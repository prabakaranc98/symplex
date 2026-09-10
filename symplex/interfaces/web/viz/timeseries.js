"use strict";
/**
 * Trajectory plots with an uncertainty fan.
 *
 * Supports several series on shared axes, a p5/p25/p50/p75/p95 fan per series,
 * scenario overlays, a crosshair that reads every series at one x, brush to
 * zoom and a log/linear toggle.
 *
 * The rule that shapes the drawing code: a missing sample is a gap. The line
 * breaks, a small tick marks where the data stops, and the caption counts the
 * gaps. Nothing is interpolated across a hole and nothing falls back to zero.
 *
 * Interaction works by re-rendering: the props are embedded in the figure, and
 * viz/interact.js calls this function again with a changed x-domain or scale
 * type. That keeps every rendered state reproducible from data alone.
 */

import {
  el,
  text,
  escapeText,
  figure,
  emptyPanel,
  svgFrame,
  embedSpec,
  uid,
  num,
  coord,
} from "./svg.js";
import { INK, MUTED, LINE, PAPER, GRID, ACCENT, seriesStyle } from "./color.js";
import { linearScale, logScale, extent, combinedExtent, niceDomain, formatNumber, tickFormatter } from "./scale.js";
import { plotArea, xAxis, yAxis, zeroLine, absentNote } from "./axis.js";
import { legend, markerPath, missingNote } from "./legend.js";

const asArray = (value) => (Array.isArray(value) ? value : []);
const isNum = (value) => typeof value === "number" && Number.isFinite(value);

const BAND_KEYS = ["p05", "p25", "p50", "p75", "p95"];

/** Split a series into runs of consecutive present samples. */
function segments(x, values, xScale, yScale) {
  const runs = [];
  let run = [];
  const gaps = [];
  let missing = 0;
  for (let i = 0; i < x.length; i++) {
    const px = xScale(x[i]);
    const py = yScale(values[i]);
    if (px === null || py === null) {
      missing++;
      if (run.length) {
        runs.push(run);
        gaps.push(run[run.length - 1]);
        run = [];
      }
      continue;
    }
    if (!run.length && runs.length) gaps.push([px, py]);
    run.push([px, py]);
  }
  if (run.length) runs.push(run);
  return { runs, gaps, missing };
}

function linePath(points) {
  if (!points.length) return "";
  return points.map((point, i) => `${i ? "L" : "M"}${num(point[0])} ${num(point[1])}`).join("");
}

/** A band polygon between two quantile arrays, broken wherever either is absent. */
function bandPath(x, lower, upper, xScale, yScale) {
  const pieces = [];
  let top = [];
  let bottom = [];
  const flush = () => {
    if (top.length > 1) {
      pieces.push(linePath(top) + "L" + bottom.slice().reverse().map((point) => `${num(point[0])} ${num(point[1])}`).join("L") + "Z");
    }
    top = [];
    bottom = [];
  };
  for (let i = 0; i < x.length; i++) {
    const px = xScale(x[i]);
    const lo = yScale(lower[i]);
    const hi = yScale(upper[i]);
    if (px === null || lo === null || hi === null) {
      flush();
      continue;
    }
    top.push([px, hi]);
    bottom.push([px, lo]);
  }
  flush();
  return pieces.join("");
}

function resolveSeries(series, index) {
  const style = seriesStyle(isNum(series.styleIndex) ? series.styleIndex : index);
  return {
    id: String(series.id ?? `series-${index}`),
    label: String(series.label ?? series.id ?? `Series ${index + 1}`),
    values: asArray(series.values),
    band: series.band && typeof series.band === "object" ? series.band : null,
    color: series.color || style.color,
    dash: series.dash === undefined ? style.dash : series.dash,
    shape: series.shape || style.shape,
    scenario: series.scenario || "",
    unit: series.unit || "",
  };
}

/**
 * Draw a multi-series trajectory chart.
 *
 * spec:
 *   x            [number]            shared abscissa
 *   series       [{id, label, values, band:{p05,p25,p50,p75,p95}, color, dash,
 *                  shape, scenario, unit}]
 *   xLabel/yLabel, xDomain, yDomain, scaleType "linear"|"log"
 *   width, height, id, title, eyebrow, note, caption, syncGroup
 */
export function timeSeriesChart(spec = {}) {
  const x = asArray(spec.x).map((value) => (isNum(value) ? value : null));
  const series = asArray(spec.series).map(resolveSeries);
  const width = isNum(spec.width) ? spec.width : 760;
  const height = isNum(spec.height) ? spec.height : 300;
  const eyebrow = spec.eyebrow || "TRAJECTORY";
  const title = spec.title || "";

  if (!x.length || !series.length) {
    return emptyPanel({
      kind: "timeseries",
      eyebrow,
      title: "No trajectory to plot.",
      message: "No samples were supplied for this chart. An empty axis would imply a measurement that does not exist.",
      caption: "Nothing is drawn because nothing was recorded. This is not a flat or zero trajectory.",
    });
  }

  const area = plotArea(width, height, { top: 16, right: 18, bottom: 42, left: 58 });
  const xSpan = spec.xDomain && isNum(spec.xDomain[0]) && isNum(spec.xDomain[1]) ? spec.xDomain : extent(x);
  if (!xSpan) {
    return emptyPanel({
      kind: "timeseries",
      eyebrow,
      title: "The time axis has no usable values.",
      message: "Every abscissa value was missing or non-finite, so no point can be placed.",
      caption: "An axis without values is left blank rather than filled in with an index.",
    });
  }
  const yCandidates = [];
  for (const entry of series) {
    yCandidates.push(entry.values.filter(isNum));
    if (entry.band) for (const key of BAND_KEYS) yCandidates.push(asArray(entry.band[key]).filter(isNum));
  }
  const rawY = spec.yDomain && isNum(spec.yDomain[0]) && isNum(spec.yDomain[1]) ? spec.yDomain : combinedExtent(yCandidates);
  const scaleType = spec.scaleType === "log" ? "log" : "linear";
  const positiveOnly = rawY ? rawY.filter(isNum) : null;
  const logUsable = scaleType === "log" && rawY && rawY[1] > 0;
  const yDomain = rawY
    ? logUsable
      ? [Math.max(rawY[0] > 0 ? rawY[0] : rawY[1] / 1000, Number.MIN_VALUE), rawY[1]]
      : niceDomain(rawY[0], rawY[1], 5)
    : null;

  const xScale = linearScale({ domain: [xSpan[0], xSpan[1]], range: [area.left, area.right] });
  const yScale = yDomain
    ? logUsable
      ? logScale({ domain: yDomain, range: [area.bottom, area.top] })
      : linearScale({ domain: yDomain, range: [area.bottom, area.top] })
    : linearScale({ domain: [0, 1], range: [area.bottom, area.top] });

  let totalMissing = 0;
  let droppedByLog = 0;
  const seed = `${spec.id || "ts"}|${x.length}|${series.map((entry) => entry.id).join(",")}|${scaleType}|${xSpan.join(":")}`;

  const bandMarkup = series
    .map((entry) => {
      if (!entry.band) return "";
      const outer = bandPath(x, asArray(entry.band.p05), asArray(entry.band.p95), xScale, yScale);
      const inner = bandPath(x, asArray(entry.band.p25), asArray(entry.band.p75), xScale, yScale);
      if (!outer && !inner) return "";
      return el("g", { class: "viz-band", "data-series": entry.id }, [
        outer
          ? el("path", { d: outer, fill: entry.color, "fill-opacity": 0.1, stroke: "none", class: "viz-band-outer" })
          : "",
        inner
          ? el("path", { d: inner, fill: entry.color, "fill-opacity": 0.2, stroke: "none", class: "viz-band-inner" })
          : "",
      ]);
    })
    .join("");

  const lineMarkup = series
    .map((entry) => {
      const { runs, gaps, missing } = segments(x, entry.values, xScale, yScale);
      totalMissing += missing;
      if (scaleType === "log") {
        droppedByLog += entry.values.filter((value) => isNum(value) && value <= 0).length;
      }
      const paths = runs.map((run) =>
        run.length === 1
          ? el("path", {
              d: markerPath(entry.shape, run[0][0], run[0][1], 3.2),
              fill: entry.shape === "cross" || entry.shape === "plus" ? "none" : entry.color,
              stroke: entry.color,
              "stroke-width": 1.4,
            })
          : el("path", {
              d: linePath(run),
              fill: "none",
              stroke: entry.color,
              "stroke-width": 1.8,
              "stroke-dasharray": entry.dash || null,
              "stroke-linejoin": "round",
              class: "viz-line",
            }),
      );
      const gapTicks = gaps.map((point) =>
        el("line", {
          x1: point[0],
          y1: point[1] - 4,
          x2: point[0],
          y2: point[1] + 4,
          stroke: MUTED,
          "stroke-width": 1,
          class: "viz-gap",
        }),
      );
      const markers =
        x.length <= 48
          ? runs.flatMap((run) =>
              run.map((point) =>
                el("path", {
                  d: markerPath(entry.shape, point[0], point[1], 3),
                  fill: entry.shape === "cross" || entry.shape === "plus" ? "none" : entry.color,
                  stroke: entry.color,
                  "stroke-width": 1.2,
                  class: "viz-marker",
                }),
              ),
            )
          : [];
      const median = entry.band && asArray(entry.band.p50).length
        ? el("path", {
            d: linePath(segments(x, asArray(entry.band.p50), xScale, yScale).runs.flat()),
            fill: "none",
            stroke: entry.color,
            "stroke-width": 1.2,
            "stroke-dasharray": "4 2",
            class: "viz-median",
          })
        : "";
      return el(
        "g",
        {
          class: "viz-series",
          "data-series": entry.id,
          "aria-label": `${entry.label}${entry.scenario ? ` (${entry.scenario})` : ""}`,
          role: "graphics-symbol",
        },
        [median, ...paths, ...gapTicks, ...markers, el("title", {}, escapeText(entry.label))],
      );
    })
    .join("");

  const anyDrawn = series.some((entry) => entry.values.some(isNum));
  const overlay = el("rect", {
    x: area.left,
    y: area.top,
    width: Math.max(1, area.right - area.left),
    height: Math.max(1, area.bottom - area.top),
    fill: "transparent",
    class: "viz-plot-overlay",
    "data-plot-left": area.left,
    "data-plot-right": area.right,
    "data-plot-top": area.top,
    "data-plot-bottom": area.bottom,
    "data-x-min": xSpan[0],
    "data-x-max": xSpan[1],
  });

  const crosshair = el(
    "g",
    { class: "viz-crosshair", "data-viz-crosshair": "", "aria-hidden": "true", hidden: "hidden" },
    el("line", {
      x1: area.left,
      y1: area.top,
      x2: area.left,
      y2: area.bottom,
      stroke: INK,
      "stroke-width": 1,
      "stroke-dasharray": "2 2",
      class: "viz-crosshair-line",
    }) + el("g", { class: "viz-crosshair-dots" }),
  );

  const brush = el("rect", {
    class: "viz-brush",
    "data-viz-brush": "",
    x: area.left,
    y: area.top,
    width: 0,
    height: Math.max(1, area.bottom - area.top),
    fill: ACCENT,
    "fill-opacity": 0.12,
    stroke: ACCENT,
    "stroke-width": 1,
    hidden: "hidden",
  });

  const svg = svgFrame({
    width,
    height,
    role: "img",
    label: title || "Trajectory chart",
    desc: `${series.length} series over ${x.filter(isNum).length} sample points on a ${scaleType} vertical scale.${
      totalMissing ? ` ${totalMissing} samples are missing and are drawn as gaps.` : ""
    }`,
    seed,
    className: "viz-ts-svg",
    children:
      xAxis({ scale: xScale, area, label: spec.xLabel || "", tickCount: 6 }) +
      yAxis({
        scale: yScale,
        area,
        label: spec.yLabel || "",
        tickCount: 5,
        format: scaleType === "log" ? (value) => formatNumber(value, 3) : null,
      }) +
      (scaleType === "linear" ? zeroLine({ scale: yScale, area, orientation: "y" }) : "") +
      el("g", { class: "viz-layer viz-layer-bands" }, bandMarkup) +
      el("g", { class: "viz-layer viz-layer-lines" }, lineMarkup) +
      (anyDrawn ? "" : absentNote({ area, message: "No finite values in range" })) +
      brush +
      crosshair +
      overlay,
  });

  const controls = `<div class="viz-chart-controls" role="group" aria-label="Chart controls"><button type="button" class="viz-toggle${
    scaleType === "log" ? " is-on" : ""
  }" data-viz-scale="${scaleType === "log" ? "linear" : "log"}" aria-pressed="${
    scaleType === "log" ? "true" : "false"
  }">Log scale</button>${
    spec.xDomain
      ? '<button type="button" class="viz-toggle" data-viz-reset-zoom>Reset zoom</button>'
      : '<span class="tiny viz-hint">Drag across the plot to zoom</span>'
  }</div>`;

  const legendMarkup = legend({
    title: series.some((entry) => entry.scenario) ? "SERIES AND SCENARIO" : "SERIES",
    items: series.map((entry) => ({
      label: entry.label + (entry.unit ? ` (${entry.unit})` : ""),
      note: entry.scenario || (entry.band ? "with p5-p95 fan" : ""),
      color: entry.color,
      dash: entry.dash,
      shape: entry.shape,
    })),
  });

  const fanNote = series.some((entry) => entry.band)
    ? " The shaded fan shows the p5-p95 and p25-p75 intervals of the supplied ensemble; it describes spread under the stated model only, not forecast accuracy."
    : "";
  const logNote =
    scaleType === "log" && droppedByLog
      ? ` ${droppedByLog} non-positive value${droppedByLog === 1 ? " has" : "s have"} no position on a log axis and ${
          droppedByLog === 1 ? "is" : "are"
        } omitted rather than clipped.`
      : "";

  return figure({
    kind: "timeseries",
    id: spec.id || null,
    eyebrow,
    title,
    note: spec.note || "",
    controls,
    svg: svg + legendMarkup + embedSpec({ chart: "timeseries", ...serialisable(spec) }),
    caption:
      (spec.caption ||
        "Lines are computed trajectories under the stated assumptions, not observations and not calibrated forecasts.") +
      fanNote +
      logNote +
      missingNote(totalMissing, "sample"),
    className: spec.syncGroup ? `sync-${escapeText(spec.syncGroup)}` : "",
    scroll: false,
  });
}

/** Strip functions and undefined so the embedded props round-trip through JSON. */
function serialisable(spec) {
  return JSON.parse(
    JSON.stringify(spec, (key, value) => (typeof value === "function" ? undefined : value)),
  );
}

/**
 * A convenience wrapper for the common case: one quantity, several scenarios,
 * each with its own uncertainty fan.
 */
export function scenarioFan({ x, scenarios, xLabel = "", yLabel = "", ...rest }) {
  const series = asArray(scenarios).map((scenario, index) => ({
    id: scenario.id || `scenario-${index}`,
    label: scenario.label || scenario.id || `Scenario ${index + 1}`,
    values: asArray(scenario.median || scenario.values),
    band: scenario.band || null,
    scenario: scenario.role || "",
    styleIndex: index,
  }));
  return timeSeriesChart({ x, series, xLabel, yLabel, ...rest });
}
