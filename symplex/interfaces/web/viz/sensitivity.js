"use strict";
/**
 * Sensitivity charts.
 *
 *   tornadoDiagram - one row per parameter, a bar from the baseline outcome to
 *     the outcome at the low and high setting of that parameter. Rows are
 *     ordered by swing, which is what makes it a tornado.
 *
 *   sobolChart - first-order against total-order indices as paired bars. The
 *     gap between them is the interaction share, and it is drawn explicitly
 *     because that gap is the whole reason to compute both.
 *
 * Both refuse to invent a baseline. A tornado without a stated baseline
 * outcome is drawn against the midpoint of the swing and says so.
 */

import { el, text, escapeText, figure, emptyPanel, svgFrame, embedSpec, num } from "./svg.js";
import { INK, MUTED, LINE, PAPER, GRID, ACCENT, WARN, divergingScale } from "./color.js";
import { linearScale, extent, combinedExtent, niceDomain, formatNumber, formatSigned, formatPercent } from "./scale.js";
import { plotArea, xAxis, zeroLine } from "./axis.js";
import { legend } from "./legend.js";

const asArray = (value) => (Array.isArray(value) ? value : []);
const isNum = (value) => typeof value === "number" && Number.isFinite(value);

/**
 * Tornado diagram.
 *
 * spec:
 *   items    [{id, label, low, high, lowSetting, highSetting, unit}]
 *   baseline outcome value at the nominal setting (optional)
 *   outcomeLabel  what the horizontal axis measures
 */
export function tornadoDiagram(spec = {}) {
  const raw = asArray(spec.items).filter((item) => item && (isNum(item.low) || isNum(item.high)));
  const eyebrow = spec.eyebrow || "SENSITIVITY";
  if (!raw.length) {
    return emptyPanel({
      kind: "sensitivity",
      eyebrow,
      title: "No sensitivity results.",
      message:
        "No parameter swings were supplied. A tornado diagram cannot be drawn from a model that has not been perturbed.",
      caption: "An empty tornado means no sensitivity analysis was run, not that the model is insensitive.",
    });
  }
  const values = raw.flatMap((item) => [item.low, item.high].filter(isNum));
  const span = extent(values);
  const baseline = isNum(spec.baseline) ? spec.baseline : (span[0] + span[1]) / 2;
  const items = raw
    .map((item, index) => ({
      id: String(item.id ?? `param-${index}`),
      label: String(item.label ?? item.id ?? `Parameter ${index + 1}`),
      low: isNum(item.low) ? item.low : null,
      high: isNum(item.high) ? item.high : null,
      lowSetting: item.lowSetting ?? "",
      highSetting: item.highSetting ?? "",
      unit: item.unit || "",
      swing:
        isNum(item.low) && isNum(item.high)
          ? Math.abs(item.high - item.low)
          : Math.abs((isNum(item.high) ? item.high : item.low) - baseline),
      index,
    }))
    .sort((a, b) => b.swing - a.swing || a.index - b.index);

  const rowHeight = 26;
  const width = isNum(spec.width) ? spec.width : 720;
  const labelWidth = Math.min(220, Math.max(110, ...items.map((item) => item.label.length * 6.2)));
  const height = items.length * rowHeight + 62;
  const area = plotArea(width, height, { top: 14, right: 22, bottom: 42, left: labelWidth + 12 });
  const domain = niceDomain(Math.min(span[0], baseline), Math.max(span[1], baseline), 5);
  const xScale = linearScale({ domain, range: [area.left, area.right] });
  const baseX = xScale(baseline);
  const colour = divergingScale({ min: -1, max: 1 });

  const rows = items
    .map((item, row) => {
      const y = area.top + row * rowHeight + rowHeight / 2;
      const parts = [];
      for (const side of ["low", "high"]) {
        const value = item[side];
        if (!isNum(value)) {
          parts.push(
            text(baseX + (side === "low" ? -8 : 8), y + 3.5, "no result", {
              "text-anchor": side === "low" ? "end" : "start",
              class: "viz-absent",
              fill: MUTED,
            }),
          );
          continue;
        }
        const px = xScale(value);
        if (px === null) continue;
        const direction = value >= baseline ? 1 : -1;
        parts.push(
          el("rect", {
            x: Math.min(baseX, px),
            y: y - 8,
            width: Math.max(1, Math.abs(px - baseX)),
            height: 16,
            fill: direction > 0 ? ACCENT : WARN,
            "fill-opacity": side === "low" ? 0.55 : 0.85,
            stroke: direction > 0 ? ACCENT : WARN,
            "stroke-width": 0.8,
            class: `viz-tornado-bar is-${side}`,
          }),
        );
      }
      const swingText =
        isNum(item.low) && isNum(item.high)
          ? `${formatNumber(item.low, 3)} to ${formatNumber(item.high, 3)}`
          : "one side only";
      return el(
        "g",
        {
          class: "viz-tornado-row",
          "data-node": item.id,
          "data-label": item.label,
          "data-detail": `Outcome ranges ${swingText}${item.unit ? ` ${item.unit}` : ""} across ${
            item.lowSetting || "the low setting"
          } and ${item.highSetting || "the high setting"}. Swing ${formatNumber(item.swing, 3)}.`,
          tabindex: "0",
          role: "graphics-symbol",
          "aria-label": `${item.label}. Outcome ranges ${swingText}.`,
        },
        [
          el("rect", { x: area.left, y: y - rowHeight / 2, width: area.right - area.left, height: rowHeight, fill: row % 2 ? GRID : PAPER, "fill-opacity": row % 2 ? 0.5 : 0 }),
          ...parts,
          text(area.left - 10, y + 3.5, item.label, { "text-anchor": "end", class: "viz-row-label", fill: INK }),
          el("title", {}, escapeText(`${item.label}: ${swingText}`)),
        ],
      );
    })
    .join("");

  const svg = svgFrame({
    width,
    height,
    role: "img",
    label: spec.title || "Tornado diagram",
    desc: `${items.length} parameters ordered by the swing they produce in ${
      spec.outcomeLabel || "the outcome"
    }, measured from a baseline of ${formatNumber(baseline, 4)}.`,
    seed: `${spec.id || "tornado"}|${items.map((item) => item.id).join(",")}`,
    className: "viz-tornado-svg",
    children:
      xAxis({ scale: xScale, area, label: spec.outcomeLabel || "", tickCount: 5 }) +
      el("g", { class: "viz-layer viz-layer-rows" }, rows) +
      (baseX === null
        ? ""
        : el("line", {
            x1: baseX,
            y1: area.top,
            x2: baseX,
            y2: area.bottom,
            stroke: INK,
            "stroke-width": 1.2,
            class: "viz-baseline",
          })) +
      (baseX === null ? "" : text(baseX, area.top - 3, "baseline", { "text-anchor": "middle", class: "viz-tick", fill: MUTED })),
  });

  return figure({
    kind: "sensitivity",
    id: spec.id || null,
    eyebrow,
    title: spec.title || "",
    note: spec.note || "",
    svg:
      svg +
      legend({
        title: "DIRECTION",
        items: [
          { label: "Outcome increases", color: ACCENT, block: true },
          { label: "Outcome decreases", color: WARN, block: true },
          { label: "Low setting (lighter) / high setting (solid)", color: MUTED, block: true },
        ],
      }) +
      embedSpec({ chart: "tornado", ...JSON.parse(JSON.stringify(spec)) }),
    caption:
      (spec.caption ||
        "Each bar is the outcome under a one-at-a-time change to a single parameter, holding the others at their nominal values.") +
      " One-at-a-time swings do not capture interactions between parameters, and the ordering is not a causal ranking." +
      (isNum(spec.baseline)
        ? ""
        : " No baseline outcome was supplied, so the midpoint of the observed swings is used as the reference and is not itself a model result."),
    scroll: false,
  });
}

/**
 * Sobol first-order against total-order indices.
 *
 * spec:
 *   items [{id, label, first, total, firstCi:[lo,hi], totalCi:[lo,hi]}]
 */
export function sobolChart(spec = {}) {
  const raw = asArray(spec.items).filter((item) => item && (isNum(item.first) || isNum(item.total)));
  const eyebrow = spec.eyebrow || "VARIANCE DECOMPOSITION";
  if (!raw.length) {
    return emptyPanel({
      kind: "sensitivity",
      eyebrow,
      title: "No variance decomposition.",
      message: "No Sobol indices were supplied for this model.",
      caption: "An absent decomposition is not a finding of zero sensitivity.",
    });
  }
  const items = raw
    .map((item, index) => ({
      id: String(item.id ?? `param-${index}`),
      label: String(item.label ?? item.id ?? `Parameter ${index + 1}`),
      first: isNum(item.first) ? item.first : null,
      total: isNum(item.total) ? item.total : null,
      firstCi: asArray(item.firstCi).filter(isNum),
      totalCi: asArray(item.totalCi).filter(isNum),
      index,
    }))
    .sort((a, b) => (b.total ?? b.first ?? 0) - (a.total ?? a.first ?? 0) || a.index - b.index);

  const rowHeight = 30;
  const width = isNum(spec.width) ? spec.width : 720;
  const labelWidth = Math.min(220, Math.max(110, ...items.map((item) => item.label.length * 6.2)));
  const height = items.length * rowHeight + 62;
  const area = plotArea(width, height, { top: 14, right: 22, bottom: 42, left: labelWidth + 12 });
  const maxValue = Math.max(
    1,
    ...items.flatMap((item) => [item.first, item.total, ...item.firstCi, ...item.totalCi].filter(isNum)),
  );
  const xScale = linearScale({ domain: [0, Math.min(1, maxValue) === maxValue ? 1 : maxValue], range: [area.left, area.right] });

  let missingPairs = 0;
  const rows = items
    .map((item, row) => {
      const y = area.top + row * rowHeight;
      const bars = [];
      const drawBar = (value, ci, offset, colour, opacity, label) => {
        if (!isNum(value)) {
          missingPairs++;
          bars.push(
            text(area.left + 6, y + offset + 9, `${label} not reported`, { class: "viz-absent", fill: MUTED }),
          );
          return;
        }
        const px = xScale(value);
        if (px === null) return;
        bars.push(
          el("rect", {
            x: area.left,
            y: y + offset,
            width: Math.max(1, px - area.left),
            height: 10,
            fill: colour,
            "fill-opacity": opacity,
            stroke: colour,
            "stroke-width": 0.7,
          }),
        );
        if (ci.length === 2) {
          const lo = xScale(ci[0]);
          const hi = xScale(ci[1]);
          if (lo !== null && hi !== null) {
            bars.push(
              el("line", { x1: lo, y1: y + offset + 5, x2: hi, y2: y + offset + 5, stroke: INK, "stroke-width": 1 }),
              el("line", { x1: lo, y1: y + offset + 1, x2: lo, y2: y + offset + 9, stroke: INK, "stroke-width": 1 }),
              el("line", { x1: hi, y1: y + offset + 1, x2: hi, y2: y + offset + 9, stroke: INK, "stroke-width": 1 }),
            );
          }
        }
      };
      drawBar(item.first, item.firstCi, 3, ACCENT, 0.85, "First order");
      drawBar(item.total, item.totalCi, 15, WARN, 0.85, "Total order");
      const interaction = isNum(item.first) && isNum(item.total) ? item.total - item.first : null;
      if (interaction !== null && interaction > 1e-6) {
        const from = xScale(item.first);
        const to = xScale(item.total);
        if (from !== null && to !== null) {
          bars.push(
            el("rect", {
              x: from,
              y: y + 15,
              width: Math.max(1, to - from),
              height: 10,
              fill: "url(#viz-interaction-hatch)",
              stroke: "none",
            }),
          );
        }
      }
      return el(
        "g",
        {
          class: "viz-sobol-row",
          "data-node": item.id,
          "data-label": item.label,
          "data-detail": `First order ${item.first === null ? "not reported" : formatPercent(item.first)}, total order ${
            item.total === null ? "not reported" : formatPercent(item.total)
          }.${interaction !== null ? ` Interaction share ${formatPercent(Math.max(0, interaction))}.` : ""}`,
          tabindex: "0",
          role: "graphics-symbol",
          "aria-label": `${item.label}. First order ${
            item.first === null ? "not reported" : formatPercent(item.first)
          }, total order ${item.total === null ? "not reported" : formatPercent(item.total)}.`,
        },
        [
          ...bars,
          text(area.left - 10, y + 17, item.label, { "text-anchor": "end", class: "viz-row-label", fill: INK }),
          el("title", {}, escapeText(item.label)),
        ],
      );
    })
    .join("");

  const svg = svgFrame({
    width,
    height,
    role: "img",
    label: spec.title || "Sobol sensitivity indices",
    desc: `${items.length} parameters with first-order and total-order variance shares. The hatched span between the two bars is the interaction share.`,
    seed: `${spec.id || "sobol"}|${items.map((item) => item.id).join(",")}`,
    className: "viz-sobol-svg",
    defs: el(
      "pattern",
      { id: "viz-interaction-hatch", width: 5, height: 5, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" },
      el("line", { x1: 0, y1: 0, x2: 0, y2: 5, stroke: INK, "stroke-width": 1.1, "stroke-opacity": 0.35 }),
    ),
    children: xAxis({ scale: xScale, area, label: spec.outcomeLabel || "Share of output variance", tickCount: 5 }) + rows,
  });

  return figure({
    kind: "sensitivity",
    id: spec.id || null,
    eyebrow,
    title: spec.title || "",
    note: spec.note || "",
    svg:
      svg +
      legend({
        title: "INDEX",
        items: [
          { label: "First order - this parameter alone", color: ACCENT, block: true },
          { label: "Total order - including its interactions", color: WARN, block: true },
          { label: "Hatched span - interaction share", color: INK, block: true },
        ],
      }) +
      embedSpec({ chart: "sobol", ...JSON.parse(JSON.stringify(spec)) }),
    caption:
      (spec.caption ||
        "Indices decompose the variance of the model output under the sampled input distribution.") +
      " They describe the model, not the world: a parameter can dominate the variance and still be irrelevant to the decision, and the indices are only as good as the assumed input ranges." +
      (missingPairs ? ` ${missingPairs} index value${missingPairs === 1 ? " was" : "s were"} not reported and ${missingPairs === 1 ? "is" : "are"} left blank rather than set to zero.` : ""),
    scroll: false,
  });
}
