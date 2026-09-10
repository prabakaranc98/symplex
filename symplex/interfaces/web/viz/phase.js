"use strict";
/**
 * Phase portraits.
 *
 * A 2D state-space plot with an optional vector-field background (quiver
 * arrows or integrated streamlines), nullclines when the caller supplies them,
 * trajectories, and fixed points marked by stability class. A 3D projection
 * option renders (x, y, z) trajectories through a fixed isometric camera.
 *
 * Stability class is shown by glyph *and* colour, so "stable" and "unstable"
 * remain distinguishable without hue. A fixed point whose stability was not
 * supplied is drawn as an open square and labelled "not classified" - it is
 * never assumed stable.
 *
 * The vector field must arrive as sampled data, not as a function: a chart that
 * evaluated a callback could not be reproduced from its own recorded props.
 */

import {
  el,
  text,
  escapeText,
  figure,
  emptyPanel,
  svgFrame,
  arrowMarker,
  embedSpec,
  uid,
  num,
} from "./svg.js";
import { INK, MUTED, LINE, PAPER, ACCENT, WARN, seriesStyle } from "./color.js";
import { linearScale, extent, combinedExtent, niceDomain, formatNumber } from "./scale.js";
import { plotArea, xAxis, yAxis, absentNote } from "./axis.js";
import { legend, markerPath } from "./legend.js";

const asArray = (value) => (Array.isArray(value) ? value : []);
const isNum = (value) => typeof value === "number" && Number.isFinite(value);

/** Stability classes, each with a distinct glyph as well as a distinct colour. */
export const STABILITY = {
  stable_node: { label: "Stable node", color: ACCENT, glyph: "filled", ring: false },
  stable_spiral: { label: "Stable spiral", color: ACCENT, glyph: "filled", ring: true },
  unstable_node: { label: "Unstable node", color: WARN, glyph: "hollow", ring: false },
  unstable_spiral: { label: "Unstable spiral", color: WARN, glyph: "hollow", ring: true },
  saddle: { label: "Saddle", color: "#5c4fa3", glyph: "saddle", ring: false },
  center: { label: "Centre", color: "#1a7f72", glyph: "half", ring: true },
  unknown: { label: "Not classified", color: MUTED, glyph: "square", ring: false },
};

function stabilityStyle(key) {
  return STABILITY[key] || STABILITY.unknown;
}

/** Fixed isometric camera for the 3D option. Deterministic by construction. */
function project3d(point, view) {
  const azimuth = isNum(view?.azimuth) ? view.azimuth : (Math.PI / 180) * 32;
  const elevation = isNum(view?.elevation) ? view.elevation : (Math.PI / 180) * 24;
  const [x, y, z] = point;
  const cosA = Math.cos(azimuth);
  const sinA = Math.sin(azimuth);
  const cosE = Math.cos(elevation);
  const sinE = Math.sin(elevation);
  return [x * cosA - y * sinA, (x * sinA + y * cosA) * sinE - z * cosE];
}

function trajectoryPoints(entry, projection, view) {
  const raw = asArray(entry.points);
  const out = [];
  for (const point of raw) {
    if (!Array.isArray(point)) continue;
    if (projection === "3d") {
      if (!isNum(point[0]) || !isNum(point[1]) || !isNum(point[2])) {
        out.push(null);
        continue;
      }
      out.push(project3d(point, view));
      continue;
    }
    if (!isNum(point[0]) || !isNum(point[1])) {
      out.push(null);
      continue;
    }
    out.push([point[0], point[1]]);
  }
  return out;
}

/** Bilinear sample of a regular vector-field grid. Returns null outside it. */
function sampleField(field, x, y) {
  const grid = field.grid || {};
  const nx = grid.nx | 0;
  const ny = grid.ny | 0;
  if (nx < 2 || ny < 2) return null;
  const tx = ((x - grid.xMin) / (grid.xMax - grid.xMin)) * (nx - 1);
  const ty = ((y - grid.yMin) / (grid.yMax - grid.yMin)) * (ny - 1);
  if (!isNum(tx) || !isNum(ty) || tx < 0 || ty < 0 || tx > nx - 1 || ty > ny - 1) return null;
  const x0 = Math.min(nx - 2, Math.floor(tx));
  const y0 = Math.min(ny - 2, Math.floor(ty));
  const fx = tx - x0;
  const fy = ty - y0;
  const at = (ix, iy) => field.vectors[iy * nx + ix];
  const v00 = at(x0, y0);
  const v10 = at(x0 + 1, y0);
  const v01 = at(x0, y0 + 1);
  const v11 = at(x0 + 1, y0 + 1);
  if (!v00 || !v10 || !v01 || !v11) return null;
  const blend = (i) =>
    v00[i] * (1 - fx) * (1 - fy) + v10[i] * fx * (1 - fy) + v01[i] * (1 - fx) * fy + v11[i] * fx * fy;
  const dx = blend(0);
  const dy = blend(1);
  return isNum(dx) && isNum(dy) ? [dx, dy] : null;
}

/** Integrate one streamline with midpoint (RK2) steps, forwards and backwards. */
function streamline(field, start, step, steps, bounds) {
  const forward = [start];
  const backward = [];
  const walk = (direction, into) => {
    let point = start.slice();
    for (let i = 0; i < steps; i++) {
      const v1 = sampleField(field, point[0], point[1]);
      if (!v1) break;
      const speed = Math.hypot(v1[0], v1[1]);
      if (!(speed > 1e-9)) break;
      const halfway = [
        point[0] + (direction * step * v1[0]) / speed / 2,
        point[1] + (direction * step * v1[1]) / speed / 2,
      ];
      const v2 = sampleField(field, halfway[0], halfway[1]) || v1;
      const speed2 = Math.hypot(v2[0], v2[1]) || speed;
      point = [
        point[0] + (direction * step * v2[0]) / speed2,
        point[1] + (direction * step * v2[1]) / speed2,
      ];
      if (
        point[0] < bounds[0] ||
        point[0] > bounds[1] ||
        point[1] < bounds[2] ||
        point[1] > bounds[3]
      )
        break;
      into.push(point.slice());
    }
  };
  walk(1, forward);
  walk(-1, backward);
  return backward.reverse().concat(forward);
}

function pathOf(points, xScale, yScale) {
  const parts = [];
  let open = false;
  for (const point of points) {
    if (!point) {
      open = false;
      continue;
    }
    const px = xScale(point[0]);
    const py = yScale(point[1]);
    if (px === null || py === null) {
      open = false;
      continue;
    }
    parts.push(`${open ? "L" : "M"}${num(px)} ${num(py)}`);
    open = true;
  }
  return parts.join("");
}

function fixedPointGlyph(style, px, py) {
  const r = 5;
  switch (style.glyph) {
    case "hollow":
      return el("circle", { cx: px, cy: py, r, fill: PAPER, stroke: style.color, "stroke-width": 2 });
    case "saddle":
      return el("path", {
        d: `M${num(px - r)} ${num(py - r)}L${num(px + r)} ${num(py + r)}M${num(px - r)} ${num(py + r)}L${num(
          px + r,
        )} ${num(py - r)}`,
        stroke: style.color,
        "stroke-width": 2.2,
        fill: "none",
      });
    case "half":
      return (
        el("circle", { cx: px, cy: py, r, fill: PAPER, stroke: style.color, "stroke-width": 1.8 }) +
        el("path", {
          d: `M${num(px)} ${num(py - r)}A${num(r)} ${num(r)} 0 0 1 ${num(px)} ${num(py + r)}Z`,
          fill: style.color,
        })
      );
    case "square":
      return el("rect", {
        x: px - r,
        y: py - r,
        width: r * 2,
        height: r * 2,
        fill: PAPER,
        stroke: style.color,
        "stroke-width": 1.8,
      });
    default:
      return el("circle", { cx: px, cy: py, r, fill: style.color, stroke: style.color, "stroke-width": 1.4 });
  }
}

/**
 * Draw a phase portrait.
 *
 * spec:
 *   trajectories [{id, label, points: [[x,y]] or [[x,y,z]], color}]
 *   field        {grid:{xMin,xMax,yMin,yMax,nx,ny}, vectors:[[dx,dy],...]}
 *   fieldStyle   "quiver" (default) or "streamlines"
 *   nullclines   [{id, label, points:[[x,y]], variable}]
 *   fixedPoints  [{x, y, stability, label, eigenvalues}]
 *   projection   "2d" (default) or "3d"; 3d needs [x,y,z] points
 *   view         {azimuth, elevation} in radians, 3d only
 */
export function phasePortrait(spec = {}) {
  const projection = spec.projection === "3d" ? "3d" : "2d";
  const view = spec.view || null;
  const trajectories = asArray(spec.trajectories).map((entry, index) => {
    const style = seriesStyle(index);
    return {
      id: String(entry.id ?? `trajectory-${index}`),
      label: String(entry.label ?? entry.id ?? `Trajectory ${index + 1}`),
      color: entry.color || style.color,
      dash: entry.dash === undefined ? style.dash : entry.dash,
      shape: entry.shape || style.shape,
      points: trajectoryPoints(entry, projection, view),
    };
  });
  const nullclines = asArray(spec.nullclines);
  const fixedPoints = asArray(spec.fixedPoints);
  const field = spec.field && Array.isArray(spec.field.vectors) ? spec.field : null;

  const hasContent =
    trajectories.some((entry) => entry.points.some(Boolean)) ||
    nullclines.length ||
    fixedPoints.length ||
    field;
  if (!hasContent) {
    return emptyPanel({
      kind: "phase",
      eyebrow: spec.eyebrow || "PHASE PORTRAIT",
      title: "No state-space data.",
      message:
        "A phase portrait needs trajectories, a vector field or fixed points. None of these were supplied.",
      caption: "An empty state space is drawn as empty, not as a stable system.",
    });
  }

  const width = isNum(spec.width) ? spec.width : 520;
  const height = isNum(spec.height) ? spec.height : 380;
  const area = plotArea(width, height, { top: 18, right: 18, bottom: 42, left: 56 });

  const xValues = [];
  const yValues = [];
  for (const entry of trajectories) {
    for (const point of entry.points) {
      if (!point) continue;
      xValues.push(point[0]);
      yValues.push(point[1]);
    }
  }
  for (const nullcline of nullclines) {
    for (const point of asArray(nullcline.points)) {
      if (!Array.isArray(point)) continue;
      const projected = projection === "3d" && point.length > 2 ? project3d(point, view) : point;
      xValues.push(projected[0]);
      yValues.push(projected[1]);
    }
  }
  for (const point of fixedPoints) {
    const projected =
      projection === "3d" && isNum(point.z) ? project3d([point.x, point.y, point.z], view) : [point.x, point.y];
    xValues.push(projected[0]);
    yValues.push(projected[1]);
  }
  if (field && field.grid) {
    xValues.push(field.grid.xMin, field.grid.xMax);
    yValues.push(field.grid.yMin, field.grid.yMax);
  }

  const xSpan = spec.xDomain && isNum(spec.xDomain[0]) ? spec.xDomain : extent(xValues);
  const ySpan = spec.yDomain && isNum(spec.yDomain[0]) ? spec.yDomain : extent(yValues);
  if (!xSpan || !ySpan) {
    return emptyPanel({
      kind: "phase",
      eyebrow: spec.eyebrow || "PHASE PORTRAIT",
      title: "No finite coordinates.",
      message: "Every supplied state-space coordinate was missing or non-finite.",
      caption: "Nothing is placed at the origin to stand in for a value that does not exist.",
    });
  }
  const xDomain = niceDomain(xSpan[0], xSpan[1], 5);
  const yDomain = niceDomain(ySpan[0], ySpan[1], 5);
  const xScale = linearScale({ domain: xDomain, range: [area.left, area.right] });
  const yScale = linearScale({ domain: yDomain, range: [area.bottom, area.top] });

  const seed = `${spec.id || "phase"}|${projection}|${trajectories.map((entry) => entry.id).join(",")}|${
    fixedPoints.length
  }`;
  const fieldArrow = uid("viz-field-arrow", MUTED);
  const flowArrow = uid("viz-flow-arrow", INK);

  let fieldMarkup = "";
  const fieldStyle = spec.fieldStyle === "streamlines" ? "streamlines" : "quiver";
  if (field && field.grid) {
    if (fieldStyle === "quiver") {
      const grid = field.grid;
      const nx = grid.nx | 0;
      const ny = grid.ny | 0;
      const magnitudes = field.vectors
        .filter((vector) => Array.isArray(vector) && isNum(vector[0]) && isNum(vector[1]))
        .map((vector) => Math.hypot(vector[0], vector[1]));
      const maxMagnitude = magnitudes.length ? Math.max(...magnitudes) : 0;
      const cellWidth = (area.right - area.left) / Math.max(1, nx - 1);
      const cellHeight = (area.bottom - area.top) / Math.max(1, ny - 1);
      const reach = Math.max(4, Math.min(cellWidth, cellHeight) * 0.42);
      const arrows = [];
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          const vector = field.vectors[iy * nx + ix];
          if (!Array.isArray(vector) || !isNum(vector[0]) || !isNum(vector[1])) continue;
          const dataX = grid.xMin + ((grid.xMax - grid.xMin) * ix) / Math.max(1, nx - 1);
          const dataY = grid.yMin + ((grid.yMax - grid.yMin) * iy) / Math.max(1, ny - 1);
          const px = xScale(dataX);
          const py = yScale(dataY);
          if (px === null || py === null) continue;
          const magnitude = Math.hypot(vector[0], vector[1]);
          if (!(magnitude > 0)) {
            arrows.push(el("circle", { cx: px, cy: py, r: 1.4, fill: MUTED, "fill-opacity": 0.5 }));
            continue;
          }
          const scaled = maxMagnitude > 0 ? (magnitude / maxMagnitude) ** 0.5 : 1;
          const length = reach * Math.max(0.25, scaled);
          const ux = (vector[0] / magnitude) * length;
          const uy = (-vector[1] / magnitude) * length;
          arrows.push(
            el("line", {
              x1: px - ux / 2,
              y1: py - uy / 2,
              x2: px + ux / 2,
              y2: py + uy / 2,
              stroke: MUTED,
              "stroke-opacity": 0.55,
              "stroke-width": 1,
              "marker-end": `url(#${fieldArrow})`,
            }),
          );
        }
      }
      fieldMarkup = arrows.join("");
    } else {
      const grid = field.grid;
      const seeds = [];
      const seedCount = isNum(spec.streamSeeds) ? Math.max(2, Math.min(14, spec.streamSeeds | 0)) : 7;
      for (let i = 0; i < seedCount; i++) {
        for (let j = 0; j < seedCount; j++) {
          seeds.push([
            grid.xMin + ((grid.xMax - grid.xMin) * (i + 0.5)) / seedCount,
            grid.yMin + ((grid.yMax - grid.yMin) * (j + 0.5)) / seedCount,
          ]);
        }
      }
      const step = Math.min(grid.xMax - grid.xMin, grid.yMax - grid.yMin) / 40;
      const bounds = [grid.xMin, grid.xMax, grid.yMin, grid.yMax];
      fieldMarkup = seeds
        .map((start) => {
          const line = streamline(field, start, step, 40, bounds);
          if (line.length < 3) return "";
          const d = pathOf(line, xScale, yScale);
          if (!d) return "";
          return el("path", {
            d,
            fill: "none",
            stroke: MUTED,
            "stroke-opacity": 0.45,
            "stroke-width": 0.9,
            "marker-end": `url(#${fieldArrow})`,
          });
        })
        .join("");
    }
  }

  const nullclineMarkup = nullclines
    .map((nullcline, index) => {
      const points = asArray(nullcline.points).map((point) =>
        projection === "3d" && Array.isArray(point) && point.length > 2 ? project3d(point, view) : point,
      );
      const d = pathOf(points, xScale, yScale);
      if (!d) return "";
      const dash = index % 2 === 0 ? "7 4" : "2 3";
      return el(
        "g",
        { class: "viz-nullcline", role: "graphics-symbol", "aria-label": `Nullcline ${nullcline.label || index + 1}` },
        el("path", { d, fill: "none", stroke: INK, "stroke-width": 1.2, "stroke-dasharray": dash }) +
          el("title", {}, escapeText(`${nullcline.label || "Nullcline"}${nullcline.variable ? ` (${nullcline.variable})` : ""}`)),
      );
    })
    .join("");

  const trajectoryMarkup = trajectories
    .map((entry) => {
      const d = pathOf(entry.points, xScale, yScale);
      if (!d) return "";
      const first = entry.points.find(Boolean);
      const last = entry.points.slice().reverse().find(Boolean);
      const startMarker =
        first && xScale(first[0]) !== null
          ? el("circle", { cx: xScale(first[0]), cy: yScale(first[1]), r: 3, fill: PAPER, stroke: entry.color, "stroke-width": 1.6 })
          : "";
      const endMarker =
        last && xScale(last[0]) !== null
          ? el("path", {
              d: markerPath(entry.shape, xScale(last[0]), yScale(last[1]), 3.6),
              fill: entry.shape === "cross" || entry.shape === "plus" ? "none" : entry.color,
              stroke: entry.color,
              "stroke-width": 1.4,
            })
          : "";
      return el(
        "g",
        { class: "viz-trajectory", "data-series": entry.id, role: "graphics-symbol", "aria-label": entry.label },
        el("path", {
          d,
          fill: "none",
          stroke: entry.color,
          "stroke-width": 1.7,
          "stroke-dasharray": entry.dash || null,
          "marker-mid": `url(#${flowArrow})`,
        }) +
          startMarker +
          endMarker +
          el("title", {}, escapeText(`${entry.label}. Start is hollow, end carries the series marker.`)),
      );
    })
    .join("");

  const fixedMarkup = fixedPoints
    .map((point, index) => {
      const projected =
        projection === "3d" && isNum(point.z) ? project3d([point.x, point.y, point.z], view) : [point.x, point.y];
      const px = xScale(projected[0]);
      const py = yScale(projected[1]);
      if (px === null || py === null) return "";
      const style = stabilityStyle(point.stability);
      const eigen = asArray(point.eigenvalues)
        .map((value) => (isNum(value) ? formatNumber(value, 3) : String(value)))
        .join(", ");
      return el(
        "g",
        {
          class: "viz-fixed-point",
          "data-node": point.id || `fixed-${index}`,
          "data-label": point.label || `Fixed point ${index + 1}`,
          "data-detail": `${style.label}.${eigen ? ` Eigenvalues: ${eigen}.` : " No eigenvalues supplied."}`,
          tabindex: "0",
          role: "graphics-symbol",
          "aria-label": `${point.label || `Fixed point ${index + 1}`}. ${style.label}.`,
        },
        [
          style.ring
            ? el("circle", { cx: px, cy: py, r: 9, fill: "none", stroke: style.color, "stroke-width": 0.9, "stroke-dasharray": "2 2" })
            : "",
          fixedPointGlyph(style, px, py),
          point.label ? text(px + 9, py - 7, String(point.label), { class: "viz-point-label", fill: INK }) : "",
          el("title", {}, escapeText(`${point.label || "Fixed point"}: ${style.label}.${eigen ? ` Eigenvalues ${eigen}.` : ""}`)),
        ],
      );
    })
    .join("");

  const usedStability = [];
  for (const point of fixedPoints) {
    const key = STABILITY[point.stability] ? point.stability : "unknown";
    if (usedStability.includes(key)) continue;
    usedStability.push(key);
  }

  const svg = svgFrame({
    width,
    height,
    role: "img",
    label: spec.title || "Phase portrait",
    desc: `${trajectories.length} trajector${trajectories.length === 1 ? "y" : "ies"}, ${
      nullclines.length
    } nullcline${nullclines.length === 1 ? "" : "s"} and ${fixedPoints.length} fixed point${
      fixedPoints.length === 1 ? "" : "s"
    } in ${projection === "3d" ? "an isometric projection of three" : "two"} state dimensions.`,
    seed,
    className: "viz-phase-svg",
    defs:
      arrowMarker(fieldArrow, MUTED, { width: 5, refX: 4.6 }) + arrowMarker(flowArrow, INK, { width: 6, refX: 5.6 }),
    children:
      xAxis({ scale: xScale, area, label: spec.xLabel || "", tickCount: 5 }) +
      yAxis({ scale: yScale, area, label: spec.yLabel || "", tickCount: 5 }) +
      el("g", { class: "viz-layer viz-layer-field" }, fieldMarkup) +
      el("g", { class: "viz-layer viz-layer-nullclines" }, nullclineMarkup) +
      el("g", { class: "viz-layer viz-layer-trajectories" }, trajectoryMarkup) +
      el("g", { class: "viz-layer viz-layer-fixed" }, fixedMarkup),
  });

  const controls =
    projection === "2d" && spec.canProject3d
      ? '<div class="viz-chart-controls" role="group" aria-label="Projection"><button type="button" class="viz-toggle" data-viz-projection="3d" aria-pressed="false">3D projection</button></div>'
      : projection === "3d"
        ? '<div class="viz-chart-controls" role="group" aria-label="Projection"><button type="button" class="viz-toggle is-on" data-viz-projection="2d" aria-pressed="true">3D projection</button></div>'
        : "";

  const legendItems = trajectories.map((entry) => ({
    label: entry.label,
    color: entry.color,
    dash: entry.dash,
    shape: entry.shape,
  }));
  const stabilityItems = usedStability.map((key) => ({
    label: STABILITY[key].label,
    color: STABILITY[key].color,
    block: true,
  }));

  return figure({
    kind: "phase",
    id: spec.id || null,
    eyebrow: spec.eyebrow || "PHASE PORTRAIT",
    title: spec.title || "",
    note: spec.note || "",
    controls,
    svg:
      svg +
      legend({ title: "TRAJECTORIES", items: legendItems }) +
      legend({ title: "FIXED POINTS", items: stabilityItems }) +
      (nullclines.length
        ? legend({
            title: "NULLCLINES",
            items: nullclines.map((nullcline, index) => ({
              label: nullcline.label || `Nullcline ${index + 1}`,
              color: INK,
              dash: index % 2 === 0 ? "7 4" : "2 3",
              shape: "circle",
            })),
          })
        : "") +
      embedSpec({ chart: "phase", ...JSON.parse(JSON.stringify(spec)) }),
    caption:
      spec.caption ||
      `Curves are integrated under the stated model, not observed paths. ${
        field ? "The background field is the supplied sampled derivative, drawn at the sample points only. " : ""
      }${
        nullclines.length ? "" : "No nullclines were supplied, so none are drawn - their absence is not a claim that none exist. "
      }${
        fixedPoints.some((point) => !STABILITY[point.stability])
          ? "Fixed points without a supplied stability class are drawn as open squares and are not assumed stable."
          : "Stability classes are as supplied by the analysis, not re-derived here."
      }`,
    scroll: false,
  });
}
