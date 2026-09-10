"use strict";
/**
 * MAP-Elites archive heatmap.
 *
 * Behavioural descriptors on the axes, one cell per behaviour bin, colour by
 * fitness, and lineage arrows from a parent elite to the child that displaced
 * or extended it.
 *
 * Two honesty rules: an unoccupied cell is drawn as an empty outline, never as
 * a zero-fitness cell, because "nothing has been tried here" and "what was
 * tried scored zero" are different facts. And coverage is reported as a count
 * of occupied cells, not as a quality claim - a full archive of bad solutions
 * is still full.
 */

import {
  el,
  text,
  escapeText,
  figure,
  emptyPanel,
  svgFrame,
  embedSpec,
  arrowMarker,
  ellipsis,
  uid,
  num,
} from "./svg.js";
import { INK, MUTED, LINE, PAPER, GRID, sequential, contrastInk, ACCENT } from "./color.js";
import { formatNumber } from "./scale.js";
import { rampLegend, legend } from "./legend.js";
import { curvePath } from "./layout.js";

const asArray = (value) => (Array.isArray(value) ? value : []);
const isNum = (value) => typeof value === "number" && Number.isFinite(value);

/**
 * spec:
 *   xBins / yBins  [string]  descriptor bin labels along each axis
 *   cells          [{x, y, fitness, id, label, parent, generation}]
 *                  x and y are bin labels or indices
 *   lineage        [{from, to}]  cell ids; drawn as parent-to-child arrows
 *   xLabel / yLabel / fitnessLabel
 */
export function archiveHeatmap(spec = {}) {
  const xBins = asArray(spec.xBins).map(String);
  const yBins = asArray(spec.yBins).map(String);
  const cells = asArray(spec.cells).filter((cell) => cell && cell.x !== undefined && cell.y !== undefined);
  const eyebrow = spec.eyebrow || "ARCHIVE";
  if (!xBins.length || !yBins.length) {
    return emptyPanel({
      kind: "archive",
      eyebrow,
      title: "No behaviour space defined.",
      message: "An archive needs its descriptor bins before its coverage can be shown.",
      caption: "A behaviour space that was never defined is not an empty one.",
    });
  }

  const xIndex = new Map(xBins.map((bin, i) => [bin, i]));
  const yIndex = new Map(yBins.map((bin, i) => [bin, i]));
  const resolve = (value, index, bins) => {
    if (typeof value === "number" && Number.isInteger(value) && value >= 0 && value < bins.length) return value;
    const found = index.get(String(value));
    return found === undefined ? null : found;
  };

  const occupied = new Map();
  for (const cell of cells) {
    const cx = resolve(cell.x, xIndex, xBins);
    const cy = resolve(cell.y, yIndex, yBins);
    if (cx === null || cy === null) continue;
    occupied.set(`${cx}:${cy}`, { ...cell, cx, cy });
  }

  const fitnesses = Array.from(occupied.values())
    .map((cell) => cell.fitness)
    .filter(isNum);
  const low = fitnesses.length ? Math.min(...fitnesses) : 0;
  const high = fitnesses.length ? Math.max(...fitnesses) : 1;
  const colourFor = (value) =>
    isNum(value) ? sequential(high === low ? 0.62 : (value - low) / (high - low)) : null;

  const cellSize = Math.max(34, Math.min(78, Math.floor(520 / Math.max(xBins.length, yBins.length))));
  const labelWidth = Math.min(150, Math.max(76, ...yBins.map((bin) => Math.min(150, bin.length * 6))));
  const topLabel = 44;
  const width = labelWidth + xBins.length * cellSize + 18;
  const height = topLabel + yBins.length * cellSize + 46;
  const centreOf = (cx, cy) => [labelWidth + cx * cellSize + cellSize / 2, topLabel + cy * cellSize + cellSize / 2];

  const grid = [];
  for (let cy = 0; cy < yBins.length; cy++) {
    for (let cx = 0; cx < xBins.length; cx++) {
      const key = `${cx}:${cy}`;
      const cell = occupied.get(key);
      const x = labelWidth + cx * cellSize;
      const y = topLabel + cy * cellSize;
      const colour = cell ? colourFor(cell.fitness) : null;
      const label = `${xBins[cx]} by ${yBins[cy]}: ${
        cell
          ? isNum(cell.fitness)
            ? `${spec.fitnessLabel || "fitness"} ${formatNumber(cell.fitness, 4)}`
            : "occupied, fitness not reported"
          : "unoccupied"
      }`;
      grid.push(
        el(
          "g",
          {
            class: `viz-archive-cell${cell ? " is-occupied" : " is-empty"}`,
            "data-node": key,
            "data-label": `${xBins[cx]} by ${yBins[cy]}`,
            "data-detail": cell
              ? `${cell.label || cell.id || "Elite"}. ${
                  isNum(cell.fitness)
                    ? `${spec.fitnessLabel || "Fitness"} ${formatNumber(cell.fitness, 4)}.`
                    : "Fitness not reported."
                }${cell.parent ? ` Parent ${cell.parent}.` : ""}${
                  isNum(cell.generation) ? ` Generation ${cell.generation}.` : ""
                }`
              : "No candidate has been evaluated in this behaviour bin. The cell is empty, not zero-scoring.",
            tabindex: "0",
            role: "graphics-symbol",
            "aria-label": label,
          },
          el("rect", {
            x: x + 1,
            y: y + 1,
            width: cellSize - 2,
            height: cellSize - 2,
            rx: 3,
            fill: colour || PAPER,
            stroke: cell ? PAPER : LINE,
            "stroke-width": cell ? 1 : 1,
            "stroke-dasharray": cell ? null : "3 3",
          }) +
            (cell && isNum(cell.fitness) && cellSize >= 44
              ? text(x + cellSize / 2, y + cellSize / 2 + 4, formatNumber(cell.fitness, 3), {
                  "text-anchor": "middle",
                  class: "viz-cell-value",
                  fill: contrastInk(colour || PAPER),
                })
              : cell
                ? ""
                : text(x + cellSize / 2, y + cellSize / 2 + 4, "-", {
                    "text-anchor": "middle",
                    class: "viz-absent",
                    fill: MUTED,
                  })) +
            el("title", {}, escapeText(label)),
        ),
      );
    }
  }

  const byId = new Map(
    Array.from(occupied.values())
      .filter((cell) => cell.id !== undefined)
      .map((cell) => [String(cell.id), cell]),
  );
  const lineageEdges = asArray(spec.lineage).length
    ? asArray(spec.lineage)
    : Array.from(occupied.values())
        .filter((cell) => cell.parent)
        .map((cell) => ({ from: cell.parent, to: cell.id }));
  const lineageArrow = uid("viz-lineage-arrow", INK);
  const lineageMarkup = lineageEdges
    .map((edge) => {
      const from = byId.get(String(edge.from));
      const to = byId.get(String(edge.to));
      if (!from || !to || from === to) return "";
      const a = centreOf(from.cx, from.cy);
      const b = centreOf(to.cx, to.cy);
      const mid = [(a[0] + b[0]) / 2 + (b[1] - a[1]) * 0.16, (a[1] + b[1]) / 2 - (b[0] - a[0]) * 0.16];
      const d = curvePath([a, mid, b], { tension: 0.5 });
      if (!d) return "";
      return el(
        "g",
        { class: "viz-lineage", role: "graphics-symbol", "aria-label": `Lineage from ${edge.from} to ${edge.to}` },
        el("path", {
          d,
          fill: "none",
          stroke: INK,
          "stroke-width": 1.1,
          "stroke-opacity": 0.55,
          "marker-end": `url(#${lineageArrow})`,
        }) + el("title", {}, escapeText(`${edge.from} produced ${edge.to}`)),
      );
    })
    .join("");

  const axisLabels =
    yBins
      .map((bin, i) =>
        text(labelWidth - 8, topLabel + i * cellSize + cellSize / 2 + 3.5, ellipsis(bin, 22), {
          "text-anchor": "end",
          class: "viz-matrix-label",
          fill: INK,
        }),
      )
      .join("") +
    xBins
      .map((bin, i) =>
        text(labelWidth + i * cellSize + cellSize / 2, topLabel - 10, ellipsis(bin, 16), {
          "text-anchor": "middle",
          class: "viz-matrix-label",
          fill: INK,
        }),
      )
      .join("") +
    (spec.xLabel
      ? text(labelWidth + (xBins.length * cellSize) / 2, height - 12, String(spec.xLabel), {
          "text-anchor": "middle",
          class: "viz-axis-label",
          fill: MUTED,
        })
      : "") +
    (spec.yLabel
      ? text(12, topLabel + (yBins.length * cellSize) / 2, String(spec.yLabel), {
          "text-anchor": "middle",
          class: "viz-axis-label",
          fill: MUTED,
          transform: `rotate(-90 12 ${num(topLabel + (yBins.length * cellSize) / 2)})`,
        })
      : "");

  const total = xBins.length * yBins.length;
  const filled = occupied.size;

  const svg = svgFrame({
    width,
    height,
    role: "img",
    label: spec.title || "MAP-Elites archive",
    desc: `${filled} of ${total} behaviour bins are occupied. Colour encodes ${
      spec.fitnessLabel || "fitness"
    } from ${formatNumber(low, 3)} to ${formatNumber(high, 3)}; empty bins are outlined.`,
    seed: `${spec.id || "archive"}|${xBins.join(",")}|${yBins.join(",")}|${filled}`,
    className: "viz-archive-svg",
    defs: arrowMarker(lineageArrow, INK, { width: 6, refX: 5.6 }),
    children:
      el("g", { class: "viz-layer viz-layer-cells" }, grid.join("")) +
      el("g", { class: "viz-layer viz-layer-lineage" }, lineageMarkup) +
      el("g", { class: "viz-layer viz-layer-labels" }, axisLabels),
  });

  return figure({
    kind: "archive",
    id: spec.id || null,
    eyebrow,
    title: spec.title || "",
    note: spec.note || "",
    svg:
      svg +
      rampLegend({
        scale: (value) => colourFor(value),
        min: low,
        max: high,
        title: spec.fitnessLabel || "FITNESS",
        format: (value) => formatNumber(value, 3),
      }) +
      legend({
        items: [
          { label: "Dashed outline - bin never evaluated", color: LINE, block: true },
          ...(lineageMarkup ? [{ label: "Arrow - parent produced child", color: INK, shape: "triangle", dash: "" }] : []),
        ],
      }) +
      embedSpec({ chart: "archive", ...JSON.parse(JSON.stringify(spec)) }),
    caption:
      (spec.caption ||
        `${filled} of ${total} behaviour bins hold an evaluated candidate.`) +
      " Coverage describes how much of the declared behaviour space has been searched; it is not a measure of solution quality, and an occupied bin holds the best candidate found so far, not a verified result. Empty bins are unevaluated, not zero-scoring.",
    scroll: true,
    minWidth: Math.min(1100, width),
  });
}
