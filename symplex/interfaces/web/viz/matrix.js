"use strict";
/**
 * Adjacency and correlation matrices as a heatmap, with seriation.
 *
 * Seriation reorders rows and columns to bring similar ones together, which is
 * what makes block structure visible; without it a matrix heatmap is mostly a
 * picture of the order the data happened to arrive in. The default method is
 * spectral: the Fiedler vector of the similarity Laplacian, found by
 * deterministic power iteration on a fixed starting vector, then sorted. There
 * is no randomness and no wall-clock dependence, so the same matrix always
 * produces the same ordering.
 *
 * The colour scale is diverging and correct at zero: the domain is made
 * symmetric so the neutral colour lands exactly on zero and equal magnitudes of
 * opposite sign are equally saturated. Cells with no value are hatched, never
 * coloured as zero.
 */

import {
  el,
  text,
  escapeText,
  figure,
  emptyPanel,
  svgFrame,
  embedSpec,
  absentPattern,
  ellipsis,
  uid,
  num,
} from "./svg.js";
import { INK, MUTED, LINE, PAPER, divergingScale, sequential, contrastInk } from "./color.js";
import { formatNumber } from "./scale.js";
import { rampLegend, legend } from "./legend.js";

const asArray = (value) => (Array.isArray(value) ? value : []);
const isNum = (value) => typeof value === "number" && Number.isFinite(value);

/**
 * Spectral seriation. Returns an index permutation.
 *
 * Similarity is |value| symmetrised; the Fiedler vector of the graph Laplacian
 * orders the nodes so that strongly connected ones end up adjacent.
 */
export function seriate(values, { iterations = 160 } = {}) {
  const n = asArray(values).length;
  if (n < 3) return Array.from({ length: n }, (_, i) => i);
  const similarity = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const a = isNum(values[i]?.[j]) ? Math.abs(values[i][j]) : 0;
      const b = isNum(values[j]?.[i]) ? Math.abs(values[j][i]) : 0;
      similarity[i][j] = i === j ? 0 : Math.max(a, b);
    }
  }
  const degree = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) sum += similarity[i][j];
    degree[i] = sum;
  }
  if (!degree.some((value) => value > 0)) return Array.from({ length: n }, (_, i) => i);

  // Deterministic start vector, orthogonal to the constant vector.
  let vector = new Float64Array(n);
  for (let i = 0; i < n; i++) vector[i] = Math.cos((i + 1) * 1.7) + ((i % 3) - 1) * 0.13;
  const centre = (v) => {
    let mean = 0;
    for (let i = 0; i < n; i++) mean += v[i];
    mean /= n;
    for (let i = 0; i < n; i++) v[i] -= mean;
    let norm = 0;
    for (let i = 0; i < n; i++) norm += v[i] * v[i];
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < n; i++) v[i] /= norm;
  };
  centre(vector);
  const maxDegree = Math.max(...degree) || 1;
  const shift = 2 * maxDegree;
  for (let step = 0; step < iterations; step++) {
    const next = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let laplacian = degree[i] * vector[i];
      for (let j = 0; j < n; j++) laplacian -= similarity[i][j] * vector[j];
      // Shifted inverse-free power iteration finds the smallest non-trivial mode.
      next[i] = shift * vector[i] - laplacian;
    }
    vector = next;
    centre(vector);
  }
  return Array.from({ length: n }, (_, i) => i).sort((a, b) => vector[a] - vector[b] || a - b);
}

/**
 * spec:
 *   labels        [string]                 row and column labels (square case)
 *   rowLabels / columnLabels                for a rectangular matrix
 *   values        [[number|null, ...], ...]
 *   diverging     true (default) for signed data; false for magnitudes
 *   seriate       true (default) to reorder rows and columns
 *   valueLabel    what a cell means
 */
export function matrixHeatmap(spec = {}) {
  const values = asArray(spec.values).map((row) => asArray(row));
  const eyebrow = spec.eyebrow || "MATRIX";
  if (!values.length || !values[0].length) {
    return emptyPanel({
      kind: "matrix",
      eyebrow,
      title: "No matrix to draw.",
      message: "No cell values were supplied.",
      caption: "An empty matrix means no relationships were computed, not that they are all zero.",
    });
  }
  const rowCount = values.length;
  const columnCount = Math.max(...values.map((row) => row.length));
  const square = rowCount === columnCount;
  const rowLabels = asArray(spec.rowLabels).length
    ? asArray(spec.rowLabels).map(String)
    : asArray(spec.labels).length
      ? asArray(spec.labels).map(String)
      : Array.from({ length: rowCount }, (_, i) => `r${i + 1}`);
  const columnLabels = asArray(spec.columnLabels).length
    ? asArray(spec.columnLabels).map(String)
    : asArray(spec.labels).length
      ? asArray(spec.labels).map(String)
      : Array.from({ length: columnCount }, (_, i) => `c${i + 1}`);

  const shouldSeriate = spec.seriate !== false && square && rowCount >= 3;
  const order = shouldSeriate ? seriate(values) : Array.from({ length: rowCount }, (_, i) => i);
  const columnOrder = shouldSeriate ? order : Array.from({ length: columnCount }, (_, i) => i);

  const finite = values.flat().filter(isNum);
  const missing = rowCount * columnCount - finite.length;
  const diverging = spec.diverging !== false;
  const low = finite.length ? Math.min(...finite) : -1;
  const high = finite.length ? Math.max(...finite) : 1;
  const colourScale = diverging
    ? divergingScale({ min: low, max: high })
    : (value) => (isNum(value) ? sequential(high === low ? 0.5 : (value - low) / (high - low)) : null);

  const cell = Math.max(
    12,
    Math.min(34, Math.floor(560 / Math.max(rowCount, columnCount))),
  );
  const labelWidth = Math.min(150, Math.max(60, ...rowLabels.map((label) => Math.min(150, label.length * 6))));
  const topLabel = Math.min(120, Math.max(46, ...columnLabels.map((label) => Math.min(120, label.length * 5.4))));
  const width = labelWidth + columnCount * cell + 16;
  const height = topLabel + rowCount * cell + 16;
  const hatchId = uid("viz-absent", `${spec.id || "matrix"}`);

  const cells = [];
  order.forEach((rowIndex, r) => {
    columnOrder.forEach((columnIndex, c) => {
      const value = values[rowIndex]?.[columnIndex];
      const x = labelWidth + c * cell;
      const y = topLabel + r * cell;
      const colour = isNum(value) ? colourScale(value) : null;
      const label = `${rowLabels[rowIndex]} to ${columnLabels[columnIndex]}: ${
        isNum(value) ? formatNumber(value, 3) : "no value recorded"
      }`;
      cells.push(
        el(
          "g",
          {
            class: "viz-cell",
            "data-node": `${rowIndex}:${columnIndex}`,
            "data-label": `${rowLabels[rowIndex]} to ${columnLabels[columnIndex]}`,
            "data-detail": isNum(value)
              ? `${spec.valueLabel || "Value"} ${formatNumber(value, 4)}.`
              : "No value was recorded for this pair. The cell is hatched, not zero.",
            tabindex: "0",
            role: "graphics-symbol",
            "aria-label": label,
          },
          el("rect", {
            x,
            y,
            width: cell,
            height: cell,
            fill: colour || `url(#${hatchId})`,
            stroke: PAPER,
            "stroke-width": 0.6,
          }) +
            (cell >= 26 && isNum(value)
              ? text(x + cell / 2, y + cell / 2 + 3.5, formatNumber(value, 2), {
                  "text-anchor": "middle",
                  class: "viz-cell-value",
                  fill: contrastInk(colour || PAPER),
                })
              : "") +
            el("title", {}, escapeText(label)),
        ),
      );
    });
  });

  const rowLabelMarkup = order
    .map((rowIndex, r) =>
      text(labelWidth - 6, topLabel + r * cell + cell / 2 + 3.5, ellipsis(rowLabels[rowIndex], 24), {
        "text-anchor": "end",
        class: "viz-matrix-label",
        fill: INK,
      }),
    )
    .join("");
  const columnLabelMarkup = columnOrder
    .map((columnIndex, c) => {
      const x = labelWidth + c * cell + cell / 2;
      return text(x, topLabel - 6, ellipsis(columnLabels[columnIndex], 20), {
        "text-anchor": "start",
        class: "viz-matrix-label",
        fill: INK,
        transform: `rotate(-60 ${num(x)} ${num(topLabel - 6)})`,
      });
    })
    .join("");

  const svg = svgFrame({
    width,
    height,
    role: "img",
    label: spec.title || "Matrix heatmap",
    desc: `${rowCount} by ${columnCount} matrix${
      shouldSeriate ? ", rows and columns reordered by spectral seriation to expose block structure" : ""
    }. Values run from ${formatNumber(low, 3)} to ${formatNumber(high, 3)}.${
      missing ? ` ${missing} cells have no value and are hatched.` : ""
    }`,
    seed: `${spec.id || "matrix"}|${rowCount}x${columnCount}|${shouldSeriate}`,
    className: "viz-matrix-svg",
    defs: absentPattern(hatchId),
    children:
      el("g", { class: "viz-layer viz-layer-cells" }, cells.join("")) +
      el("g", { class: "viz-layer viz-layer-labels" }, rowLabelMarkup + columnLabelMarkup),
  });

  const controls = square
    ? `<div class="viz-chart-controls" role="group" aria-label="Ordering"><button type="button" class="viz-toggle${
        shouldSeriate ? " is-on" : ""
      }" data-viz-seriate="${shouldSeriate ? "false" : "true"}" aria-pressed="${shouldSeriate}">Seriate</button></div>`
    : "";

  return figure({
    kind: "matrix",
    id: spec.id || null,
    eyebrow,
    title: spec.title || "",
    note: spec.note || "",
    controls,
    svg:
      svg +
      rampLegend({
        scale: colourScale,
        min: diverging ? -Math.max(Math.abs(low), Math.abs(high)) : low,
        max: diverging ? Math.max(Math.abs(low), Math.abs(high)) : high,
        mid: diverging ? 0 : null,
        title: spec.valueLabel || "VALUE",
        format: (value) => formatNumber(value, 2),
      }) +
      (missing
        ? legend({ items: [{ label: "Hatched - no value recorded", color: LINE, block: true, pattern: hatchId }] })
        : "") +
      embedSpec({ chart: "matrix", ...JSON.parse(JSON.stringify(spec)) }),
    caption:
      (spec.caption || "Cells are the supplied pairwise values.") +
      (shouldSeriate
        ? " Rows and columns are reordered by spectral seriation so that related entries sit together; the ordering is a display choice and carries no meaning of its own."
        : " Rows and columns are in the order supplied, so any apparent block structure may be an artefact of that order.") +
      (diverging ? " The colour scale is symmetric about zero, so equal magnitudes of opposite sign are equally saturated." : "") +
      (missing
        ? ` ${missing} cell${missing === 1 ? " has" : "s have"} no recorded value and ${
            missing === 1 ? "is" : "are"
          } hatched rather than drawn as zero.`
        : ""),
    scroll: true,
    minWidth: Math.min(1200, width),
  });
}
