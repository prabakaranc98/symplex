"use strict";
/**
 * Legends and marker glyphs.
 *
 * Every legend entry shows colour, dash pattern and marker shape together. A
 * reader who cannot separate two hues can still separate a dashed square from
 * a solid circle, which is the point.
 */

import { el, escapeText, num, coord, text } from "./svg.js";
import { MUTED, INK, LINE } from "./color.js";

/** Marker geometry for the shapes in color.js SHAPES, as an SVG path. */
export function markerPath(shape, cx, cy, size = 4) {
  const x = Number(cx);
  const y = Number(cy);
  const r = Number(size);
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(r)) return "";
  const p = (dx, dy) => `${num(x + dx)} ${num(y + dy)}`;
  switch (shape) {
    case "square":
      return `M${p(-r, -r)}L${p(r, -r)}L${p(r, r)}L${p(-r, r)}Z`;
    case "triangle":
      return `M${p(0, -r * 1.15)}L${p(r, r * 0.75)}L${p(-r, r * 0.75)}Z`;
    case "triangle-down":
      return `M${p(0, r * 1.15)}L${p(r, -r * 0.75)}L${p(-r, -r * 0.75)}Z`;
    case "diamond":
      return `M${p(0, -r * 1.3)}L${p(r * 1.1, 0)}L${p(0, r * 1.3)}L${p(-r * 1.1, 0)}Z`;
    case "cross":
      return `M${p(-r, -r)}L${p(r, r)}M${p(-r, r)}L${p(r, -r)}`;
    case "plus":
      return `M${p(0, -r * 1.2)}L${p(0, r * 1.2)}M${p(-r * 1.2, 0)}L${p(r * 1.2, 0)}`;
    case "hexagon": {
      const points = [];
      for (let i = 0; i < 6; i++) {
        const angle = (Math.PI / 3) * i - Math.PI / 2;
        points.push(p(r * Math.cos(angle), r * Math.sin(angle)));
      }
      return `M${points.join("L")}Z`;
    }
    default: {
      const rr = num(r);
      return `M${num(x - r)} ${num(y)}a${rr} ${rr} 0 1 0 ${num(r * 2)} 0a${rr} ${rr} 0 1 0 ${num(-r * 2)} 0Z`;
    }
  }
}

/** Is this shape drawn with a stroke rather than a fill? */
function strokeOnly(shape) {
  return shape === "cross" || shape === "plus";
}

/** One inline swatch: a short line in the series dash, with its marker on top. */
export function swatch({ color = MUTED, dash = "", shape = "circle", width = 22, height = 12 }) {
  const cy = height / 2;
  return el(
    "svg",
    { width, height, viewBox: `0 0 ${num(width)} ${num(height)}`, class: "viz-swatch", "aria-hidden": "true", focusable: "false" },
    [
      el("line", {
        x1: 1,
        y1: cy,
        x2: width - 1,
        y2: cy,
        stroke: color,
        "stroke-width": 2,
        "stroke-dasharray": dash || null,
      }),
      el("path", {
        d: markerPath(shape, width / 2, cy, 3.2),
        fill: strokeOnly(shape) ? "none" : color,
        stroke: color,
        "stroke-width": strokeOnly(shape) ? 1.6 : 1,
      }),
    ],
  );
}

/** A block swatch for categorical fills (matrix, archive, sankey). */
export function blockSwatch({ color = MUTED, pattern = null, width = 18, height = 12 }) {
  return el(
    "svg",
    { width, height, viewBox: `0 0 ${num(width)} ${num(height)}`, class: "viz-swatch", "aria-hidden": "true", focusable: "false" },
    el("rect", {
      x: 0.5,
      y: 0.5,
      width: width - 1,
      height: height - 1,
      fill: pattern ? `url(#${pattern})` : color,
      stroke: LINE,
      rx: 2,
    }),
  );
}

/**
 * The HTML legend that sits beneath a chart.
 *
 * `items` entries: {label, color, dash, shape, block, pattern, note}.
 * Rendered as a definition-free list so screen readers announce it as content
 * rather than as chart furniture.
 */
export function legend({ items, title = "", className = "", columns = 0 }) {
  const entries = (items || []).filter(Boolean);
  if (!entries.length) return "";
  const body = entries
    .map((item) => {
      const glyph = item.block
        ? blockSwatch({ color: item.color, pattern: item.pattern })
        : swatch({ color: item.color, dash: item.dash, shape: item.shape });
      return `<li class="viz-legend-item${item.muted ? " is-muted" : ""}">${glyph}<span>${escapeText(
        item.label,
      )}</span>${item.note ? `<small>${escapeText(item.note)}</small>` : ""}</li>`;
    })
    .join("");
  return `<div class="viz-legend${className ? " " + escapeText(className) : ""}"${
    columns ? ` style="--viz-legend-columns:${num(columns)}"` : ""
  }>${title ? `<span class="eyebrow">${escapeText(title)}</span>` : ""}<ul>${body}</ul></div>`;
}

/** A continuous colour ramp legend, labelled at both ends and at the midpoint. */
export function rampLegend({ scale, min, max, mid = null, title = "", steps = 24, format = String }) {
  const cells = [];
  for (let i = 0; i < steps; i++) {
    const t = steps === 1 ? 0.5 : i / (steps - 1);
    const value = min + t * (max - min);
    const color = scale(value);
    if (!color) continue;
    cells.push(
      `<i style="background:${escapeText(color)}" aria-hidden="true"></i>`,
    );
  }
  const midLabel = mid === null ? "" : `<span>${escapeText(format(mid))}</span>`;
  return `<div class="viz-ramp">${title ? `<span class="eyebrow">${escapeText(title)}</span>` : ""}<div class="viz-ramp-bar">${cells.join(
    "",
  )}</div><div class="viz-ramp-axis"><span>${escapeText(format(min))}</span>${midLabel}<span>${escapeText(
    format(max),
  )}</span></div></div>`;
}

/** An in-SVG legend, for diagrams that must stay legible when exported alone. */
export function svgLegend({ items, x, y, lineHeight = 15, fontSize = 10 }) {
  const entries = (items || []).filter(Boolean);
  if (!entries.length) return "";
  const rows = entries.map((item, i) => {
    const cy = y + i * lineHeight;
    const glyph = item.block
      ? el("rect", { x, y: cy - 5, width: 11, height: 10, fill: item.color, rx: 2 })
      : [
          el("line", {
            x1: x,
            y1: cy,
            x2: x + 16,
            y2: cy,
            stroke: item.color,
            "stroke-width": 2,
            "stroke-dasharray": item.dash || null,
          }),
          el("path", {
            d: markerPath(item.shape || "circle", x + 8, cy, 3),
            fill: strokeOnly(item.shape) ? "none" : item.color,
            stroke: item.color,
          }),
        ].join("");
    return glyph + text(x + 21, cy + 3.5, item.label, { fill: INK, "font-size": fontSize });
  });
  return el("g", { class: "viz-svg-legend", "aria-hidden": "true" }, rows);
}

/** A caption fragment naming how many values were missing and therefore not drawn. */
export function missingNote(count, noun = "value") {
  if (!count) return "";
  return ` ${count} ${noun}${count === 1 ? "" : "s"} had no value and ${
    count === 1 ? "is" : "are"
  } shown as a gap, not as zero.`;
}
