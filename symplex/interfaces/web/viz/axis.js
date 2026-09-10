"use strict";
/**
 * Axes, gridlines and plot frames.
 *
 * Axes are rendered as plain SVG groups so a chart can compose several of them.
 * Ticks come from `scale.ticks()` unless the caller supplies its own, and all
 * ticks on one axis share a decimal count so the column reads as a column.
 */

import { el, text, num, coord } from "./svg.js";
import { GRID, LINE, MUTED, INK } from "./color.js";
import { tickFormatter, NO_VALUE } from "./scale.js";

/** Standard plot margins. Left is wide enough for a 6-character tick label. */
export const MARGIN = { top: 14, right: 16, bottom: 34, left: 48 };

/**
 * Resolve an inner drawing area. Returns the pixel bounds a chart should use,
 * never negative: a chart smaller than its own margins collapses to 1×1 rather
 * than emitting an inverted range.
 */
export function plotArea(width, height, margin = MARGIN) {
  const m = { ...MARGIN, ...(margin || {}) };
  return {
    margin: m,
    left: m.left,
    top: m.top,
    right: Math.max(m.left + 1, width - m.right),
    bottom: Math.max(m.top + 1, height - m.bottom),
    width: Math.max(1, width - m.left - m.right),
    height: Math.max(1, height - m.top - m.bottom),
  };
}

/** Horizontal axis with optional full-height gridlines. */
export function xAxis({
  scale,
  area,
  ticks = null,
  format = null,
  label = "",
  grid = true,
  tickCount = 5,
  rotate = 0,
}) {
  const values = ticks || (scale.ticks ? scale.ticks(tickCount) : []);
  const formatTick = format || tickFormatter(values);
  const parts = [
    el("line", {
      x1: area.left,
      y1: area.bottom,
      x2: area.right,
      y2: area.bottom,
      stroke: LINE,
      "stroke-width": 1,
    }),
  ];
  for (const value of values) {
    const x = scale(value);
    if (x === null) continue;
    if (grid && x > area.left + 0.5 && x < area.right - 0.5) {
      parts.push(
        el("line", { x1: x, y1: area.top, x2: x, y2: area.bottom, stroke: GRID, "stroke-width": 1 }),
      );
    }
    parts.push(el("line", { x1: x, y1: area.bottom, x2: x, y2: area.bottom + 4, stroke: LINE }));
    parts.push(
      text(x, area.bottom + 15, formatTick(value), {
        "text-anchor": rotate ? "end" : "middle",
        class: "viz-tick",
        fill: MUTED,
        transform: rotate ? `rotate(${num(rotate)} ${coord(x)} ${coord(area.bottom + 15)})` : null,
      }),
    );
  }
  if (label) {
    parts.push(
      text((area.left + area.right) / 2, area.bottom + 30, label, {
        "text-anchor": "middle",
        class: "viz-axis-label",
        fill: INK,
      }),
    );
  }
  return el("g", { class: "viz-axis viz-axis-x", "aria-hidden": "true" }, parts);
}

/** Vertical axis with optional full-width gridlines. */
export function yAxis({
  scale,
  area,
  ticks = null,
  format = null,
  label = "",
  grid = true,
  tickCount = 5,
}) {
  const values = ticks || (scale.ticks ? scale.ticks(tickCount) : []);
  const formatTick = format || tickFormatter(values);
  const parts = [
    el("line", {
      x1: area.left,
      y1: area.top,
      x2: area.left,
      y2: area.bottom,
      stroke: LINE,
      "stroke-width": 1,
    }),
  ];
  for (const value of values) {
    const y = scale(value);
    if (y === null) continue;
    if (grid && y > area.top + 0.5 && y < area.bottom - 0.5) {
      parts.push(
        el("line", { x1: area.left, y1: y, x2: area.right, y2: y, stroke: GRID, "stroke-width": 1 }),
      );
    }
    parts.push(el("line", { x1: area.left - 4, y1: y, x2: area.left, y2: y, stroke: LINE }));
    parts.push(
      text(area.left - 7, y + 3.5, formatTick(value), {
        "text-anchor": "end",
        class: "viz-tick",
        fill: MUTED,
      }),
    );
  }
  if (label) {
    const cy = (area.top + area.bottom) / 2;
    parts.push(
      text(12, cy, label, {
        "text-anchor": "middle",
        class: "viz-axis-label",
        fill: INK,
        transform: `rotate(-90 12 ${coord(cy)})`,
      }),
    );
  }
  return el("g", { class: "viz-axis viz-axis-y", "aria-hidden": "true" }, parts);
}

/** Categorical axis along the bottom, one label per band. */
export function bandAxis({ scale, area, label = "", rotate = 0, maxChars = 18, format = null }) {
  const parts = [
    el("line", {
      x1: area.left,
      y1: area.bottom,
      x2: area.right,
      y2: area.bottom,
      stroke: LINE,
    }),
  ];
  for (const key of scale.domain) {
    const x = scale.center(key);
    if (x === null) continue;
    const rendered = format ? format(key) : key;
    const shown = rendered.length > maxChars ? rendered.slice(0, maxChars - 1) + "…" : rendered;
    parts.push(
      text(x, area.bottom + 14, shown, {
        "text-anchor": rotate ? "end" : "middle",
        class: "viz-tick",
        fill: MUTED,
        transform: rotate ? `rotate(${num(rotate)} ${coord(x)} ${coord(area.bottom + 14)})` : null,
      }),
    );
  }
  if (label) {
    parts.push(
      text((area.left + area.right) / 2, area.bottom + 30, label, {
        "text-anchor": "middle",
        class: "viz-axis-label",
        fill: INK,
      }),
    );
  }
  return el("g", { class: "viz-axis viz-axis-band", "aria-hidden": "true" }, parts);
}

/** A zero reference line, drawn only when zero is actually inside the domain. */
export function zeroLine({ scale, area, orientation = "y" }) {
  const position = scale(0);
  if (position === null) return "";
  if (orientation === "y") {
    if (position < area.top || position > area.bottom) return "";
    return el("line", {
      x1: area.left,
      y1: position,
      x2: area.right,
      y2: position,
      stroke: MUTED,
      "stroke-width": 1,
      "stroke-dasharray": "3 3",
      class: "viz-zero",
    });
  }
  if (position < area.left || position > area.right) return "";
  return el("line", {
    x1: position,
    y1: area.top,
    x2: position,
    y2: area.bottom,
    stroke: MUTED,
    "stroke-width": 1,
    "stroke-dasharray": "3 3",
    class: "viz-zero",
  });
}

/** A note placed inside the plot when a series could not be drawn at all. */
export function absentNote({ area, message }) {
  return text((area.left + area.right) / 2, (area.top + area.bottom) / 2, message, {
    "text-anchor": "middle",
    class: "viz-absent",
    fill: MUTED,
  });
}

export { NO_VALUE };
