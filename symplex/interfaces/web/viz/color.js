"use strict";
/**
 * Colour for the Symplex charts.
 *
 * Two rules are enforced here rather than left to callers:
 *   1. Meaning is never carried by hue alone. `seriesStyle(i)` always returns a
 *      dash pattern and a marker shape alongside the colour, so the encoding
 *      survives colour-blindness, greyscale printing and low-quality screens.
 *   2. A diverging scale is correct at zero. The domain is made symmetric about
 *      zero before mapping, so the neutral colour lands on zero exactly and a
 *      +0.1 never looks like a −0.4.
 *
 * The palette extends the interface tokens (--green #2854c7, --orange #92610f,
 * --ink #17243a) rather than replacing them.
 */

/** Interface tokens, mirrored so SVG output does not depend on CSS variables. */
export const INK = "#17243a";
export const MUTED = "#526278";
export const LINE = "#dce3ed";
export const PAPER = "#ffffff";
export const PALE = "#edf3ff";
export const ACCENT = "#2854c7";
export const WARN = "#92610f";
export const GRID = "#e7ecf5";

/** Eight hues chosen for separation under deuteranopia and protanopia. */
export const CATEGORICAL = [
  "#2854c7",
  "#92610f",
  "#1a7f72",
  "#a1447e",
  "#45608a",
  "#8a3a1c",
  "#5c4fa3",
  "#2f6f3e",
];

/** Dash patterns paired 1:1 with CATEGORICAL. Index 0 is solid. */
export const DASHES = ["", "5 3", "1.5 3", "9 3 2 3", "3 3 1 3", "11 4", "2 2", "7 2 2 2"];

/** Marker shapes paired 1:1 with CATEGORICAL. */
export const SHAPES = [
  "circle",
  "square",
  "triangle",
  "diamond",
  "cross",
  "triangle-down",
  "hexagon",
  "plus",
];

/** The redundant encoding for series `index`. Wraps, but stays 1:1 at 8 series. */
export function seriesStyle(index) {
  const i = Number.isInteger(index) && index >= 0 ? index : 0;
  return {
    color: CATEGORICAL[i % CATEGORICAL.length],
    dash: DASHES[i % DASHES.length],
    shape: SHAPES[i % SHAPES.length],
    index: i,
  };
}

function hexToRgb(hex) {
  const value = String(hex).replace("#", "");
  const full = value.length === 3 ? value.split("").map((c) => c + c).join("") : value;
  return [
    parseInt(full.slice(0, 2), 16),
    parseInt(full.slice(2, 4), 16),
    parseInt(full.slice(4, 6), 16),
  ];
}

function toHex(channel) {
  const clamped = Math.max(0, Math.min(255, Math.round(channel)));
  return clamped.toString(16).padStart(2, "0");
}

function toLinear(channel) {
  const c = channel / 255;
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

function fromLinear(channel) {
  const c = channel <= 0.0031308 ? channel * 12.92 : 1.055 * Math.pow(channel, 1 / 2.4) - 0.055;
  return c * 255;
}

/** Mix two colours in linear light, which keeps mid-ramp steps evenly bright. */
export function mix(a, b, t) {
  const ratio = Number.isFinite(t) ? Math.max(0, Math.min(1, t)) : 0;
  const left = hexToRgb(a).map(toLinear);
  const right = hexToRgb(b).map(toLinear);
  return (
    "#" +
    left
      .map((channel, i) => toHex(fromLinear(channel + (right[i] - channel) * ratio)))
      .join("")
  );
}

function ramp(stops, t) {
  const ratio = Number.isFinite(t) ? Math.max(0, Math.min(1, t)) : 0;
  const span = stops.length - 1;
  const position = ratio * span;
  const index = Math.min(span - 1, Math.floor(position));
  return mix(stops[index], stops[index + 1], position - index);
}

const SEQUENTIAL_STOPS = ["#f2f6fd", "#c3d6f6", "#87abe6", "#4b78ce", "#2854c7", "#152f75"];
const WARM_STOPS = ["#fbf5ea", "#efdcbb", "#d8b878", "#b98d3a", "#92610f", "#5c3c06"];

/** Single-hue sequential ramp on [0, 1]. Low values stay legible on paper. */
export function sequential(t) {
  return ramp(SEQUENTIAL_STOPS, t);
}

/** Warm sequential ramp, used where blue already means something else. */
export function warmSequential(t) {
  return ramp(WARM_STOPS, t);
}

/**
 * A diverging scale whose neutral colour is exactly at zero.
 *
 * The domain is made symmetric — max(|min|, |max|) on each side — so equal
 * magnitudes of opposite sign are equally saturated. `absent` values return
 * null so the caller can draw a hatch instead of a colour.
 */
export function divergingScale({ min = -1, max = 1, negative = WARM_STOPS, positive = SEQUENTIAL_STOPS } = {}) {
  const lo = Number.isFinite(min) ? min : -1;
  const hi = Number.isFinite(max) ? max : 1;
  const bound = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
  const scale = (value) => {
    if (!Number.isFinite(value)) return null;
    const t = Math.max(-1, Math.min(1, value / bound));
    return t >= 0 ? ramp(positive, t) : ramp(negative, -t);
  };
  scale.bound = bound;
  scale.domain = [-bound, bound];
  scale.neutral = positive[0];
  return scale;
}

/** Relative luminance, for choosing readable label ink over a filled cell. */
export function luminance(hex) {
  const [r, g, b] = hexToRgb(hex).map(toLinear);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** Ink colour with acceptable contrast against `background`. */
export function contrastInk(background) {
  return luminance(background) > 0.42 ? INK : "#ffffff";
}

/** Fixed colours for signed relationships in causal diagrams. */
export const POLARITY = {
  positive: { color: ACCENT, dash: "", symbol: "+", label: "Same direction" },
  negative: { color: WARN, dash: "6 3", symbol: "−", label: "Opposite direction" },
  unknown: { color: MUTED, dash: "2 3", symbol: "?", label: "Polarity not stated" },
};

/** Fixed colours for loop character. R and B also differ by badge glyph. */
export const LOOP = {
  reinforcing: { color: ACCENT, badge: "R", label: "Reinforcing" },
  balancing: { color: WARN, badge: "B", label: "Balancing" },
  mixed: { color: "#5c4fa3", badge: "M", label: "Mixed" },
  unknown: { color: MUTED, badge: "?", label: "Polarity not stated" },
};

/** Component kinds from ComplexSystemSpec, each with a redundant glyph. */
export const COMPONENT_KIND = {
  mechanistic: { color: "#2854c7", glyph: "▣", label: "Mechanistic" },
  probabilistic: { color: "#5c4fa3", glyph: "◈", label: "Probabilistic" },
  learned: { color: "#a1447e", glyph: "◆", label: "Learned" },
  symbolic: { color: "#1a7f72", glyph: "▲", label: "Symbolic" },
  agent_based: { color: "#8a3a1c", glyph: "⬢", label: "Agent-based" },
  optimization: { color: "#92610f", glyph: "◇", label: "Optimization" },
  measurement: { color: "#45608a", glyph: "◉", label: "Measurement" },
  adapter: { color: "#2f6f3e", glyph: "▭", label: "Adapter" },
};

/** State kinds from ComplexSystemSpec. */
export const STATE_KIND = {
  observed: { color: "#2854c7", glyph: "●", label: "Observed" },
  latent: { color: "#5c4fa3", glyph: "○", label: "Latent" },
  decision: { color: "#92610f", glyph: "◆", label: "Decision" },
  exogenous: { color: "#45608a", glyph: "▲", label: "Exogenous" },
};

/** Look up a kind descriptor without inventing a colour for an unknown value. */
export function kindStyle(table, key) {
  const entry = table[key];
  if (entry) return entry;
  return { color: MUTED, glyph: "?", label: "Kind not stated" };
}
