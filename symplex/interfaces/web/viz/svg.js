"use strict";
/**
 * Shared SVG primitives for the Symplex visualisation library.
 *
 * Every function here is pure and deterministic: identical input produces a
 * byte-identical string. Nothing in this module reads the clock, the DOM, or a
 * random source. Coordinates that cannot be computed are omitted rather than
 * emitted as NaN, so a broken value is visibly absent instead of silently
 * placed at the origin.
 */

/** Text that marks a value the caller did not supply. Never a zero. */
export const ABSENT = "Not available";

/** Escape for both HTML body text and quoted XML attribute values. */
export function escapeText(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[c]);
}

/** True only for real, finite numbers. Booleans and numeric strings are not numbers. */
export function isNum(value) {
  return typeof value === "number" && Number.isFinite(value);
}

/**
 * Format a coordinate. Returns null — never "NaN" — when the value is not a
 * finite number, so callers must decide what an absent coordinate means.
 */
export function num(value, dp = 2) {
  if (!isNum(value)) return null;
  const rounded = Number(value.toFixed(dp));
  return String(rounded === 0 ? 0 : rounded);
}

/** Format a coordinate, falling back to a supplied default rather than failing. */
export function coord(value, fallback = 0, dp = 2) {
  const formatted = num(value, dp);
  return formatted === null ? num(fallback, dp) ?? "0" : formatted;
}

/** Serialise attributes; null/undefined/non-finite numeric attributes are dropped. */
export function attrs(map) {
  const parts = [];
  for (const key of Object.keys(map)) {
    const value = map[key];
    if (value === null || value === undefined || value === false) continue;
    if (typeof value === "number") {
      const formatted = num(value);
      if (formatted === null) continue;
      parts.push(`${key}="${formatted}"`);
      continue;
    }
    if (value === true) {
      parts.push(`${key}="${key}"`);
      continue;
    }
    parts.push(`${key}="${escapeText(value)}"`);
  }
  return parts.length ? " " + parts.join(" ") : "";
}

/** Build an element. `children` may be a string or an array of strings. */
export function el(name, attributes = {}, children = null) {
  const body = Array.isArray(children) ? children.filter(Boolean).join("") : children || "";
  if (!body && children !== "") return `<${name}${attrs(attributes)}/>`;
  return `<${name}${attrs(attributes)}>${body}</${name}>`;
}

/** SVG <text>, always escaped, silently dropped when it has no valid position. */
export function text(x, y, value, attributes = {}) {
  if (num(x) === null || num(y) === null) return "";
  return el("text", { x, y, ...attributes }, escapeText(value));
}

/** Deterministic 32-bit FNV-1a hash, rendered base36. Stable across processes. */
export function hashCode(value) {
  let hash = 0x811c9dc5;
  const source = String(value);
  for (let i = 0; i < source.length; i++) {
    hash ^= source.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(36);
}

/**
 * A document-unique but input-stable id. Two charts of the same data on one
 * page share defs safely; two charts of different data never collide.
 */
export function uid(prefix, seed) {
  return `${prefix}-${hashCode(seed)}`;
}

/** Deterministic seeded PRNG (mulberry32). Same seed string, same sequence. */
export function seededRandom(seed) {
  let state = 0;
  const source = String(seed);
  for (let i = 0; i < source.length; i++) state = (Math.imul(state, 31) + source.charCodeAt(i)) >>> 0;
  state = (state ^ 0x9e3779b9) >>> 0;
  return function random() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Clamp with explicit bounds; returns null for values that are not numbers. */
export function clamp(value, low, high) {
  if (!isNum(value)) return null;
  return value < low ? low : value > high ? high : value;
}

/**
 * The SVG shell every chart uses.
 *
 * `role` defaults to "img" (a static picture). Charts with focusable children
 * pass role="group". A <title> and <desc> are always emitted: the title names
 * the picture, the description says what it shows.
 */
export function svgFrame({
  width,
  height,
  label,
  desc = "",
  role = "img",
  className = "",
  seed = "",
  defs = "",
  children = "",
  preserveAspectRatio = null,
}) {
  const titleId = uid("t", `${label}|${seed}`);
  const descId = uid("d", `${label}|${desc}|${seed}`);
  const described = desc ? `${titleId} ${descId}` : titleId;
  return el(
    "svg",
    {
      viewBox: `0 0 ${coord(width, 1)} ${coord(height, 1)}`,
      role,
      class: className || null,
      "aria-labelledby": titleId,
      "aria-describedby": desc ? descId : null,
      "aria-label": label,
      preserveAspectRatio,
      focusable: "false",
      "data-described": described ? null : null,
    },
    [
      el("title", { id: titleId }, escapeText(label)),
      desc ? el("desc", { id: descId }, escapeText(desc)) : "",
      defs ? el("defs", {}, defs) : "",
      children,
    ],
  );
}

/**
 * The card wrapper shared by every chart.
 *
 * `caption` is required by house rule: it states what the picture does NOT
 * establish. A chart without that sentence is a chart that overclaims.
 */
export function figure({
  kind,
  eyebrow = "",
  title = "",
  note = "",
  controls = "",
  svg = "",
  caption = "",
  className = "",
  scroll = false,
  minWidth = 0,
  id = null,
}) {
  if (!caption) throw new Error(`viz/${kind}: every chart must state what it does not establish`);
  const canvasStyle = scroll && minWidth ? ` style="min-width:${coord(minWidth, 320)}px"` : "";
  const head =
    eyebrow || title || note
      ? `<div class="viz-head">${eyebrow ? `<span class="eyebrow">${escapeText(eyebrow)}</span>` : ""}${
          title ? `<h3>${escapeText(title)}</h3>` : ""
        }${note ? `<p class="tiny viz-note">${escapeText(note)}</p>` : ""}</div>`
      : "";
  return `<figure class="viz card viz-${escapeText(kind)}${className ? " " + escapeText(className) : ""}"${
    id ? ` id="${escapeText(id)}"` : ""
  } data-viz="${escapeText(kind)}">${head}${controls ? `<div class="viz-controls">${controls}</div>` : ""}<div class="viz-canvas${
    scroll ? " viz-scroll" : ""
  }"${scroll ? ' tabindex="0" role="region" aria-label="Scrollable diagram"' : ""}><div class="viz-canvas-inner"${canvasStyle}>${svg}</div></div><figcaption class="tiny viz-caption">${escapeText(
    caption,
  )}</figcaption></figure>`;
}

/** The honest empty state: says what is missing rather than drawing nothing. */
export function emptyPanel({ kind, eyebrow = "", title, message, caption = "" }) {
  return `<figure class="viz card viz-empty viz-${escapeText(kind)}" data-viz="${escapeText(
    kind,
  )}">${eyebrow ? `<span class="eyebrow">${escapeText(eyebrow)}</span>` : ""}<div class="viz-empty-body"><strong>${escapeText(
    title,
  )}</strong><p>${escapeText(message)}</p></div>${
    caption ? `<figcaption class="tiny viz-caption">${escapeText(caption)}</figcaption>` : ""
  }</figure>`;
}

/** A hatch fill that marks a cell or band as measured-absent rather than zero. */
export function absentPattern(id) {
  return el(
    "pattern",
    { id, width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" },
    [
      el("rect", { width: 6, height: 6, fill: "#ffffff" }),
      el("line", { x1: 0, y1: 0, x2: 0, y2: 6, stroke: "#c3cddd", "stroke-width": 1.6 }),
    ],
  );
}

/** Arrowhead marker definitions, sized for 1–2px strokes. */
export function arrowMarker(id, fill, { width = 8, refX = 7.4 } = {}) {
  return el(
    "marker",
    {
      id,
      markerWidth: width,
      markerHeight: width,
      refX,
      refY: width / 2,
      orient: "auto-start-reverse",
      markerUnits: "userSpaceOnUse",
    },
    el("path", { d: `M0 0L${num(width)} ${num(width / 2)}L0 ${num(width)}Z`, fill }),
  );
}

/** Build an SVG path from points, dropping any point that is not finite. */
export function polylinePath(points, { close = false } = {}) {
  const parts = [];
  for (const point of points || []) {
    const x = num(point?.[0]);
    const y = num(point?.[1]);
    if (x === null || y === null) continue;
    parts.push(`${parts.length ? "L" : "M"}${x} ${y}`);
  }
  if (!parts.length) return "";
  return parts.join("") + (close ? "Z" : "");
}

/** Truncate for a label without breaking the escape. */
export function ellipsis(value, max) {
  const source = String(value ?? "");
  return source.length > max ? source.slice(0, Math.max(1, max - 1)) + "…" : source;
}

/** Rough advance width for the interface font at a given size; layout only. */
export function textWidth(value, size) {
  return String(value ?? "").length * size * 0.55;
}

/**
 * Greedy word wrap for SVG labels, which have no automatic line breaking.
 * Returns at most `maxLines` lines; the last one is elided if text remains.
 */
export function wrapLabel(value, maxChars, maxLines = 2) {
  const words = String(value ?? "").split(/\s+/).filter(Boolean);
  if (!words.length) return [];
  const lines = [];
  let line = "";
  for (const word of words) {
    const candidate = line ? `${line} ${word}` : word;
    if (candidate.length <= maxChars || !line) {
      line = candidate;
      continue;
    }
    lines.push(line);
    line = word;
    if (lines.length === maxLines) break;
  }
  if (lines.length < maxLines && line) lines.push(line);
  if (lines.length === maxLines) {
    const consumed = lines.join(" ").length;
    if (consumed < String(value ?? "").trim().length) {
      const last = lines[maxLines - 1];
      lines[maxLines - 1] =
        last.length + 1 <= maxChars ? `${last}\u2026` : ellipsis(last, maxChars);
    }
  }
  return lines.map((entry) => (entry.length > maxChars ? ellipsis(entry, maxChars) : entry));
}

/**
 * Embed the props a chart was drawn from, so an interaction handler can
 * re-render it exactly (brush-to-zoom, log/linear, scenario toggles) without
 * guessing. Angle brackets and ampersands are escaped so the payload can never
 * terminate its own element.
 */
export function embedSpec(spec) {
  const payload = JSON.stringify(spec ?? null)
    .replace(/&/g, "\\u0026")
    .replace(/</g, "\\u003c")
    .replace(/>/g, "\\u003e");
  return `<script type="application/json" data-viz-spec>${payload}</script>`;
}
