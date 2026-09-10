"use strict";
/**
 * Scales, domains, ticks and number formatting.
 *
 * Number formatting is implemented here rather than delegated to Intl: Intl
 * output depends on the host locale and ICU version, and this repository needs
 * the same input to produce the same string on every machine.
 *
 * Every scale returns null for a value it cannot place. Callers must treat null
 * as "absent" and omit the mark, never substitute zero.
 */

/** Rendered when a number is missing. Visibly not a value. */
export const NO_VALUE = "—";

const isNum = (value) => typeof value === "number" && Number.isFinite(value);

/** Min and max over the finite values only. Returns null when there are none. */
export function extent(values) {
  let min = Infinity;
  let max = -Infinity;
  for (const value of values || []) {
    if (!isNum(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return min === Infinity ? null : [min, max];
}

/** Extent across several arrays, for shared axes. */
export function combinedExtent(arrays) {
  const spans = (arrays || []).map(extent).filter(Boolean);
  if (!spans.length) return null;
  return [Math.min(...spans.map((s) => s[0])), Math.max(...spans.map((s) => s[1]))];
}

function decimalsFor(step) {
  if (!isNum(step) || step === 0) return 0;
  return Math.max(0, -Math.floor(Math.log10(Math.abs(step))) + 1);
}

/** Decimal places literally present in a number's shortest representation. */
function decimalsOf(value) {
  const rendered = String(value);
  if (rendered.includes("e")) {
    const [mantissa, exponent] = rendered.split("e");
    const mantissaDecimals = mantissa.includes(".") ? mantissa.split(".")[1].length : 0;
    return Math.max(0, Math.min(12, mantissaDecimals - Number(exponent)));
  }
  const dot = rendered.indexOf(".");
  return dot < 0 ? 0 : Math.min(12, rendered.length - dot - 1);
}

function roundStep(value, step) {
  return Number(value.toFixed(Math.min(20, decimalsFor(step))));
}

/** The 1/2/5/10 step at or just above the requested resolution. */
export function tickStep(min, max, count = 5) {
  if (!isNum(min) || !isNum(max) || max === min) return 0;
  const rough = Math.abs(max - min) / Math.max(1, count);
  const magnitude = Math.pow(10, Math.floor(Math.log10(rough)));
  const normalised = rough / magnitude;
  const factor = normalised >= 7.5 ? 10 : normalised >= 3.5 ? 5 : normalised >= 1.5 ? 2 : 1;
  return factor * magnitude;
}

/** Evenly spaced, human-readable ticks inside [min, max]. */
export function niceTicks(min, max, count = 5) {
  if (!isNum(min) || !isNum(max)) return [];
  if (min === max) return [min];
  const step = tickStep(min, max, count);
  if (!step) return [min, max];
  const start = Math.ceil(min / step) * step;
  const ticks = [];
  for (let i = 0; i < 1000; i++) {
    const value = roundStep(start + i * step, step);
    if (value > max + step * 1e-9) break;
    ticks.push(value);
  }
  return ticks.length ? ticks : [min, max];
}

/** Round a domain outward to the tick grid so the data never touches the frame. */
export function niceDomain(min, max, count = 5) {
  if (!isNum(min) || !isNum(max)) return null;
  if (min === max) {
    const pad = Math.abs(min) > 0 ? Math.abs(min) * 0.1 : 1;
    return [min - pad, max + pad];
  }
  const step = tickStep(min, max, count) || 1;
  return [roundStep(Math.floor(min / step) * step, step), roundStep(Math.ceil(max / step) * step, step)];
}

/** Decade ticks with 2× and 5× subdivisions when the span is narrow. */
export function logTicks(min, max, count = 6) {
  if (!isNum(min) || !isNum(max) || min <= 0 || max <= 0 || min === max) return [];
  const low = Math.floor(Math.log10(min));
  const high = Math.ceil(Math.log10(max));
  const decades = high - low;
  const factors = decades * 3 <= count + 2 ? [1, 2, 5] : [1];
  const ticks = [];
  for (let power = low; power <= high; power++) {
    for (const factor of factors) {
      const value = factor * Math.pow(10, power);
      if (value >= min * (1 - 1e-9) && value <= max * (1 + 1e-9)) ticks.push(value);
    }
  }
  return ticks;
}

function makeScale(kind, domain, range, project, invert, extras) {
  const scale = (value) => {
    if (!isNum(value)) return null;
    return project(value);
  };
  scale.kind = kind;
  scale.domain = domain;
  scale.range = range;
  scale.invert = invert;
  scale.ticks = extras.ticks;
  scale.defined = (value) => scale(value) !== null;
  return scale;
}

/** Linear scale. Values outside the domain map outside the range unless clamped. */
export function linearScale({ domain, range, clamp = false }) {
  const [d0, d1] = domain && isNum(domain[0]) && isNum(domain[1]) ? domain : [0, 1];
  const [r0, r1] = range && isNum(range[0]) && isNum(range[1]) ? range : [0, 1];
  const span = d1 - d0;
  const project = (value) => {
    const t = span === 0 ? 0.5 : (value - d0) / span;
    const bounded = clamp ? Math.max(0, Math.min(1, t)) : t;
    const pixel = r0 + bounded * (r1 - r0);
    return Number.isFinite(pixel) ? pixel : null;
  };
  const invert = (pixel) => {
    if (!isNum(pixel) || r1 === r0) return null;
    return d0 + ((pixel - r0) / (r1 - r0)) * span;
  };
  return makeScale("linear", [d0, d1], [r0, r1], project, invert, {
    ticks: (count = 5) => niceTicks(Math.min(d0, d1), Math.max(d0, d1), count),
  });
}

/**
 * Log scale. A non-positive value has no position on a log axis, so it maps to
 * null and the mark is dropped — it is never clamped to the axis floor, which
 * would draw a value that does not exist.
 */
export function logScale({ domain, range, base = 10 }) {
  const raw = domain || [1, 10];
  const d0 = isNum(raw[0]) && raw[0] > 0 ? raw[0] : 1;
  const d1 = isNum(raw[1]) && raw[1] > 0 ? raw[1] : Math.max(d0 * 10, 10);
  const [r0, r1] = range && isNum(range[0]) && isNum(range[1]) ? range : [0, 1];
  const logBase = Math.log(base);
  const l0 = Math.log(d0) / logBase;
  const l1 = Math.log(d1) / logBase;
  const span = l1 - l0;
  const project = (value) => {
    if (value <= 0) return null;
    const t = span === 0 ? 0.5 : (Math.log(value) / logBase - l0) / span;
    const pixel = r0 + t * (r1 - r0);
    return Number.isFinite(pixel) ? pixel : null;
  };
  const invert = (pixel) => {
    if (!isNum(pixel) || r1 === r0) return null;
    return Math.pow(base, l0 + ((pixel - r0) / (r1 - r0)) * span);
  };
  return makeScale("log", [d0, d1], [r0, r1], project, invert, {
    ticks: (count = 6) => logTicks(d0, d1, count),
  });
}

/** Discrete positions with padding, for categorical axes and matrix rows. */
export function bandScale({ domain, range, paddingInner = 0.2, paddingOuter = 0.1 }) {
  const keys = Array.isArray(domain) ? domain.map(String) : [];
  const [r0, r1] = range && isNum(range[0]) && isNum(range[1]) ? range : [0, 1];
  const total = r1 - r0;
  const n = keys.length;
  const step = n ? total / Math.max(1e-9, n - paddingInner + 2 * paddingOuter) : 0;
  const width = Math.max(0, step * (1 - paddingInner));
  const index = new Map(keys.map((key, i) => [key, i]));
  const scale = (key) => {
    const i = index.get(String(key));
    if (i === undefined) return null;
    return r0 + paddingOuter * step + i * step;
  };
  scale.kind = "band";
  scale.domain = keys;
  scale.range = [r0, r1];
  scale.bandwidth = width;
  scale.step = step;
  scale.center = (key) => {
    const start = scale(key);
    return start === null ? null : start + width / 2;
  };
  return scale;
}

function trimTrailingZeros(value) {
  if (!value.includes(".")) return value;
  return value.replace(/\.?0+$/, "");
}

/** Deterministic, locale-independent number formatting. */
export function formatNumber(value, significant = 4) {
  if (!isNum(value)) return NO_VALUE;
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 1e6 || magnitude < 1e-4) {
    const [mantissa, exponent] = value.toExponential(Math.max(0, significant - 1)).split("e");
    return `${trimTrailingZeros(mantissa)}e${exponent.replace("+", "")}`;
  }
  const fixed = value.toPrecision(Math.max(1, significant));
  return trimTrailingZeros(fixed.includes("e") ? String(Number(fixed)) : fixed);
}

/** A formatter that gives every tick on one axis the same decimal count. */
export function tickFormatter(ticks) {
  const steps = [];
  for (let i = 1; i < (ticks || []).length; i++) steps.push(Math.abs(ticks[i] - ticks[i - 1]));
  const step = steps.length ? Number(Math.min(...steps).toPrecision(6)) : 0;
  const decimals = step > 0 ? Math.min(6, decimalsOf(step)) : 0;
  return (value) => {
    if (!isNum(value)) return NO_VALUE;
    const magnitude = Math.abs(value);
    if (magnitude !== 0 && (magnitude >= 1e6 || magnitude < 1e-4)) return formatNumber(value, 3);
    const rendered = value.toFixed(Math.max(0, decimals));
    return rendered === "-0" || Number(rendered) === 0 ? "0" : rendered;
  };
}

/** Percentage with one decimal, used by sensitivity and archive charts. */
export function formatPercent(value, decimals = 1) {
  if (!isNum(value)) return NO_VALUE;
  const rendered = (value * 100).toFixed(decimals);
  return `${Number(rendered) === 0 ? "0" : trimTrailingZeros(rendered)}%`;
}

/** Signed formatting, for deltas where the sign carries the meaning. */
export function formatSigned(value, significant = 3) {
  if (!isNum(value)) return NO_VALUE;
  const rendered = formatNumber(value, significant);
  return value > 0 ? `+${rendered}` : rendered;
}
