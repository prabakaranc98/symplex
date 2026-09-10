"use strict";
/**
 * Bifurcation diagram: a control parameter on x, equilibria on y.
 *
 * Stable branches are solid, unstable branches dashed and lighter, and the
 * critical points where branches meet or change stability are marked with a
 * vertical rule and a label. Branch stability is taken from the caller; a
 * branch that does not declare it is drawn dotted and labelled "stability not
 * classified" rather than assumed stable.
 */

import { el, text, escapeText, figure, emptyPanel, svgFrame, embedSpec, num } from "./svg.js";
import { INK, MUTED, LINE, PAPER, ACCENT, WARN } from "./color.js";
import { linearScale, extent, combinedExtent, niceDomain, formatNumber } from "./scale.js";
import { plotArea, xAxis, yAxis, zeroLine } from "./axis.js";
import { legend } from "./legend.js";

const asArray = (value) => (Array.isArray(value) ? value : []);
const isNum = (value) => typeof value === "number" && Number.isFinite(value);

const BRANCH = {
  stable: { label: "Stable equilibrium", color: ACCENT, dash: "", width: 2 },
  unstable: { label: "Unstable equilibrium", color: WARN, dash: "6 4", width: 1.6 },
  "semi-stable": { label: "Semi-stable equilibrium", color: "#5c4fa3", dash: "10 3 2 3", width: 1.6 },
  unknown: { label: "Stability not classified", color: MUTED, dash: "2 3", width: 1.4 },
};

const CRITICAL = {
  saddle_node: "Saddle-node",
  transcritical: "Transcritical",
  pitchfork: "Pitchfork",
  hopf: "Hopf",
  fold: "Fold",
};

function branchStyle(key) {
  return BRANCH[key] || BRANCH.unknown;
}

/**
 * spec:
 *   branches   [{id, label, stability, points: [[parameter, value], ...]}]
 *   critical   [{parameter, value, kind, label}]
 *   parameterLabel / stateLabel
 */
export function bifurcationDiagram(spec = {}) {
  const branches = asArray(spec.branches).filter((branch) => asArray(branch.points).length);
  const eyebrow = spec.eyebrow || "BIFURCATION";
  if (!branches.length) {
    return emptyPanel({
      kind: "bifurcation",
      eyebrow,
      title: "No equilibrium branches.",
      message: "No continuation results were supplied, so no branch of equilibria can be drawn.",
      caption: "An empty diagram means the continuation was not run, not that the system has a single regime.",
    });
  }
  const critical = asArray(spec.critical).filter((point) => isNum(point.parameter));

  const parameters = [];
  const states = [];
  for (const branch of branches) {
    for (const point of asArray(branch.points)) {
      if (!Array.isArray(point)) continue;
      if (isNum(point[0])) parameters.push(point[0]);
      if (isNum(point[1])) states.push(point[1]);
    }
  }
  for (const point of critical) {
    parameters.push(point.parameter);
    if (isNum(point.value)) states.push(point.value);
  }
  const xSpan = extent(parameters);
  const ySpan = extent(states);
  if (!xSpan || !ySpan) {
    return emptyPanel({
      kind: "bifurcation",
      eyebrow,
      title: "No finite equilibria.",
      message: "Every supplied branch point was missing or non-finite.",
      caption: "Nothing is drawn at zero to stand in for an equilibrium that was not computed.",
    });
  }

  const width = isNum(spec.width) ? spec.width : 620;
  const height = isNum(spec.height) ? spec.height : 340;
  const area = plotArea(width, height, { top: 26, right: 20, bottom: 44, left: 58 });
  const xScale = linearScale({ domain: niceDomain(xSpan[0], xSpan[1], 5), range: [area.left, area.right] });
  const yScale = linearScale({ domain: niceDomain(ySpan[0], ySpan[1], 5), range: [area.bottom, area.top] });

  let unclassified = 0;
  const branchMarkup = branches
    .map((branch, index) => {
      const style = branchStyle(branch.stability);
      if (!BRANCH[branch.stability]) unclassified++;
      const parts = [];
      let open = false;
      for (const point of asArray(branch.points)) {
        const px = Array.isArray(point) ? xScale(point[0]) : null;
        const py = Array.isArray(point) ? yScale(point[1]) : null;
        if (px === null || py === null) {
          open = false;
          continue;
        }
        parts.push(`${open ? "L" : "M"}${num(px)} ${num(py)}`);
        open = true;
      }
      const d = parts.join("");
      if (!d) return "";
      return el(
        "g",
        {
          class: "viz-branch",
          "data-node": branch.id || `branch-${index}`,
          "data-label": branch.label || style.label,
          "data-detail": `${style.label}. ${asArray(branch.points).length} continuation points.`,
          tabindex: "0",
          role: "graphics-symbol",
          "aria-label": `${branch.label || `Branch ${index + 1}`}. ${style.label}.`,
        },
        el("path", {
          d,
          fill: "none",
          stroke: style.color,
          "stroke-width": style.width,
          "stroke-dasharray": style.dash || null,
          "stroke-linejoin": "round",
        }) + el("title", {}, escapeText(`${branch.label || "Branch"}: ${style.label}`)),
      );
    })
    .join("");

  const criticalMarkup = critical
    .map((point, index) => {
      const px = xScale(point.parameter);
      if (px === null) return "";
      const py = isNum(point.value) ? yScale(point.value) : null;
      const label = point.label || CRITICAL[point.kind] || "Critical point";
      return el(
        "g",
        {
          class: "viz-critical",
          "data-node": point.id || `critical-${index}`,
          "data-label": label,
          "data-detail": `${label} at ${spec.parameterLabel || "parameter"} = ${formatNumber(point.parameter, 4)}.${
            isNum(point.value) ? ` Equilibrium value ${formatNumber(point.value, 4)}.` : ""
          }`,
          tabindex: "0",
          role: "graphics-symbol",
          "aria-label": `${label} at parameter ${formatNumber(point.parameter, 4)}.`,
        },
        [
          el("line", {
            x1: px,
            y1: area.top,
            x2: px,
            y2: area.bottom,
            stroke: INK,
            "stroke-width": 1,
            "stroke-dasharray": "4 3",
            "stroke-opacity": 0.55,
          }),
          py === null
            ? ""
            : el("path", {
                d: `M${num(px)} ${num(py - 5.5)}L${num(px + 5.5)} ${num(py)}L${num(px)} ${num(py + 5.5)}L${num(
                  px - 5.5,
                )} ${num(py)}Z`,
                fill: PAPER,
                stroke: INK,
                "stroke-width": 1.6,
              }),
          text(px, area.top - 8, label, { "text-anchor": "middle", class: "viz-critical-label", fill: INK }),
          el("title", {}, escapeText(`${label} at ${formatNumber(point.parameter, 4)}`)),
        ],
      );
    })
    .join("");

  const usedBranches = [];
  for (const branch of branches) {
    const key = BRANCH[branch.stability] ? branch.stability : "unknown";
    if (usedBranches.includes(key)) continue;
    usedBranches.push(key);
  }

  const svg = svgFrame({
    width,
    height,
    role: "img",
    label: spec.title || "Bifurcation diagram",
    desc: `${branches.length} equilibrium branch${branches.length === 1 ? "" : "es"} against ${
      spec.parameterLabel || "the control parameter"
    }, with ${critical.length} marked critical point${critical.length === 1 ? "" : "s"}.`,
    seed: `${spec.id || "bif"}|${branches.map((branch) => branch.id).join(",")}`,
    className: "viz-bif-svg",
    children:
      xAxis({ scale: xScale, area, label: spec.parameterLabel || "", tickCount: 5 }) +
      yAxis({ scale: yScale, area, label: spec.stateLabel || "", tickCount: 5 }) +
      zeroLine({ scale: yScale, area, orientation: "y" }) +
      el("g", { class: "viz-layer viz-layer-branches" }, branchMarkup) +
      el("g", { class: "viz-layer viz-layer-critical" }, criticalMarkup),
  });

  return figure({
    kind: "bifurcation",
    id: spec.id || null,
    eyebrow,
    title: spec.title || "",
    note: spec.note || "",
    svg:
      svg +
      legend({
        title: "BRANCH STABILITY",
        items: usedBranches.map((key) => ({
          label: BRANCH[key].label,
          color: BRANCH[key].color,
          dash: BRANCH[key].dash,
          shape: "circle",
        })),
      }) +
      (critical.length
        ? legend({ title: "CRITICAL POINTS", items: [{ label: "Where a branch appears, vanishes or changes stability", color: INK, shape: "diamond", dash: "" }] })
        : "") +
      embedSpec({ chart: "bifurcation", ...JSON.parse(JSON.stringify(spec)) }),
    caption:
      (spec.caption ||
        "Branches are equilibria of the stated model traced by continuation, not observed regimes of the real system.") +
      " A critical point marks a qualitative change in the model's equilibrium structure; whether the real system crosses it depends on parameter values this diagram does not estimate." +
      (unclassified
        ? ` ${unclassified} branch${unclassified === 1 ? "" : "es"} arrived without a stability classification and ${
            unclassified === 1 ? "is" : "are"
          } drawn dotted rather than assumed stable.`
        : ""),
    scroll: false,
  });
}
