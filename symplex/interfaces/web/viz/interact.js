"use strict";
/**
 * Interaction for the viz library.
 *
 * The workspace re-renders whole views by assigning innerHTML, so per-element
 * listeners would not survive. Everything here is delegated from the document
 * and keyed off data attributes, which means a chart keeps working after any
 * number of re-renders and a chart rendered in a test has no listeners at all.
 *
 * State changes that alter what is drawn - log/linear, brush zoom, shared y,
 * seriation, 3D projection - are handled by re-rendering the chart from the
 * props embedded next to it. Every visible state is therefore reproducible
 * from data, which is the same guarantee the pure render functions give.
 */

import { timeSeriesChart } from "./timeseries.js";
import { smallMultiples } from "./smallmultiples.js";
import { matrixHeatmap } from "./matrix.js";
import { phasePortrait } from "./phase.js";
import { tornadoDiagram, sobolChart } from "./sensitivity.js";
import { bifurcationDiagram } from "./bifurcation.js";
import { sankeyDiagram } from "./sankey.js";
import { archiveHeatmap } from "./archive.js";
import { formatNumber } from "./scale.js";

const RENDERERS = {
  timeseries: timeSeriesChart,
  smallmultiples: smallMultiples,
  matrix: matrixHeatmap,
  phase: phasePortrait,
  tornado: tornadoDiagram,
  sobol: sobolChart,
  bifurcation: bifurcationDiagram,
  sankey: sankeyDiagram,
  archive: archiveHeatmap,
};

let installed = false;
const DRAG = { active: false, figure: null, startView: 0, overlay: null };

function figureOf(node) {
  return node && node.closest ? node.closest("figure.viz") : null;
}

function specOf(figure) {
  const holder = figure && figure.querySelector("script[data-viz-spec]");
  if (!holder) return null;
  try {
    return JSON.parse(holder.textContent || "null");
  } catch {
    return null;
  }
}

function rerender(figure, changes) {
  const spec = specOf(figure);
  if (!spec) return false;
  const renderer = RENDERERS[spec.chart];
  if (!renderer) return false;
  const next = { ...spec, ...changes };
  delete next.chart;
  let html;
  try {
    html = renderer(next);
  } catch {
    return false;
  }
  figure.outerHTML = html;
  return true;
}

function setReadout(figure, message) {
  const readout = figure && figure.querySelector("[data-viz-readout]");
  if (!readout) return;
  readout.textContent = message || readout.dataset.default || "";
}

function describe(node) {
  const label = node.getAttribute("data-label") || "";
  const detail = node.getAttribute("data-detail") || "";
  if (!label && !detail) return "";
  return detail ? `${label}${label ? " - " : ""}${detail}` : label;
}

/** Dim everything not adjacent to the focused node. */
function applyFocus(figure, nodeId) {
  const root = figure.querySelector("svg");
  if (!root) return;
  const adjacency = new Set([nodeId]);
  root.querySelectorAll("[data-source][data-target]").forEach((edge) => {
    const source = edge.getAttribute("data-source");
    const target = edge.getAttribute("data-target");
    if (source === nodeId) adjacency.add(target);
    if (target === nodeId) adjacency.add(source);
  });
  root.querySelectorAll(".viz-node, .viz-cld-node, .viz-stock, .viz-aux").forEach((node) => {
    const id = node.getAttribute("data-node");
    node.classList.toggle("is-focused", id === nodeId);
    node.classList.toggle("is-dimmed", nodeId !== null && !adjacency.has(id));
  });
  root.querySelectorAll(".viz-edge, .viz-link, .viz-flow, .viz-ribbon").forEach((edge) => {
    const source = edge.getAttribute("data-source");
    const target = edge.getAttribute("data-target");
    const connected = nodeId === null || source === nodeId || target === nodeId;
    edge.classList.toggle("is-dimmed", !connected);
    edge.classList.toggle("is-focused", nodeId !== null && connected);
  });
  if (nodeId === null) delete figure.dataset.focus;
  else figure.dataset.focus = nodeId;
}

/** Convert a client x coordinate into the SVG's own user units. */
function toUserX(svg, clientX) {
  const box = svg.getBoundingClientRect();
  if (!box.width) return null;
  const viewBox = (svg.getAttribute("viewBox") || "").split(/\s+/).map(Number);
  const viewWidth = viewBox.length === 4 && Number.isFinite(viewBox[2]) ? viewBox[2] : box.width;
  return ((clientX - box.left) / box.width) * viewWidth;
}

function nearestIndex(values, target) {
  let best = -1;
  let bestGap = Infinity;
  for (let i = 0; i < values.length; i++) {
    const value = values[i];
    if (typeof value !== "number" || !Number.isFinite(value)) continue;
    const gap = Math.abs(value - target);
    if (gap < bestGap) {
      bestGap = gap;
      best = i;
    }
  }
  return best;
}

function crosshairFor(figure, dataX) {
  const svg = figure.querySelector("svg");
  const overlay = figure.querySelector(".viz-plot-overlay");
  const crosshair = figure.querySelector("[data-viz-crosshair]");
  const spec = specOf(figure);
  if (!svg || !overlay || !crosshair || !spec || spec.chart !== "timeseries") return;
  const left = Number(overlay.getAttribute("data-plot-left"));
  const right = Number(overlay.getAttribute("data-plot-right"));
  const top = Number(overlay.getAttribute("data-plot-top"));
  const bottom = Number(overlay.getAttribute("data-plot-bottom"));
  const xMin = Number(overlay.getAttribute("data-x-min"));
  const xMax = Number(overlay.getAttribute("data-x-max"));
  const x = Array.isArray(spec.x) ? spec.x : [];
  const index = nearestIndex(x, dataX);
  if (index < 0 || !Number.isFinite(xMin) || xMax === xMin) {
    crosshair.setAttribute("hidden", "hidden");
    setReadout(figure, "");
    return;
  }
  const px = left + ((x[index] - xMin) / (xMax - xMin)) * (right - left);
  if (!Number.isFinite(px)) return;
  const line = crosshair.querySelector(".viz-crosshair-line");
  if (line) {
    line.setAttribute("x1", String(px));
    line.setAttribute("x2", String(px));
  }
  const dots = crosshair.querySelector(".viz-crosshair-dots");
  const readings = [];
  if (dots) {
    while (dots.firstChild) dots.removeChild(dots.firstChild);
    for (const series of spec.series || []) {
      const value = Array.isArray(series.values) ? series.values[index] : null;
      const label = series.label || series.id || "series";
      if (typeof value !== "number" || !Number.isFinite(value)) {
        readings.push(`${label}: no sample`);
        continue;
      }
      readings.push(`${label}: ${formatNumber(value, 4)}`);
    }
  }
  crosshair.removeAttribute("hidden");
  setReadout(
    figure,
    `${spec.xLabel || "x"} = ${formatNumber(x[index], 4)}  -  ${readings.join("   ")}`,
  );
  syncGroup(figure, dataX);
}

function syncGroup(origin, dataX) {
  const group = Array.from(origin.classList).find((name) => name.startsWith("sync-"));
  if (!group) return;
  document.querySelectorAll(`figure.viz.${group}`).forEach((other) => {
    if (other === origin) return;
    const overlay = other.querySelector(".viz-plot-overlay");
    const crosshair = other.querySelector("[data-viz-crosshair]");
    const spec = specOf(other);
    if (!overlay || !crosshair || !spec || spec.chart !== "timeseries") return;
    const left = Number(overlay.getAttribute("data-plot-left"));
    const right = Number(overlay.getAttribute("data-plot-right"));
    const xMin = Number(overlay.getAttribute("data-x-min"));
    const xMax = Number(overlay.getAttribute("data-x-max"));
    const px = left + ((dataX - xMin) / (xMax - xMin)) * (right - left);
    if (!Number.isFinite(px) || px < left || px > right) {
      crosshair.setAttribute("hidden", "hidden");
      return;
    }
    const line = crosshair.querySelector(".viz-crosshair-line");
    if (line) {
      line.setAttribute("x1", String(px));
      line.setAttribute("x2", String(px));
    }
    crosshair.removeAttribute("hidden");
  });
}

function clearCrosshairs() {
  document.querySelectorAll("figure.viz [data-viz-crosshair]").forEach((crosshair) => {
    crosshair.setAttribute("hidden", "hidden");
  });
  document.querySelectorAll("figure.viz [data-viz-readout]").forEach((readout) => {
    readout.textContent = readout.dataset.default || "";
  });
}

function handleToggleButton(button) {
  const figure = figureOf(button);
  if (!figure) return false;
  if (button.hasAttribute("data-viz-layer")) {
    const layer = button.getAttribute("data-viz-layer");
    const pressed = button.getAttribute("aria-pressed") !== "false";
    button.setAttribute("aria-pressed", pressed ? "false" : "true");
    button.classList.toggle("is-off", pressed);
    figure.classList.toggle(`hide-layer-${layer}`, pressed);
    return true;
  }
  if (button.hasAttribute("data-viz-scale")) {
    return rerender(figure, { scaleType: button.getAttribute("data-viz-scale") });
  }
  if (button.hasAttribute("data-viz-reset-zoom")) {
    return rerender(figure, { xDomain: null });
  }
  if (button.hasAttribute("data-viz-shared-y")) {
    return rerender(figure, { sharedY: button.getAttribute("data-viz-shared-y") === "true" });
  }
  if (button.hasAttribute("data-viz-seriate")) {
    return rerender(figure, { seriate: button.getAttribute("data-viz-seriate") === "true" });
  }
  if (button.hasAttribute("data-viz-projection")) {
    return rerender(figure, { projection: button.getAttribute("data-viz-projection"), canProject3d: true });
  }
  return false;
}

/** Install the delegated handlers. Safe to call more than once. */
export function install(root) {
  const target = root || (typeof document !== "undefined" ? document : null);
  if (!target || installed) return;
  installed = true;

  target.addEventListener("click", (event) => {
    const button = event.target.closest ? event.target.closest("button.viz-toggle") : null;
    if (button) {
      event.preventDefault();
      handleToggleButton(button);
      return;
    }
    const node = event.target.closest
      ? event.target.closest(".viz-node, .viz-cld-node, .viz-stock, .viz-aux")
      : null;
    const figure = figureOf(event.target);
    if (!figure) return;
    if (!node) {
      if (event.target.closest(".viz-canvas")) applyFocus(figure, null);
      return;
    }
    const id = node.getAttribute("data-node");
    applyFocus(figure, figure.dataset.focus === id ? null : id);
  });

  const report = (event) => {
    const carrier = event.target.closest
      ? event.target.closest("[data-detail], [data-label]")
      : null;
    const figure = figureOf(event.target);
    if (!figure) return;
    if (carrier) {
      setReadout(figure, describe(carrier));
      return;
    }
    if (event.type === "focusout" || event.type === "pointerleave") setReadout(figure, "");
  };
  target.addEventListener("pointerover", report);
  target.addEventListener("focusin", report);
  target.addEventListener("focusout", report);

  target.addEventListener("pointermove", (event) => {
    const overlay = event.target.closest ? event.target.closest(".viz-plot-overlay") : null;
    if (!overlay) return;
    const figure = figureOf(overlay);
    const svg = figure && figure.querySelector("svg");
    if (!figure || !svg) return;
    const userX = toUserX(svg, event.clientX);
    if (userX === null) return;
    const left = Number(overlay.getAttribute("data-plot-left"));
    const right = Number(overlay.getAttribute("data-plot-right"));
    const xMin = Number(overlay.getAttribute("data-x-min"));
    const xMax = Number(overlay.getAttribute("data-x-max"));
    const dataX = xMin + ((userX - left) / (right - left)) * (xMax - xMin);
    if (DRAG.active && DRAG.figure === figure) {
      const brush = figure.querySelector("[data-viz-brush]");
      if (brush) {
        const from = Math.min(DRAG.startView, userX);
        const to = Math.max(DRAG.startView, userX);
        brush.setAttribute("x", String(Math.max(left, from)));
        brush.setAttribute("width", String(Math.max(0, Math.min(right, to) - Math.max(left, from))));
        brush.removeAttribute("hidden");
      }
      DRAG.endData = dataX;
      return;
    }
    crosshairFor(figure, dataX);
  });

  target.addEventListener("pointerdown", (event) => {
    const overlay = event.target.closest ? event.target.closest(".viz-plot-overlay") : null;
    if (!overlay) return;
    const figure = figureOf(overlay);
    const svg = figure && figure.querySelector("svg");
    if (!figure || !svg) return;
    const userX = toUserX(svg, event.clientX);
    if (userX === null) return;
    const left = Number(overlay.getAttribute("data-plot-left"));
    const right = Number(overlay.getAttribute("data-plot-right"));
    const xMin = Number(overlay.getAttribute("data-x-min"));
    const xMax = Number(overlay.getAttribute("data-x-max"));
    DRAG.active = true;
    DRAG.figure = figure;
    DRAG.overlay = overlay;
    DRAG.startView = userX;
    DRAG.startData = xMin + ((userX - left) / (right - left)) * (xMax - xMin);
    DRAG.endData = DRAG.startData;
  });

  target.addEventListener("pointerup", () => {
    if (!DRAG.active) return;
    const figure = DRAG.figure;
    const from = Math.min(DRAG.startData, DRAG.endData);
    const to = Math.max(DRAG.startData, DRAG.endData);
    DRAG.active = false;
    DRAG.figure = null;
    DRAG.overlay = null;
    if (!figure) return;
    const brush = figure.querySelector("[data-viz-brush]");
    if (brush) {
      brush.setAttribute("width", "0");
      brush.setAttribute("hidden", "hidden");
    }
    const spec = specOf(figure);
    if (!spec || !Number.isFinite(from) || !Number.isFinite(to)) return;
    const span = Math.abs(to - from);
    const full = Array.isArray(spec.x) ? spec.x.filter(Number.isFinite) : [];
    const fullSpan = full.length ? Math.max(...full) - Math.min(...full) : 0;
    if (!fullSpan || span < fullSpan * 0.02) return;
    rerender(figure, { xDomain: [from, to] });
  });

  target.addEventListener("pointerleave", clearCrosshairs, true);

  target.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    document.querySelectorAll("figure.viz[data-focus]").forEach((figure) => applyFocus(figure, null));
    clearCrosshairs();
  });
}

export const renderers = RENDERERS;
