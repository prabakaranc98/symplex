"use strict";
/**
 * Symplex visualisation library.
 *
 * Vanilla ES modules, no dependencies, no network. Every chart function is pure
 * and deterministic: it takes data and returns an SVG or HTML string, and the
 * same input always produces a byte-identical result.
 *
 * The workspace views are classic scripts, so this module also publishes the
 * whole surface as `window.SymplexViz` - the same convention scene-renderer.js
 * uses for `window.SymplexScene`. Views must read it defensively:
 *
 *     const viz = typeof window !== "undefined" ? window.SymplexViz : null;
 *     if (!viz) return fallbackMarkup();
 *
 * so that the DOM-free render checks in scripts/ keep working.
 */

import * as svg from "./svg.js";
import * as color from "./color.js";
import * as scale from "./scale.js";
import * as axis from "./axis.js";
import * as legend from "./legend.js";
import * as layout from "./layout.js";
import { graphDiagram, systemGraph, componentGraphModel, graphSummary, TIME_ALIGNMENT } from "./graph.js";
import { causalLoopDiagram, systemCausalLoop, causalLoopModel } from "./causal-loop.js";
import { stockFlowDiagram, stockFlowFromSpec } from "./stock-flow.js";
import { timeSeriesChart, scenarioFan } from "./timeseries.js";
import { phasePortrait, STABILITY } from "./phase.js";
import { smallMultiples } from "./smallmultiples.js";
import { tornadoDiagram, sobolChart } from "./sensitivity.js";
import { bifurcationDiagram } from "./bifurcation.js";
import { matrixHeatmap, seriate } from "./matrix.js";
import { archiveHeatmap } from "./archive.js";
import { sankeyDiagram } from "./sankey.js";
import { install } from "./interact.js";

export * from "./svg.js";
export * from "./color.js";
export * from "./scale.js";
export * from "./axis.js";
export * from "./legend.js";
export * from "./layout.js";
export * from "./graph.js";
export * from "./causal-loop.js";
export * from "./stock-flow.js";
export * from "./timeseries.js";
export * from "./phase.js";
export * from "./smallmultiples.js";
export * from "./sensitivity.js";
export * from "./bifurcation.js";
export * from "./matrix.js";
export * from "./archive.js";
export * from "./sankey.js";
export { install } from "./interact.js";

/** The public surface, frozen so a view cannot monkey-patch a chart at runtime. */
export const SymplexViz = Object.freeze({
  version: "symplex-viz-v1",

  // Diagrams
  graphDiagram,
  systemGraph,
  componentGraphModel,
  graphSummary,
  causalLoopDiagram,
  systemCausalLoop,
  causalLoopModel,
  stockFlowDiagram,
  stockFlowFromSpec,

  // Charts
  timeSeriesChart,
  scenarioFan,
  phasePortrait,
  smallMultiples,
  tornadoDiagram,
  sobolChart,
  bifurcationDiagram,
  matrixHeatmap,
  archiveHeatmap,
  sankeyDiagram,

  // Shared utilities, exposed so views can build one-off panels in the same
  // visual language rather than inventing a second one.
  seriate,
  svg,
  color,
  scale,
  axis,
  legend,
  layout,
  tables: Object.freeze({
    TIME_ALIGNMENT,
    STABILITY,
    COMPONENT_KIND: color.COMPONENT_KIND,
    STATE_KIND: color.STATE_KIND,
    POLARITY: color.POLARITY,
    LOOP: color.LOOP,
  }),

  install,
});

if (typeof window !== "undefined") {
  window.SymplexViz = SymplexViz;
  install(document);
}

export default SymplexViz;
