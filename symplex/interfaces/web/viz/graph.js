"use strict";
/**
 * The system graph: typed components joined by declared couplings.
 *
 * Replaces the fixed three-column grid that used to stand in for a layout.
 * Nodes are placed by Sugiyama layering so the direction of information flow
 * is left to right and crossings are minimised; a graph with no usable layer
 * structure (a dense mesh, or one the caller marks) falls back to the
 * deterministic force layout.
 *
 * Interaction is deliberately conservative so it survives the workspace's
 * innerHTML re-render:
 *   - every node and edge carries a <title>, so a plain browser tooltip works
 *     with no JavaScript at all;
 *   - a readout strip below the diagram is filled in by viz/interact.js on
 *     hover and on keyboard focus, showing ports, units and time alignment;
 *   - clicking a node sets data-focus on the figure, which CSS uses to dim
 *     everything that is not adjacent to it.
 *
 * Nothing here asserts that a coupling is real. The caption says so.
 */

import {
  el,
  text,
  escapeText,
  wrapLabel,
  uid,
  figure,
  emptyPanel,
  svgFrame,
  arrowMarker,
  num,
  coord,
} from "./svg.js";
import { INK, MUTED, LINE, PAPER, PALE, ACCENT, COMPONENT_KIND, kindStyle } from "./color.js";
import { legend } from "./legend.js";
import {
  layeredLayout,
  forceLayout,
  orthogonalPath,
  curvePath,
  selfLoopPath,
  bundleParallel,
  offsetRoute,
  anchorOnRect,
} from "./layout.js";

/** Dash patterns for the declared time alignment of a coupling. */
export const TIME_ALIGNMENT = {
  synchronous: { dash: "", label: "Synchronous" },
  sample_hold: { dash: "6 3", label: "Sample and hold" },
  interpolate: { dash: "2 3", label: "Interpolated" },
  aggregate: { dash: "9 3 2 3", label: "Aggregated" },
  event_mapping: { dash: "12 4", label: "Event mapping" },
};

const asArray = (value) => (Array.isArray(value) ? value : []);

function alignmentStyle(key) {
  return TIME_ALIGNMENT[key] || { dash: "1 4", label: "Alignment not stated" };
}

/**
 * Build {nodes, edges} for the system graph from a ComplexSystemSpec.
 * Exported so callers can count, filter or reuse the model without drawing it.
 */
export function componentGraphModel(spec) {
  const source = spec || {};
  const states = new Map(asArray(source.states).map((state) => [state.id, state]));
  const units = new Map(asArray(source.units).map((unit) => [unit.id, unit]));
  const scales = new Map(asArray(source.boundary?.scales).map((scale) => [scale.id, scale]));
  const portOwner = new Map();
  const ports = new Map();
  for (const component of asArray(source.components)) {
    for (const port of asArray(component.ports)) {
      portOwner.set(port.id, component.id);
      ports.set(port.id, port);
    }
  }
  const describePort = (portId) => {
    const port = ports.get(portId);
    if (!port) return { label: portId || "Unknown port", detail: "Port not declared in this specification." };
    const state = states.get(port.state_id);
    const unit = units.get(port.unit_id);
    const scale = scales.get(port.scale_id);
    return {
      label: state?.name || port.state_id || port.id,
      unit: unit?.symbol || port.unit_id || "unit not declared",
      dimension: unit?.dimension || "dimension not declared",
      clock: scale?.clock || "clock not declared",
      step:
        scale && scale.step_seconds != null
          ? `${scale.step_seconds} s step`
          : "event indexed or unstated step",
      detail: `${state?.name || port.state_id} in ${unit?.symbol || "unstated unit"} on ${
        scale?.name || port.scale_id
      }`,
    };
  };

  const nodes = asArray(source.components).map((component) => {
    const inputs = asArray(component.ports).filter((port) => port.direction === "input");
    const outputs = asArray(component.ports).filter((port) => port.direction === "output");
    return {
      id: component.id,
      label: component.name || component.id || "Component",
      kind: component.kind,
      inputs: inputs.length,
      outputs: outputs.length,
      mechanism: component.mechanism || "",
      assumptions: asArray(component.assumptions).length,
      evidence: asArray(component.evidence_ids).length,
      href: null,
      detail: [
        component.mechanism || "No mechanism stated.",
        `${inputs.length} input port${inputs.length === 1 ? "" : "s"}, ${outputs.length} output port${
          outputs.length === 1 ? "" : "s"
        }.`,
        asArray(component.evidence_ids).length
          ? `${asArray(component.evidence_ids).length} linked evidence artifact(s).`
          : "No evidence artifacts linked.",
      ].join(" "),
    };
  });

  const edges = asArray(source.couplings)
    .map((coupling) => {
      const sourceComponent = portOwner.get(coupling.source_port);
      const targetComponent = portOwner.get(coupling.target_port);
      if (!sourceComponent || !targetComponent) return null;
      const from = describePort(coupling.source_port);
      const to = describePort(coupling.target_port);
      const conversion = coupling.conversion;
      const conversionText = conversion
        ? `target = source x ${conversion.scale} + ${conversion.offset}`
        : "No affine conversion declared";
      return {
        id: coupling.id,
        source: sourceComponent,
        target: targetComponent,
        alignment: coupling.time_alignment,
        label: alignmentStyle(coupling.time_alignment).label,
        detail: [
          `${from.label} (${from.unit}) to ${to.label} (${to.unit}).`,
          `${conversionText}.`,
          `${alignmentStyle(coupling.time_alignment).label}: ${
            coupling.alignment_rationale || "no rationale stated"
          }.`,
          `Clocks: ${from.clock} to ${to.clock}. Steps: ${from.step} to ${to.step}.`,
        ].join(" "),
        mechanism: coupling.mechanism || "",
      };
    })
    .filter(Boolean);

  return { nodes, edges };
}

function nodeBox(node, compact) {
  const label = node.label || node.id;
  const width = compact
    ? Math.max(104, Math.min(150, 30 + label.length * 5.4))
    : Math.max(132, Math.min(196, 42 + label.length * 6.1));
  return { width, height: compact ? 40 : 52 };
}

function renderNode(node, placed, compact, focusable) {
  const style = kindStyle(COMPONENT_KIND, node.kind);
  const width = placed.width;
  const height = placed.height;
  const x = placed.x - width / 2;
  const y = placed.y - height / 2;
  const maxChars = Math.max(8, Math.floor(width / (compact ? 5.6 : 6.2)));
  const lines = wrapLabel(node.label || node.id, maxChars, compact ? 1 : 2);
  const titleY = compact ? placed.y + 3.5 : placed.y - (lines.length > 1 ? 5 : 0);
  const body = [
    el("rect", {
      x,
      y,
      width,
      height,
      rx: 7,
      class: "viz-node-box",
      fill: PAPER,
      stroke: LINE,
      "stroke-width": 1,
    }),
    el("rect", { x, y, width: 4, height, rx: 2, fill: style.color, class: "viz-node-kind" }),
    ...lines.map((line, i) =>
      text(x + 12, titleY + i * 12, line, { class: "viz-node-title", fill: INK }),
    ),
    compact
      ? ""
      : text(x + 12, y + height - 9, `${style.label} - ${node.inputs} in / ${node.outputs} out`, {
          class: "viz-node-kind-label",
          fill: MUTED,
        }),
    el("title", {}, escapeText(`${node.label}. ${node.detail || ""}`)),
  ];
  return el(
    "g",
    {
      class: "viz-node",
      "data-node": node.id,
      "data-detail": node.detail || "",
      "data-label": node.label || node.id,
      "data-kind": style.label,
      tabindex: focusable ? "0" : null,
      role: focusable ? "button" : null,
      "aria-label": `${node.label}. ${style.label} component. ${node.inputs} inputs, ${node.outputs} outputs.`,
    },
    body,
  );
}

function edgeGeometry(edge, positions, compact) {
  const from = positions.get(edge.source);
  const to = positions.get(edge.target);
  if (!from || !to) return null;
  if (edge.source === edge.target) return { selfLoop: true, node: from };
  const points = edge.points && edge.points.length >= 2 ? edge.points.slice() : [[from.x, from.y], [to.x, to.y]];
  const trimStart = anchorOnRect(
    [from.x, from.y],
    points[1] || [to.x, to.y],
    from.width + 3,
    from.height + 3,
  );
  const trimEnd = anchorOnRect(
    [to.x, to.y],
    points[points.length - 2] || [from.x, from.y],
    to.width + 11,
    to.height + 11,
  );
  const middle = points.slice(1, -1);
  const route = [trimStart, ...middle, trimEnd];
  const offset = edge.bundleOffset || 0;
  return {
    selfLoop: false,
    route: offset && !middle.length ? offsetRoute(route, offset) : route,
    curved: Boolean(offset) || middle.length > 0,
    compact,
  };
}

/**
 * Draw a typed directed graph.
 *
 * options:
 *   width, height   preferred pixel box (the layout may grow the height)
 *   layout          "layered" (default) or "force"
 *   direction       "LR" (default) or "TB", layered only
 *   compact         smaller nodes; chosen automatically above 24 nodes
 *   title/eyebrow/note/caption   figure furniture
 *   kindTable       colour/glyph table keyed by node.kind
 *   legendTitle     heading for the legend
 *   readout         default text for the hover/focus readout strip
 *   id              stable DOM id, needed for focus interaction
 */
export function graphDiagram({
  nodes,
  edges,
  width = 860,
  height = 0,
  layout = "layered",
  direction = "LR",
  compact = null,
  kindTable = COMPONENT_KIND,
  eyebrow = "SYSTEM GRAPH",
  title = "",
  note = "",
  caption = "",
  legendTitle = "COMPONENT KIND",
  readout = "Hover or focus a component to read its ports, units and time alignment.",
  id = null,
  emptyTitle = "No graph to draw.",
  emptyMessage = "This specification declares no components.",
} = {}) {
  const list = asArray(nodes).filter((node) => node && node.id !== undefined && node.id !== null);
  if (!list.length) {
    return emptyPanel({
      kind: "graph",
      eyebrow,
      title: emptyTitle,
      message: emptyMessage,
      caption: caption || "Nothing is drawn because nothing was declared. An empty diagram is not evidence of a simple system.",
    });
  }
  const dense = compact === null ? list.length > 24 : compact;
  const sized = list.map((node) => ({ ...node, ...nodeBox(node, dense) }));
  const bundled = bundleParallel(asArray(edges).filter((edge) => edge && edge.source !== undefined));
  const linking = bundled.filter((edge) => edge.source !== edge.target);
  const loops = bundled.filter((edge) => edge.source === edge.target);

  const useForce = layout === "force";
  const placement = useForce
    ? forceLayout(
        { nodes: sized, edges: linking },
        { width, height: height || Math.max(360, Math.min(720, 150 + sized.length * 14)) },
      )
    : layeredLayout(
        { nodes: sized, edges: linking },
        {
          direction,
          nodeGap: dense ? 12 : 18,
          layerGap: dense ? 68 : 104,
          nodeWidth: dense ? 120 : 160,
          nodeHeight: dense ? 40 : 52,
          sweeps: sized.length > 60 ? 4 : 8,
        },
      );

  const positions = new Map(
    placement.nodes.map((placed) => [
      placed.id,
      {
        x: placed.x,
        y: placed.y,
        width: placed.width || (dense ? 120 : 160),
        height: placed.height || (dense ? 40 : 52),
      },
    ]),
  );
  const byId = new Map(sized.map((node) => [String(node.id), node]));
  const placedEdges = useForce ? placement.edges : placement.edges;

  const seed = `${list.map((node) => node.id).join(",")}|${linking
    .map((edge) => `${edge.source}>${edge.target}`)
    .join(",")}`;
  const arrowId = uid("viz-arrow", seed);
  const arrowFocusId = uid("viz-arrow-focus", seed);

  const svgWidth = Math.max(120, placement.width || width);
  const svgHeight = Math.max(120, placement.height || height || 360);

  const edgeMarkup = placedEdges
    .map((edge) => {
      const geometry = edgeGeometry(edge, positions, dense);
      if (!geometry || geometry.selfLoop) return "";
      const style = alignmentStyle(edge.alignment);
      const path = geometry.curved
        ? curvePath(geometry.route, { tension: 0.5 })
        : orthogonalPath(geometry.route, { direction, radius: dense ? 5 : 8 });
      if (!path) return "";
      return el(
        "g",
        {
          class: "viz-edge",
          "data-edge": edge.id ?? "",
          "data-source": edge.source,
          "data-target": edge.target,
          "data-detail": edge.detail || "",
          "data-label": edge.label || style.label,
          tabindex: "0",
          role: "graphics-symbol",
          "aria-label": `Coupling from ${byId.get(String(edge.source))?.label || edge.source} to ${
            byId.get(String(edge.target))?.label || edge.target
          }. ${style.label}.`,
        },
        [
          el("path", { d: path, class: "viz-edge-hit", fill: "none", stroke: "transparent", "stroke-width": 12 }),
          el("path", {
            d: path,
            class: "viz-edge-line",
            fill: "none",
            stroke: MUTED,
            "stroke-width": dense ? 1.2 : 1.5,
            "stroke-dasharray": style.dash || null,
            "marker-end": `url(#${arrowId})`,
          }),
          el("title", {}, escapeText(`${edge.id ? edge.id + ": " : ""}${edge.detail || style.label}`)),
        ],
      );
    })
    .join("");

  const loopMarkup = loops
    .map((edge, i) => {
      const placed = positions.get(edge.source);
      if (!placed) return "";
      const path = selfLoopPath({ ...placed }, { index: edge.bundleIndex || 0, size: dense ? 18 : 26 });
      if (!path) return "";
      const style = alignmentStyle(edge.alignment);
      return el(
        "g",
        {
          class: "viz-edge viz-edge-self",
          "data-edge": edge.id ?? `self-${i}`,
          "data-source": edge.source,
          "data-target": edge.target,
          "data-detail": edge.detail || "",
          "data-label": `Self coupling - ${edge.label || style.label}`,
          tabindex: "0",
          role: "graphics-symbol",
          "aria-label": `Self coupling on ${byId.get(String(edge.source))?.label || edge.source}.`,
        },
        [
          el("path", {
            d: path,
            fill: "none",
            stroke: MUTED,
            "stroke-width": 1.4,
            "stroke-dasharray": style.dash || null,
            "marker-end": `url(#${arrowId})`,
            class: "viz-edge-line",
          }),
          el("title", {}, escapeText(`${edge.id || "Self coupling"}: ${edge.detail || style.label}`)),
        ],
      );
    })
    .join("");

  const nodeMarkup = sized
    .map((node) => {
      const placed = positions.get(String(node.id));
      if (!placed) return "";
      return renderNode(node, placed, dense, true);
    })
    .join("");

  const usedKinds = [];
  for (const node of sized) {
    const key = node.kind;
    if (usedKinds.some((entry) => entry.key === key)) continue;
    const style = kindStyle(kindTable, key);
    usedKinds.push({ key, label: style.label, color: style.color });
  }
  const usedAlignments = [];
  for (const edge of linking.concat(loops)) {
    if (usedAlignments.some((entry) => entry.key === edge.alignment)) continue;
    usedAlignments.push({ key: edge.alignment, ...alignmentStyle(edge.alignment) });
  }

  const svg = svgFrame({
    width: svgWidth,
    height: svgHeight,
    role: "group",
    label: title || "System component graph",
    desc: `${sized.length} components and ${linking.length + loops.length} declared couplings, laid out by ${
      useForce ? "a deterministic force-directed" : "a layered"
    } algorithm. ${placement.crossings === undefined ? "" : `${placement.crossings} edge crossings remain.`}`,
    seed,
    className: "viz-graph-svg",
    defs: arrowMarker(arrowId, MUTED) + arrowMarker(arrowFocusId, ACCENT),
    children:
      el("g", { class: "viz-layer viz-layer-couplings" }, edgeMarkup + loopMarkup) +
      el("g", { class: "viz-layer viz-layer-components" }, nodeMarkup),
  });

  const legendMarkup =
    legend({
      title: legendTitle,
      items: usedKinds.map((entry) => ({ label: entry.label, color: entry.color, block: true })),
    }) +
    legend({
      title: "COUPLING TIME ALIGNMENT",
      items: usedAlignments.map((entry) => ({
        label: entry.label,
        color: MUTED,
        dash: entry.dash,
        shape: "circle",
      })),
    });

  const controls = `<div class="viz-layer-toggles" role="group" aria-label="Diagram layers">${[
    ["components", "Components"],
    ["couplings", "Couplings"],
  ]
    .map(
      ([key, label]) =>
        `<button type="button" class="viz-toggle" data-viz-layer="${key}" aria-pressed="true">${escapeText(
          label,
        )}</button>`,
    )
    .join("")}</div>`;

  return figure({
    kind: "graph",
    id,
    eyebrow,
    title,
    note,
    controls,
    svg:
      svg +
      `<p class="viz-readout" role="status" data-viz-readout data-default="${escapeText(
        readout,
      )}">${escapeText(readout)}</p>` +
      legendMarkup,
    caption:
      caption ||
      "Position and layer show declared connectivity and direction of flow only. They do not represent physical location, effect size, timing accuracy, or that any coupling has been observed.",
    scroll: true,
    minWidth: Math.min(1400, Math.max(320, svgWidth)),
    className: dense ? "is-dense" : "",
  });
}

/** The system graph for a ComplexSystemSpec. */
export function systemGraph(spec, options = {}) {
  const model = componentGraphModel(spec);
  const componentCount = model.nodes.length;
  return graphDiagram({
    ...model,
    eyebrow: "TYPED COMPONENT CONNECTIONS",
    title: options.title || "How the declared components are wired",
    note:
      componentCount > 24
        ? "Compact layout: this specification declares more components than fit at full size."
        : "",
    id: options.id || "viz-system-graph",
    layout: options.layout || "layered",
    direction: options.direction || "LR",
    width: options.width || 900,
    caption:
      "Each box is a proposed model component and each arrow a declared coupling with its own unit conversion and time alignment. Layout shows connectivity and flow direction, not physical position, effect size or evidence. No coupling here has been executed or validated.",
    emptyTitle: "No components are declared.",
    emptyMessage:
      "A system specification needs at least one component with typed ports before its structure can be drawn.",
  });
}

/** A compact structural summary used beside the graph. */
export function graphSummary({ nodes, edges }) {
  const list = asArray(nodes);
  const links = asArray(edges);
  const incoming = new Map(list.map((node) => [String(node.id), 0]));
  const outgoing = new Map(list.map((node) => [String(node.id), 0]));
  for (const edge of links) {
    if (outgoing.has(String(edge.source))) outgoing.set(String(edge.source), outgoing.get(String(edge.source)) + 1);
    if (incoming.has(String(edge.target))) incoming.set(String(edge.target), incoming.get(String(edge.target)) + 1);
  }
  const sources = list.filter((node) => !incoming.get(String(node.id)));
  const sinks = list.filter((node) => !outgoing.get(String(node.id)));
  const isolated = list.filter(
    (node) => !incoming.get(String(node.id)) && !outgoing.get(String(node.id)),
  );
  return {
    components: list.length,
    couplings: links.length,
    sources: sources.map((node) => node.label || node.id),
    sinks: sinks.map((node) => node.label || node.id),
    isolated: isolated.map((node) => node.label || node.id),
  };
}
