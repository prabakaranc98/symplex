"use strict";
/**
 * Stock-and-flow diagram in standard system-dynamics notation.
 *
 *   stock       rectangle, the accumulation
 *   flow        double-line pipe with a valve (bowtie) at its midpoint
 *   cloud       the boundary of the model - a source or sink outside it
 *   auxiliary   small circle, a computed quantity that is not accumulated
 *   info link   thin dashed arrow, information rather than material
 *
 * The distinction that matters: a pipe moves the quantity, an info link only
 * informs a rate. Drawing them identically is the mistake this notation exists
 * to prevent.
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
} from "./svg.js";
import { INK, MUTED, LINE, PAPER, PALE, ACCENT, POLARITY } from "./color.js";
import { legend } from "./legend.js";
import { layeredLayout, curvePath, orthogonalPath } from "./layout.js";

const asArray = (value) => (Array.isArray(value) ? value : []);

/** A cloud glyph: the model boundary, drawn where material comes from or goes. */
function cloud(x, y, scale = 1) {
  const r = 11 * scale;
  return el("path", {
    d:
      `M${num(x - r * 1.5)} ${num(y + r * 0.4)}` +
      `a${num(r * 0.62)} ${num(r * 0.62)} 0 0 1 ${num(r * 0.35)} ${num(-r * 0.95)}` +
      `a${num(r * 0.72)} ${num(r * 0.72)} 0 0 1 ${num(r * 1.15)} ${num(-r * 0.3)}` +
      `a${num(r * 0.66)} ${num(r * 0.66)} 0 0 1 ${num(r * 1.2)} ${num(r * 0.45)}` +
      `a${num(r * 0.58)} ${num(r * 0.58)} 0 0 1 ${num(-r * 0.2)} ${num(r * 0.8)}` +
      "Z",
    fill: PAPER,
    stroke: LINE,
    "stroke-width": 1.2,
    class: "viz-cloud",
  });
}

/** A valve: the bowtie that marks where a rate is set on a pipe. */
function valve(x, y, size = 7) {
  return el("path", {
    d:
      `M${num(x - size)} ${num(y - size * 0.85)}L${num(x + size)} ${num(y + size * 0.85)}` +
      `L${num(x + size)} ${num(y - size * 0.85)}L${num(x - size)} ${num(y + size * 0.85)}Z`,
    fill: PAPER,
    stroke: INK,
    "stroke-width": 1.3,
    class: "viz-valve",
  });
}

/**
 * Draw a stock-and-flow model.
 *
 * model:
 *   stocks:      [{id, label, unit, detail, initial}]
 *   flows:       [{id, label, from, to, detail, unit}]   from/to null means a cloud
 *   auxiliaries: [{id, label, detail, unit}]
 *   links:       [{source, target, polarity}]            information links
 */
export function stockFlowDiagram({
  stocks,
  flows,
  auxiliaries = [],
  links = [],
  width = 820,
  eyebrow = "STOCK AND FLOW",
  title = "",
  note = "",
  caption = "",
  id = null,
  readout = "Hover or focus a stock, flow or auxiliary to read its declared meaning.",
} = {}) {
  const stockList = asArray(stocks).filter((stock) => stock && stock.id != null);
  const flowList = asArray(flows).filter((flow) => flow && flow.id != null);
  if (!stockList.length) {
    return emptyPanel({
      kind: "stock-flow",
      eyebrow,
      title: "No accumulation is declared.",
      message:
        "A stock-and-flow picture needs at least one quantity that accumulates. None was supplied for this model.",
      caption:
        "The absence of a stock here means none was declared, not that the system has no accumulations.",
    });
  }

  // Clouds become real nodes so the layout can place them like anything else.
  const nodes = stockList.map((stock) => ({
    id: `stock:${stock.id}`,
    kind: "stock",
    label: stock.label || stock.id,
    width: Math.max(96, Math.min(180, 34 + String(stock.label || stock.id).length * 6.4)),
    height: 46,
    source: stock,
  }));
  const edges = [];
  flowList.forEach((flow, index) => {
    let from = flow.from ? `stock:${flow.from}` : null;
    let to = flow.to ? `stock:${flow.to}` : null;
    if (!from) {
      from = `cloud:in:${flow.id || index}`;
      nodes.push({ id: from, kind: "cloud", label: "", width: 40, height: 34, source: null });
    }
    if (!to) {
      to = `cloud:out:${flow.id || index}`;
      nodes.push({ id: to, kind: "cloud", label: "", width: 40, height: 34, source: null });
    }
    edges.push({ id: flow.id, source: from, target: to, flow });
  });

  const layout = layeredLayout(
    { nodes, edges },
    { direction: "LR", layerGap: 108, nodeGap: 34, nodeWidth: 140, nodeHeight: 46, padding: 22 },
  );
  const positions = new Map(layout.nodes.map((node) => [node.id, node]));
  const auxList = asArray(auxiliaries).filter((aux) => aux && aux.id != null);
  const bandHeight = layout.height || 200;
  const auxRow = bandHeight + (auxList.length ? 56 : 0);
  const svgHeight = auxRow + (auxList.length ? 46 : 16);
  const svgWidth = Math.max(320, layout.width || width);
  const auxPositions = new Map();
  auxList.forEach((aux, index) => {
    const spacing = svgWidth / (auxList.length + 1);
    auxPositions.set(String(aux.id), { x: spacing * (index + 1), y: auxRow });
  });

  const seed = `${stockList.map((s) => s.id).join(",")}|${flowList
    .map((f) => `${f.from}>${f.to}`)
    .join(",")}|${auxList.map((a) => a.id).join(",")}`;
  const pipeArrow = uid("viz-pipe-arrow", INK);
  const infoArrow = uid("viz-info-arrow", MUTED);

  const cloudMarkup = layout.nodes
    .filter((node) => String(node.id).startsWith("cloud:"))
    .map((node) =>
      el(
        "g",
        { class: "viz-cloud-group", "aria-hidden": "true" },
        cloud(node.x, node.y) + el("title", {}, "Model boundary: a source or sink outside this model"),
      ),
    )
    .join("");

  const flowMarkup = edges
    .map((edge) => {
      const from = positions.get(edge.source);
      const to = positions.get(edge.target);
      if (!from || !to) return "";
      const startX = from.x + from.width / 2 + 2;
      const endX = to.x - to.width / 2 - 8;
      const route = [
        [startX, from.y],
        [(startX + endX) / 2, from.y],
        [(startX + endX) / 2, to.y],
        [endX, to.y],
      ];
      const path = orthogonalPath(route, { direction: "LR", radius: 6 });
      if (!path) return "";
      const midX = (startX + endX) / 2;
      const midY = (from.y + to.y) / 2;
      const flow = edge.flow || {};
      const label = wrapLabel(String(flow.label || flow.id || "flow"), 20, 1)[0] || "";
      return el(
        "g",
        {
          class: "viz-flow",
          "data-node": flow.id,
          "data-label": flow.label || flow.id,
          "data-detail": flow.detail || "No rate mechanism stated.",
          tabindex: "0",
          role: "graphics-symbol",
          "aria-label": `Flow ${flow.label || flow.id}. ${flow.detail || ""}`,
        },
        [
          el("path", { d: path, fill: "none", stroke: PALE, "stroke-width": 9, class: "viz-pipe-outer" }),
          el("path", {
            d: path,
            fill: "none",
            stroke: INK,
            "stroke-width": 6.4,
            class: "viz-pipe-fill",
            "stroke-opacity": 0.06,
          }),
          el("path", {
            d: path,
            fill: "none",
            stroke: INK,
            "stroke-width": 9,
            "stroke-linecap": "butt",
            class: "viz-pipe-walls",
            "stroke-opacity": 0,
          }),
          el("path", { d: path, fill: "none", stroke: INK, "stroke-width": 1.1, class: "viz-pipe-edge", transform: "translate(0 -4.5)" }),
          el("path", {
            d: path,
            fill: "none",
            stroke: INK,
            "stroke-width": 1.1,
            class: "viz-pipe-edge",
            transform: "translate(0 4.5)",
            "marker-end": `url(#${pipeArrow})`,
          }),
          valve(midX, midY),
          text(midX, midY + 22, label, { "text-anchor": "middle", class: "viz-flow-label", fill: INK }),
          flow.unit
            ? text(midX, midY + 33, String(flow.unit), {
                "text-anchor": "middle",
                class: "viz-unit",
                fill: MUTED,
              })
            : "",
          el("title", {}, escapeText(`${flow.label || flow.id}: ${flow.detail || "No rate mechanism stated."}`)),
        ],
      );
    })
    .join("");

  const stockMarkup = stockList
    .map((stock) => {
      const at = positions.get(`stock:${stock.id}`);
      if (!at) return "";
      const lines = wrapLabel(String(stock.label || stock.id), Math.floor(at.width / 6.2), 2);
      return el(
        "g",
        {
          class: "viz-stock",
          "data-node": stock.id,
          "data-label": stock.label || stock.id,
          "data-detail": stock.detail || "No meaning stated.",
          tabindex: "0",
          role: "button",
          "aria-label": `Stock ${stock.label || stock.id}${stock.unit ? ` in ${stock.unit}` : ""}. ${
            stock.detail || ""
          }`,
        },
        [
          el("rect", {
            x: at.x - at.width / 2,
            y: at.y - at.height / 2,
            width: at.width,
            height: at.height,
            rx: 3,
            fill: PAPER,
            stroke: INK,
            "stroke-width": 1.4,
            class: "viz-stock-box",
          }),
          ...lines.map((line, i) =>
            text(at.x, at.y + (lines.length > 1 ? -2 : 2) + i * 12, line, {
              "text-anchor": "middle",
              class: "viz-node-title",
              fill: INK,
            }),
          ),
          stock.unit
            ? text(at.x, at.y + at.height / 2 - 6, String(stock.unit), {
                "text-anchor": "middle",
                class: "viz-unit",
                fill: MUTED,
              })
            : "",
          el("title", {}, escapeText(`${stock.label || stock.id}: ${stock.detail || "No meaning stated."}`)),
        ],
      );
    })
    .join("");

  const auxMarkup = auxList
    .map((aux) => {
      const at = auxPositions.get(String(aux.id));
      if (!at) return "";
      return el(
        "g",
        {
          class: "viz-aux",
          "data-node": aux.id,
          "data-label": aux.label || aux.id,
          "data-detail": aux.detail || "No definition stated.",
          tabindex: "0",
          role: "button",
          "aria-label": `Auxiliary ${aux.label || aux.id}. ${aux.detail || ""}`,
        },
        [
          el("circle", { cx: at.x, cy: at.y, r: 9, fill: PAPER, stroke: MUTED, "stroke-width": 1.3 }),
          text(at.x, at.y + 25, wrapLabel(String(aux.label || aux.id), 22, 1)[0] || "", {
            "text-anchor": "middle",
            class: "viz-aux-label",
            fill: MUTED,
          }),
          el("title", {}, escapeText(`${aux.label || aux.id}: ${aux.detail || "No definition stated."}`)),
        ],
      );
    })
    .join("");

  const anchorFor = (nodeId) => {
    const key = String(nodeId);
    if (auxPositions.has(key)) return auxPositions.get(key);
    const stockAt = positions.get(`stock:${key}`);
    if (stockAt) return { x: stockAt.x, y: stockAt.y + stockAt.height / 2 };
    const edge = edges.find((entry) => String(entry.id) === key);
    if (edge) {
      const from = positions.get(edge.source);
      const to = positions.get(edge.target);
      if (from && to) {
        const startX = from.x + from.width / 2 + 2;
        const endX = to.x - to.width / 2 - 8;
        return { x: (startX + endX) / 2, y: (from.y + to.y) / 2 };
      }
    }
    return null;
  };

  const linkMarkup = asArray(links)
    .map((link) => {
      const from = anchorFor(link.source);
      const to = anchorFor(link.target);
      if (!from || !to) return "";
      const mid = [(from.x + to.x) / 2 + (to.y - from.y) * 0.12, (from.y + to.y) / 2 - Math.abs(to.x - from.x) * 0.08];
      const path = curvePath([[from.x, from.y], mid, [to.x, to.y]], { tension: 0.5 });
      if (!path) return "";
      const style =
        link.polarity === "positive"
          ? POLARITY.positive
          : link.polarity === "negative"
            ? POLARITY.negative
            : POLARITY.unknown;
      return el(
        "g",
        { class: "viz-info-link", "aria-hidden": "true" },
        el("path", {
          d: path,
          fill: "none",
          stroke: style.color,
          "stroke-width": 1.1,
          "stroke-dasharray": "3 3",
          "marker-end": `url(#${infoArrow})`,
        }) + el("title", {}, escapeText(`Information link: ${link.source} informs ${link.target}`)),
      );
    })
    .join("");

  const svg = svgFrame({
    width: svgWidth,
    height: svgHeight,
    role: "group",
    label: title || "Stock and flow diagram",
    desc: `${stockList.length} stock${stockList.length === 1 ? "" : "s"}, ${flowList.length} flow${
      flowList.length === 1 ? "" : "s"
    } and ${auxList.length} auxiliary quantit${auxList.length === 1 ? "y" : "ies"} in system-dynamics notation.`,
    seed,
    className: "viz-sf-svg",
    defs: arrowMarker(pipeArrow, INK, { width: 9, refX: 8 }) + arrowMarker(infoArrow, MUTED, { width: 6, refX: 5.6 }),
    children:
      el("g", { class: "viz-layer viz-layer-flows" }, cloudMarkup + flowMarkup) +
      el("g", { class: "viz-layer viz-layer-stocks" }, stockMarkup) +
      el("g", { class: "viz-layer viz-layer-auxiliaries" }, auxMarkup + linkMarkup),
  });

  return figure({
    kind: "stock-flow",
    id,
    eyebrow,
    title,
    note,
    svg:
      svg +
      `<p class="viz-readout" role="status" data-viz-readout data-default="${escapeText(
        readout,
      )}">${escapeText(readout)}</p>` +
      legend({
        title: "NOTATION",
        items: [
          { label: "Stock - a quantity that accumulates", color: INK, block: true },
          { label: "Flow - a rate that moves the quantity", color: INK, dash: "", shape: "diamond" },
          { label: "Cloud - the model boundary", color: LINE, block: true },
          { label: "Auxiliary - computed, not accumulated", color: MUTED, shape: "circle", dash: "" },
          { label: "Information link - informs a rate", color: MUTED, dash: "3 3", shape: "cross" },
        ],
      }),
    caption:
      caption ||
      "Stocks accumulate, flows are rates and clouds mark the model boundary; information links inform a rate without moving anything. The structure is as declared - no rate has been estimated, integrated or validated here.",
    scroll: true,
    minWidth: Math.min(1200, svgWidth),
  });
}

/**
 * A stock-and-flow *reading* of a ComplexSystemSpec.
 *
 * ComplexSystemSpec does not declare which quantities accumulate, so this is an
 * interpretation and is labelled as one: states are read as stocks, mechanistic
 * and measurement components that consume one state and produce another are
 * read as flows, and decision or exogenous states become auxiliaries.
 */
export function stockFlowFromSpec(spec) {
  const source = spec || {};
  const units = new Map(asArray(source.units).map((unit) => [unit.id, unit]));
  const states = asArray(source.states);
  const stockStates = states.filter((state) => state.kind === "observed" || state.kind === "latent");
  const auxStates = states.filter((state) => state.kind === "decision" || state.kind === "exogenous");
  const stockIds = new Set(stockStates.map((state) => state.id));
  const flows = [];
  const links = [];
  for (const component of asArray(source.components)) {
    const inputs = asArray(component.ports).filter((port) => port.direction === "input");
    const outputs = asArray(component.ports).filter((port) => port.direction === "output");
    const target = outputs.find((port) => stockIds.has(port.state_id));
    if (!target) continue;
    const origin = inputs.find((port) => stockIds.has(port.state_id) && port.state_id !== target.state_id);
    flows.push({
      id: component.id,
      label: component.name || component.id,
      from: origin ? origin.state_id : null,
      to: target.state_id,
      unit: units.get(target.unit_id)?.symbol || "",
      detail: component.mechanism || "No mechanism stated.",
    });
    for (const port of inputs) {
      if (stockIds.has(port.state_id) && origin && port.state_id === origin.state_id) continue;
      links.push({ source: port.state_id, target: component.id, polarity: "unknown" });
    }
  }
  return {
    stocks: stockStates.map((state) => ({
      id: state.id,
      label: state.name || state.id,
      unit: units.get(state.unit_id)?.symbol || "",
      detail: state.meaning || "No meaning stated.",
    })),
    flows,
    auxiliaries: auxStates.map((state) => ({
      id: state.id,
      label: state.name || state.id,
      unit: units.get(state.unit_id)?.symbol || "",
      detail: state.meaning || "No meaning stated.",
    })),
    links,
  };
}
