"use strict";
/**
 * Sankey diagram for conserved quantities.
 *
 * Node layers come from a longest-path pass, node order within a layer from a
 * few barycentre sweeps, and ribbon thickness is proportional to the supplied
 * value with one shared scale across the whole diagram.
 *
 * Because this is used for mass and flow balance, the module checks the balance
 * itself: for every internal node it compares total inflow against total
 * outflow and reports the largest imbalance in the caption. A Sankey that
 * silently renormalises a leaking budget is worse than no Sankey.
 */

import { el, text, escapeText, figure, emptyPanel, svgFrame, embedSpec, ellipsis, num } from "./svg.js";
import { INK, MUTED, LINE, PAPER, seriesStyle } from "./color.js";
import { formatNumber, formatPercent } from "./scale.js";
import { legend } from "./legend.js";

const asArray = (value) => (Array.isArray(value) ? value : []);
const isNum = (value) => typeof value === "number" && Number.isFinite(value);

function ribbonPath(x0, y0, x1, y1, thickness) {
  const curve = (x0 + x1) / 2;
  const half = thickness / 2;
  return (
    `M${num(x0)} ${num(y0 - half)}` +
    `C${num(curve)} ${num(y0 - half)} ${num(curve)} ${num(y1 - half)} ${num(x1)} ${num(y1 - half)}` +
    `L${num(x1)} ${num(y1 + half)}` +
    `C${num(curve)} ${num(y1 + half)} ${num(curve)} ${num(y0 + half)} ${num(x0)} ${num(y0 + half)}Z`
  );
}

/**
 * spec:
 *   nodes  [{id, label, unit}]
 *   links  [{source, target, value, label}]
 *   unit   the conserved quantity's unit
 */
export function sankeyDiagram(spec = {}) {
  const nodes = asArray(spec.nodes).filter((node) => node && node.id != null);
  const eyebrow = spec.eyebrow || "FLOW BALANCE";
  const links = asArray(spec.links).filter(
    (link) => link && link.source != null && link.target != null && isNum(link.value) && link.value > 0,
  );
  if (!nodes.length || !links.length) {
    return emptyPanel({
      kind: "sankey",
      eyebrow,
      title: "No flows to balance.",
      message:
        "A Sankey needs nodes and positive-valued flows between them. Zero and missing flows are not drawn.",
      caption: "An empty diagram means no flow was measured, not that nothing moves.",
    });
  }

  const index = new Map(nodes.map((node, i) => [String(node.id), i]));
  const outgoing = nodes.map(() => []);
  const incoming = nodes.map(() => []);
  const usable = [];
  for (const link of links) {
    const si = index.get(String(link.source));
    const ti = index.get(String(link.target));
    if (si === undefined || ti === undefined || si === ti) continue;
    const entry = { ...link, si, ti };
    usable.push(entry);
    outgoing[si].push(entry);
    incoming[ti].push(entry);
  }
  if (!usable.length) {
    return emptyPanel({
      kind: "sankey",
      eyebrow,
      title: "No flow connects two declared nodes.",
      message: "Every supplied flow referenced a node that is not in the node list.",
      caption: "Unresolvable references are dropped rather than drawn against a placeholder node.",
    });
  }

  // Longest-path layering over the flow graph.
  const layer = new Array(nodes.length).fill(0);
  for (let pass = 0; pass < nodes.length; pass++) {
    let changed = false;
    for (const link of usable) {
      if (layer[link.ti] < layer[link.si] + 1) {
        layer[link.ti] = layer[link.si] + 1;
        changed = true;
      }
    }
    if (!changed) break;
  }
  const layerCount = Math.max(...layer) + 1;
  const layers = Array.from({ length: layerCount }, () => []);
  nodes.forEach((node, i) => layers[layer[i]].push(i));

  const nodeTotal = nodes.map((node, i) => {
    const inflow = incoming[i].reduce((sum, link) => sum + link.value, 0);
    const outflow = outgoing[i].reduce((sum, link) => sum + link.value, 0);
    return { inflow, outflow, size: Math.max(inflow, outflow) };
  });

  const width = isNum(spec.width) ? spec.width : 780;
  const nodeWidth = 13;
  const padding = 12;
  const marginTop = 18;
  const marginBottom = 26;
  const maxLayerTotal = Math.max(
    ...layers.map((row) => row.reduce((sum, i) => sum + nodeTotal[i].size, 0)),
    1e-9,
  );
  const maxNodesInLayer = Math.max(...layers.map((row) => row.length), 1);
  const usableHeight = Math.max(180, Math.min(520, maxLayerTotal ? 320 : 240));
  const available = usableHeight - (maxNodesInLayer - 1) * padding;
  const valueScale = available > 0 ? available / maxLayerTotal : 1;
  const height = usableHeight + marginTop + marginBottom;

  // Barycentre ordering, a few deterministic sweeps.
  const position = new Map();
  layers.forEach((row) => row.forEach((node, i) => position.set(node, i)));
  for (let sweep = 0; sweep < 6; sweep++) {
    const forward = sweep % 2 === 0;
    const order = forward
      ? Array.from({ length: layerCount }, (_, i) => i)
      : Array.from({ length: layerCount }, (_, i) => layerCount - 1 - i);
    for (const l of order) {
      const neighbours = (node) =>
        (forward ? incoming[node] : outgoing[node]).map((link) =>
          position.get(forward ? link.si : link.ti),
        );
      layers[l] = layers[l]
        .map((node, i) => {
          const values = neighbours(node).filter((value) => value !== undefined);
          return { node, i, bary: values.length ? values.reduce((a, b) => a + b, 0) / values.length : i };
        })
        .sort((a, b) => a.bary - b.bary || a.i - b.i)
        .map((entry) => entry.node);
      layers[l].forEach((node, i) => position.set(node, i));
    }
  }

  const layerX = (l) =>
    layerCount === 1
      ? 40
      : 40 + ((width - 80 - nodeWidth) * l) / (layerCount - 1);
  const box = new Map();
  layers.forEach((row, l) => {
    const total = row.reduce((sum, i) => sum + nodeTotal[i].size, 0) * valueScale + (row.length - 1) * padding;
    let cursor = marginTop + Math.max(0, (usableHeight - total) / 2);
    for (const i of row) {
      const nodeHeight = Math.max(3, nodeTotal[i].size * valueScale);
      box.set(i, { x: layerX(l), y: cursor, height: nodeHeight, layer: l });
      cursor += nodeHeight + padding;
    }
  });

  const outCursor = new Map();
  const inCursor = new Map();
  const ribbons = usable
    .slice()
    .sort((a, b) => (box.get(a.ti)?.y ?? 0) - (box.get(b.ti)?.y ?? 0) || a.value - b.value)
    .map((link, i) => {
      const from = box.get(link.si);
      const to = box.get(link.ti);
      if (!from || !to) return "";
      const thickness = Math.max(1, link.value * valueScale);
      const fromOffset = outCursor.get(link.si) || 0;
      const toOffset = inCursor.get(link.ti) || 0;
      outCursor.set(link.si, fromOffset + thickness);
      inCursor.set(link.ti, toOffset + thickness);
      const y0 = from.y + fromOffset + thickness / 2;
      const y1 = to.y + toOffset + thickness / 2;
      const style = seriesStyle(link.si);
      const label = `${nodes[link.si].label || nodes[link.si].id} to ${
        nodes[link.ti].label || nodes[link.ti].id
      }: ${formatNumber(link.value, 4)}${spec.unit ? ` ${spec.unit}` : ""}`;
      return el(
        "g",
        {
          class: "viz-ribbon",
          "data-node": `${link.source}>${link.target}`,
          "data-label": label,
          "data-detail": link.label || label,
          tabindex: "0",
          role: "graphics-symbol",
          "aria-label": label,
        },
        el("path", {
          d: ribbonPath(from.x + nodeWidth, y0, to.x, y1, thickness),
          fill: style.color,
          "fill-opacity": 0.28,
          stroke: style.color,
          "stroke-opacity": 0.4,
          "stroke-width": 0.5,
        }) + el("title", {}, escapeText(label)),
      );
    })
    .join("");

  const nodeMarkup = nodes
    .map((node, i) => {
      const at = box.get(i);
      if (!at) return "";
      const style = seriesStyle(i);
      const balance = nodeTotal[i];
      const internal = incoming[i].length && outgoing[i].length;
      const drift = internal ? balance.inflow - balance.outflow : 0;
      const label = ellipsis(String(node.label || node.id), 26);
      const anchorRight = at.layer < layerCount - 1;
      return el(
        "g",
        {
          class: "viz-sankey-node",
          "data-node": node.id,
          "data-label": node.label || node.id,
          "data-detail": `In ${formatNumber(balance.inflow, 4)}, out ${formatNumber(balance.outflow, 4)}${
            spec.unit ? ` ${spec.unit}` : ""
          }.${internal ? ` Imbalance ${formatNumber(drift, 4)}.` : " Boundary node."}`,
          tabindex: "0",
          role: "graphics-symbol",
          "aria-label": `${node.label || node.id}. In ${formatNumber(balance.inflow, 3)}, out ${formatNumber(
            balance.outflow,
            3,
          )}.`,
        },
        el("rect", {
          x: at.x,
          y: at.y,
          width: nodeWidth,
          height: at.height,
          fill: style.color,
          rx: 2,
        }) +
          text(
            anchorRight ? at.x + nodeWidth + 6 : at.x - 6,
            at.y + at.height / 2 + 3.5,
            label,
            { "text-anchor": anchorRight ? "start" : "end", class: "viz-sankey-label", fill: INK },
          ) +
          el("title", {}, escapeText(`${node.label || node.id}`)),
      );
    })
    .join("");

  let worstImbalance = 0;
  let worstNode = "";
  nodes.forEach((node, i) => {
    if (!incoming[i].length || !outgoing[i].length) return;
    const drift = Math.abs(nodeTotal[i].inflow - nodeTotal[i].outflow);
    const relative = nodeTotal[i].inflow ? drift / nodeTotal[i].inflow : drift;
    if (relative > worstImbalance) {
      worstImbalance = relative;
      worstNode = node.label || node.id;
    }
  });

  const droppedZero = links.length - usable.length;

  const svg = svgFrame({
    width,
    height,
    role: "img",
    label: spec.title || "Flow diagram",
    desc: `${nodes.length} nodes and ${usable.length} flows across ${layerCount} stages. Ribbon thickness is proportional to the flow value on one shared scale.`,
    seed: `${spec.id || "sankey"}|${nodes.map((node) => node.id).join(",")}|${usable.length}`,
    className: "viz-sankey-svg",
    children:
      el("g", { class: "viz-layer viz-layer-ribbons" }, ribbons) +
      el("g", { class: "viz-layer viz-layer-nodes" }, nodeMarkup),
  });

  return figure({
    kind: "sankey",
    id: spec.id || null,
    eyebrow,
    title: spec.title || "",
    note: spec.note || "",
    svg:
      svg +
      legend({
        items: [
          { label: `Ribbon width is proportional to value${spec.unit ? ` in ${spec.unit}` : ""}`, color: MUTED, block: true },
        ],
      }) +
      embedSpec({ chart: "sankey", ...JSON.parse(JSON.stringify(spec)) }),
    caption:
      (spec.caption || "Ribbon widths are the supplied flow values on one shared scale.") +
      (worstImbalance > 0.005
        ? ` Inflow and outflow do not balance at every node: the largest gap is ${formatPercent(
            worstImbalance,
          )} at "${worstNode}". The diagram shows the values as supplied and does not renormalise them.`
        : " Inflow and outflow balance at every internal node to within half a percent.") +
      (droppedZero
        ? ` ${droppedZero} flow${droppedZero === 1 ? "" : "s"} had no positive value and ${
            droppedZero === 1 ? "is" : "are"
          } omitted rather than drawn as a hairline.`
        : ""),
    scroll: true,
    minWidth: Math.min(1200, width),
  });
}
