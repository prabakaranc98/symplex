"use strict";
/**
 * Causal loop diagram - the canonical systems-thinking picture.
 *
 * Signed links carry a "+" or a "-" at their midpoint; declared feedback loops
 * are drawn as a highlighted ring at the loop centroid with an R (reinforcing)
 * or B (balancing) badge, and the member links are thickened in the loop
 * colour.
 *
 * Honesty rule that shapes this module: ComplexSystemSpec declares polarity for
 * a *loop*, not for each *link*. This module never back-fills the missing link
 * polarities from the loop polarity, because a reinforcing loop only constrains
 * the number of negative links, not which ones they are. Unstated link polarity
 * renders as a hollow "?" and the caption says so.
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
import { INK, MUTED, LINE, PAPER, POLARITY, LOOP, ACCENT } from "./color.js";
import { legend } from "./legend.js";
import { forceLayout, circularLayout, curvePath, selfLoopPath, offsetRoute, bundleParallel } from "./layout.js";

const asArray = (value) => (Array.isArray(value) ? value : []);

function polarityStyle(value) {
  if (value === "positive" || value === "+" || value === "same") return POLARITY.positive;
  if (value === "negative" || value === "-" || value === "opposite") return POLARITY.negative;
  return POLARITY.unknown;
}

function loopStyle(value) {
  return LOOP[value] || LOOP.unknown;
}

/** Build {nodes, edges, loops} for a causal loop diagram from a ComplexSystemSpec. */
export function causalLoopModel(spec) {
  const source = spec || {};
  const portOwner = new Map();
  for (const component of asArray(source.components)) {
    for (const port of asArray(component.ports)) portOwner.set(port.id, component.id);
  }
  const nodes = asArray(source.components).map((component) => ({
    id: component.id,
    label: component.name || component.id,
    kind: component.kind,
    detail: component.mechanism || "No mechanism stated.",
  }));
  const edges = asArray(source.couplings)
    .map((coupling) => {
      const from = portOwner.get(coupling.source_port);
      const to = portOwner.get(coupling.target_port);
      if (!from || !to) return null;
      return {
        id: coupling.id,
        source: from,
        target: to,
        polarity: coupling.polarity || "unknown",
        detail: coupling.mechanism || "No mechanism stated.",
      };
    })
    .filter(Boolean);
  const loops = asArray(source.feedback).map((loop) => ({
    id: loop.id,
    members: asArray(loop.component_ids),
    polarity: loop.polarity,
    description: loop.description || "",
    nonlinearity: loop.nonlinearity || "",
    stochasticity: loop.stochasticity || "",
  }));
  return { nodes, edges, loops };
}

function centroid(points) {
  if (!points.length) return null;
  const sum = points.reduce((acc, point) => [acc[0] + point.x, acc[1] + point.y], [0, 0]);
  return { x: sum[0] / points.length, y: sum[1] / points.length };
}

/** Signed polygon area; its sign tells us which way the loop winds on screen. */
function windingSign(points) {
  if (points.length < 3) return 1;
  let area = 0;
  for (let i = 0; i < points.length; i++) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    area += a.x * b.y - b.x * a.y;
  }
  return area >= 0 ? 1 : -1;
}

/** The loop badge: a rotating ring with R or B at the centre. */
function loopBadge(loop, position, sign, index) {
  const style = loopStyle(loop.polarity);
  const radius = 15;
  const sweep = sign > 0 ? 1 : 0;
  const start = [position.x + radius, position.y];
  const end = [position.x - radius * 0.18, position.y - radius * 0.98];
  const ring = `M${num(start[0])} ${num(start[1])}A${num(radius)} ${num(radius)} 0 1 ${sweep} ${num(
    end[0],
  )} ${num(end[1])}`;
  return el(
    "g",
    {
      class: "viz-loop-badge",
      "data-loop": loop.id || `loop-${index}`,
      "data-detail": [
        `${style.label} loop.`,
        loop.description || "No description stated.",
        loop.nonlinearity ? `Nonlinearity: ${loop.nonlinearity}.` : "",
        loop.stochasticity ? `Stochasticity: ${loop.stochasticity}.` : "",
      ]
        .filter(Boolean)
        .join(" "),
      "data-label": `${loop.id || "Loop"} - ${style.label}`,
      tabindex: "0",
      role: "graphics-symbol",
      "aria-label": `${style.label} feedback loop ${loop.id || index + 1}. ${loop.description || ""}`,
    },
    [
      el("circle", {
        cx: position.x,
        cy: position.y,
        r: radius + 5,
        fill: PAPER,
        stroke: "none",
        opacity: 0.86,
      }),
      el("path", {
        d: ring,
        fill: "none",
        stroke: style.color,
        "stroke-width": 1.6,
        "marker-end": `url(#${uid("viz-loop-arrow", style.color)})`,
      }),
      text(position.x, position.y + 5, style.badge, {
        "text-anchor": "middle",
        class: "viz-loop-letter",
        fill: style.color,
      }),
      el("title", {}, escapeText(`${loop.id || "Loop"}: ${style.label}. ${loop.description || ""}`)),
    ],
  );
}

/**
 * Draw a causal loop diagram.
 *
 * nodes:  [{id, label, detail}]
 * edges:  [{id, source, target, polarity: "positive"|"negative"|"unknown", detail, delay}]
 * loops:  [{id, members:[nodeId...], polarity: "reinforcing"|"balancing"|"mixed"|"unknown", description}]
 */
export function causalLoopDiagram({
  nodes,
  edges,
  loops = [],
  width = 760,
  height = 0,
  eyebrow = "CAUSAL LOOP DIAGRAM",
  title = "",
  note = "",
  caption = "",
  id = null,
  readout = "Hover or focus a link or loop badge to read its declared mechanism.",
} = {}) {
  const list = asArray(nodes).filter((node) => node && node.id != null);
  if (!list.length) {
    return emptyPanel({
      kind: "causal-loop",
      eyebrow,
      title: "No causal structure is declared.",
      message:
        "A causal loop diagram needs components and the couplings between them. None were found in this specification.",
      caption:
        "An absent diagram means an absent declaration. It is not evidence that the system has no feedback.",
    });
  }
  const links = bundleParallel(
    asArray(edges).filter((edge) => edge && edge.source != null && edge.target != null),
  );
  const straight = links.filter((edge) => String(edge.source) !== String(edge.target));
  const selfLinks = links.filter((edge) => String(edge.source) === String(edge.target));
  const loopList = asArray(loops).filter((loop) => asArray(loop.members).length);

  const memberSet = new Set(loopList.flatMap((loop) => loop.members.map(String)));
  const singleRing =
    loopList.length === 1 && list.length <= 9 && list.every((node) => memberSet.has(String(node.id)));

  const boxHeight = 30;
  const sized = list.map((node) => {
    const label = String(node.label || node.id);
    return { ...node, width: Math.max(78, Math.min(156, 26 + label.length * 6.2)), height: boxHeight };
  });

  const canvasHeight = height || Math.max(320, Math.min(640, 200 + sized.length * 22));
  let placed;
  if (singleRing) {
    const order = loopList[0].members.map(String);
    const ordered = order
      .map((memberId) => sized.find((node) => String(node.id) === memberId))
      .filter(Boolean);
    const radius = Math.min(width, canvasHeight) * 0.33;
    const ring = circularLayout(
      ordered.map((node) => node.id),
      { cx: width / 2, cy: canvasHeight / 2, radius },
    );
    placed = {
      nodes: ring.map((point) => {
        const node = ordered.find((entry) => String(entry.id) === point.id);
        return { id: point.id, x: point.x, y: point.y, width: node.width, height: node.height };
      }),
      width,
      height: canvasHeight,
    };
  } else {
    placed = forceLayout(
      { nodes: sized, edges: straight },
      { width, height: canvasHeight, padding: 56, edgeStrength: 0.9 },
    );
  }
  const positions = new Map(
    placed.nodes.map((point) => {
      const node = sized.find((entry) => String(entry.id) === String(point.id));
      return [
        String(point.id),
        { x: point.x, y: point.y, width: node?.width || 90, height: node?.height || boxHeight },
      ];
    }),
  );

  const seed = `${sized.map((node) => node.id).join(",")}|${straight
    .map((edge) => `${edge.source}>${edge.target}:${edge.polarity}`)
    .join(",")}|${loopList.map((loop) => `${loop.id}:${loop.polarity}`).join(",")}`;

  const loopEdgeKeys = new Map();
  loopList.forEach((loop, index) => {
    const members = loop.members.map(String);
    for (let i = 0; i < members.length; i++) {
      const key = `${members[i]}>${members[(i + 1) % members.length]}`;
      if (!loopEdgeKeys.has(key)) loopEdgeKeys.set(key, []);
      loopEdgeKeys.get(key).push(index);
    }
  });

  const edgeMarkup = straight
    .map((edge) => {
      const from = positions.get(String(edge.source));
      const to = positions.get(String(edge.target));
      if (!from || !to) return "";
      const dx = to.x - from.x;
      const dy = to.y - from.y;
      const length = Math.hypot(dx, dy) || 1;
      const startTrim = Math.min(from.width / 2 + 4, length / 2 - 1);
      const endTrim = Math.min(to.width / 2 + 10, length / 2 - 1);
      const start = [from.x + (dx / length) * startTrim, from.y + (dy / length) * startTrim];
      const end = [to.x - (dx / length) * endTrim, to.y - (dy / length) * endTrim];
      const bow = edge.bundleOffset || (Math.abs(dy) < 2 ? 0 : 0);
      const curveAmount = bow || Math.min(26, length * 0.14);
      const nx = -(dy / length) * curveAmount;
      const ny = (dx / length) * curveAmount;
      const mid = [(start[0] + end[0]) / 2 + nx, (start[1] + end[1]) / 2 + ny];
      const path = curvePath([start, mid, end], { tension: 0.5 });
      if (!path) return "";
      const style = polarityStyle(edge.polarity);
      const loopIndices = loopEdgeKeys.get(`${edge.source}>${edge.target}`) || [];
      const inLoop = loopIndices.length > 0;
      const loopColour = inLoop ? loopStyle(loopList[loopIndices[0]].polarity).color : null;
      const signX = mid[0] + (nx / (curveAmount || 1)) * 9;
      const signY = mid[1] + (ny / (curveAmount || 1)) * 9;
      return el(
        "g",
        {
          class: `viz-link${inLoop ? " in-loop" : ""}`,
          "data-edge": edge.id ?? "",
          "data-source": edge.source,
          "data-target": edge.target,
          "data-detail": `${style.label} link. ${edge.detail || "No mechanism stated."}${
            inLoop ? ` Member of ${loopIndices.map((i) => loopList[i].id || `loop ${i + 1}`).join(", ")}.` : ""
          }`,
          "data-label": `${edge.source} to ${edge.target}`,
          tabindex: "0",
          role: "graphics-symbol",
          "aria-label": `Link from ${edge.source} to ${edge.target}. ${style.label}.`,
        },
        [
          el("path", { d: path, fill: "none", stroke: "transparent", "stroke-width": 12, class: "viz-link-hit" }),
          el("path", {
            d: path,
            fill: "none",
            stroke: inLoop ? loopColour : style.color,
            "stroke-width": inLoop ? 2.2 : 1.4,
            "stroke-dasharray": style.dash || null,
            "marker-end": `url(#${uid("viz-cld-arrow", inLoop ? loopColour : style.color)})`,
            class: "viz-link-line",
          }),
          el("circle", { cx: signX, cy: signY, r: 7, fill: PAPER, stroke: style.color, "stroke-width": 1 }),
          text(signX, signY + 3.6, style.symbol, {
            "text-anchor": "middle",
            class: "viz-link-sign",
            fill: style.color,
          }),
          edge.delay
            ? el("path", {
                d: `M${num(mid[0] - 5)} ${num(mid[1] - 7)}L${num(mid[0] + 5)} ${num(mid[1] - 7)}M${num(
                  mid[0] - 5,
                )} ${num(mid[1] - 3)}L${num(mid[0] + 5)} ${num(mid[1] - 3)}`,
                stroke: MUTED,
                "stroke-width": 1.4,
              })
            : "",
          el(
            "title",
            {},
            escapeText(
              `${edge.id ? edge.id + ": " : ""}${style.label}. ${edge.detail || "No mechanism stated."}`,
            ),
          ),
        ],
      );
    })
    .join("");

  const selfMarkup = selfLinks
    .map((edge, index) => {
      const at = positions.get(String(edge.source));
      if (!at) return "";
      const style = polarityStyle(edge.polarity);
      const path = selfLoopPath(at, { index: edge.bundleIndex || 0, size: 24 });
      if (!path) return "";
      return el(
        "g",
        {
          class: "viz-link viz-link-self",
          "data-edge": edge.id ?? `self-${index}`,
          "data-detail": `Self reinforcing or damping link. ${edge.detail || "No mechanism stated."}`,
          "data-label": `${edge.source} acts on itself`,
          tabindex: "0",
          role: "graphics-symbol",
          "aria-label": `Self link on ${edge.source}. ${style.label}.`,
        },
        [
          el("path", {
            d: path,
            fill: "none",
            stroke: style.color,
            "stroke-width": 1.6,
            "stroke-dasharray": style.dash || null,
            "marker-end": `url(#${uid("viz-cld-arrow", style.color)})`,
          }),
          el("title", {}, escapeText(`${edge.id || "Self link"}: ${style.label}.`)),
        ],
      );
    })
    .join("");

  const badges = loopList
    .map((loop, index) => {
      const points = loop.members
        .map((memberId) => positions.get(String(memberId)))
        .filter(Boolean);
      if (!points.length) return "";
      const middle = centroid(points);
      if (!middle) return "";
      return loopBadge(loop, middle, windingSign(points), index);
    })
    .join("");

  const nodeMarkup = sized
    .map((node) => {
      const at = positions.get(String(node.id));
      if (!at) return "";
      const label = wrapLabel(String(node.label || node.id), Math.floor(at.width / 6.1), 1)[0] || "";
      return el(
        "g",
        {
          class: "viz-node viz-cld-node",
          "data-node": node.id,
          "data-detail": node.detail || "",
          "data-label": node.label || node.id,
          tabindex: "0",
          role: "button",
          "aria-label": `${node.label || node.id}. ${node.detail || ""}`,
        },
        [
          el("rect", {
            x: at.x - at.width / 2,
            y: at.y - at.height / 2,
            width: at.width,
            height: at.height,
            rx: at.height / 2,
            fill: PAPER,
            stroke: LINE,
            class: "viz-node-box",
          }),
          text(at.x, at.y + 4, label, { "text-anchor": "middle", class: "viz-node-title", fill: INK }),
          el("title", {}, escapeText(`${node.label || node.id}. ${node.detail || ""}`)),
        ],
      );
    })
    .join("");

  const usedPolarities = [];
  for (const edge of straight.concat(selfLinks)) {
    const style = polarityStyle(edge.polarity);
    if (usedPolarities.some((entry) => entry.symbol === style.symbol)) continue;
    usedPolarities.push(style);
  }
  const usedLoops = [];
  for (const loop of loopList) {
    const style = loopStyle(loop.polarity);
    if (usedLoops.some((entry) => entry.badge === style.badge)) continue;
    usedLoops.push(style);
  }

  const markerColours = new Set([
    POLARITY.positive.color,
    POLARITY.negative.color,
    POLARITY.unknown.color,
    ...usedLoops.map((entry) => entry.color),
  ]);

  const unknownCount = straight
    .concat(selfLinks)
    .filter((edge) => polarityStyle(edge.polarity) === POLARITY.unknown).length;

  const svg = svgFrame({
    width: placed.width || width,
    height: placed.height || canvasHeight,
    role: "group",
    label: title || "Causal loop diagram",
    desc: `${sized.length} variables, ${straight.length + selfLinks.length} signed links and ${
      loopList.length
    } declared feedback loop${loopList.length === 1 ? "" : "s"}.`,
    seed,
    className: "viz-cld-svg",
    defs:
      Array.from(markerColours)
        .map((colour) => arrowMarker(uid("viz-cld-arrow", colour), colour, { width: 7, refX: 6.6 }))
        .join("") +
      Array.from(new Set(usedLoops.map((entry) => entry.color)))
        .map((colour) => arrowMarker(uid("viz-loop-arrow", colour), colour, { width: 6, refX: 5.6 }))
        .join(""),
    children:
      el("g", { class: "viz-layer viz-layer-links" }, edgeMarkup + selfMarkup) +
      el("g", { class: "viz-layer viz-layer-components" }, nodeMarkup) +
      el("g", { class: "viz-layer viz-layer-loops" }, badges),
  });

  const controls = `<div class="viz-layer-toggles" role="group" aria-label="Diagram layers">${[
    ["links", "Links"],
    ["loops", "Loop badges"],
  ]
    .map(
      ([key, label]) =>
        `<button type="button" class="viz-toggle" data-viz-layer="${key}" aria-pressed="true">${escapeText(
          label,
        )}</button>`,
    )
    .join("")}</div>`;

  return figure({
    kind: "causal-loop",
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
      legend({
        title: "LINK POLARITY",
        items: usedPolarities.map((entry) => ({
          label: `${entry.symbol}  ${entry.label}`,
          color: entry.color,
          dash: entry.dash,
          shape: "circle",
        })),
      }) +
      legend({
        title: "LOOP CHARACTER",
        items: usedLoops.map((entry) => ({
          label: `${entry.badge}  ${entry.label}`,
          color: entry.color,
          block: true,
        })),
      }),
    caption:
      caption ||
      `A "+" means the two variables move in the same direction and a "-" that they move in opposite directions, as declared. R marks a reinforcing loop and B a balancing one.${
        unknownCount
          ? ` ${unknownCount} link${unknownCount === 1 ? " has" : "s have"} no declared polarity and ${
              unknownCount === 1 ? "is" : "are"
            } shown as "?" rather than assumed.`
          : ""
      } Loop polarity is a declared property of the whole cycle and is never used to infer the sign of an individual link. Nothing here has been estimated from data.`,
    scroll: true,
    minWidth: Math.min(1200, Math.max(320, placed.width || width)),
  });
}

/** The causal loop diagram for a ComplexSystemSpec. */
export function systemCausalLoop(spec, options = {}) {
  const model = causalLoopModel(spec);
  if (!model.nodes.length) {
    return emptyPanel({
      kind: "causal-loop",
      eyebrow: "CAUSAL LOOP DIAGRAM",
      title: "No causal structure is declared.",
      message:
        "Components and couplings are needed before feedback can be drawn. This specification declares none.",
      caption: "An absent diagram is an absent declaration, not evidence that the system lacks feedback.",
    });
  }
  return causalLoopDiagram({
    ...model,
    id: options.id || "viz-causal-loop",
    title: options.title || "Declared feedback structure",
    note: model.loops.length
      ? ""
      : "No feedback loop is declared in this specification, so no R or B badge is shown.",
    width: options.width || 780,
    height: options.height || 0,
  });
}
