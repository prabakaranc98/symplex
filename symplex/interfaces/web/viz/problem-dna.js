"use strict";
/**
 * Problem DNA as a diagram rather than a form dump.
 *
 * ProblemDNA is a decision frame: who decides, what they decide, what they are
 * trying to achieve, and how anyone would know it worked. Rendered as a list of
 * fields that structure is invisible; rendered as a spine with the boundary
 * drawn around it, the constraints sitting on the path, and the feedback
 * returning to the decision, it can be read at a glance.
 *
 * Four pictures:
 *   problemDnaDiagram   the decision spine inside its scope boundary
 *   uncertaintyStrip    assumptions, missing information and failure cost
 *   revisionDeltaChart  what changed in the inputs between two revisions
 *   lineageTimeline     successive framings and the input revision each used
 *
 * Absent fields are drawn as explicitly absent - a hatched slot with the words
 * "not stated" - because an empty column in a decision frame is a finding.
 */

import {
  el,
  text,
  escapeText,
  wrapLabel,
  ellipsis,
  textWidth,
  figure,
  emptyPanel,
  svgFrame,
  arrowMarker,
  absentPattern,
  uid,
  num,
  coord,
} from "./svg.js";
import { INK, MUTED, LINE, PAPER, PALE, GRID, ACCENT, WARN, LOOP } from "./color.js";
import { legend } from "./legend.js";
import { curvePath } from "./layout.js";

const asArray = (value) => (Array.isArray(value) ? value : []);
const isNum = (value) => typeof value === "number" && Number.isFinite(value);

const NOT_STATED = "not stated in this framing";

/** Lay out short labels as wrapped chips. Returns markup plus the height used. */
function chipRow(items, { x, y, maxWidth, fill, stroke, ink, hatch = null, strike = false, max = 12 }) {
  const shown = items.slice(0, max);
  const parts = [];
  let cursorX = x;
  let cursorY = y;
  let rowHeight = 22;
  for (const item of shown) {
    const label = ellipsis(String(item), 46);
    const width = Math.max(46, textWidth(label, 10) + 18);
    if (cursorX + width > x + maxWidth && cursorX > x) {
      cursorX = x;
      cursorY += rowHeight + 6;
    }
    parts.push(
      el("rect", {
        x: cursorX,
        y: cursorY,
        width,
        height: rowHeight,
        rx: 5,
        fill: hatch ? `url(#${hatch})` : fill,
        stroke,
        "stroke-width": 1,
      }),
    );
    parts.push(
      text(cursorX + 9, cursorY + 14.5, label, {
        class: "viz-chip-label",
        fill: ink,
        "text-decoration": strike ? "line-through" : null,
      }),
    );
    parts.push(el("title", {}, escapeText(String(item))));
    cursorX += width + 6;
  }
  if (items.length > shown.length) {
    const label = `+${items.length - shown.length} more`;
    const width = Math.max(46, textWidth(label, 10) + 18);
    if (cursorX + width > x + maxWidth && cursorX > x) {
      cursorX = x;
      cursorY += rowHeight + 6;
    }
    parts.push(
      text(cursorX + 2, cursorY + 14.5, label, { class: "viz-chip-label", fill: MUTED }),
    );
  }
  if (!shown.length) {
    parts.push(text(x, y + 14.5, NOT_STATED, { class: "viz-absent", fill: MUTED }));
  }
  return { markup: parts.join(""), height: cursorY + rowHeight - y };
}

function spineBox({ x, y, width, height, eyebrow, value, accent, id, detail }) {
  const stated = value !== null && value !== undefined && String(value).trim() !== "";
  const lines = stated ? wrapLabel(String(value), Math.floor(width / 5.9), 4) : [];
  return el(
    "g",
    {
      class: "viz-node viz-dna-box",
      "data-node": id,
      "data-label": eyebrow,
      "data-detail": detail || (stated ? String(value) : `This framing does not state the ${eyebrow.toLowerCase()}.`),
      tabindex: "0",
      role: "button",
      "aria-label": `${eyebrow}: ${stated ? String(value) : NOT_STATED}`,
    },
    [
      el("rect", {
        x,
        y,
        width,
        height,
        rx: 8,
        fill: stated ? PAPER : "#fbfcfe",
        stroke: stated ? LINE : "#c9d4e6",
        "stroke-width": 1,
        "stroke-dasharray": stated ? null : "4 3",
        class: "viz-node-box",
      }),
      el("rect", { x, y, width, height: 3, rx: 1.5, fill: accent }),
      text(x + 12, y + 20, eyebrow, { class: "viz-dna-eyebrow", fill: accent }),
      ...(stated
        ? lines.map((line, i) => text(x + 12, y + 38 + i * 13, line, { class: "viz-dna-value", fill: INK }))
        : [text(x + 12, y + 38, NOT_STATED, { class: "viz-absent", fill: MUTED })]),
      el("title", {}, escapeText(`${eyebrow}: ${stated ? String(value) : NOT_STATED}`)),
    ],
  );
}

/**
 * The decision spine.
 *
 * dna:       {beneficiary, decision, objective, actors, constraints, assumptions,
 *             missing_evidence}
 * endpoints: [{name, criterion, direction}]  from the decision frame, optional
 * boundary:  {included: [text], excluded: [text]}                    optional
 * feedback:  [{id, description, polarity}]                           optional
 */
export function problemDnaDiagram({
  dna,
  endpoints = [],
  boundary = null,
  feedback = [],
  id = "viz-problem-dna",
  title = "",
  eyebrow = "PROBLEM DNA",
  note = "",
  caption = "",
  width = 900,
} = {}) {
  if (!dna || typeof dna !== "object") {
    return emptyPanel({
      kind: "problem-dna",
      eyebrow,
      title: "No decision frame yet.",
      message:
        "The agent has not proposed a beneficiary, decision, objective or evidence gap for this problem.",
      caption: "An absent frame is not a simple problem; it is an unframed one.",
    });
  }

  const padding = 22;
  const boxWidth = 178;
  const boxGap = 44;
  const boxHeight = 108;
  const spineY = 74;
  const spineWidth = boxWidth * 4 + boxGap * 3;
  const canvasWidth = Math.max(width, spineWidth + padding * 2 + 24);
  const spineX = padding + 12 + Math.max(0, (canvasWidth - padding * 2 - 24 - spineWidth) / 2);

  const constraints = asArray(dna.constraints).map(String);
  const actors = asArray(dna.actors).map(String);
  const included = asArray(boundary?.included).map(String);
  const excluded = asArray(boundary?.excluded).map(String);
  const endpointText = asArray(endpoints).length
    ? asArray(endpoints)
        .map((endpoint) =>
          typeof endpoint === "string"
            ? endpoint
            : `${endpoint.name || endpoint.id || "endpoint"}${endpoint.direction ? ` (${endpoint.direction})` : ""}`,
        )
        .join("; ")
    : "";

  const boxes = [
    {
      id: "beneficiary",
      eyebrow: "WHO DECIDES",
      value: dna.beneficiary,
      accent: ACCENT,
      detail: dna.beneficiary ? `The result is for ${dna.beneficiary}.` : "",
    },
    {
      id: "decision",
      eyebrow: "THE DECISION",
      value: dna.decision,
      accent: ACCENT,
      detail: dna.decision ? `What is actually being chosen: ${dna.decision}` : "",
    },
    {
      id: "objective",
      eyebrow: "OBJECTIVE",
      value: dna.objective,
      accent: "#1a7f72",
      detail: dna.objective ? `What the decision is trying to achieve: ${dna.objective}` : "",
    },
    {
      id: "endpoints",
      eyebrow: "HOW WE WOULD KNOW",
      value: endpointText,
      accent: "#1a7f72",
      detail: endpointText
        ? `Measurable endpoints from the current system specification: ${endpointText}`
        : "No measurable endpoint has been declared, so success cannot yet be observed.",
    },
  ];

  const arrowId = uid("viz-dna-arrow", "spine");
  const hatchId = uid("viz-dna-hatch", id);
  const feedbackColour = feedback.length ? LOOP[feedback[0].polarity]?.color || MUTED : MUTED;
  const feedbackArrow = uid("viz-dna-feedback", feedbackColour);

  const spineMarkup = boxes
    .map((box, index) =>
      spineBox({
        ...box,
        x: spineX + index * (boxWidth + boxGap),
        y: spineY,
        width: boxWidth,
        height: boxHeight,
      }),
    )
    .join("");

  const connectors = [0, 1, 2]
    .map((index) => {
      const from = spineX + index * (boxWidth + boxGap) + boxWidth;
      const to = from + boxGap;
      const y = spineY + boxHeight / 2;
      return el("line", {
        x1: from + 4,
        y1: y,
        x2: to - 9,
        y2: y,
        stroke: MUTED,
        "stroke-width": 1.4,
        "marker-end": `url(#${arrowId})`,
      });
    })
    .join("");

  // Constraints sit on the path from objective to endpoints: they gate it.
  const gateX = spineX + 2 * (boxWidth + boxGap) + boxWidth + boxGap / 2;
  const gateY = spineY + boxHeight / 2;
  const gateMarkup = constraints.length
    ? el(
        "g",
        {
          class: "viz-dna-gate",
          "data-node": "constraints",
          "data-label": "Constraints",
          "data-detail": `${constraints.length} constraint${
            constraints.length === 1 ? "" : "s"
          } stand between the objective and an acceptable outcome: ${constraints.join("; ")}`,
          tabindex: "0",
          role: "graphics-symbol",
          "aria-label": `${constraints.length} constraints on the path from objective to endpoint.`,
        },
        el("line", {
          x1: gateX,
          y1: gateY - 15,
          x2: gateX,
          y2: gateY + 15,
          stroke: WARN,
          "stroke-width": 3,
        }) +
          text(gateX, gateY - 21, `${constraints.length} constraint${constraints.length === 1 ? "" : "s"}`, {
            "text-anchor": "middle",
            class: "viz-dna-eyebrow",
            fill: WARN,
          }) +
          el("title", {}, escapeText(constraints.join("; "))),
      )
    : "";

  // Actors hang below the decision maker and feed the decision.
  const actorY = spineY + boxHeight + 26;
  const actorBlock = chipRow(actors, {
    x: spineX,
    y: actorY,
    maxWidth: boxWidth * 2 + boxGap,
    fill: PALE,
    stroke: "#d3e0f8",
    ink: INK,
  });
  const actorLabel = text(spineX, actorY - 8, "ACTORS WHO CAN CHANGE THE OUTCOME", {
    class: "viz-dna-eyebrow",
    fill: MUTED,
  });
  const actorLink = actors.length
    ? el("path", {
        d: `M${num(spineX + 24)} ${num(actorY)}L${num(spineX + 24)} ${num(spineY + boxHeight)}`,
        stroke: LINE,
        "stroke-width": 1,
        "stroke-dasharray": "3 3",
        fill: "none",
      })
    : "";

  // Feedback returns from the endpoints to the decision.
  const feedbackY = actorY + actorBlock.height + 34;
  const feedbackMarkup = feedback.length
    ? el(
        "g",
        {
          class: "viz-dna-feedback",
          "data-node": "feedback",
          "data-label": "Feedback",
          "data-detail": feedback
            .map(
              (loop) =>
                `${loop.id ? loop.id + ": " : ""}${loop.description || "no description"} (${
                  LOOP[loop.polarity]?.label || "polarity not stated"
                })`,
            )
            .join(" | "),
          tabindex: "0",
          role: "graphics-symbol",
          "aria-label": `${feedback.length} declared feedback loop${feedback.length === 1 ? "" : "s"} returning to the decision.`,
        },
        el("path", {
          d: curvePath(
            [
              [spineX + 3 * (boxWidth + boxGap) + boxWidth / 2, spineY + boxHeight],
              [spineX + 2 * (boxWidth + boxGap), feedbackY],
              [spineX + boxWidth + boxGap + boxWidth / 2, spineY + boxHeight],
            ],
            { tension: 0.5 },
          ),
          fill: "none",
          stroke: feedbackColour,
          "stroke-width": 1.6,
          "stroke-dasharray": "7 4",
          "marker-end": `url(#${feedbackArrow})`,
        }) +
          text(
            spineX + 2 * (boxWidth + boxGap),
            feedbackY + 14,
            `${feedback.length} declared feedback loop${feedback.length === 1 ? "" : "s"} returns to the decision`,
            { "text-anchor": "middle", class: "viz-dna-eyebrow", fill: feedbackColour },
          ) +
          el("title", {}, escapeText(feedback.map((loop) => loop.description || loop.id).join("; "))),
      )
    : "";

  // The scope boundary encloses everything above; what is out of scope sits
  // outside the frame on hatched ground.
  const frameBottom = feedbackY + (feedback.length ? 26 : 0) + 12;
  const frame = el("g", { class: "viz-dna-frame", "aria-hidden": "true" }, [
    el("rect", {
      x: padding,
      y: 44,
      width: canvasWidth - padding * 2,
      height: frameBottom - 44,
      rx: 12,
      fill: "none",
      stroke: ACCENT,
      "stroke-width": 1.2,
      "stroke-dasharray": "1 0",
      "stroke-opacity": 0.4,
    }),
    text(padding + 12, 36, "INSIDE THE MODEL BOUNDARY", { class: "viz-dna-eyebrow", fill: ACCENT }),
  ]);

  const insideY = frameBottom + 22;
  const insideBlock = chipRow(included, {
    x: padding + 12,
    y: insideY,
    maxWidth: canvasWidth - padding * 2 - 24,
    fill: "#f4f8ff",
    stroke: "#d3e0f8",
    ink: INK,
  });
  const outsideY = insideY + insideBlock.height + 34;
  const outsideBlock = chipRow(excluded, {
    x: padding + 12,
    y: outsideY,
    maxWidth: canvasWidth - padding * 2 - 24,
    fill: "#fbfbfc",
    stroke: "#d6d9e0",
    ink: MUTED,
    hatch: hatchId,
    strike: true,
  });

  const scopeMarkup =
    (included.length || excluded.length
      ? el("g", { class: "viz-dna-scope" }, [
          text(padding + 12, insideY - 8, "REPRESENTED IN THIS FRAME", { class: "viz-dna-eyebrow", fill: ACCENT }),
          insideBlock.markup,
          text(padding + 12, outsideY - 8, "DELIBERATELY OUT OF SCOPE", { class: "viz-dna-eyebrow", fill: MUTED }),
          outsideBlock.markup,
        ])
      : "");

  const height =
    (included.length || excluded.length ? outsideY + outsideBlock.height + 18 : frameBottom + 18);

  const svg = svgFrame({
    width: canvasWidth,
    height,
    role: "group",
    label: title || "Problem DNA decision structure",
    desc: `Decision maker, decision, objective and endpoints on one path, with ${constraints.length} constraint${
      constraints.length === 1 ? "" : "s"
    }, ${actors.length} actor${actors.length === 1 ? "" : "s"}, ${feedback.length} feedback loop${
      feedback.length === 1 ? "" : "s"
    }, and the scope boundary drawn around the frame.`,
    seed: `${id}|${dna.decision || ""}|${constraints.length}|${actors.length}`,
    className: "viz-dna-svg",
    defs:
      arrowMarker(arrowId, MUTED, { width: 7, refX: 6.6 }) +
      arrowMarker(feedbackArrow, feedbackColour, { width: 7, refX: 6.6 }) +
      absentPattern(hatchId),
    children:
      frame +
      el("g", { class: "viz-layer viz-layer-spine" }, connectors + spineMarkup + gateMarkup) +
      el("g", { class: "viz-layer viz-layer-actors" }, actorLabel + actorLink + actorBlock.markup) +
      el("g", { class: "viz-layer viz-layer-feedback" }, feedbackMarkup) +
      el("g", { class: "viz-layer viz-layer-scope" }, scopeMarkup),
  });

  return figure({
    kind: "problem-dna",
    id,
    eyebrow,
    title,
    note,
    svg:
      svg +
      '<p class="viz-readout" role="status" data-viz-readout data-default="Hover or focus any part of the frame to read it in full.">Hover or focus any part of the frame to read it in full.</p>' +
      legend({
        title: "READING THE FRAME",
        items: [
          { label: "Stated in this framing", color: ACCENT, block: true },
          { label: "Dashed outline - not stated", color: "#c9d4e6", block: true },
          { label: "Constraint gate on the path", color: WARN, shape: "plus", dash: "" },
          { label: "Hatched and struck through - out of scope", color: MUTED, block: true },
        ],
      }),
    caption:
      caption ||
      "This is the agent's proposed framing of the decision, not a validated account of it. Endpoints come from the current system specification and describe how success would be observed, not that it has been. Anything drawn as absent was absent from the framing, which is itself a gap worth closing.",
    scroll: true,
    minWidth: Math.min(1300, canvasWidth),
  });
}

/**
 * Assumptions, missing information and the cost of being wrong, side by side.
 * `failureCost` is optional: ProblemDNA does not carry one, so when the caller
 * has nothing to pass the column is drawn hatched and labelled as unstated.
 */
export function uncertaintyStrip({
  assumptions = [],
  missing = [],
  failureCost = null,
  extrapolationLimits = [],
  id = "viz-problem-uncertainty",
  eyebrow = "WHAT COULD BE WRONG",
  title = "",
  width = 900,
  caption = "",
} = {}) {
  const columns = [
    {
      key: "assumptions",
      label: "ASSUMPTIONS TO CHALLENGE",
      items: asArray(assumptions).map(String),
      accent: WARN,
      empty: "No assumption was recorded. That is unlikely to mean there are none.",
    },
    {
      key: "missing",
      label: "MISSING INFORMATION",
      items: asArray(missing).map(String),
      accent: "#a1447e",
      empty: "No evidence gap was recorded for this framing.",
    },
    {
      key: "cost",
      label: "COST OF BEING WRONG",
      items: failureCost ? [String(failureCost)] : [],
      accent: "#8a3a1c",
      empty: "The cost of a wrong decision is not stated anywhere in this framing.",
      wide: true,
    },
  ];
  if (asArray(extrapolationLimits).length) {
    columns.push({
      key: "limits",
      label: "WHERE THE INFERENCE MAY FAIL",
      items: asArray(extrapolationLimits).map(String),
      accent: MUTED,
      empty: "No validity limit was recorded.",
    });
  }

  const padding = 18;
  const gap = 14;
  const columnWidth = Math.max(
    160,
    Math.floor((width - padding * 2 - gap * (columns.length - 1)) / columns.length),
  );
  const canvasWidth = padding * 2 + columnWidth * columns.length + gap * (columns.length - 1);
  const hatchId = uid("viz-uncertainty-hatch", id);

  const blocks = columns.map((column, index) => {
    const x = padding + index * (columnWidth + gap);
    const lines = column.items.length
      ? column.items.flatMap((item, itemIndex) =>
          wrapLabel(item, Math.floor(columnWidth / 5.6), column.wide ? 6 : 3).map((line, lineIndex) => ({
            line,
            first: lineIndex === 0,
            itemIndex,
          })),
        )
      : wrapLabel(column.empty, Math.floor(columnWidth / 5.6), 4).map((line) => ({ line, absent: true }));
    return { column, x, lines };
  });
  const maxLines = Math.max(...blocks.map((block) => block.lines.length), 1);
  const bodyTop = 44;
  const height = bodyTop + maxLines * 14 + 26;

  const markup = blocks
    .map(({ column, x, lines }) => {
      const stated = column.items.length > 0;
      return el(
        "g",
        {
          class: "viz-uncertainty-column",
          "data-node": column.key,
          "data-label": column.label,
          "data-detail": stated ? column.items.join(" | ") : column.empty,
          tabindex: "0",
          role: "graphics-symbol",
          "aria-label": `${column.label}: ${stated ? column.items.join("; ") : column.empty}`,
        },
        [
          el("rect", {
            x,
            y: 20,
            width: columnWidth,
            height: height - 34,
            rx: 8,
            fill: stated ? PAPER : `url(#${hatchId})`,
            stroke: stated ? LINE : "#c9d4e6",
            "stroke-width": 1,
            "stroke-dasharray": stated ? null : "4 3",
          }),
          el("rect", { x, y: 20, width: columnWidth, height: 3, rx: 1.5, fill: column.accent }),
          text(x + 11, 38, column.label, { class: "viz-dna-eyebrow", fill: column.accent }),
          ...lines.map((entry, i) =>
            text(x + 11 + (entry.absent ? 0 : entry.first ? 0 : 8), bodyTop + 8 + i * 14, entry.absent ? entry.line : `${entry.first ? "· " : "  "}${entry.line}`, {
              class: entry.absent ? "viz-absent" : "viz-dna-value",
              fill: entry.absent ? MUTED : INK,
            }),
          ),
          el("title", {}, escapeText(`${column.label}: ${stated ? column.items.join("; ") : column.empty}`)),
        ],
      );
    })
    .join("");

  const svg = svgFrame({
    width: canvasWidth,
    height,
    role: "group",
    label: title || "Assumptions, gaps and failure cost",
    desc: columns
      .map((column) => `${column.label}: ${column.items.length ? `${column.items.length} recorded` : "none recorded"}`)
      .join(". "),
    seed: `${id}|${columns.map((column) => column.items.length).join(",")}`,
    className: "viz-uncertainty-svg",
    defs: absentPattern(hatchId),
    children: markup,
  });

  return figure({
    kind: "problem-uncertainty",
    id,
    eyebrow,
    title,
    svg:
      svg +
      '<p class="viz-readout" role="status" data-viz-readout data-default="Hover or focus a column to read every entry.">Hover or focus a column to read every entry.</p>',
    caption:
      caption ||
      "A hatched column means the framing does not address it at all. An empty assumption list is not evidence that the framing is safe, and the cost of being wrong is the field most often left blank in exactly the decisions where it matters most.",
    scroll: true,
    minWidth: Math.min(1200, canvasWidth),
  });
}

/**
 * Input revision delta: what the agent had last time against what it has now.
 *
 * previous/current are input_revision payloads: {revision, inputs:[{id, kind, digest}]}.
 * The delta is computed here from the two inventories, the same way the host
 * computes it, so no extra endpoint is needed to draw it.
 */
export function revisionDeltaChart({
  previous = null,
  current = null,
  labels = {},
  id = "viz-input-delta",
  eyebrow = "INPUT REVISION",
  title = "",
  width = 860,
  caption = "",
} = {}) {
  if (!current || !asArray(current.inputs).length) {
    return emptyPanel({
      kind: "revision-delta",
      eyebrow,
      title: "No input inventory recorded.",
      message:
        "Input revisions are recorded when the agent takes a planning decision. None has been recorded for this problem yet.",
      caption: "No inventory means no decision has been taken against these inputs, not that there are no inputs.",
    });
  }
  const before = new Map(asArray(previous?.inputs).map((entry) => [String(entry.id), entry]));
  const after = new Map(asArray(current.inputs).map((entry) => [String(entry.id), entry]));
  const rows = [];
  for (const [key, entry] of after) {
    const old = before.get(key);
    rows.push({
      id: key,
      kind: entry.kind,
      status: !old ? "added" : old.digest !== entry.digest ? "changed" : "unchanged",
    });
  }
  for (const [key, entry] of before) {
    if (after.has(key)) continue;
    rows.push({ id: key, kind: entry.kind, status: "removed" });
  }
  const rank = { added: 0, changed: 1, removed: 2, unchanged: 3 };
  rows.sort((a, b) => rank[a.status] - rank[b.status] || a.id.localeCompare(b.id));

  const STATUS = {
    added: { label: "New since the last framing", color: ACCENT, shape: "plus" },
    changed: { label: "Changed since the last framing", color: WARN, shape: "diamond" },
    removed: { label: "Withdrawn since the last framing", color: "#8a3a1c", shape: "cross" },
    unchanged: { label: "Unchanged", color: MUTED, shape: "circle" },
  };

  const padding = 18;
  const rowHeight = 22;
  const height = padding * 2 + 26 + rows.length * rowHeight;
  const canvasWidth = Math.max(360, width);
  const markup = rows
    .map((row, index) => {
      const y = padding + 26 + index * rowHeight;
      const style = STATUS[row.status];
      const label = labels[row.id] || row.id;
      return el(
        "g",
        {
          class: "viz-delta-row",
          "data-node": row.id,
          "data-label": label,
          "data-detail": `${style.label}. Kind: ${row.kind || "unknown"}. Artifact ${row.id}.`,
          tabindex: "0",
          role: "graphics-symbol",
          "aria-label": `${label}. ${style.label}.`,
        },
        [
          el("rect", {
            x: padding,
            y: y - 1,
            width: canvasWidth - padding * 2,
            height: rowHeight - 4,
            rx: 4,
            fill: row.status === "unchanged" ? "transparent" : "#f7f9fd",
          }),
          el("rect", { x: padding, y: y - 1, width: 3, height: rowHeight - 4, rx: 1.5, fill: style.color }),
          text(padding + 14, y + 12, ellipsis(String(label), 58), {
            class: "viz-dna-value",
            fill: row.status === "removed" ? MUTED : INK,
            "text-decoration": row.status === "removed" ? "line-through" : null,
          }),
          text(canvasWidth - padding - 6, y + 12, style.label, {
            "text-anchor": "end",
            class: "viz-chip-label",
            fill: style.color,
          }),
          el("title", {}, escapeText(`${label}: ${style.label}`)),
        ],
      );
    })
    .join("");

  const counts = rows.reduce((acc, row) => {
    acc[row.status] = (acc[row.status] || 0) + 1;
    return acc;
  }, {});
  const changed = (counts.added || 0) + (counts.changed || 0) + (counts.removed || 0);

  const svg = svgFrame({
    width: canvasWidth,
    height,
    role: "group",
    label: title || "Input revision delta",
    desc: `${counts.added || 0} new, ${counts.changed || 0} changed, ${counts.removed || 0} withdrawn and ${
      counts.unchanged || 0
    } unchanged inputs between revision ${previous?.revision ? previous.revision.slice(0, 8) : "none"} and ${
      current.revision ? current.revision.slice(0, 8) : "current"
    }.`,
    seed: `${id}|${current.revision}|${previous?.revision || ""}`,
    className: "viz-delta-svg",
    children:
      text(padding, padding + 12, `${changed} of ${rows.length} inputs differ from the previous framing`, {
        class: "viz-dna-eyebrow",
        fill: INK,
      }) + markup,
  });

  return figure({
    kind: "revision-delta",
    id,
    eyebrow,
    title,
    svg:
      svg +
      '<p class="viz-readout" role="status" data-viz-readout data-default="Hover or focus a row to see its artifact id and kind.">Hover or focus a row to see its artifact id and kind.</p>' +
      legend({
        items: Object.values(STATUS).map((style) => ({
          label: style.label,
          color: style.color,
          shape: style.shape,
          dash: "",
        })),
      }),
    caption:
      caption ||
      "Computed by comparing the recorded input inventories of two revisions. A changed digest means the artifact's content differs, not that the agent read it, agreed with it, or revalidated anything that depended on the old version.",
    scroll: true,
    minWidth: Math.min(1100, canvasWidth),
  });
}

/**
 * Reframing lineage: successive ProblemDNA versions and the input revision each
 * was based on. A version whose basis predates the current inputs is marked,
 * because that is exactly when a framing has quietly gone stale.
 */
export function lineageTimeline({
  versions = [],
  currentRevision = null,
  id = "viz-dna-lineage",
  eyebrow = "REFRAMING LINEAGE",
  title = "",
  width = 880,
  caption = "",
} = {}) {
  const list = asArray(versions);
  if (!list.length) {
    return emptyPanel({
      kind: "lineage",
      eyebrow,
      title: "No framing history.",
      message: "Only one framing exists, or none has been recorded yet.",
      caption: "A single framing is not a settled one; it is an unchallenged one.",
    });
  }
  const padding = 24;
  const spacing = Math.max(140, Math.min(240, (width - padding * 2) / Math.max(1, list.length)));
  const canvasWidth = padding * 2 + spacing * Math.max(1, list.length - 1) + 180;
  const axisY = 96;
  const height = 190;
  const arrowId = uid("viz-lineage-timeline-arrow", MUTED);

  const markers = list
    .map((version, index) => {
      const x = padding + 60 + index * spacing;
      const stale = Boolean(version.stale);
      const basis = version.inputRevision || null;
      const predates = Boolean(currentRevision && basis && basis !== currentRevision);
      const colour = stale ? MUTED : predates ? WARN : ACCENT;
      const lines = wrapLabel(String(version.label || version.decision || version.id || `Version ${index + 1}`), 24, 3);
      return el(
        "g",
        {
          class: "viz-lineage-node",
          "data-node": version.id || `version-${index}`,
          "data-label": `Framing ${index + 1}`,
          "data-detail": [
            version.label || version.decision || "",
            stale ? "Superseded by a later framing." : "Current framing.",
            basis ? `Based on input revision ${String(basis).slice(0, 12)}.` : "Input basis not recorded.",
            predates
              ? "This framing predates the current inputs: later evidence and steering have not been reconciled into it."
              : "",
          ]
            .filter(Boolean)
            .join(" "),
          tabindex: "0",
          role: "graphics-symbol",
          "aria-label": `Framing ${index + 1}. ${stale ? "Superseded." : "Current."} ${
            predates ? "Predates the current inputs." : ""
          }`,
        },
        [
          el("circle", {
            cx: x,
            cy: axisY,
            r: 8,
            fill: stale ? PAPER : colour,
            stroke: colour,
            "stroke-width": 2,
          }),
          predates
            ? el("circle", { cx: x, cy: axisY, r: 13, fill: "none", stroke: WARN, "stroke-width": 1, "stroke-dasharray": "2 2" })
            : "",
          text(x, axisY - 24, `FRAMING ${index + 1}`, {
            "text-anchor": "middle",
            class: "viz-dna-eyebrow",
            fill: colour,
          }),
          ...lines.map((line, i) =>
            text(x, axisY + 26 + i * 12, line, { "text-anchor": "middle", class: "viz-chip-label", fill: INK }),
          ),
          basis
            ? text(x, axisY + 26 + lines.length * 12 + 4, `inputs ${String(basis).slice(0, 8)}`, {
                "text-anchor": "middle",
                class: "viz-absent",
                fill: MUTED,
              })
            : text(x, axisY + 26 + lines.length * 12 + 4, "input basis not recorded", {
                "text-anchor": "middle",
                class: "viz-absent",
                fill: MUTED,
              }),
          el("title", {}, escapeText(String(version.label || version.decision || version.id || ""))),
        ],
      );
    })
    .join("");

  const spine = el("line", {
    x1: padding + 40,
    y1: axisY,
    x2: padding + 60 + (list.length - 1) * spacing + 40,
    y2: axisY,
    stroke: LINE,
    "stroke-width": 1.5,
    "marker-end": `url(#${arrowId})`,
  });

  const stalest = list.filter((version) => currentRevision && version.inputRevision && version.inputRevision !== currentRevision);

  const svg = svgFrame({
    width: canvasWidth,
    height,
    role: "group",
    label: title || "Reframing lineage",
    desc: `${list.length} recorded framing${list.length === 1 ? "" : "s"} in order, each showing the input revision it was based on.`,
    seed: `${id}|${list.map((version) => version.id).join(",")}`,
    className: "viz-lineage-svg",
    defs: arrowMarker(arrowId, LINE, { width: 8, refX: 7.4 }),
    children: spine + markers,
  });

  return figure({
    kind: "lineage",
    id,
    eyebrow,
    title,
    svg:
      svg +
      '<p class="viz-readout" role="status" data-viz-readout data-default="Hover or focus a framing to see what it was based on.">Hover or focus a framing to see what it was based on.</p>' +
      legend({
        items: [
          { label: "Current framing", color: ACCENT, shape: "circle", dash: "" },
          { label: "Superseded framing", color: MUTED, shape: "circle", dash: "" },
          { label: "Ringed - based on inputs older than the current set", color: WARN, shape: "circle", dash: "2 2" },
        ],
      }),
    caption:
      (caption || "Each marker is a saved framing and the input revision it was written against.") +
      (stalest.length
        ? ` ${stalest.length} framing${stalest.length === 1 ? "" : "s"} predate${
            stalest.length === 1 ? "s" : ""
          } the current inputs: later evidence and steering have not been reconciled into ${
            stalest.length === 1 ? "it" : "them"
          }.`
        : "") +
      " A new framing preserves earlier artifacts; it does not revalidate the candidates, experiments or results that depended on them.",
    scroll: true,
    minWidth: Math.min(1300, canvasWidth),
  });
}
