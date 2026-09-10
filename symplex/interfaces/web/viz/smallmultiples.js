"use strict";
/**
 * Small multiples: one panel per state or per scenario, laid out on a grid so
 * many series can be scanned at once.
 *
 * The shared-y option matters more than it looks. With a shared scale the
 * panels are comparable and a flat series really is flat; with per-panel scales
 * every series fills its panel and a 0.1% wobble looks like a crisis. The
 * default is shared, and the caption always says which was used.
 */

import { el, text, escapeText, figure, emptyPanel, svgFrame, embedSpec, num } from "./svg.js";
import { INK, MUTED, LINE, GRID, PAPER, seriesStyle } from "./color.js";
import { linearScale, extent, combinedExtent, niceDomain, formatNumber } from "./scale.js";
import { missingNote } from "./legend.js";

const asArray = (value) => (Array.isArray(value) ? value : []);
const isNum = (value) => typeof value === "number" && Number.isFinite(value);

function panelPath(x, values, xScale, yScale) {
  const parts = [];
  let open = false;
  let missing = 0;
  for (let i = 0; i < x.length; i++) {
    const px = xScale(x[i]);
    const py = yScale(values[i]);
    if (px === null || py === null) {
      if (!isNum(values[i])) missing++;
      open = false;
      continue;
    }
    parts.push(`${open ? "L" : "M"}${num(px)} ${num(py)}`);
    open = true;
  }
  return { d: parts.join(""), missing };
}

/**
 * spec:
 *   panels    [{id, label, x, values, band:{p05,p95}, color, unit, note}]
 *   x         shared abscissa used by any panel that omits its own
 *   columns   grid columns (default: fits the panel count)
 *   sharedY   true (default) to put every panel on one vertical scale
 *   panelWidth / panelHeight
 */
export function smallMultiples(spec = {}) {
  const panels = asArray(spec.panels).filter((panel) => panel && (panel.values || panel.x));
  const eyebrow = spec.eyebrow || "SMALL MULTIPLES";
  if (!panels.length) {
    return emptyPanel({
      kind: "smallmultiples",
      eyebrow,
      title: "No panels to compare.",
      message: "No series were supplied, so there is nothing to scan side by side.",
      caption: "An empty grid means no series arrived, not that every series was flat.",
    });
  }
  const sharedX = asArray(spec.x);
  const sharedY = spec.sharedY !== false;
  const columns = Math.max(
    1,
    Math.min(6, isNum(spec.columns) ? spec.columns : Math.ceil(Math.sqrt(panels.length))),
  );
  const panelWidth = isNum(spec.panelWidth) ? spec.panelWidth : 190;
  const panelHeight = isNum(spec.panelHeight) ? spec.panelHeight : 96;

  const resolved = panels.map((panel, index) => {
    const style = seriesStyle(index);
    const x = asArray(panel.x).length ? asArray(panel.x) : sharedX;
    return {
      id: String(panel.id ?? `panel-${index}`),
      label: String(panel.label ?? panel.id ?? `Panel ${index + 1}`),
      unit: panel.unit || "",
      note: panel.note || "",
      color: panel.color || style.color,
      dash: panel.dash === undefined ? style.dash : panel.dash,
      x: x.length ? x : asArray(panel.values).map((_, i) => i),
      values: asArray(panel.values),
      band: panel.band || null,
    };
  });

  const globalY = combinedExtent(resolved.map((panel) => panel.values.filter(isNum)));
  const sharedDomain = globalY ? niceDomain(globalY[0], globalY[1], 4) : null;
  let totalMissing = 0;
  let emptyPanels = 0;

  const cells = resolved
    .map((panel) => {
      const margin = { top: 8, right: 8, bottom: 16, left: 34 };
      const left = margin.left;
      const right = panelWidth - margin.right;
      const top = margin.top;
      const bottom = panelHeight - margin.bottom;
      const xSpan = extent(panel.x);
      const localY = extent(panel.values);
      const domain = sharedY ? sharedDomain : localY ? niceDomain(localY[0], localY[1], 3) : null;
      if (!xSpan || !domain) {
        emptyPanels++;
        return `<div class="viz-multiple is-absent"><span class="viz-multiple-title">${escapeText(
          panel.label,
        )}</span><p class="viz-multiple-absent">No finite values recorded for this series.</p></div>`;
      }
      const xScale = linearScale({ domain: xSpan, range: [left, right] });
      const yScale = linearScale({ domain, range: [bottom, top] });
      const { d, missing } = panelPath(panel.x, panel.values, xScale, yScale);
      totalMissing += missing;
      const bandPath =
        panel.band && asArray(panel.band.p05).length && asArray(panel.band.p95).length
          ? (() => {
              const top05 = [];
              const bottom95 = [];
              for (let i = 0; i < panel.x.length; i++) {
                const px = xScale(panel.x[i]);
                const lo = yScale(panel.band.p05[i]);
                const hi = yScale(panel.band.p95[i]);
                if (px === null || lo === null || hi === null) continue;
                top05.push([px, hi]);
                bottom95.push([px, lo]);
              }
              if (top05.length < 2) return "";
              return (
                top05.map((point, i) => `${i ? "L" : "M"}${num(point[0])} ${num(point[1])}`).join("") +
                bottom95
                  .slice()
                  .reverse()
                  .map((point) => `L${num(point[0])} ${num(point[1])}`)
                  .join("") +
                "Z"
              );
            })()
          : "";
      const zeroY = yScale(0);
      const svg = svgFrame({
        width: panelWidth,
        height: panelHeight,
        role: "img",
        label: `${panel.label}${panel.unit ? ` in ${panel.unit}` : ""}`,
        desc: `Range ${formatNumber(domain[0], 3)} to ${formatNumber(domain[1], 3)}${
          sharedY ? " on the shared vertical scale" : " on its own vertical scale"
        }.`,
        seed: `${spec.id || "sm"}|${panel.id}|${sharedY}`,
        className: "viz-multiple-svg",
        children:
          el("rect", { x: left, y: top, width: right - left, height: bottom - top, fill: PAPER, stroke: GRID }) +
          (zeroY !== null && zeroY >= top && zeroY <= bottom
            ? el("line", { x1: left, y1: zeroY, x2: right, y2: zeroY, stroke: MUTED, "stroke-dasharray": "2 2", "stroke-width": 0.8 })
            : "") +
          (bandPath ? el("path", { d: bandPath, fill: panel.color, "fill-opacity": 0.15, stroke: "none" }) : "") +
          (d
            ? el("path", { d, fill: "none", stroke: panel.color, "stroke-width": 1.5, "stroke-dasharray": panel.dash || null })
            : "") +
          text(left - 4, top + 4, formatNumber(domain[1], 3), { "text-anchor": "end", class: "viz-tick", fill: MUTED }) +
          text(left - 4, bottom, formatNumber(domain[0], 3), { "text-anchor": "end", class: "viz-tick", fill: MUTED }),
      });
      return `<div class="viz-multiple"><span class="viz-multiple-title">${escapeText(panel.label)}${
        panel.unit ? `<small>${escapeText(panel.unit)}</small>` : ""
      }</span>${svg}${panel.note ? `<small class="viz-multiple-note">${escapeText(panel.note)}</small>` : ""}</div>`;
    })
    .join("");

  const controls = `<div class="viz-chart-controls" role="group" aria-label="Scale"><button type="button" class="viz-toggle${
    sharedY ? " is-on" : ""
  }" data-viz-shared-y="${sharedY ? "false" : "true"}" aria-pressed="${sharedY}">Shared vertical scale</button></div>`;

  return figure({
    kind: "smallmultiples",
    id: spec.id || null,
    eyebrow,
    title: spec.title || "",
    note: spec.note || "",
    controls,
    svg:
      `<div class="viz-multiples" style="--viz-columns:${num(columns)}">${cells}</div>` +
      embedSpec({ chart: "smallmultiples", ...JSON.parse(JSON.stringify(spec)) }),
    caption:
      (spec.caption || "Each panel is one series under the stated model, not an observation.") +
      (sharedY
        ? " All panels share one vertical scale, so panel heights are directly comparable."
        : " Each panel has its own vertical scale, so apparent amplitude is NOT comparable between panels.") +
      (emptyPanels
        ? ` ${emptyPanels} panel${emptyPanels === 1 ? "" : "s"} had no finite values and ${
            emptyPanels === 1 ? "is" : "are"
          } shown as empty rather than flat.`
        : "") +
      missingNote(totalMissing, "sample"),
    scroll: false,
  });
}
