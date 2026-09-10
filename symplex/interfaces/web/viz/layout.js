"use strict";
/**
 * Graph layout algorithms.
 *
 * Two layouts are implemented properly rather than approximated:
 *
 *   layeredLayout - Sugiyama. Deterministic DFS cycle breaking, longest-path
 *     layering, dummy nodes for long edges, median/barycentre ordering with
 *     adjacent transposition over several sweeps (best crossing count wins),
 *     and coordinate assignment by priority-weighted isotonic regression
 *     (PAVA). The isotonic step is the L2-optimal placement subject to the
 *     ordering and minimum-separation constraints, so dummy chains - which
 *     carry the highest weight - come out straight.
 *
 *   forceLayout - Fruchterman-Reingold with a Barnes-Hut quadtree above a
 *     node threshold and exact quadratic repulsion below it. Initial placement
 *     is a phyllotaxis spiral perturbed by a seeded PRNG, so the same graph
 *     lays out identically on every run, in every process. Nothing here reads
 *     Math.random or the clock.
 *
 * Edge routing produces orthogonal polylines, self-loop arcs and offsets for
 * bundled parallel edges.
 */

import { num, seededRandom, hashCode } from "./svg.js";

const isNum = (value) => typeof value === "number" && Number.isFinite(value);

/** Normalise loose node/edge input into indexed structures. Order is preserved. */
function prepare(nodes, edges) {
  const list = [];
  const index = new Map();
  for (const node of nodes || []) {
    const id = String(node?.id ?? "");
    if (!id || index.has(id)) continue;
    index.set(id, list.length);
    list.push({
      id,
      width: isNum(node.width) ? node.width : 0,
      height: isNum(node.height) ? node.height : 0,
      data: node,
    });
  }
  const links = [];
  const selfLoops = [];
  for (const edge of edges || []) {
    const source = String(edge?.source ?? "");
    const target = String(edge?.target ?? "");
    if (!index.has(source) || !index.has(target)) continue;
    if (source === target) {
      selfLoops.push({ ...edge, source, target });
      continue;
    }
    links.push({ ...edge, source, target, si: index.get(source), ti: index.get(target) });
  }
  return { list, index, links, selfLoops };
}

/**
 * Break cycles by reversing back edges found in a deterministic DFS.
 * Returns links with a `reversed` flag; the drawing restores the true direction.
 */
function breakCycles(count, links) {
  const outgoing = Array.from({ length: count }, () => []);
  links.forEach((link, i) => outgoing[link.si].push(i));
  const WHITE = 0;
  const GREY = 1;
  const BLACK = 2;
  const colour = new Array(count).fill(WHITE);
  const reversed = new Set();
  const visit = (start) => {
    const stack = [[start, 0]];
    colour[start] = GREY;
    while (stack.length) {
      const frame = stack[stack.length - 1];
      const [node, cursor] = frame;
      if (cursor >= outgoing[node].length) {
        colour[node] = BLACK;
        stack.pop();
        continue;
      }
      frame[1] = cursor + 1;
      const edgeIndex = outgoing[node][cursor];
      const next = links[edgeIndex].ti;
      if (colour[next] === GREY) {
        reversed.add(edgeIndex);
      } else if (colour[next] === WHITE) {
        colour[next] = GREY;
        stack.push([next, 0]);
      }
    }
  };
  for (let i = 0; i < count; i++) if (colour[i] === WHITE) visit(i);
  return links.map((link, i) =>
    reversed.has(i)
      ? { ...link, si: link.ti, ti: link.si, reversed: true }
      : { ...link, reversed: false },
  );
}

/** Longest-path layering over the acyclic graph. Sources sit on layer 0. */
function assignLayers(count, links) {
  const incoming = Array.from({ length: count }, () => []);
  const outgoing = Array.from({ length: count }, () => []);
  for (const link of links) {
    outgoing[link.si].push(link.ti);
    incoming[link.ti].push(link.si);
  }
  const degree = incoming.map((list) => list.length);
  const layer = new Array(count).fill(0);
  const queue = [];
  for (let i = 0; i < count; i++) if (!degree[i]) queue.push(i);
  let head = 0;
  let visited = 0;
  while (head < queue.length) {
    const node = queue[head++];
    visited++;
    for (const next of outgoing[node]) {
      layer[next] = Math.max(layer[next], layer[node] + 1);
      if (--degree[next] === 0) queue.push(next);
    }
  }
  // A residual cycle can only mean the cycle break missed an edge; place the
  // remainder one layer past their deepest resolved predecessor rather than
  // dropping them from the drawing.
  if (visited < count) {
    for (let i = 0; i < count; i++) {
      if (degree[i] <= 0) continue;
      layer[i] = Math.max(layer[i], Math.max(0, ...incoming[i].map((p) => layer[p] + 1)));
    }
  }
  return layer;
}

/** Count crossings between two adjacent layers with a Fenwick tree. */
function countCrossings(pairs, southSize) {
  if (pairs.length < 2) return 0;
  const sorted = pairs.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const tree = new Array(southSize + 2).fill(0);
  const add = (i) => {
    for (let k = i + 1; k <= southSize + 1; k += k & -k) tree[k]++;
  };
  const sum = (i) => {
    let total = 0;
    for (let k = i + 1; k > 0; k -= k & -k) total += tree[k];
    return total;
  };
  let crossings = 0;
  let placed = 0;
  for (const [, south] of sorted) {
    crossings += placed - sum(south);
    add(south);
    placed++;
  }
  return crossings;
}

function positionsOf(rows) {
  const map = new Map();
  rows.forEach((row) => row.forEach((key, i) => map.set(key, i)));
  return map;
}

function crossingsBetween(rows, segments, l) {
  const position = positionsOf(rows);
  const pairs = [];
  for (const segment of segments) {
    if (segment.layer !== l) continue;
    const a = position.get(segment.si);
    const b = position.get(segment.ti);
    if (a === undefined || b === undefined) continue;
    pairs.push([a, b]);
  }
  return countCrossings(pairs, rows[l + 1] ? rows[l + 1].length : 0);
}

function totalCrossings(layerCount, rows, segments) {
  let total = 0;
  for (let l = 0; l + 1 < layerCount; l++) total += crossingsBetween(rows, segments, l);
  return total;
}

function medianOf(values) {
  if (!values.length) return -1;
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  if (sorted.length % 2) return sorted[mid];
  if (sorted.length === 2) return (sorted[0] + sorted[1]) / 2;
  const left = sorted[mid - 1] - sorted[0];
  const right = sorted[sorted.length - 1] - sorted[mid];
  return left + right === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : (sorted[mid - 1] * right + sorted[mid] * left) / (left + right);
}

/** Weighted pool-adjacent-violators: the L2-optimal non-decreasing fit. */
function isotonic(values, weights) {
  const blocks = [];
  for (let i = 0; i < values.length; i++) {
    const weight = weights[i] > 0 ? weights[i] : 1e-6;
    let sum = values[i] * weight;
    let mass = weight;
    let count = 1;
    let value = sum / mass;
    while (blocks.length && blocks[blocks.length - 1].value > value) {
      const previous = blocks.pop();
      sum += previous.sum;
      mass += previous.mass;
      count += previous.count;
      value = sum / mass;
    }
    blocks.push({ sum, mass, count, value });
  }
  const out = [];
  for (const block of blocks) for (let i = 0; i < block.count; i++) out.push(block.value);
  return out;
}

/**
 * Sugiyama layered layout.
 *
 * options: {direction: "LR"|"TB", layerGap, nodeGap, nodeWidth, nodeHeight,
 *           sweeps, padding}
 * returns: {nodes, edges, selfLoops, width, height, layerCount, crossings}
 */
export function layeredLayout({ nodes, edges } = {}, options = {}) {
  const settings = {
    direction: "LR",
    layerGap: 96,
    nodeGap: 18,
    nodeWidth: 150,
    nodeHeight: 46,
    sweeps: 8,
    padding: 16,
    ...options,
  };
  const { list, links: rawLinks, selfLoops } = prepare(nodes, edges);
  if (!list.length) {
    return {
      nodes: [],
      edges: [],
      selfLoops: [],
      width: 0,
      height: 0,
      layerCount: 0,
      crossings: 0,
    };
  }
  for (const node of list) {
    if (!node.width) node.width = settings.nodeWidth;
    if (!node.height) node.height = settings.nodeHeight;
  }
  const links = breakCycles(list.length, rawLinks);
  const layer = assignLayers(list.length, links);
  const layerCount = Math.max(1, Math.max(...layer) + 1);

  const vertices = list.map((node, i) => ({
    key: i,
    id: node.id,
    dummy: false,
    layer: layer[i],
    width: node.width,
    height: node.height,
    node,
  }));
  const segments = [];
  let nextKey = vertices.length;
  for (const link of links) {
    const from = link.si;
    const to = link.ti;
    const span = layer[to] - layer[from];
    const chain = [from];
    if (span > 1) {
      for (let l = layer[from] + 1; l < layer[to]; l++) {
        const dummy = {
          key: nextKey++,
          id: `__dummy_${link.si}_${link.ti}_${l}`,
          dummy: true,
          layer: l,
          width: 1,
          height: 1,
          node: null,
        };
        vertices.push(dummy);
        chain.push(dummy.key);
      }
    }
    chain.push(to);
    link.chain = chain;
    for (let i = 0; i + 1 < chain.length; i++) {
      segments.push({ si: chain[i], ti: chain[i + 1], layer: vertices[chain[i]].layer });
    }
  }

  const order = Array.from({ length: layerCount }, () => []);
  for (const vertex of vertices) order[vertex.layer].push(vertex.key);

  const neighboursDown = new Map();
  const neighboursUp = new Map();
  for (const segment of segments) {
    if (!neighboursDown.has(segment.ti)) neighboursDown.set(segment.ti, []);
    neighboursDown.get(segment.ti).push(segment.si);
    if (!neighboursUp.has(segment.si)) neighboursUp.set(segment.si, []);
    neighboursUp.get(segment.si).push(segment.ti);
  }

  const transpose = (rows) => {
    let improved = true;
    let guard = 0;
    while (improved && guard++ < 12) {
      improved = false;
      for (let l = 0; l + 1 < layerCount; l++) {
        for (let i = 0; i + 1 < rows[l].length; i++) {
          const before =
            crossingsBetween(rows, segments, l) + (l > 0 ? crossingsBetween(rows, segments, l - 1) : 0);
          const row = rows[l];
          const swapped = row[i];
          row[i] = row[i + 1];
          row[i + 1] = swapped;
          const after =
            crossingsBetween(rows, segments, l) + (l > 0 ? crossingsBetween(rows, segments, l - 1) : 0);
          if (after < before) improved = true;
          else {
            row[i + 1] = row[i];
            row[i] = swapped;
          }
        }
      }
    }
  };

  let best = order.map((row) => row.slice());
  let bestScore = totalCrossings(layerCount, best, segments);
  const working = order.map((row) => row.slice());
  for (let sweep = 0; sweep < settings.sweeps; sweep++) {
    const down = sweep % 2 === 0;
    const range = down
      ? Array.from({ length: Math.max(0, layerCount - 1) }, (_, i) => i + 1)
      : Array.from({ length: Math.max(0, layerCount - 1) }, (_, i) => layerCount - 2 - i);
    for (const l of range) {
      const position = positionsOf(working);
      const table = down ? neighboursDown : neighboursUp;
      const entries = working[l].map((key, i) => {
        const neighbours = (table.get(key) || [])
          .map((other) => position.get(other))
          .filter((value) => value !== undefined);
        return { key, i, median: neighbours.length ? medianOf(neighbours) : -1 };
      });
      const movable = entries
        .filter((entry) => entry.median >= 0)
        .sort((a, b) => a.median - b.median || a.i - b.i);
      let cursor = 0;
      working[l] = entries.map((entry) => (entry.median >= 0 ? movable[cursor++].key : entry.key));
    }
    transpose(working);
    const score = totalCrossings(layerCount, working, segments);
    if (score < bestScore) {
      bestScore = score;
      best = working.map((row) => row.slice());
    }
  }

  const byKey = new Map(vertices.map((vertex) => [vertex.key, vertex]));
  const across = settings.direction === "LR" ? "height" : "width";
  const along = settings.direction === "LR" ? "width" : "height";
  const coordinate = new Map();
  for (const row of best) {
    let cursor = 0;
    for (const key of row) {
      const vertex = byKey.get(key);
      coordinate.set(key, cursor + vertex[across] / 2);
      cursor += vertex[across] + settings.nodeGap;
    }
  }
  const degreeOf = (key) =>
    (neighboursDown.get(key) || []).length + (neighboursUp.get(key) || []).length;
  for (let pass = 0; pass < 6; pass++) {
    const downward = pass % 2 === 0;
    const range = downward
      ? Array.from({ length: layerCount }, (_, i) => i)
      : Array.from({ length: layerCount }, (_, i) => layerCount - 1 - i);
    for (const l of range) {
      const row = best[l];
      if (!row.length) continue;
      const table = downward ? neighboursDown : neighboursUp;
      const desired = row.map((key) => {
        const neighbours = (table.get(key) || [])
          .map((other) => coordinate.get(other))
          .filter((value) => value !== undefined && isNum(value));
        return neighbours.length ? medianOf(neighbours) : coordinate.get(key);
      });
      const weights = row.map((key) => (byKey.get(key).dummy ? 12 : 1 + degreeOf(key)));
      const offsets = [];
      let cumulative = 0;
      row.forEach((key, i) => {
        if (i > 0) {
          const previous = byKey.get(row[i - 1]);
          const current = byKey.get(key);
          cumulative += (previous[across] + current[across]) / 2 + settings.nodeGap;
        }
        offsets.push(cumulative);
      });
      const fitted = isotonic(
        desired.map((value, i) => (isNum(value) ? value : offsets[i]) - offsets[i]),
        weights,
      );
      row.forEach((key, i) => coordinate.set(key, fitted[i] + offsets[i]));
    }
  }

  let minAcross = Infinity;
  let maxAcross = -Infinity;
  for (const vertex of vertices) {
    const centre = coordinate.get(vertex.key);
    if (!isNum(centre)) continue;
    minAcross = Math.min(minAcross, centre - vertex[across] / 2);
    maxAcross = Math.max(maxAcross, centre + vertex[across] / 2);
  }
  if (!isNum(minAcross)) {
    minAcross = 0;
    maxAcross = 1;
  }
  const layerExtent = [];
  for (let l = 0; l < layerCount; l++) {
    layerExtent[l] = Math.max(
      settings.direction === "LR" ? settings.nodeWidth : settings.nodeHeight,
      ...best[l].map((key) => byKey.get(key)[along] || 0),
      0,
    );
  }
  const layerStart = [];
  let alongCursor = settings.padding;
  for (let l = 0; l < layerCount; l++) {
    layerStart[l] = alongCursor + layerExtent[l] / 2;
    alongCursor += layerExtent[l] + settings.layerGap;
  }
  const alongTotal = alongCursor - settings.layerGap + settings.padding;
  const acrossTotal = maxAcross - minAcross + settings.padding * 2;

  const place = (vertex) => {
    const centreAcross = coordinate.get(vertex.key) - minAcross + settings.padding;
    const centreAlong = layerStart[vertex.layer];
    return settings.direction === "LR"
      ? { x: centreAlong, y: centreAcross }
      : { x: centreAcross, y: centreAlong };
  };

  const placedNodes = vertices
    .filter((vertex) => !vertex.dummy)
    .map((vertex) => {
      const point = place(vertex);
      return {
        id: vertex.id,
        x: point.x,
        y: point.y,
        width: vertex.width,
        height: vertex.height,
        layer: vertex.layer,
        order: best[vertex.layer].indexOf(vertex.key),
        data: vertex.node.data,
      };
    });

  const placedEdges = links.map((link) => {
    const points = link.chain.map((key) => {
      const point = place(byKey.get(key));
      return [point.x, point.y];
    });
    return {
      ...(link.data || {}),
      id: link.id,
      source: link.reversed ? link.target : link.source,
      target: link.reversed ? link.source : link.target,
      reversed: Boolean(link.reversed),
      points: link.reversed ? points.slice().reverse() : points,
    };
  });

  return {
    nodes: placedNodes,
    edges: placedEdges,
    selfLoops,
    width: settings.direction === "LR" ? alongTotal : acrossTotal,
    height: settings.direction === "LR" ? acrossTotal : alongTotal,
    layerCount,
    crossings: bestScore,
  };
}

/** Deterministic starting positions: a phyllotaxis spiral with seeded jitter. */
function seedPositions(list, width, height, random) {
  const golden = Math.PI * (3 - Math.sqrt(5));
  const radius = Math.min(width, height) * 0.42;
  return list.map((node, i) => {
    const t = list.length === 1 ? 0 : i / list.length;
    const r = radius * Math.sqrt(t + 1 / (2 * list.length));
    const angle = i * golden;
    return {
      id: node.id,
      x: width / 2 + r * Math.cos(angle) + (random() - 0.5) * 4,
      y: height / 2 + r * Math.sin(angle) + (random() - 0.5) * 4,
      width: node.width,
      height: node.height,
      data: node.data,
    };
  });
}

function applyRepulsion(point, x, y, mass, k, accumulator) {
  let dx = point.x - x;
  let dy = point.y - y;
  let distance = Math.sqrt(dx * dx + dy * dy);
  if (distance < 1e-6) {
    // Deterministic separation for coincident points: push along a direction
    // derived from the node identity rather than a random one.
    const angle = (parseInt(hashCode(point.id).slice(0, 4), 36) % 360) * (Math.PI / 180);
    dx = Math.cos(angle) * 0.01;
    dy = Math.sin(angle) * 0.01;
    distance = 0.01;
  }
  const force = (k * k * mass) / distance;
  accumulator.x += (dx / distance) * force;
  accumulator.y += (dy / distance) * force;
}

/** Barnes-Hut quadtree over the current positions. */
function buildQuadtree(points, bounds) {
  const quadrant = (cell, point) => {
    const half = cell.size / 2;
    const right = point.x >= cell.x + half ? 1 : 0;
    const below = point.y >= cell.y + half ? 1 : 0;
    return below * 2 + right;
  };
  const insert = (cell, point, depth) => {
    cell.mass += 1;
    cell.cx += (point.x - cell.cx) / cell.mass;
    cell.cy += (point.y - cell.cy) / cell.mass;
    if (depth > 20) return;
    if (!cell.children && cell.point === null && cell.mass === 1) {
      cell.point = point;
      return;
    }
    if (!cell.children) {
      cell.children = [];
      const half = cell.size / 2;
      for (let i = 0; i < 4; i++) {
        cell.children.push({
          x: cell.x + (i % 2) * half,
          y: cell.y + Math.floor(i / 2) * half,
          size: half,
          mass: 0,
          cx: 0,
          cy: 0,
          point: null,
          children: null,
        });
      }
      if (cell.point) {
        const existing = cell.point;
        cell.point = null;
        insert(cell.children[quadrant(cell, existing)], existing, depth + 1);
      }
    }
    insert(cell.children[quadrant(cell, point)], point, depth + 1);
  };
  const root = { ...bounds, mass: 0, cx: 0, cy: 0, point: null, children: null };
  for (const point of points) insert(root, point, 0);
  return root;
}

function barnesHutForce(cell, point, k, theta, accumulator) {
  if (!cell || !cell.mass) return;
  if (cell.point) {
    if (cell.point !== point) applyRepulsion(point, cell.point.x, cell.point.y, 1, k, accumulator);
    return;
  }
  const dx = cell.cx - point.x;
  const dy = cell.cy - point.y;
  const distance = Math.sqrt(dx * dx + dy * dy) || 1e-6;
  if (!cell.children || cell.size / distance < theta) {
    applyRepulsion(point, cell.cx, cell.cy, cell.mass, k, accumulator);
    return;
  }
  for (const child of cell.children) barnesHutForce(child, point, k, theta, accumulator);
}

/**
 * Fruchterman-Reingold force-directed layout.
 *
 * options: {width, height, iterations, seed, barnesHutThreshold, theta,
 *           gravity, edgeStrength, padding}
 */
export function forceLayout({ nodes, edges } = {}, options = {}) {
  const settings = {
    width: 720,
    height: 420,
    iterations: 260,
    barnesHutThreshold: 60,
    theta: 0.9,
    gravity: 0.035,
    edgeStrength: 1,
    padding: 26,
    seed: null,
    ...options,
  };
  const { list, links, selfLoops } = prepare(nodes, edges);
  if (!list.length) {
    return {
      nodes: [],
      edges: [],
      selfLoops: [],
      width: settings.width,
      height: settings.height,
      iterations: 0,
      seed: "",
    };
  }
  const seed =
    settings.seed ??
    hashCode(
      list.map((node) => node.id).join("|") +
        "//" +
        links.map((link) => `${link.source}>${link.target}`).join("|"),
    );
  const random = seededRandom(seed);
  const points = seedPositions(list, settings.width, settings.height, random);
  const byId = new Map(points.map((point) => [point.id, point]));
  const indexOf = new Map(points.map((point, i) => [point.id, i]));
  if (points.length === 1) {
    points[0].x = settings.width / 2;
    points[0].y = settings.height / 2;
  }
  const area = settings.width * settings.height;
  const k = Math.sqrt(area / points.length) * 0.72;
  const iterations = Math.max(1, Math.round(settings.iterations));
  let temperature = Math.min(settings.width, settings.height) * 0.12;
  const cooling = temperature / (iterations + 1);
  const useTree = points.length > settings.barnesHutThreshold;

  for (let step = 0; step < iterations; step++) {
    const forces = points.map(() => ({ x: 0, y: 0 }));
    if (useTree) {
      const size = Math.max(settings.width, settings.height) * 4;
      const tree = buildQuadtree(points, { x: -size / 2, y: -size / 2, size });
      points.forEach((point, i) => barnesHutForce(tree, point, k, settings.theta, forces[i]));
    } else {
      for (let i = 0; i < points.length; i++) {
        for (let j = 0; j < points.length; j++) {
          if (i === j) continue;
          applyRepulsion(points[i], points[j].x, points[j].y, 1, k, forces[i]);
        }
      }
    }
    for (const link of links) {
      const source = byId.get(link.source);
      const target = byId.get(link.target);
      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const distance = Math.sqrt(dx * dx + dy * dy) || 1e-6;
      const force = ((distance * distance) / k) * settings.edgeStrength;
      const fx = (dx / distance) * force;
      const fy = (dy / distance) * force;
      const si = indexOf.get(link.source);
      const ti = indexOf.get(link.target);
      forces[si].x += fx;
      forces[si].y += fy;
      forces[ti].x -= fx;
      forces[ti].y -= fy;
    }
    points.forEach((point, i) => {
      forces[i].x += (settings.width / 2 - point.x) * settings.gravity * k * 0.05;
      forces[i].y += (settings.height / 2 - point.y) * settings.gravity * k * 0.05;
      const magnitude = Math.sqrt(forces[i].x * forces[i].x + forces[i].y * forces[i].y) || 1e-6;
      const limited = Math.min(magnitude, temperature);
      point.x += (forces[i].x / magnitude) * limited;
      point.y += (forces[i].y / magnitude) * limited;
    });
    temperature = Math.max(0.01, temperature - cooling);
  }

  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const innerWidth = Math.max(1, settings.width - settings.padding * 2);
  const innerHeight = Math.max(1, settings.height - settings.padding * 2);
  const scale = Math.min(innerWidth / spanX, innerHeight / spanY);
  const offsetX = settings.padding + (innerWidth - spanX * scale) / 2;
  const offsetY = settings.padding + (innerHeight - spanY * scale) / 2;
  for (const point of points) {
    point.x = offsetX + (point.x - minX) * scale;
    point.y = offsetY + (point.y - minY) * scale;
  }

  return {
    nodes: points,
    edges: links.map((link) => {
      const source = byId.get(link.source);
      const target = byId.get(link.target);
      return { ...link, points: [[source.x, source.y], [target.x, target.y]] };
    }),
    selfLoops,
    width: settings.width,
    height: settings.height,
    iterations,
    seed: String(seed),
  };
}

/** Evenly spaced points on a circle; the canonical causal-loop arrangement. */
export function circularLayout(ids, { cx = 0, cy = 0, radius = 100, startAngle = -Math.PI / 2 } = {}) {
  const list = (ids || []).map(String);
  return list.map((id, i) => {
    const angle = startAngle + (i / Math.max(1, list.length)) * Math.PI * 2;
    return {
      id,
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle),
      angle,
      index: i,
    };
  });
}

/** Where a straight line from `from` towards `to` leaves the box around `from`. */
export function anchorOnRect(from, to, width, height) {
  const dx = to[0] - from[0];
  const dy = to[1] - from[1];
  if (!dx && !dy) return [from[0], from[1]];
  const halfW = width / 2;
  const halfH = height / 2;
  const scale = Math.min(
    dx === 0 ? Infinity : Math.abs(halfW / dx),
    dy === 0 ? Infinity : Math.abs(halfH / dy),
  );
  return [from[0] + dx * scale, from[1] + dy * scale];
}

/** Shorten a segment at both ends so an arrowhead does not sit under a node. */
export function trimSegment(from, to, startTrim = 0, endTrim = 0) {
  const dx = to[0] - from[0];
  const dy = to[1] - from[1];
  const length = Math.hypot(dx, dy);
  if (!length) return [from, to];
  const ux = dx / length;
  const uy = dy / length;
  return [
    [from[0] + ux * startTrim, from[1] + uy * startTrim],
    [to[0] - ux * endTrim, to[1] - uy * endTrim],
  ];
}

/**
 * Orthogonal route between two points through optional waypoints.
 * Corners are rounded by `radius`, which keeps dense diagrams readable.
 */
export function orthogonalPath(points, { radius = 7, direction = "LR" } = {}) {
  const route = [];
  for (const point of points || []) {
    if (!isNum(point?.[0]) || !isNum(point?.[1])) continue;
    route.push([point[0], point[1]]);
  }
  if (route.length < 2) return "";
  const expanded = [route[0]];
  for (let i = 1; i < route.length; i++) {
    const previous = expanded[expanded.length - 1];
    const current = route[i];
    if (direction === "LR") {
      const midX = (previous[0] + current[0]) / 2;
      if (Math.abs(previous[1] - current[1]) > 0.5) expanded.push([midX, previous[1]], [midX, current[1]]);
    } else {
      const midY = (previous[1] + current[1]) / 2;
      if (Math.abs(previous[0] - current[0]) > 0.5) expanded.push([previous[0], midY], [current[0], midY]);
    }
    expanded.push(current);
  }
  const parts = [`M${num(expanded[0][0])} ${num(expanded[0][1])}`];
  for (let i = 1; i < expanded.length; i++) {
    const previous = expanded[i - 1];
    const current = expanded[i];
    const next = expanded[i + 1];
    if (!next) {
      parts.push(`L${num(current[0])} ${num(current[1])}`);
      break;
    }
    const inLength = Math.hypot(current[0] - previous[0], current[1] - previous[1]);
    const outLength = Math.hypot(next[0] - current[0], next[1] - current[1]);
    const r = Math.min(radius, inLength / 2, outLength / 2);
    if (!(r > 0.5)) {
      parts.push(`L${num(current[0])} ${num(current[1])}`);
      continue;
    }
    const before = [
      current[0] - ((current[0] - previous[0]) / inLength) * r,
      current[1] - ((current[1] - previous[1]) / inLength) * r,
    ];
    const after = [
      current[0] + ((next[0] - current[0]) / outLength) * r,
      current[1] + ((next[1] - current[1]) / outLength) * r,
    ];
    parts.push(
      `L${num(before[0])} ${num(before[1])}Q${num(current[0])} ${num(current[1])} ${num(after[0])} ${num(after[1])}`,
    );
  }
  return parts.join("");
}

/** Catmull-Rom smoothed path, for edges that read better as curves. */
export function curvePath(points, { tension = 0.5 } = {}) {
  const route = (points || []).filter((point) => isNum(point?.[0]) && isNum(point?.[1]));
  if (route.length < 2) return "";
  if (route.length === 2) {
    return `M${num(route[0][0])} ${num(route[0][1])}L${num(route[1][0])} ${num(route[1][1])}`;
  }
  const parts = [`M${num(route[0][0])} ${num(route[0][1])}`];
  for (let i = 0; i < route.length - 1; i++) {
    const p0 = route[Math.max(0, i - 1)];
    const p1 = route[i];
    const p2 = route[i + 1];
    const p3 = route[Math.min(route.length - 1, i + 2)];
    const c1 = [p1[0] + ((p2[0] - p0[0]) / 6) * tension * 2, p1[1] + ((p2[1] - p0[1]) / 6) * tension * 2];
    const c2 = [p2[0] - ((p3[0] - p1[0]) / 6) * tension * 2, p2[1] - ((p3[1] - p1[1]) / 6) * tension * 2];
    parts.push(`C${num(c1[0])} ${num(c1[1])} ${num(c2[0])} ${num(c2[1])} ${num(p2[0])} ${num(p2[1])}`);
  }
  return parts.join("");
}

/** A self-loop arc above a node, offset by `index` when a node has several. */
export function selfLoopPath(node, { index = 0, size = 26 } = {}) {
  if (!isNum(node?.x) || !isNum(node?.y)) return "";
  const halfWidth = (node.width || 40) / 2;
  const halfHeight = (node.height || 24) / 2;
  const lift = size + index * 9;
  const left = node.x - halfWidth * 0.45;
  const right = node.x + halfWidth * 0.45;
  const top = node.y - halfHeight;
  return (
    `M${num(right)} ${num(top)}` +
    `C${num(right + lift * 0.8)} ${num(top - lift)} ${num(left - lift * 0.8)} ${num(top - lift)} ${num(left)} ${num(top)}`
  );
}

/**
 * Assign a lateral offset to edges that share a node pair so parallel
 * couplings stay individually visible instead of overprinting.
 * Input order is preserved.
 */
export function bundleParallel(edges) {
  const counts = new Map();
  const seen = new Map();
  for (const edge of edges || []) {
    const key = [String(edge.source), String(edge.target)].sort().join(" ");
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  return (edges || []).map((edge) => {
    const key = [String(edge.source), String(edge.target)].sort().join(" ");
    const index = seen.get(key) || 0;
    seen.set(key, index + 1);
    const count = counts.get(key) || 1;
    return {
      ...edge,
      bundleIndex: index,
      bundleCount: count,
      bundleOffset: count === 1 ? 0 : (index - (count - 1) / 2) * 11,
    };
  });
}

/** Offset a route sideways at its midpoint, separating bundled parallel edges. */
export function offsetRoute(points, offset) {
  if (!offset || !points || points.length < 2) return points;
  const [x1, y1] = points[0];
  const [x2, y2] = points[points.length - 1];
  const dx = x2 - x1;
  const dy = y2 - y1;
  const length = Math.hypot(dx, dy) || 1;
  const nx = -(dy / length) * offset;
  const ny = (dx / length) * offset;
  return [points[0], [(x1 + x2) / 2 + nx, (y1 + y2) / 2 + ny], points[points.length - 1]];
}

/** Every simple directed cycle up to `maxLength`, in deterministic order. */
export function findCycles(nodes, edges, { maxLength = 8, limit = 24 } = {}) {
  const ids = (nodes || []).map((node) => String(node.id ?? node));
  const outgoing = new Map(ids.map((id) => [id, []]));
  for (const edge of edges || []) {
    const source = String(edge.source);
    const target = String(edge.target);
    if (!outgoing.has(source) || !outgoing.has(target)) continue;
    outgoing.get(source).push(target);
  }
  const found = [];
  const seen = new Set();
  const rank = new Map(ids.map((id, i) => [id, i]));
  const walk = (start, node, path, visited) => {
    if (found.length >= limit) return;
    for (const next of outgoing.get(node) || []) {
      if (next === start && path.length > 1) {
        const rotated = path.slice();
        const smallest = rotated.reduce(
          (bestIndex, id, i) => (rank.get(id) < rank.get(rotated[bestIndex]) ? i : bestIndex),
          0,
        );
        const canonical = rotated.slice(smallest).concat(rotated.slice(0, smallest));
        const key = canonical.join(" ");
        if (!seen.has(key)) {
          seen.add(key);
          found.push(canonical);
        }
        continue;
      }
      if (visited.has(next) || path.length >= maxLength) continue;
      if (rank.get(next) < rank.get(start)) continue;
      visited.add(next);
      path.push(next);
      walk(start, next, path, visited);
      path.pop();
      visited.delete(next);
    }
  };
  for (const id of ids) {
    if (found.length >= limit) break;
    walk(id, id, [id], new Set([id]));
  }
  return found.sort((a, b) => a.length - b.length || a.join().localeCompare(b.join()));
}
