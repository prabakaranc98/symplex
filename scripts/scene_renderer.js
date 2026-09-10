/* Symplex local 3D trajectory viewer.
 * Build dependency versions: three@0.186.0, esbuild@0.28.2.
 * Official API: https://threejs.org/manual/en/installation.html
 * https://threejs.org/docs/pages/OrbitControls.html
 * Bundle as IIFE; include node_modules/three/LICENSE in the output banner.
 * No scene-provided assets, URLs, code, or network requests are executed.
 */
import {
  Scene, Color, PerspectiveCamera, WebGLRenderer, InstancedMesh, SphereGeometry,
  MeshStandardMaterial, Matrix4, AmbientLight, DirectionalLight, BufferGeometry,
  Float32BufferAttribute, LineBasicMaterial, LineSegments, AxesHelper, Raycaster,
  Vector2,
} from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const mounted = new Set();
const elements = new WeakMap();
const palette = [0x236f64, 0xc37546, 0x626db4, 0xb75272, 0x749146, 0x448cb0];

function assertScene(data) {
  if (!data || !Array.isArray(data.agents) || !data.agents.length || data.agents.length > 300 || !Array.isArray(data.frames) || !data.frames.length || data.frames.length > 400) throw new Error('A bounded simulation scene is required.');
  const ids = new Set(data.agents.map(a => a.id));
  if (ids.size !== data.agents.length) throw new Error('Scene IDs must be unique.');
  let count = 0, previous = -Infinity;
  for (const frame of data.frames) {
    if (!Number.isFinite(frame.time) || frame.time <= previous || !Array.isArray(frame.positions) || !frame.positions.length || frame.positions.length > 300) throw new Error('Scene frames must contain finite, ascending recorded times.');
    previous = frame.time;
    const present = new Set();
    for (const p of frame.positions) {
      if (!ids.has(p.agent_id) || present.has(p.agent_id) || ![p.x, p.y, p.z].every(Number.isFinite)) throw new Error('Scene positions must be finite and reference unique known agents.');
      present.add(p.agent_id);
      if (++count > 60000) throw new Error('Scene exceeds 60,000 recorded positions.');
    }
  }
}

function node(tag, text, styles = {}) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  Object.assign(element.style, styles);
  return element;
}

function numeric(value) {
  return Number(value).toLocaleString(undefined, { maximumSignificantDigits: 5 });
}

function mount(element, data) {
  if (!element || typeof element.replaceChildren !== 'function') throw new Error('A scene mount element is required.');
  assertScene(data);
  elements.get(element)?.dispose();
  const root = node('section', undefined, { display: 'grid', gap: '10px', width: '100%', color: '#29443d', fontFamily: 'inherit' });
  root.setAttribute('aria-label', 'Interactive 3D simulation: ' + data.title);
  const header = node('div', undefined, { display: 'flex', flexWrap: 'wrap', gap: '10px', alignItems: 'baseline' });
  header.append(node('strong', data.title), node('span', data.basis + ' · declared basis, not independently verified', { fontSize: '12px', color: '#64766e' }));
  const viewport = node('div', undefined, { position: 'relative', minHeight: '360px', height: 'min(58vh, 540px)', overflow: 'hidden', borderRadius: '14px', background: '#f3f6f2', border: '1px solid #dfe7de' });
  const toolbar = node('div', undefined, { display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center' });
  const button = (label) => {
    const b = node('button', label, { padding: '7px 12px', borderRadius: '7px', border: '1px solid #cad8cc', background: '#fff', color: '#29443d', cursor: 'pointer', font: 'inherit', fontSize: '12px' });
    b.type = 'button'; toolbar.append(b); return b;
  };
  const playButton = button('Play');
  const previousButton = button('Previous');
  const nextButton = button('Next');
  const resetButton = button('Reset camera');
  const scrub = node('input', undefined, { flex: '1 1 150px', accentColor: '#236f64', minWidth: '120px' });
  Object.assign(scrub, { type: 'range', min: '0', max: String(data.frames.length - 1), step: '1', value: '0' });
  scrub.setAttribute('aria-label', 'Recorded simulation frame');
  const timeLabel = node('output', '', { minWidth: '155px', fontSize: '12px', fontVariantNumeric: 'tabular-nums' });
  toolbar.append(scrub, timeLabel);
  const legend = node('div', undefined, { display: 'flex', flexWrap: 'wrap', gap: '8px 16px', fontSize: '12px' });
  data.agents.forEach((agent, i) => {
    const label = node('span', '● ' + agent.label, { color: '#' + palette[i % palette.length].toString(16).padStart(6, '0'), maxWidth: '260px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' });
    label.title = agent.label; legend.append(label);
  });
  const detail = node('div', '', { padding: '8px 12px', borderRadius: '8px', background: '#fff', position: 'absolute', left: '12px', bottom: '12px', fontSize: '12px', pointerEvents: 'none', maxWidth: 'calc(100% - 24px)' });
  detail.textContent = 'Drag to orbit · scroll to zoom · Shift-drag to pan · hover a point for raw values';
  viewport.append(detail);
  const frameNote = node('p', data.coordinate_frame, { margin: '0', fontSize: '12px', lineHeight: '1.5' });
  const unitsNote = node('p', 'Position units: ' + data.position_unit + '. Time: ' + data.time_unit + '. Playback shows recorded frames at 6 frames/second; no interpolated samples. Trails connect consecutive recorded samples.', { margin: '0', fontSize: '11px', lineHeight: '1.5', color: '#65766e' });
  const limits = node('details', undefined, { fontSize: '12px', color: '#65766e' });
  limits.append(node('summary', 'Interpretation and limitations'));
  const list = node('ul');
  (data.limitations || []).forEach(text => list.append(node('li', text)));
  limits.append(list);
  root.append(header, viewport, toolbar, legend, frameNote, unitsNote, limits);
  element.replaceChildren(root);

  let renderer;
  try {
    renderer = new WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'low-power' });
  } catch (error) {
    viewport.replaceChildren(node('p', 'Interactive 3D requires WebGL 2 in this browser. The recorded scene data remains available for download; a 2D image is not being substituted.', { padding: '24px', lineHeight: '1.6' }));
    toolbar.remove();
    return { dispose() { root.remove(); }, play() {}, pause() {}, setFrame() {}, available: false };
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0xf3f6f2, 1);
  renderer.domElement.style.display = 'block';
  renderer.domElement.style.width = '100%';
  renderer.domElement.style.height = '100%';
  renderer.domElement.setAttribute('aria-label', '3D trajectories; use the frame controls and raw-coordinate readout for values');
  renderer.domElement.tabIndex = 0;
  viewport.prepend(renderer.domElement);
  const world = new Scene();
  const camera = new PerspectiveCamera(42, 1, 0.01, 200);
  camera.position.set(3.5, 2.8, 4.5);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.minDistance = 0.2;
  controls.maxDistance = 40;
  controls.listenToKeyEvents(renderer.domElement);
  controls.update();
  controls.saveState();
  world.add(new AmbientLight(0xffffff, 2));
  const light = new DirectionalLight(0xffffff, 2.2);
  light.position.set(3, 5, 4); world.add(light);

  // One uniform affine display transform preserves relative coordinate proportions.
  // Divide before centering to avoid overflow for large but finite source values.
  let largest = 0;
  for (const frame of data.frames) for (const p of frame.positions) for (const axis of ['x', 'y', 'z']) largest = Math.max(largest, Math.abs(p[axis]));
  largest ||= 1;
  const bounds = { x: [Infinity, -Infinity], y: [Infinity, -Infinity], z: [Infinity, -Infinity] };
  for (const frame of data.frames) for (const p of frame.positions) for (const axis of ['x', 'y', 'z']) {
    const value = p[axis] / largest;
    bounds[axis][0] = Math.min(bounds[axis][0], value); bounds[axis][1] = Math.max(bounds[axis][1], value);
  }
  const centers = Object.fromEntries(Object.entries(bounds).map(([axis, [low, high]]) => [axis, (low + high) / 2]));
  const span = Math.max(...Object.values(bounds).map(([low, high]) => high - low));
  const scale = span > 0 ? 2 / span : 1;
  const transform = p => ['x', 'y', 'z'].map(axis => (p[axis] / largest - centers[axis]) * scale);
  const sourceBounds = node('p', ['x', 'y', 'z'].map(axis => axis.toUpperCase() + ': ' + numeric(bounds[axis][0] * largest) + ' … ' + numeric(bounds[axis][1] * largest)).join('   ·   '), { margin: '0', fontSize: '11px', color: '#65766e' });
  unitsNote.after(sourceBounds);
  const axes = new AxesHelper(1.15);
  axes.position.set(-1.2, -1.2, -1.2);
  world.add(axes);
  const sphere = new SphereGeometry(0.034, 12, 8);
  const material = new MeshStandardMaterial({ color: 0xffffff, roughness: 0.65, metalness: 0 });
  const points = new InstancedMesh(sphere, material, data.agents.length);
  points.frustumCulled = false;
  world.add(points);
  const agentIndex = new Map(data.agents.map((a, i) => [a.id, i]));
  const frames = data.frames.map(f => new Map(f.positions.map(p => [p.agent_id, p])));
  const trails = data.agents.map((agent, index) => {
    const coordinates = [], ends = [];
    for (let i = 1; i < frames.length; i++) {
      const a = frames[i - 1].get(agent.id), b = frames[i].get(agent.id);
      if (a && b) { coordinates.push(...transform(a), ...transform(b)); ends.push(i); }
    }
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new Float32BufferAttribute(coordinates, 3));
    geometry.setDrawRange(0, 0);
    const lineMaterial = new LineBasicMaterial({ color: palette[index % palette.length], transparent: true, opacity: 0.65 });
    const line = new LineSegments(geometry, lineMaterial);
    world.add(line);
    return { geometry, material: lineMaterial, ends };
  });
  let frameIndex = 0, playing = false, disposed = false, request = null, lastStep = 0;
  const matrix = new Matrix4();
  const pointColor = new Color();
  function setFrame(index) {
    if (disposed) return;
    if (!Number.isInteger(index) || index < 0 || index >= data.frames.length) throw new Error('Choose an existing recorded frame.');
    frameIndex = index;
    const frame = data.frames[index];
    points.count = frame.positions.length;
    frame.positions.forEach((p, i) => {
      matrix.makeTranslation(...transform(p));
      points.setMatrixAt(i, matrix);
      points.setColorAt(i, pointColor.setHex(palette[agentIndex.get(p.agent_id) % palette.length]));
    });
    points.instanceMatrix.needsUpdate = true;
    if (points.instanceColor) points.instanceColor.needsUpdate = true;
    points.computeBoundingSphere();
    for (const trail of trails) trail.geometry.setDrawRange(0, trail.ends.filter(end => end <= index).length * 2);
    scrub.value = String(index);
    scrub.setAttribute('aria-valuetext', 'Frame ' + (index + 1) + ', time ' + numeric(frame.time));
    timeLabel.textContent = 't = ' + numeric(frame.time) + ' · frame ' + (index + 1) + '/' + data.frames.length;
  }
  function pause() { playing = false; playButton.textContent = 'Play'; }
  function play() { if (!disposed && data.frames.length > 1) { playing = true; lastStep = performance.now(); playButton.textContent = 'Pause'; } }
  playButton.addEventListener('click', () => playing ? pause() : play());
  previousButton.addEventListener('click', () => { pause(); setFrame(Math.max(0, frameIndex - 1)); });
  nextButton.addEventListener('click', () => { pause(); setFrame(Math.min(data.frames.length - 1, frameIndex + 1)); });
  resetButton.addEventListener('click', () => controls.reset());
  scrub.addEventListener('input', () => { pause(); setFrame(Number(scrub.value)); });
  const raycaster = new Raycaster(), pointer = new Vector2();
  function hover(event) {
    const box = renderer.domElement.getBoundingClientRect();
    pointer.set((event.clientX - box.left) / box.width * 2 - 1, -(event.clientY - box.top) / box.height * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObject(points)[0];
    if (hit && hit.instanceId !== undefined) {
      const p = data.frames[frameIndex].positions[hit.instanceId];
      const label = data.agents[agentIndex.get(p.agent_id)].label;
      detail.textContent = label + ' · x=' + numeric(p.x) + ', y=' + numeric(p.y) + ', z=' + numeric(p.z);
    } else detail.textContent = 'Drag to orbit · scroll to zoom · Shift-drag to pan · hover a point for raw values';
  }
  renderer.domElement.addEventListener('pointermove', hover);
  function resize() {
    const width = Math.max(1, viewport.clientWidth), height = Math.max(1, viewport.clientHeight);
    renderer.setSize(width, height, false);
    camera.aspect = width / height; camera.updateProjectionMatrix();
  }
  const observer = new ResizeObserver(resize); observer.observe(viewport);
  const controller = {
    available: true, play, pause, setFrame,
    dispose() {
      if (disposed) return;
      disposed = true; pause();
      if (request !== null) cancelAnimationFrame(request);
      observer.disconnect();
      renderer.domElement.removeEventListener('pointermove', hover);
      controls.dispose(); sphere.dispose(); material.dispose();
      trails.forEach(t => { t.geometry.dispose(); t.material.dispose(); });
      axes.geometry.dispose(); axes.material.dispose();
      renderer.dispose(); renderer.forceContextLoss();
      root.remove(); mounted.delete(controller); elements.delete(element);
    },
  };
  function animate(now) {
    if (disposed) return;
    if (!element.isConnected) { controller.dispose(); return; }
    if (playing && now - lastStep >= 1000 / 6) { setFrame((frameIndex + 1) % data.frames.length); lastStep = now; }
    controls.update(); renderer.render(world, camera);
    request = requestAnimationFrame(animate);
  }
  setFrame(0); resize();
  mounted.add(controller); elements.set(element, controller);
  request = requestAnimationFrame(animate);
  return controller;
}

window.SymplexScene = Object.freeze({ mount, dispose() { [...mounted].forEach(viewer => viewer.dispose()); }, version: 'three-0.186.0-symplex-scene-v1' });
