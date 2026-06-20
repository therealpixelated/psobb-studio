// Browser-emulation test: verify auto-play triggers state.anim.playing=true
// after opening a known walker (dragon BML).
//
// Uses jsdom to mock the DOM, mocks THREE.js with shims sufficient for
// the model_viewer's tickAnimation/disposeMesh paths, and drives a real
// HTTP fetch against the local server.
//
// Exit code 0 on pass; non-zero on regression.

import { JSDOM } from 'jsdom';
import { readFileSync } from 'fs';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(__dirname, '..');
const HTML = readFileSync(resolve(REPO, 'static/index.html'), 'utf8');
const dom = new JSDOM(HTML, { url: 'http://127.0.0.1:8765/', runScripts: 'outside-only' });
const { window } = dom;
const { document } = window;

// Add APIs that jsdom doesn't provide
window.requestAnimationFrame = () => 1;
window.cancelAnimationFrame = () => {};
window.devicePixelRatio = 1;
window.ResizeObserver = class { observe() {} disconnect() {} };

// Mock THREE - provide minimal classes; the test doesn't render
const THREE = {
  Group: class { constructor() { this.children = []; this.userData = {}; this.isGroup = true; this.matrixWorld = { extractBasis: () => {} }; this.position = { set: () => {}, copy: () => {} }; this.scale = { set: () => {}, copy: () => {} }; this.rotation = { x: 0, y: 0, z: 0 }; this.matrix = { identity: () => {} }; } add(c) { this.children.push(c); } remove(c) { this.children = this.children.filter(x => x !== c); } traverse(fn) { fn(this); for (const c of this.children) if (c.traverse) c.traverse(fn); else fn(c); } },
  Mesh: class { constructor(geo, mat) { this.userData = {}; this.material = mat || null; this.geometry = geo || null; this.frustumCulled = true; this.visible = true; this.isMesh = true; } },
  BufferGeometry: class { constructor() { this.attributes = {}; this.index = null; this.boundingSphere = null; } setAttribute(name, attr) { this.attributes[name] = attr; } setIndex(i) { this.index = i; } dispose() {} computeVertexNormals() {} computeBoundingSphere() { this.boundingSphere = { center: { x: 0, y: 0, z: 0 }, radius: 1 }; } },
  BufferAttribute: class { constructor(arr, size) { this.array = arr; this.itemSize = size; this.count = arr.length / size; this.needsUpdate = false; } },
  Float32BufferAttribute: class { constructor(arr, size) { this.array = arr instanceof Float32Array ? arr : new Float32Array(arr); this.itemSize = size; this.count = this.array.length / size; this.needsUpdate = false; } },
  Int32BufferAttribute: class { constructor(arr, size) { this.array = arr; this.itemSize = size; } },
  TextureLoader: class { load(url, onLoad, onProg, onErr) { setTimeout(() => onLoad({ dispose: () => {}, colorSpace: null, anisotropy: 1, flipY: true, wrapS: 0, wrapT: 0, image: { width: 64, height: 64 } }), 0); } },
  Texture: class { constructor() { this.image = null; } dispose() {} },
  CanvasTexture: class { constructor(c) { this.image = c; } dispose() {} },
  Raycaster: class { constructor() {} setFromCamera() {} intersectObjects() { return []; } },
  Vector2: class { constructor(x, y) { this.x = x || 0; this.y = y || 0; } set() {} },
  Matrix4: class { constructor() {} makeScale() { return this; } compose() { return this; } makeRotationFromEuler() { return this; } makeRotationY() { return this; } setPosition() { return this; } multiply() { return this; } makeTranslation() { return this; } premultiply() { return this; } copy() { return this; } identity() { return this; } extractBasis() { return this; } },
  MeshStandardMaterial: class { constructor(opts) { Object.assign(this, opts); this.map = null; } dispose() {} },
  MeshBasicMaterial: class { constructor(opts) { Object.assign(this, opts); if (this.map === undefined) this.map = null; } dispose() {} },
  MeshLambertMaterial: class { constructor(opts) { Object.assign(this, opts); if (this.map === undefined) this.map = null; } dispose() {} },
  LineBasicMaterial: class { constructor(opts) { Object.assign(this, opts); } dispose() {} },
  LineSegments: class { constructor(g, m) { this.geometry = g; this.material = m; } },
  PerspectiveCamera: class { constructor() { this.position = { set: () => {}, copy: () => {} }; this.aspect = 1; this.matrixWorld = { extractBasis: () => {} }; } updateProjectionMatrix() {} lookAt() {} },
  WebGLRenderer: class { constructor() { this.domElement = window.document.createElement('canvas'); } setPixelRatio() {} setSize() {} render() {} dispose() {} setClearColor() {} setRenderTarget() {} },
  Scene: class { constructor() { this.background = null; this.children = []; } add(c) { this.children.push(c); } remove(c) { this.children = this.children.filter(x => x !== c); } traverse(fn) { fn(this); for (const c of this.children) if (c.traverse) c.traverse(fn); else fn(c); } },
  HemisphereLight: class {},
  DirectionalLight: class { constructor() { this.position = { set: () => {} }; } },
  Color: class { constructor(c) { this.r = 1; this.g = 1; this.b = 1; } setHSL() { return this; } },
  Vector3: class { constructor(x, y, z) { this.x = x || 0; this.y = y || 0; this.z = z || 0; } set() {} copy() { return this; } normalize() { return this; } subVectors() { return this; } cross() { return this; } applyMatrix4() { return this; } add() { return this; } sub() { return this; } divideScalar() { return this; } multiplyScalar() { return this; } addScaledVector() { return this; } length() { return 1; } },
  Box3: class { constructor() {} setFromObject() { return this; } getCenter(v) { return v; } getSize(v) { return v; } expandByPoint() {} },
  Sphere: class { constructor() { this.center = { x: 0, y: 0, z: 0 }; this.radius = 1; } setFromCenterAndRadius() {} },
  BoxGeometry: class { constructor() {} },
  PlaneGeometry: class { constructor() {} },
  CylinderGeometry: class { constructor() {} },
  SphereGeometry: class { constructor() { this.dispose = () => {}; this.clone = () => new THREE.SphereGeometry(); } },
  DoubleSide: 0,
  FrontSide: 0,
  ClampToEdgeWrapping: 0,
  RepeatWrapping: 0,
  MirroredRepeatWrapping: 0,
  SRGBColorSpace: 0,
  LinearSRGBColorSpace: 0,
};

// Inject globals into jsdom window
window.THREE = THREE;
// Wrap fetch so relative URLs get prefixed with the running server.
window.fetch = (url, opts) => {
  const u = (typeof url === 'string' && url.startsWith('/'))
    ? "http://127.0.0.1:8765" + url : url;
  return fetch(u, opts);
};
window.console = console;

// Stub things model_viewer references
window.psoEditor = { state: { tileEdits: {} } };

// Load model_viewer.js into the jsdom context
const code = readFileSync(resolve(REPO, 'static/model_viewer.js'), 'utf8');
// Strip the import statement at top
const stripped = code.replace(/^import \* as THREE.*$/m, '// THREE injected');

// Run code inside the window context
const fn = new window.Function(stripped);
try {
  fn.call(window);
} catch (e) {
  console.error("Error loading model_viewer.js:");
  console.error(e.stack || e);
  process.exit(2);
}

console.log("psoOpenSkinnedModel:", typeof window.psoOpenSkinnedModel);
console.log("Before load: psoGetAnimationPlaying =", window.psoGetAnimationPlaying());
console.log("Before load: psoGetCurrentMotion =", window.psoGetCurrentMotion());

// Test 1: with dropdown present
const modelPath = "bm_boss8_dragon.bml#boss1_s_nb_dragon.nj";
console.log("\nCalling psoOpenSkinnedModel (dropdown present)...");
try {
  await window.psoOpenSkinnedModel(modelPath);
} catch (e) {
  console.error("psoOpenSkinnedModel threw:", e.message || e);
  console.error(e.stack);
  process.exit(3);
}

// Wait for populateAnimationPanel + loadMotion to finish.
await new Promise(r => setTimeout(r, 2000));

console.log("\nAfter load:");
console.log("  motions count =", window.psoListMotions().length);
console.log("  current motion =", window.psoGetCurrentMotion());
console.log("  playing =", window.psoGetAnimationPlaying());

const playing = window.psoGetAnimationPlaying();
const motion = window.psoGetCurrentMotion();

if (!(motion && motion.toLowerCase().includes("walk") && playing === true)) {
  console.error("\n✗ FAIL: expected walk motion auto-playing");
  console.error("  motion =", motion);
  console.error("  playing =", playing);
  process.exit(1);
}
console.log("\n✓ Test 1 PASS: dragon auto-plays walk on load (with dropdown)");

// Test 2: simulate a perspective that removes the dropdown but still has
// the modal — the auto-play MUST still fire (regression guard).
console.log("\nTest 2: with dropdown removed (unified-viewport-style hide)...");
const sel = window.document.getElementById('modelAnimSel');
if (sel && sel.parentNode) sel.parentNode.removeChild(sel);
// Reset state.
const motionsBefore = window.psoListMotions();
console.log("  motions before reload:", motionsBefore.length);

// Trigger a fresh load
await window.psoOpenSkinnedModel("bm_boss2_de_rol_le.bml#boss2_b_derorure_body.nj");
await new Promise(r => setTimeout(r, 2000));

const m2 = window.psoGetCurrentMotion();
const p2 = window.psoGetAnimationPlaying();
console.log("  current motion =", m2);
console.log("  playing =", p2);
if (!(m2 && p2 === true)) {
  console.error("\n✗ FAIL: auto-play broken when dropdown is missing");
  process.exit(1);
}
console.log("\n✓ Test 2 PASS: auto-play works without dropdown");

// Test 3: bm4_ps_ma_body (the multi-form Pan Arms path).
//
// Pre-2026-04-26 the picker was verb-only and landed on
// ``move_bm4_ps_mb_body.njm`` — a motion authored for the 1-bone
// ``mb_body`` SUB-FORM, which snapped the loaded 43-bone ``ma_body``
// rig to bind pose. The four-tier resolver in
// ``formats.motion_pairing`` now detects the stem mismatch and
// demotes ``move_bm4_ps_mb_body`` to Tier 3, surfacing the matching
// ``wait2_bm4_ps_ma_body`` (43 bones, same stem) at index 0.
//
// What we actually assert: the picked motion's NJM filename includes
// the loaded inner-stem (``ma_body``), not just any ``move_*`` track.
// A ``wait*`` for the same stem is a correct outcome — the failure
// mode this test guards against is "default-pick targets the wrong
// rig", not "default-pick must contain the keyword move".
console.log("\nTest 3: bm4_ps_ma_body should auto-play a same-stem motion...");
await window.psoOpenSkinnedModel("bm4_ps_ma_body.bml#bm4_ps_ma_body.nj");
await new Promise(r => setTimeout(r, 2000));
const m3 = window.psoGetCurrentMotion();
const p3 = window.psoGetAnimationPlaying();
console.log("  current motion =", m3);
console.log("  playing =", p3);
if (!(m3 && m3.toLowerCase().includes("ma_body") && p3 === true)) {
  console.error("\n✗ FAIL: bm4 auto-play missed (wrong stem or paused)");
  console.error("  motion =", m3, "playing =", p3);
  process.exit(1);
}
console.log("\n✓ Test 3 PASS: bm4_ps_ma_body auto-plays a stem-matched motion");

// Test 4: mericarol — picker should fall through to wait (no walk/run/move).
console.log("\nTest 4: bm_ene_bm9_s_mericarol...");
await window.psoOpenSkinnedModel("bm_ene_bm9_s_mericarol.bml#bm9_s_meri_body.nj");
await new Promise(r => setTimeout(r, 2000));
const m4 = window.psoGetCurrentMotion();
const p4 = window.psoGetAnimationPlaying();
console.log("  current motion =", m4);
console.log("  playing =", p4);
if (!(m4 && m4.toLowerCase().includes("wait") && p4 === true)) {
  console.error("\n✗ FAIL: mericarol auto-play missed");
  console.error("  motion =", m4, "playing =", p4);
  process.exit(1);
}
console.log("\n✓ Test 4 PASS: mericarol auto-plays its default motion (wait)");

// Test 5: scrubber drag preserves v4's "stay paused on release" semantics.
console.log("\nTest 5: scrubber drag pauses + stays paused (v4 motion editor)...");
// First load a model so we have a motion playing
await window.psoOpenSkinnedModel("bm_boss8_dragon.bml#boss1_s_nb_dragon.nj");
await new Promise(r => setTimeout(r, 2000));
console.log("  pre-drag playing =", window.psoGetAnimationPlaying());
if (!window.psoGetAnimationPlaying()) {
  console.error("\n✗ FAIL: dragon should be playing before drag");
  process.exit(1);
}
// Simulate a user drag via psoSeekAnimationToFrame (the v3 hook used by anim_editor_panel scrubber)
window.psoSeekAnimationToFrame(10);
console.log("  post-seek playing =", window.psoGetAnimationPlaying());
if (window.psoGetAnimationPlaying() !== false) {
  console.error("\n✗ FAIL: psoSeekAnimationToFrame should pause playback (v4 semantics)");
  process.exit(1);
}
console.log("\n✓ Test 5 PASS: scrubber-driven seek pauses (v4 semantics preserved)");

console.log("\nAll tests passed.");
process.exit(0);
