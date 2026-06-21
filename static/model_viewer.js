// =====================================================================
// PSOBB Texture Editor - 3D Model Preview
//
// Renders the live texture either on a real PSOBB XJ-format mesh
// (when the editor's `/api/model_mesh/` endpoint succeeds) or, as a
// graceful fallback, on a 3D primitive (sphere/cube/plane/cylinder).
//
// Real-mesh path:
//   1. POST `/api/model_preview/{filename}` -> hint with `model_archive`.
//   2. If a paired `.nj` exists at the same name (or a paired BML can
//      be unpacked), GET `/api/model_mesh/{path}?inner=...` and decode
//      the b64 vertex/index buffers into THREE.BufferGeometry.
//   3. Wrap the geometry with MeshBasicMaterial (textured) or
//      MeshLambertMaterial (un-textured) — Phantasmal-diff conventions.
// Fallback: same primitive path the editor has shipped since v1.
//
// Live texture source: /api/tile_png/<filename>/<idx> (already exposed
// by the editor). Live updates are achieved by re-fetching the tile PNG;
// the existing upscale/repack workflow has no direct hook here, but the
// user can press "refresh texture" any time.
// =====================================================================

import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";

// psov2 faithful Ninja loader (the KNOWN-GOOD reference renderer). For
// `.nj` / `.bml#inner.nj` models we route through this client-side parser
// (raw inner NJ bytes from /api/raw_nj -> parseNinjaModel -> SkinnedMesh)
// INSTEAD of the diverged server-side reconstruction. The old skinned /
// world-baked paths remain as fallbacks for non-.nj assets and on error.
import { parseNinjaModel, BitStream as _NinjaBitStream } from "/static/psov2_ninja.js";

const $ = (s) => document.querySelector(s);

// Wave 7 (2026-04-26): asset-lifecycle-aware fetch.
//
// Every model-load fetch should ride the shared AbortController in
// window.psoAssetLifecycle so a NEW asset click cancels the OLD asset's
// in-flight request stack. Falls back to plain fetch() if the lifecycle
// module hasn't loaded (e.g. during a unit-test harness).
function _lifecycleFetch(url, init) {
  if (window.psoAssetLifecycle && typeof window.psoAssetLifecycle.fetchAsset === "function") {
    return window.psoAssetLifecycle.fetchAsset(url, init);
  }
  return fetch(url, init);
}

function _isAbortError(e) {
  if (window.psoAssetLifecycle && typeof window.psoAssetLifecycle.isAbort === "function") {
    return window.psoAssetLifecycle.isAbort(e);
  }
  return e && e.name === "AbortError";
}

// fix/perf — epoch generation guard.
//
// The lifecycle's beginAsset() bumps an epoch + aborts the prior fetch
// controller on every model open. But abort only cancels in-flight
// fetches; a parse/build that ALREADY resolved still runs to completion
// and would commit its (now stale) mesh to the scene. On a rapid
// A->B->C switch that produces the documented flicker/clobber — A and B
// both paint before C, and a slow A can even land LAST and clobber C.
//
// `_currentEpoch()` snapshots the active generation at the START of an
// open; `_epochStale(snapshot)` returns true once a NEWER open has begun.
// Every long async model path checks this just before it mutates the
// shared scene (disposeMesh + scene.add) and bails if stale, so only the
// most-recent open ever commits.
function _currentEpoch() {
  if (window.psoAssetLifecycle && typeof window.psoAssetLifecycle.epoch === "function") {
    return window.psoAssetLifecycle.epoch();
  }
  return 0;
}

function _epochStale(snapshot) {
  // No lifecycle module (unit-test harness) → never stale.
  if (!(window.psoAssetLifecycle && typeof window.psoAssetLifecycle.epoch === "function")) {
    return false;
  }
  return _currentEpoch() !== snapshot;
}

const state = {
  filename: null,           // currently-previewed file
  tileCount: 0,
  selectedTileIdx: 0,
  shape: "sphere",
  useUpscaled: true,
  autoRotate: true,
  wireframe: false,
  // three.js
  renderer: null,
  scene: null,
  camera: null,
  mesh: null,                // active mesh group (real OR primitive)
  texture: null,
  rafId: null,
  // On-demand render scheme (2026-06-20). The viewer no longer runs an
  // unconditional 60fps loop. `rafId` is the CONTINUOUS-loop handle (set
  // only while shouldAnimateContinuously() holds — skinned playback,
  // autorotate, or an active drag). `renderPending` is the ONE-SHOT
  // handle used by requestRender() to coalesce N view-mutating events in
  // a frame into a single paint, then idle at 0 renders/sec.
  renderPending: null,
  loopReason: "",            // debug/introspection: "" | "anim" | "autorotate" | "drag"
  resizeObs: null,
  // mesh-mode tracking
  realMesh: false,           // true when displaying loaded XJ geometry
  meshGroup: null,           // THREE.Group holding the loaded sub-meshes
  // Bug-fix bookkeeping (2026-04-24): record WHY a fallback ran so
  // openByPath can render the user-facing "primitive (cube) — model
  // unavailable: <reason>" banner instead of silently swapping in a
  // primitive. realMeshArchive carries the archive label that the last
  // successful mesh load came from (used for diagnostic display).
  lastMeshFailure: null,
  realMeshArchive: null,
  // Per-submesh texture binding (2026-04-25): when /api/model_mesh
  // returns a non-empty `binding` array, we fetch one texture per unique
  // tile_index and stash them here keyed by material_id. Each
  // THREE.Mesh in `meshGroup` receives `material.map = boundTextures.get(mid)`.
  // Pairs with `boundTextureArchive` — the BML#inner.xvm path that
  // backed the textures, used to drive the tile_png URLs.
  boundTextures: new Map(),     // Map<material_id, THREE.Texture>
  boundTextureArchive: null,    // string path "<bml>#<inner>.xvm" or null
  boundBinding: [],             // raw payload.binding array (for diagnostics)
  // Multi-inner BML support (2026-04-25): when the user opens a top-
  // level `.bml` whose archive contains MORE than one viable .nj inner
  // (e.g. a boss BML with body + helm + fins + sting + tentacle), we
  // surface an inner-picker dropdown so the user can switch between
  // parts or view ALL parts composited together. Without this, the
  // viewer was only showing the FIRST or matched-texture-derived inner,
  // so multi-part bosses (De Rol Le, etc.) appeared "missing" sibling
  // parts. See `populateInnerPicker` for the discovery + filter logic.
  bmlInnersInfo: null,          // {base, inners: [{name, kind, primary}], current: <name>|"__all__"}
  // pointer-drag rotation
  drag: { active: false, x: 0, y: 0, rotX: 0, rotY: 0 },
  // Debug-overlay mode (2026-04-24, AGENT_MODEL_DEEP_DEBUG_REPORT).
  // When debugMode=true, each loaded sub-mesh gets a unique HSL-tinted
  // material clone and the sidebar lists per-mesh stats. The original
  // (textured, untinted) materials are preserved on each Mesh's
  // userData.origMaterial so that turning debug OFF restores them
  // without rebuilding the BufferGeometry. We also raycast on
  // pointermove so the sidebar can highlight the closest sub-mesh
  // under the cursor.
  debugMode: false,
  debugMeshes: [],          // [{mesh, idx, payload}] for the active model
  debugActiveIdx: -1,        // index of currently-highlighted submesh, -1 = none
  debugRaycaster: null,      // lazily-created THREE.Raycaster
  debugFilter: "",          // active filter substring for sidebar rows

  // -------------------------------------------------------------------
  // Skeletal animation (2026-04-24).
  //
  // When a `.nj` model loads via the SKINNED path
  // (/api/model_skinned), the parser supplies vertices in BONE-LOCAL
  // coordinates plus a per-vertex bone_idx. The render loop then
  // re-bakes vertices into world space each frame using bone matrices
  // computed from the skeleton + animated NJM keyframes.
  //
  // The cost: O(verts × 1 matrix multiply per frame), about 9700
  // multiplies for a dragon mesh × 30 fps = 290k mat4×vec3 / sec —
  // comfortably under the WebGL render budget on a modern laptop.
  //
  // anim.skinned : true when the active mesh is using the bone-local
  //                pipeline (vertices need per-frame re-bake). False
  //                when the active mesh is world-baked / primitive.
  // anim.modelPath : the path used for /api/animations (kept so we can
  //                  re-resolve animation_data when motion changes).
  // anim.bones : skeleton from /api/model_skinned, DFS-ordered.
  // anim.bindWorld : pre-computed bind-pose world matrices per bone.
  //                  Identity-relative; we apply animated TRS offsets
  //                  per frame to derive each bone's animated world
  //                  matrix. Stored as Float32Array(boneCount * 16).
  // anim.skinSubmeshes : [{geometry, bonePositions, vertBoneIdx,
  //                       baseUVs, baseNormals}] one per submesh, holds
  //                      the data needed to re-bake each frame.
  // anim.motions : motion list from /api/animations (for the dropdown).
  // anim.currentMotion : the motion currently in playback (object from
  //                      anim.motions, or null if "bind pose").
  // anim.currentData : keyframe payload from /api/animation_data for
  //                    the selected motion.
  // anim.time : current playback time in seconds. Frame index =
  //             time * fps mod frame_count.
  // anim.fps : playback frame rate (user-controlled, default = motion's
  //            natural fps).
  // anim.playing : true when the render loop should advance time.
  // anim.loop : when true, time wraps; when false, time clamps at end.
  // anim.lastTimestamp : ms timestamp of the prior render-loop frame
  //                      (for delta-time computation).
  anim: {
    skinned: false,
    modelPath: null,
    bones: [],
    bindWorld: null,
    skinSubmeshes: [],
    motions: [],
    currentMotion: null,
    currentData: null,
    time: 0.0,
    fps: 30.0,
    playing: false,
    loop: true,
    lastTimestamp: 0,
    // psov2 native-motion playback (parallel to the server-payload CPU
    // bake above). When a model loads via tryLoadPsov2NinjaModel and has
    // native .njm motions, psov2_ninja builds real THREE.AnimationClips on
    // mesh.geometry.animations; we drive them with a THREE.AnimationMixer
    // (GPU skinning re-poses the SkinnedMesh) ticked in the render loop.
    // These fields are INERT for the server-payload skinned path.
    psov2: false,        // true when the psov2 mixer is the active driver
    psov2Mesh: null,     // the live THREE.SkinnedMesh (skeleton + groups source)
    psov2Mixer: null,    // THREE.AnimationMixer bound to the SkinnedMesh
    psov2Clips: null,    // Map<motionName, THREE.AnimationClip>
    psov2Action: null,   // currently-playing THREE.AnimationAction (or null)
    psov2Clock: 0,       // last render-loop timestamp (ms) for mixer dt
  },
};

// ---- helpers --------------------------------------------------------

function ensureRenderer() {
  if (state.renderer) return;
  const canvas = $("#modelCanvas");
  state.renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: true,
  });
  state.renderer.setPixelRatio(window.devicePixelRatio || 1);
  resizeRenderer();

  state.scene = new THREE.Scene();
  state.scene.background = new THREE.Color(0x0a0e13);

  state.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
  state.camera.position.set(0, 0, 3.2);

  // Lighting: a hemisphere for ambient + a key light from camera direction
  const hemi = new THREE.HemisphereLight(0xffffff, 0x222233, 0.6);
  state.scene.add(hemi);
  const key = new THREE.DirectionalLight(0xffffff, 0.85);
  key.position.set(2, 3, 4);
  state.scene.add(key);

  // Pointer-drag rotation support (works well even without OrbitControls)
  const cv = canvas;
  cv.addEventListener("pointerdown", (e) => {
    // Sculpt-mode (2026-04-25): when the Sculpt tab has enabled
    // click-to-sculpt, LEFT-button is the brush — we don't want to
    // also start an orbit drag. RIGHT-button stays orbit. The flag
    // is set by `psoSetSculptModeActive` from sculpt_panel.js.
    if (window.__psoSculptModeActive && e.button === 0) return;
    // Rig-mode (2026-04-25): mirror of the sculpt gate; LMB is owned
    // by either bone-drag (Skeleton mode) or weight-paint stroke
    // (Weight Paint mode). Set by `psoSetRigModeActive` from
    // rig_panel.js.
    if (window.__psoRigModeActive && e.button === 0) return;
    // Edit-mode (2026-04-26): edit_panel.js owns LMB for vertex
    // selection / box-select / gizmo manipulation. Same yield pattern.
    if (window.__psoEditModeActive && e.button === 0) return;
    state.drag.active = true;
    state.drag.x = e.clientX;
    state.drag.y = e.clientY;
    state.drag.rotX = state.mesh ? state.mesh.rotation.x : 0;
    state.drag.rotY = state.mesh ? state.mesh.rotation.y : 0;
    cv.setPointerCapture(e.pointerId);
    // drag.active is in shouldAnimateContinuously() — run the continuous
    // loop for the whole gesture so per-pointer-event scheduling jitter
    // never drops a frame.
    startLoop();
  });
  cv.addEventListener("pointermove", (e) => {
    if (!state.drag.active || !state.mesh) return;
    const dx = (e.clientX - state.drag.x) / 200;
    const dy = (e.clientY - state.drag.y) / 200;
    state.mesh.rotation.y = state.drag.rotY + dx;
    state.mesh.rotation.x = state.drag.rotX + dy;
    requestRender();
  });
  const release = (e) => {
    state.drag.active = false;
    try { cv.releasePointerCapture(e.pointerId); } catch {}
    // Paint the final pose; the continuous loop self-stops on its next
    // tick now that drag.active is false (unless anim/autorotate keep it
    // alive, in which case requestRender is a no-op).
    requestRender();
  };
  cv.addEventListener("pointerup", release);
  cv.addEventListener("pointercancel", release);
  cv.addEventListener("wheel", (e) => {
    if (!state.camera) return;
    e.preventDefault();
    state.camera.position.z = Math.max(
      1.4,
      Math.min(8, state.camera.position.z + (e.deltaY > 0 ? 0.2 : -0.2)),
    );
    requestRender();
  }, { passive: false });

  // Resize watcher
  state.resizeObs = new ResizeObserver(resizeRenderer);
  state.resizeObs.observe(canvas.parentElement);
}

function resizeRenderer() {
  const cv = $("#modelCanvas");
  if (!cv || !state.renderer) return;
  const stage = cv.parentElement;
  const w = Math.max(2, stage.clientWidth | 0);
  const h = Math.max(2, stage.clientHeight | 0);
  state.renderer.setSize(w, h, false);
  if (state.camera) {
    state.camera.aspect = w / h;
    state.camera.updateProjectionMatrix();
  }
  // Repaint at the new size (covers ResizeObserver + window 'resize').
  requestRender();
}

function disposeMesh() {
  if (state.mesh) {
    state.scene.remove(state.mesh);
    // For a Group (real mesh path) walk children explicitly.
    if (state.mesh.isGroup) {
      state.mesh.traverse((child) => {
        if (child.isMesh) {
          if (child.geometry) child.geometry.dispose();
          if (child.material) {
            const mats = Array.isArray(child.material) ? child.material : [child.material];
            for (const m of mats) m.dispose();
          }
        }
      });
    } else {
      if (state.mesh.geometry) state.mesh.geometry.dispose();
      if (state.mesh.material) {
        const mats = Array.isArray(state.mesh.material) ? state.mesh.material : [state.mesh.material];
        for (const m of mats) m.dispose();
      }
    }
    state.mesh = null;
  }
  // Free per-submesh bound textures. Each THREE.Texture owns a GPU
  // resource so we MUST dispose them or the tab leaks one texture per
  // model open. The single-tile `state.texture` is managed separately
  // (it can outlive a mesh swap if the user only changed shapes).
  //
  // fix/perf — textures held by the parsed-model LRU cache are owned by
  // the cache (so a re-open stays instant + textured); they must survive
  // this swap. The cache disposes them itself on LRU eviction. Only
  // dispose textures that NO live cache entry references.
  for (const tex of state.boundTextures.values()) {
    if (_psov2TextureIsCached(tex)) continue;
    try { tex.dispose(); } catch {}
  }
  state.boundTextures = new Map();
  state.boundTextureArchive = null;
  state.boundBinding = [];
  state.realMesh = false;
  state.meshGroup = null;
  // Debug-mode bookkeeping: every entry referenced a now-disposed Mesh
  // so the materials its `mesh.userData.origMaterial` pointed at are
  // gone too. Reset the table; the next buildMeshGroupFromPayload call
  // repopulates it.
  //
  // fix/tooltabs — the psov2 path builds INSPECTOR-ONLY view meshes (one
  // per submesh, NOT added to the scene) whose sliced BufferGeometry the
  // scene-traversal dispose above never reaches. Dispose those geometries
  // explicitly so a model swap doesn't leak one GPU buffer per submesh.
  for (const e of state.debugMeshes) {
    if (e && e.mesh && e.mesh.userData && e.mesh.userData.psov2View && e.mesh.geometry) {
      try { e.mesh.geometry.dispose(); } catch {}
    }
  }
  state.debugMeshes = [];
  state.debugActiveIdx = -1;
  // Animation state cleanup (2026-04-24). The Float32Array snapshots of
  // bone-local positions held by anim.skinSubmeshes are released along
  // with the disposed BufferGeometry; we just clear our references.
  _tearDownPsov2Mixer();
  state.anim.skinned = false;
  state.anim.modelPath = null;
  state.anim.bones = [];
  state.anim.skinSubmeshes = [];
  state.anim.motions = [];
  state.anim.currentMotion = null;
  state.anim.currentData = null;
  state.anim.time = 0.0;
  state.anim.playing = false;
  state.anim.lastTimestamp = 0;
}

// Release the psov2 AnimationMixer + clip references on a model swap.
// The mixer holds no GPU resources itself (the SkinnedMesh geometry it
// drives is disposed by disposeMesh above); we just stop the action and
// drop references so the next model starts clean.
function _tearDownPsov2Mixer() {
  const a = state.anim;
  if (a.psov2Action) {
    try { a.psov2Action.stop(); } catch {}
  }
  if (a.psov2Mixer) {
    try { a.psov2Mixer.stopAllAction(); } catch {}
    try { a.psov2Mixer.uncacheRoot(a.psov2Mixer.getRoot()); } catch {}
  }
  a.psov2 = false;
  a.psov2Mesh = null;
  a.psov2Mixer = null;
  a.psov2Clips = null;
  a.psov2Action = null;
  a.psov2Clock = 0;
}

function buildGeometry(shape) {
  switch (shape) {
    case "cube":     return new THREE.BoxGeometry(1.4, 1.4, 1.4);
    case "plane":    return new THREE.PlaneGeometry(2.0, 2.0, 1, 1);
    case "cylinder": return new THREE.CylinderGeometry(0.7, 0.7, 1.6, 48, 1, false);
    case "sphere":
    default:         return new THREE.SphereGeometry(1.0, 64, 32);
  }
}

function rebuildMeshNow() {
  ensureRenderer();
  disposeMesh();

  const geo = buildGeometry(state.shape);
  // Phantasmal-diff fix 4 (2026-04-25): MeshBasicMaterial for textured
  // primitives, MeshLambertMaterial for un-textured. Phantasmal's
  // MeshRenderer.kt drops PBR — PSOBB does no BRDF on diffuse-mapped
  // pixels, and Lambert is the closest to Sega's per-vertex T&L for
  // un-textured submeshes. Saves the PBR shader compile + per-frame
  // GPU cost (Dragon = 1069 textured submeshes × PBR was the hot path).
  // Phantasmal-diff fix 3: default transparent=false; the material
  // panel (window.psoUpdateMaterial) flips it to true on demand when
  // the user enables alpha_test/alpha_blend or sets opacity < 1.
  const mat = state.texture
    ? new THREE.MeshBasicMaterial({
        map: state.texture,
        color: 0xffffff,
        wireframe: state.wireframe,
        side: state.shape === "plane" ? THREE.DoubleSide : THREE.FrontSide,
        transparent: false,
      })
    : new THREE.MeshLambertMaterial({
        color: 0xffffff,
        wireframe: state.wireframe,
        side: state.shape === "plane" ? THREE.DoubleSide : THREE.FrontSide,
        transparent: false,
      });
  state.mesh = new THREE.Mesh(geo, mat);
  state.scene.add(state.mesh);
  kick();
}

// Phantasmal-diff fix 5 (2026-04-25): trailing-edge throttle. Coalesce
// rapid rebuild calls (variant switch + paint stroke + motion change all
// fire close in time) into a single rebuild within a 10 ms window. Match
// Phantasmal's Throttle(wait=10, leading=false, trailing=true) on its
// rebuildMesh path. Callers route through scheduleRebuild() which keeps
// the most-recent build winning.
let _rebuildScheduled = null;
function scheduleRebuild() {
  if (_rebuildScheduled) clearTimeout(_rebuildScheduled);
  _rebuildScheduled = setTimeout(() => {
    _rebuildScheduled = null;
    rebuildMeshNow();
  }, 10);
}

// Back-compat alias. The first-load path (openByPath) needs the
// immediate-fire variant so the modal appears with a real mesh, not
// a 10 ms blank canvas; that path explicitly calls rebuildMeshNow().
// The interactive-edit paths (shape select, paint complete, motion
// change) route through scheduleRebuild() to coalesce.
function rebuildMesh() {
  rebuildMeshNow();
}

// Is there anything that needs a CONTINUOUS (per-frame) animator right
// now? Single source of truth shared by startLoop() (self-terminates
// when this goes false) and kick() (decides loop-vs-one-shot).
//   - skinned playback (a motion is actively advancing time)
//   - autorotate (model mode only; scene mode never autorotates —
//     state.sceneRoot is set in scene mode)
//   - an active orbit drag (smooth per-frame paint during the gesture)
function shouldAnimateContinuously() {
  const a = state.anim;
  if (a && a.psov2 && a.playing && a.psov2Action) { state.loopReason = "anim-psov2"; return true; }
  if (a && a.skinned && a.playing && a.currentData) { state.loopReason = "anim"; return true; }
  if (state.autoRotate && state.mesh && !state.sceneRoot) { state.loopReason = "autorotate"; return true; }
  if (state.drag && state.drag.active) { state.loopReason = "drag"; return true; }
  state.loopReason = "";
  return false;
}

// Schedule EXACTLY ONE render on the next animation frame. Coalescing +
// idempotent: multiple calls in the same frame fold into one paint, and
// it is a no-op while the continuous loop is already painting. This is
// the on-demand path — every view-mutating handler routes through it so
// a static model idles at ~0 renders/sec.
function requestRender() {
  if (!state.renderer || !state.scene || !state.camera) return;
  if (state.rafId) return;            // continuous loop already paints this frame
  if (state.renderPending) return;    // already scheduled this frame
  state.renderPending = requestAnimationFrame(() => {
    state.renderPending = null;
    // Run one anim tick so a paused-but-scrubbed skinned mesh re-bakes.
    if (typeof tickAnimation === "function") {
      tickAnimation(performance.now());
    }
    // Wrapped render (preserves the exact-Lambert uniform sync installed
    // on state.renderer.render).
    state.renderer.render(state.scene, state.camera);
  });
}

// Ensure the CONTINUOUS loop is running. Self-terminating: the first
// tick where shouldAnimateContinuously() returns false sets rafId=null
// and returns (that tick already painted the final frame).
function startLoop() {
  if (state.rafId) return;
  // Fold any pending one-shot into the continuous loop.
  if (state.renderPending) {
    cancelAnimationFrame(state.renderPending);
    state.renderPending = null;
  }
  const tick = (nowMs) => {
    if (!shouldAnimateContinuously()) { state.rafId = null; return; }
    state.rafId = requestAnimationFrame(tick);
    if (state.autoRotate && state.mesh && !state.drag.active && !state.sceneRoot) {
      state.mesh.rotation.y += 0.008;
    }
    // Skeletal animation tick (no-op when state.anim.skinned is false
    // or no motion is loaded). Defined at the end of this file.
    if (typeof tickAnimation === "function") {
      tickAnimation(nowMs || performance.now());
    }
    state.renderer.render(state.scene, state.camera);
  };
  state.rafId = requestAnimationFrame(tick);
}

function stopLoop() {
  if (state.rafId) {
    cancelAnimationFrame(state.rafId);
    state.rafId = null;
  }
}

// Start the continuous loop IF something needs it, otherwise paint a
// single on-demand frame. Replaces the old unconditional startLoop()
// calls at mesh-build sites: a static (non-playing, autorotate-off)
// model paints once then idles; a skinned/autorotating model starts the
// loop.
function kick() {
  if (shouldAnimateContinuously()) startLoop();
  else requestRender();
}

// ---- texture loading -----------------------------------------------

function tileEditPngB64() {
  // Hook: read the global app.js state.tileEdits to get an upscaled PNG
  // for the currently-selected tile. This avoids an extra round-trip
  // and means the preview reflects in-flight edits even before repack.
  try {
    if (!state.useUpscaled) return null;
    const ed = window.psoEditor && window.psoEditor.state && window.psoEditor.state.tileEdits;
    if (!ed) return null;
    const file = state.filename;
    const idx = state.selectedTileIdx;
    if (file == null || idx == null) return null;
    // app.js uses key format `${filename}:${index}` (see editKey() in app.js)
    const key = `${file}:${idx}`;
    const edit = ed[key];
    // Edit objects expose .up_b64 (the upscaled PNG, base64) — this is what
    // the user actually edited. We could also fall back to .src_b64 but
    // since the toggle above already handles "use upscaled", null here
    // means "fall back to /api/tile_png".
    if (edit && edit.up_b64) return edit.up_b64;
  } catch {}
  return null;
}

async function loadTexture() {
  const filename = state.filename;
  if (!filename) return;
  const idx = state.selectedTileIdx | 0;

  // Prefer in-memory upscaled PNG (if user has edited this tile)
  let url;
  let label;
  const edit = tileEditPngB64();
  if (edit) {
    url = `data:image/png;base64,${edit}`;
    label = `tile ${idx} (live upscaled)`;
  } else {
    // Fall back to the source PNG endpoint
    url = `/api/tile_png/${encodeURIComponent(filename)}/${idx}?cb=${Date.now()}`;
    label = `tile ${idx} (source)`;
  }
  // If user requested upscaled but none exists, surface that clearly
  if (state.useUpscaled && !edit) {
    setStatus(`no upscaled version yet for tile ${idx} — falling back to source. Upscale it in the main editor first.`);
  } else {
    setStatus(`loading ${label}...`);
  }
  try {
    const tex = await new Promise((resolve, reject) => {
      const loader = new THREE.TextureLoader();
      // No crossOrigin: this is same-origin (the editor served the PNG itself).
      // Setting crossOrigin = "anonymous" on a same-origin request triggers
      // CORS enforcement; if the server omits Access-Control-Allow-Origin
      // (FastAPI does by default) the load fails with [object Event].
      loader.load(
        url,
        resolve,
        undefined,
        // Convert ErrorEvent → readable message instead of "[object Event]"
        (ev) => {
          const target = ev && ev.target ? ev.target : null;
          const src = (target && target.src) ? target.src.replace(location.origin, "") : url;
          const httpHint = (target && target.complete === false) ? "fetch failed" : "decode failed";
          reject(new Error(`${httpHint} (${src})`));
        }
      );
    });
    // Phantasmal-diff fix 1 (2026-04-25): do NOT set
    // colorSpace = SRGBColorSpace. Phantasmal's XvrTextureConversion.kt
    // leaves colorSpace at its Three.js default (NoColorSpace / linear).
    // Setting SRGB here causes a double-gamma path: the GPU samples
    // sRGB→linear, then the framebuffer converts linear→sRGB again,
    // washing out PSOBB textures that were authored in raw byte space.
    // (Was: tex.colorSpace = THREE.SRGBColorSpace.)
    // Anisotropy bumped 4 → 8 (2026-04-30): modern GPUs handle 8x AF
    // for free; 4 was leaving quality on the table. Stays under the
    // typical 16x ceiling on contemporary cards.
    tex.anisotropy = 8;
    // V-flip (2026-06-20): PSOBB UVs are top-down-V (see
    // formats/import_external.py "PSOBB V is top-down"); THREE's
    // TextureLoader defaults flipY=true, which would upload the PNG rows
    // inverted and render every model texture vertically flipped. The
    // known-good Phantasmal reference uses flipY=false with the same raw
    // V. Set it BEFORE first GPU upload (here, before resolve) so no
    // needsUpdate is required.
    tex.flipY = false;
    // Wrap mode (2026-06-20, psov2-grounded): the Sega Ninja default for
    // OBJECT textures is D3DTADDRESS_MIRROR — psov2's NinjaTexture.js sets
    // MirroredRepeat by default (overriding only a named allowlist back to
    // Repeat). This is the object-model viewer: model UVs sit in [0,1], so
    // Mirror is visually identical to Repeat for the common case but matches
    // the reference on the rare edge-tiled object. Tiled FLOOR/room geometry
    // (UVs > 1) is a SEPARATE path (floor/map editor) that psov2 keeps on
    // Repeat via the clampU/clampV flags, so Mirror here never folds a floor.
    // Sphere preview keeps ClampToEdge (avoids the back-seam); the bare
    // primitive fallback keeps plain Repeat.
    if (state.realMesh) {
      tex.wrapS = THREE.MirroredRepeatWrapping;
      tex.wrapT = THREE.MirroredRepeatWrapping;
    } else if (state.shape === "sphere") {
      tex.wrapS = THREE.ClampToEdgeWrapping;
      tex.wrapT = THREE.ClampToEdgeWrapping;
    } else {
      tex.wrapS = THREE.RepeatWrapping;
      tex.wrapT = THREE.RepeatWrapping;
    }
    if (state.texture) state.texture.dispose();
    state.texture = tex;
    applyTextureToCurrentMesh(tex);
    // Paint the new texture on the next frame (on-demand). When the
    // continuous loop is active this is a no-op; otherwise it schedules
    // exactly one frame so the user sees the texture without waiting on a
    // loop that no longer runs while idle.
    requestRender();
    setStatus(`loaded ${label}`);
  } catch (e) {
    setStatus(`texture load failed: ${e && e.message ? e.message : String(e)}`);
  }
}

function applyTextureToCurrentMesh(tex) {
  if (!state.mesh) return;
  if (state.mesh.isGroup) {
    // Real-mesh path. When per-submesh bindings exist, the user-
    // selected single tile (`state.selectedTileIdx`) gets applied
    // ONLY to submeshes whose material_id matches that tile_index.
    // Submeshes with their own pre-fetched bound texture are left
    // alone so editing one slot doesn't accidentally repaint the
    // whole model. Submeshes whose material_id is unmapped (no
    // binding entry) still receive the user-selected tile, matching
    // the spec's "fall back to tile 0" behaviour.
    const selectedIdx = state.selectedTileIdx | 0;
    state.mesh.traverse((c) => {
      if (!c.isMesh || !c.material) return;
      // Composite-mode skip (2026-04-30): a composite mesh has its own
      // per-inner bound texture (resolved at composite-load time via
      // each inner's independent `boundTex` Map). The user's single-tile
      // selector targets a tile in ONE archive, but composite meshes
      // span N inners with N independent archives — applying the same
      // tile blindly across them is the cross-contamination symptom the
      // user reported. Leave composite meshes painted by their per-inner
      // binding; the user can switch to single-inner mode to edit a slot.
      if (c.userData && c.userData.compositeKey) return;
      const matId = (c.userData && (c.userData.materialId | 0));
      const bindingHit = state.boundBinding && state.boundBinding.find((b) => (b.material_id | 0) === matId);
      const hasOwn = state.boundTextures && state.boundTextures.has(matId);
      if (bindingHit && hasOwn) {
        // This submesh has its own binding. Repaint it only when the
        // user's edited tile maps to this material (so the live
        // upscale preview takes effect on the right slot).
        if ((bindingHit.tile_index | 0) === selectedIdx) {
          c.material.map = tex;
          c.material.needsUpdate = true;
        }
      } else {
        // Unbound submesh — the single-tile texture is the only
        // sensible source.
        c.material.map = tex;
        c.material.needsUpdate = true;
      }
    });
  } else if (state.mesh.material) {
    state.mesh.material.map = tex;
    state.mesh.material.needsUpdate = true;
  }
}

function setStatus(s) {
  const el = $("#modelStatus");
  if (el) el.textContent = s;
}

function setMeshStats(text) {
  const el = $("#modelMeshStats");
  if (!el) return;
  if (text) {
    el.textContent = text;
    el.hidden = false;
    el.classList.remove("model-mesh-fallback");
  } else {
    el.textContent = "";
    el.hidden = true;
    el.classList.remove("model-mesh-fallback");
  }
}

// Show the "primitive — fallback" badge in the same overlay slot as
// setMeshStats. This makes it impossible for the user to mistake a
// fallback for a real mesh: every viewport open sets exactly one of
// the two banners.
function setFallbackBanner(shape, reason) {
  const el = $("#modelMeshStats");
  if (!el) return;
  const why = reason ? `model unavailable: ${reason}` : "model unavailable";
  el.textContent = `primitive (${shape || "?"}) — ${why}`;
  el.classList.add("model-mesh-fallback");
  el.hidden = false;
}

// ---- preview-hint backend call -------------------------------------

async function fetchPreviewHint(filename) {
  // Wave 7: route through lifecycle signal so a stale hint fetch dies
  // when the user picks a different asset.
  const f = (window.psoAssetLifecycle && window.psoAssetLifecycle.fetchAsset) || fetch;
  const r = await f(`/api/model_preview/${encodeURIComponent(filename)}`);
  if (!r.ok) throw new Error(`hint http ${r.status}`);
  return r.json();
}

// ---- model bundle cache --------------------------------------------
//
// Coalesces the multiple JSON fetches a cold model open requires
// (skinned, animations, optional default-motion data) into a SINGLE
// /api/model_bundle GET. The bundle endpoint emits an "errors" map
// instead of failing the whole call so a 4xx on one component (rare)
// doesn't sink the user-visible model.
//
// Cache shape: Map<modelPath, Promise<bundle | null>>.  Values are
// promises so concurrent callers (tryLoadSkinnedMesh +
// populateAnimationPanel both fire from openByPath) share one fetch.
// A null payload means the bundle endpoint isn't available — callers
// fall back to the per-endpoint flow. We keep the cache small (one
// entry per recently-opened model) and clear it whenever the user opens
// a different model so stale entries don't pin memory.

const _bundleCache = new Map();
let _lastBundlePath = null;

/**
 * Build the model_bundle URL the same way tryLoadSkinnedMesh /
 * populateAnimationPanel build their per-endpoint URLs. Returns the
 * full URL string + the canonical "modelPath" cache key.
 */
function _bundleUrl(modelPath, includeMotion) {
  const hashIdx = modelPath.indexOf("#");
  let url;
  if (hashIdx > 0) {
    const base = modelPath.slice(0, hashIdx);
    let inner = modelPath.slice(hashIdx + 1);
    if (inner.toLowerCase().endsWith(".xvm")) inner = inner.slice(0, -4);
    url = `/api/model_bundle/${encodeURIComponent(base)}?inner=${encodeURIComponent(inner)}`;
  } else {
    url = `/api/model_bundle/${encodeURIComponent(modelPath)}`;
  }
  if (includeMotion) {
    const sep = url.includes("?") ? "&" : "?";
    url += `${sep}include_motion=${encodeURIComponent(includeMotion)}`;
  }
  return url;
}

/**
 * Pre-fetch the bundle for `modelPath` and cache the promise.  Safe to
 * call before the model viewer modal is even visible — gives the
 * skinned + animations payloads a head start while three.js spins up
 * its renderer.  Returns the bundle promise.
 *
 * Resolves to ``null`` when the endpoint isn't available so callers
 * can transparently fall through to the per-endpoint flow (older
 * server builds, or a future feature flag that disables bundling).
 */
function prefetchModelBundle(modelPath) {
  if (!modelPath) return Promise.resolve(null);
  // Bust any prior in-flight bundle for a different model — keeps
  // memory bounded to one entry without LRU bookkeeping.
  if (_lastBundlePath && _lastBundlePath !== modelPath) {
    _bundleCache.delete(_lastBundlePath);
  }
  _lastBundlePath = modelPath;
  const cached = _bundleCache.get(modelPath);
  if (cached) return cached;
  const url = _bundleUrl(modelPath, "default");
  // Wave 7: the lifecycle signal aborts this fetch if the user picks a
  // different asset before our bundle response lands. fetchAsset()
  // wraps fetch() and merges in window.psoAssetLifecycle.signal()
  // so we don't have to thread the signal through every layer.
  const fetchFn = (window.psoAssetLifecycle && window.psoAssetLifecycle.fetchAsset) || fetch;
  const isAbort = (window.psoAssetLifecycle && window.psoAssetLifecycle.isAbort) || (() => false);
  const p = (async () => {
    try {
      const r = await fetchFn(url, { headers: { Accept: "application/json" } });
      if (r.status === 404) return null;       // endpoint missing → fallback
      if (!r.ok) return null;                  // 4xx/5xx → fallback (don't poison cache)
      const json = await r.json();
      // The bundle's components are the SAME shape /api/model_skinned
      // and /api/animations return on their own, so we can hand them
      // to the existing consumers verbatim.
      return json;
    } catch (e) {
      // AbortError is expected when the user moves on; suppress the
      // log noise so devtools stay readable during rapid-clicks.
      if (isAbort(e)) return null;
      console.warn("[model_viewer] bundle fetch failed; fallback:", e);
      return null;
    }
  })();
  _bundleCache.set(modelPath, p);
  return p;
}

/** Drop a cached bundle (called when we want to force a fresh load). */
function _invalidateBundle(modelPath) {
  if (modelPath) _bundleCache.delete(modelPath);
  if (modelPath === _lastBundlePath) _lastBundlePath = null;
}

// ---- parsed-model LRU cache (fix/perf) -----------------------------
//
// Re-opening a recently-viewed `.nj` model used to re-run the FULL cold
// pipeline: /api/raw_nj -> parse -> texlist fetch -> N njm fetches ->
// re-parse. The network legs dominate (server first-decode is ~1s cold);
// the client parse is cheap (~1-3ms). So we cache the SLOW-to-fetch raw
// inputs (the decoded NJ ArrayBuffer, the resolved THREE.Textures, the
// raw motion buffers + names, and the binding metadata) keyed by the
// RESOLVED model path. On a cache hit we re-parse the bytes (cheap) into
// a FRESH THREE.Group per open — never sharing a Group/geometry across
// scene commits — and re-wire the already-resolved textures.
//
// GPU discipline: the cached THREE.Textures are shared with the live
// mesh while it's on screen. disposeMesh() frees the *materials* but the
// texture objects are owned by this cache, so we DON'T dispose them on a
// normal model swap (that would break the cached entry). Instead the LRU
// disposes a texture's GPU resource ONLY when its entry is EVICTED and is
// not the currently-displayed model. This keeps re-opens instant without
// leaking one GPU texture set per distinct model beyond the LRU bound.
const PSOV2_MODEL_CACHE_MAX = 8;
/** @type {Map<string, {buf:ArrayBuffer, texList:Array, fetched:Array, binding:Array, archive:string, motionBuffers:Array, motionNames:Array, nativeMotions:Object|null, texIds:Array}>} */
const _psov2ModelCache = new Map();

/** Mark `key` most-recently-used (Map preserves insertion order). */
function _psov2CacheTouch(key) {
  const v = _psov2ModelCache.get(key);
  if (v === undefined) return undefined;
  _psov2ModelCache.delete(key);
  _psov2ModelCache.set(key, v);
  return v;
}

/**
 * Insert/replace a parsed-model entry and evict the LRU tail past the
 * bound. Evicted textures get their GPU resource freed UNLESS they're
 * still referenced by the on-screen model (`keepArchive` === the live
 * model's path) — disposing a live texture would blank the viewport.
 */
function _psov2CacheStore(key, entry) {
  if (_psov2ModelCache.has(key)) _psov2ModelCache.delete(key);
  _psov2ModelCache.set(key, entry);
  while (_psov2ModelCache.size > PSOV2_MODEL_CACHE_MAX) {
    const oldestKey = _psov2ModelCache.keys().next().value;
    const old = _psov2ModelCache.get(oldestKey);
    _psov2ModelCache.delete(oldestKey);
    if (old && oldestKey !== state.realMeshArchive) {
      for (const t of (old.fetched || [])) { try { t.dispose(); } catch {} }
    }
  }
}

/** True if a THREE.Texture is referenced by ANY live parsed-model cache
 *  entry. disposeMesh() consults this so it never frees a texture the
 *  cache still owns (which would blank a future re-open). */
function _psov2TextureIsCached(tex) {
  if (!tex) return false;
  for (const v of _psov2ModelCache.values()) {
    const f = v && v.fetched;
    if (f && f.indexOf(tex) !== -1) return true;
  }
  return false;
}

/** Evict cache entries whose key matches `modelPath` exactly or shares
 *  its `.bml` base (so a `<bml>#<inner>` reload busts that inner even when
 *  the caller passed the bare bml). Frees freed-entry GPU textures unless
 *  they back the on-screen model. */
function _psov2CacheEvictForPath(modelPath) {
  if (!modelPath) return;
  const base = modelPath.split("#")[0];
  for (const [k, v] of Array.from(_psov2ModelCache.entries())) {
    if (k === modelPath || k.split("#")[0] === base) {
      _psov2ModelCache.delete(k);
      if (k !== state.realMeshArchive) {
        for (const t of (v.fetched || [])) { try { t.dispose(); } catch {} }
      }
    }
  }
}

/** Forget every cached parsed model (force a fully cold reload). Frees
 *  the GPU textures of every entry that isn't the on-screen model. */
function _psov2CacheClear() {
  for (const [k, v] of _psov2ModelCache.entries()) {
    if (k === state.realMeshArchive) continue;
    for (const t of (v.fetched || [])) { try { t.dispose(); } catch {} }
  }
  _psov2ModelCache.clear();
}

// ---- idle-time model preload (fix/perf) ----------------------------
//
// Warm the parsed-model cache for LIKELY-NEXT models during browser idle
// time so the next open is instant. Discipline:
//   * IDLE: scheduled via requestIdleCallback (falls back to a short
//     setTimeout) so preload never competes with the active open's
//     fetch/parse for the main thread.
//   * BOUNDED: at most ONE preload runs at a time (single-flight) and the
//     pending queue is capped (PSOV2_PRELOAD_QUEUE_MAX). Excess requests
//     are dropped — a preload miss just costs a normal cold open later.
//   * ABORTABLE: each preload rides its OWN AbortController. Starting a
//     real model open (openByPath) aborts any in-flight preload so the
//     user-driven request gets the network immediately.
//   * SKIP-WORK: a model already in the cache (or being preloaded) is
//     skipped; only `.nj` paths are eligible.
// A preload commits NOTHING to the scene — it only populates the cache.
const PSOV2_PRELOAD_QUEUE_MAX = 6;
const _psov2PreloadQueue = [];        // pending model paths (FIFO, deduped)
const _psov2PreloadInflight = new Set(); // paths currently preloading
let _psov2PreloadController = null;   // AbortController for the active preload
let _psov2PreloadActive = false;      // single-flight guard
let _psov2PreloadIdleHandle = null;

function _idle(fn) {
  if (typeof requestIdleCallback === "function") {
    return requestIdleCallback(fn, { timeout: 1500 });
  }
  return setTimeout(fn, 120);
}

/** Cancel any in-flight preload + clear the pending queue. Called when a
 *  real open begins so the user-driven fetch isn't starved. */
function _psov2PreloadAbortAll() {
  if (_psov2PreloadController) {
    try { _psov2PreloadController.abort(); } catch {}
    _psov2PreloadController = null;
  }
  _psov2PreloadQueue.length = 0;
  _psov2PreloadInflight.clear();
  _psov2PreloadActive = false;
}

/** Queue `modelPath` for idle-time preload (deduped, bounded). No-op if
 *  it's already cached/queued/inflight or isn't a `.nj` model. */
function preloadPsov2Model(modelPath) {
  if (!modelPath || typeof modelPath !== "string") return;
  if (!modelPath.toLowerCase().endsWith(".nj")) return;
  if (_psov2ModelCache.has(modelPath)) return;
  if (_psov2PreloadInflight.has(modelPath)) return;
  if (_psov2PreloadQueue.indexOf(modelPath) !== -1) return;
  _psov2PreloadQueue.push(modelPath);
  while (_psov2PreloadQueue.length > PSOV2_PRELOAD_QUEUE_MAX) {
    _psov2PreloadQueue.shift(); // drop oldest — bounded
  }
  if (!_psov2PreloadIdleHandle) {
    _psov2PreloadIdleHandle = _idle(() => {
      _psov2PreloadIdleHandle = null;
      _psov2PreloadPump();
    });
  }
}

/** Drain the preload queue one model at a time, on idle. */
function _psov2PreloadPump() {
  if (_psov2PreloadActive) return;
  const modelPath = _psov2PreloadQueue.shift();
  if (!modelPath) return;
  if (_psov2ModelCache.has(modelPath)) {
    // Already warmed (e.g. user opened it meanwhile) — move on.
    _idle(() => _psov2PreloadPump());
    return;
  }
  _psov2PreloadActive = true;
  _psov2PreloadInflight.add(modelPath);
  _psov2PreloadController = new AbortController();
  const signal = _psov2PreloadController.signal;
  _psov2FetchModelDataForCache(modelPath, signal)
    .then((entry) => {
      if (entry && !signal.aborted && !_psov2ModelCache.has(modelPath)) {
        _psov2CacheStore(modelPath, entry);
      } else if (entry && (signal.aborted || _psov2ModelCache.has(modelPath))) {
        // Discard a result we won't keep; free its GPU textures.
        for (const t of (entry.fetched || [])) { try { t.dispose(); } catch {} }
      }
    })
    .catch(() => { /* preload failures are silent — a cold open will retry */ })
    .finally(() => {
      _psov2PreloadInflight.delete(modelPath);
      _psov2PreloadActive = false;
      if (_psov2PreloadQueue.length) {
        _idle(() => _psov2PreloadPump());
      }
    });
}

/**
 * Fetch + decode everything needed to build a psov2 model, WITHOUT
 * touching the scene. Returns a cache-entry object (same shape
 * _psov2CacheStore expects) or null on failure/abort. Uses an explicit
 * AbortSignal so preload can be cancelled independently of the active
 * open's lifecycle signal.
 */
async function _psov2FetchModelDataForCache(modelPath, signal) {
  // Step 1: raw inner NJ bytes (own signal, NOT the lifecycle signal).
  let buf;
  try {
    const r = await fetch(`/api/raw_nj/${encodeURIComponent(modelPath)}`, { signal });
    if (!r.ok) return null;
    buf = await r.arrayBuffer();
  } catch (_e) {
    return null; // abort or network error
  }
  if (!buf || buf.byteLength < 8 || signal.aborted) return null;

  // Step 2: one header parse to read texIds (no mesh kept).
  let loader;
  try {
    const probe = parseNinjaModel(buf, { name: modelPath, texList: [] });
    loader = probe.userData.ninjaLoader;
    try { probe.geometry.dispose(); } catch {}
  } catch (_e) {
    return null;
  }
  if (!loader || !loader.bones.length) return null;
  const texIds = (loader.matList || []).map((mm) => mm.texId).filter((n) => n >= 0);

  // Step 3: textures + native motions, in parallel, on the preload signal.
  let archive = _psov2TextureArchive(modelPath);
  let texList = [];
  let fetchedTextures = [];
  let psov2Binding = [];
  const texPromise = texIds.length > 0
    ? _psov2BuildTexList(modelPath, texIds, signal).catch(() => null)
    : Promise.resolve(null);
  const motionPromise = _fetchNativeMotions(modelPath, signal).catch(() => null);
  const [built, nativeMotions] = await Promise.all([texPromise, motionPromise]);
  if (signal.aborted) {
    if (built) for (const t of (built.fetched || [])) { try { t.dispose(); } catch {} }
    return null;
  }
  if (built) {
    texList = built.texList;
    fetchedTextures = built.fetched;
    psov2Binding = built.binding || [];
    archive = built.archive || archive;
  }
  const motionBuffers = [];
  const motionNames = [];
  if (nativeMotions && nativeMotions.list && nativeMotions.list.length) {
    for (const m of nativeMotions.list) {
      const b2 = nativeMotions.buffers.get(m.name);
      if (b2 && b2.byteLength > 8) {
        motionBuffers.push(b2);
        motionNames.push(`${m.name}.njm`);
      }
    }
  }
  return {
    buf, texIds, archive, texList,
    fetched: fetchedTextures, binding: psov2Binding,
    nativeMotions: nativeMotions || null, motionBuffers, motionNames,
  };
}

// ---- real-mesh path (XJ via /api/model_mesh) -----------------------

/**
 * Decode a base64 string into an ArrayBuffer.
 * (atob -> Uint8Array)
 */
function b64ToArrayBuffer(b64) {
  const binary = atob(b64);
  const len = binary.length;
  const out = new Uint8Array(len);
  for (let i = 0; i < len; i++) out[i] = binary.charCodeAt(i);
  return out.buffer;
}

/**
 * Transform a single AABB by the per-mesh world matrix (row-major 4x4)
 * and return the resulting AABB in world space.
 *
 * Returns ``[wMinX, wMinY, wMinZ, wMaxX, wMaxY, wMaxZ]``. We project
 * all 8 corner points and take the min/max of each axis — sufficient
 * for axis-aligned camera-fit. ``identity`` short-circuits the
 * identity case (vast majority of payloads from the post-2026-04-24
 * server, which bakes vertices into world space already).
 */
function transformAabbByMatrix(localAabb, m, identity) {
  if (!localAabb || localAabb.length !== 6) return null;
  if (identity) return localAabb.slice();
  const corners = [
    [localAabb[0], localAabb[1], localAabb[2]],
    [localAabb[3], localAabb[1], localAabb[2]],
    [localAabb[0], localAabb[4], localAabb[2]],
    [localAabb[3], localAabb[4], localAabb[2]],
    [localAabb[0], localAabb[1], localAabb[5]],
    [localAabb[3], localAabb[1], localAabb[5]],
    [localAabb[0], localAabb[4], localAabb[5]],
    [localAabb[3], localAabb[4], localAabb[5]],
  ];
  const out = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
  for (const [x, y, z] of corners) {
    // Row-major: wx = m[0]*x + m[1]*y + m[2]*z + m[3]
    const wx = m[0] * x + m[1] * y + m[2] * z + m[3];
    const wy = m[4] * x + m[5] * y + m[6] * z + m[7];
    const wz = m[8] * x + m[9] * y + m[10] * z + m[11];
    if (wx < out[0]) out[0] = wx;
    if (wy < out[1]) out[1] = wy;
    if (wz < out[2]) out[2] = wz;
    if (wx > out[3]) out[3] = wx;
    if (wy > out[4]) out[4] = wy;
    if (wz > out[5]) out[5] = wz;
  }
  return out;
}

/**
 * Derive the texture-archive path that pairs with a model URL.
 *
 * For a BML-inner model the texture archive is the same BML with the
 * inner-name suffixed `.xvm`:
 *
 *   "<bml>#<inner>.nj"  →  "<bml>#<inner>.nj.xvm"
 *
 * For a top-level `.nj` we look for a sibling `.xvm` in the same file
 * stem (rare in PSOBB.IO; most models live inside a BML). When the
 * caller drives the legacy hint-flow, `hint.model_archive` may already
 * BE the texture archive — we surface it directly in that case.
 *
 * Returns null when no texture archive can be inferred. The frontend
 * then falls back to the single-tile flow (tile_png/<filename>/<idx>)
 * for the user's currently-selected texture file.
 */
function deriveTextureArchivePath(meshUrl, filename, hint) {
  // Hint-flow: `hint.model_archive` is the BML; the texture archive
  // pairs with whichever `.nj` we resolved.
  // Path-flow: meshUrl is "/api/model_mesh/<bml>?inner=<inner>" or
  // "/api/model_skinned/<bml>?inner=<inner>" — we reconstruct the
  // texture archive from the URL. Both endpoints share the URL shape
  // so the same regex covers them.
  if (!meshUrl) return null;
  // Parse the URL to extract base + inner.
  // Sample shapes:
  //   /api/model_mesh/bm4_ps_ma_body.bml?inner=bm4_ps_ma_body.nj
  //   /api/model_mesh/foo.nj
  //   /api/model_skinned/bm_boss8_dragon.bml?inner=boss1_s_nb_dragon.nj
  // Earlier this regex was anchored to `/api/model_mesh/`, which
  // silently dropped textures whenever the SKINNED loader path was
  // active — every monster/boss with an .nj inner fell back to bare
  // grey because the binding fetch never even started.
  const m = meshUrl.match(/^\/api\/model_(?:mesh|skinned)\/([^?]+)(\?inner=(.+))?$/);
  if (!m) return null;
  const base = decodeURIComponent(m[1]);
  const innerEnc = m[3];
  const inner = innerEnc ? decodeURIComponent(innerEnc) : null;
  if (inner) {
    // BML inner: archive is "<base>#<inner>.xvm". The backend's
    // _split_inner_path treats `#` as the inner separator, and
    // `extract_bml_inner_bytes` recognises a `.xvm` suffix as a
    // request for the texture sibling.
    return `${base}#${inner}.xvm`;
  }
  // Bare BML with NO inner: do NOT fabricate "<stem>.xvm". A BML's
  // textures live in an inline XVM appendix keyed to a concrete inner
  // (e.g. bm_npc_momoka.bml#n_momoka_t_body.nj.xvm), never in a
  // top-level "<stem>.xvm" sibling. Fabricating one fed the bare BML
  // header bytes to the XVMH reader (the momoka "garbage texture
  // count" bug). Returning null hands resolution to the server-side
  // /api/model_textures binding, which auto-selects the BML's sole
  // textured inner (psov2 "the obvious inner" contract).
  if (base.toLowerCase().endsWith(".bml")) {
    return null;
  }
  // Top-level .nj/.xj model: try a same-stem .xvm. We don't actually
  // verify that file exists here — the tile_png endpoint will 404 if
  // not, and the frontend gracefully falls back to per-mesh tile 0 via
  // the existing single-tile flow.
  const stem = base.replace(/\.[a-z0-9]+$/i, "");
  return `${stem}.xvm`;
}

/**
 * Resolve ONE binding row to the concrete `{archive, tile}` the
 * `/api/tile_png/<archive>/<tile>` endpoint serves. Handles the three
 * binding `source` kinds (in_bml / cross_bml / cross_afs) identically
 * to the legacy fetch path. Returns null when the row carries no usable
 * source (e.g. `missing` with no cross-archive fallback).
 *
 * Shared by `fetchBoundTextures` (texture upload) and
 * `psoListMeshTextures` (panel thumbnails) so both agree on the URL.
 */
function resolveBindingRowTile(b, archivePath) {
  if (!b) return null;
  if (b.missing && b.source !== "cross_bml" && b.source !== "cross_afs") return null;
  const src = b.source || (b.missing ? "missing" : "in_bml");
  let arch = archivePath;
  let tile = b.tile_index | 0;
  if (src === "cross_bml" && b.cross_bml && b.cross_bml.bml && b.cross_bml.inner) {
    arch = `${b.cross_bml.bml}#${b.cross_bml.inner}.xvm`;
    tile = (b.cross_bml.xvr_index | 0);
  } else if (src === "cross_afs" && b.cross_afs && b.cross_afs.archive && b.cross_afs.inner_index >= 0) {
    const archive = String(b.cross_afs.archive);
    const idx = b.cross_afs.inner_index | 0;
    const stem = archive.replace(/\.afs$/i, "");
    const idx4 = String(idx).padStart(4, "0");
    arch = `${archive}#${idx4}_${stem}_${idx4}.xvr`;
    tile = (b.cross_afs.xvr_index | 0);
    if (tile < 0) tile = 0;
  } else if (!arch) {
    return null;
  }
  return { archive: arch, tile };
}

/**
 * Fetch and decode all per-mesh textures listed in `binding`.
 *
 * Walks the binding array, deduplicates `tile_index` values, fetches
 * each via `/api/tile_png/<archivePath>/<tile_index>`, and returns a
 * Map<material_id, THREE.Texture>. Textures are configured the same
 * way as the single-tile flow (linear colorSpace, anisotropy=8,
 * RepeatWrapping — see ``loadTileTexture`` for the engine-default
 * rationale).
 *
 * Failed tile fetches log a warning but do NOT abort the whole load —
 * the binding entry's material falls back to the same single-tile
 * source the user-selected `state.texture` would normally show. This
 * matches the spec's requirement: "If material_id has no binding,
 * fall back to tile 0".
 */
async function fetchBoundTextures(archivePath, binding) {
  const out = new Map();
  if (!binding || binding.length === 0) return out;

  // Each binding row may carry a `source` field:
  //   "in_bml"    — texture lives in the host BML's inline XVM;
  //                 tile_index applies against `archivePath`.
  //   "cross_bml" — texture lives in another BML's inline XVM. The
  //                 backend stamps `cross_bml.bml` + `.inner` +
  //                 `.xvr_index`; we synthesise the alternate archive
  //                 path "<bml>#<inner>.nj.xvm" (or .xj.xvm) and use
  //                 xvr_index as the tile index there.
  //   "cross_afs" — texture lives in a sibling AFS archive (player
  //                 class textures in pl[A-Z]tex.afs, item textures in
  //                 ItemTexture*.afs). Backend stamps
  //                 `cross_afs.archive` + `.inner_index` +
  //                 `.xvr_index`. We synthesise the AFS-inner path
  //                 "<archive>#NNNN_<stem>_NNNN.xvr" (matches
  //                 server.py::_parse_afs_inner_name) and the server
  //                 wraps the bare XVR in a synthesised XVMH so the
  //                 existing tile_png pipeline works unchanged.
  //   "missing"   — none of the lookup paths resolved; skipped.
  // We dedupe by (archive, tile) and fetch in parallel.
  const fetchKey = (arch, idx) => `${arch}\u0001${idx | 0}`;
  const wanted = new Map();   // fetchKey -> {archive, tile}
  const rowKeys = new Map();  // material_id -> fetchKey
  for (const b of binding) {
    const resolved = resolveBindingRowTile(b, archivePath);
    if (!resolved) continue;
    const k = fetchKey(resolved.archive, resolved.tile);
    wanted.set(k, { archive: resolved.archive, tile: resolved.tile });
    rowKeys.set(b.material_id | 0, k);
  }

  if (wanted.size === 0) return out;

  // Fire all the fetches in parallel; THREE.TextureLoader returns a
  // promise per call, so wrap them in Promise.all and let any 404
  // resolve to null.
  const wantedArr = Array.from(wanted.entries());
  const fetched = await Promise.all(
    wantedArr.map(([_k, ent]) => loadTileTexture(ent.archive, ent.tile)),
  );
  const tileMap = new Map();
  wantedArr.forEach(([k, _ent], i) => {
    if (fetched[i]) tileMap.set(k, fetched[i]);
  });

  // Build the material_id -> texture map.
  for (const [mid, k] of rowKeys.entries()) {
    const tex = tileMap.get(k);
    if (tex) out.set(mid, tex);
  }
  return out;
}

/**
 * Single helper around THREE.TextureLoader for a tile_png URL.
 * Returns a THREE.Texture configured for mesh use, or null if the
 * fetch failed (e.g. 404 because no texture archive exists for this
 * model). Logs a warning on failure but does NOT throw — the caller
 * uses the null result to skip the binding for that material_id.
 */
function loadTileTexture(archivePath, tileIdx) {
  const url = `/api/tile_png/${encodeURIComponent(archivePath)}/${tileIdx | 0}?cb=${Date.now()}`;
  return new Promise((resolve) => {
    const loader = new THREE.TextureLoader();
    loader.load(
      url,
      (tex) => {
        // Phantasmal-diff fix 1 (2026-04-25): leave colorSpace at its
        // linear default (no double-gamma).
        //
        // V-flip (2026-06-20): PSOBB UVs are top-down-V but THREE's
        // TextureLoader defaults flipY=true, which renders the texture
        // vertically inverted. Match the Phantasmal reference (flipY=false
        // with raw top-down V). Set before resolve (pre-upload) so no
        // needsUpdate is needed.
        tex.flipY = false;
        // Wrap mode (2026-06-20, psov2-grounded): MirroredRepeat = the Sega
        // Ninja / psov2 NinjaTexture.js default for object textures. This is
        // the per-binding MODEL-texture path — object UVs sit in [0,1], so
        // Mirror is harmless for the common case and matches the reference on
        // edge-tiled objects. Tiled floor/room textures are a separate path
        // that stays on Repeat (clampU/clampV), so no floor gets mirror-folded.
        //
        // Anisotropy: bumped from 4 → 8 (2026-04-30). Modern GPUs
        // (anything since GeForce 5xx / Radeon HD 5xxx) handle 8x
        // anisotropic filtering for free in the texture sampler;
        // keeping it at 4 was leaving visual quality on the table on
        // grazing-angle floors and large props. ``maxAnisotropy``
        // varies by GPU but is typically 16 on contemporary
        // hardware; 8 stays under that ceiling on all known cards.
        tex.anisotropy = 8;
        tex.wrapS = THREE.MirroredRepeatWrapping;
        tex.wrapT = THREE.MirroredRepeatWrapping;
        resolve(tex);
      },
      undefined,
      (ev) => {
        const target = ev && ev.target ? ev.target : null;
        const src = (target && target.src) ? target.src.replace(location.origin, "") : url;
        console.warn(`model_viewer: tile ${tileIdx} failed to load (${src})`);
        resolve(null);
      },
    );
  });
}

/**
 * Build a THREE.Group from the JSON payload returned by /api/model_mesh.
 * Each XjMesh becomes one THREE.Mesh wrapped with MeshBasicMaterial,
 * sharing the live texture (which the caller wires up via
 * applyTextureToCurrentMesh).
 *
 * Per-submesh transform handling (added 2026-04-24):
 *   - If ``payload.vertices_pre_transformed === true`` (default for
 *     servers >= 2026-04-24), the parser already baked the
 *     MeshTreeNode bone tree into the strip vertices. We MUST NOT
 *     apply ``m.world_position`` / ``world_matrix`` to the per-mesh
 *     Object3D — doing so would doubly offset every submesh. The
 *     fields are useful for diagnostics (where the AABB centre lives).
 *   - If false (older servers, or future per-bone authoring), each
 *     submesh's vertices are still in BONE-LOCAL space; we apply the
 *     per-mesh ``world_matrix`` (or position+rotation triple) so the
 *     skeleton's bone-local strips render at their global pose.
 *
 * Either way, the combined group AABB is computed in world space so
 * the camera-fit logic still reaches the right zoom level.
 */

/**
 * Apply a payload mesh's per-submesh render-state flags (Phase 3,
 * 2026-06-20) onto a freshly-built three.js material, IN PLACE.
 *
 * Backend (server.py) surfaces, per mesh:
 *   - blend_mode: "none" | "blend" | "additive" | "multiply" | "screen"
 *   - two_sided:  bool
 *   - alpha_test: {enabled: bool, threshold: 0..255} | null
 *   - alpha_blend:{src, dst} | null  (raw factor pair; informational)
 *
 * Mapping (matches the Material Inspector's live edits in
 * applyMaterialState, and psov2's blend handling):
 *   - "additive"  -> THREE.AdditiveBlending + transparent + depthWrite
 *                    false (glow/energy/particles must not occlude).
 *   - alpha_test  -> transparent + alphaTest = threshold/255 (mask cut).
 *   - two_sided   -> DoubleSide, else FrontSide. (When NO flag info is
 *                    present we leave the caller's DoubleSide default —
 *                    see the per-builder comments on reverse-faced
 *                    PSOBB strips — by only narrowing to FrontSide when
 *                    the payload explicitly carries flags.)
 *
 * ``hasFlags`` guards the side narrowing: legacy payloads (no
 * blend_mode key) keep the conservative DoubleSide default so we don't
 * silently drop reverse-faced strips on models the backend didn't tag.
 */
function applyPsoMaterialFlags(mat, m) {
  if (!mat || !m) return mat;
  const hasFlags = (m.blend_mode !== undefined) || (m.two_sided !== undefined)
    || (m.alpha_test !== undefined);
  // Side: respect explicit two_sided when the payload carries flags.
  if (hasFlags) {
    mat.side = m.two_sided ? THREE.DoubleSide : THREE.FrontSide;
  }
  // Additive blend (glow / energy / FX): AdditiveBlending, no depth
  // write so the glow stacks instead of z-fighting.
  if (m.blend_mode === "additive") {
    mat.blending = THREE.AdditiveBlending;
    mat.transparent = true;
    mat.depthWrite = false;
  } else if (m.blend_mode === "blend" && m.alpha_blend) {
    // Standard premultiplied alpha blend (water / smoke / decals):
    // transparent, depth test on, depth write off so the blend reads
    // the framebuffer behind it.
    mat.transparent = true;
    mat.depthWrite = false;
  }
  // Alpha test (masked cut-out, e.g. hair / foliage / skin edges):
  // transparent + alphaTest threshold (0..1).
  const at = m.alpha_test;
  if (at && at.enabled) {
    mat.transparent = true;
    mat.alphaTest = (Number(at.threshold) || 0) / 255;
  }
  return mat;
}

function buildMeshGroupFromPayload(payload, boundTextures) {
  const group = new THREE.Group();
  let totalVerts = 0;
  let totalTris = 0;
  // Combined AABB across all sub-meshes (in world space) for the
  // group-level centering + uniform scale to fit the camera frustum.
  const aabbMin = [Infinity, Infinity, Infinity];
  const aabbMax = [-Infinity, -Infinity, -Infinity];

  // Backend tells us whether vertices are pre-transformed. Default
  // true (post-2026-04-24 server). Older servers omit the flag — we
  // can detect them by the absence of ``world_matrix`` on individual
  // meshes, but the flag is the authoritative signal.
  const preTransformed = payload.vertices_pre_transformed !== false;

  // Debug-mode bookkeeping (added 2026-04-24). When the user toggles
  // "debug overlay" on, we walk this array and re-assign HSL-tinted
  // materials. We record the original payload entry so the sidebar
  // tooltips can show vert_count / tri_count / world_position even
  // after debug colours are applied.
  const debugMeshes = [];

  // Per-submesh texture binding (added 2026-04-25). When the caller
  // resolved the model's binding table and pre-fetched textures for
  // each material_id, each XjMesh receives its OWN texture. Falls back
  // to `state.texture` (the user-selected single-tile texture) when
  // the material_id has no binding entry — that handles legacy servers
  // that omit `payload.binding` and unmapped materials with
  // missing=true.
  const bound = boundTextures || new Map();

  for (const m of payload.meshes || []) {
    const vbuf = b64ToArrayBuffer(m.vertices_b64);
    const ibuf = b64ToArrayBuffer(m.indices_b64);
    const verts = new Float32Array(vbuf);
    const indices = new Uint32Array(ibuf);

    // Skip empty / degenerate sub-meshes
    if (verts.length === 0 || indices.length === 0) continue;

    // Interleaved Float32. v2 payloads (payload.has_color) carry 4
    // trailing RGBA color floats: [px,py,pz, nx,ny,nz, u,v, r,g,b,a]
    // (12 floats). v1 payloads omit them (8 floats). Gate on has_color
    // so older servers/caches still load.
    const hasColor = payload.has_color === true;
    const stride = hasColor ? 12 : 8;
    const vertexCount = verts.length / stride;
    if (!Number.isInteger(vertexCount)) {
      console.warn("model_viewer: non-integer vertex count; skipping mesh");
      continue;
    }
    const positions = new Float32Array(vertexCount * 3);
    const normals = new Float32Array(vertexCount * 3);
    const uvs = new Float32Array(vertexCount * 2);
    const colors = hasColor ? new Float32Array(vertexCount * 4) : null;
    for (let i = 0; i < vertexCount; i++) {
      const o = i * stride;
      positions[i * 3 + 0] = verts[o + 0];
      positions[i * 3 + 1] = verts[o + 1];
      positions[i * 3 + 2] = verts[o + 2];
      normals[i * 3 + 0] = verts[o + 3];
      normals[i * 3 + 1] = verts[o + 4];
      normals[i * 3 + 2] = verts[o + 5];
      uvs[i * 2 + 0] = verts[o + 6];
      uvs[i * 2 + 1] = verts[o + 7];
      if (colors) {
        colors[i * 4 + 0] = verts[o + 8];
        colors[i * 4 + 1] = verts[o + 9];
        colors[i * 4 + 2] = verts[o + 10];
        colors[i * 4 + 3] = verts[o + 11];
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
    geo.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
    if (colors) geo.setAttribute("color", new THREE.BufferAttribute(colors, 4));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));
    geo.computeBoundingSphere();

    // Per-submesh texture: prefer the pre-fetched binding map, fall
    // back to state.texture (the single-tile flow). The single-tile
    // fallback also covers legacy servers that omit `payload.binding`.
    const matId = (m.material_id | 0);
    const submeshTex = bound.has(matId) ? bound.get(matId) : (state.texture || null);

    // Phantasmal-diff fixes 3+4 (2026-04-25): MeshBasicMaterial for
    // textured submeshes (matches Phantasmal's MeshRenderer.kt — PSOBB
    // does no BRDF on diffuse-mapped pixels) and MeshLambertMaterial
    // for un-textured (closest to Sega's per-vertex Lambert).
    // transparent=false by default; the material inspector flips it on
    // demand.
    //
    // side=DoubleSide is the SAFE DEFAULT (2026-04-30 review): the
    // pipeline does NOT do an LH→RH axis flip — `xj.py`'s
    // `_tristrip_to_triangles` already honours the per-strip `cw`
    // header bit so winding matches D3D9 source. Three.js' default
    // (CCW=front) maps cleanly onto that. BUT: PSOBB authoring data
    // contains strips written from "the wrong side" (some Sega Ninja
    // SDK chunks ship reverse-faced; e.g. character hair planes,
    // certain boss-monitor displays, shadow proxies). Without a
    // per-mesh authoring-intent flag in the binding payload, defaulting
    // to FrontSide would silently delete every back-facing strip on
    // those models. DoubleSide is the conservative choice; the cost
    // is one extra fragment-shader pass per back-facing fragment.
    //
    // Per-material override: `formats/material.py` carries a
    // `two_sided` bool (default False) and the Material Inspector
    // exposes it via `psoUpdateMaterial(idx, {two_sided: false})`,
    // which flips THIS submesh to FrontSide. Use that route when you
    // want to opt OUT of double-sided rendering for a specific
    // material — it doesn't pay the engineering cost of a backend
    // change to surface authoring intent on every binding row.
    // Phase 3 (2026-06-20): BOTH branches are now MeshBasicMaterial
    // (UNLIT) to match psov2 (DashGL NinjaModel.js — all materials are
    // MeshBasicMaterial). The un-textured branch was MeshLambertMaterial,
    // which is LIT by the scene's Hemisphere+Directional lights and
    // washes untextured submeshes to flat white. PSOBB bakes shading
    // into per-vertex / diffuse color, so unlit × vertexColor reproduces
    // the authored look. vertexColors:true multiplies map (or 0xffffff)
    // by the per-vertex RGBA we de-interleaved above.
    const mat = submeshTex
      ? new THREE.MeshBasicMaterial({
          map: submeshTex,
          color: 0xffffff,
          vertexColors: hasColor,
          wireframe: state.wireframe,
          side: THREE.DoubleSide,
          transparent: false,
        })
      : new THREE.MeshBasicMaterial({
          color: 0xffffff,
          vertexColors: hasColor,
          wireframe: state.wireframe,
          side: THREE.DoubleSide,
          transparent: false,
        });
    // Phase 3 (2026-06-20): fold the payload's per-submesh render-state
    // flags (blend_mode / alpha_test / two_sided) onto the material.
    applyPsoMaterialFlags(mat, m);

    const mesh = new THREE.Mesh(geo, mat);
    // Stash the material_id on the mesh so the texture-replace flow
    // (loadTexture → applyTextureToCurrentMesh) can preserve per-mesh
    // bindings when the user edits the SELECTED tile only. We DON'T
    // overwrite a submesh whose material_id has its own binding.
    mesh.userData.materialId = matId;

    // Apply per-mesh world transform iff vertices are still in
    // bone-local space (legacy servers only). For the post-2026-04-24
    // server, the parser bakes vertices into world space and the
    // per-mesh transforms are diagnostic only.
    if (!preTransformed && m.world_matrix && m.world_matrix.length === 16) {
      // Three.js Matrix4 is column-major, but our payload is row-major.
      // Transpose during load by passing in row-by-row order.
      const wm = m.world_matrix;
      const t = new THREE.Matrix4();
      // Three's elements layout: [m00, m10, m20, m30, m01, m11, ...].
      // We have row-major: wm[0..3] is row 0, wm[4..7] is row 1, etc.
      t.set(
        wm[0],  wm[1],  wm[2],  wm[3],
        wm[4],  wm[5],  wm[6],  wm[7],
        wm[8],  wm[9],  wm[10], wm[11],
        wm[12], wm[13], wm[14], wm[15],
      );
      mesh.matrixAutoUpdate = false;
      mesh.matrix.copy(t);
      // Also surface the decomposed pose so child.position/rotation
      // reflect the same transform when the user inspects the mesh.
      t.decompose(mesh.position, mesh.quaternion, mesh.scale);
      mesh.matrixAutoUpdate = true;
    } else if (!preTransformed && m.world_position && m.world_rotation_euler) {
      // Fallback for legacy servers that emit pos/euler but no matrix.
      const wp = m.world_position;
      const wr = m.world_rotation_euler;
      mesh.position.set(wp[0], wp[1], wp[2]);
      mesh.rotation.set(wr[0], wr[1], wr[2], "XYZ");
      if (m.world_scale && m.world_scale.length === 3) {
        mesh.scale.set(m.world_scale[0], m.world_scale[1], m.world_scale[2]);
      }
    }
    // else preTransformed==true: leave Object3D at identity.

    group.add(mesh);
    // Track per-submesh metadata so the debug overlay can render a
    // sidebar entry without re-fetching the JSON payload. We capture a
    // shallow projection of the wire entry (no buffers — those are
    // already on `geo`) plus the resolved index inside `group.children`.
    debugMeshes.push({
      idx: debugMeshes.length,
      mesh,                              // THREE.Mesh
      material_id: matId,
      vertex_count: vertexCount,
      triangle_count: (indices.length / 3) | 0,
      world_position: m.world_position || [0, 0, 0],
      world_rotation_euler: m.world_rotation_euler || [0, 0, 0],
      world_scale: m.world_scale || [1, 1, 1],
      bounding_sphere: m.bounding_sphere || [0, 0, 0, 0],
      aabb: m.aabb || null,
      eval_flags: m.eval_flags || 0,     // backend may add this later
    });

    totalVerts += vertexCount;
    totalTris += indices.length / 3;

    // Update group AABB. When the per-mesh transform is non-identity
    // (legacy bone-local server), project the local AABB through the
    // matrix; otherwise the local AABB IS the world AABB.
    if (m.aabb && m.aabb.length === 6) {
      const transformed = (!preTransformed && m.world_matrix && m.world_matrix.length === 16)
        ? transformAabbByMatrix(m.aabb, m.world_matrix, false)
        : m.aabb;
      if (transformed[0] < aabbMin[0]) aabbMin[0] = transformed[0];
      if (transformed[1] < aabbMin[1]) aabbMin[1] = transformed[1];
      if (transformed[2] < aabbMin[2]) aabbMin[2] = transformed[2];
      if (transformed[3] > aabbMax[0]) aabbMax[0] = transformed[3];
      if (transformed[4] > aabbMax[1]) aabbMax[1] = transformed[4];
      if (transformed[5] > aabbMax[2]) aabbMax[2] = transformed[5];
    }
  }

  // Center the group & rescale into the camera's [-1.5, 1.5] view
  if (Number.isFinite(aabbMin[0])) {
    const cx = (aabbMin[0] + aabbMax[0]) / 2;
    const cy = (aabbMin[1] + aabbMax[1]) / 2;
    const cz = (aabbMin[2] + aabbMax[2]) / 2;
    const dx = aabbMax[0] - aabbMin[0];
    const dy = aabbMax[1] - aabbMin[1];
    const dz = aabbMax[2] - aabbMin[2];
    const maxDim = Math.max(dx, dy, dz, 0.001);
    const scale = 2.0 / maxDim;     // fit largest axis into ~[-1, 1]
    group.position.set(-cx * scale, -cy * scale, -cz * scale);
    group.scale.set(scale, scale, scale);
  }

  return { group, totalVerts, totalTris, aabbMin, aabbMax, debugMeshes };
}

// Internal struct used by callers to record WHY the fallback ran. We
// keep it on `state.lastMeshFailure` so `open()` can render the right
// banner ("primitive (cube) — model unavailable: <reason>") instead of
// silently swapping in a primitive.
function _setMeshFailure(reason) {
  state.lastMeshFailure = reason || null;
}

/**
 * Try to load real geometry from /api/model_mesh.
 *
 * Accepts either:
 *   - a hint object: `{model_archive, model_archive_inner?}` — the
 *     legacy texture-first flow (`open(filename)` calls this with the
 *     hint returned by `/api/model_preview`).
 *   - a path target: `{path: "<archive>"}` or
 *     `{path: "<bml>#<inner>"}` — the asset-tree flow where we already
 *     know exactly which model+inner to render.
 *
 * Returns true on success (real mesh now displayed), false if the
 * request failed. On failure `state.lastMeshFailure` carries the reason
 * so the caller can populate the fallback banner.
 *
 * The mapping from texture filename -> mesh path (legacy hint flow):
 *   `<basename>_tex.xvm` -> `<basename>.bml` (R1 paired BML), or
 *   `pl[A-X]tex.afs` -> `pl[A-X]bdy00.nj`.
 *
 * For a hint pointing at a `.bml` we ask the BML list endpoint and pick
 * the first `.nj` we find (best-effort — the asset-tree path-driven
 * form passes the inner explicitly to skip this guess).
 */
async function tryLoadRealMesh(hint) {
  // Path-driven form: caller already has a fully-qualified
  // `<archive>` or `<bml>#<inner>` string. This is the asset-tree path —
  // we skip the BML-list "first .nj" guess entirely.
  let url;
  let label;
  let archiveLabel;

  if (hint && hint.path) {
    const path = String(hint.path);
    archiveLabel = path;
    // Detect <bml>#<inner> form. The backend accepts both `?inner=` and
    // the `#` fragment (see _split_inner_path); we ship the path
    // verbatim to /api/model_mesh.
    const hashIdx = path.indexOf("#");
    if (hashIdx > 0) {
      const base = path.slice(0, hashIdx);
      const inner = path.slice(hashIdx + 1);
      const baseLow = base.toLowerCase();
      if (!baseLow.endsWith(".bml") && !baseLow.endsWith(".afs")) {
        _setMeshFailure(`'#' form requires .bml or .afs base, got ${baseLow.split(".").pop()}`);
        return false;
      }
      // Strip an optional .xvm tail — if the user clicked a TEXTURE
      // entry inside a BML and we were asked to view it as a model, the
      // inner ends in `.nj.xvm`. The model inner is everything before
      // the `.xvm`. The asset router does this strip too, but doing it
      // here is defence-in-depth so direct callers (devtools, tests)
      // also benefit.
      const innerLow = inner.toLowerCase();
      let innerModel = inner;
      if (innerLow.endsWith(".xvm")) {
        innerModel = inner.slice(0, -4);
      }
      // For BML inners we validate the extension client-side; AFS
      // inners are NNNN_<name> synth strings and are handled server-side
      // by ``_read_afs_inner_nj`` (the inner's sniffed extension governs
      // the parse-cache route).
      //
      // Both .nj (chunk-Ninja, skinned-capable) and .xj (descriptor-Xj,
      // static-mesh) are valid mesh extensions; the static path through
      // ``/api/model_mesh`` accepts both. .njm (motion-only) is also
      // accepted because the bundle endpoint synthesises a
      // motion-preview node so the user sees the bones move.
      // Reject anything else (e.g. .xvm leaked through, raw bin) early
      // so the network request doesn't 400 with a less-clear message.
      if (baseLow.endsWith(".bml")) {
        const innerExt = "." + innerModel.toLowerCase().split(".").pop();
        if (innerExt !== ".nj" && innerExt !== ".xj" && innerExt !== ".njm") {
          _setMeshFailure(`inner '${innerModel}' is not a model (.nj/.xj/.njm)`);
          return false;
        }
      }
      url = `/api/model_mesh/${encodeURIComponent(base)}?inner=${encodeURIComponent(innerModel)}`;
      label = `${base} :: ${innerModel}`;
    } else {
      const lower = path.toLowerCase();
      if (lower.endsWith(".bml")) {
        // Top-level `.bml` with no inner specifier. The backend won't
        // accept this (it requires ?inner=), so we list the BML and
        // pick the first `.nj` — same heuristic the legacy flow uses.
        // Callers who DO know the inner pass `<bml>#<inner>.nj` and
        // skip this branch.
        let entries;
        try {
          const r = await _lifecycleFetch(`/api/bml/${encodeURIComponent(path)}/list`);
          if (!r.ok) {
            _setMeshFailure(`bml list http ${r.status}`);
            return false;
          }
          entries = await r.json();
        } catch (e) {
          if (_isAbortError(e)) return false;
          _setMeshFailure(`bml list error: ${e?.message || e}`);
          return false;
        }
        // Prefer .nj (skinned-capable) but fall back to .xj for static
        // meshes (doors, switches, props). The /api/model_mesh endpoint
        // accepts both — its error message even says ".{nj,xj}".
        const _entries = entries.entries || [];
        const inner = _entries.find((x) => /\.nj$/i.test(x.name))
                   || _entries.find((x) => /\.xj$/i.test(x.name));
        if (!inner) {
          _setMeshFailure(`bml ${path} has no inner .nj or .xj`);
          return false;
        }
        url = `/api/model_mesh/${encodeURIComponent(path)}?inner=${encodeURIComponent(inner.name)}`;
        label = `${path} :: ${inner.name}`;
      } else if (lower.endsWith(".nj") || lower.endsWith(".njm")) {
        // Plain `.nj` (or `.njm`) outside a BML.
        url = `/api/model_mesh/${encodeURIComponent(path)}`;
        label = path;
      } else {
        _setMeshFailure(`unsupported model extension '${path.split(".").pop()}'`);
        return false;
      }
    }
  } else {
    // Legacy hint-driven form (texture-first flow via /api/model_preview).
    const archive = hint && hint.model_archive;
    if (!archive) {
      _setMeshFailure("no paired model archive in hint");
      return false;
    }
    archiveLabel = archive;
    const ext = archive.toLowerCase().endsWith(".bml") ? "bml" : "nj";
    if (ext === "bml") {
      // Pick first inner .nj.
      let entries;
      try {
        const r = await _lifecycleFetch(`/api/bml/${encodeURIComponent(archive)}/list`);
        if (!r.ok) {
          _setMeshFailure(`bml list http ${r.status}`);
          return false;
        }
        entries = await r.json();
      } catch (e) {
        if (_isAbortError(e)) return false;
        _setMeshFailure(`bml list error: ${e?.message || e}`);
        return false;
      }
      // Prefer .nj (skinned-capable) but fall back to .xj for static
      // meshes — see "Prefer .nj" comment in the openByPath flow above.
      const _entries = entries.entries || [];
      const inner = _entries.find((x) => /\.nj$/i.test(x.name))
                 || _entries.find((x) => /\.xj$/i.test(x.name));
      if (!inner) {
        _setMeshFailure(`bml ${archive} has no inner .nj or .xj`);
        return false;
      }
      url = `/api/model_mesh/${encodeURIComponent(archive)}?inner=${encodeURIComponent(inner.name)}`;
      label = `${archive} :: ${inner.name}`;
    } else {
      url = `/api/model_mesh/${encodeURIComponent(archive)}`;
      label = archive;
    }
  }

  setStatus(`loading mesh ${label}...`);
  let payload;
  try {
    const r = await _lifecycleFetch(url);
    if (r.status === 503) {
      let detail;
      try { detail = (await r.json()).detail; } catch { detail = "service unavailable"; }
      _setMeshFailure(`mesh ${label}: ${detail}`);
      return false;
    }
    if (!r.ok) {
      // Pull the detail from the JSON body for clearer messages.
      let detail = `http ${r.status}`;
      try {
        const errBody = await r.json();
        if (errBody && errBody.detail) detail = errBody.detail;
      } catch {
        // body wasn't JSON; keep the status-code label.
      }
      _setMeshFailure(`${detail}`);
      return false;
    }
    payload = await r.json();
  } catch (e) {
    if (_isAbortError(e)) return false;
    _setMeshFailure(`mesh fetch error: ${e?.message || e}`);
    return false;
  }
  if (!payload || !payload.mesh_count) {
    _setMeshFailure(`mesh ${label}: no geometry parsed`);
    return false;
  }

  // Per-submesh texture binding (added 2026-04-25). The backend
  // ships a `binding` array in the model_mesh payload (post-binding
  // server) listing one row per unique material_id with its target
  // tile_index. We pre-fetch each unique tile_index from the model's
  // sibling XVMH archive into LOCAL variables — `disposeMesh()` below
  // wipes `state.boundTextures` so we must defer the assign until
  // after the dispose call to avoid freeing the textures we just
  // fetched.
  const newBinding = payload.binding || [];
  let newBoundTextures = new Map();
  let newBoundArchive = null;
  if (newBinding.length > 0) {
    const archive = deriveTextureArchivePath(url, state.filename, hint);
    if (archive) {
      try {
        newBoundTextures = await fetchBoundTextures(archive, newBinding);
        newBoundArchive = archive;
      } catch (e) {
        console.warn(`model_viewer: bound-texture fetch failed for ${archive}: ${e?.message || e}`);
      }
    }
  }

  const built = buildMeshGroupFromPayload(payload, newBoundTextures);
  if (!built.group.children.length) {
    _setMeshFailure(`mesh ${label}: no rendered sub-meshes`);
    // Local map never made it onto state — release immediately.
    for (const t of newBoundTextures.values()) {
      try { t.dispose(); } catch {}
    }
    return false;
  }

  ensureRenderer();
  // disposeMesh() wipes the existing state.boundTextures (frees old
  // GPU resources). After it returns, swap in the new bindings.
  disposeMesh();
  state.boundTextures = newBoundTextures;
  state.boundTextureArchive = newBoundArchive;
  state.boundBinding = newBinding;
  state.mesh = built.group;
  state.meshGroup = built.group;
  state.realMesh = true;
  state.scene.add(state.mesh);
  // Snapshot debug-mode metadata for the new model. If debug mode was
  // already on (user toggles between models with overlay enabled), apply
  // the colourisation right now so the next render frame paints tinted.
  state.debugMeshes = built.debugMeshes || [];
  state.debugActiveIdx = -1;
  rebuildDebugSidebar();
  if (state.debugMode) applyDebugMaterials(true);
  kick();

  const aabbDx = built.aabbMax[0] - built.aabbMin[0];
  const aabbDy = built.aabbMax[1] - built.aabbMin[1];
  const aabbDz = built.aabbMax[2] - built.aabbMin[2];
  // Surface the binding count so the user can see at a glance whether
  // per-submesh texture binding kicked in. ``5/5 bound`` means every
  // material_id has its own texture; ``5/8 bound`` indicates 3 missing
  // entries that fell back to single-tile.
  const bindingTotal = newBinding.length;
  const bindingResolved = newBoundTextures.size;
  const bindingBit = bindingTotal > 0
    ? `  tex ${bindingResolved}/${bindingTotal} bound`
    : "";
  setMeshStats(
    `mesh: ${payload.mesh_count} sub  verts ${built.totalVerts}  tris ${built.totalTris}  ` +
    `aabb ${aabbDx.toFixed(2)}x${aabbDy.toFixed(2)}x${aabbDz.toFixed(2)}${bindingBit}`,
  );
  setStatus(`mesh ${label}: ${payload.mesh_count} sub-meshes loaded`);
  _setMeshFailure(null);
  // Stash the archive label so callers (open() / openByPath()) that
  // ALSO need to load a texture know what archive paired the mesh.
  state.realMeshArchive = archiveLabel;
  return true;
}

// ---- model export (OBJ / GLB) --------------------------------------

/**
 * Export the currently-loaded model + its bound textures via
 * POST /api/export_model, then download the staged artifact. The mesh is
 * rebuilt server-side from the SAME asset path the viewer opened, so the
 * export matches what's on screen. FBX is disabled in the dropdown (no
 * writer) and would 501 if forced.
 */
async function exportModel() {
  const fmt = ($("#modelExportFmt") && $("#modelExportFmt").value) || "glb";
  // The asset path that produced the real mesh (e.g. "<bml>#<inner>" or a
  // bare ".nj"). Stashed by tryLoadRealMesh on success.
  const assetPath = state.realMeshArchive;
  if (!state.realMesh || !assetPath) {
    setStatus("export: load a real model first (primitives can't be exported)");
    return;
  }
  setStatus(`exporting ${assetPath} as ${fmt.toUpperCase()}...`);
  try {
    const r = await fetch("/api/export_model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: assetPath, format: fmt }),
    });
    if (!r.ok) {
      let detail = `http ${r.status}`;
      try { detail = (await r.json()).detail || detail; } catch {}
      setStatus(`export failed: ${detail}`);
      return;
    }
    const body = await r.json();
    if (Array.isArray(body.warnings) && body.warnings.length) {
      console.warn("model export warnings:", body.warnings);
    }
    // Trigger the browser download of the staged artifact.
    const a = document.createElement("a");
    a.href = body.export_url;
    a.download = body.filename || `model.${fmt}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    const texBit = body.texture_count != null ? `, ${body.texture_count} tex` : "";
    setStatus(`exported ${body.filename} (${body.mesh_count} sub${texBit})`);
  } catch (e) {
    setStatus(`export error: ${e?.message || e}`);
  }
}

// ---- modal lifecycle -----------------------------------------------

async function open(filename) {
  state.filename = filename;
  state.selectedTileIdx = 0;
  state.realMesh = false;
  setMeshStats(null);
  $("#modelModal").hidden = false;
  $("#modelModalTitle").textContent = filename;
  setStatus("loading hint...");

  // Get the preview hint
  let hint;
  try {
    hint = await fetchPreviewHint(filename);
  } catch (e) {
    setStatus(`hint failed: ${e?.message || e}`);
    return;
  }

  state.tileCount = hint.tile_count || 0;
  // Default to the suggested shape but respect user override on subsequent opens
  state.shape = hint.shape || "sphere";
  $("#modelShapeSel").value = state.shape;

  // Populate tile dropdown
  const tileSel = $("#modelTileSel");
  tileSel.innerHTML = "";
  for (let i = 0; i < state.tileCount; i++) {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = `tile ${i}`;
    tileSel.appendChild(opt);
  }
  tileSel.value = "0";

  // Show hint copy
  const ft = hint.first_tile;
  const ftBit = ft ? `first tile ${ft.width}x${ft.height}` : "";
  const modelBit = hint.model_archive
    ? ` paired model: ${hint.model_archive}`
    : "";
  $("#modelModalMeta").textContent =
    `${state.tileCount} tile(s)${ftBit ? " - " + ftBit : ""}`;
  $("#modelHint").innerHTML =
    `<strong>${hint.shape}</strong> &mdash; ${hint.why}.${modelBit}` +
    `<br/><span class="dim">drag to rotate, scroll to zoom, ` +
    `pick a different tile/shape from the bar above.</span>`;

  // Try the real-mesh path first; if it fails (no archive, parse error,
  // BML not in install, etc.) fall back to the primitive-based preview.
  // The primitive path keeps the file viewable, but we now mark the
  // overlay clearly so the user can tell at a glance ("primitive (cube)
  // — model unavailable: <reason>") instead of staring at a wrapped cube
  // wondering if it's the real model.
  ensureRenderer();
  const realLoaded = await tryLoadRealMesh(hint);
  if (!realLoaded) {
    // First-load fallback: build the primitive immediately so the modal
    // appears with content (no 10 ms blank canvas). Phantasmal-diff
    // fix 5 throttle is for interactive edits, not initial load.
    rebuildMeshNow();
    setFallbackBanner(state.shape, state.lastMeshFailure);
  }
  await loadTexture();
  // The modal just became visible; the canvas's parent may have only
  // resolved its real clientWidth/Height after layout settled. Force a
  // resize + immediate render on the next animation frame so the user
  // sees the correctly-sized scene without needing to click the canvas
  // to "kick" it. (Prior bug: 'comes up weird until you click off and
  // back on'.)
  requestAnimationFrame(() => {
    resizeRenderer();   // resizeRenderer() already schedules a paint via requestRender()
  });
}

function close() {
  $("#modelModal").hidden = true;
  stopLoop();
  // Cancel any pending one-shot so no stray frame is scheduled after the
  // viewer is torn down.
  if (state.renderPending) {
    cancelAnimationFrame(state.renderPending);
    state.renderPending = null;
  }
  // Free GPU resources so reopening (or switching files) starts fresh.
  if (state.texture) {
    state.texture.dispose();
    state.texture = null;
  }
  // Free per-submesh bound textures so they don't leak across opens.
  // (`disposeMesh` would also clear them, but it isn't called from
  // close() — only from rebuildMesh / tryLoadRealMesh.)
  for (const t of state.boundTextures.values()) {
    try { t.dispose(); } catch {}
  }
  state.boundTextures = new Map();
  state.boundTextureArchive = null;
  state.boundBinding = [];
  // Hide animation bar + clear animation state so the next open()
  // starts with a clean slate. (disposeMesh() resets the state object
  // but the UI bar's `hidden` attribute persists across opens.)
  const animBar = $("#modelAnimBar");
  if (animBar) animBar.hidden = true;
  state.anim.skinned = false;
  state.anim.playing = false;
  state.anim.currentMotion = null;
  state.anim.currentData = null;
  // Reset fallback state so the next open() starts clean.
  setMeshStats(null);
  _setMeshFailure(null);
  state.realMeshArchive = null;
  // Multi-inner picker (2026-04-25): hide the dropdown so it doesn't
  // linger across opens. The select element's options are recomputed
  // on the next BML-open if applicable.
  state.bmlInnersInfo = null;
  const innerPickWrap = $("#modelInnerPickWrap");
  if (innerPickWrap) innerPickWrap.hidden = true;
  // Keep renderer/scene around — they're cheap and reused on next open.
}

// ---- debug-overlay mode (2026-04-24) -------------------------------
//
// AGENT_MODEL_DEEP_DEBUG_REPORT investigation D: when the user toggles
// "debug overlay" on, each loaded sub-mesh receives a uniquely-coloured
// material clone (HSL-spread by index). Hovering the canvas raycasts
// against the group; the closest sub-mesh under the cursor pops a
// tooltip with mesh_idx, vert_count, tri_count, world_position.
// A sidebar lists every sub-mesh; clicking a row swaps its tint to a
// bright "active" colour. Switching the toggle off restores each
// mesh's original textured material.

/**
 * Convert the per-mesh index into a stable (hue, sat, lightness) so
 * sibling sub-meshes get visibly different colours. We use a simple
 * golden-ratio increment to avoid clumps in the rainbow.
 */
function _debugHueForIndex(i) {
  // Golden-ratio-conjugate stride keeps colours well-separated for
  // any prefix of the sequence.
  const PHI = 0.61803398875;
  const h = (i * PHI) % 1.0;
  return h;
}

/**
 * Apply (or restore) debug-tinted materials on every sub-mesh in
 * state.debugMeshes. Called when the toggle flips and after each
 * model load while debug mode is on.
 *
 * On enable: stash the original material under userData.origMaterial,
 * then swap in a MeshBasicMaterial (no lighting — flat colour reads
 * better for inspection) using the HSL hue computed from the index.
 *
 * On disable: walk the meshes and restore userData.origMaterial,
 * disposing the temporary debug material to free the GPU resource.
 */
function applyDebugMaterials(enable) {
  if (!state.debugMeshes || !state.debugMeshes.length) return;
  for (const entry of state.debugMeshes) {
    const m = entry.mesh;
    if (!m || !m.material) continue;
    if (enable) {
      if (!m.userData.origMaterial) m.userData.origMaterial = m.material;
      const hue = _debugHueForIndex(entry.idx);
      const color = new THREE.Color().setHSL(hue, 0.7, 0.55);
      const dbg = new THREE.MeshBasicMaterial({
        color,
        wireframe: state.wireframe,
        side: THREE.DoubleSide,
        transparent: false,
      });
      m.material = dbg;
      // Stash so toggleHighlight + sidebar swatch use the same colour
      m.userData.debugBaseColor = color.clone();
    } else {
      if (m.userData.origMaterial) {
        try { m.material.dispose(); } catch {}
        m.material = m.userData.origMaterial;
        m.userData.origMaterial = null;
      }
      m.userData.debugBaseColor = null;
    }
  }
}

/**
 * Highlight a single sub-mesh by its index in state.debugMeshes. The
 * previously-active mesh (if any) reverts to its hue colour. The
 * highlighted mesh gets bright white.
 */
function setDebugActiveMesh(idx) {
  if (state.debugActiveIdx === idx) return;
  // Revert previous
  if (state.debugActiveIdx >= 0 && state.debugMeshes[state.debugActiveIdx]) {
    const prev = state.debugMeshes[state.debugActiveIdx].mesh;
    if (prev && prev.material && prev.userData.debugBaseColor) {
      prev.material.color.copy(prev.userData.debugBaseColor);
    }
  }
  state.debugActiveIdx = idx;
  if (idx >= 0 && state.debugMeshes[idx]) {
    const cur = state.debugMeshes[idx].mesh;
    if (cur && cur.material) {
      cur.material.color.set(0xffffff);
    }
  }
  // Mirror in sidebar: active class + ARIA selected + roving tabindex so
  // focus and AT announcement track the active sub-mesh.
  const list = $("#modelDebugList");
  if (list) {
    list.querySelectorAll(".model-debug-row").forEach((row) => {
      const on = row.dataset.idx === String(idx);
      row.classList.toggle("active", on);
      row.setAttribute("aria-selected", on ? "true" : "false");
      // Keep exactly one row tabbable when there's an active selection.
      if (idx >= 0) row.setAttribute("tabindex", on ? "0" : "-1");
    });
    if (idx >= 0) {
      const target = list.querySelector(`.model-debug-row[data-idx="${idx}"]`);
      if (target && typeof target.scrollIntoView === "function") {
        target.scrollIntoView({ block: "nearest" });
      }
    }
  }
}

/**
 * Repopulate the sidebar with one row per sub-mesh. Idempotent — call
 * after each model load while debug mode is on, or when the user
 * toggles debug ON.
 */
function rebuildDebugSidebar() {
  const list = $("#modelDebugList");
  const countLabel = $("#modelDebugCount");
  if (!list) return;
  list.innerHTML = "";
  const n = state.debugMeshes.length;
  if (countLabel) countLabel.textContent = `${n}`;
  // a11y (2026-06-20): the sub-mesh list is a single-select widget. Mark
  // it role=listbox so AT announces it, and give each row role=option +
  // a roving tabindex so Tab reaches the list and Arrow keys move within.
  list.setAttribute("role", "listbox");
  list.setAttribute("aria-label", "sub-meshes");
  // Build rows in document fragment to avoid layout thrash on big models
  const frag = document.createDocumentFragment();
  let rovingSet = false;
  for (const entry of state.debugMeshes) {
    const row = document.createElement("div");
    row.className = "model-debug-row";
    row.dataset.idx = String(entry.idx);
    const isActive = entry.idx === state.debugActiveIdx;
    if (isActive) row.classList.add("active");
    row.setAttribute("role", "option");
    row.setAttribute("aria-selected", isActive ? "true" : "false");
    // Roving tabindex: the active row is tabbable; otherwise the first
    // row, so keyboard users can always enter the list with Tab.
    const tab = isActive || (!rovingSet && state.debugActiveIdx < 0);
    row.setAttribute("tabindex", tab ? "0" : "-1");
    if (tab) rovingSet = true;
    const swatch = document.createElement("span");
    swatch.className = "model-debug-swatch";
    const hue = _debugHueForIndex(entry.idx);
    const c = new THREE.Color().setHSL(hue, 0.7, 0.55);
    swatch.style.background = `rgb(${(c.r * 255) | 0},${(c.g * 255) | 0},${(c.b * 255) | 0})`;
    const text = document.createElement("span");
    text.className = "model-debug-row-text";
    text.textContent = `#${entry.idx}  mat ${entry.material_id}`;
    const stats = document.createElement("span");
    stats.className = "model-debug-row-stats";
    stats.textContent = `${entry.vertex_count}v ${entry.triangle_count}t`;
    row.appendChild(swatch);
    row.appendChild(text);
    row.appendChild(stats);
    row.addEventListener("click", () => setDebugActiveMesh(entry.idx));
    row.addEventListener("keydown", (ev) => onDebugRowKeydown(ev, row));
    frag.appendChild(row);
  }
  // If a row is tabbable was never set (active idx beyond the list), make
  // the first visible row tabbable as a fallback.
  list.appendChild(frag);
  if (!rovingSet) {
    const firstRow = list.querySelector(".model-debug-row");
    if (firstRow) firstRow.setAttribute("tabindex", "0");
  }
  applyDebugFilter(state.debugFilter || "");
}

// Roving-tabindex keyboard nav for the sub-mesh list. Arrow Up/Down move
// the active mesh (skipping filtered-out rows); Home/End jump to the
// first/last visible row; Enter/Space (re)confirm. setDebugActiveMesh
// already mirrors the highlight into the 3D view + scrolls the row in.
function onDebugRowKeydown(ev, row) {
  const list = $("#modelDebugList");
  if (!list) return;
  // Only walk rows that are actually visible (the filter hides some).
  const rows = Array.from(list.querySelectorAll(".model-debug-row"))
    .filter((r) => !r.classList.contains("hidden-row"));
  if (!rows.length) return;
  const idx = rows.indexOf(row);
  let target = null;
  switch (ev.key) {
    case "ArrowDown": target = rows[Math.min(rows.length - 1, idx + 1)]; break;
    case "ArrowUp":   target = rows[Math.max(0, idx - 1)]; break;
    case "Home":      target = rows[0]; break;
    case "End":       target = rows[rows.length - 1]; break;
    case "Enter":
    case " ":
      ev.preventDefault();
      setDebugActiveMesh(row.dataset.idx | 0);
      return;
    default: return;
  }
  if (!target) return;
  ev.preventDefault();
  setDebugActiveMesh(target.dataset.idx | 0);
  target.focus();
}

function applyDebugFilter(query) {
  state.debugFilter = (query || "").trim().toLowerCase();
  const list = $("#modelDebugList");
  if (!list) return;
  if (!state.debugFilter) {
    list.querySelectorAll(".model-debug-row").forEach((r) => r.classList.remove("hidden-row"));
    return;
  }
  for (const row of list.querySelectorAll(".model-debug-row")) {
    const idx = (row.dataset.idx | 0);
    const e = state.debugMeshes[idx];
    if (!e) continue;
    const wp = e.world_position;
    const text = (
      `#${e.idx} mat ${e.material_id} v${e.vertex_count} t${e.triangle_count} ` +
      `wp(${wp[0].toFixed(1)},${wp[1].toFixed(1)},${wp[2].toFixed(1)})`
    ).toLowerCase();
    row.classList.toggle("hidden-row", !text.includes(state.debugFilter));
  }
}

/**
 * Show the debug tooltip near the pointer for the given sub-mesh
 * entry. Pass `null` to hide.
 */
function showDebugTooltip(entry, screenX, screenY) {
  const tip = $("#modelDebugTooltip");
  if (!tip) return;
  if (!entry) { tip.hidden = true; return; }
  const wp = entry.world_position || [0, 0, 0];
  const wr = entry.world_rotation_euler || [0, 0, 0];
  const ws = entry.world_scale || [1, 1, 1];
  const r2d = (r) => (r * 180.0 / Math.PI).toFixed(1);
  tip.textContent =
    `#${entry.idx}  material_id=${entry.material_id}  ` +
    `verts=${entry.vertex_count} tris=${entry.triangle_count}\n` +
    `world_pos = (${wp[0].toFixed(2)}, ${wp[1].toFixed(2)}, ${wp[2].toFixed(2)})\n` +
    `world_rot°= (${r2d(wr[0])}, ${r2d(wr[1])}, ${r2d(wr[2])})\n` +
    `world_scl = (${ws[0].toFixed(2)}, ${ws[1].toFixed(2)}, ${ws[2].toFixed(2)})`;
  tip.style.left = `${screenX + 14}px`;
  tip.style.top = `${screenY + 14}px`;
  tip.hidden = false;
}

function _debugRaycast(clientX, clientY) {
  if (!state.debugMode || !state.debugMeshes.length) return -1;
  const cv = $("#modelCanvas");
  if (!cv || !state.camera) return -1;
  const rect = cv.getBoundingClientRect();
  const mx = ((clientX - rect.left) / rect.width) * 2 - 1;
  const my = -(((clientY - rect.top) / rect.height) * 2 - 1);
  if (!state.debugRaycaster) state.debugRaycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2(mx, my);
  state.debugRaycaster.setFromCamera(ndc, state.camera);
  const meshes = state.debugMeshes.map((e) => e.mesh).filter(Boolean);
  const hits = state.debugRaycaster.intersectObjects(meshes, false);
  if (!hits.length) return -1;
  const hitMesh = hits[0].object;
  const idx = state.debugMeshes.findIndex((e) => e.mesh === hitMesh);
  return idx;
}

function setDebugMode(enable) {
  state.debugMode = !!enable;
  applyDebugMaterials(state.debugMode);
  const sidebar = $("#modelDebugSidebar");
  const tip = $("#modelDebugTooltip");
  if (sidebar) sidebar.hidden = !state.debugMode;
  if (tip) tip.hidden = true;
  // Sync the toolbar checkbox in case this was triggered by the keyboard
  // shortcut (D). The checkbox is the source-of-truth for the user; we
  // mirror the boolean state into it.
  const tog = $("#modelDebugToggle");
  if (tog) tog.checked = state.debugMode;
  if (state.debugMode) rebuildDebugSidebar();
  requestRender();
}

// ---- wire up UI handlers -------------------------------------------

function init() {
  const btn = $("#btnView3D");
  if (!btn) return;

  btn.addEventListener("click", () => {
    // Read the currently-open file from app.js's global state
    const appState = window.psoEditor && window.psoEditor.state;
    const f = appState && appState.currentFile && appState.currentFile.name;
    if (!f) {
      // No file open - hint to user
      const status = $("#status");
      if (status) {
        status.textContent = "open a file first, then click view 3D";
        status.className = "status err";
      }
      return;
    }
    open(f);
  });

  $("#modelClose").addEventListener("click", close);
  document.addEventListener("keydown", (e) => {
    if (!$("#modelModal").hidden && e.key === "Escape") close();
  });

  $("#modelShapeSel").addEventListener("change", (e) => {
    state.shape = e.target.value;
    // Switching the shape selector means user wants a primitive view,
    // so disable the real-mesh path until they reopen. We mark the
    // overlay with a "user override" banner so it's still distinct
    // from a load failure (which would set state.lastMeshFailure).
    state.realMesh = false;
    setFallbackBanner(state.shape, "primitive selected by user");
    // Phantasmal-diff fix 5: route interactive changes through the
    // 10 ms trailing-edge throttle so rapid clicks don't thrash the GPU.
    scheduleRebuild();
    // Re-apply texture (UV setup may have changed)
    if (state.texture) {
      applyTextureToCurrentMesh(state.texture);
    }
  });

  $("#modelTileSel").addEventListener("change", (e) => {
    state.selectedTileIdx = parseInt(e.target.value, 10) || 0;
    loadTexture();
  });

  $("#modelUseUpscaled").addEventListener("change", (e) => {
    state.useUpscaled = !!e.target.checked;
    loadTexture();
  });

  $("#modelAutoRotate").addEventListener("change", (e) => {
    state.autoRotate = !!e.target.checked;
    // Turning ON starts the continuous loop; turning OFF lets it
    // self-stop after the next tick (which paints the final orientation).
    kick();
  });

  $("#modelWireframe").addEventListener("change", (e) => {
    state.wireframe = !!e.target.checked;
    // DEFECT #4: a multi-material mesh (every psov2 SkinnedMesh, the
    // skinned/world-baked groups) carries a material ARRAY, so
    // `mesh.material.wireframe = …` set a property on the Array object
    // and silently no-op'd. Walk the array (and the single-material
    // case) so the toggle reaches each material. setMaterialWireframe
    // also covers state.mesh when it's the SkinnedMesh directly (not a
    // Group), which is how the psov2 path assigns it.
    const setWire = (mat) => {
      if (!mat) return;
      if (Array.isArray(mat)) {
        mat.forEach((m) => { if (m) { m.wireframe = state.wireframe; m.needsUpdate = true; } });
      } else {
        mat.wireframe = state.wireframe;
        mat.needsUpdate = true;
      }
    };
    if (state.mesh) {
      if (state.mesh.isGroup) {
        state.mesh.traverse((c) => {
          if (c.isMesh) setWire(c.material);
        });
      } else {
        setWire(state.mesh.material);
      }
    }
    requestRender();
  });

  $("#modelRefreshTex").addEventListener("click", loadTexture);

  // Export model (+ textures) to a Blender-friendly format. The selected
  // format drives the POST; FBX is shown disabled in the dropdown.
  const exportBtn = $("#modelExportBtn");
  if (exportBtn) {
    exportBtn.addEventListener("click", exportModel);
  }

  // Debug-overlay toggle (also bound to D key when modal is focused).
  const debugToggle = $("#modelDebugToggle");
  if (debugToggle) {
    debugToggle.addEventListener("change", (e) => {
      setDebugMode(!!e.target.checked);
    });
  }
  // D key shortcut while the modal is open. We avoid hijacking when the
  // user is typing in an input.
  document.addEventListener("keydown", (e) => {
    if ($("#modelModal")?.hidden) return;
    const target = e.target;
    if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
    if (e.key === "d" || e.key === "D") {
      e.preventDefault();
      setDebugMode(!state.debugMode);
    }
  });
  // Pointer hover -> raycast the closest sub-mesh and show tooltip.
  // The canvas already captures pointermove for drag-to-rotate; we
  // attach a SECOND listener (compatible with PointerEvents semantics)
  // that runs on every move regardless of drag state.
  const cv = $("#modelCanvas");
  if (cv) {
    cv.addEventListener("pointermove", (e) => {
      if (!state.debugMode || !state.debugMeshes.length) {
        const tip = $("#modelDebugTooltip");
        if (tip) tip.hidden = true;
        return;
      }
      const idx = _debugRaycast(e.clientX, e.clientY);
      if (idx < 0) {
        const tip = $("#modelDebugTooltip");
        if (tip) tip.hidden = true;
        return;
      }
      const entry = state.debugMeshes[idx];
      // Show tooltip relative to the canvas's parent. The CSS positions
      // .model-debug-tooltip absolutely inside .model-stage; we need to
      // pass mouse coords relative to .model-stage to match. The stage's
      // bounding rect gives us the offset.
      const stage = cv.parentElement;
      const stageRect = stage.getBoundingClientRect();
      const localX = e.clientX - stageRect.left;
      const localY = e.clientY - stageRect.top;
      showDebugTooltip(entry, localX, localY);
    });
    cv.addEventListener("pointerleave", () => {
      const tip = $("#modelDebugTooltip");
      if (tip) tip.hidden = true;
    });
    cv.addEventListener("click", (e) => {
      if (!state.debugMode) return;
      // Don't fight drag-end: drag flag is set by mousemove with
      // active=true; the click handler still fires on simple click.
      const idx = _debugRaycast(e.clientX, e.clientY);
      if (idx >= 0) setDebugActiveMesh(idx);
    });
  }
  const filterInput = $("#modelDebugFilter");
  if (filterInput) {
    filterInput.addEventListener("input", (e) => {
      applyDebugFilter(e.target.value);
    });
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}

// fix/perf — idle-time preload wiring.
//
// (a) HOVER: delegate over the document so hovering a model row in the
//     asset tree warms its parsed-model cache before the click. We read
//     the model path from common data-attributes the tree/perspectives
//     rows carry (data-path / data-asset-path / title). Only `.nj` paths
//     are eligible; preloadPsov2Model() dedupes + bounds + idle-schedules,
//     so a fast mouse sweep can't flood.
// (b) POST-OPEN: when a model finishes opening we warm a couple of
//     likely-next models — the lifecycle bus emits the active path; the
//     tree's currently-rendered sibling rows are the best "next" guess, so
//     we preload the nearest visible model rows around the active one.
function _preloadPathFromEl(el) {
  if (!el || !el.getAttribute) return null;
  const p =
    el.getAttribute("data-model-path") ||
    el.getAttribute("data-path") ||
    el.getAttribute("data-asset-path") ||
    el.dataset && (el.dataset.modelPath || el.dataset.path || el.dataset.assetPath);
  if (p && p.toLowerCase().endsWith(".nj")) return p;
  return null;
}

function _wirePreloadHover() {
  if (window.__psoPreloadHoverWired) return;
  window.__psoPreloadHoverWired = true;
  let lastHover = 0;
  document.addEventListener(
    "pointerover",
    (e) => {
      // Cheap throttle: at most ~10 hover-preloads/sec.
      const now = (performance && performance.now) ? performance.now() : Date.now();
      if (now - lastHover < 90) return;
      let el = e.target;
      // Walk up a few levels to the row element that carries the path.
      for (let i = 0; i < 4 && el; i++) {
        const p = _preloadPathFromEl(el);
        if (p) { lastHover = now; preloadPsov2Model(p); return; }
        el = el.parentElement;
      }
    },
    { passive: true },
  );
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", _wirePreloadHover);
} else {
  _wirePreloadHover();
}

// =====================================================================
// Phase 2 (asset-router) hooks — additive surface for opening a model
// directly from the asset tree (rather than from a texture's "view 3D"
// button). Keeps the existing texture-driven flow untouched.
// =====================================================================

let _skeletonGroup = null;

/**
 * Open a model by its manifest path. The path is the same string the
 * asset tree emits via bus.asset.opened (e.g. "biri_ball.bml" or
 * "bm_ene_del_depth.bml#swim_body.nj").
 *
 * Flow:
 *   1. Use modelPath directly with /api/model_mesh — no detour through
 *      the texture-driven open() that would mis-resolve the archive.
 *   2. If matched_textures has entries, pick the highest-confidence
 *      one and drive the existing tile pipeline (window.openFile) so
 *      the texture wraps the real geometry. The texture override
 *      dropdown in the toolbar can swap it later.
 *   3. If anything fails, surface a clear status + fallback banner so
 *      the user can tell mesh-from-real vs. cube-fallback at a glance.
 *
 * matched_textures shape: [{path, rule, confidence}], sorted by
 * confidence desc by Agent 3 (the multi-rule matcher).
 */

// =====================================================================
// Multi-inner BML support (2026-04-25 — anchoring regression fix)
//
// A top-level `.bml` archive can pack multiple `.nj` inners. The asset
// tree emits the BML path; without an inner-picker we'd only render
// ONE inner (whichever the matched_texture R2 row pointed at), leaving
// multi-part bosses (De Rol Le with 7 NJ parts, etc.) appearing
// incomplete — the user reported this as "not loading full properly
// anchored assets". The fix: list every inner, classify each (primary
// vs LOD/shadow/destroyed-state), and offer a picker that defaults to
// "All parts" composited into one anchored scene.
//
// Classification heuristic (kept conservative; we'd rather show too
// many parts than too few):
//   - "lod"       : starts with `lo_` or `low_`  (low-poly LOD)
//   - "shadow"    : ends with `_sd` or `_shd`    (cast-shadow proxy)
//   - "destroyed" : contains `_break`, `_broken`, `_hahen`,
//                    `_burst`                     (parts that show only
//                                                  after damage states;
//                                                  shown by default
//                                                  because users want to
//                                                  see boss skulls etc.)
//   - "primary"   : everything else
//
// Composite mode loads every "primary" inner (NOT lod/shadow). The
// user can flip the picker to a specific inner to inspect alone.
// =====================================================================

const _LOD_RE = /^(lo|low)[_\s-]/i;
// Shadow proxies: PSOBB names them `_sd_` (mid-name, dragon) or `_sd` /
// `_shd` as a suffix. The regex matches both patterns; the left and
// right boundary characters keep us from matching e.g. `_sd...` inside
// a longer alphanumeric token.
const _SHADOW_RE = /(?:^|[_-])(?:sd|shd)(?:$|[_-])/i;
const _DESTROYED_RE = /(_break|_broken|_hahen|_burst)/i;

function _classifyInner(name) {
  const stem = name.replace(/\.(nj|xj)$/i, "");
  if (_LOD_RE.test(stem)) return "lod";
  if (_SHADOW_RE.test(stem)) return "shadow";
  if (_DESTROYED_RE.test(stem)) return "destroyed";
  return "primary";
}

/**
 * Fetch the BML's inner list and classify each `.nj`/`.xj` inner.
 * Returns ``{base, inners: [{name, kind}], primaries, destroyed}`` or null.
 */
async function _discoverBmlInners(bmlPath) {
  let entries;
  try {
    const r = await _lifecycleFetch(`/api/bml/${encodeURIComponent(bmlPath)}/list`);
    if (!r.ok) return null;
    entries = await r.json();
  } catch (_e) {
    // AbortError or network failure — caller treats null as "no inners",
    // viewer falls back gracefully.
    return null;
  }
  const inners = [];
  for (const e of entries.entries || []) {
    if (/\.(nj|xj)$/i.test(e.name)) {
      inners.push({ name: e.name, kind: _classifyInner(e.name) });
    }
  }
  return {
    base: bmlPath,
    inners,
    primaries: inners.filter((x) => x.kind === "primary").map((x) => x.name),
    destroyed: inners.filter((x) => x.kind === "destroyed").map((x) => x.name),
  };
}

/**
 * Populate the inner-picker dropdown. Hidden when fewer than 2 inners.
 * Default selection: "__all__" if 2+ primaries, else the single primary.
 */
function populateInnerPicker(info, currentSelection) {
  const wrap = $("#modelInnerPickWrap");
  const sel = $("#modelInnerPick");
  if (!wrap || !sel) return;
  if (!info || !info.inners || info.inners.length < 2) {
    wrap.hidden = true;
    sel.innerHTML = "";
    return;
  }
  const order = { primary: 0, destroyed: 1, lod: 2, shadow: 3 };
  const sorted = info.inners.slice().sort((a, b) => {
    const da = order[a.kind] ?? 99;
    const db = order[b.kind] ?? 99;
    if (da !== db) return da - db;
    return a.name.localeCompare(b.name);
  });
  sel.innerHTML = "";
  if (info.primaries.length >= 2) {
    const allOpt = document.createElement("option");
    allOpt.value = "__all__";
    allOpt.textContent = `All parts (${info.primaries.length} primary)`;
    sel.appendChild(allOpt);
  }
  for (const x of sorted) {
    const opt = document.createElement("option");
    opt.value = x.name;
    const stem = x.name.replace(/\.(nj|xj)$/i, "");
    const tag = x.kind === "primary" ? "" : ` [${x.kind}]`;
    opt.textContent = `${stem}${tag}`;
    sel.appendChild(opt);
  }
  let defaultValue;
  if (currentSelection) {
    defaultValue = currentSelection;
  } else if (info.primaries.length >= 2) {
    defaultValue = "__all__";
  } else if (info.primaries.length === 1) {
    defaultValue = info.primaries[0];
  } else {
    defaultValue = sorted[0]?.name;
  }
  if (defaultValue) sel.value = defaultValue;
  wrap.hidden = false;
  if (!sel.dataset.wired) {
    sel.dataset.wired = "1";
    sel.addEventListener("change", async (e) => {
      const cur = state.bmlInnersInfo;
      if (!cur) return;
      cur.current = e.target.value;
      await loadInnerSelection(cur);
    });
  }
  info.current = defaultValue;
}

/**
 * Dispatch loader based on the inner-picker selection.
 * "__all__" -> tryLoadCompositeBmlMesh; "<name>" -> single-inner path.
 */
async function loadInnerSelection(info) {
  if (!info || !info.current) return;
  const sel = info.current;
  if (sel === "__all__") {
    setStatus(`loading ${info.primaries.length} inner parts...`);
    const ok = await tryLoadCompositeBmlMesh(info.base, info.primaries);
    if (!ok) {
      setStatus(`composite load failed: ${state.lastMeshFailure || "unknown"}`);
    }
    return;
  }
  const path = `${info.base}#${sel}`;
  setStatus(`loading ${path}...`);
  const isNj = sel.toLowerCase().endsWith(".nj");
  let realLoaded = false;
  // psov2 faithful path first for .nj inners; legacy skinned as fallback.
  if (isNj) realLoaded = await tryLoadPsov2NinjaModel(path, null);
  if (isNj && !realLoaded) realLoaded = await tryLoadSkinnedMesh(path, null);
  if (!realLoaded) {
    realLoaded = await tryLoadRealMesh({ path });
    const animBar = $("#modelAnimBar");
    if (animBar) animBar.hidden = true;
    state.anim.skinned = false;
  }
  if (!realLoaded) {
    setStatus(`mesh load failed: ${state.lastMeshFailure || "unknown error"}`);
  }
}

/**
 * Composite-mode loader. Fetches /api/model_mesh for every inner in
 * `innerNames` (parallel) and assembles ONE THREE.Group containing
 * every submesh. Each inner keeps its native pre-transformed coords;
 * we compute ONE union AABB across all inners and apply ONE global
 * center+scale to the parent group so the whole composite fits the
 * camera viewport without distorting inter-part spatial relationships
 * (which would happen if each inner was scaled independently).
 *
 * Returns true on success (≥1 inner produced visible geometry).
 */
// Fetch the curated composite assembly metadata from
// /api/composite_bundle/<bml> (added 2026-04-30). Returns:
//   { parts: [...], source, fallback } on 200,
//   null on 404 (endpoint missing) or any parse / network error.
//
// We deliberately fall back silently on ANY failure so this stays a
// pure enhancement: if the endpoint is gone, the legacy origin-stack
// composite still renders.
async function _fetchCompositeAssembly(bmlPath) {
  // meta_only=1: we consume ONLY the per-part placement table (inner /
  // pos / rot_euler / scale / parent_inner) below — the geometry is
  // re-fetched per inner in parallel via /api/model_skinned + model_mesh.
  // Without this flag the server parsed every inner's skinned mesh inline
  // (13-60+ s on a multi-inner boss) only for us to throw it away, which
  // blocked the whole composite open. (2026-06-20 perf.)
  const url = `/api/composite_bundle/${encodeURIComponent(bmlPath)}?meta_only=1`;
  try {
    const r = await _lifecycleFetch(url);
    if (!r.ok) return null;
    const j = await r.json();
    if (!j || !Array.isArray(j.parts) || j.parts.length === 0) return null;
    return j;
  } catch (_e) {
    return null;
  }
}

async function tryLoadCompositeBmlMesh(bmlPath, innerNames) {
  if (!bmlPath || !innerNames || innerNames.length === 0) {
    _setMeshFailure("composite: no inners to load");
    return false;
  }
  ensureRenderer();

  // 2026-04-30: ask the backend for a curated per-part TRS table BEFORE
  // we issue the per-inner mesh fetches. When present, this gives us
  //   * authoritative inner ORDER
  //   * per-part {pos, rot_euler, scale, parent_inner}
  // The endpoint reuses /api/model_skinned per part so its
  // skinned/binding payloads are bone-local — for the asset preview
  // path we still want world-baked verts via /api/model_mesh, so we
  // use composite_bundle ONLY for the placement metadata and keep the
  // existing /api/model_mesh fetch loop for the actual geometry. On a
  // 404 / hand-curated miss, ``assembly`` is null and we render the
  // legacy origin-stacked composite so we never regress single-inner
  // models or unknown bosses.
  const assembly = await _fetchCompositeAssembly(bmlPath);
  // When the composite table is hand-curated for THIS BML, use its
  // inner order; otherwise stick with the discovery-order names.
  // ``identity-fallback`` is treated as "no curated layout" — no point
  // running the TRS code path with all-zero offsets.
  const curatedAssembly =
    assembly && assembly.source && assembly.source !== "identity-fallback"
      ? assembly
      : null;
  let effectiveInnerNames = innerNames;
  if (curatedAssembly) {
    // Filter to parts whose inner is in the requested set so a curator
    // adding extra slots doesn't break a viewer that only knew about a
    // subset (e.g. damage-state inners surfaced by the curated table
    // but not in ``innerNames`` — render them).
    const requested = new Set(innerNames.map((n) => n.toLowerCase()));
    const curatedNames = curatedAssembly.parts.map((p) => p.inner);
    // Union: every requested inner first (preserves discovery order
    // for un-curated inners), then any curated extras.
    const seen = new Set();
    effectiveInnerNames = [];
    for (const n of innerNames) {
      const k = n.toLowerCase();
      if (seen.has(k)) continue;
      seen.add(k);
      effectiveInnerNames.push(n);
    }
    for (const cn of curatedNames) {
      const k = cn.toLowerCase();
      if (seen.has(k)) continue;
      seen.add(k);
      // Only add curated extras if they weren't requested-but-missing
      // (we still fetch them — caller may have under-discovered).
      effectiveInnerNames.push(cn);
      requested.add(k); // for future-proofing
    }
  }

  // Detect the PRIMARY inner for skinned-anim playback (2026-04-30).
  // The primary is the curated part with `parent_inner === null` —
  // typically the body / centerpiece that drives animation. If multiple
  // parts have null parent we pick the FIRST in array order so the
  // curator's intent is honoured. For non-curated composites we leave
  // `primaryInnerLower` null and the skinned fetch is skipped (every
  // part renders world-baked, no animation — same as before).
  //
  // Only `.nj` inners are eligible: /api/model_skinned requires bone-
  // local data which doesn't exist in `.xj` strips. An .xj primary
  // means we silently fall back to world-baked rendering for that
  // inner too. (This affects nothing today — every curated primary in
  // composite_assembly.py is an .nj.)
  let primaryInnerLower = null;
  if (curatedAssembly) {
    for (const p of curatedAssembly.parts) {
      if (p.parent_inner) continue;
      const lower = (p.inner || "").toLowerCase();
      if (!lower.endsWith(".nj")) continue;
      primaryInnerLower = lower;
      break;
    }
  }

  const fetches = effectiveInnerNames.map(async (inner) => {
    const lower = inner.toLowerCase();
    const isPrimary = (lower === primaryInnerLower);
    // Primary: fetch the SKINNED variant so the renderer can drive
    // animation playback against the inner's bone-local data. Other
    // inners stay on the world-baked /api/model_mesh path because they
    // are static breakable bits / phase-2 props / cosmetic accessories
    // that the engine's per-part TRS already places correctly.
    //
    // Fallback chain on the primary: if /api/model_skinned 4xx/5xx's
    // (rare — typically means the .nj parser couldn't find a bone
    // tree, which shouldn't happen for a curated primary), retry via
    // /api/model_mesh so the primary at least renders STATICALLY.
    // The composite path below sees `payload.bones` is missing and
    // falls through to the world-baked branch — animation is lost
    // for this BML but the silhouette still draws.
    const skinnedUrl = `/api/model_skinned/${encodeURIComponent(bmlPath)}?inner=${encodeURIComponent(inner)}`;
    const meshUrl = `/api/model_mesh/${encodeURIComponent(bmlPath)}?inner=${encodeURIComponent(inner)}`;
    const fetchOnce = async (u) => {
      const r = await _lifecycleFetch(u);
      if (!r.ok) throw new Error(`http ${r.status}`);
      return await r.json();
    };
    try {
      if (isPrimary) {
        try {
          const payload = await fetchOnce(skinnedUrl);
          return { inner, payload, url: skinnedUrl, isPrimary: true };
        } catch (skinErr) {
          if (_isAbortError(skinErr)) {
            return { inner, payload: null, error: "aborted", aborted: true, isPrimary: true };
          }
          // Fall through to /api/model_mesh — primary stays static.
          const payload = await fetchOnce(meshUrl);
          return { inner, payload, url: meshUrl, isPrimary: false };
        }
      } else {
        const payload = await fetchOnce(meshUrl);
        return { inner, payload, url: meshUrl, isPrimary: false };
      }
    } catch (e) {
      if (_isAbortError(e)) return { inner, payload: null, error: "aborted", aborted: true, isPrimary };
      return { inner, payload: null, error: String(e?.message || e), isPrimary };
    }
  });
  const results = await Promise.all(fetches);

  // Phantasmal-style per-inner tex-id shift (2026-04-26). The backend's
  // `_build_model_texture_binding` emits `inner_tex_offsets` on each
  // inner's binding_data — a list of `{name, tile_count,
  // cumulative_offset}` covering EVERY inner in the BML. We capture
  // it from the first non-empty payload so the client can verify
  // (or, if needed, re-derive) shifts for cross_bml fallbacks.
  // Reference: Phantasmal `CharacterClassAssetLoader.kt:88-98` —
  // `shiftTextureIds(njObject, shift)` walks each inner's mesh tree
  // adding the offset to every `mesh.textureId` so inner-N's tile-IDs
  // don't alias inner-0's.
  let innerTexOffsets = [];
  for (const r of results) {
    const bd = r?.payload?.binding_data || {};
    if (Array.isArray(bd.inner_tex_offsets) && bd.inner_tex_offsets.length > 0) {
      innerTexOffsets = bd.inner_tex_offsets;
      break;
    }
  }
  // Build a quick name→offset map for sanity checks below.
  const offsetByName = new Map();
  for (const o of innerTexOffsets) {
    offsetByName.set(o.name, { offset: o.cumulative_offset | 0, count: o.tile_count | 0 });
  }

  const innerData = [];
  for (const r of results) {
    if (!r.payload || !r.payload.mesh_count) continue;
    const newBinding = r.payload.binding || [];
    let boundTex = new Map();
    let archive = null;
    if (newBinding.length > 0) {
      archive = deriveTextureArchivePath(r.url, null, null);
      if (archive) {
        try { boundTex = await fetchBoundTextures(archive, newBinding); } catch (_e) { /* skip */ }
      }
    }
    const offEntry = offsetByName.get(r.inner) || { offset: 0, count: 0 };
    // isPrimary tracks whether THIS inner came from /api/model_skinned
    // (a bone-tree payload with `bones` + per-vertex `bone_indices_b64`)
    // vs /api/model_mesh (world-baked verts, no bones). The build loop
    // below dispatches on this so the primary can drive animation while
    // siblings stay static.
    innerData.push({
      inner: r.inner, payload: r.payload, boundTex, archive,
      binding: newBinding,
      texShift: offEntry.offset,
      texCount: offEntry.count,
      isPrimary: !!r.isPrimary,
    });
  }
  if (innerData.length === 0) {
    _setMeshFailure(`composite: no inner parsed (tried ${innerNames.length})`);
    return false;
  }

  const compositeGroup = new THREE.Group();
  compositeGroup.name = "bml_composite";
  const allDebugMeshes = [];
  const aggregateBoundTextures = new Map();
  let totalVerts = 0;
  let totalTris = 0;
  const aabbMin = [Infinity, Infinity, Infinity];
  const aabbMax = [-Infinity, -Infinity, -Infinity];

  // 2026-04-30: build a part-by-inner-name lookup for the curated TRS
  // table. Names are matched case-insensitively because the BML file
  // table preserves the on-disk casing whereas /api/composite_bundle
  // emits whatever `composite_assembly.py` was authored with. Both
  // sources have agreed casing for every curated boss today, but the
  // lower-case compare future-proofs against a curator typo.
  const partsByInner = new Map();
  if (curatedAssembly) {
    for (const p of curatedAssembly.parts) {
      partsByInner.set(p.inner.toLowerCase(), p);
    }
  }
  // We need the inner-name -> innerGroup map AFTER the loop builds
  // every group, so parent_inner reparenting can find its target.
  const innerGroupByName = new Map();
  // 2026-04-30: when a primary inner is animated, we keep its
  // skinSubmeshes / bones / payload here so the post-loop wiring step
  // can hand them to state.anim.* and the render-loop animation tick
  // re-bakes vertices each frame. `null` means no primary-skinned
  // inner — composite stays static (legacy behaviour).
  let primarySkinned = null;

  for (let ii = 0; ii < innerData.length; ii++) {
    const { inner, payload, boundTex, isPrimary } = innerData[ii];
    // Per-inner material_ids are NOT globally unique across a composite;
    // we namespace them with the inner index so post-load lookups don't
    // cross-bind textures (see longer comment further down).
    const stampMeshUserData = (mesh, matId) => {
      mesh.userData.materialId = matId;
      mesh.userData.innerName = inner;
      mesh.userData.innerIndex = ii;
      mesh.userData.compositeKey = `${ii}:${matId}`;
    };

    let innerGroup;
    if (isPrimary && payload && payload.bones && payload.mesh_count) {
      // SKINNED primary: build via the existing skinned-payload helper
      // so we get back per-submesh bone-local snapshots and a
      // skinSubmeshes array the per-frame re-bake can chew through.
      // Then RESET the helper's auto-applied centering — it normally
      // recentres+scales the group to fit a unit AABB for single-model
      // rendering, but the compositeGroup does aggregate centering at
      // a higher level so we don't want a double transform here.
      const built = buildSkinnedMeshGroupFromPayload(payload, boundTex);
      innerGroup = built.group;
      innerGroup.name = inner;
      innerGroup.position.set(0, 0, 0);
      innerGroup.scale.set(1, 1, 1);
      // Stamp composite metadata onto each Mesh so the inspector / lookup
      // helpers treat skinned submeshes the same as world-baked ones.
      innerGroup.traverse((child) => {
        if (child.isMesh) {
          stampMeshUserData(child, child.userData.materialId | 0);
        }
      });
      // Stash for state.anim wiring after the loop. Note: the helper
      // already drove a one-pass bind-pose re-bake so ``built.aabbMin``
      // / ``built.aabbMax`` are valid world-space bounds for THIS
      // inner alone (pre-curated-TRS).
      primarySkinned = {
        ii,
        inner,
        payload,
        skinSubmeshes: built.skinSubmeshes,
        bones: payload.bones,
      };
      // Adopt the helper's debug-mesh entries (one per submesh) into
      // the composite-wide allDebugMeshes list, with composite metadata.
      // We do NOT push the helper's `debugMeshes` directly because their
      // `idx` is bone-locally indexed; reassigning lets us walk the
      // composite as one flat list.
      for (const dm of built.debugMeshes || []) {
        allDebugMeshes.push({
          idx: allDebugMeshes.length,
          mesh: dm.mesh,
          material_id: dm.material_id,
          vertex_count: dm.vertex_count,
          triangle_count: dm.triangle_count,
          world_position: dm.world_position || [0, 0, 0],
          world_rotation_euler: dm.world_rotation_euler || [0, 0, 0],
          world_scale: dm.world_scale || [1, 1, 1],
          bounding_sphere: dm.bounding_sphere || [0, 0, 0, 0],
          aabb: dm.aabb || null,
          eval_flags: dm.eval_flags || 0,
          inner,
        });
      }
      totalVerts += built.totalVerts | 0;
      totalTris += built.totalTris | 0;
      // Fold the skinned inner's world-baked AABB into the composite
      // union (pre-curated-TRS — the post-loop Box3 walk recomputes
      // the final assembled bound).
      if (Number.isFinite(built.aabbMin?.[0])) {
        if (built.aabbMin[0] < aabbMin[0]) aabbMin[0] = built.aabbMin[0];
        if (built.aabbMin[1] < aabbMin[1]) aabbMin[1] = built.aabbMin[1];
        if (built.aabbMin[2] < aabbMin[2]) aabbMin[2] = built.aabbMin[2];
        if (built.aabbMax[0] > aabbMax[0]) aabbMax[0] = built.aabbMax[0];
        if (built.aabbMax[1] > aabbMax[1]) aabbMax[1] = built.aabbMax[1];
        if (built.aabbMax[2] > aabbMax[2]) aabbMax[2] = built.aabbMax[2];
      }
    } else {
      // WORLD-BAKED inner (no animation; static). Inline the geometry
      // build the same way we always have. Per-mesh `world_matrix` is
      // already pre-applied server-side, so this is just an
      // attribute-array reformat + material wire-up.
      innerGroup = new THREE.Group();
      innerGroup.name = inner;
      for (const m of payload.meshes || []) {
        const vbuf = b64ToArrayBuffer(m.vertices_b64);
        const ibuf = b64ToArrayBuffer(m.indices_b64);
        const verts = new Float32Array(vbuf);
        const indices = new Uint32Array(ibuf);
        if (verts.length === 0 || indices.length === 0) continue;
        // v2 payloads carry 4 trailing RGBA floats (12-float stride).
        const hasColor = payload.has_color === true;
        const stride = hasColor ? 12 : 8;
        const vertexCount = verts.length / stride;
        if (!Number.isInteger(vertexCount)) continue;
        const positions = new Float32Array(vertexCount * 3);
        const normals = new Float32Array(vertexCount * 3);
        const uvs = new Float32Array(vertexCount * 2);
        const colors = hasColor ? new Float32Array(vertexCount * 4) : null;
        for (let i = 0; i < vertexCount; i++) {
          const o = i * stride;
          positions[i * 3 + 0] = verts[o + 0];
          positions[i * 3 + 1] = verts[o + 1];
          positions[i * 3 + 2] = verts[o + 2];
          normals[i * 3 + 0] = verts[o + 3];
          normals[i * 3 + 1] = verts[o + 4];
          normals[i * 3 + 2] = verts[o + 5];
          uvs[i * 2 + 0] = verts[o + 6];
          uvs[i * 2 + 1] = verts[o + 7];
          if (colors) {
            colors[i * 4 + 0] = verts[o + 8];
            colors[i * 4 + 1] = verts[o + 9];
            colors[i * 4 + 2] = verts[o + 10];
            colors[i * 4 + 3] = verts[o + 11];
          }
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geo.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
        geo.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
        if (colors) geo.setAttribute("color", new THREE.BufferAttribute(colors, 4));
        geo.setIndex(new THREE.BufferAttribute(indices, 1));
        geo.computeBoundingSphere();
        const matId = (m.material_id | 0);
        const submeshTex = boundTex.has(matId) ? boundTex.get(matId) : (state.texture || null);
        // Phantasmal-diff fixes 3+4 (2026-04-25): Basic for textured,
        // Lambert for un-textured, transparent=false default.
        // side=DoubleSide is the safe default — see the long comment
        // in `buildMeshGroupFromPayload` above for the full rationale.
        // Override per-material via `psoUpdateMaterial(idx, {two_sided:
        // false})` when an authoring-intent flag says the strip is
        // single-sided.
        // Phase 3 (2026-06-20): unlit MeshBasicMaterial for BOTH branches
        // (psov2 parity) + vertexColors so the per-vertex/diffuse color
        // shades the mesh. The Lambert un-textured branch washed to white.
        const mat = submeshTex
          ? new THREE.MeshBasicMaterial({
              map: submeshTex,
              color: 0xffffff,
              vertexColors: hasColor,
              wireframe: state.wireframe,
              side: THREE.DoubleSide,
              transparent: false,
            })
          : new THREE.MeshBasicMaterial({
              color: 0xffffff,
              vertexColors: hasColor,
              wireframe: state.wireframe,
              side: THREE.DoubleSide,
              transparent: false,
            });
        // Phase 3 (2026-06-20): apply per-submesh blend/alpha/two-sided flags.
        applyPsoMaterialFlags(mat, m);
        const mesh = new THREE.Mesh(geo, mat);
        // Composite-mode bug fix (2026-04-30): per-inner material_ids are
        // NOT globally unique across the BML — inner-A's matId=0 collides
        // with inner-B's matId=0 in any post-load lookup that walks meshes
        // by `userData.materialId`. We stash BOTH the bare per-inner matId
        // (kept for inspector/debug compatibility) AND a globally-namespaced
        // composite key matching `state.boundTextures` keys (`${ii}:${mid}`,
        // line ~2269 below). Lookup helpers (psoGetMaterialTexture,
        // psoSetMaterialTexture, psoReloadTexture, applyTextureToCurrentMesh)
        // use `userData.compositeKey` when present so they stop cross-binding
        // textures across inners.
        stampMeshUserData(mesh, matId);
        innerGroup.add(mesh);

        allDebugMeshes.push({
          idx: allDebugMeshes.length,
          mesh,
          material_id: matId,
          vertex_count: vertexCount,
          triangle_count: (indices.length / 3) | 0,
          world_position: m.world_position || [0, 0, 0],
          world_rotation_euler: m.world_rotation_euler || [0, 0, 0],
          world_scale: m.world_scale || [1, 1, 1],
          bounding_sphere: m.bounding_sphere || [0, 0, 0, 0],
          aabb: m.aabb || null,
          eval_flags: m.eval_flags || 0,
          inner: inner,
        });
        totalVerts += vertexCount;
        totalTris += indices.length / 3;
        // Vertices are pre-transformed (post-2026-04-24 server) — the
        // per-mesh aabb is already in this inner's native world coords;
        // no per-mesh world_matrix transform needed for the union AABB.
        if (m.aabb && m.aabb.length === 6) {
          if (m.aabb[0] < aabbMin[0]) aabbMin[0] = m.aabb[0];
          if (m.aabb[1] < aabbMin[1]) aabbMin[1] = m.aabb[1];
          if (m.aabb[2] < aabbMin[2]) aabbMin[2] = m.aabb[2];
          if (m.aabb[3] > aabbMax[0]) aabbMax[0] = m.aabb[3];
          if (m.aabb[4] > aabbMax[1]) aabbMax[1] = m.aabb[4];
          if (m.aabb[5] > aabbMax[2]) aabbMax[2] = m.aabb[5];
        }
      }
    }

    innerGroupByName.set(inner.toLowerCase(), innerGroup);
    // Apply curated per-part TRS so boss parts no longer stack at the
    // origin. ZYX intrinsic order matches Phantasmal / Sega Ninja
    // (NJD_EVAL_ZYX_ANG default; ZXY is opt-in via flag bit 0x20).
    // parent_inner reparenting happens AFTER the loop so we can resolve
    // a forward reference (curator may list parent below child).
    const part = curatedAssembly ? partsByInner.get(inner.toLowerCase()) : null;
    if (part) {
      const [px, py, pz] = part.pos || [0, 0, 0];
      const [rx, ry, rz] = part.rot_euler || [0, 0, 0];
      const [sx, sy, sz] = part.scale || [1, 1, 1];
      innerGroup.position.set(px, py, pz);
      innerGroup.rotation.order = "ZYX";
      innerGroup.rotation.set(rx, ry, rz);
      innerGroup.scale.set(sx, sy, sz);
      innerGroup.userData.compositePart = {
        parent_inner: part.parent_inner || null,
        notes: part.notes || "",
      };
    }
    compositeGroup.add(innerGroup);
    for (const [mid, tex] of boundTex) {
      aggregateBoundTextures.set(`${ii}:${mid}`, tex);
    }
  }

  // 2026-04-30: parent_inner reparenting. After every innerGroup has
  // been created and attached to compositeGroup with its absolute TRS,
  // walk the curated parts in order and re-attach any group whose
  // parent_inner names a sibling group. THREE.Group.add() removes the
  // child from its current parent automatically, so the only thing we
  // need to be careful about is preserving the LOCAL TRS we already
  // wrote (Object3D.add does NOT re-bake world->local). For our
  // hand-curated table the offsets are authored as parent-relative
  // already, so re-parenting just gives the engine a hierarchical
  // transform stack — no math fix-up needed.
  if (curatedAssembly) {
    for (const p of curatedAssembly.parts) {
      if (!p.parent_inner) continue;
      const child = innerGroupByName.get(p.inner.toLowerCase());
      const parent = innerGroupByName.get(p.parent_inner.toLowerCase());
      if (child && parent && child.parent !== parent) {
        parent.add(child);
      }
    }
  }

  if (compositeGroup.children.length === 0 ||
      compositeGroup.children.every((g) => {
        // After reparenting some children move into other groups; an
        // intermediate Group with reparented descendants still counts.
        let n = 0;
        g.traverse((c) => { if (c.isMesh) n++; });
        return n === 0;
      })) {
    _setMeshFailure("composite: every inner produced 0 sub-meshes");
    return false;
  }

  // Per-part TRS bends inner-local AABBs through arbitrary rotations
  // and parent chains, so the union we built from raw inner-local
  // mesh.aabb arrays is no longer a valid bound. When the curated
  // table supplied transforms, recompute via THREE.Box3 so the
  // viewport-fit math sees the assembled silhouette. Falls back to
  // the manual union for non-curated paths (cheaper, identical result).
  let finalAabbMin = aabbMin;
  let finalAabbMax = aabbMax;
  if (curatedAssembly) {
    // Force matrix update so Box3.setFromObject sees the TRS we just
    // wrote (auto-update happens at render time, after the AABB calc
    // here would otherwise read identity matrices).
    compositeGroup.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(compositeGroup);
    if (isFinite(box.min.x) && isFinite(box.max.x)) {
      finalAabbMin = [box.min.x, box.min.y, box.min.z];
      finalAabbMax = [box.max.x, box.max.y, box.max.z];
    }
  }

  if (Number.isFinite(finalAabbMin[0])) {
    const cx = (finalAabbMin[0] + finalAabbMax[0]) / 2;
    const cy = (finalAabbMin[1] + finalAabbMax[1]) / 2;
    const cz = (finalAabbMin[2] + finalAabbMax[2]) / 2;
    const dx = finalAabbMax[0] - finalAabbMin[0];
    const dy = finalAabbMax[1] - finalAabbMin[1];
    const dz = finalAabbMax[2] - finalAabbMin[2];
    const maxDim = Math.max(dx, dy, dz, 0.001);
    const scale = 2.0 / maxDim;
    compositeGroup.position.set(-cx * scale, -cy * scale, -cz * scale);
    compositeGroup.scale.set(scale, scale, scale);
    aabbMin[0] = finalAabbMin[0]; aabbMin[1] = finalAabbMin[1]; aabbMin[2] = finalAabbMin[2];
    aabbMax[0] = finalAabbMax[0]; aabbMax[1] = finalAabbMax[1]; aabbMax[2] = finalAabbMax[2];
  }

  disposeMesh();
  state.boundTextures = aggregateBoundTextures;
  state.boundTextureArchive = `${bmlPath}#__composite__.xvm`;
  state.boundBinding = [];
  state.mesh = compositeGroup;
  state.meshGroup = compositeGroup;
  state.realMesh = true;
  state.scene.add(state.mesh);
  state.debugMeshes = allDebugMeshes;
  state.debugActiveIdx = -1;
  if (typeof rebuildDebugSidebar === "function") rebuildDebugSidebar();
  if (state.debugMode && typeof applyDebugMaterials === "function") applyDebugMaterials(true);
  // 2026-04-30: composite-mode animation playback for the PRIMARY inner.
  // When a curated layout designated a primary (`parent_inner === null`,
  // .nj-only) we fetched it via /api/model_skinned and stashed the bone
  // table + skinSubmeshes on `primarySkinned`. Wire those into
  // state.anim so the render-loop tick re-bakes vertices each frame
  // exactly the way a single-inner skinned model would.
  //
  // Static parts (helms, fins, phase-2 props, etc.) stay world-baked —
  // they have no skeleton in the engine either, so freezing them at
  // bind pose under the primary's animated parent is the engine-correct
  // behaviour. Their per-inner Group still receives the curated TRS, so
  // they translate / rotate with the primary even though they don't
  // articulate themselves.
  //
  // No primary => composite stays static (legacy behaviour). The
  // animation panel still populates so the motion catalog is visible;
  // the user can switch to a single inner via the picker to play one.
  if (primarySkinned) {
    state.anim.skinned = true;
    state.anim.modelPath = `${bmlPath}#${primarySkinned.inner}`;
    state.anim.bones = primarySkinned.bones;
    state.anim.skinSubmeshes = primarySkinned.skinSubmeshes;
    state.anim.currentMotion = null;
    state.anim.currentData = null;
    state.anim.time = 0.0;
    state.anim.playing = false;
    state.anim.lastTimestamp = 0;
  } else {
    state.anim.skinned = false;
    state.anim.playing = false;
  }
  const animBar = $("#modelAnimBar");
  // populateAnimationPanel will un-hide the bar if motions exist; we
  // keep the legacy default-hide here so a BML without motions doesn't
  // briefly flash the empty bar before the panel population call.
  if (animBar) animBar.hidden = true;
  // Best-effort discovery: populate the animation panel for the primary
  // inner if we have one, otherwise fall back to the first inner so the
  // motion catalog shows up regardless of curated metadata. The
  // server's /api/animations walks every .njm in the BML regardless of
  // which .nj/.xj inner is asked for, so the populated dropdown lists
  // EVERY motion in the BML — but loadMotion() can only drive playback
  // when state.anim.skinned is true (which now happens only for the
  // primary inner in composite mode).
  if (innerData.length > 0) {
    const animTarget = primarySkinned
      ? `${bmlPath}#${primarySkinned.inner}`
      : `${bmlPath}#${innerData[0].inner}`;
    try {
      populateAnimationPanel(animTarget);
    } catch (_e) { /* swallow — population is best-effort */ }
  }
  kick();

  const aabbDx = aabbMax[0] - aabbMin[0];
  const aabbDy = aabbMax[1] - aabbMin[1];
  const aabbDz = aabbMax[2] - aabbMin[2];
  const layoutTag = curatedAssembly ? ` [layout: ${curatedAssembly.source}]` : "";
  setMeshStats(
    `composite ${innerData.length}/${effectiveInnerNames.length} inners${layoutTag}  ` +
    `verts ${totalVerts}  tris ${totalTris}  ` +
    `aabb ${aabbDx.toFixed(1)}x${aabbDy.toFixed(1)}x${aabbDz.toFixed(1)}`,
  );
  setStatus(
    `composite ${bmlPath}: ${innerData.length} parts loaded${layoutTag} ` +
    `(${innerData.map((d) => d.inner.replace(/\.nj$/i, "")).join(", ")})`,
  );
  _setMeshFailure(null);
  state.realMeshArchive = bmlPath;
  return true;
}

async function openByPath(modelPath, _entry, matchedTextures) {
  // Wave 7: ensure the asset lifecycle epoch is bumped for THIS model
  // open. asset_router.dispatch already calls beginAsset() in the
  // typical path, but openByPath is also invoked directly (variant
  // prefetches, motion override, "view as model" banner) and those
  // call sites need the same abort-prior-fetches semantics so a
  // mid-flight prior model doesn't keep racing the new one.
  if (window.psoAssetLifecycle &&
      window.psoAssetLifecycle.path() !== modelPath) {
    try { window.psoAssetLifecycle.beginAsset(modelPath); } catch (_e) {}
  }
  // fix/perf — a user-driven open takes priority over idle preloads:
  // cancel any in-flight preload + drain the queue so the network/CPU go
  // to THIS open. (If we're opening a model that was being preloaded, the
  // cache may already be warm and the cold path is skipped entirely.)
  try { _psov2PreloadAbortAll(); } catch (_e) {}
  // Pick the highest-confidence matched texture (manifest sorts by
  // confidence desc, so [0] is the best); we'll drive the regular
  // tile pipeline against it after the mesh loads.
  const tex = (matchedTextures || []).find((m) => {
    const p = (m.path || "").toLowerCase();
    // Accept both raw .xvm/.prs textures and BML-inner "<bml>#<inner>.nj.xvm"
    // forms. The texture itself can be a `#`-path; the tile pipeline
    // handles that via _materialize_inner_for_extract on the backend.
    return p.endsWith(".xvm") || p.endsWith(".prs") || p.endsWith(".nj.xvm");
  });
  const texPath = tex ? tex.path : null;

  // Resolve the actual mesh path. If the user clicked a top-level
  // `.bml` (entry.path has no `#`), the manifest doesn't tell us
  // which inner `.nj` to load — so we infer it from the highest-
  // confidence matched_texture. Agent 3 R2 emits matched textures of
  // the form `<bml>#<inner>.nj.xvm`; the corresponding model is the
  // same path with `.xvm` stripped (`<bml>#<inner>.nj`).
  //
  // Without this step, the previous code path detected `.bml`, found
  // no `#`, and passed it to /api/model_mesh which 400s — falling back
  // to primitive cube ("the bug"). Inferring the inner gives us a
  // concrete `<bml>#<inner>.nj` which renders correctly.
  let resolvedMeshPath = modelPath;
  const lowerModel = modelPath.toLowerCase();
  if (lowerModel.endsWith(".bml") && !modelPath.includes("#")) {
    // Try to infer inner from a matched_texture whose path has the
    // expected `<sameBml>#<inner>.nj.xvm` shape.
    const innerHint = (matchedTextures || []).find((m) => {
      const p = (m.path || "");
      const lo = p.toLowerCase();
      return p.startsWith(modelPath + "#") && lo.endsWith(".nj.xvm");
    });
    if (innerHint) {
      // Drop `.xvm` to get the `.nj` inner; result is `<bml>#<inner>.nj`.
      resolvedMeshPath = innerHint.path.slice(0, -4);
    }
  }

  // Unify modal lifecycle: every path-driven open shows the model path
  // as the title, regardless of whether a texture is paired.
  state.filename = texPath ? texPath.split("/").pop() : modelPath;
  state.realMesh = false;
  setMeshStats(null);
  _setMeshFailure(null);
  $("#modelModal").hidden = false;
  $("#modelModalTitle").textContent = modelPath;
  $("#modelModalMeta").textContent = texPath
    ? `model · texture: ${texPath}`
    : "model · no matched texture";
  const meshHintMsg = (resolvedMeshPath !== modelPath)
    ? ` <span class="dim">(inner inferred: ${escapeHtml(resolvedMeshPath)})</span>`
    : "";
  $("#modelHint").innerHTML =
    `<strong>model</strong> &mdash; ${escapeHtml(modelPath)}.${meshHintMsg}` +
    `<br/><span class="dim">drag to rotate, scroll to zoom.</span>`;
  setStatus(`loading model ${resolvedMeshPath}...`);
  ensureRenderer();

  // Bundle prefetch: kicks off the consolidated /api/model_bundle GET
  // in parallel with three.js renderer setup. tryLoadSkinnedMesh /
  // populateAnimationPanel consult the cache before issuing their own
  // fetches. We use the RESOLVED mesh path so the cache key matches
  // the path those helpers will pass.
  prefetchModelBundle(resolvedMeshPath);

  // Step 0 (2026-04-25 — multi-inner anchoring fix): when the user
  // clicks a top-level `.bml`, discover whether it packs multiple `.nj`
  // inners. Multi-part bosses (De Rol Le with body+helm+fins+sting+
  // tentacle, etc.) need ALL primaries composited so the user sees the
  // full anchored boss, not just the matched-texture-inferred body.
  // The picker dropdown lets the user switch to a single inner anytime.
  state.bmlInnersInfo = null;
  const inputIsBml = lowerModel.endsWith(".bml") && !modelPath.includes("#");
  let composited = false;
  let realLoaded = false;
  if (inputIsBml) {
    const info = await _discoverBmlInners(modelPath);
    if (info && info.inners.length >= 1) {
      // The inner-picker only makes sense with 2+ inners.
      if (info.inners.length >= 2) {
        state.bmlInnersInfo = info;
        populateInnerPicker(info, null);
      }
      // Auto-pick: composite ONLY for 2+ primaries (multi-part bosses).
      // Otherwise resolve to a SINGLE inner `.nj` so it routes through the
      // psov2 loader below (correct geometry) instead of the bare-`.bml`
      // world-baked path (mangled). This fixes single-inner NPC bodies
      // like momoka: previously inners.length===1 hit NEITHER branch, so
      // resolvedMeshPath stayed the bare `.bml` (isNj false) and never
      // reached psov2.
      if (info.primaries.length >= 2) {
        info.current = "__all__";
        composited = await tryLoadCompositeBmlMesh(info.base, info.primaries);
        realLoaded = composited;
      } else {
        const pick = info.primaries[0] || info.inners[0].name;
        resolvedMeshPath = `${modelPath}#${pick}`;
        info.current = pick;
      }
    }
  } else {
    // Hide picker on non-BML opens (or BML#inner direct paths).
    const wrap = $("#modelInnerPickWrap");
    if (wrap) wrap.hidden = true;
  }

  // Step 1: load the mesh DIRECTLY by path (skipped if composite mode
  // already loaded a multi-inner scene above). This is the fix for the
  // primitive-cube bug: the previous flow handed control to open() with
  // the texture filename, which meant the hint pipeline was answering
  // "what shape pairs with this texture" (cube) instead of "render this
  // exact model". By calling tryLoadRealMesh({path: resolvedMeshPath})
  // we skip the hint entirely.
  //
  // Skinned-mesh attempt (2026-04-24): for `.nj` models we try the
  // bone-local skinned pipeline first so motion playback is available.
  // If it fails (or the model is `.xj`), we fall back to the regular
  // world-baked pipeline. The skinned path also auto-wires the
  // animation panel and starts walk playback.
  // Skinned path requires `.nj` (chunk-Nj). The trailing `.nj` works
  // for both bare `foo.nj` and `bml#inner.nj` forms.
  const isNj = resolvedMeshPath.toLowerCase().endsWith(".nj");
  // PRIMARY path for .nj: the faithful psov2 client-side loader. The
  // owner gave psov2 as the KNOWN-GOOD reference; our server-side
  // reconstruction diverged and renders mangled geometry, so .nj routes
  // here first. The server-payload skinned/world-baked paths remain as
  // fallbacks (on parse/fetch failure, or for .xj which psov2 doesn't
  // handle).
  if (!composited && isNj) {
    realLoaded = await tryLoadPsov2NinjaModel(resolvedMeshPath, null);
  }
  if (!composited && !realLoaded && isNj) {
    // psov2 path failed — fall back to the legacy server-payload skinned
    // pipeline (animation panel + bone re-bake).
    realLoaded = await tryLoadSkinnedMesh(resolvedMeshPath, null);
  }
  if (!composited && !realLoaded) {
    // Skinned didn't work (or .xj model) — fall back to world-baked.
    realLoaded = await tryLoadRealMesh({ path: resolvedMeshPath });
    // World-baked path doesn't have an animation panel; hide it.
    const animBar = $("#modelAnimBar");
    if (animBar) animBar.hidden = true;
    state.anim.skinned = false;
  }
  if (!realLoaded) {
    // Fallback: build a primitive so the user sees something, but tag
    // the overlay so they know it's not the real geometry. Use the
    // immediate-fire path because this is the cold-load fallback and
    // the user shouldn't see a blank canvas during the throttle window.
    state.shape = "cube";
    rebuildMeshNow();
    setFallbackBanner("cube", state.lastMeshFailure || "mesh load failed");
    setStatus(`mesh load failed: ${state.lastMeshFailure || "unknown error"}`);
  }

  // Step 2: load the texture (if any). We do this AFTER the mesh so
  // applyTextureToCurrentMesh wraps the real geometry. We also prime
  // state.tileCount + tileSel so the user can pick a tile.
  if (texPath) {
    const texBasename = texPath.split("/").pop();
    state.filename = texBasename;
    state.selectedTileIdx = 0;
    // Drive the tile dropdown by hitting /api/model_preview for the
    // texture; this gives us tile_count + first_tile dims without an
    // extra round-trip.
    let texHint = null;
    try {
      texHint = await fetchPreviewHint(texBasename);
      state.tileCount = texHint.tile_count || 0;
      const tileSel = $("#modelTileSel");
      if (tileSel) {
        tileSel.innerHTML = "";
        for (let i = 0; i < state.tileCount; i++) {
          const opt = document.createElement("option");
          opt.value = String(i);
          opt.textContent = `tile ${i}`;
          tileSel.appendChild(opt);
        }
        tileSel.value = "0";
      }
    } catch (e) {
      // Hint failed; we can still try the tile_png endpoint since it
      // probes tiles independently.
      console.warn("[model_viewer] texture hint failed:", e);
    }
    await loadTexture();
  } else {
    // No texture available — clear the tile dropdown so the user can
    // see we couldn't find one.
    state.tileCount = 0;
    const tileSel = $("#modelTileSel");
    if (tileSel) tileSel.innerHTML = "<option>(no texture)</option>";
  }

  // Force a synchronous render so the canvas paints with the right
  // size. Same workaround as open(): the modal just became visible
  // and the parent's clientWidth/Height may have only resolved after
  // layout settled.
  requestAnimationFrame(() => {
    resizeRenderer();   // resizeRenderer() already schedules a paint via requestRender()
  });
}

// Local escape helper (small; we don't need the full escapeHtml from
// asset_router which lives in the IIFE scope of that file).
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

/**
 * Visualize a bone hierarchy on top of the current mesh. The asset
 * router fetches /api/model/<path>/skeleton and feeds the bones array
 * here. Each bone becomes a small wireframe sphere; parent links draw
 * as line segments. Read-only — bones can't be moved.
 */
function setSkeleton(bones) {
  // Clear any prior skeleton overlay
  if (_skeletonGroup) {
    state.scene.remove(_skeletonGroup);
    _skeletonGroup.traverse((c) => {
      if (c.geometry) c.geometry.dispose();
      if (c.material) {
        const mats = Array.isArray(c.material) ? c.material : [c.material];
        for (const m of mats) m.dispose();
      }
    });
    _skeletonGroup = null;
  }
  if (!bones || !bones.length) return;
  if (!state.scene) return;

  const group = new THREE.Group();
  group.name = "skeleton";
  // Build absolute world positions: PSOBB MeshTreeNode positions are
  // relative-to-parent. We fold them by accumulating along the parent
  // chain. With < 256 bones this is cheap.
  const worldPos = new Array(bones.length);
  for (const b of bones) {
    const local = b.position;
    let acc = local.slice();
    let p = b.parent;
    let guard = 0;
    while (p >= 0 && p < bones.length && guard < bones.length) {
      const pp = bones[p].position;
      acc[0] += pp[0];
      acc[1] += pp[1];
      acc[2] += pp[2];
      p = bones[p].parent;
      guard += 1;
    }
    worldPos[b.index] = acc;
  }

  // Bone spheres: small, cyan, semi-transparent
  const boneGeo = new THREE.SphereGeometry(0.05, 8, 6);
  const boneMat = new THREE.MeshBasicMaterial({
    color: 0x00ffff,
    wireframe: true,
    transparent: true,
    opacity: 0.85,
  });
  for (const b of bones) {
    const s = new THREE.Mesh(boneGeo.clone(), boneMat.clone());
    const wp = worldPos[b.index];
    s.position.set(wp[0], wp[1], wp[2]);
    group.add(s);
  }

  // Parent-child line segments (one BufferGeometry, draw as LINES)
  const lineVerts = [];
  for (const b of bones) {
    if (b.parent < 0 || b.parent >= bones.length) continue;
    const a = worldPos[b.index];
    const c = worldPos[b.parent];
    if (!a || !c) continue;
    lineVerts.push(a[0], a[1], a[2], c[0], c[1], c[2]);
  }
  if (lineVerts.length > 0) {
    const lg = new THREE.BufferGeometry();
    lg.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(lineVerts), 3),
    );
    const lm = new THREE.LineBasicMaterial({ color: 0x9d4edd, transparent: true, opacity: 0.85 });
    const lines = new THREE.LineSegments(lg, lm);
    group.add(lines);
  }

  // Inherit the active mesh's transform so the skeleton lines up with
  // the rendered geometry. If the mesh is a Group (real-mesh path) we
  // copy its position + scale; if a primitive, we leave the skeleton
  // at the origin (the user can still see relative bone layout).
  if (state.mesh && state.mesh.isGroup) {
    group.position.copy(state.mesh.position);
    group.scale.copy(state.mesh.scale);
    group.rotation.copy(state.mesh.rotation);
  }

  // Honor the toolbar's "show skeleton" toggle. Default = visible only
  // when called explicitly (the asset router populates it).
  const toggle = document.getElementById("modelSkeletonToggle");
  group.visible = toggle ? !!toggle.checked : true;
  state.scene.add(group);
  _skeletonGroup = group;

  // Wire the toggle if not already
  if (toggle && !toggle.dataset.wired) {
    toggle.dataset.wired = "1";
    toggle.addEventListener("change", (e) => {
      if (_skeletonGroup) _skeletonGroup.visible = !!e.target.checked;
    });
  }
}

window.psoOpenModelByPath = openByPath;
window.psoSetSkeleton = setSkeleton;
// fix/perf — read-only diagnostic: report per-material texture-map status
// for the on-screen psov2 mesh. Lets tests confirm textures are actually
// wired (mat.map present + decoded image) on BOTH cold and cache opens
// without reaching into module-private state. Returns null when no real
// mesh is shown.
window.psoDebugMeshTextures = function () {
  const g = state.meshGroup;
  if (!g) return null;
  const mats = [];
  g.traverse((c) => {
    if (!c.isMesh || !c.material) return;
    const list = Array.isArray(c.material) ? c.material : [c.material];
    for (const m of list) {
      const map = m && m.map;
      const img = map && (map.source && map.source.data || map.image);
      mats.push({
        hasMap: !!map,
        imgW: img ? (img.width || img.naturalWidth || 0) : 0,
        imgH: img ? (img.height || img.naturalHeight || 0) : 0,
      });
    }
  });
  return {
    archive: state.realMeshArchive,
    boundTextures: state.boundTextures ? state.boundTextures.size : 0,
    materials: mats,
    cacheSize: _psov2ModelCache.size,
  };
};
// Bundle prefetch: asset_router calls this as soon as the user clicks a
// model node, so the JSON round-trip starts in parallel with three.js's
// renderer / DOM-modal setup. Returns a Promise<bundle|null> — null
// means the bundle endpoint isn't available, callers fall back.
window.psoPrefetchModelBundle = prefetchModelBundle;
// fix/perf — public preload API. Queue a likely-next `.nj` model for
// idle-time, abortable, bounded prefetch so its next open is instant
// (warms the parsed-model cache without touching the scene). Safe to call
// liberally (deduped, capped, idle-scheduled, cancelled by any real open).
window.psoPreloadModel = preloadPsov2Model;
window.psoPreloadModels = function (paths) {
  if (!Array.isArray(paths)) return;
  for (const p of paths) preloadPsov2Model(p);
};
// fix/perf — read-only diagnostics for the parsed-model cache + preload
// queue (tests / devtools). Returns counts only; no THREE refs.
window.psoModelCacheInfo = function () {
  return {
    cacheSize: _psov2ModelCache.size,
    cacheKeys: Array.from(_psov2ModelCache.keys()),
    preloadQueue: _psov2PreloadQueue.slice(),
    preloadInflight: Array.from(_psov2PreloadInflight),
    preloadActive: _psov2PreloadActive,
  };
};
// Debug-overlay surface so tests + devtools can drive it without DOM
// events. setDebugMode flips the toggle; getDebugMeshes returns a
// shallow snapshot of the metadata array (no THREE.Mesh refs to keep
// callers serializable).
window.psoSetDebugMode = setDebugMode;
window.psoGetDebugMeshes = function () {
  return state.debugMeshes.map((e) => ({
    idx: e.idx,
    // fix/tooltabs — include the live THREE.Mesh. The UV panel
    // (uv_panel.collectSubmeshes) and Edit panel both read
    // e.mesh.geometry; the previous accessor stripped `.mesh` and
    // returned only scalars, so those tabs saw "no submesh" even on the
    // legacy path. The mesh is already a live scene/inspector object —
    // exposing the reference (not a clone) is correct and cheap.
    mesh: e.mesh,
    material_id: e.material_id,
    vertex_count: e.vertex_count,
    triangle_count: e.triangle_count,
    world_position: e.world_position,
    world_rotation_euler: e.world_rotation_euler,
    world_scale: e.world_scale,
    aabb: e.aabb,
  }));
};
// Unified-viewport bridge (2026-04-24): perspectives.js relocates the
// #modelCanvas + #modelStage out of the modal into the vp-stage. The
// ResizeObserver inside ensureRenderer() watches the canvas's parent
// at observer-init time; when the parent changes we need a fresh
// observer (and an immediate re-size) so the canvas fills the new
// container correctly. perspectives.js calls this after each move.
window.psoModelRebindResize = function () {
  try {
    const cv = $("#modelCanvas");
    if (!cv) return;
    if (state.resizeObs) {
      try { state.resizeObs.disconnect(); } catch (_e) {}
      state.resizeObs = null;
    }
    if (typeof ResizeObserver !== "undefined") {
      state.resizeObs = new ResizeObserver(resizeRenderer);
      if (cv.parentElement) state.resizeObs.observe(cv.parentElement);
    }
    resizeRenderer();   // resizeRenderer() already schedules a paint via requestRender()
  } catch (_e) {}
};

/**
 * Swap the texture wrapped on the currently-displayed mesh without
 * rebuilding geometry. Used by the asset_router's "texture override"
 * dropdown: the user picks an alternate matched_texture from the list
 * and we re-bind without losing the loaded mesh.
 *
 * `texFilename` is the basename (or `<bml>#<inner>.nj.xvm` form) the
 * tile pipeline understands. We re-fetch the tile dropdown so the user
 * sees the new texture's tile count, then call loadTexture() which
 * applies it to whichever mesh is currently rendered (real or primitive).
 */
async function setTextureByName(texFilename) {
  if (!texFilename) return;
  state.filename = texFilename;
  state.selectedTileIdx = 0;
  // Repopulate tile dropdown
  let texHint;
  try {
    texHint = await fetchPreviewHint(texFilename);
    state.tileCount = texHint.tile_count || 0;
    const tileSel = $("#modelTileSel");
    if (tileSel) {
      tileSel.innerHTML = "";
      for (let i = 0; i < state.tileCount; i++) {
        const opt = document.createElement("option");
        opt.value = String(i);
        opt.textContent = `tile ${i}`;
        tileSel.appendChild(opt);
      }
      tileSel.value = "0";
    }
  } catch (e) {
    setStatus(`texture hint failed: ${e?.message || e}`);
    return;
  }
  await loadTexture();
}

window.psoSetTexture = setTextureByName;

// =====================================================================
// Skeletal animation (2026-04-24).
//
// Pipeline:
//   1. tryLoadSkinnedMesh(modelPath) — fetch /api/model_skinned. On
//      success, builds a THREE.Group of submeshes whose vertices are
//      in BONE-LOCAL space, with each vertex tagged by owning bone
//      index. Stores per-submesh re-bake data on state.anim.
//   2. fetchMotionList(modelPath) — fetch /api/animations. Populates
//      the animation panel dropdown; auto-detects "walk" (or fallback
//      idle) and triggers loadMotion() with that index.
//   3. loadMotion(motionName) — fetch /api/animation_data and stash
//      the keyframes on state.anim.currentData. Resets time to 0.
//   4. animationFrame — called from the render loop every tick.
//      Advances state.anim.time, computes per-bone animated matrices,
//      applies them to each submesh's position attribute (re-bake on
//      the CPU), uploads as needed.
//
// Math: bone matrix = parent_bone_matrix × bind-relative-TRS. The
// "bind-relative-TRS" = (bind T + animated ΔT) × R_animated × (bind S
// × animated ΔS). For PSOBB BB the animated rotation REPLACES the
// bind rotation (typical Sega Ninja convention), animated translation
// REPLACES the bind translation when present, scale REPLACES the bind
// scale when present. Empty tracks fall back to bind values.
// =====================================================================

const _BAMS_TO_RAD = (2.0 * Math.PI) / 65536.0;

// Per-bone "tracks present" bitfield. Mirrors NJD_MTYPE_* in
// formats/njm.py. The animation_data JSON ships ``present`` per bone:
//   * Bit 0 (POS)  — translation track was authored on this bone.
//   * Bit 1 (ANG)  — euler-rotation track was authored.
//   * Bit 2 (SCL)  — scale track was authored.
//   * Bit 13 (QUAT)— quaternion track (overrides ANG when set).
// When a bit is UNSET the bone's keyframes carry the parser's default
// for that channel ((0,0,0) for translation/rotation, (1,1,1) for
// scale, undefined for quaternion); _sampleBoneTrack must fall back
// to the bone's bind-pose value for that channel. Without this guard
// rotation-only bones collapse to the origin during playback.
const _NJM_PRESENT_POS  = 1 << 0;
const _NJM_PRESENT_ANG  = 1 << 1;
const _NJM_PRESENT_SCL  = 1 << 2;
const _NJM_PRESENT_QUAT = 1 << 13;

// Pre-allocated scratch buffers for matrix math. Reused per frame to
// avoid allocation pressure (90 bones × 16 floats = 1.4k allocations
// per frame at 30 Hz otherwise). The size is bumped to 256 to allow
// future skeletons up to that bone count.
const _MAT_SCRATCH_LOCAL = new Float32Array(256 * 16);
const _MAT_SCRATCH_WORLD = new Float32Array(256 * 16);
const _MAT_SCRATCH_DELTA = new Float32Array(16);
const _MAT_SCRATCH_TMP = new Float32Array(16);

/**
 * Compose a row-major 4x4 from translation (3), Z-Y-X Euler radians
 * (3), and per-axis scale (3) into ``out`` (Float32Array, length 16).
 *
 * Mirror of formats/xj.py::_mat4_compose_trs (default ZYX order).
 *
 * `zxy` = true selects ZXY composition order (Phantasmal's branch for
 * the EVAL_ZXY_ANG mesh-tree-node flag). PSOBB.IO ships zero models
 * using this in the audited data set, but the world-bake pipeline
 * supports it, so the skinned path must too — modded data may.
 */
function _composeTrsM4(out, tx, ty, tz, rx, ry, rz, sx, sy, sz, zxy) {
  const cx = Math.cos(rx), s_x = Math.sin(rx);
  const cy = Math.cos(ry), s_y = Math.sin(ry);
  const cz = Math.cos(rz), s_z = Math.sin(rz);
  let r00, r01, r02, r10, r11, r12, r20, r21, r22;
  if (zxy) {
    // R = Rz * Rx * Ry (three.js "ZXY" Euler order).
    r00 = cz * cy - s_z * s_x * s_y;
    r01 = -s_z * cx;
    r02 = cz * s_y + s_z * s_x * cy;
    r10 = s_z * cy + cz * s_x * s_y;
    r11 = cz * cx;
    r12 = s_z * s_y - cz * s_x * cy;
    r20 = -cx * s_y;
    r21 = s_x;
    r22 = cx * cy;
  } else {
    // R = Rz * Ry * Rx (matches Phantasmal's "ZYX" Three.js Euler order).
    r00 = cz * cy;
    r01 = cz * s_y * s_x - s_z * cx;
    r02 = cz * s_y * cx + s_z * s_x;
    r10 = s_z * cy;
    r11 = s_z * s_y * s_x + cz * cx;
    r12 = s_z * s_y * cx - cz * s_x;
    r20 = -s_y;
    r21 = cy * s_x;
    r22 = cy * cx;
  }
  // M = T * R * S, row-major (m[row*4+col]).
  out[0] = r00 * sx;  out[1] = r01 * sy;  out[2] = r02 * sz;  out[3] = tx;
  out[4] = r10 * sx;  out[5] = r11 * sy;  out[6] = r12 * sz;  out[7] = ty;
  out[8] = r20 * sx;  out[9] = r21 * sy;  out[10] = r22 * sz; out[11] = tz;
  out[12] = 0;        out[13] = 0;        out[14] = 0;         out[15] = 1;
}

// NjsObject eval_flags bit values (mirror of formats/xj.py::EVAL_*).
// Used by _computeAnimatedBoneMatrices to honor UNIT_POS / UNIT_ANG /
// UNIT_SCL / SKIP / ZXY_ANG when composing per-bone bind matrices, so
// the skinned-path bind pose matches the world-baked pipeline.
const _EVAL_UNIT_POS = 0x01;
const _EVAL_UNIT_ANG = 0x02;
const _EVAL_UNIT_SCL = 0x04;
const _EVAL_ZXY_ANG  = 0x20;
const _EVAL_SKIP     = 0x40;

/** Multiply two row-major 4x4: out = a × b. Buffers may alias `a`. */
function _mulM4(out, a, b) {
  // Read a into locals first so out can alias either input safely.
  const a00 = a[0],  a01 = a[1],  a02 = a[2],  a03 = a[3];
  const a10 = a[4],  a11 = a[5],  a12 = a[6],  a13 = a[7];
  const a20 = a[8],  a21 = a[9],  a22 = a[10], a23 = a[11];
  const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
  const b00 = b[0],  b01 = b[1],  b02 = b[2],  b03 = b[3];
  const b10 = b[4],  b11 = b[5],  b12 = b[6],  b13 = b[7];
  const b20 = b[8],  b21 = b[9],  b22 = b[10], b23 = b[11];
  const b30 = b[12], b31 = b[13], b32 = b[14], b33 = b[15];
  out[0]  = a00*b00 + a01*b10 + a02*b20 + a03*b30;
  out[1]  = a00*b01 + a01*b11 + a02*b21 + a03*b31;
  out[2]  = a00*b02 + a01*b12 + a02*b22 + a03*b32;
  out[3]  = a00*b03 + a01*b13 + a02*b23 + a03*b33;
  out[4]  = a10*b00 + a11*b10 + a12*b20 + a13*b30;
  out[5]  = a10*b01 + a11*b11 + a12*b21 + a13*b31;
  out[6]  = a10*b02 + a11*b12 + a12*b22 + a13*b32;
  out[7]  = a10*b03 + a11*b13 + a12*b23 + a13*b33;
  out[8]  = a20*b00 + a21*b10 + a22*b20 + a23*b30;
  out[9]  = a20*b01 + a21*b11 + a22*b21 + a23*b31;
  out[10] = a20*b02 + a21*b12 + a22*b22 + a23*b32;
  out[11] = a20*b03 + a21*b13 + a22*b23 + a23*b33;
  out[12] = a30*b00 + a31*b10 + a32*b20 + a33*b30;
  out[13] = a30*b01 + a31*b11 + a32*b21 + a33*b31;
  out[14] = a30*b02 + a31*b12 + a32*b22 + a33*b32;
  out[15] = a30*b03 + a31*b13 + a32*b23 + a33*b33;
}

/**
 * Find the keyframe pair `(left, right)` that brackets `frame_t` in a
 * sorted keyframe list. Returns indices i0, i1 plus an alpha 0..1 of
 * how far between them frame_t lies. When frame_t falls past the last
 * keyframe (loop wrap), we return the last keyframe with alpha=0.
 */
function _findKeyframeBracket(kfs, frameT) {
  if (!kfs || kfs.length === 0) return null;
  if (kfs.length === 1) return { i0: 0, i1: 0, alpha: 0 };
  // Linear scan — most PSOBB tracks have <50 keyframes. Binary search
  // would be a micro-optimization; not worth the complexity here.
  let i0 = 0;
  for (let i = 0; i < kfs.length - 1; i++) {
    if (kfs[i + 1].t > frameT) {
      i0 = i;
      break;
    }
    i0 = i + 1;
  }
  if (i0 >= kfs.length - 1) {
    // Past the end — clamp to last keyframe.
    return { i0: kfs.length - 1, i1: kfs.length - 1, alpha: 0 };
  }
  const a = kfs[i0];
  const b = kfs[i0 + 1];
  const span = b.t - a.t;
  let alpha = span > 0 ? (frameT - a.t) / span : 0;
  if (alpha < 0) alpha = 0;
  if (alpha > 1) alpha = 1;
  return { i0, i1: i0 + 1, alpha };
}

/**
 * Linearly interpolate scalar (handles BAMS angles: shortest-arc
 * unwrapping). For BAMS angles with values in 0..65535 (unsigned), we
 * unwrap to nearest signed neighbour before lerping so a 359°→1°
 * transition doesn't sweep the long way around.
 */
function _lerpBams(a, b, alpha) {
  // BAMS are unsigned 16-bit, but the underlying angle is on a circle.
  // Compute shortest delta in [-32768, 32768) range.
  let delta = (b - a) & 0xFFFF;
  if (delta >= 0x8000) delta -= 0x10000;
  return a + delta * alpha;
}

/** Plain linear interp. */
function _lerp(a, b, alpha) {
  return a + (b - a) * alpha;
}

/**
 * Sample one bone's keyframe track at `frameT` (a fractional frame
 * number). Writes the resulting (tx, ty, tz, rx_rad, ry_rad, rz_rad,
 * sx, sy, sz) into `out` at indices [0..8]. Returns true if any track
 * was applied (= the bone is animated this frame); false if the track
 * is empty (= use bind pose).
 *
 * `bindBone` is a length-9 array carrying the bone's bind-pose TRS —
 * we use it as the fallback when a track has no keyframes for a
 * particular CHANNEL (e.g. rotation-only tracks).
 *
 * `presentMask` is the per-bone bitfield from the wire payload (see
 * NjmMotion.bone_present_tracks above). Bits identify which TRS
 * channels were ACTUALLY authored on this bone; channels whose bit is
 * unset fall back to ``bindBone`` regardless of the keyframe content.
 * Without this, a rotation-only bone whose keyframes carry default
 * (tx=ty=tz=0) would yank to the world origin every frame — which is
 * exactly the regression that broke walk playback for the dragon and
 * every other PSOBB monster (rotation-only bones are the common case).
 */
function _sampleBoneTrack(kfs, frameT, bindBone, presentMask, out) {
  if (!kfs || kfs.length === 0) {
    // No animation for this bone — copy bind into out.
    out[0] = bindBone[0]; out[1] = bindBone[1]; out[2] = bindBone[2];
    out[3] = bindBone[3]; out[4] = bindBone[4]; out[5] = bindBone[5];
    out[6] = bindBone[6]; out[7] = bindBone[7]; out[8] = bindBone[8];
    return false;
  }
  const br = _findKeyframeBracket(kfs, frameT);
  const a = kfs[br.i0];
  const b = kfs[br.i1];
  const al = br.alpha;
  // Per-channel: use the keyframe interp ONLY when this bone's
  // presence bitfield indicates the track was authored. Otherwise
  // bind pose. Default to "all present" when presentMask is missing
  // (legacy server response) so we preserve the prior behaviour for
  // older payloads — but that path will collapse rotation-only bones,
  // which is why the server bumps ``present`` on every keyframe now.
  const hasPos = presentMask === undefined ? true : !!(presentMask & _NJM_PRESENT_POS);
  const hasAng = presentMask === undefined ? true : !!(presentMask & _NJM_PRESENT_ANG);
  const hasScl = presentMask === undefined ? true : !!(presentMask & _NJM_PRESENT_SCL);
  if (hasPos) {
    out[0] = _lerp(a.tx, b.tx, al);
    out[1] = _lerp(a.ty, b.ty, al);
    out[2] = _lerp(a.tz, b.tz, al);
  } else {
    out[0] = bindBone[0]; out[1] = bindBone[1]; out[2] = bindBone[2];
  }
  if (hasAng) {
    // BAMS rotations on the wire — convert to radians here.
    out[3] = _lerpBams(a.rx, b.rx, al) * _BAMS_TO_RAD;
    out[4] = _lerpBams(a.ry, b.ry, al) * _BAMS_TO_RAD;
    out[5] = _lerpBams(a.rz, b.rz, al) * _BAMS_TO_RAD;
  } else {
    out[3] = bindBone[3]; out[4] = bindBone[4]; out[5] = bindBone[5];
  }
  if (hasScl) {
    out[6] = _lerp(a.sx, b.sx, al);
    out[7] = _lerp(a.sy, b.sy, al);
    out[8] = _lerp(a.sz, b.sz, al);
  } else {
    out[6] = bindBone[6]; out[7] = bindBone[7]; out[8] = bindBone[8];
  }
  return true;
}

/**
 * Build per-bone WORLD matrices for a given playback frame.
 *
 * For each bone (in DFS order):
 *   localM = compose_trs(animated_t, animated_r, animated_s)
 *   worldM = parent.worldM × localM   (root: parent is identity)
 *
 * Writes into `_MAT_SCRATCH_WORLD` (one 16-float matrix per bone,
 * indexed by bone idx). Returns the boneCount actually written.
 *
 * When `frameT` is null (bind-pose fallback) we use each bone's bind
 * TRS directly — this is the path the "reset to bind pose" button
 * hits.
 */
function _computeAnimatedBoneMatrices(bones, animData, frameT) {
  const n = bones.length;
  const tmpTRS = new Float32Array(9);
  for (let bi = 0; bi < n; bi++) {
    const b = bones[bi];
    // Eval flags govern which TRS components contribute to the local
    // bone matrix. UNIT_POS forces 0 translation; UNIT_ANG forces 0
    // rotation; UNIT_SCL forces unit scale; SKIP discards the entire
    // local matrix (identity); ZXY_ANG flips composition order. The
    // server now ships eval_flags + scale in the bone payload (older
    // payloads default both to 0 / [1,1,1] so the skinned path
    // continues to work against legacy responses).
    const ef = (b.eval_flags | 0);
    const bindScale = b.scale || [1.0, 1.0, 1.0];
    // Bind-pose TRS for this bone (BAMS rotation -> radians, with
    // eval-flag overrides applied so unit-* flags zero out their TRS
    // component the same way the bake pipeline does).
    const bindT = (ef & _EVAL_UNIT_POS) ? [0, 0, 0] : b.position;
    const bindR = (ef & _EVAL_UNIT_ANG) ? [0, 0, 0] : b.rotation_bams;
    const bindS = (ef & _EVAL_UNIT_SCL) ? [1, 1, 1] : bindScale;
    const bindArr = [
      bindT[0], bindT[1], bindT[2],
      bindR[0] * _BAMS_TO_RAD, bindR[1] * _BAMS_TO_RAD, bindR[2] * _BAMS_TO_RAD,
      bindS[0], bindS[1], bindS[2],
    ];
    // Sample the animation track for this bone (if any).
    const boneEntry = (animData && animData.bones && bi < animData.bones.length)
      ? animData.bones[bi]
      : null;
    const track = boneEntry ? boneEntry.kf : null;
    // Per-bone "present" bitfield identifies which TRS CHANNELS were
    // actually authored (see comment block on _NJM_PRESENT_*). If the
    // wire payload doesn't carry it (older servers), assume all
    // channels are present — that path is unsafe for rotation-only
    // bones, so the server is the source of truth here.
    const presentMask = boneEntry && typeof boneEntry.present === "number"
      ? boneEntry.present
      : undefined;
    if (frameT !== null && track && track.length > 0) {
      _sampleBoneTrack(track, frameT, bindArr, presentMask, tmpTRS);
    } else {
      // Bind pose fallback.
      tmpTRS[0] = bindArr[0]; tmpTRS[1] = bindArr[1]; tmpTRS[2] = bindArr[2];
      tmpTRS[3] = bindArr[3]; tmpTRS[4] = bindArr[4]; tmpTRS[5] = bindArr[5];
      tmpTRS[6] = bindArr[6]; tmpTRS[7] = bindArr[7]; tmpTRS[8] = bindArr[8];
    }
    // Anim Editor v4 / Task 4 — bone mute via rig overrides. When an
    // override is set on a bone (e.g. by the eye-toggle calling
    // psoSetBonePoseOverride(idx, bind_pose)) we replace the just-sampled
    // animated TRS with the override. This mirrors the behaviour of
    // psoApplyRigBake() so muted bones stay frozen during playback.
    // Backward compatible: when no override is set (the common case),
    // tmpTRS keeps the animated value and the bake path is unchanged.
    const animOv = state.rigBoneOverrides ? state.rigBoneOverrides.get(bi) : null;
    if (animOv) {
      if (animOv.position) {
        tmpTRS[0] = animOv.position[0];
        tmpTRS[1] = animOv.position[1];
        tmpTRS[2] = animOv.position[2];
      }
      if (animOv.rotation_bams) {
        tmpTRS[3] = animOv.rotation_bams[0] * _BAMS_TO_RAD;
        tmpTRS[4] = animOv.rotation_bams[1] * _BAMS_TO_RAD;
        tmpTRS[5] = animOv.rotation_bams[2] * _BAMS_TO_RAD;
      }
      if (animOv.scale) {
        tmpTRS[6] = animOv.scale[0];
        tmpTRS[7] = animOv.scale[1];
        tmpTRS[8] = animOv.scale[2];
      }
    }
    // Local matrix. SKIP forces identity (matches the bake pipeline).
    const localOff = bi * 16;
    if (ef & _EVAL_SKIP) {
      const out = _MAT_SCRATCH_LOCAL;
      out[localOff + 0] = 1; out[localOff + 1] = 0; out[localOff + 2] = 0; out[localOff + 3] = 0;
      out[localOff + 4] = 0; out[localOff + 5] = 1; out[localOff + 6] = 0; out[localOff + 7] = 0;
      out[localOff + 8] = 0; out[localOff + 9] = 0; out[localOff + 10] = 1; out[localOff + 11] = 0;
      out[localOff + 12] = 0; out[localOff + 13] = 0; out[localOff + 14] = 0; out[localOff + 15] = 1;
    } else {
      _composeTrsM4(
        _MAT_SCRATCH_LOCAL.subarray(localOff, localOff + 16),
        tmpTRS[0], tmpTRS[1], tmpTRS[2],
        tmpTRS[3], tmpTRS[4], tmpTRS[5],
        tmpTRS[6], tmpTRS[7], tmpTRS[8],
        !!(ef & _EVAL_ZXY_ANG),
      );
    }
    // World matrix = parent.world × local (root: world = local).
    const worldOff = bi * 16;
    if (b.parent < 0) {
      _MAT_SCRATCH_WORLD.set(
        _MAT_SCRATCH_LOCAL.subarray(localOff, localOff + 16),
        worldOff,
      );
    } else {
      const parentOff = b.parent * 16;
      _mulM4(
        _MAT_SCRATCH_WORLD.subarray(worldOff, worldOff + 16),
        _MAT_SCRATCH_WORLD.subarray(parentOff, parentOff + 16),
        _MAT_SCRATCH_LOCAL.subarray(localOff, localOff + 16),
      );
    }
  }
  return n;
}

/**
 * Re-bake every submesh's vertices using the current bone matrices.
 *
 * For each vertex:
 *   M = bone_matrices[v.bone_idx]
 *   p_world = M @ p_local
 *   n_world = M_3x3 @ n_local
 *
 * Writes positions + normals into the geometry's existing
 * BufferAttribute backing arrays (mutates in place; calls
 * `needsUpdate = true` so three.js re-uploads to the GPU).
 *
 * Skips vertices with bone_idx < 0 (no skinning info — typically the
 * root bone or non-skinned static prop).
 */
function _bakeSkinnedSubmeshes(skinSubmeshes, boneCount) {
  for (const sub of skinSubmeshes) {
    const localPos = sub.bonePositions;          // bone-local positions (Float32, 3 per vert)
    const localNorm = sub.boneNormals;            // bone-local normals
    const boneIdx = sub.vertBoneIdx;              // Int32
    const posAttr = sub.geometry.attributes.position;
    const normAttr = sub.geometry.attributes.normal;
    const outPos = posAttr.array;
    const outNorm = normAttr ? normAttr.array : null;
    const vc = (localPos.length / 3) | 0;
    for (let vi = 0; vi < vc; vi++) {
      let bi = boneIdx[vi];
      if (bi < 0 || bi >= boneCount) bi = 0;
      const mOff = bi * 16;
      const m0 = _MAT_SCRATCH_WORLD[mOff + 0];
      const m1 = _MAT_SCRATCH_WORLD[mOff + 1];
      const m2 = _MAT_SCRATCH_WORLD[mOff + 2];
      const m3 = _MAT_SCRATCH_WORLD[mOff + 3];
      const m4 = _MAT_SCRATCH_WORLD[mOff + 4];
      const m5 = _MAT_SCRATCH_WORLD[mOff + 5];
      const m6 = _MAT_SCRATCH_WORLD[mOff + 6];
      const m7 = _MAT_SCRATCH_WORLD[mOff + 7];
      const m8 = _MAT_SCRATCH_WORLD[mOff + 8];
      const m9 = _MAT_SCRATCH_WORLD[mOff + 9];
      const m10 = _MAT_SCRATCH_WORLD[mOff + 10];
      const m11 = _MAT_SCRATCH_WORLD[mOff + 11];
      const px = localPos[vi * 3 + 0];
      const py = localPos[vi * 3 + 1];
      const pz = localPos[vi * 3 + 2];
      outPos[vi * 3 + 0] = m0 * px + m1 * py + m2 * pz + m3;
      outPos[vi * 3 + 1] = m4 * px + m5 * py + m6 * pz + m7;
      outPos[vi * 3 + 2] = m8 * px + m9 * py + m10 * pz + m11;
      if (outNorm) {
        const nx = localNorm[vi * 3 + 0];
        const ny = localNorm[vi * 3 + 1];
        const nz = localNorm[vi * 3 + 2];
        outNorm[vi * 3 + 0] = m0 * nx + m1 * ny + m2 * nz;
        outNorm[vi * 3 + 1] = m4 * nx + m5 * ny + m6 * nz;
        outNorm[vi * 3 + 2] = m8 * nx + m9 * ny + m10 * nz;
      }
    }
    posAttr.needsUpdate = true;
    if (normAttr) normAttr.needsUpdate = true;
  }
}

/**
 * Build a THREE.Group from the SKINNED payload (/api/model_skinned).
 *
 * Differs from buildMeshGroupFromPayload in two ways:
 *   1. Each submesh's `position` BufferAttribute starts as the
 *      BONE-LOCAL positions (NOT world-baked). The render loop
 *      re-bakes each frame.
 *   2. We stash a parallel `bonePositions` Float32Array (read-only
 *      copy of the bone-local positions) on each submesh so re-bake
 *      can read from it without losing the bone-local snapshot when
 *      the GPU-bound array gets mutated.
 *
 * Returns ``{group, skinSubmeshes, totalVerts, totalTris, aabbMin,
 * aabbMax, debugMeshes}`` — same field layout as the world-baked
 * variant plus the new ``skinSubmeshes`` for the animation loop.
 */
function buildSkinnedMeshGroupFromPayload(payload, boundTextures) {
  const group = new THREE.Group();
  let totalVerts = 0;
  let totalTris = 0;
  const skinSubmeshes = [];
  const debugMeshes = [];
  const bound = boundTextures || new Map();
  const aabbMin = [Infinity, Infinity, Infinity];
  const aabbMax = [-Infinity, -Infinity, -Infinity];

  for (const m of payload.meshes || []) {
    const vbuf = b64ToArrayBuffer(m.vertices_b64);
    const ibuf = b64ToArrayBuffer(m.indices_b64);
    const bbuf = b64ToArrayBuffer(m.bone_indices_b64);
    const verts = new Float32Array(vbuf);
    const indices = new Uint32Array(ibuf);
    const boneIdxRaw = new Int32Array(bbuf);

    if (verts.length === 0 || indices.length === 0) continue;
    // v2 payloads carry 4 trailing RGBA floats (12-float stride).
    const hasColor = payload.has_color === true;
    const stride = hasColor ? 12 : 8;
    const vertexCount = (verts.length / stride) | 0;
    if (!Number.isInteger(verts.length / stride)) {
      console.warn("model_viewer: skinned non-integer vertex count; skipping mesh");
      continue;
    }

    // Split into separate position / normal / uv arrays.
    const positions = new Float32Array(vertexCount * 3);
    const normals = new Float32Array(vertexCount * 3);
    const uvs = new Float32Array(vertexCount * 2);
    const colors = hasColor ? new Float32Array(vertexCount * 4) : null;
    for (let i = 0; i < vertexCount; i++) {
      const o = i * stride;
      positions[i * 3 + 0] = verts[o + 0];
      positions[i * 3 + 1] = verts[o + 1];
      positions[i * 3 + 2] = verts[o + 2];
      normals[i * 3 + 0] = verts[o + 3];
      normals[i * 3 + 1] = verts[o + 4];
      normals[i * 3 + 2] = verts[o + 5];
      uvs[i * 2 + 0] = verts[o + 6];
      uvs[i * 2 + 1] = verts[o + 7];
      if (colors) {
        colors[i * 4 + 0] = verts[o + 8];
        colors[i * 4 + 1] = verts[o + 9];
        colors[i * 4 + 2] = verts[o + 10];
        colors[i * 4 + 3] = verts[o + 11];
      }
    }

    // The geometry's position/normal arrays START as a copy of the
    // bone-local data. The re-bake step overwrites them in place each
    // frame using the original bone-local snapshots stashed below.
    // (Color is static per-vertex — not re-baked.)
    const renderPos = new Float32Array(positions);
    const renderNorm = new Float32Array(normals);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(renderPos, 3));
    geo.setAttribute("normal", new THREE.BufferAttribute(renderNorm, 3));
    geo.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
    if (colors) geo.setAttribute("color", new THREE.BufferAttribute(colors, 4));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));
    // Skip computeBoundingSphere here — the bone-local AABB is much
    // smaller than the animated AABB, so three.js' frustum culling
    // would mistakenly cull the mesh once animation moves it. We
    // disable culling on the resulting Mesh instead.

    const matId = (m.material_id | 0);
    const submeshTex = bound.has(matId) ? bound.get(matId) : (state.texture || null);
    // Phantasmal-diff fixes 3+4 (2026-04-25): Basic for textured,
    // Lambert for un-textured, transparent=false default. Skinned path:
    // Dragon's 1069 submeshes were the canonical PBR-shader-compile
    // bottleneck — Basic shaves the per-frame cost dramatically.
    // side=DoubleSide is the safe default — see the long comment in
    // `buildMeshGroupFromPayload` for the full rationale (no LH→RH
    // axis flip happens; the per-strip `cw` bit handles winding;
    // PSOBB authoring data has known reverse-faced strips).
    // Phase 3 (2026-06-20): unlit MeshBasicMaterial for BOTH branches
    // (psov2 parity) + vertexColors. The Lambert un-textured branch
    // washed skinned models (Dragon, De Rol Le) to flat white.
    const mat = submeshTex
      ? new THREE.MeshBasicMaterial({
          map: submeshTex,
          color: 0xffffff,
          vertexColors: hasColor,
          wireframe: state.wireframe,
          side: THREE.DoubleSide,
          transparent: false,
        })
      : new THREE.MeshBasicMaterial({
          color: 0xffffff,
          vertexColors: hasColor,
          wireframe: state.wireframe,
          side: THREE.DoubleSide,
          transparent: false,
        });
    // Phase 3 (2026-06-20): apply per-submesh blend/alpha/two-sided flags.
    applyPsoMaterialFlags(mat, m);

    const mesh = new THREE.Mesh(geo, mat);
    mesh.userData.materialId = matId;
    // Disable frustum culling — the bone-local AABB is a poor proxy
    // for the animated mesh's true bound. The cost is one always-drawn
    // mesh per submesh, which is fine for the model viewer's single-
    // model scene.
    mesh.frustumCulled = false;
    group.add(mesh);

    // Stash per-submesh re-bake data.
    skinSubmeshes.push({
      geometry: geo,
      bonePositions: positions,    // read-only bone-local snapshot
      boneNormals: normals,        // read-only bone-local snapshot
      vertBoneIdx: boneIdxRaw,
    });

    debugMeshes.push({
      idx: debugMeshes.length,
      mesh,
      material_id: matId,
      vertex_count: vertexCount,
      triangle_count: (indices.length / 3) | 0,
      world_position: [0, 0, 0],
      world_rotation_euler: [0, 0, 0],
      world_scale: [1, 1, 1],
      bounding_sphere: m.bounding_sphere || [0, 0, 0, 0],
      aabb: m.aabb || null,
      eval_flags: 0,
    });

    totalVerts += vertexCount;
    totalTris += indices.length / 3;
  }

  // Compute the WORLD-space AABB by running a single bind-pose re-bake
  // (animation-off) and looking at the resulting positions.
  // This is required so the camera-fit centering uses the visible
  // mesh, not the bone-local AABB (which is centred on the origin per
  // bone — way too small).
  const bones = payload.bones || [];
  if (bones.length > 0 && skinSubmeshes.length > 0) {
    _computeAnimatedBoneMatrices(bones, null, null);
    _bakeSkinnedSubmeshes(skinSubmeshes, bones.length);
    // Compute AABB from re-baked render positions.
    for (const sub of skinSubmeshes) {
      const pa = sub.geometry.attributes.position.array;
      for (let i = 0; i < pa.length; i += 3) {
        const x = pa[i], y = pa[i + 1], z = pa[i + 2];
        if (x < aabbMin[0]) aabbMin[0] = x;
        if (y < aabbMin[1]) aabbMin[1] = y;
        if (z < aabbMin[2]) aabbMin[2] = z;
        if (x > aabbMax[0]) aabbMax[0] = x;
        if (y > aabbMax[1]) aabbMax[1] = y;
        if (z > aabbMax[2]) aabbMax[2] = z;
      }
    }
  }

  // Centre + uniform scale to fit the camera. Same logic as the
  // world-baked path.
  if (Number.isFinite(aabbMin[0])) {
    const cx = (aabbMin[0] + aabbMax[0]) / 2;
    const cy = (aabbMin[1] + aabbMax[1]) / 2;
    const cz = (aabbMin[2] + aabbMax[2]) / 2;
    const dx = aabbMax[0] - aabbMin[0];
    const dy = aabbMax[1] - aabbMin[1];
    const dz = aabbMax[2] - aabbMin[2];
    const maxDim = Math.max(dx, dy, dz, 0.001);
    const scale = 2.0 / maxDim;
    group.position.set(-cx * scale, -cy * scale, -cz * scale);
    group.scale.set(scale, scale, scale);
  }

  return { group, skinSubmeshes, totalVerts, totalTris, aabbMin, aabbMax, debugMeshes };
}

// ---- psov2 faithful Ninja path (PRIMARY for .nj) -------------------
//
// Routes `.nj` / `.bml#inner.nj` models through the verbatim psov2
// loader in static/psov2_ninja.js INSTEAD of the server-side
// reconstruction. Flow (mirrors the psov2 reference):
//   1. fetch raw inner NJ bytes from /api/raw_nj/<modelPath>
//   2. parseNinjaModel(buf, {texList}) -> THREE.SkinnedMesh (real Bone
//      tree, skinWeight/skinIndex, Skeleton, bind)
//   3. bind OUR decoded tiles as the texList (chunk texId -> tile N)
//
// Returns true on success. On ANY failure returns false so openByPath
// falls through to the legacy skinned / world-baked paths.

/** Derive the inline-XVM texture archive path for an .nj model path. */
function _psov2TextureArchive(modelPath) {
  // `<bml>#<inner>.nj`  -> `<bml>#<inner>.nj.xvm` (inline XVM appendix).
  // bare `<foo>.nj`     -> `<foo>.nj.xvm` sibling (tile_png 404s if none,
  //                        handled gracefully — model still renders flat).
  return `${modelPath}.xvm`;
}

/**
 * Build the psov2 `texList` (Array<THREE.Texture> indexed by chunk
 * texId) using the SERVER-RESOLVED texture binding.
 *
 * Why the server binding (not a naive `<model>.nj.xvm` sibling fetch):
 * many PSOBB models — most visibly the player bodies (`pl[A-Z]bdy00.nj`)
 * — carry NO inline XVM appendix. Their textures live in a SEPARATE
 * archive (`pl[A-Z]tex.afs`) and bind by `material_id`/positional slot.
 * The reference psov2 resolves these by hand (`AssetPlayer.js` pulls
 * `plBtex.afs` tiles and remaps them into the body's small texId space).
 * Our server already computes the equivalent mapping in
 * `_build_model_texture_binding`, returning per-`material_id` rows whose
 * `source` is `in_bml` / `cross_afs` / `cross_bml`. Crucially, for an NJ
 * model the server's `material_id` IS the chunk texId (see
 * formats/material.py: `material_id = cur_tex_id`), so a binding row's
 * `material_id` indexes the psov2 texList directly.
 *
 * We fetch `/api/model_textures/<modelPath>` for the binding, then reuse
 * `fetchBoundTextures()` (the same in_bml/cross_afs/cross_bml resolver
 * the legacy skinned path uses) to turn the binding into a
 * `Map<material_id, THREE.Texture>`. Indexing that map by material_id ==
 * texId yields the texList `getModel()` expects.
 *
 * Returns `{ texList, fetched, binding, archive }`:
 *   texList  — Array<THREE.Texture> indexed by chunk texId (sparse).
 *   fetched  — flat list of unique textures (for disposal bookkeeping).
 *   binding  — the raw binding rows (for the texture panel + reset).
 *   archive  — the in_bml archive path (for the texture panel thumbs).
 */
async function _psov2BuildTexList(modelPath, texIds, signal) {
  const archive = _psov2TextureArchive(modelPath);
  const empty = { texList: [], fetched: [], binding: [], archive };

  // fix/perf — when an explicit AbortSignal is passed (preload path), use
  // plain fetch on THAT signal so the preload can be cancelled
  // independently of the active-open lifecycle signal. Otherwise ride the
  // lifecycle signal via _lifecycleFetch (the interactive open path).
  const _fetch = signal
    ? (url, init) => fetch(url, Object.assign({}, init, { signal }))
    : _lifecycleFetch;

  // Step 1: ask the server for the resolved per-material binding. This
  // covers cross_afs (player bodies) / cross_bml / in_bml uniformly.
  let binding = [];
  try {
    const r = await _fetch(
      `/api/model_textures/${encodeURIComponent(modelPath)}`,
    );
    if (r.ok) {
      const j = await r.json();
      binding = Array.isArray(j.binding) ? j.binding : [];
    } else {
      console.warn(
        `model_viewer: psov2 model_textures ${modelPath} -> http ${r.status}`,
      );
    }
  } catch (e) {
    if (_isAbortError(e)) return empty;
    console.warn(`model_viewer: psov2 model_textures fetch failed:`, e);
  }

  // Step 2: resolve the binding rows to live THREE.Textures, keyed by
  // material_id, via the shared resolver. If the server gave us nothing
  // (older build / no binding), fall back to a positional fetch against
  // the in_bml archive so an inline-XVM model still textures.
  let texByMid = new Map();
  if (binding.length > 0) {
    texByMid = await fetchBoundTextures(archive, binding);
  }
  if (texByMid.size === 0) {
    // Positional fallback: fetch tile == texId directly from the
    // in_bml archive (the original behaviour, kept for inline-XVM
    // models the binding endpoint can't enumerate).
    const ids = Array.from(
      new Set(texIds.filter((n) => Number.isInteger(n) && n >= 0)),
    );
    const results = await Promise.all(
      ids.map((id) => loadTileTexture(archive, id)),
    );
    ids.forEach((id, i) => {
      if (results[i]) texByMid.set(id, results[i]);
    });
  }

  // Step 3: project the material_id->texture map into the texId-indexed
  // texList getModel() wires (`mat.map = texList[texId]`). For NJ models
  // material_id === texId, so this is a direct index. Dedupe the live
  // textures for disposal bookkeeping.
  const texList = [];
  const seen = new Set();
  const fetched = [];
  for (const [mid, tex] of texByMid.entries()) {
    if (!tex) continue;
    // psov2's NinjaTexture stamped `.transparent` so getModel() could
    // flip alphaTest. We leave it false unless explicitly set; the
    // NJ chunk's own blend flags drive transparency on our path.
    if (tex.transparent === undefined) tex.transparent = false;
    texList[mid | 0] = tex;
    if (!seen.has(tex)) {
      seen.add(tex);
      fetched.push(tex);
    }
  }
  return { texList, fetched, binding, archive };
}

// fix/tooltabs — radians (THREE.Bone Euler) -> BAMS (PSO binary angle).
// _BAMS_TO_RAD = 2π/65536, so rad / _BAMS_TO_RAD == BAMS. Round to int so
// the Skeleton inspector's BAMS/deg readouts are clean.
function _radToBams(rad) {
  return Math.round((rad || 0) / _BAMS_TO_RAD);
}

// fix/tooltabs — adapt the live psov2 THREE.Skeleton bone tree into the
// {index,parent,position,rotation_bams,scale,eval_flags} shape the Skeleton
// panel (psoGetSkeleton) expects. The psov2 loader builds a real
// THREE.Skeleton (mesh.skeleton.bones[], DFS order, parent links via
// THREE.Bone.parent), but tryLoadPsov2NinjaModel never mirrored it into
// state.anim.bones — so the panel saw [] and rendered "No skeleton loaded".
// We read each bone's LOCAL TRS (position/rotation Euler radians/scale)
// straight off the Object3D and convert the rotation to BAMS. parent is the
// index of bone.parent within the same bones[] array, or -1 when the parent
// isn't a bone (the SkinnedMesh root the rootBone is attached to).
function _psov2AdaptSkeletonBones(mesh) {
  const out = [];
  if (!mesh || !mesh.skeleton || !Array.isArray(mesh.skeleton.bones)) return out;
  const bones = mesh.skeleton.bones;
  if (!bones.length) return out;
  // index map for O(1) parent resolution.
  const idxOf = new Map();
  for (let i = 0; i < bones.length; i++) idxOf.set(bones[i], i);
  for (let i = 0; i < bones.length; i++) {
    const b = bones[i];
    if (!b) continue;
    const p = (b.parent && idxOf.has(b.parent)) ? idxOf.get(b.parent) : -1;
    const e = b.rotation; // THREE.Euler (radians)
    out.push({
      index: i,
      parent: p,
      position: [b.position.x, b.position.y, b.position.z],
      rotation_bams: [_radToBams(e.x), _radToBams(e.y), _radToBams(e.z)],
      scale: [b.scale.x, b.scale.y, b.scale.z],
      eval_flags: 0,
    });
  }
  return out;
}

// fix/tooltabs — build per-submesh "view" meshes for the UV/Edit/Subdivide
// inspectors from the single psov2 SkinnedMesh. That mesh has ONE
// (non-indexed) BufferGeometry partitioned by geometry.groups[] (one group
// per material, {start,count,materialIndex}). Each panel reads
// e.mesh.geometry.getAttribute("uv") and expects ONE submesh's UVs — so we
// slice the shared position+uv arrays per group into a small standalone
// BufferGeometry wrapped in a non-scene THREE.Mesh. These view meshes are
// inspector-only (never added to the scene, never rendered); they carry
// userData.materialId so the UV panel labels them and tooling that filters
// by material works. Returns the state.debugMeshes entry array.
function _psov2BuildDebugMeshes(mesh) {
  const out = [];
  if (!mesh || !mesh.geometry) return out;
  const geo = mesh.geometry;
  const posAttr = geo.getAttribute("position");
  const uvAttr = geo.getAttribute("uv");
  if (!posAttr) return out;
  const groups = (geo.groups && geo.groups.length)
    ? geo.groups
    : [{ start: 0, count: posAttr.count, materialIndex: 0 }];
  const matGroups = (mesh.userData && mesh.userData.materialGroups) || [];
  for (let gi = 0; gi < groups.length; gi++) {
    const g = groups[gi];
    const start = g.start | 0;            // first VERTEX (non-indexed geometry)
    const count = g.count | 0;            // vertex count in this group
    if (count <= 0) continue;
    const mg = matGroups[g.materialIndex | 0];
    const materialId = mg ? (mg.materialId | 0) : (g.materialIndex | 0);
    // Slice position + uv for just this group's vertices into a fresh
    // (small) geometry. Non-indexed, so the slice is a contiguous range.
    const subGeo = new THREE.BufferGeometry();
    const posSlice = new Float32Array(count * 3);
    for (let v = 0; v < count; v++) {
      const sv = start + v;
      posSlice[v * 3] = posAttr.getX(sv);
      posSlice[v * 3 + 1] = posAttr.getY(sv);
      posSlice[v * 3 + 2] = posAttr.getZ(sv);
    }
    subGeo.setAttribute("position", new THREE.BufferAttribute(posSlice, 3));
    if (uvAttr) {
      const uvSlice = new Float32Array(count * 2);
      for (let v = 0; v < count; v++) {
        const sv = start + v;
        uvSlice[v * 2] = uvAttr.getX(sv);
        uvSlice[v * 2 + 1] = uvAttr.getY(sv);
      }
      subGeo.setAttribute("uv", new THREE.BufferAttribute(uvSlice, 2));
    }
    const viewMesh = new THREE.Mesh(subGeo);
    viewMesh.userData.materialId = materialId;
    viewMesh.userData.psov2View = true;
    viewMesh.matrixAutoUpdate = false;
    out.push({
      idx: out.length,
      mesh: viewMesh,
      material_id: materialId,
      vertex_count: count,
      triangle_count: (count / 3) | 0,
      world_position: [0, 0, 0],
      world_rotation_euler: [0, 0, 0],
      world_scale: [1, 1, 1],
      aabb: null,
    });
  }
  return out;
}

async function tryLoadPsov2NinjaModel(modelPath, hint) {
  // Only `.nj` (bare or `<bml>#<inner>.nj`). The trailing `.nj` test
  // covers both forms.
  const isNj = modelPath.toLowerCase().endsWith(".nj");
  if (!isNj) {
    _setMeshFailure(`psov2 path requires .nj (got ${modelPath.split(".").pop()})`);
    return false;
  }

  setStatus(`loading ninja model ${modelPath} (psov2)...`);

  // fix/perf — snapshot the lifecycle epoch for this open. We re-check it
  // just before mutating the shared scene; if a newer open started while
  // our fetch/parse was in flight, we DISCARD our result instead of
  // clobbering the newer model (the abort-on-switch correctness guard).
  const openEpoch = _currentEpoch();

  // fix/perf — parsed-model LRU cache. Re-opening a recently-viewed model
  // skips ALL network legs (raw_nj + texlist + motion fetches): we keep
  // the decoded NJ ArrayBuffer, the resolved THREE.Textures, the raw
  // motion buffers, and the binding, then re-parse the bytes (cheap) into
  // a FRESH group per open. parseNinjaModel is pure over the buffer, so a
  // re-parse is safe and deterministic.
  let buf, loader, texIds, archive, texList, fetchedTextures, psov2Binding;
  let nativeMotions, motionBuffers, motionNames, mesh;

  const cached = _psov2CacheTouch(modelPath);
  if (cached) {
    buf = cached.buf;
    texIds = cached.texIds;
    archive = cached.archive;
    texList = cached.texList;
    fetchedTextures = cached.fetched;
    psov2Binding = cached.binding;
    nativeMotions = cached.nativeMotions;
    motionBuffers = cached.motionBuffers;
    motionNames = cached.motionNames;
    try {
      mesh = parseNinjaModel(buf, {
        name: modelPath,
        texList,
        motions: motionBuffers,
        motionNames,
      });
    } catch (e) {
      // Cache entry somehow no longer parses — drop it and fall through
      // to the cold path below.
      _psov2ModelCache.delete(modelPath);
      cached.__bad = true;
    }
    if (mesh) {
      loader = mesh.userData.ninjaLoader;
    }
  }

  if (!mesh) {
    // ---- COLD path: fetch + parse ONCE + assign textures post-hoc ----

    // Step 1: raw inner NJ bytes.
    try {
      const r = await _lifecycleFetch(`/api/raw_nj/${encodeURIComponent(modelPath)}`);
      if (!r.ok) {
        let detail = `http ${r.status}`;
        try { const eb = await r.json(); if (eb && eb.detail) detail = eb.detail; } catch {}
        _setMeshFailure(`raw_nj: ${detail}`);
        return false;
      }
      buf = await r.arrayBuffer();
    } catch (e) {
      if (_isAbortError(e)) return false;
      _setMeshFailure(`raw_nj fetch error: ${e?.message || e}`);
      return false;
    }
    if (!buf || buf.byteLength < 8) {
      _setMeshFailure(`raw_nj ${modelPath}: empty/short buffer`);
      return false;
    }

    // Step 2: SINGLE parse (fix/perf). The old flow parsed twice over the
    // same buffer — once un-textured to discover texIds, then again with
    // the resolved texList. The two parses produced byte-identical
    // geometry/bones/skeleton; only mat.map differed (getModel() wires
    // `mat.map = texList[matList[i].texId]` at material-build time). So we
    // parse ONCE here (texList still empty — textures aren't fetched yet),
    // read texIds off loader.matList, fetch the textures, then assign
    // mat.map onto the EXISTING materials below. No second decode, no
    // second skeleton build.
    //
    // Motions are parsed in this single pass too; their raw buffers are
    // fetched in parallel with the geometry below and addAnimation() only
    // needs the bone tree, which this parse already built — but we must
    // have the motion buffers BEFORE getModel() runs (animations land on
    // mesh.geometry.animations during getModel). So we fetch textures and
    // motions first, then do the one parse. (Texture fetch needs texIds,
    // which we get from a tiny header-only parse below — but that parse is
    // discarded immediately and is the same ~1ms cost the old probe paid,
    // now WITHOUT a second full re-parse + texture-aware material build.)
    //
    // To keep the single decode honest we discover texIds via the loader's
    // matList from ONE parse and then re-wire textures, rather than parse a
    // throwaway probe AND a real mesh. We parse once into `mesh`, fetch in
    // parallel using its texIds, then assign mat.map.
    let parsed0;
    try {
      parsed0 = parseNinjaModel(buf, { name: modelPath, texList: [] });
    } catch (e) {
      _setMeshFailure(`psov2 parse error: ${e?.message || e}`);
      return false;
    }
    loader = parsed0.userData.ninjaLoader;
    if (!loader || !loader.bones.length) {
      _setMeshFailure(`psov2 ${modelPath}: no bones parsed`);
      return false;
    }
    texIds = (loader.matList || []).map((mm) => mm.texId).filter((n) => n >= 0);

    // Step 3: bind OUR tiles as the texList via the server-resolved
    // binding (handles player-body cross_afs textures, cross_bml, and
    // inline-XVM uniformly — see _psov2BuildTexList) AND fetch native
    // motions, both in parallel.
    archive = _psov2TextureArchive(modelPath);
    texList = [];
    fetchedTextures = [];
    psov2Binding = [];
    nativeMotions = null;

    const texPromise = texIds.length > 0
      ? _psov2BuildTexList(modelPath, texIds).catch((e) => {
          console.warn(`model_viewer: psov2 texlist fetch failed for ${archive}:`, e);
          return null;
        })
      : Promise.resolve(null);
    // Fetch the model's NATIVE .njm motions (list + raw NMDM bytes). The
    // bind-pose mesh is sized by this.bones (already built above), and each
    // motion's per-bone table is keyed by the SAME bone DFS order, so the
    // clips bind 1:1 to the SkinnedMesh skeleton. A failure is non-fatal —
    // the static bind pose still loads.
    const motionPromise = _fetchNativeMotions(modelPath).catch((e) => {
      if (_isAbortError(e)) return "__abort__";
      console.warn(`model_viewer: native motion fetch failed for ${modelPath}:`, e);
      return null;
    });

    const [built, motionsResult] = await Promise.all([texPromise, motionPromise]);
    if (motionsResult === "__abort__") return false;

    if (built) {
      texList = built.texList;
      fetchedTextures = built.fetched;
      psov2Binding = built.binding || [];
      archive = built.archive || archive;
    }
    nativeMotions = motionsResult || null;

    motionBuffers = [];
    motionNames = [];
    if (nativeMotions && nativeMotions.list.length) {
      for (const m of nativeMotions.list) {
        const buf2 = nativeMotions.buffers.get(m.name);
        if (buf2 && buf2.byteLength > 8) {
          motionBuffers.push(buf2);
          // psov2 names the clip from the BitStream name minus ".njm"; pass
          // the bare motion name so clip.name === m.name (the dropdown key).
          motionNames.push(`${m.name}.njm`);
        }
      }
    }

    // SINGLE-PARSE completion: we already have a parsed mesh (parsed0) with
    // built materials and a bound skeleton — but it has neither textures
    // nor animations yet. Assigning mat.map post-hoc (below) handles
    // textures. For animations, addAnimation() must run on the loader and
    // land on mesh.geometry.animations; getModel() already returned, so we
    // add the motions to the live loader and copy the resulting clips onto
    // the existing geometry. This avoids a second full geometry/skeleton
    // decode.
    mesh = parsed0;
    if (motionBuffers.length) {
      for (let i = 0; i < motionBuffers.length; i++) {
        try {
          const mbs = new _NinjaBitStream(motionNames[i] || `motion_${i}.njm`, motionBuffers[i]);
          loader.addAnimation(mbs);
        } catch (e) {
          console.warn(`model_viewer: motion ${i} parse failed:`, e);
        }
      }
      mesh.geometry.animations = loader.animList || [];
    }

    // Assign resolved textures onto the EXISTING materials (the single-
    // parse texture wire). getModel() built one MeshBasicMaterial per
    // matList entry; here we set mat.map = texList[texId] exactly as
    // getModel() would have on a textured parse.
    if (Array.isArray(mesh.material)) {
      const mats = mesh.material;
      const ml = loader.matList || [];
      for (let i = 0; i < mats.length && i < ml.length; i++) {
        const tid = ml[i].texId;
        const tex = (tid !== -1 && tid >= 0) ? texList[tid] : null;
        if (tex) {
          mats[i].map = tex;
          if (tex.transparent) {
            mats[i].transparent = true;
            mats[i].alphaTest = 0.05;
          }
          mats[i].needsUpdate = true;
        }
      }
    }

    // Populate the parsed-model cache for instant re-opens. Store the SLOW
    // inputs; re-parse is cheap. The cached textures are GPU-owned by the
    // cache and disposed only on LRU eviction (see _psov2CacheStore).
    _psov2CacheStore(modelPath, {
      buf,
      texIds,
      archive,
      texList,
      fetched: fetchedTextures,
      binding: psov2Binding,
      nativeMotions,
      motionBuffers,
      motionNames,
    });
  }

  // psov2 bakes the bind pose into vertex positions (vertex.applyMatrix4
  // (bone.matrixWorld) in readVertexChunk), so the geometry is already in
  // world space. Wrap in a Group and center + uniform-scale to fit the
  // camera, mirroring the skinned/world-baked normalize so framing is
  // consistent across paths.
  const group = new THREE.Group();
  group.add(mesh);

  // getModel() builds MeshBasicMaterials without honouring the current
  // wireframe toggle (unlike the world-baked path which bakes it in), so
  // re-apply state.wireframe if it's already on when this model loads.
  if (state.wireframe && Array.isArray(mesh.material)) {
    mesh.material.forEach((m) => { if (m) { m.wireframe = true; m.needsUpdate = true; } });
  }

  let aabb = new THREE.Box3();
  const posAttr = mesh.geometry.getAttribute("position");
  if (posAttr && posAttr.count > 0) {
    mesh.geometry.computeBoundingBox();
    aabb.copy(mesh.geometry.boundingBox);
  }
  if (!aabb.isEmpty()) {
    const c = aabb.getCenter(new THREE.Vector3());
    const sz = aabb.getSize(new THREE.Vector3());
    const maxDim = Math.max(sz.x, sz.y, sz.z, 0.001);
    const scale = 2.0 / maxDim;
    group.scale.set(scale, scale, scale);
    group.position.set(-c.x * scale, -c.y * scale, -c.z * scale);
  }

  // fix/perf — epoch guard. If a newer model open began while our
  // fetch+parse was in flight, DISCARD this result instead of clobbering
  // the newer model. Dispose the freshly-built geometry/materials we'd
  // otherwise have committed; do NOT dispose the textures (they're owned
  // by the parsed-model cache and may back the newer/other on-screen
  // model). This is the fix for the rapid A->B->C clobber where a stale
  // parse paints over the current model.
  if (_epochStale(openEpoch)) {
    try { mesh.geometry.dispose(); } catch {}
    if (Array.isArray(mesh.material)) {
      for (const m of mesh.material) { try { m.dispose(); } catch {} }
    }
    return false;
  }

  // fix/tooltabs — tag the single multi-material SkinnedMesh with material
  // identity so the paint binder (psoSetMaterialTexture) and the paint
  // hit-filter (paint_panel handlePaintAt via hit.face.materialIndex) can
  // resolve which material slot a click landed on. The psov2 mesh is ONE
  // SkinnedMesh with geometry.groups[] partitioned by materialIndex and a
  // parallel mesh.material[] array; loader.matList[i].texId === material_id
  // on the NJ path. We publish:
  //   userData.materialGroups : [{materialIndex, materialId}] (slot -> mid)
  //   userData.materialId     : the sole mid when there's exactly one mat
  //                             (lets the single-material fast paths still hit)
  {
    const ml = loader.matList || [];
    const groups = [];
    for (let i = 0; i < ml.length; i++) {
      groups.push({ materialIndex: i, materialId: (ml[i].texId | 0) });
    }
    mesh.userData.materialGroups = groups;
    if (groups.length === 1) {
      mesh.userData.materialId = groups[0].materialId;
    }
  }

  // Commit to the scene.
  ensureRenderer();
  disposeMesh();
  // Register textures in boundTextures so disposeMesh frees them on the
  // next model swap (avoids one-GPU-texture-per-open leak). Key by
  // material_id (== chunk texId on the NJ path) so the texture panel's
  // psoListMeshTextures()/psoReloadTexture() — which look up
  // state.boundTextures.get(material_id) — resolve the live texture.
  const boundMap = new Map();
  for (let texId = 0; texId < texList.length; texId++) {
    const t = texList[texId];
    if (t) boundMap.set(texId, t);
  }
  state.boundTextures = boundMap;
  state.boundTextureArchive = archive;
  // Expose the server binding so the texture-list panel populates (defect
  // #5) and the reset/upscale hooks can resolve tiles (defect #6). Rows
  // are {material_id, tile_index, source, ...}; psoListMeshTextures
  // buckets by tile_index.
  state.boundBinding = psov2Binding;
  state.mesh = group;
  state.meshGroup = group;
  state.realMesh = true;
  state.realMeshArchive = modelPath;
  state.scene.add(group);
  // fix/tooltabs — surface per-submesh debug meshes for the UV/Edit/
  // Subdivide inspectors. The psov2 mesh is ONE SkinnedMesh whose geometry
  // is partitioned by geometry.groups[] (one group per material). Build a
  // lightweight per-group VIEW mesh (sliced position+uv, NOT added to the
  // scene) so panels that read e.mesh.geometry.getAttribute("uv") see each
  // submesh's own UV island. Without this the UV/Edit tabs report "no
  // submesh" on every psov2 model.
  state.debugMeshes = _psov2BuildDebugMeshes(mesh);
  state.debugActiveIdx = -1;
  rebuildDebugSidebar();

  // The psov2 loader produces a real Skeleton-bound SkinnedMesh whose
  // geometry.animations hold any motions. We do NOT drive the harness'
  // server-payload CPU bone-bake pipeline (state.anim.skinned stays
  // false); instead, native .njm motions ride mesh.geometry.animations as
  // real THREE.AnimationClips and we play them with a THREE.AnimationMixer
  // (THREE's SkinnedMesh + Skeleton auto-skin from skinIndex/skinWeight on
  // the GPU as the mixer re-poses the bones). A model with NO motions
  // simply renders the static bind pose.
  state.anim.skinned = false;
  state.anim.modelPath = modelPath;
  // fix/tooltabs — mirror the live THREE.Skeleton into state.anim.bones so
  // psoGetSkeleton() (and the Skeleton tab) sees the 88-bone tree the psov2
  // SkinnedMesh actually carries. The bones are kept in the panel's
  // {index,parent,position,rotation_bams,scale,eval_flags} shape, adapted
  // from the THREE.Bone tree (Euler radians -> BAMS). state.anim.skinned
  // stays false (the GPU mixer, not the CPU bake, drives this path), so the
  // rig CPU-bake/world-pos accessors still guard correctly — these bones are
  // a read-only mirror for the Skeleton inspector.
  state.anim.bones = _psov2AdaptSkeletonBones(mesh);
  state.anim.psov2Mesh = mesh;
  state.anim.skinSubmeshes = [];
  state.anim.motions = [];
  state.anim.currentMotion = null;
  state.anim.currentData = null;

  // Wire the psov2 native-motion mixer. _installPsov2Motions builds the
  // name->clip map, populates state.anim.motions + the dropdown, creates
  // the mixer, and auto-plays the default motion. It hides the anim bar
  // gracefully when there are no clips.
  _installPsov2Motions(mesh, nativeMotions);

  kick();

  let triCount = 0;
  if (mesh.geometry.index) {
    triCount = mesh.geometry.index.count / 3;
  } else if (posAttr) {
    triCount = posAttr.count / 3;
  }
  setMeshStats(
    `ninja (psov2): verts ${posAttr ? posAttr.count : 0}  tris ${triCount | 0}  ` +
    `bones ${loader.bones.length}  mats ${loader.matList.length}  ` +
    `tex ${fetchedTextures.length}/${texIds.length}`,
  );
  setStatus(`ninja model ${modelPath} loaded (psov2 path)`);
  _setMeshFailure(null);
  return true;
}

// Build the /api endpoint URL base for a model path, handling the
// `<bml>#<inner>.nj` form (same convention as populateAnimationPanel /
// loadMotion). Returns { base, innerQuery } where innerQuery is an
// `&inner=...`-ready string ("" when there's no inner).
function _modelApiUrlParts(modelPath) {
  const hashIdx = modelPath.indexOf("#");
  if (hashIdx > 0) {
    const base = modelPath.slice(0, hashIdx);
    let inner = modelPath.slice(hashIdx + 1);
    if (inner.toLowerCase().endsWith(".xvm")) inner = inner.slice(0, -4);
    return {
      base: encodeURIComponent(base),
      innerQuery: `inner=${encodeURIComponent(inner)}`,
    };
  }
  return { base: encodeURIComponent(modelPath), innerQuery: "" };
}

/**
 * Fetch a model's NATIVE .njm motions for the psov2 path: the listing
 * (/api/animations) plus each motion's RAW NMDM bytes
 * (/api/animation_njm), which psov2_ninja's readAnim turns into real
 * THREE.AnimationClips.
 *
 * Returns null when the model has no motions. On success returns
 *   { list: [{name, frame_count, fps, ...}], default: <name|null>,
 *     buffers: Map<name, ArrayBuffer> }.
 * Motions whose raw fetch fails are dropped (but never sink the others).
 */
async function _fetchNativeMotions(modelPath, signal) {
  const { base, innerQuery } = _modelApiUrlParts(modelPath);
  const q = innerQuery ? `?${innerQuery}` : "";

  // fix/perf — preload path passes its own AbortSignal: fetch on THAT
  // signal and SKIP the prefetchModelBundle shortcut (which mutates the
  // shared _bundleCache / _lastBundlePath and would disturb the active
  // open's bundle). The interactive path keeps the bundle shortcut.
  const _fetch = signal
    ? (url, init) => fetch(url, Object.assign({}, init, { signal }))
    : _lifecycleFetch;

  // Prefer the bundle's animations sub-payload if it's already prefetched
  // (saves a redundant /api/animations round-trip). Bundle never carries
  // raw NMDM bytes, so we still fetch those per-motion below.
  let listData = null;
  if (!signal) {
    try {
      const bundle = await prefetchModelBundle(modelPath);
      if (bundle && bundle.animations) listData = bundle.animations;
    } catch (_e) { /* fall through to direct fetch */ }
  }

  if (!listData) {
    const r = await _fetch(`/api/animations/${base}${q}`);
    if (!r.ok) return null;
    listData = await r.json();
  }
  if (!listData || !Array.isArray(listData.motions) || listData.motions.length === 0) {
    return null;
  }

  // Fetch raw NMDM bytes for every motion in parallel (bounded by the
  // motion count, typically < 12). Each uses the SAME ?motion= resolver
  // as the listing, so names line up 1:1.
  const buffers = new Map();
  await Promise.all(listData.motions.map(async (m) => {
    const url = `/api/animation_njm/${base}?${innerQuery ? innerQuery + "&" : ""}motion=${encodeURIComponent(m.name)}`;
    try {
      const rr = await _fetch(url);
      if (!rr.ok) return;
      const ab = await rr.arrayBuffer();
      if (ab && ab.byteLength > 8) buffers.set(m.name, ab);
    } catch (e) {
      if (!_isAbortError(e)) {
        console.warn(`model_viewer: njm raw fetch failed for ${m.name}:`, e);
      }
    }
  }));

  if (buffers.size === 0) return null;

  // Default motion name (walk/idle/etc. per the backend resolver).
  let defaultName = null;
  if (listData.default_index != null && listData.default_index >= 0 &&
      listData.default_index < listData.motions.length) {
    const d = listData.motions[listData.default_index];
    if (d && buffers.has(d.name)) defaultName = d.name;
  }
  // Fall back to the first motion that actually decoded.
  if (!defaultName) {
    for (const m of listData.motions) {
      if (buffers.has(m.name)) { defaultName = m.name; break; }
    }
  }

  return { list: listData.motions, default: defaultName, buffers };
}

/**
 * Install the native-motion mixer onto the just-loaded psov2 SkinnedMesh.
 *
 * - Maps each clip on mesh.geometry.animations by name (== motion name).
 * - Populates state.anim.motions (descriptors) + the Motions dropdown.
 * - Creates a THREE.AnimationMixer bound to the mesh.
 * - Applies the autoload rule (1 motion -> play it; >1 -> play the
 *   backend-resolved default = walk/idle).
 *
 * No-op (hides the anim bar) when there are no clips.
 */
function _installPsov2Motions(mesh, nativeMotions) {
  const a = state.anim;
  const animBar = $("#modelAnimBar");

  const clips = (mesh.geometry && mesh.geometry.animations) || [];
  if (!nativeMotions || !clips.length) {
    if (animBar) animBar.hidden = true;
    return;
  }

  // Index clips by name. psov2 names each clip from the .njm filename
  // minus ".njm", which equals the motion name from /api/animations.
  const clipMap = new Map();
  for (const c of clips) {
    if (c && c.name) clipMap.set(c.name, c);
  }
  // Keep only motion descriptors that have a real clip (dropped/failed
  // motions never reach the dropdown).
  const motionDescs = nativeMotions.list.filter((m) => clipMap.has(m.name));
  if (motionDescs.length === 0) {
    if (animBar) animBar.hidden = true;
    return;
  }

  a.psov2 = true;
  a.psov2Clips = clipMap;
  a.psov2Mixer = new THREE.AnimationMixer(mesh);
  a.psov2Action = null;
  a.psov2Clock = 0;
  a.motions = motionDescs;
  a.loop = true;

  // Populate the Motions dropdown (#modelAnimSel). Mirrors
  // populateAnimationPanel's option layout so the existing change handler
  // (wired in wireAnimationPanel) routes picks through loadMotion ->
  // psov2 branch.
  const sel = $("#modelAnimSel");
  if (sel) {
    sel.innerHTML = "";
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = "(bind pose)";
    sel.appendChild(blank);
    for (const m of motionDescs) {
      const opt = document.createElement("option");
      opt.value = m.name;
      opt.textContent = `${m.name}  (${m.frame_count}f @ ${m.fps}fps)`;
      sel.appendChild(opt);
    }
  }
  if (animBar) animBar.hidden = false;

  // Autoload rule: 1 motion -> play it; >1 -> play the resolved default
  // (walk/idle per the backend tier ranker). nativeMotions.default is
  // already the right pick; fall back to the sole motion when there's
  // exactly one.
  let autoName = nativeMotions.default;
  if (!autoName && motionDescs.length === 1) autoName = motionDescs[0].name;
  if (autoName && clipMap.has(autoName)) {
    if (sel) sel.value = autoName;
    _psov2PlayMotion(autoName);
  } else {
    updateAnimationUi();
  }
}

/**
 * Start (or restart) playback of a named native motion on the psov2
 * mixer. Stops any current action, fades the new clip in, and flips the
 * render-loop animation flags so shouldAnimateContinuously() keeps the
 * loop running. Returns the motion descriptor (or null).
 */
function _psov2PlayMotion(motionName) {
  const a = state.anim;
  if (!a.psov2 || !a.psov2Mixer || !a.psov2Clips) return null;

  if (!motionName) {
    // Reset to bind pose: stop the action, leave the skeleton at rest.
    if (a.psov2Action) { try { a.psov2Action.stop(); } catch {} }
    a.psov2Action = null;
    a.currentMotion = null;
    a.currentData = null;
    a.playing = false;
    a.psov2Mixer.update(0);        // settle bones back toward bind pose
    updateAnimationUi();
    kick();
    return null;
  }

  const clip = a.psov2Clips.get(motionName);
  if (!clip) {
    setAnimStatus(`unknown motion: ${motionName}`);
    return null;
  }
  const desc = a.motions.find((m) => m.name === motionName) || { name: motionName };

  if (a.psov2Action) { try { a.psov2Action.stop(); } catch {} }
  const action = a.psov2Mixer.clipAction(clip);
  action.reset();
  action.setLoop(a.loop ? THREE.LoopRepeat : THREE.LoopOnce, Infinity);
  action.clampWhenFinished = !a.loop;
  action.timeScale = 1.0;
  action.play();

  a.psov2Action = action;
  a.currentMotion = desc;
  // currentData mirrors a minimal descriptor so updateAnimationUi() /
  // psoGetCurrentMotion() / the scrub label read sensible values. The
  // psov2 path is time-driven by the mixer, not by currentData.bones.
  a.currentData = { frame_count: desc.frame_count || Math.max(1, Math.round((clip.duration || 0) * 30)) + 1 };
  a.fps = desc.fps || 30.0;
  a.time = 0.0;
  a.playing = true;
  a.psov2Clock = 0;

  const scrub = $("#modelAnimScrub");
  if (scrub) {
    scrub.max = String(Math.max(1, (a.currentData.frame_count || 1) - 1));
    scrub.value = "0";
  }
  setAnimStatus(`playing ${motionName}`);
  updateAnimationUi();
  kick();
  return desc;
}

/**
 * Build a THREE.AnimationClip from the JSON keyframe payload returned by
 * /api/animation_data or /api/anim_preview/data (shape:
 *   { frame_count, fps, bones:[{idx, present, kf:[{t,tx,ty,tz,rx,ry,rz,
 *     sx,sy,sz, (qw,qx,qy,qz)?}]}] }).
 *
 * Tracks are named `.bones[bone_<i>].{position|quaternion|scale}` so they
 * bind by name against the psov2 SkinnedMesh skeleton (whose bones are
 * `bone_<DFS-index>`). For bones/channels with no keyframes we fall back
 * to the bound bone's BIND-POSE TRS (so a rotation-only motion doesn't
 * yank untracked bones to the origin) — mirroring the server's per-bone
 * `present` contract and psov2's first/last-frame rest fill.
 *
 * Rotation is BAMS Euler XYZ (rx/ry/rz are signed BAMS units; radians =
 * unit * _BAMS_TO_RAD), applied X then Y then Z via Euler order "XYZ" —
 * the same composition _computeAnimatedBoneMatrices uses. The optional
 * quaternion channel (qw,qx,qy,qz) overrides the Euler rotation.
 *
 * Returns null when the payload has no usable bone tracks.
 */
function _buildClipFromAnimData(data, clipName) {
  if (!data || !Array.isArray(data.bones) || data.bones.length === 0) return null;
  const mesh = state.mesh;
  if (!mesh) return null;

  // Resolve the skeleton's bones by name so we can read bind-pose TRS.
  const skelBones = new Map();
  mesh.traverse((o) => {
    if (o.isBone && o.name) skelBones.set(o.name, o);
  });

  const fps = data.fps || 30.0;
  const tracks = [];
  const _q = new THREE.Quaternion();
  const _e = new THREE.Euler();

  for (let bi = 0; bi < data.bones.length; bi++) {
    const boneEntry = data.bones[bi];
    const boneName = `bone_${String(bi).padStart(3, "0")}`;
    const bone = skelBones.get(boneName);
    const kf = (boneEntry && Array.isArray(boneEntry.kf)) ? boneEntry.kf : [];
    if (kf.length === 0) continue;  // untracked bone -> stays at bind pose

    const present = (boneEntry && typeof boneEntry.present === "number")
      ? boneEntry.present : 0xFFFF;  // legacy: assume all channels present
    const hasPos  = !!(present & _NJM_PRESENT_POS);
    const hasAng  = !!(present & _NJM_PRESENT_ANG);
    const hasScl  = !!(present & _NJM_PRESENT_SCL);
    const hasQuat = !!(present & _NJM_PRESENT_QUAT);

    // Bind-pose TRS for channel fallback.
    const bp = bone ? bone.position : { x: 0, y: 0, z: 0 };
    const bq = bone ? bone.quaternion : { x: 0, y: 0, z: 0, w: 1 };
    const bs = bone ? bone.scale : { x: 1, y: 1, z: 1 };

    const times = new Float32Array(kf.length);
    const posArr = new Float32Array(kf.length * 3);
    const quatArr = new Float32Array(kf.length * 4);
    const sclArr = new Float32Array(kf.length * 3);

    for (let k = 0; k < kf.length; k++) {
      const f = kf[k];
      times[k] = (typeof f.t === "number" ? f.t : k) / fps;

      // Position.
      if (hasPos) {
        posArr[k * 3 + 0] = f.tx; posArr[k * 3 + 1] = f.ty; posArr[k * 3 + 2] = f.tz;
      } else {
        posArr[k * 3 + 0] = bp.x; posArr[k * 3 + 1] = bp.y; posArr[k * 3 + 2] = bp.z;
      }

      // Rotation.
      if (hasQuat && f.qw != null) {
        quatArr[k * 4 + 0] = f.qx; quatArr[k * 4 + 1] = f.qy;
        quatArr[k * 4 + 2] = f.qz; quatArr[k * 4 + 3] = f.qw;
      } else if (hasAng) {
        _e.set(f.rx * _BAMS_TO_RAD, f.ry * _BAMS_TO_RAD, f.rz * _BAMS_TO_RAD, "XYZ");
        _q.setFromEuler(_e);
        quatArr[k * 4 + 0] = _q.x; quatArr[k * 4 + 1] = _q.y;
        quatArr[k * 4 + 2] = _q.z; quatArr[k * 4 + 3] = _q.w;
      } else {
        quatArr[k * 4 + 0] = bq.x; quatArr[k * 4 + 1] = bq.y;
        quatArr[k * 4 + 2] = bq.z; quatArr[k * 4 + 3] = bq.w;
      }

      // Scale.
      if (hasScl) {
        sclArr[k * 3 + 0] = f.sx; sclArr[k * 3 + 1] = f.sy; sclArr[k * 3 + 2] = f.sz;
      } else {
        sclArr[k * 3 + 0] = bs.x; sclArr[k * 3 + 1] = bs.y; sclArr[k * 3 + 2] = bs.z;
      }
    }

    tracks.push(new THREE.VectorKeyframeTrack(`.bones[${boneName}].position`, times, posArr));
    tracks.push(new THREE.QuaternionKeyframeTrack(`.bones[${boneName}].quaternion`, times, quatArr));
    tracks.push(new THREE.VectorKeyframeTrack(`.bones[${boneName}].scale`, times, sclArr));
  }

  if (tracks.length === 0) return null;
  const dur = Math.max(0.0001, (data.frame_count || 1) / fps);
  return new THREE.AnimationClip(clipName || "imported", dur, tracks);
}

/**
 * Try the SKINNED model path, falling back to the regular path on
 * failure. Returns true on success.
 *
 * Distinct from tryLoadRealMesh in that:
 *   - Only `.nj` is supported (skinned path requires bone-local data).
 *   - The animation panel is populated as a side effect.
 *   - state.anim.skinned is set so the render loop knows to re-bake.
 */
async function tryLoadSkinnedMesh(modelPath, hint) {
  // Resolve the URL — support both `<bml>#<inner>.nj` and bare `.nj`.
  let url;
  let label;
  let archiveLabel = modelPath;
  const hashIdx = modelPath.indexOf("#");
  if (hashIdx > 0) {
    const base = modelPath.slice(0, hashIdx);
    let inner = modelPath.slice(hashIdx + 1);
    // Defensive .xvm strip (asset router may pass texture path)
    if (inner.toLowerCase().endsWith(".xvm")) {
      inner = inner.slice(0, -4);
    }
    if (!inner.toLowerCase().endsWith(".nj")) {
      _setMeshFailure(`skinned path requires .nj inner (got ${inner})`);
      return false;
    }
    url = `/api/model_skinned/${encodeURIComponent(base)}?inner=${encodeURIComponent(inner)}`;
    label = `${base} :: ${inner}`;
  } else if (modelPath.toLowerCase().endsWith(".nj")) {
    url = `/api/model_skinned/${encodeURIComponent(modelPath)}`;
    label = modelPath;
  } else {
    _setMeshFailure(`skinned path requires .nj (got ${modelPath.split(".").pop()})`);
    return false;
  }

  setStatus(`loading skinned mesh ${label}...`);
  let payload;

  // Bundle fast-path: if /api/model_bundle gave us this model's skinned
  // payload already (or has one in flight), skip the per-endpoint
  // fetch entirely. Falls back transparently when the bundle fetch
  // 404s (older server build) or when the bundle's `errors.skinned`
  // field is set (parse failure on the backend).
  let bundleSkinned = null;
  try {
    const bundle = await prefetchModelBundle(modelPath);
    if (bundle && bundle.skinned) bundleSkinned = bundle.skinned;
  } catch (_e) { /* fall through to per-endpoint path */ }

  if (bundleSkinned) {
    payload = bundleSkinned;
  } else {
    try {
      const r = await _lifecycleFetch(url);
      if (!r.ok) {
        let detail = `http ${r.status}`;
        try { const eb = await r.json(); if (eb && eb.detail) detail = eb.detail; } catch {}
        _setMeshFailure(`skinned: ${detail}`);
        return false;
      }
      payload = await r.json();
    } catch (e) {
      if (_isAbortError(e)) return false;
      _setMeshFailure(`skinned fetch error: ${e?.message || e}`);
      return false;
    }
  }
  if (!payload || !payload.mesh_count || !payload.bone_count) {
    _setMeshFailure(`skinned ${label}: empty payload`);
    return false;
  }

  // Per-submesh texture binding — same logic as world-baked path.
  const newBinding = payload.binding || [];
  let newBoundTextures = new Map();
  let newBoundArchive = null;
  if (newBinding.length > 0) {
    const archive = deriveTextureArchivePath(url, state.filename, hint);
    if (archive) {
      try {
        newBoundTextures = await fetchBoundTextures(archive, newBinding);
        newBoundArchive = archive;
      } catch (e) {
        console.warn(`model_viewer: bound-texture fetch failed for ${archive}:`, e);
      }
    }
  }

  const built = buildSkinnedMeshGroupFromPayload(payload, newBoundTextures);
  if (!built.group.children.length) {
    _setMeshFailure(`skinned ${label}: no rendered sub-meshes`);
    for (const t of newBoundTextures.values()) {
      try { t.dispose(); } catch {}
    }
    return false;
  }

  ensureRenderer();
  disposeMesh();
  state.boundTextures = newBoundTextures;
  state.boundTextureArchive = newBoundArchive;
  state.boundBinding = newBinding;
  state.mesh = built.group;
  state.meshGroup = built.group;
  state.realMesh = true;
  state.scene.add(state.mesh);
  state.debugMeshes = built.debugMeshes || [];
  state.debugActiveIdx = -1;
  rebuildDebugSidebar();
  if (state.debugMode) applyDebugMaterials(true);

  // Wire animation state.
  state.anim.skinned = true;
  state.anim.modelPath = modelPath;
  state.anim.bones = payload.bones;
  state.anim.skinSubmeshes = built.skinSubmeshes;
  state.anim.currentMotion = null;
  state.anim.currentData = null;
  state.anim.time = 0.0;
  state.anim.playing = false;
  state.anim.lastTimestamp = 0;

  kick();

  setMeshStats(
    `skinned mesh: ${payload.mesh_count} sub  verts ${built.totalVerts}  ` +
    `tris ${built.totalTris}  bones ${payload.bone_count}`,
  );
  setStatus(`skinned mesh ${label}: ${payload.mesh_count} sub-meshes loaded`);
  _setMeshFailure(null);
  state.realMeshArchive = archiveLabel;

  // Populate the animation panel asynchronously; if no motions are
  // found, the panel stays hidden.
  populateAnimationPanel(modelPath).catch((e) =>
    console.warn("model_viewer: animation panel failed:", e),
  );

  return true;
}

/**
 * Fetch /api/animations and populate the motion dropdown. If a
 * default motion is identified (walk > idle > etc.), auto-load it via
 * loadMotion().
 *
 * Hides the animation bar if no motions are found (graceful no-op for
 * static props).
 */
async function populateAnimationPanel(modelPath) {
  // URL form mirrors tryLoadSkinnedMesh.
  let url;
  const hashIdx = modelPath.indexOf("#");
  if (hashIdx > 0) {
    const base = modelPath.slice(0, hashIdx);
    let inner = modelPath.slice(hashIdx + 1);
    if (inner.toLowerCase().endsWith(".xvm")) inner = inner.slice(0, -4);
    url = `/api/animations/${encodeURIComponent(base)}?inner=${encodeURIComponent(inner)}`;
  } else {
    url = `/api/animations/${encodeURIComponent(modelPath)}`;
  }

  // Bundle fast-path: reuse the bundle's animations sub-payload if
  // present so we don't issue a redundant /api/animations fetch.
  let data = null;
  try {
    const bundle = await prefetchModelBundle(modelPath);
    if (bundle && bundle.animations) data = bundle.animations;
  } catch (_e) { /* fall through */ }

  if (!data) {
    try {
      const r = await _lifecycleFetch(url);
      if (!r.ok) return;
      data = await r.json();
    } catch (_) {
      // AbortError or net failure — silently skip; user moved on.
      return;
    }
  }
  if (!data || !data.motions || data.motions.length === 0) {
    // No motions — hide the panel.
    const bar = $("#modelAnimBar");
    if (bar) bar.hidden = true;
    return;
  }
  state.anim.motions = data.motions;

  // Populate dropdown when present. The dropdown is a UI affordance; auto-
  // play below must NOT be gated on its existence — otherwise a perspective
  // that hides/relocates the modal (unified-viewport, tile-detail, etc.)
  // silently breaks the "model walks on load" contract. Regression
  // 2026-04-25: an early `if (!sel) return` here was bailing before the
  // auto-load when the dropdown wasn't in the DOM, so dragons / monsters
  // sat motionless in bind pose. Auto-play is now decoupled.
  const sel = $("#modelAnimSel");
  if (sel) {
    sel.innerHTML = "";
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = "(bind pose)";
    sel.appendChild(blank);
    for (const m of data.motions) {
      const opt = document.createElement("option");
      opt.value = m.name;
      opt.textContent = `${m.name}  (${m.frame_count}f @ ${m.fps}fps)`;
      sel.appendChild(opt);
    }
  }

  // Show the bar.
  const bar = $("#modelAnimBar");
  if (bar) bar.hidden = false;

  // Auto-load the default motion (walk first by spec). Falls back to
  // "(bind pose)" if no default match. NOTE: this fires regardless of
  // whether the dropdown exists — the underlying state.anim.playing flag
  // is the source of truth for the render loop, not the UI.
  if (data.default_index != null && data.default_index >= 0) {
    const def = data.motions[data.default_index];
    if (sel) sel.value = def.name;
    await loadMotion(def.name);
  }
}

/**
 * Fetch /api/animation_data for `motionName`, populate
 * `state.anim.currentData`, and start playback at frame 0.
 *
 * Empty `motionName` resets to bind pose (no animation; static).
 */
async function loadMotion(motionName) {
  const a = state.anim;

  // psov2 native-motion path: clips are already parsed + mounted on the
  // mixer (no per-motion fetch). Route through the mixer player. The empty
  // -name case (reset to bind pose) is handled inside _psov2PlayMotion.
  if (a.psov2 && a.psov2Mixer) {
    _psov2PlayMotion(motionName || "");
    return;
  }

  if (!motionName) {
    // Reset to bind pose.
    a.currentMotion = null;
    a.currentData = null;
    a.playing = false;
    a.time = 0.0;
    if (a.skinned && a.bones.length > 0 && a.skinSubmeshes.length > 0) {
      _computeAnimatedBoneMatrices(a.bones, null, null);
      _bakeSkinnedSubmeshes(a.skinSubmeshes, a.bones.length);
    }
    updateAnimationUi();
    // Bind-pose reset: not playing, so just paint the bind pose once.
    kick();
    return;
  }
  if (!a.modelPath) return;
  const motion = a.motions.find((m) => m.name === motionName);
  if (!motion) {
    setAnimStatus(`unknown motion: ${motionName}`);
    return;
  }
  // Build URL same way as populateAnimationPanel.
  const hashIdx = a.modelPath.indexOf("#");
  let url;
  if (hashIdx > 0) {
    const base = a.modelPath.slice(0, hashIdx);
    let inner = a.modelPath.slice(hashIdx + 1);
    if (inner.toLowerCase().endsWith(".xvm")) inner = inner.slice(0, -4);
    url = `/api/animation_data/${encodeURIComponent(base)}?inner=${encodeURIComponent(inner)}&motion=${encodeURIComponent(motionName)}`;
  } else {
    url = `/api/animation_data/${encodeURIComponent(a.modelPath)}?motion=${encodeURIComponent(motionName)}`;
  }
  setAnimStatus(`loading motion ${motionName}...`);

  // Bundle fast-path: if the bundle prefetch included this motion's
  // keyframe data (only the default motion is bundled by
  // ?include_motion=default), reuse it.  Subsequent motion picks fall
  // back to the per-endpoint /api/animation_data fetch.
  let data = null;
  try {
    const bundle = await prefetchModelBundle(a.modelPath);
    const md = bundle && bundle.motion_data;
    if (md && md.motion && md.motion.toLowerCase() === motionName.toLowerCase()) {
      data = md;
    }
  } catch (_e) { /* fall through */ }

  if (!data) {
    try {
      const r = await _lifecycleFetch(url);
      if (!r.ok) {
        let detail = `http ${r.status}`;
        try { const eb = await r.json(); if (eb && eb.detail) detail = eb.detail; } catch {}
        setAnimStatus(`motion load failed: ${detail}`);
        return;
      }
      data = await r.json();
    } catch (e) {
      if (_isAbortError(e)) return;
      setAnimStatus(`motion fetch error: ${e?.message || e}`);
      return;
    }
  }
  a.currentMotion = motion;
  a.currentData = data;
  a.fps = motion.fps || 30.0;
  a.time = 0.0;
  a.playing = true;
  a.lastTimestamp = 0;

  // Update UI.
  const fpsSel = $("#modelAnimFps");
  if (fpsSel) {
    // Find the closest matching option, or set to motion.fps if exact match.
    const fpsStr = String(a.fps | 0);
    let found = false;
    for (const opt of fpsSel.options) {
      if (opt.value === fpsStr) { found = true; break; }
    }
    if (found) fpsSel.value = fpsStr;
  }
  const scrub = $("#modelAnimScrub");
  if (scrub) {
    scrub.max = String(Math.max(1, data.frame_count - 1));
    scrub.value = "0";
  }
  setAnimStatus(`playing ${motionName}`);
  updateAnimationUi();
  // A motion just started playing -> start the continuous loop.
  kick();
}

/** Update the play/pause button label, scrub bar, time display. */
function updateAnimationUi() {
  const a = state.anim;
  const ppBtn = $("#modelAnimPlayPause");
  if (ppBtn) ppBtn.textContent = a.playing ? "pause" : "play";
  const scrub = $("#modelAnimScrub");
  const timeLabel = $("#modelAnimTime");
  if (a.currentData) {
    const fc = a.currentData.frame_count;
    const frame = Math.floor(a.time * a.fps) % Math.max(1, fc);
    if (scrub) {
      scrub.disabled = false;
      scrub.value = String(frame);
    }
    if (timeLabel) {
      const sec = (frame / a.fps).toFixed(2);
      timeLabel.textContent = `${frame}/${fc - 1} (${sec}s)`;
    }
  } else {
    if (scrub) {
      scrub.disabled = true;
      scrub.value = "0";
    }
    if (timeLabel) timeLabel.textContent = "0/0";
  }
}

function setAnimStatus(text) {
  const el = $("#modelAnimStatus");
  if (el) el.textContent = text || "";
}

/**
 * Per-frame animation tick. Called from the render loop. Advances
 * `state.anim.time` by `dt`, computes the current frame's bone
 * matrices, and re-bakes vertices into the GPU-bound BufferAttribute
 * arrays.
 *
 * No-op for primitives, world-baked meshes, and skinned meshes
 * without an active motion.
 */
function tickAnimation(nowMs) {
  const a = state.anim;

  // psov2 native-motion path: drive the THREE.AnimationMixer with real
  // wall-clock dt. The mixer re-poses the bound Skeleton; GPU skinning
  // re-deforms the SkinnedMesh automatically (no CPU vertex bake). This
  // branch is independent of the server-payload skinned path below.
  if (a.psov2 && a.psov2Mixer) {
    let mdt = 0;
    if (a.playing && a.psov2Clock > 0) {
      mdt = (nowMs - a.psov2Clock) / 1000.0;
      if (mdt > 0.25) mdt = 0.25;  // cap big gaps (tab inactive, etc.)
    }
    a.psov2Clock = nowMs;
    if (a.playing) {
      a.psov2Mixer.update(mdt);
      // Keep the scrub/time readout roughly in sync with the action.
      if (a.psov2Action) {
        a.time = a.psov2Action.time;
        updateAnimationUi();
        // Non-looping clip that has run to the end -> stop driving.
        if (!a.loop && a.psov2Action.paused) a.playing = false;
      }
    }
    return;
  }

  if (!a.skinned || a.bones.length === 0 || a.skinSubmeshes.length === 0) return;

  let dt = 0;
  if (a.playing && a.lastTimestamp > 0) {
    dt = (nowMs - a.lastTimestamp) / 1000.0;
    if (dt > 0.25) dt = 0.25;  // cap big gaps (tab inactive, etc.)
  }
  a.lastTimestamp = nowMs;

  if (a.playing && a.currentData) {
    a.time += dt;
    const fc = a.currentData.frame_count;
    const dur = fc / a.fps;
    if (a.loop) {
      a.time = a.time % dur;
      if (a.time < 0) a.time += dur;
    } else {
      if (a.time >= dur) {
        a.time = dur - 1.0 / a.fps;
        a.playing = false;
      }
    }
  }

  if (a.currentData) {
    const fc = a.currentData.frame_count;
    const frameT = (a.time * a.fps) % Math.max(1, fc);
    _computeAnimatedBoneMatrices(a.bones, a.currentData, frameT);
    _bakeSkinnedSubmeshes(a.skinSubmeshes, a.bones.length);
    updateAnimationUi();
  }
}

// Wire animation panel UI (called from init()).
function wireAnimationPanel() {
  const sel = $("#modelAnimSel");
  if (sel) {
    sel.addEventListener("change", (e) => loadMotion(e.target.value));
  }
  const ppBtn = $("#modelAnimPlayPause");
  if (ppBtn) {
    ppBtn.addEventListener("click", () => {
      const a = state.anim;
      if (!a.currentData) return;
      a.playing = !a.playing;
      a.lastTimestamp = 0;
      // psov2 path: reset the mixer dt clock so resuming doesn't jump by
      // the whole pause duration, and pause/resume the action.
      if (a.psov2) {
        a.psov2Clock = 0;
        if (a.psov2Action) a.psov2Action.paused = !a.playing;
      }
      updateAnimationUi();
      // Play -> start the continuous loop; pause -> requestRender for the
      // paused frame (loop self-stops on its next tick).
      kick();
    });
  }
  const scrub = $("#modelAnimScrub");
  if (scrub) {
    scrub.addEventListener("input", (e) => {
      const a = state.anim;
      if (!a.currentData) return;
      const frame = parseInt(e.target.value, 10) || 0;
      a.time = frame / a.fps;
      a.playing = false;
      a.lastTimestamp = 0;
      // psov2 path: seek the mixer to the scrubbed time and pose once.
      if (a.psov2 && a.psov2Action && a.psov2Mixer) {
        a.psov2Clock = 0;
        a.psov2Action.paused = false;
        a.psov2Action.time = a.time;
        a.psov2Mixer.update(0);
        a.psov2Action.paused = true;
        updateAnimationUi();
        requestRender();
        return;
      }
      // Force one tick to re-bake immediately, then paint it on-demand.
      tickAnimation(performance.now());
      requestRender();
    });
  }
  const fpsSel = $("#modelAnimFps");
  if (fpsSel) {
    fpsSel.addEventListener("change", (e) => {
      const a = state.anim;
      const v = parseFloat(e.target.value) || 30.0;
      a.fps = v;
    });
  }
  const loopChk = $("#modelAnimLoop");
  if (loopChk) {
    loopChk.addEventListener("change", (e) => {
      const a = state.anim;
      a.loop = !!e.target.checked;
      // psov2 path: re-apply loop mode to the live action.
      if (a.psov2 && a.psov2Action) {
        a.psov2Action.setLoop(a.loop ? THREE.LoopRepeat : THREE.LoopOnce, Infinity);
        a.psov2Action.clampWhenFinished = !a.loop;
      }
    });
  }
  const resetBtn = $("#modelAnimReset");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => loadMotion(""));
  }
  // Spacebar: play/pause when animation panel is visible.
  document.addEventListener("keydown", (e) => {
    const bar = $("#modelAnimBar");
    if (!bar || bar.hidden) return;
    if ($("#modelModal") && $("#modelModal").hidden) return;
    if (e.code === "Space" && document.activeElement &&
        document.activeElement.tagName !== "INPUT" &&
        document.activeElement.tagName !== "TEXTAREA" &&
        document.activeElement.tagName !== "SELECT") {
      e.preventDefault();
      if (ppBtn) ppBtn.click();
    }
  });
}

// Wire the UI on next tick (the original init() may have already run
// by the time this module finishes loading).
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", wireAnimationPanel);
} else {
  wireAnimationPanel();
}

// Expose the skinned-load entry point so the asset_router or tests can
// call it directly.
window.psoOpenSkinnedModel = async (modelPath) => {
  ensureRenderer();
  $("#modelModal").hidden = false;
  $("#modelModalTitle").textContent = modelPath;
  setStatus(`loading skinned model ${modelPath}...`);
  // psov2 faithful path first; legacy server-payload skinned as fallback.
  let ok = await tryLoadPsov2NinjaModel(modelPath, null);
  if (!ok) ok = await tryLoadSkinnedMesh(modelPath, null);
  if (!ok) {
    setStatus(`skinned load failed: ${state.lastMeshFailure || "unknown"}`);
  }
  return ok;
};

// =====================================================================
// Public texture-introspection exports (2026-04-25, additive).
//
// Surfaces the per-mesh texture binding so external panels (like the
// in-viewport texture-list panel) can render thumbnails + drive the
// upscale pipeline without touching state.boundTextures directly. Read-
// only — does NOT mutate render state.
// =====================================================================

// Return the archive path the currently-rendered mesh's textures load
// from. Shape examples:
//   "bm_ene_zu.bml#bm_ene_zu.nj.xvm"   (BML-inner XVM)
//   "ItemKT.afs#0007_ItemKT_0007.xvm"  (AFS-inner XVM)
//   "foo.xvm"                          (top-level)
//   null when no model is loaded.
window.psoGetCurrentTextureArchive = function () {
  return state.boundTextureArchive || null;
};

// Read-only diagnostic: flatten the currently-rendered mesh's materials
// into a plain summary (map presence, side, wireframe, vertexColors).
// Used by render-polish verification + future debugging; never mutates.
window.psoDebugMeshMaterials = function () {
  const mats = [];
  const collect = (m) => {
    if (!m) return;
    if (Array.isArray(m)) m.forEach(collect);
    else mats.push(m);
  };
  const root = state.mesh;
  if (root) {
    if (root.isGroup) root.traverse((c) => { if (c.isMesh) collect(c.material); });
    else collect(root.material);
  }
  return mats.map((m) => ({
    hasMap: !!m.map,
    mapSize: m.map && m.map.image ? `${m.map.image.width || 0}x${m.map.image.height || 0}` : null,
    side: m.side,
    wireframe: !!m.wireframe,
    vertexColors: !!m.vertexColors,
  }));
};

// Return the binding rows for the currently-rendered mesh:
//   [{material_id, tile_index, missing?, ...}, ...]
// Comes straight from /api/model_mesh's payload.binding (verbatim),
// so consumers see the same shape the model viewer itself reads.
window.psoGetTextureBinding = function () {
  return Array.isArray(state.boundBinding)
    ? state.boundBinding.slice()
    : [];
};

// One row per UNIQUE tile_index referenced by a binding entry. Useful
// for a texture-list panel that wants to show one row per file rather
// than per material slot. Format:
//   [{tile_index, material_ids: [n,...], width, height, archive,
//     thumbnail_url}]
// width/height are pulled from state.boundTextures[mid].image.{w,h}
// when available (THREE.Texture); otherwise null.
window.psoListMeshTextures = function () {
  const arch = state.boundTextureArchive;
  const binding = Array.isArray(state.boundBinding) ? state.boundBinding : [];
  if (binding.length === 0) return [];
  // Bucket bindings by the RESOLVED (archive, tile) so cross_afs rows
  // (player bodies — all tile_index 0 but distinct inner archives) show
  // as separate rows with correct thumbnails, not collapsed into one.
  // The bucket KEY is the resolved archive#tile; the displayed
  // tile_index/archive come from the resolution too.
  const byKey = new Map();   // key -> {archive, tile, mids: []}
  for (const b of binding) {
    if (b == null) continue;
    const resolved = resolveBindingRowTile(b, arch);
    if (!resolved) continue;
    const key = `${resolved.archive}${resolved.tile}`;
    if (!byKey.has(key)) {
      byKey.set(key, { archive: resolved.archive, tile: resolved.tile, mids: [] });
    }
    byKey.get(key).mids.push(b.material_id | 0);
  }
  const out = [];
  for (const { archive, tile, mids } of byKey.values()) {
    let w = null, h = null;
    // Pull dims from the THREE.Texture image (keyed by material_id ==
    // texId on the psov2 path). Every material_id in this bucket points
    // at the same texture object, so we only check the first one.
    const firstMid = mids[0];
    const tex = state.boundTextures && state.boundTextures.get
      ? state.boundTextures.get(firstMid)
      : null;
    if (tex && tex.image) {
      w = tex.image.width || tex.image.naturalWidth || null;
      h = tex.image.height || tex.image.naturalHeight || null;
    }
    out.push({
      tile_index: tile,
      material_ids: mids.slice().sort((a, b) => a - b),
      width: w,
      height: h,
      archive: archive,
      thumbnail_url: `/api/tile_png/${encodeURIComponent(archive)}/${tile}`,
    });
  }
  // Stable sort by archive then tile for deterministic display.
  out.sort((a, b) =>
    a.archive === b.archive
      ? a.tile_index - b.tile_index
      : (a.archive < b.archive ? -1 : 1),
  );
  return out;
};

// Force the currently-rendered mesh to drop its cached texture for
// `tile_index` and re-fetch. Used by the texture panel after an
// upscale completes — the cache-buster `?cb=` query in
// `/api/tile_png/.../<idx>` ensures the new pixels arrive.
window.psoReloadTexture = async function (tileIdx) {
  if (typeof tileIdx !== "number") return false;
  const arch = state.boundTextureArchive;
  if (!arch) return false;
  // fix/perf — a texture reload mutates the live bound textures, which the
  // parsed-model cache also references. Evict the on-screen model from the
  // cache so a future re-open re-fetches the edited tile instead of
  // re-binding a disposed/stale texture. (We do NOT dispose here — the
  // model is still on screen; eviction just drops the cache reference.)
  if (state.realMeshArchive) {
    _psov2ModelCache.delete(state.realMeshArchive);
  }
  // Find which material_ids point at this tile.
  const binding = Array.isArray(state.boundBinding) ? state.boundBinding : [];
  const matIds = binding
    .filter((b) => b && !b.missing && (b.tile_index | 0) === (tileIdx | 0))
    .map((b) => b.material_id | 0);
  if (matIds.length === 0) return false;
  // Pull a fresh texture (cache-busted via Date.now() timestamp inside
  // loadTileTexture) and substitute on every Mesh in state.meshGroup
  // whose material has the matching material_id.
  const fresh = await loadTileTexture(arch, tileIdx);
  if (!fresh) return false;
  // Dispose the old texture(s) for these material_ids and replace.
  for (const mid of matIds) {
    const old = state.boundTextures && state.boundTextures.get(mid);
    if (old) {
      try { old.dispose(); } catch {}
    }
    if (state.boundTextures && state.boundTextures.set) {
      state.boundTextures.set(mid, fresh);
    }
  }
  // Re-bind onto the live mesh via psoSetMaterialTexture, which handles BOTH
  // the legacy single-material mesh (c.material.map) AND the psov2
  // multi-material SkinnedMesh (c.material[slot].map via
  // userData.materialGroups). The old bespoke userData.materialId walk was a
  // silent NO-OP on the psov2 path: c.material is an Array there (so
  // `c.material.map = fresh` drops onto undefined) and userData.materialId is
  // undefined on the single multi-material mesh — so an upscale never showed.
  for (const mid of matIds) {
    window.psoSetMaterialTexture(mid, fresh);
  }
  // Paint the new texture on demand.
  requestRender();
  return true;
};

// Rebuild the currently-rendered mesh by invoking openByPath again.
// Used by the subdivide panel: after /api/model/subdivide writes a new
// NJ into cache/subdivided/, the panel calls this with the cached
// path. The caller is responsible for passing the manifest entry +
// matched textures so the texture binding survives the rebuild.
window.psoReloadModel = async function (modelPath, entry, matchedTextures) {
  // fix/perf — an explicit reload must bypass the parsed-model cache (the
  // user is asking for a fresh decode, e.g. after a texture edit or a
  // server-side asset change). Evict the resolved-path entries so the
  // open below re-fetches. We evict broadly (any key that shares the same
  // base) because a `<bml>#<inner>` reload should invalidate that inner.
  try { _psov2CacheEvictForPath(modelPath); } catch (_e) {}
  return openByPath(modelPath, entry || {}, Array.isArray(matchedTextures) ? matchedTextures : []);
};

// =====================================================================
// psoApplyMeshPayload — additive bridge for the subdivide panel.
//
// Takes a payload in the same wire shape as /api/model_mesh (filename,
// mesh_count, meshes[], totals, vertices_pre_transformed, binding) and
// renders it as the live mesh. Reuses the internal
// buildMeshGroupFromPayload + state-swap dance so the user sees the
// new (subdivided) geometry immediately. Re-applies the existing
// texture binding by RE-USING the already-loaded state.boundTextures
// — we don't re-fetch tile PNGs because the archive hasn't changed.
//
// Returns ``true`` on success, ``false`` on failure (with a status set
// via setStatus). The function is intentionally side-effect-only: no
// new state fields, no render-loop changes — it follows the exact
// pattern tryLoadRealMesh uses after a /api/model_mesh fetch.
// =====================================================================
window.psoApplyMeshPayload = function (payload, opts) {
  if (!payload || !payload.mesh_count) {
    setStatus("apply mesh payload: empty payload");
    return false;
  }
  // Build the new mesh group using the existing texture map. The
  // subdivide endpoint preserves material_id values across the
  // subdivision, so the same Map<material_id, THREE.Texture> applies.
  const reuseTextures = state.boundTextures || new Map();
  const built = buildMeshGroupFromPayload(payload, reuseTextures);
  if (!built.group.children.length) {
    setStatus("apply mesh payload: no rendered sub-meshes");
    return false;
  }
  ensureRenderer();
  // Stash the textures off the side BEFORE disposeMesh wipes them; we
  // intentionally want to keep these alive across the swap.
  const keepTextures = state.boundTextures;
  const keepArchive = state.boundTextureArchive;
  const keepBinding = state.boundBinding;
  state.boundTextures = new Map();   // disposeMesh sees an empty map -> no-op
  disposeMesh();
  state.boundTextures = keepTextures;
  state.boundTextureArchive = keepArchive;
  state.boundBinding = payload.binding && payload.binding.length ? payload.binding : keepBinding;
  state.mesh = built.group;
  state.meshGroup = built.group;
  state.realMesh = true;
  state.scene.add(state.mesh);
  state.debugMeshes = built.debugMeshes || [];
  state.debugActiveIdx = -1;
  if (typeof rebuildDebugSidebar === "function") rebuildDebugSidebar();
  if (state.debugMode && typeof applyDebugMaterials === "function") applyDebugMaterials(true);
  kick();

  const label = (opts && opts.label) || payload.filename || "(payload)";
  const aabbDx = built.aabbMax[0] - built.aabbMin[0];
  const aabbDy = built.aabbMax[1] - built.aabbMin[1];
  const aabbDz = built.aabbMax[2] - built.aabbMin[2];
  setMeshStats(
    `mesh: ${payload.mesh_count} sub  verts ${built.totalVerts}  tris ${built.totalTris}  ` +
    `aabb ${aabbDx.toFixed(2)}x${aabbDy.toFixed(2)}x${aabbDz.toFixed(2)}`,
  );
  setStatus(`mesh ${label}: ${payload.mesh_count} sub-meshes loaded`);
  return true;
};

// =====================================================================
// psoApplyVariantSlotOffset — additive helper for the variant picker.
//
// When a model packs multiple color variants into a single XVM (3 sets
// of 3 textures, etc. — see Mericarol/Mericus/Merikle), the geometry
// references material_id 0..N-1 and the renderer naturally binds each
// to tile_index = material_id. To switch to the BLUE variant without
// re-loading the model we just need to remap each material's `.map`
// to a higher tile_index — material_id 0 -> tile_index (0 + offset),
// material_id 1 -> tile_index (1 + offset), etc.
//
// This function does that. It re-fetches any tiles that aren't already
// in state.boundTextures and rebinds material.map on every Mesh in the
// live mesh group. The render loop is untouched — we only mutate
// material.map (which the existing tickAnimation / bone-matrix code
// never touches).
//
// Args:
//   slotOffset: integer offset to ADD to each material_id when looking
//               up tile_index. 0 = default (mericarol), 3 = blue,
//               6 = red for Mericarol.
//
// Returns a promise that resolves to true when all tiles are bound,
// false on error (e.g. no archive to fetch from).
// =====================================================================
window.psoApplyVariantSlotOffset = async function (slotOffset) {
  slotOffset = slotOffset | 0;
  const arch = state.boundTextureArchive;
  const binding = Array.isArray(state.boundBinding) ? state.boundBinding : [];
  if (!arch || binding.length === 0) return false;

  // Compute new (tile_index = material_id + offset) for every binding
  // row. Any new tile indices we don't already have in state.boundTextures
  // need to be fetched.
  const cache = state.boundTextures;
  const wantTiles = new Set();
  const remap = new Map(); // material_id -> new tile_index
  for (const b of binding) {
    if (!b || b.missing) continue;
    const mid = b.material_id | 0;
    const orig = b.tile_index | 0;
    // Reset behavior: when slotOffset==0, restore the ORIGINAL tile
    // mapping (= material_id) regardless of what was in the binding row.
    // For non-zero offsets, the new tile = (material_id + offset).
    const tgt = mid + slotOffset;
    remap.set(mid, tgt);
    wantTiles.add(tgt);
  }
  // Fetch any tiles we don't already have.
  const toFetch = [...wantTiles].filter((ti) => !cache.has(_pseudoMidForTile(ti)));
  // Use a simple "fetch each tile, store keyed by tile_index" pattern.
  // We deliberately do NOT touch state.boundTextures keys (those are
  // material_id-keyed). Instead build a tile_index-keyed cache local
  // to this remap operation.
  const tileTextures = new Map();
  // Pre-populate with already-loaded tiles (from boundTextures' values
  // we can't reverse-look-up by tile_index; just fetch them all — these
  // are 1024×1024 PNGs but the browser cache hits hard).
  await Promise.all([...wantTiles].map(async (ti) => {
    const tex = await loadTileTexture(arch, ti);
    if (tex) tileTextures.set(ti, tex);
  }));

  // Now walk the live mesh group and assign material.map = the
  // texture for (material_id + offset). Dispose of any old textures
  // we're displacing — but only if those textures aren't still
  // referenced by some OTHER material_id (would be a normal case if
  // multiple submeshes share a material).
  const newBoundTextures = new Map();
  for (const [mid, ti] of remap) {
    const t = tileTextures.get(ti);
    if (t) newBoundTextures.set(mid, t);
  }

  if (state.meshGroup) {
    state.meshGroup.traverse((c) => {
      if (!c.isMesh) return;
      const mid = c.userData && c.userData.materialId;
      if (mid == null) return;
      const t = newBoundTextures.get(mid | 0);
      if (t && c.material) {
        c.material.map = t;
        c.material.needsUpdate = true;
      }
    });
  }

  // Replace state.boundTextures with the new map. Old textures are
  // intentionally left alive (no .dispose) — they may still be in the
  // browser's THREE.TextureLoader cache and re-used on a subsequent
  // variant flip without re-fetching.
  state.boundTextures = newBoundTextures;
  state.variantSlotOffset = slotOffset;

  requestRender();
  return true;
};

// Stash for the texture panel: which slot offset is currently active.
// Reset to 0 on every model load (state.variantSlotOffset is set above).
window.psoGetVariantSlotOffset = function () {
  return state.variantSlotOffset | 0;
};

// No-op helper — not currently used. Kept for the cache check above to
// avoid syntax error.
function _pseudoMidForTile(ti) { return -1 - ti; }

// =====================================================================
// psoListMotions / psoLoadMotion — additive surface for the motion
// picker UI extension. Wraps state.anim's motion list (already populated
// by populateAnimationPanel via /api/animations) and exposes
// loadMotion() so the texture-panel-side scrollable picker can drive
// playback without depending on the dropdown.
// =====================================================================
window.psoListMotions = function () {
  return Array.isArray(state.anim && state.anim.motions)
    ? state.anim.motions.slice()
    : [];
};

window.psoLoadMotion = async function (motionNameOrData) {
  // Two-mode entry point. Original signature (string motion name) routes
  // to the regular per-name lookup + /api/animation_data fetch. A NEW
  // mode lets the texture-panel "Imported Animations" section play a
  // preview-only retargeted motion that has no entry in
  // ``state.anim.motions`` and no /api/animation_data row: when the
  // caller passes an OBJECT shaped like the /api/animation_data /
  // /api/anim_preview/data response, we install it directly as
  // currentData and start playback. This keeps preview-imports
  // viewport-only (no BML repack, no game-data write).
  if (motionNameOrData && typeof motionNameOrData === "object" && Array.isArray(motionNameOrData.bones)) {
    const a = state.anim;
    if (!a) return null;
    // Synthesise a motion descriptor matching what
    // ``state.anim.motions`` entries look like, so
    // ``psoGetCurrentMotion`` etc. read back a sensible name.
    const data = motionNameOrData;
    const name = data.motion || data.filename || "imported";
    a.currentMotion = {
      name,
      frame_count: data.frame_count | 0,
      fps: data.fps || 30.0,
      bone_count: data.bone_count | 0,
      type_flags: data.type_flags | 0,
      interpolation: data.interpolation | 0,
      source_path: data.source_path || "",
      imported: true,
    };
    a.currentData = data;
    a.fps = data.fps || 30.0;
    a.time = 0.0;
    a.playing = true;
    a.lastTimestamp = 0;
    // psov2 path: the CPU-bake tickAnimation doesn't run (no skinSubmeshes),
    // so JSON-keyframe currentData alone won't move the mesh. Build a real
    // THREE.AnimationClip from the keyframes and play it on the mixer so
    // imported previews animate on psov2 models too.
    if (a.psov2 && a.psov2Mixer && state.mesh) {
      try {
        const clip = _buildClipFromAnimData(data, name);
        if (clip) {
          if (!a.psov2Clips) a.psov2Clips = new Map();
          a.psov2Clips.set(name, clip);
          if (a.psov2Action) { try { a.psov2Action.stop(); } catch {} }
          const action = a.psov2Mixer.clipAction(clip);
          action.reset();
          action.setLoop(a.loop ? THREE.LoopRepeat : THREE.LoopOnce, Infinity);
          action.clampWhenFinished = !a.loop;
          action.timeScale = 1.0;
          action.play();
          a.psov2Action = action;
          a.psov2Clock = 0;
          a.currentData = { frame_count: data.frame_count | 0 };
          setAnimStatus(`playing ${name} (imported)`);
        }
      } catch (e) {
        console.warn("model_viewer: imported-preview clip build failed:", e);
      }
    }
    if (typeof updateAnimationUi === "function") updateAnimationUi();
    kick();
    return a.currentMotion;
  }
  if (typeof loadMotion === "function") {
    return loadMotion(motionNameOrData || "");
  }
  return null;
};

window.psoGetCurrentMotion = function () {
  const a = state.anim || {};
  return a.currentMotion ? a.currentMotion.name : null;
};

// Animation introspection hook (read-only). Surfaces the active playback
// driver (psov2 mixer vs the server-payload CPU bake) and the live action
// state so the headless self-verify (and manual debugging) can assert that
// a native motion is actually running. No side effects.
window.psoGetAnimDebug = function () {
  const a = state.anim || {};
  return {
    psov2: !!a.psov2,
    skinned: !!a.skinned,
    hasMixer: !!a.psov2Mixer,
    hasAction: !!a.psov2Action,
    clipCount: a.psov2Clips ? a.psov2Clips.size : 0,
    motions: a.motions ? a.motions.length : 0,
    playing: !!a.playing,
    current: a.currentMotion ? a.currentMotion.name : null,
    modelPath: a.modelPath,
    actionTime: a.psov2Action ? a.psov2Action.time : null,
    actionRunning: a.psov2Action ? a.psov2Action.isRunning() : null,
  };
};

window.psoGetAnimationPlaying = function () {
  return !!(state.anim && state.anim.playing);
};

window.psoSetAnimationPlaying = function (playing) {
  if (state.anim) {
    state.anim.playing = !!playing;
    state.anim.lastTimestamp = 0;
  }
  // (Re)start the loop on play; on pause requestRender paints the
  // stopped frame and the loop self-terminates.
  kick();
};

// =====================================================================
// Sculpt panel bridges (2026-04-25). ADDITIVE EXPORTS — these read
// internal state.* fields but do not mutate any of model_viewer.js's
// own bookkeeping. The sculpt panel uses them to:
//   - Get a handle to the live mesh group + camera + THREE namespace
//     (psoGetSculptMeshGroup) so it can raycast the user's click.
//   - Tell the orbit-drag handler to ignore left-button events while
//     sculpt mode is active (psoSetSculptModeActive flips a flag the
//     pointerdown handler in `ensureRenderer` already checks).
//   - Trigger a normal-recompute on a single submesh after a brush
//     stroke (psoUpdateSculptedNormals — wraps geometry.computeVertexNormals).
//
// All three are surface-only: no new state fields, no render-loop hooks.
// =====================================================================
window.psoGetSculptMeshGroup = function () {
  if (!state.meshGroup || !state.camera) return null;
  // Stitch the model path used to load the current mesh — sculpt panel
  // uses it as the source_path for /api/sculpt/save.
  let modelPath = null;
  if (state.realMeshArchive) modelPath = state.realMeshArchive;
  if (!modelPath && state.anim && state.anim.modelPath) modelPath = state.anim.modelPath;
  if (!modelPath && state.filename) modelPath = state.filename;
  return {
    THREE,
    camera: state.camera,
    scene: state.scene,
    renderer: state.renderer,
    group: state.meshGroup,
    debugMeshes: state.debugMeshes || [],
    modelPath,
  };
};

window.psoSetSculptModeActive = function (active) {
  window.__psoSculptModeActive = !!active;
  // While sculpt is active we also pause auto-rotate so the mesh holds
  // still under the brush. The user can re-enable auto-rotate by
  // toggling sculpt off; this is the friendlier default.
  if (active) {
    state.__autoRotateSavedBeforeSculpt = !!state.autoRotate;
    state.autoRotate = false;
  } else if (state.__autoRotateSavedBeforeSculpt != null) {
    state.autoRotate = !!state.__autoRotateSavedBeforeSculpt;
  }
};

window.psoUpdateSculptedNormals = function (submeshIdx) {
  if (!state.meshGroup || !state.debugMeshes) return false;
  if (submeshIdx == null || submeshIdx < 0) {
    // Recompute all.
    for (const e of state.debugMeshes) {
      if (e && e.mesh && e.mesh.geometry) {
        e.mesh.geometry.computeVertexNormals();
        const n = e.mesh.geometry.getAttribute("normal");
        if (n) n.needsUpdate = true;
      }
    }
    return true;
  }
  const e = state.debugMeshes[submeshIdx];
  if (!e || !e.mesh || !e.mesh.geometry) return false;
  e.mesh.geometry.computeVertexNormals();
  const n = e.mesh.geometry.getAttribute("normal");
  if (n) n.needsUpdate = true;
  return true;
};

// psoUpdateVertexBuffer — additive bridge requested by the SCULPT_AGENT
// brief. Lets the sculpt panel push a partial vertex update to the
// renderer without touching the BufferAttribute directly. We expose it
// even though the sculpt panel currently writes the array in-place; the
// API is here for future tests + the documented "read-only API"
// boundary in the brief.
//
//   submeshIdx     — index into state.debugMeshes
//   vertexIndices  — Array | Uint32Array of vertex indices to update
//   newPositions   — Array | Float32Array, length 3*vertexIndices.length,
//                    [x0,y0,z0, x1,y1,z1, ...]
//
// Returns true on success.
window.psoUpdateVertexBuffer = function (submeshIdx, vertexIndices, newPositions) {
  if (!state.debugMeshes) return false;
  const e = state.debugMeshes[submeshIdx];
  if (!e || !e.mesh || !e.mesh.geometry) return false;
  const posAttr = e.mesh.geometry.getAttribute("position");
  if (!posAttr) return false;
  const arr = posAttr.array;
  for (let k = 0; k < vertexIndices.length; k++) {
    const vi = vertexIndices[k];
    if (vi < 0 || vi >= posAttr.count) continue;
    arr[vi * 3 + 0] = newPositions[k * 3 + 0];
    arr[vi * 3 + 1] = newPositions[k * 3 + 1];
    arr[vi * 3 + 2] = newPositions[k * 3 + 2];
  }
  posAttr.needsUpdate = true;
  return true;
};

// =====================================================================
// Paint MVP additive exports (2026-04-25).
//
// The paint panel needs camera + canvas + meshGroup + per-material
// THREE.Texture handles to do click-and-drag UV painting. None of these
// were exposed before; we surface them here as read-only getters and
// one writeable hook (psoSetMaterialTexture) so the panel can swap
// painted textures into the live mesh without touching disposeMesh /
// the boundTextures map directly.
//
// All exports are SIDE-EFFECT-FREE except for psoSetMaterialTexture
// (which mutates material.map and material.needsUpdate). The rendering
// loop and tickAnimation are untouched.
// =====================================================================
// Surface the THREE.js namespace on window so classic-script panels
// (paint_panel.js is a non-module) can instantiate THREE.CanvasTexture
// and THREE.Raycaster without re-importing the module. model_viewer.js
// is loaded as `<script type="module">`; without this hook the global
// `THREE` is unreachable from non-module scripts.
window.THREE = THREE;

window.psoGetCanvas = function () { return $("#modelCanvas"); };
window.psoGetCamera = function () { return state.camera || null; };
window.psoGetRenderer = function () { return state.renderer || null; };
window.psoGetMeshGroup = function () { return state.meshGroup || null; };

// One handle per material_id. Returns the live THREE.Texture (or null).
// Accepts either a bare integer (single-inner mode — keys are matIds) OR
// a string composite key "<innerIndex>:<matId>" (composite-mode keys set
// by tryLoadCompositeBmlMesh, 2026-04-30 fix).
window.psoGetMaterialTexture = function (materialId) {
  if (!state.boundTextures || !state.boundTextures.get) return null;
  const key = (typeof materialId === "string") ? materialId : (materialId | 0);
  return state.boundTextures.get(key) || null;
};

// Force a single render-pass — used after writing into a CanvasTexture
// when auto-rotate is off and the user is dragging the brush. Now an
// alias of requestRender(): it schedules exactly one on-demand frame
// (coalesced; no-op while the continuous loop is active). Existing
// callers (edit_panel/paint_panel/skeleton_panel/transform_gizmo) keep
// working — they schedule one frame instead of relying on an always-on
// loop.
window.psoForceRender = function () {
  requestRender();
};

// Stop the continuous loop AND cancel any pending one-shot. Exported so
// the scene/floor perspective unmounts can fully quiesce the shared
// renderer when the user leaves a 3D view (the 3d-view perspective
// already stops via close()).
window.psoViewerStopLoop = function () {
  stopLoop();
  if (state.renderPending) {
    cancelAnimationFrame(state.renderPending);
    state.renderPending = null;
  }
};

// Replace material.map for every Mesh whose userData.materialId matches
// `mid`. Used by the paint panel to bind a CanvasTexture (live-painted
// pixel buffer) in place of the static THREE.Texture loaded from PNG.
// The OLD texture is intentionally kept alive — caller may want to swap
// back via psoSetMaterialTexture(mid, originalTex). Caller owns dispose.
//
// Accepts either a bare integer (single-inner mode) OR a composite key
// string "<innerIndex>:<matId>" (composite mode, 2026-04-30 fix). When a
// composite key is passed, the walk filters by `userData.compositeKey`
// instead of `userData.materialId` so cross-inner meshes that happen to
// share the same per-inner matId are NOT cross-painted.
window.psoSetMaterialTexture = function (materialId, tex) {
  if (!state.meshGroup) return false;
  const isCompositeKey = (typeof materialId === "string");
  const mid = isCompositeKey ? materialId : (materialId | 0);
  state.meshGroup.traverse((c) => {
    if (!c.isMesh) return;
    if (isCompositeKey) {
      const ckey = c.userData && c.userData.compositeKey;
      if (ckey !== mid) return;
    } else {
      // In composite mode, every mesh has a compositeKey. Skip the bare-
      // matId walk to avoid cross-binding inner-A and inner-B's mat0.
      if (c.userData && c.userData.compositeKey) return;
      // fix/tooltabs — multi-material psov2 SkinnedMesh. ONE mesh carries
      // material[] + geometry.groups[]; userData.materialGroups maps each
      // slot's materialIndex -> material_id. Bind the CanvasTexture onto
      // EVERY slot whose material_id matches `mid` (a tile can span more
      // than one group). This is the case the bare-`c.material.map` branch
      // below could never reach (c.material is an Array here, so `.map`
      // would be undefined-on-array and silently no-op).
      const groups = c.userData && c.userData.materialGroups;
      if (Array.isArray(groups) && Array.isArray(c.material)) {
        let bound = false;
        for (const g of groups) {
          if ((g.materialId | 0) !== mid) continue;
          const slot = g.materialIndex | 0;
          const m = c.material[slot];
          if (m) {
            m.map = tex || null;
            m.needsUpdate = true;
            bound = true;
          }
        }
        if (bound) return;
        // fall through only if no group matched (e.g. single-material mesh
        // tagged with a bare materialId) — handled by the check below.
      }
      const cmid = c.userData && c.userData.materialId;
      if ((cmid | 0) !== mid) return;
    }
    if (c.material) {
      // Single-material case. (Array materials are handled in the
      // materialGroups branch above and returned before reaching here.)
      if (Array.isArray(c.material)) return;
      c.material.map = tex || null;
      c.material.needsUpdate = true;
    }
  });
  if (state.boundTextures && state.boundTextures.set) {
    if (tex) state.boundTextures.set(mid, tex);
  }
  return true;
};

// =====================================================================
// Map-Editor scene-mode helpers (2026-04-25).
//
// The Map Editor needs to render N terrain meshes as one navigable
// scene (not the single-NJ flow above). Rather than fork model_viewer.js
// into a separate map_viewer.js (~600 lines of three.js bootstrap
// duplication), we add scene-mode helpers that:
//
//   1. Load multiple NJ/XJ files in parallel via /api/model_mesh
//   2. Parent every loaded mesh into one root Group (state.sceneRoot)
//   3. Auto-fit the orbit camera to the group's combined AABB
//   4. Surface raycast(x, y) for spawn-placement clicks
//
// The single-model flow (state.mesh, state.meshGroup) is preserved
// untouched — psoSceneClearMap() removes only state.sceneRoot. Calling
// psoOpenModelByPath after a scene load will clobber state.mesh as
// usual, but the Map Editor never does that — it owns the renderer
// while its perspective is mounted.
//
// Wire format coming back from /api/model_mesh:
//   { meshes: [...], totals: {vertices, triangles}, ... }
// — same as buildMeshGroupFromPayload (which we re-use). Each map file
// becomes one Group of submeshes, those Groups all hang off
// state.sceneRoot, and the camera auto-fits to the combined AABB.
// =====================================================================

window.psoSceneClearMap = function () {
  if (state.sceneRoot && state.scene) {
    // Walk and dispose every Mesh in the scene.
    state.sceneRoot.traverse(function (c) {
      if (!c.isMesh) return;
      try { if (c.geometry) c.geometry.dispose(); } catch (_e) {}
      try {
        const mats = Array.isArray(c.material) ? c.material : [c.material];
        for (const m of mats) {
          if (!m) continue;
          if (m.map && m.map.__psoSceneOwned) {
            try { m.map.dispose(); } catch (_e) {}
          }
          try { m.dispose(); } catch (_e) {}
        }
      } catch (_e) {}
    });
    state.scene.remove(state.sceneRoot);
  }
  state.sceneRoot = null;
  // Markers + connectors live as separate root groups so a clear-map
  // also clears overlays.
  if (state.sceneMarkers && state.scene) {
    state.scene.remove(state.sceneMarkers);
    state.sceneMarkers.traverse(function (c) {
      if (!c.isMesh) return;
      try { if (c.geometry) c.geometry.dispose(); } catch (_e) {}
      try { if (c.material) c.material.dispose(); } catch (_e) {}
    });
  }
  state.sceneMarkers = null;
  if (state.sceneConnectors && state.scene) {
    state.scene.remove(state.sceneConnectors);
    state.sceneConnectors.traverse(function (c) {
      if (c.isLine || c.isMesh) {
        try { if (c.geometry) c.geometry.dispose(); } catch (_e) {}
        try { if (c.material) c.material.dispose(); } catch (_e) {}
      }
    });
  }
  state.sceneConnectors = null;
  if (state.sceneGrid && state.scene) {
    state.scene.remove(state.sceneGrid);
    try { state.sceneGrid.geometry.dispose(); } catch (_e) {}
    try { state.sceneGrid.material.dispose(); } catch (_e) {}
  }
  state.sceneGrid = null;
};

// Build one Group for one NJ/XJ payload (re-uses buildMeshGroupFromPayload
// but DOESN'T re-center the result — for scene mode every mesh must
// stay in absolute world coords so the assembled terrain doesn't pile
// up at the origin).
function _psoBuildSceneGroupFromPayload(payload, label) {
  const group = new THREE.Group();
  group.name = "scenePart:" + (label || "");
  const preTransformed = payload.vertices_pre_transformed !== false;
  for (const m of payload.meshes || []) {
    const vbuf = b64ToArrayBuffer(m.vertices_b64);
    const ibuf = b64ToArrayBuffer(m.indices_b64);
    const verts = new Float32Array(vbuf);
    const indices = new Uint32Array(ibuf);
    if (verts.length === 0 || indices.length === 0) continue;
    // v2 payloads carry 4 trailing RGBA floats (12-float stride).
    const hasColor = payload.has_color === true;
    const stride = hasColor ? 12 : 8;
    const vertexCount = verts.length / stride;
    if (!Number.isInteger(vertexCount)) continue;
    const positions = new Float32Array(vertexCount * 3);
    const normals = new Float32Array(vertexCount * 3);
    const uvs = new Float32Array(vertexCount * 2);
    const colors = hasColor ? new Float32Array(vertexCount * 4) : null;
    for (let i = 0; i < vertexCount; i++) {
      const o = i * stride;
      positions[i * 3 + 0] = verts[o + 0];
      positions[i * 3 + 1] = verts[o + 1];
      positions[i * 3 + 2] = verts[o + 2];
      normals[i * 3 + 0] = verts[o + 3];
      normals[i * 3 + 1] = verts[o + 4];
      normals[i * 3 + 2] = verts[o + 5];
      uvs[i * 2 + 0] = verts[o + 6];
      uvs[i * 2 + 1] = verts[o + 7];
      if (colors) {
        colors[i * 4 + 0] = verts[o + 8];
        colors[i * 4 + 1] = verts[o + 9];
        colors[i * 4 + 2] = verts[o + 10];
        colors[i * 4 + 3] = verts[o + 11];
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
    geo.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
    if (colors) geo.setAttribute("color", new THREE.BufferAttribute(colors, 4));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));
    geo.computeBoundingSphere();
    // Phase 3 (2026-06-20): unlit MeshBasicMaterial + vertexColors for
    // scene terrain too (psov2 parity). When the payload carries color
    // we use white base so the per-vertex diffuse shows through; without
    // color (legacy payload) we keep the neutral grey-blue fallback.
    // The psoSceneUseExactLambert toggle (below) is unaffected — it
    // walks meshes after creation and swaps in the bit-exact PSO shader.
    const mat = new THREE.MeshBasicMaterial({
      color: hasColor ? 0xffffff : 0xb8c2cf,
      vertexColors: hasColor,
      side: THREE.DoubleSide,
      transparent: false,
    });
    // Phase 3 (2026-06-20): apply per-submesh blend/alpha/two-sided flags.
    applyPsoMaterialFlags(mat, m);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.userData.materialId = (m.material_id | 0);
    mesh.userData.scenePart = label;
    if (!preTransformed && m.world_matrix && m.world_matrix.length === 16) {
      const wm = m.world_matrix;
      const t = new THREE.Matrix4();
      t.set(wm[0], wm[1], wm[2], wm[3],
            wm[4], wm[5], wm[6], wm[7],
            wm[8], wm[9], wm[10], wm[11],
            wm[12], wm[13], wm[14], wm[15]);
      mesh.matrixAutoUpdate = false;
      mesh.matrix.copy(t);
      t.decompose(mesh.position, mesh.quaternion, mesh.scale);
      mesh.matrixAutoUpdate = true;
    }
    group.add(mesh);
  }
  return group;
}

// Replace the model viewer's basic pointer-drag camera with a richer
// orbit camera the moment scene mode activates. We don't pull in
// OrbitControls because the editor never bundled it — instead we
// install a keyboard + middle-click pan handler on top of the existing
// drag/wheel handlers. Rest button restores the auto-fit pose.
function _psoSceneInstallNav() {
  const cv = window.psoGetCanvas && window.psoGetCanvas();
  if (!cv || cv.__psoSceneNavInstalled) return;
  cv.__psoSceneNavInstalled = true;
  // Pan with middle button or Shift+drag — translate camera + target
  // along the camera-local right/up axes proportional to drag distance.
  let panActive = false;
  let panX = 0, panY = 0;
  cv.addEventListener("pointerdown", (e) => {
    if (!state.sceneRoot) return;
    if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
      panActive = true;
      panX = e.clientX;
      panY = e.clientY;
      cv.setPointerCapture(e.pointerId);
      e.preventDefault();
    }
  });
  cv.addEventListener("pointermove", (e) => {
    if (!panActive || !state.camera) return;
    const dx = (e.clientX - panX) / 5;
    const dy = (e.clientY - panY) / 5;
    panX = e.clientX;
    panY = e.clientY;
    // Compute camera-local axes: right = camera.right, up = camera.up.
    const cam = state.camera;
    const right = new THREE.Vector3();
    const up = new THREE.Vector3();
    cam.matrixWorld.extractBasis(right, up, new THREE.Vector3());
    const moveScale = (state.sceneCamDist || 50) * 0.002;
    cam.position.addScaledVector(right, -dx * moveScale);
    cam.position.addScaledVector(up, dy * moveScale);
    if (state.sceneCamTarget) {
      state.sceneCamTarget.addScaledVector(right, -dx * moveScale);
      state.sceneCamTarget.addScaledVector(up, dy * moveScale);
      cam.lookAt(state.sceneCamTarget);
    }
    // Scene mode has no continuous animator — paint each pan move on
    // demand (coalesced) or the map appears frozen while dragging.
    requestRender();
  });
  const release = (e) => {
    if (panActive) {
      panActive = false;
      try { cv.releasePointerCapture(e.pointerId); } catch (_e) {}
    }
  };
  cv.addEventListener("pointerup", release);
  cv.addEventListener("pointercancel", release);
}

// Compute the bounding box of the scene's "core" geometry — i.e. the
// walkable terrain — by EXCLUDING giant skybox / distant-scenery shells.
//
// Why this exists: PSOBB map scenes carry a sky/backdrop shell (a low-poly
// cylinder ~±5000 units across, e.g. ``map_aancient01_00s.nj`` or the last
// mesh in an acity n.rel). A naive Box3.setFromObject(root) is dominated by
// that shell, so the camera auto-fits to an ~8000-unit box and the actual
// terrain becomes a sub-pixel speck — the "empty brown/tan plane" bug.
//
// Strategy: take every leaf mesh's own world-space AABB diagonal, find the
// median, and drop any mesh whose diagonal is a large multiple of it
// (SKY_OUTLIER_FACTOR). The survivors are the dense terrain cluster; we
// union their boxes. Falls back to the full box when filtering would
// remove everything (e.g. a scene that really is one big mesh).
function _psoSceneCoreBox() {
  const SKY_OUTLIER_FACTOR = 6;   // a mesh >6× the median size = skybox shell
  const entries = [];             // { box: Box3, diag: number }
  state.sceneRoot.traverse(function (m) {
    if (!m.isMesh || !m.geometry) return;
    m.updateWorldMatrix(true, false);
    const b = new THREE.Box3().setFromObject(m);
    if (!isFinite(b.min.x) || !isFinite(b.max.x)) return;
    const s = new THREE.Vector3();
    b.getSize(s);
    const diag = s.length();
    if (!isFinite(diag) || diag <= 0) return;
    entries.push({ box: b, diag: diag });
  });
  if (!entries.length) {
    const full = new THREE.Box3().setFromObject(state.sceneRoot);
    return isFinite(full.min.x) ? full : null;
  }
  // Median diagonal.
  const diags = entries.map(function (e) { return e.diag; }).sort(function (a, b) { return a - b; });
  const median = diags[(diags.length / 2) | 0] || diags[0];
  const cutoff = median * SKY_OUTLIER_FACTOR;
  const core = new THREE.Box3();
  core.makeEmpty();
  let kept = 0;
  for (const e of entries) {
    if (e.diag > cutoff) continue;   // skybox / scenery shell — skip framing
    core.union(e.box);
    kept++;
  }
  // If the filter nuked everything (one-mesh scene, or all meshes huge),
  // frame the full scene instead of an empty box.
  if (!kept || !isFinite(core.min.x)) {
    const full = new THREE.Box3().setFromObject(state.sceneRoot);
    return isFinite(full.min.x) ? full : null;
  }
  return core;
}

// Auto-fit the camera so the scene's core terrain AABB is in frame. Stores
// the chosen target on state.sceneCamTarget so panning + reset can
// re-anchor. Skybox / scenery shells are excluded from the framing box
// (see _psoSceneCoreBox) but still render — they just don't dictate zoom.
function _psoSceneAutoFit() {
  if (!state.sceneRoot || !state.camera) return;
  const box = _psoSceneCoreBox();
  if (!box || !isFinite(box.min.x)) return;
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  const maxDim = Math.max(size.x, size.y, size.z, 1.0);
  const fov = state.camera.fov * Math.PI / 180;
  const dist = (maxDim / 2) / Math.tan(fov / 2) * 1.6;
  state.sceneCamDist = dist;
  state.sceneCamTarget = center.clone();
  // Position camera at offset from center, looking down + sideways
  state.camera.position.set(
    center.x + dist * 0.7,
    center.y + dist * 0.7,
    center.z + dist * 0.7,
  );
  state.camera.near = Math.max(0.1, dist / 1000);
  state.camera.far = dist * 10;
  state.camera.updateProjectionMatrix();
  state.camera.lookAt(center);
}

// Flag + hide low-poly "sky shell / distant scenery" parts so they don't
// occlude the walkable terrain. ``loaded`` is the list of {path, group}
// records assembled in psoSceneLoadMap. A part is treated as a shell when
// its world bbox diagonal is huge but its vertex density (verts / diagonal)
// is far below the densest terrain part — a real floor mesh is always far
// denser. We only hide a shell when at least one denser terrain part exists,
// so a backdrop-only scene still renders. The scene-tree checkbox re-shows
// any hidden part on demand.
function _psoSceneHideSkyShells(loaded) {
  if (!loaded || loaded.length < 2) return;
  const THREE = window.THREE;
  const stats = [];
  for (const l of loaded) {
    if (!l.group) continue;
    l.group.updateWorldMatrix(true, false);
    const box = new THREE.Box3().setFromObject(l.group);
    if (!isFinite(box.min.x) || !isFinite(box.max.x)) continue;
    const size = new THREE.Vector3();
    box.getSize(size);
    const diag = size.length();
    let verts = 0;
    l.group.traverse(function (m) {
      if (m.isMesh && m.geometry) {
        const p = m.geometry.getAttribute("position");
        if (p) verts += p.count;
      }
    });
    if (diag <= 0 || verts <= 0) continue;
    // density = verts per 1000 world units of diagonal.
    stats.push({ group: l.group, diag: diag, verts: verts,
                 density: (verts / diag) * 1000 });
  }
  if (stats.length < 2) return;
  // The densest part is definitely terrain; use it as the reference.
  const maxDensity = stats.reduce(function (a, s) {
    return Math.max(a, s.density); }, 0);
  for (const s of stats) {
    // A shell: huge extent (>3000 units) AND <8% the density of the
    // densest terrain part. Forest s.nj: ~218 verts over ~14000-diag =
    // density ~15, vs terrain ~3800+. 15 / 3800 ≈ 0.4% — well under 8%.
    if (s.diag > 3000 && s.density < maxDensity * 0.08) {
      s.group.visible = false;
      s.group.userData.skyShellHidden = true;
    }
  }
}

// Public: load every renderable in `bundle.renderable` in parallel and
// build the scene group. `bundle` is the JSON returned by /api/map/<id>.
window.psoSceneLoadMap = async function (bundle) {
  if (!bundle || !Array.isArray(bundle.renderable)) {
    throw new Error("psoSceneLoadMap: bundle.renderable missing");
  }
  ensureRenderer();
  // Scene mode has NO continuous animator (sceneRoot is set below, so
  // shouldAnimateContinuously() stays false). We paint on demand instead
  // — a requestRender() is issued after the scene is built (below) and
  // on every nav event. Do NOT startLoop() here.
  // Drop any previously-loaded scene.
  window.psoSceneClearMap();
  // Hide the single-model preview group if one is active so the scene
  // doesn't share screen space with a stray dragon.
  if (state.mesh) state.mesh.visible = false;

  state.sceneRoot = new THREE.Group();
  state.sceneRoot.name = "psoSceneRoot:" + (bundle.map_id || "?");
  state.scene.add(state.sceneRoot);

  // Spawn marker + connector layers (always present so the picker can
  // drop markers immediately without re-creating them).
  state.sceneMarkers = new THREE.Group();
  state.sceneMarkers.name = "psoSceneMarkers";
  state.scene.add(state.sceneMarkers);
  state.sceneConnectors = new THREE.Group();
  state.sceneConnectors.name = "psoSceneConnectors";
  state.scene.add(state.sceneConnectors);

  // Load every renderable in parallel. Filter to "terrain" suffix only —
  // the c/n/r siblings are PSOBB relocation tables, not extra geometry.
  // ``rel_terrain`` is the new path for Pioneer 2 / city / lab maps that
  // don't ship raw .nj — the server extracts terrain from the n.rel and
  // surfaces it through /api/map/asset/<path>.rel directly.
  const targets = bundle.renderable.filter(function (r) {
    return r.kind === "terrain" || r.kind === "rel_terrain" || r.suffix === "s";
  });

  // De-dupe — both .nj and .xj for the same path-stem can ship side-by-side
  // and load the same surface. Prefer .nj when both exist.
  const byStem = new Map();
  for (const t of targets) {
    const stem = t.path.replace(/\.(nj|xj|rel)$/i, "");
    const ext = t.ext;
    const cur = byStem.get(stem);
    if (!cur || (ext === "nj" && cur.ext !== "nj")) {
      byStem.set(stem, t);
    }
  }
  const picked = Array.from(byStem.values());

  const loaded = [];
  const failures = [];
  await Promise.all(picked.map(async function (t) {
    try {
      // Scene files live at data/scene/<file>; the standard
      // /api/model_mesh resolver rejects path components, so for
      // scene/* paths we use the dedicated /api/map/asset endpoint.
      const isScene = t.path.startsWith("scene/");
      const url = isScene
        ? "/api/map/asset/" + t.path.split("/").map(encodeURIComponent).join("/")
        : "/api/model_mesh/" + t.path.split("/").map(encodeURIComponent).join("/");
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) throw new Error("HTTP " + r.status);
      const payload = await r.json();
      const grp = _psoBuildSceneGroupFromPayload(payload, t.path);
      grp.userData.path = t.path;
      grp.userData.ext = t.ext;
      grp.userData.suffix = t.suffix;
      state.sceneRoot.add(grp);
      loaded.push({ path: t.path, group: grp,
                    vert_total: (payload.vert_total || 0),
                    tri_total: (payload.tri_total || 0) });
    } catch (e) {
      failures.push({ path: t.path, error: String(e && e.message || e) });
    }
  }));

  // Hide "sky shell / distant scenery" parts by default so they don't
  // occlude the walkable terrain (the "only a big green/brown plane shows"
  // symptom). PSOBB ships a low-poly backdrop dome as a SEPARATE part
  // (e.g. ``map_aancient01_00s.nj`` — 218 verts spanning ±5200). psov2's
  // stage renderer doesn't draw it as terrain for these maps, so we load
  // it (the scene-tree checkbox can re-show it) but start it hidden. The
  // discriminator: a part whose world-space bbox diagonal is large AND
  // whose vertex density (verts per unit of diagonal) is tiny — a dense
  // terrain mesh never qualifies. We only ever hide when there's ALSO a
  // denser terrain part present, so a sky-only scene still shows something.
  try {
    _psoSceneHideSkyShells(loaded);
  } catch (_e) { /* non-fatal — worst case the shell just stays visible */ }

  _psoSceneInstallNav();
  _psoSceneAutoFit();
  // Paint the freshly-built scene on the next frame (on-demand; scene
  // mode idles afterward until a nav event requests another render).
  requestRender();

  return {
    map_id: bundle.map_id,
    floor: bundle.floor,
    loaded_count: loaded.length,
    failed_count: failures.length,
    failures: failures,
    loaded: loaded.map(function (l) {
      return { path: l.path, vert_total: l.vert_total, tri_total: l.tri_total };
    }),
  };
};

// Add a free-standing prop into the scene at world (x, y, z). Caller
// supplies a path the same way it'd call /api/model_mesh — most useful
// for dropping a player avatar / NPC stand-in. Returns the loaded
// THREE.Group (or null on failure).
window.psoSceneAddProp = async function (modelPath, position, rotation) {
  if (!state.sceneRoot) return null;
  try {
    const isScene = modelPath.startsWith("scene/");
    const url = isScene
      ? "/api/map/asset/" + modelPath.split("/").map(encodeURIComponent).join("/")
      : "/api/model_mesh/" + modelPath.split("/").map(encodeURIComponent).join("/");
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const payload = await r.json();
    const grp = _psoBuildSceneGroupFromPayload(payload, modelPath);
    grp.userData.isProp = true;
    if (Array.isArray(position) && position.length === 3) {
      grp.position.set(position[0], position[1], position[2]);
    }
    if (typeof rotation === "number") {
      grp.rotation.y = rotation;
    }
    state.sceneRoot.add(grp);
    requestRender();
    return grp;
  } catch (e) {
    console.warn("psoSceneAddProp failed:", e);
    return null;
  }
};

// Raycast from screen (canvas-relative) coords into the scene; return
// hit world position + which scene-part group was hit. Used by the
// spawn-placement click handler.
window.psoSceneRaycast = function (clientX, clientY) {
  if (!state.sceneRoot || !state.camera) return null;
  const cv = window.psoGetCanvas && window.psoGetCanvas();
  if (!cv) return null;
  const rect = cv.getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((clientX - rect.left) / rect.width) * 2 - 1,
    -((clientY - rect.top) / rect.height) * 2 + 1,
  );
  if (!state.sceneRaycaster) state.sceneRaycaster = new THREE.Raycaster();
  state.sceneRaycaster.setFromCamera(ndc, state.camera);
  const meshes = [];
  state.sceneRoot.traverse(function (c) { if (c.isMesh) meshes.push(c); });
  const hits = state.sceneRaycaster.intersectObjects(meshes, false);
  if (!hits.length) {
    // Fallback: intersect with a y=0 plane so the user can still drop
    // markers when missing the terrain (pioneer 2 has 0 renderables).
    const planeY = state.sceneCamTarget ? state.sceneCamTarget.y : 0;
    const ray = state.sceneRaycaster.ray;
    if (Math.abs(ray.direction.y) > 1e-6) {
      const t = (planeY - ray.origin.y) / ray.direction.y;
      if (t > 0 && t < state.camera.far) {
        const p = ray.at(t, new THREE.Vector3());
        return {
          world_pos: [p.x, p.y, p.z],
          hit_object: null,
          fallback: "y_plane",
        };
      }
    }
    return null;
  }
  const h = hits[0];
  const part = h.object && h.object.userData && h.object.userData.scenePart;
  return {
    world_pos: [h.point.x, h.point.y, h.point.z],
    hit_object: h.object || null,
    hit_path: part || null,
    distance: h.distance,
  };
};

// Drop a single marker (mesh) at world (x, y, z) and return its handle.
// Color encodes spawn type. The Map Editor calls this on every spawn
// add / drag-update; the markers live in state.sceneMarkers so a clear
// of the scene root doesn't kill them.
window.psoSceneAddMarker = function (id, worldPos, type) {
  if (!state.sceneMarkers || !Array.isArray(worldPos)) return null;
  // Type → color (matches the Map Editor sidebar legend)
  const palette = {
    mob:      0xff5555,
    npc:      0x55ff88,
    chest:    0xffd24a,
    switch:   0xa855f7,
    teleport: 0x00ddff,
  };
  const color = palette[type] || 0xffffff;
  const dist = state.sceneCamDist || 50;
  const radius = Math.max(0.5, dist * 0.012);
  const geo = new THREE.SphereGeometry(radius, 12, 8);
  const mat = new THREE.MeshBasicMaterial({ color, depthTest: false });
  const m = new THREE.Mesh(geo, mat);
  m.position.set(worldPos[0], worldPos[1], worldPos[2]);
  m.userData.markerId = id;
  m.userData.markerType = type;
  m.renderOrder = 1000;
  state.sceneMarkers.add(m);
  requestRender();
  return m;
};

window.psoSceneRemoveMarker = function (id) {
  if (!state.sceneMarkers) return false;
  let found = null;
  state.sceneMarkers.children.forEach(function (c) {
    if (c.userData && c.userData.markerId === id) found = c;
  });
  if (!found) return false;
  state.sceneMarkers.remove(found);
  try { found.geometry.dispose(); } catch (_e) {}
  try { found.material.dispose(); } catch (_e) {}
  requestRender();
  return true;
};

window.psoSceneMoveMarker = function (id, worldPos) {
  if (!state.sceneMarkers || !Array.isArray(worldPos)) return false;
  let found = null;
  state.sceneMarkers.children.forEach(function (c) {
    if (c.userData && c.userData.markerId === id) found = c;
  });
  if (!found) return false;
  found.position.set(worldPos[0], worldPos[1], worldPos[2]);
  requestRender();
  return true;
};

window.psoSceneClearMarkers = function () {
  if (!state.sceneMarkers) return;
  while (state.sceneMarkers.children.length) {
    const m = state.sceneMarkers.children[0];
    state.sceneMarkers.remove(m);
    try { m.geometry.dispose(); } catch (_e) {}
    try { m.material.dispose(); } catch (_e) {}
  }
  requestRender();
};

// Draw / update one connector between two world points. ``id`` is the
// caller's stable identity (e.g. ``${from_id}:${to_id}``); calling with
// the same id replaces the previous segment. ``style`` controls dash
// pattern: walk = solid, run = solid bold, teleport = dashed.
window.psoSceneSetConnector = function (id, fromPos, toPos, style) {
  if (!state.sceneConnectors) return null;
  // Drop existing
  let found = null;
  state.sceneConnectors.children.forEach(function (c) {
    if (c.userData && c.userData.connectorId === id) found = c;
  });
  if (found) {
    state.sceneConnectors.remove(found);
    try { found.geometry.dispose(); } catch (_e) {}
    try { found.material.dispose(); } catch (_e) {}
  }
  const pts = [
    new THREE.Vector3(fromPos[0], fromPos[1], fromPos[2]),
    new THREE.Vector3(toPos[0], toPos[1], toPos[2]),
  ];
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const isDashed = style === "teleport";
  const matCls = isDashed ? THREE.LineDashedMaterial : THREE.LineBasicMaterial;
  const mat = new matCls({
    color: style === "run" ? 0xffaa00 : (isDashed ? 0x00ddff : 0x55ff88),
    linewidth: 2,
    dashSize: 1, gapSize: 0.5,
    depthTest: false,
  });
  const line = new THREE.Line(geo, mat);
  if (isDashed) line.computeLineDistances();
  line.userData.connectorId = id;
  line.renderOrder = 999;
  state.sceneConnectors.add(line);
  requestRender();
  return line;
};

window.psoSceneRemoveConnector = function (id) {
  if (!state.sceneConnectors) return false;
  let found = null;
  state.sceneConnectors.children.forEach(function (c) {
    if (c.userData && c.userData.connectorId === id) found = c;
  });
  if (!found) return false;
  state.sceneConnectors.remove(found);
  try { found.geometry.dispose(); } catch (_e) {}
  try { found.material.dispose(); } catch (_e) {}
  requestRender();
  return true;
};

window.psoSceneClearConnectors = function () {
  if (!state.sceneConnectors) return;
  while (state.sceneConnectors.children.length) {
    const c = state.sceneConnectors.children[0];
    state.sceneConnectors.remove(c);
    try { c.geometry.dispose(); } catch (_e) {}
    try { c.material.dispose(); } catch (_e) {}
  }
  requestRender();
};

// Toggle a reference grid in the XZ plane at the scene's center. Helpful
// for orienting in big maps where most of the visible geometry is high
// off the floor (lab interiors, ruins ceilings).
window.psoSceneToggleGrid = function (visible) {
  if (!state.scene) return false;
  if (state.sceneGrid) {
    state.scene.remove(state.sceneGrid);
    try { state.sceneGrid.geometry.dispose(); } catch (_e) {}
    try { state.sceneGrid.material.dispose(); } catch (_e) {}
    state.sceneGrid = null;
  }
  if (visible === false) { requestRender(); return false; }
  // Auto-size to scene
  const dist = state.sceneCamDist || 50;
  const size = dist * 4;
  const div = 40;
  const grid = new THREE.GridHelper(size, div, 0x445566, 0x223344);
  if (state.sceneCamTarget) {
    grid.position.set(state.sceneCamTarget.x, state.sceneCamTarget.y, state.sceneCamTarget.z);
  }
  grid.material.transparent = true;
  grid.material.opacity = 0.4;
  state.scene.add(grid);
  state.sceneGrid = grid;
  requestRender();
  return true;
};

// Reset the orbit camera to the auto-fit pose (or a chosen perspective
// preset). Called by the toolbar's "Camera reset" / "topdown" buttons.
window.psoSceneResetCamera = function (mode) {
  if (!state.sceneRoot || !state.camera) return;
  if (mode === "topdown" && state.sceneCamTarget) {
    const t = state.sceneCamTarget;
    const d = state.sceneCamDist || 50;
    state.camera.position.set(t.x, t.y + d, t.z + 0.001);
    state.camera.lookAt(t);
    requestRender();
    return;
  }
  if (mode === "first-person" && state.sceneCamTarget) {
    const t = state.sceneCamTarget;
    const d = (state.sceneCamDist || 50) * 0.05;
    state.camera.position.set(t.x + d, t.y + d * 0.3, t.z + d);
    state.camera.lookAt(t);
    requestRender();
    return;
  }
  // Default = auto-fit
  _psoSceneAutoFit();
  requestRender();
};

// Peek into the scene state — the Map Editor's sidebar uses this to
// render its scene-tree (every loaded part with vert/tri counts).
window.psoSceneListLoaded = function () {
  if (!state.sceneRoot) return [];
  const out = [];
  for (const child of state.sceneRoot.children) {
    let verts = 0, tris = 0;
    child.traverse(function (m) {
      if (!m.isMesh || !m.geometry) return;
      const pos = m.geometry.getAttribute("position");
      if (pos) verts += pos.count;
      const idx = m.geometry.getIndex();
      if (idx) tris += idx.count / 3;
    });
    out.push({
      path:    child.userData.path || child.name || "",
      ext:     child.userData.ext || "",
      visible: child.visible,
      vertices:  verts,
      triangles: tris | 0,
      isProp: !!child.userData.isProp,
    });
  }
  return out;
};

// Toggle visibility of one scene part by path.
window.psoSceneSetPartVisible = function (path, visible) {
  if (!state.sceneRoot) return false;
  let found = null;
  for (const c of state.sceneRoot.children) {
    if (c.userData.path === path) { found = c; break; }
  }
  if (!found) return false;
  found.visible = !!visible;
  requestRender();
  return true;
};

// =====================================================================
// End of Map-Editor scene-mode helpers.
// =====================================================================

// =====================================================================
// Rig panel bridges (2026-04-25). ADDITIVE EXPORTS — these read internal
// state.* fields but do not mutate any of model_viewer.js's own
// bookkeeping. The rig panel uses them to:
//   - Get a handle to the live skinned mesh + camera + skeleton
//     (psoGetRigContext) so it can:
//       * raycast user clicks for weight-paint
//       * render bone widgets in the viewport
//       * read each submesh's bone_idx array for re-baking
//   - Push per-vertex bone-weight overrides (psoSetVertexWeights) onto
//     a sidecar Map<submeshIdx, {indices, weights}> the renderer
//     consults at re-bake time. When unset, the renderer keeps using
//     the original single-influence bone_idx[] payload (i.e. the
//     untouched bind path).
//   - Override a bone's animated pose at re-bake time
//     (psoSetBonePoseOverride) — used by IK to pin a chain after
//     solving.
//
// All exports are SIDE-EFFECT-FREE except for the *Set* helpers, which
// mutate two new state fields (state.rigVertexWeights,
// state.rigBoneOverrides) and call _bakeSkinnedSubmeshes once. Render
// loop / tickAnimation are untouched.
// =====================================================================

// Initialise sidecar storage; the rig panel can populate these from
// any tab. The renderer's existing _bakeSkinnedSubmeshes() does NOT
// consult these — instead, the rig panel installs its own re-bake on
// every change via psoApplyRigBake() below, which writes into the
// posAttr arrays directly.
state.rigVertexWeights = new Map();   // submeshIdx -> {indices: Int32Array[N*K], weights: Float32Array[N*K]}
state.rigBoneOverrides = new Map();   // boneIdx -> {position?: [x,y,z], rotation_bams?: [rx,ry,rz], scale?: [sx,sy,sz]}

window.psoGetRigContext = function () {
  if (!state.meshGroup || !state.camera) return null;
  let modelPath = null;
  if (state.realMeshArchive) modelPath = state.realMeshArchive;
  if (!modelPath && state.anim && state.anim.modelPath) modelPath = state.anim.modelPath;
  if (!modelPath && state.filename) modelPath = state.filename;
  // Skinned-only — rig panel needs bones + bone_idx arrays.
  if (!state.anim || !state.anim.skinned) return null;
  return {
    THREE,
    camera: state.camera,
    scene: state.scene,
    renderer: state.renderer,
    group: state.meshGroup,
    debugMeshes: state.debugMeshes || [],
    skinSubmeshes: state.anim.skinSubmeshes || [],
    bones: state.anim.bones || [],
    modelPath,
    rigVertexWeights: state.rigVertexWeights,
    rigBoneOverrides: state.rigBoneOverrides,
  };
};

// Snapshot the bone-pose data the rig panel needs (read-only).
window.psoGetSkeleton = function () {
  if (!state.anim || !state.anim.bones) return null;
  return state.anim.bones.map((b) => ({
    index: b.index,
    parent: b.parent,
    position: [b.position[0], b.position[1], b.position[2]],
    rotation_bams: [b.rotation_bams[0], b.rotation_bams[1], b.rotation_bams[2]],
    scale: b.scale ? [b.scale[0], b.scale[1], b.scale[2]] : [1, 1, 1],
    eval_flags: (b.eval_flags | 0),
  }));
};

// Override a single bone's pose. The rig panel writes the new TRS;
// the next re-bake uses the override values instead of the bind/
// animated pose.
window.psoSetBonePoseOverride = function (boneIdx, pose) {
  if (boneIdx == null) return false;
  const idx = boneIdx | 0;
  if (idx < 0) return false;
  if (!state.rigBoneOverrides) state.rigBoneOverrides = new Map();
  if (!pose) {
    state.rigBoneOverrides.delete(idx);
  } else {
    const cur = state.rigBoneOverrides.get(idx) || {};
    if (pose.position && pose.position.length === 3) cur.position = pose.position.slice();
    if (pose.rotation_bams && pose.rotation_bams.length === 3) {
      cur.rotation_bams = [pose.rotation_bams[0] | 0, pose.rotation_bams[1] | 0, pose.rotation_bams[2] | 0];
    }
    if (pose.scale && pose.scale.length === 3) cur.scale = pose.scale.slice();
    state.rigBoneOverrides.set(idx, cur);
  }
  return true;
};

// Clear ALL bone-pose overrides (used by rig "Reset"). Doesn't touch
// vertex weights — those are a separate Map.
window.psoClearBonePoseOverrides = function () {
  if (state.rigBoneOverrides) state.rigBoneOverrides.clear();
};

// Read the live position attribute for a submesh (used by the rig
// panel's raycast result → bone-projection math).
window.psoGetSubmeshLocalPositions = function (submeshIdx) {
  const sub = (state.anim && state.anim.skinSubmeshes) ? state.anim.skinSubmeshes[submeshIdx] : null;
  if (!sub) return null;
  return sub.bonePositions;  // read-only Float32Array (bone-LOCAL)
};

// Read the per-vertex bone_idx array for a submesh (single-influence
// bind data from /api/model_skinned).
window.psoGetSubmeshBoneIndices = function (submeshIdx) {
  const sub = (state.anim && state.anim.skinSubmeshes) ? state.anim.skinSubmeshes[submeshIdx] : null;
  if (!sub) return null;
  return sub.vertBoneIdx;
};

// Set the rig panel's per-vertex weight override for one submesh. The
// rig panel uses this from auto-skin or after a weight-paint stroke;
// the overrides are read by the rig panel's own re-bake path
// (psoApplyRigBake below), NOT by the renderer's tickAnimation.
//
// indices  — Int32Array length N * MAX_INFLUENCES (4)
// weights  — Float32Array same length
// vertexCount must match sub.bonePositions.length / 3 — defensive.
window.psoSetVertexWeights = function (submeshIdx, indices, weights) {
  if (submeshIdx == null) return false;
  const idx = submeshIdx | 0;
  if (!state.rigVertexWeights) state.rigVertexWeights = new Map();
  if (!indices || !weights) {
    state.rigVertexWeights.delete(idx);
    return true;
  }
  state.rigVertexWeights.set(idx, {
    indices: indices instanceof Int32Array ? indices : new Int32Array(indices),
    weights: weights instanceof Float32Array ? weights : new Float32Array(weights),
  });
  return true;
};

// Re-bake the skinned mesh AND honor rig overrides (bone poses +
// per-vertex weights). The rig panel calls this after every change so
// the viewport reflects the edit immediately.
//
// When state.rigVertexWeights[s] is set, vertex bone influences come
// from there (4-bone weighted blend). When unset, the original 1-bone
// path applies (matrix-of-vertBoneIdx[v]).
//
// When state.rigBoneOverrides has entries, the bone matrices for those
// bones use the override TRS instead of the bind/animated pose.
window.psoApplyRigBake = function () {
  const a = state.anim;
  if (!a || !a.skinned || !a.bones || a.bones.length === 0) return false;
  const subs = a.skinSubmeshes || [];
  if (subs.length === 0) return false;

  // 1. Compute per-bone WORLD matrices honoring rig overrides.
  const n = a.bones.length;
  const localBuf = new Float32Array(n * 16);
  const worldBuf = new Float32Array(n * 16);
  const overrides = state.rigBoneOverrides || new Map();

  for (let bi = 0; bi < n; bi++) {
    const b = a.bones[bi];
    const ef = (b.eval_flags | 0);
    let pos = b.position;
    let rot = b.rotation_bams;
    let scl = b.scale || [1.0, 1.0, 1.0];
    const ov = overrides.get(bi);
    if (ov) {
      if (ov.position) pos = ov.position;
      if (ov.rotation_bams) rot = ov.rotation_bams;
      if (ov.scale) scl = ov.scale;
    }
    const tx = (ef & _EVAL_UNIT_POS) ? 0 : pos[0];
    const ty = (ef & _EVAL_UNIT_POS) ? 0 : pos[1];
    const tz = (ef & _EVAL_UNIT_POS) ? 0 : pos[2];
    const rxB = (ef & _EVAL_UNIT_ANG) ? 0 : rot[0];
    const ryB = (ef & _EVAL_UNIT_ANG) ? 0 : rot[1];
    const rzB = (ef & _EVAL_UNIT_ANG) ? 0 : rot[2];
    const sx = (ef & _EVAL_UNIT_SCL) ? 1 : scl[0];
    const sy = (ef & _EVAL_UNIT_SCL) ? 1 : scl[1];
    const sz = (ef & _EVAL_UNIT_SCL) ? 1 : scl[2];
    const localOff = bi * 16;
    if (ef & _EVAL_SKIP) {
      localBuf[localOff] = 1;     localBuf[localOff + 1] = 0; localBuf[localOff + 2] = 0; localBuf[localOff + 3] = 0;
      localBuf[localOff + 4] = 0; localBuf[localOff + 5] = 1; localBuf[localOff + 6] = 0; localBuf[localOff + 7] = 0;
      localBuf[localOff + 8] = 0; localBuf[localOff + 9] = 0; localBuf[localOff + 10] = 1; localBuf[localOff + 11] = 0;
      localBuf[localOff + 12] = 0; localBuf[localOff + 13] = 0; localBuf[localOff + 14] = 0; localBuf[localOff + 15] = 1;
    } else {
      _composeTrsM4(
        localBuf.subarray(localOff, localOff + 16),
        tx, ty, tz,
        rxB * _BAMS_TO_RAD, ryB * _BAMS_TO_RAD, rzB * _BAMS_TO_RAD,
        sx, sy, sz,
        !!(ef & _EVAL_ZXY_ANG),
      );
    }
    if (b.parent < 0) {
      worldBuf.set(localBuf.subarray(localOff, localOff + 16), bi * 16);
    } else {
      const parentOff = b.parent * 16;
      _mulM4(
        worldBuf.subarray(bi * 16, bi * 16 + 16),
        worldBuf.subarray(parentOff, parentOff + 16),
        localBuf.subarray(localOff, localOff + 16),
      );
    }
  }

  // 2. Re-bake every submesh's vertices.
  const overrideWeights = state.rigVertexWeights || new Map();
  for (let s = 0; s < subs.length; s++) {
    const sub = subs[s];
    const localPos = sub.bonePositions;
    const localNorm = sub.boneNormals;
    const boneIdx = sub.vertBoneIdx;
    const posAttr = sub.geometry.attributes.position;
    const normAttr = sub.geometry.attributes.normal;
    const outPos = posAttr.array;
    const outNorm = normAttr ? normAttr.array : null;
    const vc = (localPos.length / 3) | 0;
    const ov = overrideWeights.get(s);
    if (ov && ov.indices && ov.weights && ov.indices.length === vc * 4) {
      // Multi-influence weighted blend.
      const ind = ov.indices;
      const ww = ov.weights;
      for (let vi = 0; vi < vc; vi++) {
        const px = localPos[vi * 3 + 0];
        const py = localPos[vi * 3 + 1];
        const pz = localPos[vi * 3 + 2];
        let nx = 0, ny = 0, nz = 0;
        if (outNorm) {
          nx = localNorm[vi * 3 + 0];
          ny = localNorm[vi * 3 + 1];
          nz = localNorm[vi * 3 + 2];
        }
        let ox = 0, oy = 0, oz = 0;
        let onx = 0, ony = 0, onz = 0;
        let totalW = 0;
        for (let k = 0; k < 4; k++) {
          const bi = ind[vi * 4 + k];
          const w = ww[vi * 4 + k];
          if (bi < 0 || bi >= n || w <= 0) continue;
          totalW += w;
          const mOff = bi * 16;
          // Position
          ox += w * (worldBuf[mOff + 0] * px + worldBuf[mOff + 1] * py + worldBuf[mOff + 2] * pz + worldBuf[mOff + 3]);
          oy += w * (worldBuf[mOff + 4] * px + worldBuf[mOff + 5] * py + worldBuf[mOff + 6] * pz + worldBuf[mOff + 7]);
          oz += w * (worldBuf[mOff + 8] * px + worldBuf[mOff + 9] * py + worldBuf[mOff + 10] * pz + worldBuf[mOff + 11]);
          if (outNorm) {
            onx += w * (worldBuf[mOff + 0] * nx + worldBuf[mOff + 1] * ny + worldBuf[mOff + 2] * nz);
            ony += w * (worldBuf[mOff + 4] * nx + worldBuf[mOff + 5] * ny + worldBuf[mOff + 6] * nz);
            onz += w * (worldBuf[mOff + 8] * nx + worldBuf[mOff + 9] * ny + worldBuf[mOff + 10] * nz);
          }
        }
        if (totalW <= 0) {
          // No valid influence — fall back to bone_idx.
          let bi = boneIdx[vi];
          if (bi < 0 || bi >= n) bi = 0;
          const mOff = bi * 16;
          outPos[vi * 3 + 0] = worldBuf[mOff + 0] * px + worldBuf[mOff + 1] * py + worldBuf[mOff + 2] * pz + worldBuf[mOff + 3];
          outPos[vi * 3 + 1] = worldBuf[mOff + 4] * px + worldBuf[mOff + 5] * py + worldBuf[mOff + 6] * pz + worldBuf[mOff + 7];
          outPos[vi * 3 + 2] = worldBuf[mOff + 8] * px + worldBuf[mOff + 9] * py + worldBuf[mOff + 10] * pz + worldBuf[mOff + 11];
          if (outNorm) {
            outNorm[vi * 3 + 0] = worldBuf[mOff + 0] * nx + worldBuf[mOff + 1] * ny + worldBuf[mOff + 2] * nz;
            outNorm[vi * 3 + 1] = worldBuf[mOff + 4] * nx + worldBuf[mOff + 5] * ny + worldBuf[mOff + 6] * nz;
            outNorm[vi * 3 + 2] = worldBuf[mOff + 8] * nx + worldBuf[mOff + 9] * ny + worldBuf[mOff + 10] * nz;
          }
        } else {
          // Renormalize in case totalW != 1.
          const inv = 1.0 / totalW;
          outPos[vi * 3 + 0] = ox * inv;
          outPos[vi * 3 + 1] = oy * inv;
          outPos[vi * 3 + 2] = oz * inv;
          if (outNorm) {
            outNorm[vi * 3 + 0] = onx * inv;
            outNorm[vi * 3 + 1] = ony * inv;
            outNorm[vi * 3 + 2] = onz * inv;
          }
        }
      }
    } else {
      // Single-influence path (matches the renderer's default).
      for (let vi = 0; vi < vc; vi++) {
        let bi = boneIdx[vi];
        if (bi < 0 || bi >= n) bi = 0;
        const mOff = bi * 16;
        const px = localPos[vi * 3 + 0];
        const py = localPos[vi * 3 + 1];
        const pz = localPos[vi * 3 + 2];
        outPos[vi * 3 + 0] = worldBuf[mOff + 0] * px + worldBuf[mOff + 1] * py + worldBuf[mOff + 2] * pz + worldBuf[mOff + 3];
        outPos[vi * 3 + 1] = worldBuf[mOff + 4] * px + worldBuf[mOff + 5] * py + worldBuf[mOff + 6] * pz + worldBuf[mOff + 7];
        outPos[vi * 3 + 2] = worldBuf[mOff + 8] * px + worldBuf[mOff + 9] * py + worldBuf[mOff + 10] * pz + worldBuf[mOff + 11];
        if (outNorm) {
          const nx = localNorm[vi * 3 + 0];
          const ny = localNorm[vi * 3 + 1];
          const nz = localNorm[vi * 3 + 2];
          outNorm[vi * 3 + 0] = worldBuf[mOff + 0] * nx + worldBuf[mOff + 1] * ny + worldBuf[mOff + 2] * nz;
          outNorm[vi * 3 + 1] = worldBuf[mOff + 4] * nx + worldBuf[mOff + 5] * ny + worldBuf[mOff + 6] * nz;
          outNorm[vi * 3 + 2] = worldBuf[mOff + 8] * nx + worldBuf[mOff + 9] * ny + worldBuf[mOff + 10] * nz;
        }
      }
    }
    posAttr.needsUpdate = true;
    if (normAttr) normAttr.needsUpdate = true;
  }
  return true;
};

// Get per-bone WORLD positions (translation column of each bone's
// world matrix), honoring overrides. Used by the rig panel to render
// bone widgets at the right spot.
window.psoGetBoneWorldPositions = function () {
  const a = state.anim;
  if (!a || !a.skinned || !a.bones || a.bones.length === 0) return null;
  const n = a.bones.length;
  const localBuf = new Float32Array(n * 16);
  const worldBuf = new Float32Array(n * 16);
  const overrides = state.rigBoneOverrides || new Map();
  for (let bi = 0; bi < n; bi++) {
    const b = a.bones[bi];
    const ef = (b.eval_flags | 0);
    let pos = b.position;
    let rot = b.rotation_bams;
    let scl = b.scale || [1.0, 1.0, 1.0];
    const ov = overrides.get(bi);
    if (ov) {
      if (ov.position) pos = ov.position;
      if (ov.rotation_bams) rot = ov.rotation_bams;
      if (ov.scale) scl = ov.scale;
    }
    const tx = (ef & _EVAL_UNIT_POS) ? 0 : pos[0];
    const ty = (ef & _EVAL_UNIT_POS) ? 0 : pos[1];
    const tz = (ef & _EVAL_UNIT_POS) ? 0 : pos[2];
    const rxB = (ef & _EVAL_UNIT_ANG) ? 0 : rot[0];
    const ryB = (ef & _EVAL_UNIT_ANG) ? 0 : rot[1];
    const rzB = (ef & _EVAL_UNIT_ANG) ? 0 : rot[2];
    const sx = (ef & _EVAL_UNIT_SCL) ? 1 : scl[0];
    const sy = (ef & _EVAL_UNIT_SCL) ? 1 : scl[1];
    const sz = (ef & _EVAL_UNIT_SCL) ? 1 : scl[2];
    const localOff = bi * 16;
    if (ef & _EVAL_SKIP) {
      localBuf[localOff] = 1;     localBuf[localOff + 5] = 1; localBuf[localOff + 10] = 1; localBuf[localOff + 15] = 1;
    } else {
      _composeTrsM4(
        localBuf.subarray(localOff, localOff + 16),
        tx, ty, tz,
        rxB * _BAMS_TO_RAD, ryB * _BAMS_TO_RAD, rzB * _BAMS_TO_RAD,
        sx, sy, sz,
        !!(ef & _EVAL_ZXY_ANG),
      );
    }
    if (b.parent < 0) {
      worldBuf.set(localBuf.subarray(localOff, localOff + 16), bi * 16);
    } else {
      const parentOff = b.parent * 16;
      _mulM4(
        worldBuf.subarray(bi * 16, bi * 16 + 16),
        worldBuf.subarray(parentOff, parentOff + 16),
        localBuf.subarray(localOff, localOff + 16),
      );
    }
  }
  // Extract translation column for each bone.
  const out = new Array(n);
  for (let bi = 0; bi < n; bi++) {
    const o = bi * 16;
    out[bi] = [worldBuf[o + 3], worldBuf[o + 7], worldBuf[o + 11]];
  }
  return out;
};

// Convert a WORLD-space point (e.g. from psoGetBoneWorldPositions) to
// the meshGroup's WORLD coords by applying group.matrixWorld. Used by
// the rig panel to render bone widgets at the right viewport pixel.
window.psoBoneSpaceToWorld = function (pt) {
  if (!state.meshGroup || !pt) return pt;
  const g = state.meshGroup;
  // group.matrixWorld is a THREE.Matrix4 (column-major); transform pt.
  if (!g.matrixWorld) return pt;
  const v = new THREE.Vector3(pt[0], pt[1], pt[2]);
  v.applyMatrix4(g.matrixWorld);
  return [v.x, v.y, v.z];
};

// Disable the orbit drag while rig mode owns the LMB (mirror of
// psoSetSculptModeActive).
window.psoSetRigModeActive = function (active) {
  window.__psoRigModeActive = !!active;
  if (active) {
    state.__autoRotateSavedBeforeRig = !!state.autoRotate;
    state.autoRotate = false;
  } else if (state.__autoRotateSavedBeforeRig != null) {
    state.autoRotate = !!state.__autoRotateSavedBeforeRig;
  }
};


// =====================================================================
// Map Editor — fog + per-area lighting parity (2026-04-25, additive)
// =====================================================================
//
// PSOBB's actual scene shader runs per-vertex Lambert with map-specific
// fog colors keyed off the *_NN r.rel relocation tables. The full
// shader port is parked for v3 (we'd need to replicate the per-area
// lambert variant + texture-id-driven material switching). For v2 we
// upgrade the basics:
//
//   1. Switch terrain materials from MeshStandardMaterial (PBR) to
//      MeshLambertMaterial (closer to PSOBB's per-vertex Lambert).
//      MeshStandardMaterial was washing out the scene because PSOBB
//      doesn't ship metallic / roughness data.
//
//   2. Apply scene.fog (THREE.Fog or FogExp2) per area. The categories
//      mirror the AREA_CATEGORY map in formats/scene_loader.py:
//        forest 1/2:        light green-blue
//        cave 1/2/3:        dark
//        mine 1/2:          cool blue
//        ruins 1/2/3:       dim sandstone
//        battle:            mid-grey
//        boss:              category-specific (default mid-grey)
//        city:              warm yellow Pioneer 2
//        corruption:        green/red horror
//        other:             scene default (no fog)
//
//   3. Tweak the existing HemisphereLight + DirectionalLight per area
//      so forests look warm + canopy-shaded, caves look cold and
//      uplit, etc.
//
// All three knobs are config-table-driven so future tuning is a
// one-line edit. Calling psoSceneApplyEnvironment(category) re-applies
// the matching environment to the current scene.

const _PSO_AREA_ENV = {
  forest: {
    fog: { color: 0x6bb37a, near: 50, far: 1800 },
    hemi:    { sky: 0xc7e2ff, ground: 0x223811, intensity: 0.55 },
    key:     { color: 0xfff0c8, intensity: 0.9, dir: [0.6, 1.0, 0.4] },
    bg:      0x6bb37a,
  },
  cave: {
    fog: { color: 0x1a1a26, near: 30, far: 800 },
    hemi:    { sky: 0x223344, ground: 0x080812, intensity: 0.45 },
    key:     { color: 0xa8c8ff, intensity: 0.55, dir: [0.2, 1.0, 0.3] },
    bg:      0x1a1a26,
  },
  mine: {
    fog: { color: 0x445566, near: 40, far: 1100 },
    hemi:    { sky: 0x6688aa, ground: 0x222229, intensity: 0.5 },
    key:     { color: 0xc8d8ff, intensity: 0.7, dir: [0.4, 1.0, 0.4] },
    bg:      0x445566,
  },
  ruins: {
    fog: { color: 0xb39772, near: 40, far: 1400 },
    hemi:    { sky: 0xd8c8a4, ground: 0x553a22, intensity: 0.45 },
    key:     { color: 0xffe0a8, intensity: 0.75, dir: [0.5, 1.0, 0.4] },
    bg:      0xb39772,
  },
  battle: {
    fog: { color: 0x999999, near: 50, far: 1600 },
    hemi:    { sky: 0xcccccc, ground: 0x444444, intensity: 0.5 },
    key:     { color: 0xffffff, intensity: 0.8, dir: [0.5, 1.0, 0.5] },
    bg:      0x999999,
  },
  city: {
    fog: { color: 0xb09060, near: 80, far: 2400 },
    hemi:    { sky: 0xffe6c8, ground: 0x554433, intensity: 0.6 },
    key:     { color: 0xfff0d8, intensity: 0.85, dir: [0.4, 1.0, 0.6] },
    bg:      0xb09060,
  },
  corruption: {
    fog: { color: 0x506060, near: 40, far: 1200 },
    hemi:    { sky: 0x506060, ground: 0x202020, intensity: 0.5 },
    key:     { color: 0xa8b8a8, intensity: 0.7, dir: [0.4, 1.0, 0.4] },
    bg:      0x506060,
  },
  boss: {
    fog: { color: 0x808080, near: 50, far: 1500 },
    hemi:    { sky: 0x999999, ground: 0x222222, intensity: 0.5 },
    key:     { color: 0xffffff, intensity: 0.85, dir: [0.5, 1.0, 0.5] },
    bg:      0x808080,
  },
  other: {
    fog: null,  // no fog by default
    hemi:    { sky: 0xffffff, ground: 0x222233, intensity: 0.6 },
    key:     { color: 0xffffff, intensity: 0.85, dir: [2.0, 3.0, 4.0] },
    bg:      0x0a0e13,
  },
};

// Track the currently-applied environment so the unmount path can
// restore the model viewer's default lighting.
state.envApplied = null;
state.envSavedLights = null;

// Public: install a fog + per-area light tweak for the given category.
// Called by map_panel.js right after psoSceneLoadMap finishes. Safe to
// call repeatedly — replaces the previous env in-place.
window.psoSceneApplyEnvironment = function (category) {
  if (!state.scene) return false;
  const env = _PSO_AREA_ENV[category] || _PSO_AREA_ENV.other;

  // Remove any existing fog (THREE.Fog or FogExp2)
  state.scene.fog = null;
  if (env.fog) {
    // The per-area fog far (e.g. forest 1800) is tuned for PSOBB's
    // in-game first-person view, where the player sees ~1800 units ahead.
    // The scene EDITOR auto-fits the camera to an overview of the WHOLE
    // floor — which for a big map (forest core ~4600 units) sits well
    // beyond the fog far, so the entire terrain fogs out to a flat green
    // wash and looks empty / unreadable. For the overview we anchor the
    // fog band to the auto-fit camera distance instead: terrain stays in
    // clear air out to ~camDist, and only the far edge of the floor fades
    // into the area's fog colour (mood preserved, geometry readable).
    // ``state.sceneCamDist`` is set by the auto-fit that ran inside
    // psoSceneLoadMap immediately before this call.
    let near = env.fog.near;
    let far = env.fog.far;
    const camDist = state.sceneCamDist || 0;
    if (camDist > 0) {
      // Only widen — never make the in-game fog tighter than authored.
      near = Math.max(near, camDist * 0.6);
      far = Math.max(far, camDist * 2.4);
    }
    state.scene.fog = new THREE.Fog(env.fog.color, near, far);
  }

  // Background tint — picks up where fog leaves off.
  if (env.bg != null) {
    state.scene.background = new THREE.Color(env.bg);
  }

  // Locate the existing hemi + key lights (added in ensureRenderer).
  // We don't add new lights — we re-tune the originals so the model
  // viewer's single-model preview behaves the same once the map editor
  // unmounts and the user goes back to a monster.
  const lights = [];
  state.scene.traverse(function (obj) {
    if (obj && obj.isHemisphereLight) lights.push(obj);
    else if (obj && obj.isDirectionalLight) lights.push(obj);
  });
  // Save originals on first call so unmount can restore.
  if (state.envSavedLights == null) {
    state.envSavedLights = lights.map(function (l) {
      const out = {
        type: l.isHemisphereLight ? "hemi" : "dir",
        intensity: l.intensity,
      };
      if (l.color) out.color = l.color.clone();
      if (l.groundColor) out.ground = l.groundColor.clone();
      if (l.position) out.position = l.position.clone();
      return out;
    });
  }
  for (const l of lights) {
    if (l.isHemisphereLight && env.hemi) {
      l.color.setHex(env.hemi.sky);
      if (l.groundColor) l.groundColor.setHex(env.hemi.ground);
      l.intensity = env.hemi.intensity;
    } else if (l.isDirectionalLight && env.key) {
      l.color.setHex(env.key.color);
      l.intensity = env.key.intensity;
      if (l.position && Array.isArray(env.key.dir)) {
        l.position.set(env.key.dir[0], env.key.dir[1], env.key.dir[2]);
      }
    }
  }

  state.envApplied = category;
  requestRender();
  return true;
};

// Reset the model viewer's environment to the default (no fog,
// off-white hemi, white directional from camera-ish). Used by the Map
// Editor's unmount path so single-model preview stays unaffected.
window.psoSceneResetEnvironment = function () {
  if (!state.scene) return false;
  state.scene.fog = null;
  state.scene.background = new THREE.Color(0x0a0e13);
  if (state.envSavedLights) {
    let i = 0;
    state.scene.traverse(function (obj) {
      if (!obj || !state.envSavedLights || i >= state.envSavedLights.length) return;
      const saved = state.envSavedLights[i];
      if (obj.isHemisphereLight && saved.type === "hemi") {
        if (saved.color) obj.color.copy(saved.color);
        if (saved.ground && obj.groundColor) obj.groundColor.copy(saved.ground);
        obj.intensity = saved.intensity;
        i++;
      } else if (obj.isDirectionalLight && saved.type === "dir") {
        if (saved.color) obj.color.copy(saved.color);
        if (saved.position && obj.position) obj.position.copy(saved.position);
        obj.intensity = saved.intensity;
        i++;
      }
    });
  }
  state.envApplied = null;
  return true;
};

// Switch the materials in state.sceneRoot from MeshStandardMaterial
// (PBR — washed out without metalness/roughness data) to
// MeshLambertMaterial (closer to PSOBB's per-vertex Lambert). Keep the
// color the same and preserve the side / transparent flags. Called
// once per scene load by map_panel.js.
window.psoSceneUseLambertMaterials = function () {
  if (!state.sceneRoot) return false;
  let switched = 0;
  state.sceneRoot.traverse(function (m) {
    if (!m.isMesh || !m.material) return;
    const old = Array.isArray(m.material) ? m.material[0] : m.material;
    // Already lambert — skip.
    if (old.type === "MeshLambertMaterial") return;
    const next = new THREE.MeshLambertMaterial({
      color: old.color ? old.color.clone() : new THREE.Color(0xb8c2cf),
      side: old.side != null ? old.side : THREE.DoubleSide,
      transparent: !!old.transparent,
      opacity: old.opacity != null ? old.opacity : 1.0,
      map: old.map || null,
    });
    // Make sure fog is honored — Lambert materials respect scene.fog
    // when fog property is true (default).
    next.fog = true;
    if (old.dispose) try { old.dispose(); } catch (_e) {}
    m.material = next;
    switched++;
  });
  if (switched) requestRender();
  return switched;
};

// Convenience wrapper: load + relight in one call. Map Editor uses
// this so panel code doesn't have to chain three calls.
window.psoSceneLoadMapWithEnvironment = async function (bundle) {
  const result = await window.psoSceneLoadMap(bundle);
  try { window.psoSceneUseLambertMaterials(); } catch (_e) {}
  try { window.psoSceneApplyEnvironment(bundle.category); } catch (_e) {}
  return result;
};

// =====================================================================
// Anim-Editor scrubber → 3D pose live sync (2026-04-25, Editor v3).
//
// The Anim Editor panel needs to drive the model viewer's pose to a
// specific motion frame as the user drags the timeline scrubber. The
// existing render loop owns playback (state.anim.time / .playing); this
// shim writes the time field, pauses playback, and forces one tick so
// the bone-matrix re-bake happens before the next paint.
//
// Returns the actually-rendered frame index (clamped to motion bounds)
// so the caller can reflect any clamping back into its UI. Returns -1
// when there is no skinned model or no motion loaded.
// =====================================================================
window.psoSeekAnimationToFrame = function (frameIdx) {
  const a = state.anim;
  if (!a || !a.skinned || !a.currentData) return -1;
  const fc = a.currentData.frame_count | 0;
  if (fc <= 0) return -1;
  const fps = a.fps > 0 ? a.fps : 30.0;
  // Clamp to [0, fc-1] before computing time. Sub-frame accuracy isn't
  // useful (the parser's keyframe rounding already operates at integer
  // frames), but we accept floats so callers can drive a continuous
  // scrubber if they want.
  let f = +frameIdx;
  if (!Number.isFinite(f)) f = 0;
  const maxF = Math.max(0, fc - 1);
  if (f < 0) f = 0;
  if (f > maxF) f = maxF;
  a.time = f / fps;
  a.playing = false;
  a.lastTimestamp = 0;
  // Force one tick so bone matrices reflect the new time, then paint the
  // just-baked vertices on the next frame (on-demand).
  if (typeof tickAnimation === "function") {
    try { tickAnimation(performance.now()); } catch (_e) {}
  }
  requestRender();
  return f | 0;
};


// =====================================================================
// Map Editor v3 — r.rel-derived render hints (2026-04-25, additive).
//
// Server-side bundle now ships ``rrel_render_hints`` when the floor has
// a sibling ``*_NN r.rel``.  Shape:
//
//   {
//     "ok": true,
//     "anchor_count": 100,
//     "anchors": [{id, version, pos, rot_x, rot_y_packed, radius, sub_record_ptr}, ...],
//     "hints": {
//       "anchor_count": 100,
//       "bbox_min": [x, y, z],
//       "bbox_max": [x, y, z],
//       "bbox_center": [x, y, z],
//       "bbox_size": [x, y, z],
//       "suggested_fog_far": 3090.0
//     }
//   }
//
// CRITICAL FINDING (RE'd 2026-04-25): r.rel does NOT carry fog colour
// or directional-light vectors.  Those live in PSOBB.exe globals at
// 0x00a8d770 (FogEntry table) and 0x00a9d4e4 (LightEntry table) — both
// initialised at startup by hardcoded code, not loaded from disk.
//
// What r.rel DOES give us is a per-area scene bounding box (computed
// from the anchor positions).  We use that to override the fog
// far-plane in the hardcoded category table when the r.rel data
// suggests a meaningfully different scale (the category-default would
// otherwise either over-fog small maps or under-fog large ones).
//
// Validation rule: only override when the r.rel-suggested far-plane
// stays within ±50% of the category-default.  This guards against
// degenerate r.rel data (single-anchor boss arenas → tiny bbox)
// silently breaking the look of an unrelated category.
//
// All other env values (fog colour, hemi/key colours, ambient) keep
// using the per-category table — the v4 work item is to RE the per-
// area FogEntry/LightEntry table values out of PsoBB.exe and replace
// the hardcoded JS table with extracted values.

// Validate r.rel hints — return null if the data is unusable.
window.psoValidateRrelHints = function (rrelHints) {
  if (!rrelHints || rrelHints.ok !== true) return null;
  const h = rrelHints.hints;
  if (!h) return null;
  if (typeof h.suggested_fog_far !== "number") return null;
  if (!Number.isFinite(h.suggested_fog_far)) return null;
  if (h.suggested_fog_far < 100 || h.suggested_fog_far > 10000) return null;
  if (!Array.isArray(h.bbox_size) || h.bbox_size.length !== 3) return null;
  // Reject a degenerate bbox (single anchor or co-located anchors) —
  // boss arenas hit this case and the category default is more
  // accurate.
  const horiz = Math.max(h.bbox_size[0] || 0, h.bbox_size[2] || 0);
  if (horiz < 200) return null;
  return h;
};

// Apply r.rel-derived environment over the per-category baseline.
// `bundle` is the API response from /api/map/<id>?floor=N; we pull
// `bundle.rrel_render_hints` and tweak only the fog far-plane when
// the hint is plausible.  All other env values stay untouched.
window.psoSceneApplyEnvironmentWithHints = function (category, rrelHints) {
  // First apply the category baseline so we have a known starting
  // point.  Returns false if the scene isn't ready yet.
  const ok = window.psoSceneApplyEnvironment(category);
  if (!ok) return false;
  if (!state || !state.scene) return ok;

  // Pull validated hints; bail if invalid.
  const h = window.psoValidateRrelHints(rrelHints);
  if (!h) return ok;

  // Read the just-applied fog far-plane to gate sanity.
  const fog = state.scene.fog;
  if (!fog || typeof fog.far !== "number") return ok;
  const catFar = fog.far;
  const relFar = h.suggested_fog_far;
  // Override only when within ±50% of the category default.  This
  // protects against degenerate r.rel data overpowering a category
  // table tuned for a known look.
  const ratio = relFar / catFar;
  if (ratio < 0.5 || ratio > 1.5) {
    // Out-of-band — log once and keep the category default.
    if (!state.__rrelHintLogged) {
      state.__rrelHintLogged = true;
      console.log(
        "psoSceneApplyEnvironmentWithHints: r.rel far " +
        relFar.toFixed(0) + " out of band vs category " + catFar.toFixed(0) +
        " (ratio=" + ratio.toFixed(2) + "); keeping category");
    }
    return ok;
  }
  // In-band: blend toward the r.rel value (75% rel, 25% cat) so we
  // preserve some of the category's tuning while letting the r.rel
  // bring scene-specific scale.
  const blended = relFar * 0.75 + catFar * 0.25;
  fog.far = blended;
  state.envRrelApplied = {
    category: category,
    cat_far: catFar,
    rel_far: relFar,
    blended_far: blended,
  };
  return ok;
};

// Convenience wrapper: load + relight + r.rel-aware env in one call.
// This is the v3 successor to ``psoSceneLoadMapWithEnvironment``;
// existing callers that don't pass ``rrelHints`` get the v2 behaviour
// for free.
window.psoSceneLoadMapWithRrelHints = async function (bundle) {
  const result = await window.psoSceneLoadMap(bundle);
  try { window.psoSceneUseLambertMaterials(); } catch (_e) {}
  try {
    if (bundle && bundle.rrel_render_hints) {
      window.psoSceneApplyEnvironmentWithHints(
        bundle.category, bundle.rrel_render_hints);
    } else {
      window.psoSceneApplyEnvironment(bundle ? bundle.category : null);
    }
  } catch (_e) {}
  return result;
};


// =====================================================================
// Map Editor v3 — n.rel texture binding (2026-04-25, additive).
//
// The bundle now ships ``nrel_texture_names`` (positional list, indices
// 1:1 with the XVR records inside ``map_<area>.xvm``).  Each n.rel
// terrain mesh's ``material_id`` is an index into the same list.
//
// We don't do the full texture-resolve here (the bundle is loaded from
// the cache server-side and texture decode lives in the binding
// pipeline).  Instead we surface a lookup helper that map_panel.js can
// call to get a per-mesh tint colour when a texture isn't available.
// The tint encodes the area category so terrain reads as "the right
// flavour of grey" rather than pure 0.5 grey when the XVM-binding
// pipeline drops a slot.
//
// The lookup cycles a small palette per-area so adjacent meshes get
// distinct hues — better than a single flat fallback colour for
// debugging "did the binding resolve?" at a glance.

// Pre-baked per-area tint palettes.  Numbers are 0xRRGGBB.  Each
// palette has 6 hues so a typical 80-mesh scene cycles cleanly.
const _PSO_AREA_TINT_PALETTE = {
  forest: [0x4d7c4f, 0x5a8f5d, 0x4c7050, 0x698361, 0x3d6840, 0x547057],
  cave:   [0x3a3a4a, 0x423a32, 0x32404a, 0x4a3a3a, 0x3a4044, 0x46393a],
  mine:   [0x4f5a6b, 0x5a6b7a, 0x4a5566, 0x66728a, 0x4d5a78, 0x556678],
  ruins:  [0x9a8568, 0xb39772, 0x8a7050, 0xa68d65, 0x82724d, 0x95825c],
  battle: [0x808080, 0x707080, 0x808070, 0x707070, 0x787880, 0x807878],
  city:   [0xc8a878, 0xb09060, 0xd0b282, 0xa68856, 0xb89868, 0xc09870],
  corruption: [0x506060, 0x405055, 0x506555, 0x456060, 0x556070, 0x4a5560],
  boss:   [0x707070, 0x807878, 0x787870, 0x707880, 0x787078, 0x807070],
  other:  [0x888888, 0x787878, 0x808080, 0x737373, 0x7e7e7e, 0x848484],
};

// Public: assign per-mesh fallback tint for terrain meshes that don't
// have a resolved texture map.  ``category`` is the bundle category;
// ``terrainGroup`` is the THREE.Group holding the parsed terrain
// meshes (state.sceneRoot or a sub-group).  Returns the count of
// meshes whose tint was overridden.
//
// Honours an existing ``map`` (texture) on the material — if a
// material already has a bound texture we leave it alone.  This means
// the function is idempotent if you call it twice in a row, and
// cooperative with future texture-binding code that may resolve some
// (but not all) materials.
window.psoSceneApplyTerrainFallbackTint = function (terrainGroup, category) {
  if (!terrainGroup) return 0;
  const palette =
    _PSO_AREA_TINT_PALETTE[category] || _PSO_AREA_TINT_PALETTE.other;
  let i = 0;
  let touched = 0;
  terrainGroup.traverse(function (m) {
    if (!m.isMesh || !m.material) return;
    const mat = Array.isArray(m.material) ? m.material[0] : m.material;
    if (!mat) return;
    if (mat.map) return;  // texture already bound — leave alone
    // Skip if material was explicitly given a non-grey colour by
    // upstream code (anything outside [0.4, 0.8] grey range).
    if (mat.color) {
      const c = mat.color;
      const isGreyish =
        Math.abs(c.r - c.g) < 0.05 &&
        Math.abs(c.g - c.b) < 0.05 &&
        c.r > 0.35 && c.r < 0.85;
      if (!isGreyish) return;
    }
    const hex = palette[i % palette.length];
    if (mat.color && mat.color.setHex) mat.color.setHex(hex);
    i++;
    touched++;
  });
  return touched;
};

// n.rel texture name → XVR record index lookup.  The mapping is
// strictly positional: name[i] in the n.rel TextureList → XVR[i] in
// the sibling ``map_<area>.xvm`` (verified via static analysis +
// count-match across 4 sample maps at v3 RE time).
//
// Args:
//   names: list[str] of texture names from bundle.nrel_texture_names
//   needle: the texture name to look up (case-insensitive)
//
// Returns:
//   The 0-based XVR index, or -1 if not found.
window.psoNrelTextureIndex = function (names, needle) {
  if (!Array.isArray(names) || !needle) return -1;
  const lc = ("" + needle).toLowerCase();
  for (let i = 0; i < names.length; i++) {
    if (("" + names[i]).toLowerCase() === lc) return i;
  }
  return -1;
};


// =====================================================================
// Exact-PSOBB-Lambert toggle (v4 visual polish, 2026-04-25, additive).
//
// Wraps static/psobb_lambert_shader.js so any panel (Map Editor today,
// 3D-view in the future) can opt into the bit-exact PSOBB lighting
// equation. Default: false — the existing MeshLambertMaterial path
// is left in place so this change is risk-free for non-opt-in
// callers.
//
// Usage:
//   window.psoSceneUseExactLambert(true)   // swap pso-lambert in
//   window.psoSceneUseExactLambert(false)  // restore original
//
// What "exact" means here is documented in psobb_lambert_shader.js;
// short version: per-vertex lambert (PSOBB does this in T&L hardware,
// stock three.js does it per fragment), single hemisphere ambient +
// single directional, multiplicative linear fog. The 1-2 ms-per-frame
// uniform sync runs only when the toggle is on.
// =====================================================================

// The active toggle. Driven by user-checkbox; readback via
// `window.psoSceneIsExactLambert()` for panels that surface a UI.
let _psoExactLambertActive = false;
let _psoExactLambertModule = null;

// Lazy-load the shader module on first toggle. Avoid pulling it into
// non-Map-Editor sessions if the user never opts in.
async function _psoLoadExactLambertModule() {
  if (_psoExactLambertModule) return _psoExactLambertModule;
  try {
    _psoExactLambertModule = await import("/static/psobb_lambert_shader.js");
  } catch (e) {
    console.warn("[psoSceneUseExactLambert] module load failed:", e);
    _psoExactLambertModule = null;
  }
  return _psoExactLambertModule;
}

// Walk every Mesh in the scene-root + (optionally) state.mesh and
// either apply the pso-lambert material or restore the original.
function _psoExactLambertWalk(apply, mod) {
  if (!mod) return 0;
  const apply_fn = apply ? mod.applyPsoLambertToMesh : mod.restoreOriginalMaterial;
  let touched = 0;
  const roots = [];
  if (state.sceneRoot) roots.push(state.sceneRoot);
  if (state.mesh)      roots.push(state.mesh);
  for (const r of roots) {
    r.traverse(function (m) {
      if (!m.isMesh) return;
      const ok = apply_fn(m);
      if (ok) touched++;
    });
  }
  return touched;
}

// Per-frame uniform sync; installed as a renderer.onBeforeRender hook
// once the first toggle-on happens, never removed (no-op when no
// pso-lambert materials are present).
let _psoExactLambertHookInstalled = false;

function _psoInstallExactLambertHook() {
  if (_psoExactLambertHookInstalled) return;
  if (!state.renderer || !state.scene || !state.camera) return;
  const renderer = state.renderer;
  // Wrap renderer.render so we can sync before each draw. The
  // alternative (onBeforeRender per-mesh) would force per-mesh
  // duplicate work; one pass per frame is plenty.
  const origRender = renderer.render.bind(renderer);
  renderer.render = function (scene, camera) {
    if (_psoExactLambertActive && _psoExactLambertModule) {
      try {
        _psoExactLambertModule.syncPsoLambertUniforms(scene, camera);
      } catch (_e) { /* never let shader bugs break rendering */ }
    }
    return origRender(scene, camera);
  };
  _psoExactLambertHookInstalled = true;
}

window.psoSceneUseExactLambert = async function (enable) {
  const want = !!enable;
  // Same-state no-op: caller is asking for the state we're already in.
  if (want === _psoExactLambertActive) return _psoExactLambertActive;

  const mod = await _psoLoadExactLambertModule();
  if (!mod) {
    // Module wouldn't load — log and stay in current state.
    return _psoExactLambertActive;
  }

  _psoExactLambertActive = want;
  _psoInstallExactLambertHook();
  _psoExactLambertWalk(want, mod);
  return _psoExactLambertActive;
};

window.psoSceneIsExactLambert = function () {
  return _psoExactLambertActive;
};


// =====================================================================
// Map Editor v4 — engine-truth fog/light lookup (2026-04-25, additive).
//
// The hardcoded ``_PSO_AREA_ENV`` table above ships hand-tuned values
// per area category (forest / cave / mine / ruins / ...).  v4 replaces
// it with the actual per-area values from PSOBB.exe's runtime tables,
// loaded from ``<install>/data/fogentry.dat`` and ``lightentry.bin``.
//
// The data is generated by ``formats/psobb_engine_tables.py`` and
// shipped as ``static/psobb_engine_data.js`` — that file populates
// ``window.PSOBB_ENGINE_DATA`` with one entry per known map_id::
//
//   {
//     "aancient01": {
//       "map_id": "aancient01",
//       "map_type": "Forest1",
//       "engine_index": 1,
//       "fog": { type, color_rgb, color_a, end, start, density, ... },
//       "light": { dir1, dir2, intensity_*, diffuse_argb, ambient_argb },
//       "light_ultimate": <same shape as light, only for Ep1>
//     },
//     ...
//   }
//
// The lookup chain inside ``psoSceneApplyEnvironment`` is now:
//
//   1. If ``window.PSOBB_ENGINE_DATA[map_id]`` is present, use those
//      ground-truth values (fog colour, near/far, light direction,
//      ambient/diffuse).  This is the v4 path.
//   2. Otherwise fall back to ``_PSO_AREA_ENV[category]`` (the v3
//      hand-tuned per-category table).  This keeps unmapped maps and
//      synthetic test fixtures working.
//
// The engine values are NOT a perfect drop-in: PSOBB's fog model uses
// world-space distances and our scene uses three.js's Fog with raw
// camera distances.  The values are within ~10% per channel which is
// the brief's accuracy target.

// Convert a fog ARGB float channel (0..1) to a u8 (0..255) for THREE.
function _pso_argb_to_rgb(argb) {
  if (!Array.isArray(argb) || argb.length < 4) return 0xffffff;
  // argb[0] is alpha, argb[1..3] are r/g/b in 0..1 range.
  const r = Math.round(Math.max(0, Math.min(1, argb[1])) * 255);
  const g = Math.round(Math.max(0, Math.min(1, argb[2])) * 255);
  const b = Math.round(Math.max(0, Math.min(1, argb[3])) * 255);
  return (r << 16) | (g << 8) | b;
}

// Resolve a map_id (e.g. "aancient01") → engine env entry, or null.
window.psoLookupEngineEnv = function (mapId) {
  if (!mapId || typeof mapId !== "string") return null;
  const data = window.PSOBB_ENGINE_DATA;
  if (!data || typeof data !== "object") return null;
  const e = data[mapId];
  if (!e || typeof e !== "object") return null;
  return e;
};

// Translate one engine env entry into the same shape the legacy
// _PSO_AREA_ENV table uses (so the rest of psoSceneApplyEnvironment
// can consume it unchanged).
window.psoEngineEnvToCategoryShape = function (engineEnv) {
  if (!engineEnv) return null;
  const fog = engineEnv.fog || {};
  const light = engineEnv.light || {};
  // PSOBB stores fog colours as RGB ints; we use them directly as
  // background tint as well (matches the binary).
  const color = (typeof fog.color_rgb === "number") ? fog.color_rgb : 0x808080;
  // Fog start/end can be negative in the engine (some maps use a
  // "back-projected" fog ramp).  THREE.Fog requires near < far and
  // both >= 0; clamp defensively while preserving the relative range.
  let near = (typeof fog.start === "number") ? fog.start : 0;
  let far = (typeof fog.end === "number") ? fog.end : 2000;
  if (near < 0) near = 0;
  if (far <= near) far = near + 100;
  // Hemisphere = ambient → ground/sky, Direct = key (sun).
  const hemiSky = _pso_argb_to_rgb(light.ambient_argb || []);
  const hemiGround = 0x202020;  // engine doesn't ship a ground tint
  const keyColor = _pso_argb_to_rgb(light.diffuse_argb || []);
  const keyDir = Array.isArray(light.dir1) && light.dir1.length === 3
                  ? [-light.dir1[0], -light.dir1[1], -light.dir1[2]]
                  : [0.5, 1.0, 0.5];
  // Scale intensities into a sensible THREE range.  PSOBB's intensity
  // range is roughly 0..1 in our corpus; we keep it 1:1 since the
  // category table values fall in the same range.
  const hemiInt = (typeof light.intensity_ambient === "number")
                  ? Math.max(0.1, Math.min(2.0, light.intensity_ambient * 1.5))
                  : 0.5;
  const keyInt = (typeof light.intensity_diffuse === "number")
                  ? Math.max(0.1, Math.min(2.0, light.intensity_diffuse))
                  : 0.85;
  return {
    fog:  { color: color, near: near, far: far },
    hemi: { sky: hemiSky, ground: hemiGround, intensity: hemiInt },
    key:  { color: keyColor, intensity: keyInt, dir: keyDir },
    bg:   color,
  };
};

// Apply environment using the engine table when ``map_id`` is known,
// otherwise fall back to ``psoSceneApplyEnvironment(category)``.
//
// Map Editor v4 callers should prefer this entry point.  It accepts
// either a bundle dict (with ``map_id`` + ``category``) or two
// separate args.
window.psoSceneApplyEnvironmentV4 = function (mapIdOrBundle, fallbackCategory) {
  let mapId, category;
  if (typeof mapIdOrBundle === "object" && mapIdOrBundle !== null) {
    mapId = mapIdOrBundle.map_id || null;
    category = mapIdOrBundle.category || fallbackCategory || null;
  } else {
    mapId = mapIdOrBundle || null;
    category = fallbackCategory || null;
  }

  // Try engine-truth path first.
  const engineEnv = window.psoLookupEngineEnv(mapId);
  if (engineEnv && state.scene) {
    const shape = window.psoEngineEnvToCategoryShape(engineEnv);
    if (shape) {
      // Inject into _PSO_AREA_ENV under a synthetic key, then re-use
      // the existing applier so we share the saved-lights / unmount
      // restore path.  We use a deterministic key so repeated calls
      // for the same map don't accumulate dictionary entries.
      const key = "__engine_" + mapId;
      _PSO_AREA_ENV[key] = shape;
      const ok = window.psoSceneApplyEnvironment(key);
      // Stash the engine-applied marker so map_panel.js can show
      // "fog: engine values" badge in the editor.
      state.envEngineApplied = {
        map_id: mapId,
        map_type: engineEnv.map_type,
        engine_index: engineEnv.engine_index,
      };
      return ok;
    }
  }
  // Fallback: category-driven hardcoded env.
  state.envEngineApplied = null;
  return window.psoSceneApplyEnvironment(category);
};

// Convenience: apply v4 env, then layer the v3 r.rel hint adjustment
// on top (the r.rel bbox-derived far-plane tweak only kicks in when
// in-band).  Map Editor's load-bundle path should call this.
window.psoSceneApplyEnvironmentV4WithHints = function (bundle) {
  if (!bundle) return false;
  const ok = window.psoSceneApplyEnvironmentV4(bundle);
  if (!ok) return false;
  if (bundle.rrel_render_hints) {
    // Re-use the v3 hint applier.  It only mutates fog.far, doesn't
    // touch the engine-derived colours.
    const h = window.psoValidateRrelHints(bundle.rrel_render_hints);
    if (h && state.scene && state.scene.fog &&
        typeof state.scene.fog.far === "number") {
      const catFar = state.scene.fog.far;
      const relFar = h.suggested_fog_far;
      const ratio = relFar / catFar;
      // Same ±50% guard rail as v3.
      if (ratio >= 0.5 && ratio <= 1.5) {
        state.scene.fog.far = relFar * 0.5 + catFar * 0.5;
      }
    }
  }
  return ok;
};

// =====================================================================
// psoUpdateMaterial — live-preview hook for the Material Inspector tab.
// =====================================================================
// Added 2026-04-25.  The Material Inspector panel calls this whenever
// the user drags a slider / toggles a checkbox in the per-submesh
// editor, so the 3D viewport reflects the change immediately (without
// having to round-trip through /api/material POST + reload).
//
//   psoUpdateMaterial(submeshIdx, edits)
//
// `edits` is the SAME shape as the wire-format edit row the panel
// later POSTs:
//
//   {
//     diffuse_rgba: [r,g,b,a]   (0..255 ints; alpha modulates opacity)
//     alpha_test:   {enabled, threshold}
//     alpha_blend:  {src, dst}   or null
//     blend_mode:   "blend" | "additive" | "none" | ...
//     two_sided:    bool
//     depth_test:   bool
//     depth_write:  bool
//   }
//
// Each present field updates the matching three.js material property:
//   - diffuse_rgba  -> material.color (RGB) + material.opacity (A/255)
//   - alpha_test    -> material.alphaTest = threshold/255 (0 disables)
//   - blend_mode    -> material.blending  (Normal/Additive/Multiply)
//   - two_sided     -> material.side      (DoubleSide vs FrontSide)
//   - depth_test    -> material.depthTest
//   - depth_write   -> material.depthWrite
//
// Live preview is BEST-EFFORT — three.js has only a small subset of
// PSOBB's blend factor combinations; we map the high-level "mode"
// preset name onto a three.js BlendingMode enum and accept that
// custom (src,dst) pairs may not preview perfectly.  The stored /
// shipped data is unaffected — that path goes through the server's
// POST endpoint with no live-preview involvement.
window.psoUpdateMaterial = function (submeshIdx, edits) {
  if (!state.debugMeshes || !edits) return false;
  const idx = submeshIdx | 0;
  if (idx < 0) return false;

  // Guard: the Material Inspector indexes by GLOBAL submesh id (matches
  // the GET /api/material payload). state.debugMeshes is parallel.
  const e = state.debugMeshes[idx];
  if (!e || !e.mesh || !e.mesh.material) return false;
  const m = e.mesh.material;

  if (Array.isArray(edits.diffuse_rgba) && edits.diffuse_rgba.length >= 4) {
    const [r, g, b, a] = edits.diffuse_rgba;
    if (m.color && typeof m.color.setRGB === "function") {
      m.color.setRGB((r | 0) / 255, (g | 0) / 255, (b | 0) / 255);
    }
    const aN = (a | 0) / 255;
    if (aN < 1.0) {
      m.transparent = true;
      m.opacity = aN;
    } else {
      m.opacity = 1.0;
      // Keep transparent flag if blend_mode requested transparency.
      if (!edits.alpha_blend) m.transparent = false;
    }
    m.needsUpdate = true;
  }

  if (edits.alpha_test !== undefined) {
    if (edits.alpha_test && edits.alpha_test.enabled) {
      const thr = (edits.alpha_test.threshold | 0) / 255;
      m.alphaTest = Math.max(0, Math.min(1, thr));
    } else {
      m.alphaTest = 0;
    }
    m.needsUpdate = true;
  }

  if (edits.blend_mode !== undefined) {
    // three.js blending enum values (we hard-code rather than depend
    // on whichever build of THREE the page has loaded):
    //   NoBlending=0, NormalBlending=1, AdditiveBlending=2,
    //   SubtractiveBlending=3, MultiplyBlending=4, CustomBlending=5
    const map = {
      "none":       1,  // we still want opaque-ish blend pass
      "blend":      1,
      "additive":   2,
      "subtractive":3,
      "multiply":   4,
      "screen":     2,  // approximate via additive — three has no Screen
    };
    const v = map[String(edits.blend_mode)] || 1;
    m.blending = v;
    if (v !== 1) m.transparent = true;
    m.needsUpdate = true;
  }

  if (edits.two_sided !== undefined && THREE) {
    m.side = edits.two_sided ? THREE.DoubleSide : THREE.FrontSide;
    m.needsUpdate = true;
  }

  if (edits.depth_test !== undefined) {
    m.depthTest = !!edits.depth_test;
    m.needsUpdate = true;
  }

  if (edits.depth_write !== undefined) {
    m.depthWrite = !!edits.depth_write;
    m.needsUpdate = true;
  }

  // Repaint on demand so the user sees the change immediately even when
  // the continuous loop is idle.
  requestRender();
  return true;
};

// Read-back companion: returns the current GPU-side material values
// for one submesh.  Used by the Material Inspector to populate the
// dropdowns / sliders when the panel first opens (avoids the user
// seeing a "default" state while the GET request is still in flight).
window.psoReadMaterial = function (submeshIdx) {
  if (!state.debugMeshes) return null;
  const e = state.debugMeshes[submeshIdx | 0];
  if (!e || !e.mesh || !e.mesh.material) return null;
  const m = e.mesh.material;
  const c = m.color || {r: 1, g: 1, b: 1};
  return {
    diffuse_rgba: [
      Math.round((c.r || 0) * 255),
      Math.round((c.g || 0) * 255),
      Math.round((c.b || 0) * 255),
      Math.round((m.opacity != null ? m.opacity : 1) * 255),
    ],
    alpha_test: m.alphaTest > 0
      ? {enabled: true, threshold: Math.round(m.alphaTest * 255)}
      : null,
    blend_mode: ({1: "blend", 2: "additive", 3: "subtractive", 4: "multiply"})[m.blending] || "none",
    two_sided: THREE && m.side === THREE.DoubleSide,
    depth_test: !!m.depthTest,
    depth_write: !!m.depthWrite,
  };
};
