// =====================================================================
// PSOBB Anim Editor Panel — keyframe-level motion editor
// 2026-04-25
//
// Adds an "Anim Editor" tab to the texture-panel tab strip. Workflow:
//
//   1. User opens a model (any path supported by /api/animations).
//   2. The motion picker (existing "Motions" tab) plays a motion.
//   3. User switches to this tab → server-side parse → shows a timeline
//      scrubber, bone selector, per-bone TRS editors, save buttons.
//
// All inter-panel communication goes through the existing additive
// surface on model_viewer.js + the new /api/anim_keyframe/* endpoints:
//
//   window.psoListMotions()        the model's motion list
//   window.psoLoadMotion(name)     play a motion (same as motion picker)
//   window.psoGetCurrentMotion()   currently-playing motion name
//   window.psoSetAnimationPlaying(bool)
//   window.psoGetSkeleton()        bone metadata (idx + parent + bind TRS)
//   window.psoGetRigContext()      provides modelPath + skeleton handle
//
// Dependencies: relies on the texture panel's
//   psoTexturePanelAddTabButton + psoTexturePanelRegisterTab hooks
// (same pattern as rig_panel.js / sculpt_panel.js).
//
// Wire format: see /api/anim_keyframe/load endpoint docstring in
// server.py for the full envelope. Briefly, each motion is:
//   { name, frame_count, type_flags, interpolation, fps, bone_count,
//     bones: [{ idx, present, narrow_ang, kf: [...] }],
//     round_trip: { ... } /* opaque round-trip metadata */ }
//
// Edits mutate `bones[i].kf` in place, then we POST the whole envelope
// to /save (or /insert / /delete for surgical mutations).
// =====================================================================

(function () {
  "use strict";
  if (window.__psoAnimEditorPanelLoaded) return;
  window.__psoAnimEditorPanelLoaded = true;

  // ------------------------------------------------------------------
  // Constants + state
  // ------------------------------------------------------------------
  const TAB_NAME = "anim_editor";
  const TAB_LABEL = "Anim Editor";
  const TAB_TITLE = "keyframe editor: scrub, edit per-bone TRS, save .njm";
  const STYLE_ID = "psoAnimEditorPanelStyle";

  // NJM type bits (must match formats/njm.py).
  const NJD_POS = 1 << 0;
  const NJD_ANG = 1 << 1;
  const NJD_SCL = 1 << 2;
  const NJD_QUAT = 1 << 13;

  const BAMS_TO_DEG = 360.0 / 65536.0;
  const DEG_TO_BAMS = 65536.0 / 360.0;

  const PLAY_FPS_OPTIONS = [15, 24, 30, 60];

  // Bone-tree fallback threshold: skeletons smaller than this stay on
  // the original flat dropdown. The tree shines on 100+-bone rigs (the
  // dragon has 124).
  const BONE_TREE_THRESHOLD = 12;

  // Curve-editor canvas height — taller in "all-TRS" mode so 9 curves
  // don't crowd. drawCurveCanvas picks the height based on curveMode.
  const CURVE_CANVAS_HEIGHT = 140;
  const CURVE_CANVAS_HEIGHT_ALL = 220;
  // Curve channel options: which scalar component to plot. Y axis label
  // depends on whether this is POS (world units) or ANG (BAMS).
  // The colour column drives Task 1's multi-channel overlay rendering.
  const CURVE_CHANNELS = [
    { key: "tx", label: "POS X", trs: "t", axis: 0, kind: NJD_POS, color: "#ff5060" },
    { key: "ty", label: "POS Y", trs: "t", axis: 1, kind: NJD_POS, color: "#56e060" },
    { key: "tz", label: "POS Z", trs: "t", axis: 2, kind: NJD_POS, color: "#5090ff" },
    { key: "rx", label: "ANG X", trs: "r", axis: 0, kind: NJD_ANG, color: "#ff5060" },
    { key: "ry", label: "ANG Y", trs: "r", axis: 1, kind: NJD_ANG, color: "#56e060" },
    { key: "rz", label: "ANG Z", trs: "r", axis: 2, kind: NJD_ANG, color: "#5090ff" },
    { key: "sx", label: "SCL X", trs: "s", axis: 0, kind: NJD_SCL, color: "#ff5060" },
    { key: "sy", label: "SCL Y", trs: "s", axis: 1, kind: NJD_SCL, color: "#56e060" },
    { key: "sz", label: "SCL Z", trs: "s", axis: 2, kind: NJD_SCL, color: "#5090ff" },
  ];

  const state = {
    bodyEl: null,
    panelMounted: false,
    // Loaded motion data (full envelope from /api/anim_keyframe/load).
    motion: null,                 // mutable: edits go in here
    motionOriginal: null,         // deep clone on load — for "Compare" overlay
    motionName: "",               // e.g. "walk_boss1"
    modelPath: "",                // for /load + /import/animation/swap
    saveAsName: "",               // user-supplied save filename (defaults to motionName + ".njm")
    swapTargetSlot: "",           // for the "replace slot in BML" button
    // Selection / scrubber.
    selectedBoneIdx: 0,
    selectedFrame: 0,             // current scrubber position (integer)
    selectedKeyframeIdx: -1,      // index into bones[selectedBoneIdx].kf, or -1
    showCompare: true,            // overlay original on timeline lanes
    // Playback.
    playing: false,
    loop: true,
    fps: 30,
    playbackTimer: null,
    lastPlaybackTimestamp: 0,
    // Skeleton snapshot for the bone selector.
    skeleton: null,
    // Status pip.
    status: { state: "idle", msg: "ready" },
    // Inspector debouncer for sliders that fire many input events.
    inspectorPending: null,

    // ---- Task 1: scrubber → 3D pose live sync ------------------------
    // Throttle scrubber-driven psoSeekAnimationToFrame calls to one
    // per rAF tick so high-frequency drag events don't spam the
    // viewport's bone re-bake. `seekRafPending` holds the pending
    // frame to seek to; the rAF callback consumes it.
    seekRafPending: null,         // null | number
    seekRafScheduled: false,
    seekWasPlaying: false,        // remember playback state at drag-start

    // ---- Task 2: bone tree -------------------------------------------
    boneTreeQuery: "",            // search-filter substring (lowercased)
    boneTreeHidden: new Set(),    // bone idxs whose keyframes are bypassed
    boneTreeCollapsed: new Set(), // bone idxs whose subtree is folded

    // ---- Task 3: multi-keyframe selection ----------------------------
    // Each entry: { bone, kfIdx } — using bone+kfIdx is stable across
    // edits because we always re-resolve via bone.kf[kfIdx] before use.
    // We keep the active selection on the CURRENT bone only (drag-move
    // semantics get messy across bones) but ctrl+a / shift+click can
    // produce a within-bone multi-selection.
    selectedKfSet: new Set(),     // string keys of "<kfIdx>" on selectedBoneIdx
    // Marquee-drag state for selection box on the timeline canvas.
    marquee: null,                // { startX, currentX, startBoneLane }
    // Drag-move state for moving selected keyframes horizontally.
    kfDrag: null,                 // { anchorFrame, lastDelta }
    // Clipboard for copy/paste — holds plain kf objects, no kfIdx ref.
    kfClipboard: [],

    // ---- Task 4: curve editor ----------------------------------------
    curveOpen: false,             // user-toggled visibility (header button)
    curveChannelKey: "rx",        // which scalar channel to plot (active in single/triplet/all)
    // v4 / Task 1 — multi-channel overlay. "single" plots one curve
    // (legacy), "triplet" plots POS/ANG/SCL x/y/z (depending on the
    // current channel's TRS bucket), "all" plots all 9 TRS channels
    // with a legend. The active channel (curveChannelKey) is the only
    // one whose handles can be edited; others are read-only overlays.
    curveMode: "single",          // "single" | "triplet" | "all"
    // Per (bone, kfIdx, channelKey) bezier-handle state. Handles are in
    // *frame*/value units, expressed as offsets from the keyframe's
    // (t, value) anchor. Stored on the JS state, not in the wire format
    // (the runtime ignores them — see save-densify path).
    // Key: `${boneIdx}:${kfIdx}:${channelKey}`
    bezierHandles: new Map(),     // key -> { inDx, inDy, outDx, outDy }
    curveDrag: null,              // { kfIdx, side: "in"|"out" }
    // Densify sample stride for save: how many frames to bake between
    // adjacent bezier-curve segments. 1 = every frame.
    curveBakeStride: 1,

    // ---- v4 / Task 3: cross-bone marquee mode ----------------------
    // When "all", marquee selection treats the timeline as a single
    // surface across every bone. Selection set keys become
    // "<boneIdx>:<kfIdx>" instead of "<kfIdx>". Operations (move,
    // delete, copy/paste) iterate through bones independently.
    marqueeMode: "single",        // "single" | "all"
    // Cross-bone selection set (only used when marqueeMode === "all").
    // Each entry is "<boneIdx>:<kfIdx>". Maintained in parallel with
    // selectedKfSet — code that consumes the selection picks one based
    // on marqueeMode.
    selectedKfSetMulti: new Set(),

    // ---- v4 / Task 2: bezier handle persistence -----------------------
    // True after a successful sidecar fetch (so we don't re-fetch).
    sidecarFetched: false,
  };

  // ------------------------------------------------------------------
  // Style injection
  // ------------------------------------------------------------------
  function ensureStyleInjected() {
    if (document.getElementById(STYLE_ID)) return;
    const css = `
      .ake-block {
        padding: 6px;
        display: flex;
        flex-direction: column;
        gap: 4px;
        font-size: 11px;
        color: #c7d8ec;
        height: 100%;
        min-height: 0;
        overflow-y: auto;
      }
      .ake-empty {
        padding: 12px;
        color: #99a4b3;
        font-size: 11px;
        text-align: center;
      }
      .ake-row { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
      .ake-row label { color: #99a4b3; display: flex; align-items: center; gap: 4px; }
      .ake-row select,
      .ake-row input[type="text"],
      .ake-row input[type="number"] {
        background: #0a0e13;
        color: #c7d8ec;
        border: 1px solid #2a313a;
        font: inherit;
        padding: 1px 4px;
        border-radius: 2px;
      }
      .ake-row select { min-width: 0; }
      .ake-row input[type="text"] { flex: 1; min-width: 80px; }
      .ake-section {
        border: 1px solid #2a313a;
        border-radius: 3px;
        background: rgba(0,0,0,0.20);
        padding: 4px 6px;
      }
      .ake-section-title {
        color: #56c8c8;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
      }
      .ake-btn {
        background: transparent;
        border: 1px solid #2a313a;
        color: #99a4b3;
        cursor: pointer;
        padding: 2px 6px;
        font: inherit;
        border-radius: 2px;
      }
      .ake-btn:hover { border-color: #00ffff; color: #00ffff; }
      .ake-btn.primary { border-color: #4a90e2; color: #c7d8ec; }
      .ake-btn.primary:hover { background: rgba(74,144,226,0.18); border-color: #00ffff; color: #00ffff; }
      .ake-btn.danger { border-color: #4d2323; color: #d89090; }
      .ake-btn.danger:hover { border-color: #ff6680; color: #ff6680; }
      .ake-btn.on {
        background: rgba(0, 255, 255, 0.12);
        border-color: #00ffff;
        color: #00ffff;
      }
      .ake-btn:disabled { opacity: 0.4; cursor: not-allowed; }
      .ake-status {
        padding: 2px 6px;
        border-radius: 2px;
        font-size: 10px;
        font-variant-numeric: tabular-nums;
      }
      .ake-status.idle { color: #6c7785; }
      .ake-status.busy { color: #ffaa00; }
      .ake-status.ok   { color: #56c8c8; }
      .ake-status.err  { color: #ff6680; }
      .ake-timeline {
        position: relative;
        background: #0a0e13;
        border: 1px solid #2a313a;
        border-radius: 2px;
      }
      .ake-timeline canvas {
        display: block;
        width: 100%;
        cursor: crosshair;
      }
      .ake-timeline-labels {
        display: flex;
        justify-content: space-between;
        color: #6c7785;
        font-size: 9px;
        font-variant-numeric: tabular-nums;
        padding: 2px 4px;
      }
      .ake-tracklane {
        position: relative;
        height: 16px;
        background: rgba(0,0,0,0.30);
        border-bottom: 1px solid #15191f;
      }
      .ake-tracklane.active { background: rgba(0,255,255,0.08); }
      .ake-tracklane-label {
        position: absolute; left: 4px; top: 1px;
        color: #6c7785; font-size: 9px; pointer-events: none;
        z-index: 2;
      }
      .ake-inspector {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 4px;
        font-size: 11px;
      }
      .ake-inspector-row {
        display: contents;
      }
      .ake-axis-block {
        display: flex; flex-direction: column; gap: 2px;
      }
      .ake-axis-block label { color: #99a4b3; font-size: 10px; }
      .ake-axis-block input[type="range"] { width: 100%; }
      .ake-axis-block input[type="number"] {
        background: #0a0e13;
        color: #c7d8ec;
        border: 1px solid #2a313a;
        font: inherit;
        padding: 1px 2px;
        border-radius: 2px;
        width: 100%;
        font-variant-numeric: tabular-nums;
      }
      .ake-axis-readout { color: #6c7785; font-size: 9px; font-variant-numeric: tabular-nums; }
      .ake-channel-toggles { display: flex; gap: 6px; padding: 2px 0; }
      .ake-channel-toggles label { color: #99a4b3; font-size: 10px; gap: 3px; }
      .ake-bone-pick {
        background: #0a0e13;
        color: #c7d8ec;
        border: 1px solid #2a313a;
        font: inherit;
        padding: 1px 4px;
        border-radius: 2px;
        flex: 1;
        min-width: 80px;
      }
      .ake-frame-line {
        font-size: 10px;
        color: #6c7785;
        font-variant-numeric: tabular-nums;
      }
      .ake-actionrow {
        display: flex; gap: 4px; flex-wrap: wrap;
      }
      .ake-actionrow .grow { flex: 1; }
      .ake-playback {
        display: flex; gap: 3px; align-items: center;
        padding: 2px 0;
      }
      .ake-playback .grow { flex: 1; }
      .ake-keyframe-list {
        max-height: 100px;
        overflow-y: auto;
        font-size: 10px;
        font-variant-numeric: tabular-nums;
        background: rgba(0,0,0,0.20);
        border: 1px solid #2a313a;
        border-radius: 2px;
      }
      .ake-keyframe-row {
        display: flex; padding: 1px 4px;
        cursor: pointer; gap: 6px;
        border-bottom: 1px solid #15191f;
      }
      .ake-keyframe-row:hover { background: rgba(74,144,226,0.10); }
      .ake-keyframe-row.active { background: rgba(0,255,255,0.18); color: #c7d8ec; }
      .ake-keyframe-row.active::before { content: ">"; color: #00ffff; margin-right: 2px; }
      .ake-keyframe-row.selected { background: rgba(255, 200, 80, 0.16); }
      .ake-keyframe-row .num { color: #6c7785; min-width: 32px; }

      /* Bone tree — mirrors rig_panel.js styles but uses ake- prefix */
      .ake-tree-host {
        display: flex; gap: 4px; align-items: stretch;
      }
      .ake-tree-side {
        flex: 0 0 220px;
        max-width: 280px;
        display: flex; flex-direction: column; gap: 3px;
      }
      .ake-tree-search {
        background: #0a0e13;
        color: #c7d8ec;
        border: 1px solid #2a313a;
        font: inherit;
        padding: 1px 4px;
        border-radius: 2px;
        width: 100%;
        box-sizing: border-box;
      }
      .ake-tree-list {
        flex: 1;
        max-height: 280px;
        overflow-y: auto;
        background: rgba(0, 0, 0, 0.25);
        border: 1px solid #2a313a;
        border-radius: 2px;
        padding: 2px;
        font-size: 10px;
      }
      .ake-tree-row {
        display: flex; gap: 3px; padding: 1px 3px;
        cursor: pointer;
        border-radius: 2px;
        line-height: 1.4;
        white-space: nowrap;
      }
      .ake-tree-row:hover { background: rgba(74, 144, 226, 0.10); }
      .ake-tree-row.active {
        background: rgba(0, 255, 255, 0.12);
        outline: 1px solid #00ffff;
      }
      .ake-tree-row.hidden { opacity: 0.4; }
      .ake-tree-row.dim { color: #6c7785; }
      .ake-tree-toggle {
        width: 12px; text-align: center; user-select: none;
        color: #6c7785; cursor: pointer;
      }
      .ake-tree-toggle:hover { color: #c7d8ec; }
      .ake-tree-eye {
        width: 14px; text-align: center; user-select: none;
        color: #6c7785; cursor: pointer;
      }
      .ake-tree-eye:hover { color: #c7d8ec; }
      .ake-tree-name {
        flex: 1;
        overflow: hidden; text-overflow: ellipsis;
      }
      .ake-tree-idx {
        color: #6c7785; font-variant-numeric: tabular-nums;
        min-width: 26px; text-align: right;
      }
      .ake-tree-kfn { color: #56c8c8; min-width: 22px; text-align: right; font-variant-numeric: tabular-nums; }
      .ake-tree-empty { color: #6c7785; padding: 4px; text-align: center; }
      .ake-tree-summary { color: #6c7785; font-size: 9px; padding: 2px 4px; }

      /* Marquee selection rectangle on the timeline canvas */
      .ake-marquee {
        position: absolute;
        background: rgba(255, 200, 80, 0.10);
        border: 1px dashed rgba(255, 200, 80, 0.55);
        pointer-events: none;
        z-index: 3;
      }

      /* Context menu for selected keyframes */
      .ake-ctx-menu {
        position: fixed;
        z-index: 1000;
        background: #0e131a;
        border: 1px solid #2a313a;
        border-radius: 3px;
        padding: 2px 0;
        font-size: 11px;
        color: #c7d8ec;
        min-width: 130px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.6);
      }
      .ake-ctx-menu .item {
        padding: 3px 10px;
        cursor: pointer;
      }
      .ake-ctx-menu .item:hover { background: rgba(74,144,226,0.18); }
      .ake-ctx-menu .item.danger { color: #ff8893; }
      .ake-ctx-menu .item.disabled { color: #4d5460; cursor: default; }
      .ake-ctx-menu .sep {
        border-top: 1px solid #2a313a;
        margin: 2px 0;
      }

      /* Curve editor */
      .ake-curve-host {
        position: relative;
        background: #07090d;
        border: 1px solid #2a313a;
        border-radius: 2px;
      }
      .ake-curve-host canvas {
        display: block;
        width: 100%;
        cursor: crosshair;
      }
      .ake-curve-toolbar {
        display: flex; gap: 4px; align-items: center;
        flex-wrap: wrap;
        padding: 2px 0;
      }
      .ake-curve-note {
        color: #ffaa00;
        font-size: 10px;
        padding: 2px 4px;
      }
      /* v4 / Task 1 — channel overlay legend */
      .ake-curve-overlay-modes { display: inline-flex; gap: 2px; }
      .ake-curve-overlay-modes .ake-btn { padding: 1px 6px; font-size: 10px; }
      .ake-curve-legend {
        display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
        padding: 2px 4px; font-size: 10px; color: #99a4b3;
      }
      .ake-curve-legend-item {
        display: inline-flex; align-items: center; gap: 3px;
        cursor: pointer;
        padding: 1px 4px;
        border-radius: 2px;
      }
      .ake-curve-legend-item:hover { color: #c7d8ec; background: rgba(74,144,226,0.10); }
      .ake-curve-legend-item.active { color: #c7d8ec; outline: 1px solid #00ffff; }
      .ake-curve-legend-swatch {
        display: inline-block;
        width: 10px; height: 10px;
        border-radius: 2px;
      }
      /* v4 / Task 3 — marquee mode toggle + bone-id annotation */
      .ake-marquee-mode { display: inline-flex; gap: 2px; }
      .ake-marquee-mode .ake-btn { padding: 1px 6px; font-size: 10px; }
      .ake-keyframe-row .bone-tag {
        color: #6c7785;
        font-size: 9px;
        margin-right: 4px;
      }
    `;
    const el = document.createElement("style");
    el.id = STYLE_ID;
    el.textContent = css;
    document.head.appendChild(el);
  }

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------
  function escapeHtml(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;",
      '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  function setStatus(stateName, msg) {
    state.status = { state: stateName || "idle", msg: msg || "" };
    const node = state.bodyEl ? state.bodyEl.querySelector('[data-region="status"]') : null;
    if (node) {
      node.className = `ake-status ${state.status.state}`;
      node.textContent = state.status.msg;
    }
  }

  function bamsToDeg(b) {
    // Sign-extend the BAMS value before converting (so 0x8000+ angles
    // read as negative degrees in the UI — easier to author).
    let v = b | 0;
    if (v >= 0x8000) v -= 0x10000;
    return v * BAMS_TO_DEG;
  }

  function degToBams(d) {
    // Modulo into the signed 16-bit range.
    let v = Math.round(d * DEG_TO_BAMS);
    while (v < -0x8000) v += 0x10000;
    while (v >= 0x8000) v -= 0x10000;
    return v;
  }

  function deepClone(obj) {
    return JSON.parse(JSON.stringify(obj));
  }

  function getCurrentModelPath() {
    // Try the rig context first (has the cleanest path), then fall
    // through to other surface getters.
    if (typeof window.psoGetRigContext === "function") {
      const ctx = window.psoGetRigContext();
      if (ctx && ctx.modelPath) return ctx.modelPath;
    }
    if (typeof window.psoGetSculptMeshGroup === "function") {
      const ctx = window.psoGetSculptMeshGroup();
      if (ctx && ctx.modelPath) return ctx.modelPath;
    }
    if (typeof window.psoGetCurrentTextureArchive === "function") {
      const arch = window.psoGetCurrentTextureArchive();
      if (arch) return arch;
    }
    return "";
  }

  function getSkeleton() {
    if (typeof window.psoGetSkeleton === "function") {
      try { return window.psoGetSkeleton() || []; } catch { return []; }
    }
    return [];
  }

  // ------------------------------------------------------------------
  // Tab integration
  // ------------------------------------------------------------------
  function injectAnimEditorTab() {
    if (typeof window.psoTexturePanelAddTabButton !== "function") return false;
    if (typeof window.psoTexturePanelRegisterTab !== "function") return false;
    const ok = window.psoTexturePanelAddTabButton(TAB_NAME, TAB_LABEL, TAB_TITLE);
    window.psoTexturePanelRegisterTab(TAB_NAME, (body) => renderPanel(body));
    return ok;
  }

  function waitForPanel(deadline) {
    if (injectAnimEditorTab()) return;
    if (Date.now() > deadline) {
      console.warn("[anim_editor_panel] texture panel never appeared");
      return;
    }
    setTimeout(() => waitForPanel(deadline), 250);
  }

  // ------------------------------------------------------------------
  // Top-level render
  // ------------------------------------------------------------------
  function renderPanel(body) {
    ensureStyleInjected();
    state.bodyEl = body;
    state.panelMounted = true;
    body.innerHTML = `
      <div class="ake-block">
        <div class="ake-row">
          <strong style="color:#56c8c8">Anim Editor</strong>
          <span class="ake-status ${state.status.state}" data-region="status">${escapeHtml(state.status.msg)}</span>
          <span class="grow" style="flex:1"></span>
          <button class="ake-btn" data-act="reload" title="reload current motion from disk">Reload</button>
        </div>
        <div class="ake-row">
          <label>Motion:</label>
          <select class="ake-bone-pick" data-region="motionPicker"></select>
          <button class="ake-btn" data-act="load" title="parse the selected motion into the editor">Load</button>
        </div>
        <div data-region="editor"></div>
      </div>
    `;
    body.addEventListener("click", onPanelClick);
    body.addEventListener("change", onPanelChange);
    body.addEventListener("input", onPanelInput);
    // Keyboard shortcuts (Task 3): Ctrl+A select-all, Ctrl+D deselect,
    // Delete remove, Ctrl+C copy, Ctrl+V paste. Bound to keydown on the
    // panel body — only fire when the user is interacting with the
    // editor (i.e. the body or any descendant has focus). We use
    // pointer-down to set a focus marker so users don't have to tab.
    body.addEventListener("keydown", onPanelKey);
    // Keep the panel "alive" for keystrokes — make it focusable when
    // the user clicks anywhere inside.
    if (!body.hasAttribute("tabindex")) body.setAttribute("tabindex", "-1");
    body.addEventListener("pointerdown", () => {
      if (document.activeElement === document.body) body.focus();
    });

    refreshMotionPicker();
    refreshEditor();
  }

  function onPanelKey(ev) {
    if (!state.motion) return;
    // Ignore keys typed into editable inputs (text/number/range) so the
    // user can still type filenames + bone names.
    const ae = document.activeElement;
    if (ae && (ae.tagName === "INPUT" || ae.tagName === "TEXTAREA" || ae.tagName === "SELECT")) {
      // Allow Ctrl+A / Ctrl+D to bypass the input only on tree-search.
      if (ae.dataset && ae.dataset.region === "treeSearch") {
        // pass through — let the input handle text editing.
      } else {
        return;
      }
    }
    const ctrl = ev.ctrlKey || ev.metaKey;
    if (ctrl && (ev.key === "a" || ev.key === "A")) {
      ev.preventDefault();
      selectAllKeyframes();
      return;
    }
    if (ctrl && (ev.key === "d" || ev.key === "D")) {
      ev.preventDefault();
      _clearAllSelections();
      refreshEditor();
      return;
    }
    if (ctrl && (ev.key === "c" || ev.key === "C")) {
      ev.preventDefault();
      copySelectedKeyframes();
      return;
    }
    if (ctrl && (ev.key === "v" || ev.key === "V")) {
      ev.preventDefault();
      pasteKeyframes();
      return;
    }
    if (ev.key === "Delete" || ev.key === "Backspace") {
      if (_hasSelection()) {
        ev.preventDefault();
        deleteSelectedKeyframes();
      }
    }
  }

  // ------------------------------------------------------------------
  // Motion picker
  // ------------------------------------------------------------------
  async function refreshMotionPicker() {
    if (!state.bodyEl) return;
    const sel = state.bodyEl.querySelector('[data-region="motionPicker"]');
    if (!sel) return;
    const motions = (typeof window.psoListMotions === "function")
      ? window.psoListMotions() : [];
    const current = (typeof window.psoGetCurrentMotion === "function")
      ? window.psoGetCurrentMotion() : null;
    const targetName = state.motionName || current || (motions[0] && motions[0].name) || "";
    sel.innerHTML = motions.length === 0
      ? `<option value="">(no motions — open a model first)</option>`
      : motions.map((m) => `<option value="${escapeHtml(m.name)}" ${
        m.name === targetName ? "selected" : ""
      }>${escapeHtml(m.name)} (${m.frame_count}f)</option>`).join("");
  }

  // ------------------------------------------------------------------
  // Editor body (post-load)
  // ------------------------------------------------------------------
  function refreshEditor() {
    if (!state.bodyEl) return;
    const editor = state.bodyEl.querySelector('[data-region="editor"]');
    if (!editor) return;
    if (!state.motion) {
      editor.innerHTML = `<div class="ake-empty">No motion loaded.<br>
        Open a model + pick a motion + click "Load" above.</div>`;
      return;
    }
    const m = state.motion;
    state.skeleton = getSkeleton();
    const frameMax = Math.max(0, (m.frame_count | 0) - 1);
    const bones = m.bones || [];
    state.selectedBoneIdx = Math.min(state.selectedBoneIdx, Math.max(0, bones.length - 1));
    state.selectedFrame = Math.min(state.selectedFrame, frameMax);
    const bone = bones[state.selectedBoneIdx] || null;
    const fpsOpts = PLAY_FPS_OPTIONS.map((f) =>
      `<option value="${f}" ${f === state.fps ? "selected" : ""}>${f} fps</option>`,
    ).join("");
    editor.innerHTML = `
      <div class="ake-section">
        <div class="ake-section-title">Header</div>
        <div class="ake-frame-line">
          <span>name=<b>${escapeHtml(m.name)}</b></span> ·
          <span>frames=<b>${m.frame_count}</b></span> ·
          <span>bones=<b>${bones.length}</b></span> ·
          <span>type=<b>0x${(m.type_flags|0).toString(16)}</b></span> ·
          <span>interp=<b>${m.interpolation}</b></span>
        </div>
        <div class="ake-row">
          <label>FPS (display):
            <select data-region="fpsSel">${fpsOpts}</select>
          </label>
        </div>
      </div>

      <div class="ake-section" data-region="timeline-section">
        <div class="ake-section-title">Timeline · frame <span data-region="frameLabel">${state.selectedFrame}</span> / ${frameMax}</div>
        <div class="ake-timeline" data-region="timelineHost">
          <canvas data-region="timelineCanvas" height="${state.marqueeMode==="all"?Math.max(80, Math.min(180, 30 + bones.length * 4)):80}"></canvas>
        </div>
        <div class="ake-timeline-labels">
          <span>0</span>
          <span data-region="hover" style="color:#56c8c8"></span>
          <span>${frameMax}</span>
        </div>
        <div class="ake-row">
          <input type="range" min="0" max="${frameMax}" value="${state.selectedFrame}" step="1"
                 style="flex:1" data-region="scrub">
          <input type="number" min="0" max="${frameMax}" value="${state.selectedFrame}" step="1"
                 style="width:54px" data-region="frameNum">
          <label><input type="checkbox" data-region="compare" ${state.showCompare?"checked":""}> compare</label>
        </div>
        <div class="ake-row" style="font-size:10px">
          <label>Marquee:</label>
          <span class="ake-marquee-mode">
            <button class="ake-btn ${state.marqueeMode==="single"?"on":""}"
                    data-act="marquee-mode" data-mode="single"
                    title="marquee selects keyframes on the active bone only">single bone</button>
            <button class="ake-btn ${state.marqueeMode==="all"?"on":""}"
                    data-act="marquee-mode" data-mode="all"
                    title="marquee selects across every bone (delete/move/copy work bone-relative)">all bones</button>
          </span>
          <span style="color:#6c7785">${state.marqueeMode==="all"?multiSelectionSummary():""}</span>
        </div>
        <div class="ake-playback">
          <button class="ake-btn" data-act="rewind" title="frame 0">|&lt;</button>
          <button class="ake-btn" data-act="step-back" title="prev frame">&lt;</button>
          <button class="ake-btn ${state.playing?"on":""}" data-act="play" title="play / pause">${state.playing?"❚❚":"▶"}</button>
          <button class="ake-btn" data-act="step-fwd" title="next frame">&gt;</button>
          <button class="ake-btn ${state.loop?"on":""}" data-act="loop" title="loop playback">⟳</button>
          <span style="flex:1"></span>
          <button class="ake-btn ${state.curveOpen?"on":""}" data-act="curve-toggle" title="toggle curve editor (bezier handles)">curve</button>
        </div>
      </div>

      <div class="ake-section">
        <div class="ake-section-title">Bone</div>
        ${renderBoneSelector(bones)}
        ${renderBoneInspector(bone)}
      </div>

      <div class="ake-section">
        <div class="ake-section-title">
          Keyframes (bone ${state.selectedBoneIdx})
          <span class="ake-axis-readout" style="margin-left:auto;float:right" data-region="selSummary">${selectionSummary()}</span>
        </div>
        <div class="ake-keyframe-list" data-region="kfList">
          ${renderKeyframeList(bone)}
        </div>
        <div class="ake-actionrow" style="margin-top:4px">
          <button class="ake-btn primary" data-act="kf-insert" title="insert keyframe at scrubber from current bone state">Insert kf</button>
          <button class="ake-btn danger" data-act="kf-delete" title="delete keyframe at scrubber">Delete kf</button>
          <button class="ake-btn" data-act="bone-reset" title="restore this bone's keyframes from original">Reset bone</button>
          <span style="flex:1"></span>
          <button class="ake-btn" data-act="kf-select-all" title="select all keyframes (Ctrl+A) — bone-scoped in single mode, all bones in all-bones mode">Sel all</button>
          <button class="ake-btn" data-act="kf-deselect" title="deselect (Ctrl+D)">Desel</button>
          <button class="ake-btn" data-act="kf-delete-sel" title="delete selected keyframes" ${!_hasSelection()?"disabled":""}>Del sel (${state.marqueeMode==="all"?state.selectedKfSetMulti.size:state.selectedKfSet.size})</button>
        </div>
      </div>

      <div class="ake-section" data-region="curveSection" style="${state.curveOpen?"":"display:none"}">
        <div class="ake-section-title">Curve editor (bezier handles)</div>
        ${renderCurveEditorBlock(bone)}
      </div>

      <div class="ake-section">
        <div class="ake-section-title">Save</div>
        <div class="ake-row">
          <label>filename:</label>
          <input type="text" data-region="saveName" value="${escapeHtml(state.saveAsName || (m.name + ".njm"))}">
        </div>
        <div class="ake-actionrow">
          <button class="ake-btn primary" data-act="save" title="encode + stage to cache/njm_export/">Save</button>
          <button class="ake-btn" data-act="save-as-new" title="copy current to a new motion name">Save as new</button>
          <button class="ake-btn danger" data-act="reset-all" title="discard ALL edits to this motion">Reset all</button>
        </div>
        <div class="ake-row" style="margin-top:6px">
          <label>swap target slot:</label>
          <input type="text" data-region="swapTarget" placeholder="<inner>.njm in target BML"
                 value="${escapeHtml(state.swapTargetSlot)}">
          <button class="ake-btn" data-act="swap" title="POST /api/import/animation/swap to splice the saved .njm into the BML inner slot">Replace slot in BML</button>
        </div>
      </div>
    `;
    drawTimelineCanvas();
    // Re-attach pointer handlers after the canvas was re-created. The
    // attacher is idempotent — it tags the canvas so it only binds once.
    attachTimelineHandlers();
    if (state.curveOpen) {
      // Draw inside an rAF so the canvas gets its real layout width.
      requestAnimationFrame(() => {
        drawCurveCanvas();
        attachCurveHandlers();
      });
    }
  }

  // ------------------------------------------------------------------
  // Bone selector — flat dropdown for tiny skeletons, hierarchical
  // tree (with search + eye-toggle) for everything else. Threshold is
  // BONE_TREE_THRESHOLD (= 12). The dragon's 124 bones absolutely need
  // the tree.
  // ------------------------------------------------------------------
  function renderBoneSelector(bones) {
    const treeable = bones.length >= BONE_TREE_THRESHOLD;
    if (!treeable) {
      // Fallback dropdown — same as v2.
      return `
        <div class="ake-row">
          <label style="flex:1">
            <select class="ake-bone-pick" data-region="bonePick">
              ${bones.map((b, idx) => {
                const sk = state.skeleton[idx];
                const parent = sk ? sk.parent : -1;
                const kfn = (b.kf || []).length;
                return `<option value="${idx}" ${idx===state.selectedBoneIdx?"selected":""}>
                  bone ${idx} (parent=${parent}, kf=${kfn})
                </option>`;
              }).join("")}
            </select>
          </label>
        </div>
      `;
    }
    // Tree mode.
    return `
      <div class="ake-tree-host">
        <div class="ake-tree-side">
          <input type="text" class="ake-tree-search" placeholder="filter bones (by name)…"
                 data-region="treeSearch" value="${escapeHtml(state.boneTreeQuery)}">
          <div class="ake-tree-list" data-region="boneTree">${buildBoneTreeHtml(bones)}</div>
          <div class="ake-tree-summary">
            ${bones.length} bones · ${state.boneTreeHidden.size} hidden ·
            <button class="ake-btn" data-act="tree-collapse-all" title="collapse all">−</button>
            <button class="ake-btn" data-act="tree-expand-all" title="expand all">+</button>
          </div>
        </div>
      </div>
    `;
  }

  function buildBoneTreeHtml(bones) {
    const skel = state.skeleton || [];
    if (bones.length === 0) {
      return `<div class="ake-tree-empty">no bones</div>`;
    }
    // Build children map. Some skeletons return parent=-1 for siblings;
    // bone 0 is conventionally the root.
    const childrenOf = new Map();
    for (let i = 0; i < bones.length; i++) childrenOf.set(i, []);
    for (let i = 0; i < bones.length; i++) {
      const sk = skel[i];
      const p = sk ? (sk.parent | 0) : -1;
      if (p >= 0 && childrenOf.has(p)) childrenOf.get(p).push(i);
    }
    const q = (state.boneTreeQuery || "").toLowerCase().trim();
    // When search is active, we collect EVERY matching bone into a
    // flat list (with full ancestor names visible via indent), instead
    // of pruning the tree (which would hide parents that don't
    // themselves match). This is the classic "filterable tree"
    // pattern users expect.
    if (q) {
      const out = [];
      for (let i = 0; i < bones.length; i++) {
        const name = boneDisplayName(i, skel);
        const haystack = `${name.toLowerCase()} #${i} bone${i}`;
        if (haystack.indexOf(q) < 0) continue;
        out.push(emitBoneTreeRow(i, 0, bones, skel, /*hasChildren=*/false, /*forceFlat=*/true));
      }
      if (out.length === 0) {
        return `<div class="ake-tree-empty">no match for "${escapeHtml(q)}"</div>`;
      }
      return out.join("");
    }
    // No filter — recurse from all roots.
    const out = [];
    function emit(idx, depth) {
      const kids = childrenOf.get(idx) || [];
      const isCollapsed = state.boneTreeCollapsed.has(idx) && kids.length > 0;
      out.push(emitBoneTreeRow(idx, depth, bones, skel, kids.length > 0, false, isCollapsed));
      if (isCollapsed) return;
      for (const c of kids) emit(c, depth + 1);
    }
    for (let i = 0; i < bones.length; i++) {
      const sk = skel[i];
      const p = sk ? (sk.parent | 0) : -1;
      if (p < 0) emit(i, 0);
    }
    return out.join("");
  }

  function emitBoneTreeRow(idx, depth, bones, skel, hasChildren, forceFlat, isCollapsed) {
    const indent = "&nbsp;".repeat(Math.max(0, depth) * 2);
    const name = boneDisplayName(idx, skel);
    const isActive = idx === state.selectedBoneIdx;
    const isHidden = state.boneTreeHidden.has(idx);
    const cls = `ake-tree-row${isActive ? " active" : ""}${isHidden ? " hidden" : ""}`;
    const kfn = (bones[idx] && bones[idx].kf) ? bones[idx].kf.length : 0;
    const toggleGlyph = forceFlat ? " " : (hasChildren ? (isCollapsed ? "▶" : "▼") : "·");
    return `<div class="${cls}" data-bone-idx="${idx}">
      <span class="ake-tree-toggle" data-act="tree-toggle">${toggleGlyph}</span>
      <span class="ake-tree-eye" data-act="tree-eye" title="${isHidden?"show":"hide"} bone keyframes (visualization only)">${isHidden?"○":"●"}</span>
      <span class="ake-tree-name" data-act="tree-name">${indent}${escapeHtml(name)}</span>
      <span class="ake-tree-kfn">${kfn}</span>
      <span class="ake-tree-idx">#${idx}</span>
    </div>`;
  }

  function boneDisplayName(idx, skel) {
    if (skel && skel[idx] && skel[idx].name) return skel[idx].name;
    // If the rig panel has user-named the bones, lift those names so
    // the editor's search filter sees the same labels.
    const rigState = window.psoRigPanelState;
    if (rigState && rigState.boneNames && rigState.boneNames.get) {
      const n = rigState.boneNames.get(idx);
      if (n) return n;
    }
    return `bone${idx}`;
  }

  function refreshBoneTreeHtml() {
    if (!state.bodyEl || !state.motion) return;
    const host = state.bodyEl.querySelector('[data-region="boneTree"]');
    if (!host) return;
    const bones = state.motion.bones || [];
    host.innerHTML = buildBoneTreeHtml(bones);
  }

  // Brief one-line summary of the current keyframe selection.
  function selectionSummary() {
    if (state.marqueeMode === "all") return multiSelectionSummary();
    const n = state.selectedKfSet.size;
    if (n === 0) return "";
    if (n === 1) return "1 kf selected";
    return `${n} kfs selected`;
  }

  // v4 / Task 3 — counts the cross-bone selection. The label includes
  // the number of distinct bones touched so users get feedback about
  // marquee scope.
  function multiSelectionSummary() {
    const n = state.selectedKfSetMulti.size;
    if (n === 0) return "";
    const bones = new Set();
    for (const k of state.selectedKfSetMulti) {
      const colon = k.indexOf(":");
      if (colon > 0) bones.add(k.slice(0, colon));
    }
    return `${n} kf${n===1?"":"s"} across ${bones.size} bone${bones.size===1?"":"s"}`;
  }

  // ----- v4 / Task 3 — selection helpers --------------------------------
  // True when at least one selection (single or multi) is non-empty.
  function _hasSelection() {
    if (state.marqueeMode === "all") return state.selectedKfSetMulti.size > 0;
    return state.selectedKfSet.size > 0;
  }
  // Iterate selected kfs as { boneIdx, kfIdx } objects, regardless of
  // mode. Avoid call-site copy/paste of the parsing logic.
  function _iterSelected() {
    const out = [];
    if (state.marqueeMode === "all") {
      for (const k of state.selectedKfSetMulti) {
        const colon = k.indexOf(":");
        if (colon > 0) {
          out.push({ boneIdx: +k.slice(0, colon) | 0, kfIdx: +k.slice(colon + 1) | 0 });
        }
      }
    } else {
      for (const k of state.selectedKfSet) {
        out.push({ boneIdx: state.selectedBoneIdx | 0, kfIdx: +k | 0 });
      }
    }
    return out;
  }
  function _clearAllSelections() {
    state.selectedKfSet.clear();
    state.selectedKfSetMulti.clear();
  }

  // Per-bone TRS inspector. Uses the keyframe at state.selectedFrame
  // (exact match) when one exists; otherwise interpolates between
  // neighbouring keyframes for read-only display.
  function renderBoneInspector(bone) {
    if (!bone) {
      return `<div class="ake-empty">no bone</div>`;
    }
    const kfs = bone.kf || [];
    let kfIdx = -1;
    for (let i = 0; i < kfs.length; i++) {
      if ((kfs[i].t | 0) === state.selectedFrame) { kfIdx = i; break; }
    }
    state.selectedKeyframeIdx = kfIdx;
    const interpolated = kfIdx < 0 ? sampleBone(bone, state.selectedFrame) : null;
    const kf = kfIdx >= 0 ? kfs[kfIdx] : interpolated;
    const editable = kfIdx >= 0;
    const present = bone.present | 0;
    const hasPos = !!(present & NJD_POS);
    const hasAng = !!(present & NJD_ANG);
    const hasScl = !!(present & NJD_SCL);
    function num(v) { return Number.isFinite(+v) ? (+v).toFixed(3) : "0.000"; }
    function bdeg(v) {
      const d = bamsToDeg(v|0);
      return `${(v|0)} <span class="ake-axis-readout">(${d.toFixed(2)}°)</span>`;
    }
    return `
      <div class="ake-channel-toggles">
        <label><input type="checkbox" data-mask="pos" ${hasPos?"checked":""}> POS</label>
        <label><input type="checkbox" data-mask="ang" ${hasAng?"checked":""}> ANG</label>
        <label><input type="checkbox" data-mask="scl" ${hasScl?"checked":""}> SCL</label>
        <span class="ake-axis-readout" style="margin-left:auto">
          ${editable ? "editing keyframe" : "interpolated (no kf at this frame)"}
        </span>
      </div>
      <div class="ake-inspector">
        ${["x","y","z"].map((ax, ai) => `
          <div class="ake-axis-block">
            <label>POS ${ax.toUpperCase()}</label>
            <input type="range" min="-2000" max="2000" step="0.1"
                   value="${num(kf["t" + ax])}" data-trs="t" data-axis="${ai}"
                   ${(!editable || !hasPos)?"disabled":""}>
            <input type="number" step="0.1" value="${num(kf["t" + ax])}"
                   data-trs="t" data-axis="${ai}"
                   ${(!editable || !hasPos)?"disabled":""}>
          </div>
        `).join("")}
      </div>
      <div class="ake-inspector">
        ${["x","y","z"].map((ax, ai) => `
          <div class="ake-axis-block">
            <label>ANG ${ax.toUpperCase()} ${bdeg(kf["r" + ax])}</label>
            <input type="range" min="-32768" max="32767" step="1"
                   value="${kf["r" + ax]|0}" data-trs="r" data-axis="${ai}"
                   ${(!editable || !hasAng)?"disabled":""}>
            <input type="number" step="1" value="${kf["r" + ax]|0}"
                   data-trs="r" data-axis="${ai}"
                   ${(!editable || !hasAng)?"disabled":""}>
          </div>
        `).join("")}
      </div>
      <div class="ake-inspector">
        ${["x","y","z"].map((ax, ai) => `
          <div class="ake-axis-block">
            <label>SCL ${ax.toUpperCase()}</label>
            <input type="range" min="0.1" max="10.0" step="0.01"
                   value="${num(kf["s" + ax])}" data-trs="s" data-axis="${ai}"
                   ${(!editable || !hasScl)?"disabled":""}>
            <input type="number" step="0.01" value="${num(kf["s" + ax])}"
                   data-trs="s" data-axis="${ai}"
                   ${(!editable || !hasScl)?"disabled":""}>
          </div>
        `).join("")}
      </div>
    `;
  }

  function renderKeyframeList(bone) {
    // v4 / Task 3 — in all-bones mode the list shows the multi-selection
    // (keyframes from ANY bone), each row tagged with its bone-id. The
    // single-bone path is unchanged.
    if (state.marqueeMode === "all") {
      const m = state.motion;
      if (!m) return `<div class="ake-empty">no motion</div>`;
      const sels = [...state.selectedKfSetMulti].map((k) => {
        const colon = k.indexOf(":");
        return { bi: +k.slice(0, colon) | 0, kfIdx: +k.slice(colon + 1) | 0, key: k };
      });
      if (sels.length === 0) {
        return `<div class="ake-empty">marquee select keyframes on the timeline to populate this list</div>`;
      }
      // Sort by bone, then by frame.
      sels.sort((a, b) => {
        if (a.bi !== b.bi) return a.bi - b.bi;
        const ka = m.bones[a.bi] && m.bones[a.bi].kf && m.bones[a.bi].kf[a.kfIdx];
        const kb = m.bones[b.bi] && m.bones[b.bi].kf && m.bones[b.bi].kf[b.kfIdx];
        return ((ka && ka.t) | 0) - ((kb && kb.t) | 0);
      });
      return sels.map((sel) => {
        const b = m.bones[sel.bi];
        if (!b || !b.kf) return "";
        const kf = b.kf[sel.kfIdx];
        if (!kf) return "";
        const active = (kf.t | 0) === state.selectedFrame ? "active" : "";
        return `<div class="ake-keyframe-row ${active} selected"
                     data-kf-idx="${sel.kfIdx}" data-bone-idx="${sel.bi}">
          <span class="bone-tag" title="bone idx">b${sel.bi}</span>
          <span class="num">${kf.t}</span>
          <span style="color:#ffaa00">r=(${kf.rx|0},${kf.ry|0},${kf.rz|0})</span>
        </div>`;
      }).join("") || `<div class="ake-empty">selection went stale, marquee again</div>`;
    }
    if (!bone) return `<div class="ake-empty">no bone</div>`;
    const kfs = bone.kf || [];
    if (kfs.length === 0) return `<div class="ake-empty">no keyframes</div>`;
    return kfs.map((kf, i) => {
      const active = (kf.t | 0) === state.selectedFrame ? "active" : "";
      const sel = state.selectedKfSet.has(String(i)) ? "selected" : "";
      const tx = kf.tx, ty = kf.ty, tz = kf.tz;
      const rx = kf.rx, ry = kf.ry, rz = kf.rz;
      return `<div class="ake-keyframe-row ${active} ${sel}" data-kf-idx="${i}">
        <span class="num">${kf.t}</span>
        <span style="color:#56c8c8">t=(${(tx||0).toFixed(1)},${(ty||0).toFixed(1)},${(tz||0).toFixed(1)})</span>
        <span style="color:#ffaa00">r=(${rx|0},${ry|0},${rz|0})</span>
      </div>`;
    }).join("");
  }

  // ------------------------------------------------------------------
  // Curve editor block — only shown when state.curveOpen. Shows a
  // bezier-handle interface for the current bone's keyframes on the
  // selected scalar channel. Note: PSOBB plays motions with linear
  // interp regardless of curve shape — see densifyBezierToLinear().
  // ------------------------------------------------------------------
  function renderCurveEditorBlock(bone) {
    if (!bone) return `<div class="ake-empty">no bone</div>`;
    const present = bone.present | 0;
    // Filter channel options to ones the bone actually authors.
    const opts = CURVE_CHANNELS
      .filter((c) => (present & c.kind) !== 0)
      .map((c) => `<option value="${c.key}" ${c.key===state.curveChannelKey?"selected":""}>${c.label}</option>`)
      .join("");
    const cur = CURVE_CHANNELS.find((c) => c.key === state.curveChannelKey)
      || { label: "?", trs: "r" };
    const yLabel = cur.trs === "t" ? "value (world units)"
                 : cur.trs === "r" ? "value (BAMS, signed)"
                 : "value (×scale)";
    const modes = [
      { v: "single",  label: "single",  title: "plot one channel" },
      { v: "triplet", label: "triplet", title: "overlay X/Y/Z of the active TRS bucket" },
      { v: "all",     label: "all",     title: "overlay all 9 TRS channels" },
    ];
    const modeBtns = modes.map((m) =>
      `<button class="ake-btn ${state.curveMode===m.v?"on":""}"
               data-act="curve-mode" data-mode="${m.v}"
               title="${escapeHtml(m.title)}">${m.label}</button>`
    ).join("");
    // Build the legend for the active set of channels (Task 1).
    const activeChannels = _activeCurveChannels(bone);
    const legendHtml = activeChannels.map((c) => {
      const isActive = c.key === state.curveChannelKey;
      return `<span class="ake-curve-legend-item${isActive?" active":""}"
                    data-act="curve-legend" data-channel="${c.key}"
                    title="${escapeHtml(c.label)} — click to make active for handle editing">
                <span class="ake-curve-legend-swatch" style="background:${c.color}"></span>
                ${escapeHtml(c.label)}
              </span>`;
    }).join("");
    return `
      <div class="ake-curve-toolbar">
        <label>Channels:</label>
        <span class="ake-curve-overlay-modes">${modeBtns}</span>
        <label>Active:
          <select data-region="curveChannel">${opts || `<option>(no channels on bone)</option>`}</select>
        </label>
        <span style="flex:1"></span>
        <label>bake stride:
          <input type="number" min="1" max="60" step="1" style="width:48px"
                 data-region="bakeStride" value="${state.curveBakeStride|0}">
        </label>
        <button class="ake-btn" data-act="curve-reset" title="reset all bezier handles on this bone+channel">reset handles</button>
      </div>
      <div class="ake-curve-legend">
        ${legendHtml}
        <span style="flex:1"></span>
        <span style="color:#6c7785">x = frame · active y = ${escapeHtml(yLabel)}</span>
      </div>
      <div class="ake-curve-host" data-region="curveHost">
        <canvas data-region="curveCanvas" height="${state.curveMode==="all"?CURVE_CANVAS_HEIGHT_ALL:CURVE_CANVAS_HEIGHT}"></canvas>
      </div>
      <div class="ake-curve-note">
        Note: PSOBB plays motions with linear interpolation regardless of
        curve shape. Save bakes bezier curves into dense linear keyframes
        (one kf per <b>bake stride</b> frames).
      </div>
    `;
  }

  // Channels currently visible on the curve canvas — driven by curveMode.
  // - single:  the active channel only
  // - triplet: X/Y/Z of the active TRS bucket (same .trs key as active)
  // - all:     every channel the bone authors
  function _activeCurveChannels(bone) {
    if (!bone) return [];
    const present = bone.present | 0;
    const cur = CURVE_CHANNELS.find((c) => c.key === state.curveChannelKey);
    const mode = state.curveMode;
    if (mode === "all") {
      return CURVE_CHANNELS.filter((c) => (present & c.kind) !== 0);
    }
    if (mode === "triplet" && cur) {
      return CURVE_CHANNELS.filter((c) => c.trs === cur.trs && (present & c.kind) !== 0);
    }
    // single mode (or fallback when active channel isn't authored).
    if (cur && (present & cur.kind)) return [cur];
    return [];
  }

  // ------------------------------------------------------------------
  // Timeline canvas — draws ticks + per-bone keyframe lanes
  // ------------------------------------------------------------------
  function drawTimelineCanvas() {
    if (!state.bodyEl || !state.motion) return;
    const canvas = state.bodyEl.querySelector('[data-region="timelineCanvas"]');
    if (!canvas) return;
    const host = state.bodyEl.querySelector('[data-region="timelineHost"]');
    const dpr = window.devicePixelRatio || 1;
    const cssW = host ? Math.max(50, host.clientWidth) : 200;
    const m = state.motion;
    const boneCount = (m.bones || []).length;
    // v4 / Task 3 — taller canvas in all-bones mode so each bone gets
    // its own row. Single-bone mode keeps the legacy 80px layout.
    const cssH = state.marqueeMode === "all"
      ? Math.max(80, Math.min(180, 24 + boneCount * 4))
      : 80;
    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
    canvas.style.width = cssW + "px";
    canvas.style.height = cssH + "px";
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    const frameCount = Math.max(1, m.frame_count | 0);
    const w = cssW;
    const xPerFrame = w / Math.max(1, frameCount - 1);
    // Background.
    ctx.fillStyle = "#0a0e13";
    ctx.fillRect(0, 0, w, cssH);
    // Frame ticks (every 5/10).
    for (let f = 0; f < frameCount; f++) {
      if (f % 10 === 0) {
        ctx.strokeStyle = "#1f2730";
        ctx.beginPath();
        ctx.moveTo(f * xPerFrame, 0);
        ctx.lineTo(f * xPerFrame, cssH);
        ctx.stroke();
      } else if (f % 5 === 0) {
        ctx.fillStyle = "#1a1f26";
        ctx.fillRect(f * xPerFrame, cssH - 4, 1, 4);
      }
    }
    // v4 / Task 3 — multi-bone render path (all-bones mode).
    if (state.marqueeMode === "all") {
      _drawTimelineAllBones(ctx, m, w, cssH, xPerFrame);
      // Cache for hit-tests.
      state._timelineXPerFrame = xPerFrame;
      state._timelineWidth = cssW;
      state._timelineHeight = cssH;
      // Marquee box.
      const mq = state.marquee;
      if (mq && mq.startX != null && mq.currentX != null) {
        const x0 = Math.min(mq.startX, mq.currentX);
        const x1 = Math.max(mq.startX, mq.currentX);
        ctx.fillStyle = "rgba(255, 200, 80, 0.10)";
        ctx.strokeStyle = "rgba(255, 200, 80, 0.55)";
        ctx.lineWidth = 1;
        ctx.fillRect(x0, 0, x1 - x0, cssH);
        ctx.strokeRect(x0 + 0.5, 0.5, x1 - x0 - 1, cssH - 1);
      }
      // Scrubber line on top.
      const sx = state.selectedFrame * xPerFrame;
      ctx.strokeStyle = "#00ffff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(sx, 0);
      ctx.lineTo(sx, cssH);
      ctx.stroke();
      return;
    }
    // ----- Single-bone render path (legacy) -----------------
    const bone = (m.bones || [])[state.selectedBoneIdx];
    const lanesY = { pos: 12, ang: 36, scl: 60 };
    const lanesH = 18;
    Object.keys(lanesY).forEach((kind) => {
      ctx.fillStyle = "rgba(0,0,0,0.20)";
      ctx.fillRect(0, lanesY[kind], w, lanesH);
      ctx.fillStyle = "#6c7785";
      ctx.font = "9px sans-serif";
      ctx.textBaseline = "top";
      ctx.fillText(kind.toUpperCase(), 4, lanesY[kind] + 1);
    });
    if (bone && bone.kf) {
      // Compare lane: render the original's keyframes as faint ticks
      // beneath the active ones.
      if (state.showCompare && state.motionOriginal) {
        const orig = (state.motionOriginal.bones || [])[state.selectedBoneIdx];
        if (orig && orig.kf) {
          ctx.fillStyle = "rgba(255,255,255,0.10)";
          for (const kf of orig.kf) {
            const x = (kf.t | 0) * xPerFrame;
            ctx.fillRect(x - 1, lanesY.pos, 2, lanesH * 3 + 4);
          }
        }
      }
      const present = bone.present | 0;
      // Triangles for POS, circles for ANG, squares for SCL.
      // We don't store per-channel-per-keyframe presence, but the
      // bone's `present` mask + the kf object having the channel's
      // fields (always true after our merge) is enough. We mark a
      // keyframe as present in a lane iff the bone's present bit for
      // that lane is set.
      for (let kfIdx = 0; kfIdx < bone.kf.length; kfIdx++) {
        const kf = bone.kf[kfIdx];
        const x = (kf.t | 0) * xPerFrame;
        const isSel = state.selectedKfSet.has(String(kfIdx));
        if (isSel) {
          // Selection halo across all three lanes for this keyframe.
          ctx.fillStyle = "rgba(255, 200, 80, 0.25)";
          ctx.fillRect(x - 4, lanesY.pos - 2, 8, lanesH * 3 + 6);
          ctx.strokeStyle = "#ffd060";
          ctx.lineWidth = 1;
          ctx.strokeRect(x - 4, lanesY.pos - 2, 8, lanesH * 3 + 6);
        }
        if (present & NJD_POS) {
          drawTriangle(ctx, x, lanesY.pos + lanesH/2, 5, isSel ? "#ffd060" : "#56c8c8");
        }
        if (present & NJD_ANG) {
          drawCircle(ctx, x, lanesY.ang + lanesH/2, 4, isSel ? "#ffd060" : "#ffaa00");
        }
        if (present & NJD_SCL) {
          drawSquare(ctx, x, lanesY.scl + lanesH/2, 4, isSel ? "#ffd060" : "#a366ff");
        }
      }
    }
    // Scrubber line.
    const sx = state.selectedFrame * xPerFrame;
    ctx.strokeStyle = "#00ffff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(sx, 0);
    ctx.lineTo(sx, cssH);
    ctx.stroke();
    // Cache the px-per-frame so hit-tests can use the same value.
    state._timelineXPerFrame = xPerFrame;
    state._timelineWidth = cssW;
    state._timelineHeight = cssH;
    // Marquee selection box.
    const mq = state.marquee;
    if (mq && mq.startX != null && mq.currentX != null) {
      const x0 = Math.min(mq.startX, mq.currentX);
      const x1 = Math.max(mq.startX, mq.currentX);
      ctx.fillStyle = "rgba(255, 200, 80, 0.10)";
      ctx.strokeStyle = "rgba(255, 200, 80, 0.55)";
      ctx.lineWidth = 1;
      ctx.fillRect(x0, 0, x1 - x0, cssH);
      ctx.strokeRect(x0 + 0.5, 0.5, x1 - x0 - 1, cssH - 1);
    }
  }

  // v4 / Task 3 — draw the multi-bone timeline. Each bone gets a
  // small horizontal lane; the kf marker color encodes the bone (cycles
  // through a palette). Selected kfs get the same yellow halo as the
  // single-bone view. Bone-id is displayed at the left edge of every
  // ~8th lane so users can tell which row is which without crowding.
  function _drawTimelineAllBones(ctx, m, w, cssH, xPerFrame) {
    const bones = m.bones || [];
    if (bones.length === 0) return;
    // Compute per-bone lane y. Reserve 6px top + 4px bottom for tick
    // strip, then distribute the rest.
    const top = 4, bot = 6;
    const usable = Math.max(8, cssH - top - bot);
    const laneH = Math.max(2, usable / bones.length);
    // Palette — same hues as the curve overlay so users can spot
    // patterns. Cycle when bone count exceeds palette length.
    const PALETTE = ["#56c8c8", "#ffaa00", "#a366ff", "#56e060", "#ff5060", "#5090ff", "#ffd060"];
    ctx.font = "8px sans-serif";
    ctx.textBaseline = "top";
    for (let bi = 0; bi < bones.length; bi++) {
      const b = bones[bi];
      const y = top + bi * laneH;
      const color = PALETTE[bi % PALETTE.length];
      // Lane background — alternate stripes so neighbouring rows are
      // distinguishable.
      if (bi % 2 === 0) {
        ctx.fillStyle = "rgba(255,255,255,0.02)";
        ctx.fillRect(0, y, w, laneH);
      }
      // Bone-id label every 8 lanes (or always, if lanes are tall).
      if (laneH >= 6 || bi % 8 === 0) {
        ctx.fillStyle = bi === state.selectedBoneIdx ? "#00ffff" : "#6c7785";
        ctx.fillText(`#${bi}`, 2, y);
      }
      // Highlight the active bone's lane.
      if (bi === state.selectedBoneIdx) {
        ctx.fillStyle = "rgba(0, 255, 255, 0.06)";
        ctx.fillRect(0, y, w, laneH);
      }
      if (!b || !b.kf) continue;
      // Render keyframes as small dots; selected ones get a halo.
      const cy = y + laneH / 2;
      const dotR = Math.max(1.5, Math.min(3, laneH * 0.35));
      for (let i = 0; i < b.kf.length; i++) {
        const kf = b.kf[i];
        const x = (kf.t | 0) * xPerFrame;
        const isSel = state.selectedKfSetMulti.has(`${bi}:${i}`);
        if (isSel) {
          ctx.fillStyle = "rgba(255, 200, 80, 0.30)";
          ctx.fillRect(x - dotR - 1, y, (dotR + 1) * 2, laneH);
        }
        ctx.fillStyle = isSel ? "#ffd060" : color;
        ctx.beginPath();
        ctx.arc(x, cy, dotR, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  function drawTriangle(ctx, x, y, r, color) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(x, y - r);
    ctx.lineTo(x + r, y + r);
    ctx.lineTo(x - r, y + r);
    ctx.closePath();
    ctx.fill();
  }
  function drawCircle(ctx, x, y, r, color) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }
  function drawSquare(ctx, x, y, r, color) {
    ctx.fillStyle = color;
    ctx.fillRect(x - r, y - r, r * 2, r * 2);
  }

  // ------------------------------------------------------------------
  // Linear-interpolated read of a bone's TRS at frame `t`.
  // Used by the inspector when the scrubber is between keyframes.
  // ------------------------------------------------------------------
  function sampleBone(bone, t) {
    const kfs = bone.kf || [];
    const empty = {
      t, tx: 0, ty: 0, tz: 0, rx: 0, ry: 0, rz: 0, sx: 1, sy: 1, sz: 1,
    };
    if (kfs.length === 0) return empty;
    if (t <= kfs[0].t) return Object.assign({}, kfs[0], { t });
    if (t >= kfs[kfs.length - 1].t) return Object.assign({}, kfs[kfs.length - 1], { t });
    for (let i = 0; i < kfs.length - 1; i++) {
      const a = kfs[i], b = kfs[i + 1];
      if (t >= a.t && t <= b.t) {
        const span = Math.max(1, b.t - a.t);
        const f = (t - a.t) / span;
        return {
          t,
          tx: a.tx + (b.tx - a.tx) * f,
          ty: a.ty + (b.ty - a.ty) * f,
          tz: a.tz + (b.tz - a.tz) * f,
          // BAMS interpolation — wrap across the 0xFFFF/0 boundary
          // by picking the shortest path. The renderer does the
          // same trick (see model_viewer.js _sampleBoneTrack).
          rx: lerpBams(a.rx, b.rx, f),
          ry: lerpBams(a.ry, b.ry, f),
          rz: lerpBams(a.rz, b.rz, f),
          sx: a.sx + (b.sx - a.sx) * f,
          sy: a.sy + (b.sy - a.sy) * f,
          sz: a.sz + (b.sz - a.sz) * f,
        };
      }
    }
    return empty;
  }

  function lerpBams(a, b, f) {
    let d = (b | 0) - (a | 0);
    if (d > 0x8000) d -= 0x10000;
    if (d < -0x8000) d += 0x10000;
    let v = ((a | 0) + d * f) | 0;
    while (v < -0x8000) v += 0x10000;
    while (v > 0x7FFF) v -= 0x10000;
    return v;
  }

  // v4 / Task 1 — linear-interpolate a single scalar channel at frame
  // `t` (which may be fractional). Used by the curve canvas's click-to-
  // pick hit test: find which channel's interpolated curve passes
  // closest to the click point.
  function _interpAtFrame(bone, channel, t) {
    const kfs = (bone && bone.kf) || [];
    if (kfs.length === 0) return null;
    if (t <= kfs[0].t) return _readChannel(kfs[0], channel);
    if (t >= kfs[kfs.length - 1].t) return _readChannel(kfs[kfs.length - 1], channel);
    for (let i = 0; i < kfs.length - 1; i++) {
      const a = kfs[i], b = kfs[i + 1];
      if (t >= a.t && t <= b.t) {
        const span = Math.max(1, b.t - a.t);
        const f = (t - a.t) / span;
        if (channel === "rx" || channel === "ry" || channel === "rz") {
          return lerpBams(_readChannel(a, channel), _readChannel(b, channel), f);
        }
        const av = _readChannel(a, channel), bv = _readChannel(b, channel);
        return av + (bv - av) * f;
      }
    }
    return _readChannel(kfs[kfs.length - 1], channel);
  }

  // ------------------------------------------------------------------
  // Task 4 helpers — release ALL bone overrides this panel set via the
  // eye-toggle. Called on motion change + model change. Doesn't touch
  // overrides authored by other panels (e.g. rig_panel) because we only
  // know about ones we put into state.boneTreeHidden ourselves.
  // ------------------------------------------------------------------
  function releaseAllPanelBoneOverrides() {
    if (!state.boneTreeHidden || state.boneTreeHidden.size === 0) return;
    if (typeof window.psoSetBonePoseOverride !== "function") return;
    for (const bIdx of state.boneTreeHidden) {
      try { window.psoSetBonePoseOverride(bIdx, null); } catch (_e) {}
    }
    if (typeof window.psoSeekAnimationToFrame === "function") {
      try { window.psoSeekAnimationToFrame(state.selectedFrame || 0); } catch (_e) {}
    } else if (typeof window.psoApplyRigBake === "function") {
      try { window.psoApplyRigBake(); } catch (_e) {}
    }
  }

  // ------------------------------------------------------------------
  // Event handlers
  // ------------------------------------------------------------------
  function onPanelClick(ev) {
    const t = ev.target;
    const act = t.dataset && t.dataset.act;
    // Bone tree row click — name, eye toggle, or expand/collapse.
    const treeRow = t.closest && t.closest("[data-bone-idx]");
    if (treeRow && treeRow.classList.contains("ake-tree-row")) {
      const bIdx = +treeRow.dataset.boneIdx;
      const treeAct = t.dataset && t.dataset.act;
      if (treeAct === "tree-eye") {
        ev.stopPropagation();
        // v4 / Task 4 — wire eye-toggle to bone-pose override.
        // Toggle: if hidden, restore animation; otherwise force bind pose.
        const wasHidden = state.boneTreeHidden.has(bIdx);
        if (wasHidden) {
          state.boneTreeHidden.delete(bIdx);
          // Release: pass null pose to delete the override (see
          // psoSetBonePoseOverride contract in model_viewer.js).
          if (typeof window.psoSetBonePoseOverride === "function") {
            try { window.psoSetBonePoseOverride(bIdx, null); } catch (_e) {}
          }
        } else {
          state.boneTreeHidden.add(bIdx);
          // Push the bind-pose TRS so the renderer freezes this bone.
          // The skeleton snapshot is the source of truth — bone fields
          // mirror the runtime's bind values (BAMS rotation, world-units
          // position, unit scale). When bind metadata is missing we fall
          // back to identity, which still mutes the keyframes.
          const sk = (state.skeleton || [])[bIdx];
          const bindPose = {
            position: sk && sk.position ? [sk.position[0], sk.position[1], sk.position[2]] : [0, 0, 0],
            rotation_bams: sk && sk.rotation_bams ? [sk.rotation_bams[0]|0, sk.rotation_bams[1]|0, sk.rotation_bams[2]|0] : [0, 0, 0],
            scale: sk && sk.scale ? [sk.scale[0], sk.scale[1], sk.scale[2]] : [1, 1, 1],
          };
          if (typeof window.psoSetBonePoseOverride === "function") {
            try { window.psoSetBonePoseOverride(bIdx, bindPose); } catch (_e) {}
          }
        }
        // Force a re-bake at the current frame so the change is visible
        // even when playback is paused.
        if (typeof window.psoSeekAnimationToFrame === "function") {
          try { window.psoSeekAnimationToFrame(state.selectedFrame); } catch (_e) {}
        } else if (typeof window.psoApplyRigBake === "function") {
          try { window.psoApplyRigBake(); } catch (_e) {}
        }
        refreshBoneTreeHtml();
        return;
      }
      if (treeAct === "tree-toggle") {
        ev.stopPropagation();
        if (state.boneTreeCollapsed.has(bIdx)) state.boneTreeCollapsed.delete(bIdx);
        else state.boneTreeCollapsed.add(bIdx);
        refreshBoneTreeHtml();
        return;
      }
      // tree-name / row body → select the bone.
      state.selectedBoneIdx = bIdx;
      // Switching bones blanks the per-bone keyframe selection.
      state.selectedKfSet.clear();
      refreshEditor();
      return;
    }
    // Keyframe-row click in the side list.
    const kfIdxNode = t.closest && t.closest("[data-kf-idx]");
    if (kfIdxNode && kfIdxNode.classList.contains("ake-keyframe-row")) {
      const idx = +kfIdxNode.dataset.kfIdx;
      // v4 / Task 3 — in all-bones mode the row carries data-bone-idx;
      // honor it so users can click a row from another bone.
      const rowBoneIdx = kfIdxNode.dataset.boneIdx != null
        ? +kfIdxNode.dataset.boneIdx
        : state.selectedBoneIdx;
      const bone = state.motion && state.motion.bones[rowBoneIdx];
      if (bone && bone.kf && bone.kf[idx]) {
        if (state.marqueeMode === "all") {
          const k = `${rowBoneIdx}:${idx}`;
          if (ev.shiftKey) {
            if (state.selectedKfSetMulti.has(k)) state.selectedKfSetMulti.delete(k);
            else state.selectedKfSetMulti.add(k);
          } else {
            state.selectedKfSetMulti.clear();
            state.selectedKfSetMulti.add(k);
          }
          state.selectedBoneIdx = rowBoneIdx | 0;
        } else if (ev.shiftKey) {
          const k = String(idx);
          if (state.selectedKfSet.has(k)) state.selectedKfSet.delete(k);
          else state.selectedKfSet.add(k);
        } else {
          state.selectedKfSet.clear();
          state.selectedKfSet.add(String(idx));
        }
        state.selectedFrame = bone.kf[idx].t | 0;
        // Sync 3D model to the new scrubber position.
        seekModelToFrame(state.selectedFrame, true);
        refreshEditor();
      }
      return;
    }
    if (!act) return;
    if (act === "load") return loadMotionFromPicker();
    if (act === "reload") return reloadCurrent();
    if (act === "save") return saveMotion(false);
    if (act === "save-as-new") return saveMotion(true);
    if (act === "swap") return swapIntoBml();
    if (act === "kf-insert") return insertKeyframeHere();
    if (act === "kf-delete") return deleteKeyframeHere();
    if (act === "kf-delete-sel") return deleteSelectedKeyframes();
    if (act === "kf-select-all") return selectAllKeyframes();
    if (act === "kf-deselect") {
      _clearAllSelections();
      refreshEditor();
      return;
    }
    if (act === "bone-reset") return resetBone();
    if (act === "reset-all") return resetAll();
    if (act === "play") return togglePlay();
    if (act === "loop") { state.loop = !state.loop; refreshEditor(); return; }
    if (act === "step-fwd") return stepFrame(+1);
    if (act === "step-back") return stepFrame(-1);
    if (act === "rewind") {
      state.selectedFrame = 0;
      seekModelToFrame(0, true);
      refreshEditor();
      return;
    }
    if (act === "tree-collapse-all") {
      const bones = state.motion && state.motion.bones;
      if (bones) {
        // Collapse every bone that has children — find roots' children
        // via parent map.
        const skel = state.skeleton || [];
        for (let i = 0; i < bones.length; i++) {
          const sk = skel[i];
          if (sk && sk.parent < 0) state.boneTreeCollapsed.add(i);
        }
      }
      refreshBoneTreeHtml();
      return;
    }
    if (act === "tree-expand-all") {
      state.boneTreeCollapsed.clear();
      refreshBoneTreeHtml();
      return;
    }
    if (act === "curve-toggle") {
      state.curveOpen = !state.curveOpen;
      refreshEditor();
      // Drawing happens after layout so the canvas has a real width.
      requestAnimationFrame(drawCurveCanvas);
      return;
    }
    if (act === "curve-reset") return resetBezierHandlesOnChannel();
    // v4 / Task 1 — curve-mode toggle (single | triplet | all).
    if (act === "curve-mode") {
      const mode = t.dataset.mode;
      if (mode === "single" || mode === "triplet" || mode === "all") {
        state.curveMode = mode;
        refreshEditor();
        requestAnimationFrame(drawCurveCanvas);
      }
      return;
    }
    // v4 / Task 3 — marquee-mode toggle (single | all bones).
    if (act === "marquee-mode") {
      const mode = t.dataset.mode;
      if (mode === "single" || mode === "all") {
        state.marqueeMode = mode;
        // Switching mode wipes the OTHER selection set so we don't
        // carry stale state (e.g. switching to "all" from a single-
        // bone selection would otherwise show empty in the multi UI).
        _clearAllSelections();
        refreshEditor();
      }
      return;
    }
    // v4 / Task 1 — legend swatch click → make that channel active.
    const legendNode = t.closest && t.closest("[data-act='curve-legend']");
    if (legendNode) {
      const ch = legendNode.dataset.channel;
      if (ch && CURVE_CHANNELS.find((c) => c.key === ch)) {
        state.curveChannelKey = ch;
        refreshEditor();
        requestAnimationFrame(drawCurveCanvas);
      }
      return;
    }
  }

  function onPanelChange(ev) {
    const t = ev.target;
    if (t.dataset && t.dataset.region === "motionPicker") {
      state.motionName = t.value || "";
    }
    if (t.dataset && t.dataset.region === "fpsSel") {
      state.fps = parseInt(t.value, 10) || 30;
      refreshEditor();
    }
    if (t.dataset && t.dataset.region === "compare") {
      state.showCompare = !!t.checked;
      drawTimelineCanvas();
    }
    if (t.dataset && t.dataset.region === "bonePick") {
      state.selectedBoneIdx = parseInt(t.value, 10) | 0;
      state.selectedKfSet.clear();
      refreshEditor();
    }
    if (t.dataset && t.dataset.region === "saveName") {
      state.saveAsName = t.value || "";
    }
    if (t.dataset && t.dataset.region === "swapTarget") {
      state.swapTargetSlot = t.value || "";
    }
    if (t.dataset && t.dataset.region === "curveChannel") {
      state.curveChannelKey = t.value || "rx";
      drawCurveCanvas();
    }
    if (t.dataset && t.dataset.region === "bakeStride") {
      const v = parseInt(t.value, 10);
      state.curveBakeStride = Number.isFinite(v) ? Math.max(1, Math.min(60, v)) : 1;
    }
    if (t.dataset && t.dataset.mask) {
      togglePresentMask(t.dataset.mask, !!t.checked);
    }
  }

  function onPanelInput(ev) {
    const t = ev.target;
    if (t.dataset && t.dataset.region === "treeSearch") {
      state.boneTreeQuery = t.value || "";
      refreshBoneTreeHtml();
      return;
    }
    if (t.dataset && t.dataset.region === "scrub") {
      const f = parseInt(t.value, 10) | 0;
      state.selectedFrame = f;
      // Quick re-paint without a full re-render.
      const fl = state.bodyEl && state.bodyEl.querySelector('[data-region="frameLabel"]');
      if (fl) fl.textContent = String(f);
      const fn = state.bodyEl && state.bodyEl.querySelector('[data-region="frameNum"]');
      if (fn && fn !== t) fn.value = String(f);
      drawTimelineCanvas();
      // ----- Task 1: scrubber → 3D pose live sync -----
      // Pause playback the first time we see a drag-input. Pauses BOTH
      // the panel's tickPlayback loop (which advances state.selectedFrame)
      // AND the model viewer's state.anim.playing flag (which advances
      // its own clock for bone re-bakes).
      if (typeof window.psoGetAnimationPlaying === "function" &&
          state.seekRafPending == null && window.psoGetAnimationPlaying()) {
        state.seekWasPlaying = true;
      }
      if (state.playing) {
        // Pause the panel's local playback loop too — otherwise it
        // would fight the scrubber. Stays paused on release per spec.
        state.playing = false;
        if (state.playbackTimer) {
          cancelAnimationFrame(state.playbackTimer);
          state.playbackTimer = null;
        }
      }
      if (typeof window.psoSetAnimationPlaying === "function") {
        try { window.psoSetAnimationPlaying(false); } catch (_e) {}
      }
      // Throttle the seek call to 60Hz via rAF so high-frequency drag
      // events don't flood the viewport's bone re-bake.
      seekModelToFrame(f, /*immediate=*/false);
      // Debounce inspector refresh — every input event is too much.
      if (state.inspectorPending) clearTimeout(state.inspectorPending);
      state.inspectorPending = setTimeout(() => {
        state.inspectorPending = null;
        refreshEditor();
      }, 60);
      return;
    }
    if (t.dataset && t.dataset.region === "frameNum") {
      const v = parseInt(t.value, 10);
      if (Number.isFinite(v)) {
        state.selectedFrame = v | 0;
        seekModelToFrame(state.selectedFrame, true);
        refreshEditor();
      }
      return;
    }
    if (t.dataset && t.dataset.trs) {
      onTrsInput(t);
    }
  }

  // -------------------------------------------------------------------
  // Task 1 — scrubber → 3D pose live sync (rAF-throttled).
  // -------------------------------------------------------------------
  // Schedule a psoSeekAnimationToFrame() call for the *current* pending
  // frame on the next animation-frame tick. If a frame number is
  // already pending, we just overwrite it — only the latest value gets
  // sent. This collapses bursts of drag events down to a single seek
  // per displayed frame.
  function seekModelToFrame(frame, immediate) {
    state.seekRafPending = frame | 0;
    if (immediate) {
      flushSeek();
      return;
    }
    if (state.seekRafScheduled) return;
    state.seekRafScheduled = true;
    requestAnimationFrame(flushSeek);
  }
  function flushSeek() {
    state.seekRafScheduled = false;
    const f = state.seekRafPending;
    state.seekRafPending = null;
    if (f == null) return;
    if (typeof window.psoSeekAnimationToFrame === "function") {
      try { window.psoSeekAnimationToFrame(f); } catch (_e) {}
    }
  }

  function onTrsInput(input) {
    const m = state.motion;
    if (!m) return;
    const bone = m.bones[state.selectedBoneIdx];
    if (!bone || !bone.kf) return;
    const trs = input.dataset.trs;            // "t" | "r" | "s"
    const axis = +input.dataset.axis | 0;     // 0/1/2
    if (axis < 0 || axis > 2) return;
    const value = trs === "r" ? (parseInt(input.value, 10) | 0)
                              : parseFloat(input.value);
    if (!Number.isFinite(value)) return;
    // Find the keyframe at the current frame.
    const kfIdx = bone.kf.findIndex((k) => (k.t | 0) === state.selectedFrame);
    if (kfIdx < 0) return; // can't edit interpolated frames; user must insert
    const kf = bone.kf[kfIdx];
    const axisLetter = ["x", "y", "z"][axis];
    kf[trs + axisLetter] = value;
    // Mark this channel as authored at this keyframe — needed so the
    // server's _ake_motion_from_json round-trip emits the right per-track
    // count. Without `chan`, an edit on a merge-only keyframe (one that
    // exists in the JSON only because a sibling channel had a kf at the
    // same frame) would expand the track count on save.
    const chanBit = trs === "t" ? NJD_POS
                  : trs === "r" ? NJD_ANG
                  : trs === "s" ? NJD_SCL : 0;
    if (chanBit) kf.chan = (kf.chan | 0) | chanBit;
    // Mirror the change to the sibling input (range <-> number).
    const sibSel = `[data-trs="${trs}"][data-axis="${axis}"]`;
    state.bodyEl.querySelectorAll(sibSel).forEach((el) => {
      if (el !== input) el.value = String(value);
    });
    invalidateRoundTrip(m);
    drawTimelineCanvas();
  }

  function togglePresentMask(maskName, on) {
    const m = state.motion;
    if (!m) return;
    const bone = m.bones[state.selectedBoneIdx];
    if (!bone) return;
    const bit = maskName === "pos" ? NJD_POS
              : maskName === "ang" ? NJD_ANG
              : maskName === "scl" ? NJD_SCL : 0;
    if (!bit) return;
    let p = bone.present | 0;
    if (on) p |= bit;
    else p &= ~bit;
    bone.present = p;
    invalidateRoundTrip(m);
    refreshEditor();
  }

  function invalidateRoundTrip(m) {
    if (!m || !m.round_trip) return;
    delete m.round_trip.source_body_b64;
    delete m.round_trip.track_offset_hints;
    delete m.round_trip.trailing_size;
  }

  // ------------------------------------------------------------------
  // Mutations + persistence
  // ------------------------------------------------------------------
  async function loadMotionFromPicker() {
    const sel = state.bodyEl && state.bodyEl.querySelector('[data-region="motionPicker"]');
    const motionName = sel && sel.value;
    if (!motionName) {
      setStatus("err", "no motion selected");
      return;
    }
    const modelPath = getCurrentModelPath();
    if (!modelPath) {
      setStatus("err", "no model loaded");
      return;
    }
    setStatus("busy", `loading ${motionName}…`);
    try {
      const r = await fetch("/api/anim_keyframe/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_path: modelPath, motion_name: motionName }),
      });
      if (!r.ok) {
        const txt = await r.text();
        setStatus("err", `load failed: ${r.status} ${txt.slice(0, 80)}`);
        return;
      }
      const data = await r.json();
      state.motion = data;
      state.motionOriginal = deepClone(data);
      state.motionName = motionName;
      state.modelPath = modelPath;
      state.saveAsName = motionName + ".njm";
      state.selectedFrame = 0;
      state.selectedBoneIdx = 0;
      state.fps = Math.round(data.fps || 30);
      // Discard per-motion editor scratch state.
      state.selectedKfSet.clear();
      state.selectedKfSetMulti.clear();
      state.bezierHandles.clear();
      // v4 / Task 4 — clear the renderer-side overrides we authored from
      // eye-toggles on the previous motion. Without this, switching
      // motions would leave stale bind-pose locks on bones the user
      // muted in the prior session.
      releaseAllPanelBoneOverrides();
      state.boneTreeHidden.clear();
      state.boneTreeCollapsed.clear();
      state.boneTreeQuery = "";
      state.kfClipboard = [];
      // v4 / Task 2 — restore bezier handles from the sidecar payload
      // the server attached to the load response. Each key is a string
      // "<boneIdx>:<kfIdx>:<channelKey>" matching _bezierKey().
      let restored = 0;
      if (data && data.bezier_handles && typeof data.bezier_handles === "object") {
        for (const [k, v] of Object.entries(data.bezier_handles)) {
          if (!v || typeof v !== "object") continue;
          state.bezierHandles.set(k, {
            inDx: +v.inDx || 0,
            inDy: +v.inDy || 0,
            outDx: +v.outDx || 0,
            outDy: +v.outDy || 0,
          });
          restored++;
        }
      }
      const restoredNote = restored ? ` (+${restored} bezier handles)` : "";
      setStatus("ok", `loaded ${motionName}: ${data.bones.length} bones / ${data.frame_count} frames${restoredNote}`);
      // Sync playback — also have the model viewer play this motion.
      if (typeof window.psoLoadMotion === "function") {
        try { await window.psoLoadMotion(motionName); } catch (_) {}
      }
      refreshEditor();
    } catch (e) {
      setStatus("err", `load error: ${e?.message || e}`);
    }
  }

  function reloadCurrent() {
    if (state.motionName) loadMotionFromPicker();
  }

  async function saveMotion(asNew) {
    if (!state.motion) {
      setStatus("err", "no motion loaded");
      return;
    }
    let name = state.saveAsName || (state.motion.name + ".njm");
    if (asNew) {
      const proposed = window.prompt("Save as new motion (filename, .njm):",
        (state.motion.name + "_edit.njm"));
      if (!proposed) return;
      name = proposed;
    }
    if (!name.toLowerCase().endsWith(".njm")) name += ".njm";
    // Task 4: if the user authored bezier handles, densify them into
    // linear keyframes BEFORE encoding so PSOBB's runtime (linear-only)
    // sees the curve shape. Has no effect when no handles exist.
    let payload = state.motion;
    const hadBezier = state.bezierHandles.size > 0;
    if (hadBezier) {
      payload = bakeAllBezierToLinear(state.motion);
    }
    // v4 / Task 2 — serialise the panel's bezier handle map for the
    // sidecar. Even when handles were baked into the .njm, the original
    // (non-baked) handle state must be persisted so the next reload can
    // restore the editable curves.
    const handlesPayload = {};
    for (const [k, v] of state.bezierHandles) {
      handlesPayload[k] = {
        inDx: +v.inDx, inDy: +v.inDy,
        outDx: +v.outDx, outDy: +v.outDy,
      };
    }
    setStatus("busy", `saving ${name}${hadBezier?" (baked bezier)":""}…`);
    try {
      const body = { motion_json: payload, name };
      // Only send the field when the panel has handles — keeps the
      // wire round-trip clean for users not using the curve editor.
      if (state.bezierHandles.size > 0) body.bezier_handles = handlesPayload;
      const r = await fetch("/api/anim_keyframe/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const t = await r.text();
      let data = {};
      try { data = JSON.parse(t); } catch {}
      if (!r.ok) {
        setStatus("err", `save failed: ${r.status} ${(data.detail || t).slice(0, 80)}`);
        return;
      }
      state.saveAsName = name;
      const bakeNote = hadBezier ? ` (bezier→${data.frame_count|0}f linear)` : "";
      setStatus("ok", `saved ${name}${bakeNote} (${data.size}B md5=${data.md5 ? data.md5.slice(0,8) : "?"})`);
    } catch (e) {
      setStatus("err", `save error: ${e?.message || e}`);
    }
  }

  async function swapIntoBml() {
    if (!state.saveAsName) {
      setStatus("err", "save first");
      return;
    }
    const modelPath = state.modelPath || getCurrentModelPath();
    if (!modelPath) {
      setStatus("err", "no model path");
      return;
    }
    // The host BML is the part before "#".
    const hashIdx = modelPath.indexOf("#");
    const targetBml = hashIdx > 0 ? modelPath.slice(0, hashIdx) : modelPath;
    let targetInner = state.swapTargetSlot;
    if (!targetInner) {
      // Default: same name as the loaded motion.
      targetInner = state.motionName + ".njm";
    }
    if (!targetInner.toLowerCase().endsWith(".njm")) targetInner += ".njm";
    setStatus("busy", `swapping into ${targetBml}#${targetInner}…`);
    try {
      const r = await fetch("/api/import/animation/swap", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          njm_path: state.saveAsName,
          target_bml: targetBml,
          target_inner_to_replace: targetInner,
        }),
      });
      const t = await r.text();
      let data = {};
      try { data = JSON.parse(t); } catch {}
      if (!r.ok) {
        setStatus("err", `swap failed: ${r.status} ${(data.detail || t).slice(0, 80)}`);
        return;
      }
      setStatus("ok", `replaced slot — staged ${data.archive_name} (${data.size}B)`);
    } catch (e) {
      setStatus("err", `swap error: ${e?.message || e}`);
    }
  }

  async function insertKeyframeHere() {
    const m = state.motion;
    if (!m) return;
    const bone = m.bones[state.selectedBoneIdx];
    if (!bone) return;
    // Snapshot the CURRENT inspector values (so the new keyframe lands
    // at the user's most recent edits, not at identity). Fall back to
    // the interpolated sample if we're on an empty frame.
    const sample = sampleBone(bone, state.selectedFrame);
    const present = bone.present | 0;
    const body = {
      motion_json: m,
      bone_idx: state.selectedBoneIdx,
      frame_idx: state.selectedFrame,
    };
    if (present & NJD_POS) body.pos = [sample.tx, sample.ty, sample.tz];
    if (present & NJD_ANG) body.ang = [sample.rx | 0, sample.ry | 0, sample.rz | 0];
    if (present & NJD_SCL) body.scl = [sample.sx, sample.sy, sample.sz];
    if ((present & NJD_QUAT) && sample.qw != null) {
      body.quat = [sample.qw, sample.qx, sample.qy, sample.qz];
    }
    setStatus("busy", `inserting kf @ frame ${state.selectedFrame}…`);
    try {
      const r = await fetch("/api/anim_keyframe/insert", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const t = await r.text();
        setStatus("err", `insert failed: ${r.status} ${t.slice(0, 80)}`);
        return;
      }
      state.motion = await r.json();
      setStatus("ok", `inserted kf @ frame ${state.selectedFrame}`);
      refreshEditor();
    } catch (e) {
      setStatus("err", `insert error: ${e?.message || e}`);
    }
  }

  async function deleteKeyframeHere() {
    const m = state.motion;
    if (!m) return;
    setStatus("busy", `deleting kf @ frame ${state.selectedFrame}…`);
    try {
      const r = await fetch("/api/anim_keyframe/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          motion_json: m,
          bone_idx: state.selectedBoneIdx,
          frame_idx: state.selectedFrame,
        }),
      });
      if (!r.ok) {
        const t = await r.text();
        setStatus("err", `delete failed: ${r.status} ${t.slice(0, 80)}`);
        return;
      }
      const data = await r.json();
      state.motion = data.motion_json;
      setStatus("ok", `deleted ${data.removed} kf at frame ${state.selectedFrame}`);
      refreshEditor();
    } catch (e) {
      setStatus("err", `delete error: ${e?.message || e}`);
    }
  }

  function resetBone() {
    if (!state.motion || !state.motionOriginal) return;
    const idx = state.selectedBoneIdx;
    const orig = state.motionOriginal.bones && state.motionOriginal.bones[idx];
    if (!orig) return;
    state.motion.bones[idx] = deepClone(orig);
    invalidateRoundTrip(state.motion);
    setStatus("ok", `reverted bone ${idx} to source`);
    refreshEditor();
  }

  function resetAll() {
    if (!state.motionOriginal) return;
    if (!window.confirm("Discard ALL edits to this motion?")) return;
    state.motion = deepClone(state.motionOriginal);
    setStatus("ok", "reset all to source");
    refreshEditor();
  }

  // ------------------------------------------------------------------
  // Playback
  // ------------------------------------------------------------------
  function togglePlay() {
    state.playing = !state.playing;
    if (state.playing) {
      state.lastPlaybackTimestamp = performance.now();
      tickPlayback();
    } else if (state.playbackTimer) {
      cancelAnimationFrame(state.playbackTimer);
      state.playbackTimer = null;
    }
    refreshEditor();
  }

  function tickPlayback() {
    if (!state.playing || !state.motion) {
      state.playbackTimer = null;
      return;
    }
    const now = performance.now();
    const dt = (now - state.lastPlaybackTimestamp) / 1000.0;
    state.lastPlaybackTimestamp = now;
    const inc = dt * state.fps;
    let nextFrame = state.selectedFrame + inc;
    const max = Math.max(0, state.motion.frame_count - 1);
    if (nextFrame > max) {
      if (state.loop) nextFrame = nextFrame % Math.max(1, max + 1);
      else { nextFrame = max; state.playing = false; }
    }
    state.selectedFrame = nextFrame | 0;
    drawTimelineCanvas();
    const fl = state.bodyEl && state.bodyEl.querySelector('[data-region="frameLabel"]');
    if (fl) fl.textContent = String(state.selectedFrame);
    const scrub = state.bodyEl && state.bodyEl.querySelector('[data-region="scrub"]');
    if (scrub) scrub.value = String(state.selectedFrame);
    if (state.playing) {
      state.playbackTimer = requestAnimationFrame(tickPlayback);
    } else {
      refreshEditor();
    }
  }

  function stepFrame(delta) {
    if (!state.motion) return;
    const max = Math.max(0, state.motion.frame_count - 1);
    state.selectedFrame = Math.min(max, Math.max(0, state.selectedFrame + delta));
    seekModelToFrame(state.selectedFrame, true);
    refreshEditor();
  }

  // ------------------------------------------------------------------
  // Task 3 — Multi-keyframe selection, marquee drag, drag-to-move,
  // copy/paste/delete, keyboard shortcuts.
  // v4 extension: cross-bone selection (state.marqueeMode === "all").
  // ------------------------------------------------------------------
  function selectAllKeyframes() {
    const m = state.motion;
    if (!m) return;
    if (state.marqueeMode === "all") {
      // Select EVERY keyframe across every bone.
      state.selectedKfSetMulti.clear();
      for (let bi = 0; bi < m.bones.length; bi++) {
        const b = m.bones[bi];
        if (!b || !b.kf) continue;
        for (let i = 0; i < b.kf.length; i++) {
          state.selectedKfSetMulti.add(`${bi}:${i}`);
        }
      }
      refreshEditor();
      return;
    }
    const bone = m.bones[state.selectedBoneIdx];
    if (!bone || !bone.kf) return;
    state.selectedKfSet.clear();
    for (let i = 0; i < bone.kf.length; i++) {
      state.selectedKfSet.add(String(i));
    }
    refreshEditor();
  }

  function deleteSelectedKeyframes() {
    const m = state.motion;
    if (!m) return;
    if (state.marqueeMode === "all") {
      // v4 / Task 3 — bucket the selection by bone idx, splice each
      // bone's kf array independently. Sort kfIdxs descending per bone
      // so splicing doesn't invalidate later indices.
      const byBone = new Map();
      for (const sel of _iterSelected()) {
        if (!byBone.has(sel.boneIdx)) byBone.set(sel.boneIdx, []);
        byBone.get(sel.boneIdx).push(sel.kfIdx);
      }
      let total = 0;
      for (const [bi, idxs] of byBone) {
        const b = m.bones[bi];
        if (!b || !b.kf) continue;
        idxs.sort((a, b) => b - a);
        for (const i of idxs) {
          if (i >= 0 && i < b.kf.length) {
            b.kf.splice(i, 1);
            total++;
          }
        }
      }
      state.selectedKfSetMulti.clear();
      invalidateRoundTrip(m);
      setStatus("ok", `deleted ${total} kf across ${byBone.size} bone${byBone.size===1?"":"s"}`);
      refreshEditor();
      return;
    }
    const bone = m.bones[state.selectedBoneIdx];
    if (!bone || !bone.kf || state.selectedKfSet.size === 0) return;
    const idxs = [...state.selectedKfSet].map((s) => +s).sort((a, b) => b - a);
    for (const i of idxs) {
      if (i >= 0 && i < bone.kf.length) bone.kf.splice(i, 1);
    }
    state.selectedKfSet.clear();
    invalidateRoundTrip(m);
    setStatus("ok", `deleted ${idxs.length} keyframe${idxs.length===1?"":"s"}`);
    refreshEditor();
  }

  function copySelectedKeyframes() {
    const m = state.motion;
    if (!m) return;
    if (state.marqueeMode === "all") {
      // v4 / Task 3 — clipboard preserves boneIdx so paste can restore
      // each kf to its origin bone. Stored as { boneIdx, kf } pairs.
      const sels = _iterSelected();
      if (sels.length === 0) return;
      state.kfClipboard = sels.map((sel) => {
        const b = m.bones[sel.boneIdx];
        const kf = b && b.kf ? b.kf[sel.kfIdx] : null;
        return kf ? { boneIdx: sel.boneIdx, kf: Object.assign({}, kf) } : null;
      }).filter((x) => x);
      // Bones touched (for status).
      const bones = new Set(state.kfClipboard.map((c) => c.boneIdx));
      setStatus("ok", `copied ${state.kfClipboard.length} kf across ${bones.size} bone${bones.size===1?"":"s"}`);
      return;
    }
    const bone = m.bones[state.selectedBoneIdx];
    if (!bone || !bone.kf) return;
    const idxs = [...state.selectedKfSet].map((s) => +s).sort((a, b) => a - b);
    if (idxs.length === 0) return;
    state.kfClipboard = idxs.map((i) => Object.assign({}, bone.kf[i]));
    setStatus("ok", `copied ${idxs.length} keyframe${idxs.length===1?"":"s"}`);
  }

  function pasteKeyframes() {
    const m = state.motion;
    if (!m) return;
    if (state.kfClipboard.length === 0) return;
    // v4 / Task 3 — multi-bone clipboard payload (entries are
    // {boneIdx, kf} pairs) → paste each kf back to its origin bone,
    // anchoring the entire group at the scrubber's frame.
    const isMultiClip = state.kfClipboard.length > 0
      && typeof state.kfClipboard[0] === "object"
      && state.kfClipboard[0] !== null
      && "boneIdx" in state.kfClipboard[0]
      && "kf" in state.kfClipboard[0];
    if (isMultiClip) {
      // Anchor at the EARLIEST clipboard kf's t (across all bones).
      let anchorT = Infinity;
      for (const c of state.kfClipboard) {
        const t = c.kf.t | 0;
        if (t < anchorT) anchorT = t;
      }
      if (!Number.isFinite(anchorT)) anchorT = 0;
      const shift = (state.selectedFrame | 0) - anchorT;
      // Bucket by boneIdx, mirror the single-bone replace semantics
      // (existing kf at the destination frame is overwritten).
      const byBone = new Map();
      for (const c of state.kfClipboard) {
        const bi = c.boneIdx | 0;
        if (!byBone.has(bi)) byBone.set(bi, []);
        const t = (c.kf.t | 0) + shift;
        byBone.get(bi).push(Object.assign({}, c.kf, { t }));
      }
      const targetByBone = new Map();
      for (const [bi, newKfs] of byBone) {
        const b = m.bones[bi];
        if (!b) continue;
        if (!Array.isArray(b.kf)) b.kf = [];
        for (const k of newKfs) {
          const tt = k.t | 0;
          b.kf = b.kf.filter((x) => (x.t | 0) !== tt);
          b.kf.push(k);
        }
        b.kf.sort((a, b2) => (a.t | 0) - (b2.t | 0));
        targetByBone.set(bi, new Set(newKfs.map((k) => k.t | 0)));
      }
      invalidateRoundTrip(m);
      // Reselect pasted kfs by frame number per bone.
      state.selectedKfSetMulti.clear();
      let total = 0;
      for (const [bi, tset] of targetByBone) {
        const b = m.bones[bi];
        if (!b || !b.kf) continue;
        for (let i = 0; i < b.kf.length; i++) {
          if (tset.has(b.kf[i].t | 0)) {
            state.selectedKfSetMulti.add(`${bi}:${i}`);
            total++;
          }
        }
      }
      // If we're in single mode but pasted multi data, switch the user's
      // marquee mode so they see the result.
      if (state.marqueeMode !== "all") state.marqueeMode = "all";
      setStatus("ok", `pasted ${total} kf across ${targetByBone.size} bone${targetByBone.size===1?"":"s"}`);
      refreshEditor();
      return;
    }
    // Single-bone clipboard — legacy path (each entry is a bare kf).
    const bone = m.bones[state.selectedBoneIdx];
    if (!bone) return;
    if (!Array.isArray(bone.kf)) bone.kf = [];
    // Anchor at the scrubber: shift each clipboard kf so its first kf
    // lands at state.selectedFrame.
    const anchor = state.kfClipboard[0].t | 0;
    const newKfs = state.kfClipboard.map((k) =>
      Object.assign({}, k, { t: (state.selectedFrame + ((k.t | 0) - anchor)) | 0 }));
    // Drop any kf that already exists at that frame (we replace).
    for (const k of newKfs) {
      const tt = k.t | 0;
      bone.kf = bone.kf.filter((x) => (x.t | 0) !== tt);
      bone.kf.push(k);
    }
    bone.kf.sort((a, b) => (a.t | 0) - (b.t | 0));
    invalidateRoundTrip(m);
    // Re-select the newly-pasted kfs by frame (idx may have changed).
    state.selectedKfSet.clear();
    const targetTs = new Set(newKfs.map((k) => k.t | 0));
    for (let i = 0; i < bone.kf.length; i++) {
      if (targetTs.has(bone.kf[i].t | 0)) state.selectedKfSet.add(String(i));
    }
    setStatus("ok", `pasted ${newKfs.length} keyframe${newKfs.length===1?"":"s"}`);
    refreshEditor();
  }

  function resetSelectedToBind() {
    const m = state.motion;
    if (!m) return;
    const skel = state.skeleton || [];
    const sels = _iterSelected();
    if (sels.length === 0) return;
    let count = 0;
    for (const sel of sels) {
      const bone = m.bones[sel.boneIdx];
      if (!bone || !bone.kf) continue;
      const kf = bone.kf[sel.kfIdx];
      if (!kf) continue;
      const sk = skel[sel.boneIdx];
      // Skeleton snapshot exposes bind TRS as direct fields (position,
      // rotation_bams, scale) — see psoGetSkeleton in model_viewer.js.
      const bindPos = sk && sk.position ? sk.position : null;
      const bindRot = sk && sk.rotation_bams ? sk.rotation_bams : null;
      const bindScl = sk && sk.scale ? sk.scale : null;
      if (bindPos) {
        kf.tx = +bindPos[0]; kf.ty = +bindPos[1]; kf.tz = +bindPos[2];
      } else {
        kf.tx = 0; kf.ty = 0; kf.tz = 0;
      }
      if (bindRot) {
        kf.rx = bindRot[0] | 0; kf.ry = bindRot[1] | 0; kf.rz = bindRot[2] | 0;
      } else {
        kf.rx = 0; kf.ry = 0; kf.rz = 0;
      }
      if (bindScl) {
        kf.sx = +bindScl[0]; kf.sy = +bindScl[1]; kf.sz = +bindScl[2];
      } else {
        kf.sx = 1; kf.sy = 1; kf.sz = 1;
      }
      count++;
    }
    invalidateRoundTrip(m);
    setStatus("ok", `reset ${count} kf to bind`);
    refreshEditor();
  }

  // Move all selected keyframes by `delta` integer frames. Clamps so
  // no kf falls below 0 or exceeds frame_count-1, and de-dupes with
  // existing kfs at the destination frame (the moved kf wins).
  // v4 / Task 3 — in all-bones mode, the same delta is applied to
  // every selected kf, but each kf STAYS on its origin bone's track.
  function moveSelectedKeyframes(delta) {
    const m = state.motion;
    if (!m) return;
    const d = delta | 0;
    if (!d) return;
    const max = Math.max(0, (m.frame_count | 0) - 1);
    if (state.marqueeMode === "all") {
      // Bucket by bone, run the single-bone move algorithm per bucket.
      const byBone = new Map();
      for (const sel of _iterSelected()) {
        if (!byBone.has(sel.boneIdx)) byBone.set(sel.boneIdx, []);
        byBone.get(sel.boneIdx).push(sel.kfIdx);
      }
      const newSel = new Set();
      for (const [bi, idxs] of byBone) {
        const bone = m.bones[bi];
        if (!bone || !bone.kf) continue;
        const movedKfs = idxs.map((i) => Object.assign({}, bone.kf[i],
          { t: Math.min(max, Math.max(0, (bone.kf[i].t | 0) + d)) }));
        const movingIdxSet = new Set(idxs);
        const movedTSet = new Set(movedKfs.map((k) => k.t | 0));
        const kept = [];
        for (let i = 0; i < bone.kf.length; i++) {
          if (movingIdxSet.has(i)) continue;
          const t = bone.kf[i].t | 0;
          if (movedTSet.has(t)) continue;
          kept.push(bone.kf[i]);
        }
        kept.push(...movedKfs);
        kept.sort((a, b) => (a.t | 0) - (b.t | 0));
        bone.kf = kept;
        for (let i = 0; i < bone.kf.length; i++) {
          if (movedTSet.has(bone.kf[i].t | 0)) newSel.add(`${bi}:${i}`);
        }
      }
      state.selectedKfSetMulti = newSel;
      invalidateRoundTrip(m);
      return;
    }
    const bone = m.bones[state.selectedBoneIdx];
    if (!bone || !bone.kf) return;
    const idxs = [...state.selectedKfSet].map((s) => +s);
    if (idxs.length === 0) return;
    // Build a moved-kf list, then remove originals + drop collisions.
    const movedKfs = idxs.map((i) => Object.assign({}, bone.kf[i],
      { t: Math.min(max, Math.max(0, (bone.kf[i].t | 0) + d)) }));
    const movingIdxSet = new Set(idxs);
    const movedTSet = new Set(movedKfs.map((k) => k.t | 0));
    // Drop the originals that are moving + any pre-existing kf at the
    // destination frames that AREN'T moving.
    const kept = [];
    for (let i = 0; i < bone.kf.length; i++) {
      if (movingIdxSet.has(i)) continue;
      const t = bone.kf[i].t | 0;
      if (movedTSet.has(t)) continue;
      kept.push(bone.kf[i]);
    }
    kept.push(...movedKfs);
    kept.sort((a, b) => (a.t | 0) - (b.t | 0));
    bone.kf = kept;
    // Re-resolve selection to new indices via target frame numbers.
    state.selectedKfSet.clear();
    for (let i = 0; i < bone.kf.length; i++) {
      if (movedTSet.has(bone.kf[i].t | 0)) state.selectedKfSet.add(String(i));
    }
    invalidateRoundTrip(m);
  }

  // Build keyframe x-coordinates on the timeline for hit testing.
  // In single-bone mode, returns one entry per kf on the active bone.
  // In all-bones mode, returns ALL kfs across every bone — each entry
  // also carries `boneIdx` so the click test can resolve back to bone.
  function _kfHitMap() {
    const m = state.motion;
    if (!m) return [];
    const xPerFrame = state._timelineXPerFrame || 0;
    if (state.marqueeMode !== "all") {
      const bone = m.bones[state.selectedBoneIdx];
      if (!bone || !bone.kf) return [];
      return bone.kf.map((kf, i) => ({
        idx: i,
        t: kf.t | 0,
        x: (kf.t | 0) * xPerFrame,
        boneIdx: state.selectedBoneIdx | 0,
      }));
    }
    // All-bones mode: each bone's kfs become hits at lane y depending
    // on bone index. The y value is informational — selection happens
    // by x-range so users can sweep across all bones from the timeline.
    const out = [];
    for (let bi = 0; bi < m.bones.length; bi++) {
      const b = m.bones[bi];
      if (!b || !b.kf) continue;
      for (let i = 0; i < b.kf.length; i++) {
        const kf = b.kf[i];
        out.push({
          idx: i,
          t: kf.t | 0,
          x: (kf.t | 0) * xPerFrame,
          boneIdx: bi,
        });
      }
    }
    return out;
  }

  // ------------------------------------------------------------------
  // Timeline canvas pointer handling — marquee + drag-move + dbl-click.
  // ------------------------------------------------------------------
  function attachTimelineHandlers() {
    if (!state.bodyEl) return;
    const canvas = state.bodyEl.querySelector('[data-region="timelineCanvas"]');
    const host = state.bodyEl.querySelector('[data-region="timelineHost"]');
    if (!canvas || !host) return;
    if (canvas._akeBound) return;
    canvas._akeBound = true;

    let pointerId = null;
    let mode = null;       // "marquee" | "kfDrag" | null
    let dragStartFrame = 0;
    let dragLastDelta = 0;

    // v4 / Task 3 — selection-set helper that picks the right backing
    // store based on the current marquee mode. Cleans up branchy code
    // in the down/move handlers.
    function _selKey(h) {
      return state.marqueeMode === "all" ? `${h.boneIdx}:${h.idx}` : String(h.idx);
    }
    function _selSet() {
      return state.marqueeMode === "all" ? state.selectedKfSetMulti : state.selectedKfSet;
    }

    canvas.addEventListener("pointerdown", (ev) => {
      if (ev.button === 2) return;       // right-click handled separately
      if (!state.motion) return;
      const rect = canvas.getBoundingClientRect();
      const px = ev.clientX - rect.left;
      const xPerFrame = state._timelineXPerFrame || 0;
      const frame = xPerFrame > 0 ? Math.round(px / xPerFrame) : 0;
      const hits = _kfHitMap();
      // Pick the nearest keyframe within 6px.
      let pick = null;
      let bestDx = 6;
      for (const h of hits) {
        const dx = Math.abs(h.x - px);
        if (dx <= bestDx) { bestDx = dx; pick = h; }
      }
      if (pick) {
        const k = _selKey(pick);
        const sel = _selSet();
        if (ev.shiftKey) {
          if (sel.has(k)) sel.delete(k);
          else sel.add(k);
        } else if (!sel.has(k)) {
          sel.clear();
          sel.add(k);
        }
        // Set up drag-move from this keyframe.
        mode = "kfDrag";
        dragStartFrame = pick.t;
        dragLastDelta = 0;
        state.selectedFrame = pick.t;
        // In multi mode, also reflect picked bone into selectedBoneIdx
        // so the inspector switches automatically.
        if (state.marqueeMode === "all" && pick.boneIdx != null) {
          state.selectedBoneIdx = pick.boneIdx | 0;
        }
        seekModelToFrame(pick.t, true);
        canvas.setPointerCapture(ev.pointerId);
        pointerId = ev.pointerId;
        drawTimelineCanvas();
        refreshEditor();
      } else {
        // Empty space → start marquee selection.
        mode = "marquee";
        state.marquee = { startX: px, currentX: px };
        // Shift-marquee adds; plain marquee replaces selection.
        if (!ev.shiftKey) _selSet().clear();
        canvas.setPointerCapture(ev.pointerId);
        pointerId = ev.pointerId;
        drawTimelineCanvas();
      }
    });

    canvas.addEventListener("pointermove", (ev) => {
      if (pointerId !== ev.pointerId) return;
      const rect = canvas.getBoundingClientRect();
      const px = ev.clientX - rect.left;
      const xPerFrame = state._timelineXPerFrame || 0;
      const frame = xPerFrame > 0 ? Math.round(px / xPerFrame) : 0;
      if (mode === "marquee") {
        state.marquee.currentX = px;
        const x0 = Math.min(state.marquee.startX, state.marquee.currentX);
        const x1 = Math.max(state.marquee.startX, state.marquee.currentX);
        const hits = _kfHitMap();
        const sel = _selSet();
        for (const h of hits) {
          const k = _selKey(h);
          if (h.x >= x0 && h.x <= x1) {
            sel.add(k);
          } else if (!ev.shiftKey) {
            sel.delete(k);
          }
        }
        drawTimelineCanvas();
      } else if (mode === "kfDrag") {
        const delta = (frame - dragStartFrame) | 0;
        const inc = delta - dragLastDelta;
        if (inc !== 0) {
          moveSelectedKeyframes(inc);
          dragLastDelta = delta;
          drawTimelineCanvas();
          // Reflect into the keyframe-list view + scrubber.
          state.selectedFrame = (dragStartFrame + delta) | 0;
        }
      }
    });

    function endDrag(ev) {
      if (pointerId !== ev.pointerId) return;
      try { canvas.releasePointerCapture(pointerId); } catch (_e) {}
      const wasMode = mode;
      mode = null;
      pointerId = null;
      state.marquee = null;
      drawTimelineCanvas();
      if (wasMode === "kfDrag" || wasMode === "marquee") {
        refreshEditor();
      }
    }
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);

    // Right-click context menu.
    canvas.addEventListener("contextmenu", (ev) => {
      if (!state.motion) return;
      ev.preventDefault();
      // If right-clicked on a keyframe and that kf isn't selected, make
      // it the current single selection (Maya/Blender feel).
      const rect = canvas.getBoundingClientRect();
      const px = ev.clientX - rect.left;
      const hits = _kfHitMap();
      let pick = null;
      let bestDx = 6;
      for (const h of hits) {
        const dx = Math.abs(h.x - px);
        if (dx <= bestDx) { bestDx = dx; pick = h; }
      }
      if (pick) {
        const k = _selKey(pick);
        const sel = _selSet();
        if (!sel.has(k)) {
          sel.clear();
          sel.add(k);
          drawTimelineCanvas();
          refreshEditor();
        }
      }
      showContextMenu(ev.clientX, ev.clientY);
    });
  }

  // ------------------------------------------------------------------
  // Right-click context menu — delete / copy / paste / reset to bind.
  // ------------------------------------------------------------------
  function showContextMenu(x, y) {
    closeContextMenu();
    const n = state.marqueeMode === "all"
      ? state.selectedKfSetMulti.size
      : state.selectedKfSet.size;
    const menu = document.createElement("div");
    menu.className = "ake-ctx-menu";
    menu.style.left = x + "px";
    menu.style.top = y + "px";
    const items = [
      { act: "delete", label: `Delete (${n})`, danger: true, disabled: n === 0 },
      { act: "copy", label: `Copy (${n})`, disabled: n === 0 },
      { act: "paste", label: `Paste @ frame ${state.selectedFrame}`, disabled: state.kfClipboard.length === 0 },
      { sep: true },
      { act: "reset-bind", label: "Reset to bind pose", disabled: n === 0 },
    ];
    for (const it of items) {
      if (it.sep) {
        const s = document.createElement("div");
        s.className = "sep";
        menu.appendChild(s);
        continue;
      }
      const el = document.createElement("div");
      el.className = "item" + (it.danger ? " danger" : "") + (it.disabled ? " disabled" : "");
      el.textContent = it.label;
      if (!it.disabled) {
        el.addEventListener("click", () => {
          closeContextMenu();
          if (it.act === "delete") deleteSelectedKeyframes();
          else if (it.act === "copy") copySelectedKeyframes();
          else if (it.act === "paste") pasteKeyframes();
          else if (it.act === "reset-bind") resetSelectedToBind();
        });
      }
      menu.appendChild(el);
    }
    document.body.appendChild(menu);
    state._ctxMenu = menu;
    setTimeout(() => {
      document.addEventListener("mousedown", _ctxOutside, { capture: true });
    }, 0);
  }
  function _ctxOutside(ev) {
    if (state._ctxMenu && !state._ctxMenu.contains(ev.target)) {
      closeContextMenu();
    }
  }
  function closeContextMenu() {
    if (state._ctxMenu && state._ctxMenu.parentNode) {
      state._ctxMenu.parentNode.removeChild(state._ctxMenu);
    }
    state._ctxMenu = null;
    document.removeEventListener("mousedown", _ctxOutside, { capture: true });
  }

  // ------------------------------------------------------------------
  // Task 4 — Curve editor with bezier tangents.
  // ------------------------------------------------------------------
  function _bezierKey(boneIdx, kfIdx, channel) {
    return `${boneIdx}:${kfIdx}:${channel}`;
  }

  function _readChannel(kf, channel) {
    return +kf[channel];
  }
  function _writeChannel(kf, channel, value) {
    if (channel === "rx" || channel === "ry" || channel === "rz") {
      kf[channel] = (value | 0);
    } else {
      kf[channel] = +value;
    }
  }

  function _curveExtents(bone, channel) {
    const kfs = (bone && bone.kf) || [];
    let lo = Infinity, hi = -Infinity;
    for (const kf of kfs) {
      const v = _readChannel(kf, channel);
      if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (lo === Infinity) { lo = 0; hi = 1; }
    if (lo === hi) { lo -= 1; hi += 1; }
    // 10% padding.
    const pad = (hi - lo) * 0.1;
    return [lo - pad, hi + pad];
  }

  function drawCurveCanvas() {
    if (!state.curveOpen) return;
    if (!state.bodyEl || !state.motion) return;
    const canvas = state.bodyEl.querySelector('[data-region="curveCanvas"]');
    const host = state.bodyEl.querySelector('[data-region="curveHost"]');
    if (!canvas || !host) return;
    const dpr = window.devicePixelRatio || 1;
    const cssW = Math.max(50, host.clientWidth);
    const cssH = state.curveMode === "all" ? CURVE_CANVAS_HEIGHT_ALL : CURVE_CANVAS_HEIGHT;
    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
    canvas.style.width = cssW + "px";
    canvas.style.height = cssH + "px";
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.fillStyle = "#07090d";
    ctx.fillRect(0, 0, cssW, cssH);

    const m = state.motion;
    const bone = m.bones[state.selectedBoneIdx];
    if (!bone || !bone.kf || bone.kf.length === 0) {
      ctx.fillStyle = "#6c7785";
      ctx.font = "11px sans-serif";
      ctx.fillText("no keyframes on this bone", 8, cssH / 2);
      return;
    }
    const activeKey = state.curveChannelKey;
    const present = bone.present | 0;
    const cur = CURVE_CHANNELS.find((c) => c.key === activeKey);
    const channels = _activeCurveChannels(bone);
    if (channels.length === 0) {
      ctx.fillStyle = "#ffaa00";
      ctx.font = "11px sans-serif";
      ctx.fillText(`no channels authored on this bone`, 8, cssH / 2);
      return;
    }
    const frameCount = Math.max(1, m.frame_count | 0);
    const xPerFrame = cssW / Math.max(1, frameCount - 1);

    // Grid + scrubber line — global to all curves.
    ctx.strokeStyle = "#15191f";
    for (let f = 0; f < frameCount; f += 5) {
      ctx.beginPath();
      ctx.moveTo((f|0) * xPerFrame, 0);
      ctx.lineTo((f|0) * xPerFrame, cssH);
      ctx.stroke();
    }
    ctx.strokeStyle = "rgba(0,255,255,0.3)";
    ctx.beginPath();
    ctx.moveTo((state.selectedFrame|0) * xPerFrame, 0);
    ctx.lineTo((state.selectedFrame|0) * xPerFrame, cssH);
    ctx.stroke();

    // Per-channel y-projection. Each channel gets its OWN y-extent so
    // BAMS rotations don't squash POS-world-units to a flat line. The
    // "active" channel is the only one whose handles are editable —
    // its yScale is what handle dragging reads.
    const kfs = bone.kf;
    let activeFx = null;        // captured for the handle-edit fall-through

    for (const ch of channels) {
      const isActive = ch.key === activeKey;
      const [yLo, yHi] = _curveExtents(bone, ch.key);
      const yScale = (cssH - 16) / Math.max(1e-6, yHi - yLo);
      const fy = (v) => cssH - 8 - (v - yLo) * yScale;
      const fx = (t) => (t | 0) * xPerFrame;

      // Y=0 reference line for the ACTIVE channel only (others would
      // clutter the canvas). Drawn under the curve.
      if (isActive && yLo < 0 && yHi > 0) {
        ctx.strokeStyle = "#1a1f26";
        ctx.beginPath();
        ctx.moveTo(0, fy(0));
        ctx.lineTo(cssW, fy(0));
        ctx.stroke();
      }

      // Curve segments — bezier between consecutive keyframes.
      ctx.strokeStyle = ch.color;
      ctx.lineWidth = isActive ? 1.5 : 1.0;
      ctx.globalAlpha = isActive ? 1.0 : 0.55;
      ctx.beginPath();
      for (let i = 0; i < kfs.length - 1; i++) {
        const a = kfs[i], b = kfs[i + 1];
        const ax = fx(a.t), ay = fy(_readChannel(a, ch.key));
        const bx = fx(b.t), by = fy(_readChannel(b, ch.key));
        const aHandle = state.bezierHandles.get(_bezierKey(state.selectedBoneIdx, i, ch.key));
        const bHandle = state.bezierHandles.get(_bezierKey(state.selectedBoneIdx, i + 1, ch.key));
        const defaultDx = (b.t - a.t) / 3;
        const cax = ax + (aHandle ? aHandle.outDx : defaultDx) * xPerFrame;
        const cay = ay + (aHandle ? -aHandle.outDy * yScale : 0);
        const cbx = bx + (bHandle ? bHandle.inDx : -defaultDx) * xPerFrame;
        const cby = by + (bHandle ? -bHandle.inDy * yScale : 0);
        if (i === 0) ctx.moveTo(ax, ay);
        ctx.bezierCurveTo(cax, cay, cbx, cby, bx, by);
      }
      ctx.stroke();
      ctx.globalAlpha = 1.0;

      // Keyframe dots — only on the active channel get the saturated
      // selection halo. Inactive channels show a small dim dot so the
      // user can still see where the kfs land.
      for (let i = 0; i < kfs.length; i++) {
        const kf = kfs[i];
        const px = fx(kf.t), py = fy(_readChannel(kf, ch.key));
        if (isActive) {
          const isSel = state.selectedKfSet.has(String(i));
          ctx.fillStyle = isSel ? "#ffd060" : ch.color;
          ctx.beginPath();
          ctx.arc(px, py, 4, 0, Math.PI * 2);
          ctx.fill();
        } else {
          ctx.fillStyle = ch.color;
          ctx.globalAlpha = 0.5;
          ctx.beginPath();
          ctx.arc(px, py, 2, 0, Math.PI * 2);
          ctx.fill();
          ctx.globalAlpha = 1.0;
        }
      }

      if (isActive) {
        // Handles: only the active channel draws editable handles.
        for (let i = 0; i < kfs.length; i++) {
          const kf = kfs[i];
          const px = fx(kf.t), py = fy(_readChannel(kf, ch.key));
          const isSel = state.selectedKfSet.has(String(i));
          if (state.selectedKfSet.size === 1 && isSel) {
            const handle = state.bezierHandles.get(_bezierKey(state.selectedBoneIdx, i, ch.key))
                        || { inDx: -2, inDy: 0, outDx: 2, outDy: 0 };
            const ihx = px + handle.inDx * xPerFrame;
            const ihy = py + (-handle.inDy * yScale);
            const ohx = px + handle.outDx * xPerFrame;
            const ohy = py + (-handle.outDy * yScale);
            ctx.strokeStyle = "rgba(255,200,80,0.6)";
            ctx.beginPath();
            ctx.moveTo(ihx, ihy); ctx.lineTo(px, py); ctx.lineTo(ohx, ohy);
            ctx.stroke();
            ctx.fillStyle = "#ffd060";
            ctx.fillRect(ihx - 3, ihy - 3, 6, 6);
            ctx.fillRect(ohx - 3, ohy - 3, 6, 6);
            canvas._handleScreen = { kfIdx: i, ihx, ihy, ohx, ohy, px, py, xPerFrame, yScale };
          }
        }
        // Cache projection fns of the ACTIVE channel for click-to-pick.
        const ifx = (px) => Math.round(px / xPerFrame);
        const ify = (py) => yLo + ((cssH - 8 - py) / yScale);
        activeFx = { fx, fy, ifx, ify, xPerFrame, yScale };
      }
    }
    canvas._fxFns = activeFx || { xPerFrame };
    // Cache the channel set + their per-channel y-projection so click
    // tests can identify which curve was clicked (Task 1: click to make
    // active for handle editing).
    canvas._channelSet = channels.map((ch) => {
      const [yLo, yHi] = _curveExtents(bone, ch.key);
      const yScale = (cssH - 16) / Math.max(1e-6, yHi - yLo);
      return { key: ch.key, color: ch.color, yLo, yHi, yScale };
    });
    canvas._cssH = cssH;
  }

  function attachCurveHandlers() {
    if (!state.bodyEl) return;
    const canvas = state.bodyEl.querySelector('[data-region="curveCanvas"]');
    if (!canvas || canvas._akeCurveBound) return;
    canvas._akeCurveBound = true;
    let pid = null;
    let drag = null;
    canvas.addEventListener("pointerdown", (ev) => {
      const fns = canvas._fxFns;
      const hs = canvas._handleScreen;
      const rect = canvas.getBoundingClientRect();
      const px = ev.clientX - rect.left;
      const py = ev.clientY - rect.top;
      // Hit-test handle squares first (only valid when one kf is selected).
      if (fns && hs) {
        const inDist = Math.hypot(px - hs.ihx, py - hs.ihy);
        const outDist = Math.hypot(px - hs.ohx, py - hs.ohy);
        if (inDist <= 6 || outDist <= 6) {
          drag = { kfIdx: hs.kfIdx, side: inDist <= outDist ? "in" : "out" };
          canvas.setPointerCapture(ev.pointerId);
          pid = ev.pointerId;
          return;
        }
      }
      // v4 / Task 1 — click anywhere on a non-active curve to make it
      // active for handle editing. We hit-test each cached channel by
      // sampling its (linear-interp) y at the cursor's frame and picking
      // the one with the smallest pixel distance to the cursor.
      const set = canvas._channelSet || [];
      if (set.length > 1 && fns && fns.xPerFrame) {
        const m = state.motion;
        const bone = m && m.bones[state.selectedBoneIdx];
        if (!bone || !bone.kf || bone.kf.length === 0) return;
        const xPerFrame = fns.xPerFrame;
        const cssH = canvas._cssH || CURVE_CANVAS_HEIGHT;
        const f = px / xPerFrame;
        let best = null;
        let bestPxDist = 8;     // 8px tolerance
        for (const ch of set) {
          const v = _interpAtFrame(bone, ch.key, f);
          if (!Number.isFinite(v)) continue;
          const cy = cssH - 8 - (v - ch.yLo) * ch.yScale;
          const dy = Math.abs(cy - py);
          if (dy < bestPxDist) {
            bestPxDist = dy;
            best = ch;
          }
        }
        if (best && best.key !== state.curveChannelKey) {
          state.curveChannelKey = best.key;
          // Re-render so the legend + active dropdown reflect the swap.
          refreshEditor();
          return;
        }
      }
    });
    canvas.addEventListener("pointermove", (ev) => {
      if (pid !== ev.pointerId || !drag) return;
      const fns = canvas._fxFns;
      const hs = canvas._handleScreen;
      if (!fns || !hs) return;
      const rect = canvas.getBoundingClientRect();
      const px = ev.clientX - rect.left;
      const py = ev.clientY - rect.top;
      const dxFrames = (px - hs.px) / fns.xPerFrame;
      const dyValue = -(py - hs.py) / fns.yScale;
      const key = _bezierKey(state.selectedBoneIdx, drag.kfIdx, state.curveChannelKey);
      const cur = state.bezierHandles.get(key)
                 || { inDx: -2, inDy: 0, outDx: 2, outDy: 0 };
      if (drag.side === "in") { cur.inDx = dxFrames; cur.inDy = dyValue; }
      else { cur.outDx = dxFrames; cur.outDy = dyValue; }
      state.bezierHandles.set(key, cur);
      drawCurveCanvas();
    });
    function endDrag(ev) {
      if (pid !== ev.pointerId) return;
      try { canvas.releasePointerCapture(pid); } catch (_e) {}
      pid = null;
      drag = null;
    }
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);
  }

  function resetBezierHandlesOnChannel() {
    const m = state.motion;
    if (!m) return;
    const bone = m.bones[state.selectedBoneIdx];
    if (!bone || !bone.kf) return;
    const ch = state.curveChannelKey;
    let dropped = 0;
    for (const k of [...state.bezierHandles.keys()]) {
      const parts = k.split(":");
      if (+parts[0] === state.selectedBoneIdx && parts[2] === ch) {
        state.bezierHandles.delete(k);
        dropped++;
      }
    }
    setStatus("ok", `cleared ${dropped} bezier handles`);
    drawCurveCanvas();
  }

  // Densify the channel curve into linear keyframes — used at save
  // time. Only operates on bezier-decorated keyframes; channels that
  // never had handles applied stay untouched. Returns the new kf array
  // for the bone (sorted by frame).
  function densifyBezierToLinear(bone, boneIdx, channel) {
    if (!bone || !bone.kf || bone.kf.length < 2) return bone.kf;
    const cur = CURVE_CHANNELS.find((c) => c.key === channel);
    if (!cur) return bone.kf;
    const present = bone.present | 0;
    if (!(present & cur.kind)) return bone.kf;
    // Detect any handle on this channel; if none, no-op.
    const prefix = `${boneIdx}:`;
    let anyHandle = false;
    for (const k of state.bezierHandles.keys()) {
      if (k.startsWith(prefix) && k.endsWith(`:${channel}`)) {
        anyHandle = true; break;
      }
    }
    if (!anyHandle) return bone.kf;
    const kfs = bone.kf.slice().sort((a, b) => (a.t | 0) - (b.t | 0));
    const stride = Math.max(1, state.curveBakeStride | 0);
    const baked = [];
    // Always keep the first kf as-is.
    baked.push(Object.assign({}, kfs[0]));
    for (let i = 0; i < kfs.length - 1; i++) {
      const a = kfs[i], b = kfs[i + 1];
      const ax = a.t | 0, bx = b.t | 0;
      const span = bx - ax;
      if (span <= 0) continue;
      const av = _readChannel(a, channel), bv = _readChannel(b, channel);
      const aH = state.bezierHandles.get(_bezierKey(boneIdx, i, channel));
      const bH = state.bezierHandles.get(_bezierKey(boneIdx, i + 1, channel));
      const defaultDx = span / 3;
      // Bezier control points in (t, value) space.
      const c1t = ax + (aH ? aH.outDx : defaultDx);
      const c1v = av + (aH ? aH.outDy : 0);
      const c2t = bx + (bH ? bH.inDx : -defaultDx);
      const c2v = bv + (bH ? bH.inDy : 0);
      // Walk integer frames between ax and bx-1, sampling the cubic.
      for (let f = ax + stride; f < bx; f += stride) {
        // Solve for t_param given x(t) = (1-t)^3*ax + 3(1-t)^2*t*c1t + 3(1-t)*t^2*c2t + t^3*bx = f.
        // Brute Newton step from initial t=(f-ax)/span gives close
        // enough results for the bake — we only need integer-frame
        // samples, not sub-frame precision.
        let tp = (f - ax) / span;
        for (let iter = 0; iter < 5; iter++) {
          const omt = 1 - tp;
          const xx = omt*omt*omt*ax + 3*omt*omt*tp*c1t + 3*omt*tp*tp*c2t + tp*tp*tp*bx;
          const dx = -3*omt*omt*ax + 3*(1-4*tp+3*tp*tp)*c1t + 3*(2*tp-3*tp*tp)*c2t + 3*tp*tp*bx;
          if (Math.abs(dx) < 1e-6) break;
          tp -= (xx - f) / dx;
          if (tp < 0) tp = 0;
          if (tp > 1) tp = 1;
        }
        const omt = 1 - tp;
        const yy = omt*omt*omt*av + 3*omt*omt*tp*c1v + 3*omt*tp*tp*c2v + tp*tp*tp*bv;
        const newKf = Object.assign({}, kfs[i]);  // copy other channels from the prior kf
        newKf.t = f;
        _writeChannel(newKf, channel, yy);
        baked.push(newKf);
      }
      baked.push(Object.assign({}, b));   // anchor at end of segment
    }
    // De-dup any colliding frames (when stride lands on b.t).
    const seen = new Map();
    for (const k of baked) {
      const t = k.t | 0;
      seen.set(t, k);
    }
    return [...seen.values()].sort((a, b) => (a.t | 0) - (b.t | 0));
  }

  // Apply bezier densification to ALL bones+channels that have handles.
  // Returns a deep-clone of the motion with the bake applied.
  function bakeAllBezierToLinear(motion) {
    if (!motion) return motion;
    if (state.bezierHandles.size === 0) return motion;
    const out = deepClone(motion);
    for (let b = 0; b < (out.bones || []).length; b++) {
      const bone = out.bones[b];
      for (const ch of CURVE_CHANNELS) {
        // Only loop channels actually authored.
        const present = bone.present | 0;
        if (!(present & ch.kind)) continue;
        const newKfs = densifyBezierToLinear(bone, b, ch.key);
        if (newKfs && newKfs !== bone.kf) {
          bone.kf = newKfs;
        }
      }
    }
    invalidateRoundTrip(out);
    return out;
  }
  // Expose for tests.
  window.__akeBakeAllBezierToLinear = bakeAllBezierToLinear;
  window.__akeDensifyBezier = densifyBezierToLinear;

  // ------------------------------------------------------------------
  // Init
  // ------------------------------------------------------------------
  function init() {
    waitForPanel(Date.now() + 30_000);
    if (window.bus && typeof window.bus.on === "function") {
      window.bus.on("model.loaded", () => {
        state.motion = null;
        state.motionOriginal = null;
        state.motionName = "";
        state.modelPath = "";
        state.saveAsName = "";
        state.swapTargetSlot = "";
        state.selectedBoneIdx = 0;
        state.selectedFrame = 0;
        state.playing = false;
        state.selectedKfSet.clear();
        state.bezierHandles.clear();
        // Drop renderer-side bone overrides authored by eye-toggle on
        // the previous model (matches loadMotionFromPicker semantics).
        releaseAllPanelBoneOverrides();
        state.boneTreeHidden.clear();
        state.boneTreeCollapsed.clear();
        state.boneTreeQuery = "";
        state.kfClipboard = [];
        if (state.panelMounted) refreshMotionPicker();
        if (state.panelMounted) refreshEditor();
      });
      // Live-reload (v5 polish): when an .njm staged in cache/njm_export/
      // changes on disk, refresh the motion picker so a freshly-imported
      // animation appears without an F5. Skip events for other watched
      // dirs (painted_textures, sculpted_meshes, etc.).
      window.bus.on("cache.changed", (payload) => {
        if (!payload || !payload.path) return;
        if (payload.path.indexOf("cache/njm_export/") !== 0) return;
        // Sidecars (.preview.json) ride the same SSE stream — those
        // also matter because the picker shows preview-list items.
        if (state.panelMounted) refreshMotionPicker();
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // Devtools handles
  window.psoAnimEditorState = state;
  window.psoAnimEditorRefresh = () => {
    refreshMotionPicker();
    refreshEditor();
  };
})();
