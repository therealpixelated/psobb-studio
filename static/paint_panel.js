// =====================================================================
// PSOBB Texture Editor — UV-aware Texture Paint Panel (2026-04-25, v5).
//
// Adds a "Paint" tab to the existing in-viewport texture-panel tab strip
// (Textures / Layers / Motions / Subdivide / **Paint**) and surfaces the
// brush toolbar inside the panel body. The viewport canvas itself is the
// paint surface — clicking + dragging on the model raycasts into the
// live mesh group, converts (face, uv) into a pixel coordinate of the
// bound texture, stamps a Gaussian-falloff brush onto a CPU 2D canvas
// that backs a THREE.CanvasTexture, and re-uploads to the GPU per-frame.
//
// Tools (v5 set):
//   B brush       click+drag → coloured stamp at the UV location.
//   E eraser      click+drag → reduces alpha at the UV location.
//   I picker      click → samples colour under cursor into the swatch.
//   F fill        click → flood-fills the connected UV region.
//   S smear       click+drag → smudges pixels along the drag direction.
//   T clone       Alt+click sets source; click+drag stamps from source.
//   G gradient    click+drag defines line; renders stops on release.
//
// Layer system (v5):
//   • Each painted texture has a STACK of layers (1+).
//   • Per-layer: name, visible, opacity, blend_mode (normal / multiply /
//     screen / overlay), locked, optional alpha mask.
//   • All paint tools target the ACTIVE layer (or the active layer's
//     mask when "Paint mask" is on).
//   • Compositing (CPU-side): bottom-to-top, src-over for "normal" plus
//     RGB combiners for multiply/screen/overlay. Result drives the
//     bound CanvasTexture.
//   • Persistence: cache/painted_textures/<safe>/<idx>.png + manifest.
//
// Mouse buttons:
//   LMB           paint with the active tool.
//   RMB / MMB     orbit the camera (the existing model_viewer.js drag).
//   Wheel         (when paint-mode is on) adjust brush size.
//
// Save flow:
//   /api/paint/layer/save     writes one layer/mask + manifest, returns
//                             the recomputed composite md5.
//   /api/paint/manifest       replaces the manifest after reorder/merge.
//   /api/paint/load           re-loads the full layer stack on session
//                             restore.
//   /api/paint/build_archive  rebuilds the host BML/AFS with painted
//                             textures (operates on the composite PNG).
//
// Architectural notes — read before extending:
//   • The paint canvas is a per-tile 2D <canvas> (CPU-side). On entering
//     paint mode for a tile, we fetch /api/paint/load (or fall back to
//     /api/tile_png) and seed the layer stack.
//   • A THREE.CanvasTexture wraps the COMPOSITE canvas and replaces
//     every material.map whose materialId binds to the tile (via the
//     additive window.psoSetMaterialTexture export). The composite
//     canvas is recomputed CPU-side from the layer stack on every
//     stroke (recompositeNow()).
//   • Undo/redo is a snapshot stack of the active LAYER's ImageData
//     (not the composite). On undo we re-composite. STACK_LIMIT caps
//     memory at ~5 MB per 1024² snap.
//   • Pointer events register in CAPTURING phase on the WebGL canvas so
//     we run before model_viewer.js's orbit handler. We swallow LMB
//     events when paint-mode is on to keep the camera still.
// =====================================================================

(function () {
  "use strict";

  if (window.__psoPaintPanelLoaded) return;
  window.__psoPaintPanelLoaded = true;

  // ---- constants -----------------------------------------------------
  const STYLE_ID = "psoPaintPanelStyle";
  const STACK_LIMIT = 50;
  const DEFAULT_BRUSH_SIZE = 32;
  const DEFAULT_OPACITY = 1.0;
  const DEFAULT_HARDNESS = 0.6;
  const DEFAULT_COLOR = "#ff3344";
  const TOOLS = ["brush", "eraser", "picker", "fill", "smear", "clone", "gradient"];
  const TOOL_KEYS = {
    B: "brush", E: "eraser", I: "picker", F: "fill",
    S: "smear", T: "clone", G: "gradient",
  };
  const TOOL_LABELS = {
    brush:    ["B", "Paintbrush"],
    eraser:   ["E", "Eraser"],
    picker:   ["I", "Color picker"],
    fill:     ["F", "Fill bucket"],
    smear:    ["S", "Smear"],
    clone:    ["T", "Clone stamp"],
    gradient: ["G", "Gradient"],
  };
  const SUPPORTED_BLEND_MODES = ["normal", "multiply", "screen", "overlay"];
  const GRADIENT_KINDS = ["linear", "radial", "angular"];
  const MAX_LAYERS = 16;

  // Mask paint colors are forced to black/white (paint reveals/hides);
  // anything in between alters partial-transparency on the layer.
  const MASK_WHITE = "#ffffff";
  const MASK_BLACK = "#000000";

  // ---- DOM helpers ---------------------------------------------------
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;",
      '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
      .pso-paint-block {
        padding: 8px;
        display: flex;
        flex-direction: column;
        gap: 8px;
        font-size: 11px;
      }
      .pso-paint-block label { color: #99a4b3; }
      .pso-paint-row { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
      .pso-paint-row .grow { flex: 1; }
      .pso-paint-tools {
        display: grid;
        grid-template-columns: repeat(7, 1fr);
        gap: 3px;
      }
      .pso-paint-tools button {
        background: transparent;
        border: 1px solid #2a313a;
        color: #99a4b3;
        cursor: pointer;
        padding: 4px 2px;
        font: inherit;
        font-size: 10px;
        border-radius: 2px;
        text-align: center;
      }
      .pso-paint-tools button:hover { border-color: #4a90e2; color: #c7d8ec; }
      .pso-paint-tools button.active {
        background: rgba(0, 255, 255, 0.15);
        border-color: #00ffff;
        color: #00ffff;
      }
      .pso-paint-slider { display: flex; gap: 6px; align-items: center; flex: 1; }
      .pso-paint-slider input[type="range"] { flex: 1; }
      .pso-paint-slider .num {
        min-width: 32px;
        text-align: right;
        color: #c7d8ec;
        font-variant-numeric: tabular-nums;
      }
      .pso-paint-active {
        background: rgba(0, 0, 0, 0.25);
        border: 1px solid #2a313a;
        border-radius: 2px;
        padding: 4px 6px;
        display: flex;
        gap: 8px;
        align-items: center;
      }
      .pso-paint-active img {
        width: 32px; height: 32px; object-fit: contain;
        border: 1px solid #2a313a; border-radius: 2px;
        background: #0a0e13 url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='8' height='8'><rect width='4' height='4' fill='%23222'/><rect x='4' y='4' width='4' height='4' fill='%23222'/></svg>");
        background-size: 8px 8px;
      }
      .pso-paint-active .info { flex: 1; min-width: 0; }
      .pso-paint-active .info .nm { color: #c7d8ec; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
      .pso-paint-active .info .meta { color: #6c7785; font-size: 10px; }
      .pso-paint-tex-row {
        display: flex; gap: 6px; padding: 3px 4px; cursor: pointer;
        border: 1px solid transparent; border-radius: 2px;
      }
      .pso-paint-tex-row:hover { border-color: #4a90e2; }
      .pso-paint-tex-row.active {
        border-color: #00ffff;
        background: rgba(0, 255, 255, 0.06);
      }
      .pso-paint-tex-row img {
        width: 32px; height: 32px; object-fit: contain;
        border: 1px solid #2a313a; border-radius: 2px;
        background: #0a0e13;
      }
      .pso-paint-tex-row .nm {
        flex: 1; min-width: 0;
        color: #c7d8ec; font-size: 10px;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
      }
      .pso-paint-actions { display: flex; gap: 4px; flex-wrap: wrap; }
      .pso-paint-actions button {
        background: transparent;
        border: 1px solid #2a313a;
        color: #99a4b3;
        cursor: pointer;
        padding: 3px 6px;
        font: inherit;
        font-size: 10px;
        border-radius: 2px;
      }
      .pso-paint-actions button:hover { border-color: #4a90e2; color: #c7d8ec; }
      .pso-paint-actions button.primary { color: #00ffff; border-color: #4a90e2; }
      .pso-paint-actions button.primary:hover { background: rgba(0, 255, 255, 0.12); border-color: #00ffff; }
      .pso-paint-actions button.warning { color: #d8c890; border-color: #4d4523; }
      .pso-paint-actions button.warning:hover { border-color: #ffaa00; color: #ffaa00; }
      .pso-paint-actions button:disabled { opacity: 0.4; cursor: not-allowed; }
      .pso-paint-status {
        font-size: 10px; min-height: 12px;
        color: #6c7785;
      }
      .pso-paint-status.ok { color: #56b67a; }
      .pso-paint-status.err { color: #ff6680; }
      .pso-paint-status.busy { color: #d8c890; }
      .pso-paint-tip {
        color: #6c7785;
        font-style: italic;
        font-size: 10px;
        line-height: 1.4;
      }

      /* Brush cursor — a thin ring overlaid on the WebGL canvas to show
         brush radius. Positioned at the mouse and sized in CSS px. */
      .pso-paint-cursor {
        position: absolute;
        border: 1px solid rgba(255, 255, 255, 0.6);
        box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.6);
        border-radius: 50%;
        pointer-events: none;
        z-index: 7;
        transform: translate(-50%, -50%);
      }

      /* Paint-mode marker — body class flips orbit suppression elsewhere
         (we just toggle it for visual debug). */
      body.pso-paint-mode-on .model-stage { cursor: crosshair; }

      /* ---- Layer panel (v5) ----------------------------------------- */
      .pso-paint-layers-block {
        border: 1px solid #2a313a;
        border-radius: 2px;
        padding: 4px;
        background: rgba(0, 0, 0, 0.18);
        display: flex; flex-direction: column; gap: 4px;
      }
      .pso-paint-layers-head {
        display: flex; align-items: center; gap: 4px;
        font-size: 10px; color: #99a4b3;
      }
      .pso-paint-layers-head .grow { flex: 1; }
      .pso-paint-layers-head button {
        background: transparent; border: 1px solid #2a313a;
        color: #99a4b3; cursor: pointer;
        padding: 2px 5px; font: inherit; font-size: 10px;
        border-radius: 2px;
      }
      .pso-paint-layers-head button:hover { border-color: #4a90e2; color: #c7d8ec; }
      .pso-paint-layers-head button:disabled { opacity: 0.4; cursor: not-allowed; }
      .pso-paint-layer-list {
        display: flex; flex-direction: column-reverse;
        gap: 2px;
        max-height: 220px; overflow-y: auto;
      }
      .pso-paint-layer-row {
        display: flex; align-items: center; gap: 3px;
        padding: 2px 3px; border: 1px solid transparent;
        border-radius: 2px; cursor: grab; font-size: 10px;
      }
      .pso-paint-layer-row.active {
        border-color: #00ffff;
        background: rgba(0, 255, 255, 0.08);
      }
      .pso-paint-layer-row.dragging { opacity: 0.5; }
      .pso-paint-layer-row.drop-above {
        border-top-color: #ffaa00;
      }
      .pso-paint-layer-row.drop-below {
        border-bottom-color: #ffaa00;
      }
      .pso-paint-layer-row.mask-active {
        outline: 1px dashed #ffaa00;
      }
      .pso-paint-layer-row .eye {
        width: 16px; height: 14px;
        cursor: pointer; user-select: none;
        text-align: center;
        color: #c7d8ec;
        opacity: 0.85;
      }
      .pso-paint-layer-row .eye.hidden { opacity: 0.25; }
      .pso-paint-layer-row .nm {
        flex: 1; min-width: 0;
        color: #c7d8ec;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
      }
      .pso-paint-layer-row.locked .nm { color: #6c7785; }
      .pso-paint-layer-row .lock {
        cursor: pointer; color: #6c7785;
        font-size: 9px; padding: 1px 2px;
      }
      .pso-paint-layer-row .lock.on { color: #ffaa00; }
      .pso-paint-layer-row .mini-mask {
        width: 14px; height: 14px;
        background: #1a1d22 url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='12'><circle cx='6' cy='6' r='5' fill='%23999'/></svg>") center/contain no-repeat;
        border: 1px solid #2a313a;
        cursor: pointer;
      }
      .pso-paint-layer-row .mini-mask.has { background-color: #1a1d22; outline: 1px solid #ffaa00; }
      .pso-paint-layer-row .blend-mode {
        background: transparent; border: 1px solid #2a313a;
        color: #99a4b3; font: inherit; font-size: 9px;
        padding: 0; max-width: 64px;
      }
      .pso-paint-layer-row .opacity {
        width: 40px;
      }
      .pso-paint-layers-empty {
        font-size: 10px; color: #6c7785;
        font-style: italic; padding: 4px;
      }
      .pso-paint-layer-context {
        position: absolute;
        background: #14181d;
        border: 1px solid #2a313a;
        border-radius: 2px;
        padding: 4px 0;
        z-index: 50;
        min-width: 140px;
        font-size: 11px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.7);
      }
      .pso-paint-layer-context button {
        display: block; width: 100%;
        background: transparent; border: 0; color: #c7d8ec;
        font: inherit; font-size: 11px; text-align: left;
        padding: 4px 10px; cursor: pointer;
      }
      .pso-paint-layer-context button:hover { background: rgba(74, 144, 226, 0.15); }
      .pso-paint-layer-context button:disabled { color: #4a525c; cursor: not-allowed; }
      .pso-paint-layer-context hr {
        border: 0; border-top: 1px solid #2a313a;
        margin: 4px 0;
      }

      /* ---- Tool option strips (clone / gradient) -------------------- */
      .pso-paint-tool-options {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid #2a313a;
        border-radius: 2px;
        padding: 4px 6px;
        font-size: 10px;
        display: flex; gap: 6px; flex-wrap: wrap;
        align-items: center;
      }
      .pso-paint-tool-options .lbl { color: #99a4b3; }
      .pso-paint-tool-options select {
        background: #14181d; border: 1px solid #2a313a;
        color: #c7d8ec; font: inherit; font-size: 10px;
      }
      .pso-paint-tool-options input[type="color"] {
        width: 22px; height: 16px;
        background: transparent; border: 0; padding: 0;
        cursor: pointer;
      }
      .pso-paint-tool-options input[type="number"] {
        width: 54px;
        background: #14181d; border: 1px solid #2a313a;
        color: #c7d8ec; font: inherit; font-size: 10px;
      }
      .pso-paint-grad-stops {
        display: flex; flex-direction: column;
        gap: 3px; padding: 4px 0; min-width: 100%;
      }
      .pso-paint-grad-stop {
        display: flex; gap: 4px; align-items: center;
      }
      .pso-paint-grad-stop input[type="range"] { flex: 1; }
      .pso-paint-clone-source-pip {
        position: absolute;
        width: 16px; height: 16px;
        border-radius: 50%;
        border: 2px solid #ffaa00;
        background: rgba(255, 170, 0, 0.15);
        pointer-events: none; z-index: 7;
        transform: translate(-50%, -50%);
      }
      /* Gradient drag preview line on the WebGL canvas. */
      .pso-paint-grad-preview {
        position: absolute;
        pointer-events: none; z-index: 7;
        height: 2px;
        background: rgba(255, 255, 255, 0.85);
        box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.7);
        transform-origin: 0 50%;
      }
      .pso-paint-grad-handle {
        position: absolute;
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #ffaa00;
        border: 1px solid #14181d;
        pointer-events: none; z-index: 7;
        transform: translate(-50%, -50%);
      }
    `;
    document.head.appendChild(s);
  }

  // ---- state ---------------------------------------------------------
  const ui = {
    panel: null,           // root panel (the texture-panel) we extend
    bodyEl: null,          // panel body container (the active tab's content)
    tabsEl: null,          // panel tabs strip
    cursorEl: null,        // brush-radius overlay div
  };

  const state = {
    enabled: false,        // paint mode on/off
    activeTool: "brush",
    brushSize: DEFAULT_BRUSH_SIZE,
    opacity: DEFAULT_OPACITY,
    hardness: DEFAULT_HARDNESS,
    color: DEFAULT_COLOR,        // hex like #ff3344
    activeTile: null,            // tile_index currently being painted
    activeArchive: null,         // host archive path
    activeMaterialIds: [],       // every material_id bound to activeTile
    canvas: null,                // CPU-side <canvas> = composite output
    canvasW: 0,
    canvasH: 0,
    ctx: null,                   // 2D context of `canvas` (composite)
    threeTexture: null,          // THREE.CanvasTexture wrapping `canvas`
    originalTextures: new Map(), // material_id -> the THREE.Texture before
                                 //   we swapped in our CanvasTexture
    sourceImageBitmap: null,     // ImageBitmap of the source PNG (for Reset)
    undoStack: [],               // ImageData snapshots of the active layer
    redoStack: [],
    drag: { active: false, lastX: -1, lastY: -1 },
    pendingSave: false,

    // ---- v5: layer stack ------------------------------------------------
    // `layers` is an array of layer objects in BOTTOM->TOP order.
    // Each entry: { idx, name, visible, opacity, blend_mode, locked,
    //               canvas (HTMLCanvasElement, RGBA layer pixels),
    //               ctx (2D), maskCanvas?, maskCtx?, hasMask }
    // `activeIdx` is the layer.idx currently being painted (NOT the array
    // position; idx values are stable across reorders).
    layers: [],
    activeIdx: 0,
    nextLayerId: 1,              // monotonic ID for new layers
    paintMask: false,            // when true + layer.hasMask, paint goes
                                 // to mask channel as black/white
    suppressLoad: false,         // skip /api/paint/load fetch on next bind
    activeLayerSnapshot: null,   // ImageData captured at stroke start
    pendingAutoSave: 0,          // setTimeout id for layer-save throttle

    // ---- clone tool -----------------------------------------------------
    cloneSource: null,           // {x, y, layerIdx} or null until Alt+click
    cloneOffsetX: 0,             // src + offset = dst (set on first paint click)
    cloneOffsetY: 0,
    cloneSameLayer: true,        // toggle (true=copy active layer's pixels;
                                 //          false=copy from a chosen layer)
    cloneSrcLayerIdx: 0,         // when cloneSameLayer is false

    // ---- gradient tool --------------------------------------------------
    gradKind: "linear",
    gradStops: [
      { pos: 0.0, color: "#000000", alpha: 1.0 },
      { pos: 1.0, color: "#ffffff", alpha: 1.0 },
    ],
    gradDrag: null,              // {start:{x,y}, end:{x,y}} during drag
  };

  // ---- panel injection ----------------------------------------------
  function findPanel() {
    return document.getElementById("psoTexturePanel");
  }

  function ensureTabButton() {
    const panel = findPanel();
    if (!panel) return null;
    const tabs = panel.querySelector(".pso-tex-panel-tabs");
    if (!tabs) return null;
    let btn = tabs.querySelector('button[data-tab="paint"]');
    if (btn) return panel;
    btn = document.createElement("button");
    btn.dataset.tab = "paint";
    btn.title = "UV-aware brush — click + drag on the model";
    btn.textContent = "Paint";
    tabs.appendChild(btn);
    // Fold this newly-added tab into the texture panel's "more ▾" overflow
    // (2026-06-20) instead of widening the inline tab wall.
    if (typeof window.psoTexturePanelReflowTabs === "function") {
      window.psoTexturePanelReflowTabs();
    }
    return panel;
  }

  // texture_panel renders into the panel body via its own
  // renderActiveTab() dispatch. We can't call that — but we can listen
  // for clicks on our tab and take over the body.
  function wireTabSwitch() {
    const panel = findPanel();
    if (!panel) return;
    const tabs = panel.querySelector(".pso-tex-panel-tabs");
    if (!tabs) return;
    if (tabs.dataset.psoPaintWired === "1") return;
    tabs.dataset.psoPaintWired = "1";
    tabs.addEventListener("click", (ev) => {
      const t = ev.target;
      if (!(t instanceof HTMLButtonElement)) return;
      const which = t.getAttribute("data-tab");
      if (which === "paint") {
        // Active tab is now paint; render our panel.
        ev.preventDefault();
        ev.stopPropagation();
        // Flip the active class manually since we intercepted the event.
        tabs.querySelectorAll("button").forEach((b) => {
          b.classList.toggle("active", b === t);
        });
        renderPaintTab();
        // Auto-enable paint mode on first switch (user explicitly chose Paint).
        if (!state.enabled) togglePaintMode(true);
      } else {
        // Switched away — turn off paint mode so the model is
        // orbit-able again.
        if (state.enabled) togglePaintMode(false);
      }
    }, true);  // capture: run before texture_panel.js's own listener
  }

  // ---- tab body rendering -------------------------------------------
  function renderPaintTab() {
    const panel = findPanel();
    if (!panel) return;
    const body = panel.querySelector('[data-region="body"]');
    if (!body) return;
    ui.bodyEl = body;
    const list = (typeof window.psoListMeshTextures === "function")
      ? window.psoListMeshTextures()
      : [];
    if (!list.length) {
      body.innerHTML = `
        <div class="pso-paint-block">
          <div class="pso-paint-tip">
            No bound textures detected. Open a model with a paired texture
            archive (most BML/AFS weapons + enemies pair automatically).
          </div>
        </div>
      `;
      return;
    }
    // Auto-pick the first tile if none is active or the current one
    // has been unloaded.
    const activeStillValid = list.some((r) => r.tile_index === state.activeTile);
    const active = activeStillValid ? state.activeTile : list[0].tile_index;
    if (active !== state.activeTile) {
      bindActiveTile(active, list);
    }

    const arch = (typeof window.psoGetCurrentTextureArchive === "function")
      ? window.psoGetCurrentTextureArchive() : null;
    state.activeArchive = arch;
    const archShort = arch ? arch.split("/").pop() : "(no archive)";

    const toolBtns = TOOLS.map((tool) => {
      const [k, lbl] = TOOL_LABELS[tool];
      const cls = state.activeTool === tool ? "active" : "";
      return `<button class="${cls}" data-tool="${tool}" title="${lbl} (${k})">${k} ${lbl}</button>`;
    }).join("");

    const texList = list.map((row) => {
      const cls = row.tile_index === state.activeTile ? "active" : "";
      const dim = (row.width && row.height)
        ? `${row.width}×${row.height}` : "—";
      const url = `/api/tile_png/${encodeURIComponent(arch || "")}/${row.tile_index}?cb=${Date.now()}`;
      return `
        <div class="pso-paint-tex-row ${cls}" data-tile="${row.tile_index}" title="paint tile ${row.tile_index}">
          <img src="${escapeHtml(url)}" alt="tile ${row.tile_index}" />
          <div class="nm">tile ${row.tile_index} · ${dim}</div>
        </div>`;
    }).join("");

    body.innerHTML = `
      <div class="pso-paint-block">
        <div class="pso-paint-row">
          <label class="pso-paint-tip">
            <input type="checkbox" id="psoPaintEnable" ${state.enabled ? "checked" : ""} />
            Paint mode (LMB paints, RMB orbits)
          </label>
        </div>
        <div class="pso-paint-tools" data-region="tools">${toolBtns}</div>
        <div data-region="tool-options"></div>
        <div class="pso-paint-row">
          <label title="Brush radius in texture pixels">
            size
          </label>
          <div class="pso-paint-slider">
            <input type="range" id="psoPaintSize" min="1" max="128" step="1" value="${state.brushSize}" />
            <span class="num" id="psoPaintSizeNum">${state.brushSize}</span>
          </div>
        </div>
        <div class="pso-paint-row">
          <label>opacity</label>
          <div class="pso-paint-slider">
            <input type="range" id="psoPaintOpacity" min="0" max="100" step="1" value="${Math.round(state.opacity * 100)}" />
            <span class="num" id="psoPaintOpacityNum">${Math.round(state.opacity * 100)}</span>
          </div>
        </div>
        <div class="pso-paint-row">
          <label>hardness</label>
          <div class="pso-paint-slider">
            <input type="range" id="psoPaintHardness" min="0" max="100" step="1" value="${Math.round(state.hardness * 100)}" />
            <span class="num" id="psoPaintHardnessNum">${Math.round(state.hardness * 100)}</span>
          </div>
        </div>
        <div class="pso-paint-row">
          <label>color</label>
          <input type="color" id="psoPaintColor" value="${state.color}" />
          <span class="grow"></span>
          <button class="pso-paint-undo" title="Undo (Ctrl+Z)" data-act="undo">Undo</button>
          <button class="pso-paint-redo" title="Redo (Ctrl+Shift+Z)" data-act="redo">Redo</button>
        </div>
        <div class="pso-paint-active" data-region="active">
          <img id="psoPaintActiveThumb" alt="" />
          <div class="info">
            <div class="nm" id="psoPaintActiveName">—</div>
            <div class="meta" id="psoPaintActiveMeta">${escapeHtml(archShort)}</div>
          </div>
        </div>
        <div data-region="texlist">${texList}</div>
        <div class="pso-paint-layers-block">
          <div class="pso-paint-layers-head">
            <span class="grow">Layers</span>
            <button data-act="layer-add" title="Add layer">+</button>
            <button data-act="layer-dup" title="Duplicate active">⧉</button>
            <button data-act="layer-merge" title="Merge down">⤓</button>
            <button data-act="layer-del" title="Delete active">×</button>
          </div>
          <div class="pso-paint-layer-list" data-region="layers-list"></div>
        </div>
        <div class="pso-paint-actions">
          <button data-act="save" class="primary" title="save the layer stack to cache/painted_textures/">Save</button>
          <button data-act="reset" class="warning" title="revert active layer to source texture">Reset layer</button>
          <button data-act="build" class="primary" title="build & deploy the host archive with painted textures">Build &amp; Deploy</button>
          <button data-act="livetest" class="lt-live-button" title="stage the painted texture to cache/live_overrides/ for the combo ASI to pick up (Phase 2; consumer ASI ships separately)"><span class="lt-live-dot" aria-hidden="true"></span><span class="lt-live-label">Live test</span></button>
        </div>
        <div class="pso-paint-status" data-region="status">ready · keys: B brush, E eraser, I picker, F fill, S smear, T clone, G gradient · Ctrl+Z undo</div>
        <div class="pso-paint-tip">
          UV unwrap: raycast (face, uv) → pixel = (round(u·(W−1)), round((1−v)·(H−1))).
          Stamps are clipped at texture edges (no UV-seam wrap). Clone: Alt+click sets source.
        </div>
      </div>
    `;
    bindBodyEvents(body, list);
    refreshActiveBadge(list);
  }

  function bindBodyEvents(body, list) {
    // Tool buttons.
    body.querySelectorAll("[data-tool]").forEach((b) => {
      b.addEventListener("click", () => setTool(b.dataset.tool));
    });
    body.querySelectorAll("[data-tile]").forEach((row) => {
      row.addEventListener("click", () => {
        const ti = parseInt(row.dataset.tile, 10);
        bindActiveTile(ti, list);
        renderPaintTab();
      });
    });
    body.querySelector("#psoPaintEnable").addEventListener("change", (e) => {
      togglePaintMode(e.target.checked);
    });
    const sizeInp = body.querySelector("#psoPaintSize");
    sizeInp.addEventListener("input", (e) => {
      state.brushSize = parseInt(e.target.value, 10);
      body.querySelector("#psoPaintSizeNum").textContent = String(state.brushSize);
      updateCursor();
    });
    body.querySelector("#psoPaintOpacity").addEventListener("input", (e) => {
      state.opacity = parseInt(e.target.value, 10) / 100;
      body.querySelector("#psoPaintOpacityNum").textContent = e.target.value;
    });
    body.querySelector("#psoPaintHardness").addEventListener("input", (e) => {
      state.hardness = parseInt(e.target.value, 10) / 100;
      body.querySelector("#psoPaintHardnessNum").textContent = e.target.value;
    });
    body.querySelector("#psoPaintColor").addEventListener("input", (e) => {
      state.color = e.target.value;
    });
    // Buttons that aren't inside the layers list. We attach to all
    // [data-act] but exclude the per-layer-row ones (those live inside
    // the layers-list region and are wired by renderLayerList()).
    body.querySelectorAll("[data-act]").forEach((btn) => {
      if (btn.closest('[data-region="layers-list"]')) return;
      if (btn.tagName !== "BUTTON" && btn.tagName !== "A") return;
      btn.addEventListener("click", () => onAction(btn.dataset.act));
    });
    // Initial layer list + tool options paint.
    renderLayerList();
    renderToolOptions();
  }

  // Tool-options strip: shows clone-source toggle + gradient kind + stops.
  function renderToolOptions() {
    if (!ui.bodyEl) return;
    const host = ui.bodyEl.querySelector('[data-region="tool-options"]');
    if (!host) return;
    if (state.activeTool === "clone") {
      const layers = state.layers
        .filter((L) => L.idx !== state.activeIdx)
        .map((L) => `<option value="${L.idx}"${state.cloneSrcLayerIdx === L.idx ? " selected" : ""}>${escapeHtml(L.name)}</option>`)
        .join("");
      host.innerHTML = `
        <div class="pso-paint-tool-options">
          <span class="lbl">Clone source:</span>
          <label><input type="radio" name="cloneSrc" value="same" ${state.cloneSameLayer ? "checked" : ""}> active layer</label>
          <label><input type="radio" name="cloneSrc" value="other" ${!state.cloneSameLayer ? "checked" : ""}> other:</label>
          <select data-clone-src ${state.cloneSameLayer ? "disabled" : ""}>${layers || '<option value="">(none)</option>'}</select>
          <span class="lbl">${state.cloneSource ? `src @ (${state.cloneSource.x}, ${state.cloneSource.y})` : "Alt+click to set source"}</span>
        </div>
      `;
      host.querySelectorAll('input[name="cloneSrc"]').forEach((r) => {
        r.addEventListener("change", () => {
          state.cloneSameLayer = host.querySelector('input[value="same"]').checked;
          renderToolOptions();
        });
      });
      const sel = host.querySelector("[data-clone-src]");
      if (sel) sel.addEventListener("change", (e) => {
        state.cloneSrcLayerIdx = parseInt(e.target.value, 10);
      });
      return;
    }
    if (state.activeTool === "gradient") {
      const kindOpts = GRADIENT_KINDS.map((k) =>
        `<option value="${k}"${state.gradKind === k ? " selected" : ""}>${k}</option>`).join("");
      const stopsHtml = state.gradStops.map((s, i) => `
        <div class="pso-paint-grad-stop" data-stop-idx="${i}">
          <input type="range" min="0" max="1000" step="1" value="${Math.round(s.pos * 1000)}" data-act="pos" />
          <input type="color" value="${s.color}" data-act="color" />
          <input type="number" min="0" max="100" step="1" value="${Math.round((s.alpha != null ? s.alpha : 1) * 100)}" data-act="alpha" title="alpha %" style="width:42px;" />
          <button data-act="del" ${state.gradStops.length <= 2 ? "disabled" : ""}>×</button>
        </div>
      `).join("");
      host.innerHTML = `
        <div class="pso-paint-tool-options">
          <span class="lbl">Gradient:</span>
          <select data-grad-kind>${kindOpts}</select>
          <button data-act="grad-add-stop">+ stop</button>
        </div>
        <div class="pso-paint-grad-stops">${stopsHtml}</div>
      `;
      const sel = host.querySelector("[data-grad-kind]");
      if (sel) sel.addEventListener("change", (e) => {
        state.gradKind = e.target.value;
      });
      host.querySelectorAll(".pso-paint-grad-stop").forEach((row) => {
        const i = parseInt(row.getAttribute("data-stop-idx"), 10);
        row.querySelector('[data-act="pos"]').addEventListener("input", (e) => {
          state.gradStops[i].pos = parseInt(e.target.value, 10) / 1000;
        });
        row.querySelector('[data-act="color"]').addEventListener("input", (e) => {
          state.gradStops[i].color = e.target.value;
        });
        row.querySelector('[data-act="alpha"]').addEventListener("input", (e) => {
          state.gradStops[i].alpha = parseInt(e.target.value, 10) / 100;
        });
        const del = row.querySelector('[data-act="del"]');
        if (del && !del.disabled) {
          del.addEventListener("click", () => {
            state.gradStops.splice(i, 1);
            renderToolOptions();
          });
        }
      });
      const addBtn = host.querySelector('[data-act="grad-add-stop"]');
      if (addBtn) addBtn.addEventListener("click", () => {
        state.gradStops.push({ pos: 0.5, color: "#888888", alpha: 1.0 });
        state.gradStops.sort((a, b) => a.pos - b.pos);
        renderToolOptions();
      });
      return;
    }
    // Default: hide the strip.
    host.innerHTML = "";
  }

  function refreshActiveBadge(list) {
    if (!ui.bodyEl) return;
    const row = (list || []).find((r) => r.tile_index === state.activeTile);
    if (!row) return;
    const arch = state.activeArchive || "";
    const url = `/api/tile_png/${encodeURIComponent(arch)}/${row.tile_index}?cb=${Date.now()}`;
    const thumb = ui.bodyEl.querySelector("#psoPaintActiveThumb");
    if (thumb) thumb.src = url;
    const nm = ui.bodyEl.querySelector("#psoPaintActiveName");
    if (nm) nm.textContent = `tile ${row.tile_index} (${row.width || "?"}×${row.height || "?"})`;
  }

  // ---- layer helpers (v5) -------------------------------------------
  // Build a fresh blank layer with a canvas + ctx the same dimensions
  // as the painted texture.
  function makeLayer({ idx, name, visible, opacity, blend_mode, locked, hasMask, maskValueFill }) {
    const cv = document.createElement("canvas");
    cv.width = state.canvasW;
    cv.height = state.canvasH;
    const ctx = cv.getContext("2d", { willReadFrequently: true });
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, cv.width, cv.height);
    const layer = {
      idx,
      name: name || `Layer ${idx}`,
      visible: visible !== false,
      opacity: typeof opacity === "number" ? opacity : 1.0,
      blend_mode: SUPPORTED_BLEND_MODES.includes(blend_mode) ? blend_mode : "normal",
      locked: !!locked,
      hasMask: !!hasMask,
      canvas: cv,
      ctx,
      maskCanvas: null,
      maskCtx: null,
      dirty: true,
    };
    if (hasMask) attachMaskCanvas(layer, !!maskValueFill);
    return layer;
  }

  // Allocate the per-layer mask canvas. ``fillWhite`` (default true) means
  // the layer is initially fully revealed. Set to false to start hidden.
  function attachMaskCanvas(layer, fillWhite) {
    const mc = document.createElement("canvas");
    mc.width = state.canvasW;
    mc.height = state.canvasH;
    const ctx = mc.getContext("2d", { willReadFrequently: true });
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = (fillWhite !== false) ? "#ffffff" : "#000000";
    ctx.fillRect(0, 0, mc.width, mc.height);
    layer.maskCanvas = mc;
    layer.maskCtx = ctx;
    layer.hasMask = true;
  }

  function getLayer(idx) {
    return state.layers.find((L) => L.idx === idx) || null;
  }

  function getActiveLayer() {
    return getLayer(state.activeIdx);
  }

  // The "draw target" is the canvas whose ctx receives paint primitives.
  // This is normally the active layer's RGBA canvas, but flips to the mask
  // canvas when state.paintMask is on (and the layer has a mask).
  function getActiveDrawTarget() {
    const L = getActiveLayer();
    if (!L) return null;
    if (state.paintMask && L.hasMask && L.maskCtx) {
      return { layer: L, canvas: L.maskCanvas, ctx: L.maskCtx, isMask: true };
    }
    return { layer: L, canvas: L.canvas, ctx: L.ctx, isMask: false };
  }

  // Composite all visible layers into state.canvas (the THREE.CanvasTexture
  // backing buffer). Equivalent to formats/paint.py::composite_layers.
  // The very first visible layer is always drawn with source-over so a
  // bottom layer with a "fancy" blend_mode doesn't render onto the empty
  // composite via an undefined op.
  function recompositeNow() {
    if (!state.canvas || !state.ctx) return;
    const W = state.canvasW, H = state.canvasH;
    const ctx = state.ctx;
    ctx.save();
    ctx.globalCompositeOperation = "source-over";
    ctx.clearRect(0, 0, W, H);
    let firstDrawn = false;
    for (const L of state.layers) {
      if (!L.visible || L.opacity <= 0) continue;
      // Apply mask first (only the R channel of the mask is used; we
      // composite the mask into a temp canvas to multiply alpha).
      let layerSrc = L.canvas;
      if (L.hasMask && L.maskCanvas) {
        layerSrc = applyMaskToTempCanvas(L);
      }
      ctx.globalAlpha = L.opacity;
      ctx.globalCompositeOperation = firstDrawn
        ? canvasBlendOpFor(L.blend_mode)
        : "source-over";
      ctx.drawImage(layerSrc, 0, 0);
      firstDrawn = true;
    }
    ctx.globalAlpha = 1.0;
    ctx.globalCompositeOperation = "source-over";
    ctx.restore();
    if (state.threeTexture) state.threeTexture.needsUpdate = true;
    renderForce();
  }

  // Map our blend-mode strings to canvas globalCompositeOperation values.
  // Photoshop "normal" / "multiply" / "screen" / "overlay" map cleanly.
  function canvasBlendOpFor(mode) {
    switch (mode) {
      case "multiply": return "multiply";
      case "screen":   return "screen";
      case "overlay":  return "overlay";
      default:         return "source-over";
    }
  }

  // Build a temporary canvas that holds (layer.rgba * layer.mask.r) so the
  // outer composite step can use a regular drawImage with the chosen
  // blend mode without separately tracking the mask.
  let _maskTempCanvas = null;
  let _maskTempCtx = null;
  function applyMaskToTempCanvas(layer) {
    if (!_maskTempCanvas) {
      _maskTempCanvas = document.createElement("canvas");
    }
    if (_maskTempCanvas.width !== state.canvasW || _maskTempCanvas.height !== state.canvasH) {
      _maskTempCanvas.width = state.canvasW;
      _maskTempCanvas.height = state.canvasH;
      _maskTempCtx = _maskTempCanvas.getContext("2d");
    }
    if (!_maskTempCtx) {
      _maskTempCtx = _maskTempCanvas.getContext("2d");
    }
    const tctx = _maskTempCtx;
    tctx.save();
    tctx.globalCompositeOperation = "source-over";
    tctx.clearRect(0, 0, state.canvasW, state.canvasH);
    tctx.drawImage(layer.canvas, 0, 0);
    // destination-in keeps only where the mask is opaque (the mask is
    // pre-painted as solid grayscale in white/black; alpha=255 always).
    // We treat the mask's RGB *brightness* as the alpha multiplier by
    // converting it to a black->transparent mask first via an offscreen
    // canvas rebuild on the fly. Since the mask canvas is grayscale
    // (R=G=B), we can shortcut: copy the mask, set composite to
    // "luminosity"-like behavior by reading per-pixel.
    const maskImg = layer.maskCtx.getImageData(0, 0, state.canvasW, state.canvasH);
    const mb = maskImg.data;
    for (let i = 0; i < mb.length; i += 4) {
      const m = mb[i];                     // R channel = mask value
      mb[i + 0] = 0;
      mb[i + 1] = 0;
      mb[i + 2] = 0;
      mb[i + 3] = m;
    }
    tctx.globalCompositeOperation = "destination-in";
    tctx.putImageData(maskImg, 0, 0);
    tctx.restore();
    return _maskTempCanvas;
  }

  // Detach all of state.layers (release canvas refs). Called when binding
  // a new tile.
  function clearLayers() {
    state.layers = [];
    state.activeIdx = 0;
    state.nextLayerId = 1;
    state.paintMask = false;
  }

  // Find the smallest unused layer.idx, used when adding a new layer.
  function nextFreeLayerIdx() {
    const used = new Set(state.layers.map((L) => L.idx));
    let i = 0;
    while (used.has(i) && i < MAX_LAYERS) i++;
    return i;
  }

  // ---- tile binding (load source, create CanvasTexture) -------------
  async function bindActiveTile(tileIdx, list) {
    state.activeTile = tileIdx;
    const arch = (typeof window.psoGetCurrentTextureArchive === "function")
      ? window.psoGetCurrentTextureArchive() : null;
    state.activeArchive = arch;
    const row = (list || []).find((r) => r.tile_index === tileIdx);
    state.activeMaterialIds = (row && row.material_ids) ? row.material_ids.slice() : [];
    if (!arch) return;

    // Load the source PNG into an ImageBitmap.
    const url = `/api/tile_png/${encodeURIComponent(arch)}/${tileIdx}?cb=${Date.now()}`;
    // Wave 7: lifecycle-aware fetch; rapid-clicks abort prior tile pulls.
    const paintF = (window.psoAssetLifecycle && window.psoAssetLifecycle.fetchAsset) || fetch;
    const paintAbort = (window.psoAssetLifecycle && window.psoAssetLifecycle.isAbort) || (() => false);
    let bmp;
    try {
      const resp = await paintF(url);
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      const blob = await resp.blob();
      bmp = await createImageBitmap(blob);
    } catch (e) {
      if (paintAbort(e)) return;
      setStatus("err", `tile ${tileIdx} fetch failed: ${e.message || e}`);
      return;
    }
    state.sourceImageBitmap = bmp;
    state.canvasW = bmp.width;
    state.canvasH = bmp.height;
    if (!state.canvas) {
      state.canvas = document.createElement("canvas");
    }
    state.canvas.width = bmp.width;
    state.canvas.height = bmp.height;
    state.ctx = state.canvas.getContext("2d", { willReadFrequently: true });
    state.ctx.imageSmoothingEnabled = false;
    state.ctx.clearRect(0, 0, bmp.width, bmp.height);

    // Reset the layer stack. We'll re-populate from /api/paint/load (when a
    // saved stack exists) or seed a single layer from the source PNG.
    clearLayers();

    let loadedFromServer = false;
    if (!state.suppressLoad) {
      try {
        const lr = await paintF(
          `/api/paint/load?model_path=${encodeURIComponent(activeHostName())}` +
          `&inner=${encodeURIComponent(activeInnerName())}`
        );
        if (lr.ok) {
          const ldata = await lr.json();
          if (ldata && ldata.manifest && Array.isArray(ldata.layers) && ldata.layers.length) {
            await applyLoadedLayerStack(ldata);
            loadedFromServer = true;
          }
        }
      } catch (e) {
        console.warn("[paint] /api/paint/load failed; seeding fresh:", e);
      }
    }
    state.suppressLoad = false;

    if (!loadedFromServer) {
      // Seed a single Background layer from the source PNG.
      const layer0 = makeLayer({ idx: 0, name: "Background" });
      layer0.ctx.drawImage(bmp, 0, 0);
      state.layers.push(layer0);
      state.activeIdx = 0;
      state.nextLayerId = 1;
    }

    // Build / refresh the THREE.CanvasTexture and bind to all matching
    // materials. We snapshot the original texture so Reset / paint-mode
    // off can restore it.
    const THREE = window.THREE;
    if (THREE && state.threeTexture) {
      try { state.threeTexture.dispose(); } catch {}
    }
    state.threeTexture = new THREE.CanvasTexture(state.canvas);
    state.threeTexture.colorSpace = THREE.SRGBColorSpace;
    state.threeTexture.anisotropy = 4;
    state.threeTexture.wrapS = THREE.RepeatWrapping;
    state.threeTexture.wrapT = THREE.RepeatWrapping;
    state.threeTexture.needsUpdate = true;
    if (typeof window.psoSetMaterialTexture === "function") {
      for (const mid of state.activeMaterialIds) {
        // Stash the original on first bind so Reset can restore it.
        if (!state.originalTextures.has(mid)) {
          const orig = (typeof window.psoGetMaterialTexture === "function")
            ? window.psoGetMaterialTexture(mid) : null;
          if (orig) state.originalTextures.set(mid, orig);
        }
        window.psoSetMaterialTexture(mid, state.threeTexture);
      }
    }

    // Reset undo/redo for the new tile.
    state.undoStack = [];
    state.redoStack = [];
    pushUndo();  // initial snapshot so a Reset works after the first stroke
    recompositeNow();
    setStatus(
      "ok",
      `bound tile ${tileIdx} (${bmp.width}×${bmp.height}, ${state.layers.length} layer${state.layers.length === 1 ? "" : "s"})`,
    );
  }

  // Decode a server-supplied layer stack (manifest + base64 PNGs) into our
  // internal layer canvases. Honors visible / opacity / blend_mode / mask.
  async function applyLoadedLayerStack(payload) {
    const m = payload.manifest;
    const layers = payload.layers || [];
    // Honor manifest.width/height if present — the canvas dims should
    // already be set from the source PNG, but trust the manifest if it
    // matches (or warn otherwise).
    if (m && m.width && m.height && (m.width !== state.canvasW || m.height !== state.canvasH)) {
      console.warn(
        "[paint] manifest dim", m.width, "x", m.height,
        "differs from source", state.canvasW, "x", state.canvasH,
        "— using source",
      );
    }
    // Build layers in manifest order.
    const byIdx = new Map();
    for (const L of layers) byIdx.set(L.idx, L);
    let highestIdx = -1;
    for (const layerMeta of m.layers) {
      const idx = layerMeta.idx;
      if (idx > highestIdx) highestIdx = idx;
      const layer = makeLayer({
        idx,
        name: layerMeta.name,
        visible: layerMeta.visible,
        opacity: layerMeta.opacity,
        blend_mode: layerMeta.blend_mode,
        locked: layerMeta.locked,
        hasMask: false,  // will attach below if mask_b64 present
      });
      const data = byIdx.get(idx);
      if (data && data.png_b64) {
        try {
          const blob = await b64ToBlob(data.png_b64, "image/png");
          const bmp2 = await createImageBitmap(blob);
          layer.ctx.clearRect(0, 0, layer.canvas.width, layer.canvas.height);
          layer.ctx.drawImage(bmp2, 0, 0);
        } catch (e) {
          console.warn("[paint] failed to decode layer", idx, e);
        }
      }
      if (data && data.mask_b64 && layerMeta.has_mask) {
        try {
          const blob = await b64ToBlob(data.mask_b64, "image/png");
          const bmp3 = await createImageBitmap(blob);
          attachMaskCanvas(layer, true);
          layer.maskCtx.clearRect(0, 0, layer.maskCanvas.width, layer.maskCanvas.height);
          layer.maskCtx.drawImage(bmp3, 0, 0);
        } catch (e) {
          console.warn("[paint] failed to decode mask for layer", idx, e);
        }
      }
      state.layers.push(layer);
    }
    state.activeIdx = (typeof m.active === "number") ? m.active : (state.layers[0] && state.layers[0].idx) || 0;
    state.nextLayerId = highestIdx + 1;
  }

  function b64ToBlob(b64, mime) {
    const bin = atob(b64);
    const len = bin.length;
    const buf = new Uint8Array(len);
    for (let i = 0; i < len; i++) buf[i] = bin.charCodeAt(i);
    return Promise.resolve(new Blob([buf], { type: mime || "application/octet-stream" }));
  }

  // ---- paint mode toggle --------------------------------------------
  function togglePaintMode(on) {
    state.enabled = !!on;
    document.body.classList.toggle("pso-paint-mode-on", state.enabled);
    if (state.enabled) {
      attachCanvasListeners();
      ensureCursor();
    } else {
      detachCanvasListeners();
      removeCursor();
    }
    const enableInp = ui.bodyEl ? ui.bodyEl.querySelector("#psoPaintEnable") : null;
    if (enableInp) enableInp.checked = state.enabled;
  }

  function setTool(tool) {
    if (TOOLS.indexOf(tool) < 0) return;
    state.activeTool = tool;
    if (ui.bodyEl) {
      ui.bodyEl.querySelectorAll("[data-tool]").forEach((b) => {
        b.classList.toggle("active", b.dataset.tool === tool);
      });
      renderToolOptions();
    }
    // Tool change cancels in-progress drags + clone/grad state.
    if (state.gradDrag) {
      removeGradPreview();
      state.gradDrag = null;
    }
  }

  function setStatus(cls, msg) {
    if (!ui.bodyEl) return;
    const el = ui.bodyEl.querySelector('[data-region="status"]');
    if (!el) return;
    el.textContent = msg;
    el.className = "pso-paint-status " + cls;
  }

  // ---- canvas pointer plumbing --------------------------------------
  let _attachedCanvas = null;
  function attachCanvasListeners() {
    const cv = (typeof window.psoGetCanvas === "function")
      ? window.psoGetCanvas() : null;
    if (!cv) return;
    if (_attachedCanvas === cv) return;
    detachCanvasListeners();
    _attachedCanvas = cv;
    // Capturing phase, swallow LMB if paint is on.
    cv.addEventListener("pointerdown", onCanvasPointerDown, true);
    cv.addEventListener("pointermove", onCanvasPointerMove, true);
    cv.addEventListener("pointerup",   onCanvasPointerUp, true);
    cv.addEventListener("pointercancel", onCanvasPointerUp, true);
    cv.addEventListener("contextmenu", onCanvasContextMenu, true);
    cv.addEventListener("wheel", onCanvasWheel, { capture: true, passive: false });
    cv.addEventListener("pointerleave", onPointerLeave, true);
  }
  function detachCanvasListeners() {
    const cv = _attachedCanvas;
    if (!cv) return;
    cv.removeEventListener("pointerdown", onCanvasPointerDown, true);
    cv.removeEventListener("pointermove", onCanvasPointerMove, true);
    cv.removeEventListener("pointerup",   onCanvasPointerUp, true);
    cv.removeEventListener("pointercancel", onCanvasPointerUp, true);
    cv.removeEventListener("contextmenu", onCanvasContextMenu, true);
    cv.removeEventListener("wheel", onCanvasWheel, { capture: true });
    cv.removeEventListener("pointerleave", onPointerLeave, true);
    _attachedCanvas = null;
  }

  function onCanvasContextMenu(ev) {
    if (!state.enabled) return;
    ev.preventDefault();  // RMB orbits — block the browser menu
  }

  function onCanvasPointerDown(ev) {
    if (!state.enabled) return;
    if (ev.button !== 0) return;  // only LMB; RMB falls through to orbit
    ev.stopPropagation();
    ev.preventDefault();

    // Clone tool: Alt+click sets the source pixel and exits early.
    if (state.activeTool === "clone" && (ev.altKey || ev.metaKey)) {
      const hit = raycastMesh(ev);
      if (hit && hitIsActive(hit) && hit.uv) {
        const [px, py] = uvToPixel(hit.uv.x, hit.uv.y, state.canvasW, state.canvasH);
        setCloneSourceFromEvent(ev, px, py);
        state.cloneOffsetX = null;
        state.cloneOffsetY = null;
        moveClonePip(ev);
        setStatus("ok", `clone source @ (${px}, ${py})`);
      }
      try { ev.target.setPointerCapture(ev.pointerId); } catch {}
      return;
    }

    // Gradient tool: pointerdown starts a drag that captures start point;
    // pointerup commits.
    if (state.activeTool === "gradient") {
      const hit = raycastMesh(ev);
      if (!hit || !hitIsActive(hit) || !hit.uv) return;
      const [px, py] = uvToPixel(hit.uv.x, hit.uv.y, state.canvasW, state.canvasH);
      pushUndo();
      state.redoStack.length = 0;
      state.gradDrag = { startTex: { x: px, y: py }, endTex: { x: px, y: py },
                        startScreen: { x: ev.clientX, y: ev.clientY },
                        endScreen: { x: ev.clientX, y: ev.clientY } };
      state.drag.active = true;
      ensureGradPreview();
      updateGradPreview();
      try { ev.target.setPointerCapture(ev.pointerId); } catch {}
      return;
    }

    // Capture an explicit per-stroke snapshot for the active layer so we
    // can roll back individual strokes (separate from the layer-stack
    // composite). pushUndo() also writes to the cross-tool undo bus.
    state.drag.active = true;
    state.drag.lastX = -1;
    state.drag.lastY = -1;
    pushUndo();
    state.redoStack.length = 0;

    // Reset clone offset on the FIRST paint click of a new stroke (so a
    // subsequent stroke at a new dest re-locks the offset relative to
    // the same source).
    if (state.activeTool === "clone") {
      state.cloneOffsetX = null;
      state.cloneOffsetY = null;
    }

    handlePaintAt(ev);
    try { ev.target.setPointerCapture(ev.pointerId); } catch {}
  }

  function onCanvasPointerMove(ev) {
    if (!state.enabled) return;
    moveCursor(ev);
    moveClonePip(ev);
    if (!state.drag.active) return;
    ev.stopPropagation();
    if (state.activeTool === "gradient" && state.gradDrag) {
      const hit = raycastMesh(ev);
      if (hit && hitIsActive(hit) && hit.uv) {
        const [px, py] = uvToPixel(hit.uv.x, hit.uv.y, state.canvasW, state.canvasH);
        state.gradDrag.endTex = { x: px, y: py };
        state.gradDrag.endScreen = { x: ev.clientX, y: ev.clientY };
        updateGradPreview();
      }
      return;
    }
    handlePaintAt(ev);
  }

  function onCanvasPointerUp(ev) {
    if (!state.enabled) return;
    if (state.drag.active) {
      state.drag.active = false;
      ev.stopPropagation();
      try { ev.target.releasePointerCapture(ev.pointerId); } catch {}
      if (state.activeTool === "gradient" && state.gradDrag) {
        // Commit the gradient.
        const g = state.gradDrag;
        renderGradient(g.startTex.x, g.startTex.y, g.endTex.x, g.endTex.y);
        state.gradDrag = null;
        removeGradPreview();
        scheduleAutoSave();
      } else {
        // End of stroke — re-render once more so the GPU has the final
        // CanvasTexture upload, then queue an auto-save.
        recompositeNow();
        scheduleAutoSave();
      }
    }
  }

  function onPointerLeave() {
    if (state.drag.active) {
      state.drag.active = false;
      recompositeNow();
    }
    removeCursor();
    removeGradPreview();
  }

  function onCanvasWheel(ev) {
    if (!state.enabled) return;
    // Adjust brush size — block camera zoom.
    ev.preventDefault();
    ev.stopPropagation();
    const delta = ev.deltaY > 0 ? -2 : +2;
    state.brushSize = Math.max(1, Math.min(128, state.brushSize + delta));
    if (ui.bodyEl) {
      const sl = ui.bodyEl.querySelector("#psoPaintSize");
      const num = ui.bodyEl.querySelector("#psoPaintSizeNum");
      if (sl) sl.value = state.brushSize;
      if (num) num.textContent = String(state.brushSize);
    }
    updateCursor();
  }

  // ---- raycast → UV → pixel -----------------------------------------
  let _raycaster = null;
  function ensureRaycaster() {
    const THREE = window.THREE;
    if (!_raycaster) _raycaster = new THREE.Raycaster();
    return _raycaster;
  }

  function ndcFromEvent(ev, canvas) {
    const rect = canvas.getBoundingClientRect();
    const x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -(((ev.clientY - rect.top) / rect.height) * 2 - 1);
    return { x, y };
  }

  function raycastMesh(ev) {
    const cv = (typeof window.psoGetCanvas === "function")
      ? window.psoGetCanvas() : null;
    const cam = (typeof window.psoGetCamera === "function")
      ? window.psoGetCamera() : null;
    const grp = (typeof window.psoGetMeshGroup === "function")
      ? window.psoGetMeshGroup() : null;
    if (!cv || !cam || !grp) return null;
    const ndc = ndcFromEvent(ev, cv);
    const rc = ensureRaycaster();
    rc.setFromCamera(ndc, cam);
    const meshes = [];
    grp.traverse((c) => { if (c.isMesh) meshes.push(c); });
    if (!meshes.length) return null;
    const hits = rc.intersectObjects(meshes, false);
    if (!hits.length) return null;
    return hits[0]; // {object, face, uv, point, ...}
  }

  // fix/tooltabs — resolve the material_id a raycast hit landed on. The
  // legacy single-material path stores it on hit.object.userData.materialId.
  // The psov2 SkinnedMesh is ONE multi-material mesh: material identity is
  // PER-FACE (hit.face.materialIndex) and the slot->material_id map lives on
  // userData.materialGroups. Resolve via the face's materialIndex first, then
  // fall back to the bare userData.materialId so both paths work.
  function hitMaterialId(hit) {
    if (!hit || !hit.object) return null;
    const ud = hit.object.userData || {};
    const groups = ud.materialGroups;
    if (Array.isArray(groups) && hit.face && typeof hit.face.materialIndex === "number") {
      const g = groups[hit.face.materialIndex | 0];
      if (g) return g.materialId | 0;
    }
    if (typeof ud.materialId === "number") return ud.materialId | 0;
    return null;
  }

  // Is the hit on a material we're actively painting? null id (untagged
  // single-material mesh) is allowed through only when exactly one material
  // is active — the historical single-tile behaviour.
  function hitIsActive(hit) {
    const mid = hitMaterialId(hit);
    if (mid === null) {
      return state.activeMaterialIds.length === 1;
    }
    return state.activeMaterialIds.indexOf(mid) >= 0;
  }

  function uvToPixel(u, v, w, h) {
    if (u < 0) u = 0; else if (u > 1) u = 1;
    if (v < 0) v = 0; else if (v > 1) v = 1;
    return [Math.round(u * (w - 1)), Math.round((1 - v) * (h - 1))];
  }

  // Parse #rrggbb to [r, g, b].
  function parseColor(hex) {
    const m = /^#?([0-9a-f]{6})$/i.exec(hex || "");
    if (!m) return [255, 0, 0];
    const v = parseInt(m[1], 16);
    return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
  }

  // ---- drawing primitives (CPU canvas, JS port of formats/paint.py) -
  // All primitives target the ACTIVE LAYER's ctx (or its mask ctx when
  // state.paintMask is on). After mutation they call recompositeNow() to
  // refresh the THREE.CanvasTexture binding.
  function applyTool(px, py) {
    const tgt = getActiveDrawTarget();
    if (!tgt) return;
    if (tgt.layer.locked) {
      setStatus("err", "active layer is locked");
      return;
    }
    const tool = state.activeTool;
    if (tool === "brush" || tool === "eraser") {
      stampGaussian(tgt.ctx, px, py, tool === "eraser", tgt.isMask);
    } else if (tool === "picker") {
      pickColorAt(px, py);
    } else if (tool === "fill") {
      floodFillAt(tgt.ctx, px, py, tgt.isMask);
    } else if (tool === "smear") {
      smearAt(tgt.ctx, px, py);
    } else if (tool === "clone") {
      cloneAt(tgt, px, py);
    }
    // Gradient is dispatched separately on pointerup, not per move.
    recompositeNow();
  }

  // Gaussian-falloff stamp drawn directly with canvas radial gradient.
  // ``ctx`` is the layer/mask context being painted into. ``isMask`` flips
  // the brush color to black/white (mask paint convention).
  function stampGaussian(ctx, cx, cy, erase, isMask) {
    const r = state.brushSize;
    if (r <= 0) return;
    let cr, cg, cb;
    if (isMask) {
      // On mask: paint white to reveal, black to hide. The eraser tool
      // becomes "paint black" (hide).
      const v = erase ? 0 : 255;
      cr = cg = cb = v;
    } else {
      const c = parseColor(state.color);
      cr = c[0]; cg = c[1]; cb = c[2];
    }
    const opacity = state.opacity;
    const hard = state.hardness;

    if (erase && !isMask) {
      ctx.save();
      ctx.globalCompositeOperation = "destination-out";
      const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
      // Soft edge: alpha drops from opacity at centre to 0 at rim.
      const inner = opacity;
      const outer = 0;
      grad.addColorStop(0, `rgba(0,0,0,${inner})`);
      grad.addColorStop(Math.max(0, hard), `rgba(0,0,0,${inner})`);
      grad.addColorStop(1, `rgba(0,0,0,${outer})`);
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      return;
    }

    ctx.save();
    ctx.globalCompositeOperation = "source-over";
    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
    grad.addColorStop(0, `rgba(${cr},${cg},${cb},${opacity})`);
    // hardness=1 -> hold full alpha to the rim; hardness=0 -> linear falloff
    grad.addColorStop(Math.min(0.999, Math.max(0, hard)), `rgba(${cr},${cg},${cb},${opacity})`);
    grad.addColorStop(1, `rgba(${cr},${cg},${cb},0)`);
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  function pickColorAt(px, py) {
    // Picker reads from the COMPOSITE (state.ctx) — what the user sees.
    const W = state.canvasW, H = state.canvasH;
    if (px < 0 || py < 0 || px >= W || py >= H) return;
    const data = state.ctx.getImageData(px, py, 1, 1).data;
    const r = data[0], g = data[1], b = data[2];
    const hex = "#" + [r, g, b].map((v) => v.toString(16).padStart(2, "0")).join("");
    state.color = hex;
    if (ui.bodyEl) {
      const inp = ui.bodyEl.querySelector("#psoPaintColor");
      if (inp) inp.value = hex;
    }
    setStatus("ok", `picked color ${hex}`);
  }

  function floodFillAt(ctx, px, py, isMask) {
    const W = state.canvasW, H = state.canvasH;
    if (px < 0 || py < 0 || px >= W || py >= H) return;
    const img = ctx.getImageData(0, 0, W, H);
    const buf = img.data;
    const idx0 = (py * W + px) * 4;
    const sr = buf[idx0], sg = buf[idx0 + 1], sb = buf[idx0 + 2], sa = buf[idx0 + 3];
    let fr, fg, fb, fa;
    if (isMask) {
      const v = state.color === MASK_BLACK ? 0 : 255;
      fr = fg = fb = v;
      fa = 255;
    } else {
      const c = parseColor(state.color);
      fr = c[0]; fg = c[1]; fb = c[2];
      fa = Math.round(state.opacity * 255);
    }
    if (sr === fr && sg === fg && sb === fb && sa === fa) return;
    const tol = 8;
    const matches = (i) => (
      Math.abs(buf[i] - sr) <= tol &&
      Math.abs(buf[i + 1] - sg) <= tol &&
      Math.abs(buf[i + 2] - sb) <= tol &&
      Math.abs(buf[i + 3] - sa) <= tol
    );
    const stack = [[px, py]];
    while (stack.length) {
      const [x, y] = stack.pop();
      let lx = x;
      while (lx >= 0 && matches((y * W + lx) * 4)) lx--;
      lx++;
      let rx = x;
      while (rx < W && matches((y * W + rx) * 4)) rx++;
      rx--;
      let prevA = false, prevB = false;
      for (let cx = lx; cx <= rx; cx++) {
        const ci = (y * W + cx) * 4;
        buf[ci] = fr; buf[ci + 1] = fg; buf[ci + 2] = fb; buf[ci + 3] = fa;
        if (y > 0) {
          if (matches(((y - 1) * W + cx) * 4)) {
            if (!prevA) { stack.push([cx, y - 1]); prevA = true; }
          } else prevA = false;
        }
        if (y + 1 < H) {
          if (matches(((y + 1) * W + cx) * 4)) {
            if (!prevB) { stack.push([cx, y + 1]); prevB = true; }
          } else prevB = false;
        }
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  function smearAt(ctx, px, py) {
    const W = state.canvasW, H = state.canvasH;
    const r = state.brushSize;
    const lx = state.drag.lastX, ly = state.drag.lastY;
    state.drag.lastX = px;
    state.drag.lastY = py;
    if (lx < 0 || ly < 0) return;  // first stroke point — no direction yet
    const dx = px - lx, dy = py - ly;
    if (dx === 0 && dy === 0) return;
    const x0 = Math.max(0, px - r);
    const y0 = Math.max(0, py - r);
    const x1 = Math.min(W, px + r + 1);
    const y1 = Math.min(H, py + r + 1);
    if (x0 >= x1 || y0 >= y1) return;
    const img = ctx.getImageData(0, 0, W, H);
    const buf = img.data;
    const r2 = r * r;
    const strength = state.opacity;
    // Snapshot just the source rect we read from (could span any of the
    // image since dx/dy can be large) — easier to just snapshot whole
    // image once.
    const snap = new Uint8ClampedArray(buf);
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const ddx = x - px, ddy = y - py;
        if (ddx * ddx + ddy * ddy > r2) continue;
        const sx = x - dx, sy = y - dy;
        if (sx < 0 || sx >= W || sy < 0 || sy >= H) continue;
        const di = (y * W + x) * 4;
        const si = (sy * W + sx) * 4;
        for (let c = 0; c < 4; c++) {
          buf[di + c] = Math.round(buf[di + c] + (snap[si + c] - buf[di + c]) * strength);
        }
      }
    }
    ctx.putImageData(img, x0, y0, 0, 0, x1 - x0, y1 - y0);
  }

  // Clone stamp: copies pixels from a source layer (could be the same
  // layer) at an offset locked at first paint click. Source sample is
  // cx + dx - cloneOffsetX, cy + dy - cloneOffsetY.
  function cloneAt(tgt, px, py) {
    if (!state.cloneSource) {
      setStatus("err", "set clone source first (Alt+click)");
      return;
    }
    if (state.cloneOffsetX === null || state.cloneOffsetY === null) {
      // First paint click captures the offset. ``cloneSource`` was set
      // by Alt+click; here ``(px, py)`` is the dest of the first stamp.
      state.cloneOffsetX = px - state.cloneSource.x;
      state.cloneOffsetY = py - state.cloneSource.y;
    }
    const W = state.canvasW, H = state.canvasH;
    const r = state.brushSize;
    if (r <= 0) return;
    // Pick the source canvas. Same-layer needs a snapshot to avoid
    // self-stamping bleed.
    const srcLayer = state.cloneSameLayer
      ? tgt.layer
      : (getLayer(state.cloneSrcLayerIdx) || tgt.layer);
    const srcCanvas = srcLayer.canvas;
    const sctx = srcLayer.ctx;
    const srcImg = sctx.getImageData(0, 0, srcCanvas.width, srcCanvas.height);
    const sbuf = srcImg.data;
    // Read-modify-write the dest layer rect.
    const x0 = Math.max(0, px - r);
    const y0 = Math.max(0, py - r);
    const x1 = Math.min(W, px + r + 1);
    const y1 = Math.min(H, py + r + 1);
    if (x0 >= x1 || y0 >= y1) return;
    const dimg = tgt.ctx.getImageData(x0, y0, x1 - x0, y1 - y0);
    const dbuf = dimg.data;
    const r2 = r * r;
    const soft = 1.0 - state.hardness;
    const softSq = Math.max(soft * soft, 1e-6);
    const opacity = state.opacity;
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const ddx = x - px, ddy = y - py;
        const d2 = ddx * ddx + ddy * ddy;
        if (d2 > r2) continue;
        const sx = x - state.cloneOffsetX;
        const sy = y - state.cloneOffsetY;
        if (sx < 0 || sx >= W || sy < 0 || sy >= H) continue;
        let fa;
        if (state.hardness >= 0.999) fa = 1.0;
        else fa = Math.exp(-((d2 / r2) / softSq));
        const sa = (sbuf[(sy * W + sx) * 4 + 3] / 255.0) * fa * opacity;
        if (sa <= 0) continue;
        const si = (sy * W + sx) * 4;
        const di = ((y - y0) * (x1 - x0) + (x - x0)) * 4;
        const sr = sbuf[si + 0], sg = sbuf[si + 1], sb = sbuf[si + 2];
        const dr = dbuf[di + 0], dg = dbuf[di + 1], db = dbuf[di + 2];
        const da = dbuf[di + 3] / 255.0;
        const inv = 1.0 - sa;
        const outA = sa + da * inv;
        if (outA <= 0) {
          dbuf[di] = dbuf[di + 1] = dbuf[di + 2] = dbuf[di + 3] = 0;
          continue;
        }
        dbuf[di + 0] = Math.round((sr * sa + dr * da * inv) / outA);
        dbuf[di + 1] = Math.round((sg * sa + dg * da * inv) / outA);
        dbuf[di + 2] = Math.round((sb * sa + db * da * inv) / outA);
        dbuf[di + 3] = Math.round(outA * 255.0);
      }
    }
    tgt.ctx.putImageData(dimg, x0, y0);
  }

  // Render a gradient from (x0, y0) to (x1, y1) onto the active layer's
  // canvas. Same math as formats/paint.py::gradient_fill (linear /
  // radial / angular, clamp to nearest stop outside the axis).
  function renderGradient(x0, y0, x1, y1) {
    const tgt = getActiveDrawTarget();
    if (!tgt) return;
    if (tgt.layer.locked) {
      setStatus("err", "active layer is locked");
      return;
    }
    const W = state.canvasW, H = state.canvasH;
    const ctx = tgt.ctx;
    const stops = state.gradStops.slice().sort((a, b) => a.pos - b.pos);
    const stopRgba = stops.map((s) => {
      const c = parseColor(s.color);
      const a = Math.round((s.alpha != null ? s.alpha : 1.0) * 255);
      return [s.pos, c[0], c[1], c[2], a];
    });
    const dx = x1 - x0, dy = y1 - y0;
    const length = Math.sqrt(dx * dx + dy * dy);
    const inv_len = length > 0 ? 1.0 / length : 0;
    const nx = dx * inv_len;
    const ny = dy * inv_len;
    const kind = state.gradKind;
    const opacity = state.opacity;
    const img = ctx.getImageData(0, 0, W, H);
    const buf = img.data;
    for (let py = 0; py < H; py++) {
      for (let px = 0; px < W; px++) {
        let t;
        if (kind === "linear") {
          t = ((px - x0) * nx + (py - y0) * ny) * inv_len;
        } else if (kind === "radial") {
          const rx = px - x0, ry = py - y0;
          t = Math.sqrt(rx * rx + ry * ry) * inv_len;
        } else { // angular
          const rx = px - x0, ry = py - y0;
          let a = Math.atan2(ry, rx) - Math.atan2(dy, dx);
          a = a / (2 * Math.PI);
          t = a - Math.floor(a);
        }
        // Clamp + interp stops
        let r, g, b, a;
        if (t <= stopRgba[0][0]) {
          [, r, g, b, a] = stopRgba[0];
        } else if (t >= stopRgba[stopRgba.length - 1][0]) {
          [, r, g, b, a] = stopRgba[stopRgba.length - 1];
        } else {
          for (let i = 0; i < stopRgba.length - 1; i++) {
            const s0 = stopRgba[i], s1 = stopRgba[i + 1];
            if (t >= s0[0] && t <= s1[0]) {
              const f = (t - s0[0]) / Math.max(1e-9, (s1[0] - s0[0]));
              r = Math.round(s0[1] + (s1[1] - s0[1]) * f);
              g = Math.round(s0[2] + (s1[2] - s0[2]) * f);
              b = Math.round(s0[3] + (s1[3] - s0[3]) * f);
              a = Math.round(s0[4] + (s1[4] - s0[4]) * f);
              break;
            }
          }
        }
        const sa = (a / 255.0) * opacity;
        if (sa <= 0) continue;
        const di = (py * W + px) * 4;
        const inv = 1.0 - sa;
        const dr = buf[di + 0], dg = buf[di + 1], db = buf[di + 2];
        const da = buf[di + 3] / 255.0;
        const outA = sa + da * inv;
        if (outA <= 0) {
          buf[di] = buf[di + 1] = buf[di + 2] = buf[di + 3] = 0;
          continue;
        }
        buf[di + 0] = Math.round((r * sa + dr * da * inv) / outA);
        buf[di + 1] = Math.round((g * sa + dg * da * inv) / outA);
        buf[di + 2] = Math.round((b * sa + db * da * inv) / outA);
        buf[di + 3] = Math.round(outA * 255);
      }
    }
    ctx.putImageData(img, 0, 0);
    recompositeNow();
  }

  // ---- the actual paint dispatch -------------------------------------
  function handlePaintAt(ev) {
    if (!state.canvas || !state.threeTexture) return;
    const hit = raycastMesh(ev);
    if (!hit) return;
    // Only act if the hit's mesh is bound to one of our active material_ids.
    // That keeps us from painting on a sibling submesh that uses a
    // different texture. (Both can be visible at once.)
    //
    // fix/tooltabs — resolve the material via hit.face.materialIndex for the
    // psov2 multi-material SkinnedMesh (one mesh, per-face material), falling
    // back to userData.materialId for legacy single-material meshes. The old
    // code read userData.materialId off the whole SkinnedMesh, which is
    // undefined there -> `undefined|0 === 0`, so every stroke on a tile other
    // than material 0 was silently dropped.
    if (!hitIsActive(hit)) {
      // Hit was on a sibling mesh that isn't bound to our active tile —
      // ignore so we don't streak the wrong texture.
      return;
    }
    const uv = hit.uv;
    if (!uv) return;
    const [px, py] = uvToPixel(uv.x, uv.y, state.canvasW, state.canvasH);
    applyTool(px, py);
  }

  // ---- undo / redo ---------------------------------------------------
  // Each undo entry captures (a) the active layer's idx + (b) an
  // ImageData snapshot of that layer's canvas (or its mask, if paintMask
  // was on at the time). On undo we restore the snapshot to the layer
  // it came from and re-composite. STACK_LIMIT caps the queue.
  function pushUndo() {
    const tgt = getActiveDrawTarget();
    if (!tgt) return;
    try {
      const img = tgt.ctx.getImageData(0, 0, state.canvasW, state.canvasH);
      const entry = {
        layerIdx: tgt.layer.idx,
        isMask: tgt.isMask,
        img: img,
      };
      state.undoStack.push(entry);
      if (state.undoStack.length > STACK_LIMIT) state.undoStack.shift();
    } catch {}
    // Cross-tool undo bus integration (2026-04-25). The bus stores a
    // closure that captures the layer state BEFORE this stroke + the
    // layer state AFTER this stroke, and replays them on undo/redo.
    if (window.psoUndoBus) {
      try {
        const stack = state.undoStack;
        if (stack.length >= 2) {
          const before = stack[stack.length - 2];
          const after = stack[stack.length - 1];
          const innerName = (function () {
            try { return activeInnerName(); } catch (_e) { return ""; }
          })();
          const label = "paint stroke" + (innerName ? " (" + innerName + ")" : "");
          window.psoUndoBus.push({
            label: label,
            panelId: "paint",
            undo: function () {
              try {
                applyUndoEntry(before);
                if (state.undoStack.length > 1) {
                  const popped = state.undoStack.pop();
                  state.redoStack.push(popped);
                  if (state.redoStack.length > STACK_LIMIT) state.redoStack.shift();
                }
              } catch (e) { console.warn("[paint] bus undo threw:", e); }
            },
            redo: function () {
              try {
                applyUndoEntry(after);
                if (state.redoStack.length > 0) {
                  const next = state.redoStack.pop();
                  state.undoStack.push(next);
                  if (state.undoStack.length > STACK_LIMIT) state.undoStack.shift();
                }
              } catch (e) { console.warn("[paint] bus redo threw:", e); }
            },
          });
        }
      } catch (e) { console.warn("[paint] bus push threw:", e); }
    }
  }

  function applyUndoEntry(entry) {
    if (!entry) return;
    const L = getLayer(entry.layerIdx);
    if (!L) return;
    const ctx = entry.isMask ? L.maskCtx : L.ctx;
    if (!ctx) return;
    ctx.putImageData(entry.img, 0, 0);
    recompositeNow();
  }

  function doUndo() {
    if (state.undoStack.length <= 1) return;
    const cur = state.undoStack.pop();
    state.redoStack.push(cur);
    if (state.redoStack.length > STACK_LIMIT) state.redoStack.shift();
    const prev = state.undoStack[state.undoStack.length - 1];
    applyUndoEntry(prev);
    setStatus("ok", `undo (${state.undoStack.length} states left)`);
  }

  function doRedo() {
    if (!state.redoStack.length) return;
    const next = state.redoStack.pop();
    state.undoStack.push(next);
    applyUndoEntry(next);
    setStatus("ok", `redo (${state.redoStack.length} forward states)`);
  }

  // ---- clone source pip (Alt+click feedback) ------------------------
  function moveClonePip(ev) {
    if (state.activeTool !== "clone" || !state.cloneSource) {
      if (ui.cloneSrcPip) {
        ui.cloneSrcPip.style.display = "none";
      }
      return;
    }
    // Map source texture pixel back to a screen point by tracing the
    // current mouse hit's UV bias. For now, we just track the cursor
    // position when the source was captured (CSS overlay).
    if (!ui.cloneSrcPip) {
      ui.cloneSrcPip = document.createElement("div");
      ui.cloneSrcPip.className = "pso-paint-clone-source-pip";
      document.body.appendChild(ui.cloneSrcPip);
    }
    if (state.cloneSource && state.cloneSource._screen) {
      const s = state.cloneSource._screen;
      ui.cloneSrcPip.style.display = "block";
      ui.cloneSrcPip.style.left = `${s.x}px`;
      ui.cloneSrcPip.style.top = `${s.y}px`;
    }
  }

  function setCloneSourceFromEvent(ev, px, py) {
    state.cloneSource = { x: px, y: py, layerIdx: state.activeIdx,
                          _screen: { x: ev.clientX, y: ev.clientY } };
  }

  // ---- gradient preview overlay -------------------------------------
  function ensureGradPreview() {
    if (ui.gradLineEl) return;
    ui.gradLineEl = document.createElement("div");
    ui.gradLineEl.className = "pso-paint-grad-preview";
    document.body.appendChild(ui.gradLineEl);
    ui.gradHandle1 = document.createElement("div");
    ui.gradHandle1.className = "pso-paint-grad-handle";
    document.body.appendChild(ui.gradHandle1);
    ui.gradHandle2 = document.createElement("div");
    ui.gradHandle2.className = "pso-paint-grad-handle";
    document.body.appendChild(ui.gradHandle2);
  }
  function removeGradPreview() {
    if (ui.gradLineEl) { ui.gradLineEl.remove(); ui.gradLineEl = null; }
    if (ui.gradHandle1) { ui.gradHandle1.remove(); ui.gradHandle1 = null; }
    if (ui.gradHandle2) { ui.gradHandle2.remove(); ui.gradHandle2 = null; }
  }
  function updateGradPreview() {
    if (!state.gradDrag || !ui.gradLineEl) return;
    const a = state.gradDrag.startScreen;
    const b = state.gradDrag.endScreen;
    const dx = b.x - a.x, dy = b.y - a.y;
    const len = Math.sqrt(dx * dx + dy * dy);
    const ang = Math.atan2(dy, dx) * 180 / Math.PI;
    ui.gradLineEl.style.left = `${a.x}px`;
    ui.gradLineEl.style.top = `${a.y}px`;
    ui.gradLineEl.style.width = `${len}px`;
    ui.gradLineEl.style.transform = `rotate(${ang}deg)`;
    ui.gradHandle1.style.left = `${a.x}px`;
    ui.gradHandle1.style.top = `${a.y}px`;
    ui.gradHandle2.style.left = `${b.x}px`;
    ui.gradHandle2.style.top = `${b.y}px`;
  }

  // ---- save / reset / build / deploy ---------------------------------
  function activeInnerName() {
    // The inner name is everything after '#' in the bound archive path,
    // e.g. "bm_ene_bm9_s_mericarol.bml#bm_ene_bm9_s_mericarol.nj.xvm".
    const arch = state.activeArchive || "";
    const idx = arch.indexOf("#");
    if (idx < 0) {
      // Top-level XVM (no inner). Use the full name as inner.
      return arch.split("/").pop();
    }
    return arch.slice(idx + 1);
  }

  function activeHostName() {
    const arch = state.activeArchive || "";
    const idx = arch.indexOf("#");
    return (idx < 0 ? arch : arch.slice(0, idx)).split("/").pop();
  }

  async function onAction(act) {
    if (act === "save") return doSave();
    if (act === "reset") return doReset();
    if (act === "build") return doBuildAndDeploy();
    if (act === "undo") return doUndo();
    if (act === "redo") return doRedo();
    if (act === "livetest") return doLiveTest();
    if (act === "layer-add") return layerAdd();
    if (act === "layer-dup") return layerDuplicate(state.activeIdx);
    if (act === "layer-merge") return layerMergeDown(state.activeIdx);
    if (act === "layer-del") return layerDelete(state.activeIdx);
  }

  // Live Test: stage the painted PNG as a client-side texture override
  // under cache/live_overrides/. The combo ASI's mod_live_replace module
  // (md5 3b322158, deployed 2026-04-25) hot-reloads via D3D9 SetTexture
  // redirection using a pixel-fingerprint match on the ORIGINAL texture's
  // RGBA bytes — so we MUST send the source PNG alongside the painted PNG
  // so the server can compute MD5(source RGBA) and write it as
  // match.src_rgba_md5 in the .replace sidecar. Without that, the ASI
  // sees the staged file but skips the swap (logged as "staged but
  // inert" by the ASI). The original is preserved on first bind in
  // state.sourceImageBitmap (also used by the Reset button).
  async function doLiveTest() {
    if (!state.canvas) {
      setStatus("err", "no painted texture to live-test");
      return;
    }
    if (!window.PSOLiveTest) {
      setStatus("err", "PSOLiveTest module not loaded");
      return;
    }
    setStatus("busy", "staging for live test…");
    try {
      const dataUrl = state.canvas.toDataURL("image/png");
      const b64 = dataUrl.split(",")[1] || "";
      // Encode the source bitmap into a PNG so the server can decode it
      // bit-identically to how it was originally fetched. Drawing it onto
      // a same-size offscreen canvas + toDataURL gives byte-equivalent
      // RGBA bytes (Chromium re-encodes losslessly for image/png).
      let srcB64 = "";
      if (state.sourceImageBitmap) {
        try {
          const w = state.sourceImageBitmap.width;
          const h = state.sourceImageBitmap.height;
          const tmp = document.createElement("canvas");
          tmp.width = w;
          tmp.height = h;
          const tctx = tmp.getContext("2d");
          tctx.imageSmoothingEnabled = false;
          tctx.drawImage(state.sourceImageBitmap, 0, 0);
          const srcUrl = tmp.toDataURL("image/png");
          srcB64 = srcUrl.split(",")[1] || "";
        } catch (e) {
          console.warn("[paint] could not encode source for fingerprint:", e);
        }
      }
      const arch = activeHostName();
      const inner = activeInnerName();
      const assetPath = inner ? (arch + "#" + inner) : arch;
      const liveBody = { asset_path: assetPath, png_b64: b64 };
      if (srcB64) liveBody.src_png_b64 = srcB64;
      const result = await window.PSOLiveTest.triggerLiveTest("texture", {
        panelId: "paint",
        body: liveBody,
      });
      if (result.ok === false) {
        setStatus("err", "live-test failed: " + (result.error || "unknown"));
        return;
      }
      const dep = (result.deployed && result.deployed.override_png) || "";
      if (result.deployed && result.deployed.consumer_active === false) {
        setStatus("ok", "staged " + dep.replace(/^.*[\\\/]/, "") +
                        " (combo ASI consumer not yet running)");
      } else {
        setStatus("ok", "live: " + dep.replace(/^.*[\\\/]/, ""));
      }
    } catch (e) {
      setStatus("err", "live-test failed: " + (e.message || e));
    }
  }

  // Save the entire layer stack: every dirty layer's RGBA PNG + every
  // mask + a manifest update (so reorder/delete persists). Server
  // recomputes the flat composite from the persisted layers.
  async function doSave() {
    if (!state.canvas) return;
    if (state.pendingSave) return;
    if (!state.layers.length) return;
    state.pendingSave = true;
    setStatus("busy", "saving layer stack…");
    try {
      const host = activeHostName();
      const inner = activeInnerName();
      // First push the manifest so the server knows the current layer
      // order + metadata. (Layer PNGs follow.)
      let mResp = await fetch("/api/paint/manifest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_path: host,
          inner: inner,
          manifest: buildManifestForServer(),
        }),
      });
      if (mResp.status === 404) {
        // No layer dir yet — first save needs to seed layer 0.
      } else if (!mResp.ok) {
        // Surface the manifest write error but continue with per-layer
        // saves (which create the dir if missing).
        let det = "HTTP " + mResp.status;
        try { det = (await mResp.json()).detail || det; } catch {}
        console.warn("[paint] manifest update failed:", det);
      }
      // Save every layer's RGBA + mask in order.
      for (const L of state.layers) {
        await saveOneLayer(L);
        if (L.hasMask && L.maskCanvas) await saveOneLayer(L, /*mask*/ true);
      }
      // Final pass: re-write the manifest now that each layer exists on
      // disk. Server recomputes the composite.
      mResp = await fetch("/api/paint/manifest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_path: host,
          inner: inner,
          manifest: buildManifestForServer(),
        }),
      });
      if (!mResp.ok) {
        let det = "HTTP " + mResp.status;
        try { det = (await mResp.json()).detail || det; } catch {}
        throw new Error("manifest: " + det);
      }
      const out = await mResp.json();
      setStatus(
        "ok",
        `saved · ${state.layers.length} layer${state.layers.length === 1 ? "" : "s"}` +
          (out.composite_md5 ? ` (md5 ${out.composite_md5.slice(0, 8)})` : ""),
      );
    } catch (e) {
      setStatus("err", `save failed: ${e.message || e}`);
    } finally {
      state.pendingSave = false;
    }
  }

  // POST one layer's PNG (or mask PNG) to /api/paint/layer/save.
  async function saveOneLayer(L, isMask) {
    const cv = isMask ? L.maskCanvas : L.canvas;
    if (!cv) return;
    const dataUrl = cv.toDataURL("image/png");
    const b64 = dataUrl.split(",")[1] || "";
    const body = {
      model_path: activeHostName(),
      inner: activeInnerName(),
      layer_idx: L.idx,
      png_b64: b64,
      is_mask: !!isMask,
      name: L.name,
      visible: L.visible,
      opacity: L.opacity,
      blend_mode: L.blend_mode,
      locked: L.locked,
      has_mask: L.hasMask,
    };
    const r = await fetch("/api/paint/layer/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      let det = "HTTP " + r.status;
      try { det = (await r.json()).detail || det; } catch {}
      throw new Error(`layer ${L.idx}${isMask ? " mask" : ""}: ${det}`);
    }
  }

  // Build the manifest dict the server expects.
  function buildManifestForServer() {
    return {
      version: 1,
      model_path: activeHostName(),
      inner: activeInnerName(),
      width: state.canvasW,
      height: state.canvasH,
      active: state.activeIdx,
      layers: state.layers.map((L) => ({
        idx: L.idx,
        name: L.name,
        visible: L.visible,
        opacity: L.opacity,
        blend_mode: L.blend_mode,
        locked: L.locked,
        has_mask: L.hasMask,
      })),
    };
  }

  // Auto-save 1.5 s after the last stroke finishes. Avoids the user
  // worrying about "did I save?" while the layer state is still
  // recoverable from the manifest.
  function scheduleAutoSave() {
    if (state.pendingAutoSave) clearTimeout(state.pendingAutoSave);
    state.pendingAutoSave = setTimeout(() => {
      state.pendingAutoSave = 0;
      doSave().catch((e) => console.warn("[paint] auto-save:", e));
    }, 1500);
  }

  async function doReset() {
    if (!state.sourceImageBitmap) return;
    // Reset reverts the ACTIVE LAYER to the source texture. Other layers
    // are untouched. Push undo so the user can roll the action back.
    pushUndo();
    const tgt = getActiveDrawTarget();
    if (!tgt) return;
    if (tgt.isMask) {
      // Reset mask = fill white (fully visible).
      tgt.ctx.save();
      tgt.ctx.globalCompositeOperation = "source-over";
      tgt.ctx.fillStyle = "#ffffff";
      tgt.ctx.fillRect(0, 0, state.canvasW, state.canvasH);
      tgt.ctx.restore();
    } else {
      tgt.ctx.clearRect(0, 0, state.canvasW, state.canvasH);
      tgt.ctx.drawImage(state.sourceImageBitmap, 0, 0);
    }
    recompositeNow();
    scheduleAutoSave();
    setStatus("ok", `reset ${tgt.isMask ? "mask" : "layer"} to source`);
  }

  async function doBuildAndDeploy() {
    if (state.pendingSave) return;
    setStatus("busy", "building archive…");
    try {
      // Save current state first if we have a stroke since last save.
      // Cheap insurance — server is a no-op if the bytes match.
      await doSave();
      const host = activeHostName();
      let r = await fetch("/api/paint/build_archive", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_path: host }),
      });
      if (!r.ok) {
        let det = "HTTP " + r.status;
        try { det = (await r.json()).detail || det; } catch {}
        throw new Error("build: " + det);
      }
      const built = await r.json();
      setStatus("busy", `built ${host} (${built.size} B), deploying…`);
      r = await fetch("/api/paint/deploy", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ archive_name: host, create_backup: true }),
      });
      if (!r.ok) {
        let det = "HTTP " + r.status;
        try { det = (await r.json()).detail || det; } catch {}
        throw new Error("deploy: " + det);
      }
      const dep = await r.json();
      setStatus(
        "ok",
        `deployed ${host} (${dep.live_size} B, backup=${dep.backup_name || "n/a"})`,
      );
    } catch (e) {
      setStatus("err", `build/deploy failed: ${e.message || e}`);
    }
  }

  // ---- layer mutations (called from the layer panel UI) ------------
  function layerAdd() {
    if (state.layers.length >= MAX_LAYERS) {
      setStatus("err", `max ${MAX_LAYERS} layers`);
      return;
    }
    const idx = nextFreeLayerIdx();
    const L = makeLayer({
      idx,
      name: `Layer ${state.layers.length}`,
      visible: true, opacity: 1.0, blend_mode: "normal", locked: false,
    });
    state.layers.push(L);
    state.activeIdx = idx;
    state.nextLayerId = Math.max(state.nextLayerId, idx + 1);
    recompositeNow();
    renderLayerList();
    scheduleAutoSave();
  }

  async function layerDelete(idx) {
    if (state.layers.length <= 1) {
      setStatus("err", "can't delete the only layer");
      return;
    }
    const i = state.layers.findIndex((L) => L.idx === idx);
    if (i < 0) return;
    state.layers.splice(i, 1);
    if (state.activeIdx === idx) {
      state.activeIdx = state.layers[Math.max(0, i - 1)].idx;
    }
    // Server-side delete (so the on-disk PNG goes away too).
    try {
      await fetch("/api/paint/layer/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_path: activeHostName(),
          inner: activeInnerName(),
          layer_idx: idx,
          is_mask: false,
        }),
      });
    } catch (e) {
      console.warn("[paint] layer delete server call failed:", e);
    }
    recompositeNow();
    renderLayerList();
    scheduleAutoSave();
  }

  function layerDuplicate(idx) {
    if (state.layers.length >= MAX_LAYERS) {
      setStatus("err", `max ${MAX_LAYERS} layers`);
      return;
    }
    const src = getLayer(idx);
    if (!src) return;
    const nidx = nextFreeLayerIdx();
    const dup = makeLayer({
      idx: nidx,
      name: `${src.name} copy`,
      visible: src.visible, opacity: src.opacity,
      blend_mode: src.blend_mode, locked: false,
      hasMask: src.hasMask,
    });
    dup.ctx.drawImage(src.canvas, 0, 0);
    if (src.hasMask && src.maskCanvas) {
      attachMaskCanvas(dup, true);
      dup.maskCtx.drawImage(src.maskCanvas, 0, 0);
    }
    // Insert directly above the source.
    const i = state.layers.findIndex((L) => L.idx === idx);
    state.layers.splice(i + 1, 0, dup);
    state.activeIdx = nidx;
    recompositeNow();
    renderLayerList();
    scheduleAutoSave();
  }

  // Merge layer at array position i into the layer at position i-1
  // (i.e. the one directly below). The result respects the upper
  // layer's blend_mode + opacity + mask. The lower layer becomes the
  // composited result with blend_mode reset to "normal".
  async function layerMergeDown(idx) {
    const i = state.layers.findIndex((L) => L.idx === idx);
    if (i <= 0) {
      setStatus("err", "no layer below to merge into");
      return;
    }
    const top = state.layers[i];
    const bottom = state.layers[i - 1];
    // Apply the top layer's effective pixels onto the bottom canvas
    // using the top's blend mode + opacity + mask.
    const W = state.canvasW, H = state.canvasH;
    let topCanvas = top.canvas;
    if (top.hasMask && top.maskCanvas) {
      topCanvas = applyMaskToTempCanvas(top);
    }
    bottom.ctx.save();
    bottom.ctx.globalAlpha = top.opacity;
    bottom.ctx.globalCompositeOperation = canvasBlendOpFor(top.blend_mode);
    bottom.ctx.drawImage(topCanvas, 0, 0);
    bottom.ctx.restore();
    // Drop the top layer.
    state.layers.splice(i, 1);
    if (state.activeIdx === idx) state.activeIdx = bottom.idx;
    try {
      await fetch("/api/paint/layer/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_path: activeHostName(),
          inner: activeInnerName(),
          layer_idx: top.idx,
          is_mask: false,
        }),
      });
    } catch (_e) { /* best-effort */ }
    recompositeNow();
    renderLayerList();
    scheduleAutoSave();
  }

  // Drag-to-reorder. ``fromIdx`` and ``toIdx`` are layer.idx values; we
  // resolve to array positions and splice. ``placeAbove`` is true when
  // the target should sit above the dropped-on layer.
  function layerReorder(fromIdx, toIdx, placeAbove) {
    if (fromIdx === toIdx) return;
    const fi = state.layers.findIndex((L) => L.idx === fromIdx);
    const ti = state.layers.findIndex((L) => L.idx === toIdx);
    if (fi < 0 || ti < 0) return;
    const [moved] = state.layers.splice(fi, 1);
    let insertAt = state.layers.findIndex((L) => L.idx === toIdx);
    if (insertAt < 0) insertAt = state.layers.length;
    insertAt += (placeAbove ? 1 : 0);
    state.layers.splice(insertAt, 0, moved);
    recompositeNow();
    renderLayerList();
    scheduleAutoSave();
  }

  function layerToggleVisible(idx) {
    const L = getLayer(idx);
    if (!L) return;
    L.visible = !L.visible;
    recompositeNow();
    renderLayerList();
    scheduleAutoSave();
  }

  function layerToggleLocked(idx) {
    const L = getLayer(idx);
    if (!L) return;
    L.locked = !L.locked;
    renderLayerList();
    scheduleAutoSave();
  }

  function layerSetOpacity(idx, opacity) {
    const L = getLayer(idx);
    if (!L) return;
    L.opacity = Math.max(0, Math.min(1, opacity));
    recompositeNow();
    scheduleAutoSave();
  }

  function layerSetBlendMode(idx, mode) {
    const L = getLayer(idx);
    if (!L) return;
    if (!SUPPORTED_BLEND_MODES.includes(mode)) return;
    L.blend_mode = mode;
    recompositeNow();
    scheduleAutoSave();
  }

  function layerSetName(idx, name) {
    const L = getLayer(idx);
    if (!L) return;
    L.name = String(name || "").slice(0, 64) || L.name;
    renderLayerList();
    scheduleAutoSave();
  }

  function layerActivate(idx) {
    if (!getLayer(idx)) return;
    state.activeIdx = idx;
    state.paintMask = false;  // exit mask-paint mode on layer switch
    state.cloneSource = null;
    state.cloneOffsetX = null;
    state.cloneOffsetY = null;
    moveClonePip();
    renderLayerList();
  }

  function maskAdd(idx) {
    const L = getLayer(idx);
    if (!L) return;
    if (L.hasMask) {
      setStatus("err", "layer already has a mask");
      return;
    }
    attachMaskCanvas(L, true);
    recompositeNow();
    renderLayerList();
    scheduleAutoSave();
  }

  async function maskRemove(idx) {
    const L = getLayer(idx);
    if (!L || !L.hasMask) return;
    L.hasMask = false;
    L.maskCanvas = null;
    L.maskCtx = null;
    if (state.activeIdx === idx) state.paintMask = false;
    try {
      await fetch("/api/paint/layer/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_path: activeHostName(),
          inner: activeInnerName(),
          layer_idx: idx,
          is_mask: true,
        }),
      });
    } catch (_e) { /* best-effort */ }
    recompositeNow();
    renderLayerList();
    scheduleAutoSave();
  }

  // Bake the mask into the layer's RGBA alpha channel.
  function maskApply(idx) {
    const L = getLayer(idx);
    if (!L || !L.hasMask || !L.maskCanvas) return;
    const W = state.canvasW, H = state.canvasH;
    const limg = L.ctx.getImageData(0, 0, W, H);
    const mimg = L.maskCtx.getImageData(0, 0, W, H);
    const lb = limg.data;
    const mb = mimg.data;
    for (let i = 0; i < lb.length; i += 4) {
      // Multiply layer alpha by mask R channel (grayscale).
      lb[i + 3] = Math.round(lb[i + 3] * (mb[i] / 255));
    }
    L.ctx.putImageData(limg, 0, 0);
    // Drop the mask.
    L.hasMask = false;
    L.maskCanvas = null;
    L.maskCtx = null;
    if (state.activeIdx === idx) state.paintMask = false;
    recompositeNow();
    renderLayerList();
    scheduleAutoSave();
  }

  function toggleMaskPaint() {
    const L = getActiveLayer();
    if (!L || !L.hasMask) {
      setStatus("err", "active layer has no mask");
      state.paintMask = false;
      return;
    }
    state.paintMask = !state.paintMask;
    renderLayerList();
  }

  // ---- layer-panel UI -----------------------------------------------
  function renderLayerList() {
    if (!ui.bodyEl) return;
    const host = ui.bodyEl.querySelector('[data-region="layers-list"]');
    if (!host) return;
    if (!state.layers.length) {
      host.innerHTML = '<div class="pso-paint-layers-empty">no layers (load a tile)</div>';
      return;
    }
    // Iterate in array order; CSS column-reverse flips visually so layer 0
    // sits at the bottom (matches Photoshop convention).
    const html = state.layers.map((L) => {
      const active = L.idx === state.activeIdx;
      const cls = [
        "pso-paint-layer-row",
        active ? "active" : "",
        L.locked ? "locked" : "",
        active && state.paintMask ? "mask-active" : "",
      ].filter(Boolean).join(" ");
      const blendOpts = SUPPORTED_BLEND_MODES.map((m) => (
        `<option value="${m}"${m === L.blend_mode ? " selected" : ""}>${m}</option>`
      )).join("");
      const eye = L.visible ? "open" : "hidden";
      const eyeChar = L.visible ? "\u25C9" : "\u25CB";
      const lockChar = L.locked ? "\uD83D\uDD12" : "\uD83D\uDD13";
      const maskClass = L.hasMask ? "mini-mask has" : "mini-mask";
      const maskTitle = L.hasMask ? "mask present (click to toggle paint-mask)" : "no mask (click to add)";
      const opPct = Math.round(L.opacity * 100);
      return `<div class="${cls}" data-layer-idx="${L.idx}" draggable="true">
        <span class="eye ${eye}" title="visibility" data-act="vis">${eyeChar}</span>
        <span class="lock ${L.locked ? "on" : ""}" title="lock layer" data-act="lock">${lockChar}</span>
        <span class="nm" data-act="rename" title="${escapeHtml(L.name)}">${escapeHtml(L.name)}</span>
        <select class="blend-mode" data-act="blend" title="blend mode">${blendOpts}</select>
        <input type="number" class="opacity" min="0" max="100" step="1" value="${opPct}"
               data-act="opacity" title="opacity %" />
        <span class="${maskClass}" data-act="mask" title="${escapeHtml(maskTitle)}"></span>
      </div>`;
    }).join("");
    host.innerHTML = html;
    // Wire per-row events.
    host.querySelectorAll(".pso-paint-layer-row").forEach((row) => {
      const idx = parseInt(row.getAttribute("data-layer-idx"), 10);
      row.addEventListener("click", (ev) => {
        const t = ev.target;
        const act = t && t.getAttribute && t.getAttribute("data-act");
        if (act === "vis") { layerToggleVisible(idx); ev.stopPropagation(); return; }
        if (act === "lock") { layerToggleLocked(idx); ev.stopPropagation(); return; }
        if (act === "blend") return; // handled by change
        if (act === "opacity") return; // handled by input
        if (act === "mask") {
          const L = getLayer(idx);
          if (L && L.hasMask) {
            if (state.activeIdx !== idx) layerActivate(idx);
            toggleMaskPaint();
          } else {
            maskAdd(idx);
          }
          ev.stopPropagation();
          return;
        }
        if (act === "rename") {
          ev.stopPropagation();
          const newName = prompt("Layer name:", getLayer(idx) ? getLayer(idx).name : "");
          if (newName != null) layerSetName(idx, newName);
          return;
        }
        layerActivate(idx);
      });
      // Right-click for context menu.
      row.addEventListener("contextmenu", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        showLayerContextMenu(ev.clientX, ev.clientY, idx);
      });
      // Blend-mode select.
      const bsel = row.querySelector('[data-act="blend"]');
      if (bsel) {
        bsel.addEventListener("click", (e) => e.stopPropagation());
        bsel.addEventListener("change", (e) => {
          layerSetBlendMode(idx, e.target.value);
        });
      }
      // Opacity numeric input.
      const oin = row.querySelector('[data-act="opacity"]');
      if (oin) {
        oin.addEventListener("click", (e) => e.stopPropagation());
        oin.addEventListener("input", (e) => {
          const v = parseFloat(e.target.value);
          if (!Number.isNaN(v)) layerSetOpacity(idx, v / 100);
        });
      }
      // Drag-to-reorder.
      row.addEventListener("dragstart", (ev) => {
        ev.dataTransfer.effectAllowed = "move";
        ev.dataTransfer.setData("text/plain", String(idx));
        row.classList.add("dragging");
      });
      row.addEventListener("dragend", () => {
        host.querySelectorAll(".pso-paint-layer-row").forEach((r) => {
          r.classList.remove("dragging", "drop-above", "drop-below");
        });
      });
      row.addEventListener("dragover", (ev) => {
        ev.preventDefault();
        ev.dataTransfer.dropEffect = "move";
        const rect = row.getBoundingClientRect();
        const above = (ev.clientY - rect.top) < (rect.height / 2);
        // CSS column-reverse: visually "above" means later in the array.
        host.querySelectorAll(".pso-paint-layer-row").forEach((r) => {
          if (r !== row) r.classList.remove("drop-above", "drop-below");
        });
        row.classList.toggle("drop-above", above);
        row.classList.toggle("drop-below", !above);
      });
      row.addEventListener("drop", (ev) => {
        ev.preventDefault();
        const fromIdx = parseInt(ev.dataTransfer.getData("text/plain"), 10);
        if (Number.isNaN(fromIdx)) return;
        const rect = row.getBoundingClientRect();
        const above = (ev.clientY - rect.top) < (rect.height / 2);
        layerReorder(fromIdx, idx, !above);
      });
    });
  }

  let _ctxMenuEl = null;
  function showLayerContextMenu(x, y, layerIdx) {
    closeLayerContextMenu();
    const L = getLayer(layerIdx);
    if (!L) return;
    const menu = document.createElement("div");
    menu.className = "pso-paint-layer-context";
    menu.style.left = `${x}px`;
    menu.style.top = `${y}px`;
    const items = [
      { label: "Duplicate", act: "dup" },
      { label: "Merge down", act: "merge", disabled: state.layers.findIndex((l) => l.idx === layerIdx) === 0 },
      { label: "Delete", act: "del", disabled: state.layers.length <= 1 },
      { sep: true },
      { label: L.hasMask ? "Remove mask" : "Add mask", act: "mask" },
      { label: "Apply mask", act: "applymask", disabled: !L.hasMask },
    ];
    for (const it of items) {
      if (it.sep) {
        const hr = document.createElement("hr");
        menu.appendChild(hr);
        continue;
      }
      const b = document.createElement("button");
      b.textContent = it.label;
      if (it.disabled) b.disabled = true;
      b.addEventListener("click", () => {
        closeLayerContextMenu();
        if (it.act === "dup")        layerDuplicate(layerIdx);
        else if (it.act === "merge") layerMergeDown(layerIdx);
        else if (it.act === "del")   layerDelete(layerIdx);
        else if (it.act === "mask") {
          if (L.hasMask) maskRemove(layerIdx);
          else maskAdd(layerIdx);
        }
        else if (it.act === "applymask") maskApply(layerIdx);
      });
      menu.appendChild(b);
    }
    document.body.appendChild(menu);
    _ctxMenuEl = menu;
    setTimeout(() => {
      document.addEventListener("click", closeLayerContextMenu, { once: true });
    }, 10);
  }
  function closeLayerContextMenu() {
    if (_ctxMenuEl) {
      _ctxMenuEl.remove();
      _ctxMenuEl = null;
    }
  }

  // ---- cursor overlay ------------------------------------------------
  function ensureCursor() {
    if (ui.cursorEl) return;
    ui.cursorEl = document.createElement("div");
    ui.cursorEl.className = "pso-paint-cursor";
    document.body.appendChild(ui.cursorEl);
    updateCursor();
  }
  function removeCursor() {
    if (ui.cursorEl) {
      ui.cursorEl.remove();
      ui.cursorEl = null;
    }
  }
  function updateCursor() {
    if (!ui.cursorEl) return;
    // Brush size is in TEXTURE pixels — convert to screen pixels via
    // texture/canvas size ratio. For now, treat them as approx 1:1
    // (canvas is fitted to texture); the visual is just a hint.
    const cv = (typeof window.psoGetCanvas === "function")
      ? window.psoGetCanvas() : null;
    const px = state.brushSize;
    ui.cursorEl.style.width = `${px * 2}px`;
    ui.cursorEl.style.height = `${px * 2}px`;
    if (cv) {
      // Position at last known mouse location (set in moveCursor); if
      // never moved, keep off-screen.
      // No-op: position is updated in moveCursor.
    }
  }
  function moveCursor(ev) {
    if (!ui.cursorEl) return;
    ui.cursorEl.style.left = `${ev.clientX}px`;
    ui.cursorEl.style.top = `${ev.clientY}px`;
  }

  function renderForce() {
    if (typeof window.psoForceRender === "function") {
      window.psoForceRender();
    }
  }

  // ---- keyboard shortcuts -------------------------------------------
  function onKeyDown(ev) {
    if (!state.enabled) return;
    if (ev.target && (ev.target.tagName === "INPUT" || ev.target.tagName === "TEXTAREA")) return;
    const key = ev.key.toUpperCase();
    if (TOOL_KEYS[key]) {
      ev.preventDefault();
      setTool(TOOL_KEYS[key]);
      return;
    }
    if ((ev.ctrlKey || ev.metaKey) && key === "Z") {
      ev.preventDefault();
      if (ev.shiftKey) doRedo();
      else doUndo();
      return;
    }
  }

  // ---- bootstrap -----------------------------------------------------
  function tryInstall() {
    ensureStyle();
    const panel = ensureTabButton();
    if (!panel) {
      // Texture panel not mounted yet — try again later.
      return false;
    }
    ui.panel = panel;
    wireTabSwitch();
    document.addEventListener("keydown", onKeyDown);
    return true;
  }

  // The texture panel mounts lazily on first model open. Poll every
  // 250 ms until it appears, then stop. This is the same pattern
  // texture_panel.js uses for psoOpenModelByPath.
  let attempts = 0;
  function poll() {
    if (tryInstall()) return;
    attempts++;
    if (attempts > 240) return;  // ~60 s timeout
    setTimeout(poll, 250);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", poll);
  } else {
    poll();
  }

  // The texture panel can be torn down + rebuilt when the user leaves a
  // 3D perspective (overlap fix, 2026-06-20) — its tab strip loses the
  // manually-injected Paint button. texture_panel.js emits this right
  // after it (re)builds the panel DOM; re-install then. tryInstall() is
  // idempotent (ensureTabButton skips when the button already exists,
  // wireTabSwitch guards on a dataset flag).
  if (window.bus && typeof window.bus.on === "function") {
    window.bus.on("texture-panel.rebuilt", function () { tryInstall(); });
  }
})();
