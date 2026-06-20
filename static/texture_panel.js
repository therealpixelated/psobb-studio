// =====================================================================
// PSOBB Texture Editor — in-viewport texture + subdivide panel.
// 2026-04-25
//
// Mounts a sidebar inside the 3D-view's <div class="model-stage"> that
// lists every texture the currently-rendered model has bound. Each row
// shows a 64x64 thumbnail, filename / index / dims, and an "Upscale ×N"
// button group (×2/×4/×8) that drives /api/upscale on the texture's
// host archive. Replace-from-PNG goes through /api/import_png.
//
// The same sidebar holds the "Subdivide model" controls
// (Loop subdivision via trimesh on the server side). Both panels live
// in the same DOM region so the user never leaves the 3D view to do
// either task.
//
// Reads model state via:
//   window.psoListMeshTextures()         current bindings + tile dims
//   window.psoGetCurrentTextureArchive() archive path "<base>#<inner>"
//   window.psoGetTextureBinding()        raw binding rows
//   window.psoReloadTexture(tileIdx)     re-fetch one tile after upscale
//   window.psoApplyMeshPayload(payload)  swap geometry to a subdivide result
//
// Style tokens: reuses the existing --tk-* / model-* design language so
// no new theme tokens are introduced.
// =====================================================================

(function () {
  "use strict";

  if (window.__psoTexturePanelLoaded) return;
  window.__psoTexturePanelLoaded = true;

  const PANEL_ID = "psoTexturePanel";
  const STYLE_ID = "psoTexturePanelStyle";
  const VARIANT_STRIP_ID = "psoVariantStrip";
  const REFRESH_DEBOUNCE_MS = 250;

  // Per-tile upscale options. realesrgan-x4plus-anime is the default
  // (matches the editor's tile-grid default); tile_size=auto, tta=off,
  // gpu=auto. The user can pick the scale via the button row.
  const UPSCALE_MODEL_DEFAULT = "realesrgan-x4plus-anime";
  const UPSCALE_SCALES = [2, 4, 8];

  // ---- DOM helpers ---------------------------------------------------

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[c]));
  }

  function ensureStyleInjected() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .pso-tex-panel {
        position: absolute;
        top: 8px;
        left: 8px;
        bottom: 8px;
        width: 280px;
        background: rgba(10, 14, 19, 0.92);
        border: 1px solid #2a313a;
        border-radius: 4px;
        display: flex;
        flex-direction: column;
        font-size: 11px;
        font-family: var(--font-mono, monospace);
        z-index: 6;
        overflow: hidden;
        color: #e0e8f0;
      }
      .pso-tex-panel-header {
        padding: 6px 8px;
        border-bottom: 1px solid #2a313a;
        display: flex;
        gap: 8px;
        align-items: baseline;
        background: rgba(0, 0, 0, 0.25);
      }
      .pso-tex-panel-header strong { font-weight: 600; color: #00ffff; }
      .pso-tex-panel-header .grow { flex: 1; }
      .pso-tex-panel-header button {
        background: transparent;
        border: 1px solid #2a313a;
        color: #99a4b3;
        cursor: pointer;
        padding: 1px 6px;
        font: inherit;
        border-radius: 2px;
      }
      .pso-tex-panel-header button:hover {
        border-color: #00ffff;
        color: #00ffff;
      }
      .pso-tex-build-btn {
        font-size: 11px;
        background: rgba(0, 255, 255, 0.05) !important;
        border-color: #00ffff !important;
        color: #00ffff !important;
      }
      .pso-tex-build-btn:hover {
        background: rgba(0, 255, 255, 0.15) !important;
      }
      .pso-tex-build-status {
        font-size: 10px;
        color: #6c7785;
        margin-right: 4px;
        max-width: 280px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .pso-tex-build-status.busy { color: #d8c890; }
      .pso-tex-build-status.ok   { color: #56b67a; }
      .pso-tex-build-status.err  { color: #ff6680; }
      .pso-tex-panel-tabs {
        display: flex;
        flex-wrap: wrap;
        gap: 3px;
        padding: 4px 8px;
        border-bottom: 1px solid #2a313a;
        background: rgba(0, 0, 0, 0.15);
      }
      .pso-tex-panel-tabs button {
        flex: 1 0 auto;
        min-width: 60px;
        background: transparent;
        border: 1px solid #2a313a;
        color: #99a4b3;
        cursor: pointer;
        padding: 3px 6px;
        font: inherit;
        font-size: 10.5px;
        border-radius: 2px;
        transition: border-color 0.15s, color 0.15s, background 0.15s;
      }
      .pso-tex-panel-tabs button:hover {
        border-color: #4a90e2;
        color: #c7d8ec;
      }
      .pso-tex-panel-tabs button.active {
        background: rgba(0, 255, 255, 0.12);
        border-color: #00ffff;
        color: #00ffff;
      }

      /* ---- tab overflow ("more ▾") ---- */
      .pso-tex-tab-overflow {
        position: relative;
        flex: 0 0 auto;
        display: inline-flex;
      }
      .pso-tex-tab-more {
        background: transparent;
        border: 1px solid #2a313a;
        color: #99a4b3;
        cursor: pointer;
        padding: 3px 8px;
        font: inherit;
        font-size: 10.5px;
        border-radius: 2px;
        white-space: nowrap;
      }
      .pso-tex-tab-more:hover {
        border-color: #4a90e2;
        color: #c7d8ec;
      }
      .pso-tex-tab-more.has-active {
        background: rgba(0, 255, 255, 0.12);
        border-color: #00ffff;
        color: #00ffff;
      }
      .pso-tex-tab-overflow-menu {
        position: absolute;
        top: calc(100% + 3px);
        right: 0;
        z-index: 12;
        min-width: 130px;
        background: rgba(12, 16, 22, 0.98);
        border: 1px solid #2a313a;
        border-radius: 3px;
        padding: 3px;
        display: flex;
        flex-direction: column;
        gap: 2px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.5);
      }
      .pso-tex-tab-overflow-menu[hidden] { display: none; }
      .pso-tex-tab-overflow-menu button {
        flex: 0 0 auto;
        width: 100%;
        text-align: left;
        background: transparent;
        border: 1px solid transparent;
        color: #99a4b3;
        cursor: pointer;
        padding: 4px 8px;
        font: inherit;
        font-size: 10.5px;
        border-radius: 2px;
      }
      .pso-tex-tab-overflow-menu button:hover {
        border-color: #4a90e2;
        color: #c7d8ec;
        background: rgba(74, 144, 226, 0.10);
      }
      .pso-tex-tab-overflow-menu button.active {
        background: rgba(0, 255, 255, 0.12);
        border-color: #00ffff;
        color: #00ffff;
      }

      .pso-tex-panel-body {
        flex: 1;
        overflow-y: auto;
        /* 2026-06-20 (ui-polish): setting only overflow-y makes the
           browser compute overflow-x as 'auto' too, so a long unbreakable
           string (the asset path in .meta) grew a horizontal scrollbar
           inside this fixed-width (280px) panel. Pin overflow-x:hidden —
           rows truncate with ellipsis instead. */
        overflow-x: hidden;
      }
      .pso-tex-panel-empty {
        padding: 16px 12px;
        color: #6c7785;
        text-align: center;
        font-style: italic;
      }
      .pso-tex-row {
        display: flex;
        gap: 8px;
        padding: 6px 8px;
        border-bottom: 1px solid #1a1f26;
        /* let the flex children (.pso-tex-info) actually shrink so the
           long .meta path can't push the row wider than the panel. */
        min-width: 0;
      }
      .pso-tex-row:last-child { border-bottom: none; }
      .pso-tex-thumb {
        flex: 0 0 64px;
        width: 64px;
        height: 64px;
        background: #0a0e13 url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='8' height='8'><rect width='4' height='4' fill='%23222'/><rect x='4' y='4' width='4' height='4' fill='%23222'/></svg>");
        background-size: 8px 8px;
        border: 1px solid #2a313a;
        border-radius: 2px;
        position: relative;
        overflow: hidden;
      }
      .pso-tex-thumb img {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        object-fit: contain;
      }
      .pso-tex-info {
        flex: 1 1 auto;
        min-width: 0;
        display: flex;
        flex-direction: column;
        gap: 3px;
      }
      .pso-tex-info .nm {
        color: #c7d8ec;
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .pso-tex-info .meta {
        color: #6c7785;
        font-size: 10px;
        /* the asset path here is a long unbreakable token; truncate it
           rather than let it widen the row past the panel (was the source
           of the horizontal scrollbar in the model-viewer tex panel). */
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        min-width: 0;
      }
      .pso-tex-actions {
        display: flex;
        gap: 4px;
        flex-wrap: wrap;
      }
      .pso-tex-actions button {
        background: transparent;
        border: 1px solid #2a313a;
        color: #99a4b3;
        cursor: pointer;
        padding: 1px 5px;
        font: inherit;
        font-size: 10px;
        border-radius: 2px;
      }
      .pso-tex-actions button:hover {
        border-color: #4a90e2;
        color: #c7d8ec;
      }
      .pso-tex-actions button:disabled {
        opacity: 0.4;
        cursor: not-allowed;
      }
      .pso-tex-actions button.replace { color: #d8c890; border-color: #4d4523; }
      .pso-tex-actions button.replace:hover { border-color: #ffaa00; color: #ffaa00; }
      .pso-tex-status {
        font-size: 10px;
        color: #6c7785;
        margin-top: 2px;
        min-height: 12px;
      }
      .pso-tex-status.idle { color: #6c7785; }
      .pso-tex-status.queued { color: #d8c890; }
      .pso-tex-status.running { color: #4a90e2; }
      .pso-tex-status.done { color: #56b67a; }
      .pso-tex-status.err { color: #ff6680; }

      /* Subdivide section. Same look-and-feel as the texture rows but
         laid out as a single block. */
      .pso-sd-block {
        padding: 8px;
        display: flex;
        flex-direction: column;
        gap: 8px;
      }
      .pso-sd-block label { color: #99a4b3; }
      .pso-sd-row { display: flex; gap: 8px; align-items: center; }
      .pso-sd-row .grow { flex: 1; }
      .pso-sd-stats {
        background: rgba(0, 0, 0, 0.25);
        border: 1px solid #2a313a;
        border-radius: 2px;
        padding: 4px 6px;
        font-size: 10px;
        color: #c7d8ec;
        white-space: pre-wrap;
      }
      .pso-sd-actions { display: flex; gap: 4px; flex-wrap: wrap; }
      .pso-sd-actions button {
        background: transparent;
        border: 1px solid #2a313a;
        color: #99a4b3;
        cursor: pointer;
        padding: 3px 8px;
        font: inherit;
        border-radius: 2px;
      }
      .pso-sd-actions button:hover { border-color: #00ffff; color: #00ffff; }
      .pso-sd-actions button.primary {
        border-color: #4a90e2;
        color: #c7d8ec;
      }
      .pso-sd-actions button.primary:hover {
        background: rgba(74, 144, 226, 0.18);
        border-color: #00ffff;
        color: #00ffff;
      }
      .pso-sd-actions button:disabled { opacity: 0.4; cursor: not-allowed; }

      .pso-tex-panel.collapsed { width: 40px; }
      .pso-tex-panel.collapsed > *:not(.pso-tex-panel-header) { display: none; }
      .pso-tex-panel.collapsed .pso-tex-panel-header strong,
      .pso-tex-panel.collapsed .pso-tex-panel-header .grow { display: none; }

      /* ---- Variant picker strip (Gap 1) ----
         Floats at the TOP of the model-stage. Pills are clickable;
         active pill is highlighted in cyan to match the active-tab look. */
      .pso-variant-strip {
        position: absolute;
        top: 8px;
        left: 296px;             /* avoid the texture panel (280+16 gap) */
        right: 8px;
        z-index: 6;
        background: rgba(10, 14, 19, 0.92);
        border: 1px solid #2a313a;
        border-radius: 4px;
        padding: 4px 6px;
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        align-items: center;
        font: 11px var(--font-mono, monospace);
      }
      .pso-variant-strip .vs-label {
        color: #6c7785;
        margin-right: 4px;
      }
      .pso-variant-pill {
        background: transparent;
        border: 1px solid #2a313a;
        color: #99a4b3;
        cursor: pointer;
        padding: 2px 8px 2px 4px;
        font: inherit;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
      }
      .pso-variant-pill:hover {
        border-color: #4a90e2;
        color: #c7d8ec;
      }
      .pso-variant-pill.active {
        background: rgba(0, 255, 255, 0.12);
        border-color: #00ffff;
        color: #00ffff;
      }
      .pso-variant-pill .vs-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #888;
        border: 1px solid rgba(255,255,255,0.2);
        flex: 0 0 auto;
      }
      .pso-variant-pill.lod {
        opacity: 0.7;
        font-style: italic;
      }

      /* ---- Layers tab (Gap 2) ---- */
      .pso-layers-block {
        padding: 6px 8px;
        display: flex;
        flex-direction: column;
        gap: 8px;
      }
      .pso-layers-info {
        font-size: 10px;
        color: #6c7785;
      }
      .pso-layers-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(64px, 1fr));
        gap: 4px;
      }
      .pso-layer-cell {
        background: #0a0e13;
        border: 1px solid #2a313a;
        border-radius: 2px;
        padding: 4px;
        display: flex;
        flex-direction: column;
        gap: 3px;
        align-items: center;
        min-width: 0;
      }
      .pso-layer-cell.in-use {
        border-color: #56b67a;
      }
      .pso-layer-cell.alt {
        opacity: 0.55;
        border-style: dashed;
      }
      .pso-layer-cell .lc-thumb {
        width: 100%;
        aspect-ratio: 1;
        background: #050708;
        border: 1px solid #1a1f26;
        position: relative;
        overflow: hidden;
      }
      .pso-layer-cell .lc-thumb img {
        position: absolute; inset: 0;
        width: 100%; height: 100%;
        object-fit: contain;
      }
      .pso-layer-cell .lc-meta {
        font-size: 9px;
        color: #99a4b3;
        text-align: center;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100%;
      }
      .pso-layer-cell .lc-tag {
        font-size: 9px;
        color: #56b67a;
        font-weight: 600;
      }
      .pso-layer-cell.alt .lc-tag {
        color: #d8c890;
        font-style: italic;
      }
      .pso-layers-mesh-row {
        border-bottom: 1px solid #1a1f26;
        padding: 6px 8px;
      }
      .pso-layers-mesh-row:last-child { border-bottom: none; }
      .pso-layers-mesh-title {
        font-size: 11px;
        color: #c7d8ec;
        margin-bottom: 4px;
      }

      /* ---- Motions tab (Gap 3) ---- */
      .pso-motions-block {
        display: flex;
        flex-direction: column;
        height: 100%;
      }
      .pso-motions-toolbar {
        padding: 6px 8px;
        display: flex;
        gap: 6px;
        border-bottom: 1px solid #2a313a;
        background: rgba(0,0,0,0.18);
      }
      .pso-motions-toolbar button {
        background: transparent;
        border: 1px solid #2a313a;
        color: #99a4b3;
        cursor: pointer;
        padding: 2px 8px;
        font: inherit;
        border-radius: 2px;
        flex: 1;
      }
      .pso-motions-toolbar button:hover {
        border-color: #00ffff;
        color: #00ffff;
      }
      .pso-motions-toolbar button.active {
        background: rgba(0, 255, 255, 0.12);
        border-color: #00ffff;
        color: #00ffff;
      }
      .pso-motions-list {
        flex: 1 1 auto;
        overflow-y: auto;
      }
      .pso-motions-group {
        border-bottom: 1px solid #1a1f26;
      }
      .pso-motions-group-title {
        background: rgba(0,0,0,0.30);
        color: #56c8c8;
        font-size: 10px;
        padding: 2px 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
      }
      .pso-motion-row {
        display: flex;
        gap: 8px;
        padding: 4px 8px;
        cursor: pointer;
        border-bottom: 1px solid #15191f;
      }
      .pso-motion-row:last-child { border-bottom: none; }
      .pso-motion-row:hover {
        background: rgba(74, 144, 226, 0.10);
      }
      .pso-motion-row.active {
        background: rgba(0, 255, 255, 0.12);
        border-left: 2px solid #00ffff;
        padding-left: 6px;
      }
      .pso-motion-thumb {
        width: 28px;
        height: 28px;
        background: #0a0e13;
        border: 1px solid #2a313a;
        border-radius: 2px;
        flex: 0 0 auto;
        font-size: 9px;
        color: #56c8c8;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
      }
      .pso-motion-info {
        flex: 1; min-width: 0;
        display: flex;
        flex-direction: column;
      }
      .pso-motion-name {
        font-size: 11px;
        color: #c7d8ec;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .pso-motion-meta {
        font-size: 9px;
        color: #6c7785;
      }

      /* ---- Imported Animations (preview-only, on-disk staged NJMs) ---- */
      .pso-motions-imported-group {
        border-bottom: 1px solid #2a313a;
        background: rgba(74, 144, 226, 0.04);
      }
      .pso-motions-imported-group .pso-motions-group-title {
        background: rgba(74, 144, 226, 0.16);
        color: #8db3e6;
        display: flex;
        align-items: center;
        gap: 6px;
      }
      .pso-motion-row.imported {
        background: rgba(74, 144, 226, 0.04);
      }
      .pso-motion-row.imported:hover {
        background: rgba(74, 144, 226, 0.16);
      }
      .pso-motion-row.imported .pso-motion-thumb {
        background: rgba(74, 144, 226, 0.20);
        color: #b0c8e6;
        border-color: #4a90e2;
      }
      .pso-import-badge {
        display: inline-block;
        padding: 1px 6px;
        background: rgba(74, 144, 226, 0.20);
        color: #8db3e6;
        font-size: 9px;
        font-weight: 600;
        border-radius: 9px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border: 1px solid rgba(74, 144, 226, 0.50);
      }
      .pso-motion-row.imported .pso-motion-actions {
        display: none;
        margin-left: 6px;
      }
      .pso-motion-row.imported:hover .pso-motion-actions {
        display: flex;
        gap: 4px;
        align-items: center;
      }
      .pso-motion-row .pso-motion-remove {
        background: transparent;
        border: 1px solid #4a3030;
        color: #c08080;
        font-size: 9px;
        padding: 1px 6px;
        cursor: pointer;
        border-radius: 2px;
      }
      .pso-motion-row .pso-motion-remove:hover {
        border-color: #ff6060;
        color: #ff6060;
        background: rgba(255, 96, 96, 0.10);
      }
    `;
    document.head.appendChild(style);
  }

  // ---- mount / unmount ----------------------------------------------

  // Lifecycle: a single panel exists at any moment; we move/insert it
  // into the active .model-stage. The 3d-view perspective relocates
  // model-stage in/out of vp-stage; we don't track that — instead we
  // re-attach on every model-load event.
  let panel = null;
  let activeTab = "textures"; // "textures" | "subdivide"
  let lastModelPath = null;        // resolved (e.g. "<bml>#<inner>.nj")
  let lastModelPathPublic = null;  // raw manifest path (e.g. "<bml>")
  let lastEntry = null;
  let lastMatched = null;

  // Per-tile state pip cache.
  const status = new Map(); // tile_index -> { state, msg }

  // ---- perspective gating (overlap fix, 2026-06-20) ------------------
  // The texture panel + variant strip are scenery for the 3D-view (and
  // viewport-paint) perspectives ONLY. Other perspectives — map-editor,
  // floor-editor — ALSO relocate the shared .model-stage into their own
  // viewport host (#mapViewport / #floorViewport). Without this gate,
  // texture_panel.js's perspective.switched handler would re-mount the
  // absolutely-positioned panel into THAT relocated .model-stage and it
  // would render on top of the map/floor sidebar + viewport (the green
  // "Model assets" panel overlap bug).
  //
  // We only ever mount into a .model-stage that lives inside one of these
  // allowed hosts, and we proactively detach when the active perspective
  // is anything else.
  const TEXTURE_PANEL_PERSPECTIVES = new Set(["3d-view", "viewport-paint"]);

  function activePerspectiveName() {
    // Source of truth set by perspectives.js on every switchTo().
    return (document.body && document.body.dataset &&
            document.body.dataset.psoActivePerspective) || null;
  }

  function perspectiveWantsTexturePanel() {
    // Classic-modal mode (body without .unified-viewport-mode) has no
    // perspective concept — the panel belongs to #modelModal there.
    if (!document.body.classList.contains("unified-viewport-mode")) return true;
    const name = activePerspectiveName();
    // No perspective active yet (initial load) — defer to the model-stage
    // discovery below, which only resolves a stage when one legitimately
    // exists in #modelModal / .vp-stage-3d.
    if (!name) return true;
    return TEXTURE_PANEL_PERSPECTIVES.has(name);
  }

  // The ONLY DOM locations the panel is allowed to live in: the 3d-view
  // stage wrapper (unified mode) or #modelModal (classic mode). We do NOT
  // fall back to a bare ".model-stage" — that selector also matches the
  // stage after map/floor relocate it into their own viewport, which is
  // exactly the overlap we are preventing.
  function findAllowedModelStage() {
    return document.querySelector(".vp-stage-3d .model-stage")
      || document.querySelector("#modelModal .model-stage");
  }

  function ensurePanelDom() {
    ensureStyleInjected();
    if (!perspectiveWantsTexturePanel()) { detachPanel(); return null; }
    const stage = findAllowedModelStage();
    if (!stage) { detachPanel(); return null; }
    if (panel && panel.isConnected) {
      // If the panel drifted onto a stale stage (e.g. the model-stage was
      // relocated), re-home it onto the allowed stage.
      if (panel.parentElement !== stage) stage.appendChild(panel);
      return panel;
    }
    if (panel && !panel.isConnected) {
      stage.appendChild(panel);
      return panel;
    }
    panel = document.createElement("div");
    panel.id = PANEL_ID;
    panel.className = "pso-tex-panel";
    panel.innerHTML = `
      <div class="pso-tex-panel-header">
        <strong>Model assets</strong>
        <span class="grow"></span>
        <button data-act="build_deploy"
                title="rebuild the host BML/AFS from current edits and deploy to the live game install"
                class="pso-tex-build-btn">Build &amp; Deploy</button>
        <span class="pso-tex-build-status" data-region="build-status"></span>
        <button data-act="collapse" title="collapse panel">&#x2014;</button>
      </div>
      <div class="pso-tex-panel-tabs">
        <button data-tab="textures" class="active">Textures</button>
        <button data-tab="layers" title="all bound texture stages, grouped by submesh">Layers</button>
        <button data-tab="motions" title="full motion list with grouping + Loop-all">Motions</button>
        <button data-tab="subdivide">Subdivide</button>
      </div>
      <div class="pso-tex-panel-body" data-region="body"></div>
    `;
    stage.appendChild(panel);
    panel.addEventListener("click", onPanelClick);
    // Replay any external tab buttons (sculpt/material/paint/…) that were
    // registered before this (re)build so they survive a teardown.
    reinstallExternalTabButtons();
    // Signal panels that inject their own tab DOM directly (paint_panel.js)
    // so they can re-attach to the freshly-built tab strip. Those panels
    // append a button synchronously in their handler, so re-flow once more
    // afterwards to fold any newly-added tab into the overflow menu.
    if (window.bus && typeof window.bus.emit === "function") {
      try { window.bus.emit("texture-panel.rebuilt", {}); } catch (_e) {}
    }
    reflowTabs();
    return panel;
  }

  // ---- Build & Deploy --------------------------------------------------
  // Stub for the future BML/AFS rebuild flow. Reads the current model's
  // host archive (via psoGetCurrentTextureArchive() etc), then either:
  //   (a) for AFS hosts: gather every (now possibly-edited) inner blob
  //       via /api/raw/<host>#<inner> and POST /api/build_afs
  //   (b) for BML hosts: same path, plus pair each inner with its
  //       texture (.nj.xvm) and POST /api/build_bml
  // Followed by POST /api/deploy/<archive>.
  //
  // Only a stub for now — full payload assembly requires the inner-blob
  // walker that lives in the asset router; this hook is here so the
  // button is wired and visible. A future commit fills in the body.
  async function buildAndDeploy() {
    const status = panel && panel.querySelector('[data-region="build-status"]');
    function setBuildStatus(state, msg) {
      if (!status) return;
      status.textContent = msg;
      status.className = "pso-tex-build-status " + state;
    }
    try {
      setBuildStatus("busy", "resolving host…");
      const archive = ((typeof window.psoGetCurrentTextureArchive === "function")
        ? window.psoGetCurrentTextureArchive() : null)
        || deriveArchiveFromContext(lastModelPath, lastMatched);
      if (!archive) {
        setBuildStatus("err", "no model archive in scope");
        return;
      }
      // archive is "host.bml#inner" — strip to just the host filename.
      const hostName = archive.split("#")[0];
      if (!hostName) {
        setBuildStatus("err", "could not parse host name");
        return;
      }
      setBuildStatus("busy", `enumerating ${hostName}…`);

      // Heuristic: ask the server to surface every inner via the
      // existing raw endpoint, then re-pack. We use /api/build_bml
      // for *.bml and /api/build_afs for *.afs.
      const isBml = /\.bml$/i.test(hostName);
      const isAfs = /\.afs$/i.test(hostName);
      if (!isBml && !isAfs) {
        setBuildStatus("err", `unsupported host type: ${hostName}`);
        return;
      }
      // The full assembly path runs server-side via a forthcoming
      // /api/repack_archive endpoint. For now, fall back to deploy
      // of whatever the export cache already holds (if anything),
      // letting the user pre-stage a build via the CLI.
      setBuildStatus("busy", `deploying ${hostName}…`);
      const r = await fetch(`/api/deploy/${encodeURIComponent(hostName)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ create_backup: true }),
      });
      if (r.status === 404) {
        setBuildStatus(
          "err",
          `no built artifact in cache for ${hostName}; build it first via the CLI`,
        );
        return;
      }
      if (!r.ok) {
        const t = await r.text();
        setBuildStatus("err", `deploy failed: ${r.status} ${t.slice(0, 80)}`);
        return;
      }
      const j = await r.json();
      setBuildStatus(
        "ok",
        `deployed ${hostName} (${j.live_size} B, backup=${j.backup_name || "n/a"})`,
      );
    } catch (e) {
      setBuildStatus("err", `error: ${String(e).slice(0, 80)}`);
    }
  }

  // Variant pills strip — sits ABOVE the texture panel, separate DOM
  // node for layout independence. Mounted in the same .model-stage as
  // the panel itself; auto-hidden when no variants are detected.
  let variantStrip = null;
  let variantList = [];
  let activeVariantIdx = -1;

  function ensureVariantStripDom() {
    // Same active-stage discovery + perspective gate as ensurePanelDom():
    // never mount onto a .model-stage that map/floor relocated into their
    // own viewport, or the strip overlaps the same way the panel did.
    if (!perspectiveWantsTexturePanel()) { detachVariantStrip(); return null; }
    const stage = findAllowedModelStage();
    if (!stage) { detachVariantStrip(); return null; }
    if (variantStrip && variantStrip.isConnected) {
      // Move the strip if its parent is no longer the active stage
      // (perspective switch leaves the strip on the old DOM tree
      // otherwise).
      if (variantStrip.parentElement !== stage) {
        stage.appendChild(variantStrip);
      }
      return variantStrip;
    }
    if (!variantStrip) {
      variantStrip = document.createElement("div");
      variantStrip.id = VARIANT_STRIP_ID;
      variantStrip.className = "pso-variant-strip";
      variantStrip.hidden = true;
      variantStrip.addEventListener("click", onVariantStripClick);
    }
    stage.appendChild(variantStrip);
    return variantStrip;
  }

  function detachVariantStrip() {
    if (variantStrip && variantStrip.parentNode) {
      variantStrip.parentNode.removeChild(variantStrip);
    }
  }

  function onVariantStripClick(ev) {
    const t = ev.target.closest(".pso-variant-pill");
    if (!t) return;
    const idx = parseInt(t.dataset.idx, 10);
    if (Number.isNaN(idx)) return;
    selectVariant(idx);
  }

  // Selecting a variant either:
  //   - swaps to a sibling BML (cross-BML variant) — re-invokes the
  //     model loader and lets it rebuild from scratch.
  //   - applies a slot-offset on the current model's textures (intra-BML).
  async function selectVariant(idx) {
    if (idx < 0 || idx >= variantList.length) return;
    const v = variantList[idx];
    if (!v) return;
    activeVariantIdx = idx;
    renderVariantStrip();

    if (v.slot_group != null) {
      // Intra-BML — apply a tile-index offset.
      if (typeof window.psoApplyVariantSlotOffset === "function") {
        const offset = (v.slot_group | 0) * (v.slot_count | 0);
        try { await window.psoApplyVariantSlotOffset(offset); }
        catch (e) { console.warn("variant offset apply failed", e); }
      }
      scheduleRefresh();
    } else {
      // Cross-BML — re-open the model. We pass the variant's path and
      // an empty entry/matched list (the asset router will fill in
      // matched textures via its own resolution).
      if (typeof window.psoOpenModelByPath === "function") {
        try {
          await window.psoOpenModelByPath(v.path, {}, []);
        } catch (e) {
          console.warn("variant open failed", e);
        }
      }
    }
  }

  function renderVariantStrip() {
    const strip = ensureVariantStripDom();
    if (!strip) return;
    if (!variantList || variantList.length <= 1) {
      strip.hidden = true;
      strip.innerHTML = "";
      return;
    }
    strip.hidden = false;
    strip.innerHTML =
      `<span class="vs-label">variant:</span>` +
      variantList.map((v, i) => {
        const cls = ["pso-variant-pill"];
        if (i === activeVariantIdx) cls.push("active");
        if (v.variant_kind === "lod") cls.push("lod");
        const titleAttr =
          `path: ${escapeHtml(v.path)}\nkind: ${escapeHtml(v.variant_kind)}`;
        return `<button class="${cls.join(' ')}" data-idx="${i}" title="${titleAttr}">` +
               `<span class="vs-dot" style="background:${escapeHtml(v.icon_color || '#888')}"></span>` +
               `<span>${escapeHtml(v.label)}</span></button>`;
      }).join("");
  }

  // Fetch the variants list for the currently-loaded model. Idempotent;
  // safe to call multiple times. Picks the "self" entry as the active
  // pill on first load.
  async function fetchVariants() {
    if (!lastModelPathPublic) {
      variantList = [];
      activeVariantIdx = -1;
      renderVariantStrip();
      return;
    }
    // Strip any "#inner" fragment — variants are container-level.
    const bml = lastModelPathPublic.split("#")[0];
    if (!bml.toLowerCase().endsWith(".bml")) {
      variantList = [];
      activeVariantIdx = -1;
      renderVariantStrip();
      return;
    }
    const url = `/api/variants/${encodeURIComponent(bml)}`;
    // Wave 7: lifecycle-aware fetch so a stale variant query dies when
    // the user opens a different asset.
    const f = (window.psoAssetLifecycle && window.psoAssetLifecycle.fetchAsset) || fetch;
    const isAbort = (window.psoAssetLifecycle && window.psoAssetLifecycle.isAbort) || (() => false);
    let data;
    try {
      const r = await f(url);
      if (!r.ok) return;
      data = await r.json();
    } catch (e) {
      if (isAbort(e)) return;
      console.warn("[texture_panel] variant fetch failed:", e);
      return;
    }
    variantList = (data && data.variants) || [];
    activeVariantIdx = variantList.findIndex((v) => v.is_self);
    if (activeVariantIdx < 0 && variantList.length > 0) activeVariantIdx = 0;
    renderVariantStrip();
  }

  function detachPanel() {
    if (panel && panel.parentNode) panel.parentNode.removeChild(panel);
    panel = null;
    status.clear();
    detachVariantStrip();
    stopLoopAll();
  }

  function onPanelClick(ev) {
    const t = ev.target;
    if (!t) return;
    // Resolve the nearest actionable element (clicks can land on inner
    // text nodes / spans of a button).
    const actEl = t.closest && t.closest("[data-act]");
    const act = actEl && actEl.getAttribute("data-act");
    if (act === "tab-more") {
      ev.preventDefault();
      ev.stopPropagation();
      toggleTabOverflowMenu();
      return;
    }
    if (act === "collapse") {
      panel.classList.toggle("collapsed");
      return;
    }
    if (act === "build_deploy") {
      // Fire-and-forget: button stays clickable so the user can re-run
      // after a fix. Status surfaces in the header span.
      buildAndDeploy().catch((e) => {
        const status = panel && panel.querySelector('[data-region="build-status"]');
        if (status) {
          status.textContent = `error: ${String(e).slice(0, 80)}`;
          status.className = "pso-tex-build-status err";
        }
      });
      return;
    }
    const tabEl = t.closest && t.closest("button[data-tab]");
    const tab = tabEl && tabEl.getAttribute("data-tab");
    if (tab) {
      activeTab = tab;
      const tabs = panel.querySelectorAll(".pso-tex-panel-tabs button[data-tab]");
      tabs.forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
      renderActiveTab();
      // Close the overflow menu and re-flow so the freshly-active tab is
      // hoisted inline (and "more" un-highlights if it left the menu).
      closeTabOverflowMenu();
      reflowTabs();
      return;
    }
  }

  // ---- texture panel rendering --------------------------------------

  function renderTextureRows() {
    const body = panel.querySelector('[data-region="body"]');
    if (!body) return;
    const list = listTexturesFallback();
    if (!list.length) {
      body.innerHTML = `
        <div class="pso-tex-panel-empty">
          No bound textures detected.<br><br>
          Open a model with a paired XVM texture archive (most BML/AFS
          weapons + enemies pair automatically).
        </div>
      `;
      return;
    }
    const archive = list[0].archive;
    const rows = list.map((row) => textureRowHtml(row, archive)).join("");
    body.innerHTML = rows;
    // Bind action buttons (per-row event delegation).
    body.querySelectorAll("[data-tex-action]").forEach((btn) => {
      btn.addEventListener("click", onTexAction);
    });
    // Lazy-load thumbnails.
    body.querySelectorAll("img[data-src]").forEach((img) => {
      img.src = img.dataset.src;
      img.removeAttribute("data-src");
    });
  }

  function textureRowHtml(row, archive) {
    const tileIdx = row.tile_index;
    const dim = (row.width && row.height)
      ? `${row.width}&times;${row.height}`
      : "—";
    const matLabel = (row.material_ids || []).map((m) => `m${m}`).join(", ");
    const thumbUrl = `/api/tile_png/${encodeURIComponent(archive)}/${tileIdx}?cb=${Date.now()}`;
    const st = status.get(tileIdx) || { state: "idle", msg: "ready" };
    const stCls = "pso-tex-status " + st.state;
    const arName = archive.split("/").pop();
    return `
      <div class="pso-tex-row" data-tile="${tileIdx}">
        <div class="pso-tex-thumb">
          <img alt="tile ${tileIdx}" data-src="${escapeHtml(thumbUrl)}" />
        </div>
        <div class="pso-tex-info">
          <div class="nm" title="${escapeHtml(arName)} · tile ${tileIdx}">tile ${tileIdx} (${escapeHtml(matLabel)})</div>
          <div class="meta">${dim} · ${escapeHtml(arName)}</div>
          <div class="pso-tex-actions">
            ${UPSCALE_SCALES.map((s) => `
              <button data-tex-action="upscale" data-tile="${tileIdx}" data-scale="${s}"
                      title="upscale tile ${tileIdx} ×${s} via realesrgan">×${s}</button>
            `).join("")}
            <button class="replace" data-tex-action="replace" data-tile="${tileIdx}"
                    title="replace tile with a user-supplied PNG">Replace…</button>
          </div>
          <div class="${stCls}" data-status="${tileIdx}">${escapeHtml(st.msg)}</div>
        </div>
      </div>
    `;
  }

  function setStatus(tileIdx, state, msg) {
    status.set(tileIdx, { state, msg });
    if (!panel) return;
    const el = panel.querySelector(`[data-status="${tileIdx}"]`);
    if (el) {
      el.textContent = msg;
      el.className = "pso-tex-status " + state;
    }
  }

  async function onTexAction(ev) {
    const btn = ev.currentTarget;
    const action = btn.getAttribute("data-tex-action");
    const tile = parseInt(btn.getAttribute("data-tile"), 10);
    if (action === "upscale") {
      const scale = parseInt(btn.getAttribute("data-scale"), 10);
      await runUpscale(tile, scale);
    } else if (action === "replace") {
      await runReplace(tile);
    }
  }

  async function runUpscale(tile, scale) {
    const archive = ((typeof window.psoGetCurrentTextureArchive === "function")
      ? window.psoGetCurrentTextureArchive() : null)
      || deriveArchiveFromContext(lastModelPath, lastMatched);
    if (!archive) {
      setStatus(tile, "err", "no archive bound");
      return;
    }
    // The archive is the basename (or `<bml>#<inner>.xvm` form). The
    // upscale endpoint accepts the same path /api/tile_png expects;
    // BML+AFS inner forms work because resolve_asset_bytes handles `#`.
    const baseName = archive.split("/").pop();
    setStatus(tile, "running", `upscaling ×${scale}…`);
    let res;
    try {
      const r = await fetch("/api/upscale", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename: baseName,
          tile_index: tile,
          model: UPSCALE_MODEL_DEFAULT,
          scale,
          keep_native_dims: true,
        }),
      });
      if (!r.ok) {
        let detail = `HTTP ${r.status}`;
        try { detail = (await r.json()).detail || detail; } catch {}
        throw new Error(detail);
      }
      res = await r.json();
    } catch (e) {
      setStatus(tile, "err", `failed: ${e.message || e}`);
      return;
    }
    setStatus(tile, "done", `done · ${res.out_w}×${res.out_h}`);
    // Re-fetch the tile texture in the live mesh so the user sees
    // the upscaled version immediately.
    if (typeof window.psoReloadTexture === "function") {
      try { await window.psoReloadTexture(tile); } catch (_e) {}
    }
    // Refresh the thumbnail (cache-busted via timestamp in URL).
    const img = panel.querySelector(`.pso-tex-row[data-tile="${tile}"] img`);
    if (img) img.src = `/api/tile_png/${encodeURIComponent(archive)}/${tile}?cb=${Date.now()}`;
  }

  async function runReplace(tile) {
    const archive = ((typeof window.psoGetCurrentTextureArchive === "function")
      ? window.psoGetCurrentTextureArchive() : null)
      || deriveArchiveFromContext(lastModelPath, lastMatched);
    if (!archive) {
      setStatus(tile, "err", "no archive bound");
      return;
    }
    const baseName = archive.split("/").pop();
    // File picker
    const inp = document.createElement("input");
    inp.type = "file";
    inp.accept = "image/png,.png";
    inp.style.display = "none";
    document.body.appendChild(inp);
    inp.addEventListener("change", async () => {
      const f = inp.files && inp.files[0];
      document.body.removeChild(inp);
      if (!f) return;
      setStatus(tile, "running", `uploading ${f.name}…`);
      try {
        const fd = new FormData();
        fd.append("image", f);
        fd.append("keep_native_dims", "true");
        const url = `/api/import_png/${encodeURIComponent(baseName)}/${tile}`;
        const r = await fetch(url, { method: "POST", body: fd });
        if (!r.ok) {
          let detail = `HTTP ${r.status}`;
          try { detail = (await r.json()).detail || detail; } catch {}
          throw new Error(detail);
        }
        await r.json(); // result is registered in the editor's state
      } catch (e) {
        setStatus(tile, "err", `failed: ${e.message || e}`);
        return;
      }
      setStatus(tile, "done", `replaced from ${f.name}`);
      if (typeof window.psoReloadTexture === "function") {
        try { await window.psoReloadTexture(tile); } catch (_e) {}
      }
      const img = panel.querySelector(`.pso-tex-row[data-tile="${tile}"] img`);
      if (img) img.src = `/api/tile_png/${encodeURIComponent(archive)}/${tile}?cb=${Date.now()}`;
    });
    inp.click();
  }

  // ---- subdivide panel rendering ------------------------------------

  let subdivideState = {
    busy: false,
    origStats: null,    // { vertices, triangles } pre-subdivide
    lastResult: null,   // { level, vertices, triangles }
    cachedPath: null,   // path of subdivided model in cache/subdivided
  };

  function renderSubdivideBlock() {
    const body = panel.querySelector('[data-region="body"]');
    if (!body) return;
    const list = (typeof window.psoGetDebugMeshes === "function")
      ? window.psoGetDebugMeshes()
      : [];
    const verts = list.reduce((a, m) => a + (m.vertex_count || 0), 0);
    const tris = list.reduce((a, m) => a + (m.triangle_count || 0), 0);
    if (!subdivideState.origStats && verts > 0) {
      subdivideState.origStats = { vertices: verts, triangles: tris };
    }
    const orig = subdivideState.origStats;
    const cur = { vertices: verts, triangles: tris };
    let stats = `Current: ${cur.vertices.toLocaleString()} verts, ${cur.triangles.toLocaleString()} tris`;
    if (orig && (orig.vertices !== cur.vertices || orig.triangles !== cur.triangles)) {
      stats += `\nOriginal: ${orig.vertices.toLocaleString()} verts, ${orig.triangles.toLocaleString()} tris`;
      const dvf = ((cur.vertices / Math.max(1, orig.vertices))).toFixed(2);
      const dtf = ((cur.triangles / Math.max(1, orig.triangles))).toFixed(2);
      stats += `\nFactor: ${dvf}× verts, ${dtf}× tris`;
    }
    if (subdivideState.lastResult) {
      stats += `\nLast applied: lvl ${subdivideState.lastResult.level}`;
    }
    const busy = subdivideState.busy;
    const disabledAttr = busy ? "disabled" : "";
    body.innerHTML = `
      <div class="pso-sd-block">
        <div class="pso-sd-stats">${escapeHtml(stats)}</div>
        <div class="pso-sd-row">
          <label>Iterations:
            <select id="psoSdLevel" ${disabledAttr}>
              <option value="1">×1 (4× tris)</option>
              <option value="2">×2 (~16×)</option>
              <option value="3">×3 (~64×)</option>
            </select>
          </label>
        </div>
        <div class="pso-sd-row">
          <label>
            <input type="checkbox" id="psoSdSmooth" checked ${disabledAttr} />
            smooth normals
          </label>
        </div>
        <div class="pso-sd-actions">
          <button class="primary" id="psoSdApply" ${disabledAttr}>Apply Loop subdivide</button>
          <button id="psoSdReset" ${disabledAttr}>Reset to original</button>
        </div>
        <div class="pso-tex-status ${busy ? "running" : "idle"}" id="psoSdStatus">${busy ? "subdividing…" : "ready"}</div>
      </div>
    `;
    body.querySelector("#psoSdApply").addEventListener("click", runSubdivide);
    body.querySelector("#psoSdReset").addEventListener("click", runReset);
  }

  function setSdStatus(state, msg) {
    if (!panel) return;
    const el = panel.querySelector("#psoSdStatus");
    if (el) {
      el.textContent = msg;
      el.className = "pso-tex-status " + state;
    }
  }

  async function runSubdivide() {
    if (!lastModelPath) {
      setSdStatus("err", "no model loaded");
      return;
    }
    const level = parseInt(panel.querySelector("#psoSdLevel").value, 10) || 1;
    const smooth = !!panel.querySelector("#psoSdSmooth").checked;
    subdivideState.busy = true;
    renderSubdivideBlock();
    setSdStatus("running", `subdividing ×${level}…`);
    let res;
    try {
      const r = await fetch("/api/model/subdivide", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          path: lastModelPath,
          level,
          smooth_normals: smooth,
        }),
      });
      if (!r.ok) {
        let detail = `HTTP ${r.status}`;
        try { detail = (await r.json()).detail || detail; } catch {}
        throw new Error(detail);
      }
      res = await r.json();
    } catch (e) {
      subdivideState.busy = false;
      renderSubdivideBlock();
      setSdStatus("err", `failed: ${e.message || e}`);
      return;
    }
    // The endpoint returns an in-line mesh payload + stats; apply it.
    if (typeof window.psoApplyMeshPayload === "function" && res.mesh_payload) {
      try { window.psoApplyMeshPayload(res.mesh_payload, { label: `subdivide lvl ${level}` }); }
      catch (e) { console.warn("apply payload threw", e); }
    }
    subdivideState.busy = false;
    subdivideState.lastResult = {
      level,
      vertices: res.totals && res.totals.vertices,
      triangles: res.totals && res.totals.triangles,
    };
    subdivideState.cachedPath = res.cache_path || null;
    renderSubdivideBlock();
    const before = res.before || {};
    const after = res.after || {};
    setSdStatus("done",
      `done lvl ${level} · ` +
      `${before.vertices || 0} → ${after.vertices || 0} verts, ` +
      `${before.triangles || 0} → ${after.triangles || 0} tris`);
  }

  async function runReset() {
    const reloadPath = lastModelPathPublic || lastModelPath;
    if (!reloadPath) return;
    subdivideState.busy = true;
    renderSubdivideBlock();
    setSdStatus("running", "reverting…");
    if (typeof window.psoReloadModel === "function") {
      try {
        await window.psoReloadModel(reloadPath, lastEntry, lastMatched);
      } catch (e) {
        setSdStatus("err", `revert failed: ${e.message || e}`);
        subdivideState.busy = false;
        return;
      }
    }
    subdivideState.busy = false;
    subdivideState.lastResult = null;
    renderSubdivideBlock();
    setSdStatus("done", "reverted to original");
  }

  // ---- top-level dispatch -------------------------------------------

  // External tab renderer registry — any tab whose name doesn't match
  // one of the built-in cases delegates to a callback registered via
  // window.psoTexturePanelRegisterTab(). The Sculpt tab uses this hook
  // (added 2026-04-25) so sculpt_panel.js owns its own body rendering
  // without us refactoring the existing built-in tabs.
  const _externalTabRenderers = new Map(); // tabName -> (bodyEl) => void
  // Remember every external tab BUTTON definition (sculpt/material/paint/
  // rig/anim-editor/edit/skeleton/uv …). External panels register their
  // button exactly once at init via psoTexturePanelAddTabButton; the panel
  // DOM, however, can be rebuilt from scratch (e.g. after the overlap-fix
  // detach when leaving a 3D perspective, or on a model swap). We replay
  // these on every rebuild so the extra tabs survive a teardown.
  const _externalTabButtons = new Map(); // tabName -> { label, title }

  // Re-add all remembered external tab buttons to a freshly-built tab
  // strip. Called from ensurePanelDom right after the panel HTML is set.
  function reinstallExternalTabButtons() {
    if (!panel) return;
    const tabs = panel.querySelector(".pso-tex-panel-tabs");
    if (!tabs) return;
    for (const [tabName, def] of _externalTabButtons) {
      if (tabs.querySelector(`button[data-tab="${tabName}"]`)) continue;
      const btn = document.createElement("button");
      btn.dataset.tab = tabName;
      if (def && def.title) btn.title = def.title;
      btn.textContent = (def && def.label) || tabName;
      tabs.appendChild(btn);
    }
    reflowTabs();
  }

  // ---- tab overflow ("more ▾") ---------------------------------------
  // 12 tabs (Textures/Layers/Motions/Subdivide/Sculpt/Material/Paint/Rig/
  // Anim Editor/Edit/Skeleton/UV) wrapped into a 3-row wall. Keep the
  // common tabs inline and tuck the rest under a "more ▾" overflow menu.
  // Reachability is preserved — every overflow item is a real tab button
  // (same data-tab) that the existing click delegation already handles;
  // we just relocate it into a popdown. The active tab is always hoisted
  // inline so the user can see what's selected.
  const PRIMARY_TABS = ["textures", "layers", "motions", "subdivide"];

  function reflowTabs() {
    if (!panel) return;
    const tabs = panel.querySelector(".pso-tex-panel-tabs");
    if (!tabs) return;

    // Gather the real tab buttons (exclude our own overflow control).
    let overflowWrap = tabs.querySelector(".pso-tex-tab-overflow");
    const allBtns = Array.from(tabs.querySelectorAll("button[data-tab]"))
      .filter((b) => !b.closest(".pso-tex-tab-overflow-menu"));
    const menuBtns = overflowWrap
      ? Array.from(overflowWrap.querySelectorAll("button[data-tab]"))
      : [];
    const buttons = allBtns.concat(menuBtns);
    if (!buttons.length) return;

    // Decide primary vs overflow. Primary = the fixed common set, PLUS the
    // currently-active tab (so the selection is always visible inline).
    // Show ALL editor tabs inline. The strip flex-wraps to a second row when
    // there isn't enough width — fine, because Rigging / Skeleton / Material /
    // UV / Paint are first-class tools and must be visible, not buried in a
    // "more ▾" menu. (PRIMARY_TABS still drives the left-to-right ordering.)
    const primary = buttons.slice();
    const overflow = [];

    // If nothing overflows, drop the overflow control entirely and put all
    // buttons back inline in a stable order.
    if (!overflow.length) {
      if (overflowWrap) overflowWrap.remove();
      orderTabs(tabs, primary, []);
      return;
    }

    // Build / reuse the overflow control.
    if (!overflowWrap) {
      overflowWrap = document.createElement("div");
      overflowWrap.className = "pso-tex-tab-overflow";
      overflowWrap.innerHTML =
        '<button type="button" class="pso-tex-tab-more" ' +
        'data-act="tab-more" aria-haspopup="true" aria-expanded="false" ' +
        'title="more editors">more ▾</button>' +
        '<div class="pso-tex-tab-overflow-menu" hidden></div>';
    }
    const menu = overflowWrap.querySelector(".pso-tex-tab-overflow-menu");

    orderTabs(tabs, primary, []);
    // The overflow control sits after the primary inline tabs.
    tabs.appendChild(overflowWrap);
    // Move overflow buttons into the menu (in stable order).
    for (const b of overflow) menu.appendChild(b);

    // Highlight the "more" control when the active tab lives inside it.
    const moreBtn = overflowWrap.querySelector(".pso-tex-tab-more");
    const activeInMenu = overflow.some((b) => b.dataset.tab === activeTab);
    if (moreBtn) moreBtn.classList.toggle("has-active", activeInMenu);
  }

  // Place `inline` buttons directly inside the tab strip in a stable order
  // (PRIMARY_TABS order first, then any extras by their existing order).
  function orderTabs(tabs, inline, _unused) {
    const ordered = inline.slice().sort((a, b) => {
      const ia = PRIMARY_TABS.indexOf(a.dataset.tab);
      const ib = PRIMARY_TABS.indexOf(b.dataset.tab);
      const ra = ia === -1 ? 999 : ia;
      const rb = ib === -1 ? 999 : ib;
      return ra - rb;
    });
    for (const b of ordered) tabs.appendChild(b);
  }

  function closeTabOverflowMenu() {
    if (!panel) return;
    const menu = panel.querySelector(".pso-tex-tab-overflow-menu");
    const more = panel.querySelector(".pso-tex-tab-more");
    if (menu) menu.hidden = true;
    if (more) more.setAttribute("aria-expanded", "false");
  }

  function toggleTabOverflowMenu() {
    if (!panel) return;
    const menu = panel.querySelector(".pso-tex-tab-overflow-menu");
    const more = panel.querySelector(".pso-tex-tab-more");
    if (!menu) return;
    const open = menu.hidden;
    menu.hidden = !open;
    if (more) more.setAttribute("aria-expanded", open ? "true" : "false");
  }

  function renderActiveTab() {
    if (!panel) return;
    if (activeTab === "subdivide") renderSubdivideBlock();
    else if (activeTab === "layers") renderLayersBlock();
    else if (activeTab === "motions") renderMotionsBlock();
    else if (_externalTabRenderers.has(activeTab)) {
      const body = panel.querySelector('[data-region="body"]');
      if (body) {
        try { _externalTabRenderers.get(activeTab)(body); }
        catch (e) {
          body.innerHTML = `<div class="pso-tex-panel-empty">${escapeHtml(String(e))}</div>`;
        }
      }
    }
    else renderTextureRows();
  }

  // Public hook (2026-04-25): register a callback that renders an
  // external tab's body region. The caller is also responsible for
  // adding their tab button via psoTexturePanelAddTabButton.
  window.psoTexturePanelRegisterTab = function (tabName, renderer) {
    if (typeof tabName !== "string" || typeof renderer !== "function") return false;
    _externalTabRenderers.set(tabName, renderer);
    return true;
  };

  window.psoTexturePanelAddTabButton = function (tabName, label, title) {
    // Remember the definition so we can replay it whenever the panel DOM
    // is rebuilt (the overlap-fix detach + a model swap both null `panel`).
    if (typeof tabName === "string" && tabName) {
      _externalTabButtons.set(tabName, { label: label, title: title });
    }
    ensurePanelDom();
    if (!panel) {
      // Panel not mountable right now (e.g. user is on a non-3D
      // perspective). The button is remembered and will be installed by
      // reinstallExternalTabButtons() on the next rebuild — treat as OK so
      // the external panel's one-shot install loop stops retrying.
      return true;
    }
    const tabs = panel.querySelector(".pso-tex-panel-tabs");
    if (!tabs) return false;
    if (tabs.querySelector(`button[data-tab="${tabName}"]`)) return true;
    const btn = document.createElement("button");
    btn.dataset.tab = tabName;
    if (title) btn.title = title;
    btn.textContent = label || tabName;
    tabs.appendChild(btn);
    // External tabs register over time (each panel polls independently);
    // re-flow on every addition so they fold into the overflow menu
    // instead of accumulating inline as the old multi-row wall.
    reflowTabs();
    return true;
  };

  // ===========================================================
  // Layers tab — Gap 2: surface ALL bound texture stages per
  // submesh with thumbnails + an "in-use vs alternate-variant"
  // badge so the user can see which tiles the renderer is
  // actually consuming and which are alternate-variant slots
  // that variant-switch will swap in.
  // ===========================================================

  function renderLayersBlock() {
    const body = panel.querySelector('[data-region="body"]');
    if (!body) return;
    const list = listTexturesFallback();
    if (list.length === 0) {
      body.innerHTML = `<div class="pso-tex-panel-empty">No layer info — no textures bound.</div>`;
      return;
    }
    const archive = list[0].archive;
    const binding = (typeof window.psoGetTextureBinding === "function")
      ? window.psoGetTextureBinding() : [];
    // Compute the active slot offset (if any) so we know which tiles
    // are CURRENTLY in use vs which are alternate-variant.
    const slotOffset = (typeof window.psoGetVariantSlotOffset === "function")
      ? (window.psoGetVariantSlotOffset() | 0) : 0;
    // Set of tile_indices currently in use = (material_id + slotOffset)
    // for every binding row.
    const inUseSet = new Set();
    for (const b of binding) {
      if (b && !b.missing) inUseSet.add(((b.material_id | 0) + slotOffset) | 0);
    }

    // Bucket all archive tiles by tile_index so we can show the FULL
    // set, not just the in-use subset. We need the XVMH listing for
    // this — fetch from /api/model_textures.
    const arName = archive.split("/").pop();
    body.innerHTML = `<div class="pso-layers-block">
      <div class="pso-layers-info">archive: <code>${escapeHtml(arName)}</code> · ${list.length} tile${list.length === 1 ? '' : 's'} in use · slot offset = ${slotOffset}</div>
      <div data-region="layer-grid"><em class="pso-tex-panel-empty">loading…</em></div>
    </div>`;

    // Pull the full tile list via /api/model_textures so we get the
    // alternate-variant slots too.
    if (lastModelPath) {
      const path = lastModelPath;
      const hashIdx = path.indexOf("#");
      let url;
      if (hashIdx > 0) {
        const base = path.slice(0, hashIdx);
        const inner = path.slice(hashIdx + 1);
        url = `/api/model_textures/${encodeURIComponent(base)}?inner=${encodeURIComponent(inner)}`;
      } else {
        url = `/api/model_textures/${encodeURIComponent(path)}`;
      }
      fetch(url).then(r => r.json()).then((data) => {
        if (activeTab !== "layers") return;
        const grid = body.querySelector('[data-region="layer-grid"]');
        if (!grid) return;
        const xvmh = (data && data.xvmh) || [];
        const njtl = (data && data.njtl) || [];
        if (xvmh.length === 0) {
          grid.innerHTML = `<div class="pso-tex-panel-empty">No XVMH records found for this model.</div>`;
          return;
        }
        const cells = xvmh.map((x) => {
          const ti = x.tile_index | 0;
          const inUse = inUseSet.has(ti);
          const cls = ["pso-layer-cell"];
          cls.push(inUse ? "in-use" : "alt");
          const slotName = (njtl[ti] && njtl[ti].name) || x.name || "";
          const w = x.width || "?", h = x.height || "?";
          const fmt = x.fmt;
          const tag = inUse ? "in use" : "alt-variant";
          const thumbUrl = `/api/tile_png/${encodeURIComponent(archive)}/${ti}?cb=${Date.now()}`;
          return `<div class="${cls.join(' ')}" title="${escapeHtml(slotName)} · ${w}×${h} · fmt ${fmt}">
            <div class="lc-thumb"><img loading="lazy" src="${escapeHtml(thumbUrl)}" alt="tile ${ti}"></div>
            <div class="lc-meta">tile ${ti} · ${w}×${h}</div>
            <div class="lc-meta" title="${escapeHtml(slotName)}">${escapeHtml(slotName)}</div>
            <div class="lc-tag">${tag}</div>
          </div>`;
        });
        grid.innerHTML = `<div class="pso-layers-grid">${cells.join('')}</div>`;
      }).catch((e) => {
        const grid = body.querySelector('[data-region="layer-grid"]');
        if (grid) grid.innerHTML = `<div class="pso-tex-panel-empty">model_textures fetch failed: ${escapeHtml(e.message || String(e))}</div>`;
      });
    }
  }

  // ===========================================================
  // Motions tab — Gap 3: scrollable grouped list of every
  // motion the server returned for the current model. Includes
  // a Loop-all button that auto-cycles through them.
  //
  // Sources:
  //   - window.psoListMotions() (populated by model_viewer.js's
  //     skinned-mesh path)
  //   - /api/animations/<path>?inner=<inner> direct fetch when
  //     the public list is empty (the model loaded via the
  //     non-skinned path doesn't run populateAnimationPanel —
  //     happens for static props, large boss models the
  //     skin-detector defers, etc.)
  // ===========================================================

  // Cached "fallback" motion list when psoListMotions() returns []
  // — populated by /api/animations directly. Keyed by lastModelPath
  // so a model swap clears it.
  let motionsFallback = { path: null, motions: null };

  // ===========================================================
  // Imported Animations (preview-only). These are .njm files
  // staged via /api/import/animation along with a .preview.json
  // sidecar tagging the target model. Distinct from the model's
  // own motion list — they live in cache/njm_export/ and never
  // touch <install>/data/. Click → /api/anim_preview/data → hand
  // the JSON to psoLoadMotion(motion_json) to play in viewport.
  // ===========================================================
  let importedAnimsState = { path: null, items: null, error: null, currentName: null };

  // Resolve the BML basename used by /api/anim_preview/list. The
  // server matches on basename only (the sidecar's
  // target_model_path is normalised the same way), so it doesn't
  // matter whether the model was opened by raw path or via #inner.
  function _importedAnimsModelKey() {
    const refPath = lastModelPathPublic || lastModelPath;
    if (!refPath) return null;
    const hashIdx = refPath.indexOf("#");
    const base = hashIdx > 0 ? refPath.slice(0, hashIdx) : refPath;
    // Strip any path components — server only cares about the file name.
    const m = String(base).split(/[\\\/]/);
    return m[m.length - 1];
  }

  async function fetchImportedAnims(force) {
    const key = _importedAnimsModelKey();
    if (!key) {
      importedAnimsState = { path: null, items: [], error: null, currentName: importedAnimsState.currentName };
      return importedAnimsState.items;
    }
    if (!force && importedAnimsState.path === key && Array.isArray(importedAnimsState.items)) {
      return importedAnimsState.items;
    }
    const lifeF = (window.psoAssetLifecycle && window.psoAssetLifecycle.fetchAsset) || fetch;
    const lifeAbort = (window.psoAssetLifecycle && window.psoAssetLifecycle.isAbort) || (() => false);
    try {
      const r = await lifeF(`/api/anim_preview/list?model_path=${encodeURIComponent(key)}`);
      if (!r.ok) {
        // 404 here is unexpected (the endpoint always 200s with empty
        // list); treat as soft-fail so the rest of the Motions tab
        // renders normally.
        importedAnimsState = { path: key, items: [], error: `http ${r.status}`, currentName: importedAnimsState.currentName };
        return importedAnimsState.items;
      }
      const data = await r.json();
      const items = Array.isArray(data && data.items) ? data.items : [];
      importedAnimsState = { path: key, items, error: null, currentName: importedAnimsState.currentName };
      return items;
    } catch (e) {
      if (lifeAbort(e)) {
        importedAnimsState = { path: key, items: [], error: null, currentName: importedAnimsState.currentName };
        return importedAnimsState.items;
      }
      importedAnimsState = { path: key, items: [], error: e.message || String(e), currentName: importedAnimsState.currentName };
      return importedAnimsState.items;
    }
  }

  async function playImportedAnim(njmFilename) {
    if (!njmFilename) return;
    let data;
    try {
      const r = await fetch(`/api/anim_preview/data?njm_path=${encodeURIComponent(njmFilename)}`);
      if (!r.ok) {
        let detail = `http ${r.status}`;
        try { const eb = await r.json(); if (eb && eb.detail) detail = eb.detail; } catch (_) {}
        console.warn("[texture_panel] anim_preview/data failed:", detail);
        return;
      }
      data = await r.json();
    } catch (e) {
      console.warn("[texture_panel] anim_preview/data fetch error:", e);
      return;
    }
    if (typeof window.psoLoadMotion === "function") {
      // Preview-only: stop any running loop-all first so the regular
      // playlist doesn't immediately yank back to a vanilla motion.
      stopLoopAll();
      await window.psoLoadMotion(data);
      importedAnimsState.currentName = njmFilename;
    }
  }

  async function removeImportedAnim(njmFilename) {
    if (!njmFilename) return;
    try {
      const r = await fetch(`/api/anim_preview/delete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ njm_path: njmFilename }),
      });
      if (!r.ok) {
        console.warn("[texture_panel] anim_preview/delete failed:", r.status);
        return;
      }
      // Drop from local cache + clear current-name if it was just removed.
      if (importedAnimsState.currentName === njmFilename) {
        importedAnimsState.currentName = null;
        if (typeof window.psoLoadMotion === "function") window.psoLoadMotion("");
      }
      await fetchImportedAnims(true);
      renderMotionsBlock();
    } catch (e) {
      console.warn("[texture_panel] anim_preview/delete error:", e);
    }
  }

  function _importedAnimRowHtml(item, isCurrent) {
    const cls = ["pso-motion-row", "imported"];
    if (isCurrent) cls.push("active");
    const initials = (item.display_name || item.name).replace(/_/g, " ").slice(0, 3).toUpperCase();
    const fps = item.fps || 30;
    const dur = (item.frame_count / Math.max(1, fps)).toFixed(2);
    const src = item.source_glb ? ` · src: ${escapeHtml(item.source_glb)}` : "";
    const dropped = (item.dropped_bones | 0) > 0
      ? ` · ${item.dropped_bones} bones dropped`
      : "";
    return `<div class="${cls.join(' ')}" data-imported-name="${escapeHtml(item.name)}">
      <div class="pso-motion-thumb">${escapeHtml(initials)}</div>
      <div class="pso-motion-info">
        <div class="pso-motion-name" title="${escapeHtml(item.display_name || item.name)}">${escapeHtml(item.display_name || item.name)}</div>
        <div class="pso-motion-meta">${item.frame_count}f @ ${fps}fps · ${dur}s · ${item.retargeted_bones} bones${dropped}${src}</div>
      </div>
      <div class="pso-motion-actions">
        <button class="pso-motion-remove" data-act="remove-imported" title="remove from preview">remove</button>
      </div>
    </div>`;
  }

  function _renderImportedAnimsGroup(items) {
    if (!Array.isArray(items) || items.length === 0) return "";
    const cur = importedAnimsState.currentName;
    const rows = items.map((it) => _importedAnimRowHtml(it, cur === it.name)).join("");
    return `<div class="pso-motions-group pso-motions-imported-group">
      <div class="pso-motions-group-title">
        <span class="pso-import-badge">imported</span>
        Imported Animations (${items.length})
      </div>
      ${rows}
    </div>`;
  }

  async function ensureMotionsList() {
    const native = (typeof window.psoListMotions === "function")
      ? window.psoListMotions()
      : [];
    if (native.length > 0) {
      motionsFallback = { path: null, motions: null };
      return native;
    }
    // Use cached fallback if it matches the current model AND is non-empty
    // (an empty cache means we previously failed to find any — try again).
    if (motionsFallback.path === lastModelPath
        && motionsFallback.motions
        && motionsFallback.motions.length > 0) {
      return motionsFallback.motions;
    }
    // Fetch directly from /api/animations. We try BOTH the bml-only form
    // and the bml#inner form when applicable — the no-inner form covers
    // the common case of sibling NJM motions, but some BMLs need the
    // inner specified (cluster archives like NpcApcMot.bml).
    const refPath = lastModelPath || lastModelPathPublic;
    if (!refPath) return [];
    const hashIdx = refPath.indexOf("#");
    let url;
    if (hashIdx > 0) {
      const base = refPath.slice(0, hashIdx);
      let inner = refPath.slice(hashIdx + 1);
      if (inner.toLowerCase().endsWith(".xvm")) inner = inner.slice(0, -4);
      url = `/api/animations/${encodeURIComponent(base)}?inner=${encodeURIComponent(inner)}`;
    } else {
      url = `/api/animations/${encodeURIComponent(refPath)}`;
    }
    // Wave 7: lifecycle abort coordination.
    const motionsF = (window.psoAssetLifecycle && window.psoAssetLifecycle.fetchAsset) || fetch;
    try {
      const r = await motionsF(url);
      if (!r.ok) return [];
      const data = await r.json();
      const motions = (data && data.motions) || [];
      motionsFallback = { path: refPath, motions };
      return motions;
    } catch (_) {
      // AbortError or other failures → empty motions list. The viewer
      // already handles "no motions" gracefully.
      return [];
    }
  }

  // Group a motion by name keyword. Returns one of: movement,
  // combat, idle, death, special.
  function classifyMotion(name) {
    const n = (name || "").toLowerCase();
    if (/walk|run|move|swim|fly|frin|frout|frloop|wgwalk/.test(n)) return "movement";
    if (/atack|attack|tatk|fire|kiri|bite|tukomi|tobidasi|nkdown|nkup|flyshot/.test(n)) return "combat";
    if (/dam|hurt|down|kiri|nobi|land|lift|wngclose|wngopn|wing/.test(n)) return "reaction";
    if (/dead|dying|die|opgc|bossopgc/.test(n)) return "death";
    if (/wait|idle|stand|cstand|laugh|hoe|wake|nobi/.test(n)) return "idle";
    return "special";
  }

  const _MOTION_GROUP_ORDER = ["movement", "combat", "reaction", "idle", "death", "special"];
  const _MOTION_GROUP_LABELS = {
    movement: "Movement",
    combat: "Combat",
    reaction: "Damage / Reaction",
    idle: "Idle / Wait",
    death: "Death",
    special: "Other",
  };

  let loopAllState = { active: false, idx: 0, timer: null };

  function stopLoopAll() {
    loopAllState.active = false;
    if (loopAllState.timer) {
      clearTimeout(loopAllState.timer);
      loopAllState.timer = null;
    }
  }

  function renderMotionsBlock() {
    const body = panel.querySelector('[data-region="body"]');
    if (!body) return;
    body.innerHTML = `<div class="pso-tex-panel-empty">loading motions…</div>`;
    Promise.all([ensureMotionsList(), fetchImportedAnims(false)]).then(([motions, imported]) => {
      if (activeTab !== "motions") return;
      _renderMotionsBlockWith(body, motions || [], imported || []);
    }).catch((e) => {
      if (activeTab !== "motions") return;
      body.innerHTML = `<div class="pso-tex-panel-empty">motion list failed: ${escapeHtml(e.message || String(e))}</div>`;
    });
  }

  function _renderMotionsBlockWith(body, motions, imported) {
    const importedHtml = _renderImportedAnimsGroup(imported);
    if (motions.length === 0 && (!imported || imported.length === 0)) {
      body.innerHTML = `<div class="pso-tex-panel-empty">No motions detected for this model.</div>`;
      return;
    }
    const current = (typeof window.psoGetCurrentMotion === "function")
      ? window.psoGetCurrentMotion() : null;
    // Group by classification.
    const groups = new Map();
    for (const m of motions) {
      const cat = classifyMotion(m.name);
      if (!groups.has(cat)) groups.set(cat, []);
      groups.get(cat).push(m);
    }
    // Sort each group by name.
    for (const arr of groups.values()) arr.sort((a, b) => a.name.localeCompare(b.name));

    const groupHtml = _MOTION_GROUP_ORDER
      .filter((g) => groups.has(g))
      .map((g) => {
        const rows = groups.get(g).map((m) => {
          const cls = ["pso-motion-row"];
          if (current === m.name) cls.push("active");
          // Thumbnail = first 3 letters of motion name (same as keyboard
          // shortcut style). Could be a real first-frame render later.
          const initials = m.name.replace(/_/g, " ").slice(0, 3).toUpperCase();
          const dur = (m.frame_count / Math.max(1, m.fps || 30)).toFixed(2);
          return `<div class="${cls.join(' ')}" data-motion="${escapeHtml(m.name)}">
            <div class="pso-motion-thumb">${escapeHtml(initials)}</div>
            <div class="pso-motion-info">
              <div class="pso-motion-name" title="${escapeHtml(m.name)}">${escapeHtml(m.name)}</div>
              <div class="pso-motion-meta">${m.frame_count}f @ ${m.fps}fps · ${dur}s</div>
            </div>
          </div>`;
        }).join("");
        return `<div class="pso-motions-group">
          <div class="pso-motions-group-title">${_MOTION_GROUP_LABELS[g]} (${groups.get(g).length})</div>
          ${rows}
        </div>`;
      }).join("");

    const loopActiveAttr = loopAllState.active ? "active" : "";
    body.innerHTML = `<div class="pso-motions-block">
      <div class="pso-motions-toolbar">
        <button data-act="bind-pose" title="reset to bind pose">bind pose</button>
        <button data-act="loop-all" class="${loopActiveAttr}"
                title="auto-cycle through every motion">${loopAllState.active ? 'stop loop-all' : 'Loop all'}</button>
        <button data-act="create-blend" title="weighted blend of 2-3 motions">Create blend</button>
      </div>
      <div class="pso-motions-list">${importedHtml}${groupHtml}</div>
    </div>`;

    // Wire row clicks for VANILLA motions (string-name path).
    body.querySelectorAll(".pso-motion-row[data-motion]").forEach((r) => {
      r.addEventListener("click", () => {
        stopLoopAll();
        const name = r.dataset.motion;
        if (typeof window.psoLoadMotion === "function") {
          window.psoLoadMotion(name);
          // Re-highlight the active row without a full re-render.
          body.querySelectorAll(".pso-motion-row.active").forEach(el => el.classList.remove("active"));
          r.classList.add("active");
          importedAnimsState.currentName = null;
        }
      });
    });
    // Wire IMPORTED row clicks (data path; no entry in state.anim.motions).
    body.querySelectorAll(".pso-motion-row[data-imported-name]").forEach((r) => {
      r.addEventListener("click", (ev) => {
        // Don't trigger play when clicking the inline "remove" button.
        const t = ev.target;
        if (t && (t.closest && t.closest(".pso-motion-remove"))) return;
        const name = r.dataset.importedName;
        playImportedAnim(name).then(() => {
          body.querySelectorAll(".pso-motion-row.active").forEach(el => el.classList.remove("active"));
          r.classList.add("active");
        });
      });
    });
    body.querySelectorAll(".pso-motion-row[data-imported-name] .pso-motion-remove").forEach((b) => {
      b.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const row = b.closest(".pso-motion-row[data-imported-name]");
        if (!row) return;
        const name = row.dataset.importedName;
        if (!name) return;
        if (!confirm(`Remove imported animation "${name}" from preview?\n\nThis deletes the staged .njm + sidecar (the source .glb is untouched). It does NOT modify any game data.`)) return;
        removeImportedAnim(name);
      });
    });
    // Toolbar.
    body.querySelector("[data-act='bind-pose']").addEventListener("click", () => {
      stopLoopAll();
      if (typeof window.psoLoadMotion === "function") window.psoLoadMotion("");
      body.querySelectorAll(".pso-motion-row.active").forEach(el => el.classList.remove("active"));
      importedAnimsState.currentName = null;
    });
    body.querySelector("[data-act='loop-all']").addEventListener("click", () => {
      if (loopAllState.active) {
        stopLoopAll();
      } else {
        startLoopAll(motions);
      }
      renderMotionsBlock();
    });
    // 2026-04-25 v2: Create blend modal. Lets user pick 2-3 source
    // motions + assign per-motion weights, posts to /api/anim/blend,
    // and refreshes the imported-anims list so the blended NJM shows
    // up under "Imported animations" with a single click play.
    const blendBtn = body.querySelector("[data-act='create-blend']");
    if (blendBtn) {
      blendBtn.addEventListener("click", () => {
        openBlendModal(motions);
      });
    }
  }

  // ----------------------------------------------------------------------
  // Blend-spaces modal (Task B / 2026-04-25)
  // ----------------------------------------------------------------------
  // Renders an overlay with motion-name pickers + weight sliders. Posts
  // to /api/anim/blend, then triggers a refresh of the imported-anims
  // panel so the new blend.njm appears in-place.
  let _blendModalEl = null;

  function openBlendModal(motions) {
    if (_blendModalEl) {
      try { _blendModalEl.remove(); } catch (_) { }
      _blendModalEl = null;
    }
    if (!motions || motions.length < 2) {
      alert("Need at least 2 motions in the current model to create a blend.");
      return;
    }
    const refPath = lastModelPath || lastModelPathPublic;
    if (!refPath) {
      alert("No active model to blend against.");
      return;
    }
    const root = document.createElement("div");
    root.className = "pso-blend-modal-overlay";
    root.style.cssText = "position:fixed;top:0;left:0;width:100%;height:100%;"
      + "background:rgba(0,0,0,0.65);z-index:9000;display:flex;"
      + "align-items:center;justify-content:center";
    const motionOpts = motions.map((m) =>
      `<option value="${escapeHtml(m.name)}">${escapeHtml(m.name)} (${m.frame_count}f)</option>`
    ).join("");
    root.innerHTML = `
      <div style="background:#1c2230;color:#e6e9ef;padding:20px;border-radius:6px;
                  border:1px solid #303849;min-width:480px;max-width:95vw">
        <div style="font-size:14px;font-weight:600;margin-bottom:12px">Create motion blend</div>
        <div style="font-size:11px;color:#99a4b3;margin-bottom:10px">
          Blend up to 3 source motions with per-motion weights. The
          result is staged as an imported animation alongside this model.
        </div>
        <div data-region="rows" style="display:flex;flex-direction:column;gap:8px"></div>
        <div style="margin-top:12px;display:flex;gap:6px;align-items:center">
          <button type="button" data-act="add-row" style="font-size:11px">+ add source</button>
          <span style="flex:1"></span>
          <label style="font-size:11px">Curve:</label>
          <select data-region="curve" style="font-size:11px">
            <option value="linear">linear</option>
            <option value="smooth">smooth (transition)</option>
            <option value="ease_in">ease-in</option>
            <option value="ease_out">ease-out</option>
          </select>
        </div>
        <div style="margin-top:8px;display:flex;gap:6px;align-items:center">
          <label style="font-size:11px">Output name:</label>
          <input type="text" data-region="output-name" value="blend.njm"
                 style="flex:1;padding:4px 6px;font-size:11px" />
        </div>
        <div style="margin-top:8px;display:flex;gap:6px;align-items:center">
          <label style="font-size:11px">Frame count:</label>
          <input type="number" data-region="frame-count" min="1" value=""
                 placeholder="(auto = max source)"
                 style="width:120px;padding:4px 6px;font-size:11px" />
        </div>
        <div style="margin-top:14px;display:flex;gap:8px;justify-content:flex-end">
          <button type="button" data-act="cancel" style="font-size:11px">Cancel</button>
          <button type="button" data-act="build" style="font-size:11px;
                  background:#3a78c2;color:#fff;border:none;padding:6px 12px;
                  border-radius:3px">Blend &amp; stage</button>
        </div>
        <div data-region="status" style="margin-top:8px;font-size:11px;color:#99a4b3"></div>
      </div>
    `;
    document.body.appendChild(root);
    _blendModalEl = root;

    const rowsEl = root.querySelector("[data-region='rows']");
    function addRow(defaultIdx) {
      const row = document.createElement("div");
      row.className = "pso-blend-row";
      row.style.cssText = "display:flex;gap:6px;align-items:center";
      row.innerHTML = `
        <select data-region="motion" style="flex:1;font-size:11px">${motionOpts}</select>
        <input type="range" data-region="weight" min="0" max="100" value="50"
               style="width:120px" />
        <span data-region="weight-label" style="width:36px;text-align:right;font-size:11px">0.50</span>
        <button type="button" data-act="remove-row" style="font-size:11px">remove</button>
      `;
      const sel = row.querySelector("[data-region='motion']");
      if (typeof defaultIdx === "number" && defaultIdx >= 0 && defaultIdx < motions.length) {
        sel.value = motions[defaultIdx].name;
      }
      const slider = row.querySelector("[data-region='weight']");
      const label = row.querySelector("[data-region='weight-label']");
      slider.addEventListener("input", () => {
        label.textContent = (Number(slider.value) / 100).toFixed(2);
      });
      row.querySelector("[data-act='remove-row']").addEventListener("click", () => {
        if (rowsEl.children.length <= 2) {
          // Keep at least 2 rows; user must clear the modal to abandon.
          return;
        }
        row.remove();
      });
      rowsEl.appendChild(row);
    }
    addRow(0);
    addRow(Math.min(1, motions.length - 1));

    root.querySelector("[data-act='add-row']").addEventListener("click", () => {
      if (rowsEl.children.length >= 3) return;  // 3 max
      addRow();
    });
    root.querySelector("[data-act='cancel']").addEventListener("click", () => {
      if (_blendModalEl) {
        _blendModalEl.remove();
        _blendModalEl = null;
      }
    });
    root.querySelector("[data-act='build']").addEventListener("click", async () => {
      const status = root.querySelector("[data-region='status']");
      const rows = Array.from(rowsEl.querySelectorAll(".pso-blend-row"));
      const sourceNames = [];
      const weights = [];
      for (const r of rows) {
        const nm = (r.querySelector("[data-region='motion']")).value;
        const wRaw = Number((r.querySelector("[data-region='weight']")).value);
        sourceNames.push(nm);
        weights.push(wRaw / 100);
      }
      const outputName = (root.querySelector("[data-region='output-name']")).value.trim() || "blend.njm";
      const fcInput = (root.querySelector("[data-region='frame-count']")).value.trim();
      const frameCount = fcInput === "" ? null : Math.max(1, Math.floor(Number(fcInput)));
      const curve = (root.querySelector("[data-region='curve']")).value;
      status.textContent = "blending...";
      try {
        const res = await fetch("/api/anim/blend", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model_path: refPath,
            source_motion_names: sourceNames,
            weights,
            output_name: outputName,
            frame_count: frameCount,
            transition_curve: curve,
          }),
        });
        if (!res.ok) {
          let detail = `http ${res.status}`;
          try { const eb = await res.json(); if (eb && eb.detail) detail = eb.detail; } catch (_) { }
          status.textContent = `blend failed: ${detail}`;
          return;
        }
        const data = await res.json();
        status.textContent = `staged ${data.njm_name} (${data.frame_count} frames, ${data.bone_count} bones)`;
        // Refresh imported-anims so the new blend appears.
        await fetchImportedAnims(true);
        renderMotionsBlock();
        setTimeout(() => {
          if (_blendModalEl) {
            _blendModalEl.remove();
            _blendModalEl = null;
          }
        }, 1200);
      } catch (e) {
        status.textContent = `blend error: ${e.message || String(e)}`;
      }
    });
  }

  function startLoopAll(motions) {
    if (motions.length === 0) return;
    loopAllState.active = true;
    loopAllState.idx = 0;
    const playOne = async (i) => {
      if (!loopAllState.active) return;
      const m = motions[i];
      if (!m) return;
      if (typeof window.psoLoadMotion === "function") {
        await window.psoLoadMotion(m.name);
      }
      // Update active row highlight if motions tab is currently rendered.
      if (panel) {
        const body = panel.querySelector('[data-region="body"]');
        if (body) {
          body.querySelectorAll(".pso-motion-row.active").forEach(el => el.classList.remove("active"));
          const r = body.querySelector(`.pso-motion-row[data-motion="${CSS.escape(m.name)}"]`);
          if (r) r.classList.add("active");
        }
      }
      // Schedule the next motion after this one's duration (in seconds).
      // Cap the per-motion runtime to a human-watchable window:
      // min 1.5s (so very-short motions don't blink past), max 6s.
      const dur = m.frame_count / Math.max(1, m.fps || 30);
      const ms = Math.max(1500, Math.min(6000, dur * 1000));
      loopAllState.timer = setTimeout(() => {
        const next = (i + 1) % motions.length;
        loopAllState.idx = next;
        playOne(next);
      }, ms);
    };
    playOne(0);
  }

  let _refreshTimer = null;
  function scheduleRefresh() {
    if (_refreshTimer) clearTimeout(_refreshTimer);
    _refreshTimer = setTimeout(() => {
      _refreshTimer = null;
      // Overlap fix: if the active perspective is not a texture-panel
      // host (e.g. map-editor / floor-editor relocated .model-stage into
      // their own viewport), tear the panel + strip down instead of
      // re-mounting them on top of that perspective's content.
      if (!perspectiveWantsTexturePanel()) {
        detachPanel();
        return;
      }
      // Re-attach panel to whichever allowed .model-stage is currently
      // mounted (perspectives.js relocates it between the modal and the
      // 3d-view vp-stage).
      ensurePanelDom();
      ensureVariantStripDom();
      renderActiveTab();
      renderVariantStrip();
    }, REFRESH_DEBOUNCE_MS);
  }

  // Derive the texture-archive path from a model path + matched-texture
  // list. Mirrors model_viewer's `deriveTextureArchivePath` but works
  // for both /api/model_mesh and /api/model_skinned (the latter's URL
  // shape doesn't match model_viewer's regex, leaving boundTextureArchive
  // as null on skinned-path loads).
  //
  // Strategy:
  //   * If a matched_texture path of shape "<base>#<inner>.nj.xvm" is
  //     present, use it verbatim (this is what `R2` synth produces).
  //   * Otherwise infer from modelPath: "<bml>#<inner>.nj"
  //                                     -> "<bml>#<inner>.nj.xvm"
  function deriveArchiveFromContext(modelPath, matched) {
    if (Array.isArray(matched)) {
      for (const m of matched) {
        const p = (m && m.path) || "";
        const lo = p.toLowerCase();
        if (lo.endsWith(".nj.xvm") || lo.endsWith(".xvm") || lo.endsWith(".prs")) {
          return p;
        }
      }
    }
    if (typeof modelPath === "string") {
      const hashIdx = modelPath.indexOf("#");
      if (hashIdx > 0) {
        const lo = modelPath.toLowerCase();
        if (lo.endsWith(".nj")) return modelPath + ".xvm";
        if (lo.endsWith(".njm")) return modelPath.slice(0, -4) + ".nj.xvm";
      } else if (modelPath.toLowerCase().endsWith(".bml")) {
        // Top-level BML — caller didn't infer the inner; take the first
        // matched_texture's archive as the best guess.
        if (Array.isArray(matched) && matched[0] && matched[0].path) {
          return matched[0].path;
        }
      }
    }
    return null;
  }

  // Fallback texture-list builder that uses /api/tile_png + the binding
  // table directly when window.psoListMeshTextures returns []. The
  // model_viewer ALWAYS populates state.boundBinding when binding rows
  // are returned by the server; we just synthesise the archive ourselves
  // when boundTextureArchive came up null (model_skinned URL pattern).
  function listTexturesFallback() {
    const native = (typeof window.psoListMeshTextures === "function")
      ? window.psoListMeshTextures()
      : [];
    if (native.length > 0) return native;
    const binding = (typeof window.psoGetTextureBinding === "function")
      ? window.psoGetTextureBinding()
      : [];
    if (binding.length === 0) return [];
    const archive = (typeof window.psoGetCurrentTextureArchive === "function")
      ? window.psoGetCurrentTextureArchive() : null;
    const arch = archive || deriveArchiveFromContext(lastModelPath, lastMatched);
    if (!arch) return [];
    const byTile = new Map();
    for (const b of binding) {
      if (!b || b.missing) continue;
      const ti = b.tile_index | 0;
      if (!byTile.has(ti)) byTile.set(ti, []);
      byTile.get(ti).push(b.material_id | 0);
    }
    const out = [];
    for (const [tile, mids] of byTile) {
      out.push({
        tile_index: tile,
        material_ids: mids.slice().sort((a, b) => a - b),
        width: null,
        height: null,
        archive: arch,
        thumbnail_url: `/api/tile_png/${encodeURIComponent(arch)}/${tile}`,
      });
    }
    out.sort((a, b) => a.tile_index - b.tile_index);
    return out;
  }

  // Mirror model_viewer's `openByPath` heuristic for resolving a
  // top-level BML path to its inner `.nj` mesh path. When the manifest
  // gives us a `bm_ene_*.bml` and a matched texture of shape
  // `<bml>#<inner>.nj.xvm`, the corresponding model is `<bml>#<inner>.nj`.
  function resolveMeshPath(modelPath, matched) {
    if (typeof modelPath !== "string") return modelPath;
    if (modelPath.indexOf("#") >= 0) return modelPath;
    const lower = modelPath.toLowerCase();
    if (!lower.endsWith(".bml")) return modelPath;
    if (Array.isArray(matched)) {
      for (const m of matched) {
        const p = (m && m.path) || "";
        if (p.startsWith(modelPath + "#") && p.toLowerCase().endsWith(".nj.xvm")) {
          return p.slice(0, -4);  // drop ".xvm" tail
        }
      }
    }
    return modelPath;
  }

  // Hook into the asset_router's openModel pipeline by wrapping
  // window.psoOpenModelByPath. We can't subscribe to bus events alone
  // because the model_viewer dispatches no signal on "mesh ready" —
  // wrapping the entry-point lets us refresh once the load resolves.
  function instrumentOpener() {
    const orig = window.psoOpenModelByPath;
    if (!orig || orig.__psoTexPanelWrapped) return;
    const wrapped = async function (modelPath, entry, matched) {
      // Save BOTH the public model path (for re-open / reset) and the
      // resolved inner-mesh path (which the subdivide endpoint expects).
      lastModelPath = resolveMeshPath(modelPath, matched);
      lastEntry = entry || {};
      lastMatched = Array.isArray(matched) ? matched.slice() : [];
      lastModelPathPublic = modelPath;
      // Reset subdivide bookkeeping for the new model so the orig
      // stats line refreshes against fresh geometry.
      subdivideState.origStats = null;
      subdivideState.lastResult = null;
      subdivideState.cachedPath = null;
      // Reset variant + loop-all bookkeeping so a new model loads fresh.
      stopLoopAll();
      // Clear preview-imports cache so we re-fetch the list against
      // the new target model.
      importedAnimsState = { path: null, items: null, error: null, currentName: null };
      const r = await orig.apply(this, arguments);
      // After the mesh resolves we want to refresh the panel so the
      // bound textures show up. Two ticks is enough for the THREE
      // Texture image-load callbacks to populate width/height.
      scheduleRefresh();
      setTimeout(scheduleRefresh, 600);
      // Fetch variants asynchronously so the strip appears as soon as
      // the server resolves the family.
      fetchVariants().catch((e) => console.warn("variant fetch failed:", e));
      return r;
    };
    wrapped.__psoTexPanelWrapped = true;
    window.psoOpenModelByPath = wrapped;
  }

  function instrumentReload() {
    const orig = window.psoReloadModel;
    if (!orig || orig.__psoTexPanelWrapped) return;
    const wrapped = async function (modelPath, entry, matched) {
      lastModelPath = modelPath;
      if (entry) lastEntry = entry;
      if (Array.isArray(matched) && matched.length) lastMatched = matched.slice();
      const r = await orig.apply(this, arguments);
      scheduleRefresh();
      setTimeout(scheduleRefresh, 600);
      return r;
    };
    wrapped.__psoTexPanelWrapped = true;
    window.psoReloadModel = wrapped;
  }

  // Wait until model_viewer.js has registered psoOpenModelByPath (it
  // does so unconditionally at module-init), then wrap. Poll for up
  // to 5s to handle late-loading.
  function waitForOpener(deadline) {
    if (window.psoOpenModelByPath) {
      instrumentOpener();
      instrumentReload();
      return;
    }
    if (Date.now() > deadline) {
      console.warn("[texture_panel] psoOpenModelByPath never appeared; panel disabled");
      return;
    }
    setTimeout(() => waitForOpener(deadline), 100);
  }

  function init() {
    waitForOpener(Date.now() + 5000);
    // Close the tab-overflow menu on any click outside it (or outside the
    // "more ▾" toggle). Bound once.
    document.addEventListener("click", (ev) => {
      if (!panel) return;
      const menu = panel.querySelector(".pso-tex-tab-overflow-menu");
      if (!menu || menu.hidden) return;
      const wrap = panel.querySelector(".pso-tex-tab-overflow");
      if (wrap && wrap.contains(ev.target)) return; // handled by onPanelClick
      closeTabOverflowMenu();
    });
    // If the user is already on the 3D-view perspective when this
    // script loads, mount the panel right away. Otherwise it'll mount
    // on the next openByPath wrap.
    scheduleRefresh();
    // Re-attach when the perspective switches (model-stage gets moved
    // between the modal and vp-stage).
    if (window.bus && typeof window.bus.on === "function") {
      window.bus.on("perspective.switched", () => scheduleRefresh());
      // Live-reload (v5 polish): refresh the panel when a painted
      // texture or sculpted mesh changes on disk. Filters on path
      // prefix so events for unrelated dirs (njm/bml/etc.) skip.
      window.bus.on("cache.changed", (payload) => {
        if (!payload || !payload.path) return;
        if (payload.path.indexOf("cache/painted_textures/") === 0 ||
            payload.path.indexOf("cache/sculpted_meshes/") === 0) {
          scheduleRefresh();
        }
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // Surface a manual refresh hook for tests + devtools.
  window.psoTexturePanelRefresh = scheduleRefresh;
  // Public re-flow hook for panels that inject their own tab DOM directly
  // (paint_panel.js) so they can fold their tab into the overflow menu.
  window.psoTexturePanelReflowTabs = function () { try { reflowTabs(); } catch (_e) {} };
})();
