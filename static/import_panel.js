// =====================================================================
// PSOBB Import Panel — drag .obj/.gltf/.glb -> deployable PSOBB .nj
// 2026-04-25
//
// Adds a top-level drop zone overlay that listens for drag-drop on the
// editor at large. When the user drops a supported external file we:
//   1. POST it to /api/import/parse to get an ImportedModel JSON
//   2. Open a preview modal with:
//      - 3D preview via psoApplyMeshPayload (re-uses model_viewer)
//      - Mesh / vertex / triangle / bone count summary
//      - Skeleton template picker (pulled from /api/import/templates)
//      - Axis-flip toggle, scale slider
//      - "Convert to NJ" button -> /api/import/build_nj
//      - Optional "Replace target..." panel that lists eligible BMLs
//        and substitutes the imported NJ for an inner mesh
//      - Live Test button (re-uses the shared PSOLiveTest hook)
//
// Drop zone semantics: visible whenever a drag is over the editor with
// at least one item that has a recognized extension OR no extension
// (we accept on drop and let the server detect via magic bytes).
//
// Header button "Import 3D" opens a file picker as a fallback for
// users on tablets / VMs without a working drag-drop pipeline.
//
// Wire formats in this module mirror /api/import/* server endpoints —
// see formats/import_external.py and the docstrings in server.py for
// the JSON shapes.
// =====================================================================

(function () {
  "use strict";

  if (window.__psoImportPanelLoaded) return;
  window.__psoImportPanelLoaded = true;

  const SUPPORTED_EXTS = ["obj", "gltf", "glb", "fbx"];
  const REJECTED_EXTS = ["dae", "ply", "stl", "3ds"];

  const STYLE_ID = "psoImportPanelStyle";

  // -------------------------------------------------------------------
  // Style injection — kept self-contained so the panel doesn't depend
  // on style.css order.
  // -------------------------------------------------------------------
  function ensureStyleInjected() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .imp-drop-overlay {
        position: fixed;
        inset: 0;
        z-index: 4000;
        background: rgba(14, 17, 22, 0.85);
        display: none;
        align-items: center;
        justify-content: center;
        pointer-events: auto;
      }
      .imp-drop-overlay.active { display: flex; }
      .imp-drop-inner {
        border: 3px dashed #ffaa00;
        background: #2a1a05;
        color: #fff;
        padding: 36px 60px;
        border-radius: 14px;
        text-align: center;
        max-width: 520px;
        box-shadow: 0 30px 80px rgba(0, 0, 0, 0.6);
      }
      .imp-drop-title { font-size: 22px; font-weight: 600; margin-bottom: 10px; }
      .imp-drop-sub { font-size: 13px; color: #ddc097; margin-bottom: 8px; }
      .imp-drop-exts {
        display: inline-flex; gap: 6px; margin-top: 10px;
      }
      .imp-drop-ext {
        padding: 2px 8px; border: 1px solid #ffaa00; border-radius: 3px;
        font-size: 11px; color: #ffaa00;
      }
      .imp-modal-backdrop {
        position: fixed; inset: 0; z-index: 3500;
        background: rgba(14, 17, 22, 0.6);
        display: flex; align-items: center; justify-content: center;
      }
      .imp-modal {
        background: #181d24;
        border: 1px solid #2a313a;
        border-radius: 6px;
        width: 720px; max-width: 96vw;
        max-height: 92vh; overflow: hidden;
        display: flex; flex-direction: column;
        color: #c7d8ec; font-size: 12px;
        box-shadow: 0 30px 80px rgba(0, 0, 0, 0.7);
      }
      .imp-modal-head {
        padding: 10px 14px;
        border-bottom: 1px solid #2a313a;
        display: flex; align-items: center; gap: 8px;
        background: rgba(0, 0, 0, 0.25);
      }
      .imp-modal-title { font-weight: 600; flex: 1; color: #e6e8eb; }
      .imp-modal-x {
        cursor: pointer; padding: 0 8px;
        background: transparent; border: 1px solid #2a313a;
        color: #99a4b3; border-radius: 2px;
      }
      .imp-modal-x:hover { border-color: #ff6680; color: #ff6680; }
      .imp-modal-body {
        padding: 12px 14px; overflow: auto;
        display: grid; grid-template-columns: 1fr; gap: 10px;
      }
      .imp-section {
        border: 1px solid #2a313a; border-radius: 4px;
        padding: 8px 10px; background: rgba(0, 0, 0, 0.2);
      }
      .imp-section-title {
        font-size: 11px; color: #99a4b3; text-transform: uppercase;
        margin-bottom: 6px; letter-spacing: 0.4px;
      }
      .imp-stat-grid {
        display: grid; grid-template-columns: repeat(3, 1fr);
        gap: 4px 14px; font-variant-numeric: tabular-nums;
      }
      .imp-stat-label { color: #99a4b3; font-size: 10px; }
      .imp-stat-val { color: #c7d8ec; font-size: 12px; }
      .imp-warn {
        padding: 4px 8px; margin-top: 4px;
        border-left: 3px solid #ffaa00;
        background: rgba(255, 170, 0, 0.05);
        color: #f3d899; font-size: 11px;
      }
      .imp-form-row {
        display: grid; grid-template-columns: 120px 1fr auto;
        gap: 8px; align-items: center;
        padding: 4px 0;
      }
      .imp-form-row label { color: #99a4b3; font-size: 11px; }
      .imp-form-row select, .imp-form-row input[type=text] {
        background: #0e1116; border: 1px solid #2a313a;
        color: #c7d8ec; padding: 3px 6px; font: inherit;
        border-radius: 2px;
      }
      .imp-form-row input[type=range] { width: 100%; }
      .imp-form-row .num {
        font-variant-numeric: tabular-nums; min-width: 50px;
        text-align: right;
      }
      .imp-form-row input[type=checkbox] { transform: translateY(1px); }
      .imp-actions {
        display: flex; gap: 6px; justify-content: flex-end; flex-wrap: wrap;
        padding: 10px 14px; border-top: 1px solid #2a313a;
        background: rgba(0, 0, 0, 0.25);
      }
      .imp-btn {
        background: transparent; border: 1px solid #2a313a;
        color: #99a4b3; padding: 4px 10px; font: inherit;
        border-radius: 2px; cursor: pointer;
      }
      .imp-btn:hover { border-color: #00ffff; color: #00ffff; }
      .imp-btn:disabled { opacity: 0.4; cursor: not-allowed; }
      .imp-btn.primary { border-color: #ffaa00; color: #ffaa00; }
      .imp-btn.primary:hover { background: rgba(255, 170, 0, 0.15); }
      .imp-status {
        flex: 1; align-self: center; font-size: 11px;
        font-variant-numeric: tabular-nums;
      }
      .imp-status.ok { color: #6ee785; }
      .imp-status.err { color: #ff6680; }
      .imp-status.busy { color: #ffaa00; }
      .imp-replace-list {
        max-height: 180px; overflow: auto;
        background: rgba(0, 0, 0, 0.25);
        border: 1px solid #2a313a; border-radius: 2px;
        padding: 4px;
      }
      .imp-replace-row {
        padding: 3px 6px; cursor: pointer; font-size: 11px;
        font-family: monospace; color: #c7d8ec;
        border-radius: 2px;
      }
      .imp-replace-row:hover { background: rgba(74, 144, 226, 0.15); }
      .imp-replace-row.selected {
        background: rgba(74, 144, 226, 0.25);
        border-left: 2px solid #4a90e2;
      }
      .imp-replace-search {
        width: 100%; padding: 3px 6px; margin-bottom: 4px;
        background: #0e1116; border: 1px solid #2a313a;
        color: #c7d8ec; font: inherit; border-radius: 2px;
      }
      header button.imp-header-btn {
        background: transparent; border: 1px solid #2a313a;
        color: #c7d8ec; padding: 3px 10px; cursor: pointer;
        border-radius: 2px;
      }
      header button.imp-header-btn:hover {
        border-color: #ffaa00; color: #ffaa00;
      }
    `;
    document.head.appendChild(style);
  }

  // -------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------
  const state = {
    parsedModel: null,    // ImportedModel JSON from /api/import/parse
    parsedFilename: null,
    parsedFile: null,     // raw File handle, kept for the animation
                          // re-upload path (we don't want to ask the user
                          // to drop the same file twice).
    templates: [],        // [{name, bone_count, description, source}, ...]
    selectedTemplate: "",
    axisFlip: true,
    scale: 1.0,
    suggestedName: "",    // .nj output name
    builtNjPath: null,    // last build output (for replace)
    builtNjMd5: null,
    selectedTargetBml: null,
    selectedTargetInner: null,
    bmlIndex: [],         // {bml, inners: [{name, ext}]} from manifest
    // 2026-04-25 v2: animation import state.
    builtNjmName: null,
    builtNjmFrames: 0,
    builtNjmBoneCount: 0,
  };

  // -------------------------------------------------------------------
  // Drop overlay — covers the whole window during drag.
  // -------------------------------------------------------------------
  let dropOverlay = null;
  let dragDepth = 0;

  function createDropOverlay() {
    if (dropOverlay) return dropOverlay;
    dropOverlay = document.createElement("div");
    dropOverlay.className = "imp-drop-overlay";
    dropOverlay.innerHTML = `
      <div class="imp-drop-inner">
        <div class="imp-drop-title">Drop a 3D model to import</div>
        <div class="imp-drop-sub">.obj / .gltf / .glb / .fbx (Blender, Maya, 3ds Max, Unity, Mixamo)</div>
        <div class="imp-drop-exts">
          <span class="imp-drop-ext">obj</span>
          <span class="imp-drop-ext">gltf</span>
          <span class="imp-drop-ext">glb</span>
          <span class="imp-drop-ext">fbx</span>
        </div>
        <div class="imp-drop-sub" style="margin-top:14px;color:#99a4b3">
          parsed mesh + skeleton -> PSOBB .nj writer -> deployable
        </div>
      </div>
    `;
    document.body.appendChild(dropOverlay);
    return dropOverlay;
  }

  function hasSupportedDragItem(dt) {
    if (!dt) return false;
    // dt.types is a DOMStringList; "Files" appears for any file drag.
    if (Array.from(dt.types || []).includes("Files")) {
      return true;
    }
    return false;
  }

  function attachDropHandlers() {
    window.addEventListener("dragenter", (e) => {
      if (!hasSupportedDragItem(e.dataTransfer)) return;
      // Don't fire for drags originating inside a file input or paint canvas.
      const tag = (e.target && e.target.tagName) || "";
      if (tag === "INPUT" || tag === "CANVAS") return;
      dragDepth += 1;
      createDropOverlay().classList.add("active");
    });
    window.addEventListener("dragover", (e) => {
      if (dropOverlay && dropOverlay.classList.contains("active")) {
        e.preventDefault();
        if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
      }
    });
    window.addEventListener("dragleave", (e) => {
      if (!dropOverlay) return;
      dragDepth -= 1;
      if (dragDepth <= 0) {
        dragDepth = 0;
        dropOverlay.classList.remove("active");
      }
    });
    window.addEventListener("drop", (e) => {
      if (!dropOverlay || !dropOverlay.classList.contains("active")) return;
      e.preventDefault();
      dragDepth = 0;
      dropOverlay.classList.remove("active");
      const files = (e.dataTransfer && e.dataTransfer.files) || [];
      if (files.length === 0) return;
      // Take the first file. (Multi-file import is v2.)
      const f = files[0];
      const ext = (f.name.split(".").pop() || "").toLowerCase();
      if (REJECTED_EXTS.includes(ext)) {
        showStatus(`${ext.toUpperCase()} is not supported — re-export as .glb / .fbx / .obj`, "err");
        return;
      }
      void parseAndOpen(f);
    });
  }

  // -------------------------------------------------------------------
  // Header button — file-picker fallback.
  // -------------------------------------------------------------------
  function attachHeaderButton() {
    const header = document.querySelector("header");
    if (!header) return;
    if (header.querySelector(".imp-header-btn")) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "imp-header-btn";
    btn.title = "import an external 3D model (obj / gltf / glb / fbx)";
    btn.textContent = "Import 3D";
    // Insert before the "deploy to game" button when present.
    const deployBtn = header.querySelector("#btnDeployToGame");
    if (deployBtn) header.insertBefore(btn, deployBtn);
    else header.appendChild(btn);
    btn.addEventListener("click", () => {
      const inp = document.createElement("input");
      inp.type = "file";
      inp.accept = ".obj,.gltf,.glb,.fbx";
      inp.onchange = () => {
        const f = inp.files && inp.files[0];
        if (f) void parseAndOpen(f);
      };
      inp.click();
    });
  }

  // -------------------------------------------------------------------
  // Server calls
  // -------------------------------------------------------------------
  async function fetchTemplates() {
    if (state.templates.length > 0) return state.templates;
    try {
      const r = await fetch("/api/import/templates");
      if (!r.ok) throw new Error(r.statusText);
      const j = await r.json();
      state.templates = j.templates || [];
    } catch (e) {
      console.warn("import: failed to fetch templates", e);
      state.templates = [];
    }
    return state.templates;
  }

  async function postImportParse(file) {
    const fd = new FormData();
    fd.append("file", file, file.name);
    const r = await fetch("/api/import/parse", { method: "POST", body: fd });
    if (!r.ok) {
      let msg = r.statusText;
      try {
        const j = await r.json();
        if (j && j.detail) msg = j.detail;
      } catch (_) { }
      throw new Error(msg);
    }
    return await r.json();
  }

  async function postImportBuildNj(modelJson, name, opts) {
    const body = {
      name,
      model_json: modelJson,
      target_class: opts.target_class || null,
      axis_flip_z: !!opts.axis_flip_z,
      scale: Number(opts.scale) || 1.0,
    };
    const r = await fetch("/api/import/build_nj", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      let msg = r.statusText;
      try {
        const j = await r.json();
        if (j && j.detail) msg = j.detail;
      } catch (_) { }
      throw new Error(msg);
    }
    return await r.json();
  }

  async function postImportReplace(args) {
    const r = await fetch("/api/import/replace", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(args),
    });
    if (!r.ok) {
      let msg = r.statusText;
      try {
        const j = await r.json();
        if (j && j.detail) msg = j.detail;
      } catch (_) { }
      throw new Error(msg);
    }
    return await r.json();
  }

  // 2026-04-25 v2: animation endpoints. Re-upload the original file
  // (we kept the File handle in state.parsedFile) and let the server
  // retarget onto the chosen target's skeleton.
  async function postImportAnimation(file, opts) {
    const fd = new FormData();
    fd.append("file", file, file.name);
    fd.append("target_model_path", opts.target_model_path);
    fd.append("motion_name", opts.motion_name);
    if (opts.target_inner) fd.append("target_inner", opts.target_inner);
    fd.append("include_translation", opts.include_translation ? "true" : "false");
    fd.append("flip_z", opts.flip_z ? "true" : "false");
    fd.append("target_fps", String(opts.target_fps || 30));
    // 2026-04-25 v2: IK retarget toggle. Default true; set explicitly
    // so the server can rely on the form field's presence for parsing.
    fd.append("enable_ik", opts.enable_ik === false ? "false" : "true");
    // 2026-04-25 v3: rotation IK + mirror toggles.
    fd.append(
      "enable_ik_rotation",
      opts.enable_ik_rotation === false ? "false" : "true",
    );
    fd.append("mirror", opts.mirror ? "true" : "false");
    if (opts.bone_map_name) fd.append("bone_map_name", opts.bone_map_name);
    const r = await fetch("/api/import/animation", { method: "POST", body: fd });
    if (!r.ok) {
      let msg = r.statusText;
      try {
        const j = await r.json();
        if (j && j.detail) msg = j.detail;
      } catch (_) { }
      throw new Error(msg);
    }
    return await r.json();
  }

  // 2026-04-25 v4: blend-shape side-file exporter. Re-uploads the
  // original parsed file (we kept the File handle in state.parsedFile)
  // so the server can re-run the parse + dump shapes to JSON.
  async function postBlendShapesExport(file, modelPath) {
    const fd = new FormData();
    fd.append("file", file, file.name);
    let url = "/api/import/blend_shapes/export";
    if (modelPath) url += "?model_path=" + encodeURIComponent(modelPath);
    const r = await fetch(url, { method: "POST", body: fd });
    if (!r.ok) {
      let msg = r.statusText;
      try {
        const j = await r.json();
        if (j && j.detail) msg = j.detail;
      } catch (_) { }
      throw new Error(msg);
    }
    return await r.json();
  }

  async function postImportAnimationReplace(args) {
    const r = await fetch("/api/import/animation/replace", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(args),
    });
    if (!r.ok) {
      let msg = r.statusText;
      try {
        const j = await r.json();
        if (j && j.detail) msg = j.detail;
      } catch (_) { }
      throw new Error(msg);
    }
    return await r.json();
  }

  // -------------------------------------------------------------------
  // BML index — pull from window.psoManifest if available.
  // -------------------------------------------------------------------
  async function ensureBmlIndex() {
    if (state.bmlIndex.length > 0) return state.bmlIndex;
    // Phase 1: ask the manifest module if it surfaces a list of BMLs.
    try {
      const r = await fetch("/api/manifest");
      if (r.ok) {
        const m = await r.json();
        const bmls = m.entries
          ? m.entries.filter((e) => e.path && e.path.endsWith(".bml")).map((e) => e.path)
          : (m.files || []).filter((p) => p.endsWith(".bml"));
        // For each BML we'd need its inner list; we lazy-load on click.
        state.bmlIndex = bmls.map((b) => ({ bml: b, inners: null }));
      }
    } catch (e) {
      console.warn("import: manifest fetch failed", e);
    }
    return state.bmlIndex;
  }

  async function loadBmlInners(bmlPath) {
    const r = await fetch(`/api/bml/${encodeURIComponent(bmlPath)}`);
    if (!r.ok) return [];
    const j = await r.json();
    return (j.entries || []).map((e) => e.name).filter((n) => n.endsWith(".nj"));
  }

  // -------------------------------------------------------------------
  // UI: modal
  // -------------------------------------------------------------------
  let modalEl = null;

  function closeModal() {
    if (modalEl && modalEl.parentNode) {
      modalEl.parentNode.removeChild(modalEl);
    }
    modalEl = null;
  }

  function showStatus(msg, kind) {
    if (!modalEl) {
      // Fall back to header status pill.
      const status = document.getElementById("status");
      if (status) status.textContent = msg;
      return;
    }
    const el = modalEl.querySelector(".imp-status");
    if (!el) return;
    el.textContent = msg;
    el.className = "imp-status" + (kind ? " " + kind : "");
  }

  function buildModal(parsed, file) {
    closeModal();
    state.parsedModel = parsed;
    state.parsedFilename = file.name;
    state.parsedFile = file;
    state.suggestedName = (file.name.replace(/\.[^.]+$/, "") || "import").replace(/[^A-Za-z0-9_\-.]/g, "_") + ".nj";
    state.builtNjPath = null;
    state.builtNjMd5 = null;
    state.selectedTargetBml = null;
    state.selectedTargetInner = null;
    state.builtNjmName = null;
    state.builtNjmFrames = 0;
    state.builtNjmBoneCount = 0;

    const root = document.createElement("div");
    root.className = "imp-modal-backdrop";
    root.innerHTML = `
      <div class="imp-modal">
        <div class="imp-modal-head">
          <div class="imp-modal-title">Import 3D model: ${escapeHtml(file.name)}</div>
          <button type="button" class="imp-modal-x" data-act="close">×</button>
        </div>
        <div class="imp-modal-body">
          <div class="imp-section">
            <div class="imp-section-title">Source summary</div>
            <div class="imp-stat-grid">
              <div class="imp-stat-label">format</div>
              <div class="imp-stat-label">meshes</div>
              <div class="imp-stat-label">vertices</div>
              <div class="imp-stat-val" data-fld="format">${escapeHtml(parsed.format || "?")}</div>
              <div class="imp-stat-val" data-fld="mesh_count">${parsed.mesh_count}</div>
              <div class="imp-stat-val" data-fld="vert_total">${parsed.vert_total}</div>
              <div class="imp-stat-label">triangles</div>
              <div class="imp-stat-label">bones</div>
              <div class="imp-stat-label">size</div>
              <div class="imp-stat-val" data-fld="tri_total">${parsed.tri_total}</div>
              <div class="imp-stat-val" data-fld="bone_count">${parsed.bone_count}</div>
              <div class="imp-stat-val" data-fld="size">${file.size} B</div>
            </div>
            ${(parsed.warnings || []).map((w) =>
              `<div class="imp-warn">${escapeHtml(w)}</div>`
            ).join("")}
          </div>
          <div class="imp-section">
            <div class="imp-section-title">Convert</div>
            <div class="imp-form-row">
              <label for="impName">Output filename</label>
              <input type="text" id="impName" value="${escapeHtml(state.suggestedName)}" />
              <span></span>
            </div>
            <div class="imp-form-row">
              <label for="impTpl">Target class</label>
              <select id="impTpl">
                <option value="">(use source skeleton or 1-bone root)</option>
              </select>
              <span></span>
            </div>
            <div class="imp-form-row">
              <label for="impFlip">Flip Z (RH→LH)</label>
              <input type="checkbox" id="impFlip" ${state.axisFlip ? "checked" : ""} />
              <span></span>
            </div>
            <div class="imp-form-row">
              <label for="impScale">Scale</label>
              <input type="range" id="impScale" min="0.01" max="500" step="0.01" value="${state.scale}" />
              <span class="num" data-fld="scaleVal">${state.scale.toFixed(2)}×</span>
            </div>
          </div>
          <div class="imp-section">
            <div class="imp-section-title">Replace target (optional)</div>
            <div style="margin-bottom:6px;color:#99a4b3;font-size:11px">
              Pick a BML + inner .nj to substitute the import for. Leave blank to just build the .nj into cache/nj_export.
            </div>
            <input type="text" class="imp-replace-search" id="impReplaceSearch" placeholder="filter BMLs..." />
            <div class="imp-replace-list" id="impReplaceList">
              <div style="padding:4px;color:#99a4b3">loading...</div>
            </div>
          </div>
          ${(parsed.blend_shapes && parsed.blend_shapes.length > 0) ? `
          <div class="imp-section" id="impBlendShapesSection">
            <div class="imp-section-title">Blend shapes (${parsed.blend_shapes.length})</div>
            <div style="margin-bottom:6px;color:#99a4b3;font-size:11px">
              ${parsed.blend_shapes.length} morph target${parsed.blend_shapes.length === 1 ? "" : "s"} parsed.
              PSOBB doesn't render blend shapes — but you can export them as JSON
              for a Blender re-import workflow or a separate facial-rig pipeline.
            </div>
            <div style="margin-bottom:6px;color:#99a4b3;font-size:10px">
              ${parsed.blend_shapes.slice(0, 8).map((b) => escapeHtml(b.name)).join(", ")}${parsed.blend_shapes.length > 8 ? ", ..." : ""}
            </div>
            <div class="imp-form-row">
              <label></label>
              <div style="display:flex;gap:6px">
                <button type="button" class="imp-btn" data-act="export_blend_shapes">Export blend shapes (JSON)</button>
              </div>
              <span></span>
            </div>
            <div id="impBlendShapesSummary" style="font-size:11px;color:#99a4b3;padding:4px 0"></div>
          </div>
          ` : ""}
          ${(parsed.animations && parsed.animations.length > 0) ? `
          <div class="imp-section" id="impAnimSection">
            <div class="imp-section-title">Animations (${parsed.animations.length})</div>
            <div style="margin-bottom:6px;color:#99a4b3;font-size:11px">
              Found ${parsed.animations.length} animation${parsed.animations.length === 1 ? "" : "s"} in the source.
              Pick a target BML in "Replace target" above, then convert the animation to a deployable .njm.
            </div>
            <div class="imp-form-row">
              <label for="impAnimPick">Animation</label>
              <select id="impAnimPick">
                ${parsed.animations.map((a, i) => `<option value="${i}">${escapeHtml(a.name)} (${a.duration_seconds.toFixed(2)}s, ${a.track_count} tracks)</option>`).join("")}
              </select>
              <span></span>
            </div>
            <div class="imp-form-row">
              <label for="impAnimName">Output .njm</label>
              <input type="text" id="impAnimName" value="" placeholder="e.g. lobby_girl_typing.njm" />
              <span></span>
            </div>
            <div class="imp-form-row">
              <label for="impAnimTrans">Include translation</label>
              <input type="checkbox" id="impAnimTrans" />
              <span style="font-size:10px;color:#99a4b3">leave OFF for in-place motions (typing, idle)</span>
            </div>
            <div class="imp-form-row">
              <label for="impAnimIk">Enable IK retarget</label>
              <input type="checkbox" id="impAnimIk" checked />
              <span style="font-size:10px;color:#99a4b3">closes hand/foot world-position drift on different-length skeletons</span>
            </div>
            <div class="imp-form-row">
              <label for="impAnimIkRot">Enable rotation IK</label>
              <input type="checkbox" id="impAnimIkRot" checked />
              <span style="font-size:10px;color:#99a4b3">match the source's wrist/ankle orientation (v3)</span>
            </div>
            <div class="imp-form-row">
              <label for="impAnimMirror">Mirror left&#8596;right</label>
              <input type="checkbox" id="impAnimMirror" />
              <span style="font-size:10px;color:#99a4b3">flip the animation across YZ plane (e.g. right-hand wave -> left-hand wave)</span>
            </div>
            <div class="imp-form-row">
              <label></label>
              <div style="display:flex;gap:6px">
                <button type="button" class="imp-btn primary" data-act="anim_build" disabled>Convert animation -> NJM</button>
                <button type="button" class="imp-btn" data-act="anim_replace" disabled>Append to BML</button>
              </div>
              <span></span>
            </div>
            <div id="impAnimSummary" style="font-size:11px;color:#99a4b3;padding:4px 0"></div>
          </div>
          ` : ""}
        </div>
        <div class="imp-actions">
          <span class="imp-status">parsed; configure and convert</span>
          <button type="button" class="imp-btn" data-act="cancel">Cancel</button>
          <button type="button" class="imp-btn primary" data-act="build">Convert to NJ</button>
          <button type="button" class="imp-btn" data-act="replace" disabled>Replace + Stage BML</button>
        </div>
      </div>
    `;
    modalEl = root;
    document.body.appendChild(root);

    root.querySelectorAll("[data-act]").forEach((b) => {
      b.addEventListener("click", () => onAction(b.dataset.act));
    });

    // Hydrate template select.
    void hydrateTemplateSelect();

    // Hydrate replace list.
    void hydrateReplaceList();

    // Wire input listeners.
    const flip = root.querySelector("#impFlip");
    if (flip) flip.addEventListener("change", () => { state.axisFlip = !!flip.checked; });
    const sc = root.querySelector("#impScale");
    if (sc) sc.addEventListener("input", () => {
      state.scale = parseFloat(sc.value) || 1.0;
      const lbl = root.querySelector('[data-fld="scaleVal"]');
      if (lbl) lbl.textContent = state.scale.toFixed(2) + "×";
    });
    const tpl = root.querySelector("#impTpl");
    if (tpl) tpl.addEventListener("change", () => { state.selectedTemplate = tpl.value; });
    const search = root.querySelector("#impReplaceSearch");
    if (search) search.addEventListener("input", filterReplaceList);

    // Render preview if the viewer is mounted.
    void renderPreview(parsed);
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  async function hydrateTemplateSelect() {
    const sel = modalEl && modalEl.querySelector("#impTpl");
    if (!sel) return;
    const tpls = await fetchTemplates();
    for (const t of tpls) {
      const opt = document.createElement("option");
      opt.value = t.name;
      opt.textContent = `${t.name} (${t.bone_count} bones)`;
      opt.title = t.description || "";
      sel.appendChild(opt);
    }
    // Auto-pick a template when the source has no skeleton AND triangle
    // count is consistent with one of the templates' typical model.
    if ((state.parsedModel.bones || []).length === 0) {
      // Default to monster_humanoid as a conservative first guess.
      const def = tpls.find((t) => t.name === "monster_humanoid");
      if (def) {
        sel.value = def.name;
        state.selectedTemplate = def.name;
      }
    }
  }

  async function hydrateReplaceList() {
    const list = modalEl && modalEl.querySelector("#impReplaceList");
    if (!list) return;
    const idx = await ensureBmlIndex();
    if (!idx || idx.length === 0) {
      list.innerHTML = '<div style="padding:4px;color:#99a4b3">no BML index available</div>';
      return;
    }
    list.innerHTML = "";
    for (const ent of idx) {
      const row = document.createElement("div");
      row.className = "imp-replace-row";
      row.dataset.bml = ent.bml;
      row.textContent = ent.bml;
      row.addEventListener("click", () => onPickBml(row, ent.bml));
      list.appendChild(row);
    }
  }

  function filterReplaceList() {
    const search = modalEl && modalEl.querySelector("#impReplaceSearch");
    const list = modalEl && modalEl.querySelector("#impReplaceList");
    if (!search || !list) return;
    const term = search.value.toLowerCase().trim();
    list.querySelectorAll(".imp-replace-row").forEach((r) => {
      const txt = r.textContent.toLowerCase();
      r.style.display = (!term || txt.includes(term)) ? "" : "none";
    });
  }

  async function onPickBml(row, bml) {
    const list = modalEl && modalEl.querySelector("#impReplaceList");
    if (!list) return;
    list.querySelectorAll(".imp-replace-row").forEach((r) => r.classList.remove("selected"));
    row.classList.add("selected");
    state.selectedTargetBml = bml;
    state.selectedTargetInner = null;
    // Show inners.
    let block = row.querySelector(".imp-inner-list");
    if (!block) {
      block = document.createElement("div");
      block.className = "imp-inner-list";
      block.style.cssText = "padding:4px 0 4px 16px;font-size:10px;color:#99a4b3";
      block.innerHTML = "loading...";
      row.appendChild(block);
    }
    let inners = [];
    try {
      inners = await loadBmlInners(bml);
    } catch (e) {
      block.innerHTML = `<span style="color:#ff6680">load failed: ${escapeHtml(String(e))}</span>`;
      return;
    }
    if (inners.length === 0) {
      block.innerHTML = "<i>no .nj inners</i>";
      return;
    }
    block.innerHTML = "";
    for (const name of inners) {
      const r2 = document.createElement("div");
      r2.style.cssText = "padding:2px 6px;cursor:pointer;border-radius:2px;color:#c7d8ec";
      r2.textContent = name;
      r2.addEventListener("click", (e) => {
        e.stopPropagation();
        block.querySelectorAll("div").forEach((d) => d.style.background = "");
        r2.style.background = "rgba(74,144,226,0.25)";
        state.selectedTargetInner = name;
        const replace = modalEl && modalEl.querySelector('[data-act="replace"]');
        if (replace) replace.disabled = !state.builtNjPath;
        // Enable the animation conversion button now that we have a target.
        const animBuild = modalEl && modalEl.querySelector('[data-act="anim_build"]');
        if (animBuild) animBuild.disabled = !state.parsedFile;
        // Auto-fill the suggested .njm name if empty.
        const animName = modalEl && modalEl.querySelector("#impAnimName");
        if (animName && !animName.value) {
          const stem = (state.parsedFile && state.parsedFile.name || "anim").replace(/\.[^.]+$/, "");
          animName.value = stem.replace(/[^A-Za-z0-9_\-]/g, "_") + ".njm";
        }
        showStatus(`target: ${bml}#${name}`, "");
      });
      block.appendChild(r2);
    }
  }

  async function renderPreview(parsed) {
    if (typeof window.psoApplyMeshPayload !== "function") return;
    // psoApplyMeshPayload expects payload.mesh_count + payload.meshes[].
    // Our /api/import/parse response is already in that shape.
    try {
      const ok = window.psoApplyMeshPayload(parsed, { label: state.parsedFilename });
      if (!ok) showStatus("preview unavailable (open a model first)", "err");
    } catch (e) {
      console.warn("import: preview failed", e);
    }
  }

  async function onAction(act) {
    if (act === "close" || act === "cancel") {
      closeModal();
      return;
    }
    if (act === "build") {
      await onBuild();
      return;
    }
    if (act === "replace") {
      await onReplace();
      return;
    }
    if (act === "anim_build") {
      await onAnimBuild();
      return;
    }
    if (act === "anim_replace") {
      await onAnimReplace();
      return;
    }
    if (act === "export_blend_shapes") {
      await onExportBlendShapes();
      return;
    }
  }

  async function onExportBlendShapes() {
    if (!modalEl || !state.parsedFile) {
      showStatus("blend-shape export requires the source file", "err");
      return;
    }
    const btn = modalEl.querySelector('[data-act="export_blend_shapes"]');
    if (btn) btn.disabled = true;
    showStatus("exporting blend shapes...", "busy");
    try {
      // Use the parsed filename (stem) as the model_path hint so the
      // output JSON ends up with a recognisable name.
      const stem = (state.parsedFilename || "").replace(/\.[^.]+$/, "") || "blend_shapes";
      const r = await postBlendShapesExport(state.parsedFile, stem);
      const sumEl = modalEl.querySelector("#impBlendShapesSummary");
      if (sumEl) {
        const truncated = (r.names || []).slice(0, 6).join(", ");
        const more = (r.names || []).length > 6 ? ", ..." : "";
        sumEl.innerHTML =
          `wrote <strong>${r.path.split(/[\\/]/).pop()}</strong> (${r.size} B, md5 ${r.md5.slice(0, 8)})<br>` +
          `${r.shape_count} shape${r.shape_count === 1 ? "" : "s"}: ${escapeHtml(truncated)}${more}`;
      }
      showStatus(
        `exported ${r.shape_count} blend shape${r.shape_count === 1 ? "" : "s"} to JSON (md5 ${r.md5.slice(0, 8)})`,
        "ok",
      );
    } catch (e) {
      showStatus("blend-shape export failed: " + e.message, "err");
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  async function onBuild() {
    if (!modalEl || !state.parsedModel) return;
    const nameInput = modalEl.querySelector("#impName");
    const name = (nameInput && nameInput.value || state.suggestedName).trim();
    if (!name.endsWith(".nj")) {
      showStatus("output name must end in .nj", "err");
      return;
    }
    const buildBtn = modalEl.querySelector('[data-act="build"]');
    if (buildBtn) buildBtn.disabled = true;
    showStatus("encoding NJ...", "busy");
    try {
      const r = await postImportBuildNj(state.parsedModel, name, {
        target_class: state.selectedTemplate || null,
        axis_flip_z: state.axisFlip,
        scale: state.scale,
      });
      state.builtNjPath = r.path;
      state.builtNjMd5 = r.md5;
      showStatus(
        `built ${name}: ${r.size} B (md5 ${r.md5.slice(0, 8)}, ${r.vert_count} verts, ${r.bone_count} bones)`,
        "ok",
      );
      const rep = modalEl.querySelector('[data-act="replace"]');
      if (rep) rep.disabled = !(state.selectedTargetBml && state.selectedTargetInner);
    } catch (e) {
      showStatus("build failed: " + e.message, "err");
    } finally {
      if (buildBtn) buildBtn.disabled = false;
    }
  }

  async function onReplace() {
    if (!modalEl || !state.builtNjPath) return;
    if (!state.selectedTargetBml || !state.selectedTargetInner) {
      showStatus("pick a BML and inner .nj first", "err");
      return;
    }
    const repBtn = modalEl.querySelector('[data-act="replace"]');
    if (repBtn) repBtn.disabled = true;
    showStatus("re-packing BML...", "busy");
    try {
      // Path part of state.builtNjPath is absolute; the server expects the
      // bare filename in cache/nj_export.
      const njName = state.builtNjPath.replace(/\\/g, "/").split("/").pop();
      const r = await postImportReplace({
        import_nj_path: njName,
        target_bml: state.selectedTargetBml,
        target_inner: state.selectedTargetInner,
      });
      showStatus(
        `staged ${r.archive_name}: ${r.size} B (md5 ${r.md5.slice(0, 8)})`,
        "ok",
      );
      // If the live-test helper exists, mount a button.
      if (window.PSOLiveTest && typeof window.PSOLiveTest.triggerLiveTest === "function") {
        // Just notify the user — they hit the existing button to push.
        showStatus(
          `staged ${r.archive_name} — click "Live Test" or deploy to push`,
          "ok",
        );
      }
    } catch (e) {
      showStatus("replace failed: " + e.message, "err");
    } finally {
      if (repBtn) repBtn.disabled = false;
    }
  }

  // 2026-04-25 v2: animation conversion handlers.
  async function onAnimBuild() {
    if (!modalEl || !state.parsedFile) return;
    if (!state.selectedTargetBml) {
      showStatus("pick a BML in 'Replace target' first (the skeleton source)", "err");
      return;
    }
    const animName = modalEl.querySelector("#impAnimName");
    const motionName = (animName && animName.value || "").trim();
    if (!motionName.endsWith(".njm")) {
      showStatus(".njm filename must end in .njm", "err");
      return;
    }
    const trans = modalEl.querySelector("#impAnimTrans");
    const include_translation = !!(trans && trans.checked);
    const ikToggle = modalEl.querySelector("#impAnimIk");
    // Default to ON (matches the checkbox's default `checked` attribute)
    // when the toggle isn't present in the DOM (older modal cache).
    const enable_ik = ikToggle ? !!ikToggle.checked : true;
    // 2026-04-25 v3: rotation IK + mirror toggles. Default ON for
    // rotation IK (matches the v3 server default), OFF for mirror.
    const ikRotToggle = modalEl.querySelector("#impAnimIkRot");
    const enable_ik_rotation = ikRotToggle ? !!ikRotToggle.checked : true;
    const mirrorToggle = modalEl.querySelector("#impAnimMirror");
    const mirror = mirrorToggle ? !!mirrorToggle.checked : false;
    const btn = modalEl.querySelector('[data-act="anim_build"]');
    if (btn) btn.disabled = true;
    showStatus("retargeting animation...", "busy");
    try {
      const r = await postImportAnimation(state.parsedFile, {
        target_model_path: state.selectedTargetBml,
        motion_name: motionName,
        target_inner: state.selectedTargetInner || null,
        include_translation,
        enable_ik,
        enable_ik_rotation,
        mirror,
        flip_z: state.axisFlip,
        target_fps: 30,
      });
      state.builtNjmName = r.njm_name;
      state.builtNjmFrames = r.frame_count;
      state.builtNjmBoneCount = r.bone_count;
      const summary = modalEl.querySelector("#impAnimSummary");
      if (summary) {
        summary.innerHTML = `
          <div>retargeted <b>${r.retargeted_bones}</b> bones (dropped ${r.dropped_bones})</div>
          <div>${r.frame_count} frames @ 30 fps from <b>${escapeHtml(r.source_animation)}</b></div>
          <div>staged <b>${escapeHtml(r.njm_name)}</b> (${r.size} B, md5 ${r.md5.slice(0, 8)})</div>
        `;
      }
      showStatus(
        `built ${r.njm_name}: ${r.frame_count} frames, ${r.retargeted_bones} bones retargeted`,
        "ok",
      );
      const ar = modalEl.querySelector('[data-act="anim_replace"]');
      if (ar) ar.disabled = false;
    } catch (e) {
      showStatus("animation build failed: " + e.message, "err");
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  async function onAnimReplace() {
    if (!modalEl || !state.builtNjmName) return;
    if (!state.selectedTargetBml) {
      showStatus("pick a target BML first", "err");
      return;
    }
    const animName = modalEl.querySelector("#impAnimName");
    const motionName = (animName && animName.value || state.builtNjmName).trim().replace(/\.njm$/i, "");
    const btn = modalEl.querySelector('[data-act="anim_replace"]');
    if (btn) btn.disabled = true;
    showStatus("re-packing BML with new motion...", "busy");
    try {
      const r = await postImportAnimationReplace({
        njm_path: state.builtNjmName,
        target_bml: state.selectedTargetBml,
        target_motion_name: motionName,
        append_if_missing: true,
      });
      showStatus(
        `${r.operation} motion '${motionName}' in ${r.archive_name} (md5 ${r.md5.slice(0, 8)}) — deploy via header button`,
        "ok",
      );
    } catch (e) {
      showStatus("animation replace failed: " + e.message, "err");
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  async function parseAndOpen(file) {
    const status = document.getElementById("status");
    if (status) status.textContent = `parsing ${file.name}...`;
    try {
      const parsed = await postImportParse(file);
      buildModal(parsed, file);
    } catch (e) {
      const msg = `import parse failed: ${e.message}`;
      if (status) status.textContent = msg;
      console.warn(msg, e);
      alert(msg);
    }
  }

  // -------------------------------------------------------------------
  // Public API surface (mostly for tests + dev console).
  // -------------------------------------------------------------------
  window.psoImportPanel = {
    open: parseAndOpen,
    state,
  };

  // -------------------------------------------------------------------
  // Recent imports — surfaced via live-reload bus (v5 polish, 2026-04-25).
  // We keep an in-memory ring of the last N create/modify events on
  // cache/bml_export/ so the user can "see what just landed" without
  // having to F5 or open the asset tree manually. Exposed via
  // window.psoImportPanel.recentImports for now; a future micro-toast
  // can surface it visually.
  // -------------------------------------------------------------------
  const RECENT_IMPORTS_MAX = 16;
  state.recentImports = [];

  function attachLiveReloadListener() {
    if (!window.bus || typeof window.bus.on !== "function") {
      // bus not ready yet — retry shortly. The bus is sticky once
      // initialized so a one-off retry is safe.
      setTimeout(attachLiveReloadListener, 100);
      return;
    }
    window.bus.on("cache.changed", (payload) => {
      if (!payload || !payload.path) return;
      // Surface anything that lands in bml_export/ (the natural
      // artifact of an import-and-replace flow) AND nj_export/ (the
      // bare .nj output when no replacement was requested).
      const isImport = payload.path.indexOf("cache/bml_export/") === 0
                     || payload.path.indexOf("cache/nj_export/") === 0;
      if (!isImport) return;
      if (payload.kind === "delete") {
        state.recentImports = state.recentImports.filter((it) => it.path !== payload.path);
        return;
      }
      // Drop any prior entry with the same path so the latest mtime wins.
      state.recentImports = state.recentImports.filter((it) => it.path !== payload.path);
      state.recentImports.unshift({
        path: payload.path,
        kind: payload.kind,
        ts: payload.ts || Date.now(),
      });
      if (state.recentImports.length > RECENT_IMPORTS_MAX) {
        state.recentImports.length = RECENT_IMPORTS_MAX;
      }
    });
  }

  // -------------------------------------------------------------------
  // Boot
  // -------------------------------------------------------------------
  function boot() {
    ensureStyleInjected();
    attachDropHandlers();
    attachHeaderButton();
    attachLiveReloadListener();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
