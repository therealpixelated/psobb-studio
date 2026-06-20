// =====================================================================
// PSOBB Material Inspector / Editor Panel
// 2026-04-25
//
// Adds a "Material" tab to the existing texture-panel tab strip via
// the standard psoTexturePanelAddTabButton + psoTexturePanelRegisterTab
// hooks.  Mirrors the shape of rig_panel / sculpt_panel / paint_panel —
// no DOM-mounting magic, just a renderer callback.
//
// Per-submesh view shows:
//   - Submesh index + bound texture name (from /api/model_textures)
//   - Diffuse RGBA color picker
//   - Alpha test enable + threshold slider (0..255)
//   - Alpha blend mode dropdown (none / blend / additive / multiply / screen)
//   - Two-sided checkbox
//   - Depth test / depth write checkboxes
//   - Preset chooser: shipping-value catalog (player skin, hair/fur,
//     glass/energy, standard solid, transparent blend)
//
// Live preview: every edit calls window.psoUpdateMaterial (added by
// model_viewer.js) so the 3D viewport reflects changes immediately.
//
// Save flow:
//   POST /api/material/<bml> { inner, submeshes: [...] }
//     -> stages cache/bml_export/<bml>.bml
//     -> user can deploy via the existing /api/deploy/<archive> path
//        (the Deploy button on the texture panel works for our archive
//         too because both panels stage to the same dir)
//
// Wire-format read:
//   GET /api/material/<bml>?inner=<inner.nj>
//     -> {submesh_count, submeshes: [{idx, material_id, diffuse_rgba,
//                                     alpha_test, alpha_blend, blend_mode,
//                                     two_sided, depth_test, depth_write,
//                                     ...}]}
//
// =====================================================================

(function () {
  "use strict";

  if (window.__psoMaterialPanelLoaded) return;
  window.__psoMaterialPanelLoaded = true;

  const TAB_NAME  = "material";
  const TAB_LABEL = "Material";
  const TAB_TITLE =
    "Per-submesh material flags: diffuse / alpha-test / blend / two-sided / depth";

  const STYLE_ID = "psoMaterialPanelStyle";

  // ---- state -------------------------------------------------------

  // The per-submesh edit state.  Keyed by submesh global idx.  We
  // accumulate edits (without mutating the originally-fetched data)
  // so the user can walk away from a submesh and come back to it.
  const state = {
    modelPath: null,            // last-fetched path (e.g. "<bml>#<inner>.nj")
    bmlBase: null,              // <bml>
    inner: null,                // <inner.nj>
    submeshes: [],              // raw GET response — never mutated
    edits: new Map(),           // submeshIdx -> edit dict
    presets: [],                // GET /api/material_presets cache
    selectedSubmesh: 0,         // active submesh index in the list
    multiSelect: new Set(),     // additional submesh indices (Apply-to-selection)
    statusKind: "idle",         // idle | running | done | err
    statusMsg: "",
    materialIdToTexName: {},    // material_id -> human texture name
    busy: false,
  };

  // ---- helpers -----------------------------------------------------

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;",
      '"': "&quot;", "'": "&#39;",
    })[c]);
  }

  function _setStatus(kind, msg) {
    state.statusKind = kind;
    state.statusMsg = msg;
    const node = document.querySelector(".pso-mat-inspector .pso-mat-status");
    if (node) {
      node.dataset.kind = kind;
      node.textContent = msg || "";
    }
  }

  function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const css = `
      .pso-mat-inspector { display:flex; flex-direction:column; gap:8px;
        padding:8px; font-size:12px; color:var(--text-fg, #cfd2d6); }
      .pso-mat-toolbar { display:flex; gap:8px; align-items:center;
        flex-wrap:wrap; padding-bottom:6px;
        border-bottom:1px solid var(--bd-color, #2a2a2a); }
      .pso-mat-toolbar select, .pso-mat-toolbar button {
        background:var(--bg-2,#1f1f1f); color:inherit;
        border:1px solid var(--bd-color,#2a2a2a); padding:3px 6px;
        font-size:12px; border-radius:3px; }
      .pso-mat-toolbar button { cursor:pointer; }
      .pso-mat-toolbar button:hover { background:var(--bg-3,#2a2a2a); }
      .pso-mat-toolbar button.primary {
        background:var(--accent,#3d7be4); color:#fff;
        border-color:var(--accent,#3d7be4); }
      .pso-mat-toolbar button:disabled { opacity:.45; cursor:not-allowed; }
      .pso-mat-status { font-size:11px; padding:2px 4px; border-radius:3px;
        margin-left:auto; }
      .pso-mat-status[data-kind="running"] { color:#dba; }
      .pso-mat-status[data-kind="done"]    { color:#7c7; }
      .pso-mat-status[data-kind="err"]     { color:#e88; }
      .pso-mat-grid {
        display:grid; grid-template-columns: 200px 1fr; gap:8px;
        flex:1 1 auto; min-height:0; }
      .pso-mat-list {
        overflow-y:auto; border:1px solid var(--bd-color,#2a2a2a);
        border-radius:3px; max-height:60vh; }
      .pso-mat-list-row {
        padding:4px 6px; cursor:pointer;
        border-bottom:1px solid var(--bd-color,#2a2a2a); }
      .pso-mat-list-row:hover { background:var(--bg-2,#202020); }
      .pso-mat-list-row.active {
        background:var(--accent,#3d7be4); color:#fff; }
      .pso-mat-list-row .sub-line {
        font-size:10px; opacity:.7; padding-left:8px; }
      .pso-mat-list-row .dirty-dot {
        display:inline-block; width:6px; height:6px; border-radius:50%;
        background:#ee9133; margin-right:4px; vertical-align:middle; }
      .pso-mat-detail {
        display:flex; flex-direction:column; gap:10px;
        padding:8px; overflow-y:auto;
        border:1px solid var(--bd-color,#2a2a2a); border-radius:3px; }
      .pso-mat-row {
        display:flex; align-items:center; gap:8px; min-height:24px; }
      .pso-mat-row > label.kk {
        flex:0 0 110px; font-weight:600; }
      .pso-mat-row input[type=color] {
        background:transparent; border:1px solid var(--bd-color,#2a2a2a);
        width:36px; height:22px; padding:0; }
      .pso-mat-row input[type=range] { flex:1 1 auto; }
      .pso-mat-row input[type=number] {
        width:50px; background:var(--bg-2,#1f1f1f); color:inherit;
        border:1px solid var(--bd-color,#2a2a2a); padding:2px 4px;
        font-size:12px; }
      .pso-mat-row input[type=text] {
        background:var(--bg-2,#1f1f1f); color:inherit;
        border:1px solid var(--bd-color,#2a2a2a); padding:2px 4px;
        font-size:12px; flex:1; }
      .pso-mat-row select {
        background:var(--bg-2,#1f1f1f); color:inherit;
        border:1px solid var(--bd-color,#2a2a2a); padding:2px 4px; }
      .pso-mat-presets { display:flex; gap:4px; flex-wrap:wrap;
        padding-top:6px; border-top:1px solid var(--bd-color,#2a2a2a); }
      .pso-mat-presets button { font-size:11px; padding:3px 6px; }
      .pso-mat-detail h3 { margin:6px 0 0 0; font-size:12px;
        color:var(--accent,#3d7be4); border-bottom:1px solid var(--bd-color,#2a2a2a);
        padding-bottom:2px; }
      .pso-mat-detail .hint { font-size:10px; opacity:.65; padding-left:8px; }
      .pso-mat-empty { padding:20px; text-align:center; opacity:.55; }
      .pso-mat-multi-hint {
        font-size:10px; padding:4px 6px; background:var(--bg-2,#1f1f1f);
        border-left:3px solid var(--accent,#3d7be4); margin-top:4px; }
    `;
    const tag = document.createElement("style");
    tag.id = STYLE_ID;
    tag.textContent = css;
    document.head.appendChild(tag);
  }

  function rgbaToHex(rgba) {
    if (!rgba || rgba.length < 3) return "#ffffff";
    const r = (rgba[0] | 0).toString(16).padStart(2, "0");
    const g = (rgba[1] | 0).toString(16).padStart(2, "0");
    const b = (rgba[2] | 0).toString(16).padStart(2, "0");
    return `#${r}${g}${b}`;
  }

  function hexToRgba(hex, alpha) {
    if (!hex || hex.length < 7) return [255, 255, 255, alpha | 0];
    return [
      parseInt(hex.slice(1, 3), 16),
      parseInt(hex.slice(3, 5), 16),
      parseInt(hex.slice(5, 7), 16),
      alpha != null ? (alpha | 0) : 255,
    ];
  }

  function getCurrentValue(submeshIdx, field) {
    const edit = state.edits.get(submeshIdx);
    if (edit && edit[field] !== undefined) return edit[field];
    const sm = state.submeshes[submeshIdx];
    return sm ? sm[field] : undefined;
  }

  function setEdit(submeshIdx, field, value) {
    let edit = state.edits.get(submeshIdx);
    if (!edit) {
      edit = {submesh_idx: submeshIdx};
      state.edits.set(submeshIdx, edit);
    }
    edit[field] = value;

    // Multi-select: copy to every other selected submesh too.
    if (state.multiSelect.size > 0) {
      for (const otherIdx of state.multiSelect) {
        if (otherIdx === submeshIdx) continue;
        let oe = state.edits.get(otherIdx);
        if (!oe) { oe = {submesh_idx: otherIdx}; state.edits.set(otherIdx, oe); }
        oe[field] = value;
        applyLivePreview(otherIdx, {[field]: value});
      }
    }
    applyLivePreview(submeshIdx, {[field]: value});
  }

  function applyLivePreview(submeshIdx, partial) {
    if (typeof window.psoUpdateMaterial !== "function") return;
    // Compose a full edit object so the renderer hook sees consistent state.
    const sm = state.submeshes[submeshIdx] || {};
    const edit = state.edits.get(submeshIdx) || {};
    const merged = Object.assign(
      {
        diffuse_rgba: sm.diffuse_rgba,
        alpha_test:   sm.alpha_test,
        alpha_blend:  sm.alpha_blend,
        blend_mode:   sm.blend_mode,
        two_sided:    sm.two_sided,
        depth_test:   sm.depth_test,
        depth_write:  sm.depth_write,
      },
      edit,
      partial,
    );
    try { window.psoUpdateMaterial(submeshIdx, merged); }
    catch (_e) { /* swallow — best-effort live preview */ }
  }

  // ---- model resolution + fetch -----------------------------------

  function resolveCurrentModel() {
    const meta = (typeof window.psoGetSculptMeshGroup === "function")
      ? window.psoGetSculptMeshGroup() : null;
    if (!meta || !meta.modelPath) return null;
    return meta.modelPath;
  }

  function splitModelPath(mp) {
    // "<bml>#<inner>.nj" or "<file>.nj"
    const i = mp.indexOf("#");
    if (i < 0) return {base: mp, inner: null};
    return {base: mp.slice(0, i), inner: mp.slice(i + 1)};
  }

  async function fetchMaterials(modelPath) {
    const {base, inner} = splitModelPath(modelPath);
    let url;
    if (inner) {
      url = `/api/material/${encodeURIComponent(base)}?inner=${encodeURIComponent(inner)}`;
    } else {
      url = `/api/material/${encodeURIComponent(base)}`;
    }
    const r = await fetch(url);
    if (!r.ok) {
      let msg = `HTTP ${r.status}`;
      try { msg = (await r.json()).detail || msg; } catch {}
      throw new Error(msg);
    }
    return r.json();
  }

  async function fetchTextureNames(modelPath) {
    // Best-effort: pull /api/model_textures and build a material_id -> name map.
    const {base, inner} = splitModelPath(modelPath);
    let url;
    if (inner) {
      url = `/api/model_textures/${encodeURIComponent(base)}?inner=${encodeURIComponent(inner)}`;
    } else {
      url = `/api/model_textures/${encodeURIComponent(base)}`;
    }
    try {
      const r = await fetch(url);
      if (!r.ok) return {};
      const data = await r.json();
      const out = {};
      for (const b of (data.binding || [])) {
        out[b.material_id] = b.name || `tex#${b.tile_index}`;
      }
      return out;
    } catch (_e) {
      return {};
    }
  }

  async function fetchPresets() {
    if (state.presets.length > 0) return state.presets;
    try {
      const r = await fetch("/api/material_presets");
      if (!r.ok) return [];
      const data = await r.json();
      state.presets = data.presets || [];
      return state.presets;
    } catch (_e) {
      return [];
    }
  }

  // ---- save --------------------------------------------------------

  async function saveAll() {
    if (state.edits.size === 0) {
      _setStatus("err", "no edits to save");
      return;
    }
    if (!state.bmlBase) {
      _setStatus("err", "no model loaded");
      return;
    }
    const submeshes = [];
    for (const [idx, edit] of state.edits.entries()) {
      submeshes.push({...edit, submesh_idx: idx});
    }
    state.busy = true;
    _setStatus("running", `saving ${submeshes.length} edit${submeshes.length === 1 ? '' : 's'}…`);
    try {
      const r = await fetch(`/api/material/${encodeURIComponent(state.bmlBase)}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({inner: state.inner, submeshes}),
      });
      if (!r.ok) {
        let msg = `HTTP ${r.status}`;
        try { msg = (await r.json()).detail || msg; } catch {}
        throw new Error(msg);
      }
      const data = await r.json();
      _setStatus("done", `staged ${(data.size / 1024).toFixed(1)} KB · md5 ${(data.md5 || "").slice(0, 8)}…`);
    } catch (e) {
      _setStatus("err", `save failed: ${e.message || e}`);
    } finally {
      state.busy = false;
    }
  }

  function discardEdits() {
    state.edits.clear();
    // Reset live preview to match server-side originals.
    for (let i = 0; i < state.submeshes.length; i++) {
      applyLivePreview(i, state.submeshes[i]);
    }
    _setStatus("idle", "edits discarded");
  }

  // ---- rendering ---------------------------------------------------

  function renderPanel(body) {
    ensureStyle();
    body.innerHTML = `
      <div class="pso-mat-inspector">
        <div class="pso-mat-toolbar">
          <button data-act="reload" title="Reload from disk">Reload</button>
          <button data-act="save" class="primary" title="POST /api/material to stage a new BML">Save</button>
          <button data-act="discard" title="Drop unsaved edits">Discard</button>
          <span class="pso-mat-status" data-kind="${state.statusKind}">${escapeHtml(state.statusMsg || "")}</span>
        </div>
        <div data-region="grid" style="flex:1;"></div>
      </div>
    `;
    const tb = body.querySelector(".pso-mat-toolbar");
    tb.addEventListener("click", onToolbarClick);

    // Trigger initial fetch.
    const grid = body.querySelector('[data-region="grid"]');
    grid.innerHTML = `<div class="pso-mat-empty">Loading…</div>`;
    loadAndRender(grid);
  }

  async function loadAndRender(grid) {
    const modelPath = resolveCurrentModel();
    if (!modelPath) {
      grid.innerHTML = `<div class="pso-mat-empty">No model loaded.<br><span class="hint">Open a .nj or BML inner from the model tree first.</span></div>`;
      return;
    }
    const split = splitModelPath(modelPath);
    state.modelPath = modelPath;
    state.bmlBase = split.base;
    state.inner = split.inner;

    try {
      const [data, texNames, _presets] = await Promise.all([
        fetchMaterials(modelPath),
        fetchTextureNames(modelPath),
        fetchPresets(),
      ]);
      state.submeshes = data.submeshes || [];
      state.materialIdToTexName = texNames;
      if (state.submeshes.length === 0) {
        grid.innerHTML = `<div class="pso-mat-empty">No submeshes in this model.</div>`;
        return;
      }
      // Reset selection to first submesh if the prior one is now invalid.
      if (state.selectedSubmesh >= state.submeshes.length) {
        state.selectedSubmesh = 0;
      }
      renderGrid(grid);
    } catch (e) {
      grid.innerHTML = `<div class="pso-mat-empty">Load failed: ${escapeHtml(e.message || String(e))}</div>`;
    }
  }

  function renderGrid(grid) {
    grid.innerHTML = `
      <div class="pso-mat-grid">
        <div class="pso-mat-list" data-region="list"></div>
        <div class="pso-mat-detail" data-region="detail"></div>
      </div>
    `;
    renderList(grid.querySelector('[data-region="list"]'));
    renderDetail(grid.querySelector('[data-region="detail"]'));
  }

  function renderList(listEl) {
    const rows = state.submeshes.map((sm, i) => {
      const isActive = i === state.selectedSubmesh;
      const isMulti = state.multiSelect.has(i);
      const dirty = state.edits.has(i);
      const texName = state.materialIdToTexName[sm.material_id]
        || `tex#${sm.material_id}`;
      const cls = [
        "pso-mat-list-row",
        isActive ? "active" : "",
        isMulti ? "multi" : "",
      ].filter(Boolean).join(" ");
      return `
        <div class="${cls}" data-idx="${i}">
          ${dirty ? '<span class="dirty-dot" title="has edits"></span>' : ''}
          <span>#${i} · mat ${sm.material_id}</span>
          <div class="sub-line">${escapeHtml(texName)} · ${escapeHtml(sm.blend_mode)}${sm.two_sided ? " · 2-side" : ""}</div>
        </div>
      `;
    }).join("");
    listEl.innerHTML = rows;
    listEl.addEventListener("click", onListClick);
  }

  function renderDetail(detailEl) {
    const idx = state.selectedSubmesh;
    const sm = state.submeshes[idx];
    if (!sm) {
      detailEl.innerHTML = `<div class="pso-mat-empty">Select a submesh on the left.</div>`;
      return;
    }
    const cur = {
      diffuse_rgba: getCurrentValue(idx, "diffuse_rgba"),
      alpha_test:   getCurrentValue(idx, "alpha_test"),
      alpha_blend:  getCurrentValue(idx, "alpha_blend"),
      blend_mode:   getCurrentValue(idx, "blend_mode"),
      two_sided:    getCurrentValue(idx, "two_sided"),
      depth_test:   getCurrentValue(idx, "depth_test"),
      depth_write:  getCurrentValue(idx, "depth_write"),
    };
    const diffuse = cur.diffuse_rgba || [255, 255, 255, 255];
    const alpha = (diffuse[3] != null ? diffuse[3] : 255) | 0;
    const at = cur.alpha_test || {enabled: false, threshold: 128};
    const ab = cur.alpha_blend;
    const blendMode = cur.blend_mode || "none";
    const texName = state.materialIdToTexName[sm.material_id]
      || `tex#${sm.material_id}`;

    const presetButtons = (state.presets || []).map((p) =>
      `<button data-preset="${escapeHtml(p.key)}" title="${escapeHtml(p.description || '')}">${escapeHtml(p.label || p.key)}</button>`
    ).join("");

    const multiHint = state.multiSelect.size > 0
      ? `<div class="pso-mat-multi-hint">Multi-select: ${state.multiSelect.size + 1} submesh${state.multiSelect.size === 0 ? '' : 'es'} · edits apply to all.</div>`
      : "";

    detailEl.innerHTML = `
      <h3>Submesh #${idx}</h3>
      ${multiHint}
      <div class="pso-mat-row">
        <label class="kk">Material id</label>
        <span>${sm.material_id}</span>
        <span class="hint">${escapeHtml(texName)}</span>
      </div>
      <div class="pso-mat-row">
        <label class="kk">Mesh</label>
        <span class="hint">parent mesh ${sm.mesh_idx ?? "?"} · strip ${sm.submesh_idx_in_mesh ?? "?"}</span>
      </div>

      <h3>Color</h3>
      <div class="pso-mat-row">
        <label class="kk">Diffuse RGB</label>
        <input type="color" data-field="diffuse_rgb" value="${rgbaToHex(diffuse)}">
        <label class="kk" style="flex:0 0 auto;">A</label>
        <input type="range" data-field="diffuse_alpha" min="0" max="255" value="${alpha}">
        <input type="number" data-field="diffuse_alpha_n" min="0" max="255" value="${alpha}">
      </div>

      <h3>Alpha test</h3>
      <div class="pso-mat-row">
        <label class="kk"><input type="checkbox" data-field="alpha_test_en"${at.enabled ? " checked" : ""}> Enabled</label>
        <input type="range" data-field="alpha_test_threshold" min="0" max="255" value="${at.threshold}" ${at.enabled ? "" : "disabled"}>
        <input type="number" data-field="alpha_test_threshold_n" min="0" max="255" value="${at.threshold}" ${at.enabled ? "" : "disabled"}>
      </div>

      <h3>Blend mode</h3>
      <div class="pso-mat-row">
        <label class="kk">Mode</label>
        <select data-field="blend_mode">
          <option value="none" ${blendMode === "none" ? "selected" : ""}>none (opaque)</option>
          <option value="blend" ${blendMode === "blend" ? "selected" : ""}>blend (src_alpha / inv)</option>
          <option value="additive" ${blendMode === "additive" ? "selected" : ""}>additive (glow)</option>
          <option value="multiply" ${blendMode === "multiply" ? "selected" : ""}>multiply (decal darkens)</option>
          <option value="screen" ${blendMode === "screen" ? "selected" : ""}>screen (lighten)</option>
        </select>
        <span class="hint">${ab ? `${escapeHtml(ab.src)} / ${escapeHtml(ab.dst)}` : "default"}</span>
      </div>

      <h3>Render flags</h3>
      <div class="pso-mat-row">
        <label class="kk"><input type="checkbox" data-field="two_sided"${cur.two_sided ? " checked" : ""}> Two-sided</label>
        <span class="hint">disable backface cull</span>
      </div>
      <div class="pso-mat-row">
        <label class="kk"><input type="checkbox" data-field="depth_test"${cur.depth_test !== false ? " checked" : ""}> Depth test</label>
      </div>
      <div class="pso-mat-row">
        <label class="kk"><input type="checkbox" data-field="depth_write"${cur.depth_write !== false ? " checked" : ""}> Depth write</label>
      </div>

      <div class="pso-mat-presets">
        <span class="hint">Presets:</span>
        ${presetButtons}
      </div>
    `;
    detailEl.addEventListener("input", onDetailInput);
    detailEl.addEventListener("change", onDetailInput);
    detailEl.addEventListener("click", onDetailClick);
  }

  function rerender() {
    const body = document.querySelector(".pso-tex-panel-body, [data-region='body']");
    if (!body) return;
    const grid = body.querySelector('[data-region="grid"]');
    if (!grid) return;
    renderGrid(grid);
  }

  // ---- event handlers ---------------------------------------------

  function onToolbarClick(ev) {
    const t = ev.target.closest("button[data-act]");
    if (!t) return;
    const act = t.dataset.act;
    if (act === "reload") {
      const body = t.closest(".pso-mat-inspector");
      if (body) {
        const grid = body.parentElement.querySelector('[data-region="grid"]') || body.querySelector('[data-region="grid"]');
        loadAndRender(grid || body);
      }
      _setStatus("running", "reloading…");
      return;
    }
    if (act === "save")    { saveAll(); return; }
    if (act === "discard") { discardEdits(); rerender(); return; }
  }

  function onListClick(ev) {
    const row = ev.target.closest(".pso-mat-list-row");
    if (!row) return;
    const idx = parseInt(row.dataset.idx, 10);
    if (isNaN(idx)) return;
    if (ev.shiftKey || ev.ctrlKey || ev.metaKey) {
      // Multi-select: add to set unless it's the active one.
      if (idx === state.selectedSubmesh) return;
      if (state.multiSelect.has(idx)) state.multiSelect.delete(idx);
      else state.multiSelect.add(idx);
    } else {
      state.selectedSubmesh = idx;
      state.multiSelect.clear();
    }
    rerender();
  }

  function onDetailInput(ev) {
    const t = ev.target;
    if (!t || !t.dataset || !t.dataset.field) return;
    const idx = state.selectedSubmesh;
    const field = t.dataset.field;

    // Diffuse RGB picker.
    if (field === "diffuse_rgb") {
      const oldA = (getCurrentValue(idx, "diffuse_rgba") || [255,255,255,255])[3] | 0;
      setEdit(idx, "diffuse_rgba", hexToRgba(t.value, oldA));
      rerender();
      return;
    }
    if (field === "diffuse_alpha" || field === "diffuse_alpha_n") {
      const a = Math.max(0, Math.min(255, parseInt(t.value, 10) || 0));
      const old = getCurrentValue(idx, "diffuse_rgba") || [255, 255, 255, 255];
      setEdit(idx, "diffuse_rgba", [old[0], old[1], old[2], a]);
      rerender();
      return;
    }
    if (field === "alpha_test_en") {
      const cur = getCurrentValue(idx, "alpha_test") || {enabled: false, threshold: 128};
      setEdit(idx, "alpha_test", t.checked
        ? {enabled: true, threshold: cur.threshold || 128}
        : null);
      rerender();
      return;
    }
    if (field === "alpha_test_threshold" || field === "alpha_test_threshold_n") {
      const v = Math.max(0, Math.min(255, parseInt(t.value, 10) || 0));
      const cur = getCurrentValue(idx, "alpha_test") || {enabled: true, threshold: 128};
      setEdit(idx, "alpha_test", {enabled: cur.enabled !== false, threshold: v});
      rerender();
      return;
    }
    if (field === "blend_mode") {
      const mode = t.value;
      // Translate the high-level mode -> (src, dst) factor pair.
      const presets = {
        "none":      null,
        "blend":     {src: "src_alpha", dst: "one_minus_src_alpha"},
        "additive":  {src: "src_alpha", dst: "one"},
        "multiply":  {src: "dst_color", dst: "zero"},
        "screen":    {src: "one",       dst: "one_minus_src_color"},
      };
      setEdit(idx, "blend_mode", mode);
      setEdit(idx, "alpha_blend", presets[mode] || null);
      rerender();
      return;
    }
    if (field === "two_sided")   { setEdit(idx, "two_sided",   !!t.checked); return; }
    if (field === "depth_test")  { setEdit(idx, "depth_test",  !!t.checked); return; }
    if (field === "depth_write") { setEdit(idx, "depth_write", !!t.checked); return; }
  }

  function onDetailClick(ev) {
    const t = ev.target.closest("button[data-preset]");
    if (!t) return;
    const presetKey = t.dataset.preset;
    const preset = (state.presets || []).find((p) => p.key === presetKey);
    if (!preset) return;
    const idx = state.selectedSubmesh;
    // Apply each preset field — careful to handle null vs undefined.
    if ("alpha_test" in preset)  setEdit(idx, "alpha_test", preset.alpha_test);
    if ("alpha_blend" in preset) setEdit(idx, "alpha_blend", preset.alpha_blend);
    if ("blend_mode" in preset)  setEdit(idx, "blend_mode", preset.blend_mode);
    if ("two_sided" in preset)   setEdit(idx, "two_sided", !!preset.two_sided);
    if ("depth_test" in preset)  setEdit(idx, "depth_test", preset.depth_test !== false);
    if ("depth_write" in preset) setEdit(idx, "depth_write", preset.depth_write !== false);
    _setStatus("done", `applied preset: ${preset.label}`);
    rerender();
  }

  // ---- tab integration --------------------------------------------

  function injectMaterialTab() {
    if (typeof window.psoTexturePanelAddTabButton !== "function") return false;
    if (typeof window.psoTexturePanelRegisterTab !== "function") return false;
    const ok = window.psoTexturePanelAddTabButton(TAB_NAME, TAB_LABEL, TAB_TITLE);
    window.psoTexturePanelRegisterTab(TAB_NAME, (body) => renderPanel(body));
    return ok;
  }

  function waitForPanel(deadline) {
    if (injectMaterialTab()) return;
    if (Date.now() > deadline) {
      console.warn("[material_panel] texture panel never appeared; material inspector disabled");
      return;
    }
    setTimeout(() => waitForPanel(deadline), 250);
  }

  function init() {
    waitForPanel(Date.now() + 30_000);
    // When a new model loads, drop our state so the next time the
    // tab opens we re-fetch.
    if (window.bus && typeof window.bus.on === "function") {
      window.bus.on("model.loaded", () => {
        state.modelPath = null;
        state.bmlBase = null;
        state.inner = null;
        state.submeshes = [];
        state.edits.clear();
        state.multiSelect.clear();
        state.selectedSubmesh = 0;
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
