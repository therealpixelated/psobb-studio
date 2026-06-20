// =====================================================================
// PSOBB Texture Editor - Floor copy/create editor perspective (2026-06-20)
//
// A full-stage workspace to browse floors, Preview one in the shared
// model_viewer 3D canvas, Copy a floor into an editable DEV slot, or
// Create a floor from a GLB (upload OR imported-asset ref) via the
// build_lobby pipeline.
//
// This is a PSOPerspectives perspective CLONED from map_panel.js — NOT a
// model-viewer tab. It relocates the shared model_viewer canvas into its
// own #floorViewport (same pluck-and-restore dance as map_panel mount) and
// renders the scene via window.psoSceneLoadMapWithEnvironment(bundle).
//
// Backend contract (server.py /api/floors/*):
//   GET    /api/floors                  -> { ok, categories, floors[] }
//   GET    /api/floors/{floor_id}       -> per-floor bundle (== /api/map shape)
//   POST   /api/floors/copy             -> { ok, new_floor_id, ... }
//   POST   /api/floors/create (multipart) -> { ok, floor_id, report{...} }
//   DELETE /api/floors/{floor_id}       -> { ok, deleted[] }   (dev slots only)
//
// SAFETY: there is NO live-write verb here. Every copy/create lands in the
// server's DEV dir; the live install is read-only.
// =====================================================================

(function () {
  "use strict";

  if (window.__psoFloorPanelLoaded) return;
  window.__psoFloorPanelLoaded = true;

  if (!window.PSOPerspectives) {
    console.warn("[floor_panel] perspectives.js not loaded yet");
    return;
  }

  const state = {
    floors:       null,   // GET /api/floors payload
    selectedId:   null,
    showGrid:     false,
    // DOM caches (set in mount)
    _stage: null,
    _insp:  null,
    _restorers: [],
    _escSuppressor: null,
  };

  // ---- API helpers ----------------------------------------------------
  // Copied verbatim from map_panel.fetchJson: cache:"no-store", throws on
  // !r.ok with the response text so callers can surface a friendly status.
  async function fetchJson(url, opts) {
    const r = await fetch(url, opts || { cache: "no-store" });
    if (!r.ok) {
      let text = "";
      try { text = await r.text(); } catch (_e) {}
      throw new Error(`HTTP ${r.status}: ${text || url}`);
    }
    return r.json();
  }

  async function loadFloors() {
    state.floors = await fetchJson("/api/floors");
    return state.floors;
  }

  // ---- helpers --------------------------------------------------------
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;",
                '"': "&quot;", "'": "&#39;" })[c];
    });
  }

  function setStatus(msg, isErr) {
    const el = document.getElementById("floorToolbarStatus");
    if (!el) return;
    el.textContent = msg || "";
    el.classList.toggle("err", !!isErr);
  }

  function selectedFloor() {
    if (!state.floors || !state.selectedId) return null;
    return (state.floors.floors || []).find(f => f.floor_id === state.selectedId) || null;
  }

  // ---- toolbar --------------------------------------------------------
  function renderToolbar() {
    return `
      <div class="map-toolbar">
        <strong>Floor editor</strong>
        <button type="button" id="floorBtnCreate" class="ghost" title="author a new floor from a 3D model">+ Create floor from model</button>
        <span class="grow"></span>
        <button type="button" id="floorBtnResetCam" class="ghost" title="auto-fit camera to scene">camera reset</button>
        <label class="map-grid-toggle" title="toggle reference grid">
          <input type="checkbox" id="floorGridToggle"${state.showGrid ? " checked" : ""}/> grid
        </label>
        <button type="button" id="floorBtnRefresh" class="ghost" title="reload the floor list">refresh</button>
        <span id="floorToolbarStatus" class="dim"></span>
      </div>
    `;
  }

  // ---- floor list (left) ---------------------------------------------
  function renderFloorList() {
    const payload = state.floors || { categories: [], floors: [] };
    const cats = payload.categories || [];
    const floors = payload.floors || [];
    if (!floors.length) {
      return `<div class="dim" style="padding:10px">No floors found.<br/>`
           + `The floor service may not be available yet, or no dev slots exist.<br/>`
           + `Use <b>+ Create floor from model</b> to author one.</div>`;
    }
    // Group by area-category (dev first), mirroring map_panel's optgroup idiom.
    const groups = {};
    const order = [];
    for (const c of cats) { groups[c.id] = { label: c.label, items: [] }; order.push(c.id); }
    for (const f of floors) {
      const key = (f.source === "stock") ? (catKeyForArea(cats, f) || "other") : "dev";
      if (!groups[key]) { groups[key] = { label: key, items: [] }; order.push(key); }
      groups[key].items.push(f);
    }
    let html = '<div class="map-tree-title">Floors (' + floors.length + ')</div>';
    html += '<ul class="map-tree-list">';
    for (const cid of order) {
      const g = groups[cid];
      if (!g || !g.items.length) continue;
      html += `<li class="dim" style="margin-top:6px;font-size:10px;text-transform:uppercase">${escapeHtml(g.label)}</li>`;
      for (const f of g.items) {
        const sel = f.floor_id === state.selectedId ? ' selected' : '';
        const parts = (f.part_count != null) ? `${f.part_count} parts` : '';
        html += `<li class="map-tree-item floor-item${sel}" data-floor-id="${escapeHtml(f.floor_id)}" style="cursor:pointer">`
              + `<span class="map-tree-name" title="${escapeHtml(f.floor_id)}">${escapeHtml(f.label || f.floor_id)}</span>`
              + `<span class="map-tree-stat dim">${escapeHtml(f.area || '')} · ${parts}</span>`
              + `<span class="map-tag">${escapeHtml(f.source || '')}</span>`
              + `</li>`;
      }
    }
    html += '</ul>';
    return html;
  }

  // Best-effort: which category bucket does a stock floor belong to?
  function catKeyForArea(cats, f) {
    // The backend already tags stock floors with their area; the category
    // id list includes the canonical buckets. We map a few common areas;
    // unknown areas fall through to "other".
    const a = (f.area || "").toLowerCase();
    const known = {
      pioneer: "city", lobby: "city",
      forest: "forest", aancient: "forest",
      cave: "cave", acave: "cave",
      mine: "mine", machine: "mine",
      ruins: "ruins", aruins: "ruins",
      boss: "boss",
    };
    return known[a] || "other";
  }

  // ---- detail pane (right) -------------------------------------------
  function renderDetail() {
    const f = selectedFloor();
    if (!f) {
      return `<div class="dim" style="padding:10px">Select a floor on the left to preview, copy, or delete it.</div>`;
    }
    const isCopy = (f.source === "copy" || f.source === "glb");
    let html = '<div class="map-spawn-title">' + escapeHtml(f.label || f.floor_id) + '</div>';
    html += '<div class="imp-stat-grid" style="margin:6px 0">';
    html += `<span class="imp-stat-label">id</span><span class="imp-stat-val" style="grid-column:span 2">${escapeHtml(f.floor_id)}</span>`;
    html += `<span class="imp-stat-label">source</span><span class="imp-stat-val">${escapeHtml(f.source || '')}</span>`;
    html += `<span class="imp-stat-label">area</span><span class="imp-stat-val">${escapeHtml(f.area || '')}</span>`;
    html += `<span class="imp-stat-label">parts</span><span class="imp-stat-val">${f.part_count != null ? f.part_count : '—'}</span>`;
    html += '</div>';
    html += '<div class="map-spawn-actions" style="margin-top:8px">';
    html += `<button type="button" id="floorBtnPreview" class="ghost" title="render this floor in the viewport">Preview</button>`;
    html += `<button type="button" id="floorBtnCopy" class="ghost" title="duplicate into an editable dev slot">Copy</button>`;
    if (isCopy) {
      html += `<button type="button" id="floorBtnDelete" class="ghost" title="delete this dev slot">Delete</button>`;
    }
    html += '</div>';
    html += `<div id="floorReport" class="dim" style="margin-top:8px;font-size:11px"></div>`;
    return html;
  }

  // ---- preview (single code path) ------------------------------------
  async function previewFloor(floorId) {
    if (!floorId) return;
    setStatus(`loading ${floorId}…`);
    try {
      window.psoSceneClearMap && window.psoSceneClearMap();
      const bundle = await fetchJson(`/api/floors/${encodeURIComponent(floorId)}`);
      const loader = window.psoSceneLoadMapWithEnvironment || window.psoSceneLoadMap;
      const result = loader ? await loader(bundle) : { loaded_count: 0, failed_count: 0 };
      window.psoSceneToggleGrid && window.psoSceneToggleGrid(state.showGrid);
      window.psoSceneResetCamera && window.psoSceneResetCamera("auto");
      const fail = result.failed_count ? ` (${result.failed_count} failed)` : "";
      setStatus(`${result.loaded_count || 0} parts loaded${fail}`);
      // Surface the fidelity banners if present.
      const notes = [];
      if (bundle.root_only_preview) {
        notes.push("preview shows root nodes only; child sub-meshes are not rendered (copy preserves source bytes).");
      }
      if (bundle.single_texture_slot) {
        notes.push("all submeshes share one texture slot (build_lobby limitation).");
      }
      const rep = document.getElementById("floorReport");
      if (rep && notes.length) rep.innerHTML = notes.map(n => `<div class="imp-warn">${escapeHtml(n)}</div>`).join("");
    } catch (e) {
      console.error("[floor_panel] preview failed:", e);
      setStatus("preview failed: " + (e && e.message || e), true);
    }
  }

  // ---- copy ----------------------------------------------------------
  async function copyFloor(floorId) {
    if (!floorId) return;
    setStatus(`copying ${floorId}…`);
    try {
      const r = await fetchJson("/api/floors/copy", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ floor_id: floorId, mode: "passthrough" }),
      });
      await loadFloors();
      state.selectedId = r.new_floor_id;
      reRender();
      setStatus(`copied to ${r.new_floor_id}`);
      await previewFloor(r.new_floor_id);
    } catch (e) {
      setStatus("copy failed: " + (e && e.message || e), true);
    }
  }

  // ---- delete --------------------------------------------------------
  async function deleteFloor(floorId) {
    if (!floorId) return;
    if (!window.confirm(`Delete dev slot ${floorId}? This cannot be undone.`)) return;
    setStatus(`deleting ${floorId}…`);
    try {
      const r = await fetch(`/api/floors/${encodeURIComponent(floorId)}`, { method: "DELETE" });
      if (!r.ok) { throw new Error(await r.text()); }
      if (state.selectedId === floorId) state.selectedId = null;
      window.psoSceneClearMap && window.psoSceneClearMap();
      await loadFloors();
      reRender();
      setStatus(`deleted ${floorId}`);
    } catch (e) {
      setStatus("delete failed: " + (e && e.message || e), true);
    }
  }

  // ---- create modal --------------------------------------------------
  function openCreateModal() {
    closeCreateModal();
    const back = document.createElement("div");
    back.className = "imp-modal-backdrop";
    back.id = "floorCreateBackdrop";
    const models = manifestModelEntries();
    let modelOpts = '<option value="">(none — upload a file instead)</option>';
    for (const m of models) {
      modelOpts += `<option value="${escapeHtml(m.path)}">${escapeHtml(m.path)}</option>`;
    }
    back.innerHTML = `
      <div class="imp-modal" role="dialog" aria-label="Create floor from model">
        <div class="imp-modal-head">
          <span class="imp-modal-title">Create floor from model</span>
          <button type="button" class="imp-modal-x" id="floorCreateX">×</button>
        </div>
        <div class="imp-modal-body">
          <div class="imp-section">
            <div class="imp-section-title">Source</div>
            <div class="imp-form-row">
              <label>Upload model</label>
              <input type="file" id="floorCreateFile" accept=".glb,.gltf,.obj,.fbx"/>
              <span></span>
            </div>
            <div class="imp-form-row">
              <label>or imported asset</label>
              <select id="floorCreateSource">${modelOpts}</select>
              <span></span>
            </div>
          </div>
          <div class="imp-section">
            <div class="imp-section-title">Floor</div>
            <div class="imp-form-row">
              <label>name</label>
              <input type="text" id="floorCreateName" placeholder="myfloor" />
              <span class="dim">letters/digits/_/-</span>
            </div>
            <div class="imp-form-row">
              <label>area template</label>
              <select id="floorCreateArea">
                <option value="forest">forest</option>
                <option value="cave">cave</option>
                <option value="mine">mine</option>
                <option value="ruins">ruins</option>
                <option value="city">city</option>
                <option value="other" selected>other</option>
              </select>
              <span class="dim">lighting / fog hint</span>
            </div>
          </div>
          <div id="floorCreateReport"></div>
        </div>
        <div class="imp-actions">
          <span class="imp-status" id="floorCreateStatus"></span>
          <button type="button" class="imp-btn" id="floorCreateCancel">Cancel</button>
          <button type="button" class="imp-btn primary" id="floorCreateGo">Create</button>
        </div>
      </div>
    `;
    document.body.appendChild(back);
    document.getElementById("floorCreateX").addEventListener("click", closeCreateModal);
    document.getElementById("floorCreateCancel").addEventListener("click", closeCreateModal);
    document.getElementById("floorCreateGo").addEventListener("click", submitCreate);
    back.addEventListener("click", function (e) { if (e.target === back) closeCreateModal(); });
  }

  function closeCreateModal() {
    const b = document.getElementById("floorCreateBackdrop");
    if (b && b.parentNode) b.parentNode.removeChild(b);
  }

  function manifestModelEntries() {
    const out = [];
    try {
      if (window.PSOManifest && typeof window.PSOManifest.entries === "function") {
        for (const e of window.PSOManifest.entries()) {
          if (e && e.category === "model" && e.path) out.push(e);
        }
      }
    } catch (_e) {}
    return out.slice(0, 500);
  }

  function setCreateStatus(msg, cls) {
    const el = document.getElementById("floorCreateStatus");
    if (!el) return;
    el.textContent = msg || "";
    el.className = "imp-status" + (cls ? " " + cls : "");
  }

  async function submitCreate() {
    const fileEl = document.getElementById("floorCreateFile");
    const srcEl = document.getElementById("floorCreateSource");
    const nameEl = document.getElementById("floorCreateName");
    const areaEl = document.getElementById("floorCreateArea");
    const name = (nameEl && nameEl.value || "").trim();
    if (!name) { setCreateStatus("enter a floor name", "err"); return; }
    const file = fileEl && fileEl.files && fileEl.files[0];
    const sourcePath = srcEl && srcEl.value;
    if (!file && !sourcePath) { setCreateStatus("choose a file or an imported asset", "err"); return; }

    const fd = new FormData();
    if (file) fd.append("file", file, file.name);
    if (sourcePath) fd.append("source_path", sourcePath);
    fd.append("name", name);
    fd.append("area_template", (areaEl && areaEl.value) || "other");

    setCreateStatus("building floor…", "busy");
    const goBtn = document.getElementById("floorCreateGo");
    if (goBtn) goBtn.disabled = true;
    try {
      const r = await fetch("/api/floors/create", { method: "POST", body: fd });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) {
        const detail = (j && j.detail) || r.statusText;
        setCreateStatus("create failed: " + detail, "err");
        renderCreateReport(j && j.report, true);
        return;
      }
      setCreateStatus("created " + j.floor_id, "ok");
      renderCreateReport(j.report, false, j.floor_id);
      await loadFloors();
      state.selectedId = j.floor_id;
      reRender();
    } catch (e) {
      setCreateStatus("create failed: " + (e && e.message || e), "err");
    } finally {
      if (goBtn) goBtn.disabled = false;
    }
  }

  function renderCreateReport(report, isErr, floorId) {
    const host = document.getElementById("floorCreateReport");
    if (!host) return;
    if (!report) { host.innerHTML = ""; return; }
    let html = '<div class="imp-section"><div class="imp-section-title">Verify report</div>';
    html += '<div class="imp-stat-grid">';
    const rows = [
      ["parts", report.part_count], ["vertices", report.vertex_count],
      ["textures", report.texture_count], ["tris in", report.tri_in],
      ["tris out", report.tri_out], ["dropped child nodes", report.dropped_child_nodes],
    ];
    for (const [k, v] of rows) {
      if (v == null) continue;
      html += `<span class="imp-stat-label">${escapeHtml(k)}</span><span class="imp-stat-val">${escapeHtml(String(v))}</span>`;
    }
    html += '</div>';
    if (report.single_texture_slot) {
      html += `<div class="imp-warn">all submeshes share one texture slot (build_lobby limitation).</div>`;
    }
    for (const w of (report.warnings || [])) {
      html += `<div class="imp-warn">${escapeHtml(w)}</div>`;
    }
    for (const er of (report.errors || [])) {
      html += `<div class="imp-warn" style="border-left-color:#ff6680;color:#ff6680">${escapeHtml(er)}</div>`;
    }
    for (const fl of (report.files || [])) {
      html += `<div class="dim" style="font-family:monospace;font-size:10px">${escapeHtml(fl.name)} — ${fl.size} bytes${fl.budget_ok === false ? ' (OVER BUDGET)' : ''}</div>`;
    }
    html += '</div>';
    if (!isErr && floorId) {
      html += `<button type="button" class="imp-btn primary" id="floorCreatePreview">Preview result</button>`;
    }
    host.innerHTML = html;
    const pv = document.getElementById("floorCreatePreview");
    if (pv) pv.addEventListener("click", function () {
      closeCreateModal();
      previewFloor(floorId);
    });
  }

  // ---- rerender / rebind ---------------------------------------------
  function reRender() {
    if (!state._stage) return;
    const tb = state._stage.querySelector("#floorToolbar");
    if (tb) tb.innerHTML = renderToolbar();
    const list = state._stage.querySelector("#floorList");
    if (list) list.innerHTML = renderFloorList();
    const detail = state._stage.querySelector("#floorDetail");
    if (detail) detail.innerHTML = renderDetail();
    rebindAfterRender();
  }

  function rebindAfterRender() {
    const stage = state._stage;
    if (!stage) return;
    const $ = (sel) => stage.querySelector(sel);

    $("#floorBtnCreate") && $("#floorBtnCreate").addEventListener("click", openCreateModal);
    $("#floorBtnRefresh") && $("#floorBtnRefresh").addEventListener("click", async function () {
      try { await loadFloors(); reRender(); setStatus("list refreshed"); }
      catch (e) { setStatus("refresh failed: " + (e && e.message || e), true); }
    });
    $("#floorBtnResetCam") && $("#floorBtnResetCam").addEventListener("click", function () {
      window.psoSceneResetCamera && window.psoSceneResetCamera("auto");
    });
    const grid = $("#floorGridToggle");
    if (grid) grid.addEventListener("change", function () {
      state.showGrid = grid.checked;
      window.psoSceneToggleGrid && window.psoSceneToggleGrid(state.showGrid);
    });

    // Floor list rows
    stage.querySelectorAll(".floor-item").forEach(function (row) {
      row.addEventListener("click", function () {
        state.selectedId = row.dataset.floorId;
        const detail = stage.querySelector("#floorDetail");
        if (detail) detail.innerHTML = renderDetail();
        const list = stage.querySelector("#floorList");
        if (list) list.querySelectorAll(".floor-item").forEach(r => r.classList.remove("selected"));
        row.classList.add("selected");
        rebindDetail();
      });
    });
    rebindDetail();
  }

  function rebindDetail() {
    const stage = state._stage;
    if (!stage) return;
    const $ = (sel) => stage.querySelector(sel);
    $("#floorBtnPreview") && $("#floorBtnPreview").addEventListener("click", function () {
      previewFloor(state.selectedId);
    });
    $("#floorBtnCopy") && $("#floorBtnCopy").addEventListener("click", function () {
      copyFloor(state.selectedId);
    });
    $("#floorBtnDelete") && $("#floorBtnDelete").addEventListener("click", function () {
      deleteFloor(state.selectedId);
    });
  }

  // ---- inspector -----------------------------------------------------
  function renderInspector() {
    const insp = state._insp;
    if (!insp) return;
    let html = '<div class="vp-insp-title">Floor editor</div>';
    html += '<div class="vp-insp-help dim">Browse floors on the left. <b>Preview</b> renders one in the viewport; ';
    html += '<b>Copy</b> duplicates it into an editable dev slot; <b>+ Create floor from model</b> authors a new floor ';
    html += 'from a GLB. All edits land in the dev data dir — the live game install is never touched.</div>';
    insp.innerHTML = html;
  }

  // ---- perspective registration -------------------------------------
  window.PSOPerspectives.register("floor-editor", {
    label: "Floor editor",
    match: function (entry, file) {
      // Strictly BELOW map-editor's 90 so the Map editor stays the default
      // owner of raw map assets. The Floor editor is reached via its header
      // button + the Floors pill.
      if (entry && entry.category === "map") return 80;
      const fn = (file || "").toLowerCase();
      if (fn.startsWith("scene/map_")) return 60;
      return 0;
    },
    mount: async function (stage, insp, ctx) {
      state._stage = stage;
      state._insp  = insp;

      // Suppress Esc — tabs are the only exit in unified mode (copied from
      // map_panel mount).
      const esc = function (e) {
        if (e.key !== "Escape") return;
        const t = e.target;
        const tag = t && t.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        e.stopPropagation();
        e.preventDefault();
      };
      document.addEventListener("keydown", esc, true);
      state._escSuppressor = esc;

      stage.innerHTML = `
        <div class="map-perspective floor-perspective">
          <div id="floorToolbar"></div>
          <div class="map-body floor-body">
            <aside id="floorList" class="map-side map-side-left floor-side floor-side-left"></aside>
            <main id="floorViewport" class="map-viewport floor-viewport"></main>
            <aside id="floorDetail" class="map-side map-side-right floor-side floor-side-right"></aside>
          </div>
          <div class="map-footer"><span id="floorFooter" class="dim">floor editor — dev-only, live install never touched</span></div>
        </div>
      `;

      // Pluck the model viewer's canvas + bar into the viewport host.
      // COPIED LITERALLY from map_panel.js mount (do NOT simplify — the
      // nextSibling/parentNode bookkeeping is what restores the canvas to
      // #modelModal on unmount; skip it and the 3D area orphans black).
      const restorers = [];
      const bar = document.querySelector("#modelModal .model-bar");
      const mstage = document.querySelector("#modelModal .model-stage");
      const homeBar = bar ? bar.parentNode : null;
      const homeStage = mstage ? mstage.parentNode : null;
      const nextBar = bar ? bar.nextSibling : null;
      const nextStage = mstage ? mstage.nextSibling : null;
      const vpHost = stage.querySelector("#floorViewport");
      if (bar) {
        bar.style.display = "none";
      }
      if (mstage) vpHost.appendChild(mstage);
      restorers.push(function () {
        if (bar) bar.style.display = "";
        if (homeBar && bar) {
          if (nextBar && nextBar.parentNode === homeBar) homeBar.insertBefore(bar, nextBar);
          else homeBar.appendChild(bar);
        }
        if (homeStage && mstage) {
          if (nextStage && nextStage.parentNode === homeStage) homeStage.insertBefore(mstage, nextStage);
          else homeStage.appendChild(mstage);
        }
      });
      stage._floorRestorers = restorers;

      // Force the renderer to grab its new parent's size.
      setTimeout(function () {
        if (typeof window.psoModelRebindResize === "function") {
          window.psoModelRebindResize();
        }
        window.dispatchEvent(new Event("resize"));
      }, 80);

      // Build the panel.
      const tb = stage.querySelector("#floorToolbar");
      if (tb) tb.innerHTML = renderToolbar();
      renderInspector();

      // Initial load — degrade gracefully if the floor service 404s.
      try {
        await loadFloors();
        // Pre-select the first dev slot, else the first floor.
        const floors = (state.floors && state.floors.floors) || [];
        const firstDev = floors.find(f => f.source === "copy" || f.source === "glb");
        state.selectedId = (firstDev && firstDev.floor_id) || (floors[0] && floors[0].floor_id) || null;
        reRender();
        if (ctx && ctx.fileName && /^map_/i.test(ctx.fileName)) {
          // A floor/map leaf routed us here; nothing else to do — the user
          // picks from the list.
        }
      } catch (e) {
        console.error("[floor_panel] mount-load failed:", e);
        reRender();  // still render the empty-state placeholder
        setStatus("floor service not available: " + (e && e.message || e), true);
      }
    },
    unmount: function (stage, insp) {
      if (state._escSuppressor) {
        document.removeEventListener("keydown", state._escSuppressor, true);
        state._escSuppressor = null;
      }
      closeCreateModal();
      window.psoSceneClearMap && window.psoSceneClearMap();
      window.psoSceneResetEnvironment && window.psoSceneResetEnvironment();
      try {
        if (stage._floorRestorers) {
          stage._floorRestorers.forEach(function (f) { try { f(); } catch (_e) {} });
          stage._floorRestorers = null;
        }
      } catch (_e) {}
      state._stage = null;
      state._insp  = null;
    },
  });

  // ---- header button -------------------------------------------------
  function openPerspective() {
    const ctx = {
      path: "__floor_editor__",
      entry: { category: "map", format: "FloorScene" },
      fileName: "Floor editor",
    };
    if (window.PSOPerspectives && window.PSOPerspectives.switchTo) {
      window.PSOPerspectives.switchTo("floor-editor", ctx);
    }
  }

  function ensureHeaderButton() {
    if (document.getElementById("btnFloorEditor")) return;
    const status = document.getElementById("status");
    const header = status ? status.parentNode : null;
    if (!header) return;
    const btn = document.createElement("button");
    btn.id = "btnFloorEditor";
    btn.type = "button";
    btn.className = "ghost";
    btn.title = "Floor Editor — browse, copy, and create floors (dev-only)";
    btn.textContent = "Floor Editor";
    header.insertBefore(btn, status);
    btn.addEventListener("click", openPerspective);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", ensureHeaderButton);
  } else {
    ensureHeaderButton();
  }
})();
