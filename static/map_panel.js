// =====================================================================
// PSOBB Texture Editor - Map Editor perspective (2026-04-25)
//
// Multi-object scene viewport. Loads every NJ/XJ for one (map, floor)
// tuple in parallel and lets the user place spawns + waypoints, all
// against the same persistent Three.js scene model_viewer.js owns.
//
// Layout:
//   stage:
//     <toolbar>: map picker, floor picker, reset-camera, grid toggle
//     <body>: [scene-tree sidebar] [3d viewport] [spawn/waypoint sidebar]
//     <footer>: save/load buttons + coord readout
//   inspector: contextual help + per-spawn detail editor
//
// The stage relocates the model viewer's <canvas> into our 3d-viewport
// host, just like the existing 3d-view perspective does. On unmount we
// clear scene-mode state and put the canvas back where it belongs.
//
// Spawn / waypoint state lives in module-scope `state.edits` so
// switching tabs doesn't lose unsaved work. Save flushes to
// /api/map/edits; Load (or fresh map-pick) pulls from the same.
// =====================================================================

(function () {
  "use strict";

  if (!window.PSOPerspectives) {
    console.warn("[map_panel] perspectives.js not loaded yet");
    return;
  }

  const state = {
    catalogue:   null,        // GET /api/map/list payload
    activeMapId: null,
    activeFloor: 0,
    bundle:      null,        // GET /api/map/<id>?floor=N payload
    edits: {                  // matches scene_loader sidecar shape
      spawns: [],
      waypoints: [],
    },
    nextSpawnId: 1,           // monotonic id allocator
    selectedSpawnId: null,
    placeMode: false,         // when true, next viewport click drops a spawn
    placeType: "mob",
    connectMode: false,       // when true, two clicks in spawn list pair
    connectFirstId: null,
    showGrid: false,
    // v4 visual polish: opt-in to the bit-exact PSOBB Lambert shader
    // (psobb_lambert_shader.js). Default off to preserve the existing
    // MeshLambertMaterial behaviour for users who don't want the
    // slightly-darker / slightly-redder PSOBB-accurate look.
    exactLambert: false,
    // DOM caches (set in mount)
    _stage: null,
    _insp:  null,
    _coordReadout: null,
    _restorers: [],
    _escSuppressor: null,
    // Drag state for marker repositioning
    _dragMarkerId: null,
  };

  // ---- API helpers ----------------------------------------------------
  async function fetchJson(url, opts) {
    const r = await fetch(url, opts || { cache: "no-store" });
    if (!r.ok) {
      let text = "";
      try { text = await r.text(); } catch (_e) {}
      throw new Error(`HTTP ${r.status}: ${text || url}`);
    }
    return r.json();
  }

  async function loadCatalogue() {
    if (state.catalogue) return state.catalogue;
    state.catalogue = await fetchJson("/api/map/list");
    return state.catalogue;
  }

  async function loadMapBundle(mapId, floor) {
    return fetchJson(`/api/map/${encodeURIComponent(mapId)}?floor=${floor | 0}`);
  }

  async function loadEditsFromServer(mapId) {
    return fetchJson(`/api/map/edits/${encodeURIComponent(mapId)}`);
  }

  async function saveEditsToServer(mapId, edits) {
    return fetchJson("/api/map/edits", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        map_id:    mapId,
        spawns:    edits.spawns,
        waypoints: edits.waypoints,
      }),
    });
  }

  // ---- helpers --------------------------------------------------------
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;",
                '"': "&quot;", "'": "&#39;" })[c];
    });
  }

  function fmtCoord(v) {
    if (typeof v !== "number") return "—";
    return v.toFixed(1);
  }

  // ---- toolbar / picker ----------------------------------------------
  function renderToolbar() {
    const cat = state.catalogue || { categories: [], maps: [] };
    // Build optgroups by category
    const groups = {};
    for (const c of cat.categories) groups[c.id] = { label: c.label, maps: [] };
    for (const m of cat.maps) {
      if (!groups[m.category]) {
        groups[m.category] = { label: m.category, maps: [] };
      }
      groups[m.category].maps.push(m);
    }
    let mapsHtml = "";
    for (const cid of Object.keys(groups)) {
      const g = groups[cid];
      if (!g.maps.length) continue;
      mapsHtml += `<optgroup label="${escapeHtml(g.label)}">`;
      for (const m of g.maps) {
        const ren = m.renderable_files;
        const dim = ren === 0 ? " (no terrain)" : "";
        mapsHtml += `<option value="${escapeHtml(m.map_id)}"`
                 + (m.map_id === state.activeMapId ? " selected" : "")
                 + `>${escapeHtml(m.map_id)} - ${escapeHtml(m.label)}${dim}</option>`;
      }
      mapsHtml += `</optgroup>`;
    }
    // Floor picker depends on the chosen map
    const activeMap = (cat.maps || []).find(m => m.map_id === state.activeMapId);
    const floors = activeMap ? activeMap.floors : [];
    const floorsHtml = floors.length
      ? floors.map(f => `<option value="${f}"${f === state.activeFloor ? " selected" : ""}>floor ${f}</option>`).join("")
      : `<option value="0">no floors</option>`;

    return `
      <div class="map-toolbar">
        <label>map:
          <select id="mapPicker">${mapsHtml || '<option value="">(no maps)</option>'}</select>
        </label>
        <label>floor:
          <select id="mapFloorPicker">${floorsHtml}</select>
        </label>
        <span class="grow"></span>
        <button type="button" id="mapBtnResetCam" class="ghost" title="auto-fit camera to scene">camera reset</button>
        <button type="button" id="mapBtnTopdown"  class="ghost" title="top-down view">top-down</button>
        <button type="button" id="mapBtnFp"       class="ghost" title="first-person view (rough)">first-person</button>
        <label class="map-grid-toggle" title="toggle reference grid">
          <input type="checkbox" id="mapGridToggle"${state.showGrid ? " checked" : ""}/> grid
        </label>
        <label class="map-exact-lambert" title="swap to a custom GLSL shader that bit-exactly matches PSOBB's per-vertex Lambert lighting (hemisphere ambient + 1 directional + multiplicative fog)">
          <input type="checkbox" id="mapExactLambertToggle"${state.exactLambert ? " checked" : ""}/> Exact PSOBB shader
        </label>
        <span id="mapToolbarStatus" class="dim"></span>
      </div>
    `;
  }

  function renderSceneTree() {
    if (!window.psoSceneListLoaded) return "<div class='dim'>scene viewer not loaded</div>";
    const parts = window.psoSceneListLoaded() || [];
    if (!parts.length) {
      return `<div class="dim">no scene parts loaded yet</div>`;
    }
    let html = '<div class="map-tree-title">Scene parts (' + parts.length + ')</div>';
    html += '<ul class="map-tree-list">';
    for (const p of parts) {
      const tag = p.isProp ? '<span class="map-tag">prop</span>' : '';
      html += `<li class="map-tree-item">`
            + `<label class="map-tree-vis"><input type="checkbox" data-path="${escapeHtml(p.path)}"${p.visible ? ' checked' : ''}/></label>`
            + `<span class="map-tree-name" title="${escapeHtml(p.path)}">${escapeHtml(p.path.split('/').pop() || '?')}</span>`
            + tag
            + `<span class="map-tree-stat dim">${(p.vertices || 0).toLocaleString()}v / ${(p.triangles || 0).toLocaleString()}t</span>`
            + `</li>`;
    }
    html += '</ul>';
    html += '<div class="map-tree-actions">';
    html += '<button type="button" id="mapBtnAddProp" class="ghost" title="add a free-standing model into the scene">+ Add prop</button>';
    html += '</div>';
    return html;
  }

  function renderSpawnList() {
    const spawns = state.edits.spawns || [];
    let html = '<div class="map-spawn-title">Spawns (' + spawns.length + ')</div>';
    html += '<div class="map-spawn-actions">';
    html += `<button type="button" id="mapBtnPlaceMob"   class="ghost${state.placeMode && state.placeType==='mob'?' active':''}" title="next click drops a mob spawn">+ Mob</button>`;
    html += `<button type="button" id="mapBtnPlaceNpc"   class="ghost${state.placeMode && state.placeType==='npc'?' active':''}">+ NPC</button>`;
    html += `<button type="button" id="mapBtnPlaceChest" class="ghost${state.placeMode && state.placeType==='chest'?' active':''}">+ Chest</button>`;
    html += `<button type="button" id="mapBtnPlaceSwitch" class="ghost${state.placeMode && state.placeType==='switch'?' active':''}">+ Switch</button>`;
    html += `<button type="button" id="mapBtnPlaceTele" class="ghost${state.placeMode && state.placeType==='teleport'?' active':''}">+ Teleport</button>`;
    html += `<button type="button" id="mapBtnConnect"   class="ghost${state.connectMode?' active':''}" title="next 2 spawn clicks pair them">~ Connect</button>`;
    html += '</div>';
    if (!spawns.length) {
      html += '<div class="dim map-spawn-empty">no spawns. Pick a type then click in the viewport.</div>';
    } else {
      html += '<ul class="map-spawn-list">';
      for (const sp of spawns) {
        const sel = sp.id === state.selectedSpawnId ? ' selected' : '';
        html += `<li class="map-spawn-row map-spawn-${escapeHtml(sp.type)}${sel}" data-spawn-id="${sp.id}">`;
        html += `<span class="map-spawn-pip"></span>`;
        html += `<span class="map-spawn-id">#${sp.id}</span>`;
        html += `<span class="map-spawn-type">${escapeHtml(sp.type)}</span>`;
        html += `<span class="map-spawn-coord dim">(${fmtCoord(sp.world_pos[0])}, ${fmtCoord(sp.world_pos[1])}, ${fmtCoord(sp.world_pos[2])})</span>`;
        html += `<button type="button" class="ghost map-spawn-del" data-spawn-id="${sp.id}" title="delete spawn">x</button>`;
        html += `</li>`;
      }
      html += '</ul>';
    }
    // Waypoints
    const wp = state.edits.waypoints || [];
    html += '<div class="map-wp-title">Waypoints (' + wp.length + ')</div>';
    if (!wp.length) {
      html += '<div class="dim map-wp-empty">click "Connect" then 2 spawns to add a path.</div>';
    } else {
      html += '<ul class="map-wp-list">';
      for (let i = 0; i < wp.length; i++) {
        const w = wp[i];
        html += `<li class="map-wp-row" data-wp-i="${i}">`;
        html += `<span class="map-wp-pair">#${w.from_id} &rarr; #${w.to_id}</span>`;
        html += `<select class="map-wp-style" data-wp-i="${i}">`;
        for (const s of ["walk", "run", "teleport"]) {
          html += `<option value="${s}"${s === w.style ? ' selected' : ''}>${s}</option>`;
        }
        html += `</select>`;
        html += `<input type="number" class="map-wp-speed" data-wp-i="${i}" step="0.1" min="0" value="${w.speed}" title="patrol speed"/>`;
        html += `<button type="button" class="ghost map-wp-del" data-wp-i="${i}" title="delete">x</button>`;
        html += `</li>`;
      }
      html += '</ul>';
    }
    return html;
  }

  function renderFooter() {
    return `
      <div class="map-footer">
        <button type="button" id="mapBtnSave" title="save spawns + waypoints to cache/map_edits/">save edits</button>
        <button type="button" id="mapBtnLoad" class="ghost" title="reload from server (discards in-memory changes)">reload edits</button>
        <button type="button" id="mapBtnExportJson" class="ghost" title="download as JSON">export JSON</button>
        <span class="grow"></span>
        <span id="mapCoordReadout" class="dim">cam: — / click: —</span>
      </div>
    `;
  }

  // ---- coordinate readout -------------------------------------------
  function updateCoordReadout(extraLabel, extraVal) {
    if (!state._coordReadout) return;
    let camStr = "—";
    try {
      const cam = window.psoGetCamera && window.psoGetCamera();
      if (cam) camStr = `(${cam.position.x.toFixed(1)}, ${cam.position.y.toFixed(1)}, ${cam.position.z.toFixed(1)})`;
    } catch (_e) {}
    let txt = `cam: ${camStr}`;
    if (extraLabel) txt += ` / ${extraLabel}: ${extraVal || '—'}`;
    state._coordReadout.textContent = txt;
  }

  // ---- main load flow ------------------------------------------------
  async function setActiveMap(mapId, floor) {
    if (!mapId) return;
    state.activeMapId = mapId;
    state.activeFloor = floor | 0;
    setStatus(`loading ${mapId}/floor ${floor}…`);
    try {
      // Parallel: bundle + edits
      const [bundle, editsResp] = await Promise.all([
        loadMapBundle(mapId, floor),
        loadEditsFromServer(mapId),
      ]);
      state.bundle = bundle;
      // Re-pick active floor if server defaulted
      if (typeof bundle.floor === "number") state.activeFloor = bundle.floor;
      // Load scene + apply per-area environment (fog, lighting, lambert
      // materials). Falls through to the bare loader if the environment
      // helper isn't loaded — the page still renders, it's just brighter
      // and flatter.
      const t0 = performance.now();
      const loader = window.psoSceneLoadMapWithEnvironment
                  || window.psoSceneLoadMap;
      const result = loader
        ? await loader(bundle)
        : { loaded_count: 0, failed_count: 0 };
      const dt = (performance.now() - t0) | 0;
      // Reset edit state from sidecar
      state.edits.spawns    = (editsResp && editsResp.spawns)    || [];
      state.edits.waypoints = (editsResp && editsResp.waypoints) || [];
      state.nextSpawnId = state.edits.spawns.reduce((a, sp) => Math.max(a, sp.id + 1), 1);
      state.selectedSpawnId = null;
      // Re-add markers + connectors for loaded edits
      window.psoSceneClearMarkers && window.psoSceneClearMarkers();
      window.psoSceneClearConnectors && window.psoSceneClearConnectors();
      for (const sp of state.edits.spawns) {
        window.psoSceneAddMarker && window.psoSceneAddMarker(sp.id, sp.world_pos, sp.type);
      }
      const byId = new Map(state.edits.spawns.map(sp => [sp.id, sp]));
      for (const w of state.edits.waypoints) {
        const a = byId.get(w.from_id);
        const b = byId.get(w.to_id);
        if (!a || !b) continue;
        window.psoSceneSetConnector && window.psoSceneSetConnector(
          `${w.from_id}:${w.to_id}`, a.world_pos, b.world_pos, w.style,
        );
      }
      // Apply grid setting
      window.psoSceneToggleGrid && window.psoSceneToggleGrid(state.showGrid);

      const failNote = result.failed_count
        ? ` (${result.failed_count} parts failed)` : "";
      setStatus(`${mapId}/${state.activeFloor}: ${result.loaded_count} parts loaded${failNote} in ${dt} ms`);
      reRender();
    } catch (e) {
      console.error("[map_panel] load failed:", e);
      setStatus("load failed: " + (e && e.message || e), true);
    }
  }

  function setStatus(msg, isErr) {
    const el = document.getElementById("mapToolbarStatus");
    if (!el) return;
    el.textContent = msg || "";
    el.classList.toggle("err", !!isErr);
  }

  // ---- rerender ------------------------------------------------------
  function reRender() {
    if (!state._stage) return;
    // Toolbar
    const tb = state._stage.querySelector("#mapToolbar");
    if (tb) tb.innerHTML = renderToolbar();
    // Sidebars
    const tree = state._stage.querySelector("#mapSceneTree");
    if (tree) tree.innerHTML = renderSceneTree();
    const sl = state._stage.querySelector("#mapSpawnSide");
    if (sl) sl.innerHTML = renderSpawnList();
    // Footer / coord readout reference
    state._coordReadout = state._stage.querySelector("#mapCoordReadout");
    updateCoordReadout();
    rebindAfterRender();
  }

  function rebindAfterRender() {
    const stage = state._stage;
    if (!stage) return;
    const $ = (sel) => stage.querySelector(sel);

    // Toolbar
    const mp = $("#mapPicker");
    if (mp) mp.addEventListener("change", function () {
      setActiveMap(mp.value, 0);
    });
    const fp = $("#mapFloorPicker");
    if (fp) fp.addEventListener("change", function () {
      setActiveMap(state.activeMapId, parseInt(fp.value, 10) | 0);
    });
    $("#mapBtnResetCam") && $("#mapBtnResetCam").addEventListener("click", function () {
      window.psoSceneResetCamera && window.psoSceneResetCamera("auto");
    });
    $("#mapBtnTopdown") && $("#mapBtnTopdown").addEventListener("click", function () {
      window.psoSceneResetCamera && window.psoSceneResetCamera("topdown");
    });
    $("#mapBtnFp") && $("#mapBtnFp").addEventListener("click", function () {
      window.psoSceneResetCamera && window.psoSceneResetCamera("first-person");
    });
    const grid = $("#mapGridToggle");
    if (grid) grid.addEventListener("change", function () {
      state.showGrid = grid.checked;
      window.psoSceneToggleGrid && window.psoSceneToggleGrid(state.showGrid);
    });
    // v4 visual polish: Exact-PSOBB-Lambert toggle. Calls into the
    // additive global on model_viewer.js. Failure is silent (the
    // checkbox state still flips, but no shader swap happens) so a
    // map-editor user can't get stuck in a half-applied state.
    const exact = $("#mapExactLambertToggle");
    if (exact) exact.addEventListener("change", async function () {
      state.exactLambert = exact.checked;
      if (typeof window.psoSceneUseExactLambert === "function") {
        try {
          const result = await window.psoSceneUseExactLambert(state.exactLambert);
          // psoSceneUseExactLambert returns the actual state — sync the
          // checkbox if it differed (e.g. the shader module failed to load).
          if (result !== state.exactLambert) {
            state.exactLambert = result;
            exact.checked = result;
          }
        } catch (e) {
          console.warn("[map_panel] psoSceneUseExactLambert failed:", e);
        }
      }
    });

    // Scene tree visibility toggles
    stage.querySelectorAll(".map-tree-vis input[data-path]").forEach(function (cb) {
      cb.addEventListener("change", function () {
        window.psoSceneSetPartVisible && window.psoSceneSetPartVisible(cb.dataset.path, cb.checked);
      });
    });
    $("#mapBtnAddProp") && $("#mapBtnAddProp").addEventListener("click", openPropPicker);

    // Spawn placement buttons
    function bindPlace(btnId, type) {
      const b = $(btnId);
      if (!b) return;
      b.addEventListener("click", function () {
        if (state.placeMode && state.placeType === type) {
          state.placeMode = false;
        } else {
          state.placeMode = true;
          state.placeType = type;
          state.connectMode = false;
        }
        reRender();
      });
    }
    bindPlace("#mapBtnPlaceMob",    "mob");
    bindPlace("#mapBtnPlaceNpc",    "npc");
    bindPlace("#mapBtnPlaceChest",  "chest");
    bindPlace("#mapBtnPlaceSwitch", "switch");
    bindPlace("#mapBtnPlaceTele",   "teleport");
    $("#mapBtnConnect") && $("#mapBtnConnect").addEventListener("click", function () {
      state.connectMode = !state.connectMode;
      state.placeMode = false;
      state.connectFirstId = null;
      reRender();
    });

    // Spawn rows
    stage.querySelectorAll(".map-spawn-row").forEach(function (row) {
      const id = parseInt(row.dataset.spawnId, 10);
      row.addEventListener("click", function (e) {
        if (e.target.classList.contains("map-spawn-del")) return;
        if (state.connectMode) {
          if (state.connectFirstId == null) {
            state.connectFirstId = id;
            setStatus("connect: pick a 2nd spawn (or click ~Connect to cancel)");
          } else if (state.connectFirstId === id) {
            state.connectFirstId = null;
            setStatus("connect: pick a 2nd spawn");
          } else {
            const a = state.connectFirstId;
            const b = id;
            addWaypoint(a, b);
            state.connectFirstId = null;
            state.connectMode = false;
            reRender();
          }
        } else {
          state.selectedSpawnId = id;
          renderInspector();
          // Highlight selected spawn row
          stage.querySelectorAll(".map-spawn-row").forEach(r => r.classList.remove("selected"));
          row.classList.add("selected");
        }
      });
    });
    stage.querySelectorAll(".map-spawn-del").forEach(function (btn) {
      btn.addEventListener("click", function (e) {
        e.stopPropagation();
        const id = parseInt(btn.dataset.spawnId, 10);
        deleteSpawn(id);
      });
    });
    stage.querySelectorAll(".map-wp-del").forEach(function (btn) {
      btn.addEventListener("click", function () {
        const i = parseInt(btn.dataset.wpI, 10);
        deleteWaypoint(i);
      });
    });
    stage.querySelectorAll(".map-wp-style").forEach(function (sel) {
      sel.addEventListener("change", function () {
        const i = parseInt(sel.dataset.wpI, 10);
        const w = state.edits.waypoints[i];
        if (!w) return;
        w.style = sel.value;
        // re-render line
        const a = state.edits.spawns.find(sp => sp.id === w.from_id);
        const b = state.edits.spawns.find(sp => sp.id === w.to_id);
        if (a && b) {
          window.psoSceneSetConnector && window.psoSceneSetConnector(
            `${w.from_id}:${w.to_id}`, a.world_pos, b.world_pos, w.style);
        }
      });
    });
    stage.querySelectorAll(".map-wp-speed").forEach(function (inp) {
      inp.addEventListener("change", function () {
        const i = parseInt(inp.dataset.wpI, 10);
        const w = state.edits.waypoints[i];
        if (!w) return;
        w.speed = parseFloat(inp.value) || 0;
      });
    });

    // Footer buttons
    $("#mapBtnSave") && $("#mapBtnSave").addEventListener("click", saveEdits);
    $("#mapBtnLoad") && $("#mapBtnLoad").addEventListener("click", function () {
      if (!state.activeMapId) return;
      setActiveMap(state.activeMapId, state.activeFloor);
    });
    $("#mapBtnExportJson") && $("#mapBtnExportJson").addEventListener("click", exportJson);
  }

  // ---- inspector -----------------------------------------------------
  function renderInspector() {
    const insp = state._insp;
    if (!insp) return;
    let html = '<div class="vp-insp-title">Map Editor</div>';
    html += '<div class="vp-insp-help dim">Pick a map + floor in the toolbar. ';
    html += 'Use the +Mob / +NPC / etc. buttons to drop spawns; click ~Connect ';
    html += 'then 2 spawns to wire a waypoint. Shift+drag in the viewport pans.';
    html += '</div>';
    const sel = state.selectedSpawnId
      ? state.edits.spawns.find(sp => sp.id === state.selectedSpawnId)
      : null;
    if (sel) {
      html += '<div class="vp-insp-section">';
      html += `<div class="map-insp-title">Spawn #${sel.id} <span class="dim">(${escapeHtml(sel.type)})</span></div>`;
      html += `<label>type:`;
      html += `<select id="mapInspType">`;
      for (const t of ["mob","npc","chest","switch","teleport"]) {
        html += `<option value="${t}"${t===sel.type?' selected':''}>${t}</option>`;
      }
      html += `</select></label>`;
      html += `<label>x: <input type="number" step="0.1" id="mapInspX" value="${sel.world_pos[0]}"/></label>`;
      html += `<label>y: <input type="number" step="0.1" id="mapInspY" value="${sel.world_pos[1]}"/></label>`;
      html += `<label>z: <input type="number" step="0.1" id="mapInspZ" value="${sel.world_pos[2]}"/></label>`;
      html += `<label>rot: <input type="number" step="0.01" id="mapInspRot" value="${sel.rotation || 0}"/></label>`;
      // Type-specific fields
      const td = sel.type_data || {};
      if (sel.type === "mob") {
        html += `<label>mob_id: <input type="number" id="mapInspMobId" value="${td.mob_id || 0}"/></label>`;
        html += `<label>count: <input type="number" id="mapInspCount" value="${td.count || 1}"/></label>`;
        html += `<label>behavior: <input type="text" id="mapInspBehavior" value="${escapeHtml(td.behavior || '')}"/></label>`;
      } else if (sel.type === "npc") {
        html += `<label>name: <input type="text" id="mapInspNpcName" value="${escapeHtml(td.name || '')}"/></label>`;
        html += `<label>dialog_id: <input type="number" id="mapInspDialogId" value="${td.dialog_id || 0}"/></label>`;
      } else if (sel.type === "chest") {
        html += `<label>item_id: <input type="number" id="mapInspItemId" value="${td.item_id || 0}"/></label>`;
      } else if (sel.type === "switch") {
        html += `<label>target_id: <input type="number" id="mapInspTargetId" value="${td.target_id || 0}"/></label>`;
        html += `<label>function: <input type="text" id="mapInspFunc" value="${escapeHtml(td.function || '')}"/></label>`;
      } else if (sel.type === "teleport") {
        html += `<label>dest_map: <input type="text" id="mapInspDestMap" value="${escapeHtml(td.dest_map || '')}"/></label>`;
        html += `<label>dest_floor: <input type="number" id="mapInspDestFloor" value="${td.dest_floor || 0}"/></label>`;
      }
      html += '</div>';
    } else {
      html += '<div class="vp-insp-section dim">no spawn selected. click a spawn in the right sidebar.</div>';
    }
    html += '<div class="vp-insp-section">';
    html += '<div class="dim">Markers in viewport — click to select, then drag to move.</div>';
    html += '</div>';
    insp.innerHTML = html;

    // Bind inspector edit handlers
    function bindInsp(id, fn) {
      const el = insp.querySelector(id);
      if (el) el.addEventListener("input", fn);
    }
    if (sel) {
      bindInsp("#mapInspType", () => {
        const t = insp.querySelector("#mapInspType").value;
        if (t === sel.type) return;
        sel.type = t;
        sel.type_data = {};
        // Re-render marker color
        window.psoSceneRemoveMarker && window.psoSceneRemoveMarker(sel.id);
        window.psoSceneAddMarker    && window.psoSceneAddMarker(sel.id, sel.world_pos, sel.type);
        renderInspector();
        const sl = state._stage.querySelector("#mapSpawnSide");
        if (sl) sl.innerHTML = renderSpawnList();
        rebindAfterRender();
      });
      function bindCoord(id, idx) {
        bindInsp(id, () => {
          sel.world_pos[idx] = parseFloat(insp.querySelector(id).value) || 0;
          window.psoSceneMoveMarker && window.psoSceneMoveMarker(sel.id, sel.world_pos);
          updateConnectorsForSpawn(sel.id);
        });
      }
      bindCoord("#mapInspX", 0);
      bindCoord("#mapInspY", 1);
      bindCoord("#mapInspZ", 2);
      bindInsp("#mapInspRot", () => {
        sel.rotation = parseFloat(insp.querySelector("#mapInspRot").value) || 0;
      });
      // Type-specific
      function bindTd(id, key, asNumber) {
        bindInsp(id, () => {
          if (!sel.type_data) sel.type_data = {};
          const v = insp.querySelector(id).value;
          sel.type_data[key] = asNumber ? (parseFloat(v) || 0) : v;
        });
      }
      bindTd("#mapInspMobId",     "mob_id",     true);
      bindTd("#mapInspCount",     "count",      true);
      bindTd("#mapInspBehavior",  "behavior",   false);
      bindTd("#mapInspNpcName",   "name",       false);
      bindTd("#mapInspDialogId",  "dialog_id",  true);
      bindTd("#mapInspItemId",    "item_id",    true);
      bindTd("#mapInspTargetId",  "target_id",  true);
      bindTd("#mapInspFunc",      "function",   false);
      bindTd("#mapInspDestMap",   "dest_map",   false);
      bindTd("#mapInspDestFloor", "dest_floor", true);
    }
  }

  // ---- mutations -----------------------------------------------------
  function addSpawn(worldPos, type) {
    const sp = {
      id: state.nextSpawnId++,
      type: type || "mob",
      world_pos: [worldPos[0], worldPos[1], worldPos[2]],
      rotation: 0,
      type_data: {},
    };
    state.edits.spawns.push(sp);
    window.psoSceneAddMarker && window.psoSceneAddMarker(sp.id, sp.world_pos, sp.type);
    state.selectedSpawnId = sp.id;
    reRender();
    renderInspector();
    return sp;
  }

  function deleteSpawn(id) {
    state.edits.spawns = state.edits.spawns.filter(sp => sp.id !== id);
    state.edits.waypoints = state.edits.waypoints.filter(w => {
      const drop = (w.from_id === id || w.to_id === id);
      if (drop) {
        window.psoSceneRemoveConnector && window.psoSceneRemoveConnector(`${w.from_id}:${w.to_id}`);
      }
      return !drop;
    });
    window.psoSceneRemoveMarker && window.psoSceneRemoveMarker(id);
    if (state.selectedSpawnId === id) state.selectedSpawnId = null;
    reRender();
    renderInspector();
  }

  function addWaypoint(fromId, toId) {
    const exists = state.edits.waypoints.find(w =>
      (w.from_id === fromId && w.to_id === toId) ||
      (w.from_id === toId && w.to_id === fromId));
    if (exists) {
      setStatus(`waypoint #${fromId}-${toId} already exists`);
      return;
    }
    const w = { from_id: fromId, to_id: toId, speed: 1.0, style: "walk" };
    state.edits.waypoints.push(w);
    const a = state.edits.spawns.find(sp => sp.id === fromId);
    const b = state.edits.spawns.find(sp => sp.id === toId);
    if (a && b) {
      window.psoSceneSetConnector && window.psoSceneSetConnector(
        `${fromId}:${toId}`, a.world_pos, b.world_pos, w.style);
    }
  }

  function deleteWaypoint(idx) {
    const w = state.edits.waypoints[idx];
    if (!w) return;
    window.psoSceneRemoveConnector && window.psoSceneRemoveConnector(`${w.from_id}:${w.to_id}`);
    state.edits.waypoints.splice(idx, 1);
    reRender();
  }

  // When a spawn moves, redraw all waypoints touching it.
  function updateConnectorsForSpawn(spawnId) {
    for (const w of state.edits.waypoints) {
      if (w.from_id !== spawnId && w.to_id !== spawnId) continue;
      const a = state.edits.spawns.find(sp => sp.id === w.from_id);
      const b = state.edits.spawns.find(sp => sp.id === w.to_id);
      if (!a || !b) continue;
      window.psoSceneSetConnector && window.psoSceneSetConnector(
        `${w.from_id}:${w.to_id}`, a.world_pos, b.world_pos, w.style);
    }
  }

  // ---- viewport click handling --------------------------------------
  function onViewportPointerDown(e) {
    if (e.button !== 0) return;
    if (e.shiftKey) return;  // pan
    if (!window.psoSceneRaycast) return;
    const hit = window.psoSceneRaycast(e.clientX, e.clientY);
    if (!hit) return;
    if (state.placeMode) {
      addSpawn(hit.world_pos, state.placeType);
      // Stay in place mode so the user can drop several quickly. Click
      // the active button again to exit.
      e.preventDefault();
      return;
    }
    // Marker drag: check if hit was a marker mesh
    const obj = hit.hit_object;
    if (obj && obj.userData && obj.userData.markerId != null) {
      state._dragMarkerId = obj.userData.markerId;
      state.selectedSpawnId = obj.userData.markerId;
      renderInspector();
      const sl = state._stage.querySelector("#mapSpawnSide");
      if (sl) sl.innerHTML = renderSpawnList();
      rebindAfterRender();
      e.preventDefault();
      return;
    }
    updateCoordReadout("click", `(${fmtCoord(hit.world_pos[0])}, ${fmtCoord(hit.world_pos[1])}, ${fmtCoord(hit.world_pos[2])})`);
  }

  function onViewportPointerMove(e) {
    if (state._dragMarkerId == null) return;
    if (!window.psoSceneRaycast) return;
    const hit = window.psoSceneRaycast(e.clientX, e.clientY);
    if (!hit) return;
    const sp = state.edits.spawns.find(s => s.id === state._dragMarkerId);
    if (!sp) return;
    sp.world_pos = [hit.world_pos[0], hit.world_pos[1], hit.world_pos[2]];
    window.psoSceneMoveMarker && window.psoSceneMoveMarker(sp.id, sp.world_pos);
    updateConnectorsForSpawn(sp.id);
    updateCoordReadout("drag", `(${fmtCoord(sp.world_pos[0])}, ${fmtCoord(sp.world_pos[1])}, ${fmtCoord(sp.world_pos[2])})`);
  }

  function onViewportPointerUp(e) {
    if (state._dragMarkerId != null) {
      state._dragMarkerId = null;
      // Refresh sidebar coords + inspector
      const sl = state._stage.querySelector("#mapSpawnSide");
      if (sl) sl.innerHTML = renderSpawnList();
      rebindAfterRender();
      renderInspector();
    }
  }

  // ---- save / export -------------------------------------------------
  async function saveEdits() {
    if (!state.activeMapId) return;
    setStatus("saving…");
    try {
      const r = await saveEditsToServer(state.activeMapId, state.edits);
      setStatus(`saved ${r.spawn_count} spawns, ${r.waypoint_count} waypoints`);
    } catch (e) {
      setStatus("save failed: " + (e && e.message || e), true);
    }
  }

  function exportJson() {
    const norm = {
      version:   1,
      map_id:    state.activeMapId,
      spawns:    state.edits.spawns,
      waypoints: state.edits.waypoints,
    };
    const blob = new Blob([JSON.stringify(norm, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `map_edits_${state.activeMapId || 'unsaved'}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // ---- prop picker ---------------------------------------------------
  function openPropPicker() {
    // Minimal modal: text input for the model path. Future: tree of
    // BML/NJ models from the manifest. For v1 the user pastes a path
    // and we call psoSceneAddProp.
    const path = window.prompt(
      "Enter model path (e.g. biri_ball.bml#biri_ball.nj or scene/map_aancient01_00s.nj):",
      "");
    if (!path) return;
    if (!window.psoSceneAddProp) {
      setStatus("prop API not available", true);
      return;
    }
    const cam = window.psoGetCamera && window.psoGetCamera();
    const target = (window.psoGetCamera) ? null : null;  // currently no easy way to get camera target; use camera position
    const pos = cam ? [cam.position.x, cam.position.y, cam.position.z] : [0, 0, 0];
    window.psoSceneAddProp(path, pos, 0).then(grp => {
      if (grp) {
        setStatus("prop added: " + path);
        reRender();
      } else {
        setStatus("prop add failed (model not found?)", true);
      }
    });
  }

  // ---- perspective registration -------------------------------------
  window.PSOPerspectives.register("map-editor", {
    label: "Map editor",
    match: function (entry, file) {
      // The Map Editor is reached via the header "Map Editor" button (it's
      // a top-level workspace, not a per-asset viewport mode). Auto-route
      // ONLY when the user clicks a real map asset in the tree — never
      // appear as a tab on enemies / bosses / players / weapons / props.
      if (entry && entry.category === "map") return 90;
      const fn = (file || "").toLowerCase();
      if (fn.startsWith("scene/map_")) return 70;
      return 0;
    },
    mount: async function (stage, insp, ctx) {
      state._stage = stage;
      state._insp  = insp;

      // Suppress Esc — the renderer's close hook lives on the model
      // viewer's modal; in unified mode we want tabs to be the only exit.
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
        <div class="map-perspective">
          <div id="mapToolbar"></div>
          <div class="map-body">
            <aside id="mapSceneTree" class="map-side map-side-left"></aside>
            <main id="mapViewport" class="map-viewport"></main>
            <aside id="mapSpawnSide" class="map-side map-side-right"></aside>
          </div>
          ${renderFooter()}
        </div>
      `;

      // Pluck the model viewer's canvas + bar into the viewport host.
      const restorers = [];
      const card = document.querySelector("#modelModal .model-modal-card");
      const bar = document.querySelector("#modelModal .model-bar");
      const mstage = document.querySelector("#modelModal .model-stage");
      const homeBar = bar ? bar.parentNode : null;
      const homeStage = mstage ? mstage.parentNode : null;
      const nextBar = bar ? bar.nextSibling : null;
      const nextStage = mstage ? mstage.nextSibling : null;
      const vpHost = stage.querySelector("#mapViewport");
      // Hide the model bar in this perspective — the toolbar above
      // duplicates the relevant controls.
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
      stage._mapRestorers = restorers;

      // Force the renderer to grab its new parent's size.
      setTimeout(function () {
        if (typeof window.psoModelRebindResize === "function") {
          window.psoModelRebindResize();
        }
        window.dispatchEvent(new Event("resize"));
      }, 80);

      // Toolbar status placeholder
      const tb = stage.querySelector("#mapToolbar");
      if (tb) tb.innerHTML = renderToolbar();
      const tree = stage.querySelector("#mapSceneTree");
      if (tree) tree.innerHTML = renderSceneTree();
      const sl = stage.querySelector("#mapSpawnSide");
      if (sl) sl.innerHTML = renderSpawnList();
      state._coordReadout = stage.querySelector("#mapCoordReadout");
      rebindAfterRender();

      // Wire viewport pointer events — hits handled by the panel
      // (raycast → spawn drop / marker drag).
      const cv = window.psoGetCanvas && window.psoGetCanvas();
      if (cv && !cv.__mapPanelBound) {
        cv.addEventListener("pointerdown", onViewportPointerDown);
        cv.addEventListener("pointermove", onViewportPointerMove);
        cv.addEventListener("pointerup",   onViewportPointerUp);
        cv.__mapPanelBound = true;
        restorers.push(function () {
          cv.removeEventListener("pointerdown", onViewportPointerDown);
          cv.removeEventListener("pointermove", onViewportPointerMove);
          cv.removeEventListener("pointerup",   onViewportPointerUp);
          cv.__mapPanelBound = false;
        });
      }

      // RAF loop for camera coord readout
      let rafId = null;
      function tick() {
        rafId = requestAnimationFrame(tick);
        updateCoordReadout();
      }
      tick();
      restorers.push(function () { if (rafId) cancelAnimationFrame(rafId); });

      renderInspector();

      // Initial load — catalogue + first map
      try {
        await loadCatalogue();
        // Pick a first map: prefer aancient01 (forest) since that's the
        // canonical "rich terrain" for the smoke test. If the catalogue
        // doesn't have it, pick the first one with renderable_files > 0.
        let firstId = null;
        const maps = state.catalogue.maps || [];
        const preferred = maps.find(m => m.map_id === "aancient01");
        const richest = maps.find(m => m.renderable_files > 0);
        const ctxFile = ctx && ctx.fileName;
        // If user clicked a scene/map_*.nj path, route to that map.
        if (ctxFile) {
          const m = ctxFile.match(/^map_([a-z]+)(\d+)/i);
          if (m) {
            firstId = `${m[1].toLowerCase()}${parseInt(m[2], 10).toString().padStart(2, '0')}`;
            if (!maps.find(mp => mp.map_id === firstId)) firstId = null;
          }
        }
        firstId = firstId
          || (preferred && preferred.map_id)
          || (richest && richest.map_id)
          || (maps[0] && maps[0].map_id);
        if (firstId) {
          await setActiveMap(firstId, 0);
          renderInspector();
        } else {
          setStatus("no maps found in manifest", true);
        }
      } catch (e) {
        console.error("[map_panel] mount-load failed:", e);
        setStatus("init failed: " + (e && e.message || e), true);
      }
    },
    unmount: function (stage, insp) {
      if (state._escSuppressor) {
        document.removeEventListener("keydown", state._escSuppressor, true);
        state._escSuppressor = null;
      }
      // Stop the shared render loop + cancel any pending one-shot so the
      // viewer fully idles when we leave the map tab (scene mode shares
      // the model viewer's renderer and has no continuous animator of its
      // own — without this the loop would keep running after unmount).
      window.psoViewerStopLoop && window.psoViewerStopLoop();
      // Clear the loaded scene so it doesn't bleed into the next perspective.
      window.psoSceneClearMap && window.psoSceneClearMap();
      // Roll back per-area fog + lighting tweaks so the model viewer
      // keeps its single-model preview default appearance.
      window.psoSceneResetEnvironment && window.psoSceneResetEnvironment();
      // Restore the model viewer's modal-owned DOM nodes.
      try {
        if (stage._mapRestorers) {
          stage._mapRestorers.forEach(function (f) { try { f(); } catch (_e) {} });
          stage._mapRestorers = null;
        }
      } catch (_e) {}
      state._stage = null;
      state._insp  = null;
      state._coordReadout = null;
    },
  });

  // ---- header button -------------------------------------------------
  function openPerspective() {
    const ctx = {
      path: "__map_editor__",
      entry: { category: "map", format: "MapScene" },
      fileName: "Map editor",
    };
    if (window.PSOPerspectives && window.PSOPerspectives.switchTo) {
      window.PSOPerspectives.switchTo("map-editor", ctx);
    }
  }

  function ensureHeaderButton() {
    if (document.getElementById("btnMapEditor")) return;
    const status = document.getElementById("status");
    const header = status ? status.parentNode : null;
    if (!header) return;
    const btn = document.createElement("button");
    btn.id = "btnMapEditor";
    btn.type = "button";
    btn.className = "ghost";
    btn.title = "Map Editor — load a scene + drop spawns/waypoints";
    btn.textContent = "Map Editor";
    header.insertBefore(btn, status);
    btn.addEventListener("click", openPerspective);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", ensureHeaderButton);
  } else {
    ensureHeaderButton();
  }
})();
