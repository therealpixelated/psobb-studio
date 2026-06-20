// =====================================================================
// PSOBB Edit Panel — vertex/edge/face selection + transform gizmo.
// 2026-04-26
//
// The "Edit" tab adds Blender-style edit-mode to the 3D viewport:
//
//   Selection modes (V/E/F keys, or radio buttons):
//     - Vertex   click to add, Shift+click to multi-select,
//                Shift+drag for box-select
//     - Edge     P1 (groundwork in this file; toggle disabled until P1)
//     - Face     P1 ditto
//
//   Transform modes (G/R/S keys):
//     - G translate    selected verts move with the gizmo
//     - R rotate       (about selection centroid)
//     - S scale        (about selection centroid)
//
//   Save-back:
//     The "Save edits" button POSTs the captured vertex deltas to
//     /api/protools/save_vertex_transforms.  The server writes a JSON
//     sidecar in cache/protools_edits/<sha>.json — same shape sculpt
//     uses, so the existing /api/sculpt/build_archive path picks them
//     up unchanged.
//
//   Undo:
//     Every committed transform pushes onto window.psoUndoBus so
//     Ctrl+Z works the same as it does for paint / sculpt / mob_dsl.
//
// Mounted as a tab via psoTexturePanelRegisterTab.  Toggle is via the
// new top-toolbar "Edit Mode" button (#btnEditMode in index.html).
//
// Idempotent on multiple loads.
// =====================================================================

(function () {
  "use strict";

  if (window.__psoEditPanelLoaded) return;
  window.__psoEditPanelLoaded = true;

  const STYLE_ID = "psoEditPanelStyle";
  const TAB_NAME = "edit";
  const TAB_LABEL = "Edit";
  const TAB_TITLE = "vertex selection + transform gizmo (Blender-style edit-mode)";

  const SELMODE_VERTEX = "vertex";
  const SELMODE_EDGE = "edge";
  const SELMODE_FACE = "face";
  const VALID_SELMODES = [SELMODE_VERTEX, SELMODE_EDGE, SELMODE_FACE];

  const TFM_TRANSLATE = "translate";
  const TFM_ROTATE = "rotate";
  const TFM_SCALE = "scale";

  // ---- module state ------------------------------------------------
  const state = {
    enabled: false,           // edit-mode toggle (mirrors sculpt)
    selMode: SELMODE_VERTEX,
    activeTfm: TFM_TRANSLATE,
    activeSubmeshIdx: 0,
    // Selection: Map<submeshIdx, Set<vertexIdx>>.  Set so add/remove is fast.
    selection: new Map(),
    // Visual marker container for selected vertices (cyan dots).
    markersGroup: null,        // THREE.Group
    // Proxy Object3D the gizmo manipulates; sits at selection centroid.
    proxy: null,
    proxyDelta: null,           // {position, rotation, scale} since drag start
    // Per-stroke captured "before" arrays for undo: Map<submeshIdx, {indices: Uint32Array, before: Float32Array}>
    strokeStart: null,
    // Box-select state.
    boxSelect: { active: false, x0: 0, y0: 0, x1: 0, y1: 0, additive: false, overlayEl: null },
    // Save-back identifiers.
    modelPath: null,            // resolved e.g. "<bml>#<inner>.nj"
    sourceSha: null,            // captured first time we hit save
    // DOM refs.
    bodyEl: null,
    statusEl: null,
    selCountEl: null,
    submeshSelEl: null,
    // Hover highlight raycaster.
    raycaster: null,
  };

  // ---- styles ------------------------------------------------------
  function ensureStyleInjected() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .pso-edit-toolbar {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
        padding: 4px 0;
        font-size: 11px;
        align-items: center;
      }
      .pso-edit-toolbar label {
        display: inline-flex;
        align-items: center;
        gap: 3px;
        color: #c8d3df;
      }
      .pso-edit-toolbar select {
        background: #0e1318;
        border: 1px solid #2a313a;
        color: #d8e3ef;
        padding: 2px 6px;
        border-radius: 3px;
      }
      .pso-edit-toolbar button {
        background: #232931;
        border: 1px solid #2a313a;
        color: #c8d3df;
        padding: 2px 8px;
        cursor: pointer;
        border-radius: 3px;
      }
      .pso-edit-toolbar button:hover { background: #2d3540; }
      .pso-edit-toolbar button.active {
        background: #2d4a6e;
        border-color: #3d6e9e;
        color: #fff;
      }
      .pso-edit-toolbar button.disabled {
        opacity: 0.45;
        cursor: not-allowed;
      }
      .pso-edit-toolbar button.danger { background: #3d1f24; border-color: #5c2d35; }
      .pso-edit-toolbar button.danger:hover { background: #5c2d35; }
      .pso-edit-status {
        font-family: ui-monospace, monospace;
        font-size: 11px;
        color: #8a96a6;
        padding: 4px 0;
      }
      .pso-edit-help {
        padding: 6px 8px;
        background: #1a1f25;
        border: 1px solid #2a313a;
        border-radius: 4px;
        font-size: 11px;
        color: #c8d3df;
        margin-top: 6px;
      }
      .pso-edit-help kbd {
        background: #0e1318;
        border: 1px solid #2a313a;
        padding: 0 4px;
        border-radius: 3px;
        font-family: ui-monospace, monospace;
        font-size: 10px;
      }
      .pso-edit-pillrow {
        display: inline-flex;
        gap: 0;
        border: 1px solid #2a313a;
        border-radius: 3px;
        overflow: hidden;
      }
      .pso-edit-pillrow button {
        border: none;
        border-radius: 0;
        border-right: 1px solid #2a313a;
      }
      .pso-edit-pillrow button:last-child { border-right: none; }
      .pso-edit-modebtn {
        background: linear-gradient(90deg, #2d4a6e, #1a3050);
        color: #fff;
        border: 1px solid #3d6e9e;
        padding: 2px 10px;
        border-radius: 3px;
        cursor: pointer;
      }
      .pso-edit-modebtn.active { background: #3d6e9e; }
      .pso-edit-boxsel-overlay {
        position: absolute;
        background: rgba(60, 130, 220, 0.18);
        border: 1px solid rgba(120, 200, 255, 0.85);
        pointer-events: none;
        z-index: 50;
      }
    `;
    document.head.appendChild(style);
  }

  // ---- helpers -----------------------------------------------------
  function totalSelectedCount() {
    let n = 0;
    for (const set of state.selection.values()) n += set.size;
    return n;
  }

  function clearSelection() {
    state.selection.clear();
    rebuildMarkers();
    detachGizmo();
    updateStatus();
    if (window.bus && window.bus.emit) {
      try { window.bus.emit("edit.selectionChanged", { count: 0 }); } catch (_e) {}
    }
  }

  function selectAll() {
    const dbg = window.psoGetDebugMeshes && window.psoGetDebugMeshes();
    if (!dbg) return;
    state.selection.clear();
    for (let i = 0; i < dbg.length; i++) {
      const e = dbg[i];
      if (!e || !e.mesh || !e.mesh.geometry) continue;
      const p = e.mesh.geometry.getAttribute("position");
      if (!p) continue;
      const set = new Set();
      for (let v = 0; v < p.count; v++) set.add(v);
      state.selection.set(i, set);
    }
    rebuildMarkers();
    placeProxyAtCentroid();
    updateStatus();
  }

  function getDebugMeshes() {
    return window.psoGetDebugMeshes ? (window.psoGetDebugMeshes() || []) : [];
  }

  function getMeshGroup() {
    return window.psoGetMeshGroup ? window.psoGetMeshGroup() : null;
  }

  function getCamera() {
    return window.psoGetCamera ? window.psoGetCamera() : null;
  }

  // World-space position of vertex (submesh-local + mesh.matrixWorld).
  function vertexWorldPos(submeshIdx, vertIdx, outVec3) {
    const dbg = getDebugMeshes();
    const e = dbg[submeshIdx];
    if (!e || !e.mesh || !e.mesh.geometry) return false;
    const pa = e.mesh.geometry.getAttribute("position");
    if (!pa || vertIdx < 0 || vertIdx >= pa.count) return false;
    const THREE = window.THREE;
    if (!THREE) return false;
    e.mesh.updateMatrixWorld(true);
    const v = outVec3 || new THREE.Vector3();
    v.set(pa.array[vertIdx * 3], pa.array[vertIdx * 3 + 1], pa.array[vertIdx * 3 + 2]);
    v.applyMatrix4(e.mesh.matrixWorld);
    return v;
  }

  // ---- selection markers --------------------------------------------
  function ensureMarkersGroup() {
    const THREE = window.THREE;
    const root = getMeshGroup() ? getMeshGroup().parent : null;
    if (!THREE || !root) return null;
    if (state.markersGroup && state.markersGroup.parent === root) return state.markersGroup;
    if (state.markersGroup && state.markersGroup.parent) {
      state.markersGroup.parent.remove(state.markersGroup);
    }
    state.markersGroup = new THREE.Group();
    state.markersGroup.name = "psoEditMarkers";
    root.add(state.markersGroup);
    return state.markersGroup;
  }

  function clearMarkers() {
    if (!state.markersGroup) return;
    while (state.markersGroup.children.length) {
      const c = state.markersGroup.children[0];
      state.markersGroup.remove(c);
      try { if (c.geometry) c.geometry.dispose(); } catch (_e) {}
      try { if (c.material) c.material.dispose(); } catch (_e) {}
    }
  }

  function rebuildMarkers() {
    const THREE = window.THREE;
    if (!THREE) return;
    const grp = ensureMarkersGroup();
    if (!grp) return;
    clearMarkers();
    if (!state.enabled) return;
    const dbg = getDebugMeshes();
    if (!dbg.length) return;
    const sphereGeo = new THREE.SphereGeometry(0.02, 6, 4);
    const mat = new THREE.MeshBasicMaterial({
      color: 0x66ccff,
      transparent: true,
      opacity: 0.95,
      depthTest: false,
    });
    for (const [submeshIdx, set] of state.selection) {
      const e = dbg[submeshIdx];
      if (!e || !e.mesh) continue;
      e.mesh.updateMatrixWorld(true);
      const posAttr = e.mesh.geometry.getAttribute("position");
      if (!posAttr) continue;
      for (const vi of set) {
        if (vi < 0 || vi >= posAttr.count) continue;
        const m = new THREE.Mesh(sphereGeo, mat);
        m.renderOrder = 9998;
        const p = vertexWorldPos(submeshIdx, vi);
        if (p) m.position.copy(p);
        m.userData = { submeshIdx, vertexIdx: vi };
        grp.add(m);
      }
    }
    if (window.psoForceRender) window.psoForceRender();
  }

  function refreshMarkerPositions() {
    if (!state.markersGroup) return;
    for (const m of state.markersGroup.children) {
      const u = m.userData;
      if (!u) continue;
      const p = vertexWorldPos(u.submeshIdx, u.vertexIdx);
      if (p) m.position.copy(p);
    }
    if (window.psoForceRender) window.psoForceRender();
  }

  // ---- selection centroid + gizmo --------------------------------
  function selectionCentroid() {
    const THREE = window.THREE;
    if (!THREE) return null;
    let cx = 0, cy = 0, cz = 0, n = 0;
    const tmp = new THREE.Vector3();
    for (const [submeshIdx, set] of state.selection) {
      for (const vi of set) {
        if (vertexWorldPos(submeshIdx, vi, tmp)) {
          cx += tmp.x; cy += tmp.y; cz += tmp.z; n++;
        }
      }
    }
    if (!n) return null;
    return new THREE.Vector3(cx / n, cy / n, cz / n);
  }

  function ensureProxy() {
    const THREE = window.THREE;
    if (!THREE) return null;
    const root = getMeshGroup() ? getMeshGroup().parent : null;
    if (!root) return null;
    if (state.proxy && state.proxy.parent === root) return state.proxy;
    if (state.proxy && state.proxy.parent) state.proxy.parent.remove(state.proxy);
    state.proxy = new THREE.Object3D();
    state.proxy.name = "psoEditGizmoProxy";
    root.add(state.proxy);
    return state.proxy;
  }

  function placeProxyAtCentroid() {
    const c = selectionCentroid();
    if (!c) {
      detachGizmo();
      return;
    }
    const proxy = ensureProxy();
    if (!proxy) return;
    proxy.position.copy(c);
    proxy.rotation.set(0, 0, 0);
    proxy.scale.set(1, 1, 1);
    proxy.updateMatrixWorld(true);
    attachGizmo(proxy);
  }

  let gizmoChangeUnsub = null;
  let gizmoCommitUnsub = null;

  async function attachGizmo(proxy) {
    if (!window.psoTransformGizmo) return false;
    const ok = await window.psoTransformGizmo.attach(proxy);
    if (!ok) return false;
    window.psoTransformGizmo.setMode(state.activeTfm);
    if (gizmoChangeUnsub) { gizmoChangeUnsub(); gizmoChangeUnsub = null; }
    if (gizmoCommitUnsub) { gizmoCommitUnsub(); gizmoCommitUnsub = null; }
    gizmoChangeUnsub = window.psoTransformGizmo.onChange(onGizmoChange);
    gizmoCommitUnsub = window.psoTransformGizmo.onCommit(onGizmoCommit);
    return true;
  }

  function detachGizmo() {
    if (window.psoTransformGizmo) {
      try { window.psoTransformGizmo.detach(); } catch (_e) {}
    }
    if (gizmoChangeUnsub) { gizmoChangeUnsub(); gizmoChangeUnsub = null; }
    if (gizmoCommitUnsub) { gizmoCommitUnsub(); gizmoCommitUnsub = null; }
  }

  // Capture before-positions on first onChange of a stroke; apply
  // current proxy delta to all selected vertices each frame.
  function onGizmoChange() {
    if (!window.psoTransformGizmo) return;
    const d = window.psoTransformGizmo.getDelta();
    if (!d) return;
    if (!state.strokeStart) {
      // Capture before-positions (one-shot per drag).
      state.strokeStart = new Map();
      const dbg = getDebugMeshes();
      const THREE = window.THREE;
      const tmp = new THREE.Vector3();
      // Also capture the reference centroid (in world space) for rotate/scale pivots.
      const c = selectionCentroid();
      state.strokeCentroid = c ? c.clone() : null;
      for (const [submeshIdx, set] of state.selection) {
        const e = dbg[submeshIdx];
        if (!e || !e.mesh || !e.mesh.geometry) continue;
        const pa = e.mesh.geometry.getAttribute("position");
        if (!pa) continue;
        const indices = new Uint32Array(set.size);
        const before = new Float32Array(set.size * 3);
        let k = 0;
        for (const vi of set) {
          indices[k] = vi;
          before[k * 3 + 0] = pa.array[vi * 3 + 0];
          before[k * 3 + 1] = pa.array[vi * 3 + 1];
          before[k * 3 + 2] = pa.array[vi * 3 + 2];
          k++;
        }
        state.strokeStart.set(submeshIdx, { indices, before, mesh: e.mesh });
      }
    }
    applyDeltaToSelection(d);
    refreshMarkerPositions();
    updateStatus();
  }

  // After gizmo onCommit (mouseup): finalize the stroke, push to undo bus.
  function onGizmoCommit() {
    if (!state.strokeStart || !state.strokeStart.size) {
      state.strokeStart = null;
      return;
    }
    // Capture after-positions.
    const captured = state.strokeStart;
    const after = new Map();
    let totalVerts = 0;
    for (const [submeshIdx, rec] of captured) {
      const pa = rec.mesh.geometry.getAttribute("position");
      if (!pa) continue;
      const af = new Float32Array(rec.indices.length * 3);
      for (let k = 0; k < rec.indices.length; k++) {
        const vi = rec.indices[k];
        af[k * 3 + 0] = pa.array[vi * 3 + 0];
        af[k * 3 + 1] = pa.array[vi * 3 + 1];
        af[k * 3 + 2] = pa.array[vi * 3 + 2];
      }
      after.set(submeshIdx, af);
      totalVerts += rec.indices.length;
    }
    state.strokeStart = null;
    state.strokeCentroid = null;

    // Reset gizmo proxy back to centroid (so the gizmo doesn't drift on next drag).
    placeProxyAtCentroid();

    // Push to global undo bus.
    if (window.psoUndoBus) {
      const lbl = state.activeTfm + " " + totalVerts + " vert" + (totalVerts === 1 ? "" : "s");
      function applyAfter() {
        for (const [submeshIdx, rec] of captured) {
          const pa = rec.mesh.geometry.getAttribute("position");
          if (!pa) continue;
          const af = after.get(submeshIdx);
          for (let k = 0; k < rec.indices.length; k++) {
            const vi = rec.indices[k];
            pa.array[vi * 3] = af[k * 3];
            pa.array[vi * 3 + 1] = af[k * 3 + 1];
            pa.array[vi * 3 + 2] = af[k * 3 + 2];
          }
          pa.needsUpdate = true;
          rec.mesh.geometry.computeVertexNormals();
        }
        refreshMarkerPositions();
        placeProxyAtCentroid();
      }
      function applyBefore() {
        for (const [submeshIdx, rec] of captured) {
          const pa = rec.mesh.geometry.getAttribute("position");
          if (!pa) continue;
          for (let k = 0; k < rec.indices.length; k++) {
            const vi = rec.indices[k];
            pa.array[vi * 3] = rec.before[k * 3];
            pa.array[vi * 3 + 1] = rec.before[k * 3 + 1];
            pa.array[vi * 3 + 2] = rec.before[k * 3 + 2];
          }
          pa.needsUpdate = true;
          rec.mesh.geometry.computeVertexNormals();
        }
        refreshMarkerPositions();
        placeProxyAtCentroid();
      }
      window.psoUndoBus.push({
        label: lbl,
        panelId: "edit",
        undo: applyBefore,
        redo: applyAfter,
      });
    }

    // Mark "dirty" so the Save button knows there are pending edits.
    state.dirty = true;
    updateStatus();
  }

  function applyDeltaToSelection(delta) {
    if (!delta) return;
    const THREE = window.THREE;
    if (!THREE || !state.strokeStart) return;
    // For translate: pos += deltaPos.
    // For rotate: rotate around centroid (world space).
    // For scale: scale around centroid.
    const dbg = getDebugMeshes();
    const c = state.strokeCentroid;
    const tmp = new THREE.Vector3();
    const m4 = new THREE.Matrix4();
    let R = null;
    if (state.activeTfm === TFM_ROTATE && delta.rotation && c) {
      const e = new THREE.Euler(delta.rotation[0], delta.rotation[1], delta.rotation[2], "XYZ");
      const q = new THREE.Quaternion().setFromEuler(e);
      R = new THREE.Matrix4().makeRotationFromQuaternion(q);
    }
    let sx = 1, sy = 1, sz = 1;
    if (state.activeTfm === TFM_SCALE && delta.scale && c) {
      sx = delta.scale[0]; sy = delta.scale[1]; sz = delta.scale[2];
    }
    for (const [submeshIdx, rec] of state.strokeStart) {
      const e = dbg[submeshIdx];
      if (!e || !e.mesh || !e.mesh.geometry) continue;
      const pa = e.mesh.geometry.getAttribute("position");
      if (!pa) continue;
      e.mesh.updateMatrixWorld(true);
      const inv = new THREE.Matrix4().copy(e.mesh.matrixWorld).invert();
      for (let k = 0; k < rec.indices.length; k++) {
        const vi = rec.indices[k];
        const bx = rec.before[k * 3];
        const by = rec.before[k * 3 + 1];
        const bz = rec.before[k * 3 + 2];
        // Local before position -> world.
        tmp.set(bx, by, bz).applyMatrix4(e.mesh.matrixWorld);
        if (state.activeTfm === TFM_TRANSLATE && delta.position) {
          tmp.x += delta.position[0];
          tmp.y += delta.position[1];
          tmp.z += delta.position[2];
        } else if (state.activeTfm === TFM_ROTATE && R && c) {
          tmp.x -= c.x; tmp.y -= c.y; tmp.z -= c.z;
          tmp.applyMatrix4(R);
          tmp.x += c.x; tmp.y += c.y; tmp.z += c.z;
        } else if (state.activeTfm === TFM_SCALE && c) {
          tmp.x = c.x + (tmp.x - c.x) * sx;
          tmp.y = c.y + (tmp.y - c.y) * sy;
          tmp.z = c.z + (tmp.z - c.z) * sz;
        }
        // Back to local.
        tmp.applyMatrix4(inv);
        pa.array[vi * 3] = tmp.x;
        pa.array[vi * 3 + 1] = tmp.y;
        pa.array[vi * 3 + 2] = tmp.z;
      }
      pa.needsUpdate = true;
    }
  }

  // ---- click selection raycast --------------------------------------
  // Click on an empty area = clear; click on a vertex (within radius) = select.
  // Shift+click = additive.
  function ensureRaycaster() {
    if (state.raycaster) return state.raycaster;
    const THREE = window.THREE;
    if (!THREE) return null;
    state.raycaster = new THREE.Raycaster();
    return state.raycaster;
  }

  function pickVertexFromClick(clientX, clientY) {
    const THREE = window.THREE;
    const cam = getCamera();
    const dbg = getDebugMeshes();
    if (!THREE || !cam || !dbg.length) return null;
    const cv = window.psoGetCanvas && window.psoGetCanvas();
    if (!cv) return null;
    const rect = cv.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
    // Project every visible vertex; find the closest screen-space match.
    const proj = new THREE.Vector3();
    const tmp = new THREE.Vector3();
    let best = null, bestD2 = (12 / Math.min(rect.width, rect.height) * 2) ** 2;
    for (let s = 0; s < dbg.length; s++) {
      const e = dbg[s];
      if (!e || !e.mesh || !e.mesh.geometry) continue;
      const mesh = e.mesh;
      mesh.updateMatrixWorld(true);
      const pa = mesh.geometry.getAttribute("position");
      if (!pa) continue;
      for (let v = 0; v < pa.count; v++) {
        tmp.set(pa.array[v * 3], pa.array[v * 3 + 1], pa.array[v * 3 + 2])
          .applyMatrix4(mesh.matrixWorld);
        proj.copy(tmp).project(cam);
        const dx = proj.x - ndc.x;
        const dy = proj.y - ndc.y;
        const d2 = dx * dx + dy * dy;
        if (d2 < bestD2 && proj.z < 1.0) {
          bestD2 = d2;
          best = { submeshIdx: s, vertexIdx: v };
        }
      }
    }
    return best;
  }

  function onCanvasClick(ev) {
    if (!state.enabled) return;
    if (window.psoTransformGizmo && window.psoTransformGizmo.isDragging()) return;
    if (state.boxSelect.active) return;
    if (ev.button !== 0) return;
    const hit = pickVertexFromClick(ev.clientX, ev.clientY);
    if (!hit) {
      if (!ev.shiftKey) clearSelection();
      return;
    }
    const additive = ev.shiftKey;
    if (!additive) state.selection.clear();
    let set = state.selection.get(hit.submeshIdx);
    if (!set) { set = new Set(); state.selection.set(hit.submeshIdx, set); }
    if (additive && set.has(hit.vertexIdx)) {
      set.delete(hit.vertexIdx);
      if (!set.size) state.selection.delete(hit.submeshIdx);
    } else {
      set.add(hit.vertexIdx);
    }
    rebuildMarkers();
    placeProxyAtCentroid();
    updateStatus();
    if (window.bus && window.bus.emit) {
      try {
        window.bus.emit("edit.selectionChanged", { count: totalSelectedCount() });
      } catch (_e) {}
    }
  }

  // ---- box-select (Shift+drag) -------------------------------------
  function onBoxSelectStart(ev) {
    if (!state.enabled) return;
    if (!ev.shiftKey || ev.button !== 0) return;
    if (window.psoTransformGizmo && window.psoTransformGizmo.isDragging()) return;
    const cv = window.psoGetCanvas && window.psoGetCanvas();
    if (!cv) return;
    state.boxSelect.active = true;
    state.boxSelect.x0 = ev.clientX;
    state.boxSelect.y0 = ev.clientY;
    state.boxSelect.x1 = ev.clientX;
    state.boxSelect.y1 = ev.clientY;
    state.boxSelect.additive = ev.ctrlKey || ev.metaKey;
    // Overlay div.
    const div = document.createElement("div");
    div.className = "pso-edit-boxsel-overlay";
    document.body.appendChild(div);
    state.boxSelect.overlayEl = div;
    updateBoxSelOverlay();
  }

  function onBoxSelectMove(ev) {
    if (!state.boxSelect.active) return;
    state.boxSelect.x1 = ev.clientX;
    state.boxSelect.y1 = ev.clientY;
    updateBoxSelOverlay();
  }

  function onBoxSelectEnd(ev) {
    if (!state.boxSelect.active) return;
    state.boxSelect.active = false;
    if (state.boxSelect.overlayEl) {
      state.boxSelect.overlayEl.remove();
      state.boxSelect.overlayEl = null;
    }
    const x0 = Math.min(state.boxSelect.x0, state.boxSelect.x1);
    const x1 = Math.max(state.boxSelect.x0, state.boxSelect.x1);
    const y0 = Math.min(state.boxSelect.y0, state.boxSelect.y1);
    const y1 = Math.max(state.boxSelect.y0, state.boxSelect.y1);
    if (x1 - x0 < 3 && y1 - y0 < 3) return;  // tap, not drag
    runBoxSelect(x0, y0, x1, y1, state.boxSelect.additive);
  }

  function updateBoxSelOverlay() {
    if (!state.boxSelect.overlayEl) return;
    const x0 = Math.min(state.boxSelect.x0, state.boxSelect.x1);
    const x1 = Math.max(state.boxSelect.x0, state.boxSelect.x1);
    const y0 = Math.min(state.boxSelect.y0, state.boxSelect.y1);
    const y1 = Math.max(state.boxSelect.y0, state.boxSelect.y1);
    Object.assign(state.boxSelect.overlayEl.style, {
      left: x0 + "px",
      top: y0 + "px",
      width: (x1 - x0) + "px",
      height: (y1 - y0) + "px",
    });
  }

  function runBoxSelect(x0, y0, x1, y1, additive) {
    const THREE = window.THREE;
    const cam = getCamera();
    const dbg = getDebugMeshes();
    if (!THREE || !cam || !dbg.length) return;
    const cv = window.psoGetCanvas && window.psoGetCanvas();
    if (!cv) return;
    const rect = cv.getBoundingClientRect();
    const ndc0 = new THREE.Vector2(
      ((x0 - rect.left) / rect.width) * 2 - 1,
      -((y1 - rect.top) / rect.height) * 2 + 1,  // y0/y1 swapped because NDC y points up
    );
    const ndc1 = new THREE.Vector2(
      ((x1 - rect.left) / rect.width) * 2 - 1,
      -((y0 - rect.top) / rect.height) * 2 + 1,
    );
    const minX = Math.min(ndc0.x, ndc1.x), maxX = Math.max(ndc0.x, ndc1.x);
    const minY = Math.min(ndc0.y, ndc1.y), maxY = Math.max(ndc0.y, ndc1.y);
    if (!additive) state.selection.clear();
    const proj = new THREE.Vector3();
    const tmp = new THREE.Vector3();
    for (let s = 0; s < dbg.length; s++) {
      const e = dbg[s];
      if (!e || !e.mesh || !e.mesh.geometry) continue;
      const mesh = e.mesh;
      mesh.updateMatrixWorld(true);
      const pa = mesh.geometry.getAttribute("position");
      if (!pa) continue;
      let set = state.selection.get(s);
      for (let v = 0; v < pa.count; v++) {
        tmp.set(pa.array[v * 3], pa.array[v * 3 + 1], pa.array[v * 3 + 2])
          .applyMatrix4(mesh.matrixWorld);
        proj.copy(tmp).project(cam);
        if (proj.z >= 1.0) continue;
        if (proj.x >= minX && proj.x <= maxX && proj.y >= minY && proj.y <= maxY) {
          if (!set) { set = new Set(); state.selection.set(s, set); }
          set.add(v);
        }
      }
      if (set && !set.size) state.selection.delete(s);
    }
    rebuildMarkers();
    placeProxyAtCentroid();
    updateStatus();
  }

  // ---- pointer event wiring --------------------------------------
  let _pointerWired = false;
  function wirePointerHandlers() {
    if (_pointerWired) return;
    const cv = window.psoGetCanvas && window.psoGetCanvas();
    if (!cv) return;
    cv.addEventListener("pointerdown", function (ev) {
      if (!state.enabled) return;
      if (ev.shiftKey && ev.button === 0) {
        onBoxSelectStart(ev);
        return;
      }
    });
    document.addEventListener("pointermove", onBoxSelectMove);
    document.addEventListener("pointerup", function (ev) {
      onBoxSelectEnd(ev);
    });
    cv.addEventListener("click", onCanvasClick);
    _pointerWired = true;
  }

  // ---- save-back ---------------------------------------------------
  function resolveModelPath() {
    if (state.modelPath) return state.modelPath;
    let mp = null;
    if (window.psoGetTextureBinding) {
      const b = window.psoGetTextureBinding();
      if (b && b.archive) mp = b.archive;
    }
    if (!mp && window.psoGetCurrentTextureArchive) {
      mp = window.psoGetCurrentTextureArchive();
    }
    state.modelPath = mp;
    return mp;
  }

  function buildEditPayload() {
    // Encodes the current per-submesh displacement vs the original
    // position attribute snapshot.
    const dbg = getDebugMeshes();
    const submeshes = [];
    for (const e of dbg) {
      if (!e || !e.mesh || !e.mesh.geometry) continue;
      const pa = e.mesh.geometry.getAttribute("position");
      if (!pa) continue;
      const idx = e.idx | 0;
      // Compare to original (sculpt panel keeps original; if absent
      // we pull from a snapshot we capture on edit-mode entry).
      const orig = state.originalPositions ? state.originalPositions.get(idx) : null;
      if (!orig) continue;
      const dispXyz = [];
      const indices = [];
      for (let v = 0; v < pa.count; v++) {
        const dx = pa.array[v * 3] - orig[v * 3];
        const dy = pa.array[v * 3 + 1] - orig[v * 3 + 1];
        const dz = pa.array[v * 3 + 2] - orig[v * 3 + 2];
        if (dx !== 0 || dy !== 0 || dz !== 0) {
          indices.push(v);
          dispXyz.push(dx, dy, dz);
        }
      }
      if (indices.length) {
        submeshes.push({
          submesh_idx: idx,
          material_id: (e.mesh.userData && e.mesh.userData.materialId) | 0,
          vertex_count: pa.count,
          indices: indices,
          displacement: dispXyz,
        });
      }
    }
    return submeshes;
  }

  function captureOriginalPositions() {
    const dbg = getDebugMeshes();
    state.originalPositions = new Map();
    for (const e of dbg) {
      if (!e || !e.mesh || !e.mesh.geometry) continue;
      const pa = e.mesh.geometry.getAttribute("position");
      if (!pa) continue;
      state.originalPositions.set(e.idx | 0, new Float32Array(pa.array));
    }
  }

  async function saveEdits() {
    setStatus("busy", "saving…");
    const modelPath = resolveModelPath();
    if (!modelPath) {
      setStatus("err", "no model in scope");
      return;
    }
    const submeshes = buildEditPayload();
    if (!submeshes.length) {
      setStatus("ok", "no edits to save");
      return;
    }
    const payload = {
      model_path: modelPath,
      submeshes,
    };
    try {
      const r = await fetch("/api/protools/save_vertex_transforms", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!r.ok) {
        const t = await r.text();
        setStatus("err", "save failed: " + r.status);
        console.error("[edit] save failed:", t);
        return;
      }
      const data = await r.json();
      state.sourceSha = data.sha || null;
      state.dirty = false;
      setStatus("ok", "saved (" + submeshes.length + " submeshes; sha " +
                (data.sha ? data.sha.slice(0, 12) : "?") + ")");
    } catch (e) {
      setStatus("err", "save threw: " + (e.message || e));
      console.error(e);
    }
  }

  function setStatus(kind, msg) {
    if (!state.statusEl) return;
    state.statusEl.textContent = msg;
    state.statusEl.style.color =
      kind === "err" ? "#ef6b6b" :
      kind === "ok" ? "#7fdba0" :
      kind === "busy" ? "#f0c060" :
      "#8a96a6";
  }

  function updateStatus() {
    const n = totalSelectedCount();
    if (state.selCountEl) state.selCountEl.textContent = String(n);
    let line = `selected: ${n}  |  mode: ${state.selMode}  |  tfm: ${state.activeTfm}`;
    if (state.dirty) line += "  |  unsaved edits";
    setStatus(state.dirty ? "busy" : "info", line);
  }

  // ---- mode toggles -----------------------------------------------
  function setEditMode(active) {
    if (state.enabled === !!active) return;
    state.enabled = !!active;
    // Notify model_viewer so its pointer-drag-orbit yields LMB.
    window.__psoEditModeActive = state.enabled;
    if (state.enabled) {
      captureOriginalPositions();
      wirePointerHandlers();
      ensureMarkersGroup();
      rebuildMarkers();
      // Hotkey scope.
      if (window.psoHotkeys) window.psoHotkeys.setActiveScope("edit");
    } else {
      detachGizmo();
      clearMarkers();
      if (window.psoHotkeys) window.psoHotkeys.setActiveScope("global");
    }
    // Update top-bar button.
    const btn = document.getElementById("btnEditMode");
    if (btn) {
      btn.classList.toggle("active", state.enabled);
      btn.setAttribute("aria-pressed", state.enabled ? "true" : "false");
    }
    if (window.bus && window.bus.emit) {
      try { window.bus.emit("edit.modeChanged", { enabled: state.enabled }); } catch (_e) {}
    }
    updateStatus();
  }

  function setTransformMode(m) {
    if (m !== TFM_TRANSLATE && m !== TFM_ROTATE && m !== TFM_SCALE) return;
    state.activeTfm = m;
    if (window.psoTransformGizmo && window.psoTransformGizmo.isAttached()) {
      window.psoTransformGizmo.setMode(m);
    }
    // Update toolbar buttons.
    const tb = state.bodyEl && state.bodyEl.querySelectorAll(".pso-edit-tfm");
    if (tb) tb.forEach(function (b) {
      b.classList.toggle("active", b.dataset.tfm === m);
    });
    updateStatus();
  }

  function setSelMode(m) {
    if (!VALID_SELMODES.includes(m)) return;
    if (m !== SELMODE_VERTEX) {
      // Edge / face are P1; for now just inform.
      setStatus("err", m + " selection coming in P1; staying in vertex");
      return;
    }
    state.selMode = m;
    const tb = state.bodyEl && state.bodyEl.querySelectorAll(".pso-edit-sel");
    if (tb) tb.forEach(function (b) {
      b.classList.toggle("active", b.dataset.sel === m);
    });
    updateStatus();
  }

  // ---- hotkeys -----------------------------------------------------
  function installHotkeys() {
    if (!window.psoHotkeys || state.__hkInstalled) return;
    state.__hkInstalled = true;
    const bind = window.psoHotkeys.bind;
    bind("Tab", "edit-mode-toggle", function () {
      setEditMode(!state.enabled);
    }, { scope: "global", label: "Toggle edit-mode (Tab)" });
    bind("g", "edit-translate", function () {
      if (!state.enabled) return false;
      setTransformMode(TFM_TRANSLATE);
    }, { scope: "edit", label: "Translate (G)" });
    bind("r", "edit-rotate", function () {
      if (!state.enabled) return false;
      setTransformMode(TFM_ROTATE);
    }, { scope: "edit", label: "Rotate (R)" });
    bind("s", "edit-scale", function () {
      if (!state.enabled) return false;
      setTransformMode(TFM_SCALE);
    }, { scope: "edit", label: "Scale (S)" });
    bind("v", "edit-sel-vertex", function () {
      if (!state.enabled) return false;
      setSelMode(SELMODE_VERTEX);
    }, { scope: "edit", label: "Vertex select mode (V)" });
    bind("a", "edit-select-all", function () {
      if (!state.enabled) return false;
      selectAll();
    }, { scope: "edit", label: "Select all (A)" });
    bind("Alt+a", "edit-select-none", function () {
      if (!state.enabled) return false;
      clearSelection();
    }, { scope: "edit", label: "Deselect all (Alt+A)" });
    bind("Ctrl+s", "edit-save", function () {
      if (!state.enabled) return false;
      saveEdits();
    }, { scope: "edit", label: "Save vertex edits (Ctrl+S)" });
  }

  // ---- tab body rendering ------------------------------------------
  function renderTabBody(bodyEl) {
    ensureStyleInjected();
    state.bodyEl = bodyEl;
    bodyEl.innerHTML =
      '<div class="pso-edit-toolbar">' +
        '<button id="psoEditToggle" class="pso-edit-modebtn ' + (state.enabled ? "active" : "") + '"' +
          ' title="toggle edit-mode (Tab)">' + (state.enabled ? "Edit Mode: ON" : "Edit Mode: OFF") + '</button>' +

        '<span style="border-left:1px solid #2a313a; height:18px; margin:0 4px"></span>' +

        '<span class="dim">select:</span>' +
        '<div class="pso-edit-pillrow">' +
          '<button class="pso-edit-sel active" data-sel="vertex" title="vertex select (V)">V</button>' +
          '<button class="pso-edit-sel disabled" data-sel="edge" title="edge select (E) — P1">E</button>' +
          '<button class="pso-edit-sel disabled" data-sel="face" title="face select (F) — P1">F</button>' +
        '</div>' +

        '<span style="border-left:1px solid #2a313a; height:18px; margin:0 4px"></span>' +

        '<span class="dim">tfm:</span>' +
        '<div class="pso-edit-pillrow">' +
          '<button class="pso-edit-tfm active" data-tfm="translate" title="translate (G)">G</button>' +
          '<button class="pso-edit-tfm" data-tfm="rotate" title="rotate (R)">R</button>' +
          '<button class="pso-edit-tfm" data-tfm="scale" title="scale (S)">S</button>' +
        '</div>' +

        '<span class="grow" style="flex:1"></span>' +
        '<span class="dim">selected: <strong id="psoEditSelCount">0</strong></span>' +
        '<button id="psoEditSelAll" type="button" title="select all (A)">all</button>' +
        '<button id="psoEditSelNone" type="button" title="deselect all (Alt+A)">none</button>' +
        '<button id="psoEditSave" type="button" title="save vertex transforms (Ctrl+S)">save edits</button>' +
        '<button id="psoEditDiscard" type="button" class="danger" title="revert all uncommitted edits">revert</button>' +
      '</div>' +
      '<div class="pso-edit-status" id="psoEditStatus"></div>' +
      '<div class="pso-edit-help">' +
        '<strong>Edit-mode (Blender-style)</strong><br>' +
        '<kbd>Tab</kbd> toggle, ' +
        '<kbd>V</kbd>/<kbd>E</kbd>/<kbd>F</kbd> select mode, ' +
        '<kbd>G</kbd>/<kbd>R</kbd>/<kbd>S</kbd> translate/rotate/scale, ' +
        '<kbd>A</kbd> all, <kbd>Alt+A</kbd> none, ' +
        '<kbd>Ctrl+S</kbd> save, <kbd>Ctrl+Z</kbd> undo. ' +
        '<br><span class="dim">Click a vertex to select; ' +
        'Shift+click to add to selection; Shift+drag for box-select; ' +
        'gizmo handles transform.</span>' +
      '</div>';

    state.statusEl = bodyEl.querySelector("#psoEditStatus");
    state.selCountEl = bodyEl.querySelector("#psoEditSelCount");
    bodyEl.querySelector("#psoEditToggle").addEventListener("click", function () {
      setEditMode(!state.enabled);
      // Update label.
      this.textContent = state.enabled ? "Edit Mode: ON" : "Edit Mode: OFF";
      this.classList.toggle("active", state.enabled);
    });
    bodyEl.querySelectorAll(".pso-edit-sel").forEach(function (b) {
      b.addEventListener("click", function () {
        if (b.classList.contains("disabled")) return;
        setSelMode(b.dataset.sel);
      });
    });
    bodyEl.querySelectorAll(".pso-edit-tfm").forEach(function (b) {
      b.addEventListener("click", function () {
        setTransformMode(b.dataset.tfm);
      });
    });
    bodyEl.querySelector("#psoEditSelAll").addEventListener("click", selectAll);
    bodyEl.querySelector("#psoEditSelNone").addEventListener("click", clearSelection);
    bodyEl.querySelector("#psoEditSave").addEventListener("click", saveEdits);
    bodyEl.querySelector("#psoEditDiscard").addEventListener("click", revertAllEdits);

    installHotkeys();
    wirePointerHandlers();
    updateStatus();
  }

  function revertAllEdits() {
    if (!state.originalPositions) {
      setStatus("err", "no snapshot to revert from (toggle edit-mode first)");
      return;
    }
    const dbg = getDebugMeshes();
    for (const e of dbg) {
      if (!e || !e.mesh || !e.mesh.geometry) continue;
      const pa = e.mesh.geometry.getAttribute("position");
      if (!pa) continue;
      const orig = state.originalPositions.get(e.idx | 0);
      if (!orig) continue;
      pa.array.set(orig);
      pa.needsUpdate = true;
      e.mesh.geometry.computeVertexNormals();
    }
    state.dirty = false;
    refreshMarkerPositions();
    placeProxyAtCentroid();
    updateStatus();
    setStatus("ok", "reverted to original positions");
  }

  // ---- top-toolbar Edit Mode button (added by index.html) ---------
  function wireTopBarButton() {
    const btn = document.getElementById("btnEditMode");
    if (!btn || btn.__psoEditWired) return;
    btn.__psoEditWired = true;
    btn.addEventListener("click", function () {
      setEditMode(!state.enabled);
    });
  }

  // ---- public API -----------------------------------------------
  window.psoEditPanel = Object.freeze({
    setEditMode: setEditMode,
    isEditMode: function () { return state.enabled; },
    setTransformMode: setTransformMode,
    setSelMode: setSelMode,
    selectAll: selectAll,
    clearSelection: clearSelection,
    selectionCount: totalSelectedCount,
    saveEdits: saveEdits,
    revertAllEdits: revertAllEdits,
    getSelection: function () {
      // Snapshot for tests.
      const out = {};
      for (const [s, set] of state.selection) {
        out[s] = Array.from(set);
      }
      return out;
    },
    setSelection: function (snap) {
      // Used by tests + scripted reload.
      state.selection.clear();
      if (!snap || typeof snap !== "object") return;
      for (const k of Object.keys(snap)) {
        const list = snap[k];
        if (!Array.isArray(list)) continue;
        state.selection.set(k | 0, new Set(list.map(function (v) { return v | 0; })));
      }
      rebuildMarkers();
      placeProxyAtCentroid();
      updateStatus();
    },
  });

  function tryRegisterTab() {
    if (typeof window.psoTexturePanelRegisterTab === "function" &&
        typeof window.psoTexturePanelAddTabButton === "function") {
      window.psoTexturePanelRegisterTab(TAB_NAME, renderTabBody);
      window.psoTexturePanelAddTabButton(TAB_NAME, TAB_LABEL, TAB_TITLE);
      return true;
    }
    return false;
  }

  function init() {
    wireTopBarButton();
    if (!tryRegisterTab()) {
      let attempts = 0;
      const t = setInterval(function () {
        if (tryRegisterTab() || ++attempts > 40) clearInterval(t);
      }, 250);
    }
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
  // Top-bar button mounts late (header is in DOM at parse).
  setTimeout(wireTopBarButton, 200);
})();
