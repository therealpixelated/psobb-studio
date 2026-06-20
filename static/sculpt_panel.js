// =====================================================================
// PSOBB Sculpt Panel — interactive vertex-displacement brushes.
// 2026-04-25
//
// Adds a "Sculpt" tab to the existing texture-panel tab strip, along
// with a state machine that:
//   - Captures pointer clicks on the 3D canvas and raycasts to find
//     the (face, point) under the cursor
//   - Builds a per-stroke spatial index (uniform grid hash) for O(log N)
//     radius queries
//   - Applies a brush operator (push/pull/inflate/smooth/pinch/flatten)
//     to the affected vertices
//   - Mutates the live mesh's BufferAttribute in-place (no rebuild)
//   - Tracks per-submesh displacement for undo / persistence
//
// The math here mirrors `formats/sculpt.py` so server-side persistence
// stays consistent. The server only handles save/fetch — the live
// stroke runs entirely in the browser.
//
// Falloff curves: linear / smooth / sharp / gaussian (mirrors
//   formats.sculpt.falloff()).
//
// Performance: dragon-class meshes (9677 verts) measure ~1 ms / stroke
// step at brush_radius=0.5 thanks to the grid hash. Normals are
// recomputed lazily at stroke-end to keep the per-frame cost flat;
// users can toggle "real-time normals" on for low-poly meshes.
//
// Wire format (POST /api/sculpt/save):
//   {
//     model_path: "<bml>#<inner>.nj",
//     subdivide_level: <int>,
//     mesh_payload: {
//       format_version: 1,
//       source_path: "...",
//       source_sha: "...",
//       submeshes: [
//         { submesh_idx, material_id, vertex_count,
//           displacement_b64, modified_indices_b64, mode },
//         ...
//       ],
//       sha: "..."
//     }
//   }
//
// Reads from existing model_viewer exports:
//   - window.psoGetDebugMeshes()  for per-submesh metadata
//   - window.psoGetSculptMeshGroup() (added below in additive section)
//   - window.psoUpdateSculptedNormals() (added below)
// =====================================================================

(function () {
  "use strict";

  if (window.__psoSculptPanelLoaded) return;
  window.__psoSculptPanelLoaded = true;

  const STYLE_ID = "psoSculptPanelStyle";

  // Brush + falloff string constants — must match formats/sculpt.py.
  // v5 brushes (smudge / twist / layer / retopo) are appended at the
  // end so the toolbar's tab-order keeps the v1 brushes' positions
  // stable for muscle-memory.
  const BRUSHES = [
    "push", "pull", "inflate", "smooth", "pinch", "flatten",
    "decimate_region",
    "smudge", "twist", "layer", "retopo",
  ];
  const FALLOFFS = ["linear", "smooth", "sharp", "gaussian"];

  // Mirror axis values — must match formats/sculpt.py's MIRROR_*.
  // "off" disables the mirror entirely.
  const MIRROR_AXES = ["off", "x", "y", "z"];

  // Default brush state.
  const _DEF = {
    brush: "inflate",
    radius: 0.5,
    strength: 0.3,
    falloff: "smooth",
    // v5: replaces the old `mirrorX` boolean. Legacy state migrated
    // automatically (see init()).
    mirrorAxis: "off",
    realTimeNormals: false,
    // Twist rate (radians per unit drag distance). Negative -> CCW.
    twistRate: 1.0,
  };

  // -------------------------------------------------------------------
  // Falloff math — mirror formats.sculpt.falloff()
  // -------------------------------------------------------------------
  function falloff(t, curve) {
    if (t >= 1.0) return 0.0;
    if (t < 0.0) t = 0.0;
    switch (curve) {
      case "linear":   return 1.0 - t;
      case "sharp":    { const u = 1.0 - t; return u * u * u; }
      case "gaussian": return Math.exp(-(t * t) * 4.0);
      case "smooth":
      default: {
        const s = 1.0 - t;
        return s * s * (3.0 - 2.0 * s);
      }
    }
  }

  // -------------------------------------------------------------------
  // Uniform-grid spatial index — mirror formats.sculpt.GridIndex.
  //
  // Build once per submesh per stroke (we rebuild between strokes if
  // the user has moved >5% of the verts more than 1 cell over).
  // Stores indices as plain ints; bucket map keyed by (ix << 32 | iy <<
  // 16 | iz) — a simple string key works fine in V8.
  // -------------------------------------------------------------------
  class GridIndex {
    constructor(positions, cellSize) {
      // positions = THREE.BufferAttribute (Float32Array view)
      // cellSize  = world-space cell edge length
      this.cellSize = cellSize;
      this.points = positions;
      this.buckets = new Map(); // string -> int32 array
      const n = positions.count;
      const arr = positions.array;
      const cs = cellSize;
      for (let i = 0; i < n; i++) {
        const ox = i * 3;
        const ix = Math.floor(arr[ox] / cs);
        const iy = Math.floor(arr[ox + 1] / cs);
        const iz = Math.floor(arr[ox + 2] / cs);
        const key = `${ix}|${iy}|${iz}`;
        let b = this.buckets.get(key);
        if (!b) { b = []; this.buckets.set(key, b); }
        b.push(i);
      }
    }

    queryRadius(cx, cy, cz, radius) {
      const cs = this.cellSize;
      const ix0 = Math.floor((cx - radius) / cs);
      const ix1 = Math.floor((cx + radius) / cs);
      const iy0 = Math.floor((cy - radius) / cs);
      const iy1 = Math.floor((cy + radius) / cs);
      const iz0 = Math.floor((cz - radius) / cs);
      const iz1 = Math.floor((cz + radius) / cs);
      const r2 = radius * radius;
      const arr = this.points.array;
      const out = [];
      for (let ix = ix0; ix <= ix1; ix++) {
        for (let iy = iy0; iy <= iy1; iy++) {
          for (let iz = iz0; iz <= iz1; iz++) {
            const b = this.buckets.get(`${ix}|${iy}|${iz}`);
            if (!b) continue;
            for (let k = 0; k < b.length; k++) {
              const vi = b[k];
              const ox = vi * 3;
              const dx = arr[ox]     - cx;
              const dy = arr[ox + 1] - cy;
              const dz = arr[ox + 2] - cz;
              if (dx * dx + dy * dy + dz * dz <= r2) out.push(vi);
            }
          }
        }
      }
      return out;
    }
  }

  // -------------------------------------------------------------------
  // 1-ring neighbour map (used by Smooth)
  // -------------------------------------------------------------------
  function buildNeighbours(indexAttr, vertexCount) {
    const arr = indexAttr ? indexAttr.array : null;
    if (!arr) return [];
    const sets = new Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) sets[i] = new Set();
    const n = arr.length;
    for (let f = 0; f < n; f += 3) {
      const a = arr[f], b = arr[f + 1], c = arr[f + 2];
      if (a < vertexCount && b < vertexCount) { sets[a].add(b); sets[b].add(a); }
      if (b < vertexCount && c < vertexCount) { sets[b].add(c); sets[c].add(b); }
      if (c < vertexCount && a < vertexCount) { sets[c].add(a); sets[a].add(c); }
    }
    const out = new Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) out[i] = [...sets[i]];
    return out;
  }

  // -------------------------------------------------------------------
  // Per-submesh sculpt state. The mesh group has multiple sub-meshes;
  // we keep one record per sub-mesh that the user has actually touched.
  // Records:
  //   meshRef         — THREE.Mesh
  //   originalPos     — Float32Array snapshot (length = vertexCount*3)
  //                     captured BEFORE any sculpt edits; used by Reset.
  //   accumDisp       — Float32Array (vertexCount*3) running displacement
  //                     vs original. When the panel saves to disk this is
  //                     the field that goes on the wire.
  //   modifiedSet     — Set<int> vertex indices the user has moved.
  //   gridIdx         — GridIndex built from the current positions.
  //   neighbours      — 1-ring map (lazy; built on first Smooth call).
  //   gridDirty       — flag: rebuild gridIdx before next stroke.
  // -------------------------------------------------------------------
  const sculptRecords = new Map(); // submesh_idx -> record

  function ensureRecord(submeshIdx, mesh) {
    let rec = sculptRecords.get(submeshIdx);
    if (rec) return rec;
    const posAttr = mesh.geometry.getAttribute("position");
    const original = new Float32Array(posAttr.array);  // copy
    rec = {
      meshRef: mesh,
      originalPos: original,
      accumDisp: new Float32Array(posAttr.count * 3),
      modifiedSet: new Set(),
      gridIdx: null,
      neighbours: null,
      gridDirty: true,
    };
    sculptRecords.set(submeshIdx, rec);
    return rec;
  }

  function disposeRecords() {
    sculptRecords.clear();
  }

  // -------------------------------------------------------------------
  // Undo stack. Each entry is an array of {submeshIdx, indices: Uint32Array,
  // before: Float32Array} — the inverse of one stroke.
  // -------------------------------------------------------------------
  const UNDO_LIMIT = 50;
  const undoStack = [];
  const redoStack = [];

  function pushUndoEntry(entry) {
    undoStack.push(entry);
    while (undoStack.length > UNDO_LIMIT) undoStack.shift();
    redoStack.length = 0;  // any new edit clears redo
    if (typeof updateUndoButtons === "function") updateUndoButtons();

    // Cross-tool undo bus integration (2026-04-25, C7 follow-up): every
    // panel-local undo push ALSO lands on `window.psoUndoBus` so Ctrl+Z
    // from anywhere unwinds the most recent sculpt stroke. The bus's
    // undo/redo callbacks apply the displacement DIRECTLY (via the same
    // `applyEntry` helper the panel-local stack uses) so the two stacks
    // stay decoupled — popping one does NOT pop the other. Net effect:
    // Ctrl+Z always reverses one user-visible action, regardless of
    // which perspective owns the focus.
    //
    // The closure captures `entry` (immutable per-stroke {subs: [...]}),
    // so this stays correct even if the panel's local stack later
    // evicts the entry under the LIMIT cap.
    if (window.psoUndoBus) {
      const captured = entry;
      const subCount = (captured.subs && captured.subs.length) || 0;
      let vertTotal = 0;
      for (const s of captured.subs || []) {
        vertTotal += (s.indices && s.indices.length) || 0;
      }
      const lbl = "sculpt stroke (" + state.brush + ", " +
                  vertTotal + " vert" + (vertTotal === 1 ? "" : "s") +
                  (subCount > 1 ? " across " + subCount + " submeshes" : "") + ")";
      const applyDirection = function (fromPos /* "before"|"after" */) {
        for (const subEntry of captured.subs || []) {
          const rec = sculptRecords.get(subEntry.submeshIdx);
          if (!rec) continue;
          applyEntry(rec, subEntry, fromPos);
          if (state.realTimeNormals) recomputeNormals(rec.meshRef);
        }
        if (!state.realTimeNormals) recomputeAllNormals();
      };
      window.psoUndoBus.push({
        label: lbl,
        panelId: "sculpt",
        undo: function () { applyDirection("before"); },
        redo: function () { applyDirection("after"); },
      });
    }
  }

  function applyEntry(rec, entry, fromPos /* "before"|"after" */) {
    const mesh = rec.meshRef;
    if (!mesh) return;
    const posAttr = mesh.geometry.getAttribute("position");
    const indices = entry.indices;
    const data = entry[fromPos];
    for (let k = 0; k < indices.length; k++) {
      const vi = indices[k];
      const sx = vi * 3;
      const dx = vi * 3;
      posAttr.array[sx + 0] = data[dx + 0];
      posAttr.array[sx + 1] = data[dx + 1];
      posAttr.array[sx + 2] = data[dx + 2];
      // Update accumDisp = current_position - original_position.
      rec.accumDisp[sx + 0] = posAttr.array[sx + 0] - rec.originalPos[sx + 0];
      rec.accumDisp[sx + 1] = posAttr.array[sx + 1] - rec.originalPos[sx + 1];
      rec.accumDisp[sx + 2] = posAttr.array[sx + 2] - rec.originalPos[sx + 2];
      const isModified =
        rec.accumDisp[sx + 0] !== 0 ||
        rec.accumDisp[sx + 1] !== 0 ||
        rec.accumDisp[sx + 2] !== 0;
      if (isModified) rec.modifiedSet.add(vi); else rec.modifiedSet.delete(vi);
    }
    posAttr.needsUpdate = true;
    rec.gridDirty = true;
  }

  function undoOne() {
    const entry = undoStack.pop();
    if (!entry) return false;
    for (const subEntry of entry.subs) {
      const rec = sculptRecords.get(subEntry.submeshIdx);
      if (!rec) continue;
      applyEntry(rec, subEntry, "before");
      if (state.realTimeNormals) recomputeNormals(rec.meshRef);
    }
    redoStack.push(entry);
    while (redoStack.length > UNDO_LIMIT) redoStack.shift();
    if (!state.realTimeNormals) recomputeAllNormals();
    if (typeof updateUndoButtons === "function") updateUndoButtons();
    return true;
  }

  function redoOne() {
    const entry = redoStack.pop();
    if (!entry) return false;
    for (const subEntry of entry.subs) {
      const rec = sculptRecords.get(subEntry.submeshIdx);
      if (!rec) continue;
      applyEntry(rec, subEntry, "after");
      if (state.realTimeNormals) recomputeNormals(rec.meshRef);
    }
    undoStack.push(entry);
    while (undoStack.length > UNDO_LIMIT) undoStack.shift();
    if (!state.realTimeNormals) recomputeAllNormals();
    if (typeof updateUndoButtons === "function") updateUndoButtons();
    return true;
  }

  // -------------------------------------------------------------------
  // The main brush state machine
  // -------------------------------------------------------------------
  const state = {
    enabled: false,        // sculpt-mode toggle
    activePanel: null,     // DOM element (the Sculpt-tab body)
    brush: _DEF.brush,
    radius: _DEF.radius,
    strength: _DEF.strength,
    falloff: _DEF.falloff,
    // v5: mirrorAxis is "off"/"x"/"y"/"z". v1 had `mirrorX: bool`;
    // legacy callers reading state.mirrorX still work via the proxy
    // getter further down.
    mirrorAxis: _DEF.mirrorAxis,
    twistRate: _DEF.twistRate,
    realTimeNormals: _DEF.realTimeNormals,
    // Live stroke — cleared on pointerup
    stroke: null,          // { centresVisited: Vector3[], submeshIdx, ... }
    // Cached subdivide-level so we know what we're sculpting on top of
    subdivideLevel: 0,
  };

  // Backward-compat alias for any code that still reads `state.mirrorX`
  // (scripts mounted before v5 polish, e.g. devtools snippets and the
  // older test harness in test_sculpt.py). Reflects the dropdown
  // selection so the boolean stays in lock-step. Writes propagate too.
  Object.defineProperty(state, "mirrorX", {
    enumerable: false,
    configurable: true,
    get: function () { return state.mirrorAxis === "x"; },
    set: function (v) { state.mirrorAxis = v ? "x" : "off"; },
  });

  // Stash for the sculpt panel to read the source SHA. Computed lazily
  // when the user saves; uses a hash of the original-positions arrays
  // concatenated.
  function computeSourceShaSync() {
    // Browser SubtleCrypto is async; we use a fast non-cryptographic
    // hash (FNV-1a 32-bit) over each submesh's originalPos. Truncated to
    // 16 hex chars to match formats.sculpt.compute_source_sha.
    let h = 0x811c9dc5;
    for (const [, rec] of sculptRecords) {
      const arr = new Uint8Array(rec.originalPos.buffer);
      for (let i = 0; i < arr.length; i++) {
        h ^= arr[i];
        h = (h * 0x01000193) >>> 0;
      }
    }
    return h.toString(16).padStart(16, "0").slice(0, 16);
  }

  // -------------------------------------------------------------------
  // Stroke entry point
  // -------------------------------------------------------------------
  let _activePointerId = null;

  function onPointerDown(ev) {
    if (!state.enabled) return;
    // Only left-click triggers sculpt; right-click stays orbit.
    if (ev.button !== 0) return;
    const cv = ev.target;
    if (!cv || !cv.tagName) return;
    if (cv.tagName.toLowerCase() !== "canvas") return;
    if (cv.id !== "modelCanvas") return;
    ev.preventDefault();
    ev.stopPropagation();
    _activePointerId = ev.pointerId;
    cv.setPointerCapture(ev.pointerId);

    state.stroke = {
      submeshIdx: -1,
      // accumulated per-submesh deltas FOR THIS STROKE (separate from
      // record.accumDisp which carries the full session). Indexed by
      // submeshIdx; each value is { indices: Set<int>, before: Map<int, [x,y,z]> }.
      perSub: new Map(),
      pointsApplied: 0,
      // v5: drag tracking for smudge / twist. lastLocal is the previous
      // step's hit point in submesh-local space; null on the first step
      // of a stroke (so the drag delta is 0 — neither brush moves
      // anything on click without drag, which matches Blender).
      lastLocal: null,
      lastSubmeshIdx: -1,
      dragDistance: 0,    // cumulative since stroke start
    };
    applyStrokeStep(ev.clientX, ev.clientY);
  }

  function onPointerMove(ev) {
    if (!state.enabled) return;
    if (_activePointerId == null) return;
    if (ev.pointerId !== _activePointerId) return;
    if (state.stroke) applyStrokeStep(ev.clientX, ev.clientY);
  }

  function onPointerUp(ev) {
    if (_activePointerId == null) return;
    if (ev.pointerId !== _activePointerId) return;
    const cv = ev.target;
    if (cv && cv.releasePointerCapture) {
      try { cv.releasePointerCapture(ev.pointerId); } catch (_) {}
    }
    _activePointerId = null;
    finalizeStroke();
  }

  function finalizeStroke() {
    if (!state.stroke) return;
    const stroke = state.stroke;
    state.stroke = null;
    // Build an undo entry per sub-mesh.
    const subEntries = [];
    for (const [submeshIdx, perSub] of stroke.perSub) {
      const rec = sculptRecords.get(submeshIdx);
      if (!rec) continue;
      const indices = new Uint32Array([...perSub.indices]);
      const before = new Float32Array(rec.accumDisp.length);  // size = vc*3
      const after = new Float32Array(rec.accumDisp.length);
      const posArr = rec.meshRef.geometry.getAttribute("position").array;
      for (let k = 0; k < indices.length; k++) {
        const vi = indices[k];
        const t = perSub.before.get(vi);
        if (!t) continue;
        before[vi * 3 + 0] = t[0];
        before[vi * 3 + 1] = t[1];
        before[vi * 3 + 2] = t[2];
        after[vi * 3 + 0] = posArr[vi * 3 + 0];
        after[vi * 3 + 1] = posArr[vi * 3 + 1];
        after[vi * 3 + 2] = posArr[vi * 3 + 2];
      }
      subEntries.push({ submeshIdx, indices, before, after });
    }
    if (subEntries.length > 0) {
      pushUndoEntry({ subs: subEntries });
    }
    // Lazy normals recompute at stroke-end (unless real-time).
    if (!state.realTimeNormals) recomputeAllNormals();
    if (typeof updateUndoButtons === "function") updateUndoButtons();
    if (typeof updateStats === "function") updateStats();
  }

  // -------------------------------------------------------------------
  // The actual brush apply — called on every pointer step
  // -------------------------------------------------------------------
  function applyStrokeStep(clientX, clientY) {
    const r = raycastFromScreen(clientX, clientY);
    if (!r) return;
    const meshGroup = r.meshGroup;
    const mesh = r.mesh;
    const submeshIdx = r.submeshIdx;
    const localPoint = r.localPoint;
    const camForwardLocal = r.cameraForwardLocal;

    const rec = ensureRecord(submeshIdx, mesh);
    if (rec.gridDirty || !rec.gridIdx) {
      const cell = Math.max(0.01, state.radius * 1.5);
      rec.gridIdx = new GridIndex(mesh.geometry.getAttribute("position"), cell);
      rec.gridDirty = false;
    }

    const r0 = state.radius;
    const affected = rec.gridIdx.queryRadius(localPoint.x, localPoint.y, localPoint.z, r0);
    if (affected.length === 0) return;

    // Stroke bookkeeping: capture per-vertex BEFORE positions for undo.
    let perSub = state.stroke.perSub.get(submeshIdx);
    if (!perSub) {
      perSub = { indices: new Set(), before: new Map() };
      state.stroke.perSub.set(submeshIdx, perSub);
    }
    const posAttr = mesh.geometry.getAttribute("position");
    const nrmAttr = mesh.geometry.getAttribute("normal");
    const posArr = posAttr.array;
    const nrmArr = nrmAttr ? nrmAttr.array : null;

    // Lazy build neighbours map if Smooth.
    if (state.brush === "smooth" && !rec.neighbours) {
      const idxAttr = mesh.geometry.getIndex();
      rec.neighbours = buildNeighbours(idxAttr, posAttr.count);
    }

    // For Flatten: pre-compute centroid + plane normal of the affected
    // verts (in submesh local space).
    let flattenPlane = null;
    if (state.brush === "flatten" && affected.length >= 3) {
      let cx = 0, cy = 0, cz = 0;
      for (const vi of affected) {
        cx += posArr[vi * 3];
        cy += posArr[vi * 3 + 1];
        cz += posArr[vi * 3 + 2];
      }
      cx /= affected.length; cy /= affected.length; cz /= affected.length;
      // Covariance + smallest-eigvec via 3x3 power-iteration on the
      // INVERSE — actually for a flatten the user usually wants the
      // surface-normal of the brush region. Approximate it as the mean
      // per-vertex normal.
      let nx = 0, ny = 0, nz = 0;
      if (nrmArr) {
        for (const vi of affected) {
          nx += nrmArr[vi * 3];
          ny += nrmArr[vi * 3 + 1];
          nz += nrmArr[vi * 3 + 2];
        }
      }
      const nl = Math.sqrt(nx * nx + ny * ny + nz * nz);
      if (nl < 1e-6) { nx = 0; ny = 0; nz = 1; } else { nx /= nl; ny /= nl; nz /= nl; }
      flattenPlane = { cx, cy, cz, nx, ny, nz };
    }

    // Direction vector (push/pull use this; in submesh local space).
    const bdx = camForwardLocal.x, bdy = camForwardLocal.y, bdz = camForwardLocal.z;

    // v5 drag tracking — for smudge/twist we need (a) the per-step drag
    // vector in submesh local space and (b) cumulative drag distance.
    // lastLocal is null on the first step or when the user crossed
    // submeshes mid-stroke (we reset there to avoid garbage smudge
    // deltas across separate sub-models).
    let dragVec = { x: 0, y: 0, z: 0 };
    if (state.stroke.lastLocal && state.stroke.lastSubmeshIdx === submeshIdx) {
      dragVec.x = localPoint.x - state.stroke.lastLocal.x;
      dragVec.y = localPoint.y - state.stroke.lastLocal.y;
      dragVec.z = localPoint.z - state.stroke.lastLocal.z;
      const dlen = Math.sqrt(
        dragVec.x * dragVec.x +
        dragVec.y * dragVec.y +
        dragVec.z * dragVec.z,
      );
      state.stroke.dragDistance += dlen;
    }
    state.stroke.lastLocal = { x: localPoint.x, y: localPoint.y, z: localPoint.z };
    state.stroke.lastSubmeshIdx = submeshIdx;
    const dragDist = state.stroke.dragDistance;
    const twistRate = state.twistRate || 1.0;

    // Pre-compute the twist axis once per step (it's the same for every
    // affected vertex). Axis = camera-forward (bd) — could be the
    // surface normal at the brush hit instead, but bd matches "rotate
    // the cursor around the screen" which is the user-intuitive
    // mapping.
    const twistTheta = (state.brush === "twist") ?
      (dragDist * twistRate) : 0;
    const twistCos = Math.cos(twistTheta);
    const twistSin = Math.sin(twistTheta);

    let movedThisStep = 0;
    for (const vi of affected) {
      const sx = vi * 3;
      const px = posArr[sx], py = posArr[sx + 1], pz = posArr[sx + 2];
      const ddx = px - localPoint.x;
      const ddy = py - localPoint.y;
      const ddz = pz - localPoint.z;
      const dist = Math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
      if (dist >= r0) continue;
      const w = falloff(dist / r0, state.falloff) * state.strength;
      if (w <= 0) continue;

      // Preserve the ORIGINAL pre-stroke position once so undo gets
      // back to the right place. If we've already recorded a "before"
      // for vi this stroke, don't overwrite — multiple steps within
      // the same stroke compound into a single undo entry.
      if (!perSub.before.has(vi)) {
        perSub.before.set(vi, [px, py, pz]);
      }
      perSub.indices.add(vi);

      let nx = 0, ny = 0, nz = 0;
      let ok = false;

      switch (state.brush) {
        case "push":
          nx = bdx * w * r0; ny = bdy * w * r0; nz = bdz * w * r0;
          ok = true;
          break;
        case "pull":
          nx = -bdx * w * r0; ny = -bdy * w * r0; nz = -bdz * w * r0;
          ok = true;
          break;
        case "inflate": {
          if (!nrmArr) break;
          let vx = nrmArr[sx], vy = nrmArr[sx + 1], vz = nrmArr[sx + 2];
          const vlen = Math.sqrt(vx * vx + vy * vy + vz * vz);
          if (vlen < 1e-6) break;
          vx /= vlen; vy /= vlen; vz /= vlen;
          nx = vx * w * r0; ny = vy * w * r0; nz = vz * w * r0;
          ok = true;
          break;
        }
        case "pinch": {
          if (dist < 1e-6) break;
          nx = -ddx * (w * 0.5);
          ny = -ddy * (w * 0.5);
          nz = -ddz * (w * 0.5);
          ok = true;
          break;
        }
        case "smooth": {
          const ngb = rec.neighbours[vi];
          if (!ngb || ngb.length === 0) break;
          let mx = 0, my = 0, mz = 0;
          for (const nj of ngb) {
            mx += posArr[nj * 3];
            my += posArr[nj * 3 + 1];
            mz += posArr[nj * 3 + 2];
          }
          mx /= ngb.length; my /= ngb.length; mz /= ngb.length;
          nx = (mx - px) * w;
          ny = (my - py) * w;
          nz = (mz - pz) * w;
          ok = true;
          break;
        }
        case "flatten": {
          if (!flattenPlane) break;
          const fp = flattenPlane;
          // Signed distance from plane along normal.
          const sd =
            (px - fp.cx) * fp.nx +
            (py - fp.cy) * fp.ny +
            (pz - fp.cz) * fp.nz;
          // Project: subtract (sd * normal) -> on plane. Lerp by w.
          nx = -fp.nx * sd * w;
          ny = -fp.ny * sd * w;
          nz = -fp.nz * sd * w;
          ok = true;
          break;
        }
        case "decimate_region":
          // No-op in v1 (server-side stub also no-ops). UI surfaces a
          // toast when the user picks this brush.
          break;
        case "smudge": {
          // Translate the vertex by the per-step drag, falloff-weighted.
          // The drag vector is recomputed each step (not per stroke);
          // matches Blender's "grab+brush" feel.
          nx = dragVec.x * w;
          ny = dragVec.y * w;
          nz = dragVec.z * w;
          ok = (nx !== 0 || ny !== 0 || nz !== 0);
          break;
        }
        case "twist": {
          // Rodrigues rotation around camera-forward axis through the
          // brush centre by `twistTheta * w`. theta=0 -> no movement.
          if (twistTheta === 0) break;
          const localTheta = twistTheta * w;
          // Re-derive cos/sin per vertex so falloff applies correctly
          // (a single global cos/sin would rotate the brush's edge
          // verts as much as the centre — wrong).
          const ct = Math.cos(localTheta);
          const st = Math.sin(localTheta);
          // Rotate (px-cx, py-cy, pz-cz) around (bdx, bdy, bdz).
          const vx = -ddx, vy = -ddy, vz = -ddz; // pos - centre
          // k cross v
          const cxv = bdy * vz - bdz * vy;
          const cyv = bdz * vx - bdx * vz;
          const czv = bdx * vy - bdy * vx;
          // k dot v
          const kdv = bdx * vx + bdy * vy + bdz * vz;
          const rx = vx * ct + cxv * st + bdx * kdv * (1 - ct);
          const ry = vy * ct + cyv * st + bdy * kdv * (1 - ct);
          const rz = vz * ct + czv * st + bdz * kdv * (1 - ct);
          // Final pos = centre + rotated; delta = final - current.
          nx = (localPoint.x + rx) - px;
          ny = (localPoint.y + ry) - py;
          nz = (localPoint.z + rz) - pz;
          ok = true;
          break;
        }
        case "layer": {
          // Add a fixed layer of `strength * falloff * radius * 0.5`
          // along the per-vertex normal. Matches Blender's layer
          // behaviour; compounds on re-application.
          if (!nrmArr) break;
          let vx = nrmArr[sx], vy = nrmArr[sx + 1], vz = nrmArr[sx + 2];
          const vlen = Math.sqrt(vx * vx + vy * vy + vz * vz);
          if (vlen < 1e-6) break;
          vx /= vlen; vy /= vlen; vz /= vlen;
          // Same magnitude convention as inflate but halved so layer
          // builds geometry slowly enough for stroke-feel.
          nx = vx * w * r0 * 0.5;
          ny = vy * w * r0 * 0.5;
          nz = vz * w * r0 * 0.5;
          ok = true;
          break;
        }
        case "retopo": {
          // Region Laplacian relax. Same code path as smooth but
          // gated on `state.brush === "retopo"` so the UX label is
          // distinct ("retopologise" vs "smooth"). Future v6 will
          // expand this to a true Catmull-Clark sub-region pass.
          if (!rec.neighbours) {
            const idxAttr = mesh.geometry.getIndex();
            rec.neighbours = buildNeighbours(idxAttr, posAttr.count);
          }
          const ngb = rec.neighbours[vi];
          if (!ngb || ngb.length === 0) break;
          let mx = 0, my = 0, mz = 0;
          for (const nj of ngb) {
            mx += posArr[nj * 3];
            my += posArr[nj * 3 + 1];
            mz += posArr[nj * 3 + 2];
          }
          mx /= ngb.length; my /= ngb.length; mz /= ngb.length;
          // Stronger pull-to-mean than the smooth brush so a single
          // pass produces a visibly cleaner triangulation.
          const wRetopo = Math.min(1.0, w * 1.5);
          nx = (mx - px) * wRetopo;
          ny = (my - py) * wRetopo;
          nz = (mz - pz) * wRetopo;
          ok = true;
          break;
        }
      }

      if (!ok) continue;

      posArr[sx]     += nx;
      posArr[sx + 1] += ny;
      posArr[sx + 2] += nz;
      // Update accumDisp = current - original.
      rec.accumDisp[sx]     = posArr[sx]     - rec.originalPos[sx];
      rec.accumDisp[sx + 1] = posArr[sx + 1] - rec.originalPos[sx + 1];
      rec.accumDisp[sx + 2] = posArr[sx + 2] - rec.originalPos[sx + 2];
      rec.modifiedSet.add(vi);
      movedThisStep++;
    }

    if (movedThisStep > 0) {
      posAttr.needsUpdate = true;
      // Mirror across an arbitrary axis (X / Y / Z) — replaces the
      // v1 X-only path. The mirror finds verts within `radius` of the
      // reflected brush centre and applies the same brush logic with
      // the brush direction reflected on the matching axis. A mirrored
      // smudge gets a mirrored drag vector; a mirrored twist flips
      // sign on the matching component. Imperfect mirror (no
      // precomputed mirror map), but fine for symmetric authoring.
      const mAxis = state.mirrorAxis;
      if (mAxis && mAxis !== "off") {
        const mirrorCentre = {
          x: mAxis === "x" ? -localPoint.x : localPoint.x,
          y: mAxis === "y" ? -localPoint.y : localPoint.y,
          z: mAxis === "z" ? -localPoint.z : localPoint.z,
        };
        const mbdx = mAxis === "x" ? -bdx : bdx;
        const mbdy = mAxis === "y" ? -bdy : bdy;
        const mbdz = mAxis === "z" ? -bdz : bdz;
        // Reflect the drag vector for smudge.
        const mDragX = mAxis === "x" ? -dragVec.x : dragVec.x;
        const mDragY = mAxis === "y" ? -dragVec.y : dragVec.y;
        const mDragZ = mAxis === "z" ? -dragVec.z : dragVec.z;
        const mAffected = rec.gridIdx.queryRadius(
          mirrorCentre.x, mirrorCentre.y, mirrorCentre.z, r0,
        );
        for (const vi of mAffected) {
          const sx = vi * 3;
          const px = posArr[sx], py = posArr[sx + 1], pz = posArr[sx + 2];
          const ddx = px - mirrorCentre.x;
          const ddy = py - mirrorCentre.y;
          const ddz = pz - mirrorCentre.z;
          const dist = Math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
          if (dist >= r0) continue;
          const w = falloff(dist / r0, state.falloff) * state.strength;
          if (w <= 0) continue;
          if (!perSub.before.has(vi)) {
            perSub.before.set(vi, [px, py, pz]);
          }
          perSub.indices.add(vi);
          // Mirrored brush deltas — match the primary brush switch
          // but with reflected direction / drag.
          let dnx = 0, dny = 0, dnz = 0;
          let mok = false;
          if (state.brush === "push" || state.brush === "pull") {
            const sg = state.brush === "push" ? 1 : -1;
            dnx = mbdx * sg * w * r0;
            dny = mbdy * sg * w * r0;
            dnz = mbdz * sg * w * r0;
            mok = true;
          } else if (state.brush === "inflate" || state.brush === "layer") {
            if (!nrmArr) continue;
            let vx = nrmArr[sx], vy = nrmArr[sx + 1], vz = nrmArr[sx + 2];
            const vlen = Math.sqrt(vx * vx + vy * vy + vz * vz);
            if (vlen < 1e-6) continue;
            vx /= vlen; vy /= vlen; vz /= vlen;
            const scale = state.brush === "layer" ? 0.5 : 1.0;
            dnx = vx * w * r0 * scale;
            dny = vy * w * r0 * scale;
            dnz = vz * w * r0 * scale;
            mok = true;
          } else if (state.brush === "pinch") {
            if (dist < 1e-6) continue;
            dnx = -ddx * (w * 0.5);
            dny = -ddy * (w * 0.5);
            dnz = -ddz * (w * 0.5);
            mok = true;
          } else if (state.brush === "smudge") {
            dnx = mDragX * w;
            dny = mDragY * w;
            dnz = mDragZ * w;
            mok = (dnx !== 0 || dny !== 0 || dnz !== 0);
          } else if (state.brush === "twist") {
            // Mirror flips the rotation sign (right-hand-rule on the
            // reflected axis). Re-run Rodrigues with the reflected
            // axis + opposite-sign theta.
            if (twistTheta === 0) continue;
            const localTheta = -twistTheta * w;
            const ct = Math.cos(localTheta);
            const st = Math.sin(localTheta);
            const vx = -ddx, vy = -ddy, vz = -ddz;
            const cxv = mbdy * vz - mbdz * vy;
            const cyv = mbdz * vx - mbdx * vz;
            const czv = mbdx * vy - mbdy * vx;
            const kdv = mbdx * vx + mbdy * vy + mbdz * vz;
            const rx = vx * ct + cxv * st + mbdx * kdv * (1 - ct);
            const ry = vy * ct + cyv * st + mbdy * kdv * (1 - ct);
            const rz = vz * ct + czv * st + mbdz * kdv * (1 - ct);
            dnx = (mirrorCentre.x + rx) - px;
            dny = (mirrorCentre.y + ry) - py;
            dnz = (mirrorCentre.z + rz) - pz;
            mok = true;
          } else if (state.brush === "smooth" || state.brush === "retopo") {
            const ngb = rec.neighbours && rec.neighbours[vi];
            if (!ngb || ngb.length === 0) continue;
            let mx2 = 0, my2 = 0, mz2 = 0;
            for (const nj of ngb) {
              mx2 += posArr[nj * 3];
              my2 += posArr[nj * 3 + 1];
              mz2 += posArr[nj * 3 + 2];
            }
            mx2 /= ngb.length; my2 /= ngb.length; mz2 /= ngb.length;
            const wEff = state.brush === "retopo" ? Math.min(1.0, w * 1.5) : w;
            dnx = (mx2 - px) * wEff;
            dny = (my2 - py) * wEff;
            dnz = (mz2 - pz) * wEff;
            mok = true;
          } else continue;
          if (!mok) continue;
          posArr[sx]     += dnx;
          posArr[sx + 1] += dny;
          posArr[sx + 2] += dnz;
          rec.accumDisp[sx]     = posArr[sx]     - rec.originalPos[sx];
          rec.accumDisp[sx + 1] = posArr[sx + 1] - rec.originalPos[sx + 1];
          rec.accumDisp[sx + 2] = posArr[sx + 2] - rec.originalPos[sx + 2];
          rec.modifiedSet.add(vi);
        }
      }
      if (state.realTimeNormals) recomputeNormals(mesh);
      state.stroke.pointsApplied += movedThisStep;
    }
  }

  // -------------------------------------------------------------------
  // Raycaster — mirrors the debug-mode rayhit logic in model_viewer.js
  // -------------------------------------------------------------------
  let _raycaster = null;
  function raycastFromScreen(clientX, clientY) {
    const meta = (typeof window.psoGetSculptMeshGroup === "function")
      ? window.psoGetSculptMeshGroup() : null;
    if (!meta) return null;
    const { camera, group, debugMeshes, THREE } = meta;
    if (!group || !camera || !debugMeshes || debugMeshes.length === 0) return null;
    const cv = document.getElementById("modelCanvas");
    if (!cv) return null;
    const rect = cv.getBoundingClientRect();
    const mx = ((clientX - rect.left) / rect.width) * 2 - 1;
    const my = -(((clientY - rect.top) / rect.height) * 2 - 1);
    if (!_raycaster) _raycaster = new THREE.Raycaster();
    const ndc = new THREE.Vector2(mx, my);
    _raycaster.setFromCamera(ndc, camera);
    const meshes = debugMeshes.map((e) => e.mesh).filter(Boolean);
    const hits = _raycaster.intersectObjects(meshes, false);
    if (!hits.length) return null;
    const hitMesh = hits[0].object;
    const submeshIdx = debugMeshes.findIndex((e) => e.mesh === hitMesh);
    if (submeshIdx < 0) return null;
    // Convert hit point world->local using the mesh's matrixWorld inverse,
    // then chain up through the group's local matrix as well.
    const mInv = new THREE.Matrix4().copy(hitMesh.matrixWorld).invert();
    const hitWorld = hits[0].point.clone();
    const localPoint = hitWorld.clone().applyMatrix4(mInv);
    // Camera forward in submesh-local: take camera-forward in world,
    // strip translation, then transform via mInv.
    const camForwardWorld = new THREE.Vector3();
    camera.getWorldDirection(camForwardWorld);
    const camForwardLocal = camForwardWorld.clone();
    // Strip translation: rotate-only the inverse matrix.
    const r3 = new THREE.Matrix3().setFromMatrix4(mInv);
    camForwardLocal.applyMatrix3(r3);
    camForwardLocal.normalize();
    return {
      meshGroup: group,
      mesh: hitMesh,
      submeshIdx,
      localPoint,
      cameraForwardLocal: camForwardLocal,
    };
  }

  // -------------------------------------------------------------------
  // Normals recompute
  // -------------------------------------------------------------------
  function recomputeNormals(mesh) {
    if (!mesh || !mesh.geometry) return;
    mesh.geometry.computeVertexNormals();
    const n = mesh.geometry.getAttribute("normal");
    if (n) n.needsUpdate = true;
  }

  function recomputeAllNormals() {
    for (const [, rec] of sculptRecords) {
      recomputeNormals(rec.meshRef);
    }
  }

  // -------------------------------------------------------------------
  // Reset to source — drop all accumulated edits + re-snapshot the
  // ORIGINAL positions in case the model was re-loaded since.
  // -------------------------------------------------------------------
  function resetAll() {
    for (const [, rec] of sculptRecords) {
      const posAttr = rec.meshRef.geometry.getAttribute("position");
      posAttr.array.set(rec.originalPos);
      posAttr.needsUpdate = true;
      rec.accumDisp.fill(0);
      rec.modifiedSet.clear();
      rec.gridDirty = true;
    }
    recomputeAllNormals();
    undoStack.length = 0;
    redoStack.length = 0;
    if (typeof updateUndoButtons === "function") updateUndoButtons();
    if (typeof updateStats === "function") updateStats();
  }

  // -------------------------------------------------------------------
  // Save / Load
  // -------------------------------------------------------------------
  async function saveSculpt() {
    const meta = (typeof window.psoGetSculptMeshGroup === "function")
      ? window.psoGetSculptMeshGroup() : null;
    if (!meta) {
      _setStatus("err", "no model loaded");
      return null;
    }
    const modelPath = meta.modelPath;
    if (!modelPath) {
      _setStatus("err", "no model path");
      return null;
    }
    if (sculptRecords.size === 0) {
      _setStatus("err", "nothing sculpted yet");
      return null;
    }

    // Build the wire payload mirroring formats.sculpt.encode_sculpt_payload.
    const subs = [];
    for (const [submeshIdx, rec] of sculptRecords) {
      const vc = rec.originalPos.length / 3;
      const modified = [...rec.modifiedSet].sort((a, b) => a - b);
      const ratio = modified.length / Math.max(1, vc);
      let mode, dispBytes;
      if (modified.length > 0 && ratio <= 0.33) {
        // sparse
        const rowDisp = new Float32Array(modified.length * 3);
        for (let k = 0; k < modified.length; k++) {
          const vi = modified[k];
          rowDisp[k * 3 + 0] = rec.accumDisp[vi * 3 + 0];
          rowDisp[k * 3 + 1] = rec.accumDisp[vi * 3 + 1];
          rowDisp[k * 3 + 2] = rec.accumDisp[vi * 3 + 2];
        }
        dispBytes = new Uint8Array(rowDisp.buffer);
        mode = "sparse";
      } else {
        dispBytes = new Uint8Array(rec.accumDisp.buffer);
        mode = "dense";
      }
      const idxArr = new Uint32Array(modified);
      subs.push({
        submesh_idx: submeshIdx,
        material_id: (rec.meshRef.userData && rec.meshRef.userData.materialId) | 0,
        vertex_count: vc,
        displacement_b64: _bytesToB64(dispBytes),
        modified_indices_b64: _bytesToB64(new Uint8Array(idxArr.buffer)),
        mode,
      });
    }

    const payload = {
      format_version: 1,
      source_path: modelPath,
      source_sha: computeSourceShaSync(),
      subdivide_level: state.subdivideLevel | 0,
      smooth_normals: true,
      submeshes: subs,
      // Server recomputes a canonical sha; this seed value is ignored
      // but keeps the wire payload self-consistent.
      sha: computeSourceShaSync(),
      saved_at_ms: Date.now(),
    };

    try {
      const r = await fetch("/api/sculpt/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_path: modelPath,
          mesh_payload: payload,
          subdivide_level: state.subdivideLevel | 0,
          smooth_normals: true,
        }),
      });
      if (!r.ok) {
        let detail = `HTTP ${r.status}`;
        try { detail = (await r.json()).detail || detail; } catch {}
        throw new Error(detail);
      }
      const j = await r.json();
      _setStatus("done", `saved · sha ${j.sha.slice(0, 8)}…`);
      return j;
    } catch (e) {
      _setStatus("err", `save failed: ${e.message || e}`);
      return null;
    }
  }

  function _bytesToB64(u8) {
    let s = "";
    const chunk = 0x8000;
    for (let i = 0; i < u8.length; i += chunk) {
      s += String.fromCharCode.apply(null, u8.subarray(i, i + chunk));
    }
    return btoa(s);
  }

  // -------------------------------------------------------------------
  // UI rendering — embedded into the existing texture-panel via a tab
  // hook. We don't own the panel chrome; we wait for the "Sculpt" tab
  // to be clicked, then we own the body region.
  // -------------------------------------------------------------------
  function ensureStyleInjected() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .pso-sculpt-block {
        padding: 8px;
        display: flex;
        flex-direction: column;
        gap: 8px;
        font-size: 11px;
        color: #c7d8ec;
      }
      .pso-sculpt-block label { color: #99a4b3; display: flex; align-items: center; gap: 6px; }
      .pso-sculpt-toggle {
        display: flex;
        gap: 4px;
        align-items: center;
        padding: 4px 6px;
        border: 1px solid #2a313a;
        border-radius: 3px;
        background: rgba(0, 0, 0, 0.25);
      }
      .pso-sculpt-toggle button {
        background: transparent;
        border: 1px solid #2a313a;
        color: #99a4b3;
        cursor: pointer;
        padding: 2px 8px;
        font: inherit;
        border-radius: 2px;
        flex: 1;
      }
      .pso-sculpt-toggle button.on {
        background: rgba(255, 144, 0, 0.18);
        border-color: #ffaa00;
        color: #ffaa00;
      }
      .pso-sculpt-toggle button:hover {
        border-color: #00ffff;
      }
      .pso-sculpt-brushes {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 3px;
      }
      .pso-sculpt-brush {
        background: transparent;
        border: 1px solid #2a313a;
        color: #99a4b3;
        cursor: pointer;
        padding: 4px 6px;
        font: inherit;
        font-size: 10px;
        border-radius: 2px;
      }
      .pso-sculpt-brush.active {
        background: rgba(0, 255, 255, 0.12);
        border-color: #00ffff;
        color: #00ffff;
      }
      .pso-sculpt-brush:hover { border-color: #4a90e2; color: #c7d8ec; }
      .pso-sculpt-row { display: flex; gap: 6px; align-items: center; }
      .pso-sculpt-row .grow { flex: 1; }
      .pso-sculpt-row input[type=range] { flex: 1; }
      .pso-sculpt-row .num {
        min-width: 40px;
        font-variant-numeric: tabular-nums;
        text-align: right;
        color: #c7d8ec;
      }
      .pso-sculpt-actions {
        display: flex;
        gap: 4px;
        flex-wrap: wrap;
      }
      .pso-sculpt-actions button {
        background: transparent;
        border: 1px solid #2a313a;
        color: #99a4b3;
        cursor: pointer;
        padding: 3px 8px;
        font: inherit;
        border-radius: 2px;
      }
      .pso-sculpt-actions button:hover { border-color: #00ffff; color: #00ffff; }
      .pso-sculpt-actions button:disabled { opacity: 0.4; cursor: not-allowed; }
      .pso-sculpt-actions button.primary {
        border-color: #4a90e2;
        color: #c7d8ec;
      }
      .pso-sculpt-actions button.primary:hover {
        background: rgba(74, 144, 226, 0.18);
        border-color: #00ffff;
        color: #00ffff;
      }
      .pso-sculpt-actions button.danger {
        border-color: #4d2323;
        color: #d89090;
      }
      .pso-sculpt-actions button.danger:hover { border-color: #ff6680; color: #ff6680; }
      .pso-sculpt-stats {
        background: rgba(0, 0, 0, 0.25);
        border: 1px solid #2a313a;
        border-radius: 2px;
        padding: 4px 6px;
        font-size: 10px;
        color: #c7d8ec;
        white-space: pre-wrap;
        font-variant-numeric: tabular-nums;
      }
      .pso-sculpt-status {
        font-size: 10px;
        min-height: 12px;
      }
      .pso-sculpt-status.idle { color: #6c7785; }
      .pso-sculpt-status.running { color: #4a90e2; }
      .pso-sculpt-status.done { color: #56b67a; }
      .pso-sculpt-status.err  { color: #ff6680; }
    `;
    document.head.appendChild(style);
  }

  // The active body element where we render the Sculpt UI. Updated each
  // time the Sculpt tab is shown.
  let bodyEl = null;
  let statusEl = null;
  let undoBtn = null;
  let redoBtn = null;
  let statsEl = null;

  function _setStatus(stateStr, msg) {
    if (!statusEl) return;
    statusEl.textContent = msg;
    statusEl.className = "pso-sculpt-status " + stateStr;
  }

  function updateUndoButtons() {
    if (undoBtn) undoBtn.disabled = undoStack.length === 0;
    if (redoBtn) redoBtn.disabled = redoStack.length === 0;
  }

  function updateStats() {
    if (!statsEl) return;
    let nMod = 0, nVerts = 0, subs = 0;
    for (const [, rec] of sculptRecords) {
      subs++;
      nMod += rec.modifiedSet.size;
      nVerts += rec.originalPos.length / 3;
    }
    const undoSize = undoStack.length;
    statsEl.textContent =
      `Sculpt active: ${state.enabled ? "yes" : "no"}\n` +
      `Submeshes touched: ${subs}\n` +
      `Modified verts:   ${nMod.toLocaleString()}\n` +
      `Total verts:      ${nVerts.toLocaleString()}\n` +
      `Undo depth: ${undoSize} / ${UNDO_LIMIT}`;
  }

  function renderSculptBlock(body) {
    ensureStyleInjected();
    bodyEl = body;
    body.innerHTML = `
      <div class="pso-sculpt-block">
        <div class="pso-sculpt-toggle">
          <span class="grow" style="color:#99a4b3">Sculpt mode:</span>
          <button data-act="toggle" class="${state.enabled ? "on" : ""}"
                  title="enable click-to-sculpt (LMB sculpt, RMB orbit, wheel = radius, shift+wheel = strength)">
            ${state.enabled ? "ON (click + drag)" : "OFF"}
          </button>
        </div>
        <div>
          <div style="margin-bottom:3px;color:#99a4b3">Brush:</div>
          <div class="pso-sculpt-brushes">
            ${BRUSHES.map((b) => `
              <button class="pso-sculpt-brush ${state.brush === b ? "active" : ""}"
                      data-brush="${b}" title="${_brushTip(b)}">${_brushLabel(b)}</button>
            `).join("")}
          </div>
        </div>
        <label>Radius:
          <input type="range" min="0.05" max="3.0" step="0.01" value="${state.radius}" data-knob="radius">
          <span class="num" data-readout="radius">${state.radius.toFixed(2)}</span>
        </label>
        <label>Strength:
          <input type="range" min="0" max="1.0" step="0.01" value="${state.strength}" data-knob="strength">
          <span class="num" data-readout="strength">${state.strength.toFixed(2)}</span>
        </label>
        <label>Falloff:
          <select data-knob="falloff">
            ${FALLOFFS.map((f) => `<option value="${f}" ${state.falloff === f ? "selected" : ""}>${f}</option>`).join("")}
          </select>
        </label>
        <label title="Mirror brushed edits across the chosen axis (Off / X / Y / Z)">
          Mirror axis:
          <select data-knob="mirrorAxis">
            ${MIRROR_AXES.map((a) => `<option value="${a}" ${state.mirrorAxis === a ? "selected" : ""}>${a === "off" ? "Off" : a.toUpperCase()}</option>`).join("")}
          </select>
        </label>
        <label title="Twist rate — radians of rotation per unit drag distance. Negative = counter-clockwise.">
          Twist rate:
          <input type="range" min="-3" max="3" step="0.1" value="${state.twistRate}" data-knob="twistRate">
          <span class="num" data-readout="twistRate">${state.twistRate.toFixed(1)}</span>
        </label>
        <label>
          <input type="checkbox" data-knob="realTimeNormals" ${state.realTimeNormals ? "checked" : ""}>
          Real-time normals (slow on high-poly)
        </label>
        <div class="pso-sculpt-stats" data-region="stats"></div>
        <div class="pso-sculpt-actions">
          <button data-act="undo">Undo (Ctrl+Z)</button>
          <button data-act="redo">Redo (Ctrl+Y)</button>
          <button class="primary" data-act="save">Save sculpt</button>
          <button class="primary" data-act="encode-nj"
                  title="Save the current sculpt and bake to a deployable .nj file">
            Encode as NJ
          </button>
          <button class="danger" data-act="reset">Reset to source</button>
        </div>
        <div class="pso-sculpt-status idle" data-region="status">ready</div>
      </div>
    `;
    statusEl = body.querySelector('[data-region="status"]');
    statsEl = body.querySelector('[data-region="stats"]');
    undoBtn = body.querySelector('[data-act="undo"]');
    redoBtn = body.querySelector('[data-act="redo"]');
    body.addEventListener("click", onSculptPanelClick);
    body.querySelectorAll("[data-knob]").forEach((el) => {
      el.addEventListener("input", onKnobChange);
      el.addEventListener("change", onKnobChange);
    });
    updateUndoButtons();
    updateStats();
  }

  function _brushLabel(b) {
    return ({
      push: "push", pull: "pull", inflate: "infl",
      smooth: "smth", pinch: "pnch", flatten: "flat",
      decimate_region: "deci",
      smudge: "smdg", twist: "twst", layer: "layr", retopo: "rtpo",
    })[b] || b;
  }
  function _brushTip(b) {
    return ({
      push: "Move verts away from camera",
      pull: "Move verts toward camera",
      inflate: "Move verts along their normal (positive=outward)",
      smooth: "Laplacian — blend each vertex toward its 1-ring centroid",
      pinch: "Move verts toward brush centre",
      flatten: "Project verts onto brush region's average plane",
      decimate_region: "Quadric-edge-collapse on the brushed face set (v1: stub)",
      smudge: "Drag verts along the cursor — Blender's grab+brush",
      twist: "Rotate verts around the brush axis (drag = rotation angle)",
      layer: "Add a layer of geometry along per-vertex normals (compounds)",
      retopo: "Region retopology — relax brushed verts toward uniform spacing",
    })[b] || b;
  }

  function onSculptPanelClick(ev) {
    const t = ev.target.closest("[data-act],[data-brush]");
    if (!t) return;
    const brush = t.getAttribute("data-brush");
    if (brush) {
      state.brush = brush;
      if (brush === "decimate_region") {
        _setStatus("running", "decimate is a v1 stub — no-op for now");
      } else {
        _setStatus("idle", `brush: ${brush}`);
      }
      bodyEl.querySelectorAll(".pso-sculpt-brush").forEach((b) => {
        b.classList.toggle("active", b.dataset.brush === brush);
      });
      return;
    }
    const act = t.getAttribute("data-act");
    if (act === "toggle") {
      state.enabled = !state.enabled;
      t.classList.toggle("on", state.enabled);
      t.textContent = state.enabled ? "ON (click + drag)" : "OFF";
      _setStatus("idle", state.enabled ? "click + drag the model to sculpt" : "sculpt disabled");
      // Surface to the camera so it knows to ignore left-button orbit.
      if (typeof window.psoSetSculptModeActive === "function") {
        window.psoSetSculptModeActive(state.enabled);
      }
      updateStats();
      return;
    }
    if (act === "undo") {
      const ok = undoOne();
      if (!ok) _setStatus("idle", "nothing to undo");
      else _setStatus("done", `undid (${undoStack.length} left)`);
      updateStats();
      return;
    }
    if (act === "redo") {
      const ok = redoOne();
      if (!ok) _setStatus("idle", "nothing to redo");
      else _setStatus("done", `redid (${redoStack.length} left)`);
      updateStats();
      return;
    }
    if (act === "save") {
      _setStatus("running", "saving…");
      saveSculpt().then((j) => updateStats());
      return;
    }
    if (act === "encode-nj") {
      _setStatus("running", "saving + encoding NJ…");
      (async () => {
        // Save first so the server has a sidecar to encode from.
        const savedJson = await saveSculpt();
        if (!savedJson || !savedJson.sha) {
          _setStatus("err", "save failed; can't encode");
          return;
        }
        const meta = (typeof window.psoGetSculptMeshGroup === "function")
          ? window.psoGetSculptMeshGroup() : null;
        if (!meta || !meta.modelPath) {
          _setStatus("err", "no model loaded");
          return;
        }
        try {
          // Derive the output filename from the inner name.
          const innerName = meta.modelPath.split("#").pop() || "sculpt.nj";
          const outName = innerName.replace(/\.nj$/, "_sculpt.nj");
          const r = await fetch("/api/sculpt/build_nj", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              model_path: meta.modelPath,
              inner_idx: 0,
              sculpt_sha: savedJson.sha,
              output_name: outName,
            }),
          });
          if (!r.ok) {
            let detail = `HTTP ${r.status}`;
            try { detail = (await r.json()).detail || detail; } catch {}
            throw new Error(detail);
          }
          const j = await r.json();
          _setStatus("done",
            `encoded · ${(j.size / 1024).toFixed(1)} KB · md5 ${j.md5.slice(0, 8)}…`);
        } catch (e) {
          _setStatus("err", `encode failed: ${e.message || e}`);
        }
      })();
      return;
    }
    if (act === "reset") {
      resetAll();
      _setStatus("done", "reverted to original");
      return;
    }
  }

  function onKnobChange(ev) {
    const t = ev.target;
    const knob = t.getAttribute("data-knob");
    if (!knob) return;
    if (knob === "radius") {
      state.radius = parseFloat(t.value);
      const ro = bodyEl.querySelector('[data-readout="radius"]');
      if (ro) ro.textContent = state.radius.toFixed(2);
      // Force grid rebuild on next stroke since cell-size depends on radius.
      for (const [, rec] of sculptRecords) rec.gridDirty = true;
    } else if (knob === "strength") {
      state.strength = parseFloat(t.value);
      const ro = bodyEl.querySelector('[data-readout="strength"]');
      if (ro) ro.textContent = state.strength.toFixed(2);
    } else if (knob === "falloff") {
      state.falloff = t.value;
    } else if (knob === "mirrorAxis") {
      // Validate against the known axes; ignore anything else so a
      // stale option-value can't crash the brush loop.
      if (MIRROR_AXES.indexOf(t.value) >= 0) state.mirrorAxis = t.value;
      // Toggle the label's "mirror-active" class so the CSS rule
      // for highlighting the dropdown can see the selection.
      const lbl = t.closest("label");
      if (lbl) lbl.classList.toggle("mirror-active", state.mirrorAxis !== "off");
    } else if (knob === "mirrorX") {
      // Legacy hook: kept for any older test harness that still posts
      // a mirrorX checkbox. The setter on `state.mirrorX` re-routes
      // to `state.mirrorAxis`.
      state.mirrorX = !!t.checked;
    } else if (knob === "twistRate") {
      state.twistRate = parseFloat(t.value);
      const ro = bodyEl.querySelector('[data-readout="twistRate"]');
      if (ro) ro.textContent = state.twistRate.toFixed(1);
    } else if (knob === "realTimeNormals") {
      state.realTimeNormals = !!t.checked;
    }
  }

  // -------------------------------------------------------------------
  // Wheel = brush radius adjust; Shift+wheel = strength
  // -------------------------------------------------------------------
  function onWheel(ev) {
    if (!state.enabled) return;
    if (ev.target && ev.target.id !== "modelCanvas") return;
    ev.preventDefault();
    const dir = ev.deltaY > 0 ? -1 : 1;
    if (ev.shiftKey) {
      state.strength = Math.max(0, Math.min(1.0, state.strength + dir * 0.05));
    } else {
      state.radius = Math.max(0.05, Math.min(3.0, state.radius + dir * 0.05));
    }
    refreshKnobUi();
  }

  function refreshKnobUi() {
    if (!bodyEl) return;
    const r = bodyEl.querySelector('[data-knob="radius"]');
    const ro = bodyEl.querySelector('[data-readout="radius"]');
    const s = bodyEl.querySelector('[data-knob="strength"]');
    const so = bodyEl.querySelector('[data-readout="strength"]');
    if (r) r.value = String(state.radius);
    if (ro) ro.textContent = state.radius.toFixed(2);
    if (s) s.value = String(state.strength);
    if (so) so.textContent = state.strength.toFixed(2);
  }

  // -------------------------------------------------------------------
  // Keyboard: Ctrl+Z / Ctrl+Y for undo/redo while sculpt is active
  // -------------------------------------------------------------------
  function onKeyDown(ev) {
    if (!state.enabled) return;
    if (!ev.ctrlKey && !ev.metaKey) return;
    if (ev.key === "z" || ev.key === "Z") {
      ev.preventDefault();
      if (ev.shiftKey) { redoOne(); _setStatus("done", "redid"); }
      else { undoOne(); _setStatus("done", "undid"); }
      updateStats();
    } else if (ev.key === "y" || ev.key === "Y") {
      ev.preventDefault();
      redoOne();
      _setStatus("done", "redid");
      updateStats();
    }
  }

  // -------------------------------------------------------------------
  // Tab integration — uses the public hook surfaced by texture_panel.js.
  // psoTexturePanelAddTabButton appends a new button to the tab strip;
  // psoTexturePanelRegisterTab registers our body-renderer callback so
  // the existing dispatcher routes clicks to us without us patching its
  // internals.
  // -------------------------------------------------------------------
  function injectSculptTab() {
    if (typeof window.psoTexturePanelAddTabButton !== "function") return false;
    if (typeof window.psoTexturePanelRegisterTab !== "function") return false;
    const ok = window.psoTexturePanelAddTabButton(
      "sculpt", "Sculpt",
      "interactive vertex-displacement brushes (push, pull, inflate, smooth, pinch, flatten)",
    );
    window.psoTexturePanelRegisterTab("sculpt", (body) => renderSculptBlock(body));
    return ok;
  }

  // Wait for psoTexturePanel to mount AND for the public hook to be
  // registered. We poll up to 30 s; texture_panel registers the hook
  // on load.
  function waitForPanel(deadline) {
    if (injectSculptTab()) return;
    if (Date.now() > deadline) {
      console.warn("[sculpt_panel] texture panel never appeared; sculpt disabled");
      return;
    }
    setTimeout(() => waitForPanel(deadline), 250);
  }

  function init() {
    waitForPanel(Date.now() + 30_000);
    // Pointer events on the canvas live independently of which tab is
    // showing — sculpt-mode toggle gates them.
    const cv = document.getElementById("modelCanvas");
    if (cv) {
      cv.addEventListener("pointerdown", onPointerDown);
      cv.addEventListener("pointermove", onPointerMove);
      cv.addEventListener("pointerup", onPointerUp);
      cv.addEventListener("pointercancel", onPointerUp);
      cv.addEventListener("wheel", onWheel, { passive: false });
    } else {
      // Late attach: poll for the canvas.
      const t = setInterval(() => {
        const c = document.getElementById("modelCanvas");
        if (!c) return;
        clearInterval(t);
        c.addEventListener("pointerdown", onPointerDown);
        c.addEventListener("pointermove", onPointerMove);
        c.addEventListener("pointerup", onPointerUp);
        c.addEventListener("pointercancel", onPointerUp);
        c.addEventListener("wheel", onWheel, { passive: false });
      }, 250);
    }
    document.addEventListener("keydown", onKeyDown);
    // Reset session state every time a new model loads.
    if (window.bus && typeof window.bus.on === "function") {
      window.bus.on("model.loaded", resetSession);
    }
    // Wrap psoApplyMeshPayload so that subdivide / variant-swap /
    // any-other-mesh-rebuild also drops stale sculpt records. The
    // previous records' THREE.Mesh refs would otherwise dangle —
    // pointing at disposed geometry that's been replaced by the new
    // payload.
    //
    // Compose-with-Subdivide order: SUBDIVIDE FIRST, then SCULPT on
    // top. After subdivide the live mesh has new vertex IDs + new
    // vertex count; sculpt records (keyed by submesh + vertex index)
    // can't survive the topology change. The user's workflow is to
    // pick subdivide level first (×1/×2/×3 in the Subdivide tab) and
    // THEN sculpt the resulting denser mesh.
    //
    // Reverse order (sculpt first, then subdivide) is NOT supported in
    // v1 — subdivide reads from /api/model/subdivide which fetches the
    // ORIGINAL mesh from disk, not the live (sculpted) one. A future
    // pass can ship the sculpted mesh to the subdivide endpoint as the
    // input; deferring that to v2.
    waitForApplyMeshWrap(Date.now() + 30_000);
  }

  function resetSession() {
    disposeRecords();
    undoStack.length = 0;
    redoStack.length = 0;
    if (typeof updateUndoButtons === "function") updateUndoButtons();
    if (typeof updateStats === "function") updateStats();
  }

  function waitForApplyMeshWrap(deadline) {
    let allWired = true;
    if (typeof window.psoApplyMeshPayload === "function") {
      const orig = window.psoApplyMeshPayload;
      if (!orig.__psoSculptWrapped) {
        const wrapped = function (payload, opts) {
          // Reset BEFORE the swap so the old records' meshRefs aren't
          // touched by the post-swap cleanup paths.
          resetSession();
          return orig.apply(this, arguments);
        };
        wrapped.__psoSculptWrapped = true;
        window.psoApplyMeshPayload = wrapped;
      }
    } else allWired = false;
    if (typeof window.psoOpenModelByPath === "function") {
      const orig = window.psoOpenModelByPath;
      if (!orig.__psoSculptWrapped) {
        const wrapped = async function (modelPath, entry, matched) {
          resetSession();
          return await orig.apply(this, arguments);
        };
        wrapped.__psoSculptWrapped = true;
        window.psoOpenModelByPath = wrapped;
      }
    } else allWired = false;
    if (allWired) return;
    if (Date.now() > deadline) return;
    setTimeout(() => waitForApplyMeshWrap(deadline), 250);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // Surface a manual-test handle for devtools.
  window.psoSculptState = state;
  window.psoSculptRecords = sculptRecords;
  window.psoSculptUndo = undoOne;
  window.psoSculptRedo = redoOne;
  window.psoSculptReset = resetAll;
  window.psoSculptSave = saveSculpt;
})();
