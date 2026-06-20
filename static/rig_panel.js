// =====================================================================
// PSOBB Rig Panel — skeleton edits, weight painting, IK targets.
// 2026-04-25
//
// Adds a "Rig" tab to the existing texture-panel tab strip. Three modes
// are supported:
//
//   Skeleton mode
//     - Renders every bone as a draggable widget in 3D.
//     - Click a bone to select; the bone's TRS is editable in a TRS
//       inspector to the right of the bone tree.
//     - Drag a bone (LMB) to translate; Shift+drag to rotate; Alt+drag
//       to scale.
//     - Reparent via drag-into-tree; rename via double-click.
//
//   Weight Paint mode
//     - Pick a bone in the tree.
//     - Click+drag on the mesh to paint vertex weights toward that bone.
//     - Heatmap visualises the active bone's weights (red = high,
//       blue = zero).
//     - Brush size + strength + falloff sliders; "Normalize selection"
//       button.
//
//   IK mode
//     - "Add IK target on selected bone" places a 3D marker at the
//       bone's current world position with chain length 2.
//     - Drag the marker — FABRIK runs on the chain in real time and
//       writes overrides via psoSetBonePoseOverride.
//
// Auto-skin
//     - Heat-equation-style smoothed inverse-distance weights for every
//       submesh in one click. Talks to /api/rig/auto_skin.
//
// Save / Reset / Apply
//     - Save: POST /api/rig/save with the current skeleton + weights +
//       IK targets; returns a SHA the editor uses to re-fetch.
//     - Reset: drops all rig overrides + reverts the in-viewport mesh
//       to its bind pose.
//     - Apply: re-runs the full re-bake pass (also done automatically
//       after every interactive edit).
//
// All inter-panel communication goes through the additive exports the
// rig agent surfaced on model_viewer.js:
//
//   window.psoGetRigContext()             handle to scene/cam/skeleton
//   window.psoGetSkeleton()                read-only bone list
//   window.psoSetBonePoseOverride()        push override TRS
//   window.psoClearBonePoseOverrides()     reset all
//   window.psoGetBoneWorldPositions()      live world pos per bone
//   window.psoBoneSpaceToWorld(pt)         meshGroup local -> world
//   window.psoSetVertexWeights()           override weights for submesh
//   window.psoApplyRigBake()               re-bake skinned mesh
//   window.psoSetRigModeActive()           gate orbit drag
//   window.psoGetSubmeshLocalPositions()   bone-local positions
//   window.psoGetSubmeshBoneIndices()      bone_idx per vertex
//
// Wire format (POST /api/rig/save):
//   { model_path, rig_payload, subdivide_level }
//
// rig_payload has the same envelope `formats.rigging.encode_rig_payload`
// emits — server validates by re-decoding once and re-encodes for SHA.
// =====================================================================

(function () {
  "use strict";

  if (window.__psoRigPanelLoaded) return;
  window.__psoRigPanelLoaded = true;

  const STYLE_ID = "psoRigPanelStyle";
  const FORMAT_VERSION = 1;
  const MAX_INFLUENCES = 4;
  const _BAMS_TO_RAD = (2.0 * Math.PI) / 65536.0;
  const RAD_TO_BAMS = 1.0 / _BAMS_TO_RAD;

  // Modes the toolbar mirrors.
  const MODE_SKELETON = "skeleton";
  const MODE_WEIGHT = "weight";
  const MODE_IK = "ik";
  const VALID_MODES = [MODE_SKELETON, MODE_WEIGHT, MODE_IK];

  const FALLOFFS = ["linear", "smooth", "sharp", "gaussian"];

  // -------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------
  const state = {
    enabled: false,                // rig mode toggle (mirrors sculpt)
    mode: MODE_SKELETON,
    activeBoneIdx: -1,
    selectedBones: new Set(),      // multi-select via Ctrl/Shift
    hiddenBones: new Set(),
    boneNames: new Map(),          // boneIdx -> string
    boneWidgets: null,             // THREE.Group of bone widgets
    boneWidgetMeta: [],            // per-bone {sphere, line?, idx}
    skeleton: null,                // snapshot from psoGetSkeleton
    submeshWeights: new Map(),     // submeshIdx -> { indices: Int32Array, weights: Float32Array }
    submeshOriginalBoneIdx: new Map(),  // submeshIdx -> Int32Array (pristine)
    ikTargets: [],                 // [{ boneIdx, chainLen, target: [x,y,z], iterations, name, marker?: THREE.Object3D }]
    activeIkIdx: -1,
    weightBrush: { radius: 0.4, strength: 0.5, falloff: "smooth" },
    showHeatmap: true,
    boneOverridesUndo: [],         // [{ boneIdx, before: pose|null, after: pose|null }]
    weightUndo: [],                // [{ submeshIdx, before: {indices, weights}, after: {indices, weights} }]
    bodyEl: null,
    statusEl: null,
    treeEl: null,
    legendEl: null,
    inspectorEl: null,
    subdivideLevel: 0,
    sourceSha: null,
  };

  // -------------------------------------------------------------------
  // Style injection — mirror sculpt_panel.js
  // -------------------------------------------------------------------
  function ensureStyleInjected() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .pso-rig-block {
        padding: 8px;
        display: flex;
        flex-direction: column;
        gap: 6px;
        font-size: 11px;
        color: #c7d8ec;
      }
      .pso-rig-block label { color: #99a4b3; display: flex; align-items: center; gap: 6px; }
      .pso-rig-toggle, .pso-rig-mode {
        display: flex; gap: 4px; align-items: center;
        padding: 4px 6px;
        border: 1px solid #2a313a;
        border-radius: 3px;
        background: rgba(0, 0, 0, 0.25);
      }
      .pso-rig-toggle button, .pso-rig-mode button {
        background: transparent;
        border: 1px solid #2a313a;
        color: #99a4b3;
        cursor: pointer;
        padding: 3px 8px;
        font: inherit;
        border-radius: 2px;
      }
      .pso-rig-toggle button.on {
        background: rgba(255, 144, 0, 0.18);
        border-color: #ffaa00;
        color: #ffaa00;
      }
      .pso-rig-mode button.on {
        background: rgba(0, 255, 255, 0.12);
        border-color: #00ffff;
        color: #00ffff;
      }
      .pso-rig-mode { gap: 3px; }
      .pso-rig-mode .grow { flex: 1; }
      .pso-rig-mode button { flex: 1; }
      .pso-rig-tree {
        max-height: 220px;
        overflow-y: auto;
        background: rgba(0, 0, 0, 0.25);
        border: 1px solid #2a313a;
        border-radius: 2px;
        padding: 2px;
      }
      .pso-rig-tree-row {
        display: flex; gap: 4px; padding: 1px 4px;
        cursor: pointer;
        border-radius: 2px;
        font-size: 10px;
        line-height: 1.4;
      }
      .pso-rig-tree-row:hover { background: rgba(74, 144, 226, 0.10); }
      .pso-rig-tree-row.active {
        background: rgba(0, 255, 255, 0.12);
        outline: 1px solid #00ffff;
      }
      .pso-rig-tree-row.selected {
        background: rgba(0, 255, 255, 0.05);
      }
      .pso-rig-tree-row.hidden { opacity: 0.4; }
      .pso-rig-tree-toggle {
        width: 12px; text-align: center; user-select: none;
        color: #6c7785;
      }
      .pso-rig-tree-toggle:hover { color: #c7d8ec; }
      .pso-rig-tree-eye {
        width: 14px; text-align: center; user-select: none;
        color: #6c7785;
      }
      .pso-rig-tree-eye:hover { color: #c7d8ec; }
      .pso-rig-tree-name {
        flex: 1;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
      }
      .pso-rig-tree-name.editing {
        outline: 1px solid #00ffff;
        background: rgba(0, 0, 0, 0.5);
        padding: 0 2px;
      }
      .pso-rig-tree-idx {
        color: #6c7785; font-variant-numeric: tabular-nums;
        min-width: 22px; text-align: right;
      }
      .pso-rig-inspector {
        background: rgba(0, 0, 0, 0.25);
        border: 1px solid #2a313a;
        border-radius: 2px;
        padding: 4px 6px;
        font-size: 10px;
      }
      .pso-rig-inspector .nm { color: #00ffff; font-weight: bold; }
      .pso-rig-inspector .row {
        display: grid;
        grid-template-columns: 50px 1fr 1fr 1fr;
        gap: 2px;
        margin: 2px 0;
        align-items: center;
      }
      .pso-rig-inspector .row span { color: #99a4b3; }
      .pso-rig-inspector .row input {
        background: rgba(0, 0, 0, 0.4);
        color: #c7d8ec;
        border: 1px solid #2a313a;
        font: inherit;
        font-size: 10px;
        padding: 1px 3px;
        text-align: right;
        font-variant-numeric: tabular-nums;
        width: 100%;
        box-sizing: border-box;
      }
      .pso-rig-inspector .row input:focus { border-color: #00ffff; outline: none; }
      .pso-rig-row { display: flex; gap: 6px; align-items: center; }
      .pso-rig-row .grow { flex: 1; }
      .pso-rig-row input[type=range] { flex: 1; }
      .pso-rig-row .num {
        min-width: 40px;
        font-variant-numeric: tabular-nums;
        text-align: right;
        color: #c7d8ec;
      }
      .pso-rig-actions {
        display: flex;
        gap: 4px;
        flex-wrap: wrap;
      }
      .pso-rig-actions button {
        background: transparent;
        border: 1px solid #2a313a;
        color: #99a4b3;
        cursor: pointer;
        padding: 3px 8px;
        font: inherit;
        border-radius: 2px;
      }
      .pso-rig-actions button:hover { border-color: #00ffff; color: #00ffff; }
      .pso-rig-actions button:disabled { opacity: 0.4; cursor: not-allowed; }
      .pso-rig-actions button.primary { border-color: #4a90e2; color: #c7d8ec; }
      .pso-rig-actions button.primary:hover { background: rgba(74, 144, 226, 0.18); border-color: #00ffff; color: #00ffff; }
      .pso-rig-actions button.danger { border-color: #4d2323; color: #d89090; }
      .pso-rig-actions button.danger:hover { border-color: #ff6680; color: #ff6680; }
      .pso-rig-stats {
        background: rgba(0, 0, 0, 0.25);
        border: 1px solid #2a313a;
        border-radius: 2px;
        padding: 4px 6px;
        font-size: 10px;
        color: #c7d8ec;
        white-space: pre-wrap;
        font-variant-numeric: tabular-nums;
      }
      .pso-rig-status { font-size: 10px; min-height: 12px; }
      .pso-rig-status.idle { color: #6c7785; }
      .pso-rig-status.running { color: #4a90e2; }
      .pso-rig-status.done { color: #56b67a; }
      .pso-rig-status.err  { color: #ff6680; }
      .pso-rig-legend {
        display: flex;
        gap: 4px;
        align-items: center;
        font-size: 9px;
        color: #99a4b3;
        background: rgba(0, 0, 0, 0.25);
        border: 1px solid #2a313a;
        border-radius: 2px;
        padding: 3px 4px;
      }
      .pso-rig-legend .bar {
        flex: 1;
        height: 6px;
        background: linear-gradient(to right, #2244aa, #44ff44, #ffaa00, #ff2244);
        border-radius: 2px;
      }
      .pso-rig-iklist {
        background: rgba(0, 0, 0, 0.25);
        border: 1px solid #2a313a;
        border-radius: 2px;
        padding: 2px;
        max-height: 100px;
        overflow-y: auto;
      }
      .pso-rig-ik-row {
        display: flex; gap: 4px; padding: 2px 4px;
        cursor: pointer; font-size: 10px;
      }
      .pso-rig-ik-row:hover { background: rgba(74, 144, 226, 0.10); }
      .pso-rig-ik-row.active {
        background: rgba(0, 255, 255, 0.12);
        outline: 1px solid #00ffff;
      }
      .pso-rig-ik-row .grow { flex: 1; }
    `;
    document.head.appendChild(style);
  }

  // -------------------------------------------------------------------
  // Falloff math (mirror of formats.sculpt.falloff)
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
  // Status helpers
  // -------------------------------------------------------------------
  function setStatus(stateStr, msg) {
    if (!state.statusEl) return;
    state.statusEl.textContent = msg;
    state.statusEl.className = "pso-rig-status " + stateStr;
  }

  // -------------------------------------------------------------------
  // FNV-1a 32-bit (matches sculpt panel's source-SHA)
  // -------------------------------------------------------------------
  function computeSourceShaSync() {
    let h = 0x811c9dc5;
    const ctx = window.psoGetRigContext ? window.psoGetRigContext() : null;
    if (ctx && ctx.modelPath) {
      const bytes = new TextEncoder().encode(ctx.modelPath);
      for (const b of bytes) {
        h ^= b;
        h = (h * 0x01000193) >>> 0;
      }
    }
    return h.toString(16).padStart(16, "0").slice(0, 16);
  }

  // -------------------------------------------------------------------
  // Bone widgets — wireframe spheres + parent-child lines
  // -------------------------------------------------------------------
  function rebuildBoneWidgets() {
    const ctx = window.psoGetRigContext ? window.psoGetRigContext() : null;
    if (!ctx) return;
    const { THREE, scene, group } = ctx;
    if (state.boneWidgets) {
      try {
        if (state.boneWidgets.parent) state.boneWidgets.parent.remove(state.boneWidgets);
        state.boneWidgets.traverse((c) => {
          if (c.geometry) c.geometry.dispose();
          if (c.material) {
            if (Array.isArray(c.material)) c.material.forEach((m) => m.dispose());
            else c.material.dispose();
          }
        });
      } catch (_e) {}
      state.boneWidgets = null;
      state.boneWidgetMeta = [];
    }
    const skel = state.skeleton;
    if (!skel || skel.length === 0) return;
    const widgets = new THREE.Group();
    widgets.name = "pso-rig-bone-widgets";
    // Render in WORLD space — we'll add to scene, NOT into group, so the
    // widgets aren't affected by the group's centering scale.
    const worldPositions = window.psoGetBoneWorldPositions
      ? window.psoGetBoneWorldPositions()
      : null;
    if (!worldPositions) return;
    const meta = [];
    const sphereGeo = new THREE.SphereGeometry(1.0, 8, 6);
    for (let i = 0; i < skel.length; i++) {
      const wp = worldPositions[i];
      const groupWp = window.psoBoneSpaceToWorld
        ? window.psoBoneSpaceToWorld(wp)
        : wp;
      const mat = new THREE.MeshBasicMaterial({
        color: i === state.activeBoneIdx ? 0x00ffff : 0xffaa00,
        wireframe: true,
        depthTest: false,
        transparent: true,
        opacity: 0.9,
      });
      const sphere = new THREE.Mesh(sphereGeo, mat);
      // Size in WORLD units — scale relative to the group's scale so
      // widgets stay roughly the same screen size across models. The
      // group has been centered + scaled to fit; we use a tiny fixed
      // factor of the canvas's screen extent (0.04 world units after
      // the group's reciprocal scale).
      const groupScale = (group.scale && group.scale.x) || 1.0;
      const r = 0.04 / Math.max(0.0001, groupScale);
      sphere.scale.set(r, r, r);
      sphere.position.set(groupWp[0], groupWp[1], groupWp[2]);
      sphere.userData.boneIdx = i;
      sphere.renderOrder = 10;
      widgets.add(sphere);
      meta.push({ sphere, idx: i });
    }
    // Parent-child lines.
    const linePositions = [];
    for (let i = 0; i < skel.length; i++) {
      const b = skel[i];
      if (b.parent < 0) continue;
      const cp = window.psoBoneSpaceToWorld(worldPositions[i]);
      const pp = window.psoBoneSpaceToWorld(worldPositions[b.parent]);
      linePositions.push(pp[0], pp[1], pp[2]);
      linePositions.push(cp[0], cp[1], cp[2]);
    }
    if (linePositions.length > 0) {
      const lineGeo = new THREE.BufferGeometry();
      lineGeo.setAttribute(
        "position",
        new THREE.BufferAttribute(new Float32Array(linePositions), 3),
      );
      const lineMat = new THREE.LineBasicMaterial({
        color: 0x99aabb,
        transparent: true,
        opacity: 0.5,
        depthTest: false,
      });
      const lines = new THREE.LineSegments(lineGeo, lineMat);
      lines.renderOrder = 9;
      widgets.add(lines);
    }
    scene.add(widgets);
    state.boneWidgets = widgets;
    state.boneWidgetMeta = meta;
  }

  function refreshBoneWidgetVisibility() {
    if (!state.boneWidgets) return;
    state.boneWidgets.visible = state.enabled;
  }

  // Visit each tree row and update its highlight state.
  function refreshTree() {
    if (!state.treeEl) return;
    const rows = state.treeEl.querySelectorAll("[data-bone-idx]");
    rows.forEach((row) => {
      const bi = parseInt(row.getAttribute("data-bone-idx"), 10);
      row.classList.toggle("active", bi === state.activeBoneIdx);
      row.classList.toggle("selected", state.selectedBones.has(bi));
      row.classList.toggle("hidden", state.hiddenBones.has(bi));
    });
    if (state.boneWidgets && state.boneWidgetMeta) {
      for (const m of state.boneWidgetMeta) {
        if (!m.sphere || !m.sphere.material) continue;
        const isActive = m.idx === state.activeBoneIdx;
        const isSelected = state.selectedBones.has(m.idx);
        const isHidden = state.hiddenBones.has(m.idx);
        m.sphere.visible = !isHidden;
        m.sphere.material.color.setHex(
          isActive ? 0x00ffff : (isSelected ? 0x66ff99 : 0xffaa00),
        );
      }
    }
  }

  function refreshInspector() {
    if (!state.inspectorEl) return;
    const idx = state.activeBoneIdx;
    const skel = state.skeleton;
    if (!skel || idx < 0 || idx >= skel.length) {
      state.inspectorEl.innerHTML = `<div class="dim">no bone selected</div>`;
      return;
    }
    const b = skel[idx];
    const name = state.boneNames.get(idx) || `bone${idx}`;
    state.inspectorEl.innerHTML = `
      <div><span class="nm">${escapeHtml(name)}</span> <span style="color:#6c7785">#${idx} parent #${b.parent}</span></div>
      <div class="row">
        <span>pos</span>
        <input type="number" step="0.001" data-trs="px" value="${b.position[0].toFixed(4)}">
        <input type="number" step="0.001" data-trs="py" value="${b.position[1].toFixed(4)}">
        <input type="number" step="0.001" data-trs="pz" value="${b.position[2].toFixed(4)}">
      </div>
      <div class="row">
        <span>rot°</span>
        <input type="number" step="0.5" data-trs="rx" value="${(b.rotation_bams[0] * _BAMS_TO_RAD * 180 / Math.PI).toFixed(2)}">
        <input type="number" step="0.5" data-trs="ry" value="${(b.rotation_bams[1] * _BAMS_TO_RAD * 180 / Math.PI).toFixed(2)}">
        <input type="number" step="0.5" data-trs="rz" value="${(b.rotation_bams[2] * _BAMS_TO_RAD * 180 / Math.PI).toFixed(2)}">
      </div>
      <div class="row">
        <span>scl</span>
        <input type="number" step="0.01" data-trs="sx" value="${(b.scale ? b.scale[0] : 1).toFixed(3)}">
        <input type="number" step="0.01" data-trs="sy" value="${(b.scale ? b.scale[1] : 1).toFixed(3)}">
        <input type="number" step="0.01" data-trs="sz" value="${(b.scale ? b.scale[2] : 1).toFixed(3)}">
      </div>
      <div class="row" style="grid-template-columns: 50px 1fr">
        <span>flags</span>
        <span style="color:#6c7785">0x${b.eval_flags.toString(16).padStart(2, "0")}</span>
      </div>
    `;
    state.inspectorEl.querySelectorAll("[data-trs]").forEach((inp) => {
      inp.addEventListener("change", onTrsInput);
    });
  }

  function onTrsInput(ev) {
    const t = ev.target;
    const k = t.getAttribute("data-trs");
    const v = parseFloat(t.value);
    if (!isFinite(v)) return;
    const idx = state.activeBoneIdx;
    if (idx < 0) return;
    const skel = state.skeleton;
    if (!skel || idx >= skel.length) return;
    const b = skel[idx];
    if (k === "px") b.position[0] = v;
    else if (k === "py") b.position[1] = v;
    else if (k === "pz") b.position[2] = v;
    else if (k === "rx") b.rotation_bams[0] = (v * Math.PI / 180 * RAD_TO_BAMS) | 0;
    else if (k === "ry") b.rotation_bams[1] = (v * Math.PI / 180 * RAD_TO_BAMS) | 0;
    else if (k === "rz") b.rotation_bams[2] = (v * Math.PI / 180 * RAD_TO_BAMS) | 0;
    else if (k === "sx") b.scale = [v, b.scale ? b.scale[1] : 1, b.scale ? b.scale[2] : 1];
    else if (k === "sy") b.scale = [b.scale ? b.scale[0] : 1, v, b.scale ? b.scale[2] : 1];
    else if (k === "sz") b.scale = [b.scale ? b.scale[0] : 1, b.scale ? b.scale[1] : 1, v];
    pushBoneOverride(idx, b);
    rebake();
    rebuildBoneWidgets();
  }

  function pushBoneOverride(idx, b) {
    if (typeof window.psoSetBonePoseOverride !== "function") return;
    window.psoSetBonePoseOverride(idx, {
      position: [b.position[0], b.position[1], b.position[2]],
      rotation_bams: [b.rotation_bams[0] | 0, b.rotation_bams[1] | 0, b.rotation_bams[2] | 0],
      scale: b.scale ? [b.scale[0], b.scale[1], b.scale[2]] : [1, 1, 1],
    });
  }

  function rebake() {
    if (typeof window.psoApplyRigBake === "function") {
      window.psoApplyRigBake();
    }
  }

  // -------------------------------------------------------------------
  // Bone tree builder
  // -------------------------------------------------------------------
  function buildBoneTreeHtml() {
    const skel = state.skeleton;
    if (!skel || skel.length === 0) {
      return `<div class="dim">no skinned model loaded</div>`;
    }
    const childrenOf = new Map();
    for (let i = 0; i < skel.length; i++) {
      childrenOf.set(i, []);
    }
    for (let i = 0; i < skel.length; i++) {
      const p = skel[i].parent;
      if (p >= 0 && childrenOf.has(p)) childrenOf.get(p).push(i);
    }
    const out = [];
    function emit(idx, depth) {
      const indent = "&nbsp;".repeat(depth * 2);
      const name = state.boneNames.get(idx) || `bone${idx}`;
      const isHidden = state.hiddenBones.has(idx);
      const isActive = idx === state.activeBoneIdx;
      const isSel = state.selectedBones.has(idx);
      const cls = `pso-rig-tree-row${isActive ? " active" : ""}${isSel ? " selected" : ""}${isHidden ? " hidden" : ""}`;
      out.push(`<div class="${cls}" data-bone-idx="${idx}">
        <span class="pso-rig-tree-toggle" data-act="toggle">${childrenOf.get(idx).length > 0 ? "·" : "·"}</span>
        <span class="pso-rig-tree-eye" data-act="eye" title="${isHidden ? "show" : "hide"} bone">${isHidden ? "○" : "●"}</span>
        <span class="pso-rig-tree-name" data-act="name">${indent}${escapeHtml(name)}</span>
        <span class="pso-rig-tree-idx">#${idx}</span>
      </div>`);
      for (const c of childrenOf.get(idx)) emit(c, depth + 1);
    }
    for (let i = 0; i < skel.length; i++) {
      if (skel[i].parent < 0) emit(i, 0);
    }
    return out.join("");
  }

  function refreshTreeHtml() {
    if (!state.treeEl) return;
    state.treeEl.innerHTML = buildBoneTreeHtml();
  }

  function selectBone(idx, opts) {
    opts = opts || {};
    if (opts.append) {
      if (state.selectedBones.has(idx)) state.selectedBones.delete(idx);
      else state.selectedBones.add(idx);
    } else {
      state.selectedBones.clear();
      state.selectedBones.add(idx);
    }
    state.activeBoneIdx = idx;
    refreshTree();
    refreshInspector();
  }

  function onTreeClick(ev) {
    const target = ev.target;
    const row = target.closest("[data-bone-idx]");
    if (!row) return;
    const idx = parseInt(row.getAttribute("data-bone-idx"), 10);
    const act = target.getAttribute("data-act");
    if (act === "eye") {
      ev.stopPropagation();
      if (state.hiddenBones.has(idx)) state.hiddenBones.delete(idx);
      else state.hiddenBones.add(idx);
      refreshTree();
      const eye = row.querySelector('[data-act="eye"]');
      if (eye) eye.textContent = state.hiddenBones.has(idx) ? "○" : "●";
      return;
    }
    if (act === "name" && ev.detail >= 2) {
      // Double-click → rename
      ev.stopPropagation();
      startRenameBone(idx, target);
      return;
    }
    selectBone(idx, { append: ev.shiftKey || ev.ctrlKey });
  }

  function startRenameBone(idx, nameEl) {
    if (!nameEl) return;
    const cur = state.boneNames.get(idx) || `bone${idx}`;
    nameEl.classList.add("editing");
    nameEl.contentEditable = "true";
    nameEl.textContent = cur;
    nameEl.focus();
    // Select all
    const range = document.createRange();
    range.selectNodeContents(nameEl);
    const sel = window.getSelection();
    sel.removeAllRanges(); sel.addRange(range);
    function commit() {
      nameEl.contentEditable = "false";
      nameEl.classList.remove("editing");
      const newName = (nameEl.textContent || "").trim();
      if (newName && newName !== cur) {
        state.boneNames.set(idx, newName);
      }
      refreshTreeHtml();
      refreshInspector();
      nameEl.removeEventListener("blur", commit);
      nameEl.removeEventListener("keydown", onKey);
    }
    function onKey(e) {
      if (e.key === "Enter" || e.key === "Escape") {
        e.preventDefault();
        nameEl.blur();
      }
    }
    nameEl.addEventListener("blur", commit);
    nameEl.addEventListener("keydown", onKey);
  }

  // -------------------------------------------------------------------
  // Pointer handling — three different modes share one pointer pipe.
  // -------------------------------------------------------------------
  let _activePointerId = null;
  let _drag = null;  // { type: "bone" | "ik" | "weight", ... }
  let _raycaster = null;

  function ensureRaycaster() {
    const ctx = window.psoGetRigContext ? window.psoGetRigContext() : null;
    if (!ctx) return null;
    if (!_raycaster) _raycaster = new ctx.THREE.Raycaster();
    return ctx;
  }

  function ndcFromEvent(ev, canvas) {
    const rect = canvas.getBoundingClientRect();
    const x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -(((ev.clientY - rect.top) / rect.height) * 2 - 1);
    return { x, y };
  }

  function rayhitBone(ev) {
    const ctx = ensureRaycaster();
    if (!ctx) return null;
    if (!state.boneWidgets) return null;
    const cv = document.getElementById("modelCanvas");
    if (!cv) return null;
    const { THREE, camera } = ctx;
    const ndc = ndcFromEvent(ev, cv);
    _raycaster.setFromCamera(new THREE.Vector2(ndc.x, ndc.y), camera);
    const meshes = state.boneWidgetMeta.map((m) => m.sphere).filter(Boolean);
    const hits = _raycaster.intersectObjects(meshes, false);
    if (!hits.length) return null;
    const obj = hits[0].object;
    return { boneIdx: obj.userData.boneIdx, worldPoint: hits[0].point };
  }

  function rayhitMesh(ev) {
    const ctx = ensureRaycaster();
    if (!ctx) return null;
    const cv = document.getElementById("modelCanvas");
    if (!cv) return null;
    const { THREE, camera, debugMeshes } = ctx;
    const ndc = ndcFromEvent(ev, cv);
    _raycaster.setFromCamera(new THREE.Vector2(ndc.x, ndc.y), camera);
    const meshes = debugMeshes.map((d) => d.mesh).filter(Boolean);
    const hits = _raycaster.intersectObjects(meshes, false);
    if (!hits.length) return null;
    const hitMesh = hits[0].object;
    const submeshIdx = debugMeshes.findIndex((d) => d.mesh === hitMesh);
    if (submeshIdx < 0) return null;
    const mInv = new THREE.Matrix4().copy(hitMesh.matrixWorld).invert();
    const localPoint = hits[0].point.clone().applyMatrix4(mInv);
    return { submeshIdx, mesh: hitMesh, worldPoint: hits[0].point, localPoint };
  }

  function onPointerDown(ev) {
    if (!state.enabled) return;
    if (ev.button !== 0) return;
    const cv = ev.target;
    if (!cv || cv.id !== "modelCanvas") return;
    ev.preventDefault();
    ev.stopPropagation();
    _activePointerId = ev.pointerId;
    cv.setPointerCapture(ev.pointerId);

    if (state.mode === MODE_SKELETON) {
      const hit = rayhitBone(ev);
      if (hit) {
        selectBone(hit.boneIdx, { append: ev.shiftKey || ev.ctrlKey });
        // Begin drag.
        _drag = {
          type: "bone",
          boneIdx: hit.boneIdx,
          startWorld: hit.worldPoint.clone(),
          shift: ev.shiftKey,
          alt: ev.altKey,
          beforePose: snapshotBonePose(hit.boneIdx),
        };
      }
      return;
    }
    if (state.mode === MODE_WEIGHT) {
      const hit = rayhitMesh(ev);
      if (!hit) return;
      _drag = { type: "weight", submeshIdx: hit.submeshIdx, before: snapshotSubmeshWeights(hit.submeshIdx) };
      paintStrokeStep(ev);
      return;
    }
    if (state.mode === MODE_IK) {
      const hitIk = rayhitIkMarker(ev);
      if (hitIk) {
        state.activeIkIdx = hitIk.ikIdx;
        _drag = {
          type: "ik",
          ikIdx: hitIk.ikIdx,
          startWorld: hitIk.worldPoint.clone(),
        };
        refreshIkList();
        return;
      }
      // No marker hit; pick a bone instead.
      const hit = rayhitBone(ev);
      if (hit) selectBone(hit.boneIdx);
      return;
    }
  }

  function onPointerMove(ev) {
    if (_activePointerId == null) return;
    if (ev.pointerId !== _activePointerId) return;
    if (!_drag) return;
    if (_drag.type === "bone") boneDragStep(ev);
    else if (_drag.type === "weight") paintStrokeStep(ev);
    else if (_drag.type === "ik") ikDragStep(ev);
  }

  function onPointerUp(ev) {
    if (_activePointerId == null) return;
    if (ev.pointerId !== _activePointerId) return;
    const cv = ev.target;
    if (cv && cv.releasePointerCapture) {
      try { cv.releasePointerCapture(ev.pointerId); } catch {}
    }
    _activePointerId = null;
    if (_drag && _drag.type === "weight") finalizeWeightStroke();
    if (_drag && _drag.type === "bone") finalizeBoneDrag();
    if (_drag && _drag.type === "ik") finalizeIkDrag();
    _drag = null;
  }

  // -------------------------------------------------------------------
  // Bone drag (Skeleton mode)
  // -------------------------------------------------------------------
  function snapshotBonePose(idx) {
    const skel = state.skeleton;
    if (!skel || idx < 0 || idx >= skel.length) return null;
    const b = skel[idx];
    return {
      position: [b.position[0], b.position[1], b.position[2]],
      rotation_bams: [b.rotation_bams[0], b.rotation_bams[1], b.rotation_bams[2]],
      scale: b.scale ? [b.scale[0], b.scale[1], b.scale[2]] : [1, 1, 1],
    };
  }

  function boneDragStep(ev) {
    if (!_drag || _drag.type !== "bone") return;
    const ctx = ensureRaycaster();
    if (!ctx) return;
    const { THREE, camera } = ctx;
    const cv = document.getElementById("modelCanvas");
    if (!cv) return;
    // Project the cursor onto a plane through the bone perpendicular
    // to the camera's forward.
    const ndc = ndcFromEvent(ev, cv);
    _raycaster.setFromCamera(new THREE.Vector2(ndc.x, ndc.y), camera);
    const camForward = new THREE.Vector3();
    camera.getWorldDirection(camForward);
    const plane = new THREE.Plane();
    plane.setFromNormalAndCoplanarPoint(camForward, _drag.startWorld);
    const newPoint = new THREE.Vector3();
    if (!_raycaster.ray.intersectPlane(plane, newPoint)) return;
    // Convert world delta to mesh-group local delta.
    const ctx2 = window.psoGetRigContext();
    const group = ctx2.group;
    const groupInv = new THREE.Matrix4().copy(group.matrixWorld).invert();
    const startLocal = _drag.startWorld.clone().applyMatrix4(groupInv);
    const newLocal = newPoint.clone().applyMatrix4(groupInv);
    const dx = newLocal.x - startLocal.x;
    const dy = newLocal.y - startLocal.y;
    const dz = newLocal.z - startLocal.z;
    const skel = state.skeleton;
    const idx = _drag.boneIdx;
    if (!skel || idx < 0 || idx >= skel.length) return;
    const b = skel[idx];
    if (_drag.shift) {
      // Rotate around screen-Z axis based on cursor angle delta from
      // bone center. A simple visual rotation: dx becomes rz delta.
      const oldRot = _drag.beforePose.rotation_bams.slice();
      const rzDelta = (dx + dy) * 4096; // BAMS per local-unit
      b.rotation_bams[2] = ((oldRot[2] | 0) + rzDelta) | 0;
    } else if (_drag.alt) {
      // Uniform scale based on dx magnitude.
      const oldScl = _drag.beforePose.scale;
      const sd = 1.0 + (dx + dy) * 0.5;
      b.scale = [oldScl[0] * sd, oldScl[1] * sd, oldScl[2] * sd];
    } else {
      // Translate.
      const oldPos = _drag.beforePose.position;
      b.position[0] = oldPos[0] + dx;
      b.position[1] = oldPos[1] + dy;
      b.position[2] = oldPos[2] + dz;
    }
    pushBoneOverride(idx, b);
    rebake();
    rebuildBoneWidgets();
    refreshInspector();
  }

  function finalizeBoneDrag() {
    if (!_drag || _drag.type !== "bone") return;
    const idx = _drag.boneIdx;
    const after = snapshotBonePose(idx);
    state.boneOverridesUndo.push({ boneIdx: idx, before: _drag.beforePose, after });
    while (state.boneOverridesUndo.length > 50) state.boneOverridesUndo.shift();
    _drag = null;
  }

  // -------------------------------------------------------------------
  // Weight Paint mode
  // -------------------------------------------------------------------
  function ensureWeightsForSubmesh(submeshIdx) {
    let cur = state.submeshWeights.get(submeshIdx);
    if (cur) return cur;
    // Seed from the bind pose's single-influence bone_idx[].
    const boneIdx = (typeof window.psoGetSubmeshBoneIndices === "function")
      ? window.psoGetSubmeshBoneIndices(submeshIdx)
      : null;
    if (!boneIdx) return null;
    const vc = boneIdx.length;
    const indices = new Int32Array(vc * MAX_INFLUENCES).fill(-1);
    const weights = new Float32Array(vc * MAX_INFLUENCES);
    for (let i = 0; i < vc; i++) {
      indices[i * MAX_INFLUENCES] = boneIdx[i];
      weights[i * MAX_INFLUENCES] = boneIdx[i] >= 0 ? 1.0 : 0.0;
    }
    cur = { indices, weights };
    state.submeshWeights.set(submeshIdx, cur);
    state.submeshOriginalBoneIdx.set(submeshIdx, new Int32Array(boneIdx));
    pushWeights(submeshIdx);
    return cur;
  }

  function pushWeights(submeshIdx) {
    const w = state.submeshWeights.get(submeshIdx);
    if (!w) return;
    if (typeof window.psoSetVertexWeights === "function") {
      window.psoSetVertexWeights(submeshIdx, w.indices, w.weights);
    }
  }

  function snapshotSubmeshWeights(submeshIdx) {
    const w = state.submeshWeights.get(submeshIdx);
    if (!w) return null;
    return {
      indices: new Int32Array(w.indices),
      weights: new Float32Array(w.weights),
    };
  }

  function paintStrokeStep(ev) {
    if (state.activeBoneIdx < 0) {
      setStatus("idle", "weight paint: pick a bone in the tree first");
      return;
    }
    const hit = rayhitMesh(ev);
    if (!hit) return;
    const w = ensureWeightsForSubmesh(hit.submeshIdx);
    if (!w) return;
    const localPositions = (typeof window.psoGetSubmeshLocalPositions === "function")
      ? window.psoGetSubmeshLocalPositions(hit.submeshIdx)
      : null;
    if (!localPositions) return;
    const r = state.weightBrush.radius;
    const r2 = r * r;
    const cx = hit.localPoint.x;
    const cy = hit.localPoint.y;
    const cz = hit.localPoint.z;
    const target = state.activeBoneIdx;
    const stren = state.weightBrush.strength;
    const curve = state.weightBrush.falloff;
    const indices = w.indices;
    const weights = w.weights;
    const vc = (localPositions.length / 3) | 0;
    let touched = 0;
    for (let vi = 0; vi < vc; vi++) {
      const px = localPositions[vi * 3 + 0];
      const py = localPositions[vi * 3 + 1];
      const pz = localPositions[vi * 3 + 2];
      const dx = px - cx; const dy = py - cy; const dz = pz - cz;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 >= r2) continue;
      const dist = Math.sqrt(d2);
      const wgt = falloff(dist / r, curve) * stren;
      if (wgt <= 0) continue;
      // Find target's slot, or carve a new one.
      const off = vi * MAX_INFLUENCES;
      let slot = -1;
      for (let k = 0; k < MAX_INFLUENCES; k++) {
        if (indices[off + k] === target) { slot = k; break; }
      }
      if (slot < 0) {
        // Find an empty slot.
        for (let k = 0; k < MAX_INFLUENCES; k++) {
          if (indices[off + k] < 0) { slot = k; break; }
        }
      }
      if (slot < 0) {
        // Replace the smallest-weight slot.
        let kmin = 0;
        let wmin = weights[off];
        for (let k = 1; k < MAX_INFLUENCES; k++) {
          if (weights[off + k] < wmin) { wmin = weights[off + k]; kmin = k; }
        }
        slot = kmin;
      }
      indices[off + slot] = target;
      weights[off + slot] = Math.min(1.0, weights[off + slot] + wgt);
      // Pull-down other slots proportionally so total stays in [0, 1].
      // After increasing slot's weight, scale the remaining slots so
      // sum of all weights = 1. This mirrors what blender does.
      let sum = 0;
      for (let k = 0; k < MAX_INFLUENCES; k++) sum += weights[off + k];
      if (sum > 1.0) {
        const scale = 1.0 / sum;
        for (let k = 0; k < MAX_INFLUENCES; k++) weights[off + k] *= scale;
      }
      touched++;
    }
    if (touched > 0) {
      pushWeights(hit.submeshIdx);
      rebake();
      if (state.showHeatmap) updateHeatmapColors();
    }
  }

  function finalizeWeightStroke() {
    if (!_drag || _drag.type !== "weight") return;
    const submeshIdx = _drag.submeshIdx;
    const before = _drag.before;
    const after = snapshotSubmeshWeights(submeshIdx);
    state.weightUndo.push({ submeshIdx, before, after });
    while (state.weightUndo.length > 50) state.weightUndo.shift();
    setStatus("done", `painted weights on submesh ${submeshIdx}`);
  }

  function normalizeAllWeights() {
    let n = 0;
    for (const [si, w] of state.submeshWeights) {
      const indices = w.indices;
      const weights = w.weights;
      const vc = (indices.length / MAX_INFLUENCES) | 0;
      for (let vi = 0; vi < vc; vi++) {
        const off = vi * MAX_INFLUENCES;
        let sum = 0;
        for (let k = 0; k < MAX_INFLUENCES; k++) sum += Math.max(0, weights[off + k]);
        if (sum > 1e-9) {
          const inv = 1.0 / sum;
          for (let k = 0; k < MAX_INFLUENCES; k++) weights[off + k] = Math.max(0, weights[off + k]) * inv;
        }
      }
      pushWeights(si);
      n++;
    }
    rebake();
    return n;
  }

  // -------------------------------------------------------------------
  // Weight heatmap visualisation
  // -------------------------------------------------------------------
  function updateHeatmapColors() {
    const ctx = window.psoGetRigContext ? window.psoGetRigContext() : null;
    if (!ctx) return;
    if (state.activeBoneIdx < 0) return;
    if (!state.showHeatmap) return;
    const target = state.activeBoneIdx;
    const debugMeshes = ctx.debugMeshes || [];
    for (let s = 0; s < debugMeshes.length; s++) {
      const m = debugMeshes[s];
      if (!m || !m.mesh || !m.mesh.geometry) continue;
      const w = state.submeshWeights.get(s);
      const geo = m.mesh.geometry;
      const posAttr = geo.getAttribute("position");
      if (!posAttr) continue;
      const vc = posAttr.count;
      let colorAttr = geo.getAttribute("color");
      if (!colorAttr || colorAttr.count !== vc) {
        const arr = new Float32Array(vc * 3);
        for (let i = 0; i < arr.length; i++) arr[i] = 1.0;
        colorAttr = new ctx.THREE.BufferAttribute(arr, 3);
        geo.setAttribute("color", colorAttr);
      }
      const carr = colorAttr.array;
      if (w) {
        for (let vi = 0; vi < vc; vi++) {
          const off = vi * MAX_INFLUENCES;
          let weight = 0;
          for (let k = 0; k < MAX_INFLUENCES; k++) {
            if (w.indices[off + k] === target) {
              weight = w.weights[off + k];
              break;
            }
          }
          // Map [0..1] -> blue..green..yellow..red.
          const t = Math.max(0, Math.min(1, weight));
          // Use a simple HSV-like ramp.
          let r, g, b;
          if (t < 0.33) {
            const u = t / 0.33;
            r = 0.13 * (1 - u) + 0.27 * u;
            g = 0.27 * (1 - u) + 1.0 * u;
            b = 0.67 * (1 - u) + 0.27 * u;
          } else if (t < 0.66) {
            const u = (t - 0.33) / 0.33;
            r = 0.27 * (1 - u) + 1.0 * u;
            g = 1.0;
            b = 0.27 * (1 - u);
          } else {
            const u = (t - 0.66) / 0.34;
            r = 1.0;
            g = 1.0 * (1 - u) + 0.13 * u;
            b = 0;
          }
          carr[vi * 3 + 0] = r;
          carr[vi * 3 + 1] = g;
          carr[vi * 3 + 2] = b;
        }
      } else {
        // No weights yet — neutral white.
        for (let vi = 0; vi < vc; vi++) {
          carr[vi * 3] = 1.0; carr[vi * 3 + 1] = 1.0; carr[vi * 3 + 2] = 1.0;
        }
      }
      colorAttr.needsUpdate = true;
      // Enable vertexColors on the material.
      if (m.mesh.material) {
        m.mesh.material.vertexColors = true;
        m.mesh.material.needsUpdate = true;
      }
    }
  }

  function clearHeatmapColors() {
    const ctx = window.psoGetRigContext ? window.psoGetRigContext() : null;
    if (!ctx) return;
    const debugMeshes = ctx.debugMeshes || [];
    for (const m of debugMeshes) {
      if (m.mesh && m.mesh.material) {
        m.mesh.material.vertexColors = false;
        m.mesh.material.needsUpdate = true;
      }
    }
  }

  // -------------------------------------------------------------------
  // IK mode
  // -------------------------------------------------------------------
  function rayhitIkMarker(ev) {
    const ctx = ensureRaycaster();
    if (!ctx) return null;
    const cv = document.getElementById("modelCanvas");
    if (!cv) return null;
    const { THREE, camera } = ctx;
    const ndc = ndcFromEvent(ev, cv);
    _raycaster.setFromCamera(new THREE.Vector2(ndc.x, ndc.y), camera);
    const markers = state.ikTargets.map((t) => t.marker).filter(Boolean);
    const hits = _raycaster.intersectObjects(markers, false);
    if (!hits.length) return null;
    const obj = hits[0].object;
    const ikIdx = obj.userData.ikIdx;
    return { ikIdx, worldPoint: hits[0].point };
  }

  function addIkTarget() {
    if (state.activeBoneIdx < 0) {
      setStatus("err", "pick a bone first");
      return;
    }
    const wpos = window.psoGetBoneWorldPositions
      ? window.psoGetBoneWorldPositions()
      : null;
    if (!wpos) return;
    const localPos = wpos[state.activeBoneIdx] || [0, 0, 0];
    const worldPos = window.psoBoneSpaceToWorld(localPos);
    const ik = {
      boneIdx: state.activeBoneIdx,
      chainLen: 2,
      target: worldPos.slice(),
      iterations: 16,
      name: state.boneNames.get(state.activeBoneIdx) || `bone${state.activeBoneIdx}`,
      marker: null,
    };
    state.ikTargets.push(ik);
    state.activeIkIdx = state.ikTargets.length - 1;
    rebuildIkMarkers();
    refreshIkList();
    setStatus("done", `added IK target on bone ${ik.boneIdx}`);
  }

  function removeIkTarget(ikIdx) {
    if (ikIdx < 0 || ikIdx >= state.ikTargets.length) return;
    const ik = state.ikTargets[ikIdx];
    if (ik.marker && ik.marker.parent) {
      ik.marker.parent.remove(ik.marker);
      try { ik.marker.geometry.dispose(); ik.marker.material.dispose(); } catch (_) {}
    }
    state.ikTargets.splice(ikIdx, 1);
    if (state.activeIkIdx === ikIdx) state.activeIkIdx = -1;
    else if (state.activeIkIdx > ikIdx) state.activeIkIdx--;
    refreshIkList();
  }

  function rebuildIkMarkers() {
    const ctx = window.psoGetRigContext ? window.psoGetRigContext() : null;
    if (!ctx) return;
    const { THREE, scene } = ctx;
    for (const ik of state.ikTargets) {
      if (!ik.marker) {
        const geo = new THREE.SphereGeometry(0.05, 12, 8);
        const mat = new THREE.MeshBasicMaterial({
          color: 0xff66cc, transparent: true, opacity: 0.85,
          depthTest: false,
        });
        const m = new THREE.Mesh(geo, mat);
        m.userData.ikIdx = state.ikTargets.indexOf(ik);
        m.renderOrder = 11;
        scene.add(m);
        ik.marker = m;
      }
      ik.marker.position.set(ik.target[0], ik.target[1], ik.target[2]);
    }
  }

  function ikDragStep(ev) {
    if (!_drag || _drag.type !== "ik") return;
    const ctx = ensureRaycaster();
    if (!ctx) return;
    const { THREE, camera } = ctx;
    const cv = document.getElementById("modelCanvas");
    if (!cv) return;
    const ndc = ndcFromEvent(ev, cv);
    _raycaster.setFromCamera(new THREE.Vector2(ndc.x, ndc.y), camera);
    const camForward = new THREE.Vector3();
    camera.getWorldDirection(camForward);
    const plane = new THREE.Plane();
    plane.setFromNormalAndCoplanarPoint(camForward, _drag.startWorld);
    const newPoint = new THREE.Vector3();
    if (!_raycaster.ray.intersectPlane(plane, newPoint)) return;
    const ik = state.ikTargets[_drag.ikIdx];
    if (!ik) return;
    ik.target = [newPoint.x, newPoint.y, newPoint.z];
    if (ik.marker) ik.marker.position.set(newPoint.x, newPoint.y, newPoint.z);
    solveIk(ik);
  }

  function finalizeIkDrag() {
    if (!_drag || _drag.type !== "ik") return;
    setStatus("done", `IK target moved`);
  }

  // FABRIK in WORLD space, then write angle deltas to bone overrides.
  // We collect the chain's world joints, solve, and for each pair of
  // adjacent joints adjust the bone's local rotation so its world tip
  // points at the new joint position.
  function solveIk(ik) {
    const ctx = window.psoGetRigContext();
    if (!ctx) return;
    const skel = state.skeleton;
    if (!skel) return;
    // Build chain: end-effector first → upward through parent.
    const chain = [];
    let cur = ik.boneIdx;
    for (let i = 0; i < ik.chainLen + 1; i++) {
      if (cur < 0 || cur >= skel.length) break;
      chain.push(cur);
      cur = skel[cur].parent;
    }
    chain.reverse();  // root → end
    if (chain.length < 2) return;
    const wpos = window.psoGetBoneWorldPositions();
    if (!wpos) return;
    const chainWorld = chain.map((bi) => window.psoBoneSpaceToWorld(wpos[bi]));
    // Append a virtual "tip" past the end bone so the FABRIK end joint
    // is one past the last bone (meaningful for single-bone chains).
    // Actually simpler: feed FABRIK with (root, ..., end) only — the
    // new positions tell us where each bone should be in world space.
    const target = ik.target;
    const newPositions = fabrikSolve(chainWorld, target, ik.iterations);
    // For each bone in the chain except the root, derive a new local
    // pose so that its world position matches newPositions[i].
    for (let i = 1; i < chain.length; i++) {
      const bi = chain[i];
      const parentIdx = chain[i - 1];
      const parentWorld = newPositions[i - 1];
      // Convert newPositions[i] (world) to parent-local.
      const childWorld = newPositions[i];
      // We need parent's world matrix to invert. Easiest: use the
      // fact that worldPos[i] - worldPos[parentIdx] approximates the
      // child's local position offset from the parent — when the
      // parent's rotation has not been touched. For arm/leg-style
      // chains where bones sit along a single axis this is exact.
      // For more complex chains the result is an approximation; v2
      // can implement full quaternion-based rebase.
      const dx = childWorld[0] - parentWorld[0];
      const dy = childWorld[1] - parentWorld[1];
      const dz = childWorld[2] - parentWorld[2];
      // Convert from WORLD space to mesh-group local (group has scale).
      const groupScale = (ctx.group.scale && ctx.group.scale.x) || 1.0;
      const inv = 1.0 / Math.max(0.0001, groupScale);
      // Bone's bind position in local space.
      const b = skel[bi];
      const newLocal = [dx * inv, dy * inv, dz * inv];
      const bindLocal = b.position;
      // For the simplest "move the joint" semantic, we only update
      // the BONE'S OWN position, not its parent's rotation. This is
      // less accurate than rotation-based IK but maps cleanly to the
      // direct-translate widgets and produces visually reasonable
      // results for the tail / arm cases the spec calls out.
      window.psoSetBonePoseOverride(bi, {
        position: newLocal,
        rotation_bams: [b.rotation_bams[0], b.rotation_bams[1], b.rotation_bams[2]],
        scale: b.scale ? [b.scale[0], b.scale[1], b.scale[2]] : [1, 1, 1],
      });
      // Also update our skeleton snapshot so widgets+inspector reflect.
      skel[bi].position = newLocal;
    }
    rebake();
    rebuildBoneWidgets();
  }

  function fabrikSolve(chain, target, iterations) {
    const n = chain.length;
    if (n < 2) return chain.map((p) => p.slice());
    const pts = chain.map((p) => [p[0], p[1], p[2]]);
    const segLen = [];
    for (let i = 0; i < n - 1; i++) {
      const dx = pts[i + 1][0] - pts[i][0];
      const dy = pts[i + 1][1] - pts[i][1];
      const dz = pts[i + 1][2] - pts[i][2];
      segLen.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
    }
    const totalReach = segLen.reduce((a, b) => a + b, 0);
    const root = pts[0];
    const dx = target[0] - root[0]; const dy = target[1] - root[1]; const dz = target[2] - root[2];
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (dist > totalReach) {
      if (dist < 1e-9) return pts;
      const out = [root];
      let cum = 0;
      for (let i = 0; i < n - 1; i++) {
        cum += segLen[i];
        const t = cum / dist;
        out.push([root[0] + dx * t, root[1] + dy * t, root[2] + dz * t]);
      }
      return out;
    }
    const it = Math.max(1, iterations | 0);
    for (let pass = 0; pass < it; pass++) {
      // Backward
      pts[n - 1] = [target[0], target[1], target[2]];
      for (let i = n - 2; i >= 0; i--) {
        const ax = pts[i][0] - pts[i + 1][0];
        const ay = pts[i][1] - pts[i + 1][1];
        const az = pts[i][2] - pts[i + 1][2];
        const d = Math.sqrt(ax * ax + ay * ay + az * az);
        if (d < 1e-9) {
          pts[i] = [pts[i + 1][0] + segLen[i], pts[i + 1][1], pts[i + 1][2]];
          continue;
        }
        const r = segLen[i] / d;
        pts[i] = [pts[i + 1][0] + ax * r, pts[i + 1][1] + ay * r, pts[i + 1][2] + az * r];
      }
      // Forward
      pts[0] = root;
      for (let i = 0; i < n - 1; i++) {
        const ax = pts[i + 1][0] - pts[i][0];
        const ay = pts[i + 1][1] - pts[i][1];
        const az = pts[i + 1][2] - pts[i][2];
        const d = Math.sqrt(ax * ax + ay * ay + az * az);
        if (d < 1e-9) {
          pts[i + 1] = [pts[i][0] + segLen[i], pts[i][1], pts[i][2]];
          continue;
        }
        const r = segLen[i] / d;
        pts[i + 1] = [pts[i][0] + ax * r, pts[i][1] + ay * r, pts[i][2] + az * r];
      }
      // Convergence check
      const ex = pts[n - 1][0] - target[0];
      const ey = pts[n - 1][1] - target[1];
      const ez = pts[n - 1][2] - target[2];
      if (ex * ex + ey * ey + ez * ez < 1e-6) break;
    }
    return pts;
  }

  // -------------------------------------------------------------------
  // Auto-skin
  // -------------------------------------------------------------------
  async function autoSkin(algorithm) {
    const ctx = window.psoGetRigContext ? window.psoGetRigContext() : null;
    if (!ctx) {
      setStatus("err", "no model loaded");
      return;
    }
    if (!ctx.modelPath) {
      setStatus("err", "no model path");
      return;
    }
    setStatus("running", `auto-skin (${algorithm})...`);
    try {
      const r = await fetch("/api/rig/auto_skin", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_path: ctx.modelPath,
          inner_idx: 0,
          algorithm,
          falloff: 4.0,
          iterations: 8,
          max_influences: MAX_INFLUENCES,
        }),
      });
      if (!r.ok) {
        let detail = `HTTP ${r.status}`;
        try { detail = (await r.json()).detail || detail; } catch {}
        throw new Error(detail);
      }
      const j = await r.json();
      if (!j.weights) throw new Error("auto_skin response missing weights");
      // Apply each submesh's weights.
      for (const w of j.weights) {
        const ind = new Int32Array(b64ToBuf(w.indices_b64));
        const wt = new Float32Array(b64ToBuf(w.weights_b64));
        state.submeshWeights.set(w.submesh_idx, { indices: ind, weights: wt });
        pushWeights(w.submesh_idx);
      }
      rebake();
      if (state.showHeatmap) updateHeatmapColors();
      setStatus("done", `auto-skin done · ${j.weights.length} submesh(es)`);
    } catch (e) {
      setStatus("err", `auto-skin failed: ${e.message || e}`);
    }
  }

  function b64ToBuf(b64) {
    const bin = atob(b64);
    const len = bin.length;
    const out = new Uint8Array(len);
    for (let i = 0; i < len; i++) out[i] = bin.charCodeAt(i);
    return out.buffer;
  }

  function bufToB64(buf) {
    const u8 = (buf instanceof ArrayBuffer) ? new Uint8Array(buf) : new Uint8Array(buf.buffer || buf);
    let s = "";
    const chunk = 0x8000;
    for (let i = 0; i < u8.length; i += chunk) {
      s += String.fromCharCode.apply(null, u8.subarray(i, i + chunk));
    }
    return btoa(s);
  }

  // -------------------------------------------------------------------
  // Save / Reset
  // -------------------------------------------------------------------
  async function saveRig() {
    const ctx = window.psoGetRigContext ? window.psoGetRigContext() : null;
    if (!ctx) {
      setStatus("err", "no model loaded");
      return null;
    }
    const skel = state.skeleton;
    if (!skel) {
      setStatus("err", "no skeleton");
      return null;
    }
    const bones = skel.map((b, i) => ({
      index: i,
      parent: b.parent,
      position: [b.position[0], b.position[1], b.position[2]],
      rotation_bams: [b.rotation_bams[0] | 0, b.rotation_bams[1] | 0, b.rotation_bams[2] | 0],
      scale: b.scale ? [b.scale[0], b.scale[1], b.scale[2]] : [1, 1, 1],
      name: state.boneNames.get(i) || "",
      eval_flags: b.eval_flags | 0,
      hidden: state.hiddenBones.has(i),
    }));
    const weights = [];
    for (const [si, w] of state.submeshWeights) {
      const vc = (w.indices.length / MAX_INFLUENCES) | 0;
      weights.push({
        submesh_idx: si,
        vertex_count: vc,
        indices_b64: bufToB64(w.indices.buffer),
        weights_b64: bufToB64(w.weights.buffer),
        max_influences: MAX_INFLUENCES,
      });
    }
    const ik_targets = state.ikTargets.map((ik) => ({
      bone_idx: ik.boneIdx,
      chain_length: ik.chainLen,
      target: [ik.target[0], ik.target[1], ik.target[2]],
      iterations: ik.iterations,
      name: ik.name || "",
    }));
    const sourceSha = state.sourceSha || computeSourceShaSync();
    const payload = {
      format_version: FORMAT_VERSION,
      source_path: ctx.modelPath,
      source_sha: sourceSha,
      subdivide_level: state.subdivideLevel | 0,
      skeleton: { bones },
      weights,
      ik_targets,
      saved_at_ms: Date.now(),
    };
    try {
      const r = await fetch("/api/rig/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_path: ctx.modelPath,
          rig_payload: payload,
          subdivide_level: state.subdivideLevel | 0,
        }),
      });
      if (!r.ok) {
        let detail = `HTTP ${r.status}`;
        try { detail = (await r.json()).detail || detail; } catch {}
        throw new Error(detail);
      }
      const j = await r.json();
      setStatus("done", `saved · sha ${j.sha.slice(0, 8)}…`);
      return j;
    } catch (e) {
      setStatus("err", `save failed: ${e.message || e}`);
      return null;
    }
  }

  function resetAll() {
    state.boneOverridesUndo = [];
    state.weightUndo = [];
    state.ikTargets = [];
    state.activeIkIdx = -1;
    state.submeshWeights.clear();
    state.submeshOriginalBoneIdx.clear();
    state.hiddenBones.clear();
    state.boneNames.clear();
    if (typeof window.psoClearBonePoseOverrides === "function") {
      window.psoClearBonePoseOverrides();
    }
    if (typeof window.psoSetVertexWeights === "function") {
      // Clear per-submesh weights.
      const ctx = window.psoGetRigContext ? window.psoGetRigContext() : null;
      if (ctx && ctx.debugMeshes) {
        for (let s = 0; s < ctx.debugMeshes.length; s++) {
          window.psoSetVertexWeights(s, null, null);
        }
      }
    }
    state.skeleton = window.psoGetSkeleton ? window.psoGetSkeleton() : null;
    rebake();
    rebuildBoneWidgets();
    refreshIkMarkersClear();
    refreshTreeHtml();
    refreshInspector();
    refreshIkList();
    clearHeatmapColors();
    setStatus("done", "reverted to source");
  }

  function refreshIkMarkersClear() {
    const ctx = window.psoGetRigContext ? window.psoGetRigContext() : null;
    if (!ctx) return;
    // Markers were already attached to the scene; ikTargets may be []
    // by now, but old markers might persist if state was mutated
    // independently. Defensive cleanup.
    if (state.ikTargets.length === 0 && ctx.scene) {
      ctx.scene.children
        .filter((c) => c.userData && c.userData.ikIdx != null)
        .forEach((c) => {
          ctx.scene.remove(c);
          try { c.geometry.dispose(); c.material.dispose(); } catch (_) {}
        });
    }
  }

  // -------------------------------------------------------------------
  // Render the main panel body
  // -------------------------------------------------------------------
  function renderRigBlock(body) {
    ensureStyleInjected();
    state.bodyEl = body;
    body.innerHTML = `
      <div class="pso-rig-block">
        <div class="pso-rig-toggle">
          <span class="grow" style="color:#99a4b3">Rig mode:</span>
          <button data-act="toggle" class="${state.enabled ? "on" : ""}"
                  title="enable click-to-rig (LMB sculpt-bone / paint-weight / drag-IK)">
            ${state.enabled ? "ON (interactive)" : "OFF"}
          </button>
        </div>
        <div class="pso-rig-mode">
          ${VALID_MODES.map((m) => `
            <button data-mode="${m}" class="${state.mode === m ? "on" : ""}">${_modeLabel(m)}</button>
          `).join("")}
        </div>
        <div data-region="bone-tree-wrap">
          <div style="margin-bottom:3px;color:#99a4b3">Bones <span style="color:#6c7785" data-region="bone-count"></span>:</div>
          <div class="pso-rig-tree" data-region="bone-tree"></div>
        </div>
        <div class="pso-rig-inspector" data-region="inspector">
          <div class="dim">no bone selected</div>
        </div>
        <div data-region="weight-controls" hidden>
          <label>brush radius:
            <input type="range" min="0.05" max="3.0" step="0.01" value="${state.weightBrush.radius}" data-knob="wRadius">
            <span class="num" data-readout="wRadius">${state.weightBrush.radius.toFixed(2)}</span>
          </label>
          <label>strength:
            <input type="range" min="0" max="1.0" step="0.01" value="${state.weightBrush.strength}" data-knob="wStrength">
            <span class="num" data-readout="wStrength">${state.weightBrush.strength.toFixed(2)}</span>
          </label>
          <label>falloff:
            <select data-knob="wFalloff">
              ${FALLOFFS.map((f) => `<option value="${f}" ${state.weightBrush.falloff === f ? "selected" : ""}>${f}</option>`).join("")}
            </select>
          </label>
          <label>
            <input type="checkbox" ${state.showHeatmap ? "checked" : ""} data-knob="showHeatmap">
            heatmap (active bone)
          </label>
          <div class="pso-rig-legend" data-region="legend">
            <span>0</span><div class="bar"></div><span>1</span>
          </div>
          <div class="pso-rig-actions">
            <button data-act="auto-skin-distance" title="invert-distance auto-skin (fast)">auto-skin (dist)</button>
            <button data-act="auto-skin-heat" title="heat-equation-style smoothed auto-skin (slower)">auto-skin (heat)</button>
            <button data-act="normalize">Normalize</button>
          </div>
        </div>
        <div data-region="ik-controls" hidden>
          <div class="pso-rig-iklist" data-region="iklist"></div>
          <div class="pso-rig-actions">
            <button data-act="ik-add" title="add IK target on the selected bone">add IK target</button>
            <button data-act="ik-remove" class="danger">remove selected IK</button>
          </div>
          <label>chain length:
            <input type="range" min="1" max="6" step="1" value="2" data-knob="ikChainLen">
            <span class="num" data-readout="ikChainLen">2</span>
          </label>
        </div>
        <div class="pso-rig-stats" data-region="stats"></div>
        <div class="pso-rig-actions">
          <button data-act="save" class="primary">Save rig</button>
          <button data-act="reset" class="danger">Reset to source</button>
        </div>
        <div class="pso-rig-status idle" data-region="status">ready</div>
      </div>
    `;
    state.statusEl = body.querySelector('[data-region="status"]');
    state.treeEl = body.querySelector('[data-region="bone-tree"]');
    state.inspectorEl = body.querySelector('[data-region="inspector"]');
    state.legendEl = body.querySelector('[data-region="legend"]');
    body.addEventListener("click", onPanelClick);
    // Bone tree gets its own delegated click listener — onPanelClick
    // only fires when the click hits a [data-act] / [data-mode]
    // element, so tree rows would be ignored otherwise.
    if (state.treeEl) {
      state.treeEl.addEventListener("click", onTreeClick);
    }
    body.querySelectorAll("[data-knob]").forEach((el) => {
      el.addEventListener("input", onKnobChange);
      el.addEventListener("change", onKnobChange);
    });

    refreshSkeleton();
    refreshTreeHtml();
    refreshInspector();
    refreshModeSubregion();
    refreshIkList();
    updateStats();
  }

  function _modeLabel(m) {
    return ({
      [MODE_SKELETON]: "Skeleton",
      [MODE_WEIGHT]: "Weight Paint",
      [MODE_IK]: "IK",
    })[m] || m;
  }

  function refreshModeSubregion() {
    if (!state.bodyEl) return;
    const wc = state.bodyEl.querySelector('[data-region="weight-controls"]');
    const ic = state.bodyEl.querySelector('[data-region="ik-controls"]');
    if (wc) wc.hidden = state.mode !== MODE_WEIGHT;
    if (ic) ic.hidden = state.mode !== MODE_IK;
    if (state.mode === MODE_WEIGHT && state.showHeatmap) updateHeatmapColors();
    else if (state.mode !== MODE_WEIGHT) clearHeatmapColors();
  }

  function refreshIkList() {
    if (!state.bodyEl) return;
    const wrap = state.bodyEl.querySelector('[data-region="iklist"]');
    if (!wrap) return;
    if (state.ikTargets.length === 0) {
      wrap.innerHTML = `<div class="dim" style="padding:4px">no IK targets</div>`;
      return;
    }
    wrap.innerHTML = state.ikTargets.map((ik, i) => `
      <div class="pso-rig-ik-row${state.activeIkIdx === i ? " active" : ""}" data-ik-idx="${i}">
        <span>#${i}</span>
        <span class="grow">${escapeHtml(ik.name || `bone${ik.boneIdx}`)}</span>
        <span style="color:#6c7785">chain ${ik.chainLen}</span>
      </div>
    `).join("");
    wrap.addEventListener("click", (ev) => {
      const row = ev.target.closest("[data-ik-idx]");
      if (!row) return;
      state.activeIkIdx = parseInt(row.getAttribute("data-ik-idx"), 10);
      refreshIkList();
    }, { once: true });
  }

  function refreshSkeleton() {
    state.skeleton = window.psoGetSkeleton ? window.psoGetSkeleton() : null;
    const wrap = state.bodyEl ? state.bodyEl.querySelector('[data-region="bone-count"]') : null;
    if (wrap) {
      wrap.textContent = state.skeleton ? `(${state.skeleton.length})` : "(0)";
    }
  }

  function onPanelClick(ev) {
    const t = ev.target.closest("[data-act],[data-mode]");
    if (!t) return;
    const mode = t.getAttribute("data-mode");
    if (mode) {
      state.mode = mode;
      state.bodyEl.querySelectorAll(".pso-rig-mode button").forEach((b) => {
        b.classList.toggle("on", b.dataset.mode === mode);
      });
      refreshModeSubregion();
      setStatus("idle", `mode: ${mode}`);
      return;
    }
    const act = t.getAttribute("data-act");
    if (act === "toggle") {
      state.enabled = !state.enabled;
      t.classList.toggle("on", state.enabled);
      t.textContent = state.enabled ? "ON (interactive)" : "OFF";
      setStatus("idle", state.enabled ? "rig mode on" : "rig mode off");
      if (typeof window.psoSetRigModeActive === "function") {
        window.psoSetRigModeActive(state.enabled);
      }
      if (state.enabled) {
        refreshSkeleton();
        refreshTreeHtml();
        rebuildBoneWidgets();
        if (state.mode === MODE_WEIGHT && state.showHeatmap) updateHeatmapColors();
      } else {
        if (state.boneWidgets && state.boneWidgets.parent) {
          state.boneWidgets.parent.remove(state.boneWidgets);
        }
        clearHeatmapColors();
      }
      refreshBoneWidgetVisibility();
      updateStats();
      return;
    }
    if (act === "save") {
      setStatus("running", "saving…");
      saveRig().then(() => updateStats());
      return;
    }
    if (act === "reset") {
      resetAll();
      updateStats();
      return;
    }
    if (act === "auto-skin-distance") {
      autoSkin("distance");
      return;
    }
    if (act === "auto-skin-heat") {
      autoSkin("heat");
      return;
    }
    if (act === "normalize") {
      const n = normalizeAllWeights();
      setStatus("done", `normalized ${n} submesh(es)`);
      return;
    }
    if (act === "ik-add") {
      addIkTarget();
      return;
    }
    if (act === "ik-remove") {
      if (state.activeIkIdx < 0) return;
      removeIkTarget(state.activeIkIdx);
      return;
    }
    // Tree row click handler — delegated since the tree may have been
    // rebuilt and its handler reattached.
    if (state.treeEl && state.treeEl.contains(t)) onTreeClick(ev);
  }

  function onKnobChange(ev) {
    const t = ev.target;
    const k = t.getAttribute("data-knob");
    if (!k) return;
    if (k === "wRadius") {
      state.weightBrush.radius = parseFloat(t.value);
      const ro = state.bodyEl.querySelector('[data-readout="wRadius"]');
      if (ro) ro.textContent = state.weightBrush.radius.toFixed(2);
    } else if (k === "wStrength") {
      state.weightBrush.strength = parseFloat(t.value);
      const ro = state.bodyEl.querySelector('[data-readout="wStrength"]');
      if (ro) ro.textContent = state.weightBrush.strength.toFixed(2);
    } else if (k === "wFalloff") {
      state.weightBrush.falloff = t.value;
    } else if (k === "showHeatmap") {
      state.showHeatmap = !!t.checked;
      if (state.showHeatmap) updateHeatmapColors();
      else clearHeatmapColors();
    } else if (k === "ikChainLen") {
      const v = parseInt(t.value, 10) || 2;
      const ro = state.bodyEl.querySelector('[data-readout="ikChainLen"]');
      if (ro) ro.textContent = String(v);
      if (state.activeIkIdx >= 0 && state.activeIkIdx < state.ikTargets.length) {
        state.ikTargets[state.activeIkIdx].chainLen = v;
      }
    }
  }

  function updateStats() {
    if (!state.bodyEl) return;
    const wrap = state.bodyEl.querySelector('[data-region="stats"]');
    if (!wrap) return;
    const n = state.skeleton ? state.skeleton.length : 0;
    let nWeights = 0;
    for (const [_, w] of state.submeshWeights) {
      nWeights += w.indices.length / MAX_INFLUENCES;
    }
    wrap.textContent =
      `Rig active: ${state.enabled ? "yes" : "no"}\n` +
      `Bones:     ${n}\n` +
      `Painted submeshes: ${state.submeshWeights.size}\n` +
      `Painted verts:     ${nWeights.toLocaleString()}\n` +
      `IK targets:        ${state.ikTargets.length}\n` +
      `Selected:          ${state.selectedBones.size}`;
  }

  // -------------------------------------------------------------------
  // Tab integration
  // -------------------------------------------------------------------
  function injectRigTab() {
    if (typeof window.psoTexturePanelAddTabButton !== "function") return false;
    if (typeof window.psoTexturePanelRegisterTab !== "function") return false;
    const ok = window.psoTexturePanelAddTabButton(
      "rig", "Rig",
      "skeleton edits, weight painting, IK targets",
    );
    window.psoTexturePanelRegisterTab("rig", (body) => renderRigBlock(body));
    return ok;
  }

  function waitForPanel(deadline) {
    if (injectRigTab()) return;
    if (Date.now() > deadline) {
      console.warn("[rig_panel] texture panel never appeared; rig disabled");
      return;
    }
    setTimeout(() => waitForPanel(deadline), 250);
  }

  // -------------------------------------------------------------------
  // Init
  // -------------------------------------------------------------------
  function init() {
    waitForPanel(Date.now() + 30_000);
    const cv = document.getElementById("modelCanvas");
    if (cv) {
      cv.addEventListener("pointerdown", onPointerDown);
      cv.addEventListener("pointermove", onPointerMove);
      cv.addEventListener("pointerup", onPointerUp);
      cv.addEventListener("pointercancel", onPointerUp);
    } else {
      const t = setInterval(() => {
        const c = document.getElementById("modelCanvas");
        if (!c) return;
        clearInterval(t);
        c.addEventListener("pointerdown", onPointerDown);
        c.addEventListener("pointermove", onPointerMove);
        c.addEventListener("pointerup", onPointerUp);
        c.addEventListener("pointercancel", onPointerUp);
      }, 250);
    }
    // Reset on model load.
    if (window.bus && typeof window.bus.on === "function") {
      window.bus.on("model.loaded", () => {
        resetAll();
        refreshSkeleton();
        refreshTreeHtml();
        if (state.enabled) rebuildBoneWidgets();
      });
    }
    waitForApplyMeshWrap(Date.now() + 30_000);
  }

  function waitForApplyMeshWrap(deadline) {
    let allWired = true;
    if (typeof window.psoApplyMeshPayload === "function") {
      const orig = window.psoApplyMeshPayload;
      if (!orig.__psoRigWrapped) {
        const wrapped = function () {
          resetAll();
          return orig.apply(this, arguments);
        };
        wrapped.__psoRigWrapped = true;
        window.psoApplyMeshPayload = wrapped;
      }
    } else allWired = false;
    if (typeof window.psoOpenModelByPath === "function") {
      const orig = window.psoOpenModelByPath;
      if (!orig.__psoRigWrapped) {
        const wrapped = async function () {
          resetAll();
          return await orig.apply(this, arguments);
        };
        wrapped.__psoRigWrapped = true;
        window.psoOpenModelByPath = wrapped;
      }
    } else allWired = false;
    if (allWired) return;
    if (Date.now() > deadline) return;
    setTimeout(() => waitForApplyMeshWrap(deadline), 250);
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;",
      '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // Devtools handles
  window.psoRigState = state;
  window.psoRigSave = saveRig;
  window.psoRigReset = resetAll;
  window.psoRigAutoSkin = autoSkin;
  window.psoRigSolveIk = solveIk;
})();
