// =====================================================================
// PSOBB Skeleton Panel — read-only bone hierarchy tree.
// 2026-04-26
//
// A discoverable bone tree that lives outside the rig panel's modal
// flow.  When the user clicks a bone in the tree, the panel:
//
//   1. Highlights that bone in the 3D viewport (yellow ring + label).
//   2. Surfaces the bone's local TRS in the side inspector.
//   3. Emits `skeleton.boneSelected` on window.bus so the rig panel
//      and edit panel can react.
//
// Read-only — this panel doesn't write any data.  All bone editing
// flows through the rig panel; this is the *view* surface that makes
// the skeleton easy to navigate.
//
// Mounts as a tab in the texture-panel tab strip via
// psoTexturePanelRegisterTab + psoTexturePanelAddTabButton.
//
// Idempotent on multiple loads.
// =====================================================================

(function () {
  "use strict";

  if (window.__psoSkeletonPanelLoaded) return;
  window.__psoSkeletonPanelLoaded = true;

  const STYLE_ID = "psoSkeletonPanelStyle";
  const TAB_NAME = "skeleton";
  const TAB_LABEL = "Skeleton";
  const TAB_TITLE = "bone hierarchy tree (read-only); click a bone to highlight in viewport";

  // ---- state -------------------------------------------------------
  const state = {
    bones: [],            // [{ index, parent, position, rotation_bams, scale }]
    selectedIdx: -1,
    expanded: new Set(),  // bone indices expanded in the tree
    bodyEl: null,
    treeEl: null,
    inspectorEl: null,
    highlightMarker: null, // THREE.Object3D placed at selected bone
    boneNames: new Map(),  // boneIdx -> string
  };

  // ---- styles ------------------------------------------------------
  function ensureStyleInjected() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .pso-skel-panel-body {
        display: flex;
        gap: 8px;
        height: 100%;
        min-height: 200px;
      }
      .pso-skel-panel-tree {
        flex: 1 1 60%;
        overflow: auto;
        background: #1a1f25;
        border: 1px solid #2a313a;
        border-radius: 4px;
        padding: 6px;
        font-family: "Segoe UI", system-ui, sans-serif;
        font-size: 12px;
        max-height: 360px;
      }
      .pso-skel-panel-inspector {
        flex: 0 0 240px;
        background: #1a1f25;
        border: 1px solid #2a313a;
        border-radius: 4px;
        padding: 8px;
        font-size: 12px;
        max-height: 360px;
        overflow: auto;
      }
      .pso-skel-row {
        cursor: pointer;
        padding: 2px 4px;
        border-radius: 3px;
        white-space: nowrap;
        display: flex;
        align-items: center;
        gap: 4px;
      }
      .pso-skel-row:hover { background: #232931; }
      .pso-skel-row.active { background: #2d4a6e; color: #fff; }
      .pso-skel-row .twist {
        display: inline-block;
        width: 12px;
        text-align: center;
        font-size: 10px;
        cursor: pointer;
        user-select: none;
        color: #8a96a6;
      }
      .pso-skel-row .twist.empty { visibility: hidden; }
      .pso-skel-row .badge {
        font-size: 10px;
        color: #8a96a6;
        background: #232931;
        padding: 0 3px;
        border-radius: 2px;
      }
      .pso-skel-inspector h4 {
        margin: 0 0 6px 0;
        font-size: 12px;
        color: #c8d3df;
      }
      .pso-skel-inspector dl {
        margin: 0 0 8px 0;
        display: grid;
        grid-template-columns: 70px 1fr;
        gap: 2px 6px;
      }
      .pso-skel-inspector dt {
        color: #8a96a6;
        font-size: 11px;
      }
      .pso-skel-inspector dd {
        margin: 0;
        font-family: ui-monospace, monospace;
        font-size: 11px;
      }
      .pso-skel-empty {
        padding: 16px;
        color: #8a96a6;
        text-align: center;
      }
      .pso-skel-toolbar {
        display: flex;
        gap: 6px;
        padding: 4px 0 6px 0;
        font-size: 11px;
      }
      .pso-skel-toolbar input[type="text"] {
        flex: 1;
        background: #0e1318;
        border: 1px solid #2a313a;
        color: #d8e3ef;
        padding: 2px 6px;
        border-radius: 3px;
      }
      .pso-skel-toolbar button {
        background: #232931;
        border: 1px solid #2a313a;
        color: #c8d3df;
        padding: 2px 8px;
        cursor: pointer;
        border-radius: 3px;
      }
      .pso-skel-toolbar button:hover { background: #2d3540; }
    `;
    document.head.appendChild(style);
  }

  // ---- bone naming heuristic ---------------------------------------
  // PSOBB skeletons don't ship bone names.  We synthesize:
  //   bone_0 = "root"
  //   else use a humanoid heuristic when the parent chain depth matches
  //   known patterns, else "bone_<idx>".
  function nameBone(idx, bones) {
    if (state.boneNames.has(idx)) return state.boneNames.get(idx);
    if (idx === 0) {
      state.boneNames.set(idx, "root");
      return "root";
    }
    const b = bones[idx];
    let depth = 0;
    let cur = b;
    while (cur && cur.parent >= 0) {
      depth++;
      cur = bones[cur.parent];
      if (depth > 32) break;
    }
    const nm = "bone_" + idx + " (d" + depth + ")";
    state.boneNames.set(idx, nm);
    return nm;
  }

  // ---- tree building ------------------------------------------------
  function getChildren(parentIdx) {
    const out = [];
    for (let i = 0; i < state.bones.length; i++) {
      if ((state.bones[i].parent | 0) === parentIdx) out.push(i);
    }
    return out;
  }

  function rootBoneIdxs() {
    const out = [];
    for (let i = 0; i < state.bones.length; i++) {
      if ((state.bones[i].parent | 0) < 0) out.push(i);
    }
    return out.length ? out : [0];
  }

  function renderTree(filter) {
    const tree = state.treeEl;
    if (!tree) return;
    tree.innerHTML = "";
    if (!state.bones.length) {
      tree.innerHTML = '<div class="pso-skel-empty">No skeleton loaded.<br>Open a skinned model (.nj with bones) first.</div>';
      return;
    }
    const filterLc = (filter || "").trim().toLowerCase();

    const ul = document.createElement("ul");
    ul.style.listStyle = "none";
    ul.style.padding = "0";
    ul.style.margin = "0";
    tree.appendChild(ul);

    function emitNode(parent, idx, depth) {
      const children = getChildren(idx);
      const expanded = state.expanded.has(idx);
      const li = document.createElement("li");
      const row = document.createElement("div");
      row.className = "pso-skel-row";
      if (idx === state.selectedIdx) row.classList.add("active");
      row.style.paddingLeft = (depth * 12) + "px";

      const twist = document.createElement("span");
      twist.className = "twist" + (children.length ? "" : " empty");
      twist.textContent = children.length ? (expanded ? "▾" : "▸") : "•";
      twist.addEventListener("click", function (e) {
        e.stopPropagation();
        if (!children.length) return;
        if (expanded) state.expanded.delete(idx);
        else state.expanded.add(idx);
        renderTree(filter);
      });
      row.appendChild(twist);

      const label = document.createElement("span");
      label.className = "label";
      label.textContent = nameBone(idx, state.bones);
      row.appendChild(label);

      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = "#" + idx;
      row.appendChild(badge);

      row.addEventListener("click", function () {
        selectBone(idx);
      });
      row.addEventListener("dblclick", function () {
        // Expand all descendants on dbl-click for fast navigation.
        function recurse(i) {
          state.expanded.add(i);
          for (const c of getChildren(i)) recurse(c);
        }
        recurse(idx);
        renderTree(filter);
      });

      // Filter: keep this row if itself or any descendant matches.
      let visible = true;
      if (filterLc) {
        const matchSelf = nameBone(idx, state.bones).toLowerCase().includes(filterLc)
                       || ("#" + idx).includes(filterLc);
        let matchDesc = false;
        function descMatches(i) {
          if (matchDesc) return;
          if (nameBone(i, state.bones).toLowerCase().includes(filterLc)) {
            matchDesc = true;
            return;
          }
          for (const c of getChildren(i)) descMatches(c);
        }
        for (const c of children) descMatches(c);
        visible = matchSelf || matchDesc;
      }

      if (visible) {
        li.appendChild(row);
        parent.appendChild(li);
        if ((expanded || (filterLc && children.length)) && children.length) {
          const childUl = document.createElement("ul");
          childUl.style.listStyle = "none";
          childUl.style.padding = "0";
          childUl.style.margin = "0";
          li.appendChild(childUl);
          for (const c of children) emitNode(childUl, c, depth + 1);
        }
      }
    }
    for (const r of rootBoneIdxs()) emitNode(ul, r, 0);
  }

  function renderInspector() {
    const insp = state.inspectorEl;
    if (!insp) return;
    if (state.selectedIdx < 0 || !state.bones[state.selectedIdx]) {
      insp.innerHTML = '<div class="pso-skel-empty">No bone selected.<br>Click a row to inspect.</div>';
      return;
    }
    const b = state.bones[state.selectedIdx];
    const r = b.rotation_bams || [0, 0, 0];
    const BAMS_TO_DEG = 360.0 / 65536.0;
    const rotDeg = r.map(function (x) { return (x * BAMS_TO_DEG).toFixed(2); });

    const children = getChildren(state.selectedIdx);
    insp.innerHTML =
      '<h4>Bone #' + state.selectedIdx + '</h4>' +
      '<dl>' +
      '<dt>name</dt><dd>' + escapeHtml(nameBone(state.selectedIdx, state.bones)) + '</dd>' +
      '<dt>parent</dt><dd>' + (b.parent < 0 ? "(root)" : ("#" + b.parent)) + '</dd>' +
      '<dt>children</dt><dd>' + children.length + '</dd>' +
      '<dt>position</dt><dd>' + b.position.map(fmt).join(", ") + '</dd>' +
      '<dt>rot (BAMS)</dt><dd>' + r.join(", ") + '</dd>' +
      '<dt>rot (deg)</dt><dd>' + rotDeg.join(", ") + '</dd>' +
      '<dt>scale</dt><dd>' + (b.scale || [1, 1, 1]).map(fmt).join(", ") + '</dd>' +
      '<dt>eval flags</dt><dd>0x' + ((b.eval_flags | 0)).toString(16) + '</dd>' +
      '</dl>' +
      '<div class="dim" style="font-size:11px">' +
      'open the <strong>Rig</strong> tab to edit this bone\'s TRS or paint weights to it.' +
      '</div>';
  }

  function fmt(x) {
    if (typeof x !== "number") return String(x);
    if (Math.abs(x) < 1e-6) return "0";
    return x.toFixed(4);
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;",
                '"': "&quot;", "'": "&#39;" })[c];
    });
  }

  // ---- bone selection + viewport highlight -------------------------
  function selectBone(idx) {
    state.selectedIdx = idx;
    renderTree(state.lastFilter || "");
    renderInspector();
    if (window.bus && window.bus.emit) {
      try { window.bus.emit("skeleton.boneSelected", { boneIdx: idx }); } catch (_e) {}
    }
    placeHighlight(idx);
  }

  function placeHighlight(idx) {
    const THREE = window.THREE;
    if (!THREE) return;
    const scene = window.psoGetMeshGroup && window.psoGetMeshGroup();
    const root = scene ? scene.parent : null;
    if (!root) return;
    if (!state.highlightMarker) {
      const geo = new THREE.SphereGeometry(0.04, 12, 8);
      const mat = new THREE.MeshBasicMaterial({
        color: 0xffd400,
        wireframe: true,
        transparent: true,
        opacity: 0.95,
        depthTest: false,
      });
      const mk = new THREE.Mesh(geo, mat);
      mk.renderOrder = 9999;
      mk.name = "psoSkeletonPanelHighlight";
      root.add(mk);
      state.highlightMarker = mk;
    }
    if (!window.psoGetBoneWorldPositions) return;
    const positions = window.psoGetBoneWorldPositions();
    if (!positions || idx < 0 || idx >= positions.length) {
      state.highlightMarker.visible = false;
      return;
    }
    const p = positions[idx];
    if (!p) {
      state.highlightMarker.visible = false;
      return;
    }
    state.highlightMarker.position.set(p[0], p[1], p[2]);
    state.highlightMarker.visible = true;
    if (window.psoForceRender) window.psoForceRender();
  }

  function clearHighlight() {
    if (state.highlightMarker) state.highlightMarker.visible = false;
  }

  // ---- public API ---------------------------------------------------
  window.psoSkeletonPanel = Object.freeze({
    selectBone: selectBone,
    clearHighlight: clearHighlight,
    refreshFromViewport: refreshFromViewport,
    getSelected: function () { return state.selectedIdx; },
    getBoneCount: function () { return state.bones.length; },
  });

  function refreshFromViewport() {
    const skel = window.psoGetSkeleton && window.psoGetSkeleton();
    if (!skel || !skel.length) {
      state.bones = [];
      state.boneNames.clear();
      state.selectedIdx = -1;
      state.expanded.clear();
      renderTree();
      renderInspector();
      return false;
    }
    state.bones = skel;
    // Auto-expand depth 0 + 1 for discovery.
    state.expanded.clear();
    for (let i = 0; i < skel.length; i++) {
      let depth = 0;
      let cur = skel[i];
      while (cur && cur.parent >= 0) { depth++; cur = skel[cur.parent]; if (depth > 32) break; }
      if (depth <= 1) state.expanded.add(i);
    }
    renderTree();
    renderInspector();
    return true;
  }

  // ---- tab registration --------------------------------------------
  function renderTabBody(bodyEl) {
    ensureStyleInjected();
    state.bodyEl = bodyEl;
    bodyEl.innerHTML =
      '<div class="pso-skel-toolbar">' +
        '<input type="text" id="psoSkelFilter" placeholder="filter bone name or #idx…" autocomplete="off" />' +
        '<button id="psoSkelExpandAll" type="button">expand all</button>' +
        '<button id="psoSkelCollapseAll" type="button">collapse all</button>' +
        '<button id="psoSkelRefresh" type="button" title="re-read skeleton from current viewport">refresh</button>' +
      '</div>' +
      '<div class="pso-skel-panel-body">' +
        '<div class="pso-skel-panel-tree" id="psoSkelTree"></div>' +
        '<div class="pso-skel-panel-inspector pso-skel-inspector" id="psoSkelInspector"></div>' +
      '</div>';
    state.treeEl = bodyEl.querySelector("#psoSkelTree");
    state.inspectorEl = bodyEl.querySelector("#psoSkelInspector");
    const filt = bodyEl.querySelector("#psoSkelFilter");
    if (filt) {
      filt.addEventListener("input", function () {
        state.lastFilter = filt.value;
        renderTree(filt.value);
      });
    }
    const ea = bodyEl.querySelector("#psoSkelExpandAll");
    if (ea) ea.addEventListener("click", function () {
      for (let i = 0; i < state.bones.length; i++) state.expanded.add(i);
      renderTree(state.lastFilter || "");
    });
    const ca = bodyEl.querySelector("#psoSkelCollapseAll");
    if (ca) ca.addEventListener("click", function () {
      state.expanded.clear();
      renderTree(state.lastFilter || "");
    });
    const rf = bodyEl.querySelector("#psoSkelRefresh");
    if (rf) rf.addEventListener("click", function () {
      refreshFromViewport();
    });
    refreshFromViewport();
  }

  function tryRegisterTab() {
    if (typeof window.psoTexturePanelRegisterTab === "function" &&
        typeof window.psoTexturePanelAddTabButton === "function") {
      window.psoTexturePanelRegisterTab(TAB_NAME, renderTabBody);
      window.psoTexturePanelAddTabButton(TAB_NAME, TAB_LABEL, TAB_TITLE);
      return true;
    }
    return false;
  }

  // Refresh on model load.
  if (window.bus && window.bus.on) {
    window.bus.on("model.loaded", function () { setTimeout(refreshFromViewport, 100); });
    window.bus.on("model.skinned.loaded", function () { setTimeout(refreshFromViewport, 100); });
  }

  // Register on next tick — texture_panel.js may not be loaded yet.
  function init() {
    if (!tryRegisterTab()) {
      let attempts = 0;
      const t = setInterval(function () {
        if (tryRegisterTab() || ++attempts > 40) {
          clearInterval(t);
        }
      }, 250);
    }
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
