// =====================================================================
// PSOBB UV Panel — read-only UV-island viewer.
// 2026-04-26
//
// 2D canvas that draws the UV layout for the active submesh, optionally
// over the bound texture.  The user can:
//
//   - Pick a submesh from a dropdown (auto-populated from the loaded
//     mesh group)
//   - Toggle texture overlay on / off
//   - Toggle a checker-pattern preview (helps spot stretching)
//   - Drag-pan / wheel-zoom inside the canvas
//   - Click a vertex to highlight it (also highlights the matching
//     vertex in the 3D viewport via window.bus 'uv.vertexSelected')
//
// Read-only by design — UV editing is a P1 follow-up.  This panel
// surfaces the missing UV view the user complained about.
//
// Mounts as a tab in the texture-panel tab strip.  Uses the same
// public model-viewer API the sculpt / rig / paint panels use.
//
// Idempotent on multiple loads.
// =====================================================================

(function () {
  "use strict";

  if (window.__psoUvPanelLoaded) return;
  window.__psoUvPanelLoaded = true;

  const STYLE_ID = "psoUvPanelStyle";
  const TAB_NAME = "uv";
  const TAB_LABEL = "UV";
  const TAB_TITLE = "2D UV-island view of the selected submesh; toggle texture overlay or checker preview";

  const state = {
    canvas: null,
    ctx: null,
    submeshIdx: 0,
    submeshes: [],     // [{ idx, materialId, uv, indices, vertCount }]
    showTexture: true,
    showChecker: false,
    showWire: true,
    selectedVert: -1,
    pan: { x: 0, y: 0 },
    zoom: 1.0,
    drag: { active: false, x0: 0, y0: 0, px: 0, py: 0 },
    overlayImg: null,  // HTMLImageElement of the bound texture, when available
    bodyEl: null,
    statusEl: null,
    selectorEl: null,
  };

  function ensureStyleInjected() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .pso-uv-toolbar {
        display: flex;
        gap: 6px;
        padding: 4px 0 6px 0;
        font-size: 11px;
        align-items: center;
        flex-wrap: wrap;
      }
      .pso-uv-toolbar select {
        background: #0e1318;
        border: 1px solid #2a313a;
        color: #d8e3ef;
        padding: 2px 6px;
        border-radius: 3px;
      }
      .pso-uv-toolbar label {
        display: inline-flex;
        align-items: center;
        gap: 3px;
        color: #c8d3df;
      }
      .pso-uv-canvas-wrap {
        position: relative;
        background: #0a0e13;
        border: 1px solid #2a313a;
        border-radius: 4px;
        overflow: hidden;
      }
      #psoUvCanvas {
        display: block;
        width: 100%;
        height: 360px;
        cursor: grab;
        background:
          repeating-conic-gradient(#0d1116 0% 25%, #15191e 25% 50%) 50% / 16px 16px;
      }
      #psoUvCanvas.dragging { cursor: grabbing; }
      .pso-uv-status {
        font-size: 11px;
        color: #8a96a6;
        padding: 4px 0;
        font-family: ui-monospace, monospace;
      }
      .pso-uv-empty {
        padding: 40px;
        color: #8a96a6;
        text-align: center;
      }
      .pso-uv-toolbar button {
        background: #232931;
        border: 1px solid #2a313a;
        color: #c8d3df;
        padding: 2px 8px;
        cursor: pointer;
        border-radius: 3px;
      }
      .pso-uv-toolbar button:hover { background: #2d3540; }
    `;
    document.head.appendChild(style);
  }

  // ---- collect submeshes from the viewport -------------------------
  function collectSubmeshes() {
    const out = [];
    const dbg = window.psoGetDebugMeshes && window.psoGetDebugMeshes();
    if (!dbg || !dbg.length) return out;
    for (let i = 0; i < dbg.length; i++) {
      const e = dbg[i];
      const mesh = e && e.mesh;
      if (!mesh || !mesh.geometry) continue;
      const uvAttr = mesh.geometry.getAttribute("uv");
      const idxAttr = mesh.geometry.getIndex();
      if (!uvAttr) continue;
      out.push({
        idx: i,
        materialId: (mesh.userData && mesh.userData.materialId) | 0,
        uv: uvAttr.array,
        indices: idxAttr ? idxAttr.array : null,
        vertCount: uvAttr.count,
        triCount: idxAttr ? Math.floor(idxAttr.count / 3) : Math.floor(uvAttr.count / 3),
      });
    }
    return out;
  }

  // ---- canvas drawing -----------------------------------------------
  // UV space:    u in [0,1] -> x = u * size
  //              v in [0,1] -> y = (1 - v) * size  (Y-flipped to match game)
  function draw() {
    const cv = state.canvas;
    if (!cv || !state.ctx) return;
    const ctx = state.ctx;
    const w = cv.width;
    const h = cv.height;
    ctx.save();
    ctx.clearRect(0, 0, w, h);

    // Background — already a CSS-gradient checker behind the canvas;
    // the canvas itself stays transparent so the CSS shows through.
    // For deterministic export we still fill a translucent dark rect.
    ctx.fillStyle = "rgba(10,14,19,0.55)";
    ctx.fillRect(0, 0, w, h);

    const sub = state.submeshes[state.submeshIdx];
    if (!sub) {
      ctx.fillStyle = "#8a96a6";
      ctx.font = "12px ui-monospace, monospace";
      ctx.textAlign = "center";
      ctx.fillText("no submesh", w / 2, h / 2);
      ctx.restore();
      return;
    }

    // World transform: zoom + pan.
    const size = Math.min(w, h) * 0.9 * state.zoom;
    const ox = (w - size) / 2 + state.pan.x;
    const oy = (h - size) / 2 + state.pan.y;

    // Texture overlay — draw at native UV [0,1] mapping.
    if (state.showTexture && state.overlayImg && state.overlayImg.complete) {
      ctx.globalAlpha = 0.85;
      ctx.drawImage(state.overlayImg, ox, oy, size, size);
      ctx.globalAlpha = 1.0;
    } else if (state.showChecker) {
      // Procedural checker pattern.
      const cells = 16;
      const cw = size / cells;
      for (let cy = 0; cy < cells; cy++) {
        for (let cx = 0; cx < cells; cx++) {
          ctx.fillStyle = ((cx + cy) & 1) ? "#1c2128" : "#2a313a";
          ctx.fillRect(ox + cx * cw, oy + cy * cw, cw, cw);
        }
      }
    }

    // UV [0,1] frame.
    ctx.strokeStyle = "#3a4252";
    ctx.lineWidth = 1;
    ctx.strokeRect(ox, oy, size, size);
    ctx.fillStyle = "#576070";
    ctx.font = "10px ui-monospace, monospace";
    ctx.textAlign = "left";
    ctx.fillText("(0,0)", ox + 2, oy + size - 2);
    ctx.textAlign = "right";
    ctx.fillText("(1,1)", ox + size - 2, oy + 11);

    // UV wireframe — draw triangles.
    if (state.showWire) {
      ctx.strokeStyle = "rgba(120, 200, 255, 0.65)";
      ctx.lineWidth = 0.7;
      const uv = sub.uv;
      const idx = sub.indices;
      ctx.beginPath();
      if (idx) {
        for (let f = 0; f < idx.length; f += 3) {
          const a = idx[f], b = idx[f + 1], c = idx[f + 2];
          const ax = ox + uv[a * 2] * size;
          const ay = oy + (1 - uv[a * 2 + 1]) * size;
          const bx = ox + uv[b * 2] * size;
          const by = oy + (1 - uv[b * 2 + 1]) * size;
          const cx = ox + uv[c * 2] * size;
          const cy = oy + (1 - uv[c * 2 + 1]) * size;
          ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
          ctx.moveTo(bx, by); ctx.lineTo(cx, cy);
          ctx.moveTo(cx, cy); ctx.lineTo(ax, ay);
        }
      } else {
        for (let v = 0; v + 2 < sub.vertCount; v += 3) {
          const ax = ox + uv[v * 2] * size;
          const ay = oy + (1 - uv[v * 2 + 1]) * size;
          const bx = ox + uv[(v + 1) * 2] * size;
          const by = oy + (1 - uv[(v + 1) * 2 + 1]) * size;
          const cx = ox + uv[(v + 2) * 2] * size;
          const cy = oy + (1 - uv[(v + 2) * 2 + 1]) * size;
          ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
          ctx.moveTo(bx, by); ctx.lineTo(cx, cy);
          ctx.moveTo(cx, cy); ctx.lineTo(ax, ay);
        }
      }
      ctx.stroke();
    }

    // Vertices (small dots).
    ctx.fillStyle = "rgba(255,180,100,0.75)";
    for (let v = 0; v < sub.vertCount; v++) {
      const x = ox + sub.uv[v * 2] * size;
      const y = oy + (1 - sub.uv[v * 2 + 1]) * size;
      ctx.fillRect(x - 1, y - 1, 2, 2);
    }

    // Selected vertex highlight.
    if (state.selectedVert >= 0 && state.selectedVert < sub.vertCount) {
      const x = ox + sub.uv[state.selectedVert * 2] * size;
      const y = oy + (1 - sub.uv[state.selectedVert * 2 + 1]) * size;
      ctx.strokeStyle = "#ffd400";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.stroke();
    }

    ctx.restore();
  }

  // ---- pointer interaction ------------------------------------------
  function pickVertex(clientX, clientY) {
    const cv = state.canvas;
    if (!cv) return -1;
    const rect = cv.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const x = (clientX - rect.left) * dpr;
    const y = (clientY - rect.top) * dpr;
    const w = cv.width;
    const h = cv.height;
    const sub = state.submeshes[state.submeshIdx];
    if (!sub) return -1;
    const size = Math.min(w, h) * 0.9 * state.zoom;
    const ox = (w - size) / 2 + state.pan.x;
    const oy = (h - size) / 2 + state.pan.y;
    let best = -1, bestD2 = (8 * dpr) * (8 * dpr);
    for (let v = 0; v < sub.vertCount; v++) {
      const vx = ox + sub.uv[v * 2] * size;
      const vy = oy + (1 - sub.uv[v * 2 + 1]) * size;
      const d2 = (vx - x) * (vx - x) + (vy - y) * (vy - y);
      if (d2 < bestD2) { bestD2 = d2; best = v; }
    }
    return best;
  }

  function setSelectedVertex(v) {
    state.selectedVert = v;
    draw();
    if (window.bus && window.bus.emit) {
      try {
        window.bus.emit("uv.vertexSelected", {
          submeshIdx: state.submeshIdx,
          vertexIdx: v,
        });
      } catch (_e) {}
    }
    updateStatus();
  }

  function attachPointer() {
    const cv = state.canvas;
    if (!cv) return;
    cv.addEventListener("pointerdown", function (ev) {
      cv.setPointerCapture(ev.pointerId);
      if (ev.button === 0 && !ev.shiftKey) {
        // Click — pick a vertex.
        const v = pickVertex(ev.clientX, ev.clientY);
        if (v >= 0) {
          setSelectedVertex(v);
          return;
        }
      }
      // Otherwise start a pan-drag.
      state.drag.active = true;
      state.drag.x0 = ev.clientX;
      state.drag.y0 = ev.clientY;
      state.drag.px = state.pan.x;
      state.drag.py = state.pan.y;
      cv.classList.add("dragging");
    });
    cv.addEventListener("pointermove", function (ev) {
      if (!state.drag.active) return;
      const dpr = window.devicePixelRatio || 1;
      state.pan.x = state.drag.px + (ev.clientX - state.drag.x0) * dpr;
      state.pan.y = state.drag.py + (ev.clientY - state.drag.y0) * dpr;
      draw();
    });
    function release(ev) {
      state.drag.active = false;
      cv.classList.remove("dragging");
      try { cv.releasePointerCapture(ev.pointerId); } catch (_e) {}
    }
    cv.addEventListener("pointerup", release);
    cv.addEventListener("pointercancel", release);
    cv.addEventListener("wheel", function (ev) {
      ev.preventDefault();
      const factor = ev.deltaY > 0 ? 0.9 : 1.1;
      state.zoom = Math.max(0.2, Math.min(20, state.zoom * factor));
      draw();
      updateStatus();
    }, { passive: false });
  }

  function fitView() {
    state.pan.x = 0; state.pan.y = 0; state.zoom = 1.0;
    draw();
    updateStatus();
  }

  function updateStatus() {
    if (!state.statusEl) return;
    const sub = state.submeshes[state.submeshIdx];
    if (!sub) {
      state.statusEl.textContent = "no submesh";
      return;
    }
    const sel = state.selectedVert >= 0
      ? ` | sel #${state.selectedVert} u=${sub.uv[state.selectedVert*2].toFixed(4)} v=${sub.uv[state.selectedVert*2+1].toFixed(4)}`
      : "";
    state.statusEl.textContent =
      `submesh ${sub.idx} (mat ${sub.materialId})  ` +
      `verts ${sub.vertCount}  tris ${sub.triCount}  zoom ${(state.zoom * 100).toFixed(0)}%${sel}`;
  }

  // ---- texture overlay ---------------------------------------------
  async function loadTextureOverlay() {
    state.overlayImg = null;
    const sub = state.submeshes[state.submeshIdx];
    if (!sub) { draw(); return; }
    const tex = window.psoGetMaterialTexture && window.psoGetMaterialTexture(sub.materialId);
    if (!tex) { draw(); return; }
    // tex is THREE.Texture — its .image is HTMLImageElement | HTMLCanvasElement.
    // Use it as an Image source if possible.
    const img = tex.image;
    if (img && (img.complete || img.tagName === "CANVAS")) {
      state.overlayImg = img;
      draw();
      return;
    }
    // Fallback: try /api/model_textures or skip.
    draw();
  }

  // ---- DOM helpers --------------------------------------------------
  function updateSubmeshSelector() {
    if (!state.selectorEl) return;
    const subs = state.submeshes;
    state.selectorEl.innerHTML = subs.map(function (s) {
      return '<option value="' + s.idx + '">submesh ' + s.idx +
             ' (mat ' + s.materialId + ', ' + s.vertCount + ' v)</option>';
    }).join("");
    if (state.submeshIdx >= subs.length) state.submeshIdx = 0;
    state.selectorEl.value = String(state.submeshIdx);
  }

  function refreshFromViewport() {
    state.submeshes = collectSubmeshes();
    state.selectedVert = -1;
    if (state.submeshIdx >= state.submeshes.length) state.submeshIdx = 0;
    updateSubmeshSelector();
    if (state.canvas) {
      // Resize for retina.
      const cv = state.canvas;
      const dpr = window.devicePixelRatio || 1;
      const rect = cv.getBoundingClientRect();
      cv.width = Math.max(1, Math.floor(rect.width * dpr));
      cv.height = Math.max(1, Math.floor(rect.height * dpr));
    }
    loadTextureOverlay();
    draw();
    updateStatus();
  }

  // ---- public API ---------------------------------------------------
  window.psoUvPanel = Object.freeze({
    refreshFromViewport: refreshFromViewport,
    selectSubmesh: function (i) {
      state.submeshIdx = i | 0;
      state.selectedVert = -1;
      loadTextureOverlay();
      draw();
      updateStatus();
    },
    selectVertex: function (v) { setSelectedVertex(v | 0); },
    getSelected: function () {
      return { submeshIdx: state.submeshIdx, vertexIdx: state.selectedVert };
    },
    fitView: fitView,
  });

  // ---- tab registration --------------------------------------------
  function renderTabBody(bodyEl) {
    ensureStyleInjected();
    state.bodyEl = bodyEl;
    bodyEl.innerHTML =
      '<div class="pso-uv-toolbar">' +
        '<label>submesh: <select id="psoUvSubSel"></select></label>' +
        '<label><input type="checkbox" id="psoUvShowTex" checked /> texture</label>' +
        '<label><input type="checkbox" id="psoUvShowChecker" /> checker</label>' +
        '<label><input type="checkbox" id="psoUvShowWire" checked /> wireframe</label>' +
        '<button id="psoUvFit" type="button" title="reset zoom + pan">fit</button>' +
        '<button id="psoUvRefresh" type="button" title="re-read UVs from viewport">refresh</button>' +
        '<span class="grow" style="flex:1"></span>' +
      '</div>' +
      '<div class="pso-uv-canvas-wrap">' +
        '<canvas id="psoUvCanvas"></canvas>' +
      '</div>' +
      '<div class="pso-uv-status" id="psoUvStatus"></div>';
    state.canvas = bodyEl.querySelector("#psoUvCanvas");
    state.ctx = state.canvas.getContext("2d");
    state.statusEl = bodyEl.querySelector("#psoUvStatus");
    state.selectorEl = bodyEl.querySelector("#psoUvSubSel");
    state.selectorEl.addEventListener("change", function () {
      state.submeshIdx = parseInt(state.selectorEl.value, 10) || 0;
      state.selectedVert = -1;
      loadTextureOverlay();
      draw();
      updateStatus();
    });
    bodyEl.querySelector("#psoUvShowTex").addEventListener("change", function (ev) {
      state.showTexture = !!ev.target.checked;
      draw();
    });
    bodyEl.querySelector("#psoUvShowChecker").addEventListener("change", function (ev) {
      state.showChecker = !!ev.target.checked;
      draw();
    });
    bodyEl.querySelector("#psoUvShowWire").addEventListener("change", function (ev) {
      state.showWire = !!ev.target.checked;
      draw();
    });
    bodyEl.querySelector("#psoUvFit").addEventListener("click", fitView);
    bodyEl.querySelector("#psoUvRefresh").addEventListener("click", refreshFromViewport);
    attachPointer();
    setTimeout(refreshFromViewport, 50);
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
