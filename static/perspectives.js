// =====================================================================
// PSOBB Texture Editor - Perspective registry (2026-04-24)
//
// One persistent unified viewport for the editor's main work area.
// Each "perspective" is a mode the central stage can render in. Tabs
// across the top swap perspectives without modal popups.
//
// Lifecycle of a perspective:
//   match(entry, file) -> number   higher score wins on auto-route
//   mount(stageEl, inspectorEl, ctx)
//   unmount(stageEl, inspectorEl)
//
// `ctx` carries the active asset payload:
//   { path, entry, fileName }
//
// The registry is intentionally minimal — perspectives can do anything
// inside their stage element. They share helpers via window.PSOPerspectives:
//   active()              currently active perspective name
//   list(entry)           perspectives that match the entry, with scores
//   switchTo(name, ctx)   programmatic activation
//   refresh(ctx)          re-evaluate auto-route for a context
//
// localStorage key `psoVPLastPerspective:<path>` holds the last-used
// perspective per asset for resume-on-reopen.
// =====================================================================

(function () {
  "use strict";

  if (window.PSOPerspectives) return;

  // ---- registry --------------------------------------------------------
  const registry = new Map(); // name -> spec
  const order = [];           // insertion order for stable tab strip

  function register(name, spec) {
    if (!name || typeof name !== "string") return;
    if (registry.has(name)) {
      console.warn("[perspectives] re-registering", name);
      return;
    }
    if (typeof spec.match !== "function" ||
        typeof spec.mount !== "function" ||
        typeof spec.unmount !== "function") {
      console.warn("[perspectives] missing match/mount/unmount on", name);
      return;
    }
    registry.set(name, Object.assign({ label: name }, spec));
    order.push(name);
  }

  // ---- state ---------------------------------------------------------
  let activeName = null;
  let activeCtx = null;
  let lastTransitionId = 0;

  function $(s) { return document.querySelector(s); }

  function stageEl() { return $("#vpStage"); }
  function inspectorEl() { return $("#vpInspector"); }

  // ---- last-perspective persistence ---------------------------------
  function rememberLastPerspective(path, name) {
    if (!path) return;
    try { localStorage.setItem("psoVPLastPerspective:" + path, name); }
    catch (_e) {}
  }
  function recallLastPerspective(path) {
    if (!path) return null;
    try { return localStorage.getItem("psoVPLastPerspective:" + path); }
    catch (_e) { return null; }
  }

  // ---- list / switch -------------------------------------------------
  function list(entry, fileName) {
    const out = [];
    for (const name of order) {
      const spec = registry.get(name);
      if (!spec) continue;
      let score = 0;
      try { score = spec.match(entry, fileName) || 0; } catch (_e) { score = 0; }
      out.push({ name, label: spec.label || name, score, spec });
    }
    return out;
  }

  function bestPerspective(entry, fileName, preferred) {
    const cands = list(entry, fileName).filter((c) => c.score > 0);
    if (!cands.length) return null;
    if (preferred) {
      const pref = cands.find((c) => c.name === preferred);
      if (pref) return pref.name;
    }
    cands.sort((a, b) => b.score - a.score);
    return cands[0].name;
  }

  function active() { return activeName; }
  function activeContext() { return activeCtx; }

  async function switchTo(name, ctx) {
    if (!registry.has(name)) {
      console.warn("[perspectives] unknown perspective", name);
      return;
    }
    const stage = stageEl();
    const insp = inspectorEl();
    if (!stage || !insp) {
      console.warn("[perspectives] viewport DOM not present yet");
      return;
    }

    const myToken = ++lastTransitionId;
    ctx = ctx || activeCtx;

    // Unmount previous (if any). Wrap in try so a buggy unmount doesn't
    // wedge subsequent transitions.
    if (activeName) {
      const prev = registry.get(activeName);
      if (prev) {
        try { await prev.unmount(stage, insp); }
        catch (e) { console.error("[perspectives] unmount failed:", e); }
      }
    }

    // Race guard: another switchTo may have started while we awaited unmount.
    if (myToken !== lastTransitionId) return;

    activeName = name;
    activeCtx = ctx;

    // Reset stage + inspector content. Each mount() rebuilds.
    stage.innerHTML = "";
    insp.innerHTML = "";
    stage.classList.add("vp-stage-fade-in");

    try {
      await registry.get(name).mount(stage, insp, ctx || {});
    } catch (e) {
      console.error("[perspectives] mount failed:", e);
      stage.innerHTML = '<div class="vp-stage-empty">' +
        '<div>perspective failed to mount: ' + escapeHtml(name) + '</div>' +
        '<div class="dim">' + escapeHtml(String(e && e.message || e)) + '</div>' +
        '</div>';
    }

    // Smooth fade — short, no animation lib
    setTimeout(function () {
      try { stage.classList.remove("vp-stage-fade-in"); } catch (_e) {}
    }, 220);

    rebuildTabStrip();
    if (ctx && ctx.path) rememberLastPerspective(ctx.path, name);
    document.body.dataset.psoActivePerspective = name;
    if (window.bus) window.bus.emit("perspective.switched", { name, ctx });
  }

  // ---- auto-route on asset open --------------------------------------
  async function refresh(ctx) {
    activeCtx = ctx || activeCtx;
    if (!activeCtx) {
      rebuildTabStrip();
      return;
    }
    const recalled = recallLastPerspective(activeCtx.path);
    const best = bestPerspective(activeCtx.entry, activeCtx.fileName, recalled);
    if (best) {
      await switchTo(best, activeCtx);
    } else {
      // No perspective claimed this entry; mount a fallback if registered.
      if (registry.has("asset-info")) await switchTo("asset-info", activeCtx);
      else rebuildTabStrip();
    }
  }

  // ---- tab strip rendering ------------------------------------------
  function rebuildTabStrip() {
    const tabs = $("#vpTabs");
    if (!tabs) return;
    tabs.innerHTML = "";
    const cands = activeCtx
      ? list(activeCtx.entry, activeCtx.fileName).filter((c) => c.score > 0)
      : [];

    if (!cands.length) {
      const hint = document.createElement("span");
      hint.className = "vp-tab-empty dim";
      hint.textContent = "select an asset on the left to begin";
      tabs.appendChild(hint);
      return;
    }

    cands.sort((a, b) => b.score - a.score);

    for (let i = 0; i < cands.length; i++) {
      const c = cands[i];
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "vp-tab" + (c.name === activeName ? " active" : "");
      btn.dataset.perspective = c.name;
      btn.textContent = c.label;
      // Optional numeric badge (1..9) for keyboard hints
      if (i < 9) {
        const num = document.createElement("span");
        num.className = "vp-tab-num";
        num.textContent = String(i + 1);
        btn.appendChild(num);
      }
      btn.addEventListener("click", function () { switchTo(c.name, activeCtx); });
      tabs.appendChild(btn);
    }
  }

  // ---- keyboard 1..9 ------------------------------------------------
  function onKey(e) {
    if (!activeCtx) return;
    // Only intercept if focus isn't in an input / textarea / select
    const t = e.target;
    const tag = t && t.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (t && t.isContentEditable) return;
    // Don't capture if a modifier is held — Ctrl-1, Cmd-1 etc. are reserved
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key < "1" || e.key > "9") return;
    const cands = list(activeCtx.entry, activeCtx.fileName).filter((c) => c.score > 0);
    cands.sort((a, b) => b.score - a.score);
    const idx = e.key.charCodeAt(0) - "1".charCodeAt(0);
    if (idx < cands.length) {
      e.preventDefault();
      switchTo(cands[idx].name, activeCtx);
    }
  }

  // ---- helpers exposed to perspectives ------------------------------
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;",
                '"': "&quot;", "'": "&#39;" })[c];
    });
  }

  // ---- bus subscription ---------------------------------------------
  function wireBus() {
    if (!window.bus) {
      // bus.js not loaded yet — try again on next tick.
      setTimeout(wireBus, 30);
      return;
    }
    window.bus.on("asset.opened", function (evt) {
      if (!evt || !evt.path) return;
      const fileName = evt.path.split("/").pop();
      refresh({ path: evt.path, entry: evt.entry || {}, fileName: fileName });
    });
    // app.js continues to call openFile() for tile grids; some
    // perspectives need to know when the underlying tile data is ready.
    // We don't directly hook that here — perspectives mount lazily.
  }

  function init() {
    // Add the keyboard shortcut handler exactly once.
    if (!window.__psoVPKeyBound) {
      window.__psoVPKeyBound = true;
      document.addEventListener("keydown", onKey);
    }
    wireBus();
    rebuildTabStrip();

    // "Classic UI" toggle: header button flips body.unified-viewport-mode.
    const toggleBtn = document.getElementById("btnClassicUiToggle");
    if (toggleBtn) {
      const apply = function () {
        const on = document.body.classList.contains("unified-viewport-mode");
        toggleBtn.textContent = on ? "classic UI" : "unified UI";
        toggleBtn.title = on
          ? "switch back to the legacy modal-stack UI"
          : "switch to the unified viewport (default)";
        toggleBtn.setAttribute("aria-pressed", on ? "true" : "false");
      };
      apply();
      toggleBtn.addEventListener("click", function () {
        const turningOff = document.body.classList.contains("unified-viewport-mode");
        if (turningOff) {
          // Going classic -> unmount any active perspective so legacy DOM
          // (#fileWorkspace, modal sub-elements) goes back home.
          if (activeName) {
            const spec = registry.get(activeName);
            if (spec) {
              try { spec.unmount(stageEl(), inspectorEl()); }
              catch (_e) {}
            }
            activeName = null;
            // Don't lose the context — user can come back to unified mode.
          }
          document.body.classList.remove("unified-viewport-mode");
        } else {
          document.body.classList.add("unified-viewport-mode");
          // Going classic -> unified: re-route the active context if any.
          // If there's a current open file but no asset.opened ctx yet,
          // synthesize one from psoEditor.state.
          if (!activeCtx) {
            const editorState = window.psoEditor && window.psoEditor.state;
            const cur = editorState && editorState.currentFile;
            if (cur && cur.name) {
              activeCtx = {
                path: cur.name,
                entry: { category: "container", format: cur.name.toLowerCase().endsWith(".prs") ? "PRS" : "XVM" },
                fileName: cur.name,
              };
            }
          }
          if (activeCtx) refresh(activeCtx);
          else rebuildTabStrip();
        }
        apply();
      });
    }
  }

  // ---- public API ---------------------------------------------------
  window.PSOPerspectives = Object.freeze({
    register: register,
    list: list,
    active: active,
    activeContext: activeContext,
    switchTo: switchTo,
    refresh: refresh,
    rebuildTabStrip: rebuildTabStrip,
    escapeHtml: escapeHtml,
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // =====================================================================
  // Built-in perspectives. Defined here (single file) for easy review.
  // Each perspective treats the existing modal DOM nodes as detachable —
  // we pluck the live canvas / panel out of the modal, place it in the
  // stage, and put it back on unmount. That preserves all existing
  // event wiring inside model_viewer.js / viewport.js / app.js.
  // =====================================================================

  function makeRawUrl(path) {
    return "/api/raw/" + path.split("/").map(encodeURIComponent).join("/");
  }

  function fmtSize(n) {
    if (typeof n !== "number" || !isFinite(n) || n < 0) return "";
    if (n < 1024) return n + " B";
    const u = ["KB", "MB", "GB"];
    let v = n / 1024, i = 0;
    while (v >= 1024 && i < u.length - 1) { v /= 1024; i += 1; }
    return v.toFixed(v < 10 ? 1 : 0) + " " + u[i];
  }

  // Helper: detach a node from its parent and append elsewhere; remember
  // origin parent so we can restore on unmount.
  function detachInto(originId, targetEl) {
    const node = document.getElementById(originId);
    if (!node) return null;
    const home = node.parentNode;
    const next = node.nextSibling;
    targetEl.appendChild(node);
    return function restore() {
      if (home) {
        if (next && next.parentNode === home) home.insertBefore(node, next);
        else home.appendChild(node);
      }
    };
  }

  // ---------- TILE-GRID perspective ----------------------------------
  // Hosts the existing #fileWorkspace block (toolbar + tile grid +
  // atlas view + adv settings + progress bar). app.js continues to
  // own those controls; we just relocate them.
  register("tile-grid", {
    label: "Tile grid",
    match: function (entry, file) {
      if (!entry) return 0;
      const cat = entry.category;
      const ext = ((file || "").split(".").pop() || "").toLowerCase();
      if (cat === "texture" || cat === "container") return 100;
      if (ext === "prs" || ext === "xvm") return 80;
      return 0;
    },
    mount: function (stage, insp, ctx) {
      // Pull the live #fileWorkspace into the stage. app.js mutates
      // its child elements directly via document.querySelector — those
      // selectors keep working since we're moving live nodes.
      const restoreFW = detachInto("fileWorkspace", stage);
      stage._restorers = [restoreFW];

      // Drive openFile() if we have a flat filename and app.js exposes it.
      if (ctx && ctx.fileName && typeof window.openFile === "function") {
        // Skip if openFile is already showing the right file
        const cur = window.psoEditor && window.psoEditor.state &&
                    window.psoEditor.state.currentFile &&
                    window.psoEditor.state.currentFile.name;
        if (cur !== ctx.fileName) {
          try { window.openFile(ctx.fileName); } catch (_e) {}
        } else {
          // Make sure the workspace is visible (openFile flips this)
          const fw = document.getElementById("fileWorkspace");
          if (fw) fw.hidden = false;
          const ph = document.getElementById("placeholder");
          if (ph) ph.hidden = true;
        }
      }

      // Inspector: show inline help; the actual upscale settings live in
      // the toolbar inside #fileWorkspace (it's already verbose). Keeping
      // the inspector slim for tile-grid avoids duplicating UI.
      // 2026-06-19 anti-slop: the inspector used to repeat the
      // "view 3D / viewport / AI generate" perspective switches that the
      // tab strip above already provides — two parallel ways to do the
      // same thing. Dropped them; the inspector now carries only the
      // tile-specific help + the deploy-diff action (which is NOT a tab).
      insp.innerHTML =
        '<div class="vp-insp-title">Tile grid</div>' +
        '<div class="vp-insp-help dim">' +
        'Click a tile to enter A/B view. Press <kbd>U</kbd> to upscale all, ' +
        '<kbd>R</kbd> to repack and deploy. Drop a PNG onto a card to import ' +
        'an external upscaled version. Use the tabs above to switch to ' +
        '3D view, viewport paint or AI generate.' +
        '</div>' +
        '<div class="vp-insp-section">' +
        '<button id="vpiOpenDeployPersp" class="ghost" type="button" title="show deploy diff">deploy diff</button>' +
        '</div>';

      const odep = insp.querySelector("#vpiOpenDeployPersp");
      if (odep) odep.addEventListener("click", function () {
        // Reuse existing deploy modal show — but render as a slide-in
        // drawer styled by .deploy-as-drawer; modal class is overridden
        // by body.unified-viewport-mode.
        try {
          const m = document.getElementById("deployModal");
          if (m) m.hidden = false;
          if (typeof window.psoEditor === "object" && window.psoEditor.showDeployDiff) {
            window.psoEditor.showDeployDiff();
          } else if (typeof window.showDeployDiff === "function") {
            window.showDeployDiff();
          } else {
            // Fallback — synthesize a click on the toolbar's repack button.
            const btn = document.getElementById("btnRepack");
            if (btn) btn.click();
          }
        } catch (_e) {}
      });
    },
    unmount: function (stage, insp) {
      try {
        if (stage._restorers) stage._restorers.forEach(function (f) { try { f(); } catch (_e) {} });
        stage._restorers = null;
      } catch (_e) {}
    },
  });

  // ---------- 3D VIEW perspective ------------------------------------
  // Pulls the model viewer's <canvas> + bar + overlay out of the modal.
  // model_viewer.js continues to drive the renderer via its existing
  // querySelector(#modelCanvas) — same node, different parent.
  register("3d-view", {
    label: "3D view",
    match: function (entry, file) {
      if (!entry) return 0;
      const cat = entry.category;
      const ext = ((file || "").split(".").pop() || "").toLowerCase();
      if (cat === "model") return 100;
      // A texture asset can ALSO open in 3D, but only as a secondary
      // tab — score lower than tile-grid so auto-route prefers the tile
      // editor for textures.
      if (cat === "texture" || cat === "container") return 25;
      if (ext === "bml" || ext === "nj" || ext === "xj" || ext === "afs") return 90;
      return 0;
    },
    mount: function (stage, insp, ctx) {
      const card = document.querySelector("#modelModal .model-modal-card");
      const restorers = [];

      // Suppress Esc — model_viewer.js wires it to close() which would
      // dispose the renderer + drop GPU resources. Within the 3d-view
      // perspective the user navigates by tabs, not Esc.
      const escSuppressor = function (e) {
        if (e.key !== "Escape") return;
        const t = e.target;
        const tag = t && t.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        e.stopPropagation();
        e.preventDefault();
      };
      document.addEventListener("keydown", escSuppressor, true);
      stage._3dEscSuppress = escSuppressor;
      // Force-show the modal CSS contents (the wrapper #modelModal stays
      // hidden via body.unified-viewport-mode CSS, but its children move
      // into our stage so we don't need it visible).
      // Rip the inner card pieces out and into the stage:
      const wrapper = document.createElement("div");
      wrapper.className = "vp-stage-3d";
      stage.appendChild(wrapper);

      // Move the model-bar (toolbar) and model-stage (canvas) into the wrapper.
      const bar = document.querySelector("#modelModal .model-bar");
      const mstage = document.querySelector("#modelModal .model-stage");
      const homeBar = bar ? bar.parentNode : null;
      const homeStage = mstage ? mstage.parentNode : null;
      const nextBar = bar ? bar.nextSibling : null;
      const nextStage = mstage ? mstage.nextSibling : null;
      if (bar) wrapper.appendChild(bar);
      if (mstage) wrapper.appendChild(mstage);
      restorers.push(function () {
        if (homeBar && bar) {
          if (nextBar && nextBar.parentNode === homeBar) homeBar.insertBefore(bar, nextBar);
          else homeBar.appendChild(bar);
        }
        if (homeStage && mstage) {
          if (nextStage && nextStage.parentNode === homeStage) homeStage.insertBefore(mstage, nextStage);
          else homeStage.appendChild(mstage);
        }
      });
      stage._restorers = restorers;

      // Inspector summary text
      const path = ctx && ctx.path ? ctx.path : "";
      insp.innerHTML =
        '<div class="vp-insp-title">3D view</div>' +
        '<div class="vp-insp-help dim">' +
        'Drag to rotate, scroll to zoom. Use the toolbar above to pick a ' +
        'tile/shape; texture override (when present) lists matched textures ' +
        'from the manifest.' +
        '</div>' +
        '<div class="vp-insp-section">' +
        '<div class="vp-insp-row dim">model: <code>' + escapeHtml(path) + '</code></div>' +
        '</div>';

      // Drive open() on the underlying viewer, which in turn populates
      // the bar selectors.
      const fileName = ctx && ctx.fileName;
      const cat = ctx && ctx.entry && ctx.entry.category;
      if (cat === "model" && typeof window.psoOpenModelByPath === "function") {
        try { window.psoOpenModelByPath(ctx.path, ctx.entry, ctx.entry && ctx.entry.matched_textures || []); }
        catch (_e) {}
      } else if (fileName && typeof window.openFile === "function") {
        // Texture-driven: load file then trigger the existing open()
        try { window.openFile(fileName); } catch (_e) {}
        // Wait briefly for tile pipeline before opening the 3D viewer.
        setTimeout(function () {
          const btn = document.getElementById("btnView3D");
          if (btn) btn.click();
        }, 200);
      }
      // Force-resize the canvas to fill the new stage layout. The model
      // viewer's ResizeObserver was bound to the OLD parent (modal-stage
      // when first opened); psoModelRebindResize re-attaches it to the
      // new parent and triggers a fresh render.
      setTimeout(function () {
        if (typeof window.psoModelRebindResize === "function") {
          window.psoModelRebindResize();
        }
        window.dispatchEvent(new Event("resize"));
      }, 80);
    },
    unmount: function (stage, insp) {
      // Detach the Esc suppressor first so model_viewer's close() can
      // run normally if other code triggers a close later.
      try {
        if (stage._3dEscSuppress) {
          document.removeEventListener("keydown", stage._3dEscSuppress, true);
          stage._3dEscSuppress = null;
        }
      } catch (_e) {}
      // Tell model viewer to stop animating + drop GPU resources.
      // model_viewer.js binds its close() to #modelClose click; we
      // synthesize that. The header (incl. modelClose) stays in the
      // modal — we only moved model-bar + model-stage out.
      try {
        const closeBtn = document.getElementById("modelClose");
        if (closeBtn) closeBtn.click();
      } catch (_e) {}
      try {
        if (stage._restorers) stage._restorers.forEach(function (f) { try { f(); } catch (_e) {} });
        stage._restorers = null;
      } catch (_e) {}
    },
  });

  // ---------- VIEWPORT PAINT perspective -----------------------------
  register("viewport-paint", {
    label: "Viewport (16:9)",
    match: function (entry, file) {
      if (!entry) return 0;
      const cat = entry.category;
      const ext = ((file || "").split(".").pop() || "").toLowerCase();
      // Only textures/containers have tiles to paint over.
      if (cat === "texture" || cat === "container") return 60;
      if (ext === "prs" || ext === "xvm") return 50;
      return 0;
    },
    mount: function (stage, insp, ctx) {
      // Suppress Esc — viewport.js's internal _keyHandler closes the
      // overlay on Esc; in unified mode we want tabs to be the only
      // way out. Capture-phase + stopImmediatePropagation so the
      // viewport's listener (also capture-phase) doesn't see the event.
      const escSuppressor = function (e) {
        if (e.key !== "Escape") return;
        const t = e.target;
        const tag = t && t.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        e.stopImmediatePropagation();
        e.preventDefault();
      };
      window.addEventListener("keydown", escSuppressor, true);
      stage._vpEscSuppress = escSuppressor;

      // The viewport's <pso-viewport-canvas> Web Component is appended
      // to <body> as a fixed overlay by viewport.js. We bring it into
      // the stage by removing the position:fixed via a CSS class and
      // moving it into the stage element.
      let vp = document.querySelector("pso-viewport-canvas");
      if (!vp) {
        // viewport.js hasn't mounted yet (DOMContentLoaded race). Try to
        // induce a mount.
        const evt = new Event("DOMContentLoaded");
        try { document.dispatchEvent(evt); } catch (_e) {}
        vp = document.querySelector("pso-viewport-canvas");
      }
      const restorers = [];
      if (vp) {
        const home = vp.parentNode;
        const next = vp.nextSibling;
        vp.classList.add("vp-as-stage-content");
        vp.hidden = false;
        stage.appendChild(vp);
        restorers.push(function () {
          vp.classList.remove("vp-as-stage-content");
          vp.hidden = true;
          if (home) {
            if (next && next.parentNode === home) home.insertBefore(vp, next);
            else home.appendChild(vp);
          }
        });
        // Drive open() on the active file.
        const fileName = ctx && ctx.fileName;
        if (fileName && typeof vp.open === "function") {
          try { vp.open(fileName); } catch (_e) {}
        }
      } else {
        stage.innerHTML = '<div class="vp-stage-empty">viewport mount failed</div>';
      }
      stage._restorers = restorers;

      insp.innerHTML =
        '<div class="vp-insp-title">Viewport (16:9)</div>' +
        '<div class="vp-insp-help dim">' +
        'Paint the sprite as it would render at 1278x768. Tools (brush, ' +
        'fill, eyedropper, marquee) and undo/redo live in the toolbar ' +
        'above the canvas. Save flattens the paint layer back into per-tile ' +
        'edits.' +
        '</div>';
    },
    unmount: function (stage, insp) {
      try {
        if (stage._vpEscSuppress) {
          window.removeEventListener("keydown", stage._vpEscSuppress, true);
          stage._vpEscSuppress = null;
        }
      } catch (_e) {}
      try {
        if (stage._restorers) stage._restorers.forEach(function (f) { try { f(); } catch (_e) {} });
        stage._restorers = null;
      } catch (_e) {}
    },
  });

  // ---------- AIGEN perspective --------------------------------------
  // The AI generate panel currently lives inside the per-tile modal as
  // a tab. In unified mode we promote it to a top-level perspective
  // that operates on the currently-selected tile. The user opens a
  // tile (in tile-grid) then switches to "AI gen" — the modal opens
  // headlessly and we yank the AI panel + canvas into the stage.
  register("aigen", {
    label: "AI generate",
    match: function (entry, file) {
      if (!entry) return 0;
      const cat = entry.category;
      const ext = ((file || "").split(".").pop() || "").toLowerCase();
      // AI gen needs a tile context — only score positive for textures.
      if (cat === "texture" || cat === "container") return 20;
      if (ext === "prs" || ext === "xvm") return 15;
      return 0;
    },
    mount: function (stage, insp, ctx) {
      // Make sure the modal is "open" so app.js knows which tile is active.
      // We do NOT visually show #modal — body.unified-viewport-mode hides
      // the modal backdrop. We just need state.modalTileIdx to be set so
      // upscale/AI-gen calls go to the right tile.
      const editorState = window.psoEditor && window.psoEditor.state;
      const cur = editorState && editorState.currentFile;
      if (!cur && ctx && ctx.fileName && typeof window.openFile === "function") {
        try { window.openFile(ctx.fileName); } catch (_e) {}
      }
      const tileIdx = (editorState && editorState.modalTileIdx != null)
        ? editorState.modalTileIdx
        : 0;
      if (typeof window.openModal === "function") {
        // Bypass the unified-mode redirect: aigen owns its own DOM
        // relocation (below), so we want openModal's inner setup to run
        // against the moved nodes, not bounce into tile-detail.
        window._psoOpenModalBypass = true;
        try { window.openModal(tileIdx); }
        catch (_e) {}
        finally { window._psoOpenModalBypass = false; }
      }

      // Esc-suppression: app.js binds a global keydown handler that
      // calls closeModal() when #modal isn't .hidden. In unified mode
      // we want Esc to be a no-op inside the aigen perspective (the
      // user navigates by tabs, not by Esc). Capture-phase listener so
      // we run before app.js's bubble-phase one.
      const escSuppressor = function (e) {
        if (e.key !== "Escape") return;
        // Don't kill Esc when focus is in an input/textarea/select —
        // that would block native blur behaviour.
        const t = e.target;
        const tag = t && t.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        e.stopPropagation();
        e.preventDefault();
      };
      document.addEventListener("keydown", escSuppressor, true);
      stage._aigenEscSuppress = escSuppressor;

      // Pluck the AI panel + AB stage into the stage element.
      const wrapper = document.createElement("div");
      wrapper.className = "vp-stage-aigen";
      stage.appendChild(wrapper);

      // Tabs inside the modal (upscale|AI generate); make AI active.
      const tabUp = document.getElementById("tabUpscale");
      const tabAi = document.getElementById("tabAigen");
      if (tabAi && tabAi.click) tabAi.click();

      const restorers = [];
      const ids = ["modal-upscale-bar-host", "modal-tabs-host", "aigenPanel", "abStageHost"];

      // Take the modal sub-elements we need.
      const movedNodes = [];
      function move(elemSelector) {
        const el = document.querySelector(elemSelector);
        if (!el) return null;
        const home = el.parentNode;
        const next = el.nextSibling;
        wrapper.appendChild(el);
        movedNodes.push({ el: el, home: home, next: next });
        return el;
      }
      move("#modalUpscaleBar");
      move("#aigenPanel");
      move(".modal-anim-bar");
      const abStage = document.querySelector("#modal .ab-stage");
      if (abStage) {
        const home = abStage.parentNode;
        const next = abStage.nextSibling;
        wrapper.appendChild(abStage);
        movedNodes.push({ el: abStage, home: home, next: next });
      }

      restorers.push(function () {
        for (let i = movedNodes.length - 1; i >= 0; i--) {
          const m = movedNodes[i];
          if (!m.home) continue;
          if (m.next && m.next.parentNode === m.home) m.home.insertBefore(m.el, m.next);
          else m.home.appendChild(m.el);
        }
      });
      stage._restorers = restorers;

      // Inspector: tile picker + meta + revert
      let html = '<div class="vp-insp-title">AI generate</div>';
      html += '<div class="vp-insp-help dim">';
      html += 'Re-paint the active tile via your local AI runtime (img2img, ' +
              'inpaint, text2img, ControlNet). Pick a tile via the dropdown.';
      html += '</div>';
      html += '<div class="vp-insp-section"><label>tile: <select id="vpiAigenTileSel"></select></label></div>';
      html += '<div class="vp-insp-section">' +
              '<button id="vpiAigenRevert" class="ghost" type="button">revert</button>' +
              '</div>';
      insp.innerHTML = html;

      const sel = insp.querySelector("#vpiAigenTileSel");
      if (sel && editorState && editorState.currentFile) {
        const tiles = editorState.currentFile.tiles || [];
        sel.innerHTML = tiles.map(function (t) {
          return '<option value="' + t.index + '"' +
                 (t.index === tileIdx ? " selected" : "") +
                 ">tile " + t.index + " (" + t.width + "x" + t.height + ")</option>";
        }).join("");
        sel.addEventListener("change", function () {
          const newIdx = parseInt(sel.value, 10);
          if (Number.isFinite(newIdx) && typeof window.openModal === "function") {
            window._psoOpenModalBypass = true;
            try { window.openModal(newIdx); }
            catch (_e) {}
            finally { window._psoOpenModalBypass = false; }
          }
        });
      }
      const rev = insp.querySelector("#vpiAigenRevert");
      if (rev) rev.addEventListener("click", function () {
        const btn = document.getElementById("modalRevert");
        if (btn) btn.click();
      });
    },
    unmount: function (stage, insp) {
      try {
        if (stage._aigenEscSuppress) {
          document.removeEventListener("keydown", stage._aigenEscSuppress, true);
          stage._aigenEscSuppress = null;
        }
      } catch (_e) {}
      try {
        if (stage._restorers) stage._restorers.forEach(function (f) { try { f(); } catch (_e) {} });
        stage._restorers = null;
      } catch (_e) {}
      // Close the modal so its keyboard handlers don't fire while
      // hidden. closeModal() is exposed via app.js.
      try {
        if (typeof window.closeModal === "function") window.closeModal();
        else {
          const m = document.getElementById("modal");
          if (m) m.hidden = true;
        }
      } catch (_e) {}
    },
  });

  // ---------- ATLAS COMPOSITE perspective ----------------------------
  register("atlas-composite", {
    label: "Atlas composite",
    match: function (entry, file) {
      if (!entry) return 0;
      const cat = entry.category;
      // Score positive but lower than tile-grid for textures, so it
      // shows up as a tab without auto-claiming the asset.
      if (cat === "texture" || cat === "container") return 30;
      return 0;
    },
    mount: function (stage, insp, ctx) {
      // Atlas mode lives inside #atlasView (already part of #fileWorkspace
      // when active). Easiest: switch to tile-grid then flip the toolbar
      // toggle. We do that in two steps: ensure file is loaded, then
      // drive #atlasModeToggle.
      const fileName = ctx && ctx.fileName;
      const editorState = window.psoEditor && window.psoEditor.state;
      const cur = editorState && editorState.currentFile;
      if (fileName && (!cur || cur.name !== fileName) && typeof window.openFile === "function") {
        try { window.openFile(fileName); } catch (_e) {}
      }
      // Move the atlas view + tile grid out of #fileWorkspace into stage
      // so the toolbar buttons remain wired.
      const restoreFW = detachInto("fileWorkspace", stage);
      stage._restorers = [restoreFW];

      // Flip the atlas toggle on (after a tick for openFile pipeline)
      setTimeout(function () {
        const t = document.getElementById("atlasModeToggle");
        if (t && !t.checked) t.click();
      }, 200);

      insp.innerHTML =
        '<div class="vp-insp-title">Atlas composite</div>' +
        '<div class="vp-insp-help dim">' +
        'Edit the assembled composite as one image. The slice strip below ' +
        'shows per-tile boundaries. Click any tile to enter A/B view for it.' +
        '</div>';
    },
    unmount: function (stage, insp) {
      try {
        // Turn atlas mode off so the next perspective starts clean.
        const t = document.getElementById("atlasModeToggle");
        if (t && t.checked) t.click();
      } catch (_e) {}
      try {
        if (stage._restorers) stage._restorers.forEach(function (f) { try { f(); } catch (_e) {} });
        stage._restorers = null;
      } catch (_e) {}
    },
  });

  // ---------- AUDIO perspective --------------------------------------
  // The audio perspective lives in static/audio_panel.js (the audio suite):
  // codec badge, .pac record-picker, waveform canvas, and a DEV-only Replace
  // input. It registers itself via window.PSOPerspectives.register("audio",…)
  // after this file loads, so the stub that used to live here (a bare /api/raw
  // <audio> with the now-false "Ogg Vorbis is the only audio format PSOBB
  // ships" caption) has been removed to avoid a double registration.

  // ---------- HEX perspective ----------------------------------------
  register("hex", {
    label: "Hex",
    match: function (entry, file) {
      if (!entry) return 0;
      const cat = entry.category;
      if (cat === "script" || cat === "quest" || cat === "unknown" || cat === "map") return 90;
      // Hex is also useful for unknown-extension files.
      if (cat === "metadata") return 30;
      return 5;  // always available as a fallback tab
    },
    mount: async function (stage, insp, ctx) {
      const path = ctx && ctx.path ? ctx.path : "";
      const url = makeRawUrl(path);
      stage.innerHTML = '' +
        '<div class="vp-stage-card vp-stage-hex">' +
        '<div class="vp-hex-bar"><label>view: ' +
        '<select id="vpHexMode">' +
        '<option value="hex" selected>hex+ascii</option>' +
        '<option value="text">text</option>' +
        '</select></label>' +
        '<span class="grow"></span>' +
        '<span id="vpHexInfo" class="dim"></span>' +
        '</div>' +
        '<pre id="vpHexBody" class="vp-hex-body">loading...</pre>' +
        '</div>';
      insp.innerHTML = '<div class="vp-insp-title">Hex / text</div>' +
        '<div class="vp-insp-help dim">first 16 KB shown. <a href="' + escapeHtml(url) + '" download>download raw</a> for full bytes.</div>' +
        '<div class="vp-insp-section"><div class="vp-insp-row dim">path: <code>' + escapeHtml(path) + '</code></div></div>';

      const sel = stage.querySelector("#vpHexMode");
      const info = stage.querySelector("#vpHexInfo");
      const body = stage.querySelector("#vpHexBody");
      const HEX_PREVIEW_BYTES = 16 * 1024;
      const TEXT_PREVIEW_BYTES = 64 * 1024;
      let buf;
      try {
        const r = await fetch(url, { cache: "no-store" });
        if (!r.ok) throw new Error("HTTP " + r.status);
        buf = new Uint8Array(await r.arrayBuffer());
      } catch (e) {
        body.textContent = "(load failed: " + (e && e.message || e) + ")";
        return;
      }
      const total = buf.length;
      function render() {
        const m = sel.value;
        if (m === "text") {
          const cap = Math.min(total, TEXT_PREVIEW_BYTES);
          const slice = buf.subarray(0, cap);
          let text;
          try { text = new TextDecoder("utf-8", { fatal: false }).decode(slice); }
          catch (_e) { text = new TextDecoder("latin1").decode(slice); }
          body.textContent = text;
          info.textContent = total > cap
            ? "showing " + cap + " of " + total + " bytes"
            : total + " bytes";
        } else {
          const cap = Math.min(total, HEX_PREVIEW_BYTES);
          body.textContent = formatHexDump(buf.subarray(0, cap));
          info.textContent = total > cap
            ? "showing " + cap + " of " + total + " bytes"
            : total + " bytes";
        }
      }
      sel.addEventListener("change", render);
      render();
    },
    unmount: function () {},
  });

  function formatHexDump(u8) {
    const lines = [];
    for (let off = 0; off < u8.length; off += 16) {
      const row = u8.subarray(off, off + 16);
      let hex = "";
      let ascii = "";
      for (let i = 0; i < 16; i++) {
        if (i < row.length) {
          const b = row[i];
          hex += b.toString(16).padStart(2, "0") + " ";
          ascii += (b >= 0x20 && b < 0x7f) ? String.fromCharCode(b) : ".";
        } else {
          hex += "   ";
        }
        if (i === 7) hex += " ";
      }
      lines.push(off.toString(16).padStart(8, "0") + "  " + hex + " " + ascii);
    }
    return lines.join("\n");
  }

  // ---------- JSON perspective ---------------------------------------
  register("json", {
    label: "JSON",
    match: function (entry, file) {
      if (!entry) return 0;
      const ext = ((file || "").split(".").pop() || "").toLowerCase();
      if (entry.category === "metadata" && ext === "json") return 100;
      if (ext === "json") return 70;
      return 0;
    },
    mount: async function (stage, insp, ctx) {
      const path = ctx && ctx.path ? ctx.path : "";
      const url = makeRawUrl(path);
      stage.innerHTML =
        '<div class="vp-stage-card vp-stage-json"><pre id="vpJsonBody" class="vp-json-body">loading...</pre></div>';
      insp.innerHTML = '<div class="vp-insp-title">JSON</div>' +
        '<div class="vp-insp-help dim">pretty-printed. <a href="' + escapeHtml(url) + '" download>download raw</a> for the original.</div>';

      const body = stage.querySelector("#vpJsonBody");
      try {
        const r = await fetch(url, { cache: "no-store" });
        if (!r.ok) throw new Error("HTTP " + r.status);
        const text = await r.text();
        try { body.textContent = JSON.stringify(JSON.parse(text), null, 2); }
        catch (_e) { body.textContent = text; }
      } catch (e) {
        body.textContent = "(load failed: " + (e && e.message || e) + ")";
      }
    },
    unmount: function () {},
  });

  // ---------- ASSET-INFO perspective ---------------------------------
  // Always-available fallback tab.
  register("asset-info", {
    label: "Asset info",
    match: function (entry) {
      if (!entry) return 0;
      // Always show as a low-score tab. cinematic gets a higher score
      // since there's no other useful view for it yet.
      if (entry.category === "cinematic") return 80;
      return 1;
    },
    mount: function (stage, insp, ctx) {
      const entry = ctx && ctx.entry || {};
      const path = ctx && ctx.path || "";
      const url = makeRawUrl(path);
      const rows = [
        ["path", path],
        ["category", entry.category || "?"],
        ["format", entry.format || "?"],
        ["size", fmtSize(entry.size) + " (" + (entry.size || 0) + " bytes)"],
        ["parsable", entry.parsable || "—"],
      ];
      if (entry.compressed) rows.push(["compressed", "yes"]);
      if (entry.inner_format) rows.push(["inner format", entry.inner_format]);
      if (Array.isArray(entry.matched_textures) && entry.matched_textures.length) {
        const list = entry.matched_textures.map(function (m) {
          return m.path + " (" + m.rule + ", " + (m.confidence || 0).toFixed(2) + ")";
        }).join("\n");
        rows.push(["matched textures", list]);
      }
      if (Array.isArray(entry.warnings) && entry.warnings.length) {
        rows.push(["warnings", entry.warnings.join("\n")]);
      }
      stage.innerHTML =
        '<div class="vp-stage-card vp-stage-info">' +
        '<dl class="vp-info-body">' +
        rows.map(function (kv) {
          return "<dt>" + escapeHtml(kv[0]) + "</dt><dd>" + escapeHtml(kv[1]) + "</dd>";
        }).join("") +
        '</dl></div>';
      insp.innerHTML = '<div class="vp-insp-title">Asset info</div>' +
        '<div class="vp-insp-help dim">manifest metadata for this entry.</div>' +
        '<div class="vp-insp-section"><a class="ghost" href="' + escapeHtml(url) + '" download>download raw</a></div>';
    },
    unmount: function () {},
  });

  // ---------- DEPLOY DIFF perspective --------------------------------
  // Optional perspective: only relevant when the user has staged edits.
  // Lower score so it doesn't auto-route on asset open.
  register("deploy-diff", {
    label: "Deploy diff",
    match: function (entry, file) {
      const editorState = window.psoEditor && window.psoEditor.state;
      const hasEdits = editorState && editorState.tileEdits &&
                       Object.keys(editorState.tileEdits).length > 0;
      // Only show as a tab when there are actually edits to deploy.
      if (hasEdits) return 8;
      return 0;
    },
    mount: function (stage, insp, ctx) {
      stage.innerHTML = '<div class="vp-stage-card vp-stage-deploy">' +
        '<div class="vp-deploy-host" id="vpDeployBody">' +
        '<div class="dim">computing diff...</div>' +
        '</div>' +
        '<div class="vp-deploy-actions">' +
        '<button type="button" id="vpDeployCancel" class="ghost">cancel</button>' +
        '<button type="button" id="vpDeployConfirm">deploy now</button>' +
        '</div>' +
        '</div>';
      insp.innerHTML = '<div class="vp-insp-title">Deploy diff</div>' +
        '<div class="vp-insp-help dim">Preview rebuilt artifact + per-tile diff before writing to disk.</div>';

      // Re-use existing showDeployDiff() — it writes into #deployBody. We
      // pluck #deployBody into our stage temporarily.
      const deployBody = document.getElementById("deployBody");
      const host = stage.querySelector("#vpDeployBody");
      const home = deployBody ? deployBody.parentNode : null;
      const next = deployBody ? deployBody.nextSibling : null;
      if (deployBody && host) {
        host.innerHTML = "";
        host.appendChild(deployBody);
      }
      stage._restorers = [function () {
        if (home && deployBody) {
          if (next && next.parentNode === home) home.insertBefore(deployBody, next);
          else home.appendChild(deployBody);
        }
      }];

      // Drive the existing computation. We deliberately call
      // showDeployDiff() (NOT repackDeployFlow) so the body.pso-modal-
      // allow-deploy class is NOT set — the modal stays CSS-hidden
      // while #deployBody renders inside our vp-stage. The perspective
      // owns its own cancel/confirm buttons (below).
      try {
        if (typeof window.showDeployDiff === "function") {
          window.showDeployDiff();
        } else {
          const btn = document.getElementById("btnRepack");
          if (btn) btn.click();
        }
      } catch (_e) {}

      const cancel = stage.querySelector("#vpDeployCancel");
      if (cancel) cancel.addEventListener("click", function () {
        // Switch back to tile-grid
        switchTo("tile-grid", activeCtx);
      });
      const confirm = stage.querySelector("#vpDeployConfirm");
      if (confirm) confirm.addEventListener("click", function () {
        const real = document.getElementById("deployConfirm");
        if (real) real.click();
      });
    },
    unmount: function (stage) {
      try {
        if (stage._restorers) stage._restorers.forEach(function (f) { try { f(); } catch (_e) {} });
        stage._restorers = null;
      } catch (_e) {}
    },
  });

  // ---------- TILE-DETAIL perspective --------------------------------
  //
  // 2026-04-25 (regression-fix-modal-vs-viewport): replaces the legacy
  // fullscreen #modal A/B view for tile inspection in unified mode. We
  // pluck the SAME DOM nodes the modal owns (#modalUpscaleBar, the
  // #aigenPanel, the .modal-anim-bar, and the .ab-stage) into vp-stage,
  // then call window.openModal(tileIdx) with the bypass flag set so the
  // rest of openModal's wiring (image load, slider, anim runner setup)
  // runs against those same nodes — they're now living in the persistent
  // viewport instead of the modal backdrop.
  //
  // Lifecycle:
  //   mount()    yanks the nodes, calls openModal(idx) via the bypass.
  //   unmount()  restores the nodes home and calls closeModal() so the
  //              animation runners + canvas hidden state get reset.
  //
  // This perspective is intentionally NOT in the auto-route table —
  // match() only scores positive when the active context carries a
  // tileIdx (set by openModal's redirect path). The tab strip therefore
  // shows it only while the user has an active tile selection.
  register("tile-detail", {
    label: "Tile detail",
    match: function (entry, _file) {
      // Score positive only when the active context references a tile
      // (i.e. the user clicked a tile and openModal redirected here). We
      // don't auto-claim texture/container assets — tile-grid does that.
      if (!activeCtx || activeCtx.tileIdx == null) return 0;
      const cat = entry && entry.category;
      if (cat === "texture" || cat === "container") return 8;
      return 0;
    },
    mount: function (stage, insp, ctx) {
      const wrapper = document.createElement("div");
      wrapper.className = "vp-stage-tile-detail vp-stage-aigen";
      stage.appendChild(wrapper);

      // Suppress Esc — app.js wires it to closeModal which would dispose
      // animation state we still want.
      const escSuppressor = function (e) {
        if (e.key !== "Escape") return;
        const t = e.target;
        const tag = t && t.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        e.stopPropagation();
        e.preventDefault();
      };
      document.addEventListener("keydown", escSuppressor, true);
      stage._tdEscSuppress = escSuppressor;

      const movedNodes = [];
      function move(elemSelector) {
        const el = document.querySelector(elemSelector);
        if (!el) return null;
        const home = el.parentNode;
        const next = el.nextSibling;
        wrapper.appendChild(el);
        movedNodes.push({ el: el, home: home, next: next });
        return el;
      }
      // Toolbar (upscale settings) + AI gen panel (hidden by default,
      // user opts in via the existing modal-tabs button which moves with
      // the bar).
      const tabsBar = document.querySelector("#modal .modal-tabs");
      if (tabsBar) {
        const home = tabsBar.parentNode;
        const next = tabsBar.nextSibling;
        wrapper.appendChild(tabsBar);
        movedNodes.push({ el: tabsBar, home: home, next: next });
      }
      move("#modalUpscaleBar");
      move("#aigenPanel");
      move(".modal-anim-bar");
      const abStage = document.querySelector("#modal .ab-stage");
      if (abStage) {
        const home = abStage.parentNode;
        const next = abStage.nextSibling;
        wrapper.appendChild(abStage);
        movedNodes.push({ el: abStage, home: home, next: next });
      }
      stage._restorers = [function () {
        for (let i = movedNodes.length - 1; i >= 0; i--) {
          const m = movedNodes[i];
          if (!m.home) continue;
          if (m.next && m.next.parentNode === m.home) m.home.insertBefore(m.el, m.next);
          else m.home.appendChild(m.el);
        }
      }];

      // Inspector: tile picker + revert + nav buttons.
      const editorState = window.psoEditor && window.psoEditor.state;
      const cur = editorState && editorState.currentFile;
      const tileIdx = (ctx && ctx.tileIdx != null)
        ? ctx.tileIdx
        : (editorState && editorState.modalTileIdx != null
            ? editorState.modalTileIdx
            : 0);
      const tilesArr = (cur && cur.tiles) || [];
      let html = '<div class="vp-insp-title">Tile detail</div>';
      html += '<div class="vp-insp-help dim">';
      html += 'A/B slider compares source vs upscaled. Use the toolbar ';
      html += 'above for re-upscale / revert / animation settings. The ';
      html += 'AI generate tab shares the same canvas.';
      html += '</div>';
      html += '<div class="vp-insp-section"><label>tile: ';
      html += '<select id="vpiTdTileSel">';
      html += tilesArr.map(function (t) {
        return '<option value="' + t.index + '"' +
               (t.index === tileIdx ? " selected" : "") +
               '>tile ' + t.index +
               ' (' + t.width + '\u00d7' + t.height + ')</option>';
      }).join("");
      html += '</select></label></div>';
      html += '<div class="vp-insp-section">';
      html += '<button id="vpiTdRevert" class="ghost" type="button">revert</button> ';
      html += '<button id="vpiTdBackToGrid" class="ghost" type="button">\u2190 back to tile grid</button>';
      html += '</div>';
      insp.innerHTML = html;
      const sel = insp.querySelector("#vpiTdTileSel");
      if (sel) sel.addEventListener("change", function () {
        const newIdx = parseInt(sel.value, 10);
        if (Number.isFinite(newIdx) && typeof window.openModal === "function") {
          // Set the bypass so app.js's openModal does its inner work
          // against the relocated nodes.
          window._psoOpenModalBypass = true;
          try { window.openModal(newIdx); } finally {
            window._psoOpenModalBypass = false;
          }
          if (activeCtx) activeCtx.tileIdx = newIdx;
        }
      });
      const rev = insp.querySelector("#vpiTdRevert");
      if (rev) rev.addEventListener("click", function () {
        const btn = document.getElementById("modalRevert");
        if (btn) btn.click();
      });
      const back = insp.querySelector("#vpiTdBackToGrid");
      if (back) back.addEventListener("click", function () {
        switchTo("tile-grid", activeCtx);
      });

      // Drive openModal with the bypass set so the redirect-loop guard
      // in app.js falls through to the inner setup. The relocated nodes
      // in vp-stage receive the image / slider / anim wiring.
      window._psoOpenModalBypass = true;
      try {
        if (typeof window.openModal === "function") window.openModal(tileIdx);
      } finally {
        window._psoOpenModalBypass = false;
      }
    },
    unmount: function (stage, insp) {
      try {
        if (stage._tdEscSuppress) {
          document.removeEventListener("keydown", stage._tdEscSuppress, true);
          stage._tdEscSuppress = null;
        }
      } catch (_e) {}
      try {
        if (stage._restorers) stage._restorers.forEach(function (f) { try { f(); } catch (_e) {} });
        stage._restorers = null;
      } catch (_e) {}
      // Stop animation runners / clear modalTileIdx so re-entering
      // tile-grid doesn't render stale modal state.
      try {
        if (typeof window.closeModal === "function") window.closeModal();
        else {
          const m = document.getElementById("modal");
          if (m) m.hidden = true;
        }
      } catch (_e) {}
    },
  });

})();
