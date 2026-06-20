// =====================================================================
// PSOBB Texture Editor - Workspace persistence (2026-04-25).
//
// Save / restore which panels are open, which asset is loaded, scroll
// positions, motion-picker state, edit-mode toggles, brush settings.
// Layout-level state shouldn't reset on browser reload.
//
// Persistence layers:
//   - Auto-save every 5 s (debounced) to localStorage['pso.workspace'].
//   - Auto-save on visibilitychange when the tab goes background.
//   - Reload restores from localStorage on page load.
//   - Header buttons "Save Workspace as..." / "Load Workspace..." /
//     "Recent" call /api/workspace/{save,load,list,delete} which write
//     named JSON snapshots to cache/workspaces/<name>.json on the
//     server. localStorage is the unnamed/automatic snapshot.
//
// Saved blob shape:
//   {
//     version: 1,
//     ts: <epoch_ms>,
//     activePerspective: "tile-grid" | ... | null,
//     activePath: "<asset path>" | null,
//     selection: [paths],
//     hideAssetTree: bool, hideFileList: bool,
//     scroll: { tree: number, hex: number, ... },
//     panels: {
//       paint: { color, brushSize, hardness, opacity, tool },
//       sculpt: { brush, radius, strength, ... },
//       mob_dsl: { variant, slot, difficulty, patches: {...} },
//       motion_picker: { motionId, fps, loop, scrub },
//       texture_panel: { activeTab },
//       ...
//     },
//     hotkeyOverrides: {...},
//   }
//
// Each panel that has interesting transient state implements:
//   window.psoWorkspace.registerPanel(panelId, {
//     snapshot: () => ({...}),         // returns serialisable object
//     restore: (state) => void,        // accepts what snapshot returned
//   });
// On save we walk the registry; on restore we replay every panel's
// restore() in registration order. Panels that aren't loaded yet
// register lazily; their snapshot is held in lastLoaded until they
// register, then auto-applied.
// =====================================================================

(function () {
  "use strict";

  if (window.psoWorkspace) return;

  const LS_KEY = "pso.workspace";
  const RECENT_KEY = "pso.workspace.recent";
  const AUTOSAVE_MS = 5000;
  const MAX_RECENT = 8;
  const VERSION = 1;

  // ---- panel registry ---------------------------------------------
  // panelId -> { snapshot, restore }
  const panels = new Map();
  // Last loaded blob, kept around so panels that register AFTER a
  // restore can still pick up their state.
  let lastLoaded = null;

  function registerPanel(panelId, hooks) {
    if (typeof panelId !== "string" || !panelId) return;
    if (!hooks || typeof hooks.snapshot !== "function" || typeof hooks.restore !== "function") {
      console.warn("[workspace] registerPanel needs {snapshot, restore} fns");
      return;
    }
    panels.set(panelId, hooks);
    // If we have pending state for this panel from a previous load,
    // hand it back now (covers the scenario where the panel mounts
    // after our initial restore pass).
    if (lastLoaded && lastLoaded.panels && lastLoaded.panels[panelId]) {
      try { hooks.restore(lastLoaded.panels[panelId]); }
      catch (e) { console.warn("[workspace] restore '" + panelId + "' threw:", e); }
    }
  }

  // ---- snapshot / restore ------------------------------------------
  function captureLayout() {
    const layout = {
      activePerspective: null,
      activePath: null,
      hideAssetTree: document.body.classList.contains("hide-asset-tree"),
      hideFileList: document.body.classList.contains("hide-file-list"),
      scroll: {},
    };
    if (window.PSOPerspectives) {
      try {
        layout.activePerspective = window.PSOPerspectives.active() || null;
        const ctx = window.PSOPerspectives.activeContext();
        if (ctx && ctx.path) layout.activePath = ctx.path;
      } catch (_e) {}
    }
    // Common scroll positions worth restoring.
    const treeBody = document.querySelector("pso-asset-tree");
    if (treeBody && treeBody.shadowRoot) {
      const body = treeBody.shadowRoot.querySelector(".body");
      if (body) layout.scroll.tree = body.scrollTop || 0;
    }
    return layout;
  }

  function snapshot() {
    const blob = {
      version: VERSION,
      ts: Date.now(),
    };
    Object.assign(blob, captureLayout());
    blob.selection = window.psoSelection ? window.psoSelection.getActive().slice() : [];
    blob.hotkeyOverrides = (function () {
      try { return JSON.parse(localStorage.getItem("pso.hotkeys") || "{}"); }
      catch (_e) { return {}; }
    })();
    blob.panels = {};
    for (const [id, h] of panels) {
      try {
        const s = h.snapshot();
        if (s !== undefined) blob.panels[id] = s;
      } catch (e) {
        console.warn("[workspace] snapshot '" + id + "' threw:", e);
      }
    }
    return blob;
  }

  function restoreLayout(blob) {
    if (!blob) return;
    if (typeof blob.hideAssetTree === "boolean") {
      document.body.classList.toggle("hide-asset-tree", blob.hideAssetTree);
    }
    if (typeof blob.hideFileList === "boolean") {
      document.body.classList.toggle("hide-file-list", blob.hideFileList);
    }
    if (Array.isArray(blob.selection) && window.psoSelection) {
      window.psoSelection.replaceAll(blob.selection);
    }
    if (blob.scroll && blob.scroll.tree != null) {
      const treeBody = document.querySelector("pso-asset-tree");
      if (treeBody && treeBody.shadowRoot) {
        const body = treeBody.shadowRoot.querySelector(".body");
        if (body) {
          // requestAnimationFrame so the tree's render has a chance to fill body.
          requestAnimationFrame(function () {
            try { body.scrollTop = blob.scroll.tree; } catch (_e) {}
          });
        }
      }
    }
    // The active asset / perspective is the most fragile to restore
    // because it depends on the tree+manifest having loaded. We do a
    // best-effort: emit asset.opened on the bus once the manifest is
    // ready, then PSOPerspectives.refresh handles the rest.
    if (blob.activePath && window.bus && window.PSOManifest) {
      const tryOpen = function () {
        if (!window.PSOManifest.isLoaded()) {
          setTimeout(tryOpen, 200);
          return;
        }
        // Look up the entry — emit if found.
        const entries = window.PSOManifest.entries();
        let entry = null;
        for (const e of entries) {
          if (e && e.path === blob.activePath) { entry = e; break; }
        }
        if (entry) {
          try { window.bus.emit("asset.opened", { path: blob.activePath, entry: entry }); }
          catch (_e) {}
          // After that bus event resolves into a perspective, force the
          // remembered perspective if present.
          if (blob.activePerspective && window.PSOPerspectives) {
            setTimeout(function () {
              try {
                window.PSOPerspectives.switchTo(blob.activePerspective, {
                  path: blob.activePath,
                  entry: entry,
                  fileName: blob.activePath.split("/").pop(),
                });
              } catch (_e) {}
            }, 250);
          }
        }
      };
      tryOpen();
    }
  }

  function restorePanels(blob) {
    if (!blob || !blob.panels) return;
    for (const [id, h] of panels) {
      const state = blob.panels[id];
      if (state == null) continue;
      try { h.restore(state); }
      catch (e) { console.warn("[workspace] restore '" + id + "' threw:", e); }
    }
  }

  function restore(blob) {
    if (!blob || typeof blob !== "object") return;
    if (blob.version !== VERSION) {
      // Version skew: best-effort. We attempt to restore anyway since
      // the schema is additive; panels validate their own state.
      console.warn("[workspace] version skew (got " + blob.version + ", want " + VERSION + ")");
    }
    lastLoaded = blob;
    restoreLayout(blob);
    restorePanels(blob);
  }

  // ---- localStorage save/load --------------------------------------
  function saveLocal() {
    try {
      const blob = snapshot();
      localStorage.setItem(LS_KEY, JSON.stringify(blob));
    } catch (e) {
      // Quota exceeded? Snapshot too big? Drop the largest panels.
      console.warn("[workspace] saveLocal failed:", e);
    }
  }

  function loadLocal() {
    try {
      const raw = localStorage.getItem(LS_KEY);
      if (!raw) return null;
      return JSON.parse(raw);
    } catch (_e) {
      return null;
    }
  }

  // ---- debounced auto-save -----------------------------------------
  let saveTimer = null;
  function scheduleAutoSave() {
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(function () {
      saveTimer = null;
      saveLocal();
    }, AUTOSAVE_MS);
  }

  // ---- named server-side workspaces -------------------------------
  async function saveNamed(name) {
    if (typeof name !== "string" || !name) throw new Error("name required");
    const blob = snapshot();
    const r = await fetch("/api/workspace/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: name, blob: blob }),
    });
    if (!r.ok) {
      let det = "HTTP " + r.status;
      try { det = (await r.json()).detail || det; } catch (_e) {}
      throw new Error(det);
    }
    rememberRecent(name);
    return r.json();
  }

  async function loadNamed(name) {
    if (typeof name !== "string" || !name) throw new Error("name required");
    const r = await fetch("/api/workspace/load?name=" + encodeURIComponent(name));
    if (!r.ok) {
      let det = "HTTP " + r.status;
      try { det = (await r.json()).detail || det; } catch (_e) {}
      throw new Error(det);
    }
    const data = await r.json();
    restore(data.blob);
    rememberRecent(name);
    return data;
  }

  async function listNamed() {
    const r = await fetch("/api/workspace/list");
    if (!r.ok) return [];
    return (await r.json()).workspaces || [];
  }

  async function deleteNamed(name) {
    if (typeof name !== "string" || !name) throw new Error("name required");
    const r = await fetch("/api/workspace/delete?name=" + encodeURIComponent(name), {
      method: "POST",
    });
    return r.ok;
  }

  // Recent-list lives in localStorage so the dropdown is instant.
  function rememberRecent(name) {
    try {
      const raw = localStorage.getItem(RECENT_KEY);
      let arr = raw ? JSON.parse(raw) : [];
      if (!Array.isArray(arr)) arr = [];
      arr = arr.filter(function (n) { return n !== name; });
      arr.unshift(name);
      arr = arr.slice(0, MAX_RECENT);
      localStorage.setItem(RECENT_KEY, JSON.stringify(arr));
    } catch (_e) {}
  }

  function getRecent() {
    try {
      const raw = localStorage.getItem(RECENT_KEY);
      const arr = raw ? JSON.parse(raw) : [];
      return Array.isArray(arr) ? arr : [];
    } catch (_e) { return []; }
  }

  // ---- header UI ---------------------------------------------------
  // Append "save / load / recent" buttons to the header. Done lazily on
  // DOMContentLoaded so we don't race the existing markup.
  function ensureHeaderUI() {
    const header = document.querySelector("header");
    if (!header) return;
    if (document.getElementById("psoWsHeader")) return;
    const wrap = document.createElement("span");
    wrap.id = "psoWsHeader";
    wrap.className = "ws-header";
    wrap.innerHTML =
      '<button type="button" id="psoWsSave" class="ghost" title="save current layout + per-panel state to a named workspace (Ctrl+S)">save workspace</button>' +
      '<button type="button" id="psoWsLoad" class="ghost" title="load a named workspace from server cache">load workspace</button>' +
      '<button type="button" id="psoWsRecent" class="ghost" title="recent workspaces">recent ▾</button>';
    // Insert before the status pill so it doesn't push the layout right.
    const status = header.querySelector("#status");
    if (status) header.insertBefore(wrap, status);
    else header.appendChild(wrap);
    wrap.querySelector("#psoWsSave").addEventListener("click", onClickSave);
    wrap.querySelector("#psoWsLoad").addEventListener("click", onClickLoad);
    wrap.querySelector("#psoWsRecent").addEventListener("click", onClickRecent);
  }

  async function onClickSave() {
    const n = window.prompt("Save workspace as:", "default");
    if (!n) return;
    try {
      await saveNamed(n);
      window.psoToast ? window.psoToast("workspace saved: " + n) : console.log("workspace saved", n);
    } catch (e) {
      alert("save failed: " + (e.message || e));
    }
  }

  async function onClickLoad() {
    let names = [];
    try { names = (await listNamed()).map(function (w) { return w.name; }); }
    catch (_e) {}
    if (!names.length) { alert("no saved workspaces"); return; }
    const n = window.prompt("Load workspace name (saved: " + names.join(", ") + "):", names[0]);
    if (!n) return;
    try {
      await loadNamed(n);
      window.psoToast ? window.psoToast("workspace loaded: " + n) : console.log("workspace loaded", n);
    } catch (e) {
      alert("load failed: " + (e.message || e));
    }
  }

  function onClickRecent() {
    closeRecentMenu();
    const arr = getRecent();
    if (!arr.length) { alert("no recent workspaces — save one first"); return; }
    const menu = document.createElement("div");
    menu.id = "psoWsRecentMenu";
    menu.className = "ws-recent-menu";
    const rect = document.getElementById("psoWsRecent").getBoundingClientRect();
    menu.style.position = "fixed";
    menu.style.top = (rect.bottom + 4) + "px";
    menu.style.left = rect.left + "px";
    menu.innerHTML = '<div class="ws-recent-title">Recent (newest first)</div>' +
      arr.map(function (n) {
        return '<button type="button" class="ws-recent-item" data-n="' +
               n.replace(/&/g, "&amp;").replace(/"/g, "&quot;") + '">' +
               n.replace(/&/g, "&amp;").replace(/</g, "&lt;") + '</button>';
      }).join("");
    document.body.appendChild(menu);
    menu.addEventListener("click", async function (e) {
      const b = e.target.closest("button.ws-recent-item");
      if (!b) return;
      const n = b.dataset.n;
      closeRecentMenu();
      try {
        await loadNamed(n);
        window.psoToast ? window.psoToast("workspace loaded: " + n) : console.log("workspace loaded", n);
      } catch (e2) {
        alert("load failed: " + (e2.message || e2));
      }
    });
    setTimeout(function () {
      document.addEventListener("click", _outsideRecentClick);
    }, 0);
  }

  function _outsideRecentClick(e) {
    const m = document.getElementById("psoWsRecentMenu");
    if (!m) { document.removeEventListener("click", _outsideRecentClick); return; }
    if (m.contains(e.target)) return;
    closeRecentMenu();
  }

  function closeRecentMenu() {
    const m = document.getElementById("psoWsRecentMenu");
    if (m && m.parentNode) m.parentNode.removeChild(m);
    document.removeEventListener("click", _outsideRecentClick);
  }

  // ---- bootstrap ---------------------------------------------------
  function init() {
    ensureHeaderUI();
    // Restore from localStorage on load. Run BEFORE wiring auto-save so
    // we don't immediately overwrite the saved blob with a stub.
    try {
      const blob = loadLocal();
      if (blob) restore(blob);
    } catch (e) { console.warn("[workspace] init restore threw:", e); }

    // Wire auto-save on bus channels that signal layout-relevant
    // changes. Each emission schedules a debounced save.
    if (window.bus) {
      const channels = [
        "perspective.switched",
        "asset.opened",
        "selection.changed",
        "undo.applied",
      ];
      for (const ch of channels) window.bus.on(ch, scheduleAutoSave);
    }
    // Visibility-change: catch tab-switch / close cases.
    document.addEventListener("visibilitychange", function () {
      if (document.visibilityState === "hidden") {
        if (saveTimer) { clearTimeout(saveTimer); saveTimer = null; }
        saveLocal();
      }
    });
    // beforeunload as a last-ditch save (browsers may skip it).
    window.addEventListener("beforeunload", saveLocal);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // ---- public API --------------------------------------------------
  window.psoWorkspace = Object.freeze({
    registerPanel: registerPanel,
    snapshot: snapshot,
    restore: restore,
    saveLocal: saveLocal,
    loadLocal: loadLocal,
    saveNamed: saveNamed,
    loadNamed: loadNamed,
    listNamed: listNamed,
    deleteNamed: deleteNamed,
    getRecent: getRecent,
    scheduleAutoSave: scheduleAutoSave,
    VERSION: VERSION,
  });
})();
