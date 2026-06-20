// =====================================================================
// PSOBB Texture Editor - Cross-panel multi-selection store (2026-04-25).
//
// Currently every authoring action targets ONE asset. Power users want
// to select multiple textures in the manifest tree (Ctrl+click /
// Shift+click), or multiple mobs in the mob_dsl panel, then run a
// batch operation — "upscale all selected x2", "apply preset to all",
// "set atp_max=999 across these 12 weapons".
//
// This module is the single source of truth for the active selection.
// Panels SUBSCRIBE to the `selection.changed` bus channel to refresh
// their batch toolbar; consumers query psoSelection.getActive() at the
// moment they need to enumerate the chosen items.
//
// Public API:
//   psoSelection.add(path)        - add a single path (no-op if already in)
//   psoSelection.remove(path)     - remove
//   psoSelection.toggle(path)     - flip
//   psoSelection.clear()          - empty
//   psoSelection.has(path)        - bool
//   psoSelection.getActive()      - frozen array of paths in insertion order
//   psoSelection.size()           - count
//   psoSelection.forEach(fn)      - iterate (insertion order)
//   psoSelection.replaceAll(arr)  - bulk replace (used by tree.js shift-click)
//
// Bus channels:
//   selection.changed   { paths: string[] }   - emitted on any mutation
//
// Persistence: kept in localStorage['pso.selection'] so a refresh
// doesn't lose your in-progress batch. Bounded at 1000 entries.
// =====================================================================

(function () {
  "use strict";

  if (window.psoSelection) return;

  const LS_KEY = "pso.selection";
  const MAX_ENTRIES = 1000;

  // Use a Set for O(1) has(); also keep an array for stable insertion-
  // order iteration (matters for "apply to selection" deterministic UX).
  const set = new Set();
  const order = [];

  // ---- persistence -------------------------------------------------
  function _load() {
    try {
      const raw = localStorage.getItem(LS_KEY);
      if (!raw) return;
      const arr = JSON.parse(raw);
      if (!Array.isArray(arr)) return;
      for (const p of arr) {
        if (typeof p === "string" && p && set.size < MAX_ENTRIES) {
          if (!set.has(p)) {
            set.add(p);
            order.push(p);
          }
        }
      }
    } catch (_e) { /* ignore */ }
  }

  function _persist() {
    try {
      localStorage.setItem(LS_KEY, JSON.stringify(order));
    } catch (_e) { /* quota / disabled */ }
  }

  // ---- mutation ----------------------------------------------------
  function _emit() {
    _persist();
    try {
      if (window.bus && typeof window.bus.emit === "function") {
        window.bus.emit("selection.changed", { paths: order.slice() });
      }
    } catch (_e) {}
  }

  function add(path) {
    if (typeof path !== "string" || !path) return false;
    if (set.has(path)) return false;
    if (set.size >= MAX_ENTRIES) {
      // FIFO eviction so the head doesn't grow without bound.
      const dropped = order.shift();
      if (dropped) set.delete(dropped);
    }
    set.add(path);
    order.push(path);
    _emit();
    return true;
  }

  function remove(path) {
    if (typeof path !== "string" || !path) return false;
    if (!set.has(path)) return false;
    set.delete(path);
    const idx = order.indexOf(path);
    if (idx >= 0) order.splice(idx, 1);
    _emit();
    return true;
  }

  function toggle(path) {
    if (set.has(path)) {
      remove(path);
      return false;
    }
    add(path);
    return true;
  }

  function clear() {
    if (!set.size) return;
    set.clear();
    order.length = 0;
    _emit();
  }

  function has(path) {
    return set.has(path);
  }

  function size() {
    return set.size;
  }

  function getActive() {
    // Frozen copy so callers can't mutate the internal array.
    return Object.freeze(order.slice());
  }

  function forEach(fn) {
    if (typeof fn !== "function") return;
    for (let i = 0; i < order.length; i++) {
      try { fn(order[i], i); } catch (e) { console.error("[multi_select] forEach cb threw:", e); }
    }
  }

  function replaceAll(arr) {
    if (!Array.isArray(arr)) arr = [];
    set.clear();
    order.length = 0;
    for (const p of arr) {
      if (typeof p !== "string" || !p) continue;
      if (set.has(p)) continue;
      if (set.size >= MAX_ENTRIES) break;
      set.add(p);
      order.push(p);
    }
    _emit();
  }

  // ---- selection-count badge in the header --------------------------
  // The header markup already has #selectionCount (see index.html); we
  // just toggle visibility + update text on every change.
  function refreshHeaderBadge() {
    const el = document.getElementById("selectionCount");
    if (!el) return;
    const n = set.size;
    if (n <= 1) {
      el.hidden = true;
      el.textContent = "";
    } else {
      el.hidden = false;
      el.textContent = n + " selected";
      el.title = "click to clear selection";
      if (!el._psoMsBound) {
        el._psoMsBound = true;
        el.addEventListener("click", function () { clear(); });
      }
    }
    refreshFloatingBar();
  }

  // ---- floating batch-actions bar ----------------------------------
  // Shown anchored to the bottom of the asset tree when selection > 1.
  // Contains preset actions ("Upscale x2"), a clear button, and an
  // "Apply to selection" hook other panels can populate via
  // psoSelection.registerAction.
  let extraActions = [];
  function registerAction(spec) {
    if (!spec || typeof spec.label !== "string" || typeof spec.run !== "function") return;
    extraActions.push(spec);
    refreshFloatingBar();
  }

  function refreshFloatingBar() {
    let bar = document.getElementById("psoMsFloatingBar");
    if (set.size <= 1) {
      if (bar && bar.parentNode) bar.parentNode.removeChild(bar);
      return;
    }
    if (!bar) {
      bar = document.createElement("div");
      bar.id = "psoMsFloatingBar";
      bar.className = "ms-batch-bar";
      bar.style.position = "fixed";
      bar.style.bottom = "10px";
      bar.style.left = "50%";
      bar.style.transform = "translateX(-50%)";
      bar.style.zIndex = "3500";
      document.body.appendChild(bar);
    }
    const actions = [];
    actions.push(
      '<button type="button" class="primary" data-act="upscale-x2">Upscale x2 selected (' + set.size + ')</button>'
    );
    actions.push(
      '<button type="button" data-act="upscale-x4">Upscale x4 selected</button>'
    );
    for (let i = 0; i < extraActions.length; i++) {
      const a = extraActions[i];
      if (a.match) {
        try {
          if (!a.match(getActive())) continue;
        } catch (_e) { continue; }
      }
      actions.push('<button type="button" data-extra="' + i + '">' +
                   String(a.label).replace(/[<>&"]/g, function (c) {
                     return ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;" })[c];
                   }) +
                   '</button>');
    }
    bar.innerHTML =
      '<span class="ms-batch-count">' + set.size + ' selected</span>' +
      actions.join("") +
      '<button type="button" data-act="clear">clear</button>';
    bar.onclick = function (e) {
      const btn = e.target.closest("button");
      if (!btn) return;
      const act = btn.dataset.act;
      if (act === "clear") { clear(); return; }
      if (act === "upscale-x2") return runBuiltinUpscale(2);
      if (act === "upscale-x4") return runBuiltinUpscale(4);
      if (btn.dataset.extra != null) {
        const i = parseInt(btn.dataset.extra, 10);
        const a = extraActions[i];
        if (a) try { a.run(getActive()); } catch (e2) { console.error(e2); }
      }
    };
  }

  async function runBuiltinUpscale(scale) {
    const paths = getActive();
    if (!paths.length) return;
    if (window.psoToast) window.psoToast("upscaling " + paths.length + " selected (x" + scale + ")…");
    try {
      const r = await fetch("/api/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          op: "upscale",
          paths: paths,
          payload: { scale: scale },
        }),
      });
      if (!r.ok) {
        let det = "HTTP " + r.status;
        try { det = (await r.json()).detail || det; } catch (_e) {}
        if (window.psoToast) window.psoToast("batch upscale failed: " + det);
        return;
      }
      const data = await r.json();
      if (window.psoToast) {
        window.psoToast("batch upscale: " + data.ok + " ok, " + data.failed + " failed");
      }
    } catch (e) {
      if (window.psoToast) window.psoToast("batch upscale threw: " + (e.message || e));
    }
  }

  function _wireBus() {
    refreshHeaderBadge();
    if (!window.bus) return;
    window.bus.on("selection.changed", refreshHeaderBadge);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", _wireBus);
  } else {
    _wireBus();
  }

  _load();

  // ---- public API ---------------------------------------------------
  window.psoSelection = Object.freeze({
    add: add,
    remove: remove,
    toggle: toggle,
    clear: clear,
    has: has,
    size: size,
    getActive: getActive,
    forEach: forEach,
    replaceAll: replaceAll,
    registerAction: registerAction,
    MAX_ENTRIES: MAX_ENTRIES,
  });
})();
