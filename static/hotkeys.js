// =====================================================================
// PSOBB Texture Editor - Unified hotkey registry (2026-04-25).
//
// Currently scattered: B/E/I/F/S in paint, Ctrl+Z everywhere, perspective
// tab keys (1-9), motion-picker arrows. This module is the single
// place where global bindings live AND the source for the help overlay
// + the user-rebind UI in the Settings perspective.
//
// Public API:
//   psoHotkeys.bind(combo, actionId, callback, opts)
//     combo: "Ctrl+P" / "?" / "B" / "Ctrl+Shift+Z" — case-insensitive
//     actionId: stable string for rebinding ("open-quick-search")
//     callback: function called when combo fires
//     opts.scope: "global" (default) | "paint" | "mob_dsl" | ...
//                 panel-scoped binds only fire when the matching panel
//                 advertises itself as active via setActiveScope()
//     opts.allowInInput: if true, fires even when an INPUT is focused
//     Returns a disposer fn.
//
//   psoHotkeys.bindings()      - [{combo, actionId, scope, label}]
//                                 Reflects user overrides.
//   psoHotkeys.rebind(actionId, newCombo)
//   psoHotkeys.reset(actionId) - revert to default
//   psoHotkeys.setActiveScope(scope) - panel-active gating
//   psoHotkeys.openHelp()
//   psoHotkeys.closeHelp()
//
// Persistence:
//   localStorage['pso.hotkeys'] = { actionId: combo }
//
// Help overlay:
//   ? key opens a centered modal listing every binding grouped by
//   panel. Click an entry to rebind it (records next keypress).
//   Esc closes.
// =====================================================================

(function () {
  "use strict";

  if (window.psoHotkeys) return;

  const LS_KEY = "pso.hotkeys";

  // actionId -> { defaultCombo, scope, label, callback, allowInInput }
  const actions = new Map();
  // current combo override map: actionId -> combo
  let overrides = (function () {
    try {
      const r = localStorage.getItem(LS_KEY);
      const o = r ? JSON.parse(r) : {};
      return (o && typeof o === "object") ? o : {};
    } catch (_e) { return {}; }
  })();

  // Reverse map: comboNormalized -> [actionId, ...]
  // Rebuilt whenever bindings change.
  let comboToAction = new Map();

  // Active panel scope. Panel-scoped bindings only fire when this is
  // the panel that owns them. "global" scope always fires.
  let activeScope = "global";

  // --------- combo parsing ---------------------------------------
  // Normalize "Ctrl+Shift+Z" -> "ctrl+shift+z"; sort modifiers so
  // "Shift+Ctrl+Z" === "Ctrl+Shift+Z".
  function normalize(combo) {
    if (!combo || typeof combo !== "string") return "";
    const parts = combo.split("+").map(function (s) { return s.trim().toLowerCase(); });
    if (!parts.length) return "";
    const mods = [];
    let key = "";
    for (const p of parts) {
      if (p === "ctrl" || p === "control") mods.push("ctrl");
      else if (p === "shift") mods.push("shift");
      else if (p === "alt" || p === "option") mods.push("alt");
      else if (p === "meta" || p === "cmd" || p === "command") mods.push("meta");
      else if (p) key = p;
    }
    // Order: ctrl, shift, alt, meta — stable ordering for hash equality.
    const order = ["ctrl", "shift", "alt", "meta"];
    const sortedMods = order.filter(function (m) { return mods.indexOf(m) >= 0; });
    return [].concat(sortedMods, [key]).join("+");
  }

  function rebuildReverse() {
    comboToAction = new Map();
    for (const [actionId, def] of actions) {
      const combo = overrides[actionId] || def.defaultCombo;
      const norm = normalize(combo);
      if (!norm) continue;
      let arr = comboToAction.get(norm);
      if (!arr) { arr = []; comboToAction.set(norm, arr); }
      arr.push(actionId);
    }
  }

  function persist() {
    try {
      localStorage.setItem(LS_KEY, JSON.stringify(overrides));
    } catch (_e) {}
  }

  // --------- API ---------------------------------------------------
  function bind(combo, actionId, callback, opts) {
    if (!actionId || typeof actionId !== "string") return function () {};
    if (typeof callback !== "function") return function () {};
    opts = opts || {};
    actions.set(actionId, {
      defaultCombo: combo,
      scope: opts.scope || "global",
      label: opts.label || actionId,
      callback: callback,
      allowInInput: !!opts.allowInInput,
    });
    rebuildReverse();
    return function dispose() {
      actions.delete(actionId);
      rebuildReverse();
    };
  }

  function bindings() {
    const out = [];
    for (const [actionId, def] of actions) {
      out.push({
        actionId: actionId,
        combo: overrides[actionId] || def.defaultCombo,
        defaultCombo: def.defaultCombo,
        scope: def.scope,
        label: def.label,
      });
    }
    return out;
  }

  function rebind(actionId, newCombo) {
    if (!actions.has(actionId)) return false;
    overrides[actionId] = newCombo;
    persist();
    rebuildReverse();
    return true;
  }

  function reset(actionId) {
    if (overrides[actionId]) {
      delete overrides[actionId];
      persist();
      rebuildReverse();
    }
  }

  function setActiveScope(scope) {
    activeScope = scope || "global";
  }

  // --------- key handler -------------------------------------------
  // Shifted-symbol keys (US layout): the user CANNOT type these without
  // holding Shift, so a bind like `?` should fire on Shift+/. Reporting
  // `shift+?` from the event would never match the registered `?` and
  // the hotkey would silently fail. We strip `shift` from the combo
  // when the key char is one of these — the shift state is already
  // implicit in the character. (Other layouts produce different chars
  // for these positions, but ev.key reflects the LAYOUT-aware char so
  // the strip is safe across layouts.)
  const SHIFTED_SYMBOL_KEYS = new Set([
    "?", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")",
    "_", "+", "{", "}", "|", ":", "\"", "<", ">", "~",
  ]);

  function comboFromEvent(ev) {
    const mods = [];
    if (ev.ctrlKey) mods.push("ctrl");
    let pushShift = !!ev.shiftKey;
    if (ev.altKey) mods.push("alt");
    if (ev.metaKey) mods.push("meta");
    let k = (ev.key || "").toLowerCase();
    if (k === " ") k = "space";
    if (k === "escape") k = "escape";
    if (k === "tab") k = "tab";
    if (k === "arrowleft") k = "left";
    if (k === "arrowright") k = "right";
    if (k === "arrowup") k = "up";
    if (k === "arrowdown") k = "down";
    // Strip shift when the resulting key is a shifted-symbol — the
    // shift state is implicit in the character itself. ev.key reports
    // the layout-aware character (so `Shift+/` on US becomes `?`,
    // `Shift+'` on UK becomes `@`, etc.); whatever produced the
    // character, we want the no-shift bind to match.
    if (pushShift && SHIFTED_SYMBOL_KEYS.has(k)) {
      pushShift = false;
    }
    if (pushShift) {
      // Insert shift after ctrl, before alt — preserves the canonical
      // ordering ctrl,shift,alt,meta that `normalize` enforces.
      const ctrlIdx = mods.indexOf("ctrl");
      mods.splice(ctrlIdx + 1, 0, "shift");
    }
    return [].concat(mods, [k]).join("+");
  }

  function isTypingTarget(t) {
    if (!t) return false;
    const tag = t.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
    if (t.isContentEditable) return true;
    return false;
  }

  function onKeyDown(ev) {
    // While the rebind capture is active, we swallow the event and
    // record it as the new combo for the active rebind target.
    if (rebindTarget) {
      const norm = comboFromEvent(ev);
      // Don't register a bare modifier keypress.
      const last = norm.split("+").pop();
      if (last === "control" || last === "shift" || last === "alt" || last === "meta") return;
      ev.preventDefault();
      ev.stopPropagation();
      rebind(rebindTarget, norm);
      finishRebind();
      return;
    }

    const combo = comboFromEvent(ev);
    const arr = comboToAction.get(combo);
    if (!arr || !arr.length) return;
    const typing = isTypingTarget(ev.target);
    for (const actionId of arr) {
      const def = actions.get(actionId);
      if (!def) continue;
      if (typing && !def.allowInInput) continue;
      if (def.scope !== "global" && def.scope !== activeScope) continue;
      try {
        const r = def.callback(ev);
        if (r !== false) {
          ev.preventDefault();
          ev.stopPropagation();
        }
      } catch (e) {
        console.error("[hotkeys] '" + actionId + "' threw:", e);
      }
      // First match wins to avoid double-firing.
      return;
    }
  }

  // --------- help overlay -----------------------------------------
  let overlayEl = null;
  let rebindTarget = null;

  function openHelp() {
    if (overlayEl) return;
    overlayEl = document.createElement("div");
    overlayEl.id = "psoHkOverlay";
    overlayEl.className = "hk-overlay";
    overlayEl.innerHTML =
      '<div class="hk-card" role="dialog" aria-modal="true" aria-labelledby="psoHkTitle">' +
        '<header>' +
          '<strong id="psoHkTitle">Keyboard shortcuts</strong>' +
          '<span class="grow"></span>' +
          '<button type="button" id="psoHkResetAll" class="ghost" title="reset every binding to default">reset all</button>' +
          '<button type="button" id="psoHkClose" class="ghost" title="close (Esc)">close</button>' +
        '</header>' +
        '<div class="hk-body" id="psoHkBody"></div>' +
        '<div class="hk-foot dim">click any combo to rebind. Esc to cancel a rebind.</div>' +
      '</div>';
    document.body.appendChild(overlayEl);
    overlayEl.addEventListener("click", function (e) {
      if (e.target === overlayEl) closeHelp();
    });
    overlayEl.querySelector("#psoHkClose").addEventListener("click", closeHelp);
    overlayEl.querySelector("#psoHkResetAll").addEventListener("click", function () {
      overrides = {};
      persist();
      rebuildReverse();
      renderHelp();
    });
    renderHelp();
  }

  function closeHelp() {
    if (rebindTarget) finishRebind();
    if (overlayEl && overlayEl.parentNode) overlayEl.parentNode.removeChild(overlayEl);
    overlayEl = null;
  }

  function renderHelp() {
    if (!overlayEl) return;
    const body = overlayEl.querySelector("#psoHkBody");
    if (!body) return;
    const groups = new Map();
    for (const b of bindings()) {
      const g = b.scope || "global";
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g).push(b);
    }
    const order = ["global", "paint", "sculpt", "mob_dsl", "motion_picker"];
    const allGroups = order.filter(function (g) { return groups.has(g); });
    for (const g of groups.keys()) {
      if (allGroups.indexOf(g) < 0) allGroups.push(g);
    }
    const parts = [];
    for (const g of allGroups) {
      parts.push('<div class="hk-group">');
      parts.push('<div class="hk-group-title">' + escapeHtml(g) + '</div>');
      parts.push('<table class="hk-table">');
      for (const b of groups.get(g)) {
        const isOverride = !!overrides[b.actionId];
        parts.push('<tr>');
        parts.push('<td class="hk-label">' + escapeHtml(b.label) + '</td>');
        parts.push('<td class="hk-combo"><button type="button" class="hk-combo-btn ' +
                   (isOverride ? "hk-overridden" : "") +
                   '" data-action="' + escapeHtml(b.actionId) + '" title="click to rebind">' +
                   escapeHtml(b.combo) + '</button>' +
                   (isOverride ? ' <button type="button" class="hk-reset-btn ghost" data-reset="' +
                                 escapeHtml(b.actionId) + '" title="reset to default (' +
                                 escapeHtml(b.defaultCombo) + ')">↺</button>' : "") +
                   '</td>');
        parts.push('</tr>');
      }
      parts.push('</table>');
      parts.push('</div>');
    }
    body.innerHTML = parts.join("");
    body.querySelectorAll(".hk-combo-btn").forEach(function (btn) {
      btn.addEventListener("click", function () { startRebind(btn.dataset.action); });
    });
    body.querySelectorAll(".hk-reset-btn").forEach(function (btn) {
      btn.addEventListener("click", function (e) {
        e.stopPropagation();
        reset(btn.dataset.reset);
        renderHelp();
      });
    });
  }

  function startRebind(actionId) {
    rebindTarget = actionId;
    if (!overlayEl) return;
    const btn = overlayEl.querySelector('.hk-combo-btn[data-action="' + cssEscape(actionId) + '"]');
    if (btn) {
      btn.classList.add("hk-rebinding");
      btn.textContent = "press a key…";
    }
  }

  function finishRebind() {
    rebindTarget = null;
    renderHelp();
  }

  function cssEscape(s) {
    return String(s).replace(/[^a-zA-Z0-9_-]/g, "\\$&");
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;",
                '"': "&quot;", "'": "&#39;" })[c];
    });
  }

  // --------- defaults ---------------------------------------------
  // Bound during init() so consumers can override before the listener
  // attaches if they're loaded before us.
  function installDefaults() {
    bind("?", "open-help-overlay", function () { openHelp(); }, { label: "Show this help overlay" });
    bind("Ctrl+P", "open-quick-search", function () {
      if (window.psoQuickSearch && window.psoQuickSearch.open) window.psoQuickSearch.open();
    }, { label: "Quick search assets" });
    bind("Ctrl+Z", "undo", function () {
      if (window.psoUndoBus && window.psoUndoBus.undo) window.psoUndoBus.undo();
    }, { label: "Undo" });
    bind("Ctrl+Shift+Z", "redo", function () {
      if (window.psoUndoBus && window.psoUndoBus.redo) window.psoUndoBus.redo();
    }, { label: "Redo" });
    bind("Ctrl+Y", "redo-y", function () {
      if (window.psoUndoBus && window.psoUndoBus.redo) window.psoUndoBus.redo();
    }, { label: "Redo (alt)" });
    bind("Ctrl+S", "save-workspace", function () {
      if (window.psoWorkspace && window.psoWorkspace.saveLocal) window.psoWorkspace.saveLocal();
      // Also surface a toast so the user knows it worked.
      if (window.psoToast) window.psoToast("workspace auto-saved");
    }, { label: "Save workspace (local)" });
    bind("Tab", "cycle-perspective", function (ev) {
      if (!window.PSOPerspectives) return false;
      const ctx = window.PSOPerspectives.activeContext();
      if (!ctx) return false;
      const cands = window.PSOPerspectives.list(ctx.entry, ctx.fileName)
        .filter(function (c) { return c.score > 0; });
      if (!cands.length) return false;
      cands.sort(function (a, b) { return b.score - a.score; });
      const cur = window.PSOPerspectives.active();
      const idx = cands.findIndex(function (c) { return c.name === cur; });
      const dir = ev && ev.shiftKey ? -1 : 1;
      const next = cands[(idx + dir + cands.length) % cands.length];
      if (next) window.PSOPerspectives.switchTo(next.name, ctx);
    }, { label: "Cycle perspective tabs" });
    bind("Escape", "close-overlays", function () {
      // The undo_bus / quicksearch overlay each handle their own Esc.
      // We only handle the help overlay here.
      if (overlayEl) { closeHelp(); return; }
      // Let the existing modal handlers catch other Escs by NOT
      // consuming the event. Returning false signals the dispatch loop
      // to skip preventDefault().
      return false;
    }, { label: "Close help overlay / dialogs" });
    bind("Space", "motion-play-pause", function (ev) {
      // Only fire if the model viewer's animation panel is mounted
      // and showing — fall back to false to let the normal scroll work.
      const btn = document.getElementById("modelAnimPlayPause");
      if (btn && btn.offsetParent !== null) {
        btn.click();
        return;
      }
      return false;
    }, { label: "Play/pause active motion" });
  }

  function init() {
    installDefaults();
    document.addEventListener("keydown", onKeyDown, true);  // capture so we beat panel handlers
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // --------- public API -------------------------------------------
  window.psoHotkeys = Object.freeze({
    bind: bind,
    bindings: bindings,
    rebind: rebind,
    reset: reset,
    setActiveScope: setActiveScope,
    openHelp: openHelp,
    closeHelp: closeHelp,
    normalize: normalize,
  });
})();
