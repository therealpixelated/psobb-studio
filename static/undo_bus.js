// =====================================================================
// PSOBB Texture Editor - Cross-tool undo/redo bus (2026-04-25).
//
// Each authoring panel keeps its own internal stack (paint canvas
// snapshots, sculpt vertex deltas, mob_dsl patch dicts, etc.). The bus
// sits ABOVE those stacks: every undoable action ALSO pushes a
// {label, undo, redo, panelId} record onto a single shared deque.
// Ctrl+Z / Ctrl+Shift+Z always pop / re-apply the bus's tip, regardless
// of which perspective is mounted right now. This means a user can
// paint a stroke, switch to the mob editor, change Booma's walk_speed,
// and Ctrl+Z one-then-the-other without losing history.
//
// Public API:
//   psoUndoBus.push({label, undo, redo, panelId})
//     - label is a short human string for the status indicator
//     - undo()/redo() are sync or async fns owned by the panel
//     - panelId is "paint" / "mob_dsl" / "sculpt" / etc.
//     - Returns the entry's id (monotonic int) for callers that care
//
//   psoUndoBus.undo()       - apply tip's undo, push onto redo deque
//   psoUndoBus.redo()       - apply tip of redo deque, push back onto undo
//   psoUndoBus.peek()       - { label, panelId, ts } of the most recent entry
//   psoUndoBus.history()    - [{label, panelId, ts, id}, ...] newest-first
//   psoUndoBus.clear()      - drop everything (e.g. on workspace switch)
//
// Events on window.bus:
//   undo.pushed   { id, label, panelId }
//   undo.applied  { id, label, panelId, direction: "undo"|"redo" }
//   undo.cleared
//
// FIFO eviction at MAX_ENTRIES keeps memory bounded; the deque holds
// closures, not snapshots, so per-entry RAM is tiny. The PANEL is
// responsible for capturing the actual state into its closure.
//
// Idempotent on multiple loads.
// =====================================================================

(function () {
  "use strict";

  if (window.psoUndoBus) return;

  const MAX_ENTRIES = 200;

  // Two deques: undoDeque is "things we can undo" (newest at end), and
  // redoDeque is "things we can redo" (newest at end). On any new push,
  // we clear redoDeque — the user has branched.
  const undoDeque = [];
  const redoDeque = [];

  let nextId = 1;

  function emit(channel, payload) {
    try {
      if (window.bus && typeof window.bus.emit === "function") {
        window.bus.emit(channel, payload);
      }
    } catch (_e) {}
  }

  function push(entry) {
    if (!entry || typeof entry !== "object") return null;
    const undoFn = entry.undo;
    const redoFn = entry.redo;
    if (typeof undoFn !== "function" || typeof redoFn !== "function") {
      console.warn("[undo_bus] push() needs {undo, redo} as functions");
      return null;
    }
    const rec = {
      id: nextId++,
      label: String(entry.label || "(unnamed action)"),
      panelId: String(entry.panelId || "unknown"),
      undo: undoFn,
      redo: redoFn,
      ts: Date.now(),
    };
    undoDeque.push(rec);
    while (undoDeque.length > MAX_ENTRIES) undoDeque.shift();
    // Branching: any new action invalidates the redo chain.
    redoDeque.length = 0;
    emit("undo.pushed", { id: rec.id, label: rec.label, panelId: rec.panelId });
    return rec.id;
  }

  async function undo() {
    if (!undoDeque.length) return false;
    const rec = undoDeque.pop();
    try {
      const r = rec.undo();
      if (r && typeof r.then === "function") await r;
    } catch (e) {
      console.error("[undo_bus] undo for '" + rec.label + "' threw:", e);
      // Still treat as consumed so we don't loop on a poison entry.
    }
    redoDeque.push(rec);
    while (redoDeque.length > MAX_ENTRIES) redoDeque.shift();
    emit("undo.applied", { id: rec.id, label: rec.label, panelId: rec.panelId, direction: "undo" });
    return true;
  }

  async function redo() {
    if (!redoDeque.length) return false;
    const rec = redoDeque.pop();
    try {
      const r = rec.redo();
      if (r && typeof r.then === "function") await r;
    } catch (e) {
      console.error("[undo_bus] redo for '" + rec.label + "' threw:", e);
    }
    undoDeque.push(rec);
    while (undoDeque.length > MAX_ENTRIES) undoDeque.shift();
    emit("undo.applied", { id: rec.id, label: rec.label, panelId: rec.panelId, direction: "redo" });
    return true;
  }

  function peek() {
    if (!undoDeque.length) return null;
    const rec = undoDeque[undoDeque.length - 1];
    return { id: rec.id, label: rec.label, panelId: rec.panelId, ts: rec.ts };
  }

  function history() {
    // newest-first, summary only (no fn handles leaked).
    const out = [];
    for (let i = undoDeque.length - 1; i >= 0; i--) {
      const r = undoDeque[i];
      out.push({ id: r.id, label: r.label, panelId: r.panelId, ts: r.ts });
    }
    return out;
  }

  function clear() {
    undoDeque.length = 0;
    redoDeque.length = 0;
    emit("undo.cleared", {});
  }

  function size() {
    return { undo: undoDeque.length, redo: redoDeque.length };
  }

  // ---- status indicator + history dropdown -------------------------
  // We render a tiny pill near the perspective tab strip showing the
  // most recent action's label + panelId, plus a clickable button that
  // opens a flyout with the deque history. Toggling on click; closes on
  // outside click or Esc.
  let statusEl = null;
  let listEl = null;
  let listOpen = false;

  function ensureStatusUI() {
    if (statusEl && document.body.contains(statusEl)) return;
    const tabs = document.getElementById("vpTabs");
    const host = tabs ? tabs.parentElement : document.body;
    if (!host) return;
    statusEl = document.createElement("div");
    statusEl.className = "ub-status";
    statusEl.id = "psoUbStatus";
    statusEl.innerHTML =
      '<button type="button" class="ub-pill" id="psoUbPill" title="last action (click for history)" hidden>' +
      '<span class="ub-pill-label">no actions</span>' +
      '<span class="ub-pill-caret">▾</span>' +
      '</button>';
    // Insert after the tab strip rather than inside it (lets CSS float).
    if (tabs && tabs.parentElement) {
      tabs.parentElement.insertBefore(statusEl, tabs.nextSibling);
    } else {
      document.body.appendChild(statusEl);
    }
    statusEl.querySelector("#psoUbPill").addEventListener("click", function (e) {
      e.stopPropagation();
      toggleList();
    });
    document.addEventListener("click", function (e) {
      if (!listOpen) return;
      if (statusEl && statusEl.contains(e.target)) return;
      closeList();
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && listOpen) closeList();
    });
  }

  function refreshStatus() {
    if (!statusEl) return;
    const pill = statusEl.querySelector("#psoUbPill");
    if (!pill) return;
    const tip = peek();
    if (!tip) {
      pill.hidden = true;
      return;
    }
    pill.hidden = false;
    const labelEl = pill.querySelector(".ub-pill-label");
    if (labelEl) labelEl.textContent = "Last: " + tip.label + " (" + tip.panelId + ")";
  }

  function toggleList() {
    if (listOpen) closeList();
    else openList();
  }

  function openList() {
    if (!statusEl) return;
    closeList();
    const items = history();
    listEl = document.createElement("div");
    listEl.className = "ub-history";
    if (!items.length) {
      listEl.innerHTML = '<div class="ub-history-empty dim">no history</div>';
    } else {
      const parts = ['<div class="ub-history-title">History (newest first)</div>',
                     '<ul class="ub-history-list">'];
      const cap = Math.min(items.length, 30);
      for (let i = 0; i < cap; i++) {
        const it = items[i];
        const ago = humanAgo(it.ts);
        parts.push(
          '<li class="ub-history-item" data-id="' + it.id + '">' +
          '<span class="ub-history-panel">' + escapeHtml(it.panelId) + '</span>' +
          '<span class="ub-history-label">' + escapeHtml(it.label) + '</span>' +
          '<span class="ub-history-ago dim">' + escapeHtml(ago) + '</span>' +
          '</li>'
        );
      }
      parts.push('</ul>');
      if (items.length > cap) {
        parts.push('<div class="ub-history-more dim">+' + (items.length - cap) + ' more (clear with workspace switch)</div>');
      }
      listEl.innerHTML = parts.join("");
    }
    statusEl.appendChild(listEl);
    listOpen = true;
  }

  function closeList() {
    if (listEl && listEl.parentNode) {
      listEl.parentNode.removeChild(listEl);
    }
    listEl = null;
    listOpen = false;
  }

  function humanAgo(ts) {
    const d = Math.max(0, Date.now() - ts);
    const s = Math.round(d / 1000);
    if (s < 5) return "just now";
    if (s < 60) return s + "s ago";
    const m = Math.round(s / 60);
    if (m < 60) return m + "m ago";
    const h = Math.round(m / 60);
    return h + "h ago";
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;",
                '"': "&quot;", "'": "&#39;" })[c];
    });
  }

  // Wire status refresh to bus events.
  function wireStatus() {
    ensureStatusUI();
    refreshStatus();
    if (window.bus) {
      window.bus.on("undo.pushed", function () { refreshStatus(); });
      window.bus.on("undo.applied", function () { refreshStatus(); if (listOpen) openList(); });
      window.bus.on("undo.cleared", function () { refreshStatus(); closeList(); });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", wireStatus);
  } else {
    wireStatus();
  }

  // ---- global Ctrl+Z / Ctrl+Shift+Z handler -------------------------
  // We bind in capturing phase at the document level so we run before
  // the per-panel handlers (paint_panel.js etc. each install their own
  // keydown listener). Per-panel handlers are still useful as a
  // fallback when the panel is mounted but the bus is empty for that
  // panel — they call THEIR internal stack. Net effect: Ctrl+Z always
  // does something sensible.
  function isTypingTarget(t) {
    if (!t) return false;
    const tag = t.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
    if (t.isContentEditable) return true;
    return false;
  }

  document.addEventListener("keydown", function (ev) {
    // Don't fight the user's text-area Ctrl+Z (browser-native form undo).
    if (isTypingTarget(ev.target)) return;
    if (!(ev.ctrlKey || ev.metaKey)) return;
    const k = (ev.key || "").toLowerCase();
    if (k !== "z" && k !== "y") return;
    // Ctrl+Y is a Windows-y "redo" alias; respect both.
    const wantRedo = (k === "y") || (k === "z" && ev.shiftKey);
    if (undoDeque.length === 0 && !wantRedo) return;
    if (wantRedo && redoDeque.length === 0) return;
    ev.preventDefault();
    ev.stopPropagation();
    if (wantRedo) redo();
    else undo();
  }, true);  // capture phase

  // ---- public API ---------------------------------------------------
  window.psoUndoBus = Object.freeze({
    push: push,
    undo: undo,
    redo: redo,
    peek: peek,
    history: history,
    clear: clear,
    size: size,
    MAX_ENTRIES: MAX_ENTRIES,
  });
})();
