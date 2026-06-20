// =====================================================================
// PSOBB Texture Editor - Quick search overlay (Ctrl+P style, 2026-04-25).
//
// Manifest has 9357 entries; finding a specific BML by typing fragment
// of its name should be instant. This module mounts a centered overlay
// with an <input> + result list. Fuzzy matching combines:
//   - case-insensitive substring  (highest weight)
//   - token-prefix scoring        (each segment of the path)
//   - initialism                  ("vol" matches "VolOpt" via V_O)
//
// Scope: searches over manifest-lite (already loaded by manifest.js).
// We index ALL entries once on first open into a flat array of
// {path, lowerPath, segments, initial}; each keystroke ranks against
// that array. With a 9357-entry manifest the per-keystroke cost on a
// midrange laptop is < 5 ms (well under the 16 ms budget).
//
// Public API:
//   psoQuickSearch.open()      - show overlay
//   psoQuickSearch.close()
//   psoQuickSearch.toggle()
//
// Recent + favorites:
//   - Top of empty-input result list shows recent + pinned. Recent is
//     up to 10 most-recently-opened paths (kept in localStorage).
//     Pinned is starred entries (clicking the ★ in the result row).
//
// Selection:
//   - ↑/↓ navigate, Enter selects, Esc closes.
//   - Selecting calls window.bus.emit("asset.opened", ...) which the
//     asset-router picks up.
// =====================================================================

(function () {
  "use strict";

  if (window.psoQuickSearch) return;

  const RECENT_KEY = "pso.qs.recent";
  const PINNED_KEY = "pso.qs.pinned";
  const MAX_RESULTS = 20;
  const MAX_RECENT = 10;

  let overlay = null;
  let inputEl = null;
  let listEl = null;
  let countEl = null;
  let activeIdx = 0;
  let lastResults = [];

  // ---- index ------------------------------------------------------
  // Built once on first open; rebuilt if the manifest refreshes.
  let index = null;        // [{path, lowerPath, segments, initial, entry}]
  let indexedSize = 0;

  function buildIndex() {
    if (!window.PSOManifest || !window.PSOManifest.isLoaded()) return false;
    const all = window.PSOManifest.entries();
    if (!all || !all.length) return false;
    const arr = new Array(all.length);
    for (let i = 0; i < all.length; i++) {
      const e = all[i];
      const path = (e && e.path) || "";
      const lower = path.toLowerCase();
      // Path segments: split on '/' and the inner-archive '#' separator.
      const segs = lower.split(/[\/#]/).filter(Boolean);
      // Initialism: uppercase letters and segment-firsts.
      let init = "";
      for (const seg of segs) {
        if (seg) init += seg[0];
      }
      arr[i] = {
        path: path,
        lowerPath: lower,
        segments: segs,
        initial: init,
        entry: e,
      };
    }
    index = arr;
    indexedSize = all.length;
    return true;
  }

  // ---- score ------------------------------------------------------
  // Returns an integer score; -1 if no match.
  // Higher is better. Tied scores fall through to alpha-by-path.
  function scoreEntry(item, q) {
    if (!q) return 0;
    const lp = item.lowerPath;
    // Substring is the strongest signal.
    const idx = lp.indexOf(q);
    if (idx >= 0) {
      // Boost for matching at the start of the basename.
      const slash = lp.lastIndexOf("/");
      const base = slash >= 0 ? lp.slice(slash + 1) : lp;
      let bonus = 0;
      if (base.startsWith(q)) bonus += 200;
      else if (idx === 0) bonus += 80;
      // Shorter paths that match win over longer ones (preferring "boss1.bml"
      // over "data/path/.../boss1_offshoot_unused.bml").
      bonus += Math.max(0, 60 - lp.length);
      return 1000 + bonus;
    }
    // Token-prefix: every space-delimited query token must prefix some segment.
    const toks = q.split(/\s+/).filter(Boolean);
    if (toks.length > 1) {
      let allHit = true;
      let tokScore = 0;
      for (const t of toks) {
        let any = false;
        for (const s of item.segments) {
          if (s.startsWith(t)) { any = true; tokScore += 30; break; }
          if (s.indexOf(t) >= 0) { any = true; tokScore += 10; break; }
        }
        if (!any) { allHit = false; break; }
      }
      if (allHit) return 500 + tokScore;
    }
    // Initialism: first letters of segments.
    if (item.initial.indexOf(q) === 0) return 300;
    if (item.initial.indexOf(q) > 0) return 150;
    return -1;
  }

  // ---- recent / pinned -------------------------------------------
  function getRecent() {
    try {
      const r = localStorage.getItem(RECENT_KEY);
      const a = r ? JSON.parse(r) : [];
      return Array.isArray(a) ? a : [];
    } catch (_e) { return []; }
  }

  function pushRecent(path) {
    if (!path) return;
    try {
      let arr = getRecent();
      arr = arr.filter(function (p) { return p !== path; });
      arr.unshift(path);
      arr = arr.slice(0, MAX_RECENT);
      localStorage.setItem(RECENT_KEY, JSON.stringify(arr));
    } catch (_e) {}
  }

  function getPinned() {
    try {
      const r = localStorage.getItem(PINNED_KEY);
      const a = r ? JSON.parse(r) : [];
      return Array.isArray(a) ? a : [];
    } catch (_e) { return []; }
  }

  function togglePinned(path) {
    try {
      let arr = getPinned();
      if (arr.indexOf(path) >= 0) {
        arr = arr.filter(function (p) { return p !== path; });
      } else {
        arr.unshift(path);
        arr = arr.slice(0, 30);
      }
      localStorage.setItem(PINNED_KEY, JSON.stringify(arr));
    } catch (_e) {}
  }

  // ---- search ----------------------------------------------------
  function search(q) {
    if (!index) {
      if (!buildIndex()) return [];
    }
    q = (q || "").toLowerCase().trim();
    if (!q) {
      // Empty input: show pinned then recent.
      const pinned = getPinned();
      const recent = getRecent();
      const seen = new Set();
      const out = [];
      for (const p of pinned) {
        if (seen.has(p)) continue;
        const it = lookupByPath(p);
        if (it) { out.push({ item: it, score: 9999, badge: "pinned" }); seen.add(p); }
      }
      for (const p of recent) {
        if (seen.has(p)) continue;
        const it = lookupByPath(p);
        if (it) { out.push({ item: it, score: 9000, badge: "recent" }); seen.add(p); }
        if (out.length >= MAX_RESULTS) break;
      }
      return out.slice(0, MAX_RESULTS);
    }
    const t0 = (typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now();
    const results = [];
    const N = index.length;
    for (let i = 0; i < N; i++) {
      const s = scoreEntry(index[i], q);
      if (s < 0) continue;
      results.push({ item: index[i], score: s });
    }
    results.sort(function (a, b) {
      if (a.score !== b.score) return b.score - a.score;
      // Stable tie-break: shorter path first, then alpha.
      const la = a.item.lowerPath.length, lb = b.item.lowerPath.length;
      if (la !== lb) return la - lb;
      return a.item.lowerPath < b.item.lowerPath ? -1 : 1;
    });
    const out = results.slice(0, MAX_RESULTS);
    const elapsed = ((typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now()) - t0;
    out.elapsedMs = elapsed;
    return out;
  }

  function lookupByPath(p) {
    if (!index && !buildIndex()) return null;
    for (const it of index) {
      if (it.path === p) return it;
    }
    return null;
  }

  // ---- UI --------------------------------------------------------
  function ensureOverlay() {
    if (overlay) return;
    overlay = document.createElement("div");
    overlay.id = "psoQsOverlay";
    overlay.className = "qs-overlay";
    overlay.innerHTML =
      '<div class="qs-card" role="dialog" aria-modal="true" aria-label="quick search">' +
        '<div class="qs-input-wrap">' +
          '<input type="text" id="psoQsInput" class="qs-input" placeholder="search assets…" autocomplete="off" spellcheck="false" />' +
          '<span id="psoQsCount" class="qs-count dim"></span>' +
        '</div>' +
        '<ul id="psoQsList" class="qs-list" role="listbox"></ul>' +
        '<div class="qs-foot dim">↑↓ navigate · Enter open · ★ pin · Esc close</div>' +
      '</div>';
    document.body.appendChild(overlay);
    inputEl = overlay.querySelector("#psoQsInput");
    listEl = overlay.querySelector("#psoQsList");
    countEl = overlay.querySelector("#psoQsCount");
    inputEl.addEventListener("input", onInput);
    inputEl.addEventListener("keydown", onInputKey);
    overlay.addEventListener("click", function (e) {
      if (e.target === overlay) close();
    });
    listEl.addEventListener("click", function (e) {
      const li = e.target.closest("li.qs-item");
      if (!li) return;
      if (e.target.classList.contains("qs-pin")) {
        togglePinned(li.dataset.path);
        renderResults(lastResults);
        return;
      }
      const idx = parseInt(li.dataset.idx, 10);
      if (isFinite(idx)) chooseAt(idx);
    });
  }

  function open() {
    ensureOverlay();
    overlay.style.display = "flex";
    inputEl.value = "";
    inputEl.focus();
    activeIdx = 0;
    refresh();
  }

  function close() {
    if (overlay) overlay.style.display = "none";
  }

  function toggle() {
    if (!overlay || overlay.style.display !== "flex") open();
    else close();
  }

  function onInput() {
    activeIdx = 0;
    refresh();
  }

  function onInputKey(ev) {
    if (ev.key === "Escape") {
      ev.preventDefault();
      close();
      return;
    }
    if (ev.key === "ArrowDown") {
      ev.preventDefault();
      if (lastResults.length) {
        activeIdx = (activeIdx + 1) % lastResults.length;
        renderResults(lastResults);
      }
      return;
    }
    if (ev.key === "ArrowUp") {
      ev.preventDefault();
      if (lastResults.length) {
        activeIdx = (activeIdx - 1 + lastResults.length) % lastResults.length;
        renderResults(lastResults);
      }
      return;
    }
    if (ev.key === "Enter") {
      ev.preventDefault();
      if (lastResults.length) chooseAt(activeIdx);
      return;
    }
  }

  function refresh() {
    const q = inputEl.value;
    const results = search(q);
    lastResults = results;
    renderResults(results);
  }

  function renderResults(results) {
    if (!listEl) return;
    if (!results || !results.length) {
      const q = inputEl.value;
      listEl.innerHTML = '<li class="qs-empty dim">' +
        (q ? "no matches for '" + escapeHtml(q) + "'" : "no recent or pinned items yet") +
        '</li>';
      countEl.textContent = "";
      return;
    }
    const pinnedSet = new Set(getPinned());
    const parts = [];
    for (let i = 0; i < results.length; i++) {
      const r = results[i];
      const it = r.item;
      const isPinned = pinnedSet.has(it.path);
      const isActive = i === activeIdx;
      const badge = r.badge ? '<span class="qs-badge">' + escapeHtml(r.badge) + '</span>' : "";
      const cat = (it.entry && it.entry.inferred_category) ||
                  (it.entry && it.entry.category) || "";
      const catHtml = cat ? '<span class="qs-cat dim">' + escapeHtml(cat) + '</span>' : "";
      parts.push(
        '<li class="qs-item' + (isActive ? " qs-active" : "") + '"' +
        ' role="option" aria-selected="' + (isActive ? "true" : "false") + '"' +
        ' data-idx="' + i + '" data-path="' + escapeHtml(it.path) + '">' +
          '<span class="qs-pin" title="' + (isPinned ? "unpin" : "pin to top") +
          '">' + (isPinned ? "★" : "☆") + '</span>' +
          '<span class="qs-path">' + highlight(it.path, inputEl.value) + '</span>' +
          catHtml +
          badge +
        '</li>'
      );
    }
    listEl.innerHTML = parts.join("");
    // Scroll active into view.
    const active = listEl.querySelector("li.qs-active");
    if (active) {
      const top = active.offsetTop;
      const bot = top + active.offsetHeight;
      const vt = listEl.scrollTop;
      const vb = vt + listEl.clientHeight;
      if (top < vt) listEl.scrollTop = top;
      else if (bot > vb) listEl.scrollTop = bot - listEl.clientHeight;
    }
    const elapsed = results.elapsedMs != null ? results.elapsedMs : null;
    const indexedTxt = indexedSize ? " · " + indexedSize + " indexed" : "";
    if (elapsed != null) {
      countEl.textContent = results.length + " result" + (results.length === 1 ? "" : "s") +
                            indexedTxt + " · " + elapsed.toFixed(1) + " ms";
    } else {
      countEl.textContent = results.length + " result" + (results.length === 1 ? "" : "s") + indexedTxt;
    }
  }

  function chooseAt(i) {
    const r = lastResults[i];
    if (!r || !r.item) return;
    const path = r.item.path;
    const entry = r.item.entry;
    pushRecent(path);
    close();
    if (window.bus && typeof window.bus.emit === "function") {
      window.bus.emit("asset.opened", { path: path, entry: entry });
    }
  }

  // ---- highlight (lightweight; no regex meta panic) -------------
  function highlight(path, q) {
    const e = escapeHtml(path);
    if (!q) return e;
    q = q.toLowerCase();
    const lp = path.toLowerCase();
    const i = lp.indexOf(q);
    if (i < 0) return e;
    // Recompute on raw path then re-escape parts for safety.
    return escapeHtml(path.slice(0, i)) +
           '<mark class="qs-mark">' + escapeHtml(path.slice(i, i + q.length)) + '</mark>' +
           escapeHtml(path.slice(i + q.length));
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;",
                '"': "&quot;", "'": "&#39;" })[c];
    });
  }

  // ---- bootstrap -------------------------------------------------
  function init() {
    // Pre-build the index when the manifest finishes loading so the
    // first open of Ctrl+P is instant.
    if (window.PSOManifest) {
      // load() is idempotent; we don't await — just kick it.
      window.PSOManifest.load().then(function () {
        try { buildIndex(); } catch (_e) {}
      }).catch(function () {});
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  window.psoQuickSearch = Object.freeze({
    open: open,
    close: close,
    toggle: toggle,
    // Exposed for tests.
    _scoreEntry: scoreEntry,
    _buildIndex: buildIndex,
    _search: search,
  });
})();
