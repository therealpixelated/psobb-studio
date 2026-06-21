// PSOBB Studio — first-run onboarding + re-openable walkthrough.
// =====================================================================
// Self-contained module. Exposes window.psoOnboarding with:
//   open()          — show the guided walkthrough overlay (step 1)
//   close()         — dismiss the overlay
//   maybeAutoShow() — auto-open ONCE on first run (localStorage-gated),
//                     called by app.js init() after loadFiles().
//   refreshDataDir(info) — repaint the empty-state data-dir callout +
//                     the header #dataDir pill from a {data_dir, exists,
//                     count} object. app.js calls this from loadFiles().
//
// Design constraints (per spec):
//   * The overlay is appended to <body>, NOT inside #vpStage, so
//     perspectives.js `stage.innerHTML = ""` can't wipe it.
//   * Status chip gates on data_dir.exists + asset count, NEVER on
//     health.ok (health.ok is false on a healthy box merely because the
//     optional legacy "puyo" tool is absent).
//   * No server changes: /api/health, /api/files, /api/manifest/categories
//     already expose everything needed.
// =====================================================================

(function () {
  "use strict";

  if (window.psoOnboarding) return; // idempotent

  var SEEN_KEY = "pso.onboarding.seen.v1";

  // ── walkthrough step content ─────────────────────────────────────
  // Each step optionally spotlights an element by selector; if the
  // element is missing/hidden the card just centers.
  var STEPS = [
    {
      title: "Pick a category",
      spotlight: "#assetTree",
      body: function () {
        return (
          "Every file in your install, sorted into categories — " +
          "textures, models, maps, quests, audio, UI and more. " +
          "Click a category in the <strong>All assets</strong> tree on the " +
          "left to expand it." +
          catListHtml()
        );
      },
    },
    {
      title: "Open a model or texture",
      spotlight: "#assetTree",
      body: function () {
        return (
          "Click any asset. <strong>Textures</strong> open as a tile grid; " +
          "<strong>models</strong> open in the 3D viewer; everything else gets " +
          "a sensible viewer (audio player, hex dump, JSON)."
        );
      },
    },
    {
      title: "Switch views with the tab strip",
      spotlight: "#vpTabs",
      body: function () {
        return (
          "The tabs above the stage swap perspectives — " +
          "<strong>Tile grid</strong>, <strong>3D view</strong>, " +
          "<strong>Viewport paint</strong> and more — with no popups. " +
          "The little boxed digit on each tab is its keyboard shortcut."
        );
      },
    },
    {
      title: "Edit & upscale",
      spotlight: "#workArea",
      body: function () {
        return (
          "Press <kbd>U</kbd> to AI-upscale every tile, or drag an external " +
          "Upscayl result onto a tile card. Models support paint, sculpt, rig " +
          "and UV tabs. The <code>model:</code> / <code>scale:</code> selectors " +
          "in the toolbar choose the upscaler."
        );
      },
    },
    {
      title: "Deploy to game",
      spotlight: "#btnDeployToGame",
      body: function () {
        return (
          "Press <kbd>R</kbd> to repack a container, or <strong>deploy to " +
          "game</strong> to copy changed files from the dev mirror back into " +
          "your live PSOBB.IO install. Studio always makes a timestamped " +
          "backup first."
        );
      },
    },
  ];

  var state = { idx: 0, overlay: null };
  var liveCategories = null; // [{name,count}] once fetched

  function $(sel) { return document.querySelector(sel); }

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return {
        "&": "&amp;", "<": "&lt;", ">": "&gt;",
        '"': "&quot;", "'": "&#39;",
      }[c];
    });
  }

  // The onboarding pills mirror the asset-tree FILTER pills (tree.js
  // TAB_FILTERS, minus the catch-all "All") so the walkthrough teaches the
  // exact same vocabulary the user then sees in the tree. Each filter pill
  // aggregates one or more manifest-enum categories.
  var FILTER_PILLS = [
    { label: "Models",   cats: ["model"] },
    { label: "Textures", cats: ["texture", "container"] },
    { label: "UI",       cats: ["ui"] },
    { label: "Audio",    cats: ["audio"] },
    { label: "Quests",   cats: ["quest"] },
    { label: "Floors",   cats: ["map"] },
  ];

  function catListHtml() {
    if (!liveCategories || !liveCategories.length) return "";
    var byName = {};
    liveCategories.forEach(function (c) {
      if (c && c.name) byName[c.name] = c.count || 0;
    });
    var chips = FILTER_PILLS.map(function (p) {
      var n = p.cats.reduce(function (a, cat) { return a + (byName[cat] || 0); }, 0);
      if (n <= 0) return "";
      return (
        '<span class="ob-cat">' + esc(p.label) +
        '<span class="ob-cat-n">' + n.toLocaleString() + "</span></span>"
      );
    }).filter(Boolean).join("");
    if (!chips) return "";
    return '<div class="ob-cat-list">' + chips + "</div>";
  }

  // ── overlay rendering ────────────────────────────────────────────
  function ensureOverlay() {
    if (state.overlay) return state.overlay;
    var ov = document.createElement("div");
    ov.className = "ob-overlay";
    ov.id = "obOverlay";
    ov.hidden = true;
    ov.setAttribute("role", "dialog");
    ov.setAttribute("aria-modal", "true");
    ov.setAttribute("aria-label", "PSOBB Studio walkthrough");
    ov.innerHTML =
      '<div class="ob-card" role="document">' +
      '  <div class="ob-card-spot" id="obSpot">1</div>' +
      '  <button class="ob-card-close" id="obClose" type="button" aria-label="close walkthrough">&times;</button>' +
      '  <div class="ob-card-title" id="obTitle"></div>' +
      '  <div class="ob-card-body" id="obBody"></div>' +
      '  <div class="ob-card-foot">' +
      '    <label class="ob-noshow" style="margin-right:auto;display:flex;align-items:center;gap:6px;font-size:12px;color:var(--tk-text-mute,#9aa7b4);cursor:pointer">' +
      '      <input type="checkbox" id="obNoShow" checked> don\'t show this again</label>' +
      '    <div class="ob-dots" id="obDots"></div>' +
      '    <button class="ghost" id="obPrev" type="button">back</button>' +
      '    <button id="obNext" type="button">next</button>' +
      "  </div>" +
      "</div>";
    document.body.appendChild(ov);
    state.overlay = ov;

    ov.querySelector("#obClose").addEventListener("click", close);
    ov.querySelector("#obPrev").addEventListener("click", function () { go(-1); });
    ov.querySelector("#obNext").addEventListener("click", function () { go(1); });
    var noShow = ov.querySelector("#obNoShow");
    if (noShow) noShow.addEventListener("change", function () {
      if (noShow.checked) markSeen(); else clearSeen();
    });
    // Click on the backdrop (outside the card) dismisses.
    ov.addEventListener("mousedown", function (e) {
      if (e.target === ov) close();
    });
    // Esc closes while the overlay is open.
    document.addEventListener("keydown", function (e) {
      if (state.overlay && !state.overlay.hidden && e.key === "Escape") {
        e.stopPropagation();
        close();
      }
    });
    return ov;
  }

  function renderStep() {
    var ov = ensureOverlay();
    var step = STEPS[state.idx];
    ov.querySelector("#obSpot").textContent = String(state.idx + 1);
    ov.querySelector("#obTitle").textContent = step.title;
    ov.querySelector("#obBody").innerHTML =
      typeof step.body === "function" ? step.body() : step.body;

    var dots = ov.querySelector("#obDots");
    dots.innerHTML = "";
    for (var i = 0; i < STEPS.length; i++) {
      var d = document.createElement("span");
      d.className = "ob-dot" + (i === state.idx ? " active" : "");
      dots.appendChild(d);
    }

    var prev = ov.querySelector("#obPrev");
    var next = ov.querySelector("#obNext");
    prev.disabled = state.idx === 0;
    next.textContent = state.idx === STEPS.length - 1 ? "done" : "next";

    // Soft spotlight: briefly outline the highlighted element if visible.
    clearSpotlight();
    try {
      var el = step.spotlight ? $(step.spotlight) : null;
      if (el && el.offsetParent !== null) {
        el.classList.add("ob-spotlighted");
        el.style.boxShadow = "0 0 0 2px var(--accent-2, #4da3ff)";
        el.style.transition = "box-shadow 0.2s";
        state._spot = el;
      }
    } catch (_e) { /* non-fatal */ }
  }

  function clearSpotlight() {
    if (state._spot) {
      try {
        state._spot.classList.remove("ob-spotlighted");
        state._spot.style.boxShadow = "";
      } catch (_e) {}
      state._spot = null;
    }
  }

  function go(delta) {
    var n = state.idx + delta;
    if (n < 0) return;
    if (n >= STEPS.length) { finish(); return; }
    state.idx = n;
    renderStep();
  }

  function open() {
    ensureCategories();
    state.idx = 0;
    var ov = ensureOverlay();
    ov.hidden = false;
    renderStep();
  }

  function close() {
    if (state.overlay) state.overlay.hidden = true;
    clearSpotlight();
    markSeen();
  }

  function finish() {
    close();
    try {
      if (window.psoEditor && window.psoEditor.toast) {
        window.psoEditor.toast("you're all set — pick an asset on the left", "ok");
      }
    } catch (_e) {}
  }

  function markSeen() {
    try { localStorage.setItem(SEEN_KEY, "1"); } catch (_e) {}
  }
  function clearSeen() {
    try { localStorage.removeItem(SEEN_KEY); } catch (_e) {}
  }

  function maybeAutoShow() {
    // Auto-show DISABLED (owner request 2026-06-20). The first-run tour was
    // flashing on every refresh — aggressive static-JS caching meant the
    // updated once-gate never ran, and per-port localStorage made it
    // unreliable anyway. The owner doesn't want it auto-rendering at all.
    // The walkthrough is still available ON DEMAND via open(): the header
    // "?" button and the "Start the walkthrough" button on the empty-state
    // welcome screen. We mark it seen so nothing ever auto-triggers.
    markSeen();
  }

  // ── data-dir callout / header pill ───────────────────────────────
  function refreshDataDir(info) {
    info = info || {};
    var path = info.data_dir || "";
    var count = (typeof info.count === "number") ? info.count : null;
    // `exists` defaults to true when a count is known and >0; the health
    // probe (below) provides an authoritative value when available.
    var exists = (typeof info.exists === "boolean")
      ? info.exists
      : (count != null ? count > 0 : true);

    // Empty-state callout.
    var ddCode = $("#obDataDir");
    if (ddCode) ddCode.textContent = path || "(unknown)";
    var chip = $("#obDataDirChip");
    if (chip) {
      if (exists && (count == null || count > 0)) {
        chip.className = "ob-chip ok";
        chip.textContent = count != null
          ? count.toLocaleString() + " assets found"
          : "ready";
      } else {
        chip.className = "ob-chip warn";
        chip.textContent = exists ? "folder empty" : "folder not found";
      }
    }

    // Header pill colour.
    var hdr = $("#dataDir");
    if (hdr) {
      hdr.classList.remove("dd-ok", "dd-warn");
      hdr.classList.add(exists && (count == null || count > 0) ? "dd-ok" : "dd-warn");
    }

    // Authoritative existence check via /api/health (cheap, cached server
    // side). Gate the chip on data_dir.exists, NOT health.ok.
    fetch("/api/health")
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (h) {
        if (!h) return;
        var dd = (h.tools_resolved && h.tools_resolved.data_dir) || {};
        if (typeof dd.exists === "boolean") {
          refreshDataDir({
            data_dir: dd.path || path,
            exists: dd.exists,
            count: count,
          });
        }
      })
      .catch(function () { /* offline / older server — keep the count-based guess */ });
  }

  function ensureCategories() {
    if (liveCategories) return;
    fetch("/api/manifest/categories")
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d) return;
        liveCategories = Array.isArray(d.categories) ? d.categories : null;
        // If the overlay is showing step 1, re-render to pick up chips.
        if (state.overlay && !state.overlay.hidden && state.idx === 0) renderStep();
      })
      .catch(function () {});
  }

  // ── wiring ───────────────────────────────────────────────────────
  function wire() {
    ensureCategories();

    var help = $("#btnHelp");
    if (help) help.addEventListener("click", function () { open(); });

    var ew = $("#btnEmptyWalkthrough");
    if (ew) ew.addEventListener("click", function () { open(); });

    var eb = $("#btnEmptyBrowse");
    if (eb) eb.addEventListener("click", function () {
      markSeen();
      // Nudge focus toward the tree so keyboard users land somewhere useful.
      var tree = document.querySelector("pso-asset-tree");
      if (tree && typeof tree.focus === "function") {
        try { tree.focus(); } catch (_e) {}
      }
    });

    // "?" / Shift+/ opens help, unless a text field is focused.
    document.addEventListener("keydown", function (e) {
      if (e.key !== "?" ) return;
      var t = e.target;
      var tag = t && t.tagName ? t.tagName.toLowerCase() : "";
      if (tag === "input" || tag === "textarea" || (t && t.isContentEditable)) return;
      if (state.overlay && !state.overlay.hidden) return; // already open
      e.preventDefault();
      open();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", wire);
  } else {
    wire();
  }

  window.psoOnboarding = {
    open: open,
    close: close,
    maybeAutoShow: maybeAutoShow,
    refreshDataDir: refreshDataDir,
  };
})();
