// =====================================================================
// PSOBB Texture Editor — Live Mod-Test shared module (2026-04-25).
//
// Public surface (window.PSOLiveTest):
//   triggerLiveTest(kind, opts)   POST /api/live_test, return parsed JSON.
//                                 Surfaces a status pip + log entry into
//                                 the panel identified by opts.panelId.
//   ensureLiveButton(opts)        idempotently mount a "Live Test" button
//                                 + status pip into a panel. Re-callable
//                                 (e.g. on each render) — no leaks.
//   getConfig(force=false)        cached GET /api/live_test/config.
//   logTail(panelId, limit=3)     cached GET /api/live_test/log filtered
//                                 to the panel.
//
// Status pip lifecycle:
//   idle → preparing → applying → live   (success path)
//   idle → preparing → applying → failed (error path)
//
// The pip is a single DOM element with class "lt-status-pip" + a state
// data-attribute. Panels can place the pip anywhere by calling
// PSOLiveTest.attachPip(host, panelId); the module manages its own
// children.
//
// All CSS lives in static/style.css under the .lt-* prefix; this module
// never injects <style> blocks.
// =====================================================================

(function () {
  "use strict";

  if (window.PSOLiveTest) return;

  // -------------------------------------------------------------------
  // Module state
  // -------------------------------------------------------------------
  // panelId -> { pipEl, recentLogEl, lastFetched }
  const panels = new Map();
  // Last config fetch (cached for ~5 s; the values rarely change in a session)
  let configCache = null;
  let configCacheTs = 0;
  const CONFIG_TTL_MS = 5000;

  // -------------------------------------------------------------------
  // Tiny helpers
  // -------------------------------------------------------------------
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;",
                '"': "&quot;", "'": "&#39;" })[c];
    });
  }

  function fmtTs(ts) {
    if (typeof ts !== "number") return "";
    const d = new Date(ts * 1000);
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    const ss = String(d.getSeconds()).padStart(2, "0");
    return hh + ":" + mm + ":" + ss;
  }

  function setPipState(panelId, state, message) {
    const ent = panels.get(panelId);
    if (!ent || !ent.pipEl) return;
    // `staged` is a soft-success state used when the texture override
    // landed on disk but the ASI consumer (mod_live_replace) hasn't
    // checked in via heartbeat. Visually distinct from `live` (which
    // implies the running game has already swallowed the change).
    const valid = ["idle", "preparing", "applying", "live", "staged", "failed"];
    const st = valid.indexOf(state) >= 0 ? state : "idle";
    ent.pipEl.setAttribute("data-state", st);
    ent.pipEl.textContent = message || st;
  }

  // -------------------------------------------------------------------
  // Config + log fetch
  // -------------------------------------------------------------------
  async function getConfig(force) {
    const now = Date.now();
    if (!force && configCache && (now - configCacheTs) < CONFIG_TTL_MS) {
      return configCache;
    }
    try {
      const r = await fetch("/api/live_test/config");
      if (!r.ok) throw new Error("HTTP " + r.status);
      configCache = await r.json();
      configCacheTs = now;
      return configCache;
    } catch (e) {
      // Don't cache failures — the next call should retry.
      console.warn("[live_test] config fetch failed:", e);
      return null;
    }
  }

  async function logTail(panelId, limit) {
    limit = limit || 3;
    const url = "/api/live_test/log?limit=" + encodeURIComponent(limit) +
                (panelId ? "&panel=" + encodeURIComponent(panelId) : "");
    try {
      const r = await fetch(url);
      if (!r.ok) return [];
      const data = await r.json();
      return data.entries || [];
    } catch (e) {
      console.warn("[live_test] log fetch failed:", e);
      return [];
    }
  }

  function renderRecentLog(panelId, entries) {
    const ent = panels.get(panelId);
    if (!ent || !ent.recentLogEl) return;
    if (!entries || !entries.length) {
      ent.recentLogEl.textContent = "";
      ent.recentLogEl.hidden = true;
      return;
    }
    const html = entries.slice().reverse().map(function (it) {
      const okBadge = it.requires_manual_reload
        ? '<span class="lt-log-badge warn">manual reload</span>'
        : (it.ok ? '<span class="lt-log-badge ok">ok</span>'
                 : '<span class="lt-log-badge err">err</span>');
      const dep = it.deployed && (it.deployed.deployed_to || it.deployed.override_png) || "";
      const depShort = dep ? dep.replace(/^.*[\\\/]/, "") : "";
      return '<div class="lt-log-row">' +
             '<span class="lt-log-ts">' + escapeHtml(fmtTs(it.ts)) + '</span>' +
             okBadge +
             '<span class="lt-log-kind">' + escapeHtml(it.kind || "") + '</span>' +
             (depShort ? ' <span class="lt-log-target dim">→ ' + escapeHtml(depShort) + '</span>' : '') +
             '</div>';
    }).join("");
    ent.recentLogEl.innerHTML = html;
    ent.recentLogEl.hidden = false;
  }

  // -------------------------------------------------------------------
  // Core trigger
  // -------------------------------------------------------------------
  async function triggerLiveTest(kind, opts) {
    opts = opts || {};
    const panelId = opts.panelId || kind;
    setPipState(panelId, "preparing", "preparing…");
    const body = Object.assign({ kind: kind, panel: panelId }, opts.body || {});
    setPipState(panelId, "applying", "applying…");
    try {
      const r = await fetch("/api/live_test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const detailText = await r.text();
      let parsed = null;
      try { parsed = JSON.parse(detailText); } catch (_) { /* swallow */ }
      if (!r.ok) {
        const detail = (parsed && parsed.detail) || detailText.slice(0, 120);
        setPipState(panelId, "failed", "failed: " + detail);
        return { ok: false, error: detail };
      }
      const data = parsed || {};
      if (data.requires_manual_reload) {
        setPipState(panelId, "live",
                    "deployed (manual newserv reload required)");
      } else if (data.category === "client" &&
                 data.deployed && data.deployed.consumer_active === false) {
        // Texture override landed on disk but the ASI consumer hasn't
        // ticked its heartbeat — file is present, game hasn't picked
        // it up yet. Use the dedicated `staged` pip color so the user
        // can distinguish "successful but inert" from "fully live".
        setPipState(panelId, "staged",
                    "staged (ASI consumer not running)");
      } else if (data.category === "client" &&
                 data.deployed && data.deployed.consumer_active === true) {
        setPipState(panelId, "live", "live (ASI consumer applied)");
      } else {
        setPipState(panelId, "live", "live");
      }
      // Refresh the action log shown next to the pip.
      try {
        const tail = await logTail(panelId, 3);
        renderRecentLog(panelId, tail);
      } catch (_) { /* non-fatal */ }
      return data;
    } catch (e) {
      setPipState(panelId, "failed", "failed: " + (e.message || e));
      return { ok: false, error: e.message || String(e) };
    }
  }

  // -------------------------------------------------------------------
  // Pip + button mount
  // -------------------------------------------------------------------
  function attachPip(host, panelId) {
    if (!host) return null;
    let ent = panels.get(panelId);
    if (!ent) {
      ent = { pipEl: null, recentLogEl: null };
      panels.set(panelId, ent);
    }
    if (!ent.pipEl || !ent.pipEl.isConnected) {
      ent.pipEl = document.createElement("span");
      ent.pipEl.className = "lt-status-pip";
      ent.pipEl.setAttribute("data-state", "idle");
      ent.pipEl.textContent = "idle";
      host.appendChild(ent.pipEl);
    }
    if (!ent.recentLogEl || !ent.recentLogEl.isConnected) {
      ent.recentLogEl = document.createElement("div");
      ent.recentLogEl.className = "lt-log-list";
      ent.recentLogEl.hidden = true;
      host.appendChild(ent.recentLogEl);
    }
    // Initial fill of the action log (fire-and-forget).
    logTail(panelId, 3).then(function (entries) {
      renderRecentLog(panelId, entries);
    });
    return ent;
  }

  /**
   * Idempotently insert a "Live Test" button + status pip into a panel.
   *
   * opts:
   *   host        DOM element to insert into
   *   panelId     unique id used by the action log
   *   kind        "battle_param" | "itempmt" | "mob_dsl" | "texture"
   *   bodyBuilder ()=>{ ...request body }   (called at click time)
   *   label       optional button label override
   *   title       optional tooltip override
   *   className   optional extra class for the button
   *   beforeNode  optional sibling node to insertBefore (else appendChild)
   *
   * Returns the button element (or the existing one if already mounted).
   */
  function ensureLiveButton(opts) {
    if (!opts || !opts.host || !opts.panelId || !opts.kind) {
      console.warn("[live_test] ensureLiveButton: missing host/panelId/kind", opts);
      return null;
    }
    const btnId = "ltBtn_" + opts.panelId.replace(/[^a-z0-9_-]/gi, "_");
    let btn = opts.host.querySelector("#" + btnId);
    if (btn) return btn;
    btn = document.createElement("button");
    btn.id = btnId;
    btn.type = "button";
    btn.className = "lt-live-button" + (opts.className ? (" " + opts.className) : "");
    btn.title = opts.title ||
                "push this edit into the running game without re-launching";
    btn.innerHTML =
      '<span class="lt-live-dot" aria-hidden="true"></span>' +
      '<span class="lt-live-label">' +
      escapeHtml(opts.label || "Live Test") + '</span>';
    if (opts.beforeNode && opts.beforeNode.parentNode === opts.host) {
      opts.host.insertBefore(btn, opts.beforeNode);
    } else {
      opts.host.appendChild(btn);
    }
    btn.addEventListener("click", async function () {
      btn.disabled = true;
      try {
        const body = (typeof opts.bodyBuilder === "function")
                     ? (opts.bodyBuilder() || {})
                     : {};
        await triggerLiveTest(opts.kind, {
          panelId: opts.panelId,
          body: body,
        });
      } finally {
        btn.disabled = false;
      }
    });
    return btn;
  }

  // -------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------
  window.PSOLiveTest = Object.freeze({
    triggerLiveTest: triggerLiveTest,
    ensureLiveButton: ensureLiveButton,
    attachPip: attachPip,
    setPipState: setPipState,
    getConfig: getConfig,
    logTail: logTail,
    renderRecentLog: renderRecentLog,
  });
})();
