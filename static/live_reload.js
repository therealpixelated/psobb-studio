// =====================================================================
// PSOBB Texture Editor - Live-reload SSE bridge (v5 polish, 2026-04-25)
// =====================================================================
// Connects to GET /api/events (Server-Sent Events) and re-emits each
// `cache.changed` event onto the global window.bus / window.PSOBus so
// any panel can subscribe with one line:
//
//     window.bus.on("cache.changed", (payload) => { ... });
//
// Payload shape:
//     { path: "cache/njm_export/foo.njm", kind: "create"|"modify"|"delete" }
//
// Throttle: events for the same path within a 200 ms window collapse to
// the LAST seen kind. This prevents fast-write spam from a build script
// (which often rewrites the .njm twice in microseconds) from flooding
// the bus and triggering N redundant refreshes per panel.
//
// UI: a small "live" badge auto-mounts in the header (between the data
// dir and the action buttons) showing the connection state:
//   * pulsing green dot   - connected, watcher active
//   * solid yellow dot    - connected, just received an event (200ms flash)
//   * red x               - disconnected, will retry with backoff
//
// Reconnect: exponential backoff (1s, 2s, 4s, 8s, max 30s) when the
// EventSource errors. The backoff resets on every successful event.
// =====================================================================

(function () {
  "use strict";

  if (window.__psoLiveReloadLoaded) return;
  window.__psoLiveReloadLoaded = true;

  // ------------------------------------------------------------------
  // Tunables
  // ------------------------------------------------------------------
  const COALESCE_MS = 200;
  const BACKOFF_MS = [1000, 2000, 4000, 8000, 16000, 30000];
  const FLASH_MS = 200;
  const SOURCE_URL = "/api/events";

  // ------------------------------------------------------------------
  // State
  // ------------------------------------------------------------------
  const state = {
    es: null,                 // active EventSource
    badge: null,              // header badge element
    badgeFlashTimer: null,    // pending un-flash
    reconnectAttempts: 0,
    reconnectTimer: null,
    pending: new Map(),       // path -> {kind, ts}  for coalescing
    coalesceTimer: null,
    eventCount: 0,            // observed lifetime; used by tests
    lastEvent: null,          // last delivered event
    closed: false,
  };

  // Surface for test harnesses + devtools poking.
  window.psoLiveReload = {
    state: () => ({
      connected: state.es && state.es.readyState === 1,
      eventCount: state.eventCount,
      lastEvent: state.lastEvent,
      reconnectAttempts: state.reconnectAttempts,
    }),
    forceFlush: forceFlush,
    disconnect: disconnect,
    reconnect: connect,
  };

  // ------------------------------------------------------------------
  // Bus helper - resolve once + warn if missing.
  // ------------------------------------------------------------------
  function emit(eventName, payload) {
    if (window.bus && typeof window.bus.emit === "function") {
      window.bus.emit(eventName, payload);
      return;
    }
    if (window.PSOBus && typeof window.PSOBus.emit === "function") {
      window.PSOBus.emit(eventName, payload);
      return;
    }
    // bus.js failed to load / wrong order — stash on window so a late
    // listener can still observe it.
    if (!window.__psoLiveReloadBacklog) window.__psoLiveReloadBacklog = [];
    window.__psoLiveReloadBacklog.push({ event: eventName, payload });
  }

  // ------------------------------------------------------------------
  // Coalescing - merge events per path within COALESCE_MS.
  // ------------------------------------------------------------------
  function pushIncoming(payload) {
    if (!payload || !payload.path) return;
    const prev = state.pending.get(payload.path);
    // Merge logic: if the prior pending was a "create" and we now see
    // "delete", the file came + went mid-window; collapse to delete.
    // Generally the LAST kind wins because that reflects the disk's
    // current state at flush time.
    if (prev && prev.kind === "create" && payload.kind === "delete") {
      state.pending.set(payload.path, { kind: "delete", ts: Date.now() });
    } else {
      state.pending.set(payload.path, { kind: payload.kind, ts: Date.now() });
    }
    if (!state.coalesceTimer) {
      state.coalesceTimer = setTimeout(forceFlush, COALESCE_MS);
    }
  }

  function forceFlush() {
    if (state.coalesceTimer) {
      clearTimeout(state.coalesceTimer);
      state.coalesceTimer = null;
    }
    if (state.pending.size === 0) return;
    const batch = [];
    for (const [path, info] of state.pending) {
      batch.push({ path, kind: info.kind, ts: info.ts });
    }
    state.pending.clear();
    for (const ev of batch) {
      state.eventCount += 1;
      state.lastEvent = ev;
      emit("cache.changed", ev);
    }
    flashBadge();
  }

  // ------------------------------------------------------------------
  // Badge
  // ------------------------------------------------------------------
  function _devBadgesEnabled() {
    // The live-reload badge is a DEVELOPER hot-reload status light, not a
    // user-facing control — it reads as cryptic ("live"?) to end users.
    // Hidden by default; opt in with ?dev in the URL or localStorage
    // 'pso.devBadges'='1'. The SSE auto-reload itself still runs silently.
    try {
      if (new URLSearchParams(location.search).has("dev")) return true;
      return localStorage.getItem("pso.devBadges") === "1";
    } catch (_e) {
      return false;
    }
  }

  function ensureBadge() {
    if (!_devBadgesEnabled()) return null;
    if (state.badge && state.badge.isConnected) return state.badge;
    // Find the header. The dataDir span is a stable anchor sitting at
    // the very left of the header.
    const header = document.querySelector("header");
    if (!header) return null;
    const badge = document.createElement("span");
    badge.id = "liveReloadBadge";
    badge.className = "live-reload-badge live-reload-disconnected";
    badge.title = "live-reload: connecting...";
    badge.innerHTML = '<span class="live-reload-dot"></span><span class="live-reload-label">live</span>';
    badge.addEventListener("click", () => {
      // Click for a quick rescan + status toast. Useful when a build
      // dropped a file that hasn't propagated through SSE yet (e.g.
      // user manually wrote into cache).
      forceRescan();
    });
    // Insert right after the dataDir span (between dataDir and grow).
    const anchor = header.querySelector("#dataDir");
    if (anchor && anchor.parentNode) {
      anchor.parentNode.insertBefore(badge, anchor.nextSibling);
    } else {
      header.appendChild(badge);
    }
    state.badge = badge;
    return badge;
  }

  function setBadgeState(kind, title) {
    const b = ensureBadge();
    if (!b) return;
    b.classList.remove(
      "live-reload-connecting",
      "live-reload-connected",
      "live-reload-disconnected",
      "live-reload-flash",
    );
    b.classList.add("live-reload-" + kind);
    if (title) b.title = title;
  }

  function flashBadge() {
    const b = ensureBadge();
    if (!b) return;
    b.classList.add("live-reload-flash");
    if (state.badgeFlashTimer) clearTimeout(state.badgeFlashTimer);
    state.badgeFlashTimer = setTimeout(() => {
      b.classList.remove("live-reload-flash");
      state.badgeFlashTimer = null;
    }, FLASH_MS);
  }

  // ------------------------------------------------------------------
  // Connect / reconnect
  // ------------------------------------------------------------------
  function connect() {
    if (state.closed) return;
    if (state.es) {
      try { state.es.close(); } catch (_e) {}
      state.es = null;
    }
    setBadgeState("connecting", "live-reload: connecting...");
    let es;
    try {
      es = new EventSource(SOURCE_URL);
    } catch (e) {
      console.warn("[live_reload] EventSource construct failed:", e);
      scheduleReconnect();
      return;
    }
    state.es = es;

    es.addEventListener("ready", (ev) => {
      try {
        const data = JSON.parse(ev.data || "{}");
        setBadgeState(
          "connected",
          "live-reload: connected (" + (data.subscribers || 1) + " active)",
        );
      } catch (_e) {
        setBadgeState("connected", "live-reload: connected");
      }
      state.reconnectAttempts = 0;
    });

    es.addEventListener("cache.changed", (ev) => {
      try {
        const payload = JSON.parse(ev.data || "{}");
        pushIncoming(payload);
      } catch (e) {
        console.warn("[live_reload] bad cache.changed payload:", e, ev.data);
      }
      // The act of receiving anything is a strong "we are alive" signal.
      state.reconnectAttempts = 0;
    });

    es.addEventListener("heartbeat", () => {
      // Update tooltip with the time since last event so the user can
      // see the watcher hasn't gone silent.
      const b = ensureBadge();
      if (b) b.title = "live-reload: connected (heartbeat " + new Date().toLocaleTimeString() + ")";
    });

    es.onopen = () => {
      // onopen fires before the "ready" handshake; keep tooltip in sync.
      setBadgeState("connecting", "live-reload: handshaking...");
    };

    es.onerror = () => {
      // EventSource auto-reconnects internally, but it can stall in
      // some browser/proxy combos. Force-close + manual backoff gives
      // us deterministic recovery + a visible "disconnected" state.
      if (es.readyState === 2) {
        // CLOSED — reschedule.
        setBadgeState("disconnected", "live-reload: disconnected, reconnecting...");
        scheduleReconnect();
      } else if (es.readyState === 0) {
        // CONNECTING — let the native retry try once, but back-stop.
        setTimeout(() => {
          if (state.es === es && es.readyState !== 1) {
            setBadgeState("disconnected", "live-reload: disconnected, reconnecting...");
            try { es.close(); } catch (_e) {}
            state.es = null;
            scheduleReconnect();
          }
        }, 5000);
      }
    };
  }

  function scheduleReconnect() {
    if (state.closed) return;
    if (state.reconnectTimer) return;
    const idx = Math.min(state.reconnectAttempts, BACKOFF_MS.length - 1);
    const delay = BACKOFF_MS[idx];
    state.reconnectAttempts += 1;
    state.reconnectTimer = setTimeout(() => {
      state.reconnectTimer = null;
      connect();
    }, delay);
  }

  function disconnect() {
    state.closed = true;
    if (state.es) {
      try { state.es.close(); } catch (_e) {}
      state.es = null;
    }
    if (state.reconnectTimer) {
      clearTimeout(state.reconnectTimer);
      state.reconnectTimer = null;
    }
    setBadgeState("disconnected", "live-reload: disconnected");
  }

  function forceRescan() {
    // Trigger server-side rescan; the resulting events flow back via SSE.
    fetch("/api/events/rescan", { method: "POST" })
      .then((r) => r.json())
      .then((j) => {
        if (j && typeof j.events_fired === "number") {
          flashBadge();
          const b = ensureBadge();
          if (b) {
            const prev = b.title;
            b.title = "live-reload: forced rescan, " + j.events_fired + " events";
            setTimeout(() => { if (b.title.indexOf("forced rescan") === 0) b.title = prev; }, 2000);
          }
        }
      })
      .catch((e) => console.warn("[live_reload] rescan failed:", e));
  }

  // ------------------------------------------------------------------
  // Init
  // ------------------------------------------------------------------
  function init() {
    if (typeof EventSource === "undefined") {
      console.warn("[live_reload] EventSource not supported in this browser");
      ensureBadge();
      setBadgeState("disconnected", "live-reload: EventSource unavailable");
      return;
    }
    ensureBadge();
    connect();
    window.addEventListener("beforeunload", disconnect);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
