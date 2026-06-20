// PSOBB Modding Suite — sidecar event bus.
// =====================================================================
// Plain pub/sub wrapper attached to `window.bus` and `window.PSOBus`
// (alias used in MASTER_PLAN/04_proposed_architecture.md). New plugins
// (asset tree, model viewer extensions, animation panel, sculpt) talk
// to each other through this bus instead of reaching into app.js's
// global state object — keeps the existing 1.2.0 editor decoupled from
// the new modding-suite modules.
//
// Channels currently in use (extend freely; this file enforces no
// schema):
//   asset.opened        { path, entry }     - tree → app.js → openFile
//   tile.edited         { path, tileIdx }   - app.js → model viewer
//   model.loaded        { path, ... }
//   anim.frame          { frameIdx, ts }
//   deploy.done         { path, backupName }
//
// Idempotent on multiple loads: re-loading this script is a no-op.
// =====================================================================

(function () {
  "use strict";

  if (window.bus && window.PSOBus) {
    // Already loaded — second <script> tag would clobber listeners.
    return;
  }

  // listeners: Map<eventName, Set<fn>>. Set ensures the same fn
  // registered twice fires once; prevents leaks from reload-style
  // bindings.
  const listeners = new Map();

  function on(event, fn) {
    if (typeof event !== "string" || typeof fn !== "function") {
      console.warn("[bus] on() expects (string, function); got", event, fn);
      return () => {};
    }
    let set = listeners.get(event);
    if (!set) {
      set = new Set();
      listeners.set(event, set);
    }
    set.add(fn);
    // Return an unsubscribe shortcut — common pattern in modern buses
    // (mitt, nanoevents). Optional; off(event,fn) still works.
    return () => off(event, fn);
  }

  function off(event, fn) {
    const set = listeners.get(event);
    if (!set) return;
    if (fn) {
      set.delete(fn);
      if (set.size === 0) listeners.delete(event);
    } else {
      // off(event) alone clears the channel.
      listeners.delete(event);
    }
  }

  function emit(event, payload) {
    const set = listeners.get(event);
    if (!set || set.size === 0) return 0;
    let delivered = 0;
    // Snapshot subscribers so a handler that calls off() during dispatch
    // doesn't mutate the iteration target.
    for (const fn of Array.from(set)) {
      try {
        fn(payload);
        delivered += 1;
      } catch (e) {
        // One listener crashing must not break the others.
        console.error(`[bus] listener for '${event}' threw:`, e);
      }
    }
    return delivered;
  }

  // For debugging in DevTools: list all current channels + counts.
  function _channels() {
    const out = {};
    for (const [k, v] of listeners) out[k] = v.size;
    return out;
  }

  const bus = Object.freeze({ on, off, emit, _channels });

  // Aliases: `bus` is the short form used by Agent 5's tree.js,
  // `PSOBus` is the architecture-doc canonical name. They point to
  // the same instance so a listener registered via either resolves
  // to one channel.
  window.bus = bus;
  window.PSOBus = bus;
})();
