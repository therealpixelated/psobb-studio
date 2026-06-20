// PSOBB Modding Suite — asset lifecycle / abort coordinator.
// =====================================================================
// Wave 7 (2026-04-26). Before this module, every panel issued its own
// `fetch()` calls with no shared cancel signal. When the user rapid-
// clicked through a list of bosses, each click stacked 6-25 sub-requests
// onto a single uvicorn worker; the queue grew to 50-100 in-flight and
// the server appeared to hang for minutes.
//
// This file owns the SHARED AbortController for "the asset the user is
// currently looking at". When the user picks a NEW asset, we bump the
// epoch, abort the prior controller, and any fetches still in flight
// for the previous asset reject with AbortError — which we catch
// silently downstream (a stale fetch's resolution is irrelevant once
// the user has moved on).
//
// Usage from any panel:
//
//     const url = "/api/model_bundle/" + encodeURIComponent(path);
//     const r   = await window.psoAssetLifecycle.fetchAsset(url, {
//       headers: { Accept: "application/json" },
//     });
//     // ... AbortError is bubbled up; caller checks e.name === "AbortError"
//
// Or, panels that own their own fetch sites can read the current signal:
//
//     const sig = window.psoAssetLifecycle.signal();
//     const r   = await fetch(url, { signal: sig });
//
// Asset lifecycle entry points (asset_router.js, hotkeys' next/prev,
// quicksearch teleport) call `beginAsset(path)` BEFORE any fetch fires
// for the new asset. The previous controller's .abort() runs first, so
// the in-flight queue collapses immediately.
//
// Debounce: rapid-click protection is implemented via
// `debouncedOpen(path, openFn)`. 100 ms timer per asset_router call;
// only the LAST click within the window actually fires openFn. This
// is in addition to abort: even if the user rapid-clicks 10 different
// assets in 5 seconds, only one fetch pipeline ever runs concurrently
// because each click resets the timer + aborts the prior controller.
//
// Idempotent on multi-load (matches bus.js convention).
// =====================================================================

(function () {
  "use strict";

  if (window.psoAssetLifecycle) return;

  /** @type {{epoch: number, path: string|null, controller: AbortController|null}} */
  const state = {
    epoch: 0,
    path: null,
    controller: null,
  };

  /** Last invocation handle for `debouncedOpen`. */
  let _debounceTimer = null;
  /** Pending arguments held during the debounce window. */
  let _debouncePending = null;
  /** Configurable debounce delay (ms). 100 ms matches the spec. */
  const DEBOUNCE_MS = 100;

  /**
   * Bump the epoch and abort the prior controller. Call this whenever a
   * NEW asset is being opened. Returns the new epoch number so callers
   * can detect "did the user move on while my fetch was in flight"
   * without relying on the AbortSignal alone.
   *
   * `path` is informational only (lets devtools / bus subscribers know
   * which asset is now current); the abort behaviour is path-agnostic.
   */
  function beginAsset(path) {
    // Abort the previous controller. Any fetch still bound to its
    // signal rejects with AbortError on next microtask.
    if (state.controller) {
      try {
        state.controller.abort();
      } catch (_e) {
        // AbortController.abort() is spec'd to never throw, but defensive
        // browsers (older Edge) have been observed to throw on a
        // controller whose signal was already aborted by another caller.
        // Swallow and continue.
      }
    }
    state.epoch += 1;
    state.path = path || null;
    state.controller = new AbortController();
    // Surface lifecycle events on the bus for any panel that wants to
    // wire its own teardown (e.g. cancel a long-running canvas paint).
    if (window.bus && typeof window.bus.emit === "function") {
      try {
        window.bus.emit("asset.lifecycle.begin", {
          path: state.path,
          epoch: state.epoch,
        });
      } catch (_e) {}
    }
    return state.epoch;
  }

  /**
   * Current AbortSignal. Returns null if no asset is yet active (panels
   * should defensively pass `signal: sig || undefined` to fetch).
   */
  function signal() {
    return state.controller ? state.controller.signal : null;
  }

  /** Current epoch number (monotonic). */
  function epoch() {
    return state.epoch;
  }

  /** Current asset path (informational). */
  function path() {
    return state.path;
  }

  /**
   * Wrap fetch() so the current signal is auto-injected and `init.signal`
   * (if the caller already supplied one) is preserved as the OUTER
   * abort: an external abort still cancels even if the lifecycle hasn't
   * moved.
   *
   * Matches fetch() semantics — returns a Promise<Response>. Caller is
   * responsible for `.json()`/`.arrayBuffer()` — those go through the
   * same signal chain via the response stream, so a mid-body abort
   * cancels the body read too.
   */
  function fetchAsset(url, init) {
    const merged = Object.assign({}, init || {});
    const lifSig = signal();
    if (merged.signal && lifSig) {
      // Caller already has a signal AND we have a lifecycle signal.
      // Use AbortSignal.any when available (Chrome 116+, Firefox 124+);
      // fall back to a manual chain for older browsers.
      if (typeof AbortSignal !== "undefined" && typeof AbortSignal.any === "function") {
        merged.signal = AbortSignal.any([merged.signal, lifSig]);
      } else {
        const ctrl = new AbortController();
        const aborter = () => ctrl.abort();
        merged.signal.addEventListener("abort", aborter, { once: true });
        lifSig.addEventListener("abort", aborter, { once: true });
        merged.signal = ctrl.signal;
      }
    } else if (lifSig && !merged.signal) {
      merged.signal = lifSig;
    }
    return fetch(url, merged);
  }

  /**
   * Debounce wrapper for asset opens. The user rapid-clicks A, B, C
   * inside 100 ms; only C's openFn actually runs. Each call:
   *   1. clears any pending timer,
   *   2. records the args,
   *   3. starts a fresh 100 ms timer.
   * When the timer fires, the LAST recorded args are used and openFn is
   * invoked with them.
   *
   * IMPORTANT: this does NOT abort prior in-flight fetches (that's
   * `beginAsset`'s job). The point is to avoid even STARTING a fetch if
   * the user is mid-rapid-click. If you want both, call beginAsset()
   * inside your openFn and the chain composes cleanly.
   */
  function debouncedOpen(path, openFn) {
    _debouncePending = { path: path, fn: openFn };
    if (_debounceTimer) {
      clearTimeout(_debounceTimer);
    }
    _debounceTimer = setTimeout(function () {
      const pending = _debouncePending;
      _debouncePending = null;
      _debounceTimer = null;
      if (pending && typeof pending.fn === "function") {
        try {
          pending.fn(pending.path);
        } catch (e) {
          console.error("[asset_lifecycle] debounced open threw:", e);
        }
      }
    }, DEBOUNCE_MS);
  }

  /**
   * Detect whether an error is the "lifecycle was aborted" condition.
   * Use this in panel error-handlers to swallow expected aborts vs.
   * surfacing real network errors.
   */
  function isAbort(e) {
    if (!e) return false;
    if (e.name === "AbortError") return true;
    // DOMException with code 20 (legacy Edge / older browsers).
    if (typeof DOMException !== "undefined" && e instanceof DOMException && e.code === 20) {
      return true;
    }
    return false;
  }

  window.psoAssetLifecycle = Object.freeze({
    beginAsset,
    signal,
    epoch,
    path,
    fetchAsset,
    debouncedOpen,
    isAbort,
    DEBOUNCE_MS,
  });
})();
