// =====================================================================
// PSOBB Transform Gizmo — translate / rotate / scale gizmo for vertex
// and bone selections.  2026-04-26
//
// Wraps THREE.TransformControls (loaded from the same CDN as the rest
// of the three.js stack in model_viewer.js).  Exposes a tiny imperative
// API that edit_panel.js drives:
//
//   psoTransformGizmo.attach(target)       attach to a THREE.Object3D
//   psoTransformGizmo.detach()             remove from the scene
//   psoTransformGizmo.setMode("translate"|"rotate"|"scale")
//   psoTransformGizmo.isAttached()
//   psoTransformGizmo.getDelta()           since last reset
//   psoTransformGizmo.resetDelta()
//   psoTransformGizmo.onChange(fn)         per-frame change callback
//   psoTransformGizmo.onCommit(fn)         on mouseup callback
//   psoTransformGizmo.dispose()            full teardown
//
// The gizmo is parented to state.scene from psoGetScene equivalent
// (renderer + camera + scene come from psoGetCanvas/psoGetCamera/
// psoGetRenderer + psoGetMeshGroup's parent).
//
// Why a wrapper?  TransformControls grabs pointer events directly.  Edit
// mode needs to KNOW when the gizmo has the pointer so a click on its
// arrow doesn't also start a vertex-selection box.  We expose
// isDragging() so edit_panel.js can guard.
//
// Idempotent on multiple loads.
// =====================================================================

(function () {
  "use strict";

  if (window.psoTransformGizmo) return;

  // ---- module state -------------------------------------------------
  const state = {
    THREE: null,
    controls: null,        // THREE.TransformControls instance
    proxy: null,           // THREE.Object3D the gizmo manipulates
    mode: "translate",
    onChangeFns: [],
    onCommitFns: [],
    dragging: false,
    deltaPos: null,        // [dx, dy, dz] since last reset
    deltaRot: null,        // [drx, dry, drz] euler radians
    deltaScl: null,        // [dsx, dsy, dsz]
    startPos: null,
    startRot: null,
    startScl: null,
    importPromise: null,
  };

  function emitChange() {
    for (const fn of state.onChangeFns) {
      try { fn(); } catch (e) { console.error("[gizmo] onChange threw:", e); }
    }
  }
  function emitCommit() {
    for (const fn of state.onCommitFns) {
      try { fn(state.deltaPos, state.deltaRot, state.deltaScl); }
      catch (e) { console.error("[gizmo] onCommit threw:", e); }
    }
  }

  // ---- lazy load TransformControls ---------------------------------
  // model_viewer.js imports three.module.js as a static module.  We
  // can't directly import in a classic-script context, but we can
  // dynamic-import — same CDN.
  function ensureControls() {
    if (state.importPromise) return state.importPromise;
    state.importPromise = (async function () {
      // Wait for window.THREE to be defined (model_viewer.js sets it).
      const start = Date.now();
      while (!window.THREE && Date.now() - start < 5000) {
        await new Promise((r) => setTimeout(r, 50));
      }
      if (!window.THREE) throw new Error("[gizmo] THREE not available on window");
      state.THREE = window.THREE;
      // Pull in TransformControls from the same three version as model_viewer.
      const mod = await import("https://unpkg.com/three@0.160.0/examples/jsm/controls/TransformControls.js");
      const TransformControls = mod.TransformControls;
      if (!TransformControls) {
        throw new Error("[gizmo] TransformControls export missing");
      }
      const camera = window.psoGetCamera && window.psoGetCamera();
      const renderer = window.psoGetRenderer && window.psoGetRenderer();
      if (!camera || !renderer) {
        throw new Error("[gizmo] camera/renderer not ready (open a model first)");
      }
      const ctl = new TransformControls(camera, renderer.domElement);
      ctl.size = 0.8;
      ctl.setMode(state.mode);
      ctl.addEventListener("change", function () {
        if (state.proxy && state.startPos) {
          state.deltaPos = [
            state.proxy.position.x - state.startPos[0],
            state.proxy.position.y - state.startPos[1],
            state.proxy.position.z - state.startPos[2],
          ];
        }
        if (state.proxy && state.startRot) {
          state.deltaRot = [
            state.proxy.rotation.x - state.startRot[0],
            state.proxy.rotation.y - state.startRot[1],
            state.proxy.rotation.z - state.startRot[2],
          ];
        }
        if (state.proxy && state.startScl) {
          state.deltaScl = [
            state.proxy.scale.x / state.startScl[0],
            state.proxy.scale.y / state.startScl[1],
            state.proxy.scale.z / state.startScl[2],
          ];
        }
        emitChange();
        // Force a render pass so updates feel snappy when auto-rotate is off.
        if (window.psoForceRender) window.psoForceRender();
      });
      ctl.addEventListener("dragging-changed", function (ev) {
        state.dragging = !!ev.value;
        if (state.dragging) {
          if (state.proxy) {
            state.startPos = [state.proxy.position.x, state.proxy.position.y, state.proxy.position.z];
            state.startRot = [state.proxy.rotation.x, state.proxy.rotation.y, state.proxy.rotation.z];
            state.startScl = [state.proxy.scale.x, state.proxy.scale.y, state.proxy.scale.z];
            state.deltaPos = [0, 0, 0];
            state.deltaRot = [0, 0, 0];
            state.deltaScl = [1, 1, 1];
          }
        } else {
          // mouseup — fire commit if there was actually a delta.
          emitCommit();
        }
      });
      // Add the gizmo helper to the scene; the helper renders the arrows /
      // rings.  In three.js >=0.160 TransformControls is itself a
      // Group, so we add it directly.
      const scene = window.psoGetMeshGroup && window.psoGetMeshGroup();
      const root = scene ? scene.parent : null;
      const dest = root || (window.__psoGizmoFallbackScene || null);
      if (!dest) {
        throw new Error("[gizmo] could not locate scene root");
      }
      dest.add(ctl);
      state.controls = ctl;
      return ctl;
    })();
    return state.importPromise;
  }

  // ---- public API --------------------------------------------------
  async function attach(target) {
    if (!target) return false;
    try {
      const ctl = await ensureControls();
      state.proxy = target;
      ctl.attach(target);
      return true;
    } catch (e) {
      console.error("[gizmo] attach failed:", e);
      return false;
    }
  }

  function detach() {
    if (state.controls) {
      try { state.controls.detach(); } catch (_e) {}
    }
    state.proxy = null;
    state.startPos = null;
    state.startRot = null;
    state.startScl = null;
    state.deltaPos = null;
    state.deltaRot = null;
    state.deltaScl = null;
  }

  function setMode(m) {
    if (m !== "translate" && m !== "rotate" && m !== "scale") return false;
    state.mode = m;
    if (state.controls) {
      try { state.controls.setMode(m); } catch (_e) {}
    }
    return true;
  }

  function getMode() { return state.mode; }

  function isAttached() { return !!state.proxy; }
  function isDragging() { return !!state.dragging; }
  function getProxy() { return state.proxy; }
  function getDelta() {
    return {
      position: state.deltaPos ? state.deltaPos.slice() : null,
      rotation: state.deltaRot ? state.deltaRot.slice() : null,
      scale: state.deltaScl ? state.deltaScl.slice() : null,
    };
  }

  function resetDelta() {
    state.deltaPos = [0, 0, 0];
    state.deltaRot = [0, 0, 0];
    state.deltaScl = [1, 1, 1];
    if (state.proxy) {
      state.startPos = [state.proxy.position.x, state.proxy.position.y, state.proxy.position.z];
      state.startRot = [state.proxy.rotation.x, state.proxy.rotation.y, state.proxy.rotation.z];
      state.startScl = [state.proxy.scale.x, state.proxy.scale.y, state.proxy.scale.z];
    }
  }

  function onChange(fn) {
    if (typeof fn !== "function") return function () {};
    state.onChangeFns.push(fn);
    return function dispose() {
      const idx = state.onChangeFns.indexOf(fn);
      if (idx >= 0) state.onChangeFns.splice(idx, 1);
    };
  }
  function onCommit(fn) {
    if (typeof fn !== "function") return function () {};
    state.onCommitFns.push(fn);
    return function dispose() {
      const idx = state.onCommitFns.indexOf(fn);
      if (idx >= 0) state.onCommitFns.splice(idx, 1);
    };
  }

  function dispose() {
    detach();
    if (state.controls) {
      try { state.controls.dispose(); } catch (_e) {}
      const root = state.controls.parent;
      if (root) {
        try { root.remove(state.controls); } catch (_e) {}
      }
      state.controls = null;
    }
    state.onChangeFns.length = 0;
    state.onCommitFns.length = 0;
  }

  // ---- visibility ---------------------------------------------------
  function setVisible(v) {
    if (state.controls) state.controls.visible = !!v;
  }

  window.psoTransformGizmo = Object.freeze({
    attach: attach,
    detach: detach,
    setMode: setMode,
    getMode: getMode,
    isAttached: isAttached,
    isDragging: isDragging,
    getProxy: getProxy,
    getDelta: getDelta,
    resetDelta: resetDelta,
    onChange: onChange,
    onCommit: onCommit,
    setVisible: setVisible,
    dispose: dispose,
  });
})();
