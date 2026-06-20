// =====================================================================
// PSOBB Texture Editor - Mob AI Authoring (Tier 1) perspective.
//
// Higher-level authoring layer over BattleParamEntry. Mirrors the UX
// of static/battle_param_panel.js but groups fields by SEMANTIC labels
// (Movement / Combat / AI Behavior / Resists / Stats) and uses
// kind-aware editors:
//   - "duration_seconds": float input + " s" suffix
//   - "angle_bams":       int input + " deg" suffix
//   - "percent" / "percent_int": int input + " %%" suffix
//   - everything else:    plain number
//
// Endpoints used:
//   GET  /api/mob_dsl/schemas
//   GET  /api/mob_dsl/{mob}
//   GET  /api/mob_dsl/presets
//   GET  /api/mob_dsl/{mob}/preset/{preset}
//   POST /api/mob_dsl/compile        -> returns compiled BattleParam JSON
//   POST /api/battle_param/{variant} -> stage compiled JSON
//   POST /api/battle_param/{variant}/deploy -> deploy
//
// State is module-scope so switching mobs doesn't lose unsaved edits.
// =====================================================================

(function () {
  "use strict";

  if (!window.PSOPerspectives) {
    console.warn("[mob_dsl] perspectives.js not loaded yet");
    return;
  }

  // Per-perspective state. ``patches`` is keyed by "{slot}_{difficulty}"
  // so a user can edit Booma in Normal AND Hildebear in Ultimate without
  // collision. The compile path replays these against the loaded
  // baseline and returns a full BattleParam JSON.
  const state = {
    schemas: null,        // { [slot]: schema }
    presets: null,        // [{name,title,description,mobs}]
    variant: "on",
    difficulty: 0,        // 0..3
    slot: 0x4B,           // active mob (default Booma)
    // patches[slot][difficulty][label] = dsl_value (only changed)
    // Object-of-objects so we can key on small ints.
    patches: {},
    _inspectorHost: null,
  };

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;",
                '"': "&quot;", "'": "&#39;" })[c];
    });
  }

  // ------------------------------------------------------------------
  // API helpers
  // ------------------------------------------------------------------
  async function getSchemas() {
    const r = await fetch("/api/mob_dsl/schemas");
    if (!r.ok) throw new Error("schemas: " + r.status);
    return r.json();
  }
  async function getPresets() {
    const r = await fetch("/api/mob_dsl/presets");
    if (!r.ok) throw new Error("presets: " + r.status);
    return r.json();
  }
  async function getPresetForMob(mob, preset) {
    const r = await fetch(
      "/api/mob_dsl/" + encodeURIComponent(mob) +
      "/preset/" + encodeURIComponent(preset)
    );
    if (!r.ok) {
      const detail = await r.text().catch(() => "");
      throw new Error("preset " + preset + ": " + r.status + " " + detail);
    }
    return r.json();
  }
  async function compile(variant, mobs, stage) {
    const r = await fetch("/api/mob_dsl/compile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ variant: variant, mobs: mobs, stage: !!stage }),
    });
    if (!r.ok) {
      const detail = await r.text().catch(() => "");
      throw new Error("compile (" + r.status + "): " + detail);
    }
    return r.json();
  }
  async function deployVariant(variant) {
    const r = await fetch(
      "/api/battle_param/" + encodeURIComponent(variant) + "/deploy",
      { method: "POST" }
    );
    if (!r.ok) {
      const detail = await r.text().catch(() => "");
      throw new Error("deploy " + variant + ": " + r.status + " " + detail);
    }
    return r.json();
  }

  // ------------------------------------------------------------------
  // Patch state helpers
  // ------------------------------------------------------------------
  function getPatchFields(slot, difficulty) {
    if (!state.patches[slot]) return null;
    return state.patches[slot][difficulty] || null;
  }

  function setPatchField(slot, difficulty, label, value) {
    if (!state.patches[slot]) state.patches[slot] = {};
    if (!state.patches[slot][difficulty]) state.patches[slot][difficulty] = {};
    state.patches[slot][difficulty][label] = value;
  }

  function clearPatchField(slot, difficulty, label) {
    if (!state.patches[slot]) return;
    if (!state.patches[slot][difficulty]) return;
    delete state.patches[slot][difficulty][label];
    if (Object.keys(state.patches[slot][difficulty]).length === 0) {
      delete state.patches[slot][difficulty];
    }
    if (Object.keys(state.patches[slot]).length === 0) {
      delete state.patches[slot];
    }
  }

  function clearAllPatches() {
    state.patches = {};
  }

  function totalChangedFields() {
    let total = 0;
    for (const slot of Object.keys(state.patches)) {
      for (const d of Object.keys(state.patches[slot])) {
        total += Object.keys(state.patches[slot][d]).length;
      }
    }
    return total;
  }

  // Build the wire-format mob list for /compile and /preview:
  //   [{mob, difficulty, fields: {...}}, ...]
  function buildPatchPayload() {
    const mobs = [];
    const diffNames = ["Normal", "Hard", "VeryHard", "Ultimate"];
    for (const slotStr of Object.keys(state.patches)) {
      const slot = parseInt(slotStr, 10);
      const schema = state.schemas[slot];
      const mobName = schema ? schema.name : ("0x" + slot.toString(16).toUpperCase());
      for (const dStr of Object.keys(state.patches[slot])) {
        const d = parseInt(dStr, 10);
        const fields = state.patches[slot][d];
        if (!fields || Object.keys(fields).length === 0) continue;
        mobs.push({
          mob: mobName,
          difficulty: diffNames[d] || "all",
          fields: Object.assign({}, fields),
        });
      }
    }
    return mobs;
  }

  // ------------------------------------------------------------------
  // UI rendering
  // ------------------------------------------------------------------
  function renderToolbar(hostEl) {
    const variants = ["on", "off", "lab_on", "lab_off", "ep4_on", "ep4_off"];
    const variantOpts = variants.map(function (v) {
      return '<option value="' + escapeHtml(v) + '"' +
             (v === state.variant ? " selected" : "") + ">" +
             escapeHtml(v) + "</option>";
    }).join("");
    const diffNames = ["Normal", "Hard", "VeryHard", "Ultimate"];
    const diffOpts = diffNames.map(function (n, i) {
      return '<option value="' + i + '"' +
             (i === state.difficulty ? " selected" : "") + ">" +
             n + "</option>";
    }).join("");
    const slotEntries = Object.entries(state.schemas || {})
      .map(function (kv) { return [parseInt(kv[0], 10), kv[1].name]; })
      .sort(function (a, b) { return a[0] - b[0]; });
    const slotOpts = slotEntries.map(function (kv) {
      const v = kv[0], n = kv[1];
      return '<option value="' + v + '"' +
             (v === state.slot ? " selected" : "") + ">0x" +
             v.toString(16).padStart(2, "0").toUpperCase() + "  " +
             escapeHtml(n) + "</option>";
    }).join("");

    // Preset dropdown (filtered to ones touching the active mob).
    const presetOpts = (state.presets || []).map(function (p) {
      return '<option value="' + escapeHtml(p.name) + '">' +
             escapeHtml(p.title || p.name) + "</option>";
    }).join("");

    hostEl.innerHTML =
      '<div class="mdsl-toolbar">' +
      '<label>variant <select id="mdslVariant">' + variantOpts + '</select></label>' +
      '<label>difficulty <select id="mdslDifficulty">' + diffOpts + '</select></label>' +
      '<label>mob <select id="mdslSlot">' + slotOpts + '</select></label>' +
      '<label class="mdsl-preset-cell" title="Apply a shipped preset to the active mob">preset ' +
        '<select id="mdslPreset"><option value="">(pick a preset...)</option>' + presetOpts + '</select>' +
      '</label>' +
      '<button type="button" id="mdslApplyPreset" class="ghost" title="apply selected preset to this mob">apply</button>' +
      '<button type="button" id="mdslClearMob" class="ghost" title="clear this mob+difficulty\'s edits">clear mob</button>' +
      '<button type="button" id="mdslClearAll" class="ghost" title="clear ALL pending edits across mobs">clear all</button>' +
      '<button type="button" id="mdslCompileDeploy" class="warn" title="compile patches → stage variant → deploy to newserv">compile + deploy</button>' +
      '<span class="dim mdsl-status" id="mdslStatus"></span>' +
      '</div>';

    hostEl.querySelector("#mdslVariant").addEventListener("change", function (e) {
      state.variant = e.target.value;
      setStatus("variant: " + state.variant);
    });
    hostEl.querySelector("#mdslDifficulty").addEventListener("change", function (e) {
      state.difficulty = parseInt(e.target.value, 10) || 0;
      renderEditor();
    });
    hostEl.querySelector("#mdslSlot").addEventListener("change", function (e) {
      state.slot = parseInt(e.target.value, 10) || 0;
      renderEditor();
    });
    hostEl.querySelector("#mdslApplyPreset").addEventListener("click", onApplyPreset);
    hostEl.querySelector("#mdslClearMob").addEventListener("click", onClearMob);
    hostEl.querySelector("#mdslClearAll").addEventListener("click", onClearAll);
    hostEl.querySelector("#mdslCompileDeploy").addEventListener("click", onCompileDeploy);

    // Live Test button — compile + stage + deploy to newserv. The
    // PSOLiveTest module owns the status pip; the mdslStatus span
    // continues to show the compile-side messages.
    if (window.PSOLiveTest) {
      const toolbar = hostEl.querySelector(".mdsl-toolbar");
      const statusEl = hostEl.querySelector("#mdslStatus");
      window.PSOLiveTest.ensureLiveButton({
        host: toolbar,
        beforeNode: statusEl,
        panelId: "mob-dsl",
        kind: "mob_dsl",
        label: "Live Test",
        title: "compile patches → stage → push to newserv (live)",
        bodyBuilder: function () {
          // Note: we DO NOT compile here — live-test assumes a staged
          // file already exists. onLiveTest() below runs the compile
          // first, then explicitly POSTs /api/live_test.
          return { variant: state.variant, attempt_reload: true };
        },
      });
      // Replace the auto-bound click handler with our compile-then-live
      // workflow so users get the expected one-click behavior.
      const ltBtn = toolbar.querySelector("#ltBtn_mob-dsl");
      if (ltBtn) {
        const newBtn = ltBtn.cloneNode(true);
        ltBtn.parentNode.replaceChild(newBtn, ltBtn);
        newBtn.addEventListener("click", onLiveTest);
      }
      window.PSOLiveTest.attachPip(toolbar, "mob-dsl");
    }
  }

  function setStatus(msg, isErr) {
    const el = document.getElementById("mdslStatus");
    if (!el) return;
    el.textContent = msg || "";
    el.classList.toggle("err", !!isErr);
  }

  function fieldKindHint(kind) {
    switch (kind) {
      case "duration_seconds": return "s";
      case "angle_bams":       return "deg";
      case "percent":          return "%";
      case "percent_int":      return "%";
      default:                 return "";
    }
  }

  function fieldStep(kind) {
    if (kind === "float" || kind === "percent" || kind === "duration_seconds") {
      return "any";
    }
    return "1";
  }

  function renderEditor() {
    const host = document.getElementById("mdslEditor");
    if (!host) return;
    if (!state.schemas) {
      host.innerHTML = '<div class="dim">loading schemas...</div>';
      return;
    }
    const schema = state.schemas[state.slot];
    if (!schema) {
      host.innerHTML = '<div class="err">no schema for slot 0x' +
                        state.slot.toString(16).toUpperCase() + '</div>';
      return;
    }
    const patchFields = getPatchFields(state.slot, state.difficulty) || {};

    let html = '<div class="mdsl-mob-header">';
    html += '<span class="mdsl-mob-slot">slot 0x' +
            state.slot.toString(16).padStart(2, "0").toUpperCase() + '</span> ';
    html += '<span class="mdsl-mob-name">' + escapeHtml(schema.name) + '</span> ';
    html += '<span class="mdsl-mob-diff dim">' +
            ["Normal", "Hard", "VeryHard", "Ultimate"][state.difficulty] +
            ' / ' + escapeHtml(state.variant) + '</span>';
    if (schema.notes) {
      html += '<div class="mdsl-mob-notes dim">' + escapeHtml(schema.notes) + '</div>';
    }
    html += '</div>';

    // Group fields by FieldSpec.group
    const groups = {};
    for (const f of schema.fields) {
      if (!groups[f.group]) groups[f.group] = [];
      groups[f.group].push(f);
    }
    const groupOrder = ["Movement", "Combat", "AI Behavior", "Stats", "Resists", "Other"];
    for (const grp of groupOrder) {
      if (!groups[grp]) continue;
      html += '<div class="mdsl-group">';
      html += '<div class="mdsl-group-title">' + escapeHtml(grp) + '</div>';
      html += '<div class="mdsl-group-fields">';
      for (const f of groups[grp]) {
        const isChanged = (f.label in patchFields);
        const cur = isChanged ? patchFields[f.label] : (f.default == null ? "" : f.default);
        const tooltip = f.tooltip || "";
        const suffix = fieldKindHint(f.kind);
        html += '<label class="mdsl-field' + (isChanged ? " changed" : "") + '"' +
                (tooltip ? ' title="' + escapeHtml(tooltip) + '"' : '') + '>';
        html += '<span class="mdsl-field-label">' + escapeHtml(f.label) + '</span>';
        html += '<span class="mdsl-field-binary dim" title="binary mapping: ' +
                escapeHtml(f.binary_group + '.' + f.binary_name + ' (' + f.kind + ')') +
                '">' + escapeHtml(f.binary_group + '.' + f.binary_name) + '</span>';
        html += '<input type="number" data-label="' + escapeHtml(f.label) +
                '" data-kind="' + escapeHtml(f.kind) +
                '" value="' + (cur === "" ? "" : escapeHtml(String(cur))) + '"' +
                ' step="' + fieldStep(f.kind) + '"' +
                ' placeholder="(unset — keeps stock)" />';
        if (suffix) {
          html += '<span class="mdsl-field-suffix dim">' + escapeHtml(suffix) + '</span>';
        }
        if (isChanged) {
          html += '<button type="button" class="mdsl-field-revert ghost" data-label="' +
                  escapeHtml(f.label) + '" title="clear this edit">x</button>';
        }
        html += '</label>';
      }
      html += '</div></div>';
    }

    host.innerHTML = html;
    // wire inputs
    host.querySelectorAll("input[data-label]").forEach(function (inp) {
      // Cross-tool undo bus (2026-04-25): capture the field's value
      // when the user starts editing so we can build a {before, after}
      // closure on commit. We bind on focus + change rather than every
      // input event so the deque doesn't fill up with per-keystroke
      // entries (the user expects "one Ctrl+Z = one field commit").
      inp.addEventListener("focus", function () {
        inp._psoUbBefore = inp.value;
      });
      inp.addEventListener("change", function () {
        if (!window.psoUndoBus) return;
        if (inp._psoUbBefore == null) return;
        const before = inp._psoUbBefore;
        const after = inp.value;
        if (before === after) return;
        const slot = state.slot;
        const difficulty = state.difficulty;
        const label = inp.dataset.label;
        const kind = inp.dataset.kind;
        const schema = state.schemas && state.schemas[slot];
        const mobLabel = schema && schema.name ? schema.name : ("slot " + slot);
        const apply = function (raw) {
          inp.value = raw;
          const txt = String(raw).trim();
          if (txt === "") {
            clearPatchField(slot, difficulty, label);
          } else {
            let v;
            if (kind === "float" || kind === "percent" || kind === "duration_seconds") {
              v = parseFloat(txt);
              if (!isFinite(v)) return;
            } else {
              v = parseInt(txt, 10);
              if (!isFinite(v)) return;
            }
            setPatchField(slot, difficulty, label, v);
          }
          const lbEl = inp.closest(".mdsl-field");
          if (lbEl) {
            const isChanged = (label in (getPatchFields(slot, difficulty) || {}));
            lbEl.classList.toggle("changed", isChanged);
          }
          renderInspector();
        };
        window.psoUndoBus.push({
          label: "edit " + mobLabel + "/" + label,
          panelId: "mob_dsl",
          undo: function () { apply(before); },
          redo: function () { apply(after); },
        });
        inp._psoUbBefore = after;
      });
      inp.addEventListener("input", function () {
        const label = inp.dataset.label;
        const kind = inp.dataset.kind;
        const txt = inp.value.trim();
        if (txt === "") {
          clearPatchField(state.slot, state.difficulty, label);
        } else {
          let v;
          if (kind === "float" || kind === "percent" ||
              kind === "duration_seconds") {
            v = parseFloat(txt);
            if (!isFinite(v)) return;
          } else {
            v = parseInt(txt, 10);
            if (!isFinite(v)) return;
          }
          setPatchField(state.slot, state.difficulty, label, v);
        }
        // Live update CSS class without full re-render so the cursor stays.
        const labelEl = inp.closest(".mdsl-field");
        if (labelEl) {
          const isChanged = (label in (getPatchFields(state.slot, state.difficulty) || {}));
          labelEl.classList.toggle("changed", isChanged);
        }
        renderInspector();
      });
    });
    host.querySelectorAll(".mdsl-field-revert").forEach(function (btn) {
      btn.addEventListener("click", function (e) {
        e.preventDefault();
        const label = btn.dataset.label;
        clearPatchField(state.slot, state.difficulty, label);
        renderEditor();
        renderInspector();
      });
    });
    renderInspector();
  }

  function renderInspector() {
    const host = state._inspectorHost;
    if (!host) return;
    if (!state.schemas) {
      host.innerHTML = '<div class="vp-insp-help dim">loading...</div>';
      return;
    }
    const total = totalChangedFields();
    const payload = buildPatchPayload();
    let html = '<div class="vp-insp-title">Mob AI Authoring (Tier 1)</div>';
    html += '<div class="vp-insp-help dim">' +
            'Pick a mob + difficulty, edit named fields. Each field maps ' +
            'back to a BattleParamEntry slot. Click "compile + deploy" to ' +
            'apply your patches to the chosen variant and push to newserv.' +
            '</div>';
    html += '<div class="vp-insp-section mdsl-source">';
    html += '<dl class="mdsl-meta">';
    html += '<dt>variant</dt><dd>' + escapeHtml(state.variant) + '</dd>';
    html += '<dt>changes</dt><dd>' + total + ' field(s) across ' +
            payload.length + ' (mob, difficulty) tuple(s)</dd>';
    html += '</dl></div>';
    if (payload.length) {
      html += '<div class="vp-insp-section">';
      html += '<div class="mdsl-changes-title">pending edits</div>';
      html += '<ul class="mdsl-changes">';
      for (const p of payload) {
        html += '<li><strong>' + escapeHtml(p.mob) + '</strong> ' +
                '<span class="dim">' + escapeHtml(p.difficulty) + '</span><ul>';
        const keys = Object.keys(p.fields).sort();
        for (const k of keys) {
          html += '<li>' + escapeHtml(k) + ': <span class="mdsl-val">' +
                  escapeHtml(String(p.fields[k])) + '</span></li>';
        }
        html += '</ul></li>';
      }
      html += '</ul></div>';
      html += '<div class="vp-insp-section mdsl-source-block">';
      html += '<div class="mdsl-changes-title">DSL JSON (wire format)</div>';
      html += '<pre class="mdsl-dsl-source">' +
              escapeHtml(JSON.stringify({ mobs: payload }, null, 2)) +
              '</pre></div>';
    }
    host.innerHTML = html;
  }

  // ------------------------------------------------------------------
  // Actions
  // ------------------------------------------------------------------
  async function onApplyPreset() {
    const sel = document.getElementById("mdslPreset");
    if (!sel || !sel.value) {
      setStatus("pick a preset first", true);
      return;
    }
    const name = sel.value;
    const schema = state.schemas[state.slot];
    if (!schema) return;
    setStatus("loading preset " + name + "...");
    try {
      const r = await getPresetForMob(schema.name, name);
      if (!r.patches || !r.patches.length) {
        setStatus("preset " + name + " has no patches for " + schema.name, true);
        return;
      }
      // Merge fields from each patch onto our state. We keep this
      // simple: every patch in the preset contributes to whichever
      // (slot, difficulty) tuple it names.
      let count = 0;
      const diffNames = ["normal", "hard", "veryhard", "ultimate"];
      for (const p of r.patches) {
        let diffs = [0, 1, 2, 3];
        const d = (p.difficulty || "all").toString().toLowerCase();
        if (d !== "all" && d !== "") {
          const idx = diffNames.indexOf(d);
          if (idx >= 0) diffs = [idx];
        }
        for (const di of diffs) {
          for (const k of Object.keys(p.fields || {})) {
            setPatchField(state.slot, di, k, p.fields[k]);
            count++;
          }
        }
      }
      renderEditor();
      setStatus("applied " + count + " field(s) from " + r.title);
    } catch (e) {
      setStatus("preset failed: " + e.message, true);
    }
  }

  function onClearMob() {
    if (!state.patches[state.slot]) {
      setStatus("nothing to clear", false);
      return;
    }
    if (state.patches[state.slot][state.difficulty]) {
      delete state.patches[state.slot][state.difficulty];
    }
    if (Object.keys(state.patches[state.slot]).length === 0) {
      delete state.patches[state.slot];
    }
    renderEditor();
    setStatus("cleared mob+difficulty edits");
  }

  function onClearAll() {
    if (totalChangedFields() === 0) {
      setStatus("nothing to clear");
      return;
    }
    if (!confirm("Clear ALL pending edits across all mobs?")) return;
    clearAllPatches();
    renderEditor();
    setStatus("cleared all edits");
  }

  async function onCompileDeploy() {
    const payload = buildPatchPayload();
    if (!payload.length) {
      setStatus("no edits to compile", true);
      return;
    }
    if (!confirm("Compile " + payload.length + " (mob, difficulty) tuples + " +
                 totalChangedFields() + " field(s), then deploy to newserv variant " +
                 state.variant + "?\n\nA timestamped backup of the existing " +
                 "BattleParamEntry will be created.")) return;
    setStatus("compiling...");
    try {
      const c = await compile(state.variant, payload, true);
      setStatus("compiled (" + c.patches_applied + " patches, " + c.size +
                " bytes md5=" + c.md5.slice(0, 12) + "...) deploying...");
      const d = await deployVariant(state.variant);
      setStatus("deployed: " + d.deployed_to +
                (d.backup ? " (backup " + d.backup + ")" : ""));
    } catch (e) {
      setStatus("compile/deploy failed: " + e.message, true);
    }
  }

  // Live Test: compile patches (stage=true) → POST /api/live_test → newserv
  // gets the new BattleParamEntry + an attempt at `reload patch-indexes`.
  // The PSOLiveTest module owns the per-panel status pip and recent-action
  // log; this function only handles the compile prefix and the trigger.
  async function onLiveTest() {
    const payload = buildPatchPayload();
    if (!payload.length) {
      setStatus("no edits to live-test", true);
      window.PSOLiveTest && window.PSOLiveTest.setPipState(
        "mob-dsl", "failed", "no edits");
      return;
    }
    setStatus("compiling for live test...");
    try {
      const c = await compile(state.variant, payload, true);
      setStatus("compiled " + c.size + " bytes; pushing to newserv...");
      const result = await window.PSOLiveTest.triggerLiveTest("mob_dsl", {
        panelId: "mob-dsl",
        body: { variant: state.variant, attempt_reload: true },
      });
      if (result.ok === false) {
        setStatus("live-test failed: " + (result.error || "unknown"), true);
        return;
      }
      const dep = (result.deployed && result.deployed.deployed_to) || "";
      const requires = result.requires_manual_reload
        ? " (manual newserv reload required)" : "";
      setStatus("live: " + dep + requires);
    } catch (e) {
      setStatus("live-test failed: " + e.message, true);
      window.PSOLiveTest && window.PSOLiveTest.setPipState(
        "mob-dsl", "failed", "compile failed: " + e.message);
    }
  }

  // ------------------------------------------------------------------
  // Perspective registration
  // ------------------------------------------------------------------
  window.PSOPerspectives.register("mob-ai-authoring", {
    label: "Mob AI Authoring",
    match: function (entry, file) {
      // Score 100 only if context is explicitly mob-ai-authoring (header
      // button); else 0 (hidden from auto-route).
      if (entry && entry.category === "mob-ai") return 100;
      return 0;
    },
    mount: async function (stage, insp, ctx) {
      stage.innerHTML =
        '<div class="mdsl-perspective">' +
        '<div id="mdslToolbar"></div>' +
        '<div id="mdslEditor" class="mdsl-editor"></div>' +
        '</div>';
      state._inspectorHost = insp;

      // Lazy-load schemas + presets
      try {
        if (!state.schemas) {
          const r = await getSchemas();
          state.schemas = {};
          for (const s of r.schemas) {
            state.schemas[s.slot] = s;
          }
          // Show coverage in the inspector for transparency
          state._coverage = r.coverage || null;
        }
        if (!state.presets) {
          const r = await getPresets();
          state.presets = r.presets || [];
        }
      } catch (e) {
        stage.innerHTML = '<div class="err">' +
                          escapeHtml("init failed: " + e.message) + '</div>';
        return;
      }
      renderToolbar(stage.querySelector("#mdslToolbar"));
      renderEditor();
      renderInspector();
    },
    unmount: function (stage, insp) {
      state._inspectorHost = null;
    },
  });

  // ------------------------------------------------------------------
  // Header button: synthesize a mob-ai context and open
  // ------------------------------------------------------------------
  function openPerspective() {
    const ctx = {
      path: "__mob_ai__",
      entry: { category: "mob-ai", format: "MobAIDsl" },
      fileName: "MobAIDsl",
    };
    if (window.PSOPerspectives && window.PSOPerspectives.switchTo) {
      window.PSOPerspectives.switchTo("mob-ai-authoring", ctx);
    }
  }

  function ensureHeaderButton() {
    if (document.getElementById("btnMobAi")) return;
    const status = document.getElementById("status");
    const header = status ? status.parentNode : null;
    if (!header) return;
    const btn = document.createElement("button");
    btn.id = "btnMobAi";
    btn.type = "button";
    btn.className = "ghost";
    btn.title = "edit mob AI tuning (named fields over BattleParamEntry)";
    btn.textContent = "Mob AI";
    header.insertBefore(btn, status);
    btn.addEventListener("click", openPerspective);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", ensureHeaderButton);
  } else {
    ensureHeaderButton();
  }
})();
