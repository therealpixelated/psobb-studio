// =====================================================================
// PSOBB Texture Editor - Battle Params perspective (2026-04-25)
//
// Edits BattleParamEntry*.dat (newserv) variants. The perspective is
// "global" - it doesn't bind to a specific asset entry, so the score is
// 1 (always available as a tab). A header button (#btnBattleParams)
// synthesizes a context and switches to it.
//
// Layout:
//   - Top bar:    variant picker (on/off/lab_on/lab_off/ep4_on/ep4_off),
//                 difficulty picker (Normal/Hard/VeryHard/Ultimate),
//                 mob picker (96 slots searchable by name),
//                 reload + import-from-newserv + export-to-newserv buttons.
//   - Main panel: three tabs (Stats / Attacks / Resists / Animations)
//                 with named editors per field.
//   - Inspector:  diff summary (changed fields vs original) + deploy
//                 button.
//
// Wiring: GET /api/battle_param/config returns the newserv path and
// variants. GET /api/battle_param/<variant> returns the parsed JSON.
// POST writes to staging. /api/battle_param/<variant>/deploy promotes
// the staged file into newserv (creating a backup).
//
// The perspective stores per-variant state in module-scope `state` so
// switching variants doesn't lose unsaved edits.
// =====================================================================

(function () {
  "use strict";

  if (!window.PSOPerspectives) {
    console.warn("[battle_param] perspectives.js not loaded yet");
    return;
  }

  // Per-perspective state. Cleared when the user clicks "Reload" or
  // when the editor explicitly reloads from newserv.
  const state = {
    config: null,         // GET /api/battle_param/config response
    variant: "on",        // active variant
    difficulty: 0,        // 0..3 (Normal..Ultimate)
    slot: 0x4B,           // active mob slot (0x4B = Booma default)
    original: null,       // last-loaded server data, deep-cloned
    edited: null,         // deep-clone for editing
    slotNames: {},        // { 0x4B: "Booma", ... }
  };

  // Field metadata per record type. Keys map to underlying JSON keys.
  // type: "int" | "uint" | "float"   range: [min, max]
  const FIELD_META = {
    stats: [
      { key: "atp", label: "ATP", type: "int", range: [-32768, 32767], help: "base attack power" },
      { key: "mst", label: "MST", type: "int", range: [-32768, 32767], help: "mind / tech power" },
      { key: "evp", label: "EVP", type: "int", range: [-32768, 32767], help: "evasion" },
      { key: "hp",  label: "HP",  type: "int", range: [-32768, 32767], help: "max HP (signed in stock files)" },
      { key: "dfp", label: "DFP", type: "int", range: [-32768, 32767], help: "defense" },
      { key: "ata", label: "ATA", type: "int", range: [-32768, 32767], help: "accuracy" },
      { key: "lck", label: "LCK", type: "int", range: [-32768, 32767], help: "luck" },
      { key: "esp", label: "ESP", type: "int", range: [-32768, 32767], help: "esp" },
      { key: "hp_modifier",   label: "HP modifier",   type: "float", help: "BB Patch field_0x10" },
      { key: "dfp_modifier",  label: "DFP modifier",  type: "float", help: "BB Patch field_0x14" },
      { key: "hp_mst_modifier", label: "HP/MST mod",  type: "int",   help: "BB Patch unknown" },
      { key: "xp",  label: "XP",  type: "int", range: [-32768, 32767], help: "exp dropped" },
      { key: "field_0x1e", label: "reserved 0x1e", type: "int", range: [-32768, 32767], advanced: true },
      { key: "field_0x20", label: "reserved 0x20", type: "int", range: [-32768, 32767], advanced: true },
      { key: "field_0x22", label: "reserved 0x22", type: "int", range: [-32768, 32767], advanced: true },
    ],
    attacks: [
      { key: "min_atp", label: "min ATP", type: "int" },
      { key: "max_atp", label: "max ATP", type: "int" },
      { key: "min_ata", label: "min ATA", type: "int" },
      { key: "max_ata", label: "max ATA", type: "int" },
      { key: "distance_x", label: "distance x", type: "float", help: "attack reach (units?)" },
      { key: "angle", label: "angle (bams)", type: "uint", help: "0x10000 = full revolution" },
      { key: "distance_y", label: "distance y", type: "float" },
      { key: "unknown_a8",  label: "reserved a8",  type: "uint", advanced: true },
      { key: "unknown_a9",  label: "reserved a9",  type: "uint", advanced: true },
      { key: "unknown_a10", label: "reserved a10", type: "uint", advanced: true },
      { key: "unknown_a11", label: "reserved a11", type: "uint", advanced: true },
      { key: "unknown_a12", label: "reserved a12", type: "uint", advanced: true },
      { key: "unknown_a13", label: "reserved a13", type: "uint", advanced: true },
      { key: "unknown_a14", label: "reserved a14", type: "uint", advanced: true },
      { key: "unknown_a15", label: "reserved a15", type: "uint", advanced: true },
      { key: "unknown_a16", label: "reserved a16", type: "uint", advanced: true },
    ],
    resists: [
      { key: "evp_bonus", label: "EVP bonus",   type: "int" },
      { key: "efr",  label: "EFR (fire)",       type: "uint" },
      { key: "eic",  label: "EIC (ice)",        type: "uint" },
      { key: "eth",  label: "ETH (thunder)",    type: "uint" },
      { key: "elt",  label: "ELT (light)",      type: "uint" },
      { key: "edk",  label: "EDK (dark)",       type: "uint" },
      { key: "unknown_a6", label: "reserved a6", type: "uint", advanced: true },
      { key: "unknown_a7", label: "reserved a7", type: "uint", advanced: true },
      { key: "unknown_a8", label: "reserved a8", type: "uint", advanced: true },
      { key: "unknown_a9", label: "reserved a9", type: "uint", advanced: true },
      { key: "dfp_bonus", label: "DFP bonus", type: "int" },
    ],
    animations: [
      { key: "fparam1", label: "fparam1", type: "float", help: "varies by mob (see notes/movement-data.txt)" },
      { key: "fparam2", label: "fparam2", type: "float" },
      { key: "fparam3", label: "fparam3", type: "float" },
      { key: "fparam4", label: "fparam4", type: "float" },
      { key: "fparam5", label: "fparam5", type: "float" },
      { key: "fparam6", label: "fparam6", type: "float" },
      { key: "iparam1", label: "iparam1", type: "uint" },
      { key: "iparam2", label: "iparam2", type: "uint" },
      { key: "iparam3", label: "iparam3", type: "uint" },
      { key: "iparam4", label: "iparam4", type: "uint" },
      { key: "iparam5", label: "iparam5", type: "uint" },
      { key: "iparam6", label: "iparam6", type: "uint" },
    ],
  };

  // Per-mob field overrides for animations. Lifts named labels per
  // movement-data.txt. Falls back to the generic fparamN/iparamN
  // labels when slot has no override.
  const ANIM_LABELS_BY_SLOT = {
    // Booma family (0x4B) - identical for slots 0x4B..0x4D
    0x4B: {
      fparam1: "idle move speed",
      fparam2: "idle anim speed",
      fparam3: "engaged move speed",
      fparam4: "engaged anim speed",
      fparam5: "poison cloud dmg (Merillia)",
      fparam6: "run-away speed",
      iparam1: "low-HP threshold (0-100)",
    },
    0x4C: { /* same as 0x4B */ },
    0x4D: { /* same as 0x4B */ },
    // Hildebear/Hildeblue (0x49 / 0x4A)
    0x49: {
      fparam1: "punch attack speed",
      fparam2: "tech range",
      fparam3: "movement speed",
      fparam4: "walking anim speed",
    },
    0x4A: { /* same as 0x49 */ },
    // De Rol Le / Barba Ray (0x0F)
    0x0F: {
      fparam1: "damage (some attack)",
      fparam3: "mine explosion damage",
      fparam4: "X-position randomisation",
      fparam5: "X-pos rng probability",
      iparam1: "total HP",
      iparam2: "armor break HP threshold",
      iparam3: "mask removal HP threshold",
    },
  };

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;",
                '"': "&quot;", "'": "&#39;" })[c];
    });
  }

  // Last path segment, for both / and \ separators. Keeps absolute dev
  // paths (C:/Users/...) out of user-facing displays.
  function basename(p) {
    if (p == null) return "";
    var s = String(p).replace(/[\\/]+$/, "");
    var i = Math.max(s.lastIndexOf("/"), s.lastIndexOf("\\"));
    return i >= 0 ? s.slice(i + 1) : s;
  }

  function deepClone(o) { return JSON.parse(JSON.stringify(o)); }

  // ------------------------------------------------------------------
  // API helpers
  // ------------------------------------------------------------------
  async function getConfig() {
    const r = await fetch("/api/battle_param/config");
    if (!r.ok) throw new Error("config: " + r.status);
    return r.json();
  }
  async function getSlots() {
    const r = await fetch("/api/battle_param/slots");
    if (!r.ok) throw new Error("slots: " + r.status);
    return r.json();
  }
  async function getVariant(variant, source) {
    const url = "/api/battle_param/" + encodeURIComponent(variant) +
                (source ? "?source=" + encodeURIComponent(source) : "");
    const r = await fetch(url);
    if (!r.ok) {
      const detail = await r.text().catch(() => "");
      throw new Error("get " + variant + " (" + r.status + "): " + detail);
    }
    return r.json();
  }
  async function postVariant(variant, data) {
    const r = await fetch("/api/battle_param/" + encodeURIComponent(variant), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: data }),
    });
    if (!r.ok) {
      const detail = await r.text().catch(() => "");
      throw new Error("post " + variant + " (" + r.status + "): " + detail);
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
      throw new Error("deploy " + variant + " (" + r.status + "): " + detail);
    }
    return r.json();
  }

  // ------------------------------------------------------------------
  // UI render
  // ------------------------------------------------------------------
  function renderToolbar(hostEl) {
    const variants = (state.config && state.config.variants) || ["on"];
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
    // state.slotNames keys are stringified decimal integers (JS object
    // keys are always strings). Parse as base 10, NOT 16.
    const slotEntries = Object.entries(state.slotNames || {}).map(function (kv) {
      const idx = parseInt(kv[0], 10);
      return [idx, kv[1]];
    }).sort(function (a, b) { return a[0] - b[0]; });
    const slotOpts = slotEntries.map(function (kv) {
      const v = kv[0], n = kv[1];
      return '<option value="' + v + '"' +
             (v === state.slot ? " selected" : "") + ">0x" +
             v.toString(16).padStart(2, "0").toUpperCase() + "  " +
             escapeHtml(n) + "</option>";
    }).join("");

    hostEl.innerHTML =
      '<div class="bp-toolbar">' +
      '<label>variant <select id="bpVariant">' + variantOpts + '</select></label>' +
      '<label>difficulty <select id="bpDifficulty">' + diffOpts + '</select></label>' +
      '<label>mob <select id="bpSlot">' + slotOpts + '</select></label>' +
      '<button type="button" id="bpReload" class="ghost" title="reload from newserv (discards in-progress edits)">reload</button>' +
      '<button type="button" id="bpExport" title="serialize edits to staging (cache/battle_param_export/)">export to staging</button>' +
      '<button type="button" id="bpDeploy" class="warn" title="copy staged file to the newserv install (with backup)">deploy to newserv</button>' +
      '<span class="dim" id="bpStatus"></span>' +
      '</div>';

    hostEl.querySelector("#bpVariant").addEventListener("change", function (e) {
      state.variant = e.target.value;
      reloadFromServer();
    });
    hostEl.querySelector("#bpDifficulty").addEventListener("change", function (e) {
      state.difficulty = parseInt(e.target.value, 10) || 0;
      renderEditor();
    });
    hostEl.querySelector("#bpSlot").addEventListener("change", function (e) {
      state.slot = parseInt(e.target.value, 10) || 0;
      renderEditor();
    });
    hostEl.querySelector("#bpReload").addEventListener("click", reloadFromServer);
    hostEl.querySelector("#bpExport").addEventListener("click", onExport);
    hostEl.querySelector("#bpDeploy").addEventListener("click", onDeploy);

    // Live Test button (2026-04-25). Stages-then-deploys to newserv with
    // an attempt at `reload patch-indexes`; surfaces a status pip + the
    // last 3 actions. Mounted between bpDeploy and bpStatus so it sits
    // next to the warn-deploy button (visual hierarchy: red dot = live).
    if (window.PSOLiveTest) {
      const toolbar = hostEl.querySelector(".bp-toolbar");
      const statusEl = hostEl.querySelector("#bpStatus");
      window.PSOLiveTest.ensureLiveButton({
        host: toolbar,
        beforeNode: statusEl,
        panelId: "battle-param",
        kind: "battle_param",
        title: "stage current edits → push to newserv (live)",
      });
      // Replace auto-handler with our compile-prefix + live trigger.
      const ltBtn = toolbar.querySelector("#ltBtn_battle-param");
      if (ltBtn) {
        const newBtn = ltBtn.cloneNode(true);
        ltBtn.parentNode.replaceChild(newBtn, ltBtn);
        newBtn.addEventListener("click", onLiveTest);
      }
      window.PSOLiveTest.attachPip(toolbar, "battle-param");
    }
  }

  function setStatus(msg, isErr) {
    const el = document.getElementById("bpStatus");
    if (!el) return;
    el.textContent = msg || "";
    el.classList.toggle("err", !!isErr);
  }

  function renderEditor() {
    const host = document.getElementById("bpEditor");
    if (!host) return;
    if (!state.edited) {
      host.innerHTML = '<div class="dim">load a variant to edit.</div>';
      return;
    }
    const diff = state.edited.difficulties[state.difficulty];
    if (!diff) {
      host.innerHTML = '<div class="err">missing difficulty ' + state.difficulty + '</div>';
      return;
    }
    const ent = diff.entries[state.slot];
    if (!ent) {
      host.innerHTML = '<div class="err">missing slot 0x' + state.slot.toString(16) + '</div>';
      return;
    }
    const origDiff = state.original.difficulties[state.difficulty];
    const origEnt = origDiff && origDiff.entries[state.slot];

    let html = '<div class="bp-mob-header">';
    html += '<span class="bp-mob-slot">slot 0x' + state.slot.toString(16).padStart(2, "0").toUpperCase() + '</span> ';
    html += '<span class="bp-mob-name">' + escapeHtml(ent.name || "") + '</span>';
    html += '</div>';

    // Render one editable field. Kept editable for both the named and the
    // de-emphasized "reserved" fields; only the grouping differs.
    function renderField(group, m) {
      const v = ent[group][m.key];
      const ov = origEnt ? origEnt[group][m.key] : null;
      const changed = (v !== ov);
      const labelOverride = (group === "animations" &&
                             ANIM_LABELS_BY_SLOT[state.slot] &&
                             ANIM_LABELS_BY_SLOT[state.slot][m.key])
                            ? ANIM_LABELS_BY_SLOT[state.slot][m.key]
                            : null;
      let s = '<label class="bp-field' + (changed ? " changed" : "") + '"' +
              (m.help ? ' title="' + escapeHtml(m.help) + '"' : '') + '>';
      s += '<span class="bp-field-label">' +
           escapeHtml(m.label + (labelOverride ? " (" + labelOverride + ")" : "")) +
           '</span>';
      s += '<input type="number" data-group="' + group + '" data-key="' + m.key +
           '" value="' + (v == null ? "" : v) +
           '" step="' + (m.type === "float" ? "any" : "1") + '" />';
      if (changed) {
        s += '<span class="bp-orig dim" title="original">' +
             (ov == null ? "(null)" : escapeHtml(String(ov))) + '</span>';
      }
      s += '</label>';
      return s;
    }

    const groups = ["stats", "attacks", "resists", "animations"];
    for (const group of groups) {
      const meta = FIELD_META[group];
      const named = meta.filter(function (m) { return !m.advanced; });
      const advanced = meta.filter(function (m) { return m.advanced; });

      html += '<div class="bp-group">';
      html += '<div class="bp-group-title">' + group + '</div>';
      html += '<div class="bp-group-fields">';
      for (const m of named) html += renderField(group, m);
      html += '</div>';

      // Truly-unknown RE bytes: kept editable but collapsed by default so
      // they don't read as broken. Highlight the summary if any have edits.
      if (advanced.length) {
        const anyChanged = advanced.some(function (m) {
          const v = ent[group][m.key];
          const ov = origEnt ? origEnt[group][m.key] : null;
          return v !== ov;
        });
        html += '<details class="bp-advanced"' + (anyChanged ? ' open' : '') + '>';
        html += '<summary class="dim">Advanced / unknown bytes (' +
                advanced.length + ')</summary>';
        html += '<div class="bp-group-fields">';
        for (const m of advanced) html += renderField(group, m);
        html += '</div></details>';
      }
      html += '</div>';
    }

    host.innerHTML = html;
    // wire inputs
    host.querySelectorAll("input[data-group]").forEach(function (inp) {
      inp.addEventListener("input", function () {
        const group = inp.dataset.group;
        const key = inp.dataset.key;
        const meta = FIELD_META[group].find(function (m) { return m.key === key; });
        let v;
        if (meta && meta.type === "float") {
          v = parseFloat(inp.value);
          if (!isFinite(v)) v = 0;
        } else {
          v = parseInt(inp.value, 10);
          if (!isFinite(v)) v = 0;
        }
        ent[group][key] = v;
        // Drop the bits sidecar so the new value is what gets serialized.
        delete ent[group]["_" + key + "_bits"];
        // Update visual diff state
        const origVal = origEnt ? origEnt[group][key] : null;
        const labelEl = inp.closest(".bp-field");
        if (labelEl) labelEl.classList.toggle("changed", v !== origVal);
        renderInspector();
      });

      // Cross-tool undo bus integration (2026-04-25, C7 follow-up).
      // Same focus-then-change pattern as mob_dsl_panel: snapshot the
      // value entering the field, then push a bus entry on commit so
      // Ctrl+Z from anywhere reverts THIS edit. Per-keystroke `input`
      // events deliberately stay outside the bus — one bus entry per
      // user-visible field commit (focus → blur or focus → Enter).
      inp.addEventListener("focus", function () {
        // Capture the underlying typed value, not inp.value (text), so
        // the undo restoration round-trips back to the same JS number.
        const group = inp.dataset.group;
        const key = inp.dataset.key;
        inp._psoUbBefore = ent[group][key];
        inp._psoUbBeforeText = inp.value;
        inp._psoUbSlot = state.slot;
        inp._psoUbDifficulty = state.difficulty;
      });
      inp.addEventListener("change", function () {
        if (!window.psoUndoBus) return;
        if (inp._psoUbBefore === undefined) return;
        const group = inp.dataset.group;
        const key = inp.dataset.key;
        const before = inp._psoUbBefore;
        const after = ent[group][key];
        if (before === after) return;
        // Capture context at commit time — the user might switch slot
        // before they Ctrl+Z, so the closure must hold the slot/diff
        // that this edit ACTUALLY belongs to.
        const slotAtCommit = inp._psoUbSlot != null ? inp._psoUbSlot : state.slot;
        const diffAtCommit = inp._psoUbDifficulty != null ? inp._psoUbDifficulty : state.difficulty;
        const meta = FIELD_META[group].find(function (m) { return m.key === key; });
        const fieldLabel = (meta && meta.label) || key;
        const slotName = (state.slotNames && state.slotNames[slotAtCommit]) || ("slot 0x" + slotAtCommit.toString(16));
        const apply = function (val) {
          // Restore the underlying record value …
          const diff = state.edited && state.edited.difficulties[diffAtCommit];
          const e = diff && diff.entries[slotAtCommit];
          if (!e || !e[group]) return;
          e[group][key] = val;
          delete e[group]["_" + key + "_bits"];
          // … snap the editor back to the matching slot/difficulty if
          // the user has navigated away …
          let needsRender = false;
          if (state.slot !== slotAtCommit) {
            state.slot = slotAtCommit;
            needsRender = true;
            const slotSel = document.getElementById("bpSlot");
            if (slotSel) slotSel.value = String(slotAtCommit);
          }
          if (state.difficulty !== diffAtCommit) {
            state.difficulty = diffAtCommit;
            needsRender = true;
            const diffSel = document.getElementById("bpDifficulty");
            if (diffSel) diffSel.value = String(diffAtCommit);
          }
          if (needsRender) {
            renderEditor();
          } else {
            // Same view — just update the input + diff classes.
            inp.value = val == null ? "" : String(val);
            const origDiff2 = state.original && state.original.difficulties[diffAtCommit];
            const origEnt2 = origDiff2 && origDiff2.entries[slotAtCommit];
            const origVal = origEnt2 ? origEnt2[group][key] : null;
            const labelEl = inp.closest(".bp-field");
            if (labelEl) labelEl.classList.toggle("changed", val !== origVal);
            renderInspector();
          }
        };
        window.psoUndoBus.push({
          label: "edit " + slotName + "/" + fieldLabel,
          panelId: "battle_param",
          undo: function () { apply(before); },
          redo: function () { apply(after); },
        });
        // Re-prime so a subsequent commit on the same focus session
        // (e.g. Tab between fields without leaving the row) works.
        inp._psoUbBefore = after;
        inp._psoUbBeforeText = inp.value;
      });
    });
    renderInspector();
  }

  function renderInspector() {
    const host = state._inspectorHost;
    if (!host) return;
    if (!state.edited || !state.original) {
      host.innerHTML = '<div class="vp-insp-help dim">no variant loaded.</div>';
      return;
    }
    const changes = computeChanges();
    let html = '<div class="vp-insp-title">Battle Params</div>';
    html += '<div class="vp-insp-help dim">' +
            'Edit the BattleParamEntry slot for the active mob. Click ' +
            '"export to staging" to serialize and review; "deploy to newserv" ' +
            'copies the staged file (with timestamped backup).' +
            '</div>';
    html += '<div class="vp-insp-section bp-source">';
    html += '<dl class="bp-meta"><dt>variant</dt><dd>' + escapeHtml(state.variant) + '</dd>';
    if (state.config && state.config.newserv_dir) {
      // Show only the basename — never leak the absolute dev path. The full
      // path is still discoverable via the title tooltip for power users.
      var srcName = basename(state.config.newserv_dir) || "newserv data";
      html += '<dt>source</dt><dd class="dim wrap" title="' +
              escapeHtml(srcName) + '">' + escapeHtml(srcName) + '</dd>';
    }
    html += '<dt>changes</dt><dd>' + changes.length + '</dd>';
    html += '</dl></div>';
    if (changes.length) {
      html += '<div class="vp-insp-section">';
      html += '<div class="bp-changes-title">changed fields</div>';
      html += '<ul class="bp-changes">';
      for (const c of changes.slice(0, 50)) {
        html += '<li>' + escapeHtml(c.label) + ': ' +
                escapeHtml(String(c.from)) + ' -> ' +
                escapeHtml(String(c.to)) + '</li>';
      }
      if (changes.length > 50) {
        html += '<li class="dim">... ' + (changes.length - 50) + ' more</li>';
      }
      html += '</ul></div>';
    }
    host.innerHTML = html;
  }

  function computeChanges() {
    if (!state.original || !state.edited) return [];
    const diffs = [];
    for (let d = 0; d < 4; d++) {
      const orig = state.original.difficulties[d];
      const ed = state.edited.difficulties[d];
      if (!orig || !ed) continue;
      for (let s = 0; s < ed.entries.length; s++) {
        const o = orig.entries[s];
        const e = ed.entries[s];
        if (!o || !e) continue;
        for (const group of ["stats", "attacks", "resists", "animations"]) {
          const og = o[group], eg = e[group];
          if (!og || !eg) continue;
          for (const k of Object.keys(eg)) {
            if (k.startsWith("_") && k.endsWith("_bits")) continue;
            if (og[k] !== eg[k]) {
              diffs.push({
                difficulty: d,
                slot: s,
                group: group,
                key: k,
                from: og[k],
                to: eg[k],
                label: ["Normal", "Hard", "VeryHard", "Ultimate"][d] +
                       " / 0x" + s.toString(16).padStart(2, "0").toUpperCase() +
                       " " + (e.name || "") + " / " + group + "." + k,
              });
            }
          }
        }
      }
    }
    return diffs;
  }

  // ------------------------------------------------------------------
  // Actions
  // ------------------------------------------------------------------
  async function reloadFromServer() {
    setStatus("loading...");
    try {
      const resp = await getVariant(state.variant, "newserv");
      state.original = deepClone(resp.data);
      state.edited = deepClone(resp.data);
      renderEditor();
      // Basename only — never surface the absolute newserv path in the status.
      setStatus("loaded " + (basename(resp.source_path) || state.variant));
    } catch (e) {
      setStatus("error: " + e.message, true);
    }
  }

  async function onExport() {
    if (!state.edited) {
      setStatus("nothing to export", true);
      return;
    }
    setStatus("staging...");
    try {
      const r = await postVariant(state.variant, state.edited);
      setStatus("staged: " + r.size + " bytes md5=" + r.md5.slice(0, 12) + " ...");
    } catch (e) {
      setStatus("export failed: " + e.message, true);
    }
  }

  async function onDeploy() {
    if (!confirm("Deploy " + state.variant + " to newserv? A timestamped backup will be created.")) return;
    setStatus("deploying...");
    try {
      const r = await deployVariant(state.variant);
      setStatus("deployed: " + r.deployed_to + (r.backup ? " (backup " + r.backup + ")" : ""));
    } catch (e) {
      setStatus("deploy failed: " + e.message, true);
    }
  }

  // Live Test: stage current edits → POST /api/live_test → newserv gets the
  // new BattleParamEntry + an attempt at `reload patch-indexes`. Skips the
  // confirm dialog (button is intentionally fast-path; the action log
  // shows what happened). PSOLiveTest owns the status pip; setStatus
  // continues to surface compile-side messages.
  async function onLiveTest() {
    if (!state.edited) {
      setStatus("nothing to live-test", true);
      window.PSOLiveTest && window.PSOLiveTest.setPipState(
        "battle-param", "failed", "no data");
      return;
    }
    setStatus("staging for live test...");
    try {
      const r = await postVariant(state.variant, state.edited);
      setStatus("staged " + r.size + " bytes; pushing to newserv...");
      const result = await window.PSOLiveTest.triggerLiveTest("battle_param", {
        panelId: "battle-param",
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
        "battle-param", "failed", "stage failed: " + e.message);
    }
  }

  // ------------------------------------------------------------------
  // Perspective registration
  // ------------------------------------------------------------------
  window.PSOPerspectives.register("battle-params", {
    label: "Battle Params",
    match: function (entry, file) {
      // Score 1 if the active asset is anything; this perspective is
      // invoked manually via the header button (we synthesize a ctx
      // there). Score is non-zero so it appears in the tab strip
      // alongside other available views.
      if (entry && entry.category === "battleparam") return 100;
      // Hide by default unless user explicitly opens it via header btn.
      // Returning 0 means it doesn't appear; we override by switching
      // to it programmatically.
      return 0;
    },
    mount: async function (stage, insp, ctx) {
      stage.innerHTML =
        '<div class="bp-perspective">' +
        '<div id="bpToolbar"></div>' +
        '<div id="bpEditor" class="bp-editor"></div>' +
        '</div>';
      state._inspectorHost = insp;
      // Lazy-load config + slots if not cached.
      if (!state.config) {
        try {
          state.config = await getConfig();
        } catch (e) {
          stage.innerHTML = '<div class="err">' + escapeHtml("config error: " + e.message) + '</div>';
          return;
        }
      }
      if (!Object.keys(state.slotNames).length) {
        try {
          const r = await getSlots();
          state.slotNames = {};
          for (const k of Object.keys(r.slots)) {
            state.slotNames[parseInt(k, 16)] = r.slots[k];
          }
        } catch (e) {
          // non-fatal
        }
      }
      renderToolbar(stage.querySelector("#bpToolbar"));
      renderInspector();
      if (!state.edited) {
        await reloadFromServer();
      } else {
        renderEditor();
      }
    },
    unmount: function (stage, insp) {
      state._inspectorHost = null;
    },
  });

  // ------------------------------------------------------------------
  // Header button: synthesize a battleparam context and open
  // ------------------------------------------------------------------
  function openPerspective() {
    const ctx = {
      path: "__battleparam__",
      entry: { category: "battleparam", format: "BattleParamEntry" },
      fileName: "BattleParamEntry.dat",
    };
    if (window.PSOPerspectives && window.PSOPerspectives.switchTo) {
      window.PSOPerspectives.switchTo("battle-params", ctx);
    }
  }

  function ensureHeaderButton() {
    if (document.getElementById("btnBattleParams")) return;
    const status = document.getElementById("status");
    const header = status ? status.parentNode : null;
    if (!header) return;
    const btn = document.createElement("button");
    btn.id = "btnBattleParams";
    btn.type = "button";
    btn.className = "ghost";
    btn.title = "edit BattleParamEntry*.dat (mob stats / attacks / resists / movement)";
    btn.textContent = "Battle Params";
    header.insertBefore(btn, status);
    btn.addEventListener("click", openPerspective);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", ensureHeaderButton);
  } else {
    ensureHeaderButton();
  }
})();
