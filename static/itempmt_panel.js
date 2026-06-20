// =====================================================================
// PSOBB Texture Editor - Item PMT perspective (2026-04-25)
//
// Edits ItemPMT-bb-v4.prs (newserv canonical) / ItemPMT.prs (Booma legacy).
// The perspective is "global" — it doesn't bind to a specific asset entry,
// so the score is 0 for normal contexts and is invoked manually via a
// header button (#btnItemPmt).
//
// Layout:
//   - Top bar:    section picker (Weapons / Armors / Shields / Units /
//                 Mags / Tools / Specials / Mag Feed / Combinations),
//                 sub-class / sub-table picker (when applicable),
//                 entry picker (filtered by ID/type/skin).
//                 Reload + export-to-staging + deploy-to-newserv.
//   - Main panel: grouped field editors per active record.
//   - Inspector:  diff summary (changed fields vs original) + deploy.
//
// Wiring:
//   GET /api/itempmt/config                — newserv path + stage dir
//   GET /api/itempmt[?source=newserv|stage] — parsed JSON
//   POST /api/itempmt {data}                — stage edited PMT (PRS + .prs)
//   POST /api/itempmt/deploy                — promote stage to newserv
// =====================================================================

(function () {
  "use strict";

  if (!window.PSOPerspectives) {
    console.warn("[itempmt] perspectives.js not loaded yet");
    return;
  }

  const state = {
    config: null,
    original: null,
    edited: null,
    section: "weapons",       // active section key
    classIdx: 0,              // weapons[i].class_index OR tools[i]
    subTableIdx: 0,           // mag_feeds[i].table_index
    entryIdx: 0,              // index inside the active list
  };

  // ---- Field metadata (mirrors battle_param style) -------------------
  // Each section has a list of fields with {key, label, type, help}.
  // type: "int" | "uint" | "float" | "hex"
  const FIELD_META = {
    item_base: [
      { key: "id",          label: "id (item code)", type: "hex" },
      { key: "type",        label: "type",  type: "uint", help: "model index (per-class table)" },
      { key: "skin",        label: "skin",  type: "uint", help: "texture variant" },
      { key: "team_points", label: "team points", type: "uint" },
    ],
    weapons: [
      { key: "class_flags",  label: "class flags", type: "hex", help: "u16 bitfield (HUmar/HUnewearl/...)" },
      { key: "atp_min",      label: "ATP min",     type: "int" },
      { key: "atp_max",      label: "ATP max",     type: "int" },
      { key: "atp_required", label: "ATP req",     type: "int" },
      { key: "mst_required", label: "MST req",     type: "int" },
      { key: "ata_required", label: "ATA req",     type: "int" },
      { key: "mst",          label: "MST",         type: "int" },
      { key: "max_grind",    label: "max grind",   type: "uint" },
      { key: "photon",       label: "photon",      type: "uint" },
      { key: "special",      label: "special",     type: "int", help: "-1 = none" },
      { key: "ata",          label: "ATA",         type: "uint" },
      { key: "stat_boost_entry_index", label: "stat boost #", type: "uint" },
      { key: "projectile",   label: "projectile",  type: "uint" },
      { key: "trail1_x",     label: "trail1 x",    type: "int" },
      { key: "trail1_y",     label: "trail1 y",    type: "int" },
      { key: "trail2_x",     label: "trail2 x",    type: "int" },
      { key: "trail2_y",     label: "trail2 y",    type: "int" },
      { key: "color",        label: "color",       type: "uint" },
      { key: "tech_boost",   label: "tech boost",  type: "uint" },
      { key: "behavior_flags", label: "behavior flags", type: "hex" },
      { key: "unknown_a4",   label: "unknown_a4",  type: "uint" },
      { key: "unknown_a5",   label: "unknown_a5",  type: "uint" },
    ],
    armors: [
      { key: "dfp",                    label: "DFP base",      type: "uint" },
      { key: "evp",                    label: "EVP base",      type: "uint" },
      { key: "block_particle",         label: "block particle", type: "uint" },
      { key: "block_effect",           label: "block effect",  type: "uint" },
      { key: "class_flags",            label: "class flags",   type: "hex" },
      { key: "required_level",         label: "required level", type: "uint" },
      { key: "efr",                    label: "EFR (fire)",     type: "uint" },
      { key: "eth",                    label: "ETH (thunder)",  type: "uint" },
      { key: "eic",                    label: "EIC (ice)",      type: "uint" },
      { key: "edk",                    label: "EDK (dark)",     type: "uint" },
      { key: "elt",                    label: "ELT (light)",    type: "uint" },
      { key: "dfp_range",              label: "DFP range",     type: "uint" },
      { key: "evp_range",              label: "EVP range",     type: "uint" },
      { key: "stat_boost_entry_index", label: "stat boost #",  type: "uint" },
      { key: "tech_boost",             label: "tech boost",    type: "uint" },
      { key: "flags_type",             label: "flags type",    type: "uint", help: "0=armor, 1/2/3=variant" },
      { key: "unknown_a4",             label: "unknown_a4",    type: "uint" },
    ],
    shields: [], // same as armors below
    units: [
      { key: "stat",            label: "stat",            type: "uint" },
      { key: "stat_amount",     label: "stat amount",     type: "uint" },
      { key: "modifier_amount", label: "modifier amount", type: "int" },
    ],
    mags: [
      { key: "feed_table",   label: "feed table",   type: "uint" },
      { key: "photon_blast", label: "photon blast", type: "uint" },
      { key: "activation",   label: "activation",   type: "uint" },
      { key: "class_flags",  label: "class flags",  type: "hex" },
    ],
    tools: [
      { key: "amount",     label: "amount",     type: "uint" },
      { key: "tech",       label: "tech",       type: "uint" },
      { key: "cost",       label: "cost (sale)", type: "int" },
      { key: "item_flags", label: "item flags", type: "hex" },
    ],
    specials: [
      { key: "type",   label: "type",   type: "uint", help: "0xFFFF = none" },
      { key: "amount", label: "amount", type: "uint" },
    ],
    stat_boosts: [
      { key: "stats",   label: "stat ids", type: "uint", arrayLen: 2 },
      { key: "amounts", label: "amounts",  type: "uint", arrayLen: 2 },
    ],
    mag_feeds: [
      { key: "def",     label: "DEF",     type: "int" },
      { key: "pow",     label: "POW",     type: "int" },
      { key: "dex",     label: "DEX",     type: "int" },
      { key: "mind",    label: "MIND",    type: "int" },
      { key: "iq",      label: "IQ",      type: "int" },
      { key: "synchro", label: "synchro", type: "int" },
    ],
    combinations: [
      { key: "used_item",     label: "used item (3 bytes)",     type: "uint", arrayLen: 3 },
      { key: "equipped_item", label: "equipped item (3 bytes)", type: "uint", arrayLen: 3 },
      { key: "result_item",   label: "result item (3 bytes)",   type: "uint", arrayLen: 3 },
      { key: "mag_level",     label: "mag level",     type: "uint" },
      { key: "grind",         label: "grind",         type: "uint" },
      { key: "level",         label: "level",         type: "uint" },
      { key: "char_class",    label: "char class",    type: "uint" },
    ],
  };
  FIELD_META.shields = FIELD_META.armors; // same struct

  // Sections that are 2D (have a class/sub-table dimension).
  const SECTIONS_2D = new Set(["weapons", "tools", "mag_feeds"]);

  // Sections to expose in the section dropdown (in display order).
  const SECTION_ORDER = [
    { key: "weapons",      label: "Weapons" },
    { key: "armors",       label: "Armors (Frames)" },
    { key: "shields",      label: "Shields (Barriers)" },
    { key: "units",        label: "Units" },
    { key: "mags",         label: "Mags" },
    { key: "tools",        label: "Tools" },
    { key: "specials",     label: "Specials" },
    { key: "stat_boosts",  label: "Stat Boosts" },
    { key: "mag_feeds",    label: "Mag Feed" },
    { key: "combinations", label: "Combinations" },
  ];

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;",
                '"': "&quot;", "'": "&#39;" })[c];
    });
  }

  function deepClone(o) { return JSON.parse(JSON.stringify(o)); }

  // ------------------------------------------------------------------
  // API helpers
  // ------------------------------------------------------------------
  async function getConfig() {
    const r = await fetch("/api/itempmt/config");
    if (!r.ok) throw new Error("config: " + r.status);
    return r.json();
  }
  async function getPmt(source) {
    const url = "/api/itempmt" +
                (source ? "?source=" + encodeURIComponent(source) : "");
    const r = await fetch(url);
    if (!r.ok) {
      const detail = await r.text().catch(() => "");
      throw new Error("get pmt (" + r.status + "): " + detail);
    }
    return r.json();
  }
  async function postPmt(data) {
    const r = await fetch("/api/itempmt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: data }),
    });
    if (!r.ok) {
      const detail = await r.text().catch(() => "");
      throw new Error("post pmt (" + r.status + "): " + detail);
    }
    return r.json();
  }
  async function deployPmt() {
    const r = await fetch("/api/itempmt/deploy", { method: "POST" });
    if (!r.ok) {
      const detail = await r.text().catch(() => "");
      throw new Error("deploy (" + r.status + "): " + detail);
    }
    return r.json();
  }

  // ------------------------------------------------------------------
  // Active list lookup
  // ------------------------------------------------------------------
  function getActiveList(d) {
    if (!d) return [];
    const sec = state.section;
    if (sec === "weapons") {
      const wc = d.weapons.find((w) => w.class_index === state.classIdx);
      return wc ? wc.items : [];
    }
    if (sec === "tools") {
      const tc = d.tools.find((t) => t.class_index === state.classIdx);
      return tc ? tc.items : [];
    }
    if (sec === "mag_feeds") {
      const mf = d.mag_feeds.find((m) => m.table_index === state.subTableIdx);
      return mf ? mf.results : [];
    }
    return d[sec] || [];
  }

  function getOriginalList(d) {
    if (!d) return [];
    const sec = state.section;
    if (sec === "weapons") {
      const wc = d.weapons.find((w) => w.class_index === state.classIdx);
      return wc ? wc.items : [];
    }
    if (sec === "tools") {
      const tc = d.tools.find((t) => t.class_index === state.classIdx);
      return tc ? tc.items : [];
    }
    if (sec === "mag_feeds") {
      const mf = d.mag_feeds.find((m) => m.table_index === state.subTableIdx);
      return mf ? mf.results : [];
    }
    return d[sec] || [];
  }

  // ------------------------------------------------------------------
  // Toolbar
  // ------------------------------------------------------------------
  function renderToolbar(hostEl) {
    const data = state.edited;
    const sectionOpts = SECTION_ORDER.map((s) =>
      '<option value="' + s.key + '"' +
      (state.section === s.key ? " selected" : "") +
      '>' + escapeHtml(s.label) + '</option>'
    ).join("");

    let classOpts = "";
    let classVisible = false;
    if (state.section === "weapons" && data) {
      classVisible = true;
      const classes = data.weapons.filter((w) => (w.items || []).length > 0);
      classOpts = classes.map((w) =>
        '<option value="' + w.class_index + '"' +
        (state.classIdx === w.class_index ? " selected" : "") +
        '>0x' + w.class_index.toString(16).padStart(2, "0").toUpperCase() +
        " " + escapeHtml(w.name) + " (" + w.items.length + ")</option>"
      ).join("");
    } else if (state.section === "tools" && data) {
      classVisible = true;
      const classes = data.tools.filter((t) => (t.items || []).length > 0);
      classOpts = classes.map((t) =>
        '<option value="' + t.class_index + '"' +
        (state.classIdx === t.class_index ? " selected" : "") +
        '>0x' + t.class_index.toString(16).padStart(2, "0").toUpperCase() +
        " " + escapeHtml(t.name) + " (" + t.items.length + ")</option>"
      ).join("");
    } else if (state.section === "mag_feeds" && data) {
      classVisible = true;
      classOpts = (data.mag_feeds || []).map((m) =>
        '<option value="' + m.table_index + '"' +
        (state.subTableIdx === m.table_index ? " selected" : "") +
        '>table ' + m.table_index + ' (' + m.results.length + ' results)</option>'
      ).join("");
    }

    const list = getActiveList(data);
    const entryOpts = list.map((rec, i) => {
      let label;
      if (state.section === "specials") {
        label = "[" + i + "] type=0x" + (rec.type || 0).toString(16) + " amt=" + rec.amount;
      } else if (state.section === "stat_boosts") {
        label = "[" + i + "] stats=[" + (rec.stats || []).join(",") + "] amts=[" + (rec.amounts || []).join(",") + "]";
      } else if (state.section === "mag_feeds") {
        label = "[" + i + "] " + ["FIRST","SECOND","THIRD","FOURTH","FIFTH","SIXTH","SEVENTH","EIGHTH","NINTH","TENTH","ELEVENTH"][i] || ("[" + i + "]");
      } else if (state.section === "combinations") {
        label = "[" + i + "] " + (rec.used_item || []).map((b) => "0x" + (b||0).toString(16).padStart(2,"0")).join(" ");
      } else {
        // item entry
        const id = (rec.id != null ? "0x" + (rec.id >>> 0).toString(16).padStart(8, "0") : "?");
        label = "[" + i + "] " + id;
        if (rec.type != null) label += "  type=" + rec.type;
        if (rec.skin != null) label += "  skin=" + rec.skin;
      }
      return '<option value="' + i + '"' +
             (state.entryIdx === i ? " selected" : "") + '>' +
             escapeHtml(label) + '</option>';
    }).join("");

    hostEl.innerHTML =
      '<div class="ipmt-toolbar">' +
      '<label>section <select id="ipmtSection">' + sectionOpts + '</select></label>' +
      (classVisible ? ('<label>' +
        (state.section === "mag_feeds" ? "table" : "class") +
        ' <select id="ipmtClass">' + classOpts + '</select></label>') : '') +
      '<label>entry <select id="ipmtEntry">' + entryOpts + '</select></label>' +
      '<button type="button" id="ipmtReload" class="ghost" title="reload from newserv (discards edits)">reload</button>' +
      '<button type="button" id="ipmtExport" title="serialize edits + PRS-compress; stage in cache/itempmt_export/">export to staging</button>' +
      '<button type="button" id="ipmtDeploy" class="warn" title="copy staged file to newserv (with backup)">deploy to newserv</button>' +
      '<span class="dim" id="ipmtStatus"></span>' +
      '</div>';

    hostEl.querySelector("#ipmtSection").addEventListener("change", function (e) {
      state.section = e.target.value;
      // Reset class/entry indices to a safe default
      if (state.section === "weapons" && state.edited) {
        const wc = state.edited.weapons.find((w) => (w.items || []).length > 0);
        state.classIdx = wc ? wc.class_index : 0;
      } else if (state.section === "tools" && state.edited) {
        const tc = state.edited.tools.find((t) => (t.items || []).length > 0);
        state.classIdx = tc ? tc.class_index : 0;
      } else if (state.section === "mag_feeds") {
        state.subTableIdx = 0;
      }
      state.entryIdx = 0;
      rerender();
    });
    const cls = hostEl.querySelector("#ipmtClass");
    if (cls) {
      cls.addEventListener("change", function (e) {
        const v = parseInt(e.target.value, 10);
        if (state.section === "mag_feeds") state.subTableIdx = v;
        else state.classIdx = v;
        state.entryIdx = 0;
        rerender();
      });
    }
    hostEl.querySelector("#ipmtEntry").addEventListener("change", function (e) {
      state.entryIdx = parseInt(e.target.value, 10) || 0;
      renderEditor();
    });
    hostEl.querySelector("#ipmtReload").addEventListener("click", reloadFromServer);
    hostEl.querySelector("#ipmtExport").addEventListener("click", onExport);
    hostEl.querySelector("#ipmtDeploy").addEventListener("click", onDeploy);

    // Live Test button (2026-04-25). Stages-then-deploys to newserv with
    // an attempt at `reload patch-indexes`; surfaces a status pip + the
    // last 3 actions. Mounted between ipmtDeploy and ipmtStatus.
    if (window.PSOLiveTest) {
      const toolbar = hostEl.querySelector(".ipmt-toolbar");
      const statusEl = hostEl.querySelector("#ipmtStatus");
      window.PSOLiveTest.ensureLiveButton({
        host: toolbar,
        beforeNode: statusEl,
        panelId: "itempmt",
        kind: "itempmt",
        title: "stage current edits → push to newserv (live)",
      });
      const ltBtn = toolbar.querySelector("#ltBtn_itempmt");
      if (ltBtn) {
        const newBtn = ltBtn.cloneNode(true);
        ltBtn.parentNode.replaceChild(newBtn, ltBtn);
        newBtn.addEventListener("click", onLiveTest);
      }
      window.PSOLiveTest.attachPip(toolbar, "itempmt");
    }
  }

  function setStatus(msg, isErr) {
    const el = document.getElementById("ipmtStatus");
    if (!el) return;
    el.textContent = msg || "";
    el.classList.toggle("err", !!isErr);
  }

  // ------------------------------------------------------------------
  // Editor
  // ------------------------------------------------------------------
  function renderEditor() {
    const host = document.getElementById("ipmtEditor");
    if (!host) return;
    if (!state.edited) {
      host.innerHTML = '<div class="dim">load to edit.</div>';
      return;
    }
    const list = getActiveList(state.edited);
    const origList = getOriginalList(state.original);
    if (!list.length) {
      host.innerHTML = '<div class="dim">section is empty.</div>';
      renderInspector();
      return;
    }
    if (state.entryIdx >= list.length) state.entryIdx = 0;
    const ent = list[state.entryIdx];
    const orig = origList[state.entryIdx] || {};

    // Render groups: ItemBase (if present in record) + section-specific.
    const hasItemBase = ent.id !== undefined && ent.team_points !== undefined;
    const groups = [];
    if (hasItemBase) {
      groups.push({ name: "Item Base", fields: FIELD_META.item_base });
    }
    groups.push({ name: state.section, fields: FIELD_META[state.section] || [] });

    let html = '<div class="ipmt-entry-header">';
    html += '<span class="ipmt-entry-slot">[' + state.entryIdx + ']</span>';
    if (hasItemBase) {
      html += '<span class="ipmt-entry-id">id=0x' +
              ((ent.id >>> 0) || 0).toString(16).padStart(8, "0") + '</span>';
    }
    html += '</div>';

    for (const g of groups) {
      html += '<div class="ipmt-group">';
      html += '<div class="ipmt-group-title">' + escapeHtml(g.name) + '</div>';
      html += '<div class="ipmt-group-fields">';
      for (const m of g.fields) {
        const v = ent[m.key];
        const ov = orig[m.key];
        const isArr = m.arrayLen && Array.isArray(v);
        if (isArr) {
          html += '<label class="ipmt-field"' +
                  (m.help ? ' title="' + escapeHtml(m.help) + '"' : '') + '>';
          html += '<span class="ipmt-field-label">' + escapeHtml(m.label) + '</span>';
          html += '<span class="ipmt-array">';
          for (let i = 0; i < m.arrayLen; i++) {
            const av = v[i];
            const aov = (Array.isArray(ov) ? ov[i] : null);
            const changed = av !== aov;
            html += '<input type="number" class="ipmt-arr-cell' +
                    (changed ? " changed" : "") +
                    '" data-key="' + m.key + '" data-arr-idx="' + i +
                    '" value="' + (av == null ? "" : av) +
                    '" step="' + (m.type === "float" ? "any" : "1") +
                    '" />';
          }
          html += '</span>';
          html += '</label>';
        } else {
          const changed = v !== ov;
          html += '<label class="ipmt-field' + (changed ? " changed" : "") + '"' +
                  (m.help ? ' title="' + escapeHtml(m.help) + '"' : '') + '>';
          html += '<span class="ipmt-field-label">' + escapeHtml(m.label) + '</span>';
          if (m.type === "hex") {
            html += '<input type="text" data-key="' + m.key +
                    '" value="0x' + (((v >>> 0) || 0).toString(16)) +
                    '" />';
          } else {
            html += '<input type="number" data-key="' + m.key +
                    '" value="' + (v == null ? "" : v) +
                    '" step="' + (m.type === "float" ? "any" : "1") + '" />';
          }
          if (changed) {
            html += '<span class="ipmt-orig dim" title="original">' +
                    (ov == null ? "(null)" : escapeHtml(String(ov))) + '</span>';
          }
          html += '</label>';
        }
      }
      html += '</div></div>';
    }
    host.innerHTML = html;

    // Wire inputs
    host.querySelectorAll('input[data-key]').forEach((inp) => {
      inp.addEventListener("input", () => {
        const key = inp.dataset.key;
        const ai = inp.dataset.arrIdx;
        const meta = ([].concat(FIELD_META.item_base, FIELD_META[state.section] || []))
                     .find((m) => m.key === key);
        let v;
        if (meta && meta.type === "hex") {
          // accept "0xff" / "ff" / decimal
          const t = inp.value.trim();
          v = t.toLowerCase().startsWith("0x")
              ? parseInt(t, 16)
              : parseInt(t, 10);
          if (!isFinite(v)) v = 0;
        } else if (meta && meta.type === "float") {
          v = parseFloat(inp.value);
          if (!isFinite(v)) v = 0;
        } else {
          v = parseInt(inp.value, 10);
          if (!isFinite(v)) v = 0;
        }
        if (ai !== undefined) {
          const arr = ent[key];
          if (Array.isArray(arr)) arr[parseInt(ai, 10)] = v;
        } else {
          ent[key] = v;
          // Drop bits sidecar if a known float field
          delete ent["_" + key + "_bits"];
        }
        // Visual diff toggle
        const labelEl = inp.closest(".ipmt-field");
        if (labelEl) {
          const ov = orig[key];
          const cur = ent[key];
          const changed = ai !== undefined
              ? (Array.isArray(ov) && Array.isArray(cur) && ov[parseInt(ai, 10)] !== cur[parseInt(ai, 10)])
              : (cur !== ov);
          labelEl.classList.toggle("changed", changed);
        }
        renderInspector();
      });

      // Cross-tool undo bus integration (2026-04-25, C7 follow-up).
      // Same focus-then-change pattern as mob_dsl_panel / battle_param_panel.
      // The closure captures the SECTION + class/sub-table + entry idx
      // so that a user navigating away (different weapon class, etc.)
      // still gets the right field reverted on Ctrl+Z.
      inp.addEventListener("focus", () => {
        const key = inp.dataset.key;
        const ai = inp.dataset.arrIdx;
        if (ai !== undefined) {
          const arr = ent[key];
          inp._psoUbBefore = (Array.isArray(arr) ? arr[parseInt(ai, 10)] : undefined);
        } else {
          inp._psoUbBefore = ent[key];
        }
        inp._psoUbCtx = {
          section: state.section,
          classIdx: state.classIdx,
          subTableIdx: state.subTableIdx,
          entryIdx: state.entryIdx,
        };
      });
      inp.addEventListener("change", () => {
        if (!window.psoUndoBus) return;
        if (inp._psoUbBefore === undefined) return;
        const key = inp.dataset.key;
        const ai = inp.dataset.arrIdx;
        const before = inp._psoUbBefore;
        const after = (ai !== undefined)
            ? (Array.isArray(ent[key]) ? ent[key][parseInt(ai, 10)] : undefined)
            : ent[key];
        if (before === after) return;
        const ctx = inp._psoUbCtx || {
          section: state.section,
          classIdx: state.classIdx,
          subTableIdx: state.subTableIdx,
          entryIdx: state.entryIdx,
        };
        const meta = ([].concat(FIELD_META.item_base, FIELD_META[ctx.section] || []))
                     .find((m) => m.key === key);
        const fieldLabel = (meta && meta.label) || key
                          + (ai !== undefined ? "[" + ai + "]" : "");
        const apply = (val) => {
          // Restore the underlying record value on the SAME entry the
          // user committed against — even if they've navigated away.
          // Snap state back so the editor reflects the change.
          let needsRender = false;
          if (state.section !== ctx.section) {
            state.section = ctx.section;
            needsRender = true;
            const sel = document.getElementById("ipmtSection");
            if (sel) sel.value = ctx.section;
          }
          if (state.classIdx !== ctx.classIdx) {
            state.classIdx = ctx.classIdx;
            needsRender = true;
          }
          if (state.subTableIdx !== ctx.subTableIdx) {
            state.subTableIdx = ctx.subTableIdx;
            needsRender = true;
          }
          if (state.entryIdx !== ctx.entryIdx) {
            state.entryIdx = ctx.entryIdx;
            needsRender = true;
            const sel = document.getElementById("ipmtEntry");
            if (sel) sel.value = String(ctx.entryIdx);
          }
          // Mutate the canonical record on `state.edited` (not on the
          // cached `ent` from the original render scope, which may now
          // refer to the wrong entry after a section switch).
          const editedList = getActiveList(state.edited);
          const tgt = editedList[ctx.entryIdx];
          if (!tgt) return;
          if (ai !== undefined) {
            const arr = tgt[key];
            if (Array.isArray(arr)) arr[parseInt(ai, 10)] = val;
          } else {
            tgt[key] = val;
            delete tgt["_" + key + "_bits"];
          }
          // Always re-render the editor (cheaper than scoped DOM updates
          // here since the field tree is small per entry). When the user
          // had navigated away (different section/class/entry) we must
          // also rebuild the toolbar so its dropdowns reflect the
          // restored selection.
          if (needsRender) rerender();
          else renderEditor();
        };
        window.psoUndoBus.push({
          label: "edit " + ctx.section + "[" + ctx.entryIdx + "]/" + fieldLabel,
          panelId: "itempmt",
          undo: () => apply(before),
          redo: () => apply(after),
        });
        // Re-prime so subsequent commits in the same focus session work.
        inp._psoUbBefore = after;
      });
    });
    renderInspector();
  }

  function renderInspector() {
    const host = state._inspectorHost;
    if (!host) return;
    if (!state.edited || !state.original) {
      host.innerHTML = '<div class="vp-insp-help dim">no PMT loaded.</div>';
      return;
    }
    const changes = computeChanges();
    let html = '<div class="vp-insp-title">Item PMT</div>';
    html += '<div class="vp-insp-help dim">' +
            'Edit weapon / armor / mag / tool stats. "export to staging" ' +
            'serializes + PRS-compresses to cache/itempmt_export/. ' +
            '"deploy to newserv" copies the staged file with a timestamped backup.' +
            '</div>';
    html += '<div class="vp-insp-section ipmt-source">';
    html += '<dl class="ipmt-meta">';
    if (state.config && state.config.configured_path) {
      html += '<dt>source</dt><dd class="dim wrap">' +
              escapeHtml(state.config.configured_path) + '</dd>';
    }
    html += '<dt>changes</dt><dd>' + changes.length + '</dd>';
    html += '</dl></div>';
    if (changes.length) {
      html += '<div class="vp-insp-section">';
      html += '<div class="ipmt-changes-title">changed fields</div>';
      html += '<ul class="ipmt-changes">';
      for (const c of changes.slice(0, 80)) {
        html += '<li>' + escapeHtml(c.label) + ': ' +
                escapeHtml(String(c.from)) + ' → ' +
                escapeHtml(String(c.to)) + '</li>';
      }
      if (changes.length > 80) {
        html += '<li class="dim">... ' + (changes.length - 80) + ' more</li>';
      }
      html += '</ul></div>';
    }
    host.innerHTML = html;
  }

  // Recursively diff arrays of records / records / scalars between
  // original and edited. We only walk keys we know about — opaque
  // sections are not surfaced.
  function computeChanges() {
    const o = state.original, e = state.edited;
    if (!o || !e) return [];
    const out = [];
    const SECTION_KEYS = ["specials", "stat_boosts", "combinations",
                          "armors", "shields", "units", "mags",
                          "v1_replacement", "weapon_sale_divisors",
                          "star_values"];
    // 1D sections
    for (const k of SECTION_KEYS) {
      const oa = o[k] || [], ea = e[k] || [];
      const n = Math.max(oa.length, ea.length);
      for (let i = 0; i < n; i++) {
        const ov = oa[i], ev = ea[i];
        if (ov && ev && typeof ov === "object" && typeof ev === "object") {
          for (const fk of Object.keys(ev)) {
            if (fk.startsWith("_") && fk.endsWith("_bits")) continue;
            const a = ov[fk], b = ev[fk];
            if (Array.isArray(a) && Array.isArray(b)) {
              if (a.length !== b.length || a.some((x, j) => x !== b[j])) {
                out.push({ label: k + "[" + i + "]." + fk, from: JSON.stringify(a), to: JSON.stringify(b) });
              }
            } else if (a !== b) {
              out.push({ label: k + "[" + i + "]." + fk, from: a, to: b });
            }
          }
        } else if (typeof ev === "number" || typeof ev === "string") {
          if (ov !== ev) out.push({ label: k + "[" + i + "]", from: ov, to: ev });
        }
      }
    }
    // weapons (2D)
    for (let ci = 0; ci < (e.weapons || []).length; ci++) {
      const wo = (o.weapons || [])[ci] || { items: [] };
      const we = e.weapons[ci];
      const ol = wo.items || [], el2 = we.items || [];
      const n = Math.max(ol.length, el2.length);
      for (let i = 0; i < n; i++) {
        const a = ol[i], b = el2[i];
        if (!a || !b) continue;
        for (const fk of Object.keys(b)) {
          if (fk.startsWith("_") && fk.endsWith("_bits")) continue;
          if (a[fk] !== b[fk]) {
            // arrays
            if (Array.isArray(a[fk]) && Array.isArray(b[fk])) {
              if (a[fk].length !== b[fk].length || a[fk].some((x, j) => x !== b[fk][j])) {
                out.push({
                  label: "weapons[" + we.name + "][" + i + "]." + fk,
                  from: JSON.stringify(a[fk]),
                  to: JSON.stringify(b[fk]),
                });
              }
            } else {
              out.push({
                label: "weapons[" + we.name + "][" + i + "]." + fk,
                from: a[fk], to: b[fk],
              });
            }
          }
        }
      }
    }
    // tools (2D)
    for (let ci = 0; ci < (e.tools || []).length; ci++) {
      const to_ = (o.tools || [])[ci] || { items: [] };
      const te = e.tools[ci];
      const ol = to_.items || [], el2 = te.items || [];
      const n = Math.max(ol.length, el2.length);
      for (let i = 0; i < n; i++) {
        const a = ol[i], b = el2[i];
        if (!a || !b) continue;
        for (const fk of Object.keys(b)) {
          if (fk.startsWith("_") && fk.endsWith("_bits")) continue;
          if (Array.isArray(a[fk]) && Array.isArray(b[fk])) {
            if (a[fk].length !== b[fk].length || a[fk].some((x, j) => x !== b[fk][j])) {
              out.push({
                label: "tools[" + te.name + "][" + i + "]." + fk,
                from: JSON.stringify(a[fk]),
                to: JSON.stringify(b[fk]),
              });
            }
          } else if (a[fk] !== b[fk]) {
            out.push({
              label: "tools[" + te.name + "][" + i + "]." + fk,
              from: a[fk], to: b[fk],
            });
          }
        }
      }
    }
    // mag_feeds
    for (let ti = 0; ti < (e.mag_feeds || []).length; ti++) {
      const oo = (o.mag_feeds || [])[ti] || { results: [] };
      const ee = e.mag_feeds[ti];
      const ol = oo.results || [], el2 = ee.results || [];
      const n = Math.max(ol.length, el2.length);
      for (let i = 0; i < n; i++) {
        const a = ol[i], b = el2[i];
        if (!a || !b) continue;
        for (const fk of Object.keys(b)) {
          if (fk.startsWith("_") && fk.endsWith("_bits")) continue;
          if (Array.isArray(a[fk]) && Array.isArray(b[fk])) {
            if (a[fk].length !== b[fk].length || a[fk].some((x, j) => x !== b[fk][j])) {
              out.push({
                label: "mag_feeds[" + ti + "][" + i + "]." + fk,
                from: JSON.stringify(a[fk]),
                to: JSON.stringify(b[fk]),
              });
            }
          } else if (a[fk] !== b[fk]) {
            out.push({
              label: "mag_feeds[" + ti + "][" + i + "]." + fk,
              from: a[fk], to: b[fk],
            });
          }
        }
      }
    }
    return out;
  }

  function rerender() {
    const tb = document.getElementById("ipmtToolbar");
    if (tb) renderToolbar(tb);
    renderEditor();
  }

  // ------------------------------------------------------------------
  // Actions
  // ------------------------------------------------------------------
  async function reloadFromServer() {
    setStatus("loading...");
    try {
      const resp = await getPmt("newserv");
      state.original = deepClone(resp.data);
      state.edited = deepClone(resp.data);
      setStatus("loaded " + (resp.source_path || "") +
                " (raw=" + resp.raw_size + ")");
      // Default to the first non-empty weapon class
      const wc = state.edited.weapons.find((w) => (w.items || []).length > 0);
      state.classIdx = wc ? wc.class_index : 0;
      state.entryIdx = 0;
      rerender();
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
      const r = await postPmt(state.edited);
      setStatus("staged: raw=" + r.raw_size + "B prs=" + r.prs_size +
                "B md5(prs)=" + (r.prs_md5 || "").slice(0, 12) + " ...");
    } catch (e) {
      setStatus("export failed: " + e.message, true);
    }
  }

  async function onDeploy() {
    if (!confirm("Deploy ItemPMT to newserv? A timestamped backup will be created.")) return;
    setStatus("deploying...");
    try {
      const r = await deployPmt();
      setStatus("deployed: " + r.deployed_to +
                (r.backup ? " (backup " + r.backup + ")" : ""));
    } catch (e) {
      setStatus("deploy failed: " + e.message, true);
    }
  }

  // Live Test: stage current edits + PRS-compress → POST /api/live_test →
  // newserv gets the new ItemPMT. Best-effort `reload patch-indexes` runs
  // automatically if NEWSERV_RELOAD_URL is configured.
  async function onLiveTest() {
    if (!state.edited) {
      setStatus("nothing to live-test", true);
      window.PSOLiveTest && window.PSOLiveTest.setPipState(
        "itempmt", "failed", "no data");
      return;
    }
    setStatus("staging for live test...");
    try {
      const r = await postPmt(state.edited);
      setStatus("staged prs=" + r.prs_size + "B; pushing to newserv...");
      const result = await window.PSOLiveTest.triggerLiveTest("itempmt", {
        panelId: "itempmt",
        body: { attempt_reload: true },
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
        "itempmt", "failed", "stage failed: " + e.message);
    }
  }

  // ------------------------------------------------------------------
  // Perspective registration
  // ------------------------------------------------------------------
  window.PSOPerspectives.register("item-pmt", {
    label: "Item PMT",
    match: function (entry, file) {
      if (entry && entry.category === "itempmt") return 100;
      return 0;
    },
    mount: async function (stage, insp, ctx) {
      stage.innerHTML =
        '<div class="ipmt-perspective">' +
        '<div id="ipmtToolbar"></div>' +
        '<div id="ipmtEditor" class="ipmt-editor"></div>' +
        '</div>';
      state._inspectorHost = insp;
      if (!state.config) {
        try {
          state.config = await getConfig();
        } catch (e) {
          stage.innerHTML = '<div class="err">' + escapeHtml("config error: " + e.message) + '</div>';
          return;
        }
      }
      const tb = document.getElementById("ipmtToolbar");
      if (tb) renderToolbar(tb);
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
  // Header button
  // ------------------------------------------------------------------
  function openPerspective() {
    const ctx = {
      path: "__itempmt__",
      entry: { category: "itempmt", format: "ItemPMT" },
      fileName: "ItemPMT-bb-v4.prs",
    };
    if (window.PSOPerspectives && window.PSOPerspectives.switchTo) {
      window.PSOPerspectives.switchTo("item-pmt", ctx);
    }
  }

  function ensureHeaderButton() {
    if (document.getElementById("btnItemPmt")) return;
    const status = document.getElementById("status");
    const header = status ? status.parentNode : null;
    if (!header) return;
    const btn = document.createElement("button");
    btn.id = "btnItemPmt";
    btn.type = "button";
    btn.className = "ghost";
    btn.title = "edit ItemPMT-bb-v4.prs (newserv item parameter table)";
    btn.textContent = "Item PMT";
    header.insertBefore(btn, status);
    btn.addEventListener("click", openPerspective);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", ensureHeaderButton);
  } else {
    ensureHeaderButton();
  }
})();
