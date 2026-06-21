// PSOBB Modding Suite — <pso-asset-tree> Web Component.
// =====================================================================
// Sidebar that renders the asset manifest grouped by category. Click
// a leaf to emit `asset.opened` on the bus; app.js picks that up and
// (if the file is a known texture container) routes through openFile.
//
// Layout:
//   <pso-asset-tree>
//     ┌──────────────────────────────────┐
//     │ [search box] [refresh ⟳]         │  search filters paths
//     ├──────────────────────────────────┤
//     │ ▶ Textures (124)                 │  each category collapses
//     │   ▼ Models (87)                  │
//     │     bm_boss2_de_rol_le.bml       │  click → bus.emit
//     │     bm_ene_lappy.bml             │
//     │   ▶ Quests (33)                  │
//     │   ...                            │
//     └──────────────────────────────────┘
//
// Persists per-category expanded state in localStorage under
// `pso_tree_expanded`. Polls /api/manifest every 30 s while the
// endpoint 404s (i.e. while Agent 1 is shipping); stops polling once
// it lands.
//
// All colours / spacing come from --tk-* tokens. The cosmic backdrop
// is enabled only when the host page sets body.theme-psobb (Agent 5
// keeps the default theme intact).
// =====================================================================

(function () {
  "use strict";

  if (customElements.get("pso-asset-tree")) return; // idempotent

  const LS_EXPANDED_KEY = "pso_tree_expanded";
  const LS_FILTER_KEY = "pso_tree_filter";    // active category-tab filter
  const LS_SCROLL_KEY = "pso_tree_scroll";    // scroll position of body
  const POLL_MS = 30_000;

  // Tab-strip filter constants. "all" is the default. Other values map 1:1
  // to manifest category enum names. We keep this small (5 tabs) so the
  // strip doesn't wrap; the full category breakdown is still visible inside
  // the tree as collapsible groups.
  const TAB_FILTERS = [
    { key: "all",     label: "All",     match: null },
    { key: "model",   label: "Models",  match: ["model"] },
    { key: "texture", label: "Textures", match: ["texture", "container"] },
    { key: "ui",      label: "UI",      match: ["ui"] },
    { key: "audio",   label: "Audio",   match: ["audio"] },
    { key: "quest",   label: "Quests",  match: ["quest"] },
    // Floors live under the manifest 'map' category, so match:["map"]
    // reuses the existing canonical category (no manifest/enum change).
    // Caveat: until a dedicated 'floor' category exists, this pill aliases
    // the map category (count == maps count). Clicking a floor/map leaf
    // routes to the floor perspective via asset_router.js.
    { key: "floor",   label: "Floors",  match: ["map"] },
  ];

  // Human-readable labels for the manifest category enum.
  const CATEGORY_LABELS = {
    texture:   "Textures",
    model:     "Models",
    container: "Containers",
    quest:     "Quests",
    map:       "Maps",
    audio:     "Audio",
    ui:        "UI",
    script:    "Scripts",
    cinematic: "Cinematics",
    metadata:  "Metadata",
    animation: "Animations",
    unknown:   "Unknown",
  };

  // Inferred-category label list. Display order; categories not in this
  // list still render but at the bottom in alpha order. The asset tree
  // uses ``entry.inferred_category`` first (set by manifest.infer_category);
  // entries with no inferred_category fall back to the canonical
  // ``category`` enum so nothing is lost.
  const INFERRED_CATEGORY_ORDER = [
    "Bosses",
    "Boss Attack Effects",
    "Enemies",
    "NPCs",
    "Mags",
    "Photon Blasts",
    "Player Bodies",
    "Player Headgear",
    "Player Textures",
    "Player Rigs",
    "Player Misc",
    "Items",
    "Weapons / Items",
    "Weapon Textures",
    "Effects",
    "Objects",
    "Set Pieces",
    "Maps / Terrain",
  ];

  // ── persistence helpers ──────────────────────────────────────────

  function loadExpandedState() {
    try {
      const raw = localStorage.getItem(LS_EXPANDED_KEY);
      if (!raw) return {};
      const parsed = JSON.parse(raw);
      return (parsed && typeof parsed === "object") ? parsed : {};
    } catch (_e) {
      return {};
    }
  }

  function saveExpandedState(state) {
    try {
      localStorage.setItem(LS_EXPANDED_KEY, JSON.stringify(state));
    } catch (_e) {
      // localStorage may be disabled in some embeds — silently degrade.
    }
  }

  function loadFilterState() {
    try {
      const raw = localStorage.getItem(LS_FILTER_KEY);
      // Validate against TAB_FILTERS so a stale value can't put the UI
      // into a state that has no matching tab.
      if (raw && TAB_FILTERS.some((f) => f.key === raw)) return raw;
    } catch (_e) {}
    return "all";
  }

  function saveFilterState(key) {
    try {
      localStorage.setItem(LS_FILTER_KEY, key);
    } catch (_e) {}
  }

  function loadScrollPos() {
    try {
      const raw = localStorage.getItem(LS_SCROLL_KEY);
      if (raw == null) return 0;
      const n = parseInt(raw, 10);
      return isNaN(n) ? 0 : Math.max(0, n);
    } catch (_e) { return 0; }
  }

  function saveScrollPos(n) {
    try {
      localStorage.setItem(LS_SCROLL_KEY, String(Math.max(0, n | 0)));
    } catch (_e) {}
  }

  // Bucket an entry into the user-facing display category. Inferred
  // category (from manifest.infer_category) wins; otherwise fall back
  // to the canonical category enum's human-readable label.
  function bucketLabel(entry) {
    if (!entry) return "Unknown";
    if (typeof entry.inferred_category === "string" && entry.inferred_category) {
      return entry.inferred_category;
    }
    const cat = entry.category || "unknown";
    return CATEGORY_LABELS[cat] || cat;
  }

  // Group entries by display label. Returns
  //   { labels: ["Bosses", "Enemies", ...], byLabel: { "Bosses": [entry...] } }
  // with `labels` ordered: INFERRED_CATEGORY_ORDER first, then any
  // canonical-category labels in CATEGORY_ORDER, then alpha for
  // anything left.
  // Display label for the area-data bucket. Entries whose inferred bucket
  // is this (or whose canonical category is "map") are consumed by the
  // dedicated Map Editor / Floor Editor tabs, NOT the asset tree — so the
  // tree hides them by default (opts.excludeMap) to declutter. The Floors
  // tab re-includes them because it scopes inScope to category==="map"
  // BEFORE calling bucketEntries with excludeMap=false.
  const MAP_BUCKET_LABEL = "Maps / Terrain";

  function bucketEntries(entries, opts) {
    const excludeMap = !!(opts && opts.excludeMap);
    const byLabel = Object.create(null);
    for (const e of entries) {
      if (!e || e.deprecated) continue;
      // Owner request: keep area/terrain data out of the asset tree (it
      // lives in the Map + Floor editor tabs). Drop entries whose DISPLAY
      // bucket is "Maps / Terrain". We key on the inferred bucket label, NOT
      // the canonical "map" category, on purpose: a handful of assets
      // (pl?smp.rel rigs, cam_*/particleentry effect tables, SetDataTable
      // set-pieces) have a .rel/.gsl extension that maps to canonical
      // category=="map" yet were re-bucketed to Player Rigs / Effects /
      // Set Pieces by the DB — those are real browsable assets and must
      // stay in the tree. Guarded by excludeMap so the Floors tab (which
      // scopes to category=="map" before calling this) still shows terrain.
      if (excludeMap && bucketLabel(e) === MAP_BUCKET_LABEL) {
        continue;
      }
      const label = bucketLabel(e);
      if (!byLabel[label]) byLabel[label] = [];
      byLabel[label].push(e);
    }
    // Sort each bucket by path for stable display.
    for (const list of Object.values(byLabel)) {
      list.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
    }
    // Order labels: inferred first, then canonical, then alpha leftovers.
    const seen = new Set();
    const labels = [];
    for (const lbl of INFERRED_CATEGORY_ORDER) {
      if (byLabel[lbl] && !seen.has(lbl)) { labels.push(lbl); seen.add(lbl); }
    }
    const canonOrder = ["Models", "Textures", "Containers", "UI", "Maps",
                        "Quests", "Animations", "Audio", "Scripts",
                        "Cinematics", "Metadata", "Unknown"];
    for (const lbl of canonOrder) {
      if (byLabel[lbl] && !seen.has(lbl)) { labels.push(lbl); seen.add(lbl); }
    }
    const remaining = Object.keys(byLabel).filter((l) => !seen.has(l)).sort();
    for (const lbl of remaining) labels.push(lbl);
    return { labels, byLabel };
  }

  // ── escaping ─────────────────────────────────────────────────────
  // The manifest comes from the local FS so the threat model is low,
  // but path strings can include unusual characters; escape on render.
  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[c]));
  }

  // Pretty size like "2.4 MB" / "812 B".
  function fmtSize(n) {
    if (typeof n !== "number" || !isFinite(n) || n < 0) return "";
    if (n < 1024) return n + " B";
    const u = ["KB", "MB", "GB"];
    let v = n / 1024, i = 0;
    while (v >= 1024 && i < u.length - 1) { v /= 1024; i += 1; }
    return v.toFixed(v < 10 ? 1 : 0) + " " + u[i];
  }

  // ── friendly-name + nesting helpers ──────────────────────────────
  // Human-readable label for an entry. Prefers the global resolver
  // (asset_names.js); degrades to the raw basename if it isn't loaded.
  function friendlyName(entry) {
    if (window.PSOAssetNames && typeof window.PSOAssetNames.friendly === "function") {
      try { return window.PSOAssetNames.friendly(entry) || ""; }
      catch (_e) { /* fall through */ }
    }
    // Minimal inline fallback: server display_name, else basename.
    if (entry && entry.display_name) return entry.display_name;
    const p = (entry && entry.path) || "";
    const base = String(p).replace(/\\/g, "/").split("/").pop();
    return base.indexOf("#") >= 0 ? base.split("#").pop() : base;
  }

  // The basename a row should also show as a dim subtitle/tooltip. For
  // an archive inner ("<archive>#<inner>") this is the INNER filename;
  // for a plain file it's the bare basename.
  function rawLabel(path) {
    let p = String(path || "").replace(/\\/g, "/");
    p = p.split("/").pop();
    if (p.indexOf("#") >= 0) {
      let inner = p.split("#").pop();
      // Drop the "NNNN_" synth index prefix that AFS inners carry so the
      // subtitle reads like a real filename ("inner.nj" not "0042_inner.nj").
      inner = inner.replace(/^\d{1,5}_/, "");
      return inner;
    }
    return p;
  }

  // Short basename of an archive path (parent_archive is a relpath).
  function archiveBasename(archivePath) {
    return String(archivePath || "").replace(/\\/g, "/").split("/").pop();
  }

  // Split a bucket's flat entry list into:
  //   archives : ordered list of { key, archivePath, archiveEntry|null,
  //                                children:[entry...] }
  //   loose    : entries that are NOT archive inners (rendered as before)
  // An entry is an inner when it has a `parent_archive` AND its path
  // contains '#'. The bare archive entry (path === parent_archive, no '#')
  // becomes the parent row when present in this same bucket; otherwise we
  // synthesise a parent header from the archive path so inners always roll
  // up under an obvious parent instead of appearing as flat #-siblings.
  function splitArchives(list) {
    const byArchive = new Map();   // archivePath -> { children:[], archiveEntry }
    const looseCandidates = [];    // possibly-bare-archive entries

    for (const e of list) {
      if (!e) continue;
      const pa = e.parent_archive;
      const isInner = pa && typeof e.path === "string" && e.path.indexOf("#") >= 0;
      if (isInner) {
        let rec = byArchive.get(pa);
        if (!rec) { rec = { children: [], archiveEntry: null }; byArchive.set(pa, rec); }
        rec.children.push(e);
      } else {
        looseCandidates.push(e);
      }
    }

    // Attach any bare archive entry (its path is a key in byArchive) to
    // the matching group as the parent; everything else stays loose.
    const loose = [];
    for (const e of looseCandidates) {
      const rec = byArchive.get(e.path);
      if (rec) rec.archiveEntry = e;
      else loose.push(e);
    }

    // Stable order: archives by archive path, children by their own path.
    const archives = [];
    for (const [archivePath, rec] of byArchive) {
      rec.children.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
      archives.push({ key: archivePath, archivePath, archiveEntry: rec.archiveEntry, children: rec.children });
    }
    archives.sort((a, b) => (a.archivePath < b.archivePath ? -1 : a.archivePath > b.archivePath ? 1 : 0));
    return { archives, loose };
  }

  // ── component CSS (shadow DOM scope) ─────────────────────────────
  // All values come from --tk-* tokens. Tokens cross the shadow
  // boundary because they're inherited via custom-property cascade
  // from :root.
  const STYLE = `
    :host {
      display: flex;
      flex-direction: column;
      width: 100%;
      height: 100%;
      font-family: var(--tk-font-body, sans-serif);
      color: var(--tk-text, #e0f0ff);
      background: transparent;
      font-size: var(--tk-fs-sm, 0.9rem);
    }

    .toolbar {
      display: flex;
      align-items: center;
      gap: var(--tk-sp-2, 0.5rem);
      padding: var(--tk-sp-2, 0.5rem) var(--tk-sp-3, 0.75rem);
      border-bottom: var(--tk-border-mute, 1px solid rgba(0,255,255,0.3));
      background: var(--tk-overlay, rgba(0,0,0,0.4));
    }

    .toolbar input.search {
      flex: 1;
      min-width: 0;
      background: rgba(0, 0, 0, 0.3);
      border: var(--tk-border-mute, 1px solid rgba(0,255,255,0.3));
      border-radius: var(--tk-rad-1, 4px);
      color: var(--tk-text, #e0f0ff);
      padding: 4px 8px;
      font-family: inherit;
      font-size: var(--tk-fs-sm, 0.9rem);
      transition: border-color var(--tk-d-2, 0.3s),
                  box-shadow   var(--tk-d-2, 0.3s);
    }
    .toolbar input.search:focus {
      outline: none;
      border-color: var(--tk-blue, #00ffff);
      box-shadow: var(--tk-glow-blue, 0 0 10px #00ffff);
    }
    .toolbar button.refresh {
      background: transparent;
      border: var(--tk-border-mute, 1px solid rgba(0,255,255,0.3));
      border-radius: var(--tk-rad-1, 4px);
      color: var(--tk-blue, #00ffff);
      padding: 2px 8px;
      font: inherit;
      cursor: pointer;
      transition: border-color var(--tk-d-2, 0.3s),
                  box-shadow   var(--tk-d-2, 0.3s);
    }
    .toolbar button.refresh:hover {
      border-color: var(--tk-blue, #00ffff);
      box-shadow: var(--tk-glow-blue, 0 0 10px #00ffff);
    }

    .tab-strip {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
      padding: 4px var(--tk-sp-3, 0.75rem);
      border-bottom: var(--tk-border-mute, 1px solid rgba(0,255,255,0.3));
      background: var(--tk-overlay, rgba(0,0,0,0.4));
    }
    .tab-strip button.tab {
      background: transparent;
      border: 1px solid var(--tk-line, rgba(0,255,255,0.3));
      border-radius: var(--tk-rad-1, 4px);
      color: var(--tk-text-mute, rgba(224,240,255,0.7));
      padding: 2px 8px;
      font-family: inherit;
      font-size: var(--tk-fs-xs, 0.8rem);
      cursor: pointer;
      transition: background-color var(--tk-d-1, 0.2s),
                  border-color var(--tk-d-1, 0.2s),
                  color var(--tk-d-1, 0.2s),
                  box-shadow var(--tk-d-2, 0.3s);
    }
    .tab-strip button.tab:hover {
      border-color: var(--tk-blue, #00ffff);
      color: var(--tk-blue, #00ffff);
    }
    .tab-strip button.tab.active {
      background: var(--tk-blue, #00ffff);
      color: #000;
      border-color: var(--tk-blue, #00ffff);
      box-shadow: var(--tk-glow-blue, 0 0 10px #00ffff);
    }
    .tab-strip button.tab .tab-count {
      margin-left: 4px;
      opacity: 0.65;
      font-size: 9px;
      vertical-align: 1px;
    }

    .body {
      flex: 1;
      overflow-y: auto;
      padding: var(--tk-sp-1, 0.25rem) 0;
    }

    .placeholder {
      padding: var(--tk-sp-5, 1.5rem) var(--tk-sp-4, 1rem);
      color: var(--tk-text-mute, rgba(224,240,255,0.7));
      text-align: center;
      font-size: var(--tk-fs-sm, 0.9rem);
      line-height: var(--tk-lh-body, 1.6);
    }
    .placeholder .hint {
      color: var(--tk-text-dim, rgba(224,240,255,0.5));
      font-size: var(--tk-fs-xs, 0.8rem);
      margin-top: var(--tk-sp-2, 0.5rem);
    }
    .placeholder button {
      margin-top: var(--tk-sp-3, 0.75rem);
      background: transparent;
      border: 1px solid var(--tk-blue, #00ffff);
      border-radius: var(--tk-rad-1, 4px);
      color: var(--tk-blue, #00ffff);
      padding: 4px 12px;
      font-family: inherit;
      cursor: pointer;
      transition: all var(--tk-d-2, 0.3s) var(--tk-ease, ease);
    }
    .placeholder button:hover {
      background: var(--tk-blue, #00ffff);
      color: #000;
      box-shadow: var(--tk-glow-blue, 0 0 10px #00ffff);
    }

    .group { margin: 0; }
    .group > .header {
      display: flex;
      align-items: center;
      gap: var(--tk-sp-2, 0.5rem);
      padding: 4px var(--tk-sp-3, 0.75rem);
      cursor: pointer;
      user-select: none;
      color: var(--tk-blue, #00ffff);
      font-weight: var(--tk-fw-medium, 500);
      border-bottom: 1px solid transparent;
      transition: background-color var(--tk-d-1, 0.2s);
    }
    .group > .header:hover {
      background: rgba(0, 255, 255, 0.06);
    }
    .group > .header .twist {
      width: 10px;
      display: inline-block;
      transition: transform var(--tk-d-1, 0.2s);
      color: var(--tk-text-dim, rgba(224,240,255,0.5));
    }
    .group.expanded > .header .twist {
      transform: rotate(90deg);
    }
    .group > .header .label { flex: 1; min-width: 0; }
    .group > .header .count {
      color: var(--tk-text-mute, rgba(224,240,255,0.7));
      font-size: var(--tk-fs-xs, 0.8rem);
      font-weight: var(--tk-fw-normal, 400);
    }

    .group > .items {
      display: none;
      list-style: none;
      margin: 0;
      padding: 0;
    }
    .group.expanded > .items { display: block; }

    .item {
      padding: 3px var(--tk-sp-3, 0.75rem) 3px var(--tk-sp-6, 2rem);
      cursor: pointer;
      color: var(--tk-text, #e0f0ff);
      font-size: var(--tk-fs-xs, 0.8rem);
      word-break: break-all;
      transition: background-color var(--tk-d-1, 0.2s),
                  color            var(--tk-d-1, 0.2s);
      border-left: 2px solid transparent;
    }
    .item:hover {
      background: rgba(0, 255, 255, 0.08);
      color: var(--tk-text-strong, #fff);
      border-left-color: var(--tk-blue, #00ffff);
    }
    .item .meta {
      display: block;
      color: var(--tk-text-dim, rgba(224,240,255,0.5));
      font-size: 10px;
      margin-top: 1px;
      font-family: var(--tk-font-mono, monospace);
    }

    /* Friendly (human-readable) primary label + dim raw-filename subtitle.
       The friendly name is the prominent line; the raw filename stays
       visible underneath (never hidden) so the user can always see what
       file backs the row. */
    .item .friendly {
      display: block;
      color: var(--tk-text, #e0f0ff);
      font-size: var(--tk-fs-xs, 0.8rem);
      line-height: 1.25;
    }
    .item .raw {
      display: block;
      color: var(--tk-text-dim, rgba(224,240,255,0.5));
      font-size: 10px;
      margin-top: 1px;
      font-family: var(--tk-font-mono, monospace);
      word-break: break-all;
    }

    /* Archive parent row: a collapsible node that rolls its inner files
       up underneath it (no more flat "<archive>#<inner>" siblings). */
    .archive { list-style: none; margin: 0; padding: 0; }
    .archive > .arc-header {
      display: flex;
      align-items: center;
      gap: var(--tk-sp-2, 0.5rem);
      padding: 3px var(--tk-sp-3, 0.75rem) 3px var(--tk-sp-5, 1.5rem);
      cursor: pointer;
      user-select: none;
      color: var(--tk-text, #e0f0ff);
      font-size: var(--tk-fs-xs, 0.8rem);
      border-left: 2px solid transparent;
      transition: background-color var(--tk-d-1, 0.2s);
    }
    .archive > .arc-header:hover {
      background: rgba(0, 255, 255, 0.06);
      border-left-color: var(--tk-blue, #00ffff);
    }
    .archive > .arc-header .arc-twist {
      width: 10px;
      display: inline-block;
      flex: none;
      transition: transform var(--tk-d-1, 0.2s);
      color: var(--tk-text-dim, rgba(224,240,255,0.5));
    }
    .archive.open > .arc-header .arc-twist { transform: rotate(90deg); }
    .archive > .arc-header .arc-name { flex: 1; min-width: 0; }
    .archive > .arc-header .arc-name .friendly { color: var(--tk-text, #e0f0ff); }
    .archive > .arc-header .arc-name .raw { color: var(--tk-text-dim, rgba(224,240,255,0.5)); }
    .archive > .arc-header .arc-count {
      flex: none;
      color: var(--tk-text-mute, rgba(224,240,255,0.7));
      font-size: var(--tk-fs-xs, 0.8rem);
    }
    /* The inner-children list is hidden until the archive is open. */
    .archive > .arc-children { display: none; list-style: none; margin: 0; padding: 0; }
    .archive.open > .arc-children { display: block; }
    /* Inner rows indent one step deeper than a normal leaf. */
    .archive > .arc-children > .item {
      padding-left: var(--tk-sp-7, 2.75rem);
    }
    .item .matched {
      display: block;
      color: var(--tk-green, #00ff88);
      font-size: 9px;
      margin-top: 1px;
      font-family: var(--tk-font-mono, monospace);
      opacity: 0.8;
    }
    .item.parsable-no .meta { color: var(--tk-orange, #ffaa00); }
    .item .cat-pill {
      display: inline-block;
      margin-right: 4px;
      padding: 0 4px;
      border-radius: 2px;
      background: rgba(0,255,255,0.12);
      color: var(--tk-blue, #00ffff);
      font-size: 9px;
      font-family: var(--tk-font-mono, monospace);
      vertical-align: 1px;
    }
    .item.cat-model    .cat-pill { background: rgba(157,78,221,0.18); color: var(--tk-purple, #9d4edd); }
    .item.cat-texture  .cat-pill { background: rgba(0,255,255,0.12);  color: var(--tk-blue,   #00ffff); }
    .item.cat-audio    .cat-pill { background: rgba(0,255,136,0.15);  color: var(--tk-green,  #00ff88); }
    .item.cat-quest    .cat-pill { background: rgba(255,170,0,0.15);  color: var(--tk-orange, #ffaa00); }
    .item.cat-script   .cat-pill { background: rgba(255,170,0,0.10);  color: var(--tk-orange, #ffaa00); }
    .item.cat-cinematic .cat-pill { background: rgba(157,78,221,0.10); color: var(--tk-purple, #9d4edd); }
    .item.cat-metadata .cat-pill { background: rgba(120,120,120,0.18); color: var(--tk-text-mute, rgba(224,240,255,0.7)); }
    .item.cat-unknown  .cat-pill { background: rgba(120,120,120,0.18); color: var(--tk-text-mute, rgba(224,240,255,0.7)); }
    .item.cat-container .cat-pill { background: rgba(0,255,255,0.10); color: var(--tk-blue,   #00ffff); }

    /* Multi-select highlight (2026-04-25). When psoSelection.has(path)
       is true, the item gets ms-selected; we mark it with a left
       accent + slight bg so users can see the batch in a long tree
       without scanning checkboxes. Honors --pso-ms-bg if the host
       page wants to override the colour. */
    .item.ms-selected {
      background: var(--pso-ms-bg, rgba(217, 185, 110, 0.10));
      border-left-color: var(--pso-ms-accent, #d9b96e);
    }
    .item.ms-selected:hover {
      background: var(--pso-ms-bg-hover, rgba(217, 185, 110, 0.18));
    }

    /* Persistent active-leaf highlight (the asset currently open). */
    .item.active {
      background: rgba(37, 99, 235, 0.20);
      color: var(--tk-text-strong, #fff);
      border-left-color: #2563eb;
    }
    .item.active:hover { background: rgba(37, 99, 235, 0.28); }

    /* Keyboard focus ring — the global :focus-visible can't cross the
       shadow boundary, so define it here for the tree's roving focus. */
    .item:focus-visible,
    .group > .header:focus-visible {
      outline: 2px solid #4da3ff;
      outline-offset: -2px;
    }
    .item:focus, .group > .header:focus { outline: none; }

    .empty-cat {
      padding: 4px var(--tk-sp-6, 2rem);
      color: var(--tk-text-dim, rgba(224,240,255,0.5));
      font-style: italic;
      font-size: var(--tk-fs-xs, 0.8rem);
    }

    .stats {
      padding: var(--tk-sp-2, 0.5rem) var(--tk-sp-3, 0.75rem);
      border-top: var(--tk-border-mute, 1px solid rgba(0,255,255,0.3));
      color: var(--tk-text-mute, rgba(224,240,255,0.7));
      font-size: var(--tk-fs-xs, 0.8rem);
    }
  `;

  // ── component ─────────────────────────────────────────────────────

  class PsoAssetTree extends HTMLElement {
    constructor() {
      super();
      this.attachShadow({ mode: "open" });
      this._search = "";
      this._expanded = loadExpandedState();
      this._filter = loadFilterState();      // active tab key from TAB_FILTERS
      this._pollTimer = null;
      this._searchDebounce = null;   // setTimeout handle for search input
      this._mounted = false;
      // Multi-select anchor (2026-04-25): the path most-recently
      // single-clicked WITHOUT a modifier. Shift+click extends the
      // selection from the anchor through the clicked item using the
      // current rendered order.
      this._selAnchor = null;
      this._lastRenderedPaths = [];
      // Path of the currently-open leaf (persistent .active highlight).
      this._activePath = null;
      this._onSearchInput = this._onSearchInput.bind(this);
      this._onRefreshClick = this._onRefreshClick.bind(this);
      this._onBodyClick = this._onBodyClick.bind(this);
      this._onBodyKeydown = this._onBodyKeydown.bind(this);
      this._onTabClick = this._onTabClick.bind(this);
      this._onSelectionChanged = this._onSelectionChanged.bind(this);
    }

    connectedCallback() {
      this._mounted = true;
      this._renderShell();
      // Best-effort initial load. Failures fall through to placeholder
      // + polling; we never block the UI.
      this._tryLoad();
      // Subscribe to multi-select store changes so highlights stay in
      // sync when other panels mutate the selection (e.g. clear-all).
      if (window.bus) {
        window.bus.on("selection.changed", this._onSelectionChanged);
      }
    }

    disconnectedCallback() {
      this._mounted = false;
      if (this._pollTimer) {
        clearInterval(this._pollTimer);
        this._pollTimer = null;
      }
      if (this._searchDebounce) {
        clearTimeout(this._searchDebounce);
        this._searchDebounce = null;
      }
      if (window.bus && typeof window.bus.off === "function") {
        window.bus.off("selection.changed", this._onSelectionChanged);
      }
    }

    _onSelectionChanged() {
      // Toggle the ms-selected class on existing rendered <li>s
      // without rebuilding the DOM (cheap incremental update).
      if (!this._bodyEl || !window.psoSelection) return;
      const items = this._bodyEl.querySelectorAll("li.item");
      for (const li of items) {
        const p = li.dataset.path;
        if (!p) continue;
        li.classList.toggle("ms-selected", window.psoSelection.has(p));
      }
    }

    // ── internals ─────────────────────────────────────────────────

    _renderShell() {
      const root = this.shadowRoot;
      // Tab strip: built from TAB_FILTERS so adding a tab is one-line.
      const tabBtns = TAB_FILTERS.map((t) =>
        `<button type="button" class="tab${this._filter === t.key ? " active" : ""}" ` +
        `data-tab="${esc(t.key)}" title="${esc(t.label)}">` +
        `${esc(t.label)}<span class="tab-count" data-tab-count="${esc(t.key)}"></span>` +
        `</button>`
      ).join("");
      root.innerHTML = `
        <style>${STYLE}</style>
        <div class="toolbar">
          <input class="search" type="search"
                 placeholder="filter by path..."
                 autocomplete="off" spellcheck="false" />
          <button class="refresh" title="re-fetch /api/manifest">⟳</button>
        </div>
        <div class="tab-strip" part="tab-strip">${tabBtns}</div>
        <div class="body" part="body"></div>
        <div class="stats" part="stats"></div>
      `;
      this._searchEl = root.querySelector("input.search");
      this._refreshEl = root.querySelector("button.refresh");
      this._tabStripEl = root.querySelector(".tab-strip");
      this._bodyEl = root.querySelector(".body");
      this._statsEl = root.querySelector(".stats");

      this._searchEl.addEventListener("input", this._onSearchInput);
      this._refreshEl.addEventListener("click", this._onRefreshClick);
      this._tabStripEl.addEventListener("click", this._onTabClick);
      this._bodyEl.addEventListener("click", this._onBodyClick);
      // Keyboard navigation (2026-06-19 a11y): the tree was mouse-only.
      this._bodyEl.addEventListener("keydown", this._onBodyKeydown);
      this._bodyEl.setAttribute("role", "tree");
      this._bodyEl.setAttribute("aria-label", "asset tree");
      // Persist the body scroll position so a manifest refresh / tree
      // refresh / page reload doesn't lose the user's place in a long
      // (~9k-entry) tree.
      this._onBodyScroll = () => {
        if (this._bodyEl) saveScrollPos(this._bodyEl.scrollTop);
      };
      this._bodyEl.addEventListener("scroll", this._onBodyScroll, { passive: true });

      this._renderPlaceholder("Loading manifest…");
    }

    _onTabClick(ev) {
      const btn = ev.target.closest("button.tab");
      if (!btn) return;
      const key = btn.dataset.tab;
      if (!key || key === this._filter) return;
      this._filter = key;
      saveFilterState(key);
      // Update tab UI
      const all = this._tabStripEl.querySelectorAll("button.tab");
      for (const b of all) {
        b.classList.toggle("active", b.dataset.tab === key);
      }
      // Re-render the body with the new filter applied
      this._renderTree();
    }

    // 2026-06-20: own the "All assets" pane-header coverage label so the
    // header count matches the tree's actual top-level categories (the
    // display buckets the user navigates), labelled "categories" to agree
    // with the footer. Replaces asset_router.js's old enum-category writer
    // that showed a different (manifest-enum) count under the same noun.
    _updateCoverageHeader(allEntries) {
      const el = document.getElementById("assetTreeCoverage");
      if (!el) return;
      let total = 0;
      for (const e of allEntries) { if (e && !e.deprecated) total += 1; }
      // Match the default ("All") tree view, which hides the Maps / Terrain
      // area-data bucket, so the header's category count agrees with what
      // the user actually sees in the tree.
      const { labels } = bucketEntries(
        allEntries.filter((e) => e && !e.deprecated),
        { excludeMap: true }
      );
      const n = labels.length;
      el.textContent = `${total.toLocaleString()} entries · ${n} categor${n === 1 ? "y" : "ies"}`;
    }

    _updateTabCounts(entries) {
      // Counts are computed against the canonical category enum so the
      // tab-strip filter (Models / Textures / Audio / Quests) lines up
      // with the manifest's hard-typed category field. The tab strip
      // is intentionally orthogonal to the inferred-category bucketing
      // shown inside the body.
      if (!this._tabStripEl) return;
      const total = entries.length;
      const counts = Object.create(null);
      for (const e of entries) {
        if (!e || e.deprecated) continue;
        const c = e.category || "unknown";
        counts[c] = (counts[c] || 0) + 1;
      }
      for (const t of TAB_FILTERS) {
        const span = this._tabStripEl.querySelector(`[data-tab-count="${t.key}"]`);
        if (!span) continue;
        let n;
        if (t.match === null) {
          n = total;
        } else {
          n = 0;
          for (const cat of t.match) n += (counts[cat] || 0);
        }
        span.textContent = n ? String(n) : "";
      }
    }

    async _tryLoad() {
      try {
        await window.PSOManifest.load();
        if (!this._mounted) return;
        this._stopPolling();
        this._renderTree();
      } catch (e) {
        if (!this._mounted) return;
        const code = e && e.code;
        const status = e && e.status;
        if (code === "ENDPOINT_MISSING" || status === 404) {
          this._renderPlaceholder(
            "Manifest not yet built.",
            "The /api/manifest endpoint is not available. " +
            "It will appear once Agent 1 (manifest backend) ships. " +
            "Polling every 30 s…",
            true,
          );
          this._startPolling();
        } else {
          this._renderPlaceholder(
            "Manifest failed to load.",
            (e && e.message) || String(e),
            true,
          );
        }
      }
    }

    _startPolling() {
      if (this._pollTimer) return;
      this._pollTimer = setInterval(async () => {
        try {
          await window.PSOManifest.refresh();
          if (!this._mounted) return;
          this._stopPolling();
          this._renderTree();
        } catch (_e) {
          // keep polling silently
        }
      }, POLL_MS);
    }

    _stopPolling() {
      if (this._pollTimer) {
        clearInterval(this._pollTimer);
        this._pollTimer = null;
      }
    }

    _onSearchInput(e) {
      // Debounce (2026-06-20 perf): _renderTree re-buckets + re-sorts the
      // full ~10.5k-entry manifest and rebuilds the DOM. Doing that on
      // every keystroke makes fast typing janky in a long tree. Coalesce
      // bursts into a single render ~120 ms after the user stops typing;
      // an empty query renders immediately so clearing the box feels snappy.
      this._search = (e.target.value || "").toLowerCase();
      if (this._searchDebounce) {
        clearTimeout(this._searchDebounce);
        this._searchDebounce = null;
      }
      if (!this._search) {
        this._renderTree();
        return;
      }
      this._searchDebounce = setTimeout(() => {
        this._searchDebounce = null;
        if (this._mounted) this._renderTree();
      }, 120);
    }

    async _onRefreshClick() {
      this._renderPlaceholder("Refreshing…");
      try {
        await window.PSOManifest.refresh();
        if (!this._mounted) return;
        this._renderTree();
      } catch (e) {
        if (!this._mounted) return;
        this._renderPlaceholder(
          "Refresh failed.",
          (e && e.message) || String(e),
          true,
        );
      }
    }

    _onBodyClick(ev) {
      // Event delegation: handle group toggles + item clicks here so
      // we don't have to wire per-element listeners on every render.
      const groupHeader = ev.target.closest(".group > .header");
      if (groupHeader) {
        const group = groupHeader.parentElement;
        const cat = group.dataset.cat;
        const open = group.classList.toggle("expanded");
        this._expanded[cat] = open;
        groupHeader.setAttribute("aria-expanded", open ? "true" : "false");
        saveExpandedState(this._expanded);
        // 2026-06-19 perf: leaves are built on demand. If this group's
        // <ul> is still empty (collapsed groups render no items), opening
        // it requires a re-render to build the leaves. Collapsing keeps
        // the items in the DOM until the next full render — fine.
        if (open) {
          const itemsEl = group.querySelector(".items");
          const built = itemsEl && itemsEl.querySelector("li.item, li.empty-cat");
          if (!built) { this._renderTree(); }
        }
        return;
      }
      // Archive parent toggle (ISSUE 1): expand/collapse the inner-file
      // children rolled up under an archive row.
      const arcHeader = ev.target.closest(".archive > .arc-header");
      if (arcHeader) {
        const archive = arcHeader.parentElement;
        const arcPath = archive.dataset.arc;
        // Clicking the row OPENS the archive's composite (all inners) in ONE
        // click; only the twist (chevron) toggles expand to pick individual
        // parts. ("click the top level and just open all inners" — owner.)
        const onTwist = ev.target.closest(".arc-twist");
        if (!onTwist) {
          const entry = this._findEntry(arcPath) || { path: arcPath };
          this._setActive(arcPath);
          if (window.bus) window.bus.emit("asset.opened", { path: arcPath, entry });
          return;
        }
        const open = archive.classList.toggle("open");
        this._expanded["arc:" + arcPath] = open;
        arcHeader.setAttribute("aria-expanded", open ? "true" : "false");
        saveExpandedState(this._expanded);
        // Lazy-build children on first open (parity with the group lazy build).
        if (open) {
          const childrenEl = archive.querySelector(".arc-children");
          const built = childrenEl && childrenEl.querySelector("li.item");
          if (!built) { this._renderTree(); }
        }
        return;
      }
      const item = ev.target.closest(".item");
      if (item) {
        const path = item.dataset.path;
        const entry = this._findEntry(path);
        const sel = window.psoSelection;

        // Multi-select branch (2026-04-25):
        //   Ctrl/Cmd+click   toggle this path in the selection set
        //   Shift+click      extend selection from anchor → this path
        //                    using the current rendered order
        //   plain click      clear selection, set anchor, open asset
        if (sel && (ev.ctrlKey || ev.metaKey)) {
          ev.preventDefault();
          ev.stopPropagation();
          sel.toggle(path);
          this._selAnchor = path;
          item.classList.toggle("ms-selected", sel.has(path));
          // Don't open the file on Ctrl-click; user is building a batch.
          return;
        }
        if (sel && ev.shiftKey && this._selAnchor) {
          ev.preventDefault();
          ev.stopPropagation();
          const order = this._lastRenderedPaths;
          const a = order.indexOf(this._selAnchor);
          const b = order.indexOf(path);
          if (a >= 0 && b >= 0) {
            const lo = Math.min(a, b), hi = Math.max(a, b);
            const range = order.slice(lo, hi + 1);
            // Additive shift-click: add range to existing selection.
            // (Pure replace-with-range would be a worse UX once the
            // user has built up a batch from multiple Ctrl-clicks.)
            for (const p of range) sel.add(p);
            // Repaint highlights for the visible items.
            this._onSelectionChanged();
          }
          return;
        }
        // Plain click → clear selection (unless empty), set new anchor.
        if (sel && sel.size() > 0) {
          sel.clear();
          this._onSelectionChanged();
        }
        this._selAnchor = path;
        this._openLeaf(item, path, entry);
      }
    }

    // Open a leaf: mark it active (persistent highlight + aria-current)
    // and route through the bus. Shared by mouse click + keyboard Enter.
    _openLeaf(item, path, entry) {
      this._setActive(path);

      // Bus is the public IPC surface — app.js subscribes and
      // decides whether to delegate to openFile, model viewer, etc.
      if (window.bus) {
        window.bus.emit("asset.opened", { path, entry });
        // When the manifest is the lite shape (Phase 0.5 perf),
        // hydrate the FULL entry detail in the background and re-emit so
        // consumers can upgrade what they rendered. Fire-and-forget.
        if (window.PSOManifest && typeof window.PSOManifest.isLite === "function"
            && window.PSOManifest.isLite()
            && typeof window.PSOManifest.fetchEntryDetail === "function") {
          window.PSOManifest.fetchEntryDetail(path).then(function (full) {
            if (full) {
              window.bus.emit("asset.detail", { path, entry: full });
            }
          }).catch(function () {
            // Silent — the lite-shape entry already drove the open.
          });
        }
      }
    }

    // Persistent active-leaf highlight. Only one item is active at a
    // time; survives re-render via this._activePath.
    _setActive(path) {
      this._activePath = path;
      if (!this._bodyEl) return;
      const items = this._bodyEl.querySelectorAll("li.item");
      for (const li of items) {
        const on = li.dataset.path === path;
        li.classList.toggle("active", on);
        if (on) li.setAttribute("aria-current", "true");
        else li.removeAttribute("aria-current");
      }
    }

    // Keyboard navigation over the flat list of visible items + group
    // headers. ArrowUp/Down move the roving tabindex; Enter/Space opens
    // a leaf or toggles a group; ArrowLeft/Right collapse/expand groups.
    _onBodyKeydown(ev) {
      const focusables = Array.from(
        this._bodyEl.querySelectorAll(".group > .header, .archive > .arc-header, li.item")
      ).filter((el) => el.offsetParent !== null);
      if (!focusables.length) return;
      const active = this.shadowRoot.activeElement
        || this._bodyEl.querySelector('[tabindex="0"]');
      let idx = focusables.indexOf(active);

      const focusAt = (i) => {
        const clamped = Math.max(0, Math.min(focusables.length - 1, i));
        for (const el of focusables) el.setAttribute("tabindex", "-1");
        const tgt = focusables[clamped];
        tgt.setAttribute("tabindex", "0");
        tgt.focus();
        tgt.scrollIntoView({ block: "nearest" });
      };

      switch (ev.key) {
        case "ArrowDown": ev.preventDefault(); focusAt(idx + 1); break;
        case "ArrowUp":   ev.preventDefault(); focusAt(idx - 1); break;
        case "Home":      ev.preventDefault(); focusAt(0); break;
        case "End":       ev.preventDefault(); focusAt(focusables.length - 1); break;
        case "ArrowRight": {
          const hdr = active && active.closest && active.closest(".group > .header");
          if (hdr && !hdr.parentElement.classList.contains("expanded")) {
            ev.preventDefault();
            hdr.click();
            break;
          }
          const arc = active && active.closest && active.closest(".archive > .arc-header");
          if (arc && !arc.parentElement.classList.contains("open")) {
            ev.preventDefault();
            arc.click();
          }
          break;
        }
        case "ArrowLeft": {
          const hdr = active && active.closest && active.closest(".group > .header");
          if (hdr && hdr.parentElement.classList.contains("expanded")) {
            ev.preventDefault();
            hdr.click();
            break;
          }
          const arc = active && active.closest && active.closest(".archive > .arc-header");
          if (arc && arc.parentElement.classList.contains("open")) {
            ev.preventDefault();
            arc.click();
          }
          break;
        }
        case "Enter":
        case " ": {
          if (!active) break;
          ev.preventDefault();
          if (active.classList.contains("header")
              || active.classList.contains("arc-header")) {
            active.click();
          } else if (active.classList.contains("item")) {
            const path = active.dataset.path;
            const entry = this._findEntry(path);
            this._selAnchor = path;
            this._openLeaf(active, path, entry);
          }
          break;
        }
        default: break;
      }
    }

    _findEntry(path) {
      const all = window.PSOManifest.entries();
      for (const e of all) {
        if (e && e.path === path) return e;
      }
      return null;
    }

    _renderPlaceholder(title, hint, withRefresh) {
      this._bodyEl.innerHTML = `
        <div class="placeholder">
          <div>${esc(title)}</div>
          ${hint ? `<div class="hint">${esc(hint)}</div>` : ""}
          ${withRefresh ? `<button type="button" data-action="refresh">refresh</button>` : ""}
        </div>
      `;
      const btn = this._bodyEl.querySelector("button[data-action=refresh]");
      if (btn) btn.addEventListener("click", this._onRefreshClick);
      this._statsEl.textContent = "";
    }

    // Build the HTML for a single leaf <li>. ``ctx`` carries the shared
    // render state (selection + rendered-path order); ``opts.inner`` marks
    // an archive-child row (it shows the short inner name, not the path).
    _leafHtml(entry, ctx, opts) {
      opts = opts || {};
      const meta = [
        entry.format,
        fmtSize(entry.size),
        entry.parsable && entry.parsable !== "yes" ? entry.parsable : "",
      ].filter(Boolean).join(" · ");
      let matchedHtml = "";
      if (Array.isArray(entry.matched_textures) && entry.matched_textures.length) {
        const best = entry.matched_textures[0];
        const otherCount = entry.matched_textures.length - 1;
        const tail = otherCount > 0 ? ` (+${otherCount})` : "";
        matchedHtml = `<span class="matched">→ ${esc(best.path)}${tail}</span>`;
      }
      const cat = entry.category || "unknown";
      const isSelected = ctx.sel && ctx.sel.has(entry.path);
      const isActive = entry.path === this._activePath;
      const cls = "item cat-" + esc(cat)
        + (entry.parsable === "no" ? " parsable-no" : "")
        + (isSelected ? " ms-selected" : "")
        + (isActive ? " active" : "");
      ctx.renderedPaths.push(entry.path);
      const pill = `<span class="cat-pill">${esc(entry.format || cat || "?")}</span>`;

      // Human-readable primary label + dim raw-filename subtitle. For an
      // archive child the raw subtitle is the short inner filename; for a
      // loose file it's the bare basename. The friendly name is always
      // present (asset_names.js never returns blank).
      const friendly = friendlyName(entry);
      const raw = opts.inner ? rawLabel(entry.path)
                             : String(entry.path).replace(/\\/g, "/").split("/").pop();
      // Only show the raw subtitle when it differs from the friendly label
      // (avoids "Booma / Booma" duplication when no friendly name applies).
      const showRaw = raw && raw.toLowerCase() !== friendly.toLowerCase();

      return [
        `    <li class="${cls}" data-path="${esc(entry.path)}" title="${esc(entry.path)}"` +
        ` role="treeitem" tabindex="-1"${isActive ? ' aria-current="true"' : ""}>`,
        `      ${pill}<span class="friendly">${esc(friendly)}</span>`,
        showRaw ? `      <span class="raw">${esc(raw)}</span>` : "",
        `      <span class="meta">${esc(meta)}</span>`,
        `      ${matchedHtml}`,
        `    </li>`,
      ].filter(Boolean).join("\n");
    }

    // Build the HTML for one archive node: a collapsible <li> whose
    // children are the inner files, rolled up under a single parent row
    // (instead of flat "<archive>#<inner>" siblings). ``arc`` is a record
    // from splitArchives(). The parent label uses the bare archive entry's
    // friendly name when present, else a prettified archive basename.
    _archiveHtml(arc, ctx, opts) {
      opts = opts || {};
      const archBase = archiveBasename(arc.archivePath);
      // Friendly parent name: prefer the bare archive entry's curated name,
      // else the resolver on the archive path itself (covers e.g. an .afs
      // whose curated name lives on a child), else prettified basename.
      let parentFriendly;
      if (arc.archiveEntry) parentFriendly = friendlyName(arc.archiveEntry);
      else parentFriendly = friendlyName({ path: arc.archivePath });
      const showRawArc = archBase && archBase.toLowerCase() !== parentFriendly.toLowerCase();

      // Persisted open/closed state keyed by archive path. Single-inner
      // archives default OPEN so the obvious inner is visible at a glance;
      // multi-inner archives default to the user's saved state (collapsed).
      const single = arc.children.length === 1;
      const arcKey = "arc:" + arc.archivePath;
      const isOpen = opts.forceOpen
        ? true
        : (arcKey in this._expanded) ? !!this._expanded[arcKey] : single;

      const n = arc.children.length;
      const out = [];
      out.push(
        `    <li class="archive${isOpen ? " open" : ""}" data-arc="${esc(arc.archivePath)}" role="treeitem">`,
        `      <div class="arc-header" role="button" aria-expanded="${isOpen}" tabindex="-1" title="${esc(arc.archivePath)}">`,
        `        <span class="arc-twist">▶</span>`,
        `        <span class="arc-name">` +
          `<span class="friendly">${esc(parentFriendly)}</span>` +
          (showRawArc ? `<span class="raw">${esc(archBase)}</span>` : "") +
        `</span>`,
        `        <span class="arc-count">${n} inner${n === 1 ? "" : "s"}</span>`,
        `      </div>`,
        `      <ul class="arc-children" role="group">`,
      );
      // Build child leaves only when open (perf parity with group lazy-build).
      if (isOpen) {
        for (const child of arc.children) {
          out.push(this._leafHtml(child, ctx, { inner: true }));
        }
      }
      out.push(`      </ul>`, `    </li>`);
      return out.join("\n");
    }

    _renderTree() {
      const allEntries = window.PSOManifest.entries();

      // Update per-tab counts against the full manifest so the user
      // always sees coverage regardless of the active inferred-category
      // filter.
      this._updateTabCounts(allEntries);

      if (allEntries.length === 0) {
        this._renderPlaceholder(
          "Manifest is empty.",
          "No assets were discovered under the install root.",
          true,
        );
        return;
      }

      const q = this._search;
      // Resolve the active tab into a canonical-category allowlist.
      const activeTab = TAB_FILTERS.find((t) => t.key === this._filter);
      const tabAllow = activeTab && activeTab.match ? new Set(activeTab.match) : null;

      // Apply the tab filter first (canonical category), then bucket by
      // the inferred display label.
      const inScope = tabAllow
        ? allEntries.filter((e) => e && !e.deprecated && tabAllow.has(e.category || "unknown"))
        : allEntries;
      // Hide the "Maps / Terrain" area-data bucket from the asset tree in
      // every scope EXCEPT the dedicated Floors tab (which explicitly wants
      // category==="map"). The Map + Floor editors own that data, so it
      // doesn't belong in the asset browser.
      const isFloorTab = !!(tabAllow && tabAllow.has("map"));
      const { labels, byLabel } = bucketEntries(inScope, { excludeMap: !isFloorTab });

      let totalShown = 0;
      let totalEntries = 0;

      // Reset rendered-paths order tracker for shift-click range support.
      const renderedPaths = [];
      const sel = window.psoSelection;

      const parts = [];
      for (const label of labels) {
        const list = byLabel[label];
        // Search matches the raw path OR the friendly name so a user can
        // find "Booma" even though the file is bm_ene_re8_b_beast.bml. The
        // path match keeps archive-name searches working across nesting
        // (an inner's path still contains the archive basename).
        const filtered = q
          ? list.filter((e) =>
              e.path.toLowerCase().includes(q)
              || friendlyName(e).toLowerCase().includes(q))
          : list;
        totalEntries += list.length;
        totalShown += filtered.length;

        // Persistent expansion key uses the display label so user state
        // survives manifest rebuilds. When the user types a search,
        // matched buckets auto-expand; single-tab scopes default open
        // for less-clicking.
        const expandKey = label;
        const isExpanded = q
          ? filtered.length > 0
          : (tabAllow && tabAllow.size === 1 && labels.length === 1) ? true
          : !!this._expanded[expandKey];

        parts.push(
          `<div class="group${isExpanded ? " expanded" : ""}" data-cat="${esc(expandKey)}">`,
          `  <div class="header" role="button" aria-expanded="${isExpanded}" tabindex="-1">`,
          `    <span class="twist">▶</span>`,
          `    <span class="label">${esc(label)}</span>`,
          `    <span class="count">${filtered.length}${q && filtered.length !== list.length ? " / " + list.length : ""}</span>`,
          `  </div>`,
          `  <ul class="items" role="group">`,
        );
        // 2026-06-19 perf: only build the <li> leaves for groups that are
        // actually open (expanded or matched by a search). Previously all
        // ~9400 leaves were rendered into the DOM (28k shadow nodes) and
        // merely display:none'd, which is the main on-load / filter jank.
        // Collapsed groups now render an empty <ul>; the group toggle in
        // _onBodyClick triggers a re-render to build a group on first
        // expand. renderedPaths only tracks what's in the DOM, which is
        // exactly the set shift-click range-select can operate on.
        if (!isExpanded) {
          // leave the <ul> empty; build on expand
        } else if (filtered.length === 0) {
          parts.push(
            `    <li class="empty-cat">${q ? "no matches" : "(empty)"}</li>`,
          );
        } else {
          // ISSUE 1: roll archive inners up under a collapsible parent row
          // instead of rendering them as flat "<archive>#<inner>" siblings.
          // splitArchives groups the filtered entries by parent_archive;
          // loose (non-inner) files render exactly as before.
          const ctx = { sel, renderedPaths };
          const { archives, loose } = splitArchives(filtered);

          // Interleave loose files + archive nodes in one path-sorted
          // stream so the bucket reads in a stable, predictable order.
          const stream = [];
          for (const e of loose) stream.push({ sortPath: e.path, kind: "leaf", e });
          for (const arc of archives) {
            // Sort an archive next to where its bare path would fall.
            stream.push({ sortPath: arc.archivePath, kind: "arc", arc });
          }
          stream.sort((a, b) => (a.sortPath < b.sortPath ? -1 : a.sortPath > b.sortPath ? 1 : 0));

          for (const node of stream) {
            if (node.kind === "leaf") {
              parts.push(this._leafHtml(node.e, ctx, { inner: false }));
            } else if (node.arc.children.length === 1) {
              // Single-inner archive (the common case): render the lone inner
              // as ONE clickable row (clean inner name, opens on click)
              // instead of a redundant parent+child that forced
              // expand-then-click-inner. ("shit ux" — owner, 2026-06-20.)
              parts.push(this._leafHtml(node.arc.children[0], ctx, { inner: true }));
            } else {
              // Multi-inner: keep the collapsible parent, but its row OPENS
              // the composite on click (the twist toggles expand) — see
              // _onBodyClick. During search, force open so matches show.
              parts.push(this._archiveHtml(node.arc, ctx, { forceOpen: !!q }));
            }
          }
        }
        parts.push(`  </ul>`, `</div>`);
      }
      this._lastRenderedPaths = renderedPaths;
      this._bodyEl.innerHTML = parts.join("\n");

      // Roving-tabindex seed: make the first focusable element reachable
      // by Tab so keyboard users can enter the tree. ArrowUp/Down then
      // move focus (see _onBodyKeydown).
      const firstFocusable = this._bodyEl.querySelector(".group > .header, li.item");
      if (firstFocusable) firstFocusable.setAttribute("tabindex", "0");

      // Restore scroll position from localStorage. We do this on the
      // next tick so the new innerHTML actually has its final layout
      // before we set scrollTop. _onBodyScroll keeps the LS in sync.
      const savedScroll = loadScrollPos();
      if (savedScroll > 0) {
        // requestAnimationFrame keeps us out of the layout-thrash window.
        requestAnimationFrame(() => {
          if (!this._bodyEl) return;
          this._bodyEl.scrollTop = Math.min(savedScroll, this._bodyEl.scrollHeight);
        });
      }

      // Status footer reflects: {shown of total} {tab scope} · "{query}"
      // 2026-06-20 taxonomy reconciliation: the tree is the user's primary
      // navigation, and its top-level collapsible buckets (Bosses/Enemies/
      // NPCs/…) ARE the "categories" the user browses. We label that count
      // "categories" everywhere (footer here + the #assetTreeCoverage
      // header below) so the UI tells ONE story instead of the old
      // "10 categories (header) vs 20 groups (footer)" contradiction.
      const tabLbl = activeTab && activeTab.match ? ` · ${activeTab.label}` : "";
      const bucketLbl = labels.length > 1 ? ` · ${labels.length} categories` : "";
      this._statsEl.textContent = q
        ? `${totalShown} of ${totalEntries} shown${tabLbl}${bucketLbl} · "${q}"`
        : (tabAllow ? `${totalShown} of ${totalEntries} shown${tabLbl}${bucketLbl}`
                    : `${totalEntries} entries${bucketLbl}`);

      // Single source of truth for the "All assets" pane-header coverage
      // label (#assetTreeCoverage, light-DOM sibling of <pso-asset-tree>).
      // Always reflects the FULL unfiltered taxonomy so the header stays
      // stable while the footer tracks the active filter/search scope.
      this._updateCoverageHeader(allEntries);
    }
  }

  customElements.define("pso-asset-tree", PsoAssetTree);
})();
