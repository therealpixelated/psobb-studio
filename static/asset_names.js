// PSOBB Modding Suite — friendly asset-name resolver.
// =====================================================================
// Self-contained name map + resolver. The asset tree (and any other
// consumer) calls window.PSOAssetNames.friendly(entry) to get a
// HUMAN-READABLE label for an asset row. NEVER returns blank — falls
// back to a prettified filename so something always shows.
//
// Resolution order (first hit wins):
//   1. entry.display_name        — curated psov2 name emitted by the
//                                  server manifest (data_meta/psov2_names.json).
//                                  This is the authoritative source for
//                                  Enemies / Bosses / NPCs / Objects /
//                                  Player Bodies / Weapons / Maps.
//   2. PHOTON_BLAST table        — the PSO Mag Photon Blasts (pm_b_* /
//                                  pm_mahoujin). These are NOT in the psov2
//                                  catalogs, so they're hard facts here.
//   3. PREFIX/SUFFIX inference   — best-effort prettifier so plain files
//                                  (and uncovered inners) still read nicely.
//   4. prettified basename       — last resort: strip the dir + ext, turn
//                                  separators into spaces, title-case.
//
// All sources are committed FACTS (psov2 catalogs / owner-confirmed PB
// names / the documented plX class table). No _reference at runtime.
// =====================================================================

(function () {
  "use strict";

  if (window.PSOAssetNames) return; // idempotent

  // ── Photon Blasts (PSO Mag PBs) — owner-confirmed, not in psov2 ──────
  // pm_b_<id> bodies + the magic-circle effect. Matched on the inner /
  // file basename (lowercased, ext-stripped) by substring so the same
  // entry covers pm_b_estlla.nj, pm_b_estlla_body.njm, etc.
  const PHOTON_BLAST = [
    ["pm_b_estlla", "Estlla (Photon Blast)"],
    ["pm_b_mylla",  "Mylla (Photon Blast)"],
    ["pm_b_youlla", "Youlla (Photon Blast)"],
    ["pm_b_pilla",  "Pilla (Photon Blast)"],
    ["pm_b_golla",  "Golla (Photon Blast)"],
    ["pm_b_leilla", "Leilla (Photon Blast)"],
    ["pm_b_farlla", "Farlla (Photon Blast)"],
    ["pm_b_twins",  "Twins (Photon Blast)"],
    ["pm_mahoujin", "Magic Circle (Photon Blast)"],
  ];

  // ── Player class table (plX -> class) ──────────────────────────────
  // Authoritative mapping cross-checked against
  // _reference/psov2/public/js/AssetPlayer.js (the body/head/cap inners
  // each named class pulls from its plXnj.bml). The psov2 manifest names
  // already cover the on-disk plX files; this table is the graceful
  // fallback for any plX inner the server didn't curate, and it
  // distinguishes Body / Headgear / Hair / Cap parts (the psov2 names
  // collapse all parts to the bare class name).
  const PLAYER_CLASS = {
    a: "HUmar", b: "HUnewearl", c: "HUcast",
    d: "RAmar", e: "RAcast", f: "RAcaseal",
    g: "FOmarl", h: "FOnewm", i: "FOnewearl",
  };
  // pl<X><part><NN> -> part label
  const PLAYER_PART = {
    bdy: "Body", hed: "Headgear", hai: "Hair",
    cap: "Cap", arm: "Arms", tex: "Texture",
  };

  // ── light prefix/suffix inference for uncovered files ──────────────
  // Conservative: only fires on well-known PSOBB filename conventions so
  // we never invent a misleading name. Returns null when nothing matches
  // (caller then prettifies the basename).
  function inferFromName(base) {
    // base is lowercased, no extension, no dir, no "#NNNN_" synth prefix.

    // Player bodies/headgear: pl<class><part><NN>
    let m = base.match(/^pl([a-i])(bdy|hed|hai|cap|arm|tex)(\d*)/);
    if (m) {
      const cls = PLAYER_CLASS[m[1]];
      const part = PLAYER_PART[m[2]];
      if (cls) return part ? `${cls} ${part}` : cls;
    }
    // Player model BML: pl<class>nj
    m = base.match(/^pl([a-i])nj$/);
    if (m && PLAYER_CLASS[m[1]]) return `${PLAYER_CLASS[m[1]]} Model`;

    // Photon-blast catch-all handled by PHOTON_BLAST table above.

    // Enemy model bml: bm_ene_*  /  NPC: bm_npc_*
    m = base.match(/^bm_ene_(.+)$/);
    if (m) return prettifyToken(m[1]) + " (Enemy)";
    m = base.match(/^bm_npc_(.+)$/);
    if (m) return prettifyToken(m[1]) + " (NPC)";
    m = base.match(/^bm_obj_(.+)$/);
    if (m) return prettifyToken(m[1]) + " (Object)";

    return null;
  }

  // Turn a raw token like "de_rol_le" / "re8_b_beast" into "De Rol Le".
  function prettifyToken(tok) {
    return String(tok)
      .replace(/_+/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase())
      .trim();
  }

  // Strip directory, "#NNNN_" / "#" synth prefix, and extension; return
  // the bare lowercased stem used for table lookups.
  function innerStem(path) {
    let p = String(path || "").replace(/\\/g, "/");
    p = p.split("/").pop();          // basename (handles archive path too)
    if (p.indexOf("#") >= 0) {
      // For "<archive>#<inner>" use the inner; for synth AFS the inner
      // carries a leading "NNNN_" index — drop it.
      let inner = p.split("#").pop();
      inner = inner.replace(/^\d{1,5}_/, "");
      p = inner;
    }
    // drop extension
    const dot = p.lastIndexOf(".");
    if (dot > 0) p = p.slice(0, dot);
    return p.toLowerCase();
  }

  // Prettify a full path's basename as the always-present last resort.
  function prettifyBasename(path) {
    const stem = innerStem(path);
    if (!stem) return String(path || "");
    return prettifyToken(stem);
  }

  // ── public resolver ────────────────────────────────────────────────
  // Returns a non-empty friendly label for an entry (or a bare path
  // string). Pass the lite/full manifest entry so curated display_name
  // is honoured.
  function friendly(entryOrPath) {
    const entry = (entryOrPath && typeof entryOrPath === "object")
      ? entryOrPath : null;
    const path = entry ? entry.path : String(entryOrPath || "");

    // 1. server-curated name wins.
    if (entry && typeof entry.display_name === "string" && entry.display_name) {
      return entry.display_name;
    }

    const stem = innerStem(path);

    // 2. Photon Blasts (substring so _body / _t variants all hit).
    for (const [needle, name] of PHOTON_BLAST) {
      if (stem.indexOf(needle) >= 0) return name;
    }

    // 3. convention inference.
    const inf = inferFromName(stem);
    if (inf) return inf;

    // 4. prettified basename — never blank.
    return prettifyBasename(path) || path || "";
  }

  // True when friendly() produced something other than the raw filename
  // (so the tree can decide whether to show the filename as a subtitle).
  function hasFriendlyName(entryOrPath) {
    const entry = (entryOrPath && typeof entryOrPath === "object")
      ? entryOrPath : null;
    const path = entry ? entry.path : String(entryOrPath || "");
    if (entry && entry.display_name) return true;
    const stem = innerStem(path);
    for (const [needle] of PHOTON_BLAST) {
      if (stem.indexOf(needle) >= 0) return true;
    }
    return inferFromName(stem) !== null;
  }

  window.PSOAssetNames = Object.freeze({
    friendly,
    hasFriendlyName,
    // exposed for tests / debugging
    _innerStem: innerStem,
    _prettifyBasename: prettifyBasename,
  });
})();
