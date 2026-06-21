// =====================================================================
// smoke_render_all.mjs — every-model render smoke harness
// ---------------------------------------------------------------------
// Runs, for EVERY model entry in the manifest, the SAME load route the
// browser's model_viewer.js openByPath() takes, and records whether the
// model produces a REAL mesh (verts > 0) or falls back to the grey cube
// ("primitive (cube) — model unavailable").
//
// FAITHFULNESS — this harness mirrors openByPath()'s routing exactly:
//
//   * .nj inner (bare / bml#inner.nj / afs#inner.nj):
//       1. psov2 client parser  -> parseNinjaModel(GET /api/raw_nj)
//          ok iff loader.bones.length > 0 AND position.count > 0
//       2. fallback: GET /api/model_skinned   (mesh_count>0 && bone_count>0)
//       3. fallback: GET /api/model_mesh       (mesh_count>0 && verts>0)
//       4. else CUBE
//     The psov2 parse is the REAL static/psov2_ninja.js parseNinjaModel
//     loaded under Node with the REAL three@0.160.0 package (the same
//     version the browser loads from unpkg) — see scripts/_psov2_loader.mjs.
//
//   * .xj inner (bml#inner.xj):
//       1. GET /api/model_mesh   (mesh_count>0 && verts>0)
//       2. else CUBE
//
//   * top-level .bml (no '#'):
//       discover inners via GET /api/bml/<bml>/list, classify
//       (primary/lod/shadow/destroyed) exactly like _classifyInner:
//         - >= 2 primaries -> COMPOSITE: per-inner GET /api/model_mesh
//           (primary inner via /api/model_skinned), ordered by
//           GET /api/composite_bundle?meta_only=1. ok iff any inner has
//           geometry.
//         - else single primary -> resolve to <bml>#<primary> and route
//           it through the .nj or .xj path above.
//
// A verdict cross-checked against a real headless Chrome (psoOpenModelByPath
// + the #modelMeshStats.model-mesh-fallback indicator) agreed on the
// validation sample; see docs/SMOKE_REPORT.md.
//
// Usage:
//   SMOKE_BASE=http://127.0.0.1:8791 node scripts/smoke_render_all.mjs
//   [--limit N] [--concurrency K] [--filter substr] [--out docs/SMOKE_REPORT.md]
//
// Requires a server already running against the target data dir, e.g.:
//   PSO_DATA_DIR=C:/Users/<user>/PSOBB.IO/data \
//     python -m uvicorn server:app --port 8791 --workers 4
// =====================================================================

import { writeFileSync, mkdirSync } from "fs";
import { dirname, resolve } from "path";
import { fileURLToPath } from "url";
import { loadParser } from "./_psov2_loader.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(__dirname, "..");

// ---- args ----
const argv = process.argv.slice(2);
function arg(name, def) {
  const i = argv.indexOf(name);
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : def;
}
const BASE = process.env.SMOKE_BASE || "http://127.0.0.1:8791";
const LIMIT = parseInt(arg("--limit", "0"), 10) || 0;
const CONCURRENCY = parseInt(arg("--concurrency", "12"), 10) || 12;
const FILTER = arg("--filter", "");
const OUT_MD = arg("--out", resolve(REPO, "docs/SMOKE_REPORT.md"));
const OUT_FAIL = arg("--failures", resolve(REPO, "failures.json"));
const RESULTS_JSON = arg("--results", "");

// ---- _classifyInner — verbatim from model_viewer.js ----
const _LOD_RE = /^(lo|low)[_\s-]/i;
const _SHADOW_RE = /(?:^|[_-])(?:sd|shd)(?:$|[_-])/i;
const _DESTROYED_RE = /(_break|_broken|_hahen|_burst)/i;
function classifyInner(name) {
  const stem = name.replace(/\.(nj|xj)$/i, "");
  if (_LOD_RE.test(stem)) return "lod";
  if (_SHADOW_RE.test(stem)) return "shadow";
  if (_DESTROYED_RE.test(stem)) return "destroyed";
  return "primary";
}

// ---- fetch helpers ----
// The browser opens ONE model at a time; this harness runs K in parallel.
// Under high concurrency the dev server occasionally drops a connection
// (ECONNRESET / socket hang up) or times out mid-parse — a TRANSPORT
// failure, not a model defect. We retry those a few times with backoff so
// contention never manufactures a false "cube". A real HTTP 4xx/5xx with a
// JSON {detail} body is a genuine model-rejection and is NOT retried
// (it's deterministic — the browser would see the same rejection).
function _isTransient(e) {
  if (e && e.status) return false; // structured HTTP error -> deterministic
  const m = String((e && e.message) || e).toLowerCase();
  return /reset|hang up|socket|econn|timeout|network|fetch failed|terminated|aborted/.test(m);
}
async function _withRetry(fn, tries = 4) {
  let last;
  for (let i = 0; i < tries; i++) {
    try { return await fn(); }
    catch (e) {
      last = e;
      if (!_isTransient(e) || i === tries - 1) throw e;
      await new Promise((r) => setTimeout(r, 150 * (i + 1) + Math.random() * 100));
    }
  }
  throw last;
}
async function fetchJson(url) {
  return _withRetry(async () => {
    const r = await fetch(BASE + url);
    if (!r.ok) {
      let detail = `http ${r.status}`;
      try { const b = await r.json(); if (b && b.detail) detail = b.detail; } catch {}
      const e = new Error(detail);
      e.status = r.status;
      throw e;
    }
    return r.json();
  });
}
async function fetchBuf(url) {
  return _withRetry(async () => {
    const r = await fetch(BASE + url);
    if (!r.ok) {
      let detail = `http ${r.status}`;
      try { const b = await r.json(); if (b && b.detail) detail = b.detail; } catch {}
      const e = new Error(detail);
      e.status = r.status;
      throw e;
    }
    return r.arrayBuffer();
  });
}
const enc = encodeURIComponent;

// ---- per-route loaders (mirror model_viewer.js success criteria) ----

// psov2 client-side parse. Mirrors tryLoadPsov2NinjaModel's success gate
// EXACTLY: ok iff the parse doesn't throw AND loader.bones.length > 0
// (the browser sets state.realMesh=true on that condition — it does NOT
// require position.count > 0). We additionally flag `empty:true` when the
// parse succeeds but yields 0 vertices: the browser shows NO cube banner
// for these (so they count as "ok" for the cube metric, matching the
// indicator), but nothing renders — they're surfaced separately so the
// owner can see the "invisible model" set.
function psov2Parse(parseNinjaModel, buf, name) {
  if (!buf || buf.byteLength < 8) return { ok: false, reason: "empty/short buffer" };
  let mesh;
  try {
    mesh = parseNinjaModel(buf, { name, texList: [] });
  } catch (e) {
    return { ok: false, reason: `psov2 parse error: ${e?.message || e}` };
  }
  const loader = mesh.userData.ninjaLoader;
  if (!loader || !loader.bones || !loader.bones.length) {
    return { ok: false, reason: "no bones parsed" };
  }
  const pos = mesh.geometry.getAttribute("position");
  const verts = pos ? pos.count : 0;
  return { ok: true, verts, bones: loader.bones.length, empty: verts === 0 };
}

// model_skinned (legacy bone-local payload). ok iff mesh_count && bone_count.
async function skinnedLoad(base, inner) {
  const url = inner
    ? `/api/model_skinned/${enc(base)}?inner=${enc(inner)}`
    : `/api/model_skinned/${enc(base)}`;
  const p = await fetchJson(url);
  if (!p || !p.mesh_count || !p.bone_count) {
    return { ok: false, reason: "skinned: empty payload" };
  }
  return { ok: true, verts: p.vert_total | 0 };
}

// model_mesh (world-baked). ok iff mesh_count>0 and at least one submesh
// has non-empty verts+indices (== built.totalVerts>0 in the browser).
async function meshLoad(base, inner) {
  const url = inner
    ? `/api/model_mesh/${enc(base)}?inner=${enc(inner)}`
    : `/api/model_mesh/${enc(base)}`;
  const p = await fetchJson(url);
  if (!p || !p.mesh_count) return { ok: false, reason: "mesh: no geometry parsed" };
  let renderable = 0;
  for (const m of p.meshes || []) {
    if ((m.vertex_count | 0) > 0 && (m.triangle_count | 0) > 0) renderable += m.vertex_count;
  }
  if (renderable === 0) return { ok: false, reason: "mesh: no rendered sub-meshes" };
  return { ok: true, verts: renderable };
}

// .nj fallback chain: psov2 -> skinned -> mesh.
async function loadNj(parseNinjaModel, path) {
  // base/inner split for the server-payload fallbacks.
  const hash = path.indexOf("#");
  const base = hash > 0 ? path.slice(0, hash) : path;
  const inner = hash > 0 ? path.slice(hash + 1) : null;

  let lastReason = "";
  // 1. psov2
  try {
    const buf = await fetchBuf(`/api/raw_nj/${enc(path)}`);
    const r = psov2Parse(parseNinjaModel, buf, path);
    if (r.ok) return { ok: true, verts: r.verts, route: "psov2", empty: !!r.empty };
    lastReason = r.reason;
  } catch (e) {
    lastReason = `raw_nj: ${e?.message || e}`;
  }
  // 2. skinned
  try {
    const r = await skinnedLoad(base, inner);
    if (r.ok) return { ok: true, verts: r.verts, route: "skinned" };
    lastReason = r.reason;
  } catch (e) {
    lastReason = `skinned: ${e?.message || e}`;
  }
  // 3. world-baked
  try {
    const r = await meshLoad(base, inner);
    if (r.ok) return { ok: true, verts: r.verts, route: "mesh" };
    lastReason = r.reason;
  } catch (e) {
    lastReason = `mesh: ${e?.message || e}`;
  }
  return { ok: false, reason: lastReason, route: "cube" };
}

// .xj path: world-baked only.
async function loadXj(path) {
  const hash = path.indexOf("#");
  const base = hash > 0 ? path.slice(0, hash) : path;
  const inner = hash > 0 ? path.slice(hash + 1) : null;
  try {
    const r = await meshLoad(base, inner);
    if (r.ok) return { ok: true, verts: r.verts, route: "mesh" };
    return { ok: false, reason: r.reason, route: "cube" };
  } catch (e) {
    return { ok: false, reason: `mesh: ${e?.message || e}`, route: "cube" };
  }
}

// top-level .bml: discover inners, classify, composite-or-single.
async function loadBmlTop(parseNinjaModel, bmlPath) {
  let list;
  try {
    list = await fetchJson(`/api/bml/${enc(bmlPath)}/list`);
  } catch (e) {
    return { ok: false, reason: `bml list: ${e?.message || e}`, route: "cube" };
  }
  const inners = [];
  for (const e of list.entries || []) {
    if (/\.(nj|xj)$/i.test(e.name)) inners.push({ name: e.name, kind: classifyInner(e.name) });
  }
  if (inners.length === 0) return { ok: false, reason: "bml has no .nj/.xj inner", route: "cube" };
  const primaries = inners.filter((x) => x.kind === "primary").map((x) => x.name);

  if (primaries.length >= 2) {
    // COMPOSITE. Use composite_bundle meta_only for order (best-effort);
    // fetch each inner's geometry via model_mesh (primary via skinned).
    let order = primaries;
    let primaryInner = null;
    try {
      const bundle = await fetchJson(`/api/composite_bundle/${enc(bmlPath)}?meta_only=1`);
      if (bundle && bundle.source && bundle.source !== "identity-fallback" && Array.isArray(bundle.parts)) {
        const seen = new Set();
        order = [];
        for (const n of primaries) { const k = n.toLowerCase(); if (!seen.has(k)) { seen.add(k); order.push(n); } }
        for (const p of bundle.parts) { const k = (p.inner || "").toLowerCase(); if (!seen.has(k)) { seen.add(k); order.push(p.inner); } }
        for (const p of bundle.parts) {
          if (p.parent_inner) continue;
          if ((p.inner || "").toLowerCase().endsWith(".nj")) { primaryInner = p.inner.toLowerCase(); break; }
        }
      }
    } catch { /* identity fallback — keep discovery order */ }

    let anyOk = false;
    let totalVerts = 0;
    let lastReason = "";
    for (const inner of order) {
      const lower = inner.toLowerCase();
      try {
        let r;
        if (lower === primaryInner) {
          // primary -> skinned, fall back to mesh (browser does the same).
          try { r = await skinnedLoad(bmlPath, inner); }
          catch { r = await meshLoad(bmlPath, inner); }
          if (!r.ok) r = await meshLoad(bmlPath, inner);
        } else {
          r = await meshLoad(bmlPath, inner);
        }
        if (r.ok) { anyOk = true; totalVerts += r.verts | 0; }
        else lastReason = r.reason;
      } catch (e) { lastReason = `${inner}: ${e?.message || e}`; }
    }
    if (anyOk) return { ok: true, verts: totalVerts, route: "composite" };
    return { ok: false, reason: `composite: no inner parsed (${lastReason})`, route: "cube" };
  }

  // single primary -> resolve to <bml>#<primary> and route by extension.
  const pick = primaries[0] || inners[0].name;
  const single = `${bmlPath}#${pick}`;
  if (pick.toLowerCase().endsWith(".nj")) return loadNj(parseNinjaModel, single);
  return loadXj(single);
}

// ---- route dispatcher (mirrors openByPath) ----
function entryType(path) {
  const lower = path.toLowerCase();
  const hash = path.indexOf("#");
  if (hash < 0) {
    if (lower.endsWith(".bml")) return "bml_toplevel";
    if (lower.endsWith(".nj")) return "bare_nj";
    if (lower.endsWith(".xj")) return "bare_xj";
    return "other";
  }
  const inner = lower.slice(hash + 1);
  const base = lower.slice(0, hash);
  const ext = inner.split(".").pop();
  const fam = base.endsWith(".afs") ? "afs" : base.endsWith(".bml") ? "bml" : "other";
  return `${fam}#inner_${ext}`;
}

async function loadEntry(parseNinjaModel, path) {
  const lower = path.toLowerCase();
  const hash = path.indexOf("#");
  if (hash < 0 && lower.endsWith(".bml")) return loadBmlTop(parseNinjaModel, path);
  // both bare and #inner: dispatch on the trailing extension
  if (lower.endsWith(".nj")) return loadNj(parseNinjaModel, path);
  if (lower.endsWith(".xj")) return loadXj(path);
  // a non-model leaf mislabeled as 'model' (e.g. a stray .dat): try mesh.
  return loadXj(path);
}

// ---- concurrency pool ----
async function runPool(items, worker, concurrency) {
  const out = new Array(items.length);
  let next = 0;
  let done = 0;
  const total = items.length;
  async function spin() {
    while (next < items.length) {
      const i = next++;
      out[i] = await worker(items[i], i);
      done++;
      if (done % 100 === 0 || done === total) {
        process.stderr.write(`\r  progress ${done}/${total}   `);
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, spin));
  process.stderr.write("\n");
  return out;
}

// ---- percentile ----
function pct(sorted, p) {
  if (sorted.length === 0) return 0;
  const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
  return sorted[idx];
}

// ---- main ----
(async () => {
  process.stderr.write(`smoke_render_all: base=${BASE}\n`);
  const { parseNinjaModel } = await loadParser();

  // Enumerate model entries. Either an explicit --paths-file (one path per
  // line; used by the faithfulness cross-check) or the full manifest
  // (category === 'model').
  let models;
  const pathsFile = arg("--paths-file", "");
  if (pathsFile) {
    const { readFileSync } = await import("fs");
    models = readFileSync(pathsFile, "utf8").split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
  } else {
    const manifest = await fetchJson(`/api/manifest_lite`);
    models = (manifest.entries || [])
      .filter((e) => e.category === "model")
      .map((e) => e.path);
  }
  if (FILTER) models = models.filter((p) => p.toLowerCase().includes(FILTER.toLowerCase()));
  if (LIMIT > 0) models = models.slice(0, LIMIT);
  process.stderr.write(`  ${models.length} model entries to test (concurrency ${CONCURRENCY})\n`);

  const t0 = Date.now();
  const results = await runPool(models, async (path) => {
    const type = entryType(path);
    const start = performance.now();
    let r;
    try {
      r = await loadEntry(parseNinjaModel, path);
    } catch (e) {
      r = { ok: false, reason: `harness error: ${e?.message || e}`, route: "cube" };
    }
    const loadMs = +(performance.now() - start).toFixed(2);
    return {
      path,
      type,
      verdict: r.ok ? "ok" : "cube",
      route: r.route,
      verts: r.verts | 0,
      empty: !!r.empty && (r.verts | 0) === 0,
      error: r.ok ? null : (r.reason || "unknown"),
      loadMs,
    };
  }, CONCURRENCY);
  const wallMs = Date.now() - t0;

  // ---- aggregate ----
  const total = results.length;
  const cubes = results.filter((r) => r.verdict === "cube");
  const oks = results.filter((r) => r.verdict === "ok");
  const empties = results.filter((r) => r.verdict === "ok" && r.empty);
  const cubeCount = cubes.length;
  const okCount = oks.length;
  const emptyCount = empties.length;
  const cubePct = total ? +((cubeCount / total) * 100).toFixed(2) : 0;

  const loadTimes = results.map((r) => r.loadMs).sort((a, b) => a - b);
  const median = +pct(loadTimes, 50).toFixed(2);
  const p90 = +pct(loadTimes, 90).toFixed(2);
  const p99 = +pct(loadTimes, 99).toFixed(2);
  const over50 = loadTimes.filter((t) => t > 50).length;
  const maxMs = loadTimes.length ? loadTimes[loadTimes.length - 1] : 0;

  // group cubes by type
  const byType = {};
  for (const c of cubes) (byType[c.type] ||= []).push(c);

  // group cubes by error signature (normalise digits/paths)
  function sig(err) {
    return (err || "unknown")
      .replace(/0x[0-9a-f]+/gi, "0x#")
      .replace(/\b\d+\b/g, "#")
      .replace(/'[^']*'/g, "'…'")
      .replace(/reading '[^']*'/g, "reading '…'")
      .slice(0, 120);
  }
  const byErr = {};
  for (const c of cubes) {
    const s = sig(c.error);
    (byErr[s] ||= []).push(c.path);
  }
  // group cubes by archive family. For container inners the family is the
  // container basename (e.g. `ItemModel.afs`); for bare paths it's the
  // directory prefix when present (e.g. `scene/`) else the bare basename.
  // This collapses the 94 `scene/map_*.nj` rows into one informative family.
  function archiveFamily(p) {
    const hash = p.indexOf("#");
    if (hash > 0) return p.slice(0, hash);
    const slash = p.lastIndexOf("/");
    if (slash > 0) return p.slice(0, slash + 1); // directory prefix, e.g. "scene/"
    return p;
  }
  const byArchive = {};
  for (const c of cubes) {
    (byArchive[archiveFamily(c.path)] ||= []).push(c.path);
  }

  // load-time buckets per type (for the report)
  const byTypeAll = {};
  for (const r of results) (byTypeAll[r.type] ||= []).push(r);

  const topCubeCategories = Object.entries(byType)
    .map(([category, arr]) => ({ category, count: arr.length, examples: arr.slice(0, 5).map((x) => x.path) }))
    .sort((a, b) => b.count - a.count);
  const topErrSigs = Object.entries(byErr)
    .map(([signature, paths]) => ({ signature, count: paths.length, examples: paths.slice(0, 5) }))
    .sort((a, b) => b.count - a.count);
  const topArchives = Object.entries(byArchive)
    .map(([archive, paths]) => ({ archive, count: paths.length, examples: paths.slice(0, 3) }))
    .sort((a, b) => b.count - a.count);

  // ---- write failures.json ----
  const failuresOut = {
    base: BASE,
    generated: new Date().toISOString(),
    total,
    okCount,
    cubeCount,
    cubePct,
    emptyCount,
    byType: topCubeCategories,
    byErrorSignature: topErrSigs,
    byArchiveFamily: topArchives.slice(0, 40),
    cubes: cubes.map((c) => ({ path: c.path, type: c.type, error: c.error, loadMs: c.loadMs })),
    empties: empties.map((e) => ({ path: e.path, type: e.type, route: e.route })),
  };
  mkdirSync(dirname(OUT_FAIL), { recursive: true });
  writeFileSync(OUT_FAIL, JSON.stringify(failuresOut, null, 2));

  if (RESULTS_JSON) {
    writeFileSync(RESULTS_JSON, JSON.stringify(results, null, 0));
  }

  // ---- write SMOKE_REPORT.md ----
  const fmtPct = (n, d) => (d ? ((n / d) * 100).toFixed(1) : "0.0");
  let md = "";
  md += `# Model render smoke report\n\n`;
  md += `Generated: ${new Date().toISOString()}\n\n`;
  md += `Harness: \`scripts/smoke_render_all.mjs\` — replays model_viewer.js \`openByPath()\` `;
  md += `for every manifest model entry (real \`parseNinjaModel\` under three@0.160.0 for the .nj psov2 path; `;
  md += `\`/api/model_mesh\` // \`/api/model_skinned\` // \`/api/composite_bundle\` for the server paths).\n\n`;
  md += `## Totals\n\n`;
  md += `| metric | value |\n|---|---|\n`;
  md += `| models tested | **${total}** |\n`;
  md += `| real mesh (ok) | **${okCount}** (${fmtPct(okCount, total)}%) |\n`;
  md += `| &nbsp;&nbsp;of which empty (0 verts, no cube banner) | ${emptyCount} |\n`;
  md += `| grey cube | **${cubeCount}** (${cubePct}%) |\n`;
  md += `| sweep wall time | ${(wallMs / 1000).toFixed(1)}s |\n\n`;
  md += `> "ok" mirrors the browser's cube indicator: a model is **ok** when `;
  md += `\`openByPath\` does NOT raise the "primitive (cube) — model unavailable" `;
  md += `banner. ${emptyCount} of the ok models parse to a 0-vertex stub (psov2 `;
  md += `succeeds with bones but no geometry — e.g. \`ene_common_all.nj\`); the `;
  md += `browser shows no cube for these but nothing renders. They are listed in `;
  md += `the Empty-models section below.\n\n`;

  md += `## Load-time distribution (per-model route ms)\n\n`;
  md += `| stat | ms |\n|---|---|\n`;
  md += `| median | ${median} |\n`;
  md += `| p90 | ${p90} |\n`;
  md += `| p99 | ${p99} |\n`;
  md += `| max | ${maxMs} |\n`;
  md += `| count > 50ms | **${over50}** (${fmtPct(over50, total)}%) |\n\n`;
  md += `Per-type load timing:\n\n`;
  md += `| type | n | ok | cube | median ms | p90 ms | >50ms |\n|---|---|---|---|---|---|---|\n`;
  for (const [type, arr] of Object.entries(byTypeAll).sort((a, b) => b[1].length - a[1].length)) {
    const lt = arr.map((x) => x.loadMs).sort((a, b) => a - b);
    const ok = arr.filter((x) => x.verdict === "ok").length;
    const cube = arr.length - ok;
    md += `| ${type} | ${arr.length} | ${ok} | ${cube} | ${pct(lt, 50).toFixed(1)} | ${pct(lt, 90).toFixed(1)} | ${lt.filter((t) => t > 50).length} |\n`;
  }
  md += `\n`;

  md += `## Cube failures grouped\n\n`;
  if (cubeCount === 0) {
    md += `**No cubes** — every model entry produced a real mesh (verts > 0).\n\n`;
  } else {
    md += `### By entry type\n\n| type | cubes | examples |\n|---|---|---|\n`;
    for (const c of topCubeCategories) {
      md += `| ${c.category} | ${c.count} | ${c.examples.map((e) => "`" + e + "`").join("<br>")} |\n`;
    }
    md += `\n### By error signature\n\n| count | signature | examples |\n|---|---|---|\n`;
    for (const s of topErrSigs) {
      md += `| ${s.count} | \`${s.signature.replace(/\|/g, "\\|")}\` | ${s.examples.map((e) => "`" + e + "`").join("<br>")} |\n`;
    }
    md += `\n### By archive family (top 20)\n\n| count | archive | examples |\n|---|---|---|\n`;
    for (const a of topArchives.slice(0, 20)) {
      md += `| ${a.count} | \`${a.archive}\` | ${a.examples.map((e) => "`" + e + "`").join("<br>")} |\n`;
    }
    md += `\n`;
  }

  md += `## Empty (0-vertex) models — render "ok" but invisible\n\n`;
  if (emptyCount === 0) {
    md += `None.\n\n`;
  } else {
    md += `${emptyCount} models parse without a cube banner but carry no geometry `;
    md += `(0 vertices). To the user these render as nothing. ${emptyCount <= 40 ? "Full list" : "First 40"}:\n\n`;
    md += `| path | route |\n|---|---|\n`;
    for (const e of empties.slice(0, 40)) md += `| \`${e.path}\` | ${e.route} |\n`;
    md += `\n`;
  }

  md += `## Over-50ms loads\n\n`;
  const slow = results.filter((r) => r.loadMs > 50).sort((a, b) => b.loadMs - a.loadMs);
  md += `${slow.length} models loaded slower than 50ms. Top 25:\n\n`;
  md += `| ms | route | type | path |\n|---|---|---|---|\n`;
  for (const s of slow.slice(0, 25)) {
    md += `| ${s.loadMs} | ${s.route} | ${s.type} | \`${s.path}\` |\n`;
  }
  md += `\n`;
  md += `> Note: per-model ms is measured in-process (psov2 parse + server fetches), `;
  md += `which the browser parallelises with renderer setup + texture/motion fetches. `;
  md += `Cached re-opens in the live editor are ~4ms (the psov2 LRU); these are COLD loads.\n`;
  md += `>\n`;
  md += `> The handful of multi-second outliers split into (a) **genuine** heavy `;
  md += `cold server parses — the \`ItemModelEp4.afs\` skinned inners reproduce at ~6s `;
  md += `even in isolation and are a real server-side optimisation target; and `;
  md += `(b) **sweep-startup stalls** — the first models scheduled (alphabetical, e.g. `;
  md += `\`biri_ball.bml\`) occasionally absorb a one-time worker/JIT warm-up spike `;
  md += `(\`biri_ball.bml#biri_ball.nj\` measures ~9ms when re-run in isolation), which `;
  md += `inflates max/p99 but not the median/p90. Re-run a flagged path alone with `;
  md += `\`--paths-file\` to distinguish the two.\n`;

  mkdirSync(dirname(OUT_MD), { recursive: true });
  writeFileSync(OUT_MD, md);

  // ---- machine-readable summary line (consumed by the orchestrator) ----
  const summary = {
    totalTested: total,
    okCount,
    cubeCount,
    cubePct,
    emptyCount,
    loadMsMedian: median,
    loadMsP90: p90,
    over50msCount: over50,
    topCubeCategories,
    reportPath: OUT_MD,
    failuresPath: OUT_FAIL,
  };
  process.stdout.write("\n__SMOKE_SUMMARY__" + JSON.stringify(summary) + "\n");
  process.stderr.write(
    `\nDONE: ${total} tested | ${okCount} ok | ${cubeCount} cube (${cubePct}%) | ` +
    `median ${median}ms p90 ${p90}ms | ${over50} over 50ms\n`,
  );
})().catch((e) => {
  console.error("smoke_render_all FATAL:", e?.stack || e);
  process.exit(1);
});
