// PSOBB Modding Suite — manifest loader.
// =====================================================================
// Wraps GET /api/manifest_lite (with /api/manifest fallback) into a
// small browser-side cache so every component that needs the asset
// list (tree, search, model viewer, future sculpt panel) shares one
// fetch and one in-memory snapshot.
//
// Phase 0.5 perf: load() prefers /api/manifest_lite (~110 KB gzipped
// vs ~3.8 MB for /api/manifest). Per-entry detail
// (matched_textures, warnings, format) lazy-loads via
// PSOManifest.fetchEntryDetail(path). Older server builds without the
// lite endpoint transparently fall through to the full manifest, so
// older deployments keep working.
//
// Public surface:
//   await PSOManifest.load()                    - fetches once + caches (lite preferred)
//   await PSOManifest.fetchEntryDetail(path)    - hydrate one entry's full detail
//   PSOManifest.byCategory(cat)                 - returns entries[] filtered
//   PSOManifest.refresh()                       - busts cache + refetches
//   PSOManifest.isLoaded()                      - bool
//   PSOManifest.lastError()                     - last fetch error or null
//   PSOManifest.entries()                       - all entries (no copy)
//   PSOManifest.isLite()                        - true when current cache is the lite shape
//
// All methods are idempotent. `load()` is safe to call from many
// places concurrently — only one fetch is in flight at a time.
// =====================================================================

(function () {
  "use strict";

  if (window.PSOManifest) return; // idempotent

  // The full manifest dict matching MASTER_PLAN/manifest.schema.json.
  // null until first successful fetch.
  let cached = null;
  // True when ``cached`` is the lite shape (path/category/inferred_category/
  // size/parent_archive only). Tree consumers tolerate the missing
  // fields; per-entry-detail panels call ``fetchEntryDetail`` to
  // hydrate the full record on demand.
  let isLiteCached = false;

  // Per-path cache of FULL entries hydrated via /api/asset. Pre-populated
  // when load() reaches the full-manifest fallback path (every entry is
  // already detailed). Keyed by entry.path; values are the full entry
  // dict matching manifest.schema.json's AssetEntry.
  const detailCache = new Map();

  // In-flight detail-fetch promises, keyed by path. Ensures a flurry of
  // hover events on the same entry collapses to one fetch.
  const detailInflight = new Map();

  // Promise of the in-flight fetch, if any. Lets concurrent callers
  // share a single request without re-issuing one mid-flight.
  let inflight = null;

  // Last error, for callers (e.g. the tree) that want to render a
  // "manifest not yet built" placeholder with the underlying reason.
  let lastErr = null;

  // Generic GET-JSON helper that funnels every code path through one
  // error-shape so the tree rendering stays predictable.
  async function _getJson(url) {
    let res;
    try {
      res = await fetch(url, {
        method: "GET",
        headers: { Accept: "application/json" },
        cache: "no-store",
      });
    } catch (e) {
      lastErr = e;
      throw e;
    }
    if (res.status === 404) {
      const err = new Error(`endpoint not implemented (404): ${url}`);
      err.code = "ENDPOINT_MISSING";
      err.status = 404;
      throw err;
    }
    if (!res.ok) {
      const err = new Error(`${url} -> HTTP ${res.status}`);
      err.status = res.status;
      throw err;
    }
    try {
      return await res.json();
    } catch (e) {
      const err = new Error(`${url} response was not valid JSON`);
      err.cause = e;
      throw err;
    }
  }

  function _validateManifestShape(json) {
    return !!(
      json
      && typeof json === "object"
      && Array.isArray(json.entries)
      && json.version === 1
    );
  }

  async function _fetchOnce() {
    lastErr = null;
    // Try the lite endpoint first (Phase 0.5 perf: ~10× smaller).
    try {
      const lite = await _getJson("/api/manifest_lite");
      if (!_validateManifestShape(lite)) {
        const err = new Error("manifest_lite payload missing required fields");
        err.payload = lite;
        lastErr = err;
        throw err;
      }
      cached = lite;
      isLiteCached = true;
      return cached;
    } catch (e) {
      if (!e || e.code !== "ENDPOINT_MISSING") {
        // 4xx/5xx other than 404 → fall through to full manifest so
        // the tree still renders if only the lite endpoint is
        // misbehaving. We keep `lastErr` set in case both fail.
        lastErr = e;
      }
    }

    // Fallback: the full manifest. Older servers that lack
    // /api/manifest_lite will land here.
    try {
      const full = await _getJson("/api/manifest");
      if (!_validateManifestShape(full)) {
        const err = new Error("manifest payload missing required fields");
        err.payload = full;
        lastErr = err;
        throw err;
      }
      cached = full;
      isLiteCached = false;
      // Pre-populate the detail cache from the full manifest — every
      // entry is already detailed, so hovers / clicks don't need to
      // round-trip /api/asset.
      detailCache.clear();
      for (const ent of full.entries) {
        if (ent && ent.path) detailCache.set(ent.path, ent);
      }
      return cached;
    } catch (e) {
      lastErr = e;
      throw e;
    }
  }

  async function load() {
    if (cached) return cached;
    if (inflight) return inflight;
    inflight = _fetchOnce().finally(() => { inflight = null; });
    return inflight;
  }

  async function refresh() {
    cached = null;
    isLiteCached = false;
    detailCache.clear();
    detailInflight.clear();
    inflight = null;
    return load();
  }

  function isLoaded() {
    return cached !== null;
  }

  function isLite() {
    return isLiteCached;
  }

  function entries() {
    return cached ? cached.entries : [];
  }

  function lastError() {
    return lastErr;
  }

  /**
   * Resolve the full AssetEntry for ``path``.  Returns the cached
   * detailed entry when known, or fetches /api/asset/<path> on demand.
   * Concurrent callers for the same path share one in-flight fetch.
   *
   * Throws on hard failure (network / 4xx / 5xx); the caller should
   * fall back to the lite shape it already has when this rejects.
   */
  async function fetchEntryDetail(path) {
    if (!path) return null;
    const known = detailCache.get(path);
    if (known) return known;
    const inflightP = detailInflight.get(path);
    if (inflightP) return inflightP;
    const url = "/api/asset/" + path.split("/").map(encodeURIComponent).join("/");
    const p = (async () => {
      const json = await _getJson(url);
      detailCache.set(path, json);
      return json;
    })().finally(() => { detailInflight.delete(path); });
    detailInflight.set(path, p);
    return p;
  }

  // category enum matches manifest.schema.json:
  //   texture, model, container, quest, map, audio, ui, script,
  //   cinematic, metadata, unknown
  function byCategory(cat) {
    if (!cached) return [];
    if (typeof cat !== "string") return [];
    return cached.entries.filter((e) => e && e.category === cat);
  }

  // Group all entries into a {category: [entry,...]} map, ordered by
  // a sensible priority for the tree sidebar. Categories not in the
  // priority list fall through to alpha. `deprecated: true` entries
  // are omitted (manifest.schema.json says hide-by-default).
  const CATEGORY_ORDER = [
    "texture", "model", "container", "ui",
    "map", "quest", "audio", "script",
    "cinematic", "metadata", "unknown",
  ];

  function grouped() {
    const out = {};
    if (!cached) return out;
    for (const e of cached.entries) {
      if (!e || e.deprecated) continue;
      const cat = e.category || "unknown";
      if (!out[cat]) out[cat] = [];
      out[cat].push(e);
    }
    // Sort each bucket by path for stable display.
    for (const list of Object.values(out)) {
      list.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
    }
    // Re-key in priority order so Object.entries() walks predictably.
    const ordered = {};
    for (const cat of CATEGORY_ORDER) {
      if (out[cat]) ordered[cat] = out[cat];
    }
    // Append any unexpected categories that fell outside the enum.
    for (const cat of Object.keys(out).sort()) {
      if (!ordered[cat]) ordered[cat] = out[cat];
    }
    return ordered;
  }

  window.PSOManifest = Object.freeze({
    load,
    refresh,
    isLoaded,
    isLite,
    entries,
    byCategory,
    grouped,
    lastError,
    fetchEntryDetail,
    CATEGORY_ORDER: Object.freeze(CATEGORY_ORDER.slice()),
  });
})();
