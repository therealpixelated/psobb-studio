// =====================================================================
// PSOBB Texture Editor - Anim Library perspective (v5 polish, 2026-04-25)
// =====================================================================
// A global cross-model browser for every staged motion in
// cache/njm_export/. Exists alongside per-model "Imported Animations"
// strips in the Motions tab; this gives the user a single view of
// every retargeted .njm regardless of which BML it was authored against.
//
// Layout:
//   - Toolbar: search box, target-model filter, "select all", "select
//     none", "Refresh from disk", subscriber count
//   - Grid:    one card per .njm with name, target, source glb, frame
//     count, fps, duration, file size, md5 prefix, age, sidecar status
//   - Inspector: bulk-action panel — Delete N, Bulk rename, Export ZIP,
//     Re-retarget (lifts the Import 3D flow with a pre-selected source)
//
// Wiring:
//   GET  /api/anim_library/list      everything in cache/njm_export/
//   POST /api/anim_library/delete    {names: [...]}
//   POST /api/anim_library/rename    {renames: [{old_name, new_name}, ...]}
//   POST /api/anim_library/zip       {names: [...]} -> .zip download
//   POST /api/events/rescan          re-scan watched dirs
//   bus.cache.changed                live-reload triggers a fetch of /list
//
// Re-retarget flow:
//   The Import 3D panel (import_panel.js) already does the source -> target
//   retarget pipeline. We don't duplicate it; we synthesize a click on
//   the import-3d header button and pass through window.psoImportPanel
//   if/when it surfaces a programmatic API. For now, "Re-retarget" prefills
//   the source_glb if known and shows a hint asking the user to drop the
//   target glb / pick a target BML.
// =====================================================================

(function () {
  "use strict";

  if (!window.PSOPerspectives) {
    console.warn("[anim_library] perspectives.js not loaded yet");
    return;
  }
  if (window.__psoAnimLibraryLoaded) return;
  window.__psoAnimLibraryLoaded = true;

  // ------------------------------------------------------------------
  // State
  // ------------------------------------------------------------------
  const state = {
    items: [],            // raw response.items
    totals: null,         // {size, with_sidecar}
    filter: "",           // search text (lowercased)
    targetFilter: "",     // active target_model_name filter; "" = all
    selected: new Set(),  // Set<name>
    fetching: false,
    stageEl: null,
    inspEl: null,
    refreshTimer: null,   // debounce live-reload bursts
  };

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------
  // e2e / unit-test fixtures live in the same cache dir as authored
  // animations (test_walk_endpoint, test_e2e_*, test_anim_editor_synth).
  // Hide them from the displayed list without touching the files.
  function isTestFixture(name) {
    if (!name) return false;
    const n = String(name).toLowerCase();
    return n.startsWith("test_") || /(^|_)e2e_/.test(n);
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;",
                '"': "&quot;", "'": "&#39;" })[c];
    });
  }

  function fmtSize(n) {
    if (typeof n !== "number" || !isFinite(n) || n < 0) return "";
    if (n < 1024) return n + " B";
    const u = ["KB", "MB", "GB"];
    let v = n / 1024, i = 0;
    while (v >= 1024 && i < u.length - 1) { v /= 1024; i += 1; }
    return v.toFixed(v < 10 ? 1 : 0) + " " + u[i];
  }

  function fmtAge(ms) {
    if (!ms || typeof ms !== "number") return "";
    const delta = Math.max(0, Date.now() - ms);
    const s = Math.floor(delta / 1000);
    if (s < 60) return s + "s ago";
    const m = Math.floor(s / 60);
    if (m < 60) return m + "m ago";
    const h = Math.floor(m / 60);
    if (h < 24) return h + "h ago";
    const d = Math.floor(h / 24);
    if (d < 30) return d + "d ago";
    return new Date(ms).toLocaleDateString();
  }

  function fmtDuration(frames, fps) {
    if (!frames || !fps) return "";
    return (frames / fps).toFixed(2) + "s @ " + fps + "fps";
  }

  function toast(msg, kind) {
    if (window.psoEditor && typeof window.psoEditor.toast === "function") {
      window.psoEditor.toast(msg, kind);
      return;
    }
    if (typeof window.showToast === "function") {
      window.showToast(msg, kind);
      return;
    }
    console.log("[anim_library]", kind || "info", msg);
  }

  // ------------------------------------------------------------------
  // API
  // ------------------------------------------------------------------
  async function apiList() {
    const r = await fetch("/api/anim_library/list");
    if (!r.ok) throw new Error("list (" + r.status + ")");
    return r.json();
  }

  async function apiDelete(names) {
    const r = await fetch("/api/anim_library/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ names }),
    });
    if (!r.ok) throw new Error("delete (" + r.status + ")");
    return r.json();
  }

  async function apiRename(renames) {
    const r = await fetch("/api/anim_library/rename", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ renames }),
    });
    if (!r.ok) throw new Error("rename (" + r.status + ")");
    return r.json();
  }

  async function apiZip(names) {
    const r = await fetch("/api/anim_library/zip", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ names }),
    });
    if (!r.ok) throw new Error("zip (" + r.status + ")");
    const blob = await r.blob();
    const cd = r.headers.get("Content-Disposition") || "";
    let filename = "anim_library.zip";
    const m = /filename="([^"]+)"/.exec(cd);
    if (m) filename = m[1];
    return { blob, filename };
  }

  async function apiRescan() {
    const r = await fetch("/api/events/rescan", { method: "POST" });
    if (!r.ok) throw new Error("rescan (" + r.status + ")");
    return r.json();
  }

  // ------------------------------------------------------------------
  // Refresh
  // ------------------------------------------------------------------
  async function refresh() {
    if (state.fetching) return;
    state.fetching = true;
    try {
      const data = await apiList();
      const all = Array.isArray(data.items) ? data.items : [];
      // Hide e2e / unit-test fixtures (test_walk_endpoint, test_e2e_*,
      // test_anim_editor_synth, ...) — they're real cache files but pure
      // test junk, not authored content. Frontend filter only; the cache
      // files are left untouched on disk.
      state.items = all.filter((it) => !isTestFixture(it.name));
      state.totals = data.totals || null;
      // Drop selections for items that no longer exist on disk.
      const live = new Set(state.items.map((it) => it.name));
      for (const n of Array.from(state.selected)) {
        if (!live.has(n)) state.selected.delete(n);
      }
    } catch (e) {
      console.warn("[anim_library] list failed:", e);
      toast("anim library: " + e.message, "error");
    } finally {
      state.fetching = false;
    }
    if (state.stageEl) renderGrid();
    if (state.inspEl) renderInspector();
  }

  function debounceRefresh() {
    if (state.refreshTimer) clearTimeout(state.refreshTimer);
    state.refreshTimer = setTimeout(() => {
      state.refreshTimer = null;
      refresh();
    }, 300);
  }

  // ------------------------------------------------------------------
  // Filtering
  // ------------------------------------------------------------------
  function visibleItems() {
    const f = state.filter.trim().toLowerCase();
    const t = state.targetFilter.trim().toLowerCase();
    return state.items.filter((it) => {
      if (t && (it.target_model_name || "").toLowerCase() !== t) return false;
      if (!f) return true;
      const hay = (it.name + " " + it.display_name + " " +
                   it.source_glb + " " + it.target_model_name + " " +
                   it.source_animation).toLowerCase();
      return hay.indexOf(f) !== -1;
    });
  }

  function uniqueTargets() {
    const seen = new Map();   // basename -> count
    for (const it of state.items) {
      const t = it.target_model_name || "(none)";
      seen.set(t, (seen.get(t) || 0) + 1);
    }
    return Array.from(seen.entries()).sort();
  }

  // ------------------------------------------------------------------
  // Rendering
  // ------------------------------------------------------------------
  function renderToolbar(host) {
    const targets = uniqueTargets();
    host.innerHTML = `
      <div class="anim-lib-toolbar">
        <input type="text" class="anim-lib-search" placeholder="filter by name / source / target..." spellcheck="false">
        <select class="anim-lib-target-sel">
          <option value="">all targets (${state.items.length})</option>
          ${targets.map(([n, c]) =>
            `<option value="${escapeHtml(n)}" ${n === state.targetFilter ? "selected" : ""}>${escapeHtml(n)} (${c})</option>`
          ).join("")}
        </select>
        <button class="anim-lib-btn anim-lib-select-all ghost" title="select every visible animation">select all</button>
        <button class="anim-lib-btn anim-lib-select-none ghost" title="clear selection">select none</button>
        <span class="grow"></span>
        <span class="anim-lib-count dim" data-region="count"></span>
        <button class="anim-lib-btn anim-lib-refresh" title="re-scan cache/njm_export/ (also fires whenever a file changes via live-reload)">refresh from disk</button>
      </div>
    `;
    const search = host.querySelector(".anim-lib-search");
    if (search) {
      search.value = state.filter;
      search.addEventListener("input", () => {
        state.filter = search.value;
        renderGrid();
        renderInspector();
      });
    }
    const tsel = host.querySelector(".anim-lib-target-sel");
    if (tsel) {
      tsel.addEventListener("change", () => {
        state.targetFilter = tsel.value || "";
        renderGrid();
        renderInspector();
      });
    }
    host.querySelector(".anim-lib-select-all").addEventListener("click", () => {
      const vis = visibleItems();
      for (const it of vis) state.selected.add(it.name);
      renderGrid();
      renderInspector();
    });
    host.querySelector(".anim-lib-select-none").addEventListener("click", () => {
      state.selected.clear();
      renderGrid();
      renderInspector();
    });
    host.querySelector(".anim-lib-refresh").addEventListener("click", async () => {
      try {
        await apiRescan();
        await refresh();
        toast("Rescanned cache/njm_export/", "info");
      } catch (e) {
        toast("rescan failed: " + e.message, "error");
      }
    });
  }

  function renderGrid() {
    if (!state.stageEl) return;
    const list = state.stageEl.querySelector('[data-region="grid"]');
    const counter = state.stageEl.querySelector('[data-region="count"]');
    const vis = visibleItems();
    if (counter) {
      counter.textContent = `${vis.length} / ${state.items.length} animations`;
    }
    if (!list) return;
    if (vis.length === 0) {
      const reason = state.items.length === 0
        ? "No animations staged yet. Drop a .glb on the Import 3D panel to retarget against a PSOBB skeleton, then come back here."
        : "No animations match the current filter.";
      list.innerHTML = `<div class="anim-lib-empty">${escapeHtml(reason)}</div>`;
      return;
    }
    list.innerHTML = vis.map((it) => {
      const sel = state.selected.has(it.name);
      const dur = fmtDuration(it.frame_count, it.fps);
      const src = it.source_glb || (it.has_sidecar ? "(unknown source)" : "(legacy / no sidecar)");
      const tgt = it.target_model_name || "(none)";
      const md5 = (it.md5 || "").slice(0, 8);
      const cov = it.bone_count
        ? `${it.retargeted_bones}/${it.bone_count}` + (it.dropped_bones ? ` · ${it.dropped_bones} dropped` : "")
        : "(no sidecar)";
      const age = fmtAge(it.retargeted_at_ms || it.mtime_ms);
      return `
        <div class="anim-lib-card ${sel ? "selected" : ""}" data-name="${escapeHtml(it.name)}">
          <label class="anim-lib-card-check">
            <input type="checkbox" data-act="toggle" ${sel ? "checked" : ""} />
          </label>
          <div class="anim-lib-card-body">
            <div class="anim-lib-card-name" title="${escapeHtml(it.name)}">${escapeHtml(it.display_name || it.name)}</div>
            <div class="anim-lib-card-row dim">
              <span title="target BML basename">target: <b>${escapeHtml(tgt)}</b></span>
              <span title="source glb">src: ${escapeHtml(src)}</span>
            </div>
            <div class="anim-lib-card-row dim">
              <span title="frame count and fps">${it.frame_count}f${dur ? " · " + dur : ""}</span>
              <span title="bones mapped/total">bones: ${cov}</span>
              <span title="file size">${fmtSize(it.size)}</span>
            </div>
            <div class="anim-lib-card-row dim">
              <span title="md5 prefix">md5: ${escapeHtml(md5 || "?")}</span>
              <span title="last modified">${age}</span>
              ${it.has_sidecar ? '' : '<span class="anim-lib-tag-warn" title="no .preview.json sidecar — retargeted via /api/anim_keyframe/save instead of /api/import/animation">legacy</span>'}
            </div>
          </div>
          <div class="anim-lib-card-actions">
            <button class="anim-lib-btn ghost" data-act="preview" title="preview on whatever model this was retargeted to">preview</button>
            <button class="anim-lib-btn ghost" data-act="rename" title="rename this .njm">rename</button>
            <button class="anim-lib-btn ghost danger" data-act="delete" title="delete this .njm + its .preview.json">delete</button>
          </div>
        </div>
      `;
    }).join("");
    // Wire interactions.
    list.querySelectorAll(".anim-lib-card").forEach((card) => {
      const name = card.dataset.name;
      const cb = card.querySelector('[data-act="toggle"]');
      if (cb) {
        cb.addEventListener("change", () => {
          if (cb.checked) state.selected.add(name);
          else state.selected.delete(name);
          card.classList.toggle("selected", cb.checked);
          renderInspector();
        });
      }
      const prevBtn = card.querySelector('[data-act="preview"]');
      if (prevBtn) prevBtn.addEventListener("click", () => previewItem(name));
      const renameBtn = card.querySelector('[data-act="rename"]');
      if (renameBtn) renameBtn.addEventListener("click", () => renameOne(name));
      const delBtn = card.querySelector('[data-act="delete"]');
      if (delBtn) delBtn.addEventListener("click", () => deleteOne(name));
      // Make whole card body clickable for select-toggle (UX nicety).
      const body = card.querySelector(".anim-lib-card-body");
      if (body) {
        body.addEventListener("click", (ev) => {
          if (ev.target.closest("button")) return;
          if (ev.target.closest("input")) return;
          if (cb) {
            cb.checked = !cb.checked;
            cb.dispatchEvent(new Event("change"));
          }
        });
      }
    });
  }

  function renderInspector() {
    if (!state.inspEl) return;
    const sel = state.selected.size;
    const totalSize = state.items.reduce((s, it) => s + (it.size || 0), 0);
    const selSize = state.items
      .filter((it) => state.selected.has(it.name))
      .reduce((s, it) => s + (it.size || 0), 0);
    state.inspEl.innerHTML = `
      <div class="vp-insp-title">Anim Library</div>
      <div class="vp-insp-help dim">
        Browse every imported animation across all models. Use the
        checkboxes on each card to bulk-act on N at once.
      </div>
      <div class="vp-insp-section">
        <div class="anim-lib-stats dim">
          <div>${state.items.length} animations · ${fmtSize(totalSize)}</div>
          <div>${state.totals ? state.totals.with_sidecar : 0} with sidecar</div>
          <div><b>${sel}</b> selected · ${fmtSize(selSize)}</div>
        </div>
      </div>
      <div class="vp-insp-section anim-lib-bulk">
        <button class="anim-lib-btn anim-lib-bulk-zip" ${sel === 0 ? "disabled" : ""} title="download all selected as a ZIP for batch deploy">Export ZIP (${sel})</button>
        <button class="anim-lib-btn anim-lib-bulk-rename" ${sel === 0 ? "disabled" : ""} title="prefix-rename every selected animation">Bulk rename (${sel})</button>
        <button class="anim-lib-btn anim-lib-bulk-retarget" ${sel === 0 ? "disabled" : ""} title="re-run retarget against a different target skeleton (opens Import 3D)">Re-retarget (${sel})</button>
        <button class="anim-lib-btn danger anim-lib-bulk-delete" ${sel === 0 ? "disabled" : ""} title="delete all selected animations + their sidecars">Delete ${sel}</button>
      </div>
    `;
    const zipBtn = state.inspEl.querySelector(".anim-lib-bulk-zip");
    if (zipBtn) zipBtn.addEventListener("click", bulkZip);
    const renameBtn = state.inspEl.querySelector(".anim-lib-bulk-rename");
    if (renameBtn) renameBtn.addEventListener("click", bulkRename);
    const retargetBtn = state.inspEl.querySelector(".anim-lib-bulk-retarget");
    if (retargetBtn) retargetBtn.addEventListener("click", bulkReRetarget);
    const delBtn = state.inspEl.querySelector(".anim-lib-bulk-delete");
    if (delBtn) delBtn.addEventListener("click", bulkDelete);
  }

  // ------------------------------------------------------------------
  // Operations
  // ------------------------------------------------------------------
  function previewItem(name) {
    // Find the entry; route to /api/anim_preview/data through the
    // existing motion-picker flow if a target model is known. This
    // re-uses model_viewer.js infrastructure rather than duplicating it.
    const item = state.items.find((it) => it.name === name);
    if (!item) return;
    const target = item.target_model_path;
    if (!target) {
      toast("No target_model_path on sidecar — cannot auto-preview.", "warn");
      return;
    }
    // Open the model first — the asset_router already knows how.
    if (window.bus && window.bus.emit) {
      window.bus.emit("asset.opened", {
        path: target,
        entry: { category: "model", format: target.toLowerCase().endsWith(".bml") ? "BML" : "NJ" },
      });
    } else if (typeof window.psoOpenModelByPath === "function") {
      try { window.psoOpenModelByPath(target, {}); } catch (_e) {}
    }
    // Try to play this motion after a short delay so the model resolves.
    setTimeout(() => {
      if (typeof window.psoLoadMotion === "function") {
        try { window.psoLoadMotion(item.display_name || name); }
        catch (_e) {}
      }
    }, 500);
  }

  async function deleteOne(name) {
    if (!window.confirm(`Delete ${name}?\nThis removes the .njm + sidecar from cache/njm_export/.`)) return;
    try {
      const r = await apiDelete([name]);
      const removed = (r.results || []).filter((x) => x.removed).length;
      toast(`Deleted ${removed} animation`, "info");
      state.selected.delete(name);
      await refresh();
    } catch (e) {
      toast("delete failed: " + e.message, "error");
    }
  }

  async function renameOne(name) {
    const item = state.items.find((it) => it.name === name);
    if (!item) return;
    const proposed = window.prompt("New filename (must end with .njm):", name);
    if (!proposed || proposed === name) return;
    try {
      const r = await apiRename([{ old_name: name, new_name: proposed }]);
      const ok = (r.results || []).filter((x) => x.renamed).length;
      if (ok === 0) {
        const err = (r.results || [])[0] || {};
        toast("rename failed: " + (err.error || "unknown"), "error");
      } else {
        toast(`Renamed to ${proposed}`, "info");
        state.selected.delete(name);
        if (state.selected.has(name)) state.selected.delete(name);
        await refresh();
      }
    } catch (e) {
      toast("rename failed: " + e.message, "error");
    }
  }

  async function bulkDelete() {
    const names = Array.from(state.selected);
    if (names.length === 0) return;
    if (!window.confirm(`Delete ${names.length} animations?\nThis removes the .njm + .preview.json for each from cache/njm_export/.`)) return;
    try {
      const r = await apiDelete(names);
      const removed = (r.results || []).filter((x) => x.removed).length;
      toast(`Deleted ${removed}/${names.length}`, "info");
      state.selected.clear();
      await refresh();
    } catch (e) {
      toast("bulk delete failed: " + e.message, "error");
    }
  }

  async function bulkRename() {
    const names = Array.from(state.selected);
    if (names.length === 0) return;
    const prefix = window.prompt(
      `Bulk rename: prefix to add to ${names.length} animations:`,
      "",
    );
    if (prefix === null) return;
    if (!prefix.trim()) {
      toast("empty prefix — nothing to do", "warn");
      return;
    }
    if (!/^[A-Za-z0-9_\-]+$/.test(prefix)) {
      toast("prefix may only contain [A-Za-z0-9_-]", "warn");
      return;
    }
    const renames = names.map((n) => ({
      old_name: n,
      new_name: prefix + "_" + n,
    }));
    try {
      const r = await apiRename(renames);
      const ok = (r.results || []).filter((x) => x.renamed).length;
      const failed = (r.results || []).filter((x) => !x.renamed);
      toast(`Renamed ${ok}/${names.length}` + (failed.length ? ` · ${failed.length} failed` : ""), failed.length ? "warn" : "info");
      state.selected.clear();
      await refresh();
    } catch (e) {
      toast("bulk rename failed: " + e.message, "error");
    }
  }

  async function bulkZip() {
    const names = Array.from(state.selected);
    if (names.length === 0) return;
    try {
      const { blob, filename } = await apiZip(names);
      // Trigger a browser download.
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      toast(`Exported ${names.length} animations as ${filename}`, "info");
    } catch (e) {
      toast("zip export failed: " + e.message, "error");
    }
  }

  function bulkReRetarget() {
    const names = Array.from(state.selected);
    if (names.length === 0) return;
    // Re-retargeting requires a fresh target skeleton choice. Instead of
    // re-implementing the import_panel.js retarget pipeline here, we tell
    // the user how to do it — this single click would otherwise prompt
    // for a target BML, then loop apiAnimationRetarget for every item.
    // The Import 3D panel exposes that whole flow for one item; until
    // we expose a programmatic API on it, surface a help toast.
    toast(
      `Re-retarget for ${names.length} animations: open Import 3D, drop the same source GLB, pick a different target BML, and the import flow will re-stage each animation.`,
      "info",
    );
    if (typeof window.psoImportPanelOpen === "function") {
      try {
        window.psoImportPanelOpen({ source: "anim_library", names });
      } catch (_e) {}
    }
  }

  // ------------------------------------------------------------------
  // Perspective registration
  // ------------------------------------------------------------------
  window.PSOPerspectives.register("anim-library", {
    label: "Anim Library",
    match: function (entry) {
      // Only match when synthesized by the header button (path = __anim_library__).
      // Returning 0 hides it from auto-route on regular asset opens.
      if (entry && entry.category === "anim-library") return 100;
      return 0;
    },
    mount: async function (stage, insp) {
      state.stageEl = stage;
      state.inspEl = insp;
      stage.innerHTML = `
        <div class="anim-lib-perspective">
          <div data-region="toolbar"></div>
          <div data-region="grid" class="anim-lib-grid"></div>
        </div>
      `;
      renderToolbar(stage.querySelector('[data-region="toolbar"]'));
      renderInspector();
      await refresh();
    },
    unmount: function () {
      state.stageEl = null;
      state.inspEl = null;
    },
  });

  // ------------------------------------------------------------------
  // Header button + bus wiring
  // ------------------------------------------------------------------
  function openPerspective() {
    const ctx = {
      path: "__anim_library__",
      entry: { category: "anim-library", format: "AnimLibrary" },
      fileName: "Animation Library",
    };
    if (window.PSOPerspectives && window.PSOPerspectives.switchTo) {
      window.PSOPerspectives.switchTo("anim-library", ctx);
    }
  }

  function ensureHeaderButton() {
    if (document.getElementById("btnAnimLibrary")) return;
    const status = document.getElementById("status");
    const header = status ? status.parentNode : null;
    if (!header) return;
    const btn = document.createElement("button");
    btn.id = "btnAnimLibrary";
    btn.type = "button";
    btn.className = "ghost";
    btn.title = "browse every staged animation in cache/njm_export/ across all target models";
    btn.textContent = "Anim Library";
    header.insertBefore(btn, status);
    btn.addEventListener("click", openPerspective);
  }

  function wireBus() {
    if (!window.bus) {
      setTimeout(wireBus, 50);
      return;
    }
    // Live-reload integration: refresh whenever cache/njm_export/ changes.
    window.bus.on("cache.changed", (payload) => {
      if (!payload || !payload.path) return;
      // Only refresh if the change is in njm_export. Other dirs are
      // handled by their own panels.
      if (payload.path.indexOf("cache/njm_export/") !== 0) return;
      // If the panel is currently mounted, refresh now (debounced); else
      // skip — next mount will fetch fresh.
      if (state.stageEl) debounceRefresh();
    });
  }

  function init() {
    ensureHeaderButton();
    wireBus();
    // Surface a programmatic API for tests + integrations.
    window.psoAnimLibrary = {
      open: openPerspective,
      refresh,
      state: () => ({
        items: state.items.slice(),
        selected: Array.from(state.selected),
        filter: state.filter,
        targetFilter: state.targetFilter,
      }),
    };
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
