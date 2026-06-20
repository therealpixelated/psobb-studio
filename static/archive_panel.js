// =====================================================================
// PSOBB Modding Suite — Archive entry editor perspective (2026-06-20)
//
// First-class DUPLICATE / CREATE / DELETE / RENAME of an inner entry
// inside a container archive (AFS or BML). Renders the entry list and
// per-row actions, plus a "+ New entry" modal (upload a blob OR an
// empty/copy-first template, with a name field).
//
// Self-registering: like static/audio_panel.js, this registers the
// "archive-entries" perspective with window.PSOPerspectives AFTER
// perspectives.js has loaded — perspectives.js is NOT edited.
//
// Backend contract (server.py /api/archive/*):
//   GET    /api/archive/{name}/entries       -> { ok, kind, supported, entries[] }
//   POST   /api/archive/duplicate_entry       -> { ok, new_index?|new_entry_name?, new_path }
//   POST   /api/archive/create_entry (multipart)
//   DELETE /api/archive/entry
//   POST   /api/archive/rename_entry
//
// Writes land on the path the archive was OPENED from (DATA_DIR or LIVE,
// resolved like the readers). After any success we GET /api/manifest?
// force=1, refresh the list, and for duplicate/create auto-open the new
// entry so it is immediately editable.
//
// GSL / unknown containers -> disabled controls + a "not supported"
// note (the backend reports supported=false).
// =====================================================================
(function () {
  "use strict";

  if (window.__psoArchivePanelLoaded) return;
  window.__psoArchivePanelLoaded = true;

  if (!window.PSOPerspectives || !window.PSOPerspectives.register) {
    console.warn("[archive_panel] PSOPerspectives not available; archive editor disabled");
    return;
  }
  var esc = window.PSOPerspectives.escapeHtml || function (s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  };

  function fmtBytes(n) {
    if (typeof n !== "number" || n < 0) return "";
    if (n < 1024) return n + " B";
    var u = ["KB", "MB", "GB"], v = n / 1024, i = 0;
    while (v >= 1024 && i < u.length - 1) { v /= 1024; i += 1; }
    return v.toFixed(v < 10 ? 1 : 0) + " " + u[i];
  }

  // archive bare-name from a ctx path: strip any directory and any "#inner".
  function archiveNameFromPath(path) {
    if (!path) return "";
    var p = String(path);
    var hash = p.indexOf("#");
    if (hash >= 0) p = p.slice(0, hash);
    return p.split("/").pop();
  }

  async function fetchJson(url, opts) {
    var r = await fetch(url, opts || { cache: "no-store" });
    var j = null;
    try { j = await r.json(); } catch (_e) { j = null; }
    if (!r.ok) {
      var detail = (j && j.detail) || ("HTTP " + r.status);
      var err = new Error(detail);
      err.status = r.status;
      throw err;
    }
    return j;
  }

  function setStatus(state, msg, isErr) {
    var el = state.stage.querySelector("#arcStatus");
    if (!el) return;
    el.textContent = msg || "";
    el.className = "dim arc-status" + (isErr ? " err" : "");
  }

  // After any successful edit: refresh the manifest (force) + reload list.
  async function refreshManifestAndList(state) {
    try { await fetch("/api/manifest?force=1", { cache: "no-store" }); } catch (_e) {}
    await loadList(state);
  }

  // Open the (new) entry in the appropriate viewer so it's editable.
  function openEntry(newPath) {
    if (!newPath) return;
    // Route through the shared asset dispatcher; tree.js / asset_router.js
    // know how to open "<archive>#NNNN" addressing in the tile/hex editor.
    try {
      if (window.bus && typeof window.bus.emit === "function") {
        window.bus.emit("asset.opened", { path: newPath, entry: {} });
        return;
      }
    } catch (_e) {}
    if (typeof window.psoAssetOpen === "function") {
      window.psoAssetOpen({ path: newPath, entry: {} });
    }
  }

  // ---- entry list -----------------------------------------------------
  async function loadList(state) {
    var data;
    try {
      data = await fetchJson("/api/archive/" + encodeURIComponent(state.archive) + "/entries");
    } catch (e) {
      renderError(state, e);
      return;
    }
    state.kind = data.kind;
    state.supported = !!data.supported;
    state.entries = data.entries || [];
    renderList(state, data);
  }

  function renderError(state, e) {
    var body = state.stage.querySelector("#arcListBody");
    if (body) {
      body.innerHTML = '<div class="dim arc-empty">could not load entries: ' +
        esc(String(e && e.message || e)) + '</div>';
    }
  }

  function rowActions(state, ent) {
    var afs = state.kind === "afs";
    // AFS rename is only valid when the archive has a real name table; the
    // backend returns 409 otherwise. We always render the button and let the
    // backend reject it with a clear status (so the user learns why).
    var idAttr = afs
      ? ('data-index="' + ent.index + '"')
      : ('data-name="' + esc(ent.name) + '"');
    return '<div class="arc-row-actions">' +
      '<button type="button" class="ghost arc-dup" ' + idAttr + '>Duplicate</button>' +
      '<button type="button" class="ghost arc-ren" ' + idAttr + '>Rename</button>' +
      '<button type="button" class="ghost danger arc-del" ' + idAttr + '>Delete</button>' +
      '</div>';
  }

  function renderList(state, data) {
    var body = state.stage.querySelector("#arcListBody");
    if (!body) return;

    if (!state.supported) {
      body.innerHTML = '<div class="dim arc-empty">' +
        'Entry editing is <b>not supported</b> for this container' +
        (data && data.kind ? '' : ' (only AFS and BML archives can be edited).') +
        '</div>';
      var addBtn = state.stage.querySelector("#arcBtnNew");
      if (addBtn) { addBtn.disabled = true; addBtn.title = "not supported for this container"; }
      return;
    }

    var rows = state.entries.map(function (ent) {
      var fmt = ent.inner_format ? '<span class="arc-fmt">' + esc(ent.inner_format) + '</span>' : '';
      var comp = ent.compressed ? '<span class="arc-tag">PRS</span>' : '';
      var tex = ent.has_texture ? '<span class="arc-tag">+tex</span>' : '';
      return '<div class="arc-row">' +
        '<div class="arc-row-main">' +
        '<span class="arc-idx">' + (state.kind === "afs" ? ("#" + String(ent.index).padStart(4, "0")) : "") + '</span>' +
        '<span class="arc-name">' + esc(ent.name) + '</span>' +
        fmt + comp + tex +
        '<span class="arc-size dim">' + esc(fmtBytes(ent.size)) + '</span>' +
        '</div>' +
        rowActions(state, ent) +
        '</div>';
    }).join("");

    body.innerHTML = rows || '<div class="dim arc-empty">no entries</div>';
    wireRowActions(state);
  }

  function wireRowActions(state) {
    var body = state.stage.querySelector("#arcListBody");
    if (!body) return;
    body.querySelectorAll(".arc-dup").forEach(function (b) {
      b.addEventListener("click", function () { doDuplicate(state, b); });
    });
    body.querySelectorAll(".arc-ren").forEach(function (b) {
      b.addEventListener("click", function () { doRename(state, b); });
    });
    body.querySelectorAll(".arc-del").forEach(function (b) {
      b.addEventListener("click", function () { doDelete(state, b); });
    });
  }

  function rowRef(state, btn) {
    if (state.kind === "afs") return { index: parseInt(btn.getAttribute("data-index"), 10) };
    return { entry_name: btn.getAttribute("data-name") };
  }

  // ---- actions --------------------------------------------------------
  async function doDuplicate(state, btn) {
    var ref = rowRef(state, btn);
    var body = Object.assign({ archive: state.archive }, ref);
    setStatus(state, "duplicating…");
    try {
      var res = await fetchJson("/api/archive/duplicate_entry", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      setStatus(state, "duplicated → " + (res.new_path || ""));
      await refreshManifestAndList(state);
      openEntry(res.new_path);
    } catch (e) {
      setStatus(state, "duplicate failed: " + (e && e.message || e), true);
    }
  }

  async function doRename(state, btn) {
    var ref = rowRef(state, btn);
    var current = state.kind === "afs"
      ? (state.entries.find(function (e) { return e.index === ref.index; }) || {}).name
      : ref.entry_name;
    var nn = window.prompt("New name for " + current + ":", current || "");
    if (nn == null || nn === "" || nn === current) return;
    var body = Object.assign({ archive: state.archive, new_name: nn }, ref);
    setStatus(state, "renaming…");
    try {
      await fetchJson("/api/archive/rename_entry", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      setStatus(state, "renamed → " + nn);
      await refreshManifestAndList(state);
    } catch (e) {
      setStatus(state, "rename failed: " + (e && e.message || e), true);
    }
  }

  async function doDelete(state, btn) {
    var ref = rowRef(state, btn);
    var label = state.kind === "afs" ? ("#" + ref.index) : ref.entry_name;
    if (!window.confirm("Delete entry " + label + " from " + state.archive + "?")) return;
    var body = Object.assign({ archive: state.archive }, ref);
    setStatus(state, "deleting…");
    try {
      await fetchJson("/api/archive/entry", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      setStatus(state, "deleted " + label);
      await refreshManifestAndList(state);
    } catch (e) {
      setStatus(state, "delete failed: " + (e && e.message || e), true);
    }
  }

  // ---- "+ New entry" modal -------------------------------------------
  function openNewModal(state) {
    if (!state.supported) return;
    var afs = state.kind === "afs";
    var existing = state.stage.querySelector("#arcModal");
    if (existing) existing.remove();

    var modal = document.createElement("div");
    modal.id = "arcModal";
    modal.className = "arc-modal";
    modal.innerHTML =
      '<div class="arc-modal-card">' +
      '<div class="arc-modal-title">New entry — ' + esc(state.archive) + '</div>' +
      '<label class="arc-field">name' +
      '<input type="text" id="arcNewName" placeholder="' +
      (afs ? "optional (AFS uses index addressing)" : "required, ≤32 ASCII bytes") + '" />' +
      '</label>' +
      '<label class="arc-field">source' +
      '<select id="arcNewSource">' +
      '<option value="upload">upload a blob</option>' +
      (afs ? '<option value="empty">empty entry</option>' +
             '<option value="copy_first">copy of first entry</option>' : '') +
      '</select>' +
      '</label>' +
      '<label class="arc-field" id="arcFileWrap">file' +
      '<input type="file" id="arcNewFile" />' +
      '</label>' +
      (afs ? '' : '<label class="arc-opt"><input type="checkbox" id="arcIsCompressed" /> ' +
                   'blob is already PRS-compressed</label>') +
      '<div class="arc-modal-btns">' +
      '<button type="button" id="arcNewCancel" class="ghost">cancel</button>' +
      '<button type="button" id="arcNewCreate" class="primary">create</button>' +
      '</div>' +
      '<div id="arcNewStatus" class="dim arc-status"></div>' +
      '</div>';
    state.stage.appendChild(modal);

    var sourceSel = modal.querySelector("#arcNewSource");
    var fileWrap = modal.querySelector("#arcFileWrap");
    function syncFileVisibility() {
      fileWrap.style.display = (sourceSel.value === "upload") ? "" : "none";
    }
    sourceSel.addEventListener("change", syncFileVisibility);
    syncFileVisibility();

    modal.querySelector("#arcNewCancel").addEventListener("click", function () { modal.remove(); });
    modal.addEventListener("click", function (e) { if (e.target === modal) modal.remove(); });
    modal.querySelector("#arcNewCreate").addEventListener("click", function () {
      doCreate(state, modal);
    });
  }

  async function doCreate(state, modal) {
    var afs = state.kind === "afs";
    var name = (modal.querySelector("#arcNewName").value || "").trim();
    var source = modal.querySelector("#arcNewSource").value;
    var fileInput = modal.querySelector("#arcNewFile");
    var statusEl = modal.querySelector("#arcNewStatus");
    function st(m, e) { statusEl.textContent = m; statusEl.className = "dim arc-status" + (e ? " err" : ""); }

    if (!afs && !name) { st("BML entries need a name", true); return; }

    var fd = new FormData();
    fd.append("archive", state.archive);
    if (name) fd.append("new_name", name);
    if (source === "upload") {
      if (!fileInput.files || !fileInput.files.length) { st("pick a file to upload", true); return; }
      fd.append("file", fileInput.files[0]);
      var isC = modal.querySelector("#arcIsCompressed");
      if (isC && isC.checked) fd.append("is_compressed", "true");
    } else {
      fd.append("template", source); // "empty" | "copy_first"
    }
    st("creating…");
    try {
      var r = await fetch("/api/archive/create_entry", { method: "POST", body: fd });
      var j = null; try { j = await r.json(); } catch (_e) {}
      if (!r.ok) { st("create failed: " + ((j && j.detail) || ("HTTP " + r.status)), true); return; }
      modal.remove();
      setStatus(state, "created → " + (j.new_path || ""));
      await refreshManifestAndList(state);
      openEntry(j.new_path);
    } catch (e) {
      st("create error: " + (e && e.message || e), true);
    }
  }

  // ---- stage scaffolding ---------------------------------------------
  function renderStage(state) {
    state.stage.innerHTML =
      '<div class="arc-perspective">' +
      '<div class="arc-toolbar">' +
      '<strong>Archive entries</strong>' +
      '<span class="arc-archive dim">' + esc(state.archive) + '</span>' +
      '<button type="button" id="arcBtnNew" class="ghost">+ New entry</button>' +
      '<span class="grow"></span>' +
      '<button type="button" id="arcBtnRefresh" class="ghost">refresh</button>' +
      '<span id="arcStatus" class="dim arc-status"></span>' +
      '</div>' +
      '<div class="arc-note dim">Edits write back to the file you opened (the archive on disk) ' +
      'with an automatic backup. AFS uses positional index addressing; BML uses entry names.</div>' +
      '<div id="arcListBody" class="arc-list"><div class="dim arc-empty">loading…</div></div>' +
      '</div>';

    var newBtn = state.stage.querySelector("#arcBtnNew");
    if (newBtn) newBtn.addEventListener("click", function () { openNewModal(state); });
    var refBtn = state.stage.querySelector("#arcBtnRefresh");
    if (refBtn) refBtn.addEventListener("click", function () { loadList(state); });
  }

  function renderInspector(state) {
    if (!state.insp) return;
    state.insp.innerHTML =
      '<div class="vp-insp-title">Archive entries</div>' +
      '<div class="vp-insp-section dim">' +
      '<div class="vp-insp-row">archive: <code>' + esc(state.archive) + '</code></div>' +
      '<div class="vp-insp-row">kind: <code>' + esc(state.kind || "—") + '</code></div>' +
      '</div>' +
      '<div class="vp-insp-help dim">Duplicate copies an entry verbatim. Create adds a new ' +
      'entry from an upload or a template. Delete removes it (AFS renumbers later slots). ' +
      'Rename is name-addressed; AFS rename needs a filename table.</div>';
  }

  // ---- perspective spec ----------------------------------------------
  window.PSOPerspectives.register("archive-entries", {
    label: "Edit entries",
    match: function (entry, file) {
      // Container archives (AFS/BML) opt in. Kept BELOW the tile/model
      // editors so opening an asset still defaults to its viewer; this
      // perspective is reached via the perspective tab / asset_router.
      var fn = (file || "").toLowerCase();
      var ext = fn.split(".").pop();
      if (ext === "afs" || ext === "bml") return 50;
      if (entry && entry.parent_archive) {
        var pa = String(entry.parent_archive).toLowerCase();
        if (pa.endsWith(".afs") || pa.endsWith(".bml")) return 40;
      }
      return 0;
    },
    mount: function (stage, insp, ctx) {
      var path = ctx && ctx.path;
      // If the user opened an inner entry, edit its PARENT archive.
      var archive = "";
      if (ctx && ctx.entry && ctx.entry.parent_archive) {
        archive = String(ctx.entry.parent_archive).split("/").pop();
      } else {
        archive = archiveNameFromPath(path || (ctx && ctx.fileName) || "");
      }
      var state = { stage: stage, insp: insp, archive: archive, kind: null,
                    supported: false, entries: [] };
      stage._archiveState = state;
      if (!archive) {
        stage.innerHTML = '<div class="arc-perspective"><div class="dim arc-empty">' +
          'no archive selected</div></div>';
        return;
      }
      renderStage(state);
      renderInspector(state);
      loadList(state).then(function () { renderInspector(state); });
    },
    unmount: function (stage) {
      var m = stage && stage.querySelector && stage.querySelector("#arcModal");
      if (m) m.remove();
      if (stage && stage._archiveState) stage._archiveState = null;
    },
  });
})();
