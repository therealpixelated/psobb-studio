// =====================================================================
// PSOBB Texture Editor — asset router (2026-04-25, Phase 1+2).
//
// Single source of truth for "what happens when the user clicks a node
// in the asset tree". Wires the header toggle buttons and subscribes to
// the bus.asset.opened channel. Routes to the right viewer per category:
//
//   texture / container  → tile editor (window.openFile, defined by app.js)
//   model                → 3D model viewer (open() in model_viewer.js, exposed
//                          via window.psoOpenModelByPath; we install a
//                          fallback below if the module hasn't loaded yet)
//   audio                → #assetModal audio panel (HTML5 <audio>)
//   script               → #assetModal hex panel
//   quest                → #assetModal hex panel + size info
//   cinematic            → #assetModal info panel (no decoder yet)
//   metadata             → #assetModal JSON panel (.txt → text mode)
//   unknown              → #assetModal hex panel (last resort)
//
// All raw bytes come from /api/raw/<path> served by server.py. The hex
// dump caps at 16 KB to keep the modal snappy; the user can always hit
// "download" for the full bytes.
// =====================================================================

(function () {
  "use strict";

  if (window.__psoAssetRouterLoaded) return;
  window.__psoAssetRouterLoaded = true;

  // Hex dump caps. PSO scripts top out at ~30 KB; quests at ~1 MB. We
  // truncate for display so a careless click on a megabyte file doesn't
  // freeze the layout — the download link covers full extraction.
  const HEX_PREVIEW_BYTES = 16 * 1024;
  const TEXT_PREVIEW_BYTES = 64 * 1024;

  // Tracks which model + matched textures are currently open in the
  // model viewer modal so the texture override dropdown can drive
  // re-binding without reopening the whole 3D view.
  const modelCtx = {
    modelPath: null,
    inner: null,
    matched: [],   // [{path, rule, confidence}] from the manifest
    activeTex: null,
  };

  // ---- DOM helpers ---------------------------------------------------

  function $(s) { return document.querySelector(s); }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      })[c];
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

  function rawUrl(path) {
    return "/api/raw/" + path.split("/").map(encodeURIComponent).join("/");
  }

  // ---- assetModal lifecycle -----------------------------------------

  function openAssetModal(title, meta, downloadHref) {
    $("#assetModalTitle").textContent = title;
    $("#assetModalMeta").textContent = meta || "";
    const dl = $("#assetDownloadLink");
    if (downloadHref) {
      dl.href = downloadHref;
      dl.style.display = "";
    } else {
      dl.removeAttribute("href");
      dl.style.display = "none";
    }
    // Hide every panel; the caller un-hides exactly one.
    for (const p of document.querySelectorAll("#assetModal .asset-panel")) {
      p.hidden = true;
    }
    $("#assetModal").hidden = false;
  }

  function showPanel(id) {
    const el = document.getElementById(id);
    if (el) el.hidden = false;
  }

  function closeAssetModal() {
    // Stop any audio playing in the panel — leaving an audio element
    // behind would keep playing after the user dismissed the modal.
    const audio = $("#assetAudioEl");
    if (audio) {
      try { audio.pause(); audio.removeAttribute("src"); audio.load(); } catch (_e) {}
    }
    $("#assetModal").hidden = true;
  }

  // ---- viewers ------------------------------------------------------

  function openAudio(path, entry) {
    const url = rawUrl(path);
    openAssetModal(path, fmtSize(entry && entry.size), url);
    showPanel("assetAudio");
    const a = $("#assetAudioEl");
    a.src = url;
    a.play().catch(function () {
      // autoplay blocked is fine; controls let the user start manually
    });
  }

  async function openHexOrText(path, entry, mode) {
    const url = rawUrl(path);
    openAssetModal(path, fmtSize(entry && entry.size), url);
    showPanel("assetHex");
    const sel = $("#assetHexMode");
    sel.value = mode || "hex";
    const info = $("#assetHexInfo");
    const body = $("#assetHexBody");
    body.textContent = "loading…";
    info.textContent = "";
    // Wave 7: hex view streams in 16 KB chunks via ?offset=&limit=
    // so large BMLs (3.4 MB dragon) render INSTANTLY instead of
    // blocking on a 3+ MB JSON arraybuffer. The full file can be
    // paginated by the user via the Hex panel's mode/scroll controls;
    // this initial load is just the first chunk.
    const HEX_INITIAL_BYTES = 16 * 1024;
    const f = (window.psoAssetLifecycle && window.psoAssetLifecycle.fetchAsset) || fetch;
    const isAbort = (window.psoAssetLifecycle && window.psoAssetLifecycle.isAbort) || (() => false);
    let buf;
    let totalSize = entry && typeof entry.size === "number" ? entry.size : null;
    try {
      const sep = url.includes("?") ? "&" : "?";
      const headUrl = url + sep + "offset=0&limit=" + HEX_INITIAL_BYTES;
      const r = await f(headUrl, { cache: "no-store" });
      if (!r.ok) {
        // Server doesn't support range params yet → fall back to full fetch.
        if (r.status === 400 || r.status === 404) {
          const rFull = await f(url, { cache: "no-store" });
          if (!rFull.ok) throw new Error("HTTP " + rFull.status);
          buf = new Uint8Array(await rFull.arrayBuffer());
        } else {
          throw new Error("HTTP " + r.status);
        }
      } else {
        buf = new Uint8Array(await r.arrayBuffer());
        // X-Asset-Total carries the full file size when ranged.
        const tot = r.headers.get("X-Asset-Total");
        if (tot) totalSize = parseInt(tot, 10);
      }
    } catch (e) {
      if (isAbort(e)) return;  // user moved on; suppress spurious banner
      body.textContent = "(load failed: " + (e && e.message || e) + ")";
      return;
    }
    // `total` is the bytes we actually hold in `buf`. `fullSize` is the
    // ASSET's total bytes (from manifest entry or X-Asset-Total) so the
    // info line tells the user how much remains beyond the streamed
    // initial chunk.
    const total = buf.length;
    const fullSize = (typeof totalSize === "number" && totalSize > 0)
      ? totalSize
      : total;
    const isPartial = fullSize > total;
    function render() {
      const m = sel.value;
      if (m === "text") {
        const cap = Math.min(total, TEXT_PREVIEW_BYTES);
        const slice = buf.subarray(0, cap);
        // Best-effort UTF-8 decode; fall back to latin-1 on corrupt bytes
        let text;
        try {
          text = new TextDecoder("utf-8", { fatal: false }).decode(slice);
        } catch (_e) {
          text = new TextDecoder("latin1").decode(slice);
        }
        body.textContent = text;
        if (isPartial) {
          info.textContent = "showing first " + cap + " of " + fullSize + " bytes (streamed; load more on scroll)";
        } else {
          info.textContent = total > cap
            ? "showing " + cap + " of " + total + " bytes"
            : total + " bytes";
        }
      } else {
        const cap = Math.min(total, HEX_PREVIEW_BYTES);
        body.textContent = formatHexDump(buf.subarray(0, cap));
        if (isPartial) {
          info.textContent = "showing first " + cap + " of " + fullSize + " bytes (streamed; load more on scroll)";
        } else {
          info.textContent = total > cap
            ? "showing " + cap + " of " + total + " bytes"
            : total + " bytes";
        }
      }
    }
    sel.onchange = render;
    render();
  }

  function formatHexDump(u8) {
    // 16 bytes per line, address + hex columns + ascii trailer. Standard
    // dump format; matches `xxd -g 1` output for easy diffing.
    const lines = [];
    for (let off = 0; off < u8.length; off += 16) {
      const row = u8.subarray(off, off + 16);
      let hex = "";
      let ascii = "";
      for (let i = 0; i < 16; i++) {
        if (i < row.length) {
          const b = row[i];
          hex += b.toString(16).padStart(2, "0") + " ";
          ascii += (b >= 0x20 && b < 0x7f) ? String.fromCharCode(b) : ".";
        } else {
          hex += "   ";
        }
        if (i === 7) hex += " ";
      }
      lines.push(off.toString(16).padStart(8, "0") + "  " + hex + " " + ascii);
    }
    return lines.join("\n");
  }

  async function openMetadata(path, entry) {
    const url = rawUrl(path);
    const ext = (path.split(".").pop() || "").toLowerCase();
    // .txt files are text; everything else metadata-typed (PR2/PR3) goes
    // through the hex path. JSON panel is reserved for actual JSON
    // payloads (none currently in the install but the schema allows for
    // future ones).
    if (ext === "txt") {
      return openHexOrText(path, entry, "text");
    }
    if (ext === "json") {
      openAssetModal(path, fmtSize(entry && entry.size), url);
      showPanel("assetJson");
      const body = $("#assetJsonBody");
      body.textContent = "loading…";
      try {
        const r = await fetch(url, { cache: "no-store" });
        if (!r.ok) throw new Error("HTTP " + r.status);
        const text = await r.text();
        try {
          body.textContent = JSON.stringify(JSON.parse(text), null, 2);
        } catch (_e) {
          body.textContent = text;  // fall back to raw if not valid JSON
        }
      } catch (e) {
        body.textContent = "(load failed: " + (e && e.message || e) + ")";
      }
      return;
    }
    return openHexOrText(path, entry, "hex");
  }

  function openInfo(path, entry, headline) {
    openAssetModal(path, fmtSize(entry && entry.size), rawUrl(path));
    showPanel("assetInfo");
    const dl = $("#assetInfoBody");
    const rows = [
      ["path", path],
      ["category", entry.category],
      ["format", entry.format],
      ["size", fmtSize(entry.size) + " (" + entry.size + " bytes)"],
      ["parsable", entry.parsable || "—"],
    ];
    if (entry.compressed) rows.push(["compressed", "yes"]);
    if (entry.inner_format) rows.push(["inner format", entry.inner_format]);
    if (Array.isArray(entry.matched_textures) && entry.matched_textures.length) {
      const list = entry.matched_textures.map(function (m) {
        return m.path + " (" + m.rule + ", " + (m.confidence || 0).toFixed(2) + ")";
      }).join("\n");
      rows.push(["matched textures", list]);
    }
    if (Array.isArray(entry.warnings) && entry.warnings.length) {
      rows.push(["warnings", entry.warnings.join("\n")]);
    }
    dl.innerHTML = rows.map(function (kv) {
      return "<dt>" + escapeHtml(kv[0]) + "</dt><dd>" + escapeHtml(kv[1]) + "</dd>";
    }).join("");
    if (headline) {
      // headline shows above the dl in the panel header
      const titleEl = $("#assetModalTitle");
      titleEl.textContent = headline + " — " + path;
    }
  }

  // ---- model routing -------------------------------------------------

  // The model viewer (model_viewer.js) was originally driven from the
  // tile editor's "view 3D" button against the currently-open texture.
  // Phase 2 routes through the asset tree by manifest path, so we expose
  // a path-driven open() here. If model_viewer.js publishes its own
  // window.psoOpenModelByPath we delegate; otherwise we drive its modal
  // directly via DOM ids it owns (modelModal, modelTexOverride, ...).
  async function openModel(path, entry) {
    modelCtx.modelPath = path;
    modelCtx.inner = null;
    modelCtx.matched = Array.isArray(entry.matched_textures)
      ? entry.matched_textures.slice()
      : [];

    // 2026-04-25 (regression-fix-bound-textures): diagnostic log so we
    // can see what the model viewer received vs. what shows in the
    // texture panel. The texture panel reads window.psoGetTextureBinding
    // / .psoGetCurrentTextureArchive after the model resolves; both
    // values are populated by tryLoadSkinnedMesh / tryLoadRealMesh
    // inside model_viewer.js. If the panel shows "No bound textures
    // detected" check this log to see whether (a) the wrap event fired,
    // (b) matched_textures had .nj.xvm entries, (c) state populated.
    console.warn("[asset_router] openModel:", {
      path: path,
      matched_count: modelCtx.matched.length,
      matched: modelCtx.matched.map(function (m) { return m.path; }),
    });

    // Prefetch the model bundle in parallel with three.js spinning up
    // its renderer + the modal becoming visible. tryLoadSkinnedMesh +
    // populateAnimationPanel consult the bundle cache before issuing
    // their own /api/model_skinned + /api/animations fetches, so this
    // collapses 4-7 sequential round trips into one. No await — the
    // promise is cached by path; downstream consumers await it.
    if (typeof window.psoPrefetchModelBundle === "function") {
      try { window.psoPrefetchModelBundle(path); } catch (_e) {}
    }

    // If model_viewer.js has registered an entry point, use it.
    if (typeof window.psoOpenModelByPath === "function") {
      try {
        await window.psoOpenModelByPath(path, entry, modelCtx.matched);
        // After the open() awaits, log what the texture-binding bridge
        // sees. Helps diagnose the locked-file regression where the
        // texture panel displays "No bound textures detected" even
        // though the server returned binding rows.
        try {
          const arch = (typeof window.psoGetCurrentTextureArchive === "function")
            ? window.psoGetCurrentTextureArchive()
            : "(getter missing)";
          const binding = (typeof window.psoGetTextureBinding === "function")
            ? window.psoGetTextureBinding()
            : null;
          console.warn("[asset_router] post-open binding:", {
            archive: arch,
            binding_count: Array.isArray(binding) ? binding.length : "(getter missing)",
            binding_sample: Array.isArray(binding) ? binding.slice(0, 3) : null,
          });
        } catch (_e) {}
        populateTexOverride();
        await loadAndShowSkeleton(path, modelCtx.inner);
        return;
      } catch (e) {
        console.warn("[asset_router] psoOpenModelByPath threw:", e);
        // fall through to the fallback path below
      }
    }

    // Fallback: use the model viewer's existing pipeline by faking
    // a "view 3D" click against the matched texture. This works for
    // the common case where Agent 3 paired a `.bml` to a sibling
    // `.xvm` (R1 confidence 1.0). The matched-texture filename is
    // what the editor's tile pipeline knows how to open.
    const xvmMatch = modelCtx.matched.find(function (m) {
      const p = (m.path || "").toLowerCase();
      return p.endsWith(".xvm") || p.endsWith(".prs");
    });
    if (xvmMatch && typeof window.openFile === "function") {
      // Strip directory prefix if any — /api/files lives under DATA_DIR.
      const basename = xvmMatch.path.split("/").pop();
      window.openFile(basename);
      // Then chain into the 3D viewer via its own button.
      const btn = document.getElementById("btnView3D");
      if (btn) {
        // Wait a tick for openFile's tile load, then click view 3D.
        setTimeout(function () { btn.click(); }, 200);
      }
      populateTexOverride();
      await loadAndShowSkeleton(path, modelCtx.inner);
      return;
    }

    // Last resort: hex-dump the model file so the user sees something.
    openHexOrText(path, entry, "hex");
  }

  // Populate the texture override dropdown in the model modal toolbar.
  function populateTexOverride() {
    const wrap = document.getElementById("modelTexOverrideWrap");
    const sel = document.getElementById("modelTexOverride");
    if (!wrap || !sel) return;
    if (!modelCtx.matched.length) {
      wrap.hidden = true;
      sel.innerHTML = "";
      return;
    }
    wrap.hidden = false;
    sel.innerHTML = modelCtx.matched.map(function (m, i) {
      const conf = (m.confidence || 0).toFixed(2);
      const tail = " [" + (m.rule || "?") + " " + conf + "]";
      return "<option value=\"" + escapeHtml(m.path) + "\""
        + (i === 0 ? " selected" : "")
        + ">" + escapeHtml(m.path) + tail + "</option>";
    }).join("");
    // On change, swap which texture the model viewer is rendering.
    // We need to update BOTH the tile editor (so user can edit per-tile)
    // and the 3D viewer (so the new texture wraps the current mesh).
    // The model viewer exposes window.psoSetTexture for this — see
    // model_viewer.js. If unavailable we fall back to openFile alone,
    // which at least swaps the tile editor.
    sel.onchange = function () {
      const newTex = sel.value;
      const basename = newTex.split("/").pop();
      modelCtx.activeTex = newTex;
      if (typeof window.openFile === "function") {
        window.openFile(basename);
      }
      if (typeof window.psoSetTexture === "function") {
        window.psoSetTexture(basename);
      }
    };
  }

  // Load + display the skeleton overlay if the model has bones.
  // Currently this populates the bone-count display + visibility toggle;
  // the actual 3D rendering of bones is delegated to model_viewer.js
  // via window.psoSetSkeleton if it's available.
  async function loadAndShowSkeleton(path, inner) {
    const wrap = document.getElementById("modelSkeletonWrap");
    const cnt = document.getElementById("modelBoneCount");
    if (!wrap || !cnt) return;
    let url = "/api/model/" + encodeURIComponent(path) + "/skeleton";
    if (inner) url += "?inner=" + encodeURIComponent(inner);
    try {
      const r = await fetch(url);
      if (!r.ok) {
        // 400 from BML without inner is normal until inner is resolved.
        wrap.hidden = true;
        return;
      }
      const data = await r.json();
      const bones = (data && data.bones) || [];
      cnt.textContent = String(bones.length);
      // Only surface the toggle if there's a real skeleton (>1 bone).
      // Static props with a single root node aren't worth the toggle.
      wrap.hidden = bones.length <= 1;
      if (typeof window.psoSetSkeleton === "function") {
        window.psoSetSkeleton(bones);
      }
    } catch (_e) {
      wrap.hidden = true;
    }
  }

  // ---- "view as model" affordance for BML-inner textures ------------

  // When the user clicks a texture entry that lives inside a BML
  // (`<bml>#<inner>.nj.xvm` form), and the same BML carries a model
  // inner (`<inner>.nj`), offer a quick switch to the 3D viewer with
  // that texture pre-bound.
  //
  // We render the offer as a banner below the file workspace toolbar
  // (so it survives the openFile setStatus() that overwrites #status's
  // textContent). The banner uses #assetViewAsModelBanner — created on
  // first call, reused on subsequent calls. Clicking it routes through
  // the model-by-path entry-point.
  function offerViewAsModel(texPath, _entry) {
    // Only act on BML-inner xvm forms. The "model sibling" check
    // requires an `.nj.xvm` extension (the only inner form the BML
    // carries that pairs 1:1 with a sibling .nj).
    if (!texPath || texPath.indexOf("#") < 0) {
      hideViewAsModelBanner();
      return;
    }
    const lower = texPath.toLowerCase();
    if (!lower.endsWith(".nj.xvm")) {
      hideViewAsModelBanner();
      return;
    }
    const hashIdx = texPath.indexOf("#");
    const base = texPath.slice(0, hashIdx);
    const inner = texPath.slice(hashIdx + 1);
    // Strip the .xvm tail to recover the model inner-name.
    const modelInner = inner.slice(0, -4);
    const modelPath = base + "#" + modelInner;
    // Look the model entry up in the manifest so we know the matched
    // textures (we want to pre-bind THIS texture in the 3D viewer).
    // The manifest currently only synthesises top-level BML entries,
    // not BML-inner ones — so we ALSO accept the top-level BML entry
    // as long as its matched_textures references the texture we have.
    let modelEntry = null;
    if (window.PSOManifest && typeof window.PSOManifest.entries === "function") {
      for (const e of window.PSOManifest.entries()) {
        if (e && e.path === modelPath) { modelEntry = e; break; }
      }
      if (!modelEntry) {
        // Try the top-level BML — same `base` path as the texture.
        for (const e of window.PSOManifest.entries()) {
          if (e && e.path === base
              && e.category === "model"
              && Array.isArray(e.matched_textures)
              && e.matched_textures.some(function (m) { return m.path === texPath; })) {
            modelEntry = e;
            break;
          }
        }
      }
    }
    if (!modelEntry) {
      // No corresponding model entry — nothing to offer.
      hideViewAsModelBanner();
      return;
    }
    showViewAsModelBanner(modelPath, texPath, modelEntry);
  }

  function showViewAsModelBanner(modelPath, texPath, modelEntry) {
    let banner = document.getElementById("assetViewAsModelBanner");
    if (!banner) {
      banner = document.createElement("div");
      banner.id = "assetViewAsModelBanner";
      banner.className = "view-as-model-banner";
      // Inline styles keep this self-contained (no CSS update needed).
      banner.style.cssText = [
        "padding: 4px 12px",
        "margin: 4px 0",
        "background: rgba(157,78,221,0.12)",
        "border: 1px solid rgba(157,78,221,0.5)",
        "border-radius: 4px",
        "color: #c8a4f0",
        "font-size: 0.9em",
        "display: flex",
        "align-items: center",
        "gap: 8px",
      ].join(";");
      // Insert at top of the file workspace, before the toolbar.
      const fw = document.getElementById("fileWorkspace");
      if (fw && fw.firstChild) {
        fw.insertBefore(banner, fw.firstChild);
      } else if (fw) {
        fw.appendChild(banner);
      } else {
        // Fallback: append to body so the user still sees it.
        document.body.appendChild(banner);
      }
    }
    banner.innerHTML = "";
    const msg = document.createElement("span");
    msg.textContent =
      "This texture lives inside a BML model. ";
    banner.appendChild(msg);
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = "View as model →";
    btn.title = `Open ${modelPath} in the 3D viewer with this texture pre-bound`;
    btn.style.cssText = [
      "background: transparent",
      "border: 1px solid rgba(157,78,221,0.7)",
      "color: #c8a4f0",
      "padding: 2px 10px",
      "border-radius: 3px",
      "cursor: pointer",
      "font: inherit",
    ].join(";");
    btn.onclick = function () {
      // Reuse the normal model-open pipeline. We nudge the matched
      // texture list to put this very texture first so openByPath's
      // "highest-confidence first" heuristic selects it.
      const matched = (modelEntry.matched_textures || []).slice();
      const others = matched.filter(function (m) { return m && m.path !== texPath; });
      const merged = [{ path: texPath, rule: "user_chose", confidence: 1.0 }].concat(others);
      const augmentedEntry = Object.assign({}, modelEntry, { matched_textures: merged });
      if (typeof window.psoOpenModelByPath === "function") {
        window.psoOpenModelByPath(modelPath, augmentedEntry, merged);
      }
    };
    banner.appendChild(btn);
    const dismiss = document.createElement("button");
    dismiss.type = "button";
    dismiss.textContent = "×";
    dismiss.title = "dismiss";
    dismiss.style.cssText = [
      "background: transparent",
      "border: none",
      "color: #c8a4f0",
      "cursor: pointer",
      "font-size: 1.2em",
      "padding: 0 4px",
      "margin-left: auto",
    ].join(";");
    dismiss.onclick = hideViewAsModelBanner;
    banner.appendChild(dismiss);
    banner.style.display = "flex";
  }

  function hideViewAsModelBanner() {
    const banner = document.getElementById("assetViewAsModelBanner");
    if (banner) banner.style.display = "none";
  }

  // ---- top-level dispatch -------------------------------------------

  function dispatch(evt) {
    if (!evt || !evt.path) return;
    // Wave 7 (2026-04-26): rapid-click protection.
    //
    // Step 1 — abort any in-flight fetch chain for the previously-
    // opened asset. The shared AbortController lives in
    // window.psoAssetLifecycle; bumping its epoch cancels every
    // outstanding fetch that was tagged with the prior signal.
    //
    // Step 2 — debounce the OPEN itself by 100 ms so a user who
    // rapid-clicks A→B→C inside a single window only triggers the
    // expensive open() for C. The abort already kills A and B's
    // request stack on the FIRST click (each beginAsset call), but
    // skipping their bundle/skinned fetches entirely is cheaper than
    // letting them race the abort.
    if (window.psoAssetLifecycle) {
      try { window.psoAssetLifecycle.beginAsset(evt.path); } catch (_e) {}
      window.psoAssetLifecycle.debouncedOpen(evt.path, function () {
        _dispatchInner(evt);
      });
      return;
    }
    _dispatchInner(evt);
  }

  function _dispatchInner(evt) {
    const entry = evt.entry || {};
    const cat = entry.category || "unknown";
    const ext = ("." + (evt.path.split(".").pop() || "")).toLowerCase();

    // Tear down the BML-inner "view as model" banner whenever the user
    // navigates away — we'll re-show it below if the next click is a
    // BML-inner texture.
    hideViewAsModelBanner();

    // 2026-04-24: when unified-viewport mode is on, perspectives.js
    // owns asset routing and renders into the persistent vp-stage. The
    // legacy modal openers below still work in classic mode (toggled
    // off via header "classic UI" button). For unified mode we still
    // surface the BML-inner "view as model" hint since it's a useful
    // affordance regardless of UI mode.
    if (document.body.classList.contains("unified-viewport-mode")) {
      // .prs UI atlases are texture-editable too (PRS-compressed XVMH).
      if (cat === "texture" || cat === "container"
          || (cat === "ui" && ext === ".prs")) {
        try { offerViewAsModel(evt.path, entry); } catch (_e) {}
      }
      return;
    }

    switch (cat) {
      case "texture":
      case "container": {
        // Existing tile editor; only flat filenames are openFile-able.
        const basename = evt.path.split("/").pop();
        if (typeof window.openFile === "function") {
          window.openFile(basename);
        }
        // BML-inner texture special case: if the user clicked
        // `<bml>#<inner>.nj.xvm` and the same BML has a sibling
        // `<inner>.nj` model, offer a "view as model" link so the user
        // can switch to the 3D viewer with the texture pre-bound.
        // Doesn't auto-route — the user picked a texture and may want
        // to edit it.
        offerViewAsModel(evt.path, entry);
        return;
      }
      case "model":
        openModel(evt.path, entry);
        return;
      case "audio":
        openAudio(evt.path, entry);
        return;
      case "script":
        openHexOrText(evt.path, entry, "hex");
        return;
      case "quest":
        // .qst is a header+payload; .dat is the inner. Both hex-dump
        // until we have a real quest decoder.
        openHexOrText(evt.path, entry, "hex");
        return;
      case "cinematic":
        openInfo(evt.path, entry, "Cinematic (no inline decoder yet)");
        return;
      case "metadata":
        openMetadata(evt.path, entry);
        return;
      case "ui":
        // PNGs land here; just fetch and show them as text/hex by ext.
        if (ext === ".png") {
          // Show in a fresh tab — the editor's own UI doesn't host an
          // image preview pane, and the PNG is already a static asset.
          window.open(rawUrl(evt.path), "_blank");
          return;
        }
        // .prs UI assets are PRS-compressed XVMH atlases — route them to
        // the tile/texture editor (they're editable images), same as the
        // texture case, rather than a raw hex dump.
        if (ext === ".prs") {
          const basename = evt.path.split("/").pop();
          if (typeof window.openFile === "function") {
            window.openFile(basename);
          }
          offerViewAsModel(evt.path, entry);
          return;
        }
        openHexOrText(evt.path, entry, "hex");
        return;
      case "map":
        // Floor editor routing (2026-06-20): a clicked map/floor leaf can
        // open the Floor editor perspective (browse/copy/create floors).
        // The map editor stays the DEFAULT owner of map-category leaves in
        // unified mode (its match score 90 > floor-editor's 80); this
        // classic-mode branch offers the floor perspective when one exists.
        if (window.PSOPerspectives && typeof window.PSOPerspectives.switchTo === "function") {
          window.PSOPerspectives.switchTo("floor-editor", {
            path: evt.path, entry: entry,
            fileName: (evt.path.split("/").pop() || evt.path),
          });
          return;
        }
        // .rel files are area-script blobs; hex-dump for now.
        openHexOrText(evt.path, entry, "hex");
        return;
      case "unknown":
      default:
        openHexOrText(evt.path, entry, "hex");
        return;
    }
  }

  // ---- toggle-button wiring -----------------------------------------

  function wireToggles() {
    // "All assets" button uses the negative-class semantics (visible by default,
    // hide by adding hide-asset-tree). "Editables" button uses positive-class
    // semantics (hidden by default, show by adding show-file-list) so the
    // legacy panel stays out of the way unless the user explicitly opens it.
    function bindNeg(btnId, hideClass) {
      const btn = document.getElementById(btnId);
      if (!btn) return;
      btn.addEventListener("click", function () {
        const hiding = !document.body.classList.contains(hideClass);
        document.body.classList.toggle(hideClass, hiding);
        btn.classList.toggle("active", !hiding);
        btn.setAttribute("aria-pressed", hiding ? "false" : "true");
      });
    }
    function bindPos(btnId, showClass) {
      const btn = document.getElementById(btnId);
      if (!btn) return;
      btn.addEventListener("click", function () {
        const showing = !document.body.classList.contains(showClass);
        document.body.classList.toggle(showClass, showing);
        btn.classList.toggle("active", showing);
        btn.setAttribute("aria-pressed", showing ? "true" : "false");
      });
    }
    bindNeg("btnToggleTree",  "hide-asset-tree");
    bindPos("btnToggleFiles", "show-file-list");
  }

  // ---- modal close + keyboard ---------------------------------------

  function wireModal() {
    const close = $("#assetClose");
    if (close) close.addEventListener("click", closeAssetModal);
    document.addEventListener("keydown", function (e) {
      if (!$("#assetModal") || $("#assetModal").hidden) return;
      if (e.key === "Escape") closeAssetModal();
    });
  }

  // ---- coverage label ------------------------------------------------

  // Update the small "(N entries across M categories)" hint above the
  // tree once the manifest is loaded. Keeps the user oriented even when
  // the tree itself is collapsed/filtered.
  async function updateCoverageLabel() {
    const el = document.getElementById("assetTreeCoverage");
    if (!el) return;
    try {
      const r = await fetch("/api/manifest/categories", { cache: "no-store" });
      if (!r.ok) return;
      const data = await r.json();
      const total = data.total || 0;
      const cats = (data.categories || []).length;
      el.textContent = total + " entries · " + cats + " categories";
    } catch (_e) {
      // silently degrade — not worth a toast
    }
  }

  // ---- bus subscription ---------------------------------------------

  function wireBus() {
    if (!window.bus) return;
    window.bus.on("asset.opened", dispatch);
    // Lite manifest hydration (Phase 0.5 perf): tree.js emits this when
    // the full AssetEntry has been lazily fetched after the user
    // clicked. We use it to upgrade the model viewer's matched_textures
    // (needed for texture binding) without re-running the open flow.
    window.bus.on("asset.detail", function (evt) {
      if (!evt || !evt.path || !evt.entry) return;
      // If the model viewer is currently showing this entry, refresh
      // matched_textures + the texture override dropdown. We don't
      // re-trigger psoOpenModelByPath — that would tear down the
      // already-rendered mesh; we just nudge the override dropdown.
      if (modelCtx.modelPath === evt.path && Array.isArray(evt.entry.matched_textures)) {
        modelCtx.matched = evt.entry.matched_textures.slice();
        try { populateTexOverride(); } catch (_e) {}
      }
    });
  }

  function init() {
    wireToggles();
    wireModal();
    wireBus();
    updateCoverageLabel();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // Surface the dispatch entry-point so other modules (or the dev console)
  // can trigger a route programmatically.
  window.psoAssetOpen = dispatch;
})();
