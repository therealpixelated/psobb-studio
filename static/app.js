// PSOBB Texture Editor - frontend (V3)
// =====================================================================
// API contract (server.py):
//   GET  /api/files                      -> {files:[{name,size,mtime}], data_dir}
//   GET  /api/tiles/{filename}           -> {filename, tile_count, tiles:[{index,filename,width,height,fmt,src_png_b64}]}
//   GET  /api/models                     -> {models:[{name, native_scale, max_scale, supports_tta, description}], allowed_scales, allowed_tile_sizes}
//   POST /api/upscale  body:{filename,tile_index,model,scale,keep_native_dims,tile_size?,tta?,gpu_id?}
//                                        -> {tile_index,model,scale,out_b64,out_w,out_h,src_w,src_h}
//   POST /api/repack   body:{filename,tiles:[{tile_index,png_b64}],deploy:bool}
//                                        -> {filename,rebuilt_size,deploy_path,backup_path}
// =====================================================================

// Tunables. Anything that's a number or duration anywhere in the app should
// have a name here. Frozen so a typo at the call site can't accidentally
// mutate a constant at runtime.
const CONFIG = Object.freeze({
  // Status banner: how long an "ok" message lingers before fading.
  STATUS_OK_TTL_MS: 4000,
  // Toast: default lifetime + fade-out animation duration (must match CSS).
  TOAST_TTL_MS: 4000,
  TOAST_FADE_MS: 420,
  // API error response: max chars of body to surface in the thrown message.
  API_ERROR_BODY_TRUNC: 400,
  // Click-vs-drag heuristic on the inline A/B card slider.
  CARD_DRAG_PIX_THRESHOLD: 4,
  CARD_DRAG_TIME_THRESHOLD_MS: 250,
  // Modal slider keyboard step sizes (regular / shift).
  MODAL_SLIDER_STEP: 2,
  MODAL_SLIDER_STEP_BIG: 10,
  // After repack, how long to leave the progress bar visible.
  PROGRESS_FADE_MS: 350,
  // Modal handle: ms after open before we focus the slider handle.
  MODAL_FOCUS_DELAY_MS: 30,
  // IntersectionObserver threshold for off-screen anim pause.
  ANIM_OBSERVER_THRESHOLD: 0.05,
  // Animation: clamp values for fps and grid axes.
  ANIM_FPS_MIN: 1,
  ANIM_FPS_MAX: 60,
  ANIM_GRID_MIN: 1,
  ANIM_GRID_MAX: 64,
  ANIM_HEURISTIC_GRID_MAX: 16,
  ANIM_DEFAULT_FPS: 6,
  // Grid layout: card minimum width before zoom multiplier is applied.
  GRID_BASE_MIN_PX: 180,
  // Zoom range (matches the slider in index.html).
  ZOOM_MIN: 0.5,
  ZOOM_MAX: 4,
});

const state = {
  files: [],
  filtered: [],
  models: [],
  allowedScales: [2, 3, 4, 6, 8, 12, 16],
  allowedTileSizes: [0, 32, 64, 128, 256, 512],
  currentFile: null, // {name, tiles:[...]}
  // tileEdits keyed by `filename:tileIndex`. Each entry:
  //   { src_b64, up_b64, model, scale, native_dim:[w,h], out_dim:[w,h],
  //     tta, tile_size, gpu_id, source: "upscaler"|"import" }
  tileEdits: {},
  modalTileIdx: null,
  busyTiles: new Set(), // "filename:idx" of in-flight upscales
  upscaleAllAbort: false,
  sliderPct: 50,
  // Per-card slider position, keyed by `filename:idx`
  cardSliderPcts: {},
  // Animation runtime: keyed by `${filename}:${idx}:src` or `${filename}:${idx}:up`
  // Each entry: { rafId, lastTick, frame, cfg, srcImg, ctx, cardEl, paused }
  animRunners: new Map(),
  // IntersectionObserver for off-screen anim pause
  animObserver: null,
  // Current advanced settings (toolbar)
  fitMode: "contain",
  zoom: 1.0,
  // Multi-select: indexes within the currently-open file
  selectedIndices: new Set(),
  // Last-clicked tile index (anchor for shift-range)
  selectionAnchor: null,
  // Current visible-tile filter: {idx?:Set, dimPred?:fn, fmt?:int, hasEdit?:bool|null, raw}
  tileFilter: null,
  // Card-import target (set when "import file" button on a card is clicked)
  importTargetTileIdx: null,
  // V4 quality: most recent build-only export. Set after a successful
  // /api/repack with deploy=false. UX layer can read these to render a
  // "download PRS" link in the toolbar status.
  //   exportPath:     URL the artifact can be fetched from (string)
  //   exportFilename: suggested filename for the download
  //   exportSize:     byte size of the rebuilt artifact
  //   exportSpliced:  number of tiles that were spliced verbatim
  //   exportReencoded: number of tiles that re-encoded
  exportPath: null,
  exportFilename: null,
  exportSize: null,
  exportSpliced: null,
  exportReencoded: null,
  // Atlas mode (composite editing). When enabled and the current file has
  // a known layout, the tile grid is replaced with a single composite canvas.
  atlasMode: false,
  // {filename, composite_w, composite_h, placements:[...], skip_tiles:[...],
  //  src_b64, up_b64?}
  atlasState: null,
  atlasSliderPct: 50,
  atlasBusy: false,
};

// ----- shorthand
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const editKey = (filename, idx) => `${filename}:${idx}`;
const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));

// =====================================================================
// Animation config (per-tile, persisted in localStorage)
// =====================================================================
function animKey(filename, idx) {
  return `anim_${filename}_${idx}`;
}

function loadAnimCfg(filename, idx, srcW, srcH) {
  const k = animKey(filename, idx);
  let raw = null;
  try { raw = localStorage.getItem(k); } catch {}
  if (raw) {
    try {
      const p = JSON.parse(raw);
      // sanity
      if (Array.isArray(p.frameGrid) && p.frameGrid.length === 2) {
        return {
          frameGrid: [parseInt(p.frameGrid[0], 10) || 1, parseInt(p.frameGrid[1], 10) || 1],
          frameOrder: p.frameOrder === "col-major" ? "col-major" : "row-major",
          fps: clamp(parseInt(p.fps, 10) || CONFIG.ANIM_DEFAULT_FPS, CONFIG.ANIM_FPS_MIN, CONFIG.ANIM_FPS_MAX),
          enabled: !!p.enabled,
        };
      }
    } catch {}
  }
  // First-open heuristic: rectangular tiles -> guess a strip layout.
  let cols = 1, rows = 1;
  if (srcW && srcH) {
    if (srcW >= srcH * 2) {
      cols = Math.round(srcW / srcH);
      rows = 1;
    } else if (srcH >= srcW * 2) {
      cols = 1;
      rows = Math.round(srcH / srcW);
    }
    cols = clamp(cols, 1, CONFIG.ANIM_HEURISTIC_GRID_MAX);
    rows = clamp(rows, 1, CONFIG.ANIM_HEURISTIC_GRID_MAX);
  }
  return {
    frameGrid: [cols, rows],
    frameOrder: "row-major",
    fps: CONFIG.ANIM_DEFAULT_FPS,
    enabled: false,
  };
}

function saveAnimCfg(filename, idx, cfg) {
  try {
    localStorage.setItem(animKey(filename, idx), JSON.stringify(cfg));
  } catch {}
}

// =====================================================================
// small formatting helpers
// =====================================================================
// plural(2, "tile") -> "2 tiles"; plural(1, "tile") -> "1 tile".
// Pass an explicit plural form for irregular words: plural(3, "entry", "entries").
function plural(n, singular, pluralForm) {
  const word = (n === 1) ? singular : (pluralForm || singular + "s");
  return `${n} ${word}`;
}

// basename(path) -> last path segment, for both / and \ separators.
// Used to keep absolute dev paths out of user-facing displays.
function basename(p) {
  if (p == null) return "";
  const s = String(p).replace(/[\\/]+$/, "");
  const i = Math.max(s.lastIndexOf("/"), s.lastIndexOf("\\"));
  return i >= 0 ? s.slice(i + 1) : s;
}

// =====================================================================
// status banner
// =====================================================================
let statusTimer = null;
function setStatus(msg, kind = "info", { sticky = false } = {}) {
  const el = $("#status");
  el.textContent = msg;
  el.className = `status ${kind}`;
  el.title = msg;
  if (statusTimer) { clearTimeout(statusTimer); statusTimer = null; }
  if (!sticky && kind === "ok") {
    statusTimer = setTimeout(() => {
      if (el.textContent === msg) el.className = "status info";
    }, CONFIG.STATUS_OK_TTL_MS);
  }
}

// =====================================================================
// Toast notifications
//
// Toasts are for transient, one-shot events: "tile imported", "session
// saved", quick errors. Persistent error state still uses setStatus(..., {
// sticky: true }). Toasts auto-dismiss after 4s and stack top-right.
// =====================================================================
function toast(msg, kind = "info", { ttl = CONFIG.TOAST_TTL_MS } = {}) {
  const stack = $("#toastStack");
  if (!stack) {
    // No DOM yet (e.g. during early init) — fall back to status bar.
    setStatus(msg, kind);
    return;
  }
  const el = document.createElement("div");
  el.className = `toast toast-${kind}`;
  el.innerHTML = `<span class="toast-msg"></span><span class="toast-x" title="dismiss">&times;</span>`;
  el.querySelector(".toast-msg").textContent = msg;
  el.querySelector(".toast-x").addEventListener("click", () => dismissToast(el));
  stack.appendChild(el);
  if (ttl > 0) {
    setTimeout(() => dismissToast(el), ttl);
  }
}
function dismissToast(el) {
  if (!el || !el.parentNode) return;
  el.classList.add("fading");
  setTimeout(() => { if (el.parentNode) el.parentNode.removeChild(el); }, CONFIG.TOAST_FADE_MS);
}

// =====================================================================
// API wrapper - surfaces real backend errors to the user
// =====================================================================
async function api(path, opts) {
  let r;
  try {
    r = await fetch(path, opts);
  } catch (netErr) {
    throw new Error(`network error: ${netErr.message || netErr}`);
  }
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const txt = await r.text();
      if (txt) {
        try {
          const j = JSON.parse(txt);
          if (j && j.detail) msg = `${r.status}: ${j.detail}`;
          else msg = `${r.status}: ${txt.slice(0, CONFIG.API_ERROR_BODY_TRUNC)}`;
        } catch {
          msg = `${r.status}: ${txt.slice(0, CONFIG.API_ERROR_BODY_TRUNC)}`;
        }
      }
    } catch {}
    throw new Error(msg);
  }
  return r.json();
}

// =====================================================================
// In-memory asset cache (2026-06-19, perf).
// Re-opening an asset (its decoded tiles) is instant (<1ms) instead of
// re-hitting /api/tiles + re-decoding XVR server-side. The cache is a
// small LRU keyed by the tiles endpoint path; entries are invalidated
// when the live-reload bus reports that file changed on disk, or when a
// repack/deploy rewrites it (state mutates the source).
// =====================================================================
const TILE_CACHE_MAX = 24;
const _tileCache = new Map(); // path -> parsed /api/tiles response

function _tileCacheGet(path) {
  if (!_tileCache.has(path)) return null;
  // Touch for LRU ordering (delete + re-set moves to the end).
  const v = _tileCache.get(path);
  _tileCache.delete(path);
  _tileCache.set(path, v);
  return v;
}
function _tileCacheSet(path, value) {
  if (_tileCache.has(path)) _tileCache.delete(path);
  _tileCache.set(path, value);
  while (_tileCache.size > TILE_CACHE_MAX) {
    // Evict the oldest (first inserted) entry.
    const oldest = _tileCache.keys().next().value;
    _tileCache.delete(oldest);
  }
}
// Invalidate by filename substring (a repack of "foo.xvm" must drop its
// cached tiles so the next open re-fetches the rewritten file). Called
// with no arg to clear everything.
function invalidateTileCache(nameFragment) {
  if (!nameFragment) { _tileCache.clear(); return; }
  for (const k of Array.from(_tileCache.keys())) {
    if (k.includes(encodeURIComponent(nameFragment)) || k.includes(nameFragment)) {
      _tileCache.delete(k);
    }
  }
}

// Cached GET for endpoints whose response is stable until the underlying
// file changes (currently /api/tiles). Falls through to api() on a miss.
async function apiCached(path) {
  const hit = _tileCacheGet(path);
  if (hit) return hit;
  const data = await api(path);
  _tileCacheSet(path, data);
  return data;
}

// =====================================================================
// File list pane
// =====================================================================
async function loadFiles() {
  setStatus("loading files...", "busy", { sticky: true });
  try {
    const data = await api("/api/files");
    state.files = data.files || [];
    $("#dataDir").textContent = data.data_dir || "";
    $("#dataDir").title = data.data_dir || "";
    filterFiles();
    setStatus(`${plural(state.files.length, "file")} in data/`, "ok");
    // Feed the onboarding empty-state callout + header data-dir pill.
    // The onboarding module gates the "assets found" chip on the
    // authoritative /api/health data_dir.exists (not health.ok).
    if (window.psoOnboarding && window.psoOnboarding.refreshDataDir) {
      window.psoOnboarding.refreshDataDir({
        data_dir: data.data_dir || "",
        count: state.files.length,
      });
    }
  } catch (e) {
    setStatus(`failed to load files: ${e.message}`, "err", { sticky: true });
  }
}

function filterFiles() {
  const q = $("#filterBox").value.trim().toLowerCase();
  state.filtered = q
    ? state.files.filter((f) => f.name.toLowerCase().includes(q))
    : state.files.slice();
  renderFiles();
}

function fileHasEdits(name) {
  const prefix = `${name}:`;
  for (const k of Object.keys(state.tileEdits)) {
    if (k.startsWith(prefix)) return true;
  }
  return false;
}

function renderFiles() {
  const ul = $("#files");
  ul.innerHTML = "";
  for (const f of state.filtered) {
    const li = document.createElement("li");
    li.textContent = f.name;
    li.title = f.name;
    if (state.currentFile && state.currentFile.name === f.name) li.classList.add("active");
    if (fileHasEdits(f.name)) li.classList.add("has-edits");
    const small = document.createElement("small");
    const kb = (f.size / 1024).toFixed(1);
    const dt = f.mtime ? new Date(f.mtime * 1000).toLocaleDateString() : "";
    small.textContent = `${kb} KB${dt ? " - " + dt : ""}`;
    li.appendChild(small);
    li.onclick = () => openFile(f.name);
    ul.appendChild(li);
  }
  $("#fileCount").textContent = `${state.filtered.length}/${state.files.length}`;
  updateEditsCounter();
}

function updateEditsCounter() {
  const n = Object.keys(state.tileEdits).length;
  const pill = $("#editsCount");
  const btn = $("#btnClearEdits");
  if (n > 0) {
    pill.hidden = false;
    pill.textContent = `${n} edit${n === 1 ? "" : "s"} pending`;
    btn.hidden = false;
  } else {
    pill.hidden = true;
    btn.hidden = true;
  }
  const revertAllBtn = $("#btnRevertAll");
  if (revertAllBtn && state.currentFile) {
    revertAllBtn.disabled = !fileHasEdits(state.currentFile.name);
  }
}

// =====================================================================
// Workspace (tile grid)
// =====================================================================
async function openFile(name) {
  // Stop any animations from the previous file
  stopAllAnimRunners();
  // Clear per-file UI state on file switch (selection, tile-filter)
  state.selectedIndices.clear();
  state.selectionAnchor = null;
  state.tileFilter = null;
  const tfb = $("#tileFilterBox");
  if (tfb) { tfb.value = ""; tfb.classList.remove("bad"); }
  refreshSelectionUI();
  state.currentFile = { name, tiles: [] };
  renderFiles();
  $("#placeholder").hidden = true;
  $("#fileWorkspace").hidden = false;
  $("#curFile").textContent = name;
  $("#curMeta").textContent = "";
  const grid = $("#tileGrid");
  grid.innerHTML = `<div class="grid-msg">extracting tiles from ${escapeHtml(name)}...</div>`;
  setStatus(`opening ${name}...`, "busy", { sticky: true });
  try {
    const data = await apiCached(`/api/tiles/${encodeURIComponent(name)}`);
    if (!state.currentFile || state.currentFile.name !== name) return;
    state.currentFile.tiles = data.tiles || [];
    const first = state.currentFile.tiles[0];
    const dims = first ? `${first.width}x${first.height}` : "";
    $("#curMeta").textContent = `${data.tile_count} tile${data.tile_count === 1 ? "" : "s"}${dims ? " - first " + dims : ""}`;
    renderTileGrid();
    setStatus(`${plural(data.tile_count, "tile")} loaded from ${name}`, "ok");
    updateEditsCounter();
  } catch (e) {
    if (!state.currentFile || state.currentFile.name !== name) return;
    // Keep the raw exception (often a Python magic-bytes repr) in the console
    // for debugging, but show the user a plain, auto-dismissing explanation.
    console.warn(`[tiles] extract failed for ${name}:`, e);
    const friendly = "This file isn't an editable texture container.";
    grid.innerHTML = `<div class="grid-msg err">${escapeHtml(friendly)}</div>`;
    // Auto-dismissing toast (not a sticky status banner) + clear the busy state.
    toast(friendly, "err");
    setStatus("", "info");
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function renderTileGrid() {
  const grid = $("#tileGrid");
  grid.innerHTML = "";
  if (!state.currentFile || !state.currentFile.tiles.length) {
    grid.innerHTML = `<div class="grid-msg">no tiles in this file</div>`;
    return;
  }
  for (const t of state.currentFile.tiles) {
    grid.appendChild(buildTileCard(t));
  }
  applyTileFilter();
  refreshSelectionUI();
}

// PSOBB XVR pixel-format id -> short human label (mirrors
// formats/xvr_decode.py FMT_* enum). Unknown ids fall back to "fmtN".
const FMT_LABELS = {
  1: "BGRA", 2: "565", 3: "1555", 4: "4444", 5: "P8",
  6: "DXT1", 7: "DXT2", 8: "DXT3", 9: "DXT4", 10: "DXT5",
  11: "BGRA", 12: "565", 13: "1555", 14: "4444",
  17: "A8", 18: "X1555", 19: "BGRX",
};
function fmtLabel(fmt) {
  return FMT_LABELS[fmt] || `fmt${fmt}`;
}

// Decode-suspect detector (2026-06-19). Draws the decoded source PNG to
// a tiny offscreen canvas and measures per-channel variance + the count
// of distinct quantized colours. A real texture has spread; a bad decode
// (the known fmt6/DXT1 banding/noise) collapses to a near-flat band or a
// handful of values. Memoized per data-URL so re-renders are cheap.
const _decodeSuspectCache = new Map(); // src_b64 -> Promise<bool>
function checkDecodeSuspect(t) {
  const src = t && t.src_png_b64;
  if (!src) return Promise.resolve(false);
  // Only the block-compressed formats exhibit the known decode bug; skip
  // the cheap-to-decode uncompressed formats to avoid false positives on
  // legitimately flat UI sprites.
  const blockFmts = new Set([6, 7, 8, 9, 10]);
  if (!blockFmts.has(t.fmt)) return Promise.resolve(false);
  if (_decodeSuspectCache.has(src)) return _decodeSuspectCache.get(src);

  const p = new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      try {
        const S = 16; // downsample to 16x16 — enough signal, ~cheap
        const cv = document.createElement("canvas");
        cv.width = S; cv.height = S;
        const ctx = cv.getContext("2d", { willReadFrequently: true });
        if (!ctx) { resolve(false); return; }
        ctx.drawImage(img, 0, 0, S, S);
        const data = ctx.getImageData(0, 0, S, S).data;
        let n = 0, sr = 0, sg = 0, sb = 0, sr2 = 0, sg2 = 0, sb2 = 0;
        const seen = new Set();
        for (let i = 0; i < data.length; i += 4) {
          const r = data[i], g = data[i + 1], b = data[i + 2];
          sr += r; sg += g; sb += b;
          sr2 += r * r; sg2 += g * g; sb2 += b * b;
          // Quantize to 5 bits/channel for the distinct-colour count.
          seen.add(((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3));
          n++;
        }
        if (!n) { resolve(false); return; }
        const varR = sr2 / n - (sr / n) ** 2;
        const varG = sg2 / n - (sg / n) ** 2;
        const varB = sb2 / n - (sb / n) ** 2;
        const maxVar = Math.max(varR, varG, varB);
        // Suspect when there's almost no per-channel spread AND very few
        // distinct colours (a flat band / 2-tone block). Thresholds are
        // deliberately conservative to avoid flagging real flat sprites.
        const suspect = maxVar < 12 && seen.size <= 3;
        resolve(suspect);
      } catch (_e) {
        resolve(false);
      }
    };
    img.onerror = () => resolve(false);
    img.src = src;
  });
  _decodeSuspectCache.set(src, p);
  return p;
}

// =====================================================================
// Tile card
// =====================================================================
function buildTileCard(t) {
  const fname = state.currentFile.name;
  const key = editKey(fname, t.index);
  const edit = state.tileEdits[key];
  const busy = state.busyTiles.has(key);
  const cfg = loadAnimCfg(fname, t.index, t.width, t.height);

  const card = document.createElement("div");
  card.className = `tile-card fit-${state.fitMode}`;
  if (edit) card.classList.add("has-up");
  if (busy) card.classList.add("busy");
  if (state.selectedIndices.has(t.index)) card.classList.add("selected");
  card.dataset.idx = String(t.index);
  card.dataset.fname = fname;

  // -- picture
  const pic = document.createElement("div");
  pic.className = "pic";
  pic.style.setProperty("--tile-aspect", `${t.width}/${t.height}`);

  // The src image is the "below" layer. The up image is the "above" layer
  // and gets clipped from the LEFT (showing source on left, upscaled on right).
  const imgSrc = document.createElement("img");
  imgSrc.alt = `tile ${t.index} src`;
  imgSrc.className = "pic-src";
  imgSrc.draggable = false;
  imgSrc.src = t.src_png_b64;
  pic.appendChild(imgSrc);

  if (edit) {
    const imgUp = document.createElement("img");
    imgUp.alt = `tile ${t.index} upscaled`;
    imgUp.className = "pic-up";
    imgUp.draggable = false;
    imgUp.src = edit.up_b64;
    const pct = state.cardSliderPcts[key] != null ? state.cardSliderPcts[key] : 50;
    imgUp.style.clipPath = `inset(0 0 0 ${pct}%)`;
    pic.appendChild(imgUp);
  }

  // Animation canvases - kept hidden when not animating, rendered above the imgs.
  const canvasSrc = document.createElement("canvas");
  canvasSrc.className = "anim-src-canvas";
  canvasSrc.hidden = true;
  pic.appendChild(canvasSrc);
  if (edit) {
    const canvasUp = document.createElement("canvas");
    canvasUp.className = "anim-up-canvas";
    canvasUp.hidden = true;
    canvasUp.style.clipPath = `inset(0 0 0 ${state.cardSliderPcts[key] != null ? state.cardSliderPcts[key] : 50}%)`;
    pic.appendChild(canvasUp);
  }

  // Inline A/B handle (only when has-up)
  if (edit) {
    const handle = document.createElement("div");
    handle.className = "card-handle";
    const pct = state.cardSliderPcts[key] != null ? state.cardSliderPcts[key] : 50;
    handle.style.left = pct + "%";
    pic.appendChild(handle);

    const tip = document.createElement("div");
    tip.className = "card-tip";
    tip.textContent = `${pct}% upscaled`;
    pic.appendChild(tip);
  }

  if (edit) {
    const badge = document.createElement("div");
    badge.className = "badge";
    badge.textContent = "\u2714";
    badge.title = `${edit.source === "import" ? "imported PNG" : `upscaled with ${edit.model} x${edit.scale}${edit.tta ? " TTA" : ""}`}`;
    pic.appendChild(badge);
  }
  if (busy) {
    const ov = document.createElement("div");
    ov.className = "busy-overlay";
    ov.innerHTML = `<div class="spin"></div><span>upscaling...</span>`;
    pic.appendChild(ov);
  }

  // File-picker fallback for "import an upscaled PNG" (folder icon)
  // Shown on hover. Drop-target stylings handled in CSS via .dropping class.
  const importBtn = document.createElement("div");
  importBtn.className = "import-btn";
  importBtn.title = "import a PNG (e.g. external Upscayl output) as upscaled for this tile";
  importBtn.textContent = "\u{1F4C1}"; // folder
  importBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    state.importTargetTileIdx = t.index;
    $("#cardImportInput").click();
  });
  pic.appendChild(importBtn);

  // Click handling: opens modal if no upscale; if upscaled, drag = slider, click-without-drag = modal.
  bindCardSlider(card, pic, t);
  // Modifier-aware selection: shift/ctrl click on the card (anywhere) selects.
  // We attach this on the card itself so action-bar buttons (which stop propagation)
  // don't trigger selection.
  card.addEventListener("click", (e) => {
    if (e.shiftKey || e.ctrlKey || e.metaKey) {
      // We treat shift/ctrl click as selection-only — don't open modal.
      e.preventDefault();
      e.stopPropagation();
      toggleSelect(t.index, { shift: e.shiftKey, ctrl: e.ctrlKey || e.metaKey });
    }
  });

  card.appendChild(pic);

  // -- meta
  const meta = document.createElement("div");
  meta.className = "meta";
  const left = document.createElement("strong");
  left.textContent = `tile ${String(t.index).padStart(2, "0")}`;
  const right = document.createElement("span");
  right.textContent = `${t.width}x${t.height} `;
  const fmtTag = document.createElement("span");
  fmtTag.className = "fmt-tag";
  fmtTag.textContent = fmtLabel(t.fmt);
  fmtTag.title = `pixel format ${t.fmt}`;
  right.appendChild(fmtTag);
  meta.appendChild(left);
  meta.appendChild(right);
  card.appendChild(meta);

  // Decode-suspect check (2026-06-19). The fmt6/DXT1 banding bug is
  // server-side; here we just flag a tile whose decoded source looks
  // like a flat band / low-entropy block so the user can tell a bad
  // decode from intended art. Async + best-effort; never blocks render.
  checkDecodeSuspect(t).then((suspect) => {
    if (!suspect) return;
    // Card may have been re-rendered; resolve by data attrs.
    if (card.dataset.idx !== String(t.index) || card.dataset.fname !== fname) return;
    if (card.querySelector(".decode-badge")) return;
    card.classList.add("decode-suspect");
    const db = document.createElement("div");
    db.className = "decode-badge";
    db.textContent = "decode?";
    db.title =
      `This ${fmtLabel(t.fmt)} tile decoded to a near-flat / low-detail image — ` +
      `it may be a bad server-side decode (known fmt6/DXT1 issue), not the real art.`;
    pic.appendChild(db);
  }).catch(() => {});

  if (edit) {
    const tag = document.createElement("div");
    tag.className = "meta";
    const settingsLabel = `${edit.model} \u00B7 ${edit.scale}\u00D7${edit.tta ? " \u00B7 TTA" : ""}`;
    const dimLabel = edit.out_dim ? edit.out_dim.join("x") : "";
    tag.innerHTML =
      `<span class="model-tag" title="${escapeHtml(settingsLabel)}">${escapeHtml(settingsLabel)}</span>` +
      `<span class="dim">${escapeHtml(dimLabel)}</span>`;
    card.appendChild(tag);
  }

  // -- animation row
  const animBar = document.createElement("div");
  animBar.className = "anim-bar";
  const playBtn = document.createElement("button");
  playBtn.textContent = cfg.enabled ? "\u25A0" : "\u25B6";
  playBtn.title = "play / pause animation";
  playBtn.classList.toggle("playing", cfg.enabled);
  const gridSel = document.createElement("select");
  gridSel.title = "frame grid (cols x rows)";
  for (const opt of GRID_PRESETS) {
    const o = document.createElement("option");
    o.value = opt;
    o.textContent = opt;
    gridSel.appendChild(o);
  }
  const gridStr = `${cfg.frameGrid[0]}x${cfg.frameGrid[1]}`;
  if (GRID_PRESETS.includes(gridStr)) gridSel.value = gridStr;
  else { gridSel.value = "custom"; }
  const fpsSel = document.createElement("select");
  fpsSel.title = "frames per second";
  for (const f of [1, 2, 4, 6, 8, 12, 15, 24, 30, 60]) {
    const o = document.createElement("option");
    o.value = String(f);
    o.textContent = `${f}fps`;
    fpsSel.appendChild(o);
  }
  fpsSel.value = String(cfg.fps);
  const spacer = document.createElement("span");
  spacer.className = "anim-spacer";

  animBar.appendChild(playBtn);
  animBar.appendChild(gridSel);
  animBar.appendChild(fpsSel);
  animBar.appendChild(spacer);
  card.appendChild(animBar);

  playBtn.onclick = (e) => {
    e.stopPropagation();
    cfg.enabled = !cfg.enabled;
    saveAnimCfg(fname, t.index, cfg);
    playBtn.textContent = cfg.enabled ? "\u25A0" : "\u25B6";
    playBtn.classList.toggle("playing", cfg.enabled);
    if (cfg.enabled) startCardAnimation(card, t);
    else stopCardAnimation(card, t);
  };
  gridSel.onclick = (e) => e.stopPropagation();
  gridSel.onchange = (e) => {
    e.stopPropagation();
    const v = gridSel.value;
    if (v === "custom") {
      const ans = prompt("custom grid (cols x rows), e.g. 3x2", `${cfg.frameGrid[0]}x${cfg.frameGrid[1]}`);
      if (ans) {
        const m = ans.match(/^(\d+)\s*[xX*]\s*(\d+)$/);
        if (m) {
          cfg.frameGrid = [
            clamp(parseInt(m[1], 10), CONFIG.ANIM_GRID_MIN, CONFIG.ANIM_GRID_MAX),
            clamp(parseInt(m[2], 10), CONFIG.ANIM_GRID_MIN, CONFIG.ANIM_GRID_MAX),
          ];
        }
      }
    } else {
      const m = v.match(/^(\d+)x(\d+)$/);
      if (m) cfg.frameGrid = [parseInt(m[1], 10), parseInt(m[2], 10)];
    }
    saveAnimCfg(fname, t.index, cfg);
    if (cfg.enabled) {
      stopCardAnimation(card, t);
      startCardAnimation(card, t);
    }
  };
  fpsSel.onclick = (e) => e.stopPropagation();
  fpsSel.onchange = (e) => {
    e.stopPropagation();
    cfg.fps = clamp(parseInt(fpsSel.value, 10) || CONFIG.ANIM_DEFAULT_FPS, CONFIG.ANIM_FPS_MIN, CONFIG.ANIM_FPS_MAX);
    saveAnimCfg(fname, t.index, cfg);
    // running anim re-reads cfg.fps each tick - no need to restart
  };

  // -- per-card actions
  const actions = document.createElement("div");
  actions.className = "actions";
  const bUp = document.createElement("button");
  bUp.textContent = edit ? "re-upscale" : "upscale";
  bUp.disabled = busy;
  bUp.onclick = (e) => {
    e.stopPropagation();
    triggerUpscale(t.index);
  };
  actions.appendChild(bUp);
  const bRev = document.createElement("button");
  bRev.textContent = "revert";
  bRev.className = "danger";
  bRev.disabled = !edit || busy;
  bRev.onclick = (e) => {
    e.stopPropagation();
    revertTile(t.index);
  };
  actions.appendChild(bRev);
  card.appendChild(actions);

  // Auto-start anim if cfg.enabled
  if (cfg.enabled) {
    // wait for DOM insert, then start
    queueMicrotask(() => startCardAnimation(card, t));
  }

  // Observe for off-screen pause
  if (state.animObserver) state.animObserver.observe(card);

  return card;
}

const GRID_PRESETS = ["1x1", "2x1", "4x1", "1x2", "1x4", "2x2", "4x4", "8x1", "1x8", "5x3", "custom"];

function triggerUpscale(idx) {
  const opts = currentUpscaleOpts();
  upscaleTile(idx, opts);
}

function currentUpscaleOpts() {
  const tileSizeRaw = $("#tileSizeSel").value;
  const gpuRaw = $("#gpuSel").value;
  return {
    model: $("#modelSel").value,
    scale: parseInt($("#scaleSel").value, 10),
    keepNative: $("#keepNative").checked,
    tta: $("#ttaToggle").checked,
    tileSize: tileSizeRaw === "auto" ? null : parseInt(tileSizeRaw, 10),
    gpuId: gpuRaw === "auto" ? null : parseInt(gpuRaw, 10),
  };
}

// Replace just one card without re-rendering the entire grid
function updateTileCard(idx) {
  const t = state.currentFile && state.currentFile.tiles.find((x) => x.index === idx);
  if (!t) return;
  const grid = $("#tileGrid");
  const old = grid.querySelector(`.tile-card[data-idx="${idx}"]`);
  if (old) {
    // Stop any running anim on the old card before swap
    stopCardAnimation(old, t);
    if (state.animObserver) state.animObserver.unobserve(old);
  }
  const fresh = buildTileCard(t);
  if (old) old.replaceWith(fresh);
  else grid.appendChild(fresh);
}

// =====================================================================
// Inline A/B card slider
// =====================================================================
// Single shared drag-tracking state - only ever one slider being dragged at
// a time, so a single set of doc-level listeners is enough (set up in init()).
const _cardDragState = {
  active: false,
  card: null,
  pic: null,
  tileIdx: null,
  fname: null,
  mouseDownAt: null,
  moved: false,
};

function _cardPctFromEvent(e) {
  const ds = _cardDragState;
  if (!ds.pic) return 50;
  const rect = ds.pic.getBoundingClientRect();
  const cx = e.touches ? (e.touches[0] || e.changedTouches[0]).clientX : e.clientX;
  return clamp(((cx - rect.left) / rect.width) * 100, 0, 100);
}

function _cardDragMove(e) {
  const ds = _cardDragState;
  if (!ds.active) return;
  if (e.cancelable && !e.touches) e.preventDefault();
  const pct = _cardPctFromEvent(e);
  ds.moved = true;
  state.cardSliderPcts[editKey(ds.fname, ds.tileIdx)] = Math.round(pct);
  applyCardSlider(ds.card, pct);
  if (state.modalTileIdx === ds.tileIdx) {
    setSliderPct(pct);
  }
}

function _cardDragEnd(e) {
  const ds = _cardDragState;
  if (!ds.active) return;
  const card = ds.card;
  const idx = ds.tileIdx;
  const startAt = ds.mouseDownAt;
  const moved = ds.moved;
  ds.active = false;
  ds.card = null;
  ds.pic = null;
  ds.tileIdx = null;
  ds.fname = null;
  ds.mouseDownAt = null;
  ds.moved = false;
  if (card) card.classList.remove("dragging");

  // Click vs drag heuristic: small movement + short time = click -> open modal
  if (startAt) {
    const ex = e.clientX != null ? e.clientX : (e.changedTouches && e.changedTouches[0] ? e.changedTouches[0].clientX : startAt.x);
    const ey = e.clientY != null ? e.clientY : (e.changedTouches && e.changedTouches[0] ? e.changedTouches[0].clientY : startAt.y);
    const dx = ex - startAt.x;
    const dy = ey - startAt.y;
    const dt = Date.now() - startAt.time;
    if (!moved || (Math.hypot(dx, dy) < CONFIG.CARD_DRAG_PIX_THRESHOLD && dt < CONFIG.CARD_DRAG_TIME_THRESHOLD_MS)) {
      openModal(idx);
    }
  }
}

function setupCardDragListeners() {
  document.addEventListener("mousemove", _cardDragMove);
  document.addEventListener("mouseup", _cardDragEnd);
  document.addEventListener("touchmove", _cardDragMove, { passive: true });
  document.addEventListener("touchend", _cardDragEnd);
  document.addEventListener("touchcancel", _cardDragEnd);
}

function bindCardSlider(card, pic, t) {
  const fname = state.currentFile.name;
  const key = editKey(fname, t.index);

  const startDrag = (e, isTouch) => {
    const hasUp = !!state.tileEdits[key];
    const t0 = isTouch ? e.touches[0] : e;
    // If modifier keys are held, this is a multi-select gesture, not a drag/modal
    // — the card-level click listener handles it. Don't engage drag mode.
    const isModifier = !isTouch && (e.shiftKey || e.ctrlKey || e.metaKey);
    if (isModifier) return;
    _cardDragState.mouseDownAt = { x: t0.clientX, y: t0.clientY, time: Date.now() };
    _cardDragState.moved = false;
    if (!hasUp) {
      // No upscale yet - mousedown still allows opening modal on click. Don't
      // engage drag mode; just set mouseDownAt and let _cardDragEnd handle it.
      _cardDragState.active = true; // so _cardDragEnd fires
      _cardDragState.card = card;
      _cardDragState.pic = pic;
      _cardDragState.tileIdx = t.index;
      _cardDragState.fname = fname;
      return;
    }
    _cardDragState.active = true;
    _cardDragState.card = card;
    _cardDragState.pic = pic;
    _cardDragState.tileIdx = t.index;
    _cardDragState.fname = fname;
    card.classList.add("dragging");
    if (!isTouch) e.preventDefault();
    // Snap immediately on press
    const pct = _cardPctFromEvent(e);
    state.cardSliderPcts[key] = Math.round(pct);
    applyCardSlider(card, pct);
    if (state.modalTileIdx === t.index) setSliderPct(pct);
  };

  pic.addEventListener("mousedown", (e) => startDrag(e, false));
  pic.addEventListener("touchstart", (e) => startDrag(e, true), { passive: true });
}

function applyCardSlider(card, pct) {
  const upImg = card.querySelector(".pic .pic-up");
  const upCanvas = card.querySelector(".pic .anim-up-canvas");
  const handle = card.querySelector(".pic .card-handle");
  const tip = card.querySelector(".pic .card-tip");
  if (upImg) upImg.style.clipPath = `inset(0 0 0 ${pct}%)`;
  if (upCanvas) upCanvas.style.clipPath = `inset(0 0 0 ${pct}%)`;
  if (handle) handle.style.left = pct + "%";
  if (tip) tip.textContent = `${Math.round(pct)}% upscaled`;
}

// =====================================================================
// Animation engine
// =====================================================================
// keyed by `${filename}:${idx}:src` and `${filename}:${idx}:up` for modal use
function animKeyRunner(filename, idx, kind) {
  return `${filename}:${idx}:${kind}`;
}

function startCardAnimation(card, t) {
  const fname = state.currentFile.name;
  const cfg = loadAnimCfg(fname, t.index, t.width, t.height);
  if (!cfg.enabled) return;
  const edit = state.tileEdits[editKey(fname, t.index)];

  startCanvasAnim({
    runnerKey: animKeyRunner(fname, t.index, "src"),
    canvas: card.querySelector(".pic .anim-src-canvas"),
    imgEl: card.querySelector(".pic .pic-src"),
    cfg,
    onCfgChange: () => loadAnimCfg(fname, t.index, t.width, t.height),
  });

  if (edit) {
    startCanvasAnim({
      runnerKey: animKeyRunner(fname, t.index, "up"),
      canvas: card.querySelector(".pic .anim-up-canvas"),
      imgEl: card.querySelector(".pic .pic-up"),
      cfg,
      onCfgChange: () => loadAnimCfg(fname, t.index, t.width, t.height),
    });
  }
}

function stopCardAnimation(card, t) {
  const fname = (card && card.dataset && card.dataset.fname) || (state.currentFile && state.currentFile.name);
  if (!fname) return;
  stopCanvasAnim(animKeyRunner(fname, t.index, "src"));
  stopCanvasAnim(animKeyRunner(fname, t.index, "up"));
  // Re-show the still imgs
  if (card) {
    const sImg = card.querySelector(".pic .pic-src");
    if (sImg) sImg.style.visibility = "";
    const uImg = card.querySelector(".pic .pic-up");
    if (uImg) uImg.style.visibility = "";
    const sCv = card.querySelector(".pic .anim-src-canvas");
    if (sCv) sCv.hidden = true;
    const uCv = card.querySelector(".pic .anim-up-canvas");
    if (uCv) uCv.hidden = true;
  }
}

function stopAllAnimRunners() {
  for (const k of [...state.animRunners.keys()]) {
    stopCanvasAnim(k);
  }
}

function startCanvasAnim({ runnerKey, canvas, imgEl, cfg, onCfgChange }) {
  // Stop previous runner under this key, if any
  stopCanvasAnim(runnerKey);
  if (!canvas || !imgEl) return;

  const baseSrc = imgEl.currentSrc || imgEl.src;
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    canvas.hidden = false;
    imgEl.style.visibility = "hidden";
    sizeCanvas(canvas);
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;

    const runner = {
      raf: 0,
      lastFrameTime: 0,
      frame: 0,
      cfg,
      img,
      ctx,
      canvas,
      paused: false,
      onCfgChange,
    };
    state.animRunners.set(runnerKey, runner);
    const tick = (ts) => {
      const r = state.animRunners.get(runnerKey);
      if (!r) return;
      if (r.paused) {
        r.raf = requestAnimationFrame(tick);
        return;
      }
      // Reload latest config on each tick (cheap)
      const live = r.onCfgChange ? r.onCfgChange() : r.cfg;
      const fps = clamp(live.fps || CONFIG.ANIM_DEFAULT_FPS, CONFIG.ANIM_FPS_MIN, CONFIG.ANIM_FPS_MAX);
      const interval = 1000 / fps;
      if (!r.lastFrameTime) r.lastFrameTime = ts;
      const elapsed = ts - r.lastFrameTime;
      if (elapsed >= interval) {
        const [cols, rows] = live.frameGrid;
        const totalFrames = cols * rows;
        const fw = Math.floor(r.img.naturalWidth / cols);
        const fh = Math.floor(r.img.naturalHeight / rows);
        // resize canvas if needed
        sizeCanvas(canvas, fw, fh);
        const ctx2 = canvas.getContext("2d");
        ctx2.imageSmoothingEnabled = false;
        const idx = r.frame % totalFrames;
        let col, row;
        if (live.frameOrder === "col-major") {
          col = Math.floor(idx / rows);
          row = idx % rows;
        } else {
          col = idx % cols;
          row = Math.floor(idx / cols);
        }
        ctx2.clearRect(0, 0, canvas.width, canvas.height);
        ctx2.drawImage(r.img, col * fw, row * fh, fw, fh, 0, 0, canvas.width, canvas.height);
        r.frame = (r.frame + 1) % totalFrames;
        r.lastFrameTime = ts;
      }
      r.raf = requestAnimationFrame(tick);
    };
    runner.raf = requestAnimationFrame(tick);
  };
  img.onerror = () => {
    console.warn("[anim] failed to load", baseSrc);
  };
  img.src = baseSrc;
}

function stopCanvasAnim(runnerKey) {
  const r = state.animRunners.get(runnerKey);
  if (!r) return;
  if (r.raf) cancelAnimationFrame(r.raf);
  state.animRunners.delete(runnerKey);
  if (r.canvas) {
    const ctx = r.canvas.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, r.canvas.width, r.canvas.height);
    r.canvas.hidden = true;
  }
}

function sizeCanvas(canvas, fw, fh) {
  // Match canvas internal pixels to its CSS bounding box for crispness.
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}

function pauseRunnerByKeyPrefix(prefix, paused) {
  for (const [k, r] of state.animRunners.entries()) {
    if (k.startsWith(prefix)) r.paused = paused;
  }
}

function setupAnimObserver() {
  if (!("IntersectionObserver" in window)) return;
  state.animObserver = new IntersectionObserver((entries) => {
    for (const ent of entries) {
      const card = ent.target;
      const idx = card.dataset.idx;
      const fname = card.dataset.fname || (state.currentFile && state.currentFile.name);
      if (!fname || idx == null) continue;
      pauseRunnerByKeyPrefix(`${fname}:${idx}:`, !ent.isIntersecting);
    }
  }, { root: null, threshold: CONFIG.ANIM_OBSERVER_THRESHOLD });
}

// =====================================================================
// Upscale / revert
// =====================================================================
async function upscaleTile(idx, opts) {
  if (!state.currentFile) return false;
  const t = state.currentFile.tiles.find((x) => x.index === idx);
  if (!t) return false;
  const fname = state.currentFile.name;
  const key = editKey(fname, idx);

  if (state.busyTiles.has(key)) return false;

  state.busyTiles.add(key);
  updateTileCard(idx);
  if (state.modalTileIdx === idx) showModalSpinner(true);
  const settingsLabel = `${opts.model} x${opts.scale}${opts.tta ? " TTA" : ""}`;
  setStatus(`upscaling ${fname} tile ${idx} via ${settingsLabel}...`, "busy", { sticky: true });

  try {
    const body = {
      filename: fname,
      tile_index: idx,
      model: opts.model,
      scale: opts.scale,
      keep_native_dims: opts.keepNative,
      tta: !!opts.tta,
    };
    if (opts.tileSize !== null && opts.tileSize !== undefined) body.tile_size = opts.tileSize;
    if (opts.gpuId !== null && opts.gpuId !== undefined) body.gpu_id = opts.gpuId;
    const r = await api("/api/upscale", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state.tileEdits[key] = {
      src_b64: t.src_png_b64,
      up_b64: r.out_b64,
      model: r.model || opts.model,
      scale: r.scale || opts.scale,
      tta: !!opts.tta,
      tile_size: opts.tileSize,
      gpu_id: opts.gpuId,
      native_dim: [t.width, t.height],
      out_dim: [r.out_w, r.out_h],
    };
    setStatus(`tile ${idx} upscaled (${r.out_w}x${r.out_h} from ${t.width}x${t.height} src)`, "ok");
    return true;
  } catch (e) {
    setStatus(`tile ${idx} upscale failed: ${e.message}`, "err", { sticky: true });
    return false;
  } finally {
    state.busyTiles.delete(key);
    updateTileCard(idx);
    updateEditsCounter();
    renderFiles();
    if (state.modalTileIdx === idx) {
      showModalSpinner(false);
      refreshModalImages();
    }
  }
}

function revertTile(idx) {
  if (!state.currentFile) return;
  const fname = state.currentFile.name;
  const key = editKey(fname, idx);
  if (!state.tileEdits[key]) return;
  delete state.tileEdits[key];
  delete state.cardSliderPcts[key];
  updateTileCard(idx);
  updateEditsCounter();
  renderFiles();
  if (state.modalTileIdx === idx) refreshModalImages();
  setStatus(`tile ${idx} reverted to source`, "ok");
}

function revertAllInCurrentFile() {
  if (!state.currentFile) return;
  const fname = state.currentFile.name;
  const before = Object.keys(state.tileEdits).filter((k) => k.startsWith(fname + ":"));
  if (before.length === 0) return;
  if (!confirm(`Revert ${plural(before.length, "edited tile")} in ${fname}? This only clears in-memory edits.`)) return;
  for (const k of before) {
    delete state.tileEdits[k];
    delete state.cardSliderPcts[k];
  }
  renderTileGrid();
  updateEditsCounter();
  renderFiles();
  setStatus(`reverted ${plural(before.length, "tile")} in ${fname}`, "ok");
}

function clearAllEdits() {
  const n = Object.keys(state.tileEdits).length;
  if (n === 0) return;
  if (!confirm(`Discard all ${plural(n, "pending edit")} across every file?`)) return;
  state.tileEdits = {};
  state.cardSliderPcts = {};
  if (state.currentFile) renderTileGrid();
  updateEditsCounter();
  renderFiles();
  if (state.modalTileIdx !== null) refreshModalImages();
  setStatus(`cleared ${plural(n, "edit")}`, "ok");
}

async function upscaleAll() {
  if (!state.currentFile) return;
  const tiles = state.currentFile.tiles;
  if (!tiles.length) return;
  const opts = currentUpscaleOpts();

  state.upscaleAllAbort = false;
  const btn = $("#btnUpscaleAll");
  const origText = btn.textContent;
  btn.textContent = "stop";
  btn.title = "click again to stop after current tile";
  let stopRequested = false;
  const stopHandler = (ev) => { ev.stopPropagation(); stopRequested = true; btn.textContent = "stopping..."; btn.disabled = true; };
  btn.removeEventListener("click", upscaleAll);
  btn.addEventListener("click", stopHandler, { once: true });

  showProgress(0, tiles.length);
  let ok = 0, fail = 0;
  for (let i = 0; i < tiles.length; i++) {
    if (stopRequested) break;
    const t = tiles[i];
    const success = await upscaleTile(t.index, opts);
    if (success) ok++; else fail++;
    showProgress(i + 1, tiles.length);
  }
  hideProgress();
  btn.textContent = origText;
  btn.title = "upscale every tile (U)";
  btn.disabled = false;
  btn.removeEventListener("click", stopHandler);
  btn.addEventListener("click", upscaleAll);

  if (stopRequested) {
    setStatus(`upscale all stopped (${ok} ok, ${fail} failed)`, fail > 0 ? "err" : "info", { sticky: true });
  } else if (fail > 0) {
    setStatus(`upscale all done: ${ok} ok, ${fail} failed`, "err", { sticky: true });
  } else {
    setStatus(`upscale all done: ${plural(ok, "tile")}`, "ok");
  }
}

function showProgress(done, total) {
  const bar = $("#progressBar");
  bar.hidden = false;
  const pct = total > 0 ? (done / total) * 100 : 0;
  bar.querySelector(".bar").style.width = pct + "%";
  bar.querySelector(".lbl").textContent = `${done}/${total}`;
}
function hideProgress() {
  setTimeout(() => { $("#progressBar").hidden = true; }, CONFIG.PROGRESS_FADE_MS);
}

// =====================================================================
// Repack & deploy
//
// The user-facing entry is now `repackDeployFlow()` which opens the diff
// modal first. The actual HTTP call lives in `doRepackDeploy()` and is
// triggered when the user clicks "deploy now" inside the modal. The
// previous `repackDeploy()` shape (no modal) is kept under
// `repackDeployImmediate` for any caller that wants to bypass — currently
// none, but it's clean separation.
// =====================================================================
async function repackDeployFlow() {
  if (!state.currentFile) return;
  const fname = state.currentFile.name;
  const prefix = `${fname}:`;
  const edits = [];
  for (const [k, e] of Object.entries(state.tileEdits)) {
    if (!k.startsWith(prefix)) continue;
    const idx = parseInt(k.slice(prefix.length), 10);
    edits.push({ tile_index: idx, png_b64: e.up_b64 });
  }
  if (edits.length === 0) {
    setStatus("nothing to repack - upscale at least one tile first", "err", { sticky: true });
    toast("nothing to repack — upscale at least one tile first", "err");
    return;
  }
  // 2026-04-25 (regression-fix-modal-vs-viewport): toggle the unified-mode
  // CSS allow-class so the deploy modal renders. Removed in hideDeployDiff.
  document.body.classList.add("pso-modal-allow-deploy");
  await showDeployDiff();
}

async function doRepackDeploy() {
  if (!state.currentFile) return;
  const fname = state.currentFile.name;
  const prefix = `${fname}:`;
  const edits = [];
  for (const [k, e] of Object.entries(state.tileEdits)) {
    if (!k.startsWith(prefix)) continue;
    const idx = parseInt(k.slice(prefix.length), 10);
    edits.push({ tile_index: idx, png_b64: e.up_b64 });
  }
  if (edits.length === 0) {
    toast("nothing to repack", "err");
    return;
  }
  hideDeployDiff();
  const btn = $("#btnRepack");
  btn.disabled = true;
  btn.textContent = "repacking...";
  setStatus(`repacking ${fname} (${plural(edits.length, "tile")})...`, "busy", { sticky: true });
  try {
    const r = await api("/api/repack", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: fname, tiles: edits, deploy: true }),
    });
    const bakName = r.backup_path ? r.backup_path.split(/[\\/]/).pop() : "(no backup)";
    const sizeKb = r.rebuilt_size ? (r.rebuilt_size / 1024).toFixed(1) : "?";
    const spliceTag = (typeof r.spliced_count === "number")
      ? ` (${r.spliced_count} spliced, ${r.reencoded_count} re-encoded)`
      : "";
    setStatus(`deployed ${fname} (${sizeKb} KB)${spliceTag} - backup: ${bakName}`, "ok", { sticky: true });
    toast(`deployed ${fname} (${sizeKb} KB)${spliceTag}`, "ok");
    for (const k of Object.keys(state.tileEdits)) {
      if (k.startsWith(fname + ":")) {
        delete state.tileEdits[k];
        delete state.cardSliderPcts[k];
      }
    }
    // The source file was just rewritten on disk; drop its cached tiles
    // so the re-open below re-fetches the repacked bytes.
    invalidateTileCache(fname);
    await loadFiles();
    if (state.currentFile && state.currentFile.name === fname) {
      await openFile(fname);
    }
    updateEditsCounter();
  } catch (e) {
    setStatus(`repack failed: ${e.message}`, "err", { sticky: true });
    toast(`repack failed: ${e.message}`, "err", { ttl: 7000 });
  } finally {
    btn.disabled = false;
    btn.textContent = "repack & deploy";
  }
}


// V4 quality: build-only path that does NOT touch DATA_DIR. Returns an
// export URL the caller can download. Sets `state.exportPath` /
// `state.exportFilename` / `state.exportSize` so the UX layer can render
// a "download PRS" link. The `repack & deploy` flow is unchanged.
async function doRepackBuildOnly() {
  if (!state.currentFile) {
    toast("no file open", "err");
    return null;
  }
  const fname = state.currentFile.name;
  const prefix = `${fname}:`;
  const edits = [];
  for (const [k, e] of Object.entries(state.tileEdits)) {
    if (!k.startsWith(prefix)) continue;
    const idx = parseInt(k.slice(prefix.length), 10);
    edits.push({ tile_index: idx, png_b64: e.up_b64 });
  }
  setStatus(`building ${fname} (no deploy)...`, "busy", { sticky: true });
  try {
    const r = await api("/api/repack", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: fname, tiles: edits, deploy: false }),
    });
    state.exportPath = r.export_url || null;
    state.exportFilename = r.export_filename || fname;
    state.exportSize = r.rebuilt_size || null;
    state.exportSpliced = (typeof r.spliced_count === "number") ? r.spliced_count : null;
    state.exportReencoded = (typeof r.reencoded_count === "number") ? r.reencoded_count : null;
    const sizeKb = r.rebuilt_size ? (r.rebuilt_size / 1024).toFixed(1) : "?";
    const spliceTag = (state.exportSpliced != null)
      ? ` (${state.exportSpliced} spliced, ${state.exportReencoded} re-encoded)`
      : "";
    if (state.exportPath) {
      setStatus(`built ${fname} (${sizeKb} KB)${spliceTag} — download via toolbar`, "ok", { sticky: true });
      toast(`built ${fname} ready for download`, "ok");
    } else {
      setStatus(`built ${fname} (${sizeKb} KB)${spliceTag} — no download URL returned`, "ok", { sticky: true });
    }
    // Notify any registered listeners — UX agent can wire a button to
    // listen on the custom event and update the DOM.
    document.dispatchEvent(new CustomEvent("exportReady", { detail: {
      url: state.exportPath, filename: state.exportFilename,
      size: state.exportSize, spliced: state.exportSpliced,
      reencoded: state.exportReencoded,
    }}));
    return r;
  } catch (e) {
    setStatus(`build failed: ${e.message}`, "err", { sticky: true });
    toast(`build failed: ${e.message}`, "err", { ttl: 7000 });
    return null;
  }
}

// =====================================================================
// Modal A/B slider
//
// 2026-04-25 (regression-fix-modal-vs-viewport): In unified-viewport
// mode, the user expects the tile A/B view to render INSIDE the
// persistent stage, not pop a fullscreen modal. We achieve that by
// re-routing openModal() to switch to the "tile-detail" perspective
// (registered in perspectives.js), which yanks the same DOM pieces
// (#modalUpscaleBar, .ab-stage, .modal-anim-bar) into vp-stage and
// calls openModal() with the bypass flag set. The legacy fullscreen
// modal is now debug-only, reachable solely by setting
// window.psoUseLegacyModal = true (the classic-UI toggle was removed).
// =====================================================================
function openModal(idx) {
  if (!state.currentFile) return;
  const t = state.currentFile.tiles.find((x) => x.index === idx);
  if (!t) return;

  // Route through the tile-detail perspective when:
  //   - body has .unified-viewport-mode (default for new sessions), AND
  //   - the legacy-modal feature flag is off, AND
  //   - perspectives.js is loaded, AND
  //   - we're not already inside a perspective mount calling us
  //     (the tile-detail perspective sets _psoOpenModalBypass before
  //     calling openModal so the inner A/B init can run).
  const inUnifiedMode = document.body.classList.contains("unified-viewport-mode");
  const useLegacy = !!window.psoUseLegacyModal;
  const havePerspectives = !!window.PSOPerspectives;
  if (
    inUnifiedMode &&
    !useLegacy &&
    havePerspectives &&
    !window._psoOpenModalBypass
  ) {
    try {
      // Synthesise a perspective context from the active file. The
      // perspective's mount() will call openModal again with the bypass
      // flag, which falls through to the rest of this function.
      const ctx = {
        path: state.currentFile.name,
        entry: {
          category: "container",
          format: state.currentFile.name.toLowerCase().endsWith(".prs") ? "PRS" : "XVM",
        },
        fileName: state.currentFile.name,
        tileIdx: idx,
      };
      window.PSOPerspectives.switchTo("tile-detail", ctx);
      return;
    } catch (e) {
      console.warn("[openModal] tile-detail perspective failed; opening modal:", e);
    }
  }

  state.modalTileIdx = idx;

  $("#modal").hidden = false;
  // Set the modal aspect ratio per tile (so source + upscaled line up 1:1)
  $("#abWrap").style.setProperty("--tile-aspect", `${t.width}/${t.height}`);

  refreshModalImages();
  // Restore per-card slider state for this tile, if any
  const key = editKey(state.currentFile.name, idx);
  const pct = state.cardSliderPcts[key] != null ? state.cardSliderPcts[key] : 50;
  setSliderPct(pct);
  // Sync animation controls to this tile's cfg
  syncModalAnimControls(t);

  // AI gen tab: reset to upscale tab each open, refresh tile-keyed UI bits.
  // (We don't auto-jump to AI tab — user opts in via the tab button.)
  if (state && state.aigen && typeof aigenShowTab === "function") {
    aigenShowTab("upscale");
    if (typeof aigenLoadVariationsStrip === "function") aigenLoadVariationsStrip();
    if (state.aigen.mode === "inpaint" && typeof aigenEnsureMaskCanvas === "function") {
      aigenEnsureMaskCanvas();
    }
  }

  setTimeout(() => $("#abHandle").focus(), CONFIG.MODAL_FOCUS_DELAY_MS);
}

function closeModal() {
  $("#modal").hidden = true;
  // Stop any modal-driven anim
  if (state.modalTileIdx != null && state.currentFile) {
    stopCanvasAnim(animKeyRunner(state.currentFile.name, state.modalTileIdx, "modal-src"));
    stopCanvasAnim(animKeyRunner(state.currentFile.name, state.modalTileIdx, "modal-up"));
    $("#abSrcCanvas").hidden = true;
    $("#abDstCanvas").hidden = true;
    $("#abSrc").style.visibility = "";
    $("#abDst").style.visibility = "";
  }
  state.modalTileIdx = null;
}

function showModalSpinner(visible) {
  $("#abSpinner").hidden = !visible;
}

function refreshModalImages() {
  const idx = state.modalTileIdx;
  if (idx === null || idx === undefined || !state.currentFile) return;
  const t = state.currentFile.tiles.find((x) => x.index === idx);
  if (!t) return;
  const fname = state.currentFile.name;
  const key = editKey(fname, idx);
  const edit = state.tileEdits[key];

  $("#modalTitle").textContent = `${fname} - tile ${String(idx).padStart(2, "0")}`;
  if (edit) {
    $("#modalMeta").textContent = `(${t.width}x${t.height} fmt${t.fmt}) -> (${edit.out_dim ? edit.out_dim.join("x") : "?x?"})`;
  } else {
    $("#modalMeta").textContent = `(${t.width}x${t.height} fmt${t.fmt})`;
  }

  $("#abSrc").src = t.src_png_b64;
  if (edit) {
    $("#abDst").src = edit.up_b64;
    $("#abLabelLeft").textContent = "source";
    $("#abLabelRight").textContent = `${edit.model} x${edit.scale}${edit.tta ? " TTA" : ""}${edit.out_dim ? ` (${edit.out_dim.join("x")})` : ""}`;
    $("#abLabelRight").classList.remove("dim-mode");
    $("#abEmpty").hidden = true;
    $("#modalRevert").disabled = false;
    $("#modalUpscale").textContent = "re-upscale";
    // There's an upscale to compare against -> show the compare slider.
    $("#abHandle").style.display = "";
  } else {
    $("#abDst").src = t.src_png_b64;
    $("#abLabelLeft").textContent = "source";
    $("#abLabelRight").textContent = "(not yet upscaled)";
    $("#abLabelRight").classList.add("dim-mode");
    $("#abEmpty").hidden = false;
    $("#modalRevert").disabled = true;
    $("#modalUpscale").textContent = "upscale this tile";
    // Nothing to compare yet (both sides are the source) -> hide the slider.
    $("#abHandle").style.display = "none";
  }

  // Restart any running modal anim, since underlying images may have changed
  const animEnabled = $("#modalAnimEnable").checked;
  stopCanvasAnim(animKeyRunner(fname, idx, "modal-src"));
  stopCanvasAnim(animKeyRunner(fname, idx, "modal-up"));
  $("#abSrcCanvas").hidden = true;
  $("#abDstCanvas").hidden = true;
  $("#abSrc").style.visibility = "";
  $("#abDst").style.visibility = "";
  if (animEnabled) {
    startModalAnimation(t);
  }

  const tiles = state.currentFile.tiles;
  const pos = tiles.findIndex((x) => x.index === idx);
  $("#modalPrev").disabled = pos <= 0;
  $("#modalNext").disabled = pos < 0 || pos >= tiles.length - 1;
}

function gotoModalSibling(delta) {
  if (state.modalTileIdx === null || !state.currentFile) return;
  const tiles = state.currentFile.tiles;
  const pos = tiles.findIndex((x) => x.index === state.modalTileIdx);
  const next = pos + delta;
  if (next < 0 || next >= tiles.length) return;
  openModal(tiles[next].index);
}

function setSliderPct(pct) {
  pct = clamp(pct, 0, 100);
  state.sliderPct = pct;
  $("#abHandle").style.left = pct + "%";
  $("#abHandle").setAttribute("aria-valuenow", String(Math.round(pct)));
  // hide LEFT pct% of upscaled, revealing source on the LEFT
  $("#abDst").style.clipPath = `inset(0 0 0 ${pct}%)`;
  $("#abDstCanvas").style.clipPath = `inset(0 0 0 ${pct}%)`;
  // sync to card slider
  if (state.modalTileIdx != null && state.currentFile) {
    const key = editKey(state.currentFile.name, state.modalTileIdx);
    state.cardSliderPcts[key] = Math.round(pct);
    const card = $(`#tileGrid .tile-card[data-idx="${state.modalTileIdx}"]`);
    if (card) applyCardSlider(card, pct);
  }
}

function resetSlider() { setSliderPct(50); }

function bindSlider() {
  const wrap = $("#abWrap");
  const handle = $("#abHandle");
  let dragging = false;

  const pctFromEvent = (e) => {
    const rect = wrap.getBoundingClientRect();
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    return ((cx - rect.left) / rect.width) * 100;
  };

  wrap.addEventListener("mousedown", (e) => {
    if ($("#modal").hidden) return;
    e.preventDefault();
    dragging = true;
    setSliderPct(pctFromEvent(e));
    handle.focus();
  });
  wrap.addEventListener("touchstart", (e) => {
    if ($("#modal").hidden) return;
    dragging = true;
    setSliderPct(pctFromEvent(e));
  }, { passive: true });

  document.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    e.preventDefault();
    setSliderPct(pctFromEvent(e));
  });
  document.addEventListener("touchmove", (e) => {
    if (!dragging) return;
    setSliderPct(pctFromEvent(e));
  }, { passive: true });

  const stop = () => { dragging = false; };
  document.addEventListener("mouseup", stop);
  document.addEventListener("touchend", stop);
  document.addEventListener("touchcancel", stop);

  handle.addEventListener("keydown", (e) => {
    let step = 0;
    if (e.key === "ArrowLeft") {
      step = e.shiftKey ? -CONFIG.MODAL_SLIDER_STEP_BIG : -CONFIG.MODAL_SLIDER_STEP;
    } else if (e.key === "ArrowRight") {
      step = e.shiftKey ?  CONFIG.MODAL_SLIDER_STEP_BIG :  CONFIG.MODAL_SLIDER_STEP;
    } else if (e.key === "Home")  { setSliderPct(0); e.preventDefault(); return; }
    else if (e.key === "End")   { setSliderPct(100); e.preventDefault(); return; }
    if (step) {
      setSliderPct(state.sliderPct + step);
      e.preventDefault();
    }
  });
}

// =====================================================================
// Modal animation controls
// =====================================================================
function syncModalAnimControls(t) {
  const fname = state.currentFile.name;
  const cfg = loadAnimCfg(fname, t.index, t.width, t.height);
  $("#modalAnimEnable").checked = !!cfg.enabled;
  const gridStr = `${cfg.frameGrid[0]}x${cfg.frameGrid[1]}`;
  const sel = $("#modalAnimGrid");
  if ([...sel.options].some(o => o.value === gridStr)) sel.value = gridStr;
  else sel.value = "custom";
  $("#modalAnimCustomCols").value = cfg.frameGrid[0];
  $("#modalAnimCustomRows").value = cfg.frameGrid[1];
  $("#modalAnimCustomWrap").hidden = sel.value !== "custom";
  $("#modalAnimFps").value = String(cfg.fps);
  $("#modalAnimOrder").value = cfg.frameOrder;
  $("#modalAnimInfo").textContent = `${t.width}x${t.height} src; ${cfg.frameGrid[0] * cfg.frameGrid[1]} frames at ${cfg.fps} fps`;
}

function readModalAnimCfg() {
  const t = state.modalTileIdx != null && state.currentFile
    ? state.currentFile.tiles.find(x => x.index === state.modalTileIdx)
    : null;
  if (!t) return null;
  const fname = state.currentFile.name;
  const cur = loadAnimCfg(fname, t.index, t.width, t.height);
  let grid = cur.frameGrid.slice();
  const v = $("#modalAnimGrid").value;
  if (v === "custom") {
    const c = clamp(parseInt($("#modalAnimCustomCols").value, 10) || 1, CONFIG.ANIM_GRID_MIN, CONFIG.ANIM_GRID_MAX);
    const r = clamp(parseInt($("#modalAnimCustomRows").value, 10) || 1, CONFIG.ANIM_GRID_MIN, CONFIG.ANIM_GRID_MAX);
    grid = [c, r];
  } else {
    const m = v.match(/^(\d+)x(\d+)$/);
    if (m) grid = [parseInt(m[1], 10), parseInt(m[2], 10)];
  }
  return {
    frameGrid: grid,
    frameOrder: $("#modalAnimOrder").value === "col-major" ? "col-major" : "row-major",
    fps: clamp(parseInt($("#modalAnimFps").value, 10) || CONFIG.ANIM_DEFAULT_FPS, CONFIG.ANIM_FPS_MIN, CONFIG.ANIM_FPS_MAX),
    enabled: $("#modalAnimEnable").checked,
  };
}

function persistModalAnim() {
  const t = state.modalTileIdx != null && state.currentFile
    ? state.currentFile.tiles.find(x => x.index === state.modalTileIdx)
    : null;
  if (!t) return;
  const cfg = readModalAnimCfg();
  saveAnimCfg(state.currentFile.name, t.index, cfg);
  $("#modalAnimInfo").textContent = `${t.width}x${t.height} src; ${cfg.frameGrid[0] * cfg.frameGrid[1]} frames at ${cfg.fps} fps`;
  // Mirror to the card so the per-card animation reflects the new cfg
  const card = $(`#tileGrid .tile-card[data-idx="${t.index}"]`);
  if (card) {
    // Restart card animation to pick up new cfg
    stopCardAnimation(card, t);
    if (cfg.enabled) startCardAnimation(card, t);
    // Update the per-card play button visual
    const pb = card.querySelector(".anim-bar button");
    if (pb) {
      pb.classList.toggle("playing", cfg.enabled);
      pb.textContent = cfg.enabled ? "\u25A0" : "\u25B6";
    }
  }
  // Restart modal anim
  const fname = state.currentFile.name;
  stopCanvasAnim(animKeyRunner(fname, t.index, "modal-src"));
  stopCanvasAnim(animKeyRunner(fname, t.index, "modal-up"));
  $("#abSrcCanvas").hidden = true;
  $("#abDstCanvas").hidden = true;
  $("#abSrc").style.visibility = "";
  $("#abDst").style.visibility = "";
  if (cfg.enabled) startModalAnimation(t);
}

function startModalAnimation(t) {
  const fname = state.currentFile.name;
  const cfg = readModalAnimCfg();
  if (!cfg.enabled) return;

  startCanvasAnim({
    runnerKey: animKeyRunner(fname, t.index, "modal-src"),
    canvas: $("#abSrcCanvas"),
    imgEl: $("#abSrc"),
    cfg,
    onCfgChange: () => readModalAnimCfg(),
  });

  const edit = state.tileEdits[editKey(fname, t.index)];
  if (edit) {
    startCanvasAnim({
      runnerKey: animKeyRunner(fname, t.index, "modal-up"),
      canvas: $("#abDstCanvas"),
      imgEl: $("#abDst"),
      cfg,
      onCfgChange: () => readModalAnimCfg(),
    });
  }
}

// =====================================================================
// Toolbar advanced settings
// =====================================================================
function applyFitMode() {
  const mode = $("#fitSel").value;
  state.fitMode = mode;
  $$(".tile-card").forEach((c) => {
    c.classList.remove("fit-contain", "fit-cover", "fit-native");
    c.classList.add(`fit-${mode}`);
  });
}

function applyZoom() {
  const z = parseFloat($("#zoomRange").value);
  state.zoom = z;
  $("#zoomVal").textContent = z.toFixed(1) + "\u00D7";
  document.documentElement.style.setProperty("--grid-zoom", String(z));
  // Force re-layout of grid track sizing
  const grid = $("#tileGrid");
  if (grid) grid.style.gridTemplateColumns = `repeat(auto-fill, minmax(${CONFIG.GRID_BASE_MIN_PX * z}px, 1fr))`;
}

function setupModelInfo() {
  const sel = $("#modelSel");
  const out = $("#modelInfo");
  const update = () => {
    const m = state.models.find(x => x.name === sel.value);
    if (m) {
      out.textContent = `${m.description} (native ${m.native_scale}\u00D7, max ${m.max_scale}\u00D7)`;
    } else {
      out.textContent = "";
    }
  };
  sel.addEventListener("change", update);
  update();
}

function buildScaleOptions() {
  const sel = $("#scaleSel");
  sel.innerHTML = "";
  for (const s of state.allowedScales) {
    const o = document.createElement("option");
    o.value = String(s);
    o.textContent = `${s}\u00D7${s > 4 ? " (cascade)" : ""}`;
    if (s === 4) o.selected = true;
    sel.appendChild(o);
  }
}

// =====================================================================
// Modal upscale-bar — clone of toolbar settings, bidirectionally synced.
// Lets users tweak model/scale/TTA/tile/native/gpu without leaving the modal.
// =====================================================================
function syncModalUpscaleBar() {
  // Mirror dropdowns from toolbar
  const tModel = $("#modelSel");
  const mModel = $("#modalModelSel");
  if (tModel && mModel) {
    mModel.innerHTML = "";
    for (const o of tModel.options) {
      const c = document.createElement("option");
      c.value = o.value; c.textContent = o.textContent; c.title = o.title || "";
      if (o.disabled) c.disabled = true;
      mModel.appendChild(c);
    }
    mModel.value = tModel.value;
    mModel.disabled = tModel.disabled;
  }
  const tScale = $("#scaleSel");
  const mScale = $("#modalScaleSel");
  if (tScale && mScale) {
    mScale.innerHTML = "";
    for (const o of tScale.options) {
      const c = document.createElement("option");
      c.value = o.value; c.textContent = o.textContent;
      mScale.appendChild(c);
    }
    mScale.value = tScale.value;
  }
  // Checkboxes / value selectors
  const pairs = [
    ["#keepNative", "#modalKeepNative", "checked"],
    ["#ttaToggle", "#modalTtaToggle", "checked"],
    ["#tileSizeSel", "#modalTileSizeSel", "value"],
    ["#gpuSel", "#modalGpuSel", "value"],
  ];
  for (const [tSel, mSel, prop] of pairs) {
    const t = $(tSel), m = $(mSel);
    if (t && m) m[prop] = t[prop];
  }
}

function setupModalUpscaleBarBinding() {
  const wire = (modalId, toolbarId, prop) => {
    const m = $(modalId), t = $(toolbarId);
    if (!m || !t) return;
    // modal -> toolbar
    m.addEventListener("change", () => {
      if (t[prop] !== m[prop]) {
        t[prop] = m[prop];
        t.dispatchEvent(new Event("change", { bubbles: true }));
      }
    });
    // toolbar -> modal
    t.addEventListener("change", () => {
      if (m[prop] !== t[prop]) m[prop] = t[prop];
    });
  };
  wire("#modalModelSel", "#modelSel", "value");
  wire("#modalScaleSel", "#scaleSel", "value");
  wire("#modalKeepNative", "#keepNative", "checked");
  wire("#modalTtaToggle", "#ttaToggle", "checked");
  wire("#modalTileSizeSel", "#tileSizeSel", "value");
  wire("#modalGpuSel", "#gpuSel", "value");
}

// =====================================================================
// Tile filter (per-file, parsed live)
//
// Accepted forms (any combination, comma-separated):
//   5            single index match
//   5-10         range
//   5,7,9        list
//   1024         dim equals 1024 on either axis
//   <256         dim less than 256 on smaller axis
//   >=512        dim ≥ 512 on smaller axis
//   fmt6         match XVR format byte
//   has-edit     only tiles with an upscale registered
//   no-edit      only tiles WITHOUT an upscale registered
// Free-text outside those tokens is ignored. Empty filter shows all.
// =====================================================================
function parseTileFilter(raw) {
  if (!raw || !raw.trim()) return null;
  const idxSet = new Set();
  let dimPred = null;
  let fmt = null;
  let hasEditFlag = null; // null|true|false
  let invalid = false;

  const parts = raw.split(/[,\s]+/).filter(Boolean);
  for (const tok of parts) {
    const t = tok.toLowerCase();
    let m;
    if (t === "has-edit" || t === "hasedit" || t === "edited") {
      hasEditFlag = true;
    } else if (t === "no-edit" || t === "noedit" || t === "unedited" || t === "clean") {
      hasEditFlag = false;
    } else if ((m = t.match(/^fmt(\d+)$/))) {
      fmt = parseInt(m[1], 10);
    } else if ((m = t.match(/^(\d+)-(\d+)$/))) {
      const a = parseInt(m[1], 10), b = parseInt(m[2], 10);
      const lo = Math.min(a, b), hi = Math.max(a, b);
      for (let i = lo; i <= hi; i++) idxSet.add(i);
    } else if ((m = t.match(/^(<=|>=|<|>)(\d+)$/))) {
      const op = m[1], v = parseInt(m[2], 10);
      const prior = dimPred;
      const f = (w, h) => {
        const m = Math.min(w, h);
        if (op === "<")  return m < v;
        if (op === "<=") return m <= v;
        if (op === ">")  return m > v;
        if (op === ">=") return m >= v;
        return false;
      };
      dimPred = prior ? ((w, h) => prior(w, h) || f(w, h)) : f;
    } else if ((m = t.match(/^(\d+)$/))) {
      // Could mean either "tile index N" or "dimension N". We accept both.
      // Tile index match is added if N <= 999, dim match always added.
      const n = parseInt(m[1], 10);
      idxSet.add(n);
      const prior = dimPred;
      const f = (w, h) => (w === n || h === n);
      dimPred = prior ? ((w, h) => prior(w, h) || f(w, h)) : f;
    } else {
      invalid = true;
    }
  }
  // If user only entered dim/fmt/edit constraints (no idx tokens), idxSet is empty
  // -> means "match all indices". We model that with idxSet=null.
  return {
    raw,
    idxSet: idxSet.size ? idxSet : null,
    dimPred,
    fmt,
    hasEditFlag,
    invalid,
  };
}

function tilePassesFilter(tile, filename, filter) {
  if (!filter) return true;
  // idx clause: if any idx tokens were given, the index OR a dim match must pass.
  // Note: ambiguous bare numbers added to BOTH idxSet AND dimPred.
  let idxOk = filter.idxSet == null ? true : filter.idxSet.has(tile.index);
  let dimOk = filter.dimPred == null ? true : filter.dimPred(tile.width, tile.height);
  // OR-merge ambiguous bare numbers: if BOTH idxSet and dimPred are set, either passes.
  if (filter.idxSet != null && filter.dimPred != null) {
    if (!idxOk && !dimOk) return false;
  } else {
    if (!idxOk) return false;
    if (!dimOk) return false;
  }
  if (filter.fmt != null && tile.fmt !== filter.fmt) return false;
  if (filter.hasEditFlag !== null) {
    const hasEd = !!state.tileEdits[editKey(filename, tile.index)];
    if (filter.hasEditFlag !== hasEd) return false;
  }
  return true;
}

function applyTileFilter() {
  const grid = $("#tileGrid");
  if (!grid || !state.currentFile) return;
  const filter = state.tileFilter;
  let visible = 0;
  for (const card of grid.querySelectorAll(".tile-card")) {
    const idx = parseInt(card.dataset.idx, 10);
    const t = state.currentFile.tiles.find((x) => x.index === idx);
    if (!t) continue;
    const ok = tilePassesFilter(t, state.currentFile.name, filter);
    card.classList.toggle("filter-hidden", !ok);
    card.style.display = ok ? "" : "none";
    if (ok) visible++;
  }
  const total = state.currentFile.tiles.length;
  const cnt = $("#tileFilterCount");
  if (cnt) cnt.textContent = filter ? `${visible}/${total}` : "";
  const inp = $("#tileFilterBox");
  if (inp) inp.classList.toggle("bad", !!(filter && filter.invalid));
}

// =====================================================================
// Multi-select (within the currently-open file)
// =====================================================================
function clearSelection() {
  state.selectedIndices.clear();
  state.selectionAnchor = null;
  refreshSelectionUI();
}

function refreshSelectionUI() {
  // Update card .selected class
  for (const card of document.querySelectorAll("#tileGrid .tile-card")) {
    const idx = parseInt(card.dataset.idx, 10);
    card.classList.toggle("selected", state.selectedIndices.has(idx));
  }
  // Update header pill + batch bar
  const n = state.selectedIndices.size;
  const pill = $("#selectionCount");
  if (pill) {
    if (n > 0) {
      pill.hidden = false;
      pill.textContent = `${n} tile${n === 1 ? "" : "s"} selected`;
    } else {
      pill.hidden = true;
    }
  }
  const bar = $("#batchBar");
  if (bar) bar.hidden = n === 0;
  const cnt = $("#batchCount");
  if (cnt) cnt.textContent = n > 0 ? `${n} selected` : "";
}

function toggleSelect(idx, { shift = false, ctrl = false } = {}) {
  if (!state.currentFile) return;
  const tiles = state.currentFile.tiles;
  if (shift && state.selectionAnchor != null) {
    // Range from anchor to idx in the *visible* (filtered+ordered) list.
    const visIndices = tiles
      .map((t) => t.index)
      .filter((i) => {
        const t = tiles.find((x) => x.index === i);
        return tilePassesFilter(t, state.currentFile.name, state.tileFilter);
      });
    const a = visIndices.indexOf(state.selectionAnchor);
    const b = visIndices.indexOf(idx);
    if (a >= 0 && b >= 0) {
      const lo = Math.min(a, b), hi = Math.max(a, b);
      for (let i = lo; i <= hi; i++) state.selectedIndices.add(visIndices[i]);
    } else {
      state.selectedIndices.add(idx);
    }
  } else if (ctrl) {
    if (state.selectedIndices.has(idx)) state.selectedIndices.delete(idx);
    else state.selectedIndices.add(idx);
    state.selectionAnchor = idx;
  } else {
    // Plain click: select only this. (Anchor is always reset to this.)
    state.selectedIndices.clear();
    state.selectedIndices.add(idx);
    state.selectionAnchor = idx;
  }
  refreshSelectionUI();
}

async function batchUpscaleSelected() {
  if (!state.currentFile || state.selectedIndices.size === 0) return;
  const indices = [...state.selectedIndices].sort((a, b) => a - b);
  const opts = currentUpscaleOpts();
  if (!opts.model) {
    setStatus("no upscale model available", "err", { sticky: true });
    return;
  }
  showProgress(0, indices.length);
  let ok = 0, fail = 0;
  for (let i = 0; i < indices.length; i++) {
    const idx = indices[i];
    const success = await upscaleTile(idx, opts);
    if (success) ok++; else fail++;
    showProgress(i + 1, indices.length);
  }
  hideProgress();
  if (fail > 0) {
    toast(`batch upscale: ${ok} ok, ${fail} failed`, "err");
  } else {
    toast(`batch upscale: ${plural(ok, "tile")}`, "ok");
  }
}

function batchRevertSelected() {
  if (!state.currentFile || state.selectedIndices.size === 0) return;
  const fname = state.currentFile.name;
  const indices = [...state.selectedIndices];
  const eligible = indices.filter((i) => state.tileEdits[editKey(fname, i)]);
  if (eligible.length === 0) {
    toast("no edited tiles in selection", "info");
    return;
  }
  if (!confirm(`Revert ${plural(eligible.length, "edited tile")} in selection?`)) return;
  for (const i of eligible) {
    delete state.tileEdits[editKey(fname, i)];
    delete state.cardSliderPcts[editKey(fname, i)];
    updateTileCard(i);
  }
  updateEditsCounter();
  renderFiles();
  applyTileFilter();
  toast(`reverted ${plural(eligible.length, "tile")}`, "ok");
}

// =====================================================================
// Card import (drag-and-drop or via 'import file' button)
//
// Posts multipart/form-data to /api/import_png/{filename}/{tile_index}.
// Server returns out_b64; we register the result in state.tileEdits as
// {source: "import"} so the UI shows the same has-up state as a normal
// upscale.
// =====================================================================
async function importPngForTile(idx, file, { keepNative = true } = {}) {
  if (!state.currentFile) return false;
  const t = state.currentFile.tiles.find((x) => x.index === idx);
  if (!t) return false;
  if (!file) return false;
  if (!/\.png$|^image\/png/i.test(file.name + " " + (file.type || ""))) {
    toast(`not a PNG: ${file.name}`, "err");
    return false;
  }
  const fname = state.currentFile.name;
  const key = editKey(fname, idx);
  if (state.busyTiles.has(key)) {
    toast(`tile ${idx} already busy`, "err");
    return false;
  }
  state.busyTiles.add(key);
  updateTileCard(idx);
  setStatus(`importing ${file.name} -> tile ${idx}...`, "busy", { sticky: true });
  try {
    const fd = new FormData();
    fd.append("image", file, file.name);
    fd.append("keep_native_dims", keepNative ? "true" : "false");
    const r = await api(`/api/import_png/${encodeURIComponent(fname)}/${idx}`, {
      method: "POST",
      body: fd,
    });
    state.tileEdits[key] = {
      src_b64: t.src_png_b64,
      up_b64: r.out_b64,
      model: "imported",
      scale: r.scale_factor || 1,
      tta: false,
      tile_size: null,
      gpu_id: null,
      native_dim: [t.width, t.height],
      out_dim: [r.out_w, r.out_h],
      source: "import",
      imported_filename: r.imported_filename || file.name,
    };
    toast(`imported ${file.name} as tile ${idx} (${r.scale_factor}x source)`, "ok");
    setStatus(`imported tile ${idx}`, "ok");
    return true;
  } catch (e) {
    toast(`import failed: ${e.message}`, "err", { ttl: 7000 });
    setStatus(`import failed: ${e.message}`, "err", { sticky: true });
    return false;
  } finally {
    state.busyTiles.delete(key);
    updateTileCard(idx);
    updateEditsCounter();
    renderFiles();
    applyTileFilter();
    if (state.modalTileIdx === idx) refreshModalImages();
  }
}

// =====================================================================
// Page-wide drag-drop (drag a PNG from desktop onto any tile card)
// =====================================================================
function setupDragDrop() {
  const overlay = $("#dropOverlay");
  let dragDepth = 0;

  const isFileDrag = (e) =>
    e.dataTransfer && [...(e.dataTransfer.types || [])].includes("Files");

  document.addEventListener("dragenter", (e) => {
    if (!isFileDrag(e)) return;
    e.preventDefault();
    dragDepth++;
    if (overlay) overlay.hidden = false;
  });
  document.addEventListener("dragover", (e) => {
    if (!isFileDrag(e)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    // Highlight the card under the cursor
    const card = e.target && e.target.closest ? e.target.closest(".tile-card") : null;
    document.querySelectorAll(".tile-card.dropping").forEach((c) => {
      if (c !== card) c.classList.remove("dropping");
    });
    if (card) card.classList.add("dropping");
  });
  document.addEventListener("dragleave", (e) => {
    if (!isFileDrag(e)) return;
    dragDepth--;
    if (dragDepth <= 0) {
      dragDepth = 0;
      if (overlay) overlay.hidden = true;
      document.querySelectorAll(".tile-card.dropping").forEach((c) => c.classList.remove("dropping"));
    }
  });
  document.addEventListener("drop", async (e) => {
    if (!isFileDrag(e)) return;
    e.preventDefault();
    dragDepth = 0;
    if (overlay) overlay.hidden = true;
    document.querySelectorAll(".tile-card.dropping").forEach((c) => c.classList.remove("dropping"));

    const card = e.target && e.target.closest ? e.target.closest(".tile-card") : null;
    if (!card) {
      toast("drop a PNG onto a tile card to import it", "info");
      return;
    }
    const idx = parseInt(card.dataset.idx, 10);
    const files = [...(e.dataTransfer.files || [])];
    if (files.length === 0) return;
    if (files.length > 1) {
      toast(`only the first PNG (${files[0].name}) is imported per drop`, "info");
    }
    await importPngForTile(idx, files[0], { keepNative: $("#keepNative").checked });
  });
}

// File-picker fallback: the small "import file" button on each card opens a
// hidden <input type=file> already in the DOM, scoped to the clicked tile.
function setupCardImportPicker() {
  const inp = $("#cardImportInput");
  if (!inp) return;
  inp.addEventListener("change", async () => {
    const idx = state.importTargetTileIdx;
    state.importTargetTileIdx = null;
    if (idx == null) return;
    const f = inp.files && inp.files[0];
    if (f) await importPngForTile(idx, f, { keepNative: $("#keepNative").checked });
    inp.value = ""; // reset so re-selecting same file fires change
  });
}

// =====================================================================
// Deploy diff modal — pre-deploy summary instead of bare confirm()
// =====================================================================
async function showDeployDiff() {
  if (!state.currentFile) return;
  const fname = state.currentFile.name;
  const editedIndices = Object.keys(state.tileEdits)
    .filter((k) => k.startsWith(fname + ":"))
    .map((k) => parseInt(k.slice(fname.length + 1), 10))
    .filter((n) => !Number.isNaN(n));
  if (editedIndices.length === 0) {
    toast("nothing to repack — upscale at least one tile first", "err");
    return;
  }
  const modal = $("#deployModal");
  const body = $("#deployBody");
  body.innerHTML = `<div class="dim">computing diff...</div>`;
  modal.hidden = false;
  $("#deployConfirm").disabled = true;
  let diff = null;
  try {
    diff = await api("/api/repack_diff", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: fname, edited_indices: editedIndices }),
    });
  } catch (e) {
    body.innerHTML = `<div class="status err sticky">diff failed: ${escapeHtml(e.message)}</div>`;
    return;
  }
  const sizeKb = diff.file_size_bytes ? (diff.file_size_bytes / 1024).toFixed(1) : "?";
  const changedRows = diff.changed_indices.map((i) => {
    const t = state.currentFile.tiles.find((x) => x.index === i);
    const edit = state.tileEdits[editKey(fname, i)];
    const dims = t ? `${t.width}x${t.height}` : "";
    const src = edit && edit.source === "import" ? "imported" : (edit ? `${edit.model} ${edit.scale}x${edit.tta ? " TTA" : ""}` : "");
    return `<div class="row"><span class="idx">tile ${String(i).padStart(2, "0")}</span><span class="meta">${escapeHtml(dims)} — ${escapeHtml(src)}</span></div>`;
  }).join("");
  const unchangedRows = diff.unchanged_indices.length === 0
    ? `<div class="row"><span class="meta dim">— none —</span></div>`
    : diff.unchanged_indices.map((i) => {
        const t = state.currentFile.tiles.find((x) => x.index === i);
        const dims = t ? `${t.width}x${t.height}` : "";
        return `<div class="row"><span class="idx">tile ${String(i).padStart(2, "0")}</span><span class="meta dim">${escapeHtml(dims)} — unchanged</span></div>`;
      }).join("");
  const unknownNote = (diff.unknown_indices && diff.unknown_indices.length)
    ? `<div class="deploy-warning">${plural(diff.unknown_indices.length, "edit")} reference tile indices not present in this file: ${diff.unknown_indices.join(", ")}. Those will be ignored.</div>`
    : "";
  body.innerHTML = `
    <dl class="deploy-summary">
      <dt>file</dt><dd>${escapeHtml(diff.filename)}</dd>
      <dt>type</dt><dd>${diff.is_prs ? "PRS (compressed)" : "XVM (raw)"}</dd>
      <dt>tiles</dt><dd>${diff.tile_count} total — <strong style="color:#4ade80">${diff.changed_indices.length} changed</strong>, ${diff.unchanged_indices.length} unchanged</dd>
      <dt>size</dt><dd>${sizeKb} KB on disk</dd>
      <dt>backup</dt><dd>${escapeHtml(diff.backup_name_preview)}</dd>
    </dl>
    <div style="font-size:11px;color:#7a8290;margin-bottom:6px;">changed:</div>
    <div class="changed-list">${changedRows || `<div class="row"><span class="meta dim">— none —</span></div>`}</div>
    <div style="font-size:11px;color:#7a8290;margin:10px 0 6px 0;">unchanged:</div>
    <div class="changed-list">${unchangedRows}</div>
    ${unknownNote}
    <div class="deploy-warning">deploying overwrites the live <code>${escapeHtml(diff.filename)}</code>. The backup will be written first; you can restore via the API.</div>
  `;
  $("#deployConfirm").disabled = false;
}

function hideDeployDiff() {
  $("#deployModal").hidden = true;
  // 2026-04-25 (regression-fix-modal-vs-viewport): drop the allow-class
  // so the deploy-diff perspective can relocate #deployBody without the
  // modal wrapper showing alongside.
  document.body.classList.remove("pso-modal-allow-deploy");
}

// =====================================================================
// Deploy-to-game (dev mirror -> live PSOBB.IO data dir)
//
// This is the second deploy flow on top of the per-file repack flow:
//   - repack & deploy (#deployModal) writes a single rebuilt file into
//     the active DATA_DIR (which IS the dev mirror)
//   - deploy-to-game (#promoteModal) takes whatever has piled up in dev
//     and copies named files to the user's playable PSOBB.IO install
//
// Endpoints:
//   GET  /api/deploy/config   -> {dev_dir, live_dir, ...}
//   GET  /api/deploy/diff     -> {changed:[...], dev_only:[...], live_only:[...]}
//   POST /api/deploy/promote  -> {ok_count, fail_count, results:[...]}
// =====================================================================

// Last fetched diff payload (held while the modal is open). null when the
// modal is closed or the diff request is in flight.
let _lastDeployDiff = null;
let _promoteInFlight = false;

async function showDeployToGame() {
  const modal = $("#promoteModal");
  const body = $("#promoteBody");
  modal.hidden = false;
  $("#promoteConfirm").disabled = true;
  body.innerHTML = `<div class="dim">computing dev/live diff...</div>`;

  let cfg = null;
  let diff = null;
  try {
    cfg = await api("/api/deploy/config");
  } catch (e) {
    body.innerHTML = `<div class="status err sticky">failed to read deploy config: ${escapeHtml(e.message)}</div>`;
    return;
  }
  try {
    diff = await api("/api/deploy/diff");
  } catch (e) {
    body.innerHTML = `<div class="status err sticky">diff failed: ${escapeHtml(e.message)}</div>`;
    return;
  }
  _lastDeployDiff = diff;
  renderDeployToGame(cfg, diff);
}

function hideDeployToGame() {
  $("#promoteModal").hidden = true;
  _lastDeployDiff = null;
}

function _promoteRowHtml(cls, name, meta, checked, disabled) {
  // We render checkboxes with data-name so the click handler can pick them
  // up via delegation. CSS class drives the colour swatch.
  return (
    `<div class="row ${cls}">` +
    `<label>` +
    `<input type="checkbox" class="promote-check" data-name="${escapeHtml(name)}"${checked ? " checked" : ""}${disabled ? " disabled" : ""} />` +
    `<span class="name">${escapeHtml(name)}</span>` +
    `</label>` +
    `<span class="meta">${escapeHtml(meta)}</span>` +
    `</div>`
  );
}

function _fmtKb(bytes) {
  if (bytes == null || bytes < 0) return "?";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function renderDeployToGame(cfg, diff) {
  const body = $("#promoteBody");
  const changedRows = (diff.changed || []).map((e) => {
    const dKb = _fmtKb(e.dev_size);
    const lKb = _fmtKb(e.live_size);
    const meta = `dev ${dKb} -> live ${lKb}`;
    // Default-checked: changed files
    return _promoteRowHtml("changed", e.name, meta, true, false);
  }).join("") || `<div class="empty">no changed files</div>`;

  const devOnlyRows = (diff.dev_only || []).map((e) => {
    const meta = `${_fmtKb(e.dev_size)} (new)`;
    // Default-UNCHECKED: dev-only (these are NEW files not in the stock install,
    // user might not want to push them).
    return _promoteRowHtml("dev-only", e.name, meta, false, false);
  }).join("") || `<div class="empty">no dev-only files</div>`;

  const liveOnlyRows = (diff.live_only || []).map((e) => {
    const meta = `${_fmtKb(e.live_size)} (live-only — informational)`;
    // live_only files cannot be promoted (we have no dev source for them).
    return _promoteRowHtml("live-only", e.name, meta, false, true);
  }).join("") || `<div class="empty">no live-only files</div>`;

  const changedCount = (diff.changed || []).length;
  const devOnlyCount = (diff.dev_only || []).length;
  const liveOnlyCount = (diff.live_only || []).length;

  body.innerHTML = `
    <div class="promote-source-arrow">
      <span class="lbl">DEV:</span>
      <span class="dev-path">${escapeHtml(cfg.dev_dir)}</span>
      <span class="arrow">&rarr;</span>
      <span class="lbl">LIVE:</span>
      <span class="live-path">${escapeHtml(cfg.live_dir)}</span>
    </div>
    <div class="promote-section">
      <div class="promote-section-title">
        changed <span class="count">${changedCount}</span>
        <span class="dim" style="font-weight:normal">— same name, different bytes</span>
      </div>
      <div class="promote-list">${changedRows}</div>
    </div>
    <div class="promote-section">
      <div class="promote-section-title">
        dev-only <span class="count">${devOnlyCount}</span>
        <span class="dim" style="font-weight:normal">— in dev, missing from live (NEW files)</span>
      </div>
      <div class="promote-list">${devOnlyRows}</div>
    </div>
    <div class="promote-section">
      <div class="promote-section-title">
        live-only <span class="count">${liveOnlyCount}</span>
        <span class="dim" style="font-weight:normal">— in live but not in dev (informational; cannot promote)</span>
      </div>
      <div class="promote-list">${liveOnlyRows}</div>
    </div>
    <div class="promote-warning">
      promoting writes directly to the user's playable game install. A timestamped backup
      (<code>&lt;file&gt;.pre_promote_&lt;TS&gt;</code>) is written into <code>${escapeHtml(cfg.live_dir)}</code>
      before each overwrite (toggle below to disable).
    </div>
  `;

  // Wire up the per-checkbox change so we can keep the confirm button in sync.
  $$("#promoteBody .promote-check").forEach((cb) => {
    cb.addEventListener("change", updatePromoteConfirmState);
  });
  updatePromoteConfirmState();
}

function _selectedPromoteNames() {
  return $$("#promoteBody .promote-check:checked:not(:disabled)").map((cb) => cb.dataset.name);
}

function updatePromoteConfirmState() {
  const names = _selectedPromoteNames();
  const btn = $("#promoteConfirm");
  if (!btn) return;
  btn.disabled = names.length === 0 || _promoteInFlight;
  btn.textContent = names.length > 0
    ? `promote ${names.length} file${names.length === 1 ? "" : "s"}`
    : "promote selected";
}

function promoteSelectAllChanged() {
  // Toggle all .changed rows on; leave dev-only/live-only as user set them.
  $$("#promoteBody .row.changed .promote-check").forEach((cb) => { cb.checked = true; });
  updatePromoteConfirmState();
}

function promoteSelectNone() {
  $$("#promoteBody .promote-check").forEach((cb) => { if (!cb.disabled) cb.checked = false; });
  updatePromoteConfirmState();
}

async function doPromote() {
  if (_promoteInFlight) return;
  const names = _selectedPromoteNames();
  if (!names.length) {
    toast("nothing selected to promote", "err");
    return;
  }
  const createBackup = !!$("#promoteCreateBackup").checked;
  _promoteInFlight = true;
  updatePromoteConfirmState();
  setStatus(`promoting ${plural(names.length, "file")} to live install...`, "busy", { sticky: true });
  let r;
  try {
    r = await api("/api/deploy/promote", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ files: names, create_backup: createBackup }),
    });
  } catch (e) {
    setStatus(`promote failed: ${e.message}`, "err", { sticky: true });
    toast(`promote failed: ${e.message}`, "err", { ttl: 7000 });
    _promoteInFlight = false;
    updatePromoteConfirmState();
    return;
  }

  // Render per-file results inline; toast a one-line summary.
  const ok = Number(r.ok_count) || 0;
  const fail = Number(r.fail_count) || 0;
  if (fail === 0) {
    toast(`promoted ${ok} file${ok === 1 ? "" : "s"} to live game install`, "ok");
    setStatus(`promoted ${plural(ok, "file")} to live`, "ok");
  } else {
    toast(`promoted ${ok} ok, ${fail} failed — see modal`, "err", { ttl: 7000 });
    setStatus(`promote: ${ok} ok, ${fail} failed`, "err", { sticky: true });
  }

  // Replace the modal body with the per-file outcome list so the user
  // sees what happened. They can close manually or run another diff.
  const rows = (r.results || []).map((row) => {
    const cls = row.ok ? "ok" : "fail";
    const sign = row.ok ? "\u2714" : "\u2718";
    const tail = row.ok
      ? (row.backup_name ? `backup: ${row.backup_name}` : "no backup")
      : (row.error || "unknown error");
    return (
      `<div class="row ${cls}">` +
      `<span class="meta">${sign}</span>` +
      `<span class="name">${escapeHtml(row.name)}</span>` +
      `<span class="meta">${escapeHtml(tail)}</span>` +
      `</div>`
    );
  }).join("");
  const body = $("#promoteBody");
  body.innerHTML = `
    <div class="promote-source-arrow">
      <span class="lbl">DEV:</span>
      <span class="dev-path">${escapeHtml(r.dev_dir || "")}</span>
      <span class="arrow">&rarr;</span>
      <span class="lbl">LIVE:</span>
      <span class="live-path">${escapeHtml(r.live_dir || "")}</span>
    </div>
    <div class="promote-section">
      <div class="promote-section-title">
        results
        <span class="count">${ok} ok</span>
        ${fail > 0 ? `<span class="count" style="color:#f87171;background:#2c1518;border-color:#ef444455">${fail} failed</span>` : ""}
      </div>
      <div class="promote-list promote-result-list">${rows || `<div class="empty">no results</div>`}</div>
    </div>
  `;
  $("#promoteConfirm").disabled = true;
  $("#promoteSelectChanged").disabled = true;
  $("#promoteSelectNone").disabled = true;
  _promoteInFlight = false;
}

// =====================================================================
// Session export / import
//
// Schema (v1):
// {
//   schema: "psobb-editor-session",
//   version: 1,
//   exported_at: <unix-seconds>,
//   filename_hint: "human note",
//   settings: { model, scale, keepNative, tta, tileSize, gpuId, fitMode, zoom },
//   edits: {
//     "<filename>:<tileIndex>": {
//       src_b64?: string  (omitted to keep size down — recoverable from the
//                          on-disk file by re-extracting),
//       up_b64: string,
//       model, scale, tta, tile_size, gpu_id, native_dim, out_dim, source,
//       imported_filename?
//     }, ...
//   }
// }
// =====================================================================
function buildSessionBlob() {
  const settings = currentUpscaleOpts();
  // strip src_b64 from each edit; src is recoverable cheaply by re-extracting
  // on the receiving end. up_b64 is the load-bearing payload.
  const edits = {};
  for (const [k, e] of Object.entries(state.tileEdits)) {
    edits[k] = {
      up_b64: e.up_b64,
      model: e.model,
      scale: e.scale,
      tta: !!e.tta,
      tile_size: e.tile_size != null ? e.tile_size : null,
      gpu_id: e.gpu_id != null ? e.gpu_id : null,
      native_dim: e.native_dim || null,
      out_dim: e.out_dim || null,
      source: e.source || "upscaler",
      imported_filename: e.imported_filename || null,
    };
  }
  return {
    schema: "psobb-editor-session",
    version: 1,
    exported_at: Math.floor(Date.now() / 1000),
    filename_hint: state.currentFile ? state.currentFile.name : "",
    settings: {
      model: settings.model,
      scale: settings.scale,
      keepNative: settings.keepNative,
      tta: settings.tta,
      tileSize: settings.tileSize,
      gpuId: settings.gpuId,
      fitMode: state.fitMode,
      zoom: state.zoom,
    },
    edits,
  };
}

function exportSession() {
  const n = Object.keys(state.tileEdits).length;
  if (n === 0) {
    toast("nothing to export — no edits in memory", "err");
    return;
  }
  const blob = buildSessionBlob();
  const text = JSON.stringify(blob, null, 2);
  const b = new Blob([text], { type: "application/json" });
  const a = document.createElement("a");
  const ts = new Date(blob.exported_at * 1000).toISOString().replace(/[:.]/g, "-").slice(0, 19);
  a.href = URL.createObjectURL(b);
  a.download = `pso-editor-session_${ts}.json`;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    URL.revokeObjectURL(a.href);
    if (a.parentNode) a.parentNode.removeChild(a);
  }, 200);
  toast(`exported ${plural(n, "edit")}`, "ok");
}

async function importSession(file) {
  if (!file) return;
  let parsed;
  try {
    const text = await file.text();
    parsed = JSON.parse(text);
  } catch (e) {
    toast(`invalid JSON: ${e.message}`, "err");
    return;
  }
  if (!parsed || parsed.schema !== "psobb-editor-session") {
    toast("not a PSOBB editor session file", "err");
    return;
  }
  if (typeof parsed.version !== "number" || parsed.version > 1) {
    toast(`unsupported session version: ${parsed.version}`, "err");
    return;
  }
  const newEdits = parsed.edits || {};
  const incomingCount = Object.keys(newEdits).length;
  if (incomingCount === 0) {
    toast("session file has no edits", "info");
    return;
  }
  const existing = Object.keys(state.tileEdits).length;
  if (existing > 0) {
    if (!confirm(`Replace ${plural(existing, "pending edit")} with ${incomingCount} from session file?\n\n(Cancel to keep current edits.)`)) {
      return;
    }
  }
  // Adopt edits. We restore src_b64 lazily from current cache when we open the
  // file; for now leave it absent — the card-renderer reads tile.src_png_b64
  // from state.currentFile, not from edit.src_b64.
  state.tileEdits = {};
  state.cardSliderPcts = {};
  for (const [k, v] of Object.entries(newEdits)) {
    if (!v || typeof v !== "object" || !v.up_b64) continue;
    state.tileEdits[k] = {
      src_b64: v.src_b64 || "",
      up_b64: v.up_b64,
      model: v.model || "imported",
      scale: typeof v.scale === "number" ? v.scale : 1,
      tta: !!v.tta,
      tile_size: v.tile_size != null ? v.tile_size : null,
      gpu_id: v.gpu_id != null ? v.gpu_id : null,
      native_dim: Array.isArray(v.native_dim) ? v.native_dim : null,
      out_dim: Array.isArray(v.out_dim) ? v.out_dim : null,
      source: v.source || "session",
      imported_filename: v.imported_filename || null,
    };
  }
  // Restore settings (best-effort)
  try {
    const s = parsed.settings || {};
    if (s.model && [...$("#modelSel").options].some((o) => o.value === s.model)) {
      $("#modelSel").value = s.model;
      $("#modelSel").dispatchEvent(new Event("change", { bubbles: true }));
    }
    if (s.scale && [...$("#scaleSel").options].some((o) => Number(o.value) === Number(s.scale))) {
      $("#scaleSel").value = String(s.scale);
      $("#scaleSel").dispatchEvent(new Event("change", { bubbles: true }));
    }
    if (typeof s.keepNative === "boolean") $("#keepNative").checked = s.keepNative;
    if (typeof s.tta === "boolean") $("#ttaToggle").checked = s.tta;
    if (s.tileSize === null || s.tileSize === undefined) $("#tileSizeSel").value = "auto";
    else $("#tileSizeSel").value = String(s.tileSize);
    if (s.gpuId === null || s.gpuId === undefined) $("#gpuSel").value = "auto";
    else $("#gpuSel").value = String(s.gpuId);
    if (s.fitMode) $("#fitSel").value = s.fitMode;
    if (typeof s.zoom === "number") {
      $("#zoomRange").value = String(s.zoom);
      applyZoom();
    }
    applyFitMode();
    syncModalUpscaleBar();
  } catch {}

  // Refresh visible UI
  if (state.currentFile) {
    renderTileGrid();
    applyTileFilter();
  }
  updateEditsCounter();
  renderFiles();
  toast(`imported ${plural(incomingCount, "edit")} from session`, "ok");
}

// =====================================================================
// Atlas mode (composite editing)
//
// When the user opens a file that the backend says has a known atlas
// layout (server.py / atlas_layouts.py), the tile grid is hidden and
// replaced with a single composite canvas:
//
//   - GET  /api/atlas/{filename}   -> assemble + return composite_b64
//   - POST /api/atlas_upscale      -> upscale composite, slice -> per-tile
//   - POST /api/atlas_import       -> accept user-supplied composite PNG
//
// Each per-tile slice is registered in state.tileEdits exactly the same
// way a normal /api/upscale or /api/import_png response would be, so the
// existing repack pipeline picks it up unchanged.
// =====================================================================

// Cache of filenames that actually have an atlas layout (fetched once
// from /api/atlas_layouts). Lets refreshAtlasAvailability() skip probing
// /api/atlas/<f> for files with no layout — which 404'd on every texture
// open and spammed the console. null = not yet fetched.
let _atlasKnownSet = null;
let _atlasKnownPromise = null;
async function _getAtlasKnownSet() {
  if (_atlasKnownSet) return _atlasKnownSet;
  if (_atlasKnownPromise) return _atlasKnownPromise;
  _atlasKnownPromise = (async () => {
    try {
      const data = await api("/api/atlas_layouts");
      _atlasKnownSet = new Set((data && data.filenames) || []);
    } catch (_e) {
      // Endpoint missing/old server — fall back to an empty set so we
      // simply never probe (no atlas toggle, but also no 404 spam).
      _atlasKnownSet = new Set();
    }
    return _atlasKnownSet;
  })();
  return _atlasKnownPromise;
}

async function refreshAtlasAvailability() {
  // Probe /api/atlas/{filename} for the current file. 200 -> show the toggle
  // (default ON), 404 -> hide it (and force atlasMode off).
  const wrap = $("#atlasModeWrap");
  if (!wrap) return;
  if (!state.currentFile) {
    wrap.hidden = true;
    state.atlasMode = false;
    state.atlasState = null;
    refreshAtlasView();
    return;
  }
  const fname = state.currentFile.name;
  // Skip the probe entirely for files with no known atlas layout — this is
  // the common case (only a handful of UI splash files have layouts) and
  // probing them returns 404, which the browser logs to the console even
  // though we catch it. Consult the cached known-atlas set first.
  const known = await _getAtlasKnownSet();
  if (known && !known.has(fname)) {
    wrap.hidden = true;
    state.atlasMode = false;
    state.atlasState = null;
    if ($("#atlasModeToggle")) $("#atlasModeToggle").checked = false;
    wrap.classList.remove("atlas-active");
    refreshAtlasView();
    return;
  }
  try {
    const data = await api(`/api/atlas/${encodeURIComponent(fname)}`);
    state.atlasState = {
      filename: fname,
      composite_w: data.composite_w,
      composite_h: data.composite_h,
      placements: data.placements || [],
      skip_tiles: data.skip_tiles || [],
      src_b64: data.composite_b64,
      up_b64: null,
      meta: { kind: data.kind, source: data.source },
    };
    wrap.hidden = false;
    // Default ON when a layout exists, unless the user explicitly toggled off.
    if (state.atlasMode === false && !state._atlasUserChose) {
      state.atlasMode = true;
    }
    $("#atlasModeToggle").checked = !!state.atlasMode;
    wrap.classList.toggle("atlas-active", !!state.atlasMode);
  } catch (e) {
    // 404 = no layout, anything else = unexpected. Treat both as "no atlas".
    wrap.hidden = true;
    state.atlasMode = false;
    state.atlasState = null;
    if ($("#atlasModeToggle")) $("#atlasModeToggle").checked = false;
    wrap.classList.remove("atlas-active");
  }
  refreshAtlasView();
}

function _firstAtlasEditUpB64() {
  // If every placement has a per-tile edit registered, return them so we can
  // stitch a composite preview as the "upscaled" layer. Returning null means
  // "show source on both sides" (incomplete coverage would be misleading).
  const at = state.atlasState;
  if (!at || !at.placements) return null;
  const fname = at.filename;
  const ups = [];
  for (const p of at.placements) {
    const e = state.tileEdits[editKey(fname, p.tile_index)];
    if (!e || !e.up_b64) return null;
    ups.push({ p, e });
  }
  return ups;
}

async function buildAtlasUpComposite() {
  // Stitch per-tile up_b64 images into composite_w x composite_h, mirroring
  // _build_composite_image on the server.
  const ups = _firstAtlasEditUpB64();
  if (!ups) {
    if (state.atlasState) state.atlasState.up_b64 = null;
    return null;
  }
  const at = state.atlasState;
  const canvas = document.createElement("canvas");
  canvas.width = at.composite_w;
  canvas.height = at.composite_h;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  await Promise.all(ups.map(({ p, e }) => new Promise((res) => {
    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, p.x, p.y, p.w, p.h);
      res();
    };
    img.onerror = res;  // best-effort
    img.src = e.up_b64;
  })));
  const url = canvas.toDataURL("image/png");
  at.up_b64 = url;
  return url;
}

function _setAtlasViewVisible(showAtlas) {
  const v = $("#atlasView");
  const g = $("#tileGrid");
  if (!v || !g) return;
  v.hidden = !showAtlas;
  g.style.display = showAtlas ? "none" : "";
  // The "upscale composite" / "import composite" buttons replace
  // "upscale all" while atlas mode is active.
  const ua = $("#btnUpscaleAll");
  const au = $("#btnAtlasUpscale");
  const ai = $("#btnAtlasImport");
  if (au) au.hidden = !showAtlas;
  if (ai) ai.hidden = !showAtlas;
  if (ua) ua.hidden = !!showAtlas;
}

async function refreshAtlasView() {
  const at = state.atlasState;
  const wrap = $("#atlasModeWrap");
  if (wrap) wrap.classList.toggle("atlas-active", !!state.atlasMode && !!at);
  if (!state.atlasMode || !at) {
    _setAtlasViewVisible(false);
    return;
  }
  _setAtlasViewVisible(true);
  $("#atlasMetaText").textContent =
    `composite ${at.composite_w}x${at.composite_h} - ` +
    `${plural(at.placements.length, "placed tile")}, ${at.skip_tiles.length} skipped`;
  $("#atlasWrap").style.setProperty("--atlas-aspect", `${at.composite_w}/${at.composite_h}`);
  $("#atlasSrc").src = at.src_b64;

  const composite = await buildAtlasUpComposite();
  if (composite) {
    $("#atlasDst").src = composite;
    $("#atlasEmpty").hidden = true;
    $("#atlasLabelRight").classList.remove("dim-mode");
    $("#atlasLabelRight").textContent = "upscaled (composite)";
  } else {
    $("#atlasDst").src = at.src_b64;
    $("#atlasEmpty").hidden = false;
    $("#atlasLabelRight").classList.add("dim-mode");
    $("#atlasLabelRight").textContent = "(no composite yet)";
  }
  setAtlasSliderPct(state.atlasSliderPct);
  renderAtlasStrip();
}

function renderAtlasStrip() {
  const at = state.atlasState;
  const strip = $("#atlasStrip");
  if (!strip) return;
  strip.innerHTML = "";
  if (!at || !state.currentFile) return;
  const fname = at.filename;
  const placedIdx = new Set(at.placements.map((p) => p.tile_index));

  for (const p of at.placements) {
    const t = state.currentFile.tiles.find((x) => x.index === p.tile_index);
    const edit = state.tileEdits[editKey(fname, p.tile_index)];
    const card = document.createElement("div");
    card.className = "strip-card";
    if (edit) card.classList.add("has-up");
    card.title = `tile ${p.tile_index}: ${p.w}x${p.h} @ (${p.x},${p.y}) on composite`;
    const pic = document.createElement("div");
    pic.className = "strip-pic";
    const img = document.createElement("img");
    img.src = edit ? edit.up_b64 : (t ? t.src_png_b64 : "");
    img.alt = `tile ${p.tile_index}`;
    pic.appendChild(img);
    card.appendChild(pic);
    const meta = document.createElement("div");
    meta.className = "strip-meta";
    meta.innerHTML =
      `<strong>tile ${String(p.tile_index).padStart(2, "0")}</strong>` +
      `<div>${t ? t.width + "x" + t.height : "?"}${edit ? ' <span class="has-up-tag">edited</span>' : ""}</div>`;
    card.appendChild(meta);
    card.onclick = () => openModal(p.tile_index);
    strip.appendChild(card);
  }
  for (const idx of at.skip_tiles) {
    if (placedIdx.has(idx)) continue;  // defensive
    const t = state.currentFile.tiles.find((x) => x.index === idx);
    const card = document.createElement("div");
    card.className = "strip-card skipped";
    card.title = `tile ${idx} is not part of the atlas (e.g. pillar fill)`;
    const pic = document.createElement("div");
    pic.className = "strip-pic";
    const img = document.createElement("img");
    img.src = t ? t.src_png_b64 : "";
    img.alt = `tile ${idx}`;
    pic.appendChild(img);
    card.appendChild(pic);
    const meta = document.createElement("div");
    meta.className = "strip-meta";
    meta.innerHTML =
      `<strong>tile ${String(idx).padStart(2, "0")}</strong>` +
      `<div>${t ? t.width + "x" + t.height : "?"} <span class="skip-tag">skipped</span></div>`;
    card.appendChild(meta);
    card.style.cursor = "zoom-in";
    card.onclick = () => openModal(idx);
    strip.appendChild(card);
  }
}

function setAtlasSliderPct(pct) {
  pct = clamp(pct, 0, 100);
  state.atlasSliderPct = pct;
  const handle = $("#atlasHandle");
  const dst = $("#atlasDst");
  if (handle) {
    handle.style.left = pct + "%";
    handle.setAttribute("aria-valuenow", String(Math.round(pct)));
  }
  if (dst) dst.style.clipPath = `inset(0 0 0 ${pct}%)`;
}

function bindAtlasSlider() {
  const wrap = $("#atlasWrap");
  const handle = $("#atlasHandle");
  if (!wrap || !handle) return;
  let dragging = false;
  const pctFromEvent = (e) => {
    const rect = wrap.getBoundingClientRect();
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    return ((cx - rect.left) / rect.width) * 100;
  };
  wrap.addEventListener("mousedown", (e) => {
    if ($("#atlasView").hidden) return;
    e.preventDefault();
    dragging = true;
    setAtlasSliderPct(pctFromEvent(e));
    handle.focus();
  });
  wrap.addEventListener("touchstart", (e) => {
    if ($("#atlasView").hidden) return;
    dragging = true;
    setAtlasSliderPct(pctFromEvent(e));
  }, { passive: true });
  document.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    e.preventDefault();
    setAtlasSliderPct(pctFromEvent(e));
  });
  document.addEventListener("touchmove", (e) => {
    if (!dragging) return;
    setAtlasSliderPct(pctFromEvent(e));
  }, { passive: true });
  const stop = () => { dragging = false; };
  document.addEventListener("mouseup", stop);
  document.addEventListener("touchend", stop);
  document.addEventListener("touchcancel", stop);
  handle.addEventListener("keydown", (e) => {
    let step = 0;
    if (e.key === "ArrowLeft")  step = e.shiftKey ? -10 : -2;
    else if (e.key === "ArrowRight") step = e.shiftKey ?  10 :  2;
    else if (e.key === "Home")  { setAtlasSliderPct(0); e.preventDefault(); return; }
    else if (e.key === "End")   { setAtlasSliderPct(100); e.preventDefault(); return; }
    if (step) {
      setAtlasSliderPct(state.atlasSliderPct + step);
      e.preventDefault();
    }
  });
}

function _registerAtlasResultEdits(result, sourceLabel) {
  const fname = result.filename;
  for (const it of result.tiles) {
    const t = state.currentFile && state.currentFile.tiles.find(x => x.index === it.tile_index);
    const key = editKey(fname, it.tile_index);
    state.tileEdits[key] = {
      src_b64: t ? t.src_png_b64 : "",
      up_b64: it.out_b64,
      model: result.model || "atlas-import",
      scale: result.scale || result.scale_factor || 1,
      tta: !!result.tta,
      tile_size: result.tile_size != null ? result.tile_size : null,
      gpu_id: result.gpu_id != null ? result.gpu_id : null,
      native_dim: [it.src_w, it.src_h],
      out_dim: [it.out_w, it.out_h],
      source: sourceLabel,
    };
    delete state.cardSliderPcts[key];
    updateTileCard(it.tile_index);
  }
  updateEditsCounter();
  renderFiles();
  applyTileFilter();
}

async function atlasUpscaleComposite() {
  const at = state.atlasState;
  if (!at || !state.currentFile) {
    toast("no atlas loaded", "err");
    return;
  }
  if (state.atlasBusy) {
    toast("atlas job already running", "info");
    return;
  }
  const opts = currentUpscaleOpts();
  if (!opts.model) {
    setStatus("no upscale model available", "err", { sticky: true });
    return;
  }
  const fname = at.filename;
  state.atlasBusy = true;
  $("#atlasSpinner").hidden = false;
  $("#atlasSpinnerLabel").textContent =
    `upscaling composite ${at.composite_w}x${at.composite_h} via ${opts.model} x${opts.scale}...`;
  setStatus(
    `upscaling ${fname} composite (${at.composite_w}x${at.composite_h}) via ${opts.model} x${opts.scale}...`,
    "busy", { sticky: true }
  );
  try {
    const body = {
      filename: fname,
      model: opts.model,
      scale: opts.scale,
      keep_native_dims: opts.keepNative,
      tta: !!opts.tta,
    };
    if (opts.tileSize !== null && opts.tileSize !== undefined) body.tile_size = opts.tileSize;
    if (opts.gpuId !== null && opts.gpuId !== undefined) body.gpu_id = opts.gpuId;
    const r = await api("/api/atlas_upscale", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    _registerAtlasResultEdits(r, "atlas_upscale");
    toast(`composite upscaled: ${plural(r.tiles.length, "tile")} updated`, "ok");
    setStatus(
      `composite upscaled (${r.upscaled_w}x${r.upscaled_h}); ${plural(r.tiles.length, "tile edit")} staged`,
      "ok"
    );
  } catch (e) {
    setStatus(`atlas upscale failed: ${e.message}`, "err", { sticky: true });
    toast(`atlas upscale failed: ${e.message}`, "err", { ttl: 7000 });
  } finally {
    state.atlasBusy = false;
    $("#atlasSpinner").hidden = true;
    refreshAtlasView();
  }
}

async function atlasImportComposite(file) {
  const at = state.atlasState;
  if (!at || !state.currentFile) {
    toast("no atlas loaded", "err");
    return;
  }
  if (!file) return;
  if (!/\.png$|^image\/png/i.test(file.name + " " + (file.type || ""))) {
    toast(`not a PNG: ${file.name}`, "err");
    return;
  }
  state.atlasBusy = true;
  $("#atlasSpinner").hidden = false;
  $("#atlasSpinnerLabel").textContent = `importing composite ${file.name}...`;
  setStatus(`importing composite ${file.name}...`, "busy", { sticky: true });
  try {
    const buf = await file.arrayBuffer();
    const bytes = new Uint8Array(buf);
    let bin = "";
    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    const b64 = "data:image/png;base64," + btoa(bin);
    const r = await api("/api/atlas_import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: at.filename,
        png_b64: b64,
        keep_native_dims: $("#keepNative").checked,
      }),
    });
    _registerAtlasResultEdits(r, "atlas_import");
    toast(`composite imported (${r.imported_w}x${r.imported_h}, ${r.scale_factor}x): ${plural(r.tiles.length, "tile")} updated`, "ok");
    setStatus(
      `composite imported (${r.imported_w}x${r.imported_h}); ${plural(r.tiles.length, "tile edit")} staged`,
      "ok"
    );
  } catch (e) {
    setStatus(`atlas import failed: ${e.message}`, "err", { sticky: true });
    toast(`atlas import failed: ${e.message}`, "err", { ttl: 7000 });
  } finally {
    state.atlasBusy = false;
    $("#atlasSpinner").hidden = true;
    refreshAtlasView();
  }
}

function setupAtlasMode() {
  bindAtlasSlider();
  const tog = $("#atlasModeToggle");
  if (tog) {
    tog.addEventListener("change", () => {
      state.atlasMode = !!tog.checked;
      state._atlasUserChose = true;
      refreshAtlasView();
    });
  }
  const au = $("#btnAtlasUpscale");
  if (au) au.addEventListener("click", atlasUpscaleComposite);
  const ai = $("#btnAtlasImport");
  if (ai) ai.addEventListener("click", () => $("#atlasImportInput").click());
  const aiInput = $("#atlasImportInput");
  if (aiInput) aiInput.addEventListener("change", async (e) => {
    const f = e.target.files && e.target.files[0];
    if (f) await atlasImportComposite(f);
    e.target.value = "";
  });
}

// Hook openFile so we re-probe atlas availability on each file change. We
// keep the original under _origOpenFile and reassign the binding name openFile.
const _origOpenFile = openFile;
openFile = async function patchedOpenFile(name) {
  state.atlasState = null;
  state._atlasUserChose = false;
  await _origOpenFile(name);
  await refreshAtlasAvailability();
};

// =====================================================================
// Bootstrapping
// =====================================================================
async function loadModels() {
  try {
    const r = await api("/api/models");
    state.models = r.models || [];
    if (Array.isArray(r.allowed_scales)) state.allowedScales = r.allowed_scales;
    if (Array.isArray(r.allowed_tile_sizes)) state.allowedTileSizes = r.allowed_tile_sizes;
    const sel = $("#modelSel");
    sel.innerHTML = "";
    if (state.models.length === 0) {
      const o = document.createElement("option");
      o.value = "";
      o.textContent = "(no models found)";
      o.disabled = true;
      sel.appendChild(o);
      sel.disabled = true;
      return;
    }
    for (const m of state.models) {
      const o = document.createElement("option");
      o.value = m.name;
      o.textContent = m.name;
      o.title = m.description || "";
      sel.appendChild(o);
    }
    const pref = state.models.find((m) => m.name.includes("x4plus-anime"));
    if (pref) sel.value = pref.name;
    buildScaleOptions();
    setupModelInfo();
  } catch (e) {
    console.error("failed to load models:", e);
    setStatus(`failed to load models: ${e.message}`, "err", { sticky: true });
  }
}

function bindGlobalShortcuts() {
  document.addEventListener("keydown", (e) => {
    // Deploy-to-game (promote) modal takes precedence — same Esc handling.
    if (!$("#promoteModal").hidden) {
      if (e.key === "Escape") { hideDeployToGame(); return; }
      if (e.key === "Enter") {
        if (document.activeElement && ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
        const btn = $("#promoteConfirm");
        if (btn && !btn.disabled) { e.preventDefault(); btn.click(); }
        return;
      }
      return;
    }
    // Deploy diff modal takes precedence over the A/B modal for Esc handling.
    if (!$("#deployModal").hidden) {
      if (e.key === "Escape") { hideDeployDiff(); return; }
      if (e.key === "Enter") {
        if (document.activeElement && ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
        e.preventDefault();
        $("#deployConfirm").click();
        return;
      }
      return;
    }
    if (!$("#modal").hidden) {
      if (e.key === "Escape") { closeModal(); return; }
      if (e.key === "ArrowLeft" && document.activeElement !== $("#abHandle")) {
        e.preventDefault(); gotoModalSibling(-1); return;
      }
      if (e.key === "ArrowRight" && document.activeElement !== $("#abHandle")) {
        e.preventDefault(); gotoModalSibling(1); return;
      }
      if (e.key === "Enter") {
        if (document.activeElement && document.activeElement.tagName === "INPUT") return;
        if (document.activeElement && document.activeElement.tagName === "SELECT") return;
        e.preventDefault();
        $("#modalUpscale").click();
        return;
      }
      return;
    }
    const inField = ["INPUT", "TEXTAREA", "SELECT"].includes((e.target && e.target.tagName) || "");
    if (e.ctrlKey && (e.key === "f" || e.key === "F")) {
      e.preventDefault();
      $("#filterBox").focus();
      $("#filterBox").select();
      return;
    }
    // Ctrl+A inside the workspace selects all visible tiles
    if (e.ctrlKey && (e.key === "a" || e.key === "A") && state.currentFile) {
      // Only intercept when not focused inside a regular field
      if (inField) return;
      e.preventDefault();
      const tiles = state.currentFile.tiles.filter((t) =>
        tilePassesFilter(t, state.currentFile.name, state.tileFilter));
      state.selectedIndices = new Set(tiles.map((t) => t.index));
      state.selectionAnchor = tiles.length ? tiles[0].index : null;
      refreshSelectionUI();
      return;
    }
    if (inField) return;
    if (e.key === "Escape" && state.selectedIndices.size > 0) {
      e.preventDefault();
      clearSelection();
      return;
    }
    if (e.key === "u" || e.key === "U") {
      if (state.currentFile && !$("#btnUpscaleAll").disabled) {
        e.preventDefault();
        $("#btnUpscaleAll").click();
      }
    } else if (e.key === "r" || e.key === "R") {
      if (state.currentFile && !$("#btnRepack").disabled) {
        e.preventDefault();
        $("#btnRepack").click();
      }
    } else if (e.key === "/") {
      e.preventDefault();
      // If a file is open, prefer the in-file tile filter; else focus the file list filter.
      if (state.currentFile && $("#tileFilterBox")) {
        $("#tileFilterBox").focus();
        $("#tileFilterBox").select();
      } else {
        $("#filterBox").focus();
        $("#filterBox").select();
      }
    }
  });
}

function bindModalDismiss() {
  $("#modal").addEventListener("mousedown", (e) => {
    if (e.target === $("#modal")) closeModal();
  });
}

// =====================================================================
// AI generate (img2img / inpaint / text2img / controlnet)
//
// Lives entirely in this module — adds NO new top-level state to the
// existing `state` object aside from `state.aigen` (variations + UI).
// AI results land in state.tileEdits the same way upscaler/import results
// do, so the A/B slider, repack splice path, and session export all keep
// working unchanged.
// =====================================================================

state.aigen = {
  // Provider list cached from /api/aigen/providers; refreshed on AI tab open.
  providers: [],
  selectedProvider: null,
  models: [], // models for the selected provider
  selectedModel: null,
  mode: "img2img",
  // Per-tile mask (Uint8ClampedArray) keyed by `filename:idx`. Mask dim
  // matches source tile dim. Cleared on tile switch / explicit "clear".
  masks: {},
  // Inpaint brush state
  brush: { tool: "paint", size: 16 },
  // Mask paint pointer state
  painting: false,
  // Per-tile variations: array of {out_b64, prompt, model, seed, ts}
  variations: {},
  activeVarIdx: null,
  // Probe-refresh cooldown
  lastProbeAt: 0,
};

const AIGEN_VARS_MAX = 8;
const AIGEN_PROBE_MIN_INTERVAL_MS = 1500;

function aigenKey(fname, idx) { return editKey(fname, idx); }

async function aigenRefreshProviders({ silent = false } = {}) {
  const now = Date.now();
  if (now - state.aigen.lastProbeAt < AIGEN_PROBE_MIN_INTERVAL_MS && state.aigen.providers.length) {
    return;
  }
  state.aigen.lastProbeAt = now;
  try {
    const r = await api("/api/aigen/providers");
    state.aigen.providers = Array.isArray(r.providers) ? r.providers : [];
    aigenRenderProviderSelect();
    if (!silent) aigenSetStatus(`providers: ${state.aigen.providers.filter(p=>p.available).length}/${state.aigen.providers.length} live`, "ok");
  } catch (e) {
    if (!silent) aigenSetStatus(`provider probe failed: ${e.message}`, "err");
  }
}

function aigenRenderProviderSelect() {
  const sel = $("#aigenProviderSel");
  if (!sel) return;
  const prevValue = sel.value;
  sel.innerHTML = "";
  for (const p of state.aigen.providers) {
    const opt = document.createElement("option");
    opt.value = p.name;
    const stat = p.available ? "[ok]" : "[off]";
    opt.textContent = `${stat} ${p.label}`;
    if (!p.available) opt.disabled = true;
    opt.title = p.available
      ? `${p.label}\n${p.base_url || "in-process"}`
      : `${p.label} - not running\n${p.hint || ""}`;
    sel.appendChild(opt);
  }
  // Pick the first available, preferring the previously-selected one if it's still up.
  const findFirstAvail = () => state.aigen.providers.find(p => p.available);
  let pick = state.aigen.providers.find(p => p.name === prevValue && p.available)
          || state.aigen.providers.find(p => p.name === state.aigen.selectedProvider && p.available)
          || findFirstAvail();
  if (pick) {
    sel.value = pick.name;
    state.aigen.selectedProvider = pick.name;
    aigenLoadModelsForProvider(pick.name);
  } else {
    state.aigen.selectedProvider = null;
    aigenRenderModelSelect([]);
  }
  aigenUpdateModeButtons();
  aigenUpdateGenerateButton();
}

async function aigenLoadModelsForProvider(name) {
  if (!name) { state.aigen.models = []; aigenRenderModelSelect([]); return; }
  try {
    const r = await api(`/api/aigen/models/${encodeURIComponent(name)}`);
    state.aigen.models = Array.isArray(r.models) ? r.models : [];
    aigenRenderModelSelect(state.aigen.models);
  } catch (e) {
    state.aigen.models = [];
    aigenRenderModelSelect([]);
  }
}

function aigenRenderModelSelect(models) {
  const sel = $("#aigenModelSel");
  if (!sel) return;
  sel.innerHTML = "";
  const dflt = document.createElement("option");
  dflt.value = "";
  dflt.textContent = "(provider default)";
  sel.appendChild(dflt);
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m.name;
    opt.textContent = m.label || m.name;
    sel.appendChild(opt);
  }
  state.aigen.selectedModel = sel.value;
}

function aigenUpdateModeButtons() {
  const provider = state.aigen.providers.find(p => p.name === state.aigen.selectedProvider);
  const supported = new Set(provider ? provider.supported_modes : []);
  $$(".aigen-mode").forEach(btn => {
    const m = btn.dataset.mode;
    btn.disabled = !provider || !provider.available || !supported.has(m);
    btn.classList.toggle("active", m === state.aigen.mode && !btn.disabled);
  });
  if (provider && provider.available && !supported.has(state.aigen.mode)) {
    const first = ["img2img", "text2img", "inpaint", "controlnet"].find(m => supported.has(m));
    if (first) aigenSetMode(first);
  }
}

function aigenSetMode(mode) {
  state.aigen.mode = mode;
  $$(".aigen-mode").forEach(btn => btn.classList.toggle("active", btn.dataset.mode === mode));
  $("#aigenMaskBar").hidden = (mode !== "inpaint");
  const wrap = $("#abWrap");
  if (wrap) {
    wrap.classList.toggle("aigen-inpainting", mode === "inpaint");
  }
  if (mode === "inpaint") {
    aigenEnsureMaskCanvas();
  } else {
    const c = $("#aigenMaskCanvas");
    if (c) c.hidden = true;
  }
  aigenUpdateGenerateButton();
}

function aigenUpdateGenerateButton() {
  const btn = $("#aigenGenerate");
  if (!btn) return;
  const provider = state.aigen.providers.find(p => p.name === state.aigen.selectedProvider);
  const ok = !!(provider && provider.available && provider.supported_modes.includes(state.aigen.mode));
  btn.disabled = !ok;
  if (!provider) {
    btn.title = "no provider selected";
  } else if (!provider.available) {
    btn.title = `${provider.label} is not running. ${provider.hint || ""}`;
  } else if (!provider.supported_modes.includes(state.aigen.mode)) {
    btn.title = `${provider.label} doesn't support ${state.aigen.mode}`;
  } else {
    btn.title = `generate via ${provider.label}`;
  }
}

function aigenSetStatus(text, kind = "info") {
  const row = $("#aigenStatusRow");
  const t = $("#aigenStatusText");
  if (!row || !t) return;
  row.hidden = !text;
  t.className = `dim ${kind}`;
  t.textContent = text || "";
}

function aigenShowTab(which) {
  const upBar = $("#modalUpscaleBar");
  const aiPanel = $("#aigenPanel");
  const tUp = $("#tabUpscale");
  const tAi = $("#tabAigen");
  const hint = $("#modalTabHint");
  if (which === "aigen") {
    upBar.hidden = true;
    aiPanel.hidden = false;
    tUp.classList.remove("active");
    tAi.classList.add("active");
    tUp.setAttribute("aria-selected", "false");
    tAi.setAttribute("aria-selected", "true");
    if (hint) hint.textContent = "AI generation: localhost only";
    aigenRefreshProviders({ silent: true });
    aigenLoadVariationsStrip();
    if (state.aigen.mode === "inpaint") aigenEnsureMaskCanvas();
  } else {
    upBar.hidden = false;
    aiPanel.hidden = true;
    tUp.classList.add("active");
    tAi.classList.remove("active");
    tUp.setAttribute("aria-selected", "true");
    tAi.setAttribute("aria-selected", "false");
    if (hint) hint.textContent = "upscaling: realesrgan-ncnn-vulkan";
    const wrap = $("#abWrap");
    if (wrap) wrap.classList.remove("aigen-inpainting");
    const c = $("#aigenMaskCanvas");
    if (c) c.hidden = true;
  }
}

function aigenEnsureMaskCanvas() {
  if (state.modalTileIdx === null || !state.currentFile) return;
  const t = state.currentFile.tiles.find(x => x.index === state.modalTileIdx);
  if (!t) return;
  const c = $("#aigenMaskCanvas");
  if (!c) return;
  if (c.width !== t.width || c.height !== t.height) {
    c.width = t.width;
    c.height = t.height;
  }
  c.hidden = false;
  aigenRedrawMask();
}

function aigenRedrawMask() {
  if (state.modalTileIdx === null || !state.currentFile) return;
  const c = $("#aigenMaskCanvas");
  if (!c || c.hidden) return;
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  const key = aigenKey(state.currentFile.name, state.modalTileIdx);
  const mask = state.aigen.masks[key];
  if (!mask) return;
  const img = ctx.createImageData(c.width, c.height);
  for (let i = 0; i < mask.length; i++) {
    const v = mask[i];
    img.data[i*4 + 0] = 0;
    img.data[i*4 + 1] = v;
    img.data[i*4 + 2] = v;
    img.data[i*4 + 3] = v;
  }
  ctx.putImageData(img, 0, 0);
}

function aigenGetOrCreateMask(fname, idx, w, h) {
  const key = aigenKey(fname, idx);
  let mask = state.aigen.masks[key];
  if (!mask || mask.length !== w * h) {
    mask = new Uint8ClampedArray(w * h);
    state.aigen.masks[key] = mask;
  }
  return mask;
}

function aigenPaintAt(cssX, cssY) {
  if (state.modalTileIdx === null || !state.currentFile) return;
  const t = state.currentFile.tiles.find(x => x.index === state.modalTileIdx);
  if (!t) return;
  const c = $("#aigenMaskCanvas");
  if (!c) return;
  const rect = c.getBoundingClientRect();
  const px = Math.round((cssX - rect.left) / rect.width * c.width);
  const py = Math.round((cssY - rect.top) / rect.height * c.height);
  const r = Math.max(1, Math.round(state.aigen.brush.size));
  const fname = state.currentFile.name;
  const mask = aigenGetOrCreateMask(fname, t.index, c.width, c.height);
  const value = state.aigen.brush.tool === "erase" ? 0 : 255;
  for (let dy = -r; dy <= r; dy++) {
    const yy = py + dy;
    if (yy < 0 || yy >= c.height) continue;
    for (let dx = -r; dx <= r; dx++) {
      const xx = px + dx;
      if (xx < 0 || xx >= c.width) continue;
      const dist2 = dx*dx + dy*dy;
      if (dist2 > r*r) continue;
      mask[yy * c.width + xx] = value;
    }
  }
  aigenRedrawMask();
}

function aigenSetupMaskListeners() {
  const c = $("#aigenMaskCanvas");
  if (!c || c.dataset.listenersAttached === "1") return;
  c.dataset.listenersAttached = "1";

  c.addEventListener("pointerdown", (e) => {
    if (state.aigen.mode !== "inpaint") return;
    state.aigen.painting = true;
    try { c.setPointerCapture(e.pointerId); } catch {}
    aigenPaintAt(e.clientX, e.clientY);
    e.preventDefault();
  });
  c.addEventListener("pointermove", (e) => {
    if (!state.aigen.painting) return;
    aigenPaintAt(e.clientX, e.clientY);
  });
  c.addEventListener("pointerup", (e) => {
    state.aigen.painting = false;
    try { c.releasePointerCapture(e.pointerId); } catch {}
  });
  c.addEventListener("pointercancel", () => { state.aigen.painting = false; });
}

function aigenClearMask() {
  if (state.modalTileIdx === null || !state.currentFile) return;
  const key = aigenKey(state.currentFile.name, state.modalTileIdx);
  delete state.aigen.masks[key];
  aigenRedrawMask();
}

function aigenInvertMask() {
  if (state.modalTileIdx === null || !state.currentFile) return;
  const t = state.currentFile.tiles.find(x => x.index === state.modalTileIdx);
  if (!t) return;
  const fname = state.currentFile.name;
  const mask = aigenGetOrCreateMask(fname, t.index, t.width, t.height);
  for (let i = 0; i < mask.length; i++) mask[i] = 255 - mask[i];
  aigenRedrawMask();
}

function aigenMaskToB64() {
  if (state.modalTileIdx === null || !state.currentFile) return null;
  const t = state.currentFile.tiles.find(x => x.index === state.modalTileIdx);
  if (!t) return null;
  const fname = state.currentFile.name;
  const key = aigenKey(fname, t.index);
  const mask = state.aigen.masks[key];
  if (!mask) return null;
  const off = document.createElement("canvas");
  off.width = t.width;
  off.height = t.height;
  const ctx = off.getContext("2d");
  const img = ctx.createImageData(t.width, t.height);
  for (let i = 0; i < mask.length; i++) {
    const v = mask[i];
    img.data[i*4 + 0] = v;
    img.data[i*4 + 1] = v;
    img.data[i*4 + 2] = v;
    img.data[i*4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return off.toDataURL("image/png").split(",", 2)[1] || null;
}

function aigenMaskHasContent() {
  if (state.modalTileIdx === null || !state.currentFile) return false;
  const key = aigenKey(state.currentFile.name, state.modalTileIdx);
  const mask = state.aigen.masks[key];
  if (!mask) return false;
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] > 0) return true;
  }
  return false;
}

async function aigenGenerate() {
  if (state.modalTileIdx === null || !state.currentFile) return;
  const t = state.currentFile.tiles.find(x => x.index === state.modalTileIdx);
  if (!t) return;
  const fname = state.currentFile.name;
  const key = aigenKey(fname, t.index);
  const provider = state.aigen.providers.find(p => p.name === state.aigen.selectedProvider);
  if (!provider || !provider.available) {
    aigenSetStatus("provider unavailable; click refresh", "err");
    return;
  }
  const mode = state.aigen.mode;
  if (mode === "inpaint" && !aigenMaskHasContent()) {
    aigenSetStatus("paint a mask first (white = repaint)", "warn");
    return;
  }

  const body = {
    provider: provider.name,
    mode,
    filename: fname,
    tile_index: t.index,
    prompt: $("#aigenPrompt").value || "",
    neg_prompt: $("#aigenNegPrompt").value || "",
    denoise: parseFloat($("#aigenDenoise").value),
    cfg: parseFloat($("#aigenCfg").value),
    steps: parseInt($("#aigenSteps").value, 10),
    seed: parseInt($("#aigenSeed").value || "-1", 10),
    model: state.aigen.selectedModel || $("#aigenModelSel").value || null,
    target_w: t.width,
    target_h: t.height,
  };
  if (mode === "inpaint") {
    body.mask_b64 = aigenMaskToB64();
    if (!body.mask_b64) {
      aigenSetStatus("mask is empty", "err");
      return;
    }
  }

  if (state.busyTiles.has(key)) {
    aigenSetStatus("tile busy, try again", "warn");
    return;
  }
  state.busyTiles.add(key);
  updateTileCard(t.index);
  showModalSpinner(true);
  aigenSetStatus(`generating via ${provider.label} (${mode})...`, "info");
  $("#aigenGenerate").disabled = true;
  setStatus(`AI generating tile ${t.index} via ${provider.label}...`, "busy", { sticky: true });
  const t0 = performance.now();
  try {
    const r = await api("/api/aigen/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
    state.tileEdits[key] = {
      src_b64: t.src_png_b64,
      up_b64: r.out_b64,
      model: `aigen:${r.model || provider.name}`,
      scale: 1,
      tta: false,
      tile_size: null,
      gpu_id: null,
      native_dim: [t.width, t.height],
      out_dim: [r.out_w, r.out_h],
      source: "aigen",
      aigen_mode: mode,
      aigen_provider: provider.name,
      aigen_seed: r.seed,
      aigen_prompt: body.prompt,
    };
    const varList = state.aigen.variations[key] || [];
    varList.unshift({
      out_b64: r.out_b64,
      mode,
      provider: provider.name,
      model: r.model || "",
      seed: r.seed,
      prompt: body.prompt,
      ts: Date.now(),
    });
    if (varList.length > AIGEN_VARS_MAX) varList.length = AIGEN_VARS_MAX;
    state.aigen.variations[key] = varList;
    state.aigen.activeVarIdx = 0;
    aigenLoadVariationsStrip();
    aigenSetStatus(`done in ${elapsed}s; seed=${r.seed} model=${r.model || "default"}`, "ok");
    setStatus(`AI generated tile ${t.index} (${r.out_w}x${r.out_h})`, "ok");
  } catch (e) {
    aigenSetStatus(`generate failed: ${e.message}`, "err");
    setStatus(`AI generate failed: ${e.message}`, "err", { sticky: true });
  } finally {
    state.busyTiles.delete(key);
    updateTileCard(t.index);
    updateEditsCounter();
    renderFiles();
    showModalSpinner(false);
    refreshModalImages();
    aigenUpdateGenerateButton();
  }
}

function aigenLoadVariationsStrip() {
  const bar = $("#aigenVarsBar");
  const strip = $("#aigenVarsStrip");
  if (!bar || !strip) return;
  if (state.modalTileIdx === null || !state.currentFile) {
    bar.hidden = true;
    strip.innerHTML = "";
    return;
  }
  const key = aigenKey(state.currentFile.name, state.modalTileIdx);
  const varList = state.aigen.variations[key] || [];
  bar.hidden = (varList.length === 0);
  strip.innerHTML = "";
  varList.forEach((v, i) => {
    const div = document.createElement("div");
    div.className = "aigen-var-thumb" + (i === state.aigen.activeVarIdx ? " active" : "");
    div.style.backgroundImage = `url("data:image/png;base64,${v.out_b64}")`;
    const tooltipParts = [`V${varList.length - i}`, v.mode, v.provider];
    if (v.seed != null) tooltipParts.push(`seed ${v.seed}`);
    if (v.prompt) tooltipParts.push(`"${v.prompt.slice(0, 60)}"`);
    div.title = tooltipParts.join(" \u2022 ");
    const lbl = document.createElement("span");
    lbl.className = "var-label";
    lbl.textContent = `V${varList.length - i}`;
    div.appendChild(lbl);
    div.addEventListener("click", () => aigenActivateVariation(i));
    strip.appendChild(div);
  });
}

function aigenActivateVariation(i) {
  if (state.modalTileIdx === null || !state.currentFile) return;
  const key = aigenKey(state.currentFile.name, state.modalTileIdx);
  const varList = state.aigen.variations[key] || [];
  const v = varList[i];
  if (!v) return;
  const t = state.currentFile.tiles.find(x => x.index === state.modalTileIdx);
  if (!t) return;
  state.tileEdits[key] = {
    src_b64: t.src_png_b64,
    up_b64: v.out_b64,
    model: `aigen:${v.model || v.provider}`,
    scale: 1,
    tta: false,
    tile_size: null,
    gpu_id: null,
    native_dim: [t.width, t.height],
    out_dim: [t.width, t.height],
    source: "aigen",
    aigen_mode: v.mode,
    aigen_provider: v.provider,
    aigen_seed: v.seed,
    aigen_prompt: v.prompt,
  };
  state.aigen.activeVarIdx = i;
  refreshModalImages();
  updateTileCard(t.index);
  renderFiles();
  aigenLoadVariationsStrip();
}

function aigenClearVariations() {
  if (state.modalTileIdx === null || !state.currentFile) return;
  const key = aigenKey(state.currentFile.name, state.modalTileIdx);
  delete state.aigen.variations[key];
  state.aigen.activeVarIdx = null;
  aigenLoadVariationsStrip();
}

function setupAigenPanel() {
  const tabUp = $("#tabUpscale");
  const tabAi = $("#tabAigen");
  if (tabUp) tabUp.addEventListener("click", () => aigenShowTab("upscale"));
  if (tabAi) tabAi.addEventListener("click", () => aigenShowTab("aigen"));

  $$(".aigen-mode").forEach(btn => {
    btn.addEventListener("click", () => {
      if (btn.disabled) return;
      aigenSetMode(btn.dataset.mode);
    });
  });

  const psel = $("#aigenProviderSel");
  if (psel) psel.addEventListener("change", () => {
    state.aigen.selectedProvider = psel.value || null;
    aigenLoadModelsForProvider(state.aigen.selectedProvider);
    aigenUpdateModeButtons();
    aigenUpdateGenerateButton();
  });
  const msel = $("#aigenModelSel");
  if (msel) msel.addEventListener("change", () => {
    state.aigen.selectedModel = msel.value || null;
  });
  const ref = $("#aigenRefreshBtn");
  if (ref) ref.addEventListener("click", () => aigenRefreshProviders());

  const wireSlider = (id, valId, fmt = (v) => v) => {
    const s = $(id);
    const v = $(valId);
    if (!s || !v) return;
    const update = () => { v.textContent = fmt(s.value); };
    s.addEventListener("input", update);
    update();
  };
  wireSlider("#aigenDenoise", "#aigenDenoiseVal", (v) => parseFloat(v).toFixed(2));
  wireSlider("#aigenCfg", "#aigenCfgVal", (v) => parseFloat(v).toFixed(1));
  wireSlider("#aigenSteps", "#aigenStepsVal", (v) => v);
  wireSlider("#aigenBrushSize", "#aigenBrushSizeVal", (v) => v);
  const bs = $("#aigenBrushSize");
  if (bs) bs.addEventListener("input", () => { state.aigen.brush.size = parseInt(bs.value, 10) || 16; });

  const sr = $("#aigenSeedRand");
  if (sr) sr.addEventListener("click", () => {
    $("#aigenSeed").value = String(Math.floor(Math.random() * 0x7fffffff));
  });

  const tp = $("#aigenBrushPaint");
  const te = $("#aigenBrushErase");
  const setTool = (which) => {
    state.aigen.brush.tool = which;
    if (tp) tp.classList.toggle("active", which === "paint");
    if (te) te.classList.toggle("active", which === "erase");
  };
  if (tp) tp.addEventListener("click", () => setTool("paint"));
  if (te) te.addEventListener("click", () => setTool("erase"));
  const mc = $("#aigenMaskClear");
  if (mc) mc.addEventListener("click", aigenClearMask);
  const mi = $("#aigenMaskInvert");
  if (mi) mi.addEventListener("click", aigenInvertMask);

  const gen = $("#aigenGenerate");
  if (gen) gen.addEventListener("click", aigenGenerate);

  const vc = $("#aigenVarsClear");
  if (vc) vc.addEventListener("click", aigenClearVariations);

  aigenSetupMaskListeners();
}

async function init() {
  // Drop cached tiles for any file the live-reload watcher reports
  // changed on disk, so the in-memory cache never serves stale bytes.
  if (window.bus && typeof window.bus.on === "function") {
    window.bus.on("cache.changed", (ev) => {
      try {
        const p = ev && ev.path ? String(ev.path) : "";
        const base = p.split(/[\\/]/).pop();
        if (base) invalidateTileCache(base);
      } catch (_e) {}
    });
  }
  setupAnimObserver();
  setupCardDragListeners();
  setupDragDrop();
  setupCardImportPicker();
  setupAtlasMode();
  await loadModels();
  await loadFiles();

  // Header "Tools ▾" overflow menu (2026-06-19 anti-slop). Click to
  // open/close; clicking outside or hitting Esc closes it; selecting an
  // item inside also closes it.
  const toolsBtn = $("#btnToolsMenu");
  const toolsMenu = $("#hdrToolsMenu");
  if (toolsBtn && toolsMenu) {
    const setToolsOpen = (open) => {
      toolsMenu.hidden = !open;
      toolsBtn.setAttribute("aria-expanded", open ? "true" : "false");
    };
    toolsBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      setToolsOpen(toolsMenu.hidden);
    });
    toolsMenu.addEventListener("click", (e) => {
      // Close after any real button activation inside the menu.
      if (e.target.closest("button")) setToolsOpen(false);
    });
    document.addEventListener("click", (e) => {
      if (!toolsMenu.hidden && !e.target.closest(".hdr-tools")) setToolsOpen(false);
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !toolsMenu.hidden) setToolsOpen(false);
    });
  }

  $("#filterBox").addEventListener("input", filterFiles);
  $("#btnUpscaleAll").addEventListener("click", upscaleAll);
  $("#btnRevertAll").addEventListener("click", revertAllInCurrentFile);
  $("#btnRepack").addEventListener("click", repackDeployFlow);
  $("#btnClearEdits").addEventListener("click", clearAllEdits);
  $("#btnSettingsToggle").addEventListener("click", () => {
    const adv = $("#advSettings");
    adv.hidden = !adv.hidden;
    $("#btnSettingsToggle").textContent = adv.hidden ? "settings \u25BE" : "settings \u25B4";
  });

  // Tile filter input
  const tfb = $("#tileFilterBox");
  if (tfb) {
    tfb.addEventListener("input", () => {
      state.tileFilter = parseTileFilter(tfb.value);
      applyTileFilter();
    });
    tfb.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        tfb.value = "";
        state.tileFilter = null;
        applyTileFilter();
        tfb.blur();
      }
    });
  }

  // Batch operation buttons
  $("#btnBatchUpscale").addEventListener("click", batchUpscaleSelected);
  $("#btnBatchRevert").addEventListener("click", batchRevertSelected);
  $("#btnBatchClear").addEventListener("click", clearSelection);

  // Session save / load
  $("#btnSessionExport").addEventListener("click", exportSession);
  $("#btnSessionImport").addEventListener("click", () => $("#sessionImportInput").click());
  $("#sessionImportInput").addEventListener("change", async (e) => {
    const f = e.target.files && e.target.files[0];
    if (f) await importSession(f);
    e.target.value = "";
  });

  // Deploy diff modal
  $("#deployModalClose").addEventListener("click", hideDeployDiff);
  $("#deployCancel").addEventListener("click", hideDeployDiff);
  $("#deployConfirm").addEventListener("click", doRepackDeploy);
  $("#deployModal").addEventListener("mousedown", (e) => {
    if (e.target === $("#deployModal")) hideDeployDiff();
  });

  // Deploy-to-game (dev -> live PSOBB.IO install)
  // 2026-04-25: header-initiated flows toggle .pso-modal-allow-deploy on
  // body so the unified-mode CSS un-hides #promoteModal (kept hidden by
  // default to avoid colliding with the deploy-diff perspective).
  $("#btnDeployToGame").addEventListener("click", () => {
    document.body.classList.add("pso-modal-allow-deploy");
    showDeployToGame();
  });
  const _hideAndDropAllow = () => {
    document.body.classList.remove("pso-modal-allow-deploy");
    hideDeployToGame();
  };
  $("#promoteModalClose").addEventListener("click", _hideAndDropAllow);
  $("#promoteCancel").addEventListener("click", _hideAndDropAllow);
  $("#promoteConfirm").addEventListener("click", doPromote);
  $("#promoteSelectChanged").addEventListener("click", promoteSelectAllChanged);
  $("#promoteSelectNone").addEventListener("click", promoteSelectNone);
  $("#promoteModal").addEventListener("mousedown", (e) => {
    if (e.target === $("#promoteModal")) hideDeployToGame();
  });

  $("#fitSel").addEventListener("change", applyFitMode);
  $("#zoomRange").addEventListener("input", applyZoom);
  applyZoom();

  $("#modalClose").addEventListener("click", closeModal);
  $("#modalUpscale").addEventListener("click", async () => {
    if (state.modalTileIdx === null) return;
    const opts = currentUpscaleOpts();
    if (!opts.model) { setStatus("no upscale model available", "err", { sticky: true }); return; }
    await upscaleTile(state.modalTileIdx, opts);
  });
  $("#modalRevert").addEventListener("click", () => {
    if (state.modalTileIdx !== null) revertTile(state.modalTileIdx);
  });
  $("#modalPrev").addEventListener("click", () => gotoModalSibling(-1));
  $("#modalNext").addEventListener("click", () => gotoModalSibling(1));

  // Modal anim controls
  $("#modalAnimEnable").addEventListener("change", persistModalAnim);
  $("#modalAnimGrid").addEventListener("change", () => {
    $("#modalAnimCustomWrap").hidden = $("#modalAnimGrid").value !== "custom";
    persistModalAnim();
  });
  $("#modalAnimCustomCols").addEventListener("change", persistModalAnim);
  $("#modalAnimCustomRows").addEventListener("change", persistModalAnim);
  $("#modalAnimFps").addEventListener("change", persistModalAnim);
  $("#modalAnimOrder").addEventListener("change", persistModalAnim);

  bindSlider();
  bindGlobalShortcuts();
  bindModalDismiss();

  // Modal upscale bar: clone toolbar selectors into modal + wire two-way binding
  syncModalUpscaleBar();
  setupModalUpscaleBarBinding();
  // Also re-sync after model list reloads
  $("#modelSel").addEventListener("change", syncModalUpscaleBar);
  $("#scaleSel").addEventListener("change", syncModalUpscaleBar);

  // AI generate tab — additive; does not touch upscale path
  setupAigenPanel();
  // Initial provider probe in the background so the AI tab opens fast.
  aigenRefreshProviders({ silent: true }).catch(() => {});

  // First-run onboarding (2026-06-19). Auto-shows the guided walkthrough
  // once per browser (localStorage-gated); no-ops if already seen or if
  // an asset is already open. Re-openable any time via the header "?"
  // button or the empty-state "Start the walkthrough" button.
  if (window.psoOnboarding && window.psoOnboarding.maybeAutoShow) {
    try { window.psoOnboarding.maybeAutoShow(); } catch (_e) {}
  }
}

window.addEventListener("DOMContentLoaded", () => {
  $("#abDst").style.clipPath = "inset(0 0 0 50%)";
  // Reveal the empty-state "Asset Workshop" welcome only AFTER the app has
  // settled — it's hardcoded in index.html so it paints before JS and FLASHES
  // on every refresh before perspectives.js decides what to render. CSS hides
  // .vp-stage-empty until body.app-ready (set here after two frames).
  requestAnimationFrame(() =>
    requestAnimationFrame(() => document.body.classList.add("app-ready")),
  );
});

// V4 quality: expose the build-only entry point so the UX agent (or any
// caller) can drive it without coupling to the rest of the module.
// Reading state.exportPath / state.exportFilename after a successful
// call gives the download URL + suggested filename.
window.psoEditor = window.psoEditor || {};
window.psoEditor.doRepackBuildOnly = doRepackBuildOnly;
window.psoEditor.state = state;

// Unified-viewport bridge (2026-04-24): perspectives.js needs to drive
// modal-open/close from the central stage; expose those entry points.
window.openModal = openModal;
window.closeModal = closeModal;
window.showDeployDiff = showDeployDiff;
// Debug-only legacy-modal flag (2026-04-25, regression-fix-modal-vs-viewport).
// Default false: openModal() redirects through the "tile-detail"
// perspective (body.unified-viewport-mode is always set). Setting this
// true forces the legacy fullscreen #modal — purely a debug escape hatch;
// no UI surfaces it (the classic-UI toggle was removed 2026-06-20).
if (typeof window.psoUseLegacyModal === "undefined") {
  window.psoUseLegacyModal = false;
}
window.psoEditor.openModal = openModal;
window.psoEditor.closeModal = closeModal;
window.psoEditor.showDeployDiff = showDeployDiff;
// Bridge for viewport.js (Web Component): take the /api/viewport_paint
// response and register its per-tile slices in state.tileEdits exactly
// like _registerAtlasResultEdits. We tag the source as "viewport_paint"
// so the user can distinguish these edits from atlas/import edits.
window.psoEditor.applyViewportResult = function (rsp) {
  if (!rsp || !Array.isArray(rsp.tiles)) return;
  // Reuse the atlas registration path — shape matches.
  _registerAtlasResultEdits(rsp, "viewport_paint");
};
// Expose toast so viewport.js can surface errors uniformly.
window.psoEditor.toast = toast;
// Expose a hook so viewport.js can read per-tile animation configs
// from localStorage (set by the existing modal animation feature; key
// scheme is `anim_<filename>_<tileIndex>` -> JSON {frameGrid, frameOrder, fps, enabled}).
// Returns a map of {tile_index: cfg} for tiles whose `enabled` flag is true,
// or {} if none.
window.psoEditor.getAnimConfigs = function (filename) {
  try {
    const out = {};
    const prefix = `anim_${filename}_`;
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (!k || !k.startsWith(prefix)) continue;
      const idx = parseInt(k.slice(prefix.length), 10);
      if (!Number.isFinite(idx)) continue;
      let raw = null;
      try { raw = localStorage.getItem(k); } catch { continue; }
      if (!raw) continue;
      try {
        const p = JSON.parse(raw);
        if (!p || !p.enabled) continue;
        if (!Array.isArray(p.frameGrid) || p.frameGrid.length !== 2) continue;
        out[idx] = p;
      } catch { /* ignore bad json */ }
    }
    return out;
  } catch (_) { return {}; }
};

init();
