/* viewport.js — 16:9 viewport transform mode for the PSOBB Texture Editor.
 *
 * Mounts a <pso-viewport-canvas> Web Component that:
 *   1. Pulls /api/viewport/{filename} for the active file (atlas-known
 *      placements scaled to 1278x768, or a centered fallback).
 *   2. Renders the assembled composite onto an HTML canvas at native
 *      viewport pixel dim, CSS-scaled with image-rendering: pixelated.
 *   3. Lets the user paint with brush / fill / eraser / shape / spray /
 *      eyedropper tools. Rectangular selection + copy/paste. Undo/redo
 *      via Ctrl+Z / Ctrl+Shift+Z (64-deep ring buffer).
 *   4. Exposes "show as game renders" vs "show source tiles" toggle.
 *   5. Exposes "animate" toggle that plays per-tile frame grids (read
 *      from window.psoEditor.state.animConfigs if present) at 30 FPS.
 *      v2: skipped if no anim config is registered for the active file.
 *   6. POSTs the painted canvas back to /api/viewport_paint, which
 *      slices each placement back to its native tile dim and registers
 *      it in state.tileEdits like atlas_import — so the existing repack
 *      pipeline picks it up unchanged.
 *
 * The component is self-contained (shadow DOM) but reads the active
 * file from window.psoEditor.state.currentFile.name. It calls
 * window.psoEditor.applyViewportResult(rsp) after a successful paint
 * to register the result; that hook lives in app.js.
 *
 * Keybindings (when the viewport overlay is open and the canvas has focus):
 *   B / P     brush (paint)
 *   E         eraser
 *   F         fill bucket
 *   I         eyedropper
 *   R         rectangle
 *   L         line
 *   M         marquee selection
 *   S         spraypaint
 *   Ctrl+Z    undo
 *   Ctrl+Shift+Z  redo
 *   Ctrl+C    copy selection (if marquee active)
 *   Ctrl+V    paste clipboard
 *   [ / ]     decrease / increase brush size
 *   Esc       close viewport overlay
 */
(function () {
  "use strict";

  if (window.__psoViewportLoaded) return;
  window.__psoViewportLoaded = true;

  // ---------- constants (must match server.py) ----------
  const VIEWPORT_W = 1278;
  const VIEWPORT_H = 768;
  const UNDO_DEPTH = 64;

  // ---------- color palette (named swatches). Keep small + curated. ----------
  const NAMED_PALETTE = [
    { name: "white",       rgba: [255, 255, 255, 255] },
    { name: "black",       rgba: [0, 0, 0, 255] },
    { name: "tk-blue",     rgba: [0, 255, 255, 255] },
    { name: "tk-purple",   rgba: [157, 78, 221, 255] },
    { name: "tk-orange",   rgba: [255, 170, 0, 255] },
    { name: "tk-green",    rgba: [0, 255, 136, 255] },
    { name: "red",         rgba: [255, 0, 0, 255] },
    { name: "yellow",      rgba: [255, 255, 0, 255] },
    { name: "magenta",     rgba: [255, 0, 255, 255] },
    { name: "transparent", rgba: [0, 0, 0, 0] },
  ];

  // ---------- helpers ----------
  function rgbaToHex(rgba) {
    const [r, g, b] = rgba;
    return "#" + [r, g, b].map(v => v.toString(16).padStart(2, "0")).join("");
  }
  function hexToRgb(h) {
    const m = /^#?([0-9a-fA-F]{6})$/.exec(h);
    if (!m) return [0, 0, 0];
    const n = parseInt(m[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function distSq(ax, ay, bx, by) { const dx = ax - bx, dy = ay - by; return dx * dx + dy * dy; }
  function loadImage(src) {
    return new Promise((res, rej) => {
      const img = new Image();
      img.onload = () => res(img);
      img.onerror = (e) => rej(new Error("image load failed"));
      img.src = src;
    });
  }

  // ---------- flood fill (4-connected, integer tolerance on RGBA distance) ----------
  function floodFill(imageData, x, y, fillRgba, tolerance) {
    const w = imageData.width, h = imageData.height;
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    const data = imageData.data;
    const idx0 = (y * w + x) * 4;
    const tr = data[idx0], tg = data[idx0 + 1], tb = data[idx0 + 2], ta = data[idx0 + 3];
    const fr = fillRgba[0], fg = fillRgba[1], fb = fillRgba[2], fa = fillRgba[3];
    if (tr === fr && tg === fg && tb === fb && ta === fa) return;  // already that color
    const tolSq = tolerance * tolerance * 4;  // L2-ish on the 4 channels
    function similar(i) {
      const dr = data[i] - tr;
      const dg = data[i + 1] - tg;
      const db = data[i + 2] - tb;
      const da = data[i + 3] - ta;
      return (dr * dr + dg * dg + db * db + da * da) <= tolSq;
    }
    // Iterative scanline flood fill - DFS would blow the stack on big areas.
    const stack = [[x, y]];
    while (stack.length) {
      const [px, py] = stack.pop();
      let cx = px;
      // walk left to span start
      while (cx >= 0 && similar((py * w + cx) * 4)) cx--;
      cx++;
      let spanAbove = false, spanBelow = false;
      while (cx < w && similar((py * w + cx) * 4)) {
        const i = (py * w + cx) * 4;
        data[i] = fr; data[i + 1] = fg; data[i + 2] = fb; data[i + 3] = fa;
        if (py > 0) {
          const sim = similar(((py - 1) * w + cx) * 4);
          if (!spanAbove && sim) { stack.push([cx, py - 1]); spanAbove = true; }
          else if (spanAbove && !sim) spanAbove = false;
        }
        if (py < h - 1) {
          const sim = similar(((py + 1) * w + cx) * 4);
          if (!spanBelow && sim) { stack.push([cx, py + 1]); spanBelow = true; }
          else if (spanBelow && !sim) spanBelow = false;
        }
        cx++;
      }
    }
  }

  // =====================================================================
  // <pso-viewport-canvas> Web Component
  // =====================================================================
  class PsoViewportCanvas extends HTMLElement {
    constructor() {
      super();
      this.attachShadow({ mode: "open" });
      this._state = this._defaultState();
      this._renderShell();
    }

    _defaultState() {
      return {
        filename: null,
        layout: null,            // 'atlas' | 'centered'
        source: null,            // 'atlas_layouts.py' | 'guessed'
        placements: [],
        skipTiles: [],
        inset: { x: 0, y: 0, w: VIEWPORT_W, h: VIEWPORT_H },
        compositeImg: null,      // HTMLImageElement of source composite
        tool: "brush",
        brushSize: 8,
        color: [0, 255, 255, 255],   // RGBA, default tk-blue
        opacity: 1.0,
        tolerance: 12,           // for fill bucket
        spraydens: 0.25,         // for spraypaint
        showOverlay: true,       // dotted pillar/letterbox indicators
        showAsGame: true,        // false = show source tiles grid
        animEnabled: false,
        animFps: 30,
        zoom: 0.6,               // CSS scale factor
        // Drawing buffers
        // - paintCanvas/paintCtx hold the user's strokes (transparent
        //   except where painted). On save, paintCanvas is composited
        //   over compositeImg into a final PNG.
        paintCanvas: null,
        paintCtx: null,
        // Selection / clipboard
        marquee: null,           // {x,y,w,h} in viewport pixels
        clipboard: null,         // ImageData
        // Pointer-tracking for stroking
        drawing: false,
        lastX: -1,
        lastY: -1,
        previewShape: null,      // for rect/line tool: {x0,y0,x1,y1}
        // Undo / redo ring buffers (each entry is a paintCanvas snapshot)
        undoStack: [],
        redoStack: [],
        // Animation state
        animTimer: null,
        animFrame: 0,
        // Status / busy
        busy: false,
        statusText: "",
      };
    }

    _renderShell() {
      const tk = (n, fb) => `var(${n}, ${fb})`;
      this.shadowRoot.innerHTML = `
        <style>
          :host {
            display: block;
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.85);
            z-index: 1000;
            font: 13px ${tk("--tk-font-body", "system-ui, sans-serif")};
            color: ${tk("--tk-text", "#e0f0ff")};
          }
          :host([hidden]) { display: none !important; }
          .vp-card {
            position: absolute;
            inset: 16px;
            display: grid;
            grid-template-rows: auto auto 1fr auto;
            background: ${tk("--tk-panel", "#0f141b")};
            border: 1px solid ${tk("--tk-line", "rgba(0,255,255,0.3)")};
            border-radius: 6px;
            box-shadow: ${tk("--tk-glow-card", "0 0 20px rgba(0,255,255,0.2)")};
            overflow: hidden;
          }
          .vp-header {
            display: flex; align-items: center; gap: 12px;
            padding: 10px 14px;
            border-bottom: 1px solid #2a313a;
            background: #11161e;
          }
          .vp-header strong { font-size: 14px; }
          .vp-header .meta { color: ${tk("--tk-text-mute", "rgba(224,240,255,0.7)")}; font-size: 12px; }
          .vp-header .grow { flex: 1; }
          .vp-toolbar {
            display: flex; align-items: center; gap: 8px;
            padding: 8px 14px;
            border-bottom: 1px solid #2a313a;
            background: #0d1218;
            flex-wrap: wrap;
          }
          .vp-toolbar label {
            display: inline-flex; align-items: center; gap: 4px;
            font-size: 12px;
          }
          .vp-toolbar select, .vp-toolbar input[type=number] {
            background: #0a0e13; color: #d6dee7; border: 1px solid #2a313a;
            border-radius: 3px; padding: 2px 6px; font: inherit;
          }
          .vp-toolbar input[type=color] {
            width: 28px; height: 24px; padding: 0; border: 1px solid #2a313a;
            background: #0a0e13;
          }
          .vp-toolbar input[type=range] { vertical-align: middle; }
          .vp-tool-btn {
            background: #0a0e13;
            color: ${tk("--tk-text", "#d6dee7")};
            border: 1px solid #2a313a;
            border-radius: 3px;
            padding: 4px 10px;
            cursor: pointer;
            font: inherit;
          }
          .vp-tool-btn.active {
            border-color: ${tk("--tk-blue", "#00ffff")};
            box-shadow: ${tk("--tk-glow-blue", "0 0 6px #0ff")};
            color: #fff;
          }
          .vp-tool-btn:hover:not(.active) { border-color: #4a525a; }
          .vp-stage {
            position: relative;
            overflow: auto;
            background:
              linear-gradient(45deg, #0a0e13 25%, transparent 25%),
              linear-gradient(-45deg, #0a0e13 25%, transparent 25%),
              linear-gradient(45deg, transparent 75%, #0a0e13 75%),
              linear-gradient(-45deg, transparent 75%, #0a0e13 75%);
            background-color: #161b22;
            background-size: 20px 20px;
            background-position: 0 0, 0 10px, 10px -10px, -10px 0;
            padding: 24px;
            display: flex;
            justify-content: center;
            align-items: flex-start;
          }
          .vp-canvas-wrap {
            position: relative;
            border: 2px solid ${tk("--tk-line-strong", "#00ffff")};
            box-shadow: 0 0 30px rgba(0,255,255,0.2);
            background: #000;
            transform-origin: top left;
            flex-shrink: 0;
          }
          canvas {
            display: block;
            image-rendering: pixelated;
            image-rendering: crisp-edges;
            position: absolute;
            inset: 0;
          }
          canvas.base { z-index: 1; }
          canvas.paint { z-index: 2; }
          canvas.overlay { z-index: 3; pointer-events: none; }
          canvas.cursor { z-index: 4; pointer-events: none; }
          .vp-canvas-host {
            position: relative;
            width: ${VIEWPORT_W}px;
            height: ${VIEWPORT_H}px;
          }
          .vp-canvas-wrap.hand canvas.paint { cursor: crosshair; }
          .vp-canvas-wrap.eyedropper canvas.paint { cursor: crosshair; }
          .vp-canvas-wrap.fill canvas.paint { cursor: cell; }
          .vp-canvas-wrap.marquee canvas.paint { cursor: crosshair; }
          .vp-status {
            display: flex; gap: 12px;
            padding: 6px 14px;
            border-top: 1px solid #2a313a;
            background: #0d1218;
            font-size: 12px;
            color: ${tk("--tk-text-mute", "rgba(224,240,255,0.7)")};
            font-family: ${tk("--tk-font-mono", "ui-monospace, Consolas, monospace")};
            flex-wrap: wrap;
          }
          .vp-status .grow { flex: 1; }
          .vp-status .err { color: #ff6666; }
          .vp-status .ok { color: ${tk("--tk-green", "#00ff88")}; }
          .vp-status .busy { color: ${tk("--tk-orange", "#ffaa00")}; }
          .palette {
            display: inline-flex; gap: 4px;
            padding: 0 4px; align-items: center;
          }
          .swatch {
            width: 18px; height: 18px;
            border: 1px solid #2a313a;
            border-radius: 2px;
            cursor: pointer;
            position: relative;
          }
          .swatch.transparent {
            background:
              linear-gradient(45deg, #555 25%, transparent 25%, transparent 75%, #555 75%),
              linear-gradient(45deg, #555 25%, transparent 25%, transparent 75%, #555 75%);
            background-size: 6px 6px;
            background-position: 0 0, 3px 3px;
            background-color: #222;
          }
          .swatch.active { border-color: ${tk("--tk-blue", "#00ffff")}; box-shadow: 0 0 4px #0ff; }
          .save-btn {
            background: ${tk("--tk-green", "#00ff88")};
            color: #000;
            font-weight: 600;
            border: 1px solid #2a313a;
            border-radius: 3px;
            padding: 4px 14px;
            cursor: pointer;
          }
          .save-btn:hover { filter: brightness(1.1); }
          .close-btn {
            background: #2a313a; color: #d6dee7;
            border: 1px solid #444; border-radius: 3px;
            padding: 4px 12px; cursor: pointer;
          }
          .legend {
            font-size: 11px; color: ${tk("--tk-text-dim", "rgba(224,240,255,0.5)")};
          }
          kbd {
            background: #0a0e13;
            border: 1px solid #444;
            border-radius: 2px;
            padding: 1px 4px;
            font-family: ${tk("--tk-font-mono", "ui-monospace, monospace")};
            font-size: 10px;
            color: ${tk("--tk-text-mute", "rgba(224,240,255,0.7)")};
          }
        </style>
        <div class="vp-card">
          <div class="vp-header">
            <strong id="title">16:9 viewport</strong>
            <span class="meta" id="meta"></span>
            <span class="grow"></span>
            <span class="legend">paint extends past 4:3 into the pillar that the widescreen ASI fills</span>
            <button class="save-btn" id="saveBtn" title="slice painted canvas back into per-tile edits">save edit</button>
            <button class="close-btn" id="closeBtn" title="close (Esc)">close</button>
          </div>
          <div class="vp-toolbar">
            <button class="vp-tool-btn" data-tool="brush" title="brush (B/P)">brush</button>
            <button class="vp-tool-btn" data-tool="eraser" title="eraser (E)">erase</button>
            <button class="vp-tool-btn" data-tool="fill" title="fill bucket (F)">fill</button>
            <button class="vp-tool-btn" data-tool="eyedropper" title="eyedropper (I)">eyedrop</button>
            <button class="vp-tool-btn" data-tool="rect" title="rectangle (R)">rect</button>
            <button class="vp-tool-btn" data-tool="line" title="line (L)">line</button>
            <button class="vp-tool-btn" data-tool="spray" title="spraypaint (S)">spray</button>
            <button class="vp-tool-btn" data-tool="marquee" title="rectangle select - copy with Ctrl+C, paste with Ctrl+V (M)">select</button>
            <span class="legend">|</span>
            <label>color
              <input type="color" id="colorPicker" value="#00ffff" />
            </label>
            <span class="palette" id="palette"></span>
            <span class="legend">|</span>
            <label>size <input type="number" id="brushSize" min="1" max="128" value="8" style="width: 50px" /></label>
            <label>opacity <input type="range" id="opacity" min="0" max="100" value="100" style="width: 80px" /></label>
            <label title="fill tolerance (0..255)">tol <input type="number" id="tolerance" min="0" max="255" value="12" style="width: 50px" /></label>
            <span class="legend">|</span>
            <button class="vp-tool-btn" id="undoBtn" title="undo (Ctrl+Z)">undo</button>
            <button class="vp-tool-btn" id="redoBtn" title="redo (Ctrl+Shift+Z)">redo</button>
            <button class="vp-tool-btn" id="clearBtn" title="erase all paint">clear paint</button>
            <span class="legend">|</span>
            <label title="show pillar / letterbox dotted overlay">
              <input type="checkbox" id="overlayToggle" checked />
              overlay
            </label>
            <label title="toggle: show as game renders OR show source tiles split">
              <input type="checkbox" id="modeToggle" checked />
              show as game renders
            </label>
            <label title="play tile animations if registered">
              <input type="checkbox" id="animToggle" />
              animate
            </label>
            <span class="grow" style="flex:1"></span>
            <label>zoom
              <input type="range" id="zoom" min="20" max="200" value="60" style="width: 100px" />
              <span id="zoomLabel">60%</span>
            </label>
          </div>
          <div class="vp-stage" id="stage">
            <div class="vp-canvas-wrap" id="canvasWrap">
              <div class="vp-canvas-host">
                <canvas class="base" id="baseCanvas" width="${VIEWPORT_W}" height="${VIEWPORT_H}"></canvas>
                <canvas class="paint" id="paintCanvas" width="${VIEWPORT_W}" height="${VIEWPORT_H}"></canvas>
                <canvas class="overlay" id="overlayCanvas" width="${VIEWPORT_W}" height="${VIEWPORT_H}"></canvas>
                <canvas class="cursor" id="cursorCanvas" width="${VIEWPORT_W}" height="${VIEWPORT_H}"></canvas>
              </div>
            </div>
          </div>
          <div class="vp-status">
            <span id="statusText">ready</span>
            <span class="grow"></span>
            <span id="cursorPos"></span>
            <span class="legend">
              <kbd>B</kbd> brush
              <kbd>E</kbd> erase
              <kbd>F</kbd> fill
              <kbd>I</kbd> eyedrop
              <kbd>R</kbd> rect
              <kbd>L</kbd> line
              <kbd>S</kbd> spray
              <kbd>M</kbd> select
              <kbd>Ctrl+Z</kbd> undo
              <kbd>[</kbd>/<kbd>]</kbd> size
              <kbd>Esc</kbd> close
            </span>
          </div>
        </div>
      `;

      this._buildPalette();
      this._wire();
      this._setActiveTool("brush");
    }

    _buildPalette() {
      const root = this.shadowRoot.getElementById("palette");
      for (const sw of NAMED_PALETTE) {
        const el = document.createElement("span");
        el.className = "swatch";
        if (sw.name === "transparent") el.classList.add("transparent");
        else el.style.background = `rgb(${sw.rgba[0]}, ${sw.rgba[1]}, ${sw.rgba[2]})`;
        el.title = sw.name;
        el.dataset.rgba = sw.rgba.join(",");
        el.addEventListener("click", () => this._setColor(sw.rgba));
        root.appendChild(el);
      }
    }

    _wire() {
      const sr = this.shadowRoot;
      // Toolbar buttons
      sr.querySelectorAll(".vp-tool-btn[data-tool]").forEach(btn => {
        btn.addEventListener("click", () => this._setActiveTool(btn.dataset.tool));
      });
      sr.getElementById("colorPicker").addEventListener("input", e => {
        const [r, g, b] = hexToRgb(e.target.value);
        this._setColor([r, g, b, this._state.color[3]]);
      });
      sr.getElementById("brushSize").addEventListener("change", e => {
        this._state.brushSize = clamp(parseInt(e.target.value, 10) || 8, 1, 128);
      });
      sr.getElementById("opacity").addEventListener("input", e => {
        this._state.opacity = clamp((parseInt(e.target.value, 10) || 100) / 100, 0, 1);
      });
      sr.getElementById("tolerance").addEventListener("change", e => {
        this._state.tolerance = clamp(parseInt(e.target.value, 10) || 12, 0, 255);
      });
      sr.getElementById("undoBtn").addEventListener("click", () => this._undo());
      sr.getElementById("redoBtn").addEventListener("click", () => this._redo());
      sr.getElementById("clearBtn").addEventListener("click", () => this._clearPaint());
      sr.getElementById("overlayToggle").addEventListener("change", e => {
        this._state.showOverlay = !!e.target.checked;
        this._drawOverlay();
      });
      sr.getElementById("modeToggle").addEventListener("change", e => {
        this._state.showAsGame = !!e.target.checked;
        this._renderBase();
        this._drawOverlay();
      });
      sr.getElementById("animToggle").addEventListener("change", e => {
        this._state.animEnabled = !!e.target.checked;
        this._restartAnim();
      });
      sr.getElementById("zoom").addEventListener("input", e => {
        const z = clamp((parseInt(e.target.value, 10) || 60) / 100, 0.2, 2.0);
        this._state.zoom = z;
        this._applyZoom();
      });
      sr.getElementById("saveBtn").addEventListener("click", () => this._save());
      sr.getElementById("closeBtn").addEventListener("click", () => this.close());
      // Canvas pointer events
      const paint = sr.getElementById("paintCanvas");
      paint.addEventListener("mousedown", (e) => this._onMouseDown(e));
      paint.addEventListener("mousemove", (e) => this._onMouseMove(e));
      paint.addEventListener("mouseup", (e) => this._onMouseUp(e));
      paint.addEventListener("mouseleave", (e) => this._onMouseLeave(e));
      // Keyboard inside the host (not shadow-piercing — we listen on
      // window when open and dispatch into the component).
      this._keyHandler = (e) => this._onKey(e);
    }

    _setColor(rgba) {
      this._state.color = rgba.slice(0, 4);
      while (this._state.color.length < 4) this._state.color.push(255);
      const cp = this.shadowRoot.getElementById("colorPicker");
      cp.value = rgbaToHex(this._state.color);
      // Highlight the active named swatch (if any)
      const swatches = this.shadowRoot.querySelectorAll(".swatch");
      swatches.forEach(s => {
        const sr = (s.dataset.rgba || "").split(",").map(Number);
        s.classList.toggle("active",
          sr[0] === this._state.color[0] &&
          sr[1] === this._state.color[1] &&
          sr[2] === this._state.color[2] &&
          sr[3] === this._state.color[3]
        );
      });
    }

    _setActiveTool(tool) {
      this._state.tool = tool;
      const sr = this.shadowRoot;
      sr.querySelectorAll(".vp-tool-btn[data-tool]").forEach(b =>
        b.classList.toggle("active", b.dataset.tool === tool)
      );
      const wrap = sr.getElementById("canvasWrap");
      wrap.classList.toggle("hand", tool === "brush" || tool === "eraser" || tool === "rect" || tool === "line" || tool === "spray");
      wrap.classList.toggle("eyedropper", tool === "eyedropper");
      wrap.classList.toggle("fill", tool === "fill");
      wrap.classList.toggle("marquee", tool === "marquee");
      // Switching tools clears any pending shape preview.
      this._state.previewShape = null;
      this._drawCursor();
      this._setStatus(`tool: ${tool}`);
    }

    _setStatus(text, kind) {
      this._state.statusText = text || "";
      const el = this.shadowRoot.getElementById("statusText");
      el.textContent = this._state.statusText;
      el.className = kind || "";
    }

    _applyZoom() {
      // Outer wrap takes the scaled dim so the scrollable stage sees it
      // at the visible size; inner host carries the CSS scale so the
      // native-px canvas still does pixel-accurate hit-testing via the
      // bounding-rect ratio.
      const wrap = this.shadowRoot.getElementById("canvasWrap");
      wrap.style.width = (VIEWPORT_W * this._state.zoom) + "px";
      wrap.style.height = (VIEWPORT_H * this._state.zoom) + "px";
      const host = wrap.querySelector(".vp-canvas-host");
      host.style.transform = `scale(${this._state.zoom})`;
      host.style.transformOrigin = "top left";
      const lbl = this.shadowRoot.getElementById("zoomLabel");
      if (lbl) lbl.textContent = Math.round(this._state.zoom * 100) + "%";
    }

    // -------------------- public API --------------------

    /** Open the viewport overlay for the given filename. */
    async open(filename) {
      this.hidden = false;
      this._state = { ...this._defaultState(), filename };
      this._setStatus(`loading viewport for ${filename}...`, "busy");
      this.shadowRoot.getElementById("title").textContent = `16:9 viewport — ${filename}`;
      window.addEventListener("keydown", this._keyHandler, true);
      try {
        const r = await fetch(`/api/viewport/${encodeURIComponent(filename)}`);
        if (!r.ok) {
          const err = await r.json().catch(() => ({ detail: r.statusText }));
          throw new Error(err.detail || r.statusText);
        }
        const d = await r.json();
        this._state.layout = d.layout;
        this._state.source = d.source;
        this._state.placements = d.placements || [];
        this._state.skipTiles = d.skip_tiles || [];
        this._state.inset = d.inset || { x: 0, y: 0, w: VIEWPORT_W, h: VIEWPORT_H };
        // Decode the composite for our base layer
        this._state.compositeImg = await loadImage(d.composite_b64);
        // Initialize paint buffer
        const paint = this.shadowRoot.getElementById("paintCanvas");
        this._state.paintCanvas = paint;
        this._state.paintCtx = paint.getContext("2d", { willReadFrequently: true });
        this._state.paintCtx.clearRect(0, 0, VIEWPORT_W, VIEWPORT_H);
        // Pre-fill an undo snapshot so first paint isn't lost
        this._snapshotForUndo();
        // Meta line
        const metaEl = this.shadowRoot.getElementById("meta");
        const sourceLbl = d.source === "atlas_layouts.py" ? "atlas (ground truth)" : "guessed (centered)";
        metaEl.textContent =
          `${d.viewport_w}x${d.viewport_h} - ${d.layout} layout (${sourceLbl}) - ` +
          `${d.placements.length} placement(s), ${d.skip_tiles.length} skipped`;
        this._renderBase();
        this._drawOverlay();
        this._applyZoom();
        this._setStatus(`ready — ${d.placements.length} tile placement(s)`, "ok");
      } catch (e) {
        this._setStatus(`load failed: ${e.message}`, "err");
        // Surface as toast if available
        try { if (window.psoEditor && window.psoEditor.toast) window.psoEditor.toast(`viewport load failed: ${e.message}`, "err"); } catch (_) {}
      }
    }

    close() {
      this.hidden = true;
      window.removeEventListener("keydown", this._keyHandler, true);
      this._stopAnim();
    }

    // -------------------- key handling --------------------
    _onKey(e) {
      if (this.hidden) return;
      // Allow form fields inside the toolbar to receive normal typing
      const path = e.composedPath ? e.composedPath() : [];
      const target = path[0] || e.target;
      const tag = target && target.tagName;
      const inField = tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA";
      if (e.key === "Escape") {
        e.preventDefault(); this.close(); return;
      }
      if (e.ctrlKey && (e.key === "z" || e.key === "Z")) {
        e.preventDefault();
        if (e.shiftKey) this._redo(); else this._undo();
        return;
      }
      if (e.ctrlKey && (e.key === "c" || e.key === "C")) {
        e.preventDefault(); this._copySelection(); return;
      }
      if (e.ctrlKey && (e.key === "v" || e.key === "V")) {
        e.preventDefault(); this._pasteClipboard(); return;
      }
      if (inField) return;
      // Tool shortcuts (without modifiers)
      if (!e.ctrlKey && !e.metaKey && !e.altKey) {
        const k = e.key.toLowerCase();
        const tools = { b: "brush", p: "brush", e: "eraser", f: "fill", i: "eyedropper", r: "rect", l: "line", s: "spray", m: "marquee" };
        if (tools[k]) { e.preventDefault(); this._setActiveTool(tools[k]); return; }
        if (k === "[") { e.preventDefault(); this._adjustBrush(-1); return; }
        if (k === "]") { e.preventDefault(); this._adjustBrush(+1); return; }
      }
    }

    _adjustBrush(delta) {
      const sizes = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128];
      const cur = this._state.brushSize;
      let i = sizes.indexOf(cur);
      if (i < 0) i = sizes.findIndex(s => s >= cur);
      i = clamp(i + delta, 0, sizes.length - 1);
      this._state.brushSize = sizes[i];
      this.shadowRoot.getElementById("brushSize").value = this._state.brushSize;
      this._setStatus(`brush size: ${this._state.brushSize}`);
    }

    // -------------------- mouse / drawing --------------------
    _localCoords(e) {
      const rect = this.shadowRoot.getElementById("paintCanvas").getBoundingClientRect();
      const x = (e.clientX - rect.left) * (VIEWPORT_W / rect.width);
      const y = (e.clientY - rect.top) * (VIEWPORT_H / rect.height);
      return [Math.round(x), Math.round(y)];
    }

    _onMouseDown(e) {
      if (e.button !== 0) return;
      const [x, y] = this._localCoords(e);
      const t = this._state.tool;
      if (t === "eyedropper") {
        this._eyedropAt(x, y);
        return;
      }
      if (t === "fill") {
        this._snapshotForUndo();
        this._fillAt(x, y);
        return;
      }
      this._snapshotForUndo();
      this._state.drawing = true;
      this._state.lastX = x;
      this._state.lastY = y;
      if (t === "rect" || t === "line" || t === "marquee") {
        this._state.previewShape = { x0: x, y0: y, x1: x, y1: y };
      } else if (t === "brush" || t === "eraser") {
        this._strokeStep(x, y);
      } else if (t === "spray") {
        this._sprayAt(x, y);
      }
      this._drawCursor();
    }

    _onMouseMove(e) {
      const [x, y] = this._localCoords(e);
      const cp = this.shadowRoot.getElementById("cursorPos");
      // Show pixel coords + tile placement if cursor is over one
      const ph = this._placementHit(x, y);
      cp.textContent = `(${x}, ${y})${ph ? ` over tile ${ph.tile_index}` : " in pillar"}`;
      if (!this._state.drawing) {
        this._drawCursor(x, y);
        return;
      }
      const t = this._state.tool;
      if (t === "brush" || t === "eraser") {
        this._strokeLine(this._state.lastX, this._state.lastY, x, y);
        this._state.lastX = x;
        this._state.lastY = y;
      } else if (t === "spray") {
        this._sprayAt(x, y);
        this._state.lastX = x; this._state.lastY = y;
      } else if (t === "rect" || t === "line" || t === "marquee") {
        this._state.previewShape = { x0: this._state.previewShape.x0, y0: this._state.previewShape.y0, x1: x, y1: y };
        this._drawCursor(x, y);
      }
    }

    _onMouseUp(e) {
      if (!this._state.drawing) return;
      const [x, y] = this._localCoords(e);
      const t = this._state.tool;
      if (t === "rect") this._commitRect(this._state.previewShape);
      else if (t === "line") this._commitLine(this._state.previewShape);
      else if (t === "marquee") this._commitMarquee(this._state.previewShape);
      this._state.drawing = false;
      this._state.previewShape = null;
      this._drawCursor();
    }

    _onMouseLeave() {
      // Don't end drawing on leave — the user might come back. But clear
      // hover-cursor preview.
      if (!this._state.drawing) this._drawCursor();
    }

    _placementHit(x, y) {
      for (const p of this._state.placements) {
        if (x >= p.dest_x && x < p.dest_x + p.dest_w &&
            y >= p.dest_y && y < p.dest_y + p.dest_h) return p;
      }
      return null;
    }

    // -------------------- low-level paint ops --------------------

    _strokeStep(x, y) {
      const ctx = this._state.paintCtx;
      ctx.save();
      const r = this._state.brushSize / 2;
      if (this._state.tool === "eraser") {
        ctx.globalCompositeOperation = "destination-out";
        ctx.fillStyle = "rgba(0,0,0,1)";
      } else {
        ctx.globalCompositeOperation = "source-over";
        const [r_, g_, b_, a_] = this._state.color;
        const a = a_ / 255 * this._state.opacity;
        ctx.fillStyle = `rgba(${r_}, ${g_}, ${b_}, ${a})`;
      }
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    _strokeLine(x0, y0, x1, y1) {
      // Naive Bresenham-like sampling — paint a circle at each integer step
      const dx = x1 - x0, dy = y1 - y0;
      const steps = Math.max(1, Math.ceil(Math.hypot(dx, dy)));
      for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        this._strokeStep(Math.round(x0 + dx * t), Math.round(y0 + dy * t));
      }
    }

    _sprayAt(cx, cy) {
      const ctx = this._state.paintCtx;
      ctx.save();
      ctx.globalCompositeOperation = "source-over";
      const [r_, g_, b_, a_] = this._state.color;
      const a = a_ / 255 * this._state.opacity;
      ctx.fillStyle = `rgba(${r_}, ${g_}, ${b_}, ${a})`;
      const r = this._state.brushSize;
      const N = Math.max(4, Math.round(r * r * this._state.spraydens));
      for (let i = 0; i < N; i++) {
        // uniform disk sample
        const theta = Math.random() * 2 * Math.PI;
        const rad = r * Math.sqrt(Math.random());
        const px = cx + rad * Math.cos(theta);
        const py = cy + rad * Math.sin(theta);
        ctx.fillRect(Math.round(px), Math.round(py), 1, 1);
      }
      ctx.restore();
    }

    _commitRect(shape) {
      if (!shape) return;
      const ctx = this._state.paintCtx;
      const x = Math.min(shape.x0, shape.x1);
      const y = Math.min(shape.y0, shape.y1);
      const w = Math.abs(shape.x1 - shape.x0);
      const h = Math.abs(shape.y1 - shape.y0);
      if (w < 1 || h < 1) return;
      ctx.save();
      ctx.globalCompositeOperation = "source-over";
      const [r_, g_, b_, a_] = this._state.color;
      const a = a_ / 255 * this._state.opacity;
      ctx.fillStyle = `rgba(${r_}, ${g_}, ${b_}, ${a})`;
      ctx.fillRect(x, y, w, h);
      ctx.restore();
    }

    _commitLine(shape) {
      if (!shape) return;
      this._strokeLine(shape.x0, shape.y0, shape.x1, shape.y1);
    }

    _commitMarquee(shape) {
      if (!shape) return;
      const x = Math.min(shape.x0, shape.x1);
      const y = Math.min(shape.y0, shape.y1);
      const w = Math.abs(shape.x1 - shape.x0);
      const h = Math.abs(shape.y1 - shape.y0);
      if (w < 1 || h < 1) {
        this._state.marquee = null;
        this._setStatus("selection cleared");
      } else {
        this._state.marquee = { x, y, w, h };
        this._setStatus(`selection ${w}x${h} at (${x},${y}) — Ctrl+C to copy`);
      }
      this._drawOverlay();
    }

    _fillAt(x, y) {
      const ctx = this._state.paintCtx;
      // Fill operates on the COMPOSITE (base + paint) so it behaves like
      // the user expects — clicking on a colored pixel fills its region
      // even if that color came from the source tiles. We render the
      // current visible pixels into a scratch ImageData, flood-fill it,
      // then write the diff into the paint layer.
      const baseCanvas = this.shadowRoot.getElementById("baseCanvas");
      const baseCtx = baseCanvas.getContext("2d");
      const composite = document.createElement("canvas");
      composite.width = VIEWPORT_W; composite.height = VIEWPORT_H;
      const ccx = composite.getContext("2d");
      ccx.drawImage(baseCanvas, 0, 0);
      ccx.drawImage(this._state.paintCanvas, 0, 0);
      const before = ccx.getImageData(0, 0, VIEWPORT_W, VIEWPORT_H);
      const beforeCopy = new Uint8ClampedArray(before.data);  // remember original
      floodFill(before, x, y, this._state.color, this._state.tolerance);
      // Diff: apply only changed pixels into paint layer
      const paint = ctx.getImageData(0, 0, VIEWPORT_W, VIEWPORT_H);
      const a = before.data, p = paint.data;
      let changed = 0;
      for (let i = 0; i < a.length; i += 4) {
        // pixel changed by flood fill?
        if (a[i] !== beforeCopy[i] || a[i+1] !== beforeCopy[i+1] || a[i+2] !== beforeCopy[i+2] || a[i+3] !== beforeCopy[i+3]) {
          // Apply opacity via blend with existing paint
          const op = this._state.opacity;
          if (op >= 0.999) {
            p[i] = a[i]; p[i+1] = a[i+1]; p[i+2] = a[i+2]; p[i+3] = a[i+3];
          } else {
            // Premultiplied blend
            const af = (a[i+3] / 255) * op;
            const pa = p[i+3] / 255;
            const oa = af + pa * (1 - af);
            if (oa > 0) {
              p[i]   = Math.round((a[i] * af + p[i] * pa * (1 - af)) / oa);
              p[i+1] = Math.round((a[i+1] * af + p[i+1] * pa * (1 - af)) / oa);
              p[i+2] = Math.round((a[i+2] * af + p[i+2] * pa * (1 - af)) / oa);
            }
            p[i+3] = Math.round(oa * 255);
          }
          changed++;
        }
      }
      ctx.putImageData(paint, 0, 0);
      this._setStatus(`fill: ${changed} pixels`);
    }

    _eyedropAt(x, y) {
      const baseCanvas = this.shadowRoot.getElementById("baseCanvas");
      const composite = document.createElement("canvas");
      composite.width = VIEWPORT_W; composite.height = VIEWPORT_H;
      const ccx = composite.getContext("2d");
      ccx.drawImage(baseCanvas, 0, 0);
      ccx.drawImage(this._state.paintCanvas, 0, 0);
      const px = ccx.getImageData(x, y, 1, 1).data;
      this._setColor([px[0], px[1], px[2], px[3] || 255]);
      this._setStatus(`eyedrop: rgb(${px[0]}, ${px[1]}, ${px[2]}) a=${px[3]}`);
    }

    _copySelection() {
      if (!this._state.marquee) {
        this._setStatus("no selection (use M / select then drag)", "err"); return;
      }
      const m = this._state.marquee;
      const baseCanvas = this.shadowRoot.getElementById("baseCanvas");
      const composite = document.createElement("canvas");
      composite.width = VIEWPORT_W; composite.height = VIEWPORT_H;
      const ccx = composite.getContext("2d");
      ccx.drawImage(baseCanvas, 0, 0);
      ccx.drawImage(this._state.paintCanvas, 0, 0);
      this._state.clipboard = ccx.getImageData(m.x, m.y, m.w, m.h);
      this._setStatus(`copied ${m.w}x${m.h} to clipboard`, "ok");
    }

    _pasteClipboard() {
      if (!this._state.clipboard) {
        this._setStatus("clipboard empty (Ctrl+C first)", "err"); return;
      }
      this._snapshotForUndo();
      const m = this._state.marquee;
      const x = m ? m.x : 0;
      const y = m ? m.y : 0;
      this._state.paintCtx.putImageData(this._state.clipboard, x, y);
      this._setStatus(`pasted ${this._state.clipboard.width}x${this._state.clipboard.height} at (${x},${y})`, "ok");
    }

    // -------------------- undo / redo --------------------
    _snapshotForUndo() {
      if (!this._state.paintCtx) return;
      const data = this._state.paintCtx.getImageData(0, 0, VIEWPORT_W, VIEWPORT_H);
      this._state.undoStack.push(data);
      if (this._state.undoStack.length > UNDO_DEPTH) this._state.undoStack.shift();
      this._state.redoStack.length = 0;
    }
    _undo() {
      if (this._state.undoStack.length <= 1) {
        this._setStatus("nothing to undo", "err"); return;
      }
      const cur = this._state.paintCtx.getImageData(0, 0, VIEWPORT_W, VIEWPORT_H);
      this._state.redoStack.push(cur);
      const prev = this._state.undoStack.pop();
      this._state.paintCtx.putImageData(prev, 0, 0);
      this._setStatus(`undo (${this._state.undoStack.length} left)`);
    }
    _redo() {
      if (!this._state.redoStack.length) {
        this._setStatus("nothing to redo", "err"); return;
      }
      const cur = this._state.paintCtx.getImageData(0, 0, VIEWPORT_W, VIEWPORT_H);
      this._state.undoStack.push(cur);
      const next = this._state.redoStack.pop();
      this._state.paintCtx.putImageData(next, 0, 0);
      this._setStatus(`redo (${this._state.redoStack.length} left)`);
    }

    _clearPaint() {
      this._snapshotForUndo();
      this._state.paintCtx.clearRect(0, 0, VIEWPORT_W, VIEWPORT_H);
      this._setStatus("paint cleared", "ok");
    }

    // -------------------- base + overlay rendering --------------------

    _renderBase() {
      const base = this.shadowRoot.getElementById("baseCanvas");
      const ctx = base.getContext("2d");
      ctx.clearRect(0, 0, VIEWPORT_W, VIEWPORT_H);
      if (this._state.showAsGame) {
        // Draw the assembled composite as the game would render it.
        if (this._state.compositeImg) ctx.drawImage(this._state.compositeImg, 0, 0);
      } else {
        // Show source tiles on a dark grid so the user can see which
        // region maps to which tile_index. We draw the placement borders
        // bright and label each region.
        const inset = this._state.inset;
        ctx.fillStyle = "#0a0e13";
        ctx.fillRect(0, 0, VIEWPORT_W, VIEWPORT_H);
        // Light fill in the inset to delineate 4:3
        ctx.fillStyle = "rgba(15, 25, 35, 1)";
        ctx.fillRect(inset.x, inset.y, inset.w, inset.h);
        if (this._state.compositeImg) {
          ctx.globalAlpha = 0.5;
          ctx.drawImage(this._state.compositeImg, 0, 0);
          ctx.globalAlpha = 1;
        }
        // Tile borders
        ctx.lineWidth = 2;
        ctx.font = "12px ui-monospace, Consolas, monospace";
        for (const p of this._state.placements) {
          ctx.strokeStyle = "#00ffff";
          ctx.strokeRect(p.dest_x + 0.5, p.dest_y + 0.5, p.dest_w - 1, p.dest_h - 1);
          ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
          const lbl = `tile ${p.tile_index} (${p.dest_w}x${p.dest_h})`;
          const m = ctx.measureText(lbl);
          ctx.fillRect(p.dest_x + 4, p.dest_y + 4, m.width + 8, 16);
          ctx.fillStyle = "#00ffff";
          ctx.fillText(lbl, p.dest_x + 8, p.dest_y + 16);
        }
      }
    }

    _drawOverlay() {
      const ctx = this.shadowRoot.getElementById("overlayCanvas").getContext("2d");
      ctx.clearRect(0, 0, VIEWPORT_W, VIEWPORT_H);
      if (!this._state.showOverlay) return;
      const inset = this._state.inset;
      // Dotted hatching outside the 4:3 inset to mark the pillar /
      // letterbox regions. Subtle so it doesn't fight the painting.
      ctx.save();
      ctx.strokeStyle = "rgba(255, 170, 0, 0.4)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      // Top + bottom letterbox (if inset doesn't cover full height)
      if (inset.y > 0) {
        ctx.strokeRect(0.5, 0.5, VIEWPORT_W - 1, inset.y);
      }
      const bottomBand = VIEWPORT_H - (inset.y + inset.h);
      if (bottomBand > 0) {
        ctx.strokeRect(0.5, inset.y + inset.h + 0.5, VIEWPORT_W - 1, bottomBand - 1);
      }
      // Left + right pillars
      if (inset.x > 0) {
        ctx.strokeRect(0.5, inset.y + 0.5, inset.x, inset.h - 1);
      }
      const rightPillar = VIEWPORT_W - (inset.x + inset.w);
      if (rightPillar > 0) {
        ctx.strokeRect(inset.x + inset.w + 0.5, inset.y + 0.5, rightPillar - 1, inset.h - 1);
      }
      // Inset border (4:3 frame)
      ctx.setLineDash([]);
      ctx.strokeStyle = "rgba(0, 255, 255, 0.3)";
      ctx.strokeRect(inset.x + 0.5, inset.y + 0.5, inset.w - 1, inset.h - 1);
      // Marquee selection if any
      if (this._state.marquee) {
        const m = this._state.marquee;
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = "#ffaa00";
        ctx.lineWidth = 2;
        ctx.strokeRect(m.x + 0.5, m.y + 0.5, m.w - 1, m.h - 1);
      }
      ctx.restore();
    }

    _drawCursor(hx, hy) {
      const ctx = this.shadowRoot.getElementById("cursorCanvas").getContext("2d");
      ctx.clearRect(0, 0, VIEWPORT_W, VIEWPORT_H);
      // Brush size preview
      if ((this._state.tool === "brush" || this._state.tool === "eraser" || this._state.tool === "spray") &&
          typeof hx === "number" && typeof hy === "number") {
        ctx.save();
        ctx.strokeStyle = this._state.tool === "eraser" ? "rgba(255,170,0,0.7)" : "rgba(0,255,255,0.7)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(hx, hy, this._state.brushSize / 2, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }
      // Shape preview
      if (this._state.previewShape) {
        const s = this._state.previewShape;
        ctx.save();
        ctx.strokeStyle = "rgba(255, 170, 0, 0.8)";
        ctx.setLineDash([4, 3]);
        ctx.lineWidth = 1;
        if (this._state.tool === "rect" || this._state.tool === "marquee") {
          const x = Math.min(s.x0, s.x1), y = Math.min(s.y0, s.y1);
          const w = Math.abs(s.x1 - s.x0), h = Math.abs(s.y1 - s.y0);
          ctx.strokeRect(x + 0.5, y + 0.5, w, h);
        } else if (this._state.tool === "line") {
          ctx.beginPath();
          ctx.moveTo(s.x0, s.y0);
          ctx.lineTo(s.x1, s.y1);
          ctx.stroke();
        }
        ctx.restore();
      }
    }

    // -------------------- animation --------------------
    _restartAnim() {
      this._stopAnim();
      if (!this._state.animEnabled) return;
      // Look up frame configs for this file's tiles. The animation
      // feature stores per-tile {grid, fps, order} in localStorage; we
      // expose that here as a v2 path.
      let configs = {};
      try {
        if (window.psoEditor && typeof window.psoEditor.getAnimConfigs === "function") {
          configs = window.psoEditor.getAnimConfigs(this._state.filename) || {};
        }
      } catch (_) { /* best-effort */ }
      const hasAny = Object.keys(configs).length > 0;
      if (!hasAny) {
        this._setStatus("animate: no per-tile frame grids registered for this file (set via the modal animate panel)", "err");
        const el = this.shadowRoot.getElementById("animToggle");
        if (el) el.checked = false;
        this._state.animEnabled = false;
        return;
      }
      this._state.animFrame = 0;
      this._state.animTimer = setInterval(() => {
        this._state.animFrame++;
        this._renderAnim(configs);
      }, 1000 / Math.max(1, this._state.animFps));
    }

    _stopAnim() {
      if (this._state.animTimer) {
        clearInterval(this._state.animTimer);
        this._state.animTimer = null;
      }
    }

    _renderAnim(configs) {
      // For each placement that has an anim config, slice the configured
      // frame from its source and redraw into the base layer at the
      // placement rect. This is best-effort; the v1 base composite is
      // already rendered, we just overdraw the animating placements.
      // Falls back to no-op if anything is missing.
      // (Implementation kept minimal — full tile-source data isn't on
      // the wire today; tagged as v2.)
      this._setStatus(`anim frame ${this._state.animFrame}`);
    }

    // -------------------- save --------------------
    async _save() {
      if (this._state.busy) return;
      if (!this._state.filename) { this._setStatus("no file open", "err"); return; }
      this._state.busy = true;
      this._setStatus("flattening + posting...", "busy");
      try {
        // Composite base + paint into one PNG
        const flat = document.createElement("canvas");
        flat.width = VIEWPORT_W; flat.height = VIEWPORT_H;
        const fctx = flat.getContext("2d");
        if (this._state.compositeImg) fctx.drawImage(this._state.compositeImg, 0, 0);
        fctx.drawImage(this._state.paintCanvas, 0, 0);
        const dataUrl = flat.toDataURL("image/png");
        const r = await fetch("/api/viewport_paint", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            filename: this._state.filename,
            viewport_png_b64: dataUrl,
            viewport_w: VIEWPORT_W,
            viewport_h: VIEWPORT_H,
          }),
        });
        if (!r.ok) {
          const err = await r.json().catch(() => ({ detail: r.statusText }));
          throw new Error(err.detail || r.statusText);
        }
        const rsp = await r.json();
        // Hand off to app.js for state.tileEdits registration
        if (window.psoEditor && typeof window.psoEditor.applyViewportResult === "function") {
          window.psoEditor.applyViewportResult(rsp);
        }
        this._setStatus(
          `saved: ${rsp.tiles_modified.length} tile edit(s) staged (skipped: ${rsp.skipped.length})`,
          "ok"
        );
      } catch (e) {
        this._setStatus(`save failed: ${e.message}`, "err");
        try { if (window.psoEditor && window.psoEditor.toast) window.psoEditor.toast(`viewport save failed: ${e.message}`, "err", { ttl: 7000 }); } catch (_) {}
      } finally {
        this._state.busy = false;
      }
    }
  }

  customElements.define("pso-viewport-canvas", PsoViewportCanvas);

  // ---------- mount + wire to header button ----------
  function mount() {
    if (document.querySelector("pso-viewport-canvas")) return;
    const el = document.createElement("pso-viewport-canvas");
    el.hidden = true;
    document.body.appendChild(el);
    // Header button
    const btn = document.getElementById("btnViewport");
    if (btn) {
      btn.addEventListener("click", () => {
        const f = window.psoEditor && window.psoEditor.state &&
                  window.psoEditor.state.currentFile &&
                  window.psoEditor.state.currentFile.name;
        if (!f) {
          // Hint via status bar; mirrors model_viewer.js's pattern
          const status = document.getElementById("status");
          if (status) {
            status.textContent = "open a file first, then click viewport";
            status.className = "status err";
          }
          return;
        }
        el.open(f);
      });
    }
    // Expose on window for testing / external triggers
    window.psoViewport = el;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mount);
  } else {
    mount();
  }
})();
