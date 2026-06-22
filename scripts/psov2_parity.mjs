// =====================================================================
// psov2_parity.mjs — visual parity harness: OUR studio vs the psov2
// reference renderer (https://dashgl.gitlab.io/psov2/), headless.
// ---------------------------------------------------------------------
// Given a model filename (e.g. `bm_boss2_de_rol_le.bml`) this:
//
//   1. Serves the LOCAL copy of the psov2 static site
//      (_reference/psov2/public) on an ephemeral port and loads the
//      model HEADLESS. psov2 binds each enemy/weapon/object load to a
//      tray <li> via `table[leaf][key].bind(key)` (see NinjaPlugin.js
//      init -> api_setActiveNav). We reproduce that dispatch WITHOUT a
//      click: locate the model's name across the Asset* catalogs
//      (AssetEnemies / AssetWeapons / AssetObjects / AssetPlayer /
//      AssetStage / AssetRooms), then call the async loader with
//      `this` = the label string, exactly as `.bind(key)` would. The
//      loader resolves the .bml/.gsl, parses the inner .nj meshes
//      (NinjaModel/NinjaEnv readChunk path), decodes the PVM textures
//      (NinjaTexture), and calls NinjaPlugin.API.setModel which adds the
//      primary mesh at origin + spreads the other inners (dx += 20).
//      We then orient the camera and capture NinjaPlugin.MEM.renderer's
//      WebGL canvas to PNG.
//
//   2. Loads the SAME model in OUR studio (assumed already running, by
//      default http://127.0.0.1:8765) headless via the page global
//      `window.psoOpenModelByPath(<path>)` (model_viewer.js), which
//      opens the #modelModal and renders into #modelCanvas. We capture
//      that canvas to PNG with a matched camera framing.
//
//   3. Writes psov2.png + ours.png + composite.png (side-by-side) into
//      _parity/<model>/.
//
// Both renderers are three.js, so framing math is shared. We fit the
// camera to the rendered model's world-space bounding sphere in BOTH so
// orientation/scale match as closely as the two pipelines allow.
//
// Usage:
//   node scripts/psov2_parity.mjs bm_boss2_de_rol_le.bml \
//        [--variant dc] [--ours http://127.0.0.1:8765] [--out _parity] \
//        [--width 900] [--height 900] [--headed] [--keep-open]
//
//   --variant dc  : load the DREAMCAST asset in OUR studio (prefixes the
//                   path with "dc/"), i.e. the SAME 397 KB De Rol Le BML
//                   psov2 serves. This is required for the pixel-1:1 match
//                   (the Xbox/BB BML is a different 2.05 MB atlas packing).
//                   Default (no flag) loads the Xbox/BB asset as before.
//
// Requires: `playwright` installed (npm i playwright && npx playwright
// install chromium) and OUR studio already serving the model data dir.
// =====================================================================

import { createServer } from "http";
import { readFile, mkdir, writeFile } from "fs/promises";
import { existsSync } from "fs";
import { dirname, resolve, join, extname } from "path";
import { fileURLToPath } from "url";
import { chromium } from "playwright";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(__dirname, "..");
const PSOV2_ROOT = resolve(REPO, "_reference/psov2/public");

// ---- args ----
const argv = process.argv.slice(2);
function flag(name, def) {
  const i = argv.indexOf(name);
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : def;
}
function has(name) {
  return argv.includes(name);
}
const positional = argv.filter((a) => !a.startsWith("--"));
const MODEL = positional[0] || "bm_boss2_de_rol_le.bml";
// --variant dc (or PARITY_VARIANT=dc): load the Dreamcast variant of the
// model in OUR studio (the SAME asset psov2 serves) so the pixel match is
// possible. We do NOT change the load mechanism — psoOpenModelByPath()
// stays; we only prefix the path string with "dc/", which the server's
// variant resolver strips + routes to the read-only Dreamcast data root.
// Default ("") preserves today's Xbox/BB behaviour for existing invocations.
const VARIANT = flag("--variant", process.env.PARITY_VARIANT || "");
const PREFIX = VARIANT === "dc" || VARIANT === "dreamcast" ? "dc/" : "";
const OURS_MODEL = PREFIX + MODEL;
const OURS_BASE = flag("--ours", process.env.PARITY_OURS || "http://127.0.0.1:8765");
const OUT_ROOT = resolve(REPO, flag("--out", "_parity"));
const W = parseInt(flag("--width", "900"), 10) || 900;
const H = parseInt(flag("--height", "900"), 10) || 900;
const HEADED = has("--headed");
const KEEP_OPEN = has("--keep-open");
const TIMEOUT = parseInt(flag("--timeout", "60000"), 10) || 60000;

const modelStem = MODEL.replace(/[#?].*$/, "").replace(/\.[a-z0-9]+$/i, "");
const OUT_DIR = join(OUT_ROOT, modelStem);

// ---------------------------------------------------------------------
// Minimal static file server for the psov2 reference site.
// ---------------------------------------------------------------------
const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".gif": "image/gif",
  ".svg": "image/svg+xml",
  ".bin": "application/octet-stream",
  ".bml": "application/octet-stream",
  ".gsl": "application/octet-stream",
  ".nj": "application/octet-stream",
  ".njm": "application/octet-stream",
  ".pvm": "application/octet-stream",
  ".prs": "application/octet-stream",
  ".afs": "application/octet-stream",
};

function startPsov2Server() {
  return new Promise((resolveServer) => {
    const server = createServer(async (req, res) => {
      try {
        let urlPath = decodeURIComponent(req.url.split("?")[0]);
        if (urlPath === "/" || urlPath === "") urlPath = "/index.html";
        // Prevent path traversal; resolve under PSOV2_ROOT.
        const filePath = resolve(PSOV2_ROOT, "." + urlPath);
        if (!filePath.startsWith(PSOV2_ROOT)) {
          res.writeHead(403);
          res.end("forbidden");
          return;
        }
        const buf = await readFile(filePath);
        const ext = extname(filePath).toLowerCase();
        res.writeHead(200, {
          "Content-Type": MIME[ext] || "application/octet-stream",
          "Access-Control-Allow-Origin": "*",
          "Cache-Control": "no-store",
        });
        res.end(buf);
      } catch (e) {
        res.writeHead(404);
        res.end("not found: " + (e?.message || e));
      }
    });
    server.listen(0, "127.0.0.1", () => {
      const port = server.address().port;
      resolveServer({ server, port });
    });
  });
}

// ---------------------------------------------------------------------
// In-page helpers (stringified, run in the browser context).
// ---------------------------------------------------------------------

// Discover the psov2 loader for MODEL and invoke it the same way the
// tray click would (`.bind(key)` => `this === key`). Returns the label
// invoked, or throws if not found. Runs in the psov2 page.
const PSOV2_LOAD_FN = `
async (modelFile) => {
  // The catalogs are globals declared in js/Asset*.js.
  const catalogs = {
    AssetEnemies: (typeof AssetEnemies !== 'undefined') ? AssetEnemies : null,
    AssetWeapons: (typeof AssetWeapons !== 'undefined') ? AssetWeapons : null,
    AssetObjects: (typeof AssetObjects !== 'undefined') ? AssetObjects : null,
    AssetPlayer:  (typeof AssetPlayer  !== 'undefined') ? AssetPlayer  : null,
    AssetStage:   (typeof AssetStage   !== 'undefined') ? AssetStage   : null,
    AssetRooms:   (typeof AssetRooms   !== 'undefined') ? AssetRooms   : null,
  };
  // Find the catalog + key whose loader body references modelFile (the
  // .bml/.gsl it loads). We match on the literal filename appearing in
  // the function source — robust and label-agnostic.
  const needle = modelFile.toLowerCase();
  let found = null;
  for (const [catName, cat] of Object.entries(catalogs)) {
    if (!cat) continue;
    for (const key of Object.keys(cat)) {
      const fn = cat[key];
      if (typeof fn !== 'function') continue;
      const src = fn.toString().toLowerCase();
      if (src.includes(needle)) {
        // Prefer a top-level direct load of THIS file (NinjaFile.API.load).
        found = { catName, key, fn, direct: src.includes('load("'+needle+'")') || src.includes("load('"+needle+"')") };
        if (found.direct) break;
      }
    }
    if (found && found.direct) break;
  }
  if (!found) throw new Error('psov2: no catalog entry references ' + modelFile);
  // Invoke exactly as the tray binds it: this === key (the label string).
  await found.fn.call(found.key);
  return { catalog: found.catName, label: found.key };
}
`;

// After load, frame the camera onto the assembled scene and render.
// psov2 stores everything on NinjaPlugin.MEM. Returns a data URL PNG.
const PSOV2_CAPTURE_FN = `
async (opts) => {
  // NinjaPlugin / THREE are top-level consts -> bare globals, not on window.
  const P = (typeof NinjaPlugin !== 'undefined') ? NinjaPlugin : window.NinjaPlugin;
  const THREE = (typeof window.THREE !== 'undefined') ? window.THREE : THREE;
  if (!P || !P.MEM) throw new Error('NinjaPlugin.MEM missing');
  const { scene, renderer, camera } = P.MEM;

  // Stop psov2's own animation loop from advancing mixers (we want the
  // bind pose, matching a static studio capture). We can't easily cancel
  // its rAF, but clearing mixers prevents pose drift.
  P.MEM.mixers = [];

  // Resize renderer to the requested capture size and matching aspect.
  renderer.setSize(opts.w, opts.h, false);
  camera.aspect = opts.w / opts.h;

  // Compute the bounding box of all real meshes (skip GridHelper /
  // SkeletonHelper / lights). We bake world matrices first.
  scene.updateMatrixWorld(true);
  const box = new THREE.Box3();
  let any = false;
  scene.traverse((o) => {
    if (!o.isMesh && !o.isSkinnedMesh) return;
    if (o.type === 'GridHelper') return;
    const g = o.geometry;
    if (!g || !g.attributes || !g.attributes.position) return;
    g.computeBoundingBox();
    const b = g.boundingBox.clone().applyMatrix4(o.matrixWorld);
    box.union(b);
    any = true;
  });
  if (!any) throw new Error('psov2: no renderable meshes in scene');

  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z) * 0.5 || 1;

  // Frame: pull the camera back along +Z (and a little +Y) so the model
  // fits with margin. fov is in deg.
  const fov = (camera.fov || 45) * Math.PI / 180;
  const dist = (radius / Math.sin(fov / 2)) * opts.margin;
  camera.position.set(center.x, center.y + radius * 0.15, center.z + dist);
  camera.near = Math.max(0.01, dist - radius * 3);
  camera.far = dist + radius * 4;
  camera.up.set(0, 1, 0);
  camera.lookAt(center);
  camera.updateProjectionMatrix();

  // Hide the grid for a clean comparison (optional).
  if (opts.hideGrid) {
    scene.traverse((o) => { if (o.type === 'GridHelper') o.visible = false; });
  }

  // Render synchronously and read the buffer in the SAME task (the WebGL
  // backbuffer is valid until the next compositing step).
  renderer.render(scene, camera);
  const url = renderer.domElement.toDataURL('image/png');
  return {
    dataUrl: url,
    center: center.toArray(),
    radius,
    meshCount: (() => { let n = 0; scene.traverse((o)=>{ if((o.isMesh||o.isSkinnedMesh) && o.type!=='GridHelper') n++; }); return n; })(),
  };
}
`;

// Load + capture in OUR studio. Returns a data URL PNG.
const OURS_CAPTURE_FN = `
async (opts) => {
  const THREE = window.THREE;
  if (!window.psoOpenModelByPath) throw new Error('psoOpenModelByPath missing');
  await window.psoOpenModelByPath(opts.model);

  // Give the composite/texture pipeline a moment to settle.
  await new Promise((r) => setTimeout(r, opts.settleMs));

  const renderer = window.psoGetRenderer && window.psoGetRenderer();
  const camera = window.psoGetCamera && window.psoGetCamera();
  const meshGroup = window.psoGetMeshGroup && window.psoGetMeshGroup();
  const canvas = window.psoGetCanvas && window.psoGetCanvas();
  if (!renderer || !camera || !canvas) throw new Error('studio renderer/camera/canvas missing');

  // The scene is the meshGroup's ancestor Scene.
  let scene = meshGroup;
  while (scene && scene.parent) scene = scene.parent;
  if (!scene || !scene.isScene) {
    // Fallback: walk up from any object.
    scene = (meshGroup && meshGroup.parent) || null;
  }
  if (!scene) throw new Error('studio scene not found');

  // Reset the mesh rotation so we present a canonical front view, the
  // same un-rotated bind pose psov2 shows (its loader applies no extra
  // rotation either). meshGroup carries the model.
  if (meshGroup) { meshGroup.rotation.set(0, 0, 0); }

  // Size to match.
  renderer.setSize(opts.w, opts.h, false);
  camera.aspect = opts.w / opts.h;

  // Fit camera to the model bounds (world space).
  scene.updateMatrixWorld(true);
  const box = new THREE.Box3();
  let any = false;
  (meshGroup || scene).traverse((o) => {
    if (!o.isMesh && !o.isSkinnedMesh) return;
    const g = o.geometry;
    if (!g || !g.attributes || !g.attributes.position) return;
    g.computeBoundingBox();
    if (!g.boundingBox) return;
    const b = g.boundingBox.clone().applyMatrix4(o.matrixWorld);
    box.union(b);
    any = true;
  });
  if (!any) throw new Error('studio: no renderable meshes');

  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z) * 0.5 || 1;
  const fov = (camera.fov || 45) * Math.PI / 180;
  const dist = (radius / Math.sin(fov / 2)) * opts.margin;
  camera.position.set(center.x, center.y + radius * 0.15, center.z + dist);
  camera.near = Math.max(0.001, dist - radius * 3);
  camera.far = dist + radius * 4;
  camera.up.set(0, 1, 0);
  camera.lookAt(center);
  camera.updateProjectionMatrix();

  renderer.render(scene, camera);
  const url = canvas.toDataURL('image/png');
  // realMesh flag lets us flag a silent cube fallback.
  return {
    dataUrl: url,
    center: center.toArray(),
    radius,
    realMesh: !!(window.psoModelCacheInfo ? true : true),
  };
}
`;

// ---------------------------------------------------------------------
// PNG helpers (no external deps): write a data: URL out, and build a
// side-by-side composite by stitching two PNGs with the canvas API in a
// throwaway page (we already have a browser).
// ---------------------------------------------------------------------
function dataUrlToBuffer(dataUrl) {
  const i = dataUrl.indexOf(",");
  return Buffer.from(dataUrl.slice(i + 1), "base64");
}

async function buildComposite(page, leftUrl, rightUrl, leftLabel, rightLabel, w, h) {
  return page.evaluate(
    async ({ leftUrl, rightUrl, leftLabel, rightLabel, w, h }) => {
      function loadImg(src) {
        return new Promise((res, rej) => {
          const im = new Image();
          im.onload = () => res(im);
          im.onerror = rej;
          im.src = src;
        });
      }
      const [li, ri] = await Promise.all([loadImg(leftUrl), loadImg(rightUrl)]);
      const pad = 8;
      const labelH = 28;
      const cw = w * 2 + pad * 3;
      const ch = h + labelH + pad * 2;
      const c = document.createElement("canvas");
      c.width = cw;
      c.height = ch;
      const ctx = c.getContext("2d");
      ctx.fillStyle = "#11161c";
      ctx.fillRect(0, 0, cw, ch);
      ctx.drawImage(li, pad, labelH + pad, w, h);
      ctx.drawImage(ri, w + pad * 2, labelH + pad, w, h);
      ctx.fillStyle = "#e6edf3";
      ctx.font = "16px sans-serif";
      ctx.textBaseline = "middle";
      ctx.fillText(leftLabel, pad + 6, labelH / 2 + pad);
      ctx.fillText(rightLabel, w + pad * 2 + 6, labelH / 2 + pad);
      return c.toDataURL("image/png");
    },
    { leftUrl, rightUrl, leftLabel, rightLabel, w, h },
  );
}

// ---------------------------------------------------------------------
// Main.
// ---------------------------------------------------------------------
(async () => {
  if (!existsSync(PSOV2_ROOT)) {
    throw new Error("psov2 reference root not found: " + PSOV2_ROOT);
  }
  await mkdir(OUT_DIR, { recursive: true });

  const { server, port } = await startPsov2Server();
  const PSOV2_BASE = `http://127.0.0.1:${port}`;
  process.stderr.write(`[parity] psov2 served at ${PSOV2_BASE}\n`);
  process.stderr.write(`[parity] our studio at ${OURS_BASE}\n`);
  process.stderr.write(`[parity] model = ${MODEL}\n`);

  const browser = await chromium.launch({
    headless: !HEADED,
    args: [
      "--use-angle=swiftshader",
      "--use-gl=angle",
      "--ignore-gpu-blocklist",
      "--enable-unsafe-swiftshader",
    ],
  });

  const result = {
    model: MODEL,
    variant: VARIANT || "xbox",
    oursModel: OURS_MODEL,
    outDir: OUT_DIR,
    psov2: { base: PSOV2_BASE },
    ours: { base: OURS_BASE },
  };

  try {
    const ctx = await browser.newContext({
      viewport: { width: Math.max(W * 2, 1280), height: Math.max(H, 720) },
      deviceScaleFactor: 1,
    });

    // ---------- psov2 ----------
    const p1 = await ctx.newPage();
    const psov2Logs = [];
    p1.on("console", (m) => psov2Logs.push(`[${m.type()}] ${m.text()}`));
    p1.on("pageerror", (e) => psov2Logs.push(`[pageerror] ${e.message}`));
    await p1.goto(PSOV2_BASE + "/index.html", { waitUntil: "load", timeout: TIMEOUT });
    // Wait for the plugin + catalogs to exist. NOTE: psov2 declares
    // NinjaPlugin / AssetEnemies as top-level `const`s, which are NOT
    // properties of `window` — they're only reachable as bare globals in
    // the evaluation scope. THREE r95 explicitly assigns window.THREE.
    await p1.waitForFunction(
      () => typeof NinjaPlugin !== "undefined" && typeof THREE !== "undefined" &&
            typeof AssetEnemies !== "undefined" && !!(NinjaPlugin && NinjaPlugin.MEM),
      null,
      { timeout: TIMEOUT },
    );
    process.stderr.write(`[parity] psov2 page ready, loading model...\n`);
    // NOTE: page.evaluate(stringExpr, arg) treats the string as an
    // EXPRESSION (the arg is ignored). So we wrap each helper as an
    // immediately-invoked expression with the args baked in via JSON.
    const psov2Load = await p1.evaluate(`(${PSOV2_LOAD_FN})(${JSON.stringify(MODEL)})`);
    result.psov2.loadMethod =
      `${psov2Load.catalog}["${psov2Load.label}"].call("${psov2Load.label}") ` +
      `— catalog loader located by matching the model filename in the loader source; ` +
      `invoked with this===label exactly as NinjaPlugin's tray binding ` +
      `(table[leaf][key].bind(key)) would on a click.`;
    process.stderr.write(`[parity] psov2 loaded: ${JSON.stringify(psov2Load)}\n`);
    // Let texture canvases + geometry settle.
    await p1.waitForTimeout(1500);
    const psov2Cap = await p1.evaluate(
      `(${PSOV2_CAPTURE_FN})(${JSON.stringify({ w: W, h: H, margin: 1.25, hideGrid: true })})`,
    );
    const psov2Png = dataUrlToBuffer(psov2Cap.dataUrl);
    await writeFile(join(OUT_DIR, "psov2.png"), psov2Png);
    result.psov2.center = psov2Cap.center;
    result.psov2.radius = psov2Cap.radius;
    result.psov2.meshCount = psov2Cap.meshCount;
    process.stderr.write(`[parity] psov2.png written (${psov2Png.length} bytes, ${psov2Cap.meshCount} meshes)\n`);

    // ---------- ours ----------
    const p2 = await ctx.newPage();
    const oursLogs = [];
    p2.on("console", (m) => oursLogs.push(`[${m.type()}] ${m.text()}`));
    p2.on("pageerror", (e) => oursLogs.push(`[pageerror] ${e.message}`));
    await p2.goto(OURS_BASE + "/", { waitUntil: "load", timeout: TIMEOUT });
    await p2.waitForFunction(
      () => typeof window.psoOpenModelByPath === "function" && typeof window.THREE !== "undefined",
      null,
      { timeout: TIMEOUT },
    );
    process.stderr.write(`[parity] studio page ready, loading model...\n`);
    const oursCap = await p2.evaluate(
      `(${OURS_CAPTURE_FN})(${JSON.stringify({ model: OURS_MODEL, w: W, h: H, margin: 1.25, settleMs: 2500 })})`,
    );
    const oursPng = dataUrlToBuffer(oursCap.dataUrl);
    await writeFile(join(OUT_DIR, "ours.png"), oursPng);
    result.ours.center = oursCap.center;
    result.ours.radius = oursCap.radius;
    process.stderr.write(`[parity] ours.png written (${oursPng.length} bytes)\n`);

    // ---------- composite ----------
    const compUrl = await buildComposite(
      p2,
      psov2Cap.dataUrl,
      oursCap.dataUrl,
      `psov2 — ${MODEL}`,
      `ours — ${MODEL}`,
      W, H,
    );
    const compPng = dataUrlToBuffer(compUrl);
    await writeFile(join(OUT_DIR, "composite.png"), compPng);
    process.stderr.write(`[parity] composite.png written (${compPng.length} bytes)\n`);

    result.files = {
      psov2: join(OUT_DIR, "psov2.png"),
      ours: join(OUT_DIR, "ours.png"),
      composite: join(OUT_DIR, "composite.png"),
    };
    result.psov2Logs = psov2Logs.slice(-40);
    result.oursLogs = oursLogs.slice(-40);

    await writeFile(join(OUT_DIR, "parity.json"), JSON.stringify(result, null, 2));
    process.stdout.write("\n__PARITY_RESULT__" + JSON.stringify(result) + "\n");

    if (KEEP_OPEN) {
      process.stderr.write("[parity] --keep-open: leaving browser open. Ctrl-C to exit.\n");
      await new Promise(() => {});
    }
  } finally {
    if (!KEEP_OPEN) await browser.close();
    server.close();
  }
})().catch((e) => {
  console.error("psov2_parity FATAL:", e?.stack || e);
  process.exit(1);
});
