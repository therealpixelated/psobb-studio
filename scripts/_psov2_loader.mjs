// Shared loader: import the REAL static/psov2_ninja.js parseNinjaModel
// under Node using the REAL three@0.160.0 package (the same version the
// browser loads from unpkg). The frontend module imports THREE from a
// unpkg URL; we rewrite that single import to the locally-installed
// `three` package, then evaluate the (otherwise byte-identical) source as
// an ES module via a data: URL. Nothing about the parse math is altered —
// only the THREE *binding source* changes, exactly mirroring the test
// harness pattern in tests/test_autoplay_jsdom.mjs but with real THREE so
// the parser produces real BufferGeometry (position.count) instead of a
// mock that always reports 0 verts.

import { readFileSync } from "fs";
import { dirname, resolve } from "path";
import { fileURLToPath } from "url";
import { pathToFileURL } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(__dirname, "..");

// Resolve the local three package to an absolute file: URL so the rewritten
// import works regardless of CWD.
const threeUrl = pathToFileURL(
  resolve(REPO, "node_modules/three/build/three.module.js"),
).href;

export async function loadParser() {
  const src = readFileSync(resolve(REPO, "static/psov2_ninja.js"), "utf8");
  // Swap the unpkg THREE import for the local package. The import shape in
  // the source is exactly: import * as THREE from "https://unpkg.com/...";
  const patched = src.replace(
    /import \* as THREE from "[^"]*three\.module\.js";/,
    `import * as THREE from ${JSON.stringify(threeUrl)};`,
  );
  if (patched === src) {
    throw new Error("psov2_ninja.js THREE import not found/rewritten");
  }
  const dataUrl =
    "data:text/javascript;base64," + Buffer.from(patched, "utf8").toString("base64");
  const mod = await import(dataUrl);
  if (typeof mod.parseNinjaModel !== "function") {
    throw new Error("parseNinjaModel export missing after rewrite");
  }
  return mod;
}
