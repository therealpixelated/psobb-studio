#!/usr/bin/env python3
"""Automated visual demo — drive the live UI and assemble docs/demo.gif.

What this does
--------------
Boots ``server:app`` on a random localhost port (the same mechanism
``tests/test_visual_smoke.py`` uses), drives a headless Chromium browser
through a scripted walkthrough of the studio, screenshots each "beat", and
stitches the frames into an animated GIF with Pillow (no ffmpeg needed).

Beats (each is skipped gracefully if its data is absent — never crash):
  1. Home / asset browser with the category pills.
  2. A model open in the 3D viewer.
  3. A texture / tile view.
  4. The Floor editor (list + preview).
  5. The Audio panel (codec badge + waveform).

Outputs land in ``docs/``::

    docs/demo.gif        the assembled animation (looping)
    docs/shot_<n>_<slug>.png   one PNG per captured beat

Running it
----------
    python scripts/demo_capture.py                 # uses $PSO_DATA_DIR or ~/PSOBB.IO/data
    python scripts/demo_capture.py --data-dir D:/Games/PSOBB/data
    make demo                                       # same, via the Makefile

Bare-checkout safety
--------------------
If Playwright / Chromium isn't installed, OR no PSOBB ``data/`` directory is
present, the script prints a clear "skipping demo capture" message and exits 0
so it never fails a fresh clone or CI. It captures real frames only on a
machine that actually has game assets to show.

The frames are deterministic given the same data dir: assets are chosen by a
stable rule (a named candidate if present, else the first manifest entry of the
right category), the browser uses a fixed viewport, and animation is paused
before each shot.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
from contextlib import closing
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"

# Capture geometry. 960x600 keeps the GIF small while staying legible; the
# walkthrough panels are designed to be readable at this size.
VIEWPORT = {"width": 1100, "height": 700}
# Final GIF frame size (downscaled from the PNG captures to bound file bytes).
GIF_WIDTH = 760
GIF_FRAME_MS = 1900  # per beat
GIF_COLORS = 128     # adaptive palette size — lower = smaller file

# Neutral placeholder shown in place of any real filesystem path.
_PATH_PLACEHOLDER = "…/PSOBB/data"

# Injected into every page (via add_init_script) BEFORE app scripts run. It
# installs a MutationObserver that rewrites any absolute Windows/UNC/Unix path
# the moment it appears in the DOM, so the async data-dir UI writes can never
# leave a home path / username on screen when a screenshot is flushed. This is
# a privacy guard for the PUBLIC repo, not cosmetics.
_PATH_SCRUB_OBSERVER_JS = r"""
(() => {
  const PLACEHOLDER = '…/PSOBB/data';
  const pathRe = /([A-Za-z]:[\\/][^\s"'<>]+|\\\\[^\s"'<>]+|\/(?:home|Users)\/[^\s"'<>]+)/g;
  const scrubNode = (node) => {
    if (node.nodeType === Node.TEXT_NODE) {
      if (node.nodeValue && pathRe.test(node.nodeValue)) {
        pathRe.lastIndex = 0;
        node.nodeValue = node.nodeValue.replace(pathRe, PLACEHOLDER);
      }
      pathRe.lastIndex = 0;
    } else if (node.nodeType === Node.ELEMENT_NODE) {
      if (node.title && pathRe.test(node.title)) {
        pathRe.lastIndex = 0;
        node.title = node.title.replace(pathRe, PLACEHOLDER);
      }
      pathRe.lastIndex = 0;
      for (const c of node.childNodes) scrubNode(c);
    }
  };
  const run = () => { try { if (document.body) scrubNode(document.body); } catch (e) {} };
  const start = () => {
    run();
    try {
      const obs = new MutationObserver(() => run());
      obs.observe(document.documentElement, {
        subtree: true, childList: true, characterData: true, attributes: true,
        attributeFilter: ['title'],
      });
    } catch (e) {}
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})()
"""


# ---------------------------------------------------------------------------
# Optional-dependency gating — degrade to a clean skip, never a crash.
# ---------------------------------------------------------------------------

def _have_playwright() -> bool:
    try:
        import playwright.sync_api  # noqa: F401
        return True
    except Exception:
        return False


def _have_pillow() -> bool:
    try:
        import PIL  # noqa: F401
        return True
    except Exception:
        return False


def _resolve_data_dir(cli_data_dir: str | None) -> Path | None:
    """Return a usable PSOBB data dir, or None if there's nothing to show."""
    candidates = []
    if cli_data_dir:
        candidates.append(cli_data_dir)
    if os.environ.get("PSO_DATA_DIR"):
        candidates.append(os.environ["PSO_DATA_DIR"])
    candidates.append(os.path.expanduser("~/PSOBB.IO/data"))
    for c in candidates:
        p = Path(c).expanduser()
        if p.is_dir() and any(p.iterdir()):
            return p
    return None


# ---------------------------------------------------------------------------
# Live-server boot (mirrors tests/test_visual_smoke.py::live_server).
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ServerHandle:
    def __init__(self, instance, thread, url):
        self.instance = instance
        self.thread = thread
        self.url = url

    def stop(self) -> None:
        self.instance.should_exit = True
        self.thread.join(timeout=5.0)


def _boot_server(data_dir: Path) -> _ServerHandle | None:
    """Start uvicorn on a thread; return a handle or None if it won't start."""
    os.environ["PSO_DATA_DIR"] = str(data_dir)
    sys.path.insert(0, str(REPO_ROOT))
    # Run from the repo root so ``server:app`` and ``static/`` resolve.
    os.chdir(REPO_ROOT)
    try:
        import server  # noqa: F401
        import uvicorn
    except Exception as e:  # pragma: no cover - infra
        print(f"  server import failed: {e}")
        return None

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    config = uvicorn.Config(
        server.app, host="127.0.0.1", port=port, log_level="error",
        lifespan="off",  # skip the cold-cache prewarm; we just need pages
    )
    instance = uvicorn.Server(config)
    thread = threading.Thread(target=instance.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            try:
                s.connect(("127.0.0.1", port))
                return _ServerHandle(instance, thread, url)
            except OSError:
                time.sleep(0.1)
    instance.should_exit = True
    print("  server did not start within 20s")
    return None


# ---------------------------------------------------------------------------
# Manifest-driven asset selection (deterministic on a given data dir).
# ---------------------------------------------------------------------------

def _fetch_manifest(url: str) -> list[dict]:
    import urllib.request
    import json
    try:
        with urllib.request.urlopen(f"{url}/api/manifest_lite", timeout=60) as r:
            data = json.loads(r.read())
        return data.get("entries", []) if isinstance(data, dict) else []
    except Exception as e:
        print(f"  manifest fetch failed: {e}")
        return []


def _pick(entries: list[dict], category: str, prefer: list[str],
          avoid: tuple[str, ...] = ()) -> str | None:
    """Pick a stable asset path of ``category``: a preferred name if present,
    else the first manifest entry of that category whose filename contains
    none of ``avoid`` (used to skip dark/transparent textures that render as
    empty-looking tiles), else the first of any."""
    by_cat = [e for e in entries if e.get("category") == category]
    paths = [e.get("path", "") for e in by_cat]
    for want in prefer:
        for p in paths:
            if p.split("/")[-1].lower() == want.lower():
                return p
    if avoid:
        for p in paths:
            name = p.split("/")[-1].lower()
            if not any(a in name for a in avoid):
                return p
    return paths[0] if paths else None


def _audio_bare_root(entries: list[dict], prefer: list[str]) -> str | None:
    """The audio endpoints resolve a BARE filename directly under the data
    root (no path components). Most music lives in sub-dirs (``ogg/``,
    ``sound/``) and is therefore not drivable through the audio panel; the
    top-level intro movie usually is. Pick the first audio entry whose path
    has no slash (resolvable), preferring named candidates."""
    bare = [e.get("path", "") for e in entries
            if e.get("category") == "audio" and "/" not in e.get("path", "")]
    for want in prefer:
        for p in bare:
            if p.lower() == want.lower():
                return p
    return bare[0] if bare else None


# ---------------------------------------------------------------------------
# Beat scripts. Each returns True if it captured something, False to skip.
# All driving goes through the SAME public hooks the UI itself uses:
#   window.bus.emit("asset.opened", {path, entry})  -> perspectives router
#   click #btnFloorEditor                            -> floor perspective
# ---------------------------------------------------------------------------

def _sanitize_for_screenshot(page) -> None:
    """Scrub any absolute filesystem path from the DOM before a screenshot.

    THIS IS A SECURITY/PRIVACY STEP, not cosmetic. The committed GIF + PNGs go
    into a PUBLIC repo, and the studio prints the active data dir (which on a
    real machine contains a home path + username) into the header ``#dataDir``
    span and the onboarding ``#obDataDir`` card. We replace those with a
    neutral placeholder so no home path / username is ever baked into a
    committed image. Idempotent — safe to call before every shot.
    """
    page.evaluate(
        r"""() => {
            const PLACEHOLDER = '…/PSOBB/data';
            // Known path-bearing elements.
            for (const sel of ['#dataDir', '#obDataDir']) {
                const el = document.querySelector(sel);
                if (el) { el.textContent = PLACEHOLDER; el.title = PLACEHOLDER; }
            }
            // Belt-and-braces: rewrite any visible text node that still looks
            // like an absolute Windows/UNC/Unix path (drive letter, \\server,
            // or /home|/Users). Keeps captions like 'D:/Games/PSOBB/data' in
            // the onboarding HINT (those are example placeholders) untouched
            // only if they are inside <code> with the example marker class;
            // everything else gets neutralised.
            const pathRe = /([A-Za-z]:[\\/][^\s"'<>]+|\\\\[^\s"'<>]+|\/(?:home|Users)\/[^\s"'<>]+)/g;
            const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
            const hits = [];
            let n;
            while ((n = walker.nextNode())) {
                if (pathRe.test(n.nodeValue)) hits.push(n);
                pathRe.lastIndex = 0;
            }
            for (const t of hits) {
                t.nodeValue = t.nodeValue.replace(pathRe, PLACEHOLDER);
            }
        }"""
    )


def _open_asset(page, path: str, category: str, extra_entry: dict | None = None) -> None:
    entry = {"category": category, "path": path}
    if extra_entry:
        entry.update(extra_entry)
    page.evaluate(
        """([path, entry]) => {
            if (window.bus && window.bus.emit) {
                window.bus.emit('asset.opened', { path: path, entry: entry });
            }
        }""",
        [path, entry],
    )


def _pause_3d(page) -> None:
    """Stop auto-rotation so the model shot is stable / comparable."""
    try:
        page.evaluate(
            """() => {
                const cb = document.getElementById('modelAutoRotate');
                if (cb && cb.checked) { cb.checked = false;
                    cb.dispatchEvent(new Event('change', {bubbles:true})); }
            }"""
        )
    except Exception:
        pass


def beat_home(page, url, assets) -> bool:
    page.goto(url, wait_until="domcontentloaded", timeout=20000)
    page.wait_for_selector("h1", timeout=8000)
    # Let the asset tree + category pills hydrate.
    page.wait_for_timeout(1500)
    return True


def beat_model(page, url, assets) -> bool:
    path = assets.get("model")
    if not path:
        print("  [model] no model asset — skipping beat")
        return False
    _open_asset(page, path, "model")
    # Boss BMLs decode a lot of geometry + textures; give the mesh fetch and
    # the first Three.js paint room before pausing rotation for a stable shot.
    page.wait_for_timeout(5200)
    _pause_3d(page)
    page.wait_for_timeout(1200)
    return True


def beat_texture(page, url, assets) -> bool:
    path = assets.get("texture")
    if not path:
        print("  [texture] no texture asset — skipping beat")
        return False
    _open_asset(page, path, "texture")
    page.wait_for_timeout(2600)   # tile grid render
    return True


def beat_floor(page, url, assets) -> bool:
    btn = page.locator("#btnFloorEditor")
    try:
        btn.wait_for(timeout=4000)
    except Exception:
        print("  [floor] Floor Editor button not present — skipping beat")
        return False
    btn.click()
    # The floor list loads; a floor is pre-selected. Click its Preview button
    # so the terrain actually renders in the shared 3D canvas for the shot.
    page.wait_for_timeout(2200)
    try:
        prev = page.locator("#floorBtnPreview")
        if prev.count():
            prev.click()
    except Exception:
        pass
    page.wait_for_timeout(4200)   # terrain mesh fetch + paint
    _pause_3d(page)
    page.wait_for_timeout(1200)
    return True


def beat_audio(page, url, assets) -> bool:
    path = assets.get("audio")
    if not path:
        print("  [audio] no root-resolvable audio asset — skipping beat")
        return False
    _open_asset(page, path, "audio")
    page.wait_for_timeout(2800)   # info fetch + waveform paint
    return True


BEATS = [
    ("home",    "Asset browser — every file in the install, classified", beat_home),
    ("model",   "3D model viewer — decoded NJ/XJ mesh + textures",       beat_model),
    ("texture", "Texture tiles — XVR/PVR decoded to editable PNG",        beat_texture),
    ("floor",   "Floor editor — browse / copy / build floors (dev-only)", beat_floor),
    ("audio",   "Audio suite — codec badge + waveform",                   beat_audio),
]


# ---------------------------------------------------------------------------
# GIF assembly (Pillow only).
# ---------------------------------------------------------------------------

def _assemble_gif(png_paths: list[Path], out_path: Path) -> tuple[int, int]:
    """Stitch PNGs into a looping GIF. Returns (frame_count, byte_size)."""
    from PIL import Image

    frames = []
    for p in png_paths:
        img = Image.open(p).convert("RGB")
        # Downscale to bound file size while keeping aspect ratio.
        if img.width > GIF_WIDTH:
            h = round(img.height * GIF_WIDTH / img.width)
            img = img.resize((GIF_WIDTH, h), Image.LANCZOS)
        # Quantize to an adaptive palette — much smaller than full-color GIF.
        frames.append(img.quantize(colors=GIF_COLORS, method=Image.FASTOCTREE))

    if not frames:
        return 0, 0

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=GIF_FRAME_MS,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return len(frames), out_path.stat().st_size


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------

def run(data_dir: Path) -> int:
    from playwright.sync_api import sync_playwright

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"== demo capture (data dir present: {data_dir.name}) ==")
    handle = _boot_server(data_dir)
    if handle is None:
        print("SKIP: server would not boot — committing nothing.")
        return 0

    captured: list[tuple[str, Path]] = []
    try:
        entries = _fetch_manifest(handle.url)
        assets = {
            "model": _pick(entries, "model",
                           ["bm_boss1_dragon.bml", "bm_boss5_gryphon.bml",
                            "bm_boss7_crawfish.bml"]),
            # Prefer a clearly-visible texture for the demo: dark/transparent
            # ones (boss "core", effects) render as empty-looking tiles.
            "texture": _pick(entries, "texture",
                             ["f512_hunters.xvm", "ccconsole_j.xvm"],
                             avoid=("effect", "_nt", "core_tex", "fog",
                                    "shadow", "smoke", "indtex", "indirect")),
            "audio": _audio_bare_root(entries, ["opening_j.sfd"]),
        }
        print(f"  chosen assets: "
              f"model={assets['model']!r} texture={assets['texture']!r} "
              f"audio={assets['audio']!r}")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=1.0)
            # Pre-dismiss the first-run onboarding overlay (same trick as the
            # visual-smoke harness) — its modal would intercept pointer events.
            ctx.add_init_script(
                "try { localStorage.setItem('pso.onboarding.seen.v1', '1'); } catch (e) {}"
            )
            # Continuous path scrubber. app.js / onboarding.js populate the
            # data-dir UI from an async /api/health fetch whose .then() can
            # fire AFTER a one-shot scrub but before the screenshot flushes.
            # A MutationObserver re-neutralises any home path the instant it's
            # written, so NO absolute path is ever on screen at capture time.
            ctx.add_init_script(_PATH_SCRUB_OBSERVER_JS)
            page = ctx.new_page()

            idx = 0
            for slug, caption, fn in BEATS:
                try:
                    ok = fn(page, handle.url, assets)
                except Exception as e:
                    print(f"  [{slug}] beat raised ({e}) — skipping")
                    ok = False
                if not ok:
                    continue
                idx += 1
                _sanitize_for_screenshot(page)  # scrub home path before every shot
                shot = DOCS_DIR / f"shot_{idx}_{slug}.png"
                page.screenshot(path=str(shot), full_page=False)
                print(f"  captured beat {idx}: {slug} -> {shot.name}")
                captured.append((slug, shot))

            ctx.close()
            browser.close()
    finally:
        handle.stop()

    if not captured:
        print("SKIP: no beats captured (no showable assets present).")
        return 0

    gif_path = DOCS_DIR / "demo.gif"
    n_frames, size = _assemble_gif([p for _, p in captured], gif_path)
    print(f"== wrote {gif_path.relative_to(REPO_ROOT).as_posix()}: "
          f"{n_frames} frames, {size/1024:.0f} KB ==")
    if size > 8 * 1024 * 1024:
        print("  WARNING: GIF exceeds 8 MB — consider fewer frames / lower GIF_COLORS.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="demo_capture",
        description="Capture an automated visual demo GIF of psobb-studio.",
    )
    ap.add_argument("--data-dir", default=None,
                    help="PSOBB data dir (default $PSO_DATA_DIR or ~/PSOBB.IO/data)")
    args = ap.parse_args(argv)

    if not _have_playwright():
        print("SKIP: Playwright not installed "
              "(pip install playwright && playwright install chromium). "
              "No demo captured; this is fine on a bare checkout.")
        return 0
    if not _have_pillow():
        print("SKIP: Pillow not installed (pip install pillow). No demo captured.")
        return 0

    data_dir = _resolve_data_dir(args.data_dir)
    if data_dir is None:
        print("SKIP: no PSOBB data present "
              "(set PSO_DATA_DIR or --data-dir to a populated install). "
              "No demo captured; this is fine on a bare checkout.")
        return 0

    return run(data_dir)


if __name__ == "__main__":
    sys.exit(main())
