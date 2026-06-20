"""Visual viewport-rendering smoke harness (v4 visual polish, 2026-04-25).

The audit + coverage agents in this repo verify the wire format end-to-
end (server-side parsers round-trip, manifest hashes match, /api/*
endpoints return the right shapes). What they do NOT verify is whether
the resulting bytes actually paint pixels in the browser — that's been
left to manual Preview MCP validation. This file fills the gap with a
headless-Chromium harness that:

  1. Boots ``server:app`` on a random localhost port.
  2. Navigates a Playwright page to each MVP authoring panel.
  3. Captures a PNG screenshot of the active viewport.
  4. Sanity-asserts non-zero non-black content (>= some threshold of
     pixel diversity) so a totally white / totally black render fails
     loudly.
  5. Saves the screenshot under ``_reports/visual_smoke/`` and emits
     a Markdown report at ``_reports/visual_smoke.md``.

Gated by a ``visual`` pytest marker so CI / smoke runs can opt out
of the slow path:

  pytest -m visual                 # run only visual tests
  pytest -m "not visual"           # skip them

Skips entirely if Playwright + chromium aren't installable. Other
v4 tasks must ship even when this harness is unavailable.
"""
from __future__ import annotations

import io
import socket
import sys
import threading
import time
from contextlib import closing
from pathlib import Path

import pytest


# ---------------------------------------------------------------------
# Optional-import gating
# ---------------------------------------------------------------------

try:
    from playwright.sync_api import sync_playwright, Browser, Page
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


pytestmark = [
    pytest.mark.visual,
    pytest.mark.skipif(
        not HAS_PLAYWRIGHT,
        reason=(
            "playwright not installed — pip install playwright && "
            "playwright install chromium"
        ),
    ),
    pytest.mark.skipif(
        not HAS_PIL, reason="PIL needed for screenshot pixel-diversity assertion",
    ),
]


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "_reports" / "visual_smoke"


# ---------------------------------------------------------------------
# Live-server fixture — uvicorn on a random port, scoped session-wide.
# ---------------------------------------------------------------------

def _free_port() -> int:
    """Bind-and-release to find a free port. Race-prone but cheap."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def live_server():
    """Boot server.app on a thread; teardown cleanly on session end.

    Skips if the host environment can't keep up (e.g. server.py raises
    at import time because some optional dep is missing). Visual smoke
    is allowed to be best-effort.
    """
    try:
        import server  # noqa: F401  (import-side-effect tested)
        import uvicorn
    except Exception as e:  # pragma: no cover - infra
        pytest.skip(f"server import failed: {e}")

    port = _free_port()
    url = f"http://127.0.0.1:{port}"

    config = uvicorn.Config(
        server.app,
        host="127.0.0.1",
        port=port,
        log_level="error",
        # Strict no-lifespan if it's optional — the cache pre-warm
        # lifespans in this repo can take 5-10s on cold caches; we
        # don't need those for visual screenshots.
        lifespan="off",
    )
    server_instance = uvicorn.Server(config)

    thread = threading.Thread(target=server_instance.run, daemon=True)
    thread.start()

    # Wait for the server to start accepting connections (up to 10s).
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            try:
                s.connect(("127.0.0.1", port))
                break
            except OSError:
                time.sleep(0.1)
    else:
        server_instance.should_exit = True
        pytest.skip("server did not start within 10s")

    yield url

    server_instance.should_exit = True
    thread.join(timeout=5.0)


# ---------------------------------------------------------------------
# Playwright browser fixture — launched once per session.
# ---------------------------------------------------------------------

@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        yield b
        b.close()


@pytest.fixture
def page(browser):
    """Fresh Page per test so localStorage / cookies don't leak."""
    ctx = browser.new_context(
        viewport={"width": 1280, "height": 800},
        # Larger device-pixel-ratio for crisper screenshots without
        # blowing up the file size — 1.0 keeps PNGs at the viewport
        # size which is what diff comparison expects.
        device_scale_factor=1.0,
    )
    # Pre-dismiss the first-run onboarding walkthrough: its modal overlay
    # (#obOverlay) intercepts pointer events, so panel clicks would time out.
    # Runs before any page script, so onboarding.js sees the flag and stays closed.
    ctx.add_init_script(
        "try { localStorage.setItem('pso.onboarding.seen.v1', '1'); } catch (e) {}"
    )
    pg = ctx.new_page()
    yield pg
    ctx.close()


# ---------------------------------------------------------------------
# Pixel-diversity helper — fail loudly on all-black / all-white.
# ---------------------------------------------------------------------

def _png_pixel_diversity(png_bytes: bytes) -> dict:
    """Return summary stats about a PNG screenshot.

    A blank / black render has zero variance; a real render has
    thousands of distinct colours and a non-trivial standard-deviation
    in luminance. We use both signals because some legitimate panels
    can be very dark (the model viewer's 0x0a0e13 background) and
    flagged false-positive on luminance alone.

    Returns:
        {
            "size":           (w, h),
            "unique_colors":  int,
            "stddev_lum":     float,
            "is_solid_black": bool,
            "is_solid_white": bool,
            "looks_real":     bool,   # the "did anything render?" check
        }
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    w, h = img.size
    pixels = img.getdata()

    # Sample every Nth pixel to keep this O(1)-ish across resolutions.
    step = max(1, len(pixels) // 16384)
    sampled = list(pixels)[::step]
    seen = set(sampled)

    sum_l = 0.0
    sum_l2 = 0.0
    n = 0
    for (r, g, b) in sampled:
        # Rec.601 luminance — fine for solid-fill detection.
        l = 0.299 * r + 0.587 * g + 0.114 * b
        sum_l += l
        sum_l2 += l * l
        n += 1
    if n == 0:
        return {
            "size": (w, h), "unique_colors": 0, "stddev_lum": 0.0,
            "is_solid_black": True, "is_solid_white": False,
            "looks_real": False,
        }
    mean = sum_l / n
    var = max(0.0, sum_l2 / n - mean * mean)
    stddev = var ** 0.5

    is_solid_black = mean < 5.0 and stddev < 1.0
    is_solid_white = mean > 250.0 and stddev < 1.0

    # "Looks real": > 16 unique colours OR > 4 stddev_lum. Either
    # signal is enough — a panel could be 99% one solid colour with
    # one button text in another (low unique colors, low stddev still)
    # and we'd want to call that a render.
    looks_real = (len(seen) > 16) or (stddev > 4.0)

    return {
        "size":           (w, h),
        "unique_colors":  len(seen),
        "stddev_lum":     stddev,
        "is_solid_black": is_solid_black,
        "is_solid_white": is_solid_white,
        "looks_real":     looks_real,
    }


# ---------------------------------------------------------------------
# Per-panel scenarios.
#
# Each scenario is a tuple of (slug, description, setup_fn). The setup
# function receives an open Playwright Page already navigated to "/" and
# returns once the panel is ready for screenshot. Keep them small and
# resilient — visual smoke is a "did anything render?" tripwire, not a
# functional regression battery.
# ---------------------------------------------------------------------

def _scenario_index_page(page: "Page") -> str:
    """Just the editor's empty index — verifies static assets load."""
    page.wait_for_selector("h1", timeout=5000)
    page.wait_for_timeout(500)  # let the asset router boot
    return "Editor index (header + asset tree skeleton)"


def _scenario_battle_params(page: "Page") -> str:
    """Open the Battle Params perspective via the header button."""
    btn = page.locator("#btnBattleParams")
    btn.wait_for(timeout=5000)
    btn.click()
    page.wait_for_timeout(800)
    return "Battle Params perspective (mob stats editor)"


def _scenario_item_pmt(page: "Page") -> str:
    """Open the Item PMT perspective.

    Note: the panel registers itself with id="btnItemPmt" (lowercase 't')
    not "btnItemPMT" — naming inconsistency from the original panel
    file.
    """
    btn = page.locator("#btnItemPmt")
    btn.wait_for(timeout=5000)
    btn.click()
    page.wait_for_timeout(800)
    return "Item PMT perspective (weapons / armors / units)"


def _scenario_mob_ai(page: "Page") -> str:
    """Open the Mob AI authoring perspective."""
    btn = page.locator("#btnMobAi")
    btn.wait_for(timeout=5000)
    btn.click()
    page.wait_for_timeout(800)
    return "Mob AI authoring perspective"


def _scenario_map_editor(page: "Page") -> str:
    """Open the Map Editor perspective."""
    btn = page.locator("#btnMapEditor")
    btn.wait_for(timeout=5000)
    btn.click()
    page.wait_for_timeout(800)
    return "Map Editor perspective (terrain + spawns)"


SCENARIOS = [
    ("index",         _scenario_index_page),
    ("battle_params", _scenario_battle_params),
    ("item_pmt",      _scenario_item_pmt),
    ("mob_ai",        _scenario_mob_ai),
    ("map_editor",    _scenario_map_editor),
]


# ---------------------------------------------------------------------
# Tests + report writer
# ---------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _ensure_reports_dir():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@pytest.mark.parametrize("slug,setup_fn", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_panel_renders_non_blank(slug, setup_fn, live_server, page, request):
    """Open <panel>, screenshot, assert non-blank pixel content."""
    page.goto(live_server, wait_until="domcontentloaded", timeout=15000)
    description = setup_fn(page)

    img_bytes = page.screenshot(type="png", full_page=False)
    out_path = REPORTS_DIR / f"{slug}.png"
    out_path.write_bytes(img_bytes)

    stats = _png_pixel_diversity(img_bytes)

    # Stash for the markdown report writer.
    request.session._visual_smoke_results = getattr(
        request.session, "_visual_smoke_results", []
    )
    request.session._visual_smoke_results.append({
        "slug": slug,
        "description": description,
        "out_path": out_path,
        "stats": stats,
    })

    # The actual smoke assertion: did the page render anything that
    # isn't a solid colour?
    assert not stats["is_solid_black"], \
        f"{slug}: solid black screenshot — likely JS load failure or empty perspective"
    assert not stats["is_solid_white"], \
        f"{slug}: solid white screenshot"
    assert stats["looks_real"], \
        f"{slug}: not enough pixel diversity — unique={stats['unique_colors']}, stddev_lum={stats['stddev_lum']:.2f}"


@pytest.fixture(scope="session", autouse=True)
def _write_visual_smoke_report(request):
    """At session-end, emit ``_reports/visual_smoke.md`` with embedded screenshots."""
    yield
    results = getattr(request.session, "_visual_smoke_results", None)
    if not results:
        return
    md_path = REPO_ROOT / "_reports" / "visual_smoke.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Visual smoke harness report",
        "",
        f"Generated by ``tests/test_visual_smoke.py`` on {time.strftime('%Y-%m-%d %H:%M:%S')}.",
        "",
        "Each row screenshots one MVP authoring panel and asserts it",
        "rendered non-blank, non-solid-colour content. The harness is",
        "Playwright + headless Chromium against a transient uvicorn",
        "server bound to a random localhost port.",
        "",
        "## Coverage",
        "",
        "| Panel | Description | Unique colors | Std-dev lum | Status | Screenshot |",
        "|---|---|---:|---:|---|---|",
    ]
    for r in results:
        s = r["stats"]
        status = "PASS" if s["looks_real"] and not s["is_solid_black"] and not s["is_solid_white"] else "FAIL"
        rel_png = r["out_path"].relative_to(md_path.parent).as_posix()
        lines.append(
            f"| `{r['slug']}` | {r['description']} | "
            f"{s['unique_colors']} | {s['stddev_lum']:.2f} | "
            f"{status} | ![{r['slug']}]({rel_png}) |"
        )
    lines.append("")
    lines.append("## Re-running")
    lines.append("")
    lines.append("```")
    lines.append("pytest -m visual tests/test_visual_smoke.py")
    lines.append("```")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
