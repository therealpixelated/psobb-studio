"""Per-worker isolation for the tile-PNG disk cache.

When pytest-xdist runs ``tests/test_tile_png_cache.py`` in parallel,
every worker imports the same ``server`` module and therefore shares
the same ``cache/tile_png/v1/`` directory. The autouse fixture in
``test_tile_png_cache.py`` clears that directory between tests, so
worker A wipes worker B's atomic-rename .png.tmp mid-write and
``os.replace`` raises ``WinError 32`` ("file in use"). The L2 disk
write silently fails, the next test sees fewer cached entries than
expected, and assertions like ``s["misses"] == tile_count`` fail
non-deterministically.

The fix: route every worker's disk cache to its own subdir before
``server`` is imported. We do that by setting
``PSO_TILE_PNG_CACHE_DIR`` early enough that the module-import-time
``TILE_PNG_CACHE_DIR = ...`` line picks it up. ``server.py`` was
extended to honor that env var; default behaviour is unchanged for
non-test runs (falls back to ``CACHE_DIR / "tile_png"``).

This file lives at ``tests/conftest.py`` so pytest auto-discovers it
without any registration. Sequential pytest runs are also safe — the
worker id is ``master``, which still gets its own subdir but doesn't
collide with anyone.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def _resolve_worker_id() -> str:
    """Return pytest-xdist's worker id (gw0/gw1/...) or ``master``."""
    # pytest-xdist sets PYTEST_XDIST_WORKER for each subprocess. Vanilla
    # pytest leaves it unset, in which case "master" is fine — we still
    # get isolation from the production cache (multiple sequential
    # pytest invocations share one dir, which is the desired behaviour:
    # disk-warm tests stay disk-warm across `pytest -k ... && pytest`).
    return os.environ.get("PYTEST_XDIST_WORKER") or "master"


def _set_per_worker_tile_png_cache_dir() -> Path:
    """Override ``PSO_TILE_PNG_CACHE_DIR`` so server picks a unique path.

    We deliberately use ``cache/tile_png_test/<worker>`` under the repo
    root rather than the OS tempdir because:
      - it's visible / inspectable when a test fails;
      - cleanup happens automatically because the directory is wiped at
        the start of every test session;
      - on Windows we avoid spurious Defender quarantine on tempdirs.

    Always overrides — pytest-xdist propagates the env var from the
    controller to its workers, so a "set once" check would leave every
    worker pointing at the controller's path. We MUST re-resolve per
    worker based on PYTEST_XDIST_WORKER.
    """
    repo_root = Path(__file__).resolve().parent.parent
    base = repo_root / "cache" / "tile_png_test"
    worker = _resolve_worker_id()
    cache_dir = base / worker
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PSO_TILE_PNG_CACHE_DIR"] = str(cache_dir)
    return cache_dir


# IMPORTANT: this MUST run before ``import server`` happens anywhere in
# any test. Conftest.py is imported by pytest before any test module,
# so this is the right hook. The first ``import server`` (e.g. from
# test_tile_png_cache's ``srv`` fixture) will then read the env var.
_TILE_PNG_TEST_DIR = _set_per_worker_tile_png_cache_dir()


# Wipe the per-worker dir at session start so leftovers from a prior
# run can't masquerade as L2 hits in a fresh session. Each test in
# test_tile_png_cache.py also calls _tile_png_cache_clear in its
# autouse fixture, but that only nukes files under TILE_PNG_CACHE_DIR
# — which is now per-worker, so no cross-talk.
try:
    if _TILE_PNG_TEST_DIR.is_dir():
        for child in _TILE_PNG_TEST_DIR.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink()
            except OSError:
                pass
except OSError:
    pass


# Register custom pytest markers so ``pytest -m visual`` (and the
# inverse ``pytest -m "not visual"``) doesn't trigger an unknown-marker
# warning. ``visual`` gates the Playwright-driven smoke harness in
# tests/test_visual_smoke.py — opt-in because it spins up a real
# server + a headless Chromium.
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "visual: visual smoke harness (Playwright + headless Chromium); "
        "skipped by default unless run with '-m visual'.",
    )
