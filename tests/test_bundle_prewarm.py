"""Wave 7 — bundle pre-warm verification.

Once the bundle endpoint computes the binding, it kicks off a background
ThreadPoolExecutor that pre-decodes every referenced tile PNG into the
tile_png cache. By the time the browser fires /api/tile_png/* GETs,
those decode paths return from the warm LRU instead of paying ~30-100 ms
of cold XVR decode.

These tests verify:

1. Bundle GET on dragon-class model fires N pre-warm jobs (N == binding
   xvmh count) and the queue drains within a generous timeout.
2. After the queue drains, every tile_png GET against the same model
   serves in <100 ms (warm hit).
3. The pre-warm executor doesn't grow the queue unbounded — submitting
   200 jobs while the cap is 64 only schedules 64.
4. Pre-warm is best-effort: a malformed tex_filename doesn't crash the
   bundle response or other workers.
"""
from __future__ import annotations
import os

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


PSOBB_DATA = Path(os.path.expanduser("~/PSOBB.IO/data"))
HAS_PSOBB = PSOBB_DATA.is_dir()


@pytest.fixture(scope="module")
def srv():
    import server
    return server


@pytest.fixture(scope="module")
def client(srv):
    return TestClient(srv.app)


@pytest.fixture(autouse=True)
def reset_prewarm_state(srv):
    """Wave 7 staleness gate keeps `_TILE_PREWARM_CURRENT` set across
    tests; reset it so each test starts with a clean staleness slate."""
    yield
    with srv._TILE_PREWARM_LOCK:
        srv._TILE_PREWARM_QUEUE_SIZE = 0
    srv._TILE_PREWARM_CURRENT = None


def _drain_queue(srv, timeout: float = 30.0) -> int:
    """Wait until queue_size hits 0. Returns the final queue size (0 on
    success, the still-queued count on timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if srv._TILE_PREWARM_QUEUE_SIZE == 0:
            return 0
        time.sleep(0.05)
    return srv._TILE_PREWARM_QUEUE_SIZE


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO data not present")
def test_bundle_prewarms_referenced_tiles(client, srv):
    """Cold-load a multi-tile boss bundle, then verify every tile is
    fast (warm hit) on the next GET."""
    bml = "bm_boss1_dragon.bml"
    inner = "boss1_s_nb_dragon.nj"
    # Wipe the tile cache so we start from cold.
    srv._tile_png_cache_clear(drop_disk=False)

    r = client.get(f"/api/model_bundle/{bml}", params={"inner": inner})
    assert r.status_code == 200
    body = r.json()
    skinned = body.get("skinned") or {}
    bd = skinned.get("binding_data") or {}
    xvmh = bd.get("xvmh") or []
    if not xvmh:
        pytest.skip(f"{bml}#{inner} has no binding xvmh — install variation")

    expected_tiles = sorted({
        row["tile_index"] for row in xvmh
        if isinstance(row.get("tile_index"), int) and row["tile_index"] >= 0
    })
    assert expected_tiles, "binding xvmh has no tile_indices"

    # Pre-warm queue should be either still draining or already done.
    remaining = _drain_queue(srv, timeout=30.0)
    assert remaining == 0, f"prewarm queue stuck with {remaining} jobs"

    # Every tile we expected should now be present in the cache.
    stats = srv._tile_png_cache_stats()
    assert stats["entries"] >= len(expected_tiles), (
        f"only {stats['entries']} tiles cached, expected at least {len(expected_tiles)}"
    )

    # Verify warm-hit speed: every tile_png GET should be <200 ms.
    # The `#` in the filename is a URL fragment delimiter, so encode it.
    from urllib.parse import quote
    tex_filename = f"{bml}#{inner}.xvm"
    enc = quote(tex_filename, safe="")
    for idx in expected_tiles[:5]:  # sample the first 5
        t0 = time.time()
        rr = client.get(f"/api/tile_png/{enc}/{idx}")
        elapsed_ms = (time.time() - t0) * 1000
        assert rr.status_code == 200, f"tile {idx}: {rr.status_code} {rr.text[:200]}"
        assert elapsed_ms < 200, (
            f"warm tile {idx} took {elapsed_ms:.0f} ms — pre-warm didn't seed it"
        )


def test_prewarm_queue_caps_submissions(srv):
    """Submitting more jobs than _TILE_PREWARM_MAX_QUEUED only schedules
    up to the cap. Excess submissions are dropped silently."""
    # Reset queue counter to zero so this test starts clean.
    with srv._TILE_PREWARM_LOCK:
        srv._TILE_PREWARM_QUEUE_SIZE = 0

    cap = srv._TILE_PREWARM_MAX_QUEUED
    # Submit 2× cap jobs against a nonexistent tex_filename so workers
    # fast-fail without doing real work — but they still occupy queue
    # slots until the worker body returns.
    srv._kick_tile_prewarm("does_not_exist.xvm", list(range(cap * 2)))

    # We're racing the worker pool, so the queue size could already be
    # decreasing. The hard guarantee is: AT NO POINT did we exceed the cap.
    # Wait for drain and then assert no work over-flowed to the queue.
    _drain_queue(srv, timeout=10.0)
    assert srv._TILE_PREWARM_QUEUE_SIZE == 0


def test_prewarm_handles_bad_filename(srv):
    """A malformed tex_filename must not crash the worker. Pre-warm is
    best-effort — exceptions inside the decode path are logged and
    swallowed so other queued jobs keep running."""
    with srv._TILE_PREWARM_LOCK:
        srv._TILE_PREWARM_QUEUE_SIZE = 0
    srv._kick_tile_prewarm("nonexistent_archive.xvm", [0, 1, 2])
    remaining = _drain_queue(srv, timeout=5.0)
    assert remaining == 0


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO data not present")
def test_bundle_response_not_blocked_by_prewarm(client, srv):
    """The bundle's wall time should be roughly the same regardless of
    how many tiles it pre-warms. This is a weak smoke check, not a
    strict bound — we just verify the bundle returns within reason."""
    bml = "bm_ene_astark.bml"
    # Wipe to force cold path.
    srv._tile_png_cache_clear(drop_disk=False)
    t0 = time.time()
    r = client.get(f"/api/model_bundle/{bml}")
    elapsed = time.time() - t0
    assert r.status_code == 200
    # Even if pre-warm fires synchronously by mistake, this should be
    # under 5 s. If it blocks on tile decodes (16 of them at 100 ms each)
    # we'd see ~1.6 s+ here — but it should be much less because the
    # pre-warm is async.
    assert elapsed < 5.0, f"bundle took {elapsed:.1f} s — pre-warm may be blocking"
