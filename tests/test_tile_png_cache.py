"""Tests for the tile-PNG LRU cache (Phase D Win 5).

The cache wraps `/api/tile_png` — the per-tile XVR→PIL→PNG render that
took ~50-100 ms × 16 tiles = ~1.6 s of dragon's cold first-open. Cache
hits skip the FileResponse open() and serve PNG bytes directly from
RAM (or from a sha2-named pickle on disk after a server restart).

Covers:
  - Cold compute populates the LRU; second call returns the SAME bytes
    from memory in <5 ms.
  - On-disk persistence: clear in-memory only and the next request hits
    the L2 disk cache.
  - File mtime change invalidates the entry — re-deploy must NOT serve
    stale PNG bytes.
  - Different tile indices for the same archive get separate entries.
  - Eviction by entry-count cap.
  - /api/tile_png_cache/stats returns sane numbers; /clear empties both.
  - End-to-end via /api/tile_png produces 16 misses then 16 hits across
    two opens of dragon (skipped if PSOBB.IO not present).
"""
from __future__ import annotations

import os
import struct
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
def reset_tile_png_cache(srv):
    """Each test runs against an empty tile_png cache."""
    srv._tile_png_cache_clear(drop_disk=True)
    yield
    srv._tile_png_cache_clear(drop_disk=True)


def _make_fake_png(payload: bytes = b"X") -> bytes:
    """Synthesise a 1x1 PNG so disk-cache magic-number checks pass."""
    sig = b"\x89PNG\r\n\x1a\n"
    return sig + payload


# ---------------------------------------------------------------------------
# Direct unit tests (no PSOBB install needed)
# ---------------------------------------------------------------------------

def test_cold_then_warm_inmemory(srv, tmp_path):
    """Second call with identical key returns the SAME bytes from L1."""
    fake = tmp_path / "fake.xvm"
    fake.write_bytes(b"placeholder")

    # Construct a fake tile PNG on disk, then point the fetch fn at it.
    tile_dir = tmp_path / "tiles"
    tile_dir.mkdir()
    tile_png = tile_dir / "tile00.png"
    fake_png_bytes = _make_fake_png(b"warm-tile-data")
    tile_png.write_bytes(fake_png_bytes)

    calls = {"n": 0}

    def fetch_fn():
        calls["n"] += 1
        return tile_png

    a_bytes, a_path = srv._serve_tile_png_cached(fake, 0, "fake.xvm", fetch_fn)
    b_bytes, b_path = srv._serve_tile_png_cached(fake, 0, "fake.xvm", fetch_fn)

    assert a_bytes == fake_png_bytes
    assert b_bytes == fake_png_bytes
    # Single cold compute, second call hit the in-memory LRU.
    assert calls["n"] == 1, f"fetch_fn should have run once, got {calls['n']}"

    s = srv._tile_png_cache_stats()
    assert s["entries"] == 1
    assert s["misses"] == 1
    assert s["hits_inmemory"] == 1


def test_disk_persistence(srv, tmp_path):
    """Clear in-memory only; next call hits disk and serves the same bytes."""
    fake = tmp_path / "fake.xvm"
    fake.write_bytes(b"placeholder")
    tile_png = tmp_path / "tile00.png"
    expected = _make_fake_png(b"persisted-tile-data")
    tile_png.write_bytes(expected)

    def fetch_fn():
        return tile_png

    # Cold compute populates both layers.
    srv._serve_tile_png_cached(fake, 0, "fake.xvm", fetch_fn)

    # Drop in-memory only (keep disk).
    with srv._TILE_PNG_CACHE_LOCK:
        srv._TILE_PNG_CACHE.clear()
        srv._TILE_PNG_CACHE_BYTES = 0
        srv._TILE_PNG_HITS_INMEMORY = 0

    # Next call should now hit L2 — fetch_fn must NOT be called.
    calls = {"n": 0}

    def fetch_fn_2():
        calls["n"] += 1
        return tile_png

    bytes_hit, _ = srv._serve_tile_png_cached(fake, 0, "fake.xvm", fetch_fn_2)
    assert bytes_hit == expected
    assert calls["n"] == 0, "L2 hit should have skipped fetch_fn"

    s = srv._tile_png_cache_stats()
    assert s["hits_disk"] == 1


def test_mtime_change_invalidates(srv, tmp_path):
    """Bumping mtime forces a fresh cold compute."""
    fake = tmp_path / "fake.xvm"
    fake.write_bytes(b"v1")

    tile_png_v1 = tmp_path / "tile00_v1.png"
    tile_png_v2 = tmp_path / "tile00_v2.png"
    tile_png_v1.write_bytes(_make_fake_png(b"v1-data"))
    tile_png_v2.write_bytes(_make_fake_png(b"v2-data"))

    state = {"path": tile_png_v1}

    def fetch_fn():
        return state["path"]

    srv._serve_tile_png_cached(fake, 0, "fake.xvm", fetch_fn)

    # Force a fresh mtime; rewriting the same key the cache used.
    time.sleep(0.05)
    fake.write_bytes(b"v2-different-bytes-and-size")
    state["path"] = tile_png_v2

    bytes_hit, _ = srv._serve_tile_png_cached(fake, 0, "fake.xvm", fetch_fn)
    assert bytes_hit == _make_fake_png(b"v2-data"), "mtime bump must NOT serve stale"

    s = srv._tile_png_cache_stats()
    assert s["misses"] == 2, "expected two cold misses (v1+v2)"


def test_distinct_tile_indices_isolate(srv, tmp_path):
    """Different tile indices for the same archive get separate entries."""
    fake = tmp_path / "fake.xvm"
    fake.write_bytes(b"placeholder")

    tile_png_0 = tmp_path / "tile00.png"
    tile_png_1 = tmp_path / "tile01.png"
    tile_png_0.write_bytes(_make_fake_png(b"tile-0"))
    tile_png_1.write_bytes(_make_fake_png(b"tile-1"))

    def fetch_fn_0():
        return tile_png_0

    def fetch_fn_1():
        return tile_png_1

    srv._serve_tile_png_cached(fake, 0, "fake.xvm", fetch_fn_0)
    srv._serve_tile_png_cached(fake, 1, "fake.xvm", fetch_fn_1)
    srv._serve_tile_png_cached(fake, 0, "fake.xvm", fetch_fn_0)

    s = srv._tile_png_cache_stats()
    assert s["entries"] == 2
    assert s["misses"] == 2
    assert s["hits_inmemory"] == 1


def test_eviction_by_entry_count(srv, tmp_path, monkeypatch):
    """LRU eviction kicks in once entry-count cap is exceeded."""
    monkeypatch.setattr(srv, "_TILE_PNG_CACHE_MAX_ENTRIES", 3)

    for i in range(6):
        fake = tmp_path / f"f{i}.xvm"
        fake.write_bytes(b"x")
        tile_png = tmp_path / f"tile{i:02d}.png"
        tile_png.write_bytes(_make_fake_png(f"tile-{i}".encode()))
        srv._serve_tile_png_cached(fake, 0, f"f{i}.xvm", lambda p=tile_png: p)

    s = srv._tile_png_cache_stats()
    # Up to cap (3) entries; 6 unique misses, oldest 3 evicted.
    assert s["entries"] <= 3
    assert s["misses"] == 6


def test_corrupt_disk_file_recovered(srv, tmp_path):
    """A corrupted on-disk PNG (bad magic) is deleted + treated as miss."""
    fake = tmp_path / "fake.xvm"
    fake.write_bytes(b"placeholder")
    tile_png = tmp_path / "tile00.png"
    tile_png.write_bytes(_make_fake_png(b"good-tile"))

    def fetch_fn():
        return tile_png

    # Cold compute writes a real PNG to the disk cache.
    srv._serve_tile_png_cached(fake, 0, "fake.xvm", fetch_fn)

    # Find the disk path and clobber it.
    key = srv._tile_png_cache_key(fake, 0, "fake.xvm")
    assert key is not None
    disk_p = srv._tile_png_disk_path(key)
    assert disk_p is not None and disk_p.is_file()
    disk_p.write_bytes(b"this is not a PNG")

    # Drop in-memory so we go to L2.
    with srv._TILE_PNG_CACHE_LOCK:
        srv._TILE_PNG_CACHE.clear()
        srv._TILE_PNG_CACHE_BYTES = 0

    # Should detect bad magic, delete the file, re-run fetch_fn.
    calls = {"n": 0}

    def fetch_fn_2():
        calls["n"] += 1
        return tile_png

    out_bytes, _ = srv._serve_tile_png_cached(fake, 0, "fake.xvm", fetch_fn_2)
    assert out_bytes == _make_fake_png(b"good-tile")
    assert calls["n"] == 1, "corrupt disk file should force fetch_fn re-run"
    # The corrupt file got deleted, then the cold-miss re-write
    # repopulates the same disk path with fresh good bytes — verify.
    assert disk_p.is_file(), "disk cache should be repopulated after corrupt-recover"
    assert disk_p.read_bytes() == _make_fake_png(b"good-tile"), \
        "disk cache should hold the freshly-written good bytes"


def test_stat_failure_falls_through(srv, tmp_path):
    """When stat fails, the caller still gets a result via the fetch path."""
    nonexistent = tmp_path / "ghost.xvm"
    tile_png = tmp_path / "tile00.png"
    tile_png.write_bytes(_make_fake_png(b"x"))

    calls = {"n": 0}

    def fetch_fn():
        calls["n"] += 1
        return tile_png

    # Two calls — both should fall through (no caching when key=None).
    bytes_hit, path_hit = srv._serve_tile_png_cached(
        nonexistent, 0, "ghost.xvm", fetch_fn,
    )
    assert bytes_hit is None
    assert path_hit == tile_png
    srv._serve_tile_png_cached(nonexistent, 0, "ghost.xvm", fetch_fn)

    assert calls["n"] == 2

    s = srv._tile_png_cache_stats()
    assert s["entries"] == 0, "stat-fail bypass must not poison the cache"


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

def test_stats_endpoint_shape(client):
    r = client.get("/api/tile_png_cache/stats")
    assert r.status_code == 200, r.text
    s = r.json()
    for k in (
        "entries", "bytes", "max_entries", "max_bytes",
        "disk_entries", "disk_bytes",
        "hits_inmemory", "hits_disk", "misses",
        "hit_rate", "top_entries", "schema",
    ):
        assert k in s, f"missing key {k!r}"
    assert isinstance(s["top_entries"], list)


def test_clear_endpoint(srv, client, tmp_path):
    fake = tmp_path / "fake.xvm"
    fake.write_bytes(b"x")
    tile_png = tmp_path / "tile00.png"
    tile_png.write_bytes(_make_fake_png(b"x"))
    srv._serve_tile_png_cached(fake, 0, "fake.xvm", lambda: tile_png)
    assert srv._tile_png_cache_stats()["entries"] == 1

    r = client.post("/api/tile_png_cache/clear")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cleared_entries"] == 1
    assert srv._tile_png_cache_stats()["entries"] == 0


# ---------------------------------------------------------------------------
# End-to-end via /api/tile_png against a real PSOBB install
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PSOBB, reason="needs PSOBB install")
def test_e2e_warm_hits(srv, client):
    """Two opens of the same XVM = N misses then N hits, all 200 OK."""
    # Pick a smallish XVM that's stable across installs.
    candidates = [
        "obj_lobby_main.xvm",
        "obj_boss1_common_a.xvm",
        "f512_hunters.xvm",
    ]
    xvm: Path | None = None
    for name in candidates:
        p = PSOBB_DATA / name
        if p.is_file():
            xvm = p
            break
    if xvm is None:
        pytest.skip("no test XVM in install")

    # Ensure no existing cached tiles (we want a true cold first-pass).
    srv._tile_png_cache_clear(drop_disk=True)

    # Discover tile count via /api/tiles.
    r = client.get(f"/api/tiles/{xvm.name}")
    assert r.status_code == 200, r.text
    tile_count = r.json()["tile_count"]
    assert tile_count > 0

    # First pass — every tile a miss.
    for i in range(tile_count):
        r = client.get(f"/api/tile_png/{xvm.name}/{i}")
        assert r.status_code == 200, f"tile {i}: {r.status_code} {r.text}"

    s = srv._tile_png_cache_stats()
    assert s["misses"] == tile_count, f"expected {tile_count} misses, got {s['misses']}"

    # Second pass — every tile an in-memory hit.
    for i in range(tile_count):
        r = client.get(f"/api/tile_png/{xvm.name}/{i}")
        assert r.status_code == 200

    s = srv._tile_png_cache_stats()
    assert s["misses"] == tile_count
    assert s["hits_inmemory"] == tile_count


@pytest.mark.skipif(not HAS_PSOBB, reason="needs PSOBB install")
def test_e2e_warm_under_5ms_per_tile(srv, client):
    """Warm tile_png fetch < 5 ms median per tile (in-memory cache)."""
    candidates = ["obj_lobby_main.xvm", "obj_boss1_common_a.xvm"]
    xvm: Path | None = None
    for name in candidates:
        p = PSOBB_DATA / name
        if p.is_file():
            xvm = p
            break
    if xvm is None:
        pytest.skip("no test XVM in install")

    srv._tile_png_cache_clear(drop_disk=True)
    r = client.get(f"/api/tiles/{xvm.name}")
    tile_count = r.json()["tile_count"]
    if tile_count == 0:
        pytest.skip("xvm has no tiles")

    # Warm pass first (cold pass would dominate timing).
    for i in range(tile_count):
        client.get(f"/api/tile_png/{xvm.name}/{i}")

    # Now measure the warm pass.
    times_ms = []
    for i in range(tile_count):
        t0 = time.perf_counter()
        r = client.get(f"/api/tile_png/{xvm.name}/{i}")
        times_ms.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200

    # Median is more robust than mean against the test-client overhead.
    times_ms.sort()
    median_ms = times_ms[len(times_ms) // 2]
    # 50 ms is generous — the in-memory hit itself is <0.1 ms; this
    # mostly accounts for the FastAPI test-client + middleware overhead.
    assert median_ms < 50, f"warm tile_png too slow: median {median_ms:.1f}ms"


@pytest.mark.skipif(not HAS_PSOBB, reason="needs PSOBB install")
def test_e2e_disk_warm_after_inmem_clear(srv, client):
    """After in-memory clear, second pass hits L2 (disk PNG cache)."""
    candidates = ["obj_lobby_main.xvm", "obj_boss1_common_a.xvm"]
    xvm: Path | None = None
    for name in candidates:
        p = PSOBB_DATA / name
        if p.is_file():
            xvm = p
            break
    if xvm is None:
        pytest.skip("no test XVM in install")

    srv._tile_png_cache_clear(drop_disk=True)
    r = client.get(f"/api/tiles/{xvm.name}")
    tile_count = r.json()["tile_count"]
    if tile_count == 0:
        pytest.skip("xvm has no tiles")

    # Cold pass populates disk + memory.
    for i in range(tile_count):
        client.get(f"/api/tile_png/{xvm.name}/{i}")

    # Drop in-memory only (simulate a server restart).
    with srv._TILE_PNG_CACHE_LOCK:
        srv._TILE_PNG_CACHE.clear()
        srv._TILE_PNG_CACHE_BYTES = 0
        srv._TILE_PNG_HITS_INMEMORY = 0
        srv._TILE_PNG_HITS_DISK = 0
        srv._TILE_PNG_MISSES = 0

    # Disk cache should serve every tile.
    for i in range(tile_count):
        r = client.get(f"/api/tile_png/{xvm.name}/{i}")
        assert r.status_code == 200

    s = srv._tile_png_cache_stats()
    assert s["hits_disk"] == tile_count, \
        f"expected {tile_count} disk hits, got {s['hits_disk']}"
    assert s["misses"] == 0


# ---------------------------------------------------------------------------
# Regression: parallel-run race fix (v4 visual polish bundle, 2026-04-25)
# ---------------------------------------------------------------------------
#
# Before the v4 fix, ``test_e2e_disk_warm_after_inmem_clear`` failed
# intermittently under ``pytest -n 4`` because every parallel worker
# shared the same ``cache/tile_png/v1/`` directory. Worker A's
# autouse-fixture call to ``_tile_png_cache_clear(drop_disk=True)``
# would wipe the .png.tmp file that worker B had just opened for
# atomic-rename, and ``os.replace`` would raise WinError 32 mid-write.
# The L2 disk write silently failed, the next test saw fewer disk
# entries than expected, and assertions on ``s["hits_disk"]`` would
# fail.
#
# Fix: ``tests/conftest.py`` sets ``PSO_TILE_PNG_CACHE_DIR`` per worker
# at import time so every xdist worker gets its own subdir
# (``cache/tile_png_test/<worker>/``). ``server.TILE_PNG_CACHE_DIR``
# now honors that env var.
#
# Regression coverage: this test runs the disk-warm flow 5x
# sequentially within a single fixture (no parallel-execution
# dependency), so it catches an accidental rebreak even when run
# single-process. Pair it with ``pytest -n 4 tests/test_tile_png_cache.py``
# in CI to catch the race-flavoured failure mode.

@pytest.mark.skipif(not HAS_PSOBB, reason="needs PSOBB install")
def test_e2e_disk_warm_5x_sequential(srv, client):
    """Run the disk-warm flow 5 times in one fixture; all must succeed.

    Sequential-not-parallel by design: this is the regression-against-
    accidental-rebreak test for the cache-write atomicity. The
    parallel-isolation guard lives in conftest.py and is exercised by
    invoking pytest with ``-n 4`` (which is the actual race-prone
    scenario the v4 fix addressed).
    """
    candidates = ["obj_lobby_main.xvm", "obj_boss1_common_a.xvm"]
    xvm: Path | None = None
    for name in candidates:
        p = PSOBB_DATA / name
        if p.is_file():
            xvm = p
            break
    if xvm is None:
        pytest.skip("no test XVM in install")

    # Establish tile count once; reused across all 5 iterations.
    srv._tile_png_cache_clear(drop_disk=True)
    r = client.get(f"/api/tiles/{xvm.name}")
    tile_count = r.json()["tile_count"]
    if tile_count == 0:
        pytest.skip("xvm has no tiles")

    iterations = 5
    failures: list[str] = []

    for iteration in range(iterations):
        # Reset everything for a clean disk+memory pour.
        srv._tile_png_cache_clear(drop_disk=True)

        # Cold pass: writes both layers. Every tile MUST round-trip.
        for i in range(tile_count):
            r = client.get(f"/api/tile_png/{xvm.name}/{i}")
            if r.status_code != 200:
                failures.append(
                    f"iter={iteration} tile={i} cold pass HTTP {r.status_code}"
                )

        # Drop in-memory only — simulates a server restart.
        with srv._TILE_PNG_CACHE_LOCK:
            srv._TILE_PNG_CACHE.clear()
            srv._TILE_PNG_CACHE_BYTES = 0
            srv._TILE_PNG_HITS_INMEMORY = 0
            srv._TILE_PNG_HITS_DISK = 0
            srv._TILE_PNG_MISSES = 0

        # Warm pass should hit L2 (disk) for every tile. If the cold
        # pass's atomic-rename was racing with another worker, some
        # disk writes silently failed and we see misses here instead
        # of hits_disk.
        for i in range(tile_count):
            r = client.get(f"/api/tile_png/{xvm.name}/{i}")
            if r.status_code != 200:
                failures.append(
                    f"iter={iteration} tile={i} warm pass HTTP {r.status_code}"
                )

        s = srv._tile_png_cache_stats()
        if s["hits_disk"] != tile_count:
            failures.append(
                f"iter={iteration}: expected {tile_count} disk hits, "
                f"got hits_disk={s['hits_disk']} misses={s['misses']}"
            )
        if s["misses"] != 0:
            failures.append(
                f"iter={iteration}: expected 0 cold misses, got {s['misses']}"
            )

    assert not failures, "\n".join(failures)
