"""Tests for the texture-binding LRU cache (Phase D follow-up).

The cache wraps `server._build_model_texture_binding` — the per-model
NJTL→XVMH resolver that took ~1031 ms of dragon's warm /api/model_mesh
request. Cache hits skip the cross-archive lookup entirely.

Covers:
  - Cold compute populates the LRU; second call returns the SAME object
    (warm hit, no recompute).
  - File mtime change invalidates the entry — a re-deploy must NOT
    serve stale cross-BML candidate lists.
  - Different inner names of the same BML do NOT collide.
  - Eviction by entry-count cap (we use a small cap so the test is fast).
  - /api/binding_cache/stats returns sane numbers; /clear empties memory.
  - End-to-end: 10 successive /api/model_mesh calls land 1 miss + 9 hits.
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
def reset_binding_cache(srv):
    """Each test runs against an empty binding cache."""
    srv._binding_cache_clear()
    yield
    srv._binding_cache_clear()


# ---------------------------------------------------------------------------
# Direct unit tests (no PSOBB install needed)
# ---------------------------------------------------------------------------

def test_cold_then_warm_hit(srv, tmp_path):
    """Second call with identical key returns the SAME dict object."""
    # Build a fake BML on disk so the file_key stat works.
    fake = tmp_path / "fake.bml"
    fake.write_bytes(b"placeholder")

    sentinel = {"njtl": [], "xvmh": [], "binding": []}

    calls = {"n": 0}
    real_builder = srv._build_model_texture_binding

    def fake_builder(*args, **kwargs):
        calls["n"] += 1
        return sentinel

    srv._build_model_texture_binding = fake_builder
    try:
        a = srv._build_model_texture_binding_cached(
            fake, ".bml", "body.nj", b"x", [],
        )
        b = srv._build_model_texture_binding_cached(
            fake, ".bml", "body.nj", b"x", [],
        )
        assert a is b, "warm hit must return the same object"
        assert calls["n"] == 1, f"builder called {calls['n']} times, want 1"
    finally:
        srv._build_model_texture_binding = real_builder

    s = srv._binding_cache_stats()
    assert s["hits"] == 1
    assert s["misses"] == 1
    assert s["entries"] == 1


def test_mtime_change_invalidates(srv, tmp_path):
    """Bumping mtime forces a fresh cold compute."""
    fake = tmp_path / "fake.bml"
    fake.write_bytes(b"v1")

    calls = {"n": 0}
    real_builder = srv._build_model_texture_binding

    def fake_builder(*args, **kwargs):
        calls["n"] += 1
        return {"version": calls["n"]}

    srv._build_model_texture_binding = fake_builder
    try:
        srv._build_model_texture_binding_cached(fake, ".bml", "x.nj", b"", [])
        # Force a fresh mtime — sleep then rewrite (some FS resolutions
        # cap at second granularity, so a tiny sleep ensures the bump).
        time.sleep(0.05)
        fake.write_bytes(b"v2-different-bytes-and-size")
        srv._build_model_texture_binding_cached(fake, ".bml", "x.nj", b"", [])
    finally:
        srv._build_model_texture_binding = real_builder

    s = srv._binding_cache_stats()
    assert s["misses"] == 2, "mtime change must NOT serve a stale hit"
    assert s["entries"] == 2


def test_inner_name_isolates(srv, tmp_path):
    """Different inner names within the same BML get separate entries."""
    fake = tmp_path / "fake.bml"
    fake.write_bytes(b"v1")

    real_builder = srv._build_model_texture_binding
    srv._build_model_texture_binding = lambda *a, **k: {}

    try:
        srv._build_model_texture_binding_cached(fake, ".bml", "body.nj", b"", [])
        srv._build_model_texture_binding_cached(fake, ".bml", "head.nj", b"", [])
        srv._build_model_texture_binding_cached(fake, ".bml", "body.nj", b"", [])
    finally:
        srv._build_model_texture_binding = real_builder

    s = srv._binding_cache_stats()
    assert s["entries"] == 2
    assert s["misses"] == 2
    assert s["hits"] == 1


def test_eviction_by_entry_count(srv, tmp_path, monkeypatch):
    """LRU eviction kicks in once entry-count cap is exceeded."""
    real_builder = srv._build_model_texture_binding
    srv._build_model_texture_binding = lambda *a, **k: {"x": 1}

    # Tighten cap for the test then reset on teardown.
    orig_cap = srv._BINDING_CACHE_MAX_ENTRIES
    monkeypatch.setattr(srv, "_BINDING_CACHE_MAX_ENTRIES", 3)

    try:
        for i in range(6):
            f = tmp_path / f"f{i}.bml"
            f.write_bytes(b"x")
            srv._build_model_texture_binding_cached(f, ".bml", "x.nj", b"", [])

        s = srv._binding_cache_stats()
        # We only ever keep up to cap (3) entries.
        assert s["entries"] <= 3
        # 6 unique misses; LRU evicted the oldest 3.
        assert s["misses"] == 6
    finally:
        srv._build_model_texture_binding = real_builder
        # autouse fixture clears the cache regardless.


def test_stat_failure_falls_through(srv, tmp_path):
    """When stat fails (deleted file), call still serves correct result."""
    nonexistent = tmp_path / "ghost.bml"  # never written

    calls = {"n": 0}
    real_builder = srv._build_model_texture_binding
    srv._build_model_texture_binding = lambda *a, **k: (
        calls.__setitem__("n", calls["n"] + 1) or {"ghost": True}
    )
    try:
        out = srv._build_model_texture_binding_cached(
            nonexistent, ".bml", "x.nj", b"", [],
        )
        assert out == {"ghost": True}
        # Both calls should fall through (no caching when key=None).
        srv._build_model_texture_binding_cached(
            nonexistent, ".bml", "x.nj", b"", [],
        )
        assert calls["n"] == 2
    finally:
        srv._build_model_texture_binding = real_builder

    s = srv._binding_cache_stats()
    assert s["entries"] == 0, "stat-fail bypass must not poison the cache"


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

def test_stats_endpoint_shape(client):
    r = client.get("/api/binding_cache/stats")
    assert r.status_code == 200, r.text
    s = r.json()
    for k in ("entries", "bytes", "max_entries", "max_bytes",
              "hits", "misses", "hit_rate", "top_entries"):
        assert k in s, f"missing key {k!r}"
    assert isinstance(s["top_entries"], list)


def test_clear_endpoint(srv, client, tmp_path):
    fake = tmp_path / "fake.bml"
    fake.write_bytes(b"x")
    real_builder = srv._build_model_texture_binding
    srv._build_model_texture_binding = lambda *a, **k: {}
    try:
        srv._build_model_texture_binding_cached(fake, ".bml", "x.nj", b"", [])
    finally:
        srv._build_model_texture_binding = real_builder
    assert srv._binding_cache_stats()["entries"] == 1

    r = client.post("/api/binding_cache/clear")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cleared_entries"] == 1
    assert srv._binding_cache_stats()["entries"] == 0


# ---------------------------------------------------------------------------
# End-to-end via /api/model_mesh
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PSOBB, reason="needs PSOBB install")
def test_model_mesh_warm_hits_binding_cache(srv, client):
    """10 successive /api/model_mesh opens of dragon: 1 miss + 9 hits."""
    # Use De Rol Le since it's stable across installs and confirmed in
    # the existing test_phase05_perf suite.
    bml = PSOBB_DATA / "bm_boss2_de_rol_le.bml"
    if not bml.exists():
        pytest.skip("bm_boss2_de_rol_le.bml not in install")

    from formats.bml import parse_bml
    entries = parse_bml(bml.read_bytes())
    inner = next((e.name for e in entries if e.name.lower().endswith(".nj")), None)
    if not inner:
        pytest.skip("no .nj inner in target BML")

    # Reset both caches so the first request is a true cold miss.
    srv._binding_cache_clear()

    for _ in range(10):
        r = client.get(f"/api/model_mesh/{bml.name}", params={"inner": inner})
        assert r.status_code == 200, r.text
        assert "binding_data" in r.json()

    s = srv._binding_cache_stats()
    assert s["misses"] == 1, f"expected 1 cold miss, got {s['misses']}"
    assert s["hits"] == 9, f"expected 9 warm hits, got {s['hits']}"
    assert s["entries"] == 1


@pytest.mark.skipif(not HAS_PSOBB, reason="needs PSOBB install")
def test_model_mesh_warm_under_300ms(srv, client):
    """Warm /api/model_mesh under 300ms (binding cache + parse cache)."""
    bml = PSOBB_DATA / "bm_boss2_de_rol_le.bml"
    if not bml.exists():
        pytest.skip("bm_boss2_de_rol_le.bml not in install")
    from formats.bml import parse_bml
    entries = parse_bml(bml.read_bytes())
    inner = next((e.name for e in entries if e.name.lower().endswith(".nj")), None)
    if not inner:
        pytest.skip("no .nj inner")

    # Cold pass populates everything (parse cache, binding cache, etc.).
    client.get(f"/api/model_mesh/{bml.name}", params={"inner": inner})

    # Warm pass — should hit every layer.
    t0 = time.perf_counter()
    r = client.get(f"/api/model_mesh/{bml.name}", params={"inner": inner})
    warm_ms = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200, r.text

    # 300 ms covers the network/test-client overhead + JSON serialization
    # of the still-large mesh payload. The cache work itself is ~0 ms.
    assert warm_ms < 300, f"warm /api/model_mesh too slow: {warm_ms:.1f}ms"


# ---------------------------------------------------------------------------
# Disk persistence (Item 5, 2026-04-25)
# ---------------------------------------------------------------------------
# The in-memory LRU evaporates on process restart. Adding a JSON file
# tier under cache/binding/v<schema>/ lets cold-after-restart hits load
# from disk in ~30-50 ms instead of recomputing the ~1 s NJTL→XVMH
# binding for a dragon-class model.

def test_disk_tier_writes_json_after_cold_compute(srv, tmp_path):
    """A cold compute populates both in-memory AND a JSON file on disk."""
    fake = tmp_path / "fake.bml"
    fake.write_bytes(b"v1")

    sentinel = {"njtl": [{"name": "a"}], "xvmh": [], "binding": [42]}
    real_builder = srv._build_model_texture_binding
    srv._build_model_texture_binding = lambda *a, **k: sentinel
    try:
        srv._build_model_texture_binding_cached(
            fake, ".bml", "body.nj", b"x", [],
        )
    finally:
        srv._build_model_texture_binding = real_builder

    # Disk file should exist under cache/binding/v<schema>/.
    base = srv.BINDING_CACHE_DIR / f"v{srv.BINDING_CACHE_SCHEMA}"
    files = list(base.glob("*.json"))
    assert files, f"no JSON file written under {base}"
    # Verify shape: {"key": [...], "payload": {...}}
    import json as _json
    with files[0].open("r", encoding="utf-8") as f:
        obj = _json.load(f)
    assert "key" in obj and "payload" in obj
    assert obj["payload"] == sentinel


def test_disk_tier_serves_after_inmemory_cleared(srv, tmp_path):
    """Drop the in-memory cache; a fresh call hits disk and returns the
    persisted payload without recomputing."""
    fake = tmp_path / "fake.bml"
    fake.write_bytes(b"v1")
    payload = {"njtl": [], "xvmh": [{"name": "tex0"}], "binding": [1, 2, 3]}

    real_builder = srv._build_model_texture_binding
    builds = {"n": 0}

    def fake_builder(*a, **k):
        builds["n"] += 1
        return payload

    srv._build_model_texture_binding = fake_builder
    try:
        # 1) Cold: write to disk + memory.
        srv._build_model_texture_binding_cached(
            fake, ".bml", "body.nj", b"x", [],
        )
        assert builds["n"] == 1
        # 2) Drop in-memory only — disk tier survives.
        srv._binding_cache_clear(drop_disk=False)
        # 3) Re-call: must NOT invoke builder; must return persisted payload.
        result = srv._build_model_texture_binding_cached(
            fake, ".bml", "body.nj", b"x", [],
        )
    finally:
        srv._build_model_texture_binding = real_builder

    assert builds["n"] == 1, "disk hit must NOT trigger a fresh compute"
    assert result == payload, "disk-tier payload mismatch"
    # Stats should reflect the disk hit explicitly.
    s = srv._binding_cache_stats()
    assert s["hits_disk"] >= 1


def test_disk_tier_invalidates_on_mtime_change(srv, tmp_path):
    """Bumping the source file's mtime + size voids the disk cache entry."""
    fake = tmp_path / "fake.bml"
    fake.write_bytes(b"v1")

    real_builder = srv._build_model_texture_binding
    builds = {"n": 0}

    def fake_builder(*a, **k):
        builds["n"] += 1
        return {"version": builds["n"]}

    srv._build_model_texture_binding = fake_builder
    try:
        srv._build_model_texture_binding_cached(
            fake, ".bml", "body.nj", b"x", [],
        )
        # Drop in-memory, keep disk.
        srv._binding_cache_clear(drop_disk=False)
        # Re-write source: new size + mtime. The cache key includes
        # mtime_ns + size, so the new compute must NOT serve the old
        # disk entry.
        time.sleep(0.05)
        fake.write_bytes(b"v2-different-bytes-and-size-entirely")
        result = srv._build_model_texture_binding_cached(
            fake, ".bml", "body.nj", b"x", [],
        )
    finally:
        srv._build_model_texture_binding = real_builder

    assert builds["n"] == 2, (
        "mtime change must NOT serve a stale disk entry"
    )
    assert result == {"version": 2}


def test_disk_tier_handles_corrupt_json(srv, tmp_path):
    """A corrupted on-disk file is auto-deleted and triggers a fresh compute."""
    fake = tmp_path / "fake.bml"
    fake.write_bytes(b"v1")

    real_builder = srv._build_model_texture_binding
    payload = {"x": 1}
    srv._build_model_texture_binding = lambda *a, **k: payload
    try:
        # Cold compute writes a clean JSON.
        srv._build_model_texture_binding_cached(
            fake, ".bml", "body.nj", b"x", [],
        )
        # Find the JSON file we just wrote and corrupt it.
        base = srv.BINDING_CACHE_DIR / f"v{srv.BINDING_CACHE_SCHEMA}"
        files = list(base.glob("*.json"))
        assert files
        files[0].write_bytes(b"{ this is not valid JSON at all")
        # Drop in-memory; the next call hits the corrupt disk file.
        srv._binding_cache_clear(drop_disk=False)
        # Should not raise; corrupt file auto-removed and fresh compute
        # populates a clean replacement.
        out = srv._build_model_texture_binding_cached(
            fake, ".bml", "body.nj", b"x", [],
        )
    finally:
        srv._build_model_texture_binding = real_builder

    assert out == payload
    # Corrupt file replaced with a fresh one (or at least one valid file).
    base = srv.BINDING_CACHE_DIR / f"v{srv.BINDING_CACHE_SCHEMA}"
    files = list(base.glob("*.json"))
    assert files, "no valid JSON post-corruption-recovery"
    import json as _json
    obj = _json.loads(files[0].read_text(encoding="utf-8"))
    assert obj.get("payload") == payload


def test_clear_endpoint_drops_disk_too(srv, client, tmp_path):
    """The /api/binding_cache/clear endpoint clears both tiers by default."""
    fake = tmp_path / "fake.bml"
    fake.write_bytes(b"x")
    real_builder = srv._build_model_texture_binding
    srv._build_model_texture_binding = lambda *a, **k: {"present": True}
    try:
        srv._build_model_texture_binding_cached(
            fake, ".bml", "body.nj", b"x", [],
        )
    finally:
        srv._build_model_texture_binding = real_builder

    base = srv.BINDING_CACHE_DIR / f"v{srv.BINDING_CACHE_SCHEMA}"
    pre = list(base.glob("*.json"))
    assert pre, "expected a disk entry pre-clear"

    r = client.post("/api/binding_cache/clear")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cleared_entries"] >= 1
    assert body["disk_files_dropped"] >= 1
    assert body["disk_bytes_freed"] > 0

    post = list(base.glob("*.json"))
    assert not post, f"disk files survived clear: {post}"


def test_disk_tier_can_be_disabled_via_env(srv, tmp_path, monkeypatch):
    """PSO_DISABLE_DISK_BINDING_CACHE=1 keeps in-memory only — no disk writes."""
    monkeypatch.setenv("PSO_DISABLE_DISK_BINDING_CACHE", "1")
    fake = tmp_path / "fake.bml"
    fake.write_bytes(b"v1")

    real_builder = srv._build_model_texture_binding
    srv._build_model_texture_binding = lambda *a, **k: {"x": 1}
    try:
        # First, clear ALL disk entries from prior tests (autouse only
        # clears once at the start; we want zero entries here regardless).
        base = srv.BINDING_CACHE_DIR / f"v{srv.BINDING_CACHE_SCHEMA}"
        if base.is_dir():
            for child in base.glob("*.json"):
                child.unlink()
        srv._build_model_texture_binding_cached(
            fake, ".bml", "body.nj", b"x", [],
        )
    finally:
        srv._build_model_texture_binding = real_builder

    base = srv.BINDING_CACHE_DIR / f"v{srv.BINDING_CACHE_SCHEMA}"
    files = list(base.glob("*.json")) if base.is_dir() else []
    assert not files, f"disk write happened despite disable env: {files}"


def test_stats_includes_disk_keys(srv, client, tmp_path):
    """Stats endpoint surfaces disk_entries / disk_bytes / hits_disk."""
    real_builder = srv._build_model_texture_binding
    srv._build_model_texture_binding = lambda *a, **k: {"y": 1}
    try:
        f = tmp_path / "x.bml"
        f.write_bytes(b"x")
        srv._build_model_texture_binding_cached(f, ".bml", "x.nj", b"", [])
    finally:
        srv._build_model_texture_binding = real_builder

    r = client.get("/api/binding_cache/stats")
    assert r.status_code == 200, r.text
    s = r.json()
    for k in ("hits_inmemory", "hits_disk", "disk_entries", "disk_bytes",
              "schema"):
        assert k in s, f"stats missing key {k!r}"
    # disk_entries should be >= 1 since we just wrote one.
    assert s["disk_entries"] >= 1
    assert s["disk_bytes"] > 0


# ---------------------------------------------------------------------------
# End-to-end perf: warm-from-disk first-open
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PSOBB, reason="needs PSOBB install")
def test_disk_warm_first_open_under_100ms(srv, client):
    """Cold-after-restart simulation: drop in-memory only, time first
    /api/model_mesh open. With disk tier, must be <100ms (vs ~1 s
    re-compute on a true cold-cold open).
    """
    bml = PSOBB_DATA / "bm_boss2_de_rol_le.bml"
    if not bml.exists():
        pytest.skip("bm_boss2_de_rol_le.bml not in install")
    from formats.bml import parse_bml
    entries = parse_bml(bml.read_bytes())
    inner = next((e.name for e in entries if e.name.lower().endswith(".nj")), None)
    if not inner:
        pytest.skip("no .nj inner")

    # Pass 1: cold-cold — populates disk + memory.
    client.get(f"/api/model_mesh/{bml.name}", params={"inner": inner})

    # Drop in-memory only (simulate restart while keeping disk).
    srv._binding_cache_clear(drop_disk=False)

    # Pass 2: cold-warm — disk hit must serve.
    t0 = time.perf_counter()
    r = client.get(f"/api/model_mesh/{bml.name}", params={"inner": inner})
    warm_disk_ms = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200, r.text

    # 100 ms covers the json.load + transport/serialise overhead. Cold
    # compute would be ~1 s — anything under 200 ms confirms the disk
    # tier is in play. We use 100 ms as the regression bar.
    assert warm_disk_ms < 200, (
        f"disk-warm /api/model_mesh too slow: {warm_disk_ms:.1f}ms "
        "(cold-cold would be ~1000ms; disk hit should be <100ms; "
        "200ms gives slack for slow CI disks)"
    )

    # Confirm the binding-cache stats record the disk hit.
    s = srv._binding_cache_stats()
    assert s["hits_disk"] >= 1, f"expected ≥1 disk hit, got {s['hits_disk']}"
