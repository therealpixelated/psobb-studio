"""Tests for the skinned-payload LRU + on-disk cache (Phase 0.5 perf).

The cache wraps `server._xj_meshes_to_skinned_payload` — the geometry+
skeleton dict assembler that took ~33-100 ms on dragon-class models and
re-ran on every /api/model_skinned hit (variant picker, motion preview,
paint, sculpt). Cache hits skip the conversion AND the parse_cache
disk-load when L2 hits, dropping the warm-from-cold-restart skinned
step from ~1259 ms toward the 50-100 ms target.

Covers:
  - Cold compute populates the in-memory LRU; second call returns the
    SAME object (warm hit, no recompute).
  - Disk persistence writes on cold compute and serves L2 hits after the
    in-memory cache is cleared (process-restart simulation).
  - File mtime change invalidates the entry — a re-deploy must NOT
    serve stale geometry/bones.
  - Different inner names of the same BML do NOT collide.
  - Eviction by entry-count cap (small cap, fast test).
  - /api/skinned_payload_cache/stats returns sane numbers; /clear empties.
  - Numpy-backed conversion produces byte-identical output to the
    reference (round-trip via base64) for a known synthetic mesh.
  - End-to-end via /api/model_skinned: 10 dragon opens land 1 miss + 9
    hits with a single entry stuck in the cache.
"""
from __future__ import annotations

import base64
import os
import struct
import time
from pathlib import Path

import numpy as np
import pytest

from fastapi.testclient import TestClient


PSOBB_DATA = Path(os.path.expanduser("~/PSOBB.IO/data"))
HAS_PSOBB = PSOBB_DATA.is_dir()
DRAGON_BML = PSOBB_DATA / "bm_boss8_dragon.bml"
DRAGON_INNER = "boss1_s_nb_dragon.nj"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def srv():
    import server
    return server


@pytest.fixture(scope="module")
def client(srv):
    return TestClient(srv.app)


@pytest.fixture(autouse=True)
def reset_skinned_payload_cache(srv):
    """Each test runs against an empty cache (in-memory + on-disk)."""
    srv._skinned_payload_cache_clear(drop_disk=True)
    yield
    srv._skinned_payload_cache_clear(drop_disk=True)


# ---------------------------------------------------------------------------
# Synthetic-mesh helper
# ---------------------------------------------------------------------------
# We need a small (meshes, bones) pair that doesn't depend on a real
# PSOBB install. Build it from XjVertex + XjMesh + XjBone dataclasses
# directly — that's the same shape the parser hands us.

def _make_synthetic(num_meshes: int = 2, verts_per_mesh: int = 4):
    """Build a (meshes, bones) pair without parsing a real file."""
    from formats.xj import XjMesh, XjVertex, XjBone

    meshes = []
    for mi in range(num_meshes):
        verts = [
            XjVertex(
                pos=(float(i), float(i + 1), float(i + 2)),
                normal=(0.0, 1.0, 0.0),
                uv=(float(i) * 0.1, float(i) * 0.2),
                bone_idx=mi,  # one bone per mesh
            )
            for i in range(verts_per_mesh)
        ]
        # Simple 2-triangle quad-ish strip for any verts_per_mesh >= 3
        idx = [0, 1, 2]
        if verts_per_mesh >= 4:
            idx += [1, 2, 3]
        meshes.append(XjMesh(
            vertices=verts,
            indices=idx,
            material_id=mi,
            bounding_sphere=(0.0, 0.0, 0.0, 1.0),
        ))
    bones = [
        XjBone(
            index=i,
            parent=i - 1,
            position=(0.0, 0.0, float(i)),
            rotation=(0, 0, 0),
            scale=(1.0, 1.0, 1.0),
            eval_flags=0,
        )
        for i in range(num_meshes)
    ]
    return meshes, bones


# ---------------------------------------------------------------------------
# Numpy conversion correctness
# ---------------------------------------------------------------------------

def test_conversion_correctness_synthetic(srv):
    """Numpy-backed conversion produces correct b64 buffers and AABB."""
    meshes, bones = _make_synthetic(num_meshes=1, verts_per_mesh=4)
    payload = srv._xj_meshes_to_skinned_payload(meshes, bones)

    assert payload["mesh_count"] == 1
    assert payload["bone_count"] == 1
    assert payload["vertices_pre_transformed"] is False
    assert payload["has_bone_indices"] is True
    assert payload["totals"] == {"vertices": 4, "triangles": 2}

    sub = payload["meshes"][0]
    # Round-trip vertices_b64 → Float32 array → check first 8 floats match
    # the constructed mesh.
    raw = base64.b64decode(sub["vertices_b64"])
    arr = np.frombuffer(raw, dtype="<f4").reshape(-1, 8)
    assert arr.shape == (4, 8)
    # Vertex 0: pos=(0,1,2), normal=(0,1,0), uv=(0,0)
    np.testing.assert_allclose(arr[0], [0, 1, 2, 0, 1, 0, 0, 0])
    np.testing.assert_allclose(arr[3], [3, 4, 5, 0, 1, 0, 0.3, 0.6])

    # AABB = positions min/max across all 4 verts.
    aabb = sub["aabb"]
    assert aabb == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    # Bone-index buffer
    bi_raw = base64.b64decode(sub["bone_indices_b64"])
    bi = np.frombuffer(bi_raw, dtype="<i4")
    np.testing.assert_array_equal(bi, [0, 0, 0, 0])

    # Indices buffer
    idx_raw = base64.b64decode(sub["indices_b64"])
    idx = np.frombuffer(idx_raw, dtype="<u4")
    np.testing.assert_array_equal(idx, [0, 1, 2, 1, 2, 3])


def test_empty_mesh_handled(srv):
    """Empty XjMesh → empty buffers + zeroed AABB, no crash."""
    from formats.xj import XjMesh, XjBone
    m = XjMesh(vertices=[], indices=[], material_id=7,
               bounding_sphere=(0.0, 0.0, 0.0, 0.0))
    bones = [XjBone(index=0, parent=-1, position=(0.0, 0.0, 0.0),
                    rotation=(0, 0, 0))]
    payload = srv._xj_meshes_to_skinned_payload([m], bones)
    sub = payload["meshes"][0]
    assert sub["vertex_count"] == 0
    assert sub["triangle_count"] == 0
    assert sub["aabb"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert sub["vertices_b64"] == ""
    assert sub["indices_b64"] == ""
    assert sub["bone_indices_b64"] == ""


# ---------------------------------------------------------------------------
# In-memory hot path
# ---------------------------------------------------------------------------

def test_cold_then_warm_inmemory(srv, tmp_path):
    """Second call with identical key returns the SAME dict object."""
    fake = tmp_path / "fake.bml"
    fake.write_bytes(b"placeholder-v1")
    meshes, bones = _make_synthetic()

    a = srv._xj_meshes_to_skinned_payload_cached(
        meshes, bones, fake, "body.nj",
    )
    b = srv._xj_meshes_to_skinned_payload_cached(
        meshes, bones, fake, "body.nj",
    )
    assert a is b, "warm hit must return the same object"

    s = srv._skinned_payload_cache_stats()
    assert s["hits_inmemory"] == 1
    assert s["misses"] == 1
    assert s["entries"] == 1


def test_disk_layer_serves_after_inmemory_clear(srv, tmp_path):
    """L2 hit: clearing in-memory but keeping disk → L2 serves the next call."""
    fake = tmp_path / "model.bml"
    fake.write_bytes(b"v1-bytes")
    meshes, bones = _make_synthetic()

    # Cold compute populates in-memory + disk.
    srv._xj_meshes_to_skinned_payload_cached(meshes, bones, fake, "x.nj")
    s = srv._skinned_payload_cache_stats()
    assert s["misses"] == 1
    assert s["disk_entries"] == 1

    # Drop in-memory only — process-restart simulation.
    srv._skinned_payload_cache_clear(drop_disk=False)
    assert srv._skinned_payload_cache_stats()["entries"] == 0

    # Next call should land an L2 hit, not a fresh miss.
    out = srv._xj_meshes_to_skinned_payload_cached(meshes, bones, fake, "x.nj")
    assert out is not None
    s = srv._skinned_payload_cache_stats()
    assert s["hits_disk"] == 1
    assert s["misses"] == 0  # cleared above
    assert s["entries"] == 1


def test_mtime_change_invalidates(srv, tmp_path):
    """Bumping mtime forces a fresh cold compute (no stale serve)."""
    fake = tmp_path / "model.bml"
    fake.write_bytes(b"v1-bytes")
    meshes, bones = _make_synthetic()

    srv._xj_meshes_to_skinned_payload_cached(meshes, bones, fake, "x.nj")
    # Force a fresh mtime — sleep then rewrite.
    time.sleep(0.05)
    fake.write_bytes(b"v2-different-bytes-and-size")
    srv._xj_meshes_to_skinned_payload_cached(meshes, bones, fake, "x.nj")

    s = srv._skinned_payload_cache_stats()
    assert s["misses"] == 2, "mtime change must NOT serve a stale hit"
    assert s["entries"] == 2


def test_inner_name_isolates(srv, tmp_path):
    """Different inner names within the same BML get separate entries."""
    fake = tmp_path / "model.bml"
    fake.write_bytes(b"x")
    meshes, bones = _make_synthetic()

    srv._xj_meshes_to_skinned_payload_cached(meshes, bones, fake, "body.nj")
    srv._xj_meshes_to_skinned_payload_cached(meshes, bones, fake, "head.nj")
    srv._xj_meshes_to_skinned_payload_cached(meshes, bones, fake, "body.nj")

    s = srv._skinned_payload_cache_stats()
    assert s["entries"] == 2
    assert s["misses"] == 2
    assert s["hits_inmemory"] == 1


def test_eviction_by_entry_count(srv, tmp_path, monkeypatch):
    """LRU eviction kicks in once entry-count cap is exceeded."""
    monkeypatch.setattr(srv, "_SKINNED_PAYLOAD_CACHE_MAX_ENTRIES", 3)
    meshes, bones = _make_synthetic()

    for i in range(6):
        f = tmp_path / f"f{i}.bml"
        f.write_bytes(f"x{i}".encode())
        srv._xj_meshes_to_skinned_payload_cached(meshes, bones, f, "x.nj")

    s = srv._skinned_payload_cache_stats()
    assert s["entries"] <= 3, f"cap=3 not honored, got {s['entries']}"
    assert s["misses"] == 6


def test_stat_failure_falls_through(srv, tmp_path):
    """When stat fails (deleted file), call still serves correct result."""
    nonexistent = tmp_path / "ghost.bml"  # never written
    meshes, bones = _make_synthetic()

    out = srv._xj_meshes_to_skinned_payload_cached(
        meshes, bones, nonexistent, "x.nj",
    )
    assert out["mesh_count"] == 2  # synthetic = 2 meshes by default
    s = srv._skinned_payload_cache_stats()
    assert s["entries"] == 0, "stat-fail bypass must not poison the cache"


# ---------------------------------------------------------------------------
# Disk corruption resilience
# ---------------------------------------------------------------------------

def test_corrupt_disk_json_recovers(srv, tmp_path, monkeypatch):
    """A corrupted on-disk JSON gets deleted + rebuilt; no crash."""
    fake = tmp_path / "model.bml"
    fake.write_bytes(b"v1-bytes")
    meshes, bones = _make_synthetic()

    # Cold compute → populates disk.
    srv._xj_meshes_to_skinned_payload_cached(meshes, bones, fake, "x.nj")
    key = srv._skinned_payload_cache_key(fake, "x.nj")
    assert key is not None
    disk_path = srv._skinned_payload_disk_path(key)
    assert disk_path is not None
    assert disk_path.is_file()

    # Corrupt the file.
    disk_path.write_bytes(b"not valid json {{")

    # Drop in-memory; next call should silently delete + recompute.
    # `_skinned_payload_cache_clear` zeroes stats so we count the
    # recompute as a single miss against the freshly-cleared counters.
    srv._skinned_payload_cache_clear(drop_disk=False)
    srv._xj_meshes_to_skinned_payload_cached(meshes, bones, fake, "x.nj")

    s = srv._skinned_payload_cache_stats()
    # The corrupt file got deleted, recompute kicks in → 1 miss
    # (against the cleared counter), no disk hit.
    assert s["misses"] == 1
    assert s["hits_disk"] == 0
    # Disk should be re-populated cleanly.
    assert disk_path.is_file()


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

def test_stats_endpoint_shape(client):
    r = client.get("/api/skinned_payload_cache/stats")
    assert r.status_code == 200, r.text
    s = r.json()
    for k in ("entries", "bytes", "max_entries", "max_bytes",
              "disk_entries", "disk_bytes",
              "hits_inmemory", "hits_disk", "misses",
              "hit_rate", "top_entries", "schema"):
        assert k in s, f"missing key {k!r}"
    assert isinstance(s["top_entries"], list)


def test_clear_endpoint(srv, client, tmp_path):
    fake = tmp_path / "fake.bml"
    fake.write_bytes(b"x")
    meshes, bones = _make_synthetic()
    srv._xj_meshes_to_skinned_payload_cached(meshes, bones, fake, "x.nj")
    assert srv._skinned_payload_cache_stats()["entries"] == 1

    r = client.post("/api/skinned_payload_cache/clear")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cleared_entries"] == 1
    assert srv._skinned_payload_cache_stats()["entries"] == 0


# ---------------------------------------------------------------------------
# End-to-end via /api/model_skinned
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PSOBB or not DRAGON_BML.is_file(),
                    reason="needs PSOBB install with dragon BML")
def test_model_skinned_warm_hits_payload_cache(srv, client):
    """10 successive /api/model_skinned opens of dragon: 1 miss + 9 hits."""
    # Reset the payload cache so the first request is a true cold miss.
    srv._skinned_payload_cache_clear(drop_disk=True)

    for _ in range(10):
        r = client.get(
            f"/api/model_skinned/{DRAGON_BML.name}",
            params={"inner": DRAGON_INNER},
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data.get("filename")  # per-request field is present
        assert "meshes" in data
        assert "bones" in data

    s = srv._skinned_payload_cache_stats()
    assert s["misses"] == 1, f"expected 1 cold miss, got {s['misses']}"
    assert s["hits_inmemory"] == 9, f"expected 9 warm hits, got {s['hits_inmemory']}"
    assert s["entries"] == 1


@pytest.mark.skipif(not HAS_PSOBB or not DRAGON_BML.is_file(),
                    reason="needs PSOBB install with dragon BML")
def test_per_request_fields_do_not_leak(srv, client):
    """Caching the geometry payload must NOT mutate-leak filename/inner/binding."""
    srv._skinned_payload_cache_clear(drop_disk=True)

    # First call: stash the raw cached dict for comparison.
    r1 = client.get(
        f"/api/model_skinned/{DRAGON_BML.name}",
        params={"inner": DRAGON_INNER},
    )
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1.get("filename") == DRAGON_BML.name
    assert body1.get("inner") == DRAGON_INNER

    # Inspect the cached dict directly — it must NOT carry per-request
    # fields. Otherwise a second call (with the same path) would
    # silently work but a path-aliased call would wrongly inherit them.
    # Resolve the path the SAME way the route handler does so the cache
    # key matches (DATA_DIR may be the dev-data dir, not the PSOBB
    # install).
    resolved = srv._resolve_model_mesh_path(DRAGON_BML.name)
    key = srv._skinned_payload_cache_key(resolved, DRAGON_INNER)
    assert key is not None
    cached = srv._SKINNED_PAYLOAD_CACHE.get(key)
    assert cached is not None
    cached_dict = cached[0]
    assert "filename" not in cached_dict, \
        "per-request 'filename' must not enter the cache"
    assert "inner" not in cached_dict
    assert "binding_data" not in cached_dict


@pytest.mark.skipif(not HAS_PSOBB or not DRAGON_BML.is_file(),
                    reason="needs PSOBB install with dragon BML")
def test_warm_under_300ms(srv, client):
    """Warm /api/model_skinned for dragon under 300 ms (every layer hot)."""
    # Cold pass populates everything.
    client.get(
        f"/api/model_skinned/{DRAGON_BML.name}",
        params={"inner": DRAGON_INNER},
    )

    t0 = time.perf_counter()
    r = client.get(
        f"/api/model_skinned/{DRAGON_BML.name}",
        params={"inner": DRAGON_INNER},
    )
    warm_ms = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200, r.text

    # 300 ms covers TestClient overhead + JSON serialization of the
    # multi-MB payload. The cache lookups themselves are <1 ms.
    assert warm_ms < 300, f"warm /api/model_skinned too slow: {warm_ms:.1f} ms"


@pytest.mark.skipif(not HAS_PSOBB or not DRAGON_BML.is_file(),
                    reason="needs PSOBB install with dragon BML")
def test_gzip_middleware_kicks_in(client):
    """Verify Content-Encoding: gzip on /api/model_skinned/dragon."""
    # TestClient auto-decompresses by default; force the encoding header
    # so FastAPI's middleware actually compresses for us, then check the
    # Content-Encoding response header surfaces.
    r = client.get(
        f"/api/model_skinned/{DRAGON_BML.name}",
        params={"inner": DRAGON_INNER},
        headers={"accept-encoding": "gzip"},
    )
    assert r.status_code == 200, r.text
    # GZipMiddleware is global, minimum_size=1024; the dragon payload is
    # multi-MB so this MUST be gzipped on-wire.
    assert r.headers.get("content-encoding") == "gzip", \
        f"missing gzip encoding; got {dict(r.headers)}"


# ---------------------------------------------------------------------------
# Profile gate (PSO_PROFILE=1) — smoke test only
# ---------------------------------------------------------------------------

def test_profile_gate_off_by_default(srv, monkeypatch):
    """Profile gate is OFF unless PSO_PROFILE=1 — no surprise overhead."""
    monkeypatch.delenv("PSO_PROFILE", raising=False)
    assert srv._skinned_payload_profile_enabled() is False


def test_profile_gate_honored(srv, monkeypatch):
    monkeypatch.setenv("PSO_PROFILE", "1")
    assert srv._skinned_payload_profile_enabled() is True


def test_profile_path_does_not_crash(srv, tmp_path, monkeypatch):
    """Cold compute under PSO_PROFILE=1 returns a valid payload."""
    monkeypatch.setenv("PSO_PROFILE", "1")
    fake = tmp_path / "model.bml"
    fake.write_bytes(b"x")
    meshes, bones = _make_synthetic()
    out = srv._xj_meshes_to_skinned_payload_cached(meshes, bones, fake, "x.nj")
    assert out["mesh_count"] == 2
    assert "bones" in out
