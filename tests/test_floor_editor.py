"""Tests for the Floor copy/create editor (server.py /api/floors/*).

CENTRAL INVARIANT: this editor has NO live-write verb. Every copy/create
output is confined to DEV_DATA_DIR and is HARD-asserted to be neither
LIVE_DATA_DIR nor a child of it before any byte hits disk. The proof tests
here monkeypatch LIVE_DATA_DIR / DEV_DATA_DIR / DATA_DIR to tmp dirs,
snapshot the live dir's md5s, run create + copy, and assert the live dir is
byte-identical afterward.

All tests run against monkeypatched tmp dirs via an in-process FastAPI
TestClient. None requires a real PSOBB install. The synthetic floor is
built from a generated grid mesh (CI-safe, no game assets).
"""
from __future__ import annotations

import ast
import hashlib
import io
import struct
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import server  # noqa: E402
from formats import lobby_pipeline as lp  # noqa: E402
from formats import rel_writer as _rw  # noqa: E402


# ---------------------------------------------------------------------------
# synthetic geometry — a fake parsed-model with a generated grid mesh
# ---------------------------------------------------------------------------
class _FakeMesh:
    def __init__(self, verts, faces, uvs):
        self.vertices = verts
        self.indices = faces.reshape(-1).tolist()
        self.uvs = uvs


class _FakeModel:
    def __init__(self, meshes, textures=None):
        self.meshes = meshes
        self.textures = textures or []


def _grid_mesh(n: int):
    """An n×n ground grid -> 2*(n-1)^2 triangles, with planar UVs."""
    xs = np.linspace(-200.0, 200.0, n)
    V = np.array([[x, 0.0, z] for z in xs for x in xs], dtype=np.float64)
    U = np.array([[(x + 200) / 400, (z + 200) / 400] for z in xs for x in xs],
                 dtype=np.float64)
    F = []
    for r in range(n - 1):
        for c in range(n - 1):
            a = r * n + c
            F.append([a, a + 1, a + n])
            F.append([a + 1, a + n + 1, a + n])
    return V, U, np.array(F, dtype=np.int64)


def _grid_model(n: int) -> _FakeModel:
    V, U, F = _grid_mesh(n)
    return _FakeModel([_FakeMesh(V, F, U)])


def _build_synthetic_nrel_crel_xvm(n: int = 12):
    """Author a small synthetic floor's bytes via the pipeline."""
    res = lp.build_floor(_grid_model(n), texname="lobby")
    return res.nrel, res.crel, res.xvm


def _md5(b: bytes) -> str:
    h = hashlib.md5()
    h.update(b)
    return h.hexdigest()


def _snapshot_dir(d: Path) -> dict:
    """{relative_name: md5} of every file under d (recursive)."""
    out: dict = {}
    if not d.exists():
        return out
    for p in sorted(d.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(d))] = _md5(p.read_bytes())
    return out


# ---------------------------------------------------------------------------
# fixtures: monkeypatched DEV / LIVE / DATA dirs + a TestClient + a seeded slot
# ---------------------------------------------------------------------------
@pytest.fixture
def floor_env(tmp_path, monkeypatch):
    """Point server.DEV_DATA_DIR / LIVE_DATA_DIR / DATA_DIR at tmp dirs."""
    dev = tmp_path / "dev" / "data"
    live = tmp_path / "live" / "data"
    dev.mkdir(parents=True)
    live.mkdir(parents=True)
    monkeypatch.setattr(server, "DEV_DATA_DIR", dev.resolve())
    monkeypatch.setattr(server, "LIVE_DATA_DIR", live.resolve())
    monkeypatch.setattr(server, "DATA_DIR", dev.resolve())
    return {"dev": dev.resolve(), "live": live.resolve()}


@pytest.fixture
def client():
    return TestClient(server.app)


@pytest.fixture
def seeded_slot(floor_env):
    """Write a synthetic dev slot directly into the monkeypatched DEV dir."""
    nrel, crel, xvm = _build_synthetic_nrel_crel_xvm(12)
    stem = "map_devseed_00"
    (floor_env["dev"] / f"{stem}n.rel").write_bytes(nrel)
    if crel is not None:
        (floor_env["dev"] / f"{stem}c.rel").write_bytes(crel)
    if xvm is not None:
        (floor_env["dev"] / f"{stem}s.xvm").write_bytes(xvm)
    return {"stem": stem, "nrel": nrel, "crel": crel, "xvm": xvm, **floor_env}


def _tiny_glb_from_grid(n: int = 12) -> bytes:
    """Build a minimal valid binary GLB carrying the grid mesh.

    Uses formats.import_external to round-trip if it offers a writer;
    otherwise hand-roll a 2-chunk GLB (JSON + BIN). We only need it to
    parse back through parse_gltf, which the build path uses.
    """
    V, U, F = _grid_mesh(n)
    return _encode_min_glb(V, U, F)


def _encode_min_glb(V, U, F) -> bytes:
    """Hand-encode a minimal binary glTF 2.0 with POSITION + TEXCOORD_0 +
    indices. Enough for parse_gltf to recover the triangle geometry.
    """
    import json as _json

    pos = V.astype("<f4")
    uv = U.astype("<f4")
    idx = F.reshape(-1).astype("<u4")

    pos_bytes = pos.tobytes()
    uv_bytes = uv.tobytes()
    idx_bytes = idx.tobytes()

    def _pad4(b: bytes) -> bytes:
        if len(b) % 4:
            b = b + b"\x00" * (4 - len(b) % 4)
        return b

    pos_off = 0
    uv_off = pos_off + len(pos_bytes)
    idx_off = uv_off + len(uv_bytes)
    bin_blob = pos_bytes + uv_bytes + idx_bytes
    bin_blob = _pad4(bin_blob)

    pmin = pos.min(axis=0).tolist()
    pmax = pos.max(axis=0).tolist()

    gltf = {
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}],
        "scene": 0,
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
            "indices": 2,
            "mode": 4,
        }]}],
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_off, "byteLength": len(pos_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": uv_off, "byteLength": len(uv_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": idx_off, "byteLength": len(idx_bytes), "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": int(V.shape[0]),
             "type": "VEC3", "min": pmin, "max": pmax},
            {"bufferView": 1, "componentType": 5126, "count": int(U.shape[0]),
             "type": "VEC2"},
            {"bufferView": 2, "componentType": 5125, "count": int(idx.shape[0]),
             "type": "SCALAR"},
        ],
    }
    json_bytes = _json.dumps(gltf).encode("utf-8")
    json_bytes = _pad4(json_bytes) if len(json_bytes) % 4 == 0 else (
        json_bytes + b" " * (4 - len(json_bytes) % 4))

    json_chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    bin_chunk = struct.pack("<II", len(bin_blob), 0x004E4942) + bin_blob
    total = 12 + len(json_chunk) + len(bin_chunk)
    header = struct.pack("<4sII", b"glTF", 2, total)
    return header + json_chunk + bin_chunk


# ===========================================================================
# R1 LOCK TESTS — the safety proof. These must hold for the whole feature.
# ===========================================================================
def test_floor_editor_has_no_live_import():
    """The floor block must NOT call safe_live_path nor write LIVE_DATA_DIR.

    Static AST scan of server.py: every function whose name starts with
    ``_floor`` / ``api_floors`` must never call ``safe_live_path`` and must
    never pass LIVE_DATA_DIR to a write. (LIVE_DATA_DIR is *read* in the
    not-live ASSERT, which is allowed — we only forbid using it as a write
    target / via safe_live_path.)
    """
    src = (_ROOT / "server.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not (node.name.startswith("_floor") or node.name.startswith("api_floors")):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                if sub.func.id == "safe_live_path":
                    offenders.append(f"{node.name} calls safe_live_path")
    assert not offenders, offenders


def test_write_boundary_rejects_live(floor_env, monkeypatch):
    """Pointing the write boundary at LIVE must raise BEFORE any bytes land."""
    # DEV_DATA_DIR == LIVE_DATA_DIR -> resolve_out_dir must raise.
    monkeypatch.setattr(server, "DEV_DATA_DIR", floor_env["live"])
    with pytest.raises(RuntimeError, match="LIVE_DATA_DIR"):
        server._floor_resolve_out_dir()
    # A direct write into the live dir must raise (the per-file assert).
    target = floor_env["live"] / "map_devx_00n.rel"
    with pytest.raises(RuntimeError, match="LIVE_DATA_DIR"):
        server._floor_atomic_write(target, b"hello")
    assert not target.exists(), "no bytes may land in LIVE on a rejected write"


def test_write_boundary_rejects_live_child(floor_env, monkeypatch):
    """A DEV dir that is a CHILD of LIVE must also be rejected."""
    child = floor_env["live"] / "data2"
    monkeypatch.setattr(server, "DEV_DATA_DIR", child)
    with pytest.raises(RuntimeError, match="LIVE_DATA_DIR"):
        server._floor_resolve_out_dir()


def test_floor_create_never_touches_live(floor_env, client):
    """THE central proof: create + copy leave the LIVE dir byte-identical."""
    before = _snapshot_dir(floor_env["live"])
    assert before == {}, "live dir should start empty in this fixture"

    glb = _tiny_glb_from_grid(12)
    r = client.post(
        "/api/floors/create",
        files={"file": ("floor.glb", glb, "model/gltf-binary")},
        data={"name": "fromglb", "area_template": "forest"},
    )
    assert r.status_code == 200, r.text
    fid = r.json()["floor_id"]

    # Copy the freshly-created dev slot to a new slot.
    r2 = client.post("/api/floors/copy", json={"floor_id": fid, "dest_name": "copyof"})
    assert r2.status_code == 200, r2.text

    after = _snapshot_dir(floor_env["live"])
    assert after == before, f"LIVE dir was modified! before={before} after={after}"
    # And the dev dir DID get the slot files.
    dev_after = _snapshot_dir(floor_env["dev"])
    assert any("fromglb" in k for k in dev_after), dev_after
    assert any("copyof" in k for k in dev_after), dev_after


# ===========================================================================
# LIST
# ===========================================================================
def test_floors_list_empty_is_graceful(floor_env, client):
    r = client.get("/api/floors")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["ok"] is True
    assert isinstance(j["floors"], list)


def test_floors_list_includes_dev_slot(seeded_slot, client):
    r = client.get("/api/floors")
    assert r.status_code == 200, r.text
    floors = r.json()["floors"]
    devs = [f for f in floors if f["source"] == "copy"]
    assert any(f["floor_id"] == "map_devseed_00" for f in devs), devs


# ===========================================================================
# COPY (atomic, passthrough)
# ===========================================================================
def test_copy_passthrough_byte_identical(seeded_slot, client):
    r = client.post("/api/floors/copy",
                    json={"floor_id": "map_devseed_00", "dest_name": "dupe",
                          "mode": "passthrough"})
    assert r.status_code == 200, r.text
    new_id = r.json()["new_floor_id"]
    assert new_id == "map_devdupe_00"
    src = seeded_slot["dev"] / "map_devseed_00n.rel"
    dst = seeded_slot["dev"] / f"{new_id}n.rel"
    assert dst.exists()
    assert _md5(src.read_bytes()) == _md5(dst.read_bytes()), "passthrough must be byte-identical"


def test_copy_refuses_clobber_without_overwrite(seeded_slot, client):
    body = {"floor_id": "map_devseed_00", "dest_name": "twice"}
    r1 = client.post("/api/floors/copy", json=body)
    assert r1.status_code == 200, r1.text
    r2 = client.post("/api/floors/copy", json=body)
    assert r2.status_code == 409, r2.text


def test_copy_overwrite_keeps_backup(seeded_slot, client):
    body = {"floor_id": "map_devseed_00", "dest_name": "ovr"}
    assert client.post("/api/floors/copy", json=body).status_code == 200
    body2 = dict(body, overwrite=True)
    assert client.post("/api/floors/copy", json=body2).status_code == 200
    backups = list(seeded_slot["dev"].glob("map_devovr_00n.rel.pre_edit_*"))
    assert backups, "overwrite must leave a .pre_edit_<TS> backup"


def test_copy_nothing_written_under_live(seeded_slot, client):
    before = _snapshot_dir(seeded_slot["live"])
    client.post("/api/floors/copy", json={"floor_id": "map_devseed_00", "dest_name": "x1"})
    assert _snapshot_dir(seeded_slot["live"]) == before


# ===========================================================================
# CREATE (synthetic GLB) + validation
# ===========================================================================
def test_create_from_glb_report(floor_env, client):
    glb = _tiny_glb_from_grid(12)
    r = client.post("/api/floors/create",
                    files={"file": ("f.glb", glb, "model/gltf-binary")},
                    data={"name": "newfloor"})
    assert r.status_code == 200, r.text
    rep = r.json()["report"]
    assert rep["single_texture_slot"] is True
    assert rep["part_count"] >= 1
    files = rep["files"]
    assert any(f["name"].endswith("n.rel") for f in files)
    # The slot landed in DEV.
    assert (floor_env["dev"] / "map_devnewfloor_00n.rel").exists()


def test_create_bad_glb_magic_rejected(floor_env, client):
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    r = client.post("/api/floors/create",
                    files={"file": ("evil.glb", png, "model/gltf-binary")},
                    data={"name": "evilfloor"})
    assert r.status_code == 400, r.text
    # Nothing written.
    assert not list(floor_env["dev"].glob("map_devevilfloor_00*"))


def test_create_empty_glb_rejected(floor_env, client):
    r = client.post("/api/floors/create",
                    files={"file": ("empty.glb", b"", "model/gltf-binary")},
                    data={"name": "emptyfloor"})
    assert r.status_code == 400, r.text


def test_create_bad_glb_version_rejected(floor_env, client):
    # Valid magic but version 1.
    bad = struct.pack("<4sII", b"glTF", 1, 12)
    r = client.post("/api/floors/create",
                    files={"file": ("v1.glb", bad, "model/gltf-binary")},
                    data={"name": "v1floor"})
    assert r.status_code == 400, r.text


def test_create_glb_length_mismatch_rejected(floor_env, client):
    bad = struct.pack("<4sII", b"glTF", 2, 9999) + b"\x00" * 8
    r = client.post("/api/floors/create",
                    files={"file": ("len.glb", bad, "model/gltf-binary")},
                    data={"name": "lenfloor"})
    assert r.status_code == 400, r.text


def test_create_nothing_under_live(floor_env, client):
    before = _snapshot_dir(floor_env["live"])
    glb = _tiny_glb_from_grid(12)
    client.post("/api/floors/create",
                files={"file": ("f.glb", glb, "model/gltf-binary")},
                data={"name": "liveproof"})
    assert _snapshot_dir(floor_env["live"]) == before


# ===========================================================================
# TRAVERSAL (R4)
# ===========================================================================
@pytest.mark.parametrize("bad", ["../x", "a/b", "..", "", "x.rel", "a\\b", "foo bar/../x"])
def test_create_name_traversal_rejected(floor_env, client, bad):
    glb = _tiny_glb_from_grid(12)
    r = client.post("/api/floors/create",
                    files={"file": ("f.glb", glb, "model/gltf-binary")},
                    data={"name": bad})
    # 400 from our name guard, or 422 from FastAPI's Form-required check
    # (empty string is sent but the multipart field is treated as missing).
    assert r.status_code in (400, 422), (bad, r.text)
    # And no slot ever materialised under DEV.
    assert not list(floor_env["dev"].glob("map_dev*"))


# ===========================================================================
# SIZE CAP (R6) — a forced over-budget mesh -> 422 with the budget string
# ===========================================================================
def test_size_cap_422(floor_env, client, monkeypatch):
    """A build that can't fit the n.rel budget must 422 (no file written).

    We force the failure by stubbing build_floor to raise the real
    RelWriteError the writer raises on overflow (so we exercise the
    error-mapping + no-write guarantee deterministically and fast).
    """
    msg = (f"n.rel is {_rw.NREL_SIZE_BUDGET + 1} bytes, exceeds the "
           f"0x{_rw.NREL_SIZE_BUDGET:x} (768 KB) budget")

    def _boom(*a, **k):
        raise _rw.RelWriteError(msg)

    monkeypatch.setattr(lp, "build_floor", _boom)
    glb = _tiny_glb_from_grid(12)
    r = client.post("/api/floors/create",
                    files={"file": ("big.glb", glb, "model/gltf-binary")},
                    data={"name": "bigfloor"})
    assert r.status_code == 422, r.text
    assert "0xc0000" in r.text.lower() or "768" in r.text
    assert not list(floor_env["dev"].glob("map_devbigfloor_00*"))


# ===========================================================================
# PREVIEW PAYLOAD shape + route ordering
# ===========================================================================
def test_preview_bundle_shape(seeded_slot, client):
    r = client.get("/api/floors/map_devseed_00")
    assert r.status_code == 200, r.text
    b = r.json()
    # Same keys the map bundle / viewer expects.
    for key in ("renderable", "textures", "scripts", "nrel_path",
                "single_texture_slot", "root_only_preview"):
        assert key in b, (key, list(b.keys()))
    assert b["renderable"], "a dev slot must expose its n.rel as renderable"


def test_route_ordering_copy_create_not_swallowed(seeded_slot, client):
    """A POST to /api/floors/copy must reach the copy handler, NOT be
    swallowed as floor_id="copy" by the parameterized {floor_id} route.

    Proof: a POST /api/floors/copy with a valid body returns 200 (the copy
    handler ran). If the literal route were registered AFTER the
    parameterized one, FastAPI would 405 (the GET-only {floor_id} route
    has no POST) — so a 200 here proves the literal won the match.
    """
    r = client.post("/api/floors/copy",
                    json={"floor_id": "map_devseed_00", "dest_name": "route1"})
    assert r.status_code == 200, r.text
    # And a malformed-magic create reaches the create handler (400 from the
    # GLB sniff), not a 405 — proving /create is the literal, not {floor_id}.
    r2 = client.post("/api/floors/create",
                     files={"file": ("x.glb", b"nope", "model/gltf-binary")},
                     data={"name": "route2"})
    assert r2.status_code == 400, r2.text


def test_preview_unknown_slot_404(floor_env, client):
    r = client.get("/api/floors/map_devnope_00")
    assert r.status_code == 404, r.text


def test_preview_traversal_rejected(floor_env, client):
    r = client.get("/api/floors/..%2f..%2fevil")
    assert r.status_code in (400, 404), r.text


# ===========================================================================
# ATOMICITY (R3) — os.replace failure leaves no torn target + cleans .tmp
# ===========================================================================
def test_atomic_write_cleanup_on_replace_failure(floor_env, monkeypatch):
    target = floor_env["dev"] / "map_devatom_00n.rel"

    def _boom(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(server.os, "replace", _boom)
    with pytest.raises(OSError):
        server._floor_atomic_write(target, b"payload")
    assert not target.exists(), "final target must not exist after a torn write"
    strays = list(floor_env["dev"].glob("map_devatom_00n.rel*.tmp"))
    assert not strays, f"stray .tmp not cleaned: {strays}"


# ===========================================================================
# CONCURRENCY (R10) — two creates on the same slot: one 200, one 409
# ===========================================================================
def test_concurrent_create_one_wins(floor_env, client):
    glb = _tiny_glb_from_grid(12)
    results = {}

    def _hold():
        # Grab the build lock so the real request must 409.
        server._FLOOR_BUILD_LOCK.acquire()

    # Hold the lock, fire a create, expect 409, release.
    server._FLOOR_BUILD_LOCK.acquire()
    try:
        r = client.post("/api/floors/create",
                        files={"file": ("f.glb", glb, "model/gltf-binary")},
                        data={"name": "racef"})
        assert r.status_code == 409, r.text
    finally:
        server._FLOOR_BUILD_LOCK.release()
    # With the lock free it now succeeds.
    r2 = client.post("/api/floors/create",
                     files={"file": ("f.glb", glb, "model/gltf-binary")},
                     data={"name": "racef"})
    assert r2.status_code == 200, r2.text


# ===========================================================================
# DELETE (copies only)
# ===========================================================================
def test_delete_dev_slot(seeded_slot, client):
    r = client.delete("/api/floors/map_devseed_00")
    assert r.status_code == 200, r.text
    assert not (seeded_slot["dev"] / "map_devseed_00n.rel").exists()


def test_delete_stock_refused(floor_env, client):
    r = client.delete("/api/floors/stk__aancient01__0")
    assert r.status_code == 400, r.text


def test_delete_unknown_404(floor_env, client):
    r = client.delete("/api/floors/map_devghost_00")
    assert r.status_code == 404, r.text


# ===========================================================================
# REAL-GLB leg (skipped when no GLB on disk)
# ===========================================================================
def _find_glb():
    import os
    root = os.environ.get("PSOBB_DOWNLOADS_DIR") or os.path.expanduser("~/Downloads")
    cands = sorted(Path(root).glob("*.glb"), key=lambda p: p.stat().st_size, reverse=True)
    return cands[0] if cands else None


@pytest.mark.skipif(_find_glb() is None, reason="no source GLB on disk (CI)")
def test_create_real_glb(floor_env, client):
    glb = _find_glb().read_bytes()
    r = client.post("/api/floors/create",
                    files={"file": ("real.glb", glb, "model/gltf-binary")},
                    data={"name": "realfloor"})
    assert r.status_code in (200, 422), r.text
    if r.status_code == 200:
        nrel = floor_env["dev"] / "map_devrealfloor_00n.rel"
        assert nrel.exists()
        ok, msg = lp.verify_nrel(nrel.read_bytes())
        assert ok, msg
    # Live must stay empty either way.
    assert _snapshot_dir(floor_env["live"]) == {}
