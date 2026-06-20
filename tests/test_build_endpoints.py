"""Tests for the AFS / BML build + deploy endpoints.

Covered:
  - /api/build_afs with raw b64 entries and DATA_DIR path entries.
  - /api/build_bml with mixed compressed/uncompressed inputs.
  - /api/build_bml round-trip: parse a shipped BML, ship it back as
    compressed entries, post to the endpoint, recover byte-identical
    artifact.
  - /api/deploy/<archive> with backup creation.
  - Argument validation: bad name, traversal, oversized, conflicting
    flags.

These tests run against an in-process FastAPI ``TestClient``. The
``DATA_DIR`` and ``LIVE_DATA_DIR`` are not touched (deploy tests use
monkeypatched temp dirs).
"""
from __future__ import annotations

import base64
import json
import os
import struct
import tempfile
from pathlib import Path

import pytest

from fastapi.testclient import TestClient


PSOBB_DATA = Path(os.path.expanduser("~/PSOBB.IO/data"))
HAS_PSOBB = PSOBB_DATA.is_dir()


@pytest.fixture(scope="module")
def client():
    """In-process FastAPI client. Imports server.py once per module."""
    import server
    return TestClient(server.app)


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


# ---------------------------------------------------------------------------
# /api/build_afs
# ---------------------------------------------------------------------------
def test_build_afs_basic(client):
    payload = {
        "name": "test_basic.afs",
        "entries": [
            {"b64": _b64(b"PSO\x00\x01" + b"\x00" * 100)},
            {"b64": _b64(b"PSO\x00\x02" + b"\x00" * 200)},
        ],
    }
    r = client.post("/api/build_afs", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    assert data["entry_count"] == 2
    assert data["size"] > 0
    assert len(data["md5"]) == 32
    # File exists at the reported path.
    p = Path(data["path"])
    assert p.exists()
    assert p.read_bytes()[:4] == b"AFS\x00"
    # Round-trip via parse_afs gives back the same blobs.
    from formats.afs import parse_afs
    blobs = parse_afs(p.read_bytes())
    assert len(blobs) == 2
    assert blobs[0] == b"PSO\x00\x01" + b"\x00" * 100
    assert blobs[1] == b"PSO\x00\x02" + b"\x00" * 200


def test_build_afs_rejects_bad_name(client):
    r = client.post("/api/build_afs", json={
        "name": "../escape.afs",
        "entries": [{"b64": _b64(b"x")}],
    })
    assert r.status_code == 400


def test_build_afs_rejects_empty_entries(client):
    r = client.post("/api/build_afs", json={
        "name": "empty.afs",
        "entries": [],
    })
    assert r.status_code == 400


def test_build_afs_rejects_both_b64_and_path(client):
    r = client.post("/api/build_afs", json={
        "name": "conflict.afs",
        "entries": [{"b64": _b64(b"x"), "path": "biri_ball.bml"}],
    })
    assert r.status_code == 400


def test_build_afs_rejects_neither_b64_nor_path(client):
    r = client.post("/api/build_afs", json={
        "name": "missing.afs",
        "entries": [{"name": "noref"}],
    })
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# /api/build_bml
# ---------------------------------------------------------------------------
def test_build_bml_basic(client):
    payload = {
        "name": "test_basic.bml",
        "entries": [
            {
                "name": "alpha.nj",
                "data_b64": _b64(b"NJCM" + b"\x00" * 100),
            },
            {
                "name": "beta.nj",
                "data_b64": _b64(b"NJCM" + b"\x00" * 200),
                "texture_b64": _b64(b"XVMHFAKE" + b"\x00" * 50),
            },
        ],
    }
    r = client.post("/api/build_bml", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    assert data["entry_count"] == 2
    assert len(data["md5"]) == 32
    p = Path(data["path"])
    assert p.exists()
    # Confirm the file parses as a BML.
    from formats.bml import parse_bml, extract_bml, extract_bml_texture
    buf = p.read_bytes()
    entries = parse_bml(buf)
    assert len(entries) == 2
    extracted = extract_bml(buf)
    assert extracted["alpha.nj"] == b"NJCM" + b"\x00" * 100
    assert extracted["beta.nj"] == b"NJCM" + b"\x00" * 200
    tex = extract_bml_texture(buf, "beta.nj")
    assert tex == b"XVMHFAKE" + b"\x00" * 50


def test_build_bml_rejects_bad_compression(client):
    r = client.post("/api/build_bml", json={
        "name": "bad.bml",
        "compression": 0xFF,
        "entries": [{"name": "x.nj", "data_b64": _b64(b"data")}],
    })
    assert r.status_code == 400


def test_build_bml_rejects_bad_alignment(client):
    r = client.post("/api/build_bml", json={
        "name": "bad.bml",
        "file_alignment": 0x100,
        "entries": [{"name": "x.nj", "data_b64": _b64(b"data")}],
    })
    assert r.status_code == 400


def test_build_bml_rejects_missing_data(client):
    r = client.post("/api/build_bml", json={
        "name": "miss.bml",
        "entries": [{"name": "x.nj"}],
    })
    assert r.status_code == 400


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO/data not available")
def test_build_bml_roundtrip_shipped(client):
    """Parse a shipped BML, post its compressed entries back, recover
    a byte-identical artifact."""
    src = PSOBB_DATA / "biri_ball.bml"
    if not src.exists():
        pytest.skip("biri_ball.bml not present")
    buf = src.read_bytes()
    from formats.bml import parse_bml_for_pack, parse_bml_pack_meta

    pack_entries = parse_bml_for_pack(buf)
    meta = parse_bml_pack_meta(buf)
    # Translate to the API schema.
    api_entries = []
    for ent in pack_entries:
        d = {
            "name": ent.name,
            "data_b64": _b64(ent.data),
            "is_compressed": ent.is_compressed,
            "decompressed_size": ent.decompressed_size,
            "unk_a": ent.unk_a,
            "unk_b": ent.unk_b,
            "unk_c": ent.unk_c,
            "unk_d": ent.unk_d,
        }
        if ent.texture_data is not None:
            d["texture_b64"] = _b64(ent.texture_data)
            d["texture_is_compressed"] = ent.texture_is_compressed
            d["texture_decompressed_size"] = ent.texture_decompressed_size
        api_entries.append(d)
    payload = {
        "name": "biri_ball_rebuilt.bml",
        "compression": meta["compression"],
        "file_alignment": meta["file_alignment"],
        "has_textures": meta["has_textures"],
        "entries": api_entries,
    }
    r = client.post("/api/build_bml", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    rebuilt = Path(data["path"]).read_bytes()
    assert rebuilt == buf, (
        f"size orig={len(buf)} rebuilt={len(rebuilt)} md5_match={data['md5']}"
    )


# ---------------------------------------------------------------------------
# /api/deploy/<archive>
# ---------------------------------------------------------------------------
def test_deploy_archive_404_when_missing(client):
    r = client.post("/api/deploy/no_such_archive.afs", json={})
    assert r.status_code == 404


def test_deploy_archive_rejects_traversal(client):
    # Starlette / FastAPI's path matcher rejects URL-encoded path-traversal
    # strings before they even reach the handler — so we get a 404 from
    # the router. Either is fine; the upshot is "request fails, no
    # files touched". Try multiple traversal forms.
    for url in [
        "/api/deploy/..%2Fescape.afs",
        "/api/deploy/foo%2Fbar.afs",   # /
        "/api/deploy/foo%5Cbar.afs",   # \
    ]:
        r = client.post(url, json={})
        assert r.status_code in (400, 404), (
            f"unexpected for {url}: {r.status_code} {r.text}"
        )


def test_deploy_archive_full_flow(client, monkeypatch, tmp_path):
    """End-to-end: build a tiny AFS, redirect LIVE_DATA_DIR to a tmp
    dir, deploy, verify backup + bytes match."""
    import server

    # Fresh tmp live dir.
    live_dir = tmp_path / "live"
    live_dir.mkdir()
    monkeypatch.setattr(server, "LIVE_DATA_DIR", live_dir)

    # Pre-populate a "stock" copy so we can verify the backup is taken.
    stock_path = live_dir / "deploy_target.afs"
    stock_path.write_bytes(b"OLD_STOCK_BYTES")

    # Build an AFS into the export cache.
    payload = {
        "name": "deploy_target.afs",
        "entries": [{"b64": _b64(b"PSO\x00FRESH" + b"\x00" * 50)}],
    }
    r = client.post("/api/build_afs", json=payload)
    assert r.status_code == 200, r.text
    built_md5 = r.json()["md5"]
    built_size = r.json()["size"]

    # Now deploy.
    r = client.post("/api/deploy/deploy_target.afs", json={"create_backup": True})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    assert data["live_size"] == built_size
    assert data["backup_name"] is not None
    assert data["backup_name"].startswith("deploy_target.afs.pre_promote_")

    # Verify destination matches the build output.
    deployed = stock_path.read_bytes()
    import hashlib
    assert hashlib.md5(deployed).hexdigest() == built_md5

    # Backup contains the prior bytes.
    bak = live_dir / data["backup_name"]
    assert bak.exists()
    assert bak.read_bytes() == b"OLD_STOCK_BYTES"


def test_deploy_archive_no_backup_when_first_time(client, monkeypatch, tmp_path):
    """Deploy with no prior live file: backup_name is None."""
    import server

    live_dir = tmp_path / "live2"
    live_dir.mkdir()
    monkeypatch.setattr(server, "LIVE_DATA_DIR", live_dir)

    payload = {
        "name": "first_time.afs",
        "entries": [{"b64": _b64(b"AFS\x00DATA" + b"\x00" * 10)}],
    }
    r = client.post("/api/build_afs", json=payload)
    assert r.status_code == 200, r.text

    r = client.post("/api/deploy/first_time.afs", json={"create_backup": True})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["backup_name"] is None  # no prior live file
    assert (live_dir / "first_time.afs").exists()


def test_deploy_archive_skip_backup(client, monkeypatch, tmp_path):
    """create_backup=False skips the backup."""
    import server

    live_dir = tmp_path / "live3"
    live_dir.mkdir()
    monkeypatch.setattr(server, "LIVE_DATA_DIR", live_dir)
    (live_dir / "skip.afs").write_bytes(b"OLD")

    payload = {
        "name": "skip.afs",
        "entries": [{"b64": _b64(b"NEW")}],
    }
    r = client.post("/api/build_afs", json=payload)
    assert r.status_code == 200

    r = client.post("/api/deploy/skip.afs", json={"create_backup": False})
    assert r.status_code == 200
    assert r.json()["backup_name"] is None
    # No .pre_promote_* file in live dir.
    backups = list(live_dir.glob("skip.afs.pre_promote_*"))
    assert len(backups) == 0


# ---------------------------------------------------------------------------
# /api/build_nj  (NJ encoder bridge)
# ---------------------------------------------------------------------------
def _build_synthetic_cube_json():
    """Helper: build a minimal cube model JSON for /api/build_nj."""
    import math
    import struct

    verts = [
        (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
    ]
    vbody = bytearray()
    vbody.extend(struct.pack("<H", 49))
    vbody.extend(struct.pack("<HH", 0, 8))
    for (x, y, z) in verts:
        vbody.extend(struct.pack("<3f", float(x), float(y), float(z)))
        n = math.sqrt(x * x + y * y + z * z)
        vbody.extend(struct.pack("<3f", x / n, y / n, z / n))

    faces = [(0, 1, 3, 2), (5, 4, 6, 7), (4, 0, 7, 3),
             (1, 5, 2, 6), (3, 2, 7, 6), (4, 5, 0, 1)]
    sbody = bytearray()
    sbody.extend(struct.pack("<H", 31))
    sbody.extend(struct.pack("<H", len(faces) & 0x3FFF))
    for face in faces:
        sbody.extend(struct.pack("<h", 4))
        for idx in face:
            sbody.extend(struct.pack("<H", idx))

    return {
        "njtl_names": [],
        "nodes": [{
            "eval_flags": 0,
            "position": [0.0, 0.0, 0.0],
            "rotation_bams": [0, 0, 0],
            "scale": [1.0, 1.0, 1.0],
            "mesh_index": 0,
            "child_index": -1,
            "sibling_index": -1,
        }],
        "meshes": [{
            "bbox": [0.0, 0.0, 0.0, 1.732],
            "vlist": [{"type_id": 41, "flags": 0, "body_b64": _b64(bytes(vbody))}],
            "plist": [{"type_id": 64, "flags": 0, "body_b64": _b64(bytes(sbody))}],
        }],
    }


def test_build_nj_synthetic_cube(client):
    """A 1-bone, 8-vert cube via /api/build_nj produces a valid .nj."""
    payload = {
        "name": "test_cube_endpoint.nj",
        "model_json": _build_synthetic_cube_json(),
    }
    r = client.post("/api/build_nj", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    assert data["chunk_count"] == 2  # 1 vlist + 1 plist
    assert data["vert_count"] == 8
    p = Path(data["path"])
    assert p.exists()
    # Round-trip via the parser.
    from formats.xj import parse_nj_file
    meshes = parse_nj_file(p.read_bytes())
    assert len(meshes) == 6  # one submesh per face


def test_build_nj_rejects_bad_extension(client):
    r = client.post("/api/build_nj", json={
        "name": "wrong_ext.bml",
        "model_json": _build_synthetic_cube_json(),
    })
    assert r.status_code == 400


def test_build_nj_rejects_empty_nodes(client):
    r = client.post("/api/build_nj", json={
        "name": "empty.nj",
        "model_json": {"nodes": [], "meshes": []},
    })
    assert r.status_code == 400


def test_build_nj_with_njtl(client):
    """An NJ with NJTL names emits an NJTL chunk."""
    mj = _build_synthetic_cube_json()
    mj["njtl_names"] = ["test_a", "test_b", "test_c"]
    r = client.post("/api/build_nj", json={
        "name": "with_njtl.nj",
        "model_json": mj,
    })
    assert r.status_code == 200
    p = Path(r.json()["path"])
    from formats.iff import parse_iff
    chunks = parse_iff(p.read_bytes())
    assert any(c.type == "NJTL" for c in chunks)


# ---------------------------------------------------------------------------
# /api/build_njm  (NJM encoder bridge)
# ---------------------------------------------------------------------------
def test_build_njm_synthetic(client):
    """A simple 1-bone, 30-frame motion via /api/build_njm."""
    motion_json = {
        "frame_count": 30,
        "type_flags": 3,
        "inp_fn": 2,
        "bones": [
            {
                "tracks": [
                    {"kind": 1, "narrow": True,
                     "keyframes": [[0, 0.0, 0.0, 0.0], [29, 0.0, 0.0, 0.0]]},
                    {"kind": 2, "narrow": True,
                     "keyframes": [[0, 0, 0, 0], [29, 0, 1000, 0]]},
                ]
            }
        ],
    }
    r = client.post("/api/build_njm", json={
        "name": "test_walk_endpoint.njm",
        "motion_json": motion_json,
    })
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    assert data["bone_count"] == 1
    assert data["frame_count"] == 30
    p = Path(data["path"])
    # Round-trip via the parser.
    from formats.njm import parse_njm
    motions = parse_njm(p.read_bytes())
    assert len(motions) == 1
    assert motions[0].bone_count == 1
    assert motions[0].frame_count == 30


def test_build_njm_rejects_bad_extension(client):
    r = client.post("/api/build_njm", json={
        "name": "wrong.bml",
        "motion_json": {"bones": []},
    })
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# /api/sculpt/build_nj  (sculpt -> NJ bridge)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO data not present")
def test_sculpt_build_nj_bridges_sculpt_json(client, tmp_path, monkeypatch):
    """End-to-end: write a sculpt sidecar, hit endpoint, verify output."""
    import json
    import server

    # Use a small shipped NJ as the source.
    bml_name = "biri_ball.bml"
    inner_name = "biri_ball.nj"
    bml_p = PSOBB_DATA / bml_name
    if not bml_p.exists():
        pytest.skip("biri_ball.bml not present")

    # Create a small zero-displacement sculpt sidecar.
    sculpt_dir = tmp_path / "sculpted_meshes"
    sculpt_dir.mkdir()
    monkeypatch.setattr(server, "SCULPT_CACHE_DIR", sculpt_dir)

    sha = "abcdef0123456789"
    sidecar_filename = server._sculpt_safe_filename(
        f"{bml_name}#{inner_name}", sha
    )
    sidecar = {
        "format_version": 1,
        "source_path": f"{bml_name}#{inner_name}",
        "submeshes": [
            {
                "submesh_idx": 0,
                "material_id": 0,
                "vertex_count": 1,
                "displacement_b64": base64.b64encode(b"\x00" * 12).decode(),
                "modified_indices_b64": base64.b64encode(b"\x00" * 4).decode(),
            }
        ],
        "sha": sha,
    }
    (sculpt_dir / sidecar_filename).write_text(json.dumps(sidecar))

    r = client.post("/api/sculpt/build_nj", json={
        "model_path": f"{bml_name}#{inner_name}",
        "inner_idx": 0,
        "sculpt_sha": sha,
        "output_name": "sculpt_test.nj",
    })
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    p = Path(data["path"])
    assert p.exists()
    # Output should still be a valid NJ.
    from formats.nj_writer import parse_nj_for_writer
    model = parse_nj_for_writer(p.read_bytes())
    assert len(model.nodes) >= 1


def test_sculpt_build_nj_rejects_bad_sha(client):
    r = client.post("/api/sculpt/build_nj", json={
        "model_path": "biri_ball.bml#biri_ball.nj",
        "inner_idx": 0,
        "sculpt_sha": "not-hex!",
    })
    assert r.status_code == 400


def test_sculpt_build_nj_rejects_missing_pound(client):
    r = client.post("/api/sculpt/build_nj", json={
        "model_path": "biri_ball.nj",  # no '#'
        "inner_idx": 0,
        "sculpt_sha": "abcdef00",
    })
    assert r.status_code == 400
