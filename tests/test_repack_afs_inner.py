"""Tests for /api/repack_afs_inner — repack one inner XVM of an AFS archive.

The endpoint is the AFS-aware companion to /api/repack: it materialises
the inner via the standard afs_reader cache, runs the same xvr_codec
rebuild pipeline, and either mints an export token (deploy=False) or
splices the rebuilt bytes back into the parent AFS atomically.

Round-trip coverage:
  - empty `tiles` list ⇒ no-op rebuild ⇒ rebuilt inner XVM has XVMH magic
    and the export token returns a downloadable artifact.
  - bad inner_index ⇒ 400.
  - archive missing ⇒ 404.

Skipped when the live PSOBB install or the DEV mirror's AFS copies
aren't reachable (CI build, fresh checkout). The endpoint shouldn't
require those to exist for unit tests, but the underlying
materialise_inner code path reads the archive bytes off disk so we
can't fake it cheaply without a fixture archive.
"""
from __future__ import annotations
import os

import sys
from pathlib import Path

import pytest

from fastapi.testclient import TestClient


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Resolve a small AFS fixture. ItemKT.afs is one of the smallest shipped
# (~2 MB) so the no-op rebuild stays fast even on a cold cache.
_DEV_AFS = Path(r"C:/tmp_pso_dev/data/ItemKT.afs")
_LIVE_AFS = Path(os.path.expanduser("~/PSOBB.IO/data/ItemKT.afs"))
_HAS_AFS = _DEV_AFS.exists() or _LIVE_AFS.exists()


@pytest.fixture(scope="module")
def client():
    """In-process FastAPI client (one server import per module)."""
    import server
    return TestClient(server.app)


@pytest.mark.skipif(not _HAS_AFS, reason="ItemKT.afs missing in DEV/LIVE data dir")
def test_repack_afs_inner_noop_round_trip(client):
    """No-op repack of inner 0 should return a rebuilt XVM with XVMH magic.

    Empty tiles list ⇒ xvr_codec.rebuild splices every original .xvr
    block verbatim ⇒ rebuilt inner XVM is bit-identical to the cached
    materialised input. We export the artifact via the minted token
    and assert the on-the-wire bytes start with XVMH.
    """
    r = client.post(
        "/api/repack_afs_inner",
        json={
            "archive": "ItemKT.afs",
            "inner_index": 0,
            "tiles": [],
            "deploy": False,
        },
    )
    assert r.status_code == 200, f"unexpected status: {r.status_code} body={r.text}"
    body = r.json()

    # Shape contract.
    assert body["archive"] == "ItemKT.afs"
    assert body["inner_index"] == 0
    assert body["rebuilt_size"] > 0
    assert body["verify"]["ok"] is True
    assert body["changed_indices"] == []
    assert body["export_token"], "expected an export token for deploy=False"
    assert body["export_url"].startswith("/api/export/")

    # Deploy fields must be None when deploy=False.
    assert body["deploy_path"] is None
    assert body["afs_backup_path"] is None
    assert body["afs_original_size"] is None
    assert body["afs_new_size"] is None

    # No-op rebuild means xvr_codec spliced every tile, none re-encoded.
    if body["spliced_count"] is not None and body["reencoded_count"] is not None:
        assert body["reencoded_count"] == 0
        assert body["spliced_count"] >= 1

    # Pull the rebuilt artifact through the export endpoint and verify
    # the XVMH magic. This proves the full extract → rebuild → token
    # path is wired correctly.
    download = client.get(body["export_url"])
    assert download.status_code == 200
    payload = download.content
    assert len(payload) == body["rebuilt_size"], "exported size != reported rebuilt_size"
    assert payload[:4] == b"XVMH", f"rebuilt inner missing XVMH magic: {payload[:8].hex()}"


@pytest.mark.skipif(not _HAS_AFS, reason="ItemKT.afs missing in DEV/LIVE data dir")
def test_repack_afs_inner_bad_index_400(client):
    """An inner_index past the archive's slot count should 400, not 500."""
    # Use an in-range-but-still-out-of-bounds index. ItemKT.afs has well
    # under 1000 slots; 9999 is below the pydantic 0xFFFF cap so the
    # request reaches our handler instead of bouncing on schema 422.
    r = client.post(
        "/api/repack_afs_inner",
        json={
            "archive": "ItemKT.afs",
            "inner_index": 9999,
            "tiles": [],
            "deploy": False,
        },
    )
    # 400 (handler-level "out of range") or 404 (materialise_inner raises
    # IndexError → HTTPException(404) in the materialise helper). Both
    # are acceptable client-error codes for this case.
    assert r.status_code in (400, 404), f"expected 400/404, got {r.status_code}: {r.text}"


def test_repack_afs_inner_missing_archive_404(client):
    """A nonexistent AFS name should return 404 from the base resolver."""
    r = client.post(
        "/api/repack_afs_inner",
        json={
            "archive": "this_does_not_exist_xyz_42.afs",
            "inner_index": 0,
            "tiles": [],
            "deploy": False,
        },
    )
    assert r.status_code == 404, f"expected 404, got {r.status_code}: {r.text}"


def test_repack_afs_inner_rejects_non_afs_400(client):
    """A bare filename without .afs extension should 400."""
    r = client.post(
        "/api/repack_afs_inner",
        json={
            "archive": "Foo.xvm",
            "inner_index": 0,
            "tiles": [],
            "deploy": False,
        },
    )
    assert r.status_code == 400, f"expected 400, got {r.status_code}: {r.text}"


def test_repack_afs_inner_rejects_path_traversal_400(client):
    """A path-component-bearing archive name should 400 from the validator."""
    r = client.post(
        "/api/repack_afs_inner",
        json={
            "archive": "../etc/passwd.afs",
            "inner_index": 0,
            "tiles": [],
            "deploy": False,
        },
    )
    assert r.status_code == 400, f"expected 400, got {r.status_code}: {r.text}"
