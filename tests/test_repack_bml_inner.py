"""Tests for /api/repack_bml_inner — repack one inner XVM of a BML container.

The endpoint is the BML-aware companion to /api/repack: it materialises
the inner via the standard _materialize_inner_for_extract cache, runs
the same xvr_codec rebuild pipeline, and either mints an export token
(deploy=False) or splices the rebuilt bytes back into the parent BML
atomically using formats.bml.pack_bml.

Round-trip coverage:
  - empty `tiles` list ⇒ no-op rebuild ⇒ rebuilt inner XVM has XVMH magic
    and the export token returns a downloadable artifact.
  - bad inner_name ⇒ 404.
  - non-.xvm inner_name ⇒ 400 (mesh-payload edits go through
    /api/import/replace, not this endpoint).
  - BML missing ⇒ 404.
  - non-.bml extension ⇒ 400.
  - path-component traversal ⇒ 400.

Skipped when the live PSOBB install or the DEV mirror's BML copies
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


# Resolve a BML fixture. bm_obj_ep4_boss09_core.bml is from the EP4 boss
# (Saint-Million / Pan Arms / Ramar) and ships per the
# psobb_full_entity_map memory note. It has 3 entries each with a
# texture archive (per the BML reader output).
_DEV_BML = Path(r"C:/tmp_pso_dev/data/bm_obj_ep4_boss09_core.bml")
_LIVE_BML = Path(os.path.expanduser("~/PSOBB.IO/data/bm_obj_ep4_boss09_core.bml"))
_HAS_BML = _DEV_BML.exists() or _LIVE_BML.exists()

# The first entry's name in the BML — verified by parsing the live archive.
# This is the inner name with the '.xvm' suffix: extract_bml_texture
# resolves '<entry>.xvm' to the per-entry texture archive that follows
# the inner NJ payload inside the BML.
_TEX_INNER = "bm_obj_ep4_boss09_core01.nj.xvm"


@pytest.fixture(scope="module")
def client():
    """In-process FastAPI client (one server import per module)."""
    import server
    return TestClient(server.app)


@pytest.mark.skipif(not _HAS_BML, reason="bm_obj_ep4_boss09_core.bml missing in DEV/LIVE data dir")
def test_repack_bml_inner_noop_round_trip(client):
    """No-op repack of inner texture should return a rebuilt XVM with XVMH magic.

    Empty tiles list ⇒ xvr_codec.rebuild splices every original .xvr
    block verbatim ⇒ rebuilt inner XVM is bit-identical to the cached
    materialised input. We export the artifact via the minted token
    and assert the on-the-wire bytes start with XVMH.
    """
    r = client.post(
        "/api/repack_bml_inner",
        json={
            "bml": "bm_obj_ep4_boss09_core.bml",
            "inner_name": _TEX_INNER,
            "tiles": [],
            "deploy": False,
        },
    )
    assert r.status_code == 200, f"unexpected status: {r.status_code} body={r.text}"
    body = r.json()

    # Shape contract.
    assert body["bml"] == "bm_obj_ep4_boss09_core.bml"
    assert body["inner_name"] == _TEX_INNER
    assert body["rebuilt_size"] > 0
    assert body["verify"]["ok"] is True
    assert body["changed_indices"] == []
    assert body["export_token"], "expected an export token for deploy=False"
    assert body["export_url"].startswith("/api/export/")

    # Deploy fields must be None when deploy=False.
    assert body["deploy_path"] is None
    assert body["bml_backup_path"] is None
    assert body["bml_original_size"] is None
    assert body["bml_new_size"] is None

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


@pytest.mark.skipif(not _HAS_BML, reason="bm_obj_ep4_boss09_core.bml missing in DEV/LIVE data dir")
def test_repack_bml_inner_bad_inner_404(client):
    """An inner_name that doesn't exist in the BML should 404 from materialise."""
    r = client.post(
        "/api/repack_bml_inner",
        json={
            "bml": "bm_obj_ep4_boss09_core.bml",
            "inner_name": "this_inner_does_not_exist_42.xvm",
            "tiles": [],
            "deploy": False,
        },
    )
    # 404 from _extract_bml_inner_bytes when the entry name lookup fails.
    assert r.status_code == 404, f"expected 404, got {r.status_code}: {r.text}"


@pytest.mark.skipif(not _HAS_BML, reason="bm_obj_ep4_boss09_core.bml missing in DEV/LIVE data dir")
def test_repack_bml_inner_rejects_mesh_payload_400(client):
    """A non-.xvm inner_name (mesh payload) should 400 — that's not this endpoint's job."""
    r = client.post(
        "/api/repack_bml_inner",
        json={
            "bml": "bm_obj_ep4_boss09_core.bml",
            "inner_name": "bm_obj_ep4_boss09_core01.nj",
            "tiles": [],
            "deploy": False,
        },
    )
    assert r.status_code == 400, f"expected 400, got {r.status_code}: {r.text}"
    # Sanity: error message points the caller at the right endpoint.
    assert ".xvm" in r.text or "import/replace" in r.text


def test_repack_bml_inner_missing_bml_404(client):
    """A nonexistent BML name should return 404 from the base resolver."""
    r = client.post(
        "/api/repack_bml_inner",
        json={
            "bml": "this_does_not_exist_xyz_42.bml",
            "inner_name": "anything.xvm",
            "tiles": [],
            "deploy": False,
        },
    )
    assert r.status_code == 404, f"expected 404, got {r.status_code}: {r.text}"


def test_repack_bml_inner_rejects_non_bml_400(client):
    """A bare filename without .bml extension should 400."""
    r = client.post(
        "/api/repack_bml_inner",
        json={
            "bml": "Foo.xvm",
            "inner_name": "x.xvm",
            "tiles": [],
            "deploy": False,
        },
    )
    assert r.status_code == 400, f"expected 400, got {r.status_code}: {r.text}"


def test_repack_bml_inner_rejects_path_traversal_400(client):
    """A path-component-bearing BML name should 400 from the validator."""
    r = client.post(
        "/api/repack_bml_inner",
        json={
            "bml": "../etc/passwd.bml",
            "inner_name": "x.xvm",
            "tiles": [],
            "deploy": False,
        },
    )
    assert r.status_code == 400, f"expected 400, got {r.status_code}: {r.text}"


def test_repack_bml_inner_rejects_path_traversal_inner_400(client):
    """A path-component-bearing inner_name should 400 from the validator."""
    r = client.post(
        "/api/repack_bml_inner",
        json={
            "bml": "bm_obj_ep4_boss09_core.bml",
            "inner_name": "../escape.xvm",
            "tiles": [],
            "deploy": False,
        },
    )
    assert r.status_code == 400, f"expected 400, got {r.status_code}: {r.text}"


def test_repack_bml_inner_rejects_empty_inner_400(client):
    """Empty inner_name should 400."""
    r = client.post(
        "/api/repack_bml_inner",
        json={
            "bml": "bm_obj_ep4_boss09_core.bml",
            "inner_name": "",
            "tiles": [],
            "deploy": False,
        },
    )
    assert r.status_code == 400, f"expected 400, got {r.status_code}: {r.text}"
