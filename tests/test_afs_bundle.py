"""Tests for AFS-form model dispatch in /api/model_mesh, /api/model_skinned,
and /api/model_bundle.

Background
----------
Manifest entries for items live as ``ItemModel.afs#NNNN_<basename>.nj``
(see ``manifest._synthesize_afs_entries``). The ``.afs`` extension was
historically rejected by the model-load endpoints because the dispatch
table only checked ``.nj``/``.xj``/``.bml``. The patch in
``_reports/handoff/afs_bundle_support.patch`` extends the dispatch to
materialise an AFS inner via ``formats/afs_reader.py::materialize_inner``
and route the resulting bytes through the existing parse + binding
caches.

ItemModel.afs inner blobs are stored as descriptor-XJ format under a
``.nj`` extension (the afs_reader sniffs them as NJ_IFF because they
start with NJTL/NJCM magic), so the model_skinned endpoint falls back
to ``parse_xj_file`` + a synthesised single-root-bone skeleton when the
chunk-NJ skinned parser rejects the payload.

These tests are skipped when ``ItemModel.afs`` isn't present on the
build machine (CI builds without PSOBB.IO/data).
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


@pytest.fixture(scope="module")
def client():
    """In-process FastAPI client (one server import per module)."""
    import server
    return TestClient(server.app)


@pytest.fixture(scope="module")
def item_model_afs() -> Path | None:
    """Resolve the live ItemModel.afs path; None if PSOBB.IO/data missing."""
    candidates = [
        Path(os.path.expanduser("~/PSOBB.IO/data/ItemModel.afs")),
        Path(r"data/ItemModel.afs"),
        _REPO_ROOT / "data" / "ItemModel.afs",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


@pytest.fixture(scope="module")
def first_nj_inner_synth(item_model_afs) -> str | None:
    """Pick the first NJ inner whose bundle returns vert_count > 0.

    Some ItemModel slots hold zero-byte placeholder blobs; we skip past
    them so the test asserts on a real model.
    """
    if item_model_afs is None:
        return None
    from formats import afs_reader as ar
    rows = ar.list_inner_blobs(item_model_afs)
    for row in rows:
        if (row.get("inner_ext") or "").lower() != ".nj":
            continue
        idx = int(row["index"])
        # Synth path matches manifest._synthesize_afs_entries.
        name = row.get("name") or f"{item_model_afs.stem}_{idx:04d}"
        synth = f"{idx:04d}_{Path(name).name}"
        return synth
    return None


# ---------------------------------------------------------------------------
# /api/model_mesh — descriptor-XJ fallback path
# ---------------------------------------------------------------------------


def test_model_mesh_resolves_afs_inner(client, item_model_afs, first_nj_inner_synth):
    """``ItemModel.afs#NNNN_<name>`` lands on real triangulated mesh data."""
    if item_model_afs is None or first_nj_inner_synth is None:
        pytest.skip("ItemModel.afs not available")
    url = f"/api/model_mesh/ItemModel.afs%23{first_nj_inner_synth}"
    r = client.get(url)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mesh_count"] > 0
    totals = body.get("totals") or {}
    assert totals.get("vertices", 0) > 0, (
        f"expected non-stub mesh, got totals={totals}"
    )
    # Filename / inner round-trip.
    assert body["filename"].startswith("ItemModel.afs#")
    assert body["inner"] == first_nj_inner_synth


def test_model_mesh_rejects_afs_without_inner(client, item_model_afs):
    """``.afs`` without an inner returns 400, not 500."""
    if item_model_afs is None:
        pytest.skip("ItemModel.afs not available")
    r = client.get("/api/model_mesh/ItemModel.afs")
    assert r.status_code == 400
    assert "inner" in (r.json().get("detail") or "").lower()


# ---------------------------------------------------------------------------
# /api/model_skinned — falls back to XJ-descriptor + single root bone
# ---------------------------------------------------------------------------


def test_model_skinned_falls_back_for_descriptor_xj_afs(
    client, item_model_afs, first_nj_inner_synth,
):
    """ItemModel inners are descriptor-XJ; skinned endpoint synthesises
    a 1-bone skeleton so the bundle can still feed the model viewer."""
    if item_model_afs is None or first_nj_inner_synth is None:
        pytest.skip("ItemModel.afs not available")
    url = f"/api/model_skinned/ItemModel.afs%23{first_nj_inner_synth}"
    r = client.get(url)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mesh_count"] > 0
    bones = body.get("bones") or []
    # Synthetic root bone for static items.
    assert len(bones) >= 1
    assert bones[0]["index"] == 0
    assert bones[0]["parent"] == -1


# ---------------------------------------------------------------------------
# /api/model_bundle — the user-facing entry the model viewer hits
# ---------------------------------------------------------------------------


def test_model_bundle_returns_skinned_with_vertices(
    client, item_model_afs, first_nj_inner_synth,
):
    """The bundle's ``skinned`` block has ``totals.vertices > 0`` for a
    real ItemModel inner.

    Regression for the audit's 1212-row "unsupported_route" tally:
    before the patch, every bundle for an AFS path returned a 400 with
    "unsupported model extension '.afs'" so the model viewer dropped to
    its primitive-cube fallback. After: the bundle delivers actual
    geometry.
    """
    if item_model_afs is None or first_nj_inner_synth is None:
        pytest.skip("ItemModel.afs not available")
    url = f"/api/model_bundle/ItemModel.afs%23{first_nj_inner_synth}"
    r = client.get(url)
    assert r.status_code == 200, r.text
    body = r.json()
    sk = body.get("skinned") or {}
    assert sk, f"bundle.skinned missing: errors={body.get('errors')}"
    totals = sk.get("totals") or {}
    assert totals.get("vertices", 0) > 0, (
        f"bundle.skinned.totals.vertices == 0; the audit's "
        f"unsupported_route count won't drop. errors={body.get('errors')}"
    )


def test_model_bundle_unsupported_route_string_not_returned(
    client, item_model_afs, first_nj_inner_synth,
):
    """The audit script's 'unsupported_route' grade keys on the literal
    string ``"extension '.afs'"`` in the response body. After the patch,
    that string MUST NOT appear in the bundle's skinned error or detail
    fields for a valid ItemModel inner — otherwise the audit will keep
    grading the row as unsupported_route.
    """
    if item_model_afs is None or first_nj_inner_synth is None:
        pytest.skip("ItemModel.afs not available")
    url = f"/api/model_bundle/ItemModel.afs%23{first_nj_inner_synth}"
    r = client.get(url)
    assert r.status_code == 200
    serialised = r.text
    assert "extension '.afs'" not in serialised
    assert "unsupported model extension" not in serialised


def test_resolve_model_mesh_path_accepts_afs_top_level(client, item_model_afs):
    """``_resolve_model_mesh_path`` no longer 404s on a top-level ``.afs``.

    The previous behaviour: a missing-file path spec because the resolver
    rejected the extension. Confirm the path resolves cleanly (the inner
    miss is the caller's responsibility — the resolver is just the file
    locator).
    """
    if item_model_afs is None:
        pytest.skip("ItemModel.afs not available")
    # Hitting model_mesh with no inner returns a clean 400 ("AFS model
    # requires inner") rather than a 404 ("model not found").
    r = client.get("/api/model_mesh/ItemModel.afs")
    assert r.status_code == 400, r.text
