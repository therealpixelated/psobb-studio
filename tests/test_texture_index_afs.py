"""Regression tests for the ItemModel cross-AFS texture binding path.

These tests validate the fix that migrates the 354 ItemModel /
ItemModelEp4 inners flagged ``ok_no_textures`` by
``scripts/render_coverage_audit.py`` over to a real ``cross_afs``
binding. The fix has two pieces:

1. ``texture_index._index_one_afs`` deep-walks every blob in
   ``ItemTexture.afs`` and ``ItemTextureEp4.afs`` to emit one row per
   XVR sub-record. The schema bumps to v3.
2. ``texture_index.lookup_item_textures`` is a positional resolver:
   model slot N's K-th NJTL entry resolves to texture-archive slot N's
   K-th XVR record.

Live data: ``~/PSOBB.IO/data/`` — skipped if not present.
"""
from __future__ import annotations
import os

import json
from pathlib import Path

import pytest

from formats import texture_index as ti

_DATA = Path(os.path.expanduser("~/PSOBB.IO/data"))
_LIVE = _DATA.exists()


@pytest.mark.skipif(not _LIVE, reason="PSOBB.IO not installed")
def test_force_rebuild_emits_per_xvr_subrows(tmp_path, monkeypatch):
    """A force-rebuild populates per-XVR sub-rows for ItemTexture.

    Before the v3 schema, ItemTexture.afs blobs were indexed only as
    outer XVMH placeholders (xvr_index=-1). After the bump, every XVR
    record inside each blob also gets its own row keyed on
    ``ItemTexture_NNNN_MMMM`` so the positional resolver can pin a
    real record.
    """
    # Redirect the on-disk cache to tmp_path so we don't pollute the
    # live editor cache, AND so we can verify the schema field on disk.
    monkeypatch.setattr(ti, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(ti, "_CACHE_FILE", tmp_path / "texture_index.json")
    ti.clear_cache()

    idx = ti.build_global_texture_index(_DATA)
    # Every name -> at least one location.
    assert idx, "global texture index unexpectedly empty"

    # ItemTexture sub-rows: name shape is "ItemTexture_<NNNN>_<MMMM>".
    sub_rows = [
        loc
        for locs in idx.values() for loc in locs
        if loc.archive in ("ItemTexture.afs", "ItemTextureEp4.afs")
        and loc.xvr_index >= 0
    ]
    assert sub_rows, "no per-XVR sub-rows emitted for ItemTexture archives"
    # Sanity-check: there are at least 700 sub-rows (live data has
    # ~370 ItemTexture inners + ~502 ItemTextureEp4, with most carrying
    # 2-4 XVRs each).
    assert len(sub_rows) >= 700, (
        f"expected ≥700 ItemTexture sub-rows, got {len(sub_rows)}"
    )

    # Save + reload via the disk cache to verify the schema field.
    cache_key = ti._cache_key(_DATA)
    ti._save_cache(idx, cache_key)
    payload = json.loads((tmp_path / "texture_index.json").read_text("utf-8"))
    assert int(payload.get("version", 0)) == 3, (
        f"expected schema v3, got {payload.get('version')}"
    )


@pytest.mark.skipif(not _LIVE, reason="PSOBB.IO not installed")
def test_lookup_item_textures_returns_positional_table(tmp_path, monkeypatch):
    """ItemModel.afs#0050 has 2 NJTL slots → lookup returns 2 textures."""
    monkeypatch.setattr(ti, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(ti, "_CACHE_FILE", tmp_path / "texture_index.json")
    ti.clear_cache()

    locs = ti.lookup_item_textures(_DATA, "ItemModel.afs", 50)
    # We verified manually that ItemModel.afs#0050 references 2
    # textures (wxmS02e_z_w_huda + wxtS01c_z_w_katana01) and that
    # ItemTexture.afs#0050 holds an XVMH with 2 XVR records.
    assert len(locs) == 2, f"expected 2 textures, got {len(locs)}"
    # The list is sorted positionally so locs[K] is the K-th XVR record.
    for k, loc in enumerate(locs):
        assert loc.kind == "afs", loc
        assert loc.archive == "ItemTexture.afs", loc
        assert loc.inner_index == 50, (k, loc)
        assert loc.xvr_index == k, (k, loc)


@pytest.mark.skipif(not _LIVE, reason="PSOBB.IO not installed")
def test_lookup_item_textures_ep4_archive_pair(tmp_path, monkeypatch):
    """ItemModelEp4.afs maps to ItemTextureEp4.afs, not the Ep1-3 archive."""
    monkeypatch.setattr(ti, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(ti, "_CACHE_FILE", tmp_path / "texture_index.json")
    ti.clear_cache()

    locs = ti.lookup_item_textures(_DATA, "ItemModelEp4.afs", 0)
    assert locs, "ItemModelEp4#0 should have at least one texture"
    for loc in locs:
        assert loc.archive == "ItemTextureEp4.afs", loc
        assert loc.kind == "afs", loc
        assert loc.inner_index == 0, loc


@pytest.mark.skipif(not _LIVE, reason="PSOBB.IO not installed")
def test_lookup_item_textures_unknown_archive_returns_empty(tmp_path, monkeypatch):
    """Non-ItemModel archives should return [] cleanly."""
    monkeypatch.setattr(ti, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(ti, "_CACHE_FILE", tmp_path / "texture_index.json")
    ti.clear_cache()

    # plAtex.afs is a player-class texture archive, not an item model
    # archive — it should produce no rows here.
    locs = ti.lookup_item_textures(_DATA, "plAtex.afs", 0)
    assert locs == []
    locs = ti.lookup_item_textures(_DATA, "nonexistent.afs", 0)
    assert locs == []


@pytest.mark.skipif(not _LIVE, reason="PSOBB.IO not installed")
def test_lookup_item_textures_negative_index_returns_empty():
    """Defensive: negative inner_index should not raise."""
    locs = ti.lookup_item_textures(_DATA, "ItemModel.afs", -1)
    assert locs == []


def test_item_archive_for_recognises_known_pair():
    """ItemModel.afs <-> ItemTexture.afs and Ep4 sibling resolve."""
    assert ti.item_archive_for("ItemModel.afs") == "ItemTexture.afs"
    assert ti.item_archive_for("ItemModelEp4.afs") == "ItemTextureEp4.afs"
    # Case-insensitive on the basename for URL-encoded path callers.
    assert ti.item_archive_for("itemmodel.afs") == "ItemTexture.afs"
    # Path basename is what we test — passing a fragment with directories
    # still resolves the basename.
    assert ti.item_archive_for("foo/ItemModelEp4.afs") == "ItemTextureEp4.afs"
    # Unknown archives return None.
    assert ti.item_archive_for("plAtex.afs") is None
    assert ti.item_archive_for("") is None
    assert ti.item_archive_for("ItemSomethingElse.afs") is None


@pytest.mark.skipif(not _LIVE, reason="PSOBB.IO not installed")
def test_server_binding_uses_cross_afs_for_item_models():
    """The server-side ``_build_model_texture_binding`` call returns
    ``cross_afs`` rows for an ItemModel inner.

    This is the closest-to-end-to-end check we can make without
    spinning up the FastAPI app: the binding builder is the single
    place every model-bundle / model-textures / model-mesh endpoint
    leans on.
    """
    # Import the server-side helper. Skip if server.py isn't import-
    # able (it pulls in FastAPI; we only need the function in tests
    # where the editor environment is fully assembled).
    pytest.importorskip("fastapi")
    import server as _srv  # noqa: WPS433
    # Force a fresh on-disk index so the v3 deep walk is in effect.
    ti.clear_cache()

    afs_path = _DATA / "ItemModel.afs"
    # Pull the inner bytes through the same path as the API endpoints.
    # The inner name shape is "NNNN_<basename>.<ext>" — the audit
    # synthesises ".nj" for ItemModel inners.
    inner = "0050_ItemModel_0050.nj"
    blob, logical = _srv._read_afs_inner_nj(afs_path, inner)
    # Parse it through the cached parser so we get a representative
    # mesh list (with material_ids).
    meshes = _srv._cached_model_parse(blob, afs_path, ".afs", ".nj", inner)
    out = _srv._build_model_texture_binding(
        afs_path, ".afs", inner, blob, meshes,
    )
    assert "binding" in out, out
    # At least one binding row should resolve via cross_afs to
    # ItemTexture.afs#0050.
    cross_afs_rows = [
        row for row in out["binding"]
        if row.get("source") == "cross_afs"
    ]
    assert cross_afs_rows, (
        f"expected at least one cross_afs binding row, got "
        f"{[r.get('source') for r in out['binding']]}"
    )
    for row in cross_afs_rows:
        ca = row.get("cross_afs") or {}
        assert ca.get("archive") == "ItemTexture.afs", ca
        assert ca.get("inner_index") == 50, ca
        assert ca.get("xvr_index", -1) >= 0, ca
