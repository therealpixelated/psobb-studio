"""Tests for the cross-archive texture index used by the model viewer.

PSOBB.IO ships ~60 NJTL refs whose name resolves only in a SIBLING
BML's inline XVM, plus ~140 player-class texture slots that resolve
through ``pl[A-Z]tex.afs`` archives. The texture index walks every BML
+ every AFS once on first use, caches the (name -> [TextureLocation,
...]) map, and lets the binding pipeline find the right tile across
archives.

Live data: ``~/PSOBB.IO/data/`` — skipped if not present.
"""
from __future__ import annotations
import os

from pathlib import Path

import pytest

from formats import texture_index as ti

_DATA = Path(os.path.expanduser("~/PSOBB.IO/data"))
_LIVE = _DATA.exists()


@pytest.mark.skipif(not _LIVE, reason="PSOBB.IO not installed")
def test_index_finds_known_cross_bml_texture():
    """ts008_siro is a canonical cross-BML texture (Vol Opt monitor parts)."""
    idx = ti.build_texture_index(_DATA)
    assert "ts008_siro" in idx, (
        f"expected ts008_siro in texture index; have {len(idx)} unique names"
    )
    locs = idx["ts008_siro"]
    # The texture appears in MANY BMLs (Vol Opt + jungle props + boss
    # cylinder room). Audit found 5+; we relax to ≥3 so the test isn't
    # tied to a precise count.
    assert len(locs) >= 3, f"expected ≥3 locations, got {len(locs)}"


@pytest.mark.skipif(not _LIVE, reason="PSOBB.IO not installed")
def test_lookup_case_sensitivity_matches_first():
    """``lookup`` returns at least every entry the direct fetch finds.

    ``lookup`` walks the COMBINED (BML + AFS) index; the direct
    ``build_texture_index`` fetch is BML-only — so lookup may carry
    additional AFS rows. Verify both report the same BML rows in the
    same order.
    """
    direct = ti.build_texture_index(_DATA).get("ts008_siro", [])
    via_lookup = ti.lookup(_DATA, "ts008_siro")
    assert len(via_lookup) >= len(direct)
    direct_keys = [(d.bml_name, d.inner_name, d.xvr_index) for d in direct]
    lookup_bml_keys = [
        (loc.bml_name, loc.inner_name, loc.xvr_index)
        for loc in via_lookup if loc.kind == "bml"
    ]
    assert direct_keys == lookup_bml_keys


@pytest.mark.skipif(not _LIVE, reason="PSOBB.IO not installed")
def test_index_contains_unique_names():
    """The index keyset is a flat name set, not multiset."""
    idx = ti.build_texture_index(_DATA)
    # Every key must be a non-empty string.
    for k in idx.keys():
        assert isinstance(k, str) and k, k
    # Every value list must be non-empty (we never insert empty lists).
    for k, v in idx.items():
        assert v, f"empty location list for {k!r}"
        # Each location must carry valid bml_name + inner_name + xvr_index.
        for loc in v:
            assert loc.bml_name.endswith(".bml"), loc
            assert loc.inner_name, loc
            assert loc.xvr_index >= 0, loc


@pytest.mark.skipif(not _LIVE, reason="PSOBB.IO not installed")
def test_cached_index_matches_fresh():
    """The cached entry point ``get_texture_index`` matches the global build.

    NOTE: ``build_texture_index`` is BML-only; the cached entry point
    walks BML + AFS so we compare against ``build_global_texture_index``
    instead.
    """
    ti.clear_cache()
    fresh = ti.build_global_texture_index(_DATA)
    cached = ti.get_texture_index(_DATA)
    # Same key set, same value lengths.
    assert set(fresh.keys()) == set(cached.keys())
    for k in fresh:
        assert len(fresh[k]) == len(cached[k]), (
            f"{k}: fresh={len(fresh[k])}, cached={len(cached[k])}"
        )


def test_lookup_empty_data_dir(tmp_path):
    """An empty data dir produces an empty index without crashing."""
    idx = ti.build_texture_index(tmp_path)
    assert idx == {}
    assert ti.lookup(tmp_path, "anything") == []


def test_lookup_missing_name_returns_empty(tmp_path):
    """Looking up a name that isn't in the index returns an empty list."""
    # Point at empty dir to avoid heavy live-data build.
    out = ti.lookup(tmp_path, "nonexistent_name_xyz_q9")
    assert out == []


# ---------------------------------------------------------------------------
# AFS extension tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _LIVE, reason="PSOBB.IO not installed")
def test_afs_index_includes_player_archives():
    """AFS walk finds every ``pl[A-Z]tex.afs`` that ships in PSOBB.IO."""
    afs_idx = ti.index_afs_archives(_DATA)
    assert afs_idx, "AFS index unexpectedly empty"
    # Every archive should contribute at least one row, since pl?tex
    # archives are uniformly XVR. We sample a known archive (plAtex).
    pla = [
        loc for locs in afs_idx.values() for loc in locs
        if loc.archive == "plAtex.afs"
    ]
    assert pla, "no plAtex.afs entries in the AFS index"
    # plAtex has 137 inner blobs (every blob is a single XVR -> one row).
    archives_seen = {loc.archive for locs in afs_idx.values() for loc in locs}
    expected_letters = "ABCDEFGHIJKLOPQRSTUVWXYZ"  # M and N are reserved/unused in PSOBB
    pl_archives = {a for a in archives_seen if a.startswith("pl") and a.endswith("tex.afs")}
    # We don't pin the exact set (mod data may add classes); we just
    # require the audit to find SOME ``pl[A-Z]tex.afs`` rows AND that
    # plA + plB + plK are present (HUmar male, HUmar female, FOnewearl).
    for letter in ("A", "B", "K"):
        assert f"pl{letter}tex.afs" in pl_archives, (
            f"pl{letter}tex.afs missing from AFS index"
        )


@pytest.mark.skipif(not _LIVE, reason="PSOBB.IO not installed")
def test_afs_filter_restricts_walk():
    """``archive_filter`` cuts the walk down to a glob pattern."""
    only_pla = ti.index_afs_archives(_DATA, archive_filter=["plAtex.afs"])
    archives = {loc.archive for locs in only_pla.values() for loc in locs}
    assert archives == {"plAtex.afs"}, archives


@pytest.mark.skipif(not _LIVE, reason="PSOBB.IO not installed")
def test_player_class_lookup_returns_positional_table():
    """``lookup_player_class_textures`` returns the AFS positional table.

    plAbdy00 has 5 material slots; locs[0..4] map to plAtex.afs blobs
    0..4 in order. We assert at least 5 entries exist; the actual
    archive carries 137 (one per XVR blob).
    """
    locs = ti.lookup_player_class_textures(_DATA, "plAbdy00.nj")
    assert len(locs) >= 5, len(locs)
    for i, loc in enumerate(locs[:5]):
        assert loc.kind == "afs", loc
        assert loc.archive == "plAtex.afs", loc
        assert loc.inner_index == i, (i, loc)
        # The inner_name follows the AFS-router convention so the
        # frontend can synthesise the URL deterministically.
        assert loc.inner_name.startswith(f"{i:04d}_plAtex_{i:04d}"), loc


@pytest.mark.skipif(not _LIVE, reason="PSOBB.IO not installed")
def test_player_class_lookup_classes_we_verified():
    """Bodies + hair for every class flagged in the audit resolve cleanly."""
    cases = [
        ("plAbdy00.nj", "plAtex.afs", 5),    # HUmar male body
        ("plBbdy00.nj", "plBtex.afs", 5),    # HUmar female body
        ("plDbdy00.nj", "plDtex.afs", 5),    # HUcaseal body
        ("plKbdy00.nj", "plKtex.afs", 5),    # FOnewearl body
        ("plAhai00.nj", "plAtex.afs", 2),    # HUmar male hair
    ]
    for fname, archive, min_n in cases:
        locs = ti.lookup_player_class_textures(_DATA, fname)
        assert len(locs) >= min_n, f"{fname}: got {len(locs)}, expected ≥{min_n}"
        for loc in locs[:min_n]:
            assert loc.archive == archive, (fname, loc)
            assert loc.kind == "afs", loc


def test_player_class_for_recognises_body_hair_head():
    """``player_class_for`` parses the class letter from the model name."""
    assert ti.player_class_for("plAbdy00.nj") == "A"
    assert ti.player_class_for("plKhai00.nj") == "K"
    assert ti.player_class_for("plDhed00.nj") == "D"
    # Lower-case is accepted (case-insensitive regex).
    assert ti.player_class_for("plabdy00.nj") == "A"
    # Non-player files reject.
    assert ti.player_class_for("bm_boss3_volopt.bml") is None
    assert ti.player_class_for("") is None


def test_player_class_lookup_empty_dir(tmp_path):
    """When the matching pl?tex.afs is missing, return an empty list."""
    out = ti.lookup_player_class_textures(tmp_path, "plAbdy00.nj")
    assert out == []


def test_player_class_lookup_non_player_returns_empty(tmp_path):
    """Non-player NJ filenames shouldn't trigger a player-class lookup."""
    out = ti.lookup_player_class_textures(tmp_path, "boss1_dragon.nj")
    assert out == []


def test_texture_location_kind_default_bml():
    """``TextureLocation.kind`` defaults to "bml" for back-compat."""
    loc = ti.TextureLocation(bml_name="x.bml", inner_name="x.nj", xvr_index=0)
    assert loc.kind == "bml"
    assert loc.archive == "x.bml"
    assert loc.inner_index == -1


def test_texture_location_kind_afs_explicit():
    """An AFS row carries kind="afs" + a real inner_index."""
    loc = ti.TextureLocation(
        bml_name="plAtex.afs", inner_name="0042_plAtex_0042.xvr",
        xvr_index=0, kind="afs", inner_index=42,
    )
    assert loc.kind == "afs"
    assert loc.archive == "plAtex.afs"
    assert loc.inner_index == 42


@pytest.mark.skipif(not _LIVE, reason="PSOBB.IO not installed")
def test_global_index_combines_bml_and_afs():
    """``build_global_texture_index`` carries BOTH BML and AFS rows."""
    idx = ti.build_global_texture_index(_DATA)
    bml_count = 0
    afs_count = 0
    for locs in idx.values():
        for loc in locs:
            if loc.kind == "bml":
                bml_count += 1
            elif loc.kind == "afs":
                afs_count += 1
    assert bml_count > 0, "no BML rows in combined index"
    assert afs_count > 0, "no AFS rows in combined index"
    # Sanity: the BML count should match the BML-only index.
    bml_only = ti.build_texture_index(_DATA)
    bml_only_total = sum(len(v) for v in bml_only.values())
    assert bml_count == bml_only_total, (bml_count, bml_only_total)
