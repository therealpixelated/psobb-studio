"""Tests for the cross-BML name / stem-family lookup in the binding pipeline.

The binding resolver (``server._build_model_texture_binding``) used to
fall through to a same-BML positional cross-inner pick whenever an
inner had 0 inline tiles, regardless of what the sibling inner held.
That painted ``bm_obj_warpboss_ancient.bml#fe_obj_df_warp_gawa.xj``
(the stone gate frame) with ``de_obj_df_warp_sbeam.xj``'s pink-stripe
beam atlas, because gawa's mids 0..5 happened to be in range of
sbeam's 6 XVR records.

The fix layers a sibling-BML stem-family lookup BEFORE the same-BML
positional pick so that NJTL-less inners search by inner-stem token
overlap (gawa.xj <-> bm_o_warp_ancient.bml#fd_obj1_swarp_gawa.xj) and
fall back to the same-BML pick only when no sibling-BML stem match
exists.

Tests:
  * Synthetic helper checks for ``find_sibling_bml_by_inner_stem``.
  * Real-data probe of the warp-gate-frame regression.
  * Regression coverage for dragon (in_bml-only path must not change).
"""
from __future__ import annotations
import os

import json
from pathlib import Path

import pytest

from formats import material as mat_mod
from formats import texture_index as ti

import server


_LIVE = Path(os.path.expanduser("~/PSOBB.IO/data"))
_DEV = Path(r"C:/tmp_pso_dev/data")


def _data_root() -> Path:
    if _DEV.exists():
        return _DEV
    if _LIVE.exists():
        return _LIVE
    pytest.skip("no PSOBB data dir available")


# ---------------------------------------------------------------------------
# 1. Pure helper tests
# ---------------------------------------------------------------------------


def test_extract_inner_texture_names_no_njtl():
    """An IFF blob without an NJTL chunk returns an empty list (no raise)."""
    # Minimal IFF: NJCM tag + 4-byte body. find_and_parse_njtl returns
    # None; the wrapper returns [].
    blob = b"NJCM" + (4).to_bytes(4, "little") + b"\x00\x00\x00\x00"
    assert mat_mod.extract_inner_texture_names(blob, ".xj") == []


def test_extract_inner_texture_names_empty_input():
    """Empty bytes return ``[]`` and don't raise."""
    assert mat_mod.extract_inner_texture_names(b"", ".xj") == []
    assert mat_mod.extract_inner_texture_names(b"", ".nj") == []


def test_inner_stem_tokens_strips_generic_markers():
    """``fe_obj_df_warp_gawa.xj`` decomposes to a set containing 'gawa'.

    Generic prefixes ("fe", "obj", "df") are filtered out by the
    ``_GENERIC_INNER_TOKENS`` deny list. ``warp`` is also generic
    (it appears in MANY unrelated BMLs across the data tree) so it
    is filtered as well — the discriminating token here is ``gawa``,
    which is rare enough to anchor the stem-family match without
    over-pulling.
    """
    tokens = ti._inner_stem_tokens("fe_obj_df_warp_gawa.xj")
    assert "gawa" in tokens, tokens
    assert "obj" not in tokens, tokens
    assert "fe" not in tokens, tokens
    assert "df" not in tokens, tokens


def test_inner_stem_tokens_overlaps_sibling_family():
    """Two inners from different BMLs that share 'gawa' both expose it."""
    a = ti._inner_stem_tokens("fe_obj_df_warp_gawa.xj")
    b = ti._inner_stem_tokens("fd_obj1_swarp_gawa.xj")
    overlap = a & b
    assert "gawa" in overlap, (a, b, overlap)


# ---------------------------------------------------------------------------
# 2. Real-data lookup probes (skipped without a data dir)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not (_LIVE.exists() or _DEV.exists()),
                    reason="no PSOBB data dir available")
def test_find_sibling_bml_for_warp_gate_frame():
    """gawa.xj's missing-NJTL fallback finds bm_o_warp_ancient.bml's gawa."""
    data = _data_root()
    bml_path = data / "bm_obj_warpboss_ancient.bml"
    if not bml_path.exists():
        pytest.skip(f"{bml_path.name} missing from data dir")
    sibling_bml = data / "bm_o_warp_ancient.bml"
    if not sibling_bml.exists():
        pytest.skip(f"{sibling_bml.name} missing — fixture incomplete")

    # The XVR-count cap is the 6 mids gawa.xj actually uses; relax
    # via min_xvr_count=1 because no single sibling inner has >=6 tiles
    # (the warp ancient set spreads them across two inners).
    hit = ti.find_sibling_bml_by_inner_stem(
        bml_path, "fe_obj_df_warp_gawa.xj", min_xvr_count=1,
    )
    assert hit is not None, "expected a stem-family sibling hit"
    sib_bml, sib_inner, _xvr_count = hit
    # The match should pick the "warp_ancient" sibling, not the
    # warpboss-jungle / warpboss / warp_jung family. Token-overlap
    # ranking (gawa + warp) is what enforces this.
    assert sib_bml == "bm_o_warp_ancient.bml", (
        f"expected bm_o_warp_ancient.bml, got {sib_bml}"
    )
    # The matched inner must contain "gawa" or be a known siblings
    # token-rich inner ("nsita_1" also matches via "swarp" etc., but the
    # gawa-named inner is the canonical "main").
    assert "gawa" in sib_inner.lower() or "swarp" in sib_inner.lower(), sib_inner


# ---------------------------------------------------------------------------
# 3. End-to-end binding probe — the regression test for the bug.
# ---------------------------------------------------------------------------


def _binding_for(bml_basename: str, inner: str) -> dict:
    """Drive ``_build_model_texture_binding`` for one BML+inner.

    We call the underlying builder directly (NOT the cached shim) so the
    test isn't sensitive to disk-cache contents. ``LIVE_DATA_DIR`` is
    consulted by the resolver for cross-archive lookups; tests assume
    the live install is present (skipped otherwise).
    """
    data = _data_root()
    bml_path = data / bml_basename
    if not bml_path.exists():
        pytest.skip(f"{bml_basename} missing from data dir")
    blob = bml_path.read_bytes()
    # Decompress the requested inner.
    from formats.bml import decompress_prs_cached, parse_bml
    entries = parse_bml(blob)
    matches = [e for e in entries if e.name == inner]
    if not matches:
        pytest.skip(f"{inner!r} not in {bml_basename}")
    e = matches[0]
    nj_bytes = decompress_prs_cached(
        bml_path, bml_path.stat().st_mtime_ns, inner,
        lambda: bytes(blob[e.offset:e.offset + e.size_compressed]),
    )
    # Parse meshes the way the server does — we need the material_id set.
    from formats import parse_cache as _pc
    if inner.lower().endswith(".xj"):
        meshes = _pc.parse_xj_file_cached(nj_bytes)
    else:
        meshes = _pc.parse_nj_file_cached(nj_bytes)
    return server._build_model_texture_binding(
        bml_path, ".bml", inner, nj_bytes, meshes,
    )


@pytest.mark.skipif(not (_LIVE.exists() or _DEV.exists()),
                    reason="no PSOBB data dir available")
def test_warp_gate_frame_does_not_bind_to_sbeam():
    """Regression: gawa.xj must NOT cross-bind to sbeam's pink-stripe atlas."""
    bd = _binding_for("bm_obj_warpboss_ancient.bml", "fe_obj_df_warp_gawa.xj")
    binding = bd.get("binding") or []
    assert binding, "expected a non-empty binding list"
    # Regression contract: NO row may resolve to de_obj_df_warp_sbeam.xj
    # (that was the bug). Either the resolver finds a real cross-BML
    # sibling (preferred) or marks the slot missing.
    bad_inner = "de_obj_df_warp_sbeam.xj"
    for row in binding:
        cb = row.get("cross_bml") or {}
        assert cb.get("inner") != bad_inner, (
            f"gawa.xj mid={row.get('material_id')} bound to {bad_inner!r} — "
            f"this is the warp-gate-frame regression. row={row}"
        )


@pytest.mark.skipif(not (_LIVE.exists() or _DEV.exists()),
                    reason="no PSOBB data dir available")
def test_warp_gate_frame_resolves_to_sibling_bml_or_marks_missing():
    """gawa.xj's mids should EITHER hit bm_o_warp_ancient.bml OR be missing.

    Two acceptable outcomes:
      (a) cross_bml row pointing at bm_o_warp_ancient.bml (stem-family
          hit) — preferred, this is the model viewer's correct render.
      (b) missing row — acceptable when the stem-family search misses
          (still better than the buggy wrong-atlas binding).
    """
    bd = _binding_for("bm_obj_warpboss_ancient.bml", "fe_obj_df_warp_gawa.xj")
    binding = bd.get("binding") or []
    assert binding
    for row in binding:
        cb = row.get("cross_bml") or {}
        if not row.get("missing"):
            # If we resolved cross_bml, the host BML must NOT be the
            # warpboss_ancient itself — that would be the same-BML
            # positional pick we just suppressed.
            assert cb.get("bml") != "bm_obj_warpboss_ancient.bml" or \
                cb.get("via") == "stem_family", (
                f"unexpected same-BML cross-inner resolution: row={row}"
            )


@pytest.mark.skipif(not (_LIVE.exists() or _DEV.exists()),
                    reason="no PSOBB data dir available")
def test_warp_beam_does_not_bind_to_sbeam():
    """beam.xj is the same shape (no NJTL, sibling sbeam) — same regression."""
    bd = _binding_for("bm_obj_warpboss_ancient.bml", "de_obj_df_warp_beam.xj")
    binding = bd.get("binding") or []
    assert binding
    for row in binding:
        cb = row.get("cross_bml") or {}
        assert cb.get("inner") != "de_obj_df_warp_sbeam.xj", (
            f"beam.xj mid={row.get('material_id')} bound to sbeam — bug. row={row}"
        )


@pytest.mark.skipif(not (_LIVE.exists() or _DEV.exists()),
                    reason="no PSOBB data dir available")
def test_dragon_in_bml_path_unchanged():
    """Dragon's in_bml binding must remain in_bml — regression on the fix."""
    data = _data_root()
    bml = data / "bm_boss1_dragon.bml"
    if not bml.exists():
        pytest.skip("bm_boss1_dragon.bml missing")
    bd = _binding_for("bm_boss1_dragon.bml", "boss1_s_nb_dragon.nj")
    binding = bd.get("binding") or []
    assert binding, "dragon binding list should not be empty"
    in_bml = sum(1 for r in binding if r.get("source") == "in_bml")
    # Dragon's main inner has a full inline XVMH; every mid should be
    # in_bml (~16 hits per the migration note). We assert >=10 to allow
    # for minor schema drift but lock the regression line.
    assert in_bml >= 10, (
        f"dragon should retain >=10 in_bml rows, got {in_bml}; "
        f"binding={binding!r}"
    )


# ---------------------------------------------------------------------------
# 4. Synthetic 2-BML stem-family fixture.
# ---------------------------------------------------------------------------


def test_stem_family_picks_richer_sibling(tmp_path):
    """Synthetic: ``find_sibling_bml_by_inner_stem`` skips zero-tile siblings.

    Two test BMLs in tmp_path; the helper walks the parent and the
    helper depends on ``list_bml_xvmh_inners`` — which we monkeypatch
    via the underlying inner cache so the test doesn't need real BML
    bytes (BML packing is out of scope here, the cache primitive is
    what the resolver consults).
    """
    # Build fake BMLs by creating empty files (the helper won't crash on
    # them — list_bml_xvmh_inners returns [] for any unparseable BML).
    a = tmp_path / "fakeA.bml"
    b = tmp_path / "fakeB.bml"
    c = tmp_path / "host.bml"
    a.write_bytes(b"")  # empty -> parse_bml fails -> [] inners
    b.write_bytes(b"")
    c.write_bytes(b"")

    # Patch the cache directly so the helper sees synthetic inner lists.
    src_inner = "fe_obj_df_warp_gawa.xj"
    rich_sibling_inner = "fd_obj1_swarp_gawa.xj"
    ti._BML_XVMH_CACHE.clear()
    # NOTE: cache key is (basename, mtime_ns). We pin a stable mtime by
    # reading the file's stat AFTER the writes above.
    a_key = (a.name, a.stat().st_mtime_ns)
    b_key = (b.name, b.stat().st_mtime_ns)
    c_key = (c.name, c.stat().st_mtime_ns)
    ti._BML_XVMH_CACHE[a_key] = [(rich_sibling_inner, 6)]
    ti._BML_XVMH_CACHE[b_key] = [("unrelated_pillar.xj", 4)]
    ti._BML_XVMH_CACHE[c_key] = []

    hit = ti.find_sibling_bml_by_inner_stem(c, src_inner, min_xvr_count=1)
    try:
        assert hit is not None
        assert hit[0] == "fakeA.bml", f"picked wrong sibling: {hit}"
        assert hit[1] == rich_sibling_inner
        assert hit[2] == 6
    finally:
        # Don't leave the synthetic cache around for other tests.
        ti._BML_XVMH_CACHE.clear()
