"""Descriptor-XJ additive-blend tests (the bm_eff_ice "dark crystal" fix).

The descriptor parser used to decode the type-2 ``(src_alpha, dst_alpha)``
material entry only to advance the cursor and then discard it, so every
effect model (ice / fire / energy) rendered as a dark opaque solid. The
parser now maps that factor pair to ``XjMesh.blend_mode`` so the frontend
can apply ``THREE.AdditiveBlending`` (glow), guarded so opaque geometry
(no blend entry, or the normal-alpha (4,5) pair) keeps blend_mode="none".

Two halves:
  PART A  pure unit — ``_blend_factors_to_mode`` maps factor pairs to the
          right (mode, alpha_blend) with NO game data required.
  PART B  real-asset — bm_eff_ice's ice_break_root.xj parses to additive
          meshes, while an opaque object .xj stays "none"
          (PSOBB-data-guarded).

Run isolated::

    python -m pytest tests/test_xj_effect_blend.py -q
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pytest

from formats import xj_descriptor as xd


# ---------------------------------------------------------------------------
# PART A — pure factor-pair → mode mapping (no data)
# ---------------------------------------------------------------------------

def test_additive_pair_src_alpha_one():
    # (src_alpha=4, one=1) is the canonical additive FX pair (ice/fire).
    mode, ab = xd._blend_factors_to_mode(4, 1)
    assert mode == "additive"
    assert ab == {"src": "src_alpha", "dst": "one"}


def test_normal_alpha_pair_maps_to_none():
    # (src_alpha=4, one_minus_src_alpha=5) is normal alpha — over PSOBB's
    # opaque (alpha=255) textures it is pixel-identical to opaque, so we
    # leave it untouched (blend_mode "none") to avoid the depthWrite sort
    # risk on solid object/door geometry.
    mode, ab = xd._blend_factors_to_mode(4, 5)
    assert mode == "none"
    assert ab is None


def test_one_one_pair_is_additive():
    # (one=1, one=1) — pure additive accumulation; phantasmal's rule
    # "additive unless exactly (4,5)" makes this additive too.
    mode, ab = xd._blend_factors_to_mode(1, 1)
    assert mode == "additive"
    assert ab == {"src": "one", "dst": "one"}


def test_unknown_factor_index_is_tolerated():
    # Out-of-range factor index must not crash; it just falls back to a
    # sane name and (since != (4,5)) is treated as additive.
    mode, ab = xd._blend_factors_to_mode(99, 99)
    assert mode == "additive"
    assert ab is not None
    assert "src" in ab and "dst" in ab


# ---------------------------------------------------------------------------
# PART B — real-asset parse (PSOBB-data-guarded)
# ---------------------------------------------------------------------------

_DATA_DIRS = [
    Path(os.path.expanduser("~/PSOBB.IO/data")),
    Path(os.path.expanduser("~/EphineaPSO/data")),
]


def _data_dir():
    env = os.environ.get("PSO_DATA_DIR") or os.environ.get("PSO_XJ_TEST_DATA_DIR")
    if env and Path(env).is_dir():
        return Path(env)
    for d in _DATA_DIRS:
        if d.is_dir():
            return d
    return None


HAS_DATA = _data_dir() is not None


def _read_inner_xj(bml_path: Path, inner: str) -> bytes:
    # Use the same BML-inner extractor the server uses; import lazily so a
    # missing optional dep doesn't fail collection.
    import server  # noqa: WPS433 — test-time import of the app module

    return server._read_inner_nj_from_bml(bml_path, inner)


@pytest.mark.skipif(not HAS_DATA, reason="no PSOBB/Ephinea game data on disk")
def test_bm_eff_ice_meshes_are_additive():
    """The ice-break effect must parse to additive-blended submeshes."""
    data = _data_dir()
    bml = data / "bm_eff_ice.bml"
    if not bml.exists():
        pytest.skip("bm_eff_ice.bml not present in this data set")
    meshes = xd.parse_xj_file(_read_inner_xj(bml, "ice_break_root.xj"))
    assert meshes, "ice_break_root.xj produced no meshes"
    modes = Counter(getattr(m, "blend_mode", "none") for m in meshes)
    # The shipped ice effect carries (src_alpha, one) on its first strip
    # of every node; sticky inheritance makes every mesh additive.
    assert modes.get("additive", 0) == len(meshes), modes
    assert modes.get("none", 0) == 0, modes
    # And the raw factor pair is surfaced for the Material panel.
    add = next(m for m in meshes if getattr(m, "blend_mode", "") == "additive")
    assert add.alpha_blend == {"src": "src_alpha", "dst": "one"}


@pytest.mark.skipif(not HAS_DATA, reason="no PSOBB/Ephinea game data on disk")
def test_opaque_object_xj_stays_none():
    """A fully-opaque object .xj (normal-alpha only) must NOT become blended
    — the guardrail against regressing solid geometry."""
    data = _data_dir()
    # bm_fe_obj_aircon02 carries only (4,5) normal-alpha entries -> all none.
    bml = data / "bm_fe_obj_aircon02.bml"
    if not bml.exists():
        pytest.skip("bm_fe_obj_aircon02.bml not present in this data set")
    meshes = xd.parse_xj_file(_read_inner_xj(bml, "fe_obj_aircon02.xj"))
    assert meshes, "fe_obj_aircon02.xj produced no meshes"
    modes = Counter(getattr(m, "blend_mode", "none") for m in meshes)
    assert modes.get("additive", 0) == 0, modes
    assert modes.get("none", 0) == len(meshes), modes
