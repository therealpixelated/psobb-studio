"""Tests for formats/decimate.py — the real QEM mesh decimator.

Coverage:
  - decimate_mesh(target_tris=N) yields <= N*1.1 faces and > 0
  - decimate_mesh(target_ratio=0.5) roughly halves the triangle count
  - shape preserved: no NaNs, bbox within ~5% of original
  - decimate_to_byte_budget converges under a synthetic budget, and flags
    over_budget when the floor can't fit
  - the hand-rolled NumPy QEM fallback runs (and is exercised explicitly,
    so it's covered even when the fast backend is installed)
  - GLB legs (Cesium Man / Kenney crate) gated on the asset existing

Run ISOLATED (the full suite is timing-flaky under load):
    python -m pytest tests/test_decimate.py -q
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from formats import decimate as D


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "data" / "test_assets"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _sphere(subdivisions=4):
    """A high-poly icosphere (subdivisions=4 -> 5120 tris)."""
    m = trimesh.creation.icosphere(subdivisions=subdivisions)
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def _bbox(v):
    v = np.asarray(v).reshape(-1, 3)
    return v.min(axis=0), v.max(axis=0)


def _bbox_close(a_v, b_v, tol=0.05):
    a0, a1 = _bbox(a_v)
    b0, b1 = _bbox(b_v)
    span = np.maximum(a1 - a0, 1e-9)
    return bool(np.all(np.abs(b0 - a0) / span <= tol) and
                np.all(np.abs(b1 - a1) / span <= tol))


# --------------------------------------------------------------------------- #
# Core decimate_mesh behaviour
# --------------------------------------------------------------------------- #
def test_target_tris_respected():
    v, f = _sphere(4)
    assert len(f) == 5120
    for n in (2000, 1000, 500, 200):
        ov, of = D.decimate_mesh(v, f, target_tris=n)
        assert 0 < len(of) <= n * 1.1, f"target {n}: got {len(of)}"
        assert not np.isnan(ov).any()
        assert of.min() >= 0 and of.max() < len(ov)


def test_ratio_roughly_halves():
    v, f = _sphere(4)
    ov, of = D.decimate_mesh(v, f, target_ratio=0.5)
    ratio = len(of) / len(f)
    assert 0.4 <= ratio <= 0.6, f"ratio 0.5 gave {ratio:.3f}"


def test_ratio_quarter():
    v, f = _sphere(4)
    ov, of = D.decimate_mesh(v, f, target_ratio=0.25)
    ratio = len(of) / len(f)
    assert 0.18 <= ratio <= 0.32, f"ratio 0.25 gave {ratio:.3f}"


def test_shape_preserved_no_nan_bbox():
    v, f = _sphere(4)
    ov, of = D.decimate_mesh(v, f, target_tris=800)
    assert not np.isnan(ov).any()
    assert not np.isinf(ov).any()
    assert _bbox_close(v, ov, tol=0.05), "decimated bbox drifted >5%"
    # No degenerate faces survive the clean pass.
    assert np.all(of[:, 0] != of[:, 1])
    assert np.all(of[:, 1] != of[:, 2])
    assert np.all(of[:, 0] != of[:, 2])
    # No unreferenced vertices left dangling.
    assert set(np.unique(of)) == set(range(len(ov)))


def test_watertight_ish():
    # A watertight input should stay watertight-ish after QEM (every edge
    # shared by exactly 2 faces). We allow a tiny boundary slack.
    v, f = _sphere(4)
    ov, of = D.decimate_mesh(v, f, target_tris=600, preserve_border=True)
    tm = trimesh.Trimesh(vertices=ov, faces=of, process=False)
    # Either trimesh reports watertight, or the boundary-edge fraction is tiny.
    if not tm.is_watertight:
        edges = of[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        edges = np.sort(edges, axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        boundary_frac = (counts == 1).sum() / max(1, len(counts))
        assert boundary_frac < 0.05, f"too many boundary edges: {boundary_frac:.3f}"


def test_meta_reports_backend():
    v, f = _sphere(4)
    ov, of, uv, meta = D.decimate_mesh(v, f, target_tris=500, return_meta=True)
    assert meta["backend"] in (
        "trimesh_fast_simplification", "numpy_qem_fallback")
    assert meta["in_tris"] == 5120
    assert meta["out_tris"] == len(of)


def test_uv_resample_shape():
    v, f = _sphere(3)
    uvs = np.random.RandomState(0).rand(len(v), 2)
    ov, of, out_uv, meta = D.decimate_mesh(
        v, f, target_tris=400, uvs=uvs, return_meta=True)
    assert out_uv is not None
    assert out_uv.shape == (len(ov), 2)
    assert not np.isnan(out_uv).any()


def test_noop_when_target_exceeds():
    v, f = _sphere(2)  # 320 tris
    ov, of, uv, meta = D.decimate_mesh(v, f, target_tris=99999, return_meta=True)
    assert meta["backend"] == "noop"
    assert len(of) == len(f)


def test_requires_a_target():
    v, f = _sphere(2)
    with pytest.raises(ValueError):
        D.decimate_mesh(v, f)


def test_bad_ratio_rejected():
    v, f = _sphere(2)
    with pytest.raises(ValueError):
        D.decimate_mesh(v, f, target_ratio=1.5)
    with pytest.raises(ValueError):
        D.decimate_mesh(v, f, target_ratio=0.0)


# --------------------------------------------------------------------------- #
# Hand-rolled NumPy QEM fallback — exercised explicitly so it's always
# covered, even when fast-simplification is installed.
# --------------------------------------------------------------------------- #
def test_numpy_qem_fallback_runs():
    v, f = _sphere(3)  # 1280 tris
    ov, of = D._qem_fallback(v, f, 300, preserve_border=True)
    assert 0 < len(of) <= 300 * 1.1
    assert not np.isnan(ov).any()
    assert _bbox_close(v, ov, tol=0.06)


def test_forced_fallback_path(monkeypatch):
    # Force the dispatcher down the fallback by making the trimesh QEM
    # attempt report "unavailable".
    monkeypatch.setattr(D, "_try_trimesh_qem", lambda *a, **k: None)
    v, f = _sphere(3)
    ov, of, uv, meta = D.decimate_mesh(v, f, target_tris=350, return_meta=True)
    assert meta["backend"] == "numpy_qem_fallback"
    assert 0 < len(of) <= 350 * 1.1
    assert not np.isnan(ov).any()


# --------------------------------------------------------------------------- #
# Byte-budget binary search
# --------------------------------------------------------------------------- #
def test_byte_budget_converges():
    v, f = _sphere(5)  # ~20480 tris
    budget = 50_000
    ov, of, meta = D.decimate_to_byte_budget(
        v, f, encode_size_fn=D.estimate_rel_node_bytes, budget_bytes=budget)
    assert meta["over_budget"] is False
    assert meta["encoded_bytes"] <= budget
    # It actually used most of the budget (didn't stop way short).
    assert meta["encoded_bytes"] >= budget * 0.7
    assert D.estimate_rel_node_bytes(ov, of) <= budget
    assert meta["final_tris"] == len(of)


def test_byte_budget_input_already_fits():
    v, f = _sphere(2)  # 320 tris, tiny
    big_budget = 10_000_000
    ov, of, meta = D.decimate_to_byte_budget(
        v, f, encode_size_fn=D.estimate_rel_node_bytes, budget_bytes=big_budget)
    assert meta["over_budget"] is False
    assert meta["iters"] == 0
    assert len(of) == len(f)


def test_byte_budget_over_budget_flag():
    v, f = _sphere(4)
    # A budget so small that even the floor (200 tris) can't fit.
    floor_bytes = D.estimate_rel_node_bytes(
        *D.decimate_mesh(v, f, target_tris=D.DEFAULT_BUDGET_FLOOR_TRIS))
    tiny_budget = floor_bytes - 1000
    ov, of, meta = D.decimate_to_byte_budget(
        v, f, encode_size_fn=D.estimate_rel_node_bytes,
        budget_bytes=tiny_budget)
    assert meta["over_budget"] is True
    assert meta["overage_bytes"] > 0
    assert meta["final_tris"] > 0  # floor result still returned


# --------------------------------------------------------------------------- #
# Real GLB legs — gated on the asset existing.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["khronos_cesium_man.glb", "kenney_scifi_crate.glb"])
def test_decimate_real_glb(name):
    path = ASSETS / name
    if not path.exists():
        pytest.skip(f"asset not present: {path}")
    m = trimesh.load(str(path), process=False, force="mesh")
    v = np.asarray(m.vertices, dtype=np.float64)
    f = np.asarray(m.faces, dtype=np.int64)
    if len(f) < 200:
        pytest.skip(f"{name} too low-poly to meaningfully decimate ({len(f)} tris)")
    target = max(50, len(f) // 4)
    ov, of, uv, meta = D.decimate_mesh(v, f, target_tris=target, return_meta=True)
    assert 0 < len(of) <= target * 1.1
    assert not np.isnan(ov).any()
    assert _bbox_close(v, ov, tol=0.08), f"{name}: bbox drifted"
    assert meta["out_tris"] == len(of)
