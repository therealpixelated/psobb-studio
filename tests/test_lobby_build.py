"""Acceptance test for the offline lobby build pipeline (scripts/build_lobby.py).

The synthetic leg runs on a generated grid mesh so CI needs no game assets;
the real-GLB leg runs the full CLI against the Casinopolis model only when it
is present on disk (skipped otherwise).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import build_lobby as bl  # noqa: E402
from formats import rel_writer as _rw  # noqa: E402


def _grid_mesh(n: int):
    """An n×n ground grid -> 2*(n-1)^2 triangles, with planar UVs."""
    xs = np.linspace(-200.0, 200.0, n)
    V = np.array([[x, 0.0, z] for z in xs for x in xs], dtype=np.float64)
    U = np.array([[(x + 200) / 400, (z + 200) / 400] for z in xs for x in xs], dtype=np.float64)
    F = []
    for r in range(n - 1):
        for c in range(n - 1):
            a = r * n + c
            F.append([a, a + 1, a + n])
            F.append([a + 1, a + n + 1, a + n])
    return V, U, np.array(F, dtype=np.int64)


def test_synthetic_lobby_build_fits_and_reparses():
    # ~7k tris -> exceeds the n.rel cap at ~170 B/tri, so decimate_to_fit must engage.
    V, U, F = _grid_mesh(60)
    assert F.shape[0] > 6000

    V2, U2, F2, nrel = bl.decimate_to_fit(V, U, F, _rw.NREL_SIZE_BUDGET, "lobby")
    assert len(nrel) <= _rw.NREL_SIZE_BUDGET
    ok, msg = bl.verify_nrel(nrel)
    assert ok, msg

    crel = bl.author_crel(V2, F2, _rw.CREL_SIZE_BUDGET)
    assert crel is not None
    assert len(crel) <= _rw.CREL_SIZE_BUDGET
    ok, msg = bl.verify_crel(crel)
    assert ok, msg


def test_normals_are_unit_length():
    V, U, F = _grid_mesh(8)
    n = bl._vertex_normals(V, F)
    lens = np.linalg.norm(n, axis=1)
    assert np.allclose(lens, 1.0, atol=1e-6)


def test_decimate_to_fit_is_noop_when_small():
    # A small mesh already under budget must NOT be decimated.
    V, U, F = _grid_mesh(12)  # 2*121 = 242 tris, tiny
    V2, U2, F2, nrel = bl.decimate_to_fit(V, U, F, _rw.NREL_SIZE_BUDGET, "lobby")
    assert F2.shape[0] == F.shape[0]
    assert len(nrel) <= _rw.NREL_SIZE_BUDGET


def _find_casinopolis() -> Path | None:
    root = os.environ.get("PSOBB_DOWNLOADS_DIR") or os.path.expanduser("~/Downloads")
    cands = sorted(Path(root).glob("*.glb"), key=lambda p: p.stat().st_size, reverse=True)
    return cands[0] if cands else None


@pytest.mark.skipif(_find_casinopolis() is None, reason="no source GLB on disk (CI)")
def test_real_glb_build_passes(tmp_path):
    glb = _find_casinopolis()
    rc = bl.main(["--glb", str(glb), "--out", str(tmp_path), "--name", "map_test_01"])
    assert rc == 0
    nrel = tmp_path / "map_test_01n.rel"
    assert nrel.is_file()
    ok, msg = bl.verify_nrel(nrel.read_bytes())
    assert ok, msg
