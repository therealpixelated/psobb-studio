"""Tests for the formats/sculpt.py geometry-sculpting primitives.

Coverage:
  - Falloff curve outputs at sentinel inputs
  - GridIndex radius queries
  - 1-ring neighbour map for a small mesh
  - Each brush operator (push, pull, inflate, smooth, pinch, flatten)
  - Sparse + dense displacement encode/decode round-trip
  - apply_displacement_to_payload baking
  - /api/sculpt/save + /api/sculpt/<sha> endpoint smoke test
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

import numpy as np

from formats import sculpt as sc


# ---------------------------------------------------------------------------
# Falloff
# ---------------------------------------------------------------------------
def test_falloff_endpoints():
    for curve in sc.VALID_FALLOFFS:
        assert sc.falloff(0.0, curve) == pytest.approx(1.0, abs=1e-3)
        assert sc.falloff(1.0, curve) == 0.0
        assert sc.falloff(1.5, curve) == 0.0
        # Non-negative everywhere.
        for t in (0.1, 0.25, 0.5, 0.75, 0.9):
            assert sc.falloff(t, curve) >= 0.0


def test_falloff_smooth_monotonic():
    # The smooth curve should be monotonically decreasing on [0, 1].
    prev = sc.falloff(0.0, sc.FALLOFF_SMOOTH)
    for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        cur = sc.falloff(t, sc.FALLOFF_SMOOTH)
        assert cur <= prev + 1e-9
        prev = cur


def test_falloff_unknown_returns_smooth():
    # Unknown curve falls back to smooth (no exception).
    assert sc.falloff(0.5, "bogus") == pytest.approx(
        sc.falloff(0.5, sc.FALLOFF_SMOOTH), abs=1e-9,
    )


# ---------------------------------------------------------------------------
# GridIndex
# ---------------------------------------------------------------------------
def test_grid_index_basic():
    pts = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [5.0, 5.0, 5.0],
    ], dtype=np.float64)
    idx = sc.GridIndex.build(pts, cell_size=0.5)
    near0 = sorted(idx.query_radius((0.0, 0.0, 0.0), 0.2))
    assert near0 == [0, 1]
    near1 = sorted(idx.query_radius((1.0, 0.0, 0.0), 0.05))
    assert near1 == [2]
    far = idx.query_radius((10.0, 10.0, 10.0), 0.1)
    assert far == []
    # Radius 5 from (5,5,5) gets only the (5,5,5) point.
    big = sorted(idx.query_radius((5.0, 5.0, 5.0), 0.001))
    assert big == [3]


def test_grid_index_handles_flat_input():
    flat = [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0]
    idx = sc.GridIndex.build(flat, cell_size=0.4)
    # Round-trip a query at the origin.
    near = sorted(idx.query_radius((0.0, 0.0, 0.0), 0.1))
    assert near == [0]


# ---------------------------------------------------------------------------
# 1-ring neighbours
# ---------------------------------------------------------------------------
def test_neighbours_single_triangle():
    # 3 verts, 1 triangle: every vert is adjacent to the other two.
    indices = [0, 1, 2]
    n = sc.build_vertex_neighbours(indices, 3)
    assert n[0] == [1, 2]
    assert n[1] == [0, 2]
    assert n[2] == [0, 1]


def test_neighbours_quad_two_tris():
    # Quad as two tris: 0-1-2 and 0-2-3. Vertex 0 sees {1, 2, 3}.
    indices = [0, 1, 2, 0, 2, 3]
    n = sc.build_vertex_neighbours(indices, 4)
    assert n[0] == [1, 2, 3]
    assert n[2] == [0, 1, 3]


def test_neighbours_rejects_bad_buffer():
    with pytest.raises(ValueError, match="not divisible by 3"):
        sc.build_vertex_neighbours([0, 1], 3)


# ---------------------------------------------------------------------------
# Brush primitives
# ---------------------------------------------------------------------------
def _make_grid_mesh(n=5, spacing=0.1):
    """Create an N×N planar grid centred at origin in the XY plane,
    triangulated into 2*(N-1)^2 tris. Returns (positions, normals, indices).
    """
    verts = []
    for j in range(n):
        for i in range(n):
            x = (i - (n - 1) / 2) * spacing
            y = (j - (n - 1) / 2) * spacing
            verts.extend([x, y, 0.0])
    normals = []
    for _ in range(n * n):
        normals.extend([0.0, 0.0, 1.0])
    indices = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b = j * n + (i + 1)
            c = (j + 1) * n + i
            d = (j + 1) * n + (i + 1)
            indices.extend([a, b, c, b, d, c])
    return (
        np.array(verts, dtype=np.float64),
        np.array(normals, dtype=np.float64),
        np.array(indices, dtype=np.int64),
    )


def test_brush_inflate_pushes_along_normal():
    pos, nrm, ind = _make_grid_mesh(5, 0.2)
    centre_idx = (5 // 2) * 5 + (5 // 2)
    centre = pos.reshape(-1, 3)[centre_idx]
    affected = sc.GridIndex.build(pos.reshape(-1, 3), 0.2).query_radius(centre, 0.5)
    new_pos, moved = sc.apply_brush(
        positions=pos, normals=nrm, indices=ind,
        affected_indices=affected,
        brush=sc.BRUSH_INFLATE,
        brush_centre=centre,
        brush_direction=(0, 0, 1),
        radius=0.5,
        strength=0.5,
        falloff_curve=sc.FALLOFF_SMOOTH,
    )
    new_pos = new_pos.reshape(-1, 3)
    # Centre vertex should have moved along +z by some amount.
    assert new_pos[centre_idx, 2] > 0.05
    # Non-affected verts (corners may be > radius) should be unchanged.
    corner_idx = 0
    if corner_idx not in moved:
        assert new_pos[corner_idx, 2] == pytest.approx(0.0, abs=1e-9)


def test_brush_push_along_direction():
    pos, nrm, ind = _make_grid_mesh(3, 0.5)
    centre = (0.0, 0.0, 0.0)
    affected = list(range(9))
    new_pos, moved = sc.apply_brush(
        positions=pos, normals=nrm, indices=ind,
        affected_indices=affected,
        brush=sc.BRUSH_PUSH,
        brush_centre=centre,
        brush_direction=(1, 0, 0),
        radius=2.0,
        strength=0.4,
        falloff_curve=sc.FALLOFF_LINEAR,
    )
    new_pos = new_pos.reshape(-1, 3)
    centre_vert = 4  # (1,1) of a 3x3 grid
    assert new_pos[centre_vert, 0] > 0.0
    # All moved verts should have a positive +x delta (push along +x).
    for vi in moved:
        assert new_pos[vi, 0] >= pos.reshape(-1, 3)[vi, 0]


def test_brush_pull_opposite_of_push():
    pos, nrm, ind = _make_grid_mesh(3, 0.5)
    centre = (0.0, 0.0, 0.0)
    affected = list(range(9))
    out_push, _ = sc.apply_brush(
        positions=pos, normals=nrm, indices=ind,
        affected_indices=affected,
        brush=sc.BRUSH_PUSH,
        brush_centre=centre,
        brush_direction=(1, 0, 0),
        radius=2.0, strength=0.4,
    )
    out_pull, _ = sc.apply_brush(
        positions=pos, normals=nrm, indices=ind,
        affected_indices=affected,
        brush=sc.BRUSH_PULL,
        brush_centre=centre,
        brush_direction=(1, 0, 0),
        radius=2.0, strength=0.4,
    )
    # pull deltas == -push deltas
    delta_push = out_push.reshape(-1, 3) - pos.reshape(-1, 3)
    delta_pull = out_pull.reshape(-1, 3) - pos.reshape(-1, 3)
    assert np.allclose(delta_pull, -delta_push, atol=1e-9)


def test_brush_pinch_pulls_toward_centre():
    pos, nrm, ind = _make_grid_mesh(5, 0.2)
    centre = np.array([0.0, 0.0, 0.0])
    pre = pos.reshape(-1, 3).copy()
    affected = sc.GridIndex.build(pre, 0.2).query_radius(centre, 1.0)
    out, moved = sc.apply_brush(
        positions=pos, normals=nrm, indices=ind,
        affected_indices=affected,
        brush=sc.BRUSH_PINCH,
        brush_centre=centre,
        brush_direction=(0, 0, 1),
        radius=1.0,
        strength=1.0,
        falloff_curve=sc.FALLOFF_LINEAR,
    )
    out = out.reshape(-1, 3)
    # Every moved vert is now closer to the centre than it was.
    for vi in moved:
        d_pre = float(np.linalg.norm(pre[vi] - centre))
        d_post = float(np.linalg.norm(out[vi] - centre))
        if d_pre > 1e-6:
            assert d_post <= d_pre + 1e-9


def test_brush_smooth_brings_spike_down():
    # Create a flat plane with one vertex spiked up.
    pos, nrm, ind = _make_grid_mesh(5, 0.2)
    pos = pos.reshape(-1, 3).copy()
    spike_idx = 12  # centre of 5x5
    pos[spike_idx, 2] = 1.0
    centre = pos[spike_idx]
    affected = list(range(pos.shape[0]))
    out, moved = sc.apply_brush(
        positions=pos.reshape(-1), normals=nrm, indices=ind,
        affected_indices=affected,
        brush=sc.BRUSH_SMOOTH,
        brush_centre=centre,
        brush_direction=(0, 0, 1),
        radius=2.0,
        strength=1.0,
        falloff_curve=sc.FALLOFF_LINEAR,
    )
    out = out.reshape(-1, 3)
    assert out[spike_idx, 2] < pos[spike_idx, 2]
    # Smoothed-down value approaches the neighbour mean (which is ~0
    # because all neighbours are on the plane).
    assert out[spike_idx, 2] < 0.5


def test_brush_flatten_collapses_to_plane():
    # Bumpy mesh: alternate z-offsets on a 5x5 grid.
    pos, nrm, ind = _make_grid_mesh(5, 0.2)
    pos = pos.reshape(-1, 3).copy()
    for i in range(pos.shape[0]):
        pos[i, 2] = ((i % 2) - 0.5) * 0.4   # ±0.2
    centre = np.array([0.0, 0.0, 0.0])
    affected = list(range(pos.shape[0]))
    out, moved = sc.apply_brush(
        positions=pos.reshape(-1), normals=nrm, indices=ind,
        affected_indices=affected,
        brush=sc.BRUSH_FLATTEN,
        brush_centre=centre,
        brush_direction=(0, 0, 1),
        radius=5.0,
        strength=1.0,
        falloff_curve=sc.FALLOFF_LINEAR,
    )
    out = out.reshape(-1, 3)
    # Z-spread should be smaller after flattening.
    pre_spread = float(np.std(pos[:, 2]))
    post_spread = float(np.std(out[:, 2]))
    assert post_spread < pre_spread


def test_brush_unknown_raises():
    pos, nrm, ind = _make_grid_mesh(3, 0.2)
    with pytest.raises(ValueError, match="unknown brush"):
        sc.apply_brush(
            positions=pos, normals=nrm, indices=ind,
            affected_indices=[0],
            brush="bogus",
            brush_centre=(0, 0, 0),
            brush_direction=(0, 0, 1),
            radius=0.5, strength=0.5,
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def test_encode_decode_roundtrip_dense():
    n = 100
    disp = np.random.rand(n, 3).astype(np.float32) * 0.1
    modified = np.arange(n, dtype=np.uint32)  # all of them => dense
    sub = sc.SubmeshSculpt(
        submesh_idx=0, material_id=0, vertex_count=n,
        displacement=disp, modified_indices=modified,
    )
    payload = sc.encode_sculpt_payload("foo.bml#bar.nj", "abcd1234", [sub])
    assert payload["format_version"] == sc.SCULPT_FORMAT_VERSION
    assert payload["submeshes"][0]["mode"] == "dense"
    decoded = sc.decode_sculpt_payload(payload)
    assert len(decoded) == 1
    np.testing.assert_allclose(decoded[0].displacement, disp, atol=1e-7)


def test_encode_decode_roundtrip_sparse():
    n = 100
    disp = np.zeros((n, 3), dtype=np.float32)
    modified = np.array([3, 7, 9, 22], dtype=np.uint32)
    for i in modified:
        disp[i] = [0.1, 0.2, 0.3]
    sub = sc.SubmeshSculpt(
        submesh_idx=2, material_id=4, vertex_count=n,
        displacement=disp, modified_indices=modified,
    )
    payload = sc.encode_sculpt_payload("x.bml#y.nj", "deadbeef", [sub])
    assert payload["submeshes"][0]["mode"] == "sparse"
    decoded = sc.decode_sculpt_payload(payload)
    np.testing.assert_allclose(decoded[0].displacement, disp, atol=1e-7)
    np.testing.assert_array_equal(decoded[0].modified_indices, modified)


def test_decode_rejects_bad_format_version():
    bad = {"format_version": 99, "submeshes": []}
    with pytest.raises(ValueError, match="format_version"):
        sc.decode_sculpt_payload(bad)


def test_apply_displacement_to_payload():
    # Build a tiny "/api/model_mesh"-shape payload with one submesh of 4 verts
    n = 4
    verts = np.zeros((n, 8), dtype=np.float32)
    verts[:, 0:3] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    verts[:, 5] = 1.0  # normal.z=1
    verts[:, 6] = 0.5  # uv
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    payload = {
        "mesh_count": 1,
        "meshes": [{
            "vertices_b64": base64.b64encode(verts.tobytes()).decode("ascii"),
            "indices_b64": base64.b64encode(indices.tobytes()).decode("ascii"),
            "vertex_count": n,
            "triangle_count": 2,
            "material_id": 0,
        }],
    }
    disp = np.zeros((n, 3), dtype=np.float32)
    disp[0] = [0.1, 0.0, 0.5]
    sub = sc.SubmeshSculpt(0, 0, n, disp, np.array([0], dtype=np.uint32))
    out = sc.apply_displacement_to_payload(payload, [sub])
    out_v = np.frombuffer(
        base64.b64decode(out["meshes"][0]["vertices_b64"]), dtype=np.float32,
    ).reshape(-1, 8)
    np.testing.assert_allclose(out_v[0, 0:3], [0.1, 0.0, 0.5], atol=1e-6)
    # Other verts unchanged.
    np.testing.assert_allclose(out_v[1, 0:3], [1, 0, 0], atol=1e-6)


# ---------------------------------------------------------------------------
# Endpoint smoke
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """In-process FastAPI client. The CACHE_DIR is left at its default
    (pointing at the editor's cache/) — test sculpts land in
    cache/sculpted_meshes/ but we clean up after every test that
    creates entries.
    """
    import server
    from fastapi.testclient import TestClient
    return TestClient(server.app)


def test_sculpt_save_and_fetch_roundtrip(client, tmp_path):
    # Build a synthetic sculpt payload.
    n = 8
    disp = np.zeros((n, 3), dtype=np.float32)
    disp[0] = [0.1, 0.0, 0.0]
    disp[3] = [0.0, 0.2, 0.0]
    sub = sc.SubmeshSculpt(0, 0, n, disp, np.array([0, 3], dtype=np.uint32))
    payload = sc.encode_sculpt_payload("synth_sculpt_test.bml#x.nj", "0011223344", [sub])

    body = {
        "model_path": "synth_sculpt_test.bml#x.nj",
        "mesh_payload": payload,
    }
    r = client.post("/api/sculpt/save", json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["ok"] is True
    sha = j["sha"]
    cp = Path(j["cache_path"])
    assert cp.exists()

    # Fetch back.
    r2 = client.get(f"/api/sculpt/{sha}")
    assert r2.status_code == 200, r2.text
    j2 = r2.json()
    assert j2["ok"] is True
    fetched = j2["mesh_payload"]
    # Round-tripped payload should decode the same displacements.
    rd = sc.decode_sculpt_payload(fetched)
    np.testing.assert_allclose(rd[0].displacement, disp, atol=1e-7)

    # Cleanup.
    cp.unlink(missing_ok=True)


def test_sculpt_save_rejects_bad_payload(client):
    # mesh_payload must be a JSON object — FastAPI validates the type
    # at the request-model level (422) before our handler's content-shape
    # check (400). Both rejections are acceptable.
    r = client.post("/api/sculpt/save", json={"model_path": "x.bml", "mesh_payload": "nope"})
    assert r.status_code in (400, 422)
    # Object that LOOKS like a payload but fails decoder validation -> 400.
    r2 = client.post("/api/sculpt/save", json={
        "model_path": "x.bml",
        "mesh_payload": {"format_version": 99, "submeshes": []},
    })
    assert r2.status_code == 400


def test_sculpt_fetch_404_unknown_sha(client):
    r = client.get("/api/sculpt/0000000000000000000000000000000000000000")
    assert r.status_code == 404
