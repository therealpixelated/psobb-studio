"""Tests for the formats/rigging.py rig-editing primitives.

Coverage:
  - BonePose composition: identity, eval-flag overrides (UNIT_*/SKIP/ZXY)
  - World-matrix walk for a small chain
  - SubmeshWeights helpers: empty, from_bone_idx_array, normalize, add
  - Auto-skinning (distance + heat) on a synthetic 4-bone arm + cylinder
  - FABRIK IK convergence and reach-limit straight-line behaviour
  - 2-bone analytic IK preserves segment lengths
  - Wire-format encode/decode round-trip
  - Endpoint smoke test for /api/rig/save → /api/rig/<sha>
"""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import numpy as np
import pytest

from formats import rigging as rg


# ---------------------------------------------------------------------------
# Bone pose composition
# ---------------------------------------------------------------------------
def test_compose_identity():
    pose = rg.BonePose(index=0)
    m = rg.compose_local_matrix(pose)
    assert m == [1.0, 0.0, 0.0, 0.0,
                 0.0, 1.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 0.0,
                 0.0, 0.0, 0.0, 1.0]


def test_compose_translation():
    pose = rg.BonePose(index=0, position=(1.0, 2.0, 3.0))
    m = rg.compose_local_matrix(pose)
    assert m[3] == pytest.approx(1.0)
    assert m[7] == pytest.approx(2.0)
    assert m[11] == pytest.approx(3.0)


def test_compose_rotation_z_quarter_turn():
    # rotate 90° around Z: 0x4000 BAMS = 90°.
    pose = rg.BonePose(index=0, rotation_bams=(0, 0, 0x4000))
    m = rg.compose_local_matrix(pose)
    # Rotation should map (1, 0, 0) to (0, 1, 0).
    p_in = (1.0, 0.0, 0.0)
    p_out = rg.transform_point(m, p_in)
    assert p_out[0] == pytest.approx(0.0, abs=1e-5)
    assert p_out[1] == pytest.approx(1.0, abs=1e-5)
    assert p_out[2] == pytest.approx(0.0, abs=1e-5)


def test_compose_skip_yields_identity():
    pose = rg.BonePose(
        index=0, position=(1, 2, 3), rotation_bams=(0x1000, 0x2000, 0x3000),
        scale=(2, 3, 4), eval_flags=0x40,  # EVAL_SKIP
    )
    m = rg.compose_local_matrix(pose)
    assert m[0] == 1.0 and m[5] == 1.0 and m[10] == 1.0
    assert m[3] == 0.0 and m[7] == 0.0 and m[11] == 0.0


def test_compose_unit_pos_zeroes_translation():
    pose = rg.BonePose(
        index=0, position=(7, 8, 9),
        eval_flags=0x01,  # EVAL_UNIT_POS
    )
    m = rg.compose_local_matrix(pose)
    assert m[3] == 0.0 and m[7] == 0.0 and m[11] == 0.0


def test_world_matrices_propagate_through_chain():
    # 3-bone chain: each translates +1 along X locally.
    poses = [
        rg.BonePose(index=0, parent=-1, position=(1.0, 0.0, 0.0)),
        rg.BonePose(index=1, parent=0, position=(1.0, 0.0, 0.0)),
        rg.BonePose(index=2, parent=1, position=(1.0, 0.0, 0.0)),
    ]
    worlds = rg.compose_world_matrices(poses)
    assert worlds[0][3] == pytest.approx(1.0)
    assert worlds[1][3] == pytest.approx(2.0)
    assert worlds[2][3] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------
def test_empty_weights_layout():
    sw = rg.empty_weights(3, 5)
    assert sw.submesh_idx == 3
    assert sw.vertex_count == 5
    assert sw.bone_indices.shape == (5, rg.MAX_INFLUENCES)
    assert sw.weights.shape == (5, rg.MAX_INFLUENCES)
    assert (sw.bone_indices == -1).all()
    assert (sw.weights == 0).all()


def test_from_bone_idx_array():
    src = [0, 1, -1, 2]
    sw = rg.from_bone_idx_array(0, src)
    assert sw.vertex_count == 4
    assert sw.bone_indices[0, 0] == 0
    assert sw.weights[0, 0] == 1.0
    assert sw.bone_indices[2, 0] == -1
    assert sw.weights[2, 0] == 0.0
    # Slot 1+ is untouched (-1, 0).
    assert (sw.bone_indices[:, 1:] == -1).all()


def test_add_weight_existing_slot():
    sw = rg.empty_weights(0, 1)
    rg.add_weight(sw, 0, 5, 0.3)
    rg.add_weight(sw, 0, 5, 0.4)
    assert sw.bone_indices[0, 0] == 5
    assert sw.weights[0, 0] == pytest.approx(0.7)


def test_add_weight_new_slot_then_replace():
    sw = rg.empty_weights(0, 1)
    rg.add_weight(sw, 0, 1, 0.6)
    rg.add_weight(sw, 0, 2, 0.5)
    rg.add_weight(sw, 0, 3, 0.4)
    rg.add_weight(sw, 0, 4, 0.3)
    # 4 bones in 4 slots; adding a 5th replaces the smallest.
    rg.add_weight(sw, 0, 9, 0.9)
    assert 9 in sw.bone_indices[0].tolist()
    # The 0.3-bone (idx 4) should be the one that got replaced.
    assert 4 not in sw.bone_indices[0].tolist()


def test_normalize_weights_sums_to_one():
    sw = rg.empty_weights(0, 2)
    sw.bone_indices[0] = [0, 1, 2, -1]
    sw.weights[0] = [0.5, 0.25, 0.25, 0]
    sw.bone_indices[1] = [0, 1, 2, 3]
    sw.weights[1] = [2.0, 1.0, 0.5, 0.5]
    rg.normalize_weights(sw)
    assert sw.weights[0].sum() == pytest.approx(1.0)
    assert sw.weights[1].sum() == pytest.approx(1.0)


def test_normalize_weights_keeps_zero_rows_zero():
    sw = rg.empty_weights(0, 2)
    sw.bone_indices[0] = [0, 1, -1, -1]
    sw.weights[0] = [0.3, 0.7, 0, 0]
    # Row 1 stays all-zero.
    rg.normalize_weights(sw)
    assert sw.weights[0].sum() == pytest.approx(1.0)
    assert sw.weights[1].sum() == 0.0


# ---------------------------------------------------------------------------
# Auto-skin
# ---------------------------------------------------------------------------
def _make_arm_chain(n_bones: int = 4, spacing: float = 1.0):
    """Return a list of world matrices for n_bones along +X."""
    out = []
    for i in range(n_bones):
        out.append([
            1.0, 0.0, 0.0, i * spacing,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
    return out


def _make_cylinder_verts(n: int = 50, length: float = 3.0, radius: float = 0.3):
    """Verts arranged as a cylinder along +X with random angular jitter."""
    rng = np.random.default_rng(seed=42)
    pts = []
    for i in range(n):
        x = (i / max(1, n - 1)) * length
        # Ring around X.
        theta = 2 * np.pi * rng.random()
        y = radius * np.cos(theta)
        z = radius * np.sin(theta)
        pts.append([x, y, z])
    return np.array(pts, dtype=np.float64)


def test_autoskin_distance_assigns_nearest_bone():
    bones = _make_arm_chain(4, spacing=1.0)
    pts = _make_cylinder_verts(40, length=3.0, radius=0.1)
    sw = rg.autoskin_distance(pts, bones, falloff=4.0)
    # For each vert, the dominant bone (slot 0 — highest weight) should
    # be the nearest by index.
    for i in range(pts.shape[0]):
        nearest = int(np.argmin(
            np.array([abs(pts[i, 0] - b[3]) for b in bones])
        ))
        # Slot 0 carries highest weight after the descending sort.
        dom = int(sw.bone_indices[i, 0])
        # If two bones are roughly equidistant we accept neighbours.
        assert abs(dom - nearest) <= 1, (
            f"vert {i} at x={pts[i, 0]:.2f}: dom={dom}, nearest={nearest}"
        )


def test_autoskin_distance_normalized():
    bones = _make_arm_chain(4)
    pts = _make_cylinder_verts(20)
    sw = rg.autoskin_distance(pts, bones, falloff=2.0)
    sums = sw.weights.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-5)


def test_autoskin_heat_smoothes_weights():
    # A coarse cylinder; heat should produce smoother weights than
    # raw inverse-distance (less spiking near bone tips).
    bones = _make_arm_chain(4, spacing=1.0)
    pts = _make_cylinder_verts(30, length=3.0, radius=0.15)
    sw_d = rg.autoskin_distance(pts, bones, falloff=8.0)
    sw_h = rg.autoskin_heat(pts, bones, iterations=8)
    # Variance of slot-0 weight across verts: heat should be lower.
    # (Both must sum to 1 per row.)
    var_d = float(sw_d.weights[:, 0].std())
    var_h = float(sw_h.weights[:, 0].std())
    assert var_h <= var_d + 1e-3


def test_autoskin_dispatcher():
    bones = _make_arm_chain(2)
    pts = _make_cylinder_verts(10)
    a = rg.auto_skin(pts, bones, algorithm=rg.AUTOSKIN_DISTANCE)
    b = rg.auto_skin(pts, bones, algorithm=rg.AUTOSKIN_HEAT)
    assert a.vertex_count == 10
    assert b.vertex_count == 10
    with pytest.raises(ValueError, match="unknown algorithm"):
        rg.auto_skin(pts, bones, algorithm="nonsense")


def test_autoskin_handles_empty():
    sw = rg.auto_skin([], _make_arm_chain(2))
    assert sw.vertex_count == 0


# ---------------------------------------------------------------------------
# IK
# ---------------------------------------------------------------------------
def test_fabrik_reaches_target_in_range():
    # 3-joint chain along +X.
    chain = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    target = (1.0, 1.0, 0.0)
    out = rg.fabrik_solve(chain, target, iterations=30, tol=1e-4)
    end = out[-1]
    err = math.sqrt(
        (end[0] - target[0]) ** 2 + (end[1] - target[1]) ** 2 + (end[2] - target[2]) ** 2
    )
    assert err < 1e-3


def test_fabrik_preserves_segment_lengths_in_range():
    chain = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
    target = (1.5, 1.5, 0.5)
    out = rg.fabrik_solve(chain, target, iterations=30)
    for i in range(len(out) - 1):
        seg = math.sqrt(
            (out[i + 1][0] - out[i][0]) ** 2
            + (out[i + 1][1] - out[i][1]) ** 2
            + (out[i + 1][2] - out[i][2]) ** 2
        )
        assert seg == pytest.approx(1.0, abs=1e-2)


def test_fabrik_unreachable_straightens_chain():
    # Chain reach is 3.0; target 10 units away.
    chain = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
    target = (10.0, 0.0, 0.0)
    out = rg.fabrik_solve(chain, target, iterations=20)
    # Last joint should be at total reach (3.0) along +X.
    end = out[-1]
    assert end[0] == pytest.approx(3.0, abs=1e-2)
    assert end[1] == pytest.approx(0.0, abs=1e-2)


def test_fabrik_root_fixed():
    chain = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    target = (0.5, 0.5, 0)
    out = rg.fabrik_solve(chain, target, iterations=30)
    assert out[0] == (0.0, 0.0, 0.0)


def test_two_bone_ik_preserves_segment_lengths():
    root = (0.0, 0.0, 0.0)
    mid = (1.0, 0.0, 0.0)
    end = (2.0, 0.0, 0.0)
    target = (1.0, 1.0, 0.0)
    new_mid, new_end = rg.two_bone_ik(root, mid, end, target)
    L1 = math.sqrt(sum((new_mid[i] - root[i]) ** 2 for i in range(3)))
    L2 = math.sqrt(sum((new_end[i] - new_mid[i]) ** 2 for i in range(3)))
    assert L1 == pytest.approx(1.0, abs=1e-3)
    assert L2 == pytest.approx(1.0, abs=1e-3)


def test_two_bone_ik_end_at_target_when_reachable():
    root = (0.0, 0.0, 0.0)
    mid = (1.0, 0.0, 0.0)
    end = (2.0, 0.0, 0.0)
    target = (1.5, 1.0, 0.0)
    new_mid, new_end = rg.two_bone_ik(root, mid, end, target)
    err = math.sqrt(sum((new_end[i] - target[i]) ** 2 for i in range(3)))
    assert err < 1e-3


# ---------------------------------------------------------------------------
# Encode / decode round-trip
# ---------------------------------------------------------------------------
def test_encode_decode_roundtrip():
    bones = [
        rg.BonePose(index=0, parent=-1, position=(0, 0, 0), name="root"),
        rg.BonePose(index=1, parent=0, position=(1, 0, 0),
                    rotation_bams=(0x1000, 0, 0), name="mid"),
        rg.BonePose(index=2, parent=1, position=(1, 0, 0),
                    eval_flags=0x01, name="tip"),  # UNIT_POS
    ]
    bi = np.array([
        [0, 1, -1, -1],
        [1, 2, -1, -1],
        [2, -1, -1, -1],
    ], dtype=np.int32)
    w = np.array([
        [0.6, 0.4, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)
    weights = [rg.SubmeshWeights(
        submesh_idx=0, vertex_count=3, bone_indices=bi, weights=w,
    )]
    iks = [rg.IkTarget(
        bone_idx=2, chain_length=2, target=(0.5, 0.5, 0.0),
        iterations=10, name="hand_ik",
    )]
    payload = rg.encode_rig_payload(
        source_path="dragon.bml#dragon.nj",
        source_sha="cafebabe",
        bones=bones, weights=weights, ik_targets=iks,
        subdivide_level=2,
    )
    assert payload["format_version"] == rg.RIG_FORMAT_VERSION
    assert payload["source_path"] == "dragon.bml#dragon.nj"
    assert payload["source_sha"] == "cafebabe"
    assert payload["subdivide_level"] == 2
    assert len(payload["sha"]) == 16
    # Round-trip.
    db, dw, di = rg.decode_rig_payload(payload)
    assert len(db) == 3
    assert db[0].name == "root"
    assert db[2].eval_flags == 0x01
    assert db[1].rotation_bams == (0x1000, 0, 0)
    assert len(dw) == 1
    np.testing.assert_array_equal(dw[0].bone_indices, bi)
    np.testing.assert_allclose(dw[0].weights, w, atol=1e-7)
    assert len(di) == 1
    assert di[0].bone_idx == 2
    assert di[0].target == (0.5, 0.5, 0.0)


def test_encode_rejects_shape_mismatch():
    bones = [rg.BonePose(index=0)]
    bad_bi = np.zeros((3, 2), dtype=np.int32)  # wrong influence count
    bad_w = np.zeros((3, 2), dtype=np.float32)
    weights = [rg.SubmeshWeights(0, 3, bad_bi, bad_w)]
    with pytest.raises(ValueError, match="bone_indices shape"):
        rg.encode_rig_payload(
            source_path="x", source_sha="y",
            bones=bones, weights=weights,
        )


def test_decode_rejects_bad_format_version():
    bad = {"format_version": 99}
    with pytest.raises(ValueError, match="format_version"):
        rg.decode_rig_payload(bad)


def test_compute_source_sha_deterministic():
    a = rg.compute_source_sha(b"hello world")
    b = rg.compute_source_sha(b"hello world")
    c = rg.compute_source_sha(b"hello worle")
    assert a == b
    assert a != c
    assert len(a) == 16


# ---------------------------------------------------------------------------
# Endpoint smoke tests
# ---------------------------------------------------------------------------
def test_rig_endpoints_save_fetch_roundtrip(tmp_path, monkeypatch):
    """POST /api/rig/save then GET /api/rig/<sha> returns the same data."""
    # Lazy import so test collection works even when fastapi is missing.
    fastapi = pytest.importorskip("fastapi.testclient")
    import server  # noqa: WPS433
    from fastapi.testclient import TestClient

    # Redirect the rig cache dir into the test's tmp_path so we don't
    # litter the repo's cache/.
    monkeypatch.setattr(server, "RIG_CACHE_DIR", tmp_path)
    client = TestClient(server.app)

    payload = rg.encode_rig_payload(
        source_path="x.bml#y.nj",
        source_sha="deadbeefcafe",
        bones=[rg.BonePose(index=0)],
        weights=[rg.SubmeshWeights(
            submesh_idx=0, vertex_count=1,
            bone_indices=np.full((1, rg.MAX_INFLUENCES), -1, dtype=np.int32),
            weights=np.zeros((1, rg.MAX_INFLUENCES), dtype=np.float32),
        )],
        ik_targets=[],
    )
    r = client.post("/api/rig/save", json={
        "model_path": "x.bml#y.nj",
        "rig_payload": payload,
        "subdivide_level": 0,
    })
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["ok"] is True
    sha = j["sha"]
    # Fetch back.
    r2 = client.get(f"/api/rig/{sha}")
    assert r2.status_code == 200, r2.text
    fetched = r2.json()
    assert fetched["ok"] is True
    assert fetched["sha"] == sha
    fb_payload = fetched["rig_payload"]
    assert fb_payload["source_path"] == "x.bml#y.nj"
    assert len(fb_payload["skeleton"]["bones"]) == 1


def test_rig_fetch_404_for_unknown_sha(tmp_path, monkeypatch):
    pytest.importorskip("fastapi.testclient")
    import server
    from fastapi.testclient import TestClient

    monkeypatch.setattr(server, "RIG_CACHE_DIR", tmp_path)
    client = TestClient(server.app)
    r = client.get("/api/rig/0000000000000000")
    assert r.status_code == 404


def test_rig_save_rejects_bad_payload(tmp_path, monkeypatch):
    pytest.importorskip("fastapi.testclient")
    import server
    from fastapi.testclient import TestClient

    monkeypatch.setattr(server, "RIG_CACHE_DIR", tmp_path)
    client = TestClient(server.app)
    r = client.post("/api/rig/save", json={
        "model_path": "x.nj",
        "rig_payload": {"format_version": 99},
        "subdivide_level": 0,
    })
    assert r.status_code == 400


def test_rig_build_archive_lists_saved_rigs(tmp_path, monkeypatch):
    pytest.importorskip("fastapi.testclient")
    import server
    from fastapi.testclient import TestClient

    monkeypatch.setattr(server, "RIG_CACHE_DIR", tmp_path)
    client = TestClient(server.app)

    # Write a rig sidecar manually.
    payload = rg.encode_rig_payload(
        source_path="dragon.bml#dragon.nj",
        source_sha="abcd1234",
        bones=[rg.BonePose(index=0)],
        weights=[],
    )
    out = tmp_path / "dragon_bml__dragon__abcd1234.json"
    out.write_text(json.dumps(payload), encoding="utf-8")

    r = client.post("/api/rig/build_archive", json={"model_path": "dragon.bml"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["ok"] is True
    assert j["deployable"] is False  # NJ encoder pending
    assert any(s["model_path"] == "dragon.bml#dragon.nj" for s in j["rigs"])
