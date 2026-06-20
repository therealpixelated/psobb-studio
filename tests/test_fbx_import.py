"""Tests for ``formats.fbx_reader`` — binary FBX → ImportedModel.

Coverage:
  * parse_binary_fbx round-trip on hand-crafted byte streams (low-level
    parser correctness, including LayerElement subtree preservation).
  * parse_fbx on a static cube: vertex / triangle / normal / UV count
    matches expectations.
  * parse_fbx on a skinned humanoid: bone hierarchy in DFS order, skin
    weights normalize to sum-to-1 per vertex.
  * parse_fbx on a multi-mesh file: 2 ImportedMesh entries.
  * parse_fbx_with_animations: rotation curves convert to quaternions,
    times convert from FBX ktime → seconds.
  * imported_to_nj end-to-end: cube round-trips through the existing
    NJ encoder + parser without errors.
  * ASCII-FBX rejection (clear error pointing at the binary export
    option).
"""
from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from formats.fbx_reader import (
    FbxParseError,
    parse_binary_fbx,
    parse_fbx,
    parse_fbx_with_animations,
)
from formats.import_external import (
    imported_to_nj,
    parse_external,
)
from formats.nj_writer import encode_nj_model
from formats.xj import parse_nj_file

from tests._fbx_fixtures import (
    build_multi_mesh_fbx,
    build_skinned_humanoid_fbx,
    build_skinned_with_animation_fbx,
    build_static_cube_fbx,
)


# ---------------------------------------------------------------------------
# parse_binary_fbx — low-level parser
# ---------------------------------------------------------------------------


def test_parse_binary_fbx_static_cube_tree():
    """Verify the cube FBX parses into the expected node tree."""
    data = build_static_cube_fbx()
    root = parse_binary_fbx(data)
    top_names = [c.name for c in root.children]
    assert "FBXHeaderExtension" in top_names
    assert "GlobalSettings" in top_names
    assert "Objects" in top_names
    assert "Connections" in top_names

    objs = root.child("Objects")
    assert objs is not None
    geoms = objs.all_children("Geometry")
    assert len(geoms) == 1
    g = geoms[0]
    verts = g.child("Vertices")
    assert verts is not None and len(verts.props[0]) == 24  # 8 verts * 3
    poly = g.child("PolygonVertexIndex")
    assert poly is not None and len(poly.props[0]) == 24  # 6 quads * 4

    # CRITICAL: LayerElementNormal subtree must be preserved (the bug we
    # avoided in fbxloader). Verify it's a proper FbxRecord with children,
    # not a single int.
    le_n = g.child("LayerElementNormal")
    assert le_n is not None
    assert le_n.children, "LayerElementNormal subtree was lost (parser bug)"
    mit = le_n.child("MappingInformationType")
    assert mit is not None
    assert mit.props[0] == b"ByPolygonVertex"


def test_parse_binary_fbx_rejects_ascii_fbx():
    ascii_fbx = b"; FBX 7.4.0 project file\nFBXHeaderExtension:  {\n}\n"
    with pytest.raises(FbxParseError, match="ASCII"):
        parse_binary_fbx(ascii_fbx)


def test_parse_binary_fbx_rejects_garbage():
    with pytest.raises(FbxParseError, match="header"):
        parse_binary_fbx(b"NotAnFBXFile")


def test_parse_binary_fbx_rejects_short_data():
    with pytest.raises(FbxParseError):
        parse_binary_fbx(b"Kaydara FBX Binary  \x00\x1a\x00" + b"\x00" * 4)


# ---------------------------------------------------------------------------
# parse_fbx — static cube
# ---------------------------------------------------------------------------


def test_parse_fbx_static_cube_geometry():
    data = build_static_cube_fbx()
    m = parse_fbx(data)
    assert m.source_format == "fbx"
    assert len(m.meshes) == 1

    mesh = m.meshes[0]
    # 6 quads × 2 tris each = 12 triangles. With per-polygon-vertex
    # normals + UVs, the importer splits per output-triangle-corner so
    # attributes don't collide across faces sharing a source vertex —
    # 12 tris × 3 corners = 36 mesh verts. (More than the strict
    # per-polygon-vertex minimum of 24, but the renderer doesn't care
    # and the dedup keeps the post-flip writer simple.)
    assert mesh.indices.shape == (12, 3), f"got {mesh.indices.shape}"
    assert mesh.vertices.shape == (36, 3), f"got {mesh.vertices.shape}"
    # Normals + UVs should be parallel to the split vertex buffer.
    assert mesh.normals is not None
    assert mesh.normals.shape == (36, 3)
    assert mesh.uvs is not None
    assert mesh.uvs.shape == (36, 2)
    # All vertices live within [-1, 1].
    assert mesh.vertices.min() == pytest.approx(-1.0)
    assert mesh.vertices.max() == pytest.approx(1.0)
    # No bones in a static export.
    assert m.bones == []


def test_parse_fbx_static_cube_normals_unit_length():
    data = build_static_cube_fbx()
    m = parse_fbx(data)
    n = m.meshes[0].normals
    lens = np.linalg.norm(n, axis=1)
    # Each face normal in our fixture is already unit-length.
    assert np.allclose(lens, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# parse_fbx — skinned humanoid
# ---------------------------------------------------------------------------


def test_parse_fbx_skinned_extracts_bones_in_dfs_order():
    data = build_skinned_humanoid_fbx()
    m = parse_fbx(data)
    # 3 bones: root, hip, leg in DFS order.
    assert len(m.bones) == 3
    names = [b.name for b in m.bones]
    assert names == ["root", "hip", "leg"]
    # Hierarchy: root has parent_idx=-1, hip's parent=0, leg's parent=1.
    assert m.bones[0].parent_idx == -1
    assert m.bones[1].parent_idx == 0
    assert m.bones[2].parent_idx == 1


def test_parse_fbx_skinned_bones_have_bind_translation():
    data = build_skinned_humanoid_fbx()
    m = parse_fbx(data)
    # hip and leg both have Lcl Translation = (0, 1, 0).
    assert m.bones[1].bind_pos == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)
    assert m.bones[2].bind_pos == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)


def test_parse_fbx_skinned_weights_bind_to_leg():
    data = build_skinned_humanoid_fbx()
    m = parse_fbx(data)
    mesh = m.meshes[0]
    # All 4 vertices weighted entirely to the leg (bone idx 2).
    assert mesh.skin_weights is not None
    assert mesh.skin_indices is not None
    # Top weight should be 1.0 (we only have one influence per vertex).
    top = mesh.skin_weights.max(axis=1)
    np.testing.assert_allclose(top, 1.0, atol=1e-6)
    # The bone with that top weight should be 2 (leg).
    for v in range(mesh.skin_weights.shape[0]):
        j = int(np.argmax(mesh.skin_weights[v]))
        assert mesh.skin_indices[v, j] == 2


# ---------------------------------------------------------------------------
# parse_fbx — multi-mesh
# ---------------------------------------------------------------------------


def test_parse_fbx_multi_mesh_yields_two_imported_meshes():
    data = build_multi_mesh_fbx()
    m = parse_fbx(data)
    assert len(m.meshes) == 2
    names = sorted(mesh.name for mesh in m.meshes)
    assert names == ["Body", "Cloth"]


# ---------------------------------------------------------------------------
# parse_fbx_with_animations
# ---------------------------------------------------------------------------


def test_parse_fbx_animation_extracts_rotation_track():
    data = build_skinned_with_animation_fbx()
    m = parse_fbx_with_animations(data)
    assert len(m.animations) == 1
    anim = m.animations[0]
    # Track exists for the leg bone (idx 1), channel "rotation".
    rot_tracks = [t for t in anim.tracks if t.channel == "rotation"]
    assert len(rot_tracks) == 1
    track = rot_tracks[0]
    assert track.bone_idx == 1
    # 4 keyframes (the unified time grid we wrote on Y; X+Z share the
    # endpoints which fall inside Y's set so the union is still 4).
    assert len(track.times) == 4
    # First quaternion = identity (0° everywhere).
    qx, qy, qz, qw = track.values[0]
    assert (qx, qy, qz) == pytest.approx((0.0, 0.0, 0.0), abs=1e-5)
    assert qw == pytest.approx(1.0, abs=1e-5)
    # Last quaternion = 90° around Y.
    qx, qy, qz, qw = track.values[-1]
    assert qx == pytest.approx(0.0, abs=1e-5)
    assert qy == pytest.approx(math.sin(math.pi / 4.0), abs=1e-5)
    assert qz == pytest.approx(0.0, abs=1e-5)
    assert qw == pytest.approx(math.cos(math.pi / 4.0), abs=1e-5)


def test_parse_fbx_animation_times_in_seconds():
    data = build_skinned_with_animation_fbx()
    m = parse_fbx_with_animations(data)
    track = next(t for t in m.animations[0].tracks if t.channel == "rotation")
    # FBX KTIME = 46186158000 per second; we wrote keys at 0, 1/4, 1/2, 1 s.
    assert track.times[0] == pytest.approx(0.0, abs=1e-6)
    assert track.times[-1] == pytest.approx(1.0, abs=1e-6)
    # Animation duration should be 1.0 s.
    assert m.animations[0].duration_seconds == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Dispatch through parse_external
# ---------------------------------------------------------------------------


def test_parse_external_dispatches_fbx():
    data = build_static_cube_fbx()
    m = parse_external(data, "cube.fbx")
    assert m.source_format == "fbx"
    assert len(m.meshes) == 1


# ---------------------------------------------------------------------------
# imported_to_nj end-to-end
# ---------------------------------------------------------------------------


def test_imported_fbx_to_nj_round_trip():
    """FBX → ImportedModel → NjModel → bytes → parse_nj_file."""
    data = build_static_cube_fbx()
    m = parse_fbx(data)
    nj = imported_to_nj(m, axis_flip_z=True, scale=1.0)
    out = encode_nj_model(nj)
    parsed = parse_nj_file(out)
    # 24 split verts × 12 triangles → 12 single-triangle strips.
    assert len(parsed) == 12
    total_t = sum(len(s.indices) // 3 for s in parsed)
    assert total_t == 12


def test_imported_skinned_fbx_to_nj_round_trip():
    """Skinned cube → NJ → parse cleanly. Skin weights are present in
    the ImportedMesh but the v1 emitter doesn't drive skinning yet —
    we just verify the round-trip doesn't crash on a skinned source."""
    data = build_skinned_humanoid_fbx()
    m = parse_fbx(data)
    assert m.bones, "skinned source must produce a skeleton"
    nj = imported_to_nj(m, axis_flip_z=False, scale=1.0)
    out = encode_nj_model(nj)
    parsed = parse_nj_file(out)
    # 1 quad → 2 triangles → 2 strips.
    assert len(parsed) >= 1


# ---------------------------------------------------------------------------
# Polygon triangulation correctness
# ---------------------------------------------------------------------------


def test_triangulation_handles_quads_and_ngons():
    """Synthesize a 5-gon polygon and verify fan triangulation produces 3 tris."""
    from tests._fbx_fixtures import FbxNode, encode_fbx

    root = FbxNode("__root__")
    root.add(FbxNode("FBXHeaderExtension")).add(FbxNode("FBXVersion")).I(7400)
    objs = root.add(FbxNode("Objects"))
    g = objs.add(FbxNode("Geometry"))
    g.L(1).S("X\x00\x01Geometry").S("Mesh")
    g.add(FbxNode("Vertices")).Darr([
        0, 0, 0,  1, 0, 0,  2, 1, 0,  1, 2, 0,  0, 1, 0,
    ])
    # Single 5-vertex polygon (pentagon): 0, 1, 2, 3, 4 with end bit on last.
    g.add(FbxNode("PolygonVertexIndex")).Iarr([0, 1, 2, 3, -5])
    m = objs.add(FbxNode("Model"))
    m.L(2).S("X\x00\x01Model").S("Mesh")
    m.add(FbxNode("Version")).I(232)
    conns = root.add(FbxNode("Connections"))
    conns.add(FbxNode("C")).S("OO").L(1).L(2)
    conns.add(FbxNode("C")).S("OO").L(2).L(0)

    data = encode_fbx(root)
    parsed = parse_fbx(data)
    mesh = parsed.meshes[0]
    # A pentagon → 3 fan triangles.
    assert mesh.indices.shape == (3, 3)
