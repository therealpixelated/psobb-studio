"""Tests for ``formats.import_external`` — external-format import path.

Coverage:
  - parse_obj on a hand-crafted cube
  - parse_gltf on a hand-crafted glTF (built in-memory via pygltflib)
  - imported_to_nj end-to-end: round-trip OBJ -> NJ -> parse_nj_file
  - imported_to_nj with skeleton template
  - imported_to_nj with source skeleton (glTF skin)
  - quat -> ZYX BAMS conversion math (unit-axis rotations)
  - skin weight quantization (sum-to-255 invariant)
  - imported_to_json + imported_from_json round trip
  - load_template / list_templates
"""
from __future__ import annotations

import base64
import json
import math
import struct

import numpy as np
import pytest

from formats.import_external import (
    ImportedBone,
    ImportedMesh,
    ImportedModel,
    imported_from_json,
    imported_to_json,
    imported_to_nj,
    list_templates,
    load_template,
    parse_external,
    parse_gltf,
    parse_obj,
    quantize_skin_weights,
    quat_to_zyx_bams,
    rad_to_bams,
)
from formats.nj_writer import encode_nj_model, parse_nj_for_writer
from formats.xj import parse_nj_file, parse_skeleton


# ---------------------------------------------------------------------------
# Tiny test fixtures
# ---------------------------------------------------------------------------


_OBJ_CUBE = b"""
# minimal axis-aligned cube
v -1 -1 -1
v  1 -1 -1
v  1  1 -1
v -1  1 -1
v -1 -1  1
v  1 -1  1
v  1  1  1
v -1  1  1
vn  0  0 -1
vn  0  0  1
vn  1  0  0
vn -1  0  0
vn  0  1  0
vn  0 -1  0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
o cube
f 1/1/1 2/2/1 3/3/1 4/4/1
f 5/1/2 6/2/2 7/3/2 8/4/2
f 2/1/3 6/2/3 7/3/3 3/4/3
f 1/1/4 5/2/4 8/3/4 4/4/4
f 4/1/5 3/2/5 7/3/5 8/4/5
f 1/1/6 2/2/6 6/3/6 5/4/6
"""


def _build_minimal_gltf_cube_bytes() -> bytes:
    """Build a tiny in-memory glTF JSON cube: 8 verts, 12 triangles."""
    import pygltflib

    verts = np.array([
        -1, -1, -1,  1, -1, -1,  1,  1, -1, -1,  1, -1,
        -1, -1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1,
    ], dtype=np.float32)
    indices = np.array([
        0, 1, 2,  0, 2, 3,  4, 6, 5,  4, 7, 6,
        0, 4, 5,  0, 5, 1,  2, 6, 7,  2, 7, 3,
        1, 5, 6,  1, 6, 2,  0, 3, 7,  0, 7, 4,
    ], dtype=np.uint32)
    buf_data = verts.tobytes() + indices.tobytes()
    buf_b64 = base64.b64encode(buf_data).decode("ascii")

    g = pygltflib.GLTF2(
        asset=pygltflib.Asset(version="2.0"),
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(mesh=0)],
        meshes=[pygltflib.Mesh(primitives=[pygltflib.Primitive(
            attributes=pygltflib.Attributes(POSITION=0),
            indices=1,
            mode=4,
        )])],
        buffers=[pygltflib.Buffer(
            uri=f"data:application/octet-stream;base64,{buf_b64}",
            byteLength=len(buf_data),
        )],
        bufferViews=[
            pygltflib.BufferView(
                buffer=0, byteOffset=0,
                byteLength=verts.nbytes, target=34962,
            ),
            pygltflib.BufferView(
                buffer=0, byteOffset=verts.nbytes,
                byteLength=indices.nbytes, target=34963,
            ),
        ],
        accessors=[
            pygltflib.Accessor(
                bufferView=0, componentType=5126,
                count=8, type="VEC3",
                max=[1.0, 1.0, 1.0], min=[-1.0, -1.0, -1.0],
            ),
            pygltflib.Accessor(
                bufferView=1, componentType=5125,
                count=36, type="SCALAR",
            ),
        ],
    )
    return g.to_json().encode("utf-8")


# ---------------------------------------------------------------------------
# OBJ parser
# ---------------------------------------------------------------------------


def test_obj_parse_cube_basic():
    m = parse_obj(_OBJ_CUBE)
    assert m.source_format == "obj"
    assert len(m.meshes) == 1
    mesh = m.meshes[0]
    assert mesh.name == "cube"
    # 6 faces × 4 dedup'd corners = 24 unique (pos, uv, normal) combos.
    assert mesh.vertices.shape == (24, 3)
    # 6 faces × 2 tris each = 12 triangles.
    assert mesh.indices.shape == (12, 3)
    # No bones from OBJ.
    assert m.bones == []


def test_obj_with_no_normals_warns():
    """OBJ without ``vn`` lines emits a warning + (0,1,0) synthetic normal."""
    obj = b"""
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
f 1/1 2/2 3/3
"""
    m = parse_obj(obj)
    assert any("no normals" in w for w in m.warnings)
    # The synthetic normal is (0, 1, 0).
    assert m.meshes[0].normals[0].tolist() == [0.0, 1.0, 0.0]


def test_obj_dispatches_through_parse_external():
    m = parse_external(_OBJ_CUBE, "cube.obj")
    assert m.source_format == "obj"
    assert len(m.meshes) == 1


def test_parse_external_rejects_garbage_fbx():
    """Bytes that look-like-but-aren't an FBX file → clear error."""
    with pytest.raises(ValueError, match="header"):
        parse_external(b"\x00\x00\x00\x00", "model.fbx")


def test_parse_external_rejects_ascii_fbx():
    """ASCII FBX (text format) → clear error pointing at binary export."""
    ascii_fbx = b"; FBX 7.4.0 project file\nFBXHeaderExtension:  {\n}\n"
    with pytest.raises(ValueError, match="ASCII"):
        parse_external(ascii_fbx, "model.fbx")


def test_parse_external_accepts_binary_fbx():
    """Binary FBX dispatches through parse_external → ImportedModel."""
    from tests._fbx_fixtures import build_static_cube_fbx
    data = build_static_cube_fbx()
    m = parse_external(data, "cube.fbx")
    assert m.source_format == "fbx"
    assert len(m.meshes) == 1


def test_parse_external_sniffs_fbx_via_magic():
    """Even without a .fbx extension, the FBX header magic should route to fbx_reader."""
    from tests._fbx_fixtures import build_static_cube_fbx
    data = build_static_cube_fbx()
    m = parse_external(data, "cube.bin")  # wrong extension but FBX magic
    assert m.source_format == "fbx"


def test_parse_external_unknown_extension():
    with pytest.raises(ValueError, match="Unrecognized"):
        parse_external(b"\x00\x00", "model.xyz")


# ---------------------------------------------------------------------------
# glTF parser
# ---------------------------------------------------------------------------


def test_gltf_parse_cube():
    blob = _build_minimal_gltf_cube_bytes()
    m = parse_gltf(blob, glb=False)
    assert m.source_format == "gltf"
    assert len(m.meshes) == 1
    mesh = m.meshes[0]
    assert mesh.vertices.shape == (8, 3)
    assert mesh.indices.shape == (12, 3)


def test_gltf_dispatches_through_parse_external():
    blob = _build_minimal_gltf_cube_bytes()
    m = parse_external(blob, "cube.gltf")
    assert m.source_format == "gltf"


# ---------------------------------------------------------------------------
# Quat -> BAMS conversion
# ---------------------------------------------------------------------------


def test_quat_identity_yields_zero_bams():
    rx, ry, rz = quat_to_zyx_bams(0.0, 0.0, 0.0, 1.0)
    assert (rx, ry, rz) == (0, 0, 0)


def test_quat_90_around_x():
    """A 90° rotation around X-axis: q = (sin(45°), 0, 0, cos(45°))."""
    s = math.sin(math.pi / 4.0)
    c = math.cos(math.pi / 4.0)
    rx, ry, rz = quat_to_zyx_bams(s, 0.0, 0.0, c)
    # 90° → 0x4000 BAMs.
    assert abs(rx - 0x4000) <= 2
    assert ry == 0 or abs(ry) <= 2
    assert rz == 0 or abs(rz) <= 2


def test_quat_90_around_y():
    s = math.sin(math.pi / 4.0)
    c = math.cos(math.pi / 4.0)
    rx, ry, rz = quat_to_zyx_bams(0.0, s, 0.0, c)
    assert abs(ry - 0x4000) <= 2
    assert rx == 0 or abs(rx) <= 2
    assert rz == 0 or abs(rz) <= 2


def test_quat_90_around_z():
    s = math.sin(math.pi / 4.0)
    c = math.cos(math.pi / 4.0)
    rx, ry, rz = quat_to_zyx_bams(0.0, 0.0, s, c)
    assert abs(rz - 0x4000) <= 2


def test_rad_to_bams_round_trip():
    # 360° = 0 (wrap).
    assert rad_to_bams(0.0) == 0
    # 90° = 0x4000.
    assert rad_to_bams(math.pi / 2) == 0x4000
    # -90° = 0xC000 (signed wrap).
    assert rad_to_bams(-math.pi / 2) == 0xC000
    # 180° = 0x8000.
    assert rad_to_bams(math.pi) == 0x8000


# ---------------------------------------------------------------------------
# Skin weight quantization
# ---------------------------------------------------------------------------


def test_skin_weights_quantize_sum_to_255():
    weights = np.array([
        [0.5, 0.3, 0.2, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25, 0.25],
        [0.7, 0.2, 0.05, 0.05],
    ], dtype=np.float32)
    indices = np.array([
        [0, 1, 2, 3],
        [4, 0, 0, 0],
        [10, 20, 30, 40],
        [5, 6, 7, 8],
    ], dtype=np.uint8)
    qw, qi = quantize_skin_weights(weights, indices)
    # Each row sums to 255.
    sums = qw.astype(np.int32).sum(axis=1)
    assert (sums == 255).all(), f"sums={sums.tolist()}"


def test_skin_weights_zero_input_binds_to_bone_0():
    weights = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    indices = np.array([[10, 20, 30, 40]], dtype=np.uint8)
    qw, qi = quantize_skin_weights(weights, indices)
    assert qw[0, 0] == 255
    assert qi[0, 0] == 0
    assert qw[0, 1:].sum() == 0


# ---------------------------------------------------------------------------
# imported_to_nj end-to-end
# ---------------------------------------------------------------------------


def test_imported_obj_to_nj_round_trip():
    """OBJ -> ImportedModel -> NjModel -> bytes -> parse_nj_file works."""
    m = parse_obj(_OBJ_CUBE)
    nj = imported_to_nj(m, axis_flip_z=True, scale=1.0)
    out = encode_nj_model(nj)
    parsed = parse_nj_file(out)
    # Each strip is length 3 (one triangle per strip), so we get one
    # submesh per triangle = 12 submeshes.
    assert len(parsed) == 12
    total_t = sum(len(s.indices) // 3 for s in parsed)
    assert total_t == 12  # original triangle count preserved.


def test_imported_obj_with_template_uses_template_skeleton():
    """When the OBJ has no bones, target_class loads the template."""
    m = parse_obj(_OBJ_CUBE)
    assert len(m.bones) == 0
    nj = imported_to_nj(m, target_class="monster_humanoid", axis_flip_z=False)
    out = encode_nj_model(nj)
    bones = parse_skeleton(out)
    # monster_humanoid template = 51 bones (extracted from PSOBB Booma).
    assert len(bones) == 51


def test_imported_obj_no_template_uses_one_bone_root():
    m = parse_obj(_OBJ_CUBE)
    nj = imported_to_nj(m, target_class=None, axis_flip_z=False)
    # Without a template the converter still adds a root bone + 1
    # synthetic mesh-only node per extra mesh; for the cube (1 mesh)
    # there's no synthetic node, just the root.
    assert len(nj.nodes) == 1
    out = encode_nj_model(nj)
    bones = parse_skeleton(out)
    assert len(bones) == 1


def test_imported_with_source_bones_preserves_count():
    """When the source has bones, the converter uses them verbatim."""
    src_bones = [
        ImportedBone(
            name="root", parent_idx=-1,
            bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1),
        ),
        ImportedBone(
            name="hip", parent_idx=0,
            bind_pos=(0, 1, 0), bind_rot_quat=(0, 0, 0, 1),
        ),
        ImportedBone(
            name="leg", parent_idx=1,
            bind_pos=(0, -1, 0), bind_rot_quat=(0, 0, 0, 1),
        ),
    ]
    obj_model = parse_obj(_OBJ_CUBE)
    obj_model.bones = src_bones
    nj = imported_to_nj(obj_model, target_class=None, axis_flip_z=False)
    out = encode_nj_model(nj)
    bones = parse_skeleton(out)
    assert len(bones) == 3


def test_imported_axis_flip_z():
    """axis_flip_z=True negates Z on positions + bone bind positions."""
    src = ImportedModel(
        meshes=[ImportedMesh(
            name="x",
            vertices=np.array([[0.0, 0.0, 5.0]], dtype=np.float32),
            indices=np.zeros((0, 3), dtype=np.uint32),
        )],
    )
    nj = imported_to_nj(src, axis_flip_z=True, scale=1.0)
    # Encode + parse back to inspect baked vertex.
    out = encode_nj_model(nj)
    # No triangles → parse_nj_file returns empty; instead we verify the
    # vlist chunk's first vertex z is negated.
    parsed = parse_nj_for_writer(out)
    assert len(parsed.meshes) == 1
    vbody = parsed.meshes[0].vlist[0].body
    # body: u16 body_words, u16 base_idx, u16 count, then 12+12 per vert.
    px, py, pz = struct.unpack_from("<3f", vbody, 6)
    assert pz == pytest.approx(-5.0, abs=1e-5)


def test_imported_scale_applied():
    src = ImportedModel(
        meshes=[ImportedMesh(
            name="x",
            vertices=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            indices=np.zeros((0, 3), dtype=np.uint32),
        )],
    )
    nj = imported_to_nj(src, axis_flip_z=False, scale=100.0)
    out = encode_nj_model(nj)
    parsed = parse_nj_for_writer(out)
    vbody = parsed.meshes[0].vlist[0].body
    px, py, pz = struct.unpack_from("<3f", vbody, 6)
    assert pz == pytest.approx(100.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def test_list_templates_includes_player_body():
    names = list_templates()
    assert "player_body" in names
    assert "player_head" in names
    assert "monster_humanoid" in names


def test_load_template_player_body_has_64_bones():
    bones = load_template("player_body")
    # PSOBB plAbdy00.nj has 64 bones.
    assert len(bones) == 64
    # Root bone has parent -1.
    assert bones[0].parent_idx == -1


def test_load_template_unknown_raises():
    with pytest.raises(FileNotFoundError):
        load_template("does_not_exist")


# ---------------------------------------------------------------------------
# JSON wire-shape round trip
# ---------------------------------------------------------------------------


def test_imported_to_json_round_trip():
    m = parse_obj(_OBJ_CUBE)
    j = imported_to_json(m)
    assert j["mesh_count"] == 1
    assert j["meshes"][0]["vertex_count"] == 24
    assert j["meshes"][0]["triangle_count"] == 12

    # Round-trip through imported_from_json -> imported_to_nj.
    m2 = imported_from_json(j)
    assert len(m2.meshes) == 1
    np.testing.assert_array_almost_equal(
        m2.meshes[0].vertices, m.meshes[0].vertices, decimal=5,
    )
    nj2 = imported_to_nj(m2, axis_flip_z=False)
    out = encode_nj_model(nj2)
    parsed = parse_nj_file(out)
    assert len(parsed) > 0


# ---------------------------------------------------------------------------
# Coordinate convention smoke test (glTF source → PSOBB-flipped NJ)
# ---------------------------------------------------------------------------


def test_gltf_to_nj_flip_z_and_scale():
    blob = _build_minimal_gltf_cube_bytes()
    m = parse_gltf(blob, glb=False)
    nj = imported_to_nj(m, axis_flip_z=True, scale=10.0)
    out = encode_nj_model(nj)
    # Verify it parses cleanly back.
    parsed = parse_nj_file(out)
    assert len(parsed) > 0
    # Verify scaled bounds: each vertex should be 10x further from origin.
    for sub in parsed:
        for v in sub.vertices:
            r = math.sqrt(sum(c * c for c in v.pos))
            # Original cube had |v|=sqrt(3) ≈ 1.73; post-scale ≈ 17.3.
            assert 16.0 < r < 18.0
