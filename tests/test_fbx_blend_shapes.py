"""Tests for FBX BlendShape (morph target) extraction.

PSOBB has no morph-target rendering, so the import pipeline IGNORES
blend shapes when projecting to NjModel. The parser still recovers the
data and stashes it on ``ImportedModel.blend_shapes`` for downstream
tooling that might want to use it (e.g. a separate facial-rig export).

Coverage:
  * Synthesise a binary FBX with one Geometry + one BlendShape Deformer
    + one BlendShapeChannel + one Shape Geometry, parse it, and verify
    the BlendShape entry comes out with the correct name / indexes /
    offsets.
  * Multiple channels on the same mesh.
  * Inline ``Shape`` records (legacy form) under the Geometry node.
  * Validate that ``imported_to_nj`` ignores the shapes but documents
    the count via ``import_diagnostics`` on the returned NjModel.
"""
from __future__ import annotations

import struct
from typing import List, Sequence, Tuple

import numpy as np
import pytest

from formats.fbx_reader import parse_fbx, parse_fbx_with_animations
from formats.import_external import (
    BlendShape,
    ImportedModel,
    imported_to_nj,
)

from tests._fbx_fixtures import FbxNode, encode_fbx


# ---------------------------------------------------------------------------
# Synthetic FBX builders for blend-shape coverage
# ---------------------------------------------------------------------------


def _make_skinned_quad_geom(
    geom_id: int,
    name: str = "Body",
) -> FbxNode:
    """Build one Geometry record with 4 vertices in a 1x1 quad on XZ.

    Used as the base mesh that blend-shape Deformers attach to.
    """
    g = FbxNode("Geometry")
    g.L(geom_id).S(f"{name}\x00\x01Geometry").S("Mesh")
    g.add(FbxNode("Vertices")).Darr([
        -1.0, 0.0, -1.0,
         1.0, 0.0, -1.0,
         1.0, 0.0,  1.0,
        -1.0, 0.0,  1.0,
    ])
    g.add(FbxNode("PolygonVertexIndex")).Iarr([0, 1, 2, -4])
    le_n = g.add(FbxNode("LayerElementNormal"))
    le_n.I(0)
    le_n.add(FbxNode("Version")).I(101)
    le_n.add(FbxNode("MappingInformationType")).S("ByVertex")
    le_n.add(FbxNode("ReferenceInformationType")).S("Direct")
    le_n.add(FbxNode("Normals")).Darr([0, 1, 0,  0, 1, 0,  0, 1, 0,  0, 1, 0])
    return g


def _make_shape_geom(
    shape_id: int,
    name: str,
    indexes: Sequence[int],
    offsets: Sequence[Tuple[float, float, float]],
    normals: Sequence[Tuple[float, float, float]] = (),
) -> FbxNode:
    """Build a Geometry record with sub_type='Shape' (a blend-shape target)."""
    g = FbxNode("Geometry")
    g.L(shape_id).S(f"{name}\x00\x01Geometry").S("Shape")
    g.add(FbxNode("Indexes")).Iarr(list(indexes))
    flat_offsets: List[float] = []
    for o in offsets:
        flat_offsets.extend(o)
    g.add(FbxNode("Vertices")).Darr(flat_offsets)
    if normals:
        flat_normals: List[float] = []
        for n in normals:
            flat_normals.extend(n)
        g.add(FbxNode("Normals")).Darr(flat_normals)
    return g


def _make_blendshape_chain(
    objs: FbxNode,
    geom_id: int,
    deformer_id: int,
    channel_id: int,
    shape_id: int,
    channel_name: str,
    shape_indexes: Sequence[int],
    shape_offsets: Sequence[Tuple[float, float, float]],
    *,
    deform_percent: float = 0.0,
    shape_normals: Sequence[Tuple[float, float, float]] = (),
) -> List[FbxNode]:
    """Append the Deformer/SubDeformer/Shape chain for one blend shape.

    Returns the list of newly added nodes; the caller is responsible for
    wiring up ``Connections``.
    """
    # Deformer (BlendShape).
    bs = objs.add(FbxNode("Deformer"))
    bs.L(deformer_id).S(f"{channel_name}_BS\x00\x01Deformer").S("BlendShape")
    bs.add(FbxNode("Version")).I(100)

    # SubDeformer (BlendShapeChannel).
    ch = objs.add(FbxNode("Deformer"))
    ch.L(channel_id).S(f"{channel_name}\x00\x01SubDeformer").S("BlendShapeChannel")
    ch.add(FbxNode("Version")).I(100)
    # Properties70 with DeformPercent (FBX uses 0..100).
    p70 = ch.add(FbxNode("Properties70"))
    p = p70.add(FbxNode("P"))
    p.S("DeformPercent").S("Number").S("").S("A").D(deform_percent * 100.0)

    # Shape Geometry.
    shape = _make_shape_geom(
        shape_id, channel_name, shape_indexes, shape_offsets,
        normals=shape_normals,
    )
    objs.add(shape)
    return [bs, ch, shape]


def build_one_channel_blendshape_fbx() -> bytes:
    """Build an FBX with one Geometry + one BlendShape channel ('Smile')."""
    root = FbxNode("__root__")
    root.add(FbxNode("FBXHeaderExtension")).add(FbxNode("FBXVersion")).I(7400)
    root.add(FbxNode("GlobalSettings")).add(FbxNode("Version")).I(1000)

    objs = root.add(FbxNode("Objects"))

    geom_id = 1000
    model_id = 2000
    deformer_id = 3000
    channel_id = 3100
    shape_id = 3200

    g = _make_skinned_quad_geom(geom_id, name="Body")
    objs.add(g)

    m = objs.add(FbxNode("Model"))
    m.L(model_id).S("Body\x00\x01Model").S("Mesh")
    m.add(FbxNode("Version")).I(232)

    _make_blendshape_chain(
        objs, geom_id, deformer_id, channel_id, shape_id,
        channel_name="Smile",
        shape_indexes=[0, 2],
        shape_offsets=[(0.0, 0.5, 0.0), (0.0, 0.5, 0.0)],
        shape_normals=[(0.0, 0.0, 0.1), (0.0, 0.0, 0.1)],
        deform_percent=0.0,
    )

    conns = root.add(FbxNode("Connections"))
    conns.add(FbxNode("C")).S("OO").L(geom_id).L(model_id)
    conns.add(FbxNode("C")).S("OO").L(model_id).L(0)
    conns.add(FbxNode("C")).S("OO").L(deformer_id).L(geom_id)
    conns.add(FbxNode("C")).S("OO").L(channel_id).L(deformer_id)
    conns.add(FbxNode("C")).S("OO").L(shape_id).L(channel_id)

    return encode_fbx(root)


def build_two_channel_blendshape_fbx() -> bytes:
    """Build an FBX with one Geometry + TWO BlendShape channels."""
    root = FbxNode("__root__")
    root.add(FbxNode("FBXHeaderExtension")).add(FbxNode("FBXVersion")).I(7400)
    root.add(FbxNode("GlobalSettings")).add(FbxNode("Version")).I(1000)
    objs = root.add(FbxNode("Objects"))

    geom_id = 1000
    model_id = 2000
    deformer_id_a = 3000
    channel_a_id = 3100
    shape_a_id = 3200
    deformer_id_b = 3300
    channel_b_id = 3400
    shape_b_id = 3500

    objs.add(_make_skinned_quad_geom(geom_id, name="Body"))

    m = objs.add(FbxNode("Model"))
    m.L(model_id).S("Body\x00\x01Model").S("Mesh")
    m.add(FbxNode("Version")).I(232)

    _make_blendshape_chain(
        objs, geom_id, deformer_id_a, channel_a_id, shape_a_id,
        channel_name="Smile",
        shape_indexes=[0, 2],
        shape_offsets=[(0.0, 0.5, 0.0), (0.0, 0.5, 0.0)],
        deform_percent=0.0,
    )
    _make_blendshape_chain(
        objs, geom_id, deformer_id_b, channel_b_id, shape_b_id,
        channel_name="Frown",
        shape_indexes=[1, 3],
        shape_offsets=[(0.1, -0.2, 0.0), (-0.1, -0.2, 0.0)],
        deform_percent=0.25,  # static default weight
    )

    conns = root.add(FbxNode("Connections"))
    conns.add(FbxNode("C")).S("OO").L(geom_id).L(model_id)
    conns.add(FbxNode("C")).S("OO").L(model_id).L(0)
    # Smile chain
    conns.add(FbxNode("C")).S("OO").L(deformer_id_a).L(geom_id)
    conns.add(FbxNode("C")).S("OO").L(channel_a_id).L(deformer_id_a)
    conns.add(FbxNode("C")).S("OO").L(shape_a_id).L(channel_a_id)
    # Frown chain (shares the same BlendShape Deformer for it could be
    # one Deformer with multiple channels — but a separate Deformer is
    # also legal and simpler to build).
    conns.add(FbxNode("C")).S("OO").L(deformer_id_b).L(geom_id)
    conns.add(FbxNode("C")).S("OO").L(channel_b_id).L(deformer_id_b)
    conns.add(FbxNode("C")).S("OO").L(shape_b_id).L(channel_b_id)
    return encode_fbx(root)


def build_inline_shape_blendshape_fbx() -> bytes:
    """Build an FBX with a Shape record nested INSIDE the Geometry record.

    Some legacy FBX exporters and a handful of pre-2014 7.x writers put
    Shape records directly under the parent Geometry rather than
    chained via Connections. The parser handles this fallback path.
    """
    root = FbxNode("__root__")
    root.add(FbxNode("FBXHeaderExtension")).add(FbxNode("FBXVersion")).I(7400)
    root.add(FbxNode("GlobalSettings")).add(FbxNode("Version")).I(1000)
    objs = root.add(FbxNode("Objects"))

    geom_id = 1000
    model_id = 2000

    g = _make_skinned_quad_geom(geom_id, name="Body")
    # Inline Shape as a child of the Geometry. Note: legacy form gives
    # the Shape its own object id but doesn't connect via Connections.
    inline_shape = FbxNode("Shape")
    inline_shape.L(9999).S("InlineLegacyShape\x00\x01Geometry").S("Shape")
    inline_shape.add(FbxNode("Indexes")).Iarr([1])
    inline_shape.add(FbxNode("Vertices")).Darr([0.7, 0.0, 0.0])
    g.add(inline_shape)
    objs.add(g)

    m = objs.add(FbxNode("Model"))
    m.L(model_id).S("Body\x00\x01Model").S("Mesh")
    m.add(FbxNode("Version")).I(232)

    conns = root.add(FbxNode("Connections"))
    conns.add(FbxNode("C")).S("OO").L(geom_id).L(model_id)
    conns.add(FbxNode("C")).S("OO").L(model_id).L(0)
    return encode_fbx(root)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_parse_fbx_extracts_one_blend_shape():
    """A single-channel blend-shape FBX yields one BlendShape entry."""
    data = build_one_channel_blendshape_fbx()
    model = parse_fbx(data)

    assert len(model.blend_shapes) == 1
    bs = model.blend_shapes[0]
    assert isinstance(bs, BlendShape)
    assert bs.name == "Smile"
    assert bs.mesh_name == "Body"
    np.testing.assert_array_equal(bs.indexes, np.array([0, 2], dtype=np.int32))
    np.testing.assert_allclose(bs.offsets, np.array([
        [0.0, 0.5, 0.0],
        [0.0, 0.5, 0.0],
    ], dtype=np.float32))
    assert bs.normals is not None
    np.testing.assert_allclose(bs.normals, np.array([
        [0.0, 0.0, 0.1],
        [0.0, 0.0, 0.1],
    ], dtype=np.float32))
    # Default DeformPercent was 0 in the fixture.
    assert bs.default_weight == 0.0


def test_parse_fbx_extracts_multiple_blend_shapes():
    """Two channels on the same mesh both come through with distinct names."""
    data = build_two_channel_blendshape_fbx()
    model = parse_fbx(data)

    assert len(model.blend_shapes) == 2
    names = sorted(bs.name for bs in model.blend_shapes)
    assert names == ["Frown", "Smile"]
    # Frown had DeformPercent=25 → default_weight=0.25.
    by_name = {bs.name: bs for bs in model.blend_shapes}
    assert abs(by_name["Frown"].default_weight - 0.25) < 1e-6
    np.testing.assert_array_equal(
        by_name["Frown"].indexes, np.array([1, 3], dtype=np.int32),
    )


def test_parse_fbx_extracts_inline_shape():
    """Legacy inline Shape (under Geometry, no Connections) parses too."""
    data = build_inline_shape_blendshape_fbx()
    model = parse_fbx(data)

    assert len(model.blend_shapes) == 1
    bs = model.blend_shapes[0]
    assert bs.name == "InlineLegacyShape"
    np.testing.assert_array_equal(bs.indexes, np.array([1], dtype=np.int32))


def test_parse_fbx_no_blend_shapes_yields_empty_list():
    """An FBX without shapes has empty blend_shapes (default factory)."""
    from tests._fbx_fixtures import build_static_cube_fbx
    data = build_static_cube_fbx()
    model = parse_fbx(data)
    assert model.blend_shapes == []
    # And the warnings list shouldn't include the morph-disclaimer line.
    assert not any("blend shape" in w.lower() for w in model.warnings)


def test_parse_fbx_blend_shapes_emits_warning():
    """When shapes exist, parse_fbx emits a clarifying warning."""
    data = build_one_channel_blendshape_fbx()
    model = parse_fbx(data)
    assert any("blend shape" in w.lower() for w in model.warnings), model.warnings
    # The warning specifically calls out PSOBB's lack of morph rendering.
    assert any("PSOBB" in w for w in model.warnings)


def test_parse_fbx_with_animations_keeps_blend_shapes():
    """parse_fbx_with_animations preserves blend_shapes too (delegated path)."""
    data = build_two_channel_blendshape_fbx()
    iwa = parse_fbx_with_animations(data)
    assert len(iwa.model.blend_shapes) == 2


def test_imported_to_nj_documents_ignored_blend_shapes():
    """imported_to_nj flags blend-shape data as preserved-but-not-rendered."""
    data = build_one_channel_blendshape_fbx()
    model = parse_fbx(data)
    nj = imported_to_nj(model, axis_flip_z=False, scale=1.0)
    diag = getattr(nj, "import_diagnostics", None)
    assert diag is not None, "expected import_diagnostics attr on NjModel"
    assert diag["blend_shapes_ignored"] == 1
    assert diag["blend_shape_names"] == ["Smile"]
    assert "morph" in diag["note"].lower() or "blend" in diag["note"].lower()


def test_imported_to_nj_no_diag_for_shape_free_models():
    """A model without blend shapes carries no import_diagnostics."""
    from tests._fbx_fixtures import build_static_cube_fbx
    data = build_static_cube_fbx()
    model = parse_fbx(data)
    nj = imported_to_nj(model, axis_flip_z=False, scale=1.0)
    # Either the attribute is absent or it doesn't carry blend-shape diag.
    diag = getattr(nj, "import_diagnostics", None)
    assert diag is None or "blend_shapes_ignored" not in diag


def test_blend_shape_offsets_in_source_coords():
    """BlendShape offsets are NOT pre-multiplied by axis-flip / scale.

    Source coordinate space is preserved so callers using the data for
    other purposes can apply their own transform pipeline.
    """
    data = build_one_channel_blendshape_fbx()
    model = parse_fbx(data)
    bs = model.blend_shapes[0]
    # Fixture used Y-axis offsets of 0.5 — verify they survived parse.
    assert bs.offsets[0, 1] == pytest.approx(0.5)
    assert bs.offsets[1, 1] == pytest.approx(0.5)
    # Z is still 0 (no flip applied).
    assert bs.offsets[0, 2] == 0.0
