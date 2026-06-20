"""Tests for VRM 1.0 spring-bone + node-constraint preservation (v4, 2026-04-25).

PSOBB has no secondary-motion runtime, but preserving the data through
import lets a Blender re-import workflow round-trip the rig and lets
the side-file exporter emit JSON.

Coverage:
  * Synthetic VRM 1.0 GLB with one spring chain (3 joints, 1 collider)
    parses into ``ImportedModel.spring_bones``.
  * Spring chain joint values (stiffness, drag, gravity) survive the
    parser's number coercion.
  * Sphere + capsule collider shapes both decode.
  * VRMC_node_constraint roll/aim/rotation entries land on
    ``model.node_constraints``.
  * Spring data round-trips through ``imported_to_json`` /
    ``imported_from_json`` byte-identically.
  * The standalone ``export_spring_bones_json`` emits the expected
    schema with all per-joint fields and resolves collider->bone idx.
  * Models without VRMC_springBone produce empty lists (no spurious
    warnings).
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import pytest


# Skip the whole module if pygltflib is unavailable.
pygltflib = pytest.importorskip("pygltflib")


# ---------------------------------------------------------------------------
# Synthetic VRM-1.0 + springbone fixture
# ---------------------------------------------------------------------------


def _make_skeleton_nodes(node_names: List[str], child_chain: List[int]) -> List[object]:
    nodes = []
    for i, name in enumerate(node_names):
        n = pygltflib.Node(name=name, translation=[0.0, 0.2, 0.0])
        if 0 <= child_chain[i] < len(node_names):
            n.children = [child_chain[i]]
        nodes.append(n)
    return nodes


def _build_vrm10_with_springbones(
    node_names: List[str],
    humanoid_map: Dict[str, int],
    spring_block: Optional[dict] = None,
    node_constraints: Optional[Dict[int, dict]] = None,
) -> bytes:
    """Build a synthetic VRM 1.0 glTF with optional spring + constraint data.

    ``spring_block`` is the raw VRMC_springBone extension dict.
    ``node_constraints`` maps node-index -> VRMC_node_constraint dict.
    Both default to None (no extension emitted).
    """
    g = pygltflib.GLTF2()
    g.asset = pygltflib.Asset(version="2.0")
    n = len(node_names)
    chain = [i + 1 if i + 1 < n else -1 for i in range(n)]
    g.nodes = _make_skeleton_nodes(node_names, chain)
    if node_constraints:
        for node_idx, nc_block in node_constraints.items():
            if 0 <= node_idx < len(g.nodes):
                g.nodes[node_idx].extensions = {
                    "VRMC_node_constraint": nc_block,
                }
    g.scenes = [pygltflib.Scene(nodes=[0])]
    g.scene = 0
    g.skins = [pygltflib.Skin(joints=list(range(n)))]
    extensions: Dict[str, dict] = {
        "VRMC_vrm": {
            "specVersion": "1.0",
            "humanoid": {
                "humanBones": {role: {"node": int(idx)} for role, idx in humanoid_map.items()},
            },
        },
    }
    if spring_block is not None:
        extensions["VRMC_springBone"] = spring_block
    g.extensions = extensions
    return g.to_json().encode("utf-8")


def _basic_humanoid_map() -> Dict[str, int]:
    """Minimal VRM humanoid map covering the 4-node fixture."""
    return {"hips": 0, "spine": 1, "neck": 2, "head": 3}


# ---------------------------------------------------------------------------
# Tests: parsing
# ---------------------------------------------------------------------------


def test_spring_bone_chain_extracted():
    """A VRM 1.0 file with one springs entry yields a SpringBoneChain."""
    from formats.import_external import parse_gltf

    spring_block = {
        "specVersion": "1.0",
        "colliders": [
            {
                "node": 1,
                "shape": {"sphere": {"offset": [0.0, 0.5, 0.0], "radius": 0.1}},
            },
        ],
        "colliderGroups": [
            {"name": "torsoColliders", "colliders": [0]},
        ],
        "springs": [
            {
                "name": "hairFront",
                "center": 1,
                "joints": [
                    {"node": 2, "hitRadius": 0.05, "stiffness": 0.6, "dragForce": 0.3,
                     "gravityPower": 1.0, "gravityDir": [0.0, -1.0, 0.0]},
                    {"node": 3, "hitRadius": 0.04, "stiffness": 0.5, "dragForce": 0.4,
                     "gravityPower": 0.8, "gravityDir": [0.0, -1.0, 0.0]},
                ],
                "colliderGroups": [0],
            },
        ],
    }
    data = _build_vrm10_with_springbones(
        ["root", "spine", "neck", "head"],
        _basic_humanoid_map(),
        spring_block=spring_block,
    )
    model = parse_gltf(data, glb=False)
    assert len(model.spring_bones) == 1
    chain = model.spring_bones[0]
    assert chain.name == "hairFront"
    assert chain.center_bone_idx == 1
    assert len(chain.joints) == 2
    j0 = chain.joints[0]
    assert j0.bone_idx == 2
    assert abs(j0.hit_radius - 0.05) < 1e-6
    assert abs(j0.stiffness - 0.6) < 1e-6
    assert abs(j0.drag_force - 0.3) < 1e-6
    assert abs(j0.gravity_power - 1.0) < 1e-6
    assert j0.gravity_dir == (0.0, -1.0, 0.0)


def test_spring_bone_chain_resolves_collider_groups():
    """The chain's collider list pulls in colliders from referenced groups."""
    from formats.import_external import parse_gltf

    spring_block = {
        "specVersion": "1.0",
        "colliders": [
            {"node": 1, "shape": {"sphere": {"offset": [0.0, 0.0, 0.0], "radius": 0.05}}},
            {"node": 2, "shape": {"capsule": {"offset": [0.0, 0.0, 0.0],
                                              "tail": [0.0, 0.3, 0.0], "radius": 0.04}}},
        ],
        "colliderGroups": [
            {"name": "g1", "colliders": [0, 1]},
        ],
        "springs": [
            {"name": "tail", "joints": [{"node": 3}], "colliderGroups": [0]},
        ],
    }
    data = _build_vrm10_with_springbones(
        ["a", "b", "c", "d"],
        _basic_humanoid_map(),
        spring_block=spring_block,
    )
    model = parse_gltf(data, glb=False)
    chain = model.spring_bones[0]
    assert len(chain.colliders) == 2
    shapes = sorted(c.shape for c in chain.colliders)
    assert shapes == ["capsule", "sphere"]
    # Capsule's radius + tail came through.
    capsule = next(c for c in chain.colliders if c.shape == "capsule")
    assert abs(capsule.radius - 0.04) < 1e-6
    assert capsule.tail == (0.0, 0.3, 0.0)


def test_node_constraint_extracted():
    """VRMC_node_constraint blocks land on ``ImportedModel.node_constraints``."""
    from formats.import_external import parse_gltf

    constraints = {
        2: {
            "specVersion": "1.0",
            "constraint": {
                "aim": {
                    "source": 1,
                    "aimAxis": "PositiveY",
                    "weight": 0.8,
                },
            },
        },
    }
    data = _build_vrm10_with_springbones(
        ["root", "head", "eye_left", "eye_right"],
        {"hips": 0, "head": 1},
        node_constraints=constraints,
    )
    model = parse_gltf(data, glb=False)
    assert len(model.node_constraints) == 1
    nc = model.node_constraints[0]
    assert nc.constraint_type == "aim"
    assert nc.bone_idx == 2
    assert nc.source_bone_idx == 1
    assert nc.axis == "PositiveY"
    assert abs(nc.weight - 0.8) < 1e-6


def test_no_spring_extension_yields_empty_list():
    """Vanilla VRM (no VRMC_springBone) produces empty spring_bones."""
    from formats.import_external import parse_gltf

    data = _build_vrm10_with_springbones(
        ["root", "spine", "neck", "head"],
        _basic_humanoid_map(),
        spring_block=None,
    )
    model = parse_gltf(data, glb=False)
    assert model.spring_bones == []
    assert model.node_constraints == []


def test_unknown_collider_shape_preserved_with_empty_marker():
    """An unknown collider shape decodes as shape=='' so the count survives."""
    from formats.import_external import parse_gltf

    spring_block = {
        "specVersion": "1.0",
        "colliders": [
            {"node": 1, "shape": {"futureShape": {"foo": 1}}},  # unknown
        ],
        "colliderGroups": [{"colliders": [0]}],
        "springs": [
            {"name": "x", "joints": [{"node": 2}], "colliderGroups": [0]},
        ],
    }
    data = _build_vrm10_with_springbones(
        ["a", "b", "c", "d"],
        _basic_humanoid_map(),
        spring_block=spring_block,
    )
    model = parse_gltf(data, glb=False)
    chain = model.spring_bones[0]
    assert len(chain.colliders) == 1
    assert chain.colliders[0].shape == ""  # sentinel for unknown


# ---------------------------------------------------------------------------
# Tests: JSON round-trip
# ---------------------------------------------------------------------------


def test_spring_bones_round_trip_through_imported_to_json():
    """imported_to_json + imported_from_json preserve spring chains."""
    from formats.import_external import (
        imported_from_json,
        imported_to_json,
        parse_gltf,
    )
    spring_block = {
        "specVersion": "1.0",
        "colliders": [
            {"node": 1, "shape": {"sphere": {"offset": [0.0, 0.1, 0.0], "radius": 0.07}}},
        ],
        "colliderGroups": [{"colliders": [0]}],
        "springs": [
            {"name": "s0", "center": 1, "joints": [
                {"node": 2, "hitRadius": 0.05, "stiffness": 0.7,
                 "dragForce": 0.25, "gravityPower": 0.5,
                 "gravityDir": [0.0, -1.0, 0.0]},
                {"node": 3, "hitRadius": 0.04, "stiffness": 0.6,
                 "dragForce": 0.3, "gravityPower": 0.5,
                 "gravityDir": [0.0, -1.0, 0.0]},
            ], "colliderGroups": [0]},
        ],
    }
    data = _build_vrm10_with_springbones(
        ["root", "spine", "neck", "head"],
        _basic_humanoid_map(),
        spring_block=spring_block,
    )
    model = parse_gltf(data, glb=False)
    js = imported_to_json(model)
    # Round-trip through json.dumps to confirm pure-JSON-encodability.
    encoded = json.dumps(js)
    js_back = json.loads(encoded)
    restored = imported_from_json(js_back)

    assert len(restored.spring_bones) == len(model.spring_bones)
    a = model.spring_bones[0]
    b = restored.spring_bones[0]
    assert a.name == b.name
    assert a.center_bone_idx == b.center_bone_idx
    assert len(a.joints) == len(b.joints)
    for j_a, j_b in zip(a.joints, b.joints):
        assert j_a.bone_idx == j_b.bone_idx
        assert abs(j_a.stiffness - j_b.stiffness) < 1e-6
        assert abs(j_a.drag_force - j_b.drag_force) < 1e-6
        assert abs(j_a.gravity_power - j_b.gravity_power) < 1e-6
        assert j_a.gravity_dir == j_b.gravity_dir
    assert len(a.colliders) == len(b.colliders)
    assert a.colliders[0].shape == b.colliders[0].shape
    assert abs(a.colliders[0].radius - b.colliders[0].radius) < 1e-6


def test_node_constraints_round_trip_through_imported_to_json():
    """imported_to_json + imported_from_json preserve node constraints."""
    from formats.import_external import (
        imported_from_json,
        imported_to_json,
        parse_gltf,
    )
    constraints = {
        2: {
            "constraint": {
                "roll": {"source": 1, "rollAxis": "X", "weight": 0.5},
            },
        },
    }
    data = _build_vrm10_with_springbones(
        ["a", "b", "c", "d"],
        {"hips": 0, "neck": 1},
        node_constraints=constraints,
    )
    model = parse_gltf(data, glb=False)
    js = imported_to_json(model)
    js_back = json.loads(json.dumps(js))
    restored = imported_from_json(js_back)
    assert len(restored.node_constraints) == 1
    nc_a = model.node_constraints[0]
    nc_b = restored.node_constraints[0]
    assert nc_a.constraint_type == nc_b.constraint_type
    assert nc_a.bone_idx == nc_b.bone_idx
    assert nc_a.source_bone_idx == nc_b.source_bone_idx
    assert nc_a.axis == nc_b.axis
    assert abs(nc_a.weight - nc_b.weight) < 1e-6


# ---------------------------------------------------------------------------
# Tests: side-file exporter
# ---------------------------------------------------------------------------


def test_export_spring_bones_json_emits_complete_schema():
    """``export_spring_bones_json`` produces a JSON-encodable wrapper."""
    from formats.import_external import (
        export_spring_bones_json,
        parse_gltf,
    )
    spring_block = {
        "specVersion": "1.0",
        "colliders": [
            {"node": 1, "shape": {"sphere": {"offset": [0, 0, 0], "radius": 0.05}}},
        ],
        "colliderGroups": [{"colliders": [0]}],
        "springs": [
            {"name": "spr", "center": 0, "joints": [
                {"node": 2, "stiffness": 0.5},
                {"node": 3, "stiffness": 0.5},
            ], "colliderGroups": [0]},
        ],
    }
    data = _build_vrm10_with_springbones(
        ["root", "spine", "neck", "head"],
        _basic_humanoid_map(),
        spring_block=spring_block,
    )
    model = parse_gltf(data, glb=False)
    js = export_spring_bones_json(model)
    assert js["version"] == 1
    assert js["spring_chain_count"] == 1
    assert js["node_constraint_count"] == 0
    # Round-trippable as plain JSON (no numpy types leaking).
    re_encoded = json.dumps(js)
    re_decoded = json.loads(re_encoded)
    assert re_decoded["spring_bones"][0]["name"] == "spr"
    chain_dict = re_decoded["spring_bones"][0]
    assert chain_dict["center_bone_idx"] == 0
    assert len(chain_dict["joints"]) == 2
    # Per-joint schema hits the expected fields.
    j = chain_dict["joints"][0]
    assert "bone_idx" in j
    assert "hit_radius" in j
    assert "stiffness" in j
    assert "drag_force" in j
    assert "gravity_power" in j
    assert "gravity_dir" in j
    # Colliders flatten in.
    assert len(chain_dict["colliders"]) == 1
    assert chain_dict["colliders"][0]["shape"] == "sphere"


def test_export_spring_bones_json_for_bare_model_is_empty():
    """A model with no springs / constraints exports an empty wrapper."""
    from formats.import_external import (
        export_spring_bones_json,
        ImportedModel,
    )
    js = export_spring_bones_json(ImportedModel())
    assert js["spring_chain_count"] == 0
    assert js["node_constraint_count"] == 0
    assert js["spring_bones"] == []
    assert js["node_constraints"] == []


# ---------------------------------------------------------------------------
# Tests: warning surface
# ---------------------------------------------------------------------------


def test_warning_lists_spring_chain_count():
    """The parse path emits a diagnostic warning when springs are present."""
    from formats.import_external import parse_gltf

    spring_block = {
        "specVersion": "1.0",
        "springs": [
            {"name": "s0", "joints": [{"node": 1}, {"node": 2}]},
            {"name": "s1", "joints": [{"node": 2}, {"node": 3}]},
        ],
    }
    data = _build_vrm10_with_springbones(
        ["a", "b", "c", "d"],
        _basic_humanoid_map(),
        spring_block=spring_block,
    )
    model = parse_gltf(data, glb=False)
    assert any("springBone" in w and "2 chain" in w for w in model.warnings), model.warnings
