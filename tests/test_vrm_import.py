"""Tests for VRM rig import + retarget routing.

VRM is a glTF 2.0 extension popularised by VRoid Studio for CC0 anime
characters. Two on-disk variants are common in the wild:

  * VRM 0.x   -- ``extensions.VRM`` block, ``humanoid.humanBones`` is a
                 LIST of ``{"bone": <role>, "node": <node-idx>}`` entries.
  * VRM 1.0   -- ``extensions.VRMC_vrm`` block, ``humanoid.humanBones`` is
                 a DICT keyed by role name: ``{"hips": {"node": 1}}``.

The importer is supposed to extract the per-role bone-index map for
both formats and stash it on ``ImportedModel.vrm_humanoid_map``. The
retargeter then uses it as an authoritative routing table that
bypasses string-matching: a bone named ``"Bone001_RandomName"`` whose
VRM role is ``leftUpperArm`` must still retarget onto ``LeftArm``.

Tests in this module exercise:

  1. VRM 1.0 detection + map extraction (synthetic GLB / JSON).
  2. VRM 0.x detection + map extraction (synthetic JSON).
  3. End-to-end retarget routing via the VRM map even when bone names
     are gibberish.
  4. Round-trip through ``imported_to_json`` / ``imported_from_json``.
  5. Smoke test against any locally available CC0 VRM in
     ``data/test_assets/`` (skipped if not present).

License: free CC0 VRM characters can be sampled from VRoid Hub
(https://hub.vroid.com/), Booth-PM, or Sketchfab. We document the
search but don't ship a binary in the test fixtures (the synthetic
JSON exercises the parser path end-to-end without needing one).
"""
from __future__ import annotations

import json
import math
import pathlib
from typing import Dict, List, Optional

import pytest


# Skip the whole module if pygltflib is unavailable.
pygltflib = pytest.importorskip("pygltflib")


# ---------------------------------------------------------------------------
# Synthetic VRM fixture builders
# ---------------------------------------------------------------------------


def _make_skeleton_nodes(node_names: List[str], child_chain: List[int]) -> List[object]:
    """Build a list of ``pygltflib.Node`` for a linear bone chain.

    ``child_chain`` lists the index of each node's first child (-1 when
    the node is a leaf). All nodes get a small +Y bind translation so
    the skeleton has nonzero extent (for FK to look reasonable).
    """
    nodes = []
    for i, name in enumerate(node_names):
        n = pygltflib.Node(name=name, translation=[0.0, 0.2, 0.0])
        if 0 <= child_chain[i] < len(node_names):
            n.children = [child_chain[i]]
        nodes.append(n)
    return nodes


def _build_vrm10_glb(
    node_names: List[str],
    humanoid_map: Dict[str, int],
    *,
    spec_version: str = "1.0",
) -> bytes:
    """Build a synthetic VRM 1.0 glTF (JSON-only, no buffers).

    Returns the raw JSON bytes — small + parseable by the importer's
    ``parse_gltf(glb=False)`` path. We don't bother with buffer / mesh
    data; the VRM map extraction is purely metadata work.
    """
    g = pygltflib.GLTF2()
    g.asset = pygltflib.Asset(version="2.0")
    # Linear chain: each node's child is the next, last is leaf.
    n = len(node_names)
    chain = [i + 1 if i + 1 < n else -1 for i in range(n)]
    g.nodes = _make_skeleton_nodes(node_names, chain)
    g.scenes = [pygltflib.Scene(nodes=[0])]
    g.scene = 0
    g.skins = [pygltflib.Skin(joints=list(range(n)))]
    # VRM 1.0 humanoid block: dict keyed by role name.
    human_bones_dict = {role: {"node": int(idx)} for role, idx in humanoid_map.items()}
    g.extensions = {
        "VRMC_vrm": {
            "specVersion": spec_version,
            "humanoid": {"humanBones": human_bones_dict},
        }
    }
    return g.to_json().encode("utf-8")


def _build_vrm0_glb(
    node_names: List[str],
    humanoid_map: Dict[str, int],
) -> bytes:
    """Build a synthetic VRM 0.x glTF (JSON-only).

    VRM 0.x stores ``humanBones`` as a LIST of
    ``{"bone": role, "node": idx}`` dicts. We exercise that format
    explicitly so the parser handles both legacy + 1.0 layouts.
    """
    g = pygltflib.GLTF2()
    g.asset = pygltflib.Asset(version="2.0")
    n = len(node_names)
    chain = [i + 1 if i + 1 < n else -1 for i in range(n)]
    g.nodes = _make_skeleton_nodes(node_names, chain)
    g.scenes = [pygltflib.Scene(nodes=[0])]
    g.scene = 0
    g.skins = [pygltflib.Skin(joints=list(range(n)))]
    human_bones_list = [{"bone": role, "node": int(idx)} for role, idx in humanoid_map.items()]
    g.extensions = {
        "VRM": {
            "specVersion": "0.0",
            "humanoid": {"humanBones": human_bones_list},
        }
    }
    return g.to_json().encode("utf-8")


# ---------------------------------------------------------------------------
# Tests: detection + extraction
# ---------------------------------------------------------------------------


def test_vrm_10_humanoid_map_extracted():
    """A VRM 1.0 file's humanoid map ends up on ImportedModel."""
    from formats.import_external import parse_gltf

    node_names = ["Bone_Root", "Bone_Spine", "Bone_LU", "Bone_LF"]
    vrm_map = {
        "hips": 0,
        "spine": 1,
        "leftUpperArm": 2,
        "leftLowerArm": 3,
    }
    data = _build_vrm10_glb(node_names, vrm_map)
    model = parse_gltf(data, glb=False)

    assert model.vrm_humanoid_map == {
        "hips": 0,
        "spine": 1,
        "leftUpperArm": 2,
        "leftLowerArm": 3,
    }
    # Diagnostic warning surfaces the VRM detection.
    assert any("VRM 1.0" in w for w in model.warnings)


def test_vrm_0x_humanoid_map_extracted():
    """A VRM 0.x file's list-form humanoid map decodes the same way."""
    from formats.import_external import parse_gltf

    node_names = ["Bone_Root", "Bone_Spine", "Bone_LU"]
    vrm_map = {
        "hips": 0,
        "spine": 1,
        "leftUpperArm": 2,
    }
    data = _build_vrm0_glb(node_names, vrm_map)
    model = parse_gltf(data, glb=False)

    assert model.vrm_humanoid_map == {
        "hips": 0,
        "spine": 1,
        "leftUpperArm": 2,
    }
    assert any("VRM 0.x" in w for w in model.warnings)


def test_non_vrm_glb_has_empty_humanoid_map():
    """A vanilla glTF (no VRM extension) leaves vrm_humanoid_map empty."""
    from formats.import_external import parse_gltf

    g = pygltflib.GLTF2()
    g.asset = pygltflib.Asset(version="2.0")
    g.nodes = _make_skeleton_nodes(["a", "b"], [1, -1])
    g.scenes = [pygltflib.Scene(nodes=[0])]
    g.scene = 0
    g.skins = [pygltflib.Skin(joints=[0, 1])]
    data = g.to_json().encode("utf-8")
    model = parse_gltf(data, glb=False)
    assert model.vrm_humanoid_map == {}
    # No VRM warning either.
    assert not any("VRM" in w for w in model.warnings)


def test_vrm_humanoid_map_drops_non_skin_nodes():
    """Roles whose target node isn't in skin[0].joints are dropped silently."""
    from formats.import_external import parse_gltf

    # Skin only has nodes 0 + 1 — the VRM map references node 2 (not a joint)
    # which the importer should silently drop.
    g = pygltflib.GLTF2()
    g.asset = pygltflib.Asset(version="2.0")
    g.nodes = [
        pygltflib.Node(name="hips_node", children=[1]),
        pygltflib.Node(name="spine_node"),
        pygltflib.Node(name="orphan_node"),
    ]
    g.scenes = [pygltflib.Scene(nodes=[0])]
    g.scene = 0
    g.skins = [pygltflib.Skin(joints=[0, 1])]  # node 2 not a joint
    g.extensions = {
        "VRMC_vrm": {
            "specVersion": "1.0",
            "humanoid": {"humanBones": {
                "hips": {"node": 0},
                "spine": {"node": 1},
                "leftEye": {"node": 2},  # secondary joint outside skin
            }},
        }
    }
    data = g.to_json().encode("utf-8")
    model = parse_gltf(data, glb=False)
    assert model.vrm_humanoid_map == {"hips": 0, "spine": 1}


def test_vrm_humanoid_map_handles_malformed_entries():
    """Malformed VRM entries get skipped without raising."""
    from formats.import_external import parse_gltf

    g = pygltflib.GLTF2()
    g.asset = pygltflib.Asset(version="2.0")
    g.nodes = _make_skeleton_nodes(["a", "b"], [1, -1])
    g.scenes = [pygltflib.Scene(nodes=[0])]
    g.scene = 0
    g.skins = [pygltflib.Skin(joints=[0, 1])]
    g.extensions = {
        "VRMC_vrm": {
            "specVersion": "1.0",
            "humanoid": {"humanBones": {
                "hips": {"node": 0},      # ok
                "spine": {"node": "no"},  # invalid type
                "neck": {},               # missing node
                "head": "nonsense",       # wrong shape
            }},
        }
    }
    data = g.to_json().encode("utf-8")
    model = parse_gltf(data, glb=False)
    # Only the valid entry survives.
    assert model.vrm_humanoid_map == {"hips": 0}


def test_vrm_takes_precedence_over_vrm_legacy_when_both_present():
    """A file with both VRMC_vrm + VRM blocks prefers the 1.0 spec."""
    from formats.import_external import parse_gltf

    g = pygltflib.GLTF2()
    g.asset = pygltflib.Asset(version="2.0")
    g.nodes = _make_skeleton_nodes(["a", "b"], [1, -1])
    g.scenes = [pygltflib.Scene(nodes=[0])]
    g.scene = 0
    g.skins = [pygltflib.Skin(joints=[0, 1])]
    g.extensions = {
        "VRM": {  # legacy 0.x — ignored when 1.0 also present
            "specVersion": "0.0",
            "humanoid": {"humanBones": [{"bone": "head", "node": 0}]},
        },
        "VRMC_vrm": {
            "specVersion": "1.0",
            "humanoid": {"humanBones": {"hips": {"node": 0}}},
        },
    }
    data = g.to_json().encode("utf-8")
    model = parse_gltf(data, glb=False)
    assert model.vrm_humanoid_map == {"hips": 0}
    assert any("VRM 1.0" in w for w in model.warnings)


# ---------------------------------------------------------------------------
# Tests: retarget routing via the VRM map
# ---------------------------------------------------------------------------


def test_vrm_map_routes_gibberish_bone_names_to_canonical():
    """VRM-tagged bones with non-canonical names retarget via humanoid role.

    Setup: a 4-bone source skeleton named ``Garbage_*`` so string-match
    against the lobby_girl bone map would fail entirely. The VRM
    humanoid map tags each bone with a role; the retargeter routes via
    role -> canonical -> bone_map and recovers the mapping.
    """
    import math
    from formats.anim_retarget import (
        LOBBY_GIRL_BONE_MAP,
        retarget_animation,
        summarize_retarget,
    )
    from formats.import_external import (
        ImportedAnimation,
        ImportedBone,
        ImportedTrack,
    )

    # Source skeleton with names that DON'T match any bone-map key.
    src_skel = [
        ImportedBone(name="Garbage_Root", parent_idx=-1,
                     bind_pos=(0.0, 1.0, 0.0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="Random123", parent_idx=0,
                     bind_pos=(0.0, 0.4, 0.0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="VRoidExportedBoneNN", parent_idx=1,
                     bind_pos=(0.3, 0.0, 0.0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="Bone001_left_arm", parent_idx=2,
                     bind_pos=(0.0, -0.3, 0.0), bind_rot_quat=(0, 0, 0, 1)),
    ]
    # VRM humanoid map tags each bone with a role.
    vrm_map = {
        "hips": 0,
        "spine": 1,
        "leftUpperArm": 2,
        "leftLowerArm": 3,
    }
    # One animation track on bone 2 (leftUpperArm).
    times = [f / 30.0 for f in range(15)]
    values = []
    for f in range(15):
        ang = math.pi * 0.5 * f / 14.0
        values.append((0.0, math.sin(ang * 0.5), 0.0, math.cos(ang * 0.5)))
    src_anim = ImportedAnimation(
        name="VrmTestAnim",
        duration_seconds=times[-1],
        fps_target=30,
        tracks=[ImportedTrack(
            bone_idx=2,
            channel="rotation",
            times=times,
            values=values,
        )],
    )
    # Target = a small synthetic skeleton with the same shape as
    # lobby_girl's relevant bones. For the test we just need it large
    # enough that the bone-map indices are in range.
    n_target = max(LOBBY_GIRL_BONE_MAP.values()) + 1
    tgt_skel = [
        ImportedBone(name=f"tgt{i}", parent_idx=-1 if i == 0 else 0,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1))
        for i in range(n_target)
    ]
    motion = retarget_animation(
        src_anim, src_skel, tgt_skel,
        dict(LOBBY_GIRL_BONE_MAP),
        target_fps=30,
        flip_z=False,
        enable_ik=False,
        vrm_humanoid_map=vrm_map,
        # Disable auto-detect so we KNOW the routing came via VRM, not
        # the heuristic fallback.
        enable_auto_detect=False,
    )
    summary = summarize_retarget(motion)
    # All 4 source bones should map via the VRM path.
    assert summary["mapped_bones"] == 4
    assert summary["dropped_bones"] == 0
    # Resolution counters: 4 vrm-resolved, 0 direct, 0 auto.
    res = summary["resolution"]
    assert res["vrm_resolved"] == 4
    assert res["direct_mapped"] == 0
    assert res["auto_detected"] == 0


def test_vrm_map_falls_through_when_role_not_in_bone_map():
    """A VRM role with no bone_map entry falls through to direct/auto path."""
    from formats.anim_retarget import retarget_animation, summarize_retarget
    from formats.import_external import (
        ImportedAnimation,
        ImportedBone,
        ImportedTrack,
    )

    src_skel = [
        ImportedBone(name="LeftArm", parent_idx=-1,  # canonical name in bone_map
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="WeirdName", parent_idx=0,  # not in any map
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
    ]
    src_anim = ImportedAnimation(
        name="t", duration_seconds=0.0, fps_target=30, tracks=[],
    )
    tgt_skel = [
        ImportedBone(name="t0", parent_idx=-1,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
    ]
    bone_map = {"LeftArm": 0}  # only one key — VRM role "rightToes" won't resolve
    vrm_map = {"rightToes": 1}  # role not in bone_map
    motion = retarget_animation(
        src_anim, src_skel, tgt_skel, bone_map,
        target_fps=30, flip_z=False, enable_ik=False,
        vrm_humanoid_map=vrm_map, enable_auto_detect=False,
    )
    summary = summarize_retarget(motion)
    # bone 0 = direct match via "LeftArm". bone 1 = VRM role rightToes
    # has no bone_map entry, falls through, no auto-detect → dropped.
    assert summary["mapped_bones"] == 1
    assert summary["dropped_bones"] == 1
    assert summary["resolution"]["direct_mapped"] == 1
    assert summary["resolution"]["vrm_resolved"] == 0


# ---------------------------------------------------------------------------
# Tests: JSON wire round-trip
# ---------------------------------------------------------------------------


def test_vrm_humanoid_map_round_trips_through_json():
    """imported_to_json + imported_from_json preserve the humanoid map."""
    from formats.import_external import (
        imported_from_json,
        imported_to_json,
        parse_gltf,
    )

    node_names = ["root", "spine", "lua", "lla"]
    vrm_map = {
        "hips": 0, "spine": 1, "leftUpperArm": 2, "leftLowerArm": 3,
    }
    data = _build_vrm10_glb(node_names, vrm_map)
    model = parse_gltf(data, glb=False)

    js = imported_to_json(model)
    assert "vrm_humanoid_map" in js
    assert js["vrm_humanoid_map"] == vrm_map

    # Round-trip via the inverse helper.
    restored = imported_from_json(js)
    assert restored.vrm_humanoid_map == vrm_map


def test_non_vrm_json_omits_humanoid_map():
    """Non-VRM models don't bloat the JSON with an empty humanoid map."""
    from formats.import_external import imported_to_json, parse_gltf

    g = pygltflib.GLTF2()
    g.asset = pygltflib.Asset(version="2.0")
    g.nodes = _make_skeleton_nodes(["a", "b"], [1, -1])
    g.scenes = [pygltflib.Scene(nodes=[0])]
    g.scene = 0
    g.skins = [pygltflib.Skin(joints=[0, 1])]
    data = g.to_json().encode("utf-8")
    model = parse_gltf(data, glb=False)
    js = imported_to_json(model)
    # Empty maps shouldn't bloat the wire shape — check it's omitted.
    assert "vrm_humanoid_map" not in js


# ---------------------------------------------------------------------------
# Smoke test: any locally available CC0 VRM
# ---------------------------------------------------------------------------


def test_local_cc0_vrm_smoke():
    """If a CC0 VRM is in data/test_assets/, retarget it onto lobby_girl.

    Sourcing notes: CC0 VRoid characters can be downloaded from
      * VRoid Hub        https://hub.vroid.com/
      * Booth-PM         https://booth.pm/
      * Sketchfab        https://sketchfab.com/3d-models?features=downloadable&licenses=322a749bcfa841b29dff1e8a1bb74b0b
    Drop the .vrm file into ``data/test_assets/`` (the importer recognises
    .vrm via the GLB magic). A synthetic 20-joint humanoid (no mesh) is
    bundled at ``data/test_assets/synthetic_humanoid.vrm`` so the test
    runs in CI without requiring an external download.
    """
    asset_dir = pathlib.Path(__file__).parent.parent / "data" / "test_assets"
    if not asset_dir.exists():
        pytest.skip("no data/test_assets/ directory")
    candidates = list(asset_dir.glob("*.vrm")) + list(asset_dir.glob("*_vrm*.glb"))
    if not candidates:
        pytest.skip("no CC0 VRM in data/test_assets/ (see test docstring)")
    vrm_path = candidates[0]

    from formats.import_external import parse_gltf
    data = vrm_path.read_bytes()
    model = parse_gltf(data, glb=True)
    # A real VRM should have ≥ 1 humanoid role recovered.
    assert len(model.vrm_humanoid_map) > 0, (
        f"{vrm_path.name}: VRM extension parse failed, humanoid map empty"
    )
    # Hips should be present in EVERY VRM rig.
    assert "hips" in model.vrm_humanoid_map


def test_local_vrm_retargets_to_lobby_girl():
    """End-to-end: synthetic VRM → lobby-girl skeleton via humanoid map.

    Drives a single rotation track on ``leftUpperArm`` and verifies the
    retargeter routes via the VRM humanoid map → ``LeftArm`` →
    ``LOBBY_GIRL_BONE_MAP[LeftArm]``.
    """
    import math
    asset_dir = pathlib.Path(__file__).parent.parent / "data" / "test_assets"
    vrm_path = asset_dir / "synthetic_humanoid.vrm"
    if not vrm_path.exists():
        pytest.skip("synthetic_humanoid.vrm fixture missing")

    from formats.anim_retarget import (
        LOBBY_GIRL_BONE_MAP,
        retarget_animation,
        summarize_retarget,
    )
    from formats.import_external import (
        ImportedAnimation,
        ImportedBone,
        ImportedTrack,
        parse_gltf,
    )

    model = parse_gltf(vrm_path.read_bytes(), glb=True)
    assert len(model.bones) == 20
    assert "leftUpperArm" in model.vrm_humanoid_map

    # Build a synthetic animation that drives the leftUpperArm joint.
    larm_idx = model.vrm_humanoid_map["leftUpperArm"]
    times = [f / 30.0 for f in range(15)]
    values = [
        (0.0,
         math.sin(math.pi * 0.5 * f / 14.0 * 0.5),
         0.0,
         math.cos(math.pi * 0.5 * f / 14.0 * 0.5))
        for f in range(15)
    ]
    src_anim = ImportedAnimation(
        name="VrmTest",
        duration_seconds=times[-1],
        fps_target=30,
        tracks=[ImportedTrack(
            bone_idx=larm_idx,
            channel="rotation",
            times=times,
            values=values,
        )],
    )

    # Lobby-girl-sized target skeleton (synthetic stand-in; the test
    # exercises routing, not the kenkyu_w bind pose). Build with
    # ``parent_idx`` chains so FK / IK don't crash on missing parents.
    n_target = max(LOBBY_GIRL_BONE_MAP.values()) + 1
    tgt_skel = [
        ImportedBone(name=f"tgt{i}", parent_idx=-1 if i == 0 else 0,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1))
        for i in range(n_target)
    ]
    motion = retarget_animation(
        src_anim, model.bones, tgt_skel,
        dict(LOBBY_GIRL_BONE_MAP),
        target_fps=30,
        flip_z=False,
        enable_ik=False,
        vrm_humanoid_map=model.vrm_humanoid_map,
        # Disable auto-detect so any successful routing is via VRM only.
        enable_auto_detect=False,
    )
    summary = summarize_retarget(motion)
    # All 20 source bones should map via VRM (every one has a humanoid
    # role and all 20 roles have entries in LOBBY_GIRL_BONE_MAP).
    assert summary["mapped_bones"] == 20, (
        f"VRM routing dropped {summary['dropped_bones']} bones; "
        f"first dropped: {summary['dropped'][:5]}"
    )
    assert summary["dropped_bones"] == 0
    assert summary["resolution"]["vrm_resolved"] == 20
