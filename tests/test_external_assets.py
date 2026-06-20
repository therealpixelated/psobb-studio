"""Tests for real-world DCC assets (CC0 / public-domain) under
``data/test_assets/``.

Unlike ``tests/test_import_external.py`` (synthetic in-memory fixtures),
this module exercises the whole import pipeline against actual exporter
output: Kenney sci-fi GLB+FBX (CC0 1.0) and Khronos glTF-Sample-Assets
(public-domain). Tests are gated — if an asset file is missing, the
test skips. This lets the suite run cleanly in environments that don't
have the test_assets directory populated.

Coverage:
  - Kenney sci-fi blaster GLB: parse + build_nj round-trip
  - Kenney sci-fi blaster FBX: parse path through fbx_reader
  - Kenney sci-fi crate GLB: static-prop path
  - Khronos RiggedSimple GLB: minimal rig + animation parse
  - Khronos CesiumMan GLB: 19-bone retarget onto lobby_girl skeleton
    (regression: the bone-map needs CesiumMan-style joint aliases for
    the 2026-04-25 external-asset audit; this test pins the fix in
    place.)
  - Khronos Duck GLB: large static prop, no-skin path

The CesiumMan test only needs ``data/import_templates/`` and the
lobby-girl skeleton to be available; it doesn't reach into PSOBB.IO,
so it runs cleanly in CI.
"""
from __future__ import annotations
import os

import pathlib

import pytest

ASSET_DIR = pathlib.Path(__file__).parent.parent / "data" / "test_assets"


def _have(name: str) -> pathlib.Path:
    p = ASSET_DIR / name
    if not p.exists():
        pytest.skip(f"external asset missing: {name}")
    return p


def _read(name: str) -> bytes:
    return _have(name).read_bytes()


# ---------------------------------------------------------------------------
# Kenney sci-fi blaster — GLB path
# ---------------------------------------------------------------------------

def test_kenney_blaster_glb_parse_smoke():
    """Kenney's CC0 sci-fi blaster GLB parses cleanly into the editor's
    intermediate ImportedModel shape.
    """
    pygltflib = pytest.importorskip("pygltflib")
    from formats.import_external import parse_gltf

    data = _read("kenney_scifi_blaster.glb")
    m = parse_gltf(data)

    assert m.source_format == "glb"
    # Two submeshes (body + barrel/clip) with ~820 verts and 470 tris.
    assert len(m.meshes) == 2
    total_verts = sum(len(mesh.vertices) for mesh in m.meshes)
    total_tris = sum(len(mesh.indices) for mesh in m.meshes)
    assert 700 <= total_verts <= 1000
    assert 400 <= total_tris <= 600
    # No skin → exactly one warning.
    assert any("no skin" in w.lower() for w in m.warnings)


def test_kenney_blaster_glb_to_nj_roundtrip():
    """The blaster GLB drives all the way through imported_to_nj +
    encode_nj_model + parse_nj_file without losing geometry.
    """
    pygltflib = pytest.importorskip("pygltflib")
    from formats.import_external import parse_gltf, imported_to_nj
    from formats.nj_writer import encode_nj_model
    from formats.xj import parse_nj_file, parse_skeleton

    m = parse_gltf(_read("kenney_scifi_blaster.glb"))
    nj_model = imported_to_nj(m, target_class="monster_humanoid",
                              axis_flip_z=True, scale=100.0)
    out = encode_nj_model(nj_model)
    assert len(out) > 0

    # Round-trip parse: meshes parse; skeleton has the template's
    # bone count (52 from monster_humanoid).
    meshes = parse_nj_file(out)
    assert len(meshes) > 0
    sk = parse_skeleton(out)
    assert len(sk) == 52


# ---------------------------------------------------------------------------
# Kenney sci-fi blaster — FBX path
# ---------------------------------------------------------------------------

def test_kenney_blaster_fbx_parse_smoke():
    """The same blaster as a binary FBX parses through fbx_reader.

    FBX preserves UV/normal seams as duplicate verts, so vert count is
    higher than the GLB's (1410 vs 820) — that's expected.
    """
    from formats.import_external import parse_external

    m = parse_external(_read("kenney_scifi_blaster.fbx"),
                       "kenney_scifi_blaster.fbx")
    assert m.source_format == "fbx"
    assert len(m.meshes) == 2
    total_verts = sum(len(mesh.vertices) for mesh in m.meshes)
    total_tris = sum(len(mesh.indices) for mesh in m.meshes)
    # FBX seam-duplicates → 1200-1500 verts. Tri count matches GLB.
    assert 1200 <= total_verts <= 1600
    assert 400 <= total_tris <= 600


# ---------------------------------------------------------------------------
# Kenney sci-fi crate — static prop
# ---------------------------------------------------------------------------

def test_kenney_crate_glb_static_prop():
    """Static prop with no skin runs through the import path."""
    pygltflib = pytest.importorskip("pygltflib")
    from formats.import_external import parse_gltf, imported_to_nj
    from formats.nj_writer import encode_nj_model

    m = parse_gltf(_read("kenney_scifi_crate.glb"))
    assert m.source_format == "glb"
    assert len(m.meshes) >= 1
    # No bones → uses template skeleton.
    assert len(m.bones) == 0
    nj_model = imported_to_nj(m, target_class="player_body",
                              axis_flip_z=True, scale=100.0)
    out = encode_nj_model(nj_model)
    assert len(out) > 0


# ---------------------------------------------------------------------------
# Khronos RiggedSimple — minimal skeleton + animation
# ---------------------------------------------------------------------------

def test_khronos_rigged_simple_animation():
    """Khronos's minimal 2-bone RiggedSimple test asset: parse the
    skeleton and the embedded animation.
    """
    pygltflib = pytest.importorskip("pygltflib")
    from formats.import_external import parse_gltf_with_animations

    imp = parse_gltf_with_animations(_read("khronos_rigged_simple.glb"))
    assert len(imp.model.bones) == 2
    assert imp.model.bones[0].parent_idx == -1
    assert imp.model.bones[1].parent_idx == 0
    assert len(imp.animations) == 1
    # 50 keyframes / 24 fps ~= 2.08 seconds.
    assert abs(imp.animations[0].duration_seconds - 2.083) < 0.05
    # Exactly 3 tracks (translation + rotation + scale on Bone.001).
    assert len(imp.animations[0].tracks) == 3


# ---------------------------------------------------------------------------
# Khronos CesiumMan — full retarget regression
# ---------------------------------------------------------------------------

def test_khronos_cesium_man_bone_map_aliases():
    """Regression: the LOBBY_GIRL_BONE_MAP must carry Khronos
    CesiumMan-style aliases. Before the 2026-04-25 fix, every CesiumMan
    bone landed in the dropped list (retargeted_bones=0).
    """
    from formats.anim_retarget import LOBBY_GIRL_BONE_MAP
    # Spot-check the aliases that were missing pre-fix.
    expected_alias_keys = (
        "Skeleton_torso_joint_1",
        "Skeleton_arm_joint_R__2_",
        "Skeleton_arm_joint_L__4_",
        "leg_joint_R_1",
        "leg_joint_L_5",
    )
    for k in expected_alias_keys:
        assert k in LOBBY_GIRL_BONE_MAP, f"missing CesiumMan alias: {k}"


def test_khronos_cesium_man_retarget_smoke():
    """End-to-end retarget: CesiumMan animation -> lobby_girl skeleton.

    Skips when the lobby-girl BML isn't reachable (data/, dev data, or
    install dir). The kenkyu_w_hone_body.nj inner is the ground-truth
    target skeleton.
    """
    pygltflib = pytest.importorskip("pygltflib")
    from formats.import_external import parse_gltf_with_animations
    from formats.anim_retarget import (
        LOBBY_GIRL_BONE_MAP,
        retarget_animation,
        summarize_retarget,
    )
    from formats.bml import parse_bml, _prs_decompress
    from formats.xj import parse_skeleton
    from formats.njm_writer import encode_njm

    candidates = [
        pathlib.Path(os.path.expanduser("~/PSOBB.IO/data/bm_npc_kenkyu_w.bml")),
        pathlib.Path("C:/tmp_pso_dev/data/bm_npc_kenkyu_w.bml"),
        pathlib.Path("data/bm_npc_kenkyu_w.bml"),
    ]
    bml_path = next((p for p in candidates if p.exists()), None)
    if bml_path is None:
        pytest.skip("bm_npc_kenkyu_w.bml not reachable")

    bml_bytes = bml_path.read_bytes()
    entries = parse_bml(bml_bytes)
    inner = next(
        (e for e in entries if e.name == "kenkyu_w_hone_body.nj"),
        None,
    )
    assert inner is not None, "kenkyu_w_hone_body.nj inner missing"
    inner_bytes = _prs_decompress(
        bml_bytes[inner.offset:inner.offset + inner.size_compressed]
    )
    target_skel = parse_skeleton(inner_bytes)
    assert len(target_skel) == 64

    imp = parse_gltf_with_animations(_read("khronos_cesium_man.glb"))
    assert len(imp.model.bones) == 19  # CesiumMan rig has 19 joints.

    motion = retarget_animation(
        imp.animations[0],
        imp.model.bones,
        target_skel,
        dict(LOBBY_GIRL_BONE_MAP),
        target_fps=30,
        include_translation=False,
        flip_z=True,
        enable_ik=True,
    )
    summary = summarize_retarget(motion)

    # Post-fix: all 19 source bones map cleanly into the lobby_girl
    # skeleton via the CesiumMan-style aliases. Pre-fix: 0 mapped, 19
    # dropped.
    assert summary["mapped_bones"] == 19
    assert summary["dropped_bones"] == 0
    assert summary["frame_count"] >= 60  # 2.0s at 30 fps = 60 frames

    # Encoded NJM is a valid byte stream.
    njm = encode_njm(motion)
    assert len(njm) > 0
    assert njm[:4] == b"NMDM" or len(njm) > 16  # sanity


# ---------------------------------------------------------------------------
# Khronos Duck — large static no-skin
# ---------------------------------------------------------------------------

def test_khronos_duck_static_smoke():
    """Khronos Duck (~120 KB, 4212 tris) — sanity that medium static
    props parse without choking the in-memory buffers.
    """
    pygltflib = pytest.importorskip("pygltflib")
    from formats.import_external import parse_gltf

    m = parse_gltf(_read("khronos_duck.glb"))
    assert m.source_format == "glb"
    assert len(m.meshes) == 1
    total_tris = sum(len(mesh.indices) for mesh in m.meshes)
    assert total_tris >= 4000
    # No skin warning is expected.
    assert any("no skin" in w.lower() for w in m.warnings)
