"""Tests for ``auto_detect_bone_role`` — heuristic rig-convention recognition.

Different DCC tools and rigging systems name humanoid joints differently:

  Mixamo            : ``mixamorig:LeftArm`` (prefix-stripped)
  Unity Mecanim     : ``LeftUpperArm`` / ``LeftLowerArm`` (camelCase)
  HumanIK           : ``LeftArm`` / ``LeftForeArm`` (already canonical)
  VRM (lowercase)   : ``leftUpperArm`` / ``leftLowerArm``
  Cesium / Khronos  : ``Skeleton_arm_joint_L__2_`` (regex match)
  Blender Rigify    : ``upper_arm.L`` / ``forearm.L`` (.L/.R suffix)
  MakeHuman         : ``upperarm_l`` / ``lowerarm_l`` (_l/_r suffix)
  Free-form         : ``"Left Arm"`` / ``"right shoulder"`` (case-insensitive)

The auto-detect heuristic returns one of the canonical role strings
LOBBY_GIRL_BONE_MAP keys on (e.g. ``"LeftArm"``, ``"RightForeArm"``).
The retargeter then looks up that role in the bone_map dict to find the
target bone index — this means we can support a new convention by
adding it to the heuristic without ever modifying ``LOBBY_GIRL_BONE_MAP``.

These tests cover:

  1. Each rig convention's main joints map to the right canonical role.
  2. Random / unknown names return None.
  3. Edge cases: empty strings, prefix-only names, ambiguous names.
  4. End-to-end retarget: a Mecanim-style source skeleton routes
     correctly through ``retarget_animation`` without explicit aliases.
  5. CesiumMan regression: the explicit aliases pinned in
     ``LOBBY_GIRL_BONE_MAP`` continue to work alongside the new heuristic.
"""
from __future__ import annotations

import math
from typing import Dict, List

import pytest

from formats.anim_retarget import (
    LOBBY_GIRL_BONE_MAP,
    auto_detect_bone_role,
    retarget_animation,
    summarize_retarget,
)
from formats.import_external import (
    ImportedAnimation,
    ImportedBone,
    ImportedTrack,
)


# ---------------------------------------------------------------------------
# Per-convention sanity tables. Each row asserts: source-bone-name -> role.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,expected", [
    # Mixamo — prefix-stripped names match canonical directly.
    ("mixamorig:LeftArm", "LeftArm"),
    ("mixamorig:RightForeArm", "RightForeArm"),
    ("mixamorig:Hips", "Hips"),
    ("mixamorig:RightFoot", "RightFoot"),
    ("mixamorig:LeftHand", "LeftHand"),
    ("mixamorig1:Neck", "Neck"),
    # HumanIK = canonical Mixamo. Already-canonical inputs pass through.
    ("LeftArm", "LeftArm"),
    ("LeftForeArm", "LeftForeArm"),
    ("RightShoulder", "RightShoulder"),
])
def test_mixamo_humanik_already_canonical(name, expected):
    assert auto_detect_bone_role(name) == expected


@pytest.mark.parametrize("name,expected", [
    # Unity Mecanim canonical avatar names.
    ("Hips", "Hips"),
    ("Spine", "Spine"),
    ("Chest", "Spine1"),
    ("UpperChest", "Spine2"),
    ("LeftUpperArm", "LeftArm"),
    ("LeftLowerArm", "LeftForeArm"),
    ("LeftUpperLeg", "LeftUpLeg"),
    ("LeftLowerLeg", "LeftLeg"),
    ("RightFoot", "RightFoot"),
])
def test_unity_mecanim(name, expected):
    assert auto_detect_bone_role(name) == expected


@pytest.mark.parametrize("name,expected", [
    # VRM lowercase camelCase humanoid roles.
    ("hips", "Hips"),
    ("spine", "Spine"),
    ("chest", "Spine1"),
    ("upperChest", "Spine2"),
    ("leftShoulder", "LeftShoulder"),
    ("leftUpperArm", "LeftArm"),
    ("leftLowerArm", "LeftForeArm"),
    ("leftHand", "LeftHand"),
    ("rightUpperLeg", "RightUpLeg"),
    ("rightLowerLeg", "RightLeg"),
    ("rightFoot", "RightFoot"),
])
def test_vrm_lowercase(name, expected):
    assert auto_detect_bone_role(name) == expected


@pytest.mark.parametrize("name,expected", [
    # Cesium / Khronos sample-asset patterns.
    ("Skeleton_arm_joint_L", "LeftShoulder"),
    ("Skeleton_arm_joint_L__2_", "LeftArm"),
    ("Skeleton_arm_joint_L__3_", "LeftForeArm"),
    ("Skeleton_arm_joint_L__4_", "LeftHand"),
    ("Skeleton_arm_joint_R", "RightShoulder"),
    ("Skeleton_arm_joint_R__2_", "RightArm"),
    ("Skeleton_arm_joint_R__3_", "RightForeArm"),
    ("Skeleton_arm_joint_R__4_", "RightHand"),
    ("leg_joint_L_1", "LeftUpLeg"),
    ("leg_joint_L_2", "LeftLeg"),
    ("leg_joint_R_1", "RightUpLeg"),
    ("leg_joint_R_2", "RightLeg"),
    ("leg_joint_R_5", "RightFoot"),
    ("Skeleton_torso_joint_1", "Hips"),
    ("Skeleton_torso_joint_2", "Spine"),
    ("torso_joint_3", "Spine1"),
    ("Skeleton_neck_joint_1", "Neck"),
    ("Skeleton_neck_joint_2", "Head"),
])
def test_cesium_khronos(name, expected):
    assert auto_detect_bone_role(name) == expected


@pytest.mark.parametrize("name,expected", [
    # Blender Rigify dot-suffix conventions.
    ("upper_arm.L", "LeftArm"),
    ("upper_arm.R", "RightArm"),
    ("forearm.L", "LeftForeArm"),
    ("forearm.R", "RightForeArm"),
    ("hand.L", "LeftHand"),
    ("hand.R", "RightHand"),
    ("thigh.L", "LeftUpLeg"),
    ("thigh.R", "RightUpLeg"),
    ("shin.L", "LeftLeg"),
    ("foot.L", "LeftFoot"),
    ("shoulder.L", "LeftShoulder"),
    # Bone.L / Bone.R + numeric index variants.
    ("upper_arm.L.001", "LeftArm"),  # numeric suffix stripped
])
def test_blender_rigify_dot_suffix(name, expected):
    assert auto_detect_bone_role(name) == expected


@pytest.mark.parametrize("name,expected", [
    # MakeHuman / generic underscore-suffix.
    ("upperarm_l", "LeftArm"),
    ("upperarm_r", "RightArm"),
    ("UpperArm_L", "LeftArm"),
    ("UpperArm_R", "RightArm"),
    ("forearm_l", "LeftForeArm"),
    ("hand_l", "LeftHand"),
    ("thigh_l", "LeftUpLeg"),
    ("shin_l", "LeftLeg"),
    ("calf_l", "LeftLeg"),
    ("foot_l", "LeftFoot"),
    ("shoulder_l", "LeftShoulder"),
    ("clavicle_l", "LeftShoulder"),
    ("ankle_l", "LeftFoot"),
    ("knee_l", "LeftLeg"),
])
def test_makehuman_underscore_suffix(name, expected):
    assert auto_detect_bone_role(name) == expected


@pytest.mark.parametrize("name,expected", [
    # Generic free-form / case-insensitive substring matches.
    ("Left Arm", "LeftArm"),
    ("right shoulder", "RightShoulder"),
    ("LEFT_ARM", "LeftArm"),
    ("rightArm", "RightArm"),
    ("Right.Hand", "RightHand"),
    ("torso", "Spine"),
    ("Pelvis", "Hips"),
    ("WAIST", "Hips"),
    ("HEAD", "Head"),
    ("Left Foot", "LeftFoot"),
    ("right_calf", "RightLeg"),
    ("LeftThigh", "LeftUpLeg"),
])
def test_generic_substring(name, expected):
    assert auto_detect_bone_role(name) == expected


@pytest.mark.parametrize("name", [
    "",
    "RandomBoneName",
    "ZZZ_unmapped",
    "Camera",
    "Light_Spot",
    "ConstraintHelper",
    "ROOT_DUMMY_INVISIBLE",
    "weapon_attach_socket",
])
def test_unknown_returns_none(name):
    """Names that don't match any heuristic return None (not a wrong guess)."""
    assert auto_detect_bone_role(name) is None


def test_specific_takes_precedence_over_generic():
    """``LeftForeArm`` resolves to ForeArm even though it contains ``LeftArm``.

    The heuristic must order specific keywords before broader ones.
    """
    # If the generic-substring path were greedy, "LeftForeArm" would
    # match "leftarm" and return LeftArm. The keyword table puts
    # "leftforearm" first to prevent this.
    assert auto_detect_bone_role("LeftForeArm") == "LeftForeArm"
    assert auto_detect_bone_role("LeftLowerArm") == "LeftForeArm"
    assert auto_detect_bone_role("rightforearm") == "RightForeArm"


def test_normalisation_handles_prefixes_and_suffixes():
    """The heuristic strips Mixamo/Armature/rig prefixes before matching."""
    assert auto_detect_bone_role("Armature_LeftArm") == "LeftArm"
    assert auto_detect_bone_role("rig_RightFoot") == "RightFoot"
    assert auto_detect_bone_role("mixamorig:LeftHand") == "LeftHand"


# ---------------------------------------------------------------------------
# End-to-end retarget routing tests
# ---------------------------------------------------------------------------


def _make_retarget_test_anim(
    src_skeleton: List[ImportedBone],
    animated_bone_idx: int,
    *,
    n_frames: int = 15,
) -> ImportedAnimation:
    """Build a simple Y-rotation sweep animation on one source bone."""
    times = [f / 30.0 for f in range(n_frames)]
    values = [
        (0.0,
         math.sin(math.pi * 0.5 * f / max(1, n_frames - 1) * 0.5),
         0.0,
         math.cos(math.pi * 0.5 * f / max(1, n_frames - 1) * 0.5))
        for f in range(n_frames)
    ]
    return ImportedAnimation(
        name="autodetect_test",
        duration_seconds=times[-1],
        fps_target=30,
        tracks=[ImportedTrack(
            bone_idx=animated_bone_idx,
            channel="rotation",
            times=times,
            values=values,
        )],
    )


def _make_target_skeleton_for_lobby() -> List[ImportedBone]:
    """Return a synthetic target skeleton large enough to host LOBBY_GIRL_BONE_MAP."""
    n_target = max(LOBBY_GIRL_BONE_MAP.values()) + 1
    return [
        ImportedBone(name=f"tgt{i}", parent_idx=-1 if i == 0 else 0,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1))
        for i in range(n_target)
    ]


def test_mecanim_skeleton_retargets_via_autodetect():
    """A Unity Mecanim source rig retargets cleanly with NO explicit aliases."""
    src_skel = [
        ImportedBone(name="Hips", parent_idx=-1,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="Spine", parent_idx=0,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="LeftUpperArm", parent_idx=1,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="LeftLowerArm", parent_idx=2,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
    ]
    anim = _make_retarget_test_anim(src_skel, animated_bone_idx=2)
    motion = retarget_animation(
        anim, src_skel, _make_target_skeleton_for_lobby(),
        dict(LOBBY_GIRL_BONE_MAP),
        target_fps=30, flip_z=False, enable_ik=False,
    )
    summary = summarize_retarget(motion)
    # Hips/Spine match directly; LeftUpperArm + LeftLowerArm via auto-detect.
    assert summary["mapped_bones"] == 4
    assert summary["dropped_bones"] == 0
    res = summary["resolution"]
    # Hips + Spine are direct matches (LOBBY_GIRL_BONE_MAP carries them).
    # LeftUpperArm + LeftLowerArm need auto-detect to recover.
    assert res["auto_detected"] >= 2
    assert res["direct_mapped"] >= 2


def test_blender_rigify_skeleton_retargets_via_autodetect():
    """A Blender Rigify source rig retargets cleanly via .L/.R aliases."""
    src_skel = [
        ImportedBone(name="hips", parent_idx=-1,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="spine", parent_idx=0,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="upper_arm.L", parent_idx=1,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="forearm.L", parent_idx=2,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="hand.L", parent_idx=3,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="upper_arm.R", parent_idx=1,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="thigh.L", parent_idx=0,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="shin.L", parent_idx=6,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="foot.L", parent_idx=7,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
    ]
    anim = _make_retarget_test_anim(src_skel, animated_bone_idx=2)
    motion = retarget_animation(
        anim, src_skel, _make_target_skeleton_for_lobby(),
        dict(LOBBY_GIRL_BONE_MAP),
        target_fps=30, flip_z=False, enable_ik=False,
    )
    summary = summarize_retarget(motion)
    assert summary["mapped_bones"] == 9
    assert summary["dropped_bones"] == 0


def test_makehuman_skeleton_retargets_via_autodetect():
    """A MakeHuman-style underscore-suffix source rig retargets correctly."""
    src_skel = [
        ImportedBone(name="root", parent_idx=-1,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="upperarm_l", parent_idx=0,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="lowerarm_l", parent_idx=1,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="hand_l", parent_idx=2,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
    ]
    anim = _make_retarget_test_anim(src_skel, animated_bone_idx=1)
    motion = retarget_animation(
        anim, src_skel, _make_target_skeleton_for_lobby(),
        dict(LOBBY_GIRL_BONE_MAP),
        target_fps=30, flip_z=False, enable_ik=False,
    )
    summary = summarize_retarget(motion)
    # 'root' isn't in any map → 1 dropped; the 3 left-arm bones map.
    assert summary["mapped_bones"] == 3
    assert any("root" in d for d in summary["dropped"])
    assert summary["resolution"]["auto_detected"] == 3


def test_autodetect_disabled_drops_unknown_names():
    """With auto-detect disabled, source bones not in the map drop."""
    src_skel = [
        ImportedBone(name="LeftUpperArm", parent_idx=-1,  # Mecanim, not in map
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
    ]
    src_anim = ImportedAnimation(
        name="d", duration_seconds=0.0, fps_target=30, tracks=[],
    )
    tgt_skel = _make_target_skeleton_for_lobby()
    motion = retarget_animation(
        src_anim, src_skel, tgt_skel,
        # bone_map only has Mixamo names — Mecanim names need auto-detect.
        {"LeftArm": 5},
        target_fps=30, flip_z=False, enable_ik=False,
        enable_auto_detect=False,
    )
    summary = summarize_retarget(motion)
    assert summary["mapped_bones"] == 0
    assert summary["dropped_bones"] == 1


def test_autodetect_does_not_break_explicit_aliases():
    """LOBBY_GIRL_BONE_MAP's explicit CesiumMan aliases still work.

    Regression: the auto-detect path runs BEFORE explicit aliases would
    be needed (since the explicit ones already give a direct hit), so
    enabling the heuristic shouldn't change behaviour for assets that
    were already correctly aliased.
    """
    src_skel = [
        ImportedBone(name="Skeleton_arm_joint_R__2_", parent_idx=-1,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="leg_joint_L_5", parent_idx=-1,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
    ]
    anim = ImportedAnimation(name="d", duration_seconds=0.0, fps_target=30, tracks=[])
    tgt_skel = _make_target_skeleton_for_lobby()
    motion = retarget_animation(
        anim, src_skel, tgt_skel,
        dict(LOBBY_GIRL_BONE_MAP),
        target_fps=30, flip_z=False, enable_ik=False,
    )
    summary = summarize_retarget(motion)
    # Both should resolve via direct map (the explicit alias entries),
    # not via auto-detect (which would also resolve them).
    assert summary["mapped_bones"] == 2
    # direct_mapped >= 2 because the explicit aliases land first.
    assert summary["resolution"]["direct_mapped"] == 2


def test_resolution_precedence_explicit_over_autodetect():
    """When a name is in BOTH bone_map AND auto-detect, the explicit map wins.

    This ensures power users who add a custom alias to bone_map can
    override the heuristic — important for skeletons where the
    convention guess produces a wrong route.
    """
    src_skel = [
        ImportedBone(name="LeftUpperArm", parent_idx=-1,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
    ]
    anim = ImportedAnimation(name="d", duration_seconds=0.0, fps_target=30, tracks=[])
    tgt_skel = _make_target_skeleton_for_lobby()
    # Explicit map: route LeftUpperArm to bone 99 (different from what
    # the heuristic would produce — heuristic says LeftArm = bone 4 in
    # LOBBY_GIRL_BONE_MAP). The explicit entry must win.
    explicit_map = {"LeftUpperArm": 7, **dict(LOBBY_GIRL_BONE_MAP)}
    motion = retarget_animation(
        anim, src_skel, tgt_skel, explicit_map,
        target_fps=30, flip_z=False, enable_ik=False,
    )
    summary = summarize_retarget(motion)
    # Verify the explicit override took: target index 7, not 4 (LeftArm).
    assert summary["mapping"][0]["src"] == 0
    assert summary["mapping"][0]["tgt"] == 7
    assert summary["resolution"]["direct_mapped"] == 1
    assert summary["resolution"]["auto_detected"] == 0


def test_cesium_man_regression_still_passes():
    """CesiumMan-style joint names continue to retarget cleanly.

    The 17 explicit aliases in LOBBY_GIRL_BONE_MAP for Khronos sample
    assets remain present, so the regression pinned by the 2026-04-25
    audit doesn't reappear with the new heuristic in play.
    """
    cesium_keys = (
        "Skeleton_torso_joint_1",
        "Skeleton_torso_joint_2",
        "Skeleton_arm_joint_R__2_",
        "Skeleton_arm_joint_L__4_",
        "leg_joint_R_1",
        "leg_joint_L_5",
    )
    for k in cesium_keys:
        assert k in LOBBY_GIRL_BONE_MAP, f"CesiumMan alias {k} regressed"
    # And auto_detect_bone_role recognises them too (so dropping the
    # explicit entry would still leave the rig retargeting via the
    # heuristic — defence in depth).
    assert auto_detect_bone_role("Skeleton_arm_joint_L__2_") == "LeftArm"
    assert auto_detect_bone_role("leg_joint_R_1") == "RightUpLeg"
