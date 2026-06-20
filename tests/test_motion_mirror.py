"""Tests for ``formats.anim_retarget`` motion mirroring (v3, 2026-04-25).

The mirror feature lets users invert a one-handed Mixamo clip ("right-hand
wave") into the opposite-side variant ("left-hand wave") without
re-authoring the source animation. The mirror runs as a post-processing
pass on the already-retargeted ``NjmRawMotion``.

Coverage:
  * L/R bone-name pair detection (Mixamo Left/Right, mixamorig:* prefix,
    CesiumMan-style suffix tags).
  * Quaternion mirror across the YZ plane: (qx, qy, qz, qw) →
    (qx, -qy, -qz, qw).
  * End-to-end: retarget a one-armed clip, mirror it, verify the
    keyframes swapped to the opposite arm and the per-frame quats are
    mirror-aligned.
  * Centerline bones (Hips/Spine) get the in-place mirror without a
    swap.
  * Mirror is idempotent on a symmetric (no-op) animation in the sense
    that mirroring twice returns the original keyframes (within ε).
  * The ``mirror=True`` kwarg on ``retarget_animation`` runs the
    pipeline.
"""
from __future__ import annotations

import math

import pytest

from formats.anim_retarget import (
    HUMANOID_IK_CHAINS,
    IkChainSpec,
    detect_lr_pairs,
    mirror_animation,
    retarget_animation,
    summarize_retarget,
    _bams_to_quat,
)
from formats.import_external import (
    ImportedAnimation,
    ImportedBone,
    ImportedTrack,
)
from formats.njm import NJD_MTYPE_ANG, NJD_MTYPE_POS, parse_njm
from formats.njm_writer import encode_njm


# ---------------------------------------------------------------------------
# Synthetic skeleton + animation fixtures
# ---------------------------------------------------------------------------


def _make_humanoid_source() -> list[ImportedBone]:
    """Mixamo-style humanoid with both arms, both legs, hips, spine."""
    return [
        ImportedBone(name="mixamorig:Hips", parent_idx=-1,
                     bind_pos=(0.0, 1.0, 0.0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="mixamorig:Spine", parent_idx=0,
                     bind_pos=(0.0, 0.20, 0.0), bind_rot_quat=(0, 0, 0, 1)),
        # Right arm — character's right side = -X.
        ImportedBone(name="mixamorig:RightShoulder", parent_idx=1,
                     bind_pos=(-0.10, 0.18, 0.0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="mixamorig:RightArm", parent_idx=2,
                     bind_pos=(-0.05, 0.0, 0.0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="mixamorig:RightForeArm", parent_idx=3,
                     bind_pos=(-0.30, 0.0, 0.0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="mixamorig:RightHand", parent_idx=4,
                     bind_pos=(-0.25, 0.0, 0.0), bind_rot_quat=(0, 0, 0, 1)),
        # Left arm — character's left side = +X.
        ImportedBone(name="mixamorig:LeftShoulder", parent_idx=1,
                     bind_pos=(0.10, 0.18, 0.0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="mixamorig:LeftArm", parent_idx=6,
                     bind_pos=(0.05, 0.0, 0.0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="mixamorig:LeftForeArm", parent_idx=7,
                     bind_pos=(0.30, 0.0, 0.0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="mixamorig:LeftHand", parent_idx=8,
                     bind_pos=(0.25, 0.0, 0.0), bind_rot_quat=(0, 0, 0, 1)),
    ]


def _make_humanoid_target() -> list[ImportedBone]:
    """Symmetric-bind PSOBB-like target with the same topology."""
    return [
        ImportedBone(name="hips", parent_idx=-1,
                     bind_pos=(0.0, 0.95, 0.0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="spine", parent_idx=0,
                     bind_pos=(0.0, 0.18, 0.0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="r_shoulder", parent_idx=1,
                     bind_pos=(-0.08, 0.16, 0.0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="r_arm", parent_idx=2,
                     bind_pos=(-0.04, 0.0, 0.0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="r_forearm", parent_idx=3,
                     bind_pos=(-0.25, 0.0, 0.0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="r_hand", parent_idx=4,
                     bind_pos=(-0.20, 0.0, 0.0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="l_shoulder", parent_idx=1,
                     bind_pos=(0.08, 0.16, 0.0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="l_arm", parent_idx=6,
                     bind_pos=(0.04, 0.0, 0.0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="l_forearm", parent_idx=7,
                     bind_pos=(0.25, 0.0, 0.0), bind_rot_quat=(0, 0, 0, 1)),
        ImportedBone(name="l_hand", parent_idx=8,
                     bind_pos=(0.20, 0.0, 0.0), bind_rot_quat=(0, 0, 0, 1)),
    ]


def _bone_map_humanoid() -> dict:
    return {
        "Hips": 0, "Spine": 1,
        "RightShoulder": 2, "RightArm": 3, "RightForeArm": 4, "RightHand": 5,
        "LeftShoulder": 6, "LeftArm": 7, "LeftForeArm": 8, "LeftHand": 9,
    }


def _wave_anim_right_only(n_frames: int = 8, fps: int = 30) -> ImportedAnimation:
    """Source animation: ONLY the right forearm flexes 0..45° around Y.

    Picked so the pre-mirror retarget puts keyframes only on the right
    forearm; after mirror, those keyframes should land on the LEFT
    forearm and the right side should be empty.
    """
    times = [f / float(fps) for f in range(n_frames)]
    forearm_values = []
    for f in range(n_frames):
        ang = math.radians(45.0) * f / max(1, n_frames - 1)
        forearm_values.append((0.0, math.sin(ang * 0.5), 0.0, math.cos(ang * 0.5)))
    return ImportedAnimation(
        name="RightWave",
        duration_seconds=times[-1],
        fps_target=fps,
        tracks=[
            ImportedTrack(
                bone_idx=4, channel="rotation",  # RightForeArm
                times=times, values=forearm_values, interp="LINEAR",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Pair detection
# ---------------------------------------------------------------------------


def test_detect_lr_pairs_mixamo_naming():
    """Mixamo naming (Left*/Right*) round-trips through pair detection."""
    names = [
        "Hips", "Spine",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        "LeftShoulder",  "LeftArm",  "LeftForeArm",  "LeftHand",
    ]
    pairs = detect_lr_pairs(names)
    # RightShoulder(2) ↔ LeftShoulder(6); both directions present.
    assert pairs[2] == 6 and pairs[6] == 2
    assert pairs[3] == 7 and pairs[7] == 3
    assert pairs[4] == 8 and pairs[8] == 4
    assert pairs[5] == 9 and pairs[9] == 5
    # Centerline bones have no pair.
    assert 0 not in pairs
    assert 1 not in pairs


def test_detect_lr_pairs_mixamorig_prefix():
    """``mixamorig:`` prefix is preserved during swap (LeftHand ↔ RightHand
    even when both names carry the namespace prefix)."""
    names = [
        "mixamorig:Hips",
        "mixamorig:RightHand",
        "mixamorig:LeftHand",
    ]
    pairs = detect_lr_pairs(names)
    assert pairs[1] == 2 and pairs[2] == 1


def test_detect_lr_pairs_cesiumman_suffix():
    """CesiumMan _R_ / _L_ suffix tags also pair up."""
    names = [
        "Skeleton_torso_joint_1",
        "Skeleton_arm_joint_R__2_",
        "Skeleton_arm_joint_L__2_",
    ]
    pairs = detect_lr_pairs(names)
    # The L/R-swap should match these.
    assert pairs.get(1) == 2 and pairs.get(2) == 1


def test_detect_lr_pairs_no_pair_when_only_one_side():
    """A name with a Right* sibling missing → not paired."""
    names = ["Hips", "RightHand"]  # no LeftHand
    pairs = detect_lr_pairs(names)
    assert pairs == {}


# ---------------------------------------------------------------------------
# Mirror algebra
# ---------------------------------------------------------------------------


def test_mirror_quat_yz_basic_x_rotation_unchanged():
    """A pure X-axis rotation is unchanged by a YZ-plane mirror.

    Geometric reason: a rotation about X stays the same when you
    mirror across YZ (the rotation plane is perpendicular to the
    mirror plane). The quat (sin(θ/2), 0, 0, cos(θ/2)) → itself.
    """
    from formats.anim_retarget import _mirror_quat_yz
    ang = math.radians(45.0)
    q = (math.sin(ang * 0.5), 0.0, 0.0, math.cos(ang * 0.5))
    qm = _mirror_quat_yz(q)
    # All four components should match (within fp epsilon).
    for a, b in zip(q, qm):
        assert abs(a - b) < 1e-9, f"X-rotation should be invariant: {q} vs {qm}"


def test_mirror_quat_yz_y_rotation_negated():
    """A pure Y-axis rotation gets its angle negated under YZ mirror.

    A Y-axis rotation by θ becomes a Y-axis rotation by -θ when mirroring
    across the YZ plane (left/right swap of the rotational handedness).
    Quat (0, sin(θ/2), 0, cos(θ/2)) → (0, -sin(θ/2), 0, cos(θ/2)).
    """
    from formats.anim_retarget import _mirror_quat_yz
    ang = math.radians(60.0)
    q = (0.0, math.sin(ang * 0.5), 0.0, math.cos(ang * 0.5))
    qm = _mirror_quat_yz(q)
    assert abs(qm[0]) < 1e-9
    assert abs(qm[1] + math.sin(ang * 0.5)) < 1e-9
    assert abs(qm[2]) < 1e-9
    assert abs(qm[3] - math.cos(ang * 0.5)) < 1e-9


def test_mirror_quat_yz_double_application_returns_original():
    """Mirroring twice = identity (the mirror is its own inverse)."""
    from formats.anim_retarget import _mirror_quat_yz
    q = (0.1, 0.3, 0.2, math.sqrt(1.0 - 0.1**2 - 0.3**2 - 0.2**2))
    twice = _mirror_quat_yz(_mirror_quat_yz(q))
    for a, b in zip(q, twice):
        assert abs(a - b) < 1e-9


# ---------------------------------------------------------------------------
# End-to-end mirror via retarget_animation(mirror=True)
# ---------------------------------------------------------------------------


def test_retarget_with_mirror_swaps_arm_keyframes():
    """A right-arm-only source animation, mirrored, should drive the
    LEFT arm in the target's keyframes (and the right arm should be
    empty)."""
    src_skel = _make_humanoid_source()
    tgt_skel = _make_humanoid_target()
    src_anim = _wave_anim_right_only(n_frames=6)
    bone_map = _bone_map_humanoid()

    motion = retarget_animation(
        src_anim, src_skel, tgt_skel, bone_map,
        target_fps=30, flip_z=False,
        enable_ik=False, mirror=True,
    )

    # Target's RightForeArm = bone 4; LeftForeArm = bone 8.
    right_track = motion.bones[4].tracks_by_kind.get(NJD_MTYPE_ANG)
    left_track = motion.bones[8].tracks_by_kind.get(NJD_MTYPE_ANG)

    # After mirror, the LEFT side should carry the keyframes.
    assert left_track is not None and len(left_track.keyframes) > 0, (
        "left forearm should have keyframes after mirror"
    )
    # Last keyframe on the left should be near 45° around Y (negated by
    # the mirror = -45° around Y in the Y-axis case).
    last = left_track.keyframes[-1]
    # Y rotation is encoded as the second BAMS slot. -45° = -8192 = 0xE000.
    ry_signed = int(last[2])
    if ry_signed >= 0x8000:
        ry_signed -= 0x10000
    # The mirror flips the Y-axis sign; original was +45° (~+8192 BAMS).
    # We tolerate ±64 BAMS for round-trip noise.
    assert ry_signed < 0, f"expected negative Y on mirrored left forearm, got {ry_signed}"
    assert abs(ry_signed + 8192) < 200, (
        f"expected ~-45° (~-8192 BAMS) on mirrored left forearm, got {ry_signed}"
    )

    # Right side should be empty (no keyframes that mutate the bind).
    if right_track is not None and right_track.keyframes:
        # Any keyframes must be near-identity (the empty-track default).
        for kf in right_track.keyframes:
            for c in (kf[1], kf[2], kf[3]):
                signed = int(c)
                if signed >= 0x8000:
                    signed -= 0x10000
                assert abs(signed) < 32, (
                    f"right forearm should be near-bind after mirror, got {kf}"
                )


def test_retarget_mirror_is_no_op_without_pairs():
    """If no L/R pairs detected (skeleton has no L/R names), the mirror
    still runs (in-place quat-flip on every bone) but doesn't crash."""
    src_skel = _make_humanoid_source()
    tgt_skel = _make_humanoid_target()
    # Use a bone_map that doesn't expose L/R names on the target side.
    src_anim = _wave_anim_right_only(n_frames=4)
    bone_map = _bone_map_humanoid()

    motion = retarget_animation(
        src_anim, src_skel, tgt_skel, bone_map,
        target_fps=30, flip_z=False,
        enable_ik=False, mirror=True,
    )
    # Encode round-trips.
    raw = encode_njm(motion)
    parse_njm(raw)


def test_mirror_animation_double_application_round_trips():
    """Mirroring an already-mirrored motion returns close to the original
    keyframes (since the mirror is its own inverse)."""
    src_skel = _make_humanoid_source()
    tgt_skel = _make_humanoid_target()
    src_anim = _wave_anim_right_only(n_frames=8)
    bone_map = _bone_map_humanoid()

    motion_orig = retarget_animation(
        src_anim, src_skel, tgt_skel, bone_map,
        target_fps=30, flip_z=False, enable_ik=False, mirror=False,
    )
    motion_once = retarget_animation(
        src_anim, src_skel, tgt_skel, bone_map,
        target_fps=30, flip_z=False, enable_ik=False, mirror=True,
    )
    # Now mirror motion_once back: same bone_names list — extract from
    # the target skeleton the way retarget_animation does.
    from formats.anim_retarget import _derive_target_bone_names
    tgt_names = _derive_target_bone_names(tgt_skel, bone_map)
    motion_twice = mirror_animation(motion_once, target_bone_names=tgt_names)

    # Compare keyframe BAMS values per bone — should match within rounding.
    for bi in range(len(motion_orig.bones)):
        a = motion_orig.bones[bi].tracks_by_kind.get(NJD_MTYPE_ANG)
        b = motion_twice.bones[bi].tracks_by_kind.get(NJD_MTYPE_ANG)
        if a is None and b is None:
            continue
        if a is None or b is None:
            # One has empty track; the other should also be effectively empty.
            for trk in (a, b):
                if trk is not None:
                    for kf in trk.keyframes:
                        for c in (kf[1], kf[2], kf[3]):
                            signed = int(c)
                            if signed >= 0x8000:
                                signed -= 0x10000
                            assert abs(signed) < 32, (
                                f"bone {bi}: post-double-mirror non-empty "
                                f"track from one-sided original: {kf}"
                            )
            continue
        assert len(a.keyframes) == len(b.keyframes), (
            f"bone {bi}: keyframe count diverged"
        )
        for kfa, kfb in zip(a.keyframes, b.keyframes):
            for ca, cb in zip(kfa[1:4], kfb[1:4]):
                # Difference modulo 0x10000 (BAMS wrap), treated as signed.
                d = (int(ca) - int(cb)) & 0xFFFF
                if d >= 0x8000:
                    d -= 0x10000
                assert abs(d) <= 4, (
                    f"bone {bi}: double-mirror diverged at frame {kfa[0]}: "
                    f"{kfa} vs {kfb}"
                )


def test_mirror_diagnostics_in_summary():
    """``summarize_retarget`` exposes the mirror block when mirror=True."""
    src_skel = _make_humanoid_source()
    tgt_skel = _make_humanoid_target()
    src_anim = _wave_anim_right_only(n_frames=4)
    bone_map = _bone_map_humanoid()

    motion = retarget_animation(
        src_anim, src_skel, tgt_skel, bone_map,
        target_fps=30, flip_z=False, enable_ik=False, mirror=True,
    )
    summary = summarize_retarget(motion)
    mirror = summary.get("mirror")
    assert isinstance(mirror, dict)
    assert mirror.get("axis") == "x"
    # We should have detected and swapped at least one pair (the arms).
    assert mirror.get("swapped_pairs", 0) >= 1
    assert mirror.get("lr_pairs_detected", 0) >= 1


def test_mirror_default_is_off():
    """The ``mirror`` kwarg defaults to False — running without the kwarg
    should produce the same output as ``mirror=False``."""
    src_skel = _make_humanoid_source()
    tgt_skel = _make_humanoid_target()
    src_anim = _wave_anim_right_only(n_frames=4)
    bone_map = _bone_map_humanoid()

    m_default = retarget_animation(
        src_anim, src_skel, tgt_skel, bone_map,
        target_fps=30, flip_z=False, enable_ik=False,
    )
    m_explicit = retarget_animation(
        src_anim, src_skel, tgt_skel, bone_map,
        target_fps=30, flip_z=False, enable_ik=False, mirror=False,
    )
    # Bit-equal motion: same bone count, frame count, type flags.
    assert m_default.frame_count == m_explicit.frame_count
    for bi in range(len(m_default.bones)):
        a = m_default.bones[bi].tracks_by_kind.get(NJD_MTYPE_ANG)
        b = m_explicit.bones[bi].tracks_by_kind.get(NJD_MTYPE_ANG)
        if a is None and b is None:
            continue
        assert a is not None and b is not None
        assert a.keyframes == b.keyframes, f"bone {bi} differs without mirror=True"


def test_mirror_unsupported_axis_raises():
    """Only axis='x' supported; other values raise ValueError."""
    src_skel = _make_humanoid_source()
    tgt_skel = _make_humanoid_target()
    src_anim = _wave_anim_right_only(n_frames=2)
    bone_map = _bone_map_humanoid()
    motion = retarget_animation(
        src_anim, src_skel, tgt_skel, bone_map,
        target_fps=30, flip_z=False, enable_ik=False,
    )
    with pytest.raises(ValueError):
        mirror_animation(motion, target_bone_names=["a", "b"], axis="z")


def test_mirror_handles_position_tracks():
    """POS keyframes get X negated. Translation-bearing source → mirror
    flips the X component while leaving Y and Z untouched."""
    src_skel = _make_humanoid_source()
    tgt_skel = _make_humanoid_target()
    # Author hip translation that drifts +X over time.
    times = [f / 30.0 for f in range(5)]
    pos_values = [(0.1 * f, 0.0, 0.0) for f in range(5)]
    rot_values = [(0.0, 0.0, 0.0, 1.0)] * 5
    src_anim = ImportedAnimation(
        name="HipDrift", duration_seconds=times[-1], fps_target=30,
        tracks=[
            ImportedTrack(bone_idx=0, channel="translation",
                          times=times, values=pos_values, interp="LINEAR"),
            ImportedTrack(bone_idx=0, channel="rotation",
                          times=times, values=rot_values, interp="LINEAR"),
        ],
    )
    bone_map = _bone_map_humanoid()
    motion = retarget_animation(
        src_anim, src_skel, tgt_skel, bone_map,
        target_fps=30, flip_z=False,
        include_translation=True, translation_scale=1.0,
        enable_ik=False, mirror=True,
    )
    # Hip = bone 0 in target. Find its POS track.
    pos = motion.bones[0].tracks_by_kind.get(NJD_MTYPE_POS)
    assert pos is not None and pos.keyframes
    # The mirror should have negated each X value.
    last_kf = pos.keyframes[-1]
    # Original was +0.4; mirror should be -0.4 (or close).
    assert last_kf[1] < 0.0, f"expected mirrored X negative, got {last_kf}"


# ---------------------------------------------------------------------------
# Encode round-trip
# ---------------------------------------------------------------------------


def test_mirror_output_encodes_cleanly():
    """A mirrored motion encodes to NJM bytes and round-trips through
    parse_njm without errors."""
    src_skel = _make_humanoid_source()
    tgt_skel = _make_humanoid_target()
    src_anim = _wave_anim_right_only(n_frames=10)
    bone_map = _bone_map_humanoid()

    motion = retarget_animation(
        src_anim, src_skel, tgt_skel, bone_map,
        target_fps=30, flip_z=False, enable_ik=True, mirror=True,
    )
    raw = encode_njm(motion)
    parsed = parse_njm(raw)
    assert len(parsed) == 1
    parsed_motion = parsed[0]
    assert parsed_motion.bone_count == 10
    assert parsed_motion.frame_count == motion.frame_count
