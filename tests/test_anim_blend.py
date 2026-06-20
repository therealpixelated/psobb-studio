"""Tests for ``formats.anim_blend`` — runtime motion blending.

Coverage:
  * Two synthetic 1-bone motions: 0° rotation and 90° rotation around Z.
    Blend at 50/50 → result should be 45° rotation per frame.
  * 75/25 weighting tilts toward the 75% source.
  * Frame-count resampling: blend a 10-frame source against a 30-frame
    source at uniform 50/50 → output has 30 frames; per-output-frame
    sampling uses each source's own frame index proportionally.
  * Transition curves: smoothstep blend over time produces non-linear
    weight ramps (start ≈ source0, mid ≈ midpoint, end ≈ source1).
  * Bone-count mismatch raises ValueError.
  * Empty motions list raises ValueError.
  * Negative weights are clamped to 0; all-zero weights → uniform.
  * Round-trip: encode the blended motion to bytes and re-parse;
    bone/frame counts survive.
"""
from __future__ import annotations

import math

import pytest

from formats.anim_blend import (
    BlendNode,
    TRANSITION_LINEAR,
    TRANSITION_SMOOTH,
    VALID_TRANSITIONS,
    blend_from_node,
    blend_motions,
    summarize_blend,
)
from formats.njm import NJD_MTYPE_ANG, NJD_MTYPE_POS, parse_njm
from formats.njm_writer import (
    NjmBoneTracks,
    NjmRawMotion,
    NjmTrack,
    encode_njm,
)


# ---------------------------------------------------------------------------
# Synthetic motion fixtures
# ---------------------------------------------------------------------------


def _make_constant_rotation_motion(
    bone_count: int,
    frame_count: int,
    *,
    bone_idx: int = 0,
    rx_bams: int = 0,
    ry_bams: int = 0,
    rz_bams: int = 0,
) -> NjmRawMotion:
    """Build a motion where one bone has a constant rotation across all frames.

    Other bones get an empty ANG slot (= bind pose at runtime).
    """
    bones = []
    for bi in range(bone_count):
        bone = NjmBoneTracks()
        if bi == bone_idx:
            kfs = [(f, rx_bams, ry_bams, rz_bams) for f in range(frame_count)]
        else:
            kfs = []
        bone.tracks_by_kind[NJD_MTYPE_ANG] = NjmTrack(
            kind=NJD_MTYPE_ANG, keyframes=kfs, narrow=True,
        )
        bones.append(bone)
    motion = NjmRawMotion(
        frame_count=frame_count,
        type_flags=NJD_MTYPE_ANG,
        inp_fn=1,
        m_data_table_offset=0xC,
        bones=bones,
    )
    return motion


def _make_swept_rotation_motion(
    bone_count: int,
    frame_count: int,
    *,
    bone_idx: int = 0,
    start_rz: int = 0,
    end_rz: int = 0x4000,
) -> NjmRawMotion:
    """Build a motion sweeping Z rotation from ``start_rz`` to ``end_rz``."""
    bones = []
    for bi in range(bone_count):
        bone = NjmBoneTracks()
        if bi == bone_idx:
            kfs = []
            for f in range(frame_count):
                t = f / max(1, frame_count - 1)
                rz = int(round(start_rz + t * (end_rz - start_rz))) & 0xFFFF
                kfs.append((f, 0, 0, rz))
        else:
            kfs = []
        bone.tracks_by_kind[NJD_MTYPE_ANG] = NjmTrack(
            kind=NJD_MTYPE_ANG, keyframes=kfs, narrow=True,
        )
        bones.append(bone)
    return NjmRawMotion(
        frame_count=frame_count,
        type_flags=NJD_MTYPE_ANG,
        inp_fn=1,
        m_data_table_offset=0xC,
        bones=bones,
    )


# ---------------------------------------------------------------------------
# Blend math correctness
# ---------------------------------------------------------------------------


def test_blend_two_constant_rotations_50_50_yields_midpoint():
    """0° + 90° at 50/50 → 45° rotation per frame."""
    m0 = _make_constant_rotation_motion(bone_count=2, frame_count=5,
                                          bone_idx=0, rz_bams=0)
    m1 = _make_constant_rotation_motion(bone_count=2, frame_count=5,
                                          bone_idx=0, rz_bams=0x4000)  # 90°
    blended = blend_motions([m0, m1], [0.5, 0.5])
    assert blended.frame_count == 5
    assert len(blended.bones) == 2
    track = blended.bones[0].tracks_by_kind[NJD_MTYPE_ANG]
    assert len(track.keyframes) == 5
    # 45° = 0x2000 BAMS, ±2 LSB tolerance for slerp rounding.
    for kf in track.keyframes:
        assert abs(int(kf[3]) - 0x2000) <= 4, f"got rz={kf[3]:#06x} on frame {kf[0]}"


def test_blend_75_25_tilts_toward_higher_weight():
    """75/25 of 0°/90° → ~22.5° (closer to 0° source)."""
    m0 = _make_constant_rotation_motion(bone_count=1, frame_count=3,
                                          bone_idx=0, rz_bams=0)
    m1 = _make_constant_rotation_motion(bone_count=1, frame_count=3,
                                          bone_idx=0, rz_bams=0x4000)
    blended = blend_motions([m0, m1], [0.75, 0.25])
    track = blended.bones[0].tracks_by_kind[NJD_MTYPE_ANG]
    # 0.25 * 90° = 22.5° → 0x4000 / 4 = 0x1000.
    for kf in track.keyframes:
        assert abs(int(kf[3]) - 0x1000) <= 8, f"got rz={kf[3]:#06x}"


def test_blend_three_way_uniform_yields_average():
    """3-way uniform blend of 0° / 90° / 180° → 90° (centre of arc)."""
    m0 = _make_constant_rotation_motion(1, 3, rz_bams=0)
    m1 = _make_constant_rotation_motion(1, 3, rz_bams=0x4000)
    m2 = _make_constant_rotation_motion(1, 3, rz_bams=0x8000)  # 180°
    blended = blend_motions([m0, m1, m2], [1.0, 1.0, 1.0])
    track = blended.bones[0].tracks_by_kind[NJD_MTYPE_ANG]
    # Pairwise slerp(slerp(0, 90, 0.5), 180, 1/3):
    #   step1 = slerp(0, 90, 0.5) = 45°
    #   step2 = slerp(45, 180, 1/3) = 45 + (180-45)/3 = 90°
    for kf in track.keyframes:
        assert abs(int(kf[3]) - 0x4000) <= 16, f"got rz={kf[3]:#06x}"


def test_blend_resamples_different_frame_counts():
    """Blend of a 10-frame and 30-frame source produces 30-frame output;
    each output frame samples each source proportionally."""
    short_m = _make_swept_rotation_motion(1, 10, start_rz=0, end_rz=0x4000)
    long_m = _make_swept_rotation_motion(1, 30, start_rz=0, end_rz=0)
    blended = blend_motions([short_m, long_m], [1.0, 0.0])
    # All-weight on short_m; output should match short_m's sweep
    # resampled to 30 frames.
    assert blended.frame_count == 30
    track = blended.bones[0].tracks_by_kind[NJD_MTYPE_ANG]
    # Frame 0 → 0°, frame 29 → 90°.
    assert abs(int(track.keyframes[0][3]) - 0) <= 4
    assert abs(int(track.keyframes[-1][3]) - 0x4000) <= 4
    # Frame 14 of 30: maps to short_m frame 14 * 9 / 29 ≈ 4.345 → 43.45°.
    # 43.45 / 90 * 0x4000 ≈ 7912.
    mid = track.keyframes[14]
    expected = int(round(14 * 9 / 29 / 9 * 0x4000))
    assert abs(int(mid[3]) - expected) <= 32, (
        f"got mid={mid[3]:#06x}, expected ~{expected:#06x}"
    )


def test_blend_explicit_frame_count_overrides():
    """When ``frame_count`` is supplied, output uses that exact length."""
    m0 = _make_swept_rotation_motion(1, 5, end_rz=0x4000)
    m1 = _make_swept_rotation_motion(1, 7, end_rz=0x4000)
    blended = blend_motions([m0, m1], [0.5, 0.5], frame_count=12)
    assert blended.frame_count == 12
    track = blended.bones[0].tracks_by_kind[NJD_MTYPE_ANG]
    assert len(track.keyframes) == 12


# ---------------------------------------------------------------------------
# Transition curves
# ---------------------------------------------------------------------------


def test_smoothstep_curve_ramps_weights_over_duration():
    """SMOOTH curve: frame 0 ≈ source 0, last frame ≈ source 1, midframe ≈ midpoint."""
    m0 = _make_constant_rotation_motion(1, 11, rz_bams=0)
    m1 = _make_constant_rotation_motion(1, 11, rz_bams=0x4000)
    blended = blend_motions(
        [m0, m1], [0.5, 0.5], transition_curve=TRANSITION_SMOOTH,
    )
    track = blended.bones[0].tracks_by_kind[NJD_MTYPE_ANG]
    # Frame 0: pure source 0 → 0°.
    assert abs(int(track.keyframes[0][3]) - 0) <= 8
    # Frame 10 (last): pure source 1 → 90°.
    assert abs(int(track.keyframes[10][3]) - 0x4000) <= 8
    # Frame 5 (mid): smoothstep(0.5) = 0.5 → 45°.
    mid = track.keyframes[5]
    assert abs(int(mid[3]) - 0x2000) <= 16, f"got mid={mid[3]:#06x}"
    # Frame 2 (early): smoothstep(0.2) ≈ 0.104 → ~9.4°.
    early = track.keyframes[2]
    expected_early = int(round(0.104 * 0x4000))
    assert abs(int(early[3]) - expected_early) <= 256, (
        f"got early={early[3]:#06x}, expected ~{expected_early:#06x}"
    )


def test_unknown_curve_raises():
    m0 = _make_constant_rotation_motion(1, 3, rz_bams=0)
    m1 = _make_constant_rotation_motion(1, 3, rz_bams=0x4000)
    with pytest.raises(ValueError):
        blend_motions([m0, m1], [0.5, 0.5], transition_curve="zigzag")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_empty_motions_raises():
    with pytest.raises(ValueError):
        blend_motions([], [])


def test_weight_count_mismatch_raises():
    m0 = _make_constant_rotation_motion(1, 3)
    with pytest.raises(ValueError):
        blend_motions([m0], [0.5, 0.5])


def test_bone_count_mismatch_raises():
    m0 = _make_constant_rotation_motion(2, 3, rz_bams=0)
    m1 = _make_constant_rotation_motion(3, 3, rz_bams=0x4000)
    with pytest.raises(ValueError):
        blend_motions([m0, m1], [0.5, 0.5])


def test_zero_frame_count_raises():
    m0 = _make_constant_rotation_motion(1, 3)
    with pytest.raises(ValueError):
        blend_motions([m0], [1.0], frame_count=0)


def test_negative_weights_clamp_to_zero():
    """Negative weights become 0; the blend defaults to the others."""
    m0 = _make_constant_rotation_motion(1, 3, rz_bams=0)
    m1 = _make_constant_rotation_motion(1, 3, rz_bams=0x4000)
    blended = blend_motions([m0, m1], [-0.5, 1.0])
    track = blended.bones[0].tracks_by_kind[NJD_MTYPE_ANG]
    # Effective weights after clamp+normalise: [0, 1] → all source 1.
    for kf in track.keyframes:
        assert abs(int(kf[3]) - 0x4000) <= 8, f"got rz={kf[3]:#06x}"


def test_all_zero_weights_uniform_fallback():
    """All weights zero → uniform fallback (no NaN/Inf)."""
    m0 = _make_constant_rotation_motion(1, 3, rz_bams=0)
    m1 = _make_constant_rotation_motion(1, 3, rz_bams=0x4000)
    blended = blend_motions([m0, m1], [0.0, 0.0])
    track = blended.bones[0].tracks_by_kind[NJD_MTYPE_ANG]
    # Uniform → midpoint (45°).
    for kf in track.keyframes:
        assert abs(int(kf[3]) - 0x2000) <= 16, f"got rz={kf[3]:#06x}"


# ---------------------------------------------------------------------------
# Round-trip + summary
# ---------------------------------------------------------------------------


def test_blend_round_trips_through_njm_bytes():
    m0 = _make_swept_rotation_motion(2, 10, end_rz=0x4000)
    m1 = _make_swept_rotation_motion(2, 10, end_rz=0)
    blended = blend_motions([m0, m1], [0.6, 0.4])
    raw = encode_njm(blended)
    parsed = parse_njm(raw)
    assert len(parsed) == 1
    pm = parsed[0]
    assert pm.frame_count == 10
    assert pm.bone_count == 2


def test_blend_node_dataclass_roundtrip_via_blend_from_node():
    m0 = _make_constant_rotation_motion(1, 5, rz_bams=0)
    m1 = _make_constant_rotation_motion(1, 5, rz_bams=0x4000)
    node = BlendNode(sources=[m0, m1], weights=[0.5, 0.5])
    blended = blend_from_node(node)
    assert blended.frame_count == 5
    track = blended.bones[0].tracks_by_kind[NJD_MTYPE_ANG]
    for kf in track.keyframes:
        assert abs(int(kf[3]) - 0x2000) <= 4


def test_summarize_blend_reports_metadata():
    m0 = _make_constant_rotation_motion(1, 5, rz_bams=0)
    m1 = _make_constant_rotation_motion(1, 5, rz_bams=0x4000)
    blended = blend_motions([m0, m1], [0.6, 0.4])
    s = summarize_blend(blended)
    assert s["frame_count"] == 5
    assert s["bone_count"] == 1
    assert s["source_count"] == 2
    assert len(s["weights"]) == 2
    assert s["curve"] == TRANSITION_LINEAR
    # Weights should be normalised.
    assert abs(sum(s["weights"]) - 1.0) < 1e-6


def test_valid_transitions_constants_match_module_listing():
    assert TRANSITION_LINEAR in VALID_TRANSITIONS
    assert TRANSITION_SMOOTH in VALID_TRANSITIONS
    assert len(VALID_TRANSITIONS) >= 4


# ---------------------------------------------------------------------------
# POS track support
# ---------------------------------------------------------------------------


def _make_motion_with_pos(
    bone_count: int,
    frame_count: int,
    pos_per_frame: tuple,
) -> NjmRawMotion:
    """Build a motion with a constant POS keyframe per frame on bone 0."""
    bones = []
    for bi in range(bone_count):
        bone = NjmBoneTracks()
        bone.tracks_by_kind[NJD_MTYPE_ANG] = NjmTrack(
            kind=NJD_MTYPE_ANG, keyframes=[], narrow=True,
        )
        if bi == 0:
            pos_kfs = [(f, *pos_per_frame) for f in range(frame_count)]
            bone.tracks_by_kind[NJD_MTYPE_POS] = NjmTrack(
                kind=NJD_MTYPE_POS, keyframes=pos_kfs, narrow=True,
            )
        bones.append(bone)
    return NjmRawMotion(
        frame_count=frame_count,
        type_flags=NJD_MTYPE_POS | NJD_MTYPE_ANG,
        inp_fn=2,
        m_data_table_offset=0xC,
        bones=bones,
    )


def test_blend_lerps_positions():
    """Two POS tracks at (0,0,0) and (10,0,0) blended 50/50 → (5,0,0)."""
    m0 = _make_motion_with_pos(1, 5, (0.0, 0.0, 0.0))
    m1 = _make_motion_with_pos(1, 5, (10.0, 0.0, 0.0))
    blended = blend_motions([m0, m1], [0.5, 0.5])
    pos_track = blended.bones[0].tracks_by_kind[NJD_MTYPE_POS]
    assert len(pos_track.keyframes) == 5
    for kf in pos_track.keyframes:
        assert abs(float(kf[1]) - 5.0) < 1e-3
        assert abs(float(kf[2]) - 0.0) < 1e-3
        assert abs(float(kf[3]) - 0.0) < 1e-3
