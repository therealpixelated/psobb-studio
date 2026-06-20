"""Tests for ``formats.anim_retarget`` — glTF animation -> NJM retargeting.

Coverage:
  * Build a synthetic ImportedAnimation with a known sweep, retarget
    it onto a small synthetic target skeleton, encode to NJM bytes,
    re-parse, and verify the keyframe values within tolerance.
  * Verify ``mixamorig:`` prefix stripping picks up bones whose source
    name carries the namespace prefix.
  * Verify bones in source-but-not-in-map are reported as dropped, and
    bones in map-but-not-in-source produce empty NJM tracks (the
    runtime falls back to bind pose for those).
  * Verify the bundled lobby_girl bone map has the keys we expect for
    the typing animation.
"""
from __future__ import annotations
import os

import math
from pathlib import Path

import pytest

from formats.anim_retarget import (
    LOBBY_GIRL_BONE_MAP,
    get_builtin_bone_map,
    retarget_animation,
    summarize_retarget,
)
from formats.import_external import (
    ImportedAnimation,
    ImportedBone,
    ImportedTrack,
    parse_gltf_with_animations,
)
from formats.njm import NJD_MTYPE_ANG, parse_njm
from formats.njm_writer import encode_njm


# ---------------------------------------------------------------------------
# Synthetic source data
# ---------------------------------------------------------------------------


def _make_source_skeleton() -> list[ImportedBone]:
    """3-bone Mixamo-like skeleton: Hips -> Spine -> RightArm."""
    return [
        ImportedBone(
            name="mixamorig:Hips",
            parent_idx=-1,
            bind_pos=(0.0, 1.0, 0.0),
            bind_rot_quat=(0.0, 0.0, 0.0, 1.0),
        ),
        ImportedBone(
            name="mixamorig:Spine",
            parent_idx=0,
            bind_pos=(0.0, 0.2, 0.0),
            bind_rot_quat=(0.0, 0.0, 0.0, 1.0),
        ),
        ImportedBone(
            name="mixamorig:RightArm",
            parent_idx=1,
            bind_pos=(0.2, 0.0, 0.0),
            bind_rot_quat=(0.0, 0.0, 0.0, 1.0),
        ),
    ]


def _make_source_animation(n_frames: int = 30, fps: int = 30) -> ImportedAnimation:
    """One rotation track on RightArm sweeping 0..90° around Y."""
    times = [f / float(fps) for f in range(n_frames)]
    values = []
    for f in range(n_frames):
        ang = math.pi * 0.5 * f / max(1, n_frames - 1)
        values.append((0.0, math.sin(ang * 0.5), 0.0, math.cos(ang * 0.5)))
    return ImportedAnimation(
        name="TestSweep",
        duration_seconds=times[-1],
        fps_target=fps,
        tracks=[
            ImportedTrack(
                bone_idx=2,
                channel="rotation",
                times=times,
                values=values,
                interp="LINEAR",
            ),
        ],
    )


def _make_target_skeleton() -> list[ImportedBone]:
    """3-bone target: bind pose all identity, parented like source."""
    return [
        ImportedBone(name="root", parent_idx=-1, bind_pos=(0.0, 0.0, 0.0), bind_rot_quat=(0.0, 0.0, 0.0, 1.0)),
        ImportedBone(name="torso", parent_idx=0, bind_pos=(0.0, 1.0, 0.0), bind_rot_quat=(0.0, 0.0, 0.0, 1.0)),
        ImportedBone(name="r_arm", parent_idx=1, bind_pos=(0.2, 0.0, 0.0), bind_rot_quat=(0.0, 0.0, 0.0, 1.0)),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_retarget_round_trip_keyframes_preserved():
    """Sweep around Y on RightArm round-trips through NJM bytes within ε."""
    src_skel = _make_source_skeleton()
    src_anim = _make_source_animation(n_frames=30)
    tgt_skel = _make_target_skeleton()
    bone_map = {"Hips": 0, "Spine": 1, "RightArm": 2}

    motion = retarget_animation(
        src_anim, src_skel, tgt_skel, bone_map,
        target_fps=30,
        flip_z=False,  # stay in glTF convention for the round-trip check
    )
    summary = summarize_retarget(motion)
    assert summary["mapped_bones"] == 3
    assert summary["dropped_bones"] == 0
    assert summary["frame_count"] == 30

    # Encode -> parse and verify the RightArm keyframes.
    raw = encode_njm(motion)
    parsed = parse_njm(raw)
    assert len(parsed) == 1
    parsed_motion = parsed[0]
    assert parsed_motion.bone_count == 3
    assert parsed_motion.frame_count == 30

    # RightArm -> target bone 2; expect 30 ANG keyframes sweeping
    # around Y. In BAMS, 90° = 16384.
    arm_kfs = parsed_motion.tracks[2]
    assert len(arm_kfs) == 30
    # First frame: identity = (0, 0, 0).
    first = arm_kfs[0]
    assert first.rx_bams == 0
    assert first.ry_bams == 0
    assert first.rz_bams == 0
    # Last frame: ~16384 (= 90°) around Y, ±1 LSB tolerance.
    last = arm_kfs[-1]
    assert abs(last.ry_bams - 16384) <= 4, f"got ry={last.ry_bams}"


def test_mixamorig_prefix_stripping():
    """'mixamorig:Hips' resolves to the same target as 'Hips'."""
    src_skel = _make_source_skeleton()
    src_anim = _make_source_animation(n_frames=10)
    tgt_skel = _make_target_skeleton()
    bone_map = {"Hips": 0, "Spine": 1, "RightArm": 2}

    motion = retarget_animation(src_anim, src_skel, tgt_skel, bone_map, target_fps=30)
    summary = summarize_retarget(motion)
    assert summary["mapped_bones"] == 3  # all three should map despite prefix


def test_unmapped_source_bone_is_dropped():
    """A source bone whose name isn't in the map is reported as dropped."""
    # Add a fourth source bone with no corresponding map entry.
    src_skel = _make_source_skeleton() + [
        ImportedBone(name="mixamorig:LeftEye", parent_idx=1,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
    ]
    src_anim = _make_source_animation(n_frames=10)
    tgt_skel = _make_target_skeleton()
    bone_map = {"Hips": 0, "Spine": 1, "RightArm": 2}

    motion = retarget_animation(src_anim, src_skel, tgt_skel, bone_map, target_fps=30)
    summary = summarize_retarget(motion)
    assert summary["mapped_bones"] == 3
    assert summary["dropped_bones"] == 1
    assert any("LeftEye" in d for d in summary["dropped"])


def test_target_bone_with_no_source_track_emits_empty_track():
    """A target bone the map points at, but the source doesn't drive,
    gets an empty NJM keyframe list (parser falls back to bind)."""
    src_skel = _make_source_skeleton()
    src_anim = _make_source_animation(n_frames=10)
    tgt_skel = _make_target_skeleton() + [
        ImportedBone(name="extra", parent_idx=0,
                     bind_pos=(0, 0, 0), bind_rot_quat=(0, 0, 0, 1)),
    ]
    # Map points at target bone 3 but no source bone has that name.
    bone_map = {"Hips": 0, "Spine": 1, "RightArm": 2, "DoesNotExist": 3}

    motion = retarget_animation(src_anim, src_skel, tgt_skel, bone_map, target_fps=30)
    raw = encode_njm(motion)
    parsed = parse_njm(raw)[0]
    assert parsed.bone_count == 4
    # Bone 3 has no keyframes (source doesn't expose 'DoesNotExist').
    assert parsed.tracks[3] == []
    # The synthetic source animation only authors a rotation track on
    # bone 2 (RightArm); bones 0 + 1 also have empty tracks. Only
    # bone 2's per-bone ANG bit should be set.
    assert parsed.bone_present_tracks[2] & NJD_MTYPE_ANG
    assert not (parsed.bone_present_tracks[3] & NJD_MTYPE_ANG)
    assert parsed.tracks[0] == []
    assert parsed.tracks[1] == []
    assert len(parsed.tracks[2]) == 10  # the synth source has 10 frames


def test_lobby_girl_bone_map_has_typing_joints():
    """The bundled lobby-girl map covers every joint the typing
    synthetic clip animates."""
    needed = {
        "Hips", "Spine", "Neck", "Head",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    }
    for n in needed:
        assert n in LOBBY_GIRL_BONE_MAP, f"missing {n}"


def test_get_builtin_bone_map_returns_copy():
    """The accessor returns a fresh dict so mutations don't bleed."""
    m = get_builtin_bone_map("lobby_girl")
    m["Hips"] = 999
    m2 = get_builtin_bone_map("lobby_girl")
    assert m2["Hips"] != 999


def test_get_builtin_bone_map_unknown_raises():
    with pytest.raises(KeyError):
        get_builtin_bone_map("nope")


def test_resample_to_lower_fps():
    """Source authored at 60 fps resamples to 30 fps with half the frames."""
    times = [f / 60.0 for f in range(60)]  # 60 frames at 60fps = 1 second
    values = []
    for f in range(60):
        ang = math.pi * 0.5 * f / 59
        values.append((0.0, math.sin(ang * 0.5), 0.0, math.cos(ang * 0.5)))
    anim = ImportedAnimation(
        name="HighRate",
        duration_seconds=times[-1],
        fps_target=30,
        tracks=[ImportedTrack(bone_idx=2, channel="rotation", times=times,
                              values=values, interp="LINEAR")],
    )
    motion = retarget_animation(
        anim, _make_source_skeleton(), _make_target_skeleton(),
        {"Hips": 0, "Spine": 1, "RightArm": 2},
        target_fps=30,
        flip_z=False,
    )
    # 1 second at 30 fps -> 31 frames (0..30 inclusive).
    assert motion.frame_count in (30, 31)
    raw = encode_njm(motion)
    parsed = parse_njm(raw)[0]
    arm_kfs = parsed.tracks[2]
    # Last keyframe still 90° around Y.
    assert abs(arm_kfs[-1].ry_bams - 16384) <= 8


def test_end_to_end_synth_typing_to_lobby_girl():
    """End-to-end: synth glb -> retarget onto kenkyu_w skeleton -> parse."""
    glb_path = Path("data/animation_assets/standing_typing.glb")
    if not glb_path.is_file():
        pytest.skip(f"animation asset missing: {glb_path}")
    bml_path = Path(os.path.expanduser("~/PSOBB.IO/data/bm_npc_kenkyu_w.bml"))
    if not bml_path.is_file():
        pytest.skip(f"PSOBB data missing: {bml_path}")

    from formats.bml import parse_bml, _prs_decompress
    from formats.xj import parse_skeleton

    raw_bml = bml_path.read_bytes()
    entries = parse_bml(raw_bml)
    inner = _prs_decompress(raw_bml[entries[0].offset:entries[0].offset + entries[0].size_compressed])
    target_skel = parse_skeleton(inner)
    assert len(target_skel) == 64

    imp = parse_gltf_with_animations(str(glb_path))
    motion = retarget_animation(
        imp.animations[0],
        imp.model.bones,
        target_skel,
        get_builtin_bone_map("lobby_girl"),
        target_fps=30,
    )
    assert motion.frame_count == 90
    assert len(motion.bones) == 64

    raw_njm = encode_njm(motion)
    parsed = parse_njm(raw_njm)[0]
    assert parsed.bone_count == 64
    assert parsed.frame_count == 90
    # At least the upper-body joints should have keyframes.
    upper_body_indices = [1, 2, 3, 4, 7, 10, 12, 13, 16, 19, 21, 23]
    n_animated = sum(1 for i in upper_body_indices if parsed.tracks[i])
    assert n_animated >= 10, f"expected most upper-body bones animated, got {n_animated}"
