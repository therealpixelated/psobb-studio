"""Tests for ``formats.njm_writer`` — the NJM encoder.

Coverage:
  - Round-trip 20 representative NJMs from PSOBB.IO (skipped without).
  - Synthetic motion: 4-bone skeleton, 30 frames, sine-wave Y rotation.
  - Synthetic with absent channels: bones 0-1 POS+ANG, bones 2-3 ANG only.
  - njmotion_to_raw conversion path.
"""
from __future__ import annotations
import os

import math
import struct
from pathlib import Path

import pytest

from formats.bml import extract_bml
from formats.iff import parse_iff
from formats.njm import (
    NJD_MTYPE_ANG,
    NJD_MTYPE_POS,
    NJD_MTYPE_QUAT,
    NJD_MTYPE_SCL,
    NjmKeyframe,
    NjmMotion,
    parse_njm,
)
from formats.njm_writer import (
    NjmBoneTracks,
    NjmRawMotion,
    NjmTrack,
    encode_njm,
    njmotion_to_raw,
    parse_njm_for_writer,
)


PSOBB_DATA = Path(os.path.expanduser("~/PSOBB.IO/data"))
HAS_PSOBB = PSOBB_DATA.is_dir()


# ---------------------------------------------------------------------------
# Synthetic motion: 4-bone skeleton, 30-frame Y-axis sine rotation.
# ---------------------------------------------------------------------------


def _build_synthetic_motion(n_bones: int = 4, n_frames: int = 30) -> NjmRawMotion:
    """A POS+ANG motion: bone 0 has both, bones 1+ have ANG only.

    The Y rotation oscillates as a sine wave over n_frames. POS for
    bone 0 is constant zero (root translation = 0 throughout).
    """
    motion = NjmRawMotion(
        frame_count=n_frames,
        type_flags=NJD_MTYPE_POS | NJD_MTYPE_ANG,
        inp_fn=2,  # interp=0, element_count=2
    )
    bones = []
    for b_idx in range(n_bones):
        bone = NjmBoneTracks()
        # POS track: only bone 0.
        if b_idx == 0:
            pos_kfs = [(f, 0.0, 0.0, 0.0) for f in range(n_frames)]
            bone.tracks_by_kind[NJD_MTYPE_POS] = NjmTrack(
                kind=NJD_MTYPE_POS, keyframes=pos_kfs, narrow=True,
            )
        else:
            bone.tracks_by_kind[NJD_MTYPE_POS] = NjmTrack(
                kind=NJD_MTYPE_POS, keyframes=[], narrow=True,
            )
        # ANG track: every bone.
        ang_kfs = []
        for f in range(n_frames):
            angle_bams = int(0x10000 * 0.25 * math.sin(f / n_frames * 2.0 * math.pi))
            ang_kfs.append((f, 0, angle_bams & 0xFFFF, 0))
        bone.tracks_by_kind[NJD_MTYPE_ANG] = NjmTrack(
            kind=NJD_MTYPE_ANG, keyframes=ang_kfs, narrow=True,
        )
        bones.append(bone)
    motion.bones = bones
    return motion


def test_synthetic_motion_round_trips():
    """Encode synthetic motion → parse → encode produces stable bytes."""
    motion = _build_synthetic_motion()
    out1 = encode_njm(motion)
    motion2 = parse_njm_for_writer(out1)
    out2 = encode_njm(motion2)
    assert out1 == out2


def test_synthetic_motion_parses_via_official_parser():
    """Encoded synthetic NJM parses correctly via formats.njm.parse_njm."""
    motion = _build_synthetic_motion()
    out = encode_njm(motion)
    motions = parse_njm(out)
    assert len(motions) == 1
    parsed = motions[0]
    assert parsed.bone_count == 4
    assert parsed.frame_count == 30
    assert parsed.type_flags == 0x3
    # Bone 0 has both POS and ANG.
    assert parsed.bone_present_tracks[0] == 0x3
    # Bones 1-3 have only ANG.
    for b in range(1, 4):
        assert parsed.bone_present_tracks[b] == 0x2


def test_synthetic_present_mask_preserves_absent_pos():
    """Bones without POS keep their POS bit unset in present_tracks."""
    motion = _build_synthetic_motion()
    out = encode_njm(motion)
    motions = parse_njm(out)
    parsed = motions[0]
    # Ensure bone 1 has NO position keyframes — only ANG ones.
    pos_kfs_b1 = [
        kf for kf in parsed.tracks[1]
        if kf.tx != 0.0 or kf.ty != 0.0 or kf.tz != 0.0
    ]
    assert pos_kfs_b1 == []  # synthetic motion has no POS for bone 1


def test_synthetic_motion_with_scale():
    """POS+ANG+SCL motion (type=7); verify SCL track encodes correctly."""
    motion = NjmRawMotion(
        frame_count=10,
        type_flags=NJD_MTYPE_POS | NJD_MTYPE_ANG | NJD_MTYPE_SCL,
        inp_fn=3,  # interp=0, element_count=3
    )
    bone = NjmBoneTracks()
    bone.tracks_by_kind[NJD_MTYPE_POS] = NjmTrack(
        NJD_MTYPE_POS, [(0, 1.0, 2.0, 3.0)], narrow=True,
    )
    bone.tracks_by_kind[NJD_MTYPE_ANG] = NjmTrack(
        NJD_MTYPE_ANG, [(0, 100, 200, 300)], narrow=True,
    )
    bone.tracks_by_kind[NJD_MTYPE_SCL] = NjmTrack(
        NJD_MTYPE_SCL, [(0, 1.5, 1.5, 1.5)], narrow=True,
    )
    motion.bones = [bone]
    out = encode_njm(motion)
    parsed = parse_njm_for_writer(out)
    assert parsed.type_flags == 0x7
    assert len(parsed.bones) == 1


def test_synthetic_motion_with_quat():
    """POS+QUAT motion (type=0x2001); verify QUAT track encodes."""
    motion = NjmRawMotion(
        frame_count=5,
        type_flags=NJD_MTYPE_POS | NJD_MTYPE_QUAT,
        inp_fn=2,  # element_count=2
    )
    bone = NjmBoneTracks()
    bone.tracks_by_kind[NJD_MTYPE_POS] = NjmTrack(
        NJD_MTYPE_POS, [(0, 0.0, 0.0, 0.0)], narrow=True,
    )
    bone.tracks_by_kind[NJD_MTYPE_QUAT] = NjmTrack(
        NJD_MTYPE_QUAT, [(0, 1.0, 0.0, 0.0, 0.0)], narrow=True,
    )
    motion.bones = [bone]
    out = encode_njm(motion)
    parsed = parse_njm_for_writer(out)
    assert parsed.type_flags == 0x2001


# ---------------------------------------------------------------------------
# njmotion_to_raw conversion
# ---------------------------------------------------------------------------


def test_njmotion_to_raw_round_trip_simple():
    """Build an NjmMotion, convert to raw, encode, parse — fields preserved."""
    motion = NjmMotion(
        bone_count=2,
        frame_count=5,
        type_flags=0x3,
        interpolation=0,
        tracks=[
            [
                NjmKeyframe(time=0, tx=1.0, ty=2.0, tz=3.0,
                            rx_bams=100, ry_bams=200, rz_bams=300),
                NjmKeyframe(time=4, tx=4.0, ty=5.0, tz=6.0,
                            rx_bams=400, ry_bams=500, rz_bams=600),
            ],
            [
                NjmKeyframe(time=0, rx_bams=10, ry_bams=20, rz_bams=30),
                NjmKeyframe(time=4, rx_bams=40, ry_bams=50, rz_bams=60),
            ],
        ],
        bone_present_tracks=[0x3, 0x2],  # bone 0 has POS+ANG, bone 1 has ANG only
    )
    raw = njmotion_to_raw(motion)
    out = encode_njm(raw)
    parsed = parse_njm(out)[0]
    assert parsed.bone_count == 2
    assert parsed.bone_present_tracks == [0x3, 0x2]


# ---------------------------------------------------------------------------
# Live PSOBB round-trip
# ---------------------------------------------------------------------------


_LIVE_TEST_BMLS = [
    ("bm_boss1_dragon.bml",     "bossop_boss1_s_nb_dragon.njm",  "boss-class POS+ANG, 124 bones"),
    ("bm_boss1_dragon.bml",     "walk_boss1_s_nb_dragon.njm",    "dragon walk cycle"),
    ("bm_boss1_dragon.bml",     "fly_boss1_s_nb_dragon.njm",     "dragon fly motion"),
    ("bm_boss1_dragon.bml",     "frin_boss1_s_nb_dragon.njm",    "dragon fly-in"),
    ("bm_boss1_dragon.bml",     "frloop_boss1_s_nb_dragon.njm",  "dragon fly loop"),
    ("bm_boss1_dragon.bml",     "frout_boss1_s_nb_dragon.njm",   "dragon fly-out"),
    ("bm_boss1_dragon.bml",     "dead_boss1_s_nb_dragon.njm",    "dragon death"),
    ("bm_boss1_dragon.bml",     "down_boss1_s_nb_dragon.njm",    "dragon down"),
    ("bm_boss1_dragon.bml",     "fire_boss1_s_nb_dragon.njm",    "dragon fire breath"),
    ("bm_boss1_dragon.bml",     "flyshot_boss1_s_nb_dragon.njm", "dragon fly+shot"),
    ("bm_boss1_dragon.bml",     "land_boss1_s_nb_dragon.njm",    "dragon land"),
    ("bm_boss1_dragon.bml",     "kiri_boss1_s_nb_dragon.njm",    "dragon kiri (turn)"),
    ("bm_boss1_dragon.bml",     "stand_boss1_s_nb_dragon.njm",   "dragon stand"),
    ("bm_boss1_dragon.bml",     "wing_boss1_s_nb_dragon.njm",    "dragon wing"),
]


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO data not present")
@pytest.mark.parametrize("bml_name,inner_name,reason", _LIVE_TEST_BMLS)
def test_live_round_trip(bml_name, inner_name, reason):
    """Round-trip a representative shipped NJM."""
    bml_path = PSOBB_DATA / bml_name
    if not bml_path.exists():
        pytest.skip(f"{bml_name} not in PSOBB.IO/data")
    all_e = extract_bml(bml_path.read_bytes())
    if inner_name not in all_e:
        pytest.skip(f"{bml_name} has no entry {inner_name}")
    src = all_e[inner_name]
    motion = parse_njm_for_writer(src)
    out = encode_njm(motion)
    assert out == src, (
        f"{bml_name}#{inner_name} ({reason}): "
        f"src {len(src)} bytes, out {len(out)} bytes"
    )


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO data not present")
def test_live_round_trip_corpus_high_rate():
    """Walk every shipped NJM and verify byte-exact round-trip ≥ 95%."""
    import os

    total = exact = 0
    for fname in sorted(os.listdir(PSOBB_DATA)):
        if not fname.endswith(".bml"):
            continue
        try:
            all_e = extract_bml((PSOBB_DATA / fname).read_bytes())
        except Exception:
            continue
        for inner_name, inner in all_e.items():
            if not inner_name.endswith(".njm"):
                continue
            total += 1
            try:
                motion = parse_njm_for_writer(inner)
                out = encode_njm(motion)
            except Exception:
                continue
            if out == inner:
                exact += 1
    assert total >= 100, f"only {total} NJMs in corpus — PSOBB.IO data missing?"
    rate = exact / total
    assert rate >= 0.95, f"round-trip rate {rate*100:.1f}% < 95% ({exact}/{total})"


# ---------------------------------------------------------------------------
# Mutation tests
# ---------------------------------------------------------------------------


def test_mutate_synthetic_motion_keyframe_count():
    """Drop a keyframe from a synthetic motion; verify it still encodes."""
    motion = _build_synthetic_motion()
    # Drop the last keyframe of bone 0's ANG track.
    motion.bones[0].tracks_by_kind[NJD_MTYPE_ANG].keyframes.pop()
    motion.bones[0].tracks_by_kind[NJD_MTYPE_ANG].stored_count = None  # let writer auto-derive
    out = encode_njm(motion)
    parsed = parse_njm_for_writer(out)
    assert len(parsed.bones[0].tracks_by_kind[NJD_MTYPE_ANG].keyframes) == 29


def test_synthetic_zero_bone_motion():
    """A 0-bone motion produces a valid-but-empty NJM."""
    motion = NjmRawMotion(
        frame_count=10,
        type_flags=NJD_MTYPE_POS | NJD_MTYPE_ANG,
        inp_fn=2,
    )
    out = encode_njm(motion)
    chunks = parse_iff(out)
    assert chunks[0].type == "NMDM"
    assert len(chunks[0].data) == 12  # header only
