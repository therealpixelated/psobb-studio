"""Parity tests for the DOMAIN=models audit fix in ``formats.njm_writer``.

Audit finding (high): ``njm_writer._parse_ang_narrow`` lacked the
``frame >= frame_count`` rejection that the reading-side parser
``formats.njm._parse_euler_keyframes`` (line ``if frame < prev_frame or
frame >= frame_count``) and Phantasmal's ``parseEulerAngleKeyframes``
(Motion.kt:238, ``keyframe.frame < prev || keyframe.frame >=
frameCount``) both apply. Without it, WIDE 16-byte euler tracks whose
first u16 pair happens to be monotonic decode as NARROW garbage — e.g.
``pxuG01_A06_F_body.njm`` fc=20 bone 0 read frame 56043 / angle 65535
instead of the true wide frame 19 / angle -9493. Byte-exact round-trip
was already preserved (the encoder re-emits ``source_body`` verbatim),
but the corrupt decoded keyframes feed ``anim_blend`` / ``anim_retarget``.

These tests pin:
  1. The specific reported file decodes its euler tracks identically to
     the ground-truth reading-side parser ``njm._parse_euler_keyframes``.
  2. Across the WHOLE shipped corpus: every ANG track decoded by the
     writer value-equals the reading-side parser, AND byte-exact
     round-trip is unaffected (must stay 100%).
  3. A data-free synthetic unit test of the gate itself, so the core
     logic is covered even when game data is absent on CI.

Ground truth (oracle): Phantasmal ``Motion.kt`` (MIT) reproduced in
``formats.njm``; the writer must agree with it on decoded values.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest

from formats.iff import parse_iff
from formats.njm import _parse_euler_keyframes
from formats.njm_writer import (
    NJD_MTYPE_ANG,
    _parse_ang_narrow,
    encode_njm,
    parse_njm_for_writer,
)


# The audit corpus lives in PSOBB.IO/data (1623 motions, byte-exact
# round-trip). EphineaPSO/data is an equivalent shipped corpus; either
# satisfies the parity assertions. We never WRITE to these dirs.
_CORPUS_DIRS = [
    Path(os.path.expanduser("~/PSOBB.IO/data")),
    Path(os.path.expanduser("~/EphineaPSO/data")),
]


def _corpus_dir() -> Path | None:
    for d in _CORPUS_DIRS:
        if d.is_dir():
            return d
    return None


HAS_CORPUS = _corpus_dir() is not None


def _nmdm_body(njm_bytes: bytes) -> bytes:
    return next(c for c in parse_iff(njm_bytes) if c.type == "NMDM").data


# ---------------------------------------------------------------------------
# 1. The specific file named in the audit.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CORPUS, reason="shipped NJM corpus not present")
def test_pxu_g01_a06_euler_matches_reading_parser():
    """``pxuG01_A06_F_body.njm`` bone-0 ANG must decode WIDE, matching njm.py.

    Regression guard for the audit finding: the pre-fix monotonic-only
    narrow probe accepted this wide track as narrow garbage.
    """
    from formats.bml import extract_bml

    data = _corpus_dir()
    bml = data / "NpcApcMot.bml"
    if not bml.exists():
        pytest.skip("NpcApcMot.bml not in corpus")
    entries = extract_bml(bml.read_bytes())
    if "pxuG01_A06_F_body.njm" not in entries:
        pytest.skip("pxuG01_A06_F_body.njm not in NpcApcMot.bml")
    src = entries["pxuG01_A06_F_body.njm"]

    raw = parse_njm_for_writer(src)
    track = raw.bones[0].tracks_by_kind[NJD_MTYPE_ANG]

    # Must be the WIDE form now (the bug decoded it narrow).
    assert track.narrow is False

    # Value-parity vs the ground-truth reading-side parser on the same
    # (offset, count, frame_count).
    body = _nmdm_body(src)
    frame_count = struct.unpack_from("<I", body, 4)[0]
    off = raw.track_offset_hint[(0, NJD_MTYPE_ANG)]
    cnt = track.stored_count if track.stored_count is not None else len(track.keyframes)
    ref = _parse_euler_keyframes(body, off, cnt, frame_count)
    assert track.keyframes == ref
    # And specifically the documented values (frame 0/19, ry = -9493).
    assert ref[0] == (0, 0, -9493, 0)
    assert ref[-1][0] == 19  # last frame well within fc=20 once wide


# ---------------------------------------------------------------------------
# 2. Whole-corpus parity + byte-exact round-trip.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CORPUS, reason="shipped NJM corpus not present")
def test_corpus_euler_parity_and_roundtrip():
    """Every ANG track value-equals njm.py AND round-trip stays byte-exact."""
    from formats.bml import extract_bml

    data = _corpus_dir()
    total = exact = 0
    ang_tracks = ang_mismatch = 0
    for fname in sorted(os.listdir(data)):
        if not fname.endswith(".bml"):
            continue
        try:
            entries = extract_bml((data / fname).read_bytes())
        except Exception:
            continue
        for name, inner in entries.items():
            if not name.endswith(".njm"):
                continue
            total += 1
            try:
                raw = parse_njm_for_writer(inner)
                out = encode_njm(raw)
            except Exception:
                continue
            if out == inner:
                exact += 1
            body = _nmdm_body(inner)
            frame_count = struct.unpack_from("<I", body, 4)[0]
            for bi, bone in enumerate(raw.bones):
                t = bone.tracks_by_kind.get(NJD_MTYPE_ANG)
                if not t or not t.keyframes:
                    continue
                ang_tracks += 1
                off = raw.track_offset_hint.get((bi, NJD_MTYPE_ANG))
                cnt = t.stored_count if t.stored_count is not None else len(t.keyframes)
                ref = _parse_euler_keyframes(body, off, cnt, frame_count)
                if ref != t.keyframes:
                    ang_mismatch += 1

    assert total >= 100, f"only {total} NJMs — corpus missing?"
    # Byte-exact round-trip must be unaffected by the decode fix.
    assert exact == total, f"round-trip regressed: {exact}/{total} byte-exact"
    # Decoded euler values must agree with the reading-side parser.
    assert ang_mismatch == 0, (
        f"{ang_mismatch}/{ang_tracks} ANG tracks disagree with njm._parse_euler_keyframes"
    )


# ---------------------------------------------------------------------------
# 3. Data-free unit test of the gate itself.
# ---------------------------------------------------------------------------


def test_narrow_probe_rejects_frame_at_or_beyond_frame_count():
    """A monotonic narrow buffer whose frame >= frame_count must be rejected.

    This is the crux of the port: the OLD probe (monotonic-only) would
    accept it; the fixed probe rejects it so the caller falls back to
    wide. Exercised without any game data.
    """
    # One narrow keyframe: frame=19, three u16 angles. fc=20 accepts it,
    # fc=10 rejects it (19 >= 10).
    buf = struct.pack("<HHHH", 19, 0, 1, 2)
    ok = _parse_ang_narrow(buf, 0, 1, 20)
    assert ok == [(19, 0, 1, 2)]
    rejected = _parse_ang_narrow(buf, 0, 1, 10)
    assert rejected is None  # frame 19 is at-or-beyond frame_count 10

    # Boundary: frame == frame_count is rejected (matches Motion.kt:238
    # `frame >= frameCount`).
    buf_eq = struct.pack("<HHHH", 10, 0, 0, 0)
    assert _parse_ang_narrow(buf_eq, 0, 1, 10) is None
    assert _parse_ang_narrow(buf_eq, 0, 1, 11) == [(10, 0, 0, 0)]

    # Non-monotonic still rejected regardless of frame_count.
    buf_nm = struct.pack("<HHHH", 5, 0, 0, 0) + struct.pack("<HHHH", 3, 0, 0, 0)
    assert _parse_ang_narrow(buf_nm, 0, 2, 100) is None
