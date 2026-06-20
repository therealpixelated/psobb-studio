# Ported from MIT-licensed Phantasmal World by Daan Vanden Bosch.
# See LICENSES.md at the editor root for the verbatim MIT block.
#
# Reference (MIT):
#   psolib/src/commonMain/kotlin/world/phantasmal/psolib/fileFormats/ninja/Motion.kt
#   psolib/src/commonMain/kotlin/world/phantasmal/psolib/fileFormats/ninja/Angle.kt
#
# This module parses the Sega Ninja "NjMotion" (NJM) format that PSOBB
# Blue Burst uses for skeletal animations. Two variants ship in PSOBB.IO:
#
#   * v2 ("NMDM") — the IFF-wrapped form that lives inside a BML
#     archive's per-entry `.njm` blob (e.g.
#     ``bm_boss8_dragon.bml#walk_boss1_s_nb_dragon.njm``). Header magic
#     is ``NMDM``; we receive the IFF chunk PAYLOAD (4-byte tag + 4-byte
#     size already stripped by ``formats/iff.py``).
#
#   * BB ("Blue Burst") — the legacy/standalone .njm form Phantasmal
#     handles via the ``parseNjmBb`` helper (offset chain at
#     end-of-file). We do NOT see this in PSOBB.IO data — every shipping
#     .njm is the v2/NMDM form — but the parser supports it for
#     completeness and future-proofing against modded data.
#
# Wire shape (NMDM body, all little-endian):
#
#   u32  mDataTableOffset   — byte offset (relative to NMDM body start)
#                              of the per-bone MData array.
#   u32  frame_count        — total frames in the longest track.
#   u16  type               — bitfield of which transform tracks are
#                              present per bone. Bit 0 = position,
#                              bit 1 = euler angles (BAMS),
#                              bit 2 = scale, bit 13 = quaternion. PSOBB
#                              data uses POS+ANG (=3), POS+ANG+SCL (=7),
#                              and very rarely POS+QUAT (=0x2001).
#   u16  inpFn              — high 2 bits = interpolation mode (0=linear,
#                              1=spline, 2/3=user-fn). Low 4 bits =
#                              element_count (number of tracks per bone;
#                              equals popcount(type)).
#
#   ... padding to mDataTableOffset ...
#
#   At mDataTableOffset, an array of MData entries — one per BONE, in
#   the same depth-first order as the model's mesh-tree nodes:
#
#     element_count * u32  — track keyframe-list byte offsets
#                              (relative to NMDM body start; 0 = empty
#                              track for this bone)
#     element_count * u32  — track keyframe counts
#
#   Each track type has its own keyframe layout:
#     Position / Scale (vector):   u32 frame, f32 x, f32 y, f32 z
#     Euler angles (BAMS):
#       Narrow form (16-bit):       u16 frame, u16 rx, u16 ry, u16 rz
#         All four fields are u16; angles are BAMS (0x10000 = 360°).
#         Used when frame ids fit in u16 AND ascending.
#       Wide form (32-bit):         u32 frame, i32 rx, i32 ry, i32 rz
#         Phantasmal falls back to this if the narrow form would lose
#         monotonicity (rare; PSOBB BB data uses narrow ~99% of the time).
#     Quaternion:                  u32 frame, f32 real, f32 ix, f32 iy, f32 iz
#
# Per-bone tracks are sorted by frame ascending (the SDK guarantees
# this; we don't re-sort). Frame indices may skip — interpolation is
# handled by the consumer (see ``static/model_viewer.js`` for the
# linear-spline implementation).
#
# Public API:
#   NjmKeyframe   — one keyframe (time, t-vec, r-bams, s-vec).
#   NjmMotion     — the parsed motion (name, frame_count, fps, tracks).
#   parse_njm     — bytes → list[NjmMotion]. Always returns a list (one
#                   element per motion in the file; v2/BB both currently
#                   give exactly one).
#   guess_motion_fps — heuristic FPS picker based on motion name + flags.
"""Pure-Python parser for PSOBB Blue Burst NJM (Ninja Motion) files."""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import List, Optional

from .iff import parse_iff


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NjmKeyframe:
    """One keyframe in a per-bone track.

    Carries TRS deltas; fields not present in the source motion are
    populated with identity defaults (0,0,0 for translation/rotation,
    1,1,1 for scale) so consumers can apply unconditionally.

    Attributes
    ----------
    time:
        Frame index (integer; 0 = first frame). May skip — interpolation
        between consecutive keyframes is the consumer's responsibility.
    tx, ty, tz:
        Translation in model units (same as bone positions in the
        skeleton). Identity = (0, 0, 0).
    rx_bams, ry_bams, rz_bams:
        Rotation in Sega Ninja BAMS (signed integer; 0x10000 = 360°).
        Identity = (0, 0, 0). Convert to radians by multiplying by
        ``2π / 0x10000``. Z-Y-X Euler order to match the bind pose
        (matches ``formats/xj.py::_mat4_compose_trs``).
    sx, sy, sz:
        Per-axis scale. Identity = (1, 1, 1).
    qw, qx, qy, qz:
        Quaternion (when type bit 13 is set; otherwise None). When set,
        consumers should prefer the quaternion over the BAMS Euler.
        Identity = (1, 0, 0, 0). Phantasmal stores quaternions as
        (real, ix, iy, iz); we keep the same order.
    """
    time: int = 0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    rx_bams: int = 0
    ry_bams: int = 0
    rz_bams: int = 0
    sx: float = 1.0
    sy: float = 1.0
    sz: float = 1.0
    qw: Optional[float] = None
    qx: Optional[float] = None
    qy: Optional[float] = None
    qz: Optional[float] = None


@dataclass
class NjmMotion:
    """One parsed motion.

    Attributes
    ----------
    name:
        Source filename or chunk label (set by the caller; ``parse_njm``
        leaves it empty).
    bone_count:
        Number of MData entries (one per bone in the source mesh tree's
        DFS order). May exceed the actual mesh-tree node count when the
        motion was authored against a more-detailed skeleton; consumers
        should clamp by ``min(bone_count, mesh_bone_count)``.
    frame_count:
        Highest frame index across all tracks (i.e. the motion's length
        in frames). Frame 0 is the first frame; the last frame is at
        ``frame_count - 1`` if all tracks include it (most motions DO
        include the last frame as a keyframe).
    fps:
        Frames-per-second. Stored on the dataclass but the source NJM
        does NOT carry it — Sega Ninja motions are authored against a
        fixed 30 Hz tick (PSOBB sim runs at 30 Hz; see
        ``psobb_framerate_pacing.md`` in the user's memory). The default
        is 30.0; ``guess_motion_fps`` overrides for specific motion
        names.
    type_flags:
        Raw ``type`` u16 from the NJM header (bitfield). Bit 0 = POS,
        bit 1 = ANG, bit 2 = SCL, bit 13 = QUAT. Surfaced for
        diagnostics.
    interpolation:
        0 = linear, 1 = spline, 2/3 = user-function (PSOBB uses linear
        almost exclusively).
    tracks:
        ``tracks[bone_idx]`` is a list of ``NjmKeyframe``, sorted by
        time ascending. Empty list means "this bone has no animation —
        use the skeleton's bind pose unchanged".
    bone_present_tracks:
        ``bone_present_tracks[bone_idx]`` is a per-bone bitfield
        identifying which transform CHANNELS were ACTUALLY authored on
        that bone. Bit 0 = POS, bit 1 = ANG, bit 2 = SCL, bit 13 = QUAT.
        This is NOT the same as ``type_flags`` — ``type_flags`` is the
        motion-wide enable mask (e.g. POS+ANG=3), but a specific bone
        may have been authored with rotations only (count=0 for the POS
        track). Consumers MUST consult this when blending keyframes
        with bind pose: a keyframe with ``tx,ty,tz=0`` for a bone whose
        POS bit is unset in this mask should be IGNORED in favour of
        the bone's bind translation. Motivation: Sega Ninja's NJM stores
        empty tracks compactly (count=0, offset=0); the parser previously
        defaulted those channels to identity which yanked all
        rotation-only bones to the origin during playback. Per-bone
        track presence is the only signal that distinguishes
        "intentional zero translation" from "absent track".
    """
    name: str = ""
    bone_count: int = 0
    frame_count: int = 0
    fps: float = 30.0
    type_flags: int = 0
    interpolation: int = 0
    tracks: List[List[NjmKeyframe]] = field(default_factory=list)
    bone_present_tracks: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal: parse a single NMDM motion body.
# ---------------------------------------------------------------------------
#
# ``body`` is the bytes of an NMDM IFF chunk's PAYLOAD (i.e. NOT
# including the 8-byte IFF header). The first u32 is mDataTableOffset
# relative to ``body[0]``.

# Type bitfield (u16) — see ``MotionFlag`` in pso-blender's njm.py.
NJD_MTYPE_POS = 1 << 0     # vector — translation
NJD_MTYPE_ANG = 1 << 1     # BAMS-encoded euler rotation
NJD_MTYPE_SCL = 1 << 2     # vector — scale
NJD_MTYPE_VEC = 1 << 3
NJD_MTYPE_QUAT = 1 << 13   # quaternion (overrides ANG when both set)


def _parse_motion(body: bytes, *, v2_format: bool) -> NjmMotion:
    """Parse one motion's body bytes into an ``NjmMotion``.

    Direct port of Phantasmal's ``parseMotion`` in Motion.kt. The
    ``v2_format`` flag controls the mDataTable end-of-region detection
    rule (in v2 we shrink the end as we encounter keyframe offsets that
    point into the payload region; in BB we use the cursor's start
    position as the upper bound).

    Raises ``ValueError`` on truncated / corrupt inputs.
    """
    n = len(body)
    if n < 12:
        raise ValueError(f"NJM body too small: {n} < 12")

    m_data_table_offset, frame_count, type_flags, inp_fn = struct.unpack_from(
        "<II HH", body, 0
    )
    interpolation = (inp_fn & 0b1100_0000) >> 6
    element_count = inp_fn & 0b1111
    has_pos = bool(type_flags & NJD_MTYPE_POS)
    has_ang = bool(type_flags & NJD_MTYPE_ANG)
    has_scl = bool(type_flags & NJD_MTYPE_SCL)
    has_quat = bool(type_flags & NJD_MTYPE_QUAT)

    # Sanity caps: real PSOBB motions top out around 4 tracks per bone
    # (POS+ANG+SCL+QUAT = 4; VEC adds a 5th but PSOBB doesn't use it).
    # Phantasmal accepts 4; we accept up to 8 for forward-compat.
    if element_count > 8:
        raise ValueError(
            f"NJM element_count {element_count} too large (max 8)"
        )
    if element_count == 0:
        # Zero-track motion. Phantasmal returns an empty NjMotion; we
        # do the same (no bones to walk).
        return NjmMotion(
            bone_count=0,
            frame_count=frame_count,
            type_flags=type_flags,
            interpolation=interpolation,
            tracks=[],
        )

    # Sanity check the offset is in-bounds.
    if m_data_table_offset < 12 or m_data_table_offset > n:
        raise ValueError(
            f"NJM mDataTableOffset 0x{m_data_table_offset:x} out of bounds "
            f"(body size {n})"
        )

    # Per-bone MData entries. Each entry is element_count u32 keyframe
    # offsets followed by element_count u32 keyframe counts (= 8 *
    # element_count bytes). The end of the table is determined by
    # walking until we hit the lowest keyframe-offset we've seen
    # (Phantasmal's "shrink mDataTableEnd as we go" trick).
    m_data_table_end = n if v2_format else m_data_table_offset
    if not v2_format:
        # In BB form the table END is reset by the caller; we keep
        # m_data_table_end at the parser's starting position. PSOBB
        # data uses v2 only, so this branch is rarely exercised — but
        # honour Phantasmal's logic so the BB path remains usable.
        #
        # NOTE (audit low, dead path): Motion.kt:108 seeds the BB end to
        # ``cursor.position`` (= our body-relative 0, since we receive a
        # PRE-SLICED ``body = buf[motion_offset:]``), whereas we seed it
        # to ``m_data_table_offset``. The two SEED VALUES differ but the
        # RESULT is identical: the MData walk starts at
        # ``md_offset = m_data_table_offset`` and its guard is
        # ``md_offset + entry_size <= m_data_table_end``. With either
        # seed (0 or m_data_table_offset, which is >= 0x10) the guard is
        # immediately false, so BOTH parsers emit ZERO bones for a BB
        # motion. Not changed: 0 of 1623 shipped motions are BB-form, so
        # any behavioral edit here is unverifiable against real assets.
        m_data_table_end = m_data_table_offset

    # First pass: walk MData entries forward, collecting per-bone
    # (offsets, counts) tuples. Shrink m_data_table_end as we encounter
    # keyframe offsets (v2 only).
    md_entries: List[tuple] = []
    md_offset = m_data_table_offset
    entry_size = 8 * element_count
    # Hard cap: BB skeletons top out around 200 bones. 4096 = generous
    # ceiling that still rejects malformed offsets that would loop.
    MAX_BONES = 4096
    while md_offset + entry_size <= m_data_table_end and len(md_entries) < MAX_BONES:
        # Read element_count offsets, then element_count counts.
        offsets = struct.unpack_from(
            f"<{element_count}I", body, md_offset
        )
        counts = struct.unpack_from(
            f"<{element_count}I", body, md_offset + 4 * element_count
        )
        md_entries.append((offsets, counts))
        md_offset += entry_size

        if v2_format:
            # Shrink the table end if any of THIS bone's offsets points
            # into the payload region — which means the table can't
            # extend past that point.
            for off in offsets:
                if off != 0 and off < m_data_table_end:
                    m_data_table_end = off

    # Second pass: per bone, parse each track's keyframes.
    tracks: List[List[NjmKeyframe]] = []
    bone_present_tracks: List[int] = []
    actual_frame_count = 0
    for offsets, counts in md_entries:
        # Per-bone keyframe list — index by track-type in the order
        # Phantasmal's parseMotion consumes them: POS, ANG, SCL, QUAT.
        per_bone_kf: dict[int, List[NjmKeyframe]] = {}

        # Walk each track type that's enabled in `type`. Phantasmal's
        # `removeFirst()` pops one offset and one count per track type
        # in this fixed order.
        cursor_idx = 0

        def _next_track() -> tuple[int, int]:
            nonlocal cursor_idx
            if cursor_idx >= element_count:
                raise ValueError(
                    f"NJM bone has more enabled tracks than element_count {element_count}"
                )
            off, cnt = offsets[cursor_idx], counts[cursor_idx]
            cursor_idx += 1
            return off, cnt

        positions: List[tuple] = []
        eulers: List[tuple] = []
        scales: List[tuple] = []
        quats: List[tuple] = []
        # Per-bone presence bitfield. Bit set <=> this bone's track of
        # that type was non-empty (count > 0). When the motion-wide
        # type_flag includes POS but THIS bone has count=0 for the POS
        # track, the bit stays unset and consumers fall back to bind.
        present_bits = 0

        if has_pos:
            off, cnt = _next_track()
            if cnt:
                positions = _parse_vector_keyframes(body, off, cnt)
                if positions:
                    present_bits |= NJD_MTYPE_POS
        if has_ang:
            off, cnt = _next_track()
            if cnt:
                eulers = _parse_euler_keyframes(body, off, cnt, frame_count)
                if eulers:
                    present_bits |= NJD_MTYPE_ANG
        if has_scl:
            off, cnt = _next_track()
            if cnt:
                scales = _parse_vector_keyframes(body, off, cnt)
                if scales:
                    present_bits |= NJD_MTYPE_SCL
        if has_quat:
            off, cnt = _next_track()
            if cnt:
                quats = _parse_quaternion_keyframes(body, off, cnt)
                if quats:
                    present_bits |= NJD_MTYPE_QUAT

        # NOTE (audit low, dead path): like Phantasmal's parseMotion
        # ("TODO: all NJD_MTYPE's") this reader has NO branch for
        # NJD_MTYPE_VEC (bit 3). ``njm_writer._enabled_kinds_in_order``
        # DOES list VEC, so for a hypothetical VEC-flagged motion the two
        # would disagree on track ordering: the reader would not pop the
        # VEC slot here, leaving its offset/count attributed to the next
        # enabled kind. UNVERIFIABLE / not fixed: 0 of 1623 shipped
        # motions set bit 3 (observed type_flags ∈ {3,7,0x2001,0x2005}).
        # If VEC support is ever needed, add a ``has_vec`` branch BETWEEN
        # SCL and QUAT (Sega's MData element order is POS,ANG,SCL,VEC,
        # QUAT — see _enabled_kinds_in_order) and parse it as a vector
        # track, keeping both sides in lockstep.

        # Merge tracks into a per-frame keyframe list. We index by
        # frame number — multiple tracks may have keyframes at the same
        # frame, in which case we combine them. Tracks with sparser
        # keyframes get left as separate entries (the consumer
        # interpolates between them at render time).
        #
        # IMPORTANT: when a per-bone track is ABSENT (count=0), the
        # corresponding TRS channel on every keyframe stays at the
        # ``NjmKeyframe`` default — (0,0,0) for translation/rotation,
        # (1,1,1) for scale, None for quaternion. Consumers MUST treat
        # those channels as "use bind pose for this bone" by checking
        # ``NjmMotion.bone_present_tracks[bone_idx]``. Without this
        # guard the rotation-only bones — typical for PSOBB monster
        # animations where only the root bone moves the body and all
        # other bones rotate in place — get yanked to (0,0,0) every
        # frame, collapsing the model into a tangled mess at the origin.
        merged: dict[int, NjmKeyframe] = {}
        for (frame, x, y, z) in positions:
            kf = merged.setdefault(frame, NjmKeyframe(time=frame))
            kf.tx, kf.ty, kf.tz = x, y, z
        for (frame, rx, ry, rz) in eulers:
            kf = merged.setdefault(frame, NjmKeyframe(time=frame))
            kf.rx_bams, kf.ry_bams, kf.rz_bams = rx, ry, rz
        for (frame, sx, sy, sz) in scales:
            kf = merged.setdefault(frame, NjmKeyframe(time=frame))
            kf.sx, kf.sy, kf.sz = sx, sy, sz
        for (frame, qw, qx, qy, qz) in quats:
            kf = merged.setdefault(frame, NjmKeyframe(time=frame))
            kf.qw, kf.qx, kf.qy, kf.qz = qw, qx, qy, qz

        # Sort by time ascending (consumer expects this).
        sorted_keys = sorted(merged.keys())
        keyframes = [merged[k] for k in sorted_keys]
        tracks.append(keyframes)
        bone_present_tracks.append(present_bits)
        if keyframes:
            last_t = keyframes[-1].time
            if last_t + 1 > actual_frame_count:
                actual_frame_count = last_t + 1

    # Use the larger of the header-declared frame_count and the highest
    # observed keyframe-time+1 (some PSOBB motions appear to lie about
    # frame_count slightly).
    final_frame_count = max(frame_count, actual_frame_count)

    return NjmMotion(
        bone_count=len(tracks),
        frame_count=final_frame_count,
        type_flags=type_flags,
        interpolation=interpolation,
        tracks=tracks,
        bone_present_tracks=bone_present_tracks,
    )


def _parse_vector_keyframes(body: bytes, offset: int, count: int) -> List[tuple]:
    """Parse ``count`` (u32 frame, f32 x, f32 y, f32 z) keyframes.

    Returns ``[(frame, x, y, z), ...]``. Out-of-bounds offsets / counts
    are clipped (we return however many keyframes fit).
    """
    out: List[tuple] = []
    pos = offset
    n = len(body)
    for _ in range(count):
        if pos + 16 > n:
            break
        frame, x, y, z = struct.unpack_from("<I3f", body, pos)
        pos += 16
        out.append((frame, x, y, z))
    return out


def _parse_euler_keyframes(
    body: bytes, offset: int, count: int, frame_count: int
) -> List[tuple]:
    """Parse ``count`` BAMS-encoded euler keyframes.

    Tries the narrow (16-bit frame + 16-bit-each angle, total 8 bytes)
    form first; if any frame value would exceed ``frame_count`` or the
    sequence isn't monotonically ascending, falls back to the wide
    (32-bit frame + 32-bit-each angle, total 16 bytes) form.

    The 16-bit angles are read UNSIGNED (``<HHHH``) and stored as the
    raw 0..0xFFFF BAMS integer — we do NOT sign-extend. This matches
    Phantasmal's reference, which reads via ``cursor.uShort()`` and lets
    ``angleToRad`` multiply by 2π/65536; 0xFFFF (= -1 BAMS = -360/65536°
    ≈ -0.0055°) lands on the same point of the unit circle modulo 2π
    whether treated as 65535 or -1. (The wide form below reads SIGNED
    ``<Iiii`` because 32-bit BAMS can legitimately exceed one turn.)

    Returns ``[(frame, rx_bams, ry_bams, rz_bams), ...]`` with raw
    BAMS integers (unsigned 0..0xFFFF in the narrow form, signed in the
    wide form). The model viewer converts to radians via ``r * 2π /
    65536``.
    """
    out: List[tuple] = []
    n = len(body)
    if count == 0:
        return out

    # Try narrow form first.
    pos = offset
    narrow_ok = True
    narrow: List[tuple] = []
    prev_frame = -1
    for _ in range(count):
        if pos + 8 > n:
            narrow_ok = False
            break
        frame, rx, ry, rz = struct.unpack_from("<HHHH", body, pos)
        pos += 8
        if frame < prev_frame or frame >= frame_count:
            narrow_ok = False
            break
        narrow.append((frame, rx, ry, rz))
        prev_frame = frame

    if narrow_ok and len(narrow) == count:
        return narrow

    # Wide form fallback (32-bit each).
    out = []
    pos = offset
    for _ in range(count):
        if pos + 16 > n:
            break
        frame, rx, ry, rz = struct.unpack_from("<Iiii", body, pos)
        pos += 16
        out.append((frame, rx, ry, rz))
    return out


def _parse_quaternion_keyframes(
    body: bytes, offset: int, count: int
) -> List[tuple]:
    """Parse ``count`` (u32 frame, f32 real, f32 ix, f32 iy, f32 iz) keyframes.

    Returns ``[(frame, qw, qx, qy, qz), ...]``. (PSOBB stores
    quaternion as ``(real, imag.x, imag.y, imag.z)``; we surface
    ``qw, qx, qy, qz`` in the same order.)
    """
    out: List[tuple] = []
    pos = offset
    n = len(body)
    for _ in range(count):
        if pos + 20 > n:
            break
        frame, qw, qx, qy, qz = struct.unpack_from("<I4f", body, pos)
        pos += 20
        out.append((frame, qw, qx, qy, qz))
    return out


# ---------------------------------------------------------------------------
# Public: file-level parsers.
# ---------------------------------------------------------------------------


_NMDM_MAGIC = b"NMDM"


def parse_njm(buf: bytes) -> List[NjmMotion]:
    """Parse a complete .njm file (or NMDM IFF chunk PAYLOAD) into motions.

    Accepts either:
      * The full IFF-wrapped file (4-byte ``NMDM`` magic + 4-byte size
        + body + ... + optional ``POF0`` chunk). Most PSOBB.IO motion
        blobs land here — we route through ``formats/iff.py``.
      * A bare NMDM chunk PAYLOAD (no IFF header). Uses the same parse
        logic; the legacy BB form (offset chain at end-of-file) is
        identified by ``buf[:4] != b"NMDM"`` and routed to the BB
        parser. PSOBB.IO doesn't ship BB-form .njm so this branch is
        rarely hit.

    Returns a list of ``NjmMotion`` (always 1 element for v2; BB form
    also currently returns 1). Returns an empty list if the input has
    no recognisable motion data.

    Raises ``ValueError`` on malformed input.
    """
    if not isinstance(buf, (bytes, bytearray, memoryview)):
        raise ValueError("parse_njm: input must be bytes-like")
    if len(buf) < 16:
        return []

    buf_bytes = bytes(buf)
    head4 = buf_bytes[:4]

    if head4 == _NMDM_MAGIC:
        # Full IFF-wrapped file. Walk chunks via parse_iff so we get the
        # NMDM body without manually slicing the size field.
        try:
            chunks = parse_iff(buf_bytes)
        except ValueError as e:
            raise ValueError(f"parse_njm: IFF parse failed: {e}")
        out: List[NjmMotion] = []
        for c in chunks:
            if c.type == "NMDM":
                out.append(_parse_motion(c.data, v2_format=True))
        return out

    # Otherwise either a bare NMDM payload (no header) or a BB-form
    # file. Inspect a bit more to decide.
    #
    # Heuristic: the BB form uses an offset chain at end-of-file. If
    # the buffer ends with what looks like file-relative offsets (the
    # last 16 bytes contain at least one u32 < len(buf)), treat as BB.
    # Otherwise assume bare NMDM payload.
    n = len(buf_bytes)
    if n >= 16:
        # Read u32 at n-16; if it's a plausible offset, try BB.
        offset_at_tail, = struct.unpack_from("<I", buf_bytes, n - 16)
        if 16 <= offset_at_tail < n:
            try:
                return [_parse_njm_bb(buf_bytes)]
            except ValueError:
                # Fall through to bare-payload form.
                pass

    # Bare NMDM payload — try parsing directly.
    try:
        return [_parse_motion(buf_bytes, v2_format=True)]
    except ValueError as e:
        raise ValueError(f"parse_njm: bare-payload parse failed: {e}")


def _parse_njm_bb(buf: bytes) -> NjmMotion:
    """Parse the legacy BB (Blue Burst) .njm form.

    Layout per Phantasmal's ``parseNjmBb``:
        cursor.seekEnd(16)            # 16 bytes from end
        u32 offset1                   # → action chunk start
        cursor.seekStart(offset1)
        u32 actionOffset              # → motion struct start
        cursor.seekStart(actionOffset)
        cursor.seek(4)                # skip 4 bytes
        u32 motionOffset              # → motion body start
        cursor.seekStart(motionOffset)
        return parseMotion(...)

    We do not see this form in PSOBB.IO data; the implementation is
    here for API parity with Phantasmal and to defend against modded
    files that might use it.
    """
    n = len(buf)
    if n < 32:
        raise ValueError(f"BB-form NJM too small: {n} < 32")

    offset1, = struct.unpack_from("<I", buf, n - 16)
    if offset1 < 0 or offset1 + 4 > n:
        raise ValueError(f"BB-form offset1 out of range: 0x{offset1:x}")

    action_offset, = struct.unpack_from("<I", buf, offset1)
    if action_offset < 0 or action_offset + 8 > n:
        raise ValueError(f"BB-form actionOffset out of range: 0x{action_offset:x}")

    # Skip 4 bytes after actionOffset, read motionOffset.
    motion_offset, = struct.unpack_from("<I", buf, action_offset + 4)
    if motion_offset < 0 or motion_offset + 12 > n:
        raise ValueError(f"BB-form motionOffset out of range: 0x{motion_offset:x}")

    # parseMotion expects body bytes starting at motion_offset.
    return _parse_motion(buf[motion_offset:], v2_format=False)


# ---------------------------------------------------------------------------
# FPS heuristic for motion playback.
# ---------------------------------------------------------------------------
#
# Sega Ninja motions are authored against a fixed-tick simulation.
# PSOBB BB runs the world at 30 Hz (see ``psobb_framerate_pacing.md``
# in the user's memory). Most motions therefore look natural at 30 fps,
# but a handful (e.g. cutscene "wait" motions) are authored at lower
# framerates with sparse keyframes — we try to detect and slow those
# down.

_MOTION_FPS_HINTS = {
    # Slow ambient idle/wait motions — typically 5-15 frames at 30 Hz
    # play too fast; render at 15 to give them time. Empirically tuned
    # against PSOBB.IO's NpcApcMot.bml entries.
    "wait": 15.0,
    "idle": 30.0,
    "stand": 30.0,
    # Combat motions — full tick rate.
    "walk": 30.0,
    "run": 30.0,
    "jump": 30.0,
    "atack": 30.0,
    "attack": 30.0,
    "dam": 30.0,
    "dead": 30.0,
    "hoe": 30.0,  # hoeru = "scream/roar"
    # Cinematic appears at variable rates — try 30 as best-effort.
    "apear": 30.0,
    "apper": 30.0,
}


def guess_motion_fps(name: str) -> float:
    """Heuristic FPS for a motion based on its name.

    Returns 30.0 (PSOBB BB tick rate) as the default. Specific
    name-substrings override (e.g. "wait" → 15.0).

    The lookup is case-insensitive and substring-based; the FIRST
    matching key (in dict iteration order) wins.
    """
    if not name:
        return 30.0
    lower = name.lower()
    for key, fps in _MOTION_FPS_HINTS.items():
        if key in lower:
            return fps
    return 30.0


# ---------------------------------------------------------------------------
# Auto-detect "default" motion (walk / run / move / idle / wait / stand).
# ---------------------------------------------------------------------------
#
# The frontend wants the model to start animating as soon as it loads.
# We pick a sensible default by name-substring priority arranged in
# TIERS — every keyword within a tier is treated as equally preferred
# and we take the first match in motion-list order. Higher tiers
# strictly outrank lower tiers regardless of motion-list position.
#
#   Tier 1 — primary locomotion: ``walk`` and its near-synonym
#            ``run``. The canonical "model is moving" verb in player
#            and biped enemy BMLs.
#   Tier 2 — secondary locomotion: ``move``. Many monster BMLs
#            (bm4_ps_*, dragon sub-forms) ship NO walk/run track but
#            DO carry a ``move_*`` motion. Per the 2026-04-25 fix,
#            ``move`` outranks every non-walk alternate.
#   Tier 3 — alternate locomotion: ``swim``, ``fly``, ``frloop``,
#            ``frin`` (boss1 dragon's fly-in / fly-loop, fish-enemy
#            swims). Tracks below ``move`` but above idle so the rare
#            BML that has BOTH ``move`` AND ``fly`` plays the move.
#   Tier 4 — idle / wait / stand. NPCs, props, and other static models
#            tend to ship only an idle pose; better to play the idle
#            than to default to whatever's first (which can be the
#            "damage" or "death" motion in some BMLs, looking very
#            wrong on cold load).
#   Tier 5 — generic last-ditch (`hoe` = monster roar etc.).
#   Tier 6 — fall-through to motion 0.
#
# Per-tier matching means a BML with motions ``["damage", "fly", "move",
# "wait"]`` picks ``move`` (tier 2) over ``fly`` (tier 3), where the
# pre-2026-04-25 flat list would have picked ``fly``.

_DEFAULT_MOTION_PRIORITY_TIERS = (
    # Tier 1 — primary locomotion (walk + run)
    ("walk", "run"),
    # Tier 2 — secondary locomotion (move)
    ("move",),
    # Tier 3 — alternate locomotion (swim / fly variants)
    ("swim", "fly", "frloop", "frin"),
    # Tier 4 — idle / wait / stand fallbacks
    ("idle", "stand", "cstand", "wait"),
    # Tier 5 — generic last-ditch keywords
    ("hoe",),
)


def pick_default_motion(motion_names: List[str]) -> Optional[int]:
    """Return the index of the best "default" motion in ``motion_names``.

    Picks by tiered substring priority. Returns ``None`` if the list
    is empty; returns ``0`` if no name matches any priority keyword
    (= "use whatever's first").

    Case-insensitive comparison.

    Tier order (highest first):
        1. ``walk`` / ``run``
        2. ``move``
        3. ``swim`` / ``fly`` / ``frloop`` / ``frin``
        4. ``idle`` / ``stand`` / ``cstand`` / ``wait``
        5. ``hoe``
        6. fall-through to index 0

    History:
        Pre-2026-04-25 used a flat priority list which meant ``move``
        was outranked by ``swim`` / ``fly`` / ``frloop`` even when no
        locomotion matched. The tiered walk fixes this so a BML whose
        only locomotion verb is ``move_*`` (e.g. bm4_ps_*, monster
        sub-forms, set pieces) auto-plays correctly.
    """
    if not motion_names:
        return None
    lowers = [n.lower() for n in motion_names]
    for tier in _DEFAULT_MOTION_PRIORITY_TIERS:
        for i, name in enumerate(lowers):
            for keyword in tier:
                if keyword in name:
                    return i
    return 0


# ---------------------------------------------------------------------------
# Fast header-only parser for listing endpoints.
# ---------------------------------------------------------------------------


@dataclass
class NjmHeader:
    """NJM header summary — bone/frame count + type flags only.

    Same shape as NjmMotion but with empty tracks. Returned by
    ``parse_njm_header_only`` for callers that need to enumerate
    motions in a BML without paying the full keyframe-decode cost
    (linear in keyframe count; the dragon's walk has ~3000 keyframes).
    """
    bone_count: int = 0
    frame_count: int = 0
    type_flags: int = 0
    interpolation: int = 0


def parse_njm_header_only(buf: bytes) -> Optional[NjmHeader]:
    """Read just enough of an NJM file to populate an ``NjmHeader``.

    Returns None if the input doesn't have a recognisable NMDM header.
    Raises ValueError on hard parse failure (bad IFF wrapper, etc.).

    Bone count is computed by walking the MData table forward,
    counting entries until we hit the first keyframe-offset that
    points into the table (the v2 shrink-as-you-go trick); we don't
    decode any keyframes.
    """
    if not isinstance(buf, (bytes, bytearray, memoryview)):
        raise ValueError("parse_njm_header_only: input must be bytes-like")
    if len(buf) < 16:
        return None
    buf_bytes = bytes(buf)
    if buf_bytes[:4] == _NMDM_MAGIC:
        try:
            chunks = parse_iff(buf_bytes)
        except ValueError:
            return None
        for c in chunks:
            if c.type == "NMDM":
                return _read_njm_header(c.data)
        return None
    # Bare body fallback.
    return _read_njm_header(buf_bytes)


def _read_njm_header(body: bytes) -> Optional[NjmHeader]:
    """Read just the header + walk MData table (no keyframes)."""
    n = len(body)
    if n < 12:
        return None
    m_data_table_offset, frame_count, type_flags, inp_fn = struct.unpack_from(
        "<II HH", body, 0
    )
    interpolation = (inp_fn & 0b1100_0000) >> 6
    element_count = inp_fn & 0b1111
    if element_count == 0 or element_count > 8:
        return NjmHeader(
            bone_count=0, frame_count=frame_count,
            type_flags=type_flags, interpolation=interpolation,
        )
    if m_data_table_offset < 12 or m_data_table_offset > n:
        return None
    # Walk MData entries to count bones — same logic as
    # _parse_motion's first pass but no keyframe parsing.
    m_data_table_end = n
    md_offset = m_data_table_offset
    entry_size = 8 * element_count
    bone_count = 0
    MAX_BONES = 4096
    while md_offset + entry_size <= m_data_table_end and bone_count < MAX_BONES:
        offsets = struct.unpack_from(
            f"<{element_count}I", body, md_offset
        )
        bone_count += 1
        md_offset += entry_size
        for off in offsets:
            if off != 0 and off < m_data_table_end:
                m_data_table_end = off
    return NjmHeader(
        bone_count=bone_count,
        frame_count=frame_count,
        type_flags=type_flags,
        interpolation=interpolation,
    )


__all__ = [
    "NjmKeyframe",
    "NjmMotion",
    "NjmHeader",
    "parse_njm",
    "parse_njm_header_only",
    "guess_motion_fps",
    "pick_default_motion",
]
