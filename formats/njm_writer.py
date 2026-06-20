"""NJM (Ninja Motion v2 / NMDM) encoder for PSOBB Blue Burst.

Inverse of ``formats.njm.parse_njm``. Round-trips byte-exact for the
~1700 ``.njm`` files in PSOBB.IO.

Wire layout (NMDM body, all little-endian):

    +----------------------------------------------------+
    | 0x00  u32 m_data_table_offset (always 0xC)         |
    | 0x04  u32 frame_count                              |
    | 0x08  u16 type_flags                               |
    | 0x0A  u16 inp_fn  (high 2 bits = interpolation,    |
    |                    low 4 bits = element_count)     |
    +----------------------------------------------------+
    | 0x0C  MData table — element_count u32 offsets +    |
    |       element_count u32 counts per bone, packed    |
    |       back-to-back. Empty bones still get their    |
    |       slot (offsets=0, counts=0).                  |
    +----------------------------------------------------+
    | <table_end>  Keyframe blobs, packed in MData       |
    |              order (bone 0 POS, bone 0 ANG, ...).  |
    |              Each blob is per-track:               |
    |   POS / SCL: count * (u32 frame, 3*f32)            |
    |   ANG narrow: count * (u16 frame, 3*u16 BAMS)      |
    |   ANG wide  : count * (u32 frame, 3*i32 BAMS)      |
    |   QUAT       : count * (u32 frame, 4*f32)          |
    +----------------------------------------------------+

Plus IFF wrapper (NMDM tag + size). PSOBB also emits a sibling POF0
chunk in some motions; this writer emits an empty-pointer POF0 (0
bytes) by default and the parser is fine with it. For round-trip
byte-exact we capture and re-emit the source POF0 verbatim.

Type-flag support:
  * 100% of shipped NJMs use type_flags in {3, 7, 0x2001, 0x2005}.
  * Bit 0 = POS, bit 1 = ANG, bit 2 = SCL, bit 13 = QUAT.

Narrow vs wide euler form:
  * Narrow (8 bytes/keyframe): used when frames fit in u16 AND are
    monotonically ascending — ~94% of shipped tracks.
  * Wide (16 bytes/keyframe): used when narrow constraints fail.
  The writer auto-picks narrow when possible; the round-trippable
  parser captures source layout to preserve the chosen form.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .iff import parse_iff
from .njm import (
    NjmKeyframe,
    NjmMotion,
    NJD_MTYPE_POS, NJD_MTYPE_ANG, NJD_MTYPE_SCL, NJD_MTYPE_VEC, NJD_MTYPE_QUAT,
)


# ---------------------------------------------------------------------------
# Round-trip-friendly motion model
# ---------------------------------------------------------------------------
#
# The parsing direction in formats/njm.py merges all per-bone tracks
# into a single keyframe list per bone. That's lossy for round-trip
# (the writer can't reconstruct WHICH channels existed at each
# keyframe). So we keep a separate per-track dataclass that captures
# the wire-format layout precisely.


@dataclass
class NjmTrack:
    """One track for one bone (POS / ANG / SCL / QUAT).

    ``kind`` is one of NJD_MTYPE_* constants. Each list holds tuples
    matching the wire-format keyframe shape:
        POS / SCL : (frame: int, x: float, y: float, z: float)
        ANG narrow: (frame: int, rx: int, ry: int, rz: int) — 16-bit
                    frames + 16-bit BAMS angles
        ANG wide  : (frame: int, rx: int, ry: int, rz: int) — 32-bit
                    frames + 32-bit BAMS angles
        QUAT      : (frame: int, qw: float, qx: float, qy: float,
                     qz: float)

    ``narrow`` is True for ANG tracks stored in the 8-byte form,
    False for the 16-byte form. Ignored for non-ANG tracks.

    ``stored_count`` (when set) overrides ``len(keyframes)`` in the
    MData table — used for byte-exact round-trip of files where the
    source count doesn't match the actual number of keyframes that
    fit in the buffer (rare; ~8 boss-tentacle files in PSOBB.IO).
    None means "use len(keyframes)".
    """
    kind: int
    keyframes: List[Tuple] = field(default_factory=list)
    narrow: bool = True
    stored_count: Optional[int] = None


@dataclass
class NjmBoneTracks:
    """Per-bone track collection; one slot per channel in element order.

    ``tracks_by_kind`` maps NJD_MTYPE_* → NjmTrack. Empty when a bone
    has no animation (idle bones in the source).
    """
    tracks_by_kind: Dict[int, NjmTrack] = field(default_factory=dict)


@dataclass
class NjmRawMotion:
    """A round-trip-preserving NJM motion.

    Read with ``parse_njm_for_writer``, edited freely, and re-emitted
    with ``encode_njm`` to round-trip byte-exact.

    ``track_offset_hint`` (when present) maps (bone_idx, kind) →
    explicit byte offset for that track's keyframe blob. Used for
    byte-exact round-trip when the source padded between blobs.
    Synthetic motions leave it None and accept the default packed
    layout.

    ``trailing_size_hint`` (when present) overrides the body size,
    forcing the encoder to pad the body to that exact length. Used
    in conjunction with ``track_offset_hint`` for byte-exact files
    that include trailing zero bytes.

    ``source_body`` (when present) is the verbatim NMDM body of the
    source file. The encoder uses it as a starting buffer and overlays
    only the regions actually emitted (header + MData table + each
    track blob). This preserves the "uninitialized" memory the SEGA
    authoring tool left between tracks in NpcApcMot.bml's motions —
    bytes that look like leftover keyframe data but aren't referenced
    by any bone's MData.
    """
    frame_count: int = 0
    type_flags: int = 0
    inp_fn: int = 0  # raw u16 (interpolation high bits + element_count low 4)
    m_data_table_offset: int = 0xC
    bones: List[NjmBoneTracks] = field(default_factory=list)
    # Source POF0 verbatim, for byte-exact IFF round-trip. Empty when
    # the source had no POF0 sibling; the writer emits b"" in that case.
    pof0_bytes: bytes = b""
    # Optional layout hints for byte-exact round-trip.
    track_offset_hint: Optional[Dict[Tuple[int, int], int]] = None
    trailing_size_hint: Optional[int] = None
    source_body: Optional[bytes] = None


# ---------------------------------------------------------------------------
# Public alias for callers that prefer to round-trip via an
# ``NjmMotion`` (the simpler / lossy parser type) — encode_njm accepts
# both.


# ---------------------------------------------------------------------------
# Parser path: produces an NjmRawMotion suitable for re-emission.
# ---------------------------------------------------------------------------


def _enabled_kinds_in_order(type_flags: int) -> List[int]:
    """Return kinds (NJD_MTYPE_*) in MData entry order: POS, ANG, SCL, VEC, QUAT.

    Phantasmal's parser walks the MData track offsets in this fixed
    order (one offset per enabled kind, popped sequentially); we mirror
    that.

    NOTE (audit low, dead path): this writer lists NJD_MTYPE_VEC (bit 3)
    but the reader ``njm._parse_motion`` / Phantasmal ``parseMotion`` do
    NOT — so a VEC-flagged motion would round-trip here yet decode with a
    track-order shift on the reading side. UNVERIFIABLE: 0 of 1623
    shipped motions set bit 3 (type_flags ∈ {3,7,0x2001,0x2005}). Kept as
    is to preserve the writer's 100% byte-exact round-trip; if VEC is ever
    needed, add the matching ``has_vec`` branch to njm._parse_motion in
    lockstep (between SCL and QUAT) before relying on it.
    """
    kinds: List[int] = []
    if type_flags & NJD_MTYPE_POS:
        kinds.append(NJD_MTYPE_POS)
    if type_flags & NJD_MTYPE_ANG:
        kinds.append(NJD_MTYPE_ANG)
    if type_flags & NJD_MTYPE_SCL:
        kinds.append(NJD_MTYPE_SCL)
    if type_flags & NJD_MTYPE_VEC:
        kinds.append(NJD_MTYPE_VEC)
    if type_flags & NJD_MTYPE_QUAT:
        kinds.append(NJD_MTYPE_QUAT)
    return kinds


def _parse_pos_or_scl(body: bytes, offset: int, count: int) -> List[Tuple]:
    out: List[Tuple] = []
    pos = offset
    n = len(body)
    for _ in range(count):
        if pos + 16 > n:
            break
        frame, x, y, z = struct.unpack_from("<I3f", body, pos)
        pos += 16
        out.append((frame, x, y, z))
    return out


def _parse_ang_narrow(
    body: bytes, offset: int, count: int, frame_count: int
) -> Optional[List[Tuple]]:
    """Try parsing ``count`` narrow (8-byte) ANG keyframes.

    Returns None if the data doesn't fit the narrow shape (frames not
    monotonic, off the end of buffer, OR any frame value lands at-or-
    beyond ``frame_count``); the caller should retry wide.

    The ``frame >= frame_count`` rejection is a faithful port of the
    reading-side parser ``formats.njm._parse_euler_keyframes`` (line
    ``if frame < prev_frame or frame >= frame_count``) and Phantasmal's
    ``parseEulerAngleKeyframes`` (Motion.kt:238,
    ``keyframe.frame < prev || keyframe.frame >= frameCount``). Without
    it, a WIDE 16-byte track whose first u16 pair happens to be
    monotonic decodes as narrow garbage (e.g. ``pxuG01_A06_F_body.njm``
    fc=20: narrow reads frame 56043 / angle 65535 where the true wide
    decode is frame 19 / angle -9493). Byte-exact round-trip is
    UNAFFECTED — the encoder re-emits the source bytes verbatim via
    ``source_body`` regardless of narrow/wide — but the DECODED
    keyframes feed ``anim_blend``/``anim_retarget``, so picking the
    correct form is required for non-corrupt blends/retargets.
    """
    out: List[Tuple] = []
    pos = offset
    n = len(body)
    prev = -1
    for _ in range(count):
        if pos + 8 > n:
            return None
        frame, rx, ry, rz = struct.unpack_from("<HHHH", body, pos)
        pos += 8
        if frame < prev or frame >= frame_count:
            return None
        prev = frame
        out.append((frame, rx, ry, rz))
    return out


def _parse_ang_wide(body: bytes, offset: int, count: int) -> List[Tuple]:
    out: List[Tuple] = []
    pos = offset
    n = len(body)
    for _ in range(count):
        if pos + 16 > n:
            break
        frame, rx, ry, rz = struct.unpack_from("<Iiii", body, pos)
        pos += 16
        out.append((frame, rx, ry, rz))
    return out


def _parse_quat(body: bytes, offset: int, count: int) -> List[Tuple]:
    out: List[Tuple] = []
    pos = offset
    n = len(body)
    for _ in range(count):
        if pos + 20 > n:
            break
        frame, qw, qx, qy, qz = struct.unpack_from("<I4f", body, pos)
        pos += 20
        out.append((frame, qw, qx, qy, qz))
    return out


def parse_njm_for_writer(buf: bytes) -> NjmRawMotion:
    """Parse a complete .njm (IFF-wrapped) into a round-trippable motion.

    Captures every wire-format detail (narrow vs wide euler, source
    POF0 bytes, raw inp_fn) so ``encode_njm`` can byte-exact round-trip.

    Returns an empty motion when the input has no NMDM chunk.

    Raises ValueError on truncated / malformed input.
    """
    if not isinstance(buf, (bytes, bytearray, memoryview)):
        raise ValueError("parse_njm_for_writer: input must be bytes-like")
    chunks = parse_iff(bytes(buf))
    nmdm = next((c for c in chunks if c.type == "NMDM"), None)
    if nmdm is None:
        return NjmRawMotion()
    pof0 = next((c for c in chunks if c.type == "POF0"), None)
    pof0_bytes = bytes(pof0.data) if pof0 is not None else b""

    body = nmdm.data
    n = len(body)
    if n < 12:
        raise ValueError(f"NJM body too small: {n} < 12")

    m_data_table_offset, frame_count, type_flags, inp_fn = struct.unpack_from(
        "<II HH", body, 0
    )
    element_count = inp_fn & 0xF

    motion = NjmRawMotion(
        frame_count=frame_count,
        type_flags=type_flags,
        inp_fn=inp_fn,
        m_data_table_offset=m_data_table_offset,
        pof0_bytes=pof0_bytes,
        track_offset_hint={},
        trailing_size_hint=n,
        source_body=bytes(body),
    )

    if element_count == 0 or element_count > 8:
        return motion

    kinds_in_order = _enabled_kinds_in_order(type_flags)
    if len(kinds_in_order) != element_count:
        # Some shipped data has element_count != popcount(type_flags).
        # Phantasmal's parser still works (it pops element_count
        # offsets); the writer must replicate exactly. Use whichever
        # is smaller — this is rare anyway.
        # We honor element_count; truncate kinds_in_order if needed.
        kinds_in_order = kinds_in_order[:element_count]

    # Walk MData entries to find the table end (Phantasmal's
    # "shrink-as-you-go" trick).
    md_offset = m_data_table_offset
    m_data_table_end = n
    entry_size = 8 * element_count
    md_entries: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
    MAX_BONES = 4096
    while (
        md_offset + entry_size <= m_data_table_end
        and len(md_entries) < MAX_BONES
    ):
        offsets = struct.unpack_from(f"<{element_count}I", body, md_offset)
        counts = struct.unpack_from(
            f"<{element_count}I", body, md_offset + 4 * element_count
        )
        md_entries.append((offsets, counts))
        md_offset += entry_size
        for off in offsets:
            if off != 0 and off < m_data_table_end:
                m_data_table_end = off

    # For each bone, parse each enabled track.
    for bone_idx, (offsets, counts) in enumerate(md_entries):
        bone = NjmBoneTracks()
        for slot, kind in enumerate(kinds_in_order):
            off = offsets[slot]
            cnt = counts[slot]
            if cnt == 0 or off == 0:
                # Empty track. We still record the kind in the bone
                # so the writer emits offset=0, count=0 in the same
                # slot.
                bone.tracks_by_kind[kind] = NjmTrack(kind=kind, keyframes=[], narrow=True)
                continue
            # Capture source offset for byte-exact round-trip.
            motion.track_offset_hint[(bone_idx, kind)] = off
            if kind in (NJD_MTYPE_POS, NJD_MTYPE_SCL):
                kfs = _parse_pos_or_scl(body, off, cnt)
                bone.tracks_by_kind[kind] = NjmTrack(
                    kind=kind, keyframes=kfs, narrow=True, stored_count=cnt,
                )
            elif kind == NJD_MTYPE_ANG:
                # Try narrow first; fall back to wide. Pass frame_count
                # so the narrow probe rejects frame values at-or-beyond
                # it (matches njm._parse_euler_keyframes / Motion.kt:238)
                # — otherwise wide tracks decode as narrow garbage.
                narrow = _parse_ang_narrow(body, off, cnt, frame_count)
                if narrow is not None and len(narrow) == cnt:
                    bone.tracks_by_kind[kind] = NjmTrack(
                        kind=kind, keyframes=narrow, narrow=True, stored_count=cnt,
                    )
                else:
                    wide = _parse_ang_wide(body, off, cnt)
                    bone.tracks_by_kind[kind] = NjmTrack(
                        kind=kind, keyframes=wide, narrow=False, stored_count=cnt,
                    )
            elif kind == NJD_MTYPE_QUAT:
                kfs = _parse_quat(body, off, cnt)
                bone.tracks_by_kind[kind] = NjmTrack(
                    kind=kind, keyframes=kfs, narrow=True, stored_count=cnt,
                )
            elif kind == NJD_MTYPE_VEC:
                # VEC is rare / unused in PSOBB; keep raw bytes-by-frame.
                # Encoded as POS (16 bytes per keyframe) per Phantasmal.
                kfs = _parse_pos_or_scl(body, off, cnt)
                bone.tracks_by_kind[kind] = NjmTrack(
                    kind=kind, keyframes=kfs, narrow=True, stored_count=cnt,
                )
        motion.bones.append(bone)

    return motion


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def _encode_track_blob(track: NjmTrack) -> bytes:
    """Encode a single track's keyframe list to wire-format bytes."""
    out = bytearray()
    if track.kind in (NJD_MTYPE_POS, NJD_MTYPE_SCL, NJD_MTYPE_VEC):
        for kf in track.keyframes:
            frame, x, y, z = kf
            out.extend(struct.pack("<I3f", int(frame), float(x), float(y), float(z)))
    elif track.kind == NJD_MTYPE_ANG:
        if track.narrow:
            for kf in track.keyframes:
                frame, rx, ry, rz = kf
                # Narrow: u16 frame + 3*u16 angles. Mask to u16 to
                # accept either signed or unsigned integers.
                out.extend(struct.pack(
                    "<HHHH",
                    int(frame) & 0xFFFF,
                    int(rx) & 0xFFFF,
                    int(ry) & 0xFFFF,
                    int(rz) & 0xFFFF,
                ))
        else:
            for kf in track.keyframes:
                frame, rx, ry, rz = kf
                out.extend(struct.pack(
                    "<Iiii",
                    int(frame) & 0xFFFFFFFF,
                    int(rx),
                    int(ry),
                    int(rz),
                ))
    elif track.kind == NJD_MTYPE_QUAT:
        for kf in track.keyframes:
            frame, qw, qx, qy, qz = kf
            out.extend(struct.pack(
                "<I4f",
                int(frame), float(qw), float(qx), float(qy), float(qz),
            ))
    else:
        raise ValueError(f"_encode_track_blob: unsupported kind {track.kind}")
    return bytes(out)


def encode_njm(motion: NjmRawMotion) -> bytes:
    """Encode an ``NjmRawMotion`` to a complete .njm file (IFF-wrapped).

    Returns the bytes of an IFF-wrapped NMDM (+ POF0 if the source
    had one). Raises ValueError on inconsistent input.
    """
    kinds_in_order = _enabled_kinds_in_order(motion.type_flags)
    element_count = motion.inp_fn & 0xF

    # Compute MData table size + per-bone keyframe blobs.
    n_bones = len(motion.bones)
    mdt_off = motion.m_data_table_offset
    if mdt_off < 12:
        mdt_off = 12
    table_size = n_bones * 8 * element_count
    table_end = mdt_off + table_size

    # Two layout strategies:
    #   1. Hint-driven: use motion.track_offset_hint for each track's
    #      explicit byte offset. This preserves the source's gaps /
    #      padding for byte-exact round-trip.
    #   2. Packed: sequentially emit blobs starting at table_end. New
    #      synthetic motions use this path.
    use_hint = (
        motion.track_offset_hint is not None and len(motion.track_offset_hint) > 0
    )

    # bone_slot_offsets[b][slot] = (offset, count). count=0 means empty.
    bone_slot_offsets: List[List[Tuple[int, int]]] = []

    if use_hint:
        # Allocate body up to trailing_size_hint (else compute from
        # max track end). Place each blob at its hinted offset.
        track_blobs: Dict[Tuple[int, int], bytes] = {}
        max_end = table_end
        for bi, bone in enumerate(motion.bones):
            slot_info = []
            for kind in kinds_in_order:
                track = bone.tracks_by_kind.get(kind)
                if track is None or not track.keyframes:
                    slot_info.append((0, 0))
                    continue
                # Use hint if available; else default to packed (after
                # the previously-known max_end).
                hint_off = motion.track_offset_hint.get((bi, kind))
                if hint_off is None:
                    hint_off = max_end
                blob = _encode_track_blob(track)
                track_blobs[(bi, kind)] = blob
                stored_cnt = (
                    track.stored_count
                    if track.stored_count is not None
                    else len(track.keyframes)
                )
                slot_info.append((hint_off, stored_cnt))
                if hint_off + len(blob) > max_end:
                    max_end = hint_off + len(blob)
            bone_slot_offsets.append(slot_info)
        body_size = (
            motion.trailing_size_hint
            if motion.trailing_size_hint and motion.trailing_size_hint >= max_end
            else max_end
        )
        # Start from source_body (preserves uninitialized "filler"
        # bytes between tracks); fall back to zeros for synthetic
        # round-trip cases where source_body is missing.
        if motion.source_body is not None and len(motion.source_body) >= body_size:
            body = bytearray(motion.source_body[:body_size])
        else:
            body = bytearray(body_size)
        # Write header.
        struct.pack_into(
            "<II HH", body, 0,
            mdt_off, motion.frame_count, motion.type_flags, motion.inp_fn,
        )
        # Write MData table.
        cursor = mdt_off
        for slot_info in bone_slot_offsets:
            for (off, _cnt) in slot_info:
                struct.pack_into("<I", body, cursor, off); cursor += 4
            for _ in range(element_count - len(slot_info)):
                struct.pack_into("<I", body, cursor, 0); cursor += 4
            for (_off, cnt) in slot_info:
                struct.pack_into("<I", body, cursor, cnt); cursor += 4
            for _ in range(element_count - len(slot_info)):
                struct.pack_into("<I", body, cursor, 0); cursor += 4
        # Write blobs at their hinted offsets.
        for (bi, kind), blob in track_blobs.items():
            for slot, k in enumerate(kinds_in_order):
                if k == kind:
                    off = bone_slot_offsets[bi][slot][0]
                    body[off:off + len(blob)] = blob
                    break
    else:
        # Packed (synthetic) layout.
        cursor = table_end
        blob_buffer = bytearray()
        for bone in motion.bones:
            slot_info = []
            for kind in kinds_in_order:
                track = bone.tracks_by_kind.get(kind)
                if track is None or not track.keyframes:
                    slot_info.append((0, 0))
                    continue
                blob = _encode_track_blob(track)
                stored_cnt = (
                    track.stored_count
                    if track.stored_count is not None
                    else len(track.keyframes)
                )
                slot_info.append((cursor, stored_cnt))
                blob_buffer.extend(blob)
                cursor += len(blob)
            bone_slot_offsets.append(slot_info)

        body = bytearray()
        # Header.
        body.extend(struct.pack(
            "<II HH",
            mdt_off, motion.frame_count, motion.type_flags, motion.inp_fn,
        ))
        while len(body) < mdt_off:
            body.append(0)

        for slot_info in bone_slot_offsets:
            for (off, _cnt) in slot_info:
                body.extend(struct.pack("<I", off))
            for _ in range(element_count - len(slot_info)):
                body.extend(struct.pack("<I", 0))
            for (_off, cnt) in slot_info:
                body.extend(struct.pack("<I", cnt))
            for _ in range(element_count - len(slot_info)):
                body.extend(struct.pack("<I", 0))

        body.extend(blob_buffer)

    # IFF wrap.
    out = bytearray()
    out.extend(b"NMDM")
    out.extend(struct.pack("<I", len(body)))
    out.extend(body)
    if motion.pof0_bytes:
        out.extend(b"POF0")
        out.extend(struct.pack("<I", len(motion.pof0_bytes)))
        out.extend(motion.pof0_bytes)

    return bytes(out)


# ---------------------------------------------------------------------------
# Convenience: encode_njm_from_njmotion
# ---------------------------------------------------------------------------
#
# For callers that only have an ``NjmMotion`` (the parser type from
# ``formats.njm``) — e.g. server endpoints that decode user JSON into
# NjmMotion. We re-derive per-track keyframes from the merged
# representation using the bone_present_tracks bitmask.


def njmotion_to_raw(merged: NjmMotion) -> NjmRawMotion:
    """Convert a parsed ``NjmMotion`` to round-trip-friendly form.

    Loses some information (the source's narrow-vs-wide euler choice
    and POF0) — fall back to safe defaults: narrow euler if all
    frames + angles fit in u16 / i16 range, no POF0.
    """
    raw = NjmRawMotion(
        frame_count=merged.frame_count,
        type_flags=merged.type_flags,
        inp_fn=(merged.interpolation << 6) | bin(merged.type_flags & 0x200F).count("1"),
    )
    kinds_in_order = _enabled_kinds_in_order(merged.type_flags)

    for bone_idx, kfs in enumerate(merged.tracks):
        bone = NjmBoneTracks()
        present_mask = (
            merged.bone_present_tracks[bone_idx]
            if bone_idx < len(merged.bone_present_tracks)
            else 0
        )
        # Group keyframes by which channel they actually contributed to.
        # NjmKeyframe stores all four channels regardless of which were
        # originally authored; we use ``present_mask`` (set per-bone) to
        # decide which channels to emit.
        pos_kfs: List[Tuple] = []
        ang_kfs: List[Tuple] = []
        scl_kfs: List[Tuple] = []
        quat_kfs: List[Tuple] = []
        for kf in kfs:
            if present_mask & NJD_MTYPE_POS:
                pos_kfs.append((kf.time, kf.tx, kf.ty, kf.tz))
            if present_mask & NJD_MTYPE_ANG:
                ang_kfs.append((kf.time, kf.rx_bams, kf.ry_bams, kf.rz_bams))
            if present_mask & NJD_MTYPE_SCL:
                scl_kfs.append((kf.time, kf.sx, kf.sy, kf.sz))
            if present_mask & NJD_MTYPE_QUAT and kf.qw is not None:
                quat_kfs.append((kf.time, kf.qw, kf.qx, kf.qy, kf.qz))

        for kind in kinds_in_order:
            if kind == NJD_MTYPE_POS:
                bone.tracks_by_kind[kind] = NjmTrack(kind, pos_kfs, narrow=True)
            elif kind == NJD_MTYPE_ANG:
                # Pick narrow if all frames fit in u16 and angles fit
                # in u16 (positive 0..0xFFFF wrap). Otherwise wide.
                narrow_ok = all(
                    0 <= kf[0] < 0x10000 and
                    0 <= (kf[1] & 0xFFFF) < 0x10000 and
                    0 <= (kf[2] & 0xFFFF) < 0x10000 and
                    0 <= (kf[3] & 0xFFFF) < 0x10000
                    for kf in ang_kfs
                )
                bone.tracks_by_kind[kind] = NjmTrack(kind, ang_kfs, narrow=narrow_ok)
            elif kind == NJD_MTYPE_SCL:
                bone.tracks_by_kind[kind] = NjmTrack(kind, scl_kfs, narrow=True)
            elif kind == NJD_MTYPE_QUAT:
                bone.tracks_by_kind[kind] = NjmTrack(kind, quat_kfs, narrow=True)
            elif kind == NJD_MTYPE_VEC:
                bone.tracks_by_kind[kind] = NjmTrack(kind, [], narrow=True)

        raw.bones.append(bone)

    return raw


__all__ = [
    "NjmTrack",
    "NjmBoneTracks",
    "NjmRawMotion",
    "encode_njm",
    "parse_njm_for_writer",
    "njmotion_to_raw",
]
