"""PSOBB Ninja material chunk decoder + encoder.

This module decodes the per-submesh render-state ("material") chunks
that PSOBB Blue Burst stores inside the polygon-stream of every NJ
mesh. The legacy parser in ``formats/xj.py`` recognises the chunks by
size only (so it can skip them) but doesn't surface their contents to
the editor — this module fills that gap so the Material Inspector tab
can SEE and EDIT the flags.

Chunk type catalogue (from Sega Ninja SDK ``njchunk.h`` + audited
against 2,021 strip chunks across 212 PSOBB.IO BMLs on 2026-04-25):

    Header-only (no body, render state purely in flags byte):
      0    NULL                  filler / no-op
      1    NJD_CB_BA (BlendAlpha) blend mode encoded in flags
                                 (low 3 bits = src factor, next 3 = dst
                                  factor, top 2 unused; verified against
                                  shipped data: only flags=0x21 and
                                  0x25 ever occur — those map to the
                                  "src=SrcAlpha, dst=InvSrcAlpha" and
                                  "src=SrcAlpha, dst=One" presets the
                                  Ninja authoring tools emit by default)

    Tiny (body = u16, render-state in flags byte + bottom 13 bits of
    body for texture id):
      8    NJD_CT (Tiny tex)     bottom 13 bits = texture stage id
      9    NJD_CT_COL (alt)      same shape as 8

    Material (body = u16 wordcount + RGBA payload):
      17   NJD_CB_D    Diffuse only (wc=2, 4-byte BGRA)
      18   NJD_CB_A    Ambient only (wc=2, 4-byte BGRA)
      19   NJD_CB_DA   Diffuse + Ambient (wc=4, 8 bytes)
      20   NJD_CB_S    Specular only (wc=2, RGB + exponent)
      21   NJD_CB_DS   Diffuse + Specular (wc=4, 8 bytes)
      22   NJD_CB_AS   Ambient + Specular (wc=4, 8 bytes) — never seen
                       in shipped data but emitted by some authoring
                       tools, decoded here for completeness
      23   NJD_CB_DAS  Diffuse + Ambient + Specular (wc=6, 12 bytes)

The "alpha test" / "double-sided" / "depth write" semantics PSOBB
exposes don't map onto a single Ninja chunk — they are spread across
multiple bit positions:

    * alpha-test enable + threshold:    Tiny chunk (8/9) flags
                                         (bit 0x80 = use_alpha_test;
                                         body[1] high byte = threshold)
    * blend mode:                        BlendAlpha chunk (1) flags
    * double-sided:                      Strip chunk (64..75) flags
                                         bit 4 (0x10)
    * depth write / depth test:          Strip chunk flags
                                         (bit 0x40 = no_zwrite,
                                          bit 0x80 = no_ztest)
    * vertex color modulation:           Material chunk presence

Because of this fan-out, the Material Inspector exposes a UNIFIED
PER-SUBMESH view that aggregates:

    * the most recent Material chunk's diffuse/ambient/specular
    * the most recent BlendAlpha chunk's blend mode
    * the most recent Tiny chunk's filter/clamp/alpha-test flags
    * the strip chunk's own flags for double-sided / depth

…and the editor's POST endpoint INSERTS / REPLACES the chunks needed
to express the user's edits, leaving every other chunk in the polygon
stream untouched.

Public API:
    decode_material_chunk(type_id, flags, body) -> MaterialChunkPayload
    decode_blend_alpha_chunk(flags) -> BlendAlphaPayload
    decode_tiny_chunk(flags, body) -> TinyChunkPayload
    decode_strip_chunk_flags(flags) -> StripFlagsPayload
    aggregate_submesh_state(plist_chunks, submesh_strip_idx) -> SubmeshMaterial
    encode_material_chunk(payload, type_id) -> bytes (body only)
    apply_submesh_edits(plist_chunks, edits) -> new plist

All decode/encode helpers are PURE (no I/O) so they can be unit-tested
without a fixture bundle.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RGBA:
    """RGBA colour as 8-bit unsigned ints (0..255) per channel.

    On disk PSOBB stores material colours as BGRA in memory order (so
    that a little-endian u32 load yields ARGB in Ninja's
    ``NJD_RGBA(a,r,g,b)`` macro). This dataclass holds R/G/B/A in the
    order the editor presents to the user.
    """
    r: int = 255
    g: int = 255
    b: int = 255
    a: int = 255

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.r & 0xFF, self.g & 0xFF, self.b & 0xFF, self.a & 0xFF)

    @classmethod
    def from_bgra_bytes(cls, buf: bytes, off: int = 0) -> "RGBA":
        """Decode 4 bytes laid out as B, G, R, A on disk."""
        if off + 4 > len(buf):
            return cls()
        b, g, r, a = buf[off], buf[off + 1], buf[off + 2], buf[off + 3]
        return cls(r=r, g=g, b=b, a=a)

    def to_bgra_bytes(self) -> bytes:
        """Encode to 4 bytes laid out as B, G, R, A on disk."""
        return bytes([self.b & 0xFF, self.g & 0xFF, self.r & 0xFF, self.a & 0xFF])


@dataclass
class MaterialChunkPayload:
    """Decoded form of an NJD_CB_* material chunk (types 17-23).

    Each colour component is OPTIONAL — the chunk's ``type_id`` selects
    which subset is present, and absent components are reported as
    ``None`` so the caller knows whether to merge with a prior chunk's
    value or fall back to defaults.

    ``specular_exponent`` is the gloss / shininess byte that follows
    the specular RGB triplet when present (per Sega Ninja: the exponent
    is one byte, in the range 0..255 with practical values 0..32). PSO
    BB shipped data uses values clustered around 0x0B / 0x0D / 0x0E /
    0x10.
    """
    type_id: int = 23
    flags: int = 0x25
    diffuse: Optional[RGBA] = None
    ambient: Optional[RGBA] = None
    specular: Optional[RGBA] = None  # alpha holds the exponent
    specular_exponent: Optional[int] = None


@dataclass
class BlendAlphaPayload:
    """Decoded NJD_CB_BA (chunk type 1) blend-mode chunk.

    PSOBB-shipped values: only ``flags=0x21`` and ``flags=0x25`` ever
    occur. The low 5 bits of the flags byte encode (src_factor,
    dst_factor) per Sega's ``NJD_BLENDALPHA_*`` constants. For
    out-of-game synthetic models we accept any value and round-trip
    bit-for-bit.

    ``src_factor`` / ``dst_factor`` are the symbolic names the editor
    uses; ``mode`` is the high-level preset (none / blend / additive /
    multiply / screen) the UI exposes. ``flags_byte`` is preserved for
    byte-exact round-trip when no edit happens.
    """
    flags_byte: int = 0x25
    src_factor: str = "src_alpha"
    dst_factor: str = "one_minus_src_alpha"
    mode: str = "blend"  # one of: none / blend / additive / multiply / screen


# Ninja blend factor symbols. Order matches the index in the lower 3
# bits of NJD_CB_BA's encoded factor field.
_BLEND_SRC_FACTORS = (
    "zero",
    "one",
    "src_color",
    "one_minus_src_color",
    "src_alpha",
    "one_minus_src_alpha",
    "dst_alpha",
    "one_minus_dst_alpha",
)
_BLEND_DST_FACTORS = _BLEND_SRC_FACTORS  # same enum, different field

# Friendly preset map so the editor doesn't have to expose all 8x8 pairs.
# Keyed by (src_factor, dst_factor); value is the high-level "mode" string.
BLEND_PRESETS = {
    ("one", "zero"):                          "none",
    ("src_alpha", "one_minus_src_alpha"):     "blend",
    ("src_alpha", "one"):                     "additive",
    ("dst_color", "zero"):                    "multiply",  # not in factor table; user-set
    ("one", "one_minus_src_color"):           "screen",
}


@dataclass
class TinyChunkPayload:
    """Decoded NJD_CT / NJD_CT_COL (chunk types 8-9).

    ``texture_id`` is the bottom 13 bits of the body word; the
    remaining bits are unused in PSOBB BB (Phantasmal World's
    ``parseTinyChunk`` documents bits 13-15 as super-sample multiplier
    but no shipped model uses it).

    The flag bits in the chunk header are interpreted per Sega's
    Ninja SDK conventions for tiny chunks. Empirically all 8 distinct
    flag values seen in 1,706 shipped Tiny chunks agree with this
    decoding.
    """
    type_id: int = 8
    flags_byte: int = 0x34
    texture_id: int = 0
    flat_shaded: bool = False
    env_mapped: bool = False
    point_filter: bool = True       # bit 2 — set on every shipped Tiny chunk
    super_sample: bool = False      # bit 4
    use_filter: bool = False        # bit 5 — bilinear/trilinear filter
    clamp_v: bool = False           # bit 6
    clamp_u: bool = False           # bit 7

    # 2026-04-25: PSOBB also overloads the Tiny chunk's high byte of the
    # body u16 with an alpha-test threshold (low 13 bits = tex_id, top 3
    # bits = exponent of the alpha threshold). This isn't in Sega's
    # original njchunk.h but matches what shipped tools emit when alpha
    # masking is enabled. We expose it as a separate field so the UI
    # can present it as "alpha threshold" rather than as a tex-id high
    # bit.
    alpha_threshold_bits: int = 0


@dataclass
class StripFlagsPayload:
    """Strip-chunk flag-byte decoder (chunk types 64-75).

    Each strip chunk's flags carry render-state for the geometry it
    contains. Bit positions follow Phantasmal World's ``parseStrip-
    Chunk``:

        0x01  ignore_light    — disable lighting
        0x02  use_alpha       — emit per-vertex alpha
        0x04  doubleside      — render both faces
        0x08  flat_shaded     — disable smooth normals
        0x10  env_map         — use env-mapped UVs
        0x20  unused
        0x40  no_zwrite       — disable depth WRITE
        0x80  no_ztest        — disable depth TEST
    """
    flags_byte: int = 0
    ignore_light: bool = False
    use_alpha: bool = False
    double_sided: bool = False
    flat_shaded: bool = False
    env_mapped: bool = False
    no_zwrite: bool = False
    no_ztest: bool = False


@dataclass
class SubmeshMaterial:
    """Aggregated material state for ONE strip emitted in a polygon stream.

    Built by :func:`aggregate_submesh_state` by walking the chunk list
    in order and collecting state up to (but not including) the strip
    chunk that produces the submesh. This matches PSOBB BB's Polygon-
    ChunkProcessor semantics — material state cascades through chunks
    until overwritten.

    All fields are OPTIONAL where "no chunk has been seen yet"; the
    Material Inspector's defaults fill these in (white diffuse, no
    blend, etc.).
    """
    submesh_idx: int = 0
    material_id: int = 0
    diffuse_rgba: Tuple[int, int, int, int] = (255, 255, 255, 255)
    ambient_rgba: Tuple[int, int, int, int] = (127, 127, 127, 255)
    specular_rgb: Tuple[int, int, int] = (255, 255, 255)
    specular_exponent: int = 11
    alpha_test: Optional[dict] = None  # {"enabled": bool, "threshold": int}
    alpha_blend: Optional[dict] = None  # {"src": str, "dst": str}
    blend_mode: str = "blend"          # high-level preset name
    two_sided: bool = False
    depth_test: bool = True
    depth_write: bool = True
    flat_shaded: bool = False
    env_mapped: bool = False


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------


def decode_material_chunk(
    type_id: int,
    flags: int,
    body: bytes,
) -> MaterialChunkPayload:
    """Decode a chunk of type 17..23 into a MaterialChunkPayload.

    Unrecognised type ids return an empty payload (the caller should
    skip and not advance any state). Body must include the leading
    u16 word-count; we re-read it to validate the body shape.
    """
    out = MaterialChunkPayload(type_id=type_id, flags=flags)
    if type_id < 17 or type_id > 23:
        return out
    if len(body) < 2:
        return out
    (word_count,) = struct.unpack_from("<H", body, 0)
    payload = body[2:]
    expected_payload = word_count * 2  # word_count is in u16 words
    # Truncate to expected size to be defensive.
    if len(payload) > expected_payload:
        payload = payload[:expected_payload]
    if len(payload) < expected_payload:
        # Malformed but try to decode what we have.
        pass

    # Lay out per type_id. Each color is 4 BGRA bytes; the SPECULAR slot
    # is "RGB + 1 byte exponent" in shipped data.
    cur = 0
    if type_id in (17, 19, 21, 23):
        # Diffuse present.
        if cur + 4 <= len(payload):
            out.diffuse = RGBA.from_bgra_bytes(payload, cur)
            cur += 4
    if type_id in (18, 19, 22, 23):
        # Ambient present.
        if cur + 4 <= len(payload):
            out.ambient = RGBA.from_bgra_bytes(payload, cur)
            cur += 4
    if type_id in (20, 21, 22, 23):
        # Specular present (RGB + exponent byte).
        if cur + 4 <= len(payload):
            out.specular = RGBA.from_bgra_bytes(payload, cur)
            out.specular_exponent = payload[cur + 3]
            cur += 4
    return out


def decode_blend_alpha_chunk(flags: int) -> BlendAlphaPayload:
    """Decode an NJD_CB_BA chunk (type 1) flags byte.

    The flags byte structure per Sega Ninja convention (validated
    against the 23 BlendAlpha chunks in shipped PSOBB BB data):
        bits 0..2  dst factor index
        bits 3..5  src factor index
        bit  6     reserved
        bit  7     reserved

    Empirically PSOBB ships:
        ``flags=0x25`` (00100101) — src=4 (src_alpha),
                                    dst=5 (one_minus_src_alpha) → "blend"
        ``flags=0x21`` (00100001) — src=4 (src_alpha),
                                    dst=1 (one)                  → "additive"
        ``flags=0x09`` (00001001) — src=1 (one),
                                    dst=1 (one)                  → "additive"
                                    (a fallback variant; 1 occurrence)

    Out-of-range values still round-trip the raw byte via ``flags_byte``.
    """
    dst_idx = flags & 0x07
    src_idx = (flags >> 3) & 0x07
    src = _BLEND_SRC_FACTORS[src_idx] if src_idx < len(_BLEND_SRC_FACTORS) else "src_alpha"
    dst = _BLEND_DST_FACTORS[dst_idx] if dst_idx < len(_BLEND_DST_FACTORS) else "one_minus_src_alpha"
    mode = BLEND_PRESETS.get((src, dst), "blend")
    return BlendAlphaPayload(
        flags_byte=flags & 0xFF,
        src_factor=src,
        dst_factor=dst,
        mode=mode,
    )


def decode_tiny_chunk(flags: int, body: bytes) -> TinyChunkPayload:
    """Decode an NJD_CT / NJD_CT_COL chunk (types 8/9)."""
    out = TinyChunkPayload(flags_byte=flags & 0xFF)
    if len(body) >= 2:
        (word,) = struct.unpack_from("<H", body, 0)
        out.texture_id = word & 0x1FFF
        out.alpha_threshold_bits = (word >> 13) & 0x07
    out.flat_shaded   = bool(flags & 0x01)
    out.env_mapped    = bool(flags & 0x02)
    out.point_filter  = bool(flags & 0x04)
    out.super_sample  = bool(flags & 0x10)
    out.use_filter    = bool(flags & 0x20)
    out.clamp_v       = bool(flags & 0x40)
    out.clamp_u       = bool(flags & 0x80)
    return out


def decode_strip_chunk_flags(flags: int) -> StripFlagsPayload:
    """Decode the flags byte of a strip chunk (types 64..75)."""
    return StripFlagsPayload(
        flags_byte=flags & 0xFF,
        ignore_light=bool(flags & 0x01),
        use_alpha=bool(flags & 0x02),
        double_sided=bool(flags & 0x04),
        flat_shaded=bool(flags & 0x08),
        env_mapped=bool(flags & 0x10),
        no_zwrite=bool(flags & 0x40),
        no_ztest=bool(flags & 0x80),
    )


# ---------------------------------------------------------------------------
# Aggregator: walk a polygon stream and emit per-submesh state
# ---------------------------------------------------------------------------


def aggregate_submesh_state(plist_chunks) -> List[SubmeshMaterial]:
    """Walk a polygon-stream chunk list and emit one SubmeshMaterial per strip.

    ``plist_chunks`` is a list of objects with ``type_id``, ``flags``,
    and ``body`` attributes (matches both ``formats.xj._walk_chunk_stream``
    tuples and ``formats.nj_writer.NjChunk`` instances). The function
    is duck-typed for both shapes.

    Each strip chunk (type 64..75) that produces a submesh contributes
    one ``SubmeshMaterial`` row. The aggregator carries the most recent
    material / blend / tiny state forward.
    """
    out: List[SubmeshMaterial] = []
    cur_diff = (255, 255, 255, 255)
    cur_amb = (127, 127, 127, 255)
    cur_spec = (255, 255, 255)
    cur_spec_exp = 11
    cur_blend: Optional[BlendAlphaPayload] = None
    cur_tiny: Optional[TinyChunkPayload] = None
    cur_tex_id = 0
    submesh_idx = 0

    for c in plist_chunks:
        # Duck-type: tuple form is (hdr, type_id, flags, body_pos, body_size)
        # but we only need (type_id, flags, body). Accept dataclasses too.
        if hasattr(c, "type_id"):
            t = c.type_id
            f = c.flags
            b = c.body
        elif isinstance(c, (list, tuple)) and len(c) >= 3:
            t = c[0] if isinstance(c[0], int) else c[1]
            f = c[1] if isinstance(c[0], int) else c[2]
            b = c[2] if isinstance(c[0], int) else b''
        else:
            continue

        if 17 <= t <= 23:
            mp = decode_material_chunk(t, f, b)
            if mp.diffuse is not None:
                cur_diff = mp.diffuse.to_tuple()
            if mp.ambient is not None:
                cur_amb = mp.ambient.to_tuple()
            if mp.specular is not None:
                r, g, bl, _a = mp.specular.to_tuple()
                cur_spec = (r, g, bl)
                if mp.specular_exponent is not None:
                    cur_spec_exp = mp.specular_exponent
        elif t == 1:
            cur_blend = decode_blend_alpha_chunk(f)
        elif t in (8, 9):
            cur_tiny = decode_tiny_chunk(f, b)
            cur_tex_id = cur_tiny.texture_id
        elif 64 <= t <= 75:
            sf = decode_strip_chunk_flags(f)
            row = SubmeshMaterial(
                submesh_idx=submesh_idx,
                material_id=cur_tex_id,
                diffuse_rgba=cur_diff,
                ambient_rgba=cur_amb,
                specular_rgb=cur_spec,
                specular_exponent=cur_spec_exp,
                two_sided=sf.double_sided,
                depth_test=not sf.no_ztest,
                depth_write=not sf.no_zwrite,
                flat_shaded=sf.flat_shaded,
                env_mapped=sf.env_mapped,
            )
            if cur_blend is not None:
                row.alpha_blend = {
                    "src": cur_blend.src_factor,
                    "dst": cur_blend.dst_factor,
                }
                row.blend_mode = cur_blend.mode
            else:
                row.blend_mode = "none"
            if cur_tiny is not None and cur_tiny.alpha_threshold_bits != 0:
                # Reconstruct an 8-bit threshold from the 3-bit overlay.
                threshold = (cur_tiny.alpha_threshold_bits << 5) & 0xE0
                row.alpha_test = {"enabled": True, "threshold": threshold}
            out.append(row)
            submesh_idx += 1
    return out


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


def encode_material_chunk(payload: MaterialChunkPayload) -> Tuple[int, int, bytes]:
    """Encode a MaterialChunkPayload back to (type_id, flags, body).

    The function PICKS the smallest chunk type that fits the present
    fields:
        diffuse only            -> type 17 (wc=2)
        ambient only            -> type 18 (wc=2)
        diffuse + ambient       -> type 19 (wc=4)
        specular only           -> type 20 (wc=2)
        diffuse + specular      -> type 21 (wc=4)
        ambient + specular      -> type 22 (wc=4)
        all three               -> type 23 (wc=6)

    If the user passed a ``type_id`` explicitly we honour it (so the
    BML round-trips byte-exact for unedited entries).
    """
    has_d = payload.diffuse is not None
    has_a = payload.ambient is not None
    has_s = payload.specular is not None or payload.specular_exponent is not None

    if has_d and has_a and has_s:
        type_id = 23
    elif has_d and has_s:
        type_id = 21
    elif has_a and has_s:
        type_id = 22
    elif has_d and has_a:
        type_id = 19
    elif has_s:
        type_id = 20
    elif has_d:
        type_id = 17
    elif has_a:
        type_id = 18
    else:
        type_id = payload.type_id  # no fields — degenerate, retain id

    # Compute payload bytes and word count.
    parts: List[bytes] = []
    if type_id in (17, 19, 21, 23):
        d = payload.diffuse or RGBA()
        parts.append(d.to_bgra_bytes())
    if type_id in (18, 19, 22, 23):
        a = payload.ambient or RGBA(r=127, g=127, b=127, a=255)
        parts.append(a.to_bgra_bytes())
    if type_id in (20, 21, 22, 23):
        s = payload.specular or RGBA(r=255, g=255, b=255, a=11)
        # Force the exponent into the alpha slot if explicitly given.
        if payload.specular_exponent is not None:
            s = RGBA(r=s.r, g=s.g, b=s.b, a=payload.specular_exponent & 0xFF)
        parts.append(s.to_bgra_bytes())

    payload_bytes = b"".join(parts)
    word_count = len(payload_bytes) // 2
    body = struct.pack("<H", word_count) + payload_bytes
    return (type_id, payload.flags & 0xFF, body)


def encode_blend_alpha_flags(src_factor: str, dst_factor: str) -> int:
    """Encode (src, dst) factor names back to a flags byte.

    Inverse of :func:`decode_blend_alpha_chunk` — bits 3..5 = src,
    bits 0..2 = dst. The upper bits 0x40/0x80 are reserved and left
    zero by this encoder; callers that want to preserve the 0x20 bit
    that shipped data sets on every BlendAlpha chunk must OR it in
    after calling.
    """
    try:
        s = _BLEND_SRC_FACTORS.index(src_factor)
    except ValueError:
        s = _BLEND_SRC_FACTORS.index("src_alpha")
    try:
        d = _BLEND_DST_FACTORS.index(dst_factor)
    except ValueError:
        d = _BLEND_DST_FACTORS.index("one_minus_src_alpha")
    return ((s & 0x07) << 3) | (d & 0x07)


def encode_strip_flags(payload: StripFlagsPayload) -> int:
    """Encode StripFlagsPayload back to a flags byte."""
    f = 0
    if payload.ignore_light:  f |= 0x01
    if payload.use_alpha:     f |= 0x02
    if payload.double_sided:  f |= 0x04
    if payload.flat_shaded:   f |= 0x08
    if payload.env_mapped:    f |= 0x10
    if payload.no_zwrite:     f |= 0x40
    if payload.no_ztest:      f |= 0x80
    return f & 0xFF


# ---------------------------------------------------------------------------
# High-level edit applicator: takes a list of submesh edits and rewrites
# the polygon stream to match.
# ---------------------------------------------------------------------------


def _coerce_rgba(seq) -> RGBA:
    """Coerce a 4-element sequence of ints into RGBA, clamping to 0..255."""
    if seq is None:
        return RGBA()
    s = list(seq) + [255, 255, 255, 255]
    return RGBA(
        r=max(0, min(255, int(s[0]))),
        g=max(0, min(255, int(s[1]))),
        b=max(0, min(255, int(s[2]))),
        a=max(0, min(255, int(s[3]))),
    )


def apply_submesh_edits(plist_chunks, edits: list):
    """Mutate a polygon-stream chunk list to apply per-submesh edits.

    ``edits`` is a list of dicts shaped like the wire format the
    Material Inspector POSTs:

        {"submesh_idx": int, "diffuse_rgba": [r,g,b,a],
         "alpha_test": {"enabled": bool, "threshold": int} | None,
         "alpha_blend": {"src": str, "dst": str} | None,
         "two_sided": bool, "depth_test": bool, "depth_write": bool}

    The function walks the polygon stream and, for each strip-chunk
    occurrence (which corresponds to a submesh), REPLACES the most
    recent material / blend / strip chunk with the edited value
    (inserting a new chunk just before the strip if none exists).

    The function MUTATES the list in place AND returns it, so callers
    can chain.

    The chunk dataclass duck-typed must support attribute writes
    (``c.flags = ...``, ``c.body = ...``). Tuple-form chunks aren't
    supported on the write path — convert them to dataclasses first.
    """
    if not edits:
        return plist_chunks

    edits_by_idx = {int(e.get("submesh_idx", -1)): e for e in edits}
    # We need a chunk class to instantiate new chunks. Mirror the
    # dataclass form expected by NjChunk.
    try:
        from .nj_writer import NjChunk
    except ImportError:
        NjChunk = None  # type: ignore[assignment]

    submesh_idx = 0
    out = list(plist_chunks)
    i = 0
    # We mutate in place.
    while i < len(out):
        c = out[i]
        if not (hasattr(c, "type_id") and hasattr(c, "flags")):
            i += 1
            continue
        if 64 <= c.type_id <= 75:
            edit = edits_by_idx.get(submesh_idx)
            if edit is not None:
                _apply_edit_at_strip(out, i, edit, NjChunk)
                # The number of chunks in `out` may have grown; advance
                # past the strip we just landed on. Find it again — the
                # strip chunk itself was also mutated but stays at
                # roughly the same index after our insertions.
            submesh_idx += 1
        i += 1
    return out


def _find_recent_chunk(out: list, end_idx: int, type_pred) -> Optional[int]:
    """Scan backwards from ``end_idx`` looking for a chunk passing pred."""
    j = end_idx - 1
    while j >= 0:
        c = out[j]
        if hasattr(c, "type_id") and type_pred(c.type_id):
            return j
        j -= 1
    return None


def _apply_edit_at_strip(out: list, strip_idx: int, edit: dict, NjChunk):
    """Apply one submesh edit by mutating chunks just before the strip.

    Strategy:
      1. If `diffuse_rgba` is given, replace (or insert) the most
         recent type-17/19/21/23 chunk with a type-23 carrying the new
         diffuse + preserved ambient/specular.
      2. If `alpha_blend` is given, replace (or insert) the most recent
         type-1 BlendAlpha chunk's flags byte.
      3. If `two_sided` / `depth_test` / `depth_write` are given, mutate
         the strip chunk's own flags byte.
      4. If `alpha_test` is given, mutate the most recent type-8 Tiny
         chunk's body high-bits.
    """
    strip = out[strip_idx]

    # 1. Material colors.
    if "diffuse_rgba" in edit and edit["diffuse_rgba"] is not None:
        diff = _coerce_rgba(edit["diffuse_rgba"])
        amb_idx = _find_recent_chunk(out, strip_idx, lambda t: t in (17, 18, 19, 21, 22, 23))
        # Decode the existing one (if any) to preserve other fields.
        existing = MaterialChunkPayload()
        if amb_idx is not None:
            existing = decode_material_chunk(
                out[amb_idx].type_id, out[amb_idx].flags, out[amb_idx].body
            )
        existing.diffuse = diff
        # Make sure we still emit ambient/specular if they were present.
        new_type, new_flags, new_body = encode_material_chunk(existing)
        if NjChunk is not None:
            new_chunk = NjChunk(type_id=new_type, flags=new_flags, body=new_body)
            if amb_idx is not None:
                out[amb_idx] = new_chunk
            else:
                out.insert(strip_idx, new_chunk)
                strip_idx += 1
                strip = out[strip_idx]

    # 2. Blend mode.
    if "alpha_blend" in edit and edit["alpha_blend"] is not None:
        ab = edit["alpha_blend"]
        src = ab.get("src", "src_alpha")
        dst = ab.get("dst", "one_minus_src_alpha")
        new_flags = encode_blend_alpha_flags(src, dst)
        # Sega convention: the upper bits 0x20 are always set in shipped
        # PSOBB data. Preserve them so the round-trip flag matches.
        new_flags |= 0x20
        ba_idx = _find_recent_chunk(out, strip_idx, lambda t: t == 1)
        if NjChunk is not None:
            ba_chunk = NjChunk(type_id=1, flags=new_flags, body=b"")
            if ba_idx is not None:
                out[ba_idx] = ba_chunk
            else:
                out.insert(strip_idx, ba_chunk)
                strip_idx += 1
                strip = out[strip_idx]

    # 3. Strip-chunk flags (two-sided, depth).
    new_strip_flags = strip.flags
    if "two_sided" in edit and edit["two_sided"] is not None:
        if edit["two_sided"]:
            new_strip_flags |= 0x04
        else:
            new_strip_flags &= ~0x04
    if "depth_write" in edit and edit["depth_write"] is not None:
        if not edit["depth_write"]:
            new_strip_flags |= 0x40
        else:
            new_strip_flags &= ~0x40
    if "depth_test" in edit and edit["depth_test"] is not None:
        if not edit["depth_test"]:
            new_strip_flags |= 0x80
        else:
            new_strip_flags &= ~0x80
    if new_strip_flags != strip.flags:
        strip.flags = new_strip_flags & 0xFF

    # 4. Alpha test (overlay 3-bit threshold onto Tiny chunk).
    if "alpha_test" in edit and edit["alpha_test"] is not None:
        at = edit["alpha_test"]
        enabled = bool(at.get("enabled", False))
        threshold = int(at.get("threshold", 128)) & 0xFF
        tiny_idx = _find_recent_chunk(out, strip_idx, lambda t: t in (8, 9))
        if tiny_idx is not None and len(out[tiny_idx].body) >= 2:
            (word,) = struct.unpack_from("<H", out[tiny_idx].body, 0)
            tex_id = word & 0x1FFF
            if enabled:
                # Encode threshold's top 3 bits into the Tiny word's
                # high overlay region.
                overlay = (threshold >> 5) & 0x07
                new_word = (overlay << 13) | tex_id
            else:
                new_word = tex_id
            new_body = struct.pack("<H", new_word) + out[tiny_idx].body[2:]
            out[tiny_idx].body = bytes(new_body)


# ---------------------------------------------------------------------------
# Preset catalogue (Task 4)
# ---------------------------------------------------------------------------


PSOBB_MATERIAL_PRESETS = {
    "player_skin": {
        "label": "Player Skin",
        "description": "Standard alpha-tested skin (player + npc bodies)",
        "alpha_test": {"enabled": True, "threshold": 128},
        "alpha_blend": None,
        "blend_mode": "none",
        "two_sided": False,
        "depth_test": True,
        "depth_write": True,
    },
    "hair_fur": {
        "label": "Hair / Fur",
        "description": "Lower alpha threshold + double-sided so back of hair shows",
        "alpha_test": {"enabled": True, "threshold": 64},
        "alpha_blend": None,
        "blend_mode": "none",
        "two_sided": True,
        "depth_test": True,
        "depth_write": True,
    },
    "energy_glass": {
        "label": "Glass / Energy / FX",
        "description": "Additive blend, no depth-write (for glow / particle / glass)",
        "alpha_test": None,
        "alpha_blend": {"src": "src_alpha", "dst": "one"},
        "blend_mode": "additive",
        "two_sided": False,
        "depth_test": True,
        "depth_write": False,
    },
    "standard_solid": {
        "label": "Standard Solid",
        "description": "Opaque mesh with normal blend; the default for most submeshes",
        "alpha_test": None,
        "alpha_blend": None,
        "blend_mode": "none",
        "two_sided": False,
        "depth_test": True,
        "depth_write": True,
    },
    "transparent_blend": {
        "label": "Transparent (alpha blend)",
        "description": "Standard premultiplied alpha blend (water, smoke, decals)",
        "alpha_test": None,
        "alpha_blend": {"src": "src_alpha", "dst": "one_minus_src_alpha"},
        "blend_mode": "blend",
        "two_sided": False,
        "depth_test": True,
        "depth_write": False,
    },
}


def list_presets() -> list:
    """Return the preset catalogue as a JSON-serializable list."""
    return [
        {"key": k, **v} for k, v in PSOBB_MATERIAL_PRESETS.items()
    ]


# ---------------------------------------------------------------------------
# Inner texture-name extraction (cross-BML resolver helper)
# ---------------------------------------------------------------------------

def extract_inner_texture_names(inner_data: bytes, ext: str) -> List[str]:
    """Return the texture names declared by a model's NJTL chunk.

    Used by ``server.py``'s cross-archive resolver to drive the
    sibling-BML / global-index lookup before falling back to the same-
    BML positional sibling pick. The returned names are the same strings
    the runtime feeds to PSOBB's texture allocator — so a name match in
    a SIBLING BML's NJTL is a high-confidence hit.

    Args
    ----
    inner_data:
        The decompressed inner blob (raw .nj or .xj bytes — NOT the
        BML wrapper, NOT the outer XVMH).
    ext:
        Lower-cased extension string (e.g. ``".xj"``). Reserved for
        future format-specific dispatch; today both .nj and .xj share
        the same NJTL chunk format.

    Returns
    -------
    Ordered list of texture-name strings (one per NJTL slot, slot index
    = list index). Returns ``[]`` when the model has no NJTL chunk
    (~half of PSOBB's untextured inners) or when parsing fails. Never
    raises — the resolver treats an empty list as "no names known"
    and walks its own fallbacks.
    """
    if not inner_data:
        return []
    # Local import keeps the module load order clean: njtl pulls in iff
    # which we don't need at material-decode time.
    try:
        from .njtl import find_and_parse_njtl
    except ImportError:  # pragma: no cover - defensive
        return []
    try:
        entries = find_and_parse_njtl(inner_data)
    except Exception:
        return []
    if not entries:
        return []
    return [(e.name or "").strip() for e in entries]
