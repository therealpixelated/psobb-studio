'''
Pure-Python PVR (Sega Dreamcast PowerVR) texture ENCODER.

This is the byte-level inverse of :mod:`formats.pvr_decode`. It produces the
same ``GBIX`` (optional) + ``PVRT`` IFF container that ``decode_pvr`` reads, so
that ``decode_pvr(encode_pvr(img, ...))`` reproduces the input pixels within the
target format's quantisation tolerance (exact for the lossless paths).

Design principle — invert OUR decoder, not a textbook formula
-------------------------------------------------------------
The decoder (ported from VincentNL/pvr2image) de-twiddles by building a flat
permutation ``arr`` via :func:`pvr_decode._detwiddle` and reading
``ordered = [pixels[i] for i in arr]``. The encoder therefore re-twiddles by
inverting that exact permutation: it scatters the linear source pixels into the
twiddled layout with ``twiddled[arr[dest]] = linear[dest]``. This guarantees a
byte-exact round-trip for EVERY size the decoder accepts — including the
non-square (wide / tall) cases where pvr2image's twiddle order intentionally
differs from VrSharp's per-axis ``(tm[x]<<1)|tm[y]`` block walk. Re-deriving the
formula independently would risk disagreeing with the decoder on the wide case
(the classic PVR-encoder bug); inverting the decoder's own table cannot.

Supported (pixel_format, data_format) combinations
--------------------------------------------------
Direct-colour, NON-twiddled (lossless for ARGB8888, quantised otherwise):
    px 7 (ARGB8888) + tex 14 (BMP)                — 32bpp raw, lossless
    px 0/1/2/5 + tex 9 (Rectangle)                — 16bpp linear
    px 7 + tex 9 (Rectangle, ARGB8888 16-bit?)    — see note below

Direct-colour, TWIDDLED (re-twiddled via the inverse permutation):
    px 0/1/2/5 + tex 1  (SquareTwiddled)          — 16bpp, square only
    px 0/1/2/5 + tex 13 (RectangleTwiddled)       — 16bpp, any pow2 dims

NotImplementedError (genuinely unsupported — never emits wrong bytes):
    VQ / SmallVQ (tex 3/4/16/17)   — requires codebook generation (lossy VQ
                                     training); VrSharp's encoder also refuses
                                     these (CanEncode == false).
    Palettized 4/8-bit (tex 5/6/7/8) — needs colour-quantisation + an external
                                     PVPL palette file; out of scope here.
    YUV422 / YUV420 / BUMP (px 3/6/4) — lossy chroma / cartesian-bump synthesis
                                     with no faithful inverse.
    Mipmapped variants (tex 2/6/8/10/12/15/17/18) — the encoder emits a single
                                     (largest) level; mip-pyramid generation is
                                     a separate concern.

Public API
----------
    encode_pvr(img, pixel_format, data_format, *, global_index=None) -> bytes
    build_pvm(records, *, global_indices=None) -> bytes
'''

from __future__ import annotations

import struct
from typing import List, Optional, Sequence, Tuple

from PIL import Image

from . import pvr_decode

# Re-export the format dictionaries so callers can introspect names.
PX_MODES = pvr_decode.PX_MODES
TEX_MODES = pvr_decode.TEX_MODES


# Pixel formats we can pack to a 16-bit word (matches _read_col's inverse).
_SIXTEEN_BIT_PX = (0, 1, 2, 5)

# Data formats this encoder can produce. Everything else -> NotImplementedError.
_TWIDDLED_TEX = (1, 13)          # SquareTwiddled, RectangleTwiddled
_LINEAR_16_TEX = (9,)            # Rectangle (non-twiddled, 16bpp)
_BMP_TEX = (14,)                 # BMP / ARGB8888 raw

# Mipmapped data formats (we encode a single level only -> reject).
_MIP_TEX = (2, 4, 6, 8, 10, 12, 15, 17, 18)
# VQ / SmallVQ — codebook generation is lossy training, refused (as VrSharp).
_VQ_TEX = (3, 4, 16, 17)
# Palettized — needs colour quantisation + external PVPL palette.
_PAL_TEX = (5, 6, 7, 8)


# ---------------------------------------------------------------------------
# Per-pixel packers — exact inverses of pvr_decode._read_col.
# ---------------------------------------------------------------------------


def _q(value: int, bits: int) -> int:
    """Nearest-value quantise an 8-bit channel to ``bits`` bits.

    The decoder expands an N-bit channel back to 8-bit by ``int(n*0xff/max)``
    (5/6-bit) or ``n*0x11`` (4-bit). The mathematically faithful inverse is the
    nearest grid point ``round(value*max/255)`` — NOT truncation ``value>>shift``
    — which both minimises round-trip error (e.g. ARGB4444 31.9->34.4 dB,
    ARGB1555 38.2->40.4 dB) and still reproduces on-grid values (0, 255, ...)
    exactly. ``maxv = (1<<bits)-1``.
    """
    maxv = (1 << bits) - 1
    return min(maxv, (value * maxv + 127) // 255)


def _pack_pixel_16(px_format: int, r: int, g: int, b: int, a: int) -> int:
    """Pack one RGBA8 pixel into the 16-bit word the decoder reads back.

    These bit layouts are the exact inverse of ``pvr_decode._read_col`` for
    each 16-bit pixel format, using nearest-value quantisation (:func:`_q`) so
    re-decoding the encoded word yields the channel closest to the source.
    """
    if px_format == 0:  # ARGB1555 : A=15, R=14..10, G=9..5, B=4..0
        return (((1 if a >= 0x80 else 0) << 15)
                | (_q(r, 5) << 10)
                | (_q(g, 5) << 5)
                | _q(b, 5))
    if px_format == 1:  # RGB565 : R=15..11, G=10..5, B=4..0 (alpha dropped)
        return ((_q(r, 5) << 11)
                | (_q(g, 6) << 5)
                | _q(b, 5))
    if px_format == 2:  # ARGB4444 : A=15..12, R=11..8, G=7..4, B=3..0
        return ((_q(a, 4) << 12)
                | (_q(r, 4) << 8)
                | (_q(g, 4) << 4)
                | _q(b, 4))
    if px_format == 5:  # RGB555 : A bit ignored on decode, R=14..10,G,B
        return ((_q(r, 5) << 10)
                | (_q(g, 5) << 5)
                | _q(b, 5))
    raise NotImplementedError(
        f"encode_pvr: 16-bit packing not implemented for px_format {px_format}"
    )


# ---------------------------------------------------------------------------
# Body encoders.
# ---------------------------------------------------------------------------


def _img_to_rgba_rows(img: Image.Image) -> Tuple[int, int, List[Tuple[int, int, int, int]]]:
    """Return (w, h, [ (r,g,b,a), ... ]) row-major from a PIL image."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    w, h = img.size
    # Flatten to (r,g,b,a) tuples. ``tobytes`` is the stable, non-deprecated
    # path (``getdata`` is deprecated in Pillow 11+).
    raw = img.tobytes()  # RGBA8, row-major
    px = [tuple(raw[i:i + 4]) for i in range(0, len(raw), 4)]
    return w, h, px


def _encode_bmp(pixels: Sequence[Tuple[int, int, int, int]]) -> bytes:
    """ARGB8888 BMP body — disk order [A, B, G, R] per pixel (inverse of
    ``_read_col(14, ...)`` reading a little-endian u32 as r=hi..a=lo)."""
    out = bytearray(len(pixels) * 4)
    for i, (r, g, b, a) in enumerate(pixels):
        out[i * 4 + 0] = a & 0xFF
        out[i * 4 + 1] = b & 0xFF
        out[i * 4 + 2] = g & 0xFF
        out[i * 4 + 3] = r & 0xFF
    return bytes(out)


def _encode_linear16(px_format: int,
                     pixels: Sequence[Tuple[int, int, int, int]]) -> bytes:
    """Non-twiddled 16-bit body, row-major (Rectangle / tex 9)."""
    out = bytearray(len(pixels) * 2)
    for i, (r, g, b, a) in enumerate(pixels):
        word = _pack_pixel_16(px_format, r, g, b, a)
        struct.pack_into("<H", out, i * 2, word)
    return bytes(out)


def _encode_twiddled16(px_format: int, w: int, h: int,
                       pixels: Sequence[Tuple[int, int, int, int]]) -> bytes:
    """Twiddled 16-bit body — the inverse of the decoder's de-twiddle.

    The decoder reads ``ordered = [src[i] for i in arr]`` where ``ordered`` is
    row-major destination and ``arr`` is :func:`pvr_decode._detwiddle`. So the
    on-disk twiddled word at position ``arr[dest]`` must be the pixel that the
    decoder will place at row-major ``dest``: ``twiddled[arr[dest]] = px[dest]``.
    """
    arr = pvr_decode._detwiddle(w, h)
    if len(arr) != w * h:
        raise ValueError(
            f"encode_pvr: detwiddle map size {len(arr)} != {w*h} for {w}x{h}"
        )
    out = bytearray(len(pixels) * 2)
    for dest, (r, g, b, a) in enumerate(pixels):
        word = _pack_pixel_16(px_format, r, g, b, a)
        struct.pack_into("<H", out, arr[dest] * 2, word)
    return bytes(out)


# ---------------------------------------------------------------------------
# Validation helpers.
# ---------------------------------------------------------------------------


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _reject_unsupported(px_format: int, data_format: int, w: int, h: int) -> None:
    """Raise NotImplementedError with a precise reason for every combo we
    cannot faithfully encode. Never silently produces wrong bytes."""
    if data_format in _MIP_TEX and data_format not in (4, 17):
        # (4/17 also caught by _VQ_TEX below with a VQ-specific message.)
        raise NotImplementedError(
            f"encode_pvr: mipmapped data_format {data_format} "
            f"({TEX_MODES.get(data_format)}) not supported — only single-level "
            "encoding is implemented."
        )
    if data_format in _VQ_TEX:
        raise NotImplementedError(
            f"encode_pvr: VQ/SmallVQ data_format {data_format} "
            f"({TEX_MODES.get(data_format)}) not supported — requires VQ "
            "codebook training (lossy); the reference VrSharp encoder also "
            "refuses these (CanEncode == false)."
        )
    if data_format in _PAL_TEX:
        raise NotImplementedError(
            f"encode_pvr: palettized data_format {data_format} "
            f"({TEX_MODES.get(data_format)}) not supported — requires colour "
            "quantisation + an external PVPL palette file."
        )
    if px_format in (3, 4, 6):  # YUV422 / BUMP / YUV420
        raise NotImplementedError(
            f"encode_pvr: pixel_format {px_format} "
            f"({PX_MODES.get(px_format)}) not supported — lossy chroma / "
            "bump synthesis has no faithful inverse."
        )


# ---------------------------------------------------------------------------
# Public API — encode_pvr.
# ---------------------------------------------------------------------------


def encode_pvr(
    img: Image.Image,
    pixel_format: int,
    data_format: int,
    *,
    global_index: Optional[int] = None,
) -> bytes:
    """Encode a PIL image to a complete PVR file (optional GBIX + PVRT).

    Parameters
    ----------
    img
        Source image. Converted to RGBA if needed. Width/height must be
        powers of two for the twiddled data formats (PVR hardware
        requirement); the Rectangle / BMP formats accept any dimensions.
    pixel_format
        PVR pixel-format byte (see :data:`PX_MODES`). Supported:
        0 ARGB1555, 1 RGB565, 2 ARGB4444, 5 RGB555 (16-bit) and
        7 ARGB8888 (with tex 14 BMP).
    data_format
        PVR data-format / texture-format byte (see :data:`TEX_MODES`).
        Supported: 1 SquareTwiddled, 9 Rectangle, 13 RectangleTwiddled,
        14 BMP.
    global_index
        If not None, an unsigned 32-bit value written into a leading GBIX
        chunk (``decode_pvr`` and ``_parse_pvr_header`` both skip it). If
        None, no GBIX chunk is written.

    Returns
    -------
    bytes
        The full PVR file. ``decode_pvr`` of the result reproduces the
        input pixels within the format's quantisation tolerance.

    Raises
    ------
    NotImplementedError
        For genuinely-unsupported (px, tex) combinations (VQ, palettized,
        YUV, BUMP, mipmapped) — never emits wrong bytes for these.
    ValueError
        For malformed inputs (bad dimensions for a twiddled format, etc).
    """
    w, h, pixels = _img_to_rgba_rows(img)
    if w == 0 or h == 0:
        raise ValueError("encode_pvr: image has a zero dimension")
    if w > 0xFFFF or h > 0xFFFF:
        raise ValueError(f"encode_pvr: dimension too large ({w}x{h}); u16 max")

    _reject_unsupported(pixel_format, data_format, w, h)

    # ----- choose body encoder -----
    if data_format in _BMP_TEX:
        if pixel_format != 7:
            raise NotImplementedError(
                f"encode_pvr: BMP (tex 14) only supports px_format 7 "
                f"(ARGB8888); got {pixel_format}."
            )
        body = _encode_bmp(pixels)

    elif data_format in _LINEAR_16_TEX:
        if pixel_format not in _SIXTEEN_BIT_PX:
            raise NotImplementedError(
                f"encode_pvr: Rectangle (tex 9) supports 16-bit pixel formats "
                f"{_SIXTEEN_BIT_PX}; got {pixel_format} "
                f"({PX_MODES.get(pixel_format)})."
            )
        body = _encode_linear16(pixel_format, pixels)

    elif data_format in _TWIDDLED_TEX:
        if pixel_format not in _SIXTEEN_BIT_PX:
            raise NotImplementedError(
                f"encode_pvr: twiddled data_format {data_format} "
                f"({TEX_MODES.get(data_format)}) supports 16-bit pixel formats "
                f"{_SIXTEEN_BIT_PX}; got {pixel_format} "
                f"({PX_MODES.get(pixel_format)})."
            )
        if data_format == 1 and w != h:
            raise ValueError(
                f"encode_pvr: SquareTwiddled (tex 1) requires a square image; "
                f"got {w}x{h}. Use RectangleTwiddled (tex 13) for non-square."
            )
        if not (_is_pow2(w) and _is_pow2(h)):
            raise ValueError(
                f"encode_pvr: twiddled formats require power-of-two dimensions; "
                f"got {w}x{h}."
            )
        body = _encode_twiddled16(pixel_format, w, h, pixels)

    else:
        raise NotImplementedError(
            f"encode_pvr: data_format {data_format} "
            f"({TEX_MODES.get(data_format)}) not supported."
        )

    return _wrap_pvrt(body, pixel_format, data_format, w, h, global_index)


def _wrap_pvrt(
    body: bytes,
    pixel_format: int,
    data_format: int,
    w: int,
    h: int,
    global_index: Optional[int],
) -> bytes:
    """Assemble the optional GBIX chunk + PVRT chunk + body.

    Header layout mirrors ``pvr_decode._parse_pvr_header`` and VrSharp's
    ``PvrTextureEncoder.EncodeTexture``:

        GBIX (optional):  'GBIX' | u32 len=8 | u32 global_index | u32 0
        PVRT:             'PVRT' | u32 content_size | px,tex,u16 0 | u16 w | u16 h | body

    ``content_size`` is ``8 + len(body)`` — the number of bytes after the
    PVRT chunk-size field (px+tex+pad+w+h+body), matching what VrSharp writes
    (``textureLength - 8`` without GBIX / ``- 24`` with GBIX both equal this).
    """
    out = bytearray()
    if global_index is not None:
        gi = int(global_index) & 0xFFFFFFFF
        out += b"GBIX"
        out += struct.pack("<I", 8)          # GBIX payload length (always 8)
        out += struct.pack("<I", gi)         # the global index
        out += struct.pack("<I", 0)          # reserved / padding

    out += b"PVRT"
    out += struct.pack("<I", 8 + len(body))  # content size after this field
    out += bytes([pixel_format & 0xFF, data_format & 0xFF, 0, 0])
    out += struct.pack("<HH", w, h)
    out += body
    return bytes(out)


# ---------------------------------------------------------------------------
# build_pvm — invert the PVMH archive reader (sibling_archives._parse_pvm_records)
# ---------------------------------------------------------------------------


def build_pvm(
    records: Sequence[bytes],
    *,
    global_indices: Optional[Sequence[int]] = None,
) -> bytes:
    """Pack a sequence of complete PVR records into a PVMH archive.

    Mirrors the shape of :func:`xvr_decode.build_xvm` and inverts
    ``sibling_archives._parse_pvm_records``: a ``PVMH`` chunk header followed by
    the concatenated records. Each record is a full ``[optional GBIX] + PVRT +
    body`` blob — exactly what :func:`encode_pvr` returns — so the reader's
    GBIX/PVRT walk (which skips any leading GBIX before each PVRT) re-discovers
    them.

    A minimal header is emitted: ``'PVMH' | u32 header_size | u16 flags | u16
    count`` with ``flags = 0`` (no optional per-record tables — the reader does
    not require them) and ``header_size = 4`` (covers just flags+count). The
    first record then begins at offset ``0x08 + 4 == 0x0C``.

    Parameters
    ----------
    records
        Complete PVR record blobs (e.g. from :func:`encode_pvr`).
    global_indices
        Optional — currently advisory only. The minimal header carries no
        global-index table; per-record GBIX chunks (set via
        ``encode_pvr(global_index=...)``) are the canonical place for these.
        Provided for signature symmetry with archive builders that emit the
        table; a non-None value that disagrees with the record count raises.

    Returns
    -------
    bytes
        A ``PVMH`` archive that ``sibling_archives._parse_pvm_records`` walks
        back into the same records.
    """
    recs = [bytes(r) for r in records]
    if global_indices is not None and len(global_indices) != len(recs):
        raise ValueError(
            f"build_pvm: global_indices length {len(global_indices)} "
            f"!= record count {len(recs)}"
        )
    if len(recs) > 0xFFFF:
        raise ValueError(f"build_pvm: too many records ({len(recs)}); u16 max")

    flags = 0           # no optional per-record tables
    count = len(recs)
    header_size = 0x04  # flags(2) + count(2) — the bytes after offset 0x08

    out = bytearray()
    out += b"PVMH"
    out += struct.pack("<I", header_size)
    out += struct.pack("<H", flags)
    out += struct.pack("<H", count)
    for r in recs:
        out += r
    return bytes(out)


__all__ = [
    "PX_MODES",
    "TEX_MODES",
    "encode_pvr",
    "build_pvm",
]
