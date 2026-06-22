"""GameCube GVR (GVRT inside GVMH) texture decoder.

Counterpart to ``formats/pvr_decode.py`` (Dreamcast PVR) and
``formats/xvr_decode.py`` (Xbox XVR) for the GameCube ``GVRT`` container.
Added 2026-06-21 as part of the psov2 multivariant texture pipeline so
the BML/model binding + standalone-extract routes can decode a GameCube
inline/sibling archive instead of emitting a magenta placeholder.

This is an INDEPENDENT port of the public GameCube texture layout (the
big-endian GVRT header + the GameCube tile/block scan order). It is
structurally faithful to VrSharp's ``GvrDataCodec`` / ``GvrPixelCodec``
(SA-Tools, MIT) and Sega's GVR format, both of which document the layout
publicly. We do NOT copy code verbatim — the block loops are
re-expressed in Python.

GVRT header (big-endian), located by the ``GVRT`` magic:
    +0x00  'GVRT'
    +0x04  u32  chunk size (rest of chunk after this field)
    +0x08  u16  (often 0)            — unused here
    +0x0A  u8   (pixelFormat<<4) | dataFlags
    +0x0B  u8   dataFormat           — selects the codec below
    +0x0C  u16  width  (BE)
    +0x0E  u16  height (BE)
    +0x10  pixel data

Data formats (``dataFormat`` byte):
    0x00 Intensity4   (I4,  8x8 block, 4bpp)
    0x01 Intensity8   (I8,  8x4 block, 8bpp)
    0x02 IntensityA4  (IA4, 8x4 block, 8bpp)
    0x03 IntensityA8  (IA8, 4x4 block, 16bpp)
    0x04 Rgb565       (4x4 block, 16bpp)
    0x05 Rgb5a3       (4x4 block, 16bpp)
    0x06 Argb8888     (4x4 block, 32bpp, split AR/GB sub-blocks)
    0x0E Dxt1 / CMP   (GameCube S3TC, 8x8 macro-block of four 4x4 DXT1)

Index4 (0x08) / Index8 (0x09) need an external/internal palette which
PSO's GVMH inners do not ship inline in the cases we handle; those raise
NotImplementedError so the caller falls back to the placeholder (no
regression — the prior behaviour for GVMH was the placeholder anyway).

Returns ``(width, height, rgba_bytes)`` — row-major, top-to-bottom,
RGBA8 — matching ``pvr_decode.decode_pvr``'s contract so the server's
tile-PNG path is identical for PVR and GVR.
"""
from __future__ import annotations

import struct
from typing import List, Tuple

__all__ = ["decode_gvr", "GVR_DATA_FORMATS"]

GVRT_MAGIC = b"GVRT"

GVR_DATA_FORMATS = {
    0x00: "Intensity4",
    0x01: "Intensity8",
    0x02: "IntensityA4",
    0x03: "IntensityA8",
    0x04: "Rgb565",
    0x05: "Rgb5a3",
    0x06: "Argb8888",
    0x08: "Index4",
    0x09: "Index8",
    0x0E: "Dxt1",
}


def _u16be(buf: bytes, off: int) -> int:
    return struct.unpack_from(">H", buf, off)[0]


def _parse_gvrt_header(data: bytes) -> Tuple[int, int, int, int, int]:
    """Locate GVRT and read (data_offset, width, height, data_format,
    pixel_format). Raises ValueError on missing magic / truncation."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError("decode_gvr: input must be bytes-like")
    data = bytes(data)
    gv = data.find(GVRT_MAGIC)
    if gv < 0:
        raise ValueError("decode_gvr: no GVRT magic")
    if gv + 0x10 > len(data):
        raise ValueError("decode_gvr: truncated GVRT header")
    flags_byte = data[gv + 0x0A]
    pixel_format = (flags_byte >> 4) & 0x0F
    data_format = data[gv + 0x0B]
    width = _u16be(data, gv + 0x0C)
    height = _u16be(data, gv + 0x0E)
    if width == 0 or height == 0 or width > 4096 or height > 4096:
        raise ValueError(f"decode_gvr: implausible dims {width}x{height}")
    return (gv + 0x10, width, height, data_format, pixel_format)


def _exp(v: int, bits: int) -> int:
    """Expand a ``bits``-wide channel value to 8-bit (×255/max)."""
    maxv = (1 << bits) - 1
    return (v * 0xFF) // maxv


def _new_rgba(w: int, h: int) -> bytearray:
    return bytearray(w * h * 4)


def _put(out: bytearray, w: int, x: int, y: int, r: int, g: int, b: int, a: int):
    o = ((y * w) + x) * 4
    out[o + 0] = r & 0xFF
    out[o + 1] = g & 0xFF
    out[o + 2] = b & 0xFF
    out[o + 3] = a & 0xFF


def _decode_intensity4(data: bytes, off: int, w: int, h: int) -> bytearray:
    out = _new_rgba(w, h)
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            for y2 in range(8):
                for x2 in range(8):
                    entry = (data[off] >> ((~x2 & 0x01) * 4)) & 0x0F
                    v = _exp(entry, 4)
                    if y + y2 < h and x + x2 < w:
                        _put(out, w, x + x2, y + y2, v, v, v, 0xFF)
                    if x2 & 0x01:
                        off += 1
    return out


def _decode_intensity8(data: bytes, off: int, w: int, h: int) -> bytearray:
    out = _new_rgba(w, h)
    for y in range(0, h, 4):
        for x in range(0, w, 8):
            for y2 in range(4):
                for x2 in range(8):
                    v = data[off]
                    if y + y2 < h and x + x2 < w:
                        _put(out, w, x + x2, y + y2, v, v, v, 0xFF)
                    off += 1
    return out


def _decode_intensitya4(data: bytes, off: int, w: int, h: int) -> bytearray:
    out = _new_rgba(w, h)
    for y in range(0, h, 4):
        for x in range(0, w, 8):
            for y2 in range(4):
                for x2 in range(8):
                    px = data[off]
                    a = _exp((px >> 4) & 0x0F, 4)
                    i = _exp(px & 0x0F, 4)
                    if y + y2 < h and x + x2 < w:
                        _put(out, w, x + x2, y + y2, i, i, i, a)
                    off += 1
    return out


def _decode_intensitya8(data: bytes, off: int, w: int, h: int) -> bytearray:
    out = _new_rgba(w, h)
    for y in range(0, h, 4):
        for x in range(0, w, 4):
            for y2 in range(4):
                for x2 in range(4):
                    a = data[off]
                    i = data[off + 1]
                    if y + y2 < h and x + x2 < w:
                        _put(out, w, x + x2, y + y2, i, i, i, a)
                    off += 2
    return out


def _decode_rgb565(data: bytes, off: int, w: int, h: int) -> bytearray:
    out = _new_rgba(w, h)
    for y in range(0, h, 4):
        for x in range(0, w, 4):
            for y2 in range(4):
                for x2 in range(4):
                    px = _u16be(data, off)
                    r = _exp((px >> 11) & 0x1F, 5)
                    g = _exp((px >> 5) & 0x3F, 6)
                    b = _exp(px & 0x1F, 5)
                    if y + y2 < h and x + x2 < w:
                        _put(out, w, x + x2, y + y2, r, g, b, 0xFF)
                    off += 2
    return out


def _decode_rgb5a3(data: bytes, off: int, w: int, h: int) -> bytearray:
    out = _new_rgba(w, h)
    for y in range(0, h, 4):
        for x in range(0, w, 4):
            for y2 in range(4):
                for x2 in range(4):
                    px = _u16be(data, off)
                    if px & 0x8000:  # RGB555, opaque
                        r = _exp((px >> 10) & 0x1F, 5)
                        g = _exp((px >> 5) & 0x1F, 5)
                        b = _exp(px & 0x1F, 5)
                        a = 0xFF
                    else:            # ARGB3444
                        a = _exp((px >> 12) & 0x07, 3)
                        r = _exp((px >> 8) & 0x0F, 4)
                        g = _exp((px >> 4) & 0x0F, 4)
                        b = _exp(px & 0x0F, 4)
                    if y + y2 < h and x + x2 < w:
                        _put(out, w, x + x2, y + y2, r, g, b, a)
                    off += 2
    return out


def _decode_argb8888(data: bytes, off: int, w: int, h: int) -> bytearray:
    """ARGB8888: each 4x4 block stores AR for all 16 px (32 bytes) then
    GB for all 16 px (32 bytes). Within a block, pixel k's bytes are at
    off+2k (A,R) and off+32+2k (G,B)."""
    out = _new_rgba(w, h)
    for y in range(0, h, 4):
        for x in range(0, w, 4):
            for y2 in range(4):
                for x2 in range(4):
                    a = data[off + 0]
                    r = data[off + 1]
                    g = data[off + 32]
                    b = data[off + 33]
                    if y + y2 < h and x + x2 < w:
                        _put(out, w, x + x2, y + y2, r, g, b, a)
                    off += 2
            off += 32  # skip the GB half of this block
    return out


def _decode_dxt1(data: bytes, off: int, w: int, h: int) -> bytearray:
    """GameCube CMP (S3TC/DXT1) — 8x8 macro-blocks of four 4x4 DXT1 sub-
    blocks, BIG-ENDIAN color words and a GameCube-specific bit order."""
    out = _new_rgba(w, h)
    pal = [[0, 0, 0, 0] for _ in range(4)]
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            for y2 in range(0, 8, 4):
                for x2 in range(0, 8, 4):
                    c0 = _u16be(data, off)
                    c1 = _u16be(data, off + 2)
                    pal[0] = [
                        _exp((c0 >> 11) & 0x1F, 5),
                        _exp((c0 >> 5) & 0x3F, 6),
                        _exp(c0 & 0x1F, 5),
                        0xFF,
                    ]
                    pal[1] = [
                        _exp((c1 >> 11) & 0x1F, 5),
                        _exp((c1 >> 5) & 0x3F, 6),
                        _exp(c1 & 0x1F, 5),
                        0xFF,
                    ]
                    if c0 > c1:
                        pal[2] = [
                            (pal[0][0] * 2 + pal[1][0]) // 3,
                            (pal[0][1] * 2 + pal[1][1]) // 3,
                            (pal[0][2] * 2 + pal[1][2]) // 3,
                            0xFF,
                        ]
                        pal[3] = [
                            (pal[1][0] * 2 + pal[0][0]) // 3,
                            (pal[1][1] * 2 + pal[0][1]) // 3,
                            (pal[1][2] * 2 + pal[0][2]) // 3,
                            0xFF,
                        ]
                    else:
                        pal[2] = [
                            (pal[0][0] + pal[1][0]) // 2,
                            (pal[0][1] + pal[1][1]) // 2,
                            (pal[0][2] + pal[1][2]) // 2,
                            0xFF,
                        ]
                        pal[3] = [0, 0, 0, 0]
                    off += 4
                    for y3 in range(4):
                        row = data[off]
                        for x3 in range(4):
                            idx = (row >> (6 - (x3 * 2))) & 0x03
                            px_y = y + y2 + y3
                            px_x = x + x2 + x3
                            if px_y < h and px_x < w:
                                c = pal[idx]
                                _put(out, w, px_x, px_y, c[0], c[1], c[2], c[3])
                        off += 1
    return out


_DECODERS = {
    0x00: _decode_intensity4,
    0x01: _decode_intensity8,
    0x02: _decode_intensitya4,
    0x03: _decode_intensitya8,
    0x04: _decode_rgb565,
    0x05: _decode_rgb5a3,
    0x06: _decode_argb8888,
    0x0E: _decode_dxt1,
}


def decode_gvr(data: bytes) -> Tuple[int, int, bytes]:
    """Decode a GVR (GVRT) texture to ``(width, height, rgba_bytes)``.

    Raises ValueError on a missing/truncated header and
    NotImplementedError for palettized (Index4/Index8) inputs whose
    palette isn't inline (the caller falls back to a placeholder).
    """
    body_off, w, h, data_format, _pixel_format = _parse_gvrt_header(data)
    if data_format in (0x08, 0x09):
        raise NotImplementedError(
            "GVR Index4/Index8 (palettized) not supported without an "
            "external palette"
        )
    dec = _DECODERS.get(data_format)
    if dec is None:
        raise NotImplementedError(
            f"GVR data format 0x{data_format:02X} "
            f"({GVR_DATA_FORMATS.get(data_format, 'unknown')}) not supported"
        )
    out = dec(bytes(data), body_off, w, h)
    return (w, h, bytes(out))
