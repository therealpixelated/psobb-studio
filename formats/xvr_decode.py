"""
In-process XVM/XVR texture codec — faithful port of VrSharp (PuyoTools).

Reference (ground truth):
    _reference/PSO2-Aqua-Library/SAToolsShared/VrSharp/Xvr/
        XvrTexture.cs         (decode + pixelFormat table)
        XvrTextureEncoder.cs  (encode: BC1 opaque / BC3 alpha)
        XvrFormats.cs         (flags: Mips=1, Alpha=2)

This module REPLACES the earlier homegrown decoder that only handled
fmt 2/3/6/7 and emitted a 1x1 magenta placeholder for everything else
(silently corrupting any vanilla texture in another format — a real bug
for an upscaling pipeline). It also replaces the encode path that shelled
out to a `dxt_encoder.exe` which no longer exists on disk.

Authoritative XVR pixelFormat table (from the PSO1 executable, per
XvrTexture.cs GetFormat()). Indices 11-14 are aliases of 1-4:

    1, 11  D3DFMT_A8R8G8B8   32bpp BGRA
    2, 12  D3DFMT_R5G6B5     16bpp
    3, 13  D3DFMT_A1R5G5B5   16bpp
    4, 14  D3DFMT_A4R4G4B4   16bpp
    5      D3DFMT_P8         8bpp palettized   (needs palette; unsupported)
    6      D3DFMT_DXT1       BC1
    7      D3DFMT_DXT2       BC2 (DXT3 block layout, premultiplied alpha)
    8      D3DFMT_DXT3       BC2
    9      D3DFMT_DXT4       BC3 (DXT5 block layout, premultiplied alpha)
    10     D3DFMT_DXT5       BC3
    15     D3DFMT_YUY2       (unsupported)
    16     D3DFMT_V8U8       (unsupported)
    17     D3DFMT_A8         8bpp alpha
    18     D3DFMT_X1R5G5B5   16bpp, alpha forced opaque
    19     D3DFMT_X8R8G8B8   32bpp BGRX, alpha forced opaque

Public surface (kept stable for server.py / sibling_archives.py / tests):
    parse_xvm(blob)          -> list[dict] (idx, offset, width, height, fmt,
                                            flags, alpha, mips, data, header)
    decode_xvr(record_dict)  -> (w, h, RGBA8 bytes)   [base mip only]
    extract_to_dir(blob, out_dir, stem) -> tile manifest
    build_xvr_record(img, *, has_alpha=None, has_mips=False) -> bytes
    build_xvm(header_blob, records) -> bytes

XVRT record header (0x40 bytes), consistent with the encoder:
    0x00  'XVRT'
    0x04  size_after  (== record_len - 8)
    0x08  flags       (int32: bit0 Mips, bit1 Alpha)
    0x0C  pixelFormat (int32)
    0x10  globalIndex (GBIX)
    0x14  width        (uint16)
    0x16  height       (uint16)
    0x18  data_size    (== record_len - 0x40)
    0x1C..0x40  zero padding
"""
from __future__ import annotations

import hashlib
import io
import re
import struct
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from PIL import Image

XVR_MAGIC = b"XVRT"
XVM_MAGIC = b"XVMH"

XVRT_HEADER_SIZE = 0x40

# --- pixelFormat constants (faithful to the PSO executable table) ---------
FMT_A8R8G8B8 = 1
FMT_RGB565 = 2
FMT_ARGB1555 = 3
FMT_ARGB4444 = 4
FMT_P8 = 5
FMT_DXT1 = 6
FMT_DXT2 = 7
FMT_DXT3 = 8
FMT_DXT4 = 9
FMT_DXT5 = 10
FMT_A8R8G8B8_ALT = 11
FMT_RGB565_ALT = 12
FMT_ARGB1555_ALT = 13
FMT_ARGB4444_ALT = 14
FMT_YUY2 = 15
FMT_V8U8 = 16
FMT_A8 = 17
FMT_X1R5G5B5 = 18
FMT_X8R8G8B8 = 19

# Flags word at 0x08 (XvrFormats.XvrFlags).
FLAG_MIPS = 0x1
FLAG_ALPHA = 0x2

# Decode category per pixelFormat. "block:<fourcc>:<bpb>" routes through a
# DDS wrapper + Pillow; the rest are decoded directly.
#   bpb = bytes per 4x4 block (BC1=8, BC2/BC3=16).
_BLOCK = {
    FMT_DXT1: (b"DXT1", 8),
    FMT_DXT2: (b"DXT3", 16),   # BC2
    FMT_DXT3: (b"DXT3", 16),   # BC2
    FMT_DXT4: (b"DXT5", 16),   # BC3
    FMT_DXT5: (b"DXT5", 16),   # BC3
}
# Uncompressed formats -> bytes per pixel (for base-mip slicing).
_BPP = {
    FMT_A8R8G8B8: 4, FMT_A8R8G8B8_ALT: 4, FMT_X8R8G8B8: 4,
    FMT_RGB565: 2, FMT_RGB565_ALT: 2,
    FMT_ARGB1555: 2, FMT_ARGB1555_ALT: 2, FMT_X1R5G5B5: 2,
    FMT_ARGB4444: 2, FMT_ARGB4444_ALT: 2,
    FMT_A8: 1,
}


def _build_dds(width: int, height: int, data: bytes, fourcc: bytes, bpb: int) -> bytes:
    """Wrap raw DXT blocks with a DDS header so Pillow can decode it."""
    flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000  # CAPS|HEIGHT|WIDTH|PF|LINEARSIZE
    linear = max(1, ((width + 3) // 4)) * max(1, ((height + 3) // 4)) * bpb
    pf = struct.pack("<II4sIIIII", 32, 0x4, fourcc, 0, 0, 0, 0, 0)
    caps = struct.pack("<IIII", 0x1000, 0, 0, 0)
    hdr = b"DDS " + struct.pack("<IIIIIII", 124, flags, height, width, linear, 0, 1)
    hdr += b"\x00" * 44 + pf + caps + b"\x00" * 4
    if len(hdr) != 128:
        raise ValueError(f"bad DDS header length {len(hdr)}")
    return hdr + data


# --- 5/6/4-bit channel expansion (bit-replication; lossless via >>) --------
def _exp5(v: int) -> int:
    return (v << 3) | (v >> 2)


def _exp6(v: int) -> int:
    return (v << 2) | (v >> 4)


def _exp4(v: int) -> int:
    return (v << 4) | v


# --- direct decoders for uncompressed formats ------------------------------
def decode_rgb565(data: bytes, width: int, height: int) -> Image.Image:
    """FMT_RGB565 (2/12): 16-bit LE R5 G6 B5, opaque."""
    n = width * height
    px = struct.unpack_from(f"<{n}H", data)
    out = bytearray(n * 4)
    for i, v in enumerate(px):
        j = i * 4
        out[j] = _exp5((v >> 11) & 0x1F)
        out[j + 1] = _exp6((v >> 5) & 0x3F)
        out[j + 2] = _exp5(v & 0x1F)
        out[j + 3] = 255
    return Image.frombytes("RGBA", (width, height), bytes(out))


def encode_rgb565(img: Image.Image) -> bytes:
    """Encode a PIL Image to FMT_RGB565 raw tile bytes."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    px = img.tobytes()
    out = bytearray(len(px) // 2)
    o = 0
    for i in range(0, len(px), 4):
        v = ((px[i] >> 3) << 11) | ((px[i + 1] >> 2) << 5) | (px[i + 2] >> 3)
        out[o] = v & 0xFF
        out[o + 1] = (v >> 8) & 0xFF
        o += 2
    return bytes(out)


def decode_argb1555(data: bytes, width: int, height: int, *, force_opaque: bool = False) -> Image.Image:
    """FMT_ARGB1555 (3/13): 16-bit LE A1 R5 G5 B5. fmt 18 (X1R5G5B5) forces opaque."""
    n = width * height
    px = struct.unpack_from(f"<{n}H", data)
    out = bytearray(n * 4)
    for i, v in enumerate(px):
        j = i * 4
        out[j] = _exp5((v >> 10) & 0x1F)
        out[j + 1] = _exp5((v >> 5) & 0x1F)
        out[j + 2] = _exp5(v & 0x1F)
        out[j + 3] = 255 if (force_opaque or (v >> 15) & 1) else 0
    return Image.frombytes("RGBA", (width, height), bytes(out))


def encode_argb1555(img: Image.Image) -> bytes:
    """Encode a PIL Image to FMT_ARGB1555 raw tile bytes (alpha threshold 128)."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    px = img.tobytes()
    out = bytearray(len(px) // 2)
    o = 0
    for i in range(0, len(px), 4):
        a = 1 if px[i + 3] >= 128 else 0
        v = ((a << 15) | ((px[i] >> 3) << 10)
             | ((px[i + 1] >> 3) << 5) | (px[i + 2] >> 3))
        out[o] = v & 0xFF
        out[o + 1] = (v >> 8) & 0xFF
        o += 2
    return bytes(out)


def decode_argb4444(data: bytes, width: int, height: int) -> Image.Image:
    """FMT_ARGB4444 (4/14): 16-bit LE A4 R4 G4 B4."""
    n = width * height
    px = struct.unpack_from(f"<{n}H", data)
    out = bytearray(n * 4)
    for i, v in enumerate(px):
        j = i * 4
        out[j] = _exp4((v >> 8) & 0xF)
        out[j + 1] = _exp4((v >> 4) & 0xF)
        out[j + 2] = _exp4(v & 0xF)
        out[j + 3] = _exp4((v >> 12) & 0xF)
    return Image.frombytes("RGBA", (width, height), bytes(out))


def decode_a8r8g8b8(data: bytes, width: int, height: int, *, force_opaque: bool = False) -> Image.Image:
    """FMT_A8R8G8B8 (1/11) / X8R8G8B8 (19): 32bpp little-endian B,G,R,A bytes."""
    n = width * height
    out = bytearray(n * 4)
    for i in range(n):
        s = i * 4
        d = i * 4
        out[d] = data[s + 2]      # R
        out[d + 1] = data[s + 1]  # G
        out[d + 2] = data[s]      # B
        out[d + 3] = 255 if force_opaque else data[s + 3]  # A
    return Image.frombytes("RGBA", (width, height), bytes(out))


def decode_a8(data: bytes, width: int, height: int) -> Image.Image:
    """FMT_A8 (17): 8bpp alpha. Color forced white so visible/upscalable."""
    n = width * height
    out = bytearray(n * 4)
    for i in range(n):
        d = i * 4
        out[d] = out[d + 1] = out[d + 2] = 255
        out[d + 3] = data[i]
    return Image.frombytes("RGBA", (width, height), bytes(out))


def parse_xvm(blob: bytes) -> list[dict]:
    """Return list of texture dicts. Each: {idx, offset, width, height, fmt,
    flags, alpha, mips, data, header}.

    `data` is the full payload after the 0x40 header (base mip + any mip
    chain). `header` is the 0x40 record header (for verbatim rebuild).
    """
    if blob[:4] != XVM_MAGIC:
        raise ValueError(f"Not an XVM file (magic={blob[:4]!r})")
    count = struct.unpack_from("<I", blob, 0x08)[0]

    # Authoritative walk: follow the XVRT chain by each record's size_after
    # (+0x04), exactly like the VrSharp validator (IsValidXvrt checks
    # size_after == len-8). Robust against 'XVRT' byte sequences inside
    # pixel data. Some vanilla Ep1 jungle XVMs under-count in the header;
    # we keep walking past the declared count as long as records are valid.
    walk_offsets: list[int] = []
    off = 0x40
    while off + XVRT_HEADER_SIZE <= len(blob):
        if blob[off:off + 4] != XVR_MAGIC:
            break
        size_after = struct.unpack_from("<I", blob, off + 4)[0]
        rec_end = off + 8 + size_after
        if size_after < (XVRT_HEADER_SIZE - 8) or rec_end > len(blob):
            break
        walk_offsets.append(off)
        off = rec_end

    scan_offsets = [m.start() for m in re.finditer(re.escape(XVR_MAGIC), blob)
                    if m.start() >= 0x40]
    if walk_offsets and len(walk_offsets) >= max(count, len(scan_offsets)) - 4:
        offsets = walk_offsets
    else:
        offsets = scan_offsets
        if abs(len(offsets) - count) > 8 and not walk_offsets:
            raise ValueError(
                f"XVM declared {count} textures but found {len(offsets)} XVRT "
                f"(walk failed, scan gap > 8, refusing to decode)")

    out: list[dict] = []
    for i, off in enumerate(offsets):
        flags = struct.unpack_from("<I", blob, off + 0x08)[0]
        fmt = struct.unpack_from("<I", blob, off + 0x0C)[0]
        w, h = struct.unpack_from("<HH", blob, off + 0x14)
        dsz = struct.unpack_from("<I", blob, off + 0x18)[0]
        data = blob[off + 0x40: off + 0x40 + dsz]
        out.append({
            "idx": i,
            "offset": off,
            "width": w,
            "height": h,
            "fmt": fmt,
            "flags": flags,
            "alpha": bool(flags & FLAG_ALPHA),
            "mips": bool(flags & FLAG_MIPS),
            "data": data,
            "header": blob[off:off + 0x40],
        })
    return out


def _base_mip_size(fmt: int, w: int, h: int) -> int:
    """Bytes occupied by the largest (base) mip level for *fmt*."""
    if fmt in _BLOCK:
        _, bpb = _BLOCK[fmt]
        return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * bpb
    if fmt in _BPP:
        return w * h * _BPP[fmt]
    return 0  # unknown: caller will reject


def decode_xvr(record: dict) -> tuple[int, int, bytes]:
    """Decode a single XVR record dict to (w, h, RGBA8 bytes), base mip only.

    Raises NotImplementedError for pixel formats VrSharp/we can't decode
    (P8 palette, YUY2, V8U8) so the caller can substitute a placeholder.
    """
    w = int(record["width"])
    h = int(record["height"])
    fmt = int(record["fmt"])
    data: bytes = record["data"]

    # Slice the base mip only; a mip chain (flags&MIPS) follows it.
    base = _base_mip_size(fmt, w, h)
    if base and len(data) >= base:
        data = data[:base]

    if fmt in _BLOCK:
        fourcc, bpb = _BLOCK[fmt]
        dds = _build_dds(w, h, data, fourcc, bpb)
        img = Image.open(io.BytesIO(dds))
        img.load()
        return w, h, img.convert("RGBA").tobytes()
    if fmt in (FMT_A8R8G8B8, FMT_A8R8G8B8_ALT):
        return w, h, decode_a8r8g8b8(data, w, h).tobytes()
    if fmt == FMT_X8R8G8B8:
        return w, h, decode_a8r8g8b8(data, w, h, force_opaque=True).tobytes()
    if fmt in (FMT_RGB565, FMT_RGB565_ALT):
        return w, h, decode_rgb565(data, w, h).tobytes()
    if fmt in (FMT_ARGB1555, FMT_ARGB1555_ALT):
        return w, h, decode_argb1555(data, w, h).tobytes()
    if fmt == FMT_X1R5G5B5:
        return w, h, decode_argb1555(data, w, h, force_opaque=True).tobytes()
    if fmt in (FMT_ARGB4444, FMT_ARGB4444_ALT):
        return w, h, decode_argb4444(data, w, h).tobytes()
    if fmt == FMT_A8:
        return w, h, decode_a8(data, w, h).tobytes()
    raise NotImplementedError(f"XVR fmt {fmt} not supported")


# --- encode -----------------------------------------------------------------
# Fresh-texture encode mirrors XvrTextureEncoder.cs (BC1 opaque / BC3 alpha).
# Rebuild encode is format-PRESERVING (re-emit each tile in its original
# pixelFormat) so _verify_rebuilt's per-tile fmt check passes.
def _pad4(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    w, h = img.size
    pw, ph = max(4, (w + 3) & ~3), max(4, (h + 3) & ~3)
    if (pw, ph) != (w, h):
        padded = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
        padded.paste(img, (0, 0))
        img = padded
    return img


def _encode_bc1(img: Image.Image) -> bytes:
    from quicktex import RawTexture
    from quicktex.s3tc.bc1 import BC1Encoder
    img = _pad4(img)
    w, h = img.size
    enc = BC1Encoder(5, BC1Encoder.ColorMode.FourColor)
    return bytes(enc.encode(RawTexture.frombytes(img.tobytes(), w, h)))


def _encode_bc3(img: Image.Image) -> bytes:
    from quicktex import RawTexture
    from quicktex.s3tc.bc3 import BC3Encoder
    img = _pad4(img)
    w, h = img.size
    return bytes(BC3Encoder(5).encode(RawTexture.frombytes(img.tobytes(), w, h)))


def _encode_bc2(img: Image.Image) -> bytes:
    """BC2 / DXT3: per 4x4 block = 8-byte explicit 4-bit alpha + 8-byte BC1
    (4-color) color block. quicktex has no BC2, so synthesize it; the color
    block reuses quicktex BC1 in forced-4-color mode (BC2 always interprets
    the color block as 4-color regardless of endpoint ordering)."""
    img = _pad4(img)
    w, h = img.size
    rgba = img.tobytes()
    color = _encode_bc1(img)            # 8 bytes per 4x4 block, row-major
    out = bytearray()
    ci = 0
    for byk in range(h // 4):
        for bxk in range(w // 4):
            ablock = bytearray(8)
            for i in range(16):
                x = bxk * 4 + (i & 3)
                y = byk * 4 + (i >> 2)
                a = rgba[(y * w + x) * 4 + 3] >> 4
                ablock[i >> 1] |= a << (4 * (i & 1))
            out += ablock
            out += color[ci:ci + 8]
            ci += 8
    return bytes(out)


def encode_argb4444(img: Image.Image) -> bytes:
    """Encode to FMT_ARGB4444 raw bytes (16-bit LE A4 R4 G4 B4)."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    px = img.tobytes()
    out = bytearray(len(px) // 2)
    o = 0
    for i in range(0, len(px), 4):
        v = (((px[i + 3] >> 4) << 12) | ((px[i] >> 4) << 8)
             | ((px[i + 1] >> 4) << 4) | (px[i + 2] >> 4))
        out[o] = v & 0xFF
        out[o + 1] = (v >> 8) & 0xFF
        o += 2
    return bytes(out)


def encode_a8r8g8b8(img: Image.Image, *, force_opaque: bool = False) -> bytes:
    """Encode to FMT_A8R8G8B8 raw bytes (32bpp little-endian B,G,R,A)."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    px = img.tobytes()
    n = len(px) // 4
    out = bytearray(n * 4)
    for i in range(n):
        s = i * 4
        out[s] = px[s + 2]      # B
        out[s + 1] = px[s + 1]  # G
        out[s + 2] = px[s]      # R
        out[s + 3] = 255 if force_opaque else px[s + 3]
    return bytes(out)


def encode_a8(img: Image.Image) -> bytes:
    """Encode to FMT_A8 raw bytes (8bpp alpha)."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    px = img.tobytes()
    return bytes(px[i * 4 + 3] for i in range(len(px) // 4))


def encode_xvr_data(img: Image.Image, fmt: int) -> bytes:
    """Encode an RGBA image to raw XVR pixel data for a specific pixelFormat."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    if fmt == FMT_DXT1:
        return _encode_bc1(img)
    if fmt in (FMT_DXT2, FMT_DXT3):
        return _encode_bc2(img)
    if fmt in (FMT_DXT4, FMT_DXT5):
        return _encode_bc3(img)
    if fmt in (FMT_A8R8G8B8, FMT_A8R8G8B8_ALT):
        return encode_a8r8g8b8(img)
    if fmt == FMT_X8R8G8B8:
        return encode_a8r8g8b8(img, force_opaque=True)
    if fmt in (FMT_RGB565, FMT_RGB565_ALT):
        return encode_rgb565(img)
    if fmt in (FMT_ARGB1555, FMT_ARGB1555_ALT, FMT_X1R5G5B5):
        return encode_argb1555(img)
    if fmt in (FMT_ARGB4444, FMT_ARGB4444_ALT):
        return encode_argb4444(img)
    if fmt == FMT_A8:
        return encode_a8(img)
    raise NotImplementedError(f"XVR encode fmt {fmt} not supported")


def _has_alpha(img: Image.Image) -> bool:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    extrema = img.getextrema()
    return len(extrema) == 4 and extrema[3][0] < 255


def encode_xvr_record(
    img: Image.Image,
    fmt: int,
    *,
    flags: int = 0,
    global_index: int = 0,
) -> bytes:
    """Encode a PIL image to a complete XVRT record (0x40 header + data) in a
    specific pixelFormat. Emits base mip only (clears the MIPS flag bit)."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    data = encode_xvr_data(img, fmt)
    w, h = img.size
    flags &= ~FLAG_MIPS  # base mip only
    record_len = 0x40 + len(data)
    hdr = bytearray(0x40)
    hdr[0:4] = XVR_MAGIC
    struct.pack_into("<I", hdr, 0x04, record_len - 8)   # size_after
    struct.pack_into("<I", hdr, 0x08, flags)
    struct.pack_into("<I", hdr, 0x0C, fmt)
    struct.pack_into("<I", hdr, 0x10, global_index & 0xFFFFFFFF)
    struct.pack_into("<HH", hdr, 0x14, w, h)
    struct.pack_into("<I", hdr, 0x18, len(data))        # data_size
    return bytes(hdr) + data


def build_xvr_record(
    img: Image.Image,
    *,
    has_alpha: Optional[bool] = None,
    global_index: int = 0,
) -> bytes:
    """Encode a fresh PIL image to a complete XVRT record, choosing the format
    the way XvrTextureEncoder.cs does: BC3/DXT5 (type 0xA) when the image has
    alpha, otherwise BC1/DXT1 (type 0x6)."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    if has_alpha is None:
        has_alpha = _has_alpha(img)
    if has_alpha:
        return encode_xvr_record(img, FMT_DXT5, flags=FLAG_ALPHA,
                                 global_index=global_index)
    return encode_xvr_record(img, FMT_DXT1, flags=0, global_index=global_index)


def build_xvm(header_blob: bytes, records) -> bytes:
    """Rebuild an XVM from an XVMH header (>=0x40) + serialized XVRT records."""
    out = bytearray(header_blob[:0x40])
    if len(out) < 0x40:
        out.extend(b"\x00" * (0x40 - len(out)))
    for rec in records:
        out.extend(rec)
    return bytes(out)


def rebuild_xvm(tiles_dir, out_path=None) -> dict:
    """Rebuild an XVM from an extracted tiles directory (replaces the lost
    ``xvr_codec.py rebuild`` CLI; runs in-process).

    For each tile: splice the original ``.xvr`` record verbatim when the tile
    PNG is unchanged (md5 matches its ``.src.md5`` sidecar and the index is
    not listed in ``.force_reencode``); otherwise re-encode the PNG, PRESERVING
    the tile's original pixelFormat so the rebuilt XVM passes per-tile fmt
    verification. Returns {spliced, reencoded, data}.
    """
    tiles_dir = Path(tiles_dir)
    hdrs = sorted(tiles_dir.glob("*.xvmhdr"))
    if not hdrs:
        raise FileNotFoundError(f"no .xvmhdr in {tiles_dir}")
    stem = hdrs[0].name[:-len(".xvmhdr")]
    header = hdrs[0].read_bytes()

    force: set[int] = set()
    fmark = tiles_dir / ".force_reencode"
    if fmark.exists():
        try:
            force = {int(x) for x in fmark.read_text().split()}
        except ValueError:
            force = set()

    pat = re.compile(re.escape(stem) + r"_(\d+)_(\d+)x(\d+)$")
    tiles: list[tuple[int, Path]] = []
    for xvr in tiles_dir.glob(stem + "_*.xvr"):
        m = pat.match(xvr.stem)
        if m:
            tiles.append((int(m.group(1)), xvr))
    tiles.sort(key=lambda t: t[0])

    records: list[bytes] = []
    spliced = reencoded = 0
    for idx, xvr in tiles:
        orig = xvr.read_bytes()
        png = tiles_dir / (xvr.stem + ".png")
        md5f = tiles_dir / (xvr.stem + ".src.md5")
        changed = idx in force
        if not changed and png.exists() and md5f.exists():
            cur = hashlib.md5(png.read_bytes()).hexdigest()
            if cur != md5f.read_text().strip():
                changed = True
        if changed and png.exists():
            fmt = struct.unpack_from("<I", orig, 0x0C)[0]
            flags = struct.unpack_from("<I", orig, 0x08)[0]
            gidx = struct.unpack_from("<I", orig, 0x10)[0]
            try:
                im = Image.open(png).convert("RGBA")
                records.append(encode_xvr_record(im, fmt, flags=flags,
                                                 global_index=gidx))
                reencoded += 1
            except (NotImplementedError, OSError, ValueError):
                # Can't re-encode this format/PNG — keep the original verbatim
                # rather than corrupt the archive.
                records.append(orig)
                spliced += 1
        else:
            records.append(orig)
            spliced += 1

    out = build_xvm(header, records)
    if out_path is not None:
        Path(out_path).write_bytes(out)
    return {"spliced": spliced, "reencoded": reencoded, "data": out}


def _placeholder_png(width: int, height: int) -> bytes:
    """1x1 magenta PNG — placeholder for a pixel format we cannot decode."""
    im = Image.new("RGBA", (1, 1), (255, 0, 255, 255))
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def extract_to_dir(
    blob: bytes,
    out_dir: Path,
    stem: str,
    *,
    write_md5: bool = True,
) -> list[dict]:
    """Decode every XVR record in *blob* and write the on-disk tile layout.

    Per tile: <stem>_<idx:02d>_<W>x<H>.png (decoded RGBA), .xvr (raw record),
    and .src.md5 (md5 of the PNG, lets rebuild splice untouched tiles).
    The XVMH header goes to <stem>.xvmhdr. Returns manifest dicts:
    {index, filename, width, height, fmt}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    textures = parse_xvm(blob)
    header = blob[:0x40]
    (out_dir / f"{stem}.xvmhdr").write_bytes(header)

    manifest: list[dict] = []
    for tex in textures:
        idx = tex["idx"]
        w = tex["width"]
        h = tex["height"]
        fmt = tex["fmt"]
        name = f"{stem}_{idx:02d}_{w}x{h}"
        png_path = out_dir / f"{name}.png"
        xvr_path = out_dir / f"{name}.xvr"

        # Always write the raw .xvr sibling (rebuild needs it).
        xvr_path.write_bytes(tex["header"] + tex["data"])

        try:
            dw, dh, rgba = decode_xvr(tex)
            img = Image.frombytes("RGBA", (dw, dh), rgba)
            img.save(png_path, "PNG")
            manifest.append({
                "index": idx, "filename": png_path.name,
                "width": dw, "height": dh, "fmt": fmt,
            })
        except (NotImplementedError, ValueError):
            png_path.write_bytes(_placeholder_png(w, h))
            manifest.append({
                "index": idx, "filename": png_path.name,
                "width": w, "height": h, "fmt": -1,
            })

        if write_md5:
            md5 = hashlib.md5(png_path.read_bytes()).hexdigest()
            (out_dir / f"{name}.src.md5").write_text(md5)

    manifest.sort(key=lambda t: int(t.get("index", 0)))
    return manifest


__all__ = [
    "XVR_MAGIC", "XVM_MAGIC",
    "FMT_A8R8G8B8", "FMT_RGB565", "FMT_ARGB1555", "FMT_ARGB4444", "FMT_P8",
    "FMT_DXT1", "FMT_DXT2", "FMT_DXT3", "FMT_DXT4", "FMT_DXT5",
    "FMT_YUY2", "FMT_V8U8", "FMT_A8", "FMT_X1R5G5B5", "FMT_X8R8G8B8",
    "FLAG_MIPS", "FLAG_ALPHA",
    "parse_xvm", "decode_xvr",
    "decode_rgb565", "encode_rgb565",
    "decode_argb1555", "encode_argb1555",
    "decode_argb4444", "decode_a8r8g8b8", "decode_a8",
    "encode_argb4444", "encode_a8r8g8b8", "encode_a8",
    "encode_xvr_data", "encode_xvr_record",
    "build_xvr_record", "build_xvm", "rebuild_xvm",
    "extract_to_dir", "_build_dds",
]
