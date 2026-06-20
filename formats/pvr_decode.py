'''
Pure-Python PVR (Sega Dreamcast PowerVR) texture decoder.

Ported from VincentNL/pvr2image (MIT) — see ~/Repositories/psobb-studio/_reference/pvr2image/.
The original is a single-file PNG/BMP/TGA exporter with file IO baked in;
this port refactors it into a pure decode-to-RGBA library with no disk
side effects so we can plug it into the editor's preview pipeline.

The verbatim VincentNL MIT block is reproduced in full at the top of the
file; the "decode" class implementation is a near-verbatim translation
with the file-IO side effects (save_png / save_bmp / save_tga / log file)
stripped. Detwiddle, VQ codebook expansion, palettized 4/8-bit, BUMP /
YUV420 / YUV422 paths and mipmap-skip arithmetic are all preserved with
their original constants.

Public API:

    decode_pvr(data: bytes) -> tuple[int, int, bytes]
        Decode the LARGEST mip level of a PVR file (full IFF wrapper:
        optional GBIX header + PVRT chunk). Returns (width, height,
        RGBA8 pixel buffer with len = width*height*4).

    decode_pvr_mips(data: bytes) -> list[tuple[int, int, bytes]]
        Decode a PVR with mipmaps; returns [(w, h, rgba), ...] from
        smallest to largest. For a non-mipmapped PVR returns a single
        entry. Mip count is computed from the texture-format flag and
        the largest mip's dimensions.

Supported pixel formats (px_format byte at PVRT+0x08):
    0 = ARGB1555    1 = RGB565      2 = ARGB4444
    3 = YUV422      4 = BUMP        5 = RGB555
    6 = YUV420      7 = ARGB8888
    8 = PAL-4       9 = PAL-8       10 = AUTO

Supported texture formats (tex_format byte at PVRT+0x09):
    1  Twiddled                       2  Twiddled + Mips
    3  Twiddled VQ                    4  Twiddled VQ + Mips
    5  Pal-4 Twiddled                 6  Pal-4 Twiddled + Mips
    7  Pal-8 Twiddled                 8  Pal-8 Twiddled + Mips
    9  Rectangle                      10 Rectangle + Mips
    11 Stride                         12 Stride + Mips
    13 Twiddled Rectangle             14 BMP (ARGB8888 raw)
    15 BMP + Mips                     16 SmallVQ (codebook 16/32/128/256)
    17 SmallVQ + Mips                 18 Twiddled Alias + Mips

Palettized formats (5..8) require an accompanying PVPL palette payload.
Pass it via ``decode_pvr_with_palette(data, palette_data)``; calling
``decode_pvr`` on a palettized PVR without a palette yields a greyscale
fallback (each entry maps to (i,i,i,255)) — same behaviour as the
upstream tool.
'''

# ---------------------------------------------------------------------------
# Upstream license (VincentNL/pvr2image, verbatim):
#
# MIT License
#
# Copyright (c) 2023-2024 VincentNL
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ---------------------------------------------------------------------------

from __future__ import annotations

import io
import math
import struct
from typing import List, Optional, Tuple

# Format dictionaries for diagnostics / logging.
PX_MODES = {
    0: 'ARGB1555',
    1: 'RGB565',
    2: 'ARGB4444',
    3: 'YUV422',
    4: 'BUMP',
    5: 'RGB555',
    6: 'YUV420',
    7: 'ARGB8888',
    8: 'PAL-4',
    9: 'PAL-8',
    10: 'AUTO',
}

TEX_MODES = {
    1: 'Twiddled',
    2: 'Twiddled + Mips',
    3: 'Twiddled VQ',
    4: 'Twiddled VQ + Mips',
    5: 'Twiddled Pal4 (16-col)',
    6: 'Twiddled Pal4 + Mips (16-col)',
    7: 'Twiddled Pal8 (256-col)',
    8: 'Twiddled Pal8 + Mips (256-col)',
    9: 'Rectangle',
    10: 'Rectangle + Mips',
    11: 'Stride',
    12: 'Stride + Mips',
    13: 'Twiddled Rectangle',
    14: 'BMP',
    15: 'BMP + Mips',
    16: 'Twiddled SmallVQ',
    17: 'Twiddled SmallVQ + Mips',
    18: 'Twiddled Alias + Mips',
}


# ---------------------------------------------------------------------------
# Color decoders (one entry-point per px_format).
# ---------------------------------------------------------------------------


def _read_col(px_format: int, color):
    """Decode a single source pixel value to an RGBA tuple.

    Returns (r,g,b,a) for ARGB/RGB modes (alpha set to 0xFF when the
    source format has no alpha channel) and a 6-tuple
    (r0,g0,b0,r1,g1,b1) for YUV422 (px_format=3) where ``color`` is a
    pair (yuv0, yuv1) — see the YUV422 path.
    """
    if px_format == 0:  # ARGB1555
        a = ((color >> 15) & 0x1) * 0xff
        r = int(((color >> 10) & 0x1f) * 0xff / 0x1f)
        g = int(((color >> 5) & 0x1f) * 0xff / 0x1f)
        b = int((color & 0x1f) * 0xff / 0x1f)
        return (r, g, b, a)

    elif px_format == 1:  # RGB565
        a = 0xff
        r = int(((color >> 11) & 0x1f) * 0xff / 0x1f)
        g = int(((color >> 5) & 0x3f) * 0xff / 0x3f)
        b = int((color & 0x1f) * 0xff / 0x1f)
        return (r, g, b, a)

    elif px_format == 2:  # ARGB4444
        a = ((color >> 12) & 0xf) * 0x11
        r = ((color >> 8) & 0xf) * 0x11
        g = ((color >> 4) & 0xf) * 0x11
        b = (color & 0xf) * 0x11
        return (r, g, b, a)

    elif px_format == 5:  # RGB555
        a = 0xFF
        r = int(((color >> 10) & 0x1f) * 0xff / 0x1f)
        g = int(((color >> 5) & 0x1f) * 0xff / 0x1f)
        b = int((color & 0x1f) * 0xff / 0x1f)
        return (r, g, b, a)

    elif px_format == 7:  # ARGB8888
        a = (color >> 24) & 0xFF
        r = (color >> 16) & 0xFF
        g = (color >> 8) & 0xFF
        b = color & 0xFF
        return (r, g, b, a)

    elif px_format == 14:  # RGBA8888 (BMP body word order)
        r = (color >> 24) & 0xFF
        g = (color >> 16) & 0xFF
        b = (color >> 8) & 0xFF
        a = color & 0xFF
        return (r, g, b, a)

    elif px_format == 3:  # YUV422 — color is a 2-tuple
        yuv0, yuv1 = color
        y0 = (yuv0 >> 8) & 0xFF
        u = yuv0 & 0xFF
        y1 = (yuv1 >> 8) & 0xFF
        v = yuv1 & 0xFF
        c0 = y0 - 16
        c1 = y1 - 16
        d = u - 128
        e = v - 128
        r0 = max(0, min(255, int((298 * c0 + 409 * e + 128) >> 8)))
        g0 = max(0, min(255, int((298 * c0 - 100 * d - 208 * e + 128) >> 8)))
        b0 = max(0, min(255, int((298 * c0 + 516 * d + 128) >> 8)))
        r1 = max(0, min(255, int((298 * c1 + 409 * e + 128) >> 8)))
        g1 = max(0, min(255, int((298 * c1 - 100 * d - 208 * e + 128) >> 8)))
        b1 = max(0, min(255, int((298 * c1 + 516 * d + 128) >> 8)))
        return r0, g0, b0, r1, g1, b1

    raise ValueError(f"_read_col: unsupported px_format {px_format}")


# ---------------------------------------------------------------------------
# Twiddle index table.
# ---------------------------------------------------------------------------
#
# Verbatim port of pvr2image.decode.detwiddle. The constants come from
# Sega's hardware twiddling pattern — they're not derived in the
# original either. Returns a flat list of source-pixel indices that maps
# linear (y*w+x) destination indices to twiddled source positions.


def _detwiddle(w: int, h: int) -> List[int]:
    pat2: List[int] = []
    h_inc: List[int] = []
    arr: List[int] = []
    h_arr: List[int] = []
    index = 0

    seq = [2, 6, 2, 22, 2, 6, 2]
    pat = seq + [86] + seq + [342] + seq + [86] + seq

    for i in range(4):
        pat2 += [1366, 5462, 1366, 21846]
        pat2 += [1366, 5462, 1366, 87382] if i % 2 == 0 else [1366, 5462, 1366, 349526]

    for i in range(len(pat2)):
        h_inc.extend(pat + [pat2[i]])
    h_inc.extend(pat)

    if w > h:
        ratio = int(w / h)
        if w % 32 == 0 and w & (w - 1) != 0 or h & (h - 1) != 0:
            n = h * w
            for i in range(n):
                arr.append(i)
        else:
            cur_h_inc = {w: h_inc[0:h - 1] + [2]}
            for j in range(ratio):
                if w in cur_h_inc:
                    for i in cur_h_inc[w]:
                        h_arr.append(index)
                        index += i
                index = (len(h_arr) * h)
            v_arr = [int(x / 2) for x in h_arr]
            v_arr = v_arr[0:h]
            for val in v_arr:
                arr.extend([x + val for x in h_arr])

    elif h > w:
        ratio = int(h / w)
        cur_h_inc = {w: h_inc[0:w - 1] + [2]}
        if w in cur_h_inc:
            for i in cur_h_inc[w]:
                h_arr.append(index)
                index += i
        v_arr = [int(x / 2) for x in h_arr]
        for i in range(ratio):
            if i == 0:
                last_val = 0
            else:
                last_val = arr[-1] + 1
            for val in v_arr:
                arr.extend([last_val + x + val for x in h_arr])

    elif w == h:
        cur_h_inc = {w: h_inc[0:w - 1] + [2]}
        if w in cur_h_inc:
            for i in cur_h_inc[w]:
                h_arr.append(index)
                index += i
        v_arr = [int(x / 2) for x in h_arr]
        for val in v_arr:
            arr.extend([x + val for x in h_arr])

    return arr


# ---------------------------------------------------------------------------
# BUMP map helpers (from pvr2image.process_SR / cart_to_rgb).
# ---------------------------------------------------------------------------


def _process_SR(SR_value: int) -> Tuple[float, float, float]:
    S = (1.0 - ((SR_value >> 8) / 255.0)) * math.pi / 2
    R = (SR_value & 0xFF) / 255.0 * 2 * math.pi - 2 * math.pi * (SR_value & 0xFF > math.pi)
    red = (math.sin(S) * math.cos(R) + 1.0) * 0.5
    green = (math.sin(S) * math.sin(R) + 1.0) * 0.5
    blue = (math.cos(S) + 1.0) * 0.5
    return red, green, blue


def _cart_to_rgb(cval: Tuple[float, float, float]) -> Tuple[int, int, int]:
    return tuple(int(c * 255) for c in cval)


# ---------------------------------------------------------------------------
# YUV420 (16x16-block planar). Direct port of yuv420_to_rgb.
# ---------------------------------------------------------------------------


def _yuv420_to_rgb(f: io.BytesIO, w: int, h: int) -> List[Tuple[int, int, int, int]]:
    buffer = bytearray()
    col = w // 16
    row = h // 16

    U = [bytearray() for _ in range(8 * row)]
    V = [bytearray() for _ in range(8 * row)]
    Y01 = [bytearray() for _ in range(col)]
    Y23 = [bytearray() for _ in range(col)]

    for i in range(1, row + 1):
        for n in range(8):
            U[n + 8 * (i - 1)] = bytearray()
        for n in range(col):
            Y01[n] = bytearray()
            Y23[n] = bytearray()
        for _ in range(col):
            for n in range(8):
                U[n + 8 * (i - 1)] += f.read(0x8)
            for n in range(8):
                V[n + 8 * (i - 1)] += f.read(0x8)
            for _ in range(2):
                for n in range(8):
                    Y01[n] += f.read(0x8)
            for _ in range(2):
                for n in range(8):
                    Y23[n] += f.read(0x8)
        for n in range(col):
            buffer += Y01[n]
        for n in range(col):
            buffer += Y23[n]
    for datauv in U + V:
        buffer += datauv

    Y = list(buffer[:int(w * h)])
    U = list(buffer[int(w * h):int(w * h * 1.25)])
    V = list(buffer[int(w * h * 1.25):])

    Y = [Y[i:i + w] for i in range(0, len(Y), w)]
    U = [U[i:i + w // 2] for i in range(0, len(U), w // 2)]
    V = [V[i:i + w // 2] for i in range(0, len(V), w // 2)]

    U = [item for sublist in U for item in [item for item in sublist] * 2]
    V = [item for sublist in V for item in [item for item in sublist] * 2]
    U = [U[i:i + w] for i in range(0, len(U), w)]
    V = [V[i:i + w] for i in range(0, len(V), w)]

    data: List[Tuple[int, int, int, int]] = []
    for i in range(h):
        for j in range(w):
            i_UV = min(i // 2, len(U) - 1)
            j_UV = min(j // 2, len(U[i_UV]) - 1)
            y, u, v = Y[i][j], U[i_UV][j_UV], V[i_UV][j_UV]
            r = int(max(0, min(255, round(y + 1.402 * (v - 128)))))
            g = int(max(0, min(255, round(y - 0.344136 * (u - 128) - 0.714136 * (v - 128)))))
            b = int(max(0, min(255, round(y + 1.772 * (u - 128)))))
            data.append((r, g, b, 0xFF))
    return data


# ---------------------------------------------------------------------------
# Palette helpers (read PVPL → 256-entry RGB palette).
# ---------------------------------------------------------------------------


def _read_pal_color(mode: int, color: int) -> Tuple[int, int, int, int]:
    """Decode one palette entry. Returns (r,g,b,a).

    Ports pvr2image.read_pal but emits RGBA so palettized PVRs come back
    in the same shape as direct-color ones.
    """
    if mode == 4444:
        red = ((color >> 8) & 0xf) << 4
        green = ((color >> 4) & 0xf) << 4
        blue = (color & 0xf) << 4
        alpha = ((color >> 12) & 0xf) << 4
        return (red, green, blue, alpha)
    if mode == 555:
        red = ((color >> 10) & 0x1f) << 3
        green = ((color >> 5) & 0x1f) << 3
        blue = (color & 0x1f) << 3
        return (red, green, blue, 0xFF)
    if mode == 565:
        red = ((color >> 11) & 0x1f) << 3
        green = ((color >> 5) & 0x3f) << 2
        blue = (color & 0x1f) << 3
        return (red, green, blue, 0xFF)
    if mode == 8888:
        blue = (color >> 0) & 0xFF
        green = (color >> 8) & 0xFF
        red = (color >> 16) & 0xFF
        alpha = (color >> 24) & 0xFF
        return (red, green, blue, alpha)
    raise ValueError(f"_read_pal_color: unsupported pvp mode {mode}")


def _parse_pvp(data: bytes) -> Optional[List[Tuple[int, int, int, int]]]:
    """Parse a PVPL palette file → list of (r,g,b,a) entries.

    Returns None if the magic is wrong. Raises ValueError on truncation.
    """
    if len(data) < 0x10 or data[:4] != b"PVPL":
        return None
    pixel_type = data[0x08]
    if pixel_type == 1:
        mode = 565
    elif pixel_type == 2:
        mode = 4444
    elif pixel_type == 6:
        mode = 8888
    else:
        mode = 555
    ttl_entries = struct.unpack_from("<H", data, 0x0E)[0]

    pal: List[Tuple[int, int, int, int]] = []
    pos = 0x10
    for _ in range(ttl_entries):
        if mode != 8888:
            if pos + 2 > len(data):
                raise ValueError(f"PVPL truncated at entry {len(pal)}")
            color = struct.unpack_from("<H", data, pos)[0]
            pos += 2
        else:
            if pos + 4 > len(data):
                raise ValueError(f"PVPL truncated at entry {len(pal)}")
            color = struct.unpack_from("<I", data, pos)[0]
            pos += 4
        pal.append(_read_pal_color(mode, color))
    return pal


# ---------------------------------------------------------------------------
# Mip-skip arithmetic — copy of pvr2image.load_pvr's dimension table.
# ---------------------------------------------------------------------------


def _mip_skip(tex_format: int, w: int) -> int:
    """How many bytes to skip past in the body to land on the largest mip.

    PVRT mipmapped textures store ALL mips concatenated, smallest first
    (4×4 / 2×2 / 1×1 + extra). Returns the byte offset of the largest
    mip relative to the start of the body. Returns 0 for non-mipmapped.
    """
    if tex_format not in [2, 4, 6, 8, 10, 12, 15, 17, 18]:
        return 0
    pvr_dim = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    mip_size = [0x20, 0x80, 0x200, 0x800, 0x2000, 0x8000, 0x20000, 0x80000]
    size_adjust = {2: 4, 6: 1, 8: 2, 10: 4, 15: 8, 18: 4}
    extra_mip = {2: 0x2c, 6: 0xc, 8: 0x18, 10: 0x2c, 15: 0x54, 18: 0x30}
    # VQ mip-skip uses a different scale (codebook is shared across mips).
    if tex_format in (4, 17):
        vq_mip_size = [0x10, 0x40, 0x100, 0x400, 0x1000, 0x4000, 0x10000, 0x40000]
        vq_extra = {4: 0x6, 17: 0x6}
        mip_index = 0
        for i, dim in enumerate(pvr_dim):
            if dim == w:
                mip_index = i - 1
                break
        return sum(vq_mip_size[:mip_index]) * 1 + vq_extra[tex_format]
    mip_index = 0
    for i, dim in enumerate(pvr_dim):
        if dim == w:
            mip_index = i - 1
            break
    return sum(mip_size[:mip_index]) * size_adjust[tex_format] + extra_mip[tex_format]


# ---------------------------------------------------------------------------
# Body decoder — returns a list of RGBA pixel tuples.
# ---------------------------------------------------------------------------


def _decode_body(
    f: io.BytesIO,
    w: int,
    h: int,
    px_format: int,
    tex_format: int,
    palette: Optional[List[Tuple[int, int, int, int]]] = None,
) -> List[Tuple[int, int, int, int]]:
    """Decode the PVR body cursor to a list of (r,g,b,a) tuples in row order.

    Cursor is expected to be positioned at the start of the LARGEST mip
    (mip-skip already applied by the caller). Out is row-major
    (top-to-bottom, left-to-right) — same convention as Pillow / numpy.
    """
    twiddled = tex_format not in [9, 10, 11, 12, 14, 15]
    arr = _detwiddle(w, h) if twiddled else None

    # ----- Palettized 4 / 8-bit -----
    if tex_format in (5, 6, 7, 8):
        if tex_format in (7, 8):
            pixels = list(f.read(w * h))
            indices = [pixels[i] for i in arr]
            palette_entries = 256
        else:
            raw = bytearray(f.read(w * h // 2))
            unpacked: List[int] = []
            for byte in raw:
                unpacked.append(byte & 0x0f)
                unpacked.append((byte >> 4) & 0x0f)
            indices = [unpacked[i] for i in arr]
            palette_entries = 16

        if palette is None:
            # Greyscale fallback (alpha 255), matches pvr2image's default.
            if palette_entries == 16:
                palette = [(i * 17, i * 17, i * 17, 255) for i in range(16)]
            else:
                palette = [(i, i, i, 255) for i in range(256)]

        out: List[Tuple[int, int, int, int]] = []
        for idx in indices:
            if 0 <= idx < len(palette):
                out.append(palette[idx])
            else:
                out.append((0, 0, 0, 0))
        return out

    # ----- VQ / SmallVQ -----
    if tex_format in (3, 4, 16, 17):
        if tex_format == 16:
            if w <= 16:
                codebook_size = 16
            elif w == 32:
                codebook_size = 32
            elif w == 64:
                codebook_size = 128
            else:
                codebook_size = 256
        elif tex_format == 17:
            if w <= 16:
                codebook_size = 16
            elif w == 32:
                codebook_size = 64
            else:
                codebook_size = 256
        else:
            codebook_size = 256

        codebook: List[List[Tuple[int, int, int, int]]] = []
        if px_format != 3:
            for _ in range(codebook_size):
                block = []
                for _i in range(4):
                    pixel = int.from_bytes(f.read(2), 'little')
                    block.append(_read_col(px_format, pixel))
                codebook.append(block)
        else:
            for _ in range(codebook_size):
                block = []
                for _i in range(4):
                    pixel = int.from_bytes(f.read(2), 'little')
                    block.append(pixel)
                r0, g0, b0, r1, g1, b1 = _read_col(px_format, (block[0], block[3]))
                r2, g2, b2, r3, g3, b3 = _read_col(px_format, (block[1], block[2]))
                codebook.append([
                    (r0, g0, b0, 0xFF),
                    (r2, g2, b2, 0xFF),
                    (r3, g3, b3, 0xFF),
                    (r1, g1, b1, 0xFF),
                ])

        # VQ mip-skip happens AFTER codebook (per pvr2image).
        if tex_format in (4, 17):
            f.seek(f.tell() + _mip_skip(tex_format, w))

        bytes_to_read = (w * h) // 4
        pixel_indices = list(f.read(bytes_to_read))

        # Detwiddle index image (half-resolution).
        twi_arr = _detwiddle(w // 2, h // 2)
        image_array = [[(0, 0, 0, 0)] * w for _ in range(h)]
        i_ = 0
        for y in range(h // 2):
            for x in range(w // 2):
                cb = codebook[pixel_indices[twi_arr[i_]]]
                image_array[y * 2][x * 2] = cb[0]
                image_array[y * 2 + 1][x * 2] = cb[1]
                image_array[y * 2][x * 2 + 1] = cb[2]
                image_array[y * 2 + 1][x * 2 + 1] = cb[3]
                i_ += 1
        return [px for row in image_array for px in row]

    # ----- BMP / ARGB8888 raw -----
    if tex_format in (14, 15):
        pixels = [int.from_bytes(f.read(4), 'little') for _ in range(w * h)]
        return [_read_col(14, p) for p in pixels]

    # ----- BUMP map -----
    if px_format == 4:
        pixels = [int.from_bytes(f.read(2), 'little') for _ in range(w * h)]
        if twiddled:
            pixels = [pixels[i] for i in arr]
        out = []
        for p in pixels:
            r, g, b = _cart_to_rgb(_process_SR(p))
            out.append((r, g, b, 0xFF))
        return out

    # ----- ARGB modes (16-bit / 18 = twiddled-alias-mips) -----
    if px_format in (0, 1, 2, 5, 7, 18):
        pixels = [int.from_bytes(f.read(2), 'little') for _ in range(w * h)]
        if twiddled:
            ordered = [pixels[i] for i in arr]
        else:
            ordered = pixels
        return [_read_col(px_format, p) for p in ordered]

    # ----- YUV420 -----
    if px_format == 6:
        return _yuv420_to_rgb(f, w, h)

    # ----- YUV422 -----
    if px_format == 3:
        out = []
        if twiddled:
            i_ = 0
            offset_in = f.tell()
            for _y in range(h):
                for _x in range(0, w, 2):
                    f.seek(offset_in + (arr[i_] * 2))
                    yuv0 = int.from_bytes(f.read(2), 'little')
                    i_ += 1
                    f.seek(offset_in + (arr[i_] * 2))
                    yuv1 = int.from_bytes(f.read(2), 'little')
                    r0, g0, b0, r1, g1, b1 = _read_col(px_format, (yuv0, yuv1))
                    out.append((r0, g0, b0, 0xFF))
                    out.append((r1, g1, b1, 0xFF))
                    i_ += 1
        else:
            for _y in range(h):
                for _x in range(0, w, 2):
                    yuv0 = int.from_bytes(f.read(2), 'little')
                    yuv1 = int.from_bytes(f.read(2), 'little')
                    r0, g0, b0, r1, g1, b1 = _read_col(px_format, (yuv0, yuv1))
                    out.append((r0, g0, b0, 0xFF))
                    out.append((r1, g1, b1, 0xFF))
        return out

    raise ValueError(
        f"_decode_body: unsupported tex_format={tex_format} px_format={px_format}"
    )


# ---------------------------------------------------------------------------
# Header walker — parse GBIX + PVRT, locate body offset, w/h, formats.
# ---------------------------------------------------------------------------


def _parse_pvr_header(data: bytes):
    """Locate the PVRT chunk and read the 8-byte header.

    Returns (body_offset, width, height, px_format, tex_format).
    Raises ValueError on missing magic or truncation.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError("decode_pvr: input must be bytes-like")
    buf = bytes(data)
    if len(buf) < 0x10:
        raise ValueError(f"PVR too small: {len(buf)} bytes")

    # Locate the PVRT chunk deterministically, mirroring VrSharp's
    # PvrTexture.Initalize (PvrTexture.cs:115-134). The GBIX global index is
    # an arbitrary u32 that may itself contain the bytes 'PVRT', so we must
    # NOT scan for the magic — we compute the PVRT offset from the GBIX
    # length field instead.
    #   GBIX at 0x00:  pvrtOffset = 0x08 + le_u32(GBIX+0x04)
    #   GBIX at 0x04:  pvrtOffset = 0x0C + le_u32(GBIX+0x08)   (4-byte prefix)
    #   PVRT at 0x04:  pvrtOffset = 0x04                       (RLE prefix)
    #   PVRT at 0x00:  pvrtOffset = 0x00
    def _le32(off: int) -> int:
        return struct.unpack_from("<I", buf, off)[0]

    if buf[0x00:0x04] == b"GBIX" and len(buf) >= 0x0C:
        pvrt_off = 0x08 + _le32(0x04)
    elif len(buf) >= 0x10 and buf[0x04:0x08] == b"GBIX":
        pvrt_off = 0x0C + _le32(0x08)
    elif len(buf) >= 0x08 and buf[0x04:0x08] == b"PVRT":
        pvrt_off = 0x04
    elif buf[0x00:0x04] == b"PVRT":
        pvrt_off = 0x00
    else:
        # Last-resort scan kept only for inputs with neither a GBIX nor a
        # PVRT at the canonical positions (defensive; not normally reached).
        pvrt_off = buf.find(b"PVRT")
        if pvrt_off < 0:
            raise ValueError("PVR: 'PVRT' magic not found")

    if pvrt_off < 0 or pvrt_off + 0x10 > len(buf):
        raise ValueError("PVR: PVRT header truncated")
    if buf[pvrt_off:pvrt_off + 4] != b"PVRT":
        raise ValueError("PVR: 'PVRT' magic not found at computed offset")

    # PVRT layout:
    #   +0x00 'PVRT'
    #   +0x04 chunk size (u32, body_size + 8)
    #   +0x08 px_format (u8) tex_format (u8) padding (u16)
    #   +0x0C width (u16)  height (u16)
    #   +0x10 body
    px_format = buf[pvrt_off + 0x08]
    tex_format = buf[pvrt_off + 0x09]
    width, height = struct.unpack_from("<HH", buf, pvrt_off + 0x0C)
    body_off = pvrt_off + 0x10
    return body_off, width, height, px_format, tex_format


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def decode_pvr(
    data: bytes,
    palette_data: Optional[bytes] = None,
) -> Tuple[int, int, bytes]:
    """Decode a PVR file's largest mip to a (w, h, RGBA8) tuple.

    ``data`` is the full file contents (optional GBIX header + PVRT
    chunk). ``palette_data`` is the PVPL companion file's bytes if
    palettized — pass ``None`` for non-palettized formats or to use the
    greyscale fallback for palettized ones.

    Pixel buffer is row-major, top-to-bottom, RGBA8 (length =
    width*height*4).

    Raises ValueError on truncated / unsupported input.
    """
    body_off, w, h, px_format, tex_format = _parse_pvr_header(data)
    palette = _parse_pvp(palette_data) if palette_data else None

    f = io.BytesIO(bytes(data))
    # Mirror pvr2image.load_pvr: for the NON-VQ mipmapped formats the
    # to-largest-mip skip is applied to the body offset *here*; for VQ /
    # SmallVQ mips (tex 4 / 17) the skip happens INSIDE _decode_body, after
    # the codebook is read (see the VQ branch). Applying _mip_skip here for
    # VQ too would double-count it and truncate the index map.
    if tex_format in (4, 17):
        f.seek(body_off)
    else:
        f.seek(body_off + _mip_skip(tex_format, w))
    pixels = _decode_body(f, w, h, px_format, tex_format, palette=palette)

    out = bytearray(w * h * 4)
    for i, (r, g, b, a) in enumerate(pixels):
        out[i * 4 + 0] = r & 0xFF
        out[i * 4 + 1] = g & 0xFF
        out[i * 4 + 2] = b & 0xFF
        out[i * 4 + 3] = a & 0xFF
    return (w, h, bytes(out))


def decode_pvr_with_palette(data: bytes, palette_data: bytes) -> Tuple[int, int, bytes]:
    """Convenience alias for ``decode_pvr(data, palette_data=palette_data)``."""
    return decode_pvr(data, palette_data=palette_data)


# Source bits-per-pixel for each tex_format, mirroring VrSharp's
# PvrDataCodec.<codec>.Bpp. ARGB / direct-color codecs inherit the pixel
# codec's bpp (16 for the 16-bit formats, 32 for ARGB8888 / BMP); VQ and
# SmallVQ are 2; Index4 is 4; Index8 is 8. Used by the mipmapOffsets walk.
def _tex_bpp(px_format: int, tex_format: int) -> int:
    # Palettized: bpp is fixed by the index width, independent of px_format.
    if tex_format in (5, 6):      # Index4 (+mips)
        return 4
    if tex_format in (7, 8):      # Index8 (+mips)
        return 8
    if tex_format in (3, 4, 16, 17):  # VQ / SmallVQ (+mips)
        return 2
    # Direct-color (incl. BMP): inherit the pixel codec's bpp.
    if px_format == 7:            # ARGB8888 -> 32bpp
        return 32
    if px_format == 6:            # YUV420 (no real VrSharp codec) -> treat as 16
        return 16
    # ARGB1555 / RGB565 / ARGB4444 / YUV422 / BUMP / RGB555 -> 16bpp.
    return 16


def _vrsharp_mipmap_offsets(tex_w: int, px_format: int, tex_format: int) -> List[int]:
    """Per-level body offsets, LARGEST-FIRST (index 0 == tex_w mip).

    Direct port of PvrTexture.Initalize's mipmapOffsets loop
    (_reference/.../Pvr/PvrTexture.cs:236-255). Offsets are relative to the
    PVRT *data* offset — i.e. past any VQ codebook / embedded palette, which
    for our purposes equals the body start for the non-codebook formats and
    body start + codebook for VQ (the caller adds the codebook size).

      offset[i] for i = len-1 .. 0, size = 1, 2, 4, ...:
        offset[i] = acc
        acc += max((size * size * bpp) >> 3, 1)
      with a leading pad of (bpp>>3) for SquareTwiddledMipmaps (tex 2) and
      (3*bpp)>>3 for SquareTwiddledMipmapsAlt / Index8Mipmaps / Index4Mipmaps
      (tex 18 / 8 / 6). All other mipmapped formats start the pad at 0.
    """
    bpp = _tex_bpp(px_format, tex_format)
    n = tex_w.bit_length()  # == log2(tex_w) + 1
    offs = [0] * n
    acc = 0
    if tex_format == 2:               # SquareTwiddledMipmaps
        acc = bpp >> 3
    elif tex_format in (18, 8, 6):    # Alt / Index8Mipmaps / Index4Mipmaps
        acc = (3 * bpp) >> 3
    size = 1
    for i in range(n - 1, -1, -1):
        offs[i] = acc
        acc += max((size * size * bpp) >> 3, 1)
        size <<= 1
    return offs


def _read_vq_codebook(f: io.BytesIO, px_format: int, tex_format: int, w: int):
    """Read & decode a VQ / SmallVQ codebook from the cursor.

    Returns (codebook, codebook_byte_size). Each codebook entry is a list of
    4 RGBA tuples. Mirrors the codebook-read prologue of ``_decode_body``'s
    VQ branch so the per-mip walk can share one codebook across levels (as
    VrSharp does — the codebook precedes the mip index maps).
    """
    if tex_format == 16:
        codebook_size = 16 if w <= 16 else 32 if w == 32 else 128 if w == 64 else 256
    elif tex_format == 17:
        codebook_size = 16 if w <= 16 else 64 if w == 32 else 256
    else:
        codebook_size = 256

    start = f.tell()
    codebook: List[List[Tuple[int, int, int, int]]] = []
    if px_format != 3:
        for _ in range(codebook_size):
            block = []
            for _i in range(4):
                pixel = int.from_bytes(f.read(2), 'little')
                block.append(_read_col(px_format, pixel))
            codebook.append(block)
    else:
        for _ in range(codebook_size):
            block = []
            for _i in range(4):
                block.append(int.from_bytes(f.read(2), 'little'))
            r0, g0, b0, r1, g1, b1 = _read_col(px_format, (block[0], block[3]))
            r2, g2, b2, r3, g3, b3 = _read_col(px_format, (block[1], block[2]))
            codebook.append([
                (r0, g0, b0, 0xFF),
                (r2, g2, b2, 0xFF),
                (r3, g3, b3, 0xFF),
                (r1, g1, b1, 0xFF),
            ])
    return codebook, f.tell() - start


def _decode_vq_level(
    index_bytes: bytes,
    codebook: List[List[Tuple[int, int, int, int]]],
    w: int,
    h: int,
) -> List[Tuple[int, int, int, int]]:
    """Expand one VQ index map (shared codebook) into RGBA, half-res detwiddle.

    Mirrors the tail of ``_decode_body``'s VQ branch. The 1×1 mip is a single
    codebook lookup (no twiddle), matching VrSharp's VqMipmaps width==1 case.
    """
    if w == 1 and h == 1:
        return [codebook[index_bytes[0]][0]]
    twi_arr = _detwiddle(w // 2, h // 2)
    image_array = [[(0, 0, 0, 0)] * w for _ in range(h)]
    i_ = 0
    for y in range(h // 2):
        for x in range(w // 2):
            cb = codebook[index_bytes[twi_arr[i_]]]
            image_array[y * 2][x * 2] = cb[0]
            image_array[y * 2 + 1][x * 2] = cb[1]
            image_array[y * 2][x * 2 + 1] = cb[2]
            image_array[y * 2 + 1][x * 2 + 1] = cb[3]
            i_ += 1
    return [px for row in image_array for px in row]


def _rgba_tuples_to_bytes(pixels: List[Tuple[int, int, int, int]]) -> bytes:
    out = bytearray(len(pixels) * 4)
    for i, (r, g, b, a) in enumerate(pixels):
        out[i * 4 + 0] = r & 0xFF
        out[i * 4 + 1] = g & 0xFF
        out[i * 4 + 2] = b & 0xFF
        out[i * 4 + 3] = a & 0xFF
    return bytes(out)


def decode_pvr_mips(
    data: bytes,
    palette_data: Optional[bytes] = None,
) -> List[Tuple[int, int, bytes]]:
    """Decode all mip levels of a PVR.

    Returns a list ordered SMALLEST-FIRST (1×1 / 2×2 / ... / w×h) so
    consumers can build a mip pyramid in the same direction as the on-disk
    layout. For non-mipmapped textures returns a single entry.

    Each tuple is (mip_w, mip_h, rgba_bytes). Palette is shared across mips.

    Per-level offsets are computed with VrSharp's mipmapOffsets formula
    (see :func:`_vrsharp_mipmap_offsets`); each level is then decoded with
    the same byte-exact :func:`_decode_body` machinery (or, for VQ, a shared
    codebook + :func:`_decode_vq_level`). Only square mipmapped textures
    carry a full pyramid; rectangular / non-square mipmapped inputs fall back
    to the largest-mip-only :func:`decode_pvr`.
    """
    body_off, w, h, px_format, tex_format = _parse_pvr_header(data)
    palette = _parse_pvp(palette_data) if palette_data else None

    # Non-mipmapped: just decode the one body.
    if tex_format not in [2, 4, 6, 8, 10, 12, 15, 17, 18]:
        return [decode_pvr(data, palette_data=palette_data)]

    # The VrSharp mipmapOffsets model (and the on-disk pyramid) is only
    # defined for SQUARE twiddled/VQ/palettized mip chains. Rectangle-mips
    # (10/12) and any non-square input have no VrSharp pyramid model here, so
    # fall back to the verified largest-mip decode rather than guess.
    if w != h or w == 0 or (w & (w - 1)) != 0 or tex_format in (10, 12):
        return [decode_pvr(data, palette_data=palette_data)]

    n = w.bit_length()                 # number of levels (1×1 .. w×w)
    sizes = [1 << k for k in range(n)]  # ascending: 1, 2, 4, ..., w

    f = io.BytesIO(bytes(data))

    # ----- VQ / SmallVQ mips (tex 4 / 17): shared codebook + per-mip maps. --
    if tex_format in (4, 17):
        f.seek(body_off)
        try:
            codebook, cb_bytes = _read_vq_codebook(f, px_format, tex_format, w)
        except Exception:
            return [decode_pvr(data, palette_data=palette_data)]
        data_off = body_off + cb_bytes
        offs = _vrsharp_mipmap_offsets(w, px_format, tex_format)  # largest-first
        mips: List[Tuple[int, int, bytes]] = []
        for ai, size in enumerate(sizes):                 # ascending
            off_idx = n - 1 - ai                           # largest-first index
            level_off = data_off + offs[off_idx]
            nbytes = max((size * size) // 4, 1)
            f.seek(level_off)
            index_bytes = f.read(nbytes)
            if len(index_bytes) < nbytes:
                return [decode_pvr(data, palette_data=palette_data)]
            try:
                pixels = _decode_vq_level(index_bytes, codebook, size, size)
            except (IndexError, ValueError):
                return [decode_pvr(data, palette_data=palette_data)]
            mips.append((size, size, _rgba_tuples_to_bytes(pixels)))
        return mips if mips else [decode_pvr(data, palette_data=palette_data)]

    # ----- Direct-color / palettized mips: per-level _decode_body. ----------
    offs = _vrsharp_mipmap_offsets(w, px_format, tex_format)  # largest-first
    mips = []
    for ai, size in enumerate(sizes):                  # ascending (smallest-first)
        off_idx = n - 1 - ai                            # largest-first index
        level_off = body_off + offs[off_idx]
        f.seek(level_off)
        if tex_format in (5, 6) and size == 1:
            # PAL4 1×1 level: _decode_body reads (w*h)//2 == 0 bytes (the
            # nibble-packing round-down), so handle it directly. VrSharp's
            # Index4.Decode reads 1 byte and takes the LOW nibble for the
            # single pixel (y2&1 == 0 -> shift 0). Mirror that here.
            one = f.read(1)
            if not one:
                return [decode_pvr(data, palette_data=palette_data)]
            idx = one[0] & 0x0F
            if palette is None:
                pixels = [(idx * 17, idx * 17, idx * 17, 255)]
            elif 0 <= idx < len(palette):
                pixels = [palette[idx]]
            else:
                pixels = [(0, 0, 0, 0)]
            mips.append((size, size, _rgba_tuples_to_bytes(pixels)))
            continue
        try:
            pixels = _decode_body(f, size, size, px_format, tex_format, palette=palette)
        except (ValueError, IndexError):
            # Should not happen for a well-formed pyramid; bail to the safe path.
            return [decode_pvr(data, palette_data=palette_data)]
        mips.append((size, size, _rgba_tuples_to_bytes(pixels)))

    return mips if mips else [decode_pvr(data, palette_data=palette_data)]


# ---------------------------------------------------------------------------
# Synthetic PVR builder — used by tests to produce known-good fixtures
# without requiring real PSOBB data files (PSOBB BB ships XVR, not PVR).
# Documented separately so we can call out that this is NOT verbatim
# from VincentNL's reference (the upstream tool is decode-only).
# ---------------------------------------------------------------------------


def make_test_pvr(
    width: int,
    height: int,
    rgba_pixels: bytes,
    *,
    px_format: int = 7,  # ARGB8888
    tex_format: int = 14,  # BMP raw
) -> bytes:
    """Produce a self-contained PVR file from RGBA8 pixel data.

    Useful for tests / synthetic round-trip — the upstream pvr2image
    project is decode-only, so this is editor-side helper machinery.

    Supported (px_format, tex_format) combinations:
      (7, 14)  ARGB8888 BMP (raw, non-twiddled, lossless)
      (0, 9)   ARGB1555 Rectangle (non-twiddled; alpha 1-bit, RGB 5-bit)
      (1, 9)   RGB565   Rectangle (non-twiddled; alpha forced to 0xFF)
      (2, 9)   ARGB4444 Rectangle (non-twiddled; 4 bits per channel)

    The header layout matches what ``decode_pvr`` reads; round-tripping
    ``make_test_pvr → decode_pvr`` yields equivalent pixel bytes
    (modulo the format's natural quantisation — only px=7 is lossless).
    """
    if len(rgba_pixels) != width * height * 4:
        raise ValueError(
            f"rgba_pixels length {len(rgba_pixels)} != {width*height*4}"
        )

    body = bytearray()
    if px_format == 7 and tex_format == 14:
        # 4 bytes per pixel — disk order [A, B, G, R] (little-endian
        # word read by _read_col(14, ...) maps to (r,g,b,a)).
        for i in range(width * height):
            r = rgba_pixels[i * 4 + 0]
            g = rgba_pixels[i * 4 + 1]
            b = rgba_pixels[i * 4 + 2]
            a = rgba_pixels[i * 4 + 3]
            body += bytes([a, b, g, r])
    elif px_format == 0 and tex_format == 9:
        # ARGB1555: 2 bytes per pixel, little-endian. Bit 15=A, 14..10=R,
        # 9..5=G, 4..0=B.
        for i in range(width * height):
            r = rgba_pixels[i * 4 + 0]
            g = rgba_pixels[i * 4 + 1]
            b = rgba_pixels[i * 4 + 2]
            a = rgba_pixels[i * 4 + 3]
            word = ((1 if a >= 0x80 else 0) << 15) | \
                   ((r >> 3) << 10) | \
                   ((g >> 3) << 5) | \
                   (b >> 3)
            body += struct.pack("<H", word)
    elif px_format == 1 and tex_format == 9:
        # RGB565: 2 bytes per pixel. Bit 15..11=R, 10..5=G, 4..0=B.
        for i in range(width * height):
            r = rgba_pixels[i * 4 + 0]
            g = rgba_pixels[i * 4 + 1]
            b = rgba_pixels[i * 4 + 2]
            word = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
            body += struct.pack("<H", word)
    elif px_format == 2 and tex_format == 9:
        # ARGB4444: 2 bytes per pixel. Bit 15..12=A, 11..8=R, 7..4=G,
        # 3..0=B.
        for i in range(width * height):
            r = rgba_pixels[i * 4 + 0]
            g = rgba_pixels[i * 4 + 1]
            b = rgba_pixels[i * 4 + 2]
            a = rgba_pixels[i * 4 + 3]
            word = ((a >> 4) << 12) | ((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4)
            body += struct.pack("<H", word)
    else:
        raise NotImplementedError(
            f"make_test_pvr: unsupported (px={px_format}, tex={tex_format})"
        )

    header = bytearray()
    header += b"PVRT"
    header += struct.pack("<I", 8 + len(body))
    header += bytes([px_format, tex_format, 0, 0])
    header += struct.pack("<HH", width, height)
    return bytes(header) + bytes(body)


__all__ = [
    "PX_MODES",
    "TEX_MODES",
    "decode_pvr",
    "decode_pvr_with_palette",
    "decode_pvr_mips",
    "make_test_pvr",
]
