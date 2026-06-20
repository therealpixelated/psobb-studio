"""Parity tests for ``formats.pvr_decode`` mipmap + header handling.

These cover the two defects the PVR audit flagged plus the VQ-mips
double-skip bug uncovered while fixing them:

  1. ``decode_pvr_mips`` must return the FULL pyramid (smallest-first,
     including the LARGEST level), byte-exact per level, for the
     mipmapped tex formats (2 / 18 / 8 / 6 direct & palettized, 4 / 17 VQ).
  2. ``_parse_pvr_header`` must read the PVRT offset from the GBIX length
     field, never scan for the 'PVRT' byte pattern (which can collide with
     an arbitrary GBIX global index).
  3. ``decode_pvr`` must not double-apply the to-largest-mip skip for VQ /
     SmallVQ (the codebook-relative skip lives inside ``_decode_body``).

Ground truth is VrSharp's mipmapOffsets formula (PvrTexture.cs:236-255 /
VrTexture.cs:413-417), re-implemented here, plus a byte-exact comparison to
the runnable pvr2image oracle when its reference tree is present.

No real PSO .pvr assets exist (the game ships XVR/XVM), so fixtures are
synthesised to the exact on-disk PVRT layout VrSharp encodes.
"""
from __future__ import annotations

import importlib.util
import io
import os
import struct

import pytest

from formats import pvr_decode as P


# ---------------------------------------------------------------------------
# VrSharp model helpers (authoritative twiddle map + mipmapOffsets).
# ---------------------------------------------------------------------------
def _twiddle_map(size):
    tm = [0] * size
    for i in range(size):
        j = 0
        k = 1
        while k <= i:
            tm[i] |= (i & k) << j
            j += 1
            k <<= 1
    return tm


def _enc_argb1555(r, g, b, a):
    return ((1 if a >= 0x80 else 0) << 15) | ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3)


def _dec_argb1555(c):
    a = ((c >> 15) & 1) * 0xFF
    r = int(((c >> 10) & 0x1F) * 0xFF / 0x1F)
    g = int(((c >> 5) & 0x1F) * 0xFF / 0x1F)
    b = int((c & 0x1F) * 0xFF / 0x1F)
    return (r, g, b, a)


def _enc_argb4444(r, g, b, a):
    return ((a >> 4) << 12) | ((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4)


def _dec_argb4444(c):
    a = ((c >> 12) & 0xF) * 0x11
    r = ((c >> 8) & 0xF) * 0x11
    g = ((c >> 4) & 0xF) * 0x11
    b = (c & 0xF) * 0x11
    return (r, g, b, a)


def _vrsharp_offsets(tex_w, bpp, data_format):
    """LARGEST-FIRST per-level offsets (PvrTexture.Initalize:236-255)."""
    n = tex_w.bit_length()
    offs = [0] * n
    acc = 0
    if data_format == 0x02:
        acc = bpp >> 3
    elif data_format in (0x12, 0x08, 0x06):
        acc = (3 * bpp) >> 3
    size = 1
    for i in range(n - 1, -1, -1):
        offs[i] = acc
        acc += max((size * size * bpp) >> 3, 1)
        size <<= 1
    return offs


def _encode_twiddled_16(level_rgba, w, enc):
    tm = _twiddle_map(w)
    dest = bytearray(w * w * 2)
    si = 0
    for y in range(w):
        for x in range(w):
            r, g, b, a = level_rgba[si:si + 4]
            word = enc(r, g, b, a)
            di = ((tm[x] << 1) | tm[y]) << 1
            dest[di] = word & 0xFF
            dest[di + 1] = (word >> 8) & 0xFF
            si += 4
    return bytes(dest)


def _encode_index8(level_idx, w):
    tm = _twiddle_map(w)
    dest = bytearray(w * w)
    si = 0
    for y in range(w):
        for x in range(w):
            dest[(tm[x] << 1) | tm[y]] = level_idx[si]
            si += 1
    return bytes(dest)


def _make_levels_16(tex_w, enc, dec):
    levels = {}
    size = tex_w
    while size >= 1:
        rgba = bytearray(size * size * 4)
        for i in range(size * size):
            r = (i * 8 + size) & 0xF8
            g = (i * 4) & 0xF8
            b = (size * 16) & 0xF8
            er, eg, eb, ea = dec(enc(r, g, b, 0xFF))
            rgba[i * 4:i * 4 + 4] = bytes([er, eg, eb, ea])
        levels[size] = bytes(rgba)
        size >>= 1
    return levels


def _build_16(tex_w, px_format, tex_format, enc, dec):
    bpp = 16
    levels = _make_levels_16(tex_w, enc, dec)
    offs = _vrsharp_offsets(tex_w, bpp, tex_format)
    body = bytearray(offs[0] + ((tex_w * tex_w * bpp) >> 3))
    n = len(offs)
    for idx in range(n):
        size = 1 << (n - 1 - idx)
        eb = _encode_twiddled_16(levels[size], size, enc)
        body[offs[idx]:offs[idx] + len(eb)] = eb
    header = b"PVRT" + struct.pack("<I", 8 + len(body)) + bytes([px_format, tex_format, 0, 0]) + struct.pack("<HH", tex_w, tex_w)
    return bytes(header) + bytes(body), levels


# ---------------------------------------------------------------------------
# 1. decode_pvr_mips full pyramid — direct-color (tex 2 / 18).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tex_format,enc,dec", [
    (2, _enc_argb1555, _dec_argb1555),
    (18, _enc_argb4444, _dec_argb4444),
])
def test_decode_pvr_mips_full_pyramid_direct_color(tex_format, enc, dec):
    px = 0 if tex_format == 2 else 2
    pvr, levels = _build_16(8, px, tex_format, enc, dec)

    # largest-mip decode_pvr is byte-exact.
    w, h, rgba = P.decode_pvr(pvr)
    assert (w, h) == (8, 8)
    assert rgba == levels[8]

    mips = P.decode_pvr_mips(pvr)
    assert [(mw, mh) for (mw, mh, _) in mips] == [(1, 1), (2, 2), (4, 4), (8, 8)]
    # LARGEST level present and correct (the bug dropped it).
    assert mips[-1][2] == levels[8]
    for mw, mh, mb in mips:
        assert mb == levels[mw], f"mip {mw}x{mh} mismatch"


# ---------------------------------------------------------------------------
# 2. decode_pvr_mips full pyramid — palettized PAL8 (tex 8).
# ---------------------------------------------------------------------------
def _build_pal8(tex_w):
    offs = _vrsharp_offsets(tex_w, 8, 0x08)
    body = bytearray(offs[0] + tex_w * tex_w)
    idx_levels = {}
    size = tex_w
    while size >= 1:
        idx = bytes([(i * 7 + size) & 0xFF for i in range(size * size)])
        idx_levels[size] = idx
        size >>= 1
    n = len(offs)
    for idx_i in range(n):
        sz = 1 << (n - 1 - idx_i)
        eb = _encode_index8(idx_levels[sz], sz)
        body[offs[idx_i]:offs[idx_i] + len(eb)] = eb
    header = b"PVRT" + struct.pack("<I", 8 + len(body)) + bytes([9, 8, 0, 0]) + struct.pack("<HH", tex_w, tex_w)

    # PVPL palette (8888): ttl entries at 0x0E, entries at 0x10.
    pvpl = bytearray(b"PVPL" + struct.pack("<I", 0) + bytes([6, 0]) + struct.pack("<HHH", 0, 0, 256))
    for i in range(256):
        b = (i * 3) & 0xFF
        g = (255 - i) & 0xFF
        r = i & 0xFF
        word = (0xFF << 24) | (r << 16) | (g << 8) | b
        pvpl += struct.pack("<I", word)

    def pal_decode(idx_bytes):
        out = bytearray(len(idx_bytes) * 4)
        for k, ix in enumerate(idx_bytes):
            b = (ix * 3) & 0xFF
            g = (255 - ix) & 0xFF
            r = ix & 0xFF
            out[k * 4:k * 4 + 4] = bytes([r, g, b, 0xFF])
        return bytes(out)

    return bytes(header) + bytes(body), bytes(pvpl), idx_levels, pal_decode


def _build_pal4(tex_w):
    """tex6 PAL4 + mips, greyscale fallback palette (16-entry i*17)."""
    offs = _vrsharp_offsets(tex_w, 4, 0x06)
    body = bytearray(offs[0] + ((tex_w * tex_w) >> 1))
    idx_levels = {}
    size = tex_w
    while size >= 1:
        idx_levels[size] = bytes([(i * 3 + size) & 0xF for i in range(size * size)])
        size >>= 1

    def encode_index4(level_idx, w):
        if w == 1:
            return bytes([level_idx[0] & 0xF])
        tm = _twiddle_map(w)
        dest = bytearray((w * w) >> 1)
        si = 0
        for y in range(w):
            for x in range(w):
                dest[((tm[x] << 1) | tm[y]) >> 1] |= (level_idx[si] & 0xF) << ((y & 1) * 4)
                si += 1
        return bytes(dest)

    n = len(offs)
    for ii in range(n):
        sz = 1 << (n - 1 - ii)
        eb = encode_index4(idx_levels[sz], sz)
        body[offs[ii]:offs[ii] + len(eb)] = eb
    header = b"PVRT" + struct.pack("<I", 8 + len(body)) + bytes([8, 6, 0, 0]) + struct.pack("<HH", tex_w, tex_w)

    def gt(idx):
        out = bytearray(len(idx) * 4)
        for k, ix in enumerate(idx):
            out[k * 4:k * 4 + 4] = bytes([ix * 17, ix * 17, ix * 17, 255])
        return bytes(out)

    return bytes(header) + bytes(body), idx_levels, gt


def test_decode_pvr_mips_full_pyramid_pal4():
    # PAL4 + mips exercises the 1x1-level nibble special-case in decode_pvr_mips
    # (the _decode_body (w*h)//2 round-down would otherwise drop it).
    pvr, idx_levels, gt = _build_pal4(8)
    w, h, rgba = P.decode_pvr(pvr)
    assert (w, h) == (8, 8)
    assert rgba == gt(idx_levels[8])
    mips = P.decode_pvr_mips(pvr)
    assert [(mw, mh) for (mw, mh, _) in mips] == [(1, 1), (2, 2), (4, 4), (8, 8)]
    for mw, mh, mb in mips:
        assert mb == gt(idx_levels[mw]), f"PAL4 mip {mw} mismatch"


def test_decode_pvr_mips_full_pyramid_pal8():
    pvr, pvpl, idx_levels, pal_decode = _build_pal8(8)
    w, h, rgba = P.decode_pvr(pvr, palette_data=pvpl)
    assert (w, h) == (8, 8)
    assert rgba == pal_decode(idx_levels[8])
    mips = P.decode_pvr_mips(pvr, palette_data=pvpl)
    assert [(mw, mh) for (mw, mh, _) in mips] == [(1, 1), (2, 2), (4, 4), (8, 8)]
    for mw, mh, mb in mips:
        assert mb == pal_decode(idx_levels[mw])


# ---------------------------------------------------------------------------
# 3. VQ / SmallVQ mip pyramid (tex 4 / 17) — codebook + per-mip index maps.
#    Also guards the decode_pvr double-skip regression for VQ-mips.
# ---------------------------------------------------------------------------
def _enc_rgb565(r, g, b):
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def _dec_rgb565(c):
    r = int(((c >> 11) & 0x1F) * 0xFF / 0x1F)
    g = int(((c >> 5) & 0x3F) * 0xFF / 0x3F)
    b = int((c & 0x1F) * 0xFF / 0x1F)
    return (r, g, b, 0xFF)


def _vq_offsets(tex_w):
    n = tex_w.bit_length()
    offs = [0] * n
    acc = 0
    size = 1
    for i in range(n - 1, -1, -1):
        offs[i] = acc
        acc += max((size * size * 2) >> 3, 1)
        size <<= 1
    return offs


def _build_vq(tex_w, tex_format, codebook_size):
    codebook_rgba = []
    cb_bytes = bytearray()
    for c in range(codebook_size):
        block = []
        for p in range(4):
            r = (c * 4 + p * 16) & 0xF8
            g = (c * 2) & 0xFC
            b = (p * 64) & 0xF8
            w565 = _enc_rgb565(r, g, b)
            block.append(_dec_rgb565(w565))
            cb_bytes += struct.pack("<H", w565)
        codebook_rgba.append(block)

    offs = _vq_offsets(tex_w)
    n = len(offs)
    data = bytearray(offs[0] + max((tex_w * tex_w) // 4, 1))
    index_maps = {}
    for idx_i in range(n):
        size = 1 << (n - 1 - idx_i)
        nb = max((size * size) // 4, 1)
        if size == 1:
            imap = bytes([7 % codebook_size])
        else:
            half = size // 2
            tm = _twiddle_map(size)
            imap = bytearray(nb)
            for y in range(0, size, 2):
                for x in range(0, size, 2):
                    block_no = (y // 2) * half + (x // 2)
                    cb_index = (block_no * 5 + size) % codebook_size
                    imap[(tm[x >> 1] << 1) | tm[y >> 1]] = cb_index
            imap = bytes(imap)
        index_maps[size] = imap
        data[offs[idx_i]:offs[idx_i] + len(imap)] = imap

    body = bytes(cb_bytes) + bytes(data)
    header = b"PVRT" + struct.pack("<I", 8 + len(body)) + bytes([1, tex_format, 0, 0]) + struct.pack("<HH", tex_w, tex_w)
    return bytes(header) + bytes(body), codebook_rgba, index_maps


def _gt_vq_level(index_map, codebook_rgba, size):
    if size == 1:
        return bytes(bytearray(codebook_rgba[index_map[0]][0]))
    twi = P._detwiddle(size // 2, size // 2)
    img = [[(0, 0, 0, 0)] * size for _ in range(size)]
    i_ = 0
    for y in range(size // 2):
        for x in range(size // 2):
            cb = codebook_rgba[index_map[twi[i_]]]
            img[y * 2][x * 2] = cb[0]
            img[y * 2 + 1][x * 2] = cb[1]
            img[y * 2][x * 2 + 1] = cb[2]
            img[y * 2 + 1][x * 2 + 1] = cb[3]
            i_ += 1
    out = bytearray(size * size * 4)
    k = 0
    for row in img:
        for pxl in row:
            out[k:k + 4] = bytes(pxl)
            k += 4
    return bytes(out)


@pytest.mark.parametrize("tex_w,tex_format,cb_size", [
    (8, 4, 256),     # VQ + mips
    (32, 17, 64),    # SmallVQ + mips (32x32 -> codebook 64)
])
def test_decode_pvr_mips_vq_pyramid(tex_w, tex_format, cb_size):
    pvr, cb, imaps = _build_vq(tex_w, tex_format, cb_size)

    # Regression: decode_pvr must NOT double-skip (was IndexError / truncation).
    w, h, rgba = P.decode_pvr(pvr)
    assert (w, h) == (tex_w, tex_w)
    assert rgba == _gt_vq_level(imaps[tex_w], cb, tex_w)

    mips = P.decode_pvr_mips(pvr)
    expected_sizes = [(1 << k, 1 << k) for k in range(tex_w.bit_length())]
    assert [(mw, mh) for (mw, mh, _) in mips] == expected_sizes
    for mw, mh, mb in mips:
        assert mb == _gt_vq_level(imaps[mw], cb, mw), f"VQ mip {mw} mismatch"


# ---------------------------------------------------------------------------
# 4. GBIX 'PVRT'-collision header: must read offset from GBIX length field.
# ---------------------------------------------------------------------------
def test_gbix_pvrt_collision_does_not_misparse():
    # GBIX whose 8-byte global-index payload STARTS with the bytes 'PVRT'.
    gbix = b"GBIX" + struct.pack("<I", 8) + b"PVRT" + struct.pack("<I", 0)
    real = P.make_test_pvr(4, 4, bytes([0xFF, 0x00, 0x00, 0xFF]) * 16,
                           px_format=7, tex_format=14)
    colliding = gbix + real  # real PVRT lives at 0x08 + 8 = 0x10
    w, h, rgba = P.decode_pvr(colliding)
    assert (w, h) == (4, 4)
    assert len(rgba) == 4 * 4 * 4
    # first pixel is opaque red (proves we parsed the REAL body, not garbage).
    assert rgba[0:4] == bytes([0xFF, 0x00, 0x00, 0xFF])


def test_normal_gbix_prefix_still_decodes():
    gbix = b"GBIX" + struct.pack("<I", 8) + struct.pack("<II", 0xDEADBEEF, 0)
    real = P.make_test_pvr(2, 2, bytes([0x10, 0x20, 0x30, 0xFF]) * 4,
                           px_format=7, tex_format=14)
    w, h, rgba = P.decode_pvr(gbix + real)
    assert (w, h) == (2, 2)


def test_gbix_at_offset_4_prefix():
    # 4-byte prefix (e.g. RLE marker) then GBIX at 0x04.
    prefix = struct.pack("<I", 0)
    gbix = b"GBIX" + struct.pack("<I", 8) + struct.pack("<II", 1, 0)
    real = P.make_test_pvr(2, 2, bytes([0x00, 0x00, 0x00, 0xFF]) * 4,
                           px_format=7, tex_format=14)
    data = prefix + gbix + real
    w, h, rgba = P.decode_pvr(data)
    assert (w, h) == (2, 2)


# ---------------------------------------------------------------------------
# 5. Optional: byte-exact differential vs the RUNNABLE pvr2image oracle.
# ---------------------------------------------------------------------------
_ORACLE_PATH = "C:/tmp_pso_editor/_reference/pvr2image/pvr2image.py"


def _load_oracle():
    if not os.path.isfile(_ORACLE_PATH):
        return None
    spec = importlib.util.spec_from_file_location("pvr2image_oracle", _ORACLE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not os.path.isfile(_ORACLE_PATH), reason="pvr2image oracle not present")
def test_detwiddle_matches_runnable_oracle():
    mod = _load_oracle()
    o = mod.decode.__new__(mod.decode)
    o.flip = ''
    for (w, h) in [(4, 4), (8, 8), (16, 16), (32, 32), (8, 4), (4, 8), (16, 8), (256, 256)]:
        assert P._detwiddle(w, h) == o.detwiddle(w, h), f"detwiddle {w}x{h}"


@pytest.mark.skipif(not os.path.isfile(_ORACLE_PATH), reason="pvr2image oracle not present")
@pytest.mark.parametrize("px", [0, 1, 2, 5])
def test_read_col_matches_runnable_oracle(px):
    mod = _load_oracle()
    o = mod.decode.__new__(mod.decode)
    o.flip = ''
    for c in range(0, 0x10000, 17):  # sample the 16-bit space
        a = P._read_col(px, c)
        b = o.read_col(px, c)
        assert tuple(a[:len(b)]) == tuple(b), f"px={px} c={c}"
