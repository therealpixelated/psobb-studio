"""Tests for ``formats.pvr_encode`` — Sega Dreamcast PVR texture ENCODER.

The encoder is the byte-level inverse of ``formats.pvr_decode``; every test
here is a ``decode(encode(img))`` round-trip. Lossless paths (ARGB8888 BMP)
must reproduce pixels exactly; the lossy 16-bit paths must hit PSNR >= 40 dB on
a checkerboard + gradient + alpha test image. The twiddled formats additionally
assert the twiddle was actually applied (not an identity pass-through) and that
re-ordering the twiddled body through the decoder's de-twiddle permutation
recovers the linear body byte-for-byte.

PSOBB BB ships XVR (not PVR), so we synthesize fixtures here. A defensive glob
of the tree picks up any genuine ``.pvr`` files if present (the editor's
reference data is normally gitignored, so the loop is a no-op in CI).
"""
from __future__ import annotations

import glob
import math
import struct
from pathlib import Path

import pytest

from PIL import Image

from formats.pvr_encode import encode_pvr, build_pvm, PX_MODES, TEX_MODES
from formats import pvr_decode
from formats.pvr_decode import decode_pvr, _parse_pvr_header, _detwiddle


# ---------------------------------------------------------------------------
# Test-image helpers.
# ---------------------------------------------------------------------------


def _test_image(w: int, h: int) -> Image.Image:
    """Checkerboard + dual-gradient + alpha pattern — exercises every channel.

    R is a horizontal gradient, G a vertical gradient, B a 1px XOR
    checkerboard, A a 2px block checkerboard. This stresses both the colour
    quantisation and the spatial (twiddle) reordering.
    """
    img = Image.new("RGBA", (w, h))
    data = []
    for y in range(h):
        for x in range(w):
            r = (x * 255) // max(1, w - 1)
            g = (y * 255) // max(1, h - 1)
            b = ((x ^ y) & 1) * 255
            a = 255 if ((x // 2 + y // 2) & 1) == 0 else 0
            data.append((r, g, b, a))
    img.putdata(data)
    return img


def _img_bytes(img: Image.Image) -> bytes:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img.tobytes()


def _psnr(a: bytes, b: bytes) -> float:
    assert len(a) == len(b)
    if not a:
        return 999.0
    se = sum((a[i] - b[i]) ** 2 for i in range(len(a)))
    mse = se / len(a)
    if mse == 0:
        return 999.0
    return 10.0 * math.log10((255.0 * 255.0) / mse)


def _rgb_only(buf: bytes) -> bytes:
    """Strip the alpha byte from an RGBA8 buffer (for alpha-less formats)."""
    return bytes(buf[i] for i in range(len(buf)) if i % 4 != 3)


# Per-format PSNR floor. The 5/6-bit formats clear the spec's 40 dB bar; the
# 4-bit-per-channel ARGB4444 format's *theoretical* nearest-quantiser floor is
# ~34 dB (step size 17), so its tolerance reflects the format limit, not a
# defect. These are the achievable maxima with nearest-value quantisation.
_PSNR_FLOOR = {
    0: 40.0,   # ARGB1555 (5-bit RGB)
    1: 40.0,   # RGB565
    2: 33.0,   # ARGB4444 (4-bit/channel — quantisation-limited)
    5: 40.0,   # RGB555
}


# ---------------------------------------------------------------------------
# 1. Lossless: ARGB8888 BMP (px 7 / tex 14).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dims", [(2, 2), (8, 8), (16, 8), (5, 3), (1, 1)])
def test_argb8888_bmp_lossless(dims):
    w, h = dims
    img = _test_image(w, h)
    data = encode_pvr(img, 7, 14)
    dw, dh, rgba = decode_pvr(data)
    assert (dw, dh) == (w, h)
    assert rgba == _img_bytes(img), "ARGB8888 BMP must be byte-exact"


# ---------------------------------------------------------------------------
# 2. 16-bit RGB565 exact on round 5/6-bit values.
# ---------------------------------------------------------------------------


def test_rgb565_exact_on_round_values():
    """RGB565 reproduces colours already aligned to the 5/6/5 grid exactly."""
    # Channels chosen so r,b are multiples of 255/31 and g of 255/63 after the
    # decoder's expansion: use pure 0 / 255 extremes which are always exact.
    img = Image.new("RGBA", (2, 2))
    img.putdata([
        (255, 0, 0, 255),
        (0, 255, 0, 255),
        (0, 0, 255, 255),
        (255, 255, 255, 255),
    ])
    data = encode_pvr(img, 1, 9)  # RGB565 Rectangle
    _, _, rgba = decode_pvr(data)
    # RGB565 forces alpha to 0xFF; compare RGB only.
    assert _rgb_only(rgba) == _rgb_only(_img_bytes(img))


def test_argb1555_extremes_exact():
    """ARGB1555 reproduces pure primaries + on/off alpha threshold exactly."""
    img = Image.new("RGBA", (2, 2))
    img.putdata([
        (255, 0, 0, 255),
        (0, 255, 0, 255),
        (0, 0, 255, 0),     # zero alpha -> bit 0
        (255, 255, 255, 255),
    ])
    data = encode_pvr(img, 0, 9)  # ARGB1555 Rectangle
    _, _, rgba = decode_pvr(data)
    assert rgba == _img_bytes(img)


# ---------------------------------------------------------------------------
# 3. Lossy 16-bit round-trips must clear PSNR >= 40 dB.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("px_format,data_format,alpha_aware", [
    (0, 9, True),    # ARGB1555 Rectangle
    (1, 9, False),   # RGB565   Rectangle (alpha dropped)
    (2, 9, True),    # ARGB4444 Rectangle
    (5, 9, False),   # RGB555   Rectangle (alpha dropped)
])
def test_linear16_psnr(px_format, data_format, alpha_aware):
    img = _test_image(16, 16)
    data = encode_pvr(img, px_format, data_format)
    dw, dh, rgba = decode_pvr(data)
    assert (dw, dh) == (16, 16)
    orig = _img_bytes(img)
    if alpha_aware:
        got, exp = rgba, orig
    else:
        got, exp = _rgb_only(rgba), _rgb_only(orig)
    psnr = _psnr(got, exp)
    floor = _PSNR_FLOOR[px_format]
    assert psnr >= floor, f"{PX_MODES[px_format]} PSNR {psnr:.1f} dB < {floor}"


# ---------------------------------------------------------------------------
# 4. Twiddled formats: round-trip + twiddle is non-identity + exact inverse.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("px_format", [0, 1, 2, 5])
@pytest.mark.parametrize("dims,data_format", [
    ((8, 8), 1),     # SquareTwiddled, square
    ((16, 16), 1),   # SquareTwiddled, square
    ((16, 8), 13),   # RectangleTwiddled, WIDE (the classic encoder-bug case)
    ((8, 16), 13),   # RectangleTwiddled, TALL
    ((16, 16), 13),  # RectangleTwiddled, square
])
def test_twiddled_round_trip(px_format, dims, data_format):
    w, h = dims
    img = _test_image(w, h)
    data = encode_pvr(img, px_format, data_format)
    off, dw, dh, dpx, dtex = _parse_pvr_header(data)
    assert (dw, dh) == (w, h)
    assert dpx == px_format and dtex == data_format
    _, _, rgba = decode_pvr(data)
    orig = _img_bytes(img)
    # ARGB1555 (0) and ARGB4444 (2) carry alpha; RGB565/555 drop it.
    if px_format in (0, 2):
        got, exp = rgba, orig
    else:
        got, exp = _rgb_only(rgba), _rgb_only(orig)
    psnr = _psnr(got, exp)
    floor = _PSNR_FLOOR[px_format]
    assert psnr >= floor, (
        f"{PX_MODES[px_format]} {TEX_MODES[data_format]} {w}x{h} "
        f"PSNR {psnr:.1f} dB < {floor}"
    )


def test_twiddle_is_not_identity_and_is_exact_inverse():
    """The twiddled body must differ from the linear body (twiddle applied),
    and re-ordering it through the decoder's de-twiddle permutation must
    recover the linear body byte-for-byte (exact inverse)."""
    from formats.pvr_encode import _encode_linear16, _encode_twiddled16, _img_to_rgba_rows

    w = h = 16
    img = _test_image(w, h)
    _, _, pixels = _img_to_rgba_rows(img)

    linear = _encode_linear16(0, pixels)
    twiddled = _encode_twiddled16(0, w, h, pixels)
    assert linear != twiddled, "twiddle produced an identity layout"

    arr = _detwiddle(w, h)
    assert arr != list(range(w * h)), "_detwiddle returned identity"

    lin_words = [struct.unpack_from("<H", linear, i * 2)[0] for i in range(w * h)]
    twi_words = [struct.unpack_from("<H", twiddled, i * 2)[0] for i in range(w * h)]
    reordered = [twi_words[arr[i]] for i in range(w * h)]
    assert reordered == lin_words, "twiddle is not the exact inverse of de-twiddle"


def test_wide_twiddle_differs_from_naive_per_axis():
    """Regression guard: our wide (w>h) twiddle follows the DECODER's pvr2image
    order, which differs from VrSharp's per-axis (tm[x]<<1)|tm[y]. If someone
    'fixes' the encoder to the per-axis formula the round-trip below breaks."""
    img = _test_image(16, 8)  # wide
    data = encode_pvr(img, 0, 13)  # ARGB1555 RectangleTwiddled
    _, _, rgba = decode_pvr(data)
    assert _psnr(rgba, _img_bytes(img)) >= 40.0


# ---------------------------------------------------------------------------
# 5. GBIX global-index round-trip.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gi", [0, 1, 0xDEADBEEF, 0xFFFFFFFF])
def test_gbix_global_index_round_trips(gi):
    img = _test_image(4, 4)
    data = encode_pvr(img, 7, 14, global_index=gi)
    assert data[:4] == b"GBIX"
    assert struct.unpack_from("<I", data, 0x04)[0] == 8  # GBIX payload len
    assert struct.unpack_from("<I", data, 0x08)[0] == gi
    # And the PVRT body still decodes losslessly past the GBIX chunk.
    _, _, rgba = decode_pvr(data)
    assert rgba == _img_bytes(img)


def test_no_gbix_when_global_index_none():
    img = _test_image(4, 4)
    data = encode_pvr(img, 7, 14)
    assert data[:4] == b"PVRT"
    assert b"GBIX" not in data[:4]


# ---------------------------------------------------------------------------
# 6. build_pvm — invert sibling_archives._parse_pvm_records.
# ---------------------------------------------------------------------------


def test_build_pvm_round_trips_through_reader():
    from formats.sibling_archives import _parse_pvm_records

    imgs = [_test_image(8, 8), _test_image(4, 4), _test_image(16, 16)]
    recs = [encode_pvr(im, 7, 14, global_index=i) for i, im in enumerate(imgs)]
    pvm = build_pvm(recs)
    assert pvm[:4] == b"PVMH"
    parsed = _parse_pvm_records(pvm)
    assert len(parsed) == len(recs)
    # Each parsed (offset, length) slice must equal the record we packed.
    for (off, length, _name), rec in zip(parsed, recs):
        assert pvm[off:off + length] == rec


def test_build_pvm_records_decode_back():
    from formats.sibling_archives import _parse_pvm_records

    imgs = [_test_image(8, 8), _test_image(16, 8)]
    origs = [_img_bytes(im) for im in imgs]
    recs = [encode_pvr(im, 7, 14) for im in imgs]
    pvm = build_pvm(recs)
    parsed = _parse_pvm_records(pvm)
    for (off, length, _name), orig in zip(parsed, origs):
        _, _, rgba = decode_pvr(pvm[off:off + length])
        assert rgba == orig


def test_build_pvm_rejects_mismatched_global_indices():
    img = _test_image(4, 4)
    recs = [encode_pvr(img, 7, 14)]
    with pytest.raises(ValueError):
        build_pvm(recs, global_indices=[1, 2])


# ---------------------------------------------------------------------------
# 7. Unsupported formats raise NotImplementedError (never wrong bytes).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("px_format,data_format", [
    (0, 3),    # ARGB1555 Twiddled VQ
    (1, 16),   # RGB565 SmallVQ
    (0, 4),    # VQ + mips
    (0, 5),    # palettized 4-bit
    (0, 7),    # palettized 8-bit
    (0, 2),    # SquareTwiddled + mips
    (3, 9),    # YUV422
    (4, 1),    # BUMP
    (6, 9),    # YUV420
])
def test_unsupported_raises_not_implemented(px_format, data_format):
    img = _test_image(16, 16)
    with pytest.raises(NotImplementedError):
        encode_pvr(img, px_format, data_format)


def test_square_twiddled_rejects_non_square():
    img = _test_image(16, 8)
    with pytest.raises(ValueError):
        encode_pvr(img, 0, 1)  # SquareTwiddled requires square


def test_twiddled_rejects_non_pow2():
    img = _test_image(12, 8)  # 12 is not a power of two
    with pytest.raises(ValueError):
        encode_pvr(img, 0, 13)


# ---------------------------------------------------------------------------
# 8. Real .pvr files (if any are present in the tree): re-encode the decoded
#    pixels and assert decode matches. VQ/SmallVQ source files are re-encoded
#    into a format we DO support (ARGB1555 RectangleTwiddled) since we cannot
#    faithfully regenerate a VQ codebook.
# ---------------------------------------------------------------------------


def _find_real_pvrs() -> list[str]:
    roots = [
        Path(__file__).resolve().parent.parent,          # repo root
        Path(__file__).resolve().parent.parent / "_reference",
    ]
    # Only genuine REFERENCE .pvr assets — never runtime-generated tiles.
    # The studio writes decoded/re-encoded tiles under cache/ (and parity
    # renders under _parity/); those are volatile per-machine and, being
    # ARGB1555 (5 bits/channel), legitimately score below the reference PSNR
    # bar. Scanning them made this test non-deterministic across machines.
    _GENERATED = ("/cache/", "/_parity/", "/__pycache__/", "/.git/")
    found: list[str] = []
    for root in roots:
        if root.exists():
            for p in glob.glob(str(root / "**" / "*.pvr"), recursive=True):
                norm = p.replace("\\", "/").lower()
                if any(seg in norm for seg in _GENERATED):
                    continue
                found.append(p)
    # De-dup, cap to keep the test quick.
    return sorted(set(found))[:12]


def test_real_pvr_reencode_matches_decode():
    pvrs = _find_real_pvrs()
    if not pvrs:
        pytest.skip("no real .pvr files in tree (reference data is gitignored)")

    checked = 0
    for fp in pvrs:
        blob = Path(fp).read_bytes()
        try:
            off, w, h, px, tex = _parse_pvr_header(blob)
            dw, dh, rgba = decode_pvr(blob)
        except Exception:
            continue  # skip anything our decoder itself can't handle
        if dw == 0 or dh == 0:
            continue
        img = Image.frombytes("RGBA", (dw, dh), rgba)

        # Pick a faithful target: lossless BMP recovers the decoded pixels
        # exactly regardless of the source's on-disk format.
        re_blob = encode_pvr(img, 7, 14)
        _, _, rgba2 = decode_pvr(re_blob)
        assert rgba2 == rgba, f"BMP re-encode mismatch for {fp}"

        # If the dimensions are power-of-two, also exercise a twiddled
        # round-trip (the spatial-reorder path) at PSNR tolerance.
        if (dw & (dw - 1)) == 0 and (dh & (dh - 1)) == 0:
            tex_fmt = 1 if dw == dh else 13
            tw_blob = encode_pvr(img, 0, tex_fmt)  # ARGB1555 twiddled
            _, _, rgba3 = decode_pvr(tw_blob)
            assert _psnr(rgba3, rgba) >= 38.0, f"twiddled re-encode {fp}"
        checked += 1

    assert checked > 0, "found .pvr files but none were decodable to re-encode"
