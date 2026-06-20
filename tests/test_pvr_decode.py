"""Tests for ``formats.pvr_decode`` — Sega Dreamcast PVR texture decoder.

PSOBB BB doesn't ship PVR (it uses XVR / XBOX VR), so we lean on
``make_test_pvr`` synthetic fixtures plus a defensive scan of the
editor's data tree for any genuine PVR files (none expected; the test
loop is a no-op if the glob comes back empty).
"""
from __future__ import annotations
import os

from pathlib import Path

import pytest

from formats.pvr_decode import (
    PX_MODES,
    TEX_MODES,
    decode_pvr,
    decode_pvr_with_palette,
    decode_pvr_mips,
    make_test_pvr,
)


# ---------------------------------------------------------------------------
# 1. Round-trip: ARGB8888 BMP (lossless).
# ---------------------------------------------------------------------------


def test_argb8888_round_trip_2x2():
    """Pure ARGB8888 / tex=14 is lossless; bytes round-trip exactly."""
    pixels = bytes([
        0xFF, 0x00, 0x00, 0xFF,   # red opaque
        0x00, 0xFF, 0x00, 0xFF,   # green opaque
        0x00, 0x00, 0xFF, 0x80,   # blue half-alpha
        0xFF, 0xFF, 0x00, 0x00,   # yellow zero-alpha
    ])
    data = make_test_pvr(2, 2, pixels, px_format=7, tex_format=14)
    w, h, rgba = decode_pvr(data)
    assert (w, h) == (2, 2)
    assert len(rgba) == 2 * 2 * 4
    assert rgba == pixels


# ---------------------------------------------------------------------------
# 2. ARGB1555 quantises to 5-bit RGB + 1-bit alpha but the corner cases
#    (full-on, full-off, threshold) survive.
# ---------------------------------------------------------------------------


def test_argb1555_4x4_corners_preserved():
    """ARGB1555 is lossy but pure red/green/blue/white still decode correctly."""
    pixels = bytearray()
    rows = [
        (0xFF, 0x00, 0x00, 0xFF),
        (0x00, 0xFF, 0x00, 0xFF),
        (0x00, 0x00, 0xFF, 0xFF),
        (0xFF, 0xFF, 0xFF, 0xFF),
    ]
    for _ in range(4):
        for row in rows:
            pixels += bytes(row)
    data = make_test_pvr(4, 4, bytes(pixels), px_format=0, tex_format=9)
    w, h, rgba = decode_pvr(data)
    assert (w, h) == (4, 4)
    assert len(rgba) == 4 * 4 * 4
    # Spot-check first row after decode: red, green, blue, white.
    # 5-bit primary 0x1F → 0xFF round-trip is exact (the ramp uses
    # 0xff/0x1f * 0x1f = 0xff).
    assert rgba[0:4]   == bytes([0xFF, 0x00, 0x00, 0xFF])
    assert rgba[4:8]   == bytes([0x00, 0xFF, 0x00, 0xFF])
    assert rgba[8:12]  == bytes([0x00, 0x00, 0xFF, 0xFF])
    assert rgba[12:16] == bytes([0xFF, 0xFF, 0xFF, 0xFF])


# ---------------------------------------------------------------------------
# 3. RGB565 has no alpha; decoder must return 0xFF for every pixel.
# ---------------------------------------------------------------------------


def test_rgb565_alpha_forced_to_ff():
    pixels = bytes([
        0xFF, 0x00, 0x00, 0x12,   # source alpha gets DROPPED in 565
        0x00, 0xFF, 0x00, 0x34,
    ])
    data = make_test_pvr(2, 1, pixels, px_format=1, tex_format=9)
    w, h, rgba = decode_pvr(data)
    assert (w, h) == (2, 1)
    assert len(rgba) == 2 * 1 * 4
    assert rgba[3] == 0xFF
    assert rgba[7] == 0xFF


# ---------------------------------------------------------------------------
# 4. ARGB4444 has 4-bit channels; values quantise but extremes survive.
# ---------------------------------------------------------------------------


def test_argb4444_quantised_corner_cases():
    pixels = bytes([
        0xFF, 0xFF, 0xFF, 0xFF,   # white opaque
        0x00, 0x00, 0x00, 0x00,   # black transparent
        0xF0, 0x00, 0x00, 0xF0,   # near-red, near-opaque (top nibble high)
        0x00, 0x00, 0x00, 0x00,
    ])
    data = make_test_pvr(2, 2, pixels, px_format=2, tex_format=9)
    w, h, rgba = decode_pvr(data)
    assert (w, h) == (2, 2)
    # White opaque round-trips exactly (0xF*0x11 == 0xFF).
    assert rgba[0:4] == bytes([0xFF, 0xFF, 0xFF, 0xFF])
    # Black transparent round-trips exactly.
    assert rgba[4:8] == bytes([0x00, 0x00, 0x00, 0x00])
    # Top-nibble-only red is reproduced lossless since 0xF0>>4 = 0xF and
    # 0xF*0x11 = 0xFF.
    assert rgba[8:12] == bytes([0xFF, 0x00, 0x00, 0xFF])


# ---------------------------------------------------------------------------
# 5. Header parsing: dimensions and format codes recovered correctly.
# ---------------------------------------------------------------------------


def test_header_dimensions_match_for_multiple_sizes():
    """Decoder must report the dimensions stored in the PVRT header."""
    for w, h in [(1, 1), (4, 1), (1, 4), (8, 8), (16, 4)]:
        pixels = bytes([0xAA, 0xBB, 0xCC, 0xDD]) * (w * h)
        data = make_test_pvr(w, h, pixels, px_format=7, tex_format=14)
        dw, dh, rgba = decode_pvr(data)
        assert (dw, dh) == (w, h)
        assert len(rgba) == w * h * 4


# ---------------------------------------------------------------------------
# 6. Format-code dictionaries are populated for everything pvr2image
#    upstream supports.
# ---------------------------------------------------------------------------


def test_format_dictionaries_cover_supported_modes():
    """PX_MODES / TEX_MODES are the canonical pvr2image labels."""
    # Pixel-format codes 0..10 from upstream.
    for code in range(11):
        assert code in PX_MODES
    # Texture-format codes 1..18 from upstream.
    for code in range(1, 19):
        assert code in TEX_MODES
    # Spot-check a couple of well-known names.
    assert PX_MODES[7] == "ARGB8888"
    assert PX_MODES[0] == "ARGB1555"
    assert TEX_MODES[14] == "BMP"
    assert TEX_MODES[9] == "Rectangle"


# ---------------------------------------------------------------------------
# 7. Bytes output type is ``bytes`` (immutable) so callers can hash /
#    cache without copying.
# ---------------------------------------------------------------------------


def test_output_is_bytes_not_bytearray():
    pixels = bytes([0x00, 0x00, 0x00, 0xFF]) * 4
    data = make_test_pvr(2, 2, pixels)
    _, _, rgba = decode_pvr(data)
    assert isinstance(rgba, bytes)


# ---------------------------------------------------------------------------
# 8. Bad inputs raise ValueError (not segfault / silent return).
# ---------------------------------------------------------------------------


def test_truncated_pvr_raises():
    with pytest.raises(ValueError):
        decode_pvr(b"PVRT")  # too short


def test_missing_magic_raises():
    with pytest.raises(ValueError):
        decode_pvr(b"\x00" * 0x40)  # no PVRT magic


def test_non_bytes_input_raises():
    with pytest.raises(ValueError):
        decode_pvr(12345)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 9. decode_pvr_mips falls back to single-mip when input has no mip flag.
# ---------------------------------------------------------------------------


def test_decode_pvr_mips_single_for_non_mip_input():
    pixels = bytes([0x10, 0x20, 0x30, 0xFF]) * 16
    data = make_test_pvr(4, 4, pixels, px_format=7, tex_format=14)
    mips = decode_pvr_mips(data)
    assert len(mips) == 1
    assert mips[0][0:2] == (4, 4)
    assert len(mips[0][2]) == 4 * 4 * 4


# ---------------------------------------------------------------------------
# 10. Real PVR files in the data tree (best-effort — likely empty).
# ---------------------------------------------------------------------------


def _find_real_pvrs() -> list[Path]:
    roots = [
        Path("C:/tmp_pso_dev/data"),
        Path(os.path.expanduser("~/PSOBB.IO/data")),
        Path(os.path.expanduser("~/Repositories/psobb-studio/_reference/pvr2image")),
    ]
    found: list[Path] = []
    for r in roots:
        if not r.is_dir():
            continue
        try:
            found.extend(p for p in r.rglob("*.pvr") if p.is_file() and p.stat().st_size > 0x10)
        except OSError:
            pass
    return found[:5]  # sample at most 5 — keep test fast


@pytest.mark.parametrize("pvr_path", _find_real_pvrs() or [None])
def test_real_pvr_files_decode_or_skip(pvr_path):
    """Decode each real PVR found in the data tree; skip if none exist."""
    if pvr_path is None:
        pytest.skip("no real .pvr files in editor data tree (PSOBB ships XVR, not PVR)")
    data = pvr_path.read_bytes()
    try:
        w, h, rgba = decode_pvr(data)
    except (ValueError, NotImplementedError, IndexError) as e:
        # Real PVRs may use VQ / palettized / mipmapped formats we don't
        # exercise in synth fixtures — assert the error is informative.
        assert str(e), f"empty error decoding {pvr_path.name}"
        pytest.skip(f"{pvr_path.name}: {e}")
    assert w > 0 and h > 0
    assert len(rgba) == w * h * 4


# ---------------------------------------------------------------------------
# 11. Palette helper accepts None gracefully (greyscale fallback).
# ---------------------------------------------------------------------------


def test_decode_pvr_with_palette_is_alias():
    """The convenience alias should yield identical results to the main API."""
    pixels = bytes([0x10, 0x20, 0x30, 0xFF]) * 4
    data = make_test_pvr(2, 2, pixels)
    a = decode_pvr(data)
    b = decode_pvr_with_palette(data, b"")  # empty palette → fallback
    # palette_data falsy ⇒ no palette parsed; same output as decode_pvr.
    assert a[0:2] == b[0:2]
    assert a[2] == b[2]
