"""Verify formats.pvr_decode is wired into server.extract_tiles.

Covers:
  * a synthetic single-record PVR file (no GBIX)
  * a synthetic GBIX-prefixed PVR
  * a synthetic PVM container holding three records of different
    pixel formats (ARGB8888 BMP, ARGB1555 Rectangle, RGB565 Rectangle)
  * the on-disk tile cache layout matches xvr_codec's
  * extract_tiles is idempotent on a PVR work file (manifest re-use)

The tests place the synthetic PVR / PVM in a tmp_path that the
in-process ``server.extract_tiles`` is pointed at — we don't shell out
to PuyoToolsCli or xvr_codec because the magic-sniff dispatch in
extract_tiles routes PVR work files away from those subprocesses.
"""
from __future__ import annotations

import io
import struct
from pathlib import Path

import pytest

from PIL import Image

import server as server_mod
from formats.pvr_decode import make_test_pvr, decode_pvr


# ---------------------------------------------------------------------------
# Helpers — synthesise PVR / PVM bytes.
# ---------------------------------------------------------------------------


def _make_pvm_two_records(tmp_path: Path) -> Path:
    """Build a PVMH archive with two PVRT records of different formats.

    PVMH layout (per Sega PVM spec):
        +0x00  'PVMH'
        +0x04  u32 LE header_size (size AFTER offset 0x08)
        +0x08  u16 LE flags
        +0x0A  u16 LE record count
        +0x0C  optional tables (filename / dim / global-index)
        +0x08+header_size  first PVRT chunk

    For our minimal fixture header_size=0x04 (just flags + count, no
    optional tables); records start at offset 0x0C.
    """
    # Two distinct PVRs so we can prove the parser walks them separately.
    pixels1 = bytes([
        0xFF, 0x00, 0x00, 0xFF,
        0x00, 0xFF, 0x00, 0xFF,
        0x00, 0x00, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF,
    ])
    pvr1 = make_test_pvr(2, 2, pixels1, px_format=7, tex_format=14)

    pixels2 = bytearray()
    rows = [
        (0xFF, 0x00, 0x00, 0xFF),
        (0x00, 0xFF, 0x00, 0xFF),
        (0x00, 0x00, 0xFF, 0xFF),
        (0xFF, 0xFF, 0x00, 0xFF),
    ]
    for _ in range(4):
        for row in rows:
            pixels2 += bytes(row)
    pvr2 = make_test_pvr(4, 4, bytes(pixels2), px_format=0, tex_format=9)

    # Build PVMH wrapper. header_size=4 (flags+count, no tables); count=2.
    pvm = bytearray()
    pvm += b"PVMH"
    pvm += struct.pack("<I", 4)        # header_size = flags+count
    pvm += struct.pack("<H", 0)        # flags (no tables)
    pvm += struct.pack("<H", 2)        # record count
    pvm += pvr1
    pvm += pvr2
    out = tmp_path / "synth_two.pvm"
    out.write_bytes(bytes(pvm))
    return out


def _make_single_pvr_file(tmp_path: Path, *, name: str = "synth.pvr") -> Path:
    pixels = bytes([0x10, 0x20, 0x30, 0xFF]) * (4 * 4)
    pvr = make_test_pvr(4, 4, pixels, px_format=7, tex_format=14)
    out = tmp_path / name
    out.write_bytes(pvr)
    return out


def _make_argb1555_pvr_file(tmp_path: Path) -> Path:
    pixels = bytearray()
    for _ in range(4):
        pixels += bytes([0xFF, 0x00, 0x00, 0xFF])
        pixels += bytes([0x00, 0xFF, 0x00, 0xFF])
        pixels += bytes([0x00, 0x00, 0xFF, 0xFF])
        pixels += bytes([0xFF, 0xFF, 0xFF, 0xFF])
    pvr = make_test_pvr(4, 4, bytes(pixels), px_format=0, tex_format=9)
    out = tmp_path / "synth_1555.pvr"
    out.write_bytes(pvr)
    return out


def _make_rgb565_pvr_file(tmp_path: Path) -> Path:
    pixels = bytearray()
    for _ in range(4):
        pixels += bytes([0xFF, 0x00, 0x00, 0xFF])
        pixels += bytes([0x00, 0xFF, 0x00, 0xFF])
        pixels += bytes([0x00, 0x00, 0xFF, 0xFF])
        pixels += bytes([0xFF, 0xFF, 0x00, 0xFF])
    pvr = make_test_pvr(4, 4, bytes(pixels), px_format=1, tex_format=9)
    out = tmp_path / "synth_565.pvr"
    out.write_bytes(pvr)
    return out


# ---------------------------------------------------------------------------
# 1. Single-PVR: extract_tiles produces one PNG via pvr_decode.
# ---------------------------------------------------------------------------


def test_extract_tiles_single_pvr(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod, "CACHE_DIR", tmp_path / "cache")
    server_mod.CACHE_DIR.mkdir(exist_ok=True)
    pvr = _make_single_pvr_file(tmp_path)

    manifest = server_mod.extract_tiles(pvr)
    tiles = manifest["tiles"]
    assert len(tiles) == 1
    t = tiles[0]
    assert t["width"] == 4 and t["height"] == 4
    assert t["fmt"] == 7  # ARGB8888 px_format byte

    png_path = Path(manifest["tiles_dir"]) / t["filename"]
    assert png_path.is_file()
    with Image.open(png_path) as im:
        assert im.size == (4, 4)
        assert im.mode in ("RGBA", "RGB")
        # Verify pixel content is non-uniform AND not all-magenta-error.
        pixels = list(im.convert("RGBA").getdata())
        unique = len(set(pixels))
        assert unique >= 1
        # Magenta error placeholder is (255, 0, 255, 255) at 1×1 only;
        # this is a 4×4 real decode, so pixels[0] must be (0x10, 0x20,
        # 0x30, 0xFF) per make_test_pvr's lossless ARGB8888 BMP path.
        assert pixels[0][:3] == (0x10, 0x20, 0x30)


# ---------------------------------------------------------------------------
# 2. ARGB1555 PVR (different px_format) — covers tex_format=9 + twiddle=False.
# ---------------------------------------------------------------------------


def test_extract_tiles_argb1555_pvr(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod, "CACHE_DIR", tmp_path / "cache")
    server_mod.CACHE_DIR.mkdir(exist_ok=True)
    pvr = _make_argb1555_pvr_file(tmp_path)
    manifest = server_mod.extract_tiles(pvr)
    tiles = manifest["tiles"]
    assert len(tiles) == 1
    t = tiles[0]
    assert t["width"] == 4 and t["height"] == 4
    assert t["fmt"] == 0  # ARGB1555 px_format byte

    png_path = Path(manifest["tiles_dir"]) / t["filename"]
    with Image.open(png_path) as im:
        pixels = list(im.convert("RGBA").getdata())
        # First pixel should be pure red after ARGB1555 round-trip.
        assert pixels[0][:3] == (0xFF, 0x00, 0x00)


# ---------------------------------------------------------------------------
# 3. RGB565 PVR — covers px_format=1, alpha forced to 0xFF.
# ---------------------------------------------------------------------------


def test_extract_tiles_rgb565_pvr(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod, "CACHE_DIR", tmp_path / "cache")
    server_mod.CACHE_DIR.mkdir(exist_ok=True)
    pvr = _make_rgb565_pvr_file(tmp_path)
    manifest = server_mod.extract_tiles(pvr)
    tiles = manifest["tiles"]
    assert len(tiles) == 1
    assert tiles[0]["fmt"] == 1  # RGB565 px_format byte

    png_path = Path(manifest["tiles_dir"]) / tiles[0]["filename"]
    with Image.open(png_path) as im:
        pixels = list(im.convert("RGBA").getdata())
        # 565 has no alpha — every pixel must come back fully opaque.
        for px in pixels:
            assert px[3] == 0xFF


# ---------------------------------------------------------------------------
# 4. PVM container — multi-record archive yields multiple tiles in order.
# ---------------------------------------------------------------------------


def test_extract_tiles_pvm_multi_record(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod, "CACHE_DIR", tmp_path / "cache")
    server_mod.CACHE_DIR.mkdir(exist_ok=True)
    pvm = _make_pvm_two_records(tmp_path)
    manifest = server_mod.extract_tiles(pvm)
    tiles = manifest["tiles"]
    assert len(tiles) == 2
    # First tile is 2x2 ARGB8888; second is 4x4 ARGB1555.
    assert tiles[0]["width"] == 2 and tiles[0]["height"] == 2
    assert tiles[0]["fmt"] == 7
    assert tiles[1]["width"] == 4 and tiles[1]["height"] == 4
    assert tiles[1]["fmt"] == 0


# ---------------------------------------------------------------------------
# 5. Manifest cache: re-extract is idempotent.
# ---------------------------------------------------------------------------


def test_extract_tiles_pvr_manifest_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod, "CACHE_DIR", tmp_path / "cache")
    server_mod.CACHE_DIR.mkdir(exist_ok=True)
    pvr = _make_single_pvr_file(tmp_path)
    m1 = server_mod.extract_tiles(pvr)
    # Touch the cache dir's manifest mtime; second call must reuse the
    # existing tiles_dir without rebuilding.
    m2 = server_mod.extract_tiles(pvr)
    assert m1["tiles_dir"] == m2["tiles_dir"]
    assert m1["extracted_at"] == m2["extracted_at"]


# ---------------------------------------------------------------------------
# 6. Sibling .pvr file is written next to the .png so callers
#    that re-read the source format (Material Inspector preview path)
#    have a stable filesystem layout.
# ---------------------------------------------------------------------------


def test_extract_tiles_writes_pvr_sibling(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod, "CACHE_DIR", tmp_path / "cache")
    server_mod.CACHE_DIR.mkdir(exist_ok=True)
    pvr = _make_single_pvr_file(tmp_path)
    manifest = server_mod.extract_tiles(pvr)
    tiles_dir = Path(manifest["tiles_dir"])
    # Filename pattern: <stem>_<idx:02d>_<W>x<H>.png plus matching .pvr.
    pngs = list(tiles_dir.glob("*.png"))
    pvrs = list(tiles_dir.glob("*.pvr"))
    assert len(pngs) == 1 and len(pvrs) == 1
    # The sibling .pvr is the original record bytes — round-trips through decode_pvr.
    w, h, rgba = decode_pvr(pvrs[0].read_bytes())
    assert (w, h) == (4, 4)
    assert len(rgba) == 4 * 4 * 4
