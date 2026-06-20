"""Tests for ``formats.bml`` packer.

Coverage:
  - Synthetic round-trip on a small fresh archive.
  - Edge cases: single-entry, no-texture, mixed compression flags, empty
    list rejection, name overflow.
  - Live byte-exact round-trip on every shipped PSOBB BML when the
    install is present (skipped otherwise).
  - CLI: unpack + re-pack preserves decompressed content (size differs
    because re-encoding via ``prs.compress`` is not Sega-bit-identical;
    the parsed-payload round-trip path IS bit-identical, covered above).
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from formats import prs as _prs
from formats.bml import (
    BML_MAGIC,
    BmlPackEntry,
    COMPRESSION_NONE,
    COMPRESSION_PRS,
    DATA_ALIGNMENT_HAS_TEX,
    DATA_ALIGNMENT_NO_TEX,
    extract_bml,
    extract_bml_texture,
    pack_bml,
    parse_bml,
    parse_bml_for_pack,
    parse_bml_pack_meta,
)

PSOBB_DATA = Path(os.path.expanduser("~/PSOBB.IO/data"))
HAS_PSOBB = PSOBB_DATA.is_dir()


# ---------------------------------------------------------------------------
# Synthetic edge cases
# ---------------------------------------------------------------------------
def test_pack_rejects_empty():
    with pytest.raises(ValueError, match="at least one entry"):
        pack_bml([])


def test_pack_rejects_bad_compression():
    with pytest.raises(ValueError, match="compression"):
        pack_bml(
            [BmlPackEntry(name="a.nj", data=b"x")],
            compression=0x99,
        )


def test_pack_rejects_long_name():
    with pytest.raises(ValueError, match="exceeds 32 bytes"):
        pack_bml([BmlPackEntry(name="x" * 33, data=b"data")])


def test_pack_rejects_compressed_without_size():
    with pytest.raises(ValueError, match="positive decompressed_size"):
        pack_bml([
            BmlPackEntry(
                name="a.nj",
                data=b"\x00\x00\x00\x00",
                is_compressed=True,
                decompressed_size=0,
            )
        ])


def test_pack_rejects_compressed_with_no_compression():
    with pytest.raises(ValueError, match="container compression is not PRS"):
        pack_bml(
            [
                BmlPackEntry(
                    name="a.nj",
                    data=_prs.compress(b"x" * 100),
                    is_compressed=True,
                    decompressed_size=100,
                )
            ],
            compression=COMPRESSION_NONE,
        )


def test_pack_single_entry_no_texture():
    """Single uncompressed-input entry; verify header + table + data."""
    payload = b"X" * 1234
    entries = [BmlPackEntry(name="solo.nj", data=payload)]
    out = pack_bml(entries)
    # Header. The 4 bytes at +0x08 are the u32 magic 0x150 (BMLUtil.cs:206),
    # NOT a (compression, has_textures) pair. For a PRS container the magic
    # is always 0x150 regardless of texture presence (low byte 0x50='P',
    # high byte the constant 0x01), matching every shipped BML.
    assert struct.unpack_from("<I", out, 4)[0] == 1
    assert struct.unpack_from("<I", out, 8)[0] == BML_MAGIC
    assert out[8] == COMPRESSION_PRS  # low byte of the magic
    # Round-trip
    parsed = parse_bml(out)
    assert len(parsed) == 1
    assert parsed[0].name == "solo.nj"
    assert parsed[0].size_decompressed == len(payload)
    assert not parsed[0].has_texture
    # Decompressed content matches.
    extracted = extract_bml(out)
    assert extracted["solo.nj"] == payload


def test_pack_with_texture():
    """Entry with texture; alignment classifies as 0x20."""
    inner = b"INNER_CONTENT" * 50
    tex = b"XVMHFAKE" + b"\xab" * 200
    entries = [BmlPackEntry(name="t.nj", data=inner, texture_data=tex)]
    out = pack_bml(entries)
    # has_textures bit set; alignment is 0x20.
    assert out[9] == 1
    parsed = parse_bml(out)
    assert parsed[0].has_texture
    extracted = extract_bml(out)
    assert extracted["t.nj"] == inner
    extracted_tex = extract_bml_texture(out, "t.nj")
    assert extracted_tex == tex


def test_pack_uncompressed_container():
    """compression=NONE; data stored raw; decompressed_size == compressed_size."""
    payload = b"raw payload bytes!" * 100
    entries = [BmlPackEntry(name="raw.nj", data=payload)]
    out = pack_bml(entries, compression=COMPRESSION_NONE)
    assert out[8] == COMPRESSION_NONE
    parsed = parse_bml(out)
    assert parsed[0].size_compressed == parsed[0].size_decompressed == len(payload)
    # Slice the inner directly — no PRS wrapping.
    inner_off = parsed[0].offset
    assert out[inner_off:inner_off + len(payload)] == payload


def test_pack_multi_entry_alignment_layout():
    """Three entries, mixed sizes; verify offsets are properly aligned."""
    entries = [
        BmlPackEntry(name="a.nj", data=b"a" * 100, texture_data=b"t1" * 50),
        BmlPackEntry(name="b.nj", data=b"b" * 200, texture_data=b"t2" * 100),
        BmlPackEntry(name="c.nj", data=b"c" * 300),
    ]
    out = pack_bml(entries)
    # Alignment is 0x20 because at least one entry has a texture.
    parsed = parse_bml(out)
    assert len(parsed) == 3
    # Each inner offset must be 0x20-aligned.
    for ent in parsed:
        assert ent.offset % DATA_ALIGNMENT_HAS_TEX == 0, (
            f"entry {ent.name} offset 0x{ent.offset:x} not 0x20-aligned"
        )


def test_pack_alignment_override():
    """Force 0x800 alignment on a tex-bearing archive (rare but valid)."""
    entries = [BmlPackEntry(name="x.nj", data=b"data", texture_data=b"tex")]
    out = pack_bml(entries, file_alignment=DATA_ALIGNMENT_NO_TEX)
    parsed = parse_bml(out)
    # Reader's heuristic should still recover the layout.
    assert parsed[0].name == "x.nj"
    assert parsed[0].has_texture


def test_pack_roundtrip_synthetic_compressed():
    """Pre-compressed input round-trips: pack -> parse -> pack identical."""
    raw = b"PSO ROUNDTRIP " * 200
    pre = _prs.compress(raw)
    entries = [
        BmlPackEntry(
            name="prebuilt.nj",
            data=pre,
            is_compressed=True,
            decompressed_size=len(raw),
        )
    ]
    out_a = pack_bml(entries)
    # Parse back, repack.
    pack_entries = parse_bml_for_pack(out_a)
    meta = parse_bml_pack_meta(out_a)
    out_b = pack_bml(
        pack_entries,
        compression=meta["compression"],
        file_alignment=meta["file_alignment"],
        has_textures_override=meta["has_textures"],
    )
    assert out_a == out_b


def test_pack_preserves_unk_a():
    """The non-zero unk_a fields in shipped BMLs round-trip exactly."""
    entries = [
        BmlPackEntry(name="a.nj", data=b"x" * 50, unk_a=0),
        BmlPackEntry(name="b.nj", data=b"y" * 50, unk_a=42),
        BmlPackEntry(name="c.nj", data=b"z" * 50, unk_a=0xdeadbeef),
    ]
    out = pack_bml(entries)
    # Read back unk_a from the table.
    for i, expected in enumerate([0, 42, 0xdeadbeef]):
        ent_off = 0x40 + i * 0x40 + 32  # past name field
        cs, unk_a, ds, *_ = struct.unpack_from("<8I", out, ent_off)
        assert unk_a == expected, (
            f"entry {i} unk_a mismatch: expected {expected}, got {unk_a}"
        )


# ---------------------------------------------------------------------------
# Live shipped fixtures - skipped if PSOBB.IO isn't installed
# ---------------------------------------------------------------------------
def _shipped_bml_files() -> list[Path]:
    if not HAS_PSOBB:
        return []
    return sorted(PSOBB_DATA.glob("*.bml"))


# 10+ representative BMLs covering the shape spectrum the spec requested:
# player body, monster, boss, effect, prop, NPC motion, lying-flag set.
_REPRESENTATIVE_NAMES = [
    "biri_ball.bml",                    # small generic
    "bm4_ps_ma_body.bml",                # player body (lying-flag, 0x800 align)
    "bm_boss1_dragon.bml",               # boss
    "bm_boss3_volopt.bml",               # 42-entry largest
    "bm_eff_ice.bml",                    # effect (no textures)
    "NpcApcMot.bml",                     # NPC motion (large, 0x20 align)
    "bm_ene_astark.bml",                 # monster
    "bm_ene_balclaw.bml",                # monster
    "bm_ene_biter_body.bml",             # monster body
    "bm7_s_paa_body.bml",                # player body variant
]


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO/data not available")
@pytest.mark.parametrize("bml_name", _REPRESENTATIVE_NAMES)
def test_representative_byte_exact_roundtrip(bml_name: str):
    """The 10 representative BMLs round-trip parse->pack byte-exact."""
    path = PSOBB_DATA / bml_name
    if not path.exists():
        pytest.skip(f"{bml_name} not present in install")
    buf = path.read_bytes()
    meta = parse_bml_pack_meta(buf)
    pack_entries = parse_bml_for_pack(buf)
    rebuilt = pack_bml(
        pack_entries,
        compression=meta["compression"],
        file_alignment=meta["file_alignment"],
        has_textures_override=meta["has_textures"],
    )
    assert rebuilt == buf, (
        f"BML round-trip mismatch for {bml_name}: "
        f"orig={len(buf)} rebuilt={len(rebuilt)}"
    )


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO/data not available")
def test_all_shipped_bmls_byte_exact_roundtrip():
    """All 364 shipped BMLs must round-trip byte-exact.

    This is the regression net: any change to pack/parse that drops a
    byte will fail on at least one of the 23 lying-flag player NJ
    archives (0x800 alignment with has_textures=1) or one of the 121
    BMLs with non-zero ``unk_a`` fields.
    """
    files = _shipped_bml_files()
    assert len(files) >= 100, f"unexpectedly few BMLs: {len(files)}"
    failures = []
    for path in files:
        buf = path.read_bytes()
        try:
            meta = parse_bml_pack_meta(buf)
            pack_entries = parse_bml_for_pack(buf)
            rebuilt = pack_bml(
                pack_entries,
                compression=meta["compression"],
                file_alignment=meta["file_alignment"],
                has_textures_override=meta["has_textures"],
            )
            if rebuilt != buf:
                failures.append((path.name, len(buf), len(rebuilt)))
        except Exception as e:
            failures.append((path.name, type(e).__name__, str(e)[:120]))
    assert not failures, (
        f"{len(failures)} BMLs failed round-trip; first 5: {failures[:5]}"
    )


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO/data not available")
def test_player_body_lying_flag_preserved():
    """The 23 ``pl[A-Z]nj.bml`` archives have has_tex=1 but use 0x800 align.
    Round-tripping must preserve the lying flag exactly."""
    candidates = [p for p in PSOBB_DATA.glob("pl*nj.bml")]
    if not candidates:
        # Fall back to the representative one.
        path = PSOBB_DATA / "bm4_ps_ma_body.bml"
        if not path.exists():
            pytest.skip("no player NJ archives present")
        candidates = [path]
    for path in candidates[:3]:
        buf = path.read_bytes()
        meta = parse_bml_pack_meta(buf)
        # The signature: alignment is 0x800 even though has_textures is True.
        if meta["file_alignment"] != DATA_ALIGNMENT_NO_TEX:
            continue
        if not meta["has_textures"]:
            continue
        # Round-trip preserves both.
        entries = parse_bml_for_pack(buf)
        rebuilt = pack_bml(
            entries,
            compression=meta["compression"],
            file_alignment=meta["file_alignment"],
            has_textures_override=meta["has_textures"],
        )
        assert rebuilt == buf, f"lying-flag preservation failed on {path.name}"


# ---------------------------------------------------------------------------
# Header magic + alignment classification on fresh packs
# ---------------------------------------------------------------------------
def test_fresh_pack_writes_constant_magic_and_correct_alignment():
    """A fresh PRS pack always writes the 0x150 magic (the +0x09 byte is the
    constant magic high byte, NOT a real has_textures flag — BMLUtil.cs:206).
    The REAL texture/alignment signal is the cumulative-end-classified
    ``file_alignment``, which still tracks actual texture presence."""
    # No-texture case: magic is 0x150 (not the old malformed 0x50), and the
    # alignment correctly classifies as 0x800.
    entries = [BmlPackEntry(name="x.nj", data=b"abc")]
    out = pack_bml(entries)
    assert struct.unpack_from("<I", out, 8)[0] == BML_MAGIC
    meta = parse_bml_pack_meta(out)
    assert meta["file_alignment"] == DATA_ALIGNMENT_NO_TEX

    # With-texture case: same magic, alignment classifies as 0x20.
    entries = [BmlPackEntry(name="x.nj", data=b"abc", texture_data=b"tex")]
    out = pack_bml(entries)
    assert struct.unpack_from("<I", out, 8)[0] == BML_MAGIC
    meta = parse_bml_pack_meta(out)
    assert meta["has_textures"] is True
    assert meta["file_alignment"] == DATA_ALIGNMENT_HAS_TEX


# ---------------------------------------------------------------------------
# CLI: unpack + repack flow
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO/data not available")
def test_cli_unpack_pack_roundtrip(tmp_path: Path):
    """``unpack`` + ``pack`` round-trip preserves DECOMPRESSED content
    even if exact bytes differ (re-encoding through prs.compress is
    semantic-equivalent but not bit-identical to Sega's encoder)."""
    src = PSOBB_DATA / "biri_ball.bml"
    if not src.exists():
        pytest.skip("biri_ball.bml not present")
    out_dir = tmp_path / "biri_ball"
    out_bml = tmp_path / "biri_ball_repack.bml"

    # Unpack.
    r = subprocess.run(
        [sys.executable, "-m", "formats.bml", "unpack", str(src), str(out_dir)],
        cwd=str(Path(__file__).parent.parent),
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, f"unpack failed: {r.stderr}"
    assert (out_dir / "_bml_manifest.json").exists()
    manifest = json.loads((out_dir / "_bml_manifest.json").read_text())
    assert len(manifest["entries"]) == 5

    # Pack.
    r = subprocess.run(
        [sys.executable, "-m", "formats.bml", "pack", str(out_dir), str(out_bml)],
        cwd=str(Path(__file__).parent.parent),
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, f"pack failed: {r.stderr}"
    assert out_bml.exists()

    # Decompressed content must match the original.
    orig = extract_bml(src.read_bytes())
    new = extract_bml(out_bml.read_bytes())
    assert set(orig) == set(new)
    for k in orig:
        assert orig[k] == new[k], f"content mismatch on {k}"
    # Texture content must also match.
    for ent in parse_bml(src.read_bytes()):
        if ent.has_texture:
            tex_orig = extract_bml_texture(src.read_bytes(), ent.name)
            tex_new = extract_bml_texture(out_bml.read_bytes(), ent.name)
            assert tex_orig == tex_new, f"texture mismatch on {ent.name}"
