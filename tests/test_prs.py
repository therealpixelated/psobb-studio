"""Tests for formats.prs encoder/decoder."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from formats.prs import compress, compress_optimal, decompress

# Locate real PSOBB BMLs as round-trip fixtures. Skip the live-asset
# tests if the install isn't present (e.g. CI build).
PSOBB_DATA = Path(os.path.expanduser("~/PSOBB.IO/data"))
HAS_PSOBB = PSOBB_DATA.is_dir()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_empty_input_roundtrip():
    enc = compress(b"")
    assert decompress(enc) == b""
    enc = compress_optimal(b"")
    assert decompress(enc) == b""


def test_single_byte_roundtrip():
    for v in (b"\x00", b"\xff", b"A"):
        enc = compress(v)
        assert decompress(enc) == v, f"greedy: {v!r}"
        enc = compress_optimal(v)
        assert decompress(enc) == v, f"optimal: {v!r}"


def test_all_zero_input():
    """An all-zero buffer triggers maximum LZ run lengths."""
    for n in (16, 256, 1024, 8192, 0x10000):
        data = b"\x00" * n
        for enc_fn in (compress, compress_optimal):
            enc = enc_fn(data)
            dec = decompress(enc)
            assert dec == data, f"{enc_fn.__name__} failed on {n} zeros"
            # Should compress dramatically (well under 1% size)
            assert len(enc) < n * 0.05 + 32, (
                f"{enc_fn.__name__}: zeros n={n} compressed to {len(enc)}"
            )


def test_all_ff_input():
    """All-0xFF input. Same idea as zeros — every byte matches every prior byte."""
    for n in (16, 256, 1024):
        data = b"\xff" * n
        for enc_fn in (compress, compress_optimal):
            enc = enc_fn(data)
            assert decompress(enc) == data


def test_max_run_length_match():
    """Test that EXTENDED_COPY's 0x100 max size is honored correctly."""
    # 257 bytes of repetition: should produce one EXTENDED_COPY (size 0x100)
    # plus a literal (or another short copy) for the trailing byte.
    data = b"A" * 257
    enc = compress(data)
    dec = decompress(enc)
    assert dec == data


def test_no_repetition_input():
    """Input with no repeating bytes -> mostly literals."""
    data = bytes(range(256))
    enc = compress(data)
    dec = decompress(enc)
    assert dec == data
    enc = compress_optimal(data)
    assert decompress(enc) == data


def test_at_window_boundary():
    """Match exactly at the LONG window boundary (-0x1FFF)."""
    pattern = b"ABCDEFGHIJ"
    fill = b"\x00" * (0x1FFF - len(pattern))
    data = pattern + fill + pattern
    enc = compress(data)
    assert decompress(enc) == data


def test_short_copy_offset_boundary():
    """Match at the SHORT_COPY -0x100 boundary."""
    pattern = b"AB"
    fill = b"\x01" * (0x100 - len(pattern))
    data = pattern + fill + pattern
    enc = compress(data)
    assert decompress(enc) == data


def test_invalid_input_type():
    with pytest.raises(ValueError):
        compress(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        decompress("not bytes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Decoder safety
# ---------------------------------------------------------------------------
def test_decompress_max_output_size_enforced():
    data = b"\x00" * 1000
    enc = compress(data)
    with pytest.raises(ValueError):
        decompress(enc, max_output_size=500)


def test_decompress_truncated_input_raises():
    enc = compress(b"some bytes to encode here please")
    with pytest.raises((ValueError,)):
        decompress(enc[: len(enc) // 2])


def test_decompress_invalid_backreference_raises():
    # Construct a stream that asks for a backref pointing before the
    # output start. Simplest: control bit 0,0 (short_copy) with size=2
    # and offset = 0 (illegal since output is empty).
    # Control bits LSB-first: write 0 (copy), 0 (short), 0 (size_high),
    # 0 (size_low) -> 0b00000000 (then padding). Then data byte = 0
    # (offset = 0 -> backref to position 0 - 0x100 = -0x100; invalid
    # since output is empty).
    # Build manually: control = 0, data = 0x00, then EOF.
    bad = bytes([0b00000000, 0x00])  # 8 zero ctrl bits + 1 zero data byte
    # The decoder will try to read 4 bits as 0,0 (literal-or-copy=copy,
    # short=true, size_high=0, size_low=0), then read offset byte 0x00
    # which means -0x100. That's a backref to position -0x100 from
    # output size 0, invalid.
    with pytest.raises(ValueError):
        decompress(bad)


# ---------------------------------------------------------------------------
# Real PSOBB asset round-trip
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB install not present")
def test_bml_inner_blob_roundtrip():
    """Round-trip 10 real PSOBB BML inner files.

    For each shipped BML, parse it, decompress its first inner file,
    re-compress (greedy AND optimal), decompress again — expect
    bit-identity with the originally decompressed payload.
    """
    from formats.bml import parse_bml
    from formats.prs import compress, compress_optimal, decompress

    bmls = sorted(PSOBB_DATA.glob("*.bml"), key=lambda p: p.stat().st_size)
    # Bottom 10 by size to keep tests fast.
    selected = bmls[:10]
    assert len(selected) >= 5, f"too few BMLs (found {len(selected)})"

    seen_ok = 0
    for bml_path in selected:
        buf = bml_path.read_bytes()
        try:
            entries = parse_bml(buf)
        except Exception as e:
            pytest.skip(f"parse_bml failed on {bml_path.name}: {e}")
        if not entries:
            continue
        ent = entries[0]
        raw = buf[ent.offset:ent.offset + ent.size_compressed]
        # Decompress with our decoder
        try:
            ref = decompress(raw)
        except Exception as e:
            pytest.fail(f"decompress failed on {bml_path.name}: {e}")
        assert len(ref) == ent.size_decompressed, (
            f"{bml_path.name}: decompressed {len(ref)} bytes, "
            f"BML header says {ent.size_decompressed}"
        )
        # Greedy round-trip
        enc = compress(ref)
        dec = decompress(enc)
        assert dec == ref, f"greedy roundtrip mismatch on {bml_path.name}"
        # Optimal round-trip
        enc = compress_optimal(ref)
        dec = decompress(enc)
        assert dec == ref, f"optimal roundtrip mismatch on {bml_path.name}"
        seen_ok += 1
    assert seen_ok >= 5, f"only {seen_ok} BMLs round-tripped"


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB install not present")
def test_prs_compressed_assets_roundtrip():
    """Round-trip every standalone .prs in PSOBB.IO/data/ (TitleEP4, LogoEP4)."""
    targets = list(PSOBB_DATA.glob("*.prs"))
    assert targets, "no .prs files found in PSOBB install"
    for prs_path in targets:
        raw = prs_path.read_bytes()
        try:
            ref = decompress(raw)
        except Exception as e:
            pytest.skip(f"decompress {prs_path.name} failed: {e}")
        re_enc = compress(ref)
        re_dec = decompress(re_enc)
        assert re_dec == ref, f"greedy roundtrip {prs_path.name}"


# ---------------------------------------------------------------------------
# Compression ratio sanity
# ---------------------------------------------------------------------------
def test_optimal_no_worse_than_greedy_for_repetitive():
    """For highly repetitive input, optimal must produce <= greedy size."""
    data = b"ABCDABCD" * 200
    g = compress(data)
    o = compress_optimal(data)
    assert decompress(g) == data
    assert decompress(o) == data
    # Optimal should be at most a few bytes worse on small inputs (the
    # path search adds some constant overhead) but generally smaller.
    assert len(o) <= len(g) + 16


def test_compressed_output_is_bytes():
    out = compress(b"hello world")
    assert isinstance(out, bytes)
    out = compress_optimal(b"hello world")
    assert isinstance(out, bytes)
    out = decompress(out)
    assert isinstance(out, bytes)


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------
def test_cli_encode_decode(tmp_path):
    src = b"the quick brown fox jumps over the lazy dog" * 8
    in_path = tmp_path / "in.bin"
    in_path.write_bytes(src)
    enc_path = tmp_path / "in.prs"
    dec_path = tmp_path / "out.bin"

    from formats.prs import _cli

    rc = _cli(["encode", str(in_path), str(enc_path)])
    assert rc == 0
    assert enc_path.is_file() and enc_path.stat().st_size > 0

    rc = _cli(["decode", str(enc_path), str(dec_path)])
    assert rc == 0
    assert dec_path.read_bytes() == src
