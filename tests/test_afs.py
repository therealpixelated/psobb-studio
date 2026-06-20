"""Tests for formats.afs reader + writer.

Round-trip every shipped PSOBB AFS file byte-exactly when the live
install is present (skipped otherwise so CI keeps working). Plus a
handful of synthetic edge-case fixtures.
"""
from __future__ import annotations
import os

import struct
from pathlib import Path

import pytest

from formats.afs import (
    AFS_MAGIC,
    parse_afs,
    write_afs,
)

# Locate real PSOBB AFS files. Skip the live-asset tests if the install
# isn't present (e.g. CI build).
PSOBB_DATA = Path(os.path.expanduser("~/PSOBB.IO/data"))
HAS_PSOBB = PSOBB_DATA.is_dir()


# ---------------------------------------------------------------------------
# Edge cases — synthetic fixtures
# ---------------------------------------------------------------------------
def test_empty_archive_default_layout():
    """Zero-entry archive with the default PSOBB-compat layout."""
    out = write_afs([])
    # Default first_entry_offset = 0x80000 reserves 512 KB of zeros.
    assert len(out) == 0x80000
    assert out[:4] == AFS_MAGIC
    assert struct.unpack_from("<H", out, 4)[0] == 0
    # parse_afs round-trips an empty list.
    assert parse_afs(out) == []


def test_empty_archive_compact_layout():
    """``first_entry_offset=0`` falls back to the minimal layout."""
    out = write_afs([], first_entry_offset=0)
    # 8-byte header + 0 table entries -> aligned up to 0x800.
    assert len(out) == 0x800
    assert out[:4] == AFS_MAGIC
    assert parse_afs(out) == []


def test_single_entry_roundtrip():
    """Single inner blob; verify offset, size, and round-trip."""
    payload = b"PSO\x00" + b"\x01" * 100
    out = write_afs([payload])
    # First entry at 0x80000 by default; size = len(payload).
    assert struct.unpack_from("<H", out, 4)[0] == 1
    off, sz = struct.unpack_from("<II", out, 8)
    assert off == 0x80000
    assert sz == len(payload)
    # Round-trip through reader.
    assert parse_afs(out) == [payload]


def test_single_entry_compact_layout():
    """Compact layout: first entry just past the 0x800-aligned table."""
    payload = b"PSO\x00" + b"\x02" * 50
    out = write_afs([payload], first_entry_offset=0)
    off, sz = struct.unpack_from("<II", out, 8)
    # Header(8) + 1 table entry(8) = 16 bytes -> aligned up to 0x800.
    assert off == 0x800
    assert sz == len(payload)
    assert parse_afs(out) == [payload]


def test_multi_entry_alignment():
    """Five varied-size blobs; each entry must start at a 0x800 boundary."""
    blobs = [
        b"PSO\x00" + b"\x00" * (0x100 - 4),           # exactly aligned
        b"PSO\x00" + b"\x01" * (0x800 - 4),           # aligned to a full block
        b"PSO\x00" + b"\x02" * (0x801 - 4),           # 1 byte over a block
        b"PSO\x00" + b"\x03" * (0x10 - 4),            # tiny
        b"PSO\x00" + b"\x04" * (0x4000 - 4),          # 16 KB
    ]
    out = write_afs(blobs)
    fc = struct.unpack_from("<H", out, 4)[0]
    assert fc == len(blobs)
    last_end = 0
    for i in range(fc):
        off, sz = struct.unpack_from("<II", out, 8 + i * 8)
        assert off % 0x800 == 0, f"entry {i} offset 0x{off:x} not 0x800 aligned"
        assert sz == len(blobs[i])
        assert off >= last_end, f"entry {i} starts before previous ended"
        last_end = off + sz
    # Overall round-trip.
    assert parse_afs(out) == blobs


def test_dummy_5_entries_roundtrip():
    """Five dummy ``b'PSO\\x00\\x00\\x00\\x00\\x01' + zeros[N]`` entries.

    Mirrors the explicit verification step from the agent spec.
    """
    blobs = [
        b"PSO\x00\x00\x00\x00\x01" + b"\x00" * 100,
        b"PSO\x00\x00\x00\x00\x01" + b"\x00" * 200,
        b"PSO\x00\x00\x00\x00\x01" + b"\x00" * 0,
        b"PSO\x00\x00\x00\x00\x01" + b"\x00" * 0x900,  # spans two 0x800 blocks
        b"PSO\x00\x00\x00\x00\x01" + b"\x00" * 0x2000,
    ]
    out = write_afs(blobs)
    parsed = parse_afs(out)
    assert parsed == blobs
    fc = struct.unpack_from("<H", out, 4)[0]
    assert fc == 5
    for i in range(5):
        off, sz = struct.unpack_from("<II", out, 8 + i * 8)
        assert sz == len(blobs[i])
        assert off >= 0x80000


def test_zero_byte_entry():
    """A zero-byte slot still gets a valid (offset, 0) record."""
    out = write_afs([b"AAA", b"", b"BBB"])
    parsed = parse_afs(out)
    assert parsed == [b"AAA", b"", b"BBB"]


def test_name_table_when_provided():
    """When ``names`` is given, the writer emits a 48-byte/entry table."""
    blobs = [b"AAA", b"BBBB", b"CCCCC"]
    names = ["alpha.bin", "beta.bin", "gamma.bin"]
    out = write_afs(blobs, names=names)
    # Reader should still see the inner blobs in order (it ignores the
    # extra slot[file_count] descriptor).
    parsed = parse_afs(out)
    assert parsed == blobs
    # Verify slot[file_count] points at the name table.
    fc = struct.unpack_from("<H", out, 4)[0]
    name_off, name_sz = struct.unpack_from("<II", out, 8 + fc * 8)
    assert name_sz == 3 * 48
    # Read names back from the table.
    for i, expected in enumerate(names):
        entry_off = name_off + i * 48
        # 32-byte name + 16-byte mtime/pad
        name_bytes = out[entry_off:entry_off + 32].split(b"\x00", 1)[0]
        assert name_bytes.decode("ascii") == expected
        # mtime must be zero (matches shipped Sega/PSOBB convention).
        assert out[entry_off + 32:entry_off + 48] == b"\x00" * 16


def test_name_table_truncates_long_name():
    """Names longer than 32 bytes are silently truncated."""
    blobs = [b"hi"]
    long_name = "x" * 64
    out = write_afs(blobs, names=[long_name])
    parsed = parse_afs(out)
    assert parsed == blobs
    # Find the name table via slot[1].
    name_off, _ = struct.unpack_from("<II", out, 8 + 1 * 8)
    name_bytes = out[name_off:name_off + 32]
    assert name_bytes == b"x" * 32  # exactly 32 bytes, no NUL terminator


def test_writer_rejects_bad_inputs():
    """Argument validation surfaces as ValueError, not a crash."""
    with pytest.raises(ValueError, match="must be a list/tuple"):
        write_afs("not a list")
    with pytest.raises(ValueError, match="too many entries"):
        write_afs([b"x"] * 0x10000)
    with pytest.raises(ValueError, match="names has"):
        write_afs([b"a", b"b"], names=["only_one"])
    with pytest.raises(ValueError, match="negative"):
        write_afs([b"a"], first_entry_offset=-1)
    with pytest.raises(ValueError, match="overlap the file table"):
        write_afs([b"a"], first_entry_offset=4)
    with pytest.raises(ValueError, match="expected bytes-like"):
        write_afs(["not bytes"])


# ---------------------------------------------------------------------------
# Live shipped fixtures - skipped if PSOBB.IO isn't installed
# ---------------------------------------------------------------------------
def _shipped_afs_files() -> list[Path]:
    if not HAS_PSOBB:
        return []
    return sorted(PSOBB_DATA.glob("*.afs"))


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO/data not available")
@pytest.mark.parametrize("path", _shipped_afs_files(), ids=lambda p: p.name)
def test_shipped_afs_byte_exact_roundtrip(path: Path):
    """Every shipped AFS round-trips parse -> write byte-exact.

    Confirms our writer reproduces Sega's layout (0x80000 fixed first
    offset, 0x800 inter-blob padding, no name table).
    """
    buf = path.read_bytes()
    blobs = parse_afs(buf)
    rebuilt = write_afs(blobs)
    assert rebuilt == buf, (
        f"AFS round-trip mismatch for {path.name}: "
        f"orig={len(buf)} rebuilt={len(rebuilt)}"
    )


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO/data not available")
def test_shipped_afs_count_summary():
    """Sanity: enumerate the shipped archive set so the live count is
    visible in pytest's output even when individual cases pass."""
    files = _shipped_afs_files()
    assert len(files) >= 1, "no shipped AFS files found"
    # Every one parses without exceptions.
    for p in files:
        parse_afs(p.read_bytes())
