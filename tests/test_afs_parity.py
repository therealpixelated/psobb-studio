"""Parity tests for the AFS audit ports (DOMAIN afs).

Validates the three robustness/interop ports applied per the audit
against faithful Python models of the oracle behavior:

  Oracle: PSO2-Aqua-Library SAToolsShared/ArchiveLib/AFS.cs
          (the only oracle implementing the metadata/timestamp table in
           two placement modes plus the AFS2 magic variant)
          libpsoarchive/doc/afs.txt + src/AFS-read.c (format spec)
          newserv-sparse/src/AFSArchive.cc (canonical write alignment)

Ports under test:
  1. Reader: two-mode metadata-descriptor detection (OffsetEndTable AND
     OffsetBeforeFirstEntry).                       -> formats.afs_reader
  2. Reader+writer: true 48-byte metadata entry layout (32 name +
     6x u16 DateTime + u32 custom_data).            -> both modules
  3. Reader: AFS2 magic variant ('AFS ' = 0x20534641) recognition.
                                                     -> formats.afs

Plus a regression guard: real shipped *.afs still round-trip byte-exact
through parse_afs -> write_afs() with default args (the audit focus item).
"""
from __future__ import annotations
import os

import struct
from pathlib import Path

import pytest

from formats.afs import AFS_MAGIC, AFS2_MAGIC, parse_afs, write_afs
from formats import afs_reader

# Real shipped assets (skip when the install isn't present).
_DATA_DIRS = [
    Path(os.path.expanduser("~/EphineaPSO/data")),
    Path(os.path.expanduser("~/PSOBB.IO/data")),
]


def _shipped_afs_files() -> list[Path]:
    out: list[Path] = []
    for d in _DATA_DIRS:
        if d.is_dir():
            out.extend(sorted(d.glob("*.afs")))
    return out


# ---------------------------------------------------------------------------
# Faithful Python port of Aqua's AFSMetadata.GetBytes (AFS.cs:66-81) so we
# can assert our writer produces byte-identical metadata entries.
# ---------------------------------------------------------------------------
def _aqua_meta_entry(name: str, ts: tuple, custom: int) -> bytes:
    # 32-byte name region (name copied in, rest zero) — AFS.cs:69,79.
    buf = bytearray(32)
    nb = name.encode("ascii")
    buf[:len(nb)] = nb
    # 6x u16 DateTime (year, month, day, hour, minute, second) — AFS.cs:71-76.
    buf += struct.pack("<6H", *ts)
    # u32 custom_data — AFS.cs:77.
    buf += struct.pack("<I", custom & 0xFFFFFFFF)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Faithful builder of an Aqua "OffsetBeforeFirstEntry" archive. Models the
# placement Aqua reads at AFS.cs:176-180 / writes at AFS.cs:253-255.
# We keep alignment simple (0x800) which is all the reader needs.
# ---------------------------------------------------------------------------
_ALIGN = 0x800


def _align_up(v: int, a: int = _ALIGN) -> int:
    return (v + a - 1) & ~(a - 1)


def _build_before_first_archive(blobs, names, stamps, customs, magic=AFS_MAGIC):
    n = len(blobs)
    table_end = 8 + n * 8
    # Reserve the meta descriptor 8 bytes before the first entry; place the
    # first entry at an aligned offset that leaves room for it.
    first_off = _align_up(table_end + 8)
    # Lay out blobs.
    offsets = []
    cursor = first_off
    for b in blobs:
        offsets.append(cursor)
        cursor = _align_up(cursor + len(b))
    blobs_end = cursor
    meta_off = blobs_end
    meta_entries = b"".join(
        _aqua_meta_entry(names[i], stamps[i], customs[i]) for i in range(n)
    )
    total = _align_up(meta_off + len(meta_entries))
    out = bytearray(total)
    out[:4] = magic
    struct.pack_into("<I", out, 4, n)
    for i in range(n):
        struct.pack_into("<II", out, 8 + i * 8, offsets[i], len(blobs[i]))
    # before-first descriptor at first_off - 8 (OffsetBeforeFirstEntry).
    struct.pack_into("<II", out, first_off - 8, meta_off, len(meta_entries))
    for i, b in enumerate(blobs):
        out[offsets[i]:offsets[i] + len(b)] = b
    out[meta_off:meta_off + len(meta_entries)] = meta_entries
    return bytes(out)


def _build_end_table_archive(blobs, names, stamps, customs):
    """Aqua OffsetEndTable mode: descriptor in slot[file_count]."""
    n = len(blobs)
    table_end = 8 + n * 8
    first_off = _align_up(table_end + 8)  # leave the descriptor slot room
    offsets = []
    cursor = first_off
    for b in blobs:
        offsets.append(cursor)
        cursor = _align_up(cursor + len(b))
    meta_off = cursor
    meta_entries = b"".join(
        _aqua_meta_entry(names[i], stamps[i], customs[i]) for i in range(n)
    )
    total = _align_up(meta_off + len(meta_entries))
    out = bytearray(total)
    out[:4] = AFS_MAGIC
    struct.pack_into("<I", out, 4, n)
    for i in range(n):
        struct.pack_into("<II", out, 8 + i * 8, offsets[i], len(blobs[i]))
    struct.pack_into("<II", out, table_end, meta_off, len(meta_entries))
    for i, b in enumerate(blobs):
        out[offsets[i]:offsets[i] + len(b)] = b
    out[meta_off:meta_off + len(meta_entries)] = meta_entries
    return bytes(out)


# ===========================================================================
# Port 3: AFS2 magic recognition
# ===========================================================================
def test_afs2_magic_constant_value():
    # 0x20534641 LE = b'AFS ' (AFS.cs:16).
    assert struct.unpack("<I", AFS2_MAGIC)[0] == 0x20534641
    assert struct.unpack("<I", AFS_MAGIC)[0] == 0x00534641


def test_parse_afs_accepts_afs2_magic():
    payload = b"PSO\x00" + b"\x07" * 64
    # Build a normal archive then swap the magic to AFS2.
    out = bytearray(write_afs([payload]))
    out[:4] = AFS2_MAGIC
    assert parse_afs(bytes(out)) == [payload]


def test_parse_afs_still_rejects_garbage_magic():
    bad = bytearray(write_afs([b"x" * 8]))
    bad[:4] = b"ZZZZ"
    with pytest.raises(ValueError, match="bad magic"):
        parse_afs(bytes(bad))


# ===========================================================================
# Port 1: two-mode metadata descriptor detection
# ===========================================================================
def test_reader_recovers_names_end_table_mode():
    blobs = [b"AAA", b"BBBB", b"CCCCC"]
    names = ["alpha.bin", "beta.bin", "gamma.bin"]
    stamps = [(2020, 1, 2, 3, 4, 5)] * 3
    customs = [3, 4, 5]
    buf = _build_end_table_archive(blobs, names, stamps, customs)
    assert parse_afs(buf) == blobs
    assert afs_reader._afs_filename_table(buf) == names


def test_reader_recovers_names_before_first_mode():
    """The port's headline fix: before-first archives no longer lose names."""
    blobs = [b"AAA", b"BBBB", b"CCCCC"]
    names = ["one.nj", "two.xvm", "three.pvr"]
    stamps = [(2021, 6, 19, 12, 0, 0)] * 3
    customs = [0, 0, 0]
    buf = _build_before_first_archive(blobs, names, stamps, customs)
    # Sanity: the end-of-table slot is zero in this mode, so the OLD reader
    # (slot[file_count]-only) would have returned None.
    fc = struct.unpack_from("<H", buf, 4)[0]
    end_slot = struct.unpack_from("<II", buf, 8 + fc * 8)
    assert end_slot == (0, 0), "before-first fixture must leave end slot zero"
    # The ported reader recovers them.
    assert parse_afs(buf) == blobs
    assert afs_reader._afs_filename_table(buf) == names


def test_reader_no_meta_returns_none():
    # Our own default writer emits no meta table -> reader returns None.
    buf = write_afs([b"AAA", b"BBB"])
    assert afs_reader._afs_filename_table(buf) is None
    assert afs_reader._afs_metadata_table(buf) is None


# ===========================================================================
# Port 2: true 48-byte metadata layout (timestamp + custom_data)
# ===========================================================================
def test_reader_surfaces_timestamp_and_custom_data():
    blobs = [b"AAA", b"BBBB"]
    names = ["first", "second"]
    stamps = [(2019, 12, 31, 23, 59, 58), (2000, 1, 1, 0, 0, 0)]
    customs = [0xDEADBEEF, 3]
    buf = _build_end_table_archive(blobs, names, stamps, customs)
    table = afs_reader._afs_metadata_table(buf)
    assert table is not None
    assert [e["name"] for e in table] == names
    assert [e["timestamp"] for e in table] == stamps
    assert [e["custom_data"] for e in table] == customs


def test_writer_meta_entry_matches_aqua_layout():
    """Each 48-byte entry write_afs emits == Aqua's AFSMetadata.GetBytes."""
    blobs = [b"AAA", b"BBBB"]
    names = ["alpha", "beta"]
    stamps = [(2022, 3, 4, 5, 6, 7), (1999, 11, 12, 13, 14, 15)]
    customs = [0x11223344, 0]
    out = write_afs(blobs, names=names, meta=list(zip(stamps, customs)))
    fc = struct.unpack_from("<H", out, 4)[0]
    meta_off, meta_sz = struct.unpack_from("<II", out, 8 + fc * 8)
    assert meta_sz == fc * 48
    for i in range(fc):
        got = out[meta_off + i * 48: meta_off + (i + 1) * 48]
        want = _aqua_meta_entry(names[i], stamps[i], customs[i])
        assert got == want, f"entry {i} mismatch vs Aqua layout"


def test_writer_meta_roundtrips_through_our_reader():
    blobs = [b"x" * 10, b"y" * 2000, b""]
    names = ["a.nj", "b.xvm", "c.pvr"]
    stamps = [(2010, 5, 5, 1, 2, 3), (2011, 6, 6, 4, 5, 6), (2012, 7, 7, 7, 8, 9)]
    customs = [1, 2, 3]
    out = write_afs(blobs, names=names, meta=list(zip(stamps, customs)))
    assert parse_afs(out) == blobs
    table = afs_reader._afs_metadata_table(out)
    assert table is not None
    assert [e["name"] for e in table] == names
    assert [e["timestamp"] for e in table] == stamps
    assert [e["custom_data"] for e in table] == customs


def test_writer_meta_default_date_is_aqua_valid():
    """meta=None default trailer stays all-zero (preserves legacy layout);
    meta with timestamp=None yields a VALID (non-throwing) Aqua DateTime."""
    # Legacy default: trailer all-zero (what server.py / shipped layout use).
    out_legacy = write_afs([b"AAA"], names=["n"])
    fc = struct.unpack_from("<H", out_legacy, 4)[0]
    meta_off, _ = struct.unpack_from("<II", out_legacy, 8 + fc * 8)
    assert out_legacy[meta_off + 32: meta_off + 48] == b"\x00" * 16

    # meta with timestamp None -> sentinel (1,1,1,0,0,0): month/day != 0 so
    # Aqua's DateTime(year,month,day,..) does NOT throw.
    out_meta = write_afs([b"AAA"], names=["n"], meta=[(None, 0)])
    meta_off2, _ = struct.unpack_from("<II", out_meta, 8 + fc * 8)
    ts = struct.unpack_from("<6H", out_meta, meta_off2 + 32)
    year, month, day = ts[0], ts[1], ts[2]
    assert (year, month, day) == (1, 1, 1)
    assert month >= 1 and day >= 1  # the invariant Aqua's reader requires


def test_writer_meta_accepts_datetime():
    import datetime
    dt = datetime.datetime(2023, 8, 15, 10, 30, 45)
    out = write_afs([b"AAA"], names=["n"], meta=[(dt, 7)])
    table = afs_reader._afs_metadata_table(out)
    assert table is not None
    assert table[0]["timestamp"] == (2023, 8, 15, 10, 30, 45)
    assert table[0]["custom_data"] == 7


def test_writer_meta_length_validation():
    with pytest.raises(ValueError, match="meta has"):
        write_afs([b"a", b"b"], names=["a", "b"], meta=[((2020, 1, 1, 0, 0, 0), 0)])


def test_writer_timestamp_range_validation():
    with pytest.raises(ValueError, match="u16 range"):
        write_afs([b"a"], names=["a"], meta=[((70000, 1, 1, 0, 0, 0), 0)])
    with pytest.raises(ValueError, match="6 fields"):
        write_afs([b"a"], names=["a"], meta=[((2020, 1, 1), 0)])


# ===========================================================================
# Regression guard: real shipped *.afs still byte-exact (audit focus item)
# ===========================================================================
@pytest.mark.skipif(not _shipped_afs_files(), reason="no shipped *.afs installed")
@pytest.mark.parametrize(
    "path", _shipped_afs_files(), ids=lambda p: f"{p.parent.parent.name}/{p.name}"
)
def test_shipped_afs_byte_exact_after_ports(path: Path):
    buf = path.read_bytes()
    blobs = parse_afs(buf)
    rebuilt = write_afs(blobs)
    assert rebuilt == buf, (
        f"AFS round-trip mismatch for {path}: "
        f"orig={len(buf)} rebuilt={len(rebuilt)}"
    )
    # Real PSOBB files carry no metadata table.
    assert afs_reader._afs_filename_table(buf) is None
