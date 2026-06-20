# Ported from MIT-licensed Phantasmal World psolib by Daan Vanden Bosch.
# See LICENSES.md at the editor root for the verbatim MIT block.
#
# Reference:
#   psolib/src/commonMain/kotlin/world/phantasmal/psolib/fileFormats/Afs.kt
#
# Sega AFS archive layout (little-endian):
#
#     0x00   "AFS\0"      4-byte magic (= 0x00534641 LE)
#     0x04   u16 file_count
#     0x06   u16 padding   (typically 0; spec is silent)
#     0x08   per-file table: file_count * (u32 offset, u32 size)
#     ...    file payloads at the recorded offsets
#
# AFS files in PSOBB.IO/data/ store one PRS-compressed asset per slot
# (typically an NJ model or an XVM texture archive). This reader returns
# raw inner bytes verbatim - no PRS decompression is attempted here so
# the module stays subprocess-free.
"""Pure-Python reader and writer for the Sega AFS archive container."""
from __future__ import annotations

import struct
from typing import List, Optional, Sequence

# Header
AFS_MAGIC = b"AFS\x00"            # 0x00534641 LE - the 4 ASCII bytes (AFS1)
# AFS2 magic variant ('AFS ' = 0x20534641 LE). No PSOBB asset uses it,
# but SA-Tools / PSO2-Aqua-Library emit and accept it, so we recognise it
# on read for foreign-archive interop. Ref: PSO2-Aqua-Library
# SAToolsShared/ArchiveLib/AFS.cs:15-16,131-144.
AFS2_MAGIC = b"AFS\x20"           # 0x20534641 LE
_ACCEPTED_MAGICS = (AFS_MAGIC, AFS2_MAGIC)
_HEADER_SIZE = 8                  # magic + u16 count + u16 pad
_TABLE_ENTRY_SIZE = 8             # u32 offset + u32 size

# Sanity caps - real PSOBB AFS files have hundreds of entries, not
# millions. A bogus header that decodes a 0xFFFF count would otherwise
# allocate a 16 MB table read.
_MAX_FILE_COUNT = 0xFFFF

# Per-blob alignment used by every shipped PSOBB AFS. Each inner blob
# starts at a 0x800 boundary and is padded with NULs up to the next one.
_BLOB_ALIGN = 0x800

# PSOBB convention: every shipped *.afs has its first inner blob at a
# fixed 0x80000 (= 512 KB) absolute offset. The space between the end of
# the per-file table and 0x80000 is reserved (always zero in shipped
# files) - presumably a leftover of Sega's authoring tool reserving room
# for ~8 K table entries. We replicate this by default so write_afs()
# round-trips every shipped archive byte-exactly. Callers who need a
# more compact archive can pass first_entry_offset=0 to fall back to the
# minimal "table_end aligned up to 0x800" layout.
_PSOBB_FIRST_ENTRY_OFFSET = 0x80000

# Optional metadata table (Sega "AFS_PSO"/SA-Tools extension): emitted at
# the tail when names are provided. Each 48-byte entry is, per the
# PSO2-Aqua-Library oracle (SAToolsShared/ArchiveLib/AFS.cs:40-50,66-81):
#
#     0x00  char[32]  name        ASCII, NUL-padded
#     0x20  u16       year        \
#     0x22  u16       month        |
#     0x24  u16       day          | DateTime; Aqua's reader THROWS if
#     0x26  u16       hour         | month/day are 0 (DateTime(0,0,0,..)
#     0x28  u16       minute       | is invalid), so an all-zero timestamp
#     0x2A  u16       second      /  is NOT a valid meta table for interop.
#     0x2C  u32       custom_data  developer data; often == data size
#
# Shipped PSOBB *.afs carry NO metadata table at all (round-trip parity
# is unaffected: our default writer never emits one). When a caller DOES
# request names, write_afs emits a table whose timestamps default to a
# valid sentinel (year=1,month=1,day=1) so the result is parseable by
# Aqua's reader rather than producing the throwing 0/0/0 DateTime.
_NAME_ENTRY_SIZE = 48
_NAME_FIELD_SIZE = 32
_META_TS_OFFSET = 32             # u16 x6 timestamp block starts here
_META_CUSTOM_OFFSET = 44         # u32 custom_data
# Aqua DateTime(year,month,day,...) throws when month==0 or day==0. Use a
# valid minimum so a meta table we write can be read back by Aqua.
_META_DEFAULT_DATE = (1, 1, 1, 0, 0, 0)  # (year, month, day, h, m, s)


def parse_afs(buf: bytes) -> List[bytes]:
    """Parse a Sega AFS archive into a list of inner-file byte strings.

    Args:
        buf: full archive bytes.

    Returns:
        Ordered list of inner files, each as raw `bytes`. Order matches
        the table order in the archive (i.e. `files[i]` is entry `i`).

    Raises:
        ValueError: if the buffer is malformed - any of:
            - non-bytes input
            - shorter than the 8-byte header
            - magic mismatch
            - file_count too large to fit a table inside the buffer
            - any (offset, size) pair that escapes the buffer
            - any negative offset/size after sign extension
    """
    if not isinstance(buf, (bytes, bytearray, memoryview)):
        raise ValueError("parse_afs: input must be bytes-like")
    mv = memoryview(buf)
    n = len(mv)

    if n < _HEADER_SIZE:
        raise ValueError(
            f"parse_afs: truncated header (need {_HEADER_SIZE} bytes, have {n})"
        )
    magic = bytes(mv[:4])
    if magic not in _ACCEPTED_MAGICS:
        raise ValueError(
            f"parse_afs: bad magic {magic!r} "
            f"(expected {AFS_MAGIC!r} or {AFS2_MAGIC!r})"
        )

    # u16 file_count + u16 padding. Phantasmal World reads only the u16
    # count and seeks past the next 2 bytes.
    file_count = struct.unpack_from("<H", mv, 4)[0]
    if file_count > _MAX_FILE_COUNT:
        raise ValueError(f"parse_afs: file_count {file_count} too large")

    table_end = _HEADER_SIZE + file_count * _TABLE_ENTRY_SIZE
    if table_end > n:
        raise ValueError(
            f"parse_afs: table of {file_count} entries needs 0x{table_end:x} "
            f"bytes but buffer is only 0x{n:x}"
        )

    files: List[bytes] = []
    for i in range(file_count):
        entry_pos = _HEADER_SIZE + i * _TABLE_ENTRY_SIZE
        offset, size = struct.unpack_from("<II", mv, entry_pos)
        # u32s decode as non-negative, but be defensive in case a future
        # variant uses i32.
        if offset < 0 or size < 0:
            raise ValueError(
                f"parse_afs: entry {i} has negative offset/size "
                f"({offset}, {size})"
            )
        if offset > n:
            raise ValueError(
                f"parse_afs: entry {i} offset 0x{offset:x} exceeds "
                f"buffer 0x{n:x}"
            )
        end = offset + size
        if end > n:
            raise ValueError(
                f"parse_afs: entry {i} extends to 0x{end:x} but buffer "
                f"is only 0x{n:x} (offset=0x{offset:x}, size={size})"
            )
        files.append(bytes(mv[offset:end]))

    return files


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
def _align_up(v: int, a: int) -> int:
    """Round v up to the next multiple of a (a is a power of two)."""
    return (v + a - 1) & ~(a - 1)


def _normalize_timestamp(ts) -> tuple:
    """Coerce a timestamp into a 6-tuple of u16 (year..second).

    Accepts ``None`` (-> _META_DEFAULT_DATE), a ``datetime.datetime``, or
    an iterable of 6 ints. Each field is range-checked to fit a u16 so we
    never silently truncate. Mirrors Aqua's DateTime layout
    (AFS.cs:43-48,71-76).
    """
    if ts is None:
        return _META_DEFAULT_DATE
    # datetime.datetime duck-typing without importing datetime eagerly.
    if hasattr(ts, "year") and hasattr(ts, "second"):
        fields = (ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)
    else:
        fields = tuple(ts)
        if len(fields) != 6:
            raise ValueError(
                f"write_afs: timestamp must have 6 fields, got {len(fields)}"
            )
    out = []
    for v in fields:
        iv = int(v)
        if not (0 <= iv <= 0xFFFF):
            raise ValueError(
                f"write_afs: timestamp field {iv} out of u16 range 0..65535"
            )
        out.append(iv)
    return tuple(out)


def write_afs(
    entries: Sequence[bytes],
    names: Optional[Sequence[str]] = None,
    *,
    first_entry_offset: Optional[int] = None,
    pad_byte: int = 0x00,
    meta: Optional[Sequence[tuple]] = None,
) -> bytes:
    """Serialize a list of inner blobs as a Sega AFS archive.

    The output is byte-exact-compatible with the layout shipped with
    PSOBB: 8-byte header, ``len(entries) * 8`` byte table, NUL padding to
    ``first_entry_offset`` (default 0x80000 for PSOBB compatibility),
    each inner blob NUL-padded up to a 0x800 boundary, and an optional
    name table at the end if ``names`` is provided.

    Args:
        entries: ordered list of inner blobs; each ``bytes(...)``-like
            value becomes one slot in the archive in the given order.
        names: optional per-entry filenames. When provided, an
            AFS_PSO/SA-Tools-style 48-byte/entry metadata table is
            appended after the last padded blob, aligned to the next 0x800
            boundary. The offset+size of this table is stored in the slot
            **after** the file_count entries (i.e.
            `entry_table[file_count]` = Aqua's "OffsetEndTable" mode); the
            reader treats slot[file_count] as out-of-range so this is a
            backward-compatible extension. Each name is truncated /
            NUL-padded to 32 bytes. The 16-byte trailer is a 6x u16
            DateTime + u32 custom_data; see ``meta`` below for supplying
            real values. With ``meta=None`` the timestamp defaults to a
            VALID sentinel (0001-01-01 00:00:00) rather than all-zeros so
            the emitted table can be read back by Aqua's AFSMetadata
            reader (which throws on a 0/0/0 DateTime). Ref:
            PSO2-Aqua-Library AFS.cs:40-50,66-81,234-247.
            Default: ``None`` -> no metadata table.
        first_entry_offset: explicit absolute byte offset of the first
            inner blob. Defaults to ``0x80000`` (PSOBB convention). Pass
            ``0`` to use the minimal layout (``align_up(table_end, 0x800)``)
            for compact archives that don't need byte-equal-to-PSOBB
            round-trip.
        pad_byte: byte value for inter-blob padding. Defaults to 0x00,
            matching shipped archives.
        meta: optional per-entry ``(timestamp, custom_data)`` tuples used
            to populate the 16-byte metadata trailer when ``names`` is
            also given (ignored otherwise). ``timestamp`` may be a
            6-tuple ``(year, month, day, hour, minute, second)`` of u16
            values, a ``datetime.datetime``, or ``None`` for the default
            sentinel. ``custom_data`` is a u32. Must have the same length
            as ``entries`` when provided. Default: ``None`` -> every entry
            uses the valid sentinel date and custom_data 0.

    Returns:
        Full archive bytes, ready to write to disk.

    Raises:
        ValueError: invalid arguments (too many entries, mismatched
            ``names`` length, negative ``first_entry_offset``, or any
            entry that would push the table beyond the chosen first
            offset).
    """
    if not isinstance(entries, (list, tuple)):
        raise ValueError("write_afs: entries must be a list/tuple")
    n = len(entries)
    if n > _MAX_FILE_COUNT:
        raise ValueError(f"write_afs: too many entries {n} > {_MAX_FILE_COUNT}")
    if names is not None and len(names) != n:
        raise ValueError(
            f"write_afs: names has {len(names)} entries, expected {n}"
        )
    if meta is not None and len(meta) != n:
        raise ValueError(
            f"write_afs: meta has {len(meta)} entries, expected {n}"
        )
    if not (0 <= pad_byte <= 0xFF):
        raise ValueError(f"write_afs: pad_byte {pad_byte!r} out of range 0..255")

    # The per-file table also reserves one trailing slot for the optional
    # name-table descriptor when names are provided. PSOBB's reader (and
    # ours) ignores it because file_count is the loop bound; the older
    # AFS variant that DID consume this slot uses (off, sz) of the name
    # table here.
    table_slots = n + (1 if names is not None else 0)
    table_end = _HEADER_SIZE + table_slots * _TABLE_ENTRY_SIZE

    # Resolve first-blob offset.
    if first_entry_offset is None:
        first_off = _PSOBB_FIRST_ENTRY_OFFSET
    else:
        first_off = int(first_entry_offset)
    if first_off < 0:
        raise ValueError(f"write_afs: first_entry_offset {first_off} negative")
    if first_off == 0:
        first_off = _align_up(table_end, _BLOB_ALIGN)
    if first_off < table_end:
        raise ValueError(
            f"write_afs: first_entry_offset 0x{first_off:x} would overlap the "
            f"file table that ends at 0x{table_end:x}"
        )

    # Validate every entry is bytes-like up front so we can compute the
    # full size without surprise mid-loop.
    raw: List[bytes] = []
    for i, blob in enumerate(entries):
        if not isinstance(blob, (bytes, bytearray, memoryview)):
            raise ValueError(
                f"write_afs: entry {i} is {type(blob).__name__}, "
                f"expected bytes-like"
            )
        raw.append(bytes(blob))

    # Compute (offset, size) for each entry.
    offsets: List[int] = []
    sizes: List[int] = [len(b) for b in raw]
    cursor = first_off
    for i, sz in enumerate(sizes):
        offsets.append(cursor)
        # Each inner blob ends at the next 0x800 boundary; the next blob
        # starts there. Empty blobs still occupy one alignment unit so
        # offsets remain unique - matches Sega's behavior on the rare
        # zero-byte slot.
        cursor = _align_up(cursor + sz, _BLOB_ALIGN)
    blobs_end = cursor

    # Optional name-table tail.
    if names is not None:
        name_table_offset = blobs_end
        name_table_size = n * _NAME_ENTRY_SIZE
        # Pad the name-table region itself up to 0x800 so the file ends
        # on a 0x800 boundary - matches the shipped Sega convention.
        total_size = _align_up(name_table_offset + name_table_size, _BLOB_ALIGN)
    else:
        name_table_offset = 0
        name_table_size = 0
        total_size = blobs_end

    # Allocate the output buffer with the chosen pad byte.
    out = bytearray()
    out.extend(b"\x00" * total_size if pad_byte == 0 else bytes([pad_byte]) * total_size)

    # Header.
    struct.pack_into("<4sH H", out, 0, AFS_MAGIC, n, 0)

    # Per-file table.
    for i in range(n):
        struct.pack_into("<II", out, _HEADER_SIZE + i * _TABLE_ENTRY_SIZE,
                         offsets[i], sizes[i])
    if names is not None:
        # Slot[file_count] holds the (offset, size) of the name table.
        struct.pack_into("<II", out, _HEADER_SIZE + n * _TABLE_ENTRY_SIZE,
                         name_table_offset, name_table_size)

    # Inner blobs.
    for i, blob in enumerate(raw):
        if blob:
            out[offsets[i]:offsets[i] + len(blob)] = blob

    # Name table.
    if names is not None:
        for i, raw_name in enumerate(names):
            if not isinstance(raw_name, (str, bytes)):
                raise ValueError(
                    f"write_afs: name {i} is {type(raw_name).__name__}, "
                    f"expected str or bytes"
                )
            # Encode and truncate / NUL-pad to 32 bytes.
            if isinstance(raw_name, str):
                name_bytes = raw_name.encode("ascii", errors="replace")
            else:
                name_bytes = raw_name
            if len(name_bytes) > _NAME_FIELD_SIZE:
                name_bytes = name_bytes[:_NAME_FIELD_SIZE]
            entry_off = name_table_offset + i * _NAME_ENTRY_SIZE
            out[entry_off:entry_off + len(name_bytes)] = name_bytes
            # Timestamp + custom_data trailer.
            #
            # DEFAULT (meta is None): leave the 16-byte trailer all-zero.
            # This preserves the historical byte layout that server.py and
            # the existing test-suite depend on, and it matches the only
            # observed reality (shipped PSOBB *.afs have no meta table at
            # all). Note: an all-zero trailer is a 0/0/0 DateTime which
            # Aqua's *reader* rejects - so to produce an Aqua-interop meta
            # table the caller MUST supply ``meta`` with a valid date.
            #
            # When ``meta`` is supplied we write a real 6x u16 DateTime +
            # u32 custom_data per Aqua AFS.cs:71-77; _normalize_timestamp
            # defaults None -> 0001-01-01 (the valid minimum Aqua accepts).
            if meta is not None:
                ts_raw, custom = meta[i]
                ts = _normalize_timestamp(ts_raw)
                custom = int(custom) & 0xFFFFFFFF
                struct.pack_into("<6H", out, entry_off + _META_TS_OFFSET, *ts)
                struct.pack_into("<I", out, entry_off + _META_CUSTOM_OFFSET, custom)
            # else: trailing 16 bytes (name NUL-pad + timestamp + custom)
            # stay zero from the buffer fill.

    return bytes(out)
