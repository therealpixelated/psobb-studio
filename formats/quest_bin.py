"""Byte-faithful codec for the PSOBB compiled quest container.

This is the Layer-0 foundation for the quest pipeline: it owns the
**container** (the ``.bin`` quest-script header + label table + raw code
+ string blob, the ``.qst`` network-command transport that ships it, and
the PRS (de)compression around it). It does NOT decode the bytecode
opcodes — the CODE bytes stay raw; a sibling assembler module decodes
them.

Three layers, three pairs of functions:

* ``.qst`` transport (a stream of PSO ``44``/``13`` server commands that
  the online client receives, each file's payload PRS-compressed):
  :func:`parse_qst` / :func:`serialize_qst`.
* ``.bin`` quest script (decompressed): :func:`parse_bin` /
  :func:`serialize_bin`. This is the **parity oracle target** — the
  decompressed ``.bin`` must round-trip byte-for-byte.
* PRS: reused verbatim from :mod:`formats.prs` (byte-exact in-house). We
  never reimplement it here.

Ground truth
------------
``.bin`` header layout — ``PSOQuestHeaderBB`` from newserv
``src/QuestScript.hh``. The physical ``.bin`` always starts with three
little-endian uint32 offsets + a 0xFFFFFFFF marker::

    /* 0000 */ uint32 code_offset        (= header size; 0x122C for BB)
    /* 0004 */ uint32 label_table_offset (start of the int32 label table)
    /* 0008 */ uint32 size               (= total file size)
    /* 000C */ uint32 unknown_a1/a2      (always 0xFFFFFFFF)

so the file carves cleanly into three regions regardless of which
header version follows:

    header    = bin[0          : code_offset]
    code      = bin[code_offset : label_table_offset]
    label_tab = bin[label_table_offset : size]   (int32[] relative offsets)

For BB (``code_offset == 0x122C``) the header body decodes as
``PSOQuestHeaderBB`` (newserv QuestScript.hh ~:98-133)::

    /* 0010 */ uint16 quest_number   (0xFFFF for challenge quests)
    /* 0012 */ uint16 unknown_a6
    /* 0014 */ uint8  episode        (0=Ep1, 1=Ep2, 2=Ep4)
    /* 0015 */ uint8  max_players    (0 means no limit, i.e. 4)
    /* 0016 */ uint8  joinable
    /* 0017 */ uint8  unknown_a4
    /* 0018 */ utf16  name[0x20]            (0x40 bytes)
    /* 0058 */ utf16  short_description[0x80] (0x100 bytes)
    /* 0158 */ utf16  long_description[0x120] (0x240 bytes)
    /* 0398 */ uint32 unknown_a5
    /* 039C */ uint16 solo_unlock_flags[8]
    /* 03AC */ FloorAssignment floor_assignments[0x10]   (8 bytes each)
    /* 042C */ CreateItemMaskEntry create_item_mask_entries[0x40] (0x38 each)
    /* 122C */ <end of header>

The trailing ``floor_assignments`` / ``create_item_mask_entries`` block
is preserved verbatim as raw bytes (it is not always present in
tool-authored quests, which truncate the header — but BB quests in the
wild carry the full 0x122C). We keep ``header_extra`` (everything after
the descriptions, up to ``code_offset``) as raw bytes so the round-trip
is exact and we still expose the structured accessors.

Hooks for the older PC / DC-GC header versions are present
(``BIN_FORMAT_*``, the ``code_offset`` discriminator) but only BB is
fully structured; PC/DC carve+round-trip via the generic region model.

``.qst`` transport — newserv ``src/Quest.cc`` ``decode_qst_data`` /
``encode_qst_file``. The online BB form is a stream of 8-byte-aligned
``PSOCommandHeaderBB`` commands:

* ``0x44`` (online open-file) carrying ``S_OpenFile_BB_44_A6`` (0x50
  bytes): unused[0x22], type u16, filename[0x10] ascii, file_size u32,
  name[0x18] ascii. Total command = 8 + 0x50 = 0x58.
* ``0x13`` (write-file) carrying ``S_WriteFile_13_A7`` (0x414 bytes):
  filename[0x10] ascii, data[0x400], data_size u32 — plus BB's extra 4
  alignment pad bytes, so each chunk is 8 + 0x414 + 4 = 0x420 bytes.

The download variant uses ``0xA6``/``0xA7`` and the payload is further
DLQ-encrypted; we read the command stream for it but only the online BB
form is fully supported (it's the one we ship). Other container variants
(PC/DC/GC/XB online) are detected from the 4-byte signature and read
where cheap.

JSON shape (header + structural view; CODE stays raw bytes, not JSON):
    {
      "format": "BB",
      "code_offset": 4652,
      "label_table_offset": 59428,
      "size": 71436,
      "quest_number": 118,
      "episode": 0,
      "max_players": 0,
      "joinable": 0,
      "name": "Towards the Future",
      "short_description": "...",
      "long_description": "...",
      "label_count": 3002,
      "code_size": 54776,
      "floor_assignments": [[floor,area,type,layout_var,entities_var], ...],
    }
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from formats import prs

# ---------------------------------------------------------------------------
# .bin header constants
# ---------------------------------------------------------------------------
# The bytecode (code) offset doubles as a version discriminator: it equals
# the header size, which differs per client generation. Values from
# phantasmal-world Bin.kt (DC_GC=468, PC=920, BB=4652) cross-checked
# against newserv QuestScript.hh struct sizes (PSOQuestHeaderBB = 0x122C).
BIN_FORMAT_DC_GC = "DC_GC"
BIN_FORMAT_PC = "PC"
BIN_FORMAT_BB = "BB"

CODE_OFFSET_DC_GC = 468  # 0x1D4
CODE_OFFSET_PC = 920  # 0x398
CODE_OFFSET_BB = 4652  # 0x122C

_CODE_OFFSET_TO_FORMAT = {
    CODE_OFFSET_DC_GC: BIN_FORMAT_DC_GC,
    CODE_OFFSET_PC: BIN_FORMAT_PC,
    CODE_OFFSET_BB: BIN_FORMAT_BB,
}

# BB header field offsets (within the header region [0, code_offset)).
_BB_QUEST_NUMBER_OFF = 0x10  # u16
_BB_UNKNOWN_A6_OFF = 0x12  # u16
_BB_EPISODE_OFF = 0x14  # u8
_BB_MAX_PLAYERS_OFF = 0x15  # u8
_BB_JOINABLE_OFF = 0x16  # u8
_BB_UNKNOWN_A4_OFF = 0x17  # u8
_BB_NAME_OFF = 0x18  # utf16, 0x40 bytes (0x20 chars)
_BB_NAME_BYTES = 0x40
_BB_SHORT_DESC_OFF = 0x58  # utf16, 0x100 bytes (0x80 chars)
_BB_SHORT_DESC_BYTES = 0x100
_BB_LONG_DESC_OFF = 0x158  # utf16, 0x240 bytes (0x120 chars)
_BB_LONG_DESC_BYTES = 0x240
_BB_HEADER_EXTRA_OFF = 0x398  # floor_assignments + item-mask block start
_BB_FLOOR_ASSIGNMENTS_OFF = 0x3AC
_BB_FLOOR_ASSIGNMENT_SIZE = 8
_BB_FLOOR_ASSIGNMENT_COUNT = 0x10

# ---------------------------------------------------------------------------
# .qst transport constants (online BB form)
# ---------------------------------------------------------------------------
# PSOCommandHeaderBB: size u16, command u16, flag u32 (8 bytes).
_QST_HEADER_SIZE = 8
# S_OpenFile_BB_44_A6 (0x50 bytes): unused[0x22], type u16, filename[0x10],
# file_size u32, name[0x18].
_QST_OPENFILE_BODY_SIZE = 0x50
_QST_OPENFILE_TYPE_OFF = 0x22
_QST_OPENFILE_FILENAME_OFF = 0x24
_QST_OPENFILE_FILENAME_LEN = 0x10
_QST_OPENFILE_FILESIZE_OFF = 0x34
_QST_OPENFILE_NAME_OFF = 0x38
_QST_OPENFILE_NAME_LEN = 0x18
# S_WriteFile_13_A7 (0x414 bytes): filename[0x10], data[0x400], data_size u32.
_QST_WRITEFILE_FILENAME_LEN = 0x10
_QST_WRITEFILE_CHUNK_BODY = 0x400
_QST_WRITEFILE_BODY_SIZE = 0x414  # filename + data + data_size
_QST_BB_WRITE_PAD = 4  # BB's implicit extra 4 alignment bytes per chunk

# Command IDs.
_CMD_OPEN_ONLINE = 0x44
_CMD_OPEN_DOWNLOAD = 0xA6
_CMD_WRITE_ONLINE = 0x13
_CMD_WRITE_DOWNLOAD = 0xA7

# Big-endian signatures of the first 4 bytes (newserv decode_qst_data).
_SIG_BB_ONLINE = 0x58004400
_SIG_BB_DOWNLOAD = 0x5800A600


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _decode_utf16(buf: bytes) -> str:
    """Decode a fixed-width UTF-16LE field, stopping at the first NUL."""
    text = buf.decode("utf-16-le", errors="replace")
    nul = text.find("\x00")
    if nul >= 0:
        text = text[:nul]
    return text


def _decode_ascii(buf: bytes) -> str:
    """Decode a NUL-terminated ASCII/latin-1 field (filenames, names)."""
    end = buf.find(b"\x00")
    if end >= 0:
        buf = buf[:end]
    return buf.decode("latin-1")


def _encode_ascii(text: str, width: int) -> bytes:
    """Encode ``text`` to a fixed-width NUL-padded ASCII field."""
    raw = text.encode("latin-1")[:width]
    return raw + b"\x00" * (width - len(raw))


# ---------------------------------------------------------------------------
# .bin (decompressed quest script container)
# ---------------------------------------------------------------------------
@dataclass
class QuestBin:
    """Parsed (decompressed) ``.bin`` quest-script container.

    The CODE bytes (``code``) are kept raw — the opcode assembler is a
    separate module. ``header_raw`` is the *entire* header region
    ``[0, code_offset)`` kept verbatim so serialize can reproduce it
    byte-exact even for fields we don't model. The structured fields
    (quest_number/episode/name/...) are *views* over ``header_raw`` for
    BB; mutate via :meth:`apply_header_fields` to write them back.
    """

    fmt: str  # BIN_FORMAT_BB / _PC / _DC_GC
    code_offset: int
    label_table_offset: int
    size: int
    unknown_marker: int  # the 0xFFFFFFFF field at 0x0C

    header_raw: bytes  # bytes [0, code_offset) verbatim
    code: bytes  # bytes [code_offset, label_table_offset) verbatim
    label_offsets: List[int]  # int32[] from [label_table_offset, size)

    # Structured BB header fields (None for non-BB / truncated headers).
    quest_number: Optional[int] = None
    unknown_a6: Optional[int] = None
    episode: Optional[int] = None
    max_players: Optional[int] = None
    joinable: Optional[int] = None
    unknown_a4: Optional[int] = None
    name: Optional[str] = None
    short_description: Optional[str] = None
    long_description: Optional[str] = None
    # Raw trailing header block [0x398, code_offset) for BB (floor
    # assignments + create-item masks). Empty for truncated headers.
    header_extra: bytes = b""

    # ---- structured accessors ---------------------------------------
    @property
    def code_size(self) -> int:
        return len(self.code)

    @property
    def label_count(self) -> int:
        return len(self.label_offsets)

    def floor_assignments(self) -> List[List[int]]:
        """Return the BB floor-assignment table as a list of 5-tuples.

        Each entry is ``[floor, area, type, layout_var, entities_var]``
        (the last 3 ``unused`` bytes are dropped). Empty when the header
        is not BB or doesn't reach the table.
        """
        if self.fmt != BIN_FORMAT_BB:
            return []
        table_end = (
            _BB_FLOOR_ASSIGNMENTS_OFF
            + _BB_FLOOR_ASSIGNMENT_COUNT * _BB_FLOOR_ASSIGNMENT_SIZE
        )
        if len(self.header_raw) < table_end:
            return []
        out: List[List[int]] = []
        for i in range(_BB_FLOOR_ASSIGNMENT_COUNT):
            off = _BB_FLOOR_ASSIGNMENTS_OFF + i * _BB_FLOOR_ASSIGNMENT_SIZE
            floor, area, typ, layout_var, entities_var = self.header_raw[off:off + 5]
            out.append([floor, area, typ, layout_var, entities_var])
        return out

    # ---- JSON -------------------------------------------------------
    def to_json(self) -> Dict:
        """Structural JSON view (header + offsets/sizes; code stays raw)."""
        return {
            "format": self.fmt,
            "code_offset": self.code_offset,
            "label_table_offset": self.label_table_offset,
            "size": self.size,
            "quest_number": self.quest_number,
            "unknown_a6": self.unknown_a6,
            "episode": self.episode,
            "max_players": self.max_players,
            "joinable": self.joinable,
            "name": self.name,
            "short_description": self.short_description,
            "long_description": self.long_description,
            "code_size": self.code_size,
            "label_count": self.label_count,
            "floor_assignments": self.floor_assignments(),
        }


def _carve_bb_header(header_raw: bytes) -> Dict:
    """Decode the structured BB header fields from the header region.

    Returns a dict suitable for splatting into :class:`QuestBin`. Fields
    that fall beyond a truncated header decode as ``None``.
    """
    out: Dict = {
        "quest_number": None,
        "unknown_a6": None,
        "episode": None,
        "max_players": None,
        "joinable": None,
        "unknown_a4": None,
        "name": None,
        "short_description": None,
        "long_description": None,
        "header_extra": b"",
    }
    n = len(header_raw)
    if n >= _BB_QUEST_NUMBER_OFF + 2:
        out["quest_number"] = struct.unpack_from("<H", header_raw, _BB_QUEST_NUMBER_OFF)[0]
    if n >= _BB_UNKNOWN_A6_OFF + 2:
        out["unknown_a6"] = struct.unpack_from("<H", header_raw, _BB_UNKNOWN_A6_OFF)[0]
    if n >= _BB_UNKNOWN_A4_OFF + 1:
        out["episode"] = header_raw[_BB_EPISODE_OFF]
        out["max_players"] = header_raw[_BB_MAX_PLAYERS_OFF]
        out["joinable"] = header_raw[_BB_JOINABLE_OFF]
        out["unknown_a4"] = header_raw[_BB_UNKNOWN_A4_OFF]
    if n >= _BB_NAME_OFF + _BB_NAME_BYTES:
        out["name"] = _decode_utf16(header_raw[_BB_NAME_OFF:_BB_NAME_OFF + _BB_NAME_BYTES])
    if n >= _BB_SHORT_DESC_OFF + _BB_SHORT_DESC_BYTES:
        out["short_description"] = _decode_utf16(
            header_raw[_BB_SHORT_DESC_OFF:_BB_SHORT_DESC_OFF + _BB_SHORT_DESC_BYTES]
        )
    if n >= _BB_LONG_DESC_OFF + _BB_LONG_DESC_BYTES:
        out["long_description"] = _decode_utf16(
            header_raw[_BB_LONG_DESC_OFF:_BB_LONG_DESC_OFF + _BB_LONG_DESC_BYTES]
        )
    if n > _BB_HEADER_EXTRA_OFF:
        out["header_extra"] = header_raw[_BB_HEADER_EXTRA_OFF:]
    return out


def parse_bin(data: bytes) -> QuestBin:
    """Parse a **decompressed** ``.bin`` quest-script container.

    ``data`` must be the raw decompressed bytes (the parity oracle).
    Carves into {header, code, label table} using the three uint32
    offsets at the start, then decodes the BB structured header fields.

    Raises:
        ValueError: input too short or self-inconsistent (offsets out of
            range).
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError("parse_bin: input must be bytes-like")
    data = bytes(data)
    if len(data) < 0x10:
        raise ValueError(f"parse_bin: too short ({len(data)} bytes) to be a .bin")

    code_offset, label_table_offset, size, marker = struct.unpack_from("<IIII", data, 0)

    if not (0 < code_offset <= len(data)):
        raise ValueError(
            f"parse_bin: code_offset 0x{code_offset:X} out of range (len {len(data)})"
        )
    if not (code_offset <= label_table_offset <= len(data)):
        raise ValueError(
            f"parse_bin: label_table_offset 0x{label_table_offset:X} out of range "
            f"(code_offset 0x{code_offset:X}, len {len(data)})"
        )

    fmt = _CODE_OFFSET_TO_FORMAT.get(code_offset, BIN_FORMAT_PC)

    header_raw = data[:code_offset]
    code = data[code_offset:label_table_offset]
    label_blob = data[label_table_offset:]
    if len(label_blob) % 4 != 0:
        # Tolerate a stray tail (some tools pad); keep only whole int32s.
        label_blob = label_blob[: len(label_blob) - (len(label_blob) % 4)]
    label_offsets = list(struct.unpack(f"<{len(label_blob) // 4}I", label_blob))

    qb = QuestBin(
        fmt=fmt,
        code_offset=code_offset,
        label_table_offset=label_table_offset,
        size=size,
        unknown_marker=marker,
        header_raw=header_raw,
        code=code,
        label_offsets=label_offsets,
    )

    if fmt == BIN_FORMAT_BB:
        for k, v in _carve_bb_header(header_raw).items():
            setattr(qb, k, v)

    return qb


def serialize_bin(qb: QuestBin) -> bytes:
    """Serialize a :class:`QuestBin` back to decompressed ``.bin`` bytes.

    Byte-exact inverse of :func:`parse_bin` for any well-formed input:
    ``serialize_bin(parse_bin(buf)) == buf``. The header region is
    written from ``header_raw`` verbatim (preserving every byte we don't
    model); ``code`` and the label table follow.
    """
    header = bytes(qb.header_raw)
    code = bytes(qb.code)
    label_blob = struct.pack(f"<{len(qb.label_offsets)}I", *qb.label_offsets)

    # Recompute the offset fields so a freshly-built (not parsed) QuestBin
    # also produces a consistent file. For a parsed round-trip these equal
    # the stored values, so the result is byte-identical.
    code_offset = len(header)
    label_table_offset = code_offset + len(code)
    total = label_table_offset + len(label_blob)

    out = bytearray(header)
    # Patch the three offset uint32s + the marker at the start.
    struct.pack_into("<III", out, 0, code_offset, label_table_offset, total)
    struct.pack_into("<I", out, 0x0C, qb.unknown_marker)
    out += code
    out += label_blob
    return bytes(out)


# ---------------------------------------------------------------------------
# .qst transport
# ---------------------------------------------------------------------------
@dataclass
class QstFile:
    """One file embedded in a ``.qst`` (e.g. the ``.bin`` or the ``.dat``).

    ``data`` is the PRS-**compressed** payload as carried in the
    transport. Use :meth:`decompressed` for the raw bytes. ``raw_header``
    is the original OpenFile command bytes (header + body), preserved so
    a parsed-then-reserialized ``.qst`` keeps its exact metadata.
    """

    filename: str  # internal filename, e.g. "quest118.bin"
    file_size: int  # declared size from the OpenFile command
    data: bytes  # PRS-compressed payload (transport bytes)
    name: str = ""  # display name from the OpenFile command
    type: int = 0
    flag: int = 0
    is_download: bool = False
    raw_open_command: Optional[bytes] = None  # original OpenFile cmd bytes

    def decompressed(self) -> bytes:
        """PRS-decompress this file's payload."""
        return prs.decompress(self.data)


@dataclass
class Qst:
    """Parsed ``.qst`` transport container."""

    fmt: str  # "BB" (only the online BB form is fully supported)
    online: bool
    files: List[QstFile] = field(default_factory=list)

    def file(self, *, suffix: str) -> Optional[QstFile]:
        """Return the first embedded file whose name ends with ``suffix``."""
        for f in self.files:
            if f.filename.lower().endswith(suffix.lower()):
                return f
        return None

    @property
    def bin_file(self) -> Optional[QstFile]:
        return self.file(suffix=".bin")

    @property
    def dat_file(self) -> Optional[QstFile]:
        return self.file(suffix=".dat")

    def to_json(self) -> Dict:
        return {
            "format": self.fmt,
            "online": self.online,
            "files": [
                {
                    "filename": f.filename,
                    "name": f.name,
                    "file_size": f.file_size,
                    "type": f.type,
                    "is_download": f.is_download,
                    "compressed_size": len(f.data),
                }
                for f in self.files
            ],
        }


def _detect_qst_format(data: bytes) -> Tuple[str, bool]:
    """Detect (.qst format, online?) from the leading 4-byte signature.

    Only the BB online/download forms are fully decoded; other versions
    raise (they use a different header/struct geometry we don't carve).
    """
    if len(data) < 4:
        raise ValueError("parse_qst: input too short to detect format")
    sig = struct.unpack_from(">I", data, 0)[0]
    if sig == _SIG_BB_ONLINE:
        return BIN_FORMAT_BB, True
    if sig == _SIG_BB_DOWNLOAD:
        return BIN_FORMAT_BB, False
    # PC: 3C 00 44/A6 ; DC/GC: 44/A6 ?? 3C 00 ; XB: 44/A6 ?? 54 00.
    raise ValueError(
        f"parse_qst: unsupported/unknown .qst signature 0x{sig:08X} "
        "(only the BlueBurst online/download form is supported)"
    )


def parse_qst(data: bytes) -> Qst:
    """Parse a ``.qst`` transport (online BlueBurst form).

    Walks the PSO command stream, gathering OpenFile (``0x44``/``0xA6``)
    declarations and WriteFile (``0x13``/``0xA7``) chunks, reassembling
    each embedded file's PRS-compressed payload. The decompressed ``.bin``
    can then be obtained via :meth:`Qst.bin_file` -> :meth:`QstFile.decompressed`.

    Raises:
        ValueError: bad signature, malformed command, or chunk-order
            inconsistency.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError("parse_qst: input must be bytes-like")
    data = bytes(data)
    fmt, online = _detect_qst_format(data)

    # files keyed by internal filename, preserving first-seen order.
    order: List[str] = []
    opens: Dict[str, QstFile] = {}
    bodies: Dict[str, bytearray] = {}

    pos = 0
    n = len(data)
    align = _QST_HEADER_SIZE  # BB commands are 8-byte aligned
    saw_download = False
    saw_online = False

    while pos + _QST_HEADER_SIZE <= n:
        # BB implicit 8-byte command alignment (newserv decode_qst_data_t).
        pos = (pos + (align - 1)) & ~(align - 1)
        if pos + _QST_HEADER_SIZE > n:
            break

        size, command, flag = struct.unpack_from("<HHI", data, pos)

        if command in (_CMD_OPEN_ONLINE, _CMD_OPEN_DOWNLOAD):
            is_dl = command == _CMD_OPEN_DOWNLOAD
            saw_download |= is_dl
            saw_online |= not is_dl
            body_off = pos + _QST_HEADER_SIZE
            if body_off + _QST_OPENFILE_BODY_SIZE > n:
                raise ValueError("parse_qst: truncated OpenFile command")
            body = data[body_off:body_off + _QST_OPENFILE_BODY_SIZE]
            typ = struct.unpack_from("<H", body, _QST_OPENFILE_TYPE_OFF)[0]
            filename = _decode_ascii(
                body[_QST_OPENFILE_FILENAME_OFF:
                      _QST_OPENFILE_FILENAME_OFF + _QST_OPENFILE_FILENAME_LEN]
            )
            file_size = struct.unpack_from("<I", body, _QST_OPENFILE_FILESIZE_OFF)[0]
            name = _decode_ascii(
                body[_QST_OPENFILE_NAME_OFF:
                      _QST_OPENFILE_NAME_OFF + _QST_OPENFILE_NAME_LEN]
            )
            if filename in opens:
                raise ValueError(f"parse_qst: file opened twice: {filename!r}")
            opens[filename] = QstFile(
                filename=filename,
                file_size=file_size,
                data=b"",
                name=name,
                type=typ,
                flag=flag,
                is_download=is_dl,
                raw_open_command=data[pos:body_off + _QST_OPENFILE_BODY_SIZE],
            )
            bodies[filename] = bytearray()
            order.append(filename)
            pos = body_off + _QST_OPENFILE_BODY_SIZE

        elif command in (_CMD_WRITE_ONLINE, _CMD_WRITE_DOWNLOAD):
            saw_download |= command == _CMD_WRITE_DOWNLOAD
            saw_online |= command == _CMD_WRITE_ONLINE
            cstart = pos + _QST_HEADER_SIZE
            if cstart + _QST_WRITEFILE_BODY_SIZE > n:
                raise ValueError("parse_qst: truncated WriteFile command")
            filename = _decode_ascii(
                data[cstart:cstart + _QST_WRITEFILE_FILENAME_LEN]
            )
            data_start = cstart + _QST_WRITEFILE_FILENAME_LEN
            chunk = data[data_start:data_start + _QST_WRITEFILE_CHUNK_BODY]
            data_size = struct.unpack_from(
                "<I", data, data_start + _QST_WRITEFILE_CHUNK_BODY
            )[0]
            if data_size > _QST_WRITEFILE_CHUNK_BODY:
                raise ValueError(
                    f"parse_qst: chunk data_size 0x{data_size:X} exceeds 0x400"
                )
            if filename not in bodies:
                raise ValueError(
                    f"parse_qst: write to unopened file {filename!r}"
                )
            buf = bodies[filename]
            # flag is the chunk index (offset / 0x400); enforce order.
            if (len(buf) // _QST_WRITEFILE_CHUNK_BODY) != flag:
                raise ValueError(
                    f"parse_qst: out-of-order chunk for {filename!r} "
                    f"(expected {len(buf) // _QST_WRITEFILE_CHUNK_BODY}, got {flag})"
                )
            buf += chunk[:data_size]
            # BB stores the implicit extra 4 alignment bytes in the file.
            advance = _QST_HEADER_SIZE + _QST_WRITEFILE_BODY_SIZE + _QST_BB_WRITE_PAD
            pos += advance

        else:
            raise ValueError(
                f"parse_qst: invalid command 0x{command:X} at offset 0x{pos:X}"
            )

    if saw_download and saw_online:
        raise ValueError("parse_qst: mixed online and download commands")

    # Finalize payloads and validate declared sizes.
    files: List[QstFile] = []
    for fn in order:
        qf = opens[fn]
        qf.data = bytes(bodies[fn])
        if qf.file_size and len(qf.data) != qf.file_size:
            raise ValueError(
                f"parse_qst: file {fn!r} assembled to {len(qf.data)} bytes, "
                f"declared {qf.file_size}"
            )
        files.append(qf)

    return Qst(fmt=fmt, online=not saw_download, files=files)


def _build_open_command(qf: QstFile) -> bytes:
    """Build a BB OpenFile (0x44/0xA6) command for ``qf``.

    Prefers the preserved original bytes (so a parsed round-trip keeps the
    exact metadata); otherwise synthesizes a fresh command.
    """
    if qf.raw_open_command is not None:
        return qf.raw_open_command

    body = bytearray(_QST_OPENFILE_BODY_SIZE)
    struct.pack_into("<H", body, _QST_OPENFILE_TYPE_OFF, qf.type & 0xFFFF)
    body[_QST_OPENFILE_FILENAME_OFF:
         _QST_OPENFILE_FILENAME_OFF + _QST_OPENFILE_FILENAME_LEN] = _encode_ascii(
        qf.filename, _QST_OPENFILE_FILENAME_LEN
    )
    struct.pack_into("<I", body, _QST_OPENFILE_FILESIZE_OFF, len(qf.data))
    body[_QST_OPENFILE_NAME_OFF:
         _QST_OPENFILE_NAME_OFF + _QST_OPENFILE_NAME_LEN] = _encode_ascii(
        qf.name or qf.filename, _QST_OPENFILE_NAME_LEN
    )
    command = _CMD_OPEN_DOWNLOAD if qf.is_download else _CMD_OPEN_ONLINE
    header = struct.pack(
        "<HHI", _QST_HEADER_SIZE + _QST_OPENFILE_BODY_SIZE, command, qf.flag
    )
    return header + bytes(body)


def _build_write_commands(qf: QstFile) -> bytes:
    """Build the BB WriteFile (0x13/0xA7) chunk commands for ``qf``."""
    out = bytearray()
    command = _CMD_WRITE_DOWNLOAD if qf.is_download else _CMD_WRITE_ONLINE
    fname = _encode_ascii(qf.filename, _QST_WRITEFILE_FILENAME_LEN)
    payload = qf.data
    total = len(payload)
    chunk_no = 0
    pos = 0
    # newserv emits at least one chunk; an empty file would emit none, but
    # quest payloads are never empty so we follow the data-driven loop.
    while pos < total:
        chunk = payload[pos:pos + _QST_WRITEFILE_CHUNK_BODY]
        size = len(chunk)
        # command size = header(8) + body(0x414); BB pad is appended after.
        header = struct.pack(
            "<HHI", _QST_HEADER_SIZE + _QST_WRITEFILE_BODY_SIZE, command, chunk_no
        )
        body = bytearray(_QST_WRITEFILE_BODY_SIZE)
        body[:_QST_WRITEFILE_FILENAME_LEN] = fname
        body[_QST_WRITEFILE_FILENAME_LEN:_QST_WRITEFILE_FILENAME_LEN + size] = chunk
        struct.pack_into(
            "<I", body, _QST_WRITEFILE_FILENAME_LEN + _QST_WRITEFILE_CHUNK_BODY, size
        )
        out += header
        out += body
        out += b"\x00" * _QST_BB_WRITE_PAD  # BB alignment pad
        chunk_no += 1
        pos += _QST_WRITEFILE_CHUNK_BODY
    return bytes(out)


def serialize_qst(qst: Qst) -> bytes:
    """Serialize a :class:`Qst` back to ``.qst`` transport bytes.

    Layout (newserv ``encode_qst_file`` for BB): all OpenFile commands
    first, then all WriteFile chunk commands. For a parsed-then-
    reserialized ``.qst`` the OpenFile commands are reproduced verbatim
    from the originals, so the metadata is byte-identical; the chunk
    stream is rebuilt from the (compressed) payloads.

    PRS re-compression is the caller's concern — this packs whatever
    compressed bytes are in each :class:`QstFile`. The byte-exactness
    guarantee of this module is on the *decompressed* ``.bin`` round-trip
    (:func:`parse_bin`/:func:`serialize_bin`), not on ``.qst`` re-pack.
    """
    out = bytearray()
    for qf in qst.files:
        out += _build_open_command(qf)
    for qf in qst.files:
        out += _build_write_commands(qf)
    return bytes(out)


# ---------------------------------------------------------------------------
# Convenience: full .qst -> decompressed .bin pipeline
# ---------------------------------------------------------------------------
def extract_bin(qst: Qst) -> bytes:
    """Return the decompressed ``.bin`` bytes from a parsed ``.qst``.

    This is the parity-oracle target: ``extract_bin(parse_qst(qst_bytes))``
    equals the standalone ``*_decompressed.bin`` fixture byte-for-byte.
    """
    bf = qst.bin_file
    if bf is None:
        raise ValueError("extract_bin: .qst contains no .bin file")
    return bf.decompressed()


def extract_dat(qst: Qst) -> bytes:
    """Return the decompressed ``.dat`` bytes from a parsed ``.qst``."""
    df = qst.dat_file
    if df is None:
        raise ValueError("extract_dat: .qst contains no .dat file")
    return df.decompressed()


def build_qst(
    bin_compressed: bytes,
    dat_compressed: bytes,
    *,
    bin_filename: str = "quest.bin",
    dat_filename: str = "quest.dat",
    name: str = "",
    is_download: bool = False,
) -> Qst:
    """Build a fresh online-BB :class:`Qst` from compressed payloads.

    Convenience for the export path: takes already-PRS-compressed ``.bin``
    and ``.dat`` payloads and wraps them. The order mirrors the wild
    fixtures (``.dat`` first, then ``.bin``).
    """
    files = [
        QstFile(
            filename=dat_filename,
            file_size=len(dat_compressed),
            data=dat_compressed,
            name=name or dat_filename,
            is_download=is_download,
        ),
        QstFile(
            filename=bin_filename,
            file_size=len(bin_compressed),
            data=bin_compressed,
            name=name or bin_filename,
            is_download=is_download,
        ),
    ]
    return Qst(fmt=BIN_FORMAT_BB, online=not is_download, files=files)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def to_json(obj) -> str:
    """Pretty-print a QuestBin or Qst as JSON."""
    return json.dumps(obj.to_json(), indent=2, ensure_ascii=False)
