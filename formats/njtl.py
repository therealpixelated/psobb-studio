# Ported from MIT-licensed Phantasmal World psolib by Daan Vanden Bosch.
# (Phantasmal's `Ninja.kt` skips the NJTL chunk; the field shapes used
# here come from the GPL pso-blender's `_modelwork/pso-blender/pso_blender/njtl.py`
# which we reference for layout only — no GPL code copied. The MIT
# attribution is preserved per repo convention because every parser in
# `formats/` ports from psolib.)
#
# Reference (specification only):
#   _modelwork/pso-blender/pso_blender/njtl.py — TextureList /
#   TextureListEntry struct shapes.
#
# NJTL — "Ninja Texture List" — is the per-model texture-name table
# emitted by Sega's Ninja exporter and embedded as one IFF chunk inside
# every textured PSOBB ``.nj`` / ``.xj`` model. Each model's submeshes
# carry a ``material_id`` that is BOTH the slot index in this list AND
# the index of the matching XVR record in the model's sibling XVMH
# archive. We need the names primarily for diagnostic visibility
# (the binding can also be done positionally because the writer
# emits NJTL entries and XVR records in the SAME order — see the
# pso-blender ``TextureManager`` for the mechanism).
#
# Chunk body layout (all little-endian):
#
#     +----------------------------------------------------+
#     | 0x00  u32 elements_offset  (PTR — relative to body)|
#     | 0x04  u32 count                                    |
#     +----------------------------------------------------+
#     | elements_offset: count * TextureListEntry          |
#     |   per entry (12 bytes):                            |
#     |     0x00  u32 name_ptr  (PTR — relative to body)   |
#     |     0x04  u32 unk1      (filled by client at load) |
#     |     0x08  u32 data_ptr  (filled by client at load) |
#     +----------------------------------------------------+
#     | dispersed: NUL-terminated ASCII texture names      |
#     | (typically packed BEFORE the elements array;       |
#     |  pso-blender's writer emits the strings first then |
#     |  the entry array, then patches `elements_offset`)  |
#     +----------------------------------------------------+
#
# The pointers (`elements_offset`, `name_ptr`, plus `unk1`/`data_ptr`)
# are POF0-rewriteable in the on-disk representation, but in PSOBB.IO
# the writer stores pre-relocated values that are simple offsets from
# the chunk body start (POF0 fixes them at load time on the original
# console targets — desktop PSOBB just dereferences them as-is). We
# decode them as raw u32 offsets without consulting POF0; this matches
# the way pso-blender's writer constructs them.
"""Pure-Python reader for PSOBB's Ninja Texture List (NJTL) chunk."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List, Optional

from .iff import parse_iff

NJTL_TAG = "NJTL"

# Header at chunk-body start: u32 elements_offset, u32 count.
_HEADER_FMT = "<II"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

# Per-entry struct: u32 name_ptr, u32 unk1, u32 data_ptr.
_ENTRY_FMT = "<III"
_ENTRY_SIZE = struct.calcsize(_ENTRY_FMT)

# Sanity caps. Real PSOBB models top out at ~16 textures per model;
# 1024 is paranoid but keeps a malformed pointer from blowing up our
# parse loop.
_MAX_ENTRIES = 1024
# Upper bound on a texture name length. PSO names are 31 chars max.
_MAX_NAME_LEN = 64


@dataclass
class NjtlEntry:
    """One slot in the Ninja Texture List.

    Attributes:
        slot:  slot index (matches the ``material_id`` on submeshes
               that reference this texture).
        name:  decoded NUL-terminated ASCII texture name.
    """
    slot: int
    name: str


def _read_c_string(buf: bytes, offset: int, max_len: int = _MAX_NAME_LEN) -> str:
    """Read a NUL-terminated ASCII string starting at ``offset``.

    Bounds: refuses to read past the buffer end or beyond ``max_len``.
    Falls back to latin-1 for the rare non-ASCII byte (PSO data is
    ASCII in practice; the latin-1 branch only exists so a single bad
    byte doesn't crash the parser).
    """
    if offset < 0 or offset >= len(buf):
        raise ValueError(f"NJTL: string pointer 0x{offset:x} out of range (buf={len(buf)})")
    end = buf.find(b"\x00", offset, offset + max_len + 1)
    if end < 0:
        raise ValueError(
            f"NJTL: unterminated string at 0x{offset:x} "
            f"(no NUL within {max_len} bytes)"
        )
    raw = buf[offset:end]
    try:
        return raw.decode("ascii")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def parse_njtl_chunk(body: bytes) -> List[NjtlEntry]:
    """Parse an NJTL chunk body and return one NjtlEntry per slot.

    Args:
        body: the bytes of the NJTL chunk (header excluded — i.e. what
              ``IffChunk.data`` carries).

    Returns:
        Ordered list of NjtlEntry. Entry at index ``i`` carries
        ``slot=i`` and the texture name at the same position.

    Raises:
        ValueError: malformed input — truncated header, count above
            the sanity cap, elements_offset past the buffer, or any
            individual entry/name pointer past the buffer.
    """
    if not isinstance(body, (bytes, bytearray, memoryview)):
        raise ValueError("parse_njtl_chunk: input must be bytes-like")
    body = bytes(body)
    n = len(body)
    if n < _HEADER_SIZE:
        raise ValueError(
            f"parse_njtl_chunk: truncated header (need {_HEADER_SIZE} bytes, have {n})"
        )

    elements_off, count = struct.unpack_from(_HEADER_FMT, body, 0)
    if count == 0:
        return []
    if count > _MAX_ENTRIES:
        raise ValueError(
            f"parse_njtl_chunk: implausible count {count} (max {_MAX_ENTRIES})"
        )
    end_of_entries = elements_off + count * _ENTRY_SIZE
    if elements_off < 0 or end_of_entries > n:
        raise ValueError(
            f"parse_njtl_chunk: entries 0x{elements_off:x}..0x{end_of_entries:x} "
            f"exceed chunk body 0x{n:x}"
        )

    out: List[NjtlEntry] = []
    for i in range(count):
        entry_off = elements_off + i * _ENTRY_SIZE
        name_ptr, _unk1, _data_ptr = struct.unpack_from(_ENTRY_FMT, body, entry_off)
        # An empty / null name slot is a real possibility in malformed
        # data; we tag it as "" rather than refusing the whole list.
        if name_ptr == 0:
            out.append(NjtlEntry(slot=i, name=""))
            continue
        try:
            name = _read_c_string(body, name_ptr)
        except ValueError as e:
            # Surface a more useful error including the slot index.
            raise ValueError(f"parse_njtl_chunk: slot {i}: {e}")
        out.append(NjtlEntry(slot=i, name=name))
    return out


def find_and_parse_njtl(model_bytes: bytes) -> Optional[List[NjtlEntry]]:
    """Convenience: find the NJTL chunk in a PSO IFF model and parse it.

    Args:
        model_bytes: full PSO IFF model body (the decompressed inner
            ``.nj`` / ``.xj`` payload — NOT the BML wrapper, NOT the
            outer XVMH).

    Returns:
        The parsed NJTL entry list, or ``None`` if the model carries no
        NJTL chunk (some primitive-only models legitimately omit it).

    Raises:
        ValueError: the IFF parse fails or the NJTL chunk is malformed.
    """
    chunks = parse_iff(model_bytes)
    njtl = next((c for c in chunks if c.type == NJTL_TAG), None)
    if njtl is None:
        return None
    return parse_njtl_chunk(njtl.data)
