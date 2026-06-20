# Ported from MIT-licensed Phantasmal World psolib by Daan Vanden Bosch.
# See LICENSES.md at the editor root for the verbatim MIT block.
#
# Reference:
#   psolib/src/commonMain/kotlin/world/phantasmal/psolib/fileFormats/Iff.kt
#
# PSO uses a *little-endian* variant of the EA-IFF-85 chunked container
# format. Each chunk has an 8-byte header:
#
#     bytes 0..3   4-char ASCII type tag (e.g. "NJCM", "POF0", "NJTL")
#     bytes 4..7   u32 little-endian body size in bytes
#
# followed by `body_size` bytes of payload. Chunks are written
# back-to-back; PSO does not pad to alignment between top-level chunks
# (only the writer's POF0 builder rounds to 4 internally).
#
# This module is read-only and intentionally does *not* attempt to
# interpret POF0 (the pointer-fixup table) - callers that need POF0
# semantics should re-read it as a chunk and decode it elsewhere.
"""Pure-Python reader for PSOBB's little-endian IFF container."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List

# 4-byte type tag + u32 body size
_HEADER_FMT = "<4sI"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


@dataclass
class IffChunk:
    """One top-level IFF chunk.

    Attributes:
        type: 4-char ASCII tag (e.g. "NJCM"). If the source bytes are not
              valid ASCII the tag is decoded as latin-1 so callers can
              still inspect it without raising.
        data: raw chunk payload (does NOT include the 8-byte header).
    """
    type: str
    data: bytes


def parse_iff(buf: bytes) -> List[IffChunk]:
    """Parse a PSO little-endian IFF byte string into a list of chunks.

    Args:
        buf: full file bytes (or any sub-region that begins on a chunk
             header boundary).

    Returns:
        Ordered list of IffChunk, each with `type` (4-char str) and
        `data` (bytes payload, header excluded). Empty list if the
        buffer is empty.

    Raises:
        ValueError: if the buffer is malformed - any of:
            - non-bytes input
            - truncated header (fewer than 8 bytes left where a chunk
              would start)
            - chunk size that overflows the remaining buffer
            - header tag containing a NUL byte (mid-tag NULs indicate
              corruption / wrong endian read)
    """
    if not isinstance(buf, (bytes, bytearray, memoryview)):
        raise ValueError("parse_iff: input must be bytes-like")
    mv = memoryview(buf)
    n = len(mv)
    chunks: List[IffChunk] = []
    pos = 0

    while pos < n:
        remaining = n - pos
        if remaining < _HEADER_SIZE:
            raise ValueError(
                f"parse_iff: truncated chunk header at offset 0x{pos:x} "
                f"(need {_HEADER_SIZE} bytes, have {remaining})"
            )
        type_bytes, size = struct.unpack_from(_HEADER_FMT, mv, pos)
        # Validate the 4-byte tag - real PSO chunks are printable ASCII.
        # A NUL inside the tag is an unambiguous corruption signal (would
        # otherwise let "AB\0\0" match as type "AB").
        if b"\x00" in type_bytes:
            raise ValueError(
                f"parse_iff: invalid chunk tag {bytes(type_bytes)!r} at "
                f"offset 0x{pos:x} (contains NUL)"
            )
        try:
            type_str = type_bytes.decode("ascii")
        except UnicodeDecodeError:
            # Fall back to latin-1 so we never crash on a single odd
            # byte - but keep the type a plain str for downstream JSON
            # serialization.
            type_str = type_bytes.decode("latin-1")

        body_start = pos + _HEADER_SIZE
        body_end = body_start + size
        if size < 0 or body_end > n:
            raise ValueError(
                f"parse_iff: chunk {type_str!r} at offset 0x{pos:x} declares "
                f"size 0x{size:x} which exceeds buffer (have 0x{n - body_start:x} "
                f"bytes after header)"
            )

        chunks.append(IffChunk(type=type_str, data=bytes(mv[body_start:body_end])))
        pos = body_end

    return chunks
