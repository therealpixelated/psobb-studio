"""Magic-sniff sibling-archive discovery for model files.

PSOBB-family models are split across at least one model file (.bml /
.nj / .xj) and one or more SIBLING archive files in the same directory
that hold the textures. The pairing convention varies by build:

    Dreamcast original    .nj  + .pvm   (Sega PVM container of PVRT)
    Gamecube              .nj  + .gvm   (Sega GVM container of GVRT)
    Xbox                  .bml + inline XVMH or .xvm
    PSOBB BB / PSOBB.IO   .bml + inline XVMH or .xvm

Some builds (community ports, the Phantasmal-derived test corpora)
also ship raw-magic siblings:

    PVRT / GBIX           single PVR texture file
    GVRT                  single GVR (Gamecube) texture file
    PRS-compressed XVM/PVM/GVM (lz77 stream wrapping any of the above)
    AFS                   indexed archive of any of the above

This module sniffs files NEXT TO a model path by their FIRST FOUR
BYTES (after auto-decompressing PRS) and dispatches each match to a
small wrapper class that exposes:

    SiblingArchive.path            location on disk
    SiblingArchive.magic           4-byte magic seen (after PRS unwrap)
    SiblingArchive.list_tiles()    -> list[str] (one entry per inner texture)
    SiblingArchive.extract_tile(n) -> bytes (raw PVR/GVR/XVR bytes)

The intent is to PROVIDE A LAST-RESORT TEXTURE SOURCE for models whose
inline XVMH coverage is incomplete and whose cross-archive lookup
(``texture_index.lookup``) finds nothing. Callers extract the raw
texture bytes and hand them to ``formats.pvr_decode.decode_pvr`` (PVR /
GVR siblings) or ``xvr_codec`` (XVR siblings).

We do NOT copy any code from ``_reference/PSOBMLExtract`` (the
reimplementation is independent — the magic-table is documented
publicly in `format/bml`, `format/pvm`, and Sega's PowerVR SDK).

Public API
----------

``SIBLING_MAGICS``
    Ordered tuple of (4-byte magic, format-id) tuples we know how to
    decode. Used both by the discovery helper and by tests that want
    to enumerate "what could possibly pair with this model".

``SiblingArchive``
    Lightweight wrapper. Each subclass implements list_tiles +
    extract_tile for one container format. Subclasses are picked by
    the magic discovered at construction time.

``discover_sibling_textures(model_path)``
    Lists files in the model's directory whose stem matches the
    model's stem (or whose stem is the model's stem + a known
    suffix), returns wrappers for every one whose first 4 bytes
    (after one PRS-unwrap pass) match a known texture-archive magic.
"""
from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

from formats import prs as prs_mod

log = logging.getLogger("psobb_editor.sibling_archives")


# ---------------------------------------------------------------------------
# Magic table — ordered for unambiguous prefix match.
#
# (4-byte magic, label) — label is a short identifier used by the
# discovery output and tests.
# ---------------------------------------------------------------------------

SIBLING_MAGICS: Tuple[Tuple[bytes, str], ...] = (
    (b"PVMH", "PVM"),     # Sega PVM (container of PVRT)
    (b"GVMH", "GVM"),     # Sega GVM (container of GVRT)
    (b"XVMH", "XVM"),     # PSOBB Xbox VR archive
    (b"PVRT", "PVR"),     # Single PVR texture (GBIX may precede)
    (b"GBIX", "PVR"),     # PVR with global index header
    (b"GVRT", "GVR"),     # Single GVR texture
    (b"AFS\x00", "AFS"),  # AFS archive of textures
)

# Filename suffixes we expect to find as siblings to a model. We DON'T
# require these — magic-sniff is the source of truth — but they're a
# useful hint for "files I can stat" passes.
_KNOWN_TEX_SUFFIXES = (
    ".pvm", ".gvm", ".xvm", ".pvr", ".gvr", ".prs", ".afs",
)

# PRS-decompressed inner sniff window — PRS's leading bitmap byte and
# literal runs leave the inner header roughly intact in the first 16
# bytes for the textures we care about, but a multi-pass decompress is
# always safer.
_PRS_SNIFF_WINDOW = 64


# ---------------------------------------------------------------------------
# Public dataclass / wrappers
# ---------------------------------------------------------------------------


@dataclass
class _ParsedSiblingHeader:
    magic: str           # 4-byte magic as ASCII (or hex if not ASCII)
    raw_bytes: bytes     # decompressed bytes (after PRS unwrap if any)
    was_prs: bool        # True if we unwrapped a PRS layer
    on_disk_size: int    # original on-disk size before unwrap


def _peek_magic(data: bytes) -> Optional[Tuple[bytes, str]]:
    """Return (magic, label) if data's first 4 bytes match a known one."""
    head = data[:4]
    for magic, label in SIBLING_MAGICS:
        if head == magic:
            return (magic, label)
    return None


def _maybe_unwrap_prs(data: bytes) -> Tuple[bytes, bool]:
    """If data is PRS-compressed (no known magic at offset 0 but a known
    magic appears within the first 64 bytes, indicative of a PRS literal
    run), decompress it. Otherwise return the bytes as-is.

    Returns (decompressed_bytes, was_prs).
    """
    if not data:
        return data, False
    if _peek_magic(data) is not None:
        return data, False
    sniff = data[:_PRS_SNIFF_WINDOW]
    found_magic = False
    for magic, _ in SIBLING_MAGICS:
        if magic in sniff:
            found_magic = True
            break
    if not found_magic:
        return data, False
    try:
        return prs_mod.decompress(data), True
    except (ValueError, IndexError, RuntimeError) as e:
        log.debug("PRS unwrap failed: %s", e)
        return data, False


# ---------------------------------------------------------------------------
# Container parsers — each returns list of (offset, length, name) tuples
# pointing at one INNER texture record inside the container.
# ---------------------------------------------------------------------------


def _parse_pvm_records(blob: bytes) -> List[Tuple[int, int, str]]:
    """Walk a PVMH archive, return [(offset, length, name), ...].

    PVMH layout (Sega PVM, used by Dreamcast tools):
        +0x00  'PVMH'
        +0x04  u32 LE header size (size AFTER offset 0x08 — i.e.
                                    flags + count + optional per-record
                                    tables, NOT counting the chunk
                                    header itself)
        +0x08  u16 LE flags (bit0 = filename-table present,
                              bit1 = pixel-format table,
                              bit2 = dim table,
                              bit3 = global-index table)
        +0x0A  u16 LE record count
        +0x0C+ optional per-record entries (depend on flags)
        +0x08+headersize  first PVRT (each PVRT is 'PVRT' + u32 size + u32 hdr)
    """
    if blob[:4] != b"PVMH":
        return []
    header_size = struct.unpack_from("<I", blob, 0x04)[0]
    # header_size must cover at least the 4 bytes of flags+count.
    if header_size < 0x04 or 0x08 + header_size > len(blob):
        return []
    out: List[Tuple[int, int, str]] = []
    # Walk PVRT records starting at the end of the header.
    pos = 0x08 + header_size
    idx = 0
    while pos + 8 <= len(blob):
        if blob[pos:pos + 4] == b"GBIX":
            # GBIX: 'GBIX' + u32 (rest of GBIX chunk size) + payload + then PVRT
            gbix_size = struct.unpack_from("<I", blob, pos + 0x04)[0]
            gbix_total = 8 + gbix_size
            pvrt_off = pos + gbix_total
            if pvrt_off + 8 > len(blob) or blob[pvrt_off:pvrt_off + 4] != b"PVRT":
                break
            pvrt_size = struct.unpack_from("<I", blob, pvrt_off + 0x04)[0]
            record_len = (pvrt_off + 8 + pvrt_size) - pos
            out.append((pos, record_len, f"pvrt_{idx:04d}"))
            pos += record_len
        elif blob[pos:pos + 4] == b"PVRT":
            pvrt_size = struct.unpack_from("<I", blob, pos + 0x04)[0]
            record_len = 8 + pvrt_size
            out.append((pos, record_len, f"pvrt_{idx:04d}"))
            pos += record_len
        else:
            # No more PVRT records.
            nxt = blob.find(b"PVRT", pos)
            if nxt < 0:
                break
            pos = nxt
        idx += 1
        if idx > 1024:
            break
    return out


def _parse_gvm_records(blob: bytes) -> List[Tuple[int, int, str]]:
    """Walk a GVMH archive, return [(offset, length, name), ...].

    GVMH is the Gamecube counterpart to PVMH. Same big-picture layout
    but the inner texture chunks are GVRT (and may be preceded by GBIX
    just like PVR).
    """
    if blob[:4] != b"GVMH":
        return []
    header_size = struct.unpack_from(">I", blob, 0x04)[0]
    # GVMH uses BIG-ENDIAN size word (Gamecube-native).
    if header_size < 0x04 or 0x08 + header_size > len(blob):
        return []
    out: List[Tuple[int, int, str]] = []
    pos = 0x08 + header_size
    idx = 0
    while pos + 8 <= len(blob):
        if blob[pos:pos + 4] == b"GBIX":
            gbix_size = struct.unpack_from(">I", blob, pos + 0x04)[0]
            gbix_total = 8 + gbix_size
            gvrt_off = pos + gbix_total
            if gvrt_off + 8 > len(blob) or blob[gvrt_off:gvrt_off + 4] != b"GVRT":
                break
            gvrt_size = struct.unpack_from(">I", blob, gvrt_off + 0x04)[0]
            record_len = (gvrt_off + 8 + gvrt_size) - pos
            out.append((pos, record_len, f"gvrt_{idx:04d}"))
            pos += record_len
        elif blob[pos:pos + 4] == b"GVRT":
            gvrt_size = struct.unpack_from(">I", blob, pos + 0x04)[0]
            record_len = 8 + gvrt_size
            out.append((pos, record_len, f"gvrt_{idx:04d}"))
            pos += record_len
        else:
            nxt = blob.find(b"GVRT", pos)
            if nxt < 0:
                break
            pos = nxt
        idx += 1
        if idx > 1024:
            break
    return out


def _parse_xvm_records(blob: bytes) -> List[Tuple[int, int, str]]:
    """Walk an XVMH archive, return [(offset, length, name), ...].

    Layout matches what server._list_xvmh_records walks. We replicate
    the parse here so this module is self-contained.
    """
    if blob[:4] != b"XVMH":
        return []
    out: List[Tuple[int, int, str]] = []
    pos = 0x40
    idx = 0
    while pos + 0x40 <= len(blob):
        if blob[pos:pos + 4] != b"XVRT":
            nxt = blob.find(b"XVRT", pos)
            if nxt < 0:
                break
            pos = nxt
            continue
        body_size = struct.unpack_from("<I", blob, pos + 0x04)[0]
        record_len = 8 + body_size
        out.append((pos, record_len, f"xvrt_{idx:04d}"))
        pos += record_len
        idx += 1
        if idx > 1024:
            break
    return out


def _parse_afs_records(blob: bytes) -> List[Tuple[int, int, str]]:
    """Walk an AFS archive, return [(offset, length, name), ...].

    AFS layout: 'AFS\\0', u32 entry_count, then entry_count × (u32
    offset, u32 size), then the per-entry blobs.
    """
    if blob[:4] != b"AFS\x00":
        return []
    if len(blob) < 0x10:
        return []
    count = struct.unpack_from("<I", blob, 0x04)[0]
    if count > 8192 or count == 0:
        return []
    table_size = count * 8
    if 0x08 + table_size > len(blob):
        return []
    out: List[Tuple[int, int, str]] = []
    for i in range(count):
        off = struct.unpack_from("<I", blob, 0x08 + i * 8)[0]
        sz = struct.unpack_from("<I", blob, 0x08 + i * 8 + 4)[0]
        if off + sz > len(blob) or sz == 0:
            continue
        out.append((off, sz, f"afs_{i:04d}"))
    return out


# ---------------------------------------------------------------------------
# SiblingArchive base + per-magic subclasses
# ---------------------------------------------------------------------------


class SiblingArchive:
    """Represents one texture-archive file paired with a model.

    Public attributes:
        path            absolute Path to the on-disk file
        magic           4-byte magic AS A STRING (e.g. 'PVM', 'GVM',
                        'XVM', 'PVR', 'GVR', 'AFS')
        was_prs         True if the file was PRS-compressed on disk
        size            on-disk size in bytes

    Subclasses implement list_tiles + extract_tile for one container
    type. Use the factory ``SiblingArchive.from_path`` / discovery
    helper rather than instantiating directly.
    """

    def __init__(
        self,
        path: Path,
        magic: str,
        raw_bytes: bytes,
        *,
        was_prs: bool = False,
    ) -> None:
        self.path = path
        self.magic = magic
        self.was_prs = was_prs
        self.size = path.stat().st_size if path.exists() else len(raw_bytes)
        self._raw = raw_bytes  # post-PRS-unwrap bytes
        self._records: Optional[List[Tuple[int, int, str]]] = None

    def __repr__(self) -> str:
        n_tiles = len(self.list_tiles())
        return (
            f"<SiblingArchive path={self.path.name!r} magic={self.magic!r} "
            f"prs={self.was_prs} tiles={n_tiles}>"
        )

    def _ensure_records(self) -> None:
        if self._records is not None:
            return
        if self.magic == "PVM":
            self._records = _parse_pvm_records(self._raw)
        elif self.magic == "GVM":
            self._records = _parse_gvm_records(self._raw)
        elif self.magic == "XVM":
            self._records = _parse_xvm_records(self._raw)
        elif self.magic == "AFS":
            self._records = _parse_afs_records(self._raw)
        elif self.magic in ("PVR", "GVR"):
            # Single-record container: the entire file IS the record.
            self._records = [(0, len(self._raw), f"{self.magic.lower()}_0000")]
        else:
            self._records = []

    def list_tiles(self) -> List[str]:
        """Return the per-tile name list."""
        self._ensure_records()
        return [r[2] for r in (self._records or [])]

    def extract_tile(self, name_or_index: Union[str, int]) -> bytes:
        """Extract the raw inner-texture bytes for one tile.

        For PVM/PVR siblings this returns a complete PVR file (PVRT
        chunk, optionally GBIX-prefixed). For GVM/GVR likewise. For XVM
        an XVRT record. For AFS the inner blob (which may itself be PRS
        or another container — caller is responsible for further
        recursion via _maybe_unwrap_prs / SiblingArchive on the bytes).

        Raises:
            IndexError on out-of-range numeric indices.
            KeyError on unknown name strings.
        """
        self._ensure_records()
        records = self._records or []
        if isinstance(name_or_index, int):
            if name_or_index < 0 or name_or_index >= len(records):
                raise IndexError(
                    f"tile index {name_or_index} out of range "
                    f"(0..{len(records) - 1})"
                )
            off, sz, _ = records[name_or_index]
        else:
            for off, sz, n in records:
                if n == name_or_index:
                    break
            else:
                raise KeyError(f"no tile named {name_or_index!r}")
        return bytes(self._raw[off:off + sz])

    @classmethod
    def from_path(cls, path: Path) -> Optional["SiblingArchive"]:
        """Load a sibling archive from disk; return None if no magic matches.

        Performs ONE PRS-unwrap pass: if the on-disk file's first 4
        bytes don't match a known magic but the first 64 bytes contain
        one (PRS literal-run heuristic), we decompress and retry. The
        ``was_prs`` flag records whether this happened so callers can
        pass-through info to the cache layer.
        """
        try:
            data = path.read_bytes()
        except OSError as e:
            log.debug("SiblingArchive.from_path: read failed for %s: %s", path, e)
            return None
        if len(data) < 8:
            return None
        # Direct magic match on disk.
        m = _peek_magic(data)
        if m is not None:
            magic_bytes, label = m
            return cls(path, label, data, was_prs=False)
        # PRS unwrap fallback.
        unwrapped, was_prs = _maybe_unwrap_prs(data)
        if was_prs:
            m = _peek_magic(unwrapped)
            if m is not None:
                magic_bytes, label = m
                return cls(path, label, unwrapped, was_prs=True)
        return None


# ---------------------------------------------------------------------------
# Public discovery helper
# ---------------------------------------------------------------------------


def discover_sibling_textures(model_path: Path) -> List[SiblingArchive]:
    """List sibling-archive texture files next to a model.

    Walks ``model_path.parent`` and returns SiblingArchive wrappers for
    every file in the directory whose magic matches one of the
    SIBLING_MAGICS entries (after one PRS-unwrap pass) AND whose stem
    pairs with the model:

        plain pair:    <model>.<tex_ext>          (e.g. foo.nj + foo.pvm)
        suffix pair:   <model>_tex.<tex_ext>      (foo.nj + foo_tex.xvm)

    The match is intentionally PERMISSIVE — anything in the same
    directory that sniffs as a known texture archive AND shares the
    model's stem (or is a known one-off like a 'common' texture pack)
    counts. Callers can filter further by checking the returned
    archive's tile-count against the model's NJTL.

    Returns an empty list (not None) if nothing matches; the model
    file's existence is NOT checked here (a missing model is a 404
    handled by the caller).
    """
    out: List[SiblingArchive] = []
    parent = model_path.parent
    if not parent.exists():
        return out
    stem = model_path.stem.lower()
    # Build a canonical lookup of "files that could pair with this model
    # by stem". The match is loose: the sibling's stem starts with the
    # model's stem OR equals model.stem + '_tex'.
    candidates: List[Path] = []
    for child in parent.iterdir():
        if not child.is_file():
            continue
        if child == model_path:
            continue
        cstem = child.stem.lower()
        cext = child.suffix.lower()
        # Accept exact-stem siblings (foo.nj <-> foo.pvm) and
        # _tex-suffixed (foo.nj <-> foo_tex.xvm).
        if cstem == stem or cstem == f"{stem}_tex":
            candidates.append(child)
        elif cext in _KNOWN_TEX_SUFFIXES and stem in cstem:
            # Looser: foo_inner.xvm where 'foo' is the model stem.
            candidates.append(child)
    # Magic-sniff each candidate.
    for c in candidates:
        try:
            arc = SiblingArchive.from_path(c)
        except Exception as e:  # pragma: no cover — defensive
            log.warning("sibling-archive load failed for %s: %s", c, e)
            continue
        if arc is not None:
            out.append(arc)
    return out


__all__ = [
    "SIBLING_MAGICS",
    "SiblingArchive",
    "discover_sibling_textures",
]
