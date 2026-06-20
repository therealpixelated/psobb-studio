"""Pure in-memory entry editing for container archives (AFS / BML).

This module is the format-agnostic core behind the ``/api/archive/*``
endpoints (duplicate / create / delete / rename an inner entry). It works
entirely on ``bytes`` so it is unit-testable without touching disk: each
function takes the full archive bytes and returns the rewritten archive
bytes (plus, where relevant, the new entry's index / name).

Supported containers:
  * AFS  (Sega AFS) -- via ``formats.afs.parse_afs`` / ``write_afs``
  * BML  (PSOBB model/texture pack) -- via ``formats.bml`` pack helpers
  * GSL / unknown   -- ``archive_kind`` returns ``None``; callers 400.

Error contract (so the HTTP layer can map cleanly):
  * ``ValueError``                 -> caller maps to 422 (pack / layout error)
  * ``KeyError`` / ``IndexError``  -> caller maps to 404 (entry not found)

CRITICAL TRAPS preserved here (see the module-level notes on each fn):
  (a) Shipped PSOBB AFS carry NO filename table (``_afs_filename_table``
      returns ``None``). We KEEP ``None`` as ``None`` -- synthesising a
      table would add a 48-byte/entry section the original lacked and
      break byte-parity. AFS rename is therefore only valid when a real
      name table already exists, else ``ValueError`` (-> 409 in the API).
  (b) BML duplicate ``copy.deepcopy``s the ``BmlPackEntry`` so the
      already-PRS ``is_compressed`` payload bytes are preserved verbatim,
      and passes ``has_textures_override`` from ``parse_bml_pack_meta`` so
      the 23 player-NJ "lying header" archives round-trip byte-exact.
"""
from __future__ import annotations

import copy
from typing import List, Optional, Tuple

from formats.afs import parse_afs, write_afs
from formats.afs_reader import _afs_filename_table
from formats.bml import (
    BmlPackEntry,
    NAME_FIELD_SIZE as _BML_NAME_FIELD_SIZE,
    pack_bml,
    parse_bml_for_pack,
    parse_bml_pack_meta,
)

__all__ = [
    "archive_kind",
    "afs_duplicate",
    "afs_create",
    "afs_delete",
    "afs_rename",
    "bml_duplicate",
    "bml_create",
    "bml_delete",
    "bml_rename",
]

# Template tokens for ``afs_create`` / ``bml_create`` when no blob is given.
_TEMPLATE_EMPTY = "empty"
_TEMPLATE_COPY_FIRST = "copy_first"
_VALID_TEMPLATES = (_TEMPLATE_EMPTY, _TEMPLATE_COPY_FIRST)


def archive_kind(name: str) -> Optional[str]:
    """Sniff the container kind from a filename suffix.

    Returns ``"afs"`` for ``*.afs``, ``"bml"`` for ``*.bml``, else
    ``None`` (GSL, unknown -- the caller turns ``None`` into HTTP 400).
    """
    if not name or not isinstance(name, str):
        return None
    lower = name.lower()
    if lower.endswith(".afs"):
        return "afs"
    if lower.endswith(".bml"):
        return "bml"
    return None


# ---------------------------------------------------------------------------
# AFS
# ---------------------------------------------------------------------------
def _afs_dedup_name(names: List[str], base: str) -> str:
    """Return a name not already in ``names`` by appending ``_copy[/_N]``."""
    existing = set(names)
    if base not in existing:
        return base
    candidate = f"{base}_copy"
    if candidate not in existing:
        return candidate
    i = 2
    while f"{base}_copy{i}" in existing:
        i += 1
    return f"{base}_copy{i}"


def afs_duplicate(buf: bytes, src_index: int) -> Tuple[bytes, int]:
    """Duplicate inner blob ``src_index``; return ``(new_buf, new_index)``.

    The duplicated blob bytes are byte-identical to the source. The new
    slot is appended at the end (``new_index == old_count``). ``write_afs``
    recomputes the offset table + 0x800 alignment automatically.

    Name table: preserved EXACTLY as recovered. ``None`` stays ``None`` (no
    synthesis); when a real table exists the duplicate gets a de-duplicated
    copy of the source name.

    Raises:
        IndexError: ``src_index`` out of range (-> 404).
        ValueError: archive too full for the grown table (-> 422).
    """
    blobs = list(parse_afs(buf))
    if src_index < 0 or src_index >= len(blobs):
        raise IndexError(
            f"afs_duplicate: src_index {src_index} out of range (count={len(blobs)})"
        )
    names = _afs_filename_table(buf)
    blobs.append(blobs[src_index])
    if names is not None:
        names = list(names)
        names.append(_afs_dedup_name(names, names[src_index]))
    new_buf = write_afs(blobs, names=names)
    return new_buf, len(blobs) - 1


def _afs_template_blob(blobs: List[bytes], template: Optional[str]) -> bytes:
    """Resolve a template token into the new blob's bytes."""
    tmpl = (template or _TEMPLATE_EMPTY).lower()
    if tmpl == _TEMPLATE_EMPTY:
        return b""
    if tmpl == _TEMPLATE_COPY_FIRST:
        if not blobs:
            raise ValueError(
                "afs_create: template 'copy_first' needs at least one existing entry"
            )
        return blobs[0]
    raise ValueError(
        f"afs_create: unknown template {template!r} (expected one of {_VALID_TEMPLATES})"
    )


def afs_create(
    buf: bytes,
    blob: Optional[bytes],
    template: Optional[str] = None,
    *,
    new_name: Optional[str] = None,
) -> Tuple[bytes, int]:
    """Append a new inner blob; return ``(new_buf, new_index)``.

    The new blob is either ``blob`` (when given) or derived from
    ``template`` (``"empty"`` -> b"", ``"copy_first"`` -> a copy of slot 0).

    Name table handling mirrors ``afs_duplicate``: synthesise a name ONLY
    when the archive already carries a real table; otherwise keep
    ``None`` so the byte layout is unchanged.

    Raises:
        ValueError: bad template, or archive too full (-> 422).
    """
    blobs = list(parse_afs(buf))
    if blob is None:
        new_blob = _afs_template_blob(blobs, template)
    else:
        new_blob = bytes(blob)
    names = _afs_filename_table(buf)
    blobs.append(new_blob)
    if names is not None:
        names = list(names)
        base = new_name if new_name else f"entry_{len(blobs) - 1:04d}"
        names.append(_afs_dedup_name(names, base))
    new_buf = write_afs(blobs, names=names)
    return new_buf, len(blobs) - 1


def afs_delete(buf: bytes, index: int) -> bytes:
    """Remove inner blob ``index``; return the rewritten archive bytes.

    All later slots renumber down by one (positional addressing). The name
    table, when present, drops the matching entry.

    Raises:
        IndexError: ``index`` out of range (-> 404).
        ValueError: would leave the archive in an unwritable state (-> 422).
    """
    blobs = list(parse_afs(buf))
    if index < 0 or index >= len(blobs):
        raise IndexError(
            f"afs_delete: index {index} out of range (count={len(blobs)})"
        )
    names = _afs_filename_table(buf)
    del blobs[index]
    if names is not None:
        names = list(names)
        del names[index]
    return write_afs(blobs, names=names)


def afs_rename(buf: bytes, index: int, new_name: str) -> bytes:
    """Rename inner blob ``index`` to ``new_name``.

    ONLY valid when the archive already carries a real Sega/SA-Tools
    filename table. Shipped PSOBB AFS have none, so this raises a
    ``ValueError`` (which the API maps to 409) rather than inventing a
    table and silently changing the byte layout.

    Raises:
        IndexError: ``index`` out of range (-> 404).
        ValueError: no name table present (-> 409) or duplicate name.
    """
    blobs = list(parse_afs(buf))
    if index < 0 or index >= len(blobs):
        raise IndexError(
            f"afs_rename: index {index} out of range (count={len(blobs)})"
        )
    names = _afs_filename_table(buf)
    if names is None:
        raise ValueError(
            "afs_rename: archive has no filename table; rename would change "
            "the byte layout (shipped PSOBB AFS omit the table)"
        )
    names = list(names)
    if not new_name:
        raise ValueError("afs_rename: new_name must be non-empty")
    for i, nm in enumerate(names):
        if i != index and nm == new_name:
            raise ValueError(f"afs_rename: name {new_name!r} already exists")
    names[index] = new_name
    return write_afs(blobs, names=names)


# ---------------------------------------------------------------------------
# BML
# ---------------------------------------------------------------------------
def _bml_find_index(entries: List[BmlPackEntry], name: str) -> int:
    """Return the index of the entry named ``name`` or raise KeyError."""
    for i, ent in enumerate(entries):
        if ent.name == name:
            return i
    raise KeyError(f"BML entry {name!r} not found")


def _bml_validate_new_name(entries: List[BmlPackEntry], new_name: str,
                           *, ignore_index: Optional[int] = None) -> None:
    """Validate a fresh BML entry name: <=32 ASCII bytes and unique.

    Raises ``ValueError`` (-> 422 / 409) on an empty, over-long, non-ASCII
    or colliding name. ``ignore_index`` skips one slot (the entry being
    renamed) when checking uniqueness.
    """
    if not new_name:
        raise ValueError("BML entry name must be non-empty")
    try:
        raw = new_name.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError(f"BML entry name {new_name!r} must be ASCII")
    if len(raw) > _BML_NAME_FIELD_SIZE:
        raise ValueError(
            f"BML entry name {new_name!r} exceeds {_BML_NAME_FIELD_SIZE} bytes"
        )
    for i, ent in enumerate(entries):
        if ignore_index is not None and i == ignore_index:
            continue
        if ent.name == new_name:
            raise ValueError(f"BML entry name {new_name!r} already exists")


def _bml_repack(buf: bytes, entries: List[BmlPackEntry]) -> bytes:
    """Re-pack ``entries`` preserving the source header's round-trip meta."""
    meta = parse_bml_pack_meta(buf)
    return pack_bml(
        entries,
        compression=meta["compression"],
        file_alignment=meta["file_alignment"],
        has_textures_override=meta["has_textures"],
    )


def bml_duplicate(buf: bytes, src_name: str, new_name: str) -> bytes:
    """Duplicate the BML entry named ``src_name`` under ``new_name``.

    The duplicate is a ``copy.deepcopy`` of the source ``BmlPackEntry`` so
    its already-PRS ``is_compressed`` payload (and any texture) is
    preserved verbatim -- no lossy re-compression. The header's
    ``has_textures``/alignment/compression meta is carried through so the
    lying-header player-NJ archives round-trip.

    Raises:
        KeyError: ``src_name`` not found (-> 404).
        ValueError: bad / duplicate ``new_name`` (-> 422 / 409).
    """
    entries = parse_bml_for_pack(buf)
    idx = _bml_find_index(entries, src_name)
    _bml_validate_new_name(entries, new_name)
    dup = copy.deepcopy(entries[idx])
    dup.name = new_name
    entries.append(dup)
    return _bml_repack(buf, entries)


def bml_create(
    buf: bytes,
    new_name: str,
    blob: bytes,
    is_compressed: bool = False,
    texture: Optional[bytes] = None,
) -> bytes:
    """Append a fresh BML entry ``new_name`` carrying ``blob``.

    ``is_compressed=True`` means ``blob`` is already PRS bytes (stored
    verbatim); ``False`` means the packer PRS-compresses it. ``texture``
    is optional already-PRS XVM bytes for the entry's sibling slot.

    Note: when ``is_compressed`` is True the packer needs a
    ``decompressed_size`` to write the entry record; since we cannot know
    it without decompressing, callers should pass raw bytes
    (``is_compressed=False``) for created-from-scratch entries. We still
    support the compressed path by recording ``len(blob)`` as a best-effort
    decompressed size, matching how a same-bytes round-trip behaves.

    Raises:
        ValueError: bad / duplicate ``new_name`` (-> 422 / 409).
    """
    entries = parse_bml_for_pack(buf)
    _bml_validate_new_name(entries, new_name)
    data = bytes(blob)
    ent = BmlPackEntry(
        name=new_name,
        data=data,
        decompressed_size=len(data),
        is_compressed=bool(is_compressed),
    )
    if texture is not None and len(texture) > 0:
        tex = bytes(texture)
        ent.texture_data = tex
        ent.texture_decompressed_size = len(tex)
        ent.texture_is_compressed = True
    entries.append(ent)
    return _bml_repack(buf, entries)


def bml_delete(buf: bytes, name: str) -> bytes:
    """Remove the BML entry named ``name``; return rewritten bytes.

    Raises:
        KeyError: ``name`` not found (-> 404).
        ValueError: would leave an empty archive (-> 422).
    """
    entries = parse_bml_for_pack(buf)
    idx = _bml_find_index(entries, name)
    del entries[idx]
    # pack_bml raises ValueError on an empty list; surface it as-is.
    return _bml_repack(buf, entries)


def bml_rename(buf: bytes, old: str, new: str) -> bytes:
    """Rename the BML entry ``old`` to ``new``; return rewritten bytes.

    Raises:
        KeyError: ``old`` not found (-> 404).
        ValueError: bad / duplicate ``new`` name (-> 422 / 409).
    """
    entries = parse_bml_for_pack(buf)
    idx = _bml_find_index(entries, old)
    _bml_validate_new_name(entries, new, ignore_index=idx)
    entries[idx].name = new
    return _bml_repack(buf, entries)
