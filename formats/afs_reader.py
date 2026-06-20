"""High-level AFS unpacking helpers used by manifest.py.

Wraps ``formats.afs.parse_afs`` with:
  - per-blob inner-format sniffing (PRS / NJTL / XVMH / NJCM / etc.)
  - an on-disk cache at ``cache/afs/<archive_stem>/<index>.bin``
  - synthesised pretty paths of the form
    ``<archive>#<NNNN>_<sniffed_name>.<ext>`` for the asset tree

The cache is content-keyed by ``(stat.st_size, int(stat.st_mtime))`` of the
parent ``.afs`` so an upstream rebuild invalidates every entry.

PRS payloads are decompressed via ``formats.bml._prs_decompress``, which
defaults to the in-process Python decoder (no subprocess fork+exec).
When the legacy PuyoToolsCli path is selected (PSO_USE_PUYOTOOLSCLI=1)
and the binary is unavailable, the cache falls back to the raw
compressed bytes so the entry is still indexable; the ``parsable`` flag
lands as ``"no"``.

This module is intentionally tolerant: a single bad blob (malformed
PRS, garbage payload) just records ``warnings`` on that entry — the
manifest build never aborts.
"""
from __future__ import annotations

import logging
import os
import struct
from pathlib import Path
from typing import Iterable, Iterator, Optional

from formats import afs as afs_mod

log = logging.getLogger("psobb_editor.afs_reader")

# Inner-format magic table. Same shape as manifest._MAGIC_TABLE but
# duplicated locally so we can also detect raw NJTL chunks (PSOBB items
# typically wrap their model in a NJTL+POF0+NJCM IFF triple).
_INNER_MAGIC: list[tuple[bytes, str, str, str]] = [
    # (magic, format_id, category, extension_for_synth_path)
    (b"NJTL", "NJ_IFF", "model",     ".nj"),
    (b"NJCM", "NJ_IFF", "model",     ".nj"),
    (b"POF0", "NJ_IFF", "model",     ".nj"),
    (b"XVMH", "XVM",    "texture",   ".xvm"),
    (b"XVRT", "XVR",    "texture",   ".xvr"),
    (b"PVRT", "PVR",    "texture",   ".pvr"),
    (b"GBIX", "PVR",    "texture",   ".pvr"),
    (b"\x89PNG", "PNG",   "ui",      ".png"),
    (b"OggS", "OGG",    "audio",    ".ogg"),
    # Sega FSB / NMDM chunked anim
    (b"NMDM", "NJM",    "animation", ".njm"),
    (b"NMOT", "NJM",    "animation", ".njm"),
]

# PRS magic detection: PRS has no fixed header, but the first byte is a
# bitmap so the leading literal payload usually starts at offset 1. We
# look for any inner magic in the first ~64 bytes.
PRS_SNIFF_BYTES = 64


def sniff_inner_format(blob: bytes) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return ``(format_id, category, extension)`` for a raw inner blob.

    Direct match on the first 16 bytes of ``blob`` first; if nothing
    matches, treat ``blob`` as PRS-compressed and scan the first 64
    bytes for any known inner magic (PRS leaves literal runs intact, so
    the inner header usually appears verbatim).

    Returns ``(None, None, None)`` if no signature is found.
    """
    if not blob:
        return (None, None, None)
    head = blob[:16]
    for magic, fmt, cat, ext in _INNER_MAGIC:
        if head.startswith(magic):
            return (fmt, cat, ext)
    # PRS scan
    sniff = blob[:PRS_SNIFF_BYTES]
    for magic, fmt, cat, ext in _INNER_MAGIC:
        if magic in sniff:
            return (fmt, cat, ext)
    return (None, None, None)


def _is_prs_compressed(blob: bytes) -> bool:
    """Heuristic: PRS blobs have no header so we infer from inner magic.

    True when the leading bytes do NOT directly match any known inner
    magic but the next 64 bytes contain one — that pattern is
    characteristic of PRS literal-runs in the bitmap encoding.
    """
    if not blob:
        return False
    head = blob[:16]
    for magic, _fmt, _cat, _ext in _INNER_MAGIC:
        if head.startswith(magic):
            return False
    sniff = blob[:PRS_SNIFF_BYTES]
    for magic, _fmt, _cat, _ext in _INNER_MAGIC:
        if magic in sniff:
            return True
    return False


def _afs_meta_descriptor(buf: bytes) -> Optional[tuple[int, int]]:
    """Locate the optional metadata table's ``(offset, size)`` descriptor.

    Sega/SA-Tools AFS archives store the metadata-table descriptor in one
    of two placements (PSO2-Aqua-Library AFS.cs:171-189):

      * ``OffsetEndTable`` - the slot directly AFTER the per-file table
        (``8 + file_count*8``). This is Aqua's default and what our own
        writer emits.
      * ``OffsetBeforeFirstEntry`` - 8 bytes BEFORE the first entry's
        data offset. Used when the end-of-table slot reads zero. Aqua
        falls back to this; without it, an Aqua-written before-first
        archive silently loses every filename.

    Returns the validated ``(meta_off, meta_sz)`` or ``None`` when no
    usable descriptor is present. ``meta_sz`` is synthesised as
    ``file_count * 48`` for the before-first mode (where only the offset
    is meaningful in some writers) but verified against the buffer.
    """
    n = len(buf)
    if n < 8:
        return None
    file_count = struct.unpack_from("<H", buf, 4)[0]
    if file_count == 0:
        return None
    table_end = 8 + file_count * 8

    # Mode 1: descriptor in slot[file_count] (OffsetEndTable).
    if table_end + 8 <= n:
        extra_off, extra_sz = struct.unpack_from("<II", buf, table_end)
        if extra_off != 0 and extra_sz != 0 and extra_off >= table_end + 8 \
                and extra_off + extra_sz <= n:
            return (extra_off, extra_sz)

    # Mode 2: descriptor 8 bytes before the first entry (OffsetBeforeFirstEntry).
    # Read the first entry's data offset, then the (off, sz) pair at
    # first_off - 8. Mirrors Aqua AFS.cs:176-180.
    first_off = struct.unpack_from("<I", buf, 8)[0]
    if first_off >= 16 and first_off - 8 + 8 <= n and first_off - 8 >= table_end:
        meta_off, meta_sz = struct.unpack_from("<II", buf, first_off - 8)
        if meta_off == 0:
            return None
        # Some writers leave the size word as developer data; clamp to the
        # nominal table size and validate against the buffer.
        nominal = file_count * 48
        if meta_sz == 0 or meta_sz > nominal:
            meta_sz = nominal
        if meta_off >= table_end and meta_off + meta_sz <= n:
            return (meta_off, meta_sz)
    return None


def _afs_metadata_table(buf: bytes) -> Optional[list[dict]]:
    """Read the full Sega/SA-Tools metadata table, if present.

    Each entry is 48 bytes (PSO2-Aqua-Library AFS.cs:40-50):

        0x00  char[32]  name (ASCIIZ)
        0x20  u16 x6     DateTime: year, month, day, hour, minute, second
        0x2C  u32        custom_data

    Returns a list of ``{"name", "timestamp", "custom_data"}`` dicts (one
    per archive entry) where ``timestamp`` is the raw 6-tuple
    ``(year, month, day, hour, minute, second)`` exactly as stored (NOT a
    datetime - a 0/0/0 stamp is common and is not a valid datetime), or
    ``None`` when no usable metadata table is present.
    """
    desc = _afs_meta_descriptor(buf)
    if desc is None:
        return None
    meta_off, meta_sz = desc
    file_count = struct.unpack_from("<H", buf, 4)[0]
    out: list[dict] = []
    p = meta_off
    end = meta_off + meta_sz
    while len(out) < file_count and p + 48 <= end:
        name_bytes = bytes(buf[p:p + 32])
        name = name_bytes.split(b"\x00", 1)[0].decode("ascii", errors="replace")
        ts = struct.unpack_from("<6H", buf, p + 32)
        custom = struct.unpack_from("<I", buf, p + 44)[0]
        out.append({"name": name, "timestamp": ts, "custom_data": custom})
        p += 48
    if len(out) != file_count:
        return None
    return out


def _afs_filename_table(buf: bytes) -> Optional[list[str]]:
    """Try to read the optional Sega AFS filename table.

    Thin compatibility wrapper over ``_afs_metadata_table`` returning just
    the per-entry filename strings (the shape ``server.py`` passes back
    into ``write_afs(names=...)``). Supports BOTH descriptor placements
    (OffsetEndTable and OffsetBeforeFirstEntry); PSOBB's shipped archives
    omit the table entirely, so this returns ``None`` for them.

    Returns a list with one filename per archive entry, or ``None`` if
    the table isn't present / unreadable.
    """
    table = _afs_metadata_table(buf)
    if table is None:
        return None
    return [e["name"] for e in table]


# ------------------------------------------------------------------ public

def list_inner_blobs(afs_path: Path) -> list[dict]:
    """List every inner blob in an AFS archive.

    Returns a list of dicts ``{index, name, size, magic_hex,
    inner_format, inner_category, inner_ext, compressed}`` — one row
    per archive slot. ``name`` is sourced from the optional AFS
    filename table when present, else synthesised as
    ``"<archive_stem>_<NNNN>"``.

    Reads the archive but does NOT decompress. Cheap enough to call on
    every manifest rebuild (~6 archives × ~400 entries = 2.4k entries).
    """
    buf = afs_path.read_bytes()
    blobs = afs_mod.parse_afs(buf)
    names = _afs_filename_table(buf)
    stem = afs_path.stem
    out: list[dict] = []
    for i, blob in enumerate(blobs):
        fmt, cat, ext = sniff_inner_format(blob)
        # Synth a 4-digit-prefixed name so alpha sort matches archive
        # order. Append the sniffed format extension so the asset tree
        # can dispatch on suffix the same way it does for top-level files.
        if names and i < len(names) and names[i]:
            display = names[i]
            # Append format extension if the archive's name doesn't
            # already carry one we recognise.
            if "." not in display and ext:
                display = display + ext
        else:
            display = f"{stem}_{i:04d}{ext or ''}"
        compressed = _is_prs_compressed(blob)
        magic_hex = blob[:8].hex()
        out.append({
            "index": i,
            "name": display,
            "size": len(blob),
            "magic_hex": magic_hex,
            "inner_format": fmt,
            "inner_category": cat,
            "inner_ext": ext,
            "compressed": compressed,
        })
    return out


def cache_dir_for(afs_path: Path, root_cache_dir: Path) -> Path:
    """Return the per-archive cache subdir under ``root_cache_dir/afs/``.

    Layout: ``<root_cache_dir>/afs/<archive_stem>/`` (no archive
    extension to avoid filesystem confusion). Created lazily by callers
    that intend to write.
    """
    return root_cache_dir / "afs" / afs_path.stem


def cache_path_for_inner(afs_path: Path, index: int, root_cache_dir: Path,
                         ext: Optional[str] = None) -> Path:
    """Return the cache file path for one inner blob.

    Filename pattern: ``<NNNN>.bin`` (raw bytes) or
    ``<NNNN><ext>`` when an extension is known. Using ``.bin`` keeps the
    fallback path opaque so downstream tools don't mis-classify by
    extension when the sniffer missed.
    """
    sub = cache_dir_for(afs_path, root_cache_dir)
    if ext and ext.startswith("."):
        return sub / f"{index:04d}{ext}"
    return sub / f"{index:04d}.bin"


def _decompress_if_prs(blob: bytes, *, archive_name: str, idx: int,
                      timeout: int = 60) -> tuple[bytes, list[str]]:
    """If ``blob`` smells like PRS, try to decompress; else return as-is.

    Always returns a (bytes, warnings) tuple. PuyoToolsCli failures (not
    installed, timeout, garbage input) are caught and surfaced as
    warnings; the original raw blob is returned in that case so the
    cache writer still has something to commit.
    """
    if not _is_prs_compressed(blob):
        return blob, []
    try:
        from formats.bml import _prs_decompress  # local import: subprocess helper
    except Exception as e:  # pragma: no cover
        return blob, [f"PRS decompress unavailable: {e}"]
    try:
        out = _prs_decompress(blob, timeout=timeout)
        return out, []
    except Exception as e:
        return blob, [f"PRS decompress failed for {archive_name}#{idx}: {e}"]


def materialize_inner(afs_path: Path, index: int, root_cache_dir: Path,
                      *, force: bool = False, decompress: bool = True,
                      timeout: int = 60) -> tuple[Path, dict]:
    """Materialize one inner blob to the on-disk cache; return (path, info).

    ``info`` carries ``{size, inner_format, inner_category, inner_ext,
    compressed, warnings}`` so the caller can update the manifest entry
    without re-sniffing.

    When ``decompress=True`` (default) and the blob looks PRS-compressed,
    we run ``_prs_decompress`` and stash the decompressed bytes; the
    cached file's bytes are then directly usable by downstream parsers.
    When PuyoTools is unavailable we degrade to raw bytes and append a
    warning. ``force=True`` rebuilds the cache file unconditionally.

    Cache invalidation: keyed on ``(file size, mtime)`` of the AFS
    archive — encoded in the parent dir's contents. We rebuild whenever
    the parent stat changes.
    """
    buf = afs_path.read_bytes()
    blobs = afs_mod.parse_afs(buf)
    if index < 0 or index >= len(blobs):
        raise IndexError(f"index {index} out of range (count={len(blobs)})")
    blob = blobs[index]
    fmt, cat, ext = sniff_inner_format(blob)
    out_path = cache_path_for_inner(afs_path, index, root_cache_dir, ext=ext)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # mtime stamp: we write a sibling `.stat` file next to the cache
    # subdir to validate cache freshness across runs. Cheap and works
    # without a manifest of cache state.
    stat_file = out_path.parent / "_archive.stat"
    cur_stat = afs_path.stat()
    cur_key = f"{cur_stat.st_size}-{int(cur_stat.st_mtime)}"
    cached_key = ""
    if stat_file.exists():
        try:
            cached_key = stat_file.read_text(encoding="utf-8").strip()
        except OSError:
            cached_key = ""
    if cached_key != cur_key:
        # Wipe stale cache files for this archive (only same-stem dir).
        for p in out_path.parent.iterdir():
            if p.is_file() and p.name != "_archive.stat":
                try:
                    p.unlink()
                except OSError:
                    pass
        try:
            stat_file.write_text(cur_key, encoding="utf-8")
        except OSError:
            pass

    if not force and out_path.exists() and out_path.stat().st_size > 0:
        info = {
            "size": out_path.stat().st_size,
            "inner_format": fmt,
            "inner_category": cat,
            "inner_ext": ext,
            "compressed": _is_prs_compressed(blob),
            "warnings": [],
            "cached": True,
        }
        return out_path, info

    warnings: list[str] = []
    final_bytes = blob
    if decompress and _is_prs_compressed(blob):
        final_bytes, w = _decompress_if_prs(blob, archive_name=afs_path.name,
                                            idx=index, timeout=timeout)
        warnings.extend(w)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_bytes(final_bytes)
    os.replace(tmp, out_path)
    info = {
        "size": len(final_bytes),
        "inner_format": fmt,
        "inner_category": cat,
        "inner_ext": ext,
        "compressed": _is_prs_compressed(blob),
        "warnings": warnings,
        "cached": False,
    }
    return out_path, info


def iter_afs_archives(root: Path) -> Iterator[Path]:
    """Yield every ``.afs`` file under ``root`` (non-recursive in walk
    sense — the manifest walker already does the recursion; we just
    re-iterate the same set)."""
    for p in root.rglob("*.afs"):
        if p.is_file():
            yield p
