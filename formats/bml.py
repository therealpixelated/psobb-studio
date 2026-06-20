# Ported from MIT-licensed Phantasmal World psolib by Daan Vanden Bosch.
# See LICENSES.md at the editor root for the verbatim MIT block.
#
# Reference (specification only - no code copied; pso-blender is GPL):
#   _modelwork/pso-blender/pso_blender/bml.py
# Reference (Phantasmal World, MIT):
#   psolib/src/commonMain/kotlin/world/phantasmal/psolib/fileFormats/Bml.kt
#
# BML (Binary Model Library) - the container that holds the bulk of
# PSOBB's models + their inline texture archives. ~365 BMLs ship in
# the install. Layout (all little-endian):
#
#     +---------------------------------------------+
#     | 0x00 Header (0x40 bytes)                    |
#     |   u32 unk1                (always 0)        |
#     |   u32 file_count          (entries below)   |
#     |   u32 magic               (== 0x150)        |
#     |     (low byte 0x50='P', high byte 0x01 are  |
#     |      NOT independent compression/flag bytes)|
#     |   ... 13 * u32 unkN  (padding to 0x40)      |
#     +---------------------------------------------+
#     | 0x40 File table (file_count * 0x40 bytes)   |
#     |   per entry:                                |
#     |     32 bytes : NUL-padded ASCII name        |
#     |     u32      : compressed_size              |
#     |     u32      : unk                          |
#     |     u32      : decompressed_size            |
#     |     u32      : textures_compressed_size     |
#     |     u32      : textures_decompressed_size   |
#     |     u32 * 3  : unks                         |
#     +---------------------------------------------+
#     | align_up(0x40 + file_count*0x40, 0x800)     |
#     | = 0x800 for any reasonable count            |
#     +---------------------------------------------+
#     | File payloads (per entry, in table order):  |
#     |   PRS-compressed inner blob (NJ / XJ / NJM) |
#     |     padded to file_alignment                |
#     |   if textures_compressed_size > 0:          |
#     |     PRS-compressed XVM archive              |
#     |     padded to file_alignment                |
#     +---------------------------------------------+
#
# **Offset resolution (reader):** the reader does NOT assume a single
# global alignment. It resolves each payload's start with a faithful
# port of the C# oracle's SeekPadding (BMLUtil.cs:362-377): the first
# payload begins at the table end rounded up to 0x800, and after each
# payload the cursor is rounded up to 0x10 then advanced 0x10 at a time
# over any all-zero dword block until real data appears. This handles
# the 0x20-aligned (705/728) and 0x800-aligned (the 23 ``pl[A-Z]nj.bml``)
# layouts uniformly, with no whole-archive alignment guess and no
# total-size match — and stays correct for trailing-padded / mixed /
# out-of-tree-edited BMLs. Verified to reproduce every inner + texture
# offset across all 728 shipped Ephinea + PSOBB.IO BMLs.
#
# **file_alignment (packer only):** ``pack_bml`` lays payloads out from a
# single per-archive alignment (``_align_up(len, file_alignment)``), so
# the round-trip path classifies that alignment via the cumulative-end
# test (0x20 if the padded sum hits the file size, else 0x800). The 23
# ``pl[A-Z]nj.bml`` use 0x800 (their header magic high-byte still reads
# 0x01, but no entry carries a per-entry texture); the other 705 use
# 0x20. This is intentionally separate from the reader's SeekPadding
# resolver — see ``_classify_file_alignment``.
#
# Inner PRS decompression uses the in-process ``formats.prs`` decoder by
# default; the legacy PuyoToolsCli subprocess pipeline is retained behind
# the ``PSO_USE_PUYOTOOLSCLI=1`` env var as a one-release-cycle
# fallback. Switching to the in-process decoder eliminates the
# ~30-50 ms × N-inner-files fork+exec cost on Windows that dominated
# cold model-open time.
#
# Memoisation: callers that read the same BML inner repeatedly (notably
# the manifest walker — it touches every BML once per cold load and the
# tile cache fills lazily on first asset open) can use
# ``decompress_prs_cached(path, mtime, name) -> bytes`` which is backed
# by an LRU capped at 64 MB. The cache key includes the parent file's
# mtime so an upstream rebuild invalidates entries automatically.
"""Pure-Python reader and writer for PSOBB's Binary Model Library container."""
from __future__ import annotations

import logging
import os
import struct
import subprocess
import sys
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Sequence

from formats import prs as _prs

log = logging.getLogger("psobb_editor.bml")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEADER_SIZE = 0x40
FILE_ENTRY_SIZE = 0x40
NAME_FIELD_SIZE = 32
DATA_ALIGNMENT_NO_TEX = 0x800
DATA_ALIGNMENT_HAS_TEX = 0x20
TABLE_ALIGNMENT = 0x800

COMPRESSION_NONE = 0
COMPRESSION_PRS = 0x50  # ord('P')

# The 4 bytes at +0x08 are a single little-endian u32 *magic number*,
# not a (compression u8, has_textures u8) pair. The C# oracle
# (BMLUtil.cs:20 `public int magicNumber; //0x150`) reads them as one
# field, and it is 0x150 in 728/728 shipped BMLs (both Ephinea and
# PSOBB.IO). Our historical `compression_type = buf[8]` and
# `has_textures = buf[9]` were just the low/high bytes of this constant:
# byte8 == 0x50 ('P') and byte9 == 0x01 in every file, so the
# "compression" byte coincidentally looked like ord('P')=PRS and the
# "has_textures" flag was always 1. They carry no independent
# information. We keep COMPRESSION_PRS/COMPRESSION_NONE as the public
# `compression` knob for the packer/round-trip API (other modules import
# them), but the *reader* no longer treats byte +0x08 as a real
# compression discriminator — every shipped inner blob is PRS-compressed.
BML_MAGIC = 0x150

# Sanity caps. Real PSOBB BMLs hold tens of files, not millions; the
# largest in the install is bm_boss3_volopt.bml at 42 entries.
MAX_FILE_COUNT = 0xFFFF

# Path to PuyoToolsCli (used only by extract_* helpers, not parse_bml).
# Resolved lazily so importing this module never requires the binary.
_PUYO_DEFAULT = Path(r"C:/Tools/re/upscale-lab/tools/puyotools/PuyoToolsCli.exe")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class BmlEntry:
    """One file inside a BML container.

    Attributes:
        name: NUL-terminated ASCII name from the directory (e.g.
              "bm_obj_ep4_boss09_core01.nj"). Trailing NULs stripped.
        size_compressed: bytes occupied by the inner file in the
              container (PRS-compressed if compression_type==PRS).
        size_decompressed: byte length after PRS decompress; equals
              size_compressed when compression_type==NONE.
        has_texture: True if this entry has a sibling XVM payload
              following the inner file in the container.
        tex_size_compressed: bytes occupied by the inline XVM (always
              PRS-compressed, irrespective of compression_type), or 0
              if the entry has no texture.
        offset: absolute byte offset of the inner file's first byte
              inside the BML buffer. The texture (if present) starts
              at offset + align_up(size_compressed, file_alignment).
    """
    name: str
    size_compressed: int
    size_decompressed: int
    has_texture: bool
    tex_size_compressed: int
    offset: int


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _align_up(v: int, a: int) -> int:
    """Round v up to the next multiple of a (a is a power of two)."""
    return (v + a - 1) & ~(a - 1)


def _decode_name(buf: bytes) -> str:
    """Decode a 32-byte NUL-padded ASCII name field; strip trailing NULs."""
    return buf.split(b"\x00", 1)[0].decode("ascii", errors="replace")


def _seek_padding(mv: memoryview, offset: int) -> int:
    """Skip inter-payload padding exactly like the C# oracle's SeekPadding.

    Faithful port of BMLUtil.cs:362-377. Starting from ``offset`` (the
    byte just past the previous payload):

      1. Round the cursor up to the next 0x10 boundary.
      2. While the dword at the cursor is all-zero (and 4 bytes are
         readable), skip forward 0x10 at a time.

    This resolves the next payload's start with no whole-archive
    alignment assumption and no total-file-size match — it works for
    both the 0x20-aligned (341/364) and 0x800-aligned (the 23
    ``pl[A-Z]nj.bml``) layouts, and stays correct for trailing-padded,
    mixed-alignment, or out-of-tree-edited BMLs where the old
    cumulative-end heuristic would fall back to the (always-1) header
    flag. Verified to reproduce every inner and texture offset across
    all 728 shipped Ephinea + PSOBB.IO BMLs (5240 inner + 1320 texture
    payloads, 0 mismatches).
    """
    n = len(mv)
    # 1. Align cursor up to 0x10.
    if offset & 0xF:
        offset = (offset + 0xF) & ~0xF
    # 2. Skip all-zero 0x10 blocks until real data (or EOF).
    while offset + 4 <= n and struct.unpack_from("<i", mv, offset)[0] == 0:
        offset += 0x10
    return offset


def _walk_offsets(mv: memoryview, metas: Sequence[tuple]) -> List[tuple]:
    """Resolve absolute (inner_off, tex_off) for every entry via SeekPadding.

    ``metas`` is a list of ``(name, comp_size, decomp_size, tex_comp_size)``
    tuples in archive order (as built by ``parse_bml``). Returns a list of
    ``(inner_off, tex_off_or_None)`` in the same order.

    Mirrors the C# oracle's offset loop (BMLUtil.cs:87-140): the first
    payload starts at the table end rounded up to the next 0x800
    boundary; thereafter each payload's end is advanced past zero-padding
    by ``_seek_padding``. Replaces the previous global-file_alignment
    cumulative-end heuristic, which assumed one alignment for the whole
    archive and an exact file-size match.
    """
    n = len(mv)
    file_count = len(metas)
    table_end = HEADER_SIZE + file_count * FILE_ENTRY_SIZE
    cursor = _align_up(table_end, TABLE_ALIGNMENT)
    out: List[tuple] = []
    for (name, comp_size, _decomp, tex_comp_size) in metas:
        inner_off = cursor
        if inner_off + comp_size > n:
            raise ValueError(
                f"parse_bml: entry ({name!r}) extends to "
                f"0x{inner_off + comp_size:x} but buffer is only 0x{n:x}"
            )
        cursor = _seek_padding(mv, inner_off + comp_size)
        tex_off: Optional[int] = None
        if tex_comp_size > 0:
            tex_off = cursor
            if tex_off + tex_comp_size > n:
                raise ValueError(
                    f"parse_bml: entry ({name!r}) texture extends to "
                    f"0x{tex_off + tex_comp_size:x} but buffer is only 0x{n:x}"
                )
            cursor = _seek_padding(mv, tex_off + tex_comp_size)
        out.append((inner_off, tex_off))
    return out


def parse_bml(buf: bytes) -> List[BmlEntry]:
    """Parse a BML directory and return one BmlEntry per inner file.

    Does NOT decompress. The returned entries describe where each inner
    file lives inside `buf`; use extract_bml() / extract_bml_texture()
    to actually fetch the bytes.

    Args:
        buf: full BML file bytes.

    Returns:
        Ordered list of BmlEntry. Each entry's `offset` is the
        absolute byte offset of its inner file inside `buf`.

    Raises:
        ValueError: malformed header / table / payload region. Includes:
            - non-bytes input
            - truncated header (< 0x40 bytes)
            - file_count beyond sanity cap
            - file table truncated
            - declared payload extends past the buffer end
    """
    if not isinstance(buf, (bytes, bytearray, memoryview)):
        raise ValueError("parse_bml: input must be bytes-like")
    mv = memoryview(buf)
    n = len(mv)

    if n < HEADER_SIZE:
        raise ValueError(
            f"parse_bml: truncated header (need {HEADER_SIZE} bytes, have {n})"
        )

    # Header: u32 unk1, u32 file_count, u32 magic (== 0x150), ...
    # The 4 bytes at +0x08 are a single magic number, NOT a
    # (compression u8, has_textures u8) pair — see BML_MAGIC. We read it
    # for documentation/validation but it does not gate the parse: every
    # shipped inner blob is PRS-compressed and the high byte (the old
    # "has_textures" flag) is constant 0x01, so it carries no signal.
    _unk1, file_count, _magic = struct.unpack_from("<III", mv, 0)
    if file_count == 0 or file_count > MAX_FILE_COUNT:
        raise ValueError(f"parse_bml: invalid file_count {file_count}")

    table_end = HEADER_SIZE + file_count * FILE_ENTRY_SIZE
    if table_end > n:
        raise ValueError(
            f"parse_bml: file table needs 0x{table_end:x} bytes but buffer "
            f"is only 0x{n:x}"
        )

    # File payloads start at the next 0x800 boundary after the table.
    # pso-blender computes `align_up(file_count * 0x40, 0x800)` because
    # the header is itself exactly 0x40 (one entry size); we make the
    # full expression explicit here to avoid the off-by-one trap.
    data_start = _align_up(table_end, TABLE_ALIGNMENT)
    if data_start > n:
        raise ValueError(
            f"parse_bml: data region starts at 0x{data_start:x} but buffer "
            f"is only 0x{n:x}"
        )

    # Pass 1: read every entry's metadata (name + sizes). Offsets are
    # resolved separately by the SeekPadding walk below.
    metas: List[tuple] = []
    for i in range(file_count):
        ent_off = HEADER_SIZE + i * FILE_ENTRY_SIZE
        name_bytes = bytes(mv[ent_off:ent_off + NAME_FIELD_SIZE])
        meta_off = ent_off + NAME_FIELD_SIZE
        (
            comp_size,
            _unk_a,
            decomp_size,
            tex_comp_size,
            _tex_decomp_size,
            _unk_b,
            _unk_c,
            _unk_d,
        ) = struct.unpack_from("<8I", mv, meta_off)
        name = _decode_name(name_bytes)
        if not name:
            raise ValueError(f"parse_bml: entry {i} has empty name")
        if comp_size > n:
            raise ValueError(
                f"parse_bml: entry {i} ({name!r}) compressed_size 0x{comp_size:x} "
                f"exceeds buffer 0x{n:x}"
            )
        metas.append((name, comp_size, decomp_size, tex_comp_size))

    # Pass 2: resolve absolute payload offsets via the SeekPadding scan
    # (faithful port of the C# oracle), not a fixed global alignment.
    # ``_walk_offsets`` rounds the first payload to the 0x800 table
    # boundary, then advances past zero-padding 0x10 at a time after each
    # payload — handling the 0x20-aligned, 0x800-aligned, and any
    # edited/mixed layout uniformly without a whole-archive alignment
    # assumption or a total-size match.
    offsets = _walk_offsets(mv, metas)

    entries: List[BmlEntry] = []
    for (name, comp_size, decomp_size, tex_comp_size), (inner_off, _tex_off) in zip(
        metas, offsets
    ):
        entries.append(
            BmlEntry(
                name=name,
                size_compressed=comp_size,
                size_decompressed=decomp_size,
                has_texture=tex_comp_size > 0,
                tex_size_compressed=tex_comp_size,
                offset=inner_off,
            )
        )

    return entries


# ---------------------------------------------------------------------------
# PRS decompression
# ---------------------------------------------------------------------------
# Two paths exist:
#   1. In-process Python decoder via ``formats.prs.decompress`` (default).
#   2. PuyoToolsCli subprocess (legacy; activate via PSO_USE_PUYOTOOLSCLI=1).
#
# Path 1 is 5-10× faster than fork+exec on a typical inner blob (~5 ms
# vs ~40 ms on Windows for ~50 KB compressed input), and avoids the
# brittle "tool not found" failure mode entirely. Path 2 stays around
# for one release cycle as a safety valve in case the in-process
# decoder turns out to mishandle some edge case that PuyoToolsCli
# tolerates.
def _puyo_path() -> Path:
    """Resolve the PuyoToolsCli path - allow override via env."""
    override = os.environ.get("PSO_PUYOTOOLS")
    if override:
        return Path(override).resolve()
    return _PUYO_DEFAULT.resolve()


def _use_puyotoolscli() -> bool:
    """Return True iff the user opted into the legacy subprocess path.

    Triggered by ``PSO_USE_PUYOTOOLSCLI=1`` (or any truthy value). Default
    is False — the in-process decoder.
    """
    val = os.environ.get("PSO_USE_PUYOTOOLSCLI", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _prs_decompress_inproc(blob: bytes, *, tolerant: bool = False) -> bytes:
    """Decompress PRS in-process. Fast path; default.

    ``tolerant=True`` enables Sega's "stub XVM" recovery: if the PRS
    stream lacks a proper end marker (truncation mid-instruction or
    backreference past end-of-output), return whatever decompressed
    bytes we successfully produced instead of raising. The 10 affected
    BMLs in PSOBB.IO ship a compressed XVM whose decompressed payload IS
    a valid XVMH archive — only the trailing PRS instruction goes off
    the rails. Strict mode (default) preserves the historical behavior:
    truncated streams raise ``ValueError``.
    """
    if not blob:
        raise ValueError("PRS decompress: empty input")
    return _prs.decompress(blob, tolerant=tolerant)


def _prs_decompress_subprocess(blob: bytes, *, tool: Optional[Path] = None,
                               timeout: int = 60) -> bytes:
    """Decompress PRS via PuyoToolsCli subprocess (legacy fallback).

    Writes ``blob`` to a temp file, asks PuyoToolsCli to decompress it
    (with --overwrite so we don't have to reason about its naming
    convention), and returns the resulting bytes. Cleans up regardless
    of outcome.
    """
    if not blob:
        raise ValueError("PRS decompress: empty input")
    puyo = tool or _puyo_path()
    if not puyo.exists():
        raise RuntimeError(f"PuyoToolsCli not found at {puyo}")

    with tempfile.TemporaryDirectory(prefix="bml_prs_") as tdir:
        tdir_p = Path(tdir)
        in_path = tdir_p / "blob.prs"
        in_path.write_bytes(blob)
        # PuyoToolsCli with --overwrite replaces the input file with its
        # decompressed bytes (same name, no .prs suffix munging).
        cmd = [
            str(puyo),
            "compression",
            "decompress",
            "--overwrite",
            "-i",
            in_path.name,
        ]
        try:
            r = subprocess.run(
                cmd,
                cwd=tdir_p,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"PRS decompress timeout after {timeout}s") from e
        if r.returncode != 0:
            raise RuntimeError(
                f"PRS decompress failed (rc={r.returncode}): "
                f"stdout={r.stdout[-500:]!r} stderr={r.stderr[-500:]!r}"
            )
        # PuyoToolsCli leaves the decompressed bytes at the same path.
        if not in_path.exists():
            raise RuntimeError("PRS decompress: PuyoToolsCli left no output")
        return in_path.read_bytes()


def _prs_decompress(blob: bytes, *, tool: Optional[Path] = None,
                    timeout: int = 60, tolerant: bool = False) -> bytes:
    """Decompress a single PRS blob.

    Routes to the in-process decoder by default; the legacy subprocess
    pipeline is selected by setting ``PSO_USE_PUYOTOOLSCLI=1`` in the
    environment. ``tool`` and ``timeout`` are only used by the legacy
    path; the in-process decoder is fast enough that timeouts are
    irrelevant (a 1 MB blob takes <50 ms in pure Python).

    ``tolerant`` is honored only by the in-process decoder (Phantasmal
    PRS port) — the subprocess fallback inherits PuyoToolsCli's
    behavior, which silently produces the partial output as well.

    Raises:
        ValueError: empty input or malformed PRS stream (in-process,
            strict mode only).
        RuntimeError: subprocess unavailable / failed / timed out
            (subprocess path only).
    """
    if _use_puyotoolscli():
        return _prs_decompress_subprocess(blob, tool=tool, timeout=timeout)
    return _prs_decompress_inproc(blob, tolerant=tolerant)


# ---------------------------------------------------------------------------
# LRU cache for repeated reads of the same inner blob
# ---------------------------------------------------------------------------
# The manifest walker, asset tree, and the model viewer can all open the
# same BML inner multiple times in a single cold session (the manifest
# enumerates every entry; the user then opens the model; the texture
# panel re-asks for binding metadata; the variant detector probes for
# sibling NJTLs). Cache decompressed inner blobs so we pay the
# decompression cost once.
#
# Cache key: ``(absolute_path_str, mtime_ns, inner_name)``. The mtime
# ensures stale entries auto-evict when an upstream rebuild touches the
# parent file.
_PRS_INNER_CACHE: "OrderedDict[tuple, bytes]" = OrderedDict()
_PRS_INNER_CACHE_LOCK = Lock()
PRS_INNER_CACHE_MAX_BYTES = 64 * 1024 * 1024  # 64 MB
# Live byte counter — refreshed on insert/evict so we don't sum the
# whole dict every call (sum is O(N) over keys; the cache can hold
# hundreds of entries when the user browses dragon-class BMLs).
_PRS_INNER_CACHE_BYTES = 0


def decompress_prs_cached(path: Path, mtime_ns: int, name: str,
                          raw_provider) -> bytes:
    """Cached PRS decompress keyed by ``(path, mtime_ns, name)``.

    ``raw_provider`` is a zero-arg callable that returns the raw
    PRS-compressed bytes; we only call it on a cache miss so callers
    don't have to splice the raw blob from a memory-mapped buffer
    redundantly.

    Cache invalidation is automatic: once the parent file's mtime
    changes, the new key won't match any entry and the cache will
    refill. Old entries age out via LRU.

    Thread-safe: serialised behind a single lock. The decompression
    work itself is intentionally not under the lock — Python's
    ``formats.prs.decompress`` is pure-Python and wouldn't benefit from
    the GIL release pattern (radare2/numpy aren't in play here), so we
    just protect the dict mutations.
    """
    global _PRS_INNER_CACHE_BYTES
    key = (str(path), int(mtime_ns), name)
    with _PRS_INNER_CACHE_LOCK:
        cached = _PRS_INNER_CACHE.get(key)
        if cached is not None:
            _PRS_INNER_CACHE.move_to_end(key)
            return cached

    # Miss: do the work outside the lock.
    raw = raw_provider()
    out = _prs_decompress(raw)
    with _PRS_INNER_CACHE_LOCK:
        # Re-check in case of a race — another caller may have populated
        # the same key while we were decompressing.
        cached = _PRS_INNER_CACHE.get(key)
        if cached is not None:
            _PRS_INNER_CACHE.move_to_end(key)
            return cached
        _PRS_INNER_CACHE[key] = out
        _PRS_INNER_CACHE_BYTES += len(out)
        # Evict LRU until we fit. Bound the loop defensively in case of
        # a single oversize entry — we still allow one entry through
        # even when it exceeds the budget alone (so a 70 MB inner can
        # still be served), but we keep flushing other entries.
        while (_PRS_INNER_CACHE_BYTES > PRS_INNER_CACHE_MAX_BYTES
               and len(_PRS_INNER_CACHE) > 1):
            try:
                _evicted_key, _evicted_val = _PRS_INNER_CACHE.popitem(last=False)
            except KeyError:
                break
            _PRS_INNER_CACHE_BYTES -= len(_evicted_val)
    return out


def cache_stats() -> Dict[str, int]:
    """Return ``{entries, bytes, max_bytes}`` for the PRS inner-blob cache.

    Used by /api/health to surface cache utilisation; helps the user
    gauge whether the LRU is hot (working set fits) or thrashing.
    """
    with _PRS_INNER_CACHE_LOCK:
        return {
            "entries": len(_PRS_INNER_CACHE),
            "bytes": _PRS_INNER_CACHE_BYTES,
            "max_bytes": PRS_INNER_CACHE_MAX_BYTES,
        }


def cache_clear() -> None:
    """Drop the entire PRS inner-blob cache. Test-only; not wired to a route."""
    global _PRS_INNER_CACHE_BYTES
    with _PRS_INNER_CACHE_LOCK:
        _PRS_INNER_CACHE.clear()
        _PRS_INNER_CACHE_BYTES = 0


def extract_bml(
    buf: bytes,
    *,
    tool: Optional[Path] = None,
    timeout: int = 60,
) -> Dict[str, bytes]:
    """Extract every inner file from a BML, PRS-decompressing in place.

    Args:
        buf: full BML file bytes.
        tool: optional PuyoToolsCli override; defaults to the standard
              C:/Tools/re/upscale-lab/tools/puyotools/PuyoToolsCli.exe.
        timeout: per-entry subprocess timeout in seconds.

    Returns:
        Dict mapping entry name -> decompressed inner bytes. The dict
        is keyed exactly as the BML stores names (e.g.
        "bm_obj_ep4_boss09_core01.nj"). Texture archives are NOT
        included here - call extract_bml_texture() for those.

    Raises:
        ValueError: bad BML header (propagated from parse_bml).
        RuntimeError: PuyoToolsCli unavailable / failed / timed out.
    """
    entries = parse_bml(buf)
    out: Dict[str, bytes] = {}
    mv = memoryview(buf)
    for ent in entries:
        slice_start = ent.offset
        slice_end = slice_start + ent.size_compressed
        raw = bytes(mv[slice_start:slice_end])
        if ent.size_compressed == ent.size_decompressed and len(raw) >= 4 and raw[:4] != b"\x00\x00\x00\x00":
            # Heuristic: when comp==decomp and the leading bytes look
            # plausibly uncompressed (PSO IFF starts with NJCM/NMDM
            # which are all printable), skip the subprocess. Real PRS
            # never starts with all four zero bytes.
            # In practice every shipped BML is PRS and comp != decomp,
            # so this branch is rarely hit. Falls through to subprocess
            # otherwise to be safe.
            pass
        out[ent.name] = _prs_decompress(raw, tool=tool, timeout=timeout)
    return out


def extract_bml_texture(
    buf: bytes,
    name: str,
    *,
    tool: Optional[Path] = None,
    timeout: int = 60,
) -> Optional[bytes]:
    """Extract just the texture portion of one named entry as raw XVM.

    The texture (if present) is a PRS-compressed XVM archive that
    follows the entry's inner file in the BML buffer. We always
    PRS-decompress it - the BML's header `compression_type` byte does
    not apply to textures (per pso-blender source: "Texture archive is
    always PRS compressed").

    Args:
        buf: full BML file bytes.
        name: entry name (e.g. "bm_obj_ep4_boss09_core01.nj").
        tool: optional PuyoToolsCli override.
        timeout: subprocess timeout in seconds.

    Returns:
        Decompressed XVM bytes (begins with b"XVMH"), or None if the
        named entry exists but has no texture.

    Raises:
        ValueError: bad BML header, OR the named entry is not in the
                    archive.
        RuntimeError: PuyoToolsCli unavailable / failed / timed out.
    """
    entries = parse_bml(buf)
    mv_all = memoryview(buf)
    # Resolve every payload's exact offset once via the SeekPadding walk
    # (the same resolver parse_bml uses). This gives the texture offset
    # directly instead of re-deriving it from a hardcoded 0x20 alignment,
    # which was correct only because every shipped tex-bearing BML is
    # 0x20-aligned — SeekPadding stays correct for any alignment.
    metas = [
        (e.name, e.size_compressed, e.size_decompressed, e.tex_size_compressed)
        for e in entries
    ]
    offsets = _walk_offsets(mv_all, metas)
    for ent, (_inner_off, tex_off) in zip(entries, offsets):
        if ent.name != name:
            continue
        if not ent.has_texture or ent.tex_size_compressed == 0 or tex_off is None:
            return None
        tex_end = tex_off + ent.tex_size_compressed
        if tex_end > len(buf):
            raise ValueError(
                f"extract_bml_texture: entry {name!r} texture extends to "
                f"0x{tex_end:x} but buffer is only 0x{len(buf):x}"
            )
        raw = bytes(memoryview(buf)[tex_off:tex_end])
        # Tolerant-mode PRS recovery: 10 of PSOBB.IO's BMLs ship a "stub
        # XVM" payload whose PRS stream is unterminated — the
        # decompressed prefix is a valid XVMH archive header (the
        # runtime tolerates this because its texture allocator never
        # reaches end-of-stream). Try strict first; on EOF/truncation,
        # retry with tolerant=True and validate XVMH magic before
        # returning. This replaces an earlier "fail closed" path that
        # dropped 10 entire texture archives, leaving the affected
        # models (Vol Opt ceiling shards, Morfos walls, light/cave
        # rocks) untextured.
        try:
            return _prs_decompress(raw, tool=tool, timeout=timeout)
        except (ValueError, RuntimeError) as e:
            try:
                partial = _prs_decompress(
                    raw, tool=tool, timeout=timeout, tolerant=True,
                )
            except Exception:
                raise e
            # Only return the partial if it produced enough to be a real
            # XVMH archive (the magic and size header live in the first
            # 8 bytes; a sane archive is at least the 0x40-byte XVMH
            # block + at least one XVR record).
            if len(partial) >= 0x40 and partial[:4] == b"XVMH":
                log.info(
                    "extract_bml_texture: recovered %s/%s via tolerant PRS "
                    "(%d bytes from corrupt %d-byte stream)",
                    "<bml>", name, len(partial), len(raw),
                )
                return partial
            raise
    raise ValueError(f"extract_bml_texture: no entry named {name!r}")


# ---------------------------------------------------------------------------
# Packer (writer)
# ---------------------------------------------------------------------------
# Reverse-engineered on-disk layout (verified against all 364 shipped BMLs):
#
#   Header (0x40 bytes):
#     +0x00  u32  unk1            = 0 in every shipped BML
#     +0x04  u32  file_count
#     +0x08  u8   compression     = 0x50 ('P') or 0
#     +0x09  u8   has_textures    = 0 or 1
#     +0x0a  u16  unk2            = 0 in every shipped BML
#     +0x0c  13 * u32 unkN        = 0 in every shipped BML (52 bytes pad)
#
#   File table at +0x40 (file_count * 0x40 bytes), per entry:
#     +0x00  32 bytes  name (NUL-padded; bytes after first NUL also NUL)
#     +0x20  u32       compressed_size
#     +0x24  u32       unk_a       — non-zero in 1934/26500 entries; preserved
#                                    verbatim from input (semantics unclear,
#                                    looks like a per-archive global texture
#                                    counter on some BMLs but not all)
#     +0x28  u32       decompressed_size  (== compressed_size when not PRS)
#     +0x2c  u32       texture_compressed_size       (0 if no texture)
#     +0x30  u32       texture_decompressed_size     (0 if no texture)
#     +0x34  u32       unk_b       = 0 in every shipped BML
#     +0x38  u32       unk_c       = 0 in every shipped BML
#     +0x3c  u32       unk_d       = 0 in every shipped BML
#
#   Padding from end of table up to 0x800.
#
#   Payloads (in table order). Each entry's inner blob is followed by
#   its texture (if any). All blobs are PRS-compressed when
#   compression==0x50 (always true in the shipped corpus). Each blob —
#   inner or texture — is padded to ``file_alignment`` (0x20 OR 0x800).
#
#   Alignment classifier (verified empirical):
#     - 341/364 shipped BMLs use 0x20.
#     - 23/364 use 0x800. All 23 are the player-class ``pl[A-Z]nj.bml``
#       set: their header sets has_textures=1 but every entry has
#       texture_compressed_size==0. The on-disk layout uses 0x800.
#     - The discriminator that closes cleanly is the cumulative-end
#       sum: pick 0x20 if it matches the file size, else 0x800. The
#       reader does this; the packer can simply pick by examining the
#       input — alignment is 0x20 iff at least one entry has a non-zero
#       texture, otherwise 0x800. This produces non-lying headers.
#
#   ``BmlPackEntry.is_compressed`` flag avoids double-compression on
#   round-trip. When True the bytes are stored verbatim and we trust the
#   caller's ``decompressed_size``. When False the packer PRS-encodes
#   on the way in and computes ``decompressed_size`` automatically.

@dataclass
class BmlPackEntry:
    """One entry to feed to ``pack_bml``.

    Attributes:
        name: entry name (max 32 ASCII bytes incl. NUL terminator).
        data: inner blob bytes. If ``is_compressed`` is True these are
              treated as already-PRS-encoded; otherwise the packer
              compresses them with ``formats.prs.compress`` (or
              ``compress_optimal`` per ``optimal``).
        decompressed_size: REQUIRED when ``is_compressed`` is True;
              ignored otherwise (the packer computes it from
              ``len(data)``).
        is_compressed: True when ``data`` is already PRS bytes (used to
              preserve byte-exact round-trip; avoids re-encoding loss).
        texture_data: optional XVM bytes. Same compression flag as
              ``is_compressed`` controls. When None, the entry has no
              texture.
        texture_decompressed_size: REQUIRED when ``texture_data`` is
              compressed; ignored otherwise.
        texture_is_compressed: True when ``texture_data`` is already PRS.
        unk_a: per-entry unknown u32 at offset +0x24 in the entry record
              (preserved as-is from a parsed BML for round-trip; defaults
              to 0 for fresh entries). Looks like a per-archive global
              counter on a subset of BMLs, but the meaning is not used by
              any reader we've found, so just preserve.
        unk_b/unk_c/unk_d: u32s at +0x34, +0x38, +0x3c. Always 0 in the
              shipped corpus; expose for completeness.
    """
    name: str
    data: bytes
    decompressed_size: int = 0
    is_compressed: bool = False
    texture_data: Optional[bytes] = None
    texture_decompressed_size: int = 0
    texture_is_compressed: bool = False
    unk_a: int = 0
    unk_b: int = 0
    unk_c: int = 0
    unk_d: int = 0


def _encode_name(name: str) -> bytes:
    """Encode a 32-byte NUL-padded ASCII name field."""
    raw = name.encode("ascii")
    if len(raw) > NAME_FIELD_SIZE:
        raise ValueError(
            f"BML name {name!r} exceeds 32 bytes (got {len(raw)})"
        )
    return raw + b"\x00" * (NAME_FIELD_SIZE - len(raw))


def _classify_alignment(entries: Sequence[BmlPackEntry]) -> int:
    """Pick file_alignment based on whether any entry carries a texture.

    Mirrors the empirical truth in the shipped corpus:
      - any non-zero texture in the archive -> 0x20
      - otherwise -> 0x800

    Produces the "non-lying header" the spec requires: ``has_textures``
    flag in the packed header is set iff at least one entry actually
    has a texture, which exactly matches the alignment we choose.
    """
    for ent in entries:
        if ent.texture_data is not None and len(ent.texture_data) > 0:
            return DATA_ALIGNMENT_HAS_TEX
    return DATA_ALIGNMENT_NO_TEX


def pack_bml(
    entries: Sequence[BmlPackEntry],
    *,
    compression: int = COMPRESSION_PRS,
    optimal: bool = False,
    file_alignment: Optional[int] = None,
    has_textures_override: Optional[bool] = None,
) -> bytes:
    """Serialize a list of entries as a PSOBB BML container.

    Args:
        entries: ordered list of ``BmlPackEntry``. At least one required.
        compression: header byte at +0x08. Default 0x50 (PRS) — every
              shipped BML uses this. Pass ``COMPRESSION_NONE`` (0) for
              an uncompressed container; in that case ``is_compressed``
              on each entry must be False and the data is stored raw.
              Texture archives are ALWAYS PRS-compressed regardless of
              this byte (matches pso-blender + game behavior).
        optimal: when True, use ``prs.compress_optimal`` (slower, ~1-3%
              smaller) instead of ``prs.compress``. Only applies to
              entries with ``is_compressed=False``.
        file_alignment: explicit alignment override. Defaults to
              ``_classify_alignment(entries)`` which mirrors the shipped
              corpus's pattern. Pass ``DATA_ALIGNMENT_NO_TEX`` (0x800)
              or ``DATA_ALIGNMENT_HAS_TEX`` (0x20) to force.
        has_textures_override: explicit override for the header byte
              at +0x09. Defaults to (alignment==0x20). Used by the
              round-trip path to preserve the lying-flag set on the
              23 player NJ archives, where the parser sees has_tex=1
              but the alignment is 0x800. WITHOUT this override the
              packer always emits non-lying headers.

    Returns:
        Full BML bytes ready to write to disk.

    Raises:
        ValueError: empty entries list, name too long, missing
              decompressed_size for pre-compressed inputs, etc.
    """
    if not isinstance(entries, (list, tuple)):
        raise ValueError("pack_bml: entries must be a list/tuple")
    n = len(entries)
    if n == 0:
        raise ValueError("pack_bml: at least one entry required")
    if n > MAX_FILE_COUNT:
        raise ValueError(f"pack_bml: too many entries ({n} > {MAX_FILE_COUNT})")
    if compression not in (COMPRESSION_NONE, COMPRESSION_PRS):
        raise ValueError(
            f"pack_bml: compression must be 0x00 or 0x50, got 0x{compression:02x}"
        )

    # Resolve alignment + has_textures bit.
    if file_alignment is None:
        file_alignment = _classify_alignment(entries)
    elif file_alignment not in (DATA_ALIGNMENT_NO_TEX, DATA_ALIGNMENT_HAS_TEX):
        raise ValueError(
            f"pack_bml: file_alignment must be 0x20 or 0x800, got 0x{file_alignment:x}"
        )
    if has_textures_override is None:
        has_textures = any(
            ent.texture_data is not None and len(ent.texture_data) > 0
            for ent in entries
        )
    else:
        has_textures = bool(has_textures_override)

    # Resolve PRS encoder.
    encode_prs = _prs.compress_optimal if optimal else _prs.compress

    # Pass 1: compress (or pass-through) each blob and gather sizes.
    blobs: List[bytes] = []
    decompressed_sizes: List[int] = []
    tex_blobs: List[Optional[bytes]] = []
    tex_decompressed_sizes: List[int] = []

    for i, ent in enumerate(entries):
        if not isinstance(ent, BmlPackEntry):
            raise ValueError(
                f"pack_bml: entries[{i}] is {type(ent).__name__}, "
                f"expected BmlPackEntry"
            )
        if not isinstance(ent.data, (bytes, bytearray, memoryview)):
            raise ValueError(
                f"pack_bml: entries[{i}].data is {type(ent.data).__name__}, "
                f"expected bytes-like"
            )

        if ent.is_compressed:
            if compression != COMPRESSION_PRS:
                raise ValueError(
                    f"pack_bml: entries[{i}] is_compressed=True but "
                    f"container compression is not PRS"
                )
            if ent.decompressed_size <= 0:
                raise ValueError(
                    f"pack_bml: entries[{i}] is_compressed=True requires "
                    f"a positive decompressed_size (got {ent.decompressed_size})"
                )
            inner = bytes(ent.data)
            decomp_size = ent.decompressed_size
        else:
            raw = bytes(ent.data)
            if compression == COMPRESSION_PRS:
                inner = encode_prs(raw)
            else:
                inner = raw
            decomp_size = len(raw)

        blobs.append(inner)
        decompressed_sizes.append(decomp_size)

        if ent.texture_data is None or len(ent.texture_data) == 0:
            tex_blobs.append(None)
            tex_decompressed_sizes.append(0)
        else:
            if not isinstance(ent.texture_data, (bytes, bytearray, memoryview)):
                raise ValueError(
                    f"pack_bml: entries[{i}].texture_data is "
                    f"{type(ent.texture_data).__name__}, expected bytes-like"
                )
            if ent.texture_is_compressed:
                if ent.texture_decompressed_size <= 0:
                    raise ValueError(
                        f"pack_bml: entries[{i}] texture_is_compressed=True "
                        f"requires positive texture_decompressed_size"
                    )
                tex_blob = bytes(ent.texture_data)
                tex_dsize = ent.texture_decompressed_size
            else:
                # Textures are always PRS-compressed in BMLs (per
                # pso-blender source: "Texture archive is always PRS
                # compressed"). Same encoder as inner blobs.
                raw = bytes(ent.texture_data)
                tex_blob = encode_prs(raw)
                tex_dsize = len(raw)
            tex_blobs.append(tex_blob)
            tex_decompressed_sizes.append(tex_dsize)

    # Pass 2: compute layout — header, table, data_start, per-entry
    # offsets.
    file_count = n
    table_end = HEADER_SIZE + file_count * FILE_ENTRY_SIZE
    data_start = _align_up(table_end, TABLE_ALIGNMENT)

    cursor = data_start
    inner_offsets: List[int] = []
    tex_offsets: List[int] = []
    for i in range(file_count):
        inner_offsets.append(cursor)
        cursor += _align_up(len(blobs[i]), file_alignment)
        if tex_blobs[i] is not None:
            tex_offsets.append(cursor)
            cursor += _align_up(len(tex_blobs[i]), file_alignment)
        else:
            tex_offsets.append(0)
    total_size = cursor

    # Pass 3: build the buffer.
    out = bytearray(total_size)

    # Header at +0x00.
    struct.pack_into(
        "<II",
        out,
        0,
        0,                # unk1
        file_count,
    )
    # The 4 bytes at +0x08 are the u32 magic 0x150 (BMLUtil.cs:206
    # `head.magicNumber = 0x150`), NOT a (compression, has_textures) pair.
    # Writing the full magic keeps every shipped PRS archive byte-exact
    # (magic == 0x150 -> bytes 50 01 00 00 == compression 0x50 + the
    # constant 0x01 high byte) AND fixes fresh no-texture archives, which
    # under the old `out[9] = has_textures` rule emitted a malformed
    # 0x50 magic. ``has_textures``/``compression`` remain accepted for the
    # alignment-classification + round-trip API but no longer corrupt the
    # header. The non-PRS (COMPRESSION_NONE) container is not a shipped
    # format; for it we preserve the historical byte-pair layout.
    if compression == COMPRESSION_PRS:
        struct.pack_into("<I", out, 8, BML_MAGIC)
    else:
        out[8] = compression & 0xFF
        out[9] = 1 if has_textures else 0
    # +0x0c..+0x40 stays zero.

    # File table.
    for i, ent in enumerate(entries):
        ent_off = HEADER_SIZE + i * FILE_ENTRY_SIZE
        out[ent_off:ent_off + NAME_FIELD_SIZE] = _encode_name(ent.name)
        struct.pack_into(
            "<8I",
            out,
            ent_off + NAME_FIELD_SIZE,
            len(blobs[i]),                  # compressed_size
            ent.unk_a,                      # unk_a (preserved)
            decompressed_sizes[i],          # decompressed_size
            len(tex_blobs[i]) if tex_blobs[i] is not None else 0,  # tex_comp
            tex_decompressed_sizes[i],      # tex_decomp
            ent.unk_b,                      # unk_b
            ent.unk_c,                      # unk_c
            ent.unk_d,                      # unk_d
        )

    # Inner + texture payloads.
    for i in range(file_count):
        out[inner_offsets[i]:inner_offsets[i] + len(blobs[i])] = blobs[i]
        if tex_blobs[i] is not None:
            out[tex_offsets[i]:tex_offsets[i] + len(tex_blobs[i])] = tex_blobs[i]

    return bytes(out)


def _classify_file_alignment(mv: memoryview, entries: Sequence[BmlEntry]) -> int:
    """Pick the single ``file_alignment`` ``pack_bml`` needs to reproduce
    an archive's on-disk layout byte-for-byte.

    ``pack_bml`` lays out payloads with ``_align_up(len, file_alignment)``
    using one alignment for the whole archive, so for a byte-exact
    round-trip we must hand it the alignment the original writer used.
    The discriminator is the cumulative-end test: pick 0x20 if summing
    every payload padded to 0x20 lands exactly on the file size, else
    0x800. This is verified byte-exact on all 728 shipped BMLs (the 23
    ``pl[A-Z]nj.bml`` use 0x800; the other 705 use 0x20). Falls back to
    "any entry carries a texture -> 0x20 else 0x800" only for an edited
    archive whose size matches neither sum.

    Note: this is intentionally SEPARATE from the reader's offset
    resolver (``_walk_offsets`` / SeekPadding). The reader needs no
    global alignment; only the alignment-based *packer* does.
    """
    n = len(mv)
    file_count = struct.unpack_from("<I", mv, 4)[0]
    base = _align_up(HEADER_SIZE + file_count * FILE_ENTRY_SIZE, TABLE_ALIGNMENT)
    cumulative_20 = base
    cumulative_800 = base
    for ent in entries:
        cumulative_20 += _align_up(ent.size_compressed, DATA_ALIGNMENT_HAS_TEX)
        cumulative_800 += _align_up(ent.size_compressed, DATA_ALIGNMENT_NO_TEX)
        if ent.tex_size_compressed > 0:
            cumulative_20 += _align_up(ent.tex_size_compressed,
                                       DATA_ALIGNMENT_HAS_TEX)
            cumulative_800 += _align_up(ent.tex_size_compressed,
                                        DATA_ALIGNMENT_NO_TEX)
    if cumulative_20 == n:
        return DATA_ALIGNMENT_HAS_TEX
    if cumulative_800 == n:
        return DATA_ALIGNMENT_NO_TEX
    return (
        DATA_ALIGNMENT_HAS_TEX
        if any(ent.tex_size_compressed > 0 for ent in entries)
        else DATA_ALIGNMENT_NO_TEX
    )


def parse_bml_for_pack(buf: bytes) -> List[BmlPackEntry]:
    """Round-trip helper: parse a BML and return ``BmlPackEntry`` list
    that, when fed back to ``pack_bml`` with matching options, produces
    a byte-identical archive.

    The returned entries carry pre-compressed payloads
    (``is_compressed=True``) plus the original ``unk_a/b/c/d`` u32s.
    The caller passes ``has_textures_override`` based on the parsed
    header byte and ``file_alignment`` based on the parser's
    cumulative-end heuristic to handle the lying-header BMLs.

    Used by tests + the ``pack`` CLI's ``--from`` mode.

    Args:
        buf: full BML bytes.

    Returns:
        Tuple-style: list of pack entries, in archive order. Header
        metadata (compression, has_textures, alignment) must be read
        from the buffer separately by the caller — see
        ``parse_bml_pack_meta``.

    Raises:
        ValueError: on any parser error (propagated from parse_bml).
    """
    if not isinstance(buf, (bytes, bytearray, memoryview)):
        raise ValueError("parse_bml_for_pack: input must be bytes-like")
    mv = memoryview(buf)
    entries = parse_bml(buf)
    # The HIGH byte of the +0x08 magic (0x150) is constant 0x01, and the
    # LOW byte is 0x50 ('P'); every shipped inner blob is PRS-compressed,
    # so the round-trip always stores raw pre-compressed bytes.
    compression_type = mv[8]
    out: List[BmlPackEntry] = []
    # ``pack_bml`` reproduces the on-disk layout from a SINGLE global
    # ``file_alignment`` (it is alignment-based, not offset-list-based),
    # so we still classify the archive's alignment for the round-trip.
    # See ``_classify_file_alignment`` — it preserves the proven
    # cumulative-end test that makes raw repacking byte-exact.
    file_alignment = _classify_file_alignment(mv, entries)
    # Exact per-entry texture offsets come from the SeekPadding walk so
    # the slice is correct even if the archive's padding doesn't match a
    # uniform alignment (robustness; agrees with file_alignment on every
    # shipped BML).
    metas = [
        (e.name, e.size_compressed, e.size_decompressed, e.tex_size_compressed)
        for e in entries
    ]
    walk = _walk_offsets(mv, metas)

    for i, (ent, (_inner_off, tex_off)) in enumerate(zip(entries, walk)):
        # Inner blob bytes (raw, pre-compressed).
        inner = bytes(mv[ent.offset:ent.offset + ent.size_compressed])
        tex_data: Optional[bytes] = None
        tex_decomp = 0
        if ent.has_texture and ent.tex_size_compressed > 0 and tex_off is not None:
            tex_data = bytes(mv[tex_off:tex_off + ent.tex_size_compressed])
            # Read tex_decomp from the raw entry record.
            ent_pos = HEADER_SIZE + i * FILE_ENTRY_SIZE + NAME_FIELD_SIZE
            tex_decomp = struct.unpack_from("<I", mv, ent_pos + 16)[0]
        # Read original unk_a/b/c/d.
        ent_pos = HEADER_SIZE + i * FILE_ENTRY_SIZE + NAME_FIELD_SIZE
        _cs, unk_a, _ds, _tcs, _tds, unk_b, unk_c, unk_d = struct.unpack_from(
            "<8I", mv, ent_pos
        )

        out.append(BmlPackEntry(
            name=ent.name,
            data=inner,
            decompressed_size=ent.size_decompressed,
            is_compressed=(compression_type == COMPRESSION_PRS),
            texture_data=tex_data,
            texture_decompressed_size=tex_decomp,
            texture_is_compressed=tex_data is not None,
            unk_a=unk_a,
            unk_b=unk_b,
            unk_c=unk_c,
            unk_d=unk_d,
        ))
    return out


def parse_bml_pack_meta(buf: bytes) -> Dict[str, int]:
    """Return the archive-level metadata needed to round-trip a BML.

    Returns a dict with keys ``compression``, ``has_textures`` (bool),
    ``file_alignment`` (0x20 or 0x800). Use these as ``compression``,
    ``has_textures_override``, and ``file_alignment`` arguments to
    ``pack_bml`` for byte-exact round-trip.
    """
    if not isinstance(buf, (bytes, bytearray, memoryview)):
        raise ValueError("parse_bml_pack_meta: input must be bytes-like")
    mv = memoryview(buf)
    if len(mv) < HEADER_SIZE:
        raise ValueError("parse_bml_pack_meta: truncated header")
    # ``compression`` and ``has_textures`` are the low/high bytes of the
    # +0x08 magic (0x150 -> 0x50, 0x01) — constant across the corpus.
    # We surface them verbatim because ``pack_bml`` writes them straight
    # back (out[8]=compression, out[9]=has_textures), which is exactly
    # what makes the raw round-trip byte-exact. They are NOT trusted as
    # independent semantic fields by the reader.
    compression = mv[8]
    has_textures = mv[9] != 0
    # ``pack_bml`` is alignment-based, so classify the single alignment
    # that reproduces the on-disk layout (shared with parse_bml_for_pack).
    entries = parse_bml(buf)
    file_alignment = _classify_file_alignment(mv, entries)
    return {
        "compression": compression,
        "has_textures": has_textures,
        "file_alignment": file_alignment,
    }


# ---------------------------------------------------------------------------
# CLI: pack / unpack
# ---------------------------------------------------------------------------
# python -m formats.bml unpack <input.bml> <output_dir>
#   - Decompress every inner blob and (if present) every texture, write
#     them to <output_dir> with the entry name. Textures are written as
#     <entry_name>.xvm. Also writes a sidecar manifest <output_dir>/
#     _bml_manifest.json describing compression/has_textures/file_alignment
#     and the per-entry unk_a/b/c/d values needed for byte-exact pack.
#
# python -m formats.bml pack <input_dir> <output.bml>
#   - Reads <input_dir>/_bml_manifest.json + per-entry files and rebuilds
#     a BML. If the manifest is missing, falls back to a "fresh"
#     repack: zero unks, alignment classified from texture presence,
#     compression=PRS.

def _cli_unpack(in_path: Path, out_dir: Path) -> int:
    import json
    buf = in_path.read_bytes()
    meta = parse_bml_pack_meta(buf)
    pack_entries = parse_bml_for_pack(buf)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source": in_path.name,
        "compression": meta["compression"],
        "has_textures": meta["has_textures"],
        "file_alignment": meta["file_alignment"],
        "entries": [],
    }
    for ent in pack_entries:
        # Decompress for human-readable extraction.
        if ent.is_compressed and meta["compression"] == COMPRESSION_PRS:
            inner_decomp = _prs.decompress(ent.data)
        else:
            inner_decomp = ent.data
        (out_dir / ent.name).write_bytes(inner_decomp)
        tex_path = None
        if ent.texture_data is not None:
            tex_decomp = _prs.decompress(ent.texture_data)
            tex_name = f"{ent.name}.xvm"
            (out_dir / tex_name).write_bytes(tex_decomp)
            tex_path = tex_name
        manifest["entries"].append({
            "name": ent.name,
            "decompressed_size": ent.decompressed_size,
            "texture": tex_path,
            "texture_decompressed_size": ent.texture_decompressed_size,
            "unk_a": ent.unk_a,
            "unk_b": ent.unk_b,
            "unk_c": ent.unk_c,
            "unk_d": ent.unk_d,
        })

    (out_dir / "_bml_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"unpacked {len(pack_entries)} entries to {out_dir}", file=sys.stderr)
    return 0


def _cli_pack(in_dir: Path, out_path: Path) -> int:
    import json
    manifest_path = in_dir / "_bml_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        compression = manifest.get("compression", COMPRESSION_PRS)
        has_textures = bool(manifest.get("has_textures", True))
        file_alignment = manifest.get("file_alignment", DATA_ALIGNMENT_HAS_TEX)
        entries: List[BmlPackEntry] = []
        for em in manifest["entries"]:
            inner_path = in_dir / em["name"]
            if not inner_path.exists():
                raise FileNotFoundError(f"missing inner file: {inner_path}")
            data = inner_path.read_bytes()
            tex_data: Optional[bytes] = None
            tex_decomp = 0
            if em.get("texture"):
                tp = in_dir / em["texture"]
                if not tp.exists():
                    raise FileNotFoundError(f"missing texture file: {tp}")
                tex_data = tp.read_bytes()
                tex_decomp = em.get("texture_decompressed_size", len(tex_data))
            entries.append(BmlPackEntry(
                name=em["name"],
                data=data,
                decompressed_size=em.get("decompressed_size", len(data)),
                is_compressed=False,           # decompressed on disk
                texture_data=tex_data,
                texture_decompressed_size=tex_decomp,
                texture_is_compressed=False,
                unk_a=em.get("unk_a", 0),
                unk_b=em.get("unk_b", 0),
                unk_c=em.get("unk_c", 0),
                unk_d=em.get("unk_d", 0),
            ))
        out = pack_bml(
            entries,
            compression=compression,
            file_alignment=file_alignment,
            has_textures_override=has_textures,
        )
    else:
        # Fresh-repack mode: enumerate every file in the directory.
        # Files with a sibling <name>.xvm get textures; everything else
        # is bare. The .xvm files themselves are skipped from the
        # entry list (they're attached to their inner counterparts).
        all_files = sorted(p for p in in_dir.iterdir() if p.is_file())
        # Separate textures (<name>.xvm) from inners.
        names = {p.name for p in all_files}
        entries = []
        for p in all_files:
            if p.name == "_bml_manifest.json":
                continue
            if p.name.endswith(".xvm"):
                # Skip; will be picked up by the inner with same stem.
                stem_name = p.name[:-len(".xvm")]
                if stem_name in names:
                    continue
                # Bare .xvm — treat as inner, no texture.
            data = p.read_bytes()
            tex_path = in_dir / f"{p.name}.xvm"
            tex_data = tex_path.read_bytes() if tex_path.exists() else None
            entries.append(BmlPackEntry(
                name=p.name,
                data=data,
                is_compressed=False,
                texture_data=tex_data,
                texture_is_compressed=False,
            ))
        if not entries:
            raise ValueError(f"no entries found in {in_dir}")
        out = pack_bml(entries)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(out)
    print(
        f"packed {len(entries) if 'entries' in dir() else '?'} entries "
        f"to {out_path} ({len(out)} bytes)",
        file=sys.stderr,
    )
    return 0


def _cli(argv: Optional[List[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="python -m formats.bml")
    sub = p.add_subparsers(dest="cmd", required=True)
    p_unpack = sub.add_parser("unpack", help="extract BML to a directory")
    p_unpack.add_argument("input", type=Path, help="input .bml file")
    p_unpack.add_argument("output", type=Path, help="output directory")
    p_pack = sub.add_parser("pack", help="pack a directory into a .bml")
    p_pack.add_argument("input", type=Path, help="input directory")
    p_pack.add_argument("output", type=Path, help="output .bml file")
    args = p.parse_args(argv)
    if args.cmd == "unpack":
        return _cli_unpack(args.input, args.output)
    if args.cmd == "pack":
        return _cli_pack(args.input, args.output)
    return 1


if __name__ == "__main__":
    sys.exit(_cli())
