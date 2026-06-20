"""Comprehensive model-coverage audit for the PSOBB Texture Editor.

Walks every BML and standalone ``.nj`` / ``.xj`` in
``~/PSOBB.IO/data/`` and records:

  - format detected (Nj chunk vs Xj descriptor vs other)
  - inner-model count (BMLs only)
  - inner-texture archive presence + texture count
  - submesh / vertex / triangle counts produced by the editor's parsers
  - NJTL chunk presence + entry count
  - failure modes:
      parse_error             - parser raised
      zero_geometry           - parser returned [] meshes
      partial_geometry        - parser produced fewer triangles than
                                a side-by-side pso-blender extraction
                                (best-effort, see notes below)
      texture_missing         - model references material_ids but no
                                XVM is paired
      texture_count_mismatch  - distinct material_ids != XVM xvr_count
      unknown_inner_extension - BML inner with extension we don't dispatch

Outputs:
  - ~/Repositories/psobb-studio/MODEL_COVERAGE.csv  (one row per inner asset)
  - ~/Repositories/psobb-studio/MODEL_COVERAGE.md   (human readable summary)

Usage:
  python scripts/coverage_audit.py [--limit N] [--no-pso-blender]

Run from the editor root (``~/Repositories/psobb-studio``). The script imports the
editor's own parsers from ``formats/``, plus pso-blender's pure-Python
PRS decoder (no Blender dep) for in-process BML extraction. PuyoToolsCli
is NOT invoked — we use the embedded prs.py for ~50x speedup on a full
365-BML walk.

Optional ``--no-pso-blender`` disables side-by-side triangle counts
(PsoBlenderTris column will be -1). Useful when you only want the
editor-side stats.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import struct
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Resolve paths relative to the script so the audit can be re-run from
# any cwd without breaking imports.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = Path(os.path.expanduser("~/PSOBB.IO/data"))

sys.path.insert(0, str(ROOT))
# Editor's parsers
from formats.bml import parse_bml, BmlEntry  # noqa: E402
from formats.iff import parse_iff             # noqa: E402
from formats.xj import parse_xj_njcm, parse_nj_file, parse_skeleton, _walk_chunk_stream, _MESH_TREE_NODE_SIZE  # noqa: E402
from formats.xj_descriptor import parse_xj_descriptor, parse_xj_file  # noqa: E402

# pso-blender prs.py is pure-Python and has zero dependencies (it just
# decodes the LZ77-style stream). The package's __init__.py however
# imports bpy at top level, so we side-load just prs.py via importlib.
try:
    import importlib.util as _ilu
    _prs_path = ROOT / "_modelwork" / "pso-blender" / "pso_blender" / "prs.py"
    _spec = _ilu.spec_from_file_location("pso_blender_prs", str(_prs_path))
    if _spec is None or _spec.loader is None:
        raise ImportError(f"could not spec {_prs_path}")
    _prs_mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_prs_mod)
    prs_decompress = _prs_mod.decompress  # type: ignore
    HAS_PRS = True
except Exception as e:
    print(f"warning: pso-blender prs.py unavailable: {e}", file=sys.stderr)
    HAS_PRS = False


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass
class AssetRow:
    """One row in MODEL_COVERAGE.csv. One per inner-model asset.

    For standalone .nj files, container_path == asset_path and
    inner_name is empty.
    """
    container_path: str = ""              # e.g. ".../bm_boss1_dragon.bml" or "" for standalone
    inner_name: str = ""                  # e.g. "bm_boss1_dragon.nj"
    asset_path: str = ""                  # absolute path to the file actually parsed (for standalones)
    inner_ext: str = ""                   # ".nj" / ".xj" / ".njm" / unknown
    has_textures: bool = False            # BML entry has a paired XVM
    detected_format: str = ""             # "nj-chunk" / "xj-descriptor" / "njm-anim" / "iff-only" / "unknown"
    container_size_bytes: int = 0         # size of the BML / standalone file
    inner_size_bytes: int = 0             # size of the decompressed inner
    iff_chunks: str = ""                  # comma-list of IFF chunk types
    njtl_present: bool = False
    njtl_entry_count: int = 0             # number of texture-name strings in NJTL
    njtl_names: str = ""                  # joined; truncated to 200 chars
    xvm_xvr_count: int = 0                # archived textures (-1 if no XVM)
    xvm_total_bytes: int = 0
    submesh_count: int = 0                # editor parser output count
    vertex_count: int = 0
    triangle_count: int = 0
    distinct_material_ids: int = 0
    skeleton_bone_count: int = 0
    bone_max_depth: int = 0
    bone_avg_branching: float = 0.0
    pso_blender_meshes: int = -1
    pso_blender_verts: int = -1
    pso_blender_tris: int = -1
    chunk_type_histogram: str = ""        # e.g. "32:5,64:8,255:1"
    xj_vertex_type_histogram: str = ""    # e.g. "3:2,7:1"
    failure_class: str = ""               # "" if ok; one of the documented classes
    failure_detail: str = ""              # short detail string (truncated to ~250 chars)
    sample_first_bytes: str = ""          # first 64 bytes hex-encoded for parse-fail rows


@dataclass
class AuditSummary:
    """Aggregate stats across the whole walk."""
    total_assets: int = 0
    total_bmls: int = 0
    total_standalones: int = 0
    by_format: Dict[str, int] = field(default_factory=dict)
    by_failure_class: Dict[str, int] = field(default_factory=dict)
    nj_chunk_type_hist: Dict[int, int] = field(default_factory=dict)
    xj_vertex_type_hist: Dict[int, int] = field(default_factory=dict)
    inner_extensions: Dict[str, int] = field(default_factory=dict)
    texture_resolution: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decompress_inner(buf: bytes, ent: BmlEntry, compression_type: int) -> bytes:
    """Decompress the inner blob for a BmlEntry.

    The BML header byte at offset 8 tells us if the file uses PRS or
    NONE compression. PSOBB.IO BMLs are universally PRS (compression
    byte 0x50), but we tolerate both for completeness.
    """
    raw = bytes(buf[ent.offset:ent.offset + ent.size_compressed])
    if compression_type == 0x50:  # 'P' = PRS
        if not HAS_PRS:
            raise RuntimeError("PRS unavailable; install pso_blender or run with --no-pso-blender")
        return bytes(prs_decompress(bytearray(raw)))
    return raw  # uncompressed


def _decompress_texture(buf: bytes, ent: BmlEntry, header_byte9: int) -> Optional[bytes]:
    """Read & PRS-decompress an entry's paired XVM, if any.

    PSO BML stores the texture as a PRS-compressed XVM following the
    inner file (aligned to 0x20 if has_textures==1, else 0x800). The
    texture is ALWAYS PRS-compressed even when the inner uses NONE.

    Falls back to the editor's tolerant in-process PRS decoder when
    pso-blender's strict implementation rejects a "stub XVM" (10
    BMLs in the install whose PRS stream lacks a proper end marker
    but whose decompressed prefix IS a valid XVMH archive).
    """
    if not ent.has_texture or ent.tex_size_compressed == 0:
        return None
    align = 0x20 if header_byte9 != 0 else 0x800

    def _align_up(v: int, a: int) -> int:
        return (v + a - 1) & ~(a - 1)

    tex_off = ent.offset + _align_up(ent.size_compressed, align)
    raw = bytes(buf[tex_off:tex_off + ent.tex_size_compressed])
    if HAS_PRS:
        try:
            return bytes(prs_decompress(bytearray(raw)))
        except Exception:
            pass
    # Tolerant fallback via editor's in-process decoder.
    try:
        from formats import prs as _editor_prs
        partial = _editor_prs.decompress(raw, tolerant=True)
        if len(partial) >= 0x40 and partial[:4] == b"XVMH":
            return partial
    except Exception:
        pass
    return None


def _parse_xvm_count(blob: Optional[bytes]) -> int:
    """Read the XVMH header to extract the XVR (texture) count.

    Returns -1 if the blob isn't a valid XVMH archive. Layout (LE):
        u8[4]  "XVMH"
        u32    body_size
        u32    xvr_count
    """
    if not blob or len(blob) < 12:
        return -1
    if blob[:4] != b"XVMH":
        return -1
    try:
        return struct.unpack_from("<I", blob, 8)[0]
    except struct.error:
        return -1


def _parse_njtl(payload: bytes) -> List[str]:
    """Decode NJTL chunk payload -> list of texture name strings.

    NJTL layout (per pso-blender njtl.py):

        TextureList header:
            u32 elements_ptr
            u32 count
        TextureListEntry repeated:
            u32 name_ptr
            u32 unk1   (set by client at runtime)
            u32 data   (set by client at runtime)

    All pointers are relative to the start of the payload (post-IFF
    header). Names are NUL-terminated ASCII at the pointed-to offset.
    Returns [] when the chunk is too short or pointers are corrupt.
    """
    if len(payload) < 8:
        return []
    try:
        elements_ptr, count = struct.unpack_from("<II", payload, 0)
    except struct.error:
        return []
    if count == 0 or count > 256:  # sanity cap
        return []
    if elements_ptr + count * 12 > len(payload):
        return []
    names: List[str] = []
    for i in range(count):
        entry_off = elements_ptr + i * 12
        try:
            name_ptr = struct.unpack_from("<I", payload, entry_off)[0]
        except struct.error:
            break
        if name_ptr == 0 or name_ptr >= len(payload):
            names.append("")
            continue
        # NUL-terminated ASCII, capped at 64 chars
        end = name_ptr
        while end < len(payload) and end < name_ptr + 64 and payload[end] != 0:
            end += 1
        try:
            names.append(payload[name_ptr:end].decode("ascii", errors="replace"))
        except Exception:
            names.append("")
    return names


def _bone_tree_stats(buf: bytes) -> Tuple[int, int, float]:
    """Return (count, max_depth, avg_branching) for the .nj/.xj bone tree.

    Walks every NJCM chunk and traces the MeshTreeNode linked list.
    Trunated nodes / cycles are ignored.
    """
    try:
        chunks = parse_iff(buf)
    except Exception:
        return 0, 0, 0.0
    njcm = next((c for c in chunks if c.type == "NJCM"), None)
    if njcm is None:
        return 0, 0, 0.0
    body = njcm.data
    if len(body) < _MESH_TREE_NODE_SIZE:
        return 0, 0, 0.0

    visited: Set[int] = set()
    children_per_node: Dict[int, int] = defaultdict(int)
    stack: List[Tuple[int, int]] = [(0, 0)]
    bones: List[Tuple[int, int, int]] = []  # (off, depth, parent_off)
    max_depth = 0
    while stack:
        off, depth = stack.pop()
        if off in visited or off + _MESH_TREE_NODE_SIZE > len(body):
            continue
        visited.add(off)
        try:
            f = struct.unpack_from("<II3f3i3fII", body, off)
        except struct.error:
            continue
        child_ptr, next_ptr = f[11], f[12]
        bones.append((off, depth, -1))
        if depth > max_depth:
            max_depth = depth
        if child_ptr and child_ptr not in visited:
            stack.append((child_ptr, depth + 1))
            children_per_node[off] += 1
        if next_ptr and next_ptr not in visited:
            stack.append((next_ptr, depth))
    branching_vals = [v for v in children_per_node.values() if v > 0]
    avg_branching = (sum(branching_vals) / len(branching_vals)) if branching_vals else 0.0
    return len(bones), max_depth, avg_branching


def _detect_format(inner_ext: str, buf: bytes) -> str:
    """Decide which parser path applies and tag a coarse format string."""
    ext = inner_ext.lower()
    if ext == ".njm":
        return "njm-anim"
    if ext == ".njs":
        # NSSM (Ninja State-machine Sequence Motion) — animation-only.
        # Same surface treatment as .njm; the chunk parser returns []
        # for both because they lack an NJCM chunk.
        return "njm-anim"
    if ext == ".xj":
        return "xj-descriptor"
    if ext == ".nj":
        return "nj-chunk"
    # Try IFF anyway
    try:
        chunks = parse_iff(buf)
    except Exception:
        return "unknown"
    if any(c.type == "NJCM" for c in chunks):
        # We can't tell .nj from .xj from header alone; default to nj-chunk.
        return "nj-chunk"
    if any(c.type == "NMDM" for c in chunks):
        return "njm-anim"
    if any(c.type == "NSSM" for c in chunks):
        return "njm-anim"
    return "iff-only"


def _hist_str(d: Dict[int, int]) -> str:
    """Render a sparse {int: count} histogram as a stable comma-string."""
    return ",".join(f"{k}:{v}" for k, v in sorted(d.items()))


def _enumerate_chunk_type_histogram(buf: bytes) -> Dict[int, int]:
    """Walk every vlist + plist chunk stream and tally type_id frequencies.

    Used for finding chunk types ≥76 that the parser doesn't handle.
    Walks the same MeshTreeNode tree that the parser uses but doesn't
    interpret the chunks — only counts type_id values seen.
    """
    out: Dict[int, int] = defaultdict(int)
    try:
        chunks = parse_iff(buf)
    except Exception:
        return out
    njcm = next((c for c in chunks if c.type == "NJCM"), None)
    if njcm is None:
        return out
    body = njcm.data
    visited: Set[int] = set()
    stack: List[int] = [0]
    while stack:
        off = stack.pop()
        if off in visited or off + _MESH_TREE_NODE_SIZE > len(body):
            continue
        visited.add(off)
        try:
            f = struct.unpack_from("<II3f3i3fII", body, off)
        except struct.error:
            continue
        eval_flags, mesh_ptr = f[0], f[1]
        child_ptr, next_ptr = f[11], f[12]
        # NjMesh (24 bytes): vlist, plist, bbox(4)
        if mesh_ptr and mesh_ptr + 24 <= len(body):
            try:
                vlist_off, plist_off = struct.unpack_from("<II", body, mesh_ptr)
            except struct.error:
                vlist_off, plist_off = 0, 0
            for stream_start in (vlist_off, plist_off):
                if stream_start <= 0 or stream_start >= len(body):
                    continue
                try:
                    walked = _walk_chunk_stream(body, stream_start)
                except Exception:
                    continue
                for (_hdr, type_id, _flags, _bp, _bs) in walked:
                    out[type_id] += 1
        if child_ptr:
            stack.append(child_ptr)
        if next_ptr:
            stack.append(next_ptr)
    return out


def _enumerate_xj_vertex_types(buf: bytes) -> Dict[int, int]:
    """For .xj files, walk the descriptor tree and tally vertex_type values.

    Reads the 16-byte VertexInfoTable rows (signed i16 vertex_type at
    offset 0). Used for finding vertex types outside our 2-7 range.
    """
    out: Dict[int, int] = defaultdict(int)
    try:
        chunks = parse_iff(buf)
    except Exception:
        return out
    njcm = next((c for c in chunks if c.type == "NJCM"), None)
    if njcm is None:
        return out
    body = njcm.data
    visited: Set[int] = set()
    stack: List[int] = [0]
    while stack:
        off = stack.pop()
        if off in visited or off + _MESH_TREE_NODE_SIZE > len(body):
            continue
        visited.add(off)
        try:
            f = struct.unpack_from("<II3f3i3fII", body, off)
        except struct.error:
            continue
        model_ptr = f[1]
        child_ptr, next_ptr = f[11], f[12]
        # XjModel (44 bytes): u32 flags + 6 u32 + 4 f32
        if model_ptr and model_ptr + 44 <= len(body):
            try:
                xm = struct.unpack_from("<I 6I 4f", body, model_ptr)
            except struct.error:
                xm = None
            if xm is not None:
                vbi_off = xm[1]
                vbi_count = xm[2]
                if 0 < vbi_off < len(body) and 0 < vbi_count < 32:
                    for vi in range(vbi_count):
                        row = vbi_off + vi * 16
                        if row + 16 > len(body):
                            break
                        try:
                            vt = struct.unpack_from("<h", body, row)[0]
                        except struct.error:
                            break
                        out[vt] += 1
        if child_ptr:
            stack.append(child_ptr)
        if next_ptr:
            stack.append(next_ptr)
    return out


# ---------------------------------------------------------------------------
# pso-blender side-by-side (best-effort, doesn't import bpy)
# ---------------------------------------------------------------------------
#
# pso-blender's full xj/nj importers depend on bpy (Blender's Python
# API) which is unavailable in plain CPython. We instead invoke a
# minimal, dependency-free byte-walker that re-implements the same
# vertex/triangle counting logic — sufficient for "did we miss
# triangles?" comparisons. For triangle counts the comparison is:
#   - Nj: count strip-chunk triangles in every plist stream
#   - Xj: count tristrip indices in every strip table row
# We add up everything that pso-blender's parser would emit and
# compare to our editor's output. A delta > 5% surfaces as
# "partial_geometry".


def _pso_blender_count_nj(buf: bytes) -> Tuple[int, int, int]:
    """Approximate pso-blender's Nj parser: returns (meshes, verts, tris).

    Walks the same MeshTreeNode tree but tallies counts directly from
    chunk headers — independent code path from formats/xj.py so any
    discrepancy with our editor parser is surfaced.
    """
    try:
        chunks = parse_iff(buf)
    except Exception:
        return 0, 0, 0
    njcm = next((c for c in chunks if c.type == "NJCM"), None)
    if njcm is None:
        return 0, 0, 0
    body = njcm.data
    visited_nodes: Set[int] = set()
    stack: List[int] = [0]
    total_verts = 0
    total_tris = 0
    total_meshes = 0
    while stack:
        off = stack.pop()
        if off in visited_nodes or off + _MESH_TREE_NODE_SIZE > len(body):
            continue
        visited_nodes.add(off)
        try:
            f = struct.unpack_from("<II3f3i3fII", body, off)
        except struct.error:
            continue
        mesh_ptr = f[1]
        child_ptr, next_ptr = f[11], f[12]
        if mesh_ptr and mesh_ptr + 24 <= len(body):
            try:
                vlist_off, plist_off = struct.unpack_from("<II", body, mesh_ptr)
            except struct.error:
                vlist_off, plist_off = 0, 0
            # Vertex pass: count vertex slots
            if vlist_off and vlist_off < len(body):
                try:
                    walked = _walk_chunk_stream(body, vlist_off)
                except Exception:
                    walked = []
                for (_hdr, type_id, _flags, body_pos, body_size) in walked:
                    if 32 <= type_id <= 50 and body_size >= 4:
                        try:
                            base, vcount = struct.unpack_from("<HH", body, body_pos + 2)
                            total_verts += vcount
                        except struct.error:
                            pass
            # Polygon pass: count strip chunks → triangles
            if plist_off and plist_off < len(body):
                try:
                    walked = _walk_chunk_stream(body, plist_off)
                except Exception:
                    walked = []
                for (_hdr, type_id, _flags, body_pos, body_size) in walked:
                    if 64 <= type_id <= 75 and body_size >= 4:
                        # Strip header = 1 u16; strip_count low 14 bits.
                        try:
                            hdr = struct.unpack_from("<h", body, body_pos + 2)[0]
                            strip_count = hdr & 0x3FFF
                        except struct.error:
                            continue
                        # Walk each strip; count indices then convert
                        # to triangle counts (n-2). Only need approx.
                        cur = body_pos + 4
                        end = body_pos + body_size
                        has_uv = type_id in (65, 66, 68, 69, 71, 72)
                        has_normal = type_id in (67, 68, 69)
                        has_color = type_id in (70, 71, 72)
                        has_double_uv = type_id in (74, 75)
                        # Per-vertex extra bytes after the u16 idx
                        extra = 0
                        if has_uv:
                            extra += 4
                        if has_color:
                            extra += 4
                        if has_normal:
                            extra += 6
                        if has_double_uv:
                            extra += 8
                        per_vert = 2 + extra
                        for _strip_i in range(strip_count):
                            if cur + 2 > end:
                                break
                            try:
                                strip_header = struct.unpack_from("<h", body, cur)[0]
                            except struct.error:
                                break
                            cur += 2
                            ic = abs(strip_header)
                            # User offset bytes per triangle past the third.
                            user_offset_size = 2 * ((strip_header >> 14) & 0x3) if False else 0
                            # Skip indices + extras
                            cur += per_vert * ic
                            if ic > 2:
                                cur += user_offset_size * (ic - 2)
                            if ic >= 3:
                                total_tris += ic - 2
                        total_meshes += 1
        if child_ptr:
            stack.append(child_ptr)
        if next_ptr:
            stack.append(next_ptr)
    return total_meshes, total_verts, total_tris


def _pso_blender_count_xj(buf: bytes) -> Tuple[int, int, int]:
    """Approximate pso-blender's Xj parser: returns (meshes, verts, tris).

    Walks the descriptor tables directly (no Blender dep). Counts
    strip indices the same way the editor parser would but via an
    independent byte-walker.
    """
    try:
        chunks = parse_iff(buf)
    except Exception:
        return 0, 0, 0
    njcm = next((c for c in chunks if c.type == "NJCM"), None)
    if njcm is None:
        return 0, 0, 0
    body = njcm.data
    visited: Set[int] = set()
    stack: List[int] = [0]
    total_verts = 0
    total_tris = 0
    total_meshes = 0
    while stack:
        off = stack.pop()
        if off in visited or off + _MESH_TREE_NODE_SIZE > len(body):
            continue
        visited.add(off)
        try:
            f = struct.unpack_from("<II3f3i3fII", body, off)
        except struct.error:
            continue
        model_ptr = f[1]
        child_ptr, next_ptr = f[11], f[12]
        if model_ptr and model_ptr + 44 <= len(body):
            try:
                xm = struct.unpack_from("<I 6I 4f", body, model_ptr)
            except struct.error:
                xm = None
            if xm is not None:
                vbi_off, vbi_count = xm[1], xm[2]
                ts_off, ts_count = xm[3], xm[4]
                tts_off, tts_count = xm[5], xm[6]
                if 0 < vbi_off < len(body) and 0 < vbi_count < 32:
                    for vi in range(vbi_count):
                        row = vbi_off + vi * 16
                        if row + 16 > len(body):
                            break
                        try:
                            _vt, _flags, vto, vsize, vcount = struct.unpack_from("<hh III", body, row)
                            total_verts += vcount
                        except struct.error:
                            pass
                for tbl_off, tbl_count in ((ts_off, ts_count), (tts_off, tts_count)):
                    if tbl_off <= 0 or tbl_count <= 0 or tbl_count > 4096:
                        continue
                    for ri in range(tbl_count):
                        row = tbl_off + ri * 20
                        if row + 20 > len(body):
                            break
                        try:
                            _mat_off, _mat_size, idx_off, idx_count, _unk = struct.unpack_from("<IIII I", body, row)
                        except struct.error:
                            break
                        if idx_count >= 3 and 0 < idx_off < len(body):
                            total_tris += idx_count - 2
                        total_meshes += 1
        if child_ptr:
            stack.append(child_ptr)
        if next_ptr:
            stack.append(next_ptr)
    return total_meshes, total_verts, total_tris


# ---------------------------------------------------------------------------
# Per-asset audit
# ---------------------------------------------------------------------------


def _audit_asset(
    container_path: str,
    inner_name: str,
    asset_path: str,
    inner_ext: str,
    has_textures: bool,
    container_size: int,
    inner_bytes: bytes,
    xvm_bytes: Optional[bytes],
    do_pso_blender: bool = True,
) -> AssetRow:
    """Run all checks on one inner-model asset and produce a row."""
    row = AssetRow(
        container_path=container_path,
        inner_name=inner_name,
        asset_path=asset_path,
        inner_ext=inner_ext,
        has_textures=has_textures,
        container_size_bytes=container_size,
        inner_size_bytes=len(inner_bytes),
        sample_first_bytes=inner_bytes[:64].hex(),
    )

    # Format detection
    row.detected_format = _detect_format(inner_ext, inner_bytes)

    # IFF chunks + NJTL
    try:
        chunks = parse_iff(inner_bytes)
    except Exception as e:
        row.failure_class = "parse_error"
        row.failure_detail = f"IFF: {e}"
        return row
    row.iff_chunks = ",".join(c.type for c in chunks)
    njtl = next((c for c in chunks if c.type == "NJTL"), None)
    if njtl is not None:
        row.njtl_present = True
        names = _parse_njtl(njtl.data)
        row.njtl_entry_count = len(names)
        row.njtl_names = ("|".join(names))[:200]

    # XVM stats
    if xvm_bytes is not None:
        row.xvm_xvr_count = _parse_xvm_count(xvm_bytes)
        row.xvm_total_bytes = len(xvm_bytes)
    else:
        row.xvm_xvr_count = -1

    # Bone tree stats
    bone_count, max_depth, avg_branch = _bone_tree_stats(inner_bytes)
    row.skeleton_bone_count = bone_count
    row.bone_max_depth = max_depth
    row.bone_avg_branching = avg_branch

    # Type histograms
    if row.detected_format == "nj-chunk":
        chunk_hist = _enumerate_chunk_type_histogram(inner_bytes)
        row.chunk_type_histogram = _hist_str(chunk_hist)
    elif row.detected_format == "xj-descriptor":
        vt_hist = _enumerate_xj_vertex_types(inner_bytes)
        row.xj_vertex_type_histogram = _hist_str(vt_hist)

    # Editor parser
    try:
        if row.detected_format == "xj-descriptor":
            meshes = parse_xj_file(inner_bytes)
        elif row.detected_format == "nj-chunk":
            meshes = parse_nj_file(inner_bytes)
        elif row.detected_format == "njm-anim":
            meshes = []  # animation, no geometry
        else:
            meshes = parse_nj_file(inner_bytes)  # best effort
    except Exception as e:
        row.failure_class = "parse_error"
        row.failure_detail = f"{type(e).__name__}: {e}"[:250]
        return row

    row.submesh_count = len(meshes)
    row.vertex_count = sum(len(m.vertices) for m in meshes)
    row.triangle_count = sum(len(m.indices) // 3 for m in meshes)
    distinct_mats = {m.material_id for m in meshes}
    row.distinct_material_ids = len(distinct_mats)

    # pso-blender side-by-side
    if do_pso_blender:
        if row.detected_format == "nj-chunk":
            pm, pv, pt = _pso_blender_count_nj(inner_bytes)
        elif row.detected_format == "xj-descriptor":
            pm, pv, pt = _pso_blender_count_xj(inner_bytes)
        else:
            pm = pv = pt = -1
        row.pso_blender_meshes = pm
        row.pso_blender_verts = pv
        row.pso_blender_tris = pt

    # Failure classification (after all stats are populated)
    if row.detected_format == "njm-anim":
        # Animation files have no geometry by design — not a failure.
        return row
    if row.submesh_count == 0:
        row.failure_class = "zero_geometry"
        row.failure_detail = (
            f"chunks={row.iff_chunks}; njtl={row.njtl_entry_count}; "
            f"chunk_hist={row.chunk_type_histogram[:80]}"
        )
        return row
    # Only flag as partial if pso-blender saw substantially more triangles.
    if (
        do_pso_blender
        and row.pso_blender_tris > 0
        and row.triangle_count > 0
        and row.pso_blender_tris >= row.triangle_count * 1.05
    ):
        delta = row.pso_blender_tris - row.triangle_count
        ratio = (delta / row.pso_blender_tris) * 100
        row.failure_class = "partial_geometry"
        row.failure_detail = (
            f"editor={row.triangle_count}, pso_blender~{row.pso_blender_tris}, "
            f"missing≈{delta} ({ratio:.1f}%)"
        )
        return row
    if row.has_textures and xvm_bytes is None:
        row.failure_class = "texture_missing"
        row.failure_detail = "BML claims has_texture but XVM payload could not be decompressed"
        return row
    # Texture count mismatch: only a real failure when the model
    # references MORE distinct material slots than the XVM ships AND
    # the unmatched names cannot be resolved cross-BML. The previous
    # "distinct != xvr_count" check produced 46 false positives because
    # PSOBB models commonly use a subset of NJTL slots (distinct_mids=2
    # but xvr_count=8 is fine). With cross-BML resolution landed
    # (formats/texture_index.py), names that fail in-BML now have a
    # second chance at bind time. We surface the mismatch as a
    # diagnostic note but only fail the audit if the max material id
    # exceeds the available XVR records AND no NJTL name lookup
    # succeeds.
    if (
        row.has_textures
        and row.xvm_xvr_count >= 0
        and row.distinct_material_ids > 0
    ):
        # Audit-side cross-BML check: if the model has any NJTL names
        # that aren't covered by its inline XVM, see if the global
        # texture index has them. We import lazily so the audit can
        # run without the editor server.
        unresolved_names: list[str] = []
        try:
            from formats import texture_index as _ti
            # The njtl_names field is "|"-separated.
            njtl_names = [
                n for n in (row.njtl_names or "").split("|") if n.strip()
            ]
            if njtl_names and DATA_DIR.exists():
                for name in njtl_names:
                    name = name.strip()
                    if not name:
                        continue
                    locs = _ti.lookup(DATA_DIR, name)
                    if not locs:
                        unresolved_names.append(name)
        except Exception:
            unresolved_names = []
        if (
            row.xvm_xvr_count < row.distinct_material_ids
            and unresolved_names
        ):
            row.failure_class = "texture_count_mismatch"
            row.failure_detail = (
                f"distinct_materials={row.distinct_material_ids}, "
                f"xvm_xvr_count={row.xvm_xvr_count}, "
                f"unresolved={','.join(unresolved_names[:3])}"
            )
        return row

    return row


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _walk_bml(path: Path, do_pso_blender: bool, ignore_inner_exts: Tuple[str, ...] = ()) -> List[AssetRow]:
    """Open a BML, decompress every inner, run the audit on each."""
    rows: List[AssetRow] = []
    try:
        buf = path.read_bytes()
    except OSError as e:
        rows.append(AssetRow(
            container_path=str(path),
            inner_name="",
            asset_path=str(path),
            inner_ext="",
            failure_class="parse_error",
            failure_detail=f"read failed: {e}",
        ))
        return rows
    try:
        entries = parse_bml(buf)
    except Exception as e:
        rows.append(AssetRow(
            container_path=str(path),
            inner_name="",
            asset_path=str(path),
            inner_ext="",
            failure_class="parse_error",
            failure_detail=f"parse_bml: {e}",
            sample_first_bytes=buf[:64].hex(),
        ))
        return rows
    compression_type = buf[8] if len(buf) > 8 else 0
    header_byte9 = buf[9] if len(buf) > 9 else 0
    container_size = path.stat().st_size

    for ent in entries:
        inner_ext = Path(ent.name).suffix.lower()
        if inner_ext in ignore_inner_exts:
            continue
        # Decompress inner
        try:
            inner_bytes = _decompress_inner(buf, ent, compression_type)
        except Exception as e:
            rows.append(AssetRow(
                container_path=str(path),
                inner_name=ent.name,
                asset_path=str(path),
                inner_ext=inner_ext,
                has_textures=ent.has_texture,
                container_size_bytes=container_size,
                failure_class="parse_error",
                failure_detail=f"PRS: {e}",
            ))
            continue
        # Decompress XVM (if any)
        xvm_bytes = _decompress_texture(buf, ent, header_byte9)

        # Truncated 32-character BML name field: a few PSOBB.IO entries
        # have inner names that exactly fill the 32-byte name slot,
        # leaving nothing for the extension. ``api_model_mesh`` falls
        # back to ``.nj`` in that case (the recovered first-bytes are
        # always NJTL chunk magic). We mirror that dispatch here.
        effective_ext = inner_ext
        if (
            effective_ext == ""
            and len(ent.name) == 32
            and ent.name.endswith(".")
        ):
            effective_ext = ".nj"
        if effective_ext not in (".nj", ".xj", ".njm", ".njs"):
            # Inner with extension we don't dispatch in api_model_mesh.
            # Surface it so the agent prompt can classify (e.g. .bml inside BML?).
            row = AssetRow(
                container_path=str(path),
                inner_name=ent.name,
                asset_path=str(path),
                inner_ext=inner_ext,
                has_textures=ent.has_texture,
                container_size_bytes=container_size,
                inner_size_bytes=len(inner_bytes),
                sample_first_bytes=inner_bytes[:64].hex(),
                failure_class="unknown_inner_extension",
                failure_detail=f"inner_ext={inner_ext!r} not handled by api_model_mesh dispatch",
            )
            if xvm_bytes is not None:
                row.xvm_xvr_count = _parse_xvm_count(xvm_bytes)
                row.xvm_total_bytes = len(xvm_bytes)
            rows.append(row)
            continue

        rows.append(_audit_asset(
            container_path=str(path),
            inner_name=ent.name,
            asset_path=str(path),
            inner_ext=inner_ext,
            has_textures=ent.has_texture,
            container_size=container_size,
            inner_bytes=inner_bytes,
            xvm_bytes=xvm_bytes,
            do_pso_blender=do_pso_blender,
        ))
    return rows


def _walk_standalone(path: Path, do_pso_blender: bool) -> AssetRow:
    """Audit one top-level .nj / .xj / .njm file."""
    try:
        inner_bytes = path.read_bytes()
    except OSError as e:
        return AssetRow(
            container_path="",
            inner_name="",
            asset_path=str(path),
            inner_ext=path.suffix.lower(),
            failure_class="parse_error",
            failure_detail=f"read failed: {e}",
        )
    inner_ext = path.suffix.lower()
    return _audit_asset(
        container_path="",
        inner_name="",
        asset_path=str(path),
        inner_ext=inner_ext,
        has_textures=False,  # standalones never have a paired XVM
        container_size=path.stat().st_size,
        inner_bytes=inner_bytes,
        xvm_bytes=None,
        do_pso_blender=do_pso_blender,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, audit only the first N assets (for smoke tests).")
    ap.add_argument("--no-pso-blender", action="store_true",
                    help="Skip pso-blender side-by-side counts.")
    ap.add_argument("--out-csv", default=str(ROOT / "MODEL_COVERAGE.csv"))
    ap.add_argument("--out-md", default=str(ROOT / "MODEL_COVERAGE.md"))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    bmls = sorted(data_dir.glob("*.bml"))
    standalones = sorted([
        *data_dir.glob("*.nj"),
        *data_dir.glob("*.xj"),
        *data_dir.glob("*.njm"),
    ])

    do_pb = not args.no_pso_blender
    rows: List[AssetRow] = []
    summary = AuditSummary()
    summary.total_bmls = len(bmls)
    summary.total_standalones = len(standalones)

    print(f"# BMLs to audit: {len(bmls)}", file=sys.stderr)
    print(f"# standalones to audit: {len(standalones)}", file=sys.stderr)

    processed = 0
    # 1) BMLs
    for i, p in enumerate(bmls):
        if args.limit and processed >= args.limit:
            break
        try:
            br = _walk_bml(p, do_pso_blender=do_pb)
        except Exception:
            traceback.print_exc()
            continue
        rows.extend(br)
        processed += len(br)
        if i % 20 == 0:
            print(f"  [bml {i+1}/{len(bmls)}] {p.name}: +{len(br)} rows ({processed} total)",
                  file=sys.stderr)

    # 2) Standalones
    for i, p in enumerate(standalones):
        if args.limit and processed >= args.limit:
            break
        try:
            row = _walk_standalone(p, do_pso_blender=do_pb)
        except Exception:
            traceback.print_exc()
            continue
        rows.append(row)
        processed += 1
        if i % 50 == 0:
            print(f"  [stand {i+1}/{len(standalones)}] {p.name}", file=sys.stderr)

    summary.total_assets = len(rows)
    for row in rows:
        summary.by_format[row.detected_format] = summary.by_format.get(row.detected_format, 0) + 1
        cls = row.failure_class or "ok"
        summary.by_failure_class[cls] = summary.by_failure_class.get(cls, 0) + 1
        summary.inner_extensions[row.inner_ext or "(top)"] = summary.inner_extensions.get(row.inner_ext or "(top)", 0) + 1
        # Aggregate type histograms for the global tally.
        if row.chunk_type_histogram:
            for piece in row.chunk_type_histogram.split(","):
                k, v = piece.split(":")
                summary.nj_chunk_type_hist[int(k)] = summary.nj_chunk_type_hist.get(int(k), 0) + int(v)
        if row.xj_vertex_type_histogram:
            for piece in row.xj_vertex_type_histogram.split(","):
                k, v = piece.split(":")
                summary.xj_vertex_type_hist[int(k)] = summary.xj_vertex_type_hist.get(int(k), 0) + int(v)

    # CSV write
    fieldnames = [k for k in asdict(rows[0] if rows else AssetRow()).keys()]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            d = asdict(row)
            for k, v in list(d.items()):
                if isinstance(v, float):
                    d[k] = round(v, 3)
            w.writerow(d)
    print(f"# wrote {out_csv} ({len(rows)} rows)", file=sys.stderr)

    # MD summary
    md = io.StringIO()
    md.write("# PSOBB Model Coverage Audit\n\n")
    md.write(f"**Data dir:** `{data_dir}`\n\n")
    md.write(f"- BMLs audited: {summary.total_bmls}\n")
    md.write(f"- Standalone .nj/.xj/.njm: {summary.total_standalones}\n")
    md.write(f"- Total inner assets: {summary.total_assets}\n\n")

    md.write("## Format breakdown\n\n")
    md.write("| Format | Count |\n|---|---|\n")
    for fmt, n in sorted(summary.by_format.items(), key=lambda kv: -kv[1]):
        md.write(f"| {fmt or '(unset)'} | {n} |\n")
    md.write("\n")

    md.write("## Failure-class breakdown\n\n")
    md.write("| Class | Count |\n|---|---|\n")
    for cls, n in sorted(summary.by_failure_class.items(), key=lambda kv: -kv[1]):
        md.write(f"| {cls} | {n} |\n")
    md.write("\n")

    md.write("## Inner extension distribution\n\n")
    md.write("| Ext | Count |\n|---|---|\n")
    for ext, n in sorted(summary.inner_extensions.items(), key=lambda kv: -kv[1]):
        md.write(f"| `{ext}` | {n} |\n")
    md.write("\n")

    md.write("## Nj chunk-type histogram (across ALL parsed nj files)\n\n")
    md.write("Type IDs ≥76 (or ≤7 outside the recognised header-only band) are unhandled.\n\n")
    md.write("| type_id | count | recognised |\n|---|---|---|\n")
    recognised = set(range(0, 6)) | set(range(8, 10)) | set(range(17, 32)) | set(range(32, 51)) | set(range(56, 59)) | set(range(64, 76)) | {255}
    for k, v in sorted(summary.nj_chunk_type_hist.items()):
        ok = "yes" if k in recognised else "**NO**"
        md.write(f"| {k} | {v} | {ok} |\n")
    md.write("\n")

    md.write("## Xj vertex-type histogram (across ALL parsed xj files)\n\n")
    md.write("Types outside 2-7 are unhandled.\n\n")
    md.write("| vertex_type | count | recognised |\n|---|---|---|\n")
    for k, v in sorted(summary.xj_vertex_type_hist.items()):
        ok = "yes" if 2 <= k <= 7 else "**NO**"
        md.write(f"| {k} | {v} | {ok} |\n")
    md.write("\n")

    # Top failure samples
    by_class: Dict[str, List[AssetRow]] = defaultdict(list)
    for row in rows:
        if row.failure_class:
            by_class[row.failure_class].append(row)

    md.write("## Failure-class samples\n\n")
    for cls, items in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
        md.write(f"### `{cls}` ({len(items)} affected)\n\n")
        for row in items[:8]:
            label = (
                f"{Path(row.container_path).name}#{row.inner_name}"
                if row.container_path else Path(row.asset_path).name
            )
            md.write(f"- `{label}` (ext={row.inner_ext}): {row.failure_detail}\n")
        md.write("\n")

    Path(args.out_md).write_text(md.getvalue(), encoding="utf-8")
    print(f"# wrote {args.out_md}", file=sys.stderr)
    print(f"# done. {summary.total_assets} rows, {sum(1 for r in rows if r.failure_class)} failures", file=sys.stderr)


if __name__ == "__main__":
    main()
