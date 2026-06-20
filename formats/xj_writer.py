"""Descriptor-table ``.xj`` (Phantasmal ``Xj.kt``) geometry ENCODER.

Byte-level inverse of :func:`formats.xj_descriptor.parse_xj_descriptor` —
the 44-byte ``XjModel`` descriptor-table format PSOBB Blue Burst stores
inside ``.xj`` IFF files (chunk magic ``NJCM``).

This is NOT the chunk-based Ninja-Nj format that ``formats/xj.py`` parses
(that one is inverted by ``formats/nj_writer.py``).  The two formats share
the IFF wrapper and the 52-byte ``MeshTreeNode`` shape but differ entirely
in what a non-zero ``model_offset`` points at:

  * Nj  — a 24-byte ``NjMesh`` + variable-length chunk streams.
  * Xj  — a 44-byte ``XjModel`` + flat vertex / triangle-strip / material
          descriptor tables located elsewhere in the NJCM body.

Reused VERBATIM from :mod:`formats.nj_writer`: ``encode_pof0`` /
``decode_pof0`` (the SEGA Ninja relocation-token codec),
``encode_njtl_chunk`` (texture-name list), ``_iff_chunk`` (the 4-byte
tag + u32 size framing), and the two-pass "assign every region offset,
then write" pattern.

Output IFF layout::

    [ NJTL chunk + POF0 chunk ]   (optional; only when model.njtl_names)
    NJCM chunk + POF0 chunk        (always)

All in-buffer pointer fields are stored as **body-relative byte offsets**;
the POF0 chunk lists those field locations so the loader can relocate
them (the game adds the NJCM body base to each).

NJCM body region order (each pointer = byte offset from NJCM body start)::

    [0]   root MeshTreeNode (52B), then nodes packed 52B DFS pre-order
    [..]  XjModel headers (44B each), one per node with geometry
    [..]  VertexInfo tables (16B rows)
    [..]  vertex arrays (flat, stride = vertex_size)
    [..]  strip tables (opaque, then transparent; 20B rows)
    [..]  material / RenderStateArgs tables (16B entries)
    [..]  u16 index lists
    (NJCM body padded to 4 bytes)

POF0 pointer set (the EXACT relocation set; verified against the parser
and the spec's 128-2.xj field map)::

    per node      : +4  (model_offset)   if has model
                    +44 (child_offset)   if child
                    +48 (sibling_offset) if sibling
    per XjModel   : +4  (vinfo table)    if vinfo_count > 0
                    +12 (opaque strips)  if opaque_count > 0
                    +20 (transparent)    if transparent_count > 0
    per VertexInfo: +4  (vertex_table_offset)
    per strip row : +0  (material_table_offset) ONLY if mat_count > 0
                    +8  (index_list_offset)     always

Vertex layout the encoder emits — descriptor-XJ vtype enum, position
``f32x3`` ALWAYS at offset 0::

    type 2  (stride 24): pos@0; normal f32x3 @12
    type 3  (stride 32): pos@0; normal f32x3 @12; uv f32x2 @24  <- textured+lit
    type 4  (stride 16): pos@0; pad4 @12
    type 5  (stride 24): pos@0; pad4 @12; uv f32x2 @16

UVs and normals are RAW f32 (``vec2Float`` / ``vec3Float``) — NOT the
``/256`` 8.8-fixed-point of the chunk-Nj strips.  The encoder chooses the
tightest vtype that carries the supplied attributes.

Strip generation: the on-disk format is intrinsically a D3D triangle
strip — the engine draws ``D3DPT_TRIANGLESTRIP`` and has no triangle-list
primitive type.  An input triangle LIST is emitted as one 3-index strip
row per triangle (``index_count == 3``).  This is the simplest correct
strategy and sidesteps the parser's per-triangle parity-flip: a 3-index
strip ``[i0, i1, i2]`` triangulates back to exactly ``(i0, i1, i2)``.
Winding is NOT baked into the strip order — the parser recomputes
per-triangle winding from the vertex normals
(``_tristrip_to_triangles_normal_corrected``), so the encoder only needs
correct normals.

Public API
----------
``XjVertexData``      — one authored vertex (pos, normal?, uv?).
``XjStrip``           — one strip's index list + material entries.
``XjModelData``       — one node's geometry (vertices + opaque/transparent strips).
``XjNode``            — one mesh-tree node (TRS + optional model + child/sibling links).
``XjModelFile``       — the whole authored model (nodes + optional njtl names).
``encode_xjcm_chunk`` — ``XjModelFile -> (njcm_body, pof0_ptr_offsets)``.
``encode_xj_model``   — ``XjModelFile -> bytes`` (full IFF-wrapped ``.xj``).
``build_xj_from_meshes`` — convenience: world-space submeshes -> XjModelFile.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# Reuse the SEGA Ninja relocation codec, NJTL encoder, and IFF framing
# verbatim from the Nj writer — the .xj IFF wrapper is byte-identical.
from .nj_writer import _iff_chunk, encode_njtl_chunk, encode_pof0

# Struct formats — MUST match formats/xj_descriptor.py exactly.
_MESH_TREE_NODE_FMT = "<II3f3i3fII"   # eval, model, pos3f, rot3i, scale3f, child, sib
_MESH_TREE_NODE_SIZE = struct.calcsize(_MESH_TREE_NODE_FMT)
assert _MESH_TREE_NODE_SIZE == 52, _MESH_TREE_NODE_SIZE

_XJ_MODEL_FMT = "<I 6I 4f"            # flags, 6 table ptr/count, coll x/y/z/r
_XJ_MODEL_SIZE = struct.calcsize(_XJ_MODEL_FMT)
assert _XJ_MODEL_SIZE == 44, _XJ_MODEL_SIZE

_VIT_ROW_FMT = "<hh III"             # vtype, flags, vert_table_off, vert_size, vert_count
_VIT_ROW_SIZE = struct.calcsize(_VIT_ROW_FMT)
assert _VIT_ROW_SIZE == 16, _VIT_ROW_SIZE

_STRIP_ROW_FMT = "<IIII I"           # mat_off, mat_count, idx_off, idx_count, unk
_STRIP_ROW_SIZE = struct.calcsize(_STRIP_ROW_FMT)
assert _STRIP_ROW_SIZE == 20, _STRIP_ROW_SIZE

_MAT_ENTRY_FMT = "<I 3I"             # type, arg0, arg1, arg2
_MAT_ENTRY_SIZE = struct.calcsize(_MAT_ENTRY_FMT)
assert _MAT_ENTRY_SIZE == 16, _MAT_ENTRY_SIZE

# Default eval_flags for an authored identity-local node: UNIT_POS |
# UNIT_ANG | UNIT_SCL | BREAK (0x17).  Matches the eval_flags PSOBB sets
# on n.rel mesh-tree nodes and keeps the parser from applying any local
# transform, so authored world-space positions survive round-trip 1:1.
EVAL_UNIT_POS = 0x01
EVAL_UNIT_ANG = 0x02
EVAL_UNIT_SCL = 0x04
EVAL_BREAK = 0x10
DEFAULT_NODE_EVAL_FLAGS = EVAL_UNIT_POS | EVAL_UNIT_ANG | EVAL_UNIT_SCL | EVAL_BREAK

# Vertex-info table types and their byte maps (see module header). Each
# entry: (stride, normal_offset_or_None, uv_offset_or_None).
_VTYPE_LAYOUTS = {
    2: (24, 12, None),
    3: (32, 12, 24),
    4: (16, None, None),
    5: (24, None, 16),
}


class XjWriteError(ValueError):
    """Raised when an authored model cannot be serialised into a valid .xj."""


# ---------------------------------------------------------------------------
# Public model dataclasses
# ---------------------------------------------------------------------------


@dataclass
class XjVertexData:
    """One authored vertex.

    ``pos`` is required.  ``normal`` / ``uv`` are optional; the encoder
    picks the tightest vertex type (2/3/4/5) that carries the attributes
    present across the model's vertex set.  Components are plain floats.
    """
    pos: Tuple[float, float, float]
    normal: Optional[Tuple[float, float, float]] = None
    uv: Optional[Tuple[float, float]] = None


@dataclass
class XjMaterialEntry:
    """One 16-byte RenderStateArgs entry (type + 3 args).

    type=2 {dst_alpha, src_alpha}; type=3 {texture_id in arg0}; type=5
    {RGBA8 diffuse in arg0}.  The encoder writes author-supplied entries
    verbatim — it does not synthesise or validate semantics for the
    unmapped types (0,1,4,6,7,8).
    """
    type: int
    args: Tuple[int, int, int] = (0, 0, 0)


@dataclass
class XjStrip:
    """One triangle-strip row.

    ``indices`` are u16 strip indices into the model's flat vertex array
    (D3D triangle-strip order, no 0xFFFF restart).  ``materials`` is the
    full author-supplied RenderStateArgs entry list for this strip (a
    textured strip needs a type-3 entry); an empty list is legal and
    inherits prior GPU state in-engine.
    """
    indices: List[int] = field(default_factory=list)
    materials: List[XjMaterialEntry] = field(default_factory=list)


@dataclass
class XjModelData:
    """One node's geometry: a flat vertex array + opaque/transparent strips.

    The encoder writes a single VertexInfo table referencing one vertex
    array; strips index into it.  ``collision`` is the (x, y, z, r)
    bounding sphere — derived from the vertices when left ``None``.
    """
    vertices: List[XjVertexData] = field(default_factory=list)
    opaque_strips: List[XjStrip] = field(default_factory=list)
    transparent_strips: List[XjStrip] = field(default_factory=list)
    collision: Optional[Tuple[float, float, float, float]] = None


@dataclass
class XjNode:
    """One 52-byte mesh-tree node.

    ``model_index`` indexes into ``XjModelFile.models`` (-1 = no geometry).
    ``child_index`` / ``sibling_index`` index into ``XjModelFile.nodes``
    (-1 = none).  Default eval_flags keep the node identity-local so
    authored world-space vertices survive round-trip.
    """
    eval_flags: int = DEFAULT_NODE_EVAL_FLAGS
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_bams: Tuple[int, int, int] = (0, 0, 0)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    model_index: int = -1
    child_index: int = -1
    sibling_index: int = -1


@dataclass
class XjModelFile:
    """A complete authored descriptor-XJ model.

    ``nodes`` are in DFS pre-order (node 0 is the root).  ``models`` is
    the geometry pool nodes reference by ``model_index``.  ``njtl_names``,
    when non-empty, emits a leading NJTL chunk.
    """
    nodes: List[XjNode] = field(default_factory=list)
    models: List[XjModelData] = field(default_factory=list)
    njtl_names: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _choose_vtype(verts: Sequence[XjVertexData]) -> int:
    """Pick the tightest vertex type that carries every supplied attribute.

    pos+normal+uv -> 3; pos+normal -> 2; pos+uv -> 5; pos-only -> 4.
    A vertex set is treated as carrying an attribute if ANY vertex has it
    (the encoder fills missing per-vertex attrs with zeros so the stride
    stays uniform).
    """
    has_normal = any(v.normal is not None for v in verts)
    has_uv = any(v.uv is not None for v in verts)
    if has_normal and has_uv:
        return 3
    if has_normal:
        return 2
    if has_uv:
        return 5
    return 4


def _derive_sphere(
    verts: Sequence[XjVertexData],
) -> Tuple[float, float, float, float]:
    """AABB-center + enclosing radius — matches the parser's bounding sphere."""
    if not verts:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [v.pos[0] for v in verts]
    ys = [v.pos[1] for v in verts]
    zs = [v.pos[2] for v in verts]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    cz = (min(zs) + max(zs)) / 2.0
    r2 = 0.0
    for v in verts:
        dx = v.pos[0] - cx
        dy = v.pos[1] - cy
        dz = v.pos[2] - cz
        d2 = dx * dx + dy * dy + dz * dz
        if d2 > r2:
            r2 = d2
    return (cx, cy, cz, math.sqrt(r2) if r2 > 0 else 0.0)


def _pack_vertex_array(verts: Sequence[XjVertexData], vtype: int) -> bytes:
    """Pack a flat vertex array for ``vtype`` (stride from _VTYPE_LAYOUTS)."""
    stride, n_off, uv_off = _VTYPE_LAYOUTS[vtype]
    out = bytearray(stride * len(verts))
    for i, v in enumerate(verts):
        base = i * stride
        px, py, pz = v.pos
        struct.pack_into("<3f", out, base, float(px), float(py), float(pz))
        if n_off is not None:
            nx, ny, nz = v.normal if v.normal is not None else (0.0, 0.0, 0.0)
            struct.pack_into("<3f", out, base + n_off,
                             float(nx), float(ny), float(nz))
        if uv_off is not None:
            u, w = v.uv if v.uv is not None else (0.0, 0.0)
            struct.pack_into("<2f", out, base + uv_off, float(u), float(w))
    return bytes(out)


# ---------------------------------------------------------------------------
# NJCM body encoder (two-pass)
# ---------------------------------------------------------------------------


def encode_xjcm_chunk(model: XjModelFile) -> Tuple[bytes, List[int]]:
    """Encode the NJCM chunk body for ``model``.

    Returns ``(body_bytes, ptr_offsets)`` where ``ptr_offsets`` are the
    body-relative byte offsets of every pointer field (sorted ascending,
    4-byte aligned) — ready for ``encode_pof0``.

    Two passes: PASS A assigns every region a body-relative offset; PASS B
    packs the bytes using the resolved offsets and records the pointer
    field locations.
    """
    nodes = model.nodes
    if not nodes:
        raise XjWriteError("encode_xjcm_chunk: model has no nodes")
    for n in nodes:
        if n.model_index >= len(model.models):
            raise XjWriteError(
                f"node model_index {n.model_index} out of range "
                f"({len(model.models)} models)")
        if n.child_index >= len(nodes) or n.sibling_index >= len(nodes):
            raise XjWriteError("node child/sibling index out of range")

    n_nodes = len(nodes)

    # ---- PASS A: assign offsets ----
    node_offsets = [i * _MESH_TREE_NODE_SIZE for i in range(n_nodes)]
    cursor = n_nodes * _MESH_TREE_NODE_SIZE

    # Per node-with-geometry: a 44B XjModel header.  Map model_index ->
    # header offset (a model referenced by >1 node still gets one header,
    # keyed by model index).
    model_header_off: dict = {}
    # Per used model: the chosen vtype + packed vertex bytes (computed
    # once, reused in PASS B).
    model_vtype: dict = {}
    model_vbytes: dict = {}
    for n in nodes:
        mi = n.model_index
        if mi < 0 or mi in model_header_off:
            continue
        model_header_off[mi] = cursor
        cursor += _XJ_MODEL_SIZE
        md = model.models[mi]
        vt = _choose_vtype(md.vertices)
        model_vtype[mi] = vt
        model_vbytes[mi] = _pack_vertex_array(md.vertices, vt)

    # Per used model, in deterministic (header-offset) order: VertexInfo
    # table (1 row) -> vertex array -> opaque strip table -> transparent
    # strip table -> material tables -> index lists.
    used_models = sorted(model_header_off, key=lambda mi: model_header_off[mi])

    model_vinfo_off: dict = {}
    model_varray_off: dict = {}
    model_opaque_tbl_off: dict = {}
    model_transp_tbl_off: dict = {}
    # Per (model_index, kind, strip_index) -> (mat_off, idx_off).
    strip_mat_off: dict = {}
    strip_idx_off: dict = {}

    # 1) VertexInfo tables + vertex arrays.
    for mi in used_models:
        md = model.models[mi]
        model_vinfo_off[mi] = cursor
        cursor += _VIT_ROW_SIZE
        model_varray_off[mi] = cursor
        cursor += len(model_vbytes[mi])

    # 2) strip tables (opaque, then transparent).
    for mi in used_models:
        md = model.models[mi]
        if md.opaque_strips:
            model_opaque_tbl_off[mi] = cursor
            cursor += _STRIP_ROW_SIZE * len(md.opaque_strips)
        if md.transparent_strips:
            model_transp_tbl_off[mi] = cursor
            cursor += _STRIP_ROW_SIZE * len(md.transparent_strips)

    # 3) material tables, then index lists (per strip, both kinds).
    for mi in used_models:
        md = model.models[mi]
        for kind, strips in (("o", md.opaque_strips), ("t", md.transparent_strips)):
            for si, strip in enumerate(strips):
                if strip.materials:
                    strip_mat_off[(mi, kind, si)] = cursor
                    cursor += _MAT_ENTRY_SIZE * len(strip.materials)
    for mi in used_models:
        md = model.models[mi]
        for kind, strips in (("o", md.opaque_strips), ("t", md.transparent_strips)):
            for si, strip in enumerate(strips):
                strip_idx_off[(mi, kind, si)] = cursor
                cursor += 2 * len(strip.indices)

    body_size = (cursor + 3) & ~3  # pad to 4

    # ---- PASS B: write ----
    body = bytearray(body_size)
    ptr_offsets: List[int] = []

    # Nodes.
    for i, n in enumerate(nodes):
        n_off = node_offsets[i]
        model_off = model_header_off[n.model_index] if n.model_index >= 0 else 0
        child_off = node_offsets[n.child_index] if n.child_index >= 0 else 0
        sib_off = node_offsets[n.sibling_index] if n.sibling_index >= 0 else 0
        struct.pack_into(
            _MESH_TREE_NODE_FMT, body, n_off,
            n.eval_flags & 0xFFFFFFFF,
            model_off,
            float(n.position[0]), float(n.position[1]), float(n.position[2]),
            int(n.rotation_bams[0]), int(n.rotation_bams[1]), int(n.rotation_bams[2]),
            float(n.scale[0]), float(n.scale[1]), float(n.scale[2]),
            child_off,
            sib_off,
        )
        if n.model_index >= 0:
            ptr_offsets.append(n_off + 4)
        if n.child_index >= 0:
            ptr_offsets.append(n_off + 44)
        if n.sibling_index >= 0:
            ptr_offsets.append(n_off + 48)

    # XjModel headers.
    for mi in used_models:
        md = model.models[mi]
        hdr_off = model_header_off[mi]
        opaque_count = len(md.opaque_strips)
        transp_count = len(md.transparent_strips)
        vinfo_off = model_vinfo_off[mi]
        opaque_tbl = model_opaque_tbl_off.get(mi, 0)
        transp_tbl = model_transp_tbl_off.get(mi, 0)
        coll = md.collision if md.collision is not None else _derive_sphere(md.vertices)
        struct.pack_into(
            _XJ_MODEL_FMT, body, hdr_off,
            0,                       # flags
            vinfo_off, 1,            # vinfo table off + count (always 1 table)
            opaque_tbl, opaque_count,
            transp_tbl, transp_count,
            float(coll[0]), float(coll[1]), float(coll[2]), float(coll[3]),
        )
        ptr_offsets.append(hdr_off + 4)        # vinfo table (count always >0)
        if opaque_count > 0:
            ptr_offsets.append(hdr_off + 12)   # opaque strip table
        if transp_count > 0:
            ptr_offsets.append(hdr_off + 20)   # transparent strip table

    # VertexInfo rows + vertex arrays.
    for mi in used_models:
        md = model.models[mi]
        vt = model_vtype[mi]
        stride = _VTYPE_LAYOUTS[vt][0]
        vinfo_off = model_vinfo_off[mi]
        varr_off = model_varray_off[mi]
        struct.pack_into(
            _VIT_ROW_FMT, body, vinfo_off,
            vt, 0, varr_off, stride, len(md.vertices),
        )
        ptr_offsets.append(vinfo_off + 4)      # vertex_table_offset
        vbytes = model_vbytes[mi]
        body[varr_off:varr_off + len(vbytes)] = vbytes

    # Strip rows.
    for mi in used_models:
        md = model.models[mi]
        for kind, strips, tbl_map in (
            ("o", md.opaque_strips, model_opaque_tbl_off),
            ("t", md.transparent_strips, model_transp_tbl_off),
        ):
            if not strips:
                continue
            tbl_off = tbl_map[mi]
            for si, strip in enumerate(strips):
                row_off = tbl_off + si * _STRIP_ROW_SIZE
                mat_count = len(strip.materials)
                mat_off = strip_mat_off.get((mi, kind, si), 0)
                idx_off = strip_idx_off[(mi, kind, si)]
                struct.pack_into(
                    _STRIP_ROW_FMT, body, row_off,
                    mat_off, mat_count, idx_off, len(strip.indices), 0,
                )
                if mat_count > 0:
                    ptr_offsets.append(row_off + 0)   # material_table_offset
                ptr_offsets.append(row_off + 8)       # index_list_offset

    # Material tables.
    for (mi, kind, si), mat_off in strip_mat_off.items():
        md = model.models[mi]
        strips = md.opaque_strips if kind == "o" else md.transparent_strips
        for j, m in enumerate(strips[si].materials):
            struct.pack_into(
                _MAT_ENTRY_FMT, body, mat_off + j * _MAT_ENTRY_SIZE,
                m.type & 0xFFFFFFFF,
                m.args[0] & 0xFFFFFFFF,
                m.args[1] & 0xFFFFFFFF,
                m.args[2] & 0xFFFFFFFF,
            )

    # Index lists.
    for (mi, kind, si), idx_off in strip_idx_off.items():
        md = model.models[mi]
        strips = md.opaque_strips if kind == "o" else md.transparent_strips
        indices = strips[si].indices
        if indices:
            struct.pack_into(f"<{len(indices)}H", body, idx_off,
                             *[idx & 0xFFFF for idx in indices])

    ptr_offsets.sort()
    return bytes(body), ptr_offsets


def encode_xj_model(model: XjModelFile) -> bytes:
    """Encode ``model`` to a complete ``.xj`` IFF file.

    Layout: ``[NJTL + POF0]?`` then ``NJCM + POF0``.  The pointer fields
    inside each chunk are body-relative offsets; the POF0 chunk after it
    lists their locations for the loader.
    """
    out = bytearray()

    if model.njtl_names:
        njtl_body, njtl_ptrs = encode_njtl_chunk(model.njtl_names)
        out.extend(_iff_chunk("NJTL", njtl_body))
        out.extend(_iff_chunk("POF0", encode_pof0(njtl_ptrs)))

    njcm_body, njcm_ptrs = encode_xjcm_chunk(model)
    out.extend(_iff_chunk("NJCM", njcm_body))
    out.extend(_iff_chunk("POF0", encode_pof0(njcm_ptrs)))
    return bytes(out)


# ---------------------------------------------------------------------------
# Convenience: world-space submeshes -> XjModelFile
# ---------------------------------------------------------------------------


def build_xj_from_meshes(
    meshes: Sequence,
    *,
    njtl_names: Optional[Sequence[str]] = None,
) -> XjModelFile:
    """Build a single-node ``XjModelFile`` from world-space submeshes.

    Each input ``mesh`` must expose ``vertices`` (each with ``.pos``,
    ``.normal``, ``.uv``), an ``indices`` triangle LIST (flat groups of
    3), and ``material_id`` — i.e. the shape ``formats.xj.XjMesh`` /
    ``parse_xj_descriptor`` produces.  All submeshes are merged into one
    node with one vertex array; each input triangle becomes one 3-index
    opaque strip carrying a type-3 (TEXTURE_ID) material entry.

    A single identity-local root node (eval_flags = 0x17) is emitted so
    the authored world-space positions survive a parse round-trip 1:1.
    """
    all_verts: List[XjVertexData] = []
    strips: List[XjStrip] = []
    for mesh in meshes:
        base = len(all_verts)
        for v in mesh.vertices:
            all_verts.append(XjVertexData(
                pos=tuple(float(c) for c in v.pos),
                normal=tuple(float(c) for c in v.normal) if v.normal is not None else None,
                uv=tuple(float(c) for c in v.uv) if v.uv is not None else None,
            ))
        tex_id = int(getattr(mesh, "material_id", 0) or 0)
        if tex_id < 0:
            tex_id = 0
        idx = list(mesh.indices)
        for t in range(0, len(idx) - 2, 3):
            strips.append(XjStrip(
                indices=[base + idx[t], base + idx[t + 1], base + idx[t + 2]],
                materials=[XjMaterialEntry(type=3, args=(tex_id, 0, 0))],
            ))

    md = XjModelData(vertices=all_verts, opaque_strips=strips)
    root = XjNode(model_index=0 if all_verts else -1)
    return XjModelFile(
        nodes=[root],
        models=[md] if all_verts else [],
        njtl_names=list(njtl_names) if njtl_names else [],
    )


__all__ = [
    "XjWriteError",
    "XjVertexData",
    "XjMaterialEntry",
    "XjStrip",
    "XjModelData",
    "XjNode",
    "XjModelFile",
    "encode_xjcm_chunk",
    "encode_xj_model",
    "build_xj_from_meshes",
    "DEFAULT_NODE_EVAL_FLAGS",
]
