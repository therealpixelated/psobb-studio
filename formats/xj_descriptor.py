# Ported from MIT-licensed Phantasmal World by Daan Vanden Bosch.
# See LICENSES.md at the editor root for the verbatim MIT block.
#
# References (all MIT):
#   psolib/src/commonMain/kotlin/world/phantasmal/psolib/fileFormats/ninja/Xj.kt
#   psolib/src/commonMain/kotlin/world/phantasmal/psolib/fileFormats/ninja/Ninja.kt
#   psolib/src/commonMain/kotlin/world/phantasmal/psolib/fileFormats/ninja/NinjaObject.kt
#   psolib/src/commonMain/kotlin/world/phantasmal/psolib/fileFormats/ninja/Angle.kt
#
# This module parses the **descriptor-table XJ format** that PSOBB
# Blue Burst stores inside ``.xj`` IFF files (chunk magic ``NJCM``).
# The XJ format is the XBOX/D3D-style sibling of the chunk-based
# Ninja-Nj format that ``formats/xj.py`` parses; both files share the
# same ``NJCM`` IFF wrapper and the same 52-byte ``MeshTreeNode``
# linked-list shape, but the contents of each non-zero ``model_offset``
# differ:
#
#   - ``Nj.kt``  — model_offset → 24-byte ``NjMesh`` struct + chunk
#                 streams of variable-length vertex/strip chunks.
#   - ``Xj.kt``  — model_offset → 44-byte ``XjModel`` struct + flat
#                 vertex / triangle-strip / material descriptor tables
#                 located elsewhere in the NJCM body.
#
# Of the ~656 BML-inner models in PSOBB.IO, 263 use this descriptor-
# table format (e.g. ``bm_fe_obj_o_door01l.bml#fe_obj_o_door01l.xj``).
# The remaining ~393 use chunk-based Nj — ``formats/xj.py`` handles
# those. The dispatch happens in ``server.py``: when the inner file
# extension is ``.xj`` we call into this module; otherwise we call the
# legacy chunk parser.
#
# Layout summary (all little-endian):
#
#   IFF wrapper:
#     NJTL chunk (optional)        — texture name list; ignored.
#     NJCM chunk                   — outer model container.
#         MeshTreeNode (52 bytes, recursive linked list).
#         Each node:
#             u32  eval_flags           Ninja evaluation flags.
#             u32  model_offset         → XjModel (44 bytes), or 0.
#             f32  x, y, z              translation
#             i32  rot_x, rot_y, rot_z  Ninja-angles (BAMs)
#             f32  sx, sy, sz           scale
#             u32  child_offset
#             u32  sibling_offset
#         XjModel (44 bytes):
#             u32  flags                                  always 0 per QEdit.
#             u32  vertex_info_table_offset               → vertex info table.
#             u32  vertex_info_count                      table-row count.
#             u32  triangle_strip_table_offset            → opaque strip table.
#             u32  triangle_strip_count                   row count.
#             u32  transparent_strip_table_offset         → transparent strip table.
#             u32  transparent_strip_count                row count.
#             f32  collision_x, y, z                      bounding sphere center.
#             f32  collision_r                            bounding sphere radius.
#         VertexInfoTable row (16 bytes):
#             i16  vertex_type                            see _VERTEX_LAYOUTS
#             i16  flags                                  unused per Phantasmal
#             u32  vertex_table_offset                    → vertex array
#             u32  vertex_size                            stride in bytes
#             u32  vertex_count                           rows
#         TriangleStripTable row (20 bytes):
#             u32  material_table_offset
#             u32  material_table_size                    rows in material table
#             u32  index_list_offset
#             u32  index_count
#             u32  unk                                    typically 0; ignored.
#         Material entry (16 bytes; tagged-union, "type" is field 0):
#             type=2: u32 src_alpha, u32 dst_alpha
#             type=3: u32 texture_id
#             type=5: u8 R, u8 G, u8 B, u8 A   (diffuse color)
#             type=others: ignored.
#         Index list:
#             u16 * index_count                triangle-strip indices.
#         Vertex layouts (from Phantasmal Xj.kt):
#             type=2: f32 px,py,pz; f32 nx,ny,nz                          (24 B)
#             type=3: f32 px,py,pz; f32 nx,ny,nz; f32 u,v                 (32 B)
#             type=4: f32 px,py,pz; pad 4                                 (16 B)
#             type=5: f32 px,py,pz; pad 4; f32 u,v                        (24 B)
#             type=6: f32 px,py,pz; f32 nx,ny,nz; pad 4                   (28 B)
#             type=7: f32 px,py,pz; f32 nx,ny,nz; f32 u,v                 (36 B)
#         Other vertex types are tolerated (we read the position and skip
#         the rest of the stride).
#
# Triangle-strip semantics: each row in the strip table IS one strip;
# Phantasmal stores the indices verbatim and emits one ``XjMesh`` per
# row. We do the same and then run the indices through a strip-to-list
# converter (degenerate triangles are dropped). PSOBB's writer never
# emits the 0xFFFF strip-restart sentinel (verified by survey of all
# 92 sampled .xj inners), so we do not handle it; if a future file
# does carry one, it will produce a few degenerate triangles which
# the converter drops.
#
# Eval flags (NinjaEvaluationFlags from Phantasmal NinjaObject.kt):
#
#     bit 0  noTranslate          — ignore translation (EVAL_UNIT_POS)
#     bit 1  noRotate             — ignore rotation (EVAL_UNIT_ANG)
#     bit 2  noScale              — ignore scale (EVAL_UNIT_SCL)
#     bit 3  hidden               — do not draw this node's mesh (EVAL_HIDE)
#     bit 4  breakChildTrace      — do not recurse into children (EVAL_BREAK)
#     bit 5  zxyRotationOrder     — rotation order is ZXY (default XYZ)
#     bit 6  skip                 — skip transform (treat local M as identity)
#     bit 7  shapeSkip            — do not draw + do not recurse children
#     bit 8  clip                 — frustum-clip hint (rendering only; ignored)
#     bit 9  modifier             — modifier volume (ignored)
#
# As in ``formats/xj.py`` we honor at minimum POS/ANG/SCL/HIDE; we
# DELIBERATELY ignore BREAK / SHAPE_SKIP for traversal and rely on
# HIDE to suppress drawing. PSOBB BB data sets BREAK on intermediate
# nodes whose child links lead to real geometry, and dropping their
# subtree (the SDK semantic) loses ~100% of the model. This matches
# the empirical traversal rule in formats/xj.py for the chunk parser
# and matches Phantasmal's runtime walker.
#
# Public API:
#   XjVertex                 — re-exported from formats/xj.py (same shape).
#   XjMesh                   — re-exported from formats/xj.py (same shape).
#   parse_xj_descriptor      — NJCM payload bytes → list[XjMesh].
#   parse_xj_file            — full ``.xj`` IFF bytes → list[XjMesh].
"""Pure-Python descriptor-table ``.xj`` (Phantasmal ``Xj.kt``) reader."""
from __future__ import annotations

import math
import os
import struct
from typing import Dict, List, Optional, Tuple

from .iff import parse_iff
# Re-use the dataclass shapes defined in formats/xj.py so the
# /api/model_mesh JSON projection (`_xj_meshes_to_payload`) treats
# both parser outputs identically — same `vertices`, `indices`,
# `material_id`, `bounding_sphere`, `world_position`,
# `world_rotation_euler`, `world_scale`, `world_matrix` fields.
from .xj import XjVertex, XjMesh

# See formats/xj.py for the full rationale. Setting
# ``PSO_XJ_IGNORE_HIDE=1`` in the environment makes BOTH parsers
# disregard EVAL_HIDE / EVAL_SHAPE_SKIP, which is useful for
# debugging — though no shipping PSOBB BB model sets these flags on
# a mesh-bearing node.
_IGNORE_HIDE_DEFAULT: bool = bool(int(os.environ.get("PSO_XJ_IGNORE_HIDE", "0") or 0))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Eval flag bits — see module header. Same numerical values as the
# constants in formats/xj.py because the NjObject and XjObject share
# the NinjaEvaluationFlags layout in Phantasmal.
_EVAL_UNIT_POS = 0x01
_EVAL_UNIT_ANG = 0x02
_EVAL_UNIT_SCL = 0x04
_EVAL_HIDE = 0x08
_EVAL_BREAK = 0x10
_EVAL_ZXY_ANG = 0x20
_EVAL_SKIP = 0x40
_EVAL_SHAPE_SKIP = 0x80

# Sega Ninja "binary angle measurement" — 0x10000 == 360 degrees.
_BAMS_TO_RAD = (2.0 * math.pi) / 65536.0

# 52-byte MeshTreeNode (Phantasmal's parseSiblingObjects) — same shape
# as Nj's, intentionally. Decoded as:
#   u32  eval_flags
#   u32  model_offset
#   f32  x, y, z
#   i32  rx, ry, rz   (BAMs)
#   f32  sx, sy, sz
#   u32  child_offset
#   u32  sibling_offset
_MESH_TREE_NODE_FMT = "<II3f3i3fII"
_MESH_TREE_NODE_SIZE = struct.calcsize(_MESH_TREE_NODE_FMT)
assert _MESH_TREE_NODE_SIZE == 52, _MESH_TREE_NODE_SIZE

# 44-byte XjModel (Phantasmal's parseXjModel header). 4-byte flags
# (always 0 per QEdit) + six u32 table headers + 4-float collision
# sphere = 4 + 24 + 16 = 44.
_XJ_MODEL_FMT = "<I 6I 4f"
_XJ_MODEL_SIZE = struct.calcsize(_XJ_MODEL_FMT)
assert _XJ_MODEL_SIZE == 44, _XJ_MODEL_SIZE

# 16-byte VertexInfoTable row.
_VIT_ROW_FMT = "<hh III"
_VIT_ROW_SIZE = struct.calcsize(_VIT_ROW_FMT)
assert _VIT_ROW_SIZE == 16, _VIT_ROW_SIZE

# 20-byte TriangleStripTable row. Phantasmal actually only uses the
# first 16 bytes (4 u32s) — the trailing 4 bytes are unread. We still
# stride 20 bytes per row to match the spec.
_STRIP_ROW_FMT = "<IIII I"
_STRIP_ROW_SIZE = struct.calcsize(_STRIP_ROW_FMT)
assert _STRIP_ROW_SIZE == 20, _STRIP_ROW_SIZE

# 16-byte material entry.
_MAT_ENTRY_FMT = "<I 12s"
_MAT_ENTRY_SIZE = struct.calcsize(_MAT_ENTRY_FMT)
assert _MAT_ENTRY_SIZE == 16, _MAT_ENTRY_SIZE

# Hard cap on tree-walk depth. Real PSOBB models top out around 200
# mesh-tree nodes; this bounds malicious / corrupt inputs.
_MAX_NODES = 4096


# ---------------------------------------------------------------------------
# Strip-to-triangle-list helper (shared shape with formats/xj.py).
# ---------------------------------------------------------------------------


def _tristrip_to_triangles(strip: List[int], cw: bool = False) -> List[int]:
    """Convert a triangle-strip index list to a flat triangle list.

    Drops degenerate triangles (any pair of equal indices). Honours
    the parity flip every other triangle that triangle-strip semantics
    require. ``cw`` flips the initial winding (XJ never sets this in
    practice — strips are CCW by convention — but we accept it for
    API symmetry with ``formats/xj.py``).
    """
    out: List[int] = []
    n = len(strip)
    flip_first = cw
    for i in range(n - 2):
        a = strip[i]
        b = strip[i + 1]
        c = strip[i + 2]
        if a == b or b == c or a == c:
            continue
        if (i & 1) ^ (1 if flip_first else 0):
            out.extend((a, c, b))
        else:
            out.extend((a, b, c))
    return out


def _tristrip_to_triangles_normal_corrected(
    strip: List[int],
    positions: List[Tuple[float, float, float]],
    normals: List[Tuple[float, float, float]],
) -> List[int]:
    """Triangulate a strip with the **normal-correcting winding** that
    Phantasmal's ``NinjaGeometryConversion.kt::convertXjModel`` uses.

    This is a faithful port of the XJ triangulation loop (the ``.xj``
    descriptor format does NOT carry a per-strip winding flag the way
    the chunk-based Nj strips do, so a fixed alternating parity gets the
    winding wrong on ~16% of strips — verified by the parity sweep in
    ``tests/test_xj_parity.py`` over 316 real game ``.xj`` assets).

    Algorithm (verbatim from convertXjModel, lines 553-599 of
    ``web/src/jsMain/kotlin/.../NinjaGeometryConversion.kt``)::

        var clockwise = false
        for (j in 2 until indices.size) {
            a, b, c = indices[j-2], indices[j-1], indices[j]
            faceN   = (pb - pa) cross (pc - pa)
            if (clockwise) faceN.negate()
            // "Calculate a surface normal and reverse the vertex
            //  winding if at least 2 of the vertex normals point in
            //  the opposite direction. This hack fixes the winding for
            //  most models."
            opposite = count( dot(faceN, vertexNormal) < 0 )  over a,b,c
            if (opposite >= 2) clockwise = !clockwise
            emit( clockwise ? (b, a, c) : (a, b, c) )
            clockwise = !clockwise          // ordinary strip parity flip
        }

    Parameters
    ----------
    strip:
        Local triangle-strip indices into ``positions`` / ``normals``.
    positions:
        World-space vertex positions, one per local index. These are the
        SAME positions the strip references, AFTER the host-bone
        local-to-world bake — matching the oracle, which runs the
        winding correction on already-transformed (``builder``-space)
        positions and normal-matrix-transformed normals.
    normals:
        World-space vertex normals, one per local index. Synthesized
        up-normals ``(0, 1, 0)`` are acceptable here (the oracle uses
        the same ``Vector3(0, 1, 0)`` default); they simply never flag a
        triangle as opposite, so such strips keep the default parity.

    Returns
    -------
    list[int]
        Flat triangle index list (groups of three) into the same local
        vertex array. Degenerate triangles (a repeated index) are
        dropped from the OUTPUT, but — exactly like the oracle — the
        ``clockwise`` parity state machine still steps over them so the
        winding of subsequent triangles is unaffected. (The oracle emits
        the degenerate triangle as a zero-area face; three.js tolerates
        it. We drop it because the rest of this module's pipeline and the
        e2e ``indices`` invariants assume non-degenerate output, and a
        dropped zero-area face is visually identical.)
    """
    out: List[int] = []
    n = len(strip)
    clockwise = False
    for j in range(2, n):
        a = strip[j - 2]
        b = strip[j - 1]
        c = strip[j]

        pa = positions[a]
        pb = positions[b]
        pc = positions[c]

        # faceN = (pb - pa) x (pc - pa)
        ux = pb[0] - pa[0]; uy = pb[1] - pa[1]; uz = pb[2] - pa[2]
        vx = pc[0] - pa[0]; vy = pc[1] - pa[1]; vz = pc[2] - pa[2]
        fx = uy * vz - uz * vy
        fy = uz * vx - ux * vz
        fz = ux * vy - uy * vx
        if clockwise:
            fx = -fx; fy = -fy; fz = -fz

        na = normals[a]
        nb = normals[b]
        nc = normals[c]
        opposite = 0
        if fx * na[0] + fy * na[1] + fz * na[2] < 0.0:
            opposite += 1
        if fx * nb[0] + fy * nb[1] + fz * nb[2] < 0.0:
            opposite += 1
        if fx * nc[0] + fy * nc[1] + fz * nc[2] < 0.0:
            opposite += 1
        if opposite >= 2:
            clockwise = not clockwise

        # Emit (dropping degenerate triangles, but always stepping the
        # parity machine below so non-degenerate windings stay correct).
        if not (a == b or b == c or a == c):
            if clockwise:
                out.extend((b, a, c))
            else:
                out.extend((a, b, c))

        clockwise = not clockwise
    return out


# ---------------------------------------------------------------------------
# Matrix helpers — identical to formats/xj.py because the bone tree
# is shared between Nj and Xj. Duplicating (vs importing private
# helpers) keeps this module self-contained and lets it ship even if
# formats/xj.py's private layout shifts in a future agent edit.
# ---------------------------------------------------------------------------


def _mat4_identity() -> List[float]:
    """Return a fresh identity 4x4 row-major matrix."""
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def _mat4_mul(a: List[float], b: List[float]) -> List[float]:
    """Multiply two row-major 4x4 matrices: result = a @ b."""
    out = [0.0] * 16
    for i in range(4):
        ai0 = a[i * 4 + 0]
        ai1 = a[i * 4 + 1]
        ai2 = a[i * 4 + 2]
        ai3 = a[i * 4 + 3]
        for j in range(4):
            out[i * 4 + j] = (
                ai0 * b[0 * 4 + j]
                + ai1 * b[1 * 4 + j]
                + ai2 * b[2 * 4 + j]
                + ai3 * b[3 * 4 + j]
            )
    return out


def _mat4_compose_trs(
    pos: Tuple[float, float, float],
    rot_xyz_rad: Tuple[float, float, float],
    scale: Tuple[float, float, float],
    zxy_order: bool = False,
) -> List[float]:
    """Build T * R * S in row-major order — same convention as Nj.

    Matches Phantasmal's NjObject walker: a vector ``v`` is treated as
    a column; ``M * v`` applies M. Composition order is parent * local
    (local applies first). Default rotation order is **ZYX**
    (R = Rz*Ry*Rx — three.js' ``Euler(..., "ZYX")``); ``zxy_order=True``
    switches to ZXY (R = Rz*Rx*Ry) for nodes that set EVAL_ZXY_ANG.

    The earlier port used XYZ (R = Rx*Ry*Rz) as default; that was a
    transcription error — Phantasmal's
    ``NinjaGeometryConversion.kt::convertObject`` builds
    ``Euler(x, y, z, if (ef.zxyRotationOrder) "ZXY" else "ZYX")`` and
    three.js' "ZYX" composes Rz @ Ry @ Rx. See
    ``AGENT_MODEL_DEEP_DEBUG_REPORT.md`` for the analysis.
    """
    px, py, pz = pos
    rx, ry, rz = rot_xyz_rad
    sx, sy, sz = scale

    cx = math.cos(rx); s_x = math.sin(rx)
    cy = math.cos(ry); s_y = math.sin(ry)
    cz = math.cos(rz); s_z = math.sin(rz)

    Rx = [
        1.0, 0.0, 0.0, 0.0,
        0.0,  cx, -s_x, 0.0,
        0.0, s_x,   cx, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    Ry = [
         cy, 0.0, s_y, 0.0,
        0.0, 1.0, 0.0, 0.0,
        -s_y, 0.0,  cy, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    Rz = [
         cz, -s_z, 0.0, 0.0,
        s_z,   cz, 0.0, 0.0,
        0.0,  0.0, 1.0, 0.0,
        0.0,  0.0, 0.0, 1.0,
    ]

    if zxy_order:
        # ZXY: R = Rz * Rx * Ry  (EVAL_ZXY_ANG).
        R = _mat4_mul(_mat4_mul(Rz, Rx), Ry)
    else:
        # ZYX: R = Rz * Ry * Rx  (Phantasmal default — three.js "ZYX").
        R = _mat4_mul(_mat4_mul(Rz, Ry), Rx)

    M = list(R)
    M[0 * 4 + 3] = px
    M[1 * 4 + 3] = py
    M[2 * 4 + 3] = pz

    sxsysz1 = (sx, sy, sz, 1.0)
    for i in range(4):
        for j in range(4):
            M[i * 4 + j] *= sxsysz1[j]
    return M


def _mat4_transform_point(M: List[float], p: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Apply M (row-major 4x4) to a 3D point (homogeneous w=1)."""
    px, py, pz = p
    out_x = M[0 * 4 + 0] * px + M[0 * 4 + 1] * py + M[0 * 4 + 2] * pz + M[0 * 4 + 3]
    out_y = M[1 * 4 + 0] * px + M[1 * 4 + 1] * py + M[1 * 4 + 2] * pz + M[1 * 4 + 3]
    out_z = M[2 * 4 + 0] * px + M[2 * 4 + 1] * py + M[2 * 4 + 2] * pz + M[2 * 4 + 3]
    return out_x, out_y, out_z


def _mat4_transform_dir(M: List[float], d: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Apply M's upper-3x3 to a 3D direction (no translation, no normalize)."""
    dx, dy, dz = d
    out_x = M[0 * 4 + 0] * dx + M[0 * 4 + 1] * dy + M[0 * 4 + 2] * dz
    out_y = M[1 * 4 + 0] * dx + M[1 * 4 + 1] * dy + M[1 * 4 + 2] * dz
    out_z = M[2 * 4 + 0] * dx + M[2 * 4 + 1] * dy + M[2 * 4 + 2] * dz
    return out_x, out_y, out_z


def _decompose_world_xform(
    M: List[float],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    """Decompose a row-major 4x4 ``M = T * R * S`` into (pos, euler_xyz, scale).

    Same routine as ``formats/xj._decompose_world_xform`` — duplicated to
    keep this module independent of that one's private surface.
    """
    px = M[0 * 4 + 3]
    py = M[1 * 4 + 3]
    pz = M[2 * 4 + 3]

    def _col(i):
        return (M[0 * 4 + i], M[1 * 4 + i], M[2 * 4 + i])
    c0 = _col(0); c1 = _col(1); c2 = _col(2)
    sx = math.sqrt(c0[0] * c0[0] + c0[1] * c0[1] + c0[2] * c0[2])
    sy = math.sqrt(c1[0] * c1[0] + c1[1] * c1[1] + c1[2] * c1[2])
    sz = math.sqrt(c2[0] * c2[0] + c2[1] * c2[1] + c2[2] * c2[2])

    inv_sx = 1.0 / sx if sx > 1e-12 else 0.0
    inv_sy = 1.0 / sy if sy > 1e-12 else 0.0
    inv_sz = 1.0 / sz if sz > 1e-12 else 0.0

    r00 = c0[0] * inv_sx; r10 = c0[1] * inv_sx; r20 = c0[2] * inv_sx
    r01 = c1[0] * inv_sy; r11 = c1[1] * inv_sy; r21 = c1[2] * inv_sy
    r02 = c2[0] * inv_sz; r12 = c2[1] * inv_sz; r22 = c2[2] * inv_sz

    s = max(-1.0, min(1.0, -r02))
    ry = math.asin(s)
    if abs(r02) < 1.0 - 1e-6:
        rx = math.atan2(r12, r22)
        rz = math.atan2(r01, r00)
    else:
        rx = math.atan2(-r21, r11)
        rz = 0.0

    return (px, py, pz), (rx, ry, rz), (sx, sy, sz)


# ---------------------------------------------------------------------------
# Internal: descriptor parsers (Phantasmal Xj.kt port)
# ---------------------------------------------------------------------------


def _parse_vertex_info_tables(
    body: bytes,
    table_off: int,
    count: int,
) -> List[Tuple[int, List[Tuple[Tuple[float, float, float],
                                Optional[Tuple[float, float, float]],
                                Optional[Tuple[float, float]]]]]]:
    """Parse all rows of the vertex info table.

    Phantasmal's ``parseVertexInfoTable`` reads only the first row
    (``// TODO: parse all vertex info tables.``) but we walk all rows
    because PSOBB ships .xj files with up to 2 vertex tables (e.g.
    ``bm_eff_ice.bml#ice_root.xj``). Each table contributes vertices
    to a SHARED, monotonically-growing slot table — strips reference
    slots by absolute index, and the second table's vertices live at
    indices ``count[0]..count[0]+count[1]-1``. This is the behavior
    Phantasmal would have when the TODO is resolved (verified against
    the .xj fixtures: with only the first table, files with count=2
    miss ~30% of their vertices and produce broken strip indices).

    Returns
    -------
    list of (base_index, vertices)
        ``base_index`` is the absolute slot index of the FIRST vertex
        in the table (cumulative sum across previous tables). Each
        ``vertex`` is ``(pos, normal_or_None, uv_or_None)``.
    """
    out: List = []
    if count <= 0 or table_off <= 0:
        return out
    n = len(body)
    cumulative = 0
    for i in range(count):
        row_off = table_off + i * _VIT_ROW_SIZE
        if row_off + _VIT_ROW_SIZE > n:
            break
        vt, _flags, vto, vsize, vcount = struct.unpack_from(_VIT_ROW_FMT, body, row_off)
        if vcount <= 0 or vto <= 0 or vsize <= 0:
            out.append((cumulative, []))
            continue
        verts = _parse_vertex_array(body, vto, vsize, vcount, vt)
        out.append((cumulative, verts))
        cumulative += vcount
    return out


# Vertex layouts — Phantasmal's Xj.kt parseVertexInfoTable when-block.
# Each entry is ``(parse_fn, expected_size)``. ``parse_fn`` reads from
# ``body`` at a given offset and returns ``(normal_or_None, uv_or_None)``
# (the position is read separately because every layout starts with it).
def _vlayout_2(body: bytes, off: int, end: int):
    if off + 24 > end:
        return None, None
    nx, ny, nz = struct.unpack_from("<3f", body, off + 12)
    return (nx, ny, nz), None


def _vlayout_3(body: bytes, off: int, end: int):
    if off + 32 > end:
        return None, None
    nx, ny, nz = struct.unpack_from("<3f", body, off + 12)
    u, v = struct.unpack_from("<2f", body, off + 24)
    return (nx, ny, nz), (u, v)


def _vlayout_4(body: bytes, off: int, end: int):
    # pos + skip 4
    return None, None


def _vlayout_5(body: bytes, off: int, end: int):
    if off + 24 > end:
        return None, None
    u, v = struct.unpack_from("<2f", body, off + 16)
    return None, (u, v)


def _vlayout_6(body: bytes, off: int, end: int):
    if off + 24 > end:
        return None, None
    nx, ny, nz = struct.unpack_from("<3f", body, off + 12)
    return (nx, ny, nz), None


def _vlayout_7(body: bytes, off: int, end: int):
    if off + 36 > end:
        return None, None
    nx, ny, nz = struct.unpack_from("<3f", body, off + 12)
    u, v = struct.unpack_from("<2f", body, off + 24)
    return (nx, ny, nz), (u, v)


_VERTEX_LAYOUTS = {
    2: _vlayout_2,
    3: _vlayout_3,
    4: _vlayout_4,
    5: _vlayout_5,
    6: _vlayout_6,
    7: _vlayout_7,
}


def _parse_vertex_array(
    body: bytes,
    table_off: int,
    stride: int,
    count: int,
    vertex_type: int,
) -> List[Tuple[Tuple[float, float, float],
                Optional[Tuple[float, float, float]],
                Optional[Tuple[float, float]]]]:
    """Read ``count`` vertices of ``stride`` bytes from ``body[table_off..]``.

    Position is always at the start of each row (3 floats). The rest
    of the row is interpreted by ``vertex_type`` per the Phantasmal
    Xj.kt vertex-type when-block — see _VERTEX_LAYOUTS above. Unknown
    types are tolerated: position is read, normal/UV are left None.
    """
    n = len(body)
    out: List = []
    end = table_off + stride * count
    if end > n:
        # Truncate to whatever fits.
        max_full = max(0, (n - table_off) // stride)
        count = min(count, max_full)
    layout = _VERTEX_LAYOUTS.get(vertex_type)
    for i in range(count):
        off = table_off + i * stride
        if off + 12 > n:
            break
        x, y, z = struct.unpack_from("<3f", body, off)
        normal: Optional[Tuple[float, float, float]] = None
        uv: Optional[Tuple[float, float]] = None
        if layout is not None:
            normal, uv = layout(body, off, n)
        out.append(((x, y, z), normal, uv))
    return out


def _parse_material(body: bytes, mat_off: int, mat_size: int) -> Optional[int]:
    """Walk the material entry list and return the first texture_id seen.

    Each entry is 16 bytes; the first u32 is the entry "type". Per
    Phantasmal's ``parseTriangleStripMaterial``:
        type=2: src_alpha (u32), dst_alpha (u32)
        type=3: texture_id (u32)
        type=5: diffuse R/G/B/A (u8 each)
    Other types are ignored. We only care about the texture id for the
    XjMesh's ``material_id`` field; the JSON wire schema doesn't carry
    src/dst alpha or diffuse color today (it's a TODO for both Nj and
    Xj parsers).
    """
    if mat_size <= 0 or mat_off <= 0:
        return None
    n = len(body)
    for i in range(mat_size):
        off = mat_off + i * _MAT_ENTRY_SIZE
        if off + _MAT_ENTRY_SIZE > n:
            break
        (entry_type,) = struct.unpack_from("<I", body, off)
        if entry_type == 3:
            (tex_id,) = struct.unpack_from("<I", body, off + 4)
            return tex_id
    return None


def _parse_strip_table(
    body: bytes,
    strip_table_off: int,
    strip_count: int,
) -> List[Tuple[Optional[int], List[int]]]:
    """Read all strips from one strip table; return ``[(texture_id, indices), ...]``.

    Phantasmal's ``parseTriangleStripTable`` (Xj.kt) emits one ``XjMesh``
    per row; we mirror that. Each row has a 20-byte header pointing to
    a material table + an index list. Indices are u16 triangle-strip
    indices into the global vertex slot table.
    """
    out: List[Tuple[Optional[int], List[int]]] = []
    if strip_count <= 0 or strip_table_off <= 0:
        return out
    n = len(body)
    for i in range(strip_count):
        row_off = strip_table_off + i * _STRIP_ROW_SIZE
        if row_off + _STRIP_ROW_SIZE > n:
            break
        mat_off, mat_size, idx_off, idx_count, _unk = struct.unpack_from(
            _STRIP_ROW_FMT, body, row_off,
        )
        tex_id = _parse_material(body, mat_off, mat_size)

        if idx_count <= 0 or idx_off <= 0 or idx_off + 2 * idx_count > n:
            indices: List[int] = []
        else:
            indices = list(struct.unpack_from(f"<{idx_count}H", body, idx_off))
        out.append((tex_id, indices))
    return out


# ---------------------------------------------------------------------------
# Internal: tree walk
# ---------------------------------------------------------------------------


def _read_mesh_tree_node(body: bytes, off: int):
    """Read all 13 fields of a 52-byte MeshTreeNode at ``off``.

    Returns ``(eval_flags, model_offset, pos, rot_bams, scale,
    child_offset, sibling_offset)`` or ``None`` on truncation.
    """
    if off < 0 or off + _MESH_TREE_NODE_SIZE > len(body):
        return None
    f = struct.unpack_from(_MESH_TREE_NODE_FMT, body, off)
    eval_flags = f[0]
    model_offset = f[1]
    pos = (f[2], f[3], f[4])
    rot = (f[5], f[6], f[7])
    scale = (f[8], f[9], f[10])
    child_offset = f[11]
    sibling_offset = f[12]
    return eval_flags, model_offset, pos, rot, scale, child_offset, sibling_offset


def _walk_tree(body: bytes, root_off: int = 0) -> List[Tuple[int, int, List[float], int]]:
    """Pre-order DFS the MeshTreeNode hierarchy.

    Returns ``[(node_off, model_offset, world_matrix, eval_flags), ...]``
    for every node that has a non-zero ``model_offset``. Honors
    EVAL_UNIT_POS/ANG/SCL/SKIP/ZXY_ANG when composing the local matrix;
    DELIBERATELY ignores BREAK and SHAPE_SKIP for traversal — see
    module header for the rationale (they are render-state metadata in
    PSOBB BB data, not topology gates). HIDE / SHAPE_SKIP are surfaced
    via the returned ``eval_flags`` so the caller can skip drawing
    those nodes.
    """
    n = len(body)
    out: List[Tuple[int, int, List[float], int]] = []
    visited: set = set()
    # Stack of (node_off, parent_world_matrix). LIFO; we push siblings
    # FIRST so children pop next (preserves pre-order: parent → child →
    # sibling).
    stack: List[Tuple[int, List[float]]] = [(root_off, _mat4_identity())]

    while stack and len(out) < _MAX_NODES:
        off, parent_world = stack.pop()
        if off in visited or off + _MESH_TREE_NODE_SIZE > n:
            continue
        visited.add(off)
        node = _read_mesh_tree_node(body, off)
        if node is None:
            continue
        ef, model_offset, pos, rot_bams, scale, child_off, next_off = node

        if ef & _EVAL_SKIP:
            local_M = _mat4_identity()
        else:
            lpos = (0.0, 0.0, 0.0) if (ef & _EVAL_UNIT_POS) else pos
            if ef & _EVAL_UNIT_ANG:
                lrot = (0.0, 0.0, 0.0)
            else:
                lrot = (
                    rot_bams[0] * _BAMS_TO_RAD,
                    rot_bams[1] * _BAMS_TO_RAD,
                    rot_bams[2] * _BAMS_TO_RAD,
                )
            lscale = (1.0, 1.0, 1.0) if (ef & _EVAL_UNIT_SCL) else scale
            zxy = bool(ef & _EVAL_ZXY_ANG)
            local_M = _mat4_compose_trs(lpos, lrot, lscale, zxy_order=zxy)

        world_M = _mat4_mul(parent_world, local_M)

        if model_offset and model_offset + _XJ_MODEL_SIZE <= n:
            out.append((off, model_offset, world_M, ef))

        # Push sibling first so child is popped first (LIFO + pre-order).
        if next_off and next_off not in visited:
            stack.append((next_off, parent_world))
        if child_off and child_off not in visited:
            stack.append((child_off, world_M))

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_xj_descriptor(
    payload: bytes,
    *,
    ignore_hide: Optional[bool] = None,
) -> List[XjMesh]:
    """Parse the body of a single XJ-format ``NJCM`` chunk.

    Parameters
    ----------
    payload:
        The chunk PAYLOAD bytes (i.e. NOT including the 8-byte IFF
        header). All file-relative pointers stored inside the buffer
        are interpreted as offsets relative to the start of ``payload``.
    ignore_hide:
        When True, do NOT skip emission for nodes flagged HIDE /
        SHAPE_SKIP. Default (None) reads the env var
        ``PSO_XJ_IGNORE_HIDE``. See ``formats/xj.py`` for the rationale.

    Returns
    -------
    List[XjMesh]
        One submesh per row in each XjModel's opaque + transparent
        triangle-strip tables. Each submesh's vertices are in
        WORLD-SPACE (transformed by the owning MeshTreeNode's
        local-to-world matrix), so callers who pass these to three.js
        must NOT compose ``world_matrix`` again — set
        ``vertices_pre_transformed=True`` in the wire payload, just
        like ``formats/xj.py`` does.

    Raises
    ------
    ValueError
        On obvious corruption (truncated mesh-tree node, etc.).
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError("parse_xj_descriptor: payload must be bytes-like")
    if len(payload) < _MESH_TREE_NODE_SIZE:
        raise ValueError(
            f"parse_xj_descriptor: payload too small "
            f"({len(payload)} bytes < {_MESH_TREE_NODE_SIZE})"
        )

    body = bytes(payload)

    # Sanity-check the root MeshTreeNode looks plausible.
    root = _read_mesh_tree_node(body, 0)
    if root is None:
        raise ValueError("parse_xj_descriptor: root mesh tree node truncated")

    try:
        nodes = _walk_tree(body, root_off=0)
    except struct.error as e:
        raise ValueError(f"parse_xj_descriptor: structure read failed: {e}")
    except (IndexError, KeyError) as e:
        raise ValueError(f"parse_xj_descriptor: corrupt offsets: {e}")

    if ignore_hide is None:
        ignore_hide = _IGNORE_HIDE_DEFAULT
    out_meshes: List[XjMesh] = []
    # Stateful texture-id tracker. Each strip row may carry an empty
    # material entry list (mat_size=0 / mat_off=0); in PSOBB's renderer
    # those inherit the previously-set GPU state. Mirror that here so a
    # parser-level `material_id` lookup yields the same id the engine
    # would have rendered the strip with. Reset to -1 (== "no texture
    # bound yet"; emitted as material_id 0 fallback below) at the start
    # of every NJCM payload — render state does not leak across separate
    # NJCM blocks.
    #
    # Empirically observed in `fe_obj_kaifuku_moto.xj` etc.: the FIRST
    # strip carries a type-3 entry setting tex_id=0, then 17 strips with
    # empty entries that share that 0, then strip 18 carries type-3
    # tex_id=5 (5 sticky strips), and so on through 6 distinct ids
    # spread across 94 submeshes.
    last_tex_id: Optional[int] = None
    for (_off, model_offset, world_M, ef) in nodes:
        # HIDE / SHAPE_SKIP suppress drawing for this node only.
        # Children were already enqueued during the DFS regardless of
        # SHAPE_SKIP (see module header). Note: EVAL_BREAK is
        # intentionally NOT honored as a topology gate here.
        # ignore_hide=True (env: PSO_XJ_IGNORE_HIDE=1) bypasses this.
        if not ignore_hide and (ef & (_EVAL_HIDE | _EVAL_SHAPE_SKIP)):
            continue

        # Read XjModel header.
        try:
            xm = struct.unpack_from(_XJ_MODEL_FMT, body, model_offset)
        except struct.error:
            continue
        (
            _flags,
            vbi_off, vbi_count,
            ts_off, ts_count,
            tts_off, tts_count,
            _coll_x, _coll_y, _coll_z, _coll_r,
        ) = xm

        # Vertex slot table (concatenation of all vertex info table rows).
        # Phantasmal's TODO: walk every vertex info table; we do, because
        # PSOBB has multi-table .xj files in the wild (see survey notes
        # in module header).
        vit_blocks = _parse_vertex_info_tables(body, vbi_off, vbi_count)
        # Flatten to a single list keyed by absolute slot index.
        slots: Dict[int, Tuple[Tuple[float, float, float],
                               Optional[Tuple[float, float, float]],
                               Optional[Tuple[float, float]]]] = {}
        for base_index, verts in vit_blocks:
            for j, v in enumerate(verts):
                slots[base_index + j] = v
        if not slots:
            continue

        # Strip tables: opaque + transparent. Both are emitted as
        # XjMesh rows in Phantasmal — we keep them in one combined
        # list. The texture id (when present in the material list)
        # rides along on each strip entry.
        strips: List[Tuple[Optional[int], List[int]]] = []
        strips.extend(_parse_strip_table(body, ts_off, ts_count))
        strips.extend(_parse_strip_table(body, tts_off, tts_count))

        # Decompose the world matrix once for this node — every strip
        # under this node shares the same transform (host-bone tagging
        # convention from formats/xj.py).
        wp, wr, ws = _decompose_world_xform(world_M)
        wm_tuple = tuple(world_M)

        for (tex_id, strip_indices) in strips:
            if len(strip_indices) < 3:
                continue
            # Inherit previous GPU texture state when this strip's
            # material entry list is empty (mirrors PSOBB's render-state
            # stickiness — see ``last_tex_id`` declaration above).
            if tex_id is None:
                effective_tex_id = last_tex_id
            else:
                effective_tex_id = int(tex_id)
                last_tex_id = effective_tex_id

            # Locally renumber the strip's slot references so that
            # downstream consumers can treat the XjMesh's `vertices`
            # list as a tight, dense array. Skip strips that reference
            # missing slots (defensive — not observed in PSOBB.IO data
            # but cheap to guard against).
            local_verts: List[XjVertex] = []
            local_slot_map: Dict[int, int] = {}
            local_strip: List[int] = []
            valid = True
            for sidx in strip_indices:
                if sidx not in slots:
                    valid = False
                    break
                if sidx in local_slot_map:
                    local_strip.append(local_slot_map[sidx])
                    continue
                pos_local, normal_local, uv_local = slots[sidx]
                # Bake the vertex into world space using the host
                # node's local-to-world matrix. Matches the contract
                # established by formats/xj.py (the JS frontend reads
                # vertex data verbatim and only applies the camera /
                # editor transform on top).
                pos_world = _mat4_transform_point(world_M, pos_local)
                if normal_local is not None:
                    normal_world = _mat4_transform_dir(world_M, normal_local)
                else:
                    # Synthesize an up-pointing normal — same fallback
                    # as the chunk parser. A future improvement would
                    # be to emit per-strip face normals, but PSOBB
                    # almost always carries normals in the vertex
                    # table so this branch is rare.
                    normal_world = (0.0, 1.0, 0.0)
                uv = uv_local if uv_local is not None else (0.0, 0.0)
                local_slot_map[sidx] = len(local_verts)
                local_verts.append(XjVertex(
                    pos=pos_world,
                    normal=normal_world,
                    uv=uv,
                ))
                local_strip.append(local_slot_map[sidx])

            if not valid or not local_verts:
                continue

            # Normal-correcting winding (Phantasmal convertXjModel). The
            # XJ descriptor format carries no per-strip winding flag, so
            # a fixed alternating parity gets ~16% of strips backwards
            # (verified over 316 real .xj assets — see
            # tests/test_xj_parity.py). We feed the WORLD-SPACE positions
            # and normals we just baked into local_verts, matching the
            # oracle which corrects winding in builder/world space.
            _positions = [v.pos for v in local_verts]
            _normals = [v.normal for v in local_verts]
            tri_indices = _tristrip_to_triangles_normal_corrected(
                local_strip, _positions, _normals,
            )
            if not tri_indices:
                continue

            # Bounding sphere (axis-aligned-bbox-center, radius =
            # max-distance) — same simple approximation as Nj.
            xs = [v.pos[0] for v in local_verts]
            ys = [v.pos[1] for v in local_verts]
            zs = [v.pos[2] for v in local_verts]
            cx = (min(xs) + max(xs)) / 2.0
            cy = (min(ys) + max(ys)) / 2.0
            cz = (min(zs) + max(zs)) / 2.0
            r2 = 0.0
            for v in local_verts:
                dx = v.pos[0] - cx
                dy = v.pos[1] - cy
                dz = v.pos[2] - cz
                d2 = dx * dx + dy * dy + dz * dz
                if d2 > r2:
                    r2 = d2
            r = math.sqrt(r2) if r2 > 0 else 0.0

            out_meshes.append(XjMesh(
                vertices=local_verts,
                indices=tri_indices,
                material_id=int(effective_tex_id) if effective_tex_id is not None else 0,
                bounding_sphere=(cx, cy, cz, r),
                world_position=wp,
                world_rotation_euler=wr,
                world_scale=ws,
                world_matrix=wm_tuple,
            ))

    return out_meshes


def parse_xj_file(buf: bytes, *, ignore_hide: Optional[bool] = None) -> List[XjMesh]:
    """Parse a complete ``.xj`` IFF file (find every NJCM, dispatch).

    Parameters
    ----------
    buf:
        Full bytes of an ``.xj`` file. The file may include leading
        ``NJTL`` (texture name list) and trailing ``POF0`` chunks; we
        skip everything except ``NJCM``.
    ignore_hide:
        Forward to ``parse_xj_descriptor``; default reads
        ``PSO_XJ_IGNORE_HIDE`` env var.

    Returns
    -------
    List[XjMesh]
        Concatenated submeshes from every ``NJCM`` chunk in the file.
        Empty when the file has no ``NJCM`` chunk or no parseable
        geometry.

    Raises
    ------
    ValueError
        When the IFF wrapper is malformed.
    """
    if not isinstance(buf, (bytes, bytearray, memoryview)):
        raise ValueError("parse_xj_file: input must be bytes-like")
    chunks = parse_iff(buf)
    if not chunks:
        return []
    out: List[XjMesh] = []
    for c in chunks:
        if c.type != "NJCM":
            continue
        out.extend(parse_xj_descriptor(c.data, ignore_hide=ignore_hide))
    return out


__all__ = [
    "XjVertex",
    "XjMesh",
    "parse_xj_descriptor",
    "parse_xj_file",
]
