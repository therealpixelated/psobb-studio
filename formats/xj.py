# Ported from MIT-licensed Phantasmal World by Daan Vanden Bosch.
# See LICENSES.md at the editor root for the verbatim MIT block.
#
# References (all MIT):
#   psolib/src/commonMain/kotlin/world/phantasmal/psolib/fileFormats/ninja/Nj.kt
#   psolib/src/commonMain/kotlin/world/phantasmal/psolib/fileFormats/ninja/Xj.kt
#
# This module parses the Ninja mesh trees that PSOBB Blue Burst stores
# inside ``.nj`` IFF files (chunk magic ``NJCM``). The historical name
# of this module is ``xj.py`` — but the live PSOBB.IO data is in the
# CHUNK-BASED Ninja format (``Nj.kt``), not the descriptor-table XJ
# format (``Xj.kt``). The original author of this editor mis-labelled
# the format as "XJ"; we preserve the public type names
# (``XjVertex``, ``XjMesh``, ``parse_xj_njcm``, ``parse_nj_file``,
# ``parse_skeleton``) so that ``server.py``, the e2e suite, and the
# frontend keep compiling. Internally we run the Ninja-Nj parser.
#
# Layout summary (PSOBB Blue Burst ``.nj``, all little-endian):
#
#   NJTL chunk (optional)        — texture name list; we ignore.
#   POF0 chunk                   — pointer-fixup table; the values
#                                  stored inside NJCM are already
#                                  body-relative offsets, POF0 only
#                                  identifies which u32 fields are
#                                  pointer-typed. We do not need it
#                                  to parse, only to validate.
#   NJCM chunk                   — root NinjaModel.
#       MeshTreeNode (52 bytes, recursive linked list).
#       Each node:
#           u32  eval_flags
#           u32  mesh_ptr             → NjMesh (24 bytes), or 0.
#           f32  x, y, z              translation
#           i32  rot_x, rot_y, rot_z  Ninja-angles (BAMs)
#           f32  sx, sy, sz           scale
#           u32  child_ptr
#           u32  next_ptr
#       NjMesh (24 bytes):
#           u32  vlist_offset         → vertex chunk stream, or 0.
#           u32  plist_offset         → polygon chunk stream, or 0.
#           f32  bbox_x, bbox_y, bbox_z, bbox_radius
#       Vertex/polygon chunks: a stream of (u8 type_id, u8 flags,
#       optional u16 size, optional payload). type_id 255 = end.
#
# Vertex types we handle (chunkTypeId 32..50):
#   32 NJD_CV_SH         pos + pad4
#   33 NJD_CV_VN_SH      pos + pad4 + normal + pad4
#   34 NJD_CV            pos
#   35..40               pos + pad4 (varies; see Nj.kt)
#   41 NJD_CV_VN         pos + normal
#   42..47               pos + normal + pad4 (or NJD_CV_VN_NF: pos + normal + idx + bw)
#   48..50 NJD_CV_VNX*   pos + packed-i32 normal (+ optional pad4)
#
# Strip types we handle (chunkTypeId 64..75):
#   64        bare strip
#   65, 66    strip with u16,u16 UVs (raw / scaled)
#   67        strip with u16x3 normal
#   68, 69    strip with UVs + normal
#   70        strip with ARGB color (skipped)
#   71, 72    strip with UVs + ARGB color
#   73        bare strip (alt)
#   74, 75    strip with double UVs (skipped)
#
# Public API:
#   XjVertex       — per-vertex (pos, normal, uv).
#   XjMesh         — per-submesh (vertices, indices, material_id, bsphere).
#   XjBone         — flattened MeshTreeNode (used by /api/skeleton).
#   parse_xj_njcm  — NJCM payload bytes → list[XjMesh].
#   parse_nj_file  — full ``.nj`` IFF bytes → list[XjMesh].
#   parse_skeleton — full ``.nj`` IFF bytes → list[XjBone].
#
# Both ``parse_*`` functions raise ValueError on malformed input. They
# return an empty list when the parser cannot make sense of the data
# (the caller falls back to the primitive cube-preview in that case).
"""Pure-Python Ninja `.nj` mesh + skeleton reader for PSOBB Blue Burst.

Historical filename: ``xj.py``. The module parses the chunk-based
Ninja-Nj format that ships in PSOBB.IO `.nj` files, exposing it under
the editor's legacy ``XjVertex`` / ``XjMesh`` / ``parse_xj_*`` API.
"""
from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .iff import parse_iff


# ---------------------------------------------------------------------------
# Module configuration
# ---------------------------------------------------------------------------
#
# Set ``PSO_XJ_IGNORE_HIDE=1`` in the environment to make the parser
# ignore EVAL_HIDE / EVAL_SHAPE_SKIP at strip-emission time. The flag is
# also honored by ``formats/xj_descriptor.py`` (the descriptor-table
# .xj parser) — both expose it because the suspicion was that PSOBB BB
# stamps EVAL_HIDE on legitimately-visible mesh nodes and the parser
# is dropping their geometry.
#
# Empirically: across the 656 BML-inner models in PSOBB.IO, ZERO have
# the HIDE or SHAPE_SKIP flags set on a mesh-bearing node (audited by
# ``scripts/dump_eval_hide_audit.py`` 2026-04-24). So enabling this
# flag does NOT change rendering output for any shipping data — it
# exists purely as a knob to defend against future BB data that might
# stamp HIDE on visible meshes (e.g. mod packs, Ephinea additions).
#
# We default OFF: honor HIDE / SHAPE_SKIP per the SDK semantics. Test
# fixtures that need to compare "hidden vs not-hidden" can set the
# env var or pass ``ignore_hide=True`` to the public parsers.
_IGNORE_HIDE_DEFAULT: bool = bool(int(os.environ.get("PSO_XJ_IGNORE_HIDE", "0") or 0))


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class XjVertex:
    """Per-vertex attributes after parsing.

    All vector components are little-endian f32. Normal is always
    populated (synthesized as ``(0, 1, 0)`` for vertex chunks that
    omit it). UV is ``(0, 0)`` when not present in the strip chunk.

    ``bone_idx`` (added 2026-04-24): index of the mesh-tree node whose
    vertex chunk SUPPLIED this vertex's slot. The default value -1
    means "not skinned / unknown" — the bake-to-world parser path
    leaves this at -1 because it transforms vertices into world space
    and the bone identity is no longer needed. The bone-LOCAL skinning
    path (``parse_nj_skinned``) populates this field with the owning
    node's DFS-order index so the frontend can apply animated bone
    matrices at render time.
    """
    pos: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    uv: Tuple[float, float]
    bone_idx: int = -1
    # Per-vertex RGBA in 0..1 (2026-06-20). PSOBB bakes shading / AO /
    # tint into either a per-vertex ARGB chunk (strip types 70/71/72 or
    # vertex chunk heads 0x23/0x2a) OR the material-chunk DIFFUSE color.
    # psov2 (DashGL NinjaModel.js) multiplies an UNLIT MeshBasicMaterial
    # by this color; without it, untextured submeshes wash to white and
    # textured ones lose their authored shading. Default white = no-op.
    # Precedence (psov2 `aClr = aPos.color || this.color`): own per-vertex
    # color > material diffuse > white. Alpha is floored to 0.3 so no
    # submesh renders fully invisible (NinjaModel.js:867).
    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


@dataclass
class XjMesh:
    """A single submesh with triangulated indices, ready for three.js.

    ``indices`` is an already de-stripified triangle list (each triple
    of integers indexes into ``vertices``). ``bounding_sphere`` is
    ``(cx, cy, cz, r)`` derived from the vertex positions in the
    submesh's BONE-LOCAL frame. ``material_id`` is the texture id from
    the most recent ``Tiny`` chunk seen while emitting this strip, or 0
    if none.

    Bone-local-to-world transform tagging (added 2026-04-24):

    PSOBB Blue Burst skinned models use a tree of NjObject mesh-tree
    nodes (the "skeleton"); each node stores its TRS relative to its
    parent and (optionally) a vertex chunk + polygon chunk pair. The
    polygon chunk emits triangle strips; THE VERTICES THOSE STRIPS
    REFERENCE are stored in BONE-LOCAL coordinates inside each node's
    own vertex chunk.

    A naive parser that drops bone transforms (the previous behaviour
    of this module) renders every bone's strip in the bone's local
    frame, so head/arms/body all stack at the model origin and the
    enemy looks like exploded shards.

    To fix this without the frontend having to grow a skinning
    pipeline, the parser BAKES each vertex into world space at
    vertex-pass time using the owning node's local-to-world matrix.
    Strips emitted later then reference world-space vertex positions —
    correct even when one strip references vertex slots populated by
    multiple bones (a pattern PSOBB BB uses for cache/replay strips).

    The transform fields below are recorded for tooling/diagnostic use
    (the e2e tests assert at least one mesh has a non-zero position).
    They reflect the world matrix of the MeshTreeNode that owned the
    polygon chunk (i.e. the strip's HOST bone), NOT the matrix of the
    individual vertex's owning bone — because a single strip may pull
    from many bones, the host-node matrix is the most useful single
    pose the wire format can carry. The frontend MUST NOT apply these
    per-mesh transforms when ``vertices_pre_transformed`` is True (set
    by ``_xj_meshes_to_payload`` for every mesh emitted by this
    module) — the vertices are already in world space.

    ``world_position`` / ``world_rotation_euler`` / ``world_scale``:
        Decomposition of the host node's local-to-world transform.
    ``world_matrix``:
        16-float row-major equivalent of the same. Row-major because
        it makes test assertions readable; the frontend transposes if
        it wants column-major (Matrix4.fromArray default).
    """
    vertices: List[XjVertex]
    indices: List[int]
    material_id: int = 0
    bounding_sphere: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    # Host plist-node world transform (2026-04-24). Diagnostic — the
    # vertices in this submesh are ALREADY world-space, so the frontend
    # must NOT compose this with its mesh transform.
    world_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    world_rotation_euler: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    world_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    # Row-major 4x4: m[row*4+col].
    world_matrix: Tuple[float, ...] = (
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    # Per-submesh render-state flags (2026-06-20, Phase 3). Decoded from
    # the polygon stream's BlendAlpha (type 1), Tiny (8/9 alpha-test
    # overlay) and strip-chunk (64..75) flags that were in effect when
    # this strip emitted. The frontend maps these onto three.js material
    # state: ``blend_mode`` "additive" -> AdditiveBlending + depthWrite
    # false; ``alpha_test`` -> transparent + alphaTest threshold;
    # ``two_sided`` -> DoubleSide else FrontSide. Defaults are the
    # opaque, single-sided, no-blend baseline so older/untracked paths
    # render exactly as before.
    blend_mode: str = "none"           # none / blend / additive / multiply / screen
    two_sided: bool = False
    # ``alpha_test`` is ``{"enabled": bool, "threshold": int}`` or None.
    alpha_test: Optional[dict] = None
    # ``alpha_blend`` is ``{"src": str, "dst": str}`` or None (the raw
    # factor pair behind ``blend_mode``; surfaced for the Material panel).
    alpha_blend: Optional[dict] = None


# ---------------------------------------------------------------------------
# Internal: tristrip → triangle list
# ---------------------------------------------------------------------------


def _tristrip_to_triangles(
    strip: List[int], cw: bool = False
) -> List[int]:
    """Convert a triangle-strip index list to a flat triangle list.

    Drops degenerate triangles (any pair of equal indices). Honours
    the parity flip every other triangle that triangle-strip semantics
    require. ``cw`` flips the initial winding (signed strip header in
    chunk types 64..75; see ``parseTriangleStripChunk`` in Nj.kt).
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


# ---------------------------------------------------------------------------
# MeshTreeNode (NJS_OBJECT) eval_flags constants.
# ---------------------------------------------------------------------------
#
# Bit values are the SEGA Ninja SDK ``njchunk.h`` convention:
#
#     NJD_EVAL_UNIT_POS   0x01  ignore translation
#     NJD_EVAL_UNIT_ANG   0x02  ignore rotation
#     NJD_EVAL_UNIT_SCL   0x04  ignore scale
#     NJD_EVAL_HIDE       0x08  do not draw this node's mesh
#     NJD_EVAL_BREAK      0x10  do not recurse into this node's children
#     NJD_EVAL_ZXY_ANG    0x20  rotation order is ZXY (default is XYZ)
#     NJD_EVAL_SKIP       0x40  skip transform calculation entirely
#                              (treated as identity local M)
#     NJD_EVAL_SHAPE_SKIP 0x80  do not draw and do not recurse children
#
# Phantasmal Ninja honors at minimum POS/ANG/SCL/HIDE/BREAK; we mirror
# that here. ZXY_ANG and SKIP are honored too because they are cheap
# to support and a few PSOBB BB models (e.g. ``boss1_s_nb_dragon``)
# carry them on a handful of nodes (the histogram of eval_flag values
# in that model goes up to 30 = 0x1E).
EVAL_UNIT_POS = 0x01
EVAL_UNIT_ANG = 0x02
EVAL_UNIT_SCL = 0x04
EVAL_HIDE = 0x08
EVAL_BREAK = 0x10
EVAL_ZXY_ANG = 0x20
EVAL_SKIP = 0x40
EVAL_SHAPE_SKIP = 0x80
EVAL_CLIP = 0x100      # frustum-clip hint (rendering only; ignored)
EVAL_MODIFIER = 0x200  # modifier volume (ignored)


# ---------------------------------------------------------------------------
# 4x4 matrix helpers (row-major) — the bone-tree walker carries these.
# ---------------------------------------------------------------------------
#
# We use plain Python lists of 16 floats (row-major) to avoid pulling
# in numpy. The walker constructs O(bones) matrices per file (typically
# < 200), so the per-multiply cost is negligible compared to the chunk
# parser. Row-major was picked because it makes the JSON wire shape
# readable in tests; the JS frontend transposes if its consumer wants
# column-major (Matrix4.fromArray).
#
# Convention: a vector v is treated as a column; ``M * v`` applies M.
# Composition order: ``parent * local`` means local applies first.
# This matches Phantasmal's NjObject walker.

_BAMS_TO_RAD = (2.0 * math.pi) / 65536.0


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
    """Build a row-major 4x4 from translation, Euler rotation, and scale.

    ``rot_xyz_rad`` is ``(rx, ry, rz)`` in RADIANS. By default we use
    Phantasmal's **ZYX** Euler convention: ``R = Rz * Ry * Rx`` (the
    matrix rotates a vector by X first, then Y, then Z — i.e. intrinsic
    Z then Y then X about local axes after each prior rotation, which
    matches three.js' ``new Euler(x, y, z, "ZYX")`` semantics). When
    ``zxy_order`` is True (the ``EVAL_ZXY_ANG`` mesh-tree flag) we use
    ZXY: ``R = Rz * Rx * Ry``.

    Important: an earlier port of this function used XYZ
    (``R = Rx * Ry * Rz``) as the default — the comment claimed
    "Phantasmal default" but that was incorrect. Phantasmal's
    ``NinjaGeometryConversion.kt::convertObject`` builds the bone Euler
    as ``Euler(x, y, z, if (ef.zxyRotationOrder) "ZXY" else "ZYX")``,
    and three.js' ``"ZYX"`` ordering composes as ``Rz @ Ry @ Rx``. The
    XYZ default produced visually-twisted heads on bones with nested
    rotations (most visibly on the De Rol Le boss family); see
    ``AGENT_MODEL_DEEP_DEBUG_REPORT.md`` for the empirical comparison.

    The composed local-to-parent matrix is ``T * R * S`` — same as
    three.js' ``Object3D.updateMatrix()`` and Phantasmal's NjObject
    walker.
    """
    px, py, pz = pos
    rx, ry, rz = rot_xyz_rad
    sx, sy, sz = scale

    cx = math.cos(rx); s_x = math.sin(rx)
    cy = math.cos(ry); s_y = math.sin(ry)
    cz = math.cos(rz); s_z = math.sin(rz)

    # Per-axis rotation matrices.
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
        # ZXY: R = Rz * Rx * Ry  (intrinsic Z, X, Y; flag EVAL_ZXY_ANG).
        R = _mat4_mul(_mat4_mul(Rz, Rx), Ry)
    else:
        # ZYX: R = Rz * Ry * Rx  (Phantasmal default — three.js "ZYX").
        # Confirmed against
        # web/src/jsMain/kotlin/world/phantasmal/web/core/rendering/conversion/NinjaGeometryConversion.kt::convertObject.
        R = _mat4_mul(_mat4_mul(Rz, Ry), Rx)

    # T * R, with translation column substituted in the last column.
    M = list(R)
    M[0 * 4 + 3] = px
    M[1 * 4 + 3] = py
    M[2 * 4 + 3] = pz

    # Apply scale on the right: M = (T*R) * S, where S is diag(sx,sy,sz,1).
    # That is: scale column j by [sx,sy,sz,1][j] for j in 0..3.
    sxsysz1 = (sx, sy, sz, 1.0)
    for i in range(4):
        for j in range(4):
            M[i * 4 + j] *= sxsysz1[j]
    return M


# ---------------------------------------------------------------------------
# Internal: NjMesh struct (24 bytes) + MeshTreeNode (52 bytes)
# ---------------------------------------------------------------------------


_MESH_TREE_NODE_FMT = "<II3f3i3fII"
_MESH_TREE_NODE_SIZE = struct.calcsize(_MESH_TREE_NODE_FMT)
assert _MESH_TREE_NODE_SIZE == 52, _MESH_TREE_NODE_SIZE

# NjMesh: vlist_offset, plist_offset, bbox.x, bbox.y, bbox.z, bbox.r = 24 bytes.
_NJ_MESH_FMT = "<II4f"
_NJ_MESH_SIZE = struct.calcsize(_NJ_MESH_FMT)
assert _NJ_MESH_SIZE == 24, _NJ_MESH_SIZE


def _read_mesh_tree_node(body: bytes, off: int) -> Optional[Tuple[int, int, float, float, float, int, int]]:
    """Read one 52-byte MeshTreeNode at ``off``.

    Returns ``(eval_flags, mesh_ptr, x, y, z, child_ptr, next_ptr)``
    or ``None`` if the read would walk past the buffer. Rotations and
    scales are intentionally dropped — the bone-walker uses a separate
    helper that surfaces them.
    """
    if off < 0 or off + _MESH_TREE_NODE_SIZE > len(body):
        return None
    f = struct.unpack_from(_MESH_TREE_NODE_FMT, body, off)
    return f[0], f[1], f[2], f[3], f[4], f[11], f[12]


def _read_mesh_tree_node_full(body: bytes, off: int):
    """Read all 13 fields of a 52-byte MeshTreeNode at ``off``.

    Returns ``(eval_flags, mesh_ptr, pos, rot_bams, scale, child_ptr,
    next_ptr)`` or ``None`` on truncation. ``pos`` and ``scale`` are
    floats; ``rot_bams`` is the raw integer Ninja-angle triple. The
    transform accumulator converts these via ``_BAMS_TO_RAD``.
    """
    if off < 0 or off + _MESH_TREE_NODE_SIZE > len(body):
        return None
    f = struct.unpack_from(_MESH_TREE_NODE_FMT, body, off)
    eval_flags = f[0]
    mesh_ptr = f[1]
    pos = (f[2], f[3], f[4])
    rot = (f[5], f[6], f[7])  # BAMs (signed int32)
    scale = (f[8], f[9], f[10])
    child_ptr = f[11]
    next_ptr = f[12]
    return eval_flags, mesh_ptr, pos, rot, scale, child_ptr, next_ptr


def _read_nj_mesh(body: bytes, off: int) -> Optional[Tuple[int, int, Tuple[float, float, float, float]]]:
    """Read one 24-byte NjMesh at ``off``.

    Returns ``(vlist_offset, plist_offset, bbox)`` or ``None`` on
    out-of-range. ``bbox`` is ``(cx, cy, cz, r)``.
    """
    if off < 0 or off + _NJ_MESH_SIZE > len(body):
        return None
    f = struct.unpack_from(_NJ_MESH_FMT, body, off)
    return f[0], f[1], (f[2], f[3], f[4], f[5])


# ---------------------------------------------------------------------------
# Internal: chunk-stream parser (Nj.kt parseChunks port)
# ---------------------------------------------------------------------------
#
# Each chunk starts with ``u8 type_id, u8 flags``. The chunk's body
# size depends on ``type_id``:
#
#   0..3, 4, 5         no body          (size = 2)
#   8..9               u16 body         (size = 4)
#   17..31             2*u16 body       (size = 4 + 2 * cursor.short())
#   32..50             4*u16 body       (size = 4 + 4 * cursor.short())
#   56..58             2*u16 body       (size = 4 + 2 * cursor.short())
#   64..75             2*u16 body       (size = 4 + 2 * cursor.short())
#   255                no body, end     (size = 2)
#   else               2*u16 body       (size = 4 + 2 * cursor.short())
#
# We identify three chunk classes that influence rendering:
#   - Vertex chunks (32..50)        → contribute named vertices.
#   - Tiny chunks (8..9)            → set the current texture id.
#   - Strip chunks (64..75)         → emit triangle-strip submeshes.
#
# Material/Volume/etc chunks are recognised by size for skip purposes
# only; we don't surface their state.
#
# The parser carries per-stream state (texture id, last vertex chunk
# range) so that strip chunks can resolve their per-vertex
# UV/normal/index data correctly.


@dataclass
class _ChunkVertex:
    """One vertex slot read from a vertex chunk.

    ``index`` is the GLOBAL slot number (chunk-relative ``index +
    cursor.uShort()`` for type 37/44; otherwise sequential from the
    chunk's base index). The mesh tree's chunk processor populates
    a sparse list keyed by this index.
    """
    index: int
    pos: Tuple[float, float, float]
    normal: Optional[Tuple[float, float, float]]
    # Per-vertex RGBA (0..1) when the chunk head carries an ARGB color
    # (Nj heads 0x23 / 0x2a — psov2 NinjaModel.js:638). None means the
    # slot has no own color and the material diffuse should be used.
    color: Optional[Tuple[float, float, float, float]] = None


def _parse_vertex_chunk(
    body: bytes,
    chunk_data_pos: int,
    chunk_size: int,
    type_id: int,
    flags: int,
) -> List[_ChunkVertex]:
    """Read one ``32..50`` vertex chunk's body.

    Layout (per Nj.kt parseVertexChunk):

        u16 index, u16 vertex_count, then per-vertex:
            f32 x, f32 y, f32 z
            (vertex_format-specific extras)

    The variant data following the position depends on ``type_id``:
        32       skip 4 (always 1.0)
        33       skip 4, normal vec3, skip 4 (always 0.0)
        34       no extra
        35..40   skip 4 (or for 37: u16 idx-offset, u16 boneWeight)
        41       normal vec3
        42..47   normal vec3 + skip 4 (or for 44: u16 idx-offset, u16 bw)
        48..50   packed normal u32 (+ optional skip 4)

    Returns the parsed vertex slots in the order they appear in the
    chunk. The mesh tree's chunk processor folds them into a sparse
    list keyed by ``vertex.index``.
    """
    vertices: List[_ChunkVertex] = []
    end = chunk_data_pos + chunk_size
    # Per Nj.kt: the chunk body starts with a u16 size word that has
    # already been consumed by ``_chunk_body_size``; the vertex
    # content (index + count + per-vertex data) starts AFTER it.
    pos = chunk_data_pos + 2
    if pos + 4 > end:
        return vertices
    base_index, vertex_count = struct.unpack_from("<HH", body, pos)
    pos += 4

    for i in range(vertex_count):
        if pos + 12 > end:
            break
        x, y, z = struct.unpack_from("<3f", body, pos)
        pos += 12
        vertex_index = base_index + i
        normal: Optional[Tuple[float, float, float]] = None
        color: Optional[Tuple[float, float, float, float]] = None

        if type_id == 32:
            # NJD_CV_SH          — pos + 4 bytes (always 1.0)
            pos += 4
        elif type_id == 33:
            # NJD_CV_VN_SH       — pos + 4 + normal + 4 (always 0.0)
            pos += 4
            if pos + 12 > end:
                break
            nx, ny, nz = struct.unpack_from("<3f", body, pos)
            pos += 12
            normal = (nx, ny, nz)
            pos += 4
        elif type_id == 34:
            # NJD_CV             — pos only
            pass
        elif type_id == 35:
            # NJD_CV_VC (head 0x23) — pos + ARGB8888 vertex color (no
            # normal). The 4 bytes are laid out B,G,R,A on disk (psov2
            # NinjaModel.js:638-644 reads b,g,r,a / 255).
            if pos + 4 > end:
                break
            cb, cg, cr, ca = body[pos], body[pos + 1], body[pos + 2], body[pos + 3]
            pos += 4
            color = (cr / 255.0, cg / 255.0, cb / 255.0, ca / 255.0)
        elif 36 <= type_id <= 40:
            if type_id == 37:
                # NJD_CV_NF      — u16 idx-offset, u16 boneWeight
                if pos + 4 > end:
                    break
                idx_off, _bw = struct.unpack_from("<HH", body, pos)
                pos += 4
                vertex_index = base_index + idx_off
            else:
                # NJD_CV_D8/UF/S5/S4/IN — skip 4 (user/material flags)
                pos += 4
        elif type_id == 41:
            # NJD_CV_VN          — pos + normal
            if pos + 12 > end:
                break
            nx, ny, nz = struct.unpack_from("<3f", body, pos)
            pos += 12
            normal = (nx, ny, nz)
        elif type_id == 42:
            # NJD_CV_VNC (head 0x2a) — pos + normal + ARGB8888 color.
            # The trailing 4 bytes (which other 42..47 types use as a
            # user/material pad) are the per-vertex color here.
            if pos + 12 > end:
                break
            nx, ny, nz = struct.unpack_from("<3f", body, pos)
            pos += 12
            normal = (nx, ny, nz)
            if pos + 4 > end:
                break
            cb, cg, cr, ca = body[pos], body[pos + 1], body[pos + 2], body[pos + 3]
            pos += 4
            color = (cr / 255.0, cg / 255.0, cb / 255.0, ca / 255.0)
        elif 43 <= type_id <= 47:
            if pos + 12 > end:
                break
            nx, ny, nz = struct.unpack_from("<3f", body, pos)
            pos += 12
            normal = (nx, ny, nz)
            if type_id == 44:
                if pos + 4 > end:
                    break
                idx_off, _bw = struct.unpack_from("<HH", body, pos)
                pos += 4
                vertex_index = base_index + idx_off
            else:
                # NJD_CV_VN_D8/UF/S5/S4/IN — skip 4
                pos += 4
        elif 48 <= type_id <= 50:
            # NJD_CV_VNX*        — packed normal in 32 bits
            if pos + 4 > end:
                break
            (n,) = struct.unpack_from("<I", body, pos)
            pos += 4
            normal = (
                ((n >> 20) & 0x3FF) / 0x3FF,
                ((n >> 10) & 0x3FF) / 0x3FF,
                (n & 0x3FF) / 0x3FF,
            )
            if type_id >= 49:
                pos += 4

        vertices.append(_ChunkVertex(vertex_index, (x, y, z), normal, color))

    return vertices


# Per-vertex attributes a strip emits. None means "use the matching
# field from the global vertex slot". The strip's per-vertex normal
# and UV (when present) override the slot's.
@dataclass
class _StripVertex:
    """One per-strip-vertex slot.

    The strip header gives a list of indices into the global vertex
    slot table; each index optionally carries its own UV/normal that
    overrides whatever the slot has. PSOBB strips most commonly carry
    UVs but inherit the slot's normal.
    """
    index: int
    uv: Optional[Tuple[float, float]]
    normal: Optional[Tuple[float, float, float]]
    # Per-vertex RGBA (0..1) for strip types 70/71/72 (ARGB8888 on disk,
    # laid out B,G,R,A — Nj.kt "Ignore ARGB8888 color"). None for strips
    # that carry no color.
    color: Optional[Tuple[float, float, float, float]] = None


@dataclass
class _StripChunk:
    """One ``Strip`` chunk's parsed contents.

    A single chunk may pack multiple separate triangle-strips, each
    described by a signed length (winding flag in the sign bit).
    Material/texture state is resolved at the higher PolygonChunkProcessor
    level — we surface only the geometry here.
    """
    strips: List[Tuple[bool, List[_StripVertex]]]  # (clockwise, vertices)


def _parse_strip_chunk(
    body: bytes,
    chunk_data_pos: int,
    chunk_size: int,
    type_id: int,
    flags: int,
) -> _StripChunk:
    """Read one ``64..75`` strip chunk's body.

    Header (after type+flags): u16 user_offset_and_strip_count
    where the high 2 bits are user_offset_size_words (skipped per
    triangle past the third) and low 14 bits are strip_count.

    Per-vertex data depends on ``type_id``:
        64        index only
        65, 66    index + 2*u16 UV (UV scale = 1/255)
        67        index + 3*u16 normal (normal scale = 1/255)
        68, 69    index + 2*u16 UV + 3*u16 normal
        70        index + 4*u8 ARGB color (skipped)
        71, 72    index + 2*u16 UV + 4*u8 color (skipped)
        73        index only (alt)
        74, 75    index + 4*u16 (double-tex skipped)

    Plus per-triangle starting at the third index: ``user_offset_size``
    bytes are skipped.

    UVs are 8.8 fixed-point: raw 0x100 = 1.0. So divide by 256, not 255.
    The reader was using 1/255 (matching the legacy comment); writer in
    import_external.py and Sega's Ninja SDK both use 256. Fixed
    2026-04-30: tiled-texture seam misalignment was accumulating ~3%
    over 8 repeats, visible on cave/mine floors and other repeated tiles.
    Normals are unsigned u16 mapping 0..0xFFFF → -1..+1 in PSOBB; the
    1/255 was wrong for those too but we keep it as a separate fix
    (verify against in-game lighting before changing).
    """
    end = chunk_data_pos + chunk_size
    # Strip chunks share the "leading u16 size word already consumed"
    # convention with vertex chunks (cf. Nj.kt parseChunks for type
    # 64..75). The strip header (winding+count) follows the size word.
    pos = chunk_data_pos + 2

    if pos + 2 > end:
        return _StripChunk(strips=[])

    (header,) = struct.unpack_from("<h", body, pos)
    pos += 2
    user_offset_size = 2 * ((header >> 14) & 0x3)
    if user_offset_size < 0:
        user_offset_size = 0
    strip_count = header & 0x3FFF

    has_uv = type_id in (65, 66, 68, 69, 71, 72)
    has_normal = type_id in (67, 68, 69)
    has_color = type_id in (70, 71, 72)
    has_double_uv = type_id in (74, 75)

    UV_SCALE = 1.0 / 256.0
    NORMAL_SCALE = 1.0 / 255.0

    strips: List[Tuple[bool, List[_StripVertex]]] = []

    for _strip_i in range(strip_count):
        if pos + 2 > end:
            break
        (strip_header,) = struct.unpack_from("<h", body, pos)
        pos += 2
        clockwise = strip_header < 0
        index_count = abs(strip_header)

        verts: List[_StripVertex] = []
        for j in range(index_count):
            if pos + 2 > end:
                break
            (idx,) = struct.unpack_from("<H", body, pos)
            pos += 2
            uv: Optional[Tuple[float, float]] = None
            normal: Optional[Tuple[float, float, float]] = None
            color: Optional[Tuple[float, float, float, float]] = None
            if has_uv:
                if pos + 4 > end:
                    break
                u_raw, v_raw = struct.unpack_from("<HH", body, pos)
                pos += 4
                uv = (u_raw * UV_SCALE, v_raw * UV_SCALE)
            if has_color:
                # ARGB8888 laid out B,G,R,A on disk (Nj.kt strip color;
                # mirrors the material-chunk BGRA convention). Previously
                # skipped — now read so the model shows its authored
                # per-vertex shading instead of washing to white.
                if pos + 4 > end:
                    break
                cb, cg, cr, ca = body[pos], body[pos + 1], body[pos + 2], body[pos + 3]
                pos += 4
                color = (cr / 255.0, cg / 255.0, cb / 255.0, ca / 255.0)
            if has_normal:
                if pos + 6 > end:
                    break
                nx_r, ny_r, nz_r = struct.unpack_from("<HHH", body, pos)
                pos += 6
                normal = (nx_r * NORMAL_SCALE, ny_r * NORMAL_SCALE, nz_r * NORMAL_SCALE)
            if has_double_uv:
                if pos + 8 > end:
                    break
                pos += 8
            if j >= 2:
                pos += user_offset_size
            verts.append(_StripVertex(index=idx, uv=uv, normal=normal, color=color))

        strips.append((clockwise, verts))

    return _StripChunk(strips=strips)


# ---------------------------------------------------------------------------
# Chunk-stream walker
# ---------------------------------------------------------------------------
#
# Walks a chunk stream until it sees ``type_id == 255`` (End) or
# falls off the end of the buffer. Returns the list of
# ``(chunk_start, type_id, flags, chunk_data_pos, chunk_data_size)``
# tuples for the higher-level processor.


def _chunk_body_size(body: bytes, type_id: int, hdr_pos: int) -> int:
    """Return the BODY size (bytes after the type+flags header) for
    a chunk whose type+flags is at ``hdr_pos`` in ``body``.

    The size is determined by ``type_id`` per Nj.kt parseChunks. For
    chunks with a u16-words-prefixed body, we read ``cursor.short()``
    at ``hdr_pos+2`` and apply the appropriate multiplier.

    Returns 0 for chunks whose ``size`` is fixed at 2 (header-only).
    """
    # 0..5: header-only
    if type_id < 8:
        return 0
    # 8..9: 2-byte body (single u16)
    if 8 <= type_id <= 9:
        return 2
    if hdr_pos + 4 > len(body):
        return 0
    (body_words,) = struct.unpack_from("<h", body, hdr_pos + 2)
    if 17 <= type_id <= 31:
        return 2 + 2 * body_words
    if 32 <= type_id <= 50:
        return 2 + 4 * body_words
    if 56 <= type_id <= 58:
        return 2 + 2 * body_words
    if 64 <= type_id <= 75:
        return 2 + 2 * body_words
    if type_id == 255:
        return 0
    # Unknown — same scheme as 17..31 (Phantasmal logs and skips).
    return 2 + 2 * body_words


def _walk_chunk_stream(body: bytes, start_off: int) -> List[Tuple[int, int, int, int, int]]:
    """Walk a chunk stream starting at ``start_off`` until end-of-stream.

    Returns ``[(hdr_start, type_id, flags, body_pos, body_size), ...]``.
    Stops on type 255 (End), out-of-buffer, or a hard 4096-chunk cap.

    The cap exists purely as a safety net — a malformed file that
    returned, e.g., ``size=0`` for an unknown chunk type could cause
    the cursor to never advance.
    """
    out: List[Tuple[int, int, int, int, int]] = []
    pos = start_off
    n = len(body)
    seen_pos: set = set()
    while pos + 2 <= n and len(out) < 4096:
        if pos in seen_pos:
            # Defensive: a zero-size chunk would loop forever.
            break
        seen_pos.add(pos)
        type_id = body[pos]
        flags = body[pos + 1]
        body_size = _chunk_body_size(body, type_id, pos)
        body_pos = pos + 2
        out.append((pos, type_id, flags, body_pos, body_size))
        if type_id == 255:
            break
        # Header-only chunks have body_size == 0 per Nj.kt;
        # otherwise the body comes after the 2-byte u16 size word
        # we already consumed inside `_chunk_body_size`.
        if 0 < body_size < 2:
            # Malformed — body claims to exist but is too small to
            # even hold its own size word. Bail rather than loop.
            break
        if body_size == 0:
            pos = body_pos  # advance past type+flags only (2 bytes from header start)
        else:
            pos = body_pos + body_size
    return out


# ---------------------------------------------------------------------------
# Mesh-tree walker (chunk processor across the bone hierarchy)
# ---------------------------------------------------------------------------
#
# Phantasmal's PolygonChunkProcessor accumulates state (textureId,
# srcAlpha, dstAlpha, cached chunks) across nested trees so that strip
# chunks can pick up the most recent texture. We do the same here:
# the walk_tree DFS carries a small state struct.
#
# Vertex slots are accumulated in a single large list keyed by
# vertex_index — Phantasmal does this too, growing the list with nulls
# as needed. We use a dict for sparse storage; the strip chunks index
# into it.


class _NinjaChunkState:
    """Per-walk state carried by the DFS over mesh-tree nodes.

    Attributes
    ----------
    vertex_slots:
        Dict[int, _ChunkVertex] — accumulated globally across the
        tree. Strip chunks index into this; later writes overwrite
        earlier ones (matches Phantasmal's "fill nulls then assign").
    texture_id:
        Last seen Tiny-chunk texture id (-1 if none yet). Mapped onto
        each emitted XjMesh's ``material_id`` field.
    cached_chunks:
        Per-cache-index list of chunks. Type-4 chunks open a cache
        list (subsequent chunks accumulate into it instead of
        executing); type-5 chunks replay a previously-cached list.
    """

    __slots__ = (
        "vertex_slots",
        "texture_id",
        "cached_chunks",
        "cache_active",
        "current_world_matrix",
        "diffuse_color",
        "blend",
        "tiny_alpha_bits",
    )

    def __init__(self) -> None:
        self.vertex_slots: Dict[int, _ChunkVertex] = {}
        self.texture_id: int = -1
        # Most-recent Material-chunk DIFFUSE color (RGBA 0..1), used as
        # the default per-vertex color for strips whose vertices carry
        # no own ARGB color (psov2 `aClr = aPos.color || this.color`,
        # NinjaModel.js:863). None = no material seen yet -> white.
        self.diffuse_color: Optional[Tuple[float, float, float, float]] = None
        # Most-recent BlendAlpha chunk (type 1) decode, carried forward
        # as the active blend mode for subsequent strips. None = no blend
        # chunk seen -> opaque (Phase 3, 2026-06-20).
        self.blend = None  # Optional[BlendAlphaPayload]
        # Most-recent Tiny-chunk alpha-test threshold overlay (top 3 bits
        # of the Tiny body word). 0 = no alpha test in effect.
        self.tiny_alpha_bits: int = 0
        self.cached_chunks: Dict[int, List[Tuple[int, int, int, int, int]]] = {}
        # When non-None, all subsequent chunks in the current vlist /
        # plist are diverted into ``cached_chunks[cache_active]``
        # instead of being processed. A second type-4 closes the
        # cache (and re-opens a new one); type-5 replays a cache.
        self.cache_active: Optional[int] = None
        # The world matrix of the MeshTreeNode currently owning the
        # plist being processed. Used by ``_emit_strip_mesh`` to tag
        # each XjMesh as it pushes one. Defaults to identity so that
        # callers that bypass the tree walker (e.g. a bare NjMesh fed
        # to ``_process_polygon_chunks``) still get sane output.
        self.current_world_matrix: List[float] = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]


def _process_vertex_chunks(
    body: bytes,
    start_off: int,
    state: _NinjaChunkState,
) -> None:
    """Walk a vertex chunk stream and merge slots into ``state``.

    Vertex chunks (type 32..50) become ``_ChunkVertex`` instances;
    everything else is ignored. ``cached_chunks`` semantics apply but
    Phantasmal observes only polygon (strip) chunks honour them — we
    follow suit and treat type-4/5 in vertex streams as no-ops.

    Each parsed vertex's ``pos`` and ``normal`` is transformed into
    WORLD SPACE here using ``state.current_world_matrix``. This is
    what makes a multi-bone PSOBB BB skinned model render coherently
    (head/arms/body in their proper places) without the frontend
    having to maintain a skinning pipeline. The strips emitted later
    pull from these world-space slots, so their geometry is
    already-positioned by the time it reaches three.js.
    """
    M = state.current_world_matrix
    chunks = _walk_chunk_stream(body, start_off)
    for (hdr, type_id, flags, body_pos, body_size) in chunks:
        if 32 <= type_id <= 50:
            for cv in _parse_vertex_chunk(body, body_pos, body_size, type_id, flags):
                # Transform position by world matrix (homogeneous, w=1).
                px, py, pz = cv.pos
                wpx = M[0] * px + M[1] * py + M[2] * pz + M[3]
                wpy = M[4] * px + M[5] * py + M[6] * pz + M[7]
                wpz = M[8] * px + M[9] * py + M[10] * pz + M[11]
                # Transform normal by upper-3x3 only (no translation).
                # We pass non-uniform scale through as-is — PSOBB BB
                # bones almost always have uniform 1.0 scale, and
                # un-renormalized normals are accepted by three.js'
                # MeshStandardMaterial (which doesn't strictly require
                # unit length).
                if cv.normal is not None:
                    nx, ny, nz = cv.normal
                    wnx = M[0] * nx + M[1] * ny + M[2] * nz
                    wny = M[4] * nx + M[5] * ny + M[6] * nz
                    wnz = M[8] * nx + M[9] * ny + M[10] * nz
                    new_normal = (wnx, wny, wnz)
                else:
                    new_normal = None
                state.vertex_slots[cv.index] = _ChunkVertex(
                    index=cv.index,
                    pos=(wpx, wpy, wpz),
                    normal=new_normal,
                    color=cv.color,
                )


def _process_polygon_chunks(
    body: bytes,
    start_off: int,
    state: _NinjaChunkState,
    out_meshes: List[XjMesh],
) -> None:
    """Walk a polygon chunk stream and emit submeshes into ``out_meshes``.

    Strip chunks (type 64..75) become ``XjMesh`` instances using the
    current ``state.vertex_slots`` and ``state.texture_id``. Tiny
    chunks (type 8..9) update ``state.texture_id``. Type-4/5 implement
    the cache/replay semantics from PolygonChunkProcessor.

    Per Phantasmal's parseNjModel: every NjMesh's polygon stream is
    processed by a fresh ``PolygonChunkProcessor`` (so ``cacheList``
    starts null), while ``cachedChunks`` persists across the whole
    parseNjModel flow. We mirror that here by resetting
    ``state.cache_active`` to None at the start of each call but
    leaving ``state.cached_chunks`` populated.
    """
    state.cache_active = None
    chunks = _walk_chunk_stream(body, start_off)
    _process_polygon_chunk_list(body, chunks, state, out_meshes)


def _process_polygon_chunk_list(
    body: bytes,
    chunks: List[Tuple[int, int, int, int, int]],
    state: _NinjaChunkState,
    out_meshes: List[XjMesh],
) -> None:
    """Recursively process a list of polygon chunks honouring caching.

    Direct port of Phantasmal's ``PolygonChunkProcessor.process``.
    """
    for (hdr, type_id, flags, body_pos, body_size) in chunks:
        if state.cache_active is not None and type_id != 5:
            # We're inside a CachePolygonList block — keep accumulating.
            # Phantasmal closes the cache only when DrawPolygonList is
            # encountered; in practice live BB files use type-4 to open
            # and type-5 to close-and-replay.
            state.cached_chunks[state.cache_active].append(
                (hdr, type_id, flags, body_pos, body_size)
            )
            continue

        if type_id == 4:
            # CachePolygonList: cache_index = flags
            state.cache_active = flags
            state.cached_chunks[flags] = []
        elif type_id == 5:
            # DrawPolygonList: cache_index = flags. Replay the cache.
            state.cache_active = None
            cached = state.cached_chunks.get(flags)
            if cached:
                _process_polygon_chunk_list(body, cached, state, out_meshes)
        elif type_id == 1:
            # BlendAlpha: blend mode lives in the flags byte. Track it as
            # the active blend state for subsequent strips (Phase 3).
            try:
                from .material import decode_blend_alpha_chunk as _dbac
                state.blend = _dbac(flags)
            except Exception:
                state.blend = None
        elif 8 <= type_id <= 9:
            # Tiny: bottom 13 bits of u16 = texture_id; top 3 bits =
            # alpha-test threshold overlay (PSOBB overload).
            if body_size >= 2:
                (tex_word,) = struct.unpack_from("<H", body, body_pos)
                state.texture_id = tex_word & 0x1FFF
                state.tiny_alpha_bits = (tex_word >> 13) & 0x07
        elif 17 <= type_id <= 23:
            # Material chunk — decode its DIFFUSE color (when present)
            # and carry it forward as the default per-vertex color for
            # subsequent strips. PSOBB.IO Nj data carries NO per-vertex
            # color chunks, so THIS is the load-bearing color source
            # that fixes the flat/white look (psov2's `this.color`).
            try:
                from .material import decode_material_chunk as _dmc
                mp = _dmc(type_id, flags, body[body_pos:body_pos + body_size])
            except Exception:
                mp = None
            if mp is not None and mp.diffuse is not None:
                r, g, b, a = mp.diffuse.to_tuple()
                state.diffuse_color = (r / 255.0, g / 255.0, b / 255.0, a / 255.0)
        elif 64 <= type_id <= 75:
            sc = _parse_strip_chunk(body, body_pos, body_size, type_id, flags)
            for (cw, strip_verts) in sc.strips:
                if len(strip_verts) < 3:
                    continue
                _emit_strip_mesh(state, strip_verts, cw, out_meshes, flags)
        # Other chunks (Volume, MipmapDAdjust, SpecularExponent,
        # Unknown) carry render state we don't surface to the viewer;
        # ignored intentionally.


def _resolve_material_flags(
    state: "_NinjaChunkState",
    strip_flags: int,
) -> Tuple[str, bool, Optional[dict], Optional[dict]]:
    """Derive the per-submesh render-state flags for an emitting strip.

    Returns ``(blend_mode, two_sided, alpha_test, alpha_blend)`` from the
    state's most-recent BlendAlpha + Tiny chunks and the strip chunk's
    own flags byte (bit 0x04 = double-sided per Phantasmal's
    ``parseStripChunk``; mirrored in ``material.decode_strip_chunk_flags``).
    Defaults are the opaque/single-sided baseline. (Phase 3, 2026-06-20.)
    """
    blend_mode = "none"
    alpha_blend: Optional[dict] = None
    if state.blend is not None:
        blend_mode = state.blend.mode
        alpha_blend = {"src": state.blend.src_factor, "dst": state.blend.dst_factor}
    two_sided = bool(strip_flags & 0x04)
    alpha_test: Optional[dict] = None
    if state.tiny_alpha_bits:
        # Reconstruct an 8-bit threshold from the 3-bit overlay (mirrors
        # material.aggregate_submesh_state).
        threshold = (state.tiny_alpha_bits << 5) & 0xE0
        alpha_test = {"enabled": True, "threshold": threshold}
    return blend_mode, two_sided, alpha_test, alpha_blend


def _emit_strip_mesh(
    state: _NinjaChunkState,
    strip_verts: List[_StripVertex],
    clockwise: bool,
    out_meshes: List[XjMesh],
    strip_flags: int = 0,
) -> None:
    """Convert one parsed strip into an ``XjMesh`` and append it.

    Each strip becomes its own submesh with locally-renumbered
    indices: we walk the strip vertices and pull positions from
    ``state.vertex_slots`` (and normals; UVs come from the strip
    header itself for vertex types that carry them).

    Strips whose any index points to a missing vertex slot are
    dropped silently (Phantasmal logs a warning instead).

    Vertex positions and normals come pre-baked from
    ``state.vertex_slots`` (transformed into world space during the
    vertex pass — see ``_process_vertex_chunks``). The strip-level
    per-vertex normal override (chunks 67-69) is rotated by the host
    plist node's world matrix as a best-effort approximation; in
    practice PSOBB BB rarely uses strip-level normals.
    """
    M = state.current_world_matrix
    # Default per-vertex color for this strip when no own color is
    # present: the most-recent material-chunk diffuse, else white
    # (psov2 `aClr = aPos.color || this.color`, NinjaModel.js:863).
    default_color = state.diffuse_color if state.diffuse_color is not None else (1.0, 1.0, 1.0, 1.0)
    local_verts: List[XjVertex] = []
    local_indices: List[int] = []
    for sv in strip_verts:
        slot = state.vertex_slots.get(sv.index)
        if slot is None:
            return  # broken strip — skip whole submesh
        if sv.normal is not None:
            # Rotate strip-level normal by host plist's matrix.
            nx, ny, nz = sv.normal
            normal = (
                M[0] * nx + M[1] * ny + M[2] * nz,
                M[4] * nx + M[5] * ny + M[6] * nz,
                M[8] * nx + M[9] * ny + M[10] * nz,
            )
        else:
            normal = slot.normal or (0.0, 1.0, 0.0)
        uv = sv.uv if sv.uv is not None else (0.0, 0.0)
        # Color precedence: strip-vertex own color > vertex-chunk slot
        # color > material diffuse default. Floor alpha to 0.3 so a
        # fully-transparent authored color can't make the submesh vanish
        # (NinjaModel.js:867).
        if sv.color is not None:
            cr, cg, cb, ca = sv.color
        elif slot.color is not None:
            cr, cg, cb, ca = slot.color
        else:
            cr, cg, cb, ca = default_color
        if ca < 0.3:
            ca = 0.3
        local_indices.append(len(local_verts))
        local_verts.append(XjVertex(pos=slot.pos, normal=normal, uv=uv, color=(cr, cg, cb, ca)))

    tri_indices = _tristrip_to_triangles(local_indices, cw=clockwise)
    if not tri_indices:
        return

    # Bounding sphere from local vertex positions.
    xs = [v.pos[0] for v in local_verts]
    ys = [v.pos[1] for v in local_verts]
    zs = [v.pos[2] for v in local_verts]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    cz = (min(zs) + max(zs)) / 2.0
    r = 0.0
    for v in local_verts:
        dx = v.pos[0] - cx
        dy = v.pos[1] - cy
        dz = v.pos[2] - cz
        d2 = dx * dx + dy * dy + dz * dz
        if d2 > r:
            r = d2
    r = math.sqrt(r) if r > 0 else 0.0

    # Tag this submesh with diagnostic transform metadata.
    #
    # Because the strip's vertices are already in world space (baked
    # at vertex-pass time using the OWNING bone's matrix — the strip's
    # vertex slots can come from many bones, especially for cache/replay
    # patterns), there is no single "node-local-to-world" matrix that
    # captures the strip's pose. We therefore report:
    #   ``world_position`` = the strip's vertex AABB centre in world
    #                        space (i.e. where this sub-mesh sits in
    #                        the model). This is what the e2e tests
    #                        assert "non-zero" / "varies across
    #                        sub-meshes" against, and it's what an
    #                        editor-side debugging viewer wants to
    #                        place a label at.
    #   ``world_matrix``    = identity (vertices are already
    #                        world-space; no further transform needed).
    #   ``world_rotation_euler`` / ``world_scale`` = identity
    #                        components — see ``world_matrix``.
    #
    # The frontend MUST treat ``vertices_pre_transformed = True``
    # (set in ``server.py::_xj_meshes_to_payload``) as authoritative:
    # it MUST NOT compose ``world_position`` into ``Mesh.position``,
    # else every submesh will be doubly-offset. The field is purely a
    # diagnostic / scene-graph anchor.
    blend_mode, two_sided, alpha_test, alpha_blend = _resolve_material_flags(
        state, strip_flags
    )
    out_meshes.append(XjMesh(
        vertices=local_verts,
        indices=tri_indices,
        material_id=max(0, state.texture_id),
        bounding_sphere=(cx, cy, cz, r),
        world_position=(cx, cy, cz),
        world_rotation_euler=(0.0, 0.0, 0.0),
        world_scale=(1.0, 1.0, 1.0),
        world_matrix=tuple(_mat4_identity()),
        blend_mode=blend_mode,
        two_sided=two_sided,
        alpha_test=alpha_test,
        alpha_blend=alpha_blend,
    ))


# ---------------------------------------------------------------------------
# Tree walker (recursive over mesh-tree nodes)
# ---------------------------------------------------------------------------


def _decompose_world_xform(
    M: List[float],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    """Decompose a row-major 4x4 ``M = T * R * S`` into (pos, euler_xyz, scale).

    Translation is read directly from the last column. Per-axis scale
    is the L2-norm of each column of the upper-3x3. The unscaled
    rotation matrix R' is obtained by dividing each column by its
    scale factor; we then read Euler-XYZ angles from R' using the
    standard formula:

        ry = asin(-R'[0,2])
        if cos(ry) > eps:
            rx = atan2(R'[1,2], R'[2,2])
            rz = atan2(R'[0,1], R'[0,0])
        else:                     # gimbal lock: y near +/- pi/2
            rx = atan2(-R'[2,1], R'[1,1])
            rz = 0

    A column with zero norm collapses to scale=0 + identity rotation
    contribution (we leave the column as-is to avoid divide-by-zero).
    """
    # Translation column (row-major: row i, col 3 = M[i*4+3]).
    px = M[0 * 4 + 3]
    py = M[1 * 4 + 3]
    pz = M[2 * 4 + 3]

    # Per-column L2 norms of the upper-3x3 -> scale factors.
    def _col(i):
        return (M[0 * 4 + i], M[1 * 4 + i], M[2 * 4 + i])
    c0 = _col(0); c1 = _col(1); c2 = _col(2)
    sx = math.sqrt(c0[0] * c0[0] + c0[1] * c0[1] + c0[2] * c0[2])
    sy = math.sqrt(c1[0] * c1[0] + c1[1] * c1[1] + c1[2] * c1[2])
    sz = math.sqrt(c2[0] * c2[0] + c2[1] * c2[1] + c2[2] * c2[2])

    inv_sx = 1.0 / sx if sx > 1e-12 else 0.0
    inv_sy = 1.0 / sy if sy > 1e-12 else 0.0
    inv_sz = 1.0 / sz if sz > 1e-12 else 0.0

    # Build rotation matrix entries (R') by normalizing columns.
    r00 = c0[0] * inv_sx; r10 = c0[1] * inv_sx; r20 = c0[2] * inv_sx
    r01 = c1[0] * inv_sy; r11 = c1[1] * inv_sy; r21 = c1[2] * inv_sy
    r02 = c2[0] * inv_sz; r12 = c2[1] * inv_sz; r22 = c2[2] * inv_sz

    # Euler XYZ extraction.
    s = max(-1.0, min(1.0, -r02))
    ry = math.asin(s)
    if abs(r02) < 1.0 - 1e-6:
        rx = math.atan2(r12, r22)
        rz = math.atan2(r01, r00)
    else:
        rx = math.atan2(-r21, r11)
        rz = 0.0

    return (px, py, pz), (rx, ry, rz), (sx, sy, sz)


def _walk_tree(
    body: bytes,
    state: _NinjaChunkState,
    out_meshes: List[XjMesh],
    root_off: int = 0,
    *,
    ignore_hide: bool = False,
) -> None:
    """DFS the MeshTreeNode hierarchy, processing every node's mesh.

    Per Phantasmal's parseNjModel, vertex chunks update the global
    slot table; polygon chunks emit submeshes using the current state.
    Order matters: the root mesh's polygon list often references
    vertices that CHILD bones define (PSOBB caches the strip chunks
    at the root via type-4 CachePolygonList, then a leaf's plist
    triggers a type-5 DrawPolygonList that replays them after all
    vertex chunks have been seen).

    Each node carries a 4x4 local-to-parent transform built from its
    translation, BAMS rotation (XYZ Euler unless EVAL_ZXY_ANG flips it
    to ZXY), and per-axis scale. We accumulate world matrices during a
    pre-order DFS and tag every emitted ``XjMesh`` with the node's
    local-to-world matrix. ``EVAL_HIDE`` skips mesh emission for that
    node; ``EVAL_BREAK`` stops the descent at that node (children are
    not visited). ``EVAL_UNIT_POS/ANG/SCL`` zero out the corresponding
    portion of that node's local matrix. ``EVAL_SKIP`` collapses the
    whole local matrix to identity. ``EVAL_SHAPE_SKIP`` is treated as
    HIDE + BREAK.

    To handle the cache-replay strip pattern (root caches strips with
    type-4, leaf replays with type-5) we do TWO passes:
      1. Vertex pass — for every node with a mesh, feed its vlist into
         ``state.vertex_slots`` (in DFS pre-order).
      2. Polygon pass — for every node with a mesh, run its plist into
         ``out_meshes`` (in DFS pre-order). The current world matrix
         is recorded on ``state`` so ``_emit_strip_mesh`` can tag it.

    Cycles are broken via a visited set per pass.
    """
    n = len(body)

    def _dfs_collect_visited_nodes() -> List[Tuple[int, int, List[float], int]]:
        """Pre-order DFS the mesh tree, returning ``(off, mesh_ptr, world_matrix, eval_flags)``.

        Honors eval-flag bits: BREAK / SHAPE_SKIP prune the descent at
        the current node (children not visited); HIDE / SHAPE_SKIP
        cause us to emit ``mesh_ptr=0`` so the second pass skips strip
        emission for this node (vertex chunks still apply because
        Phantasmal's vertex slot table is flat across the whole model
        — a HIDDEN node may still feed vertex slots referenced by a
        sibling's strips).
        """
        out: List[Tuple[int, int, List[float], int]] = []
        visited: set = set()

        # Iterative DFS with explicit (off, parent_world_matrix) stack.
        # We push children AFTER siblings so that sibling pop happens
        # AFTER child pop (LIFO). This preserves Phantasmal's pre-order
        # traversal: visit node -> visit child -> visit sibling.
        stack: List[Tuple[int, List[float]]] = [(root_off, _mat4_identity())]
        # Hard cap: protect against malicious/malformed offsets that
        # might cause node count to balloon. Real PSOBB models top out
        # around 200 mesh tree nodes.
        MAX_NODES = 4096
        while stack and len(out) < MAX_NODES:
            off, parent_world = stack.pop()
            if off in visited or off + _MESH_TREE_NODE_SIZE > n:
                continue
            visited.add(off)
            full = _read_mesh_tree_node_full(body, off)
            if full is None:
                continue
            ef, mesh_ptr, pos, rot_bams, scale, child_ptr, next_ptr = full

            # Apply eval-flag overrides to local TRS.
            if ef & EVAL_SKIP:
                local_M = _mat4_identity()
            else:
                lpos = (0.0, 0.0, 0.0) if (ef & EVAL_UNIT_POS) else pos
                if ef & EVAL_UNIT_ANG:
                    lrot = (0.0, 0.0, 0.0)
                else:
                    lrot = (
                        rot_bams[0] * _BAMS_TO_RAD,
                        rot_bams[1] * _BAMS_TO_RAD,
                        rot_bams[2] * _BAMS_TO_RAD,
                    )
                lscale = (1.0, 1.0, 1.0) if (ef & EVAL_UNIT_SCL) else scale
                zxy = bool(ef & EVAL_ZXY_ANG)
                local_M = _mat4_compose_trs(lpos, lrot, lscale, zxy_order=zxy)

            world_M = _mat4_mul(parent_world, local_M)

            # Decide whether this node's mesh is drawn. HIDE or
            # SHAPE_SKIP on this node disables drawing, but we still
            # walk children (unless BREAK / SHAPE_SKIP) and we still
            # let vertex chunks contribute to the global slot table —
            # because Phantasmal's NjObject walker does the same. To
            # achieve "vertex chunks yes, strip chunks no," the second
            # pass checks the recorded eval_flags and only RUNS
            # _process_polygon_chunks when neither HIDE nor SHAPE_SKIP
            # is set. We pass the eval_flags through so it can decide.
            if mesh_ptr and mesh_ptr + _NJ_MESH_SIZE <= n:
                out.append((off, mesh_ptr, world_M, ef))

            # PSOBB BB tree-traversal rules (empirically derived from
            # ``bm_ene_gibbles_low.bml#lo_gibb_body.nj`` etc.):
            #
            # The official SEGA Ninja SDK uses ``EVAL_BREAK`` to mean
            # "stop visiting siblings" and treats ``EVAL_SHAPE_SKIP`` as
            # "stop recursing into children". In PSOBB BB data both
            # flags appear on intermediate nodes whose ``next_ptr`` and
            # ``child_ptr`` lead to real, visible geometry — honoring
            # the SDK semantics drops ~100% of the model.
            #
            # The Blender plugin ``pso-blender/pso_blender/xj.py``
            # confirms this: its tree walker (``make_mesh_tree``) pushes
            # both ``child`` and ``next`` unconditionally, treating the
            # eval flags as render-state metadata to round-trip rather
            # than topology gates. Phantasmal Ninja does the same — it
            # walks the link list strictly, then evaluates HIDE/UNIT_*
            # at draw time.
            #
            # We therefore IGNORE BREAK and SHAPE_SKIP for traversal
            # (they ride along on the XjMesh's eval_flags is enough)
            # and rely on HIDE in pass 2 to suppress drawing the
            # specific nodes the data wants hidden.
            #
            # Push sibling FIRST so the LIFO pops the child first
            # (preserving pre-order: parent, child, sibling).
            if next_ptr and next_ptr not in visited:
                stack.append((next_ptr, parent_world))
            if child_ptr and child_ptr not in visited:
                stack.append((child_ptr, world_M))
        return out

    visited_nodes = _dfs_collect_visited_nodes()

    # Pass 1: vertex chunks. Phantasmal accumulates vertex slots in
    # one big sparse list across the whole model, so order does not
    # affect the final geometry — but we keep it pre-order for
    # determinism.
    #
    # We thread the OWNING NODE's world matrix into the state slot so
    # ``_process_vertex_chunks`` can bake each vertex into world space
    # as it parses. This is the heart of the bone-fix: vertices stored
    # as world-space coordinates render coherently regardless of which
    # bone's strips later reference them (including cache-replay).
    for (_off, mesh_ptr, world_M, _ef) in visited_nodes:
        mesh = _read_nj_mesh(body, mesh_ptr)
        if mesh is None:
            continue
        vlist_off, _plist_off, _bbox = mesh
        if vlist_off:
            state.current_world_matrix = world_M
            _process_vertex_chunks(body, vlist_off, state)

    # Pass 2: polygon chunks. Each call resets state.cache_active to
    # None (matching Phantasmal's "fresh PolygonChunkProcessor per
    # parseNjModel"), but state.cached_chunks persists so a leaf's
    # type-5 (DrawPolygonList) can replay strips cached by an earlier
    # bone's type-4 (CachePolygonList).
    #
    # The world matrix for the CURRENT node is parked on
    # ``state.current_world_matrix`` so that ``_emit_strip_mesh`` can
    # tag each XjMesh as it pushes one. Phantasmal's strip emission
    # loop happens deep inside the chunk processor; threading the
    # matrix through every helper would be noisy, so the per-state
    # slot is the cleanest in-band channel.
    for (_off, mesh_ptr, world_M, ef) in visited_nodes:
        mesh = _read_nj_mesh(body, mesh_ptr)
        if mesh is None:
            continue
        _vlist_off, plist_off, _bbox = mesh
        # HIDE suppresses strip emission for THIS node only (children
        # may still draw — they were enqueued during the DFS even when
        # HIDE was set, see comments in `_dfs_collect_visited_nodes`).
        # SHAPE_SKIP also suppresses drawing for this node, on top of
        # already pruning child recursion at the DFS step.
        #
        # ignore_hide=True (env: PSO_XJ_IGNORE_HIDE=1) bypasses this
        # filter so callers can confirm the parser handles legitimate
        # HIDE-flagged data correctly. Empirically, no shipping PSOBB
        # BB model sets HIDE on a mesh-bearing node (see
        # scripts/dump_eval_hide_audit.py); the knob exists for
        # diagnostics + future-proofing against modded data.
        if not ignore_hide and (ef & (EVAL_HIDE | EVAL_SHAPE_SKIP)):
            continue
        if plist_off:
            state.current_world_matrix = world_M
            _process_polygon_chunks(body, plist_off, state, out_meshes)
    state.current_world_matrix = _mat4_identity()


# ---------------------------------------------------------------------------
# Public: parser entry points
# ---------------------------------------------------------------------------


def parse_xj_njcm(
    payload: bytes,
    pof0_payload: Optional[bytes] = None,
    *,
    ignore_hide: Optional[bool] = None,
) -> List[XjMesh]:
    """Parse the body of a single ``NJCM`` chunk into a list of submeshes.

    Parameters
    ----------
    payload:
        The chunk PAYLOAD bytes (i.e. NOT including the 8-byte IFF
        header). File-relative pointers stored inside the buffer are
        interpreted as offsets relative to the start of ``payload``.
    pof0_payload:
        Reserved for future use. The Ninja chunk parser does not need
        POF0 to find pointers — they are stored in-place — but we
        accept the parameter to keep API parity with callers that
        already pass it (server.py, e2e tests).
    ignore_hide:
        When True, do NOT skip strip emission for nodes flagged with
        EVAL_HIDE or EVAL_SHAPE_SKIP. Default (None) reads the env-var
        ``PSO_XJ_IGNORE_HIDE`` (truthy values turn this on). Empirically
        no shipping PSOBB BB data sets HIDE on a mesh-bearing node, so
        this flag is for diagnostics + modded-data future-proofing.

    Returns
    -------
    List[XjMesh]
        One submesh per ``Strip`` chunk seen during the tree walk, in
        traversal order. May be empty if the file has no strip chunks
        (e.g. corrupt files, animation-only ``.njm`` payloads passed
        in by mistake, etc).

    Raises
    ------
    ValueError
        On obvious corruption (truncated mesh-tree node, etc.).
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError("parse_xj_njcm: payload must be bytes-like")
    if len(payload) < _MESH_TREE_NODE_SIZE:
        raise ValueError(
            f"parse_xj_njcm: payload too small "
            f"({len(payload)} bytes < {_MESH_TREE_NODE_SIZE})"
        )

    body = bytes(payload)

    # Sanity-check the root MeshTreeNode looks plausible.
    root = _read_mesh_tree_node(body, 0)
    if root is None:
        raise ValueError("parse_xj_njcm: root mesh tree node truncated")

    if ignore_hide is None:
        ignore_hide = _IGNORE_HIDE_DEFAULT
    state = _NinjaChunkState()
    out: List[XjMesh] = []
    try:
        _walk_tree(body, state, out, root_off=0, ignore_hide=bool(ignore_hide))
    except struct.error as e:
        raise ValueError(f"parse_xj_njcm: structure read failed: {e}")
    except (IndexError, KeyError) as e:
        raise ValueError(f"parse_xj_njcm: corrupt offsets: {e}")

    return out


def parse_nj_file(buf: bytes, *, ignore_hide: Optional[bool] = None) -> List[XjMesh]:
    """Parse a complete ``.nj`` file (IFF container) and return all meshes.

    Parameters
    ----------
    buf:
        Full bytes of a ``.nj`` file. The file may include leading
        ``NJTL`` (texture name list) and trailing ``POF0`` chunks; we
        skip everything except ``NJCM``.
    ignore_hide:
        Forward to ``parse_xj_njcm``; see its docstring. Default reads
        the ``PSO_XJ_IGNORE_HIDE`` env var.

    Returns
    -------
    List[XjMesh]
        Concatenated submeshes from every ``NJCM`` chunk in the file.
        Empty when the file has no ``NJCM`` chunk or no parseable
        geometry. Each ``NJCM`` re-initialises the chunk-processor
        state — vertex slots and texture id do not leak between
        chunks.

    Raises
    ------
    ValueError
        When the IFF wrapper is malformed.
    """
    if not isinstance(buf, (bytes, bytearray, memoryview)):
        raise ValueError("parse_nj_file: input must be bytes-like")
    chunks = parse_iff(buf)
    if not chunks:
        return []

    out: List[XjMesh] = []
    for i, c in enumerate(chunks):
        if c.type != "NJCM":
            continue
        # POF0 (when present) is the next chunk after NJCM but the
        # chunk parser does not need it. Pass-through for API parity.
        pof0_data: Optional[bytes] = None
        if i + 1 < len(chunks) and chunks[i + 1].type == "POF0":
            pof0_data = chunks[i + 1].data
        out.extend(parse_xj_njcm(c.data, pof0_data, ignore_hide=ignore_hide))
    return out


# ---------------------------------------------------------------------------
# Skeleton (Ninja MeshTreeNode) parser
# ---------------------------------------------------------------------------
#
# Walks the NJCM mesh-tree linked-list to extract bone positions/rotations.
# This is INDEPENDENT of the mesh chunk parsing path — it only needs the
# 52-byte tree nodes — so callers who only want bones can avoid the
# (more expensive) chunk-stream walk.


@dataclass
class XjBone:
    """One node in the Ninja MeshTreeNode hierarchy.

    Attributes
    ----------
    index:
        0-based traversal index assigned during the walk. Stable for a
        given file. The root is always index 0.
    parent:
        Index of this node's parent in the same list, or -1 for the
        root.
    position:
        ``(x, y, z)`` in model space (raw Ninja units; the consumer is
        responsible for applying the same scale as the mesh group).
    rotation:
        ``(rx, ry, rz)`` integer Ninja angles (BAMs — 0x10000 = 360°).
        We surface the raw integer here to avoid lossy double-conversion;
        consumers can convert to radians via ``r * 2*pi / 0x10000``.
    scale:
        ``(sx, sy, sz)`` per-axis scale factors (1.0 = identity). Most
        bones carry (1, 1, 1) but some PSOBB BB models stash non-unit
        scale here (e.g. shaping props by axis).
    eval_flags:
        Raw u32 NjsObject eval-flag bitfield from the source mesh-tree
        node. Bits surface the SEGA Ninja SDK semantics:
        UNIT_POS=0x01 (use 0 translation), UNIT_ANG=0x02 (use 0 rot),
        UNIT_SCL=0x04 (use 1 scale), HIDE=0x08 (don't draw), BREAK=0x10,
        ZXY_ANG=0x20 (rotation order is ZXY not ZYX), SKIP=0x40 (use
        identity local matrix), SHAPE_SKIP=0x80. The skinned-path
        consumer (animation) MUST honor UNIT_*/SKIP/ZXY_ANG when
        composing per-bone bind matrices to match the world-baked
        renderer; without this, bones with UNIT_POS deviate from their
        bake-pipeline equivalent (regression spotted on De Rol Le whose
        head bones use UNIT_POS|UNIT_SCL).
    """
    index: int
    parent: int
    position: Tuple[float, float, float]
    rotation: Tuple[int, int, int]
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    eval_flags: int = 0


def _parse_mesh_tree_node_for_bones(body: bytes, offset: int):
    """Read one 52-byte MeshTreeNode at ``offset`` for the bone walker.

    Returns ``(pos, rot, scale, eval_flags, child_ptr, next_ptr)`` or
    ``None`` on truncation. Distinct from ``_read_mesh_tree_node`` above
    which returns a 7-tuple optimised for the mesh DFS.
    """
    if offset < 0 or offset + _MESH_TREE_NODE_SIZE > len(body):
        return None
    f = struct.unpack_from(_MESH_TREE_NODE_FMT, body, offset)
    eval_flags = f[0]
    pos = (f[2], f[3], f[4])
    rot = (f[5], f[6], f[7])
    scale = (f[8], f[9], f[10])
    child_ptr = f[11]
    next_ptr = f[12]
    return pos, rot, scale, eval_flags, child_ptr, next_ptr


def parse_skeleton(buf: bytes) -> List[XjBone]:
    """Parse the bone hierarchy from a complete ``.nj`` IFF byte string.

    Walks the mesh-tree linked list anchored at the start of the first
    NJCM chunk's body and flattens it into a depth-first list with
    parent indices. The root is always at index 0 with parent = -1.

    Returns
    -------
    List[XjBone]
        Ordered bone list. Empty if the input has no NJCM chunk or
        the root node fails the cheap plausibility test (positions
        all zero AND no children — typical for non-skinned meshes).

    Notes
    -----
    Heuristic guard: many PSOBB props store a single root node with
    no children, in which case we return a one-bone list. Skinned
    character files (``plX*.nj``) typically expose 20-40 bones.
    """
    if not isinstance(buf, (bytes, bytearray, memoryview)):
        raise ValueError("parse_skeleton: input must be bytes-like")
    chunks = parse_iff(buf)
    njcm = next((c for c in chunks if c.type == "NJCM"), None)
    if njcm is None:
        return []
    body = njcm.data

    bones: List[XjBone] = []
    visited: set = set()
    stack: list = [(0, -1)]
    # Hard cap so a circular pointer chain can't run forever. Real
    # PSO skeletons top out around ~50 bones.
    MAX_BONES = 256
    while stack and len(bones) < MAX_BONES:
        off, parent = stack.pop()
        if off in visited:
            continue
        visited.add(off)
        node = _parse_mesh_tree_node_for_bones(body, off)
        if node is None:
            continue
        pos, rot, scale, eval_flags, child_ptr, next_ptr = node
        my_index = len(bones)
        bones.append(XjBone(
            index=my_index,
            parent=parent,
            position=pos,
            rotation=rot,
            scale=scale,
            eval_flags=eval_flags,
        ))
        # Push next BEFORE child so child is processed first (DFS).
        # ``next_ptr`` is a SIBLING — it shares our parent.
        if next_ptr and next_ptr not in visited:
            stack.append((next_ptr, parent))
        if child_ptr and child_ptr not in visited:
            stack.append((child_ptr, my_index))

    return bones


# ---------------------------------------------------------------------------
# Skinning-friendly parser path (2026-04-24).
# ---------------------------------------------------------------------------
#
# The default ``parse_nj_file`` / ``parse_xj_njcm`` path bakes vertex
# positions into world space at the bind pose. That makes static
# rendering trivial (no bone transforms in the frontend) but rules out
# skeletal animation — once a vertex is in world space we've lost the
# bone-relative information we need to re-pose it.
#
# This second path is the "skin-friendly" output:
#
#   * Vertices stay in BONE-LOCAL coordinates (NOT world-space-baked).
#   * Each ``XjVertex.bone_idx`` records the DFS-order index of the
#     mesh-tree node whose vertex chunk supplied this vertex slot.
#   * The skeleton bone matrix at bind pose is computed in JS from the
#     ``XjBone`` hierarchy via the same TRS composition as the bake
#     parser — so static rendering still matches the world-space path.
#   * Animated rendering: at frame N the frontend computes per-bone
#     animated matrices, then applies the matrix-of-the-owning-bone to
#     each vertex (Option A re-bake from the spec; CPU-side, ~30 fps
#     for typical PSOBB monster bones counts).
#
# Wire-format note: this parser path emits ``XjVertex`` instances whose
# ``pos`` is bone-LOCAL and whose ``bone_idx`` is the owning DFS index.
# The default bake-to-world path emits ``bone_idx == -1`` to signal
# "vertex is in world space; no per-bone transform needed". The
# frontend dispatches on the first vertex's ``bone_idx`` value.
#
# We do NOT modify ``parse_xj_njcm`` itself — adding a parameter risks
# breaking other callers (the bug-hunt agent is iterating on that path
# in parallel). Instead this is a second entry point with the
# minimum-shared-code structure: we replicate the tree-walker but feed
# strips into a separate "no-bake" emitter.


def _process_vertex_chunks_skinned(
    body: bytes,
    start_off: int,
    state: "_NinjaChunkState",
    bone_idx: int,
    bone_idx_map: dict,
) -> None:
    """Variant of ``_process_vertex_chunks`` that does NOT bake to world space.

    Each parsed slot is stored at its bone-LOCAL position with
    ``bone_idx`` set to the owning node's DFS index. Used by the
    skin-friendly parser path; the world-space-baking variant remains
    the default for static rendering.

    ``bone_idx_map`` is a dict keyed by vertex slot index; we update it
    with the owning bone's DFS index. The dict lives on the parser
    caller (NOT on the slotted ``_NinjaChunkState``) so we don't have
    to widen the existing class for this code path.
    """
    chunks = _walk_chunk_stream(body, start_off)
    for (hdr, type_id, flags, body_pos, body_size) in chunks:
        if 32 <= type_id <= 50:
            for cv in _parse_vertex_chunk(body, body_pos, body_size, type_id, flags):
                # Store bone-local position; remember which bone supplied it.
                state.vertex_slots[cv.index] = _ChunkVertex(
                    index=cv.index,
                    pos=cv.pos,
                    normal=cv.normal,
                    color=cv.color,
                )
                bone_idx_map[cv.index] = bone_idx


def _emit_strip_mesh_skinned(
    state: "_NinjaChunkState",
    strip_verts: List["_StripVertex"],
    clockwise: bool,
    out_meshes: List[XjMesh],
    bone_idx_map: dict,
    strip_flags: int = 0,
) -> None:
    """Variant of ``_emit_strip_mesh`` that surfaces bone-local + bone_idx.

    The emitted ``XjVertex.pos`` is the bone-LOCAL position of its
    vertex slot (no world-matrix application). Each vertex's
    ``bone_idx`` is the DFS index of the bone whose vertex chunk
    supplied that slot — which may DIFFER between vertices in the same
    strip (PSOBB cache/replay pattern stamps strips that pull from
    multiple bones' slots).

    ``bone_idx_map`` is a ``dict[slot_idx -> bone_dfs_idx]`` populated
    by ``_process_vertex_chunks_skinned`` during the vertex pass.
    """
    default_color = state.diffuse_color if state.diffuse_color is not None else (1.0, 1.0, 1.0, 1.0)
    local_verts: List[XjVertex] = []
    local_indices: List[int] = []
    for sv in strip_verts:
        slot = state.vertex_slots.get(sv.index)
        if slot is None:
            return  # broken strip — skip whole submesh
        normal = sv.normal if sv.normal is not None else (slot.normal or (0.0, 1.0, 0.0))
        uv = sv.uv if sv.uv is not None else (0.0, 0.0)
        owner = bone_idx_map.get(sv.index, -1)
        # Color precedence: strip color > slot color > material diffuse;
        # alpha floored to 0.3 (mirrors the world-baked emitter).
        if sv.color is not None:
            cr, cg, cb, ca = sv.color
        elif slot.color is not None:
            cr, cg, cb, ca = slot.color
        else:
            cr, cg, cb, ca = default_color
        if ca < 0.3:
            ca = 0.3
        local_indices.append(len(local_verts))
        local_verts.append(XjVertex(pos=slot.pos, normal=normal, uv=uv, bone_idx=owner, color=(cr, cg, cb, ca)))

    tri_indices = _tristrip_to_triangles(local_indices, cw=clockwise)
    if not tri_indices:
        return

    # Bounding sphere from the BONE-LOCAL vertex positions. Less useful
    # than the world-space sphere (the consumer needs the animated pose
    # to derive a true bound) but stable for sanity checks.
    xs = [v.pos[0] for v in local_verts]
    ys = [v.pos[1] for v in local_verts]
    zs = [v.pos[2] for v in local_verts]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    cz = (min(zs) + max(zs)) / 2.0
    r = 0.0
    for v in local_verts:
        dx = v.pos[0] - cx
        dy = v.pos[1] - cy
        dz = v.pos[2] - cz
        d2 = dx * dx + dy * dy + dz * dz
        if d2 > r:
            r = d2
    r = math.sqrt(r) if r > 0 else 0.0

    blend_mode, two_sided, alpha_test, alpha_blend = _resolve_material_flags(
        state, strip_flags
    )
    out_meshes.append(XjMesh(
        vertices=local_verts,
        indices=tri_indices,
        material_id=max(0, state.texture_id),
        bounding_sphere=(cx, cy, cz, r),
        # No world matrix — vertices are bone-local. The consumer
        # builds per-bone matrices from the skeleton + animation.
        world_position=(0.0, 0.0, 0.0),
        world_rotation_euler=(0.0, 0.0, 0.0),
        world_scale=(1.0, 1.0, 1.0),
        world_matrix=tuple(_mat4_identity()),
        blend_mode=blend_mode,
        two_sided=two_sided,
        alpha_test=alpha_test,
        alpha_blend=alpha_blend,
    ))


def _process_polygon_chunks_skinned(
    body: bytes,
    start_off: int,
    state: "_NinjaChunkState",
    out_meshes: List[XjMesh],
    bone_idx_map: dict,
) -> None:
    """Variant of ``_process_polygon_chunks`` that calls ``_emit_strip_mesh_skinned``.

    Same cache/replay semantics as the bake variant; only the emit step
    differs.
    """
    state.cache_active = None
    chunks = _walk_chunk_stream(body, start_off)
    _process_polygon_chunk_list_skinned(body, chunks, state, out_meshes, bone_idx_map)


def _process_polygon_chunk_list_skinned(
    body: bytes,
    chunks: List[Tuple[int, int, int, int, int]],
    state: "_NinjaChunkState",
    out_meshes: List[XjMesh],
    bone_idx_map: dict,
) -> None:
    """Recursive variant matching ``_process_polygon_chunk_list``."""
    for (hdr, type_id, flags, body_pos, body_size) in chunks:
        if state.cache_active is not None and type_id != 5:
            state.cached_chunks[state.cache_active].append(
                (hdr, type_id, flags, body_pos, body_size)
            )
            continue
        if type_id == 4:
            state.cache_active = flags
            state.cached_chunks[flags] = []
        elif type_id == 5:
            state.cache_active = None
            cached = state.cached_chunks.get(flags)
            if cached:
                _process_polygon_chunk_list_skinned(body, cached, state, out_meshes, bone_idx_map)
        elif type_id == 1:
            # BlendAlpha — track active blend mode (see world-baked variant).
            try:
                from .material import decode_blend_alpha_chunk as _dbac
                state.blend = _dbac(flags)
            except Exception:
                state.blend = None
        elif 8 <= type_id <= 9:
            if body_size >= 2:
                (tex_word,) = struct.unpack_from("<H", body, body_pos)
                state.texture_id = tex_word & 0x1FFF
                state.tiny_alpha_bits = (tex_word >> 13) & 0x07
        elif 17 <= type_id <= 23:
            # Material diffuse default color (see the world-baked variant).
            try:
                from .material import decode_material_chunk as _dmc
                mp = _dmc(type_id, flags, body[body_pos:body_pos + body_size])
            except Exception:
                mp = None
            if mp is not None and mp.diffuse is not None:
                r, g, b, a = mp.diffuse.to_tuple()
                state.diffuse_color = (r / 255.0, g / 255.0, b / 255.0, a / 255.0)
        elif 64 <= type_id <= 75:
            sc = _parse_strip_chunk(body, body_pos, body_size, type_id, flags)
            for (cw, strip_verts) in sc.strips:
                if len(strip_verts) < 3:
                    continue
                _emit_strip_mesh_skinned(state, strip_verts, cw, out_meshes, bone_idx_map, flags)


def parse_xj_njcm_skinned(
    payload: bytes,
    *,
    ignore_hide: Optional[bool] = None,
) -> Tuple[List[XjMesh], List[XjBone]]:
    """Parse an NJCM payload into bone-local meshes + skeleton.

    Same input expectations as ``parse_xj_njcm`` (NJCM body bytes; not
    the IFF-wrapped file). Returns a tuple:

      * ``meshes`` — list[XjMesh] with bone-local positions and
        per-vertex ``bone_idx`` populated.
      * ``bones`` — flattened DFS bone list. The ``bone_idx`` values on
        each vertex index into this list. Includes EVERY mesh-tree
        node (skinned or not) so animation tracks line up positionally
        with PSOBB's NJM mDataTable layout.
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError("parse_xj_njcm_skinned: payload must be bytes-like")
    if len(payload) < _MESH_TREE_NODE_SIZE:
        raise ValueError(
            f"parse_xj_njcm_skinned: payload too small "
            f"({len(payload)} < {_MESH_TREE_NODE_SIZE})"
        )

    body = bytes(payload)
    root = _read_mesh_tree_node(body, 0)
    if root is None:
        raise ValueError("parse_xj_njcm_skinned: root mesh tree node truncated")

    if ignore_hide is None:
        ignore_hide = _IGNORE_HIDE_DEFAULT

    state = _NinjaChunkState()
    out_meshes: List[XjMesh] = []
    bones: List[XjBone] = []

    # Pre-pass: build the DFS bone list (same logic as parse_skeleton).
    # We need this BEFORE the chunk pass so the bone_idx assigned to
    # each vertex is the bone's final DFS index, matching the NJM
    # mDataTable order. We also keep the off→bone_idx map so the chunk
    # walker can convert mesh-tree-node offsets to bone indices.
    visited_for_bones: set = set()
    off_to_bone: dict[int, int] = {}
    stack: list = [(0, -1)]
    MAX_BONES = 4096
    while stack and len(bones) < MAX_BONES:
        off, parent = stack.pop()
        if off in visited_for_bones:
            continue
        visited_for_bones.add(off)
        node = _parse_mesh_tree_node_for_bones(body, off)
        if node is None:
            continue
        pos, rot, scale, eval_flags, child_ptr, next_ptr = node
        my_index = len(bones)
        bones.append(XjBone(
            index=my_index,
            parent=parent,
            position=pos,
            rotation=rot,
            scale=scale,
            eval_flags=eval_flags,
        ))
        off_to_bone[off] = my_index
        if next_ptr and next_ptr not in visited_for_bones:
            stack.append((next_ptr, parent))
        if child_ptr and child_ptr not in visited_for_bones:
            stack.append((child_ptr, my_index))

    # Sidecar map for bone_idx tagging (vertex slot index -> bone DFS index).
    # Lives outside _NinjaChunkState because that class uses __slots__ and
    # we want a zero-touch addition.
    bone_idx_map: dict = {}

    # Vertex pass: each node's vertex chunk loads slots, tagged with
    # the owning DFS bone index (looked up via off_to_bone).
    n = len(body)
    visited_for_chunks: set = set()
    stack2 = [0]
    visit_order: list[int] = []
    while stack2 and len(visit_order) < MAX_BONES:
        off = stack2.pop()
        if off in visited_for_chunks or off + _MESH_TREE_NODE_SIZE > n:
            continue
        visited_for_chunks.add(off)
        full = _read_mesh_tree_node_full(body, off)
        if full is None:
            continue
        ef, mesh_ptr, _pos, _rot, _scale, child_ptr, next_ptr = full
        visit_order.append(off)
        if next_ptr and next_ptr not in visited_for_chunks:
            stack2.append(next_ptr)
        if child_ptr and child_ptr not in visited_for_chunks:
            stack2.append(child_ptr)

    for off in visit_order:
        full = _read_mesh_tree_node_full(body, off)
        if full is None:
            continue
        ef, mesh_ptr, _pos, _rot, _scale, _ch, _nx = full
        if not mesh_ptr or mesh_ptr + _NJ_MESH_SIZE > n:
            continue
        mesh = _read_nj_mesh(body, mesh_ptr)
        if mesh is None:
            continue
        vlist_off, _plist_off, _bbox = mesh
        if vlist_off:
            owner = off_to_bone.get(off, -1)
            _process_vertex_chunks_skinned(body, vlist_off, state, owner, bone_idx_map)

    # Polygon pass: emit bone-local strips with per-vertex bone_idx.
    for off in visit_order:
        full = _read_mesh_tree_node_full(body, off)
        if full is None:
            continue
        ef, mesh_ptr, _pos, _rot, _scale, _ch, _nx = full
        if not mesh_ptr or mesh_ptr + _NJ_MESH_SIZE > n:
            continue
        if not ignore_hide and (ef & (EVAL_HIDE | EVAL_SHAPE_SKIP)):
            continue
        mesh = _read_nj_mesh(body, mesh_ptr)
        if mesh is None:
            continue
        _vlist_off, plist_off, _bbox = mesh
        if plist_off:
            _process_polygon_chunks_skinned(body, plist_off, state, out_meshes, bone_idx_map)

    return out_meshes, bones


def parse_nj_skinned(
    buf: bytes, *, ignore_hide: Optional[bool] = None
) -> Tuple[List[XjMesh], List[XjBone]]:
    """Parse a complete `.nj` IFF byte string into bone-local meshes + skeleton.

    Companion to ``parse_nj_file`` for the skinning-friendly path; the
    output is suitable for the model viewer's animation playback. Each
    ``XjVertex.pos`` is in BONE-LOCAL space (NOT world-baked), and the
    ``bone_idx`` field identifies the owning bone in the returned
    ``bones`` list.

    Returns ``([], [])`` for files with no NJCM chunk.
    """
    if not isinstance(buf, (bytes, bytearray, memoryview)):
        raise ValueError("parse_nj_skinned: input must be bytes-like")
    chunks = parse_iff(buf)
    if not chunks:
        return [], []
    out_meshes: List[XjMesh] = []
    out_bones: List[XjBone] = []
    for c in chunks:
        if c.type != "NJCM":
            continue
        meshes, bones = parse_xj_njcm_skinned(c.data, ignore_hide=ignore_hide)
        out_meshes.extend(meshes)
        # Skeletons from multiple NJCMs are concatenated. Real PSOBB
        # files have ONE NJCM, but we keep the loop for symmetry with
        # parse_nj_file.
        if not out_bones:
            out_bones = bones
    return out_meshes, out_bones


# ---------------------------------------------------------------------------
# Material chunk decode helper — added 2026-04-25 for the Material Inspector.
# ---------------------------------------------------------------------------
#
# The legacy parser above intentionally ignores chunk types 17-23 (it
# only needs to skip past them for size). The Material Inspector tab
# wants to SEE those chunks so the user can inspect / edit per-submesh
# diffuse / blend mode / two-sided / depth flags. Rather than refactor
# this 2k-line parser, we expose a thin shim that re-exports the
# dedicated decoder in ``formats/material.py`` so callers don't need to
# know which module owns the format spec. See ``formats/material.py``
# for the full chunk-type catalogue + per-field semantics.
def decode_material_chunk(type_id: int, flags: int, body: bytes):
    """Decode a Ninja material chunk (types 17..23).

    Thin re-export of :func:`formats.material.decode_material_chunk` —
    placed here so callers can `from formats.xj import
    decode_material_chunk` without learning about the new module.
    """
    from .material import decode_material_chunk as _impl
    return _impl(type_id, flags, body)


__all__ = [
    "XjVertex",
    "XjMesh",
    "XjBone",
    "parse_xj_njcm",
    "parse_nj_file",
    "parse_skeleton",
    "parse_nj_skinned",
    "parse_xj_njcm_skinned",
    "decode_material_chunk",
]
