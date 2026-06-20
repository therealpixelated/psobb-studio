"""PSOBB ``.rel`` relocation-container WRITER (2026-06-20).

Inverse of the parse-only reader in :mod:`formats.rel`.  A ``.rel`` file
is a *relocatable* container: a flat data section plus a pointer-fixup
table (raw little-endian u16 word-deltas — NOT POF0) plus a 32-byte
trailer.  At load time PSOBB walks the delta table, advancing an
``int*`` cursor by each delta and adding the buffer's load address to the
u32 it lands on (decompile ``load_rel_asset`` / ``load_rel_asset2`` at
``Psobb.exe-05112026.c:404134-404214`` / ``:404642-404669``).

This module implements two REL halves on top of one shared container:

  STEP 0  :func:`encode_rel_pointer_table` / :func:`decode_rel_pointer_table`
           — the load-bearing u16 word-delta codec.  Shared by c+n.rel.
  STEP 1  :func:`assemble_rel` — pad-to-32 + table + pad + 32-byte trailer.
           Generic container framing; shared by c+n.rel.
  STEP 2  :func:`parse_crel_for_writer` / :func:`encode_crel` /
           :func:`build_crel` — the c.rel model + byte-exact emitter.
  STEP 3  :func:`parse_nrel_for_writer` / :func:`encode_nrel` — the n.rel
           (node geometry, ``NrelFmt2``) model + byte-exact emitter.

The STEP-0/STEP-1 primitives are format-agnostic; the n.rel half reuses
them unchanged (it does NOT touch POF0 — n.rel geometry is XJ buffer
descriptors, not NJCM chunk streams, and the relocation table is the
same raw u16 word-delta codec as c.rel, never the NJ/POF0 token form).

c.rel on-disk layout (validated byte-exact against every ``*c.rel`` in
``PSOBB.IO/data/scene``)::

    [ per-node data, in node order ]   for each node i:
        verts[i]  : vertex_count[i] * 12   (vec3f x,y,z)
        faces[i]  : face_count[i]   * 36   (CrelFace, packed)
        mesh[i]   : 16   (CrelMesh: vcount, verts_ptr, fcount, faces_ptr)
    [ node array ]                node_count * 24   (CrelNode)
    [ node array NUL terminator ] 24 zero bytes (mesh_ptr==0 ends array)
    [ payload ]                   u32 head_ptr -> node array start
                                  (+ optional zero pad; 4 bytes is valid)
    [ pointer table ]             pointer_count * 2   (u16 word-deltas)
    [ pad ]                       NUL so (trailer_start) is 32-aligned
    [ trailer ]                   32 bytes

CrelMesh (16B)::

    +0x00 u32 vertex_count
    +0x04 u32 vertices_ptr   (RELOCATED)
    +0x08 i32 face_count
    +0x0C u32 faces_ptr      (RELOCATED)

CrelFace (36B, packed)::

    +0x00 u16 i0  +0x02 u16 i1  +0x04 u16 i2  +0x06 u16 flags
    +0x08 3*f32 normal   +0x14 3*f32 centroid   +0x20 f32 radius

CrelNode (24B)::

    +0x00 u32 mesh_ptr   (RELOCATED)
    +0x04 3*f32 center   +0x10 f32 radius   +0x14 u32 flags

pointer_count invariant for c.rel == ``3 * node_count + 1``
(2 per mesh [verts_ptr, faces_ptr] + 1 per node [mesh_ptr] + 1 payload).

A registered pointer field may legally hold the value 0 — e.g. node-0's
vertices live at file offset 0 in ``map_lobby_01c.rel`` so its
``verts_ptr`` is 0 yet still appears in the table.  The writer flags
pointer *locations* by construction, independent of the stored value.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from formats.rel import (
    NREL_FMT2_MAGIC,
    TRAILER_SIZE,
    RelParseError,
    is_n_rel,
    parse_rel,
    read_mesh_trees,
    read_nrel_chunks,
    read_nrel_header,
    read_texture_names,
)


class RelWriteError(ValueError):
    """Raised when a model cannot be serialised into a valid REL container."""


# ---------------------------------------------------------------------------
# STEP 0 — pointer-fixup table codec (raw u16 word-deltas)
# ---------------------------------------------------------------------------
#
# This is the single most load-bearing primitive: a wrong/extra/missing
# entry corrupts a non-pointer word at load time.  It is the exact
# inverse of ``formats.rel._resolve_pointer_offsets``.

_REL_MAX_DELTA_WORDS = 0xFFFF          # one u16 entry == up to 0xFFFF dwords
_REL_MAX_GAP_BYTES = _REL_MAX_DELTA_WORDS * 4   # 0x3FFFC bytes


def encode_rel_pointer_table(ptr_byte_offsets: Sequence[int]) -> bytes:
    """Encode absolute pointer byte-offsets into the u16 word-delta table.

    Each entry is ``(offset - previous_offset) / 4`` as a little-endian
    u16; the first delta is measured from base 0.  Offsets MUST be
    4-byte aligned and strictly ascending (duplicate locations are
    nonsensical — each pointer field is relocated once).

    Empty input returns ``b""`` (a c.rel always has >=1 pointer, but the
    primitive stays general).

    Raises
    ------
    RelWriteError
        If an offset is unaligned, not ascending, negative, or a gap to
        the previous offset exceeds ``0xFFFF`` words (0x3FFFC bytes).  No
        real PSOBB file has such a gap; the engine's table cannot encode
        it without a filler scheme, so we refuse rather than corrupt.
    """
    out = bytearray()
    prev = 0
    for off in ptr_byte_offsets:
        if off < 0:
            raise RelWriteError(f"pointer offset {off} is negative")
        if off & 3:
            raise RelWriteError(
                f"pointer offset 0x{off:x} is not 4-byte aligned")
        gap = off - prev
        if gap < 0:
            raise RelWriteError(
                f"pointer offset 0x{off:x} not ascending (prev 0x{prev:x})")
        delta_words = gap >> 2
        if delta_words > _REL_MAX_DELTA_WORDS:
            raise RelWriteError(
                f"pointer gap 0x{gap:x} bytes between 0x{prev:x} and "
                f"0x{off:x} exceeds the u16 delta ceiling "
                f"(0x{_REL_MAX_GAP_BYTES:x})")
        out += struct.pack("<H", delta_words)
        prev = off
    return bytes(out)


def decode_rel_pointer_table(buf: bytes, pt_off: int = 0,
                             pt_count: Optional[int] = None,
                             base: int = 0) -> List[int]:
    """Decode a u16 word-delta table into absolute pointer byte-offsets.

    Mirror of :func:`formats.rel._resolve_pointer_offsets`, re-exported
    here so the writer's tests can round-trip without reaching into a
    private function.

    Parameters
    ----------
    buf:
        Bytes holding the table (the table may be a sub-slice of a full
        file; ``pt_off`` selects where it starts).
    pt_off:
        Byte offset of the table within ``buf`` (default 0 — treat
        ``buf`` as the table itself).
    pt_count:
        Number of entries.  Defaults to ``(len(buf) - pt_off) // 2``.
    base:
        Added to every resolved offset (the engine adds the load
        address; tests use 0 to get file-relative locations).
    """
    if pt_count is None:
        pt_count = (len(buf) - pt_off) // 2
    out: List[int] = []
    prev = base
    for i in range(pt_count):
        slot = pt_off + i * 2
        if slot + 2 > len(buf):
            raise RelWriteError(f"pointer table walk OOB at entry {i}")
        delta = struct.unpack_from("<H", buf, slot)[0]
        cur = prev + delta * 4
        out.append(cur)
        prev = cur
    return out


# ---------------------------------------------------------------------------
# STEP 1 — container framing (data + table + trailer)
# ---------------------------------------------------------------------------
#
# Generic across c.rel / n.rel / r.rel: every REL is "data section,
# NUL-pad to 32, pointer table, NUL-pad so the trailer starts 32-aligned,
# 32-byte trailer".  The trailer's three offsets and the file size are
# all guaranteed % 32 == 0.

_ALIGN = 0x20  # 32-byte alignment for pt_off and trailer_start
TRAILER_FORMAT_FLAG = 1


def _pad_to(buf: bytearray, alignment: int) -> None:
    """Append NUL bytes so ``len(buf)`` is a multiple of ``alignment``."""
    rem = len(buf) % alignment
    if rem:
        buf += b"\x00" * (alignment - rem)


def assemble_rel(data: bytes,
                 ptr_offsets: Sequence[int],
                 payload_offset: int) -> bytes:
    """Frame a finished data section into a complete REL container.

    Performs: NUL-pad ``data`` to 32 bytes (-> ``pt_off``); append the
    u16 word-delta table; NUL-pad so the trailer starts 32-aligned;
    append the 32-byte trailer.

    The caller owns the data-section *content* and the set of pointer
    field *locations*; this routine only frames them.  ``ptr_offsets``
    must be sorted ascending and lie within the (padded) data section.

    Returns a ``bytes`` whose length, ``pt_off``, and ``trailer_start``
    are all ``% 32 == 0``.

    Raises
    ------
    RelWriteError
        On unsorted/unaligned/out-of-range pointer offsets, or an
        out-of-range payload offset.
    """
    # Pointer offsets and the payload must reference a u32 inside the
    # MEANINGFUL data the caller emitted — not the 32-byte alignment pad
    # that follows it (a pointer field landing in pad would be a writer
    # bug).  Validate against the unpadded length first.
    data_len = len(data)
    for off in ptr_offsets:
        if off < 0 or off + 4 > data_len:
            raise RelWriteError(
                f"pointer offset 0x{off:x} out of data section "
                f"(data_len=0x{data_len:x})")
    if payload_offset < 0 or payload_offset + 4 > data_len:
        raise RelWriteError(
            f"payload_offset 0x{payload_offset:x} out of data section "
            f"(data_len=0x{data_len:x})")

    buf = bytearray(data)
    # The data section is padded to a 32-byte boundary; that length is
    # where the pointer table begins.
    _pad_to(buf, _ALIGN)
    pt_off = len(buf)

    table = encode_rel_pointer_table(ptr_offsets)
    pt_count = len(ptr_offsets)
    buf += table

    # NUL-pad after the table so the 32-byte trailer starts at a
    # 32-aligned offset (which also makes the final file size a multiple
    # of 32, since the trailer is exactly 32 bytes).
    _pad_to(buf, _ALIGN)

    buf += struct.pack(
        "<5I",
        pt_off,
        pt_count,
        TRAILER_FORMAT_FLAG,
        0,
        payload_offset,
    )
    buf += b"\x00" * 12   # trailer +0x14..+0x1F reserved zeros

    assert len(buf) % _ALIGN == 0, "framed REL not 32-aligned (bug)"
    return bytes(buf)


# ---------------------------------------------------------------------------
# STEP 2 — c.rel (collision) model + byte-exact writer
# ---------------------------------------------------------------------------

CREL_MESH_SIZE = 16
CREL_FACE_SIZE = 36
CREL_NODE_SIZE = 24
_CREL_NODE_TERMINATOR = b"\x00" * CREL_NODE_SIZE

_CREL_MESH_FMT = "<IIiI"          # vcount, verts_ptr, fcount, faces_ptr
_CREL_FACE_FMT = "<4H7f"          # i0,i1,i2,flags, nx,ny,nz, cx,cy,cz, radius
_CREL_NODE_FMT = "<IffffI"        # mesh_ptr, x,y,z, radius, flags
assert struct.calcsize(_CREL_MESH_FMT) == CREL_MESH_SIZE
assert struct.calcsize(_CREL_FACE_FMT) == CREL_FACE_SIZE
assert struct.calcsize(_CREL_NODE_FMT) == CREL_NODE_SIZE

# Engine cap: map_collision_data.data[512]; the array is NUL-terminated.
_CREL_MAX_NODES = 512
# 64 KB budget for a custom (non-vanilla) collision hull.  Vanilla
# city00 ships 70144 bytes (>64KB) and is a valid read/round-trip
# fixture, so the cap is only enforced for freshly-built hulls.
CREL_SIZE_BUDGET = 0x10000

# Safe defaults for an authored walkable floor (see spec unknowns #2).
DEFAULT_FACE_FLAGS = 0x0001
DEFAULT_NODE_FLAGS = 0x80000101


@dataclass
class CrelFace:
    """One collision triangle.

    ``normal`` / ``centroid`` / ``radius`` are auto-derived from the
    vertex positions when left ``None`` (see :func:`build_crel`).  When a
    model is parsed from a vanilla file the stored values are captured
    verbatim so the round-trip is byte-exact (the engine's stored
    centroid/normal are not guaranteed to be the *exact* float result of
    re-deriving from verts).
    """
    i0: int
    i1: int
    i2: int
    flags: int = DEFAULT_FACE_FLAGS
    normal: Optional[Tuple[float, float, float]] = None
    centroid: Optional[Tuple[float, float, float]] = None
    radius: Optional[float] = None


@dataclass
class CrelNode:
    """One broad-phase collision node: a bounding sphere + a mesh.

    ``center`` / ``radius`` are auto-derived to conservatively enclose
    all of the node's vertices when left ``None``.
    """
    verts: List[Tuple[float, float, float]] = field(default_factory=list)
    faces: List[CrelFace] = field(default_factory=list)
    center: Optional[Tuple[float, float, float]] = None
    radius: Optional[float] = None
    flags: int = DEFAULT_NODE_FLAGS


@dataclass
class CrelModel:
    """Round-trip-preserving c.rel model.

    ``payload_pad_bytes`` records the number of trailing zero bytes the
    source file placed after the 4-byte payload head (vanilla files pad
    this region to between 4 and 32 bytes; the value is non-semantic but
    must be reproduced for a byte-exact round-trip).  Fresh hulls leave
    it 0 (minimal 4-byte payload).
    """
    nodes: List[CrelNode] = field(default_factory=list)
    payload_pad_bytes: int = 0


# ---- geometry derivation -------------------------------------------------

def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _length(a):
    return (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) ** 0.5


def _face_centroid(v0, v1, v2):
    return ((v0[0] + v1[0] + v2[0]) / 3.0,
            (v0[1] + v1[1] + v2[1]) / 3.0,
            (v0[2] + v1[2] + v2[2]) / 3.0)


def _face_normal(v0, v1, v2):
    n = _cross(_sub(v1, v0), _sub(v2, v0))
    ln = _length(n)
    if ln <= 1e-12:
        return None  # degenerate
    return (n[0] / ln, n[1] / ln, n[2] / ln)


def _face_circumradius(c, v0, v1, v2):
    return max(_length(_sub(c, v0)),
               _length(_sub(c, v1)),
               _length(_sub(c, v2)))


def _derive_node_sphere(verts):
    """AABB-midpoint center + enclosing radius for a vertex set."""
    if not verts:
        return (0.0, 0.0, 0.0), 0.0
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    center = ((min(xs) + max(xs)) * 0.5,
              (min(ys) + max(ys)) * 0.5,
              (min(zs) + max(zs)) * 0.5)
    radius = max(_length(_sub(center, v)) for v in verts)
    return center, radius


# ---- parse (byte-exact capture) ------------------------------------------

def parse_crel_for_writer(src: bytes) -> CrelModel:
    """Parse a c.rel byte buffer into a round-trip-preserving model.

    Captures every vertex/face value plus each node's sphere/flags and
    the payload padding span, exactly as stored, so that
    :func:`encode_crel` reproduces the source byte-for-byte.

    Raises
    ------
    RelWriteError
        If the buffer is not a structurally valid c.rel (bad trailer,
        OOB pointers, face index out of range, mis-sized regions).
    """
    if not isinstance(src, (bytes, bytearray, memoryview)):
        raise RelWriteError("input must be bytes-like")
    src = bytes(src)
    if len(src) < TRAILER_SIZE:
        raise RelWriteError("buffer too small for REL trailer")

    trailer_start = len(src) - TRAILER_SIZE
    pt_off, pt_count, flag, reserved, pl_off = struct.unpack_from(
        "<5I", src, trailer_start)
    if pt_off > trailer_start or pl_off > trailer_start:
        raise RelWriteError("trailer offsets exceed data section")
    if pl_off + 4 > trailer_start:
        raise RelWriteError("payload head out of range")

    head = struct.unpack_from("<I", src, pl_off)[0]
    if head + CREL_NODE_SIZE > trailer_start:
        raise RelWriteError(f"node array head 0x{head:x} out of range")

    nodes: List[CrelNode] = []
    node_off = head
    while True:
        if node_off + CREL_NODE_SIZE > trailer_start:
            raise RelWriteError("node array not NUL-terminated before table")
        mesh_ptr, cx, cy, cz, nrad, nflags = struct.unpack_from(
            _CREL_NODE_FMT, src, node_off)
        if mesh_ptr == 0:
            break  # terminator
        if mesh_ptr + CREL_MESH_SIZE > trailer_start:
            raise RelWriteError(f"mesh_ptr 0x{mesh_ptr:x} out of range")
        vcount, verts_ptr, fcount, faces_ptr = struct.unpack_from(
            _CREL_MESH_FMT, src, mesh_ptr)
        if fcount < 0:
            raise RelWriteError("negative face count")
        if verts_ptr + vcount * 12 > trailer_start:
            raise RelWriteError("vertex block out of range")
        if faces_ptr + fcount * CREL_FACE_SIZE > trailer_start:
            raise RelWriteError("face block out of range")

        verts = [
            struct.unpack_from("<3f", src, verts_ptr + i * 12)
            for i in range(vcount)
        ]
        faces: List[CrelFace] = []
        for i in range(fcount):
            off = faces_ptr + i * CREL_FACE_SIZE
            (i0, i1, i2, ff, nx, ny, nz, fcx, fcy, fcz, frad) = \
                struct.unpack_from(_CREL_FACE_FMT, src, off)
            if i0 >= vcount or i1 >= vcount or i2 >= vcount:
                raise RelWriteError(
                    f"face {i} index out of range (vcount={vcount})")
            faces.append(CrelFace(
                i0=i0, i1=i1, i2=i2, flags=ff,
                normal=(nx, ny, nz),
                centroid=(fcx, fcy, fcz),
                radius=frad,
            ))
        nodes.append(CrelNode(
            verts=verts, faces=faces,
            center=(cx, cy, cz), radius=nrad, flags=nflags,
        ))
        node_off += CREL_NODE_SIZE

    # Payload region span: from the payload head to the pointer table.
    payload_pad = pt_off - pl_off - 4
    if payload_pad < 0:
        raise RelWriteError("payload region shorter than 4 bytes")

    model = CrelModel(nodes=nodes, payload_pad_bytes=payload_pad)

    # Cross-check the relocation graph the source declared matches the
    # one our deterministic emit will produce.  A mismatch means the
    # source used a layout this writer does not model (and would silently
    # corrupt) — refuse rather than emit a wrong table.
    src_ptr_locs = decode_rel_pointer_table(src, pt_off, pt_count)
    _, expect_locs, _ = _build_crel_data(model)
    if sorted(src_ptr_locs) != sorted(expect_locs):
        raise RelWriteError(
            "source pointer graph does not match the c.rel layout model "
            f"(src {len(src_ptr_locs)} locs, model {len(expect_locs)} locs)")
    return model


# ---- encode --------------------------------------------------------------

def _build_crel_data(model: CrelModel) -> Tuple[bytes, List[int], int]:
    """Emit the c.rel data section (pre-framing).

    Returns ``(data_bytes, ptr_locations, payload_offset)``.  Faces with
    ``normal``/``centroid``/``radius`` left ``None`` are derived here;
    node sphere is derived when ``center``/``radius`` is ``None``.
    """
    if len(model.nodes) > _CREL_MAX_NODES:
        raise RelWriteError(
            f"{len(model.nodes)} nodes exceeds engine cap {_CREL_MAX_NODES}")

    buf = bytearray()
    ptr_locs: List[int] = []
    mesh_offsets: List[int] = []

    for nd in model.nodes:
        verts_off = len(buf)
        for v in nd.verts:
            buf += struct.pack("<3f", float(v[0]), float(v[1]), float(v[2]))

        faces_off = len(buf)
        for fc in nd.faces:
            v0, v1, v2 = nd.verts[fc.i0], nd.verts[fc.i1], nd.verts[fc.i2]
            cen = fc.centroid if fc.centroid is not None \
                else _face_centroid(v0, v1, v2)
            nrm = fc.normal
            if nrm is None:
                nrm = _face_normal(v0, v1, v2)
                if nrm is None:
                    # Degenerate triangle with no caller-supplied normal:
                    # the engine would skip it; we refuse so a caller that
                    # wanted it knows it was dropped/invalid.
                    raise RelWriteError(
                        "degenerate face (zero-area) needs an explicit normal")
            rad = fc.radius if fc.radius is not None \
                else _face_circumradius(cen, v0, v1, v2)
            buf += struct.pack(
                _CREL_FACE_FMT,
                fc.i0 & 0xFFFF, fc.i1 & 0xFFFF, fc.i2 & 0xFFFF,
                fc.flags & 0xFFFF,
                nrm[0], nrm[1], nrm[2],
                cen[0], cen[1], cen[2],
                float(rad),
            )

        mesh_off = len(buf)
        mesh_offsets.append(mesh_off)
        buf += struct.pack(_CREL_MESH_FMT,
                           len(nd.verts), verts_off,
                           len(nd.faces), faces_off)
        ptr_locs.append(mesh_off + 4)    # verts_ptr
        ptr_locs.append(mesh_off + 12)   # faces_ptr

    node_arr_off = len(buf)
    for nd, mesh_off in zip(model.nodes, mesh_offsets):
        node_off = len(buf)
        if nd.center is None or nd.radius is None:
            d_center, d_radius = _derive_node_sphere(nd.verts)
            cx, cy, cz = nd.center if nd.center is not None else d_center
            radius = nd.radius if nd.radius is not None else d_radius
        else:
            cx, cy, cz = nd.center
            radius = nd.radius
        buf += struct.pack(_CREL_NODE_FMT,
                           mesh_off, float(cx), float(cy), float(cz),
                           float(radius), nd.flags & 0xFFFFFFFF)
        ptr_locs.append(node_off)        # mesh_ptr

    buf += _CREL_NODE_TERMINATOR

    payload_off = len(buf)
    buf += struct.pack("<I", node_arr_off)
    ptr_locs.append(payload_off)         # payload head -> node array
    buf += b"\x00" * model.payload_pad_bytes

    return bytes(buf), ptr_locs, payload_off


def encode_crel(model: CrelModel) -> bytes:
    """Serialise a :class:`CrelModel` into a complete c.rel byte buffer.

    Byte-exact inverse of :func:`parse_crel_for_writer` for any vanilla
    c.rel.  Pointer locations are flagged by construction (one per
    verts_ptr/faces_ptr/mesh_ptr field + the payload head), so the
    relocation table is exactly ``3 * node_count + 1`` entries — even
    when a flagged field's stored value is 0.
    """
    data, ptr_locs, payload_off = _build_crel_data(model)
    ptr_locs.sort()
    return assemble_rel(data, ptr_locs, payload_off)


def build_crel(nodes: Sequence[CrelNode],
               enforce_budget: bool = True) -> bytes:
    """Convenience: build a c.rel from a list of nodes (fresh authoring).

    Auto-derives face normals/centroids/radii and node bounding spheres
    for any field left ``None``.  Emits a minimal 4-byte payload head
    (no trailing pad).

    Parameters
    ----------
    enforce_budget:
        When True (default) raises if the result exceeds the 64 KB c.rel
        budget — a freshly authored collision hull must be a simplified
        mesh that fits.  Set False to round-trip an over-budget vanilla
        file (e.g. city00 at 70144 bytes).
    """
    model = CrelModel(nodes=list(nodes), payload_pad_bytes=0)
    out = encode_crel(model)
    if enforce_budget and len(out) > CREL_SIZE_BUDGET:
        raise RelWriteError(
            f"c.rel is {len(out)} bytes, exceeds the 0x{CREL_SIZE_BUDGET:x} "
            f"(64 KB) budget — simplify the collision hull")
    return out


# ---------------------------------------------------------------------------
# Loader-fixup simulation (engine acceptance check; used by tests)
# ---------------------------------------------------------------------------

def simulate_rel_relocation(buf: bytes, base: int = 0) -> List[int]:
    """Replicate the engine's relocation walk (``load_rel_asset2``).

    Walks the u16 delta table, and for each flagged word adds ``base``.
    Returns the list of *relocated values* (i.e. ``stored_value + base``)
    in pointer-location order.  A test can assert each lands inside
    ``[base, base + pt_off)`` (the legal in-data range) or equals
    ``base`` (the legal-null case).
    """
    trailer_start = len(buf) - TRAILER_SIZE
    pt_off, pt_count, _flag, _res, _pl = struct.unpack_from(
        "<5I", buf, trailer_start)
    locs = decode_rel_pointer_table(buf, pt_off, pt_count)
    out: List[int] = []
    for loc in locs:
        if loc + 4 > pt_off:
            raise RelWriteError(
                f"relocation target 0x{loc:x} outside data section")
        stored = struct.unpack_from("<I", buf, loc)[0]
        out.append(stored + base)
    return out


# ---------------------------------------------------------------------------
# STEP 3 — n.rel (node geometry / NrelFmt2) model + byte-exact writer
# ---------------------------------------------------------------------------
#
# n.rel geometry is **XJ (D3D buffer descriptors)**, NOT NJCM: there is
# no IFF wrapper and no POF0 chunk.  The container framing is identical
# to c.rel (data + u16 word-delta table + 32-byte trailer, all %32), so
# the n.rel writer reuses STEP-0/STEP-1 unchanged.
#
# This is a ROUND-TRIP-PRESERVING (layout-hint) writer, exactly like
# ``nj_writer.parse_nj_for_writer``: it captures every understood region
# with its *source byte offset* plus the opaque bulk-geometry span, and
# rebuilds the relocation table from the captured pointer locations.  It
# does NOT re-derive XJ geometry from world-space triangles.  v1 must
# byte-exact round-trip UNMODIFIED files first; pose/pointer-graph edits
# layer in on top of the same layout.
#
# Data-section region order (lobby_01, all 32-byte aligned overall):
#   0x000  TAM / texture-animation info block (opaque; no pointers into it)
#   0xa8   static_mesh_trees[]  (16B each)   <- first reloc ptr
#   0x478  animated_mesh_trees[] (16B each; interleaved ptr+float, opaque)
#   0x4f8  NrelChunk[]          (52B each)
#   0x52c  NrelFmt2 header      (24B)        <- payload_offset
#   0x560  TextureList + entries (8B + 12B*count)
#   …bulk… XjMesh / vertex buffers / strip index lists / RenderStateArgs
#          (interleaved, NOT contiguous-by-type — kept as one opaque span)
#   …tail  texture name strings; NUL pad to 32 -> pt_off.
#
# The structured regions (TAM, tree arrays, chunks, fmt2 header, texture
# list, texture names) are parsed for *editability and validation*; the
# byte-exact v1 emit re-lays every region from the captured source bytes
# at its captured offset, fills any inter-region gap verbatim from the
# source, and rebuilds the table+trailer from the captured pointer
# locations.  Because every byte of the data section is sourced from the
# original (regions + gaps), the round-trip is exact while the model
# still exposes the structure the editor needs.

NREL_FMT2_HEADER_SIZE = 24
NREL_CHUNK_SIZE = 52
NREL_MESH_TREE_SIZE = 16
NREL_TEXLIST_ENTRY_SIZE = 12

_NREL_FMT2_HEADER_FMT = "<4sIHHfII"   # magic,unk1,chunk_count,unk2,radius,chunks,tex
_NREL_CHUNK_FMT = "<i3f3ifIIIII"      # id,pos3f,rot3i,radius,smt,amt,sc,ac,flags
_NREL_MESH_TREE_FMT = "<IIII"         # root_node_ptr,unk1,tex_anim_ptr,tree_flags
assert struct.calcsize(_NREL_FMT2_HEADER_FMT) == NREL_FMT2_HEADER_SIZE
assert struct.calcsize(_NREL_CHUNK_FMT) == NREL_CHUNK_SIZE
assert struct.calcsize(_NREL_MESH_TREE_FMT) == NREL_MESH_TREE_SIZE


@dataclass
class NrelMeshTreeRec:
    """One 16-byte ``MeshTree`` record, captured with its source offset."""
    src_offset: int
    root_node_ptr: int
    unk1: int
    tex_anim_info_ptr: int
    tree_flags: int


@dataclass
class NrelChunkRec:
    """One 52-byte ``NrelChunk`` record (visibility chunk).

    ``static_trees`` / ``animated_trees`` hold the parsed MeshTree
    records the chunk's tree-array pointers reference (multiple chunks
    may share the same tree array — e.g. ``map_aboss01n`` has two chunks
    both pointing at the tree array at 0x30).  They are decoded for
    inspection/editing; byte-exact emit reads the raw tree bytes back
    from the source span, so shared arrays are reproduced once.
    """
    src_offset: int
    id: int
    pos: Tuple[float, float, float]
    rot: Tuple[int, int, int]
    radius: float
    static_mesh_trees_ptr: int
    animated_mesh_trees_ptr: int
    static_count: int
    animated_count: int
    flags: int
    static_trees: List[NrelMeshTreeRec] = field(default_factory=list)
    animated_trees: List[NrelMeshTreeRec] = field(default_factory=list)


@dataclass
class NrelLayoutHint:
    """Source-byte layout of an n.rel data section, for byte-exact emit.

    ``raw_data`` is the entire relocatable data section verbatim
    (``src[:pointer_table_offset]``); ``pointer_offsets`` is the ordered
    relocation-graph the source declared.  Re-emitting ``raw_data`` and
    rebuilding the table from ``pointer_offsets`` reproduces the file
    byte-for-byte — this is the load-bearing guarantee.  The structured
    fields above the hint expose the same data for editing.
    """
    raw_data: bytes
    pointer_offsets: List[int]
    payload_offset: int
    data_size: int


@dataclass
class NrelModel:
    """Round-trip-preserving n.rel (NrelFmt2) model.

    The scalar header fields (``unk1``, ``unk2``, ``radius``) and the
    decoded chunk / mesh-tree / texture-name structure are exposed for
    inspection and future editing.  ``layout_hint`` carries the verbatim
    source data section + relocation graph that :func:`encode_nrel` uses
    to reproduce the file byte-for-byte.
    """
    unk1: int
    unk2: int
    radius: float
    chunk_count: int
    chunks: List[NrelChunkRec]
    texture_names: List[str]
    layout_hint: NrelLayoutHint


# ---- parse (byte-exact capture) ------------------------------------------

def parse_nrel_for_writer(src: bytes) -> NrelModel:
    """Parse an n.rel byte buffer into a round-trip-preserving model.

    Captures the verbatim data section + the full ordered relocation
    graph (so :func:`encode_nrel` is byte-exact), plus the decoded
    ``NrelFmt2`` header, chunks, mesh-trees, and texture names for
    inspection.  This is the exact inverse of the n.rel reader in
    :mod:`formats.rel`.

    Raises
    ------
    RelWriteError
        If the buffer is not a structurally valid n.rel (bad trailer,
        OOB pointers, payload magic != ``"fmt2"``).
    """
    if not isinstance(src, (bytes, bytearray, memoryview)):
        raise RelWriteError("input must be bytes-like")
    src = bytes(src)
    if len(src) < TRAILER_SIZE:
        raise RelWriteError("buffer too small for REL trailer")

    try:
        rel = parse_rel(src)
    except RelParseError as e:
        raise RelWriteError(f"not a valid REL container: {e}") from e
    if not is_n_rel(rel):
        raise RelWriteError("payload magic != 'fmt2' (not an n.rel)")

    pl_off = rel.payload_offset
    if pl_off + NREL_FMT2_HEADER_SIZE > rel.data_size:
        raise RelWriteError("n.rel payload header truncated")
    magic, unk1, chunk_count, unk2, radius, chunks_ptr, _tex_ptr = \
        struct.unpack_from(_NREL_FMT2_HEADER_FMT, src, pl_off)
    if magic != NREL_FMT2_MAGIC:
        raise RelWriteError(f"unexpected fmt2 magic {magic!r}")

    # Decode the chunk / mesh-tree structure (for inspection + edit). The
    # readers in formats.rel are tolerant; mirror their walk so a model
    # exposes the same view, but treat any structural defect as a hard
    # error here (the writer must understand what it round-trips).
    try:
        header = read_nrel_header(rel)
        raw_chunks = read_nrel_chunks(rel, header)
    except RelParseError as e:
        raise RelWriteError(f"n.rel chunk walk failed: {e}") from e
    if len(raw_chunks) != chunk_count:
        raise RelWriteError(
            f"chunk_count {chunk_count} but walked {len(raw_chunks)} chunks")

    chunks: List[NrelChunkRec] = []
    for i, c in enumerate(raw_chunks):
        chunk_off = header.chunks_ptr + i * NREL_CHUNK_SIZE

        def _trees(base_ptr: int, count: int) -> List[NrelMeshTreeRec]:
            recs: List[NrelMeshTreeRec] = []
            for t in read_mesh_trees(rel, base_ptr, count):
                # read_mesh_trees returns records in order; recompute each
                # source offset so an editor can locate them.
                recs.append(NrelMeshTreeRec(
                    src_offset=base_ptr + len(recs) * NREL_MESH_TREE_SIZE,
                    root_node_ptr=t.root_node_ptr,
                    unk1=t.unk1,
                    tex_anim_info_ptr=t.texture_animation_info_ptr,
                    tree_flags=t.tree_flags,
                ))
            return recs

        chunks.append(NrelChunkRec(
            src_offset=chunk_off,
            id=c.id,
            pos=(c.x, c.y, c.z),
            rot=(c.rot_x, c.rot_y, c.rot_z),
            radius=c.radius,
            static_mesh_trees_ptr=c.static_mesh_trees_ptr,
            animated_mesh_trees_ptr=c.animated_mesh_trees_ptr,
            static_count=c.static_mesh_tree_count,
            animated_count=c.animated_mesh_tree_count,
            flags=c.flags,
            static_trees=_trees(c.static_mesh_trees_ptr,
                                c.static_mesh_tree_count),
            animated_trees=_trees(c.animated_mesh_trees_ptr,
                                  c.animated_mesh_tree_count),
        ))

    texture_names = read_texture_names(rel)

    hint = NrelLayoutHint(
        raw_data=src[:rel.pointer_table_offset],
        pointer_offsets=list(rel.pointer_offsets),
        payload_offset=pl_off,
        data_size=rel.data_size,
    )
    return NrelModel(
        unk1=unk1,
        unk2=unk2,
        radius=radius,
        chunk_count=chunk_count,
        chunks=chunks,
        texture_names=texture_names,
        layout_hint=hint,
    )


# ---- encode --------------------------------------------------------------

def encode_nrel(model: NrelModel) -> bytes:
    """Serialise an :class:`NrelModel` into a complete n.rel byte buffer.

    Byte-exact inverse of :func:`parse_nrel_for_writer` for any vanilla
    n.rel.  Re-lays the captured data section, then rebuilds the u16
    word-delta table + 32-byte trailer via the shared
    :func:`assemble_rel` framing from the captured pointer graph.

    The pointer-graph SET emitted is exactly the one the source
    declared (no more, no less) — a missing entry leaves a stale
    file-relative offset and an extra entry corrupts a non-pointer word,
    so the writer carries the source's relocation list verbatim rather
    than re-deriving it from the wired graph.
    """
    hint = model.layout_hint
    if hint is None:
        raise RelWriteError(
            "encode_nrel requires a layout_hint (synthetic n.rel geometry "
            "authoring is a later phase)")

    data = hint.raw_data
    if len(data) != hint.data_size:
        raise RelWriteError(
            f"layout_hint data_size 0x{hint.data_size:x} != raw_data length "
            f"0x{len(data):x}")

    # The payload header still has to start with 'fmt2' and carry the
    # model's scalar fields — re-pack it into the data buffer so edits to
    # unk1/unk2/radius land while the rest of the region stays verbatim.
    # (In pure byte-exact mode this rewrites the header to the identical
    # bytes; it is what makes scalar edits possible without re-deriving
    # the geometry.)
    pl_off = hint.payload_offset
    if pl_off + NREL_FMT2_HEADER_SIZE > len(data):
        raise RelWriteError("payload header out of captured data section")
    buf = bytearray(data)
    magic, _u1, _cc, _u2, _rad, chunks_ptr, tex_ptr = struct.unpack_from(
        _NREL_FMT2_HEADER_FMT, buf, pl_off)
    struct.pack_into(
        _NREL_FMT2_HEADER_FMT, buf, pl_off,
        NREL_FMT2_MAGIC,
        model.unk1 & 0xFFFFFFFF,
        model.chunk_count & 0xFFFF,
        model.unk2 & 0xFFFF,
        float(model.radius),
        chunks_ptr,
        tex_ptr,
    )

    ptr_offsets = sorted(hint.pointer_offsets)
    return assemble_rel(bytes(buf), ptr_offsets, pl_off)


# ---------------------------------------------------------------------------
# STEP 3b — n.rel FROM-SCRATCH authoring (build_nrel_from_meshes)
# ---------------------------------------------------------------------------
#
# Builds a fresh, minimal valid n.rel from XJ submeshes + a mesh-tree +
# texture names.  Unlike ``encode_nrel`` (the layout-hint round-trip
# path), this re-derives the entire data section and the relocation
# graph from a high-level model — it is the path the editor's
# import/author pipeline uses.
#
# The geometry structs are the rel.py D3D ``XjMesh`` family (28B XjMesh /
# 16B VertexBufferContainer / 20B IndexBufferContainer / 16B
# RenderStateArgs) — NOT the 44B descriptor ``XjModel`` (that one lives
# in formats/xj_writer.py).  Output is validated by
# ``formats.rel.extract_nrel_meshes`` + ``simulate_rel_relocation``.
#
# Data-section region order (small fixed structs first, bulk geometry
# last so it can grow toward the 768 KB budget):
#   R1  static_mesh_trees[]      16B each
#   R3  NrelChunk[]              52B each
#   R4  NrelFmt2 header          24B   (= payload_offset)
#   R5  TextureList              8B    (omitted when no names)
#   R6  TextureListEntry[]       12B each
#   R7  MeshTreeNode[]           52B each (Ninja node hierarchy)
#   R8  XjMesh[]                 28B each
#   R8a VertexBufferContainer[]  16B each
#   R8b IndexBufferContainer[]   20B each
#   R8c RenderStateArgs[]        16B each
#   R9v raw vertex arrays        (no pointers)
#   R9i raw index arrays         (no pointers)
#   R10 texture name strings     (NUL-terminated, 4-aligned; no pointers)
# (TAM / animated trees are OMITTED for fresh authoring.)

# 768 KB engine budget for an authored map n.rel.
NREL_SIZE_BUDGET = 0xC0000

NREL_XJ_MESH_SIZE = 28
NREL_VBUF_CONTAINER_SIZE = 16
NREL_IBUF_CONTAINER_SIZE = 20
NREL_RSARGS_SIZE = 16
NREL_TEXLIST_SIZE = 8

_NREL_XJ_MESH_FMT = "<7I"
_NREL_VBUF_FMT = "<4I"            # vertex_format, vertices_ptr, vertex_size, vertex_count
_NREL_IBUF_FMT = "<5I"           # rs_ptr, rs_count, indices_ptr, index_count, vb_index
_NREL_RSARGS_FMT = "<4I"         # state_type, arg1, arg2, unk
_NREL_TEXLIST_FMT = "<II"        # elements_ptr, count
assert struct.calcsize(_NREL_XJ_MESH_FMT) == NREL_XJ_MESH_SIZE
assert struct.calcsize(_NREL_VBUF_FMT) == NREL_VBUF_CONTAINER_SIZE
assert struct.calcsize(_NREL_IBUF_FMT) == NREL_IBUF_CONTAINER_SIZE
assert struct.calcsize(_NREL_RSARGS_FMT) == NREL_RSARGS_SIZE

# Authored eval_flags: identity-local single leaf (UNIT_POS|ANG|SCL|BREAK).
# extract_nrel_meshes treats chunk pose as a pure translation and does
# NOT apply node rotation/scale, so v1 nodes are translation-only.
NREL_NODE_EVAL_FLAGS = 0x17

# vertex_format 3 = pos f32x3 @0, normal f32x3 @12, uv f32x2 @24 (stride
# 32) — fully round-trips via rel._VERTEX_LAYOUTS.
NREL_VERTEX_FORMAT = 3
NREL_VERTEX_STRIDE = 32

# Default chunk render-flags (semantics undecoded; a benign vanilla
# value).  Copied from a vanilla lobby_01 chunk.
DEFAULT_NREL_CHUNK_FLAGS = 0x00000000
DEFAULT_NREL_FMT2_UNK1 = 64


@dataclass
class NrelVertex:
    """One authored vertex: position + normal + UV (vertex_format 3)."""
    pos: Tuple[float, float, float]
    normal: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    uv: Tuple[float, float] = (0.0, 0.0)


@dataclass
class NrelSubmesh:
    """One authored submesh: a vertex array + a triangle-strip index list.

    ``indices`` are u16 triangle-STRIP indices into ``vertices`` (the
    engine de-stripifies via ``_strip_to_triangles``).  A trivial
    one-triangle-per-strip authoring keeps each submesh a single strip;
    callers wanting many triangles should hand a pre-built strip OR use
    :func:`nrel_submeshes_from_triangle_list`.  ``texture_id`` indexes
    the texture-name list and is written as a type-3 RenderStateArgs.
    """
    vertices: List[NrelVertex] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)
    texture_id: int = 0


@dataclass
class NrelNode:
    """One authored mesh-tree node: a TRS transform + its submeshes.

    ``submeshes`` become the node's XjMesh (one VertexBufferContainer +
    one IndexBufferContainer per submesh).  ``child_index`` /
    ``sibling_index`` index into the flat node list (-1 = none).  v1
    nodes are translation-only (see ``NREL_NODE_EVAL_FLAGS``).
    """
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_bams: Tuple[int, int, int] = (0, 0, 0)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    eval_flags: int = NREL_NODE_EVAL_FLAGS
    submeshes: List[NrelSubmesh] = field(default_factory=list)
    child_index: int = -1
    sibling_index: int = -1


def _nrel_enclosing_sphere(
    points: Sequence[Tuple[float, float, float]],
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> float:
    """Max distance from ``origin`` to any point (a chunk-enclosing radius)."""
    r = 0.0
    ox, oy, oz = origin
    for (x, y, z) in points:
        d = ((x - ox) ** 2 + (y - oy) ** 2 + (z - oz) ** 2) ** 0.5
        if d > r:
            r = d
    return r


def _build_nrel_data(
    nodes: Sequence[NrelNode],
    texture_names: Sequence[str],
    chunk_pos: Tuple[float, float, float],
    chunk_flags: int,
    fmt2_unk1: int,
) -> Tuple[bytes, List[int], int]:
    """Emit the n.rel data section (pre-framing) for a single chunk.

    Returns ``(data_bytes, ptr_locations, payload_offset)``.  ``nodes``
    are wired into ONE static mesh-tree under ONE chunk; the geometry of
    every node that has submeshes becomes an XjMesh.  Every submesh is
    emitted as a SEPARATE IndexBufferContainer (one draw-strip), so a
    triangle-list submesh splits into one strip-row per triangle to
    dodge the ``_strip_to_triangles`` parity flip.

    The pointer-location set is enumerated by recording the byte offset
    of each in-data pointer field as it is written (see the per-region
    FLAG comments).  The null child/next policy matches what
    ``parse_rel`` resolves on a vanilla file: a null child/next/anim/
    alpha/tex_anim field is NOT flagged.
    """
    n_nodes = len(nodes)
    if n_nodes == 0:
        raise RelWriteError("build_nrel_from_meshes: no nodes")

    # ---- PASS 1: assign region offsets ----
    # Reserve a small NUL header so NO meaningful pointer target lands at
    # data offset 0.  The reader helpers (read_mesh_trees,
    # read_nrel_chunks, _resolve_texture_id, ...) treat a stored pointer
    # value of 0 as "null/absent", so a real structure at offset 0 would
    # be unreachable.  Vanilla files place an (omitted-here) TAM block at
    # the front for the same effect; 16 NUL bytes suffice and nothing
    # points into them.
    _R0_RESERVED = 16
    cursor = _R0_RESERVED
    r1_trees_off = cursor
    cursor += NREL_MESH_TREE_SIZE              # one static mesh-tree
    r3_chunks_off = cursor
    cursor += NREL_CHUNK_SIZE                   # one chunk
    r4_fmt2_off = cursor
    cursor += NREL_FMT2_HEADER_SIZE

    has_tex = len(texture_names) > 0
    if has_tex:
        r5_texlist_off = cursor
        cursor += NREL_TEXLIST_SIZE
        r6_entries_off = cursor
        cursor += NREL_TEXLIST_ENTRY_SIZE * len(texture_names)
    else:
        r5_texlist_off = 0
        r6_entries_off = 0

    # R7 nodes.
    r7_nodes_off = cursor
    node_offsets = [r7_nodes_off + i * NREL_MESH_TREE_SIZE for i in range(n_nodes)]
    cursor += NREL_MESH_TREE_SIZE * n_nodes

    # Which nodes own geometry (one XjMesh each).
    geom_nodes = [i for i, nd in enumerate(nodes) if nd.submeshes]

    # SHARED VERTEX BUFFERS: submeshes that reference the SAME vertex-list
    # object (``id()`` identity — the stripified path reuses one array per
    # source mesh across all its strips) emit a SINGLE VertexBufferContainer
    # + a SINGLE on-disk vertex array.  Each submesh's IndexBufferContainer
    # then points at the shared vbuf via ``vertex_buffer_index``.  This is
    # the density win: K strips over one mesh write the vertex array ONCE,
    # not K times.  The 1-tri-per-strip baseline gives every submesh a
    # distinct 3-vertex list, so dedup is a no-op there (one vbuf each).
    vbufs_for_node: dict = {}    # node_i -> list of vertex-list objects (unique)
    sub_vbidx: dict = {}         # (node_i, sub_j) -> index into vbufs_for_node[i]
    for i in geom_nodes:
        uniq: List = []
        seen: dict = {}          # id(vertex-list) -> vbuf index
        for j, sm in enumerate(nodes[i].submeshes):
            key = id(sm.vertices)
            vb = seen.get(key)
            if vb is None:
                vb = len(uniq)
                seen[key] = vb
                uniq.append(sm.vertices)
            sub_vbidx[(i, j)] = vb
        vbufs_for_node[i] = uniq

    # R8 XjMesh structs.
    mesh_off: dict = {}
    for i in geom_nodes:
        mesh_off[i] = cursor
        cursor += NREL_XJ_MESH_SIZE

    # R8a VertexBufferContainer[] — one per UNIQUE vertex buffer per geom node.
    vbuf_off: dict = {}      # (node_i) -> base offset of its vbuf array
    vbuf_count: dict = {}
    for i in geom_nodes:
        vbuf_off[i] = cursor
        vbuf_count[i] = len(vbufs_for_node[i])
        cursor += NREL_VBUF_CONTAINER_SIZE * len(vbufs_for_node[i])

    # R8b IndexBufferContainer[] — one per submesh per geom node.
    ibuf_off: dict = {}
    ibuf_count: dict = {}
    for i in geom_nodes:
        ibuf_off[i] = cursor
        ibuf_count[i] = len(nodes[i].submeshes)
        cursor += NREL_IBUF_CONTAINER_SIZE * len(nodes[i].submeshes)

    # R8c RenderStateArgs[] — one type-3 entry per submesh.
    rs_off: dict = {}        # (node_i, sub_j) -> offset
    for i in geom_nodes:
        for j in range(len(nodes[i].submeshes)):
            rs_off[(i, j)] = cursor
            cursor += NREL_RSARGS_SIZE

    # R9v raw vertex arrays — one per UNIQUE vertex buffer.
    varr_off: dict = {}      # (node_i, vbuf_k) -> offset
    for i in geom_nodes:
        for k, vl in enumerate(vbufs_for_node[i]):
            varr_off[(i, k)] = cursor
            cursor += NREL_VERTEX_STRIDE * len(vl)

    # R9i raw index arrays — one per submesh.
    iarr_off: dict = {}
    for i in geom_nodes:
        for j, sm in enumerate(nodes[i].submeshes):
            iarr_off[(i, j)] = cursor
            cursor += 2 * len(sm.indices)

    # R10 texture name strings (4-aligned each).
    name_off: dict = {}
    if has_tex:
        for k, nm in enumerate(texture_names):
            name_off[k] = cursor
            cursor += ((len(nm.encode("ascii", errors="replace")) + 1 + 3) & ~3)

    data_size = cursor

    # ---- PASS 2: write + record pointer locations ----
    buf = bytearray(data_size)
    ptr_locs: List[int] = []

    # Compute the chunk-enclosing radius from world-space vertex
    # positions (chunk is at chunk_pos; verts are stored chunk-local
    # i.e. world = local + chunk_pos at extract time -> here vertices
    # are authored already as the local frame around chunk origin 0).
    all_local_pts: List[Tuple[float, float, float]] = []
    for nd in nodes:
        for sm in nd.submeshes:
            for v in sm.vertices:
                all_local_pts.append(v.pos)
    chunk_radius = _nrel_enclosing_sphere(all_local_pts) if all_local_pts else 1.0

    # R1: static mesh-tree (root_node_ptr -> first node).
    struct.pack_into(
        _NREL_MESH_TREE_FMT, buf, r1_trees_off,
        node_offsets[0],   # root_node_ptr
        0,                 # unk1
        0,                 # tex_anim_info_ptr (omitted)
        0,                 # tree_flags
    )
    ptr_locs.append(r1_trees_off + 0x00)   # root_node_ptr FLAG
    # tex_anim_info_ptr null -> NOT flagged.

    # R3: chunk.
    struct.pack_into(
        _NREL_CHUNK_FMT, buf, r3_chunks_off,
        0,                                  # id
        float(chunk_pos[0]), float(chunk_pos[1]), float(chunk_pos[2]),
        0, 0, 0,                            # rot BAMS
        float(chunk_radius),
        r1_trees_off,                       # static_mesh_trees_ptr
        0,                                  # animated_mesh_trees_ptr
        1,                                  # static_count
        0,                                  # animated_count
        chunk_flags & 0xFFFFFFFF,
    )
    ptr_locs.append(r3_chunks_off + 0x20)   # static_mesh_trees_ptr FLAG
    # animated_mesh_trees_ptr null (count 0) -> NOT flagged.

    # R4: NrelFmt2 header.
    tex_data_ptr = r5_texlist_off if has_tex else 0
    struct.pack_into(
        _NREL_FMT2_HEADER_FMT, buf, r4_fmt2_off,
        NREL_FMT2_MAGIC,
        fmt2_unk1 & 0xFFFFFFFF,
        1,                                  # chunk_count
        0,                                  # unk2
        float(chunk_radius),
        r3_chunks_off,                      # chunks_ptr
        tex_data_ptr,                       # texture_data_ptr
    )
    ptr_locs.append(r4_fmt2_off + 0x10)     # chunks_ptr FLAG
    if has_tex:
        ptr_locs.append(r4_fmt2_off + 0x14)  # texture_data_ptr FLAG (only with a list)

    # R5/R6: texture list + entries.
    if has_tex:
        struct.pack_into(
            _NREL_TEXLIST_FMT, buf, r5_texlist_off,
            r6_entries_off, len(texture_names),
        )
        ptr_locs.append(r5_texlist_off + 0x00)   # elements_ptr FLAG
        for k in range(len(texture_names)):
            ent = r6_entries_off + k * NREL_TEXLIST_ENTRY_SIZE
            struct.pack_into("<III", buf, ent, name_off[k], 0, 0)
            ptr_locs.append(ent + 0x00)          # name_ptr FLAG
            # +0x04, +0x08 runtime pointers (on-disk 0) -> NOT flagged.

    # R7: mesh-tree nodes.
    for i, nd in enumerate(nodes):
        n_off = node_offsets[i]
        mptr = mesh_off.get(i, 0)
        child_ptr = node_offsets[nd.child_index] if nd.child_index >= 0 else 0
        next_ptr = node_offsets[nd.sibling_index] if nd.sibling_index >= 0 else 0
        struct.pack_into(
            "<II3f3i3fII", buf, n_off,
            nd.eval_flags & 0xFFFFFFFF,
            mptr,
            float(nd.position[0]), float(nd.position[1]), float(nd.position[2]),
            int(nd.rotation_bams[0]), int(nd.rotation_bams[1]), int(nd.rotation_bams[2]),
            float(nd.scale[0]), float(nd.scale[1]), float(nd.scale[2]),
            child_ptr,
            next_ptr,
        )
        # mesh_ptr is flagged ONLY when the node owns a mesh.  A pure
        # transform node (no submeshes) keeps mesh_ptr==0 UNflagged,
        # matching vanilla.
        if i in mesh_off:
            ptr_locs.append(n_off + 0x04)        # mesh_ptr FLAG
        if nd.child_index >= 0:
            ptr_locs.append(n_off + 0x2C)        # child_ptr FLAG (non-null)
        if nd.sibling_index >= 0:
            ptr_locs.append(n_off + 0x30)        # next_ptr FLAG (non-null)

    # R8: XjMesh structs.
    for i in geom_nodes:
        m_off = mesh_off[i]
        struct.pack_into(
            _NREL_XJ_MESH_FMT, buf, m_off,
            0,                       # flags
            vbuf_off[i], vbuf_count[i],
            ibuf_off[i], ibuf_count[i],
            0, 0,                    # alpha_index_buffers_ptr/count (omitted)
        )
        ptr_locs.append(m_off + 0x04)            # vertex_buffers_ptr FLAG
        ptr_locs.append(m_off + 0x0C)            # index_buffers_ptr FLAG
        # alpha_index_buffers_ptr null (count 0) -> NOT flagged.

    # R8a: VertexBufferContainer[] — one per UNIQUE vertex buffer.
    for i in geom_nodes:
        for k, vl in enumerate(vbufs_for_node[i]):
            vb = vbuf_off[i] + k * NREL_VBUF_CONTAINER_SIZE
            struct.pack_into(
                _NREL_VBUF_FMT, buf, vb,
                NREL_VERTEX_FORMAT,
                varr_off[(i, k)],
                NREL_VERTEX_STRIDE,
                len(vl),
            )
            ptr_locs.append(vb + 0x04)           # vertices_ptr FLAG

    # R8b: IndexBufferContainer[] — one per submesh; vertex_buffer_index
    # points at the (possibly SHARED) deduped vbuf.
    for i in geom_nodes:
        for j, sm in enumerate(nodes[i].submeshes):
            ib = ibuf_off[i] + j * NREL_IBUF_CONTAINER_SIZE
            struct.pack_into(
                _NREL_IBUF_FMT, buf, ib,
                rs_off[(i, j)], 1,               # renderstate_args_ptr + count
                iarr_off[(i, j)], len(sm.indices),
                sub_vbidx[(i, j)],               # vertex_buffer_index (shared)
            )
            ptr_locs.append(ib + 0x00)           # renderstate_args_ptr FLAG (count>0)
            ptr_locs.append(ib + 0x08)           # indices_ptr FLAG

    # R8c: RenderStateArgs[] (type 3 = TEXTURE_ID).
    for i in geom_nodes:
        for j, sm in enumerate(nodes[i].submeshes):
            rs = rs_off[(i, j)]
            struct.pack_into(
                _NREL_RSARGS_FMT, buf, rs,
                3,                               # state_type = TEXTURE_ID
                sm.texture_id & 0xFFFFFFFF,      # arg1 = texture id
                0, 0,
            )
            # RenderStateArgs has no pointers.

    # R9v: raw vertex arrays (vertex_format 3 stride 32) — one per UNIQUE vbuf.
    for i in geom_nodes:
        for k, vl in enumerate(vbufs_for_node[i]):
            base = varr_off[(i, k)]
            for vi, v in enumerate(vl):
                vo = base + vi * NREL_VERTEX_STRIDE
                struct.pack_into("<3f", buf, vo,
                                 float(v.pos[0]), float(v.pos[1]), float(v.pos[2]))
                struct.pack_into("<3f", buf, vo + 12,
                                 float(v.normal[0]), float(v.normal[1]), float(v.normal[2]))
                struct.pack_into("<2f", buf, vo + 24,
                                 float(v.uv[0]), float(v.uv[1]))

    # R9i: raw index arrays.
    for i in geom_nodes:
        for j, sm in enumerate(nodes[i].submeshes):
            if sm.indices:
                struct.pack_into(
                    f"<{len(sm.indices)}H", buf, iarr_off[(i, j)],
                    *[idx & 0xFFFF for idx in sm.indices])

    # R10: texture name strings.
    if has_tex:
        for k, nm in enumerate(texture_names):
            raw = nm.encode("ascii", errors="replace") + b"\x00"
            buf[name_off[k]:name_off[k] + len(raw)] = raw

    return bytes(buf), ptr_locs, r4_fmt2_off


def build_nrel_from_meshes(
    nodes: Sequence[NrelNode],
    texture_names: Sequence[str],
    *,
    chunk_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    chunk_flags: int = DEFAULT_NREL_CHUNK_FLAGS,
    fmt2_unk1: int = DEFAULT_NREL_FMT2_UNK1,
    enforce_budget: bool = True,
) -> bytes:
    """Build a fresh n.rel from authored mesh-tree nodes + texture names.

    Parameters
    ----------
    nodes:
        Mesh-tree nodes (node 0 is the static-tree root).  Each node's
        ``submeshes`` become its XjMesh; each submesh is one draw-strip.
    texture_names:
        Texture name list; ``texture_id`` on each submesh indexes it.
        Empty -> no TextureList region (texture_data_ptr == 0).
    chunk_pos:
        World origin of the single visibility chunk.
    enforce_budget:
        When True (default) raises ``RelWriteError`` if the result
        exceeds ``NREL_SIZE_BUDGET`` (768 KB).

    Returns
    -------
    bytes
        A complete n.rel container that re-parses via ``formats.rel``,
        passes ``simulate_rel_relocation``, and (when ``enforce_budget``)
        fits the 768 KB cap.

    Raises
    ------
    RelWriteError
        On an empty node list, an out-of-range child/sibling index, or a
        budget overflow.

    Notes
    -----
    Child/sibling node chains are written and relocate correctly, but the
    v1 ``formats.rel.extract_nrel_meshes`` reader walks only the ROOT
    node of each mesh-tree (it does not descend ``child_ptr``/``next_ptr``
    in the bake path).  ``nrel_nodes_from_meshes`` therefore merges all
    submeshes into a single root node so the round-trip is complete; emit
    multi-node trees only when the consuming reader walks the chain.
    """
    for nd in nodes:
        if nd.child_index >= len(nodes) or nd.sibling_index >= len(nodes):
            raise RelWriteError("node child/sibling index out of range")

    data, ptr_locs, payload_off = _build_nrel_data(
        nodes, texture_names, chunk_pos, chunk_flags, fmt2_unk1,
    )
    ptr_locs.sort()
    # Self-check: every pointer location is unique & 4-aligned (a layout
    # bug otherwise; assemble_rel re-checks but we want a clear message).
    if len(set(ptr_locs)) != len(ptr_locs):
        raise RelWriteError("duplicate pointer location (n.rel layout bug)")

    out = assemble_rel(data, ptr_locs, payload_off)
    if enforce_budget and len(out) > NREL_SIZE_BUDGET:
        raise RelWriteError(
            f"n.rel is {len(out)} bytes, exceeds the 0x{NREL_SIZE_BUDGET:x} "
            f"(768 KB) budget — decimate the geometry")
    return out


def nrel_pointer_count(
    nodes: Sequence[NrelNode],
    texture_names: Sequence[str],
) -> int:
    """Closed-form expected relocation-entry count for the authored model.

    A numeric self-check companion to :func:`build_nrel_from_meshes`:
    a stray/missing pointer flag shows up as a count mismatch instead of
    only as a load-time crash.

    Count =
        2   (fmt2 chunks_ptr + texture_data_ptr-if-textured)  -- see below
      + 1   (chunk static_mesh_trees_ptr)
      + 1   (static mesh-tree root_node_ptr)
      + per node:   1 mesh_ptr-if-geometry + 1 child-if + 1 sibling-if
      + per geom node: 2 (vertex_buffers_ptr + index_buffers_ptr)
      + per UNIQUE vertex buffer: 1 vbuf vertices_ptr  (submeshes that share
        one vertex-list object — the stripified path — share one vbuf)
      + per submesh: 2 ibuf (rs_ptr + indices_ptr)
      + 1   texlist elements_ptr (if textured)
      + per name: 1 name_ptr
    """
    has_tex = len(texture_names) > 0
    count = 0
    count += 1                       # fmt2 chunks_ptr
    if has_tex:
        count += 1                   # fmt2 texture_data_ptr
    count += 1                       # chunk static_mesh_trees_ptr
    count += 1                       # mesh-tree root_node_ptr
    for nd in nodes:
        if nd.submeshes:
            count += 1               # mesh_ptr
        if nd.child_index >= 0:
            count += 1
        if nd.sibling_index >= 0:
            count += 1
    for nd in nodes:
        if not nd.submeshes:
            continue
        count += 2                   # XjMesh vertex_buffers_ptr + index_buffers_ptr
        # Unique vertex buffers (deduped by vertex-list object identity, the
        # same rule _build_nrel_data uses) each get one vertices_ptr.
        seen_vb: set = set()
        for sm in nd.submeshes:
            key = id(sm.vertices)
            if key not in seen_vb:
                seen_vb.add(key)
                count += 1           # vbuf vertices_ptr (one per unique buffer)
            count += 2               # ibuf rs_ptr + indices_ptr
    if has_tex:
        count += 1                   # texlist elements_ptr
        count += len(texture_names)  # name_ptrs
    return count


def _mesh_local_vertices(
    mesh, ox: float, oy: float, oz: float,
) -> List[NrelVertex]:
    """The mesh's full vertex list, chunk-local (``world - chunk_origin``)."""
    return [
        NrelVertex(
            pos=(float(v.pos[0]) - ox, float(v.pos[1]) - oy, float(v.pos[2]) - oz),
            normal=tuple(float(c) for c in v.normal),
            uv=tuple(float(c) for c in v.uv),
        )
        for v in mesh.vertices
    ]


def nrel_submeshes_stripified(
    meshes: Sequence,
    *,
    chunk_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    texture_index_of=None,
    max_strip_len: Optional[int] = None,
) -> List[NrelSubmesh]:
    """Stripify each source mesh into ONE submesh PER STRIP over a SHARED array.

    The density win: instead of one submesh (VertexInfo+Strip+Material row
    + 3 duplicated verts) per triangle, a source mesh emits ONE submesh per
    triangle-STRIP, all sharing the mesh's FULL vertex array.  Long strips
    coalesce many triangles, so the per-strip overhead amortises and no
    vertex is duplicated — far more triangles fit under the 768 KB cap.

    Strips are built by :func:`formats.strippify.stripify` (deterministic
    greedy adjacency walk).  Winding may flip per triangle, but the reader's
    ``_strip_to_triangles`` parity reproduces the same triangle VERTEX SET
    (the correctness gate — compare unordered triples, not index order).

    ``texture_id`` is per SOURCE MESH (one strip can only carry one texture
    via its RenderStateArgs), matching ``nrel_nodes_from_meshes``.  A mesh
    whose strips reference only a subset of its vertices still ships the full
    shared array; ``extract_nrel_meshes`` compacts unused slots on re-parse.
    """
    from formats.strippify import stripify

    ox, oy, oz = chunk_origin

    def _tex(mid: int) -> int:
        if texture_index_of is not None:
            return int(texture_index_of(mid))
        return max(0, int(mid))

    submeshes: List[NrelSubmesh] = []
    for mesh in meshes:
        verts = _mesh_local_vertices(mesh, ox, oy, oz)
        tex = _tex(getattr(mesh, "material_id", 0))
        idx = list(mesh.indices)
        n_tri = len(idx) // 3
        if n_tri == 0:
            continue
        faces = [idx[t:t + 3] for t in range(0, n_tri * 3, 3)]
        strips = stripify(faces, max_strip_len=max_strip_len)
        for strip in strips:
            if len(strip) < 3:
                continue
            submeshes.append(NrelSubmesh(
                vertices=verts,        # SHARED full array (same object reuse)
                indices=list(strip),   # u16 triangle-strip indices
                texture_id=tex,
            ))
    return submeshes


def nrel_nodes_from_meshes(
    meshes: Sequence,
    *,
    chunk_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    texture_index_of=None,
    stripify: bool = True,
    max_strip_len: Optional[int] = None,
) -> List[NrelNode]:
    """Convert world-space submeshes into a single-node authoring tree.

    Each input ``mesh`` exposes ``vertices`` (``.pos`` / ``.normal`` /
    ``.uv``), an ``indices`` triangle LIST, and ``material_id`` — the
    ``formats.xj.XjMesh`` / ``extract_nrel_meshes`` shape.

    With ``stripify=True`` (default) each source mesh is coalesced into long
    triangle strips over a SHARED vertex array via
    :func:`nrel_submeshes_stripified` — far fewer submeshes and no
    per-triangle vertex duplication, so much more geometry fits the 768 KB
    cap.  The round-trip triangle VERTEX SET is preserved (winding may flip;
    the reader recomputes it) — compare unordered triples, never index order.

    With ``stripify=False`` each input triangle becomes one 3-index strip
    (its own 3-vertex submesh): the correctness BASELINE, where a 3-index
    strip ``[a,b,c]`` de-strips to ``(a,b,c)`` with no parity flip.

    ``texture_index_of(material_id) -> int`` maps a source material id to
    a texture-name index; defaults to identity (clamped to >= 0).  All
    vertices are stored chunk-local: ``local = world - chunk_origin`` so
    that ``extract_nrel_meshes`` (which adds the chunk origin back)
    reproduces the world positions.
    """
    if stripify:
        submeshes = nrel_submeshes_stripified(
            meshes,
            chunk_origin=chunk_origin,
            texture_index_of=texture_index_of,
            max_strip_len=max_strip_len,
        )
        return [NrelNode(submeshes=submeshes)]

    ox, oy, oz = chunk_origin

    def _tex(mid: int) -> int:
        if texture_index_of is not None:
            return int(texture_index_of(mid))
        return max(0, int(mid))

    submeshes: List[NrelSubmesh] = []
    for mesh in meshes:
        verts = _mesh_local_vertices(mesh, ox, oy, oz)
        tex = _tex(getattr(mesh, "material_id", 0))
        idx = list(mesh.indices)
        for t in range(0, len(idx) - 2, 3):
            a, b, c = idx[t], idx[t + 1], idx[t + 2]
            # Each triangle is its own 3-vertex submesh (one 3-index
            # strip => no _strip_to_triangles parity flip).  Carry ONLY
            # the 3 referenced vertices so the vertex array isn't
            # duplicated per triangle (that would bloat the file ~10x).
            submeshes.append(NrelSubmesh(
                vertices=[verts[a], verts[b], verts[c]],
                indices=[0, 1, 2],
                texture_id=tex,
            ))
    return [NrelNode(submeshes=submeshes)]


__all__ = [
    "RelWriteError",
    # STEP 0
    "encode_rel_pointer_table",
    "decode_rel_pointer_table",
    # STEP 1
    "assemble_rel",
    # STEP 2 — c.rel
    "CrelFace",
    "CrelNode",
    "CrelModel",
    "parse_crel_for_writer",
    "encode_crel",
    "build_crel",
    "simulate_rel_relocation",
    "CREL_MESH_SIZE",
    "CREL_FACE_SIZE",
    "CREL_NODE_SIZE",
    "CREL_SIZE_BUDGET",
    "DEFAULT_FACE_FLAGS",
    "DEFAULT_NODE_FLAGS",
    # STEP 3 — n.rel
    "NrelMeshTreeRec",
    "NrelChunkRec",
    "NrelLayoutHint",
    "NrelModel",
    "parse_nrel_for_writer",
    "encode_nrel",
    "NREL_FMT2_HEADER_SIZE",
    "NREL_CHUNK_SIZE",
    "NREL_MESH_TREE_SIZE",
    # STEP 3b — n.rel from-scratch authoring
    "NrelVertex",
    "NrelSubmesh",
    "NrelNode",
    "build_nrel_from_meshes",
    "nrel_pointer_count",
    "nrel_nodes_from_meshes",
    "nrel_submeshes_stripified",
    "NREL_SIZE_BUDGET",
    "NREL_VERTEX_FORMAT",
    "NREL_VERTEX_STRIDE",
    "NREL_NODE_EVAL_FLAGS",
]
