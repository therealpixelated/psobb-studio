"""PSOBB ``.rel`` relocation-table parser (2026-04-25).

PSOBB ships some map-specific data inside ``.rel`` archives.  These are
*relocatable* containers that follow the format Sonic Adventure / Ninja
SDK uses for level data: a flat byte buffer plus a pointer-fixup table
that describes where in the buffer there are 32-bit offsets that must be
treated as pointers (rather than raw integers).

There are three flavours used by PSOBB scene/ files:

  ``*_NN c.rel``
      "collision".  A list of ``CrelNode`` records carrying mesh
      bounding info and per-face flags (water / wall / camera-only /
      …).  No render-quality vertices.

  ``*_NN n.rel``
      "node" (geometry).  Carries an ``NrelFmt2`` payload whose magic
      bytes are ``"fmt2"``.  Inside it is a list of visibility chunks,
      each pointing at one or more XJ mesh trees.  This is what the
      renderer turns into the actual terrain on Pioneer 2 / city / lab
      maps that ship NO ``*_NN s.nj`` sibling.

  ``*_NN r.rel``
      "render / scene anchors" (named map anchors with positions,
      activation radii, and per-anchor sub-config).  v3 finding: this is
      NOT fog/lighting (those live in PSOBB.exe globals at 0x00a8d770 /
      0x00a9d4e4).  r.rel carries the per-area set of named anchor
      records (teleporters, room entry points, camera ROI markers,
      etc.) — useful for deriving a scene bounding box and for the Map
      Editor's anchor-picker.  See :func:`read_rrel_anchors`.

This module documents the file format and exposes:

  :class:`RelFile`
      Dataclass holding the resolved buffer, payload entry, and
      relocation table.
  :func:`parse_rel`
      Top-level parser; returns :class:`RelFile`.
  :func:`extract_nrel_meshes`
      Specifically extract :class:`XjMesh` lists from an ``n.rel``
      buffer for the renderer.
  :func:`is_n_rel`, :func:`is_c_rel`, :func:`is_r_rel`
      Trailer-shape sniffers; safe on any byte input.

REL file format (32-byte trailer at end of file):

::

    +------------------------------------------+
    |          data section (relocatable)      |
    +------------------------------------------+ <- pointer_table_offset
    |     pointer table: u16 deltas /4         |
    |     each delta is "skip N*4 bytes from   |
    |     previous pointer to find the next    |
    |     pointer location"                    |
    +------------------------------------------+ <- file_size - 0x20
    |  +0x00: u32 pointer_table_offset         |
    |  +0x04: u32 pointer_count                |
    |  +0x08: u32 1   (always 1)               |
    |  +0x0C: u32 0                            |
    |  +0x10: u32 payload_offset (entry point) |
    |  +0x14..0x1F: zeros                      |
    +------------------------------------------+

Each pointer in the data section is a 32-bit absolute byte offset
within the data section itself; at runtime PSOBB walks the pointer
table and adds the buffer's load address to every word that the table
flags as a pointer.

For our editor we don't *apply* the relocations (we keep them as
offsets into the byte buffer), but we DO track which uint32's in the
data are pointers so that walks like NrelFmt2 → chunks → mesh trees
can dereference them safely.
"""
from __future__ import annotations

import logging
import math
import struct
from dataclasses import dataclass, field
from typing import Iterator, Optional

# BAMS (Binary Angular Measurement System): PSOBB/Ninja stores rotations as
# signed 16/32-bit angle units where a full turn == 0x10000.  psov2's
# ``BitStream.readRot3`` converts with ``angle * (2*PI / 0xFFFF)`` — we mirror
# that EXACT constant so chunk placement lines up byte-for-byte with psov2.
_BAMS_TO_RAD = (2.0 * math.pi) / 0xFFFF


def _chunk_world_matrix(
    pos: tuple[float, float, float],
    rot_bams: tuple[int, int, int],
) -> tuple[float, ...]:
    """Build a 3x4 row-major world matrix for one n.rel visibility chunk.

    Mirrors psov2 ``NinjaStage.prepare`` / ``NinjaRoom.prepare``: each section
    (chunk) places its mesh tree with a Y rotation (and, defensively, X/Z) plus
    a translation.  psov2 only multiplies ``makeRotationY(rot.y)`` then sets the
    bone position, but PSOBB chunks can in principle carry X/Z rotation too, so
    we compose the full ZYX-Euler rotation (the same order the per-bone walker
    uses) before translating.  Returns a flat 12-tuple
    ``(m00,m01,m02, m10,m11,m12, m20,m21,m22, tx,ty,tz)`` so callers can apply
    ``out = M . v`` cheaply without pulling in numpy.
    """
    rx = rot_bams[0] * _BAMS_TO_RAD
    ry = rot_bams[1] * _BAMS_TO_RAD
    rz = rot_bams[2] * _BAMS_TO_RAD
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    # R = Rz * Ry * Rx  (apply X first, then Y, then Z — matches the
    # NinjaModel/NinjaStage readBone order: makeRotationX, then Y, then Z,
    # each left-multiplied via applyMatrix → effective ZYX composition).
    m00 = cy * cz
    m01 = sx * sy * cz - cx * sz
    m02 = cx * sy * cz + sx * sz
    m10 = cy * sz
    m11 = sx * sy * sz + cx * cz
    m12 = cx * sy * sz - sx * cz
    m20 = -sy
    m21 = sx * cy
    m22 = cx * cy
    return (m00, m01, m02, m10, m11, m12, m20, m21, m22,
            pos[0], pos[1], pos[2])


def _apply_matrix(
    m: tuple[float, ...], x: float, y: float, z: float,
    translate: bool = True,
) -> tuple[float, float, float]:
    """Apply the 3x4 matrix from :func:`_chunk_world_matrix` to (x, y, z).

    ``translate=False`` skips the translation column — used for normals
    (direction vectors), which rotate with the chunk but must not shift.
    """
    ox = m[0] * x + m[1] * y + m[2] * z
    oy = m[3] * x + m[4] * y + m[5] * z
    oz = m[6] * x + m[7] * y + m[8] * z
    if translate:
        ox += m[9]
        oy += m[10]
        oz += m[11]
    return ox, oy, oz

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trailer constants (mirror pso-blender's rel.py)
# ---------------------------------------------------------------------------
TRAILER_SIZE = 0x20  # 32 bytes

# Trailer offsets, relative to trailer start (= file_size - 0x20):
TRAILER_PTR_TABLE_OFFSET = 0x00      # u32 abs offset to pointer-delta table
TRAILER_PTR_COUNT_OFFSET = 0x04      # u32 number of entries in that table
TRAILER_FORMAT_FLAG_OFFSET = 0x08    # u32 always 1 in PSOBB
TRAILER_PAYLOAD_OFFSET = 0x10        # u32 abs offset to payload (entry point)
# 0x14..0x1F are reserved zero bytes.


# Magic bytes that identify well-known payload structures inside the
# data section.  We sniff these to pick the right reader.
NREL_FMT2_MAGIC = b"fmt2"


class RelParseError(ValueError):
    """Raised when a .rel buffer is too small / malformed / not REL."""


# ---------------------------------------------------------------------------
# RelFile dataclass
# ---------------------------------------------------------------------------
@dataclass
class RelFile:
    """Decoded .rel container.

    Attributes
    ----------
    data:
        The full byte buffer.  ``data[:pointer_table_offset]`` is the
        relocatable data section; consumers reach into it with offsets.
    payload_offset:
        Absolute byte offset of the payload's entry struct.  This is
        what every higher-level reader (``CrelNode``, ``NrelFmt2``)
        starts from.
    pointer_table_offset:
        Where the u16 delta-table begins.
    pointer_count:
        How many entries are in the pointer-delta table.
    pointer_offsets:
        Resolved list of *absolute* offsets within ``data`` where a
        u32 pointer lives.  Sorted ascending.
    """
    data: bytes
    payload_offset: int
    pointer_table_offset: int
    pointer_count: int
    pointer_offsets: list[int] = field(default_factory=list)

    @property
    def data_size(self) -> int:
        """Length (bytes) of the relocatable data section.

        This excludes the pointer table + trailer at the tail.
        """
        return self.pointer_table_offset

    def is_pointer(self, abs_offset: int) -> bool:
        """``True`` if a u32 located at ``abs_offset`` is a pointer."""
        # Linear search is fine — pointer counts are typically small
        # (<5 K) and most callers ask once per dereference.  Caller can
        # build a set if the workload is hot.
        return abs_offset in self.pointer_offsets

    def read_u32(self, abs_offset: int) -> int:
        """Read a little-endian u32 at ``abs_offset``."""
        if abs_offset < 0 or abs_offset + 4 > self.data_size:
            raise RelParseError(
                f"u32 read out-of-range: {abs_offset} (data_size={self.data_size})")
        return struct.unpack_from("<I", self.data, abs_offset)[0]

    def read_u16(self, abs_offset: int) -> int:
        if abs_offset < 0 or abs_offset + 2 > self.data_size:
            raise RelParseError(
                f"u16 read out-of-range: {abs_offset} (data_size={self.data_size})")
        return struct.unpack_from("<H", self.data, abs_offset)[0]

    def read_f32(self, abs_offset: int) -> float:
        if abs_offset < 0 or abs_offset + 4 > self.data_size:
            raise RelParseError(
                f"f32 read out-of-range: {abs_offset} (data_size={self.data_size})")
        return struct.unpack_from("<f", self.data, abs_offset)[0]

    def deref(self, ptr_field_offset: int) -> int:
        """Read the u32 at ``ptr_field_offset`` interpreting it as a
        relocatable offset.

        Returns 0 (NULLPTR) if the field is null.  Raises
        ``RelParseError`` if the pointer points outside the data
        section.
        """
        v = self.read_u32(ptr_field_offset)
        if v == 0:
            return 0
        if v >= self.data_size:
            raise RelParseError(
                f"pointer at 0x{ptr_field_offset:x} -> 0x{v:x} OOB "
                f"(data_size=0x{self.data_size:x})")
        return v


# ---------------------------------------------------------------------------
# Trailer + pointer-table parsing
# ---------------------------------------------------------------------------
def _read_trailer(buf: bytes) -> tuple[int, int, int]:
    """Decode the 32-byte trailer.

    Returns
    -------
    (pointer_table_offset, pointer_count, payload_offset)
    """
    if len(buf) < TRAILER_SIZE:
        raise RelParseError(
            f"buffer too small for REL trailer: {len(buf)} bytes")
    trailer_start = len(buf) - TRAILER_SIZE
    pt_off = struct.unpack_from("<I", buf, trailer_start + TRAILER_PTR_TABLE_OFFSET)[0]
    pt_cnt = struct.unpack_from("<I", buf, trailer_start + TRAILER_PTR_COUNT_OFFSET)[0]
    pl_off = struct.unpack_from("<I", buf, trailer_start + TRAILER_PAYLOAD_OFFSET)[0]
    fmt_flag = struct.unpack_from("<I", buf, trailer_start + TRAILER_FORMAT_FLAG_OFFSET)[0]
    if fmt_flag != 1:
        # Soft-warn — every PSOBB REL we know of has 1 here, but other
        # Ninja-derived formats use this slot for a flag.  We let the
        # parse continue so the caller can decide what to do.
        log.debug("REL: unexpected format flag %d (expected 1)", fmt_flag)
    if pt_off > trailer_start:
        raise RelParseError(
            f"pointer_table_offset 0x{pt_off:x} > trailer 0x{trailer_start:x}")
    if pl_off > trailer_start:
        raise RelParseError(
            f"payload_offset 0x{pl_off:x} > trailer 0x{trailer_start:x}")
    # The pointer table itself must fit between pt_off and trailer_start.
    table_bytes = trailer_start - pt_off
    needed = pt_cnt * 2
    if needed > table_bytes:
        raise RelParseError(
            f"pointer table overflow: count={pt_cnt} needs {needed} bytes "
            f"but only {table_bytes} available")
    return pt_off, pt_cnt, pl_off


def _resolve_pointer_offsets(buf: bytes, pt_off: int, pt_cnt: int) -> list[int]:
    """Walk the u16 delta table to produce absolute pointer locations.

    The first entry's delta is added to 0 (i.e. it's an absolute offset
    in 4-byte units).  Subsequent entries' deltas are relative to the
    previous absolute offset.  Each delta is multiplied by 4 because all
    pointers in PSOBB REL files are 4-byte aligned.
    """
    out: list[int] = []
    prev = 0
    for i in range(pt_cnt):
        slot_off = pt_off + i * 2
        if slot_off + 2 > len(buf):
            raise RelParseError(f"pointer table walk OOB at entry {i}")
        delta = struct.unpack_from("<H", buf, slot_off)[0]
        cur = prev + delta * 4
        out.append(cur)
        prev = cur
    return out


def parse_rel(buf: bytes) -> RelFile:
    """Decode a ``.rel`` byte buffer.

    Parameters
    ----------
    buf:
        Full file bytes.

    Returns
    -------
    RelFile
        Parsed container with payload offset + resolved pointer table.

    Raises
    ------
    RelParseError
        On any structural defect (truncated trailer, OOB offsets,
        negative deltas).  The caller should treat REL parsing as
        best-effort — we never throw on perfectly-aligned files but
        do not attempt heroic recovery on malformed ones.
    """
    if not isinstance(buf, (bytes, bytearray, memoryview)):
        raise RelParseError("input must be bytes-like")
    buf = bytes(buf)
    pt_off, pt_cnt, pl_off = _read_trailer(buf)
    pointer_offsets = _resolve_pointer_offsets(buf, pt_off, pt_cnt)
    return RelFile(
        data=buf,
        payload_offset=pl_off,
        pointer_table_offset=pt_off,
        pointer_count=pt_cnt,
        pointer_offsets=pointer_offsets,
    )


# ---------------------------------------------------------------------------
# Trailer-shape sniffers (fast, no full parse)
# ---------------------------------------------------------------------------
def _trailer_looks_valid(buf: bytes) -> bool:
    """Return True if the last 32 bytes look like a PSOBB REL trailer."""
    if len(buf) < TRAILER_SIZE:
        return False
    trailer = buf[-TRAILER_SIZE:]
    pt_off, pt_cnt, fmt_flag, reserved, pl_off = struct.unpack("<5I", trailer[:20])
    rest = trailer[20:]
    if any(b != 0 for b in rest):
        return False
    if fmt_flag != 1 or reserved != 0:
        return False
    file_end = len(buf) - TRAILER_SIZE
    # pt_off may equal file_end when pointer count is 0 (no entries).
    # pl_off may equal file_end if data section is empty (also fine).
    if pt_off > file_end or pl_off > file_end:
        return False
    if pt_cnt * 2 > (file_end - pt_off):
        return False
    return True


def is_rel(buf: bytes) -> bool:
    """Cheap shape check.  Use before ``parse_rel`` for unknown bytes."""
    return _trailer_looks_valid(buf)


def is_n_rel(rel: RelFile) -> bool:
    """``True`` if the payload starts with the ``"fmt2"`` magic.

    n.rel files (PSOBB scene-mesh archives) are identified by their
    payload's leading magic bytes.
    """
    if rel.payload_offset + 4 > rel.data_size:
        return False
    return rel.data[rel.payload_offset:rel.payload_offset + 4] == NREL_FMT2_MAGIC


def is_c_rel(rel: RelFile) -> bool:
    """``True`` if the payload looks like a Crel (collision) container.

    Crel layout: a single ``Ptr32<CrelNode>`` field at the payload.  The
    payload's u32 pointer is registered in the relocation table; if the
    payload extends past 12 bytes, the slot at +0x08 is zero (this is
    what distinguishes c.rel from r.rel — the latter carries a non-zero
    count in that slot).

    Some city/lab maps (e.g. ``map_city00_00c.rel``) ship a c.rel whose
    payload is exactly 4 bytes (just the head pointer); in that case
    the +0x08 sniff cannot run and we fall back to the head-pointer
    check alone.
    """
    if is_n_rel(rel):
        return False
    if rel.payload_offset + 4 > rel.data_size:
        return False
    if rel.payload_offset not in rel.pointer_offsets:
        return False
    # If the payload is large enough to carry an r.rel header, the +0x08
    # slot must be zero for this to be a c.rel.  When the payload is too
    # small for an r.rel header, accept c.rel by default.
    if rel.payload_offset + 12 <= rel.data_size:
        count_field = rel.read_u32(rel.payload_offset + 8)
        if count_field != 0:
            return False
    return True


def is_r_rel(rel: RelFile) -> bool:
    """``True`` if the payload looks like an Rrel (anchor/scene) container.

    Rrel layout (decoded 2026-04-25 from PsoBB.exe + 156 sample files):

        +0x00: u32 anchors_ptr        -> RrelAnchor[count]
        +0x04: u32 reserved (0)
        +0x08: u32 count              # number of anchors
        +0x0C..+0x1C: zero padding

    We sniff: payload+0x00 is a registered pointer, AND payload+0x08 is
    a non-zero count under our sanity cap (4096 entries — empirically
    the largest r.rel ships ~250).
    """
    if is_n_rel(rel):
        return False
    if rel.payload_offset + 12 > rel.data_size:
        return False
    if rel.payload_offset not in rel.pointer_offsets:
        return False
    count = rel.read_u32(rel.payload_offset + 8)
    return 0 < count <= 4096


# ---------------------------------------------------------------------------
# n.rel payload reader: NrelFmt2 + Chunk + MeshTree → terrain mesh ptrs
# ---------------------------------------------------------------------------
#
# Layout (matches pso-blender/n_rel.py):
#
#   NrelFmt2 (24 bytes)
#     +0x00: u32 magic ('fmt2')
#     +0x04: u32 unk1
#     +0x08: u16 chunk_count
#     +0x0A: u16 unk2
#     +0x0C: f32 radius (overwritten at runtime)
#     +0x10: u32 chunks_ptr  (-> Chunk[])
#     +0x14: u32 texture_data_ptr  (-> TextureList; optional)
#
#   Chunk (52 bytes)
#     +0x00: i32 id
#     +0x04: f32 x
#     +0x08: f32 y
#     +0x0C: f32 z
#     +0x10: i32 rot_x
#     +0x14: i32 rot_y
#     +0x18: i32 rot_z
#     +0x1C: f32 radius
#     +0x20: u32 static_mesh_trees_ptr  (-> MeshTree[])
#     +0x24: u32 animated_mesh_trees_ptr
#     +0x28: u32 static_mesh_tree_count
#     +0x2C: u32 animated_mesh_tree_count
#     +0x30: u32 flags
#
#   MeshTree (16 bytes)
#     +0x00: u32 root_node_ptr  (-> XjMeshTreeNode)
#     +0x04: u32 unk1
#     +0x08: u32 texture_animation_info_ptr
#     +0x0C: u32 tree_flags
#
# The MeshTreeNode at root_node_ptr is the standard 52-byte Ninja mesh
# tree node already understood by ``formats.xj`` (52 bytes; see
# _MESH_TREE_NODE_FMT).  Its ``mesh`` field points to an XjMesh struct,
# also already handled by formats.xj.  So once we resolve the chain
# down to a MeshTreeNode we can hand off to the existing XJ pipeline.

_NREL_FMT2_HEADER_FMT = "<4sIHHfII"  # magic, unk1, chunk_count, unk2, radius, chunks, texture
_NREL_FMT2_HEADER_SIZE = struct.calcsize(_NREL_FMT2_HEADER_FMT)
assert _NREL_FMT2_HEADER_SIZE == 24, _NREL_FMT2_HEADER_SIZE

_CHUNK_FMT = "<i3f3i fI I I I I"
_CHUNK_SIZE = struct.calcsize(_CHUNK_FMT)
assert _CHUNK_SIZE == 52, _CHUNK_SIZE

_MESH_TREE_FMT = "<IIII"
_MESH_TREE_SIZE = struct.calcsize(_MESH_TREE_FMT)
assert _MESH_TREE_SIZE == 16, _MESH_TREE_SIZE


@dataclass
class NrelChunk:
    """One visibility chunk inside an n.rel.  Mirrors the on-disk struct."""
    id: int
    x: float
    y: float
    z: float
    rot_x: int
    rot_y: int
    rot_z: int
    radius: float
    static_mesh_trees_ptr: int
    animated_mesh_trees_ptr: int
    static_mesh_tree_count: int
    animated_mesh_tree_count: int
    flags: int


@dataclass
class NrelMeshTree:
    """One MeshTree inside an n.rel chunk.  Points at a Ninja MeshTreeNode."""
    root_node_ptr: int
    unk1: int
    texture_animation_info_ptr: int
    tree_flags: int


@dataclass
class NrelHeader:
    """Decoded ``NrelFmt2`` payload header."""
    chunk_count: int
    chunks_ptr: int
    texture_data_ptr: int
    radius: float


def read_nrel_header(rel: RelFile) -> NrelHeader:
    """Read the ``NrelFmt2`` struct at ``rel.payload_offset``.

    Raises
    ------
    RelParseError
        If the payload doesn't start with the ``"fmt2"`` magic.
    """
    if not is_n_rel(rel):
        raise RelParseError("not an n.rel (payload magic != 'fmt2')")
    if rel.payload_offset + _NREL_FMT2_HEADER_SIZE > rel.data_size:
        raise RelParseError("n.rel payload truncated")
    magic, unk1, chunk_count, unk2, radius, chunks_ptr, texture_data_ptr = \
        struct.unpack_from(_NREL_FMT2_HEADER_FMT, rel.data, rel.payload_offset)
    return NrelHeader(
        chunk_count=chunk_count,
        chunks_ptr=chunks_ptr,
        texture_data_ptr=texture_data_ptr,
        radius=radius,
    )


def read_nrel_chunks(rel: RelFile, header: Optional[NrelHeader] = None) -> list[NrelChunk]:
    """Read all chunks referenced by the NrelFmt2 header."""
    if header is None:
        header = read_nrel_header(rel)
    if header.chunks_ptr == 0 or header.chunk_count == 0:
        return []
    out: list[NrelChunk] = []
    base = header.chunks_ptr
    for i in range(header.chunk_count):
        off = base + i * _CHUNK_SIZE
        if off + _CHUNK_SIZE > rel.data_size:
            log.warning("n.rel chunk %d OOB; truncating walk", i)
            break
        try:
            (cid, cx, cy, cz, rx, ry, rz, cradius,
             smt_ptr, amt_ptr, smt_count, amt_count, flags) = \
                struct.unpack_from(_CHUNK_FMT, rel.data, off)
        except struct.error:
            log.warning("n.rel chunk %d unpack failed", i)
            break
        out.append(NrelChunk(
            id=cid, x=cx, y=cy, z=cz,
            rot_x=rx, rot_y=ry, rot_z=rz,
            radius=cradius,
            static_mesh_trees_ptr=smt_ptr,
            animated_mesh_trees_ptr=amt_ptr,
            static_mesh_tree_count=smt_count,
            animated_mesh_tree_count=amt_count,
            flags=flags,
        ))
    return out


def read_mesh_trees(
    rel: RelFile, base_ptr: int, count: int,
) -> list[NrelMeshTree]:
    """Read ``count`` ``MeshTree`` records starting at ``base_ptr``."""
    if base_ptr == 0 or count == 0:
        return []
    out: list[NrelMeshTree] = []
    for i in range(count):
        off = base_ptr + i * _MESH_TREE_SIZE
        if off + _MESH_TREE_SIZE > rel.data_size:
            log.warning("MeshTree %d OOB; truncating", i)
            break
        try:
            root_ptr, unk1, anim_ptr, tree_flags = \
                struct.unpack_from(_MESH_TREE_FMT, rel.data, off)
        except struct.error:
            log.warning("MeshTree %d unpack failed", i)
            break
        out.append(NrelMeshTree(
            root_node_ptr=root_ptr,
            unk1=unk1,
            texture_animation_info_ptr=anim_ptr,
            tree_flags=tree_flags,
        ))
    return out


def iter_nrel_mesh_root_offsets(rel: RelFile) -> Iterator[tuple[NrelChunk, NrelMeshTree, int]]:
    """Walk an n.rel and yield (chunk, mesh_tree, mesh_tree_node_offset).

    For each (chunk, static-mesh-tree) pair, yields the absolute
    offset in ``rel.data`` of the ``XjMeshTreeNode`` that the standard
    Ninja walker can read.  Animated mesh trees are skipped (they are
    chunked-NJ and re-use the same walker, but cost more memory and
    aren't needed for v2 terrain rendering).
    """
    header = read_nrel_header(rel)
    for chunk in read_nrel_chunks(rel, header):
        trees = read_mesh_trees(rel, chunk.static_mesh_trees_ptr,
                                chunk.static_mesh_tree_count)
        for tree in trees:
            if tree.root_node_ptr == 0:
                continue
            yield chunk, tree, tree.root_node_ptr


def extract_nrel_mesh_root_offsets(rel: RelFile) -> list[int]:
    """Convenience: list every static mesh-tree-node offset in the file.

    The caller can pipe each entry into :func:`formats.xj.walk_xj_tree`
    or build raw geometry.  We don't depend on formats.xj here so this
    module stays usable without three.js.
    """
    return [off for (_c, _t, off) in iter_nrel_mesh_root_offsets(rel)]


# ---------------------------------------------------------------------------
# Convenience: list embedded texture names from an n.rel TextureList
# ---------------------------------------------------------------------------
#
# TextureList (8 bytes)
#   +0x00: u32 elements_ptr (-> TextureListEntry[])
#   +0x04: u32 count
#
# TextureListEntry (12 bytes)
#   +0x00: u32 name_ptr (-> AlignedString)
#   +0x04: u32 unk1     (runtime client pointer)
#   +0x08: u32 data     (runtime client pointer)
#
# AlignedString is a NUL-terminated C string aligned to 4 bytes.

_TEXLIST_ENTRY_SIZE = 12


def read_texture_names(rel: RelFile) -> list[str]:
    """Return the texture names referenced by the n.rel TextureList.

    Empty list when the n.rel has no embedded texture metadata or when
    parsing fails — callers should treat the result as best-effort.
    """
    try:
        header = read_nrel_header(rel)
    except RelParseError:
        return []
    if header.texture_data_ptr == 0:
        return []
    base = header.texture_data_ptr
    if base + 8 > rel.data_size:
        return []
    try:
        elements_ptr, count = struct.unpack_from("<II", rel.data, base)
    except struct.error:
        return []
    if elements_ptr == 0 or count == 0 or count > 1024:
        return []
    out: list[str] = []
    for i in range(count):
        ent_off = elements_ptr + i * _TEXLIST_ENTRY_SIZE
        if ent_off + _TEXLIST_ENTRY_SIZE > rel.data_size:
            break
        try:
            name_ptr, _unk1, _unk2 = struct.unpack_from("<III", rel.data, ent_off)
        except struct.error:
            break
        if name_ptr == 0 or name_ptr >= rel.data_size:
            out.append("")
            continue
        # Read NUL-terminated string at name_ptr
        end = rel.data.find(b"\x00", name_ptr, rel.data_size)
        if end < 0:
            end = rel.data_size
        try:
            out.append(rel.data[name_ptr:end].decode("ascii", errors="replace"))
        except (UnicodeDecodeError, IndexError):
            out.append("")
    return out


# ---------------------------------------------------------------------------
# n.rel XjMesh — D3D-style buffer descriptor (DIFFERENT from xj_descriptor)
# ---------------------------------------------------------------------------
#
# The mesh structure inside an n.rel is the D3D-style ``XjMesh`` (28
# bytes), NOT the descriptor-table ``XjModel`` (44 bytes) that
# ``formats/xj_descriptor.py`` parses.  Same name, different layout —
# pso-blender's ``xj.py`` calls it ``XjMesh`` while psolib's ``Xj.kt``
# uses ``XjModel``.  We refer to it as ``NrelXjMesh`` here to stay
# unambiguous.
#
# Layout (from pso-blender/pso_blender/xj.py + psolib reverse engineer):
#
#   XjMesh (28 bytes)                  ← what mesh_ptr in MeshTreeNode points at
#     +0x00: u32 flags
#     +0x04: u32 vertex_buffers_ptr     -> VertexBufferContainer[]
#     +0x08: u32 vertex_buffer_count
#     +0x0C: u32 index_buffers_ptr      -> IndexBufferContainer[]
#     +0x10: u32 index_buffer_count
#     +0x14: u32 alpha_index_buffers_ptr
#     +0x18: u32 alpha_index_buffer_count
#
#   VertexBufferContainer (16 bytes)
#     +0x00: u32 vertex_format (per Phantasmal Xj.kt; same id as type=N)
#     +0x04: u32 vertices_ptr           -> raw vertex array (vertex_size bytes per row)
#     +0x08: u32 vertex_size            stride in bytes
#     +0x0C: u32 vertex_count
#
#   IndexBufferContainer (20 bytes)
#     +0x00: u32 renderstate_args_ptr   -> RenderStateArgs[]
#     +0x04: u32 renderstate_args_count
#     +0x08: u32 indices_ptr            -> u16[] (triangle-strip indices)
#     +0x0C: u32 index_count
#     +0x10: u32 vertex_buffer_index    which vertex buffer to draw against
#
#   RenderStateArgs (16 bytes)
#     +0x00: u32 state_type
#     +0x04: u32 arg1                   meaning depends on state_type
#     +0x08: u32 arg2
#     +0x0C: u32 unk
#
# RenderState types we care about for material assembly:
#   2 = BLEND_MODE      (arg1=src_alpha, arg2=dst_alpha)
#   3 = TEXTURE_ID      (arg1=texture_id)
#   5 = MATERIAL        (arg1 / arg2 packed RGBA)

_NREL_XJ_MESH_FMT = "<7I"
_NREL_XJ_MESH_SIZE = 28
assert struct.calcsize(_NREL_XJ_MESH_FMT) == _NREL_XJ_MESH_SIZE

_VBUF_CONTAINER_FMT = "<4I"
_VBUF_CONTAINER_SIZE = 16

_IBUF_CONTAINER_FMT = "<5I"
_IBUF_CONTAINER_SIZE = 20

_RSARGS_FMT = "<4I"
_RSARGS_SIZE = 16


# Vertex layouts (per pso-blender/pso_blender/xj.py).  Each row is
# ``(stride, struct format for pos, struct format for normal,
#   uv-byte-offset-or-None, color-byte-offset-or-None)``.
# We bake position into the world matrix before storing; normals are
# not transformed (terrain is mostly flat-lit; v3 will fix this).
_VERTEX_LAYOUTS = {
    # format: (stride, has_normal, uv_offset, color_offset)
    # 0: 12 bytes — pos only
    0: (12, False, None, None),
    # 1: pos + ?
    # 2: pos + normal (24)
    2: (24, True, None, None),
    # 3: pos + normal + uv (32)
    3: (32, True, 24, None),
    # 4: pos + pad (16)
    4: (16, False, None, None),
    # 5: pos + pad + uv (24) — confirmed in city00
    5: (24, False, 16, None),
    # 6: pos + normal + pad (28)
    6: (28, True, None, None),
    # 7: pos + normal + rgba + uv (36)
    7: (36, True, 28, 24),
}


def _strip_to_triangles(strip: list[int]) -> list[int]:
    """Convert a triangle strip to a triangle list.

    Drops degenerate triangles (where two indices match).  Handles
    odd/even strip vertex winding.  Does NOT special-case the 0xFFFF
    restart sentinel — PSOBB's writer doesn't emit it.
    """
    out: list[int] = []
    if len(strip) < 3:
        return out
    for i in range(len(strip) - 2):
        a, b, c = strip[i], strip[i + 1], strip[i + 2]
        if a == b or b == c or a == c:
            continue
        if i % 2 == 0:
            out.extend((a, b, c))
        else:
            out.extend((a, c, b))
    return out


def extract_nrel_meshes(rel: RelFile):
    """Walk every static mesh tree in an n.rel and return all submeshes.

    Each ``XjMesh`` referenced by a ``MeshTree`` is decoded into a list
    of ``XjVertex`` plus a flat triangle index list.  Vertices are
    pre-transformed by the owning ``MeshTreeNode``'s local-to-world
    matrix so the renderer can place them directly in scene space.

    Returns
    -------
    list[formats.xj.XjMesh]
        Empty when the n.rel has no chunks / mesh trees, or when the
        XJ-buffer walk turns up nothing.  Each output mesh corresponds
        to one (vertex buffer, index buffer) pair from the source data.

    Notes
    -----
    Unlike NJ chunks, this format is straightforward: each
    IndexBufferContainer is one strip, each VertexBufferContainer is
    a flat array.  We resolve the texture id from the
    RenderStateArgs (state_type=3 → TEXTURE_ID) when present.
    """
    # Lazy-import to keep formats.rel importable before formats.xj has
    # finished loading.
    from formats.xj import XjVertex, XjMesh  # type: ignore

    if not is_n_rel(rel):
        raise RelParseError("extract_nrel_meshes: not an n.rel")

    out: list = []
    seen_root: set[int] = set()

    for chunk, tree, root_off in iter_nrel_mesh_root_offsets(rel):
        # Each chunk's coordinate frame.  CRITICAL: chunks carry a real
        # rotation (forest aancient01: ~half the 215 chunks have
        # rot_y == 16383 ≈ 90° BAMS), not just a translation.  The old code
        # applied ONLY the translation, so every rotated chunk landed
        # un-rotated — the floor folded into a thin degenerate slab and the
        # camera auto-fit could never frame it (the "empty plane / only the
        # sky shell shows" bug).  Build the full chunk world matrix (psov2
        # NinjaStage section bone: Y-rotation + position) and transform every
        # vertex through it.
        chunk_matrix = _chunk_world_matrix(
            (chunk.x, chunk.y, chunk.z),
            (chunk.rot_x, chunk.rot_y, chunk.rot_z),
        )

        # The MeshTreeNode at root_off is 52 bytes; the actual mesh
        # struct lives at its mesh_ptr field (offset 0x04).
        if root_off in seen_root:
            continue
        seen_root.add(root_off)
        if root_off + 52 > rel.data_size:
            continue
        try:
            ef, mesh_ptr, *_rest, child_ptr, next_ptr = struct.unpack_from(
                "<II3f3i3fII", rel.data, root_off)
        except struct.error:
            continue
        # Honour HIDE / SHAPE_SKIP — but we always render terrain.
        # Eval-flag is mostly 0x17 (UNIT_POS|UNIT_ANG|UNIT_SCL|BREAK)
        # which doesn't suppress drawing.

        if mesh_ptr == 0 or mesh_ptr + _NREL_XJ_MESH_SIZE > rel.data_size:
            continue
        try:
            (flags, vb_ptr, vb_count, ib_ptr, ib_count,
             aib_ptr, aib_count) = struct.unpack_from(
                _NREL_XJ_MESH_FMT, rel.data, mesh_ptr)
        except struct.error:
            continue
        if vb_ptr == 0 or vb_count == 0:
            continue

        # Decode every VertexBufferContainer.
        vbufs: list[list] = []  # list of list[XjVertex]
        for i in range(vb_count):
            cont_off = vb_ptr + i * _VBUF_CONTAINER_SIZE
            if cont_off + _VBUF_CONTAINER_SIZE > rel.data_size:
                break
            try:
                vfmt, vptr, vsize, vcount = struct.unpack_from(
                    _VBUF_CONTAINER_FMT, rel.data, cont_off)
            except struct.error:
                break
            verts = _decode_vertex_array(rel, vfmt, vptr, vsize, vcount,
                                         chunk_matrix)
            vbufs.append(verts)

        # Decode every IndexBufferContainer.
        if ib_ptr and ib_count:
            for i in range(ib_count):
                cont_off = ib_ptr + i * _IBUF_CONTAINER_SIZE
                if cont_off + _IBUF_CONTAINER_SIZE > rel.data_size:
                    break
                try:
                    (rs_ptr, rs_count, idx_ptr, idx_count, vb_idx
                     ) = struct.unpack_from(
                        _IBUF_CONTAINER_FMT, rel.data, cont_off)
                except struct.error:
                    break
                if vb_idx >= len(vbufs):
                    continue
                tex_id = _resolve_texture_id(rel, rs_ptr, rs_count)
                strip_indices = _read_index_buffer(rel, idx_ptr, idx_count)
                tri_indices = _strip_to_triangles(strip_indices)
                if not tri_indices:
                    continue
                _emit_mesh(out, XjMesh, XjVertex, vbufs[vb_idx], tri_indices, tex_id)

        # Optionally include alpha (transparent) buffers.
        if aib_ptr and aib_count:
            for i in range(aib_count):
                cont_off = aib_ptr + i * _IBUF_CONTAINER_SIZE
                if cont_off + _IBUF_CONTAINER_SIZE > rel.data_size:
                    break
                try:
                    (rs_ptr, rs_count, idx_ptr, idx_count, vb_idx
                     ) = struct.unpack_from(
                        _IBUF_CONTAINER_FMT, rel.data, cont_off)
                except struct.error:
                    break
                if vb_idx >= len(vbufs):
                    continue
                tex_id = _resolve_texture_id(rel, rs_ptr, rs_count)
                strip_indices = _read_index_buffer(rel, idx_ptr, idx_count)
                tri_indices = _strip_to_triangles(strip_indices)
                if not tri_indices:
                    continue
                _emit_mesh(out, XjMesh, XjVertex, vbufs[vb_idx], tri_indices, tex_id)

    return out


def _decode_vertex_array(
    rel: RelFile, vertex_format: int, vptr: int, vsize: int, vcount: int,
    chunk_matrix: tuple[float, ...],
) -> list:
    """Decode ``vcount`` vertices into a list of ``XjVertex`` records.

    Vertex positions are transformed by ``chunk_matrix`` (the chunk's full
    rotation + translation; see :func:`_chunk_world_matrix`) so the assembled
    scene appears at world coordinates with each visibility chunk correctly
    rotated.  Format mapping defaults to "pos only at offset 0" when the format
    id is unknown — better to show a flat mesh than nothing.
    """
    from formats.xj import XjVertex  # type: ignore

    layout = _VERTEX_LAYOUTS.get(vertex_format)
    out: list = []
    if vptr == 0 or vcount == 0 or vsize <= 0:
        return out
    if layout is None:
        # Unknown format — try to read just position, skip the rest of
        # the stride.
        layout = (vsize, False, None, None)
    stride, has_normal, uv_off, _color_off = layout

    for i in range(vcount):
        off = vptr + i * vsize
        if off + stride > rel.data_size:
            break
        try:
            x, y, z = struct.unpack_from("<fff", rel.data, off)
        except struct.error:
            break
        wx, wy, wz = _apply_matrix(chunk_matrix, x, y, z, translate=True)
        if has_normal:
            try:
                nx, ny, nz = struct.unpack_from("<fff", rel.data, off + 12)
                # Rotate the normal with the chunk (no translation).
                nx, ny, nz = _apply_matrix(
                    chunk_matrix, nx, ny, nz, translate=False)
            except struct.error:
                nx, ny, nz = 0.0, 1.0, 0.0
        else:
            nx, ny, nz = 0.0, 1.0, 0.0
        if uv_off is not None and off + uv_off + 8 <= rel.data_size:
            try:
                u, v = struct.unpack_from("<ff", rel.data, off + uv_off)
            except struct.error:
                u, v = 0.0, 0.0
        else:
            u, v = 0.0, 0.0
        out.append(XjVertex(
            pos=(wx, wy, wz),
            normal=(nx, ny, nz),
            uv=(u, v),
        ))
    return out


def _resolve_texture_id(rel: RelFile, rs_ptr: int, rs_count: int) -> Optional[int]:
    """Walk RenderStateArgs[] looking for a TEXTURE_ID (state_type=3) entry."""
    if rs_ptr == 0 or rs_count == 0:
        return None
    for i in range(rs_count):
        off = rs_ptr + i * _RSARGS_SIZE
        if off + _RSARGS_SIZE > rel.data_size:
            return None
        try:
            stype, arg1, _arg2, _unk = struct.unpack_from(
                _RSARGS_FMT, rel.data, off)
        except struct.error:
            return None
        if stype == 3:  # TEXTURE_ID
            return arg1
    return None


def _read_index_buffer(rel: RelFile, idx_ptr: int, idx_count: int) -> list[int]:
    """Read ``idx_count`` u16 indices from the data buffer."""
    if idx_ptr == 0 or idx_count == 0:
        return []
    end = idx_ptr + idx_count * 2
    if end > rel.data_size:
        return []
    try:
        return list(struct.unpack_from(f"<{idx_count}H", rel.data, idx_ptr))
    except struct.error:
        return []


def _emit_mesh(out: list, XjMesh_cls, XjVertex_cls,
               vert_pool: list, tri_indices: list[int],
               tex_id: Optional[int]) -> None:
    """Build one XjMesh from a slice of the vertex pool.

    Drops triangles whose indices fall outside the pool (defensive — a
    malformed file might emit an out-of-bounds index).
    """
    # Compute used-slot set so we can compact the vertex pool.
    used = sorted(set(tri_indices))
    if not used:
        return
    if max(used) >= len(vert_pool):
        # Filter to in-range triangles only.
        new_tris: list[int] = []
        n = len(vert_pool)
        for i in range(0, len(tri_indices), 3):
            a, b, c = tri_indices[i], tri_indices[i + 1], tri_indices[i + 2]
            if a < n and b < n and c < n:
                new_tris.extend((a, b, c))
        tri_indices = new_tris
        used = sorted(set(tri_indices))
        if not used:
            return
    remap = {old: new for new, old in enumerate(used)}
    new_verts = [vert_pool[i] for i in used]
    new_indices = [remap[i] for i in tri_indices]
    # Compute bounding sphere
    if new_verts:
        xs = [v.pos[0] for v in new_verts]
        ys = [v.pos[1] for v in new_verts]
        zs = [v.pos[2] for v in new_verts]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        cz = sum(zs) / len(zs)
        r = max(((px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2) ** 0.5
                for px, py, pz in zip(xs, ys, zs))
    else:
        cx = cy = cz = 0.0
        r = 0.0
    out.append(XjMesh_cls(
        vertices=new_verts,
        indices=new_indices,
        material_id=(tex_id if tex_id is not None else -1),
        bounding_sphere=(cx, cy, cz, r),
    ))


# ---------------------------------------------------------------------------
# r.rel payload reader: RrelHeader + RrelAnchor → scene-anchor list
# ---------------------------------------------------------------------------
#
# Format reverse-engineered 2026-04-25 by:
#   1. Comparing 5 r.rel files (forest 1/2, cave 1, mine 1, ruins 1).
#   2. Cross-referencing PsoBB.exe at the only "r.rel" string xref
#      (push str.r.rel @ 0x00805b1c) inside fcn.008059a8.  The function
#      walks a 40-byte struct array, copying a u16 anchor id at +0x02 of
#      each entry into a destination buffer (`add edx, 0x28` at 0x805bb9
#      confirms the 40-byte stride).
#   3. Validating field interpretations against the resulting record
#      values across 5 maps (positions are sane game-world coords with
#      reasonable bbox, types are unique 16-bit ids).
#
# Layout:
#
#   RrelHeader (24 bytes)
#     +0x00: u32 anchors_ptr     -> RrelAnchor[count]    (relocated)
#     +0x04: u32 reserved        (always 0)
#     +0x08: u32 count           # number of anchors
#     +0x0C..+0x17: zeros
#
#   RrelAnchor (40 bytes)
#     +0x00: u32 anchor_id_packed  (id<<0 | version<<16 == 0x0001<<16)
#     +0x04: f32 pos_x
#     +0x08: f32 pos_y           (always 0 in observed data)
#     +0x0C: f32 pos_z
#     +0x10: f32 rot_x_radians
#     +0x14: u32 rot_y_packed    (BAMS-style; 0x3FFF=90°, 0x8000=180°)
#     +0x18: u32 unknown_18      (always 0 — reserved)
#     +0x1C: u32 unknown_1c      (always 0 — reserved)
#     +0x20: f32 radius          # activation/draw radius (spawn-distance hint)
#     +0x24: u32 sub_record_ptr  -> nested config block (variable layout)
#
# We don't decode the nested sub-records (each starts with a 16-bit
# sub-type 0x14/0x16/0x22/... and varies in size).  Walking only the
# outer table gives us:
#
#   * The set of anchor ids — useful to surface in the Map Editor's
#     spawn picker (these are the legal teleport/spawn destinations).
#   * A scene bounding box — diff between min/max pos_x / pos_z gives
#     the area's playable footprint.  Useful as a fallback for the
#     fog-far-plane heuristic when the per-area hardcoded table is
#     unavailable, and for the Map Editor's "fit camera to bounds"
#     button.
#
# IMPORTANT — fog and lighting are NOT in r.rel:
#
#   PSOBB stores per-area fog parameters in a code-resident table at
#   PsoBB.exe:0x00a8d770 (`FogEntry**`, decoded by Blue-Burst-Patch-
#   Project/newmap/fog.cpp), and per-area sunlight at 0x00a9d4e4
#   (`LightEntry**`, see sunlight.cpp).  Both are populated at startup
#   by hardcoded initializers — there is no per-area data file driving
#   them.  The Map Editor's hardcoded category table in
#   model_viewer.js::_PSO_AREA_ENV (forest=green-blue, cave=dark, etc.)
#   is the closest equivalent and remains the source of truth for
#   render-time fog/lighting.

_RREL_HEADER_FMT = "<III"          # anchors_ptr, reserved, count
_RREL_HEADER_SIZE = 12             # only first 12 bytes matter

_RREL_ANCHOR_FMT = "<I3f f I 2I f I"
_RREL_ANCHOR_SIZE = 40
assert struct.calcsize(_RREL_ANCHOR_FMT) == _RREL_ANCHOR_SIZE, \
    f"RrelAnchor size mismatch: {struct.calcsize(_RREL_ANCHOR_FMT)}"

# Cap: real r.rel files ship 32–250 anchors; cap at 4096 for safety.
_RREL_MAX_ANCHORS = 4096


@dataclass
class RrelAnchor:
    """One anchor record from an r.rel.  See module-level docstring for layout."""
    anchor_id: int             # 16-bit, unique within file
    version: int               # always 1 in PSOBB.IO
    pos: tuple[float, float, float]
    rot_x: float
    rot_y_packed: int          # BAMS-style 0..0xFFFF
    radius: float
    sub_record_ptr: int        # absolute offset into rel.data, 0 = none


@dataclass
class RrelHeader:
    """Decoded r.rel payload header."""
    anchors_ptr: int
    count: int


@dataclass
class RrelSceneHints:
    """Aggregate hints derived from the anchor list.

    These are used by the Map Editor's environment-application path as
    a fallback when the per-category table doesn't apply (e.g. boss
    arenas) AND as a "fit camera to bounds" hint.
    """
    anchor_count: int
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    bbox_center: tuple[float, float, float]
    bbox_size: tuple[float, float, float]
    # Suggested fog far-plane: longest horizontal bbox dimension.
    suggested_fog_far: float


def read_rrel_header(rel: RelFile) -> RrelHeader:
    """Read the RrelHeader struct at ``rel.payload_offset``.

    Raises
    ------
    RelParseError
        If the payload doesn't match the r.rel shape (no anchors_ptr in
        the relocation table, OR count is out-of-range).
    """
    if not is_r_rel(rel):
        raise RelParseError("not an r.rel (sniffer rejected)")
    if rel.payload_offset + _RREL_HEADER_SIZE > rel.data_size:
        raise RelParseError("r.rel payload truncated")
    anchors_ptr, _reserved, count = struct.unpack_from(
        _RREL_HEADER_FMT, rel.data, rel.payload_offset)
    if count == 0 or count > _RREL_MAX_ANCHORS:
        raise RelParseError(f"r.rel anchor count out-of-range: {count}")
    if anchors_ptr == 0 or anchors_ptr >= rel.data_size:
        raise RelParseError(
            f"r.rel anchors_ptr out-of-range: 0x{anchors_ptr:x}")
    if anchors_ptr + count * _RREL_ANCHOR_SIZE > rel.data_size:
        raise RelParseError(
            f"r.rel anchor table overflows data section: "
            f"0x{anchors_ptr:x} + {count}*40 > 0x{rel.data_size:x}")
    return RrelHeader(anchors_ptr=anchors_ptr, count=count)


def read_rrel_anchors(rel: RelFile,
                      header: Optional[RrelHeader] = None,
                      ) -> list[RrelAnchor]:
    """Decode every anchor record.  Tolerant of mid-table truncation.

    Returns an empty list when the file isn't an r.rel.  Raises only
    when the header itself fails to decode — individual anchor decode
    errors are logged and skipped (defensive against malformed quests).
    """
    if not is_r_rel(rel):
        return []
    if header is None:
        try:
            header = read_rrel_header(rel)
        except RelParseError:
            return []
    out: list[RrelAnchor] = []
    base = header.anchors_ptr
    for i in range(header.count):
        off = base + i * _RREL_ANCHOR_SIZE
        if off + _RREL_ANCHOR_SIZE > rel.data_size:
            log.warning("r.rel anchor %d OOB; truncating walk", i)
            break
        try:
            (idpacked, x, y, z, rotx, roty, u18, u1c, radius, subptr
             ) = struct.unpack_from(_RREL_ANCHOR_FMT, rel.data, off)
        except struct.error:
            log.warning("r.rel anchor %d unpack failed", i)
            break
        anchor_id = idpacked & 0xFFFF
        version = (idpacked >> 16) & 0xFFFF
        # Defensive: drop records with absurd positions (likely
        # truncation).  PSOBB.IO maps fit comfortably in [-10000, 10000].
        if not (-1e6 < x < 1e6 and -1e6 < y < 1e6 and -1e6 < z < 1e6):
            log.warning("r.rel anchor %d position out-of-range "
                        "(%g, %g, %g); dropping", i, x, y, z)
            continue
        out.append(RrelAnchor(
            anchor_id=anchor_id,
            version=version,
            pos=(x, y, z),
            rot_x=rotx,
            rot_y_packed=roty,
            radius=radius,
            sub_record_ptr=subptr,
        ))
    return out


def derive_scene_hints(anchors: list[RrelAnchor]) -> Optional[RrelSceneHints]:
    """Compute bbox + fog-far suggestion from an anchor list.

    Returns None when the list is empty.  The fog far-plane heuristic
    is ``max(horizontal_bbox_dimension, 800)`` — 800 is a floor that
    matches the smallest hardcoded category (cave) so we never get a
    nonsensically tiny fog distance from a sparse anchor set.
    """
    if not anchors:
        return None
    xs = [a.pos[0] for a in anchors]
    ys = [a.pos[1] for a in anchors]
    zs = [a.pos[2] for a in anchors]
    bbmin = (min(xs), min(ys), min(zs))
    bbmax = (max(xs), max(ys), max(zs))
    center = ((bbmin[0] + bbmax[0]) * 0.5,
              (bbmin[1] + bbmax[1]) * 0.5,
              (bbmin[2] + bbmax[2]) * 0.5)
    size = (bbmax[0] - bbmin[0],
            bbmax[1] - bbmin[1],
            bbmax[2] - bbmin[2])
    horiz = max(size[0], size[2])
    # Floor of 800 keeps fog from collapsing on small-anchor maps; cap
    # at 4000 so a sparse outlier doesn't push fog beyond the category
    # default of any area.
    suggested_far = max(800.0, min(horiz * 1.2, 4000.0))
    return RrelSceneHints(
        anchor_count=len(anchors),
        bbox_min=bbmin,
        bbox_max=bbmax,
        bbox_center=center,
        bbox_size=size,
        suggested_fog_far=suggested_far,
    )


def parse_rrel_render_hints(buf: bytes) -> dict:
    """Top-level convenience: parse an r.rel buffer → render-hint dict.

    Returns a JSON-friendly dict shape.  The outer keys are stable and
    documented as the wire shape used by /api/map/<id>?floor=N (see
    server.py::api_map_get):

        {
          "ok": True,
          "anchors": [
            {"id": 20, "version": 1,
             "pos": [x, y, z], "rot_x": 0.0, "rot_y_packed": 0x3fff,
             "radius": 198.0, "sub_record_ptr": 376},
            ...
          ],
          "hints": {
            "anchor_count": 100,
            "bbox_min": [-475, 0, -1300],
            "bbox_max": [1850, 0, 1275],
            "bbox_center": [687.5, 0, -12.5],
            "bbox_size": [2325, 0, 2575],
            "suggested_fog_far": 3090.0
          }
        }

    On any parse error returns ``{"ok": False, "error": "<reason>"}``
    so the caller can fall back to category defaults without raising.
    """
    try:
        rel = parse_rel(buf)
    except RelParseError as e:
        return {"ok": False, "error": f"rel parse: {e}"}
    if not is_r_rel(rel):
        return {"ok": False, "error": "not an r.rel"}
    try:
        header = read_rrel_header(rel)
        anchors = read_rrel_anchors(rel, header)
    except RelParseError as e:
        return {"ok": False, "error": f"r.rel parse: {e}"}
    if not anchors:
        return {"ok": False, "error": "no valid anchors"}
    hints = derive_scene_hints(anchors)
    return {
        "ok": True,
        "anchor_count": header.count,
        "anchors": [
            {
                "id": a.anchor_id,
                "version": a.version,
                "pos": list(a.pos),
                "rot_x": a.rot_x,
                "rot_y_packed": a.rot_y_packed,
                "radius": a.radius,
                "sub_record_ptr": a.sub_record_ptr,
            }
            for a in anchors
        ],
        "hints": (
            None if hints is None else {
                "anchor_count": hints.anchor_count,
                "bbox_min": list(hints.bbox_min),
                "bbox_max": list(hints.bbox_max),
                "bbox_center": list(hints.bbox_center),
                "bbox_size": list(hints.bbox_size),
                "suggested_fog_far": hints.suggested_fog_far,
            }
        ),
    }


# ---------------------------------------------------------------------------
# r.rel sub_record block — anchor +0x24 nested record (Editor v4 RE).
# ---------------------------------------------------------------------------
#
# Each ``RrelAnchor`` may carry an optional sub_record_ptr at +0x24 that
# points at a variable-layout block of *additional* data.  Across all 156
# r.rel files in PSOBB.IO every present sub_record begins with a fixed
# 0x46-byte common header:
#
#   RrelSubRecordHeader (0x46 bytes — note: header alone, list payload follows)
#     +0x00: u16 sub_type          # 0x10, 0x12, 0x14, 0x15, 0x16, 0x17
#     +0x02: u16 reserved          # always 0
#     +0x04: u32 next_ptr          # offset of the *next* sub_record's tail;
#                                    used by the runtime to walk a flat list
#                                    of sub_records (NOT a linked-list head).
#                                    For our purposes we treat it as an
#                                    opaque "end-of-record" marker.
#     +0x08: f32[3] pos            # world-space position; in 100% of
#                                    observed records this matches the
#                                    parent anchor's ``pos`` exactly
#     +0x14: f32[3] rot_euler      # rotation; observed always (0,0,0)
#     +0x20: f32[3] scale          # observed (1.0, 1.0, 1.0) on 99.5% of
#                                    records (rare exceptions: 1.12, 0.98)
#     +0x2C: u32 unk1              # always 0 in our corpus
#     +0x30: u32 unk2              # always 0
#     +0x34: u32 flags             # 0x00042513 on regular records,
#                                    0x00010001 on a few sub_type=0x15
#     +0x38: u32 color1            # 0xFFFFFFFF (or 0xC4A5A000 on 0x15)
#     +0x3C: u32 color2            # 0xFF7F7F7F default
#     +0x40: u16 magic             # 0x0240 on regular records,
#                                    0x4000 on sub_type=0x15
#     +0x42: u16 list_a_count      # number of entries in first list
#     +0x44: u16 list_b_count      # number of entries in second list
#
# After 0x46, the body holds variable-length lists keyed by the counts.
# The exact list element shape is sub-type specific — observed widths
# range from 16-byte "render hint" structs to longer arrays of u16
# section indices.  We do NOT decode the body in this module: the
# ground truth is buried in PSOBB.exe's set_data / map_object.cpp
# consumer chain, which is partly virtualized through subclass-of-
# entity vtables and would require a full RE pass to characterise per
# sub_type.  Instead we expose:
#
#   * The sub_type byte so callers can categorise records.
#   * The 0x46-byte header so callers can correlate with the parent
#     anchor (position match) and flag rare divergent records.
#   * The total record bytes (header + body) computed from the next_ptr
#     delta, so callers can carve the body out for further analysis.
#
# IMPORTANT — earlier task briefs claimed sub_type values 0x14 / 0x16 /
# 0x22 represented teleporter / door / NPC records.  That is WRONG.
# Empirically across all 156 files we see 0x10, 0x12, 0x14, 0x15, 0x16,
# 0x17 only (no 0x22), AND the same anchor_id receives different
# sub_types in different files — meaning sub_type categorises the
# record SHAPE, not the anchor's role.  The actual "this is a
# teleporter" semantics live in the anchor_id itself (20-29 are
# teleporters; 800-887 are NPC slot markers; etc.) — confirmed by the
# r.rel string consumer at ``fcn.008059a8`` which only reads
# ``anchor_id`` from each record.
#
# RE notes consolidated in ``_reports/psobb_engine_table_RE.md``.

# Header struct format.  All u32 fields little-endian; 64-bit alignment
# is implicit (the on-disk struct is naturally aligned at 4 bytes).
_RREL_SUBREC_HEADER_FMT = "<HHI 3f 3f 3f IIIII HHH"
_RREL_SUBREC_HEADER_SIZE = struct.calcsize(_RREL_SUBREC_HEADER_FMT)
assert _RREL_SUBREC_HEADER_SIZE == 0x46, _RREL_SUBREC_HEADER_SIZE


# Known sub_type values, with empirical counts from the 156-file corpus.
# Callers can use this for sanity-checking decoded records.
RREL_SUBREC_KNOWN_TYPES: frozenset[int] = frozenset({
    0x10,  # 40 records  - rare; cave/forest variant
    0x12,  # 18 records  - very rare
    0x14,  # 2842 records - most common
    0x15,  # 29 records  - special; ep1 ultimate hint?
    0x16,  # 2819 records - second most common
    0x17,  # 36 records  - rare
})


@dataclass
class RrelSubRecord:
    """One sub_record header decoded from an r.rel anchor's +0x24 ptr.

    The body bytes (everything after the 0x46-byte header) are kept as
    an opaque blob.  Callers that need to interpret per sub_type can
    parse :attr:`body_raw` themselves; we don't inflict a wrong layout
    by pretending we understand every variant.
    """
    anchor_index: int               # which RrelAnchor produced this
    abs_offset: int                 # offset in rel.data of the header
    sub_type: int                   # u16 at +0x00
    reserved: int                   # u16 at +0x02 (observed always 0)
    next_ptr: int                   # u32 at +0x04
    pos: tuple[float, float, float]
    rot_euler: tuple[float, float, float]
    scale: tuple[float, float, float]
    unk1: int                       # u32 at +0x2C (always 0 in corpus)
    unk2: int                       # u32 at +0x30 (always 0)
    flags: int                      # u32 at +0x34 (0x00042513 typical)
    color1: int                     # u32 at +0x38
    color2: int                     # u32 at +0x3C
    magic: int                      # u16 at +0x40 (0x0240 typical)
    list_a_count: int               # u16 at +0x42
    list_b_count: int               # u16 at +0x44
    body_raw: bytes                 # opaque (header EXCLUDED)
    record_total_bytes: int         # header + body length

    @property
    def is_known_type(self) -> bool:
        """``True`` if the sub_type is one we've catalogued."""
        return self.sub_type in RREL_SUBREC_KNOWN_TYPES


def _decode_subrecord_header(buf: bytes, off: int):
    """Decode the fixed 0x46-byte header at ``off``.

    Returns a tuple matching the dataclass init args (without
    body_raw / record_total_bytes / anchor_index / abs_offset).
    """
    raw = struct.unpack_from(_RREL_SUBREC_HEADER_FMT, buf, off)
    (sub_type, reserved, next_ptr,
     px, py, pz,
     rx, ry, rz,
     sx, sy, sz,
     unk1, unk2, flags, color1, color2,
     magic, lac, lbc) = raw
    return (sub_type, reserved, next_ptr,
            (px, py, pz), (rx, ry, rz), (sx, sy, sz),
            unk1, unk2, flags, color1, color2,
            magic, lac, lbc)


def parse_subrecord(rel: RelFile, anchor_index: int, ptr: int) -> Optional[RrelSubRecord]:
    """Decode the sub_record at offset ``ptr`` inside ``rel.data``.

    Returns ``None`` when ``ptr == 0`` or the header would extend past
    the data section.  Raises :class:`RelParseError` if the header
    decode itself fails.

    The ``anchor_index`` is the position of the source anchor in the
    anchor list; useful for callers that want to cross-reference back
    to the parent ``RrelAnchor``.
    """
    if ptr == 0:
        return None
    if ptr + _RREL_SUBREC_HEADER_SIZE > rel.data_size:
        return None
    try:
        (sub_type, reserved, next_ptr,
         pos, rot_e, scale,
         unk1, unk2, flags, color1, color2,
         magic, lac, lbc) = _decode_subrecord_header(rel.data, ptr)
    except struct.error as e:
        raise RelParseError(f"sub_record header decode failed at 0x{ptr:x}: {e}")

    # Body length: walk to the next sub_record header by following
    # next_ptr.  next_ptr semantics observed: it points at a position
    # AFTER the current record's body — typically the start of the
    # parent anchor's NEXT sub_record block.  When next_ptr <= ptr or
    # next_ptr > data_size we fall back to "consume only the header"
    # so we never overread.
    body_start = ptr + _RREL_SUBREC_HEADER_SIZE
    if next_ptr > body_start and next_ptr <= rel.data_size:
        body_end = next_ptr
    else:
        body_end = body_start
    body_raw = rel.data[body_start:body_end]
    total = body_end - ptr

    return RrelSubRecord(
        anchor_index=anchor_index,
        abs_offset=ptr,
        sub_type=sub_type,
        reserved=reserved,
        next_ptr=next_ptr,
        pos=pos,
        rot_euler=rot_e,
        scale=scale,
        unk1=unk1,
        unk2=unk2,
        flags=flags,
        color1=color1,
        color2=color2,
        magic=magic,
        list_a_count=lac,
        list_b_count=lbc,
        body_raw=body_raw,
        record_total_bytes=total,
    )


def read_rrel_subrecords(rel: RelFile,
                         anchors: Optional[list] = None) -> list[RrelSubRecord]:
    """Walk every anchor's sub_record_ptr and collect parsed sub_records.

    Returns a list ordered by anchor index.  Anchors with no sub_record
    are silently skipped (their entry simply doesn't appear in the
    result).  Decode errors on individual records are logged and the
    record is omitted; the function does not raise on partial failure.
    """
    if anchors is None:
        anchors = read_rrel_anchors(rel)
    out: list[RrelSubRecord] = []
    for i, a in enumerate(anchors):
        if a.sub_record_ptr == 0:
            continue
        try:
            rec = parse_subrecord(rel, i, a.sub_record_ptr)
        except RelParseError as e:
            log.warning("r.rel sub_record %d (anchor=%d) decode failed: %s",
                        i, a.anchor_id, e)
            continue
        if rec is not None:
            out.append(rec)
    return out


def summarize_subrecord_types(records: Iterator[RrelSubRecord]) -> dict[int, int]:
    """Build a histogram of ``sub_type`` counts.

    Convenience for tests / reports.  Returns ``{sub_type: count}``.
    """
    out: dict[int, int] = {}
    for r in records:
        out[r.sub_type] = out.get(r.sub_type, 0) + 1
    return out


__all__ = [
    "RelFile",
    "RelParseError",
    "NrelChunk",
    "NrelMeshTree",
    "NrelHeader",
    "RrelAnchor",
    "RrelHeader",
    "RrelSceneHints",
    "RrelSubRecord",
    "RREL_SUBREC_KNOWN_TYPES",
    "TRAILER_SIZE",
    "NREL_FMT2_MAGIC",
    "parse_rel",
    "is_rel",
    "is_n_rel",
    "is_c_rel",
    "is_r_rel",
    "read_nrel_header",
    "read_nrel_chunks",
    "read_mesh_trees",
    "iter_nrel_mesh_root_offsets",
    "extract_nrel_mesh_root_offsets",
    "read_texture_names",
    "extract_nrel_meshes",
    "read_rrel_header",
    "read_rrel_anchors",
    "derive_scene_hints",
    "parse_rrel_render_hints",
    "parse_subrecord",
    "read_rrel_subrecords",
    "summarize_subrecord_types",
]
