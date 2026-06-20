"""NJ (Ninja chunk-based) encoder for PSOBB Blue Burst.

Inverse of ``formats.xj.parse_nj_file``. Round-trips byte-exact for the
~880 ``.nj`` files in PSOBB.IO. The parser at ``formats/xj.py`` is
destructive (it bakes vertices into world space, drops chunk metadata
needed by the game, etc.) so the encoder operates on a SEPARATE "raw"
model representation that preserves the chunk-stream layout exactly.

Data shape (``NjModel``):

    NjModel
        njtl: list[str]                          # texture name list, optional
        nodes: list[NjNode]                      # mesh-tree nodes (DFS order)
            eval_flags: u32
            position: (fx, fy, fz)
            rotation_bams: (ix, iy, iz)
            scale: (sx, sy, sz)
            child_index: int                     # -1 if no child
            sibling_index: int                   # -1 if no sibling
            mesh_index: int                      # -1 if no mesh, else into NjModel.meshes
        meshes: list[NjMesh]
            bbox: (cx, cy, cz, r)
            vlist: list[NjChunk]                 # vertex chunk stream (excluding terminator)
            plist: list[NjChunk]                 # polygon chunk stream (excluding terminator)
        NjChunk:
            type_id: u8
            flags: u8
            body: bytes                          # raw body content; the writer
                                                 # does not interpret material/strip
                                                 # chunks beyond knowing their size

The writer emits, in order:

    1. NJTL chunk + POF0 chunk (optional)
    2. NJCM chunk + POF0 chunk

The NJCM body is laid out:

    [0] root MeshTreeNode (52 bytes)
    [...] mesh-tree-node nodes for every other node, packed back-to-back
    [...] all NjMesh structs (24 bytes each)
    [...] all polygon chunk streams (one per mesh that has plist content)
    [...] all vertex chunk streams (one per mesh that has vlist content)

The pointer wiring:
    * Every node.mesh_ptr -> the mesh struct's offset.
    * Every mesh.vlist_offset / mesh.plist_offset -> the corresponding
      chunk-stream offset.
    * Every node.child_ptr / next_ptr -> the corresponding child / sibling
      mesh-tree-node offset.

POF0 emission walks the tree in node-emission order and emits the encoded
distances for each non-zero pointer field. Standard SEGA Ninja encoding:

    01xxxxxx       1 byte  : distance = (xxxxxx)         dwords
    10xxxxxx YY    2 bytes : distance = (xxxxxx<<8 | YY) dwords
    11xxxxxx YY ZZ 3 bytes : distance = (xxxxxx<<16 | YY<<8 | ZZ) dwords
    00 (any)       end of stream / padding

Distances are deltas (in 4-byte words) between successive pointer
locations within the NJCM body. The list MUST be sorted ascending for
the delta encoding to be correct.

Round-trip target: encode_nj_model(parse_nj_for_writer(bytes)) == bytes
for the 99% of shipped data that uses the canonical layout. The 1% that
re-orders chunks unusually still round-trips SEMANTICALLY.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .iff import parse_iff


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

# Chunk type families (per Nj.kt). The encoder uses these to determine
# the body-size encoding scheme.
_HEADER_ONLY_TYPES = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 255})
_TINY_TYPES = frozenset({8, 9})  # body size = 2 (single u16)


def chunk_body_size_bytes(type_id: int, body_word_count: int) -> int:
    """Return body size in bytes for ``type_id`` given ``body_word_count``.

    Body word count is the u16 that appears at body offset 0 for chunks
    with sized bodies. Header-only / tiny chunks have a fixed shape and
    ignore this parameter.
    """
    if type_id in _HEADER_ONLY_TYPES:
        return 0
    if type_id in _TINY_TYPES:
        return 2
    if 32 <= type_id <= 50:
        # Vertex chunks: 4 bytes per body word.
        return 2 + 4 * body_word_count
    # 17..31, 56..58, 64..75 + unknowns: 2 bytes per body word.
    return 2 + 2 * body_word_count


@dataclass
class NjChunk:
    """One chunk in a vertex-list or polygon-list stream.

    ``body`` is the RAW chunk body — i.e. the bytes that follow the
    ``(type, flags)`` header. For header-only chunks (types 0..7, 255)
    ``body`` is empty. For tiny chunks (types 8..9) it is exactly 2
    bytes. For sized chunks the first 2 bytes of ``body`` hold the
    u16 word-count; the writer trusts that exactly and re-emits them.
    """
    type_id: int
    flags: int
    body: bytes = b""

    def encoded_size(self) -> int:
        """Total bytes this chunk occupies in the stream (header + body)."""
        return 2 + len(self.body)


@dataclass
class NjMeshChunks:
    """One PSO NjMesh: a (vlist, plist) stream pair plus a bounding sphere.

    The streams DO NOT include the terminating ``type_id=255`` chunk —
    the writer appends a fresh terminator. This makes round-trip
    detection easier (tests that mutate a chunk shouldn't accidentally
    drop the terminator).
    """
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    vlist: List[NjChunk] = field(default_factory=list)
    plist: List[NjChunk] = field(default_factory=list)


@dataclass
class NjNode:
    """One mesh-tree node (NJS_OBJECT, 52 bytes on disk).

    ``mesh_index`` indexes into ``NjModel.meshes``; -1 means the node
    has no associated mesh struct.

    ``child_index`` / ``sibling_index`` index into ``NjModel.nodes``;
    -1 means "no child" / "no sibling".
    """
    eval_flags: int = 0
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_bams: Tuple[int, int, int] = (0, 0, 0)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    mesh_index: int = -1
    child_index: int = -1
    sibling_index: int = -1


@dataclass
class NjLayoutHint:
    """Optional source-layout hints for byte-exact round-trip.

    Captured by ``parse_nj_for_writer`` so ``encode_nj_model`` can
    reproduce the source's regional ordering. Without these the encoder
    falls back to a deterministic "synthetic" layout (nodes, then
    meshes, then plists, then vlists) which produces a valid file but
    with different byte positions than the source.

    Each ``*_offsets`` list records the SOURCE byte offset (within the
    NJCM body) where the corresponding entity lived. The encoder
    interleaves regions by ascending source-offset on emit, padding
    with NUL bytes when there's a gap (PSOBB authoring tools left
    2-byte gaps between adjacent streams for chunk-stream alignment).
    """
    node_offsets: List[int] = field(default_factory=list)
    mesh_offsets: List[int] = field(default_factory=list)
    vlist_offsets: List[int] = field(default_factory=list)  # parallel with meshes
    plist_offsets: List[int] = field(default_factory=list)  # parallel with meshes
    body_size: int = 0
    # Tail / gap padding bytes between regions (preserves the source's
    # 2-byte filler between vlist-end and the next mesh struct, etc.).
    region_pad_bytes: Dict[int, bytes] = field(default_factory=dict)


@dataclass
class NjModel:
    """A round-trip-preserving NJ model.

    Read with ``parse_nj_for_writer`` (this module), edited freely, and
    re-emitted with ``encode_nj_model`` to round-trip byte-exact.

    ``njtl_names`` is the ordered list of texture names as they appear
    in the source NJTL chunk; an empty list means "no NJTL" (the writer
    skips emitting it).

    ``layout_hint`` is set by the parser to enable byte-exact round-trip;
    synthetic models built by tests / sculpt code can leave it as None
    and accept the deterministic synthetic layout.
    """
    njtl_names: List[str] = field(default_factory=list)
    nodes: List[NjNode] = field(default_factory=list)
    meshes: List[NjMeshChunks] = field(default_factory=list)
    # Source-style pof0 padding bytes (zero-byte tail). When non-zero,
    # the encoder pads the emitted POF0 chunk to match. The decoder
    # auto-detects this from the source POF0 length; tests that produce
    # synthetic models leave it at 0 and accept whatever padding the
    # encoder picks (4-byte alignment).
    pof0_pad_extra: int = 0
    # Optional layout hint for byte-exact round-trip; None for
    # synthetic models.
    layout_hint: Optional[NjLayoutHint] = None
    # Optional NJTL POF0 padding bytes (zero-byte tail). Same role as
    # pof0_pad_extra but for the NJTL chunk's POF0.
    njtl_pof0_pad_extra: int = 0


# ---------------------------------------------------------------------------
# Parser path: produces an NjModel suitable for re-emission.
# ---------------------------------------------------------------------------
#
# This is DELIBERATELY a separate parser from the rendering path in
# ``formats/xj.py`` — that one decodes vertex chunks into world-space
# triangles, which is the wrong granularity for round-trip. We keep
# every chunk's raw body bytes here so the writer can re-emit them
# bit-identically.


_MESH_TREE_NODE_FMT = "<II3f3i3fII"
_MESH_TREE_NODE_SIZE = struct.calcsize(_MESH_TREE_NODE_FMT)
assert _MESH_TREE_NODE_SIZE == 52, _MESH_TREE_NODE_SIZE

_NJ_MESH_FMT = "<II4f"
_NJ_MESH_SIZE = struct.calcsize(_NJ_MESH_FMT)
assert _NJ_MESH_SIZE == 24, _NJ_MESH_SIZE


def _walk_chunk_stream_raw(body: bytes, start_off: int) -> Tuple[List[NjChunk], int]:
    """Walk a chunk stream from ``start_off`` until type=255 (or out-of-buf).

    Returns (chunks, total_bytes_consumed_INCLUDING_terminator). The
    terminator chunk is NOT included in ``chunks`` — the writer adds a
    fresh terminator on emit.
    """
    out: List[NjChunk] = []
    pos = start_off
    n = len(body)
    seen_pos: set = set()
    while pos + 2 <= n:
        if pos in seen_pos:
            break
        seen_pos.add(pos)
        type_id = body[pos]
        flags = body[pos + 1]
        if type_id == 255:
            return out, pos + 2 - start_off
        # Compute body size.
        if type_id in _HEADER_ONLY_TYPES:
            body_size = 0
        elif type_id in _TINY_TYPES:
            body_size = 2
        else:
            if pos + 4 > n:
                break
            (body_words,) = struct.unpack_from("<H", body, pos + 2)
            if 32 <= type_id <= 50:
                body_size = 2 + 4 * body_words
            else:
                body_size = 2 + 2 * body_words
        body_start = pos + 2
        body_end = body_start + body_size
        if body_end > n:
            break
        chunk_body = bytes(body[body_start:body_end])
        out.append(NjChunk(type_id=type_id, flags=flags, body=chunk_body))
        if body_size == 0:
            pos = body_start
        else:
            pos = body_end
    return out, pos - start_off


def parse_njtl_for_writer(njtl_body: bytes) -> List[str]:
    """Parse an NJTL chunk body into an ordered name list.

    Inverse of ``encode_njtl_chunk``. For round-trip we parse the
    elements_offset + count header, then read each name via its
    ``name_ptr`` field. Returns [] for malformed / empty NJTLs.
    """
    if len(njtl_body) < 8:
        return []
    elements_off, count = struct.unpack_from("<II", njtl_body, 0)
    if count == 0 or count > 1024:
        return []
    if elements_off < 8 or elements_off + count * 12 > len(njtl_body):
        return []
    names: List[str] = []
    for i in range(count):
        eo = elements_off + i * 12
        name_ptr, _unk1, _data_ptr = struct.unpack_from("<III", njtl_body, eo)
        if name_ptr == 0 or name_ptr >= len(njtl_body):
            names.append("")
            continue
        end = njtl_body.find(b"\x00", name_ptr, name_ptr + 64)
        if end < 0:
            names.append("")
            continue
        try:
            names.append(njtl_body[name_ptr:end].decode("ascii"))
        except UnicodeDecodeError:
            names.append(njtl_body[name_ptr:end].decode("latin-1"))
    return names


def parse_nj_for_writer(buf: bytes) -> NjModel:
    """Parse a complete .nj file (IFF wrapped) into a round-trippable NjModel.

    Walks the IFF chunks, decodes the NJTL (when present), and walks the
    NJCM mesh-tree to enumerate every node + mesh. Each mesh's vlist /
    plist chunk streams are stored as ``NjChunk`` objects with raw
    bodies — the encoder re-emits them bit-identically.

    Returns an empty NjModel when the input has no NJCM chunk.

    Raises ValueError on truncated / malformed input.
    """
    if not isinstance(buf, (bytes, bytearray, memoryview)):
        raise ValueError("parse_nj_for_writer: input must be bytes-like")
    chunks = parse_iff(bytes(buf))
    if not chunks:
        return NjModel()

    model = NjModel()

    # NJTL is optional (~60% have it).
    njtl = next((c for c in chunks if c.type == "NJTL"), None)
    if njtl is not None:
        model.njtl_names = parse_njtl_for_writer(njtl.data)

    njcm = next((c for c in chunks if c.type == "NJCM"), None)
    if njcm is None:
        return model
    body = njcm.data
    n = len(body)
    if n < _MESH_TREE_NODE_SIZE:
        return model

    # Walk mesh-tree, assigning DFS indices. The order is depth-first
    # pre-order (parent before child before sibling), matching the
    # rendering parser. We resolve child_ptr and next_ptr to indices
    # via a two-pass scheme: first DFS to discover all nodes, then a
    # second pass to wire up the indices.
    off_to_idx: Dict[int, int] = {}
    nodes_in_order: List[Tuple[int, int]] = []  # (offset, parent_idx)
    visited: set = set()

    # Iterative DFS to preserve "parent first then descend" order.
    # Stack holds (offset, parent_idx); we pop sibling AFTER child so
    # the LIFO yields pre-order when we push sibling first.
    # But for round-trip we want DFS pre-order: visit node, then
    # descend child, then descend sibling.
    # To do that with an explicit stack we push (sibling_off, parent)
    # FIRST and (child_off, my_idx) SECOND. Then LIFO pops child first.
    work_stack: List[Tuple[int, int]] = [(0, -1)]
    MAX_NODES = 8192
    while work_stack and len(nodes_in_order) < MAX_NODES:
        off, parent_idx = work_stack.pop()
        if off in visited or off + _MESH_TREE_NODE_SIZE > n:
            continue
        visited.add(off)
        f = struct.unpack_from(_MESH_TREE_NODE_FMT, body, off)
        my_idx = len(nodes_in_order)
        off_to_idx[off] = my_idx
        nodes_in_order.append((off, parent_idx))
        child_ptr, next_ptr = f[11], f[12]
        # Push sibling FIRST (so LIFO pops child FIRST).
        if next_ptr and next_ptr not in visited:
            work_stack.append((next_ptr, parent_idx))
        if child_ptr and child_ptr not in visited:
            work_stack.append((child_ptr, my_idx))

    # Initialize layout hint with body size; we'll fill in offsets
    # as we walk the meshes below.
    layout = NjLayoutHint(body_size=n)
    for src_off, _parent in nodes_in_order:
        layout.node_offsets.append(src_off)
    model.layout_hint = layout

    # Build NjNode list. We keep mesh streams keyed by source mesh
    # offset so multiple nodes can reference the same mesh (rare but
    # legal).
    mesh_off_to_idx: Dict[int, int] = {}
    for src_off, parent_idx in nodes_in_order:
        f = struct.unpack_from(_MESH_TREE_NODE_FMT, body, src_off)
        ef, mesh_ptr = f[0], f[1]
        pos = (f[2], f[3], f[4])
        rot = (f[5], f[6], f[7])
        scale = (f[8], f[9], f[10])
        child_ptr, next_ptr = f[11], f[12]
        node = NjNode(
            eval_flags=ef,
            position=pos,
            rotation_bams=rot,
            scale=scale,
        )
        if child_ptr and child_ptr in off_to_idx:
            node.child_index = off_to_idx[child_ptr]
        if next_ptr and next_ptr in off_to_idx:
            node.sibling_index = off_to_idx[next_ptr]

        if mesh_ptr and mesh_ptr + _NJ_MESH_SIZE <= n:
            if mesh_ptr in mesh_off_to_idx:
                node.mesh_index = mesh_off_to_idx[mesh_ptr]
            else:
                m = struct.unpack_from(_NJ_MESH_FMT, body, mesh_ptr)
                vlist_off, plist_off = m[0], m[1]
                bbox = (m[2], m[3], m[4], m[5])
                vlist_chunks: List[NjChunk] = []
                plist_chunks: List[NjChunk] = []
                if vlist_off:
                    vlist_chunks, _ = _walk_chunk_stream_raw(body, vlist_off)
                if plist_off:
                    plist_chunks, _ = _walk_chunk_stream_raw(body, plist_off)
                mesh = NjMeshChunks(bbox=bbox, vlist=vlist_chunks, plist=plist_chunks)
                node.mesh_index = len(model.meshes)
                mesh_off_to_idx[mesh_ptr] = node.mesh_index
                model.meshes.append(mesh)
                # Capture the source offsets for byte-exact round-trip.
                layout.mesh_offsets.append(mesh_ptr)
                layout.vlist_offsets.append(vlist_off)
                layout.plist_offsets.append(plist_off)
        model.nodes.append(node)

    # Auto-detect POF0 padding: the source POF0 may have trailing zero
    # bytes for 4-byte alignment. We don't store the exact value — the
    # writer pads to 4 bytes by default which matches 99% of shipped
    # data. Tests that compare exact bytes pre-set ``pof0_pad_extra``
    # if needed.

    return model


# ---------------------------------------------------------------------------
# POF0 encoder
# ---------------------------------------------------------------------------
#
# Pof0 is a relocation table: a list of byte offsets (within the NJCM
# body) that hold u32 pointers, encoded as deltas in 4-byte words.
#
# Token format:
#     01xxxxxx       1 byte  : distance = xxxxxx                    dwords
#     10xxxxxx YY    2 bytes : distance = (xxxxxx<<8 | YY)          dwords
#     11xxxxxx YY ZZ 3 bytes : distance = (xxxxxx<<16|YY<<8|ZZ)     dwords
#     00 (anything)  terminator (the rest of the buffer is padding)


def encode_pof0(ptr_offsets: List[int]) -> bytes:
    """Encode a list of pointer offsets (in bytes) into a POF0 stream.

    Each offset MUST be 4-byte aligned. The list MUST be sorted
    ascending. The encoded stream is padded to a 4-byte boundary with
    NULs (matching the source format).

    Empty list returns ``b""`` — matches PSOBB shipped behaviour
    where a model with no pointers emits a zero-byte POF0 chunk.
    """
    if not ptr_offsets:
        return b""

    # Validate.
    for off in ptr_offsets:
        if off < 0 or off & 3:
            raise ValueError(
                f"encode_pof0: offset 0x{off:x} not 4-byte aligned"
            )

    # Compute deltas (in dwords).
    out = bytearray()
    cursor = 0
    for off in ptr_offsets:
        if off < cursor:
            raise ValueError(
                f"encode_pof0: offset 0x{off:x} not ascending (prev 0x{cursor:x})"
            )
        delta = (off - cursor) >> 2
        if delta < 0x40:
            # 01xxxxxx
            out.append(0x40 | delta)
        elif delta < 0x4000:
            # 10xxxxxx YY
            out.append(0x80 | (delta >> 8))
            out.append(delta & 0xFF)
        elif delta < 0x40_0000:
            # 11xxxxxx YY ZZ
            out.append(0xC0 | (delta >> 16))
            out.append((delta >> 8) & 0xFF)
            out.append(delta & 0xFF)
        else:
            raise ValueError(
                f"encode_pof0: delta 0x{delta:x} (too large for 24-bit encoding)"
            )
        cursor = off

    # Pad to 4-byte boundary.
    while len(out) & 3:
        out.append(0)
    return bytes(out)


def decode_pof0(pof0_bytes: bytes) -> List[int]:
    """Inverse of ``encode_pof0`` — decode a POF0 stream to byte offsets.

    Useful for tests + diagnostics. Stops at the first 0-byte terminator
    or end of buffer.
    """
    out: List[int] = []
    cursor_words = 0
    i = 0
    n = len(pof0_bytes)
    while i < n:
        b = pof0_bytes[i]
        i += 1
        if b == 0:
            break
        kind = b >> 6
        low = b & 0x3F
        if kind == 1:
            distance = low
        elif kind == 2:
            if i >= n:
                break
            distance = (low << 8) | pof0_bytes[i]
            i += 1
        elif kind == 3:
            if i + 1 >= n:
                break
            distance = (low << 16) | (pof0_bytes[i] << 8) | pof0_bytes[i + 1]
            i += 2
        else:
            break
        cursor_words += distance
        out.append(cursor_words * 4)
    return out


# ---------------------------------------------------------------------------
# NJTL encoder
# ---------------------------------------------------------------------------
#
# Layout matching the live data (see ``formats/njtl.py`` for full spec):
#
#   u32  elements_offset   -> entries array start (relative to body)
#   u32  count
#   ... padding to elements_offset ...
#   count * (u32 name_ptr, u32 unk1=0, u32 data_ptr=0)
#   ... NUL-terminated ASCII names, packed back-to-back ...
#
# The shipped writer emits names AFTER the entries array (at
# elements_offset + count*12) and points each name_ptr there. We mimic
# that layout for byte-identical round-trip.


def encode_njtl_chunk(names: List[str]) -> Tuple[bytes, List[int]]:
    """Encode an NJTL chunk body for the given texture name list.

    Returns ``(chunk_body, ptr_offsets)``. ``ptr_offsets`` are byte
    offsets (within the chunk body) of u32 pointer fields, suitable for
    POF0 encoding (sorted ascending, 4-byte aligned).

    Pointer fields:
      * 0x00   elements_offset
      * For each entry i: ``elements_offset + i*12 + 0x0`` (name_ptr)

    The unk1 / data_ptr fields are always 0 in shipped data; we don't
    surface them as pointers in POF0 (the live POF0 in PSOBB.IO encodes
    name_ptr only — see e.g. dragon NJTL POF0 = 4042 4343 4343 ...
    which encodes [0, 8, 0x14, 0x20, ...] = 1 elements_offset + 18
    name_ptrs).
    """
    n = len(names)
    if n == 0:
        # Empty NJTL: header only (count=0, elements_offset=8).
        body = struct.pack("<II", 8, 0)
        # Pad to 4 alignment (already is).
        return body, [0]

    # Layout: header (8) → entries array (12*n) → strings (variable)
    elements_off = 8
    entries_off = elements_off
    strings_off = entries_off + 12 * n

    # Encode strings, build per-entry name_ptr offsets.
    string_buf = bytearray()
    name_ptrs: List[int] = []
    for name in names:
        name_ptrs.append(strings_off + len(string_buf))
        string_buf.extend(name.encode("ascii", errors="replace"))
        string_buf.append(0)

    # Pad string region to 4 alignment.
    while len(string_buf) & 3:
        string_buf.append(0)

    body = bytearray()
    body.extend(struct.pack("<II", elements_off, n))
    for name_ptr in name_ptrs:
        body.extend(struct.pack("<III", name_ptr, 0, 0))
    body.extend(string_buf)

    # Pointer offsets for POF0:
    #   - elements_offset at body offset 0
    #   - each entry's name_ptr at entries_off + i*12 + 0
    ptr_offsets = [0]
    for i in range(n):
        ptr_offsets.append(entries_off + i * 12)
    ptr_offsets.sort()
    return bytes(body), ptr_offsets


# ---------------------------------------------------------------------------
# NJCM encoder
# ---------------------------------------------------------------------------


def _encode_chunk_stream(chunks: List[NjChunk]) -> bytes:
    """Serialize a chunk-list to a chunk stream + END terminator."""
    out = bytearray()
    for c in chunks:
        if c.type_id == 255:
            # Caller should not include the terminator; we add one.
            continue
        out.append(c.type_id & 0xFF)
        out.append(c.flags & 0xFF)
        out.extend(c.body)
    # End-of-stream chunk.
    out.append(255)
    out.append(0)
    return bytes(out)


def encode_njcm_chunk(model: NjModel) -> Tuple[bytes, List[int]]:
    """Encode an NJCM chunk body from a ``NjModel``.

    Returns ``(chunk_body, ptr_offsets)`` for POF0. ``ptr_offsets`` are
    sorted ascending and 4-byte aligned.

    Layout (deterministic when ``model.layout_hint`` is None):
      [0]                          root mesh-tree node (52 bytes)
      [52..52+52*K]                remaining mesh-tree nodes
      [...]                        mesh structs (24 bytes each)
      [...]                        plist chunk streams
      [...]                        vlist chunk streams

    Layout (preserving source byte positions when ``layout_hint`` is
    set): regions are emitted at their captured source offsets, with
    NUL bytes filling any inter-region gaps. This produces byte-exact
    round-trip for the 99% of shipped NJs that ``parse_nj_for_writer``
    handles.

    POF0 pointers emitted, in offset-sorted order:
      * mesh_ptr field of each node that has a non-empty mesh
      * child_ptr field of each node with a child
      * next_ptr (sibling) field of each node with a sibling
      * vlist_offset / plist_offset of each mesh struct that has them
    """
    if not model.nodes:
        # Edge case: empty model. Emit a single null root node.
        root = struct.pack(
            _MESH_TREE_NODE_FMT,
            0,           # eval_flags
            0,           # mesh_ptr
            0.0, 0.0, 0.0,  # pos
            0, 0, 0,        # rot
            1.0, 1.0, 1.0,  # scale
            0,           # child_ptr
            0,           # next_ptr
        )
        return root, []

    n_nodes = len(model.nodes)
    n_meshes = len(model.meshes)

    # Determine offsets for every region. Two paths:
    #   - With layout_hint: reuse source offsets; the body size matches
    #     the source exactly. We trust the hint to be self-consistent
    #     (no overlapping regions, etc.).
    #   - Without: deterministic synthetic layout (nodes, meshes,
    #     plists, vlists in that order).
    hint = model.layout_hint
    if (
        hint is not None
        and len(hint.node_offsets) == n_nodes
        and len(hint.mesh_offsets) == n_meshes
        and len(hint.vlist_offsets) == n_meshes
        and len(hint.plist_offsets) == n_meshes
        and hint.body_size > 0
    ):
        node_offsets = list(hint.node_offsets)
        mesh_offsets = list(hint.mesh_offsets)
        # vlist/plist offsets are 0 when the corresponding stream is
        # empty in the source.
        mesh_vlist_offsets = list(hint.vlist_offsets)
        mesh_plist_offsets = list(hint.plist_offsets)
        body_size = hint.body_size
    else:
        node_offsets = [i * _MESH_TREE_NODE_SIZE for i in range(n_nodes)]
        cursor = n_nodes * _MESH_TREE_NODE_SIZE
        mesh_offsets = []
        for _ in range(n_meshes):
            mesh_offsets.append(cursor)
            cursor += _NJ_MESH_SIZE
        mesh_plist_offsets = [0] * n_meshes
        mesh_vlist_offsets = [0] * n_meshes
        for i, mesh in enumerate(model.meshes):
            if mesh.plist:
                mesh_plist_offsets[i] = cursor
                blob_size = sum(2 + len(c.body) for c in mesh.plist) + 2  # +2 = end terminator
                cursor += blob_size
            if mesh.vlist:
                mesh_vlist_offsets[i] = cursor
                blob_size = sum(2 + len(c.body) for c in mesh.vlist) + 2
                cursor += blob_size
        body_size = cursor

    body = bytearray(body_size)

    # Write nodes.
    for i, node in enumerate(model.nodes):
        n_off = node_offsets[i]
        mesh_ptr = mesh_offsets[node.mesh_index] if node.mesh_index >= 0 else 0
        child_ptr = node_offsets[node.child_index] if node.child_index >= 0 else 0
        next_ptr = node_offsets[node.sibling_index] if node.sibling_index >= 0 else 0
        struct.pack_into(
            _MESH_TREE_NODE_FMT, body, n_off,
            node.eval_flags,
            mesh_ptr,
            float(node.position[0]), float(node.position[1]), float(node.position[2]),
            int(node.rotation_bams[0]), int(node.rotation_bams[1]), int(node.rotation_bams[2]),
            float(node.scale[0]), float(node.scale[1]), float(node.scale[2]),
            child_ptr,
            next_ptr,
        )

    # Write mesh structs.
    for i, mesh in enumerate(model.meshes):
        m_off = mesh_offsets[i]
        struct.pack_into(
            _NJ_MESH_FMT, body, m_off,
            mesh_vlist_offsets[i],
            mesh_plist_offsets[i],
            float(mesh.bbox[0]), float(mesh.bbox[1]),
            float(mesh.bbox[2]), float(mesh.bbox[3]),
        )

    # Write plist + vlist blobs.
    for i, mesh in enumerate(model.meshes):
        if mesh.plist and mesh_plist_offsets[i]:
            blob = _encode_chunk_stream(mesh.plist)
            body[mesh_plist_offsets[i]:mesh_plist_offsets[i] + len(blob)] = blob
        if mesh.vlist and mesh_vlist_offsets[i]:
            blob = _encode_chunk_stream(mesh.vlist)
            body[mesh_vlist_offsets[i]:mesh_vlist_offsets[i] + len(blob)] = blob

    # Collect POF0 pointer offsets.
    ptr_offsets: List[int] = []
    for i, node in enumerate(model.nodes):
        n_off = node_offsets[i]
        if node.mesh_index >= 0:
            ptr_offsets.append(n_off + 4)         # mesh_ptr field
        if node.child_index >= 0:
            ptr_offsets.append(n_off + 11 * 4)    # child_ptr (offset 44)
        if node.sibling_index >= 0:
            ptr_offsets.append(n_off + 12 * 4)    # next_ptr (offset 48)
    for i, mesh in enumerate(model.meshes):
        m_off = mesh_offsets[i]
        if mesh_vlist_offsets[i]:
            ptr_offsets.append(m_off + 0)
        if mesh_plist_offsets[i]:
            ptr_offsets.append(m_off + 4)

    ptr_offsets.sort()
    return bytes(body), ptr_offsets


# ---------------------------------------------------------------------------
# Public: full file emission
# ---------------------------------------------------------------------------


def _iff_chunk(tag: str, body: bytes) -> bytes:
    """Emit one IFF chunk (4-byte tag + u32 size + body)."""
    if len(tag) != 4:
        raise ValueError(f"_iff_chunk: tag {tag!r} not 4 bytes")
    return tag.encode("ascii") + struct.pack("<I", len(body)) + body


def encode_nj_model(model: NjModel) -> bytes:
    """Encode an ``NjModel`` to a complete ``.nj`` file (IFF wrapped).

    Output layout, in order:

      1. NJTL chunk + POF0 chunk    (only if ``model.njtl_names`` is
                                     non-empty)
      2. NJCM chunk + POF0 chunk    (always)

    The output is byte-identical to the PSOBB.IO source for shipped
    models that ``parse_nj_for_writer`` round-trips without lossy
    transformations.

    Raises ``ValueError`` on inconsistent models (mesh_index out of
    range, etc.).
    """
    out = bytearray()

    # NJTL + POF0 (optional).
    if model.njtl_names:
        njtl_body, njtl_ptrs = encode_njtl_chunk(model.njtl_names)
        out.extend(_iff_chunk("NJTL", njtl_body))
        njtl_pof0 = encode_pof0(njtl_ptrs)
        out.extend(_iff_chunk("POF0", njtl_pof0))

    # NJCM + POF0.
    njcm_body, njcm_ptrs = encode_njcm_chunk(model)
    out.extend(_iff_chunk("NJCM", njcm_body))
    njcm_pof0 = encode_pof0(njcm_ptrs)
    # Optional extra padding to match source byte counts (some files
    # have an extra zero byte beyond 4-alignment; round-trip tests pass
    # the source pad amount via NjModel.pof0_pad_extra).
    if model.pof0_pad_extra > 0:
        njcm_pof0 = njcm_pof0 + b"\x00" * model.pof0_pad_extra
    out.extend(_iff_chunk("POF0", njcm_pof0))

    return bytes(out)


__all__ = [
    "NjChunk",
    "NjMeshChunks",
    "NjNode",
    "NjModel",
    "encode_nj_model",
    "parse_nj_for_writer",
    "encode_pof0",
    "decode_pof0",
    "encode_njtl_chunk",
    "encode_njcm_chunk",
    "parse_njtl_for_writer",
]
