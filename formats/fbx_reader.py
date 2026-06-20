"""Binary FBX reader → ``formats.import_external.ImportedModel``.

This module ships a self-contained pure-Python parser for binary FBX
files (versions 7.0 / 7.1 / 7.2 / 7.3 / 7.4 / 7.5 — i.e. all of the
"FBX 2010+" exports the modern DCC ecosystem produces). It deliberately
does NOT depend on Autodesk's proprietary FBX SDK; that SDK is GPL-
incompatible, requires Windows + a 32-bit interpreter, and isn't
embeddable in a single Python wheel.

It also doesn't depend on the third-party ``fbxloader`` PyPI package,
which has a bug where ``LayerElement*`` nodes are collapsed to their
single int property (the layer index) when their child block isn't
preceded by a NULL block sentinel — fbxloader's ``parse_node``
unconditionally sets ``singleProperty = numProperties == 1 and
reader.index == endOffset``, which is true for any leaf-or-not node
with one property.

Nor does it depend on ``pyufbx``, the Cython binding around ufbx; the
v0.0.7 wrapper exposes ``vertex_position.values.data`` cast as
``const float*`` while ufbx defaults to ``ufbx_real = double``, so
every vertex coordinate comes back as garbage (the byte layout is
misinterpreted). ``load_memory`` is also a stub in 0.0.7, only
``load_file`` works — and even that goes through the same buggy cast.

We parse the binary FBX format ourselves (well-documented, see
https://github.com/iscle/binary-fbx-spec for a clean spec) and project
into ``ImportedModel`` so the rest of the import pipeline is reused
verbatim.

Coordinate convention:
    FBX (Maya / Mixamo / 3ds Max default): right-handed, Y-up, +Z forward,
    cm units (UnitScaleFactor=1.0 → 1 unit = 1 cm).

    glTF (also right-handed, Y-up, +Z forward, m units).

    PSOBB: LEFT-handed, Y-up, -Z forward, cm-ish units depending on
    asset. The ``imported_to_nj`` converter handles the axis flip + scale
    knob; we keep FBX data verbatim in source coords here, the same way
    the glTF parser does.

Skin weight handling:
    FBX stores skin as ``Deformer (type=Skin)`` with ``SubDeformer
    (type=Cluster)`` children — ONE cluster per bone. Each cluster has
    a flat ``Indexes`` array (vertex indices it influences) plus a
    parallel ``Weights`` array (per-vertex weight in [0, 1]).

    We invert this to per-vertex (joints[4], weights[4]) the same way
    glTF JOINTS_0/WEIGHTS_0 do. Vertices with > 4 influences keep the
    4 largest; the dropped weight is renormalized into the kept slots.

Animation handling:
    ``AnimationStack > AnimationLayer > AnimationCurveNode > AnimationCurve``.
    Each AnimationCurveNode targets one bone+channel (T/R/S) and owns
    3 AnimationCurves (X/Y/Z). FBX stores keyframes as
    ``KeyTime`` (int64, in 1/46186158000-second units = "ktime" =
    "TimeMode" base) and ``KeyValueFloat`` (float32).

    We resample to a per-frame grid at the import_external rotation
    layer's ``fps_target`` (default 30 Hz). Times are converted to
    seconds. Rotation is FBX Euler angles in DEGREES (the FBX rotation
    order from each Model node), NOT quaternions — we compose to a
    quaternion at extract time so ImportedTrack consumers don't need to
    care about Euler decomposition.

Open issues (v3):
    * Blend shapes (Geometry.Shape, Deformer type=BlendShapeChannel) —
      not extracted; PSOBB's xj.py doesn't render morphs anyway.
    * Pre/post-rotation on bones — most FBX files leave these zero.
      Mixamo specifically does. We record them in the bone's bind_rot_quat
      composition but only the standard Lcl Rotation is meaningfully
      driven by animation curves.
    * NURBS / NurbsCurves / Subdivision surfaces — not extracted; we
      assume polygonal meshes (which is what every game engine ships).
    * ASCII FBX (the text-format variant used for diff-friendly assets
      pre-2010) — not parsed. We raise a clear error pointing at the
      DCC's "FBX 2010+ Binary" export option.

License audit:
    Pure Python, no bundled binaries, MIT-equivalent (this project's
    license). Does NOT include any code from Autodesk's FBX SDK,
    Blender's GPL FBX importer, or fbxloader/pyufbx.
"""
from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .import_external import (
    BlendShape,
    ImportedAnimation,
    ImportedBone,
    ImportedMesh,
    ImportedModel,
    ImportedModelWithAnims,
    ImportedTrack,
)


# ---------------------------------------------------------------------------
# Binary FBX parser — produces a tree of FbxRecord nodes
# ---------------------------------------------------------------------------


@dataclass
class FbxRecord:
    """One node in the FBX tree.

    Attributes
    ----------
    name:
        ASCII node name (e.g. ``"Geometry"``, ``"LayerElementNormal"``).
    props:
        Property values (heterogeneous: int, float, str/bytes, list).
        Array-typed properties (``i``/``l``/``f``/``d``/``b``/``c``)
        come back as Python lists; scalars come back as int/float/bool.
    children:
        Nested FbxRecord nodes.
    """
    name: str
    props: List = field(default_factory=list)
    children: List["FbxRecord"] = field(default_factory=list)

    def child(self, name: str) -> Optional["FbxRecord"]:
        """Return the first child with this name, or None."""
        for c in self.children:
            if c.name == name:
                return c
        return None

    def all_children(self, name: str) -> List["FbxRecord"]:
        """Return every child with this name."""
        return [c for c in self.children if c.name == name]


class FbxParseError(Exception):
    """Raised when the FBX byte stream is malformed."""


_FBX_HEADER_SIG = b"Kaydara FBX Binary  \x00"


def parse_binary_fbx(data: bytes) -> FbxRecord:
    """Parse a binary FBX file into an FbxRecord tree.

    Args
    ----
    data:
        The raw .fbx bytes. Must start with the standard 23-byte
        signature followed by the 4-byte version int (little-endian).

    Returns
    -------
    FbxRecord
        Synthetic root whose ``children`` are the top-level FBX nodes
        (FBXHeaderExtension, GlobalSettings, Documents, Definitions,
        Objects, Connections, Takes, ...).

    Raises
    ------
    FbxParseError
        On magic mismatch, truncated data, or unknown property type.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise FbxParseError("parse_binary_fbx: data must be bytes")
    if data[:21] != _FBX_HEADER_SIG:
        # Look for ASCII FBX as a quick differentiator.
        head = bytes(data[:64])
        if b"; FBX" in head or b"FBXHeaderExtension" in head:
            raise FbxParseError(
                "ASCII FBX is not supported. Re-export as 'FBX 2010+ Binary' "
                "from your DCC (Maya/3ds Max/Blender → Export → FBX → Type: Binary)."
            )
        raise FbxParseError(
            f"not a binary FBX (header signature mismatch): {head[:24]!r}"
        )
    if len(data) < 27 + 160:
        raise FbxParseError("FBX file too short to be valid")
    version = struct.unpack("<I", data[23:27])[0]
    is_v75 = version >= 7500

    # Element header layout: pre-7.5 = 3*uint32 + uint8; 7.5+ = 3*uint64 + uint8.
    hdr_size = 24 if is_v75 else 12
    sentinel_len = 25 if is_v75 else 13

    def read_property(p: int) -> Tuple[object, int]:
        """Decode one property at offset ``p``; return (value, next_offset)."""
        typ = data[p:p+1]
        p += 1
        if typ == b"Y":
            return struct.unpack("<h", data[p:p+2])[0], p + 2
        if typ == b"C":
            return bool(data[p]), p + 1
        if typ == b"I":
            return struct.unpack("<i", data[p:p+4])[0], p + 4
        if typ == b"F":
            return struct.unpack("<f", data[p:p+4])[0], p + 4
        if typ == b"D":
            return struct.unpack("<d", data[p:p+8])[0], p + 8
        if typ == b"L":
            return struct.unpack("<q", data[p:p+8])[0], p + 8
        if typ in (b"S", b"R"):
            ln = struct.unpack("<I", data[p:p+4])[0]
            return bytes(data[p+4:p+4+ln]), p + 4 + ln
        if typ in (b"i", b"l", b"f", b"d", b"b", b"c"):
            n, encoding, comp_len = struct.unpack("<3I", data[p:p+12])
            payload_start = p + 12
            payload = bytes(data[payload_start:payload_start+comp_len])
            if encoding == 1:
                payload = zlib.decompress(payload)
            elif encoding != 0:
                raise FbxParseError(
                    f"unknown FBX array encoding {encoding} for type {typ!r}"
                )
            fmt_char = {
                b"i": "i", b"l": "q", b"f": "f", b"d": "d",
                b"b": "B", b"c": "B",
            }[typ]
            elem_size = {b"i": 4, b"l": 8, b"f": 4, b"d": 8, b"b": 1, b"c": 1}[typ]
            need = n * elem_size
            if len(payload) < need:
                raise FbxParseError(
                    f"FBX array short read: type={typ!r} expected {need} bytes, "
                    f"got {len(payload)}"
                )
            vals = list(struct.unpack(f"<{n}{fmt_char}", payload[:need]))
            return vals, p + 12 + comp_len
        raise FbxParseError(f"unknown FBX property type code: {typ!r}")

    def parse_node(p: int) -> Tuple[Optional[FbxRecord], int]:
        """Parse one node at offset ``p``; return (FbxRecord-or-None, next_offset)."""
        if is_v75:
            end_off, num_props, _prop_list_len = struct.unpack("<3Q", data[p:p+24])
        else:
            end_off, num_props, _prop_list_len = struct.unpack("<3I", data[p:p+12])
        p += hdr_size
        if end_off == 0 and num_props == 0:
            # Sentinel / NULL terminator. Consume the name_len byte (== 0).
            return None, p + 1
        name_len = data[p]
        p += 1
        name = bytes(data[p:p+name_len]).decode("ascii", errors="replace")
        p += name_len
        props: List = []
        for _ in range(num_props):
            v, p = read_property(p)
            props.append(v)
        # Children: parse until we exhaust the range, breaking at NULL records.
        children: List[FbxRecord] = []
        while p < end_off - sentinel_len + 1:
            if p >= end_off:
                break
            child, p = parse_node(p)
            if child is None:
                break
            children.append(child)
        # Skip any trailing sentinel bytes within end_off.
        return FbxRecord(name, props, children), end_off

    pos = 27
    children: List[FbxRecord] = []
    # Stop before the 160-byte footer band.
    stop_at = len(data) - 160
    while pos < stop_at:
        node, pos = parse_node(pos)
        if node is None:
            break
        children.append(node)

    return FbxRecord("__root__", [], children)


# ---------------------------------------------------------------------------
# FBX → ImportedModel projection
# ---------------------------------------------------------------------------
#
# Strategy:
#   1. Build an id_map from int64 object id -> FbxRecord for everything
#      under Objects (Geometry, Model, NodeAttribute, Deformer,
#      AnimationCurveNode, AnimationCurve, ...).
#   2. Walk Connections to build child→parent edges with a
#      relationship label ("OO" or "OP <propname>").
#   3. For every Geometry record, find its parent Model and the model's
#      parent chain. Triangulate; pull normals/UVs per the layer
#      mapping/reference rules; pull skin clusters (if any) keyed by
#      bone Model id.
#   4. Walk all Model nodes whose attribute is "LimbNode" and order
#      them DFS from root → leaves; record bind pose from
#      Lcl Translation / PreRotation / Lcl Rotation / Lcl Scaling.
#   5. (Optional) Walk AnimationStack/Layer/CurveNode/Curve and produce
#      per-bone keyframe lists.
#
# This deliberately uses "model id == bone id" — FBX has both
# NodeAttribute (the LimbNode metadata) AND Model (the transform host);
# our bones key off the Model.

_KTIME_PER_SECOND = 46186158000  # FBX time-mode unit


def _get_first_array_prop(rec: Optional[FbxRecord]) -> Optional[List]:
    """Return ``rec.child(...)``'s first list-typed property, or None."""
    if rec is None:
        return None
    for c in rec.children:
        if c.props and isinstance(c.props[0], list):
            return c.props[0]
    return None


def _get_array_child(parent: FbxRecord, name: str) -> Optional[List]:
    c = parent.child(name)
    if c is None or not c.props or not isinstance(c.props[0], list):
        return None
    return c.props[0]


def _get_string_child(parent: FbxRecord, name: str) -> Optional[str]:
    c = parent.child(name)
    if c is None or not c.props:
        return None
    v = c.props[0]
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, str):
        return v
    return None


def _split_name_class(packed: object) -> Tuple[str, str]:
    """Split FBX's 'Name\\x00\\x01Class' packing into (name, class).

    FBX object names are stored as ``"<name>\\x00\\x01<class>"`` —
    e.g. ``"Cube\\x00\\x01Geometry"``. Older 6.x files pre-date this
    convention and just use the name directly; we handle both.
    """
    if isinstance(packed, bytes):
        packed = packed.decode("utf-8", errors="replace")
    if not isinstance(packed, str):
        return "", ""
    sep = "\x00\x01"
    if sep in packed:
        n, c = packed.split(sep, 1)
        return n, c
    return packed, ""


# ---------------------------------------------------------------------------
# Connections graph
# ---------------------------------------------------------------------------


@dataclass
class _Edge:
    src: int           # child object id
    dst: int           # parent object id
    rel: str           # "OO" or "OP"
    prop: str = ""     # for OP: target property name (e.g. "Lcl Translation")


def _build_connections(root: FbxRecord) -> Tuple[Dict[int, List[_Edge]], Dict[int, List[_Edge]]]:
    """Return (out_edges, in_edges) keyed by object id.

    out_edges[id] = edges where id is the child (src)
    in_edges[id]  = edges where id is the parent (dst)
    """
    out: Dict[int, List[_Edge]] = {}
    inn: Dict[int, List[_Edge]] = {}
    conns = root.child("Connections")
    if conns is None:
        return out, inn
    for c in conns.all_children("C"):
        if not c.props:
            continue
        rel = c.props[0]
        if isinstance(rel, bytes):
            rel = rel.decode("ascii", errors="replace")
        if rel not in ("OO", "OP"):
            continue
        if len(c.props) < 3:
            continue
        try:
            src = int(c.props[1])
            dst = int(c.props[2])
        except (TypeError, ValueError):
            continue
        prop = ""
        if rel == "OP" and len(c.props) > 3:
            v = c.props[3]
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="replace")
            if isinstance(v, str):
                prop = v
        e = _Edge(src=src, dst=dst, rel=rel, prop=prop)
        out.setdefault(src, []).append(e)
        inn.setdefault(dst, []).append(e)
    return out, inn


# ---------------------------------------------------------------------------
# Object id → record map
# ---------------------------------------------------------------------------


@dataclass
class _ObjEntry:
    rec: FbxRecord
    type_name: str       # "Geometry", "Model", "Deformer", "NodeAttribute", ...
    sub_type: str = ""   # "Mesh", "LimbNode", "Skin", "Cluster", ...
    name: str = ""


def _build_id_map(root: FbxRecord) -> Dict[int, _ObjEntry]:
    out: Dict[int, _ObjEntry] = {}
    objs = root.child("Objects")
    if objs is None:
        return out
    for rec in objs.children:
        if not rec.props:
            continue
        try:
            obj_id = int(rec.props[0])
        except (TypeError, ValueError):
            continue
        name, _cls = _split_name_class(rec.props[1] if len(rec.props) > 1 else "")
        sub_type = ""
        if len(rec.props) > 2:
            v = rec.props[2]
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="replace")
            if isinstance(v, str):
                sub_type = v
        out[obj_id] = _ObjEntry(
            rec=rec,
            type_name=rec.name,
            sub_type=sub_type,
            name=name,
        )
    return out


# ---------------------------------------------------------------------------
# Mesh extraction (geometry + normals + uvs)
# ---------------------------------------------------------------------------


def _triangulate_polygons(poly_idx: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert FBX PolygonVertexIndex to (triangles, corner_to_pv, corner_to_poly).

    Returns
    -------
    triangles:
        ``(T, 3)`` uint32 — vertex indices of each output triangle. Negative-
        flagged indices in the source are decoded (``-i-1``) before
        triangulating.
    corner_to_pv:
        ``(T*3,)`` int32 — for each output corner (in ``triangles``-row-major
        order), the source polygon-vertex slot it inherits from. Used to
        reindex per-polygon-vertex layer elements (normals / uvs / colors)
        onto the triangle corner stream.
    corner_to_poly:
        ``(T*3,)`` int32 — for each output corner, the source polygon
        index (0-based). Used by ByPolygon-mapped layer elements.
    """
    out_tris: List[Tuple[int, int, int]] = []
    corner_to_pv: List[int] = []
    corner_to_poly: List[int] = []
    poly: List[int] = []
    poly_pv: List[int] = []  # source slot for each poly-vertex
    poly_idx_counter = 0
    for pv, vi in enumerate(poly_idx):
        end = vi < 0
        if end:
            vi = -vi - 1
        poly.append(int(vi))
        poly_pv.append(pv)
        if end:
            n = len(poly)
            if n >= 3:
                # Fan triangulate (v0, v1, v2), (v0, v2, v3), ...
                for i in range(1, n - 1):
                    a, b, c = poly[0], poly[i], poly[i + 1]
                    if a == b or b == c or a == c:
                        continue
                    out_tris.append((a, b, c))
                    corner_to_pv.append(poly_pv[0])
                    corner_to_pv.append(poly_pv[i])
                    corner_to_pv.append(poly_pv[i + 1])
                    corner_to_poly.extend([poly_idx_counter] * 3)
            poly = []
            poly_pv = []
            poly_idx_counter += 1
    if not out_tris:
        return (
            np.zeros((0, 3), dtype=np.uint32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )
    return (
        np.asarray(out_tris, dtype=np.uint32),
        np.asarray(corner_to_pv, dtype=np.int32),
        np.asarray(corner_to_poly, dtype=np.int32),
    )


def _extract_layer_floats(
    layer: FbxRecord,
    data_name: str,
    index_name: Optional[str],
    n_components: int,
    n_vertices: int,
    poly_idx: List[int],
    triangles: np.ndarray,
    corner_to_pv: np.ndarray,
    corner_to_poly: np.ndarray,
) -> Optional[np.ndarray]:
    """Reindex an FBX LayerElementNormal/UV/etc. block to per-vertex.

    FBX layer elements come in flavors:
      MappingInformationType ∈
        ByVertex / ByVertice / ByControlPoint   - one value per source vertex
        ByPolygonVertex                         - one value per polygon-vertex
        ByPolygon                               - one value per source polygon
        AllSame                                 - one value for the whole mesh

      ReferenceInformationType ∈
        Direct          - data is the values
        IndexToDirect   - data is the values, look up via the Index array

    PSOBB's NJ writer takes per-vertex data; we re-emit one value per
    OUTPUT triangle vertex (so ImportedMesh.vertices has one normal/uv
    per row). The downstream ``imported_to_nj`` rebuilds the strip
    chunks, which only consumes per-vertex data (after the importer's
    own dedup step in OBJ — but glTF passes through directly).

    For ByVertex mapping we reuse the source vertex's value across all
    triangles touching it; for ByPolygonVertex / ByPolygon we expand to
    a per-triangle-vertex array (which means we must duplicate vertices
    when the same source vertex has different polygon-vertex normals —
    that's a mesh-split task we handle in ``_extract_mesh`` by ALWAYS
    routing per-polygon-vertex layer elements through a vertex-split
    pass).

    Returns
    -------
    ``(N, n_components)`` float32 array indexed parallel to the *output*
    vertex pool, or None when the layer can't be projected to per-vertex
    cleanly.

    Implementation note: this returns per-**polygon-vertex** data when
    mapping is ByPolygonVertex/ByPolygon — the caller is responsible for
    expanding the position array to match. For ByVertex the result is
    parallel to the source vertices.
    """
    mit = _get_string_child(layer, "MappingInformationType") or "ByVertex"
    rit = _get_string_child(layer, "ReferenceInformationType") or "Direct"
    data = _get_array_child(layer, data_name)
    if data is None:
        return None
    idx_arr: Optional[List[int]] = None
    if index_name and rit == "IndexToDirect":
        idx_arr = _get_array_child(layer, index_name)
    arr = np.asarray(data, dtype=np.float32)
    if arr.size % n_components != 0:
        return None
    arr = arr.reshape(-1, n_components)

    def lookup(slot: int) -> np.ndarray:
        if idx_arr is not None:
            slot = idx_arr[slot]
        if 0 <= slot < arr.shape[0]:
            return arr[slot]
        return np.zeros(n_components, dtype=np.float32)

    if mit in ("ByVertex", "ByVertice", "ByControlPoint"):
        # Per source vertex. Indexed by vertex id.
        if rit == "Direct":
            if arr.shape[0] >= n_vertices:
                return arr[:n_vertices].astype(np.float32, copy=True)
            # Pad if short.
            out = np.zeros((n_vertices, n_components), dtype=np.float32)
            out[:arr.shape[0]] = arr
            return out
        # IndexToDirect with ByVertex.
        if idx_arr is not None and len(idx_arr) >= n_vertices:
            out = np.zeros((n_vertices, n_components), dtype=np.float32)
            for v in range(n_vertices):
                out[v] = lookup(v)
            return out
        return None

    if mit == "ByPolygonVertex":
        # Per polygon-vertex slot (parallel to PolygonVertexIndex).
        # We return a per-output-triangle-corner array (size = 3*T).
        n_tri = int(triangles.shape[0])
        out = np.zeros((n_tri * 3, n_components), dtype=np.float32)
        for corner, src_slot in enumerate(corner_to_pv.tolist()):
            if src_slot < 0:
                continue
            out[corner] = lookup(src_slot)
        return out

    if mit == "ByPolygon":
        # Per polygon. corner_to_poly maps each output corner to its
        # source polygon id; just look up.
        n_tri = int(triangles.shape[0])
        out = np.zeros((n_tri * 3, n_components), dtype=np.float32)
        for corner, poly_id in enumerate(corner_to_poly.tolist()):
            if poly_id < 0:
                continue
            out[corner] = lookup(poly_id)
        return out

    if mit == "AllSame":
        n_tri = int(triangles.shape[0])
        out = np.zeros((n_tri * 3, n_components), dtype=np.float32)
        if arr.shape[0] > 0:
            val = lookup(0)
            out[:] = val
        return out

    return None


def _expand_to_split_vertices(
    src_positions: np.ndarray,
    src_skin_w: Optional[np.ndarray],
    src_skin_i: Optional[np.ndarray],
    triangles: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Duplicate positions so per-polygon-vertex attributes don't collide.

    When normals or UVs are stored per-polygon-vertex, the same source
    vertex can have different values on different faces — we must
    create a new vertex slot per (source_vertex, layer_value) tuple.

    The simplest robust approach: every triangle corner gets its own
    vertex slot (no dedup). This 3x the vertex count for triangles but
    ensures every per-pv layer element is correctly attached. Skin
    weights remain shared via the source vertex id.

    Returns (positions, indices, skin_weights, skin_indices) — all
    rebuilt for the split layout. ``indices`` is ``(T, 3)`` uint32 with
    sequential corner ids.
    """
    n_tri = int(triangles.shape[0])
    if n_tri == 0:
        return src_positions.copy(), triangles.copy(), src_skin_w, src_skin_i

    out_pos = np.zeros((n_tri * 3, 3), dtype=np.float32)
    out_idx = np.zeros((n_tri, 3), dtype=np.uint32)
    out_w: Optional[np.ndarray] = None
    out_i: Optional[np.ndarray] = None
    if src_skin_w is not None and src_skin_i is not None:
        out_w = np.zeros((n_tri * 3, 4), dtype=np.float32)
        out_i = np.zeros((n_tri * 3, 4), dtype=np.uint8)

    for ti in range(n_tri):
        for ci in range(3):
            slot = ti * 3 + ci
            v_src = int(triangles[ti, ci])
            if 0 <= v_src < src_positions.shape[0]:
                out_pos[slot] = src_positions[v_src]
                if out_w is not None and out_i is not None:
                    out_w[slot] = src_skin_w[v_src]
                    out_i[slot] = src_skin_i[v_src]
            out_idx[ti, ci] = slot

    return out_pos, out_idx, out_w, out_i


def _extract_mesh(
    geom_rec: FbxRecord,
    bone_id_to_idx: Dict[int, int],
    out_edges: Dict[int, List[_Edge]],
    in_edges: Dict[int, List[_Edge]],
    id_map: Dict[int, _ObjEntry],
) -> Optional[ImportedMesh]:
    """Extract one Geometry node into an ImportedMesh.

    Skin lookup: walks ``Geometry → Deformer(Skin) → SubDeformer(Cluster)
    → Model(LimbNode)`` to attribute each cluster's per-vertex weights
    to its bone index in ``bone_id_to_idx``.
    """
    name = _split_name_class(geom_rec.props[1] if len(geom_rec.props) > 1 else "")[0]
    geom_id = int(geom_rec.props[0])

    # ---- Positions ----
    vert_arr = _get_array_child(geom_rec, "Vertices")
    if vert_arr is None or len(vert_arr) % 3 != 0:
        return None
    positions = np.asarray(vert_arr, dtype=np.float32).reshape(-1, 3)
    n_v = positions.shape[0]

    poly_idx = _get_array_child(geom_rec, "PolygonVertexIndex") or []

    triangles, corner_to_pv, corner_to_poly = _triangulate_polygons(poly_idx)

    # ---- Skin ----
    src_skin_w: Optional[np.ndarray] = None
    src_skin_i: Optional[np.ndarray] = None
    # Find a Deformer connected to this geometry.
    skin_w_lists: List[List[Tuple[int, float]]] = [[] for _ in range(n_v)]
    for edge in in_edges.get(geom_id, []):  # geom is the dst
        deformer_id = edge.src
        deformer = id_map.get(deformer_id)
        if deformer is None or deformer.type_name != "Deformer":
            continue
        if deformer.sub_type != "Skin":
            continue
        # Find SubDeformer (Cluster) children.
        for cluster_edge in in_edges.get(deformer_id, []):
            cluster_id = cluster_edge.src
            cluster = id_map.get(cluster_id)
            if cluster is None or cluster.type_name not in ("Deformer", "SubDeformer"):
                continue
            if cluster.sub_type != "Cluster":
                continue
            # Bone = the Model this cluster connects to.
            bone_id = None
            for be in out_edges.get(cluster_id, []):
                tgt = id_map.get(be.dst)
                if tgt is not None and tgt.type_name == "Model" and tgt.sub_type == "LimbNode":
                    bone_id = be.dst
                    break
            if bone_id is None:
                continue
            bone_idx = bone_id_to_idx.get(bone_id)
            if bone_idx is None:
                continue
            indexes = _get_array_child(cluster.rec, "Indexes") or []
            weights = _get_array_child(cluster.rec, "Weights") or []
            for vi, w in zip(indexes, weights):
                if 0 <= vi < n_v and w > 0.0:
                    skin_w_lists[vi].append((int(bone_idx), float(w)))

    # Project per-vertex skin lists to (4 weights, 4 indices).
    has_skin = any(skin_w_lists)
    if has_skin:
        src_skin_w = np.zeros((n_v, 4), dtype=np.float32)
        src_skin_i = np.zeros((n_v, 4), dtype=np.uint8)
        for v in range(n_v):
            lst = skin_w_lists[v]
            if not lst:
                continue
            # Keep top 4 by weight.
            lst.sort(key=lambda t: -t[1])
            kept = lst[:4]
            total = sum(w for _, w in kept)
            if total > 0:
                # Renormalize to 1.0.
                for k, (bi, w) in enumerate(kept):
                    src_skin_w[v, k] = w / total
                    src_skin_i[v, k] = bi

    # ---- Decide split layout ----
    # If any layer element is ByPolygonVertex / ByPolygon, we must split
    # vertices per-polygon-vertex so attributes don't collide.
    le_normal = geom_rec.child("LayerElementNormal")
    le_uv = geom_rec.child("LayerElementUV")
    needs_split = False
    for layer in (le_normal, le_uv):
        if layer is None:
            continue
        mit = _get_string_child(layer, "MappingInformationType") or "ByVertex"
        if mit in ("ByPolygonVertex", "ByPolygon"):
            needs_split = True
            break

    if needs_split:
        out_pos, out_indices, out_skin_w, out_skin_i = _expand_to_split_vertices(
            positions, src_skin_w, src_skin_i, triangles
        )
        # Attribute extraction now produces per-corner (== per-output-vertex) arrays.
        normals = (
            _extract_layer_floats(le_normal, "Normals", "NormalsIndex", 3,
                                   n_v, poly_idx, triangles, corner_to_pv, corner_to_poly)
            if le_normal else None
        )
        uvs = (
            _extract_layer_floats(le_uv, "UV", "UVIndex", 2,
                                   n_v, poly_idx, triangles, corner_to_pv, corner_to_poly)
            if le_uv else None
        )
        # If the layer was actually ByVertex (only one of normal/uv had the per-pv mapping),
        # the result is parallel to source vertices, not split. We need to expand it.
        n_split = out_pos.shape[0]
        if normals is not None and normals.shape[0] != n_split:
            # ByVertex result; re-index by triangle corners.
            expanded = np.zeros((n_split, 3), dtype=np.float32)
            for ti in range(triangles.shape[0]):
                for ci in range(3):
                    v_src = int(triangles[ti, ci])
                    if 0 <= v_src < normals.shape[0]:
                        expanded[ti * 3 + ci] = normals[v_src]
            normals = expanded
        if uvs is not None and uvs.shape[0] != n_split:
            expanded = np.zeros((n_split, 2), dtype=np.float32)
            for ti in range(triangles.shape[0]):
                for ci in range(3):
                    v_src = int(triangles[ti, ci])
                    if 0 <= v_src < uvs.shape[0]:
                        expanded[ti * 3 + ci] = uvs[v_src]
            uvs = expanded
        positions = out_pos
        triangles_out = out_indices
        skin_w_out = out_skin_w
        skin_i_out = out_skin_i
    else:
        # All ByVertex (or no layers). Triangles index into source vertices directly.
        normals = (
            _extract_layer_floats(le_normal, "Normals", "NormalsIndex", 3,
                                   n_v, poly_idx, triangles, corner_to_pv, corner_to_poly)
            if le_normal else None
        )
        uvs = (
            _extract_layer_floats(le_uv, "UV", "UVIndex", 2,
                                   n_v, poly_idx, triangles, corner_to_pv, corner_to_poly)
            if le_uv else None
        )
        triangles_out = triangles
        skin_w_out = src_skin_w
        skin_i_out = src_skin_i

    # FBX UVs use OpenGL convention (V increases upward); our glTF parser
    # treats glTF UVs as already-correct for PSOBB. FBX matches glTF here
    # so no flip — but Mixamo's FBX exports DO have UVs that need V-flip
    # for some tools. We don't second-guess the source; the user can
    # toggle a UV-flip in the editor's import modal if needed.

    return ImportedMesh(
        name=name or "mesh",
        vertices=positions.astype(np.float32, copy=False),
        indices=np.asarray(triangles_out, dtype=np.uint32),
        uvs=uvs,
        normals=normals,
        skin_indices=skin_i_out,
        skin_weights=skin_w_out,
        material_id=0,
    )


# ---------------------------------------------------------------------------
# Skeleton extraction
# ---------------------------------------------------------------------------


def _euler_deg_zyx_to_quat(rx_deg: float, ry_deg: float, rz_deg: float) -> Tuple[float, float, float, float]:
    """Compose ZYX-Euler degrees → quaternion (qx, qy, qz, qw).

    FBX's default RotationOrder is "EULER_XYZ" (rx applied first), which
    means the composition is R = Rz @ Ry @ Rx. The quaternion math:
        qx_axis = (sin(rx/2), 0, 0, cos(rx/2))
        qy_axis = (0, sin(ry/2), 0, cos(ry/2))
        qz_axis = (0, 0, sin(rz/2), cos(rz/2))
        q = qz * qy * qx
    """
    rx = math.radians(rx_deg) * 0.5
    ry = math.radians(ry_deg) * 0.5
    rz = math.radians(rz_deg) * 0.5
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    # qz * qy * qx — quaternion Hamilton product.
    # qy*qx:
    yx_w = cy * cx
    yx_x = cy * sx
    yx_y = sy * cx
    yx_z = -sy * sx
    # qz * (qy*qx):
    qw = cz * yx_w - sz * yx_z
    qx = cz * yx_x - sz * yx_y
    qy = cz * yx_y + sz * yx_x
    qz = cz * yx_z + sz * yx_w
    return (qx, qy, qz, qw)


def _read_property70(prop70: Optional[FbxRecord], name: str, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Read a Vector-typed P[name] entry from a Properties70 block."""
    if prop70 is None:
        return default
    for c in prop70.all_children("P"):
        if not c.props or len(c.props) < 5:
            continue
        n = c.props[0]
        if isinstance(n, bytes):
            n = n.decode("utf-8", errors="replace")
        if n != name:
            continue
        # Vector3D / Lcl Translation / etc. → props[4..6] are doubles.
        try:
            return (float(c.props[4]), float(c.props[5]), float(c.props[6]))
        except (IndexError, TypeError, ValueError):
            return default
    return default


def _extract_bones(
    id_map: Dict[int, _ObjEntry],
    out_edges: Dict[int, List[_Edge]],
    in_edges: Dict[int, List[_Edge]],
) -> Tuple[List[ImportedBone], Dict[int, int]]:
    """Return (bones, model_id_to_bone_idx).

    Walks the Model nodes whose sub_type is LimbNode (or those linked
    to a NodeAttribute LimbNode) and orders them DFS from the root.

    Each bone's bind pose is read from its own Properties70:
        Lcl Translation / PreRotation / Lcl Rotation / Lcl Scaling.
    """
    # Identify all bone Models.
    bone_ids: List[int] = []
    for obj_id, e in id_map.items():
        if e.type_name != "Model":
            continue
        if e.sub_type == "LimbNode" or e.sub_type == "Limb":
            bone_ids.append(obj_id)
            continue
        # Some FBX files mark the model "Null" but link to a NodeAttribute
        # of type LimbNode. Walk the connections to detect those.
        for ie in in_edges.get(obj_id, []):
            attr = id_map.get(ie.src)
            if attr is None or attr.type_name != "NodeAttribute":
                continue
            if attr.sub_type == "LimbNode":
                bone_ids.append(obj_id)
                break

    # Build child -> parent map among bones.
    bone_id_set = set(bone_ids)
    parent_of: Dict[int, int] = {}  # bone_id -> parent_bone_id (or -1)
    for bone_id in bone_ids:
        parent = -1
        # OO connection: bone Model -> parent Model.
        for e in out_edges.get(bone_id, []):
            if e.rel != "OO":
                continue
            if e.dst in bone_id_set:
                parent = e.dst
                break
        parent_of[bone_id] = parent

    # DFS from roots.
    roots = [b for b in bone_ids if parent_of[b] == -1]
    children_of: Dict[int, List[int]] = {b: [] for b in bone_ids}
    for b in bone_ids:
        p = parent_of[b]
        if p != -1:
            children_of.setdefault(p, []).append(b)

    ordered: List[int] = []
    visited: set = set()

    def _dfs(bid: int) -> None:
        if bid in visited:
            return
        visited.add(bid)
        ordered.append(bid)
        for c in children_of.get(bid, []):
            _dfs(c)

    for r in roots:
        _dfs(r)
    # Cover orphans.
    for b in bone_ids:
        if b not in visited:
            _dfs(b)

    # Build ImportedBone list.
    bones: List[ImportedBone] = []
    bone_id_to_idx: Dict[int, int] = {}
    for new_idx, bone_id in enumerate(ordered):
        bone_id_to_idx[bone_id] = new_idx

    for new_idx, bone_id in enumerate(ordered):
        rec = id_map[bone_id].rec
        prop70 = rec.child("Properties70")
        # Read defaults
        translation = _read_property70(prop70, "Lcl Translation", (0.0, 0.0, 0.0))
        rotation = _read_property70(prop70, "Lcl Rotation", (0.0, 0.0, 0.0))
        pre_rot = _read_property70(prop70, "PreRotation", (0.0, 0.0, 0.0))
        scaling = _read_property70(prop70, "Lcl Scaling", (1.0, 1.0, 1.0))

        # Compose bind rotation: PreRotation * Lcl Rotation (FBX standard).
        # Both come as Euler degrees XYZ.
        q_rot = _euler_deg_zyx_to_quat(*rotation)
        if pre_rot != (0.0, 0.0, 0.0):
            q_pre = _euler_deg_zyx_to_quat(*pre_rot)
            # Quaternion multiply: result = q_pre * q_rot
            q_rot = _quat_mul(q_pre, q_rot)

        parent_idx = -1
        p = parent_of[bone_id]
        if p != -1:
            parent_idx = bone_id_to_idx[p]

        bones.append(ImportedBone(
            name=id_map[bone_id].name or f"bone_{new_idx}",
            parent_idx=parent_idx,
            bind_pos=translation,
            bind_rot_quat=q_rot,
            bind_scale=scaling,
        ))

    return bones, bone_id_to_idx


def _quat_mul(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Hamilton quaternion product: result = a * b (apply b first, then a)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


# ---------------------------------------------------------------------------
# Animation extraction
# ---------------------------------------------------------------------------


def _extract_animations(
    id_map: Dict[int, _ObjEntry],
    out_edges: Dict[int, List[_Edge]],
    in_edges: Dict[int, List[_Edge]],
    bone_id_to_idx: Dict[int, int],
    bones: List[ImportedBone],
) -> List[ImportedAnimation]:
    """Walk AnimationStack → AnimationLayer → AnimationCurveNode → AnimationCurve.

    Each AnimationCurveNode targets a Model (bone) via OP connection
    (prop = 'Lcl Translation' / 'Lcl Rotation' / 'Lcl Scaling'). Each
    curve node owns 3 AnimationCurves (X/Y/Z) connected via OP with
    prop = 'd|X' / 'd|Y' / 'd|Z'.

    We pull every keyframe from every curve, union the per-frame times
    across X/Y/Z, sample missing components, and emit one ImportedTrack
    per (bone, channel).

    Rotation: FBX stores Euler **degrees** (XYZ order by default). We
    convert the per-frame Euler → quaternion at extract time so the
    track values match glTF's quaternion convention.
    """
    out: List[ImportedAnimation] = []

    stacks = [e for e in id_map.values() if e.type_name == "AnimationStack"]
    for stack in stacks:
        stack_id = next(k for k, v in id_map.items() if v is stack)
        # Layers connected as "child of stack" (rel OO).
        layer_ids: List[int] = []
        for ie in in_edges.get(stack_id, []):
            l = id_map.get(ie.src)
            if l is not None and l.type_name == "AnimationLayer":
                layer_ids.append(ie.src)
        if not layer_ids:
            continue

        tracks: List[ImportedTrack] = []
        max_t = 0.0

        for layer_id in layer_ids:
            # Curve nodes connected as children of layer.
            for ie in in_edges.get(layer_id, []):
                cnode = id_map.get(ie.src)
                if cnode is None or cnode.type_name != "AnimationCurveNode":
                    continue
                cnode_id = ie.src
                # Channel = the target property of OP edges from this node.
                # Find the target Model and channel name.
                target_bone: Optional[int] = None
                channel: str = ""
                for oe in out_edges.get(cnode_id, []):
                    if oe.rel != "OP":
                        continue
                    bone_idx = bone_id_to_idx.get(oe.dst)
                    if bone_idx is None:
                        continue
                    target_bone = bone_idx
                    p = oe.prop
                    if p in ("Lcl Translation", "Translation", "T"):
                        channel = "translation"
                    elif p in ("Lcl Rotation", "Rotation", "R"):
                        channel = "rotation"
                    elif p in ("Lcl Scaling", "Scaling", "S"):
                        channel = "scale"
                    break
                if target_bone is None or not channel:
                    continue

                # Find the X/Y/Z curves for this curve node.
                comps: Dict[str, Tuple[List[int], List[float]]] = {}
                for ce in in_edges.get(cnode_id, []):
                    if ce.rel != "OP":
                        continue
                    curve = id_map.get(ce.src)
                    if curve is None or curve.type_name != "AnimationCurve":
                        continue
                    times = _get_array_child(curve.rec, "KeyTime")
                    values = _get_array_child(curve.rec, "KeyValueFloat")
                    if not times or not values:
                        continue
                    comp = ce.prop
                    if comp.endswith("X"):
                        comps["X"] = (list(times), list(values))
                    elif comp.endswith("Y"):
                        comps["Y"] = (list(times), list(values))
                    elif comp.endswith("Z"):
                        comps["Z"] = (list(times), list(values))

                if not comps:
                    continue

                # Build a unified time grid (union of all component times).
                all_times: set = set()
                for tl, _vl in comps.values():
                    for t in tl:
                        all_times.add(int(t))
                if not all_times:
                    continue
                ordered_times = sorted(all_times)

                # Sample each component at each unified time (linear interp).
                def sample(times: List[int], values: List[float], t: int) -> float:
                    if not times:
                        return 0.0
                    if t <= times[0]:
                        return float(values[0])
                    if t >= times[-1]:
                        return float(values[-1])
                    # Binary search.
                    lo, hi = 0, len(times) - 1
                    while hi - lo > 1:
                        mid = (lo + hi) // 2
                        if times[mid] <= t:
                            lo = mid
                        else:
                            hi = mid
                    t0, t1 = times[lo], times[hi]
                    v0, v1 = float(values[lo]), float(values[hi])
                    a = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
                    return v0 + a * (v1 - v0)

                track_times: List[float] = []
                track_values: List[tuple] = []
                for ktime in ordered_times:
                    seconds = ktime / _KTIME_PER_SECOND
                    track_times.append(seconds)
                    if seconds > max_t:
                        max_t = seconds
                    if channel == "rotation":
                        rx = sample(comps.get("X", ([], []))[0], comps.get("X", ([], []))[1], ktime)
                        ry = sample(comps.get("Y", ([], []))[0], comps.get("Y", ([], []))[1], ktime)
                        rz = sample(comps.get("Z", ([], []))[0], comps.get("Z", ([], []))[1], ktime)
                        # FBX rotation default order is XYZ (rx applied first).
                        # Convert to quaternion.
                        q = _euler_deg_zyx_to_quat(rx, ry, rz)
                        track_values.append((q[0], q[1], q[2], q[3]))
                    else:
                        tx = sample(comps.get("X", ([], []))[0], comps.get("X", ([], []))[1], ktime)
                        ty = sample(comps.get("Y", ([], []))[0], comps.get("Y", ([], []))[1], ktime)
                        tz = sample(comps.get("Z", ([], []))[0], comps.get("Z", ([], []))[1], ktime)
                        track_values.append((tx, ty, tz))

                tracks.append(ImportedTrack(
                    bone_idx=target_bone,
                    channel=channel,
                    times=track_times,
                    values=track_values,
                    interp="LINEAR",
                ))

        out.append(ImportedAnimation(
            name=stack.name or "animation",
            duration_seconds=max_t,
            fps_target=30,
            tracks=tracks,
        ))

    return out


# ---------------------------------------------------------------------------
# Blend-shape (morph target) extraction
# ---------------------------------------------------------------------------
#
# FBX blend shapes follow a Deformer/SubDeformer/Shape chain rooted at
# the Geometry (Mesh):
#
#       Geometry (Mesh)
#           ↑ (OO)
#       Deformer    sub_type = "BlendShape"
#           ↑ (OO)
#       Deformer    sub_type = "BlendShapeChannel"   (one per channel)
#           ↑ (OO)
#       Geometry    sub_type = "Shape"
#               Indexes  : sparse vertex indices
#               Vertices : per-index x/y/z deltas
#               Normals  : optional per-index dx/dy/dz deltas
#
# Some FBX 6.x and a few 7.x exporters inline ``Shape`` blocks directly
# under the Geometry node. We support both layouts: walk the connection
# graph first, then sweep up any inline Shape children that weren't
# already absorbed.
#
# The BlendShapeChannel's name is the user-visible morph name (e.g.
# ``"Smile"``). Its ``DeformPercent`` is the default weight (0..100 in
# FBX, normalized to 0..1 here).


def _read_property70_scalar(prop70: Optional[FbxRecord], name: str, default: float) -> float:
    """Read a scalar P[name] entry from a Properties70 block.

    FBX scalar properties have ``props[4]`` as the value (after type tags
    in 0..3). Returns ``default`` if the property is missing or has no
    parseable value.
    """
    if prop70 is None:
        return default
    for c in prop70.all_children("P"):
        if not c.props or len(c.props) < 5:
            continue
        n = c.props[0]
        if isinstance(n, bytes):
            n = n.decode("utf-8", errors="replace")
        if n != name:
            continue
        try:
            return float(c.props[4])
        except (IndexError, TypeError, ValueError):
            return default
    return default


def _extract_one_shape(
    shape_rec: FbxRecord,
    channel_name: str,
    default_weight: float,
    mesh_name: str,
) -> Optional[BlendShape]:
    """Decode one FBX ``Geometry`` (sub_type=Shape) record into a BlendShape.

    The Shape record's ``Indexes`` array gives the sparse list of source-
    mesh vertex indices the shape moves; ``Vertices`` holds the per-index
    XYZ deltas. ``Normals`` (optional) holds the matching normal deltas.

    Returns ``None`` if either the indexes or vertices array is missing —
    a nameless / empty shape isn't worth a BlendShape entry.
    """
    indexes_raw = _get_array_child(shape_rec, "Indexes")
    verts_raw = _get_array_child(shape_rec, "Vertices")
    if indexes_raw is None or verts_raw is None:
        return None
    if len(verts_raw) % 3 != 0:
        return None
    n_idx = len(indexes_raw)
    n_offsets = len(verts_raw) // 3
    if n_idx == 0 or n_offsets == 0 or n_idx != n_offsets:
        return None
    idx_arr = np.asarray(indexes_raw, dtype=np.int32)
    off_arr = np.asarray(verts_raw, dtype=np.float32).reshape(-1, 3)
    normals_raw = _get_array_child(shape_rec, "Normals")
    normals_arr: Optional[np.ndarray] = None
    if normals_raw is not None and len(normals_raw) == 3 * n_idx:
        normals_arr = np.asarray(normals_raw, dtype=np.float32).reshape(-1, 3)

    # Shape name preference: the connecting BlendShapeChannel's name (the
    # user-visible morph label in DCCs) over the bare Shape's name (often
    # generated like "Shape.001").
    name = channel_name
    if not name and shape_rec.props and len(shape_rec.props) > 1:
        name = _split_name_class(shape_rec.props[1])[0]

    return BlendShape(
        name=name,
        indexes=idx_arr,
        offsets=off_arr,
        normals=normals_arr,
        default_weight=float(default_weight),
        mesh_name=mesh_name,
    )


def _extract_blend_shapes(
    geom_rec: FbxRecord,
    out_edges: Dict[int, List[_Edge]],
    in_edges: Dict[int, List[_Edge]],
    id_map: Dict[int, _ObjEntry],
) -> List[BlendShape]:
    """Walk a Geometry record's blend-shape chain and return BlendShape entries.

    Layout (canonical, post-FBX 2014):
        Geometry ← Deformer(BlendShape) ← Deformer(BlendShapeChannel) ←
        Geometry(Shape).

    Layout (legacy / some 7.x exporters):
        Inline ``Shape`` records as direct children of the Geometry node.

    Both are supported. Duplicates (a Shape reachable via both paths)
    are deduped by FBX object id.
    """
    geom_id = int(geom_rec.props[0]) if geom_rec.props else 0
    mesh_name = ""
    if geom_rec.props and len(geom_rec.props) > 1:
        mesh_name = _split_name_class(geom_rec.props[1])[0]

    out: List[BlendShape] = []
    consumed_ids: set = set()

    # ---- Canonical chain via the Connections graph ----
    for edge in in_edges.get(geom_id, []):
        deformer_id = edge.src
        deformer = id_map.get(deformer_id)
        if deformer is None or deformer.type_name != "Deformer":
            continue
        if deformer.sub_type != "BlendShape":
            continue
        # BlendShapeChannels (SubDeformers) → BlendShape Deformer
        for ch_edge in in_edges.get(deformer_id, []):
            channel_id = ch_edge.src
            channel = id_map.get(channel_id)
            if channel is None:
                continue
            if channel.type_name not in ("Deformer", "SubDeformer"):
                continue
            if channel.sub_type != "BlendShapeChannel":
                continue
            channel_name = channel.name
            # DeformPercent lives on the channel's Properties70.
            prop70 = channel.rec.child("Properties70")
            deform_percent = _read_property70_scalar(prop70, "DeformPercent", 0.0)
            # FBX stores DeformPercent as 0..100; normalize to 0..1.
            default_weight = max(0.0, min(1.0, deform_percent / 100.0)) if deform_percent > 1.0 else deform_percent
            # Shape Geometry → BlendShapeChannel
            for sh_edge in in_edges.get(channel_id, []):
                shape_id = sh_edge.src
                shape_entry = id_map.get(shape_id)
                if shape_entry is None or shape_entry.type_name != "Geometry":
                    continue
                if shape_entry.sub_type and shape_entry.sub_type != "Shape":
                    continue
                bs = _extract_one_shape(shape_entry.rec, channel_name, default_weight, mesh_name)
                if bs is not None:
                    out.append(bs)
                    consumed_ids.add(shape_id)

    # ---- Inline Shape records under the Geometry (legacy form) ----
    # These don't have an associated BlendShapeChannel; we use the bare
    # Shape's own name and a default weight of 0.
    for child in geom_rec.children:
        if child.name != "Shape":
            continue
        # Skip if already consumed via the connection-graph path. We
        # match by FBX object id when present.
        shape_obj_id = int(child.props[0]) if child.props else 0
        if shape_obj_id in consumed_ids:
            continue
        shape_name = ""
        if child.props and len(child.props) > 1:
            shape_name = _split_name_class(child.props[1])[0]
        bs = _extract_one_shape(child, shape_name, 0.0, mesh_name)
        if bs is not None:
            out.append(bs)

    return out


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def parse_fbx(buf_or_path: Union[bytes, bytearray, memoryview, str, Path]) -> ImportedModel:
    """Parse a binary FBX file → ImportedModel; animations are dropped (use parse_fbx_with_animations to keep them)."""
    return parse_fbx_with_animations(buf_or_path).model


def parse_fbx_with_animations(
    buf_or_path: Union[bytes, bytearray, memoryview, str, Path]
) -> ImportedModelWithAnims:
    """Parse a binary FBX file → ImportedModelWithAnims (mesh + bones + anims).

    Mirrors ``parse_gltf_with_animations`` so the import pipeline can
    treat the two source formats interchangeably. Coordinate-space
    conversion is delegated to the retargeter / ``imported_to_nj``.

    Notes
    -----
    Bone indices in the output tracks reference ``model.bones`` (i.e.
    the DFS order of LimbNode Model records). Animation tracks targeting
    Models that AREN'T joints are dropped — those are typically scene-
    graph nodes (root cameras, lights) and aren't useful for skeletal
    retargeting.
    """
    if isinstance(buf_or_path, (bytes, bytearray, memoryview)):
        data = bytes(buf_or_path)
    else:
        p = Path(buf_or_path)
        data = p.read_bytes()

    root = parse_binary_fbx(data)
    id_map = _build_id_map(root)
    out_edges, in_edges = _build_connections(root)

    bones, bone_id_to_idx = _extract_bones(id_map, out_edges, in_edges)

    meshes: List[ImportedMesh] = []
    blend_shapes: List[BlendShape] = []
    warnings: List[str] = []

    objs = root.child("Objects")
    if objs is not None:
        for rec in objs.children:
            if rec.name != "Geometry":
                continue
            sub_type = ""
            if len(rec.props) > 2:
                v = rec.props[2]
                if isinstance(v, bytes):
                    v = v.decode("utf-8", errors="replace")
                sub_type = v if isinstance(v, str) else ""
            if sub_type and sub_type not in ("Mesh", ""):
                # NurbsCurve, NurbsSurface, Shape, etc. ``Shape`` records
                # are blend-shape targets — handled below via the chain
                # walk, so we silently skip them here without a warning.
                if sub_type != "Shape":
                    warnings.append(f"skipped non-mesh Geometry: sub_type={sub_type}")
                continue
            mesh = _extract_mesh(rec, bone_id_to_idx, out_edges, in_edges, id_map)
            if mesh is not None:
                meshes.append(mesh)
            # BlendShape extraction is per-mesh: each Geometry can have
            # its own morph-target chain. We collect across all meshes
            # into a single ``blend_shapes`` list — the BlendShape
            # dataclass carries ``mesh_name`` so callers can group later.
            shapes = _extract_blend_shapes(rec, out_edges, in_edges, id_map)
            if shapes:
                blend_shapes.extend(shapes)

    if not meshes:
        warnings.append("FBX parsed but produced no meshes")
    if blend_shapes:
        # PSOBB doesn't render morphs — surface the count so the user
        # knows the data was parsed but won't appear in-game.
        warnings.append(
            f"parsed {len(blend_shapes)} blend shape(s); PSOBB has no morph rendering, "
            "data preserved on model.blend_shapes for downstream tooling"
        )

    # FBX UnitScaleFactor: convert to glTF-equivalent meters.
    # FBX default = cm (1 unit = 1 cm). PSOBB models are typically already
    # in cm-ish units — we leave scale_factor=1.0 by default; the user can
    # override via the UI scale slider.
    scale_factor = 1.0
    gs = root.child("GlobalSettings")
    if gs is not None:
        # Not strictly needed; we keep raw values and let imported_to_nj
        # apply the user's scale knob.
        pass

    animations = _extract_animations(id_map, out_edges, in_edges, bone_id_to_idx, bones)

    model = ImportedModel(
        meshes=meshes,
        bones=bones,
        bone_root=0,
        source_format="fbx",
        scale_factor=scale_factor,
        warnings=warnings,
        blend_shapes=blend_shapes,
    )
    return ImportedModelWithAnims(model=model, animations=animations)


__all__ = [
    "FbxRecord",
    "FbxParseError",
    "parse_binary_fbx",
    "parse_fbx",
    "parse_fbx_with_animations",
]
