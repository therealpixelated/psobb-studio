"""In-memory binary FBX synthesizer for the import tests.

Writes well-formed FBX 7.4 files that the v2 ``formats.fbx_reader`` can
parse end-to-end. Limited to the features we need exercised in tests:

  * Static mesh (vertices + polygon-vertex index + per-pv normals + UVs)
  * Skinned mesh (Geometry + Deformer/Skin + SubDeformer/Cluster +
    Model/LimbNode hierarchy)
  * Multi-mesh (two Geometry records sharing an Objects block)
  * Animation (AnimationStack + AnimationLayer + AnimationCurveNode +
    AnimationCurve)

The encoder follows the public binary FBX spec
(https://gist.github.com/iscle/0dbcee58be8582978d15ea3629ce3e8b) with
the following deliberate simplifications:

  * Always emits version 7400 (pre-v7.5 32-bit element headers); the
    parser handles both layouts, so this doesn't reduce coverage.
  * No zlib compression on small arrays (<128 byte payload); the
    parser handles both, but the synth assets stay readable in a hex
    editor that way.
  * Footer is 160 bytes minimum but doesn't reproduce the FBX ID hash —
    the parser only checks the FBX header and total byte count, which
    matches what ``fbxloader`` does in the wild.
"""
from __future__ import annotations

import math
import struct
import zlib
from typing import List, Optional, Sequence, Tuple


# ---- Binary FBX writer ----

_HEAD_MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"
_FOOT_ID = b"\xfa\xbc\xab\x09\xd0\xc8\xd4\x66\xb1\x76\xfb\x83\x1c\xf7\x26\x7e"
_FOOT_TAIL = b"\xf8\x5a\x8c\x6a\xde\xf5\xd9\x7e\xec\xe9\x0c\xe3\x75\x8f\x29\x0b"
_ALWAYS_BLOCK_SENTINEL = {b"AnimationStack", b"AnimationLayer"}


class FbxNode:
    """One node in a synthesized FBX tree.

    Use the chained property setters (``.I(int)``, ``.D(double)``, etc.)
    to attach values, and ``.add(child)`` to nest children.
    """

    __slots__ = ("name", "props", "props_type", "children")

    def __init__(self, name):
        self.name = name.encode("ascii") if isinstance(name, str) else name
        self.props: list = []
        self.props_type = bytearray()
        self.children: list = []

    def add(self, child: "FbxNode") -> "FbxNode":
        self.children.append(child)
        return child

    # Scalar property setters (return self for chaining).
    def I(self, v: int) -> "FbxNode":
        self.props.append(int(v)); self.props_type.append(ord("I")); return self

    def L(self, v: int) -> "FbxNode":
        self.props.append(int(v)); self.props_type.append(ord("L")); return self

    def F(self, v: float) -> "FbxNode":
        self.props.append(float(v)); self.props_type.append(ord("F")); return self

    def D(self, v: float) -> "FbxNode":
        self.props.append(float(v)); self.props_type.append(ord("D")); return self

    def Y(self, v: int) -> "FbxNode":
        self.props.append(int(v)); self.props_type.append(ord("Y")); return self

    def C(self, v: bool) -> "FbxNode":
        self.props.append(bool(v)); self.props_type.append(ord("C")); return self

    def S(self, v) -> "FbxNode":
        if isinstance(v, str):
            v = v.encode("utf-8")
        self.props.append(v); self.props_type.append(ord("S")); return self

    # Array property setters.
    def Iarr(self, vals: Sequence[int]) -> "FbxNode":
        self.props.append(("i", list(vals))); self.props_type.append(ord("i")); return self

    def Larr(self, vals: Sequence[int]) -> "FbxNode":
        self.props.append(("l", list(vals))); self.props_type.append(ord("l")); return self

    def Farr(self, vals: Sequence[float]) -> "FbxNode":
        self.props.append(("f", list(vals))); self.props_type.append(ord("f")); return self

    def Darr(self, vals: Sequence[float]) -> "FbxNode":
        self.props.append(("d", list(vals))); self.props_type.append(ord("d")); return self


def _encode_property(prop, typ) -> bytes:
    out = bytearray()
    out.append(typ)
    if typ == ord("Y"):
        out.extend(struct.pack("<h", prop))
    elif typ == ord("C"):
        out.extend(struct.pack("<B", 1 if prop else 0))
    elif typ == ord("I"):
        out.extend(struct.pack("<i", prop))
    elif typ == ord("F"):
        out.extend(struct.pack("<f", prop))
    elif typ == ord("D"):
        out.extend(struct.pack("<d", prop))
    elif typ == ord("L"):
        out.extend(struct.pack("<q", prop))
    elif typ in (ord("S"), ord("R")):
        out.extend(struct.pack("<I", len(prop)))
        out.extend(prop)
    elif typ in (ord("i"), ord("l"), ord("f"), ord("d")):
        kind, vals = prop
        fmt = {"i": "i", "l": "q", "f": "f", "d": "d"}[kind]
        payload = struct.pack(f"<{len(vals)}{fmt}", *vals)
        if len(payload) > 128:
            comp = zlib.compress(payload, 1)
            out.extend(struct.pack("<3I", len(vals), 1, len(comp)))
            out.extend(comp)
        else:
            out.extend(struct.pack("<3I", len(vals), 0, len(payload)))
            out.extend(payload)
    else:
        raise ValueError(f"unknown FBX property type: {typ:c}")
    return bytes(out)


def _encode_node(node: FbxNode, version: int, file_offset: int) -> bytes:
    is_v75 = version >= 7500
    props_bytes = bytearray()
    for p, t in zip(node.props, node.props_type):
        props_bytes.extend(_encode_property(p, t))

    write_block = bool(node.children) or node.name in _ALWAYS_BLOCK_SENTINEL
    header_size = (24 if is_v75 else 12) + 1 + len(node.name)
    sentinel_len = (25 if is_v75 else 13) if write_block else 0

    children_bytes = bytearray()
    cursor = file_offset + header_size + len(props_bytes)
    for child in node.children:
        cb = _encode_node(child, version, cursor)
        children_bytes.extend(cb)
        cursor += len(cb)
    if write_block:
        cursor += sentinel_len

    end_offset = cursor

    out = bytearray()
    if is_v75:
        out.extend(struct.pack("<3Q", end_offset, len(node.props), len(props_bytes)))
    else:
        out.extend(struct.pack("<3I", end_offset, len(node.props), len(props_bytes)))
    out.append(len(node.name))
    out.extend(node.name)
    out.extend(props_bytes)
    out.extend(children_bytes)
    if write_block:
        out.extend(b"\x00" * sentinel_len)
    return bytes(out)


def encode_fbx(root: FbxNode, version: int = 7400) -> bytes:
    """Serialize an FbxNode tree to binary FBX bytes.

    The synthetic root's children become the file's top-level nodes
    (FBXHeaderExtension, GlobalSettings, Objects, Connections, ...).
    """
    out = bytearray()
    out.extend(_HEAD_MAGIC)
    out.extend(struct.pack("<I", version))
    cursor = len(out)
    for child in root.children:
        cb = _encode_node(child, version, cursor)
        out.extend(cb)
        cursor += len(cb)
    is_v75 = version >= 7500
    null_len = 25 if is_v75 else 13
    out.extend(b"\x00" * null_len)
    out.extend(_FOOT_ID)
    pad = (16 - (len(out) % 16)) % 16
    if pad == 0:
        pad = 16
    out.extend(b"\x00" * pad)
    out.extend(struct.pack("<I", version))
    out.extend(b"\x00" * 120)
    out.extend(_FOOT_TAIL)
    return bytes(out)


# ---- High-level scene builders ----


def build_static_cube_fbx() -> bytes:
    """Build a minimal FBX with one cube mesh (8 verts, 6 quads).

    No skeleton, no animation. Tests the basic parse pipeline.
    """
    root = FbxNode("__root__")

    hdr = root.add(FbxNode("FBXHeaderExtension"))
    hdr.add(FbxNode("FBXHeaderVersion")).I(1003)
    hdr.add(FbxNode("FBXVersion")).I(7400)

    gs = root.add(FbxNode("GlobalSettings"))
    gs.add(FbxNode("Version")).I(1000)

    objs = root.add(FbxNode("Objects"))

    verts = [
        -1.0, -1.0, -1.0,
         1.0, -1.0, -1.0,
         1.0,  1.0, -1.0,
        -1.0,  1.0, -1.0,
        -1.0, -1.0,  1.0,
         1.0, -1.0,  1.0,
         1.0,  1.0,  1.0,
        -1.0,  1.0,  1.0,
    ]
    quads = [
        [0, 1, 2, 3],   # back
        [4, 7, 6, 5],   # front
        [0, 4, 5, 1],   # bottom
        [3, 2, 6, 7],   # top
        [0, 3, 7, 4],   # left
        [1, 5, 6, 2],   # right
    ]
    poly_idx = []
    for q in quads:
        for i, v in enumerate(q):
            poly_idx.append(-v - 1 if i == len(q) - 1 else v)

    geom_id = 1000000
    model_id = 2000000

    g = objs.add(FbxNode("Geometry"))
    g.L(geom_id).S("Cube\x00\x01Geometry").S("Mesh")
    g.add(FbxNode("Vertices")).Darr(verts)
    g.add(FbxNode("PolygonVertexIndex")).Iarr(poly_idx)

    # Per-polygon-vertex normals (Direct, ByPolygonVertex).
    face_normals = [
        (0, 0, -1),
        (0, 0,  1),
        (0, -1, 0),
        (0,  1, 0),
        (-1, 0, 0),
        (1,  0, 0),
    ]
    norm_arr: List[float] = []
    for fi, q in enumerate(quads):
        for _ in q:
            norm_arr.extend(face_normals[fi])
    le_n = g.add(FbxNode("LayerElementNormal"))
    le_n.I(0)
    le_n.add(FbxNode("Version")).I(101)
    le_n.add(FbxNode("MappingInformationType")).S("ByPolygonVertex")
    le_n.add(FbxNode("ReferenceInformationType")).S("Direct")
    le_n.add(FbxNode("Normals")).Darr(norm_arr)

    # IndexToDirect UVs (one set of 4 UVs reused per quad).
    uv_arr = [0.0, 0.0,  1.0, 0.0,  1.0, 1.0,  0.0, 1.0]
    uv_idx = []
    for _q in quads:
        for li in range(4):
            uv_idx.append(li)
    le_uv = g.add(FbxNode("LayerElementUV"))
    le_uv.I(0)
    le_uv.add(FbxNode("Version")).I(101)
    le_uv.add(FbxNode("MappingInformationType")).S("ByPolygonVertex")
    le_uv.add(FbxNode("ReferenceInformationType")).S("IndexToDirect")
    le_uv.add(FbxNode("UV")).Darr(uv_arr)
    le_uv.add(FbxNode("UVIndex")).Iarr(uv_idx)

    m = objs.add(FbxNode("Model"))
    m.L(model_id).S("Cube\x00\x01Model").S("Mesh")
    m.add(FbxNode("Version")).I(232)

    conns = root.add(FbxNode("Connections"))
    conns.add(FbxNode("C")).S("OO").L(geom_id).L(model_id)
    conns.add(FbxNode("C")).S("OO").L(model_id).L(0)

    return encode_fbx(root)


def build_skinned_humanoid_fbx() -> bytes:
    """Build a tiny FBX with 3 bones + a skinned 4-vertex mesh.

    Skeleton: root → hip → leg, each bone offset along +Y by 1.
    Mesh: a 1x1 quad along XZ at Y=0, weighted entirely to the leg.
    Tests the Deformer/Cluster path + bone DFS ordering.
    """
    root = FbxNode("__root__")

    hdr = root.add(FbxNode("FBXHeaderExtension"))
    hdr.add(FbxNode("FBXHeaderVersion")).I(1003)
    hdr.add(FbxNode("FBXVersion")).I(7400)

    gs = root.add(FbxNode("GlobalSettings"))
    gs.add(FbxNode("Version")).I(1000)

    objs = root.add(FbxNode("Objects"))

    geom_id = 1000000
    mesh_model_id = 1100000
    root_bone_id = 2000000
    hip_bone_id = 2100000
    leg_bone_id = 2200000
    skin_id = 3000000
    cluster_root_id = 3100000
    cluster_hip_id = 3200000
    cluster_leg_id = 3300000
    null_node_attr_id = 4000000
    limb_attr_id_root = 4100000
    limb_attr_id_hip = 4200000
    limb_attr_id_leg = 4300000

    # Geometry: 4 vertices, 1 quad.
    verts = [
        -1.0, 0.0, -1.0,
         1.0, 0.0, -1.0,
         1.0, 0.0,  1.0,
        -1.0, 0.0,  1.0,
    ]
    quads = [[0, 1, 2, 3]]
    poly_idx = []
    for q in quads:
        for i, v in enumerate(q):
            poly_idx.append(-v - 1 if i == len(q) - 1 else v)

    g = objs.add(FbxNode("Geometry"))
    g.L(geom_id).S("Body\x00\x01Geometry").S("Mesh")
    g.add(FbxNode("Vertices")).Darr(verts)
    g.add(FbxNode("PolygonVertexIndex")).Iarr(poly_idx)
    # Per-vertex normal (ByVertex) so we exercise that path too.
    le_n = g.add(FbxNode("LayerElementNormal"))
    le_n.I(0)
    le_n.add(FbxNode("Version")).I(101)
    le_n.add(FbxNode("MappingInformationType")).S("ByVertex")
    le_n.add(FbxNode("ReferenceInformationType")).S("Direct")
    le_n.add(FbxNode("Normals")).Darr([0, 1, 0,  0, 1, 0,  0, 1, 0,  0, 1, 0])

    # Mesh model.
    m = objs.add(FbxNode("Model"))
    m.L(mesh_model_id).S("Body\x00\x01Model").S("Mesh")
    m.add(FbxNode("Version")).I(232)

    # NodeAttribute (LimbNode) records — each bone Model has one.
    for attr_id in (limb_attr_id_root, limb_attr_id_hip, limb_attr_id_leg):
        na = objs.add(FbxNode("NodeAttribute"))
        na.L(attr_id).S("LimbNode\x00\x01NodeAttribute").S("LimbNode")
        na.add(FbxNode("TypeFlags")).S("Skeleton")

    # Bone Models.
    for bone_id, name, lcl in [
        (root_bone_id, "root", (0.0, 0.0, 0.0)),
        (hip_bone_id, "hip", (0.0, 1.0, 0.0)),
        (leg_bone_id, "leg", (0.0, 1.0, 0.0)),
    ]:
        bm = objs.add(FbxNode("Model"))
        bm.L(bone_id).S(f"{name}\x00\x01Model").S("LimbNode")
        bm.add(FbxNode("Version")).I(232)
        p70 = bm.add(FbxNode("Properties70"))
        p = p70.add(FbxNode("P"))
        p.S("Lcl Translation").S("Lcl Translation").S("").S("A")
        p.D(lcl[0]).D(lcl[1]).D(lcl[2])

    # Deformer (Skin) + 3 Clusters.
    sk = objs.add(FbxNode("Deformer"))
    sk.L(skin_id).S("Body\x00\x01Skin").S("Skin")
    sk.add(FbxNode("Version")).I(101)

    # Cluster for root: empty (no influence) — but we still emit it to
    # exercise the empty-cluster path.
    for cluster_id, indexes, weights in [
        (cluster_root_id, [], []),
        (cluster_hip_id, [], []),
        (cluster_leg_id, [0, 1, 2, 3], [1.0, 1.0, 1.0, 1.0]),
    ]:
        cl = objs.add(FbxNode("Deformer"))
        cl.L(cluster_id).S("Cluster\x00\x01SubDeformer").S("Cluster")
        cl.add(FbxNode("Version")).I(100)
        if indexes:
            cl.add(FbxNode("Indexes")).Iarr(indexes)
            cl.add(FbxNode("Weights")).Darr(weights)

    # Connections.
    conns = root.add(FbxNode("Connections"))
    # Geom → mesh model
    conns.add(FbxNode("C")).S("OO").L(geom_id).L(mesh_model_id)
    # Mesh model → root scene
    conns.add(FbxNode("C")).S("OO").L(mesh_model_id).L(0)
    # NodeAttribute → bone model (each)
    conns.add(FbxNode("C")).S("OO").L(limb_attr_id_root).L(root_bone_id)
    conns.add(FbxNode("C")).S("OO").L(limb_attr_id_hip).L(hip_bone_id)
    conns.add(FbxNode("C")).S("OO").L(limb_attr_id_leg).L(leg_bone_id)
    # Bone hierarchy: leg → hip → root → scene
    conns.add(FbxNode("C")).S("OO").L(leg_bone_id).L(hip_bone_id)
    conns.add(FbxNode("C")).S("OO").L(hip_bone_id).L(root_bone_id)
    conns.add(FbxNode("C")).S("OO").L(root_bone_id).L(0)
    # Skin → geometry
    conns.add(FbxNode("C")).S("OO").L(skin_id).L(geom_id)
    # Clusters → skin
    conns.add(FbxNode("C")).S("OO").L(cluster_root_id).L(skin_id)
    conns.add(FbxNode("C")).S("OO").L(cluster_hip_id).L(skin_id)
    conns.add(FbxNode("C")).S("OO").L(cluster_leg_id).L(skin_id)
    # Cluster → bone (the bone the cluster represents)
    conns.add(FbxNode("C")).S("OO").L(cluster_root_id).L(root_bone_id)
    conns.add(FbxNode("C")).S("OO").L(cluster_hip_id).L(hip_bone_id)
    conns.add(FbxNode("C")).S("OO").L(cluster_leg_id).L(leg_bone_id)

    return encode_fbx(root)


def build_multi_mesh_fbx() -> bytes:
    """Build an FBX with two separate Geometry records (body + clothing).

    Tests that the parser produces two ImportedMesh entries. No skeleton.
    """
    root = FbxNode("__root__")

    root.add(FbxNode("FBXHeaderExtension")).add(FbxNode("FBXVersion")).I(7400)

    gs = root.add(FbxNode("GlobalSettings"))
    gs.add(FbxNode("Version")).I(1000)

    objs = root.add(FbxNode("Objects"))

    body_geom_id = 1000000
    body_model_id = 1100000
    cloth_geom_id = 1200000
    cloth_model_id = 1300000

    # --- Body: a triangle ---
    g1 = objs.add(FbxNode("Geometry"))
    g1.L(body_geom_id).S("Body\x00\x01Geometry").S("Mesh")
    g1.add(FbxNode("Vertices")).Darr([0, 0, 0,  1, 0, 0,  0, 1, 0])
    g1.add(FbxNode("PolygonVertexIndex")).Iarr([0, 1, -3])

    m1 = objs.add(FbxNode("Model"))
    m1.L(body_model_id).S("Body\x00\x01Model").S("Mesh")
    m1.add(FbxNode("Version")).I(232)

    # --- Clothing: another triangle, offset ---
    g2 = objs.add(FbxNode("Geometry"))
    g2.L(cloth_geom_id).S("Cloth\x00\x01Geometry").S("Mesh")
    g2.add(FbxNode("Vertices")).Darr([0, 0, 1,  1, 0, 1,  0, 1, 1])
    g2.add(FbxNode("PolygonVertexIndex")).Iarr([0, 1, -3])

    m2 = objs.add(FbxNode("Model"))
    m2.L(cloth_model_id).S("Cloth\x00\x01Model").S("Mesh")
    m2.add(FbxNode("Version")).I(232)

    conns = root.add(FbxNode("Connections"))
    conns.add(FbxNode("C")).S("OO").L(body_geom_id).L(body_model_id)
    conns.add(FbxNode("C")).S("OO").L(body_model_id).L(0)
    conns.add(FbxNode("C")).S("OO").L(cloth_geom_id).L(cloth_model_id)
    conns.add(FbxNode("C")).S("OO").L(cloth_model_id).L(0)

    return encode_fbx(root)


def build_skinned_with_animation_fbx() -> bytes:
    """Build a skinned mesh with one rotation animation on the leg bone.

    Useful to test the AnimationStack/Layer/CurveNode/Curve walk and the
    Euler-degree → quaternion conversion.
    """
    root = FbxNode("__root__")

    hdr = root.add(FbxNode("FBXHeaderExtension"))
    hdr.add(FbxNode("FBXHeaderVersion")).I(1003)
    hdr.add(FbxNode("FBXVersion")).I(7400)

    gs = root.add(FbxNode("GlobalSettings"))
    gs.add(FbxNode("Version")).I(1000)

    objs = root.add(FbxNode("Objects"))

    geom_id = 1000000
    mesh_model_id = 1100000
    root_bone_id = 2000000
    leg_bone_id = 2100000
    skin_id = 3000000
    cluster_leg_id = 3100000
    limb_attr_id_root = 4000000
    limb_attr_id_leg = 4100000

    anim_stack_id = 5000000
    anim_layer_id = 5100000
    cnode_rot_id = 5200000
    curve_x_id = 5300000
    curve_y_id = 5400000
    curve_z_id = 5500000

    # Geom: a single triangle weighted to the leg.
    g = objs.add(FbxNode("Geometry"))
    g.L(geom_id).S("Body\x00\x01Geometry").S("Mesh")
    g.add(FbxNode("Vertices")).Darr([0, 0, 0,  1, 0, 0,  0, 0, 1])
    g.add(FbxNode("PolygonVertexIndex")).Iarr([0, 1, -3])

    # Mesh model.
    m = objs.add(FbxNode("Model"))
    m.L(mesh_model_id).S("Body\x00\x01Model").S("Mesh")
    m.add(FbxNode("Version")).I(232)

    # Two LimbNodes.
    for attr_id in (limb_attr_id_root, limb_attr_id_leg):
        na = objs.add(FbxNode("NodeAttribute"))
        na.L(attr_id).S("LimbNode\x00\x01NodeAttribute").S("LimbNode")
        na.add(FbxNode("TypeFlags")).S("Skeleton")
    for bone_id, name, lcl in [
        (root_bone_id, "root", (0.0, 0.0, 0.0)),
        (leg_bone_id, "leg", (0.0, 1.0, 0.0)),
    ]:
        bm = objs.add(FbxNode("Model"))
        bm.L(bone_id).S(f"{name}\x00\x01Model").S("LimbNode")
        bm.add(FbxNode("Version")).I(232)
        p70 = bm.add(FbxNode("Properties70"))
        p = p70.add(FbxNode("P"))
        p.S("Lcl Translation").S("Lcl Translation").S("").S("A")
        p.D(lcl[0]).D(lcl[1]).D(lcl[2])

    # Skin + Cluster.
    sk = objs.add(FbxNode("Deformer"))
    sk.L(skin_id).S("Body\x00\x01Skin").S("Skin")
    sk.add(FbxNode("Version")).I(101)

    cl = objs.add(FbxNode("Deformer"))
    cl.L(cluster_leg_id).S("Cluster\x00\x01SubDeformer").S("Cluster")
    cl.add(FbxNode("Version")).I(100)
    cl.add(FbxNode("Indexes")).Iarr([0, 1, 2])
    cl.add(FbxNode("Weights")).Darr([1.0, 1.0, 1.0])

    # AnimationStack + Layer.
    anim_stack = objs.add(FbxNode("AnimationStack"))
    anim_stack.L(anim_stack_id).S("Take001\x00\x01AnimStack").S("")

    anim_layer = objs.add(FbxNode("AnimationLayer"))
    anim_layer.L(anim_layer_id).S("BaseLayer\x00\x01AnimLayer").S("")

    # CurveNode (rotation, owns 3 curves).
    cnode = objs.add(FbxNode("AnimationCurveNode"))
    cnode.L(cnode_rot_id).S("R\x00\x01AnimCurveNode").S("")

    # 3 curves: X, Y, Z.
    # Animation: rotate leg from 0° to 90° around Y over 1 second.
    # FBX time unit: 46186158000 per second. 1s = 46186158000.
    # We use 4 keys: t=0, t=0.25s, t=0.5s, t=1s.
    KTIME = 46186158000
    keytimes_y = [0, KTIME // 4, KTIME // 2, KTIME]
    keyvalues_y = [0.0, 22.5, 45.0, 90.0]

    cx = objs.add(FbxNode("AnimationCurve"))
    cx.L(curve_x_id).S("\x00\x01AnimCurve").S("")
    cx.add(FbxNode("KeyTime")).Larr([0, KTIME])
    cx.add(FbxNode("KeyValueFloat")).Farr([0.0, 0.0])

    cy = objs.add(FbxNode("AnimationCurve"))
    cy.L(curve_y_id).S("\x00\x01AnimCurve").S("")
    cy.add(FbxNode("KeyTime")).Larr(keytimes_y)
    cy.add(FbxNode("KeyValueFloat")).Farr(keyvalues_y)

    cz = objs.add(FbxNode("AnimationCurve"))
    cz.L(curve_z_id).S("\x00\x01AnimCurve").S("")
    cz.add(FbxNode("KeyTime")).Larr([0, KTIME])
    cz.add(FbxNode("KeyValueFloat")).Farr([0.0, 0.0])

    # Connections.
    conns = root.add(FbxNode("Connections"))
    conns.add(FbxNode("C")).S("OO").L(geom_id).L(mesh_model_id)
    conns.add(FbxNode("C")).S("OO").L(mesh_model_id).L(0)
    conns.add(FbxNode("C")).S("OO").L(limb_attr_id_root).L(root_bone_id)
    conns.add(FbxNode("C")).S("OO").L(limb_attr_id_leg).L(leg_bone_id)
    conns.add(FbxNode("C")).S("OO").L(leg_bone_id).L(root_bone_id)
    conns.add(FbxNode("C")).S("OO").L(root_bone_id).L(0)
    conns.add(FbxNode("C")).S("OO").L(skin_id).L(geom_id)
    conns.add(FbxNode("C")).S("OO").L(cluster_leg_id).L(skin_id)
    conns.add(FbxNode("C")).S("OO").L(cluster_leg_id).L(leg_bone_id)
    # Animation linking:
    #   anim_layer → anim_stack (OO)
    #   cnode → anim_layer (OO)
    #   cnode → leg_bone (OP, "Lcl Rotation")
    #   curve → cnode (OP, "d|X" / "d|Y" / "d|Z")
    conns.add(FbxNode("C")).S("OO").L(anim_layer_id).L(anim_stack_id)
    conns.add(FbxNode("C")).S("OO").L(cnode_rot_id).L(anim_layer_id)
    conns.add(FbxNode("C")).S("OP").L(cnode_rot_id).L(leg_bone_id).S("Lcl Rotation")
    conns.add(FbxNode("C")).S("OP").L(curve_x_id).L(cnode_rot_id).S("d|X")
    conns.add(FbxNode("C")).S("OP").L(curve_y_id).L(cnode_rot_id).S("d|Y")
    conns.add(FbxNode("C")).S("OP").L(curve_z_id).L(cnode_rot_id).S("d|Z")

    return encode_fbx(root)


__all__ = [
    "FbxNode",
    "encode_fbx",
    "build_static_cube_fbx",
    "build_skinned_humanoid_fbx",
    "build_multi_mesh_fbx",
    "build_skinned_with_animation_fbx",
]
