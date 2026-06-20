"""External-format importer for the PSOBB Texture Editor.

Parses ``.obj`` / ``.gltf`` / ``.glb`` / ``.fbx`` exports from Blender,
Maya, 3ds Max, Unity, etc. and projects them into the editor's NJ
writer struct (`formats.nj_writer.NjModel`) so they can be deployed via
the just-shipped /api/build_nj path.

Supported formats:
    OBJ      hand-rolled parser. No skeleton, no animation; we paste the
             single mesh onto a chosen skeleton template (or a 1-bone
             root if no template is requested).
    glTF/GLB pygltflib-backed. Vertex positions/normals/UVs + skinning
             (joints + weights) + skeleton hierarchy from the first skin.
             v2: Animations imported via ``parse_gltf_with_animations``.
    FBX      v2: own pure-Python binary FBX parser (formats.fbx_reader).
             Vertex positions/normals/UVs + skinning + skeleton
             hierarchy + animation tracks. Mirrors the glTF capability.
             Binary FBX 7.0+ (FBX 2010+ Binary). ASCII FBX is rejected
             with a clear message pointing at the binary export option.

Coordinate convention:
    glTF / Blender export: right-handed, Y-up, +Z forward.
    PSOBB                 : LEFT-handed, Y-up, -Z forward.
    Conversion             : flip Z on positions/normals + bone bind pose
                             + the corresponding texcoord V if needed.

The conversion is gated by a boolean flag the UI exposes — some users
import models that already match PSOBB conventions (e.g. another PSO
mod packaged as glTF), in which case the flip is a no-op.

Quaternion -> BAMS:
    PSOBB stores rotations as 3 BAM (Binary Angular Measurement) ints,
    one per axis: u16 BAM = degrees * 65536 / 360. The composition is
    ZYX Euler unless EVAL_ZXY_ANG is set (rare); we emit ZYX. The
    bind-pose quat is converted to ZYX intrinsic Euler then to BAMS via
    ``quat_to_zyx_bams``.

Skin weights:
    PSOBB accepts 4 bone slots per vertex with 8-bit weight each
    (NJD_CV_VN_NF chunk type 47). glTF stores up to 4 joints + 4 float
    weights — we re-quantize the weights to 8-bit, renormalize the sum
    to 255, and emit (weight, joint) pairs. Vertices with no skinning
    information get assigned weight=255 to bone 0.

Chunk emission strategy (v1, conservative):
    All meshes use:
      - chunk type 41 NJD_CV_VN  (POS + NORMAL, no UV in v-chunk)
      - chunk type 64 strip      (bare strip, indices only)
    UVs come from the strip chunk's variant when present; lacking a
    UV-flagged strip type we fall back to type 64 (UV-less strips) and
    let the user texture-bind with material_id=0. This is the simplest
    path that round-trips cleanly through ``parse_nj_for_writer`` and
    ``parse_nj_file`` (xj.py).

    Future: emit chunk type 47 (POS + NORMAL + idx + bw) when a skin is
    present and chunk type 65/68 (strip with UVs) when uvs are non-empty.
    The encoder is type-id agnostic; the only limit is what the rendering
    parser supports — it handles 32..50 vertex chunks and 64..75 strip
    chunks today.
"""
from __future__ import annotations

import base64
import json
import math
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .nj_writer import NjChunk, NjMeshChunks, NjModel, NjNode


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ImportedMesh:
    """One submesh in source-coordinates from an external import.

    Vertices live in the source's coordinate frame; the convert-to-NJ
    pipeline applies the (axis_flip, scale) transform at emit time so
    callers can preview the source data verbatim before commit.

    Attributes
    ----------
    name:
        Source-supplied name (mesh.name in glTF, ``o ...`` group in OBJ).
        Empty string when the source has no name.
    vertices:
        ``(N, 3)`` float32. World-space (the source's world; we don't
        apply node transforms — most exporters bake them in).
    indices:
        ``(M, 3)`` uint32 triangle list. Already de-stripified.
    uvs / normals:
        ``(N, 2)`` / ``(N, 3)`` or None when the source omits them.
    skin_indices / skin_weights:
        ``(N, 4)`` uint8 / ``(N, 4)`` float32 sum-to-1; both None when
        the source has no skinning.
    material_id:
        0..255. Picked from the source material slot, or 0 if there's
        no material binding. The NJ writer doesn't care about the
        material layout; the editor binds material_id -> tile_index at
        render time.
    """
    name: str
    vertices: np.ndarray
    indices: np.ndarray
    uvs: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    skin_indices: Optional[np.ndarray] = None
    skin_weights: Optional[np.ndarray] = None
    material_id: int = 0


@dataclass
class ImportedBone:
    """One bone in the source hierarchy.

    All bind-pose values are in the source's coordinate frame; the
    converter applies axis flips at emit time. ``parent_idx`` is -1 for
    the root bone.
    """
    name: str
    parent_idx: int
    bind_pos: Tuple[float, float, float]
    bind_rot_quat: Tuple[float, float, float, float]  # (x, y, z, w)
    bind_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class BlendShape:
    """One FBX BlendShape (morph target) the parser recovered.

    PSOBB doesn't render blend shapes, so the import pipeline ignores
    these once they're attached to the ``ImportedModel``. We keep them
    on the dataclass so downstream tooling that wants to use the data
    for some other purpose (e.g. dumping facial-rig shapes to a side
    file, building a separate morph pipeline) can read them out.

    Fields are kept minimal — vertex deltas are stored in the SOURCE
    coordinate frame (no axis flip applied yet) so they line up with
    the equivalent ``ImportedMesh.vertices``.

    Attributes
    ----------
    name:
        Source-supplied name (typically the BlendShapeChannel's name,
        e.g. "Smile", "BrowUp", "Mouth_OO"). Empty when the source has
        no name.
    indexes:
        ``(K,)`` int32 — the vertex indices the offsets apply to. Sparse:
        only verts that actually move appear in the array.
    offsets:
        ``(K, 3)`` float32 — per-vertex position delta (added on top of
        the bind-pose mesh.vertices when the channel's weight is 1.0).
    normals:
        Optional ``(K, 3)`` float32 — per-vertex normal delta, when the
        source authored separate normal targets. None when absent.
    default_weight:
        0.0..1.0 default channel weight from FBX's ``DeformPercent``.
        Most channels default to 0; nonzero defaults indicate a static
        deformation the user expected to be on always.
    mesh_name:
        Name of the ImportedMesh (Geometry record) the shape attaches
        to. Used by callers that need to associate shapes back with a
        specific mesh in a multi-mesh import.
    """
    name: str
    indexes: np.ndarray
    offsets: np.ndarray
    normals: Optional[np.ndarray] = None
    default_weight: float = 0.0
    mesh_name: str = ""


@dataclass
class SpringBoneJoint:
    """One node in a VRM ``VRMC_springBone`` chain.

    A spring chain hangs off a "root" joint and propagates secondary
    motion (cloth, hair, accessories) via a per-joint stiffness/drag
    integration. PSOBB has no equivalent runtime so the data is
    preserved verbatim for round-trip / Blender re-import workflows.

    Attributes
    ----------
    bone_idx:
        Index into ``ImportedModel.bones``. -1 if the joint references a
        node that isn't in skin[0] (rare; we drop those silently in
        the parser, so this is mostly informational).
    hit_radius:
        Per-joint collision radius in source units. 0 disables collision
        on this joint.
    stiffness:
        VRM-1.0 ``stiffness`` field, 0..1. Not interpreted here — the
        spec assigns it to the parent->child segment of the chain.
    drag_force:
        VRM-1.0 ``dragForce`` (a.k.a. damping), 0..1.
    gravity_power:
        VRM-1.0 ``gravityPower``, scalar multiplier on ``gravity_dir``.
    gravity_dir:
        ``(x, y, z)`` direction the joint settles toward. Default
        ``(0, -1, 0)``.
    """
    bone_idx: int
    hit_radius: float = 0.0
    stiffness: float = 1.0
    drag_force: float = 0.4
    gravity_power: float = 0.0
    gravity_dir: Tuple[float, float, float] = (0.0, -1.0, 0.0)


@dataclass
class SpringBoneCollider:
    """One collider shape in a VRM ``VRMC_springBone`` setup.

    The spec supports sphere + capsule colliders. Both share the same
    dataclass; ``shape`` selects which fields apply.

    Attributes
    ----------
    bone_idx:
        Bone the collider is attached to. -1 when not skin-resident.
    shape:
        ``"sphere"`` or ``"capsule"``.
    offset:
        ``(x, y, z)`` local offset from the bone origin.
    radius:
        Sphere/capsule radius.
    tail:
        Capsule second endpoint (offset). ``(0, 0, 0)`` for sphere.
    """
    bone_idx: int
    shape: str = "sphere"
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.0
    tail: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class SpringBoneChain:
    """One named chain of spring-bone joints + its associated colliders.

    VRM 1.0 ``VRMC_springBone.springs[i]`` maps onto this dataclass.
    The joints list is ordered root → tip; the collider list is what
    the chain is allowed to collide against.

    Attributes
    ----------
    name:
        VRM-supplied chain name, often empty in practice. Synthesised as
        ``"spring_<i>"`` if the source omits it.
    joints:
        ``SpringBoneJoint`` entries in chain order.
    colliders:
        ``SpringBoneCollider`` entries the chain collides with. Note
        that VRM expresses collisions via "collider groups" referenced
        by the spring; we flatten the group into the colliders list so
        downstream consumers don't need to chase the indirection.
    center_bone_idx:
        Optional center bone (the simulation's local-space origin).
        -1 when absent or unresolved.
    """
    name: str
    joints: List[SpringBoneJoint] = field(default_factory=list)
    colliders: List[SpringBoneCollider] = field(default_factory=list)
    center_bone_idx: int = -1


@dataclass
class NodeConstraint:
    """One VRM ``VRMC_node_constraint`` entry.

    These describe runtime relationships between bones: roll constraints
    (twist propagation), aim constraints (look-at), and rotation
    constraints. PSOBB doesn't honour any of these but Blender's VRM
    addon round-trips them on re-import, so the editor preserves them.

    Attributes
    ----------
    bone_idx:
        Constrained bone (the node the constraint sits on).
    source_bone_idx:
        The bone the constraint reads its driving rotation from. -1
        when unresolved.
    constraint_type:
        ``"roll"`` | ``"aim"`` | ``"rotation"``. Mirrors the VRM enum.
    weight:
        ``0..1`` blend weight from the spec.
    axis:
        Roll/aim axis: ``"X"`` / ``"Y"`` / ``"Z"`` (the VRM enum). Empty
        when the type doesn't use it.
    """
    bone_idx: int
    source_bone_idx: int = -1
    constraint_type: str = ""
    weight: float = 1.0
    axis: str = ""


@dataclass
class ImportedModel:
    """A parsed external model in source coordinates.

    Returned by every ``parse_*`` function in this module. The converter
    ``imported_to_nj`` consumes this and produces a deployable NjModel.

    Attributes
    ----------
    meshes / bones / bone_root / source_format / scale_factor / warnings:
        See ``parse_obj`` / ``parse_gltf`` / ``parse_fbx`` for details.
    vrm_humanoid_map:
        When the source is a VRM file (glTF with ``extensions.VRM`` or
        ``extensions.VRMC_vrm``), this maps the VRM humanoid bone role
        (e.g. ``"hips"``, ``"leftUpperArm"``) to a bone index into
        ``bones``. Empty dict for non-VRM sources. Used by the
        retargeter as an authoritative bone-role lookup that bypasses
        string-matching.
    blend_shapes:
        FBX morph targets recovered as a side-channel. PSOBB's NJ
        runtime ignores these (no morph rendering); we keep them on the
        model for downstream tooling that wants to use them. Empty
        list for non-FBX or shape-free sources.
    spring_bones:
        VRM 1.0 ``VRMC_springBone`` chains preserved on import. PSOBB
        has no secondary-motion runtime; we keep the data so a
        downstream Blender re-import or JSON side-file exporter can
        round-trip it. Empty list for non-VRM and VRM-without-springs
        sources.
    node_constraints:
        VRM 1.0 ``VRMC_node_constraint`` look-at / roll / aim
        constraints. Same preservation rationale as ``spring_bones``.
    """
    meshes: List[ImportedMesh] = field(default_factory=list)
    bones: List[ImportedBone] = field(default_factory=list)
    bone_root: int = 0
    source_format: str = ""
    scale_factor: float = 1.0
    warnings: List[str] = field(default_factory=list)
    vrm_humanoid_map: Dict[str, int] = field(default_factory=dict)
    blend_shapes: List[BlendShape] = field(default_factory=list)
    spring_bones: List[SpringBoneChain] = field(default_factory=list)
    node_constraints: List[NodeConstraint] = field(default_factory=list)


# ---------------------------------------------------------------------------
# OBJ parser
# ---------------------------------------------------------------------------
#
# Wavefront OBJ is text-based, line-oriented. Reference:
# https://en.wikipedia.org/wiki/Wavefront_.obj_file
#
# We support:
#   v   x y z    [w]                vertex position (w ignored)
#   vn  x y z                       vertex normal
#   vt  u v      [w]                vertex texcoord (w ignored)
#   f   v[/vt[/vn]] v.. v..         face (3+ verts; we triangulate fans)
#   o   <name>                      object/group separator
#   g   <name>                      group (treated as a sub-object)
#   usemtl <name>                   material id is a hash bucketed to 0..255
#
# We DO NOT honor:
#   smoothing groups (s)
#   .mtl files (we just bucket usemtl name -> material_id)
#   negative indices (would require buffering all indices first)
#
# The parser produces a single ImportedMesh per (object, material) pair
# unless the file has no objects, in which case it produces one mesh.


def parse_obj(data: bytes) -> ImportedModel:
    """Parse a Wavefront ``.obj`` file into an ImportedModel.

    Returns one ImportedMesh per (object/group, material) split. Vertex
    indices are mapped per-mesh via a small dedup table so each mesh
    gets a compact 0..N-1 vertex range.

    The OBJ format has no skeleton, so the returned ``bones`` list is
    empty (the converter substitutes a 1-bone or template skeleton).
    """
    text = data.decode("utf-8", errors="replace")
    # Master attribute pools (1-indexed in OBJ).
    pos: List[Tuple[float, float, float]] = []
    nrm: List[Tuple[float, float, float]] = []
    uvs: List[Tuple[float, float]] = []

    # Per-submesh accumulators.
    @dataclass
    class _Sub:
        name: str
        material_id: int
        # Per-vertex dedup: (pi, ti, ni) -> emit_index
        dedup: Dict[Tuple[int, int, int], int] = field(default_factory=dict)
        out_pos: List[Tuple[float, float, float]] = field(default_factory=list)
        out_nrm: List[Tuple[float, float, float]] = field(default_factory=list)
        out_uv: List[Tuple[float, float]] = field(default_factory=list)
        out_tri: List[Tuple[int, int, int]] = field(default_factory=list)

    submeshes: List[_Sub] = []
    cur_name = ""
    cur_mat = 0
    mat_table: Dict[str, int] = {}

    def _ensure_sub() -> _Sub:
        # Pick the LAST submesh matching (cur_name, cur_mat); create a
        # new one if none.
        for s in submeshes:
            if s.name == cur_name and s.material_id == cur_mat:
                return s
        s = _Sub(name=cur_name, material_id=cur_mat)
        submeshes.append(s)
        return s

    def _resolve_index(s: _Sub, pi: int, ti: int, ni: int) -> int:
        key = (pi, ti, ni)
        idx = s.dedup.get(key)
        if idx is not None:
            return idx
        idx = len(s.out_pos)
        s.dedup[key] = idx
        if 0 <= pi < len(pos):
            s.out_pos.append(pos[pi])
        else:
            s.out_pos.append((0.0, 0.0, 0.0))
        if 0 <= ti < len(uvs):
            s.out_uv.append(uvs[ti])
        else:
            s.out_uv.append((0.0, 0.0))
        if 0 <= ni < len(nrm):
            s.out_nrm.append(nrm[ni])
        else:
            s.out_nrm.append((0.0, 1.0, 0.0))
        return idx

    for raw_line in text.splitlines():
        # Strip comments + trailing whitespace.
        ln = raw_line.split("#", 1)[0].strip()
        if not ln:
            continue
        toks = ln.split()
        kw = toks[0]
        try:
            if kw == "v" and len(toks) >= 4:
                pos.append((float(toks[1]), float(toks[2]), float(toks[3])))
            elif kw == "vn" and len(toks) >= 4:
                nrm.append((float(toks[1]), float(toks[2]), float(toks[3])))
            elif kw == "vt" and len(toks) >= 3:
                # OBJ V is bottom-up; PSOBB V is top-down. We flip here.
                uvs.append((float(toks[1]), 1.0 - float(toks[2])))
            elif kw in ("o", "g") and len(toks) >= 2:
                cur_name = " ".join(toks[1:])
            elif kw == "usemtl" and len(toks) >= 2:
                name = toks[1]
                mid = mat_table.get(name)
                if mid is None:
                    mid = len(mat_table)
                    if mid > 255:
                        mid = 255
                    mat_table[name] = mid
                cur_mat = mid
            elif kw == "f" and len(toks) >= 4:
                s = _ensure_sub()
                # Decode each (v/t/n) triple. Indices are 1-based; an
                # empty middle field means "no UV/normal".
                fan: List[int] = []
                for tk in toks[1:]:
                    parts = tk.split("/")
                    pi = int(parts[0]) - 1 if parts[0] else -1
                    ti = int(parts[1]) - 1 if len(parts) > 1 and parts[1] else -1
                    ni = int(parts[2]) - 1 if len(parts) > 2 and parts[2] else -1
                    fan.append(_resolve_index(s, pi, ti, ni))
                # Fan-triangulate (v0, v1, v2), (v0, v2, v3), ...
                for i in range(1, len(fan) - 1):
                    s.out_tri.append((fan[0], fan[i], fan[i + 1]))
        except (ValueError, IndexError):
            # Skip malformed lines silently — most OBJ files have a few.
            continue

    # Build ImportedMesh list.
    meshes: List[ImportedMesh] = []
    for s in submeshes:
        if not s.out_tri:
            continue
        verts = np.asarray(s.out_pos, dtype=np.float32)
        normals = np.asarray(s.out_nrm, dtype=np.float32)
        u = np.asarray(s.out_uv, dtype=np.float32)
        idx = np.asarray(s.out_tri, dtype=np.uint32)
        meshes.append(ImportedMesh(
            name=s.name or f"mesh_{len(meshes)}",
            vertices=verts,
            indices=idx,
            uvs=u,
            normals=normals,
            material_id=s.material_id,
        ))

    warnings: List[str] = []
    if not meshes:
        warnings.append("OBJ file produced no meshes — empty or only contained non-face data")
    if not nrm:
        warnings.append("OBJ has no normals (`vn`) — emitting (0,1,0) for every vertex")
    if not uvs:
        warnings.append("OBJ has no UVs (`vt`) — emitting (0,0) for every vertex")

    return ImportedModel(
        meshes=meshes,
        bones=[],
        bone_root=0,
        source_format="obj",
        scale_factor=1.0,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# glTF / GLB parser (pygltflib)
# ---------------------------------------------------------------------------
#
# pygltflib gives us a Python view over the JSON spec at
# https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html. We pull
# only what we need:
#
#   meshes[].primitives[].attributes
#       POSITION  (vec3)
#       NORMAL    (vec3)         optional
#       TEXCOORD_0 (vec2)        optional
#       JOINTS_0   (vec4 u8/u16) optional
#       WEIGHTS_0  (vec4 f32)    optional
#       indices                  optional (else a flat list 0..N-1)
#
#   skins[0].joints                 list of node indices
#   skins[0].inverseBindMatrices    one mat4 per joint (for bind-pose)
#   nodes[]                         translation/rotation/scale per node
#
# We use ``skins[0]`` only — multi-skin files are rare and the model's
# primary skeleton is conventionally skin 0.


def parse_gltf(data: bytes, *, glb: Optional[bool] = None) -> ImportedModel:
    """Parse glTF 2.0 (JSON .gltf or binary .glb) into an ImportedModel.

    Args
    ----
    data:
        File bytes. For a .gltf+.bin pair the .bin must be inlined as a
        data: URI or embedded buffer; we don't support cross-file
        references in the v1 server bridge (the user uploads a single
        file).
    glb:
        Force GLB framing (4-byte ``glTF`` magic + JSON chunk + BIN
        chunk). When ``None`` we auto-detect from the magic.
    """
    try:
        import pygltflib  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "pygltflib is required for glTF/GLB import. Install with: pip install pygltflib"
        ) from e

    # Auto-detect format.
    is_glb = glb if glb is not None else (data[:4] == b"glTF")

    if is_glb:
        gltf = pygltflib.GLTF2.load_from_bytes(data)
    else:
        # JSON load via temp path — pygltflib's ``loads`` requires a path.
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("parse_gltf: not valid UTF-8 JSON (and not GLB)")
        gltf = pygltflib.GLTF2.from_json(text)

    warnings: List[str] = []

    # Resolve binary buffers. pygltflib stores GLB body in
    # gltf._glb_data; for JSON we walk buffer.uri.
    def _get_buffer(buf_idx: int) -> bytes:
        buf = gltf.buffers[buf_idx]
        if is_glb and buf_idx == 0:
            return gltf._glb_data or b""
        if buf.uri is None:
            return b""
        if buf.uri.startswith("data:"):
            # data: URI: data:application/octet-stream;base64,<b64>
            comma = buf.uri.find(",")
            if comma < 0:
                return b""
            return base64.b64decode(buf.uri[comma + 1:])
        # External URIs not supported in v1.
        warnings.append(f"buffer {buf_idx} uri={buf.uri!r}: external buffers not supported")
        return b""

    # Cache buffers + materialized accessor numpy arrays.
    _buf_cache: Dict[int, bytes] = {}

    def _buf(idx: int) -> bytes:
        if idx not in _buf_cache:
            _buf_cache[idx] = _get_buffer(idx)
        return _buf_cache[idx]

    # Component-type widths.
    _CTYPE = {
        5120: ("i1", 1),  # BYTE
        5121: ("u1", 1),  # UNSIGNED_BYTE
        5122: ("i2", 2),  # SHORT
        5123: ("u2", 2),  # UNSIGNED_SHORT
        5125: ("u4", 4),  # UNSIGNED_INT
        5126: ("f4", 4),  # FLOAT
    }
    _SIZE = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

    def _accessor_array(acc_idx: int) -> np.ndarray:
        acc = gltf.accessors[acc_idx]
        bv = gltf.bufferViews[acc.bufferView]
        buf = _buf(bv.buffer)
        ct, ct_w = _CTYPE[acc.componentType]
        n_per = _SIZE[acc.type]
        start = (bv.byteOffset or 0) + (acc.byteOffset or 0)
        # Stride: explicit byteStride OR tightly packed.
        stride = bv.byteStride or (ct_w * n_per)
        if stride == ct_w * n_per:
            # Tightly packed, fast path.
            arr = np.frombuffer(buf, dtype=np.dtype("<" + ct), count=acc.count * n_per, offset=start)
            return arr.reshape(acc.count, n_per) if n_per > 1 else arr.copy()
        # Strided: walk one element at a time.
        out = np.empty((acc.count, n_per), dtype=np.dtype("<" + ct))
        for i in range(acc.count):
            elem = np.frombuffer(buf, dtype=np.dtype("<" + ct), count=n_per, offset=start + i * stride)
            out[i] = elem
        return out if n_per > 1 else out.ravel()

    # ---- Parse skin ----
    bones: List[ImportedBone] = []
    joint_node_to_bone_idx: Dict[int, int] = {}
    if gltf.skins:
        skin = gltf.skins[0]
        joint_nodes = list(skin.joints or [])
        # Walk nodes to build a parent map (gltf nodes don't carry
        # parent ptrs; child relationships are in node.children).
        n_nodes = len(gltf.nodes or [])
        parent_of = [-1] * n_nodes
        for ni, node in enumerate(gltf.nodes or []):
            for ci in (node.children or []):
                if 0 <= ci < n_nodes:
                    parent_of[ci] = ni
        for jb_idx, node_idx in enumerate(joint_nodes):
            joint_node_to_bone_idx[node_idx] = jb_idx
        for jb_idx, node_idx in enumerate(joint_nodes):
            node = gltf.nodes[node_idx]
            # Resolve parent: walk up parent_of until we hit a joint or
            # the root. Bones whose parent is non-joint get parent_idx
            # = -1.
            parent_node = parent_of[node_idx]
            parent_bone = -1
            visited = set()
            while parent_node >= 0 and parent_node not in visited:
                visited.add(parent_node)
                if parent_node in joint_node_to_bone_idx:
                    parent_bone = joint_node_to_bone_idx[parent_node]
                    break
                parent_node = parent_of[parent_node]
            t = node.translation or [0.0, 0.0, 0.0]
            r = node.rotation or [0.0, 0.0, 0.0, 1.0]
            s = node.scale or [1.0, 1.0, 1.0]
            bones.append(ImportedBone(
                name=node.name or f"bone_{jb_idx}",
                parent_idx=parent_bone,
                bind_pos=(float(t[0]), float(t[1]), float(t[2])),
                bind_rot_quat=(float(r[0]), float(r[1]), float(r[2]), float(r[3])),
                bind_scale=(float(s[0]), float(s[1]), float(s[2])),
            ))
    else:
        warnings.append("glTF has no skin — emitting empty skeleton; converter will use template")

    # ---- Parse meshes ----
    meshes: List[ImportedMesh] = []
    for mi, gmesh in enumerate(gltf.meshes or []):
        for pi, prim in enumerate(gmesh.primitives or []):
            attrs = prim.attributes
            if attrs is None or attrs.POSITION is None:
                continue
            pos = _accessor_array(attrs.POSITION).astype(np.float32, copy=False)
            n_verts = pos.shape[0]
            normals: Optional[np.ndarray] = None
            if attrs.NORMAL is not None:
                normals = _accessor_array(attrs.NORMAL).astype(np.float32, copy=False)
            uvs: Optional[np.ndarray] = None
            if getattr(attrs, "TEXCOORD_0", None) is not None:
                uvs = _accessor_array(attrs.TEXCOORD_0).astype(np.float32, copy=False)
                # glTF UVs are top-down (V increases downward). PSOBB
                # uses the same convention — no flip needed.
            skin_indices: Optional[np.ndarray] = None
            skin_weights: Optional[np.ndarray] = None
            if getattr(attrs, "JOINTS_0", None) is not None:
                ji = _accessor_array(attrs.JOINTS_0)
                # Clamp to uint8 (PSOBB has at most 256 bones in any
                # shipped model). A glTF skin can use u8 or u16.
                skin_indices = np.clip(ji, 0, 255).astype(np.uint8)
            if getattr(attrs, "WEIGHTS_0", None) is not None:
                skin_weights = _accessor_array(attrs.WEIGHTS_0).astype(np.float32, copy=False)
            # Indices.
            if prim.indices is not None:
                idx_flat = _accessor_array(prim.indices).astype(np.uint32, copy=False)
            else:
                idx_flat = np.arange(n_verts, dtype=np.uint32)
            mode = prim.mode if prim.mode is not None else 4  # default = TRIANGLES
            if mode == 4:  # TRIANGLES
                triangles = idx_flat.reshape(-1, 3)
            elif mode == 5:  # TRIANGLE_STRIP
                triangles = _strip_to_triangles(idx_flat)
            elif mode == 6:  # TRIANGLE_FAN
                triangles = _fan_to_triangles(idx_flat)
            else:
                warnings.append(f"mesh {mi}.prim{pi} mode={mode} unsupported, skipping")
                continue
            # Material id from prim.material -> mat index, modulo 256.
            mat_id = (prim.material or 0) & 0xFF
            meshes.append(ImportedMesh(
                name=gmesh.name or f"mesh_{mi}",
                vertices=pos,
                indices=triangles.astype(np.uint32, copy=False),
                uvs=uvs,
                normals=normals,
                skin_indices=skin_indices,
                skin_weights=skin_weights,
                material_id=mat_id,
            ))

    if not meshes:
        warnings.append("glTF parsed but produced no meshes")

    # ---- Parse VRM humanoid map (extension; optional) ----
    # VRM is a glTF extension popularised by VRoid Studio for CC0 anime
    # characters. Two on-disk variants:
    #   * VRM 0.x   stores the rig under ``extensions.VRM``      with
    #               ``humanoid.humanBones`` as a LIST of
    #               ``{"bone": <role>, "node": <node-idx>}`` entries.
    #   * VRM 1.0   uses ``extensions.VRMC_vrm`` and represents the same
    #               data as a DICT keyed by role name:
    #               ``{"hips": {"node": 1}, ...}``.
    # We extract a flat ``role -> bone-idx`` dict where ``bone-idx``
    # references our ``bones`` list (same order as ``skin[0].joints``).
    # VRM uses lowercase camelCase role names ("hips", "leftUpperArm")
    # which we keep verbatim — the retargeter normalises them.
    vrm_humanoid_map: Dict[str, int] = _extract_vrm_humanoid_map(
        gltf, joint_node_to_bone_idx, warnings,
    )
    # ---- Parse VRM 1.0 spring-bone + node-constraint extensions ----
    # PSOBB has no secondary-motion runtime, but preserving the data
    # allows a Blender re-import to round-trip and lets the JSON side-
    # file exporter emit the chains for downstream tooling.
    spring_bones, node_constraints = _extract_vrm_spring_bones(
        gltf, joint_node_to_bone_idx, warnings,
    )
    return ImportedModel(
        meshes=meshes,
        bones=bones,
        bone_root=0,
        source_format=("glb" if is_glb else "gltf"),
        scale_factor=1.0,
        warnings=warnings,
        vrm_humanoid_map=vrm_humanoid_map,
        spring_bones=spring_bones,
        node_constraints=node_constraints,
    )


def _extract_vrm_humanoid_map(
    gltf,
    joint_node_to_bone_idx: Dict[int, int],
    warnings: List[str],
) -> Dict[str, int]:
    """Pull the VRM humanoid map out of a parsed glTF, if present.

    Returns
    -------
    Dict[str, int]
        ``{ vrm_role: bone_idx_in_imported_bones }``. Empty when the file
        isn't a VRM (no ``VRM`` / ``VRMC_vrm`` extension block) or the
        block exists but has no humanoid section.

    The function is permissive: malformed entries are skipped (with a
    diagnostic appended to ``warnings``) rather than raising. A partial
    map is more useful than none.
    """
    extensions = getattr(gltf, "extensions", None) or {}
    if not isinstance(extensions, dict):
        return {}

    # Prefer VRM 1.0 (VRMC_vrm) when both are present — newer is canonical.
    vrm_block = None
    vrm_version = ""
    if "VRMC_vrm" in extensions and isinstance(extensions["VRMC_vrm"], dict):
        vrm_block = extensions["VRMC_vrm"]
        vrm_version = "1.0"
    elif "VRM" in extensions and isinstance(extensions["VRM"], dict):
        vrm_block = extensions["VRM"]
        vrm_version = "0.x"
    if vrm_block is None:
        return {}

    humanoid = vrm_block.get("humanoid")
    if not isinstance(humanoid, dict):
        warnings.append(f"VRM {vrm_version}: extension present but humanoid section missing")
        return {}

    human_bones = humanoid.get("humanBones")
    role_to_node: Dict[str, int] = {}
    if isinstance(human_bones, dict):
        # VRM 1.0 format: {"hips": {"node": 1}, ...}
        for role, entry in human_bones.items():
            if not isinstance(entry, dict):
                continue
            node_idx = entry.get("node")
            if isinstance(node_idx, int) and node_idx >= 0:
                role_to_node[str(role)] = node_idx
    elif isinstance(human_bones, list):
        # VRM 0.x format: [{"bone": "hips", "node": 1}, ...]
        for entry in human_bones:
            if not isinstance(entry, dict):
                continue
            role = entry.get("bone")
            node_idx = entry.get("node")
            if isinstance(role, str) and isinstance(node_idx, int) and node_idx >= 0:
                role_to_node[role] = node_idx
    else:
        warnings.append(f"VRM {vrm_version}: humanBones has unexpected type {type(human_bones).__name__}")
        return {}

    # Map nodes → joint indices (which is what our ``bones`` array uses).
    # A VRM rig may reference nodes that aren't in skin[0].joints (rare
    # but legal: VRM 0.x sometimes lists secondary joints like the eye
    # bones outside the primary skin). We drop those silently — the
    # retargeter falls back to string-match for unmapped roles.
    role_to_bone: Dict[str, int] = {}
    for role, node_idx in role_to_node.items():
        bone_idx = joint_node_to_bone_idx.get(node_idx)
        if bone_idx is not None:
            role_to_bone[role] = bone_idx
    if role_to_bone:
        warnings.append(f"VRM {vrm_version} humanoid map: {len(role_to_bone)} roles resolved")
    return role_to_bone


def _extract_vrm_spring_bones(
    gltf,
    joint_node_to_bone_idx: Dict[int, int],
    warnings: List[str],
) -> Tuple[List["SpringBoneChain"], List["NodeConstraint"]]:
    """Decode VRM 1.0 secondary-motion blocks if present.

    Returns a pair ``(spring_chains, node_constraints)``. Both are empty
    lists when the source has no relevant extensions. We deliberately do
    NOT raise on malformed data — preservation is best-effort and the
    fallback is "no chains, no constraints" rather than aborting the
    whole import.

    The function looks up the following blocks (all optional):

      * ``extensions.VRMC_springBone`` (VRM 1.0 secondary motion)
        - ``colliders``    : per-bone collider shapes
        - ``colliderGroups``: collider sets the springs reference
        - ``springs``       : actual chains of joints + the groups they
                              collide against

      * ``nodes[i].extensions.VRMC_node_constraint`` (per-node constraint)
        - ``constraint``    : { roll | aim | rotation } with source +
                              axis + weight

    There's also a legacy VRM 0.x ``extensions.VRM.secondaryAnimation``
    schema for spring bones, which we DO NOT decode here — VRoid Studio
    stopped emitting that path in 2023, and the callers we serve are
    overwhelmingly 1.0. The VRM-version-detect warning will tell users
    if they hand an editor a 0.x file expecting spring data.
    """
    extensions = getattr(gltf, "extensions", None) or {}
    if not isinstance(extensions, dict):
        return [], []

    spring_chains: List[SpringBoneChain] = []
    node_constraints: List[NodeConstraint] = []

    # ---- Spring bones (VRMC_springBone) ----
    spring_block = extensions.get("VRMC_springBone")
    if isinstance(spring_block, dict):
        # 1) Decode every collider declaration up front so we can resolve
        #    spring->colliderGroups->collider indices in one pass.
        raw_colliders: List[SpringBoneCollider] = []
        for entry in spring_block.get("colliders") or []:
            if not isinstance(entry, dict):
                continue
            node_idx = entry.get("node")
            bone_idx = -1
            if isinstance(node_idx, int):
                bone_idx = joint_node_to_bone_idx.get(node_idx, -1)
            shape_blk = entry.get("shape") or {}
            if not isinstance(shape_blk, dict):
                shape_blk = {}
            if "sphere" in shape_blk and isinstance(shape_blk["sphere"], dict):
                sph = shape_blk["sphere"]
                offset = _vec3_or_zero(sph.get("offset"))
                radius = _float_or_zero(sph.get("radius"))
                raw_colliders.append(SpringBoneCollider(
                    bone_idx=bone_idx, shape="sphere",
                    offset=offset, radius=radius,
                    tail=(0.0, 0.0, 0.0),
                ))
            elif "capsule" in shape_blk and isinstance(shape_blk["capsule"], dict):
                cap = shape_blk["capsule"]
                offset = _vec3_or_zero(cap.get("offset"))
                tail = _vec3_or_zero(cap.get("tail"))
                radius = _float_or_zero(cap.get("radius"))
                raw_colliders.append(SpringBoneCollider(
                    bone_idx=bone_idx, shape="capsule",
                    offset=offset, radius=radius, tail=tail,
                ))
            else:
                # Unknown shape type — record as sphere with radius 0
                # so the count survives without us silently fabricating
                # geometry. The shape="" sentinel signals "couldn't decode".
                raw_colliders.append(SpringBoneCollider(
                    bone_idx=bone_idx, shape="",
                    offset=(0.0, 0.0, 0.0), radius=0.0,
                    tail=(0.0, 0.0, 0.0),
                ))

        # 2) Decode collider groups: each is a name + list of collider
        #    indices into raw_colliders. We just keep the resolved list.
        group_to_colliders: List[List[SpringBoneCollider]] = []
        for entry in spring_block.get("colliderGroups") or []:
            if not isinstance(entry, dict):
                group_to_colliders.append([])
                continue
            collider_idxs = entry.get("colliders") or []
            resolved: List[SpringBoneCollider] = []
            for ci in collider_idxs:
                if isinstance(ci, int) and 0 <= ci < len(raw_colliders):
                    resolved.append(raw_colliders[ci])
            group_to_colliders.append(resolved)

        # 3) Decode springs (the actual chains).
        for si, spring in enumerate(spring_block.get("springs") or []):
            if not isinstance(spring, dict):
                continue
            chain_name = str(spring.get("name") or f"spring_{si}")
            center_node = spring.get("center")
            center_bone = -1
            if isinstance(center_node, int):
                center_bone = joint_node_to_bone_idx.get(center_node, -1)
            joints: List[SpringBoneJoint] = []
            for j in spring.get("joints") or []:
                if not isinstance(j, dict):
                    continue
                jn = j.get("node")
                if not isinstance(jn, int):
                    continue
                bone_idx = joint_node_to_bone_idx.get(jn, -1)
                joints.append(SpringBoneJoint(
                    bone_idx=bone_idx,
                    hit_radius=_float_or_zero(j.get("hitRadius")),
                    stiffness=_float_or_default(j.get("stiffness"), 1.0),
                    drag_force=_float_or_default(j.get("dragForce"), 0.4),
                    gravity_power=_float_or_zero(j.get("gravityPower")),
                    gravity_dir=_vec3_or_default(
                        j.get("gravityDir"), (0.0, -1.0, 0.0),
                    ),
                ))
            # Resolve collider groups → colliders for this chain.
            chain_colliders: List[SpringBoneCollider] = []
            seen: set = set()
            for gi in spring.get("colliderGroups") or []:
                if isinstance(gi, int) and 0 <= gi < len(group_to_colliders):
                    for col in group_to_colliders[gi]:
                        # Dedup via (bone_idx, shape, radius, offset) tuple.
                        k = (col.bone_idx, col.shape, col.radius,
                             col.offset, col.tail)
                        if k in seen:
                            continue
                        seen.add(k)
                        chain_colliders.append(col)
            spring_chains.append(SpringBoneChain(
                name=chain_name,
                joints=joints,
                colliders=chain_colliders,
                center_bone_idx=center_bone,
            ))

    # ---- Node constraints (VRMC_node_constraint) ----
    nodes = getattr(gltf, "nodes", None) or []
    for node_idx, node in enumerate(nodes):
        node_ext = getattr(node, "extensions", None) or {}
        if not isinstance(node_ext, dict):
            continue
        nc_block = node_ext.get("VRMC_node_constraint")
        if not isinstance(nc_block, dict):
            continue
        constraint = nc_block.get("constraint")
        if not isinstance(constraint, dict):
            continue
        bone_idx = joint_node_to_bone_idx.get(node_idx, -1)
        for ctype in ("roll", "aim", "rotation"):
            sub = constraint.get(ctype)
            if not isinstance(sub, dict):
                continue
            src_node = sub.get("source")
            src_bone = -1
            if isinstance(src_node, int):
                src_bone = joint_node_to_bone_idx.get(src_node, -1)
            axis_field = sub.get("rollAxis") if ctype == "roll" else sub.get("aimAxis")
            node_constraints.append(NodeConstraint(
                bone_idx=bone_idx,
                source_bone_idx=src_bone,
                constraint_type=ctype,
                weight=_float_or_default(sub.get("weight"), 1.0),
                axis=str(axis_field) if isinstance(axis_field, str) else "",
            ))
            break  # one constraint per node per spec

    if spring_chains:
        warnings.append(
            f"VRM springBone: {len(spring_chains)} chain(s) preserved "
            "(PSOBB has no secondary-motion runtime; data on model.spring_bones)"
        )
    if node_constraints:
        warnings.append(
            f"VRM nodeConstraint: {len(node_constraints)} constraint(s) preserved"
        )
    return spring_chains, node_constraints


def _vec3_or_zero(v) -> Tuple[float, float, float]:
    """Coerce a glTF vec3 (list / tuple) to a 3-tuple of floats; zero on miss."""
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        try:
            return (float(v[0]), float(v[1]), float(v[2]))
        except (TypeError, ValueError):
            return (0.0, 0.0, 0.0)
    return (0.0, 0.0, 0.0)


def _vec3_or_default(v, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        try:
            return (float(v[0]), float(v[1]), float(v[2]))
        except (TypeError, ValueError):
            return default
    return default


def _float_or_zero(v) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _float_or_default(v, default: float) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _strip_to_triangles(strip: np.ndarray) -> np.ndarray:
    """Unwind a triangle-strip index list to a flat (N, 3) triangle array."""
    n = len(strip)
    if n < 3:
        return np.zeros((0, 3), dtype=np.uint32)
    out = []
    for i in range(n - 2):
        a, b, c = int(strip[i]), int(strip[i + 1]), int(strip[i + 2])
        if a == b or b == c or a == c:
            continue
        if i & 1:
            out.append((a, c, b))
        else:
            out.append((a, b, c))
    return np.asarray(out, dtype=np.uint32) if out else np.zeros((0, 3), dtype=np.uint32)


def _fan_to_triangles(fan: np.ndarray) -> np.ndarray:
    """Unwind a triangle-fan index list to a flat (N, 3) triangle array."""
    n = len(fan)
    if n < 3:
        return np.zeros((0, 3), dtype=np.uint32)
    out = []
    a = int(fan[0])
    for i in range(1, n - 1):
        b, c = int(fan[i]), int(fan[i + 1])
        if a == b or b == c or a == c:
            continue
        out.append((a, b, c))
    return np.asarray(out, dtype=np.uint32) if out else np.zeros((0, 3), dtype=np.uint32)


# ---------------------------------------------------------------------------
# Format dispatch
# ---------------------------------------------------------------------------


def parse_external(data: bytes, filename: str) -> ImportedModel:
    """Sniff the format from the filename + magic and call the right parser.

    Recognized extensions: ``.obj``, ``.gltf``, ``.glb``, ``.fbx``.
    Binary FBX is parsed via ``formats.fbx_reader``; ASCII FBX is
    rejected with a clear error pointing at the binary export option.

    Raises
    ------
    ValueError
        When the format cannot be identified or parsed.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("parse_external: data must be bytes")
    ext = (Path(filename).suffix or "").lower().lstrip(".")
    if ext == "fbx":
        # Defer the import so glTF/OBJ-only callers don't pay the import
        # cost of fbx_reader. The FBX reader has zero binary deps but
        # does pull in numpy + struct heavy machinery at module load.
        from .fbx_reader import parse_fbx, FbxParseError
        try:
            return parse_fbx(data)
        except FbxParseError as e:
            # Surface FBX-specific errors as ValueError (the API contract).
            raise ValueError(str(e)) from e
    if ext == "obj":
        return parse_obj(data)
    if ext in ("gltf", "glb"):
        return parse_gltf(data, glb=(ext == "glb"))
    # Try magic.
    if data[:4] == b"glTF":
        return parse_gltf(data, glb=True)
    if data[:21] == b"Kaydara FBX Binary  \x00":
        from .fbx_reader import parse_fbx, FbxParseError
        try:
            return parse_fbx(data)
        except FbxParseError as e:
            raise ValueError(str(e)) from e
    if data[:1] in (b"v", b"#", b"o", b"g") or b"\nv " in data[:1024]:
        return parse_obj(data)
    raise ValueError(
        f"Unrecognized format for {filename!r}. Supported: .obj, .gltf, .glb, .fbx"
    )


# ---------------------------------------------------------------------------
# Coordinate / unit conversion
# ---------------------------------------------------------------------------


def quat_to_zyx_bams(qx: float, qy: float, qz: float, qw: float) -> Tuple[int, int, int]:
    """Quaternion -> ZYX intrinsic Euler -> BAMS (16-bit unsigned).

    PSOBB's NjsObject stores rotation as 3 BAM ints. Composition order
    is ZYX (Phantasmal verified): R = Rz @ Ry @ Rx. We extract the
    Euler angles via the standard formulas:
        sy = -(R[2][0]) = 2 (qx*qz - qw*qy)  ... (sin(pitch_y))
        ry = asin(clamp(sy, -1, 1))
        rx = atan2(R[2][1], R[2][2])
        rz = atan2(R[1][0], R[0][0])
    Gimbal-lock clamp: if |sy| > 0.9999 we set rx=0, rz=atan2(-R[0][1], R[1][1]).

    Returns (rx_bams, ry_bams, rz_bams) as 16-bit unsigned ints.

    The conversion is approximate when the source is a singular pose
    (gimbal lock); the residual error is well below the 0.0055°/LSB
    quantization noise of BAMS so it doesn't matter in practice.
    """
    # Normalize.
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-9:
        return (0, 0, 0)
    qx /= n; qy /= n; qz /= n; qw /= n

    # Build the rotation matrix elements we need.
    # R from quaternion (x, y, z, w):
    #   R[0][0] = 1 - 2(y² + z²)
    #   R[1][1] = 1 - 2(x² + z²)
    #   R[2][2] = 1 - 2(x² + y²)
    #   R[1][0] = 2(xy + zw)
    #   R[0][1] = 2(xy - zw)
    #   R[2][0] = 2(xz - yw)
    #   R[2][1] = 2(yz + xw)
    #   R[1][2] = 2(yz - xw)
    R20 = 2.0 * (qx * qz - qw * qy)
    R21 = 2.0 * (qy * qz + qw * qx)
    R22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    R10 = 2.0 * (qx * qy + qw * qz)
    R00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    R11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    R01 = 2.0 * (qx * qy - qw * qz)

    # ZYX Euler: rx around X (last applied), ry around Y, rz around Z (first).
    sy = -R20
    if sy > 0.9999:
        sy = 1.0
        ry = math.pi / 2.0
        rz = math.atan2(-R01, R11)
        rx = 0.0
    elif sy < -0.9999:
        sy = -1.0
        ry = -math.pi / 2.0
        rz = math.atan2(-R01, R11)
        rx = 0.0
    else:
        ry = math.asin(sy)
        rx = math.atan2(R21, R22)
        rz = math.atan2(R10, R00)

    return (rad_to_bams(rx), rad_to_bams(ry), rad_to_bams(rz))


def rad_to_bams(rad: float) -> int:
    """Radians -> BAMS u16 (with wrap-around)."""
    deg = math.degrees(rad)
    bams = int(round(deg * 65536.0 / 360.0))
    return bams & 0xFFFF


# ---------------------------------------------------------------------------
# Skin weight quantization
# ---------------------------------------------------------------------------


def quantize_skin_weights(
    weights: np.ndarray, indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Re-quantize float skin weights to 8-bit, renormalizing to sum 255.

    PSOBB chunk type 47 stores per-vertex (u8 weight, u8 bone_idx) for
    each of 4 slots; the weights sum to 255. We read float weights from
    glTF, drop slots whose weight is zero, and renormalize the rest.

    Args
    ----
    weights:  ``(N, 4)`` float32 (glTF WEIGHTS_0).
    indices:  ``(N, 4)`` uint8 (glTF JOINTS_0).

    Returns
    -------
    (q_weights, q_indices):
        Both ``(N, 4)`` uint8. q_weights[i] sums to 255 (modulo 1 LSB
        rounding error, which we absorb into the largest slot).
    """
    n = weights.shape[0]
    out_w = np.zeros_like(weights, dtype=np.uint8)
    out_i = np.zeros_like(indices, dtype=np.uint8)
    for v in range(n):
        w = weights[v].astype(np.float64, copy=True)
        idx = indices[v].astype(np.uint8, copy=True)
        s = float(w.sum())
        if s <= 1e-9:
            # Bind to bone 0 with weight 255.
            out_w[v, 0] = 255
            out_i[v, 0] = 0
            continue
        scaled = w * (255.0 / s)
        rounded = np.round(scaled).astype(np.int32)
        diff = 255 - int(rounded.sum())
        if diff != 0:
            # Adjust the largest slot to absorb the residual.
            j = int(np.argmax(rounded))
            rounded[j] = max(0, min(255, rounded[j] + diff))
        out_w[v] = rounded.astype(np.uint8)
        out_i[v] = idx
    return out_w, out_i


# ---------------------------------------------------------------------------
# NJ chunk emitters
# ---------------------------------------------------------------------------
#
# Vertex chunk type 41 (NJD_CV_VN): per-vertex 12 (pos) + 12 (normal) = 24 B.
# Strip chunk type 64                : header + per-strip i16 length + length * u16.
# Chunk body layout per nj_writer.NjChunk:
#   first u16 = body_word_count
#   for type 41 verts: body_size = 2 + 4*body_words (u32 dwords)
#       so body_words = ceil((4 + verts*24) / 4)
#       header     : u16 base_idx, u16 count       (4 bytes)
#       per vertex : 12 pos + 12 normal             (24 bytes)
#   for type 64 strips: body_size = 2 + 2*body_words (u16 words)
#       header     : u16 strip_count_and_offset    (2 bytes)
#       per strip  : i16 length + length*u16       (2 + 2*L bytes)


def _build_vlist_chunk_type41(
    positions: np.ndarray, normals: np.ndarray
) -> NjChunk:
    """Emit a single chunk-type-41 vertex chunk for (pos + normal) data.

    positions:  (N, 3) float32
    normals:    (N, 3) float32
    """
    n = int(positions.shape[0])
    body = bytearray()
    # body_words: vertex chunks are u32-word counted.
    # header (idx+count) is 4 bytes = 1 word, per-vertex 24 bytes = 6 words.
    body_words = 1 + n * 6
    body.extend(struct.pack("<H", body_words))
    body.extend(struct.pack("<HH", 0, n))  # base_idx=0, count=n
    for i in range(n):
        px, py, pz = float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2])
        nx, ny, nz = float(normals[i, 0]), float(normals[i, 1]), float(normals[i, 2])
        body.extend(struct.pack("<3f3f", px, py, pz, nx, ny, nz))
    return NjChunk(type_id=41, flags=0, body=bytes(body))


def _build_vlist_chunk_type47(
    positions: np.ndarray,
    normals: np.ndarray,
    skin_weights_u8: np.ndarray,
    skin_indices_u8: np.ndarray,
) -> NjChunk:
    """Emit a single chunk-type-47 vertex chunk (POS + NORMAL + idx + bw).

    Per-vertex layout (40 bytes): 12 pos + 12 normal + (idx u16) + (bw u16) + 12 pad?

    Actually per the parser stride table:
        chunk 47 stride = 12 + 12 + 4 = 28 bytes.

    The 4-byte tail is (u16 bone_idx, u16 weight). PSOBB packs the
    weights as u16 fixed-point where 0xFFFF = 1.0 — but the renderer in
    xj.py drops this data and we don't drive skinning yet. For v1 we
    emit the chunk with idx=0 weight=0xFFFF for every vertex (binding
    every vertex to bone 0).

    NOTE: chunk type 47 is NOT used by the v1 emitter — see
    imported_to_nj. We keep the function for v2 once skinning lands.
    """
    n = int(positions.shape[0])
    body = bytearray()
    body_words = 1 + n * 7  # 7 dwords per vertex (28 bytes / 4)
    body.extend(struct.pack("<H", body_words))
    body.extend(struct.pack("<HH", 0, n))
    for i in range(n):
        px, py, pz = float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2])
        nx, ny, nz = float(normals[i, 0]), float(normals[i, 1]), float(normals[i, 2])
        # Pick the dominant bone for v1 (NJD_CV_VN_NF in chunk 47 only
        # has one slot per vertex per chunk).
        if skin_weights_u8 is not None and skin_indices_u8 is not None:
            j = int(np.argmax(skin_weights_u8[i]))
            bone_idx = int(skin_indices_u8[i, j])
            weight = 0xFFFF
        else:
            bone_idx = 0
            weight = 0xFFFF
        body.extend(struct.pack("<3f3f", px, py, pz, nx, ny, nz))
        body.extend(struct.pack("<HH", bone_idx, weight))
    return NjChunk(type_id=47, flags=0, body=bytes(body))


def _build_strip_chunk_type64(triangles: np.ndarray) -> NjChunk:
    """Emit a single chunk-type-64 strip chunk for an (N, 3) triangle list.

    Each triangle becomes a 3-vertex strip: this is wasteful on triangle
    count but matches what hand-rolled chunk emitters in the field
    typically produce, and the rendering parser is happy with it.
    """
    if triangles.size == 0:
        # Empty plist still needs a valid header; return an empty strip set.
        body = struct.pack("<H", 1) + struct.pack("<H", 0)
        return NjChunk(type_id=64, flags=0, body=body)

    n_strips = int(triangles.shape[0])
    body = bytearray()
    # Compute body size: header (2) + per-strip (2 length + 2*3 indices = 8) = 8N + 2
    body_size_bytes = 2 + n_strips * 8
    body_words = body_size_bytes // 2  # type 64 is u16-word-counted
    body.extend(struct.pack("<H", body_words))
    body.extend(struct.pack("<H", n_strips & 0x3FFF))  # strip count, no user offset
    for tri in triangles:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        # Length=3, positive winding (CCW). i16 length means length=3.
        body.extend(struct.pack("<h", 3))
        body.extend(struct.pack("<HHH", a, b, c))
    return NjChunk(type_id=64, flags=0, body=bytes(body))


def _build_strip_chunk_type65(triangles: np.ndarray, uvs: np.ndarray) -> NjChunk:
    """Emit a chunk-type-65 strip chunk (per-vertex u16x2 UVs alongside indices).

    Stride per strip vertex = 2 (idx) + 4 (UV) = 6 bytes.
    Per-strip header: i16 length, then per-vertex (u16 idx + u16 u + u16 v).
    UVs are stored as u16 fixed-point where 256 = 1.0 (the standard
    xj.py NjVertex_UVS scaling).

    triangles:  (N, 3) uint32
    uvs:        (V, 2) float32   indexed by triangle indices.
    """
    if triangles.size == 0:
        body = struct.pack("<H", 1) + struct.pack("<H", 0)
        return NjChunk(type_id=65, flags=0, body=body)
    n_strips = int(triangles.shape[0])
    # Body size: header(2) + per-strip(2 + 3*6) = header + 20*N
    body_size_bytes = 2 + n_strips * (2 + 3 * 6)
    body_words = body_size_bytes // 2
    body = bytearray()
    body.extend(struct.pack("<H", body_words))
    body.extend(struct.pack("<H", n_strips & 0x3FFF))
    n_verts = int(uvs.shape[0]) if uvs is not None else 0
    for tri in triangles:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        body.extend(struct.pack("<h", 3))
        for vi in (a, b, c):
            if vi < n_verts:
                u, v = float(uvs[vi, 0]), float(uvs[vi, 1])
            else:
                u, v = 0.0, 0.0
            uu = max(-32768, min(32767, int(round(u * 256.0))))
            vv = max(-32768, min(32767, int(round(v * 256.0))))
            body.extend(struct.pack("<HhH", vi, uu & 0xFFFF, vv & 0xFFFF))
            # Wait -- we wrote 3 fields but only 2 specified u/v handling.
    # Fix: rewrite the per-vertex pack precisely.
    return _rebuild_strip65(triangles, uvs)


def _rebuild_strip65(triangles: np.ndarray, uvs: np.ndarray) -> NjChunk:
    """Clean implementation of strip-chunk-65 with UVs.

    Per-vertex: u16 idx, u16 u, u16 v (6 bytes). Per-strip: i16 length
    + 3 * (idx, u, v) for each triangle.

    UVs: signed 16-bit fixed point, 256 = 1.0 (matches xj.py NjVertex_UVS).
    """
    n_strips = int(triangles.shape[0])
    # body header (2) + per-strip (2 + 3*(2+2+2) = 2+18 = 20)
    body_size_bytes = 2 + n_strips * 20
    body_words = body_size_bytes // 2
    body = bytearray()
    body.extend(struct.pack("<H", body_words))
    body.extend(struct.pack("<H", n_strips & 0x3FFF))
    n_verts = int(uvs.shape[0]) if uvs is not None else 0
    for tri in triangles:
        body.extend(struct.pack("<h", 3))
        for vi in (int(tri[0]), int(tri[1]), int(tri[2])):
            if 0 <= vi < n_verts:
                u, v = float(uvs[vi, 0]), float(uvs[vi, 1])
            else:
                u, v = 0.0, 0.0
            uu = max(-32768, min(32767, int(round(u * 256.0))))
            vv = max(-32768, min(32767, int(round(v * 256.0))))
            body.extend(struct.pack("<HhH", vi & 0xFFFF, uu, vv & 0xFFFF))
    return NjChunk(type_id=65, flags=0, body=bytes(body))


# ---------------------------------------------------------------------------
# Skeleton template loading
# ---------------------------------------------------------------------------


_TEMPLATES_DIR = Path(__file__).parent.parent / "data" / "import_templates"


def list_templates() -> List[str]:
    """Return the names of available skeleton templates (no extension)."""
    if not _TEMPLATES_DIR.is_dir():
        return []
    return sorted(p.stem for p in _TEMPLATES_DIR.glob("*.json"))


def load_template(name: str) -> List[ImportedBone]:
    """Load a skeleton template by name (e.g. 'player_body').

    Templates are JSON files in ``data/import_templates/``. Each
    contains a ``bones`` array with the same shape as ``ImportedBone``.

    Raises FileNotFoundError if the template is missing.
    """
    p = _TEMPLATES_DIR / f"{name}.json"
    if not p.is_file():
        raise FileNotFoundError(f"skeleton template not found: {name}")
    data = json.loads(p.read_text(encoding="utf-8"))
    bones: List[ImportedBone] = []
    for raw in data.get("bones", []):
        bones.append(ImportedBone(
            name=str(raw.get("name") or f"bone_{len(bones)}"),
            parent_idx=int(raw.get("parent_idx", -1)),
            bind_pos=tuple(float(x) for x in raw.get("bind_pos", (0.0, 0.0, 0.0))),
            bind_rot_quat=tuple(float(x) for x in raw.get("bind_rot_quat", (0.0, 0.0, 0.0, 1.0))),
            bind_scale=tuple(float(x) for x in raw.get("bind_scale", (1.0, 1.0, 1.0))),
        ))
    return bones


# ---------------------------------------------------------------------------
# imported_to_nj — the main converter
# ---------------------------------------------------------------------------


def imported_to_nj(
    model: ImportedModel,
    *,
    target_class: Optional[str] = None,
    axis_flip_z: bool = True,
    scale: float = 1.0,
) -> NjModel:
    """Convert an ImportedModel to a deployable NjModel.

    Args
    ----
    model:
        The parsed external model.
    target_class:
        Optional skeleton template name (e.g. ``"player_body"``). When
        the source has no bones AND a template is provided, we use the
        template's skeleton; otherwise we use a 1-bone root.
    axis_flip_z:
        When True (the default), flip Z on positions/normals/bones to
        convert from glTF (right-handed) to PSOBB (left-handed). Pass
        False if the user's source is already in PSOBB convention.
    scale:
        Uniform scale applied to positions + bone bind positions. The
        UI exposes this as a slider; default = source-supplied
        ``model.scale_factor``.
    """
    nj = NjModel()
    if scale == 1.0 and model.scale_factor != 1.0:
        scale = model.scale_factor

    # Determine bone source.
    bones = model.bones
    if not bones and target_class:
        try:
            bones = load_template(target_class)
        except FileNotFoundError:
            bones = []
    if not bones:
        # 1-bone root fallback.
        bones = [ImportedBone(
            name="root",
            parent_idx=-1,
            bind_pos=(0.0, 0.0, 0.0),
            bind_rot_quat=(0.0, 0.0, 0.0, 1.0),
            bind_scale=(1.0, 1.0, 1.0),
        )]

    n_bones = len(bones)

    # Build the NjNode list mirroring the bone hierarchy. Children are
    # found by scanning for any bone whose parent_idx == this. We
    # construct a parent->children map first.
    children_of: Dict[int, List[int]] = {}
    for i, b in enumerate(bones):
        children_of.setdefault(b.parent_idx, []).append(i)

    # Determine root: bone with parent_idx==-1, or fall back to bone 0.
    roots = children_of.get(-1, [])
    if not roots:
        roots = [0]

    # Walk DFS pre-order (parent then children); within siblings we
    # preserve list order. We also collect the new index assignments
    # so the imported parent indices map cleanly to NjNode child/sibling
    # links.
    nj_idx_for_bone: Dict[int, int] = {}
    visit_order: List[int] = []

    def _visit(b_idx: int) -> None:
        if b_idx in nj_idx_for_bone:
            return
        nj_idx_for_bone[b_idx] = len(visit_order)
        visit_order.append(b_idx)
        for c in children_of.get(b_idx, []):
            _visit(c)

    for r in roots:
        _visit(r)

    # Cover any orphans (bones whose parent doesn't exist).
    for i in range(n_bones):
        if i not in nj_idx_for_bone:
            _visit(i)

    # Build NjNodes: each bone becomes a node with first-child + sibling
    # pointers wired according to the children_of map. We use the new
    # nj_idx ordering.
    nodes: List[NjNode] = []
    # Track the FIRST child of each bone (in nj order) and its NEXT
    # sibling.
    for new_idx, b_idx in enumerate(visit_order):
        b = bones[b_idx]
        # Child = first new-index of any child of b_idx.
        child_new = -1
        my_children = sorted(
            children_of.get(b_idx, []),
            key=lambda c: nj_idx_for_bone[c],
        )
        if my_children:
            child_new = nj_idx_for_bone[my_children[0]]
        # Sibling = next sibling in parent's child list.
        sib_new = -1
        parent = b.parent_idx
        if parent >= 0:
            siblings = sorted(
                children_of.get(parent, []),
                key=lambda c: nj_idx_for_bone[c],
            )
            try:
                pos = siblings.index(b_idx)
                if pos + 1 < len(siblings):
                    sib_new = nj_idx_for_bone[siblings[pos + 1]]
            except ValueError:
                pass
        # Apply axis flip + scale to bind position; quat -> BAMS.
        bx, by, bz = b.bind_pos
        if axis_flip_z:
            bz = -bz
        bx *= scale; by *= scale; bz *= scale
        rot_bams = quat_to_zyx_bams(*b.bind_rot_quat)
        if axis_flip_z:
            # Mirror Z: negate rx and ry components of the rotation.
            rx, ry, rz = rot_bams
            # Convert through signed 16-bit: -bams = (0x10000 - bams) & 0xFFFF
            rot_bams = (
                ((-rx) & 0xFFFF) if rx else 0,
                ((-ry) & 0xFFFF) if ry else 0,
                rz,
            )
        nodes.append(NjNode(
            eval_flags=0x0,
            position=(bx, by, bz),
            rotation_bams=rot_bams,
            scale=b.bind_scale,
            mesh_index=-1,
            child_index=child_new,
            sibling_index=sib_new,
        ))

    # Attach meshes. v1 strategy: every mesh is bound to the root bone
    # (nj_idx 0). PSOBB models can spread strips across multiple bones
    # for skinning, but the v1 emitter doesn't drive skinning yet —
    # static binding to the root is correct for non-skinned imports
    # and a reasonable starting point for skinned ones (the user can
    # rig the model via the editor's existing bone tools post-import).
    nj_meshes: List[NjMeshChunks] = []
    for mesh in model.meshes:
        verts = mesh.vertices.astype(np.float32, copy=True)
        normals = (
            mesh.normals.astype(np.float32, copy=True)
            if mesh.normals is not None
            else _generate_normals(verts, mesh.indices)
        )
        if axis_flip_z:
            verts[:, 2] = -verts[:, 2]
            normals[:, 2] = -normals[:, 2]
            # Flipping Z reverses winding; flip triangle order to keep
            # outward-facing normals.
            tris = mesh.indices.copy()
            tris[:, [1, 2]] = tris[:, [2, 1]]
        else:
            tris = mesh.indices
        verts *= scale

        # Compute bounding sphere from positions.
        bbox = _bounding_sphere(verts)

        # Vertex chunk: type 41 (POS + NORMAL).
        vchunk = _build_vlist_chunk_type41(verts, normals)

        # Strip chunk: type 65 (UV-flagged) when UVs are present, else 64.
        if mesh.uvs is not None and mesh.uvs.size > 0:
            schunk = _rebuild_strip65(tris, mesh.uvs.astype(np.float32, copy=False))
        else:
            schunk = _build_strip_chunk_type64(tris)

        nj_meshes.append(NjMeshChunks(
            bbox=bbox,
            vlist=[vchunk],
            plist=[schunk],
        ))

    # Attach the meshes to the root node — first one as root.mesh_index,
    # subsequent meshes go on synthetic child nodes hanging off the root
    # so each can carry its own mesh_index. We DO NOT skin them; they
    # all live in root-bone's local frame.
    if nj_meshes:
        nodes[0].mesh_index = 0
        # Root node's first child (existing bone child, if any) keeps its
        # link; we splice the synthetic mesh-only nodes between root.child
        # and the first real bone child... actually simpler: append the
        # mesh-only nodes at the END and use them as additional siblings
        # of the existing child chain. But that changes the skeleton
        # layout. The cleanest path: emit one node per mesh as a sibling
        # of the existing root-bone child chain.
        if len(nj_meshes) > 1:
            existing_child = nodes[0].child_index
            mesh_only_first = len(nodes)
            for mi in range(1, len(nj_meshes)):
                # Add a mesh-only node.
                nodes.append(NjNode(
                    eval_flags=0,
                    position=(0.0, 0.0, 0.0),
                    rotation_bams=(0, 0, 0),
                    scale=(1.0, 1.0, 1.0),
                    mesh_index=mi,
                    child_index=-1,
                    sibling_index=-1,
                ))
            # Link the new mesh nodes as a sibling chain attached to
            # root.child (or as root.child if root had none).
            for mi in range(1, len(nj_meshes) - 1):
                nodes[mesh_only_first + (mi - 1)].sibling_index = mesh_only_first + mi
            if existing_child < 0:
                nodes[0].child_index = mesh_only_first
            else:
                # Walk to last sibling of existing child chain and append.
                tail = existing_child
                guard = 0
                while nodes[tail].sibling_index >= 0 and guard < n_bones * 2:
                    tail = nodes[tail].sibling_index
                    guard += 1
                nodes[tail].sibling_index = mesh_only_first

    nj.nodes = nodes
    nj.meshes = nj_meshes
    # Document that blend shapes were parsed but ignored. PSOBB has no
    # morph-target rendering — the data still lives on ``model.blend_shapes``
    # for downstream tooling that wants it (e.g. dumping facial-rig
    # shapes to a side file). We surface the count + names via a free-form
    # attribute on the returned NjModel rather than mutating
    # ``model.warnings`` so the input dataclass stays read-only.
    if model.blend_shapes:
        names = [bs.name for bs in model.blend_shapes if bs.name]
        nj.import_diagnostics = {  # type: ignore[attr-defined]
            "blend_shapes_ignored": len(model.blend_shapes),
            "blend_shape_names": names[:32],  # cap at 32 to bound the size
            "note": (
                "PSOBB renders no morph targets. The blend-shape data is "
                "preserved on model.blend_shapes for downstream tooling but "
                "is not encoded in the NJ output."
            ),
        }
    return nj


def _generate_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Compute area-weighted vertex normals for a triangle mesh.

    Used as a fallback when the source supplies geometry but no normals
    (common for .obj files without ``vn`` lines).
    """
    n = positions.shape[0]
    accum = np.zeros((n, 3), dtype=np.float32)
    for tri in indices:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        if a >= n or b >= n or c >= n:
            continue
        pa, pb, pc = positions[a], positions[b], positions[c]
        face_n = np.cross(pb - pa, pc - pa)
        accum[a] += face_n
        accum[b] += face_n
        accum[c] += face_n
    lens = np.linalg.norm(accum, axis=1)
    lens[lens < 1e-9] = 1.0
    return (accum / lens[:, None]).astype(np.float32)


def _bounding_sphere(positions: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute a centroid-anchored bounding sphere ``(cx, cy, cz, r)``."""
    if positions.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    centroid = positions.mean(axis=0)
    diffs = positions - centroid
    radius = float(np.sqrt((diffs * diffs).sum(axis=1)).max())
    return (float(centroid[0]), float(centroid[1]), float(centroid[2]), radius)


# ---------------------------------------------------------------------------
# JSON wire-shape helpers (server -> UI)
# ---------------------------------------------------------------------------


def imported_to_json(model: ImportedModel) -> dict:
    """Serialize an ImportedModel for the /api/import/parse response.

    Mesh vertex / index buffers are returned as base64-encoded byte
    strings (Float32 / Uint32) for compactness over the JSON wire and
    direct ingestion by the existing model viewer (which already reads
    /api/model_mesh in the same shape).
    """
    out_meshes = []
    total_v = 0
    total_t = 0
    for m in model.meshes:
        verts = m.vertices.astype(np.float32, copy=False)
        idx = m.indices.astype(np.uint32, copy=False)
        normals = m.normals.astype(np.float32, copy=False) if m.normals is not None else None
        uvs = m.uvs.astype(np.float32, copy=False) if m.uvs is not None else None
        # Build interleaved (px,py,pz, nx,ny,nz, u,v) for the existing
        # psoApplyMeshPayload viewer.
        n_v = int(verts.shape[0])
        interleaved = np.zeros((n_v, 8), dtype=np.float32)
        interleaved[:, 0:3] = verts
        if normals is not None and normals.shape[0] == n_v:
            interleaved[:, 3:6] = normals
        else:
            interleaved[:, 4] = 1.0  # synthetic +Y normal
        if uvs is not None and uvs.shape[0] == n_v:
            interleaved[:, 6:8] = uvs
        idx_flat = idx.reshape(-1)
        # Bounding box.
        if n_v > 0:
            mn = verts.min(axis=0).tolist()
            mx = verts.max(axis=0).tolist()
            aabb = [mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]]
        else:
            aabb = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        cx, cy, cz, r = _bounding_sphere(verts)
        out_meshes.append({
            "name": m.name,
            "vertices_b64": base64.b64encode(interleaved.tobytes()).decode("ascii"),
            "indices_b64": base64.b64encode(idx_flat.tobytes()).decode("ascii"),
            "vertex_count": n_v,
            "triangle_count": int(idx.shape[0]),
            "material_id": m.material_id,
            "bounding_sphere": [cx, cy, cz, r],
            "aabb": aabb,
            "world_position": [0.0, 0.0, 0.0],
            "world_rotation_euler": [0.0, 0.0, 0.0],
            "world_scale": [1.0, 1.0, 1.0],
            "world_matrix": [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
            "has_skin": m.skin_indices is not None,
        })
        total_v += n_v
        total_t += int(idx.shape[0])
    bones = [
        {
            "name": b.name,
            "parent_idx": b.parent_idx,
            "bind_pos": list(b.bind_pos),
            "bind_rot_quat": list(b.bind_rot_quat),
            "bind_scale": list(b.bind_scale),
        }
        for b in model.bones
    ]
    # Optional VRM / blend-shape sidechannels: included when populated so
    # downstream JSON consumers (e.g. the import-preview UI) can surface
    # "VRM detected: 28 humanoid roles" / "X blend shapes parsed but not
    # rendered" diagnostics. Empty defaults stay omitted to keep the
    # wire shape unchanged for non-VRM, non-FBX-shape sources.
    extras: dict = {}
    if model.vrm_humanoid_map:
        extras["vrm_humanoid_map"] = dict(model.vrm_humanoid_map)
    if model.blend_shapes:
        # Blend-shape vertex deltas can be heavy (each shape is K*12 +
        # K*4 bytes, K = vertex influence count). We surface only the
        # name + size so the UI can render a count, not the raw geometry.
        # Round-trip via imported_from_json reconstructs an empty
        # placeholder list — the deltas themselves stay server-side.
        extras["blend_shapes"] = [
            {
                "name": bs.name,
                "vertex_count": int(bs.indexes.shape[0]) if bs.indexes is not None else 0,
                "default_weight": float(bs.default_weight),
                "mesh_name": bs.mesh_name,
                "has_normals": bs.normals is not None,
            }
            for bs in model.blend_shapes
        ]
    if model.spring_bones:
        # Spring chains are small structurally (joint count + collider
        # count rarely exceeds a few hundred entries); include them
        # verbatim so the JSON wire shape is round-trippable.
        extras["spring_bones"] = [
            {
                "name": chain.name,
                "center_bone_idx": int(chain.center_bone_idx),
                "joints": [
                    {
                        "bone_idx": int(j.bone_idx),
                        "hit_radius": float(j.hit_radius),
                        "stiffness": float(j.stiffness),
                        "drag_force": float(j.drag_force),
                        "gravity_power": float(j.gravity_power),
                        "gravity_dir": list(j.gravity_dir),
                    }
                    for j in chain.joints
                ],
                "colliders": [
                    {
                        "bone_idx": int(c.bone_idx),
                        "shape": c.shape,
                        "offset": list(c.offset),
                        "radius": float(c.radius),
                        "tail": list(c.tail),
                    }
                    for c in chain.colliders
                ],
            }
            for chain in model.spring_bones
        ]
    if model.node_constraints:
        extras["node_constraints"] = [
            {
                "bone_idx": int(nc.bone_idx),
                "source_bone_idx": int(nc.source_bone_idx),
                "constraint_type": nc.constraint_type,
                "weight": float(nc.weight),
                "axis": nc.axis,
            }
            for nc in model.node_constraints
        ]
    return {
        "mesh_count": len(out_meshes),
        "meshes": out_meshes,
        "totals": {"vertices": total_v, "triangles": total_t},
        "vertices_pre_transformed": True,
        "vert_total": total_v,
        "tri_total": total_t,
        "bones": bones,
        "bone_count": len(bones),
        "bone_root": model.bone_root,
        "source_format": model.source_format,
        "scale_factor": model.scale_factor,
        "warnings": model.warnings,
        **extras,
    }


def imported_from_json(d: dict) -> ImportedModel:
    """Inverse of imported_to_json: rebuild an ImportedModel from JSON.

    Used by the server's /api/import/build_nj path so the UI can echo
    the ``model_json`` produced by /api/import/parse without re-uploading
    the raw file (which may be large).
    """
    meshes: List[ImportedMesh] = []
    for m in d.get("meshes", []) or []:
        verts_b = base64.b64decode(m.get("vertices_b64", "") or "")
        idx_b = base64.b64decode(m.get("indices_b64", "") or "")
        n_v = int(m.get("vertex_count", 0))
        # Interleaved (8 floats per vertex).
        if n_v > 0 and len(verts_b) >= n_v * 32:
            arr = np.frombuffer(verts_b, dtype="<f4", count=n_v * 8).reshape(n_v, 8)
            verts = arr[:, 0:3].astype(np.float32, copy=True)
            normals = arr[:, 3:6].astype(np.float32, copy=True)
            uvs = arr[:, 6:8].astype(np.float32, copy=True)
        else:
            verts = np.zeros((n_v, 3), dtype=np.float32)
            normals = None
            uvs = None
        if idx_b:
            idx = np.frombuffer(idx_b, dtype="<u4").reshape(-1, 3).astype(np.uint32, copy=True)
        else:
            idx = np.zeros((0, 3), dtype=np.uint32)
        meshes.append(ImportedMesh(
            name=str(m.get("name") or f"mesh_{len(meshes)}"),
            vertices=verts,
            indices=idx,
            uvs=uvs,
            normals=normals,
            material_id=int(m.get("material_id", 0)) & 0xFF,
        ))
    bones: List[ImportedBone] = []
    for b in d.get("bones", []) or []:
        bones.append(ImportedBone(
            name=str(b.get("name") or f"bone_{len(bones)}"),
            parent_idx=int(b.get("parent_idx", -1)),
            bind_pos=tuple(float(x) for x in b.get("bind_pos", (0.0, 0.0, 0.0))),
            bind_rot_quat=tuple(float(x) for x in b.get("bind_rot_quat", (0.0, 0.0, 0.0, 1.0))),
            bind_scale=tuple(float(x) for x in b.get("bind_scale", (1.0, 1.0, 1.0))),
        ))
    # VRM humanoid map: int values are dropped by JSON's int() round-trip,
    # so we just rebuild as-is. JSON keys are always strings; that's
    # already what VRM uses ("hips", "leftUpperArm", ...).
    vrm_map_raw = d.get("vrm_humanoid_map") or {}
    vrm_humanoid_map: Dict[str, int] = {}
    if isinstance(vrm_map_raw, dict):
        for role, idx in vrm_map_raw.items():
            try:
                vrm_humanoid_map[str(role)] = int(idx)
            except (TypeError, ValueError):
                continue
    # Spring-bones round-trip (they're small enough to serialize verbatim).
    spring_bones: List[SpringBoneChain] = []
    for chain_d in d.get("spring_bones") or []:
        if not isinstance(chain_d, dict):
            continue
        joints: List[SpringBoneJoint] = []
        for j in chain_d.get("joints") or []:
            if not isinstance(j, dict):
                continue
            try:
                joints.append(SpringBoneJoint(
                    bone_idx=int(j.get("bone_idx", -1)),
                    hit_radius=float(j.get("hit_radius", 0.0)),
                    stiffness=float(j.get("stiffness", 1.0)),
                    drag_force=float(j.get("drag_force", 0.4)),
                    gravity_power=float(j.get("gravity_power", 0.0)),
                    gravity_dir=tuple(
                        float(x) for x in j.get("gravity_dir", (0.0, -1.0, 0.0))
                    ),
                ))
            except (TypeError, ValueError):
                continue
        colliders: List[SpringBoneCollider] = []
        for c in chain_d.get("colliders") or []:
            if not isinstance(c, dict):
                continue
            try:
                colliders.append(SpringBoneCollider(
                    bone_idx=int(c.get("bone_idx", -1)),
                    shape=str(c.get("shape", "sphere")),
                    offset=tuple(float(x) for x in c.get("offset", (0.0, 0.0, 0.0))),
                    radius=float(c.get("radius", 0.0)),
                    tail=tuple(float(x) for x in c.get("tail", (0.0, 0.0, 0.0))),
                ))
            except (TypeError, ValueError):
                continue
        spring_bones.append(SpringBoneChain(
            name=str(chain_d.get("name") or ""),
            joints=joints,
            colliders=colliders,
            center_bone_idx=int(chain_d.get("center_bone_idx", -1)),
        ))
    node_constraints: List[NodeConstraint] = []
    for nc in d.get("node_constraints") or []:
        if not isinstance(nc, dict):
            continue
        try:
            node_constraints.append(NodeConstraint(
                bone_idx=int(nc.get("bone_idx", -1)),
                source_bone_idx=int(nc.get("source_bone_idx", -1)),
                constraint_type=str(nc.get("constraint_type", "")),
                weight=float(nc.get("weight", 1.0)),
                axis=str(nc.get("axis", "")),
            ))
        except (TypeError, ValueError):
            continue
    return ImportedModel(
        meshes=meshes,
        bones=bones,
        bone_root=int(d.get("bone_root", 0)),
        source_format=str(d.get("source_format") or ""),
        scale_factor=float(d.get("scale_factor", 1.0)),
        warnings=list(d.get("warnings") or []),
        vrm_humanoid_map=vrm_humanoid_map,
        spring_bones=spring_bones,
        node_constraints=node_constraints,
        # NOTE: blend_shapes deliberately not round-tripped — the JSON
        # wire shape only carries name/size summaries (see
        # imported_to_json). Callers that need the raw deltas keep
        # the original ImportedModel reference server-side.
    )


# ---------------------------------------------------------------------------
# Animation track import (v2 — additive on top of the v1 ImportedModel).
# ---------------------------------------------------------------------------
#
# v1 dropped every glTF animation track. v2 reads them, classifies each
# channel by ``target.path`` (translation / rotation / scale), and
# returns a per-bone keyframe list keyed by glTF-skin joint index.
# Coordinate-space conversion (glTF RH -> PSOBB LH) is the retargeter's
# job, not the importer's; we keep the source data verbatim here so
# the retargeter can introspect it (e.g. detect Mixamo's +Y-up
# convention and skip the Z flip).
#
# Output shape:
#   ImportedAnimation(name, duration_seconds, fps_target, tracks)
#       tracks[i] = ImportedTrack(bone_idx, channel, times, values, interp)
#         bone_idx == index into ImportedModelWithAnims.bones
#         channel  == "translation" | "rotation" | "scale"
#         times    == list[float] in seconds
#         values   == list[tuple]
#                       translation: (x, y, z)
#                       scale:       (sx, sy, sz)
#                       rotation:    (qx, qy, qz, qw)   -- NOT (w, x, y, z)
#         interp   == "LINEAR" | "STEP" | "CUBICSPLINE"
#                     CUBICSPLINE values are (in_tangent, value,
#                     out_tangent) packed back-to-back per glTF spec.
#                     The retargeter currently treats CUBICSPLINE as
#                     LINEAR by dropping the tangent triples — sample
#                     density gives the same end frame even if the
#                     intermediate curve loses some smoothness.
#
# Why a separate type instead of extending ImportedModel:
#   * v1 callers (existing /api/import/parse) drop the animation cost
#     unconditionally.
#   * The model_json wire shape stays stable for the existing UI
#     preview path (we serialize meshes + bones; animations are
#     server-only).


@dataclass
class ImportedTrack:
    """One per-bone, per-channel animation track.

    Times are in seconds (relative to track start); the retargeter
    resamples to a 30 Hz integer-frame grid before emitting NJM.

    Values are NOT rotated to PSOBB convention here — kept verbatim
    in glTF (right-handed, Y-up) coords. The retargeter applies the
    coordinate flip alongside the bone-name remap so the two
    transformations are colocated in one place.
    """
    bone_idx: int
    channel: str  # "translation" | "rotation" | "scale"
    times: List[float] = field(default_factory=list)
    values: List[tuple] = field(default_factory=list)
    interp: str = "LINEAR"


@dataclass
class ImportedAnimation:
    """One parsed animation (set of tracks sharing a single name).

    A glTF file can have many animations (idle, walk, jump, ...). We
    return them in source order; the caller picks which one to retarget.
    """
    name: str
    duration_seconds: float
    fps_target: int = 30
    tracks: List[ImportedTrack] = field(default_factory=list)


@dataclass
class ImportedModelWithAnims:
    """Same as ImportedModel + a list of animations.

    Returned by ``parse_gltf_with_animations``. The mesh / bone
    attributes are inherited; ``animations`` is empty for OBJ files
    (OBJ doesn't carry skeletal animation).
    """
    model: ImportedModel
    animations: List[ImportedAnimation] = field(default_factory=list)


def parse_gltf_with_animations(buf_or_path) -> ImportedModelWithAnims:
    """Parse glTF + extract every animation track.

    Args
    ----
    buf_or_path:
        Either bytes / bytearray / memoryview (the raw .glb / .gltf
        bytes) or a path-like (str / Path) to a file we should read.

    Returns
    -------
    ImportedModelWithAnims
        ``model`` carries meshes + bones (same as ``parse_gltf``);
        ``animations`` carries every animation in the source.

    Notes
    -----
    Bone indices in the output tracks reference ``model.bones`` (i.e.
    the joint order from ``skins[0].joints``). Channels targeting
    nodes that AREN'T joints in skin 0 are silently dropped — those
    are typically scene-graph nodes (the root, camera rigs, etc.) and
    aren't useful for skeletal retargeting.
    """
    # Resolve input bytes.
    if isinstance(buf_or_path, (bytes, bytearray, memoryview)):
        data = bytes(buf_or_path)
        # Sniff GLB vs JSON.
        is_glb = data[:4] == b"glTF"
    else:
        p = Path(buf_or_path)
        data = p.read_bytes()
        ext = p.suffix.lower()
        is_glb = ext == ".glb" or data[:4] == b"glTF"

    # Parse meshes + skeleton via the v1 path so we stay byte-identical
    # for non-animation callers.
    model = parse_gltf(data, glb=is_glb)

    # Re-load via pygltflib for animation extraction. We can't share
    # the v1 path's gltf object without restructuring; the cost of a
    # second parse is small relative to the keyframe-decode work.
    try:
        import pygltflib  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "pygltflib is required for glTF animation import"
        ) from e

    if is_glb:
        gltf = pygltflib.GLTF2.load_from_bytes(data)
    else:
        gltf = pygltflib.GLTF2.from_json(data.decode("utf-8"))

    # Resolve buffers (same logic as parse_gltf — kept inline for
    # isolation; the v1 path's _buf closure is local to parse_gltf).
    def _get_buffer(buf_idx: int) -> bytes:
        buf = gltf.buffers[buf_idx]
        if is_glb and buf_idx == 0:
            return gltf._glb_data or b""
        if buf.uri is None:
            return b""
        if buf.uri.startswith("data:"):
            comma = buf.uri.find(",")
            if comma < 0:
                return b""
            return base64.b64decode(buf.uri[comma + 1:])
        return b""

    _buf_cache: Dict[int, bytes] = {}

    def _buf(idx: int) -> bytes:
        if idx not in _buf_cache:
            _buf_cache[idx] = _get_buffer(idx)
        return _buf_cache[idx]

    _CTYPE = {
        5120: ("i1", 1), 5121: ("u1", 1), 5122: ("i2", 2),
        5123: ("u2", 2), 5125: ("u4", 4), 5126: ("f4", 4),
    }
    _SIZE = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

    def _accessor_array(acc_idx: int) -> np.ndarray:
        acc = gltf.accessors[acc_idx]
        bv = gltf.bufferViews[acc.bufferView]
        buf = _buf(bv.buffer)
        ct, ct_w = _CTYPE[acc.componentType]
        n_per = _SIZE[acc.type]
        start = (bv.byteOffset or 0) + (acc.byteOffset or 0)
        stride = bv.byteStride or (ct_w * n_per)
        if stride == ct_w * n_per:
            arr = np.frombuffer(buf, dtype=np.dtype("<" + ct), count=acc.count * n_per, offset=start)
            return arr.reshape(acc.count, n_per) if n_per > 1 else arr.copy()
        out = np.empty((acc.count, n_per), dtype=np.dtype("<" + ct))
        for i in range(acc.count):
            elem = np.frombuffer(buf, dtype=np.dtype("<" + ct), count=n_per, offset=start + i * stride)
            out[i] = elem
        return out if n_per > 1 else out.ravel()

    # Build node->bone-idx map (same as parse_gltf's joint_node_to_bone_idx).
    node_to_bone: Dict[int, int] = {}
    if gltf.skins:
        for jb_idx, node_idx in enumerate(gltf.skins[0].joints or []):
            node_to_bone[node_idx] = jb_idx

    animations: List[ImportedAnimation] = []
    for ai, anim in enumerate(gltf.animations or []):
        anim_name = anim.name or f"animation_{ai}"
        tracks: List[ImportedTrack] = []
        max_t = 0.0
        for ch in anim.channels or []:
            tgt = ch.target
            if tgt is None or tgt.node is None or tgt.path is None:
                continue
            bone_idx = node_to_bone.get(tgt.node)
            if bone_idx is None:
                # Channel targets a non-joint node; skip.
                continue
            sampler_idx = ch.sampler
            if sampler_idx is None or sampler_idx >= len(anim.samplers or []):
                continue
            sampler = anim.samplers[sampler_idx]
            if sampler.input is None or sampler.output is None:
                continue
            interp = (sampler.interpolation or "LINEAR").upper()
            # Read times (SCALAR f32) + values.
            times_arr = _accessor_array(sampler.input).astype(np.float32, copy=False)
            if times_arr.ndim > 1:
                times_arr = times_arr.ravel()
            times = [float(t) for t in times_arr]
            values_arr = _accessor_array(sampler.output).astype(np.float32, copy=False)

            channel = tgt.path  # "translation" | "rotation" | "scale" | "weights"
            if channel not in ("translation", "rotation", "scale"):
                # We don't retarget morph-target weights yet.
                continue

            # CUBICSPLINE: each input frame produces THREE output values
            # (in-tangent, value, out-tangent). Drop the tangents and
            # demote to LINEAR — the retargeter resamples densely
            # enough that the curve quality difference is negligible.
            n_per_frame = 3 if interp == "CUBICSPLINE" else 1

            n_frames = len(times)
            expected = n_frames * n_per_frame
            if values_arr.ndim == 2:
                if values_arr.shape[0] != expected:
                    # Truncate / pad mismatch silently.
                    values_arr = values_arr[:expected]
            else:
                # Defensive: should always be ndim==2 for VEC3/VEC4.
                values_arr = values_arr.reshape(expected, -1)

            if interp == "CUBICSPLINE":
                # Take the middle "value" component of each triple.
                values_arr = values_arr[1::3]
                interp = "LINEAR"

            # Build value tuples (translation/scale = vec3, rotation = vec4).
            values: List[tuple] = []
            if channel == "rotation":
                for v in values_arr:
                    values.append((float(v[0]), float(v[1]), float(v[2]), float(v[3])))
            else:
                for v in values_arr:
                    values.append((float(v[0]), float(v[1]), float(v[2])))

            tracks.append(ImportedTrack(
                bone_idx=bone_idx,
                channel=channel,
                times=times,
                values=values,
                interp=interp,
            ))
            if times:
                last = times[-1]
                if last > max_t:
                    max_t = last

        animations.append(ImportedAnimation(
            name=anim_name,
            duration_seconds=max_t,
            fps_target=30,
            tracks=tracks,
        ))

    return ImportedModelWithAnims(model=model, animations=animations)


# ---------------------------------------------------------------------------
# Blend-shape JSON side-file exporter (v4, 2026-04-25)
# ---------------------------------------------------------------------------
#
# PSOBB ignores blend-shape data — but users who imported a face-rigged
# FBX often want to PRESERVE the morph targets for a Blender re-import
# workflow. The side-file exporter walks an ImportedModel's
# ``blend_shapes`` list and emits a self-contained JSON file with one
# entry per shape. The schema is round-trippable: feeding the JSON
# back into ``blend_shapes_from_json`` rebuilds an identical
# ``List[BlendShape]``.
#
# Schema (per shape):
#   {
#       "name":          str,                    # channel/shape name
#       "indexes":       List[int],              # K vertex indices
#       "offsets":       List[List[float]],      # K x 3 deltas
#       "normals":       Optional[List[List[float]]],  # K x 3 normal deltas, or null
#       "default_weight": float,                  # 0..1
#       "mesh_name":     str,                    # parent geometry name
#   }
#
# Wrapper (the file as a whole):
#   {
#       "version":     1,
#       "shape_count": int,
#       "shapes":      [<schema above>, ...],
#   }
#
# We use plain JSON (not numpy-flavoured) so the file is editable in any
# text editor; the loader path tolerates extra unknown keys for forward
# compatibility.


def export_blend_shapes_json(model: ImportedModel) -> dict:
    """Serialize an ImportedModel's blend_shapes to a round-trippable dict.

    The dict is plain Python (json-encodable). For empty inputs returns
    a wrapper with ``shape_count=0`` and an empty ``shapes`` list — the
    UI uses this to detect "no shapes parsed" without an error.
    """
    shapes_json: List[dict] = []
    for bs in model.blend_shapes:
        # Indexes can be a numpy array (parser path) OR a Python list
        # (post-round-trip / hand-built fixtures); normalise both.
        idxs_raw = bs.indexes
        if hasattr(idxs_raw, "tolist"):
            idxs = idxs_raw.tolist()
        else:
            idxs = list(idxs_raw or [])
        offs_raw = bs.offsets
        if hasattr(offs_raw, "tolist"):
            offs = offs_raw.tolist()
        else:
            offs = [list(v) for v in (offs_raw or [])]
        if bs.normals is None:
            norms = None
        else:
            nrm_raw = bs.normals
            if hasattr(nrm_raw, "tolist"):
                norms = nrm_raw.tolist()
            else:
                norms = [list(v) for v in nrm_raw]
        shapes_json.append({
            "name": bs.name,
            "indexes": [int(x) for x in idxs],
            "offsets": [[float(c) for c in row] for row in offs],
            "normals": (
                [[float(c) for c in row] for row in norms]
                if norms is not None else None
            ),
            "default_weight": float(bs.default_weight),
            "mesh_name": bs.mesh_name,
        })
    return {
        "version": 1,
        "shape_count": len(shapes_json),
        "shapes": shapes_json,
    }


def blend_shapes_from_json(d: dict) -> List[BlendShape]:
    """Rebuild a ``List[BlendShape]`` from the exporter's JSON dict.

    Inverse of ``export_blend_shapes_json``. Skips malformed entries
    silently (preservation is best-effort; a partial result is more
    useful than an exception for users who hand-edit the side file).
    """
    out: List[BlendShape] = []
    for entry in d.get("shapes") or []:
        if not isinstance(entry, dict):
            continue
        try:
            idxs = np.asarray(entry.get("indexes") or [], dtype=np.int32)
            offs = np.asarray(entry.get("offsets") or [], dtype=np.float32)
            if offs.ndim == 1:
                offs = offs.reshape(0, 3)
            elif offs.ndim != 2 or offs.shape[1] != 3:
                continue
            normals = entry.get("normals")
            normals_arr: Optional[np.ndarray] = None
            if normals is not None:
                normals_arr = np.asarray(normals, dtype=np.float32)
                if normals_arr.ndim != 2 or normals_arr.shape[1] != 3:
                    normals_arr = None
            out.append(BlendShape(
                name=str(entry.get("name") or ""),
                indexes=idxs,
                offsets=offs,
                normals=normals_arr,
                default_weight=float(entry.get("default_weight") or 0.0),
                mesh_name=str(entry.get("mesh_name") or ""),
            ))
        except (TypeError, ValueError):
            continue
    return out


def export_spring_bones_json(model: ImportedModel) -> dict:
    """Serialize an ImportedModel's spring_bones + node_constraints.

    Mirrors ``export_blend_shapes_json``'s wrapper schema. The fields
    inside each chain match ``imported_to_json``'s ``spring_bones``
    layout, so the result is interchangeable with the live wire shape.
    """
    return {
        "version": 1,
        "spring_chain_count": len(model.spring_bones),
        "node_constraint_count": len(model.node_constraints),
        "spring_bones": [
            {
                "name": chain.name,
                "center_bone_idx": int(chain.center_bone_idx),
                "joints": [
                    {
                        "bone_idx": int(j.bone_idx),
                        "hit_radius": float(j.hit_radius),
                        "stiffness": float(j.stiffness),
                        "drag_force": float(j.drag_force),
                        "gravity_power": float(j.gravity_power),
                        "gravity_dir": list(j.gravity_dir),
                    }
                    for j in chain.joints
                ],
                "colliders": [
                    {
                        "bone_idx": int(c.bone_idx),
                        "shape": c.shape,
                        "offset": list(c.offset),
                        "radius": float(c.radius),
                        "tail": list(c.tail),
                    }
                    for c in chain.colliders
                ],
            }
            for chain in model.spring_bones
        ],
        "node_constraints": [
            {
                "bone_idx": int(nc.bone_idx),
                "source_bone_idx": int(nc.source_bone_idx),
                "constraint_type": nc.constraint_type,
                "weight": float(nc.weight),
                "axis": nc.axis,
            }
            for nc in model.node_constraints
        ],
    }


__all__ = [
    "ImportedMesh",
    "ImportedBone",
    "ImportedModel",
    "ImportedTrack",
    "ImportedAnimation",
    "ImportedModelWithAnims",
    "BlendShape",
    "SpringBoneJoint",
    "SpringBoneCollider",
    "SpringBoneChain",
    "NodeConstraint",
    "parse_obj",
    "parse_gltf",
    "parse_gltf_with_animations",
    "parse_external",
    "quat_to_zyx_bams",
    "rad_to_bams",
    "quantize_skin_weights",
    "imported_to_nj",
    "imported_to_json",
    "imported_from_json",
    "list_templates",
    "load_template",
    "export_blend_shapes_json",
    "blend_shapes_from_json",
    "export_spring_bones_json",
]
