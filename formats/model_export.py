"""Blender-friendly model export — build OBJ + GLB bundles (with textures).

This is the INVERSE of ``formats/import_external.py``: that module reads
OBJ/glTF *into* PSOBB-space meshes; this module writes PSOBB-space meshes
*out* to OBJ and GLB so the user can open a loaded model (mesh + textures)
in Blender.

Pure + testable: nothing here touches the network, the filesystem, or the
FastAPI app. The HTTP layer (server.py) rebuilds the mesh + decodes the
bound textures, then hands us plain Python objects:

    meshes:   a list of ``ExportMesh`` (or any object/dict exposing the
              same fields) — one per submesh: positions, normals, uvs,
              triangle indices, and a ``material_id`` linking it to a tile.
    textures: ``{tile_index: png_bytes}`` — already-decoded RGBA PNGs.

CRITICAL UV CONVENTION (must match the shipped viewer, which now renders
PSOBB top-down UVs with flipY=false):

  * PSOBB stores UVs TOP-DOWN (V increases downward), same as glTF.
  * OBJ's V axis is BOTTOM-UP, so the OBJ writer emits ``1.0 - v``
    (mirrors ``import_external.py``'s OBJ-read flip at ~line 529).
  * glTF/GLB's V axis is TOP-DOWN, so the GLB writer emits ``v`` verbatim
    (mirrors ``import_external.py``'s glTF-read no-flip note at ~line 813).

Get this right per-format or the exported texture is upside-down in Blender.

The XJ/NJ parser bakes vertices into WORLD space already
(``vertices_pre_transformed``), so we DO NOT apply any world matrix here —
positions are emitted as-is. Skinned models export their rest pose; bones
are noted in the OBJ header comment / glTF ``extras`` but no skin is written.
"""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

# Type aliases for the geometry inputs. We accept plain tuples/lists so the
# server can feed us decoded ``XjVertex``/``XjMesh`` data without importing
# this module's dataclass.
Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]


@dataclass
class ExportMesh:
    """One submesh ready for export.

    ``positions`` are WORLD-space (already baked — do not transform).
    ``normals`` and ``uvs`` may be empty/None when the source lacked them.
    ``indices`` is a flat triangle list (len % 3 == 0), each int indexing
    into ``positions`` (and the parallel ``normals`` / ``uvs``).
    ``material_id`` links this submesh to a texture tile via the resolved
    binding; ``name`` is an optional human label.
    """

    positions: List[Vec3]
    indices: List[int]
    normals: Optional[List[Vec3]] = None
    uvs: Optional[List[Vec2]] = None
    material_id: int = 0
    name: str = ""


# --------------------------------------------------------------------------- #
# Input normalisation
# --------------------------------------------------------------------------- #
def _get(obj, key, default=None):
    """Read ``key`` from a dict OR an attribute from an object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_mesh(m, fallback_idx: int) -> ExportMesh:
    """Coerce a mesh-like input (ExportMesh / dict / XjMesh-ish) into an
    ExportMesh with concrete lists, validating geometry shape.
    """
    if isinstance(m, ExportMesh):
        positions = list(m.positions)
        normals = list(m.normals) if m.normals else None
        uvs = list(m.uvs) if m.uvs else None
        indices = list(m.indices)
        material_id = int(m.material_id)
        name = m.name or f"submesh_{fallback_idx}"
        return ExportMesh(positions, indices, normals, uvs, material_id, name)

    positions = [tuple(p) for p in (_get(m, "positions") or [])]
    normals_raw = _get(m, "normals")
    uvs_raw = _get(m, "uvs")
    normals = [tuple(n) for n in normals_raw] if normals_raw else None
    uvs = [tuple(uv) for uv in uvs_raw] if uvs_raw else None
    indices = [int(i) for i in (_get(m, "indices") or [])]
    material_id = int(_get(m, "material_id", 0) or 0)
    name = str(_get(m, "name", "") or f"submesh_{fallback_idx}")

    if len(indices) % 3 != 0:
        raise ValueError(
            f"submesh {fallback_idx}: index count {len(indices)} is not a "
            "multiple of 3 (not a triangle list)"
        )
    nverts = len(positions)
    if normals is not None and len(normals) != nverts:
        # Tolerate a normal/vertex mismatch by dropping normals rather than
        # corrupting the file — Blender recomputes them on import.
        normals = None
    if uvs is not None and len(uvs) != nverts:
        uvs = None
    return ExportMesh(positions, indices, normals, uvs, material_id, name)


def _iter_normalized(meshes: Sequence) -> List[ExportMesh]:
    out: List[ExportMesh] = []
    for i, m in enumerate(meshes):
        em = _normalize_mesh(m, i)
        if em.positions and em.indices:
            out.append(em)
    if not out:
        raise ValueError("no exportable geometry (every submesh was empty)")
    return out


def _material_name(material_id: int) -> str:
    return f"mat_{material_id}"


def _texture_filename(material_id: int, tile_index: int) -> str:
    """Stable PNG filename for a material's bound tile."""
    return f"tex_{tile_index}.png"


# --------------------------------------------------------------------------- #
# OBJ + MTL bundle
# --------------------------------------------------------------------------- #
def build_obj_bundle(
    meshes: Sequence,
    textures: Optional[Dict[int, bytes]] = None,
    *,
    binding: Optional[Dict[int, int]] = None,
    model_name: str = "model",
    bone_count: int = 0,
) -> Dict[str, bytes]:
    """Build a Wavefront OBJ bundle for ``meshes`` + ``textures``.

    Returns ``{filename: bytes}`` containing ``model.obj``, ``model.mtl``,
    and one ``tex_<idx>.png`` per bound texture tile.

    * One material per distinct ``material_id``; each material references
      the PNG of its bound tile (via ``binding`` material_id -> tile_index,
      defaulting to material_id == tile_index when ``binding`` is absent).
    * OBJ V is BOTTOM-UP while PSOBB V is TOP-DOWN, so V is written as
      ``1.0 - v`` (mirror of ``import_external.py`` OBJ read).
    * Positions are world-space already — no transform applied.

    ``bone_count`` (skinned rest-pose) is recorded only as a header comment.
    """
    textures = textures or {}
    norm = _iter_normalized(meshes)

    def mid_to_tile(mid: int) -> int:
        if binding is not None:
            return binding.get(mid, mid)
        return mid

    # ---- model.mtl -------------------------------------------------------- #
    mtl_lines: List[str] = [
        "# PSOBB model export (psobb-studio)",
        "# OBJ companion material library",
        "",
    ]
    files: Dict[str, bytes] = {}
    seen_mids: List[int] = []
    for em in norm:
        if em.material_id not in seen_mids:
            seen_mids.append(em.material_id)

    for mid in seen_mids:
        tile = mid_to_tile(mid)
        mtl_lines.append(f"newmtl {_material_name(mid)}")
        mtl_lines.append("Ka 0.000 0.000 0.000")
        mtl_lines.append("Kd 1.000 1.000 1.000")
        mtl_lines.append("Ks 0.000 0.000 0.000")
        mtl_lines.append("d 1.0")
        mtl_lines.append("illum 1")
        png = textures.get(tile)
        if png:
            tex_name = _texture_filename(mid, tile)
            files[tex_name] = png
            mtl_lines.append(f"map_Kd {tex_name}")
        mtl_lines.append("")

    mtl_bytes = ("\n".join(mtl_lines) + "\n").encode("utf-8")

    # ---- model.obj -------------------------------------------------------- #
    obj_lines: List[str] = [
        "# PSOBB model export (psobb-studio)",
        f"# model: {model_name}",
        f"# submeshes: {len(norm)}",
        "# UV V flipped to OBJ bottom-up (PSOBB stores top-down V)",
        "# vertices are world-space (no transform applied)",
    ]
    if bone_count:
        obj_lines.append(
            f"# skinned model: rest pose exported ({bone_count} bones, "
            "skin weights not written)"
        )
    obj_lines.append("mtllib model.mtl")
    obj_lines.append("")

    # OBJ indices are 1-based and GLOBAL across the whole file, so we keep
    # a running base offset as we append each submesh's vertices.
    v_base = 0
    has_normals_any = any(em.normals for em in norm)
    has_uvs_any = any(em.uvs for em in norm)

    for si, em in enumerate(norm):
        nverts = len(em.positions)
        obj_lines.append(f"o {em.name or f'submesh_{si}'}")
        for p in em.positions:
            obj_lines.append(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
        if em.uvs:
            for uv in em.uvs:
                # OBJ V is bottom-up; PSOBB V is top-down -> flip.
                obj_lines.append(f"vt {uv[0]:.6f} {1.0 - uv[1]:.6f}")
        if em.normals:
            for n in em.normals:
                obj_lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
        obj_lines.append(f"usemtl {_material_name(em.material_id)}")
        obj_lines.append("s off")

        local_has_uv = bool(em.uvs)
        local_has_n = bool(em.normals)
        tri = em.indices
        for t in range(0, len(tri), 3):
            a = v_base + tri[t] + 1
            b = v_base + tri[t + 1] + 1
            c = v_base + tri[t + 2] + 1
            obj_lines.append(
                "f "
                + " ".join(
                    _obj_face_vert(idx, local_has_uv, local_has_n)
                    for idx in (a, b, c)
                )
            )
        obj_lines.append("")
        v_base += nverts

    _ = (has_normals_any, has_uvs_any)  # documented intent; not needed inline
    obj_bytes = ("\n".join(obj_lines) + "\n").encode("utf-8")

    files["model.obj"] = obj_bytes
    files["model.mtl"] = mtl_bytes
    return files


def _obj_face_vert(global_index: int, has_uv: bool, has_n: bool) -> str:
    """Format one OBJ face vertex reference.

    Because we emit positions, UVs and normals 1:1 in parallel order, the
    v / vt / vn indices are identical for a given vertex.
    """
    if has_uv and has_n:
        return f"{global_index}/{global_index}/{global_index}"
    if has_uv and not has_n:
        return f"{global_index}/{global_index}"
    if has_n and not has_uv:
        return f"{global_index}//{global_index}"
    return f"{global_index}"


# --------------------------------------------------------------------------- #
# GLB bundle
# --------------------------------------------------------------------------- #
def build_glb_bundle(
    meshes: Sequence,
    textures: Optional[Dict[int, bytes]] = None,
    *,
    binding: Optional[Dict[int, int]] = None,
    model_name: str = "model",
    bone_count: int = 0,
) -> bytes:
    """Build a single binary glTF (.glb) for ``meshes`` + ``textures``.

    * One trimesh geometry per submesh; each carries a PBR material whose
      baseColorTexture is the bound tile PNG (embedded in the GLB binary
      chunk — fully self-contained, no external files).
    * glTF V is TOP-DOWN (same as PSOBB) -> UVs written verbatim (NO flip).
    * Positions are world-space already — no transform applied.

    Implemented with trimesh (already a dependency); falls back to a
    hand-rolled glTF writer only if trimesh is somehow unavailable.
    """
    textures = textures or {}
    norm = _iter_normalized(meshes)

    try:
        return _build_glb_trimesh(
            norm, textures, binding, model_name, bone_count
        )
    except ImportError:
        # trimesh/PIL/numpy missing — fall back to the manual writer so the
        # feature still works in a minimal environment.
        return _build_glb_manual(norm, textures, binding, model_name, bone_count)


def _resolve_tile(binding: Optional[Dict[int, int]], mid: int) -> int:
    if binding is not None:
        return binding.get(mid, mid)
    return mid


def _build_glb_trimesh(
    norm: List[ExportMesh],
    textures: Dict[int, bytes],
    binding: Optional[Dict[int, int]],
    model_name: str,
    bone_count: int,
) -> bytes:
    import numpy as np
    import trimesh
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import PBRMaterial
    from PIL import Image

    # Cache one decoded PIL image per tile so identical tiles share pixels.
    tile_images: Dict[int, "Image.Image"] = {}

    def tile_image(tile_idx: int):
        if tile_idx in tile_images:
            return tile_images[tile_idx]
        png = textures.get(tile_idx)
        img = None
        if png:
            try:
                img = Image.open(io.BytesIO(png)).convert("RGBA")
                img.load()
            except Exception:
                img = None
        tile_images[tile_idx] = img
        return img

    scene = trimesh.Scene()
    for si, em in enumerate(norm):
        verts = np.asarray(em.positions, dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(em.indices, dtype=np.int64).reshape(-1, 3)
        mesh = trimesh.Trimesh(
            vertices=verts, faces=faces, process=False, validate=False
        )
        if em.normals and len(em.normals) == len(em.positions):
            try:
                mesh.vertex_normals = np.asarray(
                    em.normals, dtype=np.float64
                ).reshape(-1, 3)
            except Exception:
                pass

        tile_idx = _resolve_tile(binding, em.material_id)
        img = tile_image(tile_idx)
        if em.uvs and len(em.uvs) == len(em.positions):
            # glTF V is top-down (same as PSOBB) -> NO flip.
            uv = np.asarray(em.uvs, dtype=np.float64).reshape(-1, 2)
            mat = PBRMaterial(
                name=_material_name(em.material_id),
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                metallicFactor=0.0,
                roughnessFactor=1.0,
            )
            if img is not None:
                mat.baseColorTexture = img
            mesh.visual = TextureVisuals(uv=uv, material=mat, image=img)
        else:
            # No UVs: still attach a material (untextured) so material count
            # survives the round-trip.
            mat = PBRMaterial(
                name=_material_name(em.material_id),
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            )
            mesh.visual = TextureVisuals(material=mat)

        node_name = em.name or f"submesh_{si}"
        scene.add_geometry(mesh, node_name=node_name, geom_name=node_name)

    scene.metadata["model_name"] = model_name
    if bone_count:
        scene.metadata["bone_count"] = bone_count
        scene.metadata["note"] = "rest pose; skin weights not exported"

    glb = scene.export(file_type="glb")
    if isinstance(glb, (bytes, bytearray)):
        return bytes(glb)
    # Some trimesh versions return a file-like / str; coerce.
    if hasattr(glb, "read"):
        return glb.read()
    return bytes(glb)


# --------------------------------------------------------------------------- #
# Manual glTF writer (fallback only — no third-party deps)
# --------------------------------------------------------------------------- #
def _build_glb_manual(
    norm: List[ExportMesh],
    textures: Dict[int, bytes],
    binding: Optional[Dict[int, int]],
    model_name: str,
    bone_count: int,
) -> bytes:
    """Minimal self-contained glTF 2.0 binary writer.

    Emits POSITION (+ NORMAL, TEXCOORD_0 when present) and indices for each
    submesh as a separate primitive/mesh/node, with embedded PNG textures.
    UVs are written verbatim (glTF top-down == PSOBB). This path is only
    used if trimesh is unavailable; the trimesh path is preferred.
    """
    import json as _json

    bin_chunks: List[bytes] = []
    bin_len = 0

    buffer_views: List[dict] = []
    accessors: List[dict] = []
    images: List[dict] = []
    textures_gltf: List[dict] = []
    samplers: List[dict] = [{"magFilter": 9729, "minFilter": 9987,
                             "wrapS": 10497, "wrapT": 10497}]
    materials: List[dict] = []
    meshes_gltf: List[dict] = []
    nodes: List[dict] = []

    def _align4(n: int) -> int:
        return (n + 3) & ~3

    def add_view(data: bytes, target: Optional[int] = None) -> int:
        nonlocal bin_len
        pad = _align4(bin_len) - bin_len
        if pad:
            bin_chunks.append(b"\x00" * pad)
            bin_len += pad
        offset = bin_len
        bin_chunks.append(data)
        bin_len += len(data)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        buffer_views.append(view)
        return len(buffer_views) - 1

    # Embed textures (dedup by tile index).
    tile_to_material: Dict[int, int] = {}

    def material_for_tile(tile_idx: int, mid: int) -> int:
        if tile_idx in tile_to_material:
            return tile_to_material[tile_idx]
        png = textures.get(tile_idx)
        mat: dict = {
            "name": _material_name(mid),
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
        }
        if png:
            view = add_view(png)
            images.append({"bufferView": view, "mimeType": "image/png"})
            textures_gltf.append({"sampler": 0, "source": len(images) - 1})
            mat["pbrMetallicRoughness"]["baseColorTexture"] = {
                "index": len(textures_gltf) - 1
            }
        materials.append(mat)
        midx = len(materials) - 1
        tile_to_material[tile_idx] = midx
        return midx

    for si, em in enumerate(norm):
        nverts = len(em.positions)

        # POSITION
        pos_bytes = b"".join(
            struct.pack("<3f", float(p[0]), float(p[1]), float(p[2]))
            for p in em.positions
        )
        pos_view = add_view(pos_bytes, target=34962)
        xs = [p[0] for p in em.positions]
        ys = [p[1] for p in em.positions]
        zs = [p[2] for p in em.positions]
        accessors.append({
            "bufferView": pos_view, "componentType": 5126, "count": nverts,
            "type": "VEC3",
            "min": [min(xs), min(ys), min(zs)],
            "max": [max(xs), max(ys), max(zs)],
        })
        pos_acc = len(accessors) - 1

        attributes = {"POSITION": pos_acc}

        if em.normals:
            n_bytes = b"".join(
                struct.pack("<3f", float(n[0]), float(n[1]), float(n[2]))
                for n in em.normals
            )
            n_view = add_view(n_bytes, target=34962)
            accessors.append({
                "bufferView": n_view, "componentType": 5126,
                "count": nverts, "type": "VEC3",
            })
            attributes["NORMAL"] = len(accessors) - 1

        if em.uvs:
            # glTF V is top-down (same as PSOBB) -> verbatim.
            uv_bytes = b"".join(
                struct.pack("<2f", float(uv[0]), float(uv[1])) for uv in em.uvs
            )
            uv_view = add_view(uv_bytes, target=34962)
            accessors.append({
                "bufferView": uv_view, "componentType": 5126,
                "count": nverts, "type": "VEC2",
            })
            attributes["TEXCOORD_0"] = len(accessors) - 1

        # Indices (uint32).
        idx_bytes = b"".join(struct.pack("<I", int(i)) for i in em.indices)
        idx_view = add_view(idx_bytes, target=34963)
        accessors.append({
            "bufferView": idx_view, "componentType": 5125,
            "count": len(em.indices), "type": "SCALAR",
        })
        idx_acc = len(accessors) - 1

        tile_idx = _resolve_tile(binding, em.material_id)
        mat_idx = material_for_tile(tile_idx, em.material_id)

        prim = {"attributes": attributes, "indices": idx_acc,
                "material": mat_idx, "mode": 4}
        meshes_gltf.append({"name": em.name or f"submesh_{si}",
                            "primitives": [prim]})
        nodes.append({"mesh": len(meshes_gltf) - 1,
                      "name": em.name or f"submesh_{si}"})

    bin_blob = b"".join(bin_chunks)
    gltf: dict = {
        "asset": {"version": "2.0", "generator": "psobb-studio model_export"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes_gltf,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(bin_blob)}],
        "materials": materials,
        "extras": {"model_name": model_name},
    }
    if images:
        gltf["images"] = images
        gltf["textures"] = textures_gltf
        gltf["samplers"] = samplers
    if bone_count:
        gltf["extras"]["bone_count"] = bone_count
        gltf["extras"]["note"] = "rest pose; skin weights not exported"

    json_bytes = _json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_pad = ((len(json_bytes) + 3) & ~3) - len(json_bytes)
    json_bytes += b" " * json_pad
    bin_pad = ((len(bin_blob) + 3) & ~3) - len(bin_blob)
    bin_blob += b"\x00" * bin_pad

    total = 12 + 8 + len(json_bytes) + 8 + len(bin_blob)
    out = io.BytesIO()
    out.write(struct.pack("<III", 0x46546C67, 2, total))  # 'glTF', ver 2
    out.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))  # JSON
    out.write(json_bytes)
    out.write(struct.pack("<II", len(bin_blob), 0x004E4942))  # BIN
    out.write(bin_blob)
    return out.getvalue()
