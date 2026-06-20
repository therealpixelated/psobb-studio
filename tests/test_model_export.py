"""Model export (OBJ + GLB) tests.

Covers ``formats.model_export.build_obj_bundle`` / ``build_glb_bundle`` and
the server export endpoints. The pure-builder tests are hermetic (synthetic
textured mesh); the endpoint smoke uses a real model only if one is present
under the data dir, otherwise it skips cleanly.

UV convention asserted PER FORMAT (the whole point of the feature):
  * OBJ V is BOTTOM-UP, PSOBB is TOP-DOWN -> exporter writes ``1.0 - v``.
  * GLB V is TOP-DOWN (== PSOBB) -> exporter writes ``v`` verbatim.
"""
from __future__ import annotations

import io
import struct

import pytest

from formats.model_export import (
    ExportMesh,
    build_glb_bundle,
    build_obj_bundle,
)


# --------------------------------------------------------------------------- #
# Fixtures: a single textured quad with a KNOWN UV so the V flip is provable.
# --------------------------------------------------------------------------- #
def _png_2x2() -> bytes:
    """A tiny valid 2x2 RGBA PNG (top row red, bottom row blue)."""
    from PIL import Image

    img = Image.new("RGBA", (2, 2))
    img.putpixel((0, 0), (255, 0, 0, 255))
    img.putpixel((1, 0), (255, 0, 0, 255))
    img.putpixel((0, 1), (0, 0, 255, 255))
    img.putpixel((1, 1), (0, 0, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _quad_mesh() -> ExportMesh:
    # A unit quad in the XY plane, two triangles. The bottom-left vertex
    # carries V=0.25 (a non-symmetric value so a flip is detectable).
    positions = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    normals = [(0.0, 0.0, 1.0)] * 4
    uvs = [
        (0.0, 0.25),  # the probe UV
        (1.0, 0.25),
        (1.0, 0.75),
        (0.0, 0.75),
    ]
    indices = [0, 1, 2, 0, 2, 3]
    return ExportMesh(
        positions=positions,
        indices=indices,
        normals=normals,
        uvs=uvs,
        material_id=0,
        name="quad",
    )


# --------------------------------------------------------------------------- #
# OBJ bundle
# --------------------------------------------------------------------------- #
def test_build_obj_bundle_roundtrip_and_v_flip():
    trimesh = pytest.importorskip("trimesh")

    mesh = _quad_mesh()
    textures = {0: _png_2x2()}
    files = build_obj_bundle([mesh], textures, model_name="probe")

    # Bundle contains the expected members.
    assert "model.obj" in files
    assert "model.mtl" in files
    png_members = [k for k in files if k.endswith(".png")]
    assert png_members, "no PNG written into the OBJ bundle"
    # The PNG bytes survived intact.
    assert files[png_members[0]] == textures[0]

    obj_text = files["model.obj"].decode("utf-8")
    mtl_text = files["model.mtl"].decode("utf-8")
    # MTL references the texture.
    assert "map_Kd" in mtl_text
    assert "newmtl mat_0" in mtl_text
    assert "usemtl mat_0" in obj_text
    assert "mtllib model.mtl" in obj_text

    # --- V FLIP: source V=0.25 must be written as 1-0.25 = 0.75 ----------- #
    vt_lines = [ln for ln in obj_text.splitlines() if ln.startswith("vt ")]
    assert vt_lines, "no vt lines emitted"
    first_v = float(vt_lines[0].split()[2])
    assert abs(first_v - 0.75) < 1e-5, (
        f"OBJ V not bottom-up flipped: expected 0.75, got {first_v}"
    )

    # --- Round-trip via trimesh: counts survive ---------------------------- #
    loaded = trimesh.load(
        io.BytesIO(files["model.obj"]),
        file_type="obj",
        process=False,
    )
    # trimesh may return a Scene or a single Trimesh.
    if isinstance(loaded, trimesh.Scene):
        geoms = list(loaded.geometry.values())
        total_v = sum(len(g.vertices) for g in geoms)
        total_f = sum(len(g.faces) for g in geoms)
    else:
        total_v = len(loaded.vertices)
        total_f = len(loaded.faces)
    # 4 verts, 2 faces in the source quad.
    assert total_v == 4, f"vertex count did not survive: {total_v}"
    assert total_f == 2, f"face count did not survive: {total_f}"


def test_build_obj_bundle_multi_material():
    sub0 = _quad_mesh()
    sub1 = ExportMesh(
        positions=[(2.0, 0.0, 0.0), (3.0, 0.0, 0.0), (3.0, 1.0, 0.0)],
        indices=[0, 1, 2],
        uvs=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        material_id=1,
        name="tri",
    )
    files = build_obj_bundle([sub0, sub1], {0: _png_2x2(), 1: _png_2x2()})
    mtl = files["model.mtl"].decode("utf-8")
    assert "newmtl mat_0" in mtl and "newmtl mat_1" in mtl
    obj = files["model.obj"].decode("utf-8")
    # Global 1-based indexing: second submesh's face refs start past sub0's
    # 4 verts (so includes index 5/6/7).
    assert "usemtl mat_1" in obj


# --------------------------------------------------------------------------- #
# GLB bundle
# --------------------------------------------------------------------------- #
def test_build_glb_bundle_roundtrip_textures_and_no_v_flip():
    trimesh = pytest.importorskip("trimesh")

    mesh = _quad_mesh()
    textures = {0: _png_2x2()}
    glb = build_glb_bundle([mesh], textures, model_name="probe")

    # Valid GLB container magic.
    assert glb[:4] == b"glTF", "not a GLB (bad magic)"
    assert len(glb) > 100

    loaded = trimesh.load(io.BytesIO(glb), file_type="glb", process=False)
    if isinstance(loaded, trimesh.Scene):
        geoms = list(loaded.geometry.values())
    else:
        geoms = [loaded]
    assert geoms, "GLB produced no geometry"
    total_v = sum(len(g.vertices) for g in geoms)
    total_f = sum(len(g.faces) for g in geoms)
    assert total_v == 4, f"GLB vertex count did not survive: {total_v}"
    assert total_f == 2, f"GLB face count did not survive: {total_f}"

    # --- Embedded texture survived ---------------------------------------- #
    found_tex = False
    found_uv_unflipped = False
    for g in geoms:
        vis = getattr(g, "visual", None)
        mat = getattr(vis, "material", None)
        img = getattr(mat, "baseColorTexture", None) if mat is not None else None
        if img is not None:
            found_tex = True
        uv = getattr(vis, "uv", None)
        if uv is not None and len(uv):
            # glTF V is top-down (verbatim): the probe vertex keeps V≈0.25,
            # NOT 0.75. Match by the U=0 corner.
            for u, v in uv:
                if abs(u - 0.0) < 1e-4 and abs(v - 0.25) < 1e-3:
                    found_uv_unflipped = True
    assert found_tex, "embedded baseColorTexture not recovered from GLB"
    assert found_uv_unflipped, (
        "GLB UV was flipped: expected verbatim top-down V≈0.25 to survive"
    )


def test_build_glb_manual_fallback_is_valid():
    """The dependency-free fallback writer must also emit a valid GLB."""
    from formats import model_export

    norm = [model_export._normalize_mesh(_quad_mesh(), 0)]
    glb = model_export._build_glb_manual(
        norm, {0: _png_2x2()}, None, "probe", 0
    )
    assert glb[:4] == b"glTF"
    # Header total length matches the buffer.
    (_magic, _ver, total) = struct.unpack("<III", glb[:12])
    assert total == len(glb)

    # Re-load via trimesh to prove it parses.
    trimesh = pytest.importorskip("trimesh")
    loaded = trimesh.load(io.BytesIO(glb), file_type="glb", process=False)
    geoms = (
        list(loaded.geometry.values())
        if isinstance(loaded, trimesh.Scene)
        else [loaded]
    )
    total_v = sum(len(g.vertices) for g in geoms)
    assert total_v == 4


def test_empty_meshes_raise():
    with pytest.raises(ValueError):
        build_obj_bundle([], {})
    with pytest.raises(ValueError):
        build_glb_bundle([ExportMesh(positions=[], indices=[])], {})


# --------------------------------------------------------------------------- #
# Endpoint smoke (TestClient). Skips cleanly if deps / data absent.
# --------------------------------------------------------------------------- #
def _client():
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    import server

    return fastapi_testclient.TestClient(server.app), server


def test_export_capabilities_route():
    client, _server = _client()
    r = client.get("/api/export_model/capabilities")
    assert r.status_code == 200
    caps = r.json()
    assert caps["obj"] is True
    assert caps["glb"] is True
    assert caps["fbx"] is False


def test_export_fbx_returns_501():
    client, server = _client()
    # Find any real model to target; if none, use an arbitrary path — FBX
    # must 501 BEFORE any heavy mesh work, so the path need not resolve.
    r = client.post(
        "/api/export_model",
        json={"path": "anything.nj", "format": "fbx"},
    )
    assert r.status_code == 501, r.text
    assert "fbx" in r.text.lower()


def _find_real_model(server):
    """Locate one parseable model under the data dir, else None."""
    import pathlib

    roots = []
    for attr in ("DATA_DIR", "LIVE_DATA_DIR"):
        d = getattr(server, attr, None)
        if d:
            roots.append(pathlib.Path(d))
    for root in roots:
        if not root or not root.is_dir():
            continue
        for ext in ("*.nj", "*.xj"):
            for p in root.glob(ext):
                try:
                    if p.stat().st_size > 0:
                        return p.name
                except OSError:
                    continue
    return None


@pytest.mark.parametrize("fmt", ["obj", "glb"])
def test_export_real_model_if_present(fmt):
    client, server = _client()
    model = _find_real_model(server)
    if not model:
        pytest.skip("no real .nj/.xj model under the data dir")
    r = client.post(
        "/api/export_model", json={"path": model, "format": fmt}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "export_url" in body
    assert isinstance(body.get("warnings", []), list)
    # The staged artifact is downloadable.
    dl = client.get(body["export_url"])
    assert dl.status_code == 200
    assert len(dl.content) > 0
    if fmt == "glb":
        assert dl.content[:4] == b"glTF"
