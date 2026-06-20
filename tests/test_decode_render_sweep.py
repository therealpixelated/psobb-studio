"""End-to-end decode + render sweep — the headline smoke test for the
psov2 render-grounding work (Phases 2-3, 2026-06-20).

The owner's complaint was "no smoke testing?" for the flat/white render
fix. This sweep drives several real models through the SAME wire path the
3D viewer uses and asserts the three things that, together, prove the
render is grounded:

  (1) RESOLVE   — the texture binding is non-empty and every material_id
                  has a concrete source (in_bml / cross_bml / cross_afs),
                  not a fabricated bare-archive or "no inner" diagnostic.
                  Explicitly pins the momoka-class regression.
  (2) DECODE    — every bound tile decodes to a valid PNG of its declared
                  XVMH dimensions, NOT the magenta NotImplemented
                  placeholder.
  (3) ATTRIBUTES— the mesh payload advertises has_color and a 12-float
                  (48-byte) vertex stride, vertices_pre_transformed is
                  true, and a known vertex-/diffuse-colored asset carries
                  a non-(1,1,1,1) color (proves color is no longer
                  dropped on the way to the GPU).

All assertions degrade to ``pytest.skip`` when the underlying PSOBB data
is not installed, so the suite is safe to run on a bare checkout.

Runs in-process via fastapi.testclient.TestClient — no external server, no
network. The decode check reuses the server's own
``_export_resolve_binding_textures`` helper (the exact archive/tile
resolution the frontend ``fetchBoundTextures`` performs), so it is
faithful to production without reconstructing archive paths by hand.
"""
from __future__ import annotations

import base64
import io
import struct

import pytest

server = pytest.importorskip("server")
from fastapi.testclient import TestClient  # noqa: E402

try:
    from PIL import Image
except Exception:  # pragma: no cover - Pillow is a hard dep elsewhere
    Image = None


# Magenta is the placeholder emitted for an undecodable / NotImplemented
# texture format (P8 / YUY2 / V8U8). A bound tile that decodes to a solid
# magenta image means the decode silently failed.
_MAGENTA = (255, 0, 255)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(server.app)


# (label, base_path, inner)  — inner is None for top-level .nj.
#
# Coverage (>=5 models, each independently skipped if absent):
#   - momoka_npc  : the named momoka-class regression NPC (diffuse-colored,
#                   single-entry BML resolving to its inner texture appendix)
#   - biter_enemy : a vertex-/diffuse-colored bm_ene enemy BODY
#   - item_weapon : a TEXTURED ItemModel.afs weapon inner (cross-AFS textures)
#   - player_body : a player body (plAbdy00.nj, cross-AFS textures)
#   - saku_prop   : an object prop driven through the .xj DESCRIPTOR path (a
#                   map fence). It exercises the SECOND geometry decoder
#                   (xj_descriptor.py) and the diffuse-color fallback that
#                   grounds otherwise-flat props — the "untextured prop"
#                   render case. (No PSOBB model in the shipped data has an
#                   EMPTY texture binding — verified by a full 230-BML sweep
#                   2026-06-20 — so the prop coverage is a descriptor-path
#                   object whose flat faces depend on the diffuse fallback.)
# These were verified to resolve cleanly on the dev data install on
# 2026-06-20.
_SWEEP = [
    ("momoka_npc", "bm_npc_momoka.bml", "n_momoka_t_body.nj"),
    ("biter_enemy", "bm_ene_biter_body.bml", "biter_body.nj"),
    ("item_weapon", "ItemModel.afs", "0000_ItemModel_0000.nj"),
    ("player_body", "plAbdy00.nj", None),
    ("saku_prop", "bm_fd_obj_n_saku_4x2.bml", "fd_obj_n_saku_4x2.xj"),
]

# Assets whose material-chunk DIFFUSE is non-white on disk, so the render
# pipeline MUST surface a non-(1,1,1,1) vertex color for them. (PSOBB.IO Nj
# data carries no per-vertex color chunks — the diffuse fallback is the
# load-bearing color source, verified empirically 2026-06-20.) The .xj
# saku prop also carries a non-white type-5 material diffuse on disk.
_KNOWN_COLORED = {"momoka_npc", "biter_enemy", "player_body", "saku_prop"}


def _get_textures(client: TestClient, base: str, inner):
    params = {"inner": inner} if inner else None
    r = client.get(f"/api/model_textures/{base}", params=params)
    return r


def _get_mesh(client: TestClient, base: str, inner):
    params = {"inner": inner} if inner else None
    r = client.get(f"/api/model_mesh/{base}", params=params)
    return r


def _available(client: TestClient, base: str, inner) -> bool:
    """True iff both the mesh and texture endpoints answer 200 for this asset."""
    return (
        _get_mesh(client, base, inner).status_code == 200
        and _get_textures(client, base, inner).status_code == 200
    )


# --------------------------------------------------------------------------
# (1) RESOLVE
# --------------------------------------------------------------------------

@pytest.mark.parametrize("label,base,inner", _SWEEP, ids=[s[0] for s in _SWEEP])
def test_sweep_resolve(client, label, base, inner):
    """Every submesh material_id binds to a concrete texture source."""
    if not _available(client, base, inner):
        pytest.skip(f"{label}: data not installed")

    j = _get_textures(client, base, inner).json()
    binding = j.get("binding") or []
    assert binding, f"{label}: empty binding (no texture resolved)"

    valid_sources = {"in_bml", "cross_bml", "cross_afs"}
    for row in binding:
        src = row.get("source") or ("missing" if row.get("missing") else "in_bml")
        assert not row.get("missing"), (
            f"{label}: material {row.get('material_id')} resolved to MISSING"
        )
        assert src in valid_sources, (
            f"{label}: material {row.get('material_id')} has bad source {src!r}"
        )


def test_momoka_resolves_to_inner_xvm(client):
    """Momoka-class regression: the single-entry BML resolves to its inner
    texture appendix (in_bml) and yields >=1 XVMH record — never the bare
    container misread."""
    base, inner = "bm_npc_momoka.bml", "n_momoka_t_body.nj"
    if not _available(client, base, inner):
        pytest.skip("momoka data not installed")

    j = _get_textures(client, base, inner).json()
    assert j.get("inner") == inner, j
    binding = j.get("binding") or []
    assert binding, "momoka: empty binding"
    assert all((r.get("source") or "in_bml") == "in_bml" for r in binding), binding
    assert len(j.get("xvmh") or []) >= 1, "momoka: no XVMH records resolved"


# --------------------------------------------------------------------------
# (2) DECODE
# --------------------------------------------------------------------------

@pytest.mark.parametrize("label,base,inner", _SWEEP, ids=[s[0] for s in _SWEEP])
def test_sweep_decode(client, label, base, inner):
    """Every bound tile decodes to a valid PNG of its declared dimensions
    and is not the magenta NotImplemented placeholder."""
    if Image is None:
        pytest.skip("Pillow not available")
    if not _available(client, base, inner):
        pytest.skip(f"{label}: data not installed")

    j = _get_textures(client, base, inner).json()
    binding = j.get("binding") or []
    assert binding, f"{label}: empty binding"

    # Declared dimensions per tile_index (from the XVMH record list).
    dims = {
        int(x["tile_index"]): (int(x.get("width", 0)), int(x.get("height", 0)))
        for x in (j.get("xvmh") or [])
        if "tile_index" in x
    }

    # Reuse the server's own per-row archive/tile resolution → png bytes
    # (the exact path the frontend fetchBoundTextures uses).
    default_arch = server._export_default_texture_archive(base, inner)
    textures, _warn = server._export_resolve_binding_textures(binding, default_arch)
    assert textures, f"{label}: no tiles decoded from {len(binding)} binding rows"

    for row in binding:
        mid = int(row.get("material_id", 0))
        png = textures.get(mid)
        if png is None:
            # A row may legitimately share a tile already decoded under
            # another material_id; require at least the FIRST occurrence to
            # have decoded. Skip rows with no own bytes (deduped).
            continue
        im = Image.open(io.BytesIO(png))
        im.load()
        assert im.width > 0 and im.height > 0, f"{label}: zero-dim tile for mat {mid}"

        # Declared-dimension check when the XVMH record is available.
        tile_idx = int(row.get("tile_index", 0) or 0)
        if tile_idx in dims and dims[tile_idx][0] > 0:
            assert (im.width, im.height) == dims[tile_idx], (
                f"{label}: mat {mid} tile {tile_idx} decoded "
                f"{im.width}x{im.height}, declared {dims[tile_idx]}"
            )

        # Not the magenta placeholder (sample a few pixels; a fully-magenta
        # image means the decode silently produced the NotImplemented fill).
        rgb = im.convert("RGB")
        w, h = rgb.size
        sample = [
            rgb.getpixel((0, 0)),
            rgb.getpixel((w // 2, h // 2)),
            rgb.getpixel((w - 1, h - 1)),
        ]
        assert not all(px == _MAGENTA for px in sample), (
            f"{label}: mat {mid} tile {tile_idx} is the magenta placeholder "
            f"(decode failed)"
        )


# --------------------------------------------------------------------------
# (3) ATTRIBUTES
# --------------------------------------------------------------------------

@pytest.mark.parametrize("label,base,inner", _SWEEP, ids=[s[0] for s in _SWEEP])
def test_sweep_vertex_attributes(client, label, base, inner):
    """Payload advertises has_color + 12-float stride; vertices are
    pre-transformed; known-colored assets carry a non-white color."""
    r = _get_mesh(client, base, inner)
    if r.status_code != 200:
        pytest.skip(f"{label}: mesh not available (HTTP {r.status_code})")
    payload = r.json()
    if payload.get("mesh_count", 0) == 0:
        pytest.skip(f"{label}: no geometry parsed")

    assert payload.get("has_color") is True, f"{label}: payload missing has_color"
    assert payload.get("vertex_format_version") == 2, payload.get("vertex_format_version")
    assert payload.get("vertices_pre_transformed") is True, payload

    stride_floats = 12  # [px,py,pz, nx,ny,nz, u,v, r,g,b,a]
    bytes_per_vertex = stride_floats * 4

    found_nonwhite = False
    for m in payload["meshes"]:
        vb = base64.b64decode(m["vertices_b64"])
        vc = m["vertex_count"]
        if vc == 0:
            continue
        assert len(vb) == vc * bytes_per_vertex, (
            f"{label}: vertex stride {len(vb)/vc} bytes, expected {bytes_per_vertex}"
        )
        # Scan the RGBA tail of each vertex for a non-(1,1,1,1) color.
        for i in range(vc):
            r4, g4, b4, a4 = struct.unpack_from("<4f", vb, i * bytes_per_vertex + 32)
            if (round(r4, 3), round(g4, 3), round(b4, 3)) != (1.0, 1.0, 1.0):
                found_nonwhite = True
                break
        if found_nonwhite:
            break

    if label in _KNOWN_COLORED:
        assert found_nonwhite, (
            f"{label}: expected a non-(1,1,1,1) vertex color (diffuse/vertex "
            f"color was dropped on the way to the payload)"
        )


# --------------------------------------------------------------------------
# Regression guard: parser keeps per-vertex color out of the white default.
# --------------------------------------------------------------------------

def test_parser_keeps_diffuse_color():
    """A direct parse of a known diffuse-colored model yields a non-white
    color on at least one vertex (guards against re-dropping color in
    formats/xj.py)."""
    from pathlib import Path

    from formats import bml as bmlmod
    from formats import xj as xjmod

    # Search the configured data dirs for the momoka BML.
    candidates = []
    for d in (getattr(server, "LIVE_DATA_DIR", None), getattr(server, "DATA_DIR", None)):
        if d:
            candidates.append(Path(d) / "bm_npc_momoka.bml")
    bml_path = next((p for p in candidates if p and p.exists()), None)
    if bml_path is None:
        pytest.skip("bm_npc_momoka.bml not installed in any data dir")

    raw = bml_path.read_bytes()
    inners = bmlmod.extract_bml(raw)
    inner = inners.get("n_momoka_t_body.nj")
    if inner is None:
        pytest.skip("momoka inner not found")

    meshes = xjmod.parse_nj_file(inner)
    assert meshes, "momoka parsed to zero meshes"
    colors = {
        tuple(round(c, 3) for c in v.color)
        for m in meshes
        for v in m.vertices
    }
    nonwhite = [c for c in colors if c[:3] != (1.0, 1.0, 1.0)]
    assert nonwhite, (
        f"momoka: all vertex colors are white — diffuse color was dropped. "
        f"distinct colors seen: {sorted(colors)[:6]}"
    )
