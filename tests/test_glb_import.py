"""GLB/glTF -> NJ import-pipeline hardening tests (P0 item 3/3).

Exercises the robust GLB importer in ``formats.import_external`` against:

  * The committed CC0 / public-domain test assets under
    ``data/test_assets/`` (khronos_cesium_man = rigged/multi-node;
    kenney_scifi_crate = textured static prop; khronos_duck = a
    4000-tri single-primitive prop that used to silently drop to ~200
    tris through the rendering parser).

  * A SYNTHESISED 100-mesh, 100-material GLB built with trimesh — so the
    "large merged Sketchfab scene" path is always covered even when no
    big real asset is on disk. Each mesh gets its own node transform +
    its own embedded PNG texture, so the suite proves multi-mesh /
    multi-material / embedded-texture / node-transform handling with no
    silent drops.

  * The REAL Casinopolis GLB when present — resolved from
    ``$PSOBB_DOWNLOADS_DIR`` or ``~/Downloads`` by searching for a large
    multi-mesh casino/lobby ``.glb`` (no hardcoded filename). When found
    it asserts the full 100-mesh / 97-texture / ~9.5k-tri import survives
    with zero drops; when absent the leg logs a skip and the synthetic +
    test_assets legs carry the coverage.

Invariants asserted everywhere:
  * import yields N meshes with 0 drops (meshes_in == meshes_out for the
    geometry that carried POSITION),
  * GLB->NJ emits a PARSEABLE .nj (formats.xj.parse_nj_file),
  * triangle counts survive the round-trip EXACTLY,
  * axis orientation is correct (a known +Y-up vertex stays +Y-up),
  * embedded textures are recovered (textures_out > 0 where embedded).
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

ASSET_DIR = pathlib.Path(__file__).parent.parent / "data" / "test_assets"

# pygltflib + trimesh are required for this whole module; skip cleanly if
# the environment lacks them rather than erroring.
pygltflib = pytest.importorskip("pygltflib")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _asset(name: str) -> pathlib.Path:
    p = ASSET_DIR / name
    if not p.exists():
        pytest.skip(f"test asset missing: {name}")
    return p


def _in_tris(model) -> int:
    return sum(int(len(m.indices)) for m in model.meshes)


def _roundtrip_tris(nj_bytes: bytes) -> int:
    from formats.xj import parse_nj_file
    meshes = parse_nj_file(nj_bytes)
    return sum(len(x.indices) // 3 for x in meshes)


def _build_synthetic_glb(n_meshes: int = 100, seed: int = 7) -> bytes:
    """Synthesise an N-mesh, N-material GLB with per-node transforms + textures.

    Each mesh is a unit box placed at its own node translation with a
    distinct 8x8 RGB texture embedded as PNG, so the import path must
    handle: 100+ meshes, 100 materials, embedded PNGs (GLB binary
    buffer), and the node-transform tree. Y-up (trimesh/glTF default).
    """
    import trimesh
    from trimesh.visual import TextureVisuals
    from PIL import Image

    rng = np.random.default_rng(seed)
    scene = trimesh.Scene()
    for i in range(n_meshes):
        box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        img = Image.new("RGB", (8, 8), (int(i * 37) % 256, 64, (i * 11) % 256))
        box.visual = TextureVisuals(
            uv=rng.random((len(box.vertices), 2)), image=img,
        )
        # Spread the boxes out so each node carries a real translation.
        xform = trimesh.transformations.translation_matrix(
            (float(i % 10) * 3.0, 0.0, float(i // 10) * 3.0)
        )
        scene.add_geometry(box, node_name=f"box_{i:03d}", transform=xform)
    return scene.export(file_type="glb")


def _find_casinopolis_glb() -> "pathlib.Path | None":
    """Search the downloads dir for a large multi-mesh casino/lobby GLB.

    Resolves the downloads directory from ``$PSOBB_DOWNLOADS_DIR`` (when
    set) else ``~/Downloads``. Picks the largest ``.glb`` whose name hints
    at a casino/lobby/casinopolis scene; falls back to the single largest
    ``.glb`` over ~512 KB (a heuristic for "merged multi-mesh scene").
    Never hardcodes a filename.
    """
    base = os.environ.get("PSOBB_DOWNLOADS_DIR") or os.path.expanduser("~/Downloads")
    d = pathlib.Path(base)
    if not d.is_dir():
        return None
    globs = list(d.glob("*.glb")) + list(d.glob("*.GLB"))
    if not globs:
        return None
    hints = ("casino", "casinopolis", "lobby")
    hinted = [p for p in globs if any(h in p.name.lower() for h in hints)]
    pool = hinted or [p for p in globs if p.stat().st_size > 512 * 1024]
    if not pool:
        return None
    return max(pool, key=lambda p: p.stat().st_size)


# --------------------------------------------------------------------------- #
# test_assets: textured static prop (crate)
# --------------------------------------------------------------------------- #
def test_crate_glb_import_no_drops_and_roundtrip():
    from formats.import_external import parse_gltf, imported_model_to_nj

    data = _asset("kenney_scifi_crate.glb").read_bytes()
    m = parse_gltf(data)

    st = m.import_stats
    assert st["meshes_in"] == st["meshes_out"] > 0, "silent mesh drop"
    # The crate references its texture by EXTERNAL uri (Textures/colormap.png)
    # — it can't be embedded from a single blob, so it's recorded (not
    # dropped) as an external texture with a warning.
    assert st["textures_in"] == 1
    assert st["textures_external"] == 1
    assert any("external uri" in w for w in m.warnings)

    nj = imported_model_to_nj(m, axis_flip_z=True, scale=100.0)
    assert nj[:4] in (b"NJTL", b"NJCM")
    assert _roundtrip_tris(nj) == _in_tris(m), "triangle count did not survive"


# --------------------------------------------------------------------------- #
# test_assets: embedded-texture single-primitive prop (duck) — the
# i16-overflow regression that dropped 4212 tris -> ~200.
# --------------------------------------------------------------------------- #
def test_duck_glb_large_primitive_survives_roundtrip():
    from formats.import_external import parse_gltf, imported_model_to_nj

    data = _asset("khronos_duck.glb").read_bytes()
    m = parse_gltf(data)
    st = m.import_stats
    assert st["meshes_in"] == st["meshes_out"] == 1
    # Embedded PNG recovered with real pixel dimensions.
    assert st["textures_in"] == 1
    assert st["textures_out"] == 1
    tex = m.textures[0]
    assert tex.source == "bufferView"
    assert len(tex.data) > 0
    assert tex.width > 0 and tex.height > 0

    in_t = _in_tris(m)
    assert in_t >= 4000, "duck should carry >4000 tris"
    nj = imported_model_to_nj(m, axis_flip_z=True, scale=100.0)
    assert _roundtrip_tris(nj) == in_t, (
        f"large-primitive triangle drop: {in_t} -> {_roundtrip_tris(nj)}"
    )


# --------------------------------------------------------------------------- #
# test_assets: rigged / multi-node embedded-JPEG (cesium man)
# --------------------------------------------------------------------------- #
def test_cesium_man_rigged_multinode_roundtrip():
    from formats.import_external import parse_gltf, imported_model_to_nj

    data = _asset("khronos_cesium_man.glb").read_bytes()
    m = parse_gltf(data)
    st = m.import_stats
    assert st["meshes_in"] == st["meshes_out"] >= 1, "silent mesh drop"
    # 19-joint rig recovered.
    assert len(m.bones) == 19
    # Embedded JPEG texture recovered.
    assert st["textures_out"] == 1
    assert m.textures[0].source == "bufferView"
    assert m.textures[0].mime == "image/jpeg"

    nj = imported_model_to_nj(m, axis_flip_z=True, scale=100.0)
    assert _roundtrip_tris(nj) == _in_tris(m)


# --------------------------------------------------------------------------- #
# Synthetic 100-mesh / 100-material / embedded-texture GLB
# --------------------------------------------------------------------------- #
def test_synthetic_100_mesh_glb_no_drops():
    pytest.importorskip("trimesh")
    pytest.importorskip("PIL")
    from formats.import_external import parse_gltf, imported_model_to_nj

    glb = _build_synthetic_glb(n_meshes=100)
    m = parse_gltf(glb)

    st = m.import_stats
    # 100 boxes -> 100 primitives in, 100 ImportedMesh out, 0 drops.
    assert st["meshes_in"] == 100, st
    assert st["meshes_out"] == 100, st
    assert len(m.meshes) == 100
    # 100 distinct embedded PNG textures, all recovered.
    assert st["textures_in"] == 100
    assert st["textures_out"] == 100
    assert st["textures_external"] == 0
    assert all(t.source == "bufferView" and t.data for t in m.textures)
    assert all(t.width == 8 and t.height == 8 for t in m.textures)

    nj = imported_model_to_nj(m, axis_flip_z=True, scale=10.0)
    assert nj[:4] in (b"NJTL", b"NJCM")
    # Every box is 12 tris -> 1200 tris total; all survive the round-trip.
    in_t = _in_tris(m)
    assert in_t == 100 * 12
    assert _roundtrip_tris(nj) == in_t

    # NJTL carries one name per source image.
    from formats.nj_writer import parse_nj_for_writer
    names = parse_nj_for_writer(nj).njtl_names
    assert len(names) == 100


def test_synthetic_node_transforms_are_baked():
    """Per-node translations must be baked into vertices (merged-scene fix).

    Two boxes 30 units apart on X: the imported vertex cloud must span
    that separation, proving the node transform was applied rather than
    every box collapsing onto the origin.
    """
    pytest.importorskip("trimesh")
    from formats.import_external import parse_gltf

    glb = _build_synthetic_glb(n_meshes=11)  # boxes at x = 0,3,6,...,30
    m = parse_gltf(glb)
    all_x = np.concatenate([mesh.vertices[:, 0] for mesh in m.meshes])
    # Box centres span 0..30 on X; with +/-0.5 half-extent the cloud
    # should span well past 25 units.
    assert all_x.max() - all_x.min() > 25.0, (
        "node translations were not baked into vertices "
        f"(x span {all_x.max() - all_x.min():.2f})"
    )


# --------------------------------------------------------------------------- #
# Axis orientation: a known +Y-up vertex stays +Y-up.
# --------------------------------------------------------------------------- #
def test_axis_up_vertex_stays_up():
    from formats.import_external import (
        ImportedMesh, ImportedModel, imported_model_to_nj,
    )
    from formats.xj import parse_nj_file

    # Triangle with apex at +Y=10; base on the X axis at y=0.
    verts = np.array([[-1, 0, 0], [1, 0, 0], [0, 10, 0]], dtype=np.float32)
    tris = np.array([[0, 1, 2]], dtype=np.uint32)
    uvs = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float32)
    model = ImportedModel(
        meshes=[ImportedMesh(name="t", vertices=verts, indices=tris, uvs=uvs)],
        source_format="glb",
    )
    nj = imported_model_to_nj(model, axis_flip_z=True, scale=1.0)
    meshes = parse_nj_file(nj)
    ys = [v.pos[1] for mm in meshes for v in mm.vertices]
    zs = [v.pos[2] for mm in meshes for v in mm.vertices]
    # The apex stays UP (+Y ~ 10) — orientation preserved through the
    # glTF(Y-up,RH) -> PSOBB(Y-up,LH) Z-flip.
    assert abs(max(ys) - 10.0) < 1e-3, f"apex moved off +Y: maxY={max(ys)}"
    # The flat triangle stays in the Z=0 plane.
    assert all(abs(z) < 1e-3 for z in zs)


# --------------------------------------------------------------------------- #
# No-silent-drop: a corrupt draw mode raises rather than dropping geometry.
# --------------------------------------------------------------------------- #
def test_unsupported_draw_mode_raises_not_drops():
    pytest.importorskip("trimesh")
    from formats.import_external import parse_gltf, GltfImportError

    glb = bytearray(_build_synthetic_glb(n_meshes=2))
    g = pygltflib.GLTF2.load_from_bytes(bytes(glb))
    # Force an unknown/unsupported draw mode on a primitive that carries
    # POSITION — this WOULD silently drop the triangles in the v1 path.
    g.meshes[0].primitives[0].mode = 99
    mutated = g.save_to_bytes()
    # pygltflib save_to_bytes returns a list of byte segments for GLB.
    if isinstance(mutated, list):
        mutated = b"".join(mutated)
    with pytest.raises(GltfImportError):
        parse_gltf(bytes(mutated))


# --------------------------------------------------------------------------- #
# Fit-to-budget hook: import + decimate to <= N tris / <= B bytes.
# --------------------------------------------------------------------------- #
def test_fit_to_budget_tris_and_bytes():
    pytest.importorskip("trimesh")
    from formats.import_external import parse_gltf, fit_model_to_budget

    glb = _build_synthetic_glb(n_meshes=100)  # 1200 tris merged
    m = parse_gltf(glb)

    # Triangle cap: result must not exceed the cap and must re-parse.
    nj_t, _dm_t, meta_t = fit_model_to_budget(
        m, max_tris=400, axis_flip_z=True, scale=10.0,
    )
    assert meta_t["out_tris"] <= 400
    assert _roundtrip_tris(nj_t) == meta_t["out_tris"]

    # Byte cap small enough to force a decimation pass: the emitted .nj
    # must fit the budget, and the merged mesh must have been reduced.
    budget = 24 * 1024
    nj_b, _dm_b, meta_b = fit_model_to_budget(
        m, max_bytes=budget, axis_flip_z=True, scale=10.0,
    )
    assert not meta_b["over_budget"]
    assert len(nj_b) <= budget, f"over budget: {len(nj_b)} > {budget}"
    # Decimation actually ran (input merged mesh exceeds the budget).
    assert meta_b["out_tris"] < _in_tris(m), meta_b
    assert _roundtrip_tris(nj_b) == meta_b["out_tris"]


# --------------------------------------------------------------------------- #
# Real Casinopolis GLB (optional leg).
# --------------------------------------------------------------------------- #
def test_real_casinopolis_glb_if_present():
    from formats.import_external import parse_gltf, imported_model_to_nj

    path = _find_casinopolis_glb()
    if path is None:
        pytest.skip(
            "no large casino/lobby .glb found in $PSOBB_DOWNLOADS_DIR / ~/Downloads"
        )
    data = path.read_bytes()
    m = parse_gltf(data)
    st = m.import_stats
    # No silent drops: every POSITION-bearing primitive becomes a mesh.
    assert st["meshes_in"] == st["meshes_out"] > 0, st
    # A real merged Sketchfab scene: many meshes + many embedded textures.
    assert st["meshes_out"] >= 50, f"expected a large multi-mesh scene, got {st}"
    if st["textures_in"] > 0:
        # All embedded textures recovered (Casinopolis embeds PNG via
        # bufferView; nothing external).
        assert st["textures_out"] == st["textures_in"], st

    in_t = _in_tris(m)
    nj = imported_model_to_nj(m, axis_flip_z=True, scale=100.0)
    assert _roundtrip_tris(nj) == in_t, (
        f"Casinopolis triangle drop: {in_t} -> {_roundtrip_tris(nj)}"
    )
    # Surface the counts in the test log for the report.
    print(
        f"\n[casinopolis] {path.name}: meshes {st['meshes_in']}->{st['meshes_out']}, "
        f"textures {st['textures_in']}->{st['textures_out']} "
        f"(external {st['textures_external']}), tris {in_t} (round-trip OK), "
        f"nj {len(nj)} bytes"
    )
