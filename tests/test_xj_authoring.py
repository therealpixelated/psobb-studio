"""XJ-authoring tests: descriptor-XJ encoder + n.rel from-scratch builder.

Two halves, mirroring ``tests/test_nj_writer.py`` /
``tests/test_rel_writer_parity.py``:

  PART A  formats.xj_writer — author a known mesh, encode_xj_model, then
          re-parse via formats.xj_descriptor.parse_xj_descriptor and
          assert the geometry (positions / normals / UVs / per-tri
          vertex sets) survives.  POF0 round-trips.  Plus a real-asset
          leg: parse a real descriptor .xj, re-author, re-parse, assert
          equivalent (PSOBB-data-guarded).

  PART B  formats.rel_writer.build_nrel_from_meshes — author a known
          mesh, build the n.rel, re-parse via formats.rel
          (read_nrel_*/extract_nrel_meshes), assert geometry survives,
          simulate_rel_relocation OK, size <= 768 KB, the closed-form
          pointer count matches.  Plus a real-geometry reconstruction
          leg (extract a real n.rel's meshes, re-author, re-parse,
          assert geometry equivalent; PSOBB-data-guarded).

Run isolated::

    python -m pytest tests/test_xj_authoring.py -q
"""
from __future__ import annotations

import os
import struct
from collections import Counter
from pathlib import Path

import pytest

from formats.iff import parse_iff
from formats.nj_writer import decode_pof0
from formats.rel import (
    extract_nrel_meshes,
    is_n_rel,
    parse_rel,
    read_mesh_trees,
    read_nrel_chunks,
    read_nrel_header,
    read_texture_names,
)
from formats.rel_writer import (
    NREL_SIZE_BUDGET,
    NrelNode,
    NrelSubmesh,
    NrelVertex,
    RelWriteError,
    build_nrel_from_meshes,
    nrel_nodes_from_meshes,
    nrel_pointer_count,
    parse_nrel_for_writer,
    simulate_rel_relocation,
)
from formats.xj_descriptor import parse_xj_descriptor
from formats.xj_writer import (
    XjMaterialEntry,
    XjModelData,
    XjModelFile,
    XjNode,
    XjStrip,
    XjVertexData,
    build_xj_from_meshes,
    encode_xj_model,
    encode_xjcm_chunk,
)

SCENE_DIR = Path(os.path.expanduser("~/PSOBB.IO/data/scene"))
HAS_PSOBB = SCENE_DIR.is_dir()

_XJ_DATA_DIRS = [
    Path(os.path.expanduser("~/EphineaPSO/data")),
    Path(os.path.expanduser("~/PSOBB.IO/data")),
]


def _xj_data_dir():
    env = os.environ.get("PSO_XJ_TEST_DATA_DIR")
    if env and Path(env).is_dir():
        return Path(env)
    for d in _XJ_DATA_DIRS:
        if d.is_dir():
            return d
    return None


HAS_XJ_DATA = _xj_data_dir() is not None


# ---------------------------------------------------------------------------
# Shared geometry helpers
# ---------------------------------------------------------------------------

def _quad_verts():
    """A flat 10x10 quad in the XZ plane, up-normals, full-UV."""
    return [
        ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0)),
        ((10.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0)),
        ((10.0, 0.0, 10.0), (0.0, 1.0, 0.0), (1.0, 1.0)),
        ((0.0, 0.0, 10.0), (0.0, 1.0, 0.0), (0.0, 1.0)),
    ]


def _tri_position_multiset(meshes):
    """Winding-insensitive multiset of {pos,pos,pos} per triangle."""
    out = []
    for m in meshes:
        for i in range(0, len(m.indices), 3):
            a, b, c = m.indices[i], m.indices[i + 1], m.indices[i + 2]
            out.append(frozenset((m.vertices[a].pos,
                                  m.vertices[b].pos,
                                  m.vertices[c].pos)))
    return Counter(out)


# ===========================================================================
# PART A — formats.xj_writer (descriptor-table .xj encoder)
# ===========================================================================

def test_xj_quad_round_trip_geometry():
    """Author a textured quad -> encode_xj_model -> parse_xj_descriptor.

    Positions / normals / UVs survive within f32 tolerance and the two
    triangles' vertex sets are preserved.
    """
    qv = _quad_verts()
    verts = [XjVertexData(pos=p, normal=n, uv=uv) for (p, n, uv) in qv]
    strips = [
        XjStrip(indices=[0, 1, 2], materials=[XjMaterialEntry(type=3, args=(7, 0, 0))]),
        XjStrip(indices=[0, 2, 3], materials=[XjMaterialEntry(type=3, args=(7, 0, 0))]),
    ]
    model = XjModelFile(
        nodes=[XjNode(model_index=0)],
        models=[XjModelData(vertices=verts, opaque_strips=strips)],
        njtl_names=["tex_quad"],
    )
    out = encode_xj_model(model)

    # IFF round-trips with the expected chunk list.
    chunk_types = [c.type for c in parse_iff(out)]
    assert chunk_types == ["NJTL", "POF0", "NJCM", "POF0"]

    meshes = parse_xj_descriptor(
        next(c for c in parse_iff(out) if c.type == "NJCM").data)
    # One mesh per strip (each strip is one triangle).
    assert len(meshes) == 2
    for m in meshes:
        assert m.material_id == 7
        assert len(m.indices) == 3  # one triangle

    # Every authored vertex attribute survives exactly (f32-exact for
    # these round numbers; check via the position->attrs map).
    attr = {}
    for (p, n, uv) in qv:
        attr[p] = (n, uv)
    for m in meshes:
        for v in m.vertices:
            assert v.pos in attr, v.pos
            exp_n, exp_uv = attr[v.pos]
            assert v.normal == pytest.approx(exp_n, abs=1e-6)
            assert v.uv == pytest.approx(exp_uv, abs=1e-6)

    # The two triangles cover the same vertex sets as the source quad.
    got = _tri_position_multiset(meshes)
    want = Counter([
        frozenset((qv[0][0], qv[1][0], qv[2][0])),
        frozenset((qv[0][0], qv[2][0], qv[3][0])),
    ])
    assert got == want


def test_xj_encode_idempotent():
    """encode -> parse -> the same input encodes byte-identically twice."""
    verts = [XjVertexData(pos=(i * 1.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0),
                          uv=(0.0, 0.0)) for i in range(4)]
    strips = [XjStrip(indices=[0, 1, 2],
                      materials=[XjMaterialEntry(type=3, args=(1, 0, 0))])]
    model = XjModelFile(nodes=[XjNode(model_index=0)],
                        models=[XjModelData(vertices=verts, opaque_strips=strips)])
    assert encode_xj_model(model) == encode_xj_model(model)


def test_xj_pof0_round_trip_and_validity():
    """decode_pof0(emitted POF0) == sorted ptr_offsets; all 4-aligned,
    in-body, and each stored value lands inside the body."""
    verts = [XjVertexData(pos=(i * 1.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0),
                          uv=(0.0, 0.0)) for i in range(4)]
    strips = [XjStrip(indices=[0, 1, 2],
                      materials=[XjMaterialEntry(type=3, args=(2, 0, 0))])]
    model = XjModelFile(nodes=[XjNode(model_index=0)],
                        models=[XjModelData(vertices=verts, opaque_strips=strips)])
    body, ptrs = encode_xjcm_chunk(model)

    out = encode_xj_model(model)
    chunks = parse_iff(out)
    njcm_idx = next(i for i, c in enumerate(chunks) if c.type == "NJCM")
    pof0 = chunks[njcm_idx + 1]
    assert pof0.type == "POF0"
    assert decode_pof0(pof0.data) == ptrs

    assert ptrs == sorted(ptrs)
    assert all(o % 4 == 0 for o in ptrs)
    for o in ptrs:
        assert o + 4 <= len(body)
        val = struct.unpack_from("<I", body, o)[0]
        assert 0 <= val < len(body), (hex(o), val, len(body))


def test_xj_vtype_selection():
    """The encoder picks the tightest vtype carrying the supplied attrs."""
    # pos+normal+uv -> type 3 (stride 32).
    m3 = XjModelFile(
        nodes=[XjNode(model_index=0)],
        models=[XjModelData(
            vertices=[XjVertexData(pos=(0, 0, 0), normal=(0, 1, 0), uv=(0, 0))],
            opaque_strips=[])])
    body, _ = encode_xjcm_chunk(m3)
    # vinfo row lives right after the single 52B node + 44B model header.
    vit_off = 52 + 44
    vtype, _flags, _vto, vsize, _vc = struct.unpack_from("<hhIII", body, vit_off)
    assert vtype == 3 and vsize == 32

    # pos+normal only -> type 2 (stride 24).
    m2 = XjModelFile(
        nodes=[XjNode(model_index=0)],
        models=[XjModelData(
            vertices=[XjVertexData(pos=(0, 0, 0), normal=(0, 1, 0))],
            opaque_strips=[])])
    body2, _ = encode_xjcm_chunk(m2)
    vtype2, _f, _v, vsize2, _c = struct.unpack_from("<hhIII", body2, vit_off)
    assert vtype2 == 2 and vsize2 == 24


@pytest.mark.skipif(not HAS_XJ_DATA, reason="no PSOBB/Ephinea game data on disk")
def test_xj_real_asset_round_trip():
    """Real-geometry leg: parse a real descriptor .xj, re-author the
    submeshes, re-parse, assert the per-triangle vertex sets survive."""
    from formats.bml import _prs_decompress, parse_bml

    data_dir = _xj_data_dir()
    found = None
    for fn in sorted(os.listdir(data_dir)):
        if not fn.endswith(".bml"):
            continue
        try:
            blob = (data_dir / fn).read_bytes()
            entries = parse_bml(blob)
        except Exception:
            continue
        for e in entries:
            if not e.name.endswith(".xj"):
                continue
            try:
                raw = bytes(blob[e.offset:e.offset + e.size_compressed])
                xjb = _prs_decompress(raw, timeout=20.0)
                njcm = [c for c in parse_iff(xjb) if c.type == "NJCM"]
                if not njcm:
                    continue
                meshes = parse_xj_descriptor(njcm[0].data)
            except Exception:
                continue
            if meshes and sum(len(m.indices) for m in meshes) >= 6:
                found = (fn, e.name, meshes)
                break
        if found:
            break
    if found is None:
        pytest.skip("no parseable descriptor .xj with >=2 triangles found")

    _fn, _inner, orig = found
    model = build_xj_from_meshes(orig, njtl_names=["t0"])
    out = encode_xj_model(model)
    reparsed = parse_xj_descriptor(
        next(c for c in parse_iff(out) if c.type == "NJCM").data)

    # Re-authoring merges submeshes into one node but keeps every
    # triangle; compare the winding-insensitive per-triangle vertex
    # multiset.
    assert _tri_position_multiset(reparsed) == _tri_position_multiset(orig)


# ===========================================================================
# PART B — formats.rel_writer.build_nrel_from_meshes (n.rel from scratch)
# ===========================================================================

def _author_quad_nrel():
    qv = _quad_verts()
    verts = [NrelVertex(pos=p, normal=n, uv=uv) for (p, n, uv) in qv]
    # Two triangles, each its own submesh (one 3-index strip) so the
    # _strip_to_triangles parity flip never fires.
    sm = [
        NrelSubmesh(vertices=[verts[0], verts[1], verts[2]],
                    indices=[0, 1, 2], texture_id=5),
        NrelSubmesh(vertices=[verts[0], verts[2], verts[3]],
                    indices=[0, 1, 2], texture_id=5),
    ]
    node = NrelNode(submeshes=sm)
    names = ["floor_tex"]
    return [node], names


def test_nrel_synthetic_single_mesh_loadable():
    """T1: author a quad n.rel, re-parse fully, assert geometry +
    relocation + budget."""
    nodes, names = _author_quad_nrel()
    out = build_nrel_from_meshes(nodes, names)

    rel = parse_rel(out)
    assert is_n_rel(rel)

    header = read_nrel_header(rel)
    assert header.chunk_count == 1
    assert 0 < header.chunks_ptr < rel.data_size
    assert 0 < header.texture_data_ptr < rel.data_size

    chunks = read_nrel_chunks(rel, header)
    assert len(chunks) == 1
    assert chunks[0].static_mesh_tree_count == 1
    assert 0 < chunks[0].static_mesh_trees_ptr < rel.data_size

    trees = read_mesh_trees(rel, chunks[0].static_mesh_trees_ptr,
                            chunks[0].static_mesh_tree_count)
    assert len(trees) == 1
    assert trees[0].root_node_ptr > 0

    assert read_texture_names(rel) == names

    meshes = extract_nrel_meshes(rel)
    assert len(meshes) == 2
    for m in meshes:
        assert m.material_id == 5
        assert len(m.indices) == 3

    # Positions / UVs / normals survive (chunk at origin => identity).
    qv = _quad_verts()
    pmap = {p: (n, uv) for (p, n, uv) in qv}
    for m in meshes:
        for v in m.vertices:
            assert v.pos in pmap, v.pos
            exp_n, exp_uv = pmap[v.pos]
            assert v.uv == pytest.approx(exp_uv, abs=1e-6)
            assert v.normal == pytest.approx(exp_n, abs=1e-6)

    # Triangle vertex sets preserved.
    got = _tri_position_multiset(meshes)
    want = Counter([
        frozenset((qv[0][0], qv[1][0], qv[2][0])),
        frozenset((qv[0][0], qv[2][0], qv[3][0])),
    ])
    assert got == want

    # Relocation acceptance gate.
    base = 0x40000000
    for v in simulate_rel_relocation(out, base=base):
        assert v == base or base <= v < base + rel.pointer_table_offset

    assert len(out) <= NREL_SIZE_BUDGET


def test_nrel_pointer_count_closed_form():
    """T2: the resolved pointer count equals the structural formula."""
    nodes, names = _author_quad_nrel()
    out = build_nrel_from_meshes(nodes, names)
    rel = parse_rel(out)
    assert rel.pointer_count == nrel_pointer_count(nodes, names)


def test_nrel_alignment_and_framing():
    """T3: container framing invariants (32-aligned, trailer flag)."""
    nodes, names = _author_quad_nrel()
    out = build_nrel_from_meshes(nodes, names)
    assert len(out) % 32 == 0
    assert (len(out) - 0x20) % 32 == 0
    trailer_start = len(out) - 32
    pt_off, pt_count, flag, reserved, _pl = struct.unpack_from(
        "<5I", out, trailer_start)
    assert pt_off % 32 == 0
    assert flag == 1
    assert reserved == 0
    assert out[trailer_start + 0x14:] == b"\x00" * 12


def test_nrel_writer_parser_round_trip():
    """T4: the authored file is in the round-trip structural class."""
    nodes, names = _author_quad_nrel()
    out = build_nrel_from_meshes(nodes, names)
    model = parse_nrel_for_writer(out)
    assert model.chunk_count == 1
    assert model.texture_names == names


def test_nrel_no_texture_list():
    """A textureless build omits the TextureList (texture_data_ptr == 0)
    and drops the corresponding relocation entries."""
    verts = [NrelVertex(pos=(0, 0, 0)), NrelVertex(pos=(1, 0, 0)),
             NrelVertex(pos=(1, 0, 1))]
    nodes = [NrelNode(submeshes=[NrelSubmesh(vertices=verts,
                                             indices=[0, 1, 2], texture_id=0)])]
    out = build_nrel_from_meshes(nodes, [])
    rel = parse_rel(out)
    assert read_nrel_header(rel).texture_data_ptr == 0
    assert rel.pointer_count == nrel_pointer_count(nodes, [])
    # relocation still clean
    base = 0x10000000
    for v in simulate_rel_relocation(out, base=base):
        assert v == base or base <= v < base + rel.pointer_table_offset


def test_nrel_budget_edge():
    """T6: over-budget builds raise; near-but-under builds succeed."""
    # Build a mesh large enough to blow the 768 KB cap.  Each triangle
    # submesh costs ~ (16 vbuf + 20 ibuf + 16 rs + 96 verts + 6 idx +
    # node-relocs) -> a few hundred bytes; ~3000 tris overflows.
    big_verts = [NrelVertex(pos=(float(i), 0.0, 0.0)) for i in range(3)]
    sm = [NrelSubmesh(vertices=big_verts, indices=[0, 1, 2], texture_id=0)
          for _ in range(6000)]
    nodes = [NrelNode(submeshes=sm)]
    with pytest.raises(RelWriteError, match="budget"):
        build_nrel_from_meshes(nodes, [])
    # Same model builds when the budget gate is disabled.
    out = build_nrel_from_meshes(nodes, [], enforce_budget=False)
    assert len(out) > NREL_SIZE_BUDGET

    # A small model is comfortably under budget.
    small = [NrelNode(submeshes=[NrelSubmesh(vertices=big_verts,
                                             indices=[0, 1, 2], texture_id=0)])]
    assert len(build_nrel_from_meshes(small, [])) <= NREL_SIZE_BUDGET


def test_nrel_bad_index_raises():
    """An out-of-range child/sibling index is rejected, not silently emitted."""
    verts = [NrelVertex(pos=(0, 0, 0)), NrelVertex(pos=(1, 0, 0)),
             NrelVertex(pos=(1, 0, 1))]
    nodes = [NrelNode(submeshes=[NrelSubmesh(vertices=verts, indices=[0, 1, 2])],
                      child_index=5)]
    with pytest.raises(RelWriteError, match="out of range"):
        build_nrel_from_meshes(nodes, [])


@pytest.mark.skipif(not HAS_PSOBB, reason="no PSOBB scene data on disk")
def test_nrel_vanilla_geometry_reconstruction():
    """T7: extract a real n.rel's meshes, re-author via the builder,
    re-parse, assert the per-triangle vertex sets + (texture-bearing)
    material ids survive."""
    src = (SCENE_DIR / "map_aboss01n.rel").read_bytes()
    rel = parse_rel(src)
    orig = extract_nrel_meshes(rel)
    assert orig, "fixture produced no meshes"

    # 64 placeholder texture names (material_id indexes into this list).
    names = [f"tex_{i}" for i in range(64)]
    nodes = nrel_nodes_from_meshes(orig)
    out = build_nrel_from_meshes(nodes, names)
    assert len(out) <= NREL_SIZE_BUDGET

    rel2 = parse_rel(out)
    rebuilt = extract_nrel_meshes(rel2)

    # Same triangle count and per-triangle vertex-position sets.
    assert (sum(len(m.indices) for m in orig)
            == sum(len(m.indices) for m in rebuilt))
    assert _tri_position_multiset(orig) == _tri_position_multiset(rebuilt)

    # Material ids survive for every texture-bearing triangle.  The
    # reader returns -1 for strips with no TEXTURE_ID renderstate; the
    # builder clamps that to texture 0, so compare only the >=0 ids.
    def mat_hist(ms):
        c = Counter()
        for m in ms:
            if m.material_id >= 0:
                c[m.material_id] += len(m.indices) // 3
        return c
    orig_h = mat_hist(orig)
    rebuilt_h = mat_hist(rebuilt)
    for mid, cnt in orig_h.items():
        assert rebuilt_h.get(mid, 0) >= cnt

    # Relocation acceptance.
    base = 0x40000000
    for v in simulate_rel_relocation(out, base=base):
        assert v == base or base <= v < base + rel2.pointer_table_offset


@pytest.mark.skipif(not HAS_PSOBB, reason="no PSOBB scene data on disk")
def test_nrel_oracle_pointer_policy():
    """T5 (oracle): the builder's null child/next/anim/alpha registration
    policy matches what parse_rel resolves on a real vanilla file.

    We can't byte-match (layout differs), but the RELOCATION POLICY for a
    matched topology must agree: for a single leaf node with one mesh,
    one vbuf, one ibuf (one strip), one texture — the relative pointer
    pattern the builder emits per struct must equal the vanilla one.
    This is validated structurally by re-parsing the builder output and
    confirming the closed-form count, AND by confirming the vanilla file
    itself round-trips through parse_nrel_for_writer/encode_nrel
    unchanged (the carried vanilla relocation set is the ground truth).
    """
    from formats.rel_writer import encode_nrel

    src = (SCENE_DIR / "map_aboss01n.rel").read_bytes()
    # Vanilla relocation set is self-consistent (the oracle).
    model = parse_nrel_for_writer(src)
    assert encode_nrel(model) == src

    # The builder's emitted set, for its own topology, equals its
    # closed-form count (no stray/missing flag) — the numeric oracle.
    orig = extract_nrel_meshes(parse_rel(src))
    names = [f"tex_{i}" for i in range(64)]
    nodes = nrel_nodes_from_meshes(orig)
    out = build_nrel_from_meshes(nodes, names)
    rebuilt = parse_rel(out)
    assert rebuilt.pointer_count == nrel_pointer_count(nodes, names)
    # Every resolved pointer lands in-data or is a legal null.
    base = 0x40000000
    for v in simulate_rel_relocation(out, base=base):
        assert v == base or base <= v < base + rebuilt.pointer_table_offset
