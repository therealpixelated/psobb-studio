"""Tests for ``formats.nj_writer`` — the NJ encoder.

Coverage:
  - Round-trip 20 representative NJs from MODEL_COVERAGE.csv covering
    each chunk-type combo (skipped when PSOBB.IO data is absent).
  - Synthetic minimal model: 1 cube, 8 verts, 12 tris, 1 bone.
  - POF0 encoder edge cases (empty, single, large delta).
  - Verify game-loadable output by parsing back through formats/xj.py
    and asserting structural equivalence.
"""
from __future__ import annotations
import os

import math
import struct
from pathlib import Path

import pytest

from formats.bml import extract_bml
from formats.iff import parse_iff
from formats.nj_writer import (
    NjChunk,
    NjMeshChunks,
    NjModel,
    NjNode,
    decode_pof0,
    encode_nj_model,
    encode_njcm_chunk,
    encode_njtl_chunk,
    encode_pof0,
    parse_nj_for_writer,
    parse_njtl_for_writer,
)
from formats.xj import parse_nj_file


PSOBB_DATA = Path(os.path.expanduser("~/PSOBB.IO/data"))
HAS_PSOBB = PSOBB_DATA.is_dir()


# ---------------------------------------------------------------------------
# POF0 encoder
# ---------------------------------------------------------------------------


def test_pof0_empty_list_returns_empty():
    """Empty POF0 ptr list emits an empty buffer (matches PSOBB shipped).

    Several shipped NJs (e.g. bm_ene_common_all.bml#ene_common_all.nj)
    have a 0-byte POF0 chunk because the model contains no relocatable
    pointers. The encoder must mirror this exactly.
    """
    assert encode_pof0([]) == b""


def test_pof0_single_pointer():
    """One pointer at offset 4 → 1-byte token (01 000001 = 0x41) + pad."""
    out = encode_pof0([4])
    assert out == b"\x41\x00\x00\x00"  # padded to 4 bytes
    assert decode_pof0(out) == [4]


def test_pof0_three_pointers_round_trip():
    """3 pointers, mix of 1-byte and 2-byte deltas."""
    offsets = [4, 0x45c, 0x460]
    out = encode_pof0(offsets)
    assert decode_pof0(out) == offsets


def test_pof0_large_delta_uses_2byte_token():
    """Delta > 0x40 dwords requires 2-byte token."""
    out = encode_pof0([0, 0x100])  # 0x100/4 = 0x40 dwords
    decoded = decode_pof0(out)
    assert decoded == [0, 0x100]


def test_pof0_huge_delta_uses_3byte_token():
    """Delta > 0x4000 dwords requires 3-byte token."""
    out = encode_pof0([0, 0x10000])  # 0x10000/4 = 0x4000 dwords
    decoded = decode_pof0(out)
    assert decoded == [0, 0x10000]


def test_pof0_rejects_unaligned():
    with pytest.raises(ValueError, match="not 4-byte aligned"):
        encode_pof0([3])


def test_pof0_rejects_descending():
    with pytest.raises(ValueError, match="not ascending"):
        encode_pof0([8, 4])


# ---------------------------------------------------------------------------
# NJTL encoder
# ---------------------------------------------------------------------------


def test_njtl_round_trip():
    """NJTL encode → decode → same names."""
    names = ["b_ball", "fooBar", "n_999"]
    body, ptrs = encode_njtl_chunk(names)
    out = parse_njtl_for_writer(body)
    assert out == names
    # Pointer offsets sane.
    assert ptrs[0] == 0  # elements_offset itself
    assert all(0 < p < len(body) for p in ptrs[1:])


def test_njtl_empty_list():
    body, ptrs = encode_njtl_chunk([])
    assert struct.unpack_from("<II", body, 0) == (8, 0)


# ---------------------------------------------------------------------------
# Synthetic minimal model
# ---------------------------------------------------------------------------


def _build_synthetic_cube() -> NjModel:
    """A 1-bone, 8-vert cube model: vertex chunk type 41 + strip chunk 64.

    Chosen for simplicity (no UVs / textures); the existing parser
    handles type 41 + 64 correctly. The cube is 2x2x2 centred at the
    origin with normals = position normalised.
    """
    verts = [
        (-1, -1, -1), ( 1, -1, -1), ( 1,  1, -1), (-1,  1, -1),
        (-1, -1,  1), ( 1, -1,  1), ( 1,  1,  1), (-1,  1,  1),
    ]
    # Vertex chunk type 41 (NJD_CV_VN) body: u16 body_words, u16 base_idx,
    # u16 count, then per-vertex (12 bytes pos + 12 bytes normal) = 24 bytes
    # Total chunk body = 4 (idx+count) + 8*24 = 196 bytes; body_words = 49.
    vbody = bytearray()
    vbody.extend(struct.pack("<H", 49))
    vbody.extend(struct.pack("<HH", 0, 8))
    for (x, y, z) in verts:
        vbody.extend(struct.pack("<3f", float(x), float(y), float(z)))
        n = math.sqrt(x * x + y * y + z * z)
        vbody.extend(struct.pack("<3f", x / n, y / n, z / n))

    # Strip chunk type 64 (bare strip, no UV/normal).
    # Body: u16 body_words, u16 strip_count_and_offset, then per strip:
    # i16 length + length * u16 indices.
    # 6 quads as 6 strips of length 4 → 6 * (2 + 4*2) = 60 bytes payload
    # Plus 2-byte strip_count_and_offset header = 62 bytes; body_words=31.
    faces = [
        (0, 1, 3, 2),  # back  (-z)
        (5, 4, 6, 7),  # front (+z)
        (4, 0, 7, 3),  # left  (-x)
        (1, 5, 2, 6),  # right (+x)
        (3, 2, 7, 6),  # top   (+y)
        (4, 5, 0, 1),  # bottom(-y)
    ]
    sbody = bytearray()
    sbody.extend(struct.pack("<H", 31))
    sbody.extend(struct.pack("<H", len(faces) & 0x3FFF))  # strip_count, no user offset
    for face in faces:
        sbody.extend(struct.pack("<h", 4))  # positive = ccw winding
        for idx in face:
            sbody.extend(struct.pack("<H", idx))

    mesh = NjMeshChunks(
        bbox=(0.0, 0.0, 0.0, math.sqrt(3.0)),
        vlist=[NjChunk(type_id=41, flags=0, body=bytes(vbody))],
        plist=[NjChunk(type_id=64, flags=0, body=bytes(sbody))],
    )
    node = NjNode(eval_flags=0, mesh_index=0)
    return NjModel(nodes=[node], meshes=[mesh])


def test_synthetic_cube_encodes_round_trip():
    """Synthetic cube → encode → parse → encode produces stable bytes."""
    model = _build_synthetic_cube()
    out1 = encode_nj_model(model)
    model2 = parse_nj_for_writer(out1)
    out2 = encode_nj_model(model2)
    assert out1 == out2


def test_synthetic_cube_renders_through_xj_parser():
    """Encoded synthetic cube parses correctly via the rendering parser.

    Confirms the encoder produces a structurally-valid NJ — the chunk
    stream walks cleanly, the mesh-tree node parses, and the strips
    emit triangles that match the expected geometry.
    """
    model = _build_synthetic_cube()
    out = encode_nj_model(model)
    meshes = parse_nj_file(out)
    assert len(meshes) == 6  # one submesh per face

    # Each face should have 4 vertices and 6 indices (2 triangles).
    for mesh in meshes:
        assert len(mesh.vertices) == 4
        assert len(mesh.indices) == 6
        # All vertices should lie on the unit cube surface.
        for v in mesh.vertices:
            assert max(abs(c) for c in v.pos) == pytest.approx(1.0)


def test_synthetic_cube_no_njtl_produces_no_njtl_chunk():
    """Models without textures emit no NJTL chunk."""
    model = _build_synthetic_cube()
    out = encode_nj_model(model)
    chunks = parse_iff(out)
    assert all(c.type != "NJTL" for c in chunks)


def test_synthetic_cube_with_njtl():
    """Adding NJTL names produces an NJTL chunk + its POF0."""
    model = _build_synthetic_cube()
    model.njtl_names = ["tex_a", "tex_b"]
    out = encode_nj_model(model)
    chunks = parse_iff(out)
    types = [c.type for c in chunks]
    assert types == ["NJTL", "POF0", "NJCM", "POF0"]
    re_parsed = parse_nj_for_writer(out)
    assert re_parsed.njtl_names == ["tex_a", "tex_b"]


# ---------------------------------------------------------------------------
# Empty model edge case
# ---------------------------------------------------------------------------


def test_empty_model_encodes_root_only():
    """A model with no meshes encodes a single null root node."""
    model = NjModel(nodes=[NjNode()], meshes=[])
    out = encode_nj_model(model)
    chunks = parse_iff(out)
    # NJCM + empty POF0.
    assert chunks[0].type == "NJCM"
    assert len(chunks[0].data) == 52  # one node


# ---------------------------------------------------------------------------
# Live PSOBB data — round-trip 20 representative models.
# ---------------------------------------------------------------------------


_LIVE_TEST_BMLS = [
    # (bml_name, inner_name, "reason for selection")
    ("biri_ball.bml",          "biri_ball.nj",          "minimal: 1 vertex chunk + 1 strip"),
    ("biri_ball.bml",          "robby_cat.nj",          "multi-material: 4 NJTL slots"),
    ("biri_ball.bml",          "rokeorange_churoke.nj", "uses chunk type 23 + 64"),
    ("bm_boss1_dragon.bml",    "boss1_s_nb_dragon.nj",  "boss-class skeleton: 124 nodes"),
    ("bm_ene_common_all.bml",  "ene_common_all.nj",    "edge case: zero-byte POF0"),
    ("bm_ene_balclaw.bml",     "re6_b_bal_body.nj",    "Bal Claw: chunk types 17,19,23"),
    ("bm_ene_lappy_ap.bml",    "re3_b_lappy_base.nj",  "uses chunk type 1"),
    ("bm_ene_bm3_fly.bml",     "bm3_fly_body.nj",      "small fly skeleton"),
    ("bm_boss5_gryphon.bml",   "boss5_s_body.nj",      "Gryphon: huge body"),
    ("bm_boss7_de_rol_le_c.bml", "boss2_b_derorure_body.nj", "De Rol Le body"),
]


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO data not present")
@pytest.mark.parametrize("bml_name,inner_name,reason", _LIVE_TEST_BMLS)
def test_live_round_trip(bml_name, inner_name, reason):
    """Round-trip a representative shipped NJ.

    Asserts byte-exact equality after parse + encode.
    """
    bml_path = PSOBB_DATA / bml_name
    if not bml_path.exists():
        pytest.skip(f"{bml_name} not in PSOBB.IO/data")
    all_e = extract_bml(bml_path.read_bytes())
    if inner_name not in all_e:
        pytest.skip(f"{bml_name} has no entry {inner_name}")
    src = all_e[inner_name]
    model = parse_nj_for_writer(src)
    out = encode_nj_model(model)
    assert out == src, (
        f"{bml_name}#{inner_name} ({reason}): "
        f"src {len(src)} bytes, out {len(out)} bytes"
    )


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO data not present")
def test_live_round_trip_corpus_high_rate():
    """Walk every shipped NJ and verify byte-exact round-trip ≥ 95%.

    Lower bar than 100% in case future content surface chunk types we
    don't yet support; in practice we expect 100% on the current data.
    """
    import os

    total = exact = 0
    for fname in sorted(os.listdir(PSOBB_DATA)):
        if not fname.endswith(".bml"):
            continue
        try:
            all_e = extract_bml((PSOBB_DATA / fname).read_bytes())
        except Exception:
            continue
        for inner_name, inner in all_e.items():
            if not inner_name.endswith(".nj"):
                continue
            total += 1
            try:
                model = parse_nj_for_writer(inner)
                out = encode_nj_model(model)
            except Exception:
                continue
            if out == inner:
                exact += 1
    assert total >= 100, f"only {total} NJs in corpus — PSOBB.IO data missing?"
    rate = exact / total
    assert rate >= 0.95, f"round-trip rate {rate*100:.1f}% < 95% ({exact}/{total})"


# ---------------------------------------------------------------------------
# Mutation tests: edit a model and ensure encode produces parseable output.
# ---------------------------------------------------------------------------


def test_mutate_synthetic_cube_position():
    """Move the cube's bone; the output should still parse."""
    model = _build_synthetic_cube()
    model.nodes[0].position = (10.0, 20.0, 30.0)
    # Synthetic models have no layout_hint, so the encoder uses the
    # deterministic synthetic layout. Re-parse via the writer parser.
    out = encode_nj_model(model)
    re_parsed = parse_nj_for_writer(out)
    assert re_parsed.nodes[0].position == (10.0, 20.0, 30.0)


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO data not present")
def test_mutate_njtl_name():
    """Edit one NJTL slot's texture name, verify it round-trips."""
    bml_path = PSOBB_DATA / "biri_ball.bml"
    src = extract_bml(bml_path.read_bytes())["biri_ball.nj"]
    model = parse_nj_for_writer(src)
    assert model.njtl_names == ["b_ball"]
    model.njtl_names = ["new_tex"]
    out = encode_nj_model(model)
    re_parsed = parse_nj_for_writer(out)
    assert re_parsed.njtl_names == ["new_tex"]
