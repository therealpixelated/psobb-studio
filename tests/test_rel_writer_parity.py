"""Byte-exact round-trip parity tests for ``formats.rel_writer`` (c.rel).

Mirrors ``tests/test_nj_writer.py``: PSOBB-data-guarded live fixtures,
``assert out == src`` byte-exact equality, and a corpus sweep with a
>=99% bar.

Build order (each step independently tested):

  STEP 0  the u16 word-delta pointer-table codec (no PSOBB data needed).
  STEP 1  the container framing + 32-byte alignment invariants.
  STEP 2  parse -> encode byte-exact for the worked fixture, a small
          parametrised set, the full corpus sweep, and a
          parse->encode->parse stability cross-check vs formats.rel.

Run isolated::

    python -m pytest tests/test_rel_writer_parity.py -q
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest

from formats.rel import is_c_rel, is_n_rel, parse_rel
from formats.rel_writer import (
    CrelFace,
    CrelModel,
    CrelNode,
    RelWriteError,
    assemble_rel,
    build_crel,
    decode_rel_pointer_table,
    encode_crel,
    encode_nrel,
    encode_rel_pointer_table,
    parse_crel_for_writer,
    parse_nrel_for_writer,
    simulate_rel_relocation,
)


SCENE_DIR = Path(os.path.expanduser("~/PSOBB.IO/data/scene"))
HAS_PSOBB = SCENE_DIR.is_dir()

# Worked fixtures with their proven intermediate offsets (from manual
# byte inspection of map_lobby_01c.rel and friends).
_FIXTURES = [
    # (filename, node_count, pointer_count, first_mesh_off,
    #  node_arr_off, payload_off)
    ("map_lobby_01c.rel", 10, 31, 0x48F0, 0x93DC, 0x94E4),
    ("map_aboss01c.rel", 1, 4, None, None, None),
    ("map_city00_00c.rel", 13, 40, None, None, None),
    ("map_aancient01_00c.rel", 320, 961, None, None, None),
]


# ===========================================================================
# STEP 0 — pointer-table codec (no PSOBB data)
# ===========================================================================

def test_ptr_table_empty():
    assert encode_rel_pointer_table([]) == b""
    assert decode_rel_pointer_table(b"") == []


def test_ptr_table_lobby_style_deltas():
    """The proven lobby_01-style head: first ptr at byte 0xa8 (=42 words),
    then a run of +4-word (0x10-byte) deltas.

    Verifies the offsets [0xa8, 0xb8, 0xc8, 0xd8] encode to u16 deltas
    [42, 4, 4, 4] and decode back exactly.
    """
    offsets = [0xA8, 0xB8, 0xC8, 0xD8]
    table = encode_rel_pointer_table(offsets)
    assert table == struct.pack("<4H", 42, 4, 4, 4)
    assert decode_rel_pointer_table(table) == offsets


def test_ptr_table_round_trip_random_aligned():
    offsets = [0, 4, 0x100, 0x104, 0x3FFFC, 0x7FFF8]
    table = encode_rel_pointer_table(offsets)
    assert decode_rel_pointer_table(table) == offsets
    assert len(table) == len(offsets) * 2


def test_ptr_table_zero_value_field_still_flagged():
    """A pointer at byte offset 0 (lobby node-0 verts_ptr) is a legal,
    representable entry: delta 0 from base 0."""
    assert encode_rel_pointer_table([0]) == struct.pack("<H", 0)
    assert decode_rel_pointer_table(struct.pack("<H", 0)) == [0]


def test_ptr_table_rejects_unaligned():
    with pytest.raises(RelWriteError, match="4-byte aligned"):
        encode_rel_pointer_table([6])


def test_ptr_table_rejects_descending():
    with pytest.raises(RelWriteError, match="ascending"):
        encode_rel_pointer_table([8, 4])


def test_ptr_table_rejects_gap_over_ceiling():
    # 0x40000 bytes = 0x10000 words > 0xFFFF ceiling.
    with pytest.raises(RelWriteError, match="ceiling"):
        encode_rel_pointer_table([0, 0x40000])


def test_ptr_table_max_gap_ok():
    # Exactly 0xFFFF words is allowed.
    offsets = [0, 0xFFFF * 4]
    assert decode_rel_pointer_table(encode_rel_pointer_table(offsets)) == offsets


def test_decode_with_base():
    table = encode_rel_pointer_table([0x10, 0x20])
    assert decode_rel_pointer_table(table, base=0x1000) == [0x1010, 0x1020]


# ===========================================================================
# STEP 1 — container framing + alignment invariants
# ===========================================================================

def test_assemble_alignment_invariants_tiny():
    """Hand-built data: pad-to-32, table, pad, trailer; all the %32
    invariants hold and the trailer decodes to what we put in."""
    # A 4-byte data section with a single pointer at offset 0 and the
    # payload head also at offset 0.
    data = struct.pack("<I", 0)
    out = assemble_rel(data, ptr_offsets=[0], payload_offset=0)
    assert len(out) % 32 == 0
    trailer_start = len(out) - 0x20
    assert trailer_start % 32 == 0
    pt_off, pt_count, flag, reserved, pl_off = struct.unpack_from(
        "<5I", out, trailer_start)
    assert pt_off % 32 == 0
    assert flag == 1 and reserved == 0
    assert pt_count == 1 and pl_off == 0
    assert out[trailer_start + 0x14:trailer_start + 0x20] == b"\x00" * 12
    # data section padded to 32 -> pt_off == 0x20.
    assert pt_off == 0x20
    assert decode_rel_pointer_table(out, pt_off, pt_count) == [0]


def test_assemble_padding_to_make_trailer_aligned():
    """A data+table length that is NOT already 32-aligned must get NUL
    padding before the trailer."""
    # 0x20-byte data, 3 pointers => 6-byte table => 0x26 so-far => pad to
    # 0x40, trailer at 0x40, file 0x60.
    data = b"\x00" * 0x20
    out = assemble_rel(data, ptr_offsets=[0, 4, 8], payload_offset=0)
    assert len(out) == 0x60
    assert (len(out) - 0x20) % 32 == 0


def test_assemble_rejects_pointer_out_of_data():
    with pytest.raises(RelWriteError, match="out of data section"):
        assemble_rel(b"\x00" * 8, ptr_offsets=[8], payload_offset=0)


def test_assemble_rejects_payload_out_of_data():
    with pytest.raises(RelWriteError, match="payload_offset"):
        assemble_rel(b"\x00" * 8, ptr_offsets=[0], payload_offset=8)


# ===========================================================================
# STEP 2 — c.rel parse/encode
# ===========================================================================

def test_synthetic_single_quad_floor_loadable():
    """A hand-built 2-triangle floor (no PSOBB data) is a valid c.rel:
    re-parses, classifies as c.rel, and survives the engine relocation
    simulation."""
    verts = [(-100.0, 0.0, -100.0), (100.0, 0.0, -100.0),
             (100.0, 0.0, 100.0), (-100.0, 0.0, 100.0)]
    faces = [CrelFace(0, 1, 2), CrelFace(0, 2, 3)]
    node = CrelNode(verts=verts, faces=faces)  # sphere auto-derived
    out = build_crel([node])

    rel = parse_rel(out)
    assert is_c_rel(rel)
    # pointer_count == 3*nodes + 1
    assert rel.pointer_count == 3 * 1 + 1
    # relocation simulation: every flagged word lands in-data or is null.
    base = 0x20000000
    data_end = rel.pointer_table_offset
    for v in simulate_rel_relocation(out, base):
        assert v == base or base <= v < base + data_end

    # And it round-trips through our own parser.
    model = parse_crel_for_writer(out)
    assert len(model.nodes) == 1
    assert len(model.nodes[0].faces) == 2
    assert encode_crel(model) == out


def test_synthetic_face_geometry_derivation():
    """Auto-derived normal/centroid/radius match hand-computed values."""
    verts = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 2.0)]
    out = build_crel([CrelNode(verts=verts, faces=[CrelFace(0, 1, 2)])])
    model = parse_crel_for_writer(out)
    fc = model.nodes[0].faces[0]
    # centroid is the mean of the 3 verts.
    assert fc.centroid == pytest.approx((2 / 3, 0.0, 2 / 3))
    # normal is unit length.
    nx, ny, nz = fc.normal
    assert (nx * nx + ny * ny + nz * nz) ** 0.5 == pytest.approx(1.0, abs=1e-5)


def test_build_crel_budget_enforced():
    """A hull over 64 KB is rejected by build_crel (budget gate)."""
    # ~2000 independent triangles => ~92 KB, over the 64 KB cap.
    verts = []
    faces = []
    for i in range(2000):
        b = len(verts)
        verts += [(float(i), 0.0, 0.0), (float(i) + 1, 0.0, 0.0),
                  (float(i), 0.0, 1.0)]
        faces.append(CrelFace(b, b + 1, b + 2))
    with pytest.raises(RelWriteError, match="budget"):
        build_crel([CrelNode(verts=verts, faces=faces)])


# ---- live byte-exact fixtures --------------------------------------------

@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO data not present")
def test_lobby_01c_byte_exact_with_proven_offsets():
    """The canonical worked fixture: byte-exact AND the proven
    intermediate offsets are reproduced."""
    src = (SCENE_DIR / "map_lobby_01c.rel").read_bytes()
    model = parse_crel_for_writer(src)
    assert len(model.nodes) == 10

    out = encode_crel(model)
    assert out == src, f"lobby_01c not byte-exact: {len(out)} vs {len(src)}"

    rel = parse_rel(out)
    assert rel.pointer_count == 31
    assert rel.payload_offset == 0x94E4
    # node-0 mesh at 0x48f0; its verts live at file offset 0 (verts_ptr==0
    # yet still a registered pointer — the load-bearing edge case).
    head = struct.unpack_from("<I", out, rel.payload_offset)[0]
    assert head == 0x93DC
    mesh0_ptr = struct.unpack_from("<I", out, head)[0]
    assert mesh0_ptr == 0x48F0
    vcount, verts_ptr, fcount, faces_ptr = struct.unpack_from(
        "<IIiI", out, mesh0_ptr)
    assert verts_ptr == 0  # node-0 verts at offset 0
    assert (mesh0_ptr + 4) in rel.pointer_offsets  # still flagged


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO data not present")
@pytest.mark.parametrize(
    "fname,node_count,ptr_count,first_mesh,node_arr,payload", _FIXTURES)
def test_fixture_byte_exact(fname, node_count, ptr_count, first_mesh,
                            node_arr, payload):
    path = SCENE_DIR / fname
    if not path.exists():
        pytest.skip(f"{fname} not present")
    src = path.read_bytes()
    model = parse_crel_for_writer(src)
    assert len(model.nodes) == node_count

    out = encode_crel(model)
    assert out == src, f"{fname} not byte-exact ({len(out)} vs {len(src)})"

    rel = parse_rel(out)
    assert rel.pointer_count == ptr_count
    assert rel.pointer_count == 3 * node_count + 1
    if payload is not None:
        assert rel.payload_offset == payload
        head = struct.unpack_from("<I", out, rel.payload_offset)[0]
        assert head == node_arr
        first_mesh_ptr = struct.unpack_from("<I", out, head)[0]
        assert first_mesh_ptr == first_mesh


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO data not present")
@pytest.mark.parametrize("fname,_n,_p,_f,_na,_pl", _FIXTURES)
def test_fixture_parse_encode_parse_stability(fname, _n, _p, _f, _na, _pl):
    """parse -> encode -> parse: the relocation graph, payload offset, and
    pointer count survive, cross-checked against the trusted reader."""
    path = SCENE_DIR / fname
    if not path.exists():
        pytest.skip(f"{fname} not present")
    src = path.read_bytes()
    src_rel = parse_rel(src)
    assert src_rel.pointer_offsets, "source has no pointers (bad fixture)"

    out = encode_crel(parse_crel_for_writer(src))
    out_rel = parse_rel(out)

    assert set(out_rel.pointer_offsets) == set(src_rel.pointer_offsets)
    assert out_rel.payload_offset == src_rel.payload_offset
    assert out_rel.pointer_count == src_rel.pointer_count
    assert out_rel.pointer_table_offset == src_rel.pointer_table_offset

    # Engine relocation simulation lands every flagged word in-data/null.
    base = 0x30000000
    data_end = out_rel.pointer_table_offset
    for v in simulate_rel_relocation(out, base):
        assert v == base or base <= v < base + data_end


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO data not present")
def test_corpus_sweep_byte_exact_rate():
    """Every ``*c.rel`` in scene/ must round-trip byte-exact (>=99%).

    Files that don't classify as c.rel (none expected) are skipped, not
    counted against the rate.  Any non-exact file is reported with its
    first differing offset.
    """
    files = sorted(SCENE_DIR.glob("*c.rel"))
    assert len(files) >= 100, f"only {len(files)} *c.rel — data missing?"

    total = exact = skipped = 0
    failures: list[str] = []
    for path in files:
        src = path.read_bytes()
        try:
            rel = parse_rel(src)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{path.name}: parse_rel failed: {e}")
            total += 1
            continue
        if not is_c_rel(rel):
            skipped += 1
            continue
        total += 1
        try:
            out = encode_crel(parse_crel_for_writer(src))
        except Exception as e:  # noqa: BLE001
            failures.append(f"{path.name}: encode failed: {e}")
            continue
        if out == src:
            exact += 1
        else:
            first = next((i for i in range(min(len(out), len(src)))
                          if out[i] != src[i]), None)
            failures.append(
                f"{path.name}: first diff @ 0x{first:x}" if first is not None
                else f"{path.name}: length {len(out)} vs {len(src)}")

    rate = exact / total if total else 0.0
    detail = "; ".join(failures[:20])
    assert rate >= 0.99, (
        f"c.rel byte-exact rate {rate * 100:.1f}% ({exact}/{total}, "
        f"{skipped} non-crel skipped) < 99%. Failures: {detail}")


# ---- mutation (semantic, not byte-exact) ---------------------------------

@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO data not present")
def test_mutate_node_flag_round_trips():
    """Edit a node flag, re-encode, re-parse: the change reads back and
    the pointer graph is unchanged."""
    src = (SCENE_DIR / "map_aboss01c.rel").read_bytes()
    model = parse_crel_for_writer(src)
    src_locs = set(parse_rel(src).pointer_offsets)

    model.nodes[0].flags = 0x80000120
    out = encode_crel(model)
    re = parse_crel_for_writer(out)
    assert re.nodes[0].flags == 0x80000120
    assert set(parse_rel(out).pointer_offsets) == src_locs


# ===========================================================================
# STEP 3 — n.rel (NrelFmt2 node geometry) parse/encode
# ===========================================================================

# Worked n.rel fixtures with their proven trailer facts (from byte
# inspection of the real files in PSOBB.IO/data/scene).
_NREL_FIXTURES = [
    # (filename, pointer_count, payload_off, chunk_count, tex_count)
    ("map_lobby_01n.rel", 945, 0x52C, 1, 93),
    ("map_lobby_02n.rel", None, None, None, None),
    ("map_lobby_03n.rel", None, None, None, None),
    ("map_lobby_04n.rel", None, None, None, None),
    ("map_lobby_05n.rel", None, None, None, None),
    ("map_lobby_06n.rel", None, None, None, None),
    ("map_lobby_07n.rel", None, None, None, None),
    ("map_lobby_08n.rel", None, None, None, None),
    ("map_lobby_09n.rel", None, None, None, None),
    ("map_lobby_10n.rel", None, None, None, None),
    ("map_acave01_00n.rel", 2068, 0x1A38, 59, 94),
    ("map_aboss01n.rel", 173, 0x1D8, 2, 13),
]


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO data not present")
def test_nrel_lobby01_byte_exact():
    """The canonical n.rel worked fixture: byte-exact AND the proven
    trailer/structure facts are reproduced.

    Dumps the first differing offset on failure (the single most useful
    datum when a relocation entry or a header scalar drifts).
    """
    src = (SCENE_DIR / "map_lobby_01n.rel").read_bytes()
    model = parse_nrel_for_writer(src)
    # Verified facts from the spec / byte inspection.
    assert model.chunk_count == 1
    assert len(model.chunks) == 1
    assert len(model.texture_names) == 93
    assert model.unk1 == 64
    assert model.radius == pytest.approx(800.0)
    # first chunk references the 61-entry static tree array @0xa8 and a
    # 4-entry animated tree array @0x478.
    chunk = model.chunks[0]
    assert chunk.static_count == 61
    assert chunk.animated_count == 4
    assert chunk.static_mesh_trees_ptr == 0xA8
    assert chunk.animated_mesh_trees_ptr == 0x478

    out = encode_nrel(model)
    if out != src:
        first = next((i for i in range(min(len(out), len(src)))
                      if out[i] != src[i]), None)
        raise AssertionError(
            f"lobby_01n not byte-exact: first diff @ 0x{first:x}"
            if first is not None
            else f"lobby_01n length {len(out)} vs {len(src)}")

    rel = parse_rel(out)
    assert rel.pointer_count == 945
    assert rel.payload_offset == 0x52C
    # first reloc pointer is at 0xa8 (delta 42 words from base).
    assert rel.pointer_offsets[0] == 0xA8


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO data not present")
@pytest.mark.parametrize("fname,ptr_count,payload,chunk_count,tex_count",
                         _NREL_FIXTURES)
def test_nrel_fixture_byte_exact(fname, ptr_count, payload, chunk_count,
                                 tex_count):
    """lobby_01..10n + acave01_00n + aboss01n all round-trip byte-exact."""
    path = SCENE_DIR / fname
    if not path.exists():
        pytest.skip(f"{fname} not present")
    src = path.read_bytes()
    rel_in = parse_rel(src)
    assert is_n_rel(rel_in), f"{fname} is not an n.rel"

    model = parse_nrel_for_writer(src)
    out = encode_nrel(model)
    if out != src:
        first = next((i for i in range(min(len(out), len(src)))
                      if out[i] != src[i]), None)
        raise AssertionError(
            f"{fname} not byte-exact: first diff @ 0x{first:x}"
            if first is not None
            else f"{fname} length {len(out)} vs {len(src)}")

    rel = parse_rel(out)
    if ptr_count is not None:
        assert rel.pointer_count == ptr_count
    if payload is not None:
        assert rel.payload_offset == payload
    if chunk_count is not None:
        assert model.chunk_count == chunk_count
    if tex_count is not None:
        assert len(model.texture_names) == tex_count


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO data not present")
@pytest.mark.parametrize("fname,_pc,_pl,_cc,_tc", _NREL_FIXTURES)
def test_nrel_parse_encode_parse_stability(fname, _pc, _pl, _cc, _tc):
    """parse -> encode -> parse: the relocation graph, payload offset,
    pointer count, and table offset survive, cross-checked against the
    trusted reader formats.rel.parse_rel."""
    path = SCENE_DIR / fname
    if not path.exists():
        pytest.skip(f"{fname} not present")
    src = path.read_bytes()
    src_rel = parse_rel(src)
    assert src_rel.pointer_offsets, "source has no pointers (bad fixture)"

    out = encode_nrel(parse_nrel_for_writer(src))
    out_rel = parse_rel(out)

    assert out_rel.pointer_offsets == src_rel.pointer_offsets
    assert out_rel.payload_offset == src_rel.payload_offset
    assert out_rel.pointer_count == src_rel.pointer_count
    assert out_rel.pointer_table_offset == src_rel.pointer_table_offset
    # The relocatable data section is identical too.
    assert out[:out_rel.pointer_table_offset] == \
        src[:src_rel.pointer_table_offset]

    # Engine relocation simulation lands every flagged word in-data/null.
    base = 0x40000000
    data_end = out_rel.pointer_table_offset
    for v in simulate_rel_relocation(out, base):
        assert v == base or base <= v < base + data_end


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO data not present")
def test_nrel_corpus_sweep_byte_exact_rate():
    """Every ``*n.rel`` that classifies as n.rel must round-trip
    byte-exact (>=99%).

    Files that don't classify as n.rel are skipped, not counted against
    the rate.  Any non-exact file is reported with its first differing
    offset.
    """
    files = sorted(SCENE_DIR.glob("*n.rel"))
    assert len(files) >= 100, f"only {len(files)} *n.rel — data missing?"

    total = exact = skipped = 0
    failures: list[str] = []
    for path in files:
        src = path.read_bytes()
        try:
            rel = parse_rel(src)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{path.name}: parse_rel failed: {e}")
            total += 1
            continue
        if not is_n_rel(rel):
            skipped += 1
            continue
        total += 1
        try:
            out = encode_nrel(parse_nrel_for_writer(src))
        except Exception as e:  # noqa: BLE001
            failures.append(f"{path.name}: encode failed: {e}")
            continue
        if out == src:
            exact += 1
        else:
            first = next((i for i in range(min(len(out), len(src)))
                          if out[i] != src[i]), None)
            failures.append(
                f"{path.name}: first diff @ 0x{first:x}" if first is not None
                else f"{path.name}: length {len(out)} vs {len(src)}")

    rate = exact / total if total else 0.0
    detail = "; ".join(failures[:20])
    assert rate >= 0.99, (
        f"n.rel byte-exact rate {rate * 100:.1f}% ({exact}/{total}, "
        f"{skipped} non-nrel skipped) < 99%. Failures: {detail}")


# ---- mutation (semantic, not byte-exact) ---------------------------------

@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO data not present")
def test_nrel_mutate_radius_round_trips():
    """Edit the fmt2 header radius scalar, re-encode, re-parse: the change
    reads back and the relocation graph is unchanged."""
    src = (SCENE_DIR / "map_aboss01n.rel").read_bytes()
    model = parse_nrel_for_writer(src)
    src_locs = list(parse_rel(src).pointer_offsets)

    model.radius = 1234.0
    out = encode_nrel(model)
    re = parse_nrel_for_writer(out)
    assert re.radius == pytest.approx(1234.0)
    assert list(parse_rel(out).pointer_offsets) == src_locs
    # Scalar edit must not move the data section size or payload offset.
    assert parse_rel(out).payload_offset == parse_rel(src).payload_offset
