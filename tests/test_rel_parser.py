"""Tests for ``formats.rel`` — the PSOBB ``.rel`` relocation-table parser.

Covers:
  - Trailer decoding (32-byte tail at end of file).
  - Pointer-table walk (u16 deltas multiplied by 4).
  - Sniffer functions (is_n_rel / is_c_rel / is_r_rel) on synthetic
    + real PSOBB.IO map files (when available).
  - n.rel payload reader (NrelFmt2 + Chunk + MeshTree → mesh-tree-node
    offsets that the existing XJ walker consumes).
  - Robustness against malformed buffers (truncation / OOB pointers /
    bogus counts).

Synthetic tests build a minimal valid REL by hand so we don't need
real game data.  The "real-file" tests gate on the live PSOBB.IO data
dir; they degrade to skips when the install isn't available so the
test suite still runs in CI.
"""
from __future__ import annotations

import struct
from pathlib import Path

import pytest

from formats import rel as rel_mod


PSOBB_DATA = Path(r"~\PSOBB.IO\data\scene")


# ---------------------------------------------------------------------------
# Synthetic builders
# ---------------------------------------------------------------------------
def _build_minimal_rel(payload: bytes,
                       payload_offset: int,
                       pointer_offsets: list[int]) -> bytes:
    """Construct a minimal valid REL buffer.

    Layout::

        [data: payload bytes ...padding...]
        [pointer_table: u16 deltas]
        [trailer: 32 bytes]

    The pointer table encodes ``pointer_offsets`` as deltas /4 from the
    previous absolute offset.  The first entry is delta-from-zero.
    """
    data = bytearray(payload)
    # Pad data to 4-byte alignment (deltas must be multiples of 4).
    while len(data) % 4 != 0:
        data.append(0)
    pt_off = len(data)
    # Build the pointer table
    prev = 0
    for abs_off in pointer_offsets:
        if abs_off % 4 != 0:
            raise ValueError(f"pointer offset {abs_off} not 4-aligned")
        delta = (abs_off - prev) // 4
        if delta < 0 or delta > 0xFFFF:
            raise ValueError(f"delta {delta} out of u16 range")
        data.extend(struct.pack("<H", delta))
        prev = abs_off
    # Pad pointer table so trailer starts at 4-byte boundary
    while len(data) % 4 != 0:
        data.append(0)
    trailer = struct.pack(
        "<8I",
        pt_off,
        len(pointer_offsets),
        1,
        0,
        payload_offset,
        0, 0, 0,
    )
    data.extend(trailer)
    return bytes(data)


# ---------------------------------------------------------------------------
# Trailer + pointer-table tests
# ---------------------------------------------------------------------------
def test_trailer_decode_minimal():
    """Round-trip: build a 1-pointer REL and parse the trailer."""
    payload = b"AAAA" + struct.pack("<I", 0x10) + b"\x00" * 8 + b"BBBB"
    # payload[4:8] is the only u32 pointer field — it points to offset 0x10
    # which is the "BBBB" word.
    buf = _build_minimal_rel(payload, payload_offset=0, pointer_offsets=[4])
    rel = rel_mod.parse_rel(buf)
    assert rel.payload_offset == 0
    assert rel.pointer_count == 1
    assert rel.pointer_offsets == [4]
    assert rel.is_pointer(4)
    assert not rel.is_pointer(0)
    # Reading the pointer field yields the target offset
    assert rel.read_u32(4) == 0x10


def test_pointer_table_multiple_entries():
    """Verify delta-encoding works across multiple pointers."""
    # 4 pointer slots at 0x00, 0x10, 0x20, 0x30 → deltas: 0, 4, 4, 4
    payload = b"\x00" * 0x40
    buf = _build_minimal_rel(payload, payload_offset=0,
                             pointer_offsets=[0x00, 0x10, 0x20, 0x30])
    rel = rel_mod.parse_rel(buf)
    assert rel.pointer_offsets == [0x00, 0x10, 0x20, 0x30]


def test_pointer_table_large_deltas():
    """Deltas can be up to 0xFFFF*4 = 0x3FFFC bytes."""
    payload = b"\x00" * 0x10000
    # First pointer at 0x4000, second at 0x4000 + 0x4000 = 0x8000
    buf = _build_minimal_rel(payload, payload_offset=0,
                             pointer_offsets=[0x4000, 0x8000])
    rel = rel_mod.parse_rel(buf)
    assert rel.pointer_offsets == [0x4000, 0x8000]


def test_parse_rel_rejects_truncated_buffer():
    with pytest.raises(rel_mod.RelParseError):
        rel_mod.parse_rel(b"\x00" * 10)


def test_parse_rel_rejects_bogus_pointer_table_offset():
    """Trailer says pointer table is past file end → reject."""
    payload = b"\x00" * 0x40
    buf = bytearray(payload)
    # Construct a trailer where pointer_table_offset = 0xFFFF (way past end)
    trailer = struct.pack(
        "<8I",
        0xFFFFFFF, 0, 1, 0, 0, 0, 0, 0,
    )
    buf.extend(trailer)
    with pytest.raises(rel_mod.RelParseError):
        rel_mod.parse_rel(bytes(buf))


def test_parse_rel_rejects_bogus_payload_offset():
    payload = b"\x00" * 0x40
    buf = bytearray(payload)
    pt_off = len(buf)
    buf.extend(b"\x00" * 4)  # one u16 entry padded
    trailer = struct.pack(
        "<8I",
        pt_off, 0, 1, 0,
        0xFFFFFFF,  # bogus payload offset
        0, 0, 0,
    )
    buf.extend(trailer)
    with pytest.raises(rel_mod.RelParseError):
        rel_mod.parse_rel(bytes(buf))


def test_parse_rel_rejects_pointer_table_overflow():
    """Trailer claims more pointer entries than the table can hold."""
    payload = b"\x00" * 0x40
    buf = bytearray(payload)
    pt_off = len(buf)
    buf.extend(b"\x00" * 4)  # only 4 bytes = 2 entries
    trailer = struct.pack(
        "<8I",
        pt_off, 100,  # claims 100 entries -> overflow
        1, 0, 0, 0, 0, 0,
    )
    buf.extend(trailer)
    with pytest.raises(rel_mod.RelParseError):
        rel_mod.parse_rel(bytes(buf))


def test_is_rel_sniffer_negative():
    """Random / undersized buffers should be rejected."""
    assert not rel_mod.is_rel(b"")
    assert not rel_mod.is_rel(b"\x00" * 10)
    # Right-sized but mostly garbage trailer
    assert not rel_mod.is_rel(b"\xFF" * 100)


def test_is_rel_sniffer_positive():
    payload = b"\x00" * 0x10
    buf = _build_minimal_rel(payload, payload_offset=0, pointer_offsets=[])
    assert rel_mod.is_rel(buf)


def test_relfile_read_helpers_bounds_check():
    payload = b"\x00" * 0x10
    buf = _build_minimal_rel(payload, payload_offset=0, pointer_offsets=[])
    rel = rel_mod.parse_rel(buf)
    # OOB read raises
    with pytest.raises(rel_mod.RelParseError):
        rel.read_u32(0xFFFF)
    with pytest.raises(rel_mod.RelParseError):
        rel.read_u16(0xFFFF)
    with pytest.raises(rel_mod.RelParseError):
        rel.read_f32(0xFFFF)


def test_relfile_deref_null_returns_zero():
    """A NULL pointer field (value=0) is permitted; deref returns 0."""
    payload = b"\x00" * 0x10
    buf = _build_minimal_rel(payload, payload_offset=0, pointer_offsets=[0])
    rel = rel_mod.parse_rel(buf)
    assert rel.deref(0) == 0


def test_relfile_deref_oob_raises():
    """Pointer value past data section is rejected."""
    payload = bytearray(b"\x00" * 0x10)
    # Place a u32 = 0xFFFF at offset 0
    struct.pack_into("<I", payload, 0, 0xFFFFFF)
    buf = _build_minimal_rel(bytes(payload), payload_offset=0,
                             pointer_offsets=[0])
    rel = rel_mod.parse_rel(buf)
    with pytest.raises(rel_mod.RelParseError):
        rel.deref(0)


# ---------------------------------------------------------------------------
# n.rel payload reader tests (synthetic NrelFmt2)
# ---------------------------------------------------------------------------
def _build_minimal_nrel() -> bytes:
    """Build an n.rel with one chunk + one mesh tree pointing at a fake node.

    Chunk is at offset 0x40 in the data section.  MeshTree is at 0x80.
    NrelFmt2 header is at offset 0 (= payload_offset).
    The mesh tree's root_node_ptr is set to 0xC0 (a synthetic offset
    inside data — not actually a valid Ninja node, but we only test
    that the walker reaches it).
    """
    data_size = 0x100
    data = bytearray(b"\x00" * data_size)
    # NrelFmt2 at offset 0:
    #   +0: 'fmt2'
    #   +4: unk1
    #   +8: chunk_count=1, +0xa: unk2
    #   +0xc: radius
    #   +0x10: chunks_ptr = 0x40
    #   +0x14: texture_data_ptr = 0
    struct.pack_into("<4sIHHfII", data, 0, b"fmt2", 0, 1, 0, 0.0, 0x40, 0)
    # Chunk at 0x40 (52 bytes):
    #   id=42, x=10, y=20, z=30, rot 0/0/0, radius=100,
    #   static_mesh_trees_ptr=0x80, animated=0,
    #   static_count=1, animated=0, flags=0
    struct.pack_into("<i3f3i fI I I I I", data, 0x40,
                     42, 10.0, 20.0, 30.0, 0, 0, 0, 100.0,
                     0x80, 0, 1, 0, 0)
    # MeshTree at 0x80 (16 bytes):
    #   root_node_ptr=0xC0, unk1=0, anim_info=0, tree_flags=0
    struct.pack_into("<IIII", data, 0x80, 0xC0, 0, 0, 0)
    # Pointers (the chunks_ptr field at 0x10, static_mesh_trees_ptr at
    # 0x60, root_node_ptr at 0x80) — sorted ascending.
    pointer_offsets = [0x10, 0x60, 0x80]
    return _build_minimal_rel(bytes(data), payload_offset=0,
                              pointer_offsets=pointer_offsets)


def test_synthetic_nrel_round_trip():
    buf = _build_minimal_nrel()
    rel = rel_mod.parse_rel(buf)
    assert rel_mod.is_n_rel(rel)
    assert not rel_mod.is_c_rel(rel)
    h = rel_mod.read_nrel_header(rel)
    assert h.chunk_count == 1
    assert h.chunks_ptr == 0x40
    chunks = rel_mod.read_nrel_chunks(rel, h)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.id == 42
    assert c.x == 10.0 and c.y == 20.0 and c.z == 30.0
    assert c.radius == 100.0
    assert c.static_mesh_tree_count == 1
    trees = rel_mod.read_mesh_trees(rel, c.static_mesh_trees_ptr,
                                    c.static_mesh_tree_count)
    assert len(trees) == 1
    assert trees[0].root_node_ptr == 0xC0
    offsets = rel_mod.extract_nrel_mesh_root_offsets(rel)
    assert offsets == [0xC0]


def test_read_nrel_header_rejects_non_nrel():
    """A REL whose payload doesn't start with 'fmt2' isn't an n.rel."""
    payload = b"junk" + b"\x00" * 0x40
    buf = _build_minimal_rel(payload, payload_offset=0, pointer_offsets=[])
    rel = rel_mod.parse_rel(buf)
    assert not rel_mod.is_n_rel(rel)
    with pytest.raises(rel_mod.RelParseError):
        rel_mod.read_nrel_header(rel)


def test_read_nrel_chunks_handles_zero_count():
    """An n.rel with zero chunks is valid (boss arenas, etc.)."""
    data = bytearray(b"\x00" * 0x40)
    struct.pack_into("<4sIHHfII", data, 0, b"fmt2", 0, 0, 0, 0.0, 0, 0)
    buf = _build_minimal_rel(bytes(data), payload_offset=0,
                             pointer_offsets=[])
    rel = rel_mod.parse_rel(buf)
    h = rel_mod.read_nrel_header(rel)
    assert h.chunk_count == 0
    assert rel_mod.read_nrel_chunks(rel, h) == []
    assert rel_mod.extract_nrel_mesh_root_offsets(rel) == []


# ---------------------------------------------------------------------------
# Real-file tests (gated on PSOBB.IO data dir)
# ---------------------------------------------------------------------------
def _need_data():
    if not PSOBB_DATA.exists():
        pytest.skip(f"PSOBB data dir not present: {PSOBB_DATA}")


@pytest.fixture
def city00_nrel():
    _need_data()
    p = PSOBB_DATA / "map_city00_00n.rel"
    if not p.exists():
        pytest.skip(f"missing {p}")
    return p.read_bytes()


@pytest.fixture
def aancient01_crel():
    _need_data()
    p = PSOBB_DATA / "map_aancient01_00c.rel"
    if not p.exists():
        pytest.skip(f"missing {p}")
    return p.read_bytes()


def test_real_city00_nrel_parses(city00_nrel):
    rel = rel_mod.parse_rel(city00_nrel)
    assert rel_mod.is_n_rel(rel)
    h = rel_mod.read_nrel_header(rel)
    assert h.chunk_count > 0
    assert h.chunks_ptr != 0
    chunks = rel_mod.read_nrel_chunks(rel, h)
    assert len(chunks) == h.chunk_count
    # Pioneer 2 is laid out around (228, 0, 291) — the first chunk
    # always has reasonable spatial coords.
    c0 = chunks[0]
    assert -10000 < c0.x < 10000
    assert -10000 < c0.z < 10000
    assert c0.radius > 0
    # Plenty of mesh-tree-node offsets get extracted.
    offsets = rel_mod.extract_nrel_mesh_root_offsets(rel)
    assert len(offsets) > 10
    # All offsets are inside the data section.
    for off in offsets:
        assert 0 < off < rel.data_size


def test_real_city00_nrel_texture_names(city00_nrel):
    rel = rel_mod.parse_rel(city00_nrel)
    names = rel_mod.read_texture_names(rel)
    # Pioneer 2 has hundreds of textures
    assert len(names) > 50
    # Names look like "g032_..." / "g064_..." (level texture prefixes)
    assert any(n.startswith("g0") for n in names)


def test_real_aancient01_crel_parses(aancient01_crel):
    rel = rel_mod.parse_rel(aancient01_crel)
    assert rel_mod.is_c_rel(rel)
    assert not rel_mod.is_n_rel(rel)
    # Crel has pointer at payload offset (the nodes head pointer)
    assert rel.payload_offset in rel.pointer_offsets


def test_real_city00_extract_meshes(city00_nrel):
    """End-to-end: city00 n.rel → list[XjMesh] for the renderer."""
    rel = rel_mod.parse_rel(city00_nrel)
    meshes = rel_mod.extract_nrel_meshes(rel)
    # Pioneer 2 has hundreds of submeshes
    assert len(meshes) > 50
    total_v = sum(len(m.vertices) for m in meshes)
    total_t = sum(len(m.indices) // 3 for m in meshes)
    # Plenty of geometry
    assert total_v > 1000
    assert total_t > 1000
    # Vertices look spatially reasonable for Pioneer 2 (rough bounds)
    sample = meshes[0]
    assert len(sample.vertices) > 0
    px, py, pz = sample.vertices[0].pos
    assert -2000 < px < 2000
    assert -2000 < py < 2000
    assert -2000 < pz < 2000


def test_extract_nrel_meshes_rejects_non_nrel():
    """Asking for meshes from a file that isn't an n.rel should raise."""
    payload = b"junk" + b"\x00" * 0x40
    buf = _build_minimal_rel(payload, payload_offset=0, pointer_offsets=[])
    rel = rel_mod.parse_rel(buf)
    with pytest.raises(rel_mod.RelParseError):
        rel_mod.extract_nrel_meshes(rel)


# ---------------------------------------------------------------------------
# r.rel sub_record tests (Editor v4 RE)
# ---------------------------------------------------------------------------
@pytest.fixture
def aancient01_rrel():
    _need_data()
    p = PSOBB_DATA / "map_aancient01_00r.rel"
    if not p.exists():
        pytest.skip(f"missing {p}")
    return p.read_bytes()


def test_real_aancient01_rrel_parses_subrecords(aancient01_rrel):
    """The Forest 1 floor 0 file must produce parseable sub_records."""
    rel = rel_mod.parse_rel(aancient01_rrel)
    assert rel_mod.is_r_rel(rel)
    anchors = rel_mod.read_rrel_anchors(rel)
    assert len(anchors) > 0
    subs = rel_mod.read_rrel_subrecords(rel, anchors)
    # Forest 1 ships every anchor with a sub_record.
    assert len(subs) == len(anchors)
    for s in subs:
        # Every record decodes the common header cleanly.
        assert s.sub_type in rel_mod.RREL_SUBREC_KNOWN_TYPES
        # Position matches the parent anchor (sanity).
        anchor = anchors[s.anchor_index]
        assert s.pos == pytest.approx(anchor.pos, abs=1e-3)
        # Header alone is 0x46 bytes; record_total_bytes is at least that.
        assert s.record_total_bytes >= 0x46
        # Reserved field is zero in our corpus.
        assert s.reserved == 0


def test_subrecord_summary_histogram(aancient01_rrel):
    rel = rel_mod.parse_rel(aancient01_rrel)
    subs = rel_mod.read_rrel_subrecords(rel)
    hist = rel_mod.summarize_subrecord_types(subs)
    # Forest 1 is dominated by 0x14 / 0x16 records.
    assert sum(hist.values()) == len(subs)
    common_keys = set(hist.keys()) & {0x14, 0x16}
    assert common_keys, f"expected 0x14 or 0x16, got {set(hist.keys())}"


def test_subrecord_parser_handles_null_ptr():
    """``parse_subrecord`` returns None on null pointer."""
    payload = b"\x00" * 32
    buf = _build_minimal_rel(payload, payload_offset=0, pointer_offsets=[])
    rel = rel_mod.parse_rel(buf)
    assert rel_mod.parse_subrecord(rel, anchor_index=0, ptr=0) is None


def test_subrecord_parser_handles_oob_ptr():
    """``parse_subrecord`` returns None on out-of-bounds pointer."""
    payload = b"\x00" * 32
    buf = _build_minimal_rel(payload, payload_offset=0, pointer_offsets=[])
    rel = rel_mod.parse_rel(buf)
    # Header is 0x46 bytes — any ptr that pushes past data_size returns None.
    assert rel_mod.parse_subrecord(rel, anchor_index=0,
                                    ptr=rel.data_size - 4) is None


def test_subrecord_known_types_set():
    """The catalogued sub_type set covers the 6 values seen in 156 r.rel."""
    expected = {0x10, 0x12, 0x14, 0x15, 0x16, 0x17}
    assert rel_mod.RREL_SUBREC_KNOWN_TYPES == expected


def test_subrecord_corpus_walk_no_errors():
    """Walk every r.rel in the install — none should crash the parser."""
    if not PSOBB_DATA.exists():
        pytest.skip(f"PSOBB data dir not present: {PSOBB_DATA}")
    files = list(PSOBB_DATA.glob("*r.rel"))
    if not files:
        pytest.skip("no r.rel files in install")
    total = 0
    type_hist: dict[int, int] = {}
    for path in files:
        try:
            rel = rel_mod.parse_rel(path.read_bytes())
        except rel_mod.RelParseError:
            continue
        if not rel_mod.is_r_rel(rel):
            continue
        subs = rel_mod.read_rrel_subrecords(rel)
        total += len(subs)
        for s in subs:
            type_hist[s.sub_type] = type_hist.get(s.sub_type, 0) + 1
    # Sanity: every sub_type seen across the corpus must be in our catalogue.
    assert total > 0
    for st in type_hist:
        assert st in rel_mod.RREL_SUBREC_KNOWN_TYPES, \
            f"unexpected sub_type 0x{st:04x} ({type_hist[st]} records)"
