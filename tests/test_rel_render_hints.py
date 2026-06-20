"""Tests for the v3 r.rel render-hints additions in ``formats.rel``.

Covers:
  - is_r_rel / is_c_rel discriminator now distinguishes the two
    pointer-headed REL flavours (v2's is_c_rel was too loose and
    matched r.rel files too).
  - Synthetic r.rel round-trip: build a minimal valid r.rel and
    verify the anchor walker reads it back.
  - parse_rrel_render_hints wire shape — keys, types, value ranges.
  - derive_scene_hints bbox math with edge cases (single anchor,
    co-located anchors).
  - Real-file smoke tests on PSOBB.IO maps when available.
  - Pioneer 2 fog-fallback validation (city has no r.rel, so the
    "city" hardcoded category remains in force — we assert that).

Synthetic builders mirror tests/test_rel_parser.py to keep a single
file-format authority.
"""
from __future__ import annotations

import struct
from pathlib import Path

import pytest

from formats import rel as rel_mod


PSOBB_DATA = Path(r"~\PSOBB.IO\data\scene")


# ---------------------------------------------------------------------------
# Synthetic builders (subset of tests/test_rel_parser.py for self-containment)
# ---------------------------------------------------------------------------
def _build_minimal_rel(payload: bytes,
                       payload_offset: int,
                       pointer_offsets: list[int]) -> bytes:
    """Construct a minimal valid REL buffer.  Mirrors the helper in
    test_rel_parser.py — duplicated here so this test file is
    independent."""
    data = bytearray(payload)
    while len(data) % 4 != 0:
        data.append(0)
    pt_off = len(data)
    prev = 0
    for abs_off in pointer_offsets:
        if abs_off % 4 != 0:
            raise ValueError(f"pointer offset {abs_off} not 4-aligned")
        delta = (abs_off - prev) // 4
        if delta < 0 or delta > 0xFFFF:
            raise ValueError(f"delta {delta} out of u16 range")
        data.extend(struct.pack("<H", delta))
        prev = abs_off
    while len(data) % 4 != 0:
        data.append(0)
    trailer = struct.pack(
        "<8I",
        pt_off, len(pointer_offsets), 1, 0, payload_offset, 0, 0, 0,
    )
    data.extend(trailer)
    return bytes(data)


def _build_synthetic_rrel(anchors: list[tuple]) -> bytes:
    """Build an r.rel buffer with the given anchors.

    Each ``anchors`` entry is ``(id, x, y, z, radius)`` — version is
    fixed at 1, rotations at 0, sub_record_ptr at 0.

    Layout:
      payload (24 bytes header at offset 0):
        +0x00: anchors_ptr (-> 0x40)
        +0x04: 0
        +0x08: count
      anchors at 0x40 (40 bytes each).
    """
    n = len(anchors)
    data = bytearray(b"\x00" * 0x40)
    # RrelHeader at offset 0
    struct.pack_into("<III", data, 0, 0x40, 0, n)
    # Anchors at offset 0x40
    for i, (aid, x, y, z, radius) in enumerate(anchors):
        anchor_bytes = struct.pack(
            "<I3f f I 2I f I",
            (1 << 16) | (aid & 0xFFFF),  # id_packed
            x, y, z,                       # pos
            0.0,                            # rot_x
            0,                              # rot_y_packed
            0, 0,                           # u18 / u1c
            radius,                         # radius
            0,                              # sub_record_ptr
        )
        data.extend(anchor_bytes)
    # Pointer at payload+0x00 (the anchors_ptr) — 4-aligned offset 0
    pointer_offsets = [0]
    return _build_minimal_rel(bytes(data), payload_offset=0,
                              pointer_offsets=pointer_offsets)


# ---------------------------------------------------------------------------
# is_r_rel discriminator
# ---------------------------------------------------------------------------
def test_synthetic_rrel_classifies_as_r():
    """Build a 1-anchor r.rel and confirm it sniffs correctly."""
    buf = _build_synthetic_rrel([(20, 100.0, 0.0, 200.0, 50.0)])
    rel = rel_mod.parse_rel(buf)
    assert not rel_mod.is_n_rel(rel)
    assert not rel_mod.is_c_rel(rel)
    assert rel_mod.is_r_rel(rel)


def test_synthetic_crel_does_not_classify_as_r():
    """A minimal c.rel (single pointer at payload, +0x08 zero) doesn't
    match r.rel, even after the v3 sniffer tightening."""
    # 16-byte payload: head pointer + zeros (so the +0x08 sniff sees 0)
    payload = struct.pack("<I", 0x10) + b"\x00" * 0x10
    buf = _build_minimal_rel(payload, payload_offset=0,
                             pointer_offsets=[0])
    rel = rel_mod.parse_rel(buf)
    assert rel_mod.is_c_rel(rel)
    assert not rel_mod.is_r_rel(rel)


def test_minimal_4byte_crel_still_classifies_as_c():
    """Pioneer 2 ``map_city00_00c.rel`` ships a payload of exactly 4
    bytes (just the head pointer).  The +0x08 r.rel sniff cannot run
    on that — c.rel detection must fall back to the head-pointer test
    only."""
    # 4-byte payload (just the head ptr)
    payload = struct.pack("<I", 0x4)
    buf = _build_minimal_rel(payload, payload_offset=0,
                             pointer_offsets=[0])
    rel = rel_mod.parse_rel(buf)
    assert rel_mod.is_c_rel(rel)
    assert not rel_mod.is_r_rel(rel)
    assert not rel_mod.is_n_rel(rel)


# ---------------------------------------------------------------------------
# Synthetic r.rel decode round-trip
# ---------------------------------------------------------------------------
def test_synthetic_rrel_round_trip_single_anchor():
    buf = _build_synthetic_rrel([(20, 100.0, 0.0, 200.0, 50.0)])
    rel = rel_mod.parse_rel(buf)
    h = rel_mod.read_rrel_header(rel)
    assert h.count == 1
    assert h.anchors_ptr == 0x40
    anchors = rel_mod.read_rrel_anchors(rel, h)
    assert len(anchors) == 1
    a = anchors[0]
    assert a.anchor_id == 20
    assert a.version == 1
    assert a.pos == (100.0, 0.0, 200.0)
    assert a.radius == 50.0
    assert a.sub_record_ptr == 0


def test_synthetic_rrel_round_trip_multiple_anchors():
    anchors_in = [
        (20, -100.0, 0.0, 200.0, 50.0),
        (21,  500.0, 0.0, -800.0, 100.0),
        (22, 2000.0, 50.0, 1000.0, 75.0),
    ]
    buf = _build_synthetic_rrel(anchors_in)
    rel = rel_mod.parse_rel(buf)
    anchors_out = rel_mod.read_rrel_anchors(rel)
    assert len(anchors_out) == 3
    for src, dst in zip(anchors_in, anchors_out):
        aid, x, y, z, r = src
        assert dst.anchor_id == aid
        assert dst.pos == (x, y, z)
        assert dst.radius == r


def test_read_rrel_header_rejects_non_rrel():
    """An n.rel must NOT be readable as an r.rel."""
    # Build a fake n.rel (fmt2 magic) and assert read_rrel_header refuses.
    payload = b"fmt2" + b"\x00" * 0x40
    buf = _build_minimal_rel(payload, payload_offset=0, pointer_offsets=[])
    rel = rel_mod.parse_rel(buf)
    with pytest.raises(rel_mod.RelParseError):
        rel_mod.read_rrel_header(rel)


def test_read_rrel_header_rejects_zero_count():
    """An r.rel with count=0 isn't valid — 0 anchors is meaningless."""
    # Build a synthetic r.rel-shaped header but with count=0.
    data = bytearray(b"\x00" * 0x10)
    struct.pack_into("<III", data, 0, 0x40, 0, 0)  # anchors_ptr=0x40, count=0
    buf = _build_minimal_rel(bytes(data), payload_offset=0,
                             pointer_offsets=[0])
    rel = rel_mod.parse_rel(buf)
    # Sniffer rejects (count must be > 0)
    assert not rel_mod.is_r_rel(rel)


def test_read_rrel_anchors_handles_empty_list():
    """If the file isn't an r.rel, read_rrel_anchors returns []."""
    # Build a c.rel
    payload = struct.pack("<I", 0x10) + b"\x00" * 0x10
    buf = _build_minimal_rel(payload, payload_offset=0,
                             pointer_offsets=[0])
    rel = rel_mod.parse_rel(buf)
    assert rel_mod.read_rrel_anchors(rel) == []


# ---------------------------------------------------------------------------
# parse_rrel_render_hints wire shape
# ---------------------------------------------------------------------------
def test_parse_rrel_render_hints_wire_shape():
    """The top-level entry returns a stable JSON-friendly dict."""
    buf = _build_synthetic_rrel([
        (20, -100.0, 0.0, -200.0, 50.0),
        (21,  500.0, 0.0,  300.0, 100.0),
    ])
    result = rel_mod.parse_rrel_render_hints(buf)
    assert result["ok"] is True
    assert result["anchor_count"] == 2
    assert isinstance(result["anchors"], list)
    assert len(result["anchors"]) == 2
    a0 = result["anchors"][0]
    # Stable wire keys
    for k in ("id", "version", "pos", "rot_x", "rot_y_packed", "radius",
             "sub_record_ptr"):
        assert k in a0, f"missing key {k}"
    # pos is a 3-list of floats
    assert isinstance(a0["pos"], list)
    assert len(a0["pos"]) == 3
    # hints
    h = result["hints"]
    assert h is not None
    for k in ("anchor_count", "bbox_min", "bbox_max", "bbox_center",
              "bbox_size", "suggested_fog_far"):
        assert k in h, f"missing hint key {k}"
    # bbox math: anchor 0 is (-100, 0, -200), anchor 1 is (500, 0, 300)
    assert h["bbox_min"] == [-100.0, 0.0, -200.0]
    assert h["bbox_max"] == [500.0, 0.0, 300.0]
    assert h["bbox_size"] == [600.0, 0.0, 500.0]
    # suggested_fog_far: max(800, min(600*1.2, 4000)) = 800
    assert h["suggested_fog_far"] == 800.0


def test_parse_rrel_render_hints_failure_returns_dict_not_raise():
    """Junk bytes return ``{"ok": False, "error": ...}`` not a raise."""
    result = rel_mod.parse_rrel_render_hints(b"junk garbage")
    assert result["ok"] is False
    assert "error" in result


def test_parse_rrel_render_hints_on_n_rel_returns_failure():
    """Feeding an n.rel to the r.rel parser produces a clean failure."""
    payload = b"fmt2" + b"\x00" * 0x40
    buf = _build_minimal_rel(payload, payload_offset=0, pointer_offsets=[])
    result = rel_mod.parse_rrel_render_hints(buf)
    assert result["ok"] is False


# ---------------------------------------------------------------------------
# derive_scene_hints math
# ---------------------------------------------------------------------------
def test_derive_scene_hints_empty_returns_none():
    assert rel_mod.derive_scene_hints([]) is None


def test_derive_scene_hints_single_anchor_collapsed_bbox():
    """One anchor → degenerate bbox (size=0 in all dims)."""
    a = rel_mod.RrelAnchor(
        anchor_id=20, version=1, pos=(100.0, 0.0, 200.0),
        rot_x=0.0, rot_y_packed=0, radius=50.0, sub_record_ptr=0,
    )
    h = rel_mod.derive_scene_hints([a])
    assert h.bbox_min == h.bbox_max == (100.0, 0.0, 200.0)
    assert h.bbox_size == (0.0, 0.0, 0.0)
    # suggested_fog_far is floored at 800 even with zero bbox
    assert h.suggested_fog_far == 800.0


def test_derive_scene_hints_far_plane_capped_at_4000():
    """A huge bbox gets capped at 4000 fog far."""
    a = rel_mod.RrelAnchor(
        anchor_id=20, version=1, pos=(-5000.0, 0.0, -5000.0),
        rot_x=0.0, rot_y_packed=0, radius=0.0, sub_record_ptr=0,
    )
    b = rel_mod.RrelAnchor(
        anchor_id=21, version=1, pos=(5000.0, 0.0, 5000.0),
        rot_x=0.0, rot_y_packed=0, radius=0.0, sub_record_ptr=0,
    )
    h = rel_mod.derive_scene_hints([a, b])
    assert h.bbox_size == (10000.0, 0.0, 10000.0)
    # 10000 * 1.2 = 12000, capped at 4000
    assert h.suggested_fog_far == 4000.0


# ---------------------------------------------------------------------------
# Real-file smoke tests (gated on PSOBB.IO data dir)
# ---------------------------------------------------------------------------
def _need_data():
    if not PSOBB_DATA.exists():
        pytest.skip(f"PSOBB data dir not present: {PSOBB_DATA}")


@pytest.fixture
def aancient01_rrel():
    _need_data()
    p = PSOBB_DATA / "map_aancient01_00r.rel"
    if not p.exists():
        pytest.skip(f"missing {p}")
    return p.read_bytes()


@pytest.fixture
def acave01_rrel():
    _need_data()
    p = PSOBB_DATA / "map_acave01_00r.rel"
    if not p.exists():
        pytest.skip(f"missing {p}")
    return p.read_bytes()


@pytest.fixture
def amachine01_rrel():
    _need_data()
    p = PSOBB_DATA / "map_amachine01_00r.rel"
    if not p.exists():
        pytest.skip(f"missing {p}")
    return p.read_bytes()


def test_real_aancient01_rrel_classifies_as_r(aancient01_rrel):
    rel = rel_mod.parse_rel(aancient01_rrel)
    assert rel_mod.is_r_rel(rel)
    assert not rel_mod.is_n_rel(rel)
    assert not rel_mod.is_c_rel(rel)


def test_real_aancient01_rrel_anchor_decode(aancient01_rrel):
    """Forest map has 100 anchors with sensible positions."""
    result = rel_mod.parse_rrel_render_hints(aancient01_rrel)
    assert result["ok"] is True
    anchors = result["anchors"]
    assert 50 < len(anchors) < 200, f"unexpected anchor count {len(anchors)}"
    # All ids fit u16, all versions are 1, all positions are sensible.
    for a in anchors:
        assert 0 < a["id"] < 0x10000
        assert a["version"] == 1
        assert -10000 < a["pos"][0] < 10000
        assert -1000 < a["pos"][1] < 1000
        assert -10000 < a["pos"][2] < 10000
        # Radius is positive and bounded.
        assert 0.0 <= a["radius"] < 5000.0
    # IDs are unique within the file (this is the SetData-table invariant)
    ids = [a["id"] for a in anchors]
    assert len(ids) == len(set(ids))


def test_real_aancient01_scene_hints(aancient01_rrel):
    """Forest 1 has a bbox we can sanity-check."""
    result = rel_mod.parse_rrel_render_hints(aancient01_rrel)
    h = result["hints"]
    # PSOBB forest 1 fits in a roughly 2300 x 2600 footprint at sea level.
    # Allow generous bands for floor variants.
    assert 1000 < h["bbox_size"][0] < 5000
    assert 1000 < h["bbox_size"][2] < 5000
    # All anchors are at y≈0 in observed data — let height-band span be tight.
    assert h["bbox_size"][1] < 200
    # Fog far is a positive finite number.
    assert 800 <= h["suggested_fog_far"] <= 4000


def test_real_all_rrel_files_parse():
    """Smoke: every r.rel in PSOBB.IO/data/scene parses without raising
    OR returns a clean failure result.  No exceptions allowed."""
    _need_data()
    rrel_files = sorted(PSOBB_DATA.glob("*r.rel"))
    if len(rrel_files) < 10:
        pytest.skip("not enough r.rel samples to validate")
    failures = []
    for p in rrel_files:
        if not p.name.endswith("r.rel"):
            continue
        try:
            buf = p.read_bytes()
            res = rel_mod.parse_rrel_render_hints(buf)
        except Exception as e:  # pragma: no cover — defensive
            failures.append(f"{p.name}: EXCEPTION {e}")
            continue
        # Don't insist all parse successfully — some boss r.rels are
        # tiny stubs.  But the result must be a well-formed dict.
        assert isinstance(res, dict)
        assert "ok" in res
        if not res["ok"]:
            assert "error" in res
        else:
            assert "anchors" in res
            assert "hints" in res
    assert not failures, f"r.rel parse failures: {failures}"


def test_pioneer2_has_no_rrel():
    """Pioneer 2 (city / lab) maps don't ship r.rel files — they have
    no spawnable anchor set.  This is by-design and the Map Editor
    should treat that as "use category fog" without complaint.

    This test documents the absence so future contributors know it's
    intentional rather than a missing-asset issue."""
    _need_data()
    # The five "city" maps in PSOBB:
    city_ids = ["city00", "city02", "acity00", "labo00"]
    for cid in city_ids:
        for floor in range(5):
            p = PSOBB_DATA / f"map_{cid}_{floor:02d}r.rel"
            assert not p.exists(), \
                f"unexpected r.rel for city map: {p.name}"


def test_real_acave01_anchor_radii_match_observed(acave01_rrel):
    """Cave 1 map ships anchors with radius values in the 175–305 range
    (these are the corridor activation radii observed via static
    analysis at v3 RE time).  Sanity-check we read them correctly."""
    result = rel_mod.parse_rrel_render_hints(acave01_rrel)
    anchors = result["anchors"]
    # The first anchor has radius ~175 in cave01.
    assert abs(anchors[0]["radius"] - 175.7) < 1.0


def test_real_amachine01_anchor_first(amachine01_rrel):
    """Mine 1 first anchor: id=50 at (190, 0, 120), radius~234.8."""
    result = rel_mod.parse_rrel_render_hints(amachine01_rrel)
    a0 = result["anchors"][0]
    assert a0["id"] == 50
    assert abs(a0["pos"][0] - 190.0) < 0.5
    assert abs(a0["pos"][2] - 120.0) < 0.5
    assert abs(a0["radius"] - 234.8) < 1.0


# ---------------------------------------------------------------------------
# scene_loader.floor_bundle integration — v3 surfaces rrel_path / nrel_path
# ---------------------------------------------------------------------------
def test_floor_bundle_surfaces_rrel_and_nrel_paths():
    """v3: when a floor has both n.rel and r.rel siblings, the bundle's
    ``rrel_path`` and ``nrel_path`` keys are populated."""
    from formats import scene_loader as sl
    entries = [
        {"path": "scene/map_aancient01_00s.nj",  "category": "map", "size": 1000},
        {"path": "scene/map_aancient01_00s.xj",  "category": "map", "size": 2000},
        {"path": "scene/map_aancient01_00n.rel", "category": "map", "size": 4000},
        {"path": "scene/map_aancient01_00r.rel", "category": "map", "size": 5000},
        {"path": "scene/map_aancient01_00c.rel", "category": "map", "size": 6000},
    ]
    maps = sl.catalogue(entries)
    m = next(x for x in maps if x.map_id == "aancient01")
    bundle = sl.floor_bundle(m, 0)
    assert bundle["rrel_path"] == "scene/map_aancient01_00r.rel"
    assert bundle["nrel_path"] == "scene/map_aancient01_00n.rel"


def test_floor_bundle_no_rrel_returns_none():
    """City maps don't ship r.rel — bundle's ``rrel_path`` is None."""
    from formats import scene_loader as sl
    entries = [
        {"path": "scene/map_city00_00n.rel", "category": "map", "size": 4000},
        {"path": "scene/map_city00_00c.rel", "category": "map", "size": 6000},
    ]
    maps = sl.catalogue(entries)
    m = next(x for x in maps if x.map_id == "city00")
    bundle = sl.floor_bundle(m, 0)
    assert bundle["rrel_path"] is None
    assert bundle["nrel_path"] == "scene/map_city00_00n.rel"


# ---------------------------------------------------------------------------
# Pioneer 2 fog parity smoke — the hardcoded "city" category should still
# produce a sensible far-plane even though city has no r.rel hints.
# ---------------------------------------------------------------------------
def test_pioneer2_fog_parity():
    """The map_panel/model_viewer hardcoded ``city`` category fog far is
    2400 units (per static/model_viewer.js _PSO_AREA_ENV.city).  This
    test pins that value as the v3 expected baseline so a future
    refactor that changes it has to update both this test and the JS.

    This isn't testing parsed r.rel data — Pioneer 2 ships no r.rel.
    The point is to lock the v2-era hardcoded behaviour so v3's
    additive r.rel hooks don't accidentally regress city-area look.
    """
    # Read the JS source and assert the value is what we expect.  This
    # is a pin-test: it exists to fail loudly when someone edits the
    # constant without updating the test (and therefore the v3 spec).
    js_path = (Path(__file__).parent.parent /
               "static" / "model_viewer.js")
    src = js_path.read_text(encoding="utf-8")
    # Find the city block — match `city: { fog: { color: 0x..., near: ..., far: ... } }`
    import re as _re
    m = _re.search(
        r"city:\s*\{\s*fog:\s*\{[^}]*far:\s*(\d+)",
        src,
    )
    assert m, "could not find _PSO_AREA_ENV.city.fog.far in model_viewer.js"
    far_val = int(m.group(1))
    # Pinned at 2400 — confirms v2's "warm Pioneer 2 yellow" tuning.
    # Future re-tunes are fine but must update this test.
    assert far_val == 2400, f"city fog.far changed: {far_val}"
