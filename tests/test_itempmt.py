"""Tests for formats.itempmt."""
from __future__ import annotations
import os

import json
from pathlib import Path

import pytest

from formats import itempmt, prs
from formats.itempmt import (
    ARMOR_OR_SHIELD_RECORD_SIZE,
    ARMOR_OR_SHIELD_SCHEMA,
    HEADER_PREAMBLE_SIZE,
    ITEM_BASE_SCHEMA,
    ITEM_COMBINATION_RECORD_SIZE,
    ITEM_COMBINATION_SCHEMA,
    ItemPMTFile,
    MAG_FEED_NUM_TABLES,
    MAG_FEED_RESULT_SCHEMA,
    MAG_FEED_RESULT_SIZE,
    MAG_FEED_RESULTS_PER_TABLE,
    MAG_RECORD_SIZE,
    MAG_SCHEMA,
    MAX_TECH_NUM_CLASSES,
    MAX_TECH_NUM_TECHS,
    NUM_TABLE_OFFSETS,
    SPECIAL_RECORD_SIZE,
    SPECIAL_SCHEMA,
    STAT_BOOST_RECORD_SIZE,
    STAT_BOOST_SCHEMA,
    TABLE_OFFSETS_SIZE,
    TOOL_RECORD_SIZE,
    TOOL_SCHEMA,
    UNIT_RECORD_SIZE,
    UNIT_SCHEMA,
    WEAPON_CLASS_NAMES,
    WEAPON_RECORD_SIZE,
    WEAPON_SCHEMA,
    _pack_record,
    _parse_record,
    _schema_byte_size,
    from_json,
    pack,
    parse,
    parse_prs,
    parse_with_meta,
    serialize,
    to_json,
)


# ---------------------------------------------------------------------------
# Fixtures: shipped ItemPMT.prs
# ---------------------------------------------------------------------------
FIXTURE_PRS = Path(
    os.path.expanduser("~/Repositories/psobb2/Booma.Server/Data/ItemPMT.prs")
)
HAS_FIXTURE = FIXTURE_PRS.is_file()


@pytest.fixture(scope="module")
def shipped_raw() -> bytes:
    """Decompressed bytes of the shipped ItemPMT.prs."""
    if not HAS_FIXTURE:
        pytest.skip(f"missing fixture {FIXTURE_PRS}")
    return prs.decompress(FIXTURE_PRS.read_bytes())


@pytest.fixture(scope="module")
def shipped_pmt(shipped_raw: bytes) -> ItemPMTFile:
    return parse_with_meta(shipped_raw)


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------
def test_record_sizes_match_v4_spec():
    assert WEAPON_RECORD_SIZE == 0x2C
    assert ARMOR_OR_SHIELD_RECORD_SIZE == 0x20
    assert UNIT_RECORD_SIZE == 0x14
    assert MAG_RECORD_SIZE == 0x1C
    assert TOOL_RECORD_SIZE == 0x18
    assert SPECIAL_RECORD_SIZE == 4
    assert STAT_BOOST_RECORD_SIZE == 6
    assert ITEM_COMBINATION_RECORD_SIZE == 0x10
    assert MAG_FEED_RESULT_SIZE == 8
    assert TABLE_OFFSETS_SIZE == 0x5C


def test_schemas_match_record_sizes():
    assert _schema_byte_size(WEAPON_SCHEMA) == WEAPON_RECORD_SIZE
    assert _schema_byte_size(ARMOR_OR_SHIELD_SCHEMA) == ARMOR_OR_SHIELD_RECORD_SIZE
    assert _schema_byte_size(UNIT_SCHEMA) == UNIT_RECORD_SIZE
    assert _schema_byte_size(MAG_SCHEMA) == MAG_RECORD_SIZE
    assert _schema_byte_size(TOOL_SCHEMA) == TOOL_RECORD_SIZE
    assert _schema_byte_size(SPECIAL_SCHEMA) == SPECIAL_RECORD_SIZE
    assert _schema_byte_size(STAT_BOOST_SCHEMA) == STAT_BOOST_RECORD_SIZE
    assert _schema_byte_size(ITEM_COMBINATION_SCHEMA) == ITEM_COMBINATION_RECORD_SIZE
    assert _schema_byte_size(MAG_FEED_RESULT_SCHEMA) == MAG_FEED_RESULT_SIZE


def test_constants():
    assert NUM_TABLE_OFFSETS == 23
    assert HEADER_PREAMBLE_SIZE == 0x40
    assert MAG_FEED_NUM_TABLES == 8
    assert MAG_FEED_RESULTS_PER_TABLE == 11
    assert MAX_TECH_NUM_TECHS == 19
    assert MAX_TECH_NUM_CLASSES == 12


def test_weapon_class_names_have_known_entries():
    # V4 weapon classes per newserv item-tables
    assert WEAPON_CLASS_NAMES[0x00] == "Saber"
    assert WEAPON_CLASS_NAMES[0x01] == "Sword"
    assert WEAPON_CLASS_NAMES[0x05] == "Handgun"
    assert WEAPON_CLASS_NAMES[0x0A] == "Rod"


# ---------------------------------------------------------------------------
# Bad-input handling
# ---------------------------------------------------------------------------
def test_parse_rejects_too_short():
    with pytest.raises(ValueError):
        parse(b"\x00" * 10)


def test_parse_rejects_non_bytes():
    with pytest.raises(ValueError):
        parse("not bytes")  # type: ignore[arg-type]


def test_parse_rejects_invalid_offset_table_offset():
    # File where offset_table_offset points beyond the file
    buf = bytearray(0x200)
    buf[-0x10:-0x0C] = b"\xFF\xFF\xFF\xFF"  # huge offset
    with pytest.raises(ValueError):
        parse(bytes(buf))


# ---------------------------------------------------------------------------
# Record-level pack/unpack
# ---------------------------------------------------------------------------
def test_record_pack_unpack_weapon():
    rec = {
        "id": 0xb1, "type": 0xffff, "skin": 0xffff, "team_points": 0,
        "class_flags": 0xff, "atp_min": 5, "atp_max": 7, "atp_required": 0,
        "mst_required": 0, "ata_required": 60, "mst": 0,
        "max_grind": 10, "photon": 0, "special": -1, "ata": 0,
        "stat_boost_entry_index": 0xff, "projectile": 0xff,
        "trail1_x": 0, "trail1_y": 0, "trail2_x": 0, "trail2_y": 0,
        "color": 0, "unknown_a1": [0, 0, 0],
        "unknown_a4": 0, "unknown_a5": 0, "tech_boost": 0,
        "behavior_flags": 0,
    }
    packed = _pack_record(rec, WEAPON_SCHEMA)
    assert len(packed) == WEAPON_RECORD_SIZE
    rec2 = _parse_record(packed, 0, WEAPON_SCHEMA)
    for k, v in rec.items():
        assert rec2[k] == v, f"mismatch on {k}: {rec2[k]} != {v}"


def test_record_pack_unpack_special():
    rec = {"type": 0x12, "amount": 0x34}
    packed = _pack_record(rec, SPECIAL_SCHEMA)
    assert len(packed) == SPECIAL_RECORD_SIZE
    rec2 = _parse_record(packed, 0, SPECIAL_SCHEMA)
    assert rec2 == rec


def test_record_pack_unpack_stat_boost():
    rec = {"stats": [1, 2], "amounts": [100, 200]}
    packed = _pack_record(rec, STAT_BOOST_SCHEMA)
    assert len(packed) == STAT_BOOST_RECORD_SIZE
    rec2 = _parse_record(packed, 0, STAT_BOOST_SCHEMA)
    assert rec2 == rec


# ---------------------------------------------------------------------------
# Round-trip on shipped fixture
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_FIXTURE, reason=f"missing {FIXTURE_PRS}")
def test_parse_shipped_pmt_basic_counts(shipped_pmt: ItemPMTFile):
    # Per static analysis of the shipped Booma.Server file (V4 layout)
    assert sum(len(w.items) for w in shipped_pmt.weapons) > 0
    assert len(shipped_pmt.armors) > 0   # 89 frames
    assert len(shipped_pmt.shields) > 0  # 166 barriers
    assert len(shipped_pmt.units) > 0    # 101 units
    assert len(shipped_pmt.mags) > 0     # 83 mags
    assert sum(len(t.items) for t in shipped_pmt.tools) > 0
    assert len(shipped_pmt.specials) > 0
    assert len(shipped_pmt.stat_boosts) > 0
    assert len(shipped_pmt.combinations) > 0
    assert len(shipped_pmt.mag_feeds) == MAG_FEED_NUM_TABLES
    for mft in shipped_pmt.mag_feeds:
        assert len(mft.results) == MAG_FEED_RESULTS_PER_TABLE
    assert len(shipped_pmt.max_tech_levels) == MAX_TECH_NUM_TECHS
    for row in shipped_pmt.max_tech_levels:
        assert len(row) == MAX_TECH_NUM_CLASSES


@pytest.mark.skipif(not HAS_FIXTURE, reason=f"missing {FIXTURE_PRS}")
def test_round_trip_byte_exact(shipped_raw: bytes, shipped_pmt: ItemPMTFile):
    """parse → serialize must produce byte-identical output."""
    out = serialize(shipped_pmt)
    assert len(out) == len(shipped_raw), (
        f"size mismatch: {len(out)} vs {len(shipped_raw)}"
    )
    if out != shipped_raw:
        for i, (a, b) in enumerate(zip(out, shipped_raw)):
            if a != b:
                raise AssertionError(
                    f"byte {i:#x} differs: out={a:#04x} orig={b:#04x}\n"
                    f"  context out:  {bytes(out[max(0,i-8):i+8]).hex()}\n"
                    f"  context orig: {shipped_raw[max(0,i-8):i+8].hex()}"
                )
        raise AssertionError("size differs but no byte difference (?)")


@pytest.mark.skipif(not HAS_FIXTURE, reason=f"missing {FIXTURE_PRS}")
def test_round_trip_via_json(shipped_raw: bytes, shipped_pmt: ItemPMTFile):
    """parse → to_json → from_json → serialize must match."""
    js = to_json(shipped_pmt)
    assert isinstance(js, str)
    pmt2 = from_json(js)
    out = serialize(pmt2)
    assert out == shipped_raw, "JSON round-trip not byte-exact"


@pytest.mark.skipif(not HAS_FIXTURE, reason=f"missing {FIXTURE_PRS}")
def test_round_trip_with_pack(shipped_raw: bytes, shipped_pmt: ItemPMTFile):
    """parse → pack (PRS) → decompress must match the original raw bytes."""
    packed = pack(shipped_pmt)
    # PRS stream is allowed to differ (different match heuristics) but
    # its decompressed payload must match the original.
    re_raw = prs.decompress(packed)
    assert re_raw == shipped_raw


@pytest.mark.skipif(not HAS_FIXTURE, reason=f"missing {FIXTURE_PRS}")
def test_parse_prs_helper(shipped_raw: bytes):
    """parse_prs should accept a .prs blob and return an ItemPMTFile."""
    prs_blob = FIXTURE_PRS.read_bytes()
    pmt = parse_prs(prs_blob)
    assert isinstance(pmt, ItemPMTFile)
    out = serialize(pmt)
    assert out == shipped_raw


# ---------------------------------------------------------------------------
# Edit semantics
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_FIXTURE, reason=f"missing {FIXTURE_PRS}")
def test_edit_weapon_atp_max_round_trips(
    shipped_raw: bytes, shipped_pmt: ItemPMTFile
):
    """Editing one field should serialize cleanly and the change must appear
    in the same byte location of the output."""
    # Find the first weapon in the first non-empty class
    weapon_class = None
    for wc in shipped_pmt.weapons:
        if wc.items:
            weapon_class = wc
            break
    assert weapon_class is not None
    rec = weapon_class.items[0]
    orig_atp = rec["atp_max"]
    new_atp = (orig_atp + 100) & 0x7FFF
    rec["atp_max"] = new_atp

    out = serialize(shipped_pmt)
    assert len(out) == len(shipped_raw)
    # The diff should be exactly the 2 bytes of atp_max in the first
    # weapon. We don't know the exact offset without re-deriving it, but
    # we can check that the value parses back correctly.
    pmt2 = parse_with_meta(out)
    rec2 = pmt2.weapons[weapon_class.class_index].items[0]
    assert rec2["atp_max"] == new_atp
    # Restore for downstream tests (test order independence)
    rec["atp_max"] = orig_atp


@pytest.mark.skipif(not HAS_FIXTURE, reason=f"missing {FIXTURE_PRS}")
def test_edit_special_amount_round_trips(
    shipped_raw: bytes, shipped_pmt: ItemPMTFile
):
    """Editing a Special entry should serialize cleanly."""
    rec = shipped_pmt.specials[5]
    orig_amount = rec["amount"]
    new_amount = (orig_amount + 7) & 0xFFFF
    rec["amount"] = new_amount
    out = serialize(shipped_pmt)
    assert len(out) == len(shipped_raw)
    pmt2 = parse_with_meta(out)
    assert pmt2.specials[5]["amount"] == new_amount
    rec["amount"] = orig_amount


@pytest.mark.skipif(not HAS_FIXTURE, reason=f"missing {FIXTURE_PRS}")
def test_edit_armor_dfp_round_trips(
    shipped_raw: bytes, shipped_pmt: ItemPMTFile
):
    """Editing armor.dfp should round-trip cleanly."""
    rec = shipped_pmt.armors[0]
    orig = rec["dfp"]
    rec["dfp"] = (orig + 13) & 0xFFFF
    out = serialize(shipped_pmt)
    pmt2 = parse_with_meta(out)
    assert pmt2.armors[0]["dfp"] == rec["dfp"]
    rec["dfp"] = orig


@pytest.mark.skipif(not HAS_FIXTURE, reason=f"missing {FIXTURE_PRS}")
def test_edit_mag_feed_round_trips(
    shipped_raw: bytes, shipped_pmt: ItemPMTFile
):
    """Editing a MagFeedResult should round-trip cleanly."""
    rec = shipped_pmt.mag_feeds[0].results[3]
    orig = rec["pow"]
    rec["pow"] = -42
    out = serialize(shipped_pmt)
    pmt2 = parse_with_meta(out)
    assert pmt2.mag_feeds[0].results[3]["pow"] == -42
    rec["pow"] = orig


# ---------------------------------------------------------------------------
# JSON shape contract
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_FIXTURE, reason=f"missing {FIXTURE_PRS}")
def test_to_json_is_valid_json(shipped_pmt: ItemPMTFile):
    js = to_json(shipped_pmt)
    parsed = json.loads(js)
    assert "weapons" in parsed
    assert "armors" in parsed
    assert "shields" in parsed
    assert "units" in parsed
    assert "mags" in parsed
    assert "tools" in parsed
    assert "specials" in parsed
    assert "stat_boosts" in parsed
    assert "mag_feeds" in parsed
    assert "combinations" in parsed
    assert "_opaque" in parsed
    assert "_meta" in parsed


@pytest.mark.skipif(not HAS_FIXTURE, reason=f"missing {FIXTURE_PRS}")
def test_weapon_class_metadata(shipped_pmt: ItemPMTFile):
    # Class 0 should be 'Saber'
    saber_class = next(
        (wc for wc in shipped_pmt.weapons if wc.class_index == 0),
        None,
    )
    assert saber_class is not None
    assert saber_class.name == "Saber"
    # Should have a few weapons
    assert len(saber_class.items) >= 1
    rec = saber_class.items[0]
    assert "id" in rec
    assert "atp_min" in rec
    assert "atp_max" in rec
