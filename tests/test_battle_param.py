"""Tests for formats.battle_param."""
from __future__ import annotations
import os

import json
from pathlib import Path

import pytest

from formats.battle_param import (
    ANIMATIONS_SCHEMA,
    ATTACKS_SCHEMA,
    BattleParamFile,
    DIFFICULTY_NAMES,
    ENTRIES_PER_DIFFICULTY,
    FILE_SIZE,
    NUM_DIFFICULTIES,
    PER_DIFFICULTY_SIZE,
    RESISTS_SCHEMA,
    SLOT_NAMES,
    STATS_SCHEMA,
    VALID_VARIANTS,
    VARIANT_TO_FILENAME,
    from_json,
    parse,
    serialize,
    slot_table_json,
    to_json,
)


# Locate Booma.Server fixtures (sole local source) and mark availability.
FIXTURES = Path(os.path.expanduser("~/Repositories/psobb2/Booma.Server/Data"))
HAS_FIXTURES = FIXTURES.is_dir() and any(FIXTURES.glob("BattleParamEntry*.dat"))


# ---------------------------------------------------------------------------
# Sanity / sizes
# ---------------------------------------------------------------------------
def test_file_size_constant():
    assert FILE_SIZE == 0xF600
    assert PER_DIFFICULTY_SIZE * NUM_DIFFICULTIES == FILE_SIZE
    assert NUM_DIFFICULTIES == 4
    assert ENTRIES_PER_DIFFICULTY == 0x60


def test_difficulty_names():
    assert DIFFICULTY_NAMES == ("Normal", "Hard", "VeryHard", "Ultimate")


def test_slot_table_includes_known_mobs():
    # Per Blue Burst Patch Project battleparam.h
    assert SLOT_NAMES[0x4B] == "Booma"
    assert SLOT_NAMES[0x49] == "Hildebear"
    assert SLOT_NAMES[0x0F] == "DeRolLe"
    assert SLOT_NAMES[0x12] == "Dragon"
    assert SLOT_NAMES[0x31] == "PanArms"


def test_variant_filenames():
    assert set(VARIANT_TO_FILENAME) == set(VALID_VARIANTS)
    assert VARIANT_TO_FILENAME["on"] == "BattleParamEntry_on.dat"
    assert VARIANT_TO_FILENAME["lab_on"] == "BattleParamEntry_lab_on.dat"


# ---------------------------------------------------------------------------
# Bad input handling
# ---------------------------------------------------------------------------
def test_parse_rejects_wrong_size():
    with pytest.raises(ValueError):
        parse(b"\x00" * 100)
    with pytest.raises(ValueError):
        parse(b"\x00" * (FILE_SIZE + 1))


def test_parse_rejects_non_bytes():
    with pytest.raises(ValueError):
        parse("not bytes")  # type: ignore[arg-type]


def test_serialize_rejects_wrong_difficulty_count():
    bpf = parse(b"\x00" * FILE_SIZE)
    bpf.difficulties.pop()
    with pytest.raises(ValueError):
        serialize(bpf)


# ---------------------------------------------------------------------------
# Zero-buffer round-trip
# ---------------------------------------------------------------------------
def test_zero_buffer_roundtrip():
    raw = b"\x00" * FILE_SIZE
    bpf = parse(raw)
    assert len(bpf.difficulties) == NUM_DIFFICULTIES
    for d in bpf.difficulties:
        assert len(d.entries) == ENTRIES_PER_DIFFICULTY
    out = serialize(bpf)
    assert out == raw


# ---------------------------------------------------------------------------
# Real fixture round-trip (Booma.Server BattleParamEntry*.dat)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_FIXTURES, reason="Booma.Server fixtures not present")
@pytest.mark.parametrize(
    "filename",
    [
        "BattleParamEntry.dat",
        "BattleParamEntry_on.dat",
        "BattleParamEntry_lab.dat",
        "BattleParamEntry_lab_on.dat",
        "BattleParamEntry_ep4.dat",
        "BattleParamEntry_ep4_on.dat",
    ],
)
def test_real_fixture_byte_exact_roundtrip(filename: str):
    """Parse + serialize must be byte-identical for every shipped variant."""
    path = FIXTURES / filename
    raw = path.read_bytes()
    assert len(raw) == FILE_SIZE
    bpf = parse(raw)
    out = serialize(bpf)
    assert out == raw, f"roundtrip mismatch on {filename}"


# ---------------------------------------------------------------------------
# Single-field mutation isolation
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_FIXTURES, reason="Booma.Server fixtures not present")
def test_mutate_single_field_only_changes_that_field():
    """Mutate Booma's walk_speed (animations.fparam3 = engaged move
    speed) in difficulty 0 and verify only those 4 bytes change.
    """
    raw = (FIXTURES / "BattleParamEntry_on.dat").read_bytes()
    bpf = parse(raw)
    booma_slot = 0x4B
    booma = bpf.difficulties[0].entries[booma_slot]
    old = booma.animations["fparam3"]
    # Pick a value whose float32 bit pattern overlaps no zero bytes with
    # the original (so we always produce 4 differing bytes regardless of
    # the stock value's bit pattern).
    new = -123456.789
    assert old != new, "test value coincidentally matches stock"
    booma.animations["fparam3"] = new
    # Drop the bits sidecar so the new value (not the original bit
    # pattern) gets written.
    booma.animations.pop("_fparam3_bits", None)
    out = serialize(bpf)

    # Compute byte diffs
    diffs = [(i, raw[i], out[i]) for i in range(len(raw)) if raw[i] != out[i]]
    # 4 bytes must lie in one contiguous span, but the count may be 1..4
    # depending on how many bytes coincide with the original float. We
    # just check no diff lies outside the expected 4-byte float window.
    assert 1 <= len(diffs) <= 4, f"expected 1..4 differing bytes, got {len(diffs)}"
    if diffs:
        first = diffs[0][0]
        last = diffs[-1][0]
        assert last - first < 4, (
            f"diffs span {last - first} bytes (>= 4); fields outside "
            "the mutated float changed"
        )


# ---------------------------------------------------------------------------
# JSON conversion
# ---------------------------------------------------------------------------
def test_json_roundtrip_zero():
    raw = b"\x00" * FILE_SIZE
    bpf = parse(raw, variant="on")
    text = to_json(bpf)
    bpf2 = from_json(text)
    out = serialize(bpf2)
    assert out == raw
    assert bpf2.variant == "on"


@pytest.mark.skipif(not HAS_FIXTURES, reason="Booma.Server fixtures not present")
def test_json_roundtrip_real_fixture():
    raw = (FIXTURES / "BattleParamEntry_on.dat").read_bytes()
    bpf = parse(raw, variant="on")
    text = to_json(bpf)
    # Ensure JSON is valid
    obj = json.loads(text)
    assert "difficulties" in obj
    assert len(obj["difficulties"]) == NUM_DIFFICULTIES
    bpf2 = from_json(text)
    out = serialize(bpf2)
    assert out == raw


def test_slot_table_sidecar_is_valid_json():
    txt = slot_table_json()
    obj = json.loads(txt)
    assert "slots" in obj
    # Should contain a Booma entry
    found = [v for k, v in obj["slots"].items() if v == "Booma"]
    assert found


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------
def test_schemas_sum_to_struct_sizes():
    import struct as st
    assert sum(st.calcsize(f) * c for _, f, c in STATS_SCHEMA) == 0x24
    assert sum(st.calcsize(f) * c for _, f, c in ATTACKS_SCHEMA) == 0x30
    assert sum(st.calcsize(f) * c for _, f, c in RESISTS_SCHEMA) == 0x20
    assert sum(st.calcsize(f) * c for _, f, c in ANIMATIONS_SCHEMA) == 0x30


# ---------------------------------------------------------------------------
# Slot identity preserved across round-trip
# ---------------------------------------------------------------------------
def test_slot_index_preserved():
    raw = b"\x00" * FILE_SIZE
    bpf = parse(raw)
    for d in bpf.difficulties:
        for i, ent in enumerate(d.entries):
            assert ent.slot == i
