"""Tests for formats.mob_dsl — the Tier 1 numeric authoring layer."""
from __future__ import annotations
import os

import json
import math
from pathlib import Path

import pytest

from formats import battle_param as bp_mod
from formats import mob_dsl


# Shared Booma.Server fixtures (already used by test_battle_param).
FIXTURES = Path(os.path.expanduser("~/Repositories/psobb2/Booma.Server/Data"))
HAS_FIXTURES = FIXTURES.is_dir() and (FIXTURES / "BattleParamEntry_on.dat").is_file()


# ---------------------------------------------------------------------------
# Schema coverage
# ---------------------------------------------------------------------------
def test_schema_table_covers_all_named_slots():
    """Every named slot in bp_mod.SLOT_NAMES has a schema."""
    assert set(mob_dsl.MOB_SCHEMAS) == set(bp_mod.SLOT_NAMES)
    assert len(mob_dsl.MOB_SCHEMAS) == 79


def test_coverage_summary_arithmetic():
    s = mob_dsl.coverage_summary()
    assert s["total_slots"] == 79
    assert s["named_animation_fields"] + s["generic_animation_fields"] == 79
    # We documented at least Booma family (3) + Hildebear family (2) +
    # De Rol Le (1) + bosses (~16). Sanity bound.
    assert s["named_animation_fields"] >= 15


def test_every_schema_has_universal_groups():
    for slot, schema in mob_dsl.MOB_SCHEMAS.items():
        groups = {f.group for f in schema.fields}
        assert "Stats" in groups, f"slot 0x{slot:02X} missing Stats group"
        assert "Combat" in groups, f"slot 0x{slot:02X} missing Combat group"
        assert "Resists" in groups, f"slot 0x{slot:02X} missing Resists group"


def test_every_field_has_a_resolvable_binary_target():
    """Every FieldSpec must point at a real key in the schemas."""
    valid_groups = {
        "stats": {n for n, _, _ in bp_mod.STATS_SCHEMA},
        "attacks": {n for n, _, _ in bp_mod.ATTACKS_SCHEMA},
        "resists": {n for n, _, _ in bp_mod.RESISTS_SCHEMA},
        "animations": {n for n, _, _ in bp_mod.ANIMATIONS_SCHEMA},
    }
    for slot, schema in mob_dsl.MOB_SCHEMAS.items():
        for fs in schema.fields:
            assert fs.binary_group in valid_groups, (
                f"slot 0x{slot:02X} field {fs.label!r} has bad binary_group "
                f"{fs.binary_group!r}"
            )
            assert fs.binary_name in valid_groups[fs.binary_group], (
                f"slot 0x{slot:02X} field {fs.label!r} -> "
                f"{fs.binary_group}.{fs.binary_name} not in BattleParam schema"
            )


def test_dsl_field_labels_unique_per_mob():
    for slot, schema in mob_dsl.MOB_SCHEMAS.items():
        labels = [f.label for f in schema.fields]
        dups = {x for x in labels if labels.count(x) > 1}
        assert not dups, f"slot 0x{slot:02X} has duplicate DSL labels: {dups}"


def test_hildebear_schema_has_named_anim_fields():
    """The Tier-1 brief asks specifically for Hildebear semantic fields."""
    sch = mob_dsl.MOB_SCHEMAS[0x49]
    labels = {f.label for f in sch.fields}
    for needed in ("walk_speed", "tech_cast_chance_pct", "tech_cooldown_seconds"):
        assert needed in labels, f"Hildebear missing DSL field {needed!r}"


def test_de_rol_le_schema_has_phase_thresholds():
    sch = mob_dsl.MOB_SCHEMAS[0x0F]
    labels = {f.label for f in sch.fields}
    assert "armor_break_hp" in labels
    assert "mask_off_hp" in labels
    assert "mine_spawn_rate" in labels


# ---------------------------------------------------------------------------
# Mob / difficulty resolution
# ---------------------------------------------------------------------------
def test_resolve_mob_accepts_name():
    assert mob_dsl.resolve_mob("Booma") == 0x4B
    assert mob_dsl.resolve_mob("booma") == 0x4B
    assert mob_dsl.resolve_mob("BOOMA") == 0x4B
    assert mob_dsl.resolve_mob("Hildebear") == 0x49
    assert mob_dsl.resolve_mob("DeRolLe") == 0x0F


def test_resolve_mob_accepts_hex():
    assert mob_dsl.resolve_mob("0x4B") == 0x4B
    assert mob_dsl.resolve_mob("0x4b") == 0x4B
    assert mob_dsl.resolve_mob("0x49") == 0x49


def test_resolve_mob_accepts_int():
    assert mob_dsl.resolve_mob(0x4B) == 0x4B
    assert mob_dsl.resolve_mob(15) == 0x0F


def test_resolve_mob_rejects_unknown():
    with pytest.raises(ValueError):
        mob_dsl.resolve_mob("Nyarlathotep")
    with pytest.raises(ValueError):
        mob_dsl.resolve_mob("0xFF")
    with pytest.raises(ValueError):
        mob_dsl.resolve_mob(999)


def test_resolve_difficulty():
    assert mob_dsl.resolve_difficulty(None) == [0, 1, 2, 3]
    assert mob_dsl.resolve_difficulty("all") == [0, 1, 2, 3]
    assert mob_dsl.resolve_difficulty("Normal") == [0]
    assert mob_dsl.resolve_difficulty("hard") == [1]
    assert mob_dsl.resolve_difficulty("VeryHard") == [2]
    assert mob_dsl.resolve_difficulty("Ultimate") == [3]
    assert mob_dsl.resolve_difficulty(2) == [2]


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------
def test_dsl_to_binary_conversions():
    assert mob_dsl.dsl_to_binary("int", 5) == 5
    assert mob_dsl.dsl_to_binary("float", 1.25) == pytest.approx(1.25)
    # 4.5 seconds @ 30 Hz = 135 frames
    assert mob_dsl.dsl_to_binary("duration_seconds", 4.5) == pytest.approx(135.0)
    # 90 deg → quarter turn → 0x4000
    assert mob_dsl.dsl_to_binary("angle_bams", 90) == 0x4000
    # 360 deg wraps to 0
    assert mob_dsl.dsl_to_binary("angle_bams", 360) == 0
    # 100 percent = 1.0 float
    assert mob_dsl.dsl_to_binary("percent", 100) == pytest.approx(1.0)
    # percent_int passes through, clamped
    assert mob_dsl.dsl_to_binary("percent_int", 50) == 50
    assert mob_dsl.dsl_to_binary("percent_int", 150) == 100


def test_binary_to_dsl_round_trip():
    """Both directions agree on common values."""
    cases = [
        ("duration_seconds", 4.5),
        ("angle_bams", 90),
        ("angle_bams", 180),
        ("angle_bams", 270),
        ("percent", 25.0),
        ("percent_int", 75),
        ("int", 42),
        ("float", -3.14),
    ]
    for kind, dsl in cases:
        binary = mob_dsl.dsl_to_binary(kind, dsl)
        back = mob_dsl.binary_to_dsl(kind, binary)
        assert back == pytest.approx(dsl), f"{kind}: {dsl} → {binary} → {back}"


# ---------------------------------------------------------------------------
# Patch parsing
# ---------------------------------------------------------------------------
def test_parse_patch_minimal():
    p = mob_dsl.parse_patch({"mob": "Booma", "fields": {"engaged_speed": 2.0}})
    assert p.slot == 0x4B
    assert p.mob == "Booma"
    assert p.difficulties == [0, 1, 2, 3]
    assert p.fields == {"engaged_speed": 2.0}


def test_parse_patch_difficulty():
    p = mob_dsl.parse_patch({
        "mob": "Hildebear",
        "difficulty": "Ultimate",
        "fields": {"walk_speed": 1.5},
    })
    assert p.difficulties == [3]


def test_parse_patches_top_level_dict():
    payload = {"mobs": [
        {"mob": "Booma",     "fields": {"engaged_speed": 2.0}},
        {"mob": "Hildebear", "fields": {"walk_speed": 1.5}},
    ]}
    out = mob_dsl.parse_patches(payload)
    assert len(out) == 2
    assert out[0].slot == 0x4B
    assert out[1].slot == 0x49


def test_parse_patches_rejects_bad_input():
    with pytest.raises(ValueError):
        mob_dsl.parse_patches([{"mob": "Booma"}])  # missing fields
    with pytest.raises(ValueError):
        mob_dsl.parse_patches("not a list")


# ---------------------------------------------------------------------------
# Apply patch round-trips
# ---------------------------------------------------------------------------
def test_apply_patch_to_zero_buffer():
    """A patch on a zeroed file produces non-zero bytes only at the
    targeted struct cells."""
    raw = b"\x00" * bp_mod.FILE_SIZE
    bpf = bp_mod.parse(raw, variant="on")
    patch = mob_dsl.MobPatch(
        mob="Hildebear", slot=0x49, difficulties=[0],
        fields={"walk_speed": 1.5, "tech_cast_chance_pct": 50},
    )
    mob_dsl.apply_patch_to_file(bpf, patch)
    out = bp_mod.serialize(bpf)
    diffs = [(i, raw[i], out[i]) for i in range(len(raw)) if raw[i] != out[i]]
    assert diffs, "patch produced no byte changes"
    # Each float field is 4 bytes; we touched 2 fields in 1 difficulty
    # = at most 8 bytes of diff (could be less if a byte happened to
    # be 0 in the new value's bit pattern).
    assert len(diffs) <= 8, f"too many bytes changed: {len(diffs)}"


def test_apply_patch_isolates_to_targeted_slot():
    """A Booma patch must not touch any other mob's bytes."""
    raw = b"\x00" * bp_mod.FILE_SIZE
    bpf = bp_mod.parse(raw, variant="on")
    patch = mob_dsl.MobPatch(
        mob="Booma", slot=0x4B, difficulties=[0],
        fields={"engaged_speed": 1.5},
    )
    mob_dsl.apply_patch_to_file(bpf, patch)
    # Confirm Hildebear's slot in the same difficulty is untouched.
    hb = bpf.difficulties[0].entries[0x49]
    for fs_name, _fmt, _c in bp_mod.ANIMATIONS_SCHEMA:
        assert hb.animations[fs_name] == 0, (
            f"Hildebear.animations.{fs_name} contaminated"
        )


def test_apply_patches_compose_in_order():
    """Later patches override earlier ones on the same field."""
    raw = b"\x00" * bp_mod.FILE_SIZE
    bpf = bp_mod.parse(raw, variant="on")
    p1 = mob_dsl.MobPatch(mob="Booma", slot=0x4B, difficulties=[0],
                          fields={"engaged_speed": 1.0})
    p2 = mob_dsl.MobPatch(mob="Booma", slot=0x4B, difficulties=[0],
                          fields={"engaged_speed": 3.0})
    out = mob_dsl.apply_patches(bpf, [p1, p2])
    booma = out.difficulties[0].entries[0x4B]
    assert booma.animations["fparam3"] == pytest.approx(3.0)


def test_apply_patches_does_not_mutate_input():
    raw = b"\x00" * bp_mod.FILE_SIZE
    bpf = bp_mod.parse(raw, variant="on")
    patch = mob_dsl.MobPatch(
        mob="Booma", slot=0x4B, difficulties=[0],
        fields={"engaged_speed": 5.0},
    )
    out = mob_dsl.apply_patches(bpf, [patch])
    # Original should still see fparam3 = 0
    assert bpf.difficulties[0].entries[0x4B].animations["fparam3"] == 0
    # Output sees 5.0
    assert out.difficulties[0].entries[0x4B].animations["fparam3"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Hildebear preset → walk_speed binary mapping verification
# ---------------------------------------------------------------------------
def test_hildebear_preset_maps_walk_speed_to_fparam3():
    """The brief's verification: 'compile a Hildebear preset → verify
    walk_speed maps to the right BattleParam slot'."""
    raw = b"\x00" * bp_mod.FILE_SIZE
    bpf = bp_mod.parse(raw, variant="on")
    preset = mob_dsl.load_preset("hildebear_caster")
    patches = mob_dsl.parse_patches(preset)
    out = mob_dsl.apply_patches(bpf, patches)
    hb = out.difficulties[0].entries[0x49]
    # hildebear_caster sets walk_speed: 1.0 → fparam3 := 1.0
    assert hb.animations["fparam3"] == pytest.approx(1.0)
    # tech_cast_chance_pct: 65 → fparam5 := 0.65
    assert hb.animations["fparam5"] == pytest.approx(0.65)
    # tech_cooldown_seconds: 1.5 → fparam6 := 45 frames
    assert hb.animations["fparam6"] == pytest.approx(45.0)


# ---------------------------------------------------------------------------
# Encode overlay (BattleParam → DSL)
# ---------------------------------------------------------------------------
def test_encode_overlay_empty_when_no_diff():
    raw = b"\x00" * bp_mod.FILE_SIZE
    bpf = bp_mod.parse(raw, variant="on")
    out = mob_dsl.encode_overlay(bpf, bpf)
    assert out == []


def test_encode_overlay_picks_up_single_field_change():
    raw = b"\x00" * bp_mod.FILE_SIZE
    base = bp_mod.parse(raw, variant="on")
    patched = mob_dsl.apply_patches(
        base,
        [mob_dsl.MobPatch(mob="Booma", slot=0x4B, difficulties=[1],
                          fields={"engaged_speed": 2.5})],
    )
    overlay = mob_dsl.encode_overlay(patched, base)
    assert len(overlay) == 1
    o = overlay[0]
    assert o["mob"] == "Booma"
    assert o["slot"] == 0x4B
    assert o["difficulty"] == "Hard"
    assert o["fields"] == {"engaged_speed": pytest.approx(2.5)}


def test_round_trip_through_overlay():
    """parse → patch → encode → re-apply on stock = same edited file."""
    raw = b"\x00" * bp_mod.FILE_SIZE
    base = bp_mod.parse(raw, variant="on")
    patches_in = mob_dsl.parse_patches([
        {"mob": "Booma", "difficulty": "Normal",
         "fields": {"engaged_speed": 2.0, "atp_max": 175}},
        {"mob": "Hildebear", "difficulty": "Ultimate",
         "fields": {"walk_speed": 1.4, "tech_cooldown_seconds": 3.0}},
    ])
    edited1 = mob_dsl.apply_patches(base, patches_in)
    # Encode overlay from edited1 vs base.
    overlay = mob_dsl.encode_overlay(edited1, base)
    # Re-parse overlay as patches, re-apply.
    patches_out = mob_dsl.parse_patches({"mobs": overlay})
    edited2 = mob_dsl.apply_patches(base, patches_out)
    # Serialize both — must be byte-identical.
    out1 = bp_mod.serialize(edited1)
    out2 = bp_mod.serialize(edited2)
    assert out1 == out2


# ---------------------------------------------------------------------------
# Real-fixture round-trip (only if Booma.Server fixtures are available)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_FIXTURES, reason="Booma.Server fixtures not present")
def test_real_fixture_unmodified_compile_is_identity():
    """Compiling an empty patch list against a real .dat should leave
    the bytes byte-exact (no side effects from the deep copy)."""
    raw = (FIXTURES / "BattleParamEntry_on.dat").read_bytes()
    bpf = bp_mod.parse(raw, variant="on")
    out = mob_dsl.compile_to_battle_param(bpf, [])
    assert bp_mod.serialize(out) == raw


@pytest.mark.skipif(not HAS_FIXTURES, reason="Booma.Server fixtures not present")
def test_real_fixture_single_field_diff_is_targeted():
    """Single-field patch on real fixture changes only its 4 bytes (or
    fewer if the new value's bit pattern coincides with the old)."""
    raw = (FIXTURES / "BattleParamEntry_on.dat").read_bytes()
    bpf = bp_mod.parse(raw, variant="on")
    patch = mob_dsl.MobPatch(
        mob="Booma", slot=0x4B, difficulties=[0],
        fields={"engaged_speed": -123456.789},
    )
    out_bpf = mob_dsl.apply_patches(bpf, [patch])
    out = bp_mod.serialize(out_bpf)
    diffs = [(i, raw[i], out[i]) for i in range(len(raw)) if raw[i] != out[i]]
    # Must lie in a contiguous 4-byte window.
    assert 1 <= len(diffs) <= 4, f"expected 1..4 byte diffs, got {len(diffs)}"
    if diffs:
        first, last = diffs[0][0], diffs[-1][0]
        assert last - first < 4, (
            f"diff spans {last - first}b — exceeded float window"
        )


@pytest.mark.skipif(not HAS_FIXTURES, reason="Booma.Server fixtures not present")
def test_real_fixture_compile_then_encode_overlay_recovers_input():
    """Compile a patch, encode overlay, ensure the recovered patch list
    has the same effect when re-applied."""
    raw = (FIXTURES / "BattleParamEntry_on.dat").read_bytes()
    bpf = bp_mod.parse(raw, variant="on")
    patches_in = mob_dsl.parse_patches([
        {"mob": "Booma",     "fields": {"engaged_speed": 1.7}},
        {"mob": "Hildebear", "difficulty": "Ultimate",
         "fields": {"walk_speed": 1.4, "tech_cast_chance_pct": 30}},
        {"mob": "DeRolLe",   "difficulty": "all",
         "fields": {"armor_break_hp": 8000, "mine_spawn_rate": 4.0}},
    ])
    edited = mob_dsl.apply_patches(bpf, patches_in)
    overlay = mob_dsl.encode_overlay(edited, bpf)
    # Re-apply overlay on stock; resulting bytes must match.
    re_edited = mob_dsl.apply_patches(bpf, mob_dsl.parse_patches({"mobs": overlay}))
    assert bp_mod.serialize(edited) == bp_mod.serialize(re_edited)


# ---------------------------------------------------------------------------
# Presets shipped
# ---------------------------------------------------------------------------
def test_presets_directory_has_at_least_5_presets():
    presets = mob_dsl.list_presets()
    names = {p["name"] for p in presets}
    # Tier-1 brief asks for at least booma_aggressive, booma_passive,
    # hildebear_caster, de_rol_le_unfair, boss_speedrun.
    assert "booma_aggressive" in names
    assert "booma_passive" in names
    assert "hildebear_caster" in names
    assert "de_rol_le_unfair" in names
    assert "boss_speedrun" in names
    assert len(names) >= 5


def test_every_preset_compiles_against_stock():
    """Every shipped preset must parse + compile cleanly on a zero file."""
    raw = b"\x00" * bp_mod.FILE_SIZE
    bpf = bp_mod.parse(raw, variant="on")
    for p in mob_dsl.list_presets():
        preset = mob_dsl.load_preset(p["name"])
        try:
            patches = mob_dsl.parse_patches(preset)
            out = mob_dsl.apply_patches(bpf, patches)
        except Exception as e:
            pytest.fail(f"preset {p['name']} failed: {e}")
        # Verify serialize works on the result (catches bad packs)
        bp_mod.serialize(out)


def test_load_preset_path_traversal_blocked():
    with pytest.raises(ValueError):
        mob_dsl.load_preset("../etc/passwd")
    with pytest.raises(ValueError):
        mob_dsl.load_preset("subdir/inner")
    with pytest.raises(FileNotFoundError):
        mob_dsl.load_preset("does_not_exist")


# ---------------------------------------------------------------------------
# Schema JSON serialization (the wire format)
# ---------------------------------------------------------------------------
def test_all_schemas_json_is_valid():
    payload = mob_dsl.all_schemas_json()
    # Must round-trip through json.dumps without error
    text = json.dumps(payload)
    assert "schemas" in payload
    assert "coverage" in payload
    assert len(payload["schemas"]) == 79
    parsed = json.loads(text)
    assert len(parsed["schemas"]) == 79


def test_schema_json_for_one_mob():
    j = mob_dsl.schema_json("Hildebear")
    assert j["slot"] == 0x49
    assert j["name"] == "Hildebear"
    assert any(f["label"] == "tech_cast_chance_pct" for f in j["fields"])


# ---------------------------------------------------------------------------
# YAML-lite parser (the DSL example in the brief)
# ---------------------------------------------------------------------------
YAML_LITE_SAMPLE = """
mob: hildebear
difficulty: Normal
movement:
  walk_speed: 0.8
  walk_anim_speed: 1.2
combat:
  atp_min: 60
  atp_max: 110
  swing_arc_deg: 75
  tech_cast_chance_pct: 12
  tech_cooldown_seconds: 4.5
resists:
  ice: 30
  fire: 0
"""


def test_yaml_lite_parse_brief_sample():
    parsed = mob_dsl.parse_yaml_lite(YAML_LITE_SAMPLE)
    assert parsed["mob"] == "hildebear"
    assert parsed["difficulty"] == "Normal"
    f = parsed["fields"]
    assert f["walk_speed"] == 0.8
    assert f["atp_min"] == 60
    assert f["tech_cooldown_seconds"] == 4.5
    assert f["ice"] == 30


def test_yaml_lite_parse_compiles_cleanly():
    """The parsed YAML-lite output must be a valid MobPatch payload."""
    parsed = mob_dsl.parse_yaml_lite(YAML_LITE_SAMPLE)
    patch = mob_dsl.parse_patch(parsed)
    assert patch.slot == 0x49
    assert patch.difficulties == [0]
    assert "walk_speed" in patch.fields


def test_yaml_lite_round_trip():
    """YAML → parse → emit → parse → same fields. Field ordering may
    differ; we compare as dicts."""
    parsed1 = mob_dsl.parse_yaml_lite(YAML_LITE_SAMPLE)
    text2 = mob_dsl.emit_yaml_lite(parsed1)
    parsed2 = mob_dsl.parse_yaml_lite(text2)
    assert parsed1["mob"] == parsed2["mob"]
    assert parsed1["difficulty"] == parsed2["difficulty"]
    assert parsed1["fields"] == parsed2["fields"]


# ---------------------------------------------------------------------------
# Tooltips and ranges populated for documented fields
# ---------------------------------------------------------------------------
def test_documented_fields_have_tooltips():
    """Spot-check that the fields featured in presets have tooltips."""
    sch_hb = mob_dsl.MOB_SCHEMAS[0x49]
    walk = sch_hb.field_by_label("walk_speed")
    assert walk and walk.tooltip
    cast = sch_hb.field_by_label("tech_cast_chance_pct")
    assert cast and cast.tooltip
    sch_drl = mob_dsl.MOB_SCHEMAS[0x0F]
    armor = sch_drl.field_by_label("armor_break_hp")
    assert armor and armor.tooltip
