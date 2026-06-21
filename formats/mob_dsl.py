"""Mob AI DSL Tier 1 — semantic authoring layer over BattleParamEntry.

This module wraps :mod:`formats.battle_param` with a higher-level
authoring API:

  * **Per-slot field schemas.** Each named mob slot maps one or more of
    its raw struct fields (``stats.atp``, ``animations.fparam3``, …) to
    a *named, kind-aware* DSL field (``atp_min``, ``engaged_speed``,
    …). The schema also carries semantic groups (Movement / Combat /
    Resists / AI Behavior), kind hints (durations in seconds, angles
    in degrees), default values, and short tooltips.
  * **Diff overlays as the wire format.** A "patch" is a small dict
    keyed by mob name (or numeric slot) holding only the *changed*
    fields. Patches compose in priority order onto a stock
    BattleParamFile.
  * **Round-trippable.** ``encode_overlay`` extracts the same overlay
    from a parsed BattleParamFile vs. a stock baseline; ``apply_overlay``
    bakes one back in. Tier-1 round-trip means *named* fields survive
    JSON ↔ Battle-Param ↔ JSON cycles byte-for-byte.

Field source-of-truth: newserv ``notes/movement-data.txt`` (per
the Tier 1 brief; a copy lives in the BattleParam research note for
this editor). Where a slot has no documented semantics we fall back to
a *generic* schema (raw fparam/iparam labels, stats/attack/resist still
present). That way the DSL still works for every slot — the un-named
fields just appear as their raw struct names.

JSON shape on the wire (``/api/mob_dsl/compile`` request body)::

    {
      "mobs": [
        {
          "mob": "hildebear",          # slot name OR slot id (e.g. "0x49")
          "difficulty": "Normal",       # optional, default "all"
          "fields": {
            "walk_speed": 0.8,
            "run_speed":  1.2,
            "swing_arc_deg": 75,
            "tech_cast_chance_pct": 12,
            "tech_cooldown_seconds": 4.5,
            "ice": 30,
          }
        }
      ]
    }

Compiled output is a (variant -> patched BattleParamFile JSON) map.
"""
from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from formats import battle_param as bp_mod


# ---------------------------------------------------------------------------
# Field kinds and unit conversions
# ---------------------------------------------------------------------------
# A "kind" decides:
#   - the editor widget the UI uses (slider vs number input)
#   - the unit conversion between DSL and the raw battle-param byte
#
# Conversions happen at the FieldSpec boundary; the DSL layer always
# carries the *human* unit, the raw struct always carries the *binary*
# unit.

# A kind is one of:
#   "int"               — pass-through signed int
#   "uint"              — pass-through unsigned int
#   "float"             — pass-through float32
#   "duration_seconds"  — float DSL value; binary is float32 frames @ 30 Hz
#   "duration_frames"   — int frames pass-through
#   "angle_bams"        — int DSL value (degrees 0..360); binary is BAMS uint32
#   "percent"           — float DSL value (0..100); binary is float32 0..1
#                         (used when stock files store a probability)
#   "percent_int"       — int DSL value (0..100); binary stores 0..100 directly
#                         (some HP-threshold fields work this way)
#
# We deliberately avoid coupling the conversion direction to the
# binary type; the FieldSpec.kind is enough.
GAME_TICK_HZ = 30.0


def dsl_to_binary(kind: str, dsl_value):
    """Convert a DSL-space value to the underlying battle-param value."""
    if kind == "duration_seconds":
        return float(dsl_value) * GAME_TICK_HZ  # seconds → frames
    if kind == "angle_bams":
        # 0xFFFF is the BAMS limit per BattleParam (uint16/uint32 both
        # tolerated); 360deg → 0x10000. Round to nearest int.
        return int(round((float(dsl_value) % 360.0) * (0x10000 / 360.0))) & 0xFFFFFFFF
    if kind == "percent":
        return float(dsl_value) / 100.0
    if kind == "percent_int":
        # Clamp to 0..100; the binary just stores the same int.
        v = int(round(float(dsl_value)))
        return max(0, min(100, v))
    if kind == "duration_frames":
        return int(round(float(dsl_value)))
    if kind in ("int", "uint"):
        return int(round(float(dsl_value)))
    if kind == "float":
        return float(dsl_value)
    raise ValueError(f"unknown kind {kind!r}")


def binary_to_dsl(kind: str, binary_value):
    """Convert a battle-param raw value to its DSL-space presentation."""
    if binary_value is None:
        return None
    if kind == "duration_seconds":
        try:
            return round(float(binary_value) / GAME_TICK_HZ, 4)
        except (TypeError, ValueError):
            return None
    if kind == "angle_bams":
        v = int(binary_value) & 0xFFFFFFFF
        return round((v / 0x10000) * 360.0, 3) % 360.0
    if kind == "percent":
        try:
            return round(float(binary_value) * 100.0, 3)
        except (TypeError, ValueError):
            return None
    if kind == "percent_int":
        try:
            return int(binary_value)
        except (TypeError, ValueError):
            return None
    if kind == "duration_frames":
        return int(binary_value)
    if kind in ("int", "uint"):
        return int(binary_value)
    if kind == "float":
        try:
            return float(binary_value)
        except (TypeError, ValueError):
            return None
    raise ValueError(f"unknown kind {kind!r}")


# ---------------------------------------------------------------------------
# FieldSpec / MobSchema dataclasses
# ---------------------------------------------------------------------------
@dataclass
class FieldSpec:
    """One named DSL field for a single mob slot.

    Mapping from DSL → battle-param: the (group, binary_name) pair
    locates the raw cell in ``BattleParamEntry``. The kind handles the
    unit conversion. The label is what the UI shows; the DSL key (used
    in YAML/JSON) is :attr:`label` itself, not :attr:`binary_name`.
    """
    binary_name: str  # e.g. "fparam3" — key in animations dict
    label: str        # e.g. "engaged_speed" — DSL key
    group: str        # "Movement" / "Combat" / "Resists" / "AI Behavior" / "Stats"
    kind: str         # one of the kinds above
    binary_group: str = "animations"  # "stats" / "attacks" / "resists" / "animations"
    range: Optional[Tuple[float, float]] = None
    default: Optional[float] = None
    tooltip: str = ""

    def to_json(self) -> Dict:
        return {
            "binary_name": self.binary_name,
            "binary_group": self.binary_group,
            "label": self.label,
            "group": self.group,
            "kind": self.kind,
            "range": list(self.range) if self.range else None,
            "default": self.default,
            "tooltip": self.tooltip,
        }


@dataclass
class MobSchema:
    """All named DSL fields available for one mob slot."""
    slot: int
    name: str
    fields: List[FieldSpec] = field(default_factory=list)
    canonical_iparam_meanings: Dict[int, str] = field(default_factory=dict)
    notes: str = ""

    def to_json(self) -> Dict:
        return {
            "slot": self.slot,
            "slot_hex": f"0x{self.slot:02X}",
            "name": self.name,
            "fields": [f.to_json() for f in self.fields],
            "canonical_iparam_meanings": {
                str(k): v for k, v in self.canonical_iparam_meanings.items()
            },
            "notes": self.notes,
        }

    def field_by_label(self, label: str) -> Optional[FieldSpec]:
        for f in self.fields:
            if f.label == label:
                return f
        return None


# ---------------------------------------------------------------------------
# Generic stat fields shared by every "AI-tunable" slot
# ---------------------------------------------------------------------------
# Every AI-tunable mob has the same Stats/Attacks/Resists field set.
# Authoring a mob means picking which named-Animations fields apply
# *plus* the universal stats/attacks/resists labels. We build the
# universal set once and prepend it to every schema.

def _universal_stats_fields() -> List[FieldSpec]:
    """Stats fields exposed by every AI-tunable mob.

    These map to ``BattleParamEntry.stats`` (the 0x24-byte ``BPStatsEntry``
    struct). They are the mob's raw combat numbers — what the engine plugs
    into the damage / hit / evade formulas every time it fights a player.
    """
    return [
        FieldSpec("atp", "atp",       "Stats", "int", "stats", (-32768, 32767),
                  tooltip="Attack Power. The base of every melee/ranged hit the mob lands; "
                          "higher ATP = more damage per swing. Ultimate-difficulty enemies "
                          "carry far higher ATP than Normal. e.g. a basic Booma sits near 90; "
                          "set 300+ to make it hit like a mini-boss."),
        FieldSpec("mst", "mst",       "Stats", "int", "stats", (-32768, 32767),
                  tooltip="Mind / Magic Power (MST). Scales the damage of techniques the mob "
                          "casts (Foie/Barta/Zonde, etc.). Only matters for caster mobs; melee "
                          "mobs can leave it low. e.g. 200 for a strong tech-caster."),
        FieldSpec("evp", "evp",       "Stats", "int", "stats", (-32768, 32767),
                  tooltip="Evasion (EVP). The mob's chance to dodge incoming attacks — higher "
                          "EVP means players miss it more often. Bump it to make a mob feel "
                          "slippery/annoying. e.g. 120 base, 400 to make it hard to hit."),
        FieldSpec("hp",  "hp",        "Stats", "int", "stats", (-32768, 32767),
                  tooltip="Max HP (signed 16-bit). How much damage the mob can soak before it "
                          "dies — the single biggest lever on how long a fight lasts. e.g. a "
                          "Booma ~210; set 2000 for a beefy elite. NOTE: bosses usually ignore "
                          "this and read HP from their iparam phase fields instead."),
        FieldSpec("dfp", "dfp",       "Stats", "int", "stats", (-32768, 32767),
                  tooltip="Defense (DFP). Subtracted from incoming damage, so higher DFP makes "
                          "every player hit chip less HP off the mob. e.g. 50 base; 300 makes a "
                          "tanky enemy that shrugs off weak weapons."),
        FieldSpec("ata", "ata",       "Stats", "int", "stats", (-32768, 32767),
                  tooltip="Accuracy (ATA). The mob's chance to actually CONNECT with the player "
                          "(vs being dodged). Low ATA = the mob whiffs a lot; high ATA = it "
                          "reliably lands hits. e.g. 60 base; 200 makes attacks almost always "
                          "connect."),
        FieldSpec("lck", "lck",       "Stats", "int", "stats", (-32768, 32767),
                  tooltip="Luck (LCK). Minor stat feeding the engine's critical-hit / variance "
                          "rolls. Most stock mobs leave this at 0; small effect. e.g. 0–20."),
        FieldSpec("xp",  "xp_drop",   "Stats", "int", "stats", (-32768, 32767),
                  tooltip="Experience dropped when this mob is killed. Pure reward tuning — raise "
                          "it to make a mob worth grinding, lower it to discourage farming. "
                          "e.g. a Booma gives ~40; set 500 for a high-value target."),
    ]


def _universal_combat_fields() -> List[FieldSpec]:
    """Attack-data fields exposed by every AI-tunable mob.

    These map to ``BattleParamEntry.attacks`` (newserv ``AttackData``).
    They describe the mob's *attack swing* itself — how hard a rolled hit
    lands and the physical reach/arc the engine checks for a connect.
    """
    return [
        FieldSpec("min_atp", "atp_min",      "Combat", "int",         "attacks", (0, 32767),
                  tooltip="Minimum attack power of a rolled swing. Each hit rolls a damage value "
                          "between atp_min and atp_max, so this is the FLOOR of the mob's hit. "
                          "Raising it removes the weak hits. e.g. 80."),
        FieldSpec("max_atp", "atp_max",      "Combat", "int",         "attacks", (0, 32767),
                  tooltip="Maximum attack power of a rolled swing — the CEILING of a hit's damage. "
                          "Widen the gap from atp_min for swingy/unpredictable damage; set equal to "
                          "atp_min for consistent hits. e.g. 175."),
        FieldSpec("min_ata", "ata_min",      "Combat", "int",         "attacks", (0, 32767),
                  tooltip="Minimum rolled accuracy of a swing — the floor of the to-hit roll for "
                          "this specific attack (separate from the mob's base ATA stat). e.g. 50."),
        FieldSpec("max_ata", "ata_max",      "Combat", "int",         "attacks", (0, 32767),
                  tooltip="Maximum rolled accuracy of a swing — the ceiling of the to-hit roll. "
                          "Higher = the attack lands more reliably. e.g. 110."),
        FieldSpec("distance_x", "reach_x",   "Combat", "float",       "attacks", (0, 1000),
                  tooltip="Horizontal reach of the attack, in world units. How far in FRONT of the "
                          "mob the swing/hitbox extends — bigger reach lets it hit you from farther "
                          "away (a melee enemy that can 'reach through' a step back). e.g. 30.0 for "
                          "a normal swing, 120.0 for a long lunge."),
        FieldSpec("angle",   "swing_arc_deg", "Combat", "angle_bams", "attacks", (0, 360),
                  tooltip="Swing arc / cone width of the attack, in degrees. How WIDE the attack "
                          "sweeps to the sides — a wide arc can clip players who aren't directly in "
                          "front. e.g. 45 deg = a narrow thrust, 180 deg = a broad cleave. (Stored "
                          "internally as BAMS where 0x10000 = 360 deg; the editor handles the "
                          "conversion.)"),
        FieldSpec("distance_y", "reach_y",   "Combat", "float",       "attacks", (0, 1000),
                  tooltip="Vertical reach of the attack, in world units. How far UP/DOWN the hitbox "
                          "extends — matters for tall mobs or flyers hitting grounded players (and "
                          "vice-versa). e.g. 20.0."),
    ]


def _universal_resist_fields() -> List[FieldSpec]:
    """Resist fields exposed by every AI-tunable mob.

    These map to ``BattleParamEntry.resists`` (newserv ``ResistData``).
    Elemental resists are percentages: how much of an element's damage the
    mob shrugs off. 0 = takes full elemental damage; 100 = immune.
    """
    return [
        FieldSpec("evp_bonus", "evp_bonus",  "Resists", "int",  "resists", (-32768, 32767),
                  tooltip="Bonus evasion added ON TOP of the base EVP stat. Net effect is the same "
                          "as raising EVP — the mob dodges more — but this is a separate additive "
                          "lever (use it to buff dodge without touching the base stat). e.g. +50."),
        FieldSpec("efr",  "fire",            "Resists", "uint", "resists", (0, 65535),
                  tooltip="Fire resistance %. How much Foie / fire-element weapon damage the mob "
                          "ignores. 0 = full fire damage, 100 = immune to fire. e.g. 30 = takes "
                          "30% less fire damage; set 100 on a fire-based enemy."),
        FieldSpec("eic",  "ice",             "Resists", "uint", "resists", (0, 65535),
                  tooltip="Ice resistance %. How much Barta / ice-element damage the mob ignores. "
                          "0 = full ice damage, 100 = immune. e.g. 30."),
        FieldSpec("eth",  "thunder",         "Resists", "uint", "resists", (0, 65535),
                  tooltip="Thunder resistance %. How much Zonde / lightning-element damage the mob "
                          "ignores. 0 = full damage, 100 = immune. e.g. 30."),
        FieldSpec("elt",  "light",           "Resists", "uint", "resists", (0, 65535),
                  tooltip="Light resistance %. How much Grants / light-element damage the mob "
                          "ignores. Dark enemies (Dark Falz line) are usually weak to light, so "
                          "keep this LOW on them. 0 = full damage, 100 = immune."),
        FieldSpec("edk",  "dark",            "Resists", "uint", "resists", (0, 65535),
                  tooltip="Dark resistance %. How much Megid / dark-element damage the mob ignores. "
                          "0 = full damage, 100 = immune. Note Megid can also instant-kill via a "
                          "separate roll. e.g. 30."),
        FieldSpec("dfp_bonus", "dfp_bonus",  "Resists", "int",  "resists", (-32768, 32767),
                  tooltip="Bonus defense added ON TOP of the base DFP stat. Same effect as raising "
                          "DFP — reduces all incoming physical damage — but as a separate additive "
                          "lever. e.g. +100 for an armored variant."),
    ]


# ---------------------------------------------------------------------------
# Family-specific Animations field overlays
# ---------------------------------------------------------------------------
# These are the slots whose iparam/fparam meanings are documented in
# notes/movement-data.txt or the project's RESEARCH note. Every other
# slot gets the generic fparam1..iparam6 fall-through.

def _booma_animations() -> List[FieldSpec]:
    # The Booma family stores movement + behaviour in the 12-slot
    # ``animations`` (MovementData) array. Speeds are world-units/tick;
    # ~1.0 is a stock walk, higher = faster. anim_speed is purely cosmetic
    # (how fast the legs cycle) — decouple it from move speed for a
    # comical moonwalk, or keep them matched so motion looks natural.
    return [
        FieldSpec("fparam1", "idle_speed",        "Movement",    "float", "animations", (0, 10),
                  tooltip="How fast the mob moves while it has NOT noticed you — wandering near "
                          "spawn or returning home. Low = it loiters; high = it patrols quickly. "
                          "e.g. 0.5 stock, 1.5 for a restless patroller."),
        FieldSpec("fparam2", "idle_anim_speed",   "Movement",    "float", "animations", (0, 10),
                  tooltip="Playback rate of the walk ANIMATION while idle (cosmetic only — does "
                          "not change actual move speed). Match it to idle_speed so the feet don't "
                          "slide. e.g. 1.0."),
        FieldSpec("fparam3", "engaged_speed",     "Movement",    "float", "animations", (0, 10),
                  tooltip="How fast the mob CHASES once it has aggroed onto a player — the lever "
                          "that decides whether you can out-run it. This is the main 'aggressive vs "
                          "passive' knob. e.g. 1.0 stock; 2.5 makes a Booma that runs you down."),
        FieldSpec("fparam4", "engaged_anim_speed","Movement",    "float", "animations", (0, 10),
                  tooltip="Playback rate of the run ANIMATION while chasing (cosmetic). Match it to "
                          "engaged_speed to avoid foot-sliding. e.g. 1.2."),
        FieldSpec("fparam5", "poison_cloud_dmg",  "AI Behavior", "float", "animations", (0, 1000),
                  tooltip="Damage per tick of the poison cloud the Merillia (Ep2 reskin of this "
                          "slot) leaves behind. Only meaningful on the poison-cloud variant; 0 on a "
                          "plain Booma. e.g. 25."),
        FieldSpec("fparam6", "flee_speed",        "Movement",    "float", "animations", (0, 10),
                  tooltip="How fast the mob RETREATS when it decides to back off (e.g. after taking "
                          "damage, or below its low-HP threshold). High = it kites away from you. "
                          "e.g. 1.8."),
        FieldSpec("iparam1", "low_hp_threshold",  "AI Behavior", "percent_int", "animations", (0, 100),
                  tooltip="HP percentage at which the mob switches into its 'low HP' behaviour "
                          "(typically fleeing at flee_speed instead of pressing the attack). "
                          "e.g. 25 = it starts running once it drops below 25% HP; 0 = never flee."),
    ]


def _hildebear_animations() -> List[FieldSpec]:
    # Hildebear/Hildeblue are melee-or-caster hybrids: they punch up close
    # but can also lob a technique. These params expose that decision.
    return [
        FieldSpec("fparam1", "punch_attack_speed", "AI Behavior", "float", "animations", (0, 10),
                  tooltip="Speed multiplier on the melee punch animation/attack. Higher = it swings "
                          "faster and more often, so DPS pressure goes up. e.g. 1.0 stock, 2.0 for a "
                          "flurrying brawler."),
        FieldSpec("fparam2", "tech_range",         "AI Behavior", "float", "animations", (0, 1000),
                  tooltip="Distance (world units) at which the mob will choose to CAST a technique "
                          "instead of closing for a punch. Big range = it zaps you from across the "
                          "room; small range = it prefers to melee. e.g. 200.0."),
        FieldSpec("fparam3", "walk_speed",         "Movement",    "float", "animations", (0, 10),
                  tooltip="Movement speed when approaching/repositioning (does NOT scale the walk "
                          "animation — that is walk_anim_speed). The main 'how fast does it come at "
                          "you' lever for this family. e.g. 1.0 stock, 1.6 aggressive."),
        FieldSpec("fparam4", "walk_anim_speed",    "Movement",    "float", "animations", (0, 10),
                  tooltip="Playback rate of the walk animation (cosmetic). Match it to walk_speed so "
                          "the legs don't slide. e.g. 1.0."),
        FieldSpec("fparam5", "tech_cast_chance_pct","AI Behavior","percent",  "animations", (0, 100),
                  tooltip="Probability (0–100%) that, when it could act, the mob casts a technique "
                          "rather than melee-punching. 0 = pure brawler, 100 = pure caster. "
                          "e.g. 12 = occasional spell; 65 = mostly a caster."),
        FieldSpec("fparam6", "tech_cooldown_seconds","AI Behavior","duration_seconds","animations", (0, 30),
                  tooltip="Minimum time between technique casts, in SECONDS (the editor converts to "
                          "30 fps frames internally). Lower = it spams spells; higher = rare casts. "
                          "e.g. 4.5 s stock, 1.5 s for a caster build."),
        FieldSpec("iparam1", "tech_select_seed",   "AI Behavior", "uint",  "animations", (0, 0xFFFFFFFF),
                  tooltip="Seed feeding the RNG that picks WHICH technique it casts (the Foie / "
                          "Barta / Zonde mix). Changing it reshuffles the element pattern; most "
                          "users can leave this alone. e.g. 0."),
    ]


def _de_rol_le_animations() -> List[FieldSpec]:
    # De Rol Le (Ep1 Mines boss) is multi-phase. Its iparam slots hold the
    # HP gates that drive phase transitions — these OVERRIDE the plain
    # stats.hp value for the boss.
    return [
        FieldSpec("fparam1", "swipe_damage",          "Combat",      "float", "animations", (0, 1000),
                  tooltip="Damage of the boss's phase-2 swipe attack. Raise to make the swipe a "
                          "real threat. e.g. 150."),
        FieldSpec("fparam2", "unused2",               "AI Behavior", "float", "animations",
                  tooltip="Unused slot for this boss (no observed in-game effect). Leave at stock."),
        FieldSpec("fparam3", "mine_damage",           "Combat",      "float", "animations", (0, 1000),
                  tooltip="Damage of an exploding mine the boss spawns. Pairs with mine_spawn_rate "
                          "to set how dangerous the mine field is. e.g. 120."),
        FieldSpec("fparam4", "x_position_jitter",     "AI Behavior", "float", "animations", (0, 1000),
                  tooltip="How far the boss randomly shifts along the X axis when it repositions "
                          "(gated by x_position_jitter_pct). Bigger = more erratic side-to-side "
                          "movement. e.g. 50.0."),
        FieldSpec("fparam5", "x_position_jitter_pct", "AI Behavior", "percent","animations", (0, 100),
                  tooltip="Chance per tick (0–100%) that the X-jitter above actually fires. Higher "
                          "= the boss juke-slides more often. e.g. 20."),
        FieldSpec("fparam6", "mine_spawn_rate",       "AI Behavior", "duration_seconds", "animations", (0, 30),
                  tooltip="Seconds between mine spawns (editor converts to frames). LOWER = mines "
                          "appear faster and the arena gets busier. e.g. 6.0 s stock, 2.0 s for a "
                          "punishing mine field."),
        FieldSpec("iparam1", "total_hp",              "AI Behavior", "int",  "animations", (0, 0x7FFFFFFF),
                  tooltip="The boss's TOTAL HP. This OVERRIDES the generic stats.hp for De Rol Le "
                          "(bosses read their HP from here). The biggest lever on fight length. "
                          "e.g. 12000."),
        FieldSpec("iparam2", "armor_break_hp",        "AI Behavior", "int",  "animations", (0, 0x7FFFFFFF),
                  tooltip="HP value at which the boss transitions into its armor-break phase. Set it "
                          "relative to total_hp (e.g. ~60% of total) to control when the fight "
                          "escalates. e.g. 8000."),
        FieldSpec("iparam3", "mask_off_hp",           "AI Behavior", "int",  "animations", (0, 0x7FFFFFFF),
                  tooltip="HP value at which the boss's mask comes off and the skull becomes "
                          "targetable (the real damage window opens). e.g. 3000."),
        FieldSpec("iparam4", "ult_constant_a",        "AI Behavior", "uint", "animations", (0, 0xFFFFFFFF),
                  tooltip="Ultimate-difficulty tuning constant A (timing/behaviour magic number; "
                          "default 180 on other slots). Advanced — leave at stock unless you know "
                          "the boss's frame timings."),
        FieldSpec("iparam5", "ult_constant_b",        "AI Behavior", "uint", "animations", (0, 0xFFFFFFFF),
                  tooltip="Ultimate-difficulty tuning constant B (default 120 elsewhere). Advanced — "
                          "leave at stock unless you are tuning the boss's frame timings."),
    ]


def _generic_animations() -> List[FieldSpec]:
    """Fallback set: just expose the raw fparam/iparam fields.

    Used for slots whose 12 ``animations``/MovementData params have no
    confirmed meaning yet. The values are real and DO affect the mob —
    we just can't put a friendly name on each one. By convention across
    documented slots: the low fparams tend to be movement/anim speeds and
    the iparams tend to be HP thresholds or RNG seeds, but treat that as a
    hint, not a guarantee. Edit one at a time and watch the mob in-game.
    """
    generic_f = ("Raw MovementData float for this slot (semantics not yet "
                 "reverse-engineered). On most mobs the early floats are move/"
                 "animation speeds — change ONE and observe the mob in-game to "
                 "learn what it does. See notes/movement-data.txt.")
    generic_i = ("Raw MovementData integer for this slot (semantics not yet "
                 "reverse-engineered). On documented mobs these are usually HP "
                 "phase thresholds or RNG seeds. Change ONE and observe in-game.")
    return [
        FieldSpec("fparam1", "fparam1", "AI Behavior", "float", "animations", tooltip=generic_f),
        FieldSpec("fparam2", "fparam2", "AI Behavior", "float", "animations", tooltip=generic_f),
        FieldSpec("fparam3", "fparam3", "AI Behavior", "float", "animations", tooltip=generic_f),
        FieldSpec("fparam4", "fparam4", "AI Behavior", "float", "animations", tooltip=generic_f),
        FieldSpec("fparam5", "fparam5", "AI Behavior", "float", "animations", tooltip=generic_f),
        FieldSpec("fparam6", "fparam6", "AI Behavior", "float", "animations", tooltip=generic_f),
        FieldSpec("iparam1", "iparam1", "AI Behavior", "uint",  "animations", tooltip=generic_i),
        FieldSpec("iparam2", "iparam2", "AI Behavior", "uint",  "animations", tooltip=generic_i),
        FieldSpec("iparam3", "iparam3", "AI Behavior", "uint",  "animations", tooltip=generic_i),
        FieldSpec("iparam4", "iparam4", "AI Behavior", "uint",  "animations", tooltip=generic_i),
        FieldSpec("iparam5", "iparam5", "AI Behavior", "uint",  "animations", tooltip=generic_i),
        FieldSpec("iparam6", "iparam6", "AI Behavior", "uint",  "animations", tooltip=generic_i),
    ]


def _generic_boss_animations() -> List[FieldSpec]:
    """Bosses share the iparam2/iparam3 phase-threshold pattern.

    The iparam2/iparam3 boss-phase semantic is documented for De Rol Le
    and confirmed by the boss-data pool layout (memory:r2_psobb_findings).
    Other bosses (Dragon, Vol Opt, Dark Falz, Olga Flow, Saint Million,
    Shambertin, Kondrieu) follow the same pattern.
    """
    fields = _generic_animations()
    fields[6] = FieldSpec("iparam1", "phase_hp_total", "AI Behavior", "int",  "animations", (0, 0x7FFFFFFF),
                          tooltip="The boss's TOTAL HP pool. OVERRIDES the generic stats.hp for "
                                  "this boss — this is the real lever on how long the fight lasts. "
                                  "e.g. 10000.")
    fields[7] = FieldSpec("iparam2", "phase_2_hp",     "AI Behavior", "int",  "animations", (0, 0x7FFFFFFF),
                          tooltip="HP value at which the boss flips into PHASE 2 (new attacks / "
                                  "weak point opens). Set it as a fraction of phase_hp_total to "
                                  "control pacing — e.g. ~66% of total.")
    fields[8] = FieldSpec("iparam3", "phase_3_hp",     "AI Behavior", "int",  "animations", (0, 0x7FFFFFFF),
                          tooltip="HP value at which the boss flips into PHASE 3 (final/enrage "
                                  "phase, or the kill check). Set below phase_2_hp — e.g. ~33% of "
                                  "phase_hp_total.")
    return fields


# ---------------------------------------------------------------------------
# Slot family table — which animation override applies to which slots
# ---------------------------------------------------------------------------
# Slots not listed here get _generic_animations(). "AI-tunable" =
# excludes player slots (the BattleParam file mixes player and mob
# entries; the latter live above slot 0x40 in stock files but the slot
# table from BB Patch Project includes named entries only for mobs).
#
# Every slot in bp_mod.SLOT_NAMES is treated as AI-tunable; the picker
# in the UI uses this set as its mob list.

_BOSS_SLOTS = frozenset({
    0x0F,  # De Rol Le
    0x12,  # Dragon
    0x1E,  # DarkGunner / GalGryphon
    0x21,  # VolOptForm1
    0x22,  # VolOptPillar
    0x23,  # VolOptMonitor
    0x24,  # VolOptSpire
    0x25,  # VolOptForm2
    0x26,  # VolOptPrison
    0x2B,  # OlgaFlowForm1
    0x2C,  # OlgaFlowForm2
    0x2D,  # Gael
    0x2E,  # Giel
    0x36,  # DarkFalzForm1
    0x37,  # DarkFalzForm2
    0x38,  # DarkFalzForm3
})


def _animation_fields_for_slot(slot: int) -> List[FieldSpec]:
    # Booma family
    if slot in (0x4B, 0x4C, 0x4D):  # Booma / Gobooma / Gigobooma
        return _booma_animations()
    # Hildebear family
    if slot in (0x49, 0x4A):  # Hildebear / Hildeblue
        return _hildebear_animations()
    # De Rol Le
    if slot == 0x0F:
        return _de_rol_le_animations()
    # Bosses (generic phase-HP pattern)
    if slot in _BOSS_SLOTS:
        return _generic_boss_animations()
    # Fallback — no documented semantics
    return _generic_animations()


def _has_named_animation_fields(slot: int) -> bool:
    """True iff this slot has documented animation semantics (not the
    generic fparam fall-through)."""
    return (
        slot in (0x4B, 0x4C, 0x4D)  # Booma family
        or slot in (0x49, 0x4A)      # Hildebear family
        or slot == 0x0F              # De Rol Le
        or slot in _BOSS_SLOTS       # Bosses (generic phase-HP)
    )


def _build_schema(slot: int, name: str) -> MobSchema:
    fields = []
    fields.extend(_universal_stats_fields())
    fields.extend(_universal_combat_fields())
    fields.extend(_universal_resist_fields())
    fields.extend(_animation_fields_for_slot(slot))

    canonical: Dict[int, str] = {}
    if slot == 0x0F:
        canonical = {
            1: "total_hp",
            2: "armor_break_hp",
            3: "mask_off_hp",
            4: "ult_constant_a",
            5: "ult_constant_b",
        }
    elif slot in _BOSS_SLOTS:
        canonical = {
            1: "phase_hp_total",
            2: "phase_2_hp",
            3: "phase_3_hp",
        }

    if _has_named_animation_fields(slot):
        notes = ("Stats/Combat/Resists below are universal; the Movement & AI "
                 "Behavior fields are named specifically for this mob. Hover any "
                 "field for what it does + an example value. Blank = keep the "
                 "stock value.")
    else:
        notes = ("Stats/Combat/Resists below are universal and fully documented. "
                 "This slot's Movement/AI animation params have NOT been "
                 "reverse-engineered, so they appear as raw fparam/iparam labels "
                 "— they still affect the mob; edit one at a time and watch it "
                 "in-game. See notes/movement-data.txt.")

    return MobSchema(
        slot=slot, name=name, fields=fields,
        canonical_iparam_meanings=canonical, notes=notes,
    )


# Build the full schema table. One MobSchema per entry in
# bp_mod.SLOT_NAMES (all 79 named slots).
MOB_SCHEMAS: Dict[int, MobSchema] = {
    slot: _build_schema(slot, name)
    for slot, name in sorted(bp_mod.SLOT_NAMES.items())
}

# Reverse-lookup tables for the DSL parser (slot name -> slot id).
NAME_TO_SLOT: Dict[str, int] = {
    name.lower(): slot for slot, name in bp_mod.SLOT_NAMES.items()
}


def coverage_summary() -> Dict[str, int]:
    """How many mobs have named animation fields vs the generic fallback."""
    named = sum(1 for s in MOB_SCHEMAS if _has_named_animation_fields(s))
    return {
        "total_slots": len(MOB_SCHEMAS),
        "named_animation_fields": named,
        "generic_animation_fields": len(MOB_SCHEMAS) - named,
    }


# ---------------------------------------------------------------------------
# Mob name resolution
# ---------------------------------------------------------------------------
def resolve_mob(token) -> int:
    """Resolve a mob token (string name, hex string, or int) → slot id.

    Accepts:
      - "0x4B" / "0x4b"  → 75
      - "75"             → 75
      - "Booma"/"booma"  → 75 (case-insensitive name match)

    Raises ValueError if the token is unrecognised.
    """
    if isinstance(token, int):
        if token in MOB_SCHEMAS:
            return token
        raise ValueError(f"unknown slot id 0x{token:02X}")
    if not isinstance(token, str):
        raise ValueError(f"mob token must be int or str, got {type(token).__name__}")
    s = token.strip()
    if not s:
        raise ValueError("empty mob token")
    # Hex form
    if s.lower().startswith("0x"):
        try:
            n = int(s, 16)
        except ValueError:
            raise ValueError(f"bad hex mob id {token!r}")
        if n in MOB_SCHEMAS:
            return n
        raise ValueError(f"unknown slot id {token!r}")
    # Decimal form
    if s.isdigit():
        n = int(s)
        if n in MOB_SCHEMAS:
            return n
        raise ValueError(f"unknown slot id {n}")
    # Name form (case-insensitive)
    low = s.lower()
    if low in NAME_TO_SLOT:
        return NAME_TO_SLOT[low]
    raise ValueError(f"unknown mob name {token!r}")


# ---------------------------------------------------------------------------
# Difficulty resolution
# ---------------------------------------------------------------------------
DIFFICULTY_BY_NAME: Dict[str, int] = {
    "normal": 0, "n": 0,
    "hard": 1, "h": 1,
    "veryhard": 2, "vh": 2, "very_hard": 2, "veryHard": 2,
    "ultimate": 3, "u": 3,
}


def resolve_difficulty(token) -> List[int]:
    """Resolve a difficulty selector → list of difficulty indices.

    Accepts:
      - None / "" / "all"  → [0, 1, 2, 3]
      - int 0..3            → [int]
      - string name         → [matching index]
    """
    if token is None or token == "" or (isinstance(token, str) and token.strip().lower() == "all"):
        return [0, 1, 2, 3]
    if isinstance(token, int):
        if 0 <= token < 4:
            return [token]
        raise ValueError(f"difficulty index out of range: {token}")
    if isinstance(token, str):
        low = token.strip().lower()
        if low in DIFFICULTY_BY_NAME:
            return [DIFFICULTY_BY_NAME[low]]
        raise ValueError(f"unknown difficulty {token!r}")
    raise ValueError(f"bad difficulty {token!r}")


# ---------------------------------------------------------------------------
# DSL → BattleParam apply
# ---------------------------------------------------------------------------
@dataclass
class MobPatch:
    """One mob's overlay: changed fields only."""
    mob: str            # human label (slot name)
    slot: int
    difficulties: List[int]
    fields: Dict[str, Any]
    # source for tooling: where the patch came from (preset name, etc.)
    origin: Optional[str] = None


def parse_patch(payload: Mapping) -> MobPatch:
    """Parse one mob-patch dict from the wire format.

    Expected keys:
        mob (required): slot name or id
        difficulty (optional): "all" / "Normal" / int 0..3
        fields (required): {label: value, ...}
        origin (optional): freeform string
    """
    if not isinstance(payload, Mapping):
        raise ValueError(f"mob patch must be a dict, got {type(payload).__name__}")
    if "mob" not in payload:
        raise ValueError("mob patch missing 'mob'")
    if "fields" not in payload:
        raise ValueError("mob patch missing 'fields'")

    slot = resolve_mob(payload["mob"])
    diffs = resolve_difficulty(payload.get("difficulty"))
    fields = payload["fields"]
    if not isinstance(fields, Mapping):
        raise ValueError("'fields' must be a dict")

    return MobPatch(
        mob=bp_mod.SLOT_NAMES.get(slot, f"slot_{slot:02X}"),
        slot=slot,
        difficulties=diffs,
        fields=dict(fields),
        origin=payload.get("origin"),
    )


def parse_patches(payload) -> List[MobPatch]:
    """Parse the {"mobs": [...]} or [...] wire shape into MobPatch list."""
    if isinstance(payload, Mapping):
        if "mobs" in payload:
            payload = payload["mobs"]
    if not isinstance(payload, list):
        raise ValueError("expected list of mob patches or {'mobs': [...]}")
    out: List[MobPatch] = []
    for i, p in enumerate(payload):
        try:
            out.append(parse_patch(p))
        except ValueError as e:
            raise ValueError(f"patch[{i}]: {e}")
    return out


def _apply_field(entry, schema: MobSchema, label: str, dsl_value):
    """Set one DSL field on a BattleParamEntry. Mutates in place."""
    fs = schema.field_by_label(label)
    if fs is None:
        # Allow direct binary names as a fall-through ("fparam1", etc.)
        # so authors can hit slots without named schemas.
        for grp in ("animations", "stats", "attacks", "resists"):
            target = getattr(entry, grp, None)
            if isinstance(target, dict) and label in target:
                target[label] = dsl_value
                target.pop(f"_{label}_bits", None)
                return
        raise ValueError(f"unknown field {label!r} on mob {schema.name}")

    target = getattr(entry, fs.binary_group)
    if not isinstance(target, dict):
        raise ValueError(f"battle-param entry {fs.binary_group!r} not a dict")
    binary_value = dsl_to_binary(fs.kind, dsl_value)
    target[fs.binary_name] = binary_value
    # Drop any cached float-bit sidecar; the new value should win.
    target.pop(f"_{fs.binary_name}_bits", None)


def apply_patch_to_file(bpf: bp_mod.BattleParamFile, patch: MobPatch) -> None:
    """Apply one MobPatch to a BattleParamFile in-place.

    Touches the targeted (slot, difficulty) cells and clears the
    matching float-bit sidecars so :func:`bp_mod.serialize` writes the
    new values.
    """
    schema = MOB_SCHEMAS.get(patch.slot)
    if schema is None:
        raise ValueError(f"no schema for slot 0x{patch.slot:02X}")
    for d in patch.difficulties:
        if d < 0 or d >= len(bpf.difficulties):
            raise ValueError(f"difficulty index {d} out of range")
        diff = bpf.difficulties[d]
        if patch.slot >= len(diff.entries):
            raise ValueError(f"slot 0x{patch.slot:02X} out of range")
        entry = diff.entries[patch.slot]
        for label, dsl_value in patch.fields.items():
            _apply_field(entry, schema, label, dsl_value)


def apply_patches(
    base: bp_mod.BattleParamFile,
    patches: Iterable[MobPatch],
) -> bp_mod.BattleParamFile:
    """Compose patches in priority order over a stock file.

    Returns a *deep copy* of ``base`` with all patches applied. The
    input is not mutated. Patches later in the iterable win on a key
    collision (last writer wins).
    """
    # Deep copy via the JSON round-trip so dicts (which include the
    # `_field_bits` sidecars) get fresh references.
    out = bp_mod.BattleParamFile.from_json(base.to_json())
    out.variant = base.variant
    for p in patches:
        apply_patch_to_file(out, p)
    return out


def encode_overlay(
    edited: bp_mod.BattleParamFile,
    base: bp_mod.BattleParamFile,
    *,
    difficulty: Optional[int] = None,
) -> List[Dict]:
    """Compute the minimal DSL overlay that turns ``base`` into ``edited``.

    Args:
        edited: result of authoring.
        base: stock baseline.
        difficulty: if set, restrict to one difficulty index (0..3);
            else generate a per-difficulty patch wherever differences
            exist.

    Returns:
        A list of patch dicts in the wire format (one per (slot, diff)
        tuple that has any difference). Empty list if no diffs.
    """
    diffs_to_check = [difficulty] if difficulty is not None else list(range(4))
    out: List[Dict] = []
    for d in diffs_to_check:
        if d >= len(edited.difficulties) or d >= len(base.difficulties):
            continue
        e_diff = edited.difficulties[d]
        b_diff = base.difficulties[d]
        for slot in MOB_SCHEMAS:
            if slot >= len(e_diff.entries) or slot >= len(b_diff.entries):
                continue
            e_ent = e_diff.entries[slot]
            b_ent = b_diff.entries[slot]
            schema = MOB_SCHEMAS[slot]
            patch_fields: Dict[str, Any] = {}
            for fs in schema.fields:
                e_v = (getattr(e_ent, fs.binary_group, {}) or {}).get(fs.binary_name)
                b_v = (getattr(b_ent, fs.binary_group, {}) or {}).get(fs.binary_name)
                if not _values_equal(e_v, b_v):
                    patch_fields[fs.label] = binary_to_dsl(fs.kind, e_v)
            if patch_fields:
                out.append({
                    "mob": schema.name,
                    "slot": slot,
                    "slot_hex": f"0x{slot:02X}",
                    "difficulty": bp_mod.DIFFICULTY_NAMES[d],
                    "fields": patch_fields,
                })
    return out


def _values_equal(a, b) -> bool:
    """NaN-safe equality."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, float) or isinstance(b, float):
        try:
            af = float(a)
            bf = float(b)
        except (TypeError, ValueError):
            return False
        if math.isnan(af) and math.isnan(bf):
            return True
        return af == bf
    return a == b


# ---------------------------------------------------------------------------
# Mob-DSL → BattleParam compile entry point
# ---------------------------------------------------------------------------
def compile_to_battle_param(
    base: bp_mod.BattleParamFile,
    patches: Iterable[MobPatch],
) -> bp_mod.BattleParamFile:
    """Apply DSL patches to a stock BattleParamFile.

    Convenience wrapper around :func:`apply_patches` that the server
    `/api/mob_dsl/compile` endpoint calls. Output is JSON-ready via
    ``to_json()``.
    """
    return apply_patches(base, patches)


# ---------------------------------------------------------------------------
# Preset library — JSON files in data/mob_presets/
# ---------------------------------------------------------------------------
PRESETS_DIR = Path(__file__).resolve().parent.parent / "data" / "mob_presets"


def list_presets(presets_dir: Optional[Path] = None) -> List[Dict]:
    """List shipped presets with their metadata.

    Each entry: {name, file, title, description, mobs (list of str)}.
    Used by /api/mob_dsl/presets and the UI's "Apply preset" dropdown.
    """
    d = Path(presets_dir) if presets_dir is not None else PRESETS_DIR
    out: List[Dict] = []
    if not d.is_dir():
        return out
    for path in sorted(d.glob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        mobs = []
        for p in obj.get("mobs", []):
            if isinstance(p, Mapping) and "mob" in p:
                mobs.append(str(p["mob"]))
        out.append({
            "name": path.stem,
            "file": str(path),
            "title": obj.get("title", path.stem),
            "description": obj.get("description", ""),
            "mobs": mobs,
        })
    return out


def load_preset(name: str, presets_dir: Optional[Path] = None) -> Dict:
    """Load one preset by stem name. Raises FileNotFoundError on miss."""
    d = Path(presets_dir) if presets_dir is not None else PRESETS_DIR
    # Filename safety: refuse path components.
    bare = Path(name).name
    if bare != name or "/" in name or "\\" in name:
        raise ValueError(f"invalid preset name {name!r}")
    p = d / f"{bare}.json"
    if not p.is_file():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Lightweight YAML-ish parser (for the DSL example in the brief)
# ---------------------------------------------------------------------------
# PyYAML isn't a dependency. We accept the *DSL JSON* shape directly
# (which is the wire format), plus a small "YAML-lite" subset that
# matches the example in the brief — flat key/value pairs with one
# level of nested groups.
#
# This is intentionally restrictive; complex YAML features (anchors,
# multi-line scalars, references) are not supported. Authors who want
# rich YAML can install PyYAML themselves and feed parsed dicts in.

_YAML_NUM_RE = re.compile(r"^[-+]?(\d+(\.\d+)?([eE][-+]?\d+)?|\.\d+([eE][-+]?\d+)?)$")
_YAML_HEX_RE = re.compile(r"^0x[0-9a-fA-F]+$")


def _yaml_lite_value(s: str):
    s = s.strip()
    if not s:
        return ""
    # Strip surrounding quotes
    if (s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'"):
        return s[1:-1]
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.lower() in ("null", "none", "~"):
        return None
    if _YAML_HEX_RE.match(s):
        return int(s, 16)
    if _YAML_NUM_RE.match(s):
        if "." in s or "e" in s or "E" in s:
            return float(s)
        return int(s)
    return s


def parse_yaml_lite(text: str) -> Dict:
    """Parse the YAML-lite subset to a dict.

    Supports::

        mob: hildebear
        difficulty: Normal
        movement:
          walk_speed: 0.8
          run_speed:  1.2
        combat:
          atp_min: 60
          atp_max: 110

    The "section" labels (``movement``, ``combat``, ``resists``) are
    *flattened* into ``fields`` so the result matches the DSL JSON
    shape::

        {
          "mob": "hildebear",
          "difficulty": "Normal",
          "fields": {"walk_speed": 0.8, "run_speed": 1.2, "atp_min": 60, ...}
        }

    Unknown top-level keys (except mob/slot/difficulty/origin/comment)
    are treated as section labels and flattened.
    """
    out: Dict = {"fields": {}}
    section: Optional[str] = None
    section_indent = -1

    for raw_line in text.splitlines():
        # Strip line comments (#...)
        line = raw_line
        if "#" in line:
            line = line.split("#", 1)[0]
        if not line.strip():
            continue
        # Compute indent
        stripped = line.lstrip(" \t")
        indent = len(line) - len(stripped)
        if ":" not in stripped:
            continue
        key, _, val = stripped.partition(":")
        key = key.strip()
        val = val.strip()

        if indent == 0:
            # Top-level key
            if val == "":
                # Section header
                section = key
                section_indent = -1  # set on first child
                continue
            section = None
            # Direct top-level field (mob/difficulty/origin/...)
            if key in ("mob", "slot", "difficulty", "origin", "comment", "preset"):
                out[key] = _yaml_lite_value(val)
            else:
                out["fields"][key] = _yaml_lite_value(val)
        else:
            # Nested key — flatten into fields if we're under a section
            if section_indent < 0:
                section_indent = indent
            if indent < section_indent:
                # de-dent — back to top
                section = None
                section_indent = -1
                # Treat as top-level
                if key in ("mob", "slot", "difficulty", "origin"):
                    out[key] = _yaml_lite_value(val)
                else:
                    out["fields"][key] = _yaml_lite_value(val)
            else:
                # Stay in section
                out["fields"][key] = _yaml_lite_value(val)
    return out


def emit_yaml_lite(patch: Mapping) -> str:
    """Inverse of :func:`parse_yaml_lite` — emits a flat YAML-lite block.

    Used by the UI's "view DSL source" button + by round-trip tests.
    Produces deterministic output (sorted within each section) for diff
    stability.
    """
    lines: List[str] = []
    if "mob" in patch:
        lines.append(f"mob: {_yaml_lite_emit_value(patch['mob'])}")
    if "difficulty" in patch:
        lines.append(f"difficulty: {_yaml_lite_emit_value(patch['difficulty'])}")
    if "origin" in patch:
        lines.append(f"origin: {_yaml_lite_emit_value(patch['origin'])}")
    fields = patch.get("fields", {})
    if fields:
        # Group fields by their schema "group" so the output reads
        # nicely. Look up the schema if mob is known.
        slot = None
        try:
            slot = resolve_mob(patch.get("mob")) if patch.get("mob") is not None else None
        except ValueError:
            slot = None
        groups: Dict[str, List[Tuple[str, Any]]] = {}
        if slot is not None and slot in MOB_SCHEMAS:
            schema = MOB_SCHEMAS[slot]
            label_to_group = {fs.label: fs.group for fs in schema.fields}
            for k, v in fields.items():
                grp = label_to_group.get(k, "Other")
                groups.setdefault(grp, []).append((k, v))
        else:
            groups["fields"] = list(fields.items())

        for grp, items in groups.items():
            lines.append(f"{grp.lower().replace(' ', '_')}:")
            for k, v in sorted(items, key=lambda kv: kv[0]):
                lines.append(f"  {k}: {_yaml_lite_emit_value(v)}")
    return "\n".join(lines) + "\n"


def _yaml_lite_emit_value(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if s == "" or any(ch in s for ch in ":#\n"):
        return '"' + s.replace('"', '\\"') + '"'
    return s


# ---------------------------------------------------------------------------
# Schema export helpers (for the server's GET endpoints)
# ---------------------------------------------------------------------------
def all_schemas_json() -> Dict:
    """Return every mob schema as JSON. Used by /api/mob_dsl/schemas."""
    return {
        "schemas": [s.to_json() for s in MOB_SCHEMAS.values()],
        "coverage": coverage_summary(),
    }


def schema_json(slot_or_name) -> Dict:
    """Return one mob schema as JSON. Raises ValueError on miss."""
    slot = resolve_mob(slot_or_name)
    return MOB_SCHEMAS[slot].to_json()


__all__ = [
    "FieldSpec", "MobSchema", "MobPatch",
    "MOB_SCHEMAS", "PRESETS_DIR", "GAME_TICK_HZ",
    "dsl_to_binary", "binary_to_dsl",
    "resolve_mob", "resolve_difficulty",
    "parse_patch", "parse_patches",
    "apply_patch_to_file", "apply_patches",
    "compile_to_battle_param", "encode_overlay",
    "list_presets", "load_preset",
    "parse_yaml_lite", "emit_yaml_lite",
    "all_schemas_json", "schema_json", "coverage_summary",
]
