"""Parser/serializer for PSOBB ``ItemPMT-bb-v4.prs`` (decompressed payload).

ItemPMT is the master item parameter table that defines stats for every
weapon, armor (frame), shield (barrier), unit, mag and tool that the
client knows how to render. The server holds it in a PRS-compressed
payload (`ItemPMT-bb-v4.prs` for newserv, `ItemPMT.prs` for
Booma.Server) and reads it on launch. Editing the file lets the
operator rebalance stats without touching the client install.

This module operates on the *uncompressed* bytes — `formats.prs.decompress`
gets you in, this module's `pack()` re-compresses with `formats.prs.compress`.

============================================================================
Binary layout (little-endian, V4 / Blue Burst)
============================================================================

The file's structure is documented in newserv's MIT-licensed
``src/ItemParameterTable.cc/.hh``. The footer at ``len(buf) - 0x10``
holds a single ``u32`` ``offset_table_offset`` pointing to a
``TableOffsetsV3V4`` struct (0x5C bytes) containing 23 ``u32`` offsets
into the file. The structure of each table is documented inline below.

The file is built up of:

A. **Item section** (offsets 0x000 .. ~photon_color_table)
   - 0x000-0x03F: 64 bytes of unreferenced preamble (likely ItemBase
     sentinel slots for weapon + unit, kept as opaque bytes)
   - shields:  166 × 0x20 ArmorOrShieldV4 (offset from armor_table[1])
   - frames:   89 × 0x20 ArmorOrShieldV4 (offset from armor_table[0])
   - units:    101 × 0x14 UnitV4 (offset from unit_table)
   - mags:     83 × 0x1C MagV4 (offset from mag_table)
   - tools:    27 sub-classes × variable count, each entry 0x18 ToolV4
   - weapons:  ~24 weapon classes × variable count, each entry 0x2C
               WeaponV4 (offsets from weapon_table[i].offset)

B. **Lookup tables**
   - photon_color_table: 22 × 0x24 (9 floats per entry) opaque
   - weapon_range_table: opaque (newserv: "???")
   - v1_replacement_table: u8 array (240 bytes)
   - weapon_sale_divisor_table: float[] (165 entries × 4 bytes)
   - sale_divisor_table: 4 floats (NonWeaponSaleDivisors); the rest of
     the byte slice up to star_value_table holds the 8 mag-feed
     sub-tables (8 × 11 × 8 = 704 bytes)
   - star_value_table: 816 × u8 (one per item id 0xB1..0x437)
   - special_data_table: 41 × 4 bytes (Special: u16 type, u16 amount)
   - weapon_effect_table: 474 × 16 bytes (opaque)
   - shield_effect_table: 346 × 8 bytes (opaque)
   - unknown_a1: 8 bytes opaque
   - stat_boost_table: 52 × 6 bytes (StatBoost: u8[2], u16[2])
   - max_tech_level_table: 19 techs × 12 classes (228 bytes), but
     allocated 0x19E4 = 6628 bytes (mostly unused)
   - tech_boost_table: 87 × 0x18 (TechniqueBoost[3], each 8 bytes)

C. **Item table headers** (the ArrayRef structs the table-offset table
   points to):
   - armor_table → 2 ArrayRefs at 0x147A4 (frames + barriers)
   - unit_table → 1 ArrayRef at 0x147B4
   - mag_table → 1 ArrayRef at 0x147BC
   - tool_table → 27 ArrayRefs at 0x147C4
   - weapon_table → 238 ArrayRefs at 0x1489C (most empty; ~24 active)

D. **Combination / unwrap / unsealable / ranged-special / mag-feed**
   small offset table records.

E. **TableOffsetsV3V4** + final ``offset_table_offset`` u32.

For round-trip preservation we walk the file in the order presented by
the offset table, parse every typed record we know, and capture the
remaining bytes (everything outside our parser's coverage) as opaque
blobs keyed by section. On serialization we emit in the same order and
patch typed-record bytes back in place.

============================================================================
ArrayRef binary layout
============================================================================

The newserv struct ``ArrayRef`` is on-disk ``(count: u32, offset: u32)``
not ``(offset, count)``. Static analysis of the binary against
newserv's reader confirmed the count comes first. References:
- ``ItemParameterTable.cc`` ``indirect_lookup_2d_count`` returns
  ``r.pget<ArrayRefT<BE>>(...).count``.
- File offsets reading as ``(89, 0x1500)`` for the frames sub-table:
  count=89 frames at offset 0x1500, ArmorOrShieldV4 size 0x20 → 89×0x20
  = 0xB20 → ends at 0x2020, the next sub-table's start. Confirmed.

============================================================================
JSON shape returned by `to_json`
============================================================================

::

    {
      "header_blob": "<hex of bytes [0..0x40]>",
      "weapons":   [{class: 0, items: [<WeaponEntry>, ...]}, ...],
      "armors":    [<ArmorEntry>, ...],     // frames
      "shields":   [<ShieldEntry>, ...],    // barriers
      "units":     [<UnitEntry>, ...],
      "mags":      [<MagEntry>, ...],
      "tools":     [{class: 0, items: [<ToolEntry>, ...]}, ...],
      "specials":  [<SpecialEntry>, ...],          // 41 entries
      "stat_boosts": [<StatBoostEntry>, ...],      // 52
      "mag_feeds":  [{table: 0, results: [<MagFeedResult>, ...]}],
      "combinations": [<ItemCombinationEntry>, ...],
      "v1_replacement": [<u8>, ...],               // 240 entries
      "weapon_sale_divisors": [<float>, ...],      // 165
      "sale_divisors": {armor, shield, unit, mag},
      "star_values": [<u8>, ...],                  // 816 entries
      "max_tech_levels": [[<u8>×12], ×19],         // 19 × 12 grid
      // Opaque (unparsed) blobs for byte-exact round-trip:
      "_opaque": {
        "header_preamble":      "<hex 0x40 bytes>",
        "photon_color_table":   "<hex 0x318 bytes>",
        "weapon_range_table":   "<hex 0x1338 bytes>",
        "weapon_effect_table":  "<hex 0x1DA0 bytes>",
        "shield_effect_table":  "<hex 0xAD4 bytes>",
        "unknown_a1":           "<hex 0x8 bytes>",
        "tech_boost_table":     "<hex 0x518 bytes>",
        "max_tech_level_pad":   "<hex bytes after 19×12 grid>",
        "unwrap_table":         "<hex>",
        "unsealable_table":     "<hex>",
        "ranged_special_table": "<hex>",
        "footer_offsets":       "<hex 0x5C bytes>",
        "offset_table_offset":  <int>,
      }
    }

The opaque blobs let us round-trip stock files without round-tripping
through every byte of every reverse-engineered table; they remain
fully editable by reading hex / u8 arrays.

For NaN/Inf preservation in the typed float fields we use the same
``_<field>_bits`` sidecar trick as `formats.battle_param`.
"""
from __future__ import annotations

import json
import math
import struct
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Section sizes (bytes) — verified against the shipped Booma.Server file
# ---------------------------------------------------------------------------
WEAPON_RECORD_SIZE = 0x2C
ARMOR_OR_SHIELD_RECORD_SIZE = 0x20
UNIT_RECORD_SIZE = 0x14
MAG_RECORD_SIZE = 0x1C
TOOL_RECORD_SIZE = 0x18
SPECIAL_RECORD_SIZE = 0x4
STAT_BOOST_RECORD_SIZE = 0x6
ITEM_COMBINATION_RECORD_SIZE = 0x10
MAG_FEED_RESULT_SIZE = 0x8

# Header preamble at file start (asserted by tests)
HEADER_PREAMBLE_SIZE = 0x40

# TableOffsetsV3V4 (23 × u32)
TABLE_OFFSETS_SIZE = 0x5C
NUM_TABLE_OFFSETS = 23

# Item-id range covered by star_value_table (V4)
STAR_VALUE_FIRST_ID = 0xB1
STAR_VALUE_LAST_ID = 0x437

# Mag feed: 8 sub-tables × 11 results
MAG_FEED_NUM_TABLES = 8
MAG_FEED_RESULTS_PER_TABLE = 11

# max_tech_level grid
MAX_TECH_NUM_TECHS = 19
MAX_TECH_NUM_CLASSES = 12

# Counts of small lookup tables (verified via newserv comments + binary)
NUM_SPECIALS = 0x29  # 41


# ---------------------------------------------------------------------------
# Schema-driven helpers (mirroring the battle_param style)
# ---------------------------------------------------------------------------
# Each schema is a tuple (name, struct_format, count). count > 1 means
# the field is an array; surfaced to JSON as a Python list. Float fields
# get a parallel ``_<name>_bits`` sidecar storing the int32 bit pattern,
# preserving NaN/Inf across JSON round-trip.

def _schema_byte_size(schema: Tuple[Tuple[str, str, int], ...]) -> int:
    return sum(struct.calcsize(fmt) * count for _name, fmt, count in schema)


def _parse_record(buf: bytes, off: int, schema: Tuple[Tuple[str, str, int], ...]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    cur = off
    for name, fmt, count in schema:
        elem_size = struct.calcsize(fmt)
        if count == 1:
            (val,) = struct.unpack_from(f"<{fmt}", buf, cur)
            if fmt == "f":
                bits = struct.unpack_from("<I", buf, cur)[0]
                out[name] = val
                out[f"_{name}_bits"] = bits
            else:
                out[name] = val
            cur += elem_size
        else:
            vals = list(struct.unpack_from(f"<{count}{fmt}", buf, cur))
            out[name] = vals
            if fmt == "f":
                bits = list(struct.unpack_from(f"<{count}I", buf, cur))
                out[f"_{name}_bits"] = bits
            cur += elem_size * count
    return out


def _pack_record(record: Dict[str, Any], schema: Tuple[Tuple[str, str, int], ...]) -> bytes:
    parts: List[bytes] = []
    for name, fmt, count in schema:
        if count == 1:
            if fmt == "f":
                bits_key = f"_{name}_bits"
                if bits_key in record and record[bits_key] is not None:
                    parts.append(struct.pack("<I", record[bits_key] & 0xFFFFFFFF))
                else:
                    v = record.get(name)
                    if v is None:
                        parts.append(b"\x00\x00\x00\x00")
                    else:
                        parts.append(struct.pack("<f", v))
            else:
                parts.append(struct.pack(f"<{fmt}", record[name]))
        else:
            v = record[name]
            if not isinstance(v, list) or len(v) != count:
                raise ValueError(
                    f"_pack_record: field {name!r} expected list of length "
                    f"{count}, got {type(v).__name__} {v!r}"
                )
            if fmt == "f":
                bits_key = f"_{name}_bits"
                if bits_key in record and isinstance(record[bits_key], list):
                    bits = record[bits_key]
                    if len(bits) != count:
                        raise ValueError(
                            f"_pack_record: bits sidecar for {name!r} has "
                            f"{len(bits)}, expected {count}"
                        )
                    parts.append(struct.pack(f"<{count}I", *(b & 0xFFFFFFFF for b in bits)))
                else:
                    safe = [0.0 if x is None else x for x in v]
                    parts.append(struct.pack(f"<{count}f", *safe))
            else:
                parts.append(struct.pack(f"<{count}{fmt}", *v))
    return b"".join(parts)


def _scrub_nonfinite(o: Any) -> Any:
    if isinstance(o, float):
        if math.isnan(o) or math.isinf(o):
            return None
        return o
    if isinstance(o, list):
        return [_scrub_nonfinite(x) for x in o]
    if isinstance(o, dict):
        return {k: _scrub_nonfinite(v) for k, v in o.items()}
    return o


# ---------------------------------------------------------------------------
# Field schemas
# ---------------------------------------------------------------------------
# ItemBaseV4 (0x0C): u32 id, u16 type, u16 skin, u32 team_points
ITEM_BASE_SCHEMA: Tuple[Tuple[str, str, int], ...] = (
    ("id", "I", 1),
    ("type", "H", 1),
    ("skin", "H", 1),
    ("team_points", "I", 1),
)

# WeaponV4 (0x2C). Layout from newserv ItemParameterTable.hh:
#   +0x00 ItemBaseV4 (0x0C)
#   +0x0C u16 class_flags
#   +0x0E s16 atp_min
#   +0x10 s16 atp_max
#   +0x12 s16 atp_required
#   +0x14 s16 mst_required
#   +0x16 s16 ata_required
#   +0x18 s16 mst
#   +0x1A u8  max_grind
#   +0x1B u8  photon
#   +0x1C s8  special    (-1 = no special)
#   +0x1D u8  ata
#   +0x1E u8  stat_boost_entry_index
#   +0x1F u8  projectile
#   +0x20 s8  trail1_x
#   +0x21 s8  trail1_y
#   +0x22 s8  trail2_x
#   +0x23 s8  trail2_y
#   +0x24 u8  color
#   +0x25 u8[3] unknown_a1
#   +0x28 u8 unknown_a4
#   +0x29 u8 unknown_a5
#   +0x2A u8 tech_boost
#   +0x2B u8 behavior_flags
WEAPON_SCHEMA: Tuple[Tuple[str, str, int], ...] = ITEM_BASE_SCHEMA + (
    ("class_flags", "H", 1),
    ("atp_min", "h", 1),
    ("atp_max", "h", 1),
    ("atp_required", "h", 1),
    ("mst_required", "h", 1),
    ("ata_required", "h", 1),
    ("mst", "h", 1),
    ("max_grind", "B", 1),
    ("photon", "B", 1),
    ("special", "b", 1),
    ("ata", "B", 1),
    ("stat_boost_entry_index", "B", 1),
    ("projectile", "B", 1),
    ("trail1_x", "b", 1),
    ("trail1_y", "b", 1),
    ("trail2_x", "b", 1),
    ("trail2_y", "b", 1),
    ("color", "B", 1),
    ("unknown_a1", "B", 3),
    ("unknown_a4", "B", 1),
    ("unknown_a5", "B", 1),
    ("tech_boost", "B", 1),
    ("behavior_flags", "B", 1),
)
assert _schema_byte_size(WEAPON_SCHEMA) == WEAPON_RECORD_SIZE, (
    f"WeaponV4 schema is {_schema_byte_size(WEAPON_SCHEMA)} bytes, expected "
    f"{WEAPON_RECORD_SIZE}"
)

# ArmorOrShieldV4 (0x20). Both armors (frames) and shields (barriers)
# share this layout; ``flags_type`` distinguishes them.
#   +0x00 ItemBaseV4 (0x0C)
#   +0x0C u16 dfp
#   +0x0E u16 evp
#   +0x10 u8  block_particle
#   +0x11 u8  block_effect
#   +0x12 u16 class_flags
#   +0x14 u8  required_level
#   +0x15 u8  efr
#   +0x16 u8  eth
#   +0x17 u8  eic
#   +0x18 u8  edk
#   +0x19 u8  elt
#   +0x1A u8  dfp_range
#   +0x1B u8  evp_range
#   +0x1C u8  stat_boost_entry_index
#   +0x1D u8  tech_boost
#   +0x1E u8  flags_type      (0/1/2/3 — 0 = standard armor, 1 = ?, etc.)
#   +0x1F u8  unknown_a4
ARMOR_OR_SHIELD_SCHEMA: Tuple[Tuple[str, str, int], ...] = ITEM_BASE_SCHEMA + (
    ("dfp", "H", 1),
    ("evp", "H", 1),
    ("block_particle", "B", 1),
    ("block_effect", "B", 1),
    ("class_flags", "H", 1),
    ("required_level", "B", 1),
    ("efr", "B", 1),
    ("eth", "B", 1),
    ("eic", "B", 1),
    ("edk", "B", 1),
    ("elt", "B", 1),
    ("dfp_range", "B", 1),
    ("evp_range", "B", 1),
    ("stat_boost_entry_index", "B", 1),
    ("tech_boost", "B", 1),
    ("flags_type", "B", 1),
    ("unknown_a4", "B", 1),
)
assert _schema_byte_size(ARMOR_OR_SHIELD_SCHEMA) == ARMOR_OR_SHIELD_RECORD_SIZE

# UnitV4 (0x14):
#   +0x00 ItemBaseV4 (0x0C)
#   +0x0C u16 stat
#   +0x0E u16 stat_amount
#   +0x10 s16 modifier_amount
#   +0x12 u8[2] unused
UNIT_SCHEMA: Tuple[Tuple[str, str, int], ...] = ITEM_BASE_SCHEMA + (
    ("stat", "H", 1),
    ("stat_amount", "H", 1),
    ("modifier_amount", "h", 1),
    ("unused", "B", 2),
)
assert _schema_byte_size(UNIT_SCHEMA) == UNIT_RECORD_SIZE

# MagV4 (0x1C):
#   +0x00 ItemBaseV4 (0x0C)
#   +0x0C u16 feed_table
#   +0x0E u8  photon_blast
#   +0x0F u8  activation
#   +0x10 u8[4] trigger flags  (on_pb_full, on_low_hp, on_death, on_boss)
#   +0x14 u8[4] activation flags per trigger
#   +0x18 u16 class_flags
#   +0x1A u8[2] unused
MAG_SCHEMA: Tuple[Tuple[str, str, int], ...] = ITEM_BASE_SCHEMA + (
    ("feed_table", "H", 1),
    ("photon_blast", "B", 1),
    ("activation", "B", 1),
    ("trigger_flags", "B", 4),
    ("activation_flags", "B", 4),
    ("class_flags", "H", 1),
    ("unused", "B", 2),
)
assert _schema_byte_size(MAG_SCHEMA) == MAG_RECORD_SIZE

# ToolV4 (0x18):
#   +0x00 ItemBaseV4 (0x0C)
#   +0x0C u16 amount
#   +0x0E u16 tech
#   +0x10 s32 cost
#   +0x14 u32 item_flags
TOOL_SCHEMA: Tuple[Tuple[str, str, int], ...] = ITEM_BASE_SCHEMA + (
    ("amount", "H", 1),
    ("tech", "H", 1),
    ("cost", "i", 1),
    ("item_flags", "I", 1),
)
assert _schema_byte_size(TOOL_SCHEMA) == TOOL_RECORD_SIZE

# Special (4 bytes): u16 type (0xFFFF default), u16 amount
SPECIAL_SCHEMA: Tuple[Tuple[str, str, int], ...] = (
    ("type", "H", 1),
    ("amount", "H", 1),
)
assert _schema_byte_size(SPECIAL_SCHEMA) == SPECIAL_RECORD_SIZE

# StatBoost (6 bytes): u8[2] stats, u16[2] amounts
STAT_BOOST_SCHEMA: Tuple[Tuple[str, str, int], ...] = (
    ("stats", "B", 2),
    ("amounts", "H", 2),
)
assert _schema_byte_size(STAT_BOOST_SCHEMA) == STAT_BOOST_RECORD_SIZE

# ItemCombination (0x10 bytes):
#   +0x00 u8[3] used_item
#   +0x03 u8[3] equipped_item
#   +0x06 u8[3] result_item
#   +0x09 u8 mag_level
#   +0x0A u8 grind
#   +0x0B u8 level
#   +0x0C u8 char_class
#   +0x0D u8[3] unused/padding
ITEM_COMBINATION_SCHEMA: Tuple[Tuple[str, str, int], ...] = (
    ("used_item", "B", 3),
    ("equipped_item", "B", 3),
    ("result_item", "B", 3),
    ("mag_level", "B", 1),
    ("grind", "B", 1),
    ("level", "B", 1),
    ("char_class", "B", 1),
    ("unused", "B", 3),
)
assert _schema_byte_size(ITEM_COMBINATION_SCHEMA) == ITEM_COMBINATION_RECORD_SIZE

# MagFeedResult (8 bytes): s8 def, s8 pow, s8 dex, s8 mind, s8 iq, s8 sync, u8[2] unused
MAG_FEED_RESULT_SCHEMA: Tuple[Tuple[str, str, int], ...] = (
    ("def", "b", 1),
    ("pow", "b", 1),
    ("dex", "b", 1),
    ("mind", "b", 1),
    ("iq", "b", 1),
    ("synchro", "b", 1),
    ("unused", "B", 2),
)
assert _schema_byte_size(MAG_FEED_RESULT_SCHEMA) == MAG_FEED_RESULT_SIZE

# Sale divisors (4 floats — armor, shield, unit, mag)
SALE_DIVISORS_SCHEMA: Tuple[Tuple[str, str, int], ...] = (
    ("armor", "f", 1),
    ("shield", "f", 1),
    ("unit", "f", 1),
    ("mag", "f", 1),
)
assert _schema_byte_size(SALE_DIVISORS_SCHEMA) == 0x10


# Friendly names for weapon classes 0x00..0x16 (V4). Indices not listed
# default to "class_<hex>".
WEAPON_CLASS_NAMES: Dict[int, str] = {
    0x00: "Saber",
    0x01: "Sword",
    0x02: "Dagger",
    0x03: "Partisan",
    0x04: "Slicer",
    0x05: "Handgun",
    0x06: "Rifle",
    0x07: "Mechgun",
    0x08: "Shot",
    0x09: "Cane",
    0x0A: "Rod",
    0x0B: "Wand",
    0x0C: "Photon Launcher",
    0x0D: "Twin Sword",
    0x0E: "Double Saber",
    0x0F: "Knuckle",
    0x10: "Claw",
    0x11: "Katana",
    0x12: "Twin Katana",
    0x13: "J-Cutter",
    0x14: "Photon Bow",
    0x15: "Photon Bazooka",
    0x16: "Photon Rifle",
}

# Friendly names for tool subgroups (V4: 27 classes). These follow the
# in-game item-id grouping documented by newserv `notes/items.txt`.
TOOL_CLASS_NAMES: Dict[int, str] = {
    0x00: "Monomate / Mate",
    0x01: "Monofluid / Fluid",
    0x02: "Disk / Technique Disk",
    0x03: "Telepipe",
    0x04: "Trap Vision",
    0x05: "Scape Doll",
    0x06: "Trap (Damage)",
    0x07: "Sol Atomizer",
    0x08: "Moon Atomizer",
    0x09: "Star Atomizer",
    0x0A: "Antidote / Antiparalysis",
    0x0B: "Misc Healing",
    0x0C: "Photon Drop / Sphere / Crystal",
    0x0D: "Item Tickets",
    0x0E: "Materials",
    0x0F: "Photon Ticket",
    0x10: "Cell-related",
    0x11: "Photon Misc",
    0x12: "Material",
    0x13: "Special Items",
    0x14: "Junk / Dead Items",
    0x15: "Quest Tools",
    0x16: "Christmas Items",
    0x17: "Hat / Costume",
    0x18: "Boost Item",
    0x19: "Misc Quest",
    0x1A: "Reserved",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class WeaponEntry:
    """A single weapon record (WeaponV4)."""
    fields: Dict[str, Any]


@dataclass
class WeaponClass:
    """One weapon class (sword, gun, ...) as a list of weapons."""
    class_index: int
    name: str
    items: List[Dict[str, Any]]   # raw record dicts


@dataclass
class ToolClass:
    """One tool subgroup (potions, materials, ...) as a list of tools."""
    class_index: int
    name: str
    items: List[Dict[str, Any]]


@dataclass
class MagFeedSubTable:
    """One mag-feed sub-table (11 results)."""
    table_index: int
    results: List[Dict[str, Any]]


@dataclass
class ItemPMTFile:
    """Parsed ItemPMT-bb-v4 (decompressed)."""
    header_preamble: str               # hex of first 0x40 bytes
    weapons: List[WeaponClass]         # by class index
    armors: List[Dict[str, Any]]       # frames
    shields: List[Dict[str, Any]]      # barriers
    units: List[Dict[str, Any]]
    mags: List[Dict[str, Any]]
    tools: List[ToolClass]             # by class index
    specials: List[Dict[str, Any]]     # 41 entries
    stat_boosts: List[Dict[str, Any]]  # variable
    mag_feeds: List[MagFeedSubTable]   # 8 sub-tables × 11 results
    combinations: List[Dict[str, Any]]
    v1_replacement: List[int]          # u8 array
    weapon_sale_divisors: List[float]
    sale_divisors: Dict[str, Any]      # {armor, shield, unit, mag}
    star_values: List[int]             # u8 array (covers 0xB1..0x437)
    max_tech_levels: List[List[int]]   # 19 × 12
    # Opaque sections preserved verbatim for byte-exact round-trip.
    _opaque: Dict[str, Any] = field(default_factory=dict)
    # Original-file metadata (used only for re-emit; not edited).
    _meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return _scrub_nonfinite({
            "header_preamble": self.header_preamble,
            "weapons": [
                {"class_index": wc.class_index, "name": wc.name, "items": wc.items}
                for wc in self.weapons
            ],
            "armors": self.armors,
            "shields": self.shields,
            "units": self.units,
            "mags": self.mags,
            "tools": [
                {"class_index": tc.class_index, "name": tc.name, "items": tc.items}
                for tc in self.tools
            ],
            "specials": self.specials,
            "stat_boosts": self.stat_boosts,
            "mag_feeds": [
                {"table_index": mf.table_index, "results": mf.results}
                for mf in self.mag_feeds
            ],
            "combinations": self.combinations,
            "v1_replacement": self.v1_replacement,
            "weapon_sale_divisors": self.weapon_sale_divisors,
            "sale_divisors": self.sale_divisors,
            "star_values": self.star_values,
            "max_tech_levels": self.max_tech_levels,
            "_opaque": dict(self._opaque),
            "_meta": dict(self._meta),
        })

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "ItemPMTFile":
        return cls(
            header_preamble=data["header_preamble"],
            weapons=[
                WeaponClass(
                    class_index=int(w["class_index"]),
                    name=str(w["name"]),
                    items=list(w["items"]),
                )
                for w in data.get("weapons", [])
            ],
            armors=list(data.get("armors", [])),
            shields=list(data.get("shields", [])),
            units=list(data.get("units", [])),
            mags=list(data.get("mags", [])),
            tools=[
                ToolClass(
                    class_index=int(t["class_index"]),
                    name=str(t["name"]),
                    items=list(t["items"]),
                )
                for t in data.get("tools", [])
            ],
            specials=list(data.get("specials", [])),
            stat_boosts=list(data.get("stat_boosts", [])),
            mag_feeds=[
                MagFeedSubTable(
                    table_index=int(m["table_index"]),
                    results=list(m["results"]),
                )
                for m in data.get("mag_feeds", [])
            ],
            combinations=list(data.get("combinations", [])),
            v1_replacement=list(data.get("v1_replacement", [])),
            weapon_sale_divisors=list(data.get("weapon_sale_divisors", [])),
            sale_divisors=dict(data.get("sale_divisors", {})),
            star_values=list(data.get("star_values", [])),
            max_tech_levels=[list(row) for row in data.get("max_tech_levels", [])],
            _opaque=dict(data.get("_opaque", {})),
            _meta=dict(data.get("_meta", {})),
        )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _read_array_ref(buf: bytes, off: int) -> Tuple[int, int]:
    """Read newserv ArrayRef = (count: u32, offset: u32)."""
    return struct.unpack_from("<II", buf, off)


def _read_offsets(buf: bytes) -> Tuple[int, Tuple[int, ...]]:
    """Return (offset_table_offset, tuple of 23 u32 offsets)."""
    if len(buf) < 0x10:
        raise ValueError("ItemPMT file too short for footer")
    oto = struct.unpack_from("<I", buf, len(buf) - 0x10)[0]
    if oto + TABLE_OFFSETS_SIZE > len(buf):
        raise ValueError(
            f"offset_table_offset 0x{oto:x} out of range for size 0x{len(buf):x}"
        )
    offsets = struct.unpack_from(f"<{NUM_TABLE_OFFSETS}I", buf, oto)
    return oto, offsets


# Field index into TableOffsetsV3V4 (each is u32 at i*4)
TO_WEAPON = 0
TO_ARMOR = 1
TO_UNIT = 2
TO_TOOL = 3
TO_MAG = 4
TO_V1_REPLACEMENT = 5
TO_PHOTON_COLOR = 6
TO_WEAPON_RANGE = 7
TO_WEAPON_SALE_DIVISOR = 8
TO_SALE_DIVISOR = 9
TO_MAG_FEED = 10
TO_STAR_VALUE = 11
TO_SPECIAL_DATA = 12
TO_WEAPON_EFFECT = 13
TO_STAT_BOOST = 14
TO_SHIELD_EFFECT = 15
TO_MAX_TECH_LEVEL = 16
TO_COMBINATION = 17
TO_UNKNOWN_A1 = 18
TO_TECH_BOOST = 19
TO_UNWRAP = 20
TO_UNSEALABLE = 21
TO_RANGED_SPECIAL = 22


def parse(buf: bytes) -> ItemPMTFile:
    """Parse the *decompressed* ItemPMT-bb-v4 byte stream.

    Args:
        buf: raw uncompressed bytes (you must PRS-decompress the .prs
            file first).

    Returns:
        ItemPMTFile

    Raises:
        ValueError: malformed or truncated input.
    """
    if not isinstance(buf, (bytes, bytearray, memoryview)):
        raise ValueError("parse: input must be bytes-like")
    buf = bytes(buf)
    if len(buf) < HEADER_PREAMBLE_SIZE + TABLE_OFFSETS_SIZE + 0x10:
        raise ValueError(f"buffer too small ({len(buf)} bytes)")

    oto, off = _read_offsets(buf)

    # ---- Item sections ----------------------------------------------------
    # weapon_table at off[TO_WEAPON]: starts at e.g. 0x1489c, points to an
    # array of ArrayRefs (one per weapon class). Slot count is determined
    # by the gap to the next adjacent table in offset-sorted order.
    # We use the *offsets table itself* as the upper bound because all
    # weapon ArrayRefs are stored before it.
    weapon_table_start = off[TO_WEAPON]
    # Compute end: the smallest offset > weapon_table_start, capped at oto.
    candidates_after_weapon = [o for o in off if o > weapon_table_start] + [oto]
    weapon_table_end = min(candidates_after_weapon)
    num_weapon_classes = (weapon_table_end - weapon_table_start) // 8

    weapons: List[WeaponClass] = []
    for cls_i in range(num_weapon_classes):
        ar_off = weapon_table_start + cls_i * 8
        count, data_off = _read_array_ref(buf, ar_off)
        items: List[Dict[str, Any]] = []
        if count > 0 and data_off > 0 and data_off < len(buf):
            for j in range(count):
                items.append(
                    _parse_record(buf, data_off + j * WEAPON_RECORD_SIZE, WEAPON_SCHEMA)
                )
        weapons.append(WeaponClass(
            class_index=cls_i,
            name=WEAPON_CLASS_NAMES.get(cls_i, f"class_{cls_i:02X}"),
            items=items,
        ))

    # armor_table is an array of 2 ArrayRefs: [0]=frames, [1]=barriers.
    armor_root = off[TO_ARMOR]
    frame_ar = _read_array_ref(buf, armor_root)
    shield_ar = _read_array_ref(buf, armor_root + 8)
    armors: List[Dict[str, Any]] = []
    for j in range(frame_ar[0]):
        armors.append(_parse_record(
            buf, frame_ar[1] + j * ARMOR_OR_SHIELD_RECORD_SIZE,
            ARMOR_OR_SHIELD_SCHEMA,
        ))
    shields: List[Dict[str, Any]] = []
    for j in range(shield_ar[0]):
        shields.append(_parse_record(
            buf, shield_ar[1] + j * ARMOR_OR_SHIELD_RECORD_SIZE,
            ARMOR_OR_SHIELD_SCHEMA,
        ))

    # unit_table: single ArrayRef
    unit_ar = _read_array_ref(buf, off[TO_UNIT])
    units: List[Dict[str, Any]] = []
    for j in range(unit_ar[0]):
        units.append(_parse_record(
            buf, unit_ar[1] + j * UNIT_RECORD_SIZE, UNIT_SCHEMA
        ))

    # mag_table: single ArrayRef
    mag_ar = _read_array_ref(buf, off[TO_MAG])
    mags: List[Dict[str, Any]] = []
    for j in range(mag_ar[0]):
        mags.append(_parse_record(
            buf, mag_ar[1] + j * MAG_RECORD_SIZE, MAG_SCHEMA
        ))

    # tool_table: 27 ArrayRefs (V4 num_tool_classes = 0x1B). The exact
    # count isn't in the file — we infer from the gap to weapon_table.
    tool_table_start = off[TO_TOOL]
    num_tool_classes = (off[TO_WEAPON] - tool_table_start) // 8
    tools: List[ToolClass] = []
    for cls_i in range(num_tool_classes):
        ar_off = tool_table_start + cls_i * 8
        count, data_off = _read_array_ref(buf, ar_off)
        items_t: List[Dict[str, Any]] = []
        if count > 0 and data_off > 0 and data_off < len(buf):
            for j in range(count):
                items_t.append(_parse_record(
                    buf, data_off + j * TOOL_RECORD_SIZE, TOOL_SCHEMA
                ))
        tools.append(ToolClass(
            class_index=cls_i,
            name=TOOL_CLASS_NAMES.get(cls_i, f"class_{cls_i:02X}"),
            items=items_t,
        ))

    # ---- Lookup tables ----------------------------------------------------
    # We compute table size as offset-to-next-table (from the sorted offset
    # list) — this gives us the exact byte slice each table occupies.
    sorted_offs = sorted(set([o for o in off if o > 0]) | {oto, len(buf) - 0x10})
    def section_end(start: int) -> int:
        for s in sorted_offs:
            if s > start:
                return s
        return len(buf)

    # special_data: NUM_SPECIALS × SPECIAL_RECORD_SIZE
    sd_start = off[TO_SPECIAL_DATA]
    sd_size = section_end(sd_start) - sd_start
    sd_count = sd_size // SPECIAL_RECORD_SIZE
    specials: List[Dict[str, Any]] = []
    for j in range(sd_count):
        specials.append(_parse_record(
            buf, sd_start + j * SPECIAL_RECORD_SIZE, SPECIAL_SCHEMA
        ))

    # stat_boost
    sb_start = off[TO_STAT_BOOST]
    sb_size = section_end(sb_start) - sb_start
    sb_count = sb_size // STAT_BOOST_RECORD_SIZE
    stat_boosts: List[Dict[str, Any]] = []
    for j in range(sb_count):
        stat_boosts.append(_parse_record(
            buf, sb_start + j * STAT_BOOST_RECORD_SIZE, STAT_BOOST_SCHEMA
        ))

    # combination_table: ArrayRef → ItemCombination[]
    comb_ar = _read_array_ref(buf, off[TO_COMBINATION])
    combinations: List[Dict[str, Any]] = []
    for j in range(comb_ar[0]):
        combinations.append(_parse_record(
            buf, comb_ar[1] + j * ITEM_COMBINATION_RECORD_SIZE,
            ITEM_COMBINATION_SCHEMA,
        ))

    # v1_replacement_table: u8 array; size = section size
    v1r_start = off[TO_V1_REPLACEMENT]
    v1r_size = section_end(v1r_start) - v1r_start
    v1_replacement = list(buf[v1r_start:v1r_start + v1r_size])

    # weapon_sale_divisor_table: float array
    wsd_start = off[TO_WEAPON_SALE_DIVISOR]
    wsd_size = section_end(wsd_start) - wsd_start
    wsd_count = wsd_size // 4
    weapon_sale_divisors = list(struct.unpack_from(
        f"<{wsd_count}f", buf, wsd_start
    ))
    # Float NaN/Inf preservation via parallel _bits cache stored under _meta.
    weapon_sale_divisor_bits = list(struct.unpack_from(
        f"<{wsd_count}I", buf, wsd_start
    ))

    # sale_divisor_table: 4 floats. Note: the byte slice up to
    # star_value_table contains both the NonWeaponSaleDivisors *and* the
    # 8 mag-feed sub-tables (the latter referenced via mag_feed_table's
    # MagFeedResultsListOffsets). The first 16 bytes are the divisors.
    sd_div_start = off[TO_SALE_DIVISOR]
    sale_divisors = _parse_record(buf, sd_div_start, SALE_DIVISORS_SCHEMA)

    # star_value_table
    sv_start = off[TO_STAR_VALUE]
    sv_size = section_end(sv_start) - sv_start
    star_values = list(buf[sv_start:sv_start + sv_size])

    # max_tech_level grid (19 × 12 = 228 bytes; the rest is opaque pad)
    mtl_start = off[TO_MAX_TECH_LEVEL]
    mtl_total = section_end(mtl_start) - mtl_start
    mtl_grid_bytes = MAX_TECH_NUM_TECHS * MAX_TECH_NUM_CLASSES
    if mtl_total < mtl_grid_bytes:
        mtl_grid_bytes = mtl_total
    max_tech_levels: List[List[int]] = []
    for tech in range(MAX_TECH_NUM_TECHS):
        row_start = mtl_start + tech * MAX_TECH_NUM_CLASSES
        if row_start + MAX_TECH_NUM_CLASSES > mtl_start + mtl_total:
            max_tech_levels.append([0] * MAX_TECH_NUM_CLASSES)
            continue
        max_tech_levels.append(list(buf[row_start:row_start + MAX_TECH_NUM_CLASSES]))
    mtl_pad_start = mtl_start + mtl_grid_bytes
    mtl_pad = buf[mtl_pad_start:mtl_start + mtl_total]

    # mag_feed: at off[TO_MAG_FEED] is MagFeedResultsListOffsets (8 × u32).
    mfo_start = off[TO_MAG_FEED]
    mfo_offsets = struct.unpack_from(
        f"<{MAG_FEED_NUM_TABLES}I", buf, mfo_start
    )
    mag_feeds: List[MagFeedSubTable] = []
    for ti, sub_off in enumerate(mfo_offsets):
        results: List[Dict[str, Any]] = []
        for r in range(MAG_FEED_RESULTS_PER_TABLE):
            results.append(_parse_record(
                buf, sub_off + r * MAG_FEED_RESULT_SIZE,
                MAG_FEED_RESULT_SCHEMA,
            ))
        mag_feeds.append(MagFeedSubTable(table_index=ti, results=results))

    # ---- Opaque sections -------------------------------------------------
    # We capture each opaque section as hex of its raw bytes. On
    # serialization we paste them back verbatim. This is what guarantees
    # byte-exact round-trip even for tables whose internal structure isn't
    # parsed yet (photon_color_table, weapon_range_table, weapon/shield
    # effect tables, tech boost, unwrap tree, etc).
    def hex_section(start: int) -> str:
        end = section_end(start)
        return buf[start:end].hex()

    photon_color_blob = hex_section(off[TO_PHOTON_COLOR])
    weapon_range_blob = hex_section(off[TO_WEAPON_RANGE])
    weapon_effect_blob = hex_section(off[TO_WEAPON_EFFECT])
    shield_effect_blob = hex_section(off[TO_SHIELD_EFFECT])
    unknown_a1_blob = hex_section(off[TO_UNKNOWN_A1])
    tech_boost_blob = hex_section(off[TO_TECH_BOOST])

    # Small tables at the end (unwrap, unsealable, ranged_special) — keep
    # opaque too. Each is an ArrayRef-style structure; we record the
    # ArrayRef header + its referenced data as a single contiguous slice.
    def opaque_arrayref_chain(top_off: int) -> Dict[str, Any]:
        """For a {count, offset} top-level ArrayRef, capture the header
        and the byte slice it references. Returns
        {"header": "<hex of 8 bytes>", "data": "<hex of count*N bytes>"}."""
        count, sub_off = _read_array_ref(buf, top_off)
        header_hex = buf[top_off:top_off + 8].hex()
        # Determine end of referenced data via section_end of sub_off
        data_end = section_end(sub_off)
        return {
            "header": header_hex,
            "count": count,
            "sub_offset": sub_off,
            "data": buf[sub_off:data_end].hex(),
        }

    unwrap_blob = opaque_arrayref_chain(off[TO_UNWRAP])
    unsealable_blob = opaque_arrayref_chain(off[TO_UNSEALABLE])
    ranged_special_blob = opaque_arrayref_chain(off[TO_RANGED_SPECIAL])

    # Footer offsets table verbatim
    footer_offsets_blob = buf[oto:oto + TABLE_OFFSETS_SIZE].hex()
    # Trailing padding (between offset table end and offset_table_offset
    # in the last 16 bytes)
    trailing_blob = buf[oto + TABLE_OFFSETS_SIZE:].hex()

    opaque: Dict[str, Any] = {
        "header_preamble":      buf[:HEADER_PREAMBLE_SIZE].hex(),
        "photon_color_table":   photon_color_blob,
        "weapon_range_table":   weapon_range_blob,
        "weapon_effect_table":  weapon_effect_blob,
        "shield_effect_table":  shield_effect_blob,
        "unknown_a1":           unknown_a1_blob,
        "tech_boost_table":     tech_boost_blob,
        "max_tech_level_pad":   mtl_pad.hex(),
        "unwrap_table":         unwrap_blob,
        "unsealable_table":     unsealable_blob,
        "ranged_special_table": ranged_special_blob,
        "footer_offsets":       footer_offsets_blob,
        "trailing":             trailing_blob,
        # NaN/Inf preservation for weapon_sale_divisors
        "_weapon_sale_divisor_bits": weapon_sale_divisor_bits,
    }

    meta: Dict[str, Any] = {
        "file_size":             len(buf),
        "offset_table_offset":   oto,
        "num_weapon_classes":    num_weapon_classes,
        "num_tool_classes":      num_tool_classes,
        "section_offsets":       {
            "weapon_table":             off[TO_WEAPON],
            "armor_table":              off[TO_ARMOR],
            "unit_table":               off[TO_UNIT],
            "tool_table":               off[TO_TOOL],
            "mag_table":                off[TO_MAG],
            "v1_replacement_table":     off[TO_V1_REPLACEMENT],
            "photon_color_table":       off[TO_PHOTON_COLOR],
            "weapon_range_table":       off[TO_WEAPON_RANGE],
            "weapon_sale_divisor_table": off[TO_WEAPON_SALE_DIVISOR],
            "sale_divisor_table":       off[TO_SALE_DIVISOR],
            "mag_feed_table":           off[TO_MAG_FEED],
            "star_value_table":         off[TO_STAR_VALUE],
            "special_data_table":       off[TO_SPECIAL_DATA],
            "weapon_effect_table":      off[TO_WEAPON_EFFECT],
            "stat_boost_table":         off[TO_STAT_BOOST],
            "shield_effect_table":      off[TO_SHIELD_EFFECT],
            "max_tech_level_table":     off[TO_MAX_TECH_LEVEL],
            "combination_table":        off[TO_COMBINATION],
            "unknown_a1":               off[TO_UNKNOWN_A1],
            "tech_boost_table":         off[TO_TECH_BOOST],
            "unwrap_table":             off[TO_UNWRAP],
            "unsealable_table":         off[TO_UNSEALABLE],
            "ranged_special_table":     off[TO_RANGED_SPECIAL],
        },
    }

    return ItemPMTFile(
        header_preamble=buf[:HEADER_PREAMBLE_SIZE].hex(),
        weapons=weapons,
        armors=armors,
        shields=shields,
        units=units,
        mags=mags,
        tools=tools,
        specials=specials,
        stat_boosts=stat_boosts,
        mag_feeds=mag_feeds,
        combinations=combinations,
        v1_replacement=v1_replacement,
        weapon_sale_divisors=weapon_sale_divisors,
        sale_divisors=sale_divisors,
        star_values=star_values,
        max_tech_levels=max_tech_levels,
        _opaque=opaque,
        _meta=meta,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
def serialize(pmt: ItemPMTFile) -> bytes:
    """Serialize an ItemPMTFile back to the raw decompressed byte stream.

    Strategy: we follow the layout-preserving approach. The original
    file's byte slice for every section is captured during parse — we
    re-emit the same offsets and patch in the typed-record bytes for
    the sections we know how to write.

    For round-trip-without-edit, this produces a byte-identical file.
    For round-trip-with-edits, only the typed records change; opaque
    sections stay verbatim.
    """
    if not pmt._meta or "file_size" not in pmt._meta:
        raise ValueError(
            "serialize: ItemPMTFile is missing _meta — was it built from "
            "a fresh from_json() without going through parse()? "
            "Re-parse the original file before editing."
        )
    file_size = int(pmt._meta["file_size"])
    section_offs = pmt._meta["section_offsets"]
    out = bytearray(file_size)

    # ---- Header preamble (verbatim) --------------------------------------
    preamble = bytes.fromhex(pmt.header_preamble)
    if len(preamble) != HEADER_PREAMBLE_SIZE:
        raise ValueError(
            f"serialize: header_preamble is {len(preamble)} bytes, "
            f"expected {HEADER_PREAMBLE_SIZE}"
        )
    out[:HEADER_PREAMBLE_SIZE] = preamble

    # ---- Item entries: written at original offsets -----------------------
    # We honor the original on-disk layout (shields before frames in the
    # actual file). Each entry is written via _pack_record at its
    # original offset.
    armor_root = section_offs["armor_table"]
    frame_count, frame_off = _read_array_ref(out, armor_root) if False else (0, 0)
    # We don't have the ArrayRef counts in _meta directly, so re-emit
    # the table headers below; for now, write the entry bytes at the
    # offsets stored in our parsed lists.

    # Strategy: we need to know data_off + count for each ArrayRef to
    # know where each item slot lives. The cleanest source is the
    # opaque footer_offsets blob (the original TableOffsetsV3V4 we
    # captured), plus the ArrayRefs in the file headers we keep
    # verbatim. We'll write items based on the *original* ArrayRef
    # positions — which means we need to re-emit the header section
    # that holds the ArrayRefs first, copying from a scratch slice.

    # The simplest correct approach: write all opaque sections (they
    # cover the ArrayRef header areas + any non-edited bytes), THEN
    # overwrite each item's bytes at its proper offset.

    # Step 1: write opaque sections
    _write_opaque_sections(out, pmt)

    # Step 2: now ArrayRefs (which live in our opaque headers) are
    # in place. Iterate them and write item entries.
    _write_typed_items(out, pmt, section_offs)

    # Step 3: write typed lookup tables
    _write_typed_lookups(out, pmt, section_offs)

    # Step 4: write footer offsets table + offset_table_offset
    oto = int(pmt._meta["offset_table_offset"])
    footer_offsets = bytes.fromhex(pmt._opaque.get("footer_offsets", ""))
    if len(footer_offsets) != TABLE_OFFSETS_SIZE:
        raise ValueError(
            f"serialize: footer_offsets is {len(footer_offsets)}, "
            f"expected {TABLE_OFFSETS_SIZE}"
        )
    out[oto:oto + TABLE_OFFSETS_SIZE] = footer_offsets
    trailing = bytes.fromhex(pmt._opaque.get("trailing", ""))
    out[oto + TABLE_OFFSETS_SIZE:oto + TABLE_OFFSETS_SIZE + len(trailing)] = trailing

    return bytes(out)


def _write_opaque_sections(out: bytearray, pmt: ItemPMTFile) -> None:
    """Patch opaque blobs into ``out`` at their original offsets.

    Every byte of the file outside our typed-record coverage gets set
    here. After this the only bytes we still need to overwrite are the
    typed records (items, specials, stat_boosts, combinations,
    mag_feeds, sale_divisors etc).
    """
    section_offs = pmt._meta["section_offsets"]
    file_size = len(out)

    # The simplest way: paste *every* original byte slice for sections
    # we don't fully re-emit. We have the original buffer encoded only
    # piecewise via _opaque. To avoid a full-buffer hex round-trip we
    # rebuild section-by-section.

    def write_hex(off: int, hexstr: str, section_name: str = "") -> None:
        b = bytes.fromhex(hexstr)
        if off + len(b) > file_size:
            raise ValueError(
                f"_write_opaque_sections: blob {section_name!r} ({len(b)} bytes) "
                f"at 0x{off:x} overflows file size {file_size}"
            )
        out[off:off + len(b)] = b

    write_hex(0, pmt._opaque["header_preamble"], "header_preamble")
    write_hex(section_offs["photon_color_table"],
              pmt._opaque["photon_color_table"], "photon_color_table")
    write_hex(section_offs["weapon_range_table"],
              pmt._opaque["weapon_range_table"], "weapon_range_table")
    write_hex(section_offs["weapon_effect_table"],
              pmt._opaque["weapon_effect_table"], "weapon_effect_table")
    write_hex(section_offs["shield_effect_table"],
              pmt._opaque["shield_effect_table"], "shield_effect_table")
    write_hex(section_offs["unknown_a1"],
              pmt._opaque["unknown_a1"], "unknown_a1")
    write_hex(section_offs["tech_boost_table"],
              pmt._opaque["tech_boost_table"], "tech_boost_table")

    # Item table headers (ArrayRef arrays) live in the area between
    # tech_boost_table end and the typed lookup tables. We capture them
    # via the same opaque-blob mechanism by emitting section slices
    # whose content we don't otherwise know.
    # Specifically: armor_table (16 bytes), unit_table (8), mag_table (8),
    # tool_table (27 ArrayRefs), weapon_table (N ArrayRefs).
    # These ArrayRefs ARE part of our parsed schema, so we re-emit them
    # from our parsed counts/offsets — see _emit_arrayref_headers().
    _emit_arrayref_headers(out, pmt)

    # max_tech_level_pad goes after the 19×12 grid, before the next
    # section. The grid itself is written by _write_typed_lookups.
    mtl_start = section_offs["max_tech_level_table"]
    mtl_grid_bytes = MAX_TECH_NUM_TECHS * MAX_TECH_NUM_CLASSES
    pad_hex = pmt._opaque.get("max_tech_level_pad", "")
    if pad_hex:
        write_hex(mtl_start + mtl_grid_bytes, pad_hex, "max_tech_level_pad")

    # Small ArrayRef-chain blobs: unwrap_table, unsealable_table,
    # ranged_special_table. Each carries its 8-byte header AND the
    # referenced data.
    for key in ("unwrap_table", "unsealable_table", "ranged_special_table"):
        blob = pmt._opaque.get(key)
        if not blob:
            continue
        top_off = section_offs[key]
        write_hex(top_off, blob["header"], key + ".header")
        write_hex(int(blob["sub_offset"]), blob["data"], key + ".data")

    # mag_feed_table sits at section_offs["mag_feed_table"] and points
    # to MagFeedResultsListOffsets — we re-emit from typed mag_feeds.
    _emit_mag_feed_offsets(out, pmt)


def _emit_arrayref_headers(out: bytearray, pmt: ItemPMTFile) -> None:
    """Write the per-section ArrayRef header arrays.

    Layout:
      armor_table at section_offs["armor_table"]:
          ArrayRef[0] = (count=len(armors),  offset=<original frame data offset>)
          ArrayRef[1] = (count=len(shields), offset=<original shield data offset>)
      unit_table at section_offs["unit_table"]:
          ArrayRef = (count=len(units), offset=<original units data offset>)
      mag_table at section_offs["mag_table"]:
          ArrayRef = (count=len(mags), offset=<original mags data offset>)
      tool_table at section_offs["tool_table"]:
          ArrayRef[i] = (count=len(tools[i].items), offset=<original tool[i] offset>)
      weapon_table at section_offs["weapon_table"]:
          ArrayRef[i] = (count=len(weapons[i].items), offset=<original weapon[i] offset>)

    We pull the original *offset* values from the typed item data —
    they're not stored separately, so we recover them from the parsed
    file's position information cached in _meta. If counts diverge from
    the originals (i.e. items were added/removed) the ArrayRefs still
    point to the original offsets, which is OK for round-trip but
    incorrect for new items — that's a future-work problem.

    For the round-trip case the ArrayRef contents are identical, so
    this is byte-exact.
    """
    section_offs = pmt._meta["section_offsets"]
    item_offsets = pmt._meta.get("item_offsets")
    if not item_offsets:
        # First serialize call — build from the parsed input so we know
        # where each items array lives. We use the data positions
        # captured during parse via separate keys.
        return
    # Helper
    def write_arrayref(off: int, count: int, data_off: int) -> None:
        struct.pack_into("<II", out, off, count, data_off)

    # weapon
    wt_start = section_offs["weapon_table"]
    for i, wc in enumerate(pmt.weapons):
        offs_i = item_offsets["weapons"].get(str(i)) or item_offsets["weapons"].get(i)
        if offs_i is None:
            # Empty class — write 0,0
            write_arrayref(wt_start + i * 8, 0, 0)
        else:
            write_arrayref(wt_start + i * 8, len(wc.items), offs_i)

    # tools
    tt_start = section_offs["tool_table"]
    for i, tc in enumerate(pmt.tools):
        offs_i = item_offsets["tools"].get(str(i)) or item_offsets["tools"].get(i)
        if offs_i is None:
            write_arrayref(tt_start + i * 8, 0, 0)
        else:
            write_arrayref(tt_start + i * 8, len(tc.items), offs_i)

    # armor (frames + shields)
    arm_start = section_offs["armor_table"]
    write_arrayref(arm_start,
                   len(pmt.armors),
                   item_offsets["armors"])
    write_arrayref(arm_start + 8,
                   len(pmt.shields),
                   item_offsets["shields"])
    write_arrayref(section_offs["unit_table"],
                   len(pmt.units),
                   item_offsets["units"])
    write_arrayref(section_offs["mag_table"],
                   len(pmt.mags),
                   item_offsets["mags"])
    # combination_table is a single ArrayRef
    write_arrayref(section_offs["combination_table"],
                   len(pmt.combinations),
                   item_offsets["combinations"])


def _emit_mag_feed_offsets(out: bytearray, pmt: ItemPMTFile) -> None:
    """Write the MagFeedResultsListOffsets header (8 × u32) and each sub-table."""
    section_offs = pmt._meta["section_offsets"]
    item_offsets = pmt._meta.get("item_offsets") or {}
    sub_offs = item_offsets.get("mag_feed_subs")
    if not sub_offs:
        return
    mfo_start = section_offs["mag_feed_table"]
    # Write the offsets header (8 u32 to the sub-tables)
    if len(sub_offs) != MAG_FEED_NUM_TABLES:
        raise ValueError(
            f"_emit_mag_feed_offsets: expected {MAG_FEED_NUM_TABLES} sub-tables, "
            f"got {len(sub_offs)}"
        )
    struct.pack_into(f"<{MAG_FEED_NUM_TABLES}I", out, mfo_start, *sub_offs)
    # Sub-tables are written by _write_typed_lookups.


def _write_typed_items(
    out: bytearray, pmt: ItemPMTFile, section_offs: Dict[str, int]
) -> None:
    """Patch typed item bytes (weapons/armors/shields/units/mags/tools)
    at their original offsets recorded in _meta["item_offsets"]."""
    item_offsets = pmt._meta.get("item_offsets")
    if not item_offsets:
        return  # nothing to do (parser must have failed to record)

    # Weapons
    for i, wc in enumerate(pmt.weapons):
        base = item_offsets["weapons"].get(str(i)) or item_offsets["weapons"].get(i)
        if base is None:
            continue
        for j, rec in enumerate(wc.items):
            off = base + j * WEAPON_RECORD_SIZE
            _patch(out, off, _pack_record(rec, WEAPON_SCHEMA))

    # Tools
    for i, tc in enumerate(pmt.tools):
        base = item_offsets["tools"].get(str(i)) or item_offsets["tools"].get(i)
        if base is None:
            continue
        for j, rec in enumerate(tc.items):
            off = base + j * TOOL_RECORD_SIZE
            _patch(out, off, _pack_record(rec, TOOL_SCHEMA))

    # Armors / shields
    base = item_offsets["armors"]
    for j, rec in enumerate(pmt.armors):
        off = base + j * ARMOR_OR_SHIELD_RECORD_SIZE
        _patch(out, off, _pack_record(rec, ARMOR_OR_SHIELD_SCHEMA))
    base = item_offsets["shields"]
    for j, rec in enumerate(pmt.shields):
        off = base + j * ARMOR_OR_SHIELD_RECORD_SIZE
        _patch(out, off, _pack_record(rec, ARMOR_OR_SHIELD_SCHEMA))

    # Units
    base = item_offsets["units"]
    for j, rec in enumerate(pmt.units):
        off = base + j * UNIT_RECORD_SIZE
        _patch(out, off, _pack_record(rec, UNIT_SCHEMA))

    # Mags
    base = item_offsets["mags"]
    for j, rec in enumerate(pmt.mags):
        off = base + j * MAG_RECORD_SIZE
        _patch(out, off, _pack_record(rec, MAG_SCHEMA))

    # Combinations
    base = item_offsets["combinations"]
    for j, rec in enumerate(pmt.combinations):
        off = base + j * ITEM_COMBINATION_RECORD_SIZE
        _patch(out, off, _pack_record(rec, ITEM_COMBINATION_SCHEMA))


def _write_typed_lookups(
    out: bytearray, pmt: ItemPMTFile, section_offs: Dict[str, int]
) -> None:
    """Write typed lookup tables (specials, stat_boosts, mag_feeds,
    v1_replacement, weapon_sale_divisors, sale_divisors, star_values,
    max_tech_levels)."""
    # specials
    sd = section_offs["special_data_table"]
    for j, rec in enumerate(pmt.specials):
        _patch(out, sd + j * SPECIAL_RECORD_SIZE,
               _pack_record(rec, SPECIAL_SCHEMA))

    # stat_boosts
    sb = section_offs["stat_boost_table"]
    for j, rec in enumerate(pmt.stat_boosts):
        _patch(out, sb + j * STAT_BOOST_RECORD_SIZE,
               _pack_record(rec, STAT_BOOST_SCHEMA))

    # mag_feeds
    item_offsets = pmt._meta.get("item_offsets") or {}
    sub_offs = item_offsets.get("mag_feed_subs")
    if sub_offs:
        for ti, mft in enumerate(pmt.mag_feeds):
            base = sub_offs[ti]
            for j, rec in enumerate(mft.results):
                _patch(out, base + j * MAG_FEED_RESULT_SIZE,
                       _pack_record(rec, MAG_FEED_RESULT_SCHEMA))

    # v1_replacement: u8 array
    v1r = section_offs["v1_replacement_table"]
    _patch(out, v1r, bytes(pmt.v1_replacement))

    # weapon_sale_divisor: floats (with bits sidecar)
    wsd = section_offs["weapon_sale_divisor_table"]
    bits = pmt._opaque.get("_weapon_sale_divisor_bits")
    if bits and len(bits) == len(pmt.weapon_sale_divisors):
        _patch(out, wsd, struct.pack(
            f"<{len(bits)}I", *(b & 0xFFFFFFFF for b in bits)
        ))
    else:
        # No bits sidecar — use float values, replacing NaN/None with 0.
        floats = [0.0 if (v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))))
                  else float(v) for v in pmt.weapon_sale_divisors]
        _patch(out, wsd, struct.pack(f"<{len(floats)}f", *floats))

    # sale_divisors: 4 floats
    sd_div = section_offs["sale_divisor_table"]
    _patch(out, sd_div, _pack_record(pmt.sale_divisors, SALE_DIVISORS_SCHEMA))

    # star_values: u8 array
    sv = section_offs["star_value_table"]
    _patch(out, sv, bytes(pmt.star_values))

    # max_tech_levels: 19 × 12 grid
    mtl = section_offs["max_tech_level_table"]
    grid_bytes = b""
    for row in pmt.max_tech_levels:
        if len(row) != MAX_TECH_NUM_CLASSES:
            raise ValueError(
                f"max_tech_levels row must have {MAX_TECH_NUM_CLASSES} entries"
            )
        grid_bytes += bytes(row)
    if len(pmt.max_tech_levels) != MAX_TECH_NUM_TECHS:
        raise ValueError(
            f"max_tech_levels must have {MAX_TECH_NUM_TECHS} rows"
        )
    _patch(out, mtl, grid_bytes)


def _patch(out: bytearray, off: int, data: bytes) -> None:
    if off + len(data) > len(out):
        raise ValueError(
            f"_patch: write at 0x{off:x} of {len(data)} bytes overflows "
            f"output of size {len(out)}"
        )
    out[off:off + len(data)] = data


# ---------------------------------------------------------------------------
# JSON helpers + cached item offsets
# ---------------------------------------------------------------------------
def _record_item_offsets(buf: bytes, pmt: ItemPMTFile) -> None:
    """After parse(), record each item array's data offset in _meta so
    serialize() can find it without re-walking the original buffer.
    Mutates ``pmt`` in place."""
    section_offs = pmt._meta["section_offsets"]
    item_offsets: Dict[str, Any] = {
        "weapons": {},
        "tools": {},
        "armors": 0,
        "shields": 0,
        "units": 0,
        "mags": 0,
        "combinations": 0,
        "mag_feed_subs": [],
    }
    # Weapons
    wt = section_offs["weapon_table"]
    for i in range(len(pmt.weapons)):
        count, data_off = _read_array_ref(buf, wt + i * 8)
        if count > 0:
            item_offsets["weapons"][i] = data_off
        else:
            item_offsets["weapons"][i] = 0
    # Tools
    tt = section_offs["tool_table"]
    for i in range(len(pmt.tools)):
        count, data_off = _read_array_ref(buf, tt + i * 8)
        if count > 0:
            item_offsets["tools"][i] = data_off
        else:
            item_offsets["tools"][i] = 0
    # Armor / shield
    armor_root = section_offs["armor_table"]
    _, frame_off = _read_array_ref(buf, armor_root)
    _, shield_off = _read_array_ref(buf, armor_root + 8)
    item_offsets["armors"] = frame_off
    item_offsets["shields"] = shield_off
    _, units_off = _read_array_ref(buf, section_offs["unit_table"])
    item_offsets["units"] = units_off
    _, mags_off = _read_array_ref(buf, section_offs["mag_table"])
    item_offsets["mags"] = mags_off
    _, comb_off = _read_array_ref(buf, section_offs["combination_table"])
    item_offsets["combinations"] = comb_off
    # mag_feeds
    mfo_start = section_offs["mag_feed_table"]
    sub_offs = list(struct.unpack_from(
        f"<{MAG_FEED_NUM_TABLES}I", buf, mfo_start
    ))
    item_offsets["mag_feed_subs"] = sub_offs
    pmt._meta["item_offsets"] = item_offsets


def _parse_with_offsets(buf: bytes) -> ItemPMTFile:
    pmt = parse(buf)
    _record_item_offsets(buf, pmt)
    return pmt


# ---------------------------------------------------------------------------
# Public top-level API
# ---------------------------------------------------------------------------
def to_json(pmt: ItemPMTFile) -> str:
    """Serialize an ItemPMTFile to pretty JSON."""
    return json.dumps(pmt.to_json(), indent=2, sort_keys=False)


def from_json(text: str) -> ItemPMTFile:
    """Deserialize an ItemPMTFile from JSON text."""
    return ItemPMTFile.from_json(json.loads(text))


def parse_prs(prs_bytes: bytes) -> ItemPMTFile:
    """Convenience wrapper: PRS-decompress and parse."""
    from formats import prs as prs_mod
    return parse_with_meta(prs_mod.decompress(prs_bytes))


def parse_with_meta(buf: bytes) -> ItemPMTFile:
    """Like :func:`parse` but also caches per-item byte offsets so
    serialize() works correctly. Always use this for the round-trip flow."""
    return _parse_with_offsets(buf)


def pack(pmt: ItemPMTFile) -> bytes:
    """Serialize the ItemPMT and PRS-compress it (the form newserv loads)."""
    from formats import prs as prs_mod
    raw = serialize(pmt)
    return prs_mod.compress(raw)
