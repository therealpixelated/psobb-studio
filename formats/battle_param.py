"""Parser/serializer for PSOBB ``BattleParamEntry*.dat``.

Six variants ship with newserv (and Booma.Server, the local fixture
source):
    BattleParamEntry.dat        -- ep1 offline (lab? actually generic Episode 1)
    BattleParamEntry_on.dat     -- ep1 online ultimate
    BattleParamEntry_lab.dat    -- ep2 offline
    BattleParamEntry_lab_on.dat -- ep2 online
    BattleParamEntry_ep4.dat    -- ep4 offline
    BattleParamEntry_ep4_on.dat -- ep4 online

Each file is **0xF600 = 62976 bytes**, structured as 4 difficulty arrays
each containing 96 (= 0x60) entries of 4 record types:

    +------- per difficulty (0x3D80 bytes) -------------+
    | stats[96]      0x24 bytes each ->  0x06C0  total  |
    | attacks[96]    0x30 bytes each ->  0x1200  total  |
    | resists[96]    0x20 bytes each ->  0x0C00  total  |
    | animations[96] 0x30 bytes each ->  0x1200  total  |
    +---------------------------------------------------+
    Total per difficulty: 0x3D80
    File total: 4 * 0x3D80 = 0x0F600

Layout source: ``Blue Burst Patch Project/battleparam.h`` (verified via
``BPDifficultyFile`` struct member offsets and ``sizeof(BPFile)``). The
0x3000-byte figure quoted in early research notes was for a single
record array, not the full file.

Field semantics for ``animations`` (the 12-float ``MovementData`` array
in editor terminology) are documented per-mob in newserv
``notes/movement-data.txt``. For Booma slot 0x4A:
    fparam1 (anim[0])  = idle move speed
    fparam2 (anim[1])  = idle walking animation speed
    fparam3 (anim[2])  = engaged move speed
    fparam4 (anim[3])  = engaged animation speed
    fparam5 (anim[4])  = poison cloud damage  (Merillia variant only)
    fparam6 (anim[5])  = run-away speed
    iparam1 (anim[6])  = low HP threshold (0-100)  (Merillia variant)
    iparam2-6 (anim[7..11]) = TODO

All numeric fields are little-endian. Stats include signed (atp/mst/...)
and unsigned (xp) sub-types; we expose them as the most natural Python
type and round-trip exactly.

JSON shape (used by /api/battle_param/<variant>):
    {
      "variant": "on",
      "difficulties": [
        {"name": "Normal",  "entries": [<96 entries>]},
        {"name": "Hard",    "entries": [...]},
        {"name": "VeryHard","entries": [...]},
        {"name": "Ultimate","entries": [...]}
      ]
    }
where each entry is:
    {
      "slot": 0x4A,
      "name": "Booma",
      "stats":      {<24 fields>},
      "attacks":    {<28 fields>},
      "resists":    {<22 fields>},
      "animations": {"fparam1": ..., "fparam2": ..., "iparam1": ...}
    }
"""
from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Sizes
# ---------------------------------------------------------------------------
ENTRIES_PER_DIFFICULTY = 0x60  # 96
NUM_DIFFICULTIES = 4
DIFFICULTY_NAMES = ("Normal", "Hard", "VeryHard", "Ultimate")

STATS_ENTRY_SIZE = 0x24
ATTACKS_ENTRY_SIZE = 0x30
RESISTS_ENTRY_SIZE = 0x20
ANIMATIONS_ENTRY_SIZE = 0x30

PER_DIFFICULTY_SIZE = (
    ENTRIES_PER_DIFFICULTY * STATS_ENTRY_SIZE
    + ENTRIES_PER_DIFFICULTY * ATTACKS_ENTRY_SIZE
    + ENTRIES_PER_DIFFICULTY * RESISTS_ENTRY_SIZE
    + ENTRIES_PER_DIFFICULTY * ANIMATIONS_ENTRY_SIZE
)
FILE_SIZE = PER_DIFFICULTY_SIZE * NUM_DIFFICULTIES  # 0xF600

VALID_VARIANTS = ("on", "off", "lab_on", "lab_off", "ep4_on", "ep4_off")
# Map the editor's variant token to the on-disk filename. Booma.Server
# uses "BattleParamEntry.dat" (no suffix) for offline ep1 and
# "_on.dat" for online; this mirrors newserv.
VARIANT_TO_FILENAME = {
    "off": "BattleParamEntry.dat",
    "on": "BattleParamEntry_on.dat",
    "lab_off": "BattleParamEntry_lab.dat",
    "lab_on": "BattleParamEntry_lab_on.dat",
    "ep4_off": "BattleParamEntry_ep4.dat",
    "ep4_on": "BattleParamEntry_ep4_on.dat",
}


# ---------------------------------------------------------------------------
# Slot table (mob slot index 0..0x60 -> human-readable name)
# Pulled from Blue-Burst-Patch-Project/battleparam.h ``BPStatsIndex``.
# Where multiple eps share a slot we pick the most-common ep1 name and
# include the overload in the comment dict for UI surfacing.
# ---------------------------------------------------------------------------
SLOT_NAMES: Dict[int, str] = {
    0x00: "Mothmant",          # ep4: Boota
    0x01: "Monest",            # ep4: ZeBoota
    0x02: "SavageWolf",
    0x03: "BarbarousWolf",     # ep4: BaBoota
    0x04: "PoisonLily",
    0x05: "NarLily",           # ep4: SandRappyCrater
    0x06: "SinowBeat",         # ep2: SinowBerill, ep4: DelRappyCrater
    0x07: "Canadine",          # ep2: Gee, ep4: ZuCrater
    0x08: "CanadineRing",      # ep2: PigRay, ep4: PazuzuCrater
    0x09: "Canane",            # ep2: UlRay, ep4: Astark
    0x0A: "ChaosSorcerer",
    0x0B: "BeeR",
    0x0C: "BeeL",
    0x0D: "ChaosBringer",      # ep2: Delbiter, ep4: SatelliteLizardCrater
    0x0E: "DarkBelra",         # ep4: YowieCrater
    0x0F: "DeRolLe",           # ep2: BarbaRay, ep4: Dorphon
    0x10: "DeRolLeShell",      # ep2: BarbaRayPart, ep4: DorphonEclair
    0x11: "DeRolLeMine",       # ep4: Goran
    0x12: "Dragon",            # ep2: GolDragon, ep4: PyroGoran
    0x13: "SinowGold",         # ep2: SinowSpigell, ep4: GoranDetonator
    0x17: "_unused_17",        # ep4: SandRappyDesert
    0x18: "RagRappy",          # ep4: DelRappyDesert
    0x19: "AlRappy",           # ep2: LoveRappy, ep4: MerissaA
    0x1A: "NanoDragon",        # ep2: GiGue, ep4: MerissaAA
    0x1B: "Dubchic",           # ep4: ZuDesert
    0x1C: "Gillchic",          # ep4: PazuzuDesert
    0x1D: "Garanz",            # ep4: SatelliteLizardDesert
    0x1E: "DarkGunner",        # ep2: GalGryphon, ep4: YowieDesert
    0x1F: "Bulclaw",           # ep4: Girtablulu
    0x20: "Claw",              # ep4: SaintMilionPhase1
    0x21: "VolOptForm1",       # ep4: SpinnerSaintMilion1
    0x22: "VolOptPillar",      # ep4: SaintMilionPhase2
    0x23: "VolOptMonitor",     # ep2: Epsilon, ep4: SpinnerSaintMilion2
    0x24: "VolOptSpire",       # ep2: Epsigard, ep4: ShambertinPhase1
    0x25: "VolOptForm2",       # ep2: DelLily, ep4: SpinnerShambertin1
    0x26: "VolOptPrison",      # ep2: IllGill, ep4: ShambertinPhase2
    0x27: "_unused_27",        # ep4: SpinnerShambertin2
    0x28: "_unused_28",        # ep4: KondrieuPhase1
    0x29: "_unused_29",        # ep4: SpinnerKondrieu1
    0x2A: "_unused_2A",        # ep4: KondrieuPhase2
    0x2B: "_unused_2B",        # ep2: OlgaFlowForm1, ep4: SpinnerKondrieu2
    0x2C: "OlgaFlowForm2",     # ep2 only
    0x2D: "Gael",              # ep2 only
    0x2E: "Giel",              # ep2 only
    0x30: "PofuillySlime",     # ep2: Deldepth
    0x31: "PanArms",
    0x32: "Hidoom",
    0x33: "Migium",
    0x34: "PouillySlime",
    0x35: "Darvant",
    0x36: "DarkFalzForm1",
    0x37: "DarkFalzForm2",
    0x38: "DarkFalzForm3",
    0x39: "DarvantFalz",
    0x3A: "Mericarol",         # ep2 only
    0x3B: "UlGibbon",          # ep2 only
    0x3C: "ZolGibbon",         # ep2 only
    0x3D: "Gibbles",           # ep2 only
    0x40: "Morfos",            # ep2 only
    0x41: "Recobox",           # ep2 only
    0x42: "Recon",             # ep2 only
    0x43: "SinowZoa",          # ep2 only
    0x44: "SinowZele",         # ep2 only
    0x45: "Merikle",           # ep2 only
    0x46: "Mericus",           # ep2 only
    0x48: "Dubwitch",
    0x49: "Hildebear",
    0x4A: "Hildeblue",
    0x4B: "Booma",             # ep2: Merillia
    0x4C: "Gobooma",           # ep2: Meriltas
    0x4D: "Gigobooma",
    0x4E: "GrassAssassin",
    0x4F: "EvilShark",         # ep2: Dolmolm
    0x50: "PalShark",          # ep2: Dolmdarl
    0x51: "GuilShark",
    0x52: "Delsaber",
    0x53: "Dimenian",
    0x54: "LaDimenian",
    0x55: "SoDimenian",
}


# ---------------------------------------------------------------------------
# Field schemas (drive struct.unpack/pack and JSON conversion)
# ---------------------------------------------------------------------------
# Each schema is a tuple (name, struct_format, count). count > 1 means
# the field is an array; we surface as a Python list. The struct_format
# is per-element; total field byte length = struct.calcsize(format) * count.
#
# struct format chars used:
#   h  signed 16-bit
#   H  unsigned 16-bit
#   i  signed 32-bit
#   I  unsigned 32-bit
#   f  IEEE-754 float32
#   B  unsigned 8-bit
#
# All fields are little-endian.

# StatsEntry layout (size 0x24)
#   battleparam.h ``BPStatsEntry``:
#     int16 atp, mst, evp, hp, dfp, ata, lck, esp
#     float field_0x10 (4)
#     float field_0x14 (4)
#     int   unknown_hp_mst_modifier (4)
#     int16 xp
#     int16 field_0x1e
#     int16 field_0x20
#     int16 field_0x22
STATS_SCHEMA: Tuple[Tuple[str, str, int], ...] = (
    ("atp", "h", 1),
    ("mst", "h", 1),
    ("evp", "h", 1),
    ("hp", "h", 1),
    ("dfp", "h", 1),
    ("ata", "h", 1),
    ("lck", "h", 1),
    ("esp", "h", 1),
    ("hp_modifier", "f", 1),       # field_0x10
    ("dfp_modifier", "f", 1),      # field_0x14
    ("hp_mst_modifier", "i", 1),   # unknown_hp_mst_modifier
    ("xp", "h", 1),
    ("field_0x1e", "h", 1),
    ("field_0x20", "h", 1),
    ("field_0x22", "h", 1),
)

# AttacksEntry layout (size 0x30) -- newserv ``AttackData`` (more
# semantic names than BB Patch Project):
#   int16 min_atp, max_atp, min_ata, max_ata
#   float distance_x
#   uint32 angle (bams; 0x10000 = full revolution)
#   float distance_y
#   uint16 unknown_a8 .. unknown_a11
#   uint32 unknown_a12 .. unknown_a16
ATTACKS_SCHEMA: Tuple[Tuple[str, str, int], ...] = (
    ("min_atp", "h", 1),
    ("max_atp", "h", 1),
    ("min_ata", "h", 1),
    ("max_ata", "h", 1),
    ("distance_x", "f", 1),
    ("angle", "I", 1),
    ("distance_y", "f", 1),
    ("unknown_a8", "H", 1),
    ("unknown_a9", "H", 1),
    ("unknown_a10", "H", 1),
    ("unknown_a11", "H", 1),
    ("unknown_a12", "I", 1),
    ("unknown_a13", "I", 1),
    ("unknown_a14", "I", 1),
    ("unknown_a15", "I", 1),
    ("unknown_a16", "I", 1),
)

# ResistsEntry layout (size 0x20) -- newserv ``ResistData``:
#   int16 evp_bonus
#   uint16 efr, eic, eth, elt, edk
#   uint32 unknown_a6 .. unknown_a9
#   int32 dfp_bonus
RESISTS_SCHEMA: Tuple[Tuple[str, str, int], ...] = (
    ("evp_bonus", "h", 1),
    ("efr", "H", 1),
    ("eic", "H", 1),
    ("eth", "H", 1),
    ("elt", "H", 1),
    ("edk", "H", 1),
    ("unknown_a6", "I", 1),
    ("unknown_a7", "I", 1),
    ("unknown_a8", "I", 1),
    ("unknown_a9", "I", 1),
    ("dfp_bonus", "i", 1),
)

# AnimationsEntry layout (size 0x30) -- newserv ``MovementData``:
#   float fparam1..fparam6
#   uint32 iparam1..iparam6
# Per newserv MovementData. Per ``notes/movement-data.txt`` the
# semantics of each slot are mob-specific (slot 0x4A=Booma uses fparam3
# for engaged movement speed, etc.).
ANIMATIONS_SCHEMA: Tuple[Tuple[str, str, int], ...] = (
    ("fparam1", "f", 1),
    ("fparam2", "f", 1),
    ("fparam3", "f", 1),
    ("fparam4", "f", 1),
    ("fparam5", "f", 1),
    ("fparam6", "f", 1),
    ("iparam1", "I", 1),
    ("iparam2", "I", 1),
    ("iparam3", "I", 1),
    ("iparam4", "I", 1),
    ("iparam5", "I", 1),
    ("iparam6", "I", 1),
)


# ---------------------------------------------------------------------------
# Schema-driven (un)pack helpers
# ---------------------------------------------------------------------------
def _schema_byte_size(schema: Tuple[Tuple[str, str, int], ...]) -> int:
    return sum(struct.calcsize(fmt) * count for _name, fmt, count in schema)


# Sanity-check at module load: schema sizes must match struct sizes.
assert _schema_byte_size(STATS_SCHEMA) == STATS_ENTRY_SIZE, (
    f"STATS schema is {_schema_byte_size(STATS_SCHEMA)}, expected "
    f"{STATS_ENTRY_SIZE}"
)
assert _schema_byte_size(ATTACKS_SCHEMA) == ATTACKS_ENTRY_SIZE, (
    f"ATTACKS schema is {_schema_byte_size(ATTACKS_SCHEMA)}, expected "
    f"{ATTACKS_ENTRY_SIZE}"
)
assert _schema_byte_size(RESISTS_SCHEMA) == RESISTS_ENTRY_SIZE, (
    f"RESISTS schema is {_schema_byte_size(RESISTS_SCHEMA)}, expected "
    f"{RESISTS_ENTRY_SIZE}"
)
assert _schema_byte_size(ANIMATIONS_SCHEMA) == ANIMATIONS_ENTRY_SIZE, (
    f"ANIMATIONS schema is {_schema_byte_size(ANIMATIONS_SCHEMA)}, expected "
    f"{ANIMATIONS_ENTRY_SIZE}"
)


def _parse_record(buf: bytes, off: int, schema: Tuple[Tuple[str, str, int], ...]) -> Dict:
    out: Dict = {}
    cur = off
    for name, fmt, count in schema:
        elem_size = struct.calcsize(fmt)
        if count == 1:
            (val,) = struct.unpack_from(f"<{fmt}", buf, cur)
            # Floats: preserve the raw bit pattern across JSON round-trip
            # by also storing the int32 reinterpret. Stock BattleParam
            # files do contain NaN/Inf in unused-field cells (bit pattern
            # 0xFFFFFFFF / 0x7FC00000 etc), and JSON's NaN handling is
            # lossy. We carry the bits in a sidecar field.
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


def _pack_record(record: Dict, schema: Tuple[Tuple[str, str, int], ...]) -> bytes:
    parts = []
    for name, fmt, count in schema:
        if count == 1:
            if fmt == "f":
                # Prefer bits sidecar if present (preserves NaN payloads
                # and protects against JSON's `None` / strict-encoder
                # round-trip lossiness).
                bits_key = f"_{name}_bits"
                if bits_key in record and record[bits_key] is not None:
                    parts.append(struct.pack("<I", record[bits_key] & 0xFFFFFFFF))
                else:
                    v = record[name]
                    if v is None:
                        # No usable value or bits — write zero. Should
                        # only happen for caller-built dicts that didn't
                        # come from parse(); a stock parse round-trip
                        # always carries the bits sidecar.
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
                    # Write each float; replace None with 0.0.
                    safe = [0.0 if x is None else x for x in v]
                    parts.append(struct.pack(f"<{count}f", *safe))
            else:
                parts.append(struct.pack(f"<{count}{fmt}", *v))
    return b"".join(parts)


def _scrub_nonfinite(o):
    """Recursively replace NaN/Inf floats with None so the result is
    strict-JSON serializable. Lists/dicts handled in place (returned).
    """
    import math
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
# Dataclasses (typed shapes returned to callers)
# ---------------------------------------------------------------------------
@dataclass
class BattleParamEntry:
    """One mob slot's full record set within one difficulty."""
    slot: int
    name: str
    stats: Dict
    attacks: Dict
    resists: Dict
    animations: Dict


@dataclass
class BattleParamDifficulty:
    """All 96 entries for one difficulty level."""
    name: str
    entries: List[BattleParamEntry]


@dataclass
class BattleParamFile:
    """Parsed BattleParamEntry*.dat -- 4 difficulties x 96 entries each."""
    variant: str  # "on"/"off"/"lab_on"/...
    difficulties: List[BattleParamDifficulty]

    def to_json(self) -> Dict:
        """Convert to plain JSON-able structure (no dataclass types).

        NaN/Inf floats are replaced with None — the underlying bits are
        preserved in the ``_<field>_bits`` sidecar so byte-exact
        round-trip is preserved. Strict-JSON encoders (Starlette,
        browser ``JSON.parse``) reject non-finite floats, so this is the
        only way to ship a stock file's "unused" fields safely.
        """
        return _scrub_nonfinite({
            "variant": self.variant,
            "difficulties": [
                {"name": d.name, "entries": [asdict(e) for e in d.entries]}
                for d in self.difficulties
            ],
        })

    @classmethod
    def from_json(cls, data: Dict) -> "BattleParamFile":
        """Reconstruct from to_json() output."""
        diffs: List[BattleParamDifficulty] = []
        for d in data["difficulties"]:
            entries = [
                BattleParamEntry(
                    slot=e["slot"],
                    name=e.get("name", SLOT_NAMES.get(e["slot"], f"slot_{e['slot']:02X}")),
                    stats=e["stats"],
                    attacks=e["attacks"],
                    resists=e["resists"],
                    animations=e["animations"],
                )
                for e in d["entries"]
            ]
            diffs.append(BattleParamDifficulty(name=d["name"], entries=entries))
        return cls(variant=data.get("variant", ""), difficulties=diffs)


# ---------------------------------------------------------------------------
# Parsing / serialization
# ---------------------------------------------------------------------------
def parse(buf: bytes, variant: str = "") -> BattleParamFile:
    """Parse a BattleParamEntry*.dat byte buffer.

    Args:
        buf: full file bytes; must be exactly 0xF600 = 62976 bytes.
        variant: optional label ("on"/"off"/"lab_on"/...) to embed in
            the parsed file. Used only for round-trip JSON metadata.

    Returns:
        BattleParamFile.

    Raises:
        ValueError: bad input length or invalid bytes.
    """
    if not isinstance(buf, (bytes, bytearray, memoryview)):
        raise ValueError("parse: input must be bytes-like")
    if len(buf) != FILE_SIZE:
        raise ValueError(
            f"parse: BattleParamEntry file must be {FILE_SIZE} bytes "
            f"(0x{FILE_SIZE:x}), got {len(buf)}"
        )

    diffs: List[BattleParamDifficulty] = []
    for d in range(NUM_DIFFICULTIES):
        diff_off = d * PER_DIFFICULTY_SIZE
        # Within a difficulty, the four arrays are tightly packed in
        # the order: stats, attacks, resists, animations.
        stats_off = diff_off
        attacks_off = stats_off + ENTRIES_PER_DIFFICULTY * STATS_ENTRY_SIZE
        resists_off = attacks_off + ENTRIES_PER_DIFFICULTY * ATTACKS_ENTRY_SIZE
        anims_off = resists_off + ENTRIES_PER_DIFFICULTY * RESISTS_ENTRY_SIZE

        entries: List[BattleParamEntry] = []
        for i in range(ENTRIES_PER_DIFFICULTY):
            entries.append(
                BattleParamEntry(
                    slot=i,
                    name=SLOT_NAMES.get(i, f"slot_{i:02X}"),
                    stats=_parse_record(buf, stats_off + i * STATS_ENTRY_SIZE, STATS_SCHEMA),
                    attacks=_parse_record(buf, attacks_off + i * ATTACKS_ENTRY_SIZE, ATTACKS_SCHEMA),
                    resists=_parse_record(buf, resists_off + i * RESISTS_ENTRY_SIZE, RESISTS_SCHEMA),
                    animations=_parse_record(buf, anims_off + i * ANIMATIONS_ENTRY_SIZE, ANIMATIONS_SCHEMA),
                )
            )
        diffs.append(BattleParamDifficulty(name=DIFFICULTY_NAMES[d], entries=entries))

    return BattleParamFile(variant=variant, difficulties=diffs)


def serialize(bpf: BattleParamFile) -> bytes:
    """Serialize a BattleParamFile back to the raw .dat byte stream.

    Inverse of :func:`parse`. Round-trip guarantee: ``parse(serialize(x))``
    equals ``x`` for any well-formed BattleParamFile, and
    ``serialize(parse(buf))`` equals ``buf`` for any valid 0xF600-byte
    input.
    """
    if len(bpf.difficulties) != NUM_DIFFICULTIES:
        raise ValueError(
            f"serialize: expected {NUM_DIFFICULTIES} difficulties, "
            f"got {len(bpf.difficulties)}"
        )

    parts: List[bytes] = []
    for di, diff in enumerate(bpf.difficulties):
        if len(diff.entries) != ENTRIES_PER_DIFFICULTY:
            raise ValueError(
                f"serialize: difficulty {di} has {len(diff.entries)} entries, "
                f"expected {ENTRIES_PER_DIFFICULTY}"
            )
        # Build each array in order: stats, attacks, resists, animations.
        stats_b = b"".join(_pack_record(e.stats, STATS_SCHEMA) for e in diff.entries)
        atk_b = b"".join(_pack_record(e.attacks, ATTACKS_SCHEMA) for e in diff.entries)
        res_b = b"".join(_pack_record(e.resists, RESISTS_SCHEMA) for e in diff.entries)
        anim_b = b"".join(_pack_record(e.animations, ANIMATIONS_SCHEMA) for e in diff.entries)
        parts.extend([stats_b, atk_b, res_b, anim_b])

    out = b"".join(parts)
    if len(out) != FILE_SIZE:
        raise RuntimeError(
            f"serialize: produced {len(out)} bytes, expected {FILE_SIZE}"
        )
    return out


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def to_json(bpf: BattleParamFile) -> str:
    """Pretty-print BattleParamFile as JSON (sorted keys for diff-stability)."""
    return json.dumps(bpf.to_json(), indent=2, sort_keys=False)


def from_json(text: str, variant: str = "") -> BattleParamFile:
    """Parse JSON text into a BattleParamFile."""
    obj = json.loads(text)
    bpf = BattleParamFile.from_json(obj)
    if variant and not bpf.variant:
        bpf.variant = variant
    return bpf


# ---------------------------------------------------------------------------
# Slot-table sidecar
# ---------------------------------------------------------------------------
def slot_table_json() -> str:
    """Emit the slot table as a JSON sidecar for the editor UI.

    The UI loads this once at startup and uses it for the mob picker.
    Non-coders can edit the file in-place to relabel slots; we never
    write back to it, so user edits survive editor upgrades.
    """
    table = {f"0x{slot:02X}": name for slot, name in sorted(SLOT_NAMES.items())}
    return json.dumps({"slots": table}, indent=2)
