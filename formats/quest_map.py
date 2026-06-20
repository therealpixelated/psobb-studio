"""Byte-faithful codec for the PSOBB quest map (``.dat``) container.

This is the Layer-0 *map* half of a quest, the sibling of
:mod:`formats.quest_bin` (the script ``.bin`` half). Together they
describe a playable quest: ``quest_bin`` owns the bytecode script and the
``.qst`` transport; this module owns the **static map data** — the object
placements, the enemy/NPC spawn sets, and the wave/event/action tables —
that the engine streams into each floor.

A ``.dat`` file (decompressed; it is PRS-compressed on disk and inside a
``.qst``) is a flat **sequence of typed sections**, each prefixed with a
16-byte header. Section types:

==== ======================== =============================================
type name                     payload
==== ======================== =============================================
  0  END                      empty (16 zero bytes); terminates the file
  1  OBJECT_SETS              array of 0x44-byte ``ObjectSetEntry``
  2  ENEMY_SETS               array of 0x48-byte ``EnemySetEntry``
  3  EVENTS                   events header + entry array + action stream
  4  RANDOM_ENEMY_LOCATIONS   challenge mode (opaque round-trip)
  5  RANDOM_ENEMY_DEFINITIONS challenge mode (opaque round-trip)
==== ======================== =============================================

The ``floor`` (a.k.a. ``area_id``) field in each section header is the
primary key: a single ``.dat`` describes every floor of a quest, and the
sections appear grouped by type then floor (object sets for all floors,
then enemy sets for all floors, then events). We do **not** rely on that
ordering — sections are parsed and re-emitted in *file order* so the
round-trip is byte-exact regardless of how a given quest authored them.

Byte-exactness strategy
-----------------------
The parity gate is ``serialize_dat(parse_dat(dec)) == dec`` byte-for-byte
for every recognized BB ``.dat``. We achieve it the same way
``quest_bin`` does for the ``.bin`` header: every section keeps its raw
payload bytes verbatim (:attr:`Section.raw`), and structured entries
(objects/enemies/events) are decoded as a **view** over those bytes. By
default ``serialize_dat`` re-emits each section's raw bytes unchanged, so
a parsed-then-reserialized file is identical to the input down to event
action-stream padding and any unmodeled trailing bytes. Mutating the
structured view and calling :meth:`MapFile.rebuild` recomputes the raw
section bytes from the entries (used by the future placement editor).

Structurally-decoded vs opaque
------------------------------
* OBJECT_SETS (1), ENEMY_SETS (2): fully decoded into typed entries.
* EVENTS (3): the events header, the Event1/Event2 entry array, and the
  action stream are fully decoded.
* RANDOM_ENEMY_* (4, 5): challenge-mode sections. Their internal layout
  is understood (see the struct comments below) but they are carried as
  **opaque** raw blocks for the byte-exact round-trip; we expose their
  header fields but do not re-pack the variable internal tables.

Ground truth
-----------
Struct layouts are taken from newserv (MIT) ``src/Map.hh`` /
``src/Map.cc`` — the authoritative ``MapFile`` / ``SetDataTable`` structs
the quest disassembler uses — cross-checked against phantasmal-world
(MIT) ``psolib/.../quest/Dat.kt``. The entry sizes
(``ObjectSetEntry`` = 0x44, ``EnemySetEntry`` = 0x48), the event header
(``action_stream_offset``, ``entries_offset``, ``entry_count``,
big-endian ``format`` = 0 or ``'evt2'``), the 20-byte ``Event1Entry`` /
24-byte ``Event2Entry``, and the action opcodes (0x01 end, 0x08 spawn,
0x0A unlock, 0x0B lock, 0x0C trigger) all come from those sources.

Note that phantasmal-world's ``writeDat`` is a *semantic* re-emitter: it
re-groups sections by area and regenerates event headers, so it does NOT
round-trip byte-exact. This module deliberately does not follow that
path for serialization; it preserves the original section bytes.

JSON shape (semantic view; raw bytes are not in JSON):
    {
      "section_count": 26,
      "sections": [
        {"type": 1, "type_name": "OBJECT_SETS", "floor": 0,
         "object_count": 26},
        {"type": 2, "type_name": "ENEMY_SETS", "floor": 0,
         "enemy_count": 19},
        {"type": 3, "type_name": "EVENTS", "floor": 2,
         "event_count": 8, "evt2": false},
        ...
      ],
      "objects": [{"floor": 0, "type_id": 0, "type_name": "Player Set 1",
                   "x": .., "y": .., "z": .., ...}, ...],
      "enemies": [{"floor": 0, "type_id": 4, "type_name": "Hildebear",
                   "wave_number": 1, ...}, ...],
      "events":  [{"floor": 2, "id": 41, "wave_number": 1,
                   "actions": [...]}, ...],
    }
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from formats import prs
from formats.quest_map_names import (
    ENEMY_NAMES,
    NPC_SKIN_NAMES,
    OBJECT_NAMES,
)

# ---------------------------------------------------------------------------
# Section type constants (newserv Map.hh SectionHeader.type)
# ---------------------------------------------------------------------------
SECTION_END = 0
SECTION_OBJECT_SETS = 1
SECTION_ENEMY_SETS = 2
SECTION_EVENTS = 3
SECTION_RANDOM_ENEMY_LOCATIONS = 4
SECTION_RANDOM_ENEMY_DEFINITIONS = 5

SECTION_TYPE_NAMES = {
    SECTION_END: "END",
    SECTION_OBJECT_SETS: "OBJECT_SETS",
    SECTION_ENEMY_SETS: "ENEMY_SETS",
    SECTION_EVENTS: "EVENTS",
    SECTION_RANDOM_ENEMY_LOCATIONS: "RANDOM_ENEMY_LOCATIONS",
    SECTION_RANDOM_ENEMY_DEFINITIONS: "RANDOM_ENEMY_DEFINITIONS",
}

SECTION_HEADER_SIZE = 16  # {type, section_size, floor, data_size} u32[4]

OBJECT_ENTRY_SIZE = 0x44  # 68 bytes  (newserv ObjectSetEntry)
ENEMY_ENTRY_SIZE = 0x48  # 72 bytes  (newserv EnemySetEntry)

# Events section header (newserv EventsSectionHeader, 0x10 bytes).
EVENT1_ENTRY_SIZE = 0x14  # 20 bytes (Event1Entry, format == 0)
EVENT2_ENTRY_SIZE = 0x18  # 24 bytes (Event2Entry, format == 'evt2')
EVENT_FORMAT_EVT2 = 0x65767432  # big-endian 'evt2'

# Action-stream opcodes (newserv Map.cc / phantasmal Dat.kt).
ACTION_END = 0x01
ACTION_SPAWN_NPCS = 0x08
ACTION_UNLOCK_DOOR = 0x0A
ACTION_LOCK_DOOR = 0x0B
ACTION_TRIGGER_EVENT = 0x0C


# ---------------------------------------------------------------------------
# Schema-driven entry codec (mirrors battle_param.py's approach)
# ---------------------------------------------------------------------------
# Each schema is a tuple of (name, struct_format) pairs describing the
# fixed-stride entry. Floats carry a ``_<name>_bits`` sidecar in the
# semantic view so a JSON round-trip preserves the exact bit pattern
# (NaN/Inf and -0.0 survive), exactly like battle_param.py.

# ObjectSetEntry (0x44) — newserv Map.hh.
OBJECT_SCHEMA: Tuple[Tuple[str, str], ...] = (
    ("base_type", "H"),       # 00 object type id
    ("set_flags", "H"),       # 02 runtime; unused in DAT
    ("index", "H"),           # 04 runtime; unused in DAT
    ("floor", "H"),           # 06 floor id (redundant with header)
    ("entity_id", "H"),       # 08 = index + 0x4000; runtime
    ("group", "H"),           # 0A placement group
    ("room", "H"),            # 0C room index
    ("unknown_a3", "H"),      # 0E reserved
    ("pos_x", "f"),           # 10
    ("pos_y", "f"),           # 14
    ("pos_z", "f"),           # 18
    ("angle_x", "I"),         # 1C 32-bit angle (low 16 bits significant)
    ("angle_y", "I"),         # 20
    ("angle_z", "I"),         # 24
    ("param1", "f"),          # 28
    ("param2", "f"),          # 2C
    ("param3", "f"),          # 30
    ("param4", "i"),          # 34
    ("param5", "i"),          # 38
    ("param6", "i"),          # 3C
    ("unused_obj_ptr", "I"),  # 40 runtime pointer; unused in DAT
)

# EnemySetEntry (0x48) — newserv Map.hh.
ENEMY_SCHEMA: Tuple[Tuple[str, str], ...] = (
    ("base_type", "H"),       # 00 enemy type id
    ("set_flags", "H"),       # 02 runtime; unused in DAT
    ("index", "H"),           # 04 runtime; unused in DAT
    ("num_children", "H"),    # 06 child count (0 = constructor default)
    ("floor", "H"),           # 08 floor id (redundant with header)
    ("entity_id", "H"),       # 0A = index + 0x1000; runtime
    ("room", "H"),            # 0C room index
    ("wave_number", "H"),     # 0E primary wave id
    ("wave_number2", "H"),    # 10 secondary wave id
    ("unknown_a1", "H"),      # 12 reserved
    ("pos_x", "f"),           # 14
    ("pos_y", "f"),           # 18
    ("pos_z", "f"),           # 1C
    ("angle_x", "I"),         # 20
    ("angle_y", "I"),         # 24
    ("angle_z", "I"),         # 28
    ("param1", "f"),          # 2C
    ("param2", "f"),          # 30
    ("param3", "f"),          # 34
    ("param4", "f"),          # 38
    ("param5", "f"),          # 3C
    ("param6", "h"),          # 40 (note: int16, vs object's int32)
    ("param7", "h"),          # 42
    ("unused_obj_ptr", "I"),  # 44 runtime pointer; unused in DAT
)

_FLOAT_FORMATS = {"f"}


def _schema_size(schema: Tuple[Tuple[str, str], ...]) -> int:
    return sum(struct.calcsize(fmt) for _name, fmt in schema)


assert _schema_size(OBJECT_SCHEMA) == OBJECT_ENTRY_SIZE, (
    f"OBJECT_SCHEMA is {_schema_size(OBJECT_SCHEMA)}, expected {OBJECT_ENTRY_SIZE}"
)
assert _schema_size(ENEMY_SCHEMA) == ENEMY_ENTRY_SIZE, (
    f"ENEMY_SCHEMA is {_schema_size(ENEMY_SCHEMA)}, expected {ENEMY_ENTRY_SIZE}"
)


def _unpack_entry(buf: bytes, schema: Tuple[Tuple[str, str], ...]) -> Dict:
    """Decode one fixed-stride entry buffer into a field dict.

    Floats get a ``_<name>_bits`` sidecar holding the raw u32 bit pattern
    so a JSON round-trip is exact even for NaN/Inf/-0.0 cells.
    """
    out: Dict = {}
    cur = 0
    for name, fmt in schema:
        (val,) = struct.unpack_from(f"<{fmt}", buf, cur)
        if fmt in _FLOAT_FORMATS:
            bits = struct.unpack_from("<I", buf, cur)[0]
            out[name] = val
            out[f"_{name}_bits"] = bits
        else:
            out[name] = val
        cur += struct.calcsize(fmt)
    return out


def _pack_entry(fields: Dict, schema: Tuple[Tuple[str, str], ...]) -> bytes:
    """Re-encode a field dict to its fixed-stride entry buffer.

    Inverse of :func:`_unpack_entry`. Prefers the ``_<name>_bits`` sidecar
    for floats so the bit pattern is preserved across a JSON round-trip.
    """
    parts: List[bytes] = []
    for name, fmt in schema:
        if fmt in _FLOAT_FORMATS:
            bits_key = f"_{name}_bits"
            bits = fields.get(bits_key)
            if bits is not None:
                parts.append(struct.pack("<I", bits & 0xFFFFFFFF))
            else:
                parts.append(struct.pack("<f", fields[name]))
        else:
            parts.append(struct.pack(f"<{fmt}", fields[name]))
    return b"".join(parts)


# ---------------------------------------------------------------------------
# Typed entry dataclasses (the semantic placement view)
# ---------------------------------------------------------------------------
@dataclass
class DatObject:
    """One object placement (OBJECT_SETS entry, 0x44 bytes).

    ``fields`` is the full decoded struct dict (every field above);
    ``floor`` is the owning section's floor. Friendly accessors expose
    the common fields; mutate ``fields`` then call
    :meth:`MapFile.rebuild` to write changes back to the bytes.
    """

    floor: int
    fields: Dict

    @property
    def type_id(self) -> int:
        return self.fields["base_type"]

    @property
    def type_name(self) -> str:
        return OBJECT_NAMES.get(self.type_id, f"object_{self.type_id:#06x}")

    @property
    def section(self) -> int:
        return self.fields["room"]

    @property
    def group(self) -> int:
        return self.fields["group"]

    @property
    def pos(self) -> Tuple[float, float, float]:
        return (self.fields["pos_x"], self.fields["pos_y"], self.fields["pos_z"])

    @property
    def angle(self) -> Tuple[int, int, int]:
        return (
            self.fields["angle_x"] & 0xFFFF,
            self.fields["angle_y"] & 0xFFFF,
            self.fields["angle_z"] & 0xFFFF,
        )

    @property
    def params(self) -> List:
        return [self.fields[f"param{i}"] for i in range(1, 7)]

    def to_bytes(self) -> bytes:
        return _pack_entry(self.fields, OBJECT_SCHEMA)

    def to_json(self) -> Dict:
        d = dict(self.fields)
        d["floor"] = self.floor
        d["type_name"] = self.type_name
        return d


@dataclass
class DatNpc:
    """One enemy/NPC spawn (ENEMY_SETS entry, 0x48 bytes)."""

    floor: int
    fields: Dict

    @property
    def type_id(self) -> int:
        return self.fields["base_type"]

    @property
    def type_name(self) -> str:
        # Enemy table first; fall back to NPC skin table; then hex.
        return (
            ENEMY_NAMES.get(self.type_id)
            or NPC_SKIN_NAMES.get(self.type_id)
            or f"enemy_{self.type_id:#06x}"
        )

    @property
    def section(self) -> int:
        return self.fields["room"]

    @property
    def wave_number(self) -> int:
        return self.fields["wave_number"]

    @property
    def wave_number2(self) -> int:
        return self.fields["wave_number2"]

    @property
    def num_children(self) -> int:
        return self.fields["num_children"]

    @property
    def pos(self) -> Tuple[float, float, float]:
        return (self.fields["pos_x"], self.fields["pos_y"], self.fields["pos_z"])

    @property
    def angle(self) -> Tuple[int, int, int]:
        return (
            self.fields["angle_x"] & 0xFFFF,
            self.fields["angle_y"] & 0xFFFF,
            self.fields["angle_z"] & 0xFFFF,
        )

    @property
    def params(self) -> List:
        return [self.fields[f"param{i}"] for i in range(1, 8)]

    def to_bytes(self) -> bytes:
        return _pack_entry(self.fields, ENEMY_SCHEMA)

    def to_json(self) -> Dict:
        d = dict(self.fields)
        d["floor"] = self.floor
        d["type_name"] = self.type_name
        return d


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
@dataclass
class DatEventAction:
    """One opcode in an event's post-wave action stream.

    ``opcode`` is the action byte; ``args`` holds the decoded operands
    (e.g. ``{"section_id": .., "appear_flag": ..}`` for SPAWN_NPCS).
    """

    opcode: int
    args: Dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return {
            ACTION_SPAWN_NPCS: "spawn_npcs",
            ACTION_UNLOCK_DOOR: "unlock_door",
            ACTION_LOCK_DOOR: "lock_door",
            ACTION_TRIGGER_EVENT: "trigger_event",
        }.get(self.opcode, f"action_{self.opcode:#04x}")

    def to_json(self) -> Dict:
        return {"opcode": self.opcode, "name": self.name, **self.args}


@dataclass
class DatEvent:
    """One wave event (Event1Entry / Event2Entry) + its decoded actions.

    ``fields`` is the decoded entry struct dict. ``actions`` is the
    decoded action-stream slice the event points at. ``evt2`` flags the
    challenge-mode 24-byte format.
    """

    floor: int
    evt2: bool
    fields: Dict
    actions: List[DatEventAction]

    @property
    def id(self) -> int:
        return self.fields["event_id"]

    @property
    def room(self) -> int:
        return self.fields["room"]

    @property
    def wave_number(self) -> int:
        return self.fields["wave_number"]

    @property
    def event_type(self) -> int:
        return self.fields["event_type"]

    @property
    def delay(self) -> int:
        if self.evt2:
            return self.fields["min_delay"]
        return self.fields["delay"]

    def to_json(self) -> Dict:
        d = dict(self.fields)
        d["floor"] = self.floor
        d["evt2"] = self.evt2
        d["actions"] = [a.to_json() for a in self.actions]
        return d


def _parse_event_actions(stream: bytes, start: int) -> List[DatEventAction]:
    """Decode the action-stream slice beginning at ``start``.

    Stops at the END opcode (0x01) or an unknown opcode (matching
    phantasmal-world's tolerant behaviour). This is a *view* only — the
    raw bytes drive serialization, so a decode that stops early never
    affects byte-exactness.
    """
    actions: List[DatEventAction] = []
    pos = start
    n = len(stream)
    while pos < n:
        op = stream[pos]
        pos += 1
        if op == ACTION_END:
            break
        if op == ACTION_SPAWN_NPCS:
            if pos + 4 > n:
                break
            section_id, appear_flag = struct.unpack_from("<HH", stream, pos)
            pos += 4
            actions.append(
                DatEventAction(op, {"section_id": section_id, "appear_flag": appear_flag})
            )
        elif op == ACTION_UNLOCK_DOOR:
            if pos + 2 > n:
                break
            (door_id,) = struct.unpack_from("<H", stream, pos)
            pos += 2
            actions.append(DatEventAction(op, {"door_id": door_id}))
        elif op == ACTION_LOCK_DOOR:
            if pos + 2 > n:
                break
            (door_id,) = struct.unpack_from("<H", stream, pos)
            pos += 2
            actions.append(DatEventAction(op, {"door_id": door_id}))
        elif op == ACTION_TRIGGER_EVENT:
            if pos + 4 > n:
                break
            (event_id,) = struct.unpack_from("<I", stream, pos)
            pos += 4
            actions.append(DatEventAction(op, {"event_id": event_id}))
        else:
            # Unknown opcode — stop decoding this stream (view only).
            break
    return actions


# Event1Entry schema (format == 0), 0x14 bytes.
EVENT1_SCHEMA: Tuple[Tuple[str, str], ...] = (
    ("event_id", "I"),              # 00
    ("flags", "H"),                 # 04 runtime; unused in DAT
    ("event_type", "H"),            # 06
    ("room", "H"),                  # 08
    ("wave_number", "H"),           # 0A
    ("delay", "I"),                 # 0C frames
    ("action_stream_offset", "I"),  # 10 relative to action stream base
)

# Event2Entry schema (format == 'evt2'), 0x18 bytes.
EVENT2_SCHEMA: Tuple[Tuple[str, str], ...] = (
    ("event_id", "I"),              # 00
    ("flags", "H"),                 # 04
    ("event_type", "H"),            # 06
    ("room", "H"),                  # 08
    ("wave_number", "H"),           # 0A
    ("min_delay", "H"),             # 0C
    ("max_delay", "H"),             # 0E
    ("min_enemies", "B"),           # 10
    ("max_enemies", "B"),           # 11
    ("max_waves", "H"),             # 12
    ("action_stream_offset", "I"),  # 14
)

assert _schema_size(EVENT1_SCHEMA) == EVENT1_ENTRY_SIZE
assert _schema_size(EVENT2_SCHEMA) == EVENT2_ENTRY_SIZE


# ---------------------------------------------------------------------------
# Section model
# ---------------------------------------------------------------------------
@dataclass
class Section:
    """One typed section of a ``.dat`` file, in file order.

    ``raw`` is the verbatim payload (``data_size`` bytes, i.e. everything
    after the 16-byte header). Serialization re-emits the header computed
    from ``type``/``floor``/``len(raw)`` plus ``raw`` unchanged, which is
    byte-exact for a parsed file. The structured views (``objects`` /
    ``enemies`` / ``events``) are decoded from ``raw`` lazily by
    :func:`parse_dat`; mutate them and call :meth:`MapFile.rebuild` to
    recompute ``raw``.
    """

    type: int
    floor: int
    raw: bytes

    # Decoded views (populated by parse_dat for known types).
    objects: List[DatObject] = field(default_factory=list)
    enemies: List[DatNpc] = field(default_factory=list)
    events: List[DatEvent] = field(default_factory=list)
    # For EVENTS: the raw events-section header + format flag, kept so a
    # rebuild can reproduce the exact header geometry.
    event_format: int = 0  # 0 or EVENT_FORMAT_EVT2

    @property
    def type_name(self) -> str:
        return SECTION_TYPE_NAMES.get(self.type, f"type_{self.type}")

    @property
    def section_size(self) -> int:
        return SECTION_HEADER_SIZE + len(self.raw)

    def header_bytes(self) -> bytes:
        """The 16-byte section header for this section."""
        return struct.pack(
            "<IIII", self.type, self.section_size, self.floor, len(self.raw)
        )

    def to_bytes(self) -> bytes:
        return self.header_bytes() + self.raw

    def to_json(self) -> Dict:
        d: Dict = {
            "type": self.type,
            "type_name": self.type_name,
            "floor": self.floor,
            "data_size": len(self.raw),
        }
        if self.type == SECTION_OBJECT_SETS:
            d["object_count"] = len(self.objects)
        elif self.type == SECTION_ENEMY_SETS:
            d["enemy_count"] = len(self.enemies)
        elif self.type == SECTION_EVENTS:
            d["event_count"] = len(self.events)
            d["evt2"] = self.event_format == EVENT_FORMAT_EVT2
        return d


@dataclass
class MapFile:
    """A parsed quest ``.dat`` (decompressed).

    ``sections`` is the ordered list of typed sections, exactly as they
    appear in the file. ``trailing`` is any bytes after the END marker
    (normally empty, but preserved verbatim for byte-exactness). The
    ``objects`` / ``enemies`` / ``events`` properties flatten the
    per-section views for convenient iteration.
    """

    sections: List[Section] = field(default_factory=list)
    # The END-marker bytes that terminate the file (16 zero bytes when
    # present), plus any unmodeled trailing bytes after it. Stored so the
    # serialize is byte-exact.
    end_marker: bytes = b"\x00" * SECTION_HEADER_SIZE
    trailing: bytes = b""

    # ---- flattened semantic views -----------------------------------
    @property
    def objects(self) -> List[DatObject]:
        out: List[DatObject] = []
        for s in self.sections:
            out.extend(s.objects)
        return out

    @property
    def enemies(self) -> List[DatNpc]:
        out: List[DatNpc] = []
        for s in self.sections:
            out.extend(s.enemies)
        return out

    @property
    def events(self) -> List[DatEvent]:
        out: List[DatEvent] = []
        for s in self.sections:
            out.extend(s.events)
        return out

    @property
    def floors(self) -> List[int]:
        seen = []
        for s in self.sections:
            if s.floor not in seen:
                seen.append(s.floor)
        return seen

    def objects_on_floor(self, floor: int) -> List[DatObject]:
        return [o for o in self.objects if o.floor == floor]

    def enemies_on_floor(self, floor: int) -> List[DatNpc]:
        return [e for e in self.enemies if e.floor == floor]

    def events_on_floor(self, floor: int) -> List[DatEvent]:
        return [e for e in self.events if e.floor == floor]

    def decoded_section_types(self) -> List[int]:
        """Section types that were *structurally* decoded (not opaque)."""
        decoded = set()
        for s in self.sections:
            if s.type in (SECTION_OBJECT_SETS, SECTION_ENEMY_SETS, SECTION_EVENTS):
                decoded.add(s.type)
        return sorted(decoded)

    def opaque_section_types(self) -> List[int]:
        """Section types carried as opaque round-tripped blocks."""
        opaque = set()
        for s in self.sections:
            if s.type not in (
                SECTION_OBJECT_SETS,
                SECTION_ENEMY_SETS,
                SECTION_EVENTS,
            ):
                opaque.add(s.type)
        return sorted(opaque)

    def rebuild(self) -> None:
        """Recompute each section's ``raw`` bytes from its decoded view.

        Call after mutating ``objects`` / ``enemies`` / ``events``. For
        object and enemy sections the entries are re-packed in order. For
        event sections the header + entry array + action stream are
        regenerated (standard format only; ``evt2`` sections are left as
        their original raw bytes). Sections of opaque types are
        untouched. This is the *editor* path; it does not need to be
        byte-identical to the original, only valid.
        """
        for s in self.sections:
            if s.type == SECTION_OBJECT_SETS:
                s.raw = b"".join(o.to_bytes() for o in s.objects)
            elif s.type == SECTION_ENEMY_SETS:
                s.raw = b"".join(e.to_bytes() for e in s.enemies)
            elif s.type == SECTION_EVENTS and s.event_format != EVENT_FORMAT_EVT2:
                s.raw = _build_events_section(s.events)

    def to_json(self) -> Dict:
        return {
            "section_count": len(self.sections),
            "floors": self.floors,
            "decoded_section_types": [
                SECTION_TYPE_NAMES[t] for t in self.decoded_section_types()
            ],
            "opaque_section_types": [
                SECTION_TYPE_NAMES.get(t, f"type_{t}")
                for t in self.opaque_section_types()
            ],
            "sections": [s.to_json() for s in self.sections],
            "objects": [o.to_json() for o in self.objects],
            "enemies": [e.to_json() for e in self.enemies],
            "events": [e.to_json() for e in self.events],
        }


# ---------------------------------------------------------------------------
# Section decoders (raw payload -> structured view)
# ---------------------------------------------------------------------------
def _decode_object_section(section: Section) -> None:
    raw = section.raw
    if len(raw) % OBJECT_ENTRY_SIZE != 0:
        # Tolerate a stray tail (don't lose bytes — raw still drives the
        # byte-exact serialize); decode only the whole entries.
        count = len(raw) // OBJECT_ENTRY_SIZE
    else:
        count = len(raw) // OBJECT_ENTRY_SIZE
    objs: List[DatObject] = []
    for i in range(count):
        buf = raw[i * OBJECT_ENTRY_SIZE:(i + 1) * OBJECT_ENTRY_SIZE]
        objs.append(DatObject(floor=section.floor, fields=_unpack_entry(buf, OBJECT_SCHEMA)))
    section.objects = objs


def _decode_enemy_section(section: Section) -> None:
    raw = section.raw
    count = len(raw) // ENEMY_ENTRY_SIZE
    npcs: List[DatNpc] = []
    for i in range(count):
        buf = raw[i * ENEMY_ENTRY_SIZE:(i + 1) * ENEMY_ENTRY_SIZE]
        npcs.append(DatNpc(floor=section.floor, fields=_unpack_entry(buf, ENEMY_SCHEMA)))
    section.enemies = npcs


def _decode_events_section(section: Section) -> None:
    """Decode an EVENTS section payload into a list of DatEvent (view).

    The payload begins with EventsSectionHeader (0x10), then the event
    entry array (at ``entries_offset``), then the action stream (at
    ``action_stream_offset``). All offsets are relative to the start of
    the payload (== start of the header).
    """
    raw = section.raw
    if len(raw) < 0x10:
        return
    action_stream_offset, entries_offset, entry_count = struct.unpack_from("<III", raw, 0)
    # ``format`` is big-endian (newserv be_uint32_t).
    (fmt_be,) = struct.unpack_from(">I", raw, 0x0C)
    section.event_format = fmt_be
    evt2 = fmt_be == EVENT_FORMAT_EVT2
    entry_size = EVENT2_ENTRY_SIZE if evt2 else EVENT1_ENTRY_SIZE
    schema = EVENT2_SCHEMA if evt2 else EVENT1_SCHEMA

    # Action stream is everything from action_stream_offset to end.
    stream = raw[action_stream_offset:] if action_stream_offset <= len(raw) else b""

    events: List[DatEvent] = []
    for i in range(entry_count):
        off = entries_offset + i * entry_size
        if off + entry_size > len(raw):
            break
        fields = _unpack_entry(raw[off:off + entry_size], schema)
        aso = fields["action_stream_offset"]
        actions = _parse_event_actions(stream, aso) if aso < len(stream) else []
        events.append(DatEvent(floor=section.floor, evt2=evt2, fields=fields, actions=actions))
    section.events = events


def _build_events_section(events: List[DatEvent]) -> bytes:
    """Re-emit a standard-format EVENTS section payload from DatEvents.

    Used by :meth:`MapFile.rebuild` (the editor path). Mirrors
    phantasmal-world's writeEvents layout: header, then the 20-byte entry
    array, then the action stream, padded to a 4-byte boundary with 0xFF.
    The per-event ``action_stream_offset`` fields are recomputed.
    """
    entry_count = len(events)
    entries_offset = 0x10
    action_stream_offset = entries_offset + EVENT1_ENTRY_SIZE * entry_count

    # Build the action stream and record each event's relative offset.
    stream = bytearray()
    rel_offsets: List[int] = []
    for ev in events:
        rel_offsets.append(len(stream))
        for action in ev.actions:
            op = action.opcode
            stream.append(op)
            if op == ACTION_SPAWN_NPCS:
                stream += struct.pack(
                    "<HH", action.args["section_id"], action.args["appear_flag"]
                )
            elif op in (ACTION_UNLOCK_DOOR, ACTION_LOCK_DOOR):
                stream += struct.pack("<H", action.args["door_id"])
            elif op == ACTION_TRIGGER_EVENT:
                stream += struct.pack("<I", action.args["event_id"])
        stream.append(ACTION_END)
    # Pad the action stream to a 4-byte boundary with 0xFF.
    while len(stream) % 4 != 0:
        stream.append(0xFF)

    out = bytearray()
    out += struct.pack("<III", action_stream_offset, entries_offset, entry_count)
    out += struct.pack(">I", 0)  # standard format
    for ev, rel in zip(events, rel_offsets):
        fields = dict(ev.fields)
        fields["action_stream_offset"] = rel
        out += _pack_entry(fields, EVENT1_SCHEMA)
    out += stream
    return bytes(out)


# ---------------------------------------------------------------------------
# Top-level parse / serialize
# ---------------------------------------------------------------------------
def parse_dat(data: bytes) -> MapFile:
    """Parse a **decompressed** quest ``.dat`` into a :class:`MapFile`.

    ``data`` must be the raw decompressed bytes (the parity target). The
    file is walked as a sequence of 16-byte-headed sections until an END
    (type 0) section or end-of-data. Each section's payload is kept
    verbatim *and* structurally decoded where the type is known
    (objects/enemies/events); types 4/5 are carried opaque.

    Raises:
        ValueError: input isn't bytes-like, or a section header is
            self-inconsistent (size < 16, or data_size != size - 16, or a
            size that overruns the buffer).
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError("parse_dat: input must be bytes-like")
    data = bytes(data)

    sections: List[Section] = []
    pos = 0
    n = len(data)
    end_marker = b"\x00" * SECTION_HEADER_SIZE
    trailing = b""

    while pos + SECTION_HEADER_SIZE <= n:
        sec_type, section_size, floor, data_size = struct.unpack_from("<IIII", data, pos)

        if sec_type == SECTION_END:
            # The END marker is the 16-byte (usually zero) header; keep it
            # and any trailing bytes verbatim.
            end_marker = data[pos:pos + SECTION_HEADER_SIZE]
            trailing = data[pos + SECTION_HEADER_SIZE:]
            pos += SECTION_HEADER_SIZE
            break

        if section_size < SECTION_HEADER_SIZE:
            raise ValueError(
                f"parse_dat: section @0x{pos:X} has size {section_size} < 16"
            )
        if data_size != section_size - SECTION_HEADER_SIZE:
            raise ValueError(
                f"parse_dat: section @0x{pos:X} type {sec_type}: data_size "
                f"{data_size} != section_size-16 ({section_size - 16})"
            )
        if pos + section_size > n:
            raise ValueError(
                f"parse_dat: section @0x{pos:X} type {sec_type} size "
                f"{section_size} overruns buffer (len {n})"
            )

        raw = data[pos + SECTION_HEADER_SIZE:pos + section_size]
        section = Section(type=sec_type, floor=floor, raw=raw)

        if sec_type == SECTION_OBJECT_SETS:
            _decode_object_section(section)
        elif sec_type == SECTION_ENEMY_SETS:
            _decode_enemy_section(section)
        elif sec_type == SECTION_EVENTS:
            _decode_events_section(section)
        # types 4/5 (and any unknown): opaque round-trip via section.raw.

        sections.append(section)
        pos += section_size
    else:
        # Reached end-of-data without an END marker; nothing trailing.
        end_marker = b""

    return MapFile(sections=sections, end_marker=end_marker, trailing=trailing)


def serialize_dat(mf: MapFile) -> bytes:
    """Serialize a :class:`MapFile` back to decompressed ``.dat`` bytes.

    Byte-exact inverse of :func:`parse_dat` for a parsed file:
    ``serialize_dat(parse_dat(buf)) == buf``. Each section is emitted in
    order from its (verbatim) ``raw`` bytes, followed by the END marker
    and any preserved trailing bytes.

    If you mutated the structured views, call :meth:`MapFile.rebuild`
    first to refresh the ``raw`` bytes from the entries.
    """
    out = bytearray()
    for s in mf.sections:
        out += s.to_bytes()
    out += mf.end_marker
    out += mf.trailing
    return bytes(out)


# ---------------------------------------------------------------------------
# PRS convenience (reuses formats.prs — never reimplemented here)
# ---------------------------------------------------------------------------
def decompress_dat(compressed: bytes) -> bytes:
    """PRS-decompress a ``.dat`` (the on-disk / in-``.qst`` form)."""
    return prs.decompress(compressed)


def compress_dat(decompressed: bytes) -> bytes:
    """PRS-compress a decompressed ``.dat`` for on-disk / ``.qst`` use."""
    return prs.compress(decompressed)


def parse_dat_compressed(compressed: bytes) -> MapFile:
    """Convenience: PRS-decompress then :func:`parse_dat`."""
    return parse_dat(prs.decompress(compressed))


# ---------------------------------------------------------------------------
# Multi-floor accessor (spec §5)
# ---------------------------------------------------------------------------
@dataclass
class FloorData:
    """All map entities for a single floor."""

    floor: int
    objects: List[DatObject] = field(default_factory=list)
    enemies: List[DatNpc] = field(default_factory=list)
    events: List[DatEvent] = field(default_factory=list)

    def to_json(self) -> Dict:
        return {
            "floor": self.floor,
            "objects": [o.to_json() for o in self.objects],
            "enemies": [e.to_json() for e in self.enemies],
            "events": [e.to_json() for e in self.events],
        }


def parse_dat_by_floor(data: bytes) -> Dict[int, FloorData]:
    """Parse a ``.dat`` and group its entities by floor index.

    Returns a dict keyed by floor (ascending insertion order), each value
    a :class:`FloorData` with the floor's objects, enemies and events.
    """
    mf = parse_dat(data)
    by_floor: Dict[int, FloorData] = {}
    for floor in mf.floors:
        by_floor[floor] = FloorData(
            floor=floor,
            objects=mf.objects_on_floor(floor),
            enemies=mf.enemies_on_floor(floor),
            events=mf.events_on_floor(floor),
        )
    return by_floor


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _scrub_nonfinite(o):
    """Replace NaN/Inf floats with None for strict-JSON serializability.

    The exact bits live in the ``_<name>_bits`` sidecars, so byte-exact
    round-trip survives this scrub (same approach as battle_param.py).
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


def to_json(mf: MapFile) -> str:
    """Pretty-print a :class:`MapFile` as JSON (semantic view)."""
    return json.dumps(_scrub_nonfinite(mf.to_json()), indent=2, ensure_ascii=False)
