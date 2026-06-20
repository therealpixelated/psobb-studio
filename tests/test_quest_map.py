"""Tests for formats.quest_map — the PSOBB quest ``.dat`` (map) codec.

The parity gate is the **byte-exact decompressed-.dat round-trip**:

  serialize_dat(parse_dat(dec)) == dec   BYTE-FOR-BYTE

for every recognized BB ``.dat`` — the phantasmal-world fixtures
(quest118_e, quest27_e, both extracted from their ``.qst`` via
:mod:`formats.quest_bin`) plus a sweep of the newserv quest corpus and
the tethealla ``.qst`` tree. Sections we can't yet structurally re-pack
(challenge-mode types 4/5, and ``evt2`` event sections) are carried as
opaque verbatim blocks, so the byte-exact gate still holds for them.

There is also an always-run synthetic vector (no reference data needed),
and structural-decode checks (objects/enemies/events parse into sane
fields).
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest

from formats import prs
from formats.quest_bin import parse_qst
from formats.quest_map import (
    EVENT_FORMAT_EVT2,
    SECTION_ENEMY_SETS,
    SECTION_EVENTS,
    SECTION_OBJECT_SETS,
    parse_dat,
    parse_dat_by_floor,
    serialize_dat,
    to_json,
)

# ---------------------------------------------------------------------------
# Fixture locations (gated; skip-clean if absent on a bare clone)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
_PW_REL = Path("_reference") / "phantasmal-world" / "psolib" / "src" / "commonTest" / "resources"


def _find_pw_fixtures() -> Path:
    """Locate the phantasmal-world fixture dir.

    ``_reference/`` is gitignored, so in a detached git worktree it lives
    only in the main checkout. Try the current tree first, then the
    sibling ``psobb-studio`` checkout, then a common home root.
    """
    candidates = [
        ROOT / _PW_REL,
        ROOT.parent / "psobb-studio" / _PW_REL,
        Path(os.path.expanduser("~/Repositories/psobb-studio")) / _PW_REL,
    ]
    for c in candidates:
        if (c / "quest118_e.qst").is_file():
            return c
    return ROOT / _PW_REL  # default (won't exist; HAS_PW gates it off)


PW_FIXTURES = _find_pw_fixtures()
HAS_PW = PW_FIXTURES.is_dir() and (PW_FIXTURES / "quest118_e.qst").is_file()

NEWSERV_QUESTS = Path(os.path.expanduser("~/Repositories/newserv/system/quests"))
TETHEALLA_QUESTS = PW_FIXTURES / "tethealla_v0.143_quests"


# ---------------------------------------------------------------------------
# Synthetic vector (always runs — no reference data required)
# ---------------------------------------------------------------------------
def _make_synthetic_dat() -> bytes:
    """Build a minimal but structurally valid ``.dat`` in memory.

    One OBJECT_SETS section (2 objects), one ENEMY_SETS section (1 enemy),
    one EVENTS section (2 events with a small action stream), then END.
    """
    from formats.quest_map import (
        ENEMY_ENTRY_SIZE,
        OBJECT_ENTRY_SIZE,
        _build_events_section,
        DatEvent,
        DatEventAction,
        ACTION_SPAWN_NPCS,
        ACTION_UNLOCK_DOOR,
    )

    def section(stype: int, floor: int, payload: bytes) -> bytes:
        return struct.pack("<IIII", stype, 16 + len(payload), floor, len(payload)) + payload

    # Two object entries (0x44 each). base_type at offset 0.
    obj0 = bytearray(OBJECT_ENTRY_SIZE)
    struct.pack_into("<H", obj0, 0, 0x0002)  # Teleporter
    struct.pack_into("<f", obj0, 0x10, 1.5)  # pos_x
    obj1 = bytearray(OBJECT_ENTRY_SIZE)
    struct.pack_into("<H", obj1, 0, 0x0005)  # Item
    obj_payload = bytes(obj0) + bytes(obj1)

    # One enemy entry (0x48). base_type Hildebear (4).
    en0 = bytearray(ENEMY_ENTRY_SIZE)
    struct.pack_into("<H", en0, 0, 0x0004)
    struct.pack_into("<H", en0, 0x0E, 1)  # wave_number
    enemy_payload = bytes(en0)

    # Events section via the builder.
    events = [
        DatEvent(
            floor=0,
            evt2=False,
            fields={
                "event_id": 41,
                "flags": 0,
                "event_type": 1,
                "room": 4,
                "wave_number": 1,
                "delay": 1,
                "action_stream_offset": 0,
            },
            actions=[DatEventAction(ACTION_UNLOCK_DOOR, {"door_id": 4})],
        ),
        DatEvent(
            floor=0,
            evt2=False,
            fields={
                "event_id": 42,
                "flags": 0,
                "event_type": 1,
                "room": 4,
                "wave_number": 2,
                "delay": 30,
                "action_stream_offset": 0,
            },
            actions=[DatEventAction(ACTION_SPAWN_NPCS, {"section_id": 0, "appear_flag": 0})],
        ),
    ]
    events_payload = _build_events_section(events)

    out = bytearray()
    out += section(SECTION_OBJECT_SETS, 0, obj_payload)
    out += section(SECTION_ENEMY_SETS, 0, enemy_payload)
    out += section(SECTION_EVENTS, 0, events_payload)
    out += b"\x00" * 16  # END marker
    return bytes(out)


def test_synthetic_dat_roundtrip():
    raw = _make_synthetic_dat()
    mf = parse_dat(raw)
    # Byte-exact round trip.
    assert serialize_dat(mf) == raw
    # Structural decode.
    assert len(mf.sections) == 3
    assert len(mf.objects) == 2
    assert len(mf.enemies) == 1
    assert len(mf.events) == 2
    assert mf.objects[0].type_id == 0x0002
    assert mf.objects[0].type_name == "Teleporter"
    assert mf.enemies[0].type_id == 0x0004
    assert mf.enemies[0].type_name == "Hildebear"
    assert mf.enemies[0].wave_number == 1
    assert mf.events[0].id == 41
    assert mf.events[0].actions[0].name == "unlock_door"
    assert mf.events[0].actions[0].args["door_id"] == 4
    assert mf.events[1].actions[0].name == "spawn_npcs"
    # decoded vs opaque reporting
    assert mf.decoded_section_types() == [
        SECTION_OBJECT_SETS,
        SECTION_ENEMY_SETS,
        SECTION_EVENTS,
    ]
    assert mf.opaque_section_types() == []


def test_synthetic_rebuild_roundtrip():
    """rebuild() (the editor path) reproduces a parsed file byte-exact."""
    raw = _make_synthetic_dat()
    mf = parse_dat(raw)
    mf.rebuild()
    assert serialize_dat(mf) == raw


def test_synthetic_mutation_persists():
    raw = _make_synthetic_dat()
    mf = parse_dat(raw)
    mf.objects[1].fields["pos_z"] = -42.25
    mf.objects[1].fields["_pos_z_bits"] = None  # force the float-write path
    mf.rebuild()
    reparsed = parse_dat(serialize_dat(mf))
    assert reparsed.objects[1].fields["pos_z"] == -42.25


def test_parse_dat_rejects_garbage():
    # data_size inconsistent with section_size.
    bad = struct.pack("<IIII", 1, 100, 0, 999) + b"\x00" * 84
    with pytest.raises(ValueError):
        parse_dat(bad)
    # section overruns the buffer.
    bad2 = struct.pack("<IIII", 1, 0x1000, 0, 0xFF0)
    with pytest.raises(ValueError):
        parse_dat(bad2)


def test_to_json_smoke():
    mf = parse_dat(_make_synthetic_dat())
    j = to_json(mf)
    assert "OBJECT_SETS" in j
    assert "Hildebear" in j
    assert '"section_count": 3' in j


def test_empty_dat():
    """A bare END marker parses to an empty MapFile and round-trips."""
    raw = b"\x00" * 16
    mf = parse_dat(raw)
    assert mf.sections == []
    assert serialize_dat(mf) == raw


# ---------------------------------------------------------------------------
# Parity gate — phantasmal-world fixtures (the .dat inside each .qst)
# ---------------------------------------------------------------------------
PW_CASES = [
    "quest118_e.qst",
    "quest27_e.qst",
]


def _decompressed_dat_from_qst(qst_path: Path) -> bytes:
    qst = parse_qst(qst_path.read_bytes())
    df = qst.dat_file
    assert df is not None, f"{qst_path.name}: no .dat in container"
    return df.decompressed()


@pytest.mark.skipif(not HAS_PW, reason="phantasmal-world fixtures not present")
@pytest.mark.parametrize("qst_name", PW_CASES)
def test_phantasmal_dat_byte_exact(qst_name):
    """parse_dat -> serialize_dat == the decompressed .dat, byte-for-byte.

    This is THE parity assertion the prompt requires: the phantasmal
    fixtures' .dat MUST round-trip byte-exact and must NOT be skipped.
    """
    dec = _decompressed_dat_from_qst(PW_FIXTURES / qst_name)
    mf = parse_dat(dec)
    rt = serialize_dat(mf)
    assert rt == dec, (
        f"{qst_name}: serialize_dat ({len(rt)}) != decompressed .dat ({len(dec)})"
    )
    # And it actually decoded structure (not just an opaque blob).
    assert len(mf.objects) > 0
    assert len(mf.enemies) > 0
    assert SECTION_OBJECT_SETS in mf.decoded_section_types()
    assert SECTION_ENEMY_SETS in mf.decoded_section_types()


@pytest.mark.skipif(not HAS_PW, reason="phantasmal-world fixtures not present")
def test_phantasmal_dat_against_standalone_fixture():
    """The .dat extracted from quest118_e.qst equals the standalone
    *_decompressed.dat fixture, and that round-trips byte-exact too."""
    standalone = PW_FIXTURES / "quest118_e_decompressed.dat"
    if not standalone.is_file():
        pytest.skip("standalone decompressed .dat fixture absent")
    dec = standalone.read_bytes()
    from_qst = _decompressed_dat_from_qst(PW_FIXTURES / "quest118_e.qst")
    assert from_qst == dec, "extracted .dat != standalone decompressed fixture"
    assert serialize_dat(parse_dat(dec)) == dec


@pytest.mark.skipif(not HAS_PW, reason="phantasmal-world fixtures not present")
def test_phantasmal_by_floor():
    dec = _decompressed_dat_from_qst(PW_FIXTURES / "quest118_e.qst")
    by_floor = parse_dat_by_floor(dec)
    # quest118 (Towards the Future) spans multiple floors.
    assert len(by_floor) > 1
    # Each floor's entity lists are consistent with the flat view.
    mf = parse_dat(dec)
    total_objs = sum(len(fd.objects) for fd in by_floor.values())
    assert total_objs == len(mf.objects)


# ---------------------------------------------------------------------------
# Corpus sweep — newserv quests + tethealla .qst tree
# ---------------------------------------------------------------------------
def _iter_corpus_dats():
    """Yield (label, decompressed_dat_bytes) for recognized corpus .dat.

    newserv ``.dat`` files are PRS-compressed on disk; tethealla ``.qst``
    files carry a PRS-compressed ``.dat`` inside the BB transport.
    Anything we don't recognize is skipped (not yielded).
    """
    # newserv bare PRS-compressed .dat files.
    if NEWSERV_QUESTS.is_dir():
        for p in NEWSERV_QUESTS.rglob("*.dat"):
            if not p.is_file():
                continue
            data = p.read_bytes()
            if len(data) < 20:
                continue  # tiny text stubs (cross-platform symlink files)
            try:
                dec = prs.decompress(data)
            except Exception:
                continue
            if not _looks_like_dat(dec):
                continue
            yield (p.name, dec)

    # tethealla .qst tree (BB online/download form only).
    if TETHEALLA_QUESTS.is_dir():
        for p in TETHEALLA_QUESTS.rglob("*.qst"):
            if not p.is_file():
                continue
            data = p.read_bytes()
            if len(data) < 4:
                continue
            sig = struct.unpack_from(">I", data, 0)[0]
            if sig not in (0x58004400, 0x5800A600):
                continue
            try:
                qst = parse_qst(data)
            except Exception:
                continue
            df = qst.dat_file
            if df is None:
                continue
            try:
                dec = df.decompressed()
            except Exception:
                continue
            if not _looks_like_dat(dec):
                continue
            yield (p.name, dec)


def _looks_like_dat(dec: bytes) -> bool:
    """Validity check: the whole section chain is self-consistent.

    Walks every section header to an END marker, requiring each to be
    in-range and consistent (``data_size == section_size - 16``). This
    rejects genuinely corrupt/truncated source (e.g. the tethealla
    ``lost havoc vulcan`` fixture, whose decompressed .dat is truncated
    mid-section) so the byte-exact gate only judges well-formed BB .dat
    — matching how the quest_bin sweep skips non-BB files. The codec
    itself raises ValueError on such input (see test_parse_dat_rejects_garbage).
    """
    if len(dec) < 16:
        return False
    pos = 0
    n = len(dec)
    while pos + 16 <= n:
        stype, size, _floor, dsize = struct.unpack_from("<IIII", dec, pos)
        if stype == 0:
            return True  # reached the END marker cleanly
        if stype not in (1, 2, 3, 4, 5):
            return False
        if size < 16 or dsize != size - 16 or pos + size > n:
            return False
        pos += size
    return False  # ran off the end without an END marker


def test_corpus_sweep_dat_roundtrip():
    """For every recognized corpus .dat: parse_dat->serialize_dat is exact.

    Logs ok/skipped/failed counts and which section types appeared
    (structurally-decoded 1/2/3 vs opaque 4/5, plus evt2 event sections).
    """
    cases = list(_iter_corpus_dats())
    if not cases:
        pytest.skip("no corpus .dat present (newserv / tethealla trees absent)")

    ok = 0
    failures = []
    seen_types = set()
    evt2_sections = 0
    opaque_sections = 0
    for label, dec in cases:
        try:
            mf = parse_dat(dec)
            rt = serialize_dat(mf)
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
            continue
        if rt != dec:
            failures.append(f"{label}: serialize_dat != decompressed input")
            continue
        ok += 1
        for s in mf.sections:
            seen_types.add(s.type)
            if s.type == SECTION_EVENTS and s.event_format == EVENT_FORMAT_EVT2:
                evt2_sections += 1
            if s.type not in (SECTION_OBJECT_SETS, SECTION_ENEMY_SETS, SECTION_EVENTS):
                opaque_sections += 1

    print(
        f"\n[quest_map corpus sweep] ok={ok} skipped(unrecognized, "
        f"not counted here) failures={len(failures)} total_recognized={len(cases)}"
    )
    print(
        f"  section types seen={sorted(seen_types)} "
        f"evt2_event_sections={evt2_sections} opaque(4/5)_sections={opaque_sections}"
    )
    if failures:
        sample = "\n  ".join(failures[:10])
        pytest.fail(f"{len(failures)} corpus round-trip failures:\n  {sample}")

    assert ok > 0, f"corpus present ({len(cases)} files) but nothing round-tripped"
