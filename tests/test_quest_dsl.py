"""Tests for formats.quest_dsl — the Layer-1 author-friendly Quest DSL.

These are synthetic always-run tests: they author small ``.quest`` programs
inline, compile them through the full Layer-1 -> Layer-0 pipeline, and assert
the produced bytecode is *structurally valid* (assembles, parses back through
:mod:`formats.quest_bin` with a header + label table + code section).

Coverage:
  * variables + if/while/for control flow,
  * an NPC with on_talk dialogue and a quest_flag set,
  * a thread{} (asserting the emitted asm STARTS with the mandatory sync),
  * a wave/spawn, give_item, a menu choose{},
  * a round-trip-ish lift: compile -> lift_bin -> recompile assembles
    (byte-identical where the structured lift succeeds; still-assembles for
    the asm{} fallback),
  * diagnostics carry line/col with the right exception type.

No ``_reference/`` data is needed; the codec is fully synthetic.
"""
from __future__ import annotations

import pytest

from formats import quest_asm, quest_bin
from formats import quest_dsl as dsl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _assert_assembles(asm_text: str) -> quest_asm.AssembledCode:
    """Assemble Layer-0 text and sanity-check the output."""
    ac = quest_asm.assemble(asm_text)
    assert isinstance(ac.code, (bytes, bytearray))
    assert len(ac.code) > 0
    assert isinstance(ac.label_offsets, list)
    return ac


def _assert_valid_bin(bin_bytes: bytes) -> quest_bin.QuestBin:
    """Parse a compiled .bin and assert it is structurally valid."""
    qb = quest_bin.parse_bin(bin_bytes)
    # header + code + label table all present and coherent
    assert qb.fmt == quest_bin.BIN_FORMAT_BB
    assert qb.code_offset == quest_bin.CODE_OFFSET_BB
    assert len(qb.header_raw) == qb.code_offset
    assert len(qb.code) > 0
    assert qb.label_table_offset == qb.code_offset + len(qb.code)
    assert len(qb.label_offsets) >= 1
    # round-trips through serialize without loss
    assert quest_bin.serialize_bin(qb) == bin_bytes
    return qb


def _compile_full(src: str) -> quest_bin.QuestBin:
    """Compile DSL -> asm (assert assembles) -> bin (assert valid)."""
    asm_text = dsl.compile_dsl(src)
    _assert_assembles(asm_text)
    bin_bytes = dsl.compile_dsl_to_bin(src)
    return _assert_valid_bin(bin_bytes)


# ---------------------------------------------------------------------------
# Program 1 — variables + if/while/for
# ---------------------------------------------------------------------------
PROG_CONTROL_FLOW = """
quest "Control Flow" {
    episode 1

    var kills = 0
    var target = 3

    thread main {
        while kills < target {
            kills = kills + 1
        }
        for i in 0 .. 2 {
            set_flag i
        }
        if kills == 3 {
            quest_success = 1
        } elif kills > 3 {
            quest_success = 0
        } else {
            set_flag 99
        }
    }
}
"""


def test_control_flow_compiles_and_is_valid_bin():
    qb = _compile_full(PROG_CONTROL_FLOW)
    assert qb.name == "Control Flow"
    # episode 1 (DSL) maps to the 0-indexed header byte 0.
    assert qb.episode == 0


def test_variables_use_reserved_slots_correctly():
    asm_text = dsl.compile_dsl(PROG_CONTROL_FLOW)
    # quest_success is the reserved R255 slot.
    assert "r255" in asm_text
    # ordinary user variables start at R200.
    assert "r200" in asm_text


# ---------------------------------------------------------------------------
# Program 2 — NPC with on_talk dialogue + quest_flag set
# ---------------------------------------------------------------------------
PROG_NPC = """
quest "Talk To Me" {
    episode 1

    npc Tyrel {
        skin 27
        floor 0
        section 0
        pos 100.0, 0.0, -20.0
        dir 0.0
        on_talk {
            window_msg "Good luck out there."
            quest_flag 61, 1
        }
    }
}
"""


def test_npc_with_on_talk_compiles():
    qb = _compile_full(PROG_NPC)
    assert qb.name == "Talk To Me"


def test_npc_emits_talkable_creation_opcode():
    asm_text = dsl.compile_dsl(PROG_NPC)
    # an NPC with on_talk is created talkable via npc_crptalk over a 6-reg
    # descriptor window.
    assert "npc_crptalk" in asm_text
    # the dialogue body became a subroutine with the window message.
    assert "window_msg" in asm_text
    # and the quest_flag set lowered to set_eventflag.
    assert "set_eventflag" in asm_text


# ---------------------------------------------------------------------------
# Program 3 — thread{} MUST start with the mandatory sync
# ---------------------------------------------------------------------------
PROG_THREAD = """
quest "Worker" {
    episode 1
    thread worker {
        message 1, "Working..."
        set_flag 5
    }
}
"""


def test_thread_body_starts_with_sync():
    asm_text = dsl.compile_dsl(PROG_THREAD)
    lines = [ln.strip() for ln in asm_text.splitlines()]
    label_idx = lines.index("thread_worker:")
    # The very first instruction of the thread body is `sync` — a quest
    # thread that does not start with sync crashes the game.
    assert lines[label_idx + 1] == "sync"


def test_thread_compiles_and_is_valid_bin():
    _compile_full(PROG_THREAD)


def test_every_thread_block_starts_with_sync_multi():
    src = """
    quest "Two Threads" {
        episode 1
        thread a { set_flag 1 }
        thread b { set_flag 2 }
    }
    """
    asm_text = dsl.compile_dsl(src)
    lines = [ln.strip() for ln in asm_text.splitlines()]
    for tname in ("thread_a:", "thread_b:"):
        idx = lines.index(tname)
        assert lines[idx + 1] == "sync", f"{tname} missing leading sync"


# ---------------------------------------------------------------------------
# Program 4 — wave/spawn + give_item + floor_handler
# ---------------------------------------------------------------------------
PROG_SPAWN = """
quest "Forest Ambush" {
    episode 1

    floor_handler floor=0 {
        spawn npc=8 floor=0 section=0 x=10.0 y=0.0 z=-5.0 dir=0.0
        wave npc=8 floor=0 section=0 count=3 x=20.0 y=0.0 z=0.0 dir=0.0
    }

    thread main {
        give_item 0x00, 0x01, 0x00
    }
}
"""


def test_spawn_and_wave_and_give_item():
    qb = _compile_full(PROG_SPAWN)
    assert qb.name == "Forest Ambush"
    asm_text = dsl.compile_dsl(PROG_SPAWN)
    # floor handler registered
    assert "set_floor_handler" in asm_text
    # NPC creation (non-talkable spawn uses npc_crp)
    assert "npc_crp" in asm_text
    # give_item lowers to item_create over a staged 3-register descriptor
    assert "item_create" in asm_text


def test_wave_count_spawns_multiple():
    asm_text = dsl.compile_dsl(PROG_SPAWN)
    # wave count=3 + 1 standalone spawn = 4 npc creations total in the
    # floor-handler subroutine.
    assert asm_text.count("npc_crp") >= 4


# ---------------------------------------------------------------------------
# Program 5 — menu choose{}
# ---------------------------------------------------------------------------
PROG_CHOOSE = """
quest "Pick A Door" {
    episode 1
    thread main {
        window_msg "Choose a door."
        choose "Which door?" {
            "Left"  -> { set_flag 10 }
            "Right" -> { set_flag 11 }
            "Run"   -> { quest_success = 0 }
        }
    }
}
"""


def test_choose_menu_compiles():
    qb = _compile_full(PROG_CHOOSE)
    assert qb.name == "Pick A Door"


def test_choose_lowers_to_list_and_dispatch():
    asm_text = dsl.compile_dsl(PROG_CHOOSE)
    # The choice menu uses the `list` opcode to read a selection, then a
    # jmpi_ne dispatch chain.
    assert "list" in asm_text
    assert "jmpi_ne" in asm_text


# ---------------------------------------------------------------------------
# Round-trip-ish: compile -> lift_bin -> recompile assembles
# ---------------------------------------------------------------------------
PROG_LIFTABLE = """
quest "Lift Me" {
    episode 2
    thread main {
        window_msg "Greetings."
        set_flag 10
        clear_flag 11
        message 5, "Bye now."
    }
}
"""


def test_lift_then_recompile_is_byte_identical_when_structured():
    bin_bytes = dsl.compile_dsl_to_bin(PROG_LIFTABLE)
    res = dsl.lift_bin_detailed(bin_bytes)
    # this quest is a clean boot + single thread -> fully structured lift
    assert res.fallbacks == []
    assert any("thread main" in r for r in res.recognised)
    recompiled = dsl.compile_dsl_to_bin(res.dsl_text)
    qb_a = quest_bin.parse_bin(bin_bytes)
    qb_b = quest_bin.parse_bin(recompiled)
    assert qb_b.code == qb_a.code
    assert qb_b.label_offsets == qb_a.label_offsets


def test_lift_fallback_still_assembles_and_is_byte_exact():
    # A quest with boot-time state (a var in the quest body) does not match
    # the simple boot shape, so it lifts to a whole-program asm{} block. That
    # must still recompile, and the asm{} path reassembles byte-for-byte.
    src = """
    quest "Has Boot State" {
        episode 1
        var x = 5
        thread main { set_flag 1 }
    }
    """
    bin_bytes = dsl.compile_dsl_to_bin(src)
    res = dsl.lift_bin_detailed(bin_bytes)
    assert res.fallbacks  # fell back to asm{}
    # the lifted DSL recompiles...
    recompiled = dsl.compile_dsl_to_bin(res.dsl_text)
    _assert_valid_bin(recompiled)
    # ...to byte-identical code (the asm{} block is verbatim).
    qb_a = quest_bin.parse_bin(bin_bytes)
    qb_b = quest_bin.parse_bin(recompiled)
    assert qb_b.code == qb_a.code
    assert qb_b.label_offsets == qb_a.label_offsets


def test_lift_plain_text_entry_point():
    # the convenience lift_bin() returns just the text
    bin_bytes = dsl.compile_dsl_to_bin(PROG_LIFTABLE)
    text = dsl.lift_bin(bin_bytes)
    assert text.startswith("quest ")
    assert "episode 2" in text


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def test_undefined_variable_assignment_raises_with_line_col():
    src = 'quest "t" {\n    thread main {\n        x = 5\n    }\n}\n'
    with pytest.raises(dsl.DSLSemanticError) as e:
        dsl.compile_dsl(src)
    assert e.value.line == 3
    assert e.value.col > 0
    assert "undefined variable" in e.value.message


def test_undefined_variable_reference_raises():
    src = 'quest "t" {\n    thread main {\n        var y = z\n    }\n}\n'
    with pytest.raises(dsl.DSLSemanticError) as e:
        dsl.compile_dsl(src)
    assert e.value.line == 3
    assert "undefined variable 'z'" in e.value.message


def test_unknown_construct_raises():
    src = 'quest "t" {\n    thread main {\n        frobnicate 1\n    }\n}\n'
    with pytest.raises(dsl.DSLSemanticError) as e:
        dsl.compile_dsl(src)
    assert "frobnicate" in e.value.message


def test_type_mismatch_float_to_int_raises():
    src = (
        'quest "t" {\n'
        "    thread main {\n"
        "        var n = 1\n"
        "        n = 2.5\n"
        "    }\n"
        "}\n"
    )
    with pytest.raises(dsl.DSLSemanticError) as e:
        dsl.compile_dsl(src)
    assert "float" in e.value.message.lower()


def test_unterminated_block_is_syntax_error():
    src = 'quest "t" {\n    thread main {\n        set_flag 1\n'
    with pytest.raises(dsl.DSLSyntaxError) as e:
        dsl.compile_dsl(src)
    assert e.value.line > 0


def test_reserved_register_declaration_rejected():
    src = 'quest "t" {\n    var difficulty = 3\n    thread main {}\n}\n'
    with pytest.raises(dsl.DSLSemanticError) as e:
        dsl.compile_dsl(src)
    assert "reserved" in e.value.message


def test_message_requires_id_and_text():
    src = 'quest "t" {\n    thread main {\n        message "no id"\n    }\n}\n'
    with pytest.raises(dsl.DSLSemanticError) as e:
        dsl.compile_dsl(src)
    assert e.value.line == 3


# ---------------------------------------------------------------------------
# Misc: asm{} escape hatch + reserved register access
# ---------------------------------------------------------------------------
def test_asm_escape_hatch_passes_through():
    src = """
    quest "Raw" {
        episode 1
        thread main {
            asm {
                "leti r210, 0x2A"
                "addi r210, 0x1"
            }
        }
    }
    """
    asm_text = dsl.compile_dsl(src)
    assert "leti r210, 0x2A" in asm_text
    assert "addi r210, 0x1" in asm_text
    _assert_assembles(asm_text)


def test_difficulty_reserved_register_is_readable():
    src = """
    quest "Diff" {
        episode 1
        thread main {
            var d = difficulty
            if d == 3 { set_flag 1 }
        }
    }
    """
    asm_text = dsl.compile_dsl(src)
    # difficulty is reserved R250
    assert "r250" in asm_text
    _assert_assembles(asm_text)
