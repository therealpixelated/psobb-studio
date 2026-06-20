"""Tests for ``formats/quest_opcodes.py`` — the PSOBB quest-VM opcode table.

These tests exercise the committed, self-contained opcode model (no
``_reference/`` access at runtime): the self-test passes, the table is large
enough, known opcodes resolve by code AND mnemonic, the 0xF8/0xF9 extended-page
prefix math round-trips, F_ARGS opcodes are stack-sourced, and qedit aliases
resolve.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from formats import quest_opcodes as q  # noqa: E402


def test_self_test_passes():
    assert q.check_opcode_definitions() is True


def test_at_least_400_opcodes_loaded():
    assert len(q.OPCODES) >= 400


def test_bb_is_default_version():
    assert q.DEFAULT_VERSION == "BB_V4"
    # the bulk of the table is valid on BB
    assert len(q.opcodes_for_version()) >= 400
    assert len(q.opcodes_for_version("BB_V4")) == len(q.opcodes_for_version())


# ---------------------------------------------------------------------------
# Spot-checks: known opcode code <-> mnemonic.
# ---------------------------------------------------------------------------
KNOWN = {
    0x00: "nop",
    0x01: "ret",
    0x02: "sync",
    0x04: "thread",
    0x28: "jmp",
    0x29: "call",
    0x40: "switch_jmp",
    0x41: "switch_call",
    0x48: "arg_pushr",
    0x49: "arg_pushl",
    0x4A: "arg_pushb",
    0x4B: "arg_pushw",
    0x4C: "arg_pusha",
    0x4D: "arg_pusho",
    0x4E: "arg_pushs",
}


@pytest.mark.parametrize("code,mnemonic", KNOWN.items())
def test_known_opcode_by_code(code, mnemonic):
    op = q.OPCODES[code]
    assert op.mnemonic == mnemonic
    assert op.code == code


@pytest.mark.parametrize("code,mnemonic", KNOWN.items())
def test_known_opcode_by_mnemonic(code, mnemonic):
    op = q.by_mnemonic(mnemonic)
    assert op is not None
    assert op.code == code


def test_conditional_jmps_present():
    # The conditional jump family (jmp_=, jmpi_=, jmp_!=, ujmp_>, ...) at 0x2C..
    for code in range(0x2C, 0x40):
        op = q.OPCODES.get(code)
        assert op is not None, f"missing conditional-jump opcode 0x{code:X}"
        # each takes a label operand somewhere in its signature
        assert any(p.is_label for p in op.params), f"0x{code:X} has no label operand"


# ---------------------------------------------------------------------------
# Terminators / basic-block ends.
# ---------------------------------------------------------------------------
def test_ret_and_jmp_are_terminators():
    assert q.OPCODES[0x01].is_terminator  # ret
    assert q.OPCODES[0x28].is_terminator  # jmp
    # call does NOT end a basic block (control returns)
    assert not q.OPCODES[0x29].is_terminator
    assert q.OPCODES[0x29].clears_args


# ---------------------------------------------------------------------------
# Extended-page prefix math (0xF8 -> +0x100, 0xF9 -> +0x200).
# ---------------------------------------------------------------------------
def test_prefix_encode_decode_roundtrip():
    # one-byte opcode: no prefix
    assert q.encode_prefix(0x28) == (None, 0x28)
    assert q.decode_prefix(None, 0x28) == 0x28
    # F8 page
    assert q.encode_prefix(0x101) == (0xF8, 0x01)
    assert q.decode_prefix(0xF8, 0x01) == 0x101
    assert q.decode_prefix(0xF8, 0x00) == 0x100
    assert q.decode_prefix(0xF8, 0xFF) == 0x1FF
    # F9 page
    assert q.encode_prefix(0x25D) == (0xF9, 0x5D)
    assert q.decode_prefix(0xF9, 0x5D) == 0x25D
    assert q.decode_prefix(0xF9, 0x00) == 0x200


def test_prefix_math_for_every_extended_opcode():
    for code, op in q.OPCODES.items():
        prefix, low = q.encode_prefix(code)
        assert q.decode_prefix(prefix, low) == code
        if code > 0xFF:
            assert op.is_extended
            assert prefix in (0xF8, 0xF9)
            # the stored low byte equals code & 0xFF
            assert low == (code & 0xFF)
        else:
            assert not op.is_extended
            assert prefix is None


def test_extended_pages_have_members():
    f8 = [c for c in q.OPCODES if 0x100 <= c <= 0x1FF]
    f9 = [c for c in q.OPCODES if 0x200 <= c <= 0x2FF]
    assert f8, "no F8-page opcodes loaded"
    assert f9, "no F9-page opcodes loaded"
    # a known F8 opcode and a known F9 opcode
    assert q.OPCODES[0x101].prefix == 0xF8
    assert q.OPCODES[0x201].prefix == 0xF9


def test_decode_prefix_rejects_bad_prefix():
    with pytest.raises(ValueError):
        q.decode_prefix(0xF7, 0x01)
    with pytest.raises(ValueError):
        q.decode_prefix(0xF8, 0x100)


# ---------------------------------------------------------------------------
# F_ARGS opcodes take operands from the arg stack (no inline operand bytes).
# ---------------------------------------------------------------------------
def test_arg_push_family_flagged():
    for code in range(0x48, 0x4F):
        op = q.OPCODES[code]
        assert op.is_arg_push, f"0x{code:X} not flagged F_PUSH_ARG"
        assert not op.uses_arg_stack
        assert len(op.params) == 1


def test_sample_f_args_opcodes_are_stack_sourced():
    # message (0x50), list (0x51), sound_effect (0x54), bgm (0x55) consume
    # their operands from the argument stack on BB.
    for code in (0x50, 0x51, 0x54, 0x55):
        op = q.OPCODES[code]
        assert op.uses_arg_stack, f"0x{code:X} ({op.mnemonic}) not F_ARGS"
        # an F_ARGS opcode is never itself an arg pusher
        assert not op.is_arg_push
        # and never a basic-block terminator
        assert not op.is_terminator


def test_there_are_many_f_args_opcodes():
    n = sum(1 for op in q.OPCODES.values() if op.uses_arg_stack)
    assert n >= 100


# ---------------------------------------------------------------------------
# qedit aliases resolve.
# ---------------------------------------------------------------------------
def test_qedit_alias_resolves():
    # 0x54 sound_effect has the qedit alias "se".
    op = q.OPCODES[0x54]
    assert op.qedit_alias == "se"
    assert q.by_mnemonic("se") is op
    assert q.by_mnemonic("sound_effect") is op


def test_some_qedit_aliases_present():
    aliased = [op for op in q.OPCODES.values() if op.qedit_alias]
    assert len(aliased) >= 50
    # every alias resolves back to its opcode
    for op in aliased:
        resolved = q.by_mnemonic(op.qedit_alias)
        assert resolved is not None


# ---------------------------------------------------------------------------
# Model integrity.
# ---------------------------------------------------------------------------
def test_param_types_are_enum_members():
    for op in q.OPCODES.values():
        for p in op.params:
            assert isinstance(p.type, q.ParamType)


def test_register_write_params_are_registers():
    for op in q.OPCODES.values():
        for p in op.params:
            if p.writes:
                assert p.is_register, f"{op.mnemonic}: non-register written param {p.type}"


def test_let_writes_a_register():
    # leti (0x09): W_REG, I32
    op = q.OPCODES[0x09]
    assert op.mnemonic == "leti"
    assert op.params[0].type == q.ParamType.W_REG
    assert op.params[0].writes


def test_no_duplicate_codes_in_records():
    from formats._quest_opcode_table import OPCODE_RECORDS

    codes = [rec[0] for rec in OPCODE_RECORDS]
    assert len(codes) == len(set(codes))
