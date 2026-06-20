"""Unit tests for the quest assembler/disassembler internals (DOMAIN quest).

These complement ``tests/test_quest_parity.py`` (which owns the byte-exact
round-trip acceptance gate). Here we pin down the smaller behaviours: the
disassembly model, data-region handling, comment/whitespace tolerance,
error reporting, and the version gate.
"""
from __future__ import annotations

import struct

import pytest

from formats import quest_asm as qa
from formats import quest_bin


def test_disassembly_text_has_version_and_label_count():
    asm = qa.assemble(".version BB_V4\nstart:\n    ret\n")
    dis = qa.disassemble_code(asm.code, asm.label_offsets)
    text = dis.text()
    assert text.startswith(".version BB_V4")
    assert ".label_count" in text
    assert "label_0:" in text  # implicit entry label at offset 0


def test_data_region_emitted_as_hex_and_roundtrips():
    # Build a tiny container by hand: a code label (index 0) at offset 0 that
    # decodes as `ret`, plus a data label (index 1) whose region is bytes
    # that cannot decode as instructions (0xFF is an unknown opcode).
    code = b"\x01" + b"\xff\xff\xff\xff"  # ret, then 4 raw data bytes
    label_offsets = [0, 1]  # label 0 -> code @0 ; label 1 -> data @1
    dis = qa.disassemble_code(code, label_offsets)
    text = dis.text()
    assert ".data" in text
    # Reassembling reproduces the code + label table exactly.
    asm = qa.assemble(text)
    assert asm.code == code
    assert asm.label_offsets == label_offsets


def test_comments_and_blank_lines_tolerated():
    text = (
        ".version BB_V4   // a header comment\n"
        "\n"
        "start:           ; a label\n"
        "    leti r5, 0x2A  // set a register\n"
        "    ret\n"
    )
    asm = qa.assemble(text)
    assert asm.code == b"\x09\x05" + struct.pack("<i", 0x2A) + b"\x01"


def test_unknown_opcode_in_text_raises():
    with pytest.raises(qa.AssemblyError) as e:
        qa.assemble(".version BB_V4\nstart:\n    not_a_real_opcode r0\n")
    assert "unknown opcode" in str(e.value).lower()


def test_unknown_directive_raises():
    with pytest.raises(qa.AssemblyError):
        qa.assemble(".version BB_V4\nstart:\n    .bogus 1\n")


def test_duplicate_label_raises():
    with pytest.raises(qa.AssemblyError):
        qa.assemble(".version BB_V4\nstart:\n    ret\nstart:\n    ret\n")


def test_undefined_named_label_reference_raises():
    # A non-"label_NN" undefined reference is a hard error.
    with pytest.raises(qa.AssemblyError):
        qa.assemble(".version BB_V4\nstart:\n    jmp nowhere\n")


def test_label_index_out_of_range_reference_roundtrips():
    # `label_999` with no definition is byte-faithful: it stores the raw
    # index 999 (points outside the code section). This mirrors how real
    # quests reference data/unused label indices.
    asm = qa.assemble(".version BB_V4\n.label_count 1\nstart:\n    jmp label_999\n")
    assert asm.code == b"\x28" + struct.pack("<H", 999)


def test_lone_prefix_byte_becomes_data_region():
    # A lone 0xF8 prefix with no following byte cannot decode as an
    # instruction, so the region classifier treats it as a data blob and it
    # round-trips verbatim (rather than crashing the disassembler).
    dis = qa.disassemble_code(b"\xF8", [0])
    text = dis.text()
    assert ".data F8" in text
    assert qa.assemble(text).code == b"\xF8"


def test_assemble_to_bin_splices_header_verbatim():
    # roundtrip a real-ish container: parse a minimal hand-built .bin, change
    # nothing, and confirm assemble_to_bin reproduces the file.
    # Build a minimal BB-shaped container is overkill; instead use the
    # PC/generic carve: header(16) + code + labels.
    code = b"\x01"  # ret
    header = bytearray(16)
    # code_offset=16, label_table_offset=17, size=21, marker=0xFFFFFFFF
    struct.pack_into("<IIII", header, 0, 16, 17, 21, 0xFFFFFFFF)
    label_blob = struct.pack("<I", 0)  # one label @ offset 0
    raw = bytes(header) + code + label_blob
    qb = quest_bin.parse_bin(raw)
    text = qa.disassemble_code(qb.code, qb.label_offsets, version="BB_V4").text()
    out = qa.assemble_to_bin(text, qb, version="BB_V4")
    assert out == raw


def test_float32_roundtrip_is_bit_exact():
    # A float whose shortest-decimal repr might lose precision must still
    # re-encode to the identical 32-bit pattern.
    for val in (0.1, 1.5, 3.14159, -2.0, 1e-7):
        packed = struct.pack("<f", val)
        f32 = struct.unpack("<f", packed)[0]
        text = qa.assemble(f".version BB_V4\nstart:\n    fleti r0, {f32!r}\n")
        # bytes 3..7 are the float operand (after F9 04 reg).
        assert text.code[3:7] == packed
