"""Parity tests for the quest assembler/disassembler (DOMAIN quest).

The acceptance invariant for Layer 0 is **byte-exact round-trip** of the
decompressed ``.bin`` code+label section::

    roundtrip_bin(decompressed) == decompressed

Oracles, in order of independence:

  1. A hand-transcribed mini-oracle. We encode a handful of opcodes BY
     HAND here (no ``formats`` import in the expected bytes) straight from
     the newserv ``QuestScript.cc`` wire spec, and assert our assembler
     reproduces them. This catches a wrong opcode/operand width without
     trusting our own table.

  2. Synthetic always-run vectors: small hand-built snippets exercising
     labels, jumps, arg_push sequences, cstrings, register sets, and the
     structured if/while/macro/.const sugar. We assert encodings are
     exact and that assemble -> disassemble -> assemble is stable.

  3. Real-corpus sweep (the primary oracle, since newserv's CLI isn't
     built here): the phantasmal-world fixtures (``quest118_e`` +
     ``quest27_e`` ``*_decompressed.bin``) MUST round-trip byte-exact;
     the newserv ``system/quests`` BB corpus round-trips too (gated on
     existence, skip-clean on a bare clone). Counts are logged.

The phantasmal fixtures live in the gitignored ``_reference/`` tree in the
psobb-studio *main* checkout, which is absent in a worktree/clean clone;
the real legs skip cleanly when the corpus isn't present.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest

from formats import prs, quest_bin
from formats import quest_asm as qa

# ---------------------------------------------------------------------------
# Reference corpus locations (read-only; gitignored, so absent in a bare
# clone or a worktree). Resolved home-relative (no committed username); an
# env var overrides the Repositories root for non-default checkouts.
# ---------------------------------------------------------------------------
_REPOS_ROOT = Path(os.environ.get("PSO_REPOS_ROOT", os.path.expanduser("~/Repositories")))
_PHANTASMAL_DIR = (
    _REPOS_ROOT
    / "psobb-studio/_reference/phantasmal-world"
    / "psolib/src/commonTest/resources"
)
_NEWSERV_QUESTS = _REPOS_ROOT / "newserv/system/quests"


# ===========================================================================
# 1. Independent hand-transcribed mini-oracle
# ===========================================================================
# Each expected byte string below is assembled BY HAND from the newserv
# QuestScript.cc wire format, WITHOUT importing anything from formats/. If
# our assembler drifts from the spec, these fail first.
#
# Wire facts used (newserv assemble_quest_script, BB_V4):
#   * one-byte opcode unless 0xF8/0xF9 prefix (then big-endian F8 NN / F9 NN)
#   * R_REG/W_REG: 1 byte register number
#   * I32: 4 bytes little-endian
#   * U8: 1 byte; U16: 2 bytes LE
#   * SCRIPT16 (label ref): 2 bytes LE label INDEX
#   * CSTRING (BB): UTF-16LE + u16 NUL terminator
#   * arg_pushl = 0x49 + i32; arg_pushw = 0x4B + u16; arg_pushb = 0x4A + u8;
#     arg_pushs = 0x4E + cstring
#   * an F_ARGS opcode emits as a bare opcode after its arg_push sequence
_HAND_VECTORS = [
    # ret  -> 0x01
    (".version BB_V4\nstart:\n    ret\n", b"\x01"),
    # nop  -> 0x00
    (".version BB_V4\nstart:\n    nop\n", b"\x00"),
    # leti r5, 0x2A  -> 09 05 2A000000
    (
        ".version BB_V4\nstart:\n    leti r5, 0x2A\n",
        b"\x09\x05" + struct.pack("<i", 0x2A),
    ),
    # jmp start  (start is label index 0)  -> 28 0000
    (".version BB_V4\nstart:\n    jmp start\n", b"\x28" + struct.pack("<H", 0)),
    # set_episode 0x00000001  (0xF8BC, I32, NOT F_ARGS) -> F8 BC 01000000
    (
        ".version BB_V4\nstart:\n    set_episode 0x1\n",
        b"\xF8\xBC" + struct.pack("<I", 1),
    ),
    # arg_pushl 0x12345678  -> 49 78563412
    (
        ".version BB_V4\nstart:\n    arg_pushl 0x12345678\n",
        b"\x49" + struct.pack("<I", 0x12345678),
    ),
    # window_msg "Hi" (0x5A, CSTRING, F_ARGS): structured form synthesises
    #   arg_pushs(0x4E) + "Hi"(UTF-16LE) + u16 NUL, then bare 0x5A.
    (
        '.version BB_V4\nstart:\n    window_msg "Hi"\n',
        b"\x4E" + "Hi".encode("utf-16-le") + b"\x00\x00" + b"\x5A",
    ),
]


@pytest.mark.parametrize("text,expected", _HAND_VECTORS)
def test_hand_oracle_encoding(text, expected):
    asm = qa.assemble(text)
    assert asm.code == expected, (
        f"hand-oracle mismatch: got {asm.code.hex()} want {expected.hex()}"
    )


def test_hand_oracle_extended_prefix_bytes():
    # The 0xF8/0xF9 page prefix is stored big-endian (F8 NN), per newserv's
    # put_u16b. Assemble a +0x100-page opcode and check the first two bytes.
    # 0xF8BC set_episode is on page 0x1BC; stored bytes start F8 BC.
    asm = qa.assemble(".version BB_V4\nstart:\n    set_episode 0x0\n")
    assert asm.code[0] == 0xF8
    assert asm.code[1] == 0xBC


# ===========================================================================
# 2. Synthetic always-run vectors
# ===========================================================================
def _roundtrip_text_stable(text: str) -> qa.AssembledCode:
    """assemble -> disassemble -> assemble must be a fixed point."""
    asm1 = qa.assemble(text)
    dis = qa.disassemble_code(asm1.code, asm1.label_offsets)
    asm2 = qa.assemble(dis.text())
    assert asm2.code == asm1.code, (
        f"unstable code: {asm1.code.hex()} -> {asm2.code.hex()}"
    )
    assert asm2.label_offsets == asm1.label_offsets, "unstable label table"
    return asm1


def test_synthetic_labels_and_jumps():
    text = (
        ".version BB_V4\n"
        ".label_count 3\n"
        "start:\n"
        "    jmp label_2\n"
        "label_1:\n"
        "    ret\n"
        "label_2:\n"
        "    jmp label_1\n"
    )
    asm = _roundtrip_text_stable(text)
    # start=label0 @0, label1 @ (1+2)=3, label2 @ (3+1)=4
    assert asm.label_offsets == [0, 3, 4]
    # jmp(0x28) label_2(idx2) ; ret(0x01) ; jmp(0x28) label_1(idx1)
    assert asm.code == (
        b"\x28" + struct.pack("<H", 2) + b"\x01" + b"\x28" + struct.pack("<H", 1)
    )


def test_synthetic_arg_push_literal_and_cstring():
    # Literal form: explicit arg_push + bare F_ARGS opcode.
    text = (
        ".version BB_V4\n"
        "start:\n"
        "    arg_pushl 0x10\n"
        '    arg_pushs "ab"\n'
        "    message\n"
        "    ret\n"
    )
    asm = _roundtrip_text_stable(text)
    expected = (
        b"\x49" + struct.pack("<I", 0x10)
        + b"\x4E" + "ab".encode("utf-16-le") + b"\x00\x00"
        + b"\x50"  # message (bare F_ARGS)
        + b"\x01"  # ret
    )
    assert asm.code == expected


def test_synthetic_reg_set_and_switch():
    # switch_jmp r0, [start, label_1] uses SCRIPT16_SET (count + u16 indices);
    # jmp_on label_1, [r1, r2] uses SCRIPT16 + R_REG_SET (count + reg bytes).
    text = (
        ".version BB_V4\n"
        ".label_count 2\n"
        "start:\n"
        "    switch_jmp r0, [start, label_1]\n"
        "label_1:\n"
        "    jmp_on label_1, [r1, r2]\n"
    )
    asm = _roundtrip_text_stable(text)
    # switch_jmp(0x40) r0 count=2 idx0 idx1
    expect_head = b"\x40\x00" + b"\x02" + struct.pack("<H", 0) + struct.pack("<H", 1)
    assert asm.code.startswith(expect_head)
    # jmp_on(0x2A) label_1(idx1) count=2 r1 r2
    expect_tail = b"\x2A" + struct.pack("<H", 1) + b"\x02\x01\x02"
    assert asm.code.endswith(expect_tail)


def test_synthetic_const_directive():
    a = qa.assemble(".version BB_V4\n.const FOO 0x2A\nstart:\n    leti r5, FOO\n")
    assert a.code == b"\x09\x05" + struct.pack("<i", 0x2A)


def test_synthetic_float32():
    a = qa.assemble(".version BB_V4\nstart:\n    fleti r1, 1.5\n")
    # fleti = 0x204 (page 0x2 -> stored F9 04), W_REG, FLOAT32
    assert a.code[:2] == b"\xF9\x04"
    assert a.code[2] == 0x01
    assert struct.unpack("<f", a.code[3:7])[0] == 1.5


def test_synthetic_while_expands_to_jumps():
    # while r0 < r10 { addi r0, 1 }  ->  top: jmpi/jmp-ge past body; jmp top
    text = (
        ".version BB_V4\n"
        "start:\n"
        "    leti r0, 0\n"
        "    while r0 < r10\n"
        "    addi r0, 1\n"
        "    endwhile\n"
        "    ret\n"
    )
    asm = qa.assemble(text)
    # It must contain a back-edge jmp (0x28) and a conditional jmp_ge (0x3A,
    # the inverse of '<' in reg/reg form). Just assert it assembles and that
    # disassembling the loop body and reassembling is stable.
    dis = qa.disassemble_code(asm.code, asm.label_offsets)
    asm2 = qa.assemble(dis.text())
    assert asm2.code == asm.code
    assert b"\x28" in asm.code  # back-edge jmp present
    assert b"\x3A" in asm.code  # jmp_ge (inverted '<') present


def test_synthetic_macro_expansion():
    text = (
        ".version BB_V4\n"
        "%macro setr reg, val\n"
        "    leti %reg, %val\n"
        "%endmacro\n"
        "start:\n"
        "    setr r3, 0x7\n"
        "    ret\n"
    )
    a = qa.assemble(text)
    assert a.code == b"\x09\x03" + struct.pack("<i", 7) + b"\x01"


def test_structured_args_match_literal_args():
    # The structured F_ARGS form must produce the SAME bytes as writing the
    # arg_push sequence by hand. set_floor_handler {I32, SCRIPT16} F_ARGS:
    #   I32 value 0x0 (<=0xFF) -> arg_pushb 0x00
    #   SCRIPT16 label `start` (index 0) -> arg_pushw 0x0000
    # then a bare 0x95.
    structured = qa.assemble(
        ".version BB_V4\nstart:\n    set_floor_handler 0x0, start\n"
    )
    literal = qa.assemble(
        ".version BB_V4\nstart:\n"
        "    arg_pushb 0x0\n"
        "    arg_pushw 0x0\n"
        "    set_floor_handler\n"
    )
    assert structured.code == literal.code
    # And that byte sequence, spelled out: 4A 00  4B 0000  95
    assert structured.code == b"\x4A\x00\x4B\x00\x00\x95"


# ===========================================================================
# 3a. Phantasmal-world fixtures (MUST round-trip byte-exact when present)
# ===========================================================================
def _phantasmal_decompressed(name: str) -> bytes | None:
    p = _PHANTASMAL_DIR / name
    if not p.is_file():
        return None
    return p.read_bytes()


@pytest.mark.parametrize("name", ["quest118_e_decompressed.bin", "quest27_e_decompressed.bin"])
def test_phantasmal_fixture_roundtrip_byte_exact(name):
    data = _phantasmal_decompressed(name)
    if data is None:
        pytest.skip(f"phantasmal fixture {name} not present (bare clone)")
    out = qa.roundtrip_bin(data)
    assert out == data, f"{name}: round-trip is not byte-exact"


# ===========================================================================
# 3b. newserv BB corpus sweep (gated on existence)
# ===========================================================================
def _newserv_bb_bins() -> list[Path]:
    if not _NEWSERV_QUESTS.is_dir():
        return []
    return sorted(_NEWSERV_QUESTS.glob("**/*-bb-*.bin"))


def test_newserv_bb_corpus_roundtrip(capsys):
    bins = _newserv_bb_bins()
    if not bins:
        pytest.skip("newserv quest corpus not present")

    ok = skipped = failed = 0
    failures: list[str] = []
    for path in bins:
        raw = path.read_bytes()
        try:
            dec = prs.decompress(raw)
            qb = quest_bin.parse_bin(dec)
        except Exception:
            skipped += 1
            continue
        if qb.fmt != quest_bin.BIN_FORMAT_BB:
            skipped += 1
            continue
        try:
            out = qa.roundtrip_bin(dec)
        except Exception as exc:  # noqa: BLE001 - we want the message
            failed += 1
            if len(failures) < 20:
                failures.append(f"{path.name}: {exc}")
            continue
        if out == dec:
            ok += 1
        else:
            failed += 1
            if len(failures) < 20:
                failures.append(f"{path.name}: byte-diff")

    with capsys.disabled():
        total = ok + failed
        pct = (100.0 * ok / total) if total else 0.0
        print(
            f"\n[quest-parity] newserv BB corpus: ok={ok} skipped={skipped} "
            f"failed={failed}  coverage={pct:.1f}%"
        )
        for f in failures:
            print(f"  FAIL {f}")

    # The whole shipped BB corpus must round-trip byte-exact.
    assert failed == 0, f"{failed} BB quests failed round-trip; first: {failures[:5]}"
    assert ok > 0, "expected at least one BB quest to round-trip"
