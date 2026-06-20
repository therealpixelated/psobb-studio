"""PSOBB quest-VM assembler + disassembler — Layer 0 keystone.

This module turns the **raw CODE bytes** of a decompressed ``.bin`` quest
script (carved out by :mod:`formats.quest_bin`) into human-editable
assembly text, and back again, byte-for-byte.

It sits one level above :mod:`formats.quest_opcodes` (the typed opcode
table) and :mod:`formats.quest_bin` (the container codec): quest_bin owns
the header / label table / string blob and hands us the raw ``code``
bytes plus the ``label_offsets`` table; we decode those bytes into
instructions and resolve control-flow through the label table to
*symbolic* labels.

The acceptance invariant (the Definition-of-Done) is **byte-exact
round-trip** on every shipped Blue Burst quest::

    roundtrip_bin(decompressed) == decompressed

Two text levels
---------------
There are two faithful representations of the same code, and the choice
between them is the heart of getting parity right:

* **Literal level** (what :func:`disassemble_bin` emits, and what the
  parity gate exercises): every instruction — *including* each
  ``arg_pushX`` (0x48-0x4E) — appears as its own line with its own typed
  operand, and an ``F_ARGS`` opcode (one that consumes the argument
  stack on v3+) appears bare. This is a 1:1 mirror of the bytes: the
  assembler re-emits exactly the opcode widths the original used, so the
  round-trip can never drift even if a quest pushed an argument with a
  wider/narrower ``arg_push`` than a value-magnitude heuristic would
  pick. The literal disassembly NEVER collapses an arg-stack sequence.

* **Structured level** (an *authoring convenience*, opt-in): an
  ``F_ARGS`` opcode may be written with its operands inline (``message
  0x1234, "hi"``) and the assembler synthesises the required
  ``arg_pushX`` sequence in front of it, exactly the way newserv's
  assembler does. Plus ``.const``, ``%macro``/``%endmacro``, and
  structured ``if`` / ``while`` blocks that expand to conditional jumps
  (the ca65hl trick — zero runtime cost) so Layer 0 alone already beats
  hand-writing raw jump tables.

Because the structured sugar can produce *different* bytes than a given
quest happened to use (newserv's arg_push width heuristic, a macro's
chosen labels, etc.), it is never emitted by the disassembler; only the
literal form is. That keeps the parity oracle honest.

Wire format (ground truth: newserv ``src/QuestScript.cc``
``disassemble_quest_script`` / ``assemble_quest_script``, BB_V4):

* An opcode is one byte, unless the byte is ``0xF8`` or ``0xF9`` (the
  extended-page prefixes), in which case the full opcode is the 16-bit
  big-endian value ``(prefix << 8) | next_byte`` — i.e. stored ``F8 NN``
  / ``F9 NN`` selects page +0x100 / +0x200. (See
  :func:`formats.quest_opcodes.encode_prefix`.)
* Inline operands follow per the opcode's param signature, little-endian:
  registers are 1 byte (or 4 for the ``*32`` variants), ``I8``/``U8`` 1
  byte, ``I16``/``U16`` 2, ``I32``/``U32``/``FLOAT32``/label refs 4 (or 2
  for the 16-bit label types ``SCRIPT16``/``DATA16``), a ``*_SET`` type
  is a count byte then that many elements, a ``*_SET_FIXED`` is a single
  start register (the opcode implicitly spans ``count`` consecutive
  registers), and ``CSTRING`` is a NUL-terminated UTF-16LE string on BB.
* For an ``F_ARGS`` opcode on a version that has the argument stack (v3
  and later, which includes BB), the opcode carries **no** inline
  operands: its operands were pushed by preceding ``arg_pushX`` opcodes.

Labels: control-flow targets are stored as 16/32-bit **indices into the
``.bin`` label table** (``label_offsets``), which maps index -> byte
offset within the code section. We resolve every label-typed operand to
a symbolic ``label_NN`` and emit a ``label_NN:`` definition at the code
offset that table entry points to.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from formats import quest_bin
from formats import quest_opcodes as q
from formats.quest_opcodes import DEFAULT_VERSION, Opcode, ParamType

__all__ = [
    "Instruction",
    "Disassembly",
    "AssembledCode",
    "AssemblyError",
    "disassemble_code",
    "disassemble_bin",
    "assemble",
    "assemble_to_bin",
    "roundtrip_bin",
]

# Argument-push opcode codes (0x48..0x4E) and the bare opcode bytes used
# by the structured-sugar expander. Kept here so the assembler can hide
# the F_ARGS convention without re-deriving them.
ARG_PUSHR = 0x48  # push register value
ARG_PUSHL = 0x49  # push 32-bit literal
ARG_PUSHB = 0x4A  # push 8-bit literal (also register *number* for out-params)
ARG_PUSHW = 0x4B  # push 16-bit literal (also a label index)
ARG_PUSHA = 0x4C  # push address of register
ARG_PUSHO = 0x4D  # push address of label
ARG_PUSHS = 0x4E  # push string


class AssemblyError(ValueError):
    """Raised on any malformed assembly text or undecodable code byte."""

    def __init__(self, message: str, *, line: Optional[int] = None) -> None:
        self.line = line
        if line is not None:
            message = f"line {line}: {message}"
        super().__init__(message)


# ---------------------------------------------------------------------------
# Disassembly model
# ---------------------------------------------------------------------------
@dataclass
class Instruction:
    """One decoded instruction in literal form.

    ``offset`` is the byte offset of this instruction within the code
    section; ``size`` its encoded length. ``operands`` is the list of
    already-formatted operand strings (registers as ``rN``, immediates as
    ``0x...``, labels as ``label_NN``, sets as ``[a, b]``...). ``opcode``
    is the full opcode code.
    """

    offset: int
    size: int
    opcode: int
    mnemonic: str
    operands: List[str] = field(default_factory=list)

    def text(self) -> str:
        body = self.mnemonic
        if self.operands:
            body += " " + ", ".join(self.operands)
        return body


@dataclass
class Disassembly:
    """Result of disassembling a code section."""

    version: str
    lines: List[str]
    instructions: List[Instruction]
    # label index -> code offset (mirrors the .bin label table, but only
    # the entries that actually point into the code section).
    label_offsets: List[int]

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


# ---------------------------------------------------------------------------
# Disassembler
# ---------------------------------------------------------------------------
def _u8(b: bytes, o: int) -> int:
    if o + 1 > len(b):
        raise AssemblyError(f"truncated: needed 1 byte at 0x{o:X}")
    return b[o]


def _read(fmt: str, size: int, b: bytes, o: int) -> int:
    if o + size > len(b):
        raise AssemblyError(f"truncated: needed {size} bytes at 0x{o:X}")
    return struct.unpack_from(fmt, b, o)[0]


def _label_name(index: int) -> str:
    return f"label_{index}"


def _format_reg(num: int) -> str:
    return f"r{num}"


def _format_cstring_utf16(text: str) -> str:
    """Escape a Python str as a double-quoted assembly string literal.

    Mirrors the escaping consumed by :func:`_parse_cstring`. We keep it
    minimal and reversible: backslash, double-quote, and the C escapes
    for the control characters that appear in quest text (newline, tab,
    carriage return). Everything else (including non-ASCII UTF-16) is
    emitted verbatim so the file stays human-readable; the assembler
    re-encodes it to UTF-16LE.
    """
    out = ['"']
    for ch in text:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _decode_cstring_utf16(code: bytes, off: int) -> Tuple[str, int]:
    """Decode a NUL-terminated UTF-16LE string at ``off``.

    Returns (python_str, bytes_consumed_including_terminator).
    """
    end = off
    n = len(code)
    while True:
        if end + 2 > n:
            raise AssemblyError(f"unterminated CSTRING at 0x{off:X}")
        if code[end] == 0 and code[end + 1] == 0:
            break
        end += 2
    raw = code[off:end]
    text = raw.decode("utf-16-le", errors="surrogatepass")
    return text, (end + 2 - off)


def _decode_operands(
    op: Opcode, code: bytes, off: int, version: str
) -> Tuple[List[str], int, List[int]]:
    """Decode the inline operands of ``op`` starting at ``off``.

    Returns (operand_strings, total_operand_byte_length, referenced_label_indices).

    For an F_ARGS opcode on a version with the argument stack, the opcode
    has no inline operands and this returns ([], 0, []).
    """
    operands: List[str] = []
    refs: List[int] = []
    p = off

    if op.uses_arg_stack and version_has_args(version):
        return operands, 0, refs

    for param in op.params:
        t = param.type
        if t in (ParamType.R_REG, ParamType.W_REG):
            operands.append(_format_reg(_u8(code, p)))
            p += 1
        elif t in (ParamType.R_REG32, ParamType.W_REG32):
            operands.append(_format_reg(_read("<I", 4, code, p)))
            p += 4
        elif t in (
            ParamType.R_REG_SET_FIXED,
            ParamType.W_REG_SET_FIXED,
        ):
            first = _u8(code, p)
            p += 1
            last = first + param.count - 1
            # If the implicit span would wrap past r255, emit the single
            # start register (the count is fixed by the opcode anyway) so
            # the reassembler doesn't see a backwards/invalid range.
            operands.append(f"r{first}" if last > 0xFF else f"r{first}-r{last}")
        elif t in (
            ParamType.R_REG32_SET_FIXED,
            ParamType.W_REG32_SET_FIXED,
        ):
            first = _read("<I", 4, code, p)
            p += 4
            last = first + param.count - 1
            operands.append(f"r{first}-r{last}")
        elif t == ParamType.R_REG_SET:
            count = _u8(code, p)
            p += 1
            regs = []
            for _ in range(count):
                regs.append(_format_reg(_u8(code, p)))
                p += 1
            operands.append("[" + ", ".join(regs) + "]")
        elif t in (ParamType.SCRIPT16, ParamType.DATA16):
            idx = _read("<H", 2, code, p)
            p += 2
            refs.append(idx)
            operands.append(_label_name(idx))
        elif t == ParamType.SCRIPT32:
            idx = _read("<I", 4, code, p)
            p += 4
            refs.append(idx)
            operands.append(_label_name(idx))
        elif t == ParamType.SCRIPT16_SET:
            count = _u8(code, p)
            p += 1
            labels = []
            for _ in range(count):
                idx = _read("<H", 2, code, p)
                p += 2
                refs.append(idx)
                labels.append(_label_name(idx))
            operands.append("[" + ", ".join(labels) + "]")
        elif t == ParamType.U8:
            operands.append(f"0x{_u8(code, p):02X}")
            p += 1
        elif t == ParamType.U16:
            operands.append(f"0x{_read('<H', 2, code, p):04X}")
            p += 2
        elif t in (ParamType.I32, ParamType.U32):
            operands.append(f"0x{_read('<I', 4, code, p):08X}")
            p += 4
        elif t == ParamType.FLOAT32:
            val = struct.unpack_from("<f", code, p)[0] if p + 4 <= len(code) else None
            if val is None:
                raise AssemblyError(f"truncated FLOAT32 at 0x{p:X}")
            operands.append(_format_float(val))
            p += 4
        elif t == ParamType.CSTRING:
            text, consumed = _decode_cstring_utf16(code, p)
            operands.append(_format_cstring_utf16(text))
            p += consumed
        elif t == ParamType.VECTOR4F_LIST:
            # A VECTOR4F_LIST is only ever a *label* payload (data region),
            # never an inline opcode operand. If we ever see one inline the
            # table is wrong; surface it loudly rather than silently drift.
            raise AssemblyError(
                f"{op.mnemonic}: VECTOR4F_LIST is not an inline operand type"
            )
        else:
            raise AssemblyError(f"{op.mnemonic}: unhandled param type {t.value}")

    return operands, p - off, refs


def _format_float(val: float) -> str:
    """Format a float so re-parsing yields the identical 32-bit pattern.

    ``repr`` round-trips a Python float exactly; but we stored a *32-bit*
    float, so we must round-trip through float32 again on parse. We tag
    the literal with a trailing ``f`` and emit ``repr`` of the float32
    value, which is the shortest decimal that re-reads to the same
    float64 (and hence the same float32).
    """
    return f"{val!r}f"


def version_has_args(version: str) -> bool:
    """True if ``version`` uses the v3+ argument stack (BB does)."""
    # The argument stack exists on v3 and later. In our version set that is
    # everything except the v1/v2 (DC/PC) generations. BB_V4 has it.
    return version in {
        "GC_NTE", "GC_V3", "GC_EP3TE", "GC_EP3", "XB_V3", "BB_V4",
    }


def _region_decodes_as_code(code: bytes, start: int, end: int, version: str) -> bool:
    """True if ``[start, end)`` decodes as a clean instruction sequence.

    "Clean" means: every byte is consumed by a known opcode and the last
    instruction ends *exactly* at ``end`` (no overrun, no leftover). A
    label region that fails this is a data payload (string / struct /
    vector list), emitted as a ``.data`` blob instead.
    """
    p = start
    while p < end:
        b0 = code[p]
        p += 1
        if b0 in (q.PREFIX_F8, q.PREFIX_F9):
            if p >= end:
                return False
            full = q.decode_prefix(b0, code[p])
            p += 1
        else:
            full = b0
        op = q.OPCODES.get(full)
        if op is None or op.mnemonic is None:
            return False
        try:
            _operands, opnd_len, _refs = _decode_operands(op, code, p, version)
        except AssemblyError:
            return False
        # An operand must not read past the region boundary, or this isn't
        # a self-contained code region.
        if p + opnd_len > end:
            return False
        p += opnd_len
    return p == end


def _format_data_blob(data: bytes) -> str:
    return data.hex().upper()


def disassemble_code(
    code: bytes,
    label_offsets: Sequence[int],
    *,
    version: str = DEFAULT_VERSION,
) -> Disassembly:
    """Disassemble raw code bytes to literal assembly.

    ``code`` is the raw bytes of the code section (``QuestBin.code``);
    ``label_offsets`` is the ``.bin`` label table (index -> code offset,
    with 0xFFFFFFFF or out-of-range entries meaning "no label here").

    The code section is partitioned by label offsets into regions; each
    region is classified as *code* (decodes cleanly as instructions
    filling the region) or *data* (anything else — strings, structs,
    vector lists), and data regions are emitted verbatim as ``.data``
    hex blobs. Instructions are decoded linearly across the code regions;
    a ``label_NN:`` line is placed at every offset some label index points
    to. Control-flow targets are symbolic; ``arg_pushX`` and F_ARGS
    opcodes are emitted literally (see the module docstring).
    """
    code = bytes(code)
    n = len(code)

    # Map code offset -> sorted list of label indices defined there. Only
    # offsets that land inside the code section are code labels.
    offset_to_labels: Dict[int, List[int]] = {}
    for idx, raw_off in enumerate(label_offsets):
        if 0 <= raw_off < n:
            offset_to_labels.setdefault(raw_off, []).append(idx)

    # Region boundaries: every distinct code-label offset, plus 0 and n.
    boundaries = sorted({0, n} | set(offset_to_labels))
    # Classify each [boundaries[i], boundaries[i+1]) region.
    data_spans: List[Tuple[int, int]] = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e > s and not _region_decodes_as_code(code, s, e, version):
            data_spans.append((s, e))

    def data_end_at(p: int) -> Optional[int]:
        for s, e in data_spans:
            if s <= p < e:
                return e
        return None

    # Linearly decode the code section, skipping known data spans.
    instructions: List[Instruction] = []
    data_regions: Dict[int, bytes] = {}  # start offset -> raw bytes
    pos = 0
    while pos < n:
        d_end = data_end_at(pos)
        if d_end is not None:
            data_regions[pos] = code[pos:d_end]
            pos = d_end
            continue
        start = pos
        b0 = code[pos]
        pos += 1
        if b0 in (q.PREFIX_F8, q.PREFIX_F9):
            if pos >= n:
                raise AssemblyError(f"truncated extended opcode prefix at 0x{start:X}")
            full = q.decode_prefix(b0, code[pos])
            pos += 1
        else:
            full = b0

        op = q.OPCODES.get(full)
        if op is None or op.mnemonic is None:
            raise AssemblyError(f"unknown opcode 0x{full:X} at code offset 0x{start:X}")

        operands, opnd_len, _refs = _decode_operands(op, code, pos, version)
        pos += opnd_len
        instructions.append(
            Instruction(
                offset=start,
                size=pos - start,
                opcode=full,
                mnemonic=op.mnemonic,
                operands=operands,
            )
        )

    # Validate every code label lands on an instruction or data-region
    # boundary (so the reassembler can place the label and reproduce the
    # index).
    boundary_offsets = {ins.offset for ins in instructions}
    boundary_offsets |= set(data_regions)
    boundary_offsets.add(n)
    for off in offset_to_labels:
        if off not in boundary_offsets:
            raise AssemblyError(
                f"label at 0x{off:X} is not on an instruction/data boundary"
            )

    # Emit text. Pin the original label count so the reassembler rebuilds
    # a table of the exact same length (trailing/unused 0xFFFFFFFF entries
    # included).
    lines: List[str] = [f".version {version}", f".label_count {len(label_offsets)}", ""]

    def emit_labels_at(off: int) -> None:
        for idx in sorted(offset_to_labels.get(off, [])):
            lines.append(f"{_label_name(idx)}:")

    # Walk all boundary offsets in order, emitting labels then the region.
    emit_points = sorted(
        {ins.offset for ins in instructions} | set(data_regions) | {n}
    )
    instr_by_off = {ins.offset: ins for ins in instructions}
    for off in emit_points:
        emit_labels_at(off)
        if off in data_regions:
            lines.append("    .data " + _format_data_blob(data_regions[off]))
        elif off in instr_by_off:
            lines.append("    " + instr_by_off[off].text())

    return Disassembly(
        version=version,
        lines=lines,
        instructions=instructions,
        label_offsets=list(label_offsets),
    )


def disassemble_bin(decompressed_bytes: bytes, *, version: str = DEFAULT_VERSION) -> str:
    """Disassemble a decompressed ``.bin`` to literal assembly text.

    Convenience wrapper: parses the container, disassembles the code
    section against its label table, and returns the assembly text. The
    header metadata is *not* in the text — it is carried separately
    through :func:`roundtrip_bin`. (Layer-0 parity is on the code+labels;
    the header round-trips verbatim via :mod:`formats.quest_bin`.)
    """
    qb = quest_bin.parse_bin(decompressed_bytes)
    return disassemble_code(qb.code, qb.label_offsets, version=version).text()


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------
@dataclass
class AssembledCode:
    """Result of assembling assembly text into code bytes + label table."""

    version: str
    code: bytes
    label_offsets: List[int]


# Token regexes.
_LABEL_DEF_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$")
_REG_RE = re.compile(r"^r(\d+)$")
_REG_RANGE_RE = re.compile(r"^r(\d+)-r(\d+)$")


def _strip_comment(line: str) -> str:
    """Remove a trailing ``//`` or ``;`` comment that is outside a string."""
    out = []
    in_str = False
    esc = False
    i = 0
    while i < len(line):
        ch = line[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == ";":
            break
        if ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
            break
        out.append(ch)
        i += 1
    return "".join(out)


def _split_top_level(text: str, sep: str = ",") -> List[str]:
    """Split on ``sep`` at bracket/paren depth 0 and outside strings."""
    parts: List[str] = []
    depth = 0
    in_str = False
    esc = False
    cur = []
    for ch in text:
        if in_str:
            cur.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            cur.append(ch)
        elif ch in "[(":
            depth += 1
            cur.append(ch)
        elif ch in "])":
            depth -= 1
            cur.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _parse_int(tok: str, consts: Dict[str, int]) -> int:
    tok = tok.strip()
    if tok in consts:
        return consts[tok]
    try:
        return int(tok, 0)
    except ValueError as exc:
        raise AssemblyError(f"invalid integer literal: {tok!r}") from exc


def _parse_reg(tok: str) -> int:
    m = _REG_RE.match(tok.strip())
    if not m:
        raise AssemblyError(f"invalid register operand: {tok!r}")
    num = int(m.group(1))
    if not 0 <= num <= 0xFF:
        raise AssemblyError(f"register out of range: {tok!r}")
    return num


def _parse_cstring(tok: str) -> str:
    """Parse a double-quoted assembly string literal back to a Python str."""
    tok = tok.strip()
    if len(tok) < 2 or tok[0] != '"' or tok[-1] != '"':
        raise AssemblyError(f"invalid string literal: {tok!r}")
    body = tok[1:-1]
    out = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "\\":
            i += 1
            if i >= len(body):
                raise AssemblyError("dangling backslash in string literal")
            esc = body[i]
            out.append(
                {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"'}.get(esc, esc)
            )
        else:
            out.append(ch)
        i += 1
    return "".join(out)


@dataclass
class _Emitter:
    """Growable code buffer with deferred label backpatching."""

    buf: bytearray = field(default_factory=bytearray)
    # (buffer_offset, label_name, width_in_bytes) to backpatch in pass 2.
    fixups: List[Tuple[int, str, int]] = field(default_factory=list)

    def u8(self, v: int) -> None:
        self.buf.append(v & 0xFF)

    def u16(self, v: int) -> None:
        self.buf += struct.pack("<H", v & 0xFFFF)

    def u32(self, v: int) -> None:
        self.buf += struct.pack("<I", v & 0xFFFFFFFF)

    def f32(self, v: float) -> None:
        self.buf += struct.pack("<f", v)

    def opcode(self, code: int) -> None:
        prefix, low = q.encode_prefix(code)
        if prefix is not None:
            self.buf.append(prefix)
        self.buf.append(low)

    def label_ref(self, name: str, width: int) -> None:
        self.fixups.append((len(self.buf), name, width))
        self.buf += b"\x00" * width

    @property
    def size(self) -> int:
        return len(self.buf)


def _emit_operand(
    em: _Emitter, param_type: ParamType, tok: str, consts: Dict[str, int], count: int
) -> None:
    """Emit one inline operand of the given param type from text ``tok``."""
    tok = tok.strip()
    if param_type in (ParamType.R_REG, ParamType.W_REG):
        em.u8(_parse_reg(tok))
    elif param_type in (ParamType.R_REG32, ParamType.W_REG32):
        em.u32(_parse_reg32(tok))
    elif param_type in (
        ParamType.R_REG_SET_FIXED, ParamType.W_REG_SET_FIXED,
    ):
        first = _parse_reg_set_fixed_first(tok, count)
        em.u8(first)
    elif param_type in (
        ParamType.R_REG32_SET_FIXED, ParamType.W_REG32_SET_FIXED,
    ):
        first = _parse_reg_set_fixed_first(tok, count, allow32=True)
        em.u32(first)
    elif param_type == ParamType.R_REG_SET:
        regs = _parse_set(tok)
        em.u8(len(regs))
        for r in regs:
            em.u8(_parse_reg(r))
    elif param_type in (ParamType.SCRIPT16, ParamType.DATA16):
        em.label_ref(tok, 2)
    elif param_type == ParamType.SCRIPT32:
        em.label_ref(tok, 4)
    elif param_type == ParamType.SCRIPT16_SET:
        labels = _parse_set(tok)
        em.u8(len(labels))
        for label in labels:
            em.label_ref(label.strip(), 2)
    elif param_type == ParamType.U8:
        em.u8(_parse_int(tok, consts))
    elif param_type == ParamType.U16:
        em.u16(_parse_int(tok, consts))
    elif param_type in (ParamType.I32, ParamType.U32):
        em.u32(_parse_int(tok, consts))
    elif param_type == ParamType.FLOAT32:
        em.f32(_parse_float(tok))
    elif param_type == ParamType.CSTRING:
        text = _parse_cstring(tok)
        em.buf += text.encode("utf-16-le", errors="surrogatepass")
        em.u16(0)
    else:
        raise AssemblyError(f"unhandled param type for emit: {param_type.value}")


def _parse_float(tok: str) -> float:
    tok = tok.strip()
    if tok.endswith(("f", "F")):
        tok = tok[:-1]
    try:
        return float(tok)
    except ValueError as exc:
        raise AssemblyError(f"invalid float literal: {tok!r}") from exc


def _parse_set(tok: str) -> List[str]:
    tok = tok.strip()
    if not (tok.startswith("[") and tok.endswith("]")):
        raise AssemblyError(f"set operand must be bracketed: {tok!r}")
    inner = tok[1:-1].strip()
    if not inner:
        return []
    return [p.strip() for p in _split_top_level(inner)]


def _parse_reg32(tok: str) -> int:
    """Parse a 32-bit register operand, preserving all 32 bits.

    Real quest data sometimes stores a ``*32`` register operand whose high
    24 bits are non-zero (the engine ignores them, but the bytes are real
    and must round-trip). We therefore accept the full 32-bit range here,
    unlike :func:`_parse_reg` which is the 1-byte register form.
    """
    m = _REG_RE.match(tok.strip())
    if not m:
        raise AssemblyError(f"invalid register operand: {tok!r}")
    num = int(m.group(1))
    if not 0 <= num <= 0xFFFFFFFF:
        raise AssemblyError(f"register32 out of range: {tok!r}")
    return num


def _parse_reg_set_fixed_first(tok: str, count: int, *, allow32: bool = False) -> int:
    """Parse an ``rA-rB`` (or single ``rA``) fixed set, return first reg.

    When ``allow32`` the first register keeps its full 32-bit value (the
    ``*32_SET_FIXED`` form); otherwise it is a 1-byte register.
    """
    tok = tok.strip()
    m = _REG_RANGE_RE.match(tok)
    if m:
        first, last = int(m.group(1)), int(m.group(2))
        if (last - first + 1) != count:
            raise AssemblyError(
                f"fixed register set {tok!r} spans {last - first + 1}, expected {count}"
            )
        return first & (0xFFFFFFFF if allow32 else 0xFF)
    return _parse_reg32(tok) if allow32 else _parse_reg(tok)


# Sugar handling --------------------------------------------------------------
_CONDITION_OPS = {
    # text comparator -> (reg/reg opcode mnemonic, reg/imm opcode mnemonic)
    # used by structured if/while. The jump is taken when the condition is
    # TRUE; for if/while we invert (jump *past* the body when FALSE).
    "==": ("jmp_eq", "jmpi_eq"),
    "!=": ("jmp_ne", "jmpi_ne"),
    ">": ("jmp_gt", "jmpi_gt"),
    "<": ("jmp_lt", "jmpi_lt"),
    ">=": ("jmp_ge", "jmpi_ge"),
    "<=": ("jmp_le", "jmpi_le"),
}
_INVERT = {"==": "!=", "!=": "==", ">": "<=", "<": ">=", ">=": "<", "<=": ">"}


def _preprocess(text: str) -> List[Tuple[int, str]]:
    """Expand ``.const`` / ``%macro`` / ``if`` / ``while`` sugar.

    Returns a list of (original_line_number, expanded_line) pairs in
    literal-instruction form (label defs, directives, and bare opcode
    lines). ``.const`` defines are stripped (folded into the constant
    table later); macros are inlined; if/while expand to label+jump.
    """
    raw_lines = text.split("\n")

    # Pass A: collect macros and strip their bodies.
    macros: Dict[str, Tuple[List[str], List[str]]] = {}
    stripped: List[Tuple[int, str]] = []
    macro_name: Optional[str] = None
    macro_params: List[str] = []
    macro_body: List[str] = []
    for lineno, raw in enumerate(raw_lines, 1):
        line = _strip_comment(raw).strip()
        if not line:
            if macro_name is None:
                stripped.append((lineno, ""))
            continue
        if line.startswith("%macro"):
            if macro_name is not None:
                raise AssemblyError("nested %macro is not allowed", line=lineno)
            head = line.split(None, 1)
            if len(head) < 2 or not head[1].split():
                raise AssemblyError("%macro requires a name", line=lineno)
            name_and_params = head[1].split(None, 1)
            macro_name = name_and_params[0]
            macro_params = []
            if len(name_and_params) > 1:
                for raw_param in re.split(r"[,\s]+", name_and_params[1].strip()):
                    if not raw_param:
                        continue
                    macro_params.append(raw_param.lstrip("%"))
            macro_body = []
            continue
        if line.startswith("%endmacro"):
            if macro_name is None:
                raise AssemblyError("%endmacro without %macro", line=lineno)
            macros[macro_name] = (macro_params, macro_body)
            macro_name = None
            continue
        if macro_name is not None:
            macro_body.append(line)
            continue
        stripped.append((lineno, line))

    if macro_name is not None:
        raise AssemblyError("unterminated %macro")

    # Pass B: expand macro invocations + structured control flow.
    out: List[Tuple[int, str]] = []
    auto_label = [0]
    block_stack: List[Tuple[str, str, Optional[str]]] = []  # (kind, end_label, top_label)

    def new_label() -> str:
        auto_label[0] += 1
        return f"_auto_{auto_label[0]}"

    def expand_line(lineno: int, line: str, depth: int = 0) -> None:
        if depth > 64:
            raise AssemblyError("macro expansion too deep (recursion?)", line=lineno)
        head = line.split(None, 1)
        mnem = head[0]
        rest = head[1].strip() if len(head) > 1 else ""

        if mnem == "if":
            cond = _parse_condition(rest, lineno)
            end_label = new_label()
            block_stack.append(("if", end_label, None))
            _emit_cond_jump(out, lineno, cond, end_label, invert=True)
            return
        if mnem == "while":
            cond = _parse_condition(rest, lineno)
            top_label = new_label()
            end_label = new_label()
            block_stack.append(("while", end_label, top_label))
            out.append((lineno, f"{top_label}:"))
            _emit_cond_jump(out, lineno, cond, end_label, invert=True)
            return
        if mnem in ("endif", "endwhile", "%endif", "%endwhile"):
            if not block_stack:
                raise AssemblyError(f"{mnem} without matching block", line=lineno)
            kind, end_label, top_label = block_stack.pop()
            if kind == "while" and top_label is not None:
                out.append((lineno, f"    jmp {top_label}"))
            out.append((lineno, f"{end_label}:"))
            return
        if mnem in macros:
            params, body = macros[mnem]
            args = _split_top_level(rest) if rest else []
            args = [a.strip() for a in args if a.strip() != ""]
            if len(args) != len(params):
                raise AssemblyError(
                    f"macro {mnem} expects {len(params)} args, got {len(args)}",
                    line=lineno,
                )
            subst = {"%" + p: a for p, a in zip(params, args)}
            for bline in body:
                expanded = _apply_subst(bline, subst)
                expand_line(lineno, expanded, depth + 1)
            return
        out.append((lineno, line if line.endswith(":") else "    " + line))

    for lineno, line in stripped:
        if not line:
            out.append((lineno, ""))
            continue
        if line.startswith("."):
            out.append((lineno, line))  # directive — handled later, verbatim
            continue
        if line.endswith(":"):
            out.append((lineno, line))
            continue
        expand_line(lineno, line)

    if block_stack:
        raise AssemblyError("unterminated if/while block")
    return out


def _apply_subst(line: str, subst: Dict[str, str]) -> str:
    """Replace ``%param`` placeholders in a macro body line."""
    def repl(m: re.Match) -> str:
        name = m.group(1)
        return subst.get("%" + name, m.group(0))

    return re.sub(r"%([A-Za-z_][A-Za-z0-9_]*)", repl, line)


def _parse_condition(rest: str, lineno: int) -> Tuple[str, str, str]:
    """Parse ``A OP B`` -> (lhs, op, rhs)."""
    for op in (">=", "<=", "==", "!=", ">", "<"):
        idx = rest.find(op)
        if idx >= 0:
            lhs = rest[:idx].strip()
            rhs = rest[idx + len(op):].strip()
            if not lhs or not rhs:
                raise AssemblyError(f"malformed condition: {rest!r}", line=lineno)
            return lhs, op, rhs
    raise AssemblyError(f"no comparison operator in condition: {rest!r}", line=lineno)


def _emit_cond_jump(
    out: List[Tuple[int, str]],
    lineno: int,
    cond: Tuple[str, str, str],
    target: str,
    *,
    invert: bool,
) -> None:
    lhs, op, rhs = cond
    if invert:
        op = _INVERT[op]
    reg_op, imm_op = _CONDITION_OPS[op]
    # rhs is a register (rN) -> reg/reg form; else immediate form.
    if _REG_RE.match(rhs):
        out.append((lineno, f"    {reg_op} {lhs}, {rhs}, {target}"))
    else:
        out.append((lineno, f"    {imm_op} {lhs}, {rhs}, {target}"))


# F_ARGS structured-mode arg_push synthesis ----------------------------------
def _emit_args_pushes(
    em: _Emitter, op: Opcode, arg_toks: List[str], consts: Dict[str, int]
) -> None:
    """Emit the arg_pushX sequence for an F_ARGS opcode (structured mode).

    Mirrors newserv's assembler heuristic so structured authoring produces
    the canonical bytes. (The literal disassembly path never uses this —
    it emits explicit arg_pushX lines — so this only affects hand-authored
    structured input, never the parity round-trip.)
    """
    if len(arg_toks) != len(op.params):
        raise AssemblyError(
            f"{op.mnemonic}: expected {len(op.params)} args, got {len(arg_toks)}"
        )
    for param, tok in zip(op.params, arg_toks):
        tok = tok.strip()
        t = param.type
        if t == ParamType.CSTRING:
            em.u8(ARG_PUSHS)
            text = _parse_cstring(tok)
            em.buf += text.encode("utf-16-le", errors="surrogatepass")
            em.u16(0)
        elif t in (ParamType.SCRIPT16, ParamType.SCRIPT32, ParamType.DATA16):
            # A label argument is pushed as a 16-bit literal (arg_pushw).
            em.u8(ARG_PUSHW)
            em.label_ref(tok, 2)
        elif t in (
            ParamType.R_REG, ParamType.W_REG, ParamType.R_REG32, ParamType.W_REG32,
        ):
            # Out-param register: push the register *number* via arg_pushb.
            em.u8(ARG_PUSHB)
            em.u8(_parse_reg(tok))
        elif t in (
            ParamType.R_REG_SET_FIXED, ParamType.W_REG_SET_FIXED,
            ParamType.R_REG32_SET_FIXED, ParamType.W_REG32_SET_FIXED,
        ):
            em.u8(ARG_PUSHB)
            em.u8(_parse_reg_set_fixed_first(tok, param.count))
        else:
            # Integer-valued: choose width by magnitude (newserv rule).
            val = _parse_int(tok, consts)
            uval = val & 0xFFFFFFFF
            if uval > 0xFFFF:
                em.u8(ARG_PUSHL)
                em.u32(uval)
            elif uval > 0xFF:
                em.u8(ARG_PUSHW)
                em.u16(uval)
            else:
                em.u8(ARG_PUSHB)
                em.u8(uval)
    em.opcode(op.code)


# ---------------------------------------------------------------------------
# Two-pass assembler
# ---------------------------------------------------------------------------
def assemble(text: str, *, version: str = DEFAULT_VERSION) -> AssembledCode:
    """Assemble assembly text to code bytes + a label table.

    Two passes: collect label definitions (and ``.const`` / ``.label_count``
    directives, macro/structured expansion), emit instructions recording
    deferred label references, then backpatch the 16/32-bit label indices.

    Supports both the literal form (explicit ``arg_pushX`` + bare F_ARGS
    opcode) and the structured form (F_ARGS opcode with inline operands,
    which synthesises the arg_pushX sequence). Returns an
    :class:`AssembledCode`; its ``label_offsets`` is the rebuilt table.
    """
    consts: Dict[str, int] = {}
    declared_label_count: Optional[int] = None

    # Preprocess directives that must be known before line emission:
    # .version override and .const are gathered here; .label_count too.
    expanded = _preprocess(text)

    # Gather .const definitions and .version / .label_count up front.
    for lineno, line in expanded:
        if not line:
            continue
        body = line.strip()
        if body.startswith(".const"):
            toks = body.split()
            if len(toks) != 3:
                raise AssemblyError(".const requires NAME VALUE", line=lineno)
            consts[toks[1]] = _parse_int(toks[2], consts)
        elif body.startswith(".version"):
            parts = body.split(None, 1)
            if len(parts) == 2:
                version = parts[1].strip()
        elif body.startswith(".label_count"):
            parts = body.split(None, 1)
            if len(parts) == 2:
                declared_label_count = _parse_int(parts[1].strip(), consts)

    em = _Emitter()
    # Pass 1: emit code, recording each label's offset and label refs.
    label_def_offsets: Dict[str, int] = {}
    # Preserve explicit label index when the name encodes it (label_NN).
    label_explicit_index: Dict[str, int] = {}

    for lineno, line in expanded:
        if not line:
            continue
        body = line.strip()
        if not body:
            continue
        if body.startswith("."):
            directive = body.split(None, 1)[0]
            if directive in (".version", ".const", ".label_count"):
                # Handled in the pre-pass above; nothing to emit.
                continue
            if directive == ".data":
                parts = body.split(None, 1)
                hexstr = parts[1].strip() if len(parts) > 1 else ""
                hexstr = "".join(hexstr.split())
                try:
                    em.buf += bytes.fromhex(hexstr)
                except ValueError as exc:
                    raise AssemblyError(f"invalid .data hex: {exc}", line=lineno) from exc
                continue
            raise AssemblyError(f"unknown directive: {directive}", line=lineno)

        m = _LABEL_DEF_RE.match(body)
        if m:
            name = m.group(1)
            if name in label_def_offsets:
                raise AssemblyError(f"duplicate label: {name}", line=lineno)
            label_def_offsets[name] = em.size
            mi = re.match(r"^label_(\d+)$", name)
            if mi:
                label_explicit_index[name] = int(mi.group(1))
            continue

        # An instruction line.
        toks = body.split(None, 1)
        mnem = toks[0]
        arg_text = toks[1].strip() if len(toks) > 1 else ""
        op = q.by_mnemonic(mnem)
        if op is None:
            raise AssemblyError(f"unknown opcode mnemonic: {mnem}", line=lineno)

        arg_toks = _split_top_level(arg_text) if arg_text else []
        arg_toks = [a for a in (t.strip() for t in arg_toks) if a != ""]

        use_structured_args = (
            op.uses_arg_stack
            and version_has_args(version)
            and len(arg_toks) > 0
        )
        if use_structured_args:
            try:
                _emit_args_pushes(em, op, arg_toks, consts)
            except AssemblyError as exc:
                raise AssemblyError(str(exc), line=lineno) from exc
            continue

        # Literal form: emit opcode then inline operands per the signature.
        em.opcode(op.code)
        if op.uses_arg_stack and version_has_args(version):
            # Bare F_ARGS opcode (operands already pushed). No inline bytes.
            if arg_toks:
                raise AssemblyError(
                    f"{mnem}: F_ARGS opcode takes no inline operands in literal form",
                    line=lineno,
                )
            continue
        if len(arg_toks) != len(op.params):
            raise AssemblyError(
                f"{mnem}: expected {len(op.params)} operands, got {len(arg_toks)}",
                line=lineno,
            )
        for param, tok in zip(op.params, arg_toks):
            try:
                _emit_operand(em, param.type, tok, consts, param.count)
            except AssemblyError as exc:
                raise AssemblyError(str(exc), line=lineno) from exc

    code_len = em.size

    # Build the label table (index -> offset). Honour explicit label_NN
    # indices; rebuild a table of the declared length with 0xFFFFFFFF for
    # unused entries, so round-trip reproduces trailing/unused slots.
    index_to_offset: Dict[int, int] = {}
    next_auto = 0
    used_indices = set(label_explicit_index.values())
    for name, off in label_def_offsets.items():
        if name in label_explicit_index:
            index_to_offset[label_explicit_index[name]] = off
    for name, off in label_def_offsets.items():
        if name in label_explicit_index:
            continue
        while next_auto in used_indices:
            next_auto += 1
        index_to_offset[next_auto] = off
        used_indices.add(next_auto)
        next_auto += 1

    table_len = declared_label_count
    if table_len is None:
        table_len = (max(index_to_offset) + 1) if index_to_offset else 0
    label_offsets: List[int] = []
    for i in range(table_len):
        label_offsets.append(index_to_offset.get(i, 0xFFFFFFFF))

    # Pass 2: backpatch label references. Reuse the index assignment from
    # the table build so a name and its table index always agree.
    name_to_index: Dict[str, int] = {}
    for name in label_def_offsets:
        if name in label_explicit_index:
            name_to_index[name] = label_explicit_index[name]
    auto = 0
    used2 = set(label_explicit_index.values())
    for name in label_def_offsets:
        if name in label_explicit_index:
            continue
        while auto in used2:
            auto += 1
        name_to_index[name] = auto
        used2.add(auto)
        auto += 1

    buf = bytearray(em.buf)
    for off, name, width in em.fixups:
        if name in name_to_index:
            idx = name_to_index[name]
        else:
            # An undefined reference of the form ``label_NN`` carries its
            # own numeric index — it points outside the code section (an
            # out-of-range or data-only label that has no code definition).
            # This is byte-faithful: the original opcode stored exactly NN.
            mi = re.match(r"^label_(\d+)$", name)
            if mi is None:
                raise AssemblyError(f"reference to undefined label: {name}")
            idx = int(mi.group(1))
        if width == 2:
            struct.pack_into("<H", buf, off, idx & 0xFFFF)
        elif width == 4:
            struct.pack_into("<I", buf, off, idx & 0xFFFFFFFF)
        else:
            raise AssemblyError(f"invalid label width {width}")

    assert len(buf) == code_len
    return AssembledCode(version=version, code=bytes(buf), label_offsets=label_offsets)


def assemble_to_bin(
    text: str,
    template: quest_bin.QuestBin,
    *,
    version: str = DEFAULT_VERSION,
) -> bytes:
    """Assemble ``text`` and splice it into ``template``'s container.

    The header is taken verbatim from ``template`` (Layer-0 keeps the
    header byte-exact); only the code section and label table are
    replaced with the freshly assembled output. Returns decompressed
    ``.bin`` bytes.
    """
    asm = assemble(text, version=version)
    rebuilt = quest_bin.QuestBin(
        fmt=template.fmt,
        code_offset=template.code_offset,
        label_table_offset=template.code_offset + len(asm.code),
        size=0,  # recomputed by serialize_bin
        unknown_marker=template.unknown_marker,
        header_raw=template.header_raw,
        code=asm.code,
        label_offsets=asm.label_offsets,
    )
    return quest_bin.serialize_bin(rebuilt)


def roundtrip_bin(decompressed_bytes: bytes, *, version: str = DEFAULT_VERSION) -> bytes:
    """Disassemble then reassemble a decompressed ``.bin``; rebuild the file.

    The parity oracle: for a well-formed BB quest this returns bytes
    *identical* to the input. Equivalent to::

        text = disassemble_bin(decompressed_bytes)
        return assemble_to_bin(text, parse_bin(decompressed_bytes))

    but shares the single ``parse_bin`` so the header is spliced back
    verbatim.
    """
    qb = quest_bin.parse_bin(decompressed_bytes)
    dis = disassemble_code(qb.code, qb.label_offsets, version=version)
    asm = assemble(dis.text(), version=version)
    if asm.code != qb.code:
        raise AssemblyError(
            "roundtrip_bin: reassembled code differs from original "
            f"(orig {len(qb.code)} bytes, got {len(asm.code)} bytes)"
        )
    if asm.label_offsets != qb.label_offsets:
        raise AssemblyError(
            "roundtrip_bin: reassembled label table differs from original"
        )
    rebuilt = quest_bin.QuestBin(
        fmt=qb.fmt,
        code_offset=qb.code_offset,
        label_table_offset=qb.label_table_offset,
        size=qb.size,
        unknown_marker=qb.unknown_marker,
        header_raw=qb.header_raw,
        code=asm.code,
        label_offsets=asm.label_offsets,
    )
    return quest_bin.serialize_bin(rebuilt)
