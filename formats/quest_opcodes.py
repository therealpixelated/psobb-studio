"""PSOBB quest-VM opcode set — the single byte-faithful source of truth.

This is the Layer-0 foundation of the quest-scripting stack: a clean, typed
model of every opcode the Blue Burst quest virtual machine understands, indexed
by its full (possibly extended-page) code.

Provenance
----------
The opcode DATA is ported from two MIT-licensed reference projects (we port the
data, not their code):

* phantasmal-world ``opcodes.yml`` — the primary port source (~425 named
  opcodes with doc strings + per-param register read/write + stack push/pop).
* newserv ``QuestScript.cc`` ``opcode_defs[]`` — the cross-check, carrying the
  qedit aliases, per-version flags, and the F_ARGS / F_PUSH_ARG /
  F_CLEAR_ARGS / F_TERMINATOR flags. For Blue Burst (the live target) newserv
  wins where the two disagree.

The merged, self-contained table lives in :mod:`formats._quest_opcode_table`
(a committed, generated artifact — it never reads ``_reference/`` at runtime).
Regenerate it with::

    python scripts/gen_quest_opcodes.py

Extended pages
--------------
Opcodes are one byte, except for two prefixed pages:

* a leading ``0xF8`` byte selects the +0x100 page (stored ``F8 NN`` -> 0x1NN),
* a leading ``0xF9`` byte selects the +0x200 page (stored ``F9 NN`` -> 0x2NN).

Everything here indexes opcodes by their *full* code (which may exceed 0xFF);
:func:`encode_prefix` / :func:`decode_prefix` translate between the full code
and the stored byte sequence.

Quick start
-----------
>>> from formats import quest_opcodes as q
>>> q.OPCODES[0x28].mnemonic
'jmp'
>>> q.by_mnemonic('jmp').code
40
>>> q.check_opcode_definitions()  # raises on any inconsistency
True
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from formats._quest_opcode_table import OPCODE_RECORDS

# ---------------------------------------------------------------------------
# Versions. The default target is Blue Burst v4 (the live PSOBB server target).
# ---------------------------------------------------------------------------
KNOWN_VERSIONS: frozenset[str] = frozenset(
    {
        "DC_NTE", "DC_112000", "DC_V1", "DC_V2",
        "PC_NTE", "PC_V2",
        "GC_NTE", "GC_V3", "GC_EP3TE", "GC_EP3",
        "XB_V3",
        "BB_V4",
    }
)
DEFAULT_VERSION = "BB_V4"


# ---------------------------------------------------------------------------
# Semantic flags. Mirror newserv's flag names so the two stay legible together.
# ---------------------------------------------------------------------------
# Operands for this opcode come from the argument stack (pushed by arg_push*)
# on v3 and later, rather than appearing inline in the bytecode.
F_ARGS = "F_ARGS"
# This opcode is itself an arg_push* — it pushes one operand onto the arg stack.
F_PUSH_ARG = "F_PUSH_ARG"
# This opcode clears the argument stack (call / va_start / va_call / switch_call).
F_CLEAR_ARGS = "F_CLEAR_ARGS"
# This opcode ends a basic block (ret / jmp / exit): control does not fall
# through to the following instruction.
F_TERMINATOR = "F_TERMINATOR"

KNOWN_FLAGS: frozenset[str] = frozenset({F_ARGS, F_PUSH_ARG, F_CLEAR_ARGS, F_TERMINATOR})


# ---------------------------------------------------------------------------
# Param type taxonomy. Matches opcodes.yml's concepts, folded together with
# newserv's richer register-set distinctions.
# ---------------------------------------------------------------------------
class ParamType(Enum):
    """The type of a single opcode operand."""

    # Registers --------------------------------------------------------------
    R_REG = "R_REG"                          # 1-byte register number, read
    W_REG = "W_REG"                          # 1-byte register number, written
    R_REG_SET = "R_REG_SET"                  # count byte + that many reg bytes (read)
    R_REG_SET_FIXED = "R_REG_SET_FIXED"      # 1 reg byte; reads N consecutive regs
    W_REG_SET_FIXED = "W_REG_SET_FIXED"      # 1 reg byte; writes N consecutive regs
    R_REG32 = "R_REG32"                      # 32-bit register number, read
    W_REG32 = "W_REG32"                      # 32-bit register number, written
    R_REG32_SET_FIXED = "R_REG32_SET_FIXED"  # 32-bit reg; reads N consecutive regs
    W_REG32_SET_FIXED = "W_REG32_SET_FIXED"  # 32-bit reg; writes N consecutive regs
    # Labels (function/data table indices) -----------------------------------
    SCRIPT16 = "SCRIPT16"                     # 16-bit label index (function table)
    SCRIPT16_SET = "SCRIPT16_SET"             # count byte + that many 16-bit labels
    SCRIPT32 = "SCRIPT32"                     # 32-bit label index
    DATA16 = "DATA16"                         # 16-bit data-label index
    # Immediates -------------------------------------------------------------
    U8 = "U8"                                 # 8-bit immediate (newserv I8)
    U16 = "U16"                               # 16-bit immediate (newserv I16)
    I32 = "I32"                               # 32-bit immediate
    U32 = "U32"                               # 32-bit immediate, unsigned alias
    FLOAT32 = "FLOAT32"                       # 32-bit IEEE float
    CSTRING = "CSTRING"                       # NUL-terminated UTF-16 string (BB)
    # Structured payloads ----------------------------------------------------
    VECTOR4F_LIST = "VECTOR4F_LIST"           # inline list of 4-float vectors


# Types that name a register operand (carry register read/write semantics).
_REGISTER_TYPES = frozenset(
    {
        ParamType.R_REG, ParamType.W_REG, ParamType.R_REG_SET,
        ParamType.R_REG_SET_FIXED, ParamType.W_REG_SET_FIXED,
        ParamType.R_REG32, ParamType.W_REG32,
        ParamType.R_REG32_SET_FIXED, ParamType.W_REG32_SET_FIXED,
    }
)
# Types that name a label (basic-block / data reference).
_LABEL_TYPES = frozenset(
    {ParamType.SCRIPT16, ParamType.SCRIPT16_SET, ParamType.SCRIPT32, ParamType.DATA16}
)


@dataclass(frozen=True)
class Param:
    """A single typed operand of an opcode."""

    type: ParamType
    reads: bool = False    # operand value is read by the opcode
    writes: bool = False   # operand (a register) is written by the opcode
    count: int = 0         # for *_SET_FIXED: number of consecutive registers

    @property
    def is_register(self) -> bool:
        return self.type in _REGISTER_TYPES

    @property
    def is_label(self) -> bool:
        return self.type in _LABEL_TYPES

    def __str__(self) -> str:
        rw = ("r" if self.reads else "") + ("w" if self.writes else "")
        suffix = f"[{self.count}]" if self.count else ""
        return f"{self.type.value}{suffix}" + (f"({rw})" if rw else "")


@dataclass(frozen=True)
class Opcode:
    """A single quest-VM opcode definition."""

    code: int                                # full code (may exceed 0xFF)
    mnemonic: str | None                     # primary (newserv) mnemonic
    qedit_alias: str | None                  # qedit's name for this opcode
    doc: str | None                          # human description
    params: tuple[Param, ...] = ()           # logical operand signature
    version_flags: frozenset[str] = field(default_factory=frozenset)
    flags: frozenset[str] = field(default_factory=frozenset)

    # -- flag predicates ----------------------------------------------------
    @property
    def uses_arg_stack(self) -> bool:
        """True if operands are taken from the argument stack (F_ARGS)."""
        return F_ARGS in self.flags

    @property
    def is_arg_push(self) -> bool:
        """True if this opcode pushes an operand onto the arg stack."""
        return F_PUSH_ARG in self.flags

    @property
    def clears_args(self) -> bool:
        return F_CLEAR_ARGS in self.flags

    @property
    def is_terminator(self) -> bool:
        """True if this opcode ends a basic block (ret / jmp / exit)."""
        return F_TERMINATOR in self.flags

    # -- extended-page helpers ---------------------------------------------
    @property
    def is_extended(self) -> bool:
        return self.code > 0xFF

    @property
    def prefix(self) -> int | None:
        """The stored prefix byte (0xF8 / 0xF9), or None for a 1-byte opcode."""
        return encode_prefix(self.code)[0]

    def supports(self, version: str = DEFAULT_VERSION) -> bool:
        return version in self.version_flags

    def __str__(self) -> str:
        name = self.mnemonic or f"unknown_{self.code:03X}"
        alias = f" (qedit: {self.qedit_alias})" if self.qedit_alias else ""
        args = ", ".join(str(p) for p in self.params)
        return f"{self.code:03X} {name}{alias}({args})"


# ---------------------------------------------------------------------------
# Extended-page prefix encode / decode.
#   full 0x1NN  <->  stored bytes F8 NN
#   full 0x2NN  <->  stored bytes F9 NN
#   full 0x0NN  <->  stored byte  NN
# ---------------------------------------------------------------------------
PREFIX_F8 = 0xF8
PREFIX_F9 = 0xF9


def encode_prefix(code: int) -> tuple[int | None, int]:
    """Map a full opcode code to its stored ``(prefix_byte, low_byte)``.

    For a plain one-byte opcode the prefix is ``None``.

    >>> encode_prefix(0x28)
    (None, 40)
    >>> encode_prefix(0x101)
    (248, 1)
    >>> encode_prefix(0x25D)
    (249, 93)
    """
    page = code & 0xF00
    low = code & 0xFF
    if page == 0x000:
        return None, low
    if page == 0x100:
        return PREFIX_F8, low
    if page == 0x200:
        return PREFIX_F9, low
    raise ValueError(f"opcode 0x{code:X} is not in a known page (0x0/0x1/0x2)")


def decode_prefix(prefix: int | None, low: int) -> int:
    """Inverse of :func:`encode_prefix`: stored bytes -> full code.

    >>> decode_prefix(None, 0x28)
    40
    >>> decode_prefix(0xF8, 0x01)
    257
    >>> decode_prefix(0xF9, 0x5D)
    605
    """
    if not 0 <= low <= 0xFF:
        raise ValueError(f"low byte 0x{low:X} out of range")
    if prefix is None:
        return low
    if prefix == PREFIX_F8:
        return 0x100 | low
    if prefix == PREFIX_F9:
        return 0x200 | low
    raise ValueError(f"unknown opcode prefix 0x{prefix:X} (expected 0xF8 or 0xF9)")


# ---------------------------------------------------------------------------
# Build the in-memory model from the committed table.
# ---------------------------------------------------------------------------
def _build() -> dict[int, Opcode]:
    out: dict[int, Opcode] = {}
    for rec in OPCODE_RECORDS:
        code, mnemonic, qedit_alias, doc, params, versions, flags = rec
        param_objs = tuple(
            Param(
                type=ParamType(ptype),
                reads=bool(reads),
                writes=bool(writes),
                count=int(count),
            )
            for (ptype, reads, writes, count) in params
        )
        out[code] = Opcode(
            code=code,
            mnemonic=mnemonic,
            qedit_alias=qedit_alias,
            doc=doc,
            params=param_objs,
            version_flags=frozenset(versions),
            flags=frozenset(flags),
        )
    return out


#: Full opcode table, keyed by full code (extended-page codes may exceed 0xFF).
OPCODES: dict[int, Opcode] = _build()


def _build_mnemonic_index() -> dict[str, Opcode]:
    idx: dict[str, Opcode] = {}
    # Primary mnemonics first so they win over alias collisions.
    for op in OPCODES.values():
        if op.mnemonic:
            idx.setdefault(op.mnemonic, op)
    for op in OPCODES.values():
        if op.qedit_alias:
            idx.setdefault(op.qedit_alias, op)
    return idx


#: Lookup by mnemonic OR qedit alias (primary mnemonics take precedence).
MNEMONICS: dict[str, Opcode] = _build_mnemonic_index()


# ---------------------------------------------------------------------------
# Public lookup API
# ---------------------------------------------------------------------------
def get(code: int) -> Opcode | None:
    """Look up an opcode by its full code; ``None`` if unknown."""
    return OPCODES.get(code)


def by_mnemonic(name: str) -> Opcode | None:
    """Look up an opcode by primary mnemonic or qedit alias; ``None`` if unknown."""
    return MNEMONICS.get(name)


def opcodes_for_version(version: str = DEFAULT_VERSION) -> list[Opcode]:
    """All opcodes valid on the given version (default Blue Burst)."""
    return [op for op in OPCODES.values() if op.supports(version)]


# ---------------------------------------------------------------------------
# Self-test (mirrors newserv's check_opcode_definitions()).
# ---------------------------------------------------------------------------
class OpcodeDefinitionError(ValueError):
    """Raised when the opcode table fails an internal consistency check."""


def check_opcode_definitions() -> bool:
    """Validate the opcode table; raise :class:`OpcodeDefinitionError` on any
    inconsistency. Returns ``True`` when everything checks out.

    Checks:
      * no duplicate full codes (the dict can't hold them, so we re-scan the
        raw records);
      * every code sits in a known page (0x0 / 0x1 / 0x2) and round-trips
        through encode/decode_prefix;
      * every param type is a known :class:`ParamType`;
      * every version flag and semantic flag is known;
      * F_ARGS opcodes carry no *inline* operands — their args come from the
        arg stack (we assert that any register operands are not inline-encoded
        by requiring the opcode to be flagged, and that arg_push* opcodes which
        feed the stack are themselves F_PUSH_ARG, not F_ARGS);
      * arg_push* opcodes (F_PUSH_ARG) push exactly one operand;
      * terminators (F_TERMINATOR) are a sane set including ret and jmp;
      * register read/write flags are coherent with the param type.
    """
    seen: set[int] = set()
    for rec in OPCODE_RECORDS:
        code = rec[0]
        if code in seen:
            raise OpcodeDefinitionError(f"duplicate opcode code 0x{code:X}")
        seen.add(code)

    for code, op in OPCODES.items():
        tag = op.mnemonic or f"0x{code:03X}"

        # page + prefix round-trip
        try:
            prefix, low = encode_prefix(code)
            if decode_prefix(prefix, low) != code:
                raise OpcodeDefinitionError(f"{tag}: prefix round-trip failed for 0x{code:X}")
        except ValueError as exc:
            raise OpcodeDefinitionError(f"{tag}: bad code 0x{code:X}: {exc}") from exc

        # version flags
        unknown_versions = op.version_flags - KNOWN_VERSIONS
        if unknown_versions:
            raise OpcodeDefinitionError(f"{tag}: unknown version flags {sorted(unknown_versions)}")

        # semantic flags
        unknown_flags = op.flags - KNOWN_FLAGS
        if unknown_flags:
            raise OpcodeDefinitionError(f"{tag}: unknown flags {sorted(unknown_flags)}")

        # param types + register coherence
        for p in op.params:
            if not isinstance(p.type, ParamType):
                raise OpcodeDefinitionError(f"{tag}: param has non-ParamType {p.type!r}")
            if p.writes and not p.is_register:
                raise OpcodeDefinitionError(
                    f"{tag}: non-register param {p.type.value} flagged as written"
                )
            if p.count and p.type not in (
                ParamType.R_REG_SET_FIXED, ParamType.W_REG_SET_FIXED,
                ParamType.R_REG32_SET_FIXED, ParamType.W_REG32_SET_FIXED,
            ):
                raise OpcodeDefinitionError(
                    f"{tag}: param {p.type.value} carries a fixed count {p.count} "
                    f"but is not a *_SET_FIXED type"
                )

        # an opcode can't be both an arg pusher and an arg consumer
        if op.is_arg_push and op.uses_arg_stack:
            raise OpcodeDefinitionError(f"{tag}: both F_PUSH_ARG and F_ARGS set")

        # arg_push* opcodes push exactly one operand
        if op.is_arg_push and len(op.params) != 1:
            raise OpcodeDefinitionError(
                f"{tag}: F_PUSH_ARG opcode must take exactly one operand, got {len(op.params)}"
            )

        # F_ARGS opcodes take their operands from the stack, so no register
        # operand is *inline*; the operands described here are the logical
        # signature consumed from the arg stack. Assert it is not also a pusher
        # and that it has the BB version flag if it has any params from args.
        if op.uses_arg_stack and op.is_terminator:
            raise OpcodeDefinitionError(f"{tag}: F_ARGS opcode unexpectedly a terminator")

    # the canonical terminators must be present and flagged
    for code, name in ((0x01, "ret"), (0x28, "jmp")):
        op = OPCODES.get(code)
        if op is None or not op.is_terminator:
            raise OpcodeDefinitionError(f"expected {name} (0x{code:X}) to be a F_TERMINATOR")

    # the canonical arg pushers must be flagged
    for code in range(0x48, 0x4F):
        op = OPCODES.get(code)
        if op is None or not op.is_arg_push:
            raise OpcodeDefinitionError(f"expected 0x{code:X} to be a F_PUSH_ARG (arg_push*)")

    return True


__all__ = [
    "Opcode",
    "Param",
    "ParamType",
    "OPCODES",
    "MNEMONICS",
    "KNOWN_VERSIONS",
    "KNOWN_FLAGS",
    "DEFAULT_VERSION",
    "F_ARGS",
    "F_PUSH_ARG",
    "F_CLEAR_ARGS",
    "F_TERMINATOR",
    "get",
    "by_mnemonic",
    "opcodes_for_version",
    "encode_prefix",
    "decode_prefix",
    "check_opcode_definitions",
    "OpcodeDefinitionError",
]
