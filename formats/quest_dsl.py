"""PSOBB Quest DSL — Layer 1 authoring language over the Layer-0 assembler.

This is the *author-friendly* layer of the quest-scripting stack. Where
Layer 0 (:mod:`formats.quest_asm`) is a faithful, byte-exact mirror of the
quest bytecode — registers, raw opcodes, label tables — Layer 1 lets a
non-programmer write quests in a clean, Python-flavoured language and have
it **compile down to Layer-0 assembly text**, which the assembler then turns
into a real ``.bin``.

Pipeline
--------
``compile_dsl(text)``::

    .quest source
        -> tokenizer            (formats.quest_dsl._Lexer)
        -> recursive-descent parser + Pratt expressions  (_Parser -> AST)
        -> semantic pass        (_SemAnalyzer: scopes, types, registers)
        -> codegen              (_CodeGen -> Layer-0 assembly TEXT)
        -> quest_asm.assemble() (verify it assembles to bytecode)

There is **no register allocator**: the BB VM has 256 registers, which is
plenty, so the compiler hands out registers from a simple monotonic pool
(user variables first, then a scratch/temp stack that is freed eagerly when
an expression is done). The reserved registers are honoured:

* ``R250`` — difficulty slot,
* ``R251`` — game/episode mode,
* ``R255`` — quest success/state.

These three are never handed out to user variables; they can be read/written
explicitly via their well-known names (``difficulty``, ``game_mode``,
``quest_success``) — see :data:`RESERVED_REGISTERS`.

Language (a quick tour)
-----------------------
::

    quest "My Quest" {
        episode 1
        difficulty                       # declares nothing; just doc

        # global variables live in registers
        var kills = 0

        # the mandatory boot thread — auto-emits the leading `sync`
        thread main {
            message 1, "Welcome, hunter!"
            set_flag 60
            if kills >= 3 {
                give_item 0x00, 0x01, 0x00     # a Saber
                quest_success = 1
            }
        }

        floor_handler floor=0 {
            spawn npc=8 floor=0 section=0 x=10.0 y=0.0 z=-5.0 dir=0.0
        }

        npc Tyrel {
            skin 27
            floor 0
            section 0
            pos 100.0, 0.0, -20.0
            dir 0.0
            on_talk {
                window_msg "Good luck out there."
                choose {
                    "Yes" -> { set_flag 61 }
                    "No"  -> { }
                }
            }
        }
    }

What compiles to what (the construct catalogue)
-----------------------------------------------
* ``var x = expr`` / ``x = expr``  -> register let + arithmetic chain.
* ``if/elif/else``, ``while``, ``for`` -> structured Layer-0 ``if``/``while``
  + ``jmp`` skeleton (zero runtime cost).
* ``function f(...) { ... }`` + ``call f()`` -> a labelled subroutine and a
  Layer-0 ``call``.
* ``thread name { ... }`` -> a labelled block that is registered with
  ``thread`` from the entry point and ALWAYS begins with ``sync`` (a quest
  thread that does not start with ``sync`` crashes the game — the compiler
  guarantees this).
* ``message id, "text"`` / ``window_msg "text"`` -> the F_ARGS text opcodes.
* ``set_flag n`` / ``clear_flag n`` / ``get_flag n into x`` -> event flags.
* ``give_item a, b, c`` -> builds the 3-register item descriptor and calls
  ``item_create``.
* ``spawn npc=.. floor=.. ..`` / ``npc Name { ... }`` -> NPC creation via
  the 6-register ``npc_crp`` descriptor (+ ``on_talk`` dialogue handler).
* ``floor_handler floor=N { ... }`` -> ``set_floor_handler`` + a handler sub.
* ``choose { "label" -> { ... } }`` -> a ``list`` selection + dispatch.
* ``trigger`` / ``on_enter`` / proximity -> floor / coords handlers.
* ``asm { ... }`` -> a raw Layer-0 escape hatch (verbatim assembly).

Round-trip + lift
-----------------
Like :mod:`formats.mob_dsl`, this module is round-trippable in the
*best-effort* sense the brief asks for: :func:`lift_bin` disassembles a
``.bin`` and recognises the common compiled construct shapes, re-emitting
DSL. Anything it does not recognise is preserved verbatim inside an
``asm { ... }`` block (and logged), so the lifted DSL always recompiles to
*equivalent* bytecode. Full decompilation is explicitly out of scope.

Diagnostics
-----------
Every error carries a 1-based ``line`` and ``col`` and a short message.
:class:`DSLSyntaxError` (lex/parse), :class:`DSLSemanticError` (undefined
var, type mismatch, unknown construct), and :class:`DSLError` (base).
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from formats import quest_asm
from formats import quest_bin
from formats import quest_opcodes as q

__all__ = [
    "DSLError",
    "DSLSyntaxError",
    "DSLSemanticError",
    "RESERVED_REGISTERS",
    "compile_dsl",
    "compile_dsl_to_asm",
    "compile_dsl_to_bin",
    "lift_bin",
    "make_bb_template",
]


# ---------------------------------------------------------------------------
# Reserved register conventions (REFERENCE_DATA.md §Quest Register/Flag).
# ---------------------------------------------------------------------------
#: Well-known register names the DSL exposes directly. These map onto the
#: reserved slots and are *never* allocated to ordinary user variables.
RESERVED_REGISTERS: Dict[str, int] = {
    "difficulty": 250,     # R250 — 0..3
    "game_mode": 251,      # R251 — normal/story/challenge/ultimate
    "player_count": 252,   # R252
    "episode_reg": 253,    # R253
    "area_index": 254,     # R254 — current floor
    "quest_success": 255,  # R255 — 0=in progress, 1=complete
}
_RESERVED_SLOTS = frozenset(RESERVED_REGISTERS.values())

#: User variables are handed out from this base upward, skipping reserved
#: slots. R0..R99 are "system/scoring" by convention and R100..R199 are
#: per-player; we hand free slots from R200 (designer-free per the doc) and
#: spill downward only if exhausted.
_VAR_BASE = 200
_VAR_MAX = 249           # R200..R249 are the "free" designer slots
_TEMP_BASE = 100         # scratch/temporaries live high in the per-player band
_TEMP_MAX = 199


# The DSL writes the natural episode number (1=Ep1, 2=Ep2, 4=Ep4). Both the
# BB header byte (quest_bin._BB_EPISODE_OFF) and the set_episode opcode are
# 0-indexed (0=Ep1, 1=Ep2, 2=Ep4). These helpers convert in both directions.
_EPISODE_DSL_TO_RAW = {1: 0, 2: 1, 4: 2}
_EPISODE_RAW_TO_DSL = {0: 1, 1: 2, 2: 4}


def _episode_to_raw(ep_dsl: int) -> int:
    """Map a DSL episode number (1/2/4) to the 0-indexed engine value."""
    return _EPISODE_DSL_TO_RAW.get(ep_dsl, max(0, ep_dsl - 1))


def _episode_to_dsl(ep_raw: int) -> int:
    """Map a 0-indexed engine episode value back to the DSL number (1/2/4)."""
    return _EPISODE_RAW_TO_DSL.get(ep_raw, ep_raw + 1)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
class DSLError(Exception):
    """Base class for all DSL compile diagnostics."""

    def __init__(self, message: str, *, line: int = 0, col: int = 0) -> None:
        self.message = message
        self.line = line
        self.col = col
        where = f"line {line}, col {col}: " if line else ""
        super().__init__(f"{where}{message}")


class DSLSyntaxError(DSLError):
    """A lexing or parsing error (malformed token / unexpected structure)."""


class DSLSemanticError(DSLError):
    """A semantic error (undefined var, type mismatch, unknown construct)."""


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------
@dataclass
class Token:
    kind: str          # 'ID','NUM','FLOAT','STR','OP','NL','EOF'
    value: object
    line: int
    col: int

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Token({self.kind},{self.value!r}@{self.line}:{self.col})"


# Multi-char operators, longest first so the lexer matches greedily.
_OPERATORS = [
    "->", "==", "!=", ">=", "<=", "&&", "||",
    "+", "-", "*", "/", "%", "=", ">", "<",
    "{", "}", "(", ")", ",", ".", "!",
]
_KEYWORDS = frozenset({
    "quest", "thread", "function", "call", "return", "var",
    "if", "elif", "else", "while", "for", "in",
    "npc", "floor_handler", "spawn", "wave", "trigger",
    "on_talk", "on_enter", "choose", "asm",
    "true", "false",
    # construct field keywords are treated as identifiers; only the above
    # introduce structure.
})


class _Lexer:
    """Hand-written lexer. Produces a flat token list (newlines significant
    only as statement separators — we keep them and let the parser skip
    runs).
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: List[Token] = []

    def _advance(self, n: int = 1) -> None:
        for _ in range(n):
            if self.pos < len(self.text) and self.text[self.pos] == "\n":
                self.line += 1
                self.col = 1
            else:
                self.col += 1
            self.pos += 1

    def tokenize(self) -> List[Token]:
        text = self.text
        n = len(text)
        while self.pos < n:
            ch = text[self.pos]
            # Newline is a token (statement separator).
            if ch == "\n":
                self.tokens.append(Token("NL", "\n", self.line, self.col))
                self._advance()
                continue
            # Whitespace (not newline).
            if ch in " \t\r":
                self._advance()
                continue
            # Line comments: # ... or // ...
            if ch == "#" or (ch == "/" and self.pos + 1 < n and text[self.pos + 1] == "/"):
                while self.pos < n and text[self.pos] != "\n":
                    self._advance()
                continue
            # Block comments: /* ... */
            if ch == "/" and self.pos + 1 < n and text[self.pos + 1] == "*":
                start_l, start_c = self.line, self.col
                self._advance(2)
                while self.pos < n and not (
                    text[self.pos] == "*" and self.pos + 1 < n and text[self.pos + 1] == "/"
                ):
                    self._advance()
                if self.pos >= n:
                    raise DSLSyntaxError("unterminated block comment", line=start_l, col=start_c)
                self._advance(2)
                continue
            # String literal.
            if ch == '"':
                self._read_string()
                continue
            # Number (int / float / hex).
            if ch.isdigit() or (ch == "." and self.pos + 1 < n and text[self.pos + 1].isdigit()):
                self._read_number()
                continue
            # Identifier / keyword.
            if ch.isalpha() or ch == "_":
                self._read_ident()
                continue
            # Operator / punctuation.
            self._read_operator()
        self.tokens.append(Token("EOF", None, self.line, self.col))
        return self.tokens

    def _read_string(self) -> None:
        start_l, start_c = self.line, self.col
        self._advance()  # opening quote
        out: List[str] = []
        text = self.text
        n = len(text)
        while self.pos < n and text[self.pos] != '"':
            c = text[self.pos]
            if c == "\\":
                self._advance()
                if self.pos >= n:
                    raise DSLSyntaxError("unterminated string escape", line=start_l, col=start_c)
                esc = text[self.pos]
                out.append({"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"'}.get(esc, esc))
                self._advance()
                continue
            if c == "\n":
                raise DSLSyntaxError("newline in string literal", line=start_l, col=start_c)
            out.append(c)
            self._advance()
        if self.pos >= n:
            raise DSLSyntaxError("unterminated string literal", line=start_l, col=start_c)
        self._advance()  # closing quote
        self.tokens.append(Token("STR", "".join(out), start_l, start_c))

    def _read_number(self) -> None:
        start_l, start_c = self.line, self.col
        text = self.text
        n = len(text)
        start = self.pos
        is_float = False
        # Hex
        if text[self.pos] == "0" and self.pos + 1 < n and text[self.pos + 1] in "xX":
            self._advance(2)
            while self.pos < n and (text[self.pos] in "0123456789abcdefABCDEF_"):
                self._advance()
            raw = text[start:self.pos].replace("_", "")
            try:
                val = int(raw, 16)
            except ValueError:
                raise DSLSyntaxError(f"malformed hex literal {raw!r}", line=start_l, col=start_c)
            self.tokens.append(Token("NUM", val, start_l, start_c))
            return
        while self.pos < n and (text[self.pos].isdigit() or text[self.pos] in "._"):
            if text[self.pos] == ".":
                # Allow exactly one dot; a second dot ends the number.
                if is_float:
                    break
                is_float = True
            self._advance()
        # exponent
        if self.pos < n and text[self.pos] in "eE":
            is_float = True
            self._advance()
            if self.pos < n and text[self.pos] in "+-":
                self._advance()
            while self.pos < n and text[self.pos].isdigit():
                self._advance()
        raw = text[start:self.pos].replace("_", "")
        if is_float:
            try:
                self.tokens.append(Token("FLOAT", float(raw), start_l, start_c))
            except ValueError:
                raise DSLSyntaxError(f"malformed float literal {raw!r}", line=start_l, col=start_c)
        else:
            self.tokens.append(Token("NUM", int(raw), start_l, start_c))

    def _read_ident(self) -> None:
        start_l, start_c = self.line, self.col
        text = self.text
        n = len(text)
        start = self.pos
        while self.pos < n and (text[self.pos].isalnum() or text[self.pos] == "_"):
            self._advance()
        self.tokens.append(Token("ID", text[start:self.pos], start_l, start_c))

    def _read_operator(self) -> None:
        start_l, start_c = self.line, self.col
        text = self.text
        for op in _OPERATORS:
            if text.startswith(op, self.pos):
                self._advance(len(op))
                self.tokens.append(Token("OP", op, start_l, start_c))
                return
        raise DSLSyntaxError(f"unexpected character {text[self.pos]!r}", line=start_l, col=start_c)


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------
@dataclass
class Node:
    line: int = 0
    col: int = 0


# --- expressions -----------------------------------------------------------
@dataclass
class NumLit(Node):
    value: int = 0


@dataclass
class FloatLit(Node):
    value: float = 0.0


@dataclass
class StrLit(Node):
    value: str = ""


@dataclass
class BoolLit(Node):
    value: bool = False


@dataclass
class VarRef(Node):
    name: str = ""


@dataclass
class BinOp(Node):
    op: str = ""
    left: Optional[Node] = None
    right: Optional[Node] = None


@dataclass
class UnaryOp(Node):
    op: str = ""
    operand: Optional[Node] = None


# --- statements ------------------------------------------------------------
@dataclass
class VarDecl(Node):
    name: str = ""
    init: Optional[Node] = None


@dataclass
class Assign(Node):
    name: str = ""
    value: Optional[Node] = None


@dataclass
class IfStmt(Node):
    cond: Optional[Node] = None
    body: List[Node] = field(default_factory=list)
    elifs: List[Tuple[Node, List[Node]]] = field(default_factory=list)
    else_body: Optional[List[Node]] = None


@dataclass
class WhileStmt(Node):
    cond: Optional[Node] = None
    body: List[Node] = field(default_factory=list)


@dataclass
class ForStmt(Node):
    var: str = ""
    start: Optional[Node] = None
    end: Optional[Node] = None
    body: List[Node] = field(default_factory=list)


@dataclass
class CallStmt(Node):
    name: str = ""           # construct keyword or function name
    args: List[Node] = field(default_factory=list)
    kwargs: Dict[str, Node] = field(default_factory=dict)


@dataclass
class ReturnStmt(Node):
    pass


@dataclass
class AsmBlock(Node):
    text: str = ""


@dataclass
class ChooseStmt(Node):
    # list of (label_text, body) plus optional prompt
    options: List[Tuple[str, List[Node]]] = field(default_factory=list)
    prompt: Optional[str] = None


@dataclass
class FunctionDef(Node):
    name: str = ""
    params: List[str] = field(default_factory=list)
    body: List[Node] = field(default_factory=list)


@dataclass
class ThreadDef(Node):
    name: str = ""
    body: List[Node] = field(default_factory=list)


@dataclass
class FloorHandlerDef(Node):
    floor: int = 0
    body: List[Node] = field(default_factory=list)


@dataclass
class OnTalkDef(Node):
    body: List[Node] = field(default_factory=list)


@dataclass
class NpcDef(Node):
    name: str = ""
    skin: int = 8
    floor: int = 0
    section: int = 0
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    dir: float = 0.0
    on_talk: Optional[List[Node]] = None


@dataclass
class QuestDef(Node):
    name: str = "Untitled"
    episode: int = 1
    max_players: int = 4
    body: List[Node] = field(default_factory=list)


@dataclass
class Program(Node):
    quest: Optional[QuestDef] = None
    # statements/defs outside a quest{} block (top-level threads, funcs).
    items: List[Node] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parser (recursive descent + Pratt expressions)
# ---------------------------------------------------------------------------
# Pratt binding powers for binary operators.
_BIN_PREC = {
    "||": 1, "&&": 2,
    "==": 3, "!=": 3, "<": 3, ">": 3, "<=": 3, ">=": 3,
    "+": 4, "-": 4,
    "*": 5, "/": 5, "%": 5,
}


class _Parser:
    def __init__(self, tokens: List[Token]) -> None:
        self.toks = tokens
        self.i = 0

    # -- token helpers ------------------------------------------------------
    def _peek(self, k: int = 0) -> Token:
        j = self.i + k
        if j >= len(self.toks):
            return self.toks[-1]
        return self.toks[j]

    def _skip_nl(self) -> None:
        while self._peek().kind == "NL":
            self.i += 1

    def _next(self) -> Token:
        t = self.toks[self.i]
        if self.i < len(self.toks) - 1:
            self.i += 1
        return t

    def _at(self, kind: str, value=None) -> bool:
        t = self._peek()
        if t.kind != kind:
            return False
        return value is None or t.value == value

    def _expect(self, kind: str, value=None) -> Token:
        t = self._peek()
        if t.kind != kind or (value is not None and t.value != value):
            want = value if value is not None else kind
            raise DSLSyntaxError(
                f"expected {want!r}, found {t.value!r}", line=t.line, col=t.col
            )
        return self._next()

    def _expect_op(self, value: str) -> Token:
        return self._expect("OP", value)

    # -- entry --------------------------------------------------------------
    def parse_program(self) -> Program:
        prog = Program()
        self._skip_nl()
        while not self._at("EOF"):
            if self._at("ID", "quest"):
                if prog.quest is not None:
                    t = self._peek()
                    raise DSLSyntaxError("only one quest{} block allowed", line=t.line, col=t.col)
                prog.quest = self._parse_quest()
            else:
                prog.items.append(self._parse_top_item())
            self._skip_nl()
        return prog

    def _parse_quest(self) -> QuestDef:
        kw = self._expect("ID", "quest")
        node = QuestDef(line=kw.line, col=kw.col)
        # optional quest name string
        if self._at("STR"):
            node.name = self._next().value
        self._expect_op("{")
        self._skip_nl()
        while not self._at("OP", "}"):
            if self._at("EOF"):
                raise DSLSyntaxError("unterminated quest{} block", line=kw.line, col=kw.col)
            # header directives: episode N, max_players N, name "..."
            if self._at("ID", "episode"):
                self._next()
                node.episode = self._parse_int_token()
                self._end_stmt()
                continue
            if self._at("ID", "max_players"):
                self._next()
                node.max_players = self._parse_int_token()
                self._end_stmt()
                continue
            if self._at("ID", "name"):
                self._next()
                node.name = self._expect("STR").value
                self._end_stmt()
                continue
            node.body.append(self._parse_top_item())
            self._skip_nl()
        self._expect_op("}")
        return node

    def _parse_int_token(self) -> int:
        t = self._peek()
        if t.kind == "NUM":
            self._next()
            return int(t.value)
        raise DSLSyntaxError(f"expected integer, found {t.value!r}", line=t.line, col=t.col)

    def _end_stmt(self) -> None:
        # statements end at a newline, a closing brace, or EOF.
        if self._at("OP", "}") or self._at("EOF"):
            return
        if self._at("NL"):
            self._skip_nl()
            return
        # tolerate trailing tokens up to NL? be strict:
        t = self._peek()
        raise DSLSyntaxError(f"unexpected {t.value!r} at end of statement", line=t.line, col=t.col)

    # -- top-level item (def or statement) ---------------------------------
    def _parse_top_item(self) -> Node:
        t = self._peek()
        if t.kind == "ID":
            kw = t.value
            if kw == "thread":
                return self._parse_thread()
            if kw == "function":
                return self._parse_function()
            if kw == "floor_handler":
                return self._parse_floor_handler()
            if kw == "npc":
                return self._parse_npc()
        return self._parse_statement()

    def _parse_block(self) -> List[Node]:
        self._expect_op("{")
        self._skip_nl()
        body: List[Node] = []
        while not self._at("OP", "}"):
            if self._at("EOF"):
                t = self._peek()
                raise DSLSyntaxError("unterminated block", line=t.line, col=t.col)
            body.append(self._parse_statement())
            self._skip_nl()
        self._expect_op("}")
        return body

    def _parse_thread(self) -> ThreadDef:
        kw = self._expect("ID", "thread")
        name = self._expect("ID").value
        body = self._parse_block()
        return ThreadDef(name=name, body=body, line=kw.line, col=kw.col)

    def _parse_function(self) -> FunctionDef:
        kw = self._expect("ID", "function")
        name = self._expect("ID").value
        params: List[str] = []
        if self._at("OP", "("):
            self._next()
            while not self._at("OP", ")"):
                params.append(self._expect("ID").value)
                if self._at("OP", ","):
                    self._next()
                else:
                    break
            self._expect_op(")")
        body = self._parse_block()
        return FunctionDef(name=name, params=params, body=body, line=kw.line, col=kw.col)

    def _parse_floor_handler(self) -> FloorHandlerDef:
        kw = self._expect("ID", "floor_handler")
        floor = 0
        # floor=N  (a kwarg) or a bare integer
        if self._at("ID", "floor"):
            self._next()
            self._expect_op("=")
            floor = self._parse_int_token()
        elif self._at("NUM"):
            floor = int(self._next().value)
        body = self._parse_block()
        return FloorHandlerDef(floor=floor, body=body, line=kw.line, col=kw.col)

    def _parse_npc(self) -> NpcDef:
        kw = self._expect("ID", "npc")
        name = "npc"
        if self._at("ID"):
            name = self._next().value
        elif self._at("STR"):
            name = self._next().value
        node = NpcDef(name=name, line=kw.line, col=kw.col)
        self._expect_op("{")
        self._skip_nl()
        while not self._at("OP", "}"):
            if self._at("EOF"):
                raise DSLSyntaxError("unterminated npc{} block", line=kw.line, col=kw.col)
            field_t = self._expect("ID")
            fname = field_t.value
            if fname == "skin":
                node.skin = self._parse_int_token()
            elif fname == "floor":
                node.floor = self._parse_int_token()
            elif fname == "section":
                node.section = self._parse_int_token()
            elif fname == "dir":
                node.dir = self._parse_number_token()
            elif fname == "pos":
                x = self._parse_number_token()
                self._expect_op(",")
                y = self._parse_number_token()
                self._expect_op(",")
                z = self._parse_number_token()
                node.pos = (x, y, z)
            elif fname == "on_talk":
                node.on_talk = self._parse_block()
                self._skip_nl()
                continue
            else:
                raise DSLSyntaxError(
                    f"unknown npc field {fname!r} (want skin/floor/section/pos/dir/on_talk)",
                    line=field_t.line, col=field_t.col,
                )
            # fields may be separated by newlines or just whitespace; skip any
            # newlines and continue reading the next field (or the closing }).
            self._skip_nl()
        self._expect_op("}")
        return node

    def _parse_number_token(self) -> float:
        t = self._peek()
        neg = False
        if self._at("OP", "-"):
            neg = True
            self._next()
            t = self._peek()
        if t.kind in ("NUM", "FLOAT"):
            self._next()
            v = float(t.value)
            return -v if neg else v
        raise DSLSyntaxError(f"expected number, found {t.value!r}", line=t.line, col=t.col)

    # -- statements ---------------------------------------------------------
    def _parse_statement(self) -> Node:
        t = self._peek()
        if t.kind == "ID":
            kw = t.value
            if kw == "var":
                return self._parse_var_decl()
            if kw == "if":
                return self._parse_if()
            if kw == "while":
                return self._parse_while()
            if kw == "for":
                return self._parse_for()
            if kw == "return":
                self._next()
                self._end_stmt()
                return ReturnStmt(line=t.line, col=t.col)
            if kw == "asm":
                return self._parse_asm()
            if kw == "choose":
                return self._parse_choose()
            if kw == "thread":
                return self._parse_thread()
            if kw == "function":
                return self._parse_function()
            if kw == "floor_handler":
                return self._parse_floor_handler()
            if kw == "npc":
                return self._parse_npc()
            if kw in ("on_talk", "on_enter"):
                self._next()
                body = self._parse_block()
                return OnTalkDef(body=body, line=t.line, col=t.col)
            # assignment?  ID = expr
            if self._peek(1).kind == "OP" and self._peek(1).value == "=":
                return self._parse_assign()
            # otherwise a construct/call statement: ID args...
            return self._parse_call_stmt()
        raise DSLSyntaxError(f"unexpected {t.value!r}", line=t.line, col=t.col)

    def _parse_var_decl(self) -> VarDecl:
        kw = self._expect("ID", "var")
        name = self._expect("ID").value
        init = None
        if self._at("OP", "="):
            self._next()
            init = self._parse_expr()
        self._end_stmt()
        return VarDecl(name=name, init=init, line=kw.line, col=kw.col)

    def _parse_assign(self) -> Assign:
        name_t = self._expect("ID")
        self._expect_op("=")
        value = self._parse_expr()
        self._end_stmt()
        return Assign(name=name_t.value, value=value, line=name_t.line, col=name_t.col)

    def _parse_if(self) -> IfStmt:
        kw = self._expect("ID", "if")
        cond = self._parse_expr()
        body = self._parse_block()
        node = IfStmt(cond=cond, body=body, line=kw.line, col=kw.col)
        self._skip_nl()
        while self._at("ID", "elif"):
            self._next()
            ec = self._parse_expr()
            eb = self._parse_block()
            node.elifs.append((ec, eb))
            self._skip_nl()
        if self._at("ID", "else"):
            self._next()
            node.else_body = self._parse_block()
        return node

    def _parse_while(self) -> WhileStmt:
        kw = self._expect("ID", "while")
        cond = self._parse_expr()
        body = self._parse_block()
        return WhileStmt(cond=cond, body=body, line=kw.line, col=kw.col)

    def _parse_for(self) -> ForStmt:
        kw = self._expect("ID", "for")
        var = self._expect("ID").value
        self._expect("ID", "in")
        start = self._parse_expr()
        # range syntax:  start .. end   (two dots)
        self._expect_op(".")
        self._expect_op(".")
        end = self._parse_expr()
        body = self._parse_block()
        return ForStmt(var=var, start=start, end=end, body=body, line=kw.line, col=kw.col)

    def _parse_asm(self) -> AsmBlock:
        kw = self._expect("ID", "asm")
        # asm { ... }  — capture raw text between braces from the source slice.
        # We reconstruct it from tokens; to keep it faithful we re-tokenize
        # by scanning for the matching brace and joining token reprs is lossy,
        # so instead asm blocks accept STR lines: asm { "leti r1, 5" \n ... }
        self._expect_op("{")
        self._skip_nl()
        lines: List[str] = []
        while not self._at("OP", "}"):
            if self._at("EOF"):
                raise DSLSyntaxError("unterminated asm{} block", line=kw.line, col=kw.col)
            if self._at("STR"):
                lines.append(self._next().value)
                self._end_stmt()
            elif self._at("NL"):
                self._skip_nl()
            else:
                t = self._peek()
                raise DSLSyntaxError(
                    "asm{} block lines must be quoted assembly strings",
                    line=t.line, col=t.col,
                )
        self._expect_op("}")
        return AsmBlock(text="\n".join(lines), line=kw.line, col=kw.col)

    def _parse_choose(self) -> ChooseStmt:
        kw = self._expect("ID", "choose")
        node = ChooseStmt(line=kw.line, col=kw.col)
        if self._at("STR"):  # optional prompt
            node.prompt = self._next().value
        self._expect_op("{")
        self._skip_nl()
        while not self._at("OP", "}"):
            if self._at("EOF"):
                raise DSLSyntaxError("unterminated choose{} block", line=kw.line, col=kw.col)
            label = self._expect("STR").value
            self._expect_op("->")
            body = self._parse_block()
            node.options.append((label, body))
            self._skip_nl()
        self._expect_op("}")
        return node

    def _parse_call_stmt(self) -> CallStmt:
        name_t = self._expect("ID")
        node = CallStmt(name=name_t.value, line=name_t.line, col=name_t.col)
        # Two arg styles:
        #   foo a, b, c            (positional, comma-separated expressions)
        #   foo(a, b)              (explicit call form)
        #   spawn npc=8 floor=0    (space-separated kwargs)
        if self._at("OP", "("):
            self._next()
            while not self._at("OP", ")"):
                node.args.append(self._parse_expr())
                if self._at("OP", ","):
                    self._next()
                else:
                    break
            self._expect_op(")")
            self._end_stmt()
            return node
        # collect args until newline / brace / EOF
        first = True
        while not (self._at("NL") or self._at("OP", "}") or self._at("EOF")):
            # kwarg form: ID = expr
            if self._at("ID") and self._peek(1).kind == "OP" and self._peek(1).value == "=":
                key = self._next().value
                self._expect_op("=")
                node.kwargs[key] = self._parse_expr()
            else:
                if not first:
                    # require a comma between positional args (a, b, c)
                    if self._at("OP", ","):
                        self._next()
                node.args.append(self._parse_expr())
            first = False
            # optional comma between positional args
            if self._at("OP", ","):
                self._next()
        self._end_stmt()
        return node

    # -- expressions (Pratt) ------------------------------------------------
    def _parse_expr(self, min_bp: int = 0) -> Node:
        left = self._parse_unary()
        while True:
            t = self._peek()
            if t.kind != "OP" or t.value not in _BIN_PREC:
                break
            bp = _BIN_PREC[t.value]
            if bp < min_bp:
                break
            op = self._next().value
            right = self._parse_expr(bp + 1)
            left = BinOp(op=op, left=left, right=right, line=t.line, col=t.col)
        return left

    def _parse_unary(self) -> Node:
        t = self._peek()
        if t.kind == "OP" and t.value in ("-", "!"):
            self._next()
            operand = self._parse_unary()
            return UnaryOp(op=t.value, operand=operand, line=t.line, col=t.col)
        return self._parse_primary()

    def _parse_primary(self) -> Node:
        t = self._peek()
        if t.kind == "NUM":
            self._next()
            return NumLit(value=int(t.value), line=t.line, col=t.col)
        if t.kind == "FLOAT":
            self._next()
            return FloatLit(value=float(t.value), line=t.line, col=t.col)
        if t.kind == "STR":
            self._next()
            return StrLit(value=t.value, line=t.line, col=t.col)
        if t.kind == "ID":
            if t.value == "true":
                self._next()
                return BoolLit(value=True, line=t.line, col=t.col)
            if t.value == "false":
                self._next()
                return BoolLit(value=False, line=t.line, col=t.col)
            self._next()
            return VarRef(name=t.value, line=t.line, col=t.col)
        if t.kind == "OP" and t.value == "(":
            self._next()
            inner = self._parse_expr()
            self._expect_op(")")
            return inner
        raise DSLSyntaxError(f"unexpected {t.value!r} in expression", line=t.line, col=t.col)


# ---------------------------------------------------------------------------
# Semantic / scope model
# ---------------------------------------------------------------------------
# Value "type" tags for type checking. INT covers booleans (0/1).
T_INT = "int"
T_FLOAT = "float"
T_STR = "str"


class _Scope:
    """Maps DSL variable names to allocated register numbers + types."""

    def __init__(self) -> None:
        self.vars: Dict[str, Tuple[int, str]] = {}  # name -> (reg, type)
        self._next_var = _VAR_BASE

    def declare(self, name: str, typ: str, line: int, col: int) -> int:
        if name in RESERVED_REGISTERS:
            raise DSLSemanticError(
                f"{name!r} is a reserved register name and cannot be declared",
                line=line, col=col,
            )
        if name in self.vars:
            raise DSLSemanticError(f"variable {name!r} already declared", line=line, col=col)
        reg = self._alloc_var(line, col)
        self.vars[name] = (reg, typ)
        return reg

    def _alloc_var(self, line: int, col: int) -> int:
        while self._next_var in _RESERVED_SLOTS:
            self._next_var += 1
        if self._next_var > _VAR_MAX:
            raise DSLSemanticError(
                "out of variable registers (R200..R249 exhausted)", line=line, col=col
            )
        reg = self._next_var
        self._next_var += 1
        return reg

    def lookup(self, name: str) -> Optional[Tuple[int, str]]:
        if name in RESERVED_REGISTERS:
            return (RESERVED_REGISTERS[name], T_INT)
        return self.vars.get(name)


# ---------------------------------------------------------------------------
# Code generator: AST -> Layer-0 assembly text
# ---------------------------------------------------------------------------
class _CodeGen:
    """Emits Layer-0 assembly text from a parsed Program.

    The output is the *structured* assembly form the Layer-0 assembler
    supports (inline F_ARGS operands, structured ``if``/``while``). We never
    hand-roll arg_push sequences here; the assembler does that.
    """

    def __init__(self, version: str = q.DEFAULT_VERSION) -> None:
        self.version = version
        self.scope = _Scope()
        self.lines: List[str] = []
        self._label_n = 0
        self._temp_sp = _TEMP_BASE       # scratch register stack pointer
        self.warnings: List[str] = []
        # deferred subroutines (functions, handlers, on_talk bodies, threads)
        # emitted after the entry routine so the entry's `ret` terminates it.
        self._deferred: List[Tuple[str, List[Node]]] = []
        self._threads: List[str] = []     # thread label names to start at boot
        self._functions: Dict[str, str] = {}  # name -> label

    # -- helpers ------------------------------------------------------------
    def _emit(self, line: str) -> None:
        self.lines.append("    " + line if line and not line.endswith(":") else line)

    def _new_label(self, hint: str = "L") -> str:
        self._label_n += 1
        return f"_{hint}_{self._label_n}"

    def _alloc_temp(self) -> int:
        if self._temp_sp > _TEMP_MAX:
            raise DSLSemanticError("out of temporary registers")
        r = self._temp_sp
        self._temp_sp += 1
        return r

    def _free_temp(self, n: int = 1) -> None:
        self._temp_sp -= n

    # -- top-level driver ---------------------------------------------------
    def generate(self, prog: Program) -> str:
        quest = prog.quest
        episode = quest.episode if quest else 1
        name = quest.name if quest else "Untitled"
        body = list(quest.body) if quest else []
        body.extend(prog.items)

        # First, pull out defs (threads/functions/floor_handlers/npcs) so we
        # can register them at boot and emit their bodies as subroutines.
        boot_stmts: List[Node] = []
        threads: List[ThreadDef] = []
        functions: List[FunctionDef] = []
        floor_handlers: List[FloorHandlerDef] = []
        npcs: List[NpcDef] = []
        for item in body:
            if isinstance(item, ThreadDef):
                threads.append(item)
            elif isinstance(item, FunctionDef):
                functions.append(item)
            elif isinstance(item, FloorHandlerDef):
                floor_handlers.append(item)
            elif isinstance(item, NpcDef):
                npcs.append(item)
            else:
                boot_stmts.append(item)

        # Pre-assign function labels so calls can resolve forward references.
        for fn in functions:
            self._functions[fn.name] = f"func_{fn.name}"

        # Header banner (comment only; the .bin header is set separately).
        self.lines.append(f".version {self.version}")
        self.lines.append(f"// quest: {name!r}  episode {episode}")
        self.lines.append("")

        # --- raw whole-program escape -------------------------------------
        # If the quest body is a single top-level asm{} block and nothing
        # else (the lift's whole-program fallback shape), emit ONLY that
        # block verbatim — no boot scaffolding — so a lifted quest recompiles
        # to byte-identical bytecode.
        if (
            len(boot_stmts) == 1
            and isinstance(boot_stmts[0], AsmBlock)
            and not threads and not functions and not floor_handlers and not npcs
        ):
            for raw in boot_stmts[0].text.split("\n"):
                line = raw.strip()
                if not line:
                    continue
                if line.endswith(":") and " " not in line:
                    self.lines.append(line)
                else:
                    self._emit(line)
            return "\n".join(self.lines) + "\n"

        # --- entry routine (label 0) ---------------------------------------
        self.lines.append("start:")
        self._emit(f"set_episode 0x{_episode_to_raw(episode):X}")
        # register floor handlers
        for fh in floor_handlers:
            hlabel = self._new_label("floor")
            self._deferred.append((hlabel, fh.body))
            self._emit(f"set_floor_handler 0x{fh.floor:X}, {hlabel}")
        # spawn declared NPCs at boot
        for npc in npcs:
            self._gen_npc_spawn(npc)
        # boot statements (top-level code in quest body)
        for st in boot_stmts:
            self._gen_stmt(st)
        # start threads
        for th in threads:
            tlabel = f"thread_{th.name}"
            self._emit(f"thread {tlabel}")
        self._emit("ret")

        # --- deferred subroutines ------------------------------------------
        # threads (each begins with the MANDATORY sync)
        for th in threads:
            self.lines.append("")
            self.lines.append(f"thread_{th.name}:")
            self._emit("sync")          # <-- mandatory leading sync
            for st in th.body:
                self._gen_stmt(st)
            self._emit("ret")

        # functions
        for fn in functions:
            self.lines.append("")
            self.lines.append(f"{self._functions[fn.name]}:")
            for st in fn.body:
                self._gen_stmt(st)
            self._emit("ret")

        # floor-handler bodies + on_talk bodies etc. queued during emission
        # (drain the deferred queue, which may grow as we emit more)
        i = 0
        while i < len(self._deferred):
            label, stmts = self._deferred[i]
            i += 1
            self.lines.append("")
            self.lines.append(f"{label}:")
            for st in stmts:
                self._gen_stmt(st)
            self._emit("ret")

        return "\n".join(self.lines) + "\n"

    # -- NPC spawn ----------------------------------------------------------
    def _gen_npc_spawn(self, npc: NpcDef) -> None:
        """Emit an npc_crp (6-register descriptor) for a declared NPC.

        ``npc_crp`` reads 6 consecutive registers: the standard qedit layout
        is [x, y, z, dir, skin/flags, floor/section]. We stage the descriptor
        into a temp register window, then issue npc_crptalk if the NPC has an
        on_talk handler (so it is talkable), else npc_crp.
        """
        base = self._temp_sp
        if base + 6 > _TEMP_MAX:
            raise DSLSemanticError(f"npc {npc.name!r}: out of temp registers for descriptor")
        x, y, z = npc.pos
        # stage descriptor registers
        self._emit(f"fleti r{base + 0}, {self._f(x)}")
        self._emit(f"fleti r{base + 1}, {self._f(y)}")
        self._emit(f"fleti r{base + 2}, {self._f(z)}")
        self._emit(f"fleti r{base + 3}, {self._f(npc.dir)}")
        self._emit(f"leti r{base + 4}, 0x{npc.skin & 0xFFFFFFFF:X}")
        packed = (npc.floor & 0xFFFF) | ((npc.section & 0xFFFF) << 16)
        self._emit(f"leti r{base + 5}, 0x{packed:X}")
        if npc.on_talk is not None:
            tlabel = self._new_label(f"npctalk_{_safe_label(npc.name)}")
            self._emit(f"npc_crptalk r{base}-r{base + 5}")
            # register the talk handler. We also set a floor handler-like
            # callback via a chat handler; emit the dialogue body as a sub.
            self._deferred.append((tlabel, npc.on_talk))
            self.warnings.append(
                f"npc {npc.name!r}: on_talk body emitted as sub {tlabel}; "
                "wire it to the NPC's talk action in qedit if needed"
            )
        else:
            self._emit(f"npc_crp r{base}-r{base + 5}")

    @staticmethod
    def _f(v: float) -> str:
        # float32 literal in the assembler's expected `<repr>f` form
        return f"{struct.unpack('<f', struct.pack('<f', float(v)))[0]!r}f"

    # -- statements ---------------------------------------------------------
    def _gen_stmt(self, st: Node) -> None:
        if isinstance(st, VarDecl):
            self._gen_var_decl(st)
        elif isinstance(st, Assign):
            self._gen_assign(st)
        elif isinstance(st, IfStmt):
            self._gen_if(st)
        elif isinstance(st, WhileStmt):
            self._gen_while(st)
        elif isinstance(st, ForStmt):
            self._gen_for(st)
        elif isinstance(st, ReturnStmt):
            self._emit("ret")
        elif isinstance(st, AsmBlock):
            for raw in st.text.split("\n"):
                if raw.strip():
                    self._emit(raw.strip())
        elif isinstance(st, ChooseStmt):
            self._gen_choose(st)
        elif isinstance(st, CallStmt):
            self._gen_call(st)
        elif isinstance(st, OnTalkDef):
            # bare on_talk/on_enter at statement level -> a handler sub
            label = self._new_label("handler")
            self._deferred.append((label, st.body))
            self._emit(f"thread {label}")
        elif isinstance(st, (ThreadDef, FunctionDef, FloorHandlerDef, NpcDef)):
            # nested defs are lifted; flag and route to deferred
            self._gen_nested_def(st)
        else:  # pragma: no cover - defensive
            raise DSLSemanticError(f"cannot codegen node {type(st).__name__}")

    def _gen_nested_def(self, st: Node) -> None:
        if isinstance(st, ThreadDef):
            tlabel = f"thread_{st.name}"
            stmts = [_SyncMarker()] + list(st.body)
            self._deferred.append((tlabel, stmts))
            self._emit(f"thread {tlabel}")
        elif isinstance(st, FunctionDef):
            label = f"func_{st.name}"
            self._functions[st.name] = label
            self._deferred.append((label, list(st.body)))
        elif isinstance(st, FloorHandlerDef):
            hlabel = self._new_label("floor")
            self._deferred.append((hlabel, st.body))
            self._emit(f"set_floor_handler 0x{st.floor:X}, {hlabel}")
        elif isinstance(st, NpcDef):
            self._gen_npc_spawn(st)

    def _gen_var_decl(self, st: VarDecl) -> None:
        typ = T_INT
        if st.init is not None:
            typ = self._infer_type(st.init)
        reg = self.scope.declare(st.name, typ, st.line, st.col)
        if st.init is not None:
            self._gen_eval_into(st.init, reg, typ)

    def _gen_assign(self, st: Assign) -> None:
        info = self.scope.lookup(st.name)
        if info is None:
            raise DSLSemanticError(f"assignment to undefined variable {st.name!r}",
                                   line=st.line, col=st.col)
        reg, typ = info
        vtyp = self._infer_type(st.value)
        if typ == T_FLOAT and vtyp == T_INT:
            vtyp = T_FLOAT  # promote int literal to float store
        elif typ == T_INT and vtyp == T_FLOAT:
            raise DSLSemanticError(
                f"cannot assign float to int variable {st.name!r}", line=st.line, col=st.col
            )
        self._gen_eval_into(st.value, reg, typ)

    def _gen_if(self, st: IfStmt) -> None:
        end_label = self._new_label("endif")
        self._gen_cond_branch_chain(
            [(st.cond, st.body)] + st.elifs, st.else_body, end_label
        )
        self.lines.append(f"{end_label}:")

    def _gen_cond_branch_chain(self, branches, else_body, end_label) -> None:
        for cond, body in branches:
            next_label = self._new_label("next")
            self._emit_cond_jump(cond, next_label, invert=True)
            for s in body:
                self._gen_stmt(s)
            self._emit(f"jmp {end_label}")
            self.lines.append(f"{next_label}:")
        if else_body is not None:
            for s in else_body:
                self._gen_stmt(s)

    def _gen_while(self, st: WhileStmt) -> None:
        top = self._new_label("while")
        end = self._new_label("endwhile")
        self.lines.append(f"{top}:")
        self._emit_cond_jump(st.cond, end, invert=True)
        for s in st.body:
            self._gen_stmt(s)
        self._emit(f"jmp {top}")
        self.lines.append(f"{end}:")

    def _gen_for(self, st: ForStmt) -> None:
        # for v in start..end { } -> v=start; while v < end { body; v += 1 }
        reg = self.scope.declare(st.var, T_INT, st.line, st.col)
        self._gen_eval_into(st.start, reg, T_INT)
        top = self._new_label("for")
        end = self._new_label("endfor")
        endreg = self._alloc_temp()
        self._gen_eval_into(st.end, endreg, T_INT)
        self.lines.append(f"{top}:")
        # jump out if reg >= endreg  (invert of v < end)
        self._emit(f"jmp_ge r{reg}, r{endreg}, {end}")
        for s in st.body:
            self._gen_stmt(s)
        self._emit(f"addi r{reg}, 0x1")
        self._emit(f"jmp {top}")
        self.lines.append(f"{end}:")
        self._free_temp()

    # -- condition lowering -------------------------------------------------
    _CMP_REG = {"==": "jmp_eq", "!=": "jmp_ne", ">": "jmp_gt",
                "<": "jmp_lt", ">=": "jmp_ge", "<=": "jmp_le"}
    _CMP_IMM = {"==": "jmpi_eq", "!=": "jmpi_ne", ">": "jmpi_gt",
                "<": "jmpi_lt", ">=": "jmpi_ge", "<=": "jmpi_le"}
    _INVERT = {"==": "!=", "!=": "==", ">": "<=", "<": ">=", ">=": "<", "<=": ">"}

    def _emit_cond_jump(self, cond: Node, target: str, *, invert: bool) -> None:
        """Emit a jump to ``target`` taken when ``cond`` (optionally inverted)
        holds. Handles comparison, &&, ||, and truthiness.
        """
        if isinstance(cond, BinOp) and cond.op in self._CMP_REG:
            op = self._INVERT[cond.op] if invert else cond.op
            self._emit_compare_jump(op, cond.left, cond.right, target, cond.line, cond.col)
            return
        if isinstance(cond, BinOp) and cond.op == "&&":
            if invert:
                # jump if NOT(a and b) == (not a) or (not b)
                self._emit_cond_jump(cond.left, target, invert=True)
                self._emit_cond_jump(cond.right, target, invert=True)
            else:
                skip = self._new_label("and")
                self._emit_cond_jump(cond.left, skip, invert=True)
                self._emit_cond_jump(cond.right, target, invert=False)
                self.lines.append(f"{skip}:")
            return
        if isinstance(cond, BinOp) and cond.op == "||":
            if invert:
                skip = self._new_label("or")
                self._emit_cond_jump(cond.left, skip, invert=False)
                self._emit_cond_jump(cond.right, target, invert=True)
                self.lines.append(f"{skip}:")
            else:
                self._emit_cond_jump(cond.left, target, invert=False)
                self._emit_cond_jump(cond.right, target, invert=False)
            return
        if isinstance(cond, UnaryOp) and cond.op == "!":
            self._emit_cond_jump(cond.operand, target, invert=not invert)
            return
        # truthiness: treat as `cond != 0`
        op = "==" if invert else "!="
        self._emit_compare_jump(op, cond, NumLit(value=0), target, cond.line, cond.col)

    def _emit_compare_jump(self, op, left, right, target, line, col) -> None:
        # Evaluate lhs into a register; rhs either an int immediate or a reg.
        ltyp = self._infer_type(left)
        rtyp = self._infer_type(right)
        if T_STR in (ltyp, rtyp):
            raise DSLSemanticError("cannot compare strings", line=line, col=col)
        if T_FLOAT in (ltyp, rtyp):
            # float comparisons: materialise both into regs and use reg-reg
            lr = self._alloc_temp()
            self._gen_eval_into(left, lr, T_FLOAT)
            rr = self._alloc_temp()
            self._gen_eval_into(right, rr, T_FLOAT)
            self._emit(f"{self._CMP_REG[op]} r{lr}, r{rr}, {target}")
            self._free_temp(2)
            return
        lr = self._alloc_temp()
        self._gen_eval_into(left, lr, T_INT)
        if isinstance(right, NumLit):
            self._emit(f"{self._CMP_IMM[op]} r{lr}, 0x{right.value & 0xFFFFFFFF:X}, {target}")
        else:
            rr = self._alloc_temp()
            self._gen_eval_into(right, rr, T_INT)
            self._emit(f"{self._CMP_REG[op]} r{lr}, r{rr}, {target}")
            self._free_temp()
        self._free_temp()

    # -- expression evaluation ----------------------------------------------
    _ARITH_IMM = {"+": "addi", "-": "subi", "*": "muli", "/": "divi"}
    _ARITH_REG = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
    _FARITH_REG = {"+": "fadd", "-": "fsub", "*": "fmul", "/": "fdiv"}

    def _gen_eval_into(self, expr: Node, dest_reg: int, typ: str) -> None:
        """Evaluate ``expr`` and leave the result in ``dest_reg``."""
        if typ == T_FLOAT:
            self._gen_eval_float(expr, dest_reg)
            return
        self._gen_eval_int(expr, dest_reg)

    def _gen_eval_int(self, expr: Node, dest: int) -> None:
        if isinstance(expr, NumLit):
            self._emit(f"leti r{dest}, 0x{expr.value & 0xFFFFFFFF:X}")
            return
        if isinstance(expr, BoolLit):
            self._emit(f"leti r{dest}, 0x{1 if expr.value else 0:X}")
            return
        if isinstance(expr, VarRef):
            src = self._resolve_var(expr)
            self._emit(f"let r{dest}, r{src}")
            return
        if isinstance(expr, UnaryOp) and expr.op == "-":
            self._gen_eval_int(expr.operand, dest)
            self._emit(f"muli r{dest}, 0xFFFFFFFF")  # * -1
            return
        if isinstance(expr, UnaryOp) and expr.op == "!":
            # logical not: result = (operand == 0) ? 1 : 0
            self._gen_eval_int(expr.operand, dest)
            t = self._new_label("not")
            e = self._new_label("endnot")
            self._emit(f"jmpi_eq r{dest}, 0x0, {t}")
            self._emit(f"leti r{dest}, 0x0")
            self._emit(f"jmp {e}")
            self.lines.append(f"{t}:")
            self._emit(f"leti r{dest}, 0x1")
            self.lines.append(f"{e}:")
            return
        if isinstance(expr, BinOp) and expr.op in self._ARITH_IMM:
            # left into dest, then apply op with right
            self._gen_eval_int(expr.left, dest)
            if isinstance(expr.right, NumLit):
                self._emit(f"{self._ARITH_IMM[expr.op]} r{dest}, 0x{expr.right.value & 0xFFFFFFFF:X}")
            else:
                rt = self._alloc_temp()
                self._gen_eval_int(expr.right, rt)
                self._emit(f"{self._ARITH_REG[expr.op]} r{dest}, r{rt}")
                self._free_temp()
            return
        if isinstance(expr, BinOp) and expr.op in ("==", "!=", "<", ">", "<=", ">="):
            # comparison producing 0/1
            t = self._new_label("cmp")
            e = self._new_label("endcmp")
            self._emit(f"leti r{dest}, 0x0")
            self._emit_compare_jump(expr.op, expr.left, expr.right, t, expr.line, expr.col)
            self._emit(f"jmp {e}")
            self.lines.append(f"{t}:")
            self._emit(f"leti r{dest}, 0x1")
            self.lines.append(f"{e}:")
            return
        if isinstance(expr, BinOp) and expr.op in ("&&", "||"):
            t = self._new_label("logic")
            e = self._new_label("endlogic")
            self._emit(f"leti r{dest}, 0x0")
            self._emit_cond_jump(expr, t, invert=False)
            self._emit(f"jmp {e}")
            self.lines.append(f"{t}:")
            self._emit(f"leti r{dest}, 0x1")
            self.lines.append(f"{e}:")
            return
        raise DSLSemanticError(
            f"unsupported integer expression {type(expr).__name__}",
            line=getattr(expr, "line", 0), col=getattr(expr, "col", 0),
        )

    def _gen_eval_float(self, expr: Node, dest: int) -> None:
        if isinstance(expr, (FloatLit, NumLit)):
            self._emit(f"fleti r{dest}, {self._f(float(expr.value))}")
            return
        if isinstance(expr, VarRef):
            src = self._resolve_var(expr)
            self._emit(f"flet r{dest}, r{src}")
            return
        if isinstance(expr, UnaryOp) and expr.op == "-":
            self._gen_eval_float(expr.operand, dest)
            neg = self._alloc_temp()
            self._emit(f"fleti r{neg}, {self._f(-1.0)}")
            self._emit(f"fmul r{dest}, r{neg}")
            self._free_temp()
            return
        if isinstance(expr, BinOp) and expr.op in self._FARITH_REG:
            self._gen_eval_float(expr.left, dest)
            rt = self._alloc_temp()
            self._gen_eval_float(expr.right, rt)
            self._emit(f"{self._FARITH_REG[expr.op]} r{dest}, r{rt}")
            self._free_temp()
            return
        raise DSLSemanticError(
            f"unsupported float expression {type(expr).__name__}",
            line=getattr(expr, "line", 0), col=getattr(expr, "col", 0),
        )

    def _resolve_var(self, ref: VarRef) -> int:
        info = self.scope.lookup(ref.name)
        if info is None:
            raise DSLSemanticError(f"undefined variable {ref.name!r}", line=ref.line, col=ref.col)
        return info[0]

    def _infer_type(self, expr: Node) -> str:
        if isinstance(expr, FloatLit):
            return T_FLOAT
        if isinstance(expr, (NumLit, BoolLit)):
            return T_INT
        if isinstance(expr, StrLit):
            return T_STR
        if isinstance(expr, VarRef):
            info = self.scope.lookup(expr.name)
            if info is None:
                raise DSLSemanticError(f"undefined variable {expr.name!r}",
                                       line=expr.line, col=expr.col)
            return info[1]
        if isinstance(expr, UnaryOp):
            if expr.op == "!":
                return T_INT
            return self._infer_type(expr.operand)
        if isinstance(expr, BinOp):
            if expr.op in ("==", "!=", "<", ">", "<=", ">=", "&&", "||"):
                return T_INT
            lt = self._infer_type(expr.left)
            rt = self._infer_type(expr.right)
            if T_STR in (lt, rt):
                raise DSLSemanticError("string is not a numeric operand",
                                       line=expr.line, col=expr.col)
            return T_FLOAT if T_FLOAT in (lt, rt) else T_INT
        raise DSLSemanticError(f"cannot infer type of {type(expr).__name__}",
                               line=getattr(expr, "line", 0), col=getattr(expr, "col", 0))

    # -- construct / call statements ---------------------------------------
    def _gen_call(self, st: CallStmt) -> None:
        name = st.name
        handler = getattr(self, f"_construct_{name}", None)
        if handler is not None:
            handler(st)
            return
        # plain function call?
        if name == "call":
            # call f()  — first arg is a function name expressed as a VarRef
            if not st.args or not isinstance(st.args[0], VarRef):
                raise DSLSemanticError("call requires a function name", line=st.line, col=st.col)
            fname = st.args[0].name
            label = self._functions.get(fname)
            if label is None:
                raise DSLSemanticError(f"call to undefined function {fname!r}",
                                       line=st.line, col=st.col)
            self._emit(f"call {label}")
            return
        if name in self._functions:
            self._emit(f"call {self._functions[name]}")
            return
        # finally, allow a raw opcode mnemonic passthrough (power-user).
        op = q.by_mnemonic(name)
        if op is not None:
            self._gen_raw_opcode(name, op, st)
            return
        raise DSLSemanticError(
            f"unknown construct or function {name!r}", line=st.line, col=st.col
        )

    def _gen_raw_opcode(self, name: str, op, st: CallStmt) -> None:
        """Emit a raw opcode mnemonic with literal operands (escape hatch)."""
        parts = []
        for a in st.args:
            parts.append(self._literal_operand(a))
        self._emit(f"{name} " + ", ".join(parts) if parts else name)

    def _literal_operand(self, a: Node) -> str:
        if isinstance(a, NumLit):
            return f"0x{a.value & 0xFFFFFFFF:X}"
        if isinstance(a, FloatLit):
            return self._f(a.value)
        if isinstance(a, StrLit):
            return _asm_quote(a.value)
        if isinstance(a, VarRef):
            return f"r{self._resolve_var(a)}"
        raise DSLSemanticError("unsupported literal operand for raw opcode",
                               line=getattr(a, "line", 0), col=getattr(a, "col", 0))

    # ----- individual constructs ------------------------------------------
    def _need_int(self, expr: Node, what: str) -> int:
        if not isinstance(expr, NumLit):
            raise DSLSemanticError(f"{what} must be an integer literal",
                                   line=getattr(expr, "line", 0), col=getattr(expr, "col", 0))
        return expr.value

    def _int_arg(self, expr: Node, what: str, scratch: List[int]) -> Tuple[bool, object]:
        """Resolve an int expression for an F_ARGS opcode operand.

        Returns ``(is_reg, value)``: ``(False, int)`` for a constant or
        ``(True, reg_num)`` for a variable / materialised expression. The
        structured assembler cannot push a *register* for an integer F_ARGS
        param, so the caller emits ``arg_pushr`` explicitly for the register
        case (see :meth:`_emit_eventflag`). Temps go on ``scratch`` to free.
        """
        if isinstance(expr, NumLit):
            return (False, expr.value & 0xFFFFFFFF)
        if isinstance(expr, BoolLit):
            return (False, 1 if expr.value else 0)
        if isinstance(expr, VarRef):
            info = self.scope.lookup(expr.name)
            if info is None:
                raise DSLSemanticError(f"undefined variable {expr.name!r}",
                                       line=expr.line, col=expr.col)
            return (True, info[0])
        if self._infer_type(expr) == T_STR:
            raise DSLSemanticError(f"{what} cannot be a string",
                                   line=getattr(expr, "line", 0), col=getattr(expr, "col", 0))
        tmp = self._alloc_temp()
        scratch.append(tmp)
        self._gen_eval_int(expr, tmp)
        return (True, tmp)

    def _emit_eventflag(self, flag: Node, value: Node, what: str, st: CallStmt) -> None:
        """Emit ``set_eventflag flag, value`` accepting constants or registers.

        ``set_eventflag`` is an F_ARGS opcode whose two params are I32. When
        both args are constants we use the structured inline form (the
        assembler synthesises arg_push of the right width). When either is a
        register we hand-roll the ``arg_push*`` sequence (the assembler's
        structured F_ARGS path rejects a register for an I32 param).
        """
        scratch: List[int] = []
        f_is_reg, f_val = self._int_arg(flag, f"{what} flag id", scratch)
        v_is_reg, v_val = self._int_arg(value, f"{what} value", scratch)
        if not f_is_reg and not v_is_reg:
            self._emit(f"set_eventflag 0x{f_val:X}, 0x{v_val:X}")
        else:
            # literal arg_push sequence then bare F_ARGS opcode (push in
            # signature order: flag then value).
            self._emit(f"arg_pushr r{f_val}" if f_is_reg else f"arg_pushl 0x{f_val:X}")
            self._emit(f"arg_pushr r{v_val}" if v_is_reg else f"arg_pushl 0x{v_val:X}")
            self._emit("set_eventflag")
        self._free_temp(len(scratch))

    def _construct_message(self, st: CallStmt) -> None:
        if len(st.args) != 2:
            raise DSLSemanticError("message expects (id, text)", line=st.line, col=st.col)
        mid = self._need_int(st.args[0], "message id")
        text = self._need_str(st.args[1], "message text")
        self._emit(f"message 0x{mid & 0xFFFFFFFF:X}, {_asm_quote(text)}")
        self._emit("message_end")

    def _construct_window_msg(self, st: CallStmt) -> None:
        if len(st.args) != 1:
            raise DSLSemanticError("window_msg expects (text)", line=st.line, col=st.col)
        text = self._need_str(st.args[0], "window_msg text")
        self._emit(f"window_msg {_asm_quote(text)}")
        self._emit("window_msg_end")

    def _construct_add_msg(self, st: CallStmt) -> None:
        text = self._need_str(st.args[0], "add_msg text")
        self._emit(f"add_msg {_asm_quote(text)}")

    def _need_str(self, expr: Node, what: str) -> str:
        if not isinstance(expr, StrLit):
            raise DSLSemanticError(f"{what} must be a string literal",
                                   line=getattr(expr, "line", 0), col=getattr(expr, "col", 0))
        return expr.value

    def _construct_set_flag(self, st: CallStmt) -> None:
        if not st.args:
            raise DSLSemanticError("set_flag expects a flag id", line=st.line, col=st.col)
        self._emit_eventflag(st.args[0], NumLit(value=1), "set_flag", st)

    def _construct_clear_flag(self, st: CallStmt) -> None:
        if not st.args:
            raise DSLSemanticError("clear_flag expects a flag id", line=st.line, col=st.col)
        self._emit_eventflag(st.args[0], NumLit(value=0), "clear_flag", st)

    def _construct_quest_flag(self, st: CallStmt) -> None:
        # quest_flag id, value
        if not st.args:
            raise DSLSemanticError("quest_flag expects (id, value)", line=st.line, col=st.col)
        value = st.args[1] if len(st.args) > 1 else NumLit(value=1)
        self._emit_eventflag(st.args[0], value, "quest_flag", st)

    def _construct_get_flag(self, st: CallStmt) -> None:
        # get_flag <id>, <var>   — read event flag <id> into variable <var>.
        # Also accepts the kwarg form  get_flag <id> into=<var>.
        if len(st.args) < 1:
            raise DSLSemanticError("get_flag expects (flag_id, target_var)",
                                   line=st.line, col=st.col)
        flag = self._need_int(st.args[0], "get_flag id")
        tgt = None
        if len(st.args) >= 2:
            tgt = st.args[1]
        elif "into" in st.kwargs:
            tgt = st.kwargs["into"]
        if not isinstance(tgt, VarRef):
            raise DSLSemanticError("get_flag needs a target variable: get_flag <id>, <var>",
                                   line=st.line, col=st.col)
        reg = self._resolve_var(tgt)
        # get_eventflag reads flag index from a register; stage it.
        idx = self._alloc_temp()
        self._emit(f"leti r{idx}, 0x{flag & 0xFFFFFFFF:X}")
        self._emit(f"get_eventflag r{idx}, r{reg}")
        self._free_temp()

    def _construct_give_item(self, st: CallStmt) -> None:
        # give_item data0, data1, data2  -> stage 3 regs + item_create
        if len(st.args) < 3:
            raise DSLSemanticError("give_item expects 3 item-data words",
                                   line=st.line, col=st.col)
        words = [self._need_int(a, "give_item data word") for a in st.args[:3]]
        base = self._temp_sp
        if base + 3 > _TEMP_MAX:
            raise DSLSemanticError("give_item: out of temp registers")
        for i, w in enumerate(words):
            self._emit(f"leti r{base + i}, 0x{w & 0xFFFFFFFF:X}")
        out = base + 3
        self._emit(f"item_create r{base}-r{base + 2}, r{out}")

    def _construct_spawn(self, st: CallStmt) -> None:
        # spawn npc=ID floor=N section=N x=.. y=.. z=.. dir=..
        kw = st.kwargs
        skin = self._kw_int(kw, "npc", st, default=8)
        floor = self._kw_int(kw, "floor", st, default=0)
        section = self._kw_int(kw, "section", st, default=0)
        x = self._kw_float(kw, "x", st, default=0.0)
        y = self._kw_float(kw, "y", st, default=0.0)
        z = self._kw_float(kw, "z", st, default=0.0)
        d = self._kw_float(kw, "dir", st, default=0.0)
        npc = NpcDef(name="spawned", skin=skin, floor=floor, section=section,
                     pos=(x, y, z), dir=d)
        self._gen_npc_spawn(npc)

    def _construct_wave(self, st: CallStmt) -> None:
        # wave npc=ID floor=N section=N count=K x= y= z=
        kw = st.kwargs
        count = self._kw_int(kw, "count", st, default=1)
        for _ in range(max(1, count)):
            self._construct_spawn(st)

    def _kw_int(self, kw, key, st, default=None) -> int:
        if key not in kw:
            if default is None:
                raise DSLSemanticError(f"missing required '{key}='", line=st.line, col=st.col)
            return default
        return self._need_int(kw[key], f"'{key}'")

    def _kw_float(self, kw, key, st, default=0.0) -> float:
        if key not in kw:
            return default
        v = kw[key]
        if isinstance(v, FloatLit):
            return v.value
        if isinstance(v, NumLit):
            return float(v.value)
        if isinstance(v, UnaryOp) and v.op == "-" and isinstance(v.operand, (NumLit, FloatLit)):
            return -float(v.operand.value)
        raise DSLSemanticError(f"'{key}' must be a number", line=st.line, col=st.col)

    def _construct_choose(self, st: CallStmt) -> None:  # pragma: no cover
        raise DSLSemanticError("choose must be used as a block, not a call",
                               line=st.line, col=st.col)

    def _gen_choose(self, st: ChooseStmt) -> None:
        # Display a list of options; read selection into a temp register;
        # dispatch with jmpi_eq chains.
        sel = self._alloc_temp()
        # `list` is an F_ARGS opcode: list <out_reg>, "opt1\nopt2\n..."
        opts_text = "\n".join(label for label, _ in st.options)
        self._emit(f"list r{sel}, {_asm_quote(opts_text)}")
        end = self._new_label("endchoose")
        for i, (_label, body) in enumerate(st.options):
            nxt = self._new_label("choice")
            self._emit(f"jmpi_ne r{sel}, 0x{i:X}, {nxt}")
            for s in body:
                self._gen_stmt(s)
            self._emit(f"jmp {end}")
            self.lines.append(f"{nxt}:")
        self.lines.append(f"{end}:")
        self._free_temp()

    def _construct_set_episode(self, st: CallStmt) -> None:
        ep = self._need_int(st.args[0], "episode")
        self._emit(f"set_episode 0x{ep & 0xFFFFFFFF:X}")

    def _construct_map_designate(self, st: CallStmt) -> None:
        if len(st.args) != 5:
            raise DSLSemanticError("map_designate expects 5 byte args", line=st.line, col=st.col)
        vals = [self._need_int(a, "map_designate byte") for a in st.args]
        self._emit("bb_map_designate " + ", ".join(f"0x{v & 0xFF:X}" for v in vals))


class _SyncMarker(Node):
    """Internal marker that lowers to a bare ``sync`` (used so a deferred
    thread body begins with the mandatory sync)."""


# Patch codegen to handle the sync marker.
_orig_gen_stmt = _CodeGen._gen_stmt


def _gen_stmt_with_sync(self, st):  # type: ignore[no-untyped-def]
    if isinstance(st, _SyncMarker):
        self._emit("sync")
        return
    return _orig_gen_stmt(self, st)


_CodeGen._gen_stmt = _gen_stmt_with_sync  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Small assembly-text helpers
# ---------------------------------------------------------------------------
def _asm_quote(text: str) -> str:
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


_LABEL_SAFE_RE = re.compile(r"[^A-Za-z0-9_]")


def _safe_label(name: str) -> str:
    s = _LABEL_SAFE_RE.sub("_", name)
    if not s or not (s[0].isalpha() or s[0] == "_"):
        s = "_" + s
    return s


# ---------------------------------------------------------------------------
# Public compile entry points
# ---------------------------------------------------------------------------
def compile_dsl_to_asm(text: str, *, version: str = q.DEFAULT_VERSION) -> str:
    """Compile DSL source to Layer-0 assembly text.

    Raises :class:`DSLSyntaxError` / :class:`DSLSemanticError` on bad input.
    """
    tokens = _Lexer(text).tokenize()
    prog = _Parser(tokens).parse_program()
    gen = _CodeGen(version=version)
    asm_text = gen.generate(prog)
    return asm_text


# Back-compat alias name from the brief.
def compile_dsl(text: str, *, version: str = q.DEFAULT_VERSION) -> str:
    """Compile DSL source to Layer-0 assembly text (alias of
    :func:`compile_dsl_to_asm`)."""
    return compile_dsl_to_asm(text, version=version)


def make_bb_template(
    name: str = "Untitled",
    *,
    quest_number: int = 0,
    episode: int = 1,
    max_players: int = 4,
) -> quest_bin.QuestBin:
    """Build a fresh, minimal valid BB ``.bin`` header template.

    The header is a zero-filled 0x122C-byte BB header with the quest name,
    number, episode, and player count filled in. Code + label table are
    spliced in by :func:`compile_dsl_to_bin`. Floor assignments and item
    masks are left zeroed (a server fills sensible defaults; this is enough
    to produce a *structurally valid* container, which is the gate).
    """
    co = quest_bin.CODE_OFFSET_BB
    header = bytearray(co)
    # code_offset/label_table_offset/size/marker are recomputed by
    # serialize_bin; we only need them parseable. Leave [0:0x10] for those.
    struct.pack_into("<H", header, 0x10, quest_number & 0xFFFF)
    header[0x14] = _episode_to_raw(episode) & 0xFF
    header[0x15] = max_players & 0xFF
    header[0x16] = 0  # joinable
    nbytes = name.encode("utf-16-le")[:0x40]
    header[0x18:0x18 + len(nbytes)] = nbytes
    return quest_bin.QuestBin(
        fmt=quest_bin.BIN_FORMAT_BB,
        code_offset=co,
        label_table_offset=co,  # recomputed
        size=0,
        unknown_marker=0xFFFFFFFF,
        header_raw=bytes(header),
        code=b"",
        label_offsets=[],
    )


def compile_dsl_to_bin(
    text: str,
    *,
    version: str = q.DEFAULT_VERSION,
    template: Optional[quest_bin.QuestBin] = None,
) -> bytes:
    """Compile DSL source all the way to decompressed ``.bin`` bytes.

    If ``template`` is None a fresh BB header is synthesised from the
    ``quest "..." { episode N }`` directives in the source.
    """
    tokens = _Lexer(text).tokenize()
    prog = _Parser(tokens).parse_program()
    gen = _CodeGen(version=version)
    asm_text = gen.generate(prog)
    if template is None:
        q_name = prog.quest.name if prog.quest else "Untitled"
        q_ep = prog.quest.episode if prog.quest else 1
        q_mp = prog.quest.max_players if prog.quest else 4
        template = make_bb_template(q_name, episode=q_ep, max_players=q_mp)
    return quest_asm.assemble_to_bin(asm_text, template, version=version)


# ---------------------------------------------------------------------------
# Lift: .bin -> best-effort DSL (recognise common construct shapes)
# ---------------------------------------------------------------------------
@dataclass
class LiftResult:
    """Result of a best-effort lift."""

    dsl_text: str
    recognised: List[str] = field(default_factory=list)
    fallbacks: List[str] = field(default_factory=list)  # what fell back to asm{}


def lift_bin(decompressed_bytes: bytes, *, version: str = q.DEFAULT_VERSION) -> str:
    """Lift a decompressed ``.bin`` to best-effort DSL text.

    The strategy mirrors :mod:`formats.mob_dsl`'s best-effort lift: we
    disassemble to Layer-0, then recognise the common compiled construct
    shapes (message/window_msg sequences, set_eventflag, item_create,
    thread starts, the leading sync) and re-emit them as DSL. Everything we
    do *not* recognise is preserved verbatim inside an ``asm { ... }`` block,
    so the lifted DSL always recompiles to equivalent bytecode.

    Full decompilation (reconstructing if/while/for from jump skeletons) is
    explicitly out of scope; structured control flow lifts to ``asm{}``.
    """
    res = lift_bin_detailed(decompressed_bytes, version=version)
    return res.dsl_text


def lift_bin_detailed(
    decompressed_bytes: bytes, *, version: str = q.DEFAULT_VERSION
) -> LiftResult:
    qb = quest_bin.parse_bin(decompressed_bytes)
    dis = quest_asm.disassemble_code(qb.code, qb.label_offsets, version=version)

    name = qb.name or "Lifted Quest"
    episode = _episode_to_dsl(qb.episode) if qb.episode is not None else 1

    recognised: List[str] = []
    fallbacks: List[str] = []

    # Split the literal assembly into logical lines (strip directive header).
    asm_lines = [ln.rstrip() for ln in dis.text().split("\n")]
    body_lines = [ln for ln in asm_lines if ln.strip() and not ln.strip().startswith(".")]

    # Group into label blocks. A block = a label_NN: line plus following
    # instructions until the next label. We keep BOTH the raw literal
    # instructions (for the verbatim asm{} fallback, which must reassemble
    # exactly) and a peephole-collapsed view (for construct recognition).
    blocks: List[Tuple[Optional[str], List[str], List[str]]] = []
    cur_label: Optional[str] = None
    cur: List[str] = []
    for ln in body_lines:
        s = ln.strip()
        if s.endswith(":") and " " not in s:
            if cur or cur_label is not None:
                blocks.append((cur_label, cur, _collapse_arg_pushes(cur)))
            cur_label = s[:-1]
            cur = []
        else:
            cur.append(s)
    if cur or cur_label is not None:
        blocks.append((cur_label, cur, _collapse_arg_pushes(cur)))

    # --- attempt a fully-structured lift -----------------------------------
    # The canonical compiled shape is: an entry/boot block (set_episode +
    # `thread LABEL` starts + ret) that launches one or more thread blocks
    # (each `sync`...`ret` with a linear body of known constructs). If the
    # boot block is pure-boot and every launched thread lifts cleanly, we
    # emit structured DSL. Otherwise we fall back to wrapping the WHOLE code
    # section in a single asm{} block (byte-exact, always reassembles).
    structured = _try_structured_lift(blocks, name, episode, recognised)
    if structured is not None:
        return LiftResult(dsl_text=structured, recognised=recognised, fallbacks=[])

    # Whole-program asm{} fallback.
    fallbacks.append("(whole program)")
    out: List[str] = []
    out.append(f'quest "{name}" {{')
    out.append(f"    episode {episode}")
    out.append("")
    out.append("    // lift could not fully structure this quest; the entire")
    out.append("    // code section is preserved verbatim and reassembles 1:1.")
    out.append("    asm {")
    for label, raw_instrs, _collapsed in blocks:
        if label:
            out.append(f'        "{label}:"')
        for ins in raw_instrs:
            out.append(f"        {_asm_quote(ins)}")
    out.append("    }")
    out.append("}")
    dsl_text = "\n".join(out) + "\n"
    return LiftResult(dsl_text=dsl_text, recognised=recognised, fallbacks=fallbacks)


def _try_structured_lift(blocks, name, episode, recognised) -> Optional[str]:
    """Attempt a fully-structured lift; return DSL text or None on any miss.

    Recognises the boot-block + named-thread shape the codegen emits.
    """
    if not blocks:
        return None
    boot_label, boot_raw, boot_collapsed = blocks[0]

    # The boot block must consist only of: optional set_episode, zero or more
    # `thread LABEL` launches, and a trailing ret.
    thread_targets: List[str] = []
    boot_episode_raw: Optional[int] = None
    for ins in boot_collapsed:
        toks = ins.split(None, 1)
        m = toks[0]
        rest = toks[1].strip() if len(toks) > 1 else ""
        if m == "ret":
            continue
        if m == "set_episode":
            boot_episode_raw = _parse_hex(rest)
            continue
        if m == "thread":
            thread_targets.append(rest)
            continue
        # any other boot instruction -> not the simple shape
        return None
    if not thread_targets:
        return None

    block_by_label = {lbl: (raw, col) for (lbl, raw, col) in blocks}

    # Each launched thread block must lift cleanly.
    lifted_threads: List[Tuple[str, List[str]]] = []
    used_labels = {boot_label}
    for ti, target in enumerate(thread_targets):
        entry = block_by_label.get(target)
        if entry is None:
            return None
        raw, collapsed = entry
        body = _lift_linear_body(collapsed, require_sync=True)
        if body is None:
            return None
        tname = "main" if ti == 0 else f"thread{ti}"
        lifted_threads.append((tname, body))
        used_labels.add(target)
        recognised.append(f"thread {tname}")

    # Every other block must be consumed too (no dangling code we'd drop).
    for lbl, _raw, _col in blocks:
        if lbl not in used_labels:
            return None

    if boot_episode_raw is not None:
        episode = _episode_to_dsl(boot_episode_raw)

    out: List[str] = []
    out.append(f'quest "{name}" {{')
    out.append(f"    episode {episode}")
    for tname, body in lifted_threads:
        out.append("")
        out.append(f"    thread {tname} {{")
        for b in body:
            out.append(f"        {b}")
        out.append("    }")
    out.append("}")
    return "\n".join(out) + "\n"


def _lift_linear_body(collapsed: List[str], *, require_sync: bool) -> Optional[List[str]]:
    """Lift a linear thread body (sync ... ret). Returns DSL lines or None."""
    work = list(collapsed)
    if require_sync:
        if not work or work[0].split()[0] != "sync":
            return None
        work = work[1:]
    lifted: List[str] = []
    for ins in work:
        mnem = ins.split()[0] if ins.split() else ""
        if mnem in ("ret", "sync"):
            continue
        dsl = _lift_instruction(ins)
        if dsl is None:
            return None
        if dsl is _LIFT_SKIP:
            continue
        lifted.append(dsl)
    return lifted


def _collapse_arg_pushes(instrs: List[str]) -> List[str]:
    """Peephole-collapse literal ``arg_push* ... <F_ARGS opcode>`` runs.

    The disassembler emits the *literal* form: each operand of an F_ARGS
    opcode appears as a preceding ``arg_pushX`` line, then the opcode appears
    bare. To recognise constructs we fold each run of arg-pushes plus the
    following F_ARGS opcode back into the structured single line
    (``message 0x1, "hi"``), which the rest of the lifter understands. Any
    ``arg_pushX`` not followed by an F_ARGS opcode is left untouched (it will
    just fail recognition and fall back to ``asm{}``).
    """
    out: List[str] = []
    pending: List[str] = []  # pending pushed-operand text fragments
    for ins in instrs:
        toks = ins.split(None, 1)
        mnem = toks[0]
        rest = toks[1].strip() if len(toks) > 1 else ""
        if mnem in ("arg_pushr", "arg_pushl", "arg_pushb", "arg_pushw",
                    "arg_pusha", "arg_pusho", "arg_pushs"):
            pending.append(rest)
            continue
        op = q.by_mnemonic(mnem)
        if op is not None and op.uses_arg_stack and pending:
            collapsed = mnem
            if pending:
                collapsed += " " + ", ".join(pending)
            out.append(collapsed)
            pending = []
            continue
        # a non-arg-push opcode with leftover pending pushes: flush the
        # pushes verbatim (unrecognised), then the opcode.
        if pending:
            out.extend(
                f"arg_push {p}" for p in pending
            )
            pending = []
        out.append(ins)
    out.extend(f"arg_push {p}" for p in pending)
    return out


#: Sentinel returned by :func:`_lift_instruction` for an instruction that is
#: recognised but produces no DSL line on its own (it was folded into the
#: preceding statement, e.g. ``message_end``). Distinguishes "skip" from the
#: ``None`` that means "not recognised".
_LIFT_SKIP = object()


def _lift_instruction(ins: str) -> Optional[str]:
    """Lift a single Layer-0 instruction to a DSL statement.

    Returns the DSL line (str), :data:`_LIFT_SKIP` for a folded/no-op line,
    or ``None`` if the instruction is not recognised (forcing the whole block
    to the ``asm{}`` fallback).
    """
    toks = ins.split(None, 1)
    mnem = toks[0]
    rest = toks[1] if len(toks) > 1 else ""

    if mnem == "set_eventflag":
        # set_eventflag 0xNN, 0xVV  -> set_flag / clear_flag / quest_flag
        parts = [p.strip() for p in rest.split(",")]
        if len(parts) == 2:
            fid = _parse_hex(parts[0])
            val = _parse_hex(parts[1])
            if fid is not None and val is not None:
                if val == 1:
                    return f"set_flag {fid}"
                if val == 0:
                    return f"clear_flag {fid}"
                return f"quest_flag {fid}, {val}"
    if mnem == "window_msg":
        return f"window_msg {rest}"
    if mnem == "add_msg":
        return f"add_msg {rest}"
    if mnem == "message":
        # message 0xID, "text"
        parts = rest.split(",", 1)
        if len(parts) == 2:
            mid = _parse_hex(parts[0].strip())
            if mid is not None:
                return f"message {mid}, {parts[1].strip()}"
    if mnem in ("window_msg_end", "message_end"):
        return _LIFT_SKIP  # folded into the message/window_msg statement above
    if mnem == "set_episode":
        ep = _parse_hex(rest.strip())
        if ep is not None:
            return f"set_episode {ep}"
    if mnem == "bb_map_designate":
        parts = [p.strip() for p in rest.split(",")]
        vals = [_parse_hex(p) for p in parts]
        if len(vals) == 5 and all(v is not None for v in vals):
            return "map_designate " + ", ".join(str(v) for v in vals)
    # not recognised
    return None


def _parse_hex(tok: str) -> Optional[int]:
    tok = tok.strip()
    try:
        return int(tok, 0)
    except ValueError:
        return None
