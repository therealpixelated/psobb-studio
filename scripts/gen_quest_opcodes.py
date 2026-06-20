#!/usr/bin/env python3
"""Regenerate ``formats/_quest_opcode_table.py`` — the committed, self-contained
PSOBB quest-VM opcode table.

This script is a BUILD-TIME tool. It reads the two MIT reference sources and
emits a flat Python data module that ``formats.quest_opcodes`` wraps. The
emitted module embeds the data literally so the runtime never touches
``_reference/`` (which is gitignored on clones).

Sources (both MIT-licensed; we port the DATA, not the code):

  * PORT SOURCE — phantasmal-world ``opcodes.yml``:
        _reference/phantasmal-world/psolib/srcGeneration/asm/opcodes.yml
    ~425 named opcodes as language-agnostic data: code / mnemonic / doc /
    typed params with per-param register read-write + stack push/pop.

  * CROSS-CHECK — newserv ``QuestScript.cc`` ``opcode_defs[]``:
        _reference/newserv-sparse/src/QuestScript.cc
    Carries BOTH newserv and qedit mnemonic names, per-version flags, and the
    F_ARGS / F_PUSH_ARG / F_CLEAR_ARGS / F_TERMINATOR flags. For the Blue Burst
    (BB_V4) live target we PREFER newserv where the two disagree, and log each
    discrepancy as a ``# DISCREPANCY:`` comment inside the emitted table.

Regen:
    python scripts/gen_quest_opcodes.py
    # then: make lint && pytest tests/test_quest_opcodes.py -q

The emitted file is deterministic (sorted by full code) so re-running it
produces a clean diff.

We do NOT take a PyYAML dependency (the app pins its deps and must stay light).
The opcodes.yml file is a small, consistent subset of YAML, so a tailored
chunk-based parser handles it. The parser splits the document into one chunk
per ``- code:`` item first, then parses each chunk independently — this
guarantees forward progress and termination.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
YML = ROOT / "_reference/phantasmal-world/psolib/srcGeneration/asm/opcodes.yml"
NEWSERV = ROOT / "_reference/newserv-sparse/src/QuestScript.cc"
OUT = ROOT / "formats/_quest_opcode_table.py"


# ---------------------------------------------------------------------------
# Full-code encoding for the extended pages.
#   stored bytes  F8 NN  ->  full code 0x1NN
#   stored bytes  F9 NN  ->  full code 0x2NN
# The references write these as the raw 16-bit value 0xF8NN / 0xF9NN; convert.
# ---------------------------------------------------------------------------
def raw_to_full(raw: int) -> int:
    if (raw & 0xFF00) == 0xF800:
        return 0x100 | (raw & 0xFF)
    if (raw & 0xFF00) == 0xF900:
        return 0x200 | (raw & 0xFF)
    return raw & 0xFF


# ---------------------------------------------------------------------------
# Param-type canonicalisation. Both taxonomies are collapsed onto one enum-name
# set (see formats/quest_opcodes.ParamType).
# ---------------------------------------------------------------------------
# newserv Arg::Type token -> (canonical ParamType name, default reads, default writes)
NEWSERV_TYPE = {
    "LABEL16": ("SCRIPT16", True, False),
    "LABEL16_SET": ("SCRIPT16_SET", True, False),
    "LABEL32": ("SCRIPT32", True, False),
    "SCRIPT16": ("SCRIPT16", True, False),
    "SCRIPT16_SET": ("SCRIPT16_SET", True, False),
    "SCRIPT32": ("SCRIPT32", True, False),
    "DATA16": ("DATA16", True, False),
    "CSTRING_LABEL16": ("SCRIPT16", True, False),
    "R_REG": ("R_REG", True, False),
    "W_REG": ("W_REG", False, True),
    "R_REG_SET": ("R_REG_SET", True, False),
    "R_REG_SET_FIXED": ("R_REG_SET_FIXED", True, False),
    "W_REG_SET_FIXED": ("W_REG_SET_FIXED", False, True),
    "R_REG32": ("R_REG32", True, False),
    "W_REG32": ("W_REG32", False, True),
    "R_REG32_SET_FIXED": ("R_REG32_SET_FIXED", True, False),
    "W_REG32_SET_FIXED": ("W_REG32_SET_FIXED", False, True),
    "I8": ("U8", False, False),
    "I16": ("U16", False, False),
    "I32": ("I32", False, False),
    "FLOAT32": ("FLOAT32", False, False),
    "CSTRING": ("CSTRING", False, False),
    # named-arg shortcuts that are really I32:
    "CLIENT_ID": ("I32", False, False),
    "ITEM_ID": ("I32", False, False),
    "FLOOR": ("I32", False, False),
    "VECTOR4F_LIST": ("VECTOR4F_LIST", True, False),
}

# YAML "type:" token -> canonical ParamType name (register sub-type folded in
# separately). YAML reg params carry their own read/write via the nested
# `registers:` block; non-reg params have fixed read/write semantics.
YAML_SCALAR_TYPE = {
    "int": "I32",
    "float": "FLOAT32",
    "string": "CSTRING",
    "short": "U16",
    "byte": "U8",
    "ilabel": "SCRIPT16",      # instruction/function label (16-bit table index)
    "ilabel_var": "SCRIPT16_SET",
    "dlabel": "DATA16",        # data label
    "slabel": "SCRIPT16",      # string label
    "label": "SCRIPT16",       # generic label (used by leto / arg_pusho)
    "pointer": "R_REG",        # pointer-typed register operand
    "reg_var": "R_REG_SET",    # variadic register list (switch reg etc.)
}


# ---------------------------------------------------------------------------
# YAML parser — chunk-based for guaranteed termination.
# ---------------------------------------------------------------------------
def _indent(s: str) -> int:
    return len(s) - len(s.lstrip(" "))


def _split_chunks(lines: list[str]) -> list[list[str]]:
    """Split the opcode list into per-item line chunks (each starts at a
    ``- code:`` line)."""
    chunks: list[list[str]] = []
    cur: list[str] | None = None
    for ln in lines:
        if re.match(r"^\s*- code:\s*0x[0-9a-fA-F]+\s*$", ln):
            if cur is not None:
                chunks.append(cur)
            cur = [ln]
        elif cur is not None:
            cur.append(ln)
    if cur is not None:
        chunks.append(cur)
    return chunks


def _parse_block_scalar(chunk: list[str], start: int, key_indent: int, style: str) -> tuple[int, str]:
    """Collect a YAML block scalar (``|``/``|-``/``>``/``>-``) starting after
    its key line. Returns (next_index, text)."""
    collected: list[str] = []
    base: int | None = None
    j = start
    while j < len(chunk):
        ln = chunk[j]
        if not ln.strip():
            collected.append("")
            j += 1
            continue
        ind = _indent(ln)
        if ind <= key_indent:
            break
        if base is None:
            base = ind
        collected.append(ln[base:] if len(ln) >= base else ln.strip())
        j += 1
    while collected and collected[-1] == "":
        collected.pop()
    if style in (">", ">-"):
        text = " ".join(x for x in collected if x != "").strip()
    else:
        text = "\n".join(collected).strip()
    return j, text


def _parse_params(chunk: list[str], start: int, params_indent: int) -> list[dict]:
    """Parse a ``params:`` block (list of ``- type:`` items, each optionally
    with nested ``registers:``/``read:``/``write:``)."""
    params: list[dict] = []
    j = start
    n = len(chunk)
    while j < n:
        ln = chunk[j]
        if not ln.strip():
            j += 1
            continue
        ind = _indent(ln)
        if ind <= params_indent:
            break
        m = re.match(r"^\s*- type:\s*(\S+)\s*$", ln)
        if not m:
            j += 1
            continue
        item_indent = ind
        ptype = m.group(1)
        regs: list[dict] = []
        flat_read = False
        flat_write = False
        j += 1
        # consume children of this param item
        while j < n:
            ln2 = chunk[j]
            if not ln2.strip():
                j += 1
                continue
            ind2 = _indent(ln2)
            if ind2 <= item_indent:
                break
            s2 = ln2.strip()
            if s2.startswith("registers:"):
                j += 1
                # parse nested register sub-items
                while j < n:
                    ln3 = chunk[j]
                    if not ln3.strip():
                        j += 1
                        continue
                    ind3 = _indent(ln3)
                    if ind3 <= ind2:
                        break
                    m3 = re.match(r"^\s*- type:\s*(\S+)\s*$", ln3)
                    if not m3:
                        j += 1
                        continue
                    reg_indent = ind3
                    sub = {"type": m3.group(1), "read": False, "write": False}
                    j += 1
                    while j < n:
                        ln4 = chunk[j]
                        if not ln4.strip():
                            j += 1
                            continue
                        ind4 = _indent(ln4)
                        if ind4 <= reg_indent:
                            break
                        s4 = ln4.strip()
                        if s4.startswith("read:"):
                            sub["read"] = s4.split(":", 1)[1].strip() == "true"
                        elif s4.startswith("write:"):
                            sub["write"] = s4.split(":", 1)[1].strip() == "true"
                        j += 1
                    regs.append(sub)
            elif s2.startswith("read:"):
                flat_read = s2.split(":", 1)[1].strip() == "true"
                j += 1
            elif s2.startswith("write:"):
                flat_write = s2.split(":", 1)[1].strip() == "true"
                j += 1
            else:
                j += 1
        params.append({"type": ptype, "registers": regs, "read": flat_read, "write": flat_write})
    return params


def _parse_chunk(chunk: list[str]) -> tuple[int, dict]:
    m = re.match(r"^\s*- code:\s*(0x[0-9a-fA-F]+)\s*$", chunk[0])
    raw = int(m.group(1), 16)
    full = raw_to_full(raw)
    entry: dict = {"mnemonic": None, "doc": None, "params": [], "stack": None}
    j = 1
    n = len(chunk)
    while j < n:
        ln = chunk[j]
        if not ln.strip():
            j += 1
            continue
        key_indent = _indent(ln)
        s = ln.strip()
        if s.startswith("mnemonic:"):
            entry["mnemonic"] = s.split(":", 1)[1].strip() or None
            j += 1
        elif s.startswith("stack:"):
            entry["stack"] = s.split(":", 1)[1].strip() or None
            j += 1
        elif s.startswith("doc:"):
            after = s.split(":", 1)[1].strip()
            if after and after not in ("|", "|-", ">", ">-"):
                entry["doc"] = after
                j += 1
            else:
                j, entry["doc"] = _parse_block_scalar(chunk, j + 1, key_indent, after)
        elif s.startswith("params:"):
            rest = s.split(":", 1)[1].strip()
            if rest == "[]":
                entry["params"] = []
                j += 1
            else:
                entry["params"] = _parse_params(chunk, j + 1, key_indent)
                # _parse_params doesn't return index; skip past consumed lines
                # by scanning to next top-level key or end.
                j = _skip_block(chunk, j + 1, key_indent)
        else:
            j += 1
    return full, entry


def _skip_block(chunk: list[str], start: int, key_indent: int) -> int:
    j = start
    while j < len(chunk):
        ln = chunk[j]
        if ln.strip() and _indent(ln) <= key_indent:
            break
        j += 1
    return j


def parse_yaml() -> dict[int, dict]:
    lines = YML.read_text(encoding="utf-8").splitlines()
    out: dict[int, dict] = {}
    for chunk in _split_chunks(lines):
        full, entry = _parse_chunk(chunk)
        out[full] = entry
    return out


def yaml_params_to_canonical(params: list[dict]) -> list[dict]:
    """Map YAML param dicts onto the canonical Param shape."""
    result = []
    for p in params:
        t = p["type"]
        if t == "reg":
            read = any(r["read"] for r in p["registers"]) or p["read"]
            write = any(r["write"] for r in p["registers"]) or p["write"]
            canon = "W_REG" if write else "R_REG"
            result.append({"type": canon, "read": read, "write": bool(write)})
        else:
            canon = YAML_SCALAR_TYPE.get(t, t.upper())
            read = p["read"] or (t in ("reg_var", "ilabel_var"))
            result.append({"type": canon, "read": read, "write": p["write"]})
    return result


# ---------------------------------------------------------------------------
# newserv parser — one opcode per line, regular C-array syntax.
# ---------------------------------------------------------------------------
ROW_RE = re.compile(
    r"^\s*\{\s*(0x[0-9A-Fa-f]+)\s*,\s*"            # opcode
    r'("(?:[^"\\]|\\.)*"|nullptr)\s*,\s*'            # name
    r'("(?:[^"\\]|\\.)*"|nullptr)\s*,\s*'            # qedit name
    r"\{(.*?)\}\s*,\s*"                              # args { ... }
    r"([A-Za-z0-9_ |]+)"                              # flags expression
    r"\}\s*,?\s*$"
)

# Version-range macros -> the per-version flag-name set.
VERSION_MACROS = {
    "F_V0_V2": ["DC_NTE", "DC_112000", "DC_V1", "DC_V2", "PC_NTE", "PC_V2", "GC_NTE"],
    "F_V0_V4": ["DC_NTE", "DC_112000", "DC_V1", "DC_V2", "PC_NTE", "PC_V2", "GC_NTE",
                "GC_V3", "GC_EP3TE", "GC_EP3", "XB_V3", "BB_V4"],
    "F_V05_V2": ["DC_112000", "DC_V1", "DC_V2", "PC_NTE", "PC_V2", "GC_NTE"],
    "F_V05_V4": ["DC_112000", "DC_V1", "DC_V2", "PC_NTE", "PC_V2", "GC_NTE",
                 "GC_V3", "GC_EP3TE", "GC_EP3", "XB_V3", "BB_V4"],
    "F_V1_V2": ["DC_V1", "DC_V2", "PC_NTE", "PC_V2", "GC_NTE"],
    "F_V1_V4": ["DC_V1", "DC_V2", "PC_NTE", "PC_V2", "GC_NTE",
                "GC_V3", "GC_EP3TE", "GC_EP3", "XB_V3", "BB_V4"],
    "F_V2": ["DC_V2", "PC_NTE", "PC_V2", "GC_NTE"],
    "F_V2_V3": ["DC_V2", "PC_NTE", "PC_V2", "GC_NTE", "GC_V3", "GC_EP3TE", "GC_EP3", "XB_V3"],
    "F_V2_V4": ["DC_V2", "PC_NTE", "PC_V2", "GC_NTE",
                "GC_V3", "GC_EP3TE", "GC_EP3", "XB_V3", "BB_V4"],
    "F_V3": ["GC_V3", "GC_EP3TE", "GC_EP3", "XB_V3"],
    "F_V3_V4": ["GC_V3", "GC_EP3TE", "GC_EP3", "XB_V3", "BB_V4"],
    "F_V4": ["BB_V4"],
}
SINGLE_VERSION = {
    "F_DC_NTE": "DC_NTE", "F_DC_112000": "DC_112000", "F_DC_V1": "DC_V1",
    "F_DC_V2": "DC_V2", "F_PC_NTE": "PC_NTE", "F_PC_V2": "PC_V2",
    "F_GC_NTE": "GC_NTE", "F_GC_V3": "GC_V3", "F_GC_EP3TE": "GC_EP3TE",
    "F_GC_EP3": "GC_EP3", "F_XB_V3": "XB_V3", "F_BB_V4": "BB_V4",
}
SEMANTIC_FLAGS = {"F_PUSH_ARG", "F_CLEAR_ARGS", "F_ARGS", "F_TERMINATOR"}


def split_args(arg_blob: str) -> list[str]:
    """Split a newserv args list, honouring nested {N, X} braces."""
    args: list[str] = []
    depth = 0
    cur = ""
    for ch in arg_blob:
        if ch == "{":
            depth += 1
            cur += ch
        elif ch == "}":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            if cur.strip():
                args.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        args.append(cur.strip())
    return args


def parse_newserv_arg(tok: str) -> dict:
    tok = tok.strip()
    count = 0
    if tok.startswith("{"):
        inner = split_args(tok[1:-1])
        type_tok = inner[0].strip()
        if len(inner) > 1:
            try:
                count = int(inner[1].strip(), 0)
            except ValueError:
                count = 0
    else:
        type_tok = tok
    canon, read, write = NEWSERV_TYPE.get(type_tok, (type_tok, False, False))
    return {"type": canon, "read": read, "write": write, "count": count, "raw": type_tok}


def parse_newserv() -> list[dict]:
    text = NEWSERV.read_text(encoding="utf-8")
    rows: list[dict] = []
    in_table = False
    for line in text.splitlines():
        if "opcode_defs[] = {" in line:
            in_table = True
            continue
        if in_table and line.strip() == "};":
            break
        if not in_table:
            continue
        m = ROW_RE.match(line)
        if not m:
            continue
        raw = int(m.group(1), 16)
        name = None if m.group(2) == "nullptr" else m.group(2)[1:-1]
        qedit = None if m.group(3) == "nullptr" else m.group(3)[1:-1]
        args = [parse_newserv_arg(a) for a in split_args(m.group(4))]
        flag_tokens = [t.strip() for t in m.group(5).split("|") if t.strip()]
        versions: set[str] = set()
        semantic: set[str] = set()
        for ft in flag_tokens:
            if ft in VERSION_MACROS:
                versions.update(VERSION_MACROS[ft])
            elif ft in SINGLE_VERSION:
                versions.add(SINGLE_VERSION[ft])
            elif ft in SEMANTIC_FLAGS:
                semantic.add(ft)
        rows.append(
            {
                "raw": raw,
                "full": raw_to_full(raw),
                "name": name,
                "qedit": qedit,
                "args": args,
                "versions": sorted(versions),
                "semantic": sorted(semantic),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Merge — newserv wins for BB; YAML supplies doc + per-param read/write detail.
# ---------------------------------------------------------------------------
def build_table() -> tuple[list[dict], list[str]]:
    yaml_ops = parse_yaml()
    newserv_rows = parse_newserv()

    by_full_bb: dict[int, dict] = {}
    by_full_any: dict[int, dict] = {}
    for r in newserv_rows:
        by_full_any.setdefault(r["full"], r)
        if "BB_V4" in r["versions"]:
            prev = by_full_bb.get(r["full"])
            if prev is None or len(r["versions"]) >= len(prev["versions"]):
                by_full_bb[r["full"]] = r

    all_codes = sorted(set(yaml_ops) | set(by_full_any))
    discrepancies: list[str] = []
    table: list[dict] = []

    for code in all_codes:
        y = yaml_ops.get(code)
        ns = by_full_bb.get(code) or by_full_any.get(code)
        bb = code in by_full_bb

        mnemonic = (ns["name"] if ns and ns["name"] else None) or (y["mnemonic"] if y else None)
        qedit = ns["qedit"] if ns else None
        doc = y["doc"] if y else None

        if ns:
            params = [
                {"type": a["type"], "read": a["read"], "write": a["write"], "count": a["count"]}
                for a in ns["args"]
            ]
        elif y:
            params = [
                {"type": p["type"], "read": p["read"], "write": p["write"], "count": 0}
                for p in yaml_params_to_canonical(y["params"])
            ]
        else:
            params = []

        versions = ns["versions"] if ns else (["BB_V4"] if y else [])
        semantic = list(ns["semantic"]) if ns else []

        if y and not ns:
            if y["stack"] == "pop":
                semantic.append("F_ARGS")
            elif y["stack"] == "push":
                semantic.append("F_PUSH_ARG")
        semantic = sorted(set(semantic))

        # ---- cross-check & log discrepancies ----
        # Only a true conflict if the YAML mnemonic matches NEITHER newserv's
        # primary name NOR its qedit alias (the YAML often just uses the qedit
        # name, which is no conflict — we already carry it as qedit_alias).
        if y and ns and ns["name"] and y["mnemonic"]:
            ny, nn = y["mnemonic"], ns["name"]
            qa = ns["qedit"]
            matches_alias = qa is not None and (ny == qa or _alias_ok(ny, qa))
            if ny != nn and not _alias_ok(ny, nn) and not matches_alias:
                discrepancies.append(
                    f"0x{code:03X}: mnemonic yml={ny!r} newserv={nn!r}"
                    + (f" (qedit {qa!r})" if qa else "")
                    + " -> using newserv (BB)"
                )
        if y and ns:
            yc = yaml_params_to_canonical(y["params"])
            if len(yc) != len(ns["args"]):
                discrepancies.append(
                    f"0x{code:03X} ({mnemonic}): arg count yml={len(yc)} "
                    f"newserv={len(ns['args'])} -> using newserv (BB)"
                )

        table.append(
            {
                "code": code,
                "mnemonic": mnemonic,
                "qedit_alias": qedit,
                "doc": doc,
                "params": params,
                "versions": versions,
                "semantic": semantic,
                "bb": bb,
                "source": "newserv" if ns else "yaml",
            }
        )

    return table, discrepancies


def _alias_ok(a: str, b: str) -> bool:
    """Treat trivial spelling differences as non-discrepancies (trailing
    digits / version suffixes / shared long prefix)."""
    na = re.sub(r"[_0-9]+$", "", a)
    nb = re.sub(r"[_0-9]+$", "", b)
    if na == nb:
        return True
    common = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        common += 1
    return common >= max(4, min(len(a), len(b)) - 3)


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------
def emit(table: list[dict], discrepancies: list[str]) -> str:
    L: list[str] = []
    L.append('"""PSOBB quest-VM opcode table — GENERATED, do not edit by hand.')
    L.append("")
    L.append("Regenerate with::")
    L.append("")
    L.append("    python scripts/gen_quest_opcodes.py")
    L.append("")
    L.append("Ported (DATA only) from the MIT-licensed phantasmal-world")
    L.append("``opcodes.yml`` and cross-checked against newserv's")
    L.append("``QuestScript.cc`` ``opcode_defs[]``. For Blue Burst (BB_V4) the")
    L.append("newserv definition wins where the two disagree; see the")
    L.append("DISCREPANCY log below. This module is self-contained: it embeds")
    L.append("the data literally and never reads _reference/ at runtime.")
    L.append("")
    L.append("Each record is a tuple::")
    L.append("")
    L.append("    (code, mnemonic, qedit_alias, doc, params, versions, semantic_flags)")
    L.append("")
    L.append("where ``params`` is a tuple of ``(type, read, write, count)`` and")
    L.append("``versions`` / ``semantic_flags`` are tuples of string flag names.")
    L.append('"""')
    L.append("from __future__ import annotations")
    L.append("")
    L.append(f"# Opcodes: {len(table)} total; "
             f"{sum(1 for t in table if t['bb'])} valid on Blue Burst (BB_V4).")
    L.append("")
    L.append("# ---------------------------------------------------------------------------")
    L.append("# DISCREPANCY LOG (yml vs newserv). Resolved in favour of newserv for BB.")
    L.append("# ---------------------------------------------------------------------------")
    if discrepancies:
        for d in discrepancies:
            L.append(f"# DISCREPANCY: {d}")
    else:
        L.append("# (none — mnemonics and arg counts agree within alias tolerance)")
    L.append("")
    L.append("OPCODE_RECORDS: tuple = (")
    for t in table:
        params_repr = "(" + ", ".join(
            f"({_s(p['type'])}, {p['read']}, {p['write']}, {p['count']})" for p in t["params"]
        ) + ("," if len(t["params"]) == 1 else "") + ")"
        versions_repr = "(" + ", ".join(_s(v) for v in t["versions"]) + (
            "," if len(t["versions"]) == 1 else "") + ")"
        sem_repr = "(" + ", ".join(_s(s) for s in t["semantic"]) + (
            "," if len(t["semantic"]) == 1 else "") + ")"
        L.append(
            f"    (0x{t['code']:03X}, {_s(t['mnemonic'])}, {_s(t['qedit_alias'])}, "
            f"{_s(t['doc'])}, {params_repr}, {versions_repr}, {sem_repr}),"
        )
    L.append(")")
    L.append("")
    return "\n".join(L)


def _s(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, str):
        return repr(v)
    return str(v)


def main() -> int:
    if not YML.exists() or not NEWSERV.exists():
        print(f"ERROR: reference sources not found under {ROOT / '_reference'}")
        return 2
    yaml_ops = parse_yaml()
    newserv_rows = parse_newserv()
    newserv_codes = {r["full"] for r in newserv_rows}
    table, discrepancies = build_table()
    OUT.write_text(emit(table, discrepancies), encoding="utf-8")
    bb = sum(1 for t in table if t["bb"])
    both = len(set(yaml_ops) & newserv_codes)
    yaml_only = len(set(yaml_ops) - newserv_codes)
    newserv_only = len(newserv_codes - set(yaml_ops))
    print(f"Wrote {OUT.relative_to(ROOT)}: {len(table)} opcodes "
          f"({bb} BB), {len(discrepancies)} discrepancies logged.")
    print(f"  yaml records:    {len(yaml_ops)}")
    print(f"  newserv records: {len(newserv_codes)} distinct codes "
          f"({len(newserv_rows)} rows incl. version splits)")
    print(f"  cross-checked (in BOTH): {both}; yaml-only: {yaml_only}; "
          f"newserv-only: {newserv_only}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
