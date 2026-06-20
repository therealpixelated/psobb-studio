"""EVAL_HIDE audit — count nodes with HIDE flag and how many have geometry.

Investigation B from AGENT_MODEL_DEEP_DEBUG_REPORT.md.

The hypothesis is that PSOBB BB stamps EVAL_HIDE on legitimately-
visible mesh nodes — the "skull" sub-mesh that the user reports as
missing on the De Rol Le family is plausibly one of these. This script
prints:

  - Total node count
  - Count of nodes with mesh_ptr != 0 (i.e. has a Mesh struct)
  - Count of those nodes with EVAL_HIDE set (would be SKIPPED today)
  - Count with EVAL_SHAPE_SKIP set
  - Per-flag histogram across all mesh-bearing nodes

Run::

    python scripts/dump_eval_hide_audit.py bm_boss2_de_rol_le_a.bml#boss2_b_derorure_body.nj

For comparison, also runs on a non-boss model (Hunter body, dragon)
to see if HIDE is rare (in which case ignoring it for boss2 is safe)
or common across the dataset (in which case ignoring would regress
non-boss models).
"""
from __future__ import annotations
import os

import argparse
import struct
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from formats.bml import parse_bml, _prs_decompress  # noqa: E402
from formats.iff import parse_iff  # noqa: E402

_MESH_TREE_NODE_FMT = "<II3f3i3fII"
_MESH_TREE_NODE_SIZE = struct.calcsize(_MESH_TREE_NODE_FMT)

EVAL_FLAGS = {
    0x01: "UNIT_POS",
    0x02: "UNIT_ANG",
    0x04: "UNIT_SCL",
    0x08: "HIDE",
    0x10: "BREAK",
    0x20: "ZXY_ANG",
    0x40: "SKIP",
    0x80: "SHAPE_SKIP",
}


def load_inner_bytes(path: str) -> bytes:
    install_data = Path(os.path.expanduser("~/PSOBB.IO/data"))
    if "#" in path:
        bml_name, inner = path.split("#", 1)
        bml_path = install_data / bml_name
        if not bml_path.exists():
            sys.exit(f"BML not found: {bml_path}")
        blob = bml_path.read_bytes()
        entries = parse_bml(blob)
        target = next((e for e in entries if e.name == inner), None)
        if target is None:
            sys.exit(f"inner {inner!r} not in {bml_name}")
        raw = bytes(blob[target.offset:target.offset + target.size_compressed])
        return _prs_decompress(raw, timeout=20.0)
    p = install_data / path
    if not p.exists():
        sys.exit(f"file not found: {p}")
    return p.read_bytes()


def get_njcm_body(buf: bytes) -> bytes:
    chunks = parse_iff(buf)
    for c in chunks:
        if c.type == "NJCM":
            return c.data
    sys.exit("no NJCM chunk in input")


def walk_collect(body: bytes):
    n = len(body)
    visited: set = set()
    stack = [0]
    nodes = []
    while stack:
        off = stack.pop()
        if off in visited or off + _MESH_TREE_NODE_SIZE > n:
            continue
        visited.add(off)
        f = struct.unpack_from(_MESH_TREE_NODE_FMT, body, off)
        eval_flags = f[0]
        mesh_ptr = f[1]
        child = f[11]
        nxt = f[12]
        nodes.append({"off": off, "ef": eval_flags, "mesh_ptr": mesh_ptr})
        if nxt:
            stack.append(nxt)
        if child:
            stack.append(child)
    return nodes


def fmt_flags(ef: int) -> str:
    names = [name for bit, name in EVAL_FLAGS.items() if ef & bit]
    return f"0x{ef:02x}={'|'.join(names) if names else 'NONE'}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="`<bml>#<inner>` path(s) under PSOBB.IO/data")
    args = ap.parse_args()

    for path in args.paths:
        print(f"=== {path} ===")
        try:
            buf = load_inner_bytes(path)
            body = get_njcm_body(buf)
            nodes = walk_collect(body)
        except SystemExit as e:
            print(f"  ERROR: {e}")
            continue

        total = len(nodes)
        with_mesh = [n for n in nodes if n["mesh_ptr"] != 0]
        hide = [n for n in nodes if (n["ef"] & 0x08) and n["mesh_ptr"] != 0]
        shape_skip = [n for n in nodes if (n["ef"] & 0x80) and n["mesh_ptr"] != 0]
        print(f"  total nodes: {total}")
        print(f"  with mesh_ptr: {len(with_mesh)}")
        print(f"  with HIDE flag (mesh_ptr != 0): {len(hide)}")
        print(f"  with SHAPE_SKIP flag (mesh_ptr != 0): {len(shape_skip)}")

        # Histogram of eval-flag values across mesh-bearing nodes
        cnt = Counter(n["ef"] for n in with_mesh)
        for ef, count in sorted(cnt.items()):
            print(f"    {count:>4d}x  {fmt_flags(ef)}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
