"""Per-bone audit: how many strips emit per bone, and which bones drop.

Investigation C from AGENT_MODEL_DEEP_DEBUG_REPORT.md.

Walks the De Rol Le body using the same logic as ``formats.xj`` but
reports, for each mesh-bearing node, how many strips it emitted vs
how many strips its plist tried to emit. This catches the case where
strip references resolve to MISSING vertex slots and the entire
strip is silently dropped (the "skull" disappearance).
"""
from __future__ import annotations
import os

import argparse
import struct
import sys
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from formats.bml import parse_bml, _prs_decompress  # noqa: E402
from formats.iff import parse_iff  # noqa: E402
from formats import xj as _xj  # noqa: E402

EVAL_FLAGS = {
    0x01: "UNIT_POS", 0x02: "UNIT_ANG", 0x04: "UNIT_SCL",
    0x08: "HIDE", 0x10: "BREAK", 0x20: "ZXY_ANG",
    0x40: "SKIP", 0x80: "SHAPE_SKIP",
}


def fmt_flags(ef: int) -> str:
    names = [n for bit, n in EVAL_FLAGS.items() if ef & bit]
    return f"0x{ef:02x}={'|'.join(names) if names else 'NONE'}"


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


def per_node_audit(body: bytes) -> List[Dict]:
    """Replay the same DFS as `_walk_tree` but count strips per node."""
    n = len(body)
    visited: set = set()
    stack = [(0, _xj._mat4_identity())]
    out = []
    while stack:
        off, parent_world = stack.pop()
        if off in visited or off + _xj._MESH_TREE_NODE_SIZE > n:
            continue
        visited.add(off)
        full = _xj._read_mesh_tree_node_full(body, off)
        if full is None:
            continue
        ef, mesh_ptr, pos, rot_bams, scale, child_ptr, next_ptr = full
        if ef & _xj.EVAL_SKIP:
            local_M = _xj._mat4_identity()
        else:
            lpos = (0.0, 0.0, 0.0) if (ef & _xj.EVAL_UNIT_POS) else pos
            if ef & _xj.EVAL_UNIT_ANG:
                lrot = (0.0, 0.0, 0.0)
            else:
                lrot = tuple(r * _xj._BAMS_TO_RAD for r in rot_bams)
            lscale = (1.0, 1.0, 1.0) if (ef & _xj.EVAL_UNIT_SCL) else scale
            zxy = bool(ef & _xj.EVAL_ZXY_ANG)
            local_M = _xj._mat4_compose_trs(lpos, lrot, lscale, zxy_order=zxy)
        world_M = _xj._mat4_mul(parent_world, local_M)
        out.append({
            "off": off, "ef": ef, "mesh_ptr": mesh_ptr,
            "world_pos": [world_M[3], world_M[7], world_M[11]],
            "world_M": world_M,
        })
        if next_ptr and next_ptr not in visited:
            stack.append((next_ptr, parent_world))
        if child_ptr and child_ptr not in visited:
            stack.append((child_ptr, world_M))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()

    buf = load_inner_bytes(args.path)
    body = get_njcm_body(buf)
    nodes = per_node_audit(body)

    # Build state with all vertex chunks first (like the real parser).
    state = _xj._NinjaChunkState()
    for node in nodes:
        if not node["mesh_ptr"] or node["mesh_ptr"] + _xj._NJ_MESH_SIZE > len(body):
            continue
        mesh = _xj._read_nj_mesh(body, node["mesh_ptr"])
        if mesh is None:
            continue
        vlist_off, _plist_off, _bbox = mesh
        if vlist_off:
            state.current_world_matrix = node["world_M"]
            _xj._process_vertex_chunks(body, vlist_off, state)

    print(f"total mesh-bearing nodes: {len([n for n in nodes if n['mesh_ptr']])}")
    print(f"total vertex slots filled: {len(state.vertex_slots)}")
    print()

    total_strips_attempted = 0
    total_strips_emitted = 0
    nodes_with_strips = 0
    nodes_emitting_zero = []

    for node in nodes:
        if not node["mesh_ptr"] or node["mesh_ptr"] + _xj._NJ_MESH_SIZE > len(body):
            continue
        mesh = _xj._read_nj_mesh(body, node["mesh_ptr"])
        if mesh is None:
            continue
        _vlist_off, plist_off, _bbox = mesh
        if not plist_off:
            continue
        if node["ef"] & (_xj.EVAL_HIDE | _xj.EVAL_SHAPE_SKIP):
            print(f"  HIDDEN node off={node['off']:#x} ef={fmt_flags(node['ef'])} would not emit strips")
            continue
        # Count attempts
        from formats.xj import _walk_chunk_stream, _process_polygon_chunks
        chunks = _walk_chunk_stream(body, plist_off)
        attempted = 0
        for (_h, type_id, _f, _bp, _bs) in chunks:
            if 64 <= type_id <= 75:
                # Each chunk may pack many strips; we count chunks here.
                attempted += 1
        meshes_before = []
        state.current_world_matrix = node["world_M"]
        _process_polygon_chunks(body, plist_off, state, meshes_before)
        emitted = len(meshes_before)
        total_strips_attempted += attempted
        total_strips_emitted += emitted
        if attempted > 0:
            nodes_with_strips += 1
        if attempted > 0 and emitted == 0:
            nodes_emitting_zero.append({"off": node["off"], "ef": node["ef"], "attempted": attempted})

    print(f"\nnodes with plist_off: {nodes_with_strips}")
    print(f"strip-chunks attempted: {total_strips_attempted}")
    print(f"sub-meshes emitted: {total_strips_emitted}")
    print(f"\nnodes that ATTEMPTED strips but emitted 0 (silent drops):")
    for d in nodes_emitting_zero:
        print(f"  off={d['off']:#x} ef={fmt_flags(d['ef'])} attempted={d['attempted']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
