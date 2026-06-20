"""Dump per-bone WORLD positions for a given .bml#inner.{nj,xj} model.

Investigation A from AGENT_MODEL_DEEP_DEBUG_REPORT.md.

The MeshTreeNode tree stores translation in BONE-LOCAL space and an
integer Ninja-angle (BAMS) rotation. The Euler ORDER used to compose
those angles into the local rotation matrix changes where each bone
ends up in world space — and the visible "head sub-meshes mis-rotated"
bug for the De Rol Le family is consistent with that order being
wrong.

This script walks the tree under different Euler orders ("XYZ",
"ZYX", "ZXY", "XZY", "YXZ", "YZX") and dumps the per-bone world
position for each, so we can pick the order that produces the most
plausible skeleton geometrically.

Usage::

    python scripts/dump_bone_world_positions.py \
        bm_boss2_de_rol_le_a.bml#boss2_b_derorure_body.nj \
        --orders ZYX,XYZ,ZXY \
        --json out/de_rol_le_bones.json

The default ``--orders`` list iterates through the four orders that
turn up in real Ninja-format references (Phantasmal: ZYX/ZXY,
pso-blender: XYZ/XZY).

The script intentionally lives outside ``formats/`` so it doesn't
break the test surface. It re-implements a minimal walker so we can
parameterize the rotation order without touching ``xj.py``'s default.
"""
from __future__ import annotations
import os

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Make the editor's modules importable when the script is run from any cwd.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from formats.bml import parse_bml, _prs_decompress  # noqa: E402
from formats.iff import parse_iff  # noqa: E402

_MESH_TREE_NODE_FMT = "<II3f3i3fII"
_MESH_TREE_NODE_SIZE = struct.calcsize(_MESH_TREE_NODE_FMT)
_BAMS_TO_RAD = (2.0 * math.pi) / 65536.0

# Eval flag bits — same numerical layout as both formats/xj.py and
# formats/xj_descriptor.py, so this script handles both Nj-chunk and
# Xj-descriptor models (the bone tree is identical).
EVAL_UNIT_POS = 0x01
EVAL_UNIT_ANG = 0x02
EVAL_UNIT_SCL = 0x04
EVAL_HIDE = 0x08
EVAL_BREAK = 0x10
EVAL_ZXY_ANG = 0x20
EVAL_SKIP = 0x40
EVAL_SHAPE_SKIP = 0x80


def _mat4_identity() -> List[float]:
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def _mat4_mul(a: List[float], b: List[float]) -> List[float]:
    out = [0.0] * 16
    for i in range(4):
        ai0 = a[i * 4 + 0]
        ai1 = a[i * 4 + 1]
        ai2 = a[i * 4 + 2]
        ai3 = a[i * 4 + 3]
        for j in range(4):
            out[i * 4 + j] = (
                ai0 * b[0 * 4 + j]
                + ai1 * b[1 * 4 + j]
                + ai2 * b[2 * 4 + j]
                + ai3 * b[3 * 4 + j]
            )
    return out


def _Rx(a: float) -> List[float]:
    c, s = math.cos(a), math.sin(a)
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0,   c,  -s, 0.0,
        0.0,   s,   c, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def _Ry(a: float) -> List[float]:
    c, s = math.cos(a), math.sin(a)
    return [
          c, 0.0,   s, 0.0,
        0.0, 1.0, 0.0, 0.0,
         -s, 0.0,   c, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def _Rz(a: float) -> List[float]:
    c, s = math.cos(a), math.sin(a)
    return [
          c,  -s, 0.0, 0.0,
          s,   c, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def _compose_rot(rx: float, ry: float, rz: float, order: str) -> List[float]:
    """Compose Euler angles into a row-major 4x4 rotation matrix.

    The order string follows three.js's convention: characters
    indicate the order matrices multiply LEFT-TO-RIGHT. So "ZYX"
    means R = Rz @ Ry @ Rx (intrinsic Z then Y then X).
    """
    table = {"X": _Rx(rx), "Y": _Ry(ry), "Z": _Rz(rz)}
    M = _mat4_identity()
    for ch in order:
        M = _mat4_mul(M, table[ch])
    return M


def _compose_trs(
    pos: Tuple[float, float, float],
    rot_rad: Tuple[float, float, float],
    scale: Tuple[float, float, float],
    order: str,
) -> List[float]:
    R = _compose_rot(rot_rad[0], rot_rad[1], rot_rad[2], order)
    M = list(R)
    M[0 * 4 + 3] = pos[0]
    M[1 * 4 + 3] = pos[1]
    M[2 * 4 + 3] = pos[2]
    sxsysz1 = (scale[0], scale[1], scale[2], 1.0)
    for i in range(4):
        for j in range(4):
            M[i * 4 + j] *= sxsysz1[j]
    return M


def _read_node(body: bytes, off: int):
    if off < 0 or off + _MESH_TREE_NODE_SIZE > len(body):
        return None
    f = struct.unpack_from(_MESH_TREE_NODE_FMT, body, off)
    return {
        "eval_flags": f[0],
        "mesh_ptr": f[1],
        "pos": (f[2], f[3], f[4]),
        "rot_bams": (f[5], f[6], f[7]),
        "scale": (f[8], f[9], f[10]),
        "child": f[11],
        "next": f[12],
    }


def walk_with_order(body: bytes, order: str, *, zxy_overrides_order: bool = True) -> List[Dict]:
    """Pre-order DFS yielding bone records under the given rotation order.

    When ``zxy_overrides_order`` is True (the Phantasmal/SDK behavior),
    nodes flagged with EVAL_ZXY_ANG use "ZXY" regardless of the
    caller's ``order`` choice. Set False to force the same order
    everywhere — useful for ablation testing.
    """
    n = len(body)
    out: List[Dict] = []
    visited: set = set()
    stack: List[Tuple[int, int, List[float]]] = [(0, -1, _mat4_identity())]
    while stack:
        off, parent_idx, parent_world = stack.pop()
        if off in visited or off + _MESH_TREE_NODE_SIZE > n:
            continue
        visited.add(off)
        node = _read_node(body, off)
        if node is None:
            continue
        ef = node["eval_flags"]
        if ef & EVAL_SKIP:
            local_M = _mat4_identity()
        else:
            lpos = (0.0, 0.0, 0.0) if (ef & EVAL_UNIT_POS) else node["pos"]
            if ef & EVAL_UNIT_ANG:
                lrot = (0.0, 0.0, 0.0)
            else:
                rb = node["rot_bams"]
                lrot = (rb[0] * _BAMS_TO_RAD, rb[1] * _BAMS_TO_RAD, rb[2] * _BAMS_TO_RAD)
            lscale = (1.0, 1.0, 1.0) if (ef & EVAL_UNIT_SCL) else node["scale"]
            this_order = order
            if zxy_overrides_order and (ef & EVAL_ZXY_ANG):
                this_order = "ZXY"
            local_M = _compose_trs(lpos, lrot, lscale, this_order)
        world_M = _mat4_mul(parent_world, local_M)
        my_idx = len(out)
        out.append({
            "index": my_idx,
            "parent": parent_idx,
            "off": off,
            "eval_flags": ef,
            "local_pos": list(node["pos"]),
            "rot_deg": [r * _BAMS_TO_RAD * 180.0 / math.pi for r in node["rot_bams"]],
            "local_scale": list(node["scale"]),
            "world_pos": [world_M[3], world_M[7], world_M[11]],
        })
        if node["next"] and node["next"] not in visited:
            stack.append((node["next"], parent_idx, parent_world))
        if node["child"] and node["child"] not in visited:
            stack.append((node["child"], my_idx, world_M))
    return out


def load_inner_bytes(path: str) -> bytes:
    """Resolve `<bml>#<inner>` or a bare `.nj`/`.xj` path under DATA_DIR."""
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="`<bml>#<inner>.{nj,xj}` or a bare .nj/.xj path under PSOBB.IO/data")
    ap.add_argument(
        "--orders",
        default="ZYX,XYZ,ZXY,XZY",
        help="Comma-separated list of Euler orders to test",
    )
    ap.add_argument("--json", default=None, help="Write per-order results to JSON file")
    ap.add_argument("--limit", type=int, default=40, help="Max bones to print per order")
    args = ap.parse_args()

    buf = load_inner_bytes(args.path)
    body = get_njcm_body(buf)
    orders = [s.strip() for s in args.orders.split(",") if s.strip()]
    out: Dict[str, List[Dict]] = {}
    for order in orders:
        bones = walk_with_order(body, order)
        out[order] = bones
        print(f"=== order {order} ({len(bones)} bones) ===")
        for b in bones[: args.limit]:
            wp = b["world_pos"]
            rot = b["rot_deg"]
            print(
                f"  bone[{b['index']:>3d}] parent={b['parent']:>3d} "
                f"world=({wp[0]:8.2f},{wp[1]:8.2f},{wp[2]:8.2f}) "
                f"rot_deg=({rot[0]:6.1f},{rot[1]:6.1f},{rot[2]:6.1f}) "
                f"ef=0x{b['eval_flags']:02x}"
            )
        print()
    if args.json:
        out_path = Path(args.json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
