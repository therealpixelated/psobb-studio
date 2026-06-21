"""Headless verification of the RE-derived De Rol Le bone-attach assembly.

Renders a 3-view matplotlib silhouette of the ASSEMBLED De Rol Le:
  * body inner (boss2_b_derorure_body.nj) drawn at identity (its own
    176-bone skeleton + the intrinsic pointy skull), and
  * each attack appendage (fin_a/fin_b/sting/tentacle) transformed by
    the FULL world matrix of its attach bone in the body skeleton,
    matching the model_viewer's bone-attach path.

This proves the appendages land at the head (fins, bones 33/34) and the
tail (sting/tentacle, bone 104) instead of floating at the origin — and
that the pointy skull is present in the body silhouette.

Usage:
    PSO_DATA_DIR=.../data python scripts/_verify_derolle_assembly.py \
        bm_boss2_de_rol_le.bml --out /tmp/derolle_assembled.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import dump_bone_world_positions as dbw  # noqa: E402  (sibling script)
from formats.bml import parse_bml, _prs_decompress  # noqa: E402
from formats import xj  # noqa: E402

# Map each appendage inner -> the body DFS bone index it attaches to.
# These mirror formats/composite_assembly.py's De Rol Le parts exactly.
ATTACH = {
    "boss2_b_derorure_fin_a.nj": 33,
    "boss2_b_derorure_fin_b.nj": 34,
    "boss2_b_derorure_sting.nj": 104,
    "boss2_b_derorure_tentacle.nj": 104,
}
BODY = "boss2_b_derorure_body.nj"


def _data_dir() -> Path:
    return Path(os.environ.get("PSO_DATA_DIR") or os.path.expanduser("~/PSOBB.IO/data"))


def _inner_bytes(blob: bytes, name: str) -> bytes:
    entries = parse_bml(blob)
    e = next((x for x in entries if x.name == name), None)
    if e is None:
        raise SystemExit(f"inner {name!r} not in BML")
    raw = bytes(blob[e.offset:e.offset + e.size_compressed])
    return _prs_decompress(raw, timeout=20.0)


def _bone_world_matrices(njcm_body: bytes):
    """Full 4x4 row-major world matrix per DFS bone index (ZYX order)."""
    n = len(njcm_body)
    out = []
    visited = set()
    stack = [(0, dbw._mat4_identity())]
    # Re-walk to capture full world matrices (the public walker only
    # returns world_pos). Same pre-order DFS as walk_with_order.
    stack = [(0, dbw._mat4_identity())]
    while stack:
        off, parent_world = stack.pop()
        if off in visited or off + dbw._MESH_TREE_NODE_SIZE > n:
            continue
        visited.add(off)
        node = dbw._read_node(njcm_body, off)
        if node is None:
            continue
        ef = node["eval_flags"]
        if ef & dbw.EVAL_SKIP:
            local_M = dbw._mat4_identity()
        else:
            lpos = (0.0, 0.0, 0.0) if (ef & dbw.EVAL_UNIT_POS) else node["pos"]
            if ef & dbw.EVAL_UNIT_ANG:
                lrot = (0.0, 0.0, 0.0)
            else:
                rb = node["rot_bams"]
                lrot = tuple(r * dbw._BAMS_TO_RAD for r in rb)
            lscale = (1.0, 1.0, 1.0) if (ef & dbw.EVAL_UNIT_SCL) else node["scale"]
            order = "ZXY" if (ef & dbw.EVAL_ZXY_ANG) else "ZYX"
            local_M = dbw._compose_trs(lpos, lrot, lscale, order)
        world_M = dbw._mat4_mul(parent_world, local_M)
        out.append(world_M)
        if node["next"] and node["next"] not in visited:
            stack.append((node["next"], parent_world))
        if node["child"] and node["child"] not in visited:
            stack.append((node["child"], world_M))
    return out


def _apply(M, p):
    x, y, z = p
    return (
        M[0] * x + M[1] * y + M[2] * z + M[3],
        M[4] * x + M[5] * y + M[6] * z + M[7],
        M[8] * x + M[9] * y + M[10] * z + M[11],
    )


def _inner_world_verts(njcm_bytes: bytes):
    """Return a flat list of (x,y,z) world verts for an inner (.nj)."""
    submeshes = xj.parse_nj_file(njcm_bytes)
    pts = []
    for sm in submeshes:
        for v in getattr(sm, "vertices", []) or []:
            p = v.pos  # XjVertex.pos = (x, y, z), already world-space
            pts.append((p[0], p[1], p[2]))
    return pts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bml", help="BML basename, e.g. bm_boss2_de_rol_le.bml")
    ap.add_argument("--out", default="/tmp/derolle_assembled.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blob = (_data_dir() / args.bml).read_bytes()

    body_bytes = _inner_bytes(blob, BODY)
    body_njcm = dbw.get_njcm_body(body_bytes)
    world_mats = _bone_world_matrices(body_njcm)
    print(f"body bones: {len(world_mats)}")

    # Body verts at identity.
    body_pts = _inner_world_verts(body_bytes)
    print(f"body verts: {len(body_pts)}")

    groups = [("body", body_pts, "0.6")]
    colors = {"boss2_b_derorure_fin_a.nj": "red",
              "boss2_b_derorure_fin_b.nj": "orange",
              "boss2_b_derorure_sting.nj": "green",
              "boss2_b_derorure_tentacle.nj": "blue"}
    for name, bone in ATTACH.items():
        ab = _inner_bytes(blob, name)
        local_pts = _inner_world_verts(ab)
        M = world_mats[bone] if bone < len(world_mats) else dbw._mat4_identity()
        wpts = [_apply(M, p) for p in local_pts]
        bx = world_mats[bone][3] if bone < len(world_mats) else 0
        by = world_mats[bone][7] if bone < len(world_mats) else 0
        bz = world_mats[bone][11] if bone < len(world_mats) else 0
        print(f"{name:34s} bone {bone:3d} -> attach world=({bx:7.2f},{by:7.2f},{bz:7.2f})  verts={len(wpts)}")
        groups.append((name, wpts, colors.get(name, "magenta")))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    views = [("front X-Y", 0, 1), ("top X-Z", 0, 2), ("side Z-Y", 2, 1)]
    for ax, (title, ai, bi) in zip(axes, views):
        for label, pts, col in groups:
            if not pts:
                continue
            xs = [p[ai] for p in pts]
            ys = [p[bi] for p in pts]
            ax.scatter(xs, ys, s=0.5, c=col, label=label if title.startswith("front") else None)
        ax.set_title(title)
        ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.3)
    axes[0].legend(markerscale=8, fontsize=7, loc="upper right")
    fig.suptitle(f"De Rol Le ASSEMBLED (body+skull + bone-attached appendages) — {args.bml}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=90)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
