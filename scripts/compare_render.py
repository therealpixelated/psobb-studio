"""Render PSOBB models to PNG via matplotlib (Investigation E).

For each `.bml#inner` model, this script:

  1. Parses the model via formats.xj or formats.xj_descriptor.
  2. Computes a wireframe + per-submesh-color projection in three
     viewports (XY, XZ, YZ).
  3. Saves the result as
     ``C:/tmp_pso_editor/render_compare/<bml>__<inner>.png``.

It uses matplotlib (a stdlib-adjacent dep that's already installed
for many Python distributions). If matplotlib isn't available the
script bails with a clear message — no new pip dependencies are
required to be added to the editor.

Output PNGs are deliberately small (3-panel, 1200x400) so they're
easy to skim visually for the wireframe-vs-solid comparison.

Usage::

    python scripts/compare_render.py \\
        bm_boss2_de_rol_le_a.bml#boss2_b_derorure_body.nj \\
        bm_boss2_de_rol_le.bml#boss2_b_helm_break.nj \\
        bm_boss8_dragon.bml#boss1_s_nb_dragon.nj \\
        bm_ene_gibbles_low.bml#lo_gibb_body.nj \\
        bm_fe_obj_o_door01l.bml#fe_obj_o_door01l.xj
"""
from __future__ import annotations
import os

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

OUTPUT_DIR = Path(r"C:/tmp_pso_editor/render_compare")


def _load_inner_bytes(path: str) -> Tuple[bytes, str]:
    """Resolve `<bml>#<inner>` or a bare path; return (raw_njcm_bytes, ext)."""
    from formats.bml import parse_bml, _prs_decompress

    install_data = Path(os.path.expanduser("~/PSOBB.IO/data"))
    if "#" in path:
        bml_name, inner = path.split("#", 1)
        bml_path = install_data / bml_name
        if not bml_path.exists():
            raise SystemExit(f"BML not found: {bml_path}")
        blob = bml_path.read_bytes()
        entries = parse_bml(blob)
        target = next((e for e in entries if e.name == inner), None)
        if target is None:
            raise SystemExit(f"inner {inner!r} not in {bml_name}")
        raw = bytes(blob[target.offset:target.offset + target.size_compressed])
        return _prs_decompress(raw, timeout=20.0), Path(inner).suffix.lower()
    p = install_data / path
    if not p.exists():
        raise SystemExit(f"file not found: {p}")
    return p.read_bytes(), p.suffix.lower()


def _parse_meshes(buf: bytes, ext: str):
    if ext == ".xj":
        from formats.xj_descriptor import parse_xj_file
        return parse_xj_file(buf)
    from formats.xj import parse_nj_file
    return parse_nj_file(buf)


def _render_one(path: str, out_dir: Path) -> Path:
    """Render a single model and return the saved PNG path."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        import numpy as np
    except ImportError as e:
        raise SystemExit(
            f"matplotlib unavailable; can't render comparison PNGs: {e}\n"
            "Install with `pip install matplotlib` (it's the only soft dep "
            "this script adds; the editor itself does NOT require it)."
        )

    buf, ext = _load_inner_bytes(path)
    meshes = _parse_meshes(buf, ext)
    if not meshes:
        raise SystemExit(f"no meshes parsed for {path}")

    # Collect per-submesh edges + colors
    colors = plt.cm.hsv(np.linspace(0, 1, max(len(meshes), 6))[: len(meshes)])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{path} — {len(meshes)} sub-meshes")

    plane_pairs = [
        ("X", "Y", 0, 1, "front (X right, Y up)"),
        ("X", "Z", 0, 2, "top (X right, Z down)"),
        ("Z", "Y", 2, 1, "side (Z right, Y up)"),
    ]
    for ax, (a_label, b_label, ai, bi, title) in zip(axes, plane_pairs):
        for mi, m in enumerate(meshes):
            verts = np.array([v.pos for v in m.vertices], dtype=float)
            indices = np.array(m.indices, dtype=int).reshape(-1, 3)
            if not len(verts) or not len(indices):
                continue
            edges = []
            for tri in indices:
                a, b, c = tri
                edges.append([verts[a, [ai, bi]], verts[b, [ai, bi]]])
                edges.append([verts[b, [ai, bi]], verts[c, [ai, bi]]])
                edges.append([verts[c, [ai, bi]], verts[a, [ai, bi]]])
            lc = LineCollection(
                edges, linewidths=0.4, colors=[colors[mi % len(colors)]]
            )
            ax.add_collection(lc)
        # equal axes
        all_verts = []
        for m in meshes:
            for v in m.vertices:
                all_verts.append(v.pos)
        if all_verts:
            arr = np.array(all_verts)
            xmin, ymin = arr[:, ai].min(), arr[:, bi].min()
            xmax, ymax = arr[:, ai].max(), arr[:, bi].max()
            pad_x = max((xmax - xmin) * 0.05, 1.0)
            pad_y = max((ymax - ymin) * 0.05, 1.0)
            ax.set_xlim(xmin - pad_x, xmax + pad_x)
            ax.set_ylim(ymin - pad_y, ymax + pad_y)
            ax.invert_yaxis() if a_label == "X" and b_label == "Z" else None
        ax.set_title(title)
        ax.set_xlabel(a_label)
        ax.set_ylabel(b_label)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = path.replace("#", "__").replace("/", "_").replace("\\", "_")
    out_path = out_dir / f"{safe_name}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "paths",
        nargs="*",
        default=[
            "bm_boss2_de_rol_le_a.bml#boss2_b_derorure_body.nj",
            "bm_boss2_de_rol_le_a.bml#boss2_b_helm_break.nj",
            "bm_boss2_de_rol_le.bml#boss2_b_derorure_body.nj",
            "bm_boss7_de_rol_le_c.bml#boss2_b_derorure_body.nj",
            "bm_boss8_dragon.bml#boss1_s_nb_dragon.nj",
            "bm_ene_gibbles_low.bml#lo_gibb_body.nj",
            "bm_fe_obj_o_door01l.bml#fe_obj_o_door01l.xj",
            "bm_eff_ice.bml#ice_root.xj",
        ],
        help="`<bml>#<inner>` paths under PSOBB.IO/data",
    )
    ap.add_argument(
        "--out",
        default=str(OUTPUT_DIR),
        help="Output directory for PNGs",
    )
    args = ap.parse_args()
    out_dir = Path(args.out)

    saved = []
    for path in args.paths:
        try:
            out_path = _render_one(path, out_dir)
            print(f"rendered {path} -> {out_path}")
            saved.append(str(out_path))
        except SystemExit as e:
            print(f"SKIP {path}: {e}")
    print(f"\nWrote {len(saved)} PNGs to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
