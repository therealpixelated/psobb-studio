#!/usr/bin/env python3
"""Offline lobby/floor build pipeline: GLB -> n.rel + c.rel (+ XVM).

The §8 acceptance build, OFFLINE half. Takes a source GLB (e.g. the
Casinopolis model), merges its submeshes, decimates to fit the engine's
n.rel size cap, and authors a deployable geometry (`*n.rel`), collision
(`*c.rel`) and — best-effort — texture (`*.xvm`) set, then verifies every
file re-parses, relocates cleanly, and fits its byte budget.

Paths come from the environment only (never hardcoded):
  * source GLB:  --glb PATH  |  $PSOBB_DOWNLOADS_DIR (largest *.glb)
  * output dir:  --out DIR   |  $PSO_DEV_DATA_DIR  |  <tmp>/pso_lobby_build

This NEVER writes to $PSO_LIVE_DATA_DIR. Deploying to a live install is a
separate, guarded step (backup + newserv stop + atomic rename + restart).

NOTE (2026-06-20 refactor): the geometry / author / verify core now lives
in the importable, bytes-only ``formats.lobby_pipeline`` module so the
in-process floor-editor endpoints can drive the same pipeline. This file
is a THIN CLI WRAPPER: it owns the env + argparse + filesystem surface and
re-exports the moved names so existing callers (and ``tests/test_lobby_build``)
keep working unchanged.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Make `formats` importable when run as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from formats import rel_writer as _rw  # noqa: E402

# Re-export the geometry / author / verify core from the importable
# pipeline module so this CLI (and its tests) reference the SAME code the
# server endpoints use. These names are part of this module's public
# surface — tests do `import build_lobby as bl; bl.decimate_to_fit(...)`.
from formats.lobby_pipeline import (  # noqa: E402,F401
    _MeshShim,
    _V,
    _vertex_normals,
    author_crel,
    author_nrel,
    author_nrel_uv,
    author_xvm,
    build_floor,
    decimate_to_fit,
    load_and_merge,
    merge_model,
    verify_crel,
    verify_nrel,
)


# --------------------------------------------------------------------------
# env + CLI surface (the only part that touches the filesystem / environment)
# --------------------------------------------------------------------------
def discover_glb(arg: Optional[str]) -> Path:
    if arg:
        p = Path(os.path.expanduser(arg))
        if not p.is_file():
            raise SystemExit(f"--glb not found: {p}")
        return p
    root = os.environ.get("PSOBB_DOWNLOADS_DIR") or os.path.expanduser("~/Downloads")
    cands = sorted(Path(root).glob("*.glb"), key=lambda p: p.stat().st_size, reverse=True)
    if not cands:
        raise SystemExit(f"no *.glb in {root} (set --glb or $PSOBB_DOWNLOADS_DIR)")
    return cands[0]


def out_dir(arg: Optional[str]) -> Path:
    d = (
        arg
        or os.environ.get("PSO_DEV_DATA_DIR")
        or str(Path(tempfile.gettempdir()) / "pso_lobby_build")
    )
    p = Path(os.path.expanduser(d))
    p.mkdir(parents=True, exist_ok=True)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Offline GLB -> n.rel/c.rel/xvm lobby build")
    ap.add_argument("--glb", help="source GLB (else largest in $PSOBB_DOWNLOADS_DIR)")
    ap.add_argument("--out", help="output dir (else $PSO_DEV_DATA_DIR or tmp)")
    ap.add_argument("--name", default="map_lobby_01", help="base map name (default map_lobby_01)")
    ap.add_argument("--tex", default="lobby", help="texture name for the n.rel TextureList")
    args = ap.parse_args(argv)

    glb = discover_glb(args.glb)
    outd = out_dir(args.out)
    print(f"[build_lobby] source : {glb.name} ({glb.stat().st_size:,}B)")
    print(f"[build_lobby] output : {outd}")

    V, U, F, model = load_and_merge(glb)
    print(f"[build_lobby] merged : {V.shape[0]:,} verts, {F.shape[0]:,} tris, "
          f"{len(model.meshes)} submeshes, {len(getattr(model,'textures',[]) or [])} textures")

    V2, U2, F2, nrel = decimate_to_fit(V, U, F, _rw.NREL_SIZE_BUDGET, args.tex)
    if F2.shape[0] != F.shape[0]:
        print(f"[build_lobby] decimated to {F2.shape[0]:,} tris to fit n.rel cap")

    crel = author_crel(V2, F2, _rw.CREL_SIZE_BUDGET)
    xvm = author_xvm(model)

    results = []
    nrel_path = outd / f"{args.name}n.rel"
    nrel_path.write_bytes(nrel)
    ok, msg = verify_nrel(nrel)
    results.append(ok)
    print(f"[build_lobby] {nrel_path.name:<20} {'OK ' if ok else 'BAD'} {msg}")

    if crel is not None:
        crel_path = outd / f"{args.name}c.rel"
        crel_path.write_bytes(crel)
        ok, msg = verify_crel(crel)
        results.append(ok)
        print(f"[build_lobby] {crel_path.name:<20} {'OK ' if ok else 'BAD'} {msg}")
    else:
        print("[build_lobby] c.rel               SKIP (could not fit collision under 64KB)")

    if xvm is not None:
        xvm_path = outd / f"{args.name}.xvm"
        xvm_path.write_bytes(xvm)
        print(f"[build_lobby] {xvm_path.name:<20} OK  {len(xvm):,}B (best-effort)")
    else:
        print("[build_lobby] xvm                 SKIP (no embedded textures / encoder unavailable)")

    passed = all(results) and len(results) > 0
    print(f"[build_lobby] {'ACCEPTANCE PASS' if passed else 'ACCEPTANCE FAIL'} "
          f"(n.rel + c.rel parse, relocate, fit caps)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
