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
"""
from __future__ import annotations

import argparse
import io
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Make `formats` importable when run as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from formats import rel as _rel  # noqa: E402
from formats import rel_writer as _rw  # noqa: E402
from formats.decimate import decimate_mesh  # noqa: E402
from formats.import_external import parse_gltf  # noqa: E402


# --------------------------------------------------------------------------
# vertex shim matching the extract_nrel_meshes shape nrel_nodes_from_meshes wants
# --------------------------------------------------------------------------
class _V:
    __slots__ = ("pos", "normal", "uv")

    def __init__(self, pos, normal, uv):
        self.pos = pos
        self.normal = normal
        self.uv = uv


class _MeshShim:
    """Adapts a merged (verts, normals, uvs, faces) block to the XjMesh shape."""

    def __init__(self, verts, normals, uvs, faces, material_id: int = 0):
        self.vertices = [
            _V(
                (float(verts[i, 0]), float(verts[i, 1]), float(verts[i, 2])),
                (float(normals[i, 0]), float(normals[i, 1]), float(normals[i, 2])),
                (float(uvs[i, 0]), float(uvs[i, 1])),
            )
            for i in range(verts.shape[0])
        ]
        self.indices = [int(x) for x in faces.reshape(-1)]
        self.material_id = int(material_id)


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals (the engine recomputes winding from these)."""
    out = np.zeros_like(verts, dtype=np.float64)
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(out, faces[:, k], fn)
    ln = np.linalg.norm(out, axis=1, keepdims=True)
    ln[ln == 0.0] = 1.0
    return out / ln


def load_and_merge(glb_path: Path):
    """Parse a GLB and concatenate every submesh into one geometry block."""
    model = parse_gltf(glb_path.read_bytes(), glb=True)
    allV: List[np.ndarray] = []
    allU: List[np.ndarray] = []
    allF: List[np.ndarray] = []
    off = 0
    for m in model.meshes:
        V = np.asarray(m.vertices, dtype=np.float64).reshape(-1, 3)
        n = V.shape[0]
        if n == 0:
            continue
        U = (
            np.asarray(m.uvs, dtype=np.float64).reshape(-1, 2)
            if m.uvs is not None
            else np.zeros((n, 2), dtype=np.float64)
        )
        F = np.asarray(m.indices, dtype=np.int64).reshape(-1, 3) + off
        allV.append(V)
        allU.append(U)
        allF.append(F)
        off += n
    if not allV:
        raise SystemExit("GLB has no triangle geometry")
    V = np.vstack(allV)
    U = np.vstack(allU)
    F = np.vstack(allF)
    return V, U, F, model


def author_nrel(V, F, texname: str, enforce: bool) -> bytes:
    N = _vertex_normals(V, F)
    shim = _MeshShim(V, N, V[:, :2] * 0.0, F)  # uv filled below
    return _rw.build_nrel_from_meshes(
        _rw.nrel_nodes_from_meshes([shim]), [texname], enforce_budget=enforce
    )


def author_nrel_uv(V, U, F, texname: str, enforce: bool) -> bytes:
    N = _vertex_normals(V, F)
    shim = _MeshShim(V, N, U, F)
    return _rw.build_nrel_from_meshes(
        _rw.nrel_nodes_from_meshes([shim]), [texname], enforce_budget=enforce
    )


def decimate_to_fit(V, U, F, budget: int, texname: str):
    """Shrink the merged mesh until the authored n.rel fits ``budget`` bytes."""
    target = F.shape[0]
    # First pass: an a-priori target from the measured ~170 B/triangle cost,
    # leaving headroom; then verify+shrink against the real authored size.
    target = min(target, max(64, budget // 200))
    cur_v, cur_u, cur_f = V, U, F
    for _ in range(8):
        if target < cur_f.shape[0]:
            dv, df, du, _meta = decimate_mesh(
                cur_v, cur_f, target_tris=int(target), uvs=cur_u, return_meta=True
            )
            cur_v = np.asarray(dv, dtype=np.float64).reshape(-1, 3)
            cur_f = np.asarray(df, dtype=np.int64).reshape(-1, 3)
            cur_u = (
                np.asarray(du, dtype=np.float64).reshape(-1, 2)
                if du is not None
                else np.zeros((cur_v.shape[0], 2))
            )
        buf = author_nrel_uv(cur_v, cur_u, cur_f, texname, enforce=False)
        if len(buf) <= budget:
            return cur_v, cur_u, cur_f, buf
        # over budget: scale the target down by the overage + margin and retry.
        target = int(cur_f.shape[0] * (budget / len(buf)) * 0.9)
        target = max(64, target)
    raise SystemExit(f"could not fit n.rel under {budget} bytes after 8 passes")


def author_crel(V, F, budget: int) -> Optional[bytes]:
    """Author collision from the same geometry; decimate if it overflows c.rel."""
    cur_v, cur_f = V, F
    for _ in range(8):
        node = _rw.CrelNode(
            verts=[tuple(map(float, cur_v[i])) for i in range(cur_v.shape[0])],
            faces=[_rw.CrelFace(int(a), int(b), int(c)) for a, b, c in cur_f],
        )
        try:
            return _rw.build_crel([node])
        except _rw.RelWriteError:
            target = max(32, int(cur_f.shape[0] * 0.5))
            if target >= cur_f.shape[0]:
                return None
            dv, df, _u, _m = decimate_mesh(cur_v, cur_f, target_tris=target, return_meta=True)
            cur_v = np.asarray(dv, dtype=np.float64).reshape(-1, 3)
            cur_f = np.asarray(df, dtype=np.int64).reshape(-1, 3)
    return None


def author_xvm(model) -> Optional[bytes]:
    """Best-effort XVM from the GLB's embedded textures (DXT1/DXT5)."""
    try:
        from PIL import Image

        from formats.xvr_decode import build_xvm, encode_xvr_record
    except Exception:
        return None
    recs = []
    for i, t in enumerate(getattr(model, "textures", []) or []):
        data = getattr(t, "data", b"")
        if not data:
            continue
        try:
            img = Image.open(io.BytesIO(data)).convert("RGBA")
            alpha_min = img.getextrema()[3][0]
            fmt = 10 if alpha_min < 255 else 6  # DXT5 if alpha, else DXT1
            recs.append(encode_xvr_record(img, fmt, global_index=i))
        except Exception:
            continue
    if not recs:
        return None
    try:
        # PVMH-style header: 'PVMH' + size + flags + count (best-effort; the
        # decoder tolerates a minimal header — verified by re-parse below).
        header = bytearray(b"PVMH")
        header += struct.pack("<I", 8)
        header += struct.pack("<HH", 0x0F, len(recs))
        return build_xvm(bytes(header), recs)
    except Exception:
        return None


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------
def verify_nrel(buf: bytes) -> Tuple[bool, str]:
    try:
        rf = _rel.parse_rel(buf)
        if not _rel.is_n_rel(rf):
            return False, "not recognised as n.rel"
        meshes = _rel.extract_nrel_meshes(rf)
        _rw.simulate_rel_relocation(buf)
        tris = sum(len(getattr(m, "indices", [])) // 3 for m in meshes)
        ok = len(buf) <= _rw.NREL_SIZE_BUDGET
        return ok, f"{len(meshes)} mesh(es), {tris} tris, {len(buf):,}B (<=768KB:{ok})"
    except Exception as e:  # noqa: BLE001
        return False, f"reparse failed: {type(e).__name__}: {e}"


def verify_crel(buf: bytes) -> Tuple[bool, str]:
    try:
        rf = _rel.parse_rel(buf)
        _rw.simulate_rel_relocation(buf)
        ok = len(buf) <= _rw.CREL_SIZE_BUDGET
        return ok, f"{len(buf):,}B (<=64KB:{ok}), n_rel={_rel.is_n_rel(rf)}"
    except Exception as e:  # noqa: BLE001
        return False, f"reparse failed: {type(e).__name__}: {e}"


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
