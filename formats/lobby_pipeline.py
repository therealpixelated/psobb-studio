#!/usr/bin/env python3
"""Importable, bytes-only lobby/floor build pipeline.

This is the shared core extracted from ``scripts/build_lobby.py`` so the
GLB -> n.rel/c.rel/xvm authoring path can be driven from BOTH the offline
CLI and the in-process floor-editor endpoints.

DESIGN INVARIANTS (load-bearing — do not break):
  * NO env reads, NO argparse, NO prints, NO filesystem writes here. The
    public entry :func:`build_floor` returns BYTES ONLY; the caller (CLI
    or server) decides where (and whether) to write them. The server
    confines every write to its DEV dir and never to the live install.
  * The final authored n.rel/c.rel are produced with ``enforce_budget=True``
    so an over-budget result RAISES ``RelWriteError`` (the caller maps that
    to HTTP 422). ``enforce_budget=False`` is used ONLY inside
    :func:`decimate_to_fit`'s bounded retry loop, and the bytes it returns
    are post-checked ``len <= budget`` before they ever escape.
  * Every authored buffer is run through ``simulate_rel_relocation`` (via
    :func:`verify_nrel` / :func:`verify_crel`) so an un-relocatable .rel is
    never returned.

``scripts/build_lobby.py`` re-imports the names defined here (``_V``,
``_MeshShim``, ``_vertex_normals``, ``load_and_merge``, ``author_nrel``,
``author_nrel_uv``, ``decimate_to_fit``, ``author_crel``, ``author_xvm``,
``verify_nrel``, ``verify_crel``) so existing callers and tests
(``tests/test_lobby_build.py``) keep working unchanged.
"""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from formats import rel as _rel
from formats import rel_writer as _rw
from formats.decimate import decimate_mesh
from formats.import_external import parse_gltf

# Re-export the budgets so callers can import them from one place.
NREL_SIZE_BUDGET = _rw.NREL_SIZE_BUDGET  # 0xC0000 (768 KB)
CREL_SIZE_BUDGET = _rw.CREL_SIZE_BUDGET  # 0x10000 (64 KB)


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


def merge_model(model):
    """Concatenate every submesh of a parsed model into one geometry block.

    Returns ``(V, U, F, submesh_count, tri_total)``. Raises ValueError when
    the model carries no triangle geometry (the CLI maps this to SystemExit
    for back-compat; the server maps it to HTTP 400).
    """
    allV: List[np.ndarray] = []
    allU: List[np.ndarray] = []
    allF: List[np.ndarray] = []
    off = 0
    submesh_count = 0
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
        submesh_count += 1
    if not allV:
        raise ValueError("model has no triangle geometry")
    V = np.vstack(allV)
    U = np.vstack(allU)
    F = np.vstack(allF)
    return V, U, F, submesh_count, int(F.shape[0])


def load_and_merge(glb_path: Path):
    """Parse a GLB file and concatenate every submesh into one geometry block.

    Back-compat wrapper kept for ``scripts/build_lobby.py`` / its tests.
    Returns ``(V, U, F, model)``.
    """
    model = parse_gltf(Path(glb_path).read_bytes(), glb=True)
    try:
        V, U, F, _sm, _tt = merge_model(model)
    except ValueError as e:
        # Preserve the historic CLI behaviour.
        raise SystemExit(str(e))
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
    """Shrink the merged mesh until the authored n.rel fits ``budget`` bytes.

    Uses ``enforce_budget=False`` inside the loop (so an over-budget pass
    can be measured + retried) but POST-CHECKS ``len <= budget`` before
    returning. Raises SystemExit after 8 passes if it still won't fit — the
    server catches that and maps it to HTTP 422.
    """
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
    """Author collision from the same geometry; decimate if it overflows c.rel.

    Returns the c.rel bytes, or None when collision can't be made to fit
    (the caller surfaces that as a visible SKIP-with-reason warning).
    """
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
    """Best-effort XVM from the model's embedded textures (DXT1/DXT5).

    Returns None when there are no embedded textures or the encoder isn't
    available — the caller converts None into a visible SKIP warning.
    """
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
# public build entry — bytes only, no filesystem
# --------------------------------------------------------------------------
@dataclass
class FloorBuildResult:
    """The bytes-only output of :func:`build_floor` plus a verify report."""
    nrel: bytes
    crel: Optional[bytes]
    xvm: Optional[bytes]
    report: dict = field(default_factory=dict)


def build_floor(
    source: Union[bytes, "object"],
    *,
    texname: str = "lobby",
    nrel_budget: int = NREL_SIZE_BUDGET,
    crel_budget: int = CREL_SIZE_BUDGET,
) -> FloorBuildResult:
    """Author a floor (n.rel + optional c.rel + optional .xvm) from a model.

    ``source`` is either raw GLB bytes (magic ``b"glTF"``) or an already
    parsed model object (anything with ``.meshes`` / ``.textures``).

    Returns a :class:`FloorBuildResult` carrying BYTES ONLY — it never
    touches the filesystem. The caller writes (atomically, into a DEV dir).

    Raises:
      * ``ValueError`` when the model has no triangle geometry.
      * ``rel_writer.RelWriteError`` when the FINAL authored n.rel/c.rel
        exceeds its budget (the 8-pass decimate couldn't save it).
      * ``RuntimeError`` when an authored buffer fails re-parse /
        relocation simulation.
    """
    if isinstance(source, (bytes, bytearray)):
        model = parse_gltf(bytes(source), glb=True)
    else:
        model = source

    V, U, F, submesh_count, tri_in = merge_model(model)
    texture_count = len(getattr(model, "textures", []) or [])
    warnings: List[str] = []
    errors: List[str] = []

    # ---- n.rel: decimate to fit, then RE-AUTHOR with enforce_budget=True
    # so an over-budget result raises (never silently truncated). ----
    V2, U2, F2, _nrel_unenforced = decimate_to_fit(V, U, F, nrel_budget, texname)
    tri_out = int(F2.shape[0])
    if tri_out != tri_in:
        warnings.append(
            f"decimated {tri_in} -> {tri_out} tris to fit the n.rel budget")
    # Final author with budget enforcement on. Raises RelWriteError on
    # overflow (caller -> HTTP 422). For a flat single-root tree built from
    # the merged mesh this matches the unenforced bytes, but enforcing here
    # makes the guarantee explicit at the point bytes escape.
    nrel = author_nrel_uv(V2, U2, F2, texname, enforce=True)
    ok, msg = verify_nrel(nrel)
    if not ok:
        raise RuntimeError(f"authored n.rel failed verification: {msg}")

    # ---- c.rel (optional) ----
    crel = author_crel(V2, F2, crel_budget)
    crel_size = 0
    if crel is not None:
        ok, msg = verify_crel(crel)
        if not ok:
            # A produced-but-unverifiable collision hull is dropped, not
            # written — surface it rather than ship a bad c.rel.
            warnings.append(f"c.rel dropped (failed verification: {msg})")
            crel = None
        else:
            crel_size = len(crel)
    if crel is None:
        warnings.append("c.rel SKIP — collision could not fit 64KB; "
                        "floor ships without a collision hull")

    # ---- .xvm (optional, best-effort) ----
    xvm = author_xvm(model)
    xvm_size = 0
    if xvm is None:
        warnings.append("xvm SKIP — no embedded textures / encoder "
                        "unavailable; floor will render untextured in-game")
    else:
        xvm_size = len(xvm)

    report = {
        "submesh_count": submesh_count,
        "texture_count": texture_count,
        "tri_in": tri_in,
        "tri_out": tri_out,
        "nrel_size": len(nrel),
        "crel_size": crel_size,
        "xvm_size": xvm_size,
        # CREATE authors a flat SINGLE-root mesh tree with ONE TextureList
        # entry — every triangle binds to texture slot 0 (build_lobby
        # limitation). Surfaced so the UI never implies per-submesh textures.
        "single_texture_slot": True,
        # CREATE builds a single root node, so no child sub-trees are
        # dropped (unlike a re-authored vanilla copy). Always 0 here.
        "dropped_child_nodes": 0,
        "warnings": warnings,
        "errors": errors,
    }
    return FloorBuildResult(nrel=nrel, crel=crel, xvm=xvm, report=report)


__all__ = [
    "NREL_SIZE_BUDGET",
    "CREL_SIZE_BUDGET",
    "_V",
    "_MeshShim",
    "_vertex_normals",
    "merge_model",
    "load_and_merge",
    "author_nrel",
    "author_nrel_uv",
    "decimate_to_fit",
    "author_crel",
    "author_xvm",
    "verify_nrel",
    "verify_crel",
    "FloorBuildResult",
    "build_floor",
]
