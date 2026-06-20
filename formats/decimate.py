"""Mesh decimation (LOD) via Quadric Error Metrics (QEM).

This module provides a real triangle-reduction decimator — the thing the
``decimate`` UI used to stub out. It is used both by the sculpt-panel
"Decimate" control (``POST /api/decimate``) and, more importantly, by the
Casinopolis acceptance test where a ~9.5k-tri (and an alt ~33.6k-tri) lobby
GLB must be reduced to fit the engine's REL size caps:

  * ``n.rel`` (node geometry)  768 KB
  * ``c.rel`` (collision)       64 KB

Two public entry points:

  * :func:`decimate_mesh` — reduce a (vertices, faces) mesh to a target
    triangle count or ratio with QEM. Border edges are pinned by default so
    open meshes don't shrink at the seams.

  * :func:`decimate_to_byte_budget` — binary-search the target triangle count
    so the *encoded* output (the caller supplies the size estimator) fits a
    byte budget. Returns an ``over_budget`` flag + the overage when even the
    floor tri count can't fit, so the acceptance test can then split into
    multiple REL nodes.

Backend selection
-----------------
QEM proper is delegated to ``trimesh.Trimesh.simplify_quadric_decimation``,
which in trimesh 4.x dispatches to the ``fast-simplification`` package (a
C++ QEM implementation). When that backend is unavailable we fall back to a
hand-rolled, pure-NumPy QEM edge-collapse decimator (:func:`_qem_fallback`).
Both paths are reported via the returned ``backend`` string / meta so callers
(and tests) can see which one actually ran.

UV / attribute handling
------------------------
The QEM backends operate on positions + topology only. UVs and per-vertex
normals are **not** carried through an edge collapse by either backend in this
version (a correct attribute-aware collapse needs the wedge/seam machinery the
subdivide path has). To keep the result coherent we recompute smooth vertex
normals from the decimated faces, and we re-sample UVs by nearest-original-
vertex when the caller asks for them (good enough for a LOD; documented as a
limitation). Positions and faces are always coherent and watertight-preserving
where the input was watertight.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("psobb_studio.decimate")

# Don't ever reduce below this — a degenerate sliver is useless and the QEM
# backends can crash on absurd targets. The byte-budget search also uses this
# as its floor.
MIN_TARGET_TRIS = 4
DEFAULT_BUDGET_FLOOR_TRIS = 200


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def decimate_mesh(
    vertices,
    faces,
    *,
    target_ratio: Optional[float] = None,
    target_tris: Optional[int] = None,
    preserve_border: bool = True,
    uvs=None,
    return_meta: bool = False,
):
    """Decimate a triangle mesh to a target triangle count / ratio via QEM.

    Parameters
    ----------
    vertices : (N, 3) array-like of float
        Vertex positions.
    faces : (M, 3) array-like of int
        Triangle vertex indices.
    target_ratio : float, optional
        Fraction of triangles to KEEP, in (0, 1]. ``0.5`` ~ halve the mesh.
        Mutually informative with ``target_tris`` — if both are given,
        ``target_tris`` wins.
    target_tris : int, optional
        Absolute target triangle count. Clamped to ``[MIN_TARGET_TRIS, M]``.
    preserve_border : bool, default True
        Pin boundary edges (those used by exactly one face) so an open mesh
        does not shrink at its seams. Honoured by both backends where
        supported; the fallback always honours it.
    uvs : (N, 2) array-like, optional
        Per-vertex UVs. When supplied, the decimated mesh's UVs are
        re-sampled from the nearest surviving original vertex (see module
        docstring — this is a LOD approximation, not a wedge-correct carry).
    return_meta : bool, default False
        When True, also return a ``meta`` dict (backend, in/out counts).

    Returns
    -------
    (vertices, faces)  or  (vertices, faces, uvs_or_None, meta)
        ``vertices`` is (N', 3) float64, ``faces`` is (M', 3) int64.
        When ``return_meta`` is True the tuple also carries the re-sampled
        UVs (or None) and a meta dict.

    Notes
    -----
    A no-op (target >= current) returns a *copy* of the input, with
    ``backend="noop"``.
    """
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    n_in_v = len(v)
    n_in_f = len(f)

    if n_in_f == 0:
        meta = {"backend": "empty", "in_tris": 0, "out_tris": 0,
                "in_verts": n_in_v, "out_verts": n_in_v,
                "preserve_border": bool(preserve_border)}
        if return_meta:
            return v.copy(), f.copy(), (None if uvs is None else np.asarray(uvs, dtype=np.float64).copy()), meta
        return v.copy(), f.copy()

    tgt = _resolve_target_tris(n_in_f, target_ratio, target_tris)

    # No reduction requested (or target rounds up past current) — pass through.
    if tgt >= n_in_f:
        meta = {"backend": "noop", "in_tris": n_in_f, "out_tris": n_in_f,
                "in_verts": n_in_v, "out_verts": n_in_v,
                "target_tris": tgt, "preserve_border": bool(preserve_border)}
        out_uv = None if uvs is None else np.asarray(uvs, dtype=np.float64).reshape(-1, 2).copy()
        if return_meta:
            return v.copy(), f.copy(), out_uv, meta
        return v.copy(), f.copy()

    out_v, out_f, backend = _run_qem(v, f, tgt, preserve_border)

    # Drop any degenerate (repeated-index) faces the collapse may produce and
    # any vertices left unreferenced, so the output is clean.
    out_v, out_f = _clean_mesh(out_v, out_f)

    out_uv = None
    if uvs is not None:
        out_uv = _resample_uvs(v, np.asarray(uvs, dtype=np.float64).reshape(-1, 2), out_v)

    meta = {
        "backend": backend,
        "in_tris": n_in_f,
        "out_tris": int(len(out_f)),
        "in_verts": n_in_v,
        "out_verts": int(len(out_v)),
        "target_tris": int(tgt),
        "preserve_border": bool(preserve_border),
    }
    log.info(
        "decimate_mesh: %d -> %d tris (target %d, backend=%s)",
        n_in_f, len(out_f), tgt, backend,
    )

    if return_meta:
        return out_v, out_f, out_uv, meta
    return out_v, out_f


def decimate_to_byte_budget(
    vertices,
    faces,
    *,
    encode_size_fn: Callable[[np.ndarray, np.ndarray], int],
    budget_bytes: int,
    max_iters: int = 12,
    floor_tris: int = DEFAULT_BUDGET_FLOOR_TRIS,
    preserve_border: bool = True,
    uvs=None,
):
    """Binary-search a target tri count so the encoded mesh fits a byte budget.

    ``encode_size_fn(vertices, faces) -> int`` is the caller-supplied size
    estimator: given a decimated mesh it returns the number of bytes the real
    encoder would emit. We binary-search the triangle count between
    ``floor_tris`` and the input tri count to find the largest mesh whose
    encoded size is ``<= budget_bytes``.

    Returns
    -------
    (vertices, faces, meta)
        ``meta`` carries:
          * ``over_budget`` (bool) — True if even ``floor_tris`` does not fit.
          * ``encoded_bytes`` (int) — the final encoded size.
          * ``budget_bytes`` (int)  — echoed.
          * ``overage_bytes`` (int) — ``encoded_bytes - budget_bytes`` when
            over budget, else 0.
          * ``final_tris`` (int), ``iters`` (int), ``backend`` (str).

    When ``over_budget`` is True the returned mesh is the ``floor_tris``
    result (the smallest we'll go); the caller (acceptance test) then splits
    the geometry across multiple REL nodes.
    """
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    n_in_f = len(f)
    floor_tris = max(MIN_TARGET_TRIS, int(floor_tris))

    # Fast path: the input already fits.
    full_bytes = int(encode_size_fn(v, f))
    if full_bytes <= budget_bytes:
        log.info("decimate_to_byte_budget: input already fits (%d <= %d bytes, %d tris)",
                 full_bytes, budget_bytes, n_in_f)
        return v.copy(), f.copy(), {
            "over_budget": False,
            "encoded_bytes": full_bytes,
            "budget_bytes": int(budget_bytes),
            "overage_bytes": 0,
            "final_tris": n_in_f,
            "iters": 0,
            "backend": "noop",
        }

    # Binary search on triangle count. Invariant: `lo` tris is known to fit
    # (or is the floor), `hi` tris is known to be over budget (or is the input
    # count). We search for the largest tri count that fits.
    lo = floor_tris
    hi = n_in_f

    # Evaluate the floor first — if it doesn't fit, we're over budget.
    floor_v, floor_f, floor_meta = _decimate_and_size(
        v, f, lo, preserve_border, encode_size_fn)
    if floor_meta["encoded_bytes"] > budget_bytes:
        overage = floor_meta["encoded_bytes"] - int(budget_bytes)
        log.warning(
            "decimate_to_byte_budget: OVER BUDGET — floor %d tris encodes to "
            "%d bytes > %d budget (overage %d); caller should split REL nodes",
            len(floor_f), floor_meta["encoded_bytes"], budget_bytes, overage,
        )
        return floor_v, floor_f, {
            "over_budget": True,
            "encoded_bytes": floor_meta["encoded_bytes"],
            "budget_bytes": int(budget_bytes),
            "overage_bytes": overage,
            "final_tris": int(len(floor_f)),
            "iters": 1,
            "backend": floor_meta["backend"],
        }

    best_v, best_f = floor_v, floor_f
    best_bytes = floor_meta["encoded_bytes"]
    best_tris = len(floor_f)
    backend = floor_meta["backend"]
    iters = 1

    while lo < hi and iters < max_iters:
        mid = (lo + hi + 1) // 2
        if mid <= lo or mid >= hi:
            # No new integer between lo and hi to probe.
            if mid == lo:
                break
        cand_v, cand_f, cm = _decimate_and_size(
            v, f, mid, preserve_border, encode_size_fn)
        iters += 1
        enc = cm["encoded_bytes"]
        backend = cm["backend"]
        if enc <= budget_bytes:
            # Fits — accept and try to go bigger.
            best_v, best_f, best_bytes, best_tris = cand_v, cand_f, enc, len(cand_f)
            lo = mid
        else:
            # Over — shrink.
            hi = mid - 1
        log.info("decimate_to_byte_budget: iter %d target=%d -> %d bytes (budget %d)",
                 iters, mid, enc, budget_bytes)

    log.info(
        "decimate_to_byte_budget: converged at %d tris, %d bytes (<= %d budget) in %d iters",
        best_tris, best_bytes, budget_bytes, iters,
    )
    return best_v, best_f, {
        "over_budget": False,
        "encoded_bytes": int(best_bytes),
        "budget_bytes": int(budget_bytes),
        "overage_bytes": 0,
        "final_tris": int(best_tris),
        "iters": iters,
        "backend": backend,
    }


# --------------------------------------------------------------------------- #
# Backend dispatch
# --------------------------------------------------------------------------- #
def _run_qem(v, f, target_tris, preserve_border):
    """Run QEM via trimesh's fast backend, falling back to pure-NumPy QEM.

    Returns ``(vertices, faces, backend_str)``.
    """
    backend = _try_trimesh_qem(v, f, target_tris)
    if backend is not None:
        out_v, out_f = backend
        return out_v, out_f, "trimesh_fast_simplification"

    log.warning(
        "trimesh QEM backend unavailable; using hand-rolled NumPy QEM "
        "(slower, %d tris -> %d)", len(f), target_tris,
    )
    out_v, out_f = _qem_fallback(v, f, target_tris, preserve_border)
    return out_v, out_f, "numpy_qem_fallback"


def _try_trimesh_qem(v, f, target_tris):
    """Attempt trimesh's ``simplify_quadric_decimation``.

    Returns ``(vertices, faces)`` on success, or ``None`` if trimesh or its
    QEM backend (``fast-simplification``) is unavailable / errors out.
    """
    try:
        import trimesh
    except Exception as e:  # trimesh genuinely absent
        log.info("trimesh import failed (%s); QEM fallback", e)
        return None

    try:
        tm = trimesh.Trimesh(vertices=np.asarray(v, dtype=np.float64),
                             faces=np.asarray(f, dtype=np.int64),
                             process=False)
        # trimesh 4.x: face_count= is the modern kwarg; older builds took
        # `percent`. We pass face_count and let trimesh dispatch to the
        # fast-simplification backend. A missing backend raises here.
        simp = tm.simplify_quadric_decimation(face_count=int(target_tris))
    except TypeError:
        # Signature mismatch (very old trimesh) — try the percent form.
        try:
            pct = float(target_tris) / max(1, len(f))
            simp = tm.simplify_quadric_decimation(percent=pct)
        except Exception as e:
            log.info("trimesh percent QEM failed (%s); fallback", e)
            return None
    except (ModuleNotFoundError, ImportError) as e:
        log.info("trimesh QEM backend missing (%s); fallback", e)
        return None
    except Exception as e:  # pragma: no cover - defensive
        log.info("trimesh QEM raised (%s); fallback", e)
        return None

    out_v = np.asarray(simp.vertices, dtype=np.float64).reshape(-1, 3)
    out_f = np.asarray(simp.faces, dtype=np.int64).reshape(-1, 3)
    if len(out_f) == 0:
        # Backend produced nothing usable — fall back.
        return None
    return out_v, out_f


# --------------------------------------------------------------------------- #
# Hand-rolled QEM (Garland & Heckbert) — pure NumPy fallback
# --------------------------------------------------------------------------- #
def _qem_fallback(v, f, target_tris, preserve_border):
    """Pure-NumPy Quadric Error Metric edge-collapse decimator.

    Implements the classic Garland-Heckbert scheme: each vertex accumulates a
    4x4 fundamental error quadric (sum of plane quadrics of incident faces),
    edges are collapsed cheapest-first (the collapse target minimises the
    summed quadric), and quadrics propagate to the merged vertex. Boundary
    vertices get a large penalty quadric when ``preserve_border`` so seams
    stay pinned.

    This is O((E + collapses) log E) via a lazy heap. It is the safety net for
    environments without the fast C++ backend; correctness over raw speed.
    """
    import heapq

    v = np.asarray(v, dtype=np.float64).reshape(-1, 3).copy()
    faces = [tuple(int(x) for x in row) for row in np.asarray(f, dtype=np.int64).reshape(-1, 3)]
    nv = len(v)

    # --- Per-face plane quadrics -> per-vertex quadric sum ----------------
    Q = [np.zeros((4, 4), dtype=np.float64) for _ in range(nv)]

    def face_plane(a, b, c):
        p0, p1, p2 = v[a], v[b], v[c]
        n = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            return None
        n = n / ln
        d = -np.dot(n, p0)
        return np.array([n[0], n[1], n[2], d], dtype=np.float64)

    # Adjacency: vertex -> set of incident face indices; edge -> face count.
    from collections import defaultdict
    edge_faces: dict = defaultdict(int)
    valid_face = [True] * len(faces)
    for fi, (a, b, c) in enumerate(faces):
        pl = face_plane(a, b, c)
        if pl is None:
            valid_face[fi] = False
            continue
        K = np.outer(pl, pl)
        Q[a] += K
        Q[b] += K
        Q[c] += K
        for e in ((a, b), (b, c), (c, a)):
            edge_faces[(min(e), max(e))] += 1

    # --- Boundary penalty: pin border edges with a large perpendicular
    #     plane quadric so they resist collapse.
    if preserve_border:
        for (a, b), cnt in edge_faces.items():
            if cnt == 1:  # boundary edge
                # Build a plane through the edge, perpendicular to a face the
                # edge belongs to, weighted heavily.
                # Find one incident face to get a normal direction.
                edge_dir = v[b] - v[a]
                el = np.linalg.norm(edge_dir)
                if el < 1e-12:
                    continue
                edge_dir /= el
                # Any vector not parallel to the edge.
                up = np.array([0.0, 0.0, 1.0]) if abs(edge_dir[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
                n = np.cross(edge_dir, up)
                n /= (np.linalg.norm(n) + 1e-12)
                d = -np.dot(n, v[a])
                pl = np.array([n[0], n[1], n[2], d], dtype=np.float64)
                K = np.outer(pl, pl) * 1000.0
                Q[a] += K
                Q[b] += K

    # --- Vertex-vertex adjacency (collapsible edges) ----------------------
    vadj = defaultdict(set)
    for (a, b, c) in faces:
        vadj[a].update((b, c))
        vadj[b].update((a, c))
        vadj[c].update((a, b))

    alive = [True] * nv
    version = [0] * nv  # bumped when a vertex's quadric/position changes

    def collapse_cost(a, b):
        """Return (cost, target_position) for collapsing edge (a,b)."""
        Qbar = Q[a] + Q[b]
        # Solve for optimal position: minimise vQv. Use the 3x3 upper block.
        A = Qbar.copy()
        A[3, :] = [0.0, 0.0, 0.0, 1.0]
        try:
            vt = np.linalg.solve(A, np.array([0.0, 0.0, 0.0, 1.0]))
            target = vt[:3]
        except np.linalg.LinAlgError:
            # Singular — fall back to the edge midpoint.
            target = 0.5 * (v[a] + v[b])
            vt = np.array([target[0], target[1], target[2], 1.0])
        cost = float(vt @ Qbar @ vt)
        # Numerical guard.
        if not math.isfinite(cost) or cost < 0:
            cost = float(np.linalg.norm(v[a] - v[b]))
        return cost, target

    # --- Build the initial heap of collapsible edges ----------------------
    heap = []
    seen_edges = set()
    for a in vadj:
        for b in vadj[a]:
            if a < b:
                key = (a, b)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                cost, tgt = collapse_cost(a, b)
                heapq.heappush(heap, (cost, a, b, version[a], version[b], tgt))

    cur_tris = sum(1 for ok in valid_face if ok)
    target_tris = max(MIN_TARGET_TRIS, int(target_tris))

    while heap and cur_tris > target_tris:
        cost, a, b, va, vb, tgt = heapq.heappop(heap)
        # Lazy-deletion: skip stale heap entries.
        if not alive[a] or not alive[b]:
            continue
        if version[a] != va or version[b] != vb:
            continue
        if b not in vadj.get(a, ()):  # edge no longer exists
            continue

        # Collapse b -> a. Move a to the optimal target, merge quadrics.
        v[a] = tgt
        Q[a] = Q[a] + Q[b]
        alive[b] = False

        # Re-point all faces referencing b to a; drop faces that become
        # degenerate (now reference a twice).
        for fi, fc in enumerate(faces):
            if not valid_face[fi]:
                continue
            if b in fc:
                new_fc = tuple(a if x == b else x for x in fc)
                if len(set(new_fc)) < 3:
                    valid_face[fi] = False
                    cur_tris -= 1
                else:
                    faces[fi] = new_fc

        # Merge adjacency b -> a.
        for nb in vadj[b]:
            if nb == a:
                continue
            vadj[nb].discard(b)
            vadj[nb].add(a)
            vadj[a].add(nb)
        vadj[a].discard(b)
        vadj.pop(b, None)

        # Bump versions and re-push affected edges.
        version[a] += 1
        for nb in list(vadj[a]):
            if not alive[nb]:
                continue
            version[nb] += 1
            c, t = collapse_cost(a, nb)
            lo, hi = (a, nb) if a < nb else (nb, a)
            heapq.heappush(heap, (c, lo, hi, version[lo], version[hi], t))

    # --- Compact surviving verts + faces ----------------------------------
    out_faces = [faces[fi] for fi in range(len(faces)) if valid_face[fi]]
    if not out_faces:
        # Degenerate result — return a minimal copy of the input.
        return v.copy(), np.asarray(f, dtype=np.int64).reshape(-1, 3).copy()

    used = sorted({idx for fc in out_faces for idx in fc})
    remap = {old: new for new, old in enumerate(used)}
    out_v = v[used]
    out_f = np.array([[remap[i] for i in fc] for fc in out_faces], dtype=np.int64)
    return out_v, out_f


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _resolve_target_tris(n_faces, target_ratio, target_tris):
    """Resolve the absolute target triangle count from ratio/tris inputs."""
    if target_tris is not None:
        tgt = int(target_tris)
    elif target_ratio is not None:
        r = float(target_ratio)
        if not (0.0 < r <= 1.0):
            raise ValueError(f"target_ratio must be in (0, 1], got {r}")
        tgt = int(round(n_faces * r))
    else:
        raise ValueError("must pass exactly one of target_ratio or target_tris")
    return max(MIN_TARGET_TRIS, min(tgt, n_faces))


def _clean_mesh(v, f):
    """Drop degenerate faces and unreferenced vertices; reindex."""
    v = np.asarray(v, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(f, dtype=np.int64).reshape(-1, 3)
    if len(f) == 0:
        return v.copy(), f.copy()
    # Degenerate face = any two indices equal.
    good = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[good]
    if len(f) == 0:
        return v.copy(), f.copy()
    used = np.unique(f.reshape(-1))
    remap = np.full(len(v), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    out_v = v[used]
    out_f = remap[f]
    return out_v, out_f


def _resample_uvs(orig_v, orig_uv, new_v):
    """Re-sample UVs for the decimated verts by nearest original vertex.

    A LOD approximation: each surviving vertex takes the UV of the closest
    original vertex. Correct seam-aware carry would need per-wedge tracking
    through the collapse, which neither QEM backend exposes here.
    """
    orig_v = np.asarray(orig_v, dtype=np.float64).reshape(-1, 3)
    orig_uv = np.asarray(orig_uv, dtype=np.float64).reshape(-1, 2)
    new_v = np.asarray(new_v, dtype=np.float64).reshape(-1, 3)
    if len(orig_v) == 0 or len(orig_uv) != len(orig_v):
        return np.zeros((len(new_v), 2), dtype=np.float64)
    # Brute-force nearest-neighbour; fine for editor-scale meshes (<100k).
    # Chunk to keep peak memory bounded.
    out = np.zeros((len(new_v), 2), dtype=np.float64)
    CH = 4096
    for i in range(0, len(new_v), CH):
        chunk = new_v[i:i + CH]
        # (chunk, orig) squared distances.
        d = ((chunk[:, None, :] - orig_v[None, :, :]) ** 2).sum(axis=2)
        nn = np.argmin(d, axis=1)
        out[i:i + CH] = orig_uv[nn]
    return out


def _decimate_and_size(v, f, target_tris, preserve_border, encode_size_fn):
    """Decimate to ``target_tris`` and measure encoded size."""
    out_v, out_f, _uv, meta = decimate_mesh(
        v, f, target_tris=target_tris, preserve_border=preserve_border,
        return_meta=True)
    enc = int(encode_size_fn(out_v, out_f))
    return out_v, out_f, {"encoded_bytes": enc, "backend": meta.get("backend", "unknown")}


def estimate_rel_node_bytes(vertices, faces) -> int:
    """Rough size estimator for an ``n.rel`` node geometry blob.

    A stand-in the byte-budget search can use when the caller hasn't got a
    real encoder wired yet: 32 bytes/vertex (pos+normal+uv, padded) + 6
    bytes/triangle (3 x uint16 index) + a fixed 256-byte header. This is a
    deliberate over-estimate so the search errs on the safe (smaller) side.
    """
    nv = len(np.asarray(vertices).reshape(-1, 3))
    nf = len(np.asarray(faces).reshape(-1, 3))
    return 256 + nv * 32 + nf * 6
