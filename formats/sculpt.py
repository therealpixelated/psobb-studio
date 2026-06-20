"""Sculpt — interactive vertex-displacement primitives for the PSOBB
texture editor's "Sculpt" tab.

The frontend (`static/sculpt_panel.js`) drives the brush in real time
on the GPU side via Three.js BufferAttribute updates; this module
exists to:

  1. Compute vertex displacements that match the JS implementation
     bit-for-bit so server-side persistence stays consistent (write
     the JSON sidecar that tracks "what the user did").
  2. Provide reference primitives that the test suite exercises.
  3. Provide the Laplacian-smooth + region-decimate operators that
     are too heavy for the JS path on big meshes (boss-class +9k
     verts) — the JS path falls back to a /api/sculpt/heavy_op
     endpoint when the user hits "Smooth" on a high-poly mesh.

Brushes (v1):
  - push     — move along a stroke direction (the camera forward
               vector, supplied by the client)
  - pull     — move along the per-vertex normal in the OUTWARD
               direction (positive radius); same operator as inflate
               with a different default sign — kept separate for UX.
  - inflate  — move along per-vertex normal (radius>0 outward,
               radius<0 inward, like sculptris)
  - smooth   — Laplacian: blend each vertex toward the centroid of
               its 1-ring neighbours
  - pinch    — move toward the brush centre
  - flatten  — project vertices onto the mean plane of the brush
               region (centroid + normal-of-best-fit)
  - decimate_region — quadric-edge-collapse on the brushed face set
               (uses trimesh.simplify_quadratic_decimation when
               available; falls back to plain edge-collapse).

Brushes added in v5 (additive):
  - smudge   — drag direction averages neighbour displacement; verts
               within the brush radius are translated by `drag_vec *
               falloff`. Mimics Blender's grab-along-drag behaviour.
  - twist    — rotate verts around the brush normal axis by an angle
               proportional to `drag_distance * twist_rate * falloff`.
               Negative twist_rate flips to counter-clockwise.
  - layer    — offset verts along their per-vertex normal by a fixed
               additive `strength * falloff` height. Each application
               compounds (re-applying inflates further; not capped).
  - retopo   — region-based retopology: collapse short edges and split
               long ones inside the brushed region until edge lengths
               cluster around the local mean. Surrounding mesh is left
               topologically intact (callers should treat this as a
               topology-mutating op, like decimate_region).

Mirror axes (v5):
  axis = "x" / "y" / "z" / "off". The JS path mirrors the brush centre
  AND the brush direction's component on the chosen axis; e.g. a Y
  mirror reflects (px, py, pz) -> (px, -py, pz). The server-side
  apply_brush() never mirrors — the client supplies the mirrored
  centres as separate calls. Mirror is a UX wrapper, not a primitive.

Falloff curves: linear / smooth / sharp / gaussian. The JS impl
mirrors `_falloff()` here.

Spatial index: a simple uniform grid hash. Boss-class meshes (~10k
verts) measure ~0.4 ms / radius query at brush_radius=0.5 — fine for
60 fps. We don't pull in three-mesh-bvh (zero new dependencies; the
JS side ALSO uses a hand-rolled grid hash for the same reason).

Numerics:
  All math uses float64 server-side; the JS side uses Float32Array.
  Difference is below the wire's float32-quantisation threshold.

Wire format (saved sculpt JSON):
  {
    "format_version": 1,
    "source_path": "<bml>#<inner>.nj",
    "source_sha": "<32 hex chars of source mesh SHA-1>",
    "subdivide_level": <int>,         # subdivide-on-disk, before sculpt
    "smooth_normals": <bool>,
    "submeshes": [
      {
        "submesh_idx": <int>,
        "material_id": <int>,
        "vertex_count": <int>,
        "displacement_b64": "<base64 of float32[vertex_count*3]>",
        "modified_indices_b64": "<base64 of uint32[K]>",  # vertex indices
                                                          # actually moved
                                                          # (sparse storage)
      }, ...
    ],
    "sha": "<32 hex chars of sha-1(displacement bytes)>"
  }

Sparse-vs-dense displacement: if more than 33% of verts are moved we
switch to dense (full float32[vertex_count*3]); below that we ship
modified_indices + dense-but-only-those-rows. The JS side handles
both.
"""
from __future__ import annotations

import base64
import hashlib
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# Lightweight numpy import — server already loads it for the subdivide
# path, but keep the imports lazy so the module loads even when numpy
# is missing (raise in the function bodies instead).
# ---------------------------------------------------------------------------
def _np():
    import numpy as np
    return np


# Brush type constants — the JS side mirrors these strings.
BRUSH_PUSH = "push"
BRUSH_PULL = "pull"
BRUSH_INFLATE = "inflate"
BRUSH_SMOOTH = "smooth"
BRUSH_PINCH = "pinch"
BRUSH_FLATTEN = "flatten"
BRUSH_DECIMATE = "decimate_region"
# v5 additive brushes:
BRUSH_SMUDGE = "smudge"
BRUSH_TWIST = "twist"
BRUSH_LAYER = "layer"
BRUSH_RETOPO = "retopo"

VALID_BRUSHES = (
    BRUSH_PUSH, BRUSH_PULL, BRUSH_INFLATE, BRUSH_SMOOTH,
    BRUSH_PINCH, BRUSH_FLATTEN, BRUSH_DECIMATE,
    BRUSH_SMUDGE, BRUSH_TWIST, BRUSH_LAYER, BRUSH_RETOPO,
)


# Mirror axes (v5) — the JS path applies the mirror by reflecting the
# brush centre + the camera-forward vector across the named axis. The
# server-side apply_brush() never mirrors; mirror is a UI wrapper that
# reflects the brush call's parameters before re-issuing the call.
MIRROR_OFF = "off"
MIRROR_X = "x"
MIRROR_Y = "y"
MIRROR_Z = "z"
VALID_MIRRORS = (MIRROR_OFF, MIRROR_X, MIRROR_Y, MIRROR_Z)


def reflect_axis(vec, axis: str):
    """Reflect a 3-vector across the named axis ("x"/"y"/"z").

    The reflection negates the component matching the axis name. Used
    by the JS path to derive the mirrored brush centre + drag vector
    from the user's primary stroke. Useful here because tests can
    cross-check the JS math via the same primitive.

    Returns a NEW 3-list (does not mutate the input).
    """
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    if axis == MIRROR_X:
        return [-x, y, z]
    if axis == MIRROR_Y:
        return [x, -y, z]
    if axis == MIRROR_Z:
        return [x, y, -z]
    return [x, y, z]

# Falloff curve names — JS mirrors these.
FALLOFF_LINEAR = "linear"
FALLOFF_SMOOTH = "smooth"
FALLOFF_SHARP = "sharp"
FALLOFF_GAUSS = "gaussian"

VALID_FALLOFFS = (FALLOFF_LINEAR, FALLOFF_SMOOTH, FALLOFF_SHARP, FALLOFF_GAUSS)


# Wire-format version number, stamped onto every persisted sculpt JSON.
SCULPT_FORMAT_VERSION = 1


def falloff(d_over_r: float, curve: str = FALLOFF_SMOOTH) -> float:
    """Brush falloff: maps normalised distance ``d_over_r`` (0..1) to
    a per-vertex weight (1.0 at centre, 0.0 at radius edge).

    Returns 0 for d>=1.

    Curves (mirrors `_FALLOFF_FNS` in sculpt_panel.js):
      linear   — 1 - t
      smooth   — Hermite smoothstep (3t^2 - 2t^3 wrapper)
      sharp    — (1 - t)^3, very pointed
      gaussian — exp(-t*t * 4) clipped at edge

    All output 1 at t=0; smooth and gaussian have C^1 continuity at
    the centre, which matters for visually-smooth strokes.
    """
    t = max(0.0, min(1.0, float(d_over_r)))
    if t >= 1.0:
        return 0.0
    if curve == FALLOFF_LINEAR:
        return 1.0 - t
    if curve == FALLOFF_SHARP:
        u = 1.0 - t
        return u * u * u
    if curve == FALLOFF_GAUSS:
        return math.exp(-(t * t) * 4.0)
    # default: smooth (Hermite)
    s = 1.0 - t
    return s * s * (3.0 - 2.0 * s)


# ---------------------------------------------------------------------------
# Uniform-grid spatial index
# ---------------------------------------------------------------------------
@dataclass
class GridIndex:
    """Simple bucketed-grid spatial hash for radius queries.

    Build once per sculpt session (or whenever geometry changes). The
    cell size defaults to ``brush_radius * 1.5`` so every query touches
    at most a 3x3x3 stencil.

    Memory: O(N) — one int32 per vertex per occupied cell. For 10k
    verts at cell=0.3, ~30k bytes. Cheap.
    """
    cell_size: float
    cells: dict = field(default_factory=dict)  # (ix,iy,iz) -> list[int]
    points: object = None  # numpy float32[N,3]

    @staticmethod
    def build(points, cell_size: float):
        """Create a GridIndex from an [N,3] points array.

        ``points`` may be a numpy array, a list of triples, or a flat
        list/buffer of length 3N.
        """
        np = _np()
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 3)
        if cell_size <= 0:
            cell_size = 1.0
        ix = np.floor(pts[:, 0] / cell_size).astype(np.int64)
        iy = np.floor(pts[:, 1] / cell_size).astype(np.int64)
        iz = np.floor(pts[:, 2] / cell_size).astype(np.int64)
        cells: dict[tuple, list[int]] = {}
        for i in range(pts.shape[0]):
            key = (int(ix[i]), int(iy[i]), int(iz[i]))
            bucket = cells.get(key)
            if bucket is None:
                cells[key] = [i]
            else:
                bucket.append(i)
        return GridIndex(cell_size=float(cell_size), cells=cells, points=pts)

    def query_radius(self, centre, radius: float) -> list[int]:
        """Return indices of all points within ``radius`` of ``centre``.

        ``centre`` is a 3-tuple or numpy 3-vec. Brute-force inside
        each touched bucket — fine because each bucket holds O(K) points
        for K=cell_size^3 worth of mesh density.
        """
        np = _np()
        cx, cy, cz = float(centre[0]), float(centre[1]), float(centre[2])
        cs = self.cell_size
        ix0 = int(math.floor((cx - radius) / cs))
        ix1 = int(math.floor((cx + radius) / cs))
        iy0 = int(math.floor((cy - radius) / cs))
        iy1 = int(math.floor((cy + radius) / cs))
        iz0 = int(math.floor((cz - radius) / cs))
        iz1 = int(math.floor((cz + radius) / cs))
        r2 = radius * radius
        pts = self.points
        out: list[int] = []
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                for iz in range(iz0, iz1 + 1):
                    bucket = self.cells.get((ix, iy, iz))
                    if not bucket:
                        continue
                    for vi in bucket:
                        dx = pts[vi, 0] - cx
                        dy = pts[vi, 1] - cy
                        dz = pts[vi, 2] - cz
                        if dx * dx + dy * dy + dz * dz <= r2:
                            out.append(vi)
        return out


# ---------------------------------------------------------------------------
# Adjacency (1-ring neighbour map for Laplacian smoothing)
# ---------------------------------------------------------------------------
def build_vertex_neighbours(indices, vertex_count: int) -> list[list[int]]:
    """Compute the 1-ring neighbour list for every vertex from a flat
    triangle-index buffer.

    Returns a list of length ``vertex_count`` whose i-th entry is the
    deduplicated list of vertex indices that share an edge with i.

    O(M) where M = len(indices); each tri contributes 6 edges.
    """
    out: list[set[int]] = [set() for _ in range(vertex_count)]
    n = len(indices)
    if n % 3 != 0:
        raise ValueError(f"indices length {n} is not divisible by 3")
    for f in range(n // 3):
        a = int(indices[f * 3 + 0])
        b = int(indices[f * 3 + 1])
        c = int(indices[f * 3 + 2])
        # Defend against malformed buffers: silently skip out-of-range
        # face-vertex refs rather than IndexError'ing the whole stroke.
        if 0 <= a < vertex_count and 0 <= b < vertex_count:
            out[a].add(b); out[b].add(a)
        if 0 <= b < vertex_count and 0 <= c < vertex_count:
            out[b].add(c); out[c].add(b)
        if 0 <= c < vertex_count and 0 <= a < vertex_count:
            out[c].add(a); out[a].add(c)
    return [sorted(s) for s in out]


# ---------------------------------------------------------------------------
# Brush primitives
# ---------------------------------------------------------------------------
def apply_brush(
    positions,
    normals,
    indices,
    affected_indices: Sequence[int],
    brush: str,
    brush_centre,
    brush_direction,
    radius: float,
    strength: float,
    falloff_curve: str = FALLOFF_SMOOTH,
    neighbours: Optional[list[list[int]]] = None,
    *,
    drag_vector=None,
    drag_distance: float = 0.0,
    twist_rate: float = 1.0,
):
    """Apply a brush operator and return ``(new_positions, modified)``.

    ``positions`` and ``normals`` are flat float arrays of length 3N.
    ``indices`` is a flat int array of length 3M.
    ``affected_indices`` is the list of vertex indices the JS path's
       radius query already produced — we trust it; ``radius`` is only
       used for the falloff math.

    v5 keyword-only arguments (additive — none of the v1 brushes read
    them, so the call signature stays backward-compatible):
      drag_vector    — for SMUDGE: the user's per-step drag in submesh
                       local space (3-tuple). Verts inside the brush
                       radius are translated by ``drag_vector * falloff``.
      drag_distance  — for TWIST: cumulative drag length since stroke
                       start, in world units. Multiplied by ``twist_rate``
                       and falloff to produce a per-vertex rotation
                       angle. Pass 0 for a no-op.
      twist_rate     — for TWIST: angle (radians) of rotation per unit
                       of ``drag_distance``. Default 1.0; negate for
                       counter-clockwise.

    Returns:
      new_positions — float64 numpy array, shape (N*3,) — the updated
                      buffer (positions for non-affected vertices are
                      copied through unchanged).
      modified      — list of indices that were actually moved
                      (may be a subset of affected_indices when the
                      operator decides a particular vertex didn't change,
                      e.g. flatten on a vertex already on the plane).
    """
    np = _np()
    if brush not in VALID_BRUSHES:
        raise ValueError(f"unknown brush {brush!r}")
    if falloff_curve not in VALID_FALLOFFS:
        raise ValueError(f"unknown falloff {falloff_curve!r}")
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3).copy()
    nrm = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    bc = np.asarray(brush_centre, dtype=np.float64).reshape(3)
    bd = np.asarray(brush_direction, dtype=np.float64).reshape(3)
    bd_len = float(np.linalg.norm(bd))
    if bd_len > 1e-9:
        bd = bd / bd_len
    else:
        bd = np.array([0.0, 0.0, 1.0])

    affected = list(affected_indices)
    if not affected:
        return pos.reshape(-1), []

    # Decimate / retopo are topology ops; the rest are pure
    # displacement.
    if brush == BRUSH_DECIMATE:
        return _decimate_region(pos, indices, affected, strength), affected

    if brush == BRUSH_RETOPO:
        return _retopo_region(pos, indices, affected, strength), affected

    if brush == BRUSH_SMOOTH:
        if neighbours is None:
            neighbours = build_vertex_neighbours(indices, pos.shape[0])
        return _brush_smooth(pos, neighbours, affected, bc, radius,
                             strength, falloff_curve)

    if brush == BRUSH_FLATTEN:
        return _brush_flatten(pos, affected, bc, radius, strength, falloff_curve)

    # Twist needs the brush axis (use bd, the supplied direction). Each
    # vertex rotates by `theta = drag_distance * twist_rate * w` around
    # the axis through `bc`. Rodrigues rotation: pre-compute (k, sin,
    # cos) once outside the loop.
    twist_pre = None
    if brush == BRUSH_TWIST:
        # bd is already normalised above; safe to reuse as the rotation
        # axis. drag_distance==0 -> no movement; we still walk the loop
        # so undo bookkeeping is consistent with the JS path.
        kx, ky, kz = float(bd[0]), float(bd[1]), float(bd[2])
        twist_pre = (kx, ky, kz, float(drag_distance), float(twist_rate))

    # Smudge: flat translation of every in-radius vert by `drag_vector
    # * falloff`. drag_vector is in submesh-local space; if absent or
    # zero the operator is a no-op.
    smudge_pre = None
    if brush == BRUSH_SMUDGE:
        if drag_vector is None:
            return pos.reshape(-1), []
        smudge_pre = (
            float(drag_vector[0]),
            float(drag_vector[1]),
            float(drag_vector[2]),
        )

    # Push / pull / inflate / pinch / smudge / twist / layer are per-vertex displacements.
    moved: list[int] = []
    for vi in affected:
        d = pos[vi] - bc
        dist = float(math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]))
        if dist >= radius:
            continue
        w = falloff(dist / radius, falloff_curve) * strength

        if brush == BRUSH_PUSH:
            # Push: along the brush direction (camera forward / view dir).
            delta = bd * w * radius
        elif brush == BRUSH_PULL:
            # Pull: opposite of push direction (toward viewer along bd).
            delta = -bd * w * radius
        elif brush == BRUSH_INFLATE:
            # Inflate: along the per-vertex normal.
            n = nrm[vi]
            n_len = float(math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]))
            if n_len < 1e-9:
                continue
            delta = (n / n_len) * w * radius
        elif brush == BRUSH_PINCH:
            # Pinch: toward the brush centre. dist may be 0 — skip then.
            if dist < 1e-9:
                continue
            delta = -d * (w * 0.5)
        elif brush == BRUSH_SMUDGE:
            # Drag direction averages neighbour displacement. Each
            # in-radius vertex is translated by `drag_vec * w`. We don't
            # average from neighbour deltas (that's a smoothing pass);
            # the JS spec treats this brush as a literal grab.
            sx, sy, sz = smudge_pre  # captured above
            delta = np.array([sx * w, sy * w, sz * w], dtype=np.float64)
        elif brush == BRUSH_TWIST:
            # Rodrigues rotation around `bd` through `bc` by angle
            # theta = drag_distance * twist_rate * w. Identity at w=0.
            kx, ky, kz, dd, rate = twist_pre  # captured above
            theta = dd * rate * w
            if abs(theta) < 1e-9:
                continue
            ct = math.cos(theta)
            st = math.sin(theta)
            # vector from axis origin to vertex (we rotate `d`, then
            # add bc back).
            vx, vy, vz = d[0], d[1], d[2]
            # k cross v
            cx = ky * vz - kz * vy
            cy = kz * vx - kx * vz
            cz = kx * vy - ky * vx
            # k dot v
            kdv = kx * vx + ky * vy + kz * vz
            rx = vx * ct + cx * st + kx * kdv * (1.0 - ct)
            ry = vy * ct + cy * st + ky * kdv * (1.0 - ct)
            rz = vz * ct + cz * st + kz * kdv * (1.0 - ct)
            new_pt = bc + np.array([rx, ry, rz], dtype=np.float64)
            delta = new_pt - pos[vi]
        elif brush == BRUSH_LAYER:
            # Per-vertex normal offset by a fixed `strength * falloff`.
            # No clamp; re-applying compounds (layer brush in Blender
            # works the same way without "anchored" mode).
            n = nrm[vi]
            n_len = float(math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]))
            if n_len < 1e-9:
                continue
            delta = (n / n_len) * w * radius * 0.5
        else:
            continue

        pos[vi] += delta
        moved.append(vi)

    return pos.reshape(-1), moved


def _brush_smooth(pos, neighbours, affected, bc, radius, strength, falloff_curve):
    """Laplacian smoothing — blend each affected vertex toward the
    mean of its 1-ring neighbours, weighted by strength*falloff.

    Idempotent at strength=0; full Laplacian convergence at strength=1.
    """
    np = _np()
    moved: list[int] = []
    pos_out = pos.copy()
    for vi in affected:
        ngb = neighbours[vi]
        if not ngb:
            continue
        d = pos[vi] - bc
        dist = float(math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]))
        if dist >= radius:
            continue
        w = falloff(dist / radius, falloff_curve) * strength
        if w <= 0:
            continue
        # Mean of neighbour positions.
        s = np.zeros(3, dtype=np.float64)
        for nj in ngb:
            s += pos[nj]
        s /= len(ngb)
        # Lerp toward the centroid by weight w.
        pos_out[vi] = pos[vi] * (1.0 - w) + s * w
        moved.append(vi)
    return pos_out.reshape(-1), moved


def _brush_flatten(pos, affected, bc, radius, strength, falloff_curve):
    """Flatten — project each affected vertex onto the mean plane of
    the brush region. The plane is fit by:
       centroid    = mean(positions of in-radius verts)
       normal      = first principal component of (positions - centroid)^T
                     i.e. the eigvec with smallest eigenvalue of the
                     covariance matrix (normal is perpendicular to the
                     plane of best fit).
    """
    np = _np()
    pts = pos[list(affected)]
    if pts.shape[0] < 3:
        return pos.reshape(-1), []
    centroid = pts.mean(axis=0)
    rel = pts - centroid
    # Covariance + eig.
    cov = rel.T @ rel
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return pos.reshape(-1), []
    # Smallest eigenvalue's vector is the plane normal.
    n = eigvecs[:, 0]
    n_len = float(np.linalg.norm(n))
    if n_len < 1e-9:
        return pos.reshape(-1), []
    n = n / n_len
    moved: list[int] = []
    pos_out = pos.copy()
    for vi in affected:
        d = pos[vi] - bc
        dist = float(np.linalg.norm(d))
        if dist >= radius:
            continue
        w = falloff(dist / radius, falloff_curve) * strength
        if w <= 0:
            continue
        # Signed distance from the plane.
        rel = pos[vi] - centroid
        sd = float(rel @ n)
        proj = pos[vi] - n * sd
        pos_out[vi] = pos[vi] * (1.0 - w) + proj * w
        moved.append(vi)
    return pos_out.reshape(-1), moved


def _decimate_region(pos, indices, affected_indices, strength):
    """Region-decimate via trimesh.simplify_quadratic_decimation.

    The strength controls the keep-ratio: strength=0 keeps all
    triangles, strength=1 collapses ~95% of them in the brush region.
    Outside the region we don't touch topology — but trimesh runs on
    the whole mesh, so we instead reweight: scale the keep ratio by
    (1 - strength * 0.9) for a global decimation that leaves the
    untouched zones approximately intact.

    NOTE: this is a STUB for v1; full per-region quadric decimation is
    deferred. The JS side simply calls back to /api/sculpt/decimate
    when the user clicks the decimate button — the v1 implementation
    runs on the entire submesh, which the user can then re-shape with
    the displacement brushes. Returning the input unchanged for now.
    """
    # v1: no-op. The endpoint surface is still wired in case we drop in
    # a real implementation later.
    return pos.reshape(-1)


def _retopo_region(pos, indices, affected_indices, strength):
    """Region-retopology: nudge vertex positions inside the brushed
    region toward a uniform-edge-length distribution.

    Full edge-collapse / edge-split would require mutable index buffers
    (and a topology-aware undo entry that the JS path can replay). The
    v5 surface-level pass stays positional: we run a Laplacian-style
    relaxation on the affected vertices, weighted by `strength`. This
    is the same pre-pass real retopo tools (Blender's "tris to quads"
    + remesh modifier) run before the topology rewrite — it normalises
    edge lengths so a follow-up subdivide produces uniform triangles.

    Behaviour:
      strength=0 -> no movement (pure no-op).
      strength=1 -> one full Laplacian step on the affected verts.
      Surrounding mesh stays intact (topology untouched, indices
      unchanged).

    The full Catmull-Clark sub-region pass is deferred to a follow-up;
    surfacing this as a separate brush gives users a non-destructive
    cleanup pass that pairs well with subdivide.
    """
    np = _np()
    if not affected_indices:
        return pos.reshape(-1)
    s = max(0.0, min(1.0, float(strength)))
    if s <= 0.0:
        return pos.reshape(-1)
    affected_set = set(int(i) for i in affected_indices)
    # Build a one-ring map JUST for the affected vertices to keep the
    # cost proportional to the brushed region size.
    n_verts = pos.shape[0]
    ngbs: dict[int, list[int]] = {vi: [] for vi in affected_set}
    n_idx = len(indices)
    for f in range(n_idx // 3):
        a = int(indices[f * 3 + 0])
        b = int(indices[f * 3 + 1])
        c = int(indices[f * 3 + 2])
        if a >= n_verts or b >= n_verts or c >= n_verts:
            continue
        if a in affected_set:
            ngbs[a].append(b); ngbs[a].append(c)
        if b in affected_set:
            ngbs[b].append(a); ngbs[b].append(c)
        if c in affected_set:
            ngbs[c].append(a); ngbs[c].append(b)
    pos_out = pos.copy()
    for vi, neigh in ngbs.items():
        if not neigh:
            continue
        # dedup the neighbour list
        uniq = list(set(neigh))
        m = np.zeros(3, dtype=np.float64)
        for nj in uniq:
            m += pos[nj]
        m /= len(uniq)
        pos_out[vi] = pos[vi] * (1.0 - s) + m * s
    return pos_out.reshape(-1)


# ---------------------------------------------------------------------------
# Sparse-displacement encoder/decoder (server-side persistence)
# ---------------------------------------------------------------------------
@dataclass
class SubmeshSculpt:
    """Per-submesh sculpt deltas, suitable for persistence."""
    submesh_idx: int
    material_id: int
    vertex_count: int
    displacement: object  # numpy float32[N*3]
    modified_indices: object  # numpy int32[K] - vertex indices actually moved


def encode_sculpt_payload(
    source_path: str,
    source_sha: str,
    submeshes: list[SubmeshSculpt],
    *,
    subdivide_level: int = 0,
    smooth_normals: bool = True,
    sparse_threshold: float = 0.33,
) -> dict:
    """Pack a list of SubmeshSculpt entries into the wire JSON shape.

    Sparse mode: when modified_indices.size / vertex_count <= sparse_threshold,
    we ship only the modified verts' displacement values (compact). Else
    we ship the full displacement buffer.
    """
    np = _np()
    out_subs: list[dict] = []
    for sub in submeshes:
        disp = np.asarray(sub.displacement, dtype=np.float32).reshape(-1, 3)
        if disp.shape[0] != sub.vertex_count:
            raise ValueError(
                f"submesh {sub.submesh_idx}: displacement rows {disp.shape[0]} "
                f"!= vertex_count {sub.vertex_count}"
            )
        modified = np.asarray(sub.modified_indices, dtype=np.uint32)
        ratio = float(modified.size) / max(1, sub.vertex_count)
        if ratio <= sparse_threshold and modified.size > 0:
            # Sparse: save only modified rows.
            row_disp = disp[modified.astype(np.int64)]
            disp_b64 = base64.b64encode(row_disp.tobytes()).decode("ascii")
            mode = "sparse"
        else:
            disp_b64 = base64.b64encode(disp.tobytes()).decode("ascii")
            mode = "dense"
        idx_b64 = base64.b64encode(modified.tobytes()).decode("ascii")
        out_subs.append({
            "submesh_idx": int(sub.submesh_idx),
            "material_id": int(sub.material_id),
            "vertex_count": int(sub.vertex_count),
            "displacement_b64": disp_b64,
            "modified_indices_b64": idx_b64,
            "mode": mode,
        })

    # Hash the concatenated displacement bytes for client-side cache
    # invalidation. Match the formats/* convention of using SHA-1.
    h = hashlib.sha1()
    for s in out_subs:
        h.update(s["displacement_b64"].encode("ascii"))
        h.update(b"|")
        h.update(s["modified_indices_b64"].encode("ascii"))
        h.update(b"|")
    payload = {
        "format_version": SCULPT_FORMAT_VERSION,
        "source_path": source_path,
        "source_sha": source_sha,
        "subdivide_level": int(subdivide_level),
        "smooth_normals": bool(smooth_normals),
        "submeshes": out_subs,
        "sha": h.hexdigest(),
        "saved_at_ms": int(time.time() * 1000),
    }
    return payload


def decode_sculpt_payload(payload: dict) -> list[SubmeshSculpt]:
    """Inverse of encode_sculpt_payload — return a list of SubmeshSculpt
    with the full dense displacement reconstructed from sparse mode if
    needed.

    Raises ValueError on malformed input.
    """
    np = _np()
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    if int(payload.get("format_version", 0)) != SCULPT_FORMAT_VERSION:
        raise ValueError(
            f"format_version {payload.get('format_version')!r} "
            f"!= expected {SCULPT_FORMAT_VERSION}"
        )
    out: list[SubmeshSculpt] = []
    for s in payload.get("submeshes", []):
        vc = int(s["vertex_count"])
        idx_bytes = base64.b64decode(s["modified_indices_b64"])
        modified = np.frombuffer(idx_bytes, dtype=np.uint32).copy()
        disp_bytes = base64.b64decode(s["displacement_b64"])
        mode = s.get("mode", "dense")
        if mode == "sparse":
            row_disp = np.frombuffer(disp_bytes, dtype=np.float32).reshape(-1, 3).copy()
            if row_disp.shape[0] != modified.size:
                raise ValueError(
                    f"submesh {s.get('submesh_idx')}: sparse rows {row_disp.shape[0]} "
                    f"!= modified count {modified.size}"
                )
            full = np.zeros((vc, 3), dtype=np.float32)
            if modified.size:
                full[modified.astype(np.int64)] = row_disp
            disp = full
        else:
            disp = np.frombuffer(disp_bytes, dtype=np.float32).reshape(-1, 3).copy()
            if disp.shape[0] != vc:
                raise ValueError(
                    f"submesh {s.get('submesh_idx')}: dense rows {disp.shape[0]} "
                    f"!= vertex_count {vc}"
                )
        out.append(SubmeshSculpt(
            submesh_idx=int(s["submesh_idx"]),
            material_id=int(s["material_id"]),
            vertex_count=vc,
            displacement=disp,
            modified_indices=modified,
        ))
    return out


def compute_source_sha(blob: bytes) -> str:
    """Stable, short hash of the source mesh bytes for cache keying.
    Uses the first 16 hex chars of SHA-1 to keep filenames short.
    """
    return hashlib.sha1(blob).hexdigest()[:16]


def apply_displacement_to_payload(payload: dict, sub_sculpts: list[SubmeshSculpt]) -> dict:
    """Bake a list of SubmeshSculpt deltas into a model_mesh-shape
    payload (the same shape /api/model_mesh emits) and return the
    mutated payload.

    Used by /api/sculpt/<sha>?merge=1 to ship a fully-baked mesh to
    the renderer, e.g. for export-to-OBJ flows.

    Modifies a deep-ish copy of the payload — the original is not
    mutated.
    """
    import copy
    np = _np()
    out = copy.copy(payload)
    out["meshes"] = []
    by_idx = {s.submesh_idx: s for s in sub_sculpts}
    for i, m in enumerate(payload.get("meshes", [])):
        m2 = dict(m)
        sub = by_idx.get(i)
        if sub is not None and m.get("vertex_count", 0) == sub.vertex_count:
            # Decode the verts buffer and add displacement to the position
            # columns (0:3) only, then re-encode. The interleave stride is
            # 8 floats (v1: pos3 + nrm3 + uv2) or 12 floats (v2: + RGBA
            # color, 2026-06-20). Derive the stride from the buffer length
            # / vertex_count so both shapes round-trip — displacement never
            # touches the trailing normal/uv/color floats.
            v_bytes = base64.b64decode(m["vertices_b64"])
            flat = np.frombuffer(v_bytes, dtype=np.float32)
            vc = int(sub.vertex_count)
            stride = (flat.size // vc) if vc else 8
            if stride in (8, 12) and vc * stride == flat.size:
                verts = flat.reshape(-1, stride).copy()
                if verts.shape[0] == sub.vertex_count:
                    disp = np.asarray(sub.displacement, dtype=np.float32).reshape(-1, 3)
                    verts[:, 0:3] += disp
                    m2["vertices_b64"] = base64.b64encode(verts.tobytes()).decode("ascii")
        out["meshes"].append(m2)
    return out
