"""Rigging — skeleton edits, vertex weights, IK targets for the PSOBB
texture editor's "Rig" tab (sister to paint / sculpt).

The frontend (`static/rig_panel.js`) drives skeleton-widget interaction
in real time on the GPU side via Three.js BufferAttribute updates +
custom Object3D skeleton helpers; this module exists to:

  1. Compute auto-skinning weights (heat-equation / inverse-distance
     falloff) so a fresh mesh can be rigged without hand-painting every
     vertex.
  2. Provide reference IK solvers (FABRIK + 2-bone analytic) that the
     test suite exercises.
  3. Provide weight-normalisation, blur, and bone-pose composition
     utilities the JS path mirrors bit-for-bit.
  4. Persist the rig as a JSON sidecar at
     ``cache/rigs/<safe_path>__<sha>.json`` (same envelope as
     formats/sculpt.py — encoder/decoder pair, base64-packed Float32
     buffers, source-mesh SHA so cache invalidates when geometry
     changes).

Wire format (saved rig JSON, format_version=1):
  {
    "format_version": 1,
    "source_path":  "<bml>#<inner>.nj",
    "source_sha":   "<16 hex>",
    "subdivide_level": <int>,            # subdivide-on-disk before rig
    "skeleton": {
      "bones": [
        {"index": 0, "parent": -1,
         "position": [tx, ty, tz],
         "rotation_bams": [rx, ry, rz],   # int BAMS (Sega Ninja)
         "scale":    [sx, sy, sz],
         "name":     "<optional rename>",
         "eval_flags": <int>,
         "hidden":   <bool>},
        ...
      ]
    },
    "weights": [
      {"submesh_idx": <int>,
       "vertex_count": <int>,
       "indices_b64": "<base64 of int32[N*MAX_INFLUENCES]>",   # bone idx, -1 = empty
       "weights_b64": "<base64 of float32[N*MAX_INFLUENCES]>"
      },
      ...
    ],
    "ik_targets": [
      {"bone_idx":   <int>,            # end-effector
       "chain_length": <int>,          # bones to include in solve
       "target":      [x, y, z],       # world-space target
       "iterations":  <int>,
       "name":        "<optional>"}, ...
    ],
    "sha":           "<16 hex of (skeleton+weights+ik) bytes>",
    "saved_at_ms":   <int>
  }

Numerics:
  Float64 server-side, float32 on the wire. Bone rotations are stored
  in BAMS (raw 16-bit Sega Ninja angles, 0x10000 = 360°) so they
  round-trip with the parser without precision loss; the JS side
  converts to radians via _BAMS_TO_RAD before composing matrices.

Conventions:
  MAX_INFLUENCES — capped at 4 (the PSOBB convention; XJ vertices have
    one bone slot but the rigging layer extends to 4 for the editor's
    weight-paint flow; the NJ encoder downgrades to one slot at compile
    time when phase A4 ships).
"""
from __future__ import annotations

import base64
import hashlib
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence


def _np():
    """Lazy numpy import — keeps the module loadable without numpy
    so isolated unit tests for non-numeric helpers can still run.
    """
    import numpy as np
    return np


# ---------------------------------------------------------------------------
# Constants — JS mirrors these.
# ---------------------------------------------------------------------------
RIG_FORMAT_VERSION = 1
MAX_INFLUENCES = 4

# Auto-skin algorithm tags.
AUTOSKIN_DISTANCE = "distance"     # 1 / dist^falloff, normalize
AUTOSKIN_HEAT = "heat"             # Pinocchio-style heat eqn (smoothed)

VALID_AUTOSKIN = (AUTOSKIN_DISTANCE, AUTOSKIN_HEAT)


# ---------------------------------------------------------------------------
# Bone pose composition
# ---------------------------------------------------------------------------
_BAMS_TO_RAD = (2.0 * math.pi) / 65536.0


@dataclass
class BonePose:
    """One bone's edited pose, separate from the source XjBone bind pose.

    Mirrors the wire shape's per-bone fields (sans skeleton-tree
    metadata which the parser owns).

    Attributes
    ----------
    index:
        DFS index in the source skeleton. Keys back into
        XjBone.index for round-trip with the source.
    parent:
        Parent index in the rig (after any reparent the user has
        applied). The wire format keeps this so the editor can
        reproduce the user's tree structure on reopen even when the
        source bind-pose tree differs.
    position:
        (tx, ty, tz) — bone-local translation. Floats.
    rotation_bams:
        (rx, ry, rz) — int BAMS. Round-trips with XjBone.rotation.
    scale:
        (sx, sy, sz) — bone-local scale, defaults to (1, 1, 1).
    name:
        Optional user-assigned name; "" or None when not renamed.
    eval_flags:
        Mirror of XjBone.eval_flags (UNIT_POS / UNIT_ANG / UNIT_SCL /
        SKIP / ZXY_ANG bits) — preserved verbatim so the bake pipeline
        produces the same matrix.
    hidden:
        UI hint — does NOT discard the bone from the skeleton; the
        renderer still needs every bone for vertex skinning.
    """
    index: int
    parent: int = -1
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_bams: tuple[int, int, int] = (0, 0, 0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    name: str = ""
    eval_flags: int = 0
    hidden: bool = False


def compose_local_matrix(pose: BonePose) -> list[float]:
    """Compose this bone's local TRS into a 4x4 row-major matrix.

    Mirror of the JS ``_composeTrsM4`` in model_viewer.js — same
    rotation order (ZYX by default; ZXY when EVAL_ZXY_ANG=0x20 is
    set on eval_flags) and same eval-flag overrides.

    Returns a length-16 list in row-major layout (m[row*4+col]).
    """
    EVAL_UNIT_POS = 0x01
    EVAL_UNIT_ANG = 0x02
    EVAL_UNIT_SCL = 0x04
    EVAL_ZXY_ANG  = 0x20
    EVAL_SKIP     = 0x40

    ef = int(pose.eval_flags)
    if ef & EVAL_SKIP:
        return [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
    tx, ty, tz = pose.position
    if ef & EVAL_UNIT_POS:
        tx = ty = tz = 0.0
    rx_b, ry_b, rz_b = pose.rotation_bams
    if ef & EVAL_UNIT_ANG:
        rx_b = ry_b = rz_b = 0
    sx, sy, sz = pose.scale
    if ef & EVAL_UNIT_SCL:
        sx = sy = sz = 1.0

    rx = rx_b * _BAMS_TO_RAD
    ry = ry_b * _BAMS_TO_RAD
    rz = rz_b * _BAMS_TO_RAD
    cx = math.cos(rx); s_x = math.sin(rx)
    cy = math.cos(ry); s_y = math.sin(ry)
    cz = math.cos(rz); s_z = math.sin(rz)
    if ef & EVAL_ZXY_ANG:
        # R = Rz * Rx * Ry
        r00 = cz * cy - s_z * s_x * s_y
        r01 = -s_z * cx
        r02 = cz * s_y + s_z * s_x * cy
        r10 = s_z * cy + cz * s_x * s_y
        r11 = cz * cx
        r12 = s_z * s_y - cz * s_x * cy
        r20 = -cx * s_y
        r21 = s_x
        r22 = cx * cy
    else:
        # R = Rz * Ry * Rx (ZYX)
        r00 = cz * cy
        r01 = cz * s_y * s_x - s_z * cx
        r02 = cz * s_y * cx + s_z * s_x
        r10 = s_z * cy
        r11 = s_z * s_y * s_x + cz * cx
        r12 = s_z * s_y * cx - cz * s_x
        r20 = -s_y
        r21 = cy * s_x
        r22 = cy * cx
    return [
        r00 * sx,  r01 * sy,  r02 * sz,  tx,
        r10 * sx,  r11 * sy,  r12 * sz,  ty,
        r20 * sx,  r21 * sy,  r22 * sz,  tz,
        0.0,        0.0,        0.0,        1.0,
    ]


def matmul4(a: list[float], b: list[float]) -> list[float]:
    """Multiply two row-major 4x4 matrices: return ``a @ b``.

    Pure-Python; called once per bone per save (≪ 100 calls). Lists
    keep the on-disk JSON-serialisable shape; numpy is optional.
    """
    out = [0.0] * 16
    for r in range(4):
        for c in range(4):
            s = 0.0
            for k in range(4):
                s += a[r * 4 + k] * b[k * 4 + c]
            out[r * 4 + c] = s
    return out


def compose_world_matrices(poses: list[BonePose]) -> list[list[float]]:
    """Walk the bone tree and produce per-bone world matrices.

    Each bone's world matrix = parent.world @ bone.local. The root's
    parent is identity. Bones must be supplied in DFS / topological
    order (parent index < own index).
    """
    out: list[list[float]] = []
    for i, pose in enumerate(poses):
        local = compose_local_matrix(pose)
        if pose.parent < 0:
            out.append(local)
        else:
            if pose.parent >= i:
                # Defensive: misordered tree. Treat as root.
                out.append(local)
                continue
            out.append(matmul4(out[pose.parent], local))
    return out


def transform_point(m: list[float], p: Sequence[float]) -> tuple[float, float, float]:
    """Transform a 3-point by the row-major 4x4 in ``m``."""
    x = m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3]
    y = m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7]
    z = m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11]
    return (x, y, z)


# ---------------------------------------------------------------------------
# Per-vertex weights
# ---------------------------------------------------------------------------
@dataclass
class SubmeshWeights:
    """Per-vertex bone-influence storage for one submesh.

    `bone_indices` is shape (vertex_count, MAX_INFLUENCES) Int32 — the
    bone DFS index, or -1 for unused slots. `weights` is the same
    shape, Float32, summing to ≤ 1.0 per row (≤ 1.0 because the
    NJ-baseline single-influence path uses just slot 0).
    """
    submesh_idx: int
    vertex_count: int
    bone_indices: object  # numpy int32[N, MAX_INFLUENCES]
    weights: object       # numpy float32[N, MAX_INFLUENCES]


def empty_weights(submesh_idx: int, vertex_count: int) -> SubmeshWeights:
    """Make a SubmeshWeights with all-empty (-1, 0.0) slots.

    Used as the seed for auto-skin and as the v1 default when no
    explicit weights have been authored.
    """
    np = _np()
    bi = np.full((vertex_count, MAX_INFLUENCES), -1, dtype=np.int32)
    w = np.zeros((vertex_count, MAX_INFLUENCES), dtype=np.float32)
    return SubmeshWeights(
        submesh_idx=int(submesh_idx),
        vertex_count=int(vertex_count),
        bone_indices=bi,
        weights=w,
    )


def from_bone_idx_array(
    submesh_idx: int,
    bone_idx: Sequence[int],
) -> SubmeshWeights:
    """Convert a flat ``Int32[N]`` of single-influence bone indices
    (the shape /api/model_skinned ships) into a 4-influence
    SubmeshWeights.

    Slot 0 = the source bone with weight 1.0; slots 1..3 = (-1, 0.0).
    """
    np = _np()
    arr = np.asarray(bone_idx, dtype=np.int32).reshape(-1)
    n = int(arr.size)
    bi = np.full((n, MAX_INFLUENCES), -1, dtype=np.int32)
    w = np.zeros((n, MAX_INFLUENCES), dtype=np.float32)
    bi[:, 0] = arr
    w[:, 0] = 1.0
    # Mark missing-bone vertices as having no influence (slot 0 = -1).
    miss = arr < 0
    if miss.any():
        bi[miss, 0] = -1
        w[miss, 0] = 0.0
    return SubmeshWeights(
        submesh_idx=int(submesh_idx),
        vertex_count=n,
        bone_indices=bi,
        weights=w,
    )


def normalize_weights(sw: SubmeshWeights) -> SubmeshWeights:
    """Renormalize so each vertex's weights sum to exactly 1.0.

    Vertices whose raw sum is 0 (no influences assigned) keep their
    zero rows so downstream code can detect them. Negative weights
    are clamped to 0 first.
    """
    np = _np()
    w = np.asarray(sw.weights, dtype=np.float32).copy()
    np.maximum(w, 0.0, out=w)
    sums = w.sum(axis=1, keepdims=True)
    # Preserve all-zero rows (nothing to normalize).
    nz = sums.flatten() > 1e-9
    if nz.any():
        w[nz] /= sums[nz]
    sw.weights = w
    return sw


def add_weight(sw: SubmeshWeights, vert_idx: int, bone_idx: int, delta: float) -> None:
    """Increment the weight at (vert, bone) by ``delta``.

    If the bone isn't already an influence on the vertex, the lowest
    weighted slot is replaced (or an empty -1 slot if available).
    Other influences are NOT renormalized here — the caller decides
    when to call ``normalize_weights`` (for the JS-side weight-paint
    flow we normalize at brush-stroke end, not per-step, to keep
    numerical stability while smearing).
    """
    np = _np()
    bi = sw.bone_indices
    w = sw.weights
    if vert_idx < 0 or vert_idx >= sw.vertex_count:
        raise IndexError(f"vert_idx {vert_idx} out of range 0..{sw.vertex_count}")
    # Find existing slot.
    row_bi = bi[vert_idx]
    row_w = w[vert_idx]
    for k in range(MAX_INFLUENCES):
        if int(row_bi[k]) == bone_idx:
            row_w[k] = float(max(0.0, row_w[k] + delta))
            return
    # Find -1 slot.
    for k in range(MAX_INFLUENCES):
        if int(row_bi[k]) < 0:
            row_bi[k] = bone_idx
            row_w[k] = float(max(0.0, delta))
            return
    # Replace the smallest-weight slot.
    k_min = int(np.argmin(row_w))
    row_bi[k_min] = bone_idx
    row_w[k_min] = float(max(0.0, delta))


# ---------------------------------------------------------------------------
# Auto-skin: weight assignment from bone proximity
# ---------------------------------------------------------------------------
def autoskin_distance(
    positions,
    bones_world: list[list[float]],
    *,
    falloff: float = 4.0,
    max_influences: int = MAX_INFLUENCES,
) -> SubmeshWeights:
    """Compute weights from inverse-distance: w(v, b) = 1 / dist^falloff.

    For each vertex, the top ``max_influences`` closest bones are kept
    and their weights are normalized to sum to 1. Bones supplied as
    world-space matrices (length 16, row-major); the bone "origin" is
    matrix[3,7,11] (translation column).

    O(N_vert * N_bone) — fine for dragon-class skeletons (≤128 bones)
    and ≤10k verts. The JS path mirrors this so a re-run on the live
    mesh produces identical weights.
    """
    np = _np()
    pts = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    n_vert = int(pts.shape[0])
    n_bone = len(bones_world)
    if n_bone == 0 or n_vert == 0:
        return empty_weights(0, n_vert)
    # Bone positions (translation column of each world matrix).
    bp = np.array(
        [(m[3], m[7], m[11]) for m in bones_world],
        dtype=np.float64,
    )
    # All pairwise distances. dx[i, j] = pts[i] - bp[j].
    diff = pts[:, None, :] - bp[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    # Inverse distance with falloff. Avoid div-by-zero at coincident
    # vertex/bone (rare in real meshes; common in synthetic tests).
    eps = 1e-6
    inv = 1.0 / np.power(dist + eps, max(0.5, float(falloff)))
    # Top-K bones per vertex.
    k = min(max_influences, n_bone)
    if k <= 0:
        return empty_weights(0, n_vert)
    # argsort ascending → pick last k indices for top-K (largest inv).
    top_idx = np.argsort(inv, axis=1)[:, -k:]
    bi = np.full((n_vert, max_influences), -1, dtype=np.int32)
    w = np.zeros((n_vert, max_influences), dtype=np.float32)
    rows = np.arange(n_vert)[:, None].repeat(k, axis=1)
    bi[:, :k] = top_idx[:, ::-1].astype(np.int32)
    w[:, :k] = inv[rows, top_idx][:, ::-1].astype(np.float32)
    # Normalize per row.
    sums = w[:, :k].sum(axis=1, keepdims=True)
    nz = (sums.flatten() > 1e-12)
    if nz.any():
        w[nz, :k] /= sums[nz]
    return SubmeshWeights(
        submesh_idx=0,
        vertex_count=n_vert,
        bone_indices=bi,
        weights=w,
    )


def autoskin_heat(
    positions,
    bones_world: list[list[float]],
    *,
    iterations: int = 8,
    max_influences: int = MAX_INFLUENCES,
) -> SubmeshWeights:
    """Approximate Pinocchio-style heat-equation weights.

    The full Pinocchio paper diffuses heat from each bone across the
    mesh's Laplacian. We use a cheaper proxy that produces visually
    similar results for character meshes:

      1. Seed weights via inverse-distance (falloff=2.0).
      2. Smooth across vertex k-NN graph for ``iterations`` steps,
         keeping the per-bone column normalized after each pass.
      3. Top-K cull and renormalize.

    Suitable for v1 — full Pinocchio diffusion is deferred to v2 where
    we'd plumb a sparse Laplacian solver. This proxy preserves the
    "weights are smooth across nearby verts" property the user
    expects from heat-equation skinning while running in O(N * iter)
    instead of O(N^3) (factorisation of the cotangent matrix).
    """
    np = _np()
    pts = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    n_vert = int(pts.shape[0])
    n_bone = len(bones_world)
    if n_bone == 0 or n_vert == 0:
        return empty_weights(0, n_vert)
    bp = np.array(
        [(m[3], m[7], m[11]) for m in bones_world],
        dtype=np.float64,
    )
    diff = pts[:, None, :] - bp[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    eps = 1e-6
    seed = 1.0 / np.power(dist + eps, 2.0)
    # Per-vertex normalize (so each vertex's row sums to 1 at start).
    seed /= seed.sum(axis=1, keepdims=True) + eps
    # k-NN for smoothing — k=8 is enough for boss-class meshes.
    k_nn = min(8, n_vert)
    if k_nn <= 1:
        # Trivial mesh — skip smoothing.
        smoothed = seed
    else:
        # Pairwise vertex distances; pick k-nearest (excluding self).
        vv_diff = pts[:, None, :] - pts[None, :, :]
        vv_dist = np.sqrt((vv_diff * vv_diff).sum(axis=2))
        # argsort ascending; index 0 is self (distance 0).
        knn = np.argsort(vv_dist, axis=1)[:, 1:k_nn + 1]
        smoothed = seed.copy()
        # Iterative smoothing: each row replaced by mean of itself + neighbours.
        for _ in range(max(0, int(iterations))):
            nbr_mean = smoothed[knn].mean(axis=1)  # (N, n_bone)
            smoothed = 0.5 * smoothed + 0.5 * nbr_mean
            # Per-row normalize so sum(weights) == 1 across bones.
            row_sum = smoothed.sum(axis=1, keepdims=True) + eps
            smoothed = smoothed / row_sum
    # Top-K cull.
    k = min(max_influences, n_bone)
    top_idx = np.argsort(smoothed, axis=1)[:, -k:]
    bi = np.full((n_vert, max_influences), -1, dtype=np.int32)
    w = np.zeros((n_vert, max_influences), dtype=np.float32)
    rows = np.arange(n_vert)[:, None].repeat(k, axis=1)
    bi[:, :k] = top_idx[:, ::-1].astype(np.int32)
    w[:, :k] = smoothed[rows, top_idx][:, ::-1].astype(np.float32)
    sums = w[:, :k].sum(axis=1, keepdims=True)
    nz = (sums.flatten() > 1e-12)
    if nz.any():
        w[nz, :k] /= sums[nz]
    return SubmeshWeights(
        submesh_idx=0, vertex_count=n_vert,
        bone_indices=bi, weights=w,
    )


def auto_skin(
    positions,
    bones_world: list[list[float]],
    *,
    algorithm: str = AUTOSKIN_DISTANCE,
    falloff: float = 4.0,
    iterations: int = 8,
    max_influences: int = MAX_INFLUENCES,
) -> SubmeshWeights:
    """Dispatcher for the auto-skin algorithms above.

    Returns a SubmeshWeights with ``submesh_idx=0`` (caller patches
    that field for the multi-submesh path).
    """
    if algorithm not in VALID_AUTOSKIN:
        raise ValueError(f"unknown algorithm {algorithm!r}; expected one of {VALID_AUTOSKIN}")
    if algorithm == AUTOSKIN_HEAT:
        return autoskin_heat(
            positions, bones_world,
            iterations=iterations, max_influences=max_influences,
        )
    return autoskin_distance(
        positions, bones_world,
        falloff=falloff, max_influences=max_influences,
    )


# ---------------------------------------------------------------------------
# IK targets
# ---------------------------------------------------------------------------
@dataclass
class IkTarget:
    """One IK constraint in the rig.

    Attributes
    ----------
    bone_idx:
        End-effector bone (where the chain ends — the bone whose tip
        chases the target).
    chain_length:
        How many bones up the parent chain to include in the solve
        (2 = one elbow joint, 3 = two intermediate joints, etc.).
    target:
        World-space target position (x, y, z).
    iterations:
        FABRIK iteration cap. 10-20 is typical; large values improve
        accuracy but cost CPU. Default 16.
    name:
        Optional user-facing label.
    """
    bone_idx: int
    chain_length: int = 2
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    iterations: int = 16
    name: str = ""


def fabrik_solve(
    chain_positions: list[Sequence[float]],
    target: Sequence[float],
    *,
    iterations: int = 16,
    tol: float = 1e-3,
) -> list[tuple[float, float, float]]:
    """Forward-And-Backward-Reaching IK on a chain of joint positions.

    Parameters
    ----------
    chain_positions:
        Joint positions in WORLD space, ordered from ROOT to END.
        Length N >= 2; segment lengths are computed from the input.
    target:
        World-space target the END joint should reach.
    iterations:
        Cap on the forward+backward passes.
    tol:
        Convergence threshold; loop stops when the end joint is
        within this many world units of the target.

    Returns
    -------
    list[(x, y, z)]
        New joint positions, same order as input. Lengths between
        consecutive joints are preserved (within ``tol``).

    Algorithm
    ---------
    Aristidou-Lasenby FABRIK (2011). Each iteration:

      Backward pass — set end = target, then for each joint i from
      N-2 down to 0: set joint i so that it sits at the correct
      segment-length away from joint i+1, in the direction from
      joint i+1 → joint i (preserving direction). This satisfies the
      "end at target" constraint.

      Forward pass — set root = original root, then for each joint i
      from 1 up to N-1: set joint i so that it sits at the correct
      segment-length away from joint i-1, in the direction from
      joint i-1 → joint i. This satisfies the "root at fixed
      position" constraint.

    Repeat until the end is close enough to the target.
    """
    n = len(chain_positions)
    if n < 2:
        # Trivial — return the input as-is.
        return [(p[0], p[1], p[2]) for p in chain_positions]
    pts = [(float(p[0]), float(p[1]), float(p[2])) for p in chain_positions]
    seg_len = [
        math.sqrt(
            (pts[i + 1][0] - pts[i][0]) ** 2
            + (pts[i + 1][1] - pts[i][1]) ** 2
            + (pts[i + 1][2] - pts[i][2]) ** 2
        )
        for i in range(n - 1)
    ]
    total_reach = sum(seg_len)
    root = pts[0]
    # If target is unreachable from root (out of total reach), just
    # straighten the chain toward it.
    dx = target[0] - root[0]; dy = target[1] - root[1]; dz = target[2] - root[2]
    dist_to_target = math.sqrt(dx * dx + dy * dy + dz * dz)
    if dist_to_target > total_reach:
        if dist_to_target < 1e-9:
            return pts
        out = [root]
        cum = 0.0
        for i in range(n - 1):
            cum += seg_len[i]
            t = cum / dist_to_target
            out.append((
                root[0] + dx * t,
                root[1] + dy * t,
                root[2] + dz * t,
            ))
        return out
    # FABRIK passes.
    target_v = (float(target[0]), float(target[1]), float(target[2]))
    for _ in range(max(1, int(iterations))):
        # Backward.
        pts[-1] = target_v
        for i in range(n - 2, -1, -1):
            ax = pts[i][0] - pts[i + 1][0]
            ay = pts[i][1] - pts[i + 1][1]
            az = pts[i][2] - pts[i + 1][2]
            d = math.sqrt(ax * ax + ay * ay + az * az)
            if d < 1e-9:
                # Coincident — push along arbitrary axis.
                pts[i] = (pts[i + 1][0] + seg_len[i], pts[i + 1][1], pts[i + 1][2])
                continue
            r = seg_len[i] / d
            pts[i] = (
                pts[i + 1][0] + ax * r,
                pts[i + 1][1] + ay * r,
                pts[i + 1][2] + az * r,
            )
        # Forward.
        pts[0] = root
        for i in range(n - 1):
            ax = pts[i + 1][0] - pts[i][0]
            ay = pts[i + 1][1] - pts[i][1]
            az = pts[i + 1][2] - pts[i][2]
            d = math.sqrt(ax * ax + ay * ay + az * az)
            if d < 1e-9:
                pts[i + 1] = (pts[i][0] + seg_len[i], pts[i][1], pts[i][2])
                continue
            r = seg_len[i] / d
            pts[i + 1] = (
                pts[i][0] + ax * r,
                pts[i][1] + ay * r,
                pts[i][2] + az * r,
            )
        # Convergence check.
        ex = pts[-1][0] - target_v[0]
        ey = pts[-1][1] - target_v[1]
        ez = pts[-1][2] - target_v[2]
        if ex * ex + ey * ey + ez * ez < tol * tol:
            break
    return pts


def two_bone_ik(
    root: Sequence[float],
    mid: Sequence[float],
    end: Sequence[float],
    target: Sequence[float],
    pole: Optional[Sequence[float]] = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Analytic 2-bone IK (root → mid → end → target) using the law
    of cosines.

    Faster + more stable than FABRIK for the common arm/leg case.
    Returns the new ``(mid, end)`` joint positions; root is fixed.

    The pole vector points from the root toward the joint's preferred
    bend direction (knee for a leg, elbow for an arm). When ``None``,
    we default to "+Y" or the original mid's direction so the bend
    plane is preserved as much as possible.
    """
    rx, ry, rz = float(root[0]), float(root[1]), float(root[2])
    mx, my, mz = float(mid[0]), float(mid[1]), float(mid[2])
    ex, ey, ez = float(end[0]), float(end[1]), float(end[2])
    tx, ty, tz = float(target[0]), float(target[1]), float(target[2])
    # Original segment lengths (preserved).
    L1 = math.sqrt((mx - rx) ** 2 + (my - ry) ** 2 + (mz - rz) ** 2)
    L2 = math.sqrt((ex - mx) ** 2 + (ey - my) ** 2 + (ez - mz) ** 2)
    # Distance to target, clamped to be reachable.
    dx = tx - rx; dy = ty - ry; dz = tz - rz
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if dist < 1e-9:
        return (mx, my, mz), (ex, ey, ez)  # degenerate
    eff = max(min(dist, L1 + L2 - 1e-6), abs(L1 - L2) + 1e-6)
    # Direction along root→target (unit).
    udx = dx / dist; udy = dy / dist; udz = dz / dist
    # Pole direction (perpendicular to udx,udy,udz).
    if pole is not None:
        pdx = float(pole[0]) - rx
        pdy = float(pole[1]) - ry
        pdz = float(pole[2]) - rz
    else:
        # Default pole: original mid offset perpendicular to root→target.
        omx = mx - rx; omy = my - ry; omz = mz - rz
        # Project off ud: pole = orig_mid - dot(orig_mid, ud) * ud.
        dot = omx * udx + omy * udy + omz * udz
        pdx = omx - dot * udx
        pdy = omy - dot * udy
        pdz = omz - dot * udz
    # If pole is parallel/zero, fall back to a fixed axis.
    pdl = math.sqrt(pdx * pdx + pdy * pdy + pdz * pdz)
    if pdl < 1e-6:
        # Pick an axis perpendicular to ud.
        if abs(udy) < 0.9:
            pdx, pdy, pdz = 0.0, 1.0, 0.0
        else:
            pdx, pdy, pdz = 1.0, 0.0, 0.0
        # Re-orthogonalise.
        dot = pdx * udx + pdy * udy + pdz * udz
        pdx -= dot * udx; pdy -= dot * udy; pdz -= dot * udz
        pdl = math.sqrt(pdx * pdx + pdy * pdy + pdz * pdz)
    pdx /= pdl; pdy /= pdl; pdz /= pdl
    # Cosine of the angle at root via law of cosines.
    cos_a = (L1 * L1 + eff * eff - L2 * L2) / (2.0 * L1 * eff)
    cos_a = max(-1.0, min(1.0, cos_a))
    sin_a = math.sqrt(max(0.0, 1.0 - cos_a * cos_a))
    new_mx = rx + L1 * (cos_a * udx + sin_a * pdx)
    new_my = ry + L1 * (cos_a * udy + sin_a * pdy)
    new_mz = rz + L1 * (cos_a * udz + sin_a * pdz)
    # End sits along (target - new_mid) at distance L2.
    emx = tx - new_mx; emy = ty - new_my; emz = tz - new_mz
    eml = math.sqrt(emx * emx + emy * emy + emz * emz)
    if eml < 1e-9:
        new_ex, new_ey, new_ez = tx, ty, tz
    else:
        sc = L2 / eml
        new_ex = new_mx + emx * sc
        new_ey = new_my + emy * sc
        new_ez = new_mz + emz * sc
    return (new_mx, new_my, new_mz), (new_ex, new_ey, new_ez)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def encode_rig_payload(
    *,
    source_path: str,
    source_sha: str,
    bones: list[BonePose],
    weights: list[SubmeshWeights],
    ik_targets: list[IkTarget] | None = None,
    subdivide_level: int = 0,
) -> dict:
    """Pack a rig into the wire JSON shape.

    The returned dict is JSON-serialisable; callers pass it to
    ``json.dumps`` and write to disk.
    """
    np = _np()
    bone_dicts: list[dict] = []
    for b in bones:
        bone_dicts.append({
            "index": int(b.index),
            "parent": int(b.parent),
            "position": [float(b.position[0]), float(b.position[1]), float(b.position[2])],
            "rotation_bams": [int(b.rotation_bams[0]), int(b.rotation_bams[1]), int(b.rotation_bams[2])],
            "scale": [float(b.scale[0]), float(b.scale[1]), float(b.scale[2])],
            "name": str(b.name or ""),
            "eval_flags": int(b.eval_flags),
            "hidden": bool(b.hidden),
        })
    weight_dicts: list[dict] = []
    for w in weights:
        bi = np.asarray(w.bone_indices, dtype=np.int32)
        ww = np.asarray(w.weights, dtype=np.float32)
        if bi.shape != (w.vertex_count, MAX_INFLUENCES):
            raise ValueError(
                f"submesh {w.submesh_idx}: bone_indices shape {bi.shape} "
                f"!= ({w.vertex_count}, {MAX_INFLUENCES})"
            )
        if ww.shape != (w.vertex_count, MAX_INFLUENCES):
            raise ValueError(
                f"submesh {w.submesh_idx}: weights shape {ww.shape} "
                f"!= ({w.vertex_count}, {MAX_INFLUENCES})"
            )
        weight_dicts.append({
            "submesh_idx": int(w.submesh_idx),
            "vertex_count": int(w.vertex_count),
            "indices_b64": base64.b64encode(bi.tobytes()).decode("ascii"),
            "weights_b64": base64.b64encode(ww.tobytes()).decode("ascii"),
            "max_influences": int(MAX_INFLUENCES),
        })
    ik_dicts: list[dict] = []
    for tgt in (ik_targets or []):
        ik_dicts.append({
            "bone_idx": int(tgt.bone_idx),
            "chain_length": int(tgt.chain_length),
            "target": [float(tgt.target[0]), float(tgt.target[1]), float(tgt.target[2])],
            "iterations": int(tgt.iterations),
            "name": str(tgt.name or ""),
        })
    payload = {
        "format_version": RIG_FORMAT_VERSION,
        "source_path": str(source_path),
        "source_sha": str(source_sha),
        "subdivide_level": int(subdivide_level),
        "skeleton": {"bones": bone_dicts},
        "weights": weight_dicts,
        "ik_targets": ik_dicts,
        "saved_at_ms": int(time.time() * 1000),
    }
    # Stable hash over the payload's data fields. SHA-1 truncated to
    # 16 hex chars (matches the sculpt module's convention so file
    # listings sort consistently).
    h = hashlib.sha1()
    for b in bone_dicts:
        h.update(struct.pack("<iiif",
                             b["index"], b["parent"],
                             int(b["eval_flags"]), float(b["position"][0])))
        h.update(struct.pack("<fff",
                             float(b["position"][1]), float(b["position"][2]),
                             float(b["scale"][0])))
        h.update(struct.pack("<iii",
                             int(b["rotation_bams"][0]),
                             int(b["rotation_bams"][1]),
                             int(b["rotation_bams"][2])))
        h.update(b["name"].encode("utf-8"))
    for w in weight_dicts:
        h.update(b"|w|")
        h.update(w["indices_b64"].encode("ascii"))
        h.update(b"|")
        h.update(w["weights_b64"].encode("ascii"))
    for t in ik_dicts:
        h.update(b"|i|")
        h.update(struct.pack("<iif",
                             t["bone_idx"], t["chain_length"],
                             float(t["target"][0])))
        h.update(struct.pack("<fff",
                             float(t["target"][1]),
                             float(t["target"][2]),
                             float(t["iterations"])))
    payload["sha"] = h.hexdigest()[:16]
    return payload


def decode_rig_payload(payload: dict) -> tuple[
    list[BonePose], list[SubmeshWeights], list[IkTarget]
]:
    """Inverse of ``encode_rig_payload``.

    Raises ValueError on malformed input or version mismatch.
    """
    np = _np()
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    fv = int(payload.get("format_version", 0))
    if fv != RIG_FORMAT_VERSION:
        raise ValueError(
            f"format_version {fv!r} != expected {RIG_FORMAT_VERSION}"
        )
    bones: list[BonePose] = []
    for b in payload.get("skeleton", {}).get("bones", []):
        pos = b.get("position", [0.0, 0.0, 0.0])
        rot = b.get("rotation_bams", [0, 0, 0])
        scl = b.get("scale", [1.0, 1.0, 1.0])
        bones.append(BonePose(
            index=int(b["index"]),
            parent=int(b.get("parent", -1)),
            position=(float(pos[0]), float(pos[1]), float(pos[2])),
            rotation_bams=(int(rot[0]), int(rot[1]), int(rot[2])),
            scale=(float(scl[0]), float(scl[1]), float(scl[2])),
            name=str(b.get("name", "") or ""),
            eval_flags=int(b.get("eval_flags", 0)),
            hidden=bool(b.get("hidden", False)),
        ))
    weights: list[SubmeshWeights] = []
    for w in payload.get("weights", []):
        vc = int(w["vertex_count"])
        max_inf = int(w.get("max_influences", MAX_INFLUENCES))
        if max_inf != MAX_INFLUENCES:
            # Accept legacy/forward shapes as long as we can reshape.
            pass
        bi_bytes = base64.b64decode(w["indices_b64"])
        ww_bytes = base64.b64decode(w["weights_b64"])
        bi = np.frombuffer(bi_bytes, dtype=np.int32).copy()
        ww = np.frombuffer(ww_bytes, dtype=np.float32).copy()
        if bi.size != vc * max_inf or ww.size != vc * max_inf:
            raise ValueError(
                f"submesh {w.get('submesh_idx')}: bytes ({bi.size}, {ww.size}) "
                f"don't match vc*max_inf {vc * max_inf}"
            )
        bi = bi.reshape(vc, max_inf)
        ww = ww.reshape(vc, max_inf)
        # Pad / trim to MAX_INFLUENCES so downstream code can assume
        # the canonical width.
        if bi.shape[1] != MAX_INFLUENCES:
            new_bi = np.full((vc, MAX_INFLUENCES), -1, dtype=np.int32)
            new_ww = np.zeros((vc, MAX_INFLUENCES), dtype=np.float32)
            cw = min(bi.shape[1], MAX_INFLUENCES)
            new_bi[:, :cw] = bi[:, :cw]
            new_ww[:, :cw] = ww[:, :cw]
            bi, ww = new_bi, new_ww
        weights.append(SubmeshWeights(
            submesh_idx=int(w["submesh_idx"]),
            vertex_count=vc,
            bone_indices=bi,
            weights=ww,
        ))
    ik_targets: list[IkTarget] = []
    for t in payload.get("ik_targets", []):
        tgt = t.get("target", [0.0, 0.0, 0.0])
        ik_targets.append(IkTarget(
            bone_idx=int(t["bone_idx"]),
            chain_length=int(t.get("chain_length", 2)),
            target=(float(tgt[0]), float(tgt[1]), float(tgt[2])),
            iterations=int(t.get("iterations", 16)),
            name=str(t.get("name", "") or ""),
        ))
    return bones, weights, ik_targets


def compute_source_sha(blob: bytes) -> str:
    """Stable, short hash of the source mesh bytes for cache keying.

    Mirrors ``formats.sculpt.compute_source_sha`` so a model edited in
    both Sculpt and Rig keys to the same SHA prefix.
    """
    return hashlib.sha1(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------
__all__ = [
    "RIG_FORMAT_VERSION",
    "MAX_INFLUENCES",
    "AUTOSKIN_DISTANCE",
    "AUTOSKIN_HEAT",
    "VALID_AUTOSKIN",
    "BonePose",
    "SubmeshWeights",
    "IkTarget",
    "compose_local_matrix",
    "compose_world_matrices",
    "matmul4",
    "transform_point",
    "empty_weights",
    "from_bone_idx_array",
    "normalize_weights",
    "add_weight",
    "autoskin_distance",
    "autoskin_heat",
    "auto_skin",
    "fabrik_solve",
    "two_bone_ik",
    "encode_rig_payload",
    "decode_rig_payload",
    "compute_source_sha",
]
