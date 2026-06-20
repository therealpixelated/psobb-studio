"""Animation blend-spaces — runtime motion blending for ``NjmRawMotion``.

A blend-space takes 2-3 source motions and produces a new motion whose
per-bone, per-frame rotation/translation is a weighted slerp/lerp of
the inputs. This is the offline analogue of what a runtime engine
would do for "idle ↔ walk" or "walk → run" blends — we precompute the
blend at edit time and ship a single .njm.

The module exists so that:
  * The 3D editor can blend motions a user picks (no game-engine round
    trip needed; results are saved as a deployable .njm).
  * The /api/anim/blend server endpoint surfaces the same primitive
    to the front-end via JSON.

Blend math
----------
For each TARGET bone, for each TARGET frame f:

  Q_blend(f) = slerp_chain(
      Q_motion0(f) ^ w0,
      Q_motion1(f) ^ w1,
      Q_motion2(f) ^ w2,  # (optional)
  )

Multi-way slerp is implemented via successive pairwise slerps with
re-normalised weights:

  Q_01 = slerp(Q_0, Q_1, w_1 / (w_0 + w_1))
  Q_blend = slerp(Q_01, Q_2, w_2 / (w_0 + w_1 + w_2))

This is the standard "n-way slerp" trick — produces stable results as
long as the weights are non-negative. Translation tracks (POS) lerp
linearly (additive). Scale is (rarely used) lerped linearly.

Frame-count handling
--------------------
Source motions may have different frame counts. We resample each
source to the OUTPUT frame count by treating its keyframe list as a
LINEAR-interpolated function of frame index, then blending sample-by
-sample. Simpler than re-fitting keyframes; for offline use the
overhead is fine (bake at edit time, ship once).

Bone-count handling
-------------------
All sources must share the SAME bone count (same skeleton). We don't
attempt to retarget here — that's ``anim_retarget``'s job. If the
inputs disagree, ``blend_motions`` raises ValueError with the
mismatched counts so the caller can route to the retargeter.

Constants & wire format
-----------------------
This module emits standard ``NjmRawMotion`` objects that round-trip
through ``njm_writer.encode_njm`` byte-for-byte the same as
synthetic motions from the retargeter. Output ``inp_fn`` mirrors
the source motions' (1 element_count = ANG only; 2 = POS+ANG).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .njm import (
    NJD_MTYPE_ANG,
    NJD_MTYPE_POS,
    NJD_MTYPE_SCL,
)
from .njm_writer import NjmBoneTracks, NjmRawMotion, NjmTrack


# ---------------------------------------------------------------------------
# Curve types
# ---------------------------------------------------------------------------

# Smooth in-out curves are the most common transition shape; "linear"
# matches the per-frame weight ramps you'd see in a runtime blend tree.
TRANSITION_LINEAR = "linear"
TRANSITION_SMOOTH = "smooth"   # smoothstep: 3t^2 - 2t^3
TRANSITION_EASE_IN = "ease_in"
TRANSITION_EASE_OUT = "ease_out"
VALID_TRANSITIONS = (
    TRANSITION_LINEAR,
    TRANSITION_SMOOTH,
    TRANSITION_EASE_IN,
    TRANSITION_EASE_OUT,
)


def _apply_transition_curve(t: float, curve: str) -> float:
    """Map ``t`` ∈ [0, 1] through one of the named curves.

    Used by the ``BlendNode`` time-varying weight calculation; the
    static-weight ``blend_motions`` path multiplies the curve-shaped
    per-frame weight against the user's per-motion weight at each
    frame.
    """
    t = max(0.0, min(1.0, float(t)))
    if curve == TRANSITION_SMOOTH:
        return t * t * (3.0 - 2.0 * t)
    if curve == TRANSITION_EASE_IN:
        return t * t
    if curve == TRANSITION_EASE_OUT:
        return 1.0 - (1.0 - t) * (1.0 - t)
    # default LINEAR
    return t


# ---------------------------------------------------------------------------
# BlendNode dataclass — describes a blend's authoring intent
# ---------------------------------------------------------------------------


@dataclass
class BlendNode:
    """Authoring description of one blend.

    Attributes
    ----------
    sources:
        List of source ``NjmRawMotion`` (already parsed). They MUST
        share a bone count; differing frame counts are resampled.
    weights:
        Per-source weights. Must match ``len(sources)``. Need not sum
        to 1; the blender renormalises. Negative weights are clamped
        to 0.
    frame_count:
        Output motion's frame count. When None, we use the longest
        source's frame count.
    transition_curve:
        One of ``VALID_TRANSITIONS`` — how the per-frame interpolation
        weight ramps over time. Static (constant) blending uses
        ``TRANSITION_LINEAR`` with the same weight at every frame; the
        other curves animate the blend factor across the motion's
        duration. Use SMOOTH for "fade idle in over the first half,
        out over the second half" effects.
    """
    sources: List[NjmRawMotion] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    frame_count: Optional[int] = None
    transition_curve: str = TRANSITION_LINEAR


# ---------------------------------------------------------------------------
# Quaternion utilities (lifted from anim_retarget; private fork to
# keep this module self-contained)
# ---------------------------------------------------------------------------


def _quat_normalize(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    qx, qy, qz, qw = q
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return (qx / n, qy / n, qz / n, qw / n)


def _quat_dot(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


def _quat_slerp(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    t: float,
) -> Tuple[float, float, float, float]:
    """Spherical linear interpolation; returns a unit quaternion."""
    a = _quat_normalize(a)
    b = _quat_normalize(b)
    d = _quat_dot(a, b)
    if d < 0.0:
        b = (-b[0], -b[1], -b[2], -b[3])
        d = -d
    if d > 0.9995:
        # Linear blend then renormalise (avoids the small-angle div-by-0).
        return _quat_normalize((
            a[0] + t * (b[0] - a[0]),
            a[1] + t * (b[1] - a[1]),
            a[2] + t * (b[2] - a[2]),
            a[3] + t * (b[3] - a[3]),
        ))
    theta_0 = math.acos(min(1.0, max(-1.0, d)))
    theta = theta_0 * t
    sin_theta_0 = math.sin(theta_0)
    if sin_theta_0 < 1e-9:
        return a
    s_a = math.sin(theta_0 - theta) / sin_theta_0
    s_b = math.sin(theta) / sin_theta_0
    return (
        s_a * a[0] + s_b * b[0],
        s_a * a[1] + s_b * b[1],
        s_a * a[2] + s_b * b[2],
        s_a * a[3] + s_b * b[3],
    )


# Convert ZYX BAMS → quaternion and back.
_BAMS_TO_RAD = math.pi * 2.0 / 0x10000


def _bams_to_quat(rx_b: int, ry_b: int, rz_b: int) -> Tuple[float, float, float, float]:
    rx = (rx_b if rx_b < 0x8000 else rx_b - 0x10000) * _BAMS_TO_RAD
    ry = (ry_b if ry_b < 0x8000 else ry_b - 0x10000) * _BAMS_TO_RAD
    rz = (rz_b if rz_b < 0x8000 else rz_b - 0x10000) * _BAMS_TO_RAD
    cx, sx = math.cos(rx * 0.5), math.sin(rx * 0.5)
    cy, sy = math.cos(ry * 0.5), math.sin(ry * 0.5)
    cz, sz = math.cos(rz * 0.5), math.sin(rz * 0.5)
    qw = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz
    return (qx, qy, qz, qw)


def _quat_to_zyx_bams(q: Tuple[float, float, float, float]) -> Tuple[int, int, int]:
    """Mirror of ``import_external.quat_to_zyx_bams`` (kept inline here
    so this module doesn't pull import_external for its tiny surface).
    """
    qx, qy, qz, qw = _quat_normalize(q)
    sy = 2.0 * (qx * qz - qw * qy)
    sy = max(-1.0, min(1.0, -sy))
    if abs(sy) > 0.99999:
        ry = math.copysign(math.pi * 0.5, sy)
        rx = 0.0
        rz = math.atan2(-2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz))
    else:
        ry = math.asin(sy)
        rx = math.atan2(2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy))
        rz = math.atan2(2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qy * qy + qz * qz))
    rx_b = int(round(rx / _BAMS_TO_RAD)) & 0xFFFF
    ry_b = int(round(ry / _BAMS_TO_RAD)) & 0xFFFF
    rz_b = int(round(rz / _BAMS_TO_RAD)) & 0xFFFF
    return (rx_b, ry_b, rz_b)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _sample_ang_at_frame(
    keyframes: Sequence[Tuple],
    frame: float,
    bind_quat: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> Tuple[float, float, float, float]:
    """Slerp the ANG track's keyframes to find the rotation at ``frame``.

    ``frame`` may be fractional. When the track has no keyframes, we
    return ``bind_quat`` (the per-bone fall-back the runtime applies).
    """
    if not keyframes:
        return bind_quat
    if len(keyframes) == 1:
        return _bams_to_quat(int(keyframes[0][1]) & 0xFFFF,
                             int(keyframes[0][2]) & 0xFFFF,
                             int(keyframes[0][3]) & 0xFFFF)
    # Find bracketing keyframes.
    if frame <= keyframes[0][0]:
        return _bams_to_quat(int(keyframes[0][1]) & 0xFFFF,
                             int(keyframes[0][2]) & 0xFFFF,
                             int(keyframes[0][3]) & 0xFFFF)
    if frame >= keyframes[-1][0]:
        kf = keyframes[-1]
        return _bams_to_quat(int(kf[1]) & 0xFFFF, int(kf[2]) & 0xFFFF, int(kf[3]) & 0xFFFF)
    for i in range(len(keyframes) - 1):
        f0 = keyframes[i][0]
        f1 = keyframes[i + 1][0]
        if f0 <= frame <= f1:
            if f1 - f0 <= 1e-9:
                kf = keyframes[i]
                return _bams_to_quat(int(kf[1]) & 0xFFFF, int(kf[2]) & 0xFFFF, int(kf[3]) & 0xFFFF)
            t = (frame - f0) / (f1 - f0)
            q0 = _bams_to_quat(int(keyframes[i][1]) & 0xFFFF,
                               int(keyframes[i][2]) & 0xFFFF,
                               int(keyframes[i][3]) & 0xFFFF)
            q1 = _bams_to_quat(int(keyframes[i + 1][1]) & 0xFFFF,
                               int(keyframes[i + 1][2]) & 0xFFFF,
                               int(keyframes[i + 1][3]) & 0xFFFF)
            return _quat_slerp(q0, q1, t)
    return bind_quat


def _sample_pos_at_frame(
    keyframes: Sequence[Tuple],
    frame: float,
    fallback: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[float, float, float]:
    """Linear-interp a POS track at ``frame``."""
    if not keyframes:
        return fallback
    if len(keyframes) == 1:
        return (
            float(keyframes[0][1]),
            float(keyframes[0][2]),
            float(keyframes[0][3]),
        )
    if frame <= keyframes[0][0]:
        return (
            float(keyframes[0][1]),
            float(keyframes[0][2]),
            float(keyframes[0][3]),
        )
    if frame >= keyframes[-1][0]:
        kf = keyframes[-1]
        return (float(kf[1]), float(kf[2]), float(kf[3]))
    for i in range(len(keyframes) - 1):
        f0 = keyframes[i][0]
        f1 = keyframes[i + 1][0]
        if f0 <= frame <= f1:
            if f1 - f0 <= 1e-9:
                kf = keyframes[i]
                return (float(kf[1]), float(kf[2]), float(kf[3]))
            t = (frame - f0) / (f1 - f0)
            x0, y0, z0 = float(keyframes[i][1]), float(keyframes[i][2]), float(keyframes[i][3])
            x1, y1, z1 = float(keyframes[i + 1][1]), float(keyframes[i + 1][2]), float(keyframes[i + 1][3])
            return (
                x0 + t * (x1 - x0),
                y0 + t * (y1 - y0),
                z0 + t * (z1 - z0),
            )
    return fallback


# ---------------------------------------------------------------------------
# Multi-way blend
# ---------------------------------------------------------------------------


def _normalize_weights(weights: Sequence[float]) -> List[float]:
    """Clip negatives to 0, normalise to sum=1.

    Returns the input length unchanged. When all weights are zero
    (or absent), falls back to uniform weighting so the blend is
    well-defined.
    """
    clipped = [max(0.0, float(w)) for w in weights]
    s = sum(clipped)
    if s <= 1e-9:
        n = max(1, len(clipped))
        return [1.0 / n] * len(clipped)
    return [w / s for w in clipped]


def _slerp_n_way(
    quats: Sequence[Tuple[float, float, float, float]],
    weights: Sequence[float],
) -> Tuple[float, float, float, float]:
    """Blend N quaternions with the given weights (sum = 1).

    Implementation: pairwise slerp with running cumulative weight.
    For two-way input (N=2), this is exactly slerp(q0, q1, w1).
    For three-way: slerp(slerp(q0, q1, w1/(w0+w1)), q2, w2).
    """
    if not quats:
        return (0.0, 0.0, 0.0, 1.0)
    if len(quats) == 1:
        return _quat_normalize(quats[0])
    cum_w = float(weights[0])
    out = quats[0]
    for i in range(1, len(quats)):
        wi = float(weights[i])
        denom = cum_w + wi
        if denom <= 1e-9:
            continue
        t = wi / denom
        out = _quat_slerp(out, quats[i], t)
        cum_w = denom
    return out


def _lerp_n_way(
    vecs: Sequence[Tuple[float, float, float]],
    weights: Sequence[float],
) -> Tuple[float, float, float]:
    """Linear weighted average of vec3 inputs."""
    if not vecs:
        return (0.0, 0.0, 0.0)
    sx = sy = sz = 0.0
    sum_w = 0.0
    for v, w in zip(vecs, weights):
        sx += v[0] * w
        sy += v[1] * w
        sz += v[2] * w
        sum_w += w
    if sum_w <= 1e-9:
        return (0.0, 0.0, 0.0)
    return (sx / sum_w, sy / sum_w, sz / sum_w)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def blend_motions(
    motions: Sequence[NjmRawMotion],
    weights: Sequence[float],
    *,
    frame_count: Optional[int] = None,
    transition_curve: str = TRANSITION_LINEAR,
    bind_quats: Optional[List[Tuple[float, float, float, float]]] = None,
) -> NjmRawMotion:
    """Blend N source motions into a single output ``NjmRawMotion``.

    Parameters
    ----------
    motions:
        Sequence of source motions; must share bone count.
    weights:
        Per-motion weights (length matches motions). Negative values
        are clamped to 0; the sum is renormalised internally.
    frame_count:
        Output frame count. None → max of source frame counts.
    transition_curve:
        One of ``VALID_TRANSITIONS``. With curves != LINEAR, the
        per-source weight ramps from 0 → user_weight as
        ``apply_curve(frame / max_frame)`` for source[1..],
        decreasing complementary share for source[0]. This makes a
        2-source blend act as a "transition from motion0 to motion1
        over the duration".
    bind_quats:
        Optional per-bone bind quats used as a fall-back when a
        source has no keyframes for that bone (the runtime uses the
        bind pose in that case). Length must match the bone count
        when supplied; passed through from the caller's skeleton-
        parse step.

    Returns
    -------
    NjmRawMotion in packed-layout form (no source-body hint), ready
    to be encoded with ``encode_njm``.

    Raises
    ------
    ValueError
        On bone-count mismatch, empty motion list, or unknown curve.
    """
    if not motions:
        raise ValueError("blend_motions: motions list is empty")
    if len(motions) != len(weights):
        raise ValueError(
            f"blend_motions: motions ({len(motions)}) and weights ({len(weights)}) length mismatch",
        )
    if transition_curve not in VALID_TRANSITIONS:
        raise ValueError(
            f"blend_motions: unknown transition_curve {transition_curve!r}; "
            f"expected one of {VALID_TRANSITIONS}",
        )
    bone_count = len(motions[0].bones)
    for m in motions[1:]:
        if len(m.bones) != bone_count:
            raise ValueError(
                f"blend_motions: bone count mismatch (got {[len(m.bones) for m in motions]})",
            )
    if frame_count is None:
        frame_count = max(int(m.frame_count) for m in motions)
    if frame_count <= 0:
        raise ValueError(f"blend_motions: frame_count must be positive, got {frame_count}")

    static_weights = _normalize_weights(weights)
    has_pos = any(
        bone.tracks_by_kind.get(NJD_MTYPE_POS) and bone.tracks_by_kind[NJD_MTYPE_POS].keyframes
        for m in motions for bone in m.bones
    )

    # Per-motion frame mapping: for source motion s with frame_count
    # M_s, output frame f maps to source frame f * (M_s - 1) / (frame_count - 1)
    # so the start/end keyframes of each source align with output 0/end.
    src_frame_count = [max(1, int(m.frame_count)) for m in motions]
    if frame_count == 1:
        frame_scales = [0.0 for _ in motions]
    else:
        frame_scales = [
            (sf - 1) / float(frame_count - 1) for sf in src_frame_count
        ]

    # Build per-output-bone tracks.
    out_bones: List[NjmBoneTracks] = [NjmBoneTracks() for _ in range(bone_count)]

    bind_q = bind_quats or [(0.0, 0.0, 0.0, 1.0)] * bone_count

    for f in range(frame_count):
        # Per-frame weight evaluation (curve only matters when N>=2).
        if transition_curve == TRANSITION_LINEAR or len(motions) == 1:
            frame_weights = list(static_weights)
        else:
            curve_t = _apply_transition_curve(
                f / float(frame_count - 1) if frame_count > 1 else 0.0,
                transition_curve,
            )
            # 2-source: w0 = (1 - curve_t) * static_w0_norm,
            #           w1 = curve_t * static_w1_norm + ... etc.
            # For >2, we ramp every source after the first; the first
            # gets the complementary weight. This matches "fade from
            # source0 to source1+2+...".
            if len(motions) == 2:
                w0 = (1.0 - curve_t)
                w1 = curve_t
                frame_weights = _normalize_weights([w0, w1])
            else:
                # 3+ sources: source0 ramps down, others ramp up uniformly.
                ramped = [(1.0 - curve_t)] + [curve_t * sw for sw in static_weights[1:]]
                frame_weights = _normalize_weights(ramped)

        for bi in range(bone_count):
            quats: List[Tuple[float, float, float, float]] = []
            poss: List[Tuple[float, float, float]] = []
            for si, m in enumerate(motions):
                src_bone = m.bones[bi]
                src_f = f * frame_scales[si]
                ang_track = src_bone.tracks_by_kind.get(NJD_MTYPE_ANG)
                pos_track = src_bone.tracks_by_kind.get(NJD_MTYPE_POS)
                quats.append(_sample_ang_at_frame(
                    ang_track.keyframes if ang_track else [],
                    src_f, bind_q[bi],
                ))
                if has_pos:
                    poss.append(_sample_pos_at_frame(
                        pos_track.keyframes if pos_track else [],
                        src_f,
                    ))
            blended_q = _slerp_n_way(quats, frame_weights)
            rx_b, ry_b, rz_b = _quat_to_zyx_bams(blended_q)
            ang_track = out_bones[bi].tracks_by_kind.get(NJD_MTYPE_ANG)
            if ang_track is None:
                ang_track = NjmTrack(kind=NJD_MTYPE_ANG, keyframes=[], narrow=True)
                out_bones[bi].tracks_by_kind[NJD_MTYPE_ANG] = ang_track
            ang_track.keyframes.append((f, rx_b, ry_b, rz_b))
            if has_pos:
                bx, by, bz = _lerp_n_way(poss, frame_weights)
                pos_track = out_bones[bi].tracks_by_kind.get(NJD_MTYPE_POS)
                if pos_track is None:
                    pos_track = NjmTrack(kind=NJD_MTYPE_POS, keyframes=[], narrow=True)
                    out_bones[bi].tracks_by_kind[NJD_MTYPE_POS] = pos_track
                pos_track.keyframes.append((f, float(bx), float(by), float(bz)))

    type_flags = NJD_MTYPE_ANG | (NJD_MTYPE_POS if has_pos else 0)
    element_count = bin(type_flags).count("1")
    motion = NjmRawMotion(
        frame_count=frame_count,
        type_flags=type_flags,
        inp_fn=element_count,  # interp = 0 (linear), low 4 bits = element_count
        m_data_table_offset=0xC,
        bones=out_bones,
    )
    # Mark every bone as having both POS and ANG slots (so the encoder
    # emits the canonical layout). Empty tracks are tolerated by the
    # writer but every bone needs a slot in the canonical kinds order.
    for bone in out_bones:
        if NJD_MTYPE_ANG not in bone.tracks_by_kind:
            bone.tracks_by_kind[NJD_MTYPE_ANG] = NjmTrack(
                kind=NJD_MTYPE_ANG, keyframes=[], narrow=True,
            )
        if has_pos and NJD_MTYPE_POS not in bone.tracks_by_kind:
            bone.tracks_by_kind[NJD_MTYPE_POS] = NjmTrack(
                kind=NJD_MTYPE_POS, keyframes=[], narrow=True,
            )

    # Stash blend authoring on the motion for diagnostics (mirrors the
    # _retarget_dropped/_retarget_mapped pattern on retarget output).
    motion._blend_weights = list(static_weights)  # type: ignore[attr-defined]
    motion._blend_curve = transition_curve  # type: ignore[attr-defined]
    motion._blend_source_count = len(motions)  # type: ignore[attr-defined]
    return motion


def blend_from_node(
    node: BlendNode,
    *,
    bind_quats: Optional[List[Tuple[float, float, float, float]]] = None,
) -> NjmRawMotion:
    """Convenience: invoke ``blend_motions`` from a ``BlendNode`` dataclass."""
    return blend_motions(
        node.sources,
        node.weights,
        frame_count=node.frame_count,
        transition_curve=node.transition_curve,
        bind_quats=bind_quats,
    )


def summarize_blend(motion: NjmRawMotion) -> dict:
    """Diagnostics for ``/api/anim/blend``: mirror the retarget summary
    style so the front-end can render both with the same component.
    """
    return {
        "frame_count": int(motion.frame_count),
        "bone_count": len(motion.bones),
        "weights": list(getattr(motion, "_blend_weights", []) or []),
        "curve": str(getattr(motion, "_blend_curve", TRANSITION_LINEAR)),
        "source_count": int(getattr(motion, "_blend_source_count", 0) or 0),
    }


__all__ = [
    "TRANSITION_LINEAR",
    "TRANSITION_SMOOTH",
    "TRANSITION_EASE_IN",
    "TRANSITION_EASE_OUT",
    "VALID_TRANSITIONS",
    "BlendNode",
    "blend_motions",
    "blend_from_node",
    "summarize_blend",
]
