"""Synthetic procedural "standing typing" animation generator.

Used as the Task-1 fallback when no public-domain typing motion can be
sourced cleanly online. Produces a 90-frame, 30 fps glTF .glb file
containing:

  * A standard humanoid skeleton with 22 joints whose names match the
    Mixamo / Mecanim convention (Hips, Spine, Spine1, Neck, Head,
    LeftShoulder, LeftArm, LeftForeArm, LeftHand, RightShoulder,
    RightArm, RightForeArm, RightHand, LeftUpLeg, LeftLeg, LeftFoot,
    RightUpLeg, RightLeg, RightFoot, plus three optional jaw / eye
    bones that have no animation but match what Mixamo writes).
  * One animation called "StandingTyping" with rotation tracks on
    every animated bone (no translation tracks except the hips, which
    sit at bind pose). LINEAR interpolation, 90 keyframes per track at
    1/30 second spacing.

Pose authoring:
  * Standing idle: knees slightly flexed, feet shoulder-width apart,
    hips at bind-pose origin (Y up).
  * Arms held forward at ~30° elbow flex, hands hovering at hip-height
    on either side of an imaginary keyboard.
  * Wrists wiggle on a 3 Hz sine wave with low amplitude (~5°) — gives
    the "fingers tapping" illusion without per-finger bones.
  * Forearms rock on a 1.5 Hz sine wave with ~3° amplitude — adds the
    forward/back micro-motion of typing.
  * Torso sways on a 0.5 Hz sine wave with ~2° amplitude — keeps the
    pose from looking rigid.
  * Head bobs on a 0.5 Hz sine wave with ~1.5° amplitude.

Why a glTF:
  * The Task-2 parser ingests glTF directly; using a glTF round-trip
    means the synthetic generator + the imported real-Mixamo path
    share the SAME pipeline and we don't need a second code path.
  * pygltflib is already installed.
  * The author tooling is one Python module; no external binaries.

License: this module + its output are CC0 / public domain. The output
glb is a pure procedural generation, no third-party motion-capture
data is referenced.
"""
from __future__ import annotations

import base64
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Skeleton authoring (T-pose bind, Y-up, +Z forward — Mixamo convention)
# ---------------------------------------------------------------------------
#
# Bind positions are in metres so they survive the glTF importer's
# default scale. Limb lengths target a 1.7 m human. Arms are held
# slightly out (45° from the torso) to ease the "lift forward to type"
# pose.


@dataclass
class _BoneAuthor:
    name: str
    parent: int  # -1 for root
    bind_translation: Tuple[float, float, float]  # local space, metres
    # Bind rotation is identity for every bone — the rest pose has
    # everything axis-aligned to the parent.


def _make_skeleton() -> List[_BoneAuthor]:
    """Build the 22-bone humanoid skeleton.

    Names match the Mixamo joint convention so the retargeter's
    auto-name table works without the user supplying a custom map.
    """
    return [
        _BoneAuthor("Hips",            -1, (0.00,  0.95, 0.00)),
        _BoneAuthor("Spine",            0, (0.00,  0.10, 0.00)),
        _BoneAuthor("Spine1",           1, (0.00,  0.20, 0.00)),
        _BoneAuthor("Neck",             2, (0.00,  0.20, 0.00)),
        _BoneAuthor("Head",             3, (0.00,  0.10, 0.00)),
        _BoneAuthor("LeftShoulder",     2, (0.10,  0.18, 0.00)),
        _BoneAuthor("LeftArm",          5, (0.12,  0.00, 0.00)),
        _BoneAuthor("LeftForeArm",      6, (0.30,  0.00, 0.00)),
        _BoneAuthor("LeftHand",         7, (0.25,  0.00, 0.00)),
        _BoneAuthor("RightShoulder",    2, (-0.10, 0.18, 0.00)),
        _BoneAuthor("RightArm",         9, (-0.12, 0.00, 0.00)),
        _BoneAuthor("RightForeArm",    10, (-0.30, 0.00, 0.00)),
        _BoneAuthor("RightHand",       11, (-0.25, 0.00, 0.00)),
        _BoneAuthor("LeftUpLeg",        0, (0.10, -0.05, 0.00)),
        _BoneAuthor("LeftLeg",         13, (0.00, -0.45, 0.00)),
        _BoneAuthor("LeftFoot",        14, (0.00, -0.45, 0.05)),
        _BoneAuthor("RightUpLeg",       0, (-0.10, -0.05, 0.00)),
        _BoneAuthor("RightLeg",        16, (0.00, -0.45, 0.00)),
        _BoneAuthor("RightFoot",       17, (0.00, -0.45, 0.05)),
        _BoneAuthor("Jaw",              4, (0.00, -0.05, 0.07)),
        _BoneAuthor("LeftEye",          4, (0.03,  0.03, 0.08)),
        _BoneAuthor("RightEye",         4, (-0.03, 0.03, 0.08)),
    ]


# ---------------------------------------------------------------------------
# Quaternion helpers (axis-angle -> [x, y, z, w])
# ---------------------------------------------------------------------------


def _quat_axis_angle(ax: float, ay: float, az: float, angle_rad: float) -> Tuple[float, float, float, float]:
    """Build a unit quaternion from an axis (will be normalised) + angle.

    Returns ``(qx, qy, qz, qw)`` matching the glTF rotation channel
    convention.
    """
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    ax /= n; ay /= n; az /= n
    half = angle_rad * 0.5
    s = math.sin(half)
    return (ax * s, ay * s, az * s, math.cos(half))


def _quat_mul(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Hamilton product ``a * b``; both inputs in (x, y, z, w) order."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


# ---------------------------------------------------------------------------
# Pose authoring — per-frame quaternion per bone
# ---------------------------------------------------------------------------
#
# Conventions:
#   * Forward arm typing pose: shoulder pitched ~70° forward, elbow
#     flexed ~30°. The shoulder's forward axis is +Z, the elbow's flex
#     axis is +X (perpendicular to upper-arm).
#   * Wrist wiggle: small +X (pitch) + small +Y (yaw) sine combo.
#   * Sway: hips on a slow Y-axis sine, head + spine on a slower one.


def _typing_pose(bone_name: str, t_seconds: float, fps: int) -> Tuple[float, float, float, float]:
    """Return the per-frame rotation quaternion for ``bone_name`` at ``t_seconds``.

    Identity (0, 0, 0, 1) for bones that don't move (legs, feet, etc.).
    """
    # Shorthand frequencies.
    f_wrist = 3.0    # finger taps ~3 Hz
    f_forearm = 1.5  # forearm bob
    f_torso = 0.5    # gentle sway
    f_head = 0.5

    if bone_name == "Hips":
        # Tiny lateral sway.
        ang = math.radians(0.4 * math.sin(2.0 * math.pi * f_torso * t_seconds))
        return _quat_axis_angle(0.0, 0.0, 1.0, ang)

    if bone_name == "Spine":
        ang = math.radians(0.8 * math.sin(2.0 * math.pi * f_torso * t_seconds + 0.7))
        return _quat_axis_angle(1.0, 0.0, 0.0, ang)

    if bone_name == "Spine1":
        ang = math.radians(1.0 * math.sin(2.0 * math.pi * f_torso * t_seconds + 0.3))
        return _quat_axis_angle(0.0, 0.0, 1.0, ang)

    if bone_name == "Neck":
        ang = math.radians(0.6 * math.sin(2.0 * math.pi * f_head * t_seconds))
        return _quat_axis_angle(1.0, 0.0, 0.0, ang)

    if bone_name == "Head":
        # Small vertical bob like watching the keyboard.
        pitch = math.radians(7.0 + 1.5 * math.sin(2.0 * math.pi * f_head * t_seconds))
        return _quat_axis_angle(1.0, 0.0, 0.0, pitch)

    # Arms — both shoulders pitched forward + slight inward for the
    # "hands on keyboard" pose. We bake the static pitch as a constant
    # rotation and add a small per-frame oscillation.
    if bone_name == "LeftShoulder":
        # Tiny micro-shrug.
        return _quat_axis_angle(0.0, 0.0, 1.0,
                                math.radians(0.5 * math.sin(2.0 * math.pi * f_torso * t_seconds + 1.0)))
    if bone_name == "RightShoulder":
        return _quat_axis_angle(0.0, 0.0, 1.0,
                                math.radians(0.5 * math.sin(2.0 * math.pi * f_torso * t_seconds + 1.5)))

    if bone_name == "LeftArm":
        # 70° forward (around +X if the arm extends along +X in bind),
        # plus 20° inward (around +Y).
        base = _quat_axis_angle(0.0, 0.0, 1.0, math.radians(-70.0))
        inward = _quat_axis_angle(0.0, 1.0, 0.0, math.radians(-20.0))
        bob = _quat_axis_angle(0.0, 0.0, 1.0,
                                math.radians(2.0 * math.sin(2.0 * math.pi * f_forearm * t_seconds)))
        return _quat_mul(_quat_mul(base, inward), bob)

    if bone_name == "RightArm":
        base = _quat_axis_angle(0.0, 0.0, 1.0, math.radians(70.0))
        inward = _quat_axis_angle(0.0, 1.0, 0.0, math.radians(20.0))
        bob = _quat_axis_angle(0.0, 0.0, 1.0,
                                math.radians(2.0 * math.sin(2.0 * math.pi * f_forearm * t_seconds + math.pi)))
        return _quat_mul(_quat_mul(base, inward), bob)

    if bone_name == "LeftForeArm":
        # Static elbow flex + 1.5 Hz forearm pump.
        flex = _quat_axis_angle(0.0, 1.0, 0.0, math.radians(60.0))
        pump = _quat_axis_angle(0.0, 1.0, 0.0,
                                 math.radians(3.0 * math.sin(2.0 * math.pi * f_forearm * t_seconds)))
        return _quat_mul(flex, pump)

    if bone_name == "RightForeArm":
        flex = _quat_axis_angle(0.0, 1.0, 0.0, math.radians(-60.0))
        pump = _quat_axis_angle(0.0, 1.0, 0.0,
                                 math.radians(3.0 * math.sin(2.0 * math.pi * f_forearm * t_seconds + 1.0)))
        return _quat_mul(flex, pump)

    if bone_name == "LeftHand":
        # Wrist wiggle on the typing axis (yaw + pitch).
        yaw = _quat_axis_angle(0.0, 1.0, 0.0,
                                math.radians(4.0 * math.sin(2.0 * math.pi * f_wrist * t_seconds)))
        pitch = _quat_axis_angle(1.0, 0.0, 0.0,
                                  math.radians(3.0 * math.sin(2.0 * math.pi * f_wrist * t_seconds + 0.5)))
        return _quat_mul(yaw, pitch)

    if bone_name == "RightHand":
        yaw = _quat_axis_angle(0.0, 1.0, 0.0,
                                math.radians(4.0 * math.sin(2.0 * math.pi * f_wrist * t_seconds + math.pi / 2)))
        pitch = _quat_axis_angle(1.0, 0.0, 0.0,
                                  math.radians(3.0 * math.sin(2.0 * math.pi * f_wrist * t_seconds + math.pi)))
        return _quat_mul(yaw, pitch)

    if bone_name == "LeftUpLeg":
        # Knees slightly bent for "standing posture", ~5° pitch back.
        return _quat_axis_angle(1.0, 0.0, 0.0, math.radians(-5.0))
    if bone_name == "RightUpLeg":
        return _quat_axis_angle(1.0, 0.0, 0.0, math.radians(-5.0))
    if bone_name == "LeftLeg":
        return _quat_axis_angle(1.0, 0.0, 0.0, math.radians(8.0))
    if bone_name == "RightLeg":
        return _quat_axis_angle(1.0, 0.0, 0.0, math.radians(8.0))
    # Feet, jaw, eyes — bind pose.
    return (0.0, 0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# glTF / GLB writer
# ---------------------------------------------------------------------------
#
# We produce a SKINNED model with a single empty mesh. The skeleton is
# emitted as proper glTF nodes + a `skins[0]` block with all joints.
# The animation is one `animations[0]` with rotation channels for the
# bones whose pose isn't bind-identity (we still emit ALL bones for
# coverage, the retargeter will skip identity quats).


def _build_glb(
    bones: List[_BoneAuthor],
    frame_count: int,
    fps: int,
    motion_name: str = "StandingTyping",
) -> bytes:
    """Author the glTF 2.0 + binary buffer and pack as a .glb.

    Layout of the binary buffer:

        +-----------------------------------------+
        | inverseBindMatrices  (n_bones * 64 B)   |
        | animation input times (n_frames * 4 B)  |
        | per-bone rotation samples (n_frames     |
        |   * 16 B per bone, n_bones bones)       |
        +-----------------------------------------+

    glTF requires every bufferView's byteOffset to be aligned to its
    component's size. We pad with zeros after the inverseBindMatrices
    block because the input-times accessor is f32 (4-byte aligned) so
    no extra padding needed; same for the f32x4 quaternion samples.
    """
    n_bones = len(bones)

    # ---- inverseBindMatrices: identity per bone in WORLD space ----
    # Per glTF spec, IBM is the inverse of the bone's WORLD bind
    # transform. For a hierarchy of translation-only bones, the
    # inverse-world-translation has the bone's position negated.
    world_pos: Dict[int, Tuple[float, float, float]] = {}
    for i, b in enumerate(bones):
        if b.parent < 0:
            world_pos[i] = b.bind_translation
        else:
            wp = world_pos[b.parent]
            world_pos[i] = (
                wp[0] + b.bind_translation[0],
                wp[1] + b.bind_translation[1],
                wp[2] + b.bind_translation[2],
            )
    ibm_blob = bytearray()
    for i in range(n_bones):
        wx, wy, wz = world_pos[i]
        # Column-major 4x4 with translation in column 3 (rows are
        # interleaved, so position 12, 13, 14 in flat layout).
        m = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            -wx, -wy, -wz, 1.0,
        ]
        for v in m:
            ibm_blob.extend(struct.pack("<f", v))
    ibm_offset = 0
    ibm_size = len(ibm_blob)

    # ---- animation input times ----
    times_blob = bytearray()
    for f in range(frame_count):
        times_blob.extend(struct.pack("<f", f / float(fps)))
    times_offset = ibm_size
    times_size = len(times_blob)

    # ---- per-bone rotation samples ----
    # glTF stores rotation samples as VEC4 f32 in (x, y, z, w) order.
    rot_blobs: List[bytes] = []
    rot_offsets: List[int] = []
    cursor = times_offset + times_size
    for i, b in enumerate(bones):
        rot_offsets.append(cursor)
        blob = bytearray()
        for f in range(frame_count):
            t = f / float(fps)
            qx, qy, qz, qw = _typing_pose(b.name, t, fps)
            blob.extend(struct.pack("<ffff", qx, qy, qz, qw))
        rot_blobs.append(bytes(blob))
        cursor += len(blob)

    bin_buffer = bytes(ibm_blob) + bytes(times_blob) + b"".join(rot_blobs)
    bin_size = len(bin_buffer)
    # GLB chunks must be 4-byte aligned.
    if bin_size & 3:
        bin_buffer = bin_buffer + b"\0" * (4 - (bin_size & 3))

    # ---- glTF JSON ----
    nodes = []
    for i, b in enumerate(bones):
        node = {
            "name": b.name,
            "translation": list(b.bind_translation),
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0],
        }
        children = [j for j, c in enumerate(bones) if c.parent == i]
        if children:
            node["children"] = children
        nodes.append(node)

    # Single root scene whose only top-level node is the skeleton root
    # (bone with parent==-1, conventionally bone 0).
    scene_roots = [i for i, b in enumerate(bones) if b.parent < 0]

    # Buffer views.
    buffer_views = [
        {"buffer": 0, "byteOffset": ibm_offset, "byteLength": ibm_size},
        {"buffer": 0, "byteOffset": times_offset, "byteLength": times_size},
    ]
    # One bufferView per bone's rotation samples.
    rot_bv_indices: List[int] = []
    for i in range(n_bones):
        bv_idx = len(buffer_views)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": rot_offsets[i],
            "byteLength": len(rot_blobs[i]),
        })
        rot_bv_indices.append(bv_idx)

    # Accessors.
    accessors = [
        {  # inverseBindMatrices
            "bufferView": 0,
            "componentType": 5126,  # FLOAT
            "count": n_bones,
            "type": "MAT4",
        },
        {  # input times
            "bufferView": 1,
            "componentType": 5126,
            "count": frame_count,
            "type": "SCALAR",
            "min": [0.0],
            "max": [(frame_count - 1) / float(fps)],
        },
    ]
    rot_acc_indices: List[int] = []
    for i in range(n_bones):
        acc_idx = len(accessors)
        accessors.append({
            "bufferView": rot_bv_indices[i],
            "componentType": 5126,
            "count": frame_count,
            "type": "VEC4",
        })
        rot_acc_indices.append(acc_idx)

    # Skin.
    skin = {
        "name": "TypingSkeleton",
        "joints": list(range(n_bones)),
        "inverseBindMatrices": 0,
        "skeleton": scene_roots[0],
    }

    # Animation.
    channels = []
    samplers = []
    for i in range(n_bones):
        smp_idx = len(samplers)
        samplers.append({
            "input": 1,                   # times accessor
            "output": rot_acc_indices[i], # rotation accessor
            "interpolation": "LINEAR",
        })
        channels.append({
            "sampler": smp_idx,
            "target": {"node": i, "path": "rotation"},
        })

    gltf = {
        "asset": {"version": "2.0", "generator": "psobb_editor.synth_typing"},
        "scene": 0,
        "scenes": [{"nodes": scene_roots}],
        "nodes": nodes,
        "skins": [skin],
        "buffers": [{"byteLength": len(bin_buffer)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "animations": [{
            "name": motion_name,
            "channels": channels,
            "samplers": samplers,
        }],
    }
    json_text = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    # Pad JSON to 4 bytes with spaces (glTF spec requires).
    if len(json_text) & 3:
        json_text = json_text + b" " * (4 - (len(json_text) & 3))

    # Build GLB.
    glb = bytearray()
    glb.extend(b"glTF")
    glb.extend(struct.pack("<I", 2))  # version
    total_len = 12 + 8 + len(json_text) + 8 + len(bin_buffer)
    glb.extend(struct.pack("<I", total_len))
    # JSON chunk.
    glb.extend(struct.pack("<I", len(json_text)))
    glb.extend(b"JSON")
    glb.extend(json_text)
    # BIN chunk.
    glb.extend(struct.pack("<I", len(bin_buffer)))
    glb.extend(b"BIN\0")
    glb.extend(bin_buffer)
    return bytes(glb)


def write_typing_glb(out_path: Path, *, frame_count: int = 90, fps: int = 30) -> Path:
    """Write the synthetic typing animation to ``out_path``.

    Returns the resolved path. ``frame_count`` defaults to 90 (= 3
    seconds at 30 fps), which gives one full Spine sway period (T=2 s)
    plus extra room for the asynchronous wrist wiggles to look varied.
    """
    bones = _make_skeleton()
    glb = _build_glb(bones, frame_count, fps)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_bytes(glb)
    import os as _os
    _os.replace(tmp, out_path)
    return out_path


__all__ = ["write_typing_glb"]


if __name__ == "__main__":
    # Allow `python -m formats.synth_typing data/animation_assets/standing_typing.glb`
    import sys
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/animation_assets/standing_typing.glb")
    out = write_typing_glb(target)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
