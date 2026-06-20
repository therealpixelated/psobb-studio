"""Tests for ``formats.import_external.parse_gltf_with_animations``.

Coverage:
  * Author a minimal in-memory glTF with one rotation animation;
    verify the parser pulls the track shape, sample count, and
    interpolation mode.
  * Author a CUBICSPLINE rotation track; verify it gets demoted to
    LINEAR and the tangent triples are dropped.
  * Verify a translation channel is extracted as 3-tuples (not 4).
  * Run against ``data/animation_assets/standing_typing.glb`` (the
    synthetic typing motion) so the file is exercised by CI and the
    bone count + animation name don't regress.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import pytest

from formats.import_external import (
    ImportedAnimation,
    ImportedTrack,
    parse_gltf_with_animations,
)


# ---------------------------------------------------------------------------
# In-memory glTF authoring helpers
# ---------------------------------------------------------------------------


def _build_minimal_glb(
    *,
    n_frames: int = 4,
    n_bones: int = 2,
    interp: str = "LINEAR",
    include_translation: bool = False,
) -> bytes:
    """Build a tiny glTF with one rotation animation (and optional pos).

    Skeleton: bone 0 = root at origin, bone 1 = child at (1, 0, 0).
    Rotation track on bone 1 sweeps 0->90° around Y over n_frames.
    Optional translation track on bone 0 walks 0..1 along +X over the
    same time range.
    """
    # Times.
    times = [i * 0.1 for i in range(n_frames)]  # 0, 0.1, 0.2, ...
    times_blob = b"".join(struct.pack("<f", t) for t in times)

    # IBM (identity for both bones — we don't care about skin
    # correctness for the parser test).
    ibm_blob = b""
    for _ in range(n_bones):
        ibm_blob += struct.pack(
            "<16f",
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        )

    # Rotation samples on bone 1.
    if interp == "CUBICSPLINE":
        # 3x output values per input frame: in-tangent, value, out-tangent
        sample_count = n_frames * 3
    else:
        sample_count = n_frames
    rot_blob = bytearray()
    for fi in range(n_frames):
        ang = math.pi * 0.5 * fi / max(1, n_frames - 1)  # 0..π/2
        qy = math.sin(ang * 0.5)
        qw = math.cos(ang * 0.5)
        if interp == "CUBICSPLINE":
            # Tangents = zero, value as above.
            rot_blob += struct.pack("<4f", 0, 0, 0, 0)  # in-tangent
            rot_blob += struct.pack("<4f", 0, qy, 0, qw)
            rot_blob += struct.pack("<4f", 0, 0, 0, 0)  # out-tangent
        else:
            rot_blob += struct.pack("<4f", 0, qy, 0, qw)

    # Optional translation samples on bone 0.
    pos_blob = b""
    if include_translation:
        for fi in range(n_frames):
            x = fi / max(1, n_frames - 1)
            pos_blob += struct.pack("<3f", x, 0, 0)

    # Layout the binary buffer.
    ibm_off = 0
    times_off = ibm_off + len(ibm_blob)
    rot_off = times_off + len(times_blob)
    pos_off = rot_off + len(rot_blob)

    buf = ibm_blob + times_blob + rot_blob + pos_blob
    if len(buf) & 3:
        buf = buf + b"\0" * (4 - (len(buf) & 3))

    # Build glTF JSON.
    nodes = [
        {"name": "BoneRoot",  "translation": [0, 0, 0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]},
        {"name": "BoneChild", "translation": [1, 0, 0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]},
    ]
    nodes[0]["children"] = [1]
    skin = {
        "joints": [0, 1],
        "inverseBindMatrices": 0,
        "skeleton": 0,
    }
    buffer_views = [
        {"buffer": 0, "byteOffset": ibm_off,   "byteLength": len(ibm_blob)},
        {"buffer": 0, "byteOffset": times_off, "byteLength": len(times_blob)},
        {"buffer": 0, "byteOffset": rot_off,   "byteLength": len(rot_blob)},
    ]
    accessors = [
        {"bufferView": 0, "componentType": 5126, "count": n_bones, "type": "MAT4"},
        {"bufferView": 1, "componentType": 5126, "count": n_frames, "type": "SCALAR",
         "min": [0.0], "max": [times[-1]]},
        {"bufferView": 2, "componentType": 5126, "count": sample_count, "type": "VEC4"},
    ]
    channels = [
        {"sampler": 0, "target": {"node": 1, "path": "rotation"}},
    ]
    samplers = [
        {"input": 1, "output": 2, "interpolation": interp},
    ]
    if include_translation:
        buffer_views.append({"buffer": 0, "byteOffset": pos_off, "byteLength": len(pos_blob)})
        accessors.append(
            {"bufferView": 3, "componentType": 5126, "count": n_frames, "type": "VEC3"}
        )
        samplers.append({"input": 1, "output": 3, "interpolation": "LINEAR"})
        channels.append({"sampler": 1, "target": {"node": 0, "path": "translation"}})

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "skins": [skin],
        "buffers": [{"byteLength": len(buf)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "animations": [{
            "name": "Test",
            "channels": channels,
            "samplers": samplers,
        }],
    }
    json_text = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    if len(json_text) & 3:
        json_text = json_text + b" " * (4 - (len(json_text) & 3))

    glb = bytearray()
    glb.extend(b"glTF")
    glb.extend(struct.pack("<I", 2))
    total_len = 12 + 8 + len(json_text) + 8 + len(buf)
    glb.extend(struct.pack("<I", total_len))
    glb.extend(struct.pack("<I", len(json_text)))
    glb.extend(b"JSON")
    glb.extend(json_text)
    glb.extend(struct.pack("<I", len(buf)))
    glb.extend(b"BIN\0")
    glb.extend(buf)
    return bytes(glb)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_minimal_rotation_animation_parses():
    """One LINEAR rotation track on a 2-bone skeleton round-trips."""
    glb = _build_minimal_glb(n_frames=4, n_bones=2, interp="LINEAR")
    imp = parse_gltf_with_animations(glb)

    assert len(imp.model.bones) == 2
    assert imp.model.bones[0].name == "BoneRoot"
    assert imp.model.bones[1].name == "BoneChild"
    assert imp.model.bones[1].parent_idx == 0

    assert len(imp.animations) == 1
    anim = imp.animations[0]
    assert anim.name == "Test"
    assert len(anim.tracks) == 1
    track = anim.tracks[0]
    assert track.bone_idx == 1  # rotation targeted bone 1
    assert track.channel == "rotation"
    assert track.interp == "LINEAR"
    assert len(track.times) == 4
    assert len(track.values) == 4
    # First frame is identity (angle=0); last is 90° around Y.
    assert track.values[0] == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-5)
    qx, qy, qz, qw = track.values[-1]
    assert qx == pytest.approx(0.0, abs=1e-5)
    assert qy == pytest.approx(math.sin(math.pi / 4), abs=1e-5)
    assert qz == pytest.approx(0.0, abs=1e-5)
    assert qw == pytest.approx(math.cos(math.pi / 4), abs=1e-5)
    # Duration = times[-1].
    assert anim.duration_seconds == pytest.approx(0.3, abs=1e-5)


def test_cubicspline_demoted_to_linear():
    """CUBICSPLINE rotation track has tangents stripped + interp -> LINEAR."""
    glb = _build_minimal_glb(n_frames=4, n_bones=2, interp="CUBICSPLINE")
    imp = parse_gltf_with_animations(glb)
    track = imp.animations[0].tracks[0]
    assert track.interp == "LINEAR"
    assert len(track.values) == 4  # NOT 12 (tangents dropped)
    # Last frame should be 90° around Y (the value, not a tangent).
    qx, qy, qz, qw = track.values[-1]
    assert qy == pytest.approx(math.sin(math.pi / 4), abs=1e-5)
    assert qw == pytest.approx(math.cos(math.pi / 4), abs=1e-5)


def test_translation_extracted_as_vec3():
    """A translation channel produces 3-tuples, not 4-tuples."""
    glb = _build_minimal_glb(n_frames=4, n_bones=2, include_translation=True)
    imp = parse_gltf_with_animations(glb)
    anim = imp.animations[0]
    # Two tracks now: rotation on bone 1, translation on bone 0.
    by_ch = {(t.bone_idx, t.channel): t for t in anim.tracks}
    assert (1, "rotation") in by_ch
    assert (0, "translation") in by_ch
    pos_track = by_ch[(0, "translation")]
    assert len(pos_track.values) == 4
    assert all(len(v) == 3 for v in pos_track.values)
    # Walks 0 -> 1 along +X.
    assert pos_track.values[0] == pytest.approx((0.0, 0.0, 0.0), abs=1e-5)
    assert pos_track.values[-1] == pytest.approx((1.0, 0.0, 0.0), abs=1e-5)


def test_synthetic_typing_glb_parses():
    """The shipped synthetic typing animation parses cleanly."""
    p = Path("data/animation_assets/standing_typing.glb")
    if not p.is_file():
        pytest.skip(f"animation asset missing: {p}")
    imp = parse_gltf_with_animations(str(p))
    assert len(imp.model.bones) == 22
    assert len(imp.animations) == 1
    anim = imp.animations[0]
    assert anim.name == "StandingTyping"
    # 22 rotation channels, no translation in this synthetic clip.
    assert len(anim.tracks) == 22
    assert all(t.channel == "rotation" for t in anim.tracks)
    assert all(len(t.values) == 90 for t in anim.tracks)
    # Duration ~3 seconds.
    assert anim.duration_seconds == pytest.approx(89 / 30.0, abs=1e-3)
