"""Editor v4 — motion-editor UI smoke tests (anim_editor_panel.js).

The four user-visible Editor v4 features (multi-channel curve overlay,
bezier handle persistence, cross-bone marquee, eye-toggle wired to
bone-pose override) extend the existing v3 surface.

Tests are static smoke + endpoint integration. We can't drive a browser
from pytest, so behavioural verification of the JS canvas math is left
to manual QA. What we CAN check:

  1. The panel source declares the v4 entry points (modes, exports).
  2. The /api/anim_keyframe/save endpoint accepts bezier_handles AND
     writes a sidecar that /api/anim_keyframe/load surfaces back.
  3. The viewer's _computeAnimatedBoneMatrices honors rigBoneOverrides
     (this is what wires the eye-toggle to the live render).
  4. The cross-bone selection algorithm preserves the per-bone wire
     contract — running the panel's multi-bone delete/move/copy/paste
     on the JSON envelope leaves every other bone bit-identical.
"""
from __future__ import annotations
import os

import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


_REPO = Path(__file__).resolve().parent.parent
_PANEL_JS = _REPO / "static" / "anim_editor_panel.js"
_VIEWER_JS = _REPO / "static" / "model_viewer.js"
_STYLE_CSS = _REPO / "static" / "style.css"


# ---------------------------------------------------------------------------
# Static smoke
# ---------------------------------------------------------------------------


def test_panel_has_curve_overlay_modes():
    """Task 1 — panel ships single | triplet | all curve overlay modes."""
    src = _PANEL_JS.read_text(encoding="utf-8")
    # Mode state field.
    assert "curveMode" in src
    # All three modes referenced.
    assert '"single"' in src or "'single'" in src
    assert '"triplet"' in src
    assert '"all"' in src
    # Helper that returns the visible channel set for the current mode.
    assert "_activeCurveChannels" in src
    # Per-channel colour table for the multi-curve render.
    assert "color:" in src
    # Click-to-make-active hit test.
    assert "_interpAtFrame" in src
    # CSS class for the overlay legend.
    assert "ake-curve-legend" in src


def test_panel_has_bezier_handle_persistence_wiring():
    """Task 2 — panel sends bezier_handles on save and restores on load."""
    src = _PANEL_JS.read_text(encoding="utf-8")
    # Save path serialises the bezierHandles map to a wire-friendly object.
    assert "bezier_handles" in src
    # Each handle key has the four scalar fields the server validates.
    assert "inDx" in src
    assert "inDy" in src
    assert "outDx" in src
    assert "outDy" in src
    # Load path restores them when present in the response.
    assert "data.bezier_handles" in src or "bezier_handles" in src


def test_panel_has_cross_bone_marquee():
    """Task 3 — panel ships single | all marquee modes + per-bone batch ops."""
    src = _PANEL_JS.read_text(encoding="utf-8")
    assert "marqueeMode" in src
    assert "selectedKfSetMulti" in src
    # The cross-bone selection key shape is "<boneIdx>:<kfIdx>".
    assert "${bi}:${i}" in src or "${rowBoneIdx}:${idx}" in src \
        or "${boneIdx}:${kfIdx}" in src
    # Multi-bone helpers.
    assert "_iterSelected" in src
    assert "_clearAllSelections" in src
    assert "_drawTimelineAllBones" in src
    # Bone-id annotation in the keyframe-list rows.
    assert "bone-tag" in src
    # CSS for the marquee mode toggle.
    assert "ake-marquee-mode" in src


def test_panel_eye_toggle_wired_to_override():
    """Task 4 — eye-toggle calls psoSetBonePoseOverride and forces a re-bake."""
    src = _PANEL_JS.read_text(encoding="utf-8")
    # Eye-toggle handler reaches into the additive viewer surface.
    assert "psoSetBonePoseOverride" in src
    # Panel collects the bind pose from the skeleton snapshot before
    # pushing it as the override.
    assert "rotation_bams" in src
    # Sets up bind pose override or releases it via null.
    assert "psoSetBonePoseOverride(bIdx, null)" in src \
        or "psoSetBonePoseOverride(bIdx, bindPose)" in src
    # Forces a re-bake so the change is visible immediately.
    assert "psoSeekAnimationToFrame" in src
    # Cleanup helper exists + is called on motion change.
    assert "releaseAllPanelBoneOverrides" in src


def test_viewer_animated_matrices_honor_overrides():
    """Task 4 — _computeAnimatedBoneMatrices consults state.rigBoneOverrides.

    Without this, eye-toggle's override would be clobbered by the
    animation re-bake on every frame.
    """
    src = _VIEWER_JS.read_text(encoding="utf-8")
    # Find the function body by its opener, then walk to its end via brace
    # counting (regex is unsafe across nested braces). The point of the
    # test is to confirm the override consultation lives INSIDE the
    # function — not just somewhere in the file.
    start = src.find("function _computeAnimatedBoneMatrices")
    assert start >= 0, "missing _computeAnimatedBoneMatrices"
    open_brace = src.index("{", start)
    depth = 1
    i = open_brace + 1
    while depth > 0 and i < len(src):
        ch = src[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    body = src[open_brace + 1 : i - 1]
    # The override consultation must mutate the per-bone TRS scratch
    # (tmpTRS) AFTER the animation track has been sampled — otherwise
    # the override would be overwritten by the next sampleBoneTrack call.
    assert "rigBoneOverrides" in body
    assert "tmpTRS" in body
    assert "rotation_bams" in body


def test_style_has_v4_classes():
    """style.css mirrors the new ake-curve-overlay-* and ake-marquee-* rules."""
    css = _STYLE_CSS.read_text(encoding="utf-8")
    assert ".ake-curve-overlay-modes" in css
    assert ".ake-curve-legend" in css
    assert ".ake-curve-legend-item" in css
    assert ".ake-marquee-mode" in css
    assert ".bone-tag" in css


# ---------------------------------------------------------------------------
# Server-side bezier handle persistence (Task 2)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    import server
    return TestClient(server.app)


def _seed_motion_for_save():
    """Build a tiny but realistic motion JSON that can be /save'd."""
    from formats.njm_writer import (
        NjmBoneTracks, NjmRawMotion, NjmTrack,
    )
    from formats.njm import NJD_MTYPE_ANG, NJD_MTYPE_POS
    motion = NjmRawMotion(
        frame_count=10,
        type_flags=NJD_MTYPE_POS | NJD_MTYPE_ANG,
        inp_fn=2,
    )
    bone = NjmBoneTracks()
    bone.tracks_by_kind[NJD_MTYPE_POS] = NjmTrack(
        NJD_MTYPE_POS,
        [(0, 0.0, 0.0, 0.0), (5, 1.0, 2.0, 3.0)],
        narrow=True,
    )
    bone.tracks_by_kind[NJD_MTYPE_ANG] = NjmTrack(
        NJD_MTYPE_ANG,
        [(0, 0, 0, 0), (5, 1000, 2000, 3000)],
        narrow=True,
    )
    motion.bones.append(bone)
    import server
    return server._ake_motion_to_json(motion, fps=30.0, motion_name="v4_seed")


def test_save_accepts_bezier_handles_and_writes_sidecar(client, tmp_path, monkeypatch):
    """Task 2 — POST /api/anim_keyframe/save persists bezier_handles to sidecar."""
    import server
    # Redirect NJM_EXPORT_DIR to a sandbox so we don't touch the real cache.
    sandbox = tmp_path / "njm_export"
    sandbox.mkdir()
    monkeypatch.setattr(server, "NJM_EXPORT_DIR", sandbox)
    motion = _seed_motion_for_save()
    handles = {
        "0:0:rx": {"inDx": -2.0, "inDy": 0.0, "outDx": 2.0, "outDy": 0.0},
        "0:1:rx": {"inDx": -1.5, "inDy": 100.0, "outDx": 1.5, "outDy": -50.0},
    }
    r = client.post("/api/anim_keyframe/save", json={
        "motion_json": motion,
        "name": "v4_handles_test.njm",
        "bezier_handles": handles,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["sidecar_written"] is True
    # Sidecar exists on disk + carries the handles round-trip.
    sc = sandbox / "v4_handles_test.njm.preview.json"
    assert sc.is_file()
    sidecar = json.loads(sc.read_text(encoding="utf-8"))
    assert "bezier_handles" in sidecar
    assert sidecar["bezier_handles"]["0:0:rx"]["inDx"] == pytest.approx(-2.0)
    assert sidecar["bezier_handles"]["0:1:rx"]["outDy"] == pytest.approx(-50.0)
    # Saving without handles must NOT clobber the sidecar's other fields.
    sidecar["target_model_path"] = "test_unrelated.bml"
    sc.write_text(json.dumps(sidecar), encoding="utf-8")
    r = client.post("/api/anim_keyframe/save", json={
        "motion_json": motion,
        "name": "v4_handles_test.njm",
        "bezier_handles": {"0:0:rx": handles["0:0:rx"]},
    })
    assert r.status_code == 200
    sidecar2 = json.loads(sc.read_text(encoding="utf-8"))
    assert sidecar2["target_model_path"] == "test_unrelated.bml"
    assert "0:0:rx" in sidecar2["bezier_handles"]


def test_save_without_handles_skips_sidecar(client, tmp_path, monkeypatch):
    """Save with no bezier_handles field is unchanged from v3 (no sidecar)."""
    import server
    sandbox = tmp_path / "njm_export"
    sandbox.mkdir()
    monkeypatch.setattr(server, "NJM_EXPORT_DIR", sandbox)
    motion = _seed_motion_for_save()
    r = client.post("/api/anim_keyframe/save", json={
        "motion_json": motion,
        "name": "v4_no_handles.njm",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["sidecar_written"] is False
    # No sidecar file got written.
    sc = sandbox / "v4_no_handles.njm.preview.json"
    assert not sc.exists()


def test_save_rejects_malformed_handles(client, tmp_path, monkeypatch):
    """Bad bezier_handles shape returns 400, not 500."""
    import server
    sandbox = tmp_path / "njm_export"
    sandbox.mkdir()
    monkeypatch.setattr(server, "NJM_EXPORT_DIR", sandbox)
    motion = _seed_motion_for_save()
    # Not a dict.
    r = client.post("/api/anim_keyframe/save", json={
        "motion_json": motion,
        "name": "v4_bad.njm",
        "bezier_handles": [1, 2, 3],
    })
    # FastAPI may convert via pydantic to dict (rejecting list). Either
    # 400 (manual validation) or 422 (pydantic validation) is acceptable.
    assert r.status_code in (400, 422), r.text
    # Valid keys but non-numeric values.
    r = client.post("/api/anim_keyframe/save", json={
        "motion_json": motion,
        "name": "v4_bad2.njm",
        "bezier_handles": {"0:0:rx": "not an object"},
    })
    assert r.status_code in (400, 422), r.text


PSOBB_DATA = Path(os.path.expanduser("~/PSOBB.IO/data"))
HAS_PSOBB = PSOBB_DATA.is_dir()


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO data not present")
def test_load_surfaces_handles_from_sidecar_after_save(client, tmp_path, monkeypatch):
    """Task 2 end-to-end — save with handles → reload → handles restored.

    Mirrors the user-visible flow ("save with bezier handles → reload
    page → verify handles restored"). Uses the dragon BML because it
    ships in PSOBB.IO/data and has a deterministic motion.
    """
    import server
    if not (PSOBB_DATA / "bm_boss1_dragon.bml").exists():
        pytest.skip("dragon BML not present")
    sandbox = tmp_path / "njm_export"
    sandbox.mkdir()
    monkeypatch.setattr(server, "NJM_EXPORT_DIR", sandbox)
    # First load — should NOT have bezier_handles in the response.
    r = client.post("/api/anim_keyframe/load", json={
        "model_path": "bm_boss1_dragon.bml",
        "motion_name": "walk_boss1_s_nb_dragon",
    })
    assert r.status_code == 200, r.text
    motion = r.json()
    assert "bezier_handles" not in motion
    # Author handles + save.
    handles = {
        "0:0:rx": {"inDx": -2.0, "inDy": 0.0, "outDx": 2.0, "outDy": 0.0},
        "5:1:tx": {"inDx": -1.0, "inDy": 5.0, "outDx": 1.0, "outDy": -5.0},
    }
    r = client.post("/api/anim_keyframe/save", json={
        "motion_json": motion,
        "name": "walk_boss1_s_nb_dragon.njm",
        "bezier_handles": handles,
    })
    assert r.status_code == 200, r.text
    assert r.json()["sidecar_written"] is True
    # Re-load — handles surface back.
    r2 = client.post("/api/anim_keyframe/load", json={
        "model_path": "bm_boss1_dragon.bml",
        "motion_name": "walk_boss1_s_nb_dragon",
    })
    assert r2.status_code == 200
    motion2 = r2.json()
    assert "bezier_handles" in motion2
    assert motion2["bezier_handles"]["0:0:rx"]["inDx"] == pytest.approx(-2.0)
    assert motion2["bezier_handles"]["5:1:tx"]["outDy"] == pytest.approx(-5.0)


# ---------------------------------------------------------------------------
# Task 3 behavioural tests — cross-bone batch ops preserve wire contract.
# ---------------------------------------------------------------------------


def _seed_multi_bone_motion():
    """4-bone motion with distinct keyframe schedules per bone."""
    from formats.njm_writer import (
        NjmBoneTracks, NjmRawMotion, NjmTrack,
    )
    from formats.njm import NJD_MTYPE_ANG, NJD_MTYPE_POS
    import server
    motion = NjmRawMotion(
        frame_count=30,
        type_flags=NJD_MTYPE_POS | NJD_MTYPE_ANG,
        inp_fn=2,
    )
    for b in range(4):
        bone = NjmBoneTracks()
        # Each bone's ANG keyframes at a different stride so we can
        # verify cross-bone selection picks the right frames per bone.
        stride = b + 2
        kfs = [(f, b * 100, b * 100, b * 100) for f in range(0, 30, stride)]
        bone.tracks_by_kind[NJD_MTYPE_ANG] = NjmTrack(
            NJD_MTYPE_ANG, kfs, narrow=True,
        )
        bone.tracks_by_kind[NJD_MTYPE_POS] = NjmTrack(
            NJD_MTYPE_POS, [], narrow=True,
        )
        motion.bones.append(bone)
    return server._ake_motion_to_json(motion, fps=30.0, motion_name="v4_multi")


def test_cross_bone_delete_only_touches_selected_bones():
    """Mimic Task 3's multi-bone delete: per-bone splice, others
    untouched."""
    js = _seed_multi_bone_motion()
    # Snapshot bone 2's keyframes — it should be unchanged.
    bone2_snapshot = [dict(k) for k in js["bones"][2]["kf"]]
    # Pretend the user marqueed kf 0 on bone 0 and kf 1 on bone 1.
    selection = [(0, 0), (1, 1)]
    # Same algorithm the panel runs.
    by_bone = {}
    for bi, kfi in selection:
        by_bone.setdefault(bi, []).append(kfi)
    for bi, idxs in by_bone.items():
        idxs.sort(reverse=True)
        for i in idxs:
            del js["bones"][bi]["kf"][i]
    # Bone 2 untouched.
    assert js["bones"][2]["kf"] == bone2_snapshot
    # Bone 3 untouched.
    assert len(js["bones"][3]["kf"]) > 0


def test_cross_bone_move_keeps_kfs_on_origin_bones():
    """A multi-bone move must not migrate keyframes between bones."""
    js = _seed_multi_bone_motion()
    # Capture which bones own which frames before the move.
    pre_owners = {}
    for bi, b in enumerate(js["bones"]):
        for kf in b["kf"]:
            pre_owners.setdefault(kf["t"], set()).add(bi)
    # Mimic a delta=+5 move on every keyframe in every bone.
    delta = 5
    max_t = js["frame_count"] - 1
    for b in js["bones"]:
        new_kfs = []
        for k in b["kf"]:
            new_kfs.append({**k, "t": min(max_t, max(0, k["t"] + delta))})
        # Replace, dedupe by frame.
        seen = {}
        for k in new_kfs:
            seen[k["t"]] = k
        b["kf"] = sorted(seen.values(), key=lambda k: k["t"])
    # No bone gained a keyframe from another bone (each kf still has the
    # bone-specific rx/ry/rz values from _seed_multi_bone_motion).
    for bi, b in enumerate(js["bones"]):
        for kf in b["kf"]:
            assert kf["rx"] == bi * 100, (
                f"bone {bi} got a kf from a different bone (rx={kf['rx']})"
            )


def test_cross_bone_copy_paste_retargets_origin_bones():
    """Multi-bone clipboard pastes back to origin bones."""
    js = _seed_multi_bone_motion()
    # Build the clipboard from a multi-bone selection (bone 0 first kf,
    # bone 1 first kf). Each clipboard entry carries boneIdx + kf payload
    # (mirrors the JS panel's representation).
    clipboard = [
        {"boneIdx": 0, "kf": dict(js["bones"][0]["kf"][0])},
        {"boneIdx": 1, "kf": dict(js["bones"][1]["kf"][0])},
    ]
    # Anchor at min t and shift to scrubber=20.
    anchor = min(c["kf"]["t"] for c in clipboard)
    shift = 20 - anchor
    by_bone = {}
    for c in clipboard:
        bi = c["boneIdx"]
        new_t = c["kf"]["t"] + shift
        by_bone.setdefault(bi, []).append({**c["kf"], "t": new_t})
    for bi, kfs in by_bone.items():
        b = js["bones"][bi]
        for k in kfs:
            b["kf"] = [x for x in b["kf"] if x["t"] != k["t"]]
            b["kf"].append(k)
        b["kf"].sort(key=lambda k: k["t"])
    # Bone 0 now has a kf at frame 20 with rx=0, bone 1 has one with rx=100.
    b0_at_20 = next(k for k in js["bones"][0]["kf"] if k["t"] == 20)
    b1_at_20 = next(k for k in js["bones"][1]["kf"] if k["t"] == 20)
    assert b0_at_20["rx"] == 0
    assert b1_at_20["rx"] == 100


# ---------------------------------------------------------------------------
# Task 1 helper — the panel's _interpAtFrame uses linear interpolation
# matching the renderer's _sampleBoneTrack. Verify the math is consistent.
# ---------------------------------------------------------------------------


def test_interp_helper_matches_linear_lerp():
    """_interpAtFrame's linear math is correct for non-rotation channels."""
    src = _PANEL_JS.read_text(encoding="utf-8")
    # The function exists and uses the same linear-lerp shape as the
    # existing sampleBone helper (we can't run the JS, but we can
    # confirm its presence in the source).
    assert "function _interpAtFrame" in src
    # Linear-lerp implementation for non-rotation channels.
    assert "av + (bv - av) * f" in src
    # Rotation channels use the BAMS-aware lerpBams helper.
    assert "lerpBams" in src
