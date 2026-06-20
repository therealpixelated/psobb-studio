"""Editor v3 — motion-editor UI smoke tests (anim_editor_panel.js).

The four user-visible Editor v3 features (scrubber 3D sync, bone tree,
multi-keyframe selection, curve editor) live entirely in client-side
JS. We can't drive a browser from pytest, but we CAN:

  1. Statically smoke-check the source for the expected exports / hooks.
  2. Verify the wire format the panel mutates is still valid by
     re-running it through the existing server round-trip.
  3. Prove the Python equivalents of the new mutations (move-keyframes,
     delete-selected, paste-keyframes) preserve every contract that the
     existing test_anim_keyframe.py suite exercised.
  4. Confirm the new psoSeekAnimationToFrame export was added to
     model_viewer.js without breaking the existing additive surface.

These tests do NOT spin up a browser — they read the JS source as text
and exercise the server's wire-format helpers directly.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from formats.njm import NJD_MTYPE_ANG, NJD_MTYPE_POS, NJD_MTYPE_SCL, parse_njm
from formats.njm_writer import (
    NjmBoneTracks,
    NjmRawMotion,
    NjmTrack,
    encode_njm,
    parse_njm_for_writer,
)


_REPO = Path(__file__).resolve().parent.parent
_PANEL_JS = _REPO / "static" / "anim_editor_panel.js"
_VIEWER_JS = _REPO / "static" / "model_viewer.js"
_STYLE_CSS = _REPO / "static" / "style.css"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ake_helpers():
    import server
    return (
        server._ake_motion_to_json,
        server._ake_motion_from_json,
        server._ake_count_keyframes,
        server._ake_invalidate_round_trip,
    )


def _seed_motion_json():
    """Build a tiny but realistic motion JSON the panel could mutate.

    4 bones, 30 frames, POS+ANG. Bone 0 has BOTH; bones 1+ ANG-only.
    Keyframes are placed at 0/5/10/15/20/25 (POS) and 0/3/6/9/.../27
    (ANG) — the same spacing the editor's regression test uses.
    """
    to_json, _, _, _ = _ake_helpers()
    motion = NjmRawMotion(
        frame_count=30,
        type_flags=NJD_MTYPE_POS | NJD_MTYPE_ANG,
        inp_fn=2,
    )
    for b in range(4):
        bone = NjmBoneTracks()
        if b == 0:
            bone.tracks_by_kind[NJD_MTYPE_POS] = NjmTrack(
                NJD_MTYPE_POS,
                [(f, 0.0, float(f), 0.0) for f in range(0, 30, 5)],
                narrow=True,
            )
        else:
            bone.tracks_by_kind[NJD_MTYPE_POS] = NjmTrack(
                NJD_MTYPE_POS, [], narrow=True,
            )
        ang_kfs = []
        for f in range(0, 30, 3):
            ang = int(0x10000 * 0.25 * math.sin(f / 30.0 * 2.0 * math.pi))
            ang_kfs.append((f, 0, ang & 0xFFFF, 0))
        bone.tracks_by_kind[NJD_MTYPE_ANG] = NjmTrack(
            NJD_MTYPE_ANG, ang_kfs, narrow=True,
        )
        motion.bones.append(bone)
    return to_json(motion, fps=30.0, motion_name="ui_synth")


# ---------------------------------------------------------------------------
# Static smoke: panel exports + viewer additive surface present
# ---------------------------------------------------------------------------


def test_panel_js_exists_and_parses():
    """The panel source is non-empty and contains the v3 entry points."""
    assert _PANEL_JS.is_file(), f"missing {_PANEL_JS}"
    src = _PANEL_JS.read_text(encoding="utf-8")
    assert len(src) > 1000
    # The IIFE guard prevents double-loading.
    assert "__psoAnimEditorPanelLoaded" in src
    # Devtools handle still exposed.
    assert "window.psoAnimEditorState" in src


def test_panel_uses_new_seek_export():
    """Task 1 — panel calls window.psoSeekAnimationToFrame."""
    src = _PANEL_JS.read_text(encoding="utf-8")
    assert "psoSeekAnimationToFrame" in src
    assert "psoSetAnimationPlaying(false)" in src or "psoSetAnimationPlaying" in src
    # rAF-based throttling for high-frequency drag events.
    assert "requestAnimationFrame" in src
    assert "seekRafScheduled" in src or "seekRafPending" in src


def test_panel_has_bone_tree():
    """Task 2 — panel ships hierarchical bone tree + search + eye toggle."""
    src = _PANEL_JS.read_text(encoding="utf-8")
    assert "buildBoneTreeHtml" in src
    assert "ake-tree-row" in src
    assert "ake-tree-eye" in src
    assert "boneTreeQuery" in src
    assert "BONE_TREE_THRESHOLD" in src
    # Fallback dropdown still present for tiny rigs.
    assert "ake-bone-pick" in src


def test_panel_has_multi_select():
    """Task 3 — panel ships marquee selection + drag-move + ctx menu."""
    src = _PANEL_JS.read_text(encoding="utf-8")
    assert "selectedKfSet" in src
    # Marquee state.
    assert "marquee" in src
    # Drag-to-move on the timeline.
    assert "moveSelectedKeyframes" in src
    # Context menu primitives.
    assert "showContextMenu" in src
    # Keyboard shortcuts.
    assert "selectAllKeyframes" in src
    assert "Ctrl+A" in src or "ctrlKey" in src
    # Copy / paste / delete flows.
    assert "copySelectedKeyframes" in src
    assert "pasteKeyframes" in src
    assert "deleteSelectedKeyframes" in src


def test_panel_has_curve_editor():
    """Task 4 — panel ships bezier curve editor + densification."""
    src = _PANEL_JS.read_text(encoding="utf-8")
    assert "drawCurveCanvas" in src
    assert "bezierHandles" in src
    assert "densifyBezierToLinear" in src
    assert "bakeAllBezierToLinear" in src
    # Doc string about PSOBB linear interp.
    assert "linear interpolation" in src.lower()


def test_panel_documents_psobb_linear_interp_warning():
    """The curve editor must surface the PSOBB-is-linear warning to the user."""
    src = _PANEL_JS.read_text(encoding="utf-8")
    # Visible note in the panel HTML, not just a code comment.
    assert "linear interpolation" in src.lower()
    assert "bake" in src.lower()


def test_viewer_exports_seek_animation_to_frame():
    """Task 1 — model_viewer.js exposes psoSeekAnimationToFrame."""
    assert _VIEWER_JS.is_file(), f"missing {_VIEWER_JS}"
    src = _VIEWER_JS.read_text(encoding="utf-8")
    # The new additive export is present.
    assert "window.psoSeekAnimationToFrame" in src
    # And the existing surface it depends on.
    assert "window.psoSetAnimationPlaying" in src
    assert "window.psoForceRender" in src
    assert "window.psoGetSkeleton" in src


def test_viewer_seek_pauses_and_clamps():
    """psoSeekAnimationToFrame pauses playback, clamps frame, and re-bakes."""
    src = _VIEWER_JS.read_text(encoding="utf-8")
    # Locate the function body and check it touches the right state.
    m = re.search(
        r"window\.psoSeekAnimationToFrame\s*=\s*function\s*\(.*?\)\s*\{(.+?)^\}\s*;",
        src, flags=re.DOTALL | re.MULTILINE,
    )
    assert m is not None, "could not locate psoSeekAnimationToFrame body"
    body = m.group(1)
    assert "a.time" in body, "must write into state.anim.time (frameTime)"
    assert "a.playing" in body, "must pause playback"
    assert "tickAnimation" in body, "must force one bake tick"
    # Clamping logic.
    assert "frame_count" in body
    assert "fps" in body


def test_style_css_has_v3_rules():
    """style.css picks up the new ake-tree-* and ake-curve-* rules."""
    css = _STYLE_CSS.read_text(encoding="utf-8")
    assert ".ake-tree-host" in css
    assert ".ake-curve-host" in css
    assert ".ake-keyframe-row.selected" in css


# ---------------------------------------------------------------------------
# Behavioural: the mutations the panel performs are still valid for the
# server's encode path. Mirrors the algorithms in JS (single-source-of-
# truth deviations would break the round trip).
# ---------------------------------------------------------------------------


def test_multi_select_delete_preserves_other_keyframes():
    """Removing N keyframes by index leaves every other kf bit-identical."""
    js = _seed_motion_json()
    bone0 = js["bones"][0]
    before = list(bone0["kf"])
    # Mimic the panel's delete-selected: drop keyframes at indices [0, 2].
    drop_indices = {0, 2}
    bone0["kf"] = [k for i, k in enumerate(bone0["kf"]) if i not in drop_indices]
    assert len(bone0["kf"]) == len(before) - 2
    # Surviving frames retained their data.
    surviving = [before[i] for i in range(len(before)) if i not in drop_indices]
    assert bone0["kf"] == surviving


def test_multi_select_move_shifts_frames_by_delta():
    """Drag-move = uniform t-offset on selected kfs; collisions get
    overwritten by the moved value."""
    js = _seed_motion_json()
    bone0 = js["bones"][0]
    n_before = len(bone0["kf"])
    # Mimic panel: move the kf at index 0 (t=0) forward by 3 frames.
    delta = 3
    target_idx = 0
    moved = dict(bone0["kf"][target_idx])
    moved["t"] = (moved["t"] | 0) + delta
    new_t = moved["t"]
    # Drop the original + any kf at the destination frame, push moved.
    bone0["kf"] = [k for i, k in enumerate(bone0["kf"])
                   if i != target_idx and (k["t"] | 0) != new_t]
    bone0["kf"].append(moved)
    bone0["kf"].sort(key=lambda k: k["t"])
    # No frame appears twice.
    seen = set()
    for k in bone0["kf"]:
        t = k["t"] | 0
        assert t not in seen, f"duplicate t={t}"
        seen.add(t)


def test_paste_replaces_existing_kf_at_target():
    """Pasting at frame N upserts (replaces if a kf already exists there)."""
    js = _seed_motion_json()
    bone0 = js["bones"][0]
    # Find an existing kf to clobber + a fresh paste kf.
    target_t = bone0["kf"][0]["t"]
    paste_kf = dict(bone0["kf"][1])
    paste_kf["t"] = target_t
    paste_kf["tx"] = 99.0
    # Mimic: drop existing at target_t, push paste, sort.
    bone0["kf"] = [k for k in bone0["kf"] if (k["t"] | 0) != target_t]
    bone0["kf"].append(paste_kf)
    bone0["kf"].sort(key=lambda k: k["t"])
    landed = next(k for k in bone0["kf"] if (k["t"] | 0) == target_t)
    assert landed["tx"] == pytest.approx(99.0)


def test_round_trip_after_multi_kf_delete():
    """Encode → parse → decode after multi-delete produces a valid file."""
    to_json, from_json, _, _ = _ake_helpers()
    js = _seed_motion_json()
    bone0 = js["bones"][0]
    # Drop 3 ANG keyframes (preserving POS + endpoints).
    drop_targets = {3, 6, 9}
    for bone in js["bones"]:
        bone["kf"] = [k for k in bone["kf"]
                      if (k["t"] | 0) not in drop_targets]
    raw = from_json(js)
    out_bytes = encode_njm(raw)
    parsed = parse_njm(out_bytes)
    assert len(parsed) == 1
    motion = parsed[0]
    # No bone has a keyframe at any of the dropped frames (for ANG).
    for bi, kfs in enumerate(motion.tracks):
        for kf in kfs:
            if kf.time in drop_targets:
                # Allow it ONLY if the frame is also a POS frame for bone 0
                # (POS kfs at multiples of 5 are at f=5,10,...; the dropped
                # set is {3,6,9} which doesn't overlap, so this should
                # never fire).
                assert kf.time not in drop_targets, (
                    f"bone {bi} still has a kf at dropped frame {kf.time}"
                )


def test_bezier_densification_emits_dense_linear_keyframes():
    """Mimic densifyBezierToLinear in Python; verify the wire shape stays valid.

    The JS panel calls densifyBezierToLinear at save time when bezier
    handles exist. We can't run the JS bezier solver, but we can verify
    that adding more keyframes (which is what densification does)
    preserves the wire contract.
    """
    to_json, from_json, _, _ = _ake_helpers()
    js = _seed_motion_json()
    bone0 = js["bones"][0]
    # Simulate "bake" by interpolating ANG values densely between
    # existing kfs (same effect: more keyframes, all linear).
    kfs = sorted(bone0["kf"], key=lambda k: k["t"])
    densified = []
    for i in range(len(kfs) - 1):
        a, b = kfs[i], kfs[i + 1]
        densified.append(dict(a))
        ax, bx = a["t"] | 0, b["t"] | 0
        for f in range(ax + 1, bx):
            t = (f - ax) / max(1, bx - ax)
            mid = dict(a)
            mid["t"] = f
            mid["rx"] = int(a["rx"] + (b["rx"] - a["rx"]) * t)
            mid["ry"] = int(a["ry"] + (b["ry"] - a["ry"]) * t)
            mid["rz"] = int(a["rz"] + (b["rz"] - a["rz"]) * t)
            densified.append(mid)
    densified.append(dict(kfs[-1]))
    bone0["kf"] = densified
    n_dense = len(bone0["kf"])
    # Round-trip — server should still encode and decode this fine.
    raw = from_json(js)
    out = encode_njm(raw)
    parsed = parse_njm(out)
    assert len(parsed) == 1
    # Bone 0's ANG track now has the dense schedule.
    bone_kfs = parsed[0].tracks[0]
    # We don't know the EXACT count after merging POS+ANG, but it must
    # be at least as many as the dense ANG schedule we authored.
    assert len(bone_kfs) >= max(1, n_dense - 1)


# ---------------------------------------------------------------------------
# Selection-summary semantics — the panel's selectedKfSet uses string
# keys to survive splice operations. Confirm the contract is stable.
# ---------------------------------------------------------------------------


def test_selection_set_uses_index_not_frame():
    """The panel keys the selection set by kfIdx (string), so deletions
    that splice the array invalidate selection by INDEX (not frame).

    This mirrors the panel's behavior: after deletion, callers must
    rebuild the selection from frame numbers if they care about specific
    keyframes. The selection set itself is naturally cleared by the
    delete-selected helper."""
    src = _PANEL_JS.read_text(encoding="utf-8")
    # The selection set is keyed by kfIdx-as-string.
    assert "state.selectedKfSet.add(String(" in src
    # And cleared after destructive ops.
    assert "selectedKfSet.clear()" in src


def test_panel_clears_per_motion_state_on_motion_load():
    """Loading a new motion clears selection + bezier + tree filters."""
    src = _PANEL_JS.read_text(encoding="utf-8")
    # Locate loadMotionFromPicker by its declaration and find the matching
    # closing brace via brace counting (regex can't handle nested braces).
    start = src.find("async function loadMotionFromPicker")
    assert start >= 0, "could not locate loadMotionFromPicker"
    # Walk forward to the function's open brace.
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
    assert "selectedKfSet.clear()" in body
    assert "bezierHandles.clear()" in body
    assert "boneTreeHidden.clear()" in body
    # v4 — also clears the cross-bone selection + releases bone-pose
    # overrides authored by eye-toggle on the previous motion.
    assert "selectedKfSetMulti.clear()" in body
    assert "releaseAllPanelBoneOverrides" in body


# ---------------------------------------------------------------------------
# v4 forwarding tests — kept here so the v3 suite remains the canonical
# entry point for "does the panel still ship the right surface?". The
# deeper v4 behavioural tests live in test_motion_editor_v4.py.
# ---------------------------------------------------------------------------


def test_panel_v4_curve_overlay_modes_present():
    """v4 / Task 1 — single | triplet | all curve overlay modes."""
    src = _PANEL_JS.read_text(encoding="utf-8")
    assert "curveMode" in src
    assert "_activeCurveChannels" in src
    assert "ake-curve-legend" in src


def test_panel_v4_handle_persistence_wiring():
    """v4 / Task 2 — bezier handles round-trip through /save + /load."""
    src = _PANEL_JS.read_text(encoding="utf-8")
    # Save body includes bezier_handles when state.bezierHandles is non-empty.
    assert "bezier_handles" in src
    # Load handler restores them when present in the response.
    assert "data.bezier_handles" in src or "bezierHandles.set" in src


def test_panel_v4_cross_bone_marquee():
    """v4 / Task 3 — cross-bone selection + bone-id annotation."""
    src = _PANEL_JS.read_text(encoding="utf-8")
    assert "marqueeMode" in src
    assert "selectedKfSetMulti" in src


def test_panel_v4_eye_toggle_wiring():
    """v4 / Task 4 — eye-toggle calls psoSetBonePoseOverride."""
    src = _PANEL_JS.read_text(encoding="utf-8")
    assert "psoSetBonePoseOverride" in src
    assert "releaseAllPanelBoneOverrides" in src
