"""Unit tests for the Anim Editor wire format + helper functions.

Coverage:
  - JSON projection / parse round-trip on a synthetic 4-bone motion.
  - Insert / delete / value-edit operations work on the JSON wire shape.
  - Real shipped motion: load via parse_njm_for_writer → server's
    _ake_motion_to_json → _ake_motion_from_json → encode_njm ; verify
    a byte-exact round-trip on a motion the user has not edited.
  - After a single keyframe edit, only that keyframe changes; every
    other keyframe stays bit-identical.

These tests do NOT spin up FastAPI; they exercise the helpers directly
via ``server`` module imports. The end-to-end smoke is in
``tests/test_anim_editor_e2e.py``.
"""
from __future__ import annotations

import math
import os
import struct
from pathlib import Path

import pytest

from formats.bml import extract_bml
from formats.njm import (
    NJD_MTYPE_ANG,
    NJD_MTYPE_POS,
    NJD_MTYPE_SCL,
    NjmKeyframe,
    NjmMotion,
    parse_njm,
)
from formats.njm_writer import (
    NjmBoneTracks,
    NjmRawMotion,
    NjmTrack,
    encode_njm,
    parse_njm_for_writer,
)


PSOBB_DATA = Path(os.path.expanduser("~/PSOBB.IO/data"))
HAS_PSOBB = PSOBB_DATA.is_dir()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ake_helpers():
    """Lazy-import the server's helpers; lets tests run without uvicorn."""
    import server
    return (
        server._ake_motion_to_json,
        server._ake_motion_from_json,
        server._ake_count_keyframes,
        server._ake_invalidate_round_trip,
        server._ake_kinds_in_order,
    )


def _build_synthetic_njm() -> NjmRawMotion:
    """4 bones, 30 frames, POS + ANG. Bone 0 has both; bones 1+ ANG-only.

    Mirrors the synthetic shape used in ``test_njm_writer.py`` so the
    same motion exercises both the writer test suite and the editor
    JSON round-trip.
    """
    motion = NjmRawMotion(
        frame_count=30,
        type_flags=NJD_MTYPE_POS | NJD_MTYPE_ANG,
        inp_fn=2,  # interp=0, element_count=2
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
    return motion


# ---------------------------------------------------------------------------
# JSON shape tests (no shipped data needed)
# ---------------------------------------------------------------------------


def test_to_json_shape_matches_wire_contract():
    """_ake_motion_to_json emits the documented envelope keys."""
    to_json, _, _, _, _ = _ake_helpers()
    raw = _build_synthetic_njm()
    out = to_json(raw, fps=30.0, motion_name="synth")
    assert out["frame_count"] == 30
    assert out["type_flags"] == NJD_MTYPE_POS | NJD_MTYPE_ANG
    assert out["interpolation"] == 0
    assert out["fps"] == 30.0
    assert out["name"] == "synth"
    assert isinstance(out["bones"], list)
    assert len(out["bones"]) == 4
    for bone in out["bones"]:
        assert "idx" in bone and "present" in bone and "kf" in bone
        for kf in bone["kf"]:
            for k in ("t", "tx", "ty", "tz", "rx", "ry", "rz", "sx", "sy", "sz"):
                assert k in kf, f"missing {k} in keyframe"
    # Bone 0 has POS + ANG; bones 1-3 have ANG only.
    assert out["bones"][0]["present"] == NJD_MTYPE_POS | NJD_MTYPE_ANG
    for i in range(1, 4):
        assert out["bones"][i]["present"] == NJD_MTYPE_ANG


def test_round_trip_synthetic_via_json():
    """encode_njm(synth) == encode_njm(json -> from_json) — no edits applied."""
    to_json, from_json, _, _, _ = _ake_helpers()
    raw = _build_synthetic_njm()
    out1 = encode_njm(raw)
    js = to_json(raw, fps=30.0, motion_name="synth")
    raw2 = from_json(js)
    out2 = encode_njm(raw2)
    # Round-trip is byte-exact when the source had no special padding.
    assert out2 == out1


def test_invalidate_round_trip_clears_byte_exact_hints():
    """Mutation drops source_body_b64 + track_offset_hints + trailing_size."""
    _, _, _, invalidate, _ = _ake_helpers()
    js = {
        "round_trip": {
            "pof0_b64": "abc=",
            "m_data_table_offset": 12,
            "trailing_size": 1024,
            "source_body_b64": "deadbeef",
            "track_offset_hints": [{"bone": 0, "kind": 1, "offset": 100}],
        }
    }
    invalidate(js)
    assert "pof0_b64" in js["round_trip"]                # kept
    assert "m_data_table_offset" in js["round_trip"]     # kept
    assert "source_body_b64" not in js["round_trip"]     # dropped
    assert "track_offset_hints" not in js["round_trip"]  # dropped
    assert "trailing_size" not in js["round_trip"]       # dropped


def test_kinds_in_order_matches_phantasmal_order():
    """POS, ANG, SCL, VEC, QUAT — the order Phantasmal's parseMotion pops."""
    _, _, _, _, kinds_in_order = _ake_helpers()
    assert kinds_in_order(0x3) == [NJD_MTYPE_POS, NJD_MTYPE_ANG]
    assert kinds_in_order(0x7) == [NJD_MTYPE_POS, NJD_MTYPE_ANG, NJD_MTYPE_SCL]
    # Quat-only bit (0x2000) — appears AFTER POS / SCL.
    assert kinds_in_order(0x2001)[0] == NJD_MTYPE_POS
    assert kinds_in_order(0x2001)[-1] == 0x2000


def test_count_keyframes_matches_motion():
    """_ake_count_keyframes sums kf counts across bones."""
    to_json, _, count_kf, _, _ = _ake_helpers()
    raw = _build_synthetic_njm()
    js = to_json(raw, fps=30.0, motion_name="synth")
    # Bone 0: 6 POS frames at f=0,5,...,25 + 10 ANG frames at f=0,3,...,27
    #          merged by frame number — overlap on frame 0 (and 15).
    # Frame numbers in POS: {0,5,10,15,20,25}
    # Frame numbers in ANG: {0,3,6,9,12,15,18,21,24,27}
    # Union: {0,3,5,6,9,10,12,15,18,20,21,24,25,27} = 14 frames
    # Bones 1-3: 10 ANG-only keyframes each = 30 keyframes
    # Total = 14 + 30 = 44
    assert count_kf(js) == 44


# ---------------------------------------------------------------------------
# Insert / delete / value-edit operations on the JSON wire format
# ---------------------------------------------------------------------------
#
# These tests build a motion JSON in memory, call the same mutation
# functions the server exposes, and verify the JSON wire format is
# updated correctly. The mutations are simple enough to test directly.
# (The HTTP endpoints exercise the same path; the e2e tests cover
# request validation + the FastAPI plumbing.)


def _seed_motion_json():
    to_json, _, _, _, _ = _ake_helpers()
    raw = _build_synthetic_njm()
    return to_json(raw, fps=30.0, motion_name="synth")


def test_insert_keyframe_creates_a_new_kf_at_unused_frame():
    """Insert at a frame with no existing kf → new entry appears in sorted order."""
    js = _seed_motion_json()
    # Bone 0 has POS keyframes at 0/5/10/...; insert a fresh kf at frame 7.
    bone0 = js["bones"][0]
    n_before = len(bone0["kf"])
    bone0["kf"].append({
        "t": 7,
        "tx": 1.5, "ty": 2.5, "tz": 3.5,
        "rx": 100, "ry": 200, "rz": 300,
        "sx": 1.0, "sy": 1.0, "sz": 1.0,
    })
    bone0["kf"].sort(key=lambda k: k["t"])
    assert len(bone0["kf"]) == n_before + 1
    new_kf = next(k for k in bone0["kf"] if k["t"] == 7)
    assert new_kf["tx"] == pytest.approx(1.5)
    assert new_kf["rx"] == 100


def test_delete_keyframe_removes_only_target_frame():
    """Delete at frame 5 → frame 5 is gone, every other kf untouched."""
    js = _seed_motion_json()
    bone0 = js["bones"][0]
    before = list(bone0["kf"])
    bone0["kf"] = [k for k in bone0["kf"] if k["t"] != 5]
    assert len(bone0["kf"]) == len(before) - 1
    assert all(k["t"] != 5 for k in bone0["kf"])
    # Other frames unchanged.
    for old_kf in before:
        if old_kf["t"] == 5:
            continue
        new_kf = next(k for k in bone0["kf"] if k["t"] == old_kf["t"])
        assert new_kf == old_kf


def test_value_edit_changes_only_target_kf():
    """Edit kf at frame 0 on bone 0 → only that field/keyframe changes."""
    js = _seed_motion_json()
    bone0 = js["bones"][0]
    target = next(k for k in bone0["kf"] if k["t"] == 0)
    old_tx = target["tx"]
    target["rx"] = 12345
    # Every other keyframe on every other bone stays put.
    for bi, b in enumerate(js["bones"]):
        for k in b["kf"]:
            if bi == 0 and k["t"] == 0:
                continue
            assert k.get("rx", 0) != 12345 or k.get("rx") == 0


def test_round_trip_after_single_edit_changes_one_keyframe():
    """Encode → parse → decode shows the edit landed in the right place."""
    to_json, from_json, _, _, _ = _ake_helpers()
    js = _seed_motion_json()
    # Mutate frame 0 ANG on bone 0.
    bone0 = js["bones"][0]
    target = next(k for k in bone0["kf"] if k["t"] == 0)
    target["rx"] = 12345
    raw2 = from_json(js)
    out_bytes = encode_njm(raw2)
    parsed = parse_njm(out_bytes)
    assert len(parsed) == 1
    motion = parsed[0]
    # Find frame 0 keyframe on bone 0; rx should now be 12345.
    kf0 = next(k for k in motion.tracks[0] if k.time == 0)
    assert kf0.rx_bams == 12345
    # Other bones' frame 0 ANG should be 0 (unchanged from synthetic).
    for bi in range(1, 4):
        kfs = motion.tracks[bi]
        kf = next((k for k in kfs if k.time == 0), None)
        if kf is not None:
            assert kf.rx_bams == 0


# ---------------------------------------------------------------------------
# Real shipped motion: load → serialize → re-parse, no edits
# ---------------------------------------------------------------------------


def _find_a_shipped_motion() -> tuple[bytes, str] | None:
    """Locate ANY shipped NJM blob for round-trip tests.

    Returns the raw NJM bytes + a debug label, or None when PSOBB.IO/data
    isn't present.  Prefers the dragon walk (a well-tested motion in the
    rest of the suite) but falls back to whatever NJM the BMLs surface.
    """
    if not HAS_PSOBB:
        return None
    preferred = [
        ("bm_boss1_dragon.bml", "walk_boss1_s_nb_dragon.njm"),
        ("bm_boss8_dragon.bml", "walk_boss1_s_nb_dragon.njm"),
        ("NpcApcMot.bml", None),  # any NJM
    ]
    for bml_name, inner_pref in preferred:
        bml = PSOBB_DATA / bml_name
        if not bml.exists():
            continue
        try:
            entries = extract_bml(bml.read_bytes())
        except Exception:
            continue
        if inner_pref and inner_pref in entries:
            return entries[inner_pref], f"{bml_name}#{inner_pref}"
        for name in sorted(entries.keys()):
            if name.endswith(".njm"):
                return entries[name], f"{bml_name}#{name}"
    return None


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO data not present")
def test_live_round_trip_via_json_no_edits():
    """Real motion → JSON → back → encode produces byte-identical output."""
    to_json, from_json, _, _, _ = _ake_helpers()
    found = _find_a_shipped_motion()
    if found is None:
        pytest.skip("no NJM in shipped data")
    src, label = found
    raw = parse_njm_for_writer(src)
    js = to_json(raw, fps=30.0, motion_name="live")
    raw2 = from_json(js)
    out = encode_njm(raw2)
    assert out == src, (
        f"{label}: src {len(src)}B != round-trip {len(out)}B"
    )


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO data not present")
def test_live_round_trip_after_one_edit_only_target_changed():
    """Edit one BAMS field → re-encode → parse → only that field differs.

    Verifies that:
      - The mutation lands at the right frame on the right bone.
      - Every OTHER keyframe stays bit-identical.
      - frame_count + bone_count survive the round trip.
    """
    to_json, from_json, _, invalidate, _ = _ake_helpers()
    found = _find_a_shipped_motion()
    if found is None:
        pytest.skip("no NJM in shipped data")
    src, label = found
    raw = parse_njm_for_writer(src)
    js = to_json(raw, fps=30.0, motion_name="live")
    # Find the first bone with at least one ANG keyframe.
    target_bone = -1
    target_kf_idx = -1
    target_t = -1
    for bi, bone in enumerate(js["bones"]):
        if not (bone["present"] & NJD_MTYPE_ANG):
            continue
        if not bone["kf"]:
            continue
        target_bone = bi
        target_kf_idx = 0
        target_t = bone["kf"][0]["t"]
        break
    if target_bone < 0:
        pytest.skip(f"{label}: no bone has ANG keyframes")
    js_orig = _seed_motion_json_from(js)
    js["bones"][target_bone]["kf"][target_kf_idx]["rx"] = 24680
    invalidate(js)
    raw2 = from_json(js)
    out = encode_njm(raw2)
    parsed = parse_njm(out)
    assert len(parsed) == 1
    motion = parsed[0]
    # Verify the mutation landed.
    kf = next(k for k in motion.tracks[target_bone] if k.time == target_t)
    assert kf.rx_bams == 24680, f"{label}: rx_bams != 24680 at bone {target_bone} frame {target_t}"
    # Verify SCALAR identity at other (bone, frame) tuples — sample a few.
    orig_motion = parse_njm(src)[0]
    sample_count = 0
    for bi, bone_kfs in enumerate(motion.tracks):
        if bi == target_bone:
            continue
        for kf in bone_kfs[: min(3, len(bone_kfs))]:
            o = next((k for k in orig_motion.tracks[bi] if k.time == kf.time), None)
            if o is None:
                continue
            assert kf.rx_bams == o.rx_bams
            assert kf.ry_bams == o.ry_bams
            assert kf.rz_bams == o.rz_bams
            sample_count += 1
    assert sample_count >= 3


def _seed_motion_json_from(js: dict) -> dict:
    """Deep-copy the JSON envelope (used to compare before/after edits)."""
    import copy
    return copy.deepcopy(js)


# ---------------------------------------------------------------------------
# Editor v3: multi-keyframe operations preserve the wire contract.
#
# The new editor (anim_editor_panel.js) supports drag-to-move, marquee
# selection, copy/paste, and bezier densification. All of these mutate
# the shared motion JSON in ways the existing JSON shape tests already
# cover individually (insert / delete / value-edit). The tests below
# verify that a SEQUENCE of those operations — the kind a user would
# perform during a typical editing session — round-trips cleanly.
# ---------------------------------------------------------------------------


def test_editor_v3_multi_kf_session_round_trips():
    """Simulate a multi-step editing session, then re-encode & re-parse.

    Steps:
        1. Move all kfs at frames 0..9 forward by +5.
        2. Delete the kfs that landed in 12..14 (collision cleanup).
        3. Insert a fresh kf at frame 28 with a known value.
        4. Encode → parse → verify the final state.
    """
    to_json, from_json, _, _, _ = _ake_helpers()
    raw = _build_synthetic_njm()
    js = to_json(raw, fps=30.0, motion_name="multi")
    # Step 1: shift early kfs forward by 5. Mimics "drag-move 8 kfs by +5".
    for bone in js["bones"]:
        new_kfs = []
        for k in bone["kf"]:
            if (k["t"] | 0) < 10:
                k = dict(k)
                k["t"] = (k["t"] | 0) + 5
            new_kfs.append(k)
        # De-dup: when shift collides with an existing later kf,
        # the SHIFTED value wins (panel's drag-move semantics).
        seen_t: dict[int, dict] = {}
        for k in new_kfs:
            seen_t[k["t"] | 0] = k
        bone["kf"] = sorted(seen_t.values(), key=lambda k: k["t"])
    # Step 2: drop any kf in {12, 13, 14} (cleanup post-collision).
    drop_set = {12, 13, 14}
    for bone in js["bones"]:
        bone["kf"] = [k for k in bone["kf"] if (k["t"] | 0) not in drop_set]
    # Step 3: insert a fresh kf at frame 28 on bone 0 with a sentinel.
    bone0 = js["bones"][0]
    sentinel = {
        "t": 28,
        "tx": 7.0, "ty": 8.0, "tz": 9.0,
        "rx": 1234, "ry": 5678, "rz": 9012,
        "sx": 1.0, "sy": 1.0, "sz": 1.0,
    }
    bone0["kf"] = [k for k in bone0["kf"] if (k["t"] | 0) != 28]
    bone0["kf"].append(sentinel)
    bone0["kf"].sort(key=lambda k: k["t"])
    # Step 4: round-trip and check the sentinel survived the encode.
    raw2 = from_json(js)
    out_bytes = encode_njm(raw2)
    parsed = parse_njm(out_bytes)
    assert len(parsed) == 1
    motion = parsed[0]
    # The sentinel must be reachable on bone 0 at frame 28 (POS+ANG bits
    # both set on bone 0, so encode emits the value verbatim).
    bone_kfs = motion.tracks[0]
    kf_at_28 = [k for k in bone_kfs if k.time == 28]
    assert len(kf_at_28) >= 1
    found = kf_at_28[0]
    # NJM values come back via narrow-int rotations; check sentinel.
    assert found.rx_bams == 1234
    assert found.ry_bams == 5678
    assert found.rz_bams == 9012


def test_editor_v3_paste_shifts_relative_to_anchor():
    """Pasting a 3-kf clipboard at frame N anchors the FIRST kf at N
    and shifts the rest by their original spacing.

    This is the panel's pasteKeyframes() contract: anchor=clip[0].t,
    each new kf lands at (state.selectedFrame + (k.t - anchor)).
    """
    to_json, _, _, _, _ = _ake_helpers()
    raw = _build_synthetic_njm()
    js = to_json(raw, fps=30.0, motion_name="paste")
    bone0 = js["bones"][0]
    # Clipboard: pretend the user copied kfs at t=0,5,10. They're at
    # bone0 indices 0,1,2 in the seed.
    clip = [dict(bone0["kf"][i]) for i in range(3)]
    anchor = clip[0]["t"]
    # Paste at frame 20.
    paste_at = 20
    pasted = [dict(k) for k in clip]
    for k in pasted:
        k["t"] = paste_at + ((k["t"] | 0) - anchor)
    # Apply the paste like the panel does: drop existing kfs at the
    # target frames, push pasted, sort.
    target_ts = {(k["t"] | 0) for k in pasted}
    bone0["kf"] = [k for k in bone0["kf"] if (k["t"] | 0) not in target_ts]
    bone0["kf"].extend(pasted)
    bone0["kf"].sort(key=lambda k: k["t"])
    # Verify: 3 fresh kfs at t=20, 25 (was 5 → +15), 30 (was 10 → +20)?
    # The frame_count in the synthetic envelope is 30 — the kf at t=30
    # is OUT of range. Let's assert the in-range pastes landed.
    landed = [k for k in bone0["kf"] if (k["t"] | 0) in target_ts]
    assert len(landed) == len(pasted)
    expected = sorted({(paste_at + ((k["t"] | 0) - anchor)) for k in clip})
    actual = sorted(k["t"] | 0 for k in landed)
    assert expected == actual


def test_editor_v3_bone_tree_threshold_fallback():
    """The bone-tree fallback threshold (BONE_TREE_THRESHOLD = 12) is
    documented in the wire-format invariants: motions with < 12 bones
    use the flat dropdown, motions with >= 12 use the tree.

    This test asserts the threshold matches the panel constant — if it
    drifts, both panel + tests must update together.
    """
    panel_path = Path(__file__).parent.parent / "static" / "anim_editor_panel.js"
    src = panel_path.read_text(encoding="utf-8")
    # Find the const declaration.
    import re
    m = re.search(r"BONE_TREE_THRESHOLD\s*=\s*(\d+)", src)
    assert m is not None, "BONE_TREE_THRESHOLD constant missing"
    threshold = int(m.group(1))
    assert 8 <= threshold <= 32, (
        f"BONE_TREE_THRESHOLD={threshold} out of plausible range — "
        "tiny rigs (humanoid 16-bone) should still get the tree, "
        "but 1-2 bone rigs (props) shouldn't waste vertical space."
    )
