"""End-to-end smoke for /api/anim_keyframe/{load,save,insert,delete}.

Spins up an in-process FastAPI ``TestClient`` and walks the edit
workflow on a real shipped motion (skipped without PSOBB.IO/data):

    1. POST /api/anim_keyframe/load   → fetch motion JSON
    2. Inspect the wire shape
    3. Mutate one bone's keyframe via /insert
    4. Save via /save → cache/njm_export/<name>.njm appears on disk
    5. Re-load via /load → mutation persisted (modulo round-trip)
    6. Delete via /delete → kf is gone from JSON

Does NOT touch DATA_DIR or LIVE_DATA_DIR.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PSOBB_DATA = Path(os.path.expanduser("~/PSOBB.IO/data"))
HAS_PSOBB = PSOBB_DATA.is_dir()


@pytest.fixture(scope="module")
def client():
    import server
    return TestClient(server.app)


def _pick_target_bml() -> tuple[str, str] | None:
    """Find a shipped BML that ships at least one .njm we can load.

    Returns (bml_name, motion_name) or None when no candidate exists.
    Prefers dragon's walk; falls back to the first .njm in NpcApcMot.
    """
    if not HAS_PSOBB:
        return None
    from formats.bml import extract_bml
    candidates = [
        ("bm_boss1_dragon.bml", "walk_boss1_s_nb_dragon"),
        ("bm_boss8_dragon.bml", "walk_boss1_s_nb_dragon"),
    ]
    for bml_name, motion in candidates:
        if (PSOBB_DATA / bml_name).exists():
            return bml_name, motion
    # Fallback: NpcApcMot.bml — pick first .njm.
    npc = PSOBB_DATA / "NpcApcMot.bml"
    if not npc.exists():
        return None
    try:
        entries = extract_bml(npc.read_bytes())
    except Exception:
        return None
    for name in sorted(entries.keys()):
        if name.endswith(".njm"):
            stem = name[:-4]
            return "NpcApcMot.bml", stem
    return None


# ---------------------------------------------------------------------------
# Smoke: routes are registered and reject malformed bodies
# ---------------------------------------------------------------------------


def test_routes_are_registered(client):
    """All four endpoints answer (with 4xx for bogus bodies)."""
    r = client.post("/api/anim_keyframe/load", json={"model_path": "", "motion_name": ""})
    # Empty paths should return 400/404, not 500.
    assert r.status_code in (400, 404, 422)
    r = client.post("/api/anim_keyframe/save", json={"motion_json": {}, "name": "bad"})
    # bad name (missing .njm extension) → 400.
    assert r.status_code in (400, 422)
    r = client.post("/api/anim_keyframe/insert", json={"motion_json": {}, "bone_idx": 0, "frame_idx": 0})
    assert r.status_code in (400, 422)
    r = client.post("/api/anim_keyframe/delete", json={"motion_json": {}, "bone_idx": 0, "frame_idx": 0})
    assert r.status_code in (400, 422)


def test_save_rejects_traversal(client):
    """Save endpoint enforces bare-name rule (no path components)."""
    r = client.post("/api/anim_keyframe/save", json={
        "motion_json": {"frame_count": 1, "type_flags": 0x3, "inp_fn": 2, "bones": []},
        "name": "../escape.njm",
    })
    assert r.status_code == 400


def test_save_rejects_non_njm_extension(client):
    """Save endpoint requires .njm suffix."""
    r = client.post("/api/anim_keyframe/save", json={
        "motion_json": {"frame_count": 1, "type_flags": 0x3, "inp_fn": 2, "bones": []},
        "name": "noext.bin",
    })
    assert r.status_code == 400


def test_save_synthetic_motion_via_json_round_trip(client):
    """Save a tiny synthetic motion JSON; verify the .njm appears on disk."""
    # Build a minimum valid POS+ANG JSON envelope.
    motion_json = {
        "frame_count": 5,
        "type_flags": 0x3,                # POS|ANG
        "interpolation": 0,
        "inp_fn": 2,                       # element_count=2
        "fps": 30.0,
        "bones": [
            {
                "idx": 0,
                "present": 0x3,           # POS+ANG
                "narrow_ang": True,
                "kf": [
                    {"t": 0,
                     "tx": 0.0, "ty": 0.0, "tz": 0.0,
                     "rx": 0,   "ry": 0,   "rz": 0,
                     "sx": 1.0, "sy": 1.0, "sz": 1.0},
                    {"t": 4,
                     "tx": 1.0, "ty": 2.0, "tz": 3.0,
                     "rx": 100, "ry": 200, "rz": 300,
                     "sx": 1.0, "sy": 1.0, "sz": 1.0},
                ],
            },
        ],
    }
    r = client.post("/api/anim_keyframe/save", json={
        "motion_json": motion_json,
        "name": "test_anim_editor_synth.njm",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["bone_count"] == 1
    assert body["frame_count"] == 5
    assert body["size"] > 0
    assert len(body["md5"]) == 32
    p = Path(body["path"])
    assert p.exists(), f"{p} not found"
    assert p.read_bytes()[:4] == b"NMDM"


def test_insert_returns_updated_motion(client):
    """Insert at a fresh frame appends a kf and bumps frame_count."""
    motion_json = {
        "frame_count": 5,
        "type_flags": 0x3,
        "interpolation": 0,
        "inp_fn": 2,
        "bones": [
            {"idx": 0, "present": 0x3, "narrow_ang": True, "kf": []},
        ],
    }
    r = client.post("/api/anim_keyframe/insert", json={
        "motion_json": motion_json,
        "bone_idx": 0,
        "frame_idx": 10,
        "pos": [1.0, 2.0, 3.0],
        "ang": [123, 456, 789],
    })
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["frame_count"] == 11   # bumped to frame_idx+1
    assert len(out["bones"][0]["kf"]) == 1
    kf = out["bones"][0]["kf"][0]
    assert kf["t"] == 10
    assert kf["tx"] == pytest.approx(1.0)
    assert kf["rx"] == 123


def test_delete_removes_target_kf(client):
    """Delete at an existing frame removes that kf only."""
    motion_json = {
        "frame_count": 5,
        "type_flags": 0x3,
        "interpolation": 0,
        "inp_fn": 2,
        "bones": [
            {
                "idx": 0, "present": 0x3, "narrow_ang": True,
                "kf": [
                    {"t": 0, "tx": 0, "ty": 0, "tz": 0,
                     "rx": 0, "ry": 0, "rz": 0,
                     "sx": 1, "sy": 1, "sz": 1},
                    {"t": 2, "tx": 1, "ty": 2, "tz": 3,
                     "rx": 10, "ry": 20, "rz": 30,
                     "sx": 1, "sy": 1, "sz": 1},
                ],
            },
        ],
    }
    r = client.post("/api/anim_keyframe/delete", json={
        "motion_json": motion_json,
        "bone_idx": 0,
        "frame_idx": 2,
    })
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["ok"] is True
    assert out["removed"] == 1
    kfs = out["motion_json"]["bones"][0]["kf"]
    assert len(kfs) == 1
    assert kfs[0]["t"] == 0


def test_delete_noop_at_unknown_frame(client):
    """Delete at a frame with no kf returns removed=0."""
    motion_json = {
        "frame_count": 5,
        "type_flags": 0x3,
        "interpolation": 0,
        "inp_fn": 2,
        "bones": [{"idx": 0, "present": 0x3, "narrow_ang": True, "kf": []}],
    }
    r = client.post("/api/anim_keyframe/delete", json={
        "motion_json": motion_json,
        "bone_idx": 0,
        "frame_idx": 99,
    })
    assert r.status_code == 200
    assert r.json()["removed"] == 0


def test_insert_rejects_out_of_range_bone(client):
    """bone_idx >= len(bones) → 400."""
    motion_json = {"frame_count": 5, "type_flags": 0x3, "inp_fn": 2, "bones": []}
    r = client.post("/api/anim_keyframe/insert", json={
        "motion_json": motion_json,
        "bone_idx": 5,
        "frame_idx": 0,
    })
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Live workflow against shipped data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO data not present")
def test_live_load_save_roundtrip(client):
    """Load a real motion → save (no edits) → reload → identical bones."""
    pick = _pick_target_bml()
    if pick is None:
        pytest.skip("no shipped NJM available")
    bml_name, motion_name = pick
    # /load
    r = client.post("/api/anim_keyframe/load", json={
        "model_path": bml_name,
        "motion_name": motion_name,
    })
    assert r.status_code == 200, r.text
    motion_json = r.json()
    bone_count = motion_json["bone_count"]
    frame_count = motion_json["frame_count"]
    assert bone_count > 0
    assert frame_count > 0
    # /save (with a custom name to avoid clobbering anything important)
    r = client.post("/api/anim_keyframe/save", json={
        "motion_json": motion_json,
        "name": "test_e2e_roundtrip.njm",
    })
    assert r.status_code == 200, r.text
    sav = r.json()
    assert sav["ok"] is True
    assert sav["bone_count"] == bone_count
    assert sav["frame_count"] == frame_count
    p = Path(sav["path"])
    assert p.exists()
    # Re-parse the saved file → bone_count + frame_count match.
    from formats.njm import parse_njm
    parsed = parse_njm(p.read_bytes())
    assert len(parsed) == 1
    assert parsed[0].bone_count == bone_count
    assert parsed[0].frame_count == frame_count


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO data not present")
def test_live_edit_one_kf_persists(client):
    """Load → mutate one keyframe via insert → save → reload → mutation present."""
    pick = _pick_target_bml()
    if pick is None:
        pytest.skip("no shipped NJM available")
    bml_name, motion_name = pick
    r = client.post("/api/anim_keyframe/load", json={
        "model_path": bml_name,
        "motion_name": motion_name,
    })
    assert r.status_code == 200
    motion_json = r.json()
    # Find the first bone with at least one keyframe and ANG channel.
    target_bone = -1
    target_t = -1
    for bi, bone in enumerate(motion_json["bones"]):
        if not (bone["present"] & 0x2):  # ANG bit
            continue
        if bone["kf"]:
            target_bone = bi
            target_t = bone["kf"][0]["t"]
            break
    if target_bone < 0:
        pytest.skip("no ANG keyframes in chosen motion")
    # Insert (upsert) at the existing frame with new ANG value.
    r = client.post("/api/anim_keyframe/insert", json={
        "motion_json": motion_json,
        "bone_idx": target_bone,
        "frame_idx": target_t,
        "ang": [4321, 5432, 6543],
    })
    assert r.status_code == 200, r.text
    motion_json = r.json()
    # Save.
    r = client.post("/api/anim_keyframe/save", json={
        "motion_json": motion_json,
        "name": "test_e2e_edited.njm",
    })
    assert r.status_code == 200, r.text
    p = Path(r.json()["path"])
    assert p.exists()
    # Verify the mutation by re-parsing the saved file.
    from formats.njm import parse_njm
    parsed = parse_njm(p.read_bytes())[0]
    kf = next(k for k in parsed.tracks[target_bone] if k.time == target_t)
    assert kf.rx_bams == 4321
    assert kf.ry_bams == 5432
    assert kf.rz_bams == 6543


@pytest.mark.skipif(not HAS_PSOBB, reason="PSOBB.IO data not present")
def test_live_load_rejects_unknown_motion(client):
    """Unknown motion names → 404 (not 500)."""
    pick = _pick_target_bml()
    if pick is None:
        pytest.skip("no shipped NJM available")
    bml_name, _ = pick
    r = client.post("/api/anim_keyframe/load", json={
        "model_path": bml_name,
        "motion_name": "definitely_not_a_real_motion_xyz",
    })
    assert r.status_code == 404
