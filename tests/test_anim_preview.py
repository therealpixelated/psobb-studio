"""Tests for the editor preview-only animation import flow.

Covers:
  - /api/anim_preview/list filters by target_model_path basename and
    returns {} when no sidecar matches.
  - /api/anim_preview/data returns the same wire shape as
    /api/animation_data (frame_count / fps / bones[*].kf) so the
    frontend's psoLoadMotion(json) call works without translation.
  - /api/anim_preview/delete removes both the .njm and the sidecar,
    and is idempotent (re-deleting returns 200, removed=[]).
  - /api/import/animation now writes the .preview.json sidecar with
    the expected fields when given a target_model_path.
  - The lobby-girl staged demo at cache/njm_export/lobby_girl_typing.njm
    has a sidecar pointing at bm_npc_kenkyu_w.bml and parses cleanly.
  - Negative: njm_path traversal is rejected; non-existent files 404.
  - Verify the live game install at ~/PSOBB.IO/data/
    NpcApcMot.bml is byte-identical to PSOBB ship state (the wrong
    -scope finishing-line agent's output didn't deploy).

These tests use an in-process FastAPI ``TestClient`` and the real
``cache/njm_export/`` directory — they're hermetic per-test (each
test cleans up its own files) but DO touch disk to exercise the
sidecar format end-to-end.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pytest

from fastapi.testclient import TestClient


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(scope="module")
def client():
    """In-process FastAPI client. Imports server.py once per module."""
    import server
    return TestClient(server.app)


@pytest.fixture(scope="module")
def njm_export_dir():
    import server
    return server.NJM_EXPORT_DIR


@pytest.fixture(autouse=True)
def _defensive_test_file_cleanup(njm_export_dir):
    """Wipe any ``test_anim_preview_*.njm`` left behind by a prior run.

    Each test in this module already cleans up its own stage files via
    the ``_cleanup`` helper, but order-dependent flake reports indicate
    that under load (parallel test runners, slow disk) a stage file
    can occasionally survive into the next test and skew the
    ``/api/anim_preview/list`` results.

    Yields then runs again post-test for symmetry. Only touches files
    whose name starts with ``test_anim_preview_`` so we don't disturb
    the shipped ``lobby_girl_typing.*`` demo or any external tooling
    that places .njms in the export dir.
    """
    def _wipe():
        for p in njm_export_dir.glob("test_anim_preview_*"):
            try:
                p.unlink()
            except OSError:
                pass
    _wipe()
    yield
    _wipe()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_minimal_njm() -> bytes:
    """Author the smallest valid NJM the parser will accept.

    Reuses ``formats.anim_retarget`` + ``encode_njm`` with a tiny 2-bone
    synthetic skeleton + 1-track animation, so the test-side .njm is
    a real round-trippable motion. The bytes are reused across the
    suite via a module-level cache (the synthesis is the same every
    time).
    """
    global _MINIMAL_NJM_CACHE
    if _MINIMAL_NJM_CACHE is not None:
        return _MINIMAL_NJM_CACHE

    import math
    from formats.anim_retarget import retarget_animation
    from formats.import_external import (
        ImportedAnimation, ImportedBone, ImportedTrack,
    )
    from formats.njm_writer import encode_njm

    src_skel = [
        ImportedBone(name="Hips", parent_idx=-1,
                     bind_pos=(0.0, 1.0, 0.0), bind_rot_quat=(0.0, 0.0, 0.0, 1.0)),
        ImportedBone(name="Spine", parent_idx=0,
                     bind_pos=(0.0, 0.2, 0.0), bind_rot_quat=(0.0, 0.0, 0.0, 1.0)),
    ]
    tgt_skel = [
        ImportedBone(name="root", parent_idx=-1,
                     bind_pos=(0.0, 0.0, 0.0), bind_rot_quat=(0.0, 0.0, 0.0, 1.0)),
        ImportedBone(name="torso", parent_idx=0,
                     bind_pos=(0.0, 1.0, 0.0), bind_rot_quat=(0.0, 0.0, 0.0, 1.0)),
    ]
    n_frames = 4
    times = [f / 30.0 for f in range(n_frames)]
    values = []
    for f in range(n_frames):
        ang = math.pi * 0.5 * f / max(1, n_frames - 1)
        values.append((0.0, math.sin(ang * 0.5), 0.0, math.cos(ang * 0.5)))
    src_anim = ImportedAnimation(
        name="MinimalTest",
        duration_seconds=times[-1],
        fps_target=30,
        tracks=[
            ImportedTrack(
                bone_idx=1, channel="rotation",
                times=times, values=values, interp="LINEAR",
            ),
        ],
    )
    motion = retarget_animation(
        src_anim, src_skel, tgt_skel,
        bone_map={"Hips": 0, "Spine": 1},
        target_fps=30, flip_z=False,
    )
    _MINIMAL_NJM_CACHE = encode_njm(motion)
    return _MINIMAL_NJM_CACHE


_MINIMAL_NJM_CACHE: bytes | None = None


def _stage_njm(njm_export_dir: Path, name: str, sidecar: dict | None) -> Path:
    """Write a tiny valid .njm + optional .preview.json sidecar.

    Returns the path to the .njm. Caller must delete both on cleanup.
    """
    p = njm_export_dir / name
    p.write_bytes(_make_minimal_njm())
    if sidecar is not None:
        sc = njm_export_dir / (name + ".preview.json")
        sc.write_text(json.dumps(sidecar), encoding="utf-8")
    return p


def _cleanup(*paths: Path) -> None:
    for p in paths:
        try:
            if isinstance(p, Path) and p.exists():
                p.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# /api/anim_preview/list
# ---------------------------------------------------------------------------


def test_anim_preview_list_filters_by_basename(client, njm_export_dir):
    """Sidecar with target_model_path matching basename surfaces; mismatch hides."""
    matched = "test_anim_preview_match.njm"
    unmatched = "test_anim_preview_other.njm"
    p1 = _stage_njm(njm_export_dir, matched, {
        "target_model_path": "bm_npc_kenkyu_w.bml",
        "source_glb": "test.glb",
        "retargeted_at_ms": 100,
        "frame_count": 4, "bone_count": 1, "fps": 30,
        "retargeted_bones": 1, "dropped_bones": 0,
        "njm_md5": "abc",
    })
    p2 = _stage_njm(njm_export_dir, unmatched, {
        "target_model_path": "bm_other_model.bml",
        "source_glb": "test.glb",
        "retargeted_at_ms": 200,
        "frame_count": 4, "bone_count": 1, "fps": 30,
        "retargeted_bones": 1, "dropped_bones": 0,
        "njm_md5": "def",
    })
    sc1 = njm_export_dir / (matched + ".preview.json")
    sc2 = njm_export_dir / (unmatched + ".preview.json")
    try:
        r = client.get("/api/anim_preview/list", params={"model_path": "bm_npc_kenkyu_w.bml"})
        assert r.status_code == 200
        data = r.json()
        names = [it["name"] for it in data["items"]]
        assert matched in names
        assert unmatched not in names

        # Path components should be stripped before matching.
        r2 = client.get("/api/anim_preview/list", params={"model_path": os.path.expanduser("~/PSOBB.IO/data/bm_npc_kenkyu_w.bml")})
        assert r2.status_code == 200
        names2 = [it["name"] for it in r2.json()["items"]]
        assert matched in names2

        # Case-insensitive match.
        r3 = client.get("/api/anim_preview/list", params={"model_path": "BM_NPC_KENKYU_W.BML"})
        assert r3.status_code == 200
        names3 = [it["name"] for it in r3.json()["items"]]
        assert matched in names3
    finally:
        _cleanup(p1, p2, sc1, sc2)


def test_anim_preview_list_skips_njms_without_sidecar(client, njm_export_dir):
    """A staged .njm with no sidecar is silently filtered out."""
    nosidecar = "test_anim_preview_nosidecar.njm"
    p = _stage_njm(njm_export_dir, nosidecar, sidecar=None)
    try:
        r = client.get("/api/anim_preview/list", params={"model_path": "anything.bml"})
        assert r.status_code == 200
        names = [it["name"] for it in r.json()["items"]]
        assert nosidecar not in names
    finally:
        _cleanup(p)


def test_anim_preview_list_returns_empty_for_unknown_model(client):
    """Endpoint 200s with empty list for models that have no imports."""
    r = client.get("/api/anim_preview/list", params={"model_path": "nothing_imports_here.bml"})
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 0
    assert data["items"] == []


def test_anim_preview_list_sorts_most_recent_first(client, njm_export_dir):
    older = "test_anim_preview_older.njm"
    newer = "test_anim_preview_newer.njm"
    p1 = _stage_njm(njm_export_dir, older, {
        "target_model_path": "test_sort.bml",
        "retargeted_at_ms": 100,
        "frame_count": 4, "bone_count": 1, "fps": 30,
        "retargeted_bones": 1, "dropped_bones": 0,
    })
    p2 = _stage_njm(njm_export_dir, newer, {
        "target_model_path": "test_sort.bml",
        "retargeted_at_ms": 999,
        "frame_count": 4, "bone_count": 1, "fps": 30,
        "retargeted_bones": 1, "dropped_bones": 0,
    })
    sc1 = njm_export_dir / (older + ".preview.json")
    sc2 = njm_export_dir / (newer + ".preview.json")
    try:
        r = client.get("/api/anim_preview/list", params={"model_path": "test_sort.bml"})
        assert r.status_code == 200
        items = r.json()["items"]
        assert items[0]["name"] == newer
        assert items[-1]["name"] == older
    finally:
        _cleanup(p1, p2, sc1, sc2)


# ---------------------------------------------------------------------------
# /api/anim_preview/data
# ---------------------------------------------------------------------------


def test_anim_preview_data_returns_animation_data_shape(client, njm_export_dir):
    """The data endpoint returns the same fields /api/animation_data does."""
    name = "test_anim_preview_data.njm"
    p = _stage_njm(njm_export_dir, name, {
        "target_model_path": "test_data.bml",
        "retargeted_at_ms": 100,
        "frame_count": 4, "bone_count": 1, "fps": 30,
    })
    sc = njm_export_dir / (name + ".preview.json")
    try:
        r = client.get("/api/anim_preview/data", params={"njm_path": name})
        assert r.status_code == 200
        data = r.json()
        # Required fields shared with /api/animation_data
        for k in ("filename", "motion", "frame_count", "fps", "bone_count",
                  "type_flags", "interpolation", "bones"):
            assert k in data, f"missing required field: {k}"
        # imported flag distinguishes it from the regular endpoint.
        assert data.get("imported") is True
        assert data["filename"] == name
        assert data["motion"] == name[:-4]
        assert data["target_model_path"] == "test_data.bml"
        assert isinstance(data["bones"], list)
        # Each bone has the same shape /api/animation_data uses.
        for b in data["bones"]:
            assert "idx" in b and "kf" in b and "present" in b
    finally:
        _cleanup(p, sc)


def test_anim_preview_data_404_on_missing(client):
    r = client.get("/api/anim_preview/data", params={"njm_path": "totally_does_not_exist.njm"})
    assert r.status_code == 404


def test_anim_preview_data_rejects_traversal(client):
    for bad in (
        "../escape.njm",
        "subdir/foo.njm",
        "foo\\bar.njm",
        "foo.njm.bak",       # not .njm
        "",
        "weird;name.njm",    # forbidden char
    ):
        r = client.get("/api/anim_preview/data", params={"njm_path": bad})
        assert r.status_code in (400, 404, 422), (
            f"expected reject for {bad!r}, got {r.status_code}: {r.text}"
        )


# ---------------------------------------------------------------------------
# /api/anim_preview/delete
# ---------------------------------------------------------------------------


def test_anim_preview_delete_removes_both_files(client, njm_export_dir):
    name = "test_anim_preview_delete.njm"
    p = _stage_njm(njm_export_dir, name, {
        "target_model_path": "delete_test.bml",
        "retargeted_at_ms": 100,
        "frame_count": 4, "bone_count": 1, "fps": 30,
    })
    sc = njm_export_dir / (name + ".preview.json")
    assert p.exists() and sc.exists()
    r = client.post("/api/anim_preview/delete", json={"njm_path": name})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert any(name in s for s in body["removed"])
    assert any("preview.json" in s for s in body["removed"])
    assert not p.exists()
    assert not sc.exists()


def test_anim_preview_delete_idempotent(client):
    """Re-deleting a file that's already gone is a no-op (200 with empty removed)."""
    r = client.post("/api/anim_preview/delete", json={"njm_path": "absolutely_not_there.njm"})
    assert r.status_code == 200
    assert r.json()["removed"] == []


def test_anim_preview_delete_rejects_traversal(client):
    for bad in ("../escape.njm", "subdir/foo.njm", "foo\\bar.njm", "no_ext"):
        r = client.post("/api/anim_preview/delete", json={"njm_path": bad})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# /api/import/animation: sidecar emission
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not Path("data/animation_assets/standing_typing.glb").is_file()
    and not Path(_REPO_ROOT / "data/animation_assets/standing_typing.glb").is_file(),
    reason="typing.glb missing — skipping import smoke test",
)
def test_import_animation_writes_preview_sidecar(client, njm_export_dir):
    """End-to-end: POST /api/import/animation -> sidecar exists + has expected keys."""
    glb_path = (_REPO_ROOT / "data/animation_assets/standing_typing.glb")
    if not glb_path.is_file():
        pytest.skip("typing.glb missing")
    out_name = "test_import_sidecar.njm"
    cleanup_njm = njm_export_dir / out_name
    cleanup_sc = njm_export_dir / (out_name + ".preview.json")
    _cleanup(cleanup_njm, cleanup_sc)
    psobb_data = Path(os.path.expanduser("~/PSOBB.IO/data/bm_npc_kenkyu_w.bml"))
    if not psobb_data.is_file():
        pytest.skip("bm_npc_kenkyu_w.bml not in PSOBB.IO/data")
    try:
        with open(glb_path, "rb") as f:
            files = {"file": ("typing.glb", f, "model/gltf-binary")}
            data = {
                "target_model_path": "bm_npc_kenkyu_w.bml",
                "motion_name": out_name,
                "include_translation": "false",
                "target_fps": "30",
                "flip_z": "true",
            }
            r = client.post("/api/import/animation", files=files, data=data)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        assert body["njm_name"] == out_name
        # Sidecar exists + has expected fields.
        assert cleanup_sc.is_file(), "sidecar missing after /api/import/animation"
        sc_data = json.loads(cleanup_sc.read_text(encoding="utf-8"))
        assert sc_data["target_model_path"] == "bm_npc_kenkyu_w.bml"
        assert sc_data["frame_count"] == body["frame_count"]
        assert sc_data["bone_count"] == body["bone_count"]
        assert sc_data["retargeted_bones"] == body["retargeted_bones"]
        assert sc_data["njm_md5"] == body["md5"]
        assert sc_data["fps"] == 30
        # And it shows up in /api/anim_preview/list.
        r2 = client.get("/api/anim_preview/list", params={"model_path": "bm_npc_kenkyu_w.bml"})
        assert r2.status_code == 200
        names = [it["name"] for it in r2.json()["items"]]
        assert out_name in names
    finally:
        _cleanup(cleanup_njm, cleanup_sc)


# ---------------------------------------------------------------------------
# Lobby-girl preview demo (Task 3)
# ---------------------------------------------------------------------------


def test_lobby_girl_typing_preview_demo(client, njm_export_dir):
    """The shipped lobby_girl_typing.njm + sidecar appear in the preview list.

    This is the Task 3 demo — the staged animation for the
    bm_npc_kenkyu_w.bml model is the proof-point that the import
    pipeline + preview UX works end-to-end.
    """
    p = njm_export_dir / "lobby_girl_typing.njm"
    sc = njm_export_dir / "lobby_girl_typing.njm.preview.json"
    if not p.is_file() or not sc.is_file():
        pytest.skip("lobby_girl_typing demo files missing — re-run import pipeline")
    sidecar = json.loads(sc.read_text(encoding="utf-8"))
    assert sidecar["target_model_path"].lower() == "bm_npc_kenkyu_w.bml"
    assert sidecar["frame_count"] == 90
    assert sidecar["fps"] == 30
    # The retargeted bone count from the lobby_girl bone map (19 mapped
    # source bones; 3 dropped — Jaw + LeftEye + RightEye).
    assert sidecar["retargeted_bones"] == 19
    assert sidecar["dropped_bones"] == 3

    # /api/anim_preview/list surfaces it.
    r = client.get("/api/anim_preview/list", params={"model_path": "bm_npc_kenkyu_w.bml"})
    assert r.status_code == 200
    names = [it["name"] for it in r.json()["items"]]
    assert "lobby_girl_typing.njm" in names

    # /api/anim_preview/data parses the file.
    r2 = client.get("/api/anim_preview/data", params={"njm_path": "lobby_girl_typing.njm"})
    assert r2.status_code == 200
    data = r2.json()
    assert data["frame_count"] == 90
    assert data["fps"] == 30.0
    assert data["target_model_path"] == "bm_npc_kenkyu_w.bml"
    assert len(data["bones"]) == data["bone_count"]


# ---------------------------------------------------------------------------
# Wrong-scope override sanity (Task 2)
# ---------------------------------------------------------------------------


def test_live_npcapcmot_is_pristine():
    """~/PSOBB.IO/data/NpcApcMot.bml must be vanilla PSOBB.

    The finishing-line agent's wrong-scope swap must NOT have leaked
    into <install>/data/. Live install mtime must match the original
    PSOBB ship state's md5: ``24562faeca14e36ae2fe8fab55bf2474``.
    """
    live = Path(os.path.expanduser("~/PSOBB.IO/data/NpcApcMot.bml"))
    if not live.is_file():
        pytest.skip("PSOBB.IO/data/NpcApcMot.bml missing — not on user's machine")
    md5 = hashlib.md5(live.read_bytes()).hexdigest()
    assert md5 == "24562faeca14e36ae2fe8fab55bf2474", (
        f"Live NpcApcMot.bml is NOT pristine PSOBB ship state (got {md5!r}). "
        f"The wrong-scope finishing-line agent's swap may have been deployed. "
        f"Restore from cache/njm_export/.WRONG_SCOPE_DO_NOT_DEPLOY backup."
    )


def test_wrong_scope_staged_npcapcmot_is_renamed():
    """cache/bml_export/NpcApcMot.bml must NOT be a deploy-ready artifact.

    The finishing-line agent's mis-scoped output should have been
    renamed to ``.WRONG_SCOPE_DO_NOT_DEPLOY`` so the deploy endpoint
    won't pick it up. (If it exists at all — the user may also have
    deleted it outright.)
    """
    bad = Path(r"C:/tmp_pso_editor/cache/bml_export/NpcApcMot.bml")
    assert not bad.exists(), (
        "cache/bml_export/NpcApcMot.bml exists — the wrong-scope staged "
        "BML must be deleted or renamed to .WRONG_SCOPE_DO_NOT_DEPLOY"
    )


# ---------------------------------------------------------------------------
# Sidecar shape contract
# ---------------------------------------------------------------------------


def test_sidecar_required_fields(njm_export_dir):
    """The sidecar JSON has all fields the list endpoint reads."""
    p = njm_export_dir / "lobby_girl_typing.njm.preview.json"
    if not p.is_file():
        pytest.skip("lobby_girl_typing sidecar missing")
    sc = json.loads(p.read_text(encoding="utf-8"))
    required = (
        "target_model_path",
        "source_glb",
        "retargeted_at_ms",
        "retargeted_bones",
        "dropped_bones",
        "frame_count",
        "bone_count",
        "fps",
    )
    for k in required:
        assert k in sc, f"missing sidecar field: {k}"


# ---------------------------------------------------------------------------
# NPC motion fallback (extends NpcApcMot.bml fallback to bm_n_*/bm_npc_*)
# ---------------------------------------------------------------------------


def test_npc_body_bml_pulls_motions_from_npcapcmot(client):
    """``bm_npc_*.bml`` and ``bm_n_*.bml`` lack inline NJMs but the
    fallback should expose every motion from NpcApcMot.bml so the
    receptionists / civilians pick up animations.

    Regression for the `_resolve_motion_sources` fallback list — before
    this fix, only ``pl*`` matched. Now ``bm_n_*`` and ``bm_npc_*`` also
    qualify (the patch in `_reports/handoff/npc_motion_fallback.patch`).
    """
    # Skip if PSOBB.IO/data isn't present (CI builds).
    psobb = Path(os.path.expanduser("~/PSOBB.IO/data"))
    if not psobb.is_dir() or not (psobb / "NpcApcMot.bml").is_file():
        pytest.skip("PSOBB.IO/data + NpcApcMot.bml not available")

    for host in ("bm_npc_kenkyu_w.bml", "bm_n_emw_i_body.bml"):
        if not (psobb / host).is_file():
            continue
        r = client.get(f"/api/animations/{host}")
        assert r.status_code == 200, r.text
        body = r.json()
        # Should pick up the master pack's 120 motions when the host BML
        # has no NJM siblings of its own. The exact count is the NPCApcMot
        # entry-count; we just assert "many", not a fixed number, to keep
        # the test resilient to install-side .bml swaps.
        assert body["motion_count"] > 50, (
            f"{host}: expected NpcApcMot.bml fallback, got "
            f"motion_count={body['motion_count']}"
        )
        labels = [m.get("source_path", "") for m in body["motions"][:5]]
        assert all(s.startswith("NpcApcMot.bml") for s in labels), (
            f"{host}: fallback labels should reference NpcApcMot.bml, "
            f"got {labels}"
        )


def test_player_class_motion_fallback_still_works(client):
    """The original player-class path (``pl*``) must keep working after
    the fallback list extension.

    Regression guard: ensures we extended the OR clause without
    accidentally narrowing the player-class branch.
    """
    psobb = Path(os.path.expanduser("~/PSOBB.IO/data"))
    if not psobb.is_dir() or not (psobb / "NpcApcMot.bml").is_file():
        pytest.skip("PSOBB.IO/data + NpcApcMot.bml not available")

    # plAnj.bml is a player-class container that has no inline NJMs.
    pl_bml = psobb / "plAnj.bml"
    if not pl_bml.is_file():
        pytest.skip("plAnj.bml not available")

    r = client.get("/api/animations/plAnj.bml")
    assert r.status_code == 200, r.text
    body = r.json()
    # The player-class fallback should pick up some motions — exact
    # count irrelevant, just that the fallback fires.
    assert body["motion_count"] > 0, (
        f"plAnj.bml: player-class NpcApcMot fallback regressed "
        f"(motion_count=0)"
    )
