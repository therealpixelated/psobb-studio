"""Tests for /api/anim_library/{list,delete,rename,zip}.

Exercises the global animation library backend without depending on
real PSOBB.IO assets — we synthesize a few .njm + .preview.json files
in cache/njm_export/, then walk through every endpoint.
"""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def srv():
    if "server" in sys.modules:
        del sys.modules["server"]
    import server  # noqa: F401
    return sys.modules["server"]


@pytest.fixture
def client(srv):
    return TestClient(srv.app)


@pytest.fixture
def fixture_njms(srv):
    """Create a handful of fake .njm + sidecar pairs in cache/njm_export/.

    Cleans up at teardown so the user's own staged animations stay safe.
    Names are uniquely prefixed so we can identify them across the entire
    library (if the user happens to have other animations).
    """
    d = srv.NJM_EXPORT_DIR
    d.mkdir(parents=True, exist_ok=True)
    PREFIX = "__anim_lib_test_"
    fixtures = [
        # (name, sidecar)
        (f"{PREFIX}walk.njm", {
            "njm_md5": "deadbeef" * 4,
            "frame_count": 30,
            "bone_count": 24,
            "fps": 30.0,
            "source_glb": "test_walk.glb",
            "source_animation": "Walk",
            "target_model_path": "bm_boss1_dragon.bml#dragon_body",
            "retargeted_at_ms": 1700000000000,
            "retargeted_bones": 22,
            "dropped_bones": 2,
            "bone_map": "auto",
        }),
        (f"{PREFIX}run.njm", {
            "njm_md5": "feedface" * 4,
            "frame_count": 16,
            "bone_count": 24,
            "fps": 60.0,
            "source_glb": "test_run.glb",
            "source_animation": "Run",
            "target_model_path": "bm_boss1_dragon.bml#dragon_body",
            "retargeted_at_ms": 1700001000000,
            "retargeted_bones": 24,
            "dropped_bones": 0,
            "bone_map": "explicit",
        }),
        # No sidecar — simulates an .njm staged via /api/anim_keyframe/save.
        (f"{PREFIX}legacy.njm", None),
    ]
    created: list[Path] = []
    for name, sidecar in fixtures:
        p = d / name
        p.write_bytes(b"\x00" * 64)
        created.append(p)
        if sidecar is not None:
            sc = d / (name + ".preview.json")
            sc.write_text(json.dumps(sidecar), encoding="utf-8")
            created.append(sc)

    yield [name for name, _ in fixtures]

    for p in created:
        try:
            p.unlink()
        except OSError:
            pass
    # Also nuke anything renamed off our prefix in the test (rename moves them).
    for p in d.glob(PREFIX + "*"):
        try:
            p.unlink()
        except OSError:
            pass
    # And anything renamed via prefix in tests:
    for p in d.glob("__renamed_" + PREFIX + "*"):
        try:
            p.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# /list
# ---------------------------------------------------------------------------


def test_list_returns_fixtures(client, fixture_njms):
    r = client.get("/api/anim_library/list")
    assert r.status_code == 200
    data = r.json()
    assert "items" in data
    assert "totals" in data
    names = {it["name"] for it in data["items"]}
    for fn in fixture_njms:
        assert fn in names, f"fixture {fn} missing from {names}"


def test_list_entries_have_required_fields(client, fixture_njms):
    r = client.get("/api/anim_library/list")
    items = r.json()["items"]
    walk = next(it for it in items if it["name"].endswith("walk.njm"))
    # All sidecar-derived fields must be present.
    for k in ("display_name", "size", "mtime_ms", "md5",
              "frame_count", "bone_count", "fps",
              "source_glb", "source_animation",
              "target_model_path", "target_model_name",
              "retargeted_at_ms", "retargeted_bones", "dropped_bones",
              "bone_map", "has_sidecar"):
        assert k in walk, f"missing key {k}"
    assert walk["has_sidecar"] is True
    assert walk["frame_count"] == 30
    assert walk["fps"] == 30.0
    assert walk["target_model_name"] == "bm_boss1_dragon.bml"


def test_list_handles_legacy_njm_without_sidecar(client, fixture_njms):
    r = client.get("/api/anim_library/list")
    items = r.json()["items"]
    legacy = next(it for it in items if it["name"].endswith("legacy.njm"))
    assert legacy["has_sidecar"] is False
    # md5 still computed even without sidecar (read-and-hash fallback).
    assert legacy["md5"], f"md5 missing for legacy njm: {legacy}"
    assert len(legacy["md5"]) == 32  # md5 hex len


def test_list_totals_match(client, fixture_njms):
    r = client.get("/api/anim_library/list")
    data = r.json()
    items = data["items"]
    totals = data["totals"]
    assert totals["size"] == sum(it["size"] for it in items)
    sidecared = sum(1 for it in items if it["has_sidecar"])
    assert totals["with_sidecar"] == sidecared


# ---------------------------------------------------------------------------
# /delete
# ---------------------------------------------------------------------------


def test_delete_removes_njm_and_sidecar(client, srv, fixture_njms):
    target = "__anim_lib_test_walk.njm"
    r = client.post("/api/anim_library/delete", json={"names": [target]})
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True
    res = j["results"]
    assert len(res) == 1
    assert res[0]["removed"] is True
    # Both file + sidecar gone.
    d = srv.NJM_EXPORT_DIR
    assert not (d / target).exists()
    assert not (d / (target + ".preview.json")).exists()


def test_delete_idempotent(client, fixture_njms):
    """Calling delete on an already-removed file reports removed=False but no error."""
    target = "__anim_lib_test_walk.njm"
    client.post("/api/anim_library/delete", json={"names": [target]})
    r = client.post("/api/anim_library/delete", json={"names": [target]})
    assert r.status_code == 200
    res = r.json()["results"][0]
    assert res["removed"] is False
    assert res.get("error") in (None, "")


def test_delete_rejects_path_traversal(client):
    r = client.post("/api/anim_library/delete", json={"names": ["../foo.njm"]})
    assert r.status_code == 200  # batch ok with per-item error
    res = r.json()["results"][0]
    assert res["removed"] is False
    assert "forbidden" in (res["error"] or "") or "must be a bare filename" in (res["error"] or "")


def test_delete_rejects_too_many(client):
    big = ["x{}.njm".format(i) for i in range(1100)]
    r = client.post("/api/anim_library/delete", json={"names": big})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# /rename
# ---------------------------------------------------------------------------


def test_rename_moves_file_and_sidecar(client, srv, fixture_njms):
    new_name = "__renamed___anim_lib_test_walk.njm"
    r = client.post("/api/anim_library/rename", json={"renames": [{
        "old_name": "__anim_lib_test_walk.njm",
        "new_name": new_name,
    }]})
    assert r.status_code == 200
    j = r.json()
    assert j["results"][0]["renamed"] is True
    assert j["results"][0]["sidecar_moved"] is True
    d = srv.NJM_EXPORT_DIR
    assert (d / new_name).exists()
    assert (d / (new_name + ".preview.json")).exists()
    assert not (d / "__anim_lib_test_walk.njm").exists()


def test_rename_rejects_existing_target(client, fixture_njms):
    r = client.post("/api/anim_library/rename", json={"renames": [{
        "old_name": "__anim_lib_test_walk.njm",
        "new_name": "__anim_lib_test_run.njm",  # already exists
    }]})
    assert r.status_code == 200
    res = r.json()["results"][0]
    assert res["renamed"] is False
    assert "exists" in (res["error"] or "").lower()


def test_rename_validates_extension(client):
    r = client.post("/api/anim_library/rename", json={"renames": [{
        "old_name": "foo.njm",
        "new_name": "bar.txt",
    }]})
    assert r.status_code == 200
    res = r.json()["results"][0]
    assert res["renamed"] is False
    assert "must end with .njm" in (res["error"] or "").lower()


# ---------------------------------------------------------------------------
# /zip
# ---------------------------------------------------------------------------


def test_zip_packs_njms(client, fixture_njms):
    r = client.post(
        "/api/anim_library/zip",
        json={"names": ["__anim_lib_test_walk.njm", "__anim_lib_test_run.njm"]},
    )
    assert r.status_code == 200
    assert r.headers.get("content-type") == "application/zip"
    assert int(r.headers.get("X-Anim-Count", "0")) == 2
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    assert "njm/__anim_lib_test_walk.njm" in names
    assert "njm/__anim_lib_test_run.njm" in names
    # Sidecar present for the items that had one.
    assert "sidecar/__anim_lib_test_walk.njm.preview.json" in names
    assert "sidecar/__anim_lib_test_run.njm.preview.json" in names


def test_zip_skips_missing_files_but_still_succeeds(client, fixture_njms):
    r = client.post(
        "/api/anim_library/zip",
        json={"names": ["__anim_lib_test_walk.njm", "__nope_does_not_exist.njm"]},
    )
    assert r.status_code == 200
    assert int(r.headers.get("X-Anim-Count", "0")) == 1


def test_zip_404_when_nothing_matches(client):
    r = client.post(
        "/api/anim_library/zip",
        json={"names": ["__nope_a.njm", "__nope_b.njm"]},
    )
    assert r.status_code == 404


def test_zip_rejects_empty_list(client):
    r = client.post("/api/anim_library/zip", json={"names": []})
    assert r.status_code == 400


def test_zip_rejects_path_traversal(client):
    r = client.post("/api/anim_library/zip", json={"names": ["../etc.njm"]})
    assert r.status_code == 400
