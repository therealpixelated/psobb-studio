"""Tests for the Map Editor — formats.scene_loader + /api/map/* endpoints.

Covers:
  - scene_loader.catalogue() builds 30+ maps grouped into 9 categories.
  - floor_bundle() returns {renderable, textures, scripts, ...} per floor.
  - validate_edits_payload() rejects malformed inputs (bad type, missing
    fields, dangling waypoint references, self-loops, dup ids).
  - /api/map/list           returns the picker payload.
  - /api/map/<id>           returns the bundle, defaults to lowest floor
                            when the requested one doesn't exist.
  - /api/map/<id>           rejects bad map_ids (path injection guard).
  - /api/map/edits          POST validates + writes to cache/map_edits/.
  - /api/map/edits/<id>     GET reads it back; returns empty for unknown.

The endpoint tests use an in-process FastAPI TestClient so they
don't need a live server, and they DON'T touch the live data dir.
The cache write goes to the real cache/map_edits/ — we clean up after.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from formats import scene_loader as sl


# ---------------------------------------------------------------------------
# scene_loader unit tests (no I/O — synthetic manifest entries)
# ---------------------------------------------------------------------------
def _ent(path: str, size: int = 1024, category: str = "map") -> dict:
    return {"path": path, "size": size, "category": category}


SAMPLE_ENTRIES = [
    _ent("scene/map_aancient01_00s.nj", 815808),
    _ent("scene/map_aancient01_00s.xj", 384096),
    _ent("scene/map_aancient01_00s.xvm", 200000),
    _ent("scene/map_aancient01_00c.rel", 461984),
    _ent("scene/map_aancient01_00n.rel", 815808),
    _ent("scene/map_aancient01_00r.rel", 34336),
    _ent("scene/map_aancient01_00bm.bin", 12000),
    _ent("scene/map_aancient01_01s.nj", 700000),
    _ent("scene/map_aancient01_01s.xj", 350000),
    _ent("scene/map_machine01_00s.nj", 500000),
    _ent("scene/map_jungle04_05s.nj", 600000),  # tests floor 5 (not 0..4)
    _ent("scene/map_boss09_00.tam", 8000),
    _ent("scene/map_city00_00c.rel", 100000),  # city has only rel files
    _ent("scene/map_city00.xvm", 50000),       # alt-name asset (no floor)
    # Non-map / unrelated paths — must be ignored
    _ent("scene/random_unrelated.bin"),
    _ent("biri_ball.bml", category="model"),
    _ent("scene/teststuff.txt"),
]


def test_catalogue_groups_by_map():
    maps = sl.catalogue(SAMPLE_ENTRIES)
    by_id = {m.map_id: m for m in maps}
    assert "aancient01" in by_id
    assert "machine01"  in by_id
    assert "jungle04"   in by_id
    assert "boss09"     in by_id
    assert "city00"     in by_id

    a = by_id["aancient01"]
    assert a.category == "forest"
    assert a.area == "aancient"
    assert a.area_num == 1
    assert sorted(a.floors.keys()) == [0, 1]
    # Floor 0 should have the .nj, .xj, .xvm, three .rels, .bin
    floor0 = a.floors[0]
    paths = sorted(p.path for p in floor0)
    assert "scene/map_aancient01_00s.nj" in paths
    assert "scene/map_aancient01_00s.xj" in paths
    assert "scene/map_aancient01_00s.xvm" in paths
    assert sum(1 for p in floor0 if p.ext == "rel") == 3


def test_catalogue_classifies_categories():
    maps = sl.catalogue(SAMPLE_ENTRIES)
    by_id = {m.map_id: m for m in maps}
    assert by_id["aancient01"].category == "forest"
    assert by_id["machine01"].category  == "mine"
    assert by_id["jungle04"].category   == "corruption"
    assert by_id["boss09"].category     == "boss"
    assert by_id["city00"].category     == "city"


def test_floor_bundle_groups_assets_by_kind():
    maps = sl.catalogue(SAMPLE_ENTRIES)
    info = next(m for m in maps if m.map_id == "aancient01")
    bundle = sl.floor_bundle(info, 0)
    assert bundle["map_id"] == "aancient01"
    assert bundle["floor"] == 0
    assert len(bundle["renderable"]) == 2  # nj + xj
    assert len(bundle["textures"])   == 1  # xvm
    assert len(bundle["scripts"])    == 3  # 3 .rel files
    # Renderable kind is "terrain" for the .s suffix files
    for r in bundle["renderable"]:
        assert r["kind"] == "terrain"


def test_floor_bundle_unknown_floor_returns_empty_lists():
    maps = sl.catalogue(SAMPLE_ENTRIES)
    info = next(m for m in maps if m.map_id == "aancient01")
    # Floor 99 doesn't exist — bundle returns empty lists
    bundle = sl.floor_bundle(info, 99)
    assert bundle["renderable"] == []
    assert bundle["textures"] == []


def test_make_picker_payload_has_categories_and_maps():
    maps = sl.catalogue(SAMPLE_ENTRIES)
    p = sl.make_picker_payload(maps)
    cats = {c["id"] for c in p["categories"]}
    # All declared categories must exist
    assert {"city", "forest", "cave", "mine", "ruins", "battle",
            "corruption", "boss", "other"}.issubset(cats)
    # Every map has the required fields
    for m in p["maps"]:
        assert "map_id" in m
        assert "category" in m
        assert "floors" in m
        assert isinstance(m["floors"], list)


def test_catalogue_handles_empty_entries():
    maps = sl.catalogue([])
    assert maps == []


def test_catalogue_skips_non_map_paths():
    maps = sl.catalogue([
        _ent("biri_ball.bml", category="model"),
        _ent("foo.txt"),
        _ent("scene/notmap.bin"),
    ])
    assert maps == []


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------
def test_validate_minimal_valid():
    ok, err = sl.validate_edits_payload({
        "map_id": "aancient01",
        "spawns": [],
        "waypoints": [],
    })
    assert ok, err


def test_validate_rejects_bad_map_id():
    ok, err = sl.validate_edits_payload({
        "map_id": "../etc/passwd",
        "spawns": [],
        "waypoints": [],
    })
    assert not ok
    assert "map_id" in err


def test_validate_rejects_unknown_spawn_type():
    ok, err = sl.validate_edits_payload({
        "map_id": "aancient01",
        "spawns": [{
            "id": 1,
            "type": "rocket",
            "world_pos": [0, 0, 0],
        }],
        "waypoints": [],
    })
    assert not ok
    assert "type" in err


def test_validate_rejects_dup_spawn_ids():
    ok, err = sl.validate_edits_payload({
        "map_id": "aancient01",
        "spawns": [
            {"id": 1, "type": "mob", "world_pos": [0, 0, 0]},
            {"id": 1, "type": "npc", "world_pos": [1, 1, 1]},
        ],
        "waypoints": [],
    })
    assert not ok
    assert "duplicate" in err


def test_validate_rejects_waypoint_with_missing_endpoints():
    ok, err = sl.validate_edits_payload({
        "map_id": "aancient01",
        "spawns": [{"id": 1, "type": "mob", "world_pos": [0, 0, 0]}],
        "waypoints": [{"from_id": 1, "to_id": 99}],
    })
    assert not ok
    assert "missing" in err


def test_validate_rejects_waypoint_self_loop():
    ok, err = sl.validate_edits_payload({
        "map_id": "aancient01",
        "spawns": [{"id": 1, "type": "mob", "world_pos": [0, 0, 0]}],
        "waypoints": [{"from_id": 1, "to_id": 1}],
    })
    assert not ok
    assert "self-loop" in err


def test_validate_rejects_bad_waypoint_style():
    ok, err = sl.validate_edits_payload({
        "map_id": "aancient01",
        "spawns": [
            {"id": 1, "type": "mob", "world_pos": [0, 0, 0]},
            {"id": 2, "type": "mob", "world_pos": [1, 0, 0]},
        ],
        "waypoints": [{"from_id": 1, "to_id": 2, "style": "fly"}],
    })
    assert not ok
    assert "style" in err


def test_validate_rejects_bad_world_pos():
    ok, err = sl.validate_edits_payload({
        "map_id": "aancient01",
        "spawns": [{"id": 1, "type": "mob", "world_pos": [0, 0]}],  # only 2 floats
        "waypoints": [],
    })
    assert not ok
    assert "world_pos" in err


def test_normalize_strips_extra_keys():
    raw = {
        "map_id": "aancient01",
        "spawns": [{
            "id": 1,
            "type": "mob",
            "world_pos": [10, 20, 30],
            "rotation": 0.5,
            "type_data": {"mob_id": 0x4B, "count": 3},
            "extra_garbage": "ignored",
        }],
        "waypoints": [],
    }
    norm = sl.normalize_edits_payload(raw)
    assert "extra_garbage" not in norm["spawns"][0]
    assert norm["spawns"][0]["world_pos"] == [10.0, 20.0, 30.0]
    assert norm["version"] == sl.SPAWN_FILE_VERSION


# ---------------------------------------------------------------------------
# Endpoint tests via TestClient
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    """In-process FastAPI client. Imports server.py once per module."""
    import server
    return TestClient(server.app)


@pytest.fixture
def cleanup_test_sidecar():
    """Yield, then clean any test sidecar JSON we wrote."""
    yield
    import server
    for fn in ("aancient01.json", "test_dummy.json", "machine01.json"):
        p = server.MAP_EDITS_DIR / fn
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass


def test_api_map_list_returns_categories_and_maps(client):
    r = client.get("/api/map/list")
    assert r.status_code == 200, r.text
    data = r.json()
    assert "categories" in data
    assert "maps" in data
    assert len(data["categories"]) >= 9
    # The forest category must have at least one entry — aancient01
    # ships in every PSOBB.IO build
    forests = [m for m in data["maps"] if m["category"] == "forest"]
    assert any(m["map_id"] == "aancient01" for m in forests)


def test_api_map_get_aancient01_floor0(client):
    r = client.get("/api/map/aancient01?floor=0")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["map_id"] == "aancient01"
    assert b["floor"] == 0
    # At least one renderable terrain (.nj or .xj)
    assert len(b["renderable"]) > 0


def test_api_map_get_unknown_id_404(client):
    r = client.get("/api/map/zzzzzz99")
    assert r.status_code == 404


def test_api_map_get_rejects_bad_id(client):
    r = client.get("/api/map/..bad")
    # 400 (regex reject) or 404 (no match) — both are fine; no 200
    assert r.status_code in (400, 404)


def test_api_map_get_unknown_floor_falls_back(client):
    # Aancient01 doesn't have floor 99; server should default
    r = client.get("/api/map/aancient01?floor=99")
    assert r.status_code == 200
    b = r.json()
    # Returned floor != 99 (defaulted to lowest)
    assert b["floor"] != 99


def test_api_map_edits_load_unknown_returns_empty(client):
    # Pick a map_id that matches the regex but won't have a sidecar yet.
    r = client.get("/api/map/edits/test01")
    assert r.status_code == 200
    data = r.json()
    assert data["exists"] is False
    assert data["spawns"] == []
    assert data["waypoints"] == []


def test_api_map_edits_save_load_roundtrip(client, cleanup_test_sidecar):
    payload = {
        "map_id": "aancient01",
        "spawns": [
            {
                "id": 1,
                "type": "mob",
                "world_pos": [10.5, 0.0, 5.0],
                "rotation": 0.0,
                "type_data": {"mob_id": 0x4B, "count": 3, "behavior": "patrol"},
            },
            {
                "id": 2,
                "type": "npc",
                "world_pos": [20.0, 0.0, 5.0],
                "rotation": 1.57,
                "type_data": {"dialog_id": 42, "name": "Guide"},
            },
        ],
        "waypoints": [
            {"from_id": 1, "to_id": 2, "speed": 1.5, "style": "walk"},
        ],
    }
    r = client.post("/api/map/edits", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["spawn_count"] == 2
    assert body["waypoint_count"] == 1

    # Read it back
    r = client.get("/api/map/edits/aancient01")
    assert r.status_code == 200
    data = r.json()
    assert data["exists"] is True
    assert len(data["spawns"]) == 2
    assert data["spawns"][0]["type"] == "mob"
    assert data["spawns"][0]["type_data"]["mob_id"] == 0x4B


def test_api_map_edits_save_rejects_bad_map_id(client):
    r = client.post("/api/map/edits", json={
        "map_id": "../escape",
        "spawns": [],
        "waypoints": [],
    })
    assert r.status_code == 400


def test_api_map_edits_save_rejects_invalid_spawn(client):
    r = client.post("/api/map/edits", json={
        "map_id": "aancient01",
        "spawns": [{"id": 1, "type": "rocket", "world_pos": [0, 0, 0]}],
        "waypoints": [],
    })
    assert r.status_code == 400


def test_api_map_edits_save_rejects_dangling_waypoint(client):
    r = client.post("/api/map/edits", json={
        "map_id": "aancient01",
        "spawns": [{"id": 1, "type": "mob", "world_pos": [0, 0, 0]}],
        "waypoints": [{"from_id": 1, "to_id": 99}],
    })
    assert r.status_code == 400
