"""Smoke tests for scripts/render_coverage_audit.py.

The audit script needs a running server (it hits /api/manifest_lite +
/api/model_bundle), so we don't try to invoke it end-to-end here.
Instead we exercise the pure-function bits: the grading helpers + the
prefix-list classification.

A full end-to-end run is exercised manually by:
    python scripts/render_coverage_audit.py
which writes _reports/render_coverage.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add scripts/ to sys.path so we can import the audit module.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import render_coverage_audit as rca  # noqa: E402


def test_no_anim_prefixes_cover_static_props():
    """Static props (bm_obj_, bm_eff_, set_, np_obj_) are pre-classified."""
    assert any(p.startswith("bm_obj") for p in rca.NO_ANIM_PREFIXES)
    assert any(p.startswith("bm_eff") for p in rca.NO_ANIM_PREFIXES)
    assert any(p.startswith("set_") for p in rca.NO_ANIM_PREFIXES)


def test_no_anim_prefixes_cover_player_parts():
    """Player hair / cap / hed should be pre-classified as no-anim.

    They're decoration that shares animations with the parent body.
    """
    for part in ("hai", "cap", "hed", "bdy", "arm", "lwr", "ext", "fac"):
        assert any(part in p for p in rca.NO_ANIM_PREFIXES), (
            f"missing prefix for player part {part!r}"
        )


def test_no_anim_prefixes_cover_npc_bodies():
    """NPC bodies (bm_n_, bm_npc_) are pre-classified as no-anim.

    They get their motions from NpcApcMot.bml at runtime; the per-body
    BMLs ship without sibling NJMs.
    """
    assert "bm_n_" in rca.NO_ANIM_PREFIXES
    assert "bm_npc_" in rca.NO_ANIM_PREFIXES


def test_grade_bundle_handles_missing_skinned():
    """Bundle response with skinned=null is graded missing_skinned."""
    target = {
        "key": "x.bml#y.nj", "container": "x.bml", "inner": "y.nj",
        "ext": ".nj", "infered_category": "",
    }
    body = {
        "skinned": {"meshes": [], "bones": []},
        "animations": {"motions": []},
    }
    row = rca._grade_bundle(target, 200, body)
    assert row["status"] == "missing_skinned"
    assert "no meshes" in row["note"]


def test_grade_bundle_handles_unsupported_xj():
    """A 400 saying .xj cannot be skinned is unsupported_route."""
    target = {
        "key": "x.bml#y.xj", "container": "x.bml", "inner": "y.xj",
        "ext": ".xj", "infered_category": "",
    }
    body = {"detail": "skinned mesh requires .nj inner (got '.xj')"}
    row = rca._grade_bundle(target, 400, body)
    assert row["status"] == "unsupported_route"


def test_grade_bundle_marks_ok_with_textures():
    """A populated skinned + animations + textures payload is ``ok``."""
    target = {
        "key": "x.bml#y.nj", "container": "x.bml", "inner": "y.nj",
        "ext": ".nj", "infered_category": "Bosses",
    }
    body = {
        "skinned": {
            "meshes": [{"vertex_count": 8}],
            "bones": [{"name": "root"}, {"name": "spine"}],
            "binding_data": {
                "njtl": [{"slot": 0, "name": "tex0"}],
                "binding": [{"material_id": 0, "tile_index": 0,
                             "missing": False, "source": "in_bml",
                             "name": "tex0"}],
            },
        },
        "animations": {"motions": [{"name": "walk"}, {"name": "run"}]},
    }
    row = rca._grade_bundle(target, 200, body)
    assert row["status"] == "ok"
    assert row["n_textures"] == 1
    assert row["n_animations"] == 2
    assert row["has_skinned"] is True


def test_grade_bundle_marks_no_textures_when_njtl_unresolved():
    """NJTL declares names but binding has 0 resolved -> ok_no_textures."""
    target = {
        "key": "x.bml#y.nj", "container": "x.bml", "inner": "y.nj",
        "ext": ".nj", "infered_category": "",
    }
    body = {
        "skinned": {
            "meshes": [{"vertex_count": 8}],
            "bones": [{"name": "root"}],
            "binding_data": {
                "njtl": [{"slot": 0, "name": "tex0"}],
                "binding": [{"material_id": 0, "missing": True,
                             "source": "unmatched", "name": "tex0"}],
            },
        },
        "animations": {"motions": [{"name": "walk"}]},
    }
    row = rca._grade_bundle(target, 200, body)
    assert row["status"] == "ok_no_textures"


def test_grade_bundle_marks_no_anim_for_static_props():
    """Static prop families (bm_obj_) with 0 motions are ``ok``, not ok_no_animations."""
    target = {
        "key": "bm_obj_box.bml#box.nj", "container": "bm_obj_box.bml",
        "inner": "box.nj", "ext": ".nj", "infered_category": "Objects",
    }
    body = {
        "skinned": {
            "meshes": [{"vertex_count": 8}],
            "bones": [{"name": "root"}],
            "binding_data": {"njtl": [], "binding": []},
        },
        "animations": {"motions": []},
    }
    row = rca._grade_bundle(target, 200, body)
    assert row["status"] == "ok"


def test_grade_bundle_flags_no_anim_for_unexpected_family():
    """A family NOT in NO_ANIM_PREFIXES with 0 motions is ok_no_animations."""
    target = {
        "key": "bm_unknown.bml#x.nj", "container": "bm_unknown.bml",
        "inner": "x.nj", "ext": ".nj", "infered_category": "",
    }
    body = {
        "skinned": {
            "meshes": [{"vertex_count": 8}],
            "bones": [{"name": "root"}],
            "binding_data": {"njtl": [], "binding": []},
        },
        "animations": {"motions": []},
    }
    row = rca._grade_bundle(target, 200, body)
    assert row["status"] == "ok_no_animations"


def test_grade_scene_marks_ok_when_meshes_present():
    """Scene asset with meshes -> ok."""
    target = {
        "key": "scene/map_a.nj", "container": "", "inner": "",
        "ext": ".nj", "infered_category": "Maps / Terrain",
        "path": "scene/map_a.nj",
    }
    body = {"meshes": [{"vertex_count": 24, "triangle_count": 12}]}
    row = rca._grade_scene(target, 200, body)
    assert row["status"] == "ok"
    assert row["has_skinned"] is True


def test_grade_scene_marks_missing_when_empty():
    target = {
        "key": "scene/empty.nj", "container": "", "inner": "",
        "ext": ".nj", "infered_category": "Maps / Terrain",
        "path": "scene/empty.nj",
    }
    body = {"meshes": []}
    row = rca._grade_scene(target, 200, body)
    assert row["status"] == "missing_skinned"


def test_grade_mesh_only_marks_ok_when_meshes_present():
    """Mesh-only fall-through for .xj inners."""
    target = {
        "key": "x.bml#y.xj", "container": "x.bml", "inner": "y.xj",
        "ext": ".xj", "infered_category": "",
    }
    body = {
        "meshes": [{"vertex_count": 8}],
        "binding_data": {
            "njtl": [{"slot": 0, "name": "t"}],
            "binding": [{"material_id": 0, "missing": False,
                         "source": "in_bml", "name": "t", "tile_index": 0}],
        },
    }
    row = rca._grade_mesh_only(target, 200, body)
    assert row["status"] == "ok"
    assert row["n_textures"] == 1
