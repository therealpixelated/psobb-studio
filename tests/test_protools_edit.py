"""Pro-tools edit-mode tests — Edit tab (vertex selection + transform gizmo).

Covers:
  - /api/protools/save_vertex_transforms round-trip + sha stability
  - /api/protools/<sha> re-fetch
  - /api/protools/list/<path> filter by host archive
  - Validation errors (mismatched lengths, out-of-range indices, body-too-big)
  - JS module sanity: every new file is loaded by index.html and exposes
    its public API surface
  - Edit-mode integration with the existing undo bus

Each save uses a synthetic mesh path so we don't hit cache pollution.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import server
    return TestClient(server.app)


@pytest.fixture(autouse=True)
def _clean_protools_dir():
    """Wipe any test_*-prefixed sidecars between tests."""
    import server
    for p in server.PROTOOLS_EDITS_DIR.glob("*test_*.json"):
        try:
            p.unlink()
        except OSError:
            pass
    yield
    for p in server.PROTOOLS_EDITS_DIR.glob("*test_*.json"):
        try:
            p.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Save / fetch / list round-trip
# ---------------------------------------------------------------------------

def _payload(model_path: str = "test_model.bml#test_model.nj"):
    """Synthetic save payload with one submesh edit."""
    return {
        "model_path": model_path,
        "submeshes": [
            {
                "submesh_idx": 0,
                "material_id": 17,
                "vertex_count": 100,
                "indices": [3, 5, 8, 12],
                "displacement": [
                    0.1, 0.0, 0.0,
                    0.0, 0.2, 0.0,
                    0.0, 0.0, -0.3,
                    0.05, 0.05, 0.05,
                ],
            }
        ],
        "subdivide_level": 0,
    }


def test_save_returns_sha_and_writes_file(client):
    r = client.post("/api/protools/save_vertex_transforms", json=_payload())
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    assert isinstance(data["sha"], str)
    assert len(data["sha"]) == 16
    assert data["size"] > 0
    assert "test_model" in data["cache_path"]
    assert os.path.exists(data["cache_path"])


def test_save_sha_is_deterministic(client):
    r1 = client.post("/api/protools/save_vertex_transforms", json=_payload())
    r2 = client.post("/api/protools/save_vertex_transforms", json=_payload())
    assert r1.json()["sha"] == r2.json()["sha"]


def test_save_then_fetch_round_trips(client):
    r = client.post("/api/protools/save_vertex_transforms", json=_payload())
    sha = r.json()["sha"]
    f = client.get(f"/api/protools/{sha}")
    assert f.status_code == 200, f.text
    out = f.json()
    assert out["ok"] is True
    assert out["sha"] == sha
    ep = out["edit_payload"]
    assert ep["model_path"] == "test_model.bml#test_model.nj"
    assert len(ep["submeshes"]) == 1
    assert ep["submeshes"][0]["indices"] == [3, 5, 8, 12]


def test_list_filters_by_host(client):
    # Save two with different host archives.
    p1 = _payload("test_alpha.bml#a.nj")
    p2 = _payload("test_beta.bml#b.nj")
    client.post("/api/protools/save_vertex_transforms", json=p1)
    client.post("/api/protools/save_vertex_transforms", json=p2)

    r = client.get("/api/protools/list/test_alpha.bml")
    assert r.status_code == 200, r.text
    data = r.json()
    paths = [e["model_path"] for e in data["edits"]]
    assert any("test_alpha" in p for p in paths)
    assert not any("test_beta" in p for p in paths)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_save_empty_submeshes_400(client):
    bad = {"model_path": "x.nj", "submeshes": []}
    r = client.post("/api/protools/save_vertex_transforms", json=bad)
    assert r.status_code == 400, r.text


def test_save_displacement_length_mismatch_422(client):
    bad = _payload()
    bad["submeshes"][0]["displacement"] = [0.1, 0.2]  # need 12, have 2
    r = client.post("/api/protools/save_vertex_transforms", json=bad)
    assert r.status_code == 422, r.text
    assert "displacement length" in r.text.lower() or "displacement" in r.text


def test_save_index_out_of_range_422(client):
    bad = _payload()
    bad["submeshes"][0]["indices"] = [3, 5, 8, 9999]
    r = client.post("/api/protools/save_vertex_transforms", json=bad)
    assert r.status_code == 422, r.text


def test_save_negative_index_422(client):
    bad = _payload()
    bad["submeshes"][0]["indices"] = [3, -1, 8, 12]
    r = client.post("/api/protools/save_vertex_transforms", json=bad)
    assert r.status_code == 422, r.text


def test_fetch_unknown_sha_404(client):
    r = client.get("/api/protools/cafef00d12345678")
    assert r.status_code == 404


def test_fetch_invalid_sha_400(client):
    r = client.get("/api/protools/notahexsha!!!")
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Sha stability vs payload changes
# ---------------------------------------------------------------------------

def test_sha_changes_with_model_path(client):
    a = client.post("/api/protools/save_vertex_transforms",
                    json=_payload("test_a.nj")).json()["sha"]
    b = client.post("/api/protools/save_vertex_transforms",
                    json=_payload("test_b.nj")).json()["sha"]
    assert a != b


def test_sha_changes_with_displacement(client):
    p1 = _payload()
    p2 = _payload()
    p2["submeshes"][0]["displacement"][0] += 0.01
    a = client.post("/api/protools/save_vertex_transforms", json=p1).json()["sha"]
    b = client.post("/api/protools/save_vertex_transforms", json=p2).json()["sha"]
    assert a != b


# ---------------------------------------------------------------------------
# JS bundle sanity — every new module is wired in index.html
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_index_html_loads_protools_modules():
    html = (PROJECT_ROOT / "static" / "index.html").read_text(encoding="utf-8")
    for src in (
        "transform_gizmo.js",
        "edit_panel.js",
        "skeleton_panel.js",
        "uv_panel.js",
    ):
        assert f"/static/{src}" in html, f"{src} not loaded by index.html"


def test_index_html_has_edit_mode_button():
    html = (PROJECT_ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="btnEditMode"' in html, "Edit Mode toolbar button missing"


def test_protools_modules_exist_on_disk():
    for fn in (
        "transform_gizmo.js",
        "edit_panel.js",
        "skeleton_panel.js",
        "uv_panel.js",
    ):
        p = PROJECT_ROOT / "static" / fn
        assert p.exists() and p.stat().st_size > 100, f"{fn} missing or empty"


# ---------------------------------------------------------------------------
# JS module API surface — string-grep validation (jsdom test would need the
# whole three.js stack; we settle for checking the public exports exist
# in the source).
# ---------------------------------------------------------------------------

def test_edit_panel_exposes_public_api():
    src = (PROJECT_ROOT / "static" / "edit_panel.js").read_text(encoding="utf-8")
    for sym in (
        "window.psoEditPanel",
        "setEditMode",
        "selectAll",
        "clearSelection",
        "saveEdits",
        "selectionCount",
    ):
        assert sym in src, f"edit_panel.js missing {sym}"


def test_transform_gizmo_exposes_public_api():
    src = (PROJECT_ROOT / "static" / "transform_gizmo.js").read_text(encoding="utf-8")
    for sym in (
        "window.psoTransformGizmo",
        "attach",
        "detach",
        "setMode",
        "isAttached",
        "isDragging",
        "onCommit",
    ):
        assert sym in src, f"transform_gizmo.js missing {sym}"


def test_skeleton_panel_exposes_public_api():
    src = (PROJECT_ROOT / "static" / "skeleton_panel.js").read_text(encoding="utf-8")
    for sym in (
        "window.psoSkeletonPanel",
        "selectBone",
        "refreshFromViewport",
        "psoTexturePanelRegisterTab",
    ):
        assert sym in src, f"skeleton_panel.js missing {sym}"


def test_uv_panel_exposes_public_api():
    src = (PROJECT_ROOT / "static" / "uv_panel.js").read_text(encoding="utf-8")
    for sym in (
        "window.psoUvPanel",
        "refreshFromViewport",
        "selectVertex",
        "selectSubmesh",
    ):
        assert sym in src, f"uv_panel.js missing {sym}"


def test_edit_panel_pushes_to_undo_bus():
    """Edit ops must push to the global undo bus so Ctrl+Z works."""
    src = (PROJECT_ROOT / "static" / "edit_panel.js").read_text(encoding="utf-8")
    assert "psoUndoBus" in src, "edit_panel must push to global undo bus"
    assert 'panelId: "edit"' in src, "edit_panel pushes must tag panelId='edit'"


def test_edit_panel_yields_lmb_to_orbit_when_off():
    """The model viewer must yield LMB to edit_panel only when edit mode is ON."""
    src = (PROJECT_ROOT / "static" / "model_viewer.js").read_text(encoding="utf-8")
    assert "__psoEditModeActive" in src, (
        "model_viewer.js must check __psoEditModeActive in pointerdown"
    )


# ---------------------------------------------------------------------------
# Hotkey wiring — Tab toggles edit mode; G/R/S set transform mode.
# ---------------------------------------------------------------------------

def test_edit_panel_binds_blender_hotkeys():
    src = (PROJECT_ROOT / "static" / "edit_panel.js").read_text(encoding="utf-8")
    for combo in ("Tab", '"g"', '"r"', '"s"', '"v"', '"a"', "Ctrl+s"):
        assert combo in src, f"edit_panel.js missing hotkey {combo!r}"


# ---------------------------------------------------------------------------
# Cache directory exists at server startup
# ---------------------------------------------------------------------------

def test_protools_edits_dir_exists():
    import server
    assert server.PROTOOLS_EDITS_DIR.exists()
    assert server.PROTOOLS_EDITS_DIR.is_dir()


# ---------------------------------------------------------------------------
# Plan file exists and lists the shipped P0 features
# ---------------------------------------------------------------------------

def test_plan_file_documents_p0():
    plan = (PROJECT_ROOT / "_reports" / "protools_plan.md").read_text(encoding="utf-8")
    for keyword in ("P0", "vertex selection", "transform gizmo", "UV viewer"):
        assert keyword.lower() in plan.lower(), (
            f"plan file missing {keyword!r}"
        )
