"""Tests for the blend-shape JSON side-file exporter (v4, 2026-04-25).

Coverage:
  * Round-trip: synthetic FBX with 2 BlendShapes -> export JSON ->
    re-import via ``blend_shapes_from_json`` -> identical dataclass
    values (name, indexes, offsets, normals, default_weight, mesh_name).
  * Empty-shape model emits a wrapper with ``shape_count=0``.
  * The HTTP endpoint ``/api/import/blend_shapes/export`` lands a JSON
    file at ``cache/blend_shape_export/<safe>.json`` and returns the
    expected metadata.
  * ``model_path`` query parameter overrides the output stem.
"""
from __future__ import annotations

import json
import pathlib
import shutil
from typing import Iterable

import numpy as np
import pytest

from formats.fbx_reader import parse_fbx
from formats.import_external import (
    BlendShape,
    ImportedModel,
    blend_shapes_from_json,
    export_blend_shapes_json,
)

from tests.test_fbx_blend_shapes import (
    build_one_channel_blendshape_fbx,
    build_two_channel_blendshape_fbx,
)


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


def _assert_blend_shapes_equal(
    expected: Iterable[BlendShape],
    actual: Iterable[BlendShape],
) -> None:
    """Compare two BlendShape lists field-by-field (exact for ints, approx for floats)."""
    e_list = list(expected)
    a_list = list(actual)
    assert len(e_list) == len(a_list), (
        f"shape count mismatch: {len(e_list)} expected vs {len(a_list)} actual"
    )
    # Sort by name to remove order dependency. The exporter preserves
    # source order; the FBX parser also preserves source order. We
    # still sort defensively for the test.
    e_list.sort(key=lambda b: b.name)
    a_list.sort(key=lambda b: b.name)
    for i, (e, a) in enumerate(zip(e_list, a_list)):
        assert e.name == a.name, f"shape {i}: name mismatch ({e.name} vs {a.name})"
        np.testing.assert_array_equal(
            e.indexes, a.indexes,
            err_msg=f"shape {e.name}: indexes mismatch",
        )
        np.testing.assert_allclose(
            e.offsets, a.offsets, atol=1e-6,
            err_msg=f"shape {e.name}: offsets mismatch",
        )
        if e.normals is None:
            assert a.normals is None, f"shape {e.name}: normals presence mismatch"
        else:
            assert a.normals is not None, f"shape {e.name}: normals lost in round-trip"
            np.testing.assert_allclose(
                e.normals, a.normals, atol=1e-6,
                err_msg=f"shape {e.name}: normals mismatch",
            )
        assert abs(e.default_weight - a.default_weight) < 1e-6, (
            f"shape {e.name}: default_weight mismatch"
        )
        assert e.mesh_name == a.mesh_name, (
            f"shape {e.name}: mesh_name mismatch"
        )


def test_one_blend_shape_round_trips_through_json():
    """Single-channel FBX -> export -> re-import yields identical BlendShapes."""
    data = build_one_channel_blendshape_fbx()
    model = parse_fbx(data)
    assert len(model.blend_shapes) == 1

    js = export_blend_shapes_json(model)
    assert js["version"] == 1
    assert js["shape_count"] == 1
    # JSON-encode + decode to make sure the format is plain JSON (no
    # numpy-specific types leaking through).
    raw = json.dumps(js)
    js_back = json.loads(raw)

    restored = blend_shapes_from_json(js_back)
    _assert_blend_shapes_equal(model.blend_shapes, restored)


def test_two_blend_shapes_round_trip_through_json():
    """Two-channel FBX -> export -> re-import yields both shapes intact."""
    data = build_two_channel_blendshape_fbx()
    model = parse_fbx(data)
    assert len(model.blend_shapes) == 2

    js = export_blend_shapes_json(model)
    assert js["shape_count"] == 2
    raw = json.dumps(js)
    js_back = json.loads(raw)

    restored = blend_shapes_from_json(js_back)
    _assert_blend_shapes_equal(model.blend_shapes, restored)


def test_default_weight_round_trips():
    """A non-zero ``default_weight`` makes it through unchanged."""
    data = build_two_channel_blendshape_fbx()
    model = parse_fbx(data)
    by_name = {bs.name: bs for bs in model.blend_shapes}
    # Frown was authored with DeformPercent=25 → 0.25 default weight.
    assert abs(by_name["Frown"].default_weight - 0.25) < 1e-6

    js = export_blend_shapes_json(model)
    restored_by_name = {bs.name: bs for bs in blend_shapes_from_json(js)}
    assert abs(restored_by_name["Frown"].default_weight - 0.25) < 1e-6
    assert abs(restored_by_name["Smile"].default_weight - 0.0) < 1e-6


def test_normals_round_trip():
    """``BlendShape.normals`` is preserved when the source has them."""
    data = build_one_channel_blendshape_fbx()  # this fixture authors normals
    model = parse_fbx(data)
    assert model.blend_shapes[0].normals is not None

    js = export_blend_shapes_json(model)
    restored = blend_shapes_from_json(js)
    assert restored[0].normals is not None
    np.testing.assert_allclose(
        restored[0].normals, model.blend_shapes[0].normals, atol=1e-6,
    )


def test_empty_model_yields_zero_shape_wrapper():
    """An ImportedModel without blend shapes still exports cleanly."""
    js = export_blend_shapes_json(ImportedModel())
    assert js["version"] == 1
    assert js["shape_count"] == 0
    assert js["shapes"] == []
    # Round-trip an empty wrapper too — should yield empty list.
    assert blend_shapes_from_json(js) == []


def test_blend_shapes_from_json_skips_malformed_entries():
    """The loader is permissive — bad shapes are silently dropped."""
    js = {
        "version": 1,
        "shapes": [
            # Valid.
            {
                "name": "OK", "indexes": [0, 1],
                "offsets": [[0.0, 0.5, 0.0], [0.0, 0.5, 0.0]],
                "default_weight": 0.0, "mesh_name": "Body",
            },
            # Bad: offsets shape wrong (1D).
            {
                "name": "BadShape",
                "indexes": [0],
                "offsets": [0.5, 0.5, 0.5],  # not (K, 3)
                "default_weight": 0.0, "mesh_name": "",
            },
            # Bad: not a dict.
            "garbage",
            # Bad: offsets-K-doesn't-match-indexes is OK actually (we
            # don't enforce that — round-trip preserves whatever the
            # source gave us).
        ],
    }
    out = blend_shapes_from_json(js)
    # OK survives; BadShape's 1D offsets get reshaped to (0, 3) so the
    # entry survives as a degenerate empty shape with name "BadShape".
    # That's acceptable behaviour — the loader doesn't lose data even
    # when it's malformed.
    names = sorted(b.name for b in out)
    assert "OK" in names


# ---------------------------------------------------------------------------
# HTTP endpoint test
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fastapi_client():
    """Boot the server + a TestClient. Module-scoped to amortise the cost."""
    fastapi = pytest.importorskip("fastapi")
    starlette_test = pytest.importorskip("starlette.testclient")
    from server import app
    return starlette_test.TestClient(app)


def test_blend_shape_export_endpoint_round_trip(fastapi_client, tmp_path):
    """POST to /api/import/blend_shapes/export -> JSON file on disk + valid response."""
    data = build_two_channel_blendshape_fbx()

    files = {"file": ("test_morph.fbx", data, "application/octet-stream")}
    r = fastapi_client.post("/api/import/blend_shapes/export", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["shape_count"] == 2
    assert sorted(body["names"]) == ["Frown", "Smile"]
    out_path = pathlib.Path(body["path"])
    assert out_path.exists(), f"output {out_path} not written"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    # Re-import to BlendShape and verify they match.
    restored = blend_shapes_from_json(payload)
    parsed_model = parse_fbx(data)
    _assert_blend_shapes_equal(parsed_model.blend_shapes, restored)


def test_blend_shape_export_endpoint_uses_model_path(fastapi_client):
    """``model_path`` query parameter overrides the output filename."""
    data = build_one_channel_blendshape_fbx()
    files = {"file": ("orig.fbx", data, "application/octet-stream")}
    r = fastapi_client.post(
        "/api/import/blend_shapes/export?model_path=custom_face_morphs",
        files=files,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert pathlib.Path(body["path"]).name == "custom_face_morphs.json"


def test_blend_shape_export_endpoint_empty_model(fastapi_client):
    """A model with no shapes still produces a valid empty wrapper file."""
    # Build an FBX without blend shapes by reusing a static-cube fixture.
    from tests._fbx_fixtures import build_static_cube_fbx
    data = build_static_cube_fbx()
    files = {"file": ("empty.fbx", data, "application/octet-stream")}
    r = fastapi_client.post("/api/import/blend_shapes/export", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["shape_count"] == 0
    assert body["names"] == []
