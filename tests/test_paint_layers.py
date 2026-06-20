"""Tests for the v5 layer-stack additions in formats/paint.py + server.

Covers:
  - manifest validation (default + sanitization)
  - blend modes (normal / multiply / screen / overlay) at both edge
    cases (transparent + opaque) and a mid-alpha case
  - composite_layers stacks bottom-to-top with mask + opacity
  - clone_stamp copies pixels at the right offset, falloff included
  - gradient_fill linear / radial / angular
  - apply_mask_to_layer bakes alpha multiplier into layer alpha
  - /api/paint/layer/save round-trip via TestClient
  - /api/paint/load returns the saved stack
  - /api/paint/manifest reorder + delete
  - flat-PNG -> layer-dir migration
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pytest

from PIL import Image

from formats import paint as _paint


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------
def test_default_manifest_shape():
    m = _paint.make_default_manifest(model_path="x.bml", inner="x.xvm", width=64, height=64)
    assert m["version"] == 1
    assert m["model_path"] == "x.bml"
    assert m["inner"] == "x.xvm"
    assert m["width"] == 64 and m["height"] == 64
    assert m["active"] == 0
    assert len(m["layers"]) == 1
    L0 = m["layers"][0]
    assert L0["idx"] == 0
    assert L0["visible"] is True
    assert L0["opacity"] == 1.0
    assert L0["blend_mode"] == "normal"
    assert L0["locked"] is False
    assert L0["has_mask"] is False


def test_validate_manifest_rejects_bad_blend_mode():
    bad = _paint.make_default_manifest(model_path="x", inner="y", width=4, height=4)
    bad["layers"][0]["blend_mode"] = "exotic"
    with pytest.raises(ValueError):
        _paint.validate_manifest(bad)


def test_validate_manifest_clamps_opacity():
    m = _paint.make_default_manifest(model_path="x", inner="y", width=4, height=4)
    m["layers"][0]["opacity"] = 2.5  # over 1.0
    norm = _paint.validate_manifest(m)
    assert norm["layers"][0]["opacity"] == 1.0


def test_validate_manifest_rejects_duplicate_idx():
    m = {
        "version": 1, "model_path": "x", "inner": "y",
        "width": 4, "height": 4, "active": 0,
        "layers": [
            {"idx": 0, "name": "a", "visible": True, "opacity": 1.0,
             "blend_mode": "normal", "locked": False, "has_mask": False},
            {"idx": 0, "name": "b", "visible": True, "opacity": 1.0,
             "blend_mode": "normal", "locked": False, "has_mask": False},
        ],
    }
    with pytest.raises(ValueError):
        _paint.validate_manifest(m)


def test_validate_manifest_renumbers_active():
    """If `active` points outside the layer list, fall back to the first."""
    m = _paint.make_default_manifest(model_path="x", inner="y", width=4, height=4)
    m["active"] = 99
    norm = _paint.validate_manifest(m)
    assert norm["active"] == 0


# ---------------------------------------------------------------------------
# Blend modes
# ---------------------------------------------------------------------------
def _solid_layer(w: int, h: int, rgba: tuple[int, int, int, int]) -> bytes:
    buf = bytearray()
    for _ in range(w * h):
        buf.extend(rgba)
    return bytes(buf)


def _meta(idx: int, *, blend="normal", opacity=1.0, visible=True, has_mask=False):
    return {
        "idx": idx,
        "name": f"L{idx}",
        "visible": visible,
        "opacity": opacity,
        "blend_mode": blend,
        "locked": False,
        "has_mask": has_mask,
    }


def test_composite_normal_topwin():
    """Topmost opaque layer wins on normal blend."""
    w = h = 4
    layer0 = _solid_layer(w, h, (0, 0, 255, 255))   # blue
    layer1 = _solid_layer(w, h, (255, 0, 0, 255))   # red on top
    out = _paint.composite_layers(
        [(layer0, _meta(0), None), (layer1, _meta(1), None)],
        w, h,
    )
    assert out[0:4] == bytearray([255, 0, 0, 255])


def test_composite_normal_half_alpha():
    """50% alpha top should average with the bottom in normal blend."""
    w = h = 2
    layer0 = _solid_layer(w, h, (0, 255, 0, 255))         # opaque green
    layer1 = _solid_layer(w, h, (255, 0, 0, 128))         # red @ 50%
    out = _paint.composite_layers(
        [(layer0, _meta(0), None), (layer1, _meta(1), None)],
        w, h,
    )
    # Porter-Duff over: out_r ~= sr*sa + dr*(1-sa) ~= 255*0.5 + 0*0.5 ~= 127
    # out_g ~= 0*0.5 + 255*0.5 ~= 127
    r, g, b, a = out[0], out[1], out[2], out[3]
    assert 120 <= r <= 135
    assert 120 <= g <= 135
    assert b == 0
    assert a == 255


def test_composite_multiply_clamps_to_zero():
    w = h = 2
    layer0 = _solid_layer(w, h, (200, 200, 200, 255))
    layer1 = _solid_layer(w, h, (0, 0, 0, 255))           # multiply by zero -> zero
    out = _paint.composite_layers(
        [(layer0, _meta(0), None), (layer1, _meta(1, blend="multiply"), None)],
        w, h,
    )
    assert out[0] == 0 and out[1] == 0 and out[2] == 0


def test_composite_screen_lightens():
    w = h = 2
    layer0 = _solid_layer(w, h, (100, 100, 100, 255))
    layer1 = _solid_layer(w, h, (200, 200, 200, 255))
    out = _paint.composite_layers(
        [(layer0, _meta(0), None), (layer1, _meta(1, blend="screen"), None)],
        w, h,
    )
    # 1 - (1-100/255)*(1-200/255) = 1 - 0.608*0.215 = 0.869 -> ~221
    assert 215 <= out[0] <= 225


def test_composite_overlay_branches():
    w = h = 4
    # Bottom row 0 dim, row 2 bright -> overlay should multiply on row 0,
    # screen on row 2.
    bottom = bytearray()
    for y in range(h):
        for _ in range(w):
            v = 50 if y < 2 else 200
            bottom.extend((v, v, v, 255))
    top = _solid_layer(w, h, (128, 128, 128, 255))
    out = _paint.composite_layers(
        [(bytes(bottom), _meta(0), None), (top, _meta(1, blend="overlay"), None)],
        w, h,
    )
    # Bottom row (dim): overlay = 2*S*D = 2 * 50/255 * 128/255 ~= 0.197 -> ~50
    dim_idx = (0 * w + 0) * 4
    assert out[dim_idx + 0] < 100
    # Top row (bright): overlay = 1 - 2*(1-S)*(1-D) = 1 - 2*0.498*0.215 = 0.786 -> ~200
    bright_idx = (2 * w + 0) * 4
    assert out[bright_idx + 0] > 150


def test_composite_skips_invisible_layers():
    w = h = 2
    layer0 = _solid_layer(w, h, (10, 20, 30, 255))
    layer1 = _solid_layer(w, h, (200, 200, 200, 255))
    out = _paint.composite_layers(
        [(layer0, _meta(0), None), (layer1, _meta(1, visible=False), None)],
        w, h,
    )
    # Layer 1 hidden -> see layer 0.
    assert out[0:4] == bytearray([10, 20, 30, 255])


def test_composite_with_mask_hides_pixels():
    w = h = 4
    layer0 = _solid_layer(w, h, (0, 0, 0, 255))            # opaque black
    layer1 = _solid_layer(w, h, (255, 0, 0, 255))          # opaque red on top
    # Mask: left half white (visible), right half black (hidden)
    mask = bytearray()
    for y in range(h):
        for x in range(w):
            v = 255 if x < w // 2 else 0
            mask.extend((v, v, v, 255))
    out = _paint.composite_layers(
        [(layer0, _meta(0), None), (layer1, _meta(1, has_mask=True), bytes(mask))],
        w, h,
    )
    # Left half visible: red
    assert out[(0 * w + 0) * 4: (0 * w + 0) * 4 + 4] == bytearray([255, 0, 0, 255])
    # Right half hidden: layer0 (black) shows through.
    i = (0 * w + (w - 1)) * 4
    assert out[i: i + 4] == bytearray([0, 0, 0, 255])


def test_composite_layer_opacity_is_applied():
    w = h = 2
    layer0 = _solid_layer(w, h, (0, 255, 0, 255))
    layer1 = _solid_layer(w, h, (255, 0, 0, 255))           # red
    out = _paint.composite_layers(
        [(layer0, _meta(0), None), (layer1, _meta(1, opacity=0.0), None)],
        w, h,
    )
    # Top fully transparent via opacity=0 -> see green.
    assert out[0:4] == bytearray([0, 255, 0, 255])


# ---------------------------------------------------------------------------
# clone_stamp
# ---------------------------------------------------------------------------
def test_clone_copies_with_offset():
    w = h = 16
    src = bytearray()
    for y in range(h):
        for x in range(w):
            v = 0
            if x < 4 and y < 4:
                v = 255
            src.extend((v, 0, 0, 255 if v else 0))
    dst = bytearray(w * h * 4)
    # Source is the (0..3, 0..3) red square; clone to (10, 10, layerIdx).
    # offset_x=10-0=10, offset_y=10-0=10 (so dst pixel = src pixel + offset).
    rect = _paint.clone_stamp(
        dst, w, h, 10, 10, 4,
        src_buf=bytes(src), src_w=w, src_h=h,
        src_offset_x=10, src_offset_y=10,
        opacity=1.0, hardness=1.0,
    )
    assert rect[0] <= 6 and rect[2] >= 14
    # Pixel (10, 10) should be opaque red (we copied from (0, 0)).
    i = (10 * w + 10) * 4
    assert dst[i + 0] == 255 and dst[i + 3] == 255


def test_clone_falloff_softens_edges():
    w = h = 24
    src = bytearray()
    for _ in range(w * h):
        src.extend((255, 255, 255, 255))  # opaque white everywhere
    dst = bytearray(w * h * 4)
    _paint.clone_stamp(
        dst, w, h, 12, 12, 8,
        src_buf=bytes(src), src_w=w, src_h=h,
        src_offset_x=0, src_offset_y=0,
        opacity=1.0, hardness=0.0,
    )
    centre = (12 * w + 12) * 4
    rim = (4 * w + 12) * 4   # 8 px above centre, exactly at the rim
    assert dst[centre + 3] > dst[rim + 3]
    assert dst[rim + 3] > 0  # rim still partial — soft brush


def test_clone_zero_radius_noop():
    w = h = 8
    src = bytearray(_solid_layer(w, h, (255, 0, 0, 255)))
    dst = bytearray(w * h * 4)
    rect = _paint.clone_stamp(
        dst, w, h, 4, 4, 0,
        src_buf=bytes(src), src_w=w, src_h=h,
        src_offset_x=0, src_offset_y=0,
    )
    assert rect == (0, 0, 0, 0)
    assert all(v == 0 for v in dst)


# ---------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------
def test_gradient_linear_horizontal():
    w = h = 16
    dst = bytearray(w * h * 4)
    _paint.gradient_fill(
        dst, w, h,
        x0=0, y0=h // 2, x1=w - 1, y1=h // 2,
        stops=[(0.0, (255, 0, 0, 255)), (1.0, (0, 0, 255, 255))],
        kind="linear",
    )
    # Left edge should be red.
    i = (0 * w + 0) * 4
    assert dst[i + 0] >= 200 and dst[i + 2] < 50
    # Right edge should be blue.
    j = (0 * w + (w - 1)) * 4
    assert dst[j + 2] >= 200 and dst[j + 0] < 50


def test_gradient_radial_centre_to_edge():
    w = h = 32
    dst = bytearray(w * h * 4)
    _paint.gradient_fill(
        dst, w, h,
        x0=w / 2, y0=h / 2, x1=w / 2 + (w / 2), y1=h / 2,
        stops=[(0.0, (0, 0, 0, 255)), (1.0, (255, 255, 255, 255))],
        kind="radial",
    )
    centre = ((h // 2) * w + (w // 2)) * 4
    corner = (0 * w + 0) * 4
    # Centre dark, corner light.
    assert dst[centre + 0] < 50
    assert dst[corner + 0] > 200


def test_gradient_angular_full_sweep():
    w = h = 16
    dst = bytearray(w * h * 4)
    _paint.gradient_fill(
        dst, w, h,
        x0=w / 2, y0=h / 2, x1=w - 1, y1=h / 2,
        stops=[(0.0, (255, 0, 0, 255)), (0.5, (0, 255, 0, 255)), (1.0, (255, 0, 0, 255))],
        kind="angular",
    )
    # The gradient sweeps; we just verify it didn't blow up + filled the buffer.
    assert any(v != 0 for v in dst)


def test_gradient_rejects_empty_stops():
    w = h = 4
    dst = bytearray(w * h * 4)
    with pytest.raises(ValueError):
        _paint.gradient_fill(dst, w, h, x0=0, y0=0, x1=1, y1=0, stops=[])


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------
def test_make_blank_mask_white_default():
    mask = _paint.make_blank_mask(8, 8)
    assert len(mask) == 8 * 8 * 4
    # First pixel: white, alpha 255.
    assert mask[0] == 255 and mask[1] == 255 and mask[2] == 255 and mask[3] == 255


def test_apply_mask_to_layer_kills_alpha_where_black():
    w = h = 4
    rgba = bytearray(_solid_layer(w, h, (255, 128, 64, 255)))
    # Build a half-and-half mask.
    mask = bytearray()
    for y in range(h):
        for x in range(w):
            v = 255 if x < w // 2 else 0
            mask.extend((v, v, v, 255))
    _paint.apply_mask_to_layer(rgba, bytes(mask), w, h)
    # Left half: alpha unchanged.
    li = (0 * w + 0) * 4
    assert rgba[li + 3] == 255
    # Right half: alpha 0.
    ri = (0 * w + (w - 1)) * 4
    assert rgba[ri + 3] == 0
    # RGB unchanged everywhere.
    assert rgba[ri + 0] == 255 and rgba[ri + 1] == 128 and rgba[ri + 2] == 64


# ---------------------------------------------------------------------------
# safe_painted_dirname
# ---------------------------------------------------------------------------
def test_safe_painted_dirname_drops_extension():
    name = _paint.safe_painted_dirname("foo.bml", "bar.xvm")
    assert not name.endswith(".png")
    # And matches safe_painted_basename minus ".png".
    assert _paint.safe_painted_basename("foo.bml", "bar.xvm") == name + ".png"


# ---------------------------------------------------------------------------
# API round-trip
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    import server
    return TestClient(server.app)


def _png_b64(w: int, h: int, color: tuple[int, int, int, int]) -> str:
    raw = bytearray(color * (w * h))
    enc = _paint.rgba_to_png_bytes(bytes(raw), w, h)
    return base64.b64encode(enc).decode("ascii")


def test_api_layer_save_creates_dir_and_manifest(client, tmp_path, monkeypatch):
    """First /api/paint/layer/save creates dir + manifest."""
    import server
    fresh = tmp_path / "painted"
    fresh.mkdir()
    monkeypatch.setattr(server, "PAINTED_TEX_DIR", fresh)
    monkeypatch.setattr(server, "PAINTED_LAYER_ROOT", fresh)

    body = {
        "model_path": "fake.bml",
        "inner": "fake.nj.xvm",
        "layer_idx": 0,
        "png_b64": _png_b64(16, 16, (200, 50, 100, 255)),
        "name": "Background",
        "blend_mode": "normal",
    }
    r = client.post("/api/paint/layer/save", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    assert data["manifest"]["layers"][0]["idx"] == 0
    # Layer dir should exist.
    layer_dir = fresh / _paint.safe_painted_dirname("fake.bml", "fake.nj.xvm")
    assert layer_dir.exists()
    assert (layer_dir / "0.png").exists()
    assert (layer_dir / "manifest.json").exists()
    # Composite was written.
    flat = fresh / _paint.safe_painted_basename("fake.bml", "fake.nj.xvm")
    assert flat.exists()


def test_api_layer_save_rejects_oversize(client, tmp_path, monkeypatch):
    """Layer dimensions must match an existing manifest."""
    import server
    fresh = tmp_path / "painted"
    fresh.mkdir()
    monkeypatch.setattr(server, "PAINTED_TEX_DIR", fresh)
    monkeypatch.setattr(server, "PAINTED_LAYER_ROOT", fresh)

    # Save layer 0 at 16x16.
    client.post("/api/paint/layer/save", json={
        "model_path": "fake.bml",
        "inner": "fake.nj.xvm",
        "layer_idx": 0,
        "png_b64": _png_b64(16, 16, (200, 50, 100, 255)),
    })
    # Now try to add layer 1 at 32x32 — should 400.
    r = client.post("/api/paint/layer/save", json={
        "model_path": "fake.bml",
        "inner": "fake.nj.xvm",
        "layer_idx": 1,
        "png_b64": _png_b64(32, 32, (0, 0, 0, 255)),
    })
    assert r.status_code == 400


def test_api_paint_load_returns_stack(client, tmp_path, monkeypatch):
    import server
    fresh = tmp_path / "painted"
    fresh.mkdir()
    monkeypatch.setattr(server, "PAINTED_TEX_DIR", fresh)
    monkeypatch.setattr(server, "PAINTED_LAYER_ROOT", fresh)

    client.post("/api/paint/layer/save", json={
        "model_path": "x.bml", "inner": "x.xvm",
        "layer_idx": 0, "png_b64": _png_b64(8, 8, (10, 20, 30, 255)),
    })
    client.post("/api/paint/layer/save", json={
        "model_path": "x.bml", "inner": "x.xvm",
        "layer_idx": 1, "png_b64": _png_b64(8, 8, (40, 50, 60, 200)),
        "blend_mode": "multiply", "opacity": 0.7,
    })
    r = client.get("/api/paint/load", params={"model_path": "x.bml", "inner": "x.xvm"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    m = data["manifest"]
    assert m is not None
    assert len(m["layers"]) == 2
    L1 = next(L for L in m["layers"] if L["idx"] == 1)
    assert L1["blend_mode"] == "multiply"
    assert abs(L1["opacity"] - 0.7) < 1e-6
    assert len(data["layers"]) == 2
    # Both layers' PNGs are non-empty base64.
    for L in data["layers"]:
        assert L["png_b64"]


def test_api_manifest_reorder_persists(client, tmp_path, monkeypatch):
    import server
    fresh = tmp_path / "painted"
    fresh.mkdir()
    monkeypatch.setattr(server, "PAINTED_TEX_DIR", fresh)
    monkeypatch.setattr(server, "PAINTED_LAYER_ROOT", fresh)

    # Save two layers.
    client.post("/api/paint/layer/save", json={
        "model_path": "y.bml", "inner": "y.xvm",
        "layer_idx": 0, "png_b64": _png_b64(8, 8, (255, 0, 0, 255)),
    })
    client.post("/api/paint/layer/save", json={
        "model_path": "y.bml", "inner": "y.xvm",
        "layer_idx": 1, "png_b64": _png_b64(8, 8, (0, 255, 0, 255)),
    })
    # Push manifest with reversed order.
    new_mf = _paint.make_default_manifest(model_path="y.bml", inner="y.xvm", width=8, height=8)
    new_mf["layers"] = [
        {"idx": 1, "name": "B", "visible": True, "opacity": 1.0,
         "blend_mode": "normal", "locked": False, "has_mask": False},
        {"idx": 0, "name": "A", "visible": True, "opacity": 1.0,
         "blend_mode": "normal", "locked": False, "has_mask": False},
    ]
    new_mf["active"] = 0
    r = client.post("/api/paint/manifest", json={
        "model_path": "y.bml", "inner": "y.xvm", "manifest": new_mf,
    })
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["ok"] is True
    # First layer in the new manifest is now layer 1 (green).
    assert out["manifest"]["layers"][0]["idx"] == 1
    # Composite should be GREEN now (layer 1 was on top after reorder; but
    # we reversed -> bottom = layer 1 (green), top = layer 0 (red), so
    # we still see the top opaque red). Just verify the reordered order
    # round-trips correctly via /api/paint/load.
    r2 = client.get("/api/paint/load", params={"model_path": "y.bml", "inner": "y.xvm"})
    m = r2.json()["manifest"]
    assert [L["idx"] for L in m["layers"]] == [1, 0]


def test_api_layer_delete_drops_manifest_entry(client, tmp_path, monkeypatch):
    import server
    fresh = tmp_path / "painted"
    fresh.mkdir()
    monkeypatch.setattr(server, "PAINTED_TEX_DIR", fresh)
    monkeypatch.setattr(server, "PAINTED_LAYER_ROOT", fresh)

    for i in range(3):
        client.post("/api/paint/layer/save", json={
            "model_path": "z.bml", "inner": "z.xvm",
            "layer_idx": i, "png_b64": _png_b64(4, 4, (i * 50, 0, 0, 255)),
        })
    # Delete layer 1.
    r = client.post("/api/paint/layer/delete", json={
        "model_path": "z.bml", "inner": "z.xvm", "layer_idx": 1, "is_mask": False,
    })
    assert r.status_code == 200
    out = r.json()
    assert {L["idx"] for L in out["manifest"]["layers"]} == {0, 2}
    # On-disk PNG is gone.
    layer_dir = fresh / _paint.safe_painted_dirname("z.bml", "z.xvm")
    assert not (layer_dir / "1.png").exists()


def test_api_layer_delete_refuses_only_layer(client, tmp_path, monkeypatch):
    import server
    fresh = tmp_path / "painted"
    fresh.mkdir()
    monkeypatch.setattr(server, "PAINTED_TEX_DIR", fresh)
    monkeypatch.setattr(server, "PAINTED_LAYER_ROOT", fresh)

    client.post("/api/paint/layer/save", json={
        "model_path": "w.bml", "inner": "w.xvm",
        "layer_idx": 0, "png_b64": _png_b64(4, 4, (1, 2, 3, 255)),
    })
    r = client.post("/api/paint/layer/delete", json={
        "model_path": "w.bml", "inner": "w.xvm", "layer_idx": 0, "is_mask": False,
    })
    assert r.status_code == 400


def test_api_mask_save_and_delete(client, tmp_path, monkeypatch):
    import server
    fresh = tmp_path / "painted"
    fresh.mkdir()
    monkeypatch.setattr(server, "PAINTED_TEX_DIR", fresh)
    monkeypatch.setattr(server, "PAINTED_LAYER_ROOT", fresh)

    client.post("/api/paint/layer/save", json={
        "model_path": "mk.bml", "inner": "mk.xvm",
        "layer_idx": 0, "png_b64": _png_b64(8, 8, (0, 0, 0, 255)),
    })
    # Save a mask for layer 0.
    r = client.post("/api/paint/layer/save", json={
        "model_path": "mk.bml", "inner": "mk.xvm",
        "layer_idx": 0, "png_b64": _png_b64(8, 8, (255, 255, 255, 255)),
        "is_mask": True,
    })
    assert r.status_code == 200, r.text
    layer_dir = fresh / _paint.safe_painted_dirname("mk.bml", "mk.xvm")
    assert (layer_dir / "0_mask.png").exists()
    # Layer should report has_mask=True now.
    m = r.json()["manifest"]
    assert m["layers"][0]["has_mask"] is True
    # Now delete just the mask.
    r2 = client.post("/api/paint/layer/delete", json={
        "model_path": "mk.bml", "inner": "mk.xvm",
        "layer_idx": 0, "is_mask": True,
    })
    assert r2.status_code == 200
    assert not (layer_dir / "0_mask.png").exists()
    # And the layer survives without the mask flag.
    assert r2.json()["manifest"]["layers"][0]["has_mask"] is False


def test_api_paint_load_migrates_legacy_flat_png(client, tmp_path, monkeypatch):
    """A flat-PNG cache entry should auto-convert to a layer stack on /load."""
    import server
    fresh = tmp_path / "painted"
    fresh.mkdir()
    monkeypatch.setattr(server, "PAINTED_TEX_DIR", fresh)
    monkeypatch.setattr(server, "PAINTED_LAYER_ROOT", fresh)

    # Drop a flat <safe>.png pretending it was written by the v4 endpoint.
    flat_name = _paint.safe_painted_basename("legacy.bml", "legacy.xvm")
    raw = bytearray()
    for _ in range(16 * 16):
        raw.extend((10, 20, 30, 255))
    flat_path = fresh / flat_name
    flat_path.write_bytes(_paint.rgba_to_png_bytes(bytes(raw), 16, 16))

    # /api/paint/load should migrate.
    r = client.get("/api/paint/load", params={
        "model_path": "legacy.bml", "inner": "legacy.xvm",
    })
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["manifest"] is not None
    assert len(data["manifest"]["layers"]) == 1
    # Layer dir now exists.
    layer_dir = fresh / _paint.safe_painted_dirname("legacy.bml", "legacy.xvm")
    assert layer_dir.exists()
    assert (layer_dir / "0.png").exists()
    assert (layer_dir / "manifest.json").exists()


def test_api_paint_active_reports_layer_count(client, tmp_path, monkeypatch):
    import server
    fresh = tmp_path / "painted"
    fresh.mkdir()
    monkeypatch.setattr(server, "PAINTED_TEX_DIR", fresh)
    monkeypatch.setattr(server, "PAINTED_LAYER_ROOT", fresh)

    for i in range(3):
        client.post("/api/paint/layer/save", json={
            "model_path": "lc.bml", "inner": "lc.xvm",
            "layer_idx": i, "png_b64": _png_b64(4, 4, (i, 0, 0, 255)),
        })
    r = client.get("/api/paint/active")
    assert r.status_code == 200, r.text
    data = r.json()
    entry = next(e for e in data["painted"] if e["model_path"] == "lc.bml")
    assert entry["layer_count"] == 3
