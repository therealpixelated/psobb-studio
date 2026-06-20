"""Tests for the Texture Paint MVP.

Covers:
  - UV unwrap math (uv_to_pixel / pixel_to_uv round-trip)
  - Brush stamp shape + Gaussian falloff
  - Alpha-over compositing (clipped + un-clipped)
  - Flood fill correctness on a striped buffer
  - Smear stamp direction
  - safe_painted_basename round-trip + path-injection guard
  - /api/paint/save endpoint round-trip
  - /api/paint/active listing
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest

from PIL import Image

from formats import paint as _paint
from formats.paint import (
    alpha_over,
    flood_fill,
    pixel_to_uv,
    rgba_to_png_bytes,
    safe_painted_basename,
    smear_stamp,
    stamp_circle,
    uv_to_pixel,
)


# ---------------------------------------------------------------------------
# UV math
# ---------------------------------------------------------------------------
def test_uv_origin_corner_round_trip():
    """V=0 maps to bottom row (PIL height-1); V=1 maps to top row 0."""
    # 256x256 texture: u=0,v=0 -> (0, 255).
    assert uv_to_pixel(0.0, 0.0, 256, 256) == (0, 255)
    # u=1,v=1 -> (255, 0).
    assert uv_to_pixel(1.0, 1.0, 256, 256) == (255, 0)
    # u=0.5,v=0.5 -> centre.
    px, py = uv_to_pixel(0.5, 0.5, 256, 256)
    assert px in (127, 128) and py in (127, 128)


def test_uv_clamps_out_of_range():
    # Negative or >1 UVs are clamped, not wrapped.
    assert uv_to_pixel(-0.2, 1.5, 64, 64) == (0, 0)
    assert uv_to_pixel(1.7, -0.1, 64, 64) == (63, 63)


def test_uv_pixel_round_trip():
    for w, h in [(64, 64), (256, 128), (32, 100)]:
        for px in (0, w // 2, w - 1):
            for py in (0, h // 2, h - 1):
                u, v = pixel_to_uv(px, py, w, h)
                px2, py2 = uv_to_pixel(u, v, w, h)
                assert (px, py) == (px2, py2), \
                    f"round-trip failed: ({px},{py})->{(u, v)}->({px2},{py2})"


# ---------------------------------------------------------------------------
# Brush stamp
# ---------------------------------------------------------------------------
def test_stamp_circle_shape():
    s = stamp_circle(8, (255, 0, 0, 255), hardness=1.0)
    side = 8 * 2 + 1
    assert len(s) == side * side * 4
    # Centre pixel: r=255, g=0, b=0, a=255
    centre = (side * side // 2) * 4
    assert s[centre + 0] == 255
    assert s[centre + 3] == 255
    # Far corner (>radius): a=0
    corner = 0  # (0,0) — distance sqrt(side^2*2)/2 > 8
    assert s[corner + 3] == 0


def test_stamp_softness_falloff():
    """Softer brushes (hardness<1) drop to ~exp(-1) at the rim."""
    s = stamp_circle(10, (200, 200, 200, 255), hardness=0.0)
    # Pixel at distance 10 (the rim) should be alpha ~= exp(-1)*255 ~= 93.
    side = 10 * 2 + 1
    cx = cy = 10
    rim_x = cx + 10
    rim_y = cy
    i = (rim_y * side + rim_x) * 4
    assert 60 < s[i + 3] < 120, f"rim alpha {s[i + 3]} not in soft-Gaussian band"


def test_stamp_invalid_args():
    with pytest.raises(ValueError):
        stamp_circle(-1, (0, 0, 0, 255))
    with pytest.raises(ValueError):
        stamp_circle(4, (0, 0, 0, 255), hardness=2.0)


# ---------------------------------------------------------------------------
# Alpha-over compositing
# ---------------------------------------------------------------------------
def test_alpha_over_paints_centre():
    # 32x32 transparent destination.
    w = h = 32
    dst = bytearray(w * h * 4)  # all zeros = transparent black
    stamp = stamp_circle(4, (255, 0, 0, 255), hardness=1.0)
    rect = alpha_over(dst, w, h, stamp, 9, 9, 16, 16)
    assert rect == (12, 12, 21, 21), f"unexpected rect {rect}"
    # Centre pixel of dst is opaque red.
    i = (16 * w + 16) * 4
    assert (dst[i + 0], dst[i + 1], dst[i + 2], dst[i + 3]) == (255, 0, 0, 255)
    # Outside the stamp is still transparent.
    i = (0 * w + 0) * 4
    assert dst[i + 3] == 0


def test_alpha_over_clips_at_edge():
    """Stamp landing in the corner should clip cleanly, no out-of-bounds."""
    w = h = 16
    dst = bytearray(w * h * 4)
    stamp = stamp_circle(4, (0, 255, 0, 255), hardness=1.0)
    # Place at (0, 0) — only the bottom-right quadrant of the stamp lands.
    rect = alpha_over(dst, w, h, stamp, 9, 9, 0, 0)
    assert rect == (0, 0, 5, 5), f"clipped rect should be (0,0,5,5), got {rect}"
    # Pixel (0,0) is opaque green.
    assert dst[0:4] == bytearray([0, 255, 0, 255])


def test_alpha_over_off_buffer_no_op():
    w = h = 16
    dst = bytearray(w * h * 4)
    snap = bytes(dst)
    stamp = stamp_circle(4, (255, 255, 255, 255), hardness=1.0)
    rect = alpha_over(dst, w, h, stamp, 9, 9, -100, -100)
    assert rect == (0, 0, 0, 0)
    assert bytes(dst) == snap


def test_alpha_over_erase_mode():
    """Erase mode reduces destination alpha by stamp alpha."""
    w = h = 8
    # Fully-opaque red destination.
    dst = bytearray()
    for _ in range(w * h):
        dst.extend((255, 0, 0, 255))
    stamp = stamp_circle(2, (0, 0, 0, 255), hardness=1.0)
    alpha_over(dst, w, h, stamp, 5, 5, 4, 4, erase=True)
    # Centre should be fully erased (alpha=0).
    i = (4 * w + 4) * 4
    assert dst[i + 3] == 0
    # RGB unchanged on the centre (erase only touches alpha).
    assert dst[i + 0] == 255


# ---------------------------------------------------------------------------
# Flood fill
# ---------------------------------------------------------------------------
def test_flood_fill_filled_count():
    w = h = 8
    # All-zero (transparent black) buffer; fill should hit every pixel.
    dst = bytearray(w * h * 4)
    n = flood_fill(dst, w, h, 0, 0, (0, 255, 0, 255))
    assert n == w * h
    # Every pixel is now opaque green.
    assert dst[0:4] == bytearray([0, 255, 0, 255])
    assert dst[-4:] == bytearray([0, 255, 0, 255])


def test_flood_fill_respects_barrier():
    w = h = 16
    dst = bytearray(w * h * 4)
    # Build a vertical barrier of red pixels at column 8.
    for y in range(h):
        i = (y * w + 8) * 4
        dst[i:i + 4] = bytearray([255, 0, 0, 255])
    # Fill from (0,0) — should reach columns 0..7 only.
    filled = flood_fill(dst, w, h, 0, 0, (0, 255, 0, 255))
    assert filled == 8 * h
    # Column 9 is still transparent.
    assert dst[(0 * w + 9) * 4 + 3] == 0
    # Column 7 is green.
    assert dst[(0 * w + 7) * 4: (0 * w + 7) * 4 + 4] == bytearray([0, 255, 0, 255])


def test_flood_fill_no_op_when_seed_already_target():
    w = h = 4
    dst = bytearray()
    for _ in range(w * h):
        dst.extend((100, 100, 100, 255))
    n = flood_fill(dst, w, h, 0, 0, (100, 100, 100, 255))
    assert n == 0


# ---------------------------------------------------------------------------
# Smear stamp
# ---------------------------------------------------------------------------
def test_smear_pulls_along_drag():
    w = h = 16
    dst = bytearray(w * h * 4)
    # Lay down a horizontal red bar at row 8, columns 0..3.
    for x in range(4):
        i = (8 * w + x) * 4
        dst[i:i + 4] = bytearray([255, 0, 0, 255])
    # Smear from (3, 8) toward (10, 8) with a small radius and full strength.
    smear_stamp(dst, w, h, 10, 8, 4, dx=7, dy=0, strength=1.0)
    # Pixel at (10, 8) should now have some red — the smear pulled the
    # red bar across.
    i = (8 * w + 10) * 4
    assert dst[i + 0] > 100, "expected red to be smeared toward (10, 8)"


def test_smear_zero_drag_is_no_op():
    w = h = 8
    dst = bytearray(w * h * 4)
    for _ in range(w * h):
        pass
    snap = bytes(dst)
    smear_stamp(dst, w, h, 4, 4, 3, dx=0, dy=0, strength=1.0)
    assert bytes(dst) == snap


# ---------------------------------------------------------------------------
# safe_painted_basename
# ---------------------------------------------------------------------------
def test_safe_basename_simple():
    assert safe_painted_basename("foo.bml", "bar.nj.xvm") == "foo.bml__bar.nj.xvm.png"


def test_safe_basename_strips_separators():
    # Paths with traversal pieces / hash separators are flattened.
    out = safe_painted_basename("a/b\\c#d.bml", "x/y.xvm")
    assert "/" not in out and "\\" not in out and "#" not in out
    # The "__" must still be the unique separator.
    parts = out.split("__")
    assert len(parts) == 2


def test_safe_basename_inner_extension_added():
    # Inner without ".png" gets one appended.
    out = safe_painted_basename("foo.bml", "bar.xvm")
    assert out.endswith(".png")


# ---------------------------------------------------------------------------
# PIL helpers
# ---------------------------------------------------------------------------
def test_png_round_trip_via_pil():
    w, h = 16, 16
    src = bytearray()
    for i in range(w * h):
        src.extend((i % 256, (i * 3) % 256, (i * 7) % 256, 255))
    raw = bytes(src)
    enc = rgba_to_png_bytes(raw, w, h)
    # PIL re-decode gives back the same RGBA bytes.
    im = Image.open(io.BytesIO(enc))
    assert im.size == (w, h)
    rt = bytes(im.convert("RGBA").tobytes())
    assert rt == raw


# ---------------------------------------------------------------------------
# Paint API round-trip via TestClient
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    """In-process FastAPI client. Imports server.py once per module."""
    from fastapi.testclient import TestClient
    import server
    return TestClient(server.app)


def _make_png_b64(w: int, h: int, color: tuple[int, int, int, int]) -> str:
    raw = bytearray(color * (w * h))
    enc = rgba_to_png_bytes(bytes(raw), w, h)
    return base64.b64encode(enc).decode("ascii")


def test_api_paint_save_and_active(client, tmp_path, monkeypatch):
    """Save a painted PNG, then list it via /api/paint/active."""
    import server

    # Redirect cache dir to a tmp directory to keep the test hermetic.
    fresh_dir = tmp_path / "painted"
    fresh_dir.mkdir()
    monkeypatch.setattr(server, "PAINTED_TEX_DIR", fresh_dir)

    body = {
        "model_path": "fake_model.bml",
        "inner": "fake_model.nj.xvm",
        "png_b64": _make_png_b64(32, 32, (200, 50, 100, 255)),
    }
    r = client.post("/api/paint/save", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    assert data["width"] == 32 and data["height"] == 32
    cache_path = Path(data["cache_path"])
    assert cache_path.exists()
    assert cache_path.parent == fresh_dir

    # /api/paint/active should report it.
    r2 = client.get("/api/paint/active")
    assert r2.status_code == 200, r2.text
    listing = r2.json()
    assert listing["ok"] is True
    names = [e["basename"] for e in listing["painted"]]
    assert data["basename"] in names
    # Reverse-engineered fields match.
    match = next(e for e in listing["painted"] if e["basename"] == data["basename"])
    assert match["model_path"] == "fake_model.bml"
    assert match["inner"] == "fake_model.nj.xvm"


def test_api_paint_save_rejects_non_png(client, tmp_path, monkeypatch):
    import server
    fresh_dir = tmp_path / "painted"
    fresh_dir.mkdir()
    monkeypatch.setattr(server, "PAINTED_TEX_DIR", fresh_dir)

    body = {
        "model_path": "x.bml",
        "inner": "x.nj.xvm",
        "png_b64": base64.b64encode(b"NOT A PNG").decode("ascii"),
    }
    r = client.post("/api/paint/save", json=body)
    assert r.status_code == 400


def test_api_paint_build_archive_404_when_no_paint(client, tmp_path, monkeypatch):
    """Build_archive should 404 when no painted PNG exists for the host."""
    import server
    fresh_dir = tmp_path / "painted"
    fresh_dir.mkdir()
    monkeypatch.setattr(server, "PAINTED_TEX_DIR", fresh_dir)

    r = client.post("/api/paint/build_archive", json={"model_path": "nonexistent_archive.bml"})
    assert r.status_code == 404


def test_api_paint_save_path_traversal_rejected(client, tmp_path, monkeypatch):
    """Even with adversarial input, the cache write stays in PAINTED_TEX_DIR."""
    import server
    fresh_dir = tmp_path / "painted"
    fresh_dir.mkdir()
    monkeypatch.setattr(server, "PAINTED_TEX_DIR", fresh_dir)

    # safe_painted_basename strips separators, so the file lands inside the dir.
    body = {
        "model_path": "../../etc/passwd",
        "inner": "x",
        "png_b64": _make_png_b64(8, 8, (0, 0, 0, 255)),
    }
    r = client.post("/api/paint/save", json=body)
    # The save must either succeed (with a sanitized name) or refuse —
    # what it must NOT do is write outside PAINTED_TEX_DIR.
    if r.status_code == 200:
        cache = Path(r.json()["cache_path"])
        assert fresh_dir.resolve() == cache.resolve().parent
        # Filename must not contain any path separators that would escape.
        assert "/" not in cache.name and "\\" not in cache.name
