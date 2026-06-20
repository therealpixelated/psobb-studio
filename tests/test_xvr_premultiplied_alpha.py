"""Parity tests for the DXT2 / DXT4 premultiplied-alpha decode fix.

Background (ground truth):
  Direct3D 9 defines D3DFMT_DXT2 (pixelFormat 7) and D3DFMT_DXT4 (9) as the
  SAME BC2 / BC3 block layout as DXT3 (8) / DXT5 (10) but with RGB stored
  ALREADY multiplied by alpha (premultiplied alpha). This is confirmed by the
  VrSharp reference (DdsTexUtil.cs: ``IsPMAlpha() ? DXT2 : DXT3`` and
  ``IsPMAlpha() ? DXT4 : DXT5``) and by the live PSOBB texture registry, which
  reports format 7 as "DXT2".

  Pillow / Pfim decode the BC2/BC3 blocks byte-faithfully but as STRAIGHT
  alpha. For a texel with partial alpha the recovered RGB therefore stays
  scaled down (dark) — a straight-alpha renderer then shows it over-darkened /
  washed-out. ``formats/xvr_decode`` now undoes the premultiply on decode for
  fmt 7/9 only (RGB' = min(255, RGB*255/A) for 0 < A < 255), leaving DXT3/DXT5
  (straight alpha) and any fully-opaque texel byte-for-byte unchanged.

These tests pin that behaviour:
  * opaque textures (incl. the real technic.xvm fmt7 fire) are unchanged,
  * partial-alpha fmt7/fmt9 round-trips recover straight RGB,
  * fmt8/fmt10 (straight) are never touched by the unpremultiply,
  * the gate is exactly {7, 9}.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from formats import xvr_decode as X

REPO_ROOT = Path(__file__).resolve().parent.parent
PSOBB_DATA = Path(os.path.expanduser("~/PSOBB.IO/data"))


def _decode_with_premult(rec: dict, premult_set) -> bytes:
    """Decode ``rec`` forcing a specific _PREMULT membership (test helper)."""
    saved = X._PREMULT
    X._PREMULT = premult_set
    try:
        _, _, rgba = X.decode_xvr(rec)
        return rgba
    finally:
        X._PREMULT = saved


def _make_rec(img: Image.Image, fmt: int) -> dict:
    data = X.encode_xvr_data(img, fmt)
    w, h = img.size
    return {"width": w, "height": h, "fmt": fmt, "flags": X.FLAG_ALPHA, "data": data}


# ---------------------------------------------------------------------------
# Gate / mapping sanity
# ---------------------------------------------------------------------------


def test_premult_gate_is_exactly_dxt2_and_dxt4():
    # Strictly the premultiplied variants — NOT DXT3 (8) / DXT5 (10).
    assert X._PREMULT == frozenset((X.FMT_DXT2, X.FMT_DXT4))
    assert X.FMT_DXT2 == 7 and X.FMT_DXT4 == 9
    assert X.FMT_DXT3 not in X._PREMULT and X.FMT_DXT5 not in X._PREMULT


def test_unpremultiply_helper_leaves_opaque_and_clear_verbatim():
    # A==255 (opaque) and A==0 (clear) texels must pass through unchanged;
    # only 0<A<255 is rescaled.
    px = bytes([10, 20, 30, 255,    # opaque -> unchanged
                40, 50, 60, 0,      # clear  -> unchanged (even RGB kept)
                25, 50, 75, 128])   # half   -> rescaled *255/128
    out = X._unpremultiply_rgba(px)
    a = np.frombuffer(out, np.uint8).reshape(-1, 4)
    assert list(a[0]) == [10, 20, 30, 255]
    assert list(a[1]) == [40, 50, 60, 0]
    # 25*255/128 = 49.8 -> 50, 50*255/128=99.6->100, 75*255/128=149.4->149
    assert list(a[2]) == [50, 100, 149, 128]


def test_unpremultiply_clamps_to_255():
    # RGB above its alpha (impossible for true premult data, but be safe):
    # 200 * 255 / 100 = 510 -> clamps to 255, no overflow.
    px = bytes([200, 10, 5, 100])
    out = X._unpremultiply_rgba(px)
    a = np.frombuffer(out, np.uint8).reshape(-1, 4)
    assert list(a[0]) == [255, 26, 13, 100]


def test_all_opaque_is_a_noop():
    px = bytes([1, 2, 3, 255, 4, 5, 6, 255])
    assert X._unpremultiply_rgba(px) == px


# ---------------------------------------------------------------------------
# Decode behaviour, synthetic round-trips
# ---------------------------------------------------------------------------


def test_partial_alpha_fmt7_recovers_straight_rgb():
    # Bright color held at a low, uniform alpha. Straight decode (no fix)
    # returns the dark premultiplied RGB; with the fix it recovers ~bright.
    w = h = 16
    src = np.zeros((h, w, 4), np.uint8)
    src[..., 0] = 230
    src[..., 1] = 40
    src[..., 2] = 200
    src[..., 3] = 64  # 25% alpha
    img = Image.frombytes("RGBA", (w, h), src.tobytes())
    rec = _make_rec(img, X.FMT_DXT2)

    straight = np.frombuffer(
        _decode_with_premult(rec, frozenset()), np.uint8).reshape(h, w, 4)
    fixed = np.frombuffer(
        _decode_with_premult(rec, frozenset((7, 9))), np.uint8).reshape(h, w, 4)

    # Straight (buggy) decode is dark — well below the source brightness.
    assert straight[0, 0, 0] < 120
    # Fixed decode is bright — close to the 230 source R (BC quantization at
    # very low alpha leaves some error, so allow a generous tolerance but it
    # must clear the straight-decode value by a wide margin).
    assert fixed[0, 0, 0] > 180
    assert fixed[0, 0, 0] - straight[0, 0, 0] > 80
    # Alpha channel is preserved (~64; BC2 alpha is 4-bit so allow +/-).
    assert abs(int(fixed[0, 0, 3]) - 64) <= 17


def test_partial_alpha_fmt9_recovers_straight_rgb():
    w = h = 16
    src = np.zeros((h, w, 4), np.uint8)
    src[..., 0] = 60
    src[..., 1] = 210
    src[..., 2] = 90
    src[..., 3] = 80
    img = Image.frombytes("RGBA", (w, h), src.tobytes())
    rec = _make_rec(img, X.FMT_DXT4)

    straight = np.frombuffer(
        _decode_with_premult(rec, frozenset()), np.uint8).reshape(h, w, 4)
    fixed = np.frombuffer(
        _decode_with_premult(rec, frozenset((7, 9))), np.uint8).reshape(h, w, 4)
    assert fixed[0, 0, 1] > straight[0, 0, 1] + 80  # green recovered


def test_fmt8_straight_alpha_is_never_unpremultiplied():
    # DXT3 (8) is straight alpha: decode must NOT rescale RGB, regardless of
    # alpha. Stored RGB == decoded RGB (within BC2 quantization).
    w = h = 16
    src = np.zeros((h, w, 4), np.uint8)
    src[..., 0] = 200
    src[..., 1] = 100
    src[..., 2] = 50
    src[..., 3] = 64
    img = Image.frombytes("RGBA", (w, h), src.tobytes())
    rec = _make_rec(img, X.FMT_DXT3)  # encoder stores straight (no premult)
    dec = np.frombuffer(
        _decode_with_premult(rec, frozenset((7, 9))), np.uint8).reshape(h, w, 4)
    # The full straight color survives (no *255/A amplification).
    assert abs(int(dec[0, 0, 0]) - 200) <= 8
    assert abs(int(dec[0, 0, 1]) - 100) <= 8


def test_encode_decode_roundtrip_symmetry_fmt7():
    # encode_xvr_data(fmt7) premultiplies, decode_xvr(fmt7) unpremultiplies:
    # a straight-alpha PNG survives the round trip (modulo BC2 loss).
    w = h = 32
    rng = np.random.default_rng(7)
    src = rng.integers(40, 220, size=(h, w, 4), dtype=np.uint8)
    src[..., 3] = np.clip(src[..., 3], 96, 255)  # keep alpha decently high
    img = Image.frombytes("RGBA", (w, h), src.tobytes())
    _, _, out = X.decode_xvr(_make_rec(img, X.FMT_DXT2))
    o = np.frombuffer(out, np.uint8).reshape(h, w, 4).astype(np.int32)
    s = src.astype(np.int32)
    # Mean absolute RGB error stays modest (BC2 + premult round-trip).
    mae = np.abs(o[..., :3] - s[..., :3]).mean()
    assert mae < 40, f"round-trip MAE too high: {mae:.1f}"


# ---------------------------------------------------------------------------
# Parity against the REAL game data (opaque fmt7 must be untouched)
# ---------------------------------------------------------------------------


def _opaque_fmt7_records():
    """Yield (name, rec) for fully-opaque fmt7 tiles in real PSOBB.IO data."""
    if not PSOBB_DATA.exists():
        return []
    found = []
    for p in sorted(PSOBB_DATA.glob("*.xvm")):
        try:
            blob = p.read_bytes()
            if blob[:4] != b"XVMH":
                continue
            for r in X.parse_xvm(blob):
                if r["fmt"] != X.FMT_DXT2:
                    continue
                _, _, rgba = X.decode_xvr(r)
                alpha = np.frombuffer(rgba, np.uint8).reshape(-1, 4)[:, 3]
                if int(alpha.min()) == 255:  # fully opaque
                    found.append((f"{p.name}#{r['idx']}", r))
                    if len(found) >= 4:
                        return found
        except (OSError, ValueError):
            continue
    return found


_OPAQUE_FMT7 = _opaque_fmt7_records()


@pytest.mark.skipif(not _OPAQUE_FMT7,
                    reason="no opaque fmt7 (DXT2) tile in ~/PSOBB.IO/data")
def test_real_opaque_fmt7_unchanged_by_unpremultiply():
    """The fix must be a perfect no-op on opaque fmt7 textures (e.g. the
    technic.xvm fire) — bytes identical with and without the unpremultiply."""
    for name, rec in _OPAQUE_FMT7:
        with_fix = _decode_with_premult(rec, frozenset((7, 9)))
        without = _decode_with_premult(rec, frozenset())
        assert with_fix == without, f"{name}: opaque fmt7 changed by unpremultiply"


def test_technic_fmt7_is_opaque_fire_when_available():
    """Specifically pin the technic.xvm fmt7 tile (the 512x512 fire) as opaque
    so the fix is known to leave the user's 'looks correct' fire alone."""
    p = PSOBB_DATA / "technic.xvm"
    if not p.exists():
        pytest.skip("technic.xvm not present")
    recs = [r for r in X.parse_xvm(p.read_bytes()) if r["fmt"] == X.FMT_DXT2]
    assert recs, "technic.xvm has no fmt7 tile"
    _, _, rgba = X.decode_xvr(recs[0])
    alpha = np.frombuffer(rgba, np.uint8).reshape(-1, 4)[:, 3]
    assert int(alpha.min()) == 255, "technic fmt7 fire expected fully opaque"
