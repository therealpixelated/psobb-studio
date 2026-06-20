"""Regression tests for the upscale pipeline alpha + max-dim fixes
(server.py Bug A & Bug B, 2026-05-01).

Bug A — RealESRGAN's animevideov3 model is RGB-only. The wrapper must
detect a meaningful alpha channel on the source PNG, split RGB/A, run
the binary on RGB only, Lanczos-resize the alpha to the upscaled dim,
threshold to binary at 128, and recombine into RGBA. Without this fix
PSOBB DXT1 punch-through textures (foliage / hair / particles) lost
their alpha through the pipeline and rendered as black holes / missing
particles in-game.

Bug B — PSOBB silently fails to load textures whose dimensions exceed
1024 on either axis. ``_cascade_upscale`` must:
  * short-circuit when the SOURCE is already at/above the cap (no
    upscale, copy through);
  * Lanczos-down newly-upscaled OUTPUT to PSOBB_MAX_TEXTURE_DIM on each
    axis independently so non-square tiles keep their aspect.

The realesrgan binary is heavy (Vulkan init + GPU work) and isn't
available in every test environment, so every test here monkeypatches
``server._run_realesrgan_inner`` with a synthetic that performs a
deterministic Lanczos upscale. That keeps the test surface focused on
the wrapper logic — alpha preservation + dim cap — which is where both
bugs lived.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture(scope="module")
def srv():
    import server
    return server


def _make_punchthrough_rgba(path: Path, w: int = 64, h: int = 64) -> Path:
    """Synthesize an RGBA PNG with a binary alpha pattern.

    Layout: top half opaque (alpha=255) red, bottom half fully
    transparent (alpha=0) red. Mimics the punch-through pattern of a
    DXT1 foliage / hair / particle tile — exactly the case Bug A broke.
    """
    img = Image.new("RGBA", (w, h), (255, 0, 0, 255))
    pixels = img.load()
    half = h // 2
    for y in range(half, h):
        for x in range(w):
            pixels[x, y] = (255, 0, 0, 0)
    img.save(path)
    return path


def _make_solid_rgb(path: Path, w: int, h: int, color=(0, 128, 255)) -> Path:
    """Synthesize a plain RGB PNG (no alpha channel at all)."""
    Image.new("RGB", (w, h), color).save(path)
    return path


def _make_solid_rgba(path: Path, w: int, h: int) -> Path:
    """Synthesize an RGBA PNG with trivial (uniformly-255) alpha.

    Used as a control: the wrapper should treat this exactly like RGB
    and skip the split/recombine path.
    """
    Image.new("RGBA", (w, h), (32, 200, 64, 255)).save(path)
    return path


def _patch_inner_to_lanczos(monkeypatch, srv) -> dict:
    """Replace ``_run_realesrgan_inner`` with a deterministic Lanczos
    upscale. Returns a dict the test can read to count invocations and
    check call args.

    The real binary writes upscaled RGB to ``dst``. We do the same with
    PIL so the wrapper code path (split → call binary → resize alpha →
    recombine → save) is exercised end-to-end without spawning Vulkan.
    """
    state: dict = {"calls": 0, "last_args": None}

    def fake_inner(src, dst, model, binary_scale, *, tile_size=None,
                   tta=False, gpu_id=None):
        state["calls"] += 1
        state["last_args"] = {
            "src": Path(src), "dst": Path(dst), "model": model,
            "binary_scale": binary_scale,
        }
        with Image.open(src) as im:
            sw, sh = im.size
            up = im.convert("RGB").resize(
                (sw * binary_scale, sh * binary_scale),
                Image.Resampling.LANCZOS,
            )
            up.save(dst)

    monkeypatch.setattr(srv, "_run_realesrgan_inner", fake_inner)
    return state


# ---------------------------------------------------------------------------
# Bug A — alpha preservation
# ---------------------------------------------------------------------------

def test_alpha_roundtrip(srv, tmp_path, monkeypatch):
    """Punch-through alpha survives the upscale.

    Source: 64x64 RGBA, top half opaque + bottom half transparent.
    Expected after 4x: 256x256 RGBA, top half alpha=255, bottom alpha=0,
    with a thresholded boundary (no anti-aliased gradients) so DXT1 can
    still represent the result. We allow a 1-row tolerance at the seam
    for Lanczos kernel slop on the alpha channel before threshold.
    """
    state = _patch_inner_to_lanczos(monkeypatch, srv)

    src = _make_punchthrough_rgba(tmp_path / "src.png", w=64, h=64)
    dst = tmp_path / "dst.png"

    srv._run_realesrgan(src, dst, "realesr-animevideov3-x4", 4)

    assert state["calls"] == 1, (
        "wrapper should invoke the inner binary exactly once for an RGBA src"
    )
    # The inner was invoked on the *RGB-only* sidecar, not the original.
    assert state["last_args"]["src"].name.endswith("_rgb.png"), (
        "wrapper must hand the binary an RGB-only PNG, not the RGBA original "
        f"(got: {state['last_args']['src'].name})"
    )
    # Side files must be cleaned up.
    assert not (tmp_path / "src_rgb.png").exists(), "RGB temp leaked"
    assert not (tmp_path / "src_a.png").exists(), "alpha temp leaked"

    # Output is RGBA at 4x.
    with Image.open(dst) as out:
        assert out.mode == "RGBA", f"expected RGBA output, got {out.mode}"
        assert out.size == (256, 256), f"expected 256x256, got {out.size}"
        out_rgba = out.convert("RGBA").copy()

    # Alpha is binary post-threshold (only 0 or 255).
    alpha = out_rgba.split()[3]
    histogram = alpha.histogram()
    nonbinary = sum(histogram[1:255])
    assert nonbinary == 0, (
        f"alpha must be binary (0 or 255) post-threshold; "
        f"found {nonbinary} pixels with intermediate values"
    )

    # Top half ≥99% alpha=255, bottom half ≥99% alpha=0.
    half = 256 // 2
    a_pixels = list(alpha.tobytes())  # raw bytes is one byte per pixel for L mode
    top = a_pixels[: 256 * half]
    bottom = a_pixels[256 * half:]
    top_opaque_pct = sum(1 for v in top if v == 255) / len(top)
    bot_transparent_pct = sum(1 for v in bottom if v == 0) / len(bottom)
    assert top_opaque_pct >= 0.99, (
        f"top half should be ≥99% opaque after threshold; got {top_opaque_pct:.4f}"
    )
    assert bot_transparent_pct >= 0.99, (
        f"bottom half should be ≥99% transparent after threshold; "
        f"got {bot_transparent_pct:.4f}"
    )


def test_rgb_source_skips_split_path(srv, tmp_path, monkeypatch):
    """Pure-RGB source must use the legacy fast path (no temp files,
    inner called on the original src directly)."""
    state = _patch_inner_to_lanczos(monkeypatch, srv)

    src = _make_solid_rgb(tmp_path / "rgb.png", 32, 32)
    dst = tmp_path / "out.png"

    srv._run_realesrgan(src, dst, "realesr-animevideov3-x4", 4)

    assert state["calls"] == 1
    # Crucially the inner was invoked on the ORIGINAL src, not a
    # split-out RGB sidecar.
    assert state["last_args"]["src"] == src, (
        "RGB src should bypass the split wrapper; inner must see the original"
    )
    assert not (tmp_path / "rgb_rgb.png").exists()
    assert not (tmp_path / "rgb_a.png").exists()


def test_trivially_opaque_rgba_skips_split_path(srv, tmp_path, monkeypatch):
    """RGBA source with uniformly-255 alpha is functionally RGB and
    should skip the split path to avoid pointless I/O."""
    state = _patch_inner_to_lanczos(monkeypatch, srv)

    src = _make_solid_rgba(tmp_path / "opaque.png", 32, 32)
    dst = tmp_path / "out.png"

    srv._run_realesrgan(src, dst, "realesr-animevideov3-x4", 4)

    assert state["calls"] == 1
    assert state["last_args"]["src"] == src, (
        "trivially-opaque RGBA should also bypass the split path"
    )


# ---------------------------------------------------------------------------
# Bug B — max-texture-dim cap
# ---------------------------------------------------------------------------

def _cascade_with_fake_inner(srv, monkeypatch, src, target_dir, base_name,
                              model, requested_scale):
    """Helper: patch inner, call _cascade_upscale, return final path."""
    _patch_inner_to_lanczos(monkeypatch, srv)
    return srv._cascade_upscale(
        src, target_dir, base_name, model, requested_scale,
    )


def test_dim_cap_clamp(srv, tmp_path, monkeypatch):
    """256x256 src @ scale=8 → output is exactly 1024x1024 (not 2048).

    Without the cap, two passes of a 4x model produce 4096x4096, then
    Lanczos-down to the requested 8x lands at 2048x2048 — and the game
    fails to load. With the cap, the final output is clamped to
    PSOBB_MAX_TEXTURE_DIM on each axis.
    """
    src = _make_solid_rgb(tmp_path / "s.png", 256, 256)
    out = _cascade_with_fake_inner(
        srv, monkeypatch, src, tmp_path, "t256", "realesr-animevideov3-x4", 8,
    )
    with Image.open(out) as im:
        assert im.size == (1024, 1024), (
            f"expected 1024x1024 (cap), got {im.size}"
        )


def test_already_at_cap(srv, tmp_path, monkeypatch):
    """1024x1024 src @ scale=2 → output is 1024x1024 (no upscale work).

    Per spec: when the source is already at/above the cap on any axis
    we short-circuit and return the source unchanged. The output dim
    therefore equals the source dim, NOT src*scale.
    """
    state = _patch_inner_to_lanczos(monkeypatch, srv)

    src = _make_solid_rgb(tmp_path / "big.png", 1024, 1024)
    out = srv._cascade_upscale(
        src, tmp_path, "tcap", "realesr-animevideov3-x4", 2,
    )
    with Image.open(out) as im:
        assert im.size == (1024, 1024), (
            f"already-at-cap source must not be downscaled OR upscaled; "
            f"expected 1024x1024, got {im.size}"
        )
    # The binary should never have been invoked — short-circuit copies src through.
    assert state["calls"] == 0, (
        f"already-at-cap path must not run the upscaler; got {state['calls']} calls"
    )


def test_non_square(srv, tmp_path, monkeypatch):
    """256x128 src @ scale=8 → output is exactly 1024x512 (each dim
    capped independently, aspect preserved).

    Without per-axis cap, a naive single-dim cap or aspect-preserving
    fit-within-cap would produce 1024x512 OR 1024x1024 OR 2048x1024 —
    only the first matches PSOBB's actual requirement.
    """
    src = _make_solid_rgb(tmp_path / "rect.png", 256, 128)
    out = _cascade_with_fake_inner(
        srv, monkeypatch, src, tmp_path, "trect", "realesr-animevideov3-x4", 8,
    )
    with Image.open(out) as im:
        assert im.size == (1024, 512), (
            f"non-square src must cap each axis independently; "
            f"expected 1024x512, got {im.size}"
        )


def test_below_cap_unaffected(srv, tmp_path, monkeypatch):
    """256x256 src @ scale=4 → output is 1024x1024 — exactly at cap, no
    clamping needed. Sanity check that the cap doesn't kick in early."""
    src = _make_solid_rgb(tmp_path / "ok.png", 256, 256)
    out = _cascade_with_fake_inner(
        srv, monkeypatch, src, tmp_path, "tok", "realesr-animevideov3-x4", 4,
    )
    with Image.open(out) as im:
        assert im.size == (1024, 1024)


def test_cap_with_alpha_preserves_alpha(srv, tmp_path, monkeypatch):
    """Combined regression: punch-through RGBA src @ scale=8 should
    still produce RGBA at the capped 1024x1024 dim with binary alpha.

    Bug A and Bug B share the same pipeline; this guards against a
    refactor reintroducing either one in isolation.
    """
    _patch_inner_to_lanczos(monkeypatch, srv)
    src = _make_punchthrough_rgba(tmp_path / "ap.png", w=256, h=256)
    out = srv._cascade_upscale(
        src, tmp_path, "tap", "realesr-animevideov3-x4", 8,
    )
    with Image.open(out) as im:
        assert im.size == (1024, 1024), f"dim cap failed: got {im.size}"
        assert im.mode == "RGBA", f"alpha lost: mode={im.mode}"
        # Final downscale (Lanczos-down to cap) intermediates may have
        # introduced fractional alpha — but the upscaler-side wrapper
        # already thresholded once. Allow up to 1% non-binary pixels in
        # the band where Lanczos sampling crossed the seam.
        alpha = im.split()[3]
        hist = alpha.histogram()
        nonbinary = sum(hist[1:255])
        total = im.size[0] * im.size[1]
        assert nonbinary / total <= 0.01, (
            f"alpha lost binary character: {nonbinary}/{total} non-binary "
            f"({100 * nonbinary / total:.2f}%)"
        )
