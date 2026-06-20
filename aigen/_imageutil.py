"""Shared image utilities for the aigen providers.

Tiles in PSOBB are typically 64-1024 px. SDXL/FLUX-class models are
trained at 1024×1024 — feeding them a 64×64 input directly produces
the "low-res training noise" patterns the model learned to reproduce.
We work around this by:

  1. Pre-resampling the source up to a working resolution that the
     model handles well (default 768-1024 short side).
  2. Running the generation at that working resolution.
  3. Lanczos-resampling the output back to the tile's native dim.

Step (3) is the same Lanczos-down the existing realesrgan path
performs, so the final PNG is bit-compatible with everything
downstream of /api/upscale (cache layout, repack flow, etc.).
"""
from __future__ import annotations

import base64
import io
from typing import Optional, Tuple

from PIL import Image


# Working resolution targets — keep short side >= 512, prefer 1024 if
# the target model is SDXL-class. This is a soft default; the caller
# can override via GenRequest.work_w/work_h if they know better.
DEFAULT_WORK_MIN = 512
DEFAULT_WORK_MAX = 1024


def b64_to_image(b64: str) -> Image.Image:
    """Decode a base64 PNG (with or without data: prefix) into an RGBA PIL image."""
    if not b64:
        raise ValueError("empty base64 image")
    if "," in b64 and b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    return img.convert("RGBA")


def image_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image as base64 (no data: prefix)."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def pick_work_dims(
    src_w: int,
    src_h: int,
    *,
    target: int = 1024,
    min_short: int = DEFAULT_WORK_MIN,
    snap: int = 8,
) -> Tuple[int, int]:
    """Choose a working resolution that's friendly to SDXL-class models.

    Rules:
      * Short side >= ``min_short`` (avoids the "low-res training noise"
        artifact regime).
      * Long side <= ``target`` so we don't spend too much time / VRAM
        upscaling tiny tiles to giant working resolutions.
      * Aspect ratio preserved (roughly — snapped to multiples of ``snap``
        which most VAEs require).
      * If the source is already large enough we keep it (snapped).
    """
    if src_w <= 0 or src_h <= 0:
        return target, target
    short = min(src_w, src_h)
    long_ = max(src_w, src_h)
    if short >= min_short and long_ <= target:
        # Already in the sweet spot — just snap.
        w = max(snap, (src_w // snap) * snap)
        h = max(snap, (src_h // snap) * snap)
        return w, h
    # Scale so the short side hits min_short, but cap the long side at target.
    scale_short = min_short / short
    scale_long = target / long_
    scale = min(scale_short, scale_long) if (long_ * scale_short) > target else scale_short
    w = max(snap, int(round(src_w * scale)))
    h = max(snap, int(round(src_h * scale)))
    # Re-snap to multiples of `snap`
    w = (w // snap) * snap or snap
    h = (h // snap) * snap or snap
    return w, h


def resample_to(
    img: Image.Image,
    w: int,
    h: int,
    *,
    method: int = Image.Resampling.LANCZOS,
) -> Image.Image:
    """Lanczos-resize an RGBA image to (w, h). Cheap fast path if already at size."""
    if img.size == (w, h):
        return img
    return img.convert("RGBA").resize((w, h), method)


def prepare_src_for_gen(
    src_b64: str,
    *,
    work_w: Optional[int] = None,
    work_h: Optional[int] = None,
    src_w: int = 0,
    src_h: int = 0,
) -> tuple[Image.Image, int, int, int, int]:
    """Decode a source PNG and resample to a model-friendly working size.

    Returns ``(work_img, work_w, work_h, src_w, src_h)`` where ``src_w/h``
    is the original tile dim (used post-gen to resample back to native).
    """
    img = b64_to_image(src_b64)
    sw, sh = img.size
    if src_w <= 0:
        src_w = sw
    if src_h <= 0:
        src_h = sh
    if work_w is None or work_h is None:
        ww, wh = pick_work_dims(sw, sh)
    else:
        ww, wh = work_w, work_h
    work = resample_to(img, ww, wh)
    return work, ww, wh, src_w, src_h


def finalize_output(
    img: Image.Image,
    *,
    target_w: int,
    target_h: int,
) -> str:
    """Lanczos-resample the model output to the final tile dim and return b64."""
    out = resample_to(img, target_w, target_h)
    return image_to_b64(out)
