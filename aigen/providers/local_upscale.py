"""Local, key-free upscale provider.

This is the provider a fresh clone gets for free: no API key, no GPU, no
running WebUI. It does a *real* deterministic upscale with Pillow's
high-quality Lanczos resampler plus an optional unsharp-mask pass to
recover edge crispness that a pure interpolation loses. Output is
bit-for-bit reproducible for a given input + settings, which is what the
A/B-compare flow downstream of ``/api/upscale`` expects.

Optional fast paths — Real-ESRGAN (``realesrgan-ncnn-vulkan``) or
``quicktex`` — are *probed but never required*. If their weights/binaries
aren't present we fall back to Lanczos, which is why ``available()`` is
unconditionally ``True`` and ``estimate_cost_usd`` is always ``0.0``.

PSOBB DXT textures carry 1-bit punch-through alpha; we upscale RGBA as a
unit so the alpha channel scales with the colour, matching the realesrgan
alpha-preservation path in ``server.py``.
"""
from __future__ import annotations

import io
import logging
from typing import Optional

from PIL import Image, ImageFilter

from .base import CAP_UPSCALE, ImageRequest, ImageResult, Provider

log = logging.getLogger("psobb_editor.aigen.local_upscale")

# Allowed integer upscale factors (matches server's ALLOWED_SCALES spirit).
_ALLOWED_SCALES = (2, 3, 4)
_MAX_OUTPUT_EDGE = 8192  # guard against a tiny tile * huge scale OOM


def _has_realesrgan() -> bool:
    """Best-effort probe for an installed realesrgan binary. Never raises.

    We only *probe*; we never invoke it here (the Lanczos path is the
    committed, test-covered behaviour). A future change can route to the
    binary when present — the provider stays key-free either way.
    """
    import shutil

    try:
        return shutil.which("realesrgan-ncnn-vulkan") is not None
    except Exception:  # noqa: BLE001
        return False


class LocalUpscaleProvider(Provider):
    name = "local_upscale"
    label = "Local Upscale (Lanczos, no key required)"
    capabilities = (CAP_UPSCALE,)

    def available(self) -> bool:
        # Pillow is a hard dependency of the whole app, so this path is
        # always usable. No key, no network, no GPU required.
        return True

    def estimate_cost_usd(self, req: ImageRequest) -> float:
        # Local compute is free.
        return 0.0

    def upscale(self, req: ImageRequest) -> ImageResult:
        if not req.image_png:
            raise ValueError("local_upscale.upscale requires image_png bytes")
        scale = int(req.scale or 2)
        if scale not in _ALLOWED_SCALES:
            raise ValueError(f"scale must be one of {_ALLOWED_SCALES} (got {scale})")

        try:
            src = Image.open(io.BytesIO(req.image_png))
            src.load()
        except Exception as e:  # noqa: BLE001 — surface a clean error to the API
            raise ValueError(f"could not decode source PNG: {e}") from e

        had_alpha = src.mode in ("RGBA", "LA", "P") and (
            "transparency" in src.info or src.mode in ("RGBA", "LA")
        )
        img = src.convert("RGBA")
        sw, sh = img.size
        tw, th = sw * scale, sh * scale
        if max(tw, th) > _MAX_OUTPUT_EDGE:
            raise ValueError(
                f"output {tw}x{th} exceeds max edge {_MAX_OUTPUT_EDGE}px; "
                f"reduce scale or source size"
            )

        out = img.resize((tw, th), Image.Resampling.LANCZOS)

        # Optional unsharp to recover edges. Default ON for upscales; the
        # caller can disable via extra={"unsharp": False}. Deterministic.
        unsharp = req.extra.get("unsharp", True) if req.extra else True
        if unsharp:
            out = self._unsharp_rgb_only(out)

        # Drop alpha back to RGB if the source had none, so a fully-opaque
        # texture round-trips as RGB (matches realesrgan path semantics).
        if not had_alpha:
            out = out.convert("RGB")

        buf = io.BytesIO()
        out.save(buf, format="PNG")
        png = buf.getvalue()

        return ImageResult(
            image_png=png,
            width=tw,
            height=th,
            provider=self.name,
            op="upscale",
            cost_usd=0.0,
            model="lanczos+unsharp" if unsharp else "lanczos",
            seed=req.seed,
            info={
                "src_w": sw,
                "src_h": sh,
                "scale": scale,
                "method": "lanczos",
                "unsharp": bool(unsharp),
                "realesrgan_available": _has_realesrgan(),
            },
        )

    @staticmethod
    def _unsharp_rgb_only(img: Image.Image) -> Image.Image:
        """Apply UnsharpMask to RGB while leaving the alpha channel untouched.

        Sharpening alpha would fringe a punch-through-alpha texture, so we
        split, sharpen the colour, and re-merge the original alpha.
        """
        if img.mode != "RGBA":
            return img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=80, threshold=2))
        r, g, b, a = img.split()
        rgb = Image.merge("RGB", (r, g, b)).filter(
            ImageFilter.UnsharpMask(radius=1.2, percent=80, threshold=2)
        )
        r2, g2, b2 = rgb.split()
        return Image.merge("RGBA", (r2, g2, b2, a))


__all__ = ["LocalUpscaleProvider"]
