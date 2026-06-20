"""HuggingFace Diffusers provider — in-process generation.

This provider is **strictly opt-in**: it requires the user to install
``diffusers``, ``torch``, ``transformers``, and ``accelerate`` in the
editor's Python environment themselves. We never trigger that
installation. If the imports fail, the provider reports
``unavailable`` and the UI doesn't light up.

Default model is ``stabilityai/stable-diffusion-xl-base-1.0`` for
img2img/txt2img and ``diffusers/stable-diffusion-xl-1.0-inpainting-
0.1`` for inpaint. Both auto-download on first use to
``~/.cache/huggingface/hub/`` (~6.5 GB each); the user is warned about
the disk footprint via ``/api/aigen/providers``.

Implementation note: pipelines are cached so subsequent calls don't
re-load the weights (which costs 10-30 s + a few GB of VRAM). The
cache is keyed on ``(model_id, mode)``; switching modes for the same
model swaps pipeline instances but keeps the underlying weights.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from . import (
    GenRequest,
    GenResult,
    MODE_IMG2IMG,
    MODE_INPAINT,
    MODE_TXT2IMG,
    Provider,
)
from ._imageutil import (
    b64_to_image,
    finalize_output,
    image_to_b64,
    prepare_src_for_gen,
    resample_to,
)

log = logging.getLogger("psobb_editor.aigen.diffusers")


class DiffusersProvider(Provider):
    name = "diffusers"
    label = "HuggingFace Diffusers (in-process)"
    base_url = ""  # in-process
    supported_modes = (MODE_TXT2IMG, MODE_IMG2IMG, MODE_INPAINT)

    DEFAULT_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
    DEFAULT_INPAINT = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

    def __init__(self) -> None:
        self._pipeline_cache: dict[tuple[str, str], Any] = {}
        self._import_ok: Optional[bool] = None
        self._import_error = ""

    # ----- helpers -----
    def _try_imports(self) -> bool:
        """Probe heavy imports lazily and cache the result."""
        if self._import_ok is not None:
            return self._import_ok
        try:
            import torch  # noqa: F401
            import diffusers  # noqa: F401
            import transformers  # noqa: F401
            self._import_ok = True
        except ImportError as e:
            self._import_ok = False
            self._import_error = str(e)
            log.info("diffusers provider unavailable: %s", e)
        return self._import_ok

    def _device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _dtype(self):
        try:
            import torch
            return torch.float16 if self._device() == "cuda" else torch.float32
        except ImportError:
            return None

    def _get_pipeline(self, mode: str, model_id: str):
        key = (model_id, mode)
        cached = self._pipeline_cache.get(key)
        if cached is not None:
            return cached
        try:
            import torch
            from diffusers import (
                StableDiffusionXLImg2ImgPipeline,
                StableDiffusionXLInpaintPipeline,
                StableDiffusionXLPipeline,
            )
        except ImportError as e:
            raise RuntimeError(f"diffusers not installed: {e}") from e

        cls_map = {
            MODE_TXT2IMG: StableDiffusionXLPipeline,
            MODE_IMG2IMG: StableDiffusionXLImg2ImgPipeline,
            MODE_INPAINT: StableDiffusionXLInpaintPipeline,
        }
        cls = cls_map.get(mode)
        if cls is None:
            raise ValueError(f"diffusers provider can't handle mode {mode!r}")

        dtype = self._dtype()
        kwargs = {"torch_dtype": dtype} if dtype is not None else {}
        log.info("loading diffusers pipeline %s for mode %s (this may take a while)", model_id, mode)
        pipe = cls.from_pretrained(model_id, **kwargs)
        pipe.to(self._device())
        try:
            pipe.set_progress_bar_config(disable=True)
        except (AttributeError, RuntimeError):
            pass
        self._pipeline_cache[key] = pipe
        return pipe

    # ----- Provider ABC -----
    def is_available(self) -> bool:
        return self._try_imports()

    def list_models(self) -> list[dict]:
        if not self._try_imports():
            return []
        # Diffusers has no "what's locally cached" introspection that's
        # reliable across versions. We surface a curated default set
        # plus whatever the user has used so far in this process.
        seen_ids = {self.DEFAULT_BASE, self.DEFAULT_INPAINT}
        for (mid, _mode) in self._pipeline_cache.keys():
            seen_ids.add(mid)
        return [{"name": mid, "label": mid} for mid in sorted(seen_ids)]

    def generate(self, req: GenRequest) -> GenResult:
        if not self._try_imports():
            raise RuntimeError(
                f"diffusers backend unavailable: {self._import_error or 'install diffusers torch transformers accelerate'}"
            )
        if req.mode == MODE_TXT2IMG:
            return self._txt2img(req)
        if req.mode == MODE_IMG2IMG:
            return self._img2img(req)
        if req.mode == MODE_INPAINT:
            return self._inpaint(req)
        raise ValueError(f"unsupported mode for diffusers provider: {req.mode}")

    # ----- modes -----
    def _resolve_seed(self, seed: int):
        try:
            import torch
            if seed is None or seed < 0:
                seed = int(time.time() * 1000) & 0xFFFFFFFF
            return torch.Generator(device=self._device()).manual_seed(int(seed)), int(seed)
        except ImportError:
            return None, int(seed) if seed is not None else 0

    def _txt2img(self, req: GenRequest) -> GenResult:
        ww = req.work_w or 1024
        wh = req.work_h or 1024
        model_id = req.model or self.DEFAULT_BASE
        pipe = self._get_pipeline(MODE_TXT2IMG, model_id)
        gen, used_seed = self._resolve_seed(req.seed)
        t0 = time.time()
        result = pipe(
            prompt=req.prompt or "",
            negative_prompt=req.neg_prompt or "",
            num_inference_steps=int(max(1, req.steps)),
            guidance_scale=float(max(0.0, req.cfg)),
            width=int(ww),
            height=int(wh),
            generator=gen,
        )
        out_img = result.images[0].convert("RGBA")
        target_w = req.target_w or ww
        target_h = req.target_h or wh
        out_b64 = finalize_output(out_img, target_w=target_w, target_h=target_h)
        return GenResult(
            out_b64=out_b64,
            out_w=target_w,
            out_h=target_h,
            seed=used_seed,
            generation_time_s=round(time.time() - t0, 2),
            model=model_id,
            provider=self.name,
            mode=req.mode,
            info={"work_w": ww, "work_h": wh, "device": self._device()},
        )

    def _img2img(self, req: GenRequest) -> GenResult:
        if not req.src_b64:
            raise ValueError("img2img requires src_b64")
        work, ww, wh, sw, sh = prepare_src_for_gen(
            req.src_b64,
            work_w=req.work_w,
            work_h=req.work_h,
            src_w=req.src_w,
            src_h=req.src_h,
        )
        target_w = req.target_w or sw
        target_h = req.target_h or sh
        model_id = req.model or self.DEFAULT_BASE
        pipe = self._get_pipeline(MODE_IMG2IMG, model_id)
        gen, used_seed = self._resolve_seed(req.seed)
        t0 = time.time()
        result = pipe(
            prompt=req.prompt or "",
            negative_prompt=req.neg_prompt or "",
            image=work.convert("RGB"),
            strength=float(max(0.0, min(1.0, req.denoise))),
            num_inference_steps=int(max(1, req.steps)),
            guidance_scale=float(max(0.0, req.cfg)),
            generator=gen,
        )
        out_img = result.images[0].convert("RGBA")
        out_b64 = finalize_output(out_img, target_w=target_w, target_h=target_h)
        return GenResult(
            out_b64=out_b64,
            out_w=target_w,
            out_h=target_h,
            seed=used_seed,
            generation_time_s=round(time.time() - t0, 2),
            model=model_id,
            provider=self.name,
            mode=req.mode,
            info={"work_w": ww, "work_h": wh, "device": self._device()},
        )

    def _inpaint(self, req: GenRequest) -> GenResult:
        if not req.src_b64:
            raise ValueError("inpaint requires src_b64")
        if not req.mask_b64:
            raise ValueError("inpaint requires mask_b64")
        work, ww, wh, sw, sh = prepare_src_for_gen(
            req.src_b64,
            work_w=req.work_w,
            work_h=req.work_h,
            src_w=req.src_w,
            src_h=req.src_h,
        )
        target_w = req.target_w or sw
        target_h = req.target_h or sh
        model_id = req.model or self.DEFAULT_INPAINT
        pipe = self._get_pipeline(MODE_INPAINT, model_id)
        mask = b64_to_image(req.mask_b64)
        mask = resample_to(mask, ww, wh)
        gen, used_seed = self._resolve_seed(req.seed)
        t0 = time.time()
        result = pipe(
            prompt=req.prompt or "",
            negative_prompt=req.neg_prompt or "",
            image=work.convert("RGB"),
            mask_image=mask.convert("L"),
            strength=float(max(0.0, min(1.0, req.denoise))),
            num_inference_steps=int(max(1, req.steps)),
            guidance_scale=float(max(0.0, req.cfg)),
            generator=gen,
            width=int(ww),
            height=int(wh),
        )
        out_img = result.images[0].convert("RGBA")
        out_b64 = finalize_output(out_img, target_w=target_w, target_h=target_h)
        return GenResult(
            out_b64=out_b64,
            out_w=target_w,
            out_h=target_h,
            seed=used_seed,
            generation_time_s=round(time.time() - t0, 2),
            model=model_id,
            provider=self.name,
            mode=req.mode,
            info={"work_w": ww, "work_h": wh, "device": self._device()},
        )
