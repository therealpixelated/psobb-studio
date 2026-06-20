"""AUTOMATIC1111-compatible WebUI provider.

Talks to the standard ``/sdapi/v1/*`` HTTP surface that ships with
AUTOMATIC1111, Forge, SD-Next, and reForge. The user must launch
their WebUI with ``--api`` and (recommended) ``--listen 127.0.0.1``.

V1 covers img2img / inpaint / txt2img. ControlNet is supported by
attaching the ``alwayson_scripts.controlnet`` payload that the
``sd-webui-controlnet`` extension expects; if the user doesn't have
that extension installed the field is ignored by the WebUI.
"""
from __future__ import annotations

import json
import logging
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from . import (
    GenRequest,
    GenResult,
    MODE_CONTROLNET,
    MODE_IMG2IMG,
    MODE_INPAINT,
    MODE_TXT2IMG,
    Provider,
)
from ._imageutil import (
    finalize_output,
    image_to_b64,
    prepare_src_for_gen,
    b64_to_image,
    resample_to,
)

log = logging.getLogger("psobb_editor.aigen.a1111")


def _socket_open(url: str, *, timeout: float = 0.4) -> bool:
    """Fast TCP probe — return True if (host, port) accepts a connection.

    On Windows, urllib's RST handling on a closed port can take ~2 s,
    which makes the providers endpoint feel sluggish. A direct socket
    probe with a 400ms timeout fails fast and lets us still report
    accurate liveness within sub-second.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
    except Exception:  # noqa: BLE001
        return False


class A1111Provider(Provider):
    """HTTP client for AUTOMATIC1111-compatible WebUIs."""

    name = "a1111"
    label = "AUTOMATIC1111-compatible WebUI"
    base_url = "http://127.0.0.1:7860"
    supported_modes = (MODE_TXT2IMG, MODE_IMG2IMG, MODE_INPAINT, MODE_CONTROLNET)

    def __init__(self, base_url: Optional[str] = None) -> None:
        if base_url:
            self.base_url = base_url.rstrip("/")
        # Cache liveness for a short window — the providers endpoint is
        # called on every UI refresh and we don't want to spam the WebUI.
        self._last_probe_ok = False
        self._last_probe_at = 0.0
        self._probe_ttl_s = 5.0

    # ----- HTTP helpers -----
    def _get(self, path: str, *, timeout: float = 5.0) -> dict | list:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
        return json.loads(body.decode("utf-8") or "{}")

    def _post(self, path: str, payload: dict, *, timeout: float = 600.0) -> dict:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8", errors="replace")[:1000]
            except Exception:  # noqa: BLE001
                err_body = ""
            raise RuntimeError(f"A1111 HTTP {e.code} on {path}: {err_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"A1111 unreachable at {self.base_url}: {e.reason}") from e
        return json.loads(raw.decode("utf-8") or "{}")

    # ----- Provider ABC -----
    def is_available(self) -> bool:
        now = time.time()
        if (now - self._last_probe_at) < self._probe_ttl_s:
            return self._last_probe_ok
        ok = _socket_open(self.base_url, timeout=0.4)
        if ok:
            try:
                # Confirm it's actually a WebUI (not just any process on 7860).
                self._get("/sdapi/v1/options", timeout=2.0)
            except (RuntimeError, OSError, json.JSONDecodeError) as e:
                log.debug("A1111 socket-open but HTTP probe failed: %s", e)
                ok = False
        self._last_probe_ok = ok
        self._last_probe_at = now
        return ok

    def list_models(self) -> list[dict]:
        try:
            self._check()
        except RuntimeError:
            return []
        out: list[dict] = []
        try:
            models = self._get("/sdapi/v1/sd-models", timeout=5.0)
            if isinstance(models, list):
                for m in models:
                    out.append({
                        "name": m.get("title") or m.get("model_name") or "",
                        "label": m.get("model_name") or m.get("title") or "",
                        "hash": m.get("hash"),
                        "filename": m.get("filename"),
                    })
        except (RuntimeError, OSError, json.JSONDecodeError) as e:
            log.warning("A1111 list_models failed: %s", e)
        return out

    def generate(self, req: GenRequest) -> GenResult:
        self._check()
        if req.mode == MODE_TXT2IMG:
            return self._txt2img(req)
        if req.mode == MODE_IMG2IMG:
            return self._img2img(req)
        if req.mode == MODE_INPAINT:
            return self._img2img(req, inpaint=True)
        if req.mode == MODE_CONTROLNET:
            # ControlNet is just img2img with extra alwayson_scripts payload.
            return self._img2img(req, with_controlnet=True)
        raise ValueError(f"unsupported mode: {req.mode}")

    # ----- mode-specific routing -----
    def _common_payload(self, req: GenRequest, *, work_w: int, work_h: int) -> dict:
        payload: dict = {
            "prompt": req.prompt or "",
            "negative_prompt": req.neg_prompt or "",
            "steps": int(max(1, req.steps)),
            "cfg_scale": float(max(0.0, req.cfg)),
            "seed": int(req.seed),
            "width": int(work_w),
            "height": int(work_h),
            "sampler_name": "Euler a",
            "n_iter": 1,
            "batch_size": 1,
            "send_images": True,
            "save_images": False,
        }
        if req.model:
            # A1111 lets you override the model by passing the title via override_settings.
            payload["override_settings"] = {"sd_model_checkpoint": req.model}
            payload["override_settings_restore_afterwards"] = True
        return payload

    def _txt2img(self, req: GenRequest) -> GenResult:
        # text2img doesn't need a source image; pick a safe default working size.
        ww = req.work_w or 1024
        wh = req.work_h or 1024
        payload = self._common_payload(req, work_w=ww, work_h=wh)
        if req.controlnet:
            self._attach_controlnet(payload, req.controlnet)
        t0 = time.time()
        resp = self._post("/sdapi/v1/txt2img", payload)
        return self._build_result(resp, req, ww, wh, t0)

    def _img2img(
        self,
        req: GenRequest,
        *,
        inpaint: bool = False,
        with_controlnet: bool = False,
    ) -> GenResult:
        if not req.src_b64:
            raise ValueError("img2img/inpaint requires src_b64")
        work, ww, wh, src_w, src_h = prepare_src_for_gen(
            req.src_b64,
            work_w=req.work_w,
            work_h=req.work_h,
            src_w=req.src_w,
            src_h=req.src_h,
        )
        target_w = req.target_w or src_w
        target_h = req.target_h or src_h
        payload = self._common_payload(req, work_w=ww, work_h=wh)
        payload["init_images"] = [image_to_b64(work)]
        payload["denoising_strength"] = float(max(0.0, min(1.0, req.denoise)))
        payload["resize_mode"] = 1  # crop/fit — we already pre-sized

        if inpaint:
            if not req.mask_b64:
                raise ValueError("inpaint requires mask_b64")
            mask = b64_to_image(req.mask_b64)
            mask = resample_to(mask, ww, wh)
            payload["mask"] = image_to_b64(mask)
            payload["inpainting_fill"] = 1  # original — keep texture, repaint mask
            payload["inpaint_full_res"] = 0
            payload["inpainting_mask_invert"] = 0  # white = repaint
            payload["mask_blur"] = 4

        if with_controlnet and req.controlnet:
            self._attach_controlnet(payload, req.controlnet)
        elif req.controlnet:
            # ControlNet was attached to a non-CN-mode request — still allowed,
            # the WebUI ignores the script if the extension isn't loaded.
            self._attach_controlnet(payload, req.controlnet)

        t0 = time.time()
        resp = self._post("/sdapi/v1/img2img", payload)
        return self._build_result(resp, req, target_w, target_h, t0)

    # ----- ControlNet helper -----
    def _attach_controlnet(self, payload: dict, cn: dict) -> None:
        """Attach a single ControlNet unit via alwayson_scripts.

        Expected ``cn`` shape: {model, weight, image_b64, module?}.
        Anything missing falls back to safe defaults; if the WebUI doesn't
        have the controlnet extension the field is harmlessly ignored.
        """
        unit = {
            "input_image": cn.get("image_b64") or "",
            "module": cn.get("module") or "none",
            "model": cn.get("model") or "",
            "weight": float(cn.get("weight", 1.0)),
            "resize_mode": "Just Resize",
            "lowvram": False,
            "processor_res": 512,
            "guidance_start": 0.0,
            "guidance_end": 1.0,
            "control_mode": "Balanced",
            "pixel_perfect": True,
        }
        payload.setdefault("alwayson_scripts", {})["controlnet"] = {"args": [unit]}

    # ----- Response packing -----
    def _build_result(
        self,
        resp: dict,
        req: GenRequest,
        target_w: int,
        target_h: int,
        t0: float,
    ) -> GenResult:
        imgs = resp.get("images") or []
        if not imgs:
            raise RuntimeError(f"A1111 returned no images. info={resp.get('info','')[:200]}")
        # First image is the result (subsequent are CN previews etc.).
        out_img = b64_to_image(imgs[0])
        # Resample down to the requested final dim. Mirrors the Lanczos-down
        # the realesrgan path uses so the cache layout stays consistent.
        out_b64 = finalize_output(out_img, target_w=target_w, target_h=target_h)
        # A1111 packs the actually-used seed inside the JSON-encoded `info`.
        seed = req.seed
        model_used = req.model or ""
        info_dict: dict = {}
        try:
            info_str = resp.get("info") or ""
            if info_str:
                info_dict = json.loads(info_str)
                seed = int(info_dict.get("seed", seed))
                if not model_used:
                    model_used = info_dict.get("sd_model_name") or ""
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return GenResult(
            out_b64=out_b64,
            out_w=target_w,
            out_h=target_h,
            seed=seed,
            generation_time_s=round(time.time() - t0, 2),
            model=model_used,
            provider=self.name,
            mode=req.mode,
            info={
                "work_w": req.work_w,
                "work_h": req.work_h,
                "denoise": req.denoise,
                "steps": req.steps,
                "cfg": req.cfg,
                "raw_info_keys": sorted(info_dict.keys())[:20] if info_dict else [],
            },
        )
