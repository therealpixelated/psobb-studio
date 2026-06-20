"""ComfyUI provider — POSTs a workflow graph to ``/prompt``, polls
``/history/{id}`` for the result, then fetches the image bytes via
``/view``.

V1 ships three hard-coded workflow templates (txt2img / img2img /
inpaint) keyed on SDXL-class checkpoints. The workflow is just a Python
literal here so users don't need to drop JSON files in the install
directory.

Users with bespoke ComfyUI graphs can swap their own template into
``WORKFLOWS`` after import; we don't ship a UI for that in v1 (the
A1111 path is simpler for casual users). Tags in the templates that
need to be filled at request time look like ``__SLOT_*__``.

The user must launch ComfyUI on the default port (8188); we don't
auto-detect alternate ports.
"""
from __future__ import annotations

import copy
import json
import logging
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Optional

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

log = logging.getLogger("psobb_editor.aigen.comfy")


def _socket_open(url: str, *, timeout: float = 0.4) -> bool:
    """Fast TCP probe — see a1111._socket_open for rationale."""
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


# ---------------------------------------------------------------------------
# Workflow templates. Each is a "API format" workflow JSON. Slots filled
# at request time are marked by SLOT_* placeholder values.
#
# Keep these small and focused — the editor isn't a full ComfyUI
# replacement, it's a thin wrapper for users who already have ComfyUI
# running and want to call into it for img2img.
# ---------------------------------------------------------------------------
_TXT2IMG_TEMPLATE = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": "__SLOT_SEED__",
            "steps": "__SLOT_STEPS__",
            "cfg": "__SLOT_CFG__",
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "__SLOT_MODEL__"},
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": "__SLOT_WIDTH__",
            "height": "__SLOT_HEIGHT__",
            "batch_size": 1,
        },
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "__SLOT_PROMPT__", "clip": ["4", 1]},
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "__SLOT_NEG_PROMPT__", "clip": ["4", 1]},
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "psobb_editor", "images": ["8", 0]},
    },
}


_IMG2IMG_TEMPLATE = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": "__SLOT_SEED__",
            "steps": "__SLOT_STEPS__",
            "cfg": "__SLOT_CFG__",
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": "__SLOT_DENOISE__",
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["12", 0],
        },
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "__SLOT_MODEL__"},
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "__SLOT_PROMPT__", "clip": ["4", 1]},
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "__SLOT_NEG_PROMPT__", "clip": ["4", 1]},
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "psobb_editor", "images": ["8", 0]},
    },
    "10": {
        "class_type": "LoadImage",
        "inputs": {"image": "__SLOT_INPUT_IMAGE__"},
    },
    "12": {
        "class_type": "VAEEncode",
        "inputs": {"pixels": ["10", 0], "vae": ["4", 2]},
    },
}


_INPAINT_TEMPLATE = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": "__SLOT_SEED__",
            "steps": "__SLOT_STEPS__",
            "cfg": "__SLOT_CFG__",
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": "__SLOT_DENOISE__",
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["13", 0],
        },
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "__SLOT_MODEL__"},
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "__SLOT_PROMPT__", "clip": ["4", 1]},
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "__SLOT_NEG_PROMPT__", "clip": ["4", 1]},
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "psobb_editor", "images": ["8", 0]},
    },
    "10": {
        "class_type": "LoadImage",
        "inputs": {"image": "__SLOT_INPUT_IMAGE__"},
    },
    "11": {
        "class_type": "LoadImage",
        "inputs": {"image": "__SLOT_MASK_IMAGE__"},
    },
    "13": {
        "class_type": "VAEEncodeForInpaint",
        "inputs": {
            "pixels": ["10", 0],
            "vae": ["4", 2],
            "mask": ["12", 0],
            "grow_mask_by": 6,
        },
    },
    "12": {
        "class_type": "ImageToMask",
        "inputs": {"image": ["11", 0], "channel": "red"},
    },
}


WORKFLOWS = {
    MODE_TXT2IMG: _TXT2IMG_TEMPLATE,
    MODE_IMG2IMG: _IMG2IMG_TEMPLATE,
    MODE_INPAINT: _INPAINT_TEMPLATE,
}


def _fill(template: dict, slots: dict) -> dict:
    """Walk the workflow JSON and substitute __SLOT_*__ markers."""
    out = copy.deepcopy(template)

    def walk(node):
        if isinstance(node, dict):
            for k, v in list(node.items()):
                node[k] = walk(v)
            return node
        if isinstance(node, list):
            return [walk(x) for x in node]
        if isinstance(node, str) and node.startswith("__SLOT_") and node.endswith("__"):
            key = node[7:-2]  # strip __SLOT_ and __
            if key not in slots:
                raise KeyError(f"workflow slot {key!r} not provided")
            return slots[key]
        return node

    return walk(out)


class ComfyProvider(Provider):
    """ComfyUI provider — submit workflow, poll history, fetch image."""

    name = "comfy"
    label = "ComfyUI"
    base_url = "http://127.0.0.1:8188"
    supported_modes = (MODE_TXT2IMG, MODE_IMG2IMG, MODE_INPAINT)

    # Reasonable default model name — users override via list_models() pick.
    default_model = "sd_xl_base_1.0.safetensors"

    def __init__(self, base_url: Optional[str] = None) -> None:
        if base_url:
            self.base_url = base_url.rstrip("/")
        self._client_id = str(uuid.uuid4())
        self._last_probe_ok = False
        self._last_probe_at = 0.0
        self._probe_ttl_s = 5.0

    # ----- HTTP helpers -----
    def _get(self, path: str, *, timeout: float = 5.0) -> dict | list:
        url = f"{self.base_url}{path}"
        with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8") or "{}")

    def _get_bytes(self, path: str, *, timeout: float = 30.0) -> bytes:
        url = f"{self.base_url}{path}"
        with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as resp:
            return resp.read()

    def _post_json(self, path: str, payload: dict, *, timeout: float = 30.0) -> dict:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as e:
            try:
                err = e.read().decode("utf-8", errors="replace")[:1000]
            except Exception:  # noqa: BLE001
                err = ""
            raise RuntimeError(f"ComfyUI HTTP {e.code} on {path}: {err}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"ComfyUI unreachable at {self.base_url}: {e.reason}") from e

    def _upload_image(self, png_bytes: bytes, name_hint: str) -> str:
        """Upload a PNG to ComfyUI's input dir and return the assigned filename."""
        url = f"{self.base_url}/upload/image"
        boundary = uuid.uuid4().hex
        body = []
        body.append(f"--{boundary}".encode())
        body.append(
            f'Content-Disposition: form-data; name="image"; filename="{name_hint}"'.encode()
        )
        body.append(b"Content-Type: image/png")
        body.append(b"")
        body.append(png_bytes)
        body.append(f"--{boundary}".encode())
        body.append(b'Content-Disposition: form-data; name="overwrite"')
        body.append(b"")
        body.append(b"true")
        body.append(f"--{boundary}--".encode())
        body.append(b"")
        data = b"\r\n".join(body)
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                resp_json = json.loads(resp.read().decode("utf-8") or "{}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"ComfyUI upload failed: {e}") from e
        # Server returns {"name": "filename.png", "subfolder": "", "type": "input"}
        return resp_json.get("name", name_hint)

    # ----- Provider ABC -----
    def is_available(self) -> bool:
        now = time.time()
        if (now - self._last_probe_at) < self._probe_ttl_s:
            return self._last_probe_ok
        ok = _socket_open(self.base_url, timeout=0.4)
        if ok:
            try:
                self._get("/system_stats", timeout=2.0)
            except (RuntimeError, OSError, json.JSONDecodeError) as e:
                log.debug("ComfyUI socket-open but HTTP probe failed: %s", e)
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
            # ComfyUI's `/object_info` exposes the checkpoints loader's
            # available files. The shape is deeply nested; we walk it
            # carefully and tolerate missing keys (ComfyUI versions vary).
            info = self._get("/object_info/CheckpointLoaderSimple", timeout=5.0)
            if isinstance(info, dict):
                node = info.get("CheckpointLoaderSimple") or {}
                req_inputs = (node.get("input") or {}).get("required") or {}
                ckpt = req_inputs.get("ckpt_name")
                names = []
                if isinstance(ckpt, list) and ckpt:
                    first = ckpt[0]
                    if isinstance(first, list):
                        names = first
                for n in names:
                    out.append({"name": n, "label": n})
        except (RuntimeError, OSError, json.JSONDecodeError) as e:
            log.warning("ComfyUI list_models failed: %s", e)
        return out

    def generate(self, req: GenRequest) -> GenResult:
        self._check()
        if req.mode not in WORKFLOWS:
            raise ValueError(f"ComfyUI provider doesn't support mode {req.mode!r}")
        return self._run_mode(req)

    # ----- mode-agnostic runner -----
    def _run_mode(self, req: GenRequest) -> GenResult:
        # Compute working dim and prepare source(s) if needed.
        ww = req.work_w or 1024
        wh = req.work_h or 1024
        target_w = req.target_w or req.src_w or ww
        target_h = req.target_h or req.src_h or wh
        slots: dict = {
            "SEED": int(req.seed) if req.seed >= 0 else int(time.time() * 1000) & 0xFFFFFFFF,
            "STEPS": int(max(1, req.steps)),
            "CFG": float(max(0.0, req.cfg)),
            "DENOISE": float(max(0.0, min(1.0, req.denoise))),
            "MODEL": req.model or self.default_model,
            "PROMPT": req.prompt or "",
            "NEG_PROMPT": req.neg_prompt or "",
            "WIDTH": int(ww),
            "HEIGHT": int(wh),
        }

        if req.mode in (MODE_IMG2IMG, MODE_INPAINT):
            if not req.src_b64:
                raise ValueError(f"{req.mode} requires src_b64")
            work_img, ww2, wh2, sw, sh = prepare_src_for_gen(
                req.src_b64,
                work_w=req.work_w,
                work_h=req.work_h,
                src_w=req.src_w,
                src_h=req.src_h,
            )
            slots["WIDTH"] = ww2
            slots["HEIGHT"] = wh2
            target_w = req.target_w or sw
            target_h = req.target_h or sh
            png_bytes = _img_to_png_bytes(work_img)
            slots["INPUT_IMAGE"] = self._upload_image(png_bytes, "psobb_in.png")

        if req.mode == MODE_INPAINT:
            if not req.mask_b64:
                raise ValueError("inpaint requires mask_b64")
            mask_img = b64_to_image(req.mask_b64)
            mask_img = resample_to(mask_img, slots["WIDTH"], slots["HEIGHT"])
            mask_bytes = _img_to_png_bytes(mask_img)
            slots["MASK_IMAGE"] = self._upload_image(mask_bytes, "psobb_mask.png")

        workflow = _fill(WORKFLOWS[req.mode], slots)
        t0 = time.time()
        submit = self._post_json(
            "/prompt", {"prompt": workflow, "client_id": self._client_id},
            timeout=15.0,
        )
        prompt_id = submit.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI did not return a prompt_id; got {submit!r}")

        # Poll history. ComfyUI is async; result appears once the worker
        # finishes the queue. We give it a generous 5-minute window.
        deadline = time.time() + 300.0
        history: dict = {}
        while time.time() < deadline:
            try:
                history = self._get(f"/history/{prompt_id}", timeout=5.0)
            except (RuntimeError, OSError, json.JSONDecodeError):
                history = {}
            if isinstance(history, dict) and prompt_id in history:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("ComfyUI generation timed out (5 min)")

        # Find the SaveImage node's output.
        outputs = (history.get(prompt_id) or {}).get("outputs") or {}
        save_node_id = "9"
        node_out = outputs.get(save_node_id) or {}
        files = node_out.get("images") or []
        if not files:
            raise RuntimeError(f"ComfyUI returned no images. outputs={list(outputs.keys())}")
        first = files[0]
        params = (
            f"?filename={first.get('filename', '')}"
            f"&subfolder={first.get('subfolder', '')}"
            f"&type={first.get('type', 'output')}"
        )
        png_bytes = self._get_bytes(f"/view{params}", timeout=30.0)

        from io import BytesIO

        from PIL import Image
        out_img = Image.open(BytesIO(png_bytes)).convert("RGBA")
        out_b64 = finalize_output(out_img, target_w=target_w, target_h=target_h)

        return GenResult(
            out_b64=out_b64,
            out_w=target_w,
            out_h=target_h,
            seed=slots["SEED"],
            generation_time_s=round(time.time() - t0, 2),
            model=slots["MODEL"],
            provider=self.name,
            mode=req.mode,
            info={
                "prompt_id": prompt_id,
                "work_w": slots["WIDTH"],
                "work_h": slots["HEIGHT"],
            },
        )


def _img_to_png_bytes(img) -> bytes:
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
