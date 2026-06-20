"""AI image-generation provider plugins for the PSOBB Texture Editor.

Each provider exposes the same Provider ABC; the FastAPI layer calls
``is_available()`` for liveness probing and ``generate()`` /
``inpaint()`` for the actual work. Providers are auto-detected — users
can run any combination of A1111 / ComfyUI / Diffusers (or none) and
the UI lights up the modes that are actually reachable.

V1 ships three concrete providers:

  * a1111      - HTTP client for AUTOMATIC1111-compatible WebUIs
                 (covers stock A1111, Forge, SD-Next, reForge — all
                 share the same /sdapi/v1/* surface).
  * comfy      - HTTP client for ComfyUI's /prompt + /history API.
  * diffusers  - In-process HuggingFace Diffusers (lazy-loaded; only
                 enabled if the user installs the heavy deps themselves).

Adding a new provider is "drop a file in this directory + register in
``ALL_PROVIDERS``". The HTTP/API layer doesn't need changes.

Privacy: all v1 providers point at localhost. Any future remote
provider MUST go through ``_is_local_url()`` and surface a clear
"sends tile data to X" warning in the UI.
"""
from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

log = logging.getLogger("psobb_editor.aigen")


# ---------------------------------------------------------------------------
# Generation modes — flat enum-ish strings so the JSON API is human-readable.
# ---------------------------------------------------------------------------
MODE_TXT2IMG = "text2img"
MODE_IMG2IMG = "img2img"
MODE_INPAINT = "inpaint"
MODE_CONTROLNET = "controlnet"

ALL_MODES = (MODE_TXT2IMG, MODE_IMG2IMG, MODE_INPAINT, MODE_CONTROLNET)


# ---------------------------------------------------------------------------
# Result dataclass — providers return this; the FastAPI layer maps to JSON.
# ---------------------------------------------------------------------------
@dataclass
class GenResult:
    out_b64: str  # PNG bytes, base64-encoded (no data: prefix)
    out_w: int
    out_h: int
    seed: int
    generation_time_s: float
    model: str
    provider: str
    mode: str
    info: dict = field(default_factory=dict)  # provider-specific metadata


@dataclass
class GenRequest:
    """Normalized request — providers receive this."""

    mode: str
    src_b64: Optional[str] = None  # source PNG (img2img / inpaint / controlnet)
    src_w: int = 0
    src_h: int = 0
    prompt: str = ""
    neg_prompt: str = ""
    denoise: float = 0.6  # 0..1, higher = more change from src
    steps: int = 30
    cfg: float = 7.0
    seed: int = -1  # -1 = random
    mask_b64: Optional[str] = None  # inpaint only — white=repaint, black=preserve
    controlnet: Optional[dict] = None  # {model, weight, image_b64}
    model: Optional[str] = None  # provider-specific model id; None = provider default
    target_w: Optional[int] = None  # final output dim; defaults to src dim
    target_h: Optional[int] = None
    work_w: Optional[int] = None  # working resolution; defaults to max(src, 512)
    work_h: Optional[int] = None


# ---------------------------------------------------------------------------
# Provider ABC. Concrete providers live in sibling modules and are
# instantiated lazily via ``get_providers()`` below.
# ---------------------------------------------------------------------------
class Provider(abc.ABC):
    name: str = ""  # short id used in API payloads
    label: str = ""  # human-readable name
    base_url: str = ""  # for HTTP providers; "" for in-process
    supported_modes: tuple = ()

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Cheap liveness probe. Must return quickly (<3s). Cache as appropriate."""

    @abc.abstractmethod
    def list_models(self) -> list[dict]:
        """Return a list of {name, label?, capabilities?} dicts. Provider-specific."""

    @abc.abstractmethod
    def generate(self, req: GenRequest) -> GenResult:
        """Run a generation. Raises RuntimeError on provider failure."""

    def inpaint(self, req: GenRequest) -> GenResult:
        """Convenience: most providers route inpaint through generate()."""
        if req.mode != MODE_INPAINT:
            raise ValueError(f"inpaint() called with mode={req.mode!r}")
        return self.generate(req)

    # ----- helpers shared by HTTP providers -----
    def _check(self) -> None:
        """Raise RuntimeError if the provider is not actually reachable."""
        if not self.is_available():
            raise RuntimeError(f"provider {self.name!r} is not available right now")


# ---------------------------------------------------------------------------
# Localhost guard — all v1 providers must point at 127.0.0.1 / localhost.
# Future agents adding remote providers must surface a "sends data to X"
# warning in the UI. We expose this as a helper rather than a hard guard
# so an explicit override path stays open.
# ---------------------------------------------------------------------------
_LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1", "0.0.0.0")


def is_local_url(url: str) -> bool:
    if not url:
        return True  # in-process providers
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return False
    return host in _LOCAL_HOSTS


# ---------------------------------------------------------------------------
# Provider registry. Concrete classes are imported lazily so that an
# ImportError in (say) the diffusers module doesn't take the others
# down with it.
# ---------------------------------------------------------------------------
_PROVIDERS_CACHE: dict[str, Provider] = {}


def get_provider(name: str) -> Optional[Provider]:
    """Resolve a provider by name; cached. Returns None on import failure."""
    if name in _PROVIDERS_CACHE:
        return _PROVIDERS_CACHE[name]
    cls = None
    try:
        if name == "a1111":
            from .a1111 import A1111Provider
            cls = A1111Provider
        elif name == "comfy":
            from .comfy import ComfyProvider
            cls = ComfyProvider
        elif name == "diffusers":
            from .diffusers_provider import DiffusersProvider
            cls = DiffusersProvider
    except ImportError as e:
        log.info("provider %r unavailable: %s", name, e)
        return None
    if cls is None:
        return None
    inst = cls()
    _PROVIDERS_CACHE[name] = inst
    return inst


def all_provider_names() -> tuple[str, ...]:
    return ("a1111", "comfy", "diffusers")


def list_providers_status() -> list[dict]:
    """Build the response payload for GET /api/aigen/providers."""
    out = []
    for n in all_provider_names():
        p = get_provider(n)
        if p is None:
            out.append({
                "name": n,
                "label": _STATIC_LABELS.get(n, n),
                "status": "import_failed",
                "available": False,
                "base_url": "",
                "supported_modes": list(_STATIC_MODES.get(n, [])),
                "is_local": True,
                "hint": _STATIC_HINTS.get(n, ""),
            })
            continue
        try:
            avail = p.is_available()
        except (RuntimeError, OSError) as e:
            log.debug("provider %s liveness probe failed: %s", n, e)
            avail = False
        out.append({
            "name": p.name,
            "label": p.label,
            "status": "available" if avail else "unavailable",
            "available": bool(avail),
            "base_url": p.base_url,
            "supported_modes": list(p.supported_modes),
            "is_local": is_local_url(p.base_url),
            "hint": _STATIC_HINTS.get(n, ""),
        })
    return out


# Static metadata used when a provider can't be instantiated (e.g. diffusers
# without torch installed). Keeps /api/aigen/providers from going dark.
_STATIC_LABELS = {
    "a1111": "AUTOMATIC1111-compatible WebUI",
    "comfy": "ComfyUI",
    "diffusers": "HuggingFace Diffusers (in-process)",
}
_STATIC_MODES = {
    "a1111": (MODE_TXT2IMG, MODE_IMG2IMG, MODE_INPAINT, MODE_CONTROLNET),
    "comfy": (MODE_TXT2IMG, MODE_IMG2IMG, MODE_INPAINT),
    "diffusers": (MODE_TXT2IMG, MODE_IMG2IMG, MODE_INPAINT),
}
_STATIC_HINTS = {
    "a1111": "Start AUTOMATIC1111 / Forge / SD-Next with --api on port 7860.",
    "comfy": "Start ComfyUI on port 8188 (default).",
    "diffusers": "pip install diffusers torch transformers accelerate (one-time).",
}


def hint_for(name: str) -> str:
    """Public accessor for the human-readable startup hint per provider."""
    return _STATIC_HINTS.get(name, "")


__all__ = [
    "Provider",
    "GenRequest",
    "GenResult",
    "MODE_TXT2IMG",
    "MODE_IMG2IMG",
    "MODE_INPAINT",
    "MODE_CONTROLNET",
    "ALL_MODES",
    "get_provider",
    "all_provider_names",
    "list_providers_status",
    "is_local_url",
    "hint_for",
]
