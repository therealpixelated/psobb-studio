"""ComfyUI provider (MVP wrapper).

Talks to a ComfyUI instance **only if** ``COMFYUI_URL`` is set in the
environment *and* the host is reachable via a fast TCP probe. With no URL
configured (the default for a fresh clone, and for the test suite),
:meth:`available` short-circuits to ``False`` and the provider is never
exercised — no socket is opened.

Cost is ``0.0``: ComfyUI is self-hosted, so there's no per-call API
charge. It is, however, gated behind ``available()`` because it needs a
running server. The actual graph-submission logic is intentionally
minimal here; the v1 :mod:`aigen.comfy` module carries the full
workflow templates. This MVP provider exists to slot ComfyUI into the
new :class:`~aigen.providers.base.Provider` / cost-model abstraction.
"""
from __future__ import annotations

import logging
import os
import socket
from typing import Optional
from urllib.parse import urlparse

from .base import CAP_GENERATE, ImageRequest, ImageResult, Provider

log = logging.getLogger("psobb_editor.aigen.comfyui")

ENV_COMFYUI_URL = "COMFYUI_URL"


def _tcp_reachable(url: str, *, timeout: float = 0.4) -> bool:
    """Fast TCP connect probe. Never raises; returns False on any failure."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        if not host:
            return False
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
    except Exception:  # noqa: BLE001
        return False


class ComfyUIProvider(Provider):
    name = "comfyui"
    label = "ComfyUI (self-hosted)"
    capabilities = (CAP_GENERATE,)

    def __init__(self, url: Optional[str] = None) -> None:
        # Resolve URL at construction OR lazily at probe time so a test can
        # set the env after import. We re-read the env in available() to
        # stay honest about the current process state.
        self._explicit_url = url

    def _url(self) -> str:
        if self._explicit_url:
            return self._explicit_url
        return (os.environ.get(ENV_COMFYUI_URL) or "").strip()

    def available(self) -> bool:
        url = self._url()
        if not url:
            return False  # not configured — never probe
        return _tcp_reachable(url)

    def estimate_cost_usd(self, req: ImageRequest) -> float:
        # Self-hosted: no per-call charge.
        return 0.0

    def generate(self, req: ImageRequest) -> ImageResult:
        # Guard: refuse to run when not configured/reachable. In the MVP we
        # do not ship the full graph submission here (see aigen.comfy for
        # the v1 workflow templates); this keeps the test suite from ever
        # touching the network while still wiring ComfyUI into the registry.
        url = self._url()
        if not url:
            raise RuntimeError(
                f"ComfyUI is not configured; set {ENV_COMFYUI_URL} to its base URL"
            )
        if not _tcp_reachable(url):
            raise RuntimeError(f"ComfyUI at {url} is not reachable")
        # Intentionally not implemented in the MVP — the abstraction is the
        # deliverable, and we must never make a real call in tests/import.
        raise NotImplementedError(
            "ComfyUI generate() is wired but not implemented in the MVP; "
            "use aigen.comfy for the full workflow path"
        )


__all__ = ["ComfyUIProvider"]
