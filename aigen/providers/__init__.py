"""Provider abstraction for AI-assisted texture work (P5 MVP).

This subpackage is the *plumbing* layer for AI texture generation /
upscaling. It is deliberately separate from the v1 in-process
``aigen.Provider`` family (a1111 / comfy / diffusers) which speaks the
``GenRequest`` / ``GenResult`` shapes: those are the legacy "talk to a
running WebUI" providers. This MVP layer adds three things the v1 layer
lacks and that texture-modding actually needs:

  1. A uniform :class:`Provider` ABC with a cost model
     (``estimate_cost_usd``) so the server can reason about spend
     *before* it happens.
  2. A :class:`ProviderRegistry` that surfaces only the providers that
     are actually usable on this machine right now.
  3. A key-free, always-available local upscaler so a fresh clone with
     no API keys and no GPU still does real, deterministic work.

Design rules:
  * NOTHING here makes a network call at import time or in tests.
  * The default budget is ZERO — a fresh install can run the local
    (cost=0) path but is *blocked* from any paid (cost>0) provider until
    the operator opts in with ``AIGEN_*_BUDGET_USD`` env vars.
  * Providers that need a key/URL report ``available() is False`` when
    that key/URL is absent, and are therefore never exercised.
"""
from __future__ import annotations

from .base import (
    BudgetExceeded,
    ImageRequest,
    ImageResult,
    Provider,
    ProviderRegistry,
    default_registry,
)
from .comfyui import ComfyUIProvider
from .local_upscale import LocalUpscaleProvider
from .stability import StabilityProvider

__all__ = [
    "Provider",
    "ProviderRegistry",
    "ImageRequest",
    "ImageResult",
    "BudgetExceeded",
    "default_registry",
    "LocalUpscaleProvider",
    "ComfyUIProvider",
    "StabilityProvider",
]
