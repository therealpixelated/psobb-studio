"""Provider ABC + registry for the AI-gen MVP.

A :class:`Provider` is a single backend that can ``generate`` (text/img
-> img) and/or ``upscale`` (img -> bigger img). Every provider answers
three cheap questions before it ever does work:

  * :attr:`name`           — stable id used in the API.
  * :meth:`available`      — is this usable right now? (key/URL/binary
                             present and, for remote ones, reachable).
  * :meth:`estimate_cost_usd` — what would this request cost? ``0.0`` for
                             a local/free path; ``> 0`` for a paid API.

The cost estimate is what the :class:`~aigen.budget.BudgetGuard`
consults *before* the work happens, so a budgeted-out request is
rejected without spending anything.

``generate`` and ``upscale`` both return an :class:`ImageResult`. A
provider that only does one of the two raises :class:`NotImplementedError`
from the other (declare which it supports via :attr:`capabilities`).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional

# Re-export so callers can do ``from aigen.providers.base import BudgetExceeded``.
from ..budget import BudgetExceeded  # noqa: F401

# Capability flags.
CAP_GENERATE = "generate"
CAP_UPSCALE = "upscale"


@dataclass
class ImageRequest:
    """Normalized request handed to a provider.

    A single shape covers both generate and upscale; fields not relevant
    to a given op are simply ignored by that provider. ``image_png`` is
    raw PNG bytes (the source for upscale / img2img); ``prompt`` drives
    text2img / img2img guidance.
    """

    image_png: Optional[bytes] = None  # source PNG bytes (upscale / img2img)
    prompt: str = ""
    negative_prompt: str = ""
    scale: int = 2  # upscale factor (2/3/4); ignored by pure-gen providers
    width: int = 0  # explicit target dim for generate (0 = provider default)
    height: int = 0
    seed: int = -1  # -1 = random / provider default
    steps: int = 20
    model: Optional[str] = None  # provider-specific model id; None = default
    extra: dict = field(default_factory=dict)  # provider-specific knobs


@dataclass
class ImageResult:
    """What a provider returns from ``generate`` / ``upscale``."""

    image_png: bytes  # output PNG bytes
    width: int
    height: int
    provider: str
    op: str  # "generate" | "upscale"
    cost_usd: float = 0.0
    model: str = ""
    seed: int = -1
    info: dict = field(default_factory=dict)


class Provider(abc.ABC):
    """Abstract base for an AI-gen backend."""

    #: Stable short id (lowercase, no spaces) used in API payloads.
    name: str = ""
    #: Human-readable label for the UI.
    label: str = ""
    #: Which ops this provider implements; subset of {CAP_GENERATE, CAP_UPSCALE}.
    capabilities: tuple = ()

    @abc.abstractmethod
    def available(self) -> bool:
        """Cheap liveness/config probe. Must NOT make a network call in the
        common case (a TCP probe with a short timeout is acceptable for
        remote providers, but a provider with no URL configured must
        short-circuit to ``False`` before probing). Never raises.
        """

    @abc.abstractmethod
    def estimate_cost_usd(self, req: ImageRequest) -> float:
        """Estimate the USD cost of ``req`` on this provider. ``0.0`` = free."""

    def generate(self, req: ImageRequest) -> ImageResult:
        """Text/img -> img. Default: unsupported."""
        raise NotImplementedError(f"{self.name!r} does not support generate")

    def upscale(self, req: ImageRequest) -> ImageResult:
        """Img -> larger img. Default: unsupported."""
        raise NotImplementedError(f"{self.name!r} does not support upscale")

    # ----- shared metadata helper -----
    def describe(self) -> dict:
        """JSON-friendly metadata for ``GET /api/aigen/providers``.

        ``available()`` is probed defensively — a misbehaving probe must
        never take the whole listing down.
        """
        try:
            avail = bool(self.available())
        except Exception:  # noqa: BLE001 — defensive; keep the listing alive
            avail = False
        # A representative cost-model sample (a tiny 64x64 2x upscale) so the
        # UI can show "free" vs "paid" without guessing.
        sample = ImageRequest(scale=2, width=64, height=64)
        try:
            sample_cost = float(self.estimate_cost_usd(sample))
        except Exception:  # noqa: BLE001
            sample_cost = 0.0
        return {
            "name": self.name,
            "label": self.label,
            "available": avail,
            "capabilities": list(self.capabilities),
            "cost_model": "free" if sample_cost <= 0 else "paid",
            "sample_cost_usd": sample_cost,
            "key_free": sample_cost <= 0,
        }


class ProviderRegistry:
    """Holds the known providers and answers "which are usable right now"."""

    def __init__(self, providers: Optional[list[Provider]] = None) -> None:
        self._providers: dict[str, Provider] = {}
        for p in providers or []:
            self.register(p)

    def register(self, provider: Provider) -> None:
        if not provider.name:
            raise ValueError("provider must have a non-empty name")
        self._providers[provider.name] = provider

    def get(self, name: str) -> Optional[Provider]:
        return self._providers.get(name)

    def all(self) -> list[Provider]:
        """All registered providers, in registration order."""
        return list(self._providers.values())

    def available(self) -> list[Provider]:
        """Only the providers whose ``available()`` returns True."""
        out = []
        for p in self._providers.values():
            try:
                if p.available():
                    out.append(p)
            except Exception:  # noqa: BLE001 — a bad probe never breaks the list
                continue
        return out

    def describe_all(self) -> list[dict]:
        """Metadata for every registered provider (available or not)."""
        return [p.describe() for p in self._providers.values()]


def default_registry() -> ProviderRegistry:
    """Build a registry with the three MVP providers.

    Local imports keep ``base`` import-light and avoid a cycle (the
    concrete providers import from this module).
    """
    from .local_upscale import LocalUpscaleProvider
    from .comfyui import ComfyUIProvider
    from .stability import StabilityProvider

    return ProviderRegistry(
        [
            LocalUpscaleProvider(),
            ComfyUIProvider(),
            StabilityProvider(),
        ]
    )


__all__ = [
    "Provider",
    "ProviderRegistry",
    "ImageRequest",
    "ImageResult",
    "BudgetExceeded",
    "CAP_GENERATE",
    "CAP_UPSCALE",
    "default_registry",
]
