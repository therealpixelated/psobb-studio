"""Stability AI provider STUB (paid API — out of scope for live calls).

This provider models a *paid* backend so the budget guard has something
to gate. It is :meth:`available` only when ``STABILITY_API_KEY`` is set
in the environment; with no key (the default everywhere, including tests)
it reports ``False`` and is never exercised.

``estimate_cost_usd`` returns a non-zero figure so that — under the
default zero budget — any attempt to route a request here is rejected by
:class:`~aigen.budget.BudgetGuard` *before* any request is built or sent.

``generate`` assembles the request payload but DELIBERATELY does not send
it: real paid network calls are out of scope for this MVP, and nothing in
this file may ever hit the network at import or in tests. The build step
is factored out (:meth:`build_request`) so it can be unit-tested offline
without a key if desired, but the test suite does not require a key and
never calls the live path.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from .base import CAP_GENERATE, ImageRequest, ImageResult, Provider

log = logging.getLogger("psobb_editor.aigen.stability")

ENV_STABILITY_API_KEY = "STABILITY_API_KEY"

# Stability's hosted endpoint. Present for request-building only; this MVP
# never POSTs to it.
DEFAULT_HOST = "https://api.stability.ai"
DEFAULT_ENDPOINT = "/v2beta/stable-image/generate/core"

# Flat per-image price estimate (USD) used by the cost model. Real pricing
# is credit-based and model-dependent; this conservative constant is only
# used to make the guard treat the provider as "paid" (cost > 0).
PRICE_PER_IMAGE_USD = 0.04


class StabilityProvider(Provider):
    name = "stability"
    label = "Stability AI (hosted, paid)"
    capabilities = (CAP_GENERATE,)

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._explicit_key = api_key

    def _api_key(self) -> str:
        if self._explicit_key:
            return self._explicit_key
        return (os.environ.get(ENV_STABILITY_API_KEY) or "").strip()

    def available(self) -> bool:
        # Available iff a key is present. No network probe — having a key is
        # the gate; reachability is the live path's problem (out of scope).
        return bool(self._api_key())

    def estimate_cost_usd(self, req: ImageRequest) -> float:
        # Non-zero so the budget guard blocks this under the default 0 budget.
        return PRICE_PER_IMAGE_USD

    def build_request(self, req: ImageRequest) -> dict:
        """Assemble the request *descriptor* (no secrets, no network I/O).

        Returns a dict describing the call that *would* be made. The API
        key is intentionally NOT included in the returned dict so it can be
        logged/inspected safely. Real submission is out of scope.
        """
        return {
            "url": DEFAULT_HOST + DEFAULT_ENDPOINT,
            "method": "POST",
            "fields": {
                "prompt": req.prompt or "",
                "negative_prompt": req.negative_prompt or "",
                "output_format": "png",
                "seed": req.seed if req.seed and req.seed >= 0 else 0,
                "model": req.model or "sd3-medium",
            },
            "auth": "Authorization: Bearer <STABILITY_API_KEY>",  # placeholder, never the real key
        }

    def generate(self, req: ImageRequest) -> ImageResult:
        key = self._api_key()
        if not key:
            raise RuntimeError(
                f"Stability is not configured; set {ENV_STABILITY_API_KEY}"
            )
        # Build the descriptor (offline, safe) but refuse to actually call
        # out — real paid calls are explicitly out of scope for this MVP.
        _descriptor = self.build_request(req)  # noqa: F841 — documents intent
        raise NotImplementedError(
            "Stability generate() is a stub; real paid API calls are out of "
            "scope for the MVP. The request descriptor is built but not sent."
        )


__all__ = ["StabilityProvider", "ENV_STABILITY_API_KEY", "PRICE_PER_IMAGE_USD"]
