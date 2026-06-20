"""Tests for the AI-gen MVP — provider abstraction, budget guard, endpoints.

NO NETWORK. Every test here runs fully offline:

  * The local-upscale provider needs no keys and does real Pillow work.
  * ComfyUI / Stability providers are asserted UNAVAILABLE without their
    env (COMFYUI_URL / STABILITY_API_KEY) and are never invoked.
  * The FastAPI endpoints are driven via TestClient; the only provider
    that actually runs is the free local one.

Isolation: tests that need a particular env (budget, fake key) set it
explicitly and restore it, and they point the budget ledger at a tmp
path so a real install's ledger is never touched and tests don't
collide. ``python -m pytest tests/test_aigen.py -q`` passes standalone.
"""
from __future__ import annotations

import io
import os

import pytest
from PIL import Image

from aigen.budget import (
    ENV_DAILY_BUDGET,
    ENV_LEDGER_PATH,
    ENV_SESSION_BUDGET,
    BudgetExceeded,
    BudgetGuard,
)
from aigen.providers import (
    ImageRequest,
    ProviderRegistry,
    default_registry,
)
from aigen.providers.comfyui import ComfyUIProvider, ENV_COMFYUI_URL
from aigen.providers.local_upscale import LocalUpscaleProvider
from aigen.providers.stability import (
    ENV_STABILITY_API_KEY,
    PRICE_PER_IMAGE_USD,
    StabilityProvider,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _png_bytes(w: int, h: int, color=(120, 180, 60, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", (w, h), color).save(buf, "PNG")
    return buf.getvalue()


def _png_size(data: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(data)).size


@pytest.fixture
def clean_aigen_env(monkeypatch, tmp_path):
    """Strip all AIGEN/provider env so a test starts from the default state,
    and route the budget ledger at a throwaway tmp file."""
    for k in (
        ENV_STABILITY_API_KEY,
        ENV_COMFYUI_URL,
        ENV_SESSION_BUDGET,
        ENV_DAILY_BUDGET,
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(ENV_LEDGER_PATH, str(tmp_path / "ledger.json"))
    return tmp_path


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_registry_lists_providers(clean_aigen_env):
    reg = default_registry()
    names = [p.name for p in reg.all()]
    assert "local_upscale" in names
    assert "comfyui" in names
    assert "stability" in names
    # Only the key-free local provider is available with no env set.
    avail = [p.name for p in reg.available()]
    assert avail == ["local_upscale"]


def test_registry_describe_cost_model(clean_aigen_env):
    reg = default_registry()
    by_name = {d["name"]: d for d in reg.describe_all()}
    assert by_name["local_upscale"]["cost_model"] == "free"
    assert by_name["local_upscale"]["key_free"] is True
    assert by_name["local_upscale"]["available"] is True
    # Stability is the paid one; it is NOT key-free and NOT available w/o key.
    assert by_name["stability"]["cost_model"] == "paid"
    assert by_name["stability"]["key_free"] is False
    assert by_name["stability"]["available"] is False


def test_registry_register_requires_name():
    reg = ProviderRegistry()
    bad = LocalUpscaleProvider()
    bad.name = ""
    with pytest.raises(ValueError):
        reg.register(bad)


# ---------------------------------------------------------------------------
# Budget guard
# ---------------------------------------------------------------------------
def test_budget_default_zero_blocks_cost_and_allows_free(tmp_path):
    g = BudgetGuard(
        session_budget_usd=0.0,
        daily_budget_usd=0.0,
        ledger_path=tmp_path / "ledger.json",
    )
    assert g.session_budget == 0.0 and g.daily_budget == 0.0
    # Free path always passes, even at zero budget.
    g.check(0.0)
    # Any cost>0 request is rejected.
    with pytest.raises(BudgetExceeded) as ei:
        g.check(0.04)
    assert ei.value.scope == "session"
    assert ei.value.remaining == 0.0


def test_budget_reads_env_default_zero(clean_aigen_env):
    # No AIGEN_*_BUDGET_USD env => both default to 0 => generation disabled.
    g = BudgetGuard()
    assert g.session_budget == 0.0
    assert g.daily_budget == 0.0
    assert g.snapshot()["generation_enabled"] is False
    with pytest.raises(BudgetExceeded):
        g.check(0.01)


def test_budget_allows_within_caps_and_records(tmp_path):
    g = BudgetGuard(
        session_budget_usd=1.0,
        daily_budget_usd=1.0,
        ledger_path=tmp_path / "ledger.json",
    )
    g.check(0.4)  # fits
    g.record(0.4)
    assert g.session_spent == pytest.approx(0.4)
    g.check(0.6)  # exactly to the cap is OK
    g.record(0.6)
    # Now the next non-zero request must fail (session exhausted).
    with pytest.raises(BudgetExceeded):
        g.check(0.01)


def test_budget_daily_persists_via_ledger(tmp_path):
    ledger = tmp_path / "ledger.json"
    g1 = BudgetGuard(session_budget_usd=10.0, daily_budget_usd=1.0, ledger_path=ledger)
    g1.record(0.5)
    assert ledger.exists()
    # A fresh guard (simulating a server restart) reloads today's daily spend.
    g2 = BudgetGuard(session_budget_usd=10.0, daily_budget_usd=1.0, ledger_path=ledger)
    assert g2.daily_spent == pytest.approx(0.5)
    # Session spend, however, is per-process and resets.
    assert g2.session_spent == 0.0


# ---------------------------------------------------------------------------
# Local upscale provider (key-free, deterministic, real work)
# ---------------------------------------------------------------------------
def test_local_upscale_2x_dimensions():
    p = LocalUpscaleProvider()
    assert p.available() is True
    assert p.estimate_cost_usd(ImageRequest()) == 0.0
    src = _png_bytes(32, 48)
    res = p.upscale(ImageRequest(image_png=src, scale=2))
    assert res.width == 64 and res.height == 96
    assert res.cost_usd == 0.0
    assert res.provider == "local_upscale"
    # Output is a valid, larger PNG.
    ow, oh = _png_size(res.image_png)
    assert (ow, oh) == (64, 96)
    assert len(res.image_png) > 0


def test_local_upscale_deterministic():
    p = LocalUpscaleProvider()
    src = _png_bytes(40, 40, color=(200, 30, 90, 255))
    a = p.upscale(ImageRequest(image_png=src, scale=3)).image_png
    b = p.upscale(ImageRequest(image_png=src, scale=3)).image_png
    assert a == b  # bit-for-bit reproducible


def test_local_upscale_bad_inputs():
    p = LocalUpscaleProvider()
    with pytest.raises(ValueError):
        p.upscale(ImageRequest(image_png=None, scale=2))
    with pytest.raises(ValueError):
        p.upscale(ImageRequest(image_png=_png_bytes(16, 16), scale=5))


# ---------------------------------------------------------------------------
# ComfyUI / Stability are unavailable without env and never called
# ---------------------------------------------------------------------------
def test_comfyui_unavailable_without_url(clean_aigen_env):
    p = ComfyUIProvider()
    assert p.available() is False  # no COMFYUI_URL => never probes a socket
    # generate() must refuse rather than hit the network.
    with pytest.raises(RuntimeError):
        p.generate(ImageRequest(prompt="hi"))


def test_stability_unavailable_without_key(clean_aigen_env):
    p = StabilityProvider()
    assert p.available() is False
    assert p.estimate_cost_usd(ImageRequest()) == PRICE_PER_IMAGE_USD
    with pytest.raises(RuntimeError):
        p.generate(ImageRequest(prompt="hi"))


def test_stability_build_request_has_no_secret(clean_aigen_env):
    # Even WITH a fake key, build_request must not embed the real secret.
    p = StabilityProvider(api_key="sk-FAKE-TEST-KEY")
    assert p.available() is True
    desc = p.build_request(ImageRequest(prompt="forest tile", seed=7))
    assert "sk-FAKE-TEST-KEY" not in repr(desc)
    assert desc["fields"]["prompt"] == "forest tile"
    # generate() is a stub even with a key — never makes a real call.
    with pytest.raises(NotImplementedError):
        p.generate(ImageRequest(prompt="x"))


# ---------------------------------------------------------------------------
# Endpoints (FastAPI TestClient) — local provider works, budgeted one 402s
# ---------------------------------------------------------------------------
@pytest.fixture
def server_client(monkeypatch, tmp_path):
    """Import server with a clean, zero-budget env and a tmp ledger, then
    reset the server's lazy MVP singletons so they pick up our env."""
    for k in (ENV_STABILITY_API_KEY, ENV_COMFYUI_URL, ENV_SESSION_BUDGET, ENV_DAILY_BUDGET):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(ENV_LEDGER_PATH, str(tmp_path / "ledger.json"))
    import server
    from fastapi.testclient import TestClient

    # Force a fresh registry + budget guard built under THIS env.
    server._AIGEN_REGISTRY = None
    server._AIGEN_BUDGET = None
    client = TestClient(server.app)
    yield client, server
    server._AIGEN_REGISTRY = None
    server._AIGEN_BUDGET = None


def test_endpoint_providers_lists_mvp(server_client):
    client, _ = server_client
    r = client.get("/api/aigen/providers")
    assert r.status_code == 200
    body = r.json()
    assert "mvp_providers" in body
    names = {p["name"] for p in body["mvp_providers"]}
    assert {"local_upscale", "comfyui", "stability"} <= names
    assert body["budget"]["generation_enabled"] is False


def test_endpoint_local_upscale_returns_larger_png(server_client):
    client, _ = server_client
    src = _png_bytes(24, 24)
    r = client.post(
        "/api/aigen/upscale",
        files={"file": ("tile.png", src, "image/png")},
        data={"provider": "local_upscale", "scale": "2"},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"
    ow, oh = _png_size(r.content)
    assert (ow, oh) == (48, 48)  # 24*2
    assert r.headers["X-Aigen-Cost-Usd"] == "0.0000"


def test_endpoint_budgeted_provider_blocked_at_zero(server_client, monkeypatch):
    client, server = server_client
    # Give stability a fake key so it becomes AVAILABLE, but keep budget 0.
    monkeypatch.setenv(ENV_STABILITY_API_KEY, "sk-FAKE-FOR-TEST")
    # Rebuild the registry so stability.available() re-reads the env.
    server._AIGEN_REGISTRY = None
    src = _png_bytes(16, 16)
    r = client.post(
        "/api/aigen/upscale",
        files={"file": ("tile.png", src, "image/png")},
        data={"provider": "stability", "scale": "2"},
    )
    # Provider is available (has a key) but cost>0 under the zero budget =>
    # 402 Payment Required. NEVER a 200, never silently spends.
    assert r.status_code == 402, r.text
    assert "budget" in r.text.lower()


def test_endpoint_unknown_provider_400(server_client):
    client, _ = server_client
    src = _png_bytes(16, 16)
    r = client.post(
        "/api/aigen/upscale",
        files={"file": ("tile.png", src, "image/png")},
        data={"provider": "does_not_exist", "scale": "2"},
    )
    assert r.status_code == 400


def test_endpoint_unavailable_provider_503(server_client):
    client, _ = server_client
    # comfyui has no URL set => unavailable => 503 (never touches network).
    src = _png_bytes(16, 16)
    r = client.post(
        "/api/aigen/upscale",
        files={"file": ("tile.png", src, "image/png")},
        data={"provider": "comfyui", "scale": "2"},
    )
    assert r.status_code == 503
