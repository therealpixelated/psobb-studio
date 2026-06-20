"""Smoke tests for the Live Mod-Test endpoints.

Covered:
  - GET  /api/live_test/config          shape + fields.
  - GET  /api/live_test/log             empty -> entries shape.
  - POST /api/live_test                 unknown kind -> 400.
  - POST /api/live_test                 battle_param without staged file -> 404.
  - POST /api/live_test                 battle_param with valid staged file:
                                        - copies to a redirected newserv dir,
                                        - records an action log entry,
                                        - returns expected shape.
  - POST /api/live_test                 itempmt rejects staged_path traversal.
  - POST /api/live_test                 texture rejects non-PNG bytes.
  - POST /api/live_test                 texture stages PNG to live_overrides.
  - POST /api/live_test/newserv_reload  503 when not configured.

Server.py is imported once; test fixtures monkeypatch the resolver for
``newserv install`` so we never touch the user's real install.
"""
from __future__ import annotations

import base64
import io
import json
import struct
from pathlib import Path
from unittest.mock import patch

import pytest

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """In-process FastAPI client. Imports server.py once per module."""
    import server
    return TestClient(server.app)


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _make_png_bytes() -> bytes:
    """Tiny 1x1 RGBA PNG, valid header + minimal data."""
    # Hard-coded byte-exact 1x1 PNG (RGBA red pixel). Avoids pulling PIL
    # at smoke-test time; smaller than 80 bytes.
    return bytes.fromhex(
        "89504e470d0a1a0a"                        # PNG magic
        "0000000d49484452"                        # IHDR length=13
        "00000001000000010806000000"             # 1x1 RGBA, no compression/filter/interlace
        "1f15c489"                                # IHDR CRC
        "0000000d49444154"                        # IDAT length=13 (placeholder; not deflate-valid)
        "789c63f80f00010101"                     # ...
        "00000000"                                # CRC (intentionally zero — header-only check)
        "0000000049454e44"                        # IEND
        "ae426082"                                # IEND CRC
    )


def _make_real_png_bytes() -> bytes:
    """Byte-exact 1x1 RGBA PNG (red pixel) that PIL can actually decode.

    Used for tests that exercise the src_png_b64 fingerprint path —
    those need a PNG that survives PIL.Image.open(), unlike the tiny
    header-only fixture above.
    """
    return bytes.fromhex(
        "89504e470d0a1a0a"
        "0000000d4948445200000001000000010806000000"
        "1f15c489"
        "0000000d49444154789c63f8cfc0f01f00050001ff89993d1d"
        "0000000049454e44ae426082"
    )


# ---------------------------------------------------------------------------
# /api/live_test/config
# ---------------------------------------------------------------------------
def test_config_returns_expected_shape(client):
    r = client.get("/api/live_test/config")
    assert r.status_code == 200, r.text
    data = r.json()
    # Required keys
    for k in (
        "newserv_battleparam_dir",
        "newserv_itempmt_path",
        "newserv_reload_url",
        "newserv_reload_available",
        "live_overrides_dir",
        "kinds_supported",
        "texture_override_consumer_active",
    ):
        assert k in data, f"missing {k}"
    assert "battle_param" in data["kinds_supported"]
    assert "itempmt" in data["kinds_supported"]
    assert "mob_dsl" in data["kinds_supported"]
    assert "texture" in data["kinds_supported"]
    assert isinstance(data["newserv_reload_available"], bool)
    # Phase 2 isn't shipped yet; the consumer flag is False.
    assert data["texture_override_consumer_active"] is False


# ---------------------------------------------------------------------------
# /api/live_test/log
# ---------------------------------------------------------------------------
def test_log_returns_entries_list(client):
    r = client.get("/api/live_test/log")
    assert r.status_code == 200, r.text
    data = r.json()
    assert "entries" in data
    assert isinstance(data["entries"], list)


def test_log_filters_by_panel(client):
    r = client.get("/api/live_test/log?panel=does-not-exist")
    assert r.status_code == 200, r.text
    assert r.json() == {"entries": []}


# ---------------------------------------------------------------------------
# /api/live_test - validation
# ---------------------------------------------------------------------------
def test_unknown_kind_400(client):
    r = client.post("/api/live_test", json={"kind": "frob"})
    assert r.status_code == 400, r.text
    assert "unknown kind" in r.json().get("detail", "").lower()


def test_battle_param_missing_variant_400(client):
    r = client.post("/api/live_test", json={"kind": "battle_param"})
    assert r.status_code == 400, r.text


def test_battle_param_invalid_variant_400(client):
    r = client.post("/api/live_test", json={"kind": "battle_param", "variant": "bogus"})
    assert r.status_code == 400, r.text


def test_battle_param_no_staged_file_404(client, tmp_path, monkeypatch):
    """Empty staging dir → 404."""
    import server
    monkeypatch.setattr(server, "BATTLE_PARAM_STAGE_DIR", tmp_path)
    r = client.post("/api/live_test", json={"kind": "battle_param", "variant": "on"})
    assert r.status_code == 404, r.text


# ---------------------------------------------------------------------------
# /api/live_test - happy path (battle_param)
# ---------------------------------------------------------------------------
def test_battle_param_full_deploy_to_redirected_dir(client, tmp_path, monkeypatch):
    """Stage a synthetic file + redirect newserv dir + verify deploy + log."""
    import server
    from formats import battle_param as bp_mod

    # Redirect both the staging dir AND the newserv dir to tmp_path.
    stage_dir = tmp_path / "stage"
    ns_dir = tmp_path / "ns"
    stage_dir.mkdir()
    ns_dir.mkdir()
    monkeypatch.setattr(server, "BATTLE_PARAM_STAGE_DIR", stage_dir)
    monkeypatch.setattr(server, "_resolve_newserv_battleparam_dir", lambda: ns_dir)

    # Place a zero-buffer staged file of the correct size.
    fname = bp_mod.VARIANT_TO_FILENAME["on"]
    (stage_dir / fname).write_bytes(b"\x00" * bp_mod.FILE_SIZE)

    r = client.post("/api/live_test", json={
        "kind": "battle_param",
        "variant": "on",
        "panel": "test-bp",
        "attempt_reload": False,
    })
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    assert data["category"] == "server"
    assert data["deployed"]["size"] == bp_mod.FILE_SIZE
    assert Path(data["deployed"]["deployed_to"]).is_file()
    assert data["reload"]["attempted"] is False
    # Required-manual when no reload-url is configured.
    assert data["requires_manual_reload"] is True

    # Verify the action log was updated with this entry.
    log_r = client.get("/api/live_test/log?panel=test-bp")
    assert log_r.status_code == 200
    entries = log_r.json()["entries"]
    assert any(e["panel"] == "test-bp" and e["kind"] == "battle_param"
               for e in entries)


def test_mob_dsl_routes_through_battle_param(client, tmp_path, monkeypatch):
    """kind=mob_dsl is the same chain as kind=battle_param."""
    import server
    from formats import battle_param as bp_mod

    stage_dir = tmp_path / "stage"
    ns_dir = tmp_path / "ns"
    stage_dir.mkdir()
    ns_dir.mkdir()
    monkeypatch.setattr(server, "BATTLE_PARAM_STAGE_DIR", stage_dir)
    monkeypatch.setattr(server, "_resolve_newserv_battleparam_dir", lambda: ns_dir)
    fname = bp_mod.VARIANT_TO_FILENAME["off"]
    (stage_dir / fname).write_bytes(b"\x00" * bp_mod.FILE_SIZE)

    r = client.post("/api/live_test", json={
        "kind": "mob_dsl",
        "variant": "off",
        "attempt_reload": False,
    })
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    assert data["category"] == "server"


# ---------------------------------------------------------------------------
# /api/live_test - itempmt traversal protection
# ---------------------------------------------------------------------------
def test_itempmt_rejects_staged_path_traversal(client, tmp_path, monkeypatch):
    """staged_path must live inside ITEMPMT_STAGE_DIR."""
    import server
    # Point _resolve_newserv_itempmt to a real-looking path so we get past
    # the dir-existence check before the traversal check.
    fake_target = tmp_path / "ns" / "ItemPMT-bb-v4.prs"
    fake_target.parent.mkdir()
    fake_target.write_bytes(b"\x00")
    monkeypatch.setattr(server, "_resolve_newserv_itempmt", lambda: fake_target)

    # An absolute path *outside* the staging dir.
    attack = tmp_path / "evil.prs"
    attack.write_bytes(b"\x00" * 16)
    r = client.post("/api/live_test", json={
        "kind": "itempmt",
        "staged_path": str(attack),
    })
    assert r.status_code == 400, r.text
    assert "staged_path escapes" in r.json().get("detail", "")


# ---------------------------------------------------------------------------
# /api/live_test - texture (Phase 2 staging)
# ---------------------------------------------------------------------------
def test_texture_requires_asset_path(client):
    r = client.post("/api/live_test", json={
        "kind": "texture",
        "png_b64": _b64(b"\x89PNG\r\n\x1a\n"),
    })
    assert r.status_code == 400, r.text


def test_texture_requires_png_b64(client):
    r = client.post("/api/live_test", json={
        "kind": "texture",
        "asset_path": "x.bml#y",
    })
    assert r.status_code == 400, r.text


def test_texture_rejects_non_png(client):
    """png_b64 must start with the PNG magic bytes."""
    r = client.post("/api/live_test", json={
        "kind": "texture",
        "asset_path": "x.bml#y",
        "png_b64": _b64(b"NOTAPNG" + b"\x00" * 32),
    })
    assert r.status_code == 400, r.text


def test_texture_stages_override(client, tmp_path, monkeypatch):
    """Valid texture call writes a .png + .replace pair to live_overrides_dir."""
    import server
    monkeypatch.setattr(server, "LIVE_OVERRIDES_DIR", tmp_path)

    asset = "bm_ene_bm9_s_mericarol.bml#mericarol_body"
    r = client.post("/api/live_test", json={
        "kind": "texture",
        "asset_path": asset,
        "png_b64": _b64(_make_png_bytes()),
    })
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    assert data["category"] == "client"
    # Override files exist on disk
    out_png = Path(data["deployed"]["override_png"])
    out_meta = Path(data["deployed"]["override_meta"])
    assert out_png.is_file()
    assert out_meta.is_file()
    # .replace sidecar holds the asset_path verbatim
    meta = json.loads(out_meta.read_text())
    assert meta["asset_path"] == asset
    # No heartbeat staged — consumer is dormant; warning should be present.
    assert data["deployed"]["consumer_active"] is False
    assert "warning" in data["deployed"]


# ---------------------------------------------------------------------------
# Heartbeat detection — Phase 2 ASI hookup
# ---------------------------------------------------------------------------
def test_consumer_alive_when_heartbeat_fresh(client, tmp_path, monkeypatch):
    """Touching `_consumer_heartbeat` flips texture_override_consumer_active."""
    import server
    monkeypatch.setattr(server, "LIVE_OVERRIDES_DIR", tmp_path)
    # Drop a fresh heartbeat (epoch now, file just created).
    (tmp_path / "_consumer_heartbeat").write_text(
        '{"ts": 0, "swaps": 0, "overrides": 0, "ver": 1}')
    r = client.get("/api/live_test/config")
    assert r.status_code == 200, r.text
    assert r.json()["texture_override_consumer_active"] is True


def test_consumer_alive_via_alternate_sentinel(client, tmp_path, monkeypatch):
    """`_consumer_alive` filename is also accepted (per task spec)."""
    import server
    monkeypatch.setattr(server, "LIVE_OVERRIDES_DIR", tmp_path)
    (tmp_path / "_consumer_alive").write_text("ok")
    r = client.get("/api/live_test/config")
    assert r.status_code == 200, r.text
    assert r.json()["texture_override_consumer_active"] is True


def test_consumer_dormant_when_heartbeat_stale(client, tmp_path, monkeypatch):
    """Heartbeat older than LIVE_OVERRIDES_HEARTBEAT_STALE_S → dormant."""
    import server
    import os
    monkeypatch.setattr(server, "LIVE_OVERRIDES_DIR", tmp_path)
    sentinel = tmp_path / "_consumer_heartbeat"
    sentinel.write_text("stale")
    # Backdate the file by 60 seconds — well past the 10 s stale window.
    old = time.time() - 60.0
    os.utime(sentinel, (old, old))
    r = client.get("/api/live_test/config")
    assert r.status_code == 200, r.text
    assert r.json()["texture_override_consumer_active"] is False


def test_texture_response_consumer_active_when_alive(client, tmp_path, monkeypatch):
    """Texture POST surfaces consumer_active=True when heartbeat is fresh."""
    import server
    monkeypatch.setattr(server, "LIVE_OVERRIDES_DIR", tmp_path)
    (tmp_path / "_consumer_heartbeat").write_text("alive")
    r = client.post("/api/live_test", json={
        "kind": "texture",
        "asset_path": "x.bml#y",
        "png_b64": _b64(_make_png_bytes()),
    })
    assert r.status_code == 200, r.text
    deployed = r.json()["deployed"]
    assert deployed["consumer_active"] is True
    # has_fingerprint False because the request didn't include src_png_b64.
    assert deployed["has_fingerprint"] is False
    # Even with consumer alive, an override without a fingerprint can't
    # match anything — the UI gets a warning explaining why the swap
    # won't fire. (The override IS on disk, so the file path is still
    # surfaced.)
    assert "warning" in deployed
    assert "src_png_b64" in deployed["warning"]


# Need `time` import for the stale-mtime test above. The original module
# imports `import time` indirectly through other paths — make it explicit
# so the new test runs even on a clean import.
import time  # noqa: E402, F401


def test_texture_with_src_png_writes_fingerprint(client, tmp_path, monkeypatch):
    """When src_png_b64 is supplied, the .replace gets a match block."""
    import hashlib as _h
    import server
    monkeypatch.setattr(server, "LIVE_OVERRIDES_DIR", tmp_path)
    real_png = _make_real_png_bytes()
    r = client.post("/api/live_test", json={
        "kind": "texture",
        "asset_path": "z.bml#zz",
        "png_b64": _b64(real_png),
        "src_png_b64": _b64(real_png),
    })
    assert r.status_code == 200, r.text
    deployed = r.json()["deployed"]
    assert deployed["has_fingerprint"] is True
    meta = json.loads(Path(deployed["override_meta"]).read_text())
    assert "match" in meta
    assert meta["match"]["width"] == 1
    assert meta["match"]["height"] == 1
    # Fingerprint is md5 of the decoded RGBA pixels (RGBA byte order,
    # top-down). For a 1x1 fully-red pixel: ff 00 00 ff -> md5.
    expected = _h.md5(b"\xff\x00\x00\xff").hexdigest()
    assert meta["match"]["src_rgba_md5"] == expected


def test_texture_rejects_invalid_src_png(client, tmp_path, monkeypatch):
    """src_png_b64 is validated for PNG magic + decodability."""
    import server
    monkeypatch.setattr(server, "LIVE_OVERRIDES_DIR", tmp_path)
    r = client.post("/api/live_test", json={
        "kind": "texture",
        "asset_path": "z.bml#zz",
        "png_b64": _b64(_make_real_png_bytes()),
        "src_png_b64": _b64(b"NOTAPNG" + b"\x00" * 32),
    })
    assert r.status_code == 400, r.text


# ---------------------------------------------------------------------------
# /api/live_test/newserv_reload
# ---------------------------------------------------------------------------
def test_newserv_reload_503_when_not_configured(client, monkeypatch):
    import server
    monkeypatch.setattr(server, "NEWSERV_RELOAD_URL", "")
    r = client.post("/api/live_test/newserv_reload")
    assert r.status_code == 503, r.text
    assert "NEWSERV_RELOAD_URL" in r.json().get("detail", "")


def test_newserv_reload_calls_sidecar(client, monkeypatch):
    """When NEWSERV_RELOAD_URL is set, _try_newserv_reload runs through."""
    import server
    monkeypatch.setattr(server, "NEWSERV_RELOAD_URL", "http://127.0.0.1:9/reload")
    monkeypatch.setattr(server, "_try_newserv_reload",
                        lambda: (True, "reload OK (200): {}"))
    r = client.post("/api/live_test/newserv_reload")
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True
