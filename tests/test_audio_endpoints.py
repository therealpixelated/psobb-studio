"""Integration tests for the audio-suite server endpoints (server.py
/api/audio/*).

CARDINAL SAFETY INVARIANT under test: the Replace verb writes DEV ONLY.
``test_replace_deploy_dev_only_live_byte_identical`` snapshots the live dir's
md5s, performs a deploy=true replace, and asserts the live dir is byte-
identical afterward (mirrors test_floor_editor.test_floor_create_never_touches_live).

Other gates: ffmpeg-absent degrades to 501 not 500; .sfd/.adx replace -> 400;
a concurrent write -> 409; routing (.pac -> audio/PAC, opening_j.sfd ->
audio/SFD). All run against monkeypatched tmp DEV/LIVE dirs via TestClient;
none requires a real PSOBB install or ffmpeg.
"""
from __future__ import annotations

import hashlib
import io
import struct
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import server  # noqa: E402
from formats import audio_pac as ap  # noqa: E402
from formats import audio_codec as ac  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _snapshot_dir(d: Path) -> dict:
    out: dict = {}
    if not d.exists():
        return out
    for p in sorted(d.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(d))] = _md5(p.read_bytes())
    return out


def _wav_record(pcm: bytes, pad: int = 8) -> bytes:
    return ap.WFX_SIG + b"data" + struct.pack("<I", len(pcm)) + pcm + (b"\x00" * pad)


def _synthetic_pac(n: int = 3) -> bytes:
    out = bytearray()
    for i in range(n):
        nsamp = 200 * (i + 1)
        pcm = (np.sin(np.arange(nsamp) / (3 + i)) * 9000).astype(np.int16).tobytes()
        out += _wav_record(pcm, pad=(8, 12, 0)[i % 3])
    return bytes(out)


def _native_wav(nsamp: int = 400) -> bytes:
    """A 22050/mono/16 WAV (PSOBB-native — needs no ffmpeg to ingest)."""
    pcm = (np.sin(np.arange(nsamp) / 5) * 12000).astype(np.int16).tobytes()
    return ap.pcm_to_wav(pcm)


def _asf_stub() -> bytes:
    """Bytes carrying the ASF GUID so routing classifies as SFD."""
    return bytes.fromhex("3026b2758e66cf11a6d900aa0062ce6c") + b"\x00" * 256


# ---------------------------------------------------------------------------
# fixtures: monkeypatched DEV / LIVE dirs + a seeded .pac
# ---------------------------------------------------------------------------
@pytest.fixture
def audio_env(tmp_path, monkeypatch):
    dev = tmp_path / "dev" / "data"
    live = tmp_path / "live" / "data"
    dev.mkdir(parents=True)
    live.mkdir(parents=True)
    monkeypatch.setattr(server, "DEV_DATA_DIR", dev.resolve())
    monkeypatch.setattr(server, "LIVE_DATA_DIR", live.resolve())
    monkeypatch.setattr(server, "DATA_DIR", dev.resolve())
    return {"dev": dev.resolve(), "live": live.resolve()}


@pytest.fixture
def client():
    return TestClient(server.app)


@pytest.fixture
def seeded_pac(audio_env):
    blob = _synthetic_pac(3)
    (audio_env["dev"] / "forest.pac").write_bytes(blob)
    return {"name": "forest.pac", "blob": blob, **audio_env}


@pytest.fixture
def no_ffmpeg(monkeypatch):
    """Force ffmpeg_available() False everywhere the suite checks."""
    monkeypatch.setattr(ac, "ffmpeg_path", lambda: None)
    return True


# ---------------------------------------------------------------------------
# /api/audio/info
# ---------------------------------------------------------------------------
def test_info_pac(seeded_pac, client):
    r = client.get("/api/audio/info", params={"path": "forest.pac"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["container"] == "PAC"
    assert j["decode_kind"] == "pac"
    assert j["record_count"] == 3
    assert j["replace_supported"] is True
    assert all(rec["structured"] for rec in j["records"])


def test_info_missing_404(audio_env, client):
    r = client.get("/api/audio/info", params={"path": "nope.pac"})
    assert r.status_code == 404


def test_info_non_audio_400(audio_env, client):
    (audio_env["dev"] / "foo.xvm").write_bytes(b"XVMH" + b"\x00" * 32)
    r = client.get("/api/audio/info", params={"path": "foo.xvm"})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# /api/audio/decode
# ---------------------------------------------------------------------------
def test_decode_pac_record_returns_wav(seeded_pac, client):
    r = client.get("/api/audio/decode", params={"path": "forest.pac", "record": 1})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.content[:4] == b"RIFF" and r.content[8:12] == b"WAVE"
    # The decoded PCM matches the bank record.
    bank = ap.parse_pac(seeded_pac["blob"])
    pcm, _r, _c, _b = ap.wav_to_pcm(r.content)
    assert pcm == bank.records[1].pcm


def test_decode_pac_record_out_of_range_404(seeded_pac, client):
    r = client.get("/api/audio/decode", params={"path": "forest.pac", "record": 99})
    assert r.status_code == 404


def test_decode_ogg_without_ffmpeg_is_501_not_500(audio_env, client, no_ffmpeg):
    (audio_env["dev"] / "bgm.ogg").write_bytes(b"OggS" + b"\x00" * 64)
    r = client.get("/api/audio/decode", params={"path": "bgm.ogg"})
    assert r.status_code == 501, f"expected 501 (degrade), got {r.status_code}: {r.text}"


def test_decode_sfd_without_ffmpeg_is_501_not_500(audio_env, client, no_ffmpeg):
    (audio_env["dev"] / "opening_j.sfd").write_bytes(_asf_stub())
    r = client.get("/api/audio/decode", params={"path": "opening_j.sfd"})
    assert r.status_code == 501


@pytest.mark.skipif(not ac.ffmpeg_available(), reason="ffmpeg not installed")
def test_decode_ogg_with_ffmpeg_returns_wav(audio_env, client):
    """Positive ffmpeg path: a real Ogg decodes to WAV (skipped if no ffmpeg)."""
    # Encode a real Ogg from synthetic PCM using the codec's own encoder.
    pcm = (np.sin(np.arange(4410) / 5) * 12000).astype(np.int16).tobytes()
    ogg = ac.encode_ogg(ap.pcm_to_wav(pcm), in_kind="wav")
    assert ogg[:4] == b"OggS"
    (audio_env["dev"] / "bgm.ogg").write_bytes(ogg)
    r = client.get("/api/audio/decode", params={"path": "bgm.ogg"})
    assert r.status_code == 200, r.text
    assert r.content[:4] == b"RIFF" and r.content[8:12] == b"WAVE"


# ---------------------------------------------------------------------------
# /api/audio/waveform
# ---------------------------------------------------------------------------
def test_waveform_pac_buckets(seeded_pac, client):
    r = client.get("/api/audio/waveform",
                   params={"path": "forest.pac", "record": 0, "buckets": 64})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["buckets"] == 64
    assert len(j["min"]) == len(j["max"]) == len(j["rms"]) == 64


# ---------------------------------------------------------------------------
# /api/audio/replace — preview (deploy=false) touches NOTHING
# ---------------------------------------------------------------------------
def test_replace_preview_writes_nothing(seeded_pac, client):
    before_dev = _snapshot_dir(seeded_pac["dev"])
    before_live = _snapshot_dir(seeded_pac["live"])
    wav = _native_wav(300)
    r = client.post(
        "/api/audio/replace",
        data={"path": "forest.pac", "record": "1", "deploy": "false"},
        files={"file": ("clip.wav", wav, "audio/wav")},
    )
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["deployed"] is False
    assert j["export_url"].startswith("/api/export/")
    # Neither tree changed.
    assert _snapshot_dir(seeded_pac["dev"]) == before_dev
    assert _snapshot_dir(seeded_pac["live"]) == before_live


# ---------------------------------------------------------------------------
# THE SAFETY CONTRACT: deploy=true writes DEV only; LIVE byte-identical
# ---------------------------------------------------------------------------
def test_replace_deploy_dev_only_live_byte_identical(seeded_pac, client):
    """deploy=true writes DEV; the LIVE dir is byte-identical afterward."""
    # Seed a LIVE copy too, to prove it is never touched.
    (seeded_pac["live"] / "forest.pac").write_bytes(seeded_pac["blob"])
    before_live = _snapshot_dir(seeded_pac["live"])
    assert before_live, "live should hold the seeded bank"

    wav = _native_wav(256)
    r = client.post(
        "/api/audio/replace",
        data={"path": "forest.pac", "record": "1", "deploy": "true"},
        files={"file": ("clip.wav", wav, "audio/wav")},
    )
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["deployed"] is True
    # The DEV bank actually changed and is a child of DEV.
    dev_target = Path(j["path"])
    assert dev_target.parent == seeded_pac["dev"]
    assert _md5(dev_target.read_bytes()) != _md5(seeded_pac["blob"])
    # A .pre_promote backup was taken in DEV.
    assert j["backup_path"] is not None
    assert Path(j["backup_path"]).exists()
    assert Path(j["backup_path"]).parent == seeded_pac["dev"]

    # LIVE is byte-identical.
    after_live = _snapshot_dir(seeded_pac["live"])
    assert after_live == before_live, (
        f"LIVE dir was modified! before={before_live} after={after_live}")

    # And the rebuilt DEV bank re-parses with the swapped record + byte-exact.
    new_bank = ap.parse_pac(dev_target.read_bytes())
    assert ap.write_pac(new_bank) == dev_target.read_bytes()
    pcm_in, _r, _c, _b = ap.wav_to_pcm(wav)
    assert new_bank.records[1].pcm == pcm_in


def test_replace_deploy_no_live_copy_still_dev_only(seeded_pac, client):
    """Even with no LIVE copy present, deploy writes only DEV and leaves LIVE
    empty (the assert_not_live guard never lets a write escape DEV)."""
    before_live = _snapshot_dir(seeded_pac["live"])
    assert before_live == {}
    wav = _native_wav(200)
    r = client.post(
        "/api/audio/replace",
        data={"path": "forest.pac", "record": "0", "deploy": "true"},
        files={"file": ("clip.wav", wav, "audio/wav")},
    )
    assert r.status_code == 200, r.text
    assert _snapshot_dir(seeded_pac["live"]) == {}


# ---------------------------------------------------------------------------
# replace rejects non-targets
# ---------------------------------------------------------------------------
def test_replace_rejects_sfd_400(audio_env, client):
    (audio_env["dev"] / "opening_j.sfd").write_bytes(_asf_stub())
    wav = _native_wav(100)
    r = client.post(
        "/api/audio/replace",
        data={"path": "opening_j.sfd", "record": "0", "deploy": "false"},
        files={"file": ("clip.wav", wav, "audio/wav")},
    )
    assert r.status_code == 400, r.text


def test_replace_rejects_adx_400(audio_env, client):
    (audio_env["dev"] / "roar.adx").write_bytes(b"\x80\x00" + b"\x00" * 64)
    wav = _native_wav(100)
    r = client.post(
        "/api/audio/replace",
        data={"path": "roar.adx", "record": "0", "deploy": "false"},
        files={"file": ("clip.wav", wav, "audio/wav")},
    )
    # .adx is not even an audio container the suite handles -> 400.
    assert r.status_code == 400


def test_replace_unstructured_pac_disabled_400(audio_env, client):
    # An opaque bank (no 'data' chunks) is not a safe replace target.
    (audio_env["dev"] / "weird.pac").write_bytes(ap.WFX_SIG + b"\x00" * 512)
    wav = _native_wav(100)
    r = client.post(
        "/api/audio/replace",
        data={"path": "weird.pac", "record": "0", "deploy": "true"},
        files={"file": ("clip.wav", wav, "audio/wav")},
    )
    assert r.status_code == 400


def test_replace_ogg_upload_passthrough_dev_only(audio_env, client):
    """A like-for-like .ogg upload is a byte copy needing no ffmpeg; writes DEV."""
    (audio_env["dev"] / "bgm.ogg").write_bytes(b"OggS" + b"\x00" * 64)
    before_live = _snapshot_dir(audio_env["live"])
    new_ogg = b"OggS" + b"\x11" * 128
    r = client.post(
        "/api/audio/replace",
        data={"path": "bgm.ogg", "deploy": "true"},
        files={"file": ("new.ogg", new_ogg, "audio/ogg")},
    )
    assert r.status_code == 200, r.text
    target = audio_env["dev"] / "bgm.ogg"
    assert target.read_bytes() == new_ogg
    assert _snapshot_dir(audio_env["live"]) == before_live


def test_replace_wav_to_ogg_without_ffmpeg_is_501(audio_env, client, no_ffmpeg):
    (audio_env["dev"] / "bgm.ogg").write_bytes(b"OggS" + b"\x00" * 64)
    wav = _native_wav(100)
    r = client.post(
        "/api/audio/replace",
        data={"path": "bgm.ogg", "deploy": "false"},
        files={"file": ("clip.wav", wav, "audio/wav")},
    )
    assert r.status_code == 501, r.text


# ---------------------------------------------------------------------------
# lock -> 409 on concurrent write
# ---------------------------------------------------------------------------
def test_replace_lock_409(seeded_pac, client, monkeypatch):
    """Hold the per-bank lock and prove a second replace gets 409."""
    lk = server._get_lock(server._AUDIO_LOCKS, "forest.pac", server.MAX_AUDIO_LOCKS)
    assert lk.acquire(blocking=False)
    try:
        wav = _native_wav(100)
        r = client.post(
            "/api/audio/replace",
            data={"path": "forest.pac", "record": "0", "deploy": "true"},
            files={"file": ("clip.wav", wav, "audio/wav")},
        )
        assert r.status_code == 409, r.text
    finally:
        lk.release()


# ---------------------------------------------------------------------------
# routing — manifest classification (.pac -> audio/PAC, .sfd -> audio/SFD)
# ---------------------------------------------------------------------------
def test_routing_pac_is_audio_pac(tmp_path):
    import manifest
    p = tmp_path / "forest.pac"
    p.write_bytes(_synthetic_pac(1))
    e = manifest.classify(p)
    assert e["category"] == "audio"
    assert e["format"] == "PAC"


def test_routing_sfd_is_audio_sfd(tmp_path):
    import manifest
    p = tmp_path / "opening_j.sfd"
    p.write_bytes(_asf_stub())
    e = manifest.classify(p)
    assert e["category"] == "audio"
    assert e["format"] == "SFD"


def test_routing_wav_is_audio_wav(tmp_path):
    import manifest
    p = tmp_path / "clip.wav"
    p.write_bytes(_native_wav(50))
    e = manifest.classify(p)
    assert e["category"] == "audio"
    assert e["format"] == "WAV"
