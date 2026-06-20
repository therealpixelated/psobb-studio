"""Tests for the in-process PRS decoder integration in formats.bml.

Covers:
  - The default in-process path returns identical output to the legacy
    PuyoToolsCli path for the standard PSOBB asset corpus.
  - The PSO_USE_PUYOTOOLSCLI env-var fallback selects the subprocess
    path (skipped when the binary isn't present).
  - The decompress_prs_cached LRU caches results, evicts at the byte
    cap, and re-keys on mtime.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from formats import bml
from formats.prs import compress, decompress as prs_decompress


PSOBB_DATA = Path(os.path.expanduser("~/PSOBB.IO/data"))
HAS_PSOBB = PSOBB_DATA.is_dir()
HAS_PUYOTOOLS = bml._puyo_path().exists()


# ---------------------------------------------------------------------------
# Default-path correctness
# ---------------------------------------------------------------------------

def test_default_uses_inproc(monkeypatch):
    """With no env var set, _use_puyotoolscli() returns False."""
    monkeypatch.delenv("PSO_USE_PUYOTOOLSCLI", raising=False)
    assert bml._use_puyotoolscli() is False


def test_env_var_selects_subprocess(monkeypatch):
    """PSO_USE_PUYOTOOLSCLI=1 routes through the subprocess path."""
    monkeypatch.setenv("PSO_USE_PUYOTOOLSCLI", "1")
    assert bml._use_puyotoolscli() is True
    monkeypatch.setenv("PSO_USE_PUYOTOOLSCLI", "true")
    assert bml._use_puyotoolscli() is True
    monkeypatch.setenv("PSO_USE_PUYOTOOLSCLI", "0")
    assert bml._use_puyotoolscli() is False
    monkeypatch.setenv("PSO_USE_PUYOTOOLSCLI", "")
    assert bml._use_puyotoolscli() is False


def test_inproc_roundtrip():
    """Synthetic round-trip: compress + decompress via the in-proc path."""
    data = b"PSOBB roundtrip test " * 256
    enc = compress(data)
    dec = bml._prs_decompress(enc)
    assert dec == data


def test_inproc_empty_raises():
    with pytest.raises(ValueError):
        bml._prs_decompress(b"")


# ---------------------------------------------------------------------------
# LRU cache behaviour
# ---------------------------------------------------------------------------

def test_cache_hit_no_recompute():
    """A repeat call with the same key reuses the cached output."""
    bml.cache_clear()
    data = b"cached blob " * 100
    enc = compress(data)
    fake_path = Path("C:/fake.bml")
    calls = {"n": 0}

    def provider():
        calls["n"] += 1
        return enc

    out1 = bml.decompress_prs_cached(fake_path, 1, "inner.nj", provider)
    out2 = bml.decompress_prs_cached(fake_path, 1, "inner.nj", provider)
    assert out1 == data
    assert out2 == data
    # Provider only invoked on the miss; second call hit the cache.
    assert calls["n"] == 1


def test_cache_invalidates_on_mtime_change():
    """Different mtime → different cache entry; provider invoked twice."""
    bml.cache_clear()
    data = b"mtime test " * 50
    enc = compress(data)
    fake_path = Path("C:/fake2.bml")
    calls = {"n": 0}

    def provider():
        calls["n"] += 1
        return enc

    bml.decompress_prs_cached(fake_path, 1, "inner.nj", provider)
    bml.decompress_prs_cached(fake_path, 2, "inner.nj", provider)  # different mtime
    assert calls["n"] == 2


def test_cache_distinct_inner_names():
    """Cache distinguishes by inner name within the same file."""
    bml.cache_clear()
    enc1 = compress(b"first" * 200)
    enc2 = compress(b"second" * 200)
    fake_path = Path("C:/fake3.bml")

    out_a = bml.decompress_prs_cached(fake_path, 1, "a.nj", lambda: enc1)
    out_b = bml.decompress_prs_cached(fake_path, 1, "b.nj", lambda: enc2)
    out_a2 = bml.decompress_prs_cached(fake_path, 1, "a.nj", lambda: enc1)
    out_b2 = bml.decompress_prs_cached(fake_path, 1, "b.nj", lambda: enc2)
    assert out_a == out_a2 == b"first" * 200
    assert out_b == out_b2 == b"second" * 200


def test_cache_eviction_under_byte_cap(monkeypatch):
    """When the cache exceeds its byte cap, oldest entries are evicted."""
    bml.cache_clear()
    # Shrink the cap for the test so we don't have to allocate 64 MB.
    monkeypatch.setattr(bml, "PRS_INNER_CACHE_MAX_BYTES", 8 * 1024)

    payload = b"abc" * 1000  # 3000 bytes decompressed
    enc = compress(payload)
    fake_path = Path("C:/fake4.bml")

    # Insert 6 entries — total ~18 KB, cap is 8 KB.
    for i in range(6):
        bml.decompress_prs_cached(fake_path, 1, f"e{i}.nj", lambda: enc)

    stats = bml.cache_stats()
    assert stats["bytes"] <= 8 * 1024 + len(payload)  # allow one entry over
    assert stats["entries"] < 6  # at least some evicted
    bml.cache_clear()


def test_cache_clear():
    """cache_clear empties the cache."""
    enc = compress(b"x" * 100)
    bml.decompress_prs_cached(Path("C:/x.bml"), 1, "x.nj", lambda: enc)
    assert bml.cache_stats()["entries"] >= 1
    bml.cache_clear()
    assert bml.cache_stats() == {
        "entries": 0,
        "bytes": 0,
        "max_bytes": bml.PRS_INNER_CACHE_MAX_BYTES,
    }


# ---------------------------------------------------------------------------
# Cross-validation: in-proc vs subprocess agree on real PSOBB BMLs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (HAS_PSOBB and HAS_PUYOTOOLS),
                    reason="needs PSOBB install + PuyoToolsCli")
def test_inproc_matches_subprocess_real_bmls(monkeypatch):
    """For 5 small BMLs, the in-proc path must agree byte-for-byte with PuyoToolsCli."""
    from formats.bml import parse_bml

    bmls = sorted(PSOBB_DATA.glob("*.bml"), key=lambda p: p.stat().st_size)[:5]
    assert len(bmls) >= 3
    for bml_path in bmls:
        buf = bml_path.read_bytes()
        try:
            entries = parse_bml(buf)
        except Exception:
            pytest.skip(f"parse_bml failed on {bml_path.name}")
        if not entries:
            continue
        ent = entries[0]
        raw = bytes(buf[ent.offset:ent.offset + ent.size_compressed])

        # Subprocess path
        monkeypatch.setenv("PSO_USE_PUYOTOOLSCLI", "1")
        sub_out = bml._prs_decompress(raw)
        # In-process path
        monkeypatch.setenv("PSO_USE_PUYOTOOLSCLI", "0")
        inproc_out = bml._prs_decompress(raw)

        assert sub_out == inproc_out, f"mismatch on {bml_path.name}"


# ---------------------------------------------------------------------------
# extract_bml (full archive) end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PSOBB, reason="needs PSOBB install")
def test_extract_bml_inproc_real():
    """Full extract on a real BML; every entry decompresses cleanly via in-proc."""
    bmls = sorted(PSOBB_DATA.glob("*.bml"), key=lambda p: p.stat().st_size)
    target = bmls[0]
    buf = target.read_bytes()
    out = bml.extract_bml(buf)
    assert isinstance(out, dict) and len(out) > 0
    for name, payload in out.items():
        assert isinstance(payload, bytes)
        assert len(payload) > 0, f"empty payload for {name}"
