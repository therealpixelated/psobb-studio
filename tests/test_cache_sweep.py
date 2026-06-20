"""Unit tests for `server._sweep_cache_dir`.

Audit M-3 (2026-05-01) added a periodic age + total-size sweep to the
disk tier of `parse_cache` and `bml_inner`. Both caches were previously
unbounded on disk. The helper walks a directory, deletes files older
than `max_age_days` first, then deletes oldest-first by mtime if the
remaining total still exceeds `max_total_bytes`.

These tests build synthetic files of known size + mtime in a tmp dir
and verify each sweep branch in isolation:

  - empty / missing dir → no-op, zeroed stats dict
  - age sweep deletes only old files
  - size sweep deletes oldest-first when over the cap
  - combined: age first, then size on the survivors
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def srv():
    import server
    return server


def _mkfile(p: Path, size: int, age_days: float) -> Path:
    """Create a file of `size` bytes with mtime `age_days` ago. Returns p."""
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00" * size)
    target = time.time() - (age_days * 86400.0)
    os.utime(p, (target, target))
    return p


def test_missing_dir_returns_zero_stats(srv, tmp_path):
    """A nonexistent dir yields a zeroed stats dict, never raises."""
    out = srv._sweep_cache_dir(
        tmp_path / "does_not_exist", 1024 * 1024, 30,
    )
    assert out == {"deleted_count": 0, "freed_bytes": 0, "remaining_bytes": 0}


def test_age_sweep_drops_old_keeps_fresh(srv, tmp_path):
    """Files older than max_age_days are deleted; younger files survive."""
    old1 = _mkfile(tmp_path / "old1.bin", 1024, age_days=45)
    old2 = _mkfile(tmp_path / "old2.bin", 2048, age_days=60)
    fresh = _mkfile(tmp_path / "fresh.bin", 4096, age_days=5)
    # cap is huge so only the age branch fires.
    out = srv._sweep_cache_dir(tmp_path, 1024 * 1024 * 1024, 30)
    assert out["deleted_count"] == 2
    assert out["freed_bytes"] == 1024 + 2048
    assert not old1.exists()
    assert not old2.exists()
    assert fresh.exists()
    # remaining_bytes only counts SURVIVORS; the fresh file is 4096.
    assert out["remaining_bytes"] == 4096


def test_size_sweep_drops_oldest_first(srv, tmp_path):
    """Under-age but over-cap → delete oldest-first until under cap."""
    # All within max_age_days so age sweep is a no-op.
    a = _mkfile(tmp_path / "a.bin", 1000, age_days=10)
    b = _mkfile(tmp_path / "b.bin", 1000, age_days=5)
    c = _mkfile(tmp_path / "c.bin", 1000, age_days=1)
    # Cap is 1500 → must delete two of the three. `a` (oldest) goes
    # first, then `b`, leaving `c` (1000 bytes ≤ 1500).
    out = srv._sweep_cache_dir(tmp_path, 1500, 30)
    assert out["deleted_count"] == 2
    assert out["freed_bytes"] == 2000
    assert not a.exists()
    assert not b.exists()
    assert c.exists()
    assert out["remaining_bytes"] == 1000


def test_age_then_size_combined(srv, tmp_path):
    """Age sweep first, then size sweep on survivors."""
    # One ancient file (deleted by age regardless).
    ancient = _mkfile(tmp_path / "ancient.bin", 500, age_days=100)
    # Three within-age files: oldest will be evicted by size sweep.
    keep_old = _mkfile(tmp_path / "kold.bin", 800, age_days=20)
    mid = _mkfile(tmp_path / "kmid.bin", 800, age_days=10)
    young = _mkfile(tmp_path / "kyoung.bin", 800, age_days=2)
    # Age cap = 30 days (drops `ancient` only).
    # Size cap = 1700 bytes → after age sweep total is 2400 → must drop
    # `keep_old` (oldest survivor at 800) leaving 1600 bytes.
    out = srv._sweep_cache_dir(tmp_path, 1700, 30)
    assert not ancient.exists()
    assert not keep_old.exists()
    assert mid.exists()
    assert young.exists()
    assert out["deleted_count"] == 2
    assert out["freed_bytes"] == 500 + 800
    assert out["remaining_bytes"] == 1600


def test_recursive_walk_finds_nested(srv, tmp_path):
    """Files in subdirs are swept too — bml_inner uses <digest>/<file> layout."""
    nested = _mkfile(tmp_path / "sub1" / "deep" / "file.bin", 2000, age_days=50)
    out = srv._sweep_cache_dir(tmp_path, 1024 * 1024, 30)
    assert not nested.exists()
    assert out["deleted_count"] == 1
    assert out["freed_bytes"] == 2000
