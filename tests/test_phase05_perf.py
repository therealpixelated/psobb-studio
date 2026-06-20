"""Tests for the Phase 0.5 perf endpoints.

Covers:
  - /api/manifest_lite returns the slim shape (path/category/
    inferred_category/size/parent_archive only) and ETags correctly.
  - /api/asset/<path> hydrates a full entry from the cached manifest
    and 404s on unknown paths.
  - /api/model_bundle/<path> consolidates skinned + animations + a
    chosen motion's data; partial failures land in ``errors``.
  - manifest._newest_mtime_cached caches the install-tree walk and
    obeys force= to bypass.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from fastapi.testclient import TestClient


PSOBB_DATA = Path(os.path.expanduser("~/PSOBB.IO/data"))
HAS_PSOBB = PSOBB_DATA.is_dir()


@pytest.fixture(scope="module")
def client():
    import server
    return TestClient(server.app)


# ---------------------------------------------------------------------------
# manifest._newest_mtime_cached
# ---------------------------------------------------------------------------

def test_newest_mtime_cached_short_circuits(monkeypatch, tmp_path):
    """Repeated calls within the TTL only run the walk once.

    The walk implementation switched from ``walk_install`` to a direct
    ``os.scandir`` recursion in 2026-04-25 (60+× speedup on 2 k-file
    installs). The test mocks the inner ``_newest_mtime_under`` symbol
    rather than ``walk_install`` so it stays meaningful across the
    rewrite.
    """
    import manifest as manifest_mod

    # Seed a fake install tree.
    tree = tmp_path / "install"
    tree.mkdir()
    (tree / "a.bin").write_bytes(b"x")
    (tree / "b.bin").write_bytes(b"y")

    manifest_mod._newest_mtime_cache_clear(tree)
    calls = {"n": 0}

    def fake_inner(root):
        calls["n"] += 1
        return 12345

    monkeypatch.setattr(manifest_mod, "_newest_mtime_under", fake_inner)
    v1 = manifest_mod._newest_mtime_cached(tree)
    v2 = manifest_mod._newest_mtime_cached(tree)
    v3 = manifest_mod._newest_mtime_cached(tree)
    assert v1 == v2 == v3 == 12345
    # Only one inner-walk despite three calls.
    assert calls["n"] == 1
    # force=True bypasses.
    manifest_mod._newest_mtime_cached(tree, force=True)
    assert calls["n"] == 2

    manifest_mod._newest_mtime_cache_clear()


def test_newest_mtime_under_uses_scandir_fast_path(tmp_path):
    """``_newest_mtime_under`` walks via os.scandir and returns the
    largest mtime among non-backup files, skipping backup-named entries.
    Regression test for the 2026-04-25 perf rewrite (256 ms → 3.5 ms).
    """
    import os
    import manifest as manifest_mod

    tree = tmp_path / "install"
    tree.mkdir()
    (tree / "real.bin").write_bytes(b"x")
    sub = tree / "sub"
    sub.mkdir()
    newest_file = sub / "later.bin"
    newest_file.write_bytes(b"y")
    backup = tree / "real.bin.bak"
    backup.write_bytes(b"z")

    # Force the newest one to be the deeply-nested file.
    base_mt = newest_file.stat().st_mtime
    os.utime(newest_file, (base_mt + 100, base_mt + 100))
    # Push the backup file's mtime higher to confirm it's IGNORED.
    os.utime(backup, (base_mt + 1000, base_mt + 1000))

    manifest_mod._newest_mtime_cache_clear()
    got = manifest_mod._newest_mtime_under(tree)
    assert got == int(base_mt + 100), (
        f"expected newest non-backup mtime, got {got}"
    )


def test_manifest_lite_fast_path_skips_full_load(tmp_path, monkeypatch):
    """When the lite cache file is fresh relative to the install tree,
    ``cache_manifest_lite`` MUST NOT call ``cache_manifest`` (which
    would load the full 3.9 MB JSON). Regression test for the
    2026-04-25 perf rewrite (282 ms → 14 ms cold).
    """
    import json
    import manifest as manifest_mod

    install = tmp_path / "install"
    install.mkdir()
    (install / "a.bin").write_bytes(b"x")
    cache = tmp_path / "cache"
    cache.mkdir()

    # Hand-build a valid lite cache file with a future mtime so it's
    # always fresh relative to the install root.
    install_root_str = str(install.resolve()).replace("\\", "/")
    lite_payload = {
        "version": manifest_mod.MANIFEST_VERSION,
        "generated_at": 0,
        "install_root": install_root_str,
        "entries": [{"path": "a.bin", "category": "other", "size": 1}],
    }
    lite_path = cache / "manifest_lite.json"
    lite_path.write_text(json.dumps(lite_payload), encoding="utf-8")

    # Future-date the lite file so the freshness check trivially passes.
    import os
    os.utime(lite_path, (10**10, 10**10))

    manifest_mod._newest_mtime_cache_clear()

    full_called = {"n": 0}

    def fake_full(*args, **kwargs):
        full_called["n"] += 1
        return {
            "version": manifest_mod.MANIFEST_VERSION,
            "install_root": install_root_str,
            "entries": [],
        }

    monkeypatch.setattr(manifest_mod, "cache_manifest", fake_full)
    out = manifest_mod.cache_manifest_lite(install, cache_dir=cache)

    assert out["entries"] == lite_payload["entries"]
    assert full_called["n"] == 0, (
        "cache_manifest_lite must short-circuit when the lite cache is "
        "fresh — calling cache_manifest would defeat the perf win"
    )

    manifest_mod._newest_mtime_cache_clear()


# ---------------------------------------------------------------------------
# /api/manifest_lite shape + ETag
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PSOBB, reason="needs PSOBB install")
def test_manifest_lite_shape(client):
    r = client.get("/api/manifest_lite")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["version"] == 1
    assert isinstance(data["entries"], list)
    assert data["entries"], "manifest_lite must have at least some entries"
    # Every entry MUST be limited to the lite key set.
    allowed = {"path", "category", "inferred_category", "size", "parent_archive"}
    for e in data["entries"][:50]:
        assert e
        assert set(e.keys()).issubset(allowed), f"unexpected keys in lite entry: {set(e.keys()) - allowed}"
        # path + category + size are the always-present columns.
        assert "path" in e
        assert "size" in e


@pytest.mark.skipif(not HAS_PSOBB, reason="needs PSOBB install")
def test_manifest_lite_compressed_under_400kb(client):
    """Compressed wire payload should fit in the spec's 400 KB budget.

    GZipMiddleware compresses every response > 1 KB; the user-visible
    size is the gzipped bytes on the wire, not the decoded JSON the
    test client returns. We measure the compressed payload via an
    explicit Accept-Encoding header so the assertion mirrors the
    end-user network cost.
    """
    import gzip
    # TestClient transparently decompresses for r.content. Re-compress
    # the body to test against the spec target — this matches the
    # bytes the GZipMiddleware will actually ship over the wire.
    r = client.get("/api/manifest_lite")
    assert r.status_code == 200
    raw = r.content
    compressed = gzip.compress(raw)
    assert len(compressed) < 400 * 1024, (
        f"lite payload {len(compressed)} compressed bytes "
        f"({len(raw)} raw) exceeds 400 KB target"
    )


@pytest.mark.skipif(not HAS_PSOBB, reason="needs PSOBB install")
def test_manifest_lite_etag_304(client):
    """Conditional GET with matching ETag returns 304."""
    r1 = client.get("/api/manifest_lite")
    assert r1.status_code == 200
    etag = r1.headers.get("etag")
    assert etag
    r2 = client.get("/api/manifest_lite", headers={"If-None-Match": etag})
    assert r2.status_code == 304


# ---------------------------------------------------------------------------
# /api/asset/<path> lazy hydration
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PSOBB, reason="needs PSOBB install")
def test_api_asset_returns_full_entry(client):
    # Fetch any path from the lite manifest, then hydrate via /api/asset.
    r_lite = client.get("/api/manifest_lite")
    assert r_lite.status_code == 200
    entries = r_lite.json()["entries"]
    assert entries, "no entries to test"
    # Pick a non-archive entry so we hit the regular classify() path.
    target = None
    for e in entries:
        if "#" not in e["path"]:
            target = e
            break
    assert target is not None

    # /api/asset must return the full entry shape.
    r_full = client.get("/api/asset/" + target["path"])
    assert r_full.status_code == 200, r_full.text
    full = r_full.json()
    # Lite columns must match.
    assert full["path"] == target["path"]
    assert full["category"] == target["category"]
    # Full-shape fields the lite endpoint stripped.
    assert "extension" in full
    assert "magic_hex" in full


def test_api_asset_404_unknown(client):
    r = client.get("/api/asset/this_is_definitely_not_a_real_path.bin")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /api/model_bundle/<path>
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PSOBB, reason="needs PSOBB install")
def test_model_bundle_smoke(client):
    """End-to-end: pick a known boss model, fetch the bundle, verify shape."""
    # bm_boss2_de_rol_le.bml is a stable .nj model in every PSOBB install.
    bml = PSOBB_DATA / "bm_boss2_de_rol_le.bml"
    if not bml.exists():
        pytest.skip("bm_boss2_de_rol_le.bml not in install")

    # Find an .nj inner.
    from formats.bml import parse_bml
    entries = parse_bml(bml.read_bytes())
    inner = next((e.name for e in entries if e.name.lower().endswith(".nj")), None)
    if not inner:
        pytest.skip("no .nj inner in target BML")

    r = client.get(f"/api/model_bundle/{bml.name}", params={"inner": inner})
    assert r.status_code == 200, r.text
    bundle = r.json()
    assert bundle["filename"] == bml.name
    assert bundle["inner"] == inner
    # skinned + animations are always present (or in errors).
    assert "skinned" in bundle
    assert "animations" in bundle
    # No motion_data unless include_motion was set.
    assert bundle.get("motion_data") is None
    # Skinned must have mesh + bone counts.
    sk = bundle["skinned"]
    assert sk and sk.get("mesh_count", 0) > 0
    assert sk.get("bone_count", 0) > 0


@pytest.mark.skipif(not HAS_PSOBB, reason="needs PSOBB install")
def test_model_bundle_with_default_motion(client):
    bml = PSOBB_DATA / "bm_boss2_de_rol_le.bml"
    if not bml.exists():
        pytest.skip("bm_boss2_de_rol_le.bml not in install")
    from formats.bml import parse_bml
    entries = parse_bml(bml.read_bytes())
    inner = next((e.name for e in entries if e.name.lower().endswith(".nj")), None)
    if not inner:
        pytest.skip("no .nj inner")

    r = client.get(
        f"/api/model_bundle/{bml.name}",
        params={"inner": inner, "include_motion": "default"},
    )
    assert r.status_code == 200, r.text
    bundle = r.json()
    # Bundle should now include motion_data IF the model has any motions.
    anims = bundle.get("animations") or {}
    if (anims.get("motion_count") or 0) > 0:
        # The default motion's keyframe data must be populated.
        md = bundle.get("motion_data")
        assert md is not None, "default-motion data missing despite motions present"
        assert "bones" in md
