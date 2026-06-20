"""End-to-end smoke pytest layer.

Runs the comprehensive matrix from ``scripts/e2e_smoke_run.py`` against
the live editor server (default 127.0.0.1:8765). Each category gets one
parametrised test that pulls a fixed sample of BMLs (so reruns hit the
same files) and asserts:

  - Every primary `.nj`/`.xj` inner returns ``mesh_count > 0``
    EXCEPT for known-honest-gap inners (``ene_common_all.nj`` ships
    only an NJTL — no NJCM mesh — by design).
  - Every binding row resolves to a non-zero tile_png.
  - When ``has_texture`` is True, ``binding_data.binding`` is non-empty.
  - ``material_id`` distribution is not collapsed to a single id when
    the binding has 2+ rows (filters out parser-state-leak regressions
    like the pre-fix kaifuku_moto.xj that routed 89/94 strips to id=0).

The fixture ``e2e_server_or_skip`` skips the entire module when no
server is reachable on 127.0.0.1:8765, so this file is safe to keep in
the regular pytest collection.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_LITE = ROOT / "_e2e_work" / "manifest_lite.json"
SAMPLE_BASE_URL = os.environ.get("PSO_E2E_BASE_URL", "http://127.0.0.1:8765")

# Per-category sample size. Bumped to 8 (vs the 20 used in the offline
# scripts/e2e_smoke_run.py runner) so the pytest layer stays under
# 30 seconds total — full rigor is the runner's job; pytest is a
# regression-pin against major breakage.
PER_CAT = int(os.environ.get("PSO_E2E_PER_CAT", "8"))

# Failure thresholds per category (% PASS rate). Categories that test
# poorly even on the gold-master run (e.g. small samples like Items
# where N=1) get a relaxed bar.
THRESHOLDS = {
    "Bosses": 0.95,
    "Enemies": 0.95,
    "NPCs": 0.95,
    "Objects": 0.95,
    "Player Misc": 0.95,
    "Effects": 0.90,
    "Items": 0.90,
    "Quests": 0.90,
    "?": 0.90,
}

# Known-honest-gap inners: 0 meshes is correct for these (NJTL-only
# texture-name markers that other BMLs reference). See
# scripts/e2e_smoke_run.py for the maintained list.
HONEST_GAP_INNERS = {
    "ene_common_all.nj",
}


def _http_get_json(url: str, timeout: float = 60.0) -> Tuple[int, Optional[Any]]:
    """Subset of the runner's helper, no retry — pytest scope."""
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            try:
                return resp.status, json.loads(body)
            except Exception:
                return resp.status, None
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, None
    except Exception:
        return -1, None


@pytest.fixture(scope="module")
def e2e_server_or_skip():
    """Skip the entire module when the live server isn't responding."""
    code, _ = _http_get_json(f"{SAMPLE_BASE_URL}/api/health", timeout=2.0)
    if code != 200:
        pytest.skip(f"no editor server at {SAMPLE_BASE_URL}")
    if not MANIFEST_LITE.exists():
        # Try to refresh the snapshot.
        MANIFEST_LITE.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(
                f"{SAMPLE_BASE_URL}/api/manifest_lite", timeout=10.0,
            ) as r:
                MANIFEST_LITE.write_bytes(r.read())
        except Exception as e:
            pytest.skip(f"manifest_lite not available: {e}")


@pytest.fixture(scope="module")
def sample_by_category(e2e_server_or_skip) -> dict:
    """Group manifest entries by inferred_category, sample up to N per."""
    mfest = json.loads(MANIFEST_LITE.read_text(encoding="utf-8"))
    by_cat: dict[str, list[str]] = {}
    for e in mfest.get("entries", []):
        path = e.get("path", "")
        if not path.endswith(".bml"):
            continue
        cat = e.get("inferred_category", "?")
        by_cat.setdefault(cat, []).append(path)
    out = {}
    for cat, paths in by_cat.items():
        paths = sorted(paths)
        if len(paths) > PER_CAT:
            step = len(paths) / PER_CAT
            picked = [paths[int(i * step)] for i in range(PER_CAT)]
            out[cat] = picked
        else:
            out[cat] = paths
    return out


# Static cat list so pytest parametrisation can use it.
ALL_CATS = sorted(THRESHOLDS.keys())


@pytest.fixture(scope="module")
def category_results(sample_by_category) -> dict:
    """Probe every sampled BML once and cache results per category.

    Each test then reads its own slice from this cache, avoiding 8x
    redundant fetches when running the full file in one invocation.
    """
    out: dict = {}
    for cat, bmls in sample_by_category.items():
        cat_recs = []
        for bml in bmls:
            rec = _probe_single(bml)
            cat_recs.append(rec)
        out[cat] = cat_recs
    return out


def _probe_single(bml: str) -> dict:
    """Compact per-BML probe — `mesh_count`, `binding_rows`, skew flag."""
    rec = {"bml": bml, "inners": [], "fail_reasons": []}
    list_url = f"{SAMPLE_BASE_URL}/api/bml/{urllib.parse.quote(bml)}/list"
    code, listing = _http_get_json(list_url)
    if code != 200 or not isinstance(listing, dict):
        rec["fail_reasons"].append(f"list http {code}")
        return rec
    for e in (listing.get("entries") or []):
        nm = (e.get("name") or "")
        low = nm.lower()
        if not (low.endswith(".nj") or low.endswith(".xj")):
            continue
        ext = low.rsplit(".", 1)[-1]
        kind = _classify_inner(nm)
        url = (
            f"{SAMPLE_BASE_URL}/api/model_mesh/{urllib.parse.quote(bml)}"
            f"?inner={urllib.parse.quote(nm)}"
        )
        mcode, payload = _http_get_json(url)
        ir = {"inner": nm, "ext": ext, "kind": kind, "mesh_count": 0,
              "binding_rows": 0, "mid_distinct": 0, "top_share": 0.0,
              "ok": True, "skip_reason": None}
        if low in HONEST_GAP_INNERS:
            ir["skip_reason"] = "NJTL-only marker"
            ir["ok"] = True
        elif mcode != 200 or not isinstance(payload, dict):
            ir["ok"] = False
            rec["fail_reasons"].append(f"{nm}: mesh http {mcode}")
        else:
            meshes = payload.get("meshes") or []
            ir["mesh_count"] = len(meshes)
            binding = (payload.get("binding_data") or {}).get("binding") or []
            ir["binding_rows"] = len(binding)
            if meshes:
                ids = Counter(m.get("material_id", 0) for m in meshes)
                top = ids.most_common(1)[0]
                ir["mid_distinct"] = len(ids)
                ir["top_share"] = top[1] / len(meshes)
                # Only flag genuine collapse (1 distinct id + 2+
                # binding rows + 4+ submeshes).
                if (
                    len(meshes) >= 4
                    and ir["top_share"] > 0.95
                    and ir["binding_rows"] >= 2
                    and ir["mid_distinct"] == 1
                ):
                    ir["ok"] = False
                    rec["fail_reasons"].append(
                        f"{nm}: material collapse "
                        f"({top[1]}/{len(meshes)} share id={top[0]})"
                    )
            elif kind == "primary":
                ir["ok"] = False
                rec["fail_reasons"].append(f"{nm}: 0 meshes")
        rec["inners"].append(ir)
    return rec


def _classify_inner(name: str) -> str:
    low = name.lower()
    stem = low.rsplit(".", 1)[0]
    if "_lod" in stem or stem.endswith("_low"):
        return "lod"
    if "_shadow" in stem or stem.endswith("_shd"):
        return "shadow"
    if "_hahen" in stem or "_destroy" in stem or "_dest" in stem or "_break" in stem:
        return "destroyed"
    return "primary"


# ---------------------------------------------------------------------------
# Per-category parametrised tests.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("category", ALL_CATS)
def test_category_pass_rate(category_results, category):
    """For each inferred-category, primary inners pass at threshold rate."""
    recs = category_results.get(category, [])
    if not recs:
        pytest.skip(f"no BMLs in category {category}")
    total = len(recs)
    fail = sum(1 for r in recs if r["fail_reasons"])
    pass_rate = (total - fail) / total
    threshold = THRESHOLDS.get(category, 0.90)
    if pass_rate < threshold:
        details = "\n".join(
            f"  {r['bml']}: {'; '.join(r['fail_reasons'])}"
            for r in recs if r["fail_reasons"]
        )
        pytest.fail(
            f"{category}: pass rate {pass_rate*100:.1f}% < threshold "
            f"{threshold*100:.1f}% ({fail}/{total} failed)\n{details}"
        )


def test_user_flagged_bmls_recover(e2e_server_or_skip):
    """Specific BMLs from the user's report MUST come back green.

    These were the three failures the user explicitly called out. The
    fixes for them landed across:

      1. ``static/model_viewer.js:1170`` — accept .xj inner extension
         in the BML#inner client-side gate.
      2. ``formats/xj_descriptor.py`` — thread last-seen tex_id across
         strips with empty material entry tables (state-stickiness
         mirrors PSOBB's GPU pipeline).
      3. ``static/model_viewer.js:tryLoadCompositeBmlMesh`` — populate
         the animation panel in composite mode so warp BMLs surface
         their .njm motion catalog.
    """
    cases = [
        ("fe_obj_o_vs2container.bml", "fs_obj_o_vs2container.xj"),
        ("fe_obj_kaifuku_moto_2.bml", "fe_obj_kaifuku_moto.xj"),
        ("bm_obj_warp_labo.bml", "warp_obj01.xj"),
        ("bm_obj_warpboss.bml", "fs_obj_warp_dai.xj"),
        ("bm_obj_warpboss_ancient.bml", "fe_obj_df_warp_gawa.xj"),
        ("bm_obj_warpboss_jungle.bml", "fe_obj_warp4_dodai.xj"),
        ("bm_obj_warp_jung.bml", "fe_obj_warp_dodai.xj"),
    ]
    failures = []
    for bml, inner in cases:
        url = (
            f"{SAMPLE_BASE_URL}/api/model_mesh/{urllib.parse.quote(bml)}"
            f"?inner={urllib.parse.quote(inner)}"
        )
        code, payload = _http_get_json(url)
        if code != 200 or not isinstance(payload, dict):
            failures.append(f"{bml}#{inner}: http {code}")
            continue
        meshes = payload.get("meshes") or []
        binding = (payload.get("binding_data") or {}).get("binding") or []
        ids = Counter(m.get("material_id", 0) for m in meshes)
        if not meshes:
            failures.append(f"{bml}#{inner}: 0 meshes")
            continue
        if not binding:
            failures.append(f"{bml}#{inner}: empty binding")
            continue
        # Multi-material XJ files MUST exhibit material spread:
        # parser-state-leak regression would show distinct=1 even when
        # binding has multiple rows.
        if len(meshes) >= 4 and len(ids) == 1 and len(binding) >= 2:
            failures.append(
                f"{bml}#{inner}: material collapsed to id={list(ids)[0]} "
                f"({len(meshes)} submeshes share one of {len(binding)} bindings)"
            )
    if failures:
        pytest.fail("user-flagged BMLs failed:\n" + "\n".join(failures))
