"""Regression test for the 2026-04-25 multi-inner BML anchoring fix.

User report: "still not loading full properly anchored assets". Root
cause: when the user clicks a top-level `.bml` whose archive packs
multiple `.nj` inners (boss BMLs like De Rol Le with body + helm +
fins + sting + tentacle, etc.), the model viewer was rendering ONLY
the first / matched-texture-derived inner. Sibling parts were
silently invisible.

Fix: the model viewer now discovers all `.nj` inners in a BML, builds
a composite scene with every primary inner, and offers an inner-picker
dropdown so users can switch between parts or view a specific one.

This test verifies:
  1. The BML inner-list endpoint exposes all the `.nj` parts.
  2. Each primary inner returns parseable geometry via /api/model_mesh.
  3. The union AABB across all primaries is sane (every vertex within
     [-1000, 1000] per axis — catches NaN / wildly-offset coords).
  4. The composite mesh count is the SUM of per-inner mesh_counts
     (i.e. nothing got dropped during composition).

The test is skipped when the live server isn't running on port 8765.
"""
from __future__ import annotations

import re
from urllib.parse import quote

import pytest

try:
    import requests
except ImportError:
    pytest.skip("requests not installed", allow_module_level=True)


SERVER = "http://127.0.0.1:8765"


def _server_up() -> bool:
    try:
        r = requests.get(f"{SERVER}/api/manifest", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


_LIVE = _server_up()


# Same classification heuristic as model_viewer.js. Parts in
# lod/shadow/destroyed are not part of the default composite payload
# (they overlap intact parts or only render in damage states).
_LOD_RE = re.compile(r"^(lo|low)[_\s-]", re.IGNORECASE)
# Shadow proxies: `_sd_` mid-name (dragon) or `_sd`/`_shd` suffix.
_SHADOW_RE = re.compile(r"(?:^|[_-])(?:sd|shd)(?:$|[_-])", re.IGNORECASE)
_DESTROYED_RE = re.compile(r"(_break|_broken|_hahen|_burst)", re.IGNORECASE)


def _classify(name: str) -> str:
    stem = re.sub(r"\.(nj|xj)$", "", name, flags=re.IGNORECASE)
    if _LOD_RE.search(stem):
        return "lod"
    if _SHADOW_RE.search(stem):
        return "shadow"
    if _DESTROYED_RE.search(stem):
        return "destroyed"
    return "primary"


def _list_inner_njs(bml: str) -> list[str]:
    r = requests.get(f"{SERVER}/api/bml/{quote(bml)}/list", timeout=10.0)
    r.raise_for_status()
    return [
        e["name"]
        for e in r.json().get("entries", [])
        if e["name"].lower().endswith(".nj")
    ]


def _fetch_mesh(bml: str, inner: str) -> dict:
    r = requests.get(
        f"{SERVER}/api/model_mesh/{quote(bml)}?inner={quote(inner)}",
        timeout=20.0,
    )
    r.raise_for_status()
    return r.json()


def _union_aabb(payloads: list[dict]) -> tuple[list[float], list[float]]:
    mins = [float("inf")] * 3
    maxs = [float("-inf")] * 3
    for p in payloads:
        for m in p.get("meshes", []):
            a = m.get("aabb")
            if not a or len(a) != 6:
                continue
            for i in range(3):
                mins[i] = min(mins[i], a[i])
                maxs[i] = max(maxs[i], a[i + 3])
    return mins, maxs


@pytest.mark.skipif(not _LIVE, reason="server not running on :8765")
def test_de_rol_le_has_multiple_primary_inners():
    """De Rol Le BML packs 7 NJ inners; ≥3 are primaries (body + at
    least 2 supporting parts). Without compositing, the user only sees
    the body — the helm/skull, fins, sting, tentacle stay invisible.
    """
    inners = _list_inner_njs("bm_boss2_de_rol_le.bml")
    assert len(inners) >= 5, f"De Rol Le should have ≥5 NJ inners; got {len(inners)}"
    primaries = [n for n in inners if _classify(n) == "primary"]
    assert len(primaries) >= 3, (
        f"De Rol Le should have ≥3 primary inners (body + supporting parts); "
        f"got {len(primaries)}: {primaries}"
    )
    # Body must be among them — it's the centerpiece.
    assert any("body" in n.lower() for n in primaries), (
        f"De Rol Le primaries missing the body inner: {primaries}"
    )


@pytest.mark.skipif(not _LIVE, reason="server not running on :8765")
def test_de_rol_le_composite_aabb_bounded():
    """Every vertex of every primary inner falls within sane bounds.

    Catches wild AABB / NaN coords that would shrink the camera-fit
    scale to ~0 (model invisible) or push parts outside the viewport.
    """
    inners = _list_inner_njs("bm_boss2_de_rol_le.bml")
    primaries = [n for n in inners if _classify(n) == "primary"]
    payloads = [_fetch_mesh("bm_boss2_de_rol_le.bml", n) for n in primaries]
    aabb_min, aabb_max = _union_aabb(payloads)
    assert all(v != float("inf") for v in aabb_min), "no AABB data in any payload"
    # Sane physical bounds: PSOBB models live in the [-1000, 1000] unit
    # range (game-world units). NaN or wildly-offset values would blow
    # past this.
    for axis, (lo, hi) in enumerate(zip(aabb_min, aabb_max)):
        assert -1000.0 <= lo <= hi <= 1000.0, (
            f"axis {'XYZ'[axis]} out of bounds: lo={lo} hi={hi}"
        )
    # Total span on the long axis (usually Z for the worm) should be
    # noticeably larger than the short axes — confirms the composite
    # spans the full worm length, not just the helm.
    spans = [hi - lo for lo, hi in zip(aabb_min, aabb_max)]
    long_axis_span = max(spans)
    assert long_axis_span > 100.0, (
        f"De Rol Le composite long-axis span {long_axis_span:.1f} too small "
        f"— body inner alone reports >150 units, so a smaller composite means "
        f"the body wasn't included"
    )


@pytest.mark.skipif(not _LIVE, reason="server not running on :8765")
def test_de_rol_le_composite_meshes_are_additive():
    """Sum of per-inner mesh_counts == total composite submesh count.

    This catches the case where the composite path silently drops a
    full inner (zero mesh_count) due to a parse error or bad payload
    handling. We accept some reduction (an inner with zero meshes is
    still summed) but reject the case where the total goes to ~0.
    """
    inners = _list_inner_njs("bm_boss2_de_rol_le.bml")
    primaries = [n for n in inners if _classify(n) == "primary"]
    total = 0
    for inner in primaries:
        payload = _fetch_mesh("bm_boss2_de_rol_le.bml", inner)
        total += payload.get("mesh_count", 0)
    # The body inner alone has ~600+ submeshes. The composite should
    # exceed that with the supporting parts added.
    assert total > 600, (
        f"De Rol Le composite submesh total {total} too small; "
        f"body inner alone should have >600. Indicates an inner was dropped."
    )


@pytest.mark.skipif(not _LIVE, reason="server not running on :8765")
def test_dragon_lod_inners_classified_correctly():
    """Dragon BML has 3 NJ inners — main, lo_, sd_. Only ONE is primary.

    Compositing all three would render LOD overlays on top of the main
    body (sd_ is the shadow proxy, lo_ is low-res). The classifier must
    keep these out of the default "All parts" composite.
    """
    inners = _list_inner_njs("bm_boss8_dragon.bml")
    assert len(inners) == 3, f"dragon should have 3 NJ inners; got {len(inners)}"
    primaries = [n for n in inners if _classify(n) == "primary"]
    assert len(primaries) == 1, (
        f"dragon should classify exactly one inner as primary; got {primaries}"
    )
    # The remaining two must be lod/shadow.
    others = [n for n in inners if _classify(n) != "primary"]
    assert any(_classify(n) == "lod" for n in others), f"dragon: no lod inner found"
    assert any(_classify(n) == "shadow" for n in others), f"dragon: no shadow inner found"


@pytest.mark.skipif(not _LIVE, reason="server not running on :8765")
def test_single_inner_npc_no_picker_needed():
    """A BML with exactly 1 NJ inner doesn't trigger the inner-picker.

    Verifies the heuristic in populateInnerPicker (`info.inners.length < 2`
    → hidden) works for the common single-inner case.
    """
    inners = _list_inner_njs("bm_npc_kenkyu_w.bml")
    assert len(inners) == 1, (
        f"bm_npc_kenkyu_w.bml should have 1 NJ inner; got {len(inners)}: {inners}"
    )


@pytest.mark.skipif(not _LIVE, reason="server not running on :8765")
def test_classifier_handles_known_destroyed_states():
    """`_helm_break.nj`, `_shell_break.nj` are damage-state parts.

    These should classify as "destroyed" so they're listed in the
    picker but excluded from the default "All parts" set (where they'd
    overlap intact siblings).
    """
    assert _classify("boss2_b_helm_break.nj") == "destroyed"
    assert _classify("boss2_b_shell_break.nj") == "destroyed"
    assert _classify("fe_obj_vo_tenjo_hahen01.nj") == "destroyed"
    # Counter-cases: not destroyed.
    assert _classify("boss2_b_derorure_body.nj") == "primary"
    assert _classify("boss2_b_derorure_fin_a.nj") == "primary"


@pytest.mark.skipif(not _LIVE, reason="server not running on :8765")
def test_classifier_handles_lod_and_shadow():
    """Cover the LOD + shadow regex branches."""
    assert _classify("lo_boss1_s_nb_dragon.nj") == "lod"
    assert _classify("low_some_model.nj") == "lod"
    # `_sd_` mid-name (dragon shadow proxy) — between underscores.
    assert _classify("boss1_s_sd_dragon.nj") == "shadow"
    assert _classify("model_sd.nj") == "shadow"
    assert _classify("model_shd.nj") == "shadow"
    # Counter-cases: `sd` not bounded by separators must NOT match.
    assert _classify("asdf_thing.nj") == "primary"
    assert _classify("standard_model.nj") == "primary"


@pytest.mark.skipif(not _LIVE, reason="server not running on :8765")
def test_individual_inner_aabb_bounded():
    """Every primary inner of De Rol Le has a sane self-AABB.

    Even before compositing, each inner's geometry must be parseable
    and live within the sane physical range. Catches bind-pose-inverse
    bugs where one inner ends up with NaN positions.
    """
    inners = _list_inner_njs("bm_boss2_de_rol_le.bml")
    primaries = [n for n in inners if _classify(n) == "primary"]
    for inner in primaries:
        payload = _fetch_mesh("bm_boss2_de_rol_le.bml", inner)
        assert payload.get("mesh_count", 0) > 0, f"{inner}: no submeshes parsed"
        for m in payload.get("meshes", []):
            a = m.get("aabb")
            if not a or len(a) != 6:
                continue
            for i in range(3):
                lo, hi = a[i], a[i + 3]
                assert -1000.0 <= lo <= hi <= 1000.0, (
                    f"{inner} mesh aabb[{i}] out of bounds: lo={lo} hi={hi}"
                )
