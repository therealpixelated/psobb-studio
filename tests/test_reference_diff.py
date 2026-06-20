"""Wire-format pins for the Phantasmal-style renderer interface.

This suite asserts that ``/api/model_bundle/<bml>#<inner>`` returns the
exact shape a Phantasmal World renderer would consume, so the four
in-flight regression-hunt agents (anim-preview, texture-binding,
viewport, model-deep-debug) can apply targeted fixes without
re-discovering the contract.

Scope: API contract diff against the reference renderer's expectations
(`web/src/jsMain/kotlin/world/phantasmal/web/viewer/...` from
DaanVandenBosch/phantasmal-world). NOT a behavioural / visual diff.

Gated by ``PSO_REF_DIFF_TESTS=1`` so CI doesn't break when the manifest
evolves. Run locally with::

    PSO_REF_DIFF_TESTS=1 pytest tests/test_reference_diff.py -v

See ``_reports/regression_diff_vs_phantasmal.md`` for the full diff
report. The reference Kotlin source files live verbatim under
``_reports/reference_phantasmal_extracts/``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


pytestmark = pytest.mark.skipif(
    os.environ.get("PSO_REF_DIFF_TESTS") != "1",
    reason="PSO_REF_DIFF_TESTS=1 not set — gated reference-diff suite",
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    """In-process FastAPI client. Imports server.py once per module."""
    import server  # noqa: F401  (sets up app at import time)
    return TestClient(server.app)


# ---------------------------------------------------------------------------
# Surveyed BMLs from AGENT_TEXTURE_BINDING_REPORT.md. Each row's binding
# count + name_match status was captured live against the same data
# directory used in production. We re-run the same probes here so the
# regression-hunt agents can see drift.
# ---------------------------------------------------------------------------

# (path, inner, expected_binding_count, expected_name_match)
SURVEYED_MODELS = [
    ("bm4_ps_ma_body.bml", "bm4_ps_ma_body.nj", 10, True),
    ("bm_boss1_dragon.bml", "boss1_s_nb_dragon.nj", 16, True),
    ("bm_boss8_dragon.bml", "boss1_s_nb_dragon.nj", 13, True),
    ("bm_boss5_gryphon.bml", "boss5_s_body.nj", 6, True),
    ("bm_ene_gibbles_low.bml", "lo_gibb_body.nj", 3, True),
]


def _bundle_url(bml: str, inner: str) -> str:
    """``/api/model_bundle/<bml>#<inner>`` with proper percent-encoding."""
    from urllib.parse import quote
    target = f"{bml}#{inner}"
    return f"/api/model_bundle/{quote(target, safe='')}"


# ---------------------------------------------------------------------------
# Contract: the bundle endpoint MUST return the keys a Phantasmal-style
# renderer expects. If the backend ever drops `binding_data` or renames
# `bones[i].rotation_bams`, the reference renderer port we plan can't
# work — these tests catch that BEFORE the four agents apply fixes.
# ---------------------------------------------------------------------------


def _assert_phantasmal_skinned_shape(skinned: dict) -> None:
    """Assert ``skinned`` payload matches Phantasmal's NjObject contract.

    Phantasmal's ``ninjaObjectToSkinnedMesh`` consumes a tree of NjObject
    nodes where each ``model.meshes[i]`` carries ``textureId`` and
    ``vertices[]`` w/ ``boneWeights[4]`` and ``boneIndices[4]``. Our wire
    format is denormalised but each field has a 1:1 reference equivalent.
    """
    # Top-level skinned shape.
    for k in ("mesh_count", "meshes", "bones", "bone_count", "vert_total",
              "tri_total", "vertices_pre_transformed", "has_bone_indices"):
        assert k in skinned, f"missing skinned.{k!r} (Phantasmal renderer needs it)"

    # binding_data is OUR addition (Phantasmal does it in-process via
    # textureIds() + xvrTextures.getOrNull(idx)). Backend should still
    # surface it for the diagnostic banner.
    assert "binding_data" in skinned, "binding_data missing — texture-binding agent's fix relied on this"
    bd = skinned["binding_data"]
    for k in ("njtl", "xvmh", "binding", "name_match"):
        assert k in bd, f"binding_data.{k!r} missing"

    # Bone shape — Phantasmal's NjObject has position/rotation/scale +
    # eval_flags. We use rotation_bams (BAMS-encoded), reference uses
    # raw radians. The renderer port handles the conversion in
    # NinjaGeometryConversion.kt by detecting the eval_flags bit.
    assert isinstance(skinned["bones"], list)
    if skinned["bones"]:
        b0 = skinned["bones"][0]
        for k in ("index", "parent", "position", "rotation_bams", "scale", "eval_flags"):
            assert k in b0, f"bones[0].{k!r} missing — required for reference port"
        assert len(b0["position"]) == 3
        assert len(b0["rotation_bams"]) == 3
        assert len(b0["scale"]) == 3

    # Per-mesh shape. Reference's NjMesh has indices + (bone_index, weight)
    # per vertex + a textureId. We pack vertices + boneIndices as base64
    # blobs and surface material_id.
    assert isinstance(skinned["meshes"], list)
    if skinned["meshes"]:
        m0 = skinned["meshes"][0]
        for k in ("vertices_b64", "indices_b64", "material_id"):
            assert k in m0, f"meshes[0].{k!r} missing — reference renderer port needs it"
        # Optional: bone_indices_b64 is required for skinned models.
        if skinned["has_bone_indices"]:
            assert "bone_indices_b64" in m0


def _assert_phantasmal_animations_shape(anim: dict) -> None:
    """Phantasmal expects a ``motions`` list with frame_count + fps + bone_count."""
    for k in ("motions", "motion_count", "default_index", "skeleton_bone_count"):
        assert k in anim, f"animations.{k!r} missing"
    if anim["motions"]:
        m0 = anim["motions"][0]
        for k in ("index", "name", "frame_count", "fps", "bone_count", "interpolation"):
            assert k in m0, f"motions[0].{k!r} missing — reference auto-pick needs it"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bml,inner,exp_binding,exp_name_match", SURVEYED_MODELS)
def test_bundle_shape_matches_reference(
    client: TestClient,
    bml: str,
    inner: str,
    exp_binding: int,
    exp_name_match: bool,
) -> None:
    """Wire shape stays compatible with a Phantasmal-style renderer."""
    r = client.get(_bundle_url(bml, inner))
    assert r.status_code == 200, f"{bml}#{inner}: {r.status_code} {r.text[:200]}"
    body = r.json()

    # Top-level keys. ``errors`` is only emitted when there are errors —
    # successful loads omit it. ``motion_data`` is null when no motion
    # was requested via ?include_motion=, the others are always present.
    for k in ("filename", "skinned", "animations", "motion_data"):
        assert k in body, f"top-level {k!r} missing"

    skinned = body["skinned"]
    assert skinned is not None, f"{bml}#{inner}: skinned was null (parse failure?)"
    _assert_phantasmal_skinned_shape(skinned)

    # Binding count + name_match — pinned from AGENT_TEXTURE_BINDING_REPORT.md.
    bd = skinned["binding_data"]
    assert len(bd["binding"]) == exp_binding, (
        f"{bml}#{inner}: binding count drifted from "
        f"{exp_binding} to {len(bd['binding'])} — texture-binding agent "
        f"may need to re-survey"
    )
    assert bd["name_match"] is exp_name_match, (
        f"{bml}#{inner}: name_match flipped — NJTL→XVMH alignment broke"
    )

    # Animations — at least the structure must be there even if empty.
    if body["animations"] is not None:
        _assert_phantasmal_animations_shape(body["animations"])


def test_bundle_walk_motion_reaches_dragon(client: TestClient) -> None:
    """Reference's PSO_FRAME_RATE = 30; dragon's walk@13 must agree."""
    r = client.get(
        _bundle_url("bm_boss8_dragon.bml", "boss1_s_nb_dragon.nj") + "?include_motion=walk"
    )
    assert r.status_code == 200
    body = r.json()
    md = body["motion_data"]
    assert md is not None, "include_motion=walk yielded no motion_data for dragon"
    assert md["fps"] == 30.0, f"fps drifted from 30 to {md['fps']} — Phantasmal's PSO_FRAME_RATE_DOUBLE assumes 30"
    # Dragon walk frame-count is fixed by the .njm file; if it changes
    # someone re-baked the asset.
    assert md["frame_count"] >= 30, f"dragon walk frame_count {md['frame_count']} suspiciously short"
    assert md["bone_count"] == 124, f"dragon bone count drifted: {md['bone_count']}"


def test_bundle_kenkyu_w_no_walk_falls_through(client: TestClient) -> None:
    """Lobby girl has no walk — reference would show bind pose; we have an
    Imported Animations sidecar fallback that the anim-preview agent owns.
    """
    r = client.get(
        _bundle_url("bm_npc_kenkyu_w.bml", "kenkyu_w_hone_body.nj") + "?include_motion=walk"
    )
    assert r.status_code == 200
    body = r.json()
    # No walk in the engine-supplied list.
    assert body["motion_data"] is None
    err = body.get("errors") or {}
    assert err.get("motion_data"), "expected an errors.motion_data hint for missing walk"
    # But the animations list is non-empty (120 motions).
    a = body["animations"]
    assert a is not None
    assert a["motion_count"] >= 100, f"kenkyu_w motion list shrunk to {a['motion_count']}"


def test_reference_extracts_present() -> None:
    """The reference Kotlin extracts MUST be on disk for the four agents
    to consult. If someone deleted them, future audits lose ground truth.
    """
    base = _REPO_ROOT / "_reports" / "reference_phantasmal_extracts"
    expected = (
        "MeshRenderer.kt",
        "CharacterClassAssetLoader.kt",
        "XvrTextureConversion.kt",
        "MeshBuilder_summary.md",
    )
    missing = [n for n in expected if not (base / n).is_file()]
    assert not missing, f"reference extracts missing: {missing!r}"


def test_diff_report_present_and_referenced() -> None:
    """The diff report itself is part of the contract — its existence is
    pinned so the four agents can rely on the URLs and snippets inside.
    """
    report = _REPO_ROOT / "_reports" / "regression_diff_vs_phantasmal.md"
    assert report.is_file(), f"missing {report}"
    text = report.read_text(encoding="utf-8")
    # Sanity: the report MUST cite the reference URLs so a future
    # archaeologist can find them again.
    for url_marker in (
        "raw.githubusercontent.com/DaanVandenBosch/phantasmal-world",
        "MirroredRepeatWrapping",
        "PSO_FRAME_RATE",
        "ZYX",
    ):
        assert url_marker in text, f"diff report missing pin: {url_marker!r}"
