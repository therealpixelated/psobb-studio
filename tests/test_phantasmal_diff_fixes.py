"""Static regression tests for the Phantasmal-diff visual fixes.

Pin the conventions from ``_reports/regression_diff_vs_phantasmal.md`` so a
future refactor doesn't accidentally regress the textured-submesh path
back to ``MeshStandardMaterial`` (PBR), drop the
``MirroredRepeatWrapping``, or re-enable the SRGBColorSpace double-gamma.

These are SOURCE-LEVEL assertions on ``static/model_viewer.js`` — they
don't require a running server or a WebGL context. The frontend smoke
test (``test_autoplay_regression.py``) covers the live behaviour.

Conventions enforced (all from the reference Phantasmal renderer):

  * Textured submeshes use ``MeshBasicMaterial`` (Phantasmal's
    ``MeshRenderer.kt``); ``MeshStandardMaterial`` must NOT appear in
    submesh-creation paths.
  * Un-textured paths use ``MeshLambertMaterial`` (closest to Sega's
    per-vertex T&L).
  * Texture wrap mode is ``MirroredRepeatWrapping`` for the real-mesh
    + non-sphere primitive paths (Sega's D3DTADDRESS_MIRROR).
  * No ``tex.colorSpace = THREE.SRGBColorSpace`` anywhere — Phantasmal
    leaves colorSpace at the linear default.
  * The trailing-edge throttle helper (``scheduleRebuild``) exists and
    wraps ``rebuildMeshNow`` with a 10 ms timeout.

Note: the source file legitimately mentions ``MeshStandardMaterial`` in
historical comment text (the audit notes about WHY the material was
swapped). We strip line comments before checking constructor sites so
those don't trigger false positives.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MODEL_VIEWER = REPO / "static" / "model_viewer.js"


def _read_source() -> str:
    assert MODEL_VIEWER.is_file(), f"missing {MODEL_VIEWER}"
    return MODEL_VIEWER.read_text(encoding="utf-8")


def _strip_line_comments(text: str) -> str:
    """Drop ``//`` comments so doc-style mentions don't trigger checks.

    Block comments (``/* ... */``) are left intact because the file uses
    them sparingly; a quick stripping is enough to avoid the few JSDoc
    blocks that name MeshStandardMaterial in prose.
    """
    out_lines = []
    for line in text.splitlines():
        # Find first `//` not inside a string. The codebase doesn't use
        # `//` inside strings except in URLs, which always have a colon
        # before them. A naive split is fine here for the regex check.
        idx = line.find("//")
        if idx >= 0:
            # Make sure we're not inside a string literal at idx. Quick
            # heuristic: count balanced quotes before idx.
            prefix = line[:idx]
            quote_count = prefix.count('"') - prefix.count('\\"')
            if quote_count % 2 == 0:
                line = prefix
        out_lines.append(line)
    out = "\n".join(out_lines)
    # Also strip /* ... */ blocks (single-line and multi-line)
    out = re.sub(r"/\*.*?\*/", "", out, flags=re.DOTALL)
    return out


# --------------------------------------------------------------------- Fix 1


def test_no_srgb_colorspace_on_textures() -> None:
    """Phantasmal-diff fix 1: no ``tex.colorSpace = THREE.SRGBColorSpace``.

    Setting SRGBColorSpace on PSOBB textures causes a double-gamma path
    (sRGB→linear sample, then linear→sRGB framebuffer convert) that
    washes out the colour. Phantasmal's XvrTextureConversion.kt leaves
    colorSpace at its Three.js default.
    """
    src = _strip_line_comments(_read_source())
    assert "THREE.SRGBColorSpace" not in src, (
        "tex.colorSpace = THREE.SRGBColorSpace must not appear in "
        "static/model_viewer.js — it triggers double-gamma. Use the "
        "default linear color space."
    )


# --------------------------------------------------------------------- Fix 2


def test_mirrored_repeat_wrapping_used_for_textures() -> None:
    """Phantasmal-diff fix 2: textures use ``MirroredRepeatWrapping``.

    Sega's Ninja engine assumes D3DTADDRESS_MIRROR. Plain RepeatWrapping
    creates seams on edge-tiled textures (chequerboards, mirror panels).
    """
    src = _read_source()
    # The two texture-loader sites should each set MirroredRepeatWrapping.
    # We allow plain RepeatWrapping in the historical comment notes only;
    # constructor-side assignments must be Mirrored.
    mirrored = src.count("THREE.MirroredRepeatWrapping")
    assert mirrored >= 4, (
        f"expected at least 4 MirroredRepeatWrapping assignments "
        f"(2 wrapS + 2 wrapT across single-tile + per-binding paths), "
        f"got {mirrored}"
    )

    # No real-mesh path should still use plain RepeatWrapping. The
    # comment-stripped text shouldn't have wrapS/wrapT = RepeatWrapping
    # except where the user overrode the shape to a primitive (we kept
    # plain Repeat for non-real-mesh non-sphere fallback).
    src_no_comments = _strip_line_comments(src)
    # Find every "tex.wrapS = THREE.RepeatWrapping" (constructor-style)
    plain_repeat = re.findall(
        r"tex\.wrap[ST]\s*=\s*THREE\.RepeatWrapping", src_no_comments,
    )
    # Allow at most 2 (the non-real, non-sphere primitive fallback path:
    # wrapS + wrapT). If more, someone forgot to switch a real-mesh site.
    assert len(plain_repeat) <= 2, (
        f"too many plain RepeatWrapping assignments ({len(plain_repeat)}); "
        f"expected ≤2 (only the primitive fallback path may use it)"
    )


# --------------------------------------------------------------------- Fix 3


def test_textured_submeshes_default_to_transparent_false() -> None:
    """Phantasmal-diff fix 3: ``transparent: false`` is the default
    for submesh materials (textured AND un-textured fallback).

    Phantasmal sets transparent only when the material chunk reports
    alpha_test or alpha_blend. The material panel
    (window.psoUpdateMaterial) flips it on demand for user edits. Note:
    this rule applies to SUBMESH materials only — debug overlays (e.g.
    the cyan bone-sphere helper) are allowed to set transparent:true
    because they're authored as semi-transparent affordances. We
    distinguish via the presence of ``map:`` (textured) or by being part
    of the rebuildMeshNow primitive path.
    """
    src = _strip_line_comments(_read_source())
    # Find all MeshBasicMaterial / MeshLambertMaterial constructor sites
    # WITH a ``map:`` field (textured submesh path). Those MUST be
    # transparent: false.
    pattern = re.compile(
        r"new THREE\.Mesh(?:Basic|Lambert)Material\(\{[^}]*?\bmap:[^}]*?transparent:\s*(true|false)",
        re.DOTALL,
    )
    matches = pattern.findall(src)
    assert matches, (
        "no textured submesh MeshBasicMaterial/MeshLambertMaterial sites "
        "found at all — fix 4 may have missed the textured paths"
    )
    assert all(m == "false" for m in matches), (
        f"expected all textured submesh constructor sites to use "
        f"transparent: false (Phantasmal-diff fix 3); offenders: "
        f"{[m for m in matches if m != 'false']}"
    )


def test_untextured_submesh_fallbacks_default_to_transparent_false() -> None:
    """Phantasmal-diff fix 3 (un-textured branch).

    The MeshLambertMaterial fallback for submeshes WITHOUT a bound
    texture must also default to transparent: false.
    """
    src = _strip_line_comments(_read_source())
    # MeshLambertMaterial sites with ``color: 0xffffff`` (un-textured
    # submesh fallback in rebuildMeshNow + 3 submesh paths).
    pattern = re.compile(
        r"new THREE\.MeshLambertMaterial\(\{[^}]*?color:\s*0xffffff[^}]*?transparent:\s*(true|false)",
        re.DOTALL,
    )
    matches = pattern.findall(src)
    assert matches, "no un-textured MeshLambertMaterial sites found"
    assert all(m == "false" for m in matches), (
        f"expected un-textured Lambert fallback sites to use "
        f"transparent: false; offenders: "
        f"{[m for m in matches if m != 'false']}"
    )


# --------------------------------------------------------------------- Fix 4


def test_no_meshstandardmaterial_in_submesh_paths() -> None:
    """Phantasmal-diff fix 4: textured submeshes use MeshBasicMaterial.

    PSOBB does no BRDF on diffuse-mapped pixels, so PBR is wasted GPU
    cost (Dragon's 1069 submeshes × PBR shader compile was the worst
    offender). Phantasmal uses MeshBasicMaterial for textured paths and
    MeshLambertMaterial for un-textured.
    """
    src = _strip_line_comments(_read_source())
    # Constructor sites: should be ZERO MeshStandardMaterial in the
    # comment-stripped source.
    constructor_sites = re.findall(
        r"new THREE\.MeshStandardMaterial\(", src,
    )
    assert not constructor_sites, (
        f"found {len(constructor_sites)} MeshStandardMaterial constructor "
        f"sites — Phantasmal-diff fix 4 requires MeshBasicMaterial "
        f"(textured) / MeshLambertMaterial (un-textured)"
    )

    # Sanity: the new types are actually being used.
    basic_sites = re.findall(r"new THREE\.MeshBasicMaterial\(", src)
    lambert_sites = re.findall(r"new THREE\.MeshLambertMaterial\(", src)
    assert len(basic_sites) >= 4, (
        f"expected ≥4 MeshBasicMaterial sites (rebuildMeshNow + 3 "
        f"submesh paths: world-baked, composite, skinned); got {len(basic_sites)}"
    )
    assert len(lambert_sites) >= 4, (
        f"expected ≥4 MeshLambertMaterial sites (rebuildMeshNow "
        f"un-textured branch + 3 un-textured submesh fallbacks + "
        f"scene/terrain); got {len(lambert_sites)}"
    )


# --------------------------------------------------------------------- Fix 5


def test_schedule_rebuild_throttle_exists() -> None:
    """Phantasmal-diff fix 5: 10 ms trailing-edge throttle on rebuild.

    Phantasmal uses Throttle(wait=10, leading=false, trailing=true) on
    its rebuildMesh path. We replicate it as ``scheduleRebuild()``
    wrapping ``rebuildMeshNow()`` with a 10 ms setTimeout.
    """
    src = _strip_line_comments(_read_source())
    # The rebuild-now function must exist with that name.
    assert re.search(r"function\s+rebuildMeshNow\s*\(", src), (
        "rebuildMeshNow() must exist — it's the immediate-fire path "
        "preserved for first-load (so the modal doesn't show a blank "
        "canvas during the throttle window)."
    )
    # The throttle wrapper.
    assert re.search(r"function\s+scheduleRebuild\s*\(", src), (
        "scheduleRebuild() must exist — Phantasmal-diff fix 5 trailing "
        "edge throttle wrapper for interactive rebuild calls."
    )
    # And it should set a setTimeout with 10 ms wait. Use a balanced
    # brace scan since the body has nested {} (the setTimeout arrow fn).
    sched_match = re.search(r"function\s+scheduleRebuild\s*\(", src)
    assert sched_match, "couldn't locate scheduleRebuild start"
    start = sched_match.end()
    # Skip to the opening brace of the body
    while start < len(src) and src[start] != "{":
        start += 1
    depth = 0
    end = start
    while end < len(src):
        if src[end] == "{":
            depth += 1
        elif src[end] == "}":
            depth -= 1
            if depth == 0:
                end += 1
                break
        end += 1
    body = src[start:end]
    assert "setTimeout" in body, "scheduleRebuild must use setTimeout"
    assert re.search(r"\b10\b", body), (
        "scheduleRebuild must use 10 ms wait — Phantasmal's exact value"
    )
    # The interactive handler (#modelShapeSel change) must call
    # scheduleRebuild() — not rebuildMesh()/rebuildMeshNow() directly.
    # We slice forward from #modelShapeSel and balance braces to extract
    # just the change-handler body.
    sm = re.search(r'#modelShapeSel"\)\.addEventListener\(', src)
    assert sm, "couldn't find #modelShapeSel.addEventListener call"
    cur = sm.end()
    # skip past the (event, handler) opening paren; find first { of fn body
    while cur < len(src) and src[cur] != "{":
        cur += 1
    # balance braces
    depth = 0
    body_end = cur
    while body_end < len(src):
        if src[body_end] == "{":
            depth += 1
        elif src[body_end] == "}":
            depth -= 1
            if depth == 0:
                body_end += 1
                break
        body_end += 1
    handler_body = src[cur:body_end]
    assert "scheduleRebuild" in handler_body, (
        "#modelShapeSel change handler must route through scheduleRebuild() "
        "(Phantasmal-diff fix 5: interactive edits coalesce within 10 ms). "
        f"Handler body: {handler_body[:300]!r}"
    )


# --------------------------------------------------------------------- Sanity


def test_exact_lambert_toggle_path_preserved() -> None:
    """The opt-in PSOBB-exact Lambert toggle (psoSceneUseExactLambert)
    must still be intact — it's the bit-exact lighting path the Map
    Editor uses, and we must not have disturbed it.
    """
    src = _read_source()
    assert "psoSceneUseExactLambert" in src
    assert "psobb_lambert_shader.js" in src
