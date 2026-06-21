"""Unit tests for ``formats/composite_assembly.py``.

Verifies the curated multi-inner BML placement table contract:

  * The De Rol Le entry exists and exposes a multi-part assembly
    (the user-visible failure mode being a single-part fallback,
    which would still render every inner stacked at world origin).
  * Every part's TRS values are finite, non-NaN floats — a single
    NaN here would propagate into the JSON response and crash the
    Godot frontend's matrix composition.
  * Lookup is case-insensitive and tolerates path-prefixed input
    (the ``/api/composite_bundle/{path:path}`` endpoint passes the
    raw URL path through).
  * Unknown BMLs return ``None`` so the endpoint can decide between
    a 404 and an identity-fallback synthesis.

Pure unit test — does NOT require the live server and does NOT
touch ``PSOBB.IO/data``. Safe to run in any environment that has
the editor's Python path importable.
"""
from __future__ import annotations

import math

import pytest

from formats.composite_assembly import (
    CompositeAssembly,
    CompositePart,
    COMPOSITE_TABLE,
    lookup_composite,
)


# ---------------------------------------------------------------------------
# De Rol Le — the canonical multi-part boss the table was built for.
# ---------------------------------------------------------------------------


def test_de_rol_le_is_in_the_table():
    """The base ``bm_boss2_de_rol_le.bml`` must have curated data.

    The whole point of the composite endpoint is to fix De Rol Le's
    "all parts at origin" rendering bug. Falling back to identity
    here would defeat the purpose.
    """
    assembly = lookup_composite("bm_boss2_de_rol_le.bml")
    assert assembly is not None, (
        "De Rol Le has no curated entry — composite endpoint will fall "
        "back to identity placement and the user will see all parts "
        "stacked at world origin again."
    )
    assert isinstance(assembly, CompositeAssembly)
    assert assembly.source != "identity-fallback", (
        f"De Rol Le entry is marked identity-fallback "
        f"({assembly.source!r}) — the curated layout was lost."
    )


def test_de_rol_le_is_body_only_no_floating_parts():
    """De Rol Le renders body-ONLY by design (2026-06-21).

    The BML ships 7 NJ inners, but a STATIC TRS cannot place the six
    non-body inners on the curved/animated body — they floated off
    detached ("De Rol Le still fucked up", owner). Two are damage states
    (helm_break / shell_break) that must never show on the intact boss;
    the four appendages (fins, sting, tentacle) attach to specific body
    BONES in-game (offsets live in PSOBB.exe, not the assets). So the
    default assembly is the body serpent alone — its own recognizable
    form — until per-bone attachment is recovered.
    """
    assembly = lookup_composite("bm_boss2_de_rol_le.bml")
    assert assembly is not None
    inners = [p.inner_nj.lower() for p in assembly.parts]
    assert any("body" in i for i in inners), "lost the body part"
    for bad in ("helm_break", "shell_break"):
        assert not any(bad in i for i in inners), (
            f"{bad} is a damage state and must not show on the intact boss"
        )


def test_de_rol_le_includes_the_body():
    """The body inner is the centerpiece — every other part anchors
    off it. If we lose the body the composite is just floating limbs.
    """
    assembly = lookup_composite("bm_boss2_de_rol_le.bml")
    assert assembly is not None
    body_parts = [
        p for p in assembly.parts
        if "body" in p.inner_nj.lower()
    ]
    assert body_parts, (
        f"De Rol Le composite missing a *body* inner; "
        f"got: {[p.inner_nj for p in assembly.parts]}"
    )


def test_de_rol_le_alt_variant_present():
    """The ``_a`` variant ships in PSOBB.IO too. Same inner names,
    same layout — the table aliases both basenames so they get the
    same composite.
    """
    a = lookup_composite("bm_boss2_de_rol_le.bml")
    a_alt = lookup_composite("bm_boss2_de_rol_le_a.bml")
    assert a is not None and a_alt is not None
    assert len(a_alt.parts) == len(a.parts), (
        f"alt variant has different part count "
        f"({len(a_alt.parts)} vs {len(a.parts)}); they should mirror "
        f"each other."
    )


# ---------------------------------------------------------------------------
# Universal field-validity guards (apply to every entry in the table).
# ---------------------------------------------------------------------------


def _assert_finite_triplet(triplet, label: str) -> None:
    """Assert a 3-element tuple of finite floats."""
    assert isinstance(triplet, tuple), f"{label}: expected tuple, got {type(triplet).__name__}"
    assert len(triplet) == 3, f"{label}: expected 3 elements, got {len(triplet)}"
    for axis, v in zip("xyz", triplet):
        assert isinstance(v, (int, float)), (
            f"{label}.{axis}: expected number, got {type(v).__name__}"
        )
        fv = float(v)
        assert math.isfinite(fv), f"{label}.{axis}: not finite ({v!r})"
        assert not math.isnan(fv), f"{label}.{axis}: NaN ({v!r})"


@pytest.mark.parametrize(
    "key", sorted(COMPOSITE_TABLE.keys()),
)
def test_every_table_entry_has_finite_trs(key):
    """No entry may carry NaN / inf / non-numeric TRS values.

    A single bad float here would surface as ``"NaN"`` in the JSON
    response (json.dumps on float('nan') emits the bare token), then
    crash the frontend's matrix composition.
    """
    assembly = COMPOSITE_TABLE[key]
    assert assembly.parts, f"{key}: empty parts list"
    for i, part in enumerate(assembly.parts):
        _assert_finite_triplet(part.pos, f"{key}.parts[{i}].pos")
        _assert_finite_triplet(part.rot_euler, f"{key}.parts[{i}].rot_euler")
        _assert_finite_triplet(part.scale, f"{key}.parts[{i}].scale")


@pytest.mark.parametrize(
    "key", sorted(COMPOSITE_TABLE.keys()),
)
def test_every_table_entry_has_inner_names(key):
    """Each part must reference a non-empty inner-NJ name."""
    assembly = COMPOSITE_TABLE[key]
    for i, part in enumerate(assembly.parts):
        assert isinstance(part.inner_nj, str), (
            f"{key}.parts[{i}].inner_nj: not a string"
        )
        assert part.inner_nj.strip(), (
            f"{key}.parts[{i}].inner_nj: empty / whitespace"
        )
        # Sanity: parent_inner, when set, must point at one of the
        # other inners in the same assembly.
        if part.parent_inner is not None:
            siblings = {p.inner_nj for p in assembly.parts}
            assert part.parent_inner in siblings, (
                f"{key}.parts[{i}] references parent {part.parent_inner!r} "
                f"that is not in the assembly's own inner list "
                f"{sorted(siblings)}"
            )


# ---------------------------------------------------------------------------
# Lookup helper behaviour.
# ---------------------------------------------------------------------------


def test_lookup_is_case_insensitive():
    """The endpoint accepts path strings as-typed; case-insensitive
    lookup means a stray uppercase letter in the URL doesn't drop the
    user back to the identity-fallback path.
    """
    base = lookup_composite("bm_boss2_de_rol_le.bml")
    upper = lookup_composite("BM_BOSS2_DE_ROL_LE.BML")
    mixed = lookup_composite("Bm_Boss2_De_Rol_Le.Bml")
    assert base is not None
    assert upper is base, "uppercase lookup did not return the same entry"
    assert mixed is base, "mixed-case lookup did not return the same entry"


def test_lookup_strips_path_prefix():
    """Endpoint may pass a relative or absolute path; only the
    basename matters for lookup.
    """
    base = lookup_composite("bm_boss2_de_rol_le.bml")
    rel = lookup_composite("data/bm_boss2_de_rol_le.bml")
    abs_fwd = lookup_composite("C:/Users/foo/PSOBB.IO/data/bm_boss2_de_rol_le.bml")
    abs_bwd = lookup_composite(r"C:\Users\foo\PSOBB.IO\data\bm_boss2_de_rol_le.bml")
    assert base is not None
    assert rel is base
    assert abs_fwd is base
    assert abs_bwd is base


def test_lookup_strips_inner_suffix():
    """The ``base#inner`` API form should still resolve to the BML
    entry — the composite endpoint always returns ALL parts, but a
    sloppy caller might URL-encode the same path the model_bundle
    endpoint accepts.
    """
    base = lookup_composite("bm_boss2_de_rol_le.bml")
    with_inner = lookup_composite("bm_boss2_de_rol_le.bml#boss2_b_derorure_body.nj")
    assert base is not None
    assert with_inner is base


def test_lookup_returns_none_for_unknown_bml():
    """Unknown BMLs return None so the endpoint can fall back to
    identity placement of every primary inner.
    """
    assert lookup_composite("bm_does_not_exist.bml") is None
    assert lookup_composite("bm_npc_random.bml") is None


def test_lookup_handles_falsy_input():
    """Defensive: empty string / None / non-string -> None, never
    raise. The endpoint's path validator already rejects those, but
    the helper is allowed to be called from other code paths too.
    """
    assert lookup_composite("") is None
    assert lookup_composite("   ") is None
    assert lookup_composite(None) is None  # type: ignore[arg-type]
    assert lookup_composite(123) is None   # type: ignore[arg-type]
