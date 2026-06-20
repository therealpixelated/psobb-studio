# Wave 2 / Agent B (2026-04-26).
#
# Inner-discovery completeness regression tests. Asserts that for every BML
# in the Wave-2 audit set, ``formats.bml.parse_bml`` returns at least the
# expected number of ``.nj`` + ``.xj`` model inners. The expected counts
# come from ``formats.psobb_engine_tables.EXPECTED_BML_INNER_COUNTS``,
# which is the verified ground truth produced by walking the BMLs on
# disk (and cross-checked against the C# PSOBMLExtract reader).
#
# The "missing inners" hypothesis the user reported (Sil Dragon: feet,
# tail, no eyeballs) is NOT an inner-discovery problem — Sil Dragon truly
# has only 2 ``.nj`` files; the visible parts live INSIDE
# ``boss1_s_nb_dragon.nj`` as a chunked NJ tree. These tests guard
# against a future regression where the walker starts dropping inners
# (e.g. a misguided "skip non-.nj" early-exit) and the user STARTS
# losing inners.
"""Regression tests for BML model-inner discovery completeness."""
from __future__ import annotations
import os

from pathlib import Path

import pytest

from formats.bml import parse_bml
from formats.psobb_engine_tables import (
    EXPECTED_BML_INNER_COUNTS,
    expected_bml_inner_count,
)


# Where the live install lives. Tests skip when the install is absent so
# CI / dev boxes without PSOBB.IO checked out still pass.
_INSTALL_DATA_DIR = Path(os.path.expanduser("~/PSOBB.IO/data"))


def _count_model_inners(bml_path: Path) -> tuple[int, list[str]]:
    """Return ``(count, names)`` for a BML's ``.nj``/``.xj`` inners.

    Helper for the parametrised test below; surfaced as a function so
    debugging output (failed assertion message) carries the actual
    inner-name list rather than just the count.
    """
    entries = parse_bml(bml_path.read_bytes())
    names = [
        ent.name for ent in entries
        if ent.name.lower().endswith((".nj", ".xj"))
    ]
    return len(names), names


# ---------------------------------------------------------------------------
# Per-BML completeness assertion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bml_filename,expected_count", sorted(EXPECTED_BML_INNER_COUNTS.items()))
def test_bml_meets_expected_inner_count(bml_filename: str, expected_count: int) -> None:
    """For every audited BML, the walker returns >= expected ``.nj``/``.xj``.

    Skips when the install copy of the BML is missing (CI env). In a dev
    box with PSOBB.IO present, every audited BML must hit at least its
    expected count or the test fails — that's the point of having an
    audit.
    """
    bml_path = _INSTALL_DATA_DIR / bml_filename
    if not bml_path.exists():
        pytest.skip(f"install copy of {bml_filename} not present")

    actual_count, names = _count_model_inners(bml_path)
    assert actual_count >= expected_count, (
        f"{bml_filename}: walker reports {actual_count} .nj/.xj inners "
        f"({names!r}); expected >= {expected_count}. "
        f"See _reports/inner_discovery_audit.md."
    )


# ---------------------------------------------------------------------------
# Spec-required spot checks (preserve the explicit assertions the task
# brief called out, even though they're subsumed by the parametrised
# pass — having them by name makes a regression's failure trace
# self-documenting).
# ---------------------------------------------------------------------------

def test_sil_dragon_has_at_least_2_inners() -> None:
    """Sil Dragon's BML carries main + shadow (2 .nj). Per audit ground truth.

    The user's task brief said "should have >= 7 inners (head, body, tail,
    4 legs, eyes)" — but the audit confirmed that's a misconception:
    those 7 visible parts live inside the SINGLE
    ``boss1_s_nb_dragon.nj`` NJ tree, not as separate inners. The real
    ground truth, verified against PSOBMLExtract, is 2 inners.
    """
    bml_path = _INSTALL_DATA_DIR / "bm_boss1_dragon.bml"
    if not bml_path.exists():
        pytest.skip("install copy of bm_boss1_dragon.bml not present")
    actual, names = _count_model_inners(bml_path)
    assert actual >= 2, f"bm_boss1_dragon.bml: got {names!r}"
    assert "boss1_s_nb_dragon.nj" in names
    assert "boss1_s_sd_dragon.nj" in names


def test_de_rol_le_has_at_least_5_inners() -> None:
    """De Rol Le's BML carries body + fin_a + fin_b + sting + tentacle.

    Plus 2 destruction-state proxies (``helm_break`` + ``shell_break``)
    bringing the verified ground truth to 7. The brief asked for >= 5
    so we keep the threshold permissive.
    """
    bml_path = _INSTALL_DATA_DIR / "bm_boss2_de_rol_le.bml"
    if not bml_path.exists():
        pytest.skip("install copy of bm_boss2_de_rol_le.bml not present")
    actual, names = _count_model_inners(bml_path)
    assert actual >= 5, f"bm_boss2_de_rol_le.bml: got {names!r}"
    # Spec-required pieces.
    expected_pieces = [
        "boss2_b_derorure_body.nj",
        "boss2_b_derorure_fin_a.nj",
        "boss2_b_derorure_fin_b.nj",
        "boss2_b_derorure_sting.nj",
        "boss2_b_derorure_tentacle.nj",
    ]
    missing = [p for p in expected_pieces if p not in names]
    assert not missing, f"bm_boss2_de_rol_le.bml missing pieces: {missing!r}"


# ---------------------------------------------------------------------------
# expected_bml_inner_count() helper API contract
# ---------------------------------------------------------------------------

def test_expected_bml_inner_count_returns_int_for_known_bml() -> None:
    """Helper returns the audited int for a BML present in the table."""
    assert expected_bml_inner_count("bm_boss1_dragon.bml") == 2
    assert expected_bml_inner_count("bm_boss2_de_rol_le.bml") == 7


def test_expected_bml_inner_count_returns_none_for_unknown_bml() -> None:
    """Helper returns None for any BML not in the audit."""
    assert expected_bml_inner_count("bm_zzz_does_not_exist.bml") is None
    assert expected_bml_inner_count("") is None
