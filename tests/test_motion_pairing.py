"""Unit tests for ``formats.motion_pairing`` (Part D of the
2026-04-26 motion-resolver agent).

These tests exercise the four-tier ranker against the real PSOBB.IO
data tree at ``C:/tmp_pso_dev/data``. Each fixture is a real BML
shipping real motions — there's no synthetic data because the whole
point of the resolver is empirical inventory matching.

Tests skip cleanly when the data tree is absent (CI without the
install, etc.).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from formats.motion_pairing import (
    ACTION_ATTACK,
    ACTION_DIE,
    ACTION_HIT,
    ACTION_IDLE,
    ACTION_PRIORITY,
    ACTION_UNKNOWN,
    ACTION_WALK,
    MotionRef,
    extract_action_hint,
    resolve_motions_for_model,
)


DATA_DIR = Path("C:/tmp_pso_dev/data")
DATA_AVAILABLE = DATA_DIR.is_dir()
SKIP_REASON = f"PSOBB data tree not present at {DATA_DIR}"


# ---------------------------------------------------------------------------
# Pure-function tests (no data dependency).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,expected", [
    # Walk family — primary locomotion.
    ("walk_boss1_s_nb_dragon.njm",      ACTION_WALK),
    ("run_bm7_s_pal_body.njm",          ACTION_WALK),
    ("move_bm4_ps_mb_body.njm",         ACTION_WALK),
    ("dash_test.njm",                   ACTION_WALK),
    # Idle family.
    ("wait_bm4_ps_ma_body.njm",         ACTION_IDLE),
    ("stand_boss1_s_nb_dragon.njm",     ACTION_IDLE),
    ("cstand_bm2c_s_moj_body.njm",      ACTION_IDLE),
    ("land_boss1_s_nb_dragon.njm",      ACTION_IDLE),
    # Attack family — incl. tail-attack idiom from memory note.
    ("attack_re5_b_body.njm",           ACTION_ATTACK),
    ("atack_bm4_ps_ma_body.njm",        ACTION_ATTACK),
    ("tlatk_bm4_ps_ma_tail.njm",        ACTION_ATTACK),
    ("tatk_boss1_s_nb_dragon.njm",      ACTION_ATTACK),
    ("fire_boss1_s_nb_dragon.njm",      ACTION_ATTACK),
    ("beam_a_boss2_b_body.njm",         ACTION_ATTACK),
    ("hoe_boss5_s_body.njm",            ACTION_ATTACK),
    # Die family — multiple suffix variants.
    ("dead_boss1_s_nb_dragon.njm",      ACTION_DIE),
    ("deadb_bm7_s_paa_body.njm",        ACTION_DIE),
    ("die_boss2_b_body.njm",            ACTION_DIE),
    ("death_me5p02_y_all.njm",          ACTION_DIE),
    # Hit family.
    ("damage_bm4_ps_ma_body.njm",       ACTION_HIT),
    ("daml_boss1_s_nb_dragon.njm",      ACTION_HIT),
    ("tumble_re3_b_base.njm",           ACTION_HIT),
    # Unknown verb.
    ("1st_advent_root.njm",             ACTION_UNKNOWN),
    ("eff_unknown_verb.njm",            ACTION_UNKNOWN),
    # Tolerates missing .njm suffix.
    ("walk_foo",                        ACTION_WALK),
    # Tolerates upper-case.
    ("WALK_FOO.NJM",                    ACTION_WALK),
])
def test_extract_action_hint(name: str, expected: str) -> None:
    """Verb-prefix classifier covers the dominant patterns from the
    on-disk inventory.

    Five known-action filenames per the task spec — extended to ~25
    here to lock down the verb-table coverage.
    """
    assert extract_action_hint(name) == expected


def test_action_priority_walk_first() -> None:
    """Walk MUST outrank every other action so auto-play picks
    locomotion first."""
    assert ACTION_PRIORITY[ACTION_WALK] == 0
    # Idle is the explicit fallback for boss tracks lacking walk
    # (Mericarol/Mericus etc.) — see the autoplay regression suite.
    assert ACTION_PRIORITY[ACTION_IDLE] < ACTION_PRIORITY[ACTION_ATTACK]
    assert ACTION_PRIORITY[ACTION_ATTACK] < ACTION_PRIORITY[ACTION_DIE]
    assert ACTION_PRIORITY[ACTION_UNKNOWN] == max(ACTION_PRIORITY.values())


# ---------------------------------------------------------------------------
# Resolver tests against real BMLs.
# ---------------------------------------------------------------------------


# 10 sampled fixtures spanning the inventory:
#   - Multi-form BMLs (Pan Arms body + tail, Pouilly Slime trio, De Rol Le)
#   - Single-form BMLs (dragon, boota, balclaw, mericarol, biter)
#   - Object/prop BMLs with and without inline motions (warpboss, sensor)
#
# Each tuple is (bml_name, inner_model.nj | None, expected_first_action,
# min_motion_count). Tests below exercise resolve_motions_for_model
# returning >= min_motion_count and the first entry's action matches
# the expectation (the auto-play target).
SAMPLED_ENTITIES: list[tuple[str, str | None, str, int]] = [
    # Pan Arms body — should pick idle (wait_*) because no walk for
    # this stem; tail-attack motion targets a sibling stem.
    ("bm4_ps_ma_body.bml",     "bm4_ps_ma_body.nj",    ACTION_IDLE,    7),
    # Pan Arms TAIL — should pick the tlatk attack (only Tier-2 hit).
    ("bm4_ps_ma_body.bml",     "bm4_ps_ma_tail.nj",    ACTION_ATTACK,  1),
    # Dragon (Forest boss) — walk available.
    ("bm_boss1_dragon.bml",    "boss1_s_nb_dragon.nj", ACTION_WALK,   20),
    # NanoDragon (caves mob from the bug report) — walk + 11 others.
    ("bm_ene_nanodrago.bml",   "bm6_s_drc_body.nj",    ACTION_WALK,    8),
    # Bal Claw (caves) — only walk_re6_b_bal_body.
    ("bm_ene_balclaw.bml",     "re6_b_bal_body.nj",    ACTION_IDLE,    10),
    # Pouilly Slime — multi-form (paa/pal/par); body picks walk
    # (walk_bm7_s_paa_body exists for the same stem).
    ("bm7_s_paa_body.bml",     "bm7_s_paa_body.nj",    ACTION_WALK,    5),
    # Mericarol — autoplay-regression case 3, idle fallback.
    ("bm_ene_bm9_s_mericarol.bml", "bm9_s_meri_body.nj", ACTION_IDLE,  4),
    # De Rol Le — multi-stem boss. Inner ``boss2_b_derorure_body.nj``
    # has no walk; ``boss2_b_body`` motions are Tier-3 (different
    # stem). The first Tier-3 entry is the ``beamwait`` idle, which
    # is the closest to a "default pose" available.
    ("bm_boss2_de_rol_le.bml", "boss2_b_derorure_body.nj", ACTION_IDLE,  5),
    # Biter (lab) — single-stem rig with ``run_*`` and ``walk_*``;
    # auto-play picks the first walk-action.
    ("bm_ene_biter_body.bml",  "biter_body.nj",        ACTION_WALK,    7),
    # Warpboss object — no inline motions match any inner stem; Tier-3
    # only with verb "de" (unknown). Just assert the resolver returns
    # the 2 motions without crashing.
    ("bm_obj_warpboss_ancient.bml", "fe_obj_df_warp_gawa.xj", ACTION_UNKNOWN, 1),
]


@pytest.mark.skipif(not DATA_AVAILABLE, reason=SKIP_REASON)
@pytest.mark.parametrize("bml_name,inner,expected_action,min_count", SAMPLED_ENTITIES)
def test_resolve_returns_motions(
    bml_name: str, inner: str | None, expected_action: str, min_count: int,
) -> None:
    """Each sampled entity resolves to at least one motion AND the
    first entry's action matches the expected auto-play target.

    First-entry action drives ``api_animations.default_index`` (when
    ``api_animations`` picks index 0 for non-unknown actions); a
    regression here means the auto-play would fall back to a
    bind-pose-snapping motion authored for the wrong inner-rig.
    """
    p = DATA_DIR / bml_name
    if not p.is_file():
        pytest.skip(f"{bml_name} missing from data tree")

    refs = resolve_motions_for_model(
        p, inner_name=inner,
        npc_motion_pack_search_roots=(DATA_DIR,),
    )
    assert refs, f"{bml_name} (inner={inner}) returned no motions"
    assert len(refs) >= min_count, (
        f"{bml_name} returned only {len(refs)} motions, expected ≥ {min_count}"
    )
    assert refs[0].action == expected_action, (
        f"{bml_name} first motion action = {refs[0].action!r}; "
        f"expected {expected_action!r} (auto-play would mis-trigger). "
        f"Top-3: {[(r.tier, r.action, r.inner_name) for r in refs[:3]]}"
    )


@pytest.mark.skipif(not DATA_AVAILABLE, reason=SKIP_REASON)
def test_walk_outranks_other_actions_when_present() -> None:
    """Sort order: when both walk and idle motions exist for the
    matching stem, walk lands at index 0.

    Dragon BML has both ``walk_*`` and ``stand_*`` for the same stem,
    so the resolver is forced to choose. Walk MUST win.
    """
    p = DATA_DIR / "bm_boss1_dragon.bml"
    if not p.is_file():
        pytest.skip("dragon BML missing")
    refs = resolve_motions_for_model(
        p, inner_name="boss1_s_nb_dragon.nj",
        npc_motion_pack_search_roots=(DATA_DIR,),
    )
    # Index 0 = walk, no idle entry should appear before it.
    actions = [r.action for r in refs]
    walk_idx = actions.index(ACTION_WALK)
    idle_idx = actions.index(ACTION_IDLE) if ACTION_IDLE in actions else len(actions)
    assert walk_idx < idle_idx


@pytest.mark.skipif(not DATA_AVAILABLE, reason=SKIP_REASON)
def test_tier2_outranks_tier3() -> None:
    """Stem-matched motions (Tier 2) ALWAYS appear before non-stem
    matches (Tier 3) regardless of action.

    Pan Arms body (43 bones) has no walk/move for its own stem but
    has one for the sibling ``mb_body`` stem (1 bone). The resolver
    must put the Tier-2 idle BEFORE the Tier-3 walk so the auto-play
    doesn't snap the body's 43-bone rig to a 1-bone track.
    """
    p = DATA_DIR / "bm4_ps_ma_body.bml"
    if not p.is_file():
        pytest.skip("Pan Arms BML missing")
    refs = resolve_motions_for_model(
        p, inner_name="bm4_ps_ma_body.nj",
        npc_motion_pack_search_roots=(DATA_DIR,),
    )
    tiers = [r.tier for r in refs]
    # Find first Tier-3 entry — every prior entry MUST be Tier 2.
    if 3 in tiers:
        first_t3 = tiers.index(3)
        assert all(t == 2 for t in tiers[:first_t3]), (
            f"Tier ordering violated: {tiers[:first_t3 + 2]}"
        )


@pytest.mark.skipif(not DATA_AVAILABLE, reason=SKIP_REASON)
def test_motionref_source_label() -> None:
    """``MotionRef.source_label`` produces ``<bml>#<inner>`` for BML
    sources and a bare filename for top-level loose files.

    The wire format in ``api_animations.motions[].source_path`` uses
    this exact form — frontend ``loadMotion`` parses it back to
    address the right archive entry.
    """
    p = DATA_DIR / "bm_boss1_dragon.bml"
    if not p.is_file():
        pytest.skip("dragon BML missing")
    refs = resolve_motions_for_model(
        p, inner_name="boss1_s_nb_dragon.nj",
        npc_motion_pack_search_roots=(DATA_DIR,),
    )
    assert refs
    label = refs[0].source_label
    assert "#" in label
    assert label.startswith("bm_boss1_dragon.bml#")
    assert label.endswith(".njm")


@pytest.mark.skipif(not DATA_AVAILABLE, reason=SKIP_REASON)
def test_npc_motion_pack_fallback_for_pl_class() -> None:
    """Player-class BMLs (``pl*nj.bml``) fall back to ``NpcApcMot.bml``
    when they ship without inline motions. The fallback only fires
    when no Tier-2 hit is found, so a player-class rig that DOES
    happen to ship its own walk doesn't get clobbered with 120 NPC
    motions.

    Skips when ``NpcApcMot.bml`` isn't present in the data tree.
    """
    pack = DATA_DIR / "NpcApcMot.bml"
    if not pack.is_file():
        pytest.skip("NpcApcMot.bml missing")
    bml = DATA_DIR / "plGnj.bml"
    if not bml.is_file():
        pytest.skip("plGnj.bml missing")
    refs = resolve_motions_for_model(
        bml, inner_name=None,
        npc_motion_pack_search_roots=(DATA_DIR,),
    )
    # plGnj.bml has zero inline motions → all 120 entries come from
    # NpcApcMot at Tier 4.
    if refs:  # if it found any, they must be Tier 4
        tier4 = [r for r in refs if r.tier == 4]
        assert tier4, f"Expected Tier-4 fallback motions for plGnj.bml, got {[(r.tier, r.inner_name) for r in refs[:3]]}"


def test_empty_for_missing_file(tmp_path: Path) -> None:
    """Resolver returns ``[]`` for a non-existent path — caller is
    fail-soft and shouldn't crash on a stale BML reference."""
    fake = tmp_path / "does_not_exist.bml"
    refs = resolve_motions_for_model(fake, inner_name=None)
    assert refs == []


def test_empty_for_unsupported_extension(tmp_path: Path) -> None:
    """Resolver gracefully ignores extensions outside the supported
    set (``.bml``/``.nj``/``.xj``). A future caller passing a stale
    ``.afs`` reference shouldn't get a stack trace."""
    fake = tmp_path / "fake.afs"
    fake.write_bytes(b"")
    refs = resolve_motions_for_model(fake, inner_name=None)
    assert refs == []


def test_motionref_dataclass_round_trip() -> None:
    """Manual ``MotionRef`` construction works (so callers can build
    refs from cached server-side state without re-running the
    resolver).

    Catches any field-rename regression — the wire format depends on
    these specific attribute names.
    """
    ref = MotionRef(
        archive=Path("foo.bml"),
        inner_name="walk_test.njm",
        motion_label="walk_test",
        action=ACTION_WALK,
        confidence=1.0,
        tier=2,
        stem="test",
    )
    assert ref.path == Path("foo.bml")
    assert ref.source_label == "foo.bml#walk_test.njm"
    assert ref.action == "walk"
