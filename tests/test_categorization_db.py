"""Tests for the JSON-driven asset categorizer in ``manifest.py``.

Covers:
  * the 5 user-reported miscategorizations (bm4_ps_ma_body etc.) the
    research agent's categorization_db.json was built to fix,
  * a sample of additional rules drawn from the 100+ patterns in the DB,
  * the fallback (unknown path → no inferred_category),
  * rule precedence (more-specific patterns must win over less-specific
    ones — load-bearing for ``bm_ene_boss09_*`` vs ``bm_ene_*`` etc.).

The DB itself lives at ``_reports/categorization_db.json`` (relative to
the editor root) and is treated as the source of truth for all category
labels asserted in this file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from manifest import (
    _CATEGORY_DB_PATH,
    _category_db_cache_clear,
    _load_category_db,
    _match_pattern,
    infer_category,
    infer_category_full,
)


# ---------------------------------------------------------------------------
# DB sanity
# ---------------------------------------------------------------------------

def test_category_db_file_exists():
    """The JSON DB file must exist on disk for the categorizer to work."""
    assert _CATEGORY_DB_PATH.exists(), (
        f"categorization_db.json missing at {_CATEGORY_DB_PATH}"
    )


def test_category_db_loads_and_has_rules():
    """Loading the DB returns the expected top-level shape with rules."""
    _category_db_cache_clear()
    db = _load_category_db()
    assert isinstance(db, dict)
    assert isinstance(db.get("rules"), list)
    assert len(db["rules"]) >= 100, (
        f"DB shrank unexpectedly: {len(db['rules'])} rules; expected >=100"
    )
    # Every rule must have a pattern + category — anything else is
    # a malformed DB entry.
    for r in db["rules"]:
        assert isinstance(r, dict), f"non-dict rule: {r!r}"
        assert r.get("pattern"), f"rule missing pattern: {r!r}"
        assert r.get("category"), f"rule missing category: {r!r}"


def test_category_db_has_fallback():
    """The DB declares its fallback label."""
    db = _load_category_db()
    assert db.get("fallback") == "Uncategorized"


# ---------------------------------------------------------------------------
# The 5 user-reported miscategorizations
#
# These are the entire reason this DB exists — if any of these regress
# the editor goes back to mislabeling Caves enemies as Player Bodies etc.
# ---------------------------------------------------------------------------

def test_bm4_ps_ma_body_is_enemies_caves():
    """``bm4_ps_ma_body.bml`` must be categorized as an Enemies entry.

    The user reported it surfacing under Player Bodies because the
    legacy categorizer had a literal ``startswith('bm4_ps_')`` rule
    that matched the player-body bucket. The actual asset is the
    Sinow Beat / Sinow Gold (EP1 Caves) enemy family.
    """
    info = infer_category_full("bm4_ps_ma_body.bml")
    assert info is not None
    assert info["category"] == "Enemies"
    # Subcategory should mention Caves (legacy 'ma' subparts).
    assert "EP1 Caves" in info["subcategory"]
    # The in-game name ties it to Sinow Beat / Sinow Gold.
    assert "Sinow Beat" in info["in_game_name"]


def test_item_model_ep4_index_0297_is_mags():
    """``ItemModelEp4.afs#0297_..`` must route to Items.

    Lives in the EP4 mag-range slice of the AFS table; the legacy
    rule lumped everything in ``ItemModel*.afs`` into "Weapons / Items"
    without distinguishing weapons from mags.
    """
    info = infer_category_full(
        "ItemModelEp4.afs#0297_ItemModelEp4_0297.nj",
        parent_archive="ItemModelEp4.afs",
    )
    assert info is not None
    assert info["category"] == "Items"
    # The DB distinguishes EP1/2 vs EP4 item models via a per-archive
    # rule. Inner blob 0x297 should land in the EP4-archive subcat.
    assert "EP4" in info["subcategory"]


def test_bm_obj_warpboss_ancient_is_dark_falz_teleporter():
    """``bm_obj_warpboss_ancient.bml`` is the Dark Falz Boss Teleporter."""
    info = infer_category_full("bm_obj_warpboss_ancient.bml")
    assert info is not None
    assert info["category"] == "Objects"
    # Subcategory = boss-area warp; in-game name = Dark Falz teleporter.
    assert "warp" in info["subcategory"].lower() or "boss" in info["subcategory"].lower()
    assert "Dark Falz" in info["in_game_name"]


def test_bm_boss1_dragon_is_sil_dragon():
    """``bm_boss1_dragon.bml`` is the EP1 Forest boss Sil Dragon."""
    info = infer_category_full("bm_boss1_dragon.bml")
    assert info is not None
    assert info["category"] == "Bosses"
    assert "EP1" in info["subcategory"]
    assert "Sil Dragon" in info["in_game_name"]


def test_bm_boss8_dragon_is_gol_dragon():
    """``bm_boss8_dragon.bml`` is the EP4 Crater boss Gol Dragon.

    This is the asset the user pointed at for the "8 vs 1 dragon" naming
    confusion. Without the per-pattern rule both fell into a generic
    "Bosses" bucket with no way to distinguish them.
    """
    info = infer_category_full("bm_boss8_dragon.bml")
    assert info is not None
    assert info["category"] == "Bosses"
    assert "EP4" in info["subcategory"]
    assert "Gol Dragon" in info["in_game_name"]


# ---------------------------------------------------------------------------
# Sample of additional rules from across the 100+ DB entries
# ---------------------------------------------------------------------------

def test_player_body_glob_pattern():
    """``pl?bdy00.nj`` glob — single-char wildcard for the class slot."""
    assert infer_category("plAbdy00.nj") == "Player Bodies"
    assert infer_category("plRbdy00.nj") == "Player Bodies"
    assert infer_category("plZbdy00.nj") == "Player Bodies"


def test_player_class_headgear_glob():
    """``pl?cap??.nj`` matches FOmar caps / similar two-digit variants."""
    info = infer_category_full("plAcap06.nj")
    assert info is not None
    assert info["category"] == "Player Headgear"


def test_map_data_and_event_routing():
    """``map_*.bin`` is terrain, ``map_*.evt`` is a quest event script."""
    assert infer_category("map_ancient_e.bin") == "Maps / Terrain"
    info = infer_category_full("map_ancient_e.evt")
    assert info is not None
    assert info["category"] == "Quests"


def test_scene_dir_pattern():
    """The ``scene/*`` path-fragment rule maps to Maps / Terrain."""
    assert infer_category("scene/forest1/forest1.xvm") == "Maps / Terrain"
    assert infer_category("scene/cave/cave_01.nrel") == "Maps / Terrain"


def test_audio_dir_patterns():
    """``ogg/*`` is Audio (BGM); ``sound/*`` is Audio (SFX)."""
    info = infer_category_full("ogg/lobby.ogg")
    assert info is not None
    assert info["category"] == "Audio"
    assert "BGM" in info["in_game_name"]
    info = infer_category_full("sound/sfx/jingle.adx")
    assert info is not None
    assert info["category"] == "Audio"


def test_metadata_unitxt_pattern():
    """``unitxt_*.prs`` localized strings → Metadata."""
    assert infer_category("unitxt_e.prs") == "Metadata"
    assert infer_category("unitxt_j.prs") == "Metadata"


def test_ui_title_pattern():
    """``TitleEP4.prs`` is the EP4 title splash; LogoEP4.prs is dead."""
    info = infer_category_full("TitleEP4.prs")
    assert info is not None
    assert info["category"] == "UI"
    info_logo = infer_category_full("LogoEP4.prs")
    assert info_logo is not None
    assert info_logo["category"] == "UI"
    assert "dead" in info_logo["subcategory"].lower() or "dead" in info_logo["in_game_name"].lower()


def test_effects_pattern():
    """``bm_eff_*`` is Effects."""
    assert infer_category("bm_eff_ice.bml") == "Effects"


def test_npc_rico_pattern():
    """The Red Ring Rico story-NPC rule."""
    info = infer_category_full("rico_body.bml")
    assert info is not None
    assert info["category"] == "NPCs"
    assert "Rico" in info["in_game_name"]


def test_afs_inner_blob_via_parent_archive():
    """AFS inner blobs match by ``parent_archive`` not by basename.

    The synthesised path looks like ``ItemModel.afs#0042_inner.nj`` but
    the rule is keyed on the archive name (the bit before ``#``). This
    test exercises the AFS dispatch branch in ``_match_pattern``.
    """
    info = infer_category_full(
        "ItemModel.afs#0042_ItemModel_0042.nj",
        parent_archive="ItemModel.afs",
    )
    assert info is not None
    assert info["category"] == "Items"


def test_item_kt_routes_to_ui():
    """``ItemKT*.afs`` is the inventory-icon atlas — a UI asset."""
    info = infer_category_full(
        "ItemKT.afs#0000_ItemKT_0000.xvm",
        parent_archive="ItemKT.afs",
    )
    assert info is not None
    assert info["category"] == "UI"


# ---------------------------------------------------------------------------
# Fallback + ordering invariants
# ---------------------------------------------------------------------------

def test_unknown_path_returns_none():
    """An unknown path must return None (the canonical fallback signal).

    The ``inferred_category`` field is then absent from the AssetEntry
    and the asset tree groups by the canonical ``category`` instead.
    """
    assert infer_category("this_does_not_exist.totallyfake") is None
    assert infer_category_full("this_does_not_exist.totallyfake") is None


def test_empty_path_is_safe():
    """Defensive: empty input doesn't blow up."""
    assert infer_category("") is None
    assert infer_category(None) is None  # type: ignore[arg-type]


def test_specific_pattern_beats_general():
    """``bm_obj_boss8_*`` (Bosses) must beat ``bm_obj_*`` (Objects).

    This is the 'rule precedence' invariant — more-specific patterns
    appear before less-specific ones in the DB and the matcher MUST
    walk the list in order so the specific one wins.
    """
    # bm_obj_boss8_demoroom.bml -> Bosses (specific)
    assert infer_category("bm_obj_boss8_demoroom.bml") == "Bosses"
    # bm_obj_warpboss.bml -> Objects (specific subcategory)
    assert infer_category("bm_obj_warpboss.bml") == "Objects"
    # bm_obj_geenest.bml -> Objects (less specific)
    assert infer_category("bm_obj_geenest.bml") == "Objects"


def test_bm_boss_pattern_wins_over_generic_bm():
    """``bm_boss1_dragon*`` must match the Bosses bucket, not be lost
    in some generic ``bm_*`` rule (there isn't one, but the test
    documents the invariant)."""
    assert infer_category("bm_boss1_dragon.bml") == "Bosses"
    assert infer_category("bm_boss1_dragon_a.bml") == "Bosses"
    # Also verify glob suffix variants are caught (the trailing
    # ``*.bml`` on the rule allows ``_a.bml`` etc.).
    assert infer_category("bm_boss2_de_rol_le.bml") == "Bosses"
    assert infer_category("bm_boss2_de_rol_le_a.bml") == "Bosses"


def test_bm_ene_boss09_routes_to_bosses_not_enemies():
    """``bm_ene_boss09*`` must match the Bosses rule (it's the EP4 final
    boss family) even though the prefix looks like an enemy entry. The
    DB ordering puts the specific rule before any generic ``bm_ene_*``."""
    info = infer_category_full("bm_ene_boss09_a.bml")
    assert info is not None
    assert info["category"] == "Bosses"


# ---------------------------------------------------------------------------
# Direct exercise of the matcher helper for thoroughness
# ---------------------------------------------------------------------------

def test_match_pattern_basename_glob():
    assert _match_pattern("bm_boss1_dragon.bml", "", "", "bm_boss1_dragon*.bml")
    assert _match_pattern("bm_boss1_dragon_a.bml", "", "", "bm_boss1_dragon*.bml")
    assert not _match_pattern("bm_boss2_de_rol_le.bml", "", "", "bm_boss1_dragon*.bml")


def test_match_pattern_afs_archive():
    """AFS pattern matches by archive prefix (case-insensitive)."""
    assert _match_pattern(
        "itemmodel_0042.nj", "", "ItemModel.afs", "ItemModel.afs#*"
    )
    assert _match_pattern(
        "itemmodelep4_0297.nj", "", "ItemModelEp4.afs", "ItemModelEp4.afs#*"
    )
    # Wrong archive must not match.
    assert not _match_pattern(
        "itemmodel_0042.nj", "", "ItemModel.afs", "ItemModelEp4.afs#*"
    )


def test_match_pattern_afs_glob_archive():
    """``ItemKT*.afs#*`` should match any ItemKT-flavoured archive."""
    assert _match_pattern(
        "x.xvm", "", "ItemKT.afs", "ItemKT*.afs#*"
    )
    assert _match_pattern(
        "x.xvm", "", "ItemKTEp4.afs", "ItemKT*.afs#*"
    )


def test_match_pattern_path_fragment():
    """``scene/*`` matches anything under scene/."""
    assert _match_pattern("foo.xvm", "scene", "", "scene/*")
    assert _match_pattern("foo.xvm", "scene/forest", "", "scene/*")
    assert _match_pattern("foo.xvm", "data/scene/forest", "", "scene/*")
    assert not _match_pattern("foo.xvm", "data/maps", "", "scene/*")


def test_match_pattern_empty_pattern_is_false():
    """An empty pattern never matches (defensive)."""
    assert not _match_pattern("anything", "", "", "")


def test_archive_pattern_propagates_to_inner_blobs():
    """A pattern keyed on the archive name (e.g. ``pl?tex.afs``) must
    also tag every inner blob of that archive — otherwise AFS-inner
    blobs would silently fall back to ``Uncategorized`` even when the
    archive itself is well-known. The matcher checks ``parent_archive``
    against plain-glob patterns as a fallback after the basename try.
    """
    # plAtex.afs#0000_..., parent_archive='plAtex.afs', pattern='pl?tex.afs'
    assert infer_category(
        "plAtex.afs#0000_plAtex_0000.xvr",
        parent_archive="plAtex.afs",
    ) == "Player Misc"
    assert infer_category(
        "plRtex.afs#0123_plRtex_0123.xvr",
        parent_archive="plRtex.afs",
    ) == "Player Misc"
    # plZsmpnj.afs has its own pattern matching the archive
    assert infer_category(
        "plZsmpnj.afs#0000_plZsmpnj_0000.nj",
        parent_archive="plZsmpnj.afs",
    ) == "Player Misc"
