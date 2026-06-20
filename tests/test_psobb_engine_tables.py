"""Tests for the newserv-derived entity tables in ``psobb_engine_tables``.

The Section-2 portion of ``formats.psobb_engine_tables`` is a verbatim
port of newserv's ``Map.cc`` 518-entry id->class map (MIT). These
tests verify that every documented boss / cave-mob / NPC class is
discoverable by id, and that the per-floor enemy palette + SETDATA
basenames are reachable via the public API.

Sister file: ``tests/test_engine_tables.py`` covers the older
fog/light loading half of the same module — both files target the
same import path but exercise disjoint feature surfaces.
"""
from __future__ import annotations

from formats import psobb_engine_tables as PET


# --------------------------------------------------------------------------
# Boss class lookups (Map.cc:2942-2958)
# --------------------------------------------------------------------------
class TestBossLookups:
    def test_sil_dragon_class(self):
        d = PET.lookup_enemy(0x00C0)
        assert d is not None
        # The BB-version-preferred row is the Episode-1 Dragon.
        assert d.class_name == "TBoss1Dragon"
        assert d.display_name == "Dragon"

    def test_dragon_id_has_two_episode_variants(self):
        defs = PET.all_defs_for_type(PET.ENEMY_TABLE_ROWS, 0x00C0)
        names = {d.class_name for d in defs}
        # Same id is reused for the Episode-2 Gal Gryphon boss.
        assert "TBoss1Dragon" in names
        assert "TBoss5Gryphon" in names

    def test_de_rol_le(self):
        d = PET.lookup_enemy(0x00C1)
        assert d is not None
        assert d.class_name == "TBoss2DeRolLe"
        assert d.display_name == "De Rol Le"

    def test_vol_opt_main(self):
        d = PET.lookup_enemy(0x00C2)
        assert d is not None
        assert d.class_name == "TBoss3Volopt"

    def test_vol_opt_sub_parts(self):
        # Vol Opt's secondary subparts each have a distinct id.
        for tid, expected in [
            (0x00C3, "TBoss3VoloptP01"),
            (0x00C4, "TBoss3VoloptCore"),
            (0x00C5, "TBoss3VoloptP02"),
            (0x00C6, "TBoss3VoloptMonitor"),
            (0x00C7, "TBoss3VoloptHiraisin"),
        ]:
            d = PET.lookup_enemy(tid)
            assert d is not None, f"missing id 0x{tid:04X}"
            assert d.class_name == expected

    def test_dark_falz(self):
        d = PET.lookup_enemy(0x00C8)
        assert d is not None
        assert d.class_name == "TBoss4DarkFalz"

    def test_gol_dragon(self):
        d = PET.lookup_enemy(0x00CC)
        assert d is not None
        assert d.class_name == "TBoss8Dragon"
        assert d.display_name == "Gol Dragon"


# --------------------------------------------------------------------------
# Mid-game enemy classes
# --------------------------------------------------------------------------
class TestEnemyLookups:
    def test_mericarol(self):
        d = PET.lookup_enemy(0x00D6)
        assert d is not None
        assert d.class_name == "TObjEneBm9Mericarol"

    def test_pal_shark_class_is_shared(self):
        # Pal Shark / Evil Shark / Guil Shark all share the
        # TObjEneShark class; the variant is selected via param6.
        d = PET.lookup_enemy(0x0063)
        assert d is not None
        assert d.class_name == "TObjEneShark"

    def test_sinow_beat_gold_share_class(self):
        # Sinow Beat / Sinow Gold are both TObjEneMe3ShinowaReal,
        # gated by param2 (>= 1.0 = Gold).
        d = PET.lookup_enemy(0x0082)
        assert d is not None
        assert d.class_name == "TObjEneMe3ShinowaReal"

    def test_lookup_unknown_returns_none(self):
        # Definitely-out-of-range id.
        assert PET.lookup_enemy(0xBEEF) is None
        assert PET.lookup_object(0xBEEF) is None


# --------------------------------------------------------------------------
# Object lookups (Map.cc:618-2492)
# --------------------------------------------------------------------------
class TestObjectLookups:
    def test_player_set_id_zero(self):
        d = PET.lookup_object(0x0000)
        assert d is not None
        assert d.class_name == "TObjPlayerSet"

    def test_warp_forest(self):
        d = PET.lookup_object(0x0002)
        assert d is not None
        assert d.class_name == "TObjAreaWarpForest"

    def test_boss_warp(self):
        d = PET.lookup_object(0x0019)
        assert d is not None
        assert d.class_name == "TObjWarpBoss"


# --------------------------------------------------------------------------
# Per-floor enemy palette + SETDATA basenames
# --------------------------------------------------------------------------
class TestPerFloorIndex:
    def test_forest1_includes_box_and_rappy(self):
        slots = PET.PER_FLOOR_ENEMY_INDEX[(1, 0x01)]
        # 0x01 = Box, 0x05 = Rag Rappy, 0x0B = Chao
        assert 0x01 in slots
        assert 0x05 in slots
        assert 0x0B in slots

    def test_cave1_includes_pan_arms_and_pal_shark(self):
        slots = PET.PER_FLOOR_ENEMY_INDEX[(1, 0x03)]
        # 0x23 = Pan Arms, 0x30 = Pal Shark
        assert 0x23 in slots
        assert 0x30 in slots

    def test_mine1_includes_sinow_beat(self):
        slots = PET.PER_FLOOR_ENEMY_INDEX[(1, 0x06)]
        # 0x26 = Sinow Beat, 0x27 = Sinow Gold
        assert 0x26 in slots
        assert 0x27 in slots

    def test_dragon_floor_only_has_box(self):
        slots = PET.PER_FLOOR_ENEMY_INDEX[(1, 0x0B)]
        assert slots == (0x01,)

    def test_pioneer2_has_no_enemies(self):
        assert PET.PER_FLOOR_ENEMY_INDEX[(1, 0x00)] == ()


class TestSetDataNames:
    def test_forest1(self):
        assert PET.SETDATA_NAMES[(1, 0x01)] == "map_forest01_00"

    def test_boss01_dragon(self):
        assert PET.SETDATA_NAMES[(1, 0x0B)] == "map_boss01"

    def test_visuallobby(self):
        assert PET.SETDATA_NAMES[(1, 0x0F)] == "map_visuallobby"

    def test_episode2_seabed(self):
        assert PET.SETDATA_NAMES[(2, 0x1C)] == "map_seabed01"

    def test_episode4_wilds(self):
        assert PET.SETDATA_NAMES[(4, 0x24)] == "map_wilds01"


# --------------------------------------------------------------------------
# Source-shape sanity (port-fidelity guards)
# --------------------------------------------------------------------------
class TestPortFidelity:
    def test_total_row_count_matches_newserv(self):
        # 359 object rows + 159 enemy rows = 518 — the headline number
        # in `_reports/external_tools_survey.md`.
        assert len(PET.OBJECT_TABLE_ROWS) == 359
        assert len(PET.ENEMY_TABLE_ROWS) == 159
        assert len(PET.OBJECT_TABLE_ROWS) + len(PET.ENEMY_TABLE_ROWS) == 518

    def test_no_collapse_loses_unique_ids(self):
        # Each canonical lookup table should contain a row for every
        # distinct type_id seen in the source rows.
        obj_ids = {row[0] for row in PET.OBJECT_TABLE_ROWS}
        ene_ids = {row[0] for row in PET.ENEMY_TABLE_ROWS}
        assert obj_ids == set(PET.OBJECT_TABLE.keys())
        assert ene_ids == set(PET.ENEMY_TABLE.keys())

    def test_version_flag_bits_includes_bb(self):
        assert "F_V4" in PET.VERSION_FLAG_BITS
        assert PET.VERSION_FLAG_BITS["F_V4"] == 0x2000
        assert PET.VERSION_FLAG_BITS["F_V0_V4"] & PET.F_V4_MASK
