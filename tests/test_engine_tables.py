"""Tests for ``formats.psobb_engine_tables`` (Editor v4 RE).

The fixture data lives under the user's PSOBB.IO install — these tests
skip when the install isn't available, so the suite stays clean for
CI runs that don't have access to the proprietary game data.
"""
from __future__ import annotations
import os

import struct
from pathlib import Path

import pytest

from formats import psobb_engine_tables as PET


_INSTALL = Path(os.path.expanduser("~/PSOBB.IO"))
_DATA = _INSTALL / "data"
_HAS_INSTALL = (_DATA / "fogentry.dat").exists() and (_DATA / "lightentry.bin").exists()


pytestmark = pytest.mark.skipif(
    not _HAS_INSTALL,
    reason="PSOBB.IO install not present (fogentry.dat / lightentry.bin missing)",
)


# --------------------------------------------------------------------------
# Constants & sanity
# --------------------------------------------------------------------------
class TestConstants:
    def test_fog_entry_size(self):
        assert PET.FOG_ENTRY_SIZE == 64

    def test_fog_table_count(self):
        assert PET.FOG_TABLE_COUNT == 256

    def test_fog_table_bytes(self):
        # The PSOBB runtime allocates 0x4000 bytes for the fog table —
        # this is the source of truth (binary at 0x005bdb8d).
        assert PET.FOG_TABLE_COUNT * PET.FOG_ENTRY_SIZE == 0x4000

    def test_light_entry_size(self):
        # 17 floats per BBPP sunlight.cpp.
        assert PET.LIGHT_ENTRY_SIZE == 68

    def test_light_table_count(self):
        # 48 normal + 48 ultimate-ep1 = 96.
        assert PET.LIGHT_TABLE_COUNT == 96


# --------------------------------------------------------------------------
# Fog table loading
# --------------------------------------------------------------------------
class TestFogTable:
    def test_load_returns_count(self):
        fog = PET.load_fog_table()
        assert len(fog) == PET.FOG_TABLE_COUNT

    def test_pioneer2_entry(self):
        # Index 0 is Pioneer2_Ep1.  Live values from the install:
        #   type=1, color=#2d3f30, end=700, start=100.
        # These won't change unless the player edits fogentry.dat.
        fog = PET.load_fog_table()
        e0 = fog[0]
        assert e0.type == 1
        assert e0.color_rgb_hex == 0x2D3F30
        assert e0.end == pytest.approx(700.0, abs=0.5)
        assert e0.start == pytest.approx(100.0, abs=0.5)

    def test_forest1_entry(self):
        fog = PET.load_fog_table()
        e1 = fog[1]
        assert e1.type == 1
        # Pale-green fog colour for forest.
        assert e1.color_rgb_hex == 0x80FFAA

    def test_cave1_entry(self):
        fog = PET.load_fog_table()
        e3 = fog[3]
        # Torch-orange fog
        assert e3.color_rgb_hex == 0xFF7000

    def test_color_bgra_decode(self):
        """The disk byte order is BGRA but ``color_rgb_hex`` swizzles to RGB."""
        fog = PET.load_fog_table()
        e0 = fog[0]
        # On disk: B=0x30, G=0x3f, R=0x2d, A=0x00 → RGB hex 0x2d3f30
        assert e0.color_b == 0x30
        assert e0.color_g == 0x3F
        assert e0.color_r == 0x2D
        assert e0.color_rgb_hex == (e0.color_r << 16) | (e0.color_g << 8) | e0.color_b

    def test_to_dict_shape(self):
        fog = PET.load_fog_table()
        d = fog[0].to_dict()
        # Every JSON-shipping field present.
        for k in ("type", "color_rgb", "color_a", "end", "start",
                  "density", "transition", "end_pulse_distance",
                  "start_pulse_distance"):
            assert k in d


# --------------------------------------------------------------------------
# Light table loading
# --------------------------------------------------------------------------
class TestLightTable:
    def test_load_returns_count(self):
        lt = PET.load_light_table()
        assert len(lt) == PET.LIGHT_TABLE_COUNT

    def test_dir_vector_shape(self):
        lt = PET.load_light_table()
        e0 = lt[0]
        assert isinstance(e0.dir1, tuple)
        assert len(e0.dir1) == 3
        assert isinstance(e0.dir2, tuple)
        assert len(e0.dir2) == 3

    def test_argb_shape(self):
        lt = PET.load_light_table()
        e0 = lt[0]
        assert len(e0.diffuse_argb) == 4
        assert len(e0.ambient_argb) == 4
        # All entries observed have intensity components in [0, 2] range.
        assert 0.0 <= e0.intensity_diffuse <= 5.0
        assert 0.0 <= e0.intensity_ambient <= 5.0

    def test_to_dict_shape(self):
        lt = PET.load_light_table()
        d = lt[0].to_dict()
        for k in ("dir1", "dir2",
                  "intensity_specular", "intensity_diffuse", "intensity_ambient",
                  "diffuse_argb", "ambient_argb"):
            assert k in d


# --------------------------------------------------------------------------
# map_id_to_engine_index
# --------------------------------------------------------------------------
class TestMapIdResolution:
    def test_pioneer2(self):
        assert PET.map_id_to_engine_index("city00") == 0
        assert PET.map_id_to_engine_index("acity00") == 0
        assert PET.map_id_to_engine_index("labo00") == 0

    def test_forest_alt_alias(self):
        # ancient01 and aancient01 both map to Forest1 (engine index 1).
        assert PET.map_id_to_engine_index("ancient01") == 1
        assert PET.map_id_to_engine_index("aancient01") == 1
        assert PET.map_id_to_engine_index("ancient02") == 2
        assert PET.map_id_to_engine_index("aancient02") == 2

    def test_caves(self):
        assert PET.map_id_to_engine_index("cave01") == 3
        assert PET.map_id_to_engine_index("cave02") == 4
        assert PET.map_id_to_engine_index("cave03") == 5
        assert PET.map_id_to_engine_index("acave01") == 3

    def test_mines(self):
        assert PET.map_id_to_engine_index("machine01") == 6
        assert PET.map_id_to_engine_index("machine02") == 7

    def test_ruins(self):
        assert PET.map_id_to_engine_index("ruins01") == 8
        assert PET.map_id_to_engine_index("ruins02") == 9

    def test_episode_2(self):
        assert PET.map_id_to_engine_index("city01") == 18
        assert PET.map_id_to_engine_index("jungle01") == 19  # Temple_A

    def test_episode_4(self):
        assert PET.map_id_to_engine_index("wilds01") == 36
        assert PET.map_id_to_engine_index("desert01") == 41

    def test_unknown_map_id(self):
        assert PET.map_id_to_engine_index("garbage_99") is None
        assert PET.map_id_to_engine_index("") is None
        assert PET.map_id_to_engine_index(None) is None  # type: ignore[arg-type]

    def test_ultimate_offset(self):
        """Ultimate Ep1 light entries live at base+48 in lightentry.bin."""
        # Forest1 normal = 1; ultimate = 49.
        assert PET.map_id_to_engine_index("aancient01") == 1
        assert PET.map_id_to_engine_index("aancient01", ultimate=True) == 49
        # Ep2+ has no ultimate offset.
        assert PET.map_id_to_engine_index("city01", ultimate=True) == 18


# --------------------------------------------------------------------------
# Top-level builder
# --------------------------------------------------------------------------
class TestEnvTable:
    def test_builds_for_known_ids(self):
        table = PET.build_engine_env_table()
        # Sanity: at least every Episode-1 dungeon / boss should be present.
        for mid in ("city00", "aancient01", "cave01", "machine01",
                    "ruins01", "boss01", "boss04"):
            assert mid in table, f"missing {mid}"

    def test_env_record_shape(self):
        table = PET.build_engine_env_table()
        e = table["aancient01"]
        assert e.map_id == "aancient01"
        assert e.map_type == "Forest1"
        assert e.engine_index == 1
        assert isinstance(e.fog, PET.FogEntry)
        assert isinstance(e.light, PET.LightEntry)
        # Forest1 has an ultimate light entry; light_ultimate must be set.
        assert e.light_ultimate is not None
        assert isinstance(e.light_ultimate, PET.LightEntry)

    def test_env_to_dict_serialisable(self):
        import json
        table = PET.build_engine_env_table()
        d = {k: v.to_dict() for k, v in list(table.items())[:5]}
        # Round-tripping through JSON shouldn't lose anything we serialise.
        s = json.dumps(d)
        assert "fog" in s and "light" in s

    def test_env_dict_top_level(self):
        d = PET.build_engine_env_dict()
        assert isinstance(d, dict)
        assert "aancient01" in d
        assert d["aancient01"]["fog"]["type"] in (0, 1, 2)


# --------------------------------------------------------------------------
# Cross-check: live FogEntry record sizes match what the runtime expects
# --------------------------------------------------------------------------
class TestStructAlignment:
    def test_fog_struct_packed(self):
        """Confirm we don't accidentally pull in alignment padding."""
        # 4(u32) + 4(4 u8) + 12(3 f32) + 4(u32) + 8(2 f32) + 4(f32)
        # + 4(u32) + 4(f32) + 4(u32) + 4(f32) + 4(u32) + 8(8 u8) = 64
        size = struct.calcsize("<I 4B fff I ff f I f I f I 8B")
        assert size == PET.FOG_ENTRY_SIZE

    def test_light_struct_packed(self):
        size = struct.calcsize("<17f")
        assert size == PET.LIGHT_ENTRY_SIZE
