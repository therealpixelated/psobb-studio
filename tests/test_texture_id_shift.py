"""Texture-id shift regression tests for multi-inner BML composites.

Pin Phantasmal's `shiftTextureIds` invariant in our backend: when a BML
packs N inners, each inner's binding response must report a per-inner
``cumulative_offset`` so the composite renderer can keep inner-N's
texture-id range disjoint from inner-0..N-1.

The synthetic test below builds a 2-inner BML in memory (a fake
filesystem record with a stubbed `_compute_bml_inner_tex_offsets`) and
checks the offset arithmetic. The live-data tests downstream (curl
smoke) verify the server returns the field for real BMLs like
``bm_obj_warpboss_ancient.bml``.

Reference:
  - Phantasmal ``CharacterClassAssetLoader.kt:88-98`` — `shiftTextureIds`
  - Phantasmal ``EntityAssetLoader.kt:127-132`` — multi-inner addChild
  - ``_reports/phantasmal_asset_loading_spec.md`` §3.4
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


# --------------------------------------------------------------------------
# Pure logic — verify cumulative offset arithmetic
# --------------------------------------------------------------------------
class TestCumulativeOffsetArithmetic:
    def test_two_inners_three_then_two(self):
        """Phantasmal-style: inner-0 has 3 tiles, inner-1 has 2.
        Inner-1's submesh-0 must reference tile-3 (not tile-0)."""
        per_inner_counts = [3, 2]
        offsets = []
        cumulative = 0
        for tc in per_inner_counts:
            offsets.append(cumulative)
            cumulative += tc
        assert offsets == [0, 3]
        # For inner-1, submesh-0 (mat_id 0) shifts to tile 0 + 3 = 3.
        assert 0 + offsets[1] == 3
        # Inner-0's submesh-0 stays at tile 0 (no double-shift).
        assert 0 + offsets[0] == 0

    def test_three_inners_5_3_2(self):
        per_inner_counts = [5, 3, 2]
        offsets = []
        cumulative = 0
        for tc in per_inner_counts:
            offsets.append(cumulative)
            cumulative += tc
        assert offsets == [0, 5, 8]
        # Inner-2's submesh-1 lands at tile 8 + 1 = 9.
        assert 1 + offsets[2] == 9

    def test_single_inner_no_shift(self):
        per_inner_counts = [4]
        offsets = []
        cumulative = 0
        for tc in per_inner_counts:
            offsets.append(cumulative)
            cumulative += tc
        # No second inner — shift table has just the one zero.
        assert offsets == [0]


# --------------------------------------------------------------------------
# Server helper: _compute_bml_inner_tex_offsets shape
# --------------------------------------------------------------------------
class TestComputeBmlInnerTexOffsets:
    """Shape-of-result tests for the helper added in 2026-04-26.

    The helper uses ``parse_bml`` + ``extract_bml_texture`` +
    ``_list_xvmh_records`` from the production code path; we mock
    those out and feed in synthetic data so the test runs without a
    live install.
    """

    def _make_entry(self, name, has_tex):
        # parse_bml entries expose .name and .has_texture.
        class _E:
            pass
        e = _E()
        e.name = name
        e.has_texture = has_tex
        return e

    def test_two_inner_synthetic(self):
        import server

        entries = [
            self._make_entry("inner_a.nj", True),
            self._make_entry("inner_b.nj", False),
        ]
        # Each inner's xvm: 3 records for inner_a, 0 for inner_b.
        # _compute_bml_inner_tex_offsets now magic-routes via
        # _list_texture_records (XVMH/PVMH/GVMH) instead of assuming XVMH.
        with patch.object(server, "parse_bml", return_value=entries), \
             patch.object(server, "extract_bml_texture", return_value=b"FAKE_XVM"), \
             patch.object(server, "_list_texture_records",
                          return_value=[{"tile_index": i} for i in range(3)]):
            from pathlib import Path
            with patch.object(Path, "read_bytes", return_value=b"FAKE_BML"):
                offsets = server._compute_bml_inner_tex_offsets(Path("fake.bml"))
        # Two inners reported, in order.
        assert len(offsets) == 2
        assert offsets[0]["name"] == "inner_a.nj"
        assert offsets[0]["tile_count"] == 3
        assert offsets[0]["cumulative_offset"] == 0
        assert offsets[1]["name"] == "inner_b.nj"
        # inner_b has no texture so tile_count = 0.
        assert offsets[1]["tile_count"] == 0
        # Cumulative offset for inner_b is 0 + 3 = 3.
        assert offsets[1]["cumulative_offset"] == 3

    def test_synthetic_three_inner_mixed_counts(self):
        import server

        entries = [
            self._make_entry("a.nj", True),
            self._make_entry("b.nj", True),
            self._make_entry("c.xj", True),
        ]
        # Per-inner record counts via a mock side_effect.
        counts = iter([
            [{"tile_index": i} for i in range(3)],   # a.nj -> 3
            [{"tile_index": i} for i in range(2)],   # b.nj -> 2
            [{"tile_index": i} for i in range(4)],   # c.xj -> 4
        ])
        with patch.object(server, "parse_bml", return_value=entries), \
             patch.object(server, "extract_bml_texture", return_value=b"FAKE"), \
             patch.object(server, "_list_texture_records",
                          side_effect=lambda *_a, **_k: next(counts)):
            from pathlib import Path
            with patch.object(Path, "read_bytes", return_value=b"FAKE"):
                offsets = server._compute_bml_inner_tex_offsets(Path("fake.bml"))
        assert [o["tile_count"] for o in offsets] == [3, 2, 4]
        assert [o["cumulative_offset"] for o in offsets] == [0, 3, 5]
        # Inner-1's submesh-0 lands at tile 0 + 3 = 3 in shared space.
        assert 0 + offsets[1]["cumulative_offset"] == 3
        # Inner-0's submesh-0 lands at tile 0 (no shift).
        assert 0 + offsets[0]["cumulative_offset"] == 0

    def test_skips_njm_and_other_non_geometry(self):
        import server

        entries = [
            self._make_entry("body.nj", True),
            self._make_entry("walk.njm", False),       # animation, not geometry
            self._make_entry("body.xvm", False),       # raw texture
        ]
        with patch.object(server, "parse_bml", return_value=entries), \
             patch.object(server, "extract_bml_texture", return_value=b"FAKE"), \
             patch.object(server, "_list_texture_records",
                          return_value=[{"tile_index": 0}]):
            from pathlib import Path
            with patch.object(Path, "read_bytes", return_value=b"FAKE"):
                offsets = server._compute_bml_inner_tex_offsets(Path("fake.bml"))
        # Only the .nj entry should appear.
        assert len(offsets) == 1
        assert offsets[0]["name"] == "body.nj"
        assert offsets[0]["tile_count"] == 1
        assert offsets[0]["cumulative_offset"] == 0

    def test_unparseable_bml_returns_empty(self):
        import server

        with patch.object(server, "parse_bml",
                          side_effect=ValueError("bad magic")):
            from pathlib import Path
            with patch.object(Path, "read_bytes", return_value=b"GARBAGE"):
                offsets = server._compute_bml_inner_tex_offsets(Path("fake.bml"))
        assert offsets == []
