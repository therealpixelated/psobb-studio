"""Atlas layout knowledge base for the PSOBB Texture Editor.

Some PRS/XVM bundles hold tiles that are *spatially* tiled — the engine
renders multiple tiles edge-to-edge to form a single composite splash /
poster / billboard. Editing such a file tile-by-tile makes any AI upscaler
(or human texture artist) fight an artificial seam, because each tile
lacks the surrounding context.

This module hardcodes — by filename — the spatial relationship between a
file's tiles, so the editor's "atlas mode" can:

  1. Stitch the live source tiles into one composite image.
  2. Show that composite as a single editable canvas.
  3. After upscale (or external import), slice the composite back into
     per-tile crops at native dim, and register each crop as a normal
     edit in the existing repack pipeline.

Layout schema
-------------

Each entry is keyed by the file's basename (case-sensitive) and has:

    kind            : str   — informational, "screen_atlas" for now
    composite_w     : int   — assembled canvas width  (px)
    composite_h     : int   — assembled canvas height (px)
    placements      : list of dicts, each:
        tile_index  : int   — which tile from the file goes here
        x, y, w, h  : int   — destination rect on the composite
        uv_box      : 4-tuple (u0,v0,u1,v1) of source UV coverage,
                      default (0,0,1,1).  Allows future support for
                      sub-tile crops; for the LogoEP4 splash we use
                      full (0,0,1,1) since the engine samples the whole
                      tile.
    skip_tiles      : list of tile_index values NOT in the atlas
                      (e.g. pillar fills the user shouldn't edit through
                      the composite — they round-trip via the tile grid).
    source          : str — provenance comment.

The first entry in this file MUST exactly reflect the on-screen layout
documented in C:/tmp_logo_focus/LAYOUT.md and verified visually against
research/screen_truth_native.png. The composite is the rectangle that
contains the four big tiles in their *engine-spatial* order, which is:

    tile_0 (top-left screen) | tile_2 (top-right screen)
    tile_1 (bot-left screen) | tile_3 (bot-right screen)

This is NOT the file-order 2x2 grid (tile_0,1 over 2,3); it's the
true on-screen spatial order. See LAYOUT.md "Verdict" for why this
matters.

Sanity-check at module load
---------------------------

For each layout we assert:
  * placements stay inside the composite bounds
  * tile_index values appear at most once
  * uv_box is well-formed
"""
from __future__ import annotations

from typing import Optional


# Per-placement default UV box: full tile, no sub-cropping.
DEFAULT_UV = (0.0, 0.0, 1.0, 1.0)


def _placement(
    tile_index: int,
    x: int,
    y: int,
    w: int,
    h: int,
    uv_box: Optional[tuple[float, float, float, float]] = None,
) -> dict:
    """Build a placement dict, defaulting uv_box to (0,0,1,1)."""
    return {
        "tile_index": tile_index,
        "x": int(x),
        "y": int(y),
        "w": int(w),
        "h": int(h),
        "uv_box": tuple(uv_box) if uv_box is not None else DEFAULT_UV,
    }


# ---------------------------------------------------------------------------
# Hardcoded layouts.
#
# Add a new entry by:
#   1. Researching the engine's per-tile screen rect (e.g. via static
#      analysis of the render fcn, or live capture of quads.csv).
#   2. Building a 2D rectangle that wraps all big-content tiles in the
#      engine's spatial order — usually the bounding box of the screen
#      rects, scaled up to native tile dim so no resampling is needed.
#   3. Listing each (tile_index, x, y, w, h) placement.
#   4. Adding any pure-fill / non-content tiles to skip_tiles.
# ---------------------------------------------------------------------------
ATLAS_LAYOUTS: dict[str, dict] = {
    "LogoEP4.prs": {
        "kind": "screen_atlas",
        # Composite is the bounding box of the 4 big tiles laid out in
        # their on-screen 2x2 order. Each big tile is 1024x1024 native;
        # placing them at native dim avoids any resampling on assembly.
        "composite_w": 2048,
        "composite_h": 2048,
        "placements": [
            # Engine-spatial order:
            #   tile_0 | tile_2
            #   tile_1 | tile_3
            # (LAYOUT.md section 2: "tile_0 is top-left screen", etc.)
            _placement(0,    0,    0, 1024, 1024),  # top-left
            _placement(2, 1024,    0, 1024, 1024),  # top-right
            _placement(1,    0, 1024, 1024, 1024),  # bottom-left
            _placement(3, 1024, 1024, 1024, 1024),  # bottom-right
        ],
        # Tiles 4..7 are pure-white pillar fills (right-edge of the 4:3
        # canvas). They round-trip losslessly and are NOT part of the
        # composite the user edits.
        "skip_tiles": [4, 5, 6, 7],
        "source": "C:/tmp_logo_focus/LAYOUT.md (static analysis 2026-04-25)",
    },
}


def _validate_layout(filename: str, layout: dict) -> None:
    """Assert a layout entry is well-formed. Called at module import."""
    cw = int(layout["composite_w"])
    ch = int(layout["composite_h"])
    if cw <= 0 or ch <= 0:
        raise ValueError(f"{filename}: composite must have positive size, got {cw}x{ch}")

    seen: set[int] = set()
    for p in layout["placements"]:
        idx = p["tile_index"]
        if idx in seen:
            raise ValueError(f"{filename}: tile_index {idx} appears twice in placements")
        seen.add(idx)
        x, y, w, h = p["x"], p["y"], p["w"], p["h"]
        if w <= 0 or h <= 0:
            raise ValueError(f"{filename}: placement tile {idx} has non-positive size {w}x{h}")
        if x < 0 or y < 0:
            raise ValueError(f"{filename}: placement tile {idx} has negative offset ({x},{y})")
        if x + w > cw or y + h > ch:
            raise ValueError(
                f"{filename}: placement tile {idx} ({x},{y} +{w}x{h}) escapes "
                f"composite {cw}x{ch}"
            )
        u0, v0, u1, v1 = p["uv_box"]
        if not (0.0 <= u0 < u1 <= 1.0 and 0.0 <= v0 < v1 <= 1.0):
            raise ValueError(
                f"{filename}: placement tile {idx} uv_box {p['uv_box']} is malformed"
            )

    skip = layout.get("skip_tiles", [])
    for s in skip:
        if s in seen:
            raise ValueError(
                f"{filename}: tile {s} cannot be both placed and skipped"
            )


# Validate every entry at import time so a broken layout fails the server
# at startup, not at first request.
for _fn, _ly in ATLAS_LAYOUTS.items():
    _validate_layout(_fn, _ly)


def get_layout(filename: str) -> Optional[dict]:
    """Return the layout dict for `filename`, or None if no layout is known."""
    return ATLAS_LAYOUTS.get(filename)


def has_layout(filename: str) -> bool:
    """True if the given filename has a known atlas layout."""
    return filename in ATLAS_LAYOUTS
