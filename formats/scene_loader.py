"""Scene-loader for the Map Editor perspective (2026-04-25).

PSOBB ships terrain / props / lights as a tree of files under
``data/scene/``. Each map has the shape::

    map_<area><NN>_<floorNN><suffix>.{nj,xj,xvm,rel,bin,tam,scc,tls,njm}

For example, ``map_aancient01_00s.nj`` is "ancient ruins, area 1, floor 0,
suffix s (= the static terrain mesh)".

This module is the catalogue + asset-bundler. It reads the existing
manifest and groups every scene file by ``(area, area_num, floor)`` so the
Map Editor can:

  1. Show a dropdown of *maps* (28 areas grouped into ~9 categories),
  2. Pick a *floor* (most maps have 5 floors),
  3. Load every NJ/XJ/XVM file for that (map, floor) tuple in parallel.

The Map Editor does **not** parse the .rel files (those are PSOBB's
relocation-aware quest scripts and are out of scope for v1). It also
does not parse `.bin` collision data — only the renderable terrain.

Scope:
  - **Read-only API** — the Map Editor's spawn/waypoint edits go to a
    separate JSON sidecar at ``cache/map_edits/<map_id>.json`` (see
    ``server.py::api_map_edits_save``). No code in this module writes
    to disk.
  - **Pure** — no FastAPI / threading / HTTP. The server layer wraps
    this with caching + HTTP serialization.

The classification table at :data:`AREA_CATEGORY` mirrors the PSOBB
canonical area-id table. Every entry that doesn't match an explicit row
falls into the ``other`` bucket.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional


# Match PSOBB's canonical area names: aancient01, acave02, machine01, ...
# Pattern: "map_" + area letters + (01..09) + "_" + floor digits + suffix.
# Real-world examples from the manifest:
#   map_aancient01_00c.rel
#   map_aancient01_00n.rel
#   map_aancient01_00r.rel
#   map_aancient01_00s.nj
#   map_aancient01_00s.xj
#   map_aancient01_00s.xvm
#   map_aancient01_00.tam
#   map_aancient01_00bm.bin
_MAP_FILE_RE = re.compile(
    r"^map_([a-z]+)(\d+)_(\d+)([a-z]*)\.([a-z0-9]+)$",
    re.IGNORECASE,
)


# PSOBB area → user-facing category. Areas not listed are "other".
# Order matters only for dropdown grouping; the Map Editor sorts by
# (category_order_index, area_label).
AREA_CATEGORY: dict[str, tuple[str, str]] = {
    # area prefix    → (category, friendly label)
    "city":     ("city",       "Pioneer 2 / City"),
    "acity":    ("city",       "Pioneer 2 / City (alt)"),
    "labo":     ("city",       "Lab / Pioneer 2"),
    "ancient":  ("forest",     "Forest"),
    "aancient": ("forest",     "Forest (alt)"),
    "cave":     ("cave",       "Cave"),
    "acave":    ("cave",       "Cave (alt)"),
    "machine":  ("mine",       "Mine"),
    "amachine": ("mine",       "Mine (alt)"),
    "ruins":    ("ruins",      "Ruins"),
    "vs":       ("battle",     "Battle (Versus)"),
    "jungle":   ("corruption", "Jungle / Episode II"),
    "wilds":    ("corruption", "Wilds"),
    "desert":   ("corruption", "Desert / Episode IV"),
    "crater":   ("corruption", "Crater"),
    "seabed":   ("corruption", "Seabed"),
    "space":    ("corruption", "Space"),
    "boss":     ("boss",       "Boss arena"),
    "test":     ("other",      "Test"),
}

# Extension classification — only RENDERABLE assets count for the Map
# Editor's load-bundle. Other extensions are listed for diagnostic
# completeness (so the UI can show "this map ships 4 NJM motions" etc).
RENDERABLE_EXTS = ("nj", "xj")
TEXTURE_EXTS = ("xvm",)
SCRIPT_EXTS = ("rel",)
COLLISION_EXTS = ("bin",)
ANIMATION_EXTS = ("njm",)
OTHER_EXTS = ("tam", "scc", "tls")


@dataclass(frozen=True)
class MapAsset:
    """One file inside a (map, floor) tuple.

    ``floor`` and ``suffix`` come from the filename. ``suffix`` is the
    optional letter cluster between the floor digits and the extension —
    PSOBB uses 's' for static (renderable terrain), 'c' / 'n' / 'r' for
    relocation tables, 'bm' for collision binaries, etc.
    """
    path: str             # manifest-relative ("scene/map_aancient01_00s.nj")
    area: str             # "aancient"
    area_num: int         # 1
    floor: int            # 0
    suffix: str           # "s"
    ext: str              # "nj" / "xj" / "xvm" / "rel" / ...
    size: int = 0         # bytes from the manifest entry


@dataclass
class MapInfo:
    """One pickable entry in the Map Editor's "map picker" dropdown.

    A *map* is a (area, area_num) pair — e.g. ("aancient", 1). Within a
    map the ``floors`` table lists every floor that has at least one
    asset on disk (most have 0..4, boss arenas have 0..2, city has 0).
    """
    map_id: str           # "aancient01" — the PSOBB convention
    area: str             # "aancient"
    area_num: int         # 1
    category: str         # "forest"
    label: str            # "Forest (alt)"
    floors: dict[int, list[MapAsset]] = field(default_factory=dict)
    # Aggregated stats (filled by ``catalogue()``)
    total_files: int = 0
    renderable_files: int = 0
    total_bytes: int = 0


def _parse_filename(path: str) -> Optional[MapAsset]:
    """Parse one ``scene/map_*`` path. Return None if it's not a map file."""
    name = path.rsplit("/", 1)[-1]
    m = _MAP_FILE_RE.match(name)
    if not m:
        return None
    area, area_num, floor, suffix, ext = m.groups()
    try:
        area_num_i = int(area_num)
        floor_i = int(floor)
    except ValueError:
        return None
    return MapAsset(
        path=path,
        area=area.lower(),
        area_num=area_num_i,
        floor=floor_i,
        suffix=suffix.lower(),
        ext=ext.lower(),
    )


def _classify_area(area: str) -> tuple[str, str]:
    """Return (category, label) for an area prefix. Falls back to 'other'."""
    return AREA_CATEGORY.get(area.lower(), ("other", area.capitalize()))


def catalogue(manifest_entries: Iterable[dict]) -> list[MapInfo]:
    """Build a list of :class:`MapInfo` from manifest entries.

    Args:
      manifest_entries: iterable of dicts shaped like the manifest's
        ``entries`` list: ``{path, category, size, ...}``.

    Returns:
      A list of :class:`MapInfo` in stable (category, area, area_num)
      order. Each MapInfo's ``floors`` dict groups its assets by floor.
      Maps with zero renderable assets are still listed (e.g. boss
      arenas can be terrain-only, no .nj on some floors) — the picker
      shows them grayed-out.
    """
    by_id: dict[str, MapInfo] = {}
    for entry in manifest_entries:
        path = entry.get("path")
        if not path or not isinstance(path, str):
            continue
        if not path.startswith("scene/"):
            continue
        if entry.get("category") != "map":
            # The manifest's ``inferred_category`` is "Maps / Terrain"
            # for these but the canonical category is "map" — guard
            # against future re-classification.
            pass
        asset = _parse_filename(path)
        if asset is None:
            continue
        # Splice the size from the manifest entry (frozen dataclass —
        # we rebuild with the size attached).
        if entry.get("size"):
            asset = MapAsset(
                path=asset.path,
                area=asset.area,
                area_num=asset.area_num,
                floor=asset.floor,
                suffix=asset.suffix,
                ext=asset.ext,
                size=int(entry["size"]),
            )
        map_id = f"{asset.area}{asset.area_num:02d}"
        if map_id not in by_id:
            cat, label = _classify_area(asset.area)
            by_id[map_id] = MapInfo(
                map_id=map_id,
                area=asset.area,
                area_num=asset.area_num,
                category=cat,
                label=label,
            )
        info = by_id[map_id]
        info.floors.setdefault(asset.floor, []).append(asset)
        info.total_files += 1
        info.total_bytes += asset.size
        if asset.ext in RENDERABLE_EXTS:
            info.renderable_files += 1
        # n.rel files (suffix="n") provide REL-fallback terrain for
        # maps that ship no .nj/.xj. Count them so the picker doesn't
        # gray out city maps as "(no terrain)".
        elif asset.ext in SCRIPT_EXTS and asset.suffix == "n":
            info.renderable_files += 1
    # Stable sort: (category-rank, area, area_num)
    cat_rank = {c: i for i, c in enumerate(
        ("city", "forest", "cave", "mine", "ruins", "battle",
         "corruption", "boss", "other"))}
    out = sorted(
        by_id.values(),
        key=lambda mi: (cat_rank.get(mi.category, 99), mi.area, mi.area_num),
    )
    # Sort each floor's assets so the renderable ones come first (UI
    # convenience — ".nj" before "_00bm.bin").
    ext_rank = {ext: i for i, ext in enumerate(
        RENDERABLE_EXTS + TEXTURE_EXTS + COLLISION_EXTS +
        ANIMATION_EXTS + SCRIPT_EXTS + OTHER_EXTS)}
    for info in out:
        for floor_list in info.floors.values():
            floor_list.sort(key=lambda a: (ext_rank.get(a.ext, 99), a.path))
    return out


def floor_bundle(info: MapInfo, floor: int) -> dict:
    """Return a JSON-friendly bundle for one (map, floor) tuple.

    Used by the GET /api/map/<map_id>?floor=N endpoint. Shape::

        {
          "map_id": "aancient01",
          "floor": 0,
          "category": "forest",
          "label": "Forest (alt)",
          "renderable": [
            {"path": "scene/map_aancient01_00s.nj",
             "kind": "terrain",   # renderable, terrain mesh
             "ext": "nj",
             "suffix": "s",
             "size": 815808},
            {"path": "scene/map_aancient01_00s.xj",
             "kind": "terrain",
             "ext": "xj",
             "suffix": "s",
             "size": 384096}
          ],
          "textures": [
            {"path": "scene/map_aancient01_00s.xvm", "size": ...}
          ],
          "scripts": [
            {"path": "scene/map_aancient01_00c.rel", ...}
          ],
          "animations": [...],
          "other": [...],
          "rrel_path": "scene/map_aancient01_00r.rel",  # v3 — siblings
          "nrel_path": "scene/map_aancient01_00n.rel"   # v3
        }

    The frontend looks at ``renderable`` first — those are the meshes
    it parents into the scene Group. ``textures`` are siblings used
    by the binding pipeline (already wired in /api/model_mesh).
    Everything else is diagnostic.

    REL fallback: when a (map, floor) ships NO ``*_NN s.{nj,xj}`` but
    DOES ship a ``*_NN n.rel`` (Pioneer 2 / city / lab maps), the n.rel
    is exposed as a synthetic ``"rel_terrain"`` renderable so the
    frontend can route through the dedicated /api/map/asset_rel/ path.

    v3 additions (2026-04-25):
      * ``rrel_path``: relative path to the sibling ``*_NN r.rel`` if
        the floor ships one.  Server.py uses this to populate
        ``rrel_render_hints`` (anchor list + bbox-derived fog far) in
        the API response so the Map Editor can refine its environment.
      * ``nrel_path``: the n.rel even when there's a real .nj/.xj
        terrain — exposes the embedded ``rel_texture_names`` table
        regardless of which terrain source the renderer picks.
    """
    assets = info.floors.get(floor, [])

    def _kind_for(asset: MapAsset) -> str:
        # ``s`` (static) suffix is the renderable terrain. The other
        # NJ/XJ siblings (``c`` / ``n`` / ``r``) are PSOBB relocation
        # tables — we still expose them but flag them as "rel" so the
        # frontend can hide / dim them.
        if asset.ext in RENDERABLE_EXTS:
            return "terrain" if asset.suffix == "s" else "rel"
        if asset.ext in TEXTURE_EXTS:
            return "texture"
        if asset.ext in SCRIPT_EXTS:
            return "script"
        if asset.ext in COLLISION_EXTS:
            return "collision"
        if asset.ext in ANIMATION_EXTS:
            return "animation"
        return "other"

    out: dict[str, list[dict]] = {
        "renderable": [],
        "textures": [],
        "scripts": [],
        "animations": [],
        "other": [],
    }
    has_real_terrain = False
    nrel_asset: Optional[MapAsset] = None
    rrel_asset: Optional[MapAsset] = None
    for a in assets:
        rec = {
            "path": a.path,
            "kind": _kind_for(a),
            "ext": a.ext,
            "suffix": a.suffix,
            "size": a.size,
        }
        if a.ext in RENDERABLE_EXTS:
            out["renderable"].append(rec)
            if a.suffix == "s":
                has_real_terrain = True
        elif a.ext in TEXTURE_EXTS:
            out["textures"].append(rec)
        elif a.ext in SCRIPT_EXTS:
            out["scripts"].append(rec)
            # Track candidate n.rel and r.rel siblings.
            if a.suffix == "n" and nrel_asset is None:
                nrel_asset = a
            elif a.suffix == "r" and rrel_asset is None:
                rrel_asset = a
        elif a.ext in ANIMATION_EXTS:
            out["animations"].append(rec)
        else:
            out["other"].append(rec)

    # REL fallback: if the floor has no .nj/.xj but does have an n.rel,
    # surface the n.rel as a "rel_terrain" renderable so the frontend
    # routes through the rel-extraction endpoint.
    if not has_real_terrain and nrel_asset is not None:
        out["renderable"].append({
            "path": nrel_asset.path,
            "kind": "rel_terrain",
            "ext": nrel_asset.ext,
            "suffix": nrel_asset.suffix,
            "size": nrel_asset.size,
        })

    return {
        "map_id": info.map_id,
        "area": info.area,
        "area_num": info.area_num,
        "floor": floor,
        "category": info.category,
        "label": info.label,
        # v3: surface the sibling rel paths so server.py can pull
        # render-hints (r.rel) and texture-name lists (n.rel) into the
        # bundle response.  Either may be None on floors that don't
        # ship that flavour (e.g. Pioneer 2 ships no r.rel; some lab
        # boss arenas ship no n.rel).
        "rrel_path": (rrel_asset.path if rrel_asset is not None else None),
        "nrel_path": (nrel_asset.path if nrel_asset is not None else None),
        **out,
    }


def floors_for(info: MapInfo) -> list[int]:
    """Return floors sorted ascending."""
    return sorted(info.floors.keys())


def list_categories(maps: Iterable[MapInfo]) -> dict[str, list[str]]:
    """Group map_ids by category for the picker grouping headers."""
    out: dict[str, list[str]] = {}
    for m in maps:
        out.setdefault(m.category, []).append(m.map_id)
    for k in out:
        out[k].sort()
    return out


def make_picker_payload(maps: list[MapInfo]) -> dict:
    """Wire shape for GET /api/map/list. Slim — no MapAsset details."""
    return {
        "categories": [
            {"id": "city",       "label": "City / Pioneer 2"},
            {"id": "forest",     "label": "Forest"},
            {"id": "cave",       "label": "Cave"},
            {"id": "mine",       "label": "Mine"},
            {"id": "ruins",      "label": "Ruins"},
            {"id": "battle",     "label": "Battle (Versus)"},
            {"id": "corruption", "label": "Corruption / EP IV"},
            {"id": "boss",       "label": "Boss arena"},
            {"id": "other",      "label": "Other"},
        ],
        "maps": [
            {
                "map_id":           m.map_id,
                "area":             m.area,
                "area_num":         m.area_num,
                "category":         m.category,
                "label":            m.label,
                "floors":           floors_for(m),
                "total_files":      m.total_files,
                "renderable_files": m.renderable_files,
                "total_bytes":      m.total_bytes,
            }
            for m in maps
        ],
    }


# ---------------------------------------------------------------------------
# Sidecar JSON shape — spawn / waypoint edits
# ---------------------------------------------------------------------------

VALID_SPAWN_TYPES = ("mob", "npc", "chest", "switch", "teleport")
VALID_WAYPOINT_STYLES = ("walk", "run", "teleport")
SPAWN_FILE_VERSION = 1


def validate_edits_payload(payload: dict) -> tuple[bool, str]:
    """Validate a POST /api/map/edits body.

    Returns (ok, error_message). On success ``error_message`` is "".
    """
    if not isinstance(payload, dict):
        return False, "payload must be a JSON object"
    map_id = payload.get("map_id")
    if not isinstance(map_id, str) or not re.match(r"^[a-z]+\d+$", map_id):
        return False, "map_id must match ^[a-z]+\\d+$"
    spawns = payload.get("spawns") or []
    if not isinstance(spawns, list):
        return False, "spawns must be a list"
    waypoints = payload.get("waypoints") or []
    if not isinstance(waypoints, list):
        return False, "waypoints must be a list"
    seen_ids: set[int] = set()
    for i, sp in enumerate(spawns):
        if not isinstance(sp, dict):
            return False, f"spawns[{i}] not a dict"
        sid = sp.get("id")
        if not isinstance(sid, int):
            return False, f"spawns[{i}].id must be int"
        if sid in seen_ids:
            return False, f"duplicate spawn id {sid}"
        seen_ids.add(sid)
        stype = sp.get("type")
        if stype not in VALID_SPAWN_TYPES:
            return False, f"spawns[{i}].type {stype!r} not in {VALID_SPAWN_TYPES}"
        wp = sp.get("world_pos")
        if not (isinstance(wp, (list, tuple)) and len(wp) == 3
                and all(isinstance(v, (int, float)) for v in wp)):
            return False, f"spawns[{i}].world_pos must be [x,y,z] floats"
        rot = sp.get("rotation", 0.0)
        if not isinstance(rot, (int, float)):
            return False, f"spawns[{i}].rotation must be float"
        if "type_data" in sp and not isinstance(sp["type_data"], dict):
            return False, f"spawns[{i}].type_data must be a dict"
    for i, w in enumerate(waypoints):
        if not isinstance(w, dict):
            return False, f"waypoints[{i}] not a dict"
        a = w.get("from_id")
        b = w.get("to_id")
        if not isinstance(a, int) or not isinstance(b, int):
            return False, f"waypoints[{i}].(from_id|to_id) must be int"
        if a == b:
            return False, f"waypoints[{i}] self-loop ({a}->{b})"
        if a not in seen_ids or b not in seen_ids:
            return False, f"waypoints[{i}] references missing spawn"
        style = w.get("style", "walk")
        if style not in VALID_WAYPOINT_STYLES:
            return False, f"waypoints[{i}].style not in {VALID_WAYPOINT_STYLES}"
        speed = w.get("speed", 1.0)
        if not isinstance(speed, (int, float)):
            return False, f"waypoints[{i}].speed must be float"
    return True, ""


def normalize_edits_payload(payload: dict) -> dict:
    """Strip extra keys + coerce numeric types. Caller must have validated."""
    out_spawns = []
    for sp in payload.get("spawns") or []:
        td = sp.get("type_data") or {}
        out_spawns.append({
            "id":         int(sp["id"]),
            "type":       sp["type"],
            "world_pos":  [float(v) for v in sp["world_pos"]],
            "rotation":   float(sp.get("rotation", 0.0)),
            "type_data":  td if isinstance(td, dict) else {},
        })
    out_waypoints = []
    for w in payload.get("waypoints") or []:
        out_waypoints.append({
            "from_id":  int(w["from_id"]),
            "to_id":    int(w["to_id"]),
            "speed":    float(w.get("speed", 1.0)),
            "style":    w.get("style", "walk"),
        })
    return {
        "version":   SPAWN_FILE_VERSION,
        "map_id":    payload["map_id"],
        "spawns":    out_spawns,
        "waypoints": out_waypoints,
    }


__all__ = [
    "MapAsset",
    "MapInfo",
    "AREA_CATEGORY",
    "RENDERABLE_EXTS",
    "TEXTURE_EXTS",
    "VALID_SPAWN_TYPES",
    "VALID_WAYPOINT_STYLES",
    "SPAWN_FILE_VERSION",
    "catalogue",
    "floor_bundle",
    "floors_for",
    "list_categories",
    "make_picker_payload",
    "validate_edits_payload",
    "normalize_edits_payload",
]
