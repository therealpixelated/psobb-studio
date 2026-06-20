"""PSOBB engine fog/light table loaders (2026-04-25, Editor v4 RE).

The PSOBB renderer reads two flat lookup tables for per-area fog and
sunlight parameters:

* ``fogentry.dat`` — 256 × 64-byte ``FogEntry`` records.
* ``lightentry.bin`` — 96 × 68-byte ``LightEntry`` records (48 normal
  + 48 ultimate Episode 1 entries).

Both files live in ``<install>/data/`` and are read at startup by
PsoBB.exe (see Editor v3 RE notes in ``_reports/psobb_engine_table_RE.md``).
The runtime indices into these tables are the canonical PSOBB
``MapType`` enum values (0..46) — the same values used by
``Blue-Burst-Patch-Project/Blue Burst Patch Project/map.h`` and exposed
through ``GetCurrentMap()`` at runtime.

Layouts mirror the structs in ``newmap/fog.cpp`` and ``newmap/sunlight.cpp``
of the BBPP source tree (both ``#pragma pack(push, 1)``).

Field naming preserves the BBPP convention so cross-references are
trivial.

Public surface:

  :func:`load_fog_table`
      Read ``fogentry.dat`` and return ``list[FogEntry]``.
  :func:`load_light_table`
      Read ``lightentry.bin`` and return ``list[LightEntry]``.
  :func:`map_id_to_engine_index`
      Convert an editor ``map_id`` (e.g. ``"aancient01"``) plus optional
      sub-area into the PSOBB engine table index.
  :func:`build_engine_env_table`
      Top-level convenience: return a JSON-friendly dict keyed by
      editor ``map_id`` with the resolved fog + light entries.

The decoded tables are READ-ONLY ground truth from the install — we
never patch them. Editor consumers (``static/psobb_engine_data.js``)
ship a generated snapshot rather than parsing on every boot.
"""
from __future__ import annotations
import os

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sizes mandated by the file format. The PSOBB runtime reads exactly these
# many bytes; the runtime allocation in ``fcn.005bdb50`` is 0x4000 bytes
# (= 256 * 64) — confirmed against the binary at 0x005bdb8d.
# ---------------------------------------------------------------------------
FOG_ENTRY_SIZE = 64
FOG_TABLE_BYTES = 0x4000
FOG_TABLE_COUNT = FOG_TABLE_BYTES // FOG_ENTRY_SIZE  # = 256

LIGHT_ENTRY_SIZE = 68      # 17 * f32 (BBPP sunlight.cpp)
LIGHT_TABLE_COUNT = 96     # 48 normal + 48 ultimate-ep1


# ---------------------------------------------------------------------------
# struct formats. Both are #pragma pack(push, 1) — no alignment padding.
#
# FogEntry (64 bytes), per BBPP ``newmap/fog.cpp``:
#
#   u32 type
#   u8  color_b, color_g, color_r, color_a    (BGRA, NOT RGBA — matches D3D)
#   f32 end, start, density
#   u32 unk1
#   f32 unk2, unk3
#   f32 end_pulse_distance
#   u32 unk4
#   f32 start_pulse_distance
#   u32 unk5
#   f32 transition
#   u32 unk6
#   u8  unk7, unk8
#   u8  lerped_field6
#   u8  unk9
#   u8  lerped_field7
#   u8  unk10
#   u8  lerped_field8
#   u8  unk11
# ---------------------------------------------------------------------------
_FOG_FMT = "<I 4B fff I ff f I f I f I 8B"
assert struct.calcsize(_FOG_FMT) == FOG_ENTRY_SIZE, struct.calcsize(_FOG_FMT)


# LightEntry (68 bytes), 17 contiguous floats per BBPP ``newmap/sunlight.cpp``:
#
#   x1, y1, z1            # primary direction (sun vector)
#   x2, y2, z2            # secondary direction (some maps use a fill light)
#   intensity_specular
#   intensity_diffuse
#   intensity_ambient
#   diffuse_argb (a, r, g, b)
#   ambient_argb (a, r, g, b)
_LIGHT_FMT = "<17f"
assert struct.calcsize(_LIGHT_FMT) == LIGHT_ENTRY_SIZE


# ---------------------------------------------------------------------------
# Dataclasses — one Python class per on-disk struct.
# ---------------------------------------------------------------------------
@dataclass
class FogEntry:
    """Decoded ``FogEntry`` record from ``fogentry.dat``.

    The colour is stored on disk as little-endian BGRA (Direct3D 8 default).
    We surface both forms: ``color_bgra`` keeps the wire byte order while
    ``color_rgba_int`` is the integer the JS THREE.Color() helper expects.
    """
    type: int                       # 0 = no fog, 1 = linear, 2 = exp/depth
    color_b: int                    # 0..255
    color_g: int
    color_r: int
    color_a: int
    end: float                      # fog far plane
    start: float                    # fog near plane
    density: float                  # only used when type != 1
    unk1: int
    unk2: float
    unk3: float
    end_pulse_distance: float
    unk4: int
    start_pulse_distance: float
    unk5: int
    transition: float
    unk6: int
    unk7: int
    unk8: int
    lerped_field6: int
    unk9: int
    lerped_field7: int
    unk10: int
    lerped_field8: int
    unk11: int

    @property
    def color_rgba_int(self) -> int:
        """0xRRGGBBAA — the JSON / JS-friendly representation."""
        return (self.color_r << 24) | (self.color_g << 16) | \
               (self.color_b << 8) | self.color_a

    @property
    def color_rgb_hex(self) -> int:
        """0xRRGGBB — matches the JS ``new THREE.Color(0xRRGGBB)`` API."""
        return (self.color_r << 16) | (self.color_g << 8) | self.color_b

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "color_rgb": self.color_rgb_hex,
            "color_a": self.color_a,
            "end": self.end,
            "start": self.start,
            "density": self.density,
            "transition": self.transition,
            "end_pulse_distance": self.end_pulse_distance,
            "start_pulse_distance": self.start_pulse_distance,
        }


@dataclass
class LightEntry:
    """Decoded ``LightEntry`` from ``lightentry.bin``.

    Two direction vectors are stored — observed values suggest ``dir1`` is
    the primary sun and ``dir2`` is a secondary highlight axis used by
    some sky shaders. The ARGB colours are floats in [0, 1] range.
    """
    dir1: tuple[float, float, float]
    dir2: tuple[float, float, float]
    intensity_specular: float
    intensity_diffuse: float
    intensity_ambient: float
    diffuse_argb: tuple[float, float, float, float]
    ambient_argb: tuple[float, float, float, float]

    def to_dict(self) -> dict:
        return {
            "dir1": list(self.dir1),
            "dir2": list(self.dir2),
            "intensity_specular": self.intensity_specular,
            "intensity_diffuse": self.intensity_diffuse,
            "intensity_ambient": self.intensity_ambient,
            "diffuse_argb": list(self.diffuse_argb),
            "ambient_argb": list(self.ambient_argb),
        }


# ---------------------------------------------------------------------------
# MapType enum (PSOBB engine indices).  Mirrors
# Blue-Burst-Patch-Project/Blue Burst Patch Project/map.h::MapType.
# These ARE the FogEntry / LightEntry table indices at runtime.
# ---------------------------------------------------------------------------
MAP_TYPE: dict[str, int] = {
    "Pioneer2_Ep1":     0,
    "Forest1":          1,
    "Forest2":          2,
    "Cave1":            3,
    "Cave2":            4,
    "Cave3":            5,
    "Mines1":           6,
    "Mines2":           7,
    "Ruins1":           8,
    "Ruins2":           9,
    "Ruins3":          10,
    "Boss_Dragon":     11,
    "Boss_Derolle":    12,
    "Boss_Volopt":     13,
    "Boss_Darkfalz":   14,
    "Lobby":           15,
    "Battle_Spaceship": 16,
    "Battle_Ruins":    17,
    "Pioneer2_Ep2":    18,
    "Temple_A":        19,
    "Temple_B":        20,
    "Spaceship_A":     21,
    "Spaceship_B":     22,
    "CCA":             23,
    "Jungle_East":     24,
    "Jungle_North":    25,
    "Mountain":        26,
    "Seaside":         27,
    "Seabed_Upper":    28,
    "Seabed_Lower":    29,
    "Boss_Galgryphon": 30,
    "Boss_Olgaflow":   31,
    "Boss_Barbaray":   32,
    "Boss_Goldragon":  33,
    "Seaside_Night":   34,
    "Tower":           35,
    "Wilds1":          36,
    "Wilds2":          37,
    "Wilds3":          38,
    "Wilds4":          39,
    "Crater":          40,
    "Desert1":         41,
    "Desert2":         42,
    "Desert3":         43,
    "Boss_Saintmilion": 44,
    "Pioneer2_Ep4":    45,
    "Test_Area":       46,
}


# Editor map_id (area + area_num) → MapType name. The editor uses the
# scene_loader's AREA_CATEGORY plus a numeric suffix; we resolve both
# the canonical (e.g. "ancient01") and the alt-prefix (``"aancient01"``)
# spellings.
#
# Where a single editor map_id covers multiple sub-areas inside one
# MapType (e.g. ``aancient01`` corresponds to Forest1 across all 5
# floors 0..4), we map all of them to the same engine index — the
# fog/light table is per-MapType, not per-floor.
MAP_ID_TO_TYPE: dict[str, str] = {
    # ---- Episode 1 city / lobby ----
    # Pioneer 2 ships in three flavours: city (normal), acity (alt /
    # ultimate), and labo (lab variant).  All three share the same fog
    # entry — the engine reads index 0 for any Pioneer-2-Ep1 visit.
    "city00":           "Pioneer2_Ep1",
    "acity00":          "Pioneer2_Ep1",
    "labo00":           "Pioneer2_Ep1",
    # ---- Episode 1 dungeons ----
    # Each "areaNN" is one MapType.  Ancient/aancient share the same
    # MapType (alt-prefix is just texture re-skin for the 'a' set).
    # Ancient03..05 ship terrain but no MapType slot in the BBPP enum
    # — they are quest-only maps reusing Forest1 fog at runtime, so we
    # alias them onto Forest1.
    "ancient01":        "Forest1",
    "aancient01":       "Forest1",
    "ancient02":        "Forest2",
    "aancient02":       "Forest2",
    "ancient03":        "Forest1",
    "aancient03":       "Forest1",
    "ancient04":        "Forest1",
    "aancient04":       "Forest1",
    "ancient05":        "Forest1",
    "aancient05":       "Forest1",
    "cave01":           "Cave1",
    "acave01":          "Cave1",
    "cave02":           "Cave2",
    "acave02":          "Cave2",
    "cave03":           "Cave3",
    "acave03":          "Cave3",
    "machine01":        "Mines1",
    "amachine01":       "Mines1",
    "machine02":        "Mines2",
    "amachine02":       "Mines2",
    "ruins01":          "Ruins1",
    "ruins02":          "Ruins2",
    "ruins03":          "Ruins3",
    # ---- Episode 1 bosses ----
    "boss01":           "Boss_Dragon",
    "aboss01":          "Boss_Dragon",
    "boss02":           "Boss_Derolle",
    "aboss02":          "Boss_Derolle",
    "boss03":           "Boss_Volopt",
    "aboss03":          "Boss_Volopt",
    "boss04":           "Boss_Darkfalz",
    "aboss04":          "Boss_Darkfalz",
    # ---- Episode 2 city ----
    "city01":           "Pioneer2_Ep2",
    "city02":           "Pioneer2_Ep4",
    # ---- Episode 2 dungeons ----
    "jungle01":         "Temple_A",
    "jungle02":         "Temple_B",
    "jungle03":         "Spaceship_A",
    "jungle04":         "Spaceship_B",
    "jungle05":         "CCA",
    "jungle06":         "Jungle_East",
    "jungle07":         "Jungle_North",
    "jungle08":         "Mountain",
    "jungle09":         "Seaside",
    "space01":          "Spaceship_A",
    "space02":          "Spaceship_B",
    # ---- Episode 2 seabed ----
    "seabed01":         "Seabed_Upper",
    "seabed02":         "Seabed_Lower",
    # ---- Episode 2 bosses ----
    "boss05":           "Boss_Galgryphon",
    "boss06":           "Boss_Olgaflow",
    "boss07":           "Boss_Barbaray",
    "boss08":           "Boss_Goldragon",
    # ---- Episode 4 ----
    "wilds01":          "Wilds1",
    "wilds02":          "Wilds2",
    "wilds03":          "Wilds3",
    "wilds04":          "Wilds4",
    "crater01":         "Crater",
    "desert01":         "Desert1",
    "desert02":         "Desert2",
    "desert03":         "Desert3",
    "boss09":           "Boss_Saintmilion",
    # ---- Misc ----
    "vs01":             "Battle_Spaceship",
    "vs02":             "Battle_Ruins",
    "test01":           "Test_Area",
    "test02":           "Test_Area",
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
DEFAULT_INSTALL = Path(os.path.expanduser("~/PSOBB.IO"))


def _resolve_data_dir(install_root: Optional[Path]) -> Path:
    """Return the ``<install>/data`` directory.  Falls back to default."""
    root = install_root if install_root is not None else DEFAULT_INSTALL
    return Path(root) / "data"


# ---------------------------------------------------------------------------
# Fog table loader
# ---------------------------------------------------------------------------
def _decode_fog_entry(buf: bytes, off: int) -> FogEntry:
    rec = struct.unpack_from(_FOG_FMT, buf, off)
    return FogEntry(*rec)


def load_fog_table(install_root: Optional[Path] = None) -> list[FogEntry]:
    """Read ``data/fogentry.dat`` and decode all 256 entries.

    Raises:
        FileNotFoundError: if ``fogentry.dat`` is missing.
        ValueError: if the file size is not exactly 16384 bytes.
    """
    path = _resolve_data_dir(install_root) / "fogentry.dat"
    data = path.read_bytes()
    if len(data) != FOG_TABLE_BYTES:
        raise ValueError(
            f"fogentry.dat size {len(data)} != expected {FOG_TABLE_BYTES}")
    out: list[FogEntry] = []
    for i in range(FOG_TABLE_COUNT):
        out.append(_decode_fog_entry(data, i * FOG_ENTRY_SIZE))
    return out


# ---------------------------------------------------------------------------
# Light table loader
# ---------------------------------------------------------------------------
def _decode_light_entry(buf: bytes, off: int) -> LightEntry:
    f = struct.unpack_from(_LIGHT_FMT, buf, off)
    return LightEntry(
        dir1=(f[0], f[1], f[2]),
        dir2=(f[3], f[4], f[5]),
        intensity_specular=f[6],
        intensity_diffuse=f[7],
        intensity_ambient=f[8],
        diffuse_argb=(f[9], f[10], f[11], f[12]),
        ambient_argb=(f[13], f[14], f[15], f[16]),
    )


def load_light_table(install_root: Optional[Path] = None) -> list[LightEntry]:
    """Read ``data/lightentry.bin`` and decode all 96 entries.

    Raises:
        FileNotFoundError: if ``lightentry.bin`` is missing.
        ValueError: if the file size is not exactly 96 * 68 bytes.
    """
    path = _resolve_data_dir(install_root) / "lightentry.bin"
    data = path.read_bytes()
    expected = LIGHT_TABLE_COUNT * LIGHT_ENTRY_SIZE
    if len(data) != expected:
        raise ValueError(
            f"lightentry.bin size {len(data)} != expected {expected}")
    out: list[LightEntry] = []
    for i in range(LIGHT_TABLE_COUNT):
        out.append(_decode_light_entry(data, i * LIGHT_ENTRY_SIZE))
    return out


# ---------------------------------------------------------------------------
# map_id → engine index resolution
# ---------------------------------------------------------------------------
def map_id_to_engine_index(map_id: str, *, ultimate: bool = False) -> Optional[int]:
    """Return the FogEntry/LightEntry index for an editor ``map_id``.

    Args:
      map_id: The PSOBB-style map_id (``"aancient01"`` etc.).
      ultimate: If True and the map is in Episode 1, returns the
        ultimate-ep1 LightEntry index (engine offsets ep1 by +48 in
        ultimate mode — see BBPP ``sunlight.cpp::ReplaceMapSunlight``).
        Note: there is NO ultimate offset for FogEntry — fog is
        difficulty-agnostic.

    Returns:
      The integer index, or ``None`` when ``map_id`` is unknown.
    """
    if not map_id:
        return None
    name = MAP_ID_TO_TYPE.get(map_id)
    if name is None:
        return None
    base = MAP_TYPE[name]
    if ultimate and base <= MAP_TYPE["Boss_Darkfalz"]:
        # Ep1 ultimate uses entries 48..95 in lightentry.bin.
        return base + 48
    return base


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------
@dataclass
class EngineEnv:
    """Combined fog + light entry for one map_id, plus diagnostics."""
    map_id: str
    map_type: str
    engine_index: int
    fog: FogEntry
    light: LightEntry
    light_ultimate: Optional[LightEntry] = None

    def to_dict(self) -> dict:
        out = {
            "map_id": self.map_id,
            "map_type": self.map_type,
            "engine_index": self.engine_index,
            "fog": self.fog.to_dict(),
            "light": self.light.to_dict(),
        }
        if self.light_ultimate is not None:
            out["light_ultimate"] = self.light_ultimate.to_dict()
        return out


def build_engine_env_table(
    install_root: Optional[Path] = None,
    map_ids: Optional[Iterable[str]] = None,
) -> dict[str, EngineEnv]:
    """Return ``{map_id: EngineEnv}`` for every catalogued ``map_id``.

    When ``map_ids`` is None, every entry in ``MAP_ID_TO_TYPE`` is
    resolved.  Unknown ids are silently skipped.  The returned dict is
    suitable for direct serialisation to JSON.

    The fog index is the canonical MapType slot; the light index uses
    the same slot for normal mode plus a sibling ``light_ultimate``
    when the map is in Episode 1 and an ultimate-mode entry exists.
    """
    fog_table = load_fog_table(install_root)
    light_table = load_light_table(install_root)

    out: dict[str, EngineEnv] = {}
    keys = list(MAP_ID_TO_TYPE.keys()) if map_ids is None else list(map_ids)
    for mid in keys:
        idx = map_id_to_engine_index(mid)
        if idx is None or idx >= FOG_TABLE_COUNT or idx >= LIGHT_TABLE_COUNT:
            continue
        type_name = MAP_ID_TO_TYPE[mid]
        ult_idx = map_id_to_engine_index(mid, ultimate=True)
        ult_entry = (
            light_table[ult_idx]
            if ult_idx is not None
            and ult_idx != idx
            and ult_idx < LIGHT_TABLE_COUNT
            else None
        )
        out[mid] = EngineEnv(
            map_id=mid,
            map_type=type_name,
            engine_index=idx,
            fog=fog_table[idx],
            light=light_table[idx],
            light_ultimate=ult_entry,
        )
    return out


def build_engine_env_dict(install_root: Optional[Path] = None) -> dict:
    """JSON-friendly dict of every map_id → env.  Used by the JS exporter."""
    return {mid: env.to_dict()
            for mid, env in build_engine_env_table(install_root).items()}


__all__ = [
    "FogEntry",
    "LightEntry",
    "EngineEnv",
    "FOG_ENTRY_SIZE",
    "FOG_TABLE_COUNT",
    "LIGHT_ENTRY_SIZE",
    "LIGHT_TABLE_COUNT",
    "MAP_TYPE",
    "MAP_ID_TO_TYPE",
    "load_fog_table",
    "load_light_table",
    "map_id_to_engine_index",
    "build_engine_env_table",
    "build_engine_env_dict",
]



# ===========================================================================
# Section 2 - DAT entity definitions (objects + enemies)
# ===========================================================================
#
# Ported from newserv ``src/Map.cc`` (MIT) lines 604-3149, the canonical
# id -> C++ class table for PSO map objects and enemies. Newserv decodes
# every map .dat through this lookup; the same table tells us, for each
# 16-bit ``base_type``, the engine class that will be constructed.
#
# Source-of-truth: fuzziqersoftware/newserv (MIT)
#   https://github.com/fuzziqersoftware/newserv
#   src/Map.cc :: dat_object_definitions   (lines 604-2493)
#   src/Map.cc :: dat_enemy_definitions    (lines 2495-3149)
#
# License: newserv ships under MIT. We port the DATA verbatim and re-
# implement the Python around it from scratch - no newserv code is
# copied beyond the literal table contents (id, version mask, area
# mask, class name, prose comment).

# Re-import the helper tables maintained in a sibling sub-module so the
# verbose row arrays below stay close to where they're consumed.
from formats._psobb_engine_tables_extra import (  # noqa: E402
    PER_FLOOR_ENEMY_INDEX,
    SETDATA_NAMES,
)

VERSION_FLAG_BITS: dict[str, int] = {
    "F_V0_V1":  0x001C,
    "F_V0_V2":  0x01FC,
    "F_V0_V4":  0x33FC,
    "F_V1_V4":  0x33F0,
    "F_V2":     0x01E0,
    "F_V2_V4":  0x33E0,
    "F_V3_V4":  0x3200,
    "F_V4":     0x2000,
    "F_GC":     0x0F00,
    "F_EP3":    0x0C00,
}
F_V4_MASK = 0x2000     # PSOBB Blue-Burst version bit


@dataclass
class EntityDef:
    """One id <-> class binding from newserv's table.

    Multiple ``EntityDef``s may share the same ``type_id`` (e.g. id
    0x00C0 is ``TBoss1Dragon`` on Episode 1 but ``TBoss5Gryphon`` on
    Episode 2). The ``version_flag`` / ``area_mask`` pair identifies
    which version+area combination resolves to this row at runtime.
    """
    type_id: int
    class_name: str
    display_name: str | None = None
    bml_inner_count_hint: int | None = None
    notes: str | None = None
    version_flag: str = "F_V0_V4"
    area_mask: int = 0


def _friendly_name_from(class_name: str, comment: str | None) -> str | None:
    """Promote inline-comment text to a display name when sensible."""
    if not comment:
        return None
    c = comment.strip()
    if not c or c.upper() == "TODO":
        return None
    for sep in (". ", " (", ";"):
        i = c.find(sep)
        if i > 0:
            c = c[:i]
            break
    return c.strip(" ,.") or None


def _rows_to_entity_dict(
    rows: list[tuple[int, str, int, str, str | None]],
) -> dict[int, EntityDef]:
    """Pick a representative EntityDef per ``type_id``.

    Prefers rows with the BB version bit (F_V4) so the editor sees BB
    classes by default; falls back to the first encountered row for
    ids that have no BB representative (e.g. EP3-only entries).
    """
    out: dict[int, EntityDef] = {}
    seen_first: dict[int, EntityDef] = {}
    for tid, vf, am, name, comment in rows:
        ed = EntityDef(
            type_id=tid,
            class_name=name,
            display_name=_friendly_name_from(name, comment),
            notes=comment or None,
            version_flag=vf,
            area_mask=am,
        )
        if tid not in seen_first:
            seen_first[tid] = ed
        ver_bits = VERSION_FLAG_BITS.get(vf, 0)
        if (ver_bits & F_V4_MASK) and tid not in out:
            out[tid] = ed
    for tid, ed in seen_first.items():
        out.setdefault(tid, ed)
    return out


def all_defs_for_type(
    rows: list[tuple[int, str, int, str, str | None]],
    type_id: int,
) -> list[EntityDef]:
    """Return every newserv row matching ``type_id`` (multi-version aware)."""
    out: list[EntityDef] = []
    for tid, vf, am, name, comment in rows:
        if tid == type_id:
            out.append(EntityDef(
                type_id=tid,
                class_name=name,
                display_name=_friendly_name_from(name, comment),
                notes=comment or None,
                version_flag=vf,
                area_mask=am,
            ))
    return out


# ---------------------------------------------------------------------------
# Raw row arrays - verbatim from newserv. Format:
#   (type_id, version_flag, area_mask, class_name, comment_or_None)
# ---------------------------------------------------------------------------
OBJECT_TABLE_ROWS: list[tuple[int, str, int, str, str | None]] = [
    (0x0000, 'F_V0_V4', 0x00007FFFFFFFFFFF, 'TObjPlayerSet', None),
    (0x0000, 'F_EP3', 0x0000000000008001, 'TObjPlayerSet', None),
    (0x0001, 'F_V0_V4', 0x00006FFFFFFFFFFF, 'TObjParticle', None),
    (0x0001, 'F_EP3', 0x0000000000008003, 'TObjParticle', None),
    (0x0002, 'F_V0_V4', 0x00007FF3C07C78FF, 'TObjAreaWarpForest', None),
    (0x0003, 'F_V0_V4', 0x00007FFC3FFF78FF, 'TObjMapWarpForest', None),
    (0x0004, 'F_V0_V4', 0x00006FFC3FFF87FF, 'TObjLight', None),
    (0x0004, 'F_EP3', 0x0000000000008003, 'TObjLight', None),
    (0x0005, 'F_V0_V2', 0x000000000000073F, 'TItem', None),
    (0x0006, 'F_V0_V2', 0x0000000000037FFF, 'TObjEnvSound', None),
    (0x0006, 'F_V3_V4', 0x00006FF0BFFF27FF, 'TObjEnvSound', None),
    (0x0006, 'F_EP3', 0x0000000000000001, 'TObjEnvSound', None),
    (0x0007, 'F_V0_V4', 0x00006FFFFFFF7FFF, 'TObjFogCollision', None),
    (0x0007, 'F_EP3', 0x0000000000000001, 'TObjFogCollision', None),
    (0x0008, 'F_V0_V4', 0x00007FFFFFFF7FFF, 'TObjEvtCollision', None),
    (0x0008, 'F_EP3', 0x0000000000000001, 'TObjEvtCollision', None),
    (0x0009, 'F_V0_V4', 0x000060000004073F, 'TObjCollision', None),
    (0x0009, 'F_EP3', 0x0000000000000001, 'TObjCollision', None),
    (0x000A, 'F_V0_V4', 0x00005FFC3FFB07FE, 'TOMineIcon01', None),
    (0x000B, 'F_V0_V4', 0x00005FFC3FFB07FE, 'TOMineIcon02', None),
    (0x000C, 'F_V0_V4', 0x00005FFC3FFB07FE, 'TOMineIcon03', None),
    (0x000D, 'F_V0_V4', 0x00005FFC3FFB07FE, 'TOMineIcon04', None),
    (0x000E, 'F_V0_V4', 0x00005FFFFFF83FFE, 'TObjRoomId', None),
    (0x000F, 'F_V0_V4', 0x00004000000000F6, 'TOSensorGeneral01', None),
    (0x0011, 'F_V0_V4', 0x000040000000411E, 'TEF_LensFlare', None),
    (0x0012, 'F_V0_V4', 0x00006FFFFFFC7FFF, 'TObjQuestCol', None),
    (0x0012, 'F_EP3', 0x0000000000000001, 'TObjQuestCol', None),
    (0x0013, 'F_V0_V4', 0x00004FFC3FF807FE, 'TOHealGeneral', None),
    (0x0014, 'F_V0_V4', 0x0000600C3F87073F, 'TObjMapCsn', None),
    (0x0014, 'F_EP3', 0x0000000000000001, 'TObjMapCsn', None),
    (0x0015, 'F_V0_V4', 0x00006FFFFFFC7FFF, 'TObjQuestColA', None),
    (0x0015, 'F_EP3', 0x0000000000000001, 'TObjQuestColA', None),
    (0x0016, 'F_V0_V4', 0x00006FFFFFFCFFFF, 'TObjItemLight', None),
    (0x0016, 'F_EP3', 0x0000000000008001, 'TObjItemLight', None),
    (0x0017, 'F_V0_V4', 0x00004FFFFFF8FFFE, 'TObjRaderCol', None),
    (0x0017, 'F_EP3', 0x0000000000008000, 'TObjRaderCol', None),
    (0x0018, 'F_V0_V4', 0x00004FFFFFF87FFE, 'TObjFogCollisionSwitch', None),
    (0x0019, 'F_V0_V4', 0x00006FFC3FFC04A5, 'TObjWarpBoss', None),
    (0x001A, 'F_V1_V4', 0x0000600000040001, 'TObjSinBoard', None),
    (0x001A, 'F_EP3', 0x0000000000000001, 'TObjSinBoard', None),
    (0x001B, 'F_V1_V4', 0x00005000000078FE, 'TObjAreaWarpQuest', None),
    (0x001C, 'F_V1_V4', 0x0000500080004000, 'TObjAreaWarpEnding', None),
    (0x001D, 'F_V2_V4', 0x0000400000000002, 'TEffStarLight2D_Base', None),
    (0x001E, 'F_V2_V4', 0x000041F1001A0006, '__LENS_FLARE__', None),
    (0x001F, 'F_V2_V4', 0x00004FFC3FFB07FE, 'TObjRaderHideCol', None),
    (0x0020, 'F_V2_V4', 0x00006FFC3FFF07FF, 'TOSwitchItem', None),
    (0x0021, 'F_V2_V4', 0x00006FFC3FFF07FF, 'TOSymbolchatColli', None),
    (0x0022, 'F_V2_V4', 0x00004FFC3FFB07FE, 'TOKeyCol', None),
    (0x0023, 'F_V2_V4', 0x00004FFC3FFB07FE, 'TOAttackableCol', None),
    (0x0024, 'F_V2_V4', 0x0000600FFF9F07FF, 'TOSwitchAttack', None),
    (0x0025, 'F_V2_V4', 0x00006FFC3FFF07FF, 'TOSwitchTimer', None),
    (0x0026, 'F_V2_V4', 0x00006FFC3FFF07FF, 'TOChatSensor', None),
    (0x0027, 'F_V3_V4', 0x00004FFFFFFC0000, 'TObjRaderIcon', None),
    (0x0028, 'F_V3_V4', 0x00006FFCBFFF27F7, 'TObjEnvSoundEx', None),
    (0x0028, 'F_EP3', 0x0000000000000001, 'TObjEnvSoundEx', None),
    (0x0029, 'F_V3_V4', 0x00006FFCBFFF27F7, 'TObjEnvSoundGlobal', None),
    (0x0029, 'F_EP3', 0x0000000000000001, 'TObjEnvSoundGlobal', None),
    (0x0040, 'F_V0_V4', 0x0000600000040001, 'TShopGenerator', None),
    (0x0040, 'F_EP3', 0x0000000000000001, 'TShopGenerator', None),
    (0x0041, 'F_V0_V4', 0x0000600000040001, 'TObjLuker', None),
    (0x0041, 'F_EP3', 0x0000000000000001, 'TObjLuker', None),
    (0x0042, 'F_V0_V4', 0x0000600000040001, 'TObjBgmCol', None),
    (0x0042, 'F_EP3', 0x0000000000000001, 'TObjBgmCol', None),
    (0x0043, 'F_V0_V4', 0x0000600000040001, 'TObjCityMainWarp', None),
    (0x0044, 'F_V0_V4', 0x0000600000040001, 'TObjCityAreaWarp', None),
    (0x0044, 'F_EP3', 0x0000000000000001, 'TObjCityAreaWarp', None),
    (0x0045, 'F_V0_V4', 0x0000600000040001, 'TObjCityMapWarp', None),
    (0x0046, 'F_V0_V4', 0x0000600000000001, 'TObjCityDoor_Shop', 'Door to shop area'),
    (0x0047, 'F_V0_V4', 0x0000600000000001, 'TObjCityDoor_Guild', "Door to Hunter's Guild"),
    (0x0048, 'F_V0_V4', 0x0000600000000001, 'TObjCityDoor_Warp', 'Door to Ragol warp'),
    (0x0049, 'F_V0_V4', 0x0000600000000001, 'TObjCityDoor_Med', 'Door to Medical Center'),
    (0x004A, 'F_V0_V4', 0x0000600000000001, '__ELEVATOR__', None),
    (0x004B, 'F_V0_V4', 0x0000600000040001, 'TObjCity_Season_EasterEgg', None),
    (0x004C, 'F_V0_V4', 0x0000600000040001, 'TObjCity_Season_ValentineHeart', None),
    (0x004D, 'F_V0_V4', 0x0000600000040001, 'TObjCity_Season_XmasTree', None),
    (0x004E, 'F_V0_V4', 0x0000600000040001, 'TObjCity_Season_XmasWreath', None),
    (0x004F, 'F_V0_V4', 0x0000600000040001, 'TObjCity_Season_HalloweenPumpkin', None),
    (0x0050, 'F_V0_V4', 0x0000600000040001, 'TObjCity_Season_21_21', None),
    (0x0051, 'F_V0_V4', 0x0000600000040001, 'TObjCity_Season_SonicAdv2', None),
    (0x0052, 'F_V0_V4', 0x0000600000040001, 'TObjCity_Season_Board', None),
    (0x0053, 'F_V0_V4', 0x0000600400040001, 'TObjCity_Season_FireWorkCtrl', None),
    (0x0054, 'F_V0_V4', 0x0000600000000001, 'TObjCityDoor_Lobby', None),
    (0x0055, 'F_V2_V4', 0x0000600000040001, 'TObjCityMainWarpChallenge', None),
    (0x0056, 'F_V3_V4', 0x0000400000040000, 'TODoorLabo', None),
    (0x0057, 'F_V3_V4', 0x0000600000040001, 'TObjTradeCollision', None),
    (0x0057, 'F_EP3', 0x0000000000000001, 'TObjTradeCollision', None),
    (0x0058, 'F_EP3', 0x0000000000000001, 'TObjDeckCollision', None),
    (0x0080, 'F_V0_V4', 0x0000400000000006, 'TObjDoor', None),
    (0x0081, 'F_V0_V4', 0x00004FF00078003E, 'TObjDoorKey', None),
    (0x0082, 'F_V0_V4', 0x00004FF0000300FE, 'TObjLazerFenceNorm', None),
    (0x0083, 'F_V0_V4', 0x00004FF03FFB00FE, 'TObjLazerFence4', None),
    (0x0084, 'F_V0_V4', 0x00004FFC3FFB00FE, 'TLazerFenceSw', None),
    (0x0085, 'F_V0_V4', 0x00004E000F800006, 'TKomorebi', None),
    (0x0086, 'F_V0_V4', 0x00004E0000000006, 'TButterfly', None),
    (0x0087, 'F_V0_V4', 0x0000400000000006, 'TMotorcycle', None),
    (0x0088, 'F_V0_V4', 0x00004FF0B00000FE, 'TObjContainerBase2', None),
    (0x0088, 'F_EP3', 0x0000000000000002, 'TObjContainerBase2', None),
    (0x0089, 'F_V0_V4', 0x0000400000000006, 'TObjTank', None),
    (0x008A, 'F_V0_V2', 0x0000000000000006, 'TObjBattery', None),
    (0x008B, 'F_V0_V1', 0x0000000000000406, 'TObjComputer', None),
    (0x008B, 'F_V2_V4', 0x00004FFC3FFB07FE, 'TObjComputer', None),
    (0x008C, 'F_V0_V1', 0x0000000000000006, 'TObjContainerIdo', None),
    (0x008C, 'F_V2_V4', 0x000040000000000E, 'TObjContainerIdo', None),
    (0x008D, 'F_V0_V4', 0x00004000000027FE, 'TOCapsuleAncient01', None),
    (0x008E, 'F_V0_V4', 0x00004FF0000000F6, 'TOBarrierEnergy01', None),
    (0x008F, 'F_V0_V4', 0x0000400000000006, 'TObjHashi', None),
    (0x0090, 'F_V0_V4', 0x00004FFC3FFB00C6, 'TOKeyGenericSw', None),
    (0x0091, 'F_V0_V4', 0x00004FF0300000FE, 'TObjContainerEnemy', None),
    (0x0092, 'F_V0_V4', 0x00005E00B00078FE, 'TObjContainerBase', None),
    (0x0093, 'F_V0_V4', 0x00004FF0300000FE, 'TObjContainerAbeEnemy', None),
    (0x0095, 'F_V0_V4', 0x00004FF0000000FE, 'TObjContainerNoItem', None),
    (0x0096, 'F_V0_V4', 0x00004FFC3FFB07FE, 'TObjLazerFenceExtra', None),
    (0x00C0, 'F_V0_V4', 0x00004FFC3FFB0038, 'TOKeyCave01', None),
    (0x00C1, 'F_V0_V4', 0x0000400000000038, 'TODoorCave01', None),
    (0x00C2, 'F_V0_V4', 0x0000400000000038, 'TODoorCave02', None),
    (0x00C3, 'F_V0_V4', 0x0000400800780038, 'TOHangceilingCave01*', None),
    (0x00C4, 'F_V0_V4', 0x0000400000000030, 'TOSignCave01', None),
    (0x00C5, 'F_V0_V4', 0x0000400000000030, 'TOSignCave02', None),
    (0x00C6, 'F_V0_V4', 0x0000400000000030, 'TOSignCave03', None),
    (0x00C7, 'F_V0_V4', 0x0000400000000030, 'TOAirconCave01', None),
    (0x00C8, 'F_V0_V4', 0x0000400000000030, 'TOAirconCave02', None),
    (0x00C9, 'F_V0_V4', 0x000041F000000030, 'TORevlightCave01', None),
    (0x00CB, 'F_V0_V4', 0x0000400000000010, 'TORainbowCave01', None),
    (0x00CC, 'F_V0_V4', 0x0000400030000010, 'TOKurage', None),
    (0x00CD, 'F_V0_V4', 0x00004E0000610010, 'TODragonflyCave01', None),
    (0x00CE, 'F_V0_V4', 0x0000400000000038, 'TODoorCave03', None),
    (0x00CF, 'F_V0_V4', 0x00004008000000F8, 'TOBind', None),
    (0x00D0, 'F_V0_V4', 0x0000400000000020, 'TOCakeshopCave01', None),
    (0x00D1, 'F_V0_V4', 0x0000400000000008, 'TORockCaveS01', None),
    (0x00D2, 'F_V0_V4', 0x0000400000000008, 'TORockCaveM01', None),
    (0x00D3, 'F_V0_V4', 0x00004FF000000008, 'TORockCaveL01', None),
    (0x00D4, 'F_V0_V4', 0x0000000000000010, 'TORockCaveS02', None),
    (0x00D5, 'F_V0_V4', 0x0000000000000010, 'TORockCaveM02', None),
    (0x00D6, 'F_V0_V4', 0x0000000000000010, 'TORockCaveL02', None),
    (0x00D7, 'F_V0_V4', 0x0000000000000010, 'TORockCaveSS02', None),
    (0x00D8, 'F_V0_V4', 0x0000000000000010, 'TORockCaveSM02', None),
    (0x00D9, 'F_V0_V4', 0x0000000000000010, 'TORockCaveSL02', None),
    (0x00DA, 'F_V0_V4', 0x0000000000000020, 'TORockCaveS03', None),
    (0x00DB, 'F_V0_V4', 0x0000000000000020, 'TORockCaveM03', None),
    (0x00DC, 'F_V0_V4', 0x0000000000000020, 'TORockCaveL03', None),
    (0x00DE, 'F_V2_V4', 0x00004FFC3FFB07FE, 'TODummyKeyCave01', None),
    (0x00DF, 'F_V2_V4', 0x0000400000000008, 'TORockCaveBL01', None),
    (0x00E0, 'F_V2_V4', 0x0000400000000010, 'TORockCaveBL02', None),
    (0x00E1, 'F_V2_V4', 0x0000400000000020, 'TORockCaveBL03', None),
    (0x0100, 'F_V0_V4', 0x00004000000000C0, 'TODoorMachine01', None),
    (0x0101, 'F_V0_V1', 0x00000000000000C0, 'TOKeyMachine01', None),
    (0x0101, 'F_V2_V4', 0x00004FF0007B00C6, 'TOKeyMachine01', None),
    (0x0102, 'F_V0_V4', 0x00000000000000C0, 'TODoorMachine02', None),
    (0x0102, 'F_V4', 0x00004E0000000000, '__EP4_DOOR__', None),
    (0x0103, 'F_V0_V4', 0x00004008000000C0, 'TOCapsuleMachine01', None),
    (0x0104, 'F_V0_V4', 0x00004008000000C0, 'TOComputerMachine01', None),
    (0x0105, 'F_V0_V4', 0x00004008000000C0, 'TOMonitorMachine01', None),
    (0x0106, 'F_V0_V4', 0x00004000000000C0, 'TODragonflyMachine01', None),
    (0x0107, 'F_V0_V4', 0x00004000000000C0, 'TOLightMachine01', None),
    (0x0108, 'F_V0_V4', 0x00004000000000C0, 'TOExplosiveMachine01', None),
    (0x0109, 'F_V0_V4', 0x00004000000000C0, 'TOExplosiveMachine02', None),
    (0x010A, 'F_V0_V4', 0x00004000000000C0, 'TOExplosiveMachine03', None),
    (0x010B, 'F_V0_V4', 0x00004000000000C0, 'TOSparkMachine01', None),
    (0x010C, 'F_V0_V4', 0x00004000000000C0, 'TOHangerMachine01', None),
    (0x0130, 'F_V0_V4', 0x0000400000002000, 'TODoorVoShip', None),
    (0x0140, 'F_V0_V4', 0x0000400000000700, 'TObjGoalWarpAncient', None),
    (0x0141, 'F_V0_V4', 0x0000400000000700, 'TObjMapWarpAncient', None),
    (0x0142, 'F_V0_V4', 0x0000400000000700, 'TOKeyAncient02', None),
    (0x0143, 'F_V0_V4', 0x0000400000000700, 'TOKeyAncient03', None),
    (0x0144, 'F_V0_V4', 0x0000400000000100, 'TODoorAncient01', 'Usually used in Ruins 1'),
    (0x0145, 'F_V0_V4', 0x0000400000000400, 'TODoorAncient03', 'Usually used in Ruins 3'),
    (0x0146, 'F_V0_V4', 0x0000400000000200, 'TODoorAncient04', 'Usually used in Ruins 2'),
    (0x0147, 'F_V0_V4', 0x0000400000000100, 'TODoorAncient05', 'Usually used in Ruins 1'),
    (0x0148, 'F_V0_V4', 0x0000400000000200, 'TODoorAncient06', 'Usually used in Ruins 2'),
    (0x0149, 'F_V0_V4', 0x0000400000000400, 'TODoorAncient07', 'Usually used in Ruins 3'),
    (0x014A, 'F_V0_V4', 0x0000400000000700, 'TODoorAncient08', None),
    (0x014B, 'F_V0_V4', 0x0000400000000700, 'TODoorAncient09', None),
    (0x014C, 'F_V0_V4', 0x0000400000000700, 'TOSensorAncient01', None),
    (0x014D, 'F_V0_V4', 0x0000400000000700, 'TOKeyAncient01', None),
    (0x014E, 'F_V0_V4', 0x00004FF000000700, 'TOFenceAncient01', '4x2'),
    (0x014F, 'F_V0_V4', 0x00004FF000000700, 'TOFenceAncient02', '6x2'),
    (0x0150, 'F_V0_V4', 0x0000400000000700, 'TOFenceAncient03', '4x4'),
    (0x0151, 'F_V0_V4', 0x0000400000000700, 'TOFenceAncient04', '6x4'),
    (0x0152, 'F_V0_V4', 0x00004E000F800700, 'TContainerAncient01', None),
    (0x0153, 'F_V0_V4', 0x0000400000780700, 'TOTrapAncient01', None),
    (0x0154, 'F_V0_V4', 0x0000400000000700, 'TOTrapAncient02', None),
    (0x0155, 'F_V0_V4', 0x0000400000000700, 'TOMonumentAncient01', None),
    (0x0156, 'F_V0_V4', 0x0000400000000094, 'TOMonumentAncient02', None),
    (0x0159, 'F_V0_V4', 0x0000400000000700, 'TOWreckAncient01', None),
    (0x015A, 'F_V0_V4', 0x0000400000000700, 'TOWreckAncient02', None),
    (0x015B, 'F_V0_V4', 0x0000400000000700, 'TOWreckAncient03', None),
    (0x015C, 'F_V0_V4', 0x0000400000000700, 'TOWreckAncient04', None),
    (0x015D, 'F_V0_V4', 0x0000400000000700, 'TOWreckAncient05', None),
    (0x015E, 'F_V0_V4', 0x0000400000000700, 'TOWreckAncient06', None),
    (0x015F, 'F_V0_V4', 0x0000400000000700, 'TOWreckAncient07', None),
    (0x0160, 'F_V0_V4', 0x0000400000002000, 'TObjWarpBoss03', None),
    (0x0160, 'F_V0_V4', 0x00004FF030600700, 'TObjFogCollisionPoison', None),
    (0x0161, 'F_V0_V4', 0x00004003007B0700, 'TOContainerAncientItemCommon', None),
    (0x0162, 'F_V0_V4', 0x00004003007B0700, 'TOContainerAncientItemRare', None),
    (0x0163, 'F_V0_V4', 0x00004000007B0700, 'TOContainerAncientEnemyCommon', None),
    (0x0164, 'F_V0_V4', 0x00004000007B0700, 'TOContainerAncientEnemyRare', None),
    (0x0165, 'F_V2_V4', 0x00004000007B0700, 'TOContainerAncientItemNone', None),
    (0x0166, 'F_V2_V4', 0x0000400000000700, 'TOWreckAncientBrakable05', None),
    (0x0167, 'F_V2_V4', 0x0000400C3FF807C0, 'TOTrapAncient02R', None),
    (0x0170, 'F_V0_V4', 0x0000400000614000, 'TOBoss4Bird', None),
    (0x0171, 'F_V0_V4', 0x0000400000004000, 'TOBoss4Tower', None),
    (0x0172, 'F_V0_V4', 0x0000400000004000, 'TOBoss4Rock', None),
    (0x0173, 'F_V0_V2', 0x0000000000004000, 'TOSoulDF', None),
    (0x0174, 'F_V0_V2', 0x0000000000004000, 'TOButterflyDF', None),
    (0x0180, 'F_V0_V4', 0x0000400000008000, 'TObjInfoCol', None),
    (0x0180, 'F_EP3', 0x0000000000008000, 'TObjInfoCol', None),
    (0x0181, 'F_V0_V4', 0x0000400000008000, 'TObjWarpLobby', None),
    (0x0181, 'F_EP3', 0x0000000000008000, 'TObjWarpLobby', None),
    (0x0182, 'F_V3_V4', 0x0000400000008000, 'TObjLobbyMain', None),
    (0x0182, 'F_EP3', 0x0000000000008000, 'TObjLobbyMain', None),
    (0x0183, 'F_V3_V4', 0x0000400000008000, '__LOBBY_PIGEON__', None),
    (0x0183, 'F_EP3', 0x0000000000008002, '__LOBBY_PIGEON__', None),
    (0x0184, 'F_V3_V4', 0x0000400000008000, 'TObjButterflyLobby', None),
    (0x0184, 'F_EP3', 0x0000000000008002, 'TObjButterflyLobby', None),
    (0x0185, 'F_V3_V4', 0x0000400000008000, 'TObjRainbowLobby', None),
    (0x0185, 'F_EP3', 0x0000000000008002, 'TObjRainbowLobby', None),
    (0x0186, 'F_V3_V4', 0x0000400000008000, 'TObjKabochaLobby', None),
    (0x0186, 'F_EP3', 0x0000000000008000, 'TObjKabochaLobby', None),
    (0x0187, 'F_V3_V4', 0x0000400000008000, 'TObjStendGlassLobby', None),
    (0x0187, 'F_EP3', 0x0000000000008000, 'TObjStendGlassLobby', None),
    (0x0188, 'F_V3_V4', 0x0000400000008000, 'TObjCurtainLobby', None),
    (0x0188, 'F_EP3', 0x0000000000008000, 'TObjCurtainLobby', None),
    (0x0189, 'F_V3_V4', 0x0000400000008000, 'TObjWeddingLobby', None),
    (0x0189, 'F_EP3', 0x0000000000008000, 'TObjWeddingLobby', None),
    (0x018A, 'F_V3_V4', 0x0000400000008000, 'TObjTreeLobby', None),
    (0x018A, 'F_EP3', 0x0000000000008000, 'TObjTreeLobby', None),
    (0x018B, 'F_V3_V4', 0x0000400000008000, 'TObjSuisouLobby', None),
    (0x018B, 'F_EP3', 0x0000000000008000, 'TObjSuisouLobby', None),
    (0x018C, 'F_V3_V4', 0x0000400000008000, 'TObjParticleLobby', None),
    (0x018C, 'F_EP3', 0x0000000000008000, 'TObjParticleLobby', None),
    (0x018D, 'F_EP3', 0x0000000000008000, 'TObjLobbyTable', None),
    (0x018E, 'F_EP3', 0x0000000000008000, 'TObjJukeBox', None),
    (0x0190, 'F_V2_V4', 0x0000400000610000, 'TObjCamera', None),
    (0x0191, 'F_V2_V4', 0x0000400800610000, 'TObjTuitate', None),
    (0x0192, 'F_V2_V4', 0x0000400000610000, 'TObjDoaEx01', None),
    (0x0193, 'F_V2_V4', 0x0000400800610000, 'TObjBigTuitate', None),
    (0x01A0, 'F_V2_V4', 0x00004000001A0000, 'TODoorVS2Door01', None),
    (0x01A1, 'F_V2_V4', 0x00004000001A0000, 'TOVS2Wreck01', 'Partly-broken wall (like breakable wall)'),
    (0x01A2, 'F_V2_V4', 0x00004000001A0000, 'TOVS2Wreck02', 'Broken column'),
    (0x01A3, 'F_V2_V4', 0x00004000001A0000, 'TOVS2Wreck03', 'Broken wall pieces lying flat'),
    (0x01A4, 'F_V2_V4', 0x00004000001A0000, 'TOVS2Wreck04', 'Column'),
    (0x01A5, 'F_V2_V4', 0x00004000001A0000, 'TOVS2Wreck05', 'Broken toppled column'),
    (0x01A6, 'F_V2_V4', 0x00004000001A0000, 'TOVS2Wreck06', 'Truncated conic monument'),
    (0x01A7, 'F_V2_V4', 0x00004000001A0000, 'TOVS2Wall01', None),
    (0x01A8, 'F_V2_V4', 0x000041F1001A0000, '__LENS_FLARE_SWITCH_COLLISION__', None),
    (0x01A9, 'F_V2_V4', 0x00004000001A0000, 'TObjHashiVersus1', 'Small brown rising bridge'),
    (0x01AA, 'F_V2_V4', 0x00004000001A0000, 'TObjHashiVersus2', 'Long rising bridge'),
    (0x01AB, 'F_V3_V4', 0x0000400000180000, 'TODoorFourLightRuins', 'Temple'),
    (0x01C0, 'F_V3_V4', 0x0000000000600000, 'TODoorFourLightSpace', 'Spaceship'),
    (0x0200, 'F_V3_V4', 0x000041FC4F800000, 'TObjContainerJung', None),
    (0x0201, 'F_V3_V4', 0x0000400CFF800000, 'TObjWarpJung', None),
    (0x0202, 'F_V3_V4', 0x0000400C0F800000, 'TObjDoorJung', None),
    (0x0203, 'F_V3_V4', 0x0000400C4F800000, 'TObjContainerJungEx', None),
    (0x0203, 'F_V4', 0x000001F000000000, 'TObjContainerBase(0203)', None),
    (0x0204, 'F_V3_V4', 0x0000400000800000, 'TODoorJungleMain', None),
    (0x0205, 'F_V3_V4', 0x0000400C0F800000, 'TOKeyJungleMain', None),
    (0x0206, 'F_V3_V4', 0x000040040F800000, 'TORockJungleS01', 'Small rock'),
    (0x0207, 'F_V3_V4', 0x000040040F800000, 'TORockJungleM01', 'Small 3-rock wall'),
    (0x0208, 'F_V3_V4', 0x000040040F800000, 'TORockJungleL01', None),
    (0x0209, 'F_V3_V4', 0x000040040F800000, 'TOGrassJungle', None),
    (0x020A, 'F_V3_V4', 0x0000400C0F800000, 'TObjWarpJungMain', None),
    (0x020B, 'F_V3_V4', 0x0000400040800000, 'TBGLightningCtrl', None),
    (0x020C, 'F_V3_V4', 0x00004E0C0B000000, '__WHITE_BIRD__', None),
    (0x020D, 'F_V3_V4', 0x000040080B000000, '__ORANGE_BIRD__', None),
    (0x020E, 'F_V3_V4', 0x0000400C0F800000, 'TObjContainerJungEnemy', None),
    (0x020F, 'F_V3_V4', 0x0000400C3F800000, 'TOTrapChainSawDamage', None),
    (0x0210, 'F_V3_V4', 0x0000400C3F800000, 'TOTrapChainSawKey', None),
    (0x0211, 'F_V3_V4', 0x00004E0003800000, 'TOBiwaMushi', None),
    (0x0211, 'F_EP3', 0x0000000000000002, 'TOBiwaMushi', None),
    (0x0212, 'F_V3_V4', 0x000040080F800000, '__SEAGULL__', None),
    (0x0212, 'F_EP3', 0x0000000000000002, '__SEAGULL__', None),
    (0x0213, 'F_V3_V4', 0x00004E040F000000, 'TOJungleDesign', None),
    (0x0220, 'F_V3_V4', 0x0000400439008000, 'TObjFish', None),
    (0x0220, 'F_EP3', 0x0000000000008002, 'TObjFish', None),
    (0x0221, 'F_V3_V4', 0x0000400030000000, 'TODoorFourLightSeabed', 'Blue edges'),
    (0x0222, 'F_V3_V4', 0x0000400030000000, 'TODoorFourLightSeabedU', None),
    (0x0223, 'F_V3_V4', 0x0000400830000000, 'TObjSeabedSuiso_CH', None),
    (0x0224, 'F_V3_V4', 0x0000400030000000, 'TObjSeabedSuisoBrakable', None),
    (0x0225, 'F_V3_V4', 0x0000400030000000, 'TOMekaFish00', 'Blue'),
    (0x0226, 'F_V3_V4', 0x0000400030000000, 'TOMekaFish01', 'Red'),
    (0x0227, 'F_V3_V4', 0x0000400030000000, '__DOLPHIN__', None),
    (0x0228, 'F_V3_V4', 0x0000400C3F800000, 'TOTrapSeabed01', None),
    (0x0229, 'F_V3_V4', 0x0000400FFFF80000, 'TOCapsuleLabo', None),
    (0x0240, 'F_V3_V4', 0x0000400040000000, 'TObjParticle', None),
    (0x0280, 'F_V3_V4', 0x0000400100000000, '__BARBA_RAY_TELEPORTER__', None),
    (0x02A0, 'F_V3_V4', 0x0000400200000000, 'TObjLiveCamera', None),
    (0x02B0, 'F_V3_V4', 0x00004E0C0F800700, 'TContainerAncient01R', None),
    (0x02B1, 'F_V3_V4', 0x0000400000040000, 'TObjLaboDesignBase(0)', 'Computer console'),
    (0x02B1, 'F_EP3', 0x0000000000000001, 'TObjLaboDesignBase(0)', 'Computer console'),
    (0x02B2, 'F_V3_V4', 0x0000400000040000, 'TObjLaboDesignBase(1)', 'Computer console (alternate colors)'),
    (0x02B2, 'F_EP3', 0x0000000000000001, 'TObjLaboDesignBase(1)', 'Computer console (alternate colors)'),
    (0x02B3, 'F_V3_V4', 0x0000400000040000, 'TObjLaboDesignBase(2)', 'Chair'),
    (0x02B3, 'F_EP3', 0x0000000000000001, 'TObjLaboDesignBase(2)', 'Chair'),
    (0x02B4, 'F_V3_V4', 0x0000400000040000, 'TObjLaboDesignBase(3)', 'Orange wall'),
    (0x02B4, 'F_EP3', 0x0000000000000001, 'TObjLaboDesignBase(3)', 'Orange wall'),
    (0x02B5, 'F_V3_V4', 0x0000400000040000, 'TObjLaboDesignBase(4)', 'Gray/blue wall'),
    (0x02B5, 'F_EP3', 0x0000000000000001, 'TObjLaboDesignBase(4)', 'Gray/blue wall'),
    (0x02B6, 'F_V3_V4', 0x0000400000040000, 'TObjLaboDesignBase(5)', 'Long table'),
    (0x02B6, 'F_EP3', 0x0000000000000001, 'TObjLaboDesignBase(5)', 'Long table'),
    (0x02B7, 'F_GC', 0x0000000000040001, 'TObjGbAdvance', None),
    (0x02B8, 'F_V3_V4', 0x00006FFFFFFC7FFF, 'TObjQuestColALock2', None),
    (0x02B8, 'F_EP3', 0x0000000000000001, 'TObjQuestColALock2', None),
    (0x02B9, 'F_V3_V4', 0x00007FFC3FFF78FF, 'TObjMapForceWarp', None),
    (0x02B9, 'F_EP3', 0x0000000000000001, 'TObjMapForceWarp', None),
    (0x02BA, 'F_V3_V4', 0x00006FFFFFFC7FFF, 'TObjQuestCol2', None),
    (0x02BA, 'F_EP3', 0x0000000000000001, 'TObjQuestCol2', None),
    (0x02BB, 'F_V3_V4', 0x0000400000040000, 'TODoorLaboNormal', None),
    (0x02BC, 'F_V3_V4', 0x0000400080000000, 'TObjAreaWarpEndingJung', None),
    (0x02BD, 'F_V3_V4', 0x0000400000040000, 'TObjLaboMapWarp', None),
    (0x02D0, 'F_EP3', 0x0000000000000002, 'TObjKazariCard', None),
    (0x02D1, 'F_EP3', 0x0000000000000001, 'TObj_FloatingCardMaterial_Dark', None),
    (0x02D2, 'F_EP3', 0x0000000000000001, 'TObj_FloatingCardMaterial_Hero', None),
    (0x02D3, 'F_EP3', 0x0000000000000001, 'TObjCardCityMapWarp(0)', 'Battle counter warp (blue lines)'),
    (0x02D9, 'F_EP3', 0x0000000000000001, 'TObjCardCityMapWarp(1)', 'Battle counter warp (green lines; unused)'),
    (0x02E3, 'F_EP3', 0x0000000000000001, 'TObjCardCityMapWarp(2)', 'Lobby warp (yellow lines)'),
    (0x02D4, 'F_EP3', 0x0000000000000001, 'TObjCardCityDoor(0)', 'Yellow V-pattern (to deck edit room)'),
    (0x02D5, 'F_EP3', 0x0000000000000001, 'TObjCardCityDoor(1)', 'Blue V-pattern (to battle entry counter)'),
    (0x02D8, 'F_EP3', 0x0000000000000001, 'TObjCardCityDoor(2)', 'Green V-pattern (unused)'),
    (0x02DF, 'F_EP3', 0x0000000000000001, 'TObjCardCityDoor(3)', 'Blue X-pattern (to lobby teleporter)'),
    (0x02E0, 'F_EP3', 0x0000000000000001, 'TObjCardCityDoor(4)', 'Gray (to chief)'),
    (0x02DC, 'F_EP3', 0x0000000000000001, 'TObjCardCityDoor_Closed(0)', 'Yellow V-pattern (to deck edit room)'),
    (0x02DD, 'F_EP3', 0x0000000000000001, 'TObjCardCityDoor_Closed(1)', 'Blue V-pattern (to battle entry counter)'),
    (0x02DE, 'F_EP3', 0x0000000000000001, 'TObjCardCityDoor_Closed(2)', 'Green V-pattern (unused)'),
    (0x02E1, 'F_EP3', 0x0000000000000001, 'TObjCardCityDoor_Closed(3)', 'Opaque gray X-pattern'),
    (0x02E2, 'F_EP3', 0x0000000000000001, 'TObjCardCityDoor_Closed(4)', 'Gray (to chief)'),
    (0x02D6, 'F_EP3', 0x0000000000000002, 'TObjKazariGeyserMizu', None),
    (0x02D7, 'F_EP3', 0x0000000000000002, 'TObjSetCardColi', None),
    (0x02DA, 'F_EP3', 0x0000000000000001, 'TOFlyMekaHero', None),
    (0x02DB, 'F_EP3', 0x0000000000000001, 'TOFlyMekaDark', None),
    (0x02E4, 'F_EP3', 0x0000000000008001, 'TObjSinBoardCard', None),
    (0x02E5, 'F_EP3', 0x0000000000000001, 'TObjCityMoji', None),
    (0x02E6, 'F_EP3', 0x0000000000000001, 'TObjCityWarpOff', None),
    (0x02E7, 'F_EP3', 0x0000000000000001, 'TObjFlyCom', None),
    (0x02E8, 'F_EP3', 0x0000000000000001, '__UNKNOWN_02E8__', None),
    (0x0300, 'F_V4', 0x00005FF000000000, '__EP4_LIGHT__', None),
    (0x0301, 'F_V4', 0x00004FF000000000, '__WILDS_CRATER_CACTUS__', None),
    (0x0302, 'F_V4', 0x00004FF000000000, '__WILDS_CRATER_BROWN_ROCK__', None),
    (0x0303, 'F_V4', 0x00004FF000000000, '__WILDS_CRATER_BROWN_ROCK_DESTRUCTIBLE__', None),
    (0x0340, 'F_V4', 0x0000400000000000, '__UNKNOWN_0340__', None),
    (0x0341, 'F_V4', 0x0000400000000000, '__UNKNOWN_0341__', None),
    (0x0380, 'F_V4', 0x00004E0000000000, '__POISON_PLANT__', None),
    (0x0381, 'F_V4', 0x00004E0000000000, '__UNKNOWN_0381__', None),
    (0x0382, 'F_V4', 0x00004E0000000000, '__UNKNOWN_0382__', None),
    (0x0383, 'F_V4', 0x00004E0000000000, '__DESERT_OOZE_PLANT__', None),
    (0x0385, 'F_V4', 0x00004E0000000000, '__UNKNOWN_0385__', None),
    (0x0386, 'F_V4', 0x00004FF000000000, '__WILDS_CRATER_BLACK_ROCKS__', None),
    (0x0387, 'F_V4', 0x00004E0000000000, '__UNKNOWN_0387__', None),
    (0x0388, 'F_V4', 0x00004E0000000000, '__UNKNOWN_0388__', None),
    (0x0389, 'F_V4', 0x0000400000000000, '__GAME_FLAG_SET_CLEAR_ZONE__', None),
    (0x038A, 'F_V4', 0x0000400000000000, '__HP_DRAIN_ZONE__', None),
    (0x038B, 'F_V4', 0x00004E0000000000, '__FALLING_STALACTITE__', None),
    (0x038C, 'F_V4', 0x00004E0000000000, '__DESERT_PLANT_SOLID__', None),
    (0x038D, 'F_V4', 0x00004E0000000000, '__DESERT_CRYSTALS_BOX__', None),
    (0x038E, 'F_V4', 0x0000400000000000, '__EP4_TEST_DOOR__', None),
    (0x038F, 'F_V4', 0x00004E0000000000, '__BEEHIVE__', None),
    (0x0390, 'F_V4', 0x00004E0000000000, '__EP4_TEST_PARTICLE__', None),
    (0x0391, 'F_V4', 0x00004E0000000000, '__HEAT__', None),
    (0x03C0, 'F_V4', 0x0000500000000000, '__EP4_BOSS_EGG__', None),
    (0x03C1, 'F_V4', 0x0000500000000000, '__EP4_BOSS_ROCK_SPAWNER__', None),
]

ENEMY_TABLE_ROWS: list[tuple[int, str, int, str, str | None]] = [
    (0x0001, 'F_V0_V4', 0x0000200000000001, 'TObjNpcFemaleBase', 'Woman with red hair and purple outfit'),
    (0x0001, 'F_EP3', 0x0000000000000001, 'TObjNpcFemaleBase', 'Woman with red hair and purple outfit'),
    (0x0002, 'F_V0_V4', 0x0000200000000001, 'TObjNpcFemaleChild', 'Shorter version of the above'),
    (0x0002, 'F_EP3', 0x0000000000000001, 'TObjNpcFemaleChild', 'Shorter version of the above'),
    (0x0003, 'F_V0_V4', 0x0000200000040001, 'TObjNpcFemaleDwarf', 'Woman wearing green outfit'),
    (0x0003, 'F_EP3', 0x0000000000000001, 'TObjNpcFemaleDwarf', 'Woman wearing green outfit'),
    (0x0004, 'F_V0_V4', 0x0000200000000001, 'TObjNpcFemaleFat', "Woman outside Hunter's Guild"),
    (0x0004, 'F_EP3', 0x0000000000000001, 'TObjNpcFemaleFat', "Woman outside Hunter's Guild"),
    (0x0005, 'F_V0_V4', 0x0000200000000001, 'TObjNpcFemaleMacho', 'Tool shop woman'),
    (0x0005, 'F_EP3', 0x0000000000000001, 'TObjNpcFemaleMacho', 'Tool shop woman'),
    (0x0006, 'F_V0_V4', 0x0000200000040001, 'TObjNpcFemaleOld', 'Older woman with yellow/red outfit'),
    (0x0006, 'F_EP3', 0x0000000000000001, 'TObjNpcFemaleOld', 'Older woman with yellow/red outfit'),
    (0x0007, 'F_V0_V4', 0x0000200000000001, 'TObjNpcFemaleTall', 'Woman walking around inside shop area'),
    (0x0007, 'F_EP3', 0x0000000000000001, 'TObjNpcFemaleTall', 'Woman walking around inside shop area'),
    (0x0008, 'F_V0_V4', 0x0000200000008001, 'TObjNpcMaleBase', 'Similar appearance to weapon shop man'),
    (0x0008, 'F_EP3', 0x0000000000008001, 'TObjNpcMaleBase', 'Similar appearance to weapon shop man'),
    (0x0009, 'F_V0_V4', 0x0000200000040001, 'TObjNpcMaleChild', 'Kid wearing purple'),
    (0x0009, 'F_EP3', 0x0000000000000001, 'TObjNpcMaleChild', 'Kid wearing purple'),
    (0x000A, 'F_V0_V4', 0x0000200000000001, 'TObjNpcMaleDwarf', 'Man outside Medical Center'),
    (0x000A, 'F_EP3', 0x0000000000000001, 'TObjNpcMaleDwarf', 'Man outside Medical Center'),
    (0x000B, 'F_V0_V4', 0x0000200000040001, 'TObjNpcMaleFat', 'Armor shop man'),
    (0x000B, 'F_EP3', 0x0000000000000001, 'TObjNpcMaleFat', 'Armor shop man'),
    (0x000C, 'F_V0_V4', 0x0000200000000001, 'TObjNpcMaleMacho', 'Weapon shop man'),
    (0x000C, 'F_EP3', 0x0000000000000001, 'TObjNpcMaleMacho', 'Weapon shop man'),
    (0x000D, 'F_V0_V4', 0x0000200000040001, 'TObjNpcMaleOld', 'Man near telepipe locations'),
    (0x000D, 'F_EP3', 0x0000000000000001, 'TObjNpcMaleOld', 'Man near telepipe locations'),
    (0x000E, 'F_V0_V4', 0x0000200000040001, 'TObjNpcMaleTall', 'Man wearing turquoise'),
    (0x000E, 'F_EP3', 0x0000000000000001, 'TObjNpcMaleTall', 'Man wearing turquoise'),
    (0x0019, 'F_V0_V4', 0x00003FF000040001, 'TObjNpcSoldierBase', 'Man right of the Ragol warp door'),
    (0x0019, 'F_EP3', 0x0000000000000001, 'TObjNpcSoldierBase', 'Man right of the Ragol warp door'),
    (0x001A, 'F_V0_V4', 0x0000200000000001, 'TObjNpcSoldierMacho', 'Man left of the Ragol warp door'),
    (0x001A, 'F_EP3', 0x0000000000000001, 'TObjNpcSoldierMacho', 'Man left of the Ragol warp door'),
    (0x001B, 'F_V0_V4', 0x0000200000040001, 'TObjNpcGovernorBase', 'Principal Tyrell'),
    (0x001B, 'F_EP3', 0x0000000000000001, 'TObjNpcGovernorBase', 'Principal Tyrell'),
    (0x001C, 'F_V0_V4', 0x0000200000040001, 'TObjNpcConnoisseur', 'Tekker'),
    (0x001D, 'F_V0_V4', 0x0000200000040021, 'TObjNpcCloakroomBase', 'Bank woman'),
    (0x001E, 'F_V0_V4', 0x0000200000000001, 'TObjNpcExpertBase', 'Man in front of bank'),
    (0x001F, 'F_V0_V4', 0x0000200000040001, 'TObjNpcNurseBase', 'Nurses in Medical Center'),
    (0x0020, 'F_V0_V4', 0x0000200000040001, 'TObjNpcSecretaryBase', 'Irene'),
    (0x0020, 'F_EP3', 0x0000000000000001, 'TObjNpcSecretaryBase', 'Karen'),
    (0x0021, 'F_V0_V4', 0x0000200000000001, 'TObjNpcHHM00', 'TODO'),
    (0x0021, 'F_EP3', 0x0000000000000001, 'TObjNpcHHM00', 'TODO'),
    (0x0022, 'F_V0_V4', 0x0000200000000001, 'TObjNpcNHW00', 'TODO'),
    (0x0022, 'F_EP3', 0x0000000000000001, 'TObjNpcNHW00', 'TODO'),
    (0x0024, 'F_V0_V4', 0x0000200000000001, 'TObjNpcHRM00', 'TODO'),
    (0x0025, 'F_V0_V4', 0x0000200000040001, 'TObjNpcARM00', 'TODO'),
    (0x0026, 'F_V0_V4', 0x0000200000040001, 'TObjNpcARW00', 'TODO'),
    (0x0026, 'F_EP3', 0x0000000000000001, 'TObjNpcARW00', 'TODO'),
    (0x0027, 'F_V0_V4', 0x0000200000040001, 'TObjNpcHFW00', 'TODO'),
    (0x0027, 'F_EP3', 0x0000000000000001, 'TObjNpcHFW00', 'TODO'),
    (0x0028, 'F_V0_V4', 0x0000200000040001, 'TObjNpcNFM00', 'TODO'),
    (0x0028, 'F_EP3', 0x0000000000000001, 'TObjNpcNFM00', 'TODO'),
    (0x0029, 'F_V0_V4', 0x00003C0000000001, 'TObjNpcNFW00', 'TODO'),
    (0x0029, 'F_EP3', 0x0000000000000001, 'TObjNpcNFW00', 'TODO'),
    (0x002B, 'F_V0_V4', 0x0000200000000001, 'TObjNpcNHW01', 'TODO'),
    (0x002C, 'F_V0_V4', 0x0000200000000001, 'TObjNpcAHM01', 'TODO'),
    (0x002D, 'F_V0_V4', 0x0000200000000001, 'TObjNpcHRM01', 'TODO'),
    (0x0030, 'F_V0_V4', 0x0000200000000001, 'TObjNpcHFW01', 'TODO'),
    (0x0031, 'F_V0_V4', 0x0000200000040001, 'TObjNpcNFM01', 'TODO'),
    (0x0031, 'F_EP3', 0x0000000000000001, 'TObjNpcNFM01', 'TODO'),
    (0x0032, 'F_V0_V4', 0x00002C0000000001, 'TObjNpcNFW01', 'TODO'),
    (0x0045, 'F_V0_V4', 0x00000FF40F800006, 'TObjNpcLappy', 'Rappy NPC'),
    (0x0046, 'F_V0_V4', 0x0000000000000004, 'TObjNpcMoja', 'Small Hildebear NPC'),
    (0x0047, 'F_V2', 0x0000000000000004, 'TObjNpcRico', 'Rico'),
    (0x00A9, 'F_V0_V4', 0x0000000000000600, 'TObjNpcBringer', 'Dark Bringer NPC'),
    (0x00D0, 'F_V3_V4', 0x0000200000040001, 'TObjNpcKenkyu', 'Ep2 armor shop man'),
    (0x00D1, 'F_V3_V4', 0x0000200000040001, 'TObjNpcSoutokufu', 'Natasha Milarose'),
    (0x00D2, 'F_V3_V4', 0x0000000000040000, 'TObjNpcHosa', 'Dan'),
    (0x00D3, 'F_V3_V4', 0x000000F000040000, 'TObjNpcKenkyuW', 'Ep2 tool shop woman'),
    (0x00D6, 'F_EP3', 0x0000000000000001, 'TObjNpcHeroGovernor', 'Morgue Chief'),
    (0x00D7, 'F_EP3', 0x0000000000000001, 'TObjNpcHeroGovernor', 'Morgue Chief (direct alias of 00D6)'),
    (0x00F0, 'F_V3_V4', 0x0000000000040000, 'TObjNpcHosa2', 'Man next to room with warp to Lab'),
    (0x00F1, 'F_V3_V4', 0x0000000000040000, 'TObjNpcKenkyu2', 'Ep2 weapon shop man'),
    (0x00F2, 'F_V3_V4', 0x0000000000040000, 'TObjNpcNgcBase(0x00F2)', 'TODO'),
    (0x00F3, 'F_V3_V4', 0x00003FF000040000, 'TObjNpcNgcBase(0x00F3)', 'TODO'),
    (0x00F4, 'F_V3_V4', 0x00003FF030040000, 'TObjNpcNgcBase(0x00F4)', 'TODO'),
    (0x00F5, 'F_V3_V4', 0x0000000000040000, 'TObjNpcNgcBase(0x00F5)', 'TODO'),
    (0x00F6, 'F_V3_V4', 0x000000080F840000, 'TObjNpcNgcBase(0x00F6)', 'TODO'),
    (0x00F7, 'F_V3_V4', 0x0000000000040000, 'TObjNpcNgcBase(0x00F7)', 'Nol'),
    (0x00F8, 'F_V3_V4', 0x0000000000040000, 'TObjNpcNgcBase(0x00F8)', 'Elly'),
    (0x00F9, 'F_V3_V4', 0x0000000000040000, 'TObjNpcNgcBase(0x00F9)', 'Woman with cyan hair'),
    (0x00FA, 'F_V3_V4', 0x0000000000040000, 'TObjNpcNgcBase(0x00FA)', 'Woman with bright red hair'),
    (0x00FB, 'F_V3_V4', 0x0000000000040000, 'TObjNpcNgcBase(0x00FB)', 'Man with blue hair near the Ep2 Medical Center'),
    (0x00FC, 'F_V3_V4', 0x0000000000040000, 'TObjNpcNgcBase(0x00FC)', "Man in room next to Ep2 Hunter's Guild"),
    (0x00FD, 'F_V3_V4', 0x000000040F840000, 'TObjNpcNgcBase(0x00FD)', 'TODO'),
    (0x00FE, 'F_V3_V4', 0x0000000000040000, 'TObjNpcNgcBase(0x00FE)', "Episode 2 Hunter's Guild woman"),
    (0x00FF, 'F_V3_V4', 0x0000000000040000, 'TObjNpcNgcBase(0x00FF)', 'Woman near room with teleporter to VR areas'),
    (0x0100, 'F_V4', 0x0000200000040001, '__MOMOKA__', 'Momoka'),
    (0x0110, 'F_EP3', 0x0000000000000001, 'TObjNpcWalkingMeka_Hero', 'Small talking robot in Morgue'),
    (0x0111, 'F_EP3', 0x0000000000000001, 'TObjNpcWalkingMeka_Dark', 'Small talking robot in Morgue'),
    (0x00D4, 'F_EP3', 0x0000000000000001, 'TObjNpcHeroScientist', None),
    (0x00D5, 'F_EP3', 0x0000000000000001, 'TObjNpcHeroScientist', None),
    (0x0112, 'F_EP3', 0x0000000000000001, 'TObjNpcHeroAide', None),
    (0x0118, 'F_V4', 0x00007FF000000000, '__QUEST_NPC__', None),
    (0x0033, 'F_V3_V4', 0x0000200FFFFFFFFF, 'TObjNpcEnemy', None),
    (0x0033, 'F_EP3', 0x0000000000008001, 'TObjNpcEnemy', None),
    (0x0040, 'F_V0_V4', 0x00000000001B0004, 'TObjEneMoja', None),
    (0x0041, 'F_V0_V4', 0x00004FF000180006, 'TObjEneLappy', None),
    (0x0042, 'F_V0_V4', 0x0000000000180006, 'TObjEneBm3FlyNest', None),
    (0x0043, 'F_V0_V4', 0x0000000000600006, 'TObjEneBm5Wolf', None),
    (0x0044, 'F_V0_V4', 0x0000000000000006, 'TObjEneBeast', None),
    (0x0060, 'F_V0_V4', 0x00000000001B0018, 'TObjGrass', None),
    (0x0061, 'F_V0_V4', 0x0000000800180038, 'TObjEneRe2Flower', None),
    (0x0062, 'F_V0_V4', 0x0000000000000038, 'TObjEneNanoDrago', None),
    (0x0063, 'F_V0_V4', 0x0000000000030038, 'TObjEneShark', None),
    (0x0064, 'F_V0_V4', 0x0000000000000030, 'TObjEneSlime', None),
    (0x0065, 'F_V0_V4', 0x0000000000600028, 'TObjEnePanarms', None),
    (0x0080, 'F_V0_V4', 0x00000000006000C0, 'TObjEneDubchik', None),
    (0x0081, 'F_V0_V4', 0x00000000002000C0, 'TObjEneGyaranzo', None),
    (0x0082, 'F_V0_V4', 0x00000000000300C0, 'TObjEneMe3ShinowaReal', None),
    (0x0083, 'F_V0_V4', 0x00000000000000C0, 'TObjEneMe1Canadin', None),
    (0x0084, 'F_V0_V4', 0x00000000000000C0, 'TObjEneMe1CanadinLeader', None),
    (0x0085, 'F_V0_V4', 0x00000000006000C0, 'TOCtrlDubchik', None),
    (0x00A0, 'F_V0_V4', 0x0000000000630300, 'TObjEneSaver', None),
    (0x00A1, 'F_V0_V4', 0x0000000000400500, 'TObjEneRe4Sorcerer', None),
    (0x00A2, 'F_V0_V4', 0x0000000000000600, 'TObjEneDarkGunner', None),
    (0x00A3, 'F_V0_V4', 0x0000000000000600, 'TObjEneDarkGunCenter', None),
    (0x00A4, 'F_V0_V4', 0x0000000000030600, 'TObjEneDf2Bringer', None),
    (0x00A5, 'F_V0_V4', 0x0000000000180500, 'TObjEneRe7Berura', None),
    (0x00A6, 'F_V0_V4', 0x0000000000180700, 'TObjEneDimedian', None),
    (0x00A7, 'F_V0_V4', 0x0000000000000700, 'TObjEneBalClawBody', None),
    (0x00A8, 'F_V0_V4', 0x0000000000000700, 'TObjEneBalClawClaw', None),
    (0x00C0, 'F_V0_V4', 0x0000000000000800, 'TBoss1Dragon', 'Dragon'),
    (0x00C0, 'F_V3_V4', 0x0000000040000000, 'TBoss5Gryphon', 'Gal Gryphon'),
    (0x00C1, 'F_V0_V4', 0x0000000000001000, 'TBoss2DeRolLe', 'De Rol Le'),
    (0x00C2, 'F_V0_V4', 0x0000000000002000, 'TBoss3Volopt', 'Main control object'),
    (0x00C3, 'F_V0_V4', 0x0000000000002000, 'TBoss3VoloptP01', 'Phase 1 (x6; one for each big monitor)'),
    (0x00C4, 'F_V0_V4', 0x0000000000002000, 'TBoss3VoloptCore', 'Core'),
    (0x00C5, 'F_V0_V4', 0x0000000000002000, 'TBoss3VoloptP02', 'Phase 2'),
    (0x00C6, 'F_V0_V4', 0x0000000000002000, 'TBoss3VoloptMonitor', 'Monitor (x24; 4 for each wall)'),
    (0x00C7, 'F_V0_V4', 0x0000000000002000, 'TBoss3VoloptHiraisin', 'Pillar (lightning rod)'),
    (0x00C8, 'F_V0_V4', 0x0000000000004000, 'TBoss4DarkFalz', 'Dark Falz'),
    (0x00CA, 'F_V3_V4', 0x0000000080000000, 'TBoss6PlotFalz', 'Olga Flow'),
    (0x00CB, 'F_V3_V4', 0x0000000100000000, 'TBoss7DeRolLeC', 'Barba Ray'),
    (0x00CC, 'F_V3_V4', 0x0000000200000000, 'TBoss8Dragon', 'Gol Dragon'),
    (0x00D4, 'F_V3_V4', 0x000000000F800000, 'TObjEneMe3StelthReal', None),
    (0x00D5, 'F_V3_V4', 0x000000040F800000, 'TObjEneMerillLia', None),
    (0x00D6, 'F_V3_V4', 0x000000080F800000, 'TObjEneBm9Mericarol', None),
    (0x00D7, 'F_V3_V4', 0x000000040F800000, 'TObjEneBm5GibonU', None),
    (0x00D8, 'F_V3_V4', 0x000000080F800000, 'TObjEneGibbles', None),
    (0x00D9, 'F_V3_V4', 0x000000040F800000, 'TObjEneMe1Gee', None),
    (0x00DA, 'F_V3_V4', 0x000000080F800000, 'TObjEneMe1GiGue', None),
    (0x00DB, 'F_V3_V4', 0x0000000030000000, 'TObjEneDelDepth', None),
    (0x00DC, 'F_V3_V4', 0x0000000830000000, 'TObjEneDellBiter', None),
    (0x00DD, 'F_V3_V4', 0x0000000430000000, 'TObjEneDolmOlm', None),
    (0x00DE, 'F_V3_V4', 0x0000000030000000, 'TObjEneMorfos', None),
    (0x00DF, 'F_V3_V4', 0x0000000C30000000, 'TObjEneRecobox', None),
    (0x00E0, 'F_V3_V4', 0x0000000030000000, 'TObjEneMe3SinowZoaReal', None),
    (0x00E0, 'F_V3_V4', 0x0000000800000000, 'TObjEneEpsilonBody', None),
    (0x00E1, 'F_V3_V4', 0x0000000800000000, 'TObjEneIllGill', None),
    (0x0110, 'F_V4', 0x000041F000000000, '__ASTARK__', None),
    (0x0111, 'F_V4', 0x00004FF000000000, '__SATELLITE_LIZARD_YOWIE__', None),
    (0x0112, 'F_V4', 0x00004E0000000000, '__MERISSA_A__', None),
    (0x0113, 'F_V4', 0x00004E0000000000, '__GIRTABLULU__', None),
    (0x0114, 'F_V4', 0x00004FF000000000, '__ZU__', None),
    (0x0115, 'F_V4', 0x000041F000000000, '__BOOTA_FAMILY__', None),
    (0x0116, 'F_V4', 0x000041F000000000, '__DORPHON__', None),
    (0x0117, 'F_V4', 0x00004E0000000000, '__GORAN_FAMILY__', None),
    (0x0119, 'F_V4', 0x0000100000000000, '__EPISODE_4_BOSS__', None),
]


OBJECT_TABLE: dict[int, EntityDef] = _rows_to_entity_dict(OBJECT_TABLE_ROWS)
ENEMY_TABLE: dict[int, EntityDef] = _rows_to_entity_dict(ENEMY_TABLE_ROWS)


# ---------------------------------------------------------------------------
# Expected BML inner counts (Wave 2 / Agent B, 2026-04-26)
# ---------------------------------------------------------------------------
# Ground-truth ``.nj`` + ``.xj`` model-inner counts for high-value BMLs,
# verified by walking the BML on disk with ``formats.bml.parse_bml`` and
# cross-checked against ``_reference/PSOBMLExtract/PSOBMLExtract/BMLUtil.cs``.
#
# Keys are the BML filename WITHOUT directory. The value is the count of
# entries whose lowercased name ends in ``.nj`` or ``.xj`` — i.e. the
# entries the asset tree's inner picker surfaces as "model parts".
# Animation entries (``.njm`` / ``.njs``) are NOT counted here; those go
# through the animation discovery path which is owned by parallel agent C.
#
# This map is consulted by ``server.py``'s inner-discovery validator: when
# the server is asked for the inner list of a BML named here and the
# discovered count differs from the expected count, a WARN is logged and
# the request still completes — we never fail-closed on a mismatch.
#
# To extend: add ``"<bml_name>": <int>`` after walking the file with
# ``parse_bml``; document any ``_break`` / ``_hahen`` / shadow / LOD
# inners that contributed to the count in the trailing comment.
# See ``_reports/inner_discovery_audit.md`` for the diagnostic walk.

EXPECTED_BML_INNER_COUNTS: dict[str, int] = {
    # --- Bosses ---
    "bm_boss1_dragon.bml":           2,    # nb_dragon (main tree) + sd_dragon (shadow proxy)
    "bm_boss1_dragon_a.bml":         2,    # episode-1 alt; identical inner topology to the base
    "bm_boss8_dragon.bml":           3,    # main + lo_main + sd (Gol Dragon, ep2)
    "bm_boss2_de_rol_le.bml":        7,    # body + fin_a + fin_b + sting + tentacle + helm_break + shell_break
    "bm_boss7_de_rol_le_c.bml":      6,    # ep4-leftover variant: body + tentacle + helm_break + shell_break + ikada_hahen + hige_at01
    "bm_boss5_gryphon.bml":          4,    # s_body + ss_body + lo_s_body + lo_ss_body
    "bm_boss7_crawfish.bml":         2,    # minibaru_body + minibaru_body_b
    "bm_boss3_volopt.bml":           25,   # 8 monitor variants × 3 (aka/ao/hakai) + ceiling shards + Pillar parts
    # --- Enemies ---
    "bm_ene_bm9_s_mericarol.bml":    1,    # bm9_s_meri_body.nj
    "bm_ene_re8_merill_lia.bml":     2,    # beast_wola + srbeast_wola
    "bm_ene_re4_sorcerer.bml":       2,    # sorcer_body + bit (orbital)
    "bm_ene_recobox.bml":            6,    # me7_all + me7_box_all + bomb_body + parts01-03
    "bm_ene_lappy.bml":              2,    # base + s_base (color variants)
    "bm_ene_bm1_shark.bml":          3,    # 3 size tiers
    # --- Sub-bosses / NPC family ---
    "bm4_ps_ma_body.bml":            6,    # Sinow Beat: ma_body + ma_tail + mar_body + mar_tail + mb_body + mbr_body
    # --- Map objects ---
    "bm_obj_warpboss_ancient.bml":   3,    # warp_gawa + warp_beam + warp_sbeam (all .xj)
    "bm_obj_warpboss_jungle.bml":    2,    # warp4_dodai + warp4_jogo_light (all .xj)
    "bm_fe_obj_o_door01l.bml":       1,    # door01l.xj
    "bm_obj_ep4_bee_a.bml":          1,    # bee_a.nj
    "bm_o_explosive_machine.bml":    9,    # bapipe + baswitch + batank, each in 3 variants
}


def expected_bml_inner_count(bml_filename: str) -> int | None:
    """Return the expected ``.nj``/``.xj`` inner count for a BML, if known.

    Returns ``None`` for any BML not in ``EXPECTED_BML_INNER_COUNTS``;
    callers should treat that as "no ground truth, accept whatever the
    walker reports".
    """
    return EXPECTED_BML_INNER_COUNTS.get(bml_filename)


def lookup_enemy(type_id: int) -> EntityDef | None:
    """Return canonical EntityDef for an enemy ``type_id`` (BB-preferred)."""
    return ENEMY_TABLE.get(type_id)


def lookup_object(type_id: int) -> EntityDef | None:
    """Return canonical EntityDef for an object ``type_id`` (BB-preferred)."""
    return OBJECT_TABLE.get(type_id)


# ---------------------------------------------------------------------------
# Populate ``bml_inner_count_hint`` on the canonical EntityDef for a few
# class names with stable, unambiguous BML mappings. The hint is a soft
# guideline: when more than one BML can host this class (e.g. shared
# bodies across episodes), we pick the LARGEST observed count so a
# walker that returns fewer parts triggers a warning. Single-entry
# BMLs leave the hint as the exact count.
# ---------------------------------------------------------------------------
_CLASS_TO_INNER_HINT: dict[str, int] = {
    "TBoss1Dragon":       2,
    "TBoss5Gryphon":      4,
    "TBoss2DeRolLe":      7,
    "TBoss8Dragon":       3,
}
for _ed in ENEMY_TABLE.values():
    _hint = _CLASS_TO_INNER_HINT.get(_ed.class_name)
    if _hint is not None:
        _ed.bml_inner_count_hint = _hint
del _ed  # don't pollute module ns


__all__ += [
    "EntityDef",
    "VERSION_FLAG_BITS",
    "OBJECT_TABLE",
    "ENEMY_TABLE",
    "OBJECT_TABLE_ROWS",
    "ENEMY_TABLE_ROWS",
    "PER_FLOOR_ENEMY_INDEX",
    "SETDATA_NAMES",
    "EXPECTED_BML_INNER_COUNTS",
    "expected_bml_inner_count",
    "lookup_enemy",
    "lookup_object",
    "all_defs_for_type",
]
