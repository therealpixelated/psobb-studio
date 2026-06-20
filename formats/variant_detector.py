"""
PSOBB monster / NPC color-variant detector.

PSOBB ships many enemy models as VARIANT pairs sharing geometry but with
different texture sets — Mericarol/Mericus/Merikle, Booma/Gobooma/Gigobooma,
Hildebear/Hildeblue, Sinow Beat/Sinow Gold, etc. Two distinct packing styles
appear in the data:

  1. CROSS-BML variants — different .bml files in data/ that share a stem
     prefix and differ only by a suffix. Examples: ``bm_ene_bm1_shark.bml``
     vs ``bm_ene_bm1_shark_a.bml`` (state variant), ``bm_ene_lappy.bml`` vs
     ``bm_ene_lappy_es.bml``/``_hw.bml``/``_xs.bml`` (per-episode skin).

  2. INTRA-BML variants — a single BML whose NJTL chunk references multiple
     "slots" of the same texture (typical pattern: 3 base textures + 3 b-suffixed
     + 3 c-suffixed = same geometry rendered with three color palettes). The
     Mericarol BML is the textbook case: one ``bm9_s_meri_body.nj`` referencing
     9 NJTL slots that group into 3 sets of 3 → mericarol/mericus/merikle.

This module surfaces BOTH styles via a single ``detect_variants(bml_path)``
entry-point. It returns a list of variant descriptors that the frontend can
turn into clickable pills above the model viewport.

Public API:
  - VariantInfo (dataclass)  one variant entry
  - detect_variants(bml_path, *, data_dir=None, include_self=True) -> list[VariantInfo]

Heuristic priority (for cross-BML detection, applied in order):
  - exact-prefix + suffix in {_a, _b, _c, _low, _high}  → tag "lod" / "damaged"
  - exact-prefix + suffix in {_es, _hw, _xs, _es4, _hw4, _xs4}  → tag "color"
                                                                   (Episode skin)
  - exact-prefix + suffix _ap[_es|_hw|_xs]  → tag "color" (Pal Lappy variants)
  - exact-prefix + extra path tokens (all-alpha) → tag "color" by default

For intra-BML detection we parse the NJTL chunk and group slots by trimming a
trailing single-letter color marker (b/c/d/r). When ≥2 groups of equal size
appear, each group is a variant.

There is also a HARDCODED table of well-known families that overrides the
automatic suffix grouping when the file basenames are non-obvious — kept small
and easy to extend.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .bml import parse_bml, _prs_decompress
from .njtl import find_and_parse_njtl


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class VariantInfo:
    """One detected variant of a model.

    Attributes:
      path:          API path the frontend can open (e.g. ``bm_ene_lappy_es.bml``
                     for cross-BML; ``bm_ene_bm9_s_mericarol.bml#bm9_s_meri_body.nj?slot_group=1``
                     for intra-BML).
      label:         Human display label (e.g. "Lappy (Easter)", "Mericus").
      variant_kind:  "color" | "lod" | "damaged" | "self".
      icon_color:    Suggested swatch color (hex). Filled lazily from the
                     dominant XVR color sample when available; otherwise
                     a fallback hue keyed off the variant's index in the family.
      slot_group:    Optional 0-based offset into NJTL slot ranges for intra-BML
                     variants. ``None`` for cross-BML variants.
      slot_count:    Optional NJTL-slot stride for intra-BML variants. ``None``
                     for cross-BML. Together (slot_group, slot_count) tell the
                     renderer how to remap material_id → tile_index.
      is_self:       True iff this entry IS the model the user opened (so the
                     UI knows which pill to highlight).
    """
    path: str
    label: str
    variant_kind: str = "color"
    icon_color: str = "#888888"
    slot_group: Optional[int] = None
    slot_count: Optional[int] = None
    is_self: bool = False


# ---------------------------------------------------------------------------
# Suffix tables and family hardcodes
# ---------------------------------------------------------------------------

# Suffixes that map to LOD/damaged variants (we still surface them so the
# user can preview the lower-detail / hurt model).
_LOD_SUFFIXES = ("_low", "_high")
# State suffixes — typically the "alternate state" of the same model
# (damaged, secondary phase, etc.). Tagged "damaged" because that's the
# most common semantic in PSOBB data.
_STATE_SUFFIXES = ("_a", "_b", "_c")
# Color-skin suffixes — Episode-themed palettes from PSO data.
_EPISODE_SUFFIXES = (
    "_es", "_es4",   # Easter
    "_hw", "_hw4",   # Halloween
    "_xs", "_xs4",   # Christmas (Xmas)
)
# Pal Lappy variants prefix the episode suffix with "_ap".
_PAL_SUFFIXES = ("_ap", "_ap_es", "_ap_hw", "_ap_xs")

# All known suffixes (priority order; longest first so "_ap_es" wins over "_ap").
_ALL_SUFFIXES = (
    _PAL_SUFFIXES
    + _EPISODE_SUFFIXES
    + _LOD_SUFFIXES
    + _STATE_SUFFIXES
)

# Hardcoded family table for well-known PSOBB monster pairs whose basenames
# don't follow the simple stem+suffix rule. Keyed by canonical "family stem"
# (stripped basename), each value is a list of (file basename, label,
# variant_kind) tuples. The file basename is matched as-is in data/.
#
# When a model's basename appears in any family entry, the detector returns
# the entire family (with self flagged).
_FAMILY_TABLE: dict[str, list[tuple[str, str, str]]] = {
    # Booma / Gobooma / Gigobooma — same engine class, just different
    # textures. Hardcoded because the suffix is just a numeric step.
    "bm_ene_bm1_shark": [
        ("bm_ene_bm1_shark.bml", "Booma", "color"),
        ("bm_ene_bm1_shark_a.bml", "Gobooma", "color"),
    ],
    # Lappy family — Episode skins (Easter / Halloween / Xmas).
    "bm_ene_lappy": [
        ("bm_ene_lappy.bml", "Rag Rappy", "color"),
        ("bm_ene_lappy_es.bml", "Rag Rappy (Easter)", "color"),
        ("bm_ene_lappy_hw.bml", "Rag Rappy (Halloween)", "color"),
        ("bm_ene_lappy_xs.bml", "Rag Rappy (Christmas)", "color"),
        ("bm_ene_lappy_es4.bml", "Rag Rappy (Easter Ep4)", "color"),
        ("bm_ene_lappy_hw4.bml", "Rag Rappy (Halloween Ep4)", "color"),
        ("bm_ene_lappy_xs4.bml", "Rag Rappy (Christmas Ep4)", "color"),
        ("bm_ene_lappy_ap.bml", "Pal Rappy", "color"),
        ("bm_ene_lappy_ap_es.bml", "Pal Rappy (Easter)", "color"),
        ("bm_ene_lappy_ap_hw.bml", "Pal Rappy (Halloween)", "color"),
        ("bm_ene_lappy_ap_xs.bml", "Pal Rappy (Christmas)", "color"),
        ("bm_ene_sandlappy.bml", "Sand Rappy", "color"),
    ],
    # Hildebear / Hildeblue — single BML in some PSOBB.IO drops, two in
    # others. Detector falls through to NJTL grouping when bm_ene_hild* is
    # missing the second file.
    "bm_ene_hildebear": [
        ("bm_ene_hildebear.bml", "Hildebear", "color"),
        ("bm_ene_hildeblue.bml", "Hildeblue", "color"),
    ],
    "bm_ene_de_rol_le": [
        ("bm_boss2_de_rol_le.bml", "De Rol Le (Ep1)", "color"),
        ("bm_boss2_de_rol_le_a.bml", "De Rol Le (state A)", "damaged"),
        ("bm_boss7_de_rol_le.bml", "De Rol Le (Ep4)", "color"),
        ("bm_boss7_de_rol_le_c.bml", "De Rol Le (Ep4 state C)", "damaged"),
    ],
    # Sinow Beat / Sinow Gold (Ep1 + Ep2) - paired.
    "bm_ene_sinow": [
        ("bm_ene_sinow.bml", "Sinow Beat", "color"),
        ("bm_ene_sinow_a.bml", "Sinow Gold", "color"),
    ],
    # Wolf / GalWolf - bm5 wolf family.
    "bm_ene_bm5_wolf": [
        ("bm_ene_bm5_wolf.bml", "Wolf", "color"),
        ("bm_ene_bm5_wolf_a.bml", "Gal Wolf", "color"),
    ],
}

# Friendly display-name override for family stems that aren't covered by
# _FAMILY_TABLE. Maps a regex on the basename's stem-without-suffix to a
# display family name. When no entry matches, we use the stem verbatim as
# display name + the suffix as the variant tag.
_DISPLAY_NAME_OVERRIDES = [
    (re.compile(r"^bm_ene_bm9_s_mericarol$", re.I), "Mericarol"),
    (re.compile(r"^bm_ene_bm5_gibon_u$", re.I),     "Sinow"),
    (re.compile(r"^bm_ene_re8_b_beast$", re.I),     "Berserker Beast"),
    (re.compile(r"^bm_ene_dubchik$", re.I),         "Dubchik"),
    (re.compile(r"^bm_ene_grass$", re.I),           "Grass Assassin"),
    (re.compile(r"^bm_ene_cgrass$", re.I),          "Crimson Assassin"),
    (re.compile(r"^bm_boss1_dragon$", re.I),        "Dragon"),
    (re.compile(r"^bm_boss2_de_rol_le$", re.I),     "De Rol Le"),
]

# Default per-suffix label generator (used when not in _FAMILY_TABLE and no
# _DISPLAY_NAME_OVERRIDES match).
_SUFFIX_DISPLAY = {
    "_a":      "alt state A",
    "_b":      "alt state B",
    "_c":      "alt state C",
    "_low":    "low-poly",
    "_high":   "high-poly",
    "_es":     "Easter",
    "_es4":    "Easter Ep4",
    "_hw":     "Halloween",
    "_hw4":    "Halloween Ep4",
    "_xs":     "Christmas",
    "_xs4":    "Christmas Ep4",
    "_ap":     "Pal",
    "_ap_es":  "Pal Easter",
    "_ap_hw":  "Pal Halloween",
    "_ap_xs":  "Pal Christmas",
}

# Default kind for each suffix family.
_SUFFIX_KIND = {
    "_a":      "damaged",
    "_b":      "damaged",
    "_c":      "damaged",
    "_low":    "lod",
    "_high":   "lod",
    "_es":     "color",
    "_es4":    "color",
    "_hw":     "color",
    "_hw4":    "color",
    "_xs":     "color",
    "_xs4":    "color",
    "_ap":     "color",
    "_ap_es":  "color",
    "_ap_hw":  "color",
    "_ap_xs":  "color",
}

# Fallback palette for variants that don't have a sampled XVR color.
# 8 swatches used round-robin by family index.
_FALLBACK_SWATCHES = [
    "#7faf42",  # green
    "#5b8df0",  # blue
    "#e54848",  # red
    "#e8c542",  # yellow
    "#a960e8",  # purple
    "#56c8c8",  # cyan
    "#e8893a",  # orange
    "#d96bb8",  # pink
]


def _split_basename_suffix(stem: str) -> tuple[str, str]:
    """Return (base_without_suffix, suffix_or_empty) using the longest match
    in ``_ALL_SUFFIXES``. ``stem`` should NOT include the ``.bml`` extension.

    Example::
        _split_basename_suffix("bm_ene_lappy_ap_es") -> ("bm_ene_lappy", "_ap_es")
        _split_basename_suffix("bm_ene_dubchik")     -> ("bm_ene_dubchik", "")
    """
    lo = stem.lower()
    # Sort suffixes longest-first so "_ap_es" wins over "_ap" and "_es".
    for suf in sorted(_ALL_SUFFIXES, key=len, reverse=True):
        if lo.endswith(suf) and len(lo) > len(suf):
            return stem[: -len(suf)], stem[-len(suf):]
    return stem, ""


def _display_name_for_stem(stem_no_suffix: str) -> str:
    """Pretty-print a stem (basename minus suffix). Uses
    ``_DISPLAY_NAME_OVERRIDES`` first, falls back to a sanitized stem.
    """
    for rx, name in _DISPLAY_NAME_OVERRIDES:
        if rx.match(stem_no_suffix):
            return name
    # Strip the bm_ene_ / bm_boss_ / bm_obj_ prefixes for readability.
    s = stem_no_suffix
    for pfx in ("bm_ene_", "bm_boss_", "bm_obj_", "bm_"):
        if s.lower().startswith(pfx):
            s = s[len(pfx):]
            break
    # Replace underscores with spaces; capitalize.
    s = s.replace("_", " ").strip()
    if s:
        s = s[0].upper() + s[1:]
    return s or "(unknown)"


def _label_for(basename: str, stem_no_suffix: str, suffix: str) -> str:
    """Compose the human display label for one variant entry."""
    family = _display_name_for_stem(stem_no_suffix)
    if suffix and suffix in _SUFFIX_DISPLAY:
        return f"{family} ({_SUFFIX_DISPLAY[suffix]})"
    if suffix:
        return f"{family} ({suffix.lstrip('_')})"
    return family


# ---------------------------------------------------------------------------
# Cross-BML variant discovery
# ---------------------------------------------------------------------------


def _scan_data_dir_for_siblings(
    data_dir: Path, stem_no_suffix: str, ext: str = ".bml"
) -> list[Path]:
    """Return all files in ``data_dir`` whose stem starts with
    ``stem_no_suffix`` followed by an empty string OR one of the known
    suffixes. ext-filtered (default .bml).
    """
    if not data_dir.exists() or not data_dir.is_dir():
        return []
    out: list[Path] = []
    lo_stem = stem_no_suffix.lower()
    for entry in sorted(data_dir.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() != ext.lower():
            continue
        e_stem = entry.stem.lower()
        # Exact match (no suffix) — same model, same file.
        if e_stem == lo_stem:
            out.append(entry)
            continue
        # Stem must start with prefix + "_" so we don't match
        # "bm_ene_lappy_ap" when looking for "bm_ene_lap".
        if not e_stem.startswith(lo_stem + "_"):
            continue
        # The trailing piece must be a recognised suffix.
        tail = e_stem[len(lo_stem):]
        if tail in _ALL_SUFFIXES:
            out.append(entry)
    return out


def _detect_cross_bml_variants(
    bml_path: Path, data_dir: Path
) -> list[VariantInfo]:
    """Find sibling BMLs that share a stem prefix with ``bml_path``.

    Walks the data dir for known-suffix permutations of the input's
    basename. Excludes the input itself from the result list.
    """
    self_stem = bml_path.stem
    self_no_suffix, self_suffix = _split_basename_suffix(self_stem)

    # First check the hardcoded family table — if the input's basename appears
    # in any family entry, return the entire family.
    self_basename = bml_path.name.lower()
    for fam_stem, fam_entries in _FAMILY_TABLE.items():
        if any(e[0].lower() == self_basename for e in fam_entries):
            out: list[VariantInfo] = []
            for i, (fname, label, kind) in enumerate(fam_entries):
                full = data_dir / fname
                if not full.exists():
                    continue
                out.append(VariantInfo(
                    path=fname,
                    label=label,
                    variant_kind=kind,
                    icon_color=_FALLBACK_SWATCHES[i % len(_FALLBACK_SWATCHES)],
                    is_self=(fname.lower() == self_basename),
                ))
            return out

    # Otherwise scan for stem siblings.
    siblings = _scan_data_dir_for_siblings(data_dir, self_no_suffix, ext=".bml")
    out: list[VariantInfo] = []
    for i, p in enumerate(siblings):
        s_stem = p.stem
        _, s_suffix = _split_basename_suffix(s_stem)
        kind = _SUFFIX_KIND.get(s_suffix, "color") if s_suffix else "color"
        label = _label_for(p.name, self_no_suffix, s_suffix)
        out.append(VariantInfo(
            path=p.name,
            label=label,
            variant_kind=kind,
            icon_color=_FALLBACK_SWATCHES[i % len(_FALLBACK_SWATCHES)],
            is_self=(p.name.lower() == self_basename),
        ))
    return out


# ---------------------------------------------------------------------------
# Intra-BML variant discovery (NJTL slot grouping)
# ---------------------------------------------------------------------------


def _njtl_slot_groups(slot_names: list[str]) -> list[list[int]]:
    """Detect contiguous groups of NJTL slots that share a base name with
    one of {"", "b", "c", "d", "r"} as a TRAILING suffix on the slot name.

    Algorithm: strip a single-letter trailing suffix (b/c/d/r) from each
    slot name; collect the order of "stripped names" seen; then walk the
    list and split into groups where each group has the same set of stripped
    names in order.

    Example NJTL list (Mericarol)::
        s256_bmmeri01,  s256_bmmeri02,  s256_bmmeri03,
        s256_bmmeri01b, s256_bmmeri02b, s256_bmmeri03b,
        s256_bmmeri01c, s256_bmmeri02c, s256_bmmeri03c

    Returns ``[[0,1,2], [3,4,5], [6,7,8]]`` (the 3 variant groups).

    Example NJTL list with no grouping (single-set)::
        s256_dragon01, s256_dragon02
    Returns ``[[0,1]]`` (one group covering all slots).
    """
    if not slot_names:
        return []

    # Strip a single trailing letter (b/c/d/r) from each name; remember the
    # base for grouping.
    bases: list[str] = []
    suffixes: list[str] = []
    for n in slot_names:
        s = n.strip()
        # Strip a single trailing letter when it follows a digit:
        # "s256_bmmeri01b" -> base "s256_bmmeri01", suffix "b"
        # "s256_bmmeri01"  -> base "s256_bmmeri01", suffix ""
        # We restrict the trailing letter to {b,c,d,r} so we don't
        # accidentally bin texture sets that happen to end in "a".
        if (
            len(s) >= 2
            and s[-1].lower() in {"b", "c", "d", "r"}
            and s[-2].isdigit()
        ):
            bases.append(s[:-1])
            suffixes.append(s[-1].lower())
        else:
            bases.append(s)
            suffixes.append("")

    # Identify the "first group" — the contiguous prefix of slots whose
    # suffixes are empty. If any slot has a non-empty suffix, the slot count
    # of the first group equals the number of empty-suffix slots seen.
    first_group_len = 0
    for sfx in suffixes:
        if sfx == "":
            first_group_len += 1
        else:
            break
    if first_group_len == 0:
        # No empty-suffix prefix — no detectable variant grouping.
        return [list(range(len(slot_names)))]

    # If every slot has an empty suffix, all slots = single group.
    if first_group_len == len(slot_names):
        return [list(range(len(slot_names)))]

    # Verify the remainder is divisible into groups of first_group_len whose
    # stripped names match the first group's stripped names in order.
    n = len(slot_names)
    if n % first_group_len != 0:
        # Mismatched layout — bail to single group.
        return [list(range(n))]

    first_group_bases = bases[:first_group_len]
    groups: list[list[int]] = [list(range(first_group_len))]
    cursor = first_group_len
    while cursor < n:
        chunk_bases = bases[cursor : cursor + first_group_len]
        if chunk_bases != first_group_bases:
            # Doesn't line up — fall back to single group.
            return [list(range(n))]
        groups.append(list(range(cursor, cursor + first_group_len)))
        cursor += first_group_len

    return groups


def _intra_bml_variant_label(suffix_marker: str, idx: int) -> tuple[str, str]:
    """Map a single-letter suffix marker (and group index) to a
    (label, variant_kind) pair.

    Mericarol uses ``""`` / ``"b"`` / ``"c"`` for green/blue/red. Other
    PSOBB enemies use the same idiom: 1st group is the "default" color,
    2nd is "blue/lighter", 3rd is "red/darker".
    """
    if idx == 0:
        return ("default", "color")
    if suffix_marker == "b":
        return ("blue variant", "color")
    if suffix_marker == "c":
        return ("red variant", "color")
    if suffix_marker == "d":
        return ("dark variant", "color")
    if suffix_marker == "r":
        return ("rare variant", "color")
    return (f"variant {idx}", "color")


# Per-family override for intra-BML variant LABELS. When the BML stem
# (without _suffix) matches the key, the value is a tuple of explicit
# variant labels indexed by ``slot_group_idx``.
#
# Source: PSOBB enemy nomenclature where the same NJTL trick packs 3
# canonical color variants into one BML. Mericarol/Mericus/Merikle is
# the textbook case.
_INTRA_BML_LABEL_OVERRIDES: dict[str, list[str]] = {
    "bm_ene_bm9_s_mericarol": ["Mericarol", "Mericus", "Merikle"],
    # Add more 3-color enemies as discovered. Hildebear/Hildeblue are
    # cross-BML in PSOBB.IO so they live in _FAMILY_TABLE; intra-BML
    # 3-color packs are the exception.
}


def _detect_intra_bml_variants(
    bml_path: Path, *, max_inner_bytes: int = 8 * 1024 * 1024
) -> list[VariantInfo]:
    """Inspect a BML's first .nj inner; if its NJTL splits into ≥2 equal
    groups by single-letter trailing suffix, surface each group as a
    variant. Returns empty list if the BML has no NJTL or the slots don't
    group cleanly.

    ``max_inner_bytes`` caps how much of the NJ we'll PRS-decompress to
    bound memory in case of pathological inputs.
    """
    try:
        blob = bml_path.read_bytes()
    except OSError:
        return []

    try:
        entries = parse_bml(blob)
    except ValueError:
        return []

    nj_entry = next((e for e in entries if e.name.lower().endswith(".nj")), None)
    if nj_entry is None:
        return []

    if nj_entry.size_compressed > max_inner_bytes:
        return []

    try:
        raw = bytes(blob[nj_entry.offset : nj_entry.offset + nj_entry.size_compressed])
        nj_bytes = _prs_decompress(raw)
    except (RuntimeError, ValueError):
        return []

    try:
        njtl = find_and_parse_njtl(nj_bytes) or []
    except ValueError:
        return []

    if len(njtl) < 4:
        # Need at least a 2-group split (≥2 slots per group, ≥2 groups).
        return []

    slot_names = [e.name for e in njtl]
    groups = _njtl_slot_groups(slot_names)
    if len(groups) < 2:
        return []

    # Build VariantInfo for each group. The "self" entry is group 0 (the
    # default) — we use a fragment marker so the frontend can request the
    # same model with a different slot offset.
    out: list[VariantInfo] = []
    stem_no_suffix = _split_basename_suffix(bml_path.stem)[0]
    family_name = _display_name_for_stem(stem_no_suffix)
    inner_name = nj_entry.name

    # Look up an explicit per-family label list (Mericarol/Mericus/Merikle).
    label_override = _INTRA_BML_LABEL_OVERRIDES.get(stem_no_suffix.lower())

    for gi, slot_indices in enumerate(groups):
        # Pick the suffix marker by sampling the LAST char of the FIRST slot
        # name in the group (assumed-stable).
        marker = ""
        if slot_indices:
            n = slot_names[slot_indices[0]]
            if len(n) >= 2 and n[-2].isdigit() and n[-1].isalpha():
                marker = n[-1].lower()

        if label_override and gi < len(label_override):
            label = label_override[gi]
            kind = "color"
        else:
            label_marker, kind = _intra_bml_variant_label(marker, gi)
            label = f"{family_name} ({label_marker})" if gi > 0 else f"{family_name}"

        # We encode the slot offset and stride into the path via a custom
        # query-string: the frontend re-fetches the model with the offset
        # and re-binds material_id → tile_index = material_id + offset.
        slot_count = len(slot_indices)
        slot_group_idx = gi  # 0/1/2 for green/blue/red etc.

        path = bml_path.name + "#" + inner_name
        out.append(VariantInfo(
            path=path,
            label=label,
            variant_kind=kind,
            icon_color=_FALLBACK_SWATCHES[gi % len(_FALLBACK_SWATCHES)],
            slot_group=slot_group_idx,
            slot_count=slot_count,
            is_self=(gi == 0),  # default-group is "self"
        ))
    return out


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def detect_variants(
    bml_path: Path,
    *,
    data_dir: Optional[Path] = None,
    include_self: bool = True,
) -> list[VariantInfo]:
    """Return the list of color/state variants the frontend should offer
    for the model at ``bml_path``.

    Detection runs in this order:
      1. Cross-BML siblings via the hardcoded family table OR stem+suffix
         scan in ``data_dir``.
      2. Intra-BML NJTL slot grouping (Mericarol-style 3-color packing).

    If both succeed, both lists are returned (cross-BML first). If only
    one matches, only that list is returned. Empty list = "no variants
    detected; show no picker".

    ``include_self``: when False, drop the entry that is_self==True from
    the returned list. Default True (frontend wants to show all so the
    current model is highlighted in the picker).
    """
    if data_dir is None:
        data_dir = bml_path.parent

    out: list[VariantInfo] = []

    # 1. Cross-BML variants.
    cross = _detect_cross_bml_variants(bml_path, data_dir)
    out.extend(cross)

    # 2. Intra-BML variants. We do this REGARDLESS of cross results — a
    # model might have BOTH (e.g. a low-poly sibling AND in-BML color
    # variants).
    intra = _detect_intra_bml_variants(bml_path)
    out.extend(intra)

    if not include_self:
        out = [v for v in out if not v.is_self]

    return out


__all__ = ["VariantInfo", "detect_variants"]
