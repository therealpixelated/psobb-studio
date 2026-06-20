"""Texture <-> model multi-rule matcher.

Phase A delivers this as a manifest annotation; Phase E later adds the UI.

Six rules, ordered by confidence (highest first):

  R1: ``<basename>_tex.xvm`` sibling of a ``.bml``               -> 1.0
  R2: BML-internal XVM (extracted via ``formats.bml``)            -> 0.95
  R3: NJTL chunk lookup (parsed via ``formats.iff``)              -> 0.9
  R4: AFS player-tex (``pl[A-X]bdy00.nj`` -> ``pl<class>tex.afs``) -> 0.85
  R5: ItemModel/ItemTexture pair by ordinal                       -> 0.7
  R6: ``map_*.rel`` -> ``.xvm`` siblings (and the inverse)        -> 0.5

Hard dependencies on Agent 2 (``formats.iff`` / ``formats.afs``) and Agent 4
(``formats.bml``) are wrapped in try/except so this module imports cleanly
whether or not those siblings have shipped. When a dependency is missing,
the corresponding rule emits a placeholder ``Match`` whose ``partial`` flag
is true and whose ``rule`` string is suffixed ``-stub`` so callers can
distinguish a real hit from a graceful degrade.

Spec: ``MASTER_PLAN/02_asset_format_atlas.md`` section "Texture <-> model
mapping rules" + ``MASTER_PLAN/manifest.schema.json`` ``matched_textures``.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# --------------------------------------------------------------------------
# Optional sibling-module imports. Each is feature-detected at import time.
# A missing module falls through to a stub path that produces a Match with
# rule="<R*>-stub", confidence=<rule_default>, partial=True so downstream
# manifest serialization can flag the entry as "needs follow-up".
# --------------------------------------------------------------------------

try:  # Agent 2 deliverable
    from . import iff as _iff  # type: ignore
    _HAS_IFF = hasattr(_iff, "parse_iff")
except ImportError:  # pragma: no cover
    _iff = None  # type: ignore[assignment]
    _HAS_IFF = False

try:  # Agent 2 deliverable
    from . import afs as _afs  # type: ignore
    _HAS_AFS = hasattr(_afs, "parse_afs")
except ImportError:  # pragma: no cover
    _afs = None  # type: ignore[assignment]
    _HAS_AFS = False

try:  # Agent 4 deliverable
    from . import bml as _bml  # type: ignore
    _HAS_BML = hasattr(_bml, "parse_bml") or hasattr(_bml, "extract_bml")
except ImportError:  # pragma: no cover
    _bml = None  # type: ignore[assignment]
    _HAS_BML = False


# --------------------------------------------------------------------------
# Public types
# --------------------------------------------------------------------------

# Mirrors the BACKUP_FRAGMENTS list in server.py - keep in sync.
_BACKUP_FRAGMENTS = (".pre_", ".suspect_", ".parked_", ".bad_", ".disabled")


@dataclass
class Match:
    """A single texture candidate for a model file.

    Attributes
    ----------
    path:
        Absolute path to the texture asset (or to the model itself for stubs
        whose dependency is unavailable).
    rule:
        Short rule identifier - one of "R1".."R6", optionally suffixed
        ``-stub`` if the rule degraded.
    confidence:
        Float in ``[0.0, 1.0]``. The confidence of the parent rule -
        unaffected by stub degradation so callers can still prioritize.
    partial:
        ``True`` when this match is a placeholder because the underlying
        format reader (BML / IFF / AFS) wasn't available at runtime.
    detail:
        Free-form rule-specific payload (e.g. ``{"ordinal": 12}`` for R5).
        Not serialized into the manifest by default.
    """

    path: Path
    rule: str
    confidence: float
    partial: bool = False
    detail: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Backup / sibling filtering
# --------------------------------------------------------------------------

def _is_backup_path(p: Path) -> bool:
    """True when ``p`` is a ``.pre_*`` / ``.SUSPECT_*`` / ``.BAD_*`` sibling."""
    name_lower = p.name.lower()
    if name_lower.startswith("pre_"):
        return True
    for frag in _BACKUP_FRAGMENTS:
        if frag in name_lower:
            return True
    return False


def _exists_clean(p: Path) -> bool:
    """``p`` must exist and not be a backup variant."""
    return p.exists() and not _is_backup_path(p)


# --------------------------------------------------------------------------
# Rule implementations
# --------------------------------------------------------------------------

def _rule_r1_tex_xvm_sibling(model_path: Path) -> list[Match]:
    """R1: ``<basename>_tex.xvm`` sibling - confidence 1.0.

    Examples:
      ``bm_obj_ep4_boss09_core.bml`` -> ``bm_obj_ep4_boss09_core_tex.xvm``
    """
    sibling = model_path.parent / f"{model_path.stem}_tex.xvm"
    if _exists_clean(sibling):
        return [Match(path=sibling, rule="R1", confidence=1.0)]
    return []


def _rule_r2_bml_internal(model_path: Path) -> list[Match]:
    """R2: BML-internal XVM - confidence 0.95.

    Walks the BML file table and returns a Match per inline XVM entry.
    Falls back to a header-scan if ``formats.bml`` isn't available - we can
    still detect *whether* there's an inline texture cheaply (the BML header
    byte at offset 9 is the ``has_textures`` flag) so the stub is honest.
    """
    if model_path.suffix.lower() != ".bml":
        return []

    if _HAS_BML and _bml is not None:
        try:
            buf = model_path.read_bytes()
        except OSError:
            return []
        try:
            entries = _bml.parse_bml(buf)  # type: ignore[union-attr]
        except Exception:
            # Malformed BML - degrade rather than blow up the whole walk.
            return [Match(path=model_path, rule="R2-stub", confidence=0.95, partial=True,
                          detail={"reason": "parse_bml raised"})]
        out: list[Match] = []
        for e in entries:
            # An entry contributes an R2 match only if it carries an inline
            # texture archive. The accessor name varies between sketches
            # (Agent 4's report will pin it down) - tolerate either.
            has_tex = (
                getattr(e, "has_texture", False)
                or getattr(e, "has_textures", False)
                or getattr(e, "tex_size_compressed", 0) > 0
            )
            if has_tex:
                ent_name = getattr(e, "name", "")
                # Synthesize a virtual path "<bml>::<entry>.xvm" - the
                # extracted bytes live inside the archive, so consumers must
                # call extract_bml() to materialize them. Using the archive
                # path with a fragment keeps manifest entries unique.
                virt = model_path.with_name(f"{model_path.name}#{ent_name}.xvm")
                out.append(Match(path=virt, rule="R2", confidence=0.95,
                                 detail={"bml_entry": ent_name}))
        return out

    # Fallback: cheap header sniff. If the BML header says has_textures=1 we
    # emit a single stub so callers know the rule fired even though we can't
    # name the inner XVMs. If has_textures=0 we emit nothing - the rule
    # simply doesn't apply.
    try:
        with open(model_path, "rb") as f:
            head = f.read(16)
    except OSError:
        return []
    if len(head) < 16:
        return []
    has_tex_flag = head[9]
    if has_tex_flag:
        return [Match(path=model_path, rule="R2-stub", confidence=0.95, partial=True,
                      detail={"reason": "formats.bml not loaded; has_textures=1"})]
    return []


def _read_iff_chunks_fallback(buf: bytes) -> list[tuple[str, bytes]]:
    """Minimal IFF walker used when ``formats.iff`` isn't loaded.

    Layout (PSO LE flavor): repeated ``<4-char type><u32 le size><payload>``,
    aligned to 4 bytes between chunks. We bail on the first malformed chunk.
    """
    out: list[tuple[str, bytes]] = []
    off = 0
    n = len(buf)
    safety = 0
    while off + 8 <= n and safety < 64:
        safety += 1
        try:
            t = buf[off:off + 4].decode("ascii")
        except UnicodeDecodeError:
            break
        if not all(32 <= b < 127 for b in buf[off:off + 4]):
            break
        sz = struct.unpack_from("<I", buf, off + 4)[0]
        payload_start = off + 8
        payload_end = payload_start + sz
        if payload_end > n:
            break
        out.append((t, buf[payload_start:payload_end]))
        # IFF aligns to 4
        next_off = payload_end
        if next_off % 4:
            next_off += 4 - (next_off % 4)
        if next_off <= off:  # defensive against zero-size loops
            break
        off = next_off
    return out


def _extract_njtl_names(njtl_payload: bytes) -> list[str]:
    """Pull texture names out of an NJTL chunk payload.

    NJTL layout (per Phantasmal World ``Xj.kt`` / pso-blender ``njtl.py``):
      u32 elements_ptr, u32 count
      array of TextureListEntry { u32 name_ptr, u32 unk, u32 data }
      strings are 4-byte aligned, NUL-terminated ASCII

    File pointers in NJ chunks are RELATIVE to the chunk body. Since the
    POF0 fixup hasn't been applied here, we treat the pointers as offsets
    inside the payload buffer.
    """
    if len(njtl_payload) < 8:
        return []
    elements_off, count = struct.unpack_from("<II", njtl_payload, 0)
    if count == 0 or count > 256:  # sanity cap
        return []
    if elements_off + count * 12 > len(njtl_payload):
        return []
    names: list[str] = []
    for i in range(count):
        entry_off = elements_off + i * 12
        if entry_off + 12 > len(njtl_payload):
            break
        name_ptr = struct.unpack_from("<I", njtl_payload, entry_off)[0]
        if name_ptr == 0 or name_ptr >= len(njtl_payload):
            continue
        # Read a NUL-terminated ASCII string starting at name_ptr.
        end = njtl_payload.find(b"\x00", name_ptr)
        if end < 0 or end - name_ptr > 64:
            continue
        try:
            names.append(njtl_payload[name_ptr:end].decode("ascii"))
        except UnicodeDecodeError:
            continue
    return names


def _rule_r3_njtl_lookup(model_path: Path) -> list[Match]:
    """R3: NJTL chunk lookup - confidence 0.9.

    Reads the ``.nj`` file's IFF chunks, pulls the texture names out of any
    ``NJTL`` chunk, and probes the model's parent dir for ``<name>.xvm``
    or ``<name>.prs`` siblings. If ``formats.iff`` is loaded we use it;
    otherwise we fall back to a tiny inline IFF walker.
    """
    if model_path.suffix.lower() != ".nj":
        return []

    try:
        buf = model_path.read_bytes()
    except OSError:
        return []

    chunks: list[tuple[str, bytes]]
    if _HAS_IFF and _iff is not None:
        try:
            parsed = _iff.parse_iff(buf)  # type: ignore[union-attr]
        except Exception:
            return [Match(path=model_path, rule="R3-stub", confidence=0.9, partial=True,
                          detail={"reason": "parse_iff raised"})]
        chunks = []
        for c in parsed:
            t = getattr(c, "type", None) or getattr(c, "type_name", None)
            data = getattr(c, "data", None) or getattr(c, "payload", None)
            if t and data is not None:
                chunks.append((t, data))
    else:
        chunks = _read_iff_chunks_fallback(buf)

    njtl_payloads = [d for (t, d) in chunks if t == "NJTL"]
    if not njtl_payloads:
        return []

    names: list[str] = []
    for payload in njtl_payloads:
        names.extend(_extract_njtl_names(payload))

    if not names:
        # NJTL existed but we couldn't name its entries. Still useful signal -
        # tell the manifest we tried.
        return [Match(path=model_path, rule="R3-stub", confidence=0.9, partial=True,
                      detail={"reason": "NJTL present but no parseable names"})]

    out: list[Match] = []
    seen: set[Path] = set()
    parent = model_path.parent
    for n in names:
        for ext in (".xvm", ".prs"):
            candidate = parent / f"{n}{ext}"
            if candidate in seen:
                continue
            if _exists_clean(candidate):
                seen.add(candidate)
                out.append(Match(path=candidate, rule="R3", confidence=0.9,
                                 detail={"njtl_name": n}))
    return out


_PL_BDY_RE = re.compile(r"^pl([A-Z])bdy(\d{2})\.nj$", re.IGNORECASE)


def _rule_r4_player_afs(model_path: Path) -> list[Match]:
    """R4: AFS player-tex - confidence 0.85.

    ``pl[A-X]bdy00.nj`` -> ``pl<class>tex.afs`` (per character class).

    The spec says ``pl[A-X]bdy00.nj`` but the live install has bodies for
    other ordinals too (``plAbdy01.nj`` etc) - they all reference the same
    per-class AFS. We accept any two-digit body suffix.
    """
    m = _PL_BDY_RE.match(model_path.name)
    if not m:
        return []
    char_class = m.group(1).upper()
    if char_class > "X":  # only A..X are valid char-class letters
        return []
    afs = model_path.parent / f"pl{char_class}tex.afs"
    if _exists_clean(afs):
        return [Match(path=afs, rule="R4", confidence=0.85,
                      detail={"char_class": char_class, "body_index": m.group(2)})]
    return []


# Item AFS pair table. ``ItemModel.afs`` <-> ``ItemTexture.afs`` is the base
# pair; the Episode 4 expansion adds an ``Ep4`` suffix on both sides.
_ITEM_AFS_PAIRS = [
    ("ItemModel.afs", "ItemTexture.afs"),
    ("ItemModelEp4.afs", "ItemTextureEp4.afs"),
]


def _rule_r5_item_afs(model_path: Path) -> list[Match]:
    """R5: ItemModel/ItemTexture pair by ordinal - confidence 0.7.

    The matcher returns the *paired AFS file* as a match; the per-ordinal
    drill-down (which slot inside the texture AFS goes with which slot
    inside the model AFS) is left for downstream consumers - they get the
    pair pointer plus an ``ordinal`` detail field set to ``None`` to signal
    "all ordinals" until ``formats.afs`` lands.
    """
    name = model_path.name
    parent = model_path.parent
    out: list[Match] = []
    for model_afs, tex_afs in _ITEM_AFS_PAIRS:
        if name.lower() == model_afs.lower():
            candidate = parent / tex_afs
            if _exists_clean(candidate):
                out.append(Match(path=candidate, rule="R5", confidence=0.7,
                                 detail={"pair": (model_afs, tex_afs), "ordinal": None}))
            break
    return out


_MAP_AREA_RE = re.compile(r"^(map_[a-z]+\d+)(?:_(\d+)([a-z]*))?$", re.IGNORECASE)


def _map_area_key(stem: str) -> str | None:
    """Reduce a ``map_*`` stem to its area-level prefix.

    ``map_aancient01_00s`` -> ``map_aancient01``
    ``map_aancient01_00n`` -> ``map_aancient01``
    ``map_aancient01``     -> ``map_aancient01``

    Returns ``None`` for non-conforming names (callers fall back to the
    full stem so we don't over-match).
    """
    m = _MAP_AREA_RE.match(stem)
    if not m:
        return None
    return m.group(1)


def _rule_r6_map_prefix(model_path: Path) -> list[Match]:
    """R6: ``map_*.rel`` -> ``.xvm`` siblings (and the inverse) - confidence 0.5.

    PSOBB scene assets use a layered naming scheme:

      ``map_<area><nn>``           area + sub-area number
      ``map_<area><nn>_<mm><kind>`` area + sub-area + room + kind suffix

    Where ``kind`` is ``c`` (collision), ``n`` (node mesh), ``r`` (render)
    for ``.rel`` and ``s`` (surface) for ``.xvm``. Pairings inside an area
    are dense, so we return every ``.xvm`` whose area-key matches the
    requesting ``.rel`` (and vice versa).
    """
    name = model_path.name
    if not name.lower().startswith("map_"):
        return []

    out: list[Match] = []
    parent = model_path.parent
    stem = model_path.stem
    suffix = model_path.suffix.lower()
    area = _map_area_key(stem) or stem

    if suffix == ".rel":
        # Find every map_<area>*.xvm sibling - unrestricted by sub-area to
        # account for shared-environment tilesets.
        try:
            candidates = sorted(parent.glob(f"{area}*.xvm"))
        except OSError:
            candidates = []
        for c in candidates:
            if _exists_clean(c) and c != model_path:
                out.append(Match(path=c, rule="R6", confidence=0.5))
    elif suffix == ".xvm":
        # Find every map_<area>*.rel sibling.
        try:
            candidates = sorted(parent.glob(f"{area}*.rel"))
        except OSError:
            candidates = []
        for c in candidates:
            if _exists_clean(c) and c != model_path:
                out.append(Match(path=c, rule="R6", confidence=0.5))
    return out


# --------------------------------------------------------------------------
# Top-level entry point
# --------------------------------------------------------------------------

# Each rule fn signature: (model_path: Path) -> list[Match]
_RULES: tuple[Callable[[Path], list[Match]], ...] = (
    _rule_r1_tex_xvm_sibling,
    _rule_r2_bml_internal,
    _rule_r3_njtl_lookup,
    _rule_r4_player_afs,
    _rule_r5_item_afs,
    _rule_r6_map_prefix,
)


def match_textures(model_path: Path, install_root: Path) -> list[Match]:
    """Return all texture candidates for ``model_path``, ranked by confidence.

    Parameters
    ----------
    model_path:
        Absolute path to a model-side asset (``.bml``, ``.nj``, ``.afs``,
        ``.rel``, or ``.xvm`` for the R6 inverse). Backup-named files are
        ignored.
    install_root:
        Absolute path to the PSOBB install (parent of ``data/``). Currently
        only used to anchor relative-path serialization and for future
        cross-directory rules; the implementation is happy with any value
        as long as ``model_path`` exists.

    Returns
    -------
    list[Match]
        Ordered descending by ``confidence``. Within a confidence tier the
        original rule order (R1 before R2 before R3 ...) is preserved.
        Duplicates - same ``path`` matched by multiple rules - are coalesced
        to the highest-confidence rule.
    """
    if not isinstance(model_path, Path):
        model_path = Path(model_path)
    if _is_backup_path(model_path):
        return []

    # ``install_root`` is currently advisory; reserved for future
    # cross-directory rules (e.g. peeking at ``<root>/index/...``).
    _ = install_root

    matches: list[Match] = []
    for rule_fn in _RULES:
        try:
            matches.extend(rule_fn(model_path))
        except Exception:
            # A misbehaving rule must not poison the others. Swallow and
            # carry on - the manifest can call us again once formats.bml
            # etc. ship a fixed implementation.
            continue

    # Coalesce duplicates by path, keeping the highest confidence.
    by_path: dict[Path, Match] = {}
    for m in matches:
        prev = by_path.get(m.path)
        if prev is None or m.confidence > prev.confidence:
            by_path[m.path] = m

    # Sort: confidence desc, then by rule label (so R1 < R2 < ... < R6 wins
    # among equal confidences thanks to ascending rule order).
    return sorted(by_path.values(), key=lambda x: (-x.confidence, x.rule))


# --------------------------------------------------------------------------
# Manifest integration helper - Agent 1 (or whoever owns ``manifest.py``)
# can call this to project Match objects into the schema's
# ``matched_textures`` shape. We keep the projection here so the matcher's
# domain model stays cohesive.
# --------------------------------------------------------------------------

def matches_to_manifest_field(matches: list[Match], install_root: Path) -> list[dict]:
    """Project ``Match`` objects into ``matched_textures`` schema rows.

    The schema requires ``{path, rule, confidence}`` per row, with ``path``
    expressed as a forward-slash relative path under ``install_root``. We
    silently fall back to the absolute path for matches whose ``path`` lies
    outside the install (shouldn't happen in practice, but a malformed
    fixture shouldn't crash the manifest build).
    """
    rows: list[dict] = []
    try:
        root_resolved = install_root.resolve()
    except OSError:
        root_resolved = install_root
    for m in matches:
        try:
            rel = m.path.resolve().relative_to(root_resolved).as_posix()
        except (OSError, ValueError):
            rel = m.path.as_posix()
        # Strip any "-stub" suffix from the rule for schema compliance -
        # the schema enum only allows R1..R6. Stub origin is preserved
        # via the partial flag at the Match layer; downstream JSON loses
        # that distinction, which is acceptable for v1.
        rule_clean = m.rule.split("-", 1)[0]
        rows.append({
            "path": rel,
            "rule": rule_clean,
            "confidence": float(m.confidence),
        })
    return rows


__all__ = [
    "Match",
    "match_textures",
    "matches_to_manifest_field",
]
