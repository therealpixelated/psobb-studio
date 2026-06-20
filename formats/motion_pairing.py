"""Tier-ranked NJM motion resolver for PSOBB.IO models.

Background
----------

Every PSOBB BML that animates packs its motions next to its model under
the convention ``<verb>_<modelstem>.njm``. The model stem is the inner
``.nj`` file's basename, NOT the BML's basename — e.g.
``bm_boss1_dragon.bml`` ships ``boss1_s_nb_dragon.nj`` plus 26
``*_boss1_s_nb_dragon.njm`` motions.

Server's existing :func:`server._resolve_motion_sources` already
discovers every inner ``.njm`` in a BML and falls back to
``NpcApcMot.bml`` for player / pioneer NPC bodies. That works perfectly
when the BML has motions for **one** model stem. It falls down on the
67 multi-form BMLs (Pan Arms, De Rol Le, Vol Opt, Pouilly Slime, etc.)
because the auto-play pick is action-only — opening
``bm4_ps_ma_body.bml?inner=bm4_ps_ma_body.nj`` (43-bone body) causes
the picker to choose ``move_bm4_ps_mb_body.njm`` (1-bone single-arm
form) and the body snaps to bind pose because the bone counts disagree.

This module fixes that by ranking motion candidates with **stem
affinity** in addition to action priority:

  * Tier 1 — engine-table override (reserved; no hints declared today)
  * Tier 2 — same-BML siblings whose post-verb stem matches the loaded
            inner-model stem (e.g. ``walk_boss1_s_nb_dragon`` matches
            ``boss1_s_nb_dragon.nj``)
  * Tier 3 — same-BML siblings with any other stem (so multi-form BMLs
            still expose every motion, just deprioritised)
  * Tier 4 — ``NpcApcMot.bml`` fallback for ``pl*`` / ``bm_n_*`` /
            ``bm_npc_*`` host BMLs that ship without inline motions.

Within each tier, candidates are ordered by an **action priority** —
``walk > idle > attack > die > hit > spawn > fly > despawn > unknown``
— so the frontend's existing auto-play (``populateAnimationPanel`` →
``loadMotion(default_name)``) lands on the locomotion track first.

Empirical inventory backing the tiers lives in
``_reports/motion_inventory.md``.

Public surface
--------------

* :class:`MotionRef` — one resolved motion candidate.
* :func:`extract_action_hint` — verb-prefix → action-class classifier.
* :func:`resolve_motions_for_model` — top-level entrypoint.

The resolver is read-only (never mutates BMLs) and side-effect-free
beyond ``read_bytes`` calls — server.py wraps it in the same memoizing
caches that already serve ``/api/animations``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# We deliberately do NOT import server-side caches here — keeping the
# module pure makes it trivially unit-testable and reusable from the
# offline manifest builder.
from formats.bml import parse_bml


# ---------------------------------------------------------------------------
# Action-hint extraction.
#
# Verb-prefix mapping derived from the 1623-entry data inventory.
# The order matters when multiple keywords could match: more specific
# verbs (e.g. ``cstand``, ``deadb``) MUST come before their substrings
# (e.g. ``stand``, ``dead``).
# ---------------------------------------------------------------------------
ACTION_WALK = "walk"
ACTION_IDLE = "idle"
ACTION_ATTACK = "attack"
ACTION_DIE = "die"
ACTION_HIT = "hit"
ACTION_SPAWN = "spawn"
ACTION_FLY = "fly"
ACTION_DESPAWN = "despawn"
ACTION_UNKNOWN = "unknown"

# Lower case, exact verb-prefix match (strict). The PSOBB asset team
# was consistent enough that exact keys cover ~85 % of motions; the
# remainder fall through as "unknown" which is fine — the frontend
# treats unknown as last-priority and the user can still pick them.
ACTION_VERBS: dict[str, str] = {
    # walk / run / move (dominant Tier-2 hits)
    "walk": ACTION_WALK,
    "wlk": ACTION_WALK,
    "run": ACTION_WALK,
    "move": ACTION_WALK,
    "dash": ACTION_WALK,
    "walkl": ACTION_WALK,
    "walkr": ACTION_WALK,
    "wgwalk": ACTION_WALK,
    # idle / wait / stand
    "idle": ACTION_IDLE,
    "wait": ACTION_IDLE,
    "wait2": ACTION_IDLE,
    "stand": ACTION_IDLE,
    "cstand": ACTION_IDLE,
    "await": ACTION_IDLE,
    "land": ACTION_IDLE,
    "odori": ACTION_IDLE,
    "joy": ACTION_IDLE,
    "laugh": ACTION_IDLE,
    "cry": ACTION_IDLE,
    # attack
    "attack": ACTION_ATTACK,
    "atack": ACTION_ATTACK,
    "atk": ACTION_ATTACK,
    "tlatk": ACTION_ATTACK,
    "tatk": ACTION_ATTACK,
    "fire": ACTION_ATTACK,
    "beam": ACTION_ATTACK,
    "laser": ACTION_ATTACK,
    "shoot": ACTION_ATTACK,
    "bite": ACTION_ATTACK,
    "hoe": ACTION_ATTACK,
    "balattack": ACTION_ATTACK,
    "kiri": ACTION_ATTACK,
    "tukomi": ACTION_ATTACK,
    "balshout": ACTION_ATTACK,
    # die
    "die": ACTION_DIE,
    "dead": ACTION_DIE,
    "death": ACTION_DIE,
    "deada": ACTION_DIE,
    "deadb": ACTION_DIE,
    "deadg": ACTION_DIE,
    "deadl": ACTION_DIE,
    "deadr": ACTION_DIE,
    "deads": ACTION_DIE,
    "baldie": ACTION_DIE,
    # hit
    "hit": ACTION_HIT,
    "damage": ACTION_HIT,
    "damag": ACTION_HIT,
    "daml": ACTION_HIT,
    "dams": ACTION_HIT,
    "damfly": ACTION_HIT,
    "damgrd": ACTION_HIT,
    "tumble": ACTION_HIT,
    "down": ACTION_HIT,
    "lift": ACTION_HIT,
    "baldamage": ACTION_HIT,
    # spawn / appear
    "wake": ACTION_SPAWN,
    "wakeup": ACTION_SPAWN,
    "wake2": ACTION_SPAWN,
    "appear": ACTION_SPAWN,
    "apear": ACTION_SPAWN,
    "apper": ACTION_SPAWN,
    "enter": ACTION_SPAWN,
    "start": ACTION_SPAWN,
    "tobidasi": ACTION_SPAWN,
    "advent": ACTION_SPAWN,
    # despawn
    "kie": ACTION_DESPAWN,
    "exit": ACTION_DESPAWN,
    # locomotion variants — flying / swimming
    "fly": ACTION_FLY,
    "flyshot": ACTION_FLY,
    "frin": ACTION_FLY,
    "frout": ACTION_FLY,
    "frloop": ACTION_FLY,
    "swim": ACTION_FLY,
}

# Pre-sorted longest-first list for the sub-string fallback in
# :func:`extract_action_hint`. Iterating longest-first means
# ``balwait`` matches ``"wait"`` rather than the empty match, and
# ``cldamage`` matches ``"damage"`` (6 chars) before ``"dam"`` (which
# isn't a key anyway, but the sort discipline keeps the lookup
# predictable). Cached at module import for hot-path reuse.
_ACTION_VERBS_BY_LEN_DESC: tuple[str, ...] = tuple(sorted(
    ACTION_VERBS.keys(), key=lambda k: -len(k),
))


# Priority ranking for default-pick. Lower number = higher priority.
ACTION_PRIORITY: dict[str, int] = {
    ACTION_WALK:    0,
    ACTION_IDLE:    1,
    ACTION_ATTACK:  2,
    ACTION_DIE:     3,
    ACTION_HIT:     4,
    ACTION_SPAWN:   5,
    ACTION_FLY:     6,
    ACTION_DESPAWN: 7,
    ACTION_UNKNOWN: 8,
}


# Within an action class, some verbs are strictly preferable as the
# auto-play target. ``wait``-style poses are the canonical idle; mood
# variants like ``laugh``/``joy``/``cry``/``odori`` are situational
# and look weird as a default. Lower number = preferred.
#
# Verbs not listed sort at 50 — falls AFTER the curated favourites but
# BEFORE the "unknown" action (whose own priority is 8 at the action
# tier). This three-level ordering keeps the autoplay regression's
# Mericarol case picking ``wait_*`` over ``laugh_*`` even though both
# classify as idle, while still ranking either above any attack/die/hit.
_VERB_SUB_PRIORITY: dict[str, int] = {
    # idle favourites
    "wait": 0, "stand": 1, "cstand": 2, "idle": 3, "await": 4, "land": 5,
    "wait2": 6,
    # idle long-tail — explicitly deprioritised
    "laugh": 30, "joy": 30, "cry": 30, "odori": 30,
    # walk favourites — ``walk`` outranks ``run`` outranks ``move``.
    "walk": 0, "run": 1, "move": 2, "dash": 3,
    "walkl": 5, "walkr": 5, "wgwalk": 5,
    # attack favourites — generic ``attack`` beats body-part-specific
    # variants so a multi-form BML's "main" attack lands first.
    "attack": 0, "atack": 1, "atk": 2,
    "fire": 5, "beam": 5, "shoot": 5, "bite": 5,
    "tatk": 8, "tlatk": 9, "hoe": 10,
}


def extract_action_hint(motion_name: str) -> str:
    """Classify an NJM filename's action.

    Looks at the verb prefix (everything before the FIRST underscore)
    and maps it via :data:`ACTION_VERBS`. Returns ``"unknown"`` when
    the verb is not in the table.

    Stripping ``.njm`` is optional; the function tolerates both forms.

    Some BMLs prefix the verb with a 2-letter body-part designator
    (``bal``, ``bc``, ``cl`` in Bal Claw; ``f``, ``r``, ``l`` in
    Vol Opt's directional attacks). When the exact verb misses the
    table we fall back to a sub-string scan against the longest
    known keywords first — ``balwait`` falls through to
    ``"wait"`` → idle. The sub-string scan only fires when the exact
    lookup fails so well-formed names hit the fast path.

    >>> extract_action_hint("walk_boss1_s_nb_dragon.njm")
    'walk'
    >>> extract_action_hint("tlatk_bm4_ps_ma_tail")
    'attack'
    >>> extract_action_hint("balwait_re6_b_bal_body.njm")
    'idle'
    >>> extract_action_hint("1st_advent_root.njm")
    'unknown'
    """
    s = motion_name.lower()
    if s.endswith(".njm"):
        s = s[:-4]
    if "_" not in s:
        verb = s
    else:
        verb, _ = s.split("_", 1)

    direct = ACTION_VERBS.get(verb)
    if direct is not None:
        return direct

    # Sub-string fallback for body-part-prefixed verbs (balwait,
    # bcwait, clattack, cldie, etc.). Iterate longest keys first to
    # avoid premature ``walk``-in-``walking`` style false positives.
    for key in _ACTION_VERBS_BY_LEN_DESC:
        if key in verb:
            return ACTION_VERBS[key]
    return ACTION_UNKNOWN


def _strip_motion_stem(motion_name: str) -> str:
    """Return the post-verb stem of an NJM filename.

    ``walk_boss1_s_nb_dragon.njm`` → ``boss1_s_nb_dragon``.

    The stem is what ``MotionRef`` matches against the loaded
    ``.nj``/``.xj`` model basename. Empty string when the input has
    no underscore (e.g. a top-level loose ``.njm`` like
    ``fs_obj_hiraishin_a.njm``).
    """
    s = motion_name.lower()
    if s.endswith(".njm"):
        s = s[:-4]
    if "_" not in s:
        return ""
    _, stem = s.split("_", 1)
    return stem


# ---------------------------------------------------------------------------
# Resolver dataclass.
# ---------------------------------------------------------------------------


@dataclass
class MotionRef:
    """One discovered NJM motion candidate.

    Attributes
    ----------
    archive
        Path to the BML (or top-level loose ``.njm``) that contains
        this motion. Equals ``path`` when there's no archive wrapper.
    inner_name
        Inner entry name within the BML (e.g. ``walk_boss1_s_nb_dragon.njm``).
        Empty string for top-level loose files.
    motion_label
        Display name without the ``.njm`` suffix.
    action
        One of the ``ACTION_*`` constants — derived via
        :func:`extract_action_hint`.
    confidence
        ``0.0..1.0`` — higher means the resolver is more confident the
        motion belongs to the requested model. Tier-2 stem matches get
        ``1.0``; Tier-3 (same-BML, different stem) get ``0.6``;
        Tier-4 (NpcApcMot fallback) gets ``0.3``. Reserved Tier-1 hint
        gets ``1.0``.
    tier
        Numeric tier (1..4) for debugging / priority sort tie-breaks.
    stem
        The post-verb stem extracted from ``inner_name``.
    """

    archive: Path
    inner_name: str
    motion_label: str
    action: str
    confidence: float
    tier: int
    stem: str

    @property
    def path(self) -> Path:
        """Backwards-compatible alias for the archive path.

        Older callers that thought of NJM as a loose file expect a
        ``path`` attribute; we keep both names so existing wire-format
        glue stays one-line.
        """
        return self.archive

    @property
    def source_label(self) -> str:
        """Wire-friendly source label, ``<bml>#<inner>`` or ``<file>``."""
        if self.inner_name:
            return f"{self.archive.name}#{self.inner_name}"
        return self.archive.name


# ---------------------------------------------------------------------------
# Tier helpers.
# ---------------------------------------------------------------------------


def _model_inner_stem(model_path: Path, inner_name: Optional[str]) -> str:
    """Compute the basename-stem to match motion entries against.

    Rules:

    * If ``inner_name`` is provided (e.g. ``boss1_s_nb_dragon.nj``),
      strip its extension. This is the canonical case — the user is
      viewing one specific inner model from a multi-inner BML.
    * Otherwise fall back to ``model_path.stem``. For a top-level
      ``foo.nj`` that's just ``foo``; for a BML with no specified
      inner, the BML's stem isn't perfect (``bm_boss1_dragon`` ≠
      ``boss1_s_nb_dragon``) but we still use it as a hint — Tier 3
      will surface every same-BML motion regardless.
    """
    if inner_name:
        s = inner_name.lower()
        for ext in (".nj", ".xj", ".njm"):
            if s.endswith(ext):
                s = s[: -len(ext)]
                break
        return s
    return model_path.stem.lower()


def _list_bml_motions(bml_path: Path) -> list[str]:
    """Return inner ``.njm`` names inside a BML, lowercase-suffixed.

    Returns an empty list on parse failure — the resolver treats a
    broken BML as "no motions discovered" rather than raising, so a
    single bad archive doesn't sink the panel.
    """
    try:
        entries = parse_bml(bml_path.read_bytes())
    except (OSError, ValueError):
        return []
    return [
        e.name
        for e in entries
        if e.name.lower().endswith(".njm")
    ]


# Action-priority sort key. Tier-2 entries sort before Tier-3 entries
# regardless of action; within a tier action priority tie-breaks; within
# an action the verb sub-priority picks the canonical idle (``wait`` over
# ``laugh``) and locomotion (``walk`` over ``run``).
def _sort_key(m: MotionRef) -> tuple[int, int, int, str]:
    name_lower = m.inner_name.lower()
    if name_lower.endswith(".njm"):
        name_lower = name_lower[:-4]
    verb = name_lower.split("_", 1)[0] if "_" in name_lower else name_lower
    sub = _VERB_SUB_PRIORITY.get(verb, 50)
    return (
        m.tier,
        ACTION_PRIORITY.get(m.action, 99),
        sub,
        m.inner_name.lower(),
    )


# ---------------------------------------------------------------------------
# Top-level resolver.
# ---------------------------------------------------------------------------

# BMLs that lack inline motions and pull from NpcApcMot.bml.
_NPC_HOST_PREFIXES = ("pl", "bm_n_", "bm_npc_")
_NPC_MOTION_PACK = "NpcApcMot.bml"


def resolve_motions_for_model(
    model_path: Path,
    inner_name: Optional[str] = None,
    *,
    npc_motion_pack_search_roots: tuple[Path, ...] = (),
) -> list[MotionRef]:
    """Discover NJM motion candidates for a model.

    Parameters
    ----------
    model_path
        Path to the BML or top-level ``.nj``/``.xj`` the viewer is
        loading. MUST exist on disk; the resolver does not validate.
    inner_name
        For BML containers, the specific inner model the viewer is
        rendering (e.g. ``"boss1_s_nb_dragon.nj"``). Used to compute
        Tier-2 stem affinity. Pass ``None`` for top-level standalone
        models.
    npc_motion_pack_search_roots
        Additional directories to search for ``NpcApcMot.bml`` (the
        Tier-4 fallback). The resolver tries ``model_path.parent`` first
        and then each of these in order. Caller (server.py) supplies
        ``DATA_DIR`` and ``LIVE_DATA_DIR`` here.

    Returns
    -------
    list[MotionRef]
        Sorted by ``(tier, action_priority, inner_name)``. The first
        entry is the resolver's best guess for "default auto-play",
        which the existing frontend picker uses unchanged.
    """
    out: list[MotionRef] = []
    target_stem = _model_inner_stem(model_path, inner_name)
    ext = model_path.suffix.lower()

    if ext == ".bml":
        # Tier 2 + Tier 3: every inner .njm in this BML.
        for nm in _list_bml_motions(model_path):
            stem = _strip_motion_stem(nm)
            action = extract_action_hint(nm)
            label = nm[:-4] if nm.lower().endswith(".njm") else nm
            if stem and target_stem and (
                stem == target_stem
                or stem == target_stem.lower()
                or target_stem.endswith(stem)
                or stem.endswith(target_stem)
            ):
                # Tier 2 — exact or suffix-overlap match.
                tier = 2
                conf = 1.0
            else:
                # Tier 3 — same archive, different stem.
                tier = 3
                conf = 0.6
            out.append(MotionRef(
                archive=model_path,
                inner_name=nm,
                motion_label=label,
                action=action,
                confidence=conf,
                tier=tier,
                stem=stem,
            ))

        # Tier 4 — NpcApcMot.bml fallback for player / NPC body BMLs.
        # We only fire it when no Tier-2 (stem-matched) motions exist;
        # firing unconditionally would clutter every monster BML's list
        # with 120 unrelated NPC motions.
        host_lower = model_path.name.lower()
        is_npc_host = any(host_lower.startswith(p) for p in _NPC_HOST_PREFIXES)
        has_tier2 = any(m.tier == 2 for m in out)
        if is_npc_host and not has_tier2:
            search_roots = (model_path.parent, *npc_motion_pack_search_roots)
            seen_pack: set[Path] = set()
            for root in search_roots:
                pack = (root / _NPC_MOTION_PACK)
                try:
                    pack_resolved = pack.resolve()
                except OSError:
                    continue
                if pack_resolved in seen_pack:
                    continue
                seen_pack.add(pack_resolved)
                if not pack_resolved.is_file():
                    continue
                for nm in _list_bml_motions(pack_resolved):
                    label = nm[:-4] if nm.lower().endswith(".njm") else nm
                    out.append(MotionRef(
                        archive=pack_resolved,
                        inner_name=nm,
                        motion_label=label,
                        action=extract_action_hint(nm),
                        confidence=0.3,
                        tier=4,
                        stem=_strip_motion_stem(nm),
                    ))
                # Stop after the first hit; we don't want to double-list
                # motions if both DATA_DIR and LIVE_DATA_DIR carry the pack.
                break
    elif ext in (".nj", ".xj"):
        # Top-level standalone model. Look for sibling ``.njm`` files in
        # the same directory. PSOBB.IO's only loose .njm files are
        # under ``data/scene/`` (cinematic camera tracks); they don't
        # pair with any standalone .nj. We honour the layout for
        # completeness / mod compatibility.
        parent = model_path.parent
        try:
            siblings = sorted(parent.glob("*.njm"))
        except OSError:
            siblings = []
        for sib in siblings:
            if not sib.is_file():
                continue
            label = sib.stem
            stem = _strip_motion_stem(sib.name)
            action = extract_action_hint(sib.name)
            # Stem affinity for top-level files: exact name match
            # (foo.nj ↔ foo.njm) is Tier 2; everything else is Tier 3.
            if stem == target_stem or sib.stem.lower() == target_stem:
                tier, conf = 2, 1.0
            else:
                tier, conf = 3, 0.6
            out.append(MotionRef(
                archive=sib,
                inner_name="",
                motion_label=label,
                action=action,
                confidence=conf,
                tier=tier,
                stem=stem,
            ))

    out.sort(key=_sort_key)
    return out


__all__ = [
    "ACTION_WALK",
    "ACTION_IDLE",
    "ACTION_ATTACK",
    "ACTION_DIE",
    "ACTION_HIT",
    "ACTION_SPAWN",
    "ACTION_FLY",
    "ACTION_DESPAWN",
    "ACTION_UNKNOWN",
    "ACTION_VERBS",
    "ACTION_PRIORITY",
    "MotionRef",
    "extract_action_hint",
    "resolve_motions_for_model",
]
