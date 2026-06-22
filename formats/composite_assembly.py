# Composite multi-inner BML placement table for PSOBB Blue Burst bosses.
#
# Background
# ----------
# Multi-part bosses (De Rol Le, Vol Opt, Dragon, Dark Falz, Olga Flow,
# Pan Arms, ...) ship as a single ``.bml`` archive holding several
# top-level ``.nj`` inners — one per body part. Each inner's NJCM mesh
# tree is rooted at its OWN local origin: a probe across all 13 boss
# BMLs in PSOBB.IO confirmed that every primary inner has root
# MeshTreeNode TRS = (pos=0,0,0; rot=0; scale=1) with eval_flags
# carrying NJD_EVAL_UNIT_POS|UNIT_ANG|UNIT_SCL (0x07) — explicit
# identity placement at file scope.
#
# The actual per-part offsets are not in the assets. They are emitted
# by the running game's entity-init / sub-entity-spawn code, which
# allocates a child entity for each body part and writes its world
# TRS based on hard-coded constants (boss layout) plus the parent's
# orientation (e.g. De Rol Le's helm tracks the body's spine bone,
# Vol Opt's pillars are static around the room centre).
#
# Without that data, the model viewer renders every inner at the
# world origin and the user sees a stack of meshes piled on top of
# each other instead of a coherent boss.
#
# Data sources investigated (2026-04-30)
# --------------------------------------
# 1. **pso-blender** (``_modelwork/pso-blender/``) — community
#    Blender exporter that supports BML/XJ. No composite assembly
#    table; users construct hierarchy manually with Object Parenting
#    in Blender. Verified absent: no ``boss_parts.py``, no
#    composite/derole/derorure/volopt symbols anywhere in the
#    Python source.
# 2. **Phantasmal World** (``_reference/phantasmal-world/``) — Kotlin
#    quest editor. Renders entities by ``cls`` ID (server-side game
#    object), NOT by per-BML composition. Has no boss-part assembly
#    logic either. Confirmed by grepping the entire psolib + web
#    tree for derole/composite/sub_part/inner_offset.
# 3. **Static analysis of PSOBB.exe** — the placement constants do
#    exist in the running game. The De Rol Le entity singleton lives
#    at ``0x00A43CE0`` (``derolle_global``, Ghidra label confirmed).
#    Each boss has a constructor that allocates the sub-entities and
#    writes their initial TRS. Pulling the literal floats requires
#    Ghidra decomp of those constructors plus runtime tracing — not
#    in scope for the initial composite endpoint.
#
# Strategy
# --------
# Encode what we can recover from screenshots / wiki / level-design
# observations as a hand-curated literal table keyed by BML basename
# (case-insensitive). For bosses we have nothing on, return a
# single-part identity-fallback assembly so the endpoint still
# renders SOMETHING (the body) instead of failing the whole bundle.
#
# All placements use **PSOBB world units**: 1 unit ~= 1 cm in the
# game's D3D9 left-handed Y-up world (Y is up, +Z forward, +X right).
# Per-axis scale 1.0 = unchanged. ``rot_euler`` uses Phantasmal's
# default **ZYX intrinsic** order to match ``formats/xj.py`` and the
# Sega Ninja SDK default (NJD_EVAL_ZXY_ANG = 0x20 OPT-IN flag selects
# ZXY; absence = ZYX). Angles are radians.
#
# Convention summary:
#   pos:        (x, y, z) in world units, child-relative to parent_inner
#               (or world-absolute when parent_inner is None).
#   rot_euler:  (rx, ry, rz) radians, intrinsic ZYX (Phantasmal /
#               three.js "ZYX") unless ``notes`` says otherwise.
#   scale:      (sx, sy, sz) multiplicative.
#
# This module is intentionally a static literal table: the placement
# data we have is sparse (one boss with educated-guess offsets, the
# rest are TODO), and a Python dict beats SQLite for "10 entries
# accessed once per /api/composite_bundle call".
"""Composite multi-inner BML placement table for PSOBB bosses.

Provides ``CompositePart`` / ``CompositeAssembly`` dataclasses and a
``lookup_composite(bml_path)`` helper used by ``/api/composite_bundle``
in ``server.py``.

Most boss BMLs ship without per-part placement data in their inner
NJCM trees (every inner roots at the world origin). The placement
constants live inside PSOBB.exe entity-init code and have to be
recovered manually. This table is the curated set of offsets we have
recovered so far; missing bosses fall back to identity placement at
the world origin.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompositePart:
    """One body part inside a composite multi-inner boss BML.

    Attributes
    ----------
    inner_nj:
        The NJ entry name as it appears in the BML directory (case
        sensitive — must match ``BmlEntry.name`` exactly so the inner
        reader can locate the slice).
    pos:
        ``(x, y, z)`` world-unit offset. Interpreted relative to
        ``parent_inner`` if set, otherwise absolute world-space.
    rot_euler:
        ``(rx, ry, rz)`` radians, intrinsic **ZYX** order (Phantasmal /
        three.js default). Use ``ZYX`` unless ``notes`` documents the
        ZXY exception (NJD_EVAL_ZXY_ANG=0x20 in the engine flag bits).
    scale:
        Per-axis ``(sx, sy, sz)`` multiplier. ``(1, 1, 1)`` = no change.
    parent_inner:
        Name of another inner in the same BML whose pose this part
        rides on (e.g. De Rol Le's helm follows the body). When
        ``None`` the part is world-absolute.
    parent_bone:
        DFS node index into the PARENT inner's NJCM skeleton. When set,
        the fragment is attached as a CHILD of that bone and rides the
        bone's animated world matrix — the in-game behaviour for an
        appendage hung off a specific body joint. The index is the same
        pre-order-DFS node number the engine's ``DerolleGetModelNode``
        uses (verified: studio ``xj.parse_skeleton`` node index == engine
        node index, 1:1). ``local_offset`` is then applied in the bone's
        local frame. ``None`` means no bone attachment (the part uses the
        plain ``pos`` / ``rot_euler`` / ``scale`` TRS under
        ``parent_inner`` instead). Requires ``parent_inner`` to name the
        inner whose skeleton owns the bone.
    local_offset:
        ``(x, y, z)`` offset in the parent BONE's local frame, applied
        on top of the bone's animated world matrix. Only meaningful when
        ``parent_bone`` is set. ``(0, 0, 0)`` = sit exactly at the bone
        (the faithful default — the bone matrix already carries the
        in-game attach position so no hand-curated magic offset is
        needed).
    notes:
        Free-form provenance / caveat string. Use ``"placement
        unknown — TODO"`` for parts we couldn't resolve so the
        frontend can warn the user.
    """
    inner_nj: str
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot_euler: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    parent_inner: Optional[str] = None
    parent_bone: Optional[int] = None
    local_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    notes: str = ""


@dataclass(frozen=True)
class CompositeAssembly:
    """Assembly description for one BML file.

    Attributes
    ----------
    bml_path:
        Lowercase BML basename (e.g. ``"bm_boss2_de_rol_le.bml"``).
        The lookup helper normalises caller input to lowercase before
        hashing into the registry, so this should always be lowercase.
    parts:
        Ordered list of ``CompositePart``. Order matters for the
        wire response (frontend may render in array order); typically
        the body / centerpiece part is first so a frontend that
        truncates at N parts still shows the silhouette correctly.
    source:
        Provenance tag — one of ``"pso-blender"``,
        ``"static-analysis-fcn-0xNNNNNNNN"``, ``"hand-curated"``,
        ``"psov2-catalog"``, ``"bml-root-trs"``, or
        ``"identity-fallback"``. Surfaced verbatim in the API response
        so callers can decide which tier of data they trust.
    layout:
        The psov2 NinjaPlugin catalog kind that governs how the parts
        are laid out by the frontend. This mirrors the reference
        renderer's per-catalog ``api_setModel`` / ``api_setItem`` /
        ``api_setWeapon`` / ``api_setPlayer`` behaviour 1:1 (the owner
        wants a pixel side-by-side match):

          * ``"enemy"``  — primary (``parts[0]``) at the world origin;
            every other part laid in a straight +X row at
            ``x = 20 * i`` (``i`` = 1-based index of the non-primary
            part). Matches ``NinjaPlugin.api_setModel``.
          * ``"object"`` — identical +X row, step 20. Matches
            ``api_setItem``.
          * ``"weapon"`` — primary at origin; extras at ``x = 10 * i``
            (step 10). Matches ``api_setWeapon``.
          * ``"player"`` — NO spread. ``parts[0]`` (the body) sits at
            the origin and owns the full skeleton; every other part is
            attached as a CHILD of a body skeleton bone (psov2 attaches
            head / hair / cap to ``body.skeleton.bones[59]``) so it
            rides the bone's world transform and stands on the
            character. Matches ``api_setPlayer`` + the per-class
            ``AssetPlayer`` handler's ``bones[59].add(...)`` calls.
          * ``""`` / ``None`` — legacy hand-curated absolute-TRS /
            bone-attach layout (the pre-psov2 placement table). The
            frontend honours each part's literal ``pos`` /
            ``rot_euler`` / ``scale`` / ``parent_inner`` /
            ``parent_bone``.

        When ``layout`` is one of the psov2 kinds the frontend IGNORES
        the per-part literal ``pos`` (it derives the spread from the
        part index) — the part ORDER is what matters, matching psov2's
        curated ``keys[]`` order exactly.
    """
    bml_path: str
    parts: List[CompositePart]
    source: str
    layout: str = ""


# ---------------------------------------------------------------------------
# Curated placement table
# ---------------------------------------------------------------------------
#
# Literal Python dict keyed by lowercase BML basename. Add entries
# here as we recover more boss-part data via static analysis or the
# pso-blender community.
#
# Coverage status (2026-04-30):
#   bm_boss1_dragon.bml          IDENTITY-FALLBACK (single primary inner)
#   bm_boss2_de_rol_le.bml       HAND-CURATED (best-effort layout)
#   bm_boss2_de_rol_le_a.bml     HAND-CURATED (alt variant, same layout)
#   bm_boss3_volopt.bml          TODO (constructors at 0x00A44804+)
#   bm_boss7_de_rol_le_c.bml     HAND-CURATED-MINIMAL (Challenge variant)
#   bm_boss8_dragon.bml          IDENTITY-FALLBACK (single primary inner)
#   bm_boss4_*, bm_boss5_*,
#   bm_boss9_*                   FILES NOT PRESENT in PSOBB.IO/data
#                                (see psobb_full_entity_map.md for cls
#                                refs — Olga Flow unitxt 78, Falz cls
#                                cluster 0x00A4xxxx).


# De Rol Le placement notes (RE-derived, 2026-06-21)
# ---------------------------------------------------
# Ground truth from the decompiled psobb.exe (Psobb.exe-05112026.c) AND a
# direct skeleton parse of bm_boss2_de_rol_le.bml (both cross-checked):
#
#   THE POINTY SKULL IS INTRINSIC TO THE BODY MODEL.
#   boss2_b_derorure_body.nj (model_files[2], 176 bones) already contains
#   the complete crest the owner wants: skull base = DFS node 0x4d (=77),
#   chain 77 -> 84 -> 85 -> 86 -> 87 -> 88 rising to the highest point
#   Y=+12.7 at bone 88. The engine renders the alive/resting boss from
#   this ONE inner via a single NJ-tree draw (g_RenderNJ_1). The "goofy
#   mesh on its head" bug was hand-placing helm_break (broken-armor
#   DEBRIS, 80 sub-meshes) onto the head — the exact thing to NOT do.
#
# So the FAITHFUL rest pose = the body alone (skull included), and the
# other inners are attack / damage states that attach to SPECIFIC BODY
# BONES (not a world offset). We restore the four ATTACK appendages via
# the parent_bone mechanism so they ride the body skeleton in-game-faithful
# positions; placement = sit exactly at the bone (local_offset 0,0,0), the
# bone matrix carries the in-game attach point (no hand-curated magic).
#
# Bone indices below are the verified DFS node indices (engine ==
# studio, 1:1):
#   * head / face crown : bones 33, 34 (parent of skull node 77)
#   * tail base         : bone 104 (Z=-124, deep tail)
#
# LAYOUT 2026-06-21 (psov2-catalog, owner directive "1:1 with psov2"):
# psov2's AssetEnemies "De Rol Le" handler (NinjaPlugin api_setModel) draws
# the body inner at the ORIGIN and SPREADS the other inners in a +X row
# (mesh.position.x = dx; dx += 20). The curated keys[] ORDER is
# authoritative (NOT BML directory order):
#   body (origin) -> fin_b(20) -> fin_a(40) -> sting(60) -> tentacle(80)
#   -> helm_break(100) -> shell_break(120)
# All fragments share the SAME body PVM (boss2_b_derorure_body.pvm) — psov2
# passes the body `tex` to every fragment loader. The earlier bone-attach
# placement (fins on the head, sting/tentacle on the tail, _break omitted)
# was the in-game-faithful layout but it is NOT what psov2's parity view
# shows; the owner wants the side-by-side to match psov2, so we lay the
# parts out exactly as api_setModel does.
# psov2 ref: _reference/psov2/public/js/AssetEnemies.js L846-910.
_DE_ROL_LE_PARTS: List[CompositePart] = [
    # Body — the segmented carapace + legs + tail + head. This single .nj
    # is the primary; rendered at the origin and drives animation. psov2
    # textures it (and every spread fragment) with boss2_b_derorure_body.pvm.
    CompositePart(
        inner_nj="boss2_b_derorure_body.nj",
        pos=(0.0, 0.0, 0.0),
        notes="body — primary at origin (shared body PVM). psov2 keys[] order.",
    ),
    # Spread fragments — laid in a +X row by the frontend (x=20,40,...) in
    # this exact psov2 keys[] order. fin_b BEFORE fin_a (psov2 order).
    CompositePart(inner_nj="boss2_b_derorure_fin_b.nj", notes="spread (psov2 keys[0])"),
    CompositePart(inner_nj="boss2_b_derorure_fin_a.nj", notes="spread (psov2 keys[1])"),
    CompositePart(inner_nj="boss2_b_derorure_sting.nj", notes="spread (psov2 keys[2])"),
    CompositePart(inner_nj="boss2_b_derorure_tentacle.nj", notes="spread (psov2 keys[3])"),
    CompositePart(inner_nj="boss2_b_helm_break.nj", notes="breakable-armor debris — spread (psov2 keys[4])"),
    CompositePart(inner_nj="boss2_b_shell_break.nj", notes="breakable shell debris — spread (psov2 keys[5])"),
]


# Challenge-mode De Rol Le variant (bm_boss7_de_rol_le_c.bml). Not present
# in psov2's catalog (no parity reference), so we lay it out with the same
# psov2 ENEMY convention the base variant uses: body primary at the origin
# + the remaining inners spread in a +X row (the frontend computes the
# step from the part index). Same body skeleton (boss2_b_derorure_body.nj).
_DE_ROL_LE_C_PARTS: List[CompositePart] = [
    CompositePart(
        inner_nj="boss2_b_derorure_body.nj",
        pos=(0.0, 0.0, 0.0),
        notes="body — primary at origin (Challenge variant).",
    ),
    CompositePart(inner_nj="boss2_b_derorure_tentacle.nj", notes="spread"),
    CompositePart(inner_nj="hige_at01_tentacle.nj", notes="Challenge-only hige tentacle — spread"),
]


# Dragon — single primary inner (`boss1_s_nb_dragon.nj`). The other
# two NJ entries (`lo_*`, `*_sd_*`) are LOD / shadow proxies that
# the BML viewer's classifier already filters out of the default
# composite. So there's nothing to assemble — return identity.
_DRAGON_PARTS: List[CompositePart] = [
    CompositePart(
        inner_nj="boss1_s_nb_dragon.nj",
        pos=(0.0, 0.0, 0.0),
        rot_euler=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        parent_inner=None,
        notes="dragon body — single primary inner; LOD/shadow inners excluded",
    ),
]


# Vol Opt placement notes (hand-curated, 2026-04-30)
# ---------------------------------------------------
# Vol Opt is a TWO-PHASE boss: a wall-mounted "computer face" (phase 1
# the player breaks) and a snake-like body (phase 2 that emerges after
# phase 1 dies). In-game the two phases NEVER appear simultaneously;
# for static asset preview we lay them out side-by-side / stacked so
# the user can inspect each variant of the boss in one shot.
#
# BML inner inventory (probed 2026-04-30 against bm_boss3_volopt.bml):
#   me5p01_y_all.nj          — Phase 1 monolithic mesh (wall + face)
#   me5p02_y_all.nj          — Phase 2 monolithic mesh (snake form)
#   me5p02_y_all_parts.nj    — Phase 2 alt with separable parts
#   me5p02_y_cage.nj         — Phase 2 surrounding cage prop
#   me5p02_y_pillar.nj       — Phase 2 central pillar
#   me5p02_y_broken01.nj     — Phase 2 damage-state geometry
#   me5p02_y_missile.nj      — Phase 2 projectile (missile)
#   me5_y_all.nj             — combined "all" form (Challenge variant)
#   fe_obj_vo_futa_moto.nj   — door / lid base (room geometry)
#   fe_obj_vo_tenjo_hahen01.nj   — ceiling fragment 1
#   fe_obj_vo_tenjo_hahen02.nj   — ceiling fragment 2
#   fs_obj_hiraishin_a.nj    — lightning rod prop
#   fe_obj_hira_kage.nj      — lightning rod shadow plane
#
# (Plus a dozen .xj monitor variants — fe_obj_vo_mo_*_aka/ao/hakai —
# which are PER-MONITOR red/blue/destroyed state textures rendered as
# part of phase 1's material atlas, not as separate composite parts.)
#
# Source-tag rationale: keeping this "hand-curated" because the offsets
# below are TODO-flagged best-effort, NOT engine constants. The actual
# phase-1 wall layout, phase-2 snake spawn point, and pillar positions
# live in the constructor cluster at 0x00A447D4..0x00A44BF0 (see
# psobb_full_entity_map.md). Recovering them needs Ghidra decomp of:
#   * voloptcontrol_constructor    @ 0x00A447D4 (top-level controller)
#   * player_hit_volopt_core       @ 0x00A44804 (phase 1 body)
#   * player_hit_volopt_monitor    @ 0x00A449D0 (per-monitor entity)
#   * player_hit_volopt_pillar     @ 0x00A44A18 (pillar entity)
#   * init_voloptform2_global_config @ 0x00A44BF0 (phase 2 init)
#
# TODO(static-analysis): pull literal floats from the constructor
# cluster above and replace these visual-separation guesses with the
# engine's own per-part TRS. See psobb_r2_findings.md for the cluster
# context (Vol Opt class descriptor list at 0xA44C00..0xA44CBC).
# LAYOUT 2026-06-21 (psov2-catalog, owner directive "1:1 with psov2"):
# psov2's AssetEnemies "Vol Opt" handler (api_setModel) draws me5p02_y_all.nj
# (phase-2 snake form) at the ORIGIN as the primary, then SPREADS the
# curated keys[] list in a +X row (x=20,40,...). The first THREE keys[]
# entries in psov2 (me5p02_y_broken01 / me5_y_all / me5p01_y_all) are
# COMMENTED OUT, so they are NOT rendered — we drop them too. Each spread
# fragment is textured with its OWN per-fragment .pvm (the frontend's
# per-inner texture binding already does this). The exact keys[] ORDER is
# authoritative.
# psov2 ref: _reference/psov2/public/js/AssetEnemies.js L1209-1382.
_VOLOPT_PARTS: List[CompositePart] = [
    # Primary — phase-2 snake form at the origin (psov2 modelLoader.parse).
    CompositePart(
        inner_nj="me5p02_y_all.nj",
        pos=(0.0, 0.0, 0.0),
        notes="phase 2 (snake form) — primary at origin (psov2 modelLoader).",
    ),
    # Spread keys[] in psov2 order (the 3 commented-out entries are dropped).
    CompositePart(inner_nj="fe_obj_hira_kage.nj", notes="spread"),
    CompositePart(inner_nj="fe_obj_vo_mo_dai_aka.nj", notes="spread"),
    CompositePart(inner_nj="fe_obj_vo_mo_dai_ao.nj", notes="spread"),
    CompositePart(inner_nj="fe_obj_vo_mo_dai_hakai.nj", notes="spread"),
    CompositePart(inner_nj="fe_obj_vo_mo_sho01_aka.nj", notes="spread"),
    CompositePart(inner_nj="fe_obj_vo_mo_sho01_ao.nj", notes="spread"),
    CompositePart(inner_nj="fe_obj_vo_mo_sho01_hakai.nj", notes="spread"),
    CompositePart(inner_nj="fe_obj_vo_mo_sho02_aka.nj", notes="spread"),
    CompositePart(inner_nj="fe_obj_vo_mo_sho02_ao.nj", notes="spread"),
    CompositePart(inner_nj="fe_obj_vo_mo_sho02_hakai.nj", notes="spread"),
    CompositePart(inner_nj="fe_obj_vo_mo_sho03_aka.nj", notes="spread"),
    CompositePart(inner_nj="fe_obj_vo_mo_sho03_ao.nj", notes="spread"),
    CompositePart(inner_nj="fe_obj_vo_mo_sho03_hakai.nj", notes="spread"),
    CompositePart(inner_nj="fe_obj_vo_futa_moto.nj", notes="spread"),
    CompositePart(inner_nj="fe_obj_vo_tenjo_hahen01.nj", notes="spread"),
    CompositePart(inner_nj="fe_obj_vo_tenjo_hahen02.nj", notes="spread"),
    CompositePart(inner_nj="fs_obj_hiraishin_a.nj", notes="spread"),
    CompositePart(inner_nj="me5p02_y_cage.nj", notes="spread"),
    CompositePart(inner_nj="me5p02_y_missile.nj", notes="spread"),
    CompositePart(inner_nj="me5p02_y_pillar.nj", notes="spread"),
]


# Pan Arms placement notes (hand-curated, 2026-04-30)
# ----------------------------------------------------
# Pan Arms exposes 3 mesh inners in `bm7_s_paa_body.bml`:
#   bm7_s_paa_body.nj        — combined Pan Arms (gattai / fused state)
#   bm7_s_pal_body.nj        — Migium (left fragment after split)
#   bm7_s_par_body.nj        — Hidoom  (right fragment after split)
#
# In-game these three forms NEVER co-render: gattai is the resting
# fused state and "bunri" (split) replaces the single Pan Arms entity
# with two sibling entities (Hidoom + Migium) at fixed lateral offsets
# on either side of the previous fused position. For static asset
# preview we lay all three out side-by-side so the user can compare.
#
# Reference: psobb_full_entity_map.md — panarms_constructor (cls
# 0x00A742A8) handles fused state; sub-entity allocator at
# 0x00A74370 spawns the Hidoom/Migium pair. Lateral offset is roughly
# 50 units (pre-scale) but actual constants live in those constructors.
#
# TODO(static-analysis): recover the engine's Hidoom/Migium spawn
# offset from panarms_split_state (call site of 0x00A74370).
_PAN_ARMS_PARTS: List[CompositePart] = [
    # Centre — the fused Pan Arms.
    CompositePart(
        inner_nj="bm7_s_paa_body.nj",
        pos=(0.0, 0.0, 0.0),
        rot_euler=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        parent_inner=None,
        notes=(
            "Pan Arms fused (gattai) — primary skinned mesh; centre of "
            "preview layout."
        ),
    ),
    # Left fragment — Migium. Offset left so the silhouette is visible
    # without overlapping the fused mesh.
    CompositePart(
        inner_nj="bm7_s_pal_body.nj",
        pos=(-100.0, 0.0, 0.0),
        rot_euler=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        parent_inner=None,
        notes=(
            "Migium (left split fragment) — visual-separation offset "
            "for preview. In-game spawned at fused position when Pan "
            "Arms enters bunri (split) state. TODO: engine constants "
            "from panarms_split_state (0x00A74370 sub-entity alloc)."
        ),
    ),
    # Right fragment — Hidoom.
    CompositePart(
        inner_nj="bm7_s_par_body.nj",
        pos=(100.0, 0.0, 0.0),
        rot_euler=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        parent_inner=None,
        notes=(
            "Hidoom (right split fragment) — visual-separation offset "
            "for preview. TODO: engine constants from panarms_split_state."
        ),
    ),
]


# Dark Falz placement notes (hand-curated, 2026-04-30)
# -----------------------------------------------------
# `darkfalz_dat.bml` ships THREE phases plus shared accessory inners:
#   df1_s_body.nj            — Phase 1 body (the floating priest)
#   df1_s_da_heada.nj        — Phase 1 head variant a
#   df1_s_db_heada.nj        — Phase 1 head variant b
#   df1_s_dc_heada.nj        — Phase 1 head variant c
#   df1_s_dodai.nj           — Phase 1 base / pedestal
#   df1_s_simobe.nj          — Phase 1 servant (the Sigh of a God)
#   df1_s_waist.nj           — Phase 1 waist segment
#   df1_anzen.nj             — Phase 1 safety / arena geometry
#   df2_s_body.nj            — Phase 2 body
#   df2_s_dodai1.nj          — Phase 2 base / pedestal
#   df3_s_body.nj            — Phase 3 body (final form)
#   df3_s_wing.nj            — Phase 3 wings
#   df3_sl_body.nj           — Phase 3 (light variant?) body
#   df3_sl_wing.nj           — Phase 3 (light variant?) wings
#   df_event_tower.nj        — cutscene tower prop
#   df_rikomiraju_body.nj    — Rico (cutscene character)
#   fd_obj813_face.nj        — face overlay
#   fd_obj813_flower01.nj    — flower / bloom prop
#   rico_ring3_rico_ring.nj  — Rico's ring prop
#
# Three phases never co-render. Lay them out left/centre/right with
# the centrepiece of each phase ("body" inner) anchored as primary;
# heads/wings parented to their respective phase body so they track.
#
# TODO(static-analysis): recover constants from the Falz constructor
# cluster — see Falz cls cluster note in psobb_full_entity_map.md.
_DARK_FALZ_PARTS: List[CompositePart] = [
    # Phase 1 body — left.
    CompositePart(
        inner_nj="df1_s_body.nj",
        pos=(-300.0, 0.0, 0.0),
        rot_euler=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        parent_inner=None,
        notes="phase 1 body — left of preview layout. TODO: engine constants.",
    ),
    CompositePart(
        inner_nj="df1_s_da_heada.nj",
        pos=(0.0, 80.0, 0.0),
        rot_euler=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        parent_inner="df1_s_body.nj",
        notes="phase 1 head variant a — best-effort head attach. TODO.",
    ),
    CompositePart(
        inner_nj="df1_s_dodai.nj",
        pos=(0.0, -50.0, 0.0),
        rot_euler=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        parent_inner="df1_s_body.nj",
        notes="phase 1 pedestal — best-effort below body. TODO.",
    ),
    # Phase 2 body — centre.
    CompositePart(
        inner_nj="df2_s_body.nj",
        pos=(0.0, 0.0, 0.0),
        rot_euler=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        parent_inner=None,
        notes="phase 2 body — centre of preview layout. TODO: engine constants.",
    ),
    CompositePart(
        inner_nj="df2_s_dodai1.nj",
        pos=(0.0, -50.0, 0.0),
        rot_euler=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        parent_inner="df2_s_body.nj",
        notes="phase 2 pedestal — best-effort below body. TODO.",
    ),
    # Phase 3 body — right.
    CompositePart(
        inner_nj="df3_s_body.nj",
        pos=(300.0, 0.0, 0.0),
        rot_euler=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        parent_inner=None,
        notes="phase 3 body — right of preview layout. TODO: engine constants.",
    ),
    CompositePart(
        inner_nj="df3_s_wing.nj",
        pos=(0.0, 30.0, 0.0),
        rot_euler=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        parent_inner="df3_s_body.nj",
        notes="phase 3 wings — best-effort upper-body attach. TODO.",
    ),
]


# Olga Flow placement notes (hand-curated, 2026-04-30)
# -----------------------------------------------------
# `boss06_plotfalz_dat.bml` is the (Ep2) Olga Flow / "PlotFalz" archive
# despite the file name. Inner inventory:
#   flow_body.nj             — Olga Flow phase-1 body (light form)
#   flow_dark_body.nj        — Olga Flow phase-2 body (dark form)
#   flow_sord.nj             — Olga Flow's sword
#   bossgc_pf01_leg.nj       — Plotfalz phase-1 leg
#   bossgc_pf02_body.nj      — Plotfalz phase-2 body
#   bossgc_pf02l_body.nj     — Plotfalz phase-2 (variant L?) body
#   bossgc_pf01_kirai.nj     — Plotfalz phase-1 mine prop
#   bossgc_pf_mag.nj         — Plotfalz mag prop
#   bssgc_pf01_body.nj       — Plotfalz phase-1 body (note typo: `bssgc`)
#   flowen_aa_t_ok_head.nj   — flow enemy attack head accessory
#   flowen_az_t_body.nj      — flow enemy body accessory
#   bossgc_laser.nj          — laser prop
#   op_flow_mark.nj          — opening cutscene marker
#   wxmS02_k_m_sp_rbuki.nj   — Olga Flow weapon (`rbuki`)
#   lo_*  — LOD proxies (excluded)
#
# Lay Olga Flow's two phases side-by-side; Plotfalz parts are kept off
# to the right since they're a separate boss in the same archive.
#
# TODO(static-analysis): split this entry into two CompositeAssemblys
# (one for Olga Flow proper, one for Plotfalz) once we know which BML
# basename each fragment loads from in-game. As of 2026-04-30 the BML
# loader unifies them under one `boss06_plotfalz_dat.bml` request, so
# a single composite is the pragmatic choice.
_OLGA_FLOW_PARTS: List[CompositePart] = [
    # Olga Flow phase 1 (light form) — centre-left.
    CompositePart(
        inner_nj="flow_body.nj",
        pos=(-150.0, 0.0, 0.0),
        rot_euler=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        parent_inner=None,
        notes=(
            "Olga Flow phase 1 (light) — primary; left half of preview. "
            "TODO: engine constants from Olga Flow constructor."
        ),
    ),
    # Olga Flow phase 2 (dark form) — centre-right.
    CompositePart(
        inner_nj="flow_dark_body.nj",
        pos=(150.0, 0.0, 0.0),
        rot_euler=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        parent_inner=None,
        notes=(
            "Olga Flow phase 2 (dark) — right half of preview. "
            "TODO: engine constants."
        ),
    ),
    # Olga Flow's sword — held in the right hand of either phase. We
    # parent to the dark body since it's the climactic phase.
    CompositePart(
        inner_nj="flow_sord.nj",
        pos=(60.0, 0.0, 0.0),
        rot_euler=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        parent_inner="flow_dark_body.nj",
        notes=(
            "Olga Flow sword — best-effort right-hand attach. "
            "TODO: bone-relative hand offset."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Player class assemblies (psov2-catalog layout = "player")
# ---------------------------------------------------------------------------
#
# psov2's AssetPlayer handler for each class loads the body inner as a
# SkinnedMesh that owns the full skeleton, loads head / hair / cap as
# separate meshes, and PARENTS them onto ``body.skeleton.bones[59]`` (the
# neck/head bone) BEFORE calling ``NinjaPlugin.API.setPlayer`` — which adds
# ONLY the body to the scene (head/hair/cap ride along as bone children)
# and applies NO dx spread. We replicate that 1:1: the body is the primary
# (origin, animation root) and head/hair/cap are bone-attached to body bone
# 59. The body inner owns 64 bones for these classes (verified: bone 59 is
# the neck, parent 56) so the attach index is in range.
#
# Inner names per class are the exact ``bml["plXbdy00.nj"]`` /
# ``plXhed00.nj`` / ``plXhai00.nj`` / ``plXcap00.nj`` strings psov2 parses.
# The OTHER inners in the BML (alternate hair/cap variants) go only into
# psov2's mdlList (export bookkeeping), never the scene — so we leave them
# OUT of the assembly; the composite path renders only the curated parts.
# psov2 ref: _reference/psov2/public/js/AssetPlayer.js L53-751.
#
# The attach bone for head/hair/cap. psov2 hard-codes bones[59] for every
# humanoid player class.
_PLAYER_NECK_BONE = 59


def _player_parts(
    prefix: str,
    *,
    hair: bool = True,
    cap: bool = False,
) -> List[CompositePart]:
    """Build the curated player part list for a class prefix.

    ``prefix`` is the single uppercase class letter psov2 uses in the
    inner names, e.g. ``"A"`` for Humar (plAbdy00.nj / plAhed00.nj /
    plAhai00.nj). Every class has a body + head; ``hair`` / ``cap`` toggle
    the optional pieces (Hucast/Racast/Racaseal/Fonewearl-cap etc. differ).
    """
    parts: List[CompositePart] = [
        CompositePart(
            inner_nj=f"pl{prefix}bdy00.nj",
            pos=(0.0, 0.0, 0.0),
            notes="player body — primary at origin; owns the skeleton.",
        ),
        CompositePart(
            inner_nj=f"pl{prefix}hed00.nj",
            parent_inner=f"pl{prefix}bdy00.nj",
            parent_bone=_PLAYER_NECK_BONE,
            notes="head — attached to body neck bone 59 (psov2 bones[59].add).",
        ),
    ]
    if hair:
        parts.append(
            CompositePart(
                inner_nj=f"pl{prefix}hai00.nj",
                parent_inner=f"pl{prefix}bdy00.nj",
                parent_bone=_PLAYER_NECK_BONE,
                notes="hair — attached to body neck bone 59.",
            )
        )
    if cap:
        parts.append(
            CompositePart(
                inner_nj=f"pl{prefix}cap00.nj",
                parent_inner=f"pl{prefix}bdy00.nj",
                parent_bone=_PLAYER_NECK_BONE,
                notes="cap — attached to body neck bone 59.",
            )
        )
    return parts


# Per-class hair/cap presence mirrors AssetPlayer.js exactly:
#   Humar(A)     : body + head + hair                 (L53-127)
#   Hunewearl(B) : body + head + hair                 (L129-206)
#   Hucast(C)    : body + head                        (L208-270)
#   Ramar(D)     : body + head + hair + cap           (L272-359)
#   Racast(E)    : body + head                        (L361-423)
#   Racaseal(F)  : body + head                        (L425-487)
#   Fomarl(G)    : body + head + hair + cap           (L489-573)
#   Fonewm(H)    : body + head + hair + cap           (L575-662)
#   Fonewearl(I) : body + head + hair + cap           (L664-751)
_PLAYER_CLASS_FILES: Dict[str, Tuple[str, bool, bool]] = {
    # bml basename             : (prefix, has_hair, has_cap)
    "planj.bml": ("A", True, False),   # Humar
    "plbnj.bml": ("B", True, False),   # Hunewearl
    "plcnj.bml": ("C", False, False),  # Hucast
    "pldnj.bml": ("D", True, True),    # Ramar
    "plenj.bml": ("E", False, False),  # Racast
    "plfnj.bml": ("F", False, False),  # Racaseal
    "plgnj.bml": ("G", True, True),    # Fomarl
    "plhnj.bml": ("H", True, True),    # Fonewm
    "plinj.bml": ("I", True, True),    # Fonewearl
}


COMPOSITE_TABLE: Dict[str, CompositeAssembly] = {
    "bm_boss2_de_rol_le.bml": CompositeAssembly(
        bml_path="bm_boss2_de_rol_le.bml",
        parts=_DE_ROL_LE_PARTS,
        # Normal multi-part assembly: body (with the intrinsic pointy
        # skull) as the animation root + four ATTACK appendages bone-
        # attached to the body skeleton (fins on the head, sting/tentacle
        # on the tail). The skull is in the body inner itself — NOT a
        # separate part — so the faithful render shows the crest. The
        # helm/shell damage states are deliberately not in the parts list.
        source="re-derived-bone-attach",
    ),
    # The "_a" variant ships with byte-identical inner names (probed
    # 2026-04-30). Reuse the same parts list — frozen dataclass so
    # sharing the list reference is safe.
    "bm_boss2_de_rol_le_a.bml": CompositeAssembly(
        bml_path="bm_boss2_de_rol_le_a.bml",
        parts=_DE_ROL_LE_PARTS,
        source="re-derived-bone-attach",
    ),
    "bm_boss7_de_rol_le_c.bml": CompositeAssembly(
        bml_path="bm_boss7_de_rol_le_c.bml",
        parts=_DE_ROL_LE_C_PARTS,
        source="hand-curated",
    ),
    "bm_boss1_dragon.bml": CompositeAssembly(
        bml_path="bm_boss1_dragon.bml",
        parts=_DRAGON_PARTS,
        source="identity-fallback",
    ),
    "bm_boss8_dragon.bml": CompositeAssembly(
        bml_path="bm_boss8_dragon.bml",
        parts=_DRAGON_PARTS,
        source="identity-fallback",
    ),
    # Vol Opt — phase 1 (wall) + phase 2 (snake) + phase-2 props.
    # See _VOLOPT_PARTS for inner inventory and TODO list (engine
    # constants live in voloptcontrol_constructor cluster
    # 0x00A447D4..0x00A44BF0).
    "bm_boss3_volopt.bml": CompositeAssembly(
        bml_path="bm_boss3_volopt.bml",
        parts=_VOLOPT_PARTS,
        source="hand-curated",
    ),
    # The "_ap" Vol Opt variant has byte-identical inner names to the
    # base BML (probed 2026-04-30). Reuse the same parts list — frozen
    # dataclass so sharing the list reference is safe.
    "bm_boss3_volopt_ap.bml": CompositeAssembly(
        bml_path="bm_boss3_volopt_ap.bml",
        parts=_VOLOPT_PARTS,
        source="hand-curated",
    ),
    # Pan Arms — fused (gattai) + Hidoom + Migium. See _PAN_ARMS_PARTS.
    "bm7_s_paa_body.bml": CompositeAssembly(
        bml_path="bm7_s_paa_body.bml",
        parts=_PAN_ARMS_PARTS,
        source="hand-curated",
    ),
    # Dark Falz — three phases laid out left/centre/right. See
    # _DARK_FALZ_PARTS for inner inventory.
    "darkfalz_dat.bml": CompositeAssembly(
        bml_path="darkfalz_dat.bml",
        parts=_DARK_FALZ_PARTS,
        source="hand-curated",
    ),
    # Olga Flow — light + dark phases plus sword. The shared
    # `boss06_plotfalz_dat.bml` archive also contains Plotfalz parts
    # (Ep2 Falz variant); we curate the Olga Flow side only. See
    # _OLGA_FLOW_PARTS.
    "boss06_plotfalz_dat.bml": CompositeAssembly(
        bml_path="boss06_plotfalz_dat.bml",
        parts=_OLGA_FLOW_PARTS,
        source="hand-curated",
    ),
}


# ---------------------------------------------------------------------------
# Lookup helper
# ---------------------------------------------------------------------------


def _normalise_bml_key(bml_path: str) -> str:
    """Reduce a path-or-basename to the lowercase basename used as the
    registry key.

    Accepts:
        "bm_boss2_de_rol_le.bml"            -> "bm_boss2_de_rol_le.bml"
        "BM_Boss2_De_Rol_Le.bml"            -> "bm_boss2_de_rol_le.bml"
        "data/bm_boss2_de_rol_le.bml"       -> "bm_boss2_de_rol_le.bml"
        "C:/Users/.../bm_boss2_de_rol_le.bml" -> "bm_boss2_de_rol_le.bml"

    Forward-slash and backslash are both honoured as separators so the
    helper works on Windows-style and POSIX-style input. Leading/
    trailing whitespace is stripped to be robust against frontend
    URL-encoding quirks.
    """
    if not isinstance(bml_path, str):
        return ""
    stripped = bml_path.strip()
    if not stripped:
        return ""
    # Take the rightmost path component using either separator. We
    # avoid pathlib here so this stays cheap (called once per
    # /api/composite_bundle request, but no reason to allocate a
    # full Path object for a basename split).
    last_fwd = stripped.rfind("/")
    last_bwd = stripped.rfind("\\")
    cut = max(last_fwd, last_bwd)
    if cut >= 0:
        stripped = stripped[cut + 1:]
    # The endpoint accepts the BML+inner ``base#inner`` form; for a
    # composite request we want only the BML basename, not the inner.
    hash_at = stripped.find("#")
    if hash_at >= 0:
        stripped = stripped[:hash_at]
    return stripped.lower()


def lookup_composite(bml_path: str) -> Optional[CompositeAssembly]:
    """Look up a composite assembly by BML path / basename.

    Case-insensitive. Accepts a bare basename, a relative path, or
    an absolute path; only the rightmost path component is used as
    the key. The ``base#inner`` API form is honoured by stripping
    the inner suffix before lookup.

    Returns ``None`` for unknown BMLs — the caller should fall back
    to identity placement (single-inner composite at world origin)
    rather than failing the request, so the user still sees the
    centerpiece even when we have no composite metadata.
    """
    key = _normalise_bml_key(bml_path)
    if not key:
        return None
    return COMPOSITE_TABLE.get(key)


__all__ = [
    "CompositePart",
    "CompositeAssembly",
    "COMPOSITE_TABLE",
    "lookup_composite",
]
