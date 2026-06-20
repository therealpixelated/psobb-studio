"""Skeleton retargeting: external glTF animation -> PSOBB NJM.

Inputs:
  * ``ImportedAnimation`` from formats.import_external
    (per-bone keyframe tracks in glTF / Mixamo space)
  * Source skeleton (the glTF file's bones, indexed in the same way
    the animation tracks reference)
  * Target skeleton (a list of ``XjBone`` parsed from the destination
    PSOBB ``.nj`` file)
  * A bone-name map (Mixamo joint name -> target bone index)

Output:
  * An ``NjmRawMotion`` ready to feed into ``njm_writer.encode_njm``.

Algorithm:
  1. Resample every source track to the target frame grid (default
     30 Hz) via linear interpolation. Quaternion tracks slerp; vector
     tracks lerp.
  2. For each TARGET bone, walk the bone-name map to find the
     matching source bone (or skip if absent).
  3. Convert source rotations from glTF (right-handed, Y-up) to PSOBB
     (left-handed, Y-up) via a Z-mirror — see ``_mirror_quat_z`` for
     the algebra.
  4. Compose with the target bone's bind-pose inverse so the emitted
     keyframe represents the LOCAL delta the renderer applies on top
     of the bind. (Without this step a bone whose Mixamo bind has
     identity quat but whose PSOBB bind has a non-identity rotation
     ends up flying off when the animation drives it.)
  5. Convert each composed quat -> ZYX BAMS (matches the bind-pose
     convention used by ``import_external.quat_to_zyx_bams``).
  6. Pack into ``NjmTrack`` per bone, set the per-bone present-mask
     bit for the channels we filled, and return the assembled
     ``NjmRawMotion``.

Edge cases:
  * Source bone with no track (e.g. legs) -> target bone gets no
    track for that channel, which the parser/runtime falls back to
    bind pose for that bone (per ``njm.bone_present_tracks``).
  * Target bone with no source map -> same; track stays empty.
  * Differing FPS between source and target -> resample on the
    target grid.
  * Translation: many Mixamo animations bake hip translation; we
    optionally pass it through (scaled by the source-vs-target
    skeleton bounding-box ratio so the character doesn't fly off
    the screen on a different scale). ``include_translation=False``
    is the typing-animation default — we want rotation-only since
    the lobby girls stay in place at their counter.

IK retargeting (v2, 2026-04-25):
  Different-length arms cause hand-position drift: a Mixamo elbow at
  60° flex with a 30 cm upper arm puts the wrist somewhere that an
  identical 60° on a 25 cm PSOBB upper arm cannot reach. The
  ``enable_ik`` path layers an IK pass over the 1:1 quat copy:

    1. Build the source skeleton's per-frame world transforms by
       walking source.bind_pos + animated rotations (resampled to the
       target frame grid).
    2. Build the target skeleton's per-frame world transforms by
       walking the post-retarget local poses we just assembled.
    3. For each end-effector chain (hand / foot), compute the source
       world position of the END bone and call ``fabrik_solve`` on
       the target's chain (default chain length: 4 = wrist + forearm
       + upper arm + shoulder).
    4. Convert the new joint positions back to per-bone local
       rotations and overwrite the corresponding NJM tracks.

  IK is positional only — the wrist/hand orientation continues to come
  from the 1:1 quat copy. v3 will add rotation-IK so the hand pose
  matches the source's wrist orientation as well.

Bone-alias auto-detection (v3, 2026-04-25):
  Different rig conventions name the same humanoid joints differently:
  Mixamo ``mixamorig:LeftArm``, Unity Mecanim ``LeftUpperArm``,
  Cesium ``Skeleton_arm_joint_L__2_``, Blender Rigify ``upper_arm.L``,
  MakeHuman ``upperarm_l``, free-form ``"left arm"``. Adding every
  alias by hand to ``LOBBY_GIRL_BONE_MAP`` doesn't scale.
  ``auto_detect_bone_role`` maps an unknown source-bone name to one of
  the canonical roles the explicit map already keys on (``"LeftArm"``,
  ``"RightForeArm"``, ``"Hips"``, etc.) when it matches a known
  convention. The retargeter calls it for any source bone whose
  normalised name isn't in ``bone_map`` directly.

VRM humanoid map (v3, 2026-04-25):
  VRM (a glTF extension) tags each bone node with a humanoid bone
  ROLE (``hips`` / ``leftUpperArm`` / ...) in the file metadata. When
  the source carries a VRM humanoid map (``ImportedModel.vrm_humanoid_map``)
  the retargeter uses it as an authoritative routing table that
  bypasses string-matching entirely. VRM role names are translated
  to canonical roles via ``_VRM_ROLE_TO_CANONICAL``; the result feeds
  into the same ``bone_map`` lookup the explicit-name path uses.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .import_external import (
    ImportedAnimation,
    ImportedBone,
    ImportedTrack,
    quat_to_zyx_bams,
)
from .njm import (
    NJD_MTYPE_ANG,
    NJD_MTYPE_POS,
    NJD_MTYPE_SCL,
)
from .njm_writer import NjmBoneTracks, NjmRawMotion, NjmTrack
from .rigging import fabrik_solve as _rigging_fabrik_solve


# ---------------------------------------------------------------------------
# BoneNameMap helpers
# ---------------------------------------------------------------------------
#
# A BoneNameMap is conceptually:
#       Mixamo / standard joint name (str)  ->  PSOBB bone index (int)
#
# We surface a few prebuilt maps for the targets we ship with:
#   * "lobby_girl"       — bm_npc_kenkyu_w / bm_npc_momoka / bm_npc_hosa
#                           (all three share the 64-bone npc humanoid
#                           skeleton; see psobb_npc_skeleton notes).
#   * "monster_humanoid" — for the existing player_body template.
# Custom callers can build their own dict and pass it directly to
# ``retarget_animation``.

BoneNameMap = Dict[str, int]


# Lobby NPC (kenkyu_w / momoka / hosa) — 64-bone humanoid, conventions
# discovered by introspecting the bone hierarchy in
# ``bm_npc_kenkyu_w.bml#kenkyu_w_hone_body.nj``:
#
#   bone 0  = root
#   bone 1  = hips / pelvis (Y up by 11.68)
#   bone 2  = spine / torso (Z-rotation -90 brings bone-local +X to
#             world up, so all upper-body offsets are expressed
#             along +X relative to bone 2's local frame)
#   bone 3  = right clavicle  (X = right side of body in bone 2's frame)
#   bone 4  = right upper-arm joint (parented under 3)
#   bone 5  = right elbow / forearm joint (4.31 along bone 4's +X)
#   bone 7  = right wrist
#   bone 10 = right hand metacarpal (offset 1.43 along +Z = forward)
#   bone 12-19 = mirror left arm
#   bone 21 = neck
#   bone 23 = head (offset 9.03 along +X = up)
#   bone 22 = jaw
#   bone 24-25 = pelvis / lower-body root
#   bone 26-30 = right leg + foot
#   bone 39-43 = left leg + foot
#
# The map below covers the joints the typing animation drives. Bones
# we don't map (legs, head/jaw) keep their bind pose, which is what
# we want for a "stand and type" motion.
LOBBY_GIRL_BONE_MAP: BoneNameMap = {
    "Hips":          1,
    "Spine":         2,
    "Spine1":        2,   # aliased — kenkyu skeleton has only one spine
    "Spine2":        2,
    "Neck":          21,
    "Head":          23,
    # Right arm (Mixamo "Right" = character's right = our +X side).
    "RightShoulder": 3,
    "RightArm":      4,
    "RightForeArm":  7,
    "RightHand":     10,
    # Left arm.
    "LeftShoulder":  12,
    "LeftArm":       13,
    "LeftForeArm":   16,
    "LeftHand":      19,
    # Legs — typing animation freezes them at bind, but we map
    # for completeness so other motions (walk, jog) can use the
    # same map.
    "RightUpLeg":    26,
    "RightLeg":      28,
    "RightFoot":     29,
    "LeftUpLeg":     39,
    "LeftLeg":       41,
    "LeftFoot":      42,
    # ---- Khronos CesiumMan / Cesium-style sample skeletons (2026-04-25
    # external-asset smoke test). These appear in the Khronos
    # glTF-Sample-Assets and propagate into many Sketchfab CC0 assets
    # rigged via the same exporter. The naming is a 19-joint humanoid:
    #   torso_joint_1 = pelvis/hips (rare absent prefix), Skeleton_*
    #   prefix means the joint sits under a "Skeleton" group node.
    #   _N_ suffix is the chain-segment index (_1_ = root-most).
    # This is purely additive — Mixamo "Hips" / "Spine" still resolve
    # exactly as before; these names just give us a fallback path on
    # Khronos-style rigs without forcing a custom bone_map call.
    "Skeleton_torso_joint_1": 1,   # pelvis/hips
    "Skeleton_torso_joint_2": 2,   # spine
    "torso_joint_3":          2,   # upper spine, aliased to single PSOBB spine
    "Skeleton_neck_joint_1":  21,  # neck
    "Skeleton_neck_joint_2":  23,  # head
    # CesiumMan arm chain: _4_ = shoulder, _3_ = upper arm, _2_ = forearm,
    # _1_ = hand (when present). The base joint without index is the
    # shoulder (mirrors Mixamo's "Right/LeftShoulder").
    "Skeleton_arm_joint_R":     3,
    "Skeleton_arm_joint_R__2_": 4,
    "Skeleton_arm_joint_R__3_": 7,
    "Skeleton_arm_joint_R__4_": 10,
    "Skeleton_arm_joint_L":     12,
    "Skeleton_arm_joint_L__2_": 13,
    "Skeleton_arm_joint_L__3_": 16,
    "Skeleton_arm_joint_L__4_": 19,
    # CesiumMan leg chain: _1_ = hip, _2_ = thigh/knee, _3_ = shin,
    # _5_ = foot (CesiumMan's _4_ is empty — toe placeholder).
    "leg_joint_R_1": 26,
    "leg_joint_R_2": 28,
    "leg_joint_R_3": 28,
    "leg_joint_R_5": 29,
    "leg_joint_L_1": 39,
    "leg_joint_L_2": 41,
    "leg_joint_L_3": 41,
    "leg_joint_L_5": 42,
}


# Mixamo prepends "mixamorig:" to every joint name. The retargeter
# strips that prefix when looking up the map, so callers don't need
# to bake the prefix into their dicts.
def _normalize_joint_name(name: str) -> str:
    """Strip Mixamo's namespace prefix and any leading colon."""
    if not name:
        return ""
    n = name
    # Drop common prefixes seen in the wild.
    for prefix in ("mixamorig:", "mixamorig1:", "Armature_", "rig_"):
        if n.startswith(prefix):
            n = n[len(prefix):]
            break
    return n


# ---------------------------------------------------------------------------
# Bone-role auto-detection
# ---------------------------------------------------------------------------
#
# Returns a CANONICAL role string keyed on the existing
# LOBBY_GIRL_BONE_MAP entries:
#   Hips, Spine, Spine1, Spine2, Neck, Head,
#   LeftShoulder, LeftArm, LeftForeArm, LeftHand,
#   RightShoulder, RightArm, RightForeArm, RightHand,
#   LeftUpLeg, LeftLeg, LeftFoot,
#   RightUpLeg, RightLeg, RightFoot.
#
# Detection is conservative: when the heuristic isn't certain, return
# None and let the retargeter drop the bone (with a diagnostic in
# ``summarize_retarget(...)["dropped"]``). False positives are worse
# than misses — a wrong route flings the wrist to the wrong shoulder.


# VRM humanoid bone roles → our canonical role names.
# VRM uses lowercase camelCase ("leftUpperArm"). Reference:
#   https://github.com/vrm-c/vrm-specification/tree/master/specification/0.0
#   https://github.com/vrm-c/vrm-specification/tree/master/specification/VRMC_vrm-1.0
# We map the standard 21 humanoid roles (hips through both arms+legs+head).
# ``upperChest`` is a VRM-specific extra spine bone — we route it to
# LOBBY_GIRL's single Spine bone. ``leftToes``/``rightToes`` we drop
# silently (lobby_girl has no toe bones).
_VRM_ROLE_TO_CANONICAL: Dict[str, str] = {
    "hips":            "Hips",
    "spine":           "Spine",
    "chest":           "Spine1",
    "upperChest":      "Spine2",
    "neck":            "Neck",
    "head":            "Head",
    "leftShoulder":    "LeftShoulder",
    "leftUpperArm":    "LeftArm",
    "leftLowerArm":    "LeftForeArm",
    "leftHand":        "LeftHand",
    "rightShoulder":   "RightShoulder",
    "rightUpperArm":   "RightArm",
    "rightLowerArm":   "RightForeArm",
    "rightHand":       "RightHand",
    "leftUpperLeg":    "LeftUpLeg",
    "leftLowerLeg":    "LeftLeg",
    "leftFoot":        "LeftFoot",
    "rightUpperLeg":   "RightUpLeg",
    "rightLowerLeg":   "RightLeg",
    "rightFoot":       "RightFoot",
}


# Cesium / Khronos sample-asset joint patterns. Captures the chain
# index from names like "Skeleton_arm_joint_R__2_". Indexed:
#   _1_ or no suffix = base (shoulder for arm, hip for leg)
#   _2_ = upper arm / thigh-knee
#   _3_ = forearm / shin
#   _4_ = hand / placeholder (sometimes empty)
#   _5_ = toe / foot
_CESIUM_ARM_RE = re.compile(r"^Skeleton_arm_joint_([LR])(?:__(\d+)_)?$")
_CESIUM_LEG_RE = re.compile(r"^(?:Skeleton_)?leg_joint_([LR])(?:_(\d+))?$")
_CESIUM_TORSO_RE = re.compile(r"^(?:Skeleton_)?torso_joint_(\d+)$")
_CESIUM_NECK_RE = re.compile(r"^(?:Skeleton_)?neck_joint_(\d+)$")

_CESIUM_ARM_INDEX_TO_ROLE = {
    None: "Shoulder", "1": "Shoulder",
    "2": "Arm",
    "3": "ForeArm",
    "4": "Hand",
}
_CESIUM_LEG_INDEX_TO_ROLE = {
    None: "UpLeg", "1": "UpLeg",
    "2": "Leg",
    "3": "Leg",      # _3_ is shin in CesiumMan; lobby_girl has only one knee bone
    "4": "Leg",      # placeholder
    "5": "Foot",
}


# Unity Mecanim canonical names (humanoid avatar). Direct alias map:
#   the Mecanim names are PascalCase camelCase that mostly already match
#   our canonical except for "Upper"/"Lower" wording vs "Arm"/"ForeArm".
_MECANIM_TO_CANONICAL: Dict[str, str] = {
    # Spine + head
    "Hips":         "Hips",
    "Spine":        "Spine",
    "Chest":        "Spine1",
    "UpperChest":   "Spine2",
    "Neck":         "Neck",
    "Head":         "Head",
    # Arms — Mecanim says "UpperArm/LowerArm", Mixamo says "Arm/ForeArm"
    "LeftShoulder":  "LeftShoulder",
    "LeftUpperArm":  "LeftArm",
    "LeftLowerArm":  "LeftForeArm",
    "LeftHand":      "LeftHand",
    "RightShoulder": "RightShoulder",
    "RightUpperArm": "RightArm",
    "RightLowerArm": "RightForeArm",
    "RightHand":     "RightHand",
    # Legs
    "LeftUpperLeg":  "LeftUpLeg",
    "LeftLowerLeg":  "LeftLeg",
    "LeftFoot":      "LeftFoot",
    "RightUpperLeg": "RightUpLeg",
    "RightLowerLeg": "RightLeg",
    "RightFoot":     "RightFoot",
    # HumanIK / Mixamo canonical (already match our role names; kept here
    # so the auto-detect path covers them too — detection becomes a
    # one-table O(1) lookup instead of needing a fall-through case).
    "LeftArm":       "LeftArm",
    "LeftForeArm":   "LeftForeArm",
    "RightArm":      "RightArm",
    "RightForeArm":  "RightForeArm",
    "LeftUpLeg":     "LeftUpLeg",
    "LeftLeg":       "LeftLeg",
    "RightUpLeg":    "RightUpLeg",
    "RightLeg":      "RightLeg",
    "Spine1":        "Spine1",
    "Spine2":        "Spine2",
}


# Generic body-part substring keywords (case-insensitive, after suffix
# strip). Matched against a normalised string with no separators (so
# "Left Arm", "left_arm", "LeftArm", "left.arm" all collapse to "leftarm").
# Order matters: more specific keys first so "leftforearm" doesn't
# match "leftarm" by accident.
_GENERIC_KEYWORDS: List[Tuple[str, str]] = [
    # Arms — most specific first.
    ("leftforearm",   "LeftForeArm"),
    ("leftlowerarm",  "LeftForeArm"),
    ("leftshoulder",  "LeftShoulder"),
    ("leftupperarm",  "LeftArm"),
    ("leftarm",       "LeftArm"),
    ("lefthand",      "LeftHand"),
    ("rightforearm",  "RightForeArm"),
    ("rightlowerarm", "RightForeArm"),
    ("rightshoulder", "RightShoulder"),
    ("rightupperarm", "RightArm"),
    ("rightarm",      "RightArm"),
    ("righthand",     "RightHand"),
    # Legs.
    ("leftupperleg",  "LeftUpLeg"),
    ("leftupleg",     "LeftUpLeg"),
    ("leftthigh",     "LeftUpLeg"),
    ("leftlowerleg",  "LeftLeg"),
    ("leftshin",      "LeftLeg"),
    ("leftcalf",      "LeftLeg"),
    ("leftleg",       "LeftLeg"),     # last to avoid eating UpperLeg
    ("leftfoot",      "LeftFoot"),
    ("leftankle",     "LeftFoot"),
    ("rightupperleg", "RightUpLeg"),
    ("rightupleg",    "RightUpLeg"),
    ("rightthigh",    "RightUpLeg"),
    ("rightlowerleg", "RightLeg"),
    ("rightshin",     "RightLeg"),
    ("rightcalf",     "RightLeg"),
    ("rightleg",      "RightLeg"),
    ("rightfoot",     "RightFoot"),
    ("rightankle",    "RightFoot"),
    # Torso + head.
    ("upperchest",    "Spine2"),
    ("chest",         "Spine1"),
    ("spine2",        "Spine2"),
    ("spine1",        "Spine1"),
    ("spine",         "Spine"),
    ("torso",         "Spine"),
    ("hips",          "Hips"),
    ("hip",           "Hips"),
    ("pelvis",        "Hips"),
    ("waist",         "Hips"),
    ("neck",          "Neck"),
    ("head",          "Head"),
]


def _strip_side_suffix(name: str) -> Tuple[str, Optional[str]]:
    """Return (base, side) where side is "L" / "R" or None.

    Recognises Blender Rigify (``.L``/``.R``), MakeHuman/Mecanim
    (``_L``/``_R``/``_l``/``_r``), and a trailing standalone L/R after
    a separator. Case-insensitive on the side letter.
    """
    if not name:
        return name, None
    # Match ".L" / ".R" / "_L" / "_l" / "_R" / "_r" suffixes.
    m = re.search(r"[._-]([LlRr])$", name)
    if m:
        side = m.group(1).upper()
        return name[:m.start()], side
    # Match ".left" / "_right" full-word suffix (some rigs).
    m = re.search(r"[._-](left|right)$", name, re.IGNORECASE)
    if m:
        side = "L" if m.group(1).lower() == "left" else "R"
        return name[:m.start()], side
    return name, None


def _strip_index_suffix(name: str) -> Tuple[str, Optional[int]]:
    """Strip a numeric suffix like ``.001`` / ``_01`` and return (base, idx)."""
    if not name:
        return name, None
    m = re.search(r"[._-](\d+)$", name)
    if m:
        try:
            return name[:m.start()], int(m.group(1))
        except ValueError:
            pass
    return name, None


def auto_detect_bone_role(bone_name: str) -> Optional[str]:
    """Heuristically detect the canonical retarget role for a source bone.

    Returns one of the canonical role strings used as keys in
    ``LOBBY_GIRL_BONE_MAP`` (e.g. ``"Hips"``, ``"LeftArm"``,
    ``"RightForeArm"``) when the bone-name matches a known rig
    convention. Returns ``None`` when the heuristic isn't confident.

    Detection priority (first match wins):
      1. Trivial: name is already canonical (in ``_MECANIM_TO_CANONICAL``).
      2. VRM lowercase camelCase (``leftUpperArm`` → ``LeftArm``).
      3. Cesium / Khronos sample assets (regex match).
      4. Blender Rigify ``.L``/``.R`` suffix + body-part word.
      5. Underscore-suffix conventions (``upperarm_l`` → ``LeftArm``).
      6. Generic substring (case-insensitive, separator-stripped).

    The function is INTENTIONALLY conservative — when in doubt it
    returns None so the retargeter falls through to "drop with
    diagnostic". A wrong role is worse than no route.

    Examples
    --------
    >>> auto_detect_bone_role("LeftArm")
    'LeftArm'
    >>> auto_detect_bone_role("leftUpperArm")
    'LeftArm'
    >>> auto_detect_bone_role("Skeleton_arm_joint_R__2_")
    'RightArm'
    >>> auto_detect_bone_role("upper_arm.L")
    'LeftArm'
    >>> auto_detect_bone_role("UpperArm_R")
    'RightArm'
    >>> auto_detect_bone_role("right shoulder")
    'RightShoulder'
    >>> auto_detect_bone_role("RandomBoneName") is None
    True
    """
    if not bone_name:
        return None
    raw = _normalize_joint_name(bone_name)
    if not raw:
        return None

    # 1. Direct canonical / Mecanim alias hit.
    role = _MECANIM_TO_CANONICAL.get(raw)
    if role:
        return role

    # 2. VRM lowercase camelCase. Direct table hit on raw.
    role = _VRM_ROLE_TO_CANONICAL.get(raw)
    if role:
        return role

    # 3. Cesium / Khronos sample-asset patterns.
    m = _CESIUM_ARM_RE.match(raw)
    if m:
        side, idx_str = m.group(1), m.group(2)
        side_word = "Left" if side == "L" else "Right"
        suffix = _CESIUM_ARM_INDEX_TO_ROLE.get(idx_str)
        if suffix is not None:
            return side_word + suffix
    m = _CESIUM_LEG_RE.match(raw)
    if m:
        side, idx_str = m.group(1), m.group(2)
        side_word = "Left" if side == "L" else "Right"
        suffix = _CESIUM_LEG_INDEX_TO_ROLE.get(idx_str)
        if suffix is not None:
            return side_word + suffix
    m = _CESIUM_TORSO_RE.match(raw)
    if m:
        idx = int(m.group(1))
        if idx == 1:
            return "Hips"
        if idx == 2:
            return "Spine"
        if idx == 3:
            return "Spine1"
    m = _CESIUM_NECK_RE.match(raw)
    if m:
        idx = int(m.group(1))
        return "Neck" if idx == 1 else "Head"

    # 4 + 5. Strip a trailing numeric suffix first (Blender duplicates
    # bones with ``.001`` / ``.002`` suffixes after import) THEN strip
    # the side suffix. Order matters: ``upper_arm.L.001`` needs the
    # numeric tail gone before ``.L`` is at the trailing position the
    # side regex can match.
    base, _idx = _strip_index_suffix(raw)
    base, side = _strip_side_suffix(base)
    base, _idx2 = _strip_index_suffix(base)

    # Try the suffix-stripped base against Mecanim / VRM tables (e.g.
    # "Hand" matches "LeftHand" when side=L).
    if side is not None:
        side_word = "Left" if side == "L" else "Right"
        # Direct base-word lookup against the canonical rolls.
        base_lower = base.lower().replace(" ", "").replace("-", "").replace("_", "").replace(".", "")
        # Body-part word table without side prefix.
        _BARE_BASE_TO_SUFFIX: Dict[str, str] = {
            "arm":         "Arm",
            "upperarm":    "Arm",
            "shoulder":    "Shoulder",
            "clavicle":    "Shoulder",
            "lowerarm":    "ForeArm",
            "forearm":     "ForeArm",
            "elbow":       "ForeArm",
            "hand":        "Hand",
            "wrist":       "Hand",
            "leg":         "Leg",
            "lowerleg":    "Leg",
            "shin":        "Leg",
            "calf":        "Leg",
            "knee":        "Leg",
            "upperleg":    "UpLeg",
            "upleg":       "UpLeg",
            "thigh":       "UpLeg",
            "foot":        "Foot",
            "ankle":       "Foot",
        }
        suffix = _BARE_BASE_TO_SUFFIX.get(base_lower)
        if suffix:
            return side_word + suffix

    # 6. Generic substring keyword match. Strip separators + lowercase.
    # We do this on the ORIGINAL (post-prefix-strip) name so multi-word
    # inputs like "Right Shoulder" / "right_arm" / "right.arm" collapse
    # to "rightshoulder" / "rightarm".
    canon = re.sub(r"[\s._\-:]", "", raw).lower()
    for keyword, mapped_role in _GENERIC_KEYWORDS:
        if keyword in canon:
            return mapped_role

    return None


def get_builtin_bone_map(name: str) -> BoneNameMap:
    """Return one of the bundled bone-name -> PSOBB-index maps.

    Currently provides:
      * ``"lobby_girl"`` — for the 64-bone NPC humanoid skeleton used
        by every ``bm_npc_*.bml`` file.

    Raises KeyError if ``name`` is unknown.
    """
    if name == "lobby_girl":
        return dict(LOBBY_GIRL_BONE_MAP)
    raise KeyError(f"unknown bone map: {name!r}")


# ---------------------------------------------------------------------------
# IK chain definitions
# ---------------------------------------------------------------------------
#
# An ``IkChainSpec`` describes one end-effector chain to retarget via IK.
# The chain is identified by the SOURCE bone names (Mixamo joint names)
# from root → end. Resolution to TARGET bone indices happens via the
# bone_map.
#
# Conventional humanoid chains:
#   * Right arm:  RightShoulder → RightArm → RightForeArm → RightHand
#   * Left arm:   LeftShoulder  → LeftArm  → LeftForeArm  → LeftHand
#   * Right leg:  RightUpLeg    → RightLeg → RightFoot
#   * Left leg:   LeftUpLeg     → LeftLeg  → LeftFoot
#
# Each chain runs its own FABRIK solve; chains run independently so
# bone overlap (e.g. shoulder shared with spine) doesn't cascade.

@dataclass(frozen=True)
class IkChainSpec:
    """One IK chain: a list of source-bone NAMES from root to end-effector.

    The chain lookup happens at retarget time by joining the source
    skeleton's bone names + the bone_map dict to resolve target indices.
    Names that don't resolve are silently dropped, so a partial map
    (e.g. only the right arm in the bone_map) won't fail.

    ``pole_bone_name`` (v3, 2026-04-25) is an optional source-bone name
    used as a pole-vector hint for FABRIK. If the elbow/knee bend ends
    up on the wrong side of the chain (mirrored through the start↔end
    axis), the IK pass mirrors the chain back so the bend points toward
    the pole. If None, no pole correction runs (legacy v2 behaviour).

    ``pole_axis`` is a fallback direction (in world space, expressed in
    the target skeleton's frame) used when ``pole_bone_name`` doesn't
    resolve. Defaults to (0, 0, 1) — Z-forward, away from the body for
    a humanoid in T-pose. Set to None to fully disable pole correction
    even without a named bone.
    """
    name: str
    bone_names: Tuple[str, ...]
    enabled: bool = True
    pole_bone_name: Optional[str] = None
    pole_axis: Optional[Tuple[float, float, float]] = (0.0, 0.0, 1.0)


# Default chain set for the lobby_girl skeleton (and any other Mixamo-
# based humanoid retarget). These names are what Mixamo emits AFTER the
# ``mixamorig:`` prefix has been stripped, matching ``LOBBY_GIRL_BONE_MAP``.
#
# v3 (2026-04-25): pole-vector hints. Arms get the SHOULDER bone as the
# pole anchor — the elbow should bend roughly toward the shoulder's
# Z-forward direction (humanoid arms swing forward from the shoulder
# in T-pose, so "shoulder Z" is a stable hint regardless of arm pose).
# Legs get the hip as the pole anchor for the same reason.
HUMANOID_IK_CHAINS: Tuple[IkChainSpec, ...] = (
    IkChainSpec("right_arm",
                ("RightShoulder", "RightArm", "RightForeArm", "RightHand"),
                pole_bone_name="RightShoulder"),
    IkChainSpec("left_arm",
                ("LeftShoulder",  "LeftArm",  "LeftForeArm",  "LeftHand"),
                pole_bone_name="LeftShoulder"),
    IkChainSpec("right_leg",
                ("RightUpLeg",    "RightLeg", "RightFoot"),
                pole_bone_name="RightUpLeg"),
    IkChainSpec("left_leg",
                ("LeftUpLeg",     "LeftLeg",  "LeftFoot"),
                pole_bone_name="LeftUpLeg"),
)


# ---------------------------------------------------------------------------
# IK chain auto-inference (v4, 2026-04-25)
# ---------------------------------------------------------------------------
#
# When the source skeleton uses a bone-naming convention ``HUMANOID_IK_CHAINS``
# doesn't cover (the chain's ``bone_names`` won't resolve via
# ``bone_map``), we want to fall back to inferring the chains directly
# from the rig topology.
#
# Strategy:
#   1. Detect end-effector roles via ``auto_detect_bone_role`` (or a
#      caller-supplied bone_role_map).
#   2. From each end-effector, walk the parent chain UP the skeleton
#      until we either hit the chain length we expect (3 for legs,
#      4 for arms) or run out of parents.
#   3. The deepest chain bone the walk visits becomes the chain root.
#   4. Pole bone defaults to the chain root (matches HUMANOID_IK_CHAINS'
#      shoulder/hip convention); pole axis is derived from the bend
#      direction in the source bind pose (perpendicular to the
#      start↔end axis through the middle joint).
#
# The output ``IkChainSpec`` uses ROLE names (e.g. ``"RightHand"``) for
# bone_names — same as ``HUMANOID_IK_CHAINS``. That way the chain
# resolves through whichever bone_map / VRM map the caller has wired up.


_INFER_CHAIN_END_ROLES: Tuple[Tuple[str, str, int], ...] = (
    # (end_effector_role, chain_short_name, expected_length)
    ("RightHand", "right_arm", 4),
    ("LeftHand",  "left_arm",  4),
    ("RightFoot", "right_leg", 3),
    ("LeftFoot",  "left_leg",  3),
)


def _bone_role_for_inference(
    bone_name: str,
    bone_role_map: Optional[Dict[str, str]],
) -> Optional[str]:
    """Resolve ``bone_name`` -> canonical role.

    If ``bone_role_map`` carries an explicit entry, that takes
    precedence (lets the caller override e.g. a misnamed VRoid bone);
    otherwise we fall back to ``auto_detect_bone_role``.
    """
    if bone_role_map is not None:
        # Lookup tolerant of casing / whitespace differences.
        role = bone_role_map.get(bone_name)
        if role:
            return role
        # Try the normalised name as a backup key.
        norm = _normalize_joint_name(bone_name)
        if norm:
            for k, v in bone_role_map.items():
                if _normalize_joint_name(k) == norm:
                    return v
    return auto_detect_bone_role(bone_name)


def _walk_parent_chain(
    skeleton: Sequence,
    end_idx: int,
    max_len: int,
) -> List[int]:
    """Walk from ``end_idx`` up parents, collecting up to ``max_len`` bones.

    Returns a list ordered ROOT → END (so the natural FABRIK input
    order). Stops when:
      * the chain reaches ``max_len`` bones, OR
      * we run out of parents (parent_idx < 0).

    Detects cycles (a malformed rig where a parent eventually points
    back to a descendant) by giving up after ``len(skeleton)`` hops.
    """
    chain: List[int] = []
    cur = int(end_idx)
    seen: set = set()
    n = len(skeleton)
    for _ in range(n + 1):
        if cur < 0 or cur >= n or cur in seen:
            break
        seen.add(cur)
        chain.append(cur)
        if len(chain) >= max_len:
            break
        try:
            parent = int(skeleton[cur].parent_idx)
        except (AttributeError, ValueError, TypeError):
            break
        if parent < 0:
            break
        cur = parent
    chain.reverse()
    return chain


def _infer_pole_axis(
    skeleton: Sequence,
    chain_indices: List[int],
) -> Tuple[float, float, float]:
    """Derive a default pole-vector axis from the source bind pose.

    The middle joint's perpendicular displacement from the start→end
    axis indicates which way the elbow / knee bends. The unit vector of
    that perpendicular is the pole hint — it's stable per-rig as long
    as the bind pose isn't fully straight.

    For straight chains (or chains shorter than 3) we fall back to
    ``(0, 0, 1)`` (Z-forward), matching ``HUMANOID_IK_CHAINS``.
    """
    if len(chain_indices) < 3:
        return (0.0, 0.0, 1.0)
    # Compute world bind positions by accumulating bind_pos along parents.
    world: Dict[int, Tuple[float, float, float]] = {}
    n = len(skeleton)
    for i in range(n):
        bone = skeleton[i]
        try:
            parent = int(bone.parent_idx)
        except (AttributeError, TypeError):
            parent = -1
        bp = tuple(float(x) for x in (
            getattr(bone, "bind_pos", None)
            or getattr(bone, "translation", None)
            or (0.0, 0.0, 0.0)
        ))[:3]
        if parent < 0 or parent >= n or parent not in world:
            world[i] = bp
        else:
            par = world[parent]
            world[i] = (par[0] + bp[0], par[1] + bp[1], par[2] + bp[2])
    start = world.get(chain_indices[0])
    mid = world.get(chain_indices[1])
    end = world.get(chain_indices[-1])
    if start is None or mid is None or end is None:
        return (0.0, 0.0, 1.0)
    ax = end[0] - start[0]; ay = end[1] - start[1]; az = end[2] - start[2]
    a_len = math.sqrt(ax * ax + ay * ay + az * az)
    if a_len < 1e-6:
        return (0.0, 0.0, 1.0)
    ax /= a_len; ay /= a_len; az /= a_len
    mx = mid[0] - start[0]; my = mid[1] - start[1]; mz = mid[2] - start[2]
    proj = mx * ax + my * ay + mz * az
    perpx = mx - proj * ax
    perpy = my - proj * ay
    perpz = mz - proj * az
    perp_len = math.sqrt(perpx * perpx + perpy * perpy + perpz * perpz)
    if perp_len < 1e-6:
        # Bind pose is straight; nothing to infer. Z-forward fallback.
        return (0.0, 0.0, 1.0)
    return (perpx / perp_len, perpy / perp_len, perpz / perp_len)


def infer_ik_chains_from_skeleton(
    skeleton: Sequence,
    bone_role_map: Optional[Dict[str, str]] = None,
) -> List[IkChainSpec]:
    """Auto-build ``IkChainSpec`` entries by walking parent chains.

    Used as a FALLBACK when the source's bone-naming convention isn't
    covered by ``HUMANOID_IK_CHAINS`` directly. The returned chains use
    ROLE names (``"RightHand"`` etc.) for ``bone_names`` so they
    resolve via the same ``bone_map`` the rest of the retarget pipeline
    uses.

    Args
    ----
    skeleton:
        The source skeleton (list of ``ImportedBone`` or any object
        carrying ``name`` + ``parent_idx`` + optional ``bind_pos``).
    bone_role_map:
        Optional ``{bone_name: canonical_role}`` override. Empty / None
        falls back to ``auto_detect_bone_role`` for every bone name.

    Returns
    -------
    list[IkChainSpec]
        One spec per detectable chain (right_arm, left_arm, right_leg,
        left_leg). Specs are skipped silently when the corresponding
        end-effector isn't found OR the parent walk produces fewer than
        2 bones (FABRIK needs ≥ 2). The list is empty when nothing maps.

    Notes
    -----
    The function NEVER mutates the input. Multiple bones with the same
    role: we take the first match, which is consistent with how
    ``LOBBY_GIRL_BONE_MAP`` already collapses duplicates.
    """
    n = len(skeleton)
    if n == 0:
        return []
    # 1) Bone idx for each canonical role we care about (end effector
    #    + chain anchors).
    role_to_bone: Dict[str, int] = {}
    for i in range(n):
        bone = skeleton[i]
        nm = getattr(bone, "name", "") or ""
        role = _bone_role_for_inference(nm, bone_role_map)
        if role and role not in role_to_bone:
            role_to_bone[role] = i
    out: List[IkChainSpec] = []
    # Map idx -> role lookup so we can label the inferred chain bones.
    bone_to_role: Dict[int, str] = {v: k for k, v in role_to_bone.items()}
    for end_role, chain_name, expected_len in _INFER_CHAIN_END_ROLES:
        end_idx = role_to_bone.get(end_role)
        if end_idx is None:
            continue
        chain_idxs = _walk_parent_chain(skeleton, end_idx, expected_len)
        if len(chain_idxs) < 2:
            continue
        # Label each chain bone with whatever role we have, falling back
        # to a synthetic ``inferred_<idx>`` name. Synthetic names won't
        # resolve in any bone_map but they keep the chain length right
        # for FABRIK; the resolver in ``_resolve_ik_chains_for_targets``
        # drops unresolved entries gracefully.
        bone_names: List[str] = []
        for idx in chain_idxs:
            role = bone_to_role.get(idx)
            if role:
                bone_names.append(role)
            else:
                # Use the source bone NAME as a hopeful fallback — many
                # bone_maps include the name verbatim.
                src_name = getattr(skeleton[idx], "name", "") or f"inferred_{idx}"
                bone_names.append(src_name)
        # Pole bone: chain root (mirrors HUMANOID_IK_CHAINS' shoulder/hip).
        pole_role = bone_names[0] if bone_names else None
        pole_axis = _infer_pole_axis(skeleton, chain_idxs)
        out.append(IkChainSpec(
            name=chain_name,
            bone_names=tuple(bone_names),
            pole_bone_name=pole_role,
            pole_axis=pole_axis,
        ))
    return out


# ---------------------------------------------------------------------------
# Quaternion / interpolation helpers
# ---------------------------------------------------------------------------


def _quat_normalize(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    qx, qy, qz, qw = q
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return (qx / n, qy / n, qz / n, qw / n)


def _quat_dot(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


def _quat_slerp(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    t: float,
) -> Tuple[float, float, float, float]:
    """Spherical linear interpolation between two unit quaternions.

    Falls back to lerp + renormalise when the two quats are very
    close (avoids numerical noise from acos(near-1)).
    """
    a = _quat_normalize(a)
    b = _quat_normalize(b)
    d = _quat_dot(a, b)
    if d < 0:
        b = (-b[0], -b[1], -b[2], -b[3])
        d = -d
    if d > 0.9995:
        # Linear-blend + renormalise.
        rx = a[0] + t * (b[0] - a[0])
        ry = a[1] + t * (b[1] - a[1])
        rz = a[2] + t * (b[2] - a[2])
        rw = a[3] + t * (b[3] - a[3])
        return _quat_normalize((rx, ry, rz, rw))
    theta_0 = math.acos(min(1.0, max(-1.0, d)))
    theta = theta_0 * t
    sin_theta_0 = math.sin(theta_0)
    if sin_theta_0 < 1e-9:
        return a
    s_a = math.sin(theta_0 - theta) / sin_theta_0
    s_b = math.sin(theta) / sin_theta_0
    return (
        s_a * a[0] + s_b * b[0],
        s_a * a[1] + s_b * b[1],
        s_a * a[2] + s_b * b[2],
        s_a * a[3] + s_b * b[3],
    )


def _quat_mul(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Hamilton product ``a * b`` (each quat as (x, y, z, w))."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quat_inverse(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Inverse (= conjugate, for unit quats) of ``q``."""
    q = _quat_normalize(q)
    return (-q[0], -q[1], -q[2], q[3])


def _mirror_quat_z(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Reflect a rotation through the Z=0 plane.

    This converts a glTF (right-handed, +Z forward) rotation into
    PSOBB (left-handed, -Z forward). Algebra: reflecting an axis
    through a plane is equivalent to negating the components of the
    rotation that LIE in that plane and the rotation angle's sign.

    For a rotation r=(qx, qy, qz, qw), the Z-mirrored rotation is
    (qx, qy, -qz, qw)... but the sign of the rotation angle around
    that axis ALSO flips, which is captured by also negating qw.
    Combined: ``(qx, qy, -qz, -qw) ≡ -(−qx, −qy, qz, qw)``.

    Empirical sanity check: a +90° Yaw quat (0, sin45, 0, cos45)
    becomes a -90° Yaw quat after the flip — which matches the
    visual mirror through the screen plane.
    """
    return (q[0], q[1], -q[2], -q[3])


def _vec_lerp(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    t: float,
) -> Tuple[float, float, float]:
    return (
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    )


def _resample_track(
    track: ImportedTrack,
    target_times: Sequence[float],
) -> List[tuple]:
    """Resample a track at the given list of target times.

    For rotation tracks this slerps; for translation/scale it lerps.
    Out-of-range target times clamp to the source endpoints.

    Returns a list of value tuples in the same order as
    ``target_times``.
    """
    if not track.times:
        return []
    src_times = track.times
    src_vals = track.values
    n = len(src_times)
    out: List[tuple] = []
    cursor = 0  # last src index we examined; src_times is monotonic
    for t in target_times:
        if t <= src_times[0]:
            out.append(src_vals[0])
            continue
        if t >= src_times[n - 1]:
            out.append(src_vals[n - 1])
            continue
        # Advance cursor.
        while cursor + 1 < n and src_times[cursor + 1] < t:
            cursor += 1
        t0 = src_times[cursor]
        t1 = src_times[cursor + 1]
        v0 = src_vals[cursor]
        v1 = src_vals[cursor + 1]
        if t1 - t0 <= 1e-12:
            out.append(v0)
            continue
        a = (t - t0) / (t1 - t0)
        if track.channel == "rotation":
            out.append(_quat_slerp(v0, v1, a))
        else:
            out.append(_vec_lerp(v0, v1, a))
    return out


# ---------------------------------------------------------------------------
# Target-bind retrieval helpers
# ---------------------------------------------------------------------------
#
# We accept either:
#   * ``XjBone`` (formats.xj) — ``rotation`` is BAMS (int triple),
#     ``position`` is float vec3, ``scale`` is float vec3.
#   * ``ImportedBone`` (formats.import_external) — ``bind_rot_quat``
#     is float quat, ``bind_pos`` is vec3.
# The retargeter handles both. Detection is structural — we sniff
# ``rotation`` (BAMS) vs ``bind_rot_quat`` (quat).


def _target_bind_quat(bone) -> Tuple[float, float, float, float]:
    """Return the bind-pose rotation as a (qx, qy, qz, qw) quaternion."""
    if hasattr(bone, "bind_rot_quat"):
        q = tuple(bone.bind_rot_quat)
        return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    # XjBone path: BAMS triple in (rx, ry, rz) order; we convert ZYX
    # Euler -> quat. Sega Ninja's ZYX intrinsic order matches
    # quat_to_zyx_bams, so we invert: build R = Rz @ Ry @ Rx and
    # decompose to a quaternion.
    rx_b, ry_b, rz_b = bone.rotation
    rx = (rx_b if rx_b < 0x8000 else rx_b - 0x10000) * (math.pi * 2.0 / 0x10000)
    ry = (ry_b if ry_b < 0x8000 else ry_b - 0x10000) * (math.pi * 2.0 / 0x10000)
    rz = (rz_b if rz_b < 0x8000 else rz_b - 0x10000) * (math.pi * 2.0 / 0x10000)
    # Quat from ZYX Euler: q = qz * qy * qx (extrinsic Z then Y then X
    # is the same as intrinsic Z*Y*X applied to a vector).
    cx, sx = math.cos(rx * 0.5), math.sin(rx * 0.5)
    cy, sy = math.cos(ry * 0.5), math.sin(ry * 0.5)
    cz, sz = math.cos(rz * 0.5), math.sin(rz * 0.5)
    # Standard ZYX Euler -> quat formulas:
    qw = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz
    return (qx, qy, qz, qw)


def _target_bone_count(target_skeleton: Sequence) -> int:
    return len(target_skeleton)


# ---------------------------------------------------------------------------
# Forward kinematics — compute world-space bone positions from a pose
# ---------------------------------------------------------------------------
#
# We need this for both the SOURCE side (to know where Mixamo's wrist
# wants to land in world coords) and the TARGET side (to know where
# the 1:1 quat copy already put the wrist before IK fixup).
#
# The math is intentionally minimal — just enough to read joint
# positions out. We don't reuse rigging.compose_world_matrices because
# it's hard-wired to BAMS/BonePose; here we keep everything in float
# quaternion space.


def _quat_to_mat3(q: Tuple[float, float, float, float]) -> Tuple[float, ...]:
    """Build the 3x3 rotation matrix (row-major) for unit quat ``q``."""
    qx, qy, qz, qw = _quat_normalize(q)
    xx = qx * qx; yy = qy * qy; zz = qz * qz
    xy = qx * qy; xz = qx * qz; yz = qy * qz
    wx = qw * qx; wy = qw * qy; wz = qw * qz
    return (
        1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy),
        2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
        2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy),
    )


def _mat3_mul_vec(m: Tuple[float, ...], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    )


def _mat3_mul(a: Tuple[float, ...], b: Tuple[float, ...]) -> Tuple[float, ...]:
    return (
        a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
        a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
        a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
        a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
        a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
        a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
        a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
        a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
        a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
    )


def _bone_parents(skeleton: Sequence) -> List[int]:
    """Extract a list of parent indices from either ImportedBone or XjBone."""
    parents: List[int] = []
    for b in skeleton:
        if hasattr(b, "parent_idx"):
            parents.append(int(b.parent_idx))
        elif hasattr(b, "parent"):
            parents.append(int(b.parent))
        else:
            parents.append(-1)
    return parents


def _bone_position(b) -> Tuple[float, float, float]:
    """Read the bind-pose translation from an ImportedBone or XjBone."""
    if hasattr(b, "bind_pos"):
        p = b.bind_pos
    elif hasattr(b, "position"):
        p = b.position
    else:
        return (0.0, 0.0, 0.0)
    return (float(p[0]), float(p[1]), float(p[2]))


def _forward_kinematics(
    skeleton: Sequence,
    local_quats: List[Tuple[float, float, float, float]],
) -> List[Tuple[Tuple[float, float, float], Tuple[float, ...]]]:
    """Compute (world_pos, world_rot_mat) for every bone.

    The skeleton's bind translations are interpreted in the parent's
    LOCAL frame (the standard Sega Ninja / glTF convention). For each
    bone we compose:

        world_rot[i]  = world_rot[parent] @ local_quat[i]_as_mat
        world_pos[i]  = world_rot[parent] @ bone_local_pos + world_pos[parent]

    For root bones, the parent's world is identity. We return matrices
    rather than quaternions for the rotation channel because per-frame
    multiplication of 3x3s is cheap and avoids quaternion sign-flipping
    bugs at hierarchy junctions.
    """
    n = len(skeleton)
    parents = _bone_parents(skeleton)
    out: List[Tuple[Tuple[float, float, float], Tuple[float, ...]]] = []
    identity_mat3 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    for i, bone in enumerate(skeleton):
        local_p = _bone_position(bone)
        local_r = _quat_to_mat3(local_quats[i] if i < len(local_quats) else (0.0, 0.0, 0.0, 1.0))
        p = parents[i] if i < len(parents) else -1
        if p < 0 or p >= i or p >= len(out):
            world_p = local_p
            world_r = local_r
        else:
            par_p, par_r = out[p]
            tx, ty, tz = _mat3_mul_vec(par_r, local_p)
            world_p = (par_p[0] + tx, par_p[1] + ty, par_p[2] + tz)
            world_r = _mat3_mul(par_r, local_r)
        out.append((world_p, world_r))
        # We don't currently need ``identity_mat3`` but keep the binding
        # so a mypy-annotated future refactor doesn't lose the constant.
        _ = identity_mat3
    return out


def _bind_quat_for_source(b: ImportedBone) -> Tuple[float, float, float, float]:
    """Return the ImportedBone's bind quaternion in (x, y, z, w) order."""
    q = tuple(b.bind_rot_quat)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _solve_chain_to_target(
    chain_world_positions: List[Tuple[float, float, float]],
    target_world: Tuple[float, float, float],
    *,
    iterations: int = 16,
    tol: float = 1e-3,
) -> List[Tuple[float, float, float]]:
    """Wrapper around ``rigging.fabrik_solve``.

    Wraps the rigging.py FABRIK solver so this module's call sites
    don't have to know about that module's exact return shape; also
    keeps a single seam if we want to swap solvers in v3.
    """
    return list(_rigging_fabrik_solve(
        chain_world_positions, target_world,
        iterations=iterations, tol=tol,
    ))


def _rotation_to_align(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
) -> Tuple[float, float, float, float]:
    """Quaternion that rotates direction vector ``a`` to direction ``b``.

    Both inputs need not be unit length; the function normalises. Used
    by the IK feedback step to derive the per-bone local rotation that
    points each segment along the FABRIK-solved direction.
    """
    ax, ay, az = a; bx, by, bz = b
    al = math.sqrt(ax * ax + ay * ay + az * az)
    bl = math.sqrt(bx * bx + by * by + bz * bz)
    if al < 1e-9 or bl < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    ax /= al; ay /= al; az /= al
    bx /= bl; by /= bl; bz /= bl
    d = ax * bx + ay * by + az * bz
    if d > 0.99999:
        return (0.0, 0.0, 0.0, 1.0)
    if d < -0.99999:
        # 180° about an axis perpendicular to ``a``.
        if abs(ax) < 0.9:
            ux, uy, uz = 1.0, 0.0, 0.0
        else:
            ux, uy, uz = 0.0, 1.0, 0.0
        # Cross to make perpendicular.
        cx = ay * uz - az * uy
        cy = az * ux - ax * uz
        cz = ax * uy - ay * ux
        cl = math.sqrt(cx * cx + cy * cy + cz * cz) or 1.0
        return (cx / cl, cy / cl, cz / cl, 0.0)
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    s = math.sqrt(2.0 * (1.0 + d))
    inv_s = 1.0 / s
    return (cx * inv_s, cy * inv_s, cz * inv_s, s * 0.5)


def _resolve_ik_chains_for_targets(
    source_skeleton: Sequence,
    target_skeleton: Sequence,
    bone_map: BoneNameMap,
    chains: Sequence[IkChainSpec],
    disabled_chain_names: Sequence[str],
) -> List[Tuple[str, List[int], List[int], Optional[int], Optional[Tuple[float, float, float]]]]:
    """For each chain, resolve ``(name, src_idxs, tgt_idxs, pole_src, pole_axis)``.

    Skips chains that resolve to fewer than 2 bones on either side
    (FABRIK needs ≥ 2). Returns an empty list when nothing maps —
    that's fine, the IK pass becomes a no-op.

    ``pole_src`` is the SOURCE bone index for the pole-vector hint, or
    None if the spec didn't name one or the name didn't resolve.
    ``pole_axis`` is the chain's fallback pole direction (or None to
    disable pole correction entirely on this chain).

    Resolution priority for each chain entry name:
      1. NORMALISED source bone name (handles direct Mixamo-style
         conventions like "RightForeArm" verbatim).
      2. ROLE auto-detect (v4): if the chain entry is a canonical role
         (e.g. "RightForeArm") and no normalised match was found, scan
         the source skeleton for any bone whose ``auto_detect_bone_role``
         result matches. This makes inferred-chain fallback work on
         skeletons whose names don't lexically match the role.
    """
    src_name_to_idx: Dict[str, int] = {}
    for i, sb in enumerate(source_skeleton):
        norm = _normalize_joint_name(sb.name)
        if norm:
            src_name_to_idx.setdefault(norm, i)
    # v4: role -> first-bone-index lookup for the auto-detect fallback.
    src_role_to_idx: Dict[str, int] = {}
    for i, sb in enumerate(source_skeleton):
        nm = getattr(sb, "name", "") or ""
        role = auto_detect_bone_role(nm)
        if role and role not in src_role_to_idx:
            src_role_to_idx[role] = i
    n_target = len(target_skeleton)
    disabled_set = set(disabled_chain_names or ())
    out: List[Tuple[str, List[int], List[int], Optional[int], Optional[Tuple[float, float, float]]]] = []

    def _resolve_src(nm: str) -> Optional[int]:
        si = src_name_to_idx.get(nm)
        if si is not None:
            return si
        # v4 fallback: chain entry is a role; look up via auto-detect.
        return src_role_to_idx.get(nm)

    for chain in chains:
        if not chain.enabled or chain.name in disabled_set:
            continue
        src_idxs: List[int] = []
        tgt_idxs: List[int] = []
        for nm in chain.bone_names:
            si = _resolve_src(nm)
            ti = bone_map.get(nm)
            if si is None or ti is None:
                continue
            if not (0 <= ti < n_target):
                continue
            src_idxs.append(si)
            tgt_idxs.append(ti)
        if len(src_idxs) >= 2 and len(tgt_idxs) >= 2:
            pole_src: Optional[int] = None
            if chain.pole_bone_name:
                pole_src = _resolve_src(chain.pole_bone_name)
            out.append((chain.name, src_idxs, tgt_idxs, pole_src, chain.pole_axis))
    return out


def _world_position_gap(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
) -> float:
    dx = a[0] - b[0]; dy = a[1] - b[1]; dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _quat_angle_deg(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """Angular distance between two unit quats in degrees.

    The delta rotation between unit quaternions has |w| = cos(θ/2);
    inverting gives the angle θ between the two orientations. We use
    the absolute value of the dot product because q and -q represent
    the same rotation.
    """
    d = abs(_quat_dot(_quat_normalize(a), _quat_normalize(b)))
    d = max(-1.0, min(1.0, d))
    return math.degrees(2.0 * math.acos(d))


def _resolve_pole_direction(
    src_world: List[Tuple[Tuple[float, float, float], Tuple[float, ...]]],
    src_idxs: List[int],
    pole_src: Optional[int],
    pole_axis: Optional[Tuple[float, float, float]],
) -> Optional[Tuple[float, float, float]]:
    """Compute the pole-vector reference direction in world space.

    Two paths:
      * If ``pole_src`` resolved to a source bone, use that bone's
        world Z-forward (column 2 of its rotation matrix). This tracks
        the source's pose — e.g. RightShoulder Z-forward swings with
        the body. Lobby_girl-style upright humanoids: this is roughly
        +Z away from the chest, which matches the expected elbow bend.
      * Otherwise, fall back to ``pole_axis`` interpreted in the chain
        ROOT's world frame: rotate (pole_axis) by the root's world rot.
        Default pole_axis (0, 0, 1) gives Z-forward of the chain root.

    Returns None if both paths give a near-zero vector or if pole_axis
    is explicitly None.
    """
    if pole_src is not None and 0 <= pole_src < len(src_world):
        m = src_world[pole_src][1]
        # Column 2 of a row-major 3x3: indices 2, 5, 8.
        d = (m[2], m[5], m[8])
        n = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        if n > 1e-9:
            return (d[0] / n, d[1] / n, d[2] / n)
    if pole_axis is None:
        return None
    # Fallback: rotate pole_axis by the chain root's world rotation.
    if not src_idxs or src_idxs[0] >= len(src_world):
        return None
    root_mat = src_world[src_idxs[0]][1]
    rotated = _mat3_mul_vec(root_mat, pole_axis)
    n = math.sqrt(rotated[0] ** 2 + rotated[1] ** 2 + rotated[2] ** 2)
    if n < 1e-9:
        return None
    return (rotated[0] / n, rotated[1] / n, rotated[2] / n)


def _chain_needs_pole_flip(
    chain: List[Tuple[float, float, float]],
    pole_dir: Tuple[float, float, float],
) -> bool:
    """Check if the elbow/knee bend points opposite the pole hint.

    Computes the perpendicular component of the middle joint relative
    to the start↔end axis. If that perpendicular's dot with ``pole_dir``
    is strongly negative, the bend is on the wrong side of the chain
    plane and a flip is warranted.

    Two-bone chains (length 2) have no middle joint to check, so this
    returns False. For 3+ bone chains we use the second joint as the
    representative bend point.
    """
    if len(chain) < 3:
        return False
    start = chain[0]
    end = chain[-1]
    mid = chain[1]
    ax = end[0] - start[0]; ay = end[1] - start[1]; az = end[2] - start[2]
    a_len = math.sqrt(ax * ax + ay * ay + az * az)
    if a_len < 1e-9:
        return False
    ax /= a_len; ay /= a_len; az /= a_len
    mx = mid[0] - start[0]; my = mid[1] - start[1]; mz = mid[2] - start[2]
    proj = mx * ax + my * ay + mz * az
    perpx = mx - proj * ax
    perpy = my - proj * ay
    perpz = mz - proj * az
    perp_len = math.sqrt(perpx * perpx + perpy * perpy + perpz * perpz)
    if perp_len < 1e-6:
        # Chain is straight — no bend to flip.
        return False
    # Dot with pole hint.
    dot = perpx * pole_dir[0] + perpy * pole_dir[1] + perpz * pole_dir[2]
    return dot < 0.0


def _mirror_chain_across_axis(
    chain: List[Tuple[float, float, float]],
) -> List[Tuple[float, float, float]]:
    """Mirror each interior joint across the start↔end axis.

    For pole-vector correction: when FABRIK lands the elbow on the
    wrong side, reflecting the perpendicular component flips the bend
    plane while keeping segment lengths and end positions intact.

    The endpoints are preserved exactly; only joints 1..N-2 get mirrored.
    """
    if len(chain) < 3:
        return list(chain)
    start = chain[0]
    end = chain[-1]
    ax = end[0] - start[0]; ay = end[1] - start[1]; az = end[2] - start[2]
    a_len = math.sqrt(ax * ax + ay * ay + az * az)
    if a_len < 1e-9:
        return list(chain)
    ax /= a_len; ay /= a_len; az /= a_len
    out: List[Tuple[float, float, float]] = [start]
    for j in chain[1:-1]:
        mx = j[0] - start[0]; my = j[1] - start[1]; mz = j[2] - start[2]
        proj = mx * ax + my * ay + mz * az
        # Perpendicular component (this is what we mirror).
        perpx = mx - proj * ax
        perpy = my - proj * ay
        perpz = mz - proj * az
        # Mirrored joint: same axial component, perpendicular negated.
        out.append((
            start[0] + proj * ax - perpx,
            start[1] + proj * ay - perpy,
            start[2] + proj * az - perpz,
        ))
    out.append(end)
    return out


# ---------------------------------------------------------------------------
# Motion mirroring (v3, 2026-04-25)
# ---------------------------------------------------------------------------
#
# Many Mixamo clips author one-sided motions ("right-hand wave"); users
# sometimes want the mirrored variant ("left-hand wave"). The mirroring
# operates on already-retargeted ``NjmRawMotion`` objects:
#
#   1. Identify pairs of bones whose source-side names are symmetric
#      (Left*↔Right* or mixamorig:Left*↔mixamorig:Right* etc).
#   2. Swap each pair's track keyframes (positions + rotations).
#   3. Negate X axis on positions, mirror rotations across the YZ plane
#      (= multiply each quaternion by (-1, 1, 1, -1) component-wise on
#      (qx, qy, qz, qw)).


_MIRROR_PAIRS: Tuple[Tuple[str, str], ...] = (
    # Most-specific Mixamo names first so "LeftUpLeg" doesn't get
    # eaten by the bare "Left" -> "Right" rule.
    ("LeftUpLeg", "RightUpLeg"),
    ("LeftUp", "RightUp"),
    ("LeftFore", "RightFore"),
    ("LeftHand", "RightHand"),
    ("LeftFoot", "RightFoot"),
    ("LeftLeg", "RightLeg"),
    ("LeftArm", "RightArm"),
    ("LeftShoulder", "RightShoulder"),
    ("Left", "Right"),
    # Lower-case variants used by PSOBB-style target names (l_arm,
    # r_arm, etc.) and other engines that camelCase joint roots.
    ("left_", "right_"),
    ("l_", "r_"),
    # Single-letter prefix conventions (UpperCase variant).
    ("L_", "R_"),
    # _R_ and _L_ as suffix tags (CesiumMan-style).
    ("_L_", "_R_"),
)


def _swap_lr_in_name(name: str) -> Optional[str]:
    """Return the L/R-swapped variant of ``name``, or None if no pattern matched.

    Tries token replacements in order; returns the first match (most
    specific patterns first to avoid e.g. "LeftHand" turning into
    "RightHand" via a pure "Left" → "Right" swap when a more specific
    pair exists). Patterns are anchored at the start of the basename
    (after any namespace-style prefix like "mixamorig:").
    """
    if not name:
        return None
    # Strip mixamorig: prefix for matching, but preserve it on output.
    prefix = ""
    rest = name
    for p in ("mixamorig:", "mixamorig1:", "Armature_", "rig_"):
        if rest.startswith(p):
            prefix = p
            rest = rest[len(p):]
            break
    for left_token, right_token in _MIRROR_PAIRS:
        if rest.startswith(left_token):
            return prefix + right_token + rest[len(left_token):]
        if rest.startswith(right_token):
            return prefix + left_token + rest[len(right_token):]
    # Try mid-name match for the suffix-style pairs (CesiumMan style).
    for left_token, right_token in _MIRROR_PAIRS:
        if left_token in rest:
            swapped = rest.replace(left_token, "\x00")
            swapped = swapped.replace(right_token, left_token)
            swapped = swapped.replace("\x00", right_token)
            if swapped != rest:
                return prefix + swapped
    return None


def detect_lr_pairs(bone_names: Sequence[str]) -> Dict[int, int]:
    """Build an index → mirrored-index map from a list of bone names.

    For each bone whose name has an L/R counterpart in ``bone_names``,
    map its index to the counterpart's index. Self-mappings (centerline
    bones with no opposite) are omitted.

    Used by ``mirror_animation`` to know which target-bone tracks to
    swap. Bones without a sibling (e.g. ``Hips``, ``Spine``, ``Neck``)
    keep their tracks but receive the X-flip / quat-mirror.
    """
    name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(bone_names)}
    out: Dict[int, int] = {}
    for i, n in enumerate(bone_names):
        if i in out:
            continue
        sw = _swap_lr_in_name(n)
        if sw is not None and sw in name_to_idx:
            j = name_to_idx[sw]
            if j != i:
                out[i] = j
                out[j] = i
    return out


def _mirror_quat_yz(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Mirror a rotation across the YZ plane (= negate X axis).

    Algebra: a reflection through a plane perpendicular to axis ``a``
    transforms a unit quaternion (qx, qy, qz, qw) by negating the
    components NOT along ``a`` and the sign of the rotation. For the
    YZ plane (X-axis perpendicular):
        (qx, qy, qz, qw) → (qx, -qy, -qz, qw)
    Equivalently, multiply by (-1, 1, 1, -1) on (qx, qy, qz, qw) —
    which is the canonical form mentioned in the spec.

    Either form represents the same rotation (q and -q are equivalent),
    so we use the (qx, -qy, -qz, qw) form for clarity. This matches
    the geometry of an X-axis flip: a right-hand rotation about Y
    becomes a left-hand rotation about Y (= negated rotation), keeping
    the rotation about X unchanged.
    """
    return (q[0], -q[1], -q[2], q[3])


def mirror_animation(
    motion: NjmRawMotion,
    *,
    target_bone_names: Optional[Sequence[str]] = None,
    axis: str = "x",
) -> NjmRawMotion:
    """Return a left↔right mirrored copy of ``motion``.

    Args
    ----
    motion:
        The retargeted motion to mirror. Not mutated; a fresh
        NjmRawMotion is returned.
    target_bone_names:
        Bone names IN THE TARGET SKELETON ORDER, used to detect L/R
        pairs. When None, pair detection is skipped and only the
        per-frame quat mirror is applied (works for symmetric
        animations). Pass the target skeleton's bone-name list to get
        the full pair-swap behaviour.
    axis:
        Mirror plane. Currently only ``"x"`` (= mirror across the YZ
        plane, swap left↔right in a Y-up character). Reserved for
        future Z-mirror support.

    Notes
    -----
    The position keyframe X coordinates get negated; the rotation
    keyframes are decoded BAMS → quat → mirror → BAMS. We re-encode
    via ``quat_to_zyx_bams`` so the rounding is applied once (the
    writer's narrow/wide layout is preserved by the existing track
    metadata).

    Returns the mirrored motion. Diagnostics are stashed on
    ``motion._retarget_mirror`` (number of pairs swapped, etc.).
    """
    if axis != "x":
        raise ValueError(f"unsupported mirror axis: {axis!r} (only 'x' for now)")

    # Detect L/R bone pairs (if names available).
    pair_map: Dict[int, int] = {}
    if target_bone_names:
        pair_map = detect_lr_pairs(list(target_bone_names))

    n_bones = len(motion.bones)

    # Build new bone list. We deep-copy keyframes so the source motion
    # stays untouched.
    new_bones: List[NjmBoneTracks] = [NjmBoneTracks() for _ in range(n_bones)]

    swapped_pairs = 0
    seen: set = set()
    for ti in range(n_bones):
        if ti in seen:
            continue
        partner = pair_map.get(ti)
        if partner is not None and partner != ti and partner not in seen:
            # Swap-and-mirror: take partner's tracks, mirror them, put on ti.
            _set_mirrored_tracks(new_bones[ti], motion.bones[partner])
            _set_mirrored_tracks(new_bones[partner], motion.bones[ti])
            seen.add(ti)
            seen.add(partner)
            swapped_pairs += 1
        else:
            # Centerline bone: mirror in place (no swap).
            _set_mirrored_tracks(new_bones[ti], motion.bones[ti])
            seen.add(ti)

    new_motion = NjmRawMotion(
        frame_count=motion.frame_count,
        type_flags=motion.type_flags,
        inp_fn=motion.inp_fn,
        m_data_table_offset=motion.m_data_table_offset,
        bones=new_bones,
    )
    # Carry forward retargeting diagnostics so the server's summary
    # view still has access to them.
    for attr in ("_retarget_dropped", "_retarget_mapped", "_retarget_ik"):
        v = getattr(motion, attr, None)
        if v is not None:
            setattr(new_motion, attr, v)
    new_motion._retarget_mirror = {  # type: ignore[attr-defined]
        "axis": axis,
        "swapped_pairs": swapped_pairs,
        "lr_pairs_detected": len(pair_map) // 2,
    }
    return new_motion


def _set_mirrored_tracks(dst: NjmBoneTracks, src: NjmBoneTracks) -> None:
    """Copy ``src``'s POS and ANG tracks into ``dst`` with the X-mirror applied.

    POS keyframes: negate X.
    ANG keyframes: BAMS → quat → mirror_yz → BAMS.
    Other track kinds (scale, evt) pass through unchanged.
    """
    for kind, track in src.tracks_by_kind.items():
        if kind == NJD_MTYPE_POS:
            new_kfs: List[Tuple] = []
            for kf in track.keyframes:
                # POS keyframes are (frame, x, y, z) tuples.
                if len(kf) >= 4:
                    new_kfs.append((kf[0], -kf[1], kf[2], kf[3]))
                else:
                    new_kfs.append(kf)
            dst.tracks_by_kind[kind] = NjmTrack(
                kind=kind,
                keyframes=new_kfs,
                narrow=track.narrow,
            )
        elif kind == NJD_MTYPE_ANG:
            new_kfs = []
            for kf in track.keyframes:
                if len(kf) >= 4:
                    f, rx, ry, rz = kf[0], kf[1], kf[2], kf[3]
                    # Decode BAMS → quat (the offset stored in NJM is
                    # bind_inv * full; we mirror the offset directly,
                    # which mirrors the WORLD rotation given the matched
                    # mirroring of the bind). For an asymmetric bind
                    # this isn't exact, but for L↔R-symmetric humanoid
                    # binds — the only case where mirroring a clip makes
                    # sense — it does the right thing.
                    q = _bams_to_quat(int(rx) & 0xFFFF, int(ry) & 0xFFFF, int(rz) & 0xFFFF)
                    qm = _mirror_quat_yz(q)
                    rxm, rym, rzm = quat_to_zyx_bams(*qm)
                    new_kfs.append((f, rxm, rym, rzm))
                else:
                    new_kfs.append(kf)
            dst.tracks_by_kind[kind] = NjmTrack(
                kind=kind,
                keyframes=new_kfs,
                narrow=track.narrow,
            )
        else:
            # Pass-through for any non-POS/ANG track.
            dst.tracks_by_kind[kind] = NjmTrack(
                kind=kind,
                keyframes=list(track.keyframes),
                narrow=track.narrow,
            )


# ---------------------------------------------------------------------------
# Main retarget pipeline
# ---------------------------------------------------------------------------


def retarget_animation(
    source_anim: ImportedAnimation,
    source_skeleton: List[ImportedBone],
    target_skeleton: Sequence,
    bone_map: BoneNameMap,
    *,
    target_fps: int = 30,
    include_translation: bool = False,
    translation_scale: Optional[float] = None,
    target_pos_bone: Optional[int] = None,
    flip_z: bool = True,
    enable_ik: bool = True,
    enable_ik_rotation: bool = True,
    ik_chains: Optional[Sequence[IkChainSpec]] = None,
    disabled_ik_chains: Sequence[str] = (),
    ik_threshold: float = 1e-3,
    ik_iterations: int = 16,
    mirror: bool = False,
    vrm_humanoid_map: Optional[Dict[str, int]] = None,
    enable_auto_detect: bool = True,
) -> NjmRawMotion:
    """Retarget a glTF animation onto a PSOBB skeleton.

    Args
    ----
    source_anim:
        Output of ``parse_gltf_with_animations``.
    source_skeleton:
        ``ImportedModel.bones`` (from the same glTF file).
    target_skeleton:
        Either a ``list[XjBone]`` (parsed from the destination .nj
        via ``formats.xj.parse_skeleton``) OR a ``list[ImportedBone]``
        (e.g. when retargeting to one of the editor's templates).
    bone_map:
        Source-name -> target-bone-index dict. Mixamo's
        ``mixamorig:`` prefix is auto-stripped; case-sensitive.
    target_fps:
        Resample rate. PSOBB sim runs at 30 Hz, so 30 is the right
        default for any motion that needs to look right in-game.
    include_translation:
        When True, emit POS tracks scaled by ``translation_scale``.
        Default False so locomotion-baked-into-hips animations don't
        slide the receptionist away from her counter.
    translation_scale:
        Multiplier for source -> target translation. When None and
        ``include_translation`` is True, we estimate it from the ratio
        of source-skeleton-AABB to target-skeleton-AABB Y extents.
    target_pos_bone:
        Which target bone to write the POS track to (when
        ``include_translation`` is True). Defaults to the bone that
        ``Hips`` resolves to.
    flip_z:
        When True (default), apply the glTF -> PSOBB Z mirror to
        every rotation. Pass False if the source is already in PSOBB
        convention (e.g. you're retargeting between two PSOBB
        skeletons).
    enable_ik:
        When True (default), run an IK pass over each end-effector
        chain (hands + feet by default) to pull the target's wrist /
        ankle world position to match the source's. Closes the
        bone-length-mismatch gap that 1:1 quat copy can't.
    enable_ik_rotation:
        When True (default, v3 2026-04-25), after the positional IK
        pulls the chain end to the source's world position, also
        rotate the end-effector bone (wrist / ankle) so its WORLD
        rotation matches the source's. Without this, a Mixamo wrist
        twist doesn't propagate to the target hand. Disable to
        reproduce v2 baseline behaviour.
    ik_chains:
        Override the default ``HUMANOID_IK_CHAINS`` list. Each
        IkChainSpec lists source-bone names from chain root to
        end-effector (e.g. shoulder→arm→forearm→hand).
    disabled_ik_chains:
        Chain names to skip even when ``enable_ik`` is True. E.g.
        ``["right_arm"]`` disables right-arm IK while keeping legs +
        left arm. Useful when the source animation drives one arm
        only and the other should snap back to bind.
    ik_threshold:
        World-position gap threshold in target units below which a
        chain is considered already-aligned and IK is skipped for
        that frame (avoids running iterations when the 1:1 copy
        already nailed the position). Default 1e-3.
    ik_iterations:
        FABRIK iteration cap per chain per frame. 16 is plenty for
        the 4-bone arm case. Lower if you need more frame-rate; the
        residual error scales as ``O(1/iter)`` for typical cases.
    mirror:
        When True (v3 2026-04-25), apply a left↔right post-processing
        pass that swaps each ``Left*`` track with its ``Right*``
        counterpart and mirrors per-frame quats / translations across
        the YZ plane. Useful for converting one-handed Mixamo clips
        (e.g. "right-hand wave") into their mirrored variant without
        re-authoring the source animation. Default False.
    vrm_humanoid_map:
        VRM-extension humanoid map (``role -> source_bone_idx``) from
        ``ImportedModel.vrm_humanoid_map``. When supplied, source bones
        that appear in this map route through the VRM role → canonical
        role → ``bone_map`` chain instead of relying on bone-name
        string matching. Authoritative for VRoid Studio / Booth-PM CC0
        characters where every humanoid joint is explicitly tagged.
        Pass ``None`` (default) to disable VRM routing and fall back
        to name-only resolution.
    enable_auto_detect:
        When True (default), source bones not directly in ``bone_map``
        are passed through ``auto_detect_bone_role`` to recover a
        canonical role from rig-convention naming (Mecanim, Cesium,
        Rigify, MakeHuman, free-form). The detected role is then
        looked up in ``bone_map`` to find the target bone. Pass False
        to disable the heuristic (useful for tests that need to verify
        the explicit-map path in isolation).

    Returns
    -------
    NjmRawMotion ready to feed into ``njm_writer.encode_njm``.
    """
    n_target = _target_bone_count(target_skeleton)

    # Build target frame grid.
    duration = source_anim.duration_seconds
    if duration <= 0.0:
        duration = (max(
            (t.times[-1] for t in source_anim.tracks if t.times),
            default=0.0,
        ))
    n_frames = max(1, int(round(duration * target_fps)) + 1)
    target_times = [f / float(target_fps) for f in range(n_frames)]

    # Group source tracks by (bone_idx, channel).
    src_by_key: Dict[Tuple[int, str], ImportedTrack] = {}
    for tr in source_anim.tracks:
        src_by_key[(tr.bone_idx, tr.channel)] = tr

    # Resolve src_bone -> target_bone via the name map (with prefix-strip),
    # the optional VRM humanoid map, and the auto-detect heuristic.
    #
    # Resolution priority (first hit wins):
    #   1. VRM humanoid map (if supplied): role(vrm)->canonical->bone_map
    #   2. Direct: normalised(name) in bone_map
    #   3. auto_detect_bone_role(name): canonical->bone_map (when enabled)
    #
    # Using a tiered lookup means a VRM file with a non-canonical bone
    # name (e.g. user-renamed an exported bone) still routes correctly
    # via its tagged humanoid role; meanwhile non-VRM rigs get the same
    # name-string + heuristic fallback as before.
    src_to_tgt: Dict[int, int] = {}
    dropped: List[str] = []
    # Invert the VRM humanoid map so source-bone-idx → VRM role lookup is
    # O(1) inside the resolution loop. Skips entries whose target index
    # is out of source-skeleton range (defensive: VRM files can reference
    # nodes outside the primary skin in pathological cases).
    src_idx_to_vrm_role: Dict[int, str] = {}
    if vrm_humanoid_map:
        n_src = len(source_skeleton)
        for vrm_role, src_idx in vrm_humanoid_map.items():
            if 0 <= int(src_idx) < n_src:
                src_idx_to_vrm_role[int(src_idx)] = str(vrm_role)
    auto_detected = 0
    vrm_resolved = 0
    for si, sb in enumerate(source_skeleton):
        norm = _normalize_joint_name(sb.name)
        # Path 1: VRM humanoid role → canonical → bone_map.
        if si in src_idx_to_vrm_role:
            vrm_role = src_idx_to_vrm_role[si]
            canonical = _VRM_ROLE_TO_CANONICAL.get(vrm_role)
            if canonical is not None and canonical in bone_map:
                tgt = bone_map[canonical]
                if 0 <= tgt < n_target:
                    src_to_tgt[si] = tgt
                    vrm_resolved += 1
                    continue
        # Path 2: direct bone-name match against the supplied map.
        if norm in bone_map:
            tgt = bone_map[norm]
            if 0 <= tgt < n_target:
                src_to_tgt[si] = tgt
                continue
            else:
                dropped.append(f"{sb.name}: target bone {tgt} out of range")
                continue
        # Path 3: auto-detect heuristic on the name (when enabled).
        if enable_auto_detect:
            role = auto_detect_bone_role(sb.name)
            if role is not None and role in bone_map:
                tgt = bone_map[role]
                if 0 <= tgt < n_target:
                    src_to_tgt[si] = tgt
                    auto_detected += 1
                    continue
        dropped.append(f"{sb.name}: not in map")

    # Find target POS bone if needed.
    if include_translation and target_pos_bone is None:
        if "Hips" in bone_map:
            target_pos_bone = bone_map["Hips"]
        elif src_to_tgt:
            # Pick the lowest target index (likely the hip/root).
            target_pos_bone = min(src_to_tgt.values())

    # Estimate translation scale (source bbox vs target bbox Y extents).
    if include_translation and translation_scale is None:
        translation_scale = _estimate_skeleton_scale(source_skeleton, target_skeleton)

    # Build per-target-bone tracks.
    target_bones: List[NjmBoneTracks] = [NjmBoneTracks() for _ in range(n_target)]

    # First pass: rotation per bone.
    for src_idx, tgt_idx in src_to_tgt.items():
        rot_track = src_by_key.get((src_idx, "rotation"))
        if rot_track is None or not rot_track.values:
            continue
        sampled = _resample_track(rot_track, target_times)
        if not sampled:
            continue

        # Convert each sampled quat into the LOCAL delta the PSOBB
        # renderer applies on top of bone bind. Algorithm:
        #   delta = bind_target⁻¹ * (mirror_z(q_source) ?  apply ?)
        # The simplest correct mapping is to write the source rotation
        # AS the bone's local rotation (overriding bind). That works
        # because both Mixamo and PSOBB skeletons are authored in
        # T-pose and the source animation is interpreted relative to
        # T-pose. We keep ``bind_target⁻¹`` composition behind a
        # comment in case the empirical preview indicates otherwise.
        target_bind = _target_bind_quat(target_skeleton[tgt_idx])
        target_bind_inv = _quat_inverse(target_bind)

        ang_kfs: List[Tuple[int, int, int, int]] = []
        for f, q in enumerate(sampled):
            if flip_z:
                q = _mirror_quat_z(q)
            # Compose with target-bind-inverse so the emitted angles
            # encode the ABSOLUTE bone rotation (renderer multiplies
            # by bind to get the world rotation, so we cancel the
            # bind out here). For a source whose Mixamo bind is
            # identity (typical), this just rotates by the inverse
            # bind rotation — i.e. if PSOBB's bone has bind ZRot=-90°
            # and we want the bone WORLD-aligned, we emit ZRot=+90°.
            local = _quat_mul(target_bind_inv, q)
            rx, ry, rz = quat_to_zyx_bams(*local)
            ang_kfs.append((f, rx, ry, rz))

        target_bones[tgt_idx].tracks_by_kind[NJD_MTYPE_ANG] = NjmTrack(
            kind=NJD_MTYPE_ANG,
            keyframes=ang_kfs,
            narrow=True,
        )

    # IK pass: nudge end-effector chains to match source world positions.
    # Runs BEFORE the translation pass so a hip-translated source still
    # gets correct hand placement (translation moves the whole rig; IK
    # fixes the per-joint gap that bone-length differences introduce).
    ik_diagnostics: Dict[str, object] = {"chains": [], "frames_solved": 0}
    if enable_ik and src_to_tgt:
        ik_diagnostics = _apply_ik_pass(
            source_anim=source_anim,
            source_skeleton=source_skeleton,
            target_skeleton=target_skeleton,
            bone_map=bone_map,
            src_to_tgt=src_to_tgt,
            src_by_key=src_by_key,
            target_times=target_times,
            target_bones=target_bones,
            flip_z=flip_z,
            chains=tuple(ik_chains) if ik_chains is not None else HUMANOID_IK_CHAINS,
            disabled_chain_names=disabled_ik_chains,
            threshold=ik_threshold,
            iterations=ik_iterations,
            apply_rotation_ik=enable_ik_rotation,
        )

    # Second pass: translation on the hip bone, if requested.
    if include_translation and target_pos_bone is not None:
        # Find the source hip bone: it's the one with channel
        # "translation" that maps to ``target_pos_bone``. If multiple,
        # pick the first.
        src_hip = None
        for src_idx, tgt_idx in src_to_tgt.items():
            if tgt_idx == target_pos_bone and (src_idx, "translation") in src_by_key:
                src_hip = src_idx
                break
        if src_hip is not None:
            pos_track = src_by_key[(src_hip, "translation")]
            sampled = _resample_track(pos_track, target_times)
            if sampled:
                pos_kfs: List[Tuple[int, float, float, float]] = []
                k = float(translation_scale or 1.0)
                for f, p in enumerate(sampled):
                    x, y, z = p
                    if flip_z:
                        z = -z
                    pos_kfs.append((f, x * k, y * k, z * k))
                # Make sure the bone has BOTH POS and ANG tracks if
                # we already wrote ANG (so the encoder emits two
                # entries in the MData slot in POS-first order).
                target_bones[target_pos_bone].tracks_by_kind[NJD_MTYPE_POS] = NjmTrack(
                    kind=NJD_MTYPE_POS,
                    keyframes=pos_kfs,
                    narrow=True,
                )

    # Determine type_flags: POS+ANG if any bone has POS, else ANG.
    has_any_pos = any(
        NJD_MTYPE_POS in b.tracks_by_kind and b.tracks_by_kind[NJD_MTYPE_POS].keyframes
        for b in target_bones
    )
    type_flags = NJD_MTYPE_ANG | (NJD_MTYPE_POS if has_any_pos else 0)

    # Ensure every bone has the required track slots in canonical order
    # (POS then ANG) — empty tracks render as offset=0, count=0.
    for bone in target_bones:
        if has_any_pos and NJD_MTYPE_POS not in bone.tracks_by_kind:
            bone.tracks_by_kind[NJD_MTYPE_POS] = NjmTrack(
                kind=NJD_MTYPE_POS, keyframes=[], narrow=True,
            )
        if NJD_MTYPE_ANG not in bone.tracks_by_kind:
            bone.tracks_by_kind[NJD_MTYPE_ANG] = NjmTrack(
                kind=NJD_MTYPE_ANG, keyframes=[], narrow=True,
            )

    element_count = bin(type_flags).count("1")
    inp_fn = element_count  # interp = 0 (linear), low 4 bits = element_count

    motion = NjmRawMotion(
        frame_count=n_frames,
        type_flags=type_flags,
        inp_fn=inp_fn,
        m_data_table_offset=0xC,
        bones=target_bones,
    )
    # Stash diagnostics on the motion for the server to surface.
    motion._retarget_dropped = dropped  # type: ignore[attr-defined]
    motion._retarget_mapped = list(src_to_tgt.items())  # type: ignore[attr-defined]
    motion._retarget_ik = ik_diagnostics  # type: ignore[attr-defined]
    motion._retarget_resolution = {  # type: ignore[attr-defined]
        "vrm_resolved": vrm_resolved,
        "auto_detected": auto_detected,
        "direct_mapped": len(src_to_tgt) - vrm_resolved - auto_detected,
    }

    # ---- Optional mirror post-processing (v3, 2026-04-25) -------------------
    # Build the target-bone names list using whatever attribute the skeleton
    # exposes; we feed this to mirror_animation so it can detect L/R pairs
    # by name. For target_skeletons whose entries don't carry a useful
    # ``name`` (XjBone parsed from a real PSOBB file is unnamed — bones are
    # identified by index), we fall back to deriving names from the bone_map
    # itself (reverse map: tgt_idx → first source name that resolves to it).
    if mirror:
        tgt_names = _derive_target_bone_names(target_skeleton, bone_map)
        motion = mirror_animation(motion, target_bone_names=tgt_names, axis="x")
    return motion


def _derive_target_bone_names(
    target_skeleton: Sequence,
    bone_map: BoneNameMap,
) -> List[str]:
    """Best-effort target-bone-name list for L/R-pair detection.

    PSOBB ``XjBone`` entries don't carry a ``name`` field — the format
    identifies bones by index alone. To still mirror sensibly we
    reverse-map the bone_map: for each target index, pick the FIRST
    source bone name that maps to it. This gives us names like
    "RightHand", "LeftHand", etc. on the target side, which the
    L/R-pair detection then picks up cleanly.

    For target skeletons whose entries DO have a ``name`` attribute
    (e.g. ``ImportedBone``), prefer that over the reverse-map name —
    the entry's own name is more authoritative.
    """
    n = len(target_skeleton)
    out: List[str] = [""] * n
    # Pass 1: use the bone's own ``name`` attribute when present.
    for i, b in enumerate(target_skeleton):
        nm = getattr(b, "name", None)
        if isinstance(nm, str) and nm:
            out[i] = nm
    # Pass 2: fill remaining indices from the reverse bone_map.
    for src_name, tgt_idx in bone_map.items():
        if 0 <= tgt_idx < n and not out[tgt_idx]:
            out[tgt_idx] = src_name
    return out


def _bams_to_quat(rx_b: int, ry_b: int, rz_b: int) -> Tuple[float, float, float, float]:
    """Inverse of ``quat_to_zyx_bams``: BAMS triple → (qx, qy, qz, qw).

    ZYX intrinsic order — same convention as ``_target_bind_quat``'s
    XjBone branch but accepting raw signed angle ints.
    """
    rx = (rx_b if rx_b < 0x8000 else rx_b - 0x10000) * (math.pi * 2.0 / 0x10000)
    ry = (ry_b if ry_b < 0x8000 else ry_b - 0x10000) * (math.pi * 2.0 / 0x10000)
    rz = (rz_b if rz_b < 0x8000 else rz_b - 0x10000) * (math.pi * 2.0 / 0x10000)
    cx, sx = math.cos(rx * 0.5), math.sin(rx * 0.5)
    cy, sy = math.cos(ry * 0.5), math.sin(ry * 0.5)
    cz, sz = math.cos(rz * 0.5), math.sin(rz * 0.5)
    qw = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz
    return (qx, qy, qz, qw)


def _scaled_skeleton_for_target(
    source_skeleton: List[ImportedBone],
    target_skeleton: Sequence,
    src_to_tgt: Dict[int, int],
) -> List[ImportedBone]:
    """Build a virtual SOURCE skeleton scaled to TARGET bone-segment lengths.

    The source skeleton's bone offsets reflect the source character's
    proportions (Mixamo's 1.7 m human). For the IK pass we want to
    measure where the source's hand WOULD land if we picked up the
    source's joint angles and dropped them into a target-sized rig.
    Otherwise, "match the source's wrist position" would just shrink
    the target arm to Mixamo length, undoing the whole reason we're
    retargeting.

    This helper returns a copy of ``source_skeleton`` with each mapped
    bone's translation rescaled by the per-bone length ratio (target
    segment length / source segment length). Unmapped bones keep their
    original offsets (their world positions don't matter for the IK
    chains we care about).
    """
    out: List[ImportedBone] = []
    src_parents = _bone_parents(source_skeleton)
    tgt_parents = _bone_parents(target_skeleton)
    for si, sb in enumerate(source_skeleton):
        ox, oy, oz = _bone_position(sb)
        scaled_pos = (ox, oy, oz)
        sp = src_parents[si]
        ti = src_to_tgt.get(si)
        # Special case: root bone with a target mapping. Use the target's
        # bind position directly so the scaled-source FK starts at the
        # same world origin as the target FK.
        if sp < 0 and ti is not None and 0 <= ti < len(target_skeleton):
            scaled_pos = _bone_position(target_skeleton[ti])
        if sp >= 0 and ti is not None:
            tp_idx_for_self = ti
            # Find target's parent in mapped space: walk source's parent
            # chain until we find one with a mapped target.
            walker = sp
            tp_for_parent: Optional[int] = None
            while walker >= 0:
                tp_for_parent = src_to_tgt.get(walker)
                if tp_for_parent is not None:
                    break
                walker = src_parents[walker] if walker < len(src_parents) else -1
            if tp_for_parent is not None:
                # Walk the target side from tp_for_parent → tp_idx_for_self
                # to compute the cumulative translation along that
                # target sub-chain. For a directly-parented pair this is
                # just target_skeleton[ti].position. When the target
                # has fewer bones than the source between them, summing
                # is the right answer.
                cum_t = (0.0, 0.0, 0.0)
                cur = tp_idx_for_self
                guard = 64
                while cur >= 0 and cur != tp_for_parent and guard > 0:
                    pos = _bone_position(target_skeleton[cur])
                    cum_t = (cum_t[0] + pos[0], cum_t[1] + pos[1], cum_t[2] + pos[2])
                    cur = tgt_parents[cur] if cur < len(tgt_parents) else -1
                    guard -= 1
                # Compute the matching cumulative on the source side.
                cum_s = (0.0, 0.0, 0.0)
                cur_s = si
                guard = 64
                while cur_s >= 0 and cur_s != sp and guard > 0:
                    pos = _bone_position(source_skeleton[cur_s])
                    cum_s = (cum_s[0] + pos[0], cum_s[1] + pos[1], cum_s[2] + pos[2])
                    cur_s = src_parents[cur_s] if cur_s < len(src_parents) else -1
                    guard -= 1
                # Replace the SOURCE local translation with the TARGET's
                # cumulative offset divided across the same chain — for
                # the simple parent-child case this is exactly the
                # target's bone offset, which is what we want.
                scaled_pos = cum_t if cum_t != (0.0, 0.0, 0.0) else (ox, oy, oz)
        out.append(ImportedBone(
            name=sb.name,
            parent_idx=sb.parent_idx,
            bind_pos=scaled_pos,
            bind_rot_quat=sb.bind_rot_quat,
            bind_scale=sb.bind_scale,
        ))
    return out


def _apply_ik_pass(
    *,
    source_anim: ImportedAnimation,
    source_skeleton: List[ImportedBone],
    target_skeleton: Sequence,
    bone_map: BoneNameMap,
    src_to_tgt: Dict[int, int],
    src_by_key: Dict[Tuple[int, str], ImportedTrack],
    target_times: Sequence[float],
    target_bones: List[NjmBoneTracks],
    flip_z: bool,
    chains: Sequence[IkChainSpec],
    disabled_chain_names: Sequence[str],
    threshold: float,
    iterations: int,
    apply_rotation_ik: bool = True,
) -> Dict[str, object]:
    """Run the per-frame IK pass; mutate ``target_bones`` in place.

    Algorithm summary
    -----------------
    For each frame:
      1. Build the local-quat list for source + target by reading the
         previously-resampled rotation tracks and the per-bone bind
         poses (for unanimated bones).
      2. Run forward kinematics on both skeletons to get world-space
         joint positions.
      3. For each enabled chain:
           - Get source's end-effector world position.
           - Get target's chain world positions (length 2..4).
           - Skip if gap < threshold.
           - Run FABRIK. The new joint positions imply per-bone local
             rotation deltas (the rotation that points each bone's
             original direction at the new direction).
           - Pole-vector check (v3): if the elbow/knee bend ended up on
             the wrong side of the start↔end axis (dot < 0 with the
             pole hint direction), mirror the chain across that axis
             so the bend points toward the pole.
           - Convert each delta to a quaternion and overlay it on the
             target's local rotation, then re-emit the BAMS triple in
             the corresponding ``target_bones`` ANG track.
           - Rotation IK (v3): if ``apply_rotation_ik`` is True, after
             positional IK, write the end-effector's local rotation as
             ``parent_world_inv * source_end_world_rot`` so the wrist
             twist matches the source.
    """
    n_target = len(target_skeleton)
    chain_resolutions = _resolve_ik_chains_for_targets(
        source_skeleton, target_skeleton, bone_map, chains, disabled_chain_names,
    )
    inferred_used = False
    if not chain_resolutions:
        # Fallback (v4, 2026-04-25): the explicit chains didn't resolve
        # any usable end-effector pair. This happens with non-Mixamo,
        # non-Cesium, non-VRM rigs whose bone names don't match the
        # canonical role table. Try the inferred chains — they walk
        # parent links from each detected end-effector and produce
        # IkChainSpec entries that should resolve via the bone_map's
        # role-keyed entries even if the source's bone NAMES don't.
        inferred = infer_ik_chains_from_skeleton(source_skeleton)
        if inferred:
            chain_resolutions = _resolve_ik_chains_for_targets(
                source_skeleton, target_skeleton, bone_map,
                tuple(inferred), disabled_chain_names,
            )
            if chain_resolutions:
                inferred_used = True
    if not chain_resolutions:
        return {"chains": [], "frames_solved": 0,
                "rotation_ik_enabled": bool(apply_rotation_ik),
                "inferred": False}

    # We need the SOURCE animation's per-frame quaternions to drive
    # source FK. Resample once per source bone (not just the bones
    # that have a target map — the chain's intermediate bones contribute
    # to FK regardless).
    src_quats_per_frame: List[List[Tuple[float, float, float, float]]] = []
    for f in range(len(target_times)):
        src_quats_per_frame.append([
            _bind_quat_for_source(b) for b in source_skeleton
        ])
    for src_idx, sb in enumerate(source_skeleton):
        rot_track = src_by_key.get((src_idx, "rotation"))
        if rot_track is None or not rot_track.values:
            continue
        sampled = _resample_track(rot_track, target_times)
        for f, q in enumerate(sampled):
            if f < len(src_quats_per_frame):
                src_quats_per_frame[f][src_idx] = (
                    float(q[0]), float(q[1]), float(q[2]), float(q[3]),
                )

    # TARGET local quats per frame from the rotation tracks we just
    # built (re-decoded from BAMS so we work in the same space the
    # writer will round-trip).
    tgt_bind_quats: List[Tuple[float, float, float, float]] = [
        _target_bind_quat(b) for b in target_skeleton
    ]
    tgt_quats_per_frame: List[List[Tuple[float, float, float, float]]] = []
    n_frames = len(target_times)
    for f in range(n_frames):
        tgt_quats_per_frame.append(list(tgt_bind_quats))
    for ti in range(n_target):
        ang_track = target_bones[ti].tracks_by_kind.get(NJD_MTYPE_ANG)
        if ang_track is None or not ang_track.keyframes:
            continue
        for kf in ang_track.keyframes:
            f, rx, ry, rz = kf[0], kf[1], kf[2], kf[3]
            if 0 <= f < n_frames:
                # The encoded rotation is the OFFSET we pre-multiplied by
                # bind_inv. So the WORLD rotation of the bone (relative
                # to its parent) is bind * encoded.
                offset_q = _bams_to_quat(int(rx) & 0xFFFF, int(ry) & 0xFFFF, int(rz) & 0xFFFF)
                # local_full = bind * offset
                local_full = _quat_mul(tgt_bind_quats[ti], offset_q)
                tgt_quats_per_frame[f][ti] = local_full

    # Build a virtual scaled SOURCE skeleton so source-FK measurements
    # correspond to TARGET sizes (otherwise IK would shrink the target
    # arm to source length and undo the retarget).
    scaled_source_skel = _scaled_skeleton_for_target(
        source_skeleton, target_skeleton, src_to_tgt,
    )

    chain_stats: List[Dict[str, object]] = [
        {"name": cn, "frames_above_threshold": 0,
         "max_gap_before": 0.0, "max_gap_after": 0.0,
         "first_frame_before": None, "first_frame_after": None,
         "src_indices": list(si), "tgt_indices": list(ti),
         "pole_corrections": 0,
         "rotation_ik_max_err_deg": 0.0,
         "rotation_ik_frames_applied": 0}
        for (cn, si, ti, _ps, _pa) in chain_resolutions
    ]
    frames_solved = 0
    EVAL_ZXY_ANG = 0x20

    for f in range(n_frames):
        src_quats = src_quats_per_frame[f]
        tgt_quats = tgt_quats_per_frame[f]
        # Forward kinematics — once per frame, used by ALL chains.
        src_world = _forward_kinematics(scaled_source_skel, src_quats)
        tgt_world = _forward_kinematics(target_skeleton, tgt_quats)
        any_chain_solved = False
        for chain_i, (cn, src_idxs, tgt_idxs, pole_src, pole_axis) in enumerate(chain_resolutions):
            # Source end-effector world position.
            src_end_pos = src_world[src_idxs[-1]][0]
            # Target chain world positions.
            tgt_chain_world = [tgt_world[ti][0] for ti in tgt_idxs]
            tgt_end_pos = tgt_chain_world[-1]
            gap_before = _world_position_gap(src_end_pos, tgt_end_pos)
            stat = chain_stats[chain_i]
            if gap_before > stat["max_gap_before"]:
                stat["max_gap_before"] = gap_before
                if stat["first_frame_before"] is None:
                    stat["first_frame_before"] = f
            if gap_before <= threshold:
                # Even if positional IK is skipped, rotation IK still
                # runs below (the wrist may be in the right place but
                # twisted differently from the source).
                solved_chain = False
            else:
                stat["frames_above_threshold"] += 1
                solved_chain = True
            if solved_chain:
                # Solve FABRIK on the chain.
                new_chain = _solve_chain_to_target(
                    tgt_chain_world, src_end_pos,
                    iterations=iterations, tol=threshold,
                )
                # Pole-vector check (v3): if the elbow/knee bent the
                # wrong way (e.g. through the body), reflect the chain
                # across the start↔end axis so the bend points toward
                # the pole hint.
                pole_dir = _resolve_pole_direction(
                    src_world, src_idxs, pole_src, pole_axis,
                )
                if pole_dir is not None and len(new_chain) >= 3:
                    if _chain_needs_pole_flip(new_chain, pole_dir):
                        new_chain = _mirror_chain_across_axis(new_chain)
                        stat["pole_corrections"] += 1
                new_end_pos = new_chain[-1]
                gap_after = _world_position_gap(src_end_pos, new_end_pos)
                if gap_after > stat["max_gap_after"]:
                    stat["max_gap_after"] = gap_after
                    if stat["first_frame_after"] is None:
                        stat["first_frame_after"] = f
                # Convert new joint positions back to per-bone local
                # rotations. For each bone i in the chain, the original
                # bone-axis direction was (old_chain[i+1] - old_chain[i])
                # and the new is (new_chain[i+1] - new_chain[i]). The
                # per-bone delta-rotation that aligns the two — applied in
                # PARENT world space — is what we need to add to the bone's
                # current local rotation.
                for k in range(len(tgt_idxs) - 1):
                    ti = tgt_idxs[k]
                    old_dir = (
                        tgt_chain_world[k + 1][0] - tgt_chain_world[k][0],
                        tgt_chain_world[k + 1][1] - tgt_chain_world[k][1],
                        tgt_chain_world[k + 1][2] - tgt_chain_world[k][2],
                    )
                    new_dir = (
                        new_chain[k + 1][0] - new_chain[k][0],
                        new_chain[k + 1][1] - new_chain[k][1],
                        new_chain[k + 1][2] - new_chain[k][2],
                    )
                    world_delta = _rotation_to_align(old_dir, new_dir)
                    # The bone's local rotation lives in its PARENT's frame.
                    # Convert ``world_delta`` from world to parent-local:
                    #   local_delta = parent_world_inv * world_delta * parent_world
                    # Approximation: we apply directly in world space when
                    # the parent's world rotation is approximately identity
                    # (true for many lobby_girl chains because the upper-body
                    # bones inherit a near-identity parent). The accurate path
                    # is below.
                    parent_idx = -1
                    if hasattr(target_skeleton[ti], "parent_idx"):
                        parent_idx = int(target_skeleton[ti].parent_idx)
                    elif hasattr(target_skeleton[ti], "parent"):
                        parent_idx = int(target_skeleton[ti].parent)
                    if parent_idx >= 0 and parent_idx < n_target:
                        par_pos, par_rot_mat = tgt_world[parent_idx]
                        # par_rot_mat is a 3x3 row-major; convert to quat.
                        par_q = _mat3_to_quat(par_rot_mat)
                        par_q_inv = _quat_inverse(par_q)
                        # local_delta = par_inv * world_delta * par
                        local_delta = _quat_mul(par_q_inv, _quat_mul(world_delta, par_q))
                    else:
                        local_delta = world_delta
                    # New local rotation for bone ``ti`` (offset form for
                    # the writer): bind_inv * local_delta * (current_local_full)
                    # but current_local_full = bind * current_offset, so
                    # new_offset = bind_inv * local_delta * bind * current_offset
                    # = bind_inv * (local_delta * current_full)
                    current_full = tgt_quats[ti]
                    new_full = _quat_mul(local_delta, current_full)
                    # Update tgt_quats so the next chain (e.g. opposite arm)
                    # sees the post-IK pose.
                    tgt_quats[ti] = new_full
                    # Re-encode as offset for the NJM writer.
                    bind_inv = _quat_inverse(tgt_bind_quats[ti])
                    offset_q = _quat_mul(bind_inv, new_full)
                    rx_b, ry_b, rz_b = quat_to_zyx_bams(*offset_q)
                    # Find this frame's keyframe and overwrite, or add a new one.
                    ang_track = target_bones[ti].tracks_by_kind.get(NJD_MTYPE_ANG)
                    if ang_track is None:
                        # Bone wasn't animated by 1:1 quat copy; create a track.
                        ang_track = NjmTrack(
                            kind=NJD_MTYPE_ANG, keyframes=[], narrow=True,
                        )
                        target_bones[ti].tracks_by_kind[NJD_MTYPE_ANG] = ang_track
                    _replace_or_insert_ang_kf(ang_track, f, rx_b, ry_b, rz_b)
                any_chain_solved = True

            # ---- Rotation IK (v3) -------------------------------------------
            # After positional IK, the end-effector bone is in the right
            # WORLD POSITION but its LOCAL rotation is still whatever the
            # 1:1 quat copy wrote, plus whatever the parent's chain
            # rotation now imposes. To make a Mixamo wrist twist propagate,
            # set the end bone's WORLD rotation to match the source's, then
            # subtract the parent's NEW world rotation to get the local
            # rotation we need to write back.
            if apply_rotation_ik and len(tgt_idxs) >= 2 and len(src_idxs) >= 2:
                # Source's end-effector world rotation.
                src_end_idx = src_idxs[-1]
                src_end_world_mat = src_world[src_end_idx][1]
                src_end_world_q = _mat3_to_quat(src_end_world_mat)
                if flip_z:
                    src_end_world_q = _mirror_quat_z(src_end_world_q)
                # Target's end bone: parent world (after IK chain mutation
                # — re-FK to capture the rotation changes we just wrote).
                # We refresh just the parent here rather than the whole
                # skeleton because only the chain's bones moved.
                tgt_world_after = _forward_kinematics(target_skeleton, tgt_quats)
                ti_end = tgt_idxs[-1]
                parent_idx = -1
                if hasattr(target_skeleton[ti_end], "parent_idx"):
                    parent_idx = int(target_skeleton[ti_end].parent_idx)
                elif hasattr(target_skeleton[ti_end], "parent"):
                    parent_idx = int(target_skeleton[ti_end].parent)
                if 0 <= parent_idx < n_target:
                    par_q_after = _mat3_to_quat(tgt_world_after[parent_idx][1])
                else:
                    par_q_after = (0.0, 0.0, 0.0, 1.0)
                par_q_after_inv = _quat_inverse(par_q_after)
                # local_full = parent_world_inv * source_world_rot
                new_full_end = _quat_mul(par_q_after_inv, src_end_world_q)
                # Track previous full quat to measure rotation IK delta.
                prev_full = tgt_quats[ti_end]
                tgt_quats[ti_end] = new_full_end
                # Encode as offset for the NJM writer.
                bind_inv_end = _quat_inverse(tgt_bind_quats[ti_end])
                offset_end = _quat_mul(bind_inv_end, new_full_end)
                rx_b, ry_b, rz_b = quat_to_zyx_bams(*offset_end)
                ang_track = target_bones[ti_end].tracks_by_kind.get(NJD_MTYPE_ANG)
                if ang_track is None:
                    ang_track = NjmTrack(
                        kind=NJD_MTYPE_ANG, keyframes=[], narrow=True,
                    )
                    target_bones[ti_end].tracks_by_kind[NJD_MTYPE_ANG] = ang_track
                _replace_or_insert_ang_kf(ang_track, f, rx_b, ry_b, rz_b)
                stat["rotation_ik_frames_applied"] += 1
                # Track max angular delta (for diagnostics; report in deg).
                err_deg = _quat_angle_deg(prev_full, new_full_end)
                if err_deg > stat["rotation_ik_max_err_deg"]:
                    stat["rotation_ik_max_err_deg"] = err_deg
                any_chain_solved = True

        if any_chain_solved:
            frames_solved += 1
            # Refresh tgt_world for the NEXT frame's parent-rotation
            # math (tgt_quats was mutated above). We don't need to
            # recompute now — next frame's loop top does FK again.

    return {
        "chains": chain_stats,
        "frames_solved": frames_solved,
        "frame_count": n_frames,
        "rotation_ik_enabled": bool(apply_rotation_ik),
        "inferred": bool(inferred_used),
    }


def _mat3_to_quat(m: Tuple[float, ...]) -> Tuple[float, float, float, float]:
    """Convert a 3x3 row-major rotation matrix to a unit quaternion.

    Standard Shepperd's method (the maximum-trace branch).
    """
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = m
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    return _quat_normalize((qx, qy, qz, qw))


def _replace_or_insert_ang_kf(
    track: NjmTrack,
    frame: int,
    rx: int, ry: int, rz: int,
) -> None:
    """Set the ANG keyframe at ``frame`` to (rx, ry, rz), inserting if absent.

    Keeps the keyframe list ordered by ``frame``. Used by the IK pass
    to overwrite the 1:1-copy keyframe values without disturbing other
    frames or the narrow/wide layout choice.
    """
    rx_u = int(rx) & 0xFFFF
    ry_u = int(ry) & 0xFFFF
    rz_u = int(rz) & 0xFFFF
    for i, kf in enumerate(track.keyframes):
        if kf[0] == frame:
            track.keyframes[i] = (frame, rx_u, ry_u, rz_u)
            return
        if kf[0] > frame:
            track.keyframes.insert(i, (frame, rx_u, ry_u, rz_u))
            return
    track.keyframes.append((frame, rx_u, ry_u, rz_u))


def _estimate_skeleton_scale(
    source: List[ImportedBone],
    target: Sequence,
) -> float:
    """Estimate a translation scale by ratio of skeleton Y-extents.

    For a 1.7 m Mixamo skeleton retargeting onto a PSOBB skeleton
    where Y-extent is ~25 units, this returns ~14.7. Rough but good
    enough for "don't fly off the screen".
    """
    def _yspread(bones: Sequence) -> float:
        ys = []
        for b in bones:
            if hasattr(b, "bind_pos"):
                ys.append(float(b.bind_pos[1]))
            elif hasattr(b, "position"):
                ys.append(float(b.position[1]))
        if not ys:
            return 1.0
        return max(ys) - min(ys)
    s_y = _yspread(source)
    t_y = _yspread(target)
    if s_y < 1e-6 or t_y < 1e-6:
        return 1.0
    return t_y / s_y


# ---------------------------------------------------------------------------
# Diagnostics: surface mapping for the server
# ---------------------------------------------------------------------------


def summarize_retarget(motion: NjmRawMotion) -> Dict[str, object]:
    """Return mapping diagnostics from a retargeted motion.

    Used by the server endpoints to surface "you mapped X bones, we
    dropped Y" so the user can iterate on the bone-name map.
    """
    dropped = getattr(motion, "_retarget_dropped", None) or []
    mapped = getattr(motion, "_retarget_mapped", None) or []
    ik = getattr(motion, "_retarget_ik", None) or {}
    mirror = getattr(motion, "_retarget_mirror", None) or {}
    resolution = getattr(motion, "_retarget_resolution", None) or {}
    return {
        "frame_count": motion.frame_count,
        "bone_count": len(motion.bones),
        "mapped_bones": len(mapped),
        "dropped_bones": len(dropped),
        "mapping": [{"src": s, "tgt": t} for (s, t) in mapped],
        "dropped": list(dropped)[:50],
        "ik": ik,
        "mirror": mirror,
        "resolution": resolution,
    }


__all__ = [
    "BoneNameMap",
    "LOBBY_GIRL_BONE_MAP",
    "HUMANOID_IK_CHAINS",
    "IkChainSpec",
    "auto_detect_bone_role",
    "detect_lr_pairs",
    "get_builtin_bone_map",
    "infer_ik_chains_from_skeleton",
    "mirror_animation",
    "retarget_animation",
    "summarize_retarget",
]
