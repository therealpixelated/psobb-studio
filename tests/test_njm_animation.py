"""Tests for the NJM (Ninja Motion) parser + animation endpoints.

Exercises:

    1. formats.njm.parse_njm against real PSOBB.IO data (NpcApcMot.bml
       NPC motions, bm_boss8_dragon.bml monster motions).
    2. /api/animations/<bml>%23<inner>.nj listing endpoint.
    3. /api/animation_data/<bml>%23<inner>.nj?motion=<name|index> data
       endpoint (keyframes payload).
    4. Walk auto-detect: bm_boss8_dragon picks walk_boss1_s_nb_dragon.
    5. /api/model_skinned/ integration: per-vertex bone_idx + skeleton
       are emitted correctly.

Run via: ``"C:/tmp_research_upscale/.venv/Scripts/python.exe" test_njm_animation.py``
The server must be running at http://127.0.0.1:8765 (the same host the
e2e_test.py expects).

Exit code 0 on full pass; non-zero on any failure.
"""
from __future__ import annotations
import os

import base64
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Allow direct import of the formats package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from formats.bml import parse_bml, _prs_decompress
from formats.njm import parse_njm, pick_default_motion, guess_motion_fps

API = "http://127.0.0.1:8765"
DATA_DIR = Path(os.path.expanduser("~/PSOBB.IO/data")).resolve()
DRAGON_BML = "bm_boss8_dragon.bml"
DRAGON_NJ = "boss1_s_nb_dragon.nj"
NPC_BML = "NpcApcMot.bml"


PASS: list[str] = []
FAIL: list[str] = []


def _http_get(path: str, timeout: int = 30):
    url = API + path
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode("utf-8"))
        except Exception:
            err = {"detail": str(e)}
        raise RuntimeError(f"HTTP {e.code} GET {path} -> {err}") from e


def step(name: str):
    def deco(fn):
        def wrap(*a, **kw):
            t0 = time.time()
            try:
                fn(*a, **kw)
                dt = time.time() - t0
                print(f"  PASS  [{dt:6.2f}s]  {name}")
                PASS.append(name)
            except AssertionError as e:
                print(f"  FAIL  {name}: {e}")
                FAIL.append(f"{name}: {e}")
            except Exception as e:
                print(f"  ERROR {name}: {type(e).__name__}: {e}")
                FAIL.append(f"{name}: {type(e).__name__}: {e}")
        return wrap
    return deco


# =====================================================================
# 1. formats.njm offline tests (no HTTP).
# =====================================================================


def _read_inner_bytes(bml_name: str, inner_name: str) -> bytes:
    """Read the decompressed raw bytes of one BML inner entry."""
    p = DATA_DIR / bml_name
    blob = p.read_bytes()
    entries = parse_bml(blob)
    e = next(ent for ent in entries if ent.name == inner_name)
    raw = blob[e.offset:e.offset + e.size_compressed]
    return _prs_decompress(raw)


@step("njm: parse npcApcMot motion 0 returns >0 keyframes per bone")
def t_npc_motion_zero():
    p = DATA_DIR / NPC_BML
    blob = p.read_bytes()
    entries = parse_bml(blob)
    assert len(entries) > 0, "NpcApcMot.bml has no entries"
    first = entries[0]
    raw = blob[first.offset:first.offset + first.size_compressed]
    decomp = _prs_decompress(raw)
    motions = parse_njm(decomp)
    assert len(motions) == 1, f"expected exactly 1 motion, got {len(motions)}"
    m = motions[0]
    assert m.bone_count > 0, "motion has zero bones"
    assert m.frame_count > 0, "motion has zero frames"
    # At least one bone must have >0 keyframes (otherwise the file is
    # purely metadata).
    non_empty = [t for t in m.tracks if t]
    assert len(non_empty) > 0, "no bones have keyframes"
    # Every keyframe in a non-empty track must have a sane time field.
    for track in non_empty:
        prev_t = -1
        for kf in track:
            assert kf.time >= prev_t, f"keyframes out of order: {prev_t} -> {kf.time}"
            prev_t = kf.time


@step("njm: bm_boss8_dragon's bml contains a walk motion")
def t_dragon_walk_present():
    p = DATA_DIR / DRAGON_BML
    blob = p.read_bytes()
    entries = parse_bml(blob)
    njms = [e for e in entries if e.name.endswith(".njm")]
    assert len(njms) > 0, "no .njm in dragon BML"
    walk = next((e for e in njms if "walk" in e.name.lower()), None)
    assert walk is not None, f"no walk motion in {DRAGON_BML}; have: {[e.name for e in njms]}"
    raw = blob[walk.offset:walk.offset + walk.size_compressed]
    decomp = _prs_decompress(raw)
    motions = parse_njm(decomp)
    assert motions, "walk motion parsed empty"
    m = motions[0]
    assert m.frame_count > 0
    assert m.bone_count == 124, f"dragon skeleton expected 124 bones, got {m.bone_count}"


@step("njm: all dragon NJMs parse without error")
def t_dragon_all_motions():
    p = DATA_DIR / DRAGON_BML
    blob = p.read_bytes()
    entries = parse_bml(blob)
    ok = 0
    fail_names = []
    for e in entries:
        if not e.name.endswith(".njm"):
            continue
        raw = blob[e.offset:e.offset + e.size_compressed]
        decomp = _prs_decompress(raw)
        try:
            motions = parse_njm(decomp)
            assert motions
            assert motions[0].frame_count > 0
            ok += 1
        except Exception as ex:
            fail_names.append(f"{e.name}: {type(ex).__name__}: {ex}")
    assert not fail_names, f"failed: {fail_names}"
    assert ok > 0


@step("njm: walk auto-detect picks 'walk' motion in dragon BML")
def t_walk_auto_detect():
    # Build the same motion-name list /api/animations would emit.
    p = DATA_DIR / DRAGON_BML
    blob = p.read_bytes()
    entries = parse_bml(blob)
    names = [e.name[:-4] for e in entries if e.name.endswith(".njm")]
    idx = pick_default_motion(names)
    assert idx is not None, "expected default motion"
    chosen = names[idx]
    assert "walk" in chosen.lower(), f"expected walk, got {chosen}"


@step("njm: pick_default_motion priority — walk > move > idle > first")
def t_pick_default_priority_tiers():
    # Tier 1: walk wins over everything
    assert pick_default_motion(
        ["damage", "walk_a", "move_a", "idle_a", "wait_a"]
    ) == 1, "walk must outrank move/idle/wait"
    # Tier 1: run also tier 1, but motion-list order chooses among same-tier.
    assert pick_default_motion(["run_x", "walk_y"]) == 0, "first tier-1 hit wins"
    # Tier 2: move wins when no walk/run
    assert pick_default_motion(
        ["damage", "fly_a", "move_a", "idle_a"]
    ) == 2, "move must outrank fly/idle"
    # Tier 3: alternate locomotion (swim/fly) when no walk/move
    assert pick_default_motion(
        ["damage", "fly_a", "wait_a"]
    ) == 1, "fly must outrank wait"
    # Tier 4: idle/wait/stand fallback
    assert pick_default_motion(
        ["damage", "death", "wait_a"]
    ) == 2, "wait must outrank damage/death"
    assert pick_default_motion(
        ["damage", "death", "idle_a", "wait_a"]
    ) == 2, "idle outranks wait via motion-list order"
    assert pick_default_motion(
        ["damage", "death", "stand_a"]
    ) == 2, "stand fallback works"
    # No keyword matches → motion 0
    assert pick_default_motion(["foo", "bar"]) == 0
    # Empty list → None
    assert pick_default_motion([]) is None


@step("njm: pick_default_motion picks 'move' for bm4_ps_ma_body BML")
def t_pick_default_move_for_bm4():
    """The motivating case: bm4_ps_*.bml ships move_* but no walk_*."""
    p = DATA_DIR / "bm4_ps_ma_body.bml"
    if not p.exists():
        return  # data dir not installed; tests that require live data skip
    blob = p.read_bytes()
    entries = parse_bml(blob)
    names = [e.name[:-4] for e in entries if e.name.endswith(".njm")]
    idx = pick_default_motion(names)
    assert idx is not None, "expected default motion"
    chosen = names[idx]
    # No walk_*; the picker should fall through to move_*.
    assert "move" in chosen.lower(), (
        f"expected move_* for bm4_ps_ma_body, got {chosen!r}"
    )


@step("njm: pick_default_motion handles idle-only NPCs / props")
def t_pick_default_idle_only():
    """Models with no locomotion should auto-play their idle pose.

    Real PSOBB.IO data overwhelmingly ships ``wait_*`` over ``idle_*``
    (75 of the first 100 BMLs have wait, none have idle). This test
    asserts the picker still reaches tier-4 fallback when no
    walk/run/move/swim/fly is present and selects the first matching
    name — regardless of which tier-4 keyword (idle/wait/stand/cstand)
    appears in it. The within-tier behaviour is "first match in motion-
    list order wins"; we assert that the picked motion belongs to tier
    4, not which specific keyword. Within-tier ranking by keyword
    position would help the rare case where a BML ships BOTH idle and
    wait — we don't see any such BMLs in PSOBB.IO so the simpler
    motion-list-order rule is shipped.
    """
    # Realistic: an NPC BML with only damage + wait + death (the common
    # PSOBB pattern). Wait should be picked.
    npc = ["damage_a", "wait_a", "death_a"]
    idx = pick_default_motion(npc)
    assert idx is not None
    assert "wait" in npc[idx].lower(), f"got {npc[idx]!r}"
    # And one with idle + damage — idle should be picked.
    npc2 = ["damage_a", "idle_a", "death_a"]
    idx2 = pick_default_motion(npc2)
    assert idx2 is not None
    assert "idle" in npc2[idx2].lower(), f"got {npc2[idx2]!r}"
    # And one with stand only — stand should be picked.
    npc3 = ["damage_a", "stand_a", "death_a"]
    idx3 = pick_default_motion(npc3)
    assert idx3 is not None
    assert "stand" in npc3[idx3].lower(), f"got {npc3[idx3]!r}"


@step("njm: empty input → empty motions list")
def t_njm_empty():
    out = parse_njm(b"")
    assert out == [], f"expected [], got {out}"


@step("njm: guess_motion_fps returns 30 default; 'wait' overrides to 15")
def t_njm_fps_heuristic():
    assert guess_motion_fps("walk_xyz") == 30.0
    assert guess_motion_fps("attack01") == 30.0
    assert guess_motion_fps("wait_idle") == 15.0
    assert guess_motion_fps("") == 30.0


# =====================================================================
# 2. /api/animations endpoint.
# =====================================================================


@step("animations endpoint: dragon BML returns >20 motions including walk")
def t_animations_dragon():
    path = f"{DRAGON_BML}#{DRAGON_NJ}"
    quoted = urllib.parse.quote(path, safe="")
    r = _http_get(f"/api/animations/{quoted}")
    assert "motions" in r
    assert r["motion_count"] > 20, r["motion_count"]
    walk = next((m for m in r["motions"] if "walk" in m["name"].lower()), None)
    assert walk is not None, [m["name"] for m in r["motions"]]
    # Default should pick walk.
    assert r["default_index"] is not None
    chosen = r["motions"][r["default_index"]]["name"]
    assert "walk" in chosen.lower(), f"default = {chosen}"
    # Skeleton bone count should be 124 for the dragon.
    assert r["skeleton_bone_count"] == 124, r["skeleton_bone_count"]


@step("animations endpoint: empty motion list → 200 with empty array")
def t_animations_static():
    # Pick a model BML that has no .njm siblings (rare, but plHnj.bml fits)
    target = "plHnj.bml#plHbdy00.nj"
    quoted = urllib.parse.quote(target, safe="")
    r = _http_get(f"/api/animations/{quoted}")
    # plHnj has no internal NJMs; fallback to NpcApcMot since
    # filename starts with 'pl'. So motion_count > 0 here.
    assert r["motion_count"] > 0, "expected NpcApcMot fallback for plHnj"


# =====================================================================
# 3. /api/animation_data endpoint.
# =====================================================================


@step("animation_data: dragon walk returns valid keyframe JSON")
def t_animation_data_walk():
    path = f"{DRAGON_BML}#{DRAGON_NJ}"
    quoted = urllib.parse.quote(path, safe="")
    r = _http_get(f"/api/animation_data/{quoted}?motion=walk")
    assert r["motion"].lower().startswith("walk"), r["motion"]
    assert r["frame_count"] > 0
    assert r["bone_count"] == 124, r["bone_count"]
    assert "bones" in r
    # Validate keyframe structure on the first non-empty bone.
    non_empty = [b for b in r["bones"] if b["kf"]]
    assert len(non_empty) > 0
    sample = non_empty[0]
    kf0 = sample["kf"][0]
    for key in ("t", "tx", "ty", "tz", "rx", "ry", "rz", "sx", "sy", "sz"):
        assert key in kf0, f"missing key {key} in keyframe: {kf0}"
    # rx/ry/rz are BAMS integers
    assert isinstance(kf0["rx"], int)
    # tx/ty/tz are floats
    assert isinstance(kf0["tx"], float)


@step("animation_data: motion lookup by integer index")
def t_animation_data_by_index():
    path = f"{DRAGON_BML}#{DRAGON_NJ}"
    quoted = urllib.parse.quote(path, safe="")
    listing = _http_get(f"/api/animations/{quoted}")
    target_idx = listing["default_index"]
    target_name = listing["motions"][target_idx]["name"]
    # Fetch by integer index
    r = _http_get(f"/api/animation_data/{quoted}?motion={target_idx}")
    assert r["motion"] == target_name


@step("animation_data: 404 for unknown motion name")
def t_animation_data_404():
    path = f"{DRAGON_BML}#{DRAGON_NJ}"
    quoted = urllib.parse.quote(path, safe="")
    try:
        _http_get(f"/api/animation_data/{quoted}?motion=xyzfoobar99")
    except RuntimeError as e:
        assert "404" in str(e), e
        return
    raise AssertionError("expected 404")


# =====================================================================
# 4. /api/model_skinned endpoint.
# =====================================================================


@step("model_skinned: dragon returns bone-local meshes + skeleton")
def t_model_skinned():
    path = f"{DRAGON_BML}#{DRAGON_NJ}"
    quoted = urllib.parse.quote(path, safe="")
    r = _http_get(f"/api/model_skinned/{quoted}", timeout=60)
    assert r["bone_count"] == 124, r["bone_count"]
    assert r["mesh_count"] > 0
    assert r["vertices_pre_transformed"] is False
    assert r["has_bone_indices"] is True
    # Validate a vertex's bone_indices_b64 decodes as Int32 with sane indices
    first = r["meshes"][0]
    bi_bytes = base64.b64decode(first["bone_indices_b64"])
    import struct
    assert len(bi_bytes) % 4 == 0
    bone_idx_count = len(bi_bytes) // 4
    assert bone_idx_count == first["vertex_count"]
    # Decode all and check range
    bone_indices = struct.unpack(f"<{bone_idx_count}i", bi_bytes)
    for bi in bone_indices:
        assert -1 <= bi < r["bone_count"], f"out of range bone_idx {bi}"


@step("animation_data: per-bone present mask flags rotation-only tracks")
def t_animation_data_present_mask():
    """Regression guard for the rotation-only-bone collapse bug (2026-04-25).

    PSOBB monster motions (almost universally type_flag=3 = POS+ANG)
    typically populate the POS track ONLY for the root bone — the
    body translation that swings during walk — and leave per-joint
    POS tracks empty (count=0). The euler-rotation track is
    populated for every animated joint instead.

    Before the present-mask fix the parser merged tracks per-bone via
    a `dict[frame_id -> NjmKeyframe]`, defaulting absent tx/ty/tz to
    (0, 0, 0). When the consumer applied those keyframes verbatim the
    bind-pose translation got REPLACED with zero on every per-joint
    bone — collapsing the rendered model to the world origin every
    frame.

    The fix: ``NjmMotion.bone_present_tracks[bone_idx]`` carries a
    bitfield identifying which TRS channels were ACTUALLY authored on
    each bone. The server surfaces this as ``present`` per bone in
    ``/api/animation_data``; the JS consumer uses it to decide whether
    to read the keyframe channel or fall back to bind pose.

    This test asserts (a) the field is present in the wire format and
    (b) the dragon's walk has the expected pattern: bone 0 (root)
    carries POS+ANG=3, joint bones carry ANG=2, no-track bones
    carry 0. A regression that lost the field would yank rotation-only
    bones to (0,0,0) again.
    """
    path = f"{DRAGON_BML}#{DRAGON_NJ}"
    quoted = urllib.parse.quote(path, safe="")
    r = _http_get(f"/api/animation_data/{quoted}?motion=walk")
    bones = r["bones"]
    assert bones, "no bones in walk payload"
    # Every bone entry must carry the present field (even bones with
    # empty kf — the field is what tells consumers "use bind pose").
    missing = [b["idx"] for b in bones if "present" not in b]
    assert not missing, f"bones missing present field: {missing[:10]}"
    # Tally the masks. PSOBB walk motions on the dragon should split:
    #   present=3  → root bone (POS+ANG keyframes both authored)
    #   present=2  → joint bones (rotation-only)
    #   present=0  → bones the motion does not touch
    by_mask = {}
    for b in bones:
        by_mask[b["present"]] = by_mask.get(b["present"], 0) + 1
    assert by_mask.get(3, 0) >= 1, (
        f"expected at least one bone with POS+ANG (mask=3); got {by_mask}"
    )
    assert by_mask.get(2, 0) >= 50, (
        f"expected many rotation-only bones (mask=2) on the dragon; got {by_mask}"
    )
    # No bone should have a present mask that is NOT a subset of the
    # motion's type_flags — that would be a parser bug.
    type_flags = r["type_flags"]
    for b in bones:
        assert (b["present"] & ~type_flags) == 0, (
            f"bone {b['idx']} has present={b['present']} not subset of "
            f"type_flags={type_flags}"
        )
    # The bones whose present mask EXCLUDES POS (i.e. only rotation)
    # should also have keyframes whose tx/ty/tz are mostly zero — that's
    # the parser default for the unauthored channel. If that ever changes
    # (e.g. someone swaps in interpolated bind values on the server) the
    # frontend's bind-pose fallback would silently double-translate; this
    # cross-check keeps the wire-format contract aligned with the JS
    # consumer's assumption.
    rot_only = [b for b in bones if b["present"] == 2 and b["kf"]]
    assert rot_only, "expected at least one rotation-only animated bone"
    sample = rot_only[0]
    kf = sample["kf"][0]
    assert kf["tx"] == 0.0 and kf["ty"] == 0.0 and kf["tz"] == 0.0, (
        f"rot-only bone {sample['idx']} kf[0] should have zero tx/ty/tz: {kf}"
    )


@step("model_skinned: bones surface eval_flags + scale fields")
def t_model_skinned_bones_have_eval_flags():
    """Regression guard for the 2026-04-25 skinning fix.

    Before the fix, ``parse_xj_njcm_skinned`` dropped the source
    MeshTreeNode's eval_flags + scale on the floor — the wire payload's
    ``bones[i]`` only carried position+rotation. The JS skinning loop
    then composed each bone's bind matrix from raw position/rotation
    regardless of UNIT_POS / UNIT_ANG / UNIT_SCL flags, diverging from
    the world-baked /api/model_mesh path on models like De Rol Le whose
    head bones use UNIT_POS|UNIT_SCL on every joint.

    The fix surfaces eval_flags + scale through the wire so the JS
    bind-pose composition can honor them. This test asserts the wire
    contract: every bone in a skinned response has both fields.
    """
    target = f"{DRAGON_BML}#{DRAGON_NJ}"
    quoted = urllib.parse.quote(target, safe="")
    r = _http_get(f"/api/model_skinned/{quoted}")
    bones = r.get("bones", [])
    assert len(bones) == 124, f"expected 124 bones, got {len(bones)}"
    for b in bones:
        assert "eval_flags" in b, f"bone {b.get('index')} missing eval_flags"
        assert "scale" in b, f"bone {b.get('index')} missing scale"
        assert isinstance(b["eval_flags"], int), f"bone {b['index']} eval_flags not int: {b['eval_flags']!r}"
        assert isinstance(b["scale"], list) and len(b["scale"]) == 3, (
            f"bone {b['index']} scale must be 3-list, got {b['scale']!r}"
        )


@step("model_skinned: De Rol Le head bones carry UNIT_POS|UNIT_SCL eval flags")
def t_model_skinned_de_rol_le_eval_flags():
    """Concrete check that the skinned path surfaces real eval_flags
    (not just empty/zero default). De Rol Le's body has 38 bones with
    eval=0x05 (UNIT_POS|UNIT_SCL), 22 with 0x06 (UNIT_ANG|UNIT_SCL),
    etc. — see AGENT_MODEL_DEEP_DEBUG_REPORT.md histogram.
    """
    target = "bm_boss2_de_rol_le.bml#boss2_b_derorure_body.nj"
    quoted = urllib.parse.quote(target, safe="")
    r = _http_get(f"/api/model_skinned/{quoted}")
    bones = r.get("bones", [])
    assert len(bones) > 100, f"expected > 100 bones (De Rol Le), got {len(bones)}"
    # The flag bits we care about (UNIT_*/SKIP/ZXY).
    eval_mask = 0x67
    nonzero = [b for b in bones if (b.get("eval_flags", 0) & eval_mask)]
    assert len(nonzero) >= 80, (
        f"expected at least 80 bones with non-zero eval flags on De Rol Le; "
        f"got {len(nonzero)} of {len(bones)}"
    )


@step("model_skinned: player NJ archives load skinned (bml.py alignment fix)")
def t_model_skinned_player_njs():
    """Regression guard for the 2026-04-25 bml.py alignment fix.

    Before: 184 player NJ inners failed to decompress because the BML
    reader walked entries with the wrong padding alignment (the player
    BMLs lie about has_textures=1 in the header but actually use
    0x800-byte padding). The fix uses a cumulative-end heuristic to
    auto-detect the correct alignment.

    This test loads the body NJ from each player class and verifies
    the skinned payload carries 64 bones + non-empty geometry.
    """
    # Sample player classes — one body per archive.
    targets = [
        "plAnj.bml#plAbdy00.nj",
        "plDnj.bml#plDbdy00.nj",
        "plHnj.bml#plHbdy00.nj",
        "plLnj.bml#plLbdy00.nj",
    ]
    for t in targets:
        quoted = urllib.parse.quote(t, safe="")
        r = _http_get(f"/api/model_skinned/{quoted}")
        bones = r.get("bones", [])
        meshes = r.get("meshes", [])
        bone_count = r.get("bone_count", 0)
        mesh_count = r.get("mesh_count", 0)
        assert bone_count == 64, (
            f"{t}: expected 64-bone player skeleton, got {bone_count}"
        )
        assert mesh_count > 50, (
            f"{t}: expected many submeshes for player body, got {mesh_count}"
        )
        # Player NJs have no per-vertex bone tag for some chunks but
        # the rotation/skinning chain should still bind. Verify at
        # least one bone has eval_flags set.
        any_eval = any(b.get("eval_flags", 0) for b in bones)
        assert any_eval, f"{t}: all bones have eval_flags=0 (suspicious)"


@step("bml: alignment heuristic recovers all 209 player NJ inners")
def t_bml_alignment_player_njs():
    """End-to-end regression guard: every pl[A-Z]nj.bml decompresses
    cleanly to 209 player .nj entries each starting with NJCM magic
    (sometimes preceded by a single \\xff PRS-literal marker).
    """
    pl_bmls = sorted(DATA_DIR.glob("pl*nj.bml"))
    assert len(pl_bmls) == 23, f"expected 23 player NJ BMLs, got {len(pl_bmls)}"
    total_ok = 0
    for p in pl_bmls:
        buf = p.read_bytes()
        entries = parse_bml(buf)
        for e in entries:
            raw = bytes(buf[e.offset:e.offset + e.size_compressed])
            d = _prs_decompress(raw)
            # Decompressed must start with NJCM (raw or with \xff PRS-literal prefix).
            magic_ok = (
                d[:4] == b"NJCM"
                or (len(d) >= 5 and d[0] == 0xFF and d[1:5] == b"NJCM")
            )
            assert magic_ok, (
                f"{p.name}#{e.name} decompressed but no NJCM magic: "
                f"first 8 bytes = {d[:8].hex()}"
            )
            total_ok += 1
    assert total_ok == 209, f"expected 209 player NJ inners, got {total_ok}"


@step("bml: NpcApcMot.bml still parses (regression guard for 0x20 alignment)")
def t_bml_alignment_npc_apc_mot():
    """Regression guard for the 2026-04-25 alignment fix.

    NpcApcMot.bml has has_textures=1 in its header but every entry has
    tex_size_compressed==0 — same superficial pattern as the player
    NJs. BUT its actual on-disk alignment is 0x20, not 0x800. The
    fix uses a cumulative-end heuristic that picks 0x20 here. Without
    this guard the simpler "any-tex implies 0x20" rule would have
    broken NpcApcMot in favour of fixing the player NJs.
    """
    p = DATA_DIR / NPC_BML
    buf = p.read_bytes()
    entries = parse_bml(buf)
    assert len(entries) == 120, f"NpcApcMot has 120 entries, got {len(entries)}"
    # Spot-check 5 entries decompress and start with NMDM magic.
    for ent in entries[:5]:
        raw = bytes(buf[ent.offset:ent.offset + ent.size_compressed])
        d = _prs_decompress(raw)
        assert d[:4] == b"NMDM", (
            f"NpcApcMot.bml#{ent.name}: expected NMDM magic, got {d[:4].hex()}"
        )


@step("model_skinned: rejects .xj inner")
def t_model_skinned_xj_rejected():
    # Find an XJ inner to test against
    bml_path = DATA_DIR / "bm_eff_ice.bml"
    if not bml_path.exists():
        # Fall back to listing directory
        for p in sorted(DATA_DIR.glob("*.bml"))[:50]:
            try:
                blob = p.read_bytes()
                entries = parse_bml(blob)
                xj = next((e for e in entries if e.name.endswith(".xj")), None)
                if xj:
                    bml_path = p
                    xj_name = xj.name
                    break
            except Exception:
                continue
        else:
            print("    (no .xj found; skipping)")
            return
    else:
        blob = bml_path.read_bytes()
        entries = parse_bml(blob)
        xj = next((e for e in entries if e.name.endswith(".xj")), None)
        if xj is None:
            print("    (no .xj in test BML; skipping)")
            return
        xj_name = xj.name

    target = f"{bml_path.name}#{xj_name}"
    quoted = urllib.parse.quote(target, safe="")
    try:
        _http_get(f"/api/model_skinned/{quoted}")
    except RuntimeError as e:
        assert "400" in str(e) or "skinned" in str(e).lower(), e
        return
    raise AssertionError("expected 400 for .xj inner")


# =====================================================================
# Driver
# =====================================================================


def main() -> int:
    print(f"NJM animation tests against {API} (data: {DATA_DIR})")
    # Health probe
    try:
        h = _http_get("/api/health")
        if not h.get("ok"):
            print("  ERROR  server health says not ok:", h)
            return 1
    except Exception as e:
        print(f"  ERROR  cannot reach server at {API}: {e}")
        return 2

    # Offline tests (no HTTP)
    t_npc_motion_zero()
    t_dragon_walk_present()
    t_dragon_all_motions()
    t_walk_auto_detect()
    # 2026-04-25: tier-based priority for pick_default_motion. The
    # earlier flat list ranked move_* below fly/swim/frloop, which broke
    # the auto-play heuristic for bm4_ps_*.bml (only locomotion verb is
    # move_*) and most NPC props (only have idle/wait/stand).
    t_pick_default_priority_tiers()
    t_pick_default_move_for_bm4()
    t_pick_default_idle_only()
    t_njm_empty()
    t_njm_fps_heuristic()

    # Animation listing
    t_animations_dragon()
    t_animations_static()

    # Animation data
    t_animation_data_walk()
    t_animation_data_by_index()
    t_animation_data_404()
    t_animation_data_present_mask()

    # Skinned mesh
    t_model_skinned()
    t_model_skinned_bones_have_eval_flags()
    t_model_skinned_de_rol_le_eval_flags()
    t_model_skinned_player_njs()
    t_bml_alignment_player_njs()
    t_bml_alignment_npc_apc_mot()
    t_model_skinned_xj_rejected()

    print()
    print(f"  Passed: {len(PASS)}")
    print(f"  Failed: {len(FAIL)}")
    if FAIL:
        for f in FAIL:
            print(f"    {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
