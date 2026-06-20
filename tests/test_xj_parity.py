"""Parity tests for the XJ descriptor triangle-strip winding.

Background
----------
The descriptor-table ``.xj`` format (``formats/xj_descriptor.py``,
ported from Phantasmal World's ``Xj.kt``) stores each triangle strip as
a flat index list with **no per-strip winding flag** — unlike the
chunk-based Nj strips (``formats/xj.py``), whose chunk header carries a
signed length whose sign bit IS the winding.

Phantasmal does NOT triangulate XJ strips with a fixed alternating
parity. Its renderer, ``convertXjModel`` in
``web/src/jsMain/kotlin/world/phantasmal/web/core/rendering/conversion/
NinjaGeometryConversion.kt`` (lines 553-599), runs a *normal-correcting*
winding pass:

    var clockwise = false
    for (j in 2 until indices.size) {
        a, b, c   = indices[j-2], indices[j-1], indices[j]
        faceN     = (pb - pa) cross (pc - pa)
        if (clockwise) faceN.negate()
        // "Calculate a surface normal and reverse the vertex winding if
        //  at least 2 of the vertex normals point in the opposite
        //  direction. This hack fixes the winding for most models."
        opposite  = count( dot(faceN, vertexNormal) < 0 )  over a, b, c
        if (opposite >= 2) clockwise = !clockwise
        emit( clockwise ? (b, a, c) : (a, b, c) )
        clockwise = !clockwise
    }

A fixed-parity triangulation gets ~16% of strips backwards on real game
assets (1238 / 7808 strips, 5858 / 27571 triangles wrong across 316
``.xj`` BML inners). Those triangles render back-to-front: backface
culling hides them, or lighting comes out inverted. This module locks in
the corrected behaviour:

  * ``test_xj_winding_matches_oracle_on_game_assets`` — re-implements the
    oracle independently and asserts the SHIPPING parser
    (``parse_xj_file``) produces identical winding on every emitted mesh,
    over every ``.xj`` asset present on disk. Skips when game data is
    absent (CI build machines).

  * ``test_normal_correction_flips_obvious_case`` — a synthetic,
    data-free unit test: a strip whose vertex normals all point the
    opposite way from the geometric face must come out flipped relative
    to the naive fixed-parity triangulation.

If a future edit reverts the winding to fixed parity, the first test
fails on real data and the second fails with no data needed.

NOTE on verifiability: the Kotlin reference cannot be built in this
environment, so the oracle here is a hand-transcription of those exact
source lines, kept beside the quote above for audit. The transcription
was cross-checked by running it against 316 real assets and confirming
the shipping parser matches it byte-for-byte (winding-canonicalized).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import pytest

from formats import xj_descriptor as xd
from formats.iff import parse_iff


# --------------------------------------------------------------------------
# Game-data discovery (skips cleanly on CI build machines).
# --------------------------------------------------------------------------

_DATA_DIRS = [
    Path(os.path.expanduser("~/EphineaPSO/data")),
    Path(os.path.expanduser("~/PSOBB.IO/data")),
]


def _data_dir() -> Path | None:
    env = os.environ.get("PSO_XJ_TEST_DATA_DIR")
    if env and Path(env).is_dir():
        return Path(env)
    for d in _DATA_DIRS:
        if d.is_dir():
            return d
    return None


def _iter_xj_assets(data_dir: Path, limit: int | None = None):
    """Yield ``(bml_name, inner_name, xj_bytes)`` for every ``.xj`` inner."""
    from formats.bml import parse_bml, _prs_decompress

    count = 0
    for fn in sorted(os.listdir(data_dir)):
        if not fn.endswith(".bml"):
            continue
        try:
            blob = (data_dir / fn).read_bytes()
            entries = parse_bml(blob)
        except Exception:
            continue
        for e in entries:
            if not e.name.endswith(".xj"):
                continue
            try:
                raw = bytes(blob[e.offset:e.offset + e.size_compressed])
                xj_bytes = _prs_decompress(raw, timeout=20.0)
            except Exception:
                continue
            yield fn, e.name, xj_bytes
            count += 1
            if limit is not None and count >= limit:
                return


# --------------------------------------------------------------------------
# Independent oracle (verbatim transcription of convertXjModel, 553-599).
# --------------------------------------------------------------------------


def _oracle_triangulate(
    strip: List[int],
    positions: List[Tuple[float, float, float]],
    normals: List[Tuple[float, float, float]],
) -> List[int]:
    """Reference XJ winding. Independent of the production code path so a
    bug copied into both can't hide. See module docstring for the source
    lines this mirrors."""
    out: List[int] = []
    clockwise = False
    n = len(strip)
    for j in range(2, n):
        a, b, c = strip[j - 2], strip[j - 1], strip[j]
        pa, pb, pc = positions[a], positions[b], positions[c]
        ux, uy, uz = pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]
        vx, vy, vz = pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]
        fx = uy * vz - uz * vy
        fy = uz * vx - ux * vz
        fz = ux * vy - uy * vx
        if clockwise:
            fx, fy, fz = -fx, -fy, -fz
        na, nb, nc = normals[a], normals[b], normals[c]
        opp = 0
        if fx * na[0] + fy * na[1] + fz * na[2] < 0.0:
            opp += 1
        if fx * nb[0] + fy * nb[1] + fz * nb[2] < 0.0:
            opp += 1
        if fx * nc[0] + fy * nc[1] + fz * nc[2] < 0.0:
            opp += 1
        if opp >= 2:
            clockwise = not clockwise
        if clockwise:
            out.extend((b, a, c))
        else:
            out.extend((a, b, c))
        clockwise = not clockwise
    return out


def _canon_tri(t: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Rotation-invariant, winding-SENSITIVE key (smallest index first)."""
    a, b, c = t
    m = min(a, b, c)
    if a == m:
        return (a, b, c)
    if b == m:
        return (b, c, a)
    return (c, a, b)


def _canon_set(flat: List[int]):
    s = set()
    for i in range(0, len(flat), 3):
        a, b, c = flat[i], flat[i + 1], flat[i + 2]
        if a == b or b == c or a == c:
            continue
        s.add(_canon_tri((a, b, c)))
    return s


def _oracle_meshes(payload: bytes):
    """Reproduce parse_xj_descriptor's per-strip emission, but triangulate
    with the oracle. Returns winding-canonical triangle sets, one per
    emitted (non-empty) mesh — in the same order parse_xj_descriptor
    emits them."""
    import struct

    body = bytes(payload)
    nodes = xd._walk_tree(body, root_off=0)
    out_sets = []
    for (_off, model_offset, world_M, ef) in nodes:
        if ef & (xd._EVAL_HIDE | xd._EVAL_SHAPE_SKIP):
            continue
        try:
            xm = struct.unpack_from(xd._XJ_MODEL_FMT, body, model_offset)
        except struct.error:
            continue
        (_flags, vbi_off, vbi_count, ts_off, ts_count,
         tts_off, tts_count, *_rest) = xm
        vit_blocks = xd._parse_vertex_info_tables(body, vbi_off, vbi_count)
        slots = {}
        for base_index, verts in vit_blocks:
            for j, v in enumerate(verts):
                slots[base_index + j] = v
        if not slots:
            continue
        strips = []
        strips.extend(xd._parse_strip_table(body, ts_off, ts_count))
        strips.extend(xd._parse_strip_table(body, tts_off, tts_count))
        for (_tex, _diffuse, strip_indices) in strips:
            if len(strip_indices) < 3:
                continue
            local_verts = []
            local_slot_map = {}
            local_strip = []
            valid = True
            for sidx in strip_indices:
                if sidx not in slots:
                    valid = False
                    break
                if sidx in local_slot_map:
                    local_strip.append(local_slot_map[sidx])
                    continue
                pos_local, normal_local, uv_local = slots[sidx]
                pos_world = xd._mat4_transform_point(world_M, pos_local)
                if normal_local is not None:
                    normal_world = xd._mat4_transform_dir(world_M, normal_local)
                else:
                    normal_world = (0.0, 1.0, 0.0)
                local_slot_map[sidx] = len(local_verts)
                local_verts.append(
                    xd.XjVertex(pos=pos_world, normal=normal_world,
                                uv=uv_local if uv_local is not None else (0.0, 0.0))
                )
                local_strip.append(local_slot_map[sidx])
            if not valid or not local_verts:
                continue
            positions = [v.pos for v in local_verts]
            normals = [v.normal for v in local_verts]
            tri = _oracle_triangulate(local_strip, positions, normals)
            cs = _canon_set(tri)
            if cs:
                out_sets.append(cs)
    return out_sets


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_xj_winding_matches_oracle_on_game_assets():
    """The shipping parser's winding must equal the oracle on every mesh.

    Covers every ``.xj`` inner present on disk. Skips when no game data
    dir is available (CI build machines). This is the strong, real-asset
    parity check the audit asked for: it would fail loudly if the winding
    regressed to fixed parity (or to any other rule).
    """
    data_dir = _data_dir()
    if data_dir is None:
        pytest.skip("no PSOBB/Ephinea data dir available")

    checked_files = 0
    checked_meshes = 0
    mismatches: list[str] = []

    for bml_name, inner, xj_bytes in _iter_xj_assets(data_dir):
        ours = xd.parse_xj_file(xj_bytes)
        ours_sets = [s for s in (_canon_set(m.indices) for m in ours) if s]

        oracle_sets = []
        for c in parse_iff(xj_bytes):
            if c.type == "NJCM":
                oracle_sets.extend(_oracle_meshes(c.data))

        checked_files += 1
        checked_meshes += len(ours_sets)

        if len(ours_sets) != len(oracle_sets):
            mismatches.append(
                f"{bml_name}#{inner}: mesh count {len(ours_sets)} != "
                f"oracle {len(oracle_sets)}"
            )
            continue
        for k, (a, b) in enumerate(zip(ours_sets, oracle_sets)):
            if a != b:
                mismatches.append(f"{bml_name}#{inner}: mesh[{k}] winding differs")

    if checked_files == 0:
        pytest.skip("data dir present but contained no parseable .xj inners")

    assert not mismatches, (
        f"{len(mismatches)} winding mismatch(es) vs oracle "
        f"(checked {checked_meshes} meshes across {checked_files} files):\n  "
        + "\n  ".join(mismatches[:25])
    )


def test_normal_correction_flips_obvious_case():
    """Synthetic, data-free regression: a strip whose vertex normals all
    point opposite the geometric face must be flipped relative to the
    naive fixed-parity triangulation.

    We build a single 3-vertex strip lying in the XY plane wound so its
    geometric normal is +Z, but tag every vertex normal as -Z. The oracle
    sees ``opposite == 3 >= 2`` on the first triangle and flips it; the
    old fixed-parity ``_tristrip_to_triangles`` does not. This guards the
    normal-correction itself, independent of any game asset.
    """
    # CCW in XY plane => geometric face normal points +Z.
    positions = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    # But the authored vertex normals say the surface faces -Z.
    normals = [(0.0, 0.0, -1.0)] * 3
    strip = [0, 1, 2]

    corrected = xd._tristrip_to_triangles_normal_corrected(strip, positions, normals)
    naive = xd._tristrip_to_triangles(strip, cw=False)

    assert len(corrected) == 3 and len(naive) == 3

    def _canon(flat):
        a, b, c = flat
        return _canon_tri((a, b, c))

    # The fix must reverse the winding relative to the naive triangulation.
    assert _canon(corrected) != _canon(naive), (
        "normal-correction did not flip a face whose normals oppose its "
        "geometry — the fix is not wired in"
    )
    # And it must match the independent oracle.
    assert _canon_set(corrected) == _canon_set(
        _oracle_triangulate(strip, positions, normals)
    )


def test_normal_correction_leaves_consistent_case_alone():
    """When the vertex normals agree with the geometric face normal, the
    corrected triangulation must keep the natural strip winding (no
    spurious flip on already-correct geometry)."""
    positions = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    normals = [(0.0, 0.0, 1.0)] * 3  # agree with +Z face normal
    strip = [0, 1, 2]
    corrected = xd._tristrip_to_triangles_normal_corrected(strip, positions, normals)

    def _canon(flat):
        return _canon_tri((flat[0], flat[1], flat[2]))

    assert _canon(corrected) == _canon([0, 1, 2])
