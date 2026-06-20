"""Tests for the triangle-strip optimizer (``formats.strippify``) and its
n.rel authoring wiring (``formats.rel_writer.nrel_submeshes_stripified`` /
``nrel_nodes_from_meshes(stripify=True)``).

The correctness GATE is winding-insensitive: the engine recomputes winding
from vertex normals and the reader's ``_strip_to_triangles`` applies
alternating parity, so a round trip need only reproduce the same triangle
VERTEX SET (unordered triples), never the same index order.  Every
equality assertion here compares vertex SETS, never index order.

Run isolated::

    python -m pytest tests/test_strippify.py -q
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from formats import rel as _rel
from formats import rel_writer as _rw
from formats.rel_writer import (
    NREL_SIZE_BUDGET,
    build_nrel_from_meshes,
    nrel_nodes_from_meshes,
    nrel_submeshes_stripified,
)
from formats.strippify import (
    destripify,
    stripify,
    strip_triangle_count,
    triangle_set,
)


# ---------------------------------------------------------------------------
# geometry fixtures (no game assets needed)
# ---------------------------------------------------------------------------
def _grid(n: int):
    """An n×n ground grid -> 2*(n-1)^2 triangles, with positions + planar UVs."""
    xs = np.linspace(-200.0, 200.0, n)
    V = np.array([[x, 0.0, z] for z in xs for x in xs], dtype=np.float64)
    U = np.array([[(x + 200) / 400, (z + 200) / 400] for z in xs for x in xs],
                 dtype=np.float64)
    F = []
    for r in range(n - 1):
        for c in range(n - 1):
            a = r * n + c
            F.append([a, a + 1, a + n])
            F.append([a + 1, a + n + 1, a + n])
    return V, U, np.array(F, dtype=np.int64)


def _uv_sphere(stacks: int, slices: int):
    """A UV-sphere -> positions + (stacks*slices*2-ish) triangles."""
    verts = []
    for i in range(stacks + 1):
        phi = np.pi * i / stacks
        for j in range(slices + 1):
            theta = 2 * np.pi * j / slices
            x = np.sin(phi) * np.cos(theta)
            y = np.cos(phi)
            z = np.sin(phi) * np.sin(theta)
            verts.append([x, y, z])
    V = np.array(verts, dtype=np.float64)
    stride = slices + 1
    F = []
    for i in range(stacks):
        for j in range(slices):
            a = i * stride + j
            b = a + 1
            c = a + stride
            d = c + 1
            F.append([a, b, c])
            F.append([b, d, c])
    F = np.array(F, dtype=np.int64)
    # A real sphere top/bottom rows contain degenerate triangles (poles) —
    # keep them so the stripifier's degenerate handling is exercised.
    return V, F


class _Mesh:
    """Minimal XjMesh-shaped object for nrel_*_from_meshes."""

    class _V:
        __slots__ = ("pos", "normal", "uv")

        def __init__(self, pos, normal, uv):
            self.pos = pos
            self.normal = normal
            self.uv = uv

    def __init__(self, V, F, U=None, material_id=0):
        if U is None:
            U = np.zeros((V.shape[0], 2), dtype=np.float64)
        self.vertices = [
            self._V((float(V[i, 0]), float(V[i, 1]), float(V[i, 2])),
                    (0.0, 1.0, 0.0),
                    (float(U[i, 0]), float(U[i, 1])))
            for i in range(V.shape[0])
        ]
        self.indices = [int(x) for x in np.asarray(F, dtype=np.int64).reshape(-1)]
        self.material_id = int(material_id)


def _destrip_set(strips) -> Counter:
    """Winding-insensitive triangle multiset from de-stripifying strips."""
    out: Counter = Counter()
    for s in strips:
        for (a, b, c) in destripify(s):
            out[frozenset((a, b, c))] += 1
    return out


def _reparse_tri_position_multiset(meshes) -> Counter:
    """Winding-insensitive {pos,pos,pos}-per-triangle multiset (reader shape)."""
    out: Counter = Counter()
    for m in meshes:
        for i in range(0, len(m.indices), 3):
            a, b, c = m.indices[i], m.indices[i + 1], m.indices[i + 2]
            out[frozenset((m.vertices[a].pos, m.vertices[b].pos,
                           m.vertices[c].pos))] += 1
    return out


# ===========================================================================
# (a) de-stripification reproduces the EXACT triangle vertex-set
# ===========================================================================
@pytest.mark.parametrize("n", [3, 4, 8, 16, 24])
def test_grid_strips_destripify_to_exact_triangle_set(n):
    _V, _U, F = _grid(n)
    strips = stripify(F)
    assert _destrip_set(strips) == triangle_set(F)


def test_sphere_strips_destripify_to_exact_triangle_set():
    V, F = _uv_sphere(8, 12)
    strips = stripify(F)
    # Poles introduce degenerate faces; triangle_set/destripify both drop
    # them, so the SETS still match exactly.
    assert _destrip_set(strips) == triangle_set(F)


# ===========================================================================
# (b) every (non-degenerate) triangle is covered EXACTLY once
# ===========================================================================
def test_every_triangle_covered_exactly_once():
    _V, _U, F = _grid(20)
    strips = stripify(F)
    want = triangle_set(F)
    got = _destrip_set(strips)
    # Same multiset == same coverage with the same multiplicities (exactly
    # once for a manifold grid; duplicates preserved if present).
    assert got == want
    assert strip_triangle_count(strips) == sum(want.values())
    # No triangle appears MORE than its source multiplicity.
    for tri, cnt in got.items():
        assert cnt == want[tri]


def test_duplicate_and_orphan_triangles_handled():
    # dup [0,1,2] twice, an orphan [7,8,9], a degenerate [5,5,6] (dropped).
    F = np.array([[0, 1, 2], [0, 1, 2], [2, 3, 4], [5, 5, 6], [7, 8, 9]],
                 dtype=np.int64)
    strips = stripify(F)
    want = triangle_set(F)
    assert want[frozenset((0, 1, 2))] == 2          # duplicate survives twice
    assert frozenset((5, 5, 6)) not in want         # degenerate excluded
    assert _destrip_set(strips) == want


# ===========================================================================
# determinism — same input -> same strips (no RNG / clock)
# ===========================================================================
def test_stripify_is_deterministic():
    _V, _U, F = _grid(18)
    a = stripify(F)
    b = stripify(F)
    assert a == b
    # list-of-lists input gives the same result as the ndarray input.
    c = stripify([list(map(int, row)) for row in F.tolist()])
    assert a == c


def test_max_strip_len_caps_and_preserves_set():
    _V, _U, F = _grid(16)
    strips = stripify(F, max_strip_len=5)
    assert all(len(s) <= 5 for s in strips)
    assert _destrip_set(strips) == triangle_set(F)


# ===========================================================================
# (c) stripified n.rel has FEWER submeshes AND is SMALLER, same triangle set
# ===========================================================================
def test_stripified_nrel_fewer_submeshes_and_smaller():
    V, U, F = _grid(30)              # 1682 tris
    mesh = _Mesh(V, F, U, material_id=0)
    names = ["floor"]

    nodes_base = nrel_nodes_from_meshes([mesh], stripify=False)
    nodes_strip = nrel_nodes_from_meshes([mesh], stripify=True)

    n_sub_base = len(nodes_base[0].submeshes)
    n_sub_strip = len(nodes_strip[0].submeshes)
    assert n_sub_base == F.shape[0]          # one submesh per triangle
    assert n_sub_strip < n_sub_base          # FEWER submeshes

    out_base = build_nrel_from_meshes(nodes_base, names, enforce_budget=False)
    out_strip = build_nrel_from_meshes(nodes_strip, names, enforce_budget=False)
    assert len(out_strip) < len(out_base)    # SMALLER

    # both re-parse to the SAME triangle vertex-position set.
    mb = _rel.extract_nrel_meshes(_rel.parse_rel(out_base))
    ms = _rel.extract_nrel_meshes(_rel.parse_rel(out_strip))
    set_b = _reparse_tri_position_multiset(mb)
    set_s = _reparse_tri_position_multiset(ms)
    assert set_b == set_s
    # and that set equals the source geometry's (positions quantised to
    # f32 — the n.rel stores vertex_format-3 positions as 32-bit floats).
    Vq = V.astype(np.float32)
    pos = {i: (float(Vq[i, 0]), float(Vq[i, 1]), float(Vq[i, 2]))
           for i in range(Vq.shape[0])}
    src = Counter(frozenset((pos[int(a)], pos[int(b)], pos[int(c)]))
                  for a, b, c in F.tolist())
    assert set_s == src

    # closed-form pointer count holds for the stripified topology.
    assert (_rel.parse_rel(out_strip).pointer_count
            == _rw.nrel_pointer_count(nodes_strip, names))

    # relocation is clean.
    base = 0x40000000
    rel_s = _rel.parse_rel(out_strip)
    for v in _rw.simulate_rel_relocation(out_strip, base=base):
        assert v == base or base <= v < base + rel_s.pointer_table_offset


def test_stripified_shares_one_vertex_buffer_per_mesh():
    """Stripification reuses ONE vertex array across all of a mesh's strips,
    so the on-disk vertex bytes are written ONCE (the density mechanism)."""
    V, U, F = _grid(20)
    mesh = _Mesh(V, F, U)
    sms = nrel_submeshes_stripified([mesh])
    assert len(sms) > 1
    # every strip submesh references the SAME vertex-list object.
    first = sms[0].vertices
    assert all(sm.vertices is first for sm in sms)
    assert len(first) == V.shape[0]          # the mesh's FULL vertex array


# ===========================================================================
# (d) budget density win: a mesh that does NOT fit at 1-tri DOES fit stripped
# ===========================================================================
def test_budget_density_win():
    # Sized so the one-triangle-per-strip author overflows 768 KB but the
    # stripified author fits comfortably.
    V, U, F = _grid(60)              # 6962 tris
    mesh = _Mesh(V, F, U)
    names = ["floor"]

    nodes_base = nrel_nodes_from_meshes([mesh], stripify=False)
    out_base = build_nrel_from_meshes(nodes_base, names, enforce_budget=False)
    assert len(out_base) > NREL_SIZE_BUDGET          # baseline DOES NOT fit

    nodes_strip = nrel_nodes_from_meshes([mesh], stripify=True)
    out_strip = build_nrel_from_meshes(nodes_strip, names, enforce_budget=True)
    assert len(out_strip) <= NREL_SIZE_BUDGET        # stripified DOES fit

    # same geometry survives the round trip (winding-insensitive).
    ms = _rel.extract_nrel_meshes(_rel.parse_rel(out_strip))
    assert sum(len(m.indices) // 3 for m in ms) == F.shape[0]


def test_lobby_pipeline_authors_more_tris_under_budget():
    """The lobby build authors stripified geometry by default, so a fixed
    mesh fits at a HIGHER triangle count than the 1-tri path would allow."""
    from formats.lobby_pipeline import author_nrel_uv

    V, U, F = _grid(60)              # 6962 tris
    # 1-tri path overflows at full resolution...
    with pytest.raises(_rw.RelWriteError, match="budget"):
        author_nrel_uv(V, U, F, "lobby", enforce=True, stripify=False)
    # ...the default (stripified) path fits the SAME full-resolution mesh.
    buf = author_nrel_uv(V, U, F, "lobby", enforce=True, stripify=True)
    assert len(buf) <= NREL_SIZE_BUDGET
    ms = _rel.extract_nrel_meshes(_rel.parse_rel(buf))
    assert sum(len(m.indices) // 3 for m in ms) == F.shape[0]
