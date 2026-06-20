#!/usr/bin/env python3
"""Deterministic greedy triangle-strip generator (NvTriStrip / SA3D style).

Turns an ``(M, 3)`` triangle-index array into a list of triangle STRIPS,
each a flat vertex-index sequence over a SHARED vertex array.  The n.rel
authoring path (``formats.rel_writer.nrel_submeshes_stripified``) emits one
submesh per strip, collapsing the per-triangle overhead that the
one-triangle-per-strip baseline pays (a VertexInfo + Strip + Material row
and 3 duplicated vertices PER triangle).

Why a custom stripifier (no external dep)
-----------------------------------------
The reference ``SA3D.Modeling`` ``Strippify`` is a C# greedy adjacency walk;
we reproduce its essentials in pure NumPy/stdlib:

  * Coalesce triangles that share an edge into one strip.
  * When the current strip cannot be extended, restart a new strip from the
    lowest-index unused triangle.
  * Fully DETERMINISTIC — no ``random`` / no clock.  Every tie (which
    neighbour to walk into, which triangle to seed from) breaks toward the
    LOWEST triangle index, so the same input always yields the same strips.

Correctness contract (THE gate)
-------------------------------
The engine recomputes winding from vertex normals, and the reader's
``formats.rel._strip_to_triangles`` de-stripifies with ALTERNATING parity
(even position ``i`` -> ``(a,b,c)``, odd -> ``(a,c,b)``).  So a strip we emit
need only reproduce the same triangle *vertex set* once de-stripified — the
per-triangle winding may flip.  :func:`destripify` here mirrors
``_strip_to_triangles`` exactly so tests can assert the round-trip triangle
SET is identical.

Public API
----------
``stripify(faces, *, max_strip_len=None) -> list[list[int]]``
    Greedy strips.  ``faces`` is an ``(M, 3)`` array / sequence of triples.
``destripify(strip) -> list[tuple[int, int, int]]``
    Mirror of the reader's parity de-stripification (for tests / self-check).
``triangle_set(faces) -> collections.Counter``
    Winding-insensitive multiset of triangle vertex-index frozensets.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["stripify", "destripify", "triangle_set", "strip_triangle_count"]


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------
def _as_faces(faces) -> np.ndarray:
    """Coerce ``faces`` to a contiguous ``(M, 3)`` int64 array."""
    arr = np.asarray(faces, dtype=np.int64)
    if arr.size == 0:
        return arr.reshape(0, 3)
    arr = arr.reshape(-1, 3)
    return arr


def _is_degenerate(t: Tuple[int, int, int]) -> bool:
    a, b, c = t
    return a == b or b == c or a == c


# ---------------------------------------------------------------------------
# de-stripification (mirror of formats.rel._strip_to_triangles)
# ---------------------------------------------------------------------------
def destripify(strip: Sequence[int]) -> List[Tuple[int, int, int]]:
    """Replicate ``formats.rel._strip_to_triangles`` parity de-stripification.

    Even strip position ``i`` -> ``(a, b, c)``; odd -> ``(a, c, b)``.
    Degenerate triangles (two equal indices) are dropped, matching the
    reader.  Returns a list of (a, b, c) index triples.
    """
    out: List[Tuple[int, int, int]] = []
    n = len(strip)
    if n < 3:
        return out
    for i in range(n - 2):
        a, b, c = strip[i], strip[i + 1], strip[i + 2]
        if a == b or b == c or a == c:
            continue
        if i % 2 == 0:
            out.append((a, b, c))
        else:
            out.append((a, c, b))
    return out


def triangle_set(faces) -> Counter:
    """Winding-insensitive multiset of triangle vertex-index frozensets.

    Degenerate triangles (a face with a repeated index) are dropped — they
    carry no area and the reader drops them too, so they can never survive a
    round trip and must not count toward coverage.
    """
    arr = _as_faces(faces)
    c: Counter = Counter()
    for row in arr.tolist():
        t = (int(row[0]), int(row[1]), int(row[2]))
        if _is_degenerate(t):
            continue
        c[frozenset(t)] += 1
    return c


def strip_triangle_count(strips: Sequence[Sequence[int]]) -> int:
    """Total non-degenerate triangles produced by de-stripifying ``strips``."""
    return sum(len(destripify(s)) for s in strips)


# ---------------------------------------------------------------------------
# adjacency
# ---------------------------------------------------------------------------
def _build_adjacency(
    tris: List[Tuple[int, int, int]],
) -> Dict[Tuple[int, int], List[int]]:
    """Map an undirected edge (lo, hi) -> sorted list of incident triangle ids.

    Only NON-degenerate triangles are indexed.  The incident lists are kept
    sorted ascending so neighbour selection is deterministic (lowest id first).
    """
    edge_tris: Dict[Tuple[int, int], List[int]] = {}
    for ti, (a, b, c) in enumerate(tris):
        if _is_degenerate((a, b, c)):
            continue
        for (u, v) in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_tris.setdefault(key, []).append(ti)
    # lists already appended in ascending ti order
    return edge_tris


def _third_vertex(t: Tuple[int, int, int], e0: int, e1: int) -> Optional[int]:
    """Return the vertex of triangle ``t`` not in edge {e0, e1} (or None)."""
    edge = {e0, e1}
    for v in t:
        if v not in edge:
            return v
    return None


# ---------------------------------------------------------------------------
# greedy strip walk
# ---------------------------------------------------------------------------
def stripify(
    faces,
    *,
    max_strip_len: Optional[int] = None,
) -> List[List[int]]:
    """Greedy adjacency-walk triangle stripifier.

    Parameters
    ----------
    faces :
        An ``(M, 3)`` array / sequence of triangle vertex-index triples.
    max_strip_len :
        Optional cap on the vertex count of a single strip (``None`` =
        unbounded).  When a strip would exceed this it is closed and a new
        one is started; useful to keep individual index buffers small.

    Returns
    -------
    list[list[int]]
        One vertex-index sequence per strip.  De-stripifying every strip
        (via :func:`destripify`, == the reader's parity rule) reproduces the
        EXACT input triangle set (degenerate input faces excluded).  Output
        is fully deterministic for a given input.

    Notes
    -----
    The walk:
      1. Seed from the lowest-index unused, non-degenerate triangle as
         ``[a, b, c]``.
      2. Repeatedly extend through the last edge ``(prev, last)``: find the
         lowest-index unused triangle sharing that edge, append its third
         vertex.  ``_strip_to_triangles`` parity guarantees the appended
         triangle's vertex set matches.
      3. When no neighbour is available (or the cap is hit), close the strip
         and seed a new one.
    Degenerate, duplicate, and orphan (no shared-edge neighbour) triangles
    are each handled — duplicates become their own short strips, orphans
    become 3-index strips.
    """
    arr = _as_faces(faces)
    tris: List[Tuple[int, int, int]] = [
        (int(r[0]), int(r[1]), int(r[2])) for r in arr.tolist()
    ]
    n = len(tris)
    used = [False] * n

    # Degenerate triangles are never emitted (the reader drops them).
    for ti, t in enumerate(tris):
        if _is_degenerate(t):
            used[ti] = True

    edge_tris = _build_adjacency(tris)

    def _pick_neighbour(e0: int, e1: int, forbid: int) -> int:
        """Lowest-index unused triangle sharing edge {e0,e1} (excluding forbid)."""
        key = (e0, e1) if e0 < e1 else (e1, e0)
        best = -1
        for cand in edge_tris.get(key, ()):  # ascending order
            if cand == forbid or used[cand]:
                continue
            best = cand
            break
        return best

    strips: List[List[int]] = []

    # Seed triangles strictly in ascending index order for determinism.
    for seed in range(n):
        if used[seed]:
            continue
        used[seed] = True
        a, b, c = tris[seed]
        strip: List[int] = [a, b, c]
        # Last edge of the strip is (strip[-2], strip[-1]).
        prev, last = b, c
        cur = seed

        while max_strip_len is None or len(strip) < max_strip_len:
            nxt = _pick_neighbour(prev, last, cur)
            if nxt < 0:
                break
            used[nxt] = True
            third = _third_vertex(tris[nxt], prev, last)
            if third is None:
                # Shared edge collapsed (shouldn't happen for valid tris);
                # stop extending rather than emit a bad index.
                break
            strip.append(third)
            prev, last = last, third
            cur = nxt

        strips.append(strip)

    return strips
