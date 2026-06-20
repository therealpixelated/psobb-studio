"""Material-id propagation tests for chunked-NJ models.

The dragon (`bm_boss1_dragon.bml#boss1_s_nb_dragon.nj`) renders 734
sub-meshes with 16 distinct named textures (foot, kuti, tail, eye,
horn, body, jaw, etc.). The chunk-walker in ``formats/xj.py``
propagates ``state.texture_id`` via Tiny chunks (types 8/9), and each
emitted sub-mesh inherits the most-recently-seen texture id as its
``material_id``.

A regression in the chunk-walker (most likely: forgetting to thread
``state`` through ``_walk_tree`` so each node restarts at -1, or
mis-handling the type-4/5 cache-replay path so cached strips lose
their texture context) would manifest as a heavily-skewed material
distribution where 90%+ of the sub-meshes inherit a single material
(usually 0 = the first foot, since that's the first Tiny chunk in
the dragon's polygon stream).

These tests guard against that regression in two layers:

  1. *Synthetic:* a hand-built NjModel with five sub-meshes
     interleaved with five Tiny chunks (texture ids 0..4) round-trips
     through ``encode_nj_model`` + ``parse_xj_njcm`` and produces five
     sub-meshes whose ``material_id`` matches the surrounding Tiny
     chunk.

  2. *Real-data sanity:* if the dragon BML is present in the test
     data dir, parse it and assert the material distribution is
     well-spread — no single ``material_id`` may account for more than
     50% of sub-meshes. (For dragon: max share is ~15%.)

The real-data test is skipped (rather than failed) when the BML
isn't available — the editor's data dir is not part of the repo.
"""
from __future__ import annotations

import math
import os
import struct
from collections import Counter
from pathlib import Path

import pytest

from formats.nj_writer import (
    NjChunk,
    NjMeshChunks,
    NjModel,
    NjNode,
    encode_nj_model,
)
from formats.xj import parse_nj_file


# ---------------------------------------------------------------------------
# Synthetic chunk-stream helpers.
# ---------------------------------------------------------------------------


def _vertex_chunk_3pts(start_index: int) -> NjChunk:
    """Build a vertex chunk (type 34 = NJD_CV — three position-only
    vertices starting at ``start_index``).

    Per ``formats/xj.py::_chunk_body_size``, types 32..50 use::

        body_size = 2 + 4 * body_words      # body_words is in DWORDS

    and the body layout that ``_parse_vertex_chunk`` consumes (after
    the size word) is::

        u16 base_index
        u16 vertex_count
        per-vertex: f32 x, f32 y, f32 z   (type 34 has no extras)

    For 3 vertices the per-vertex content is 36 bytes; with the
    4-byte (u16+u16) base/count header we write 40 bytes of
    "post-size-word" payload, i.e. 10 dwords.
    """
    nverts = 3
    pos_payload = b""
    for i in range(nverts):
        x = float(start_index + i)
        y = float(start_index)
        z = 0.0
        pos_payload += struct.pack("<fff", x, y, z)
    # Post-size-word body: u16 base_index, u16 nverts, then per-vertex.
    after_size = struct.pack("<HH", start_index, nverts) + pos_payload
    # _chunk_body_size returns 2 + 4 * body_words, so:
    # full_body = 2-byte size word + 4 * body_words bytes
    #   => body_words = (len(after_size)) / 4
    assert len(after_size) % 4 == 0, "type-34 payload must be DWORD-aligned"
    body_words = len(after_size) // 4
    body = struct.pack("<H", body_words) + after_size
    return NjChunk(type_id=34, flags=0, body=bytes(body))


def _tiny_chunk(texture_id: int) -> NjChunk:
    """Build a Tiny chunk (type 8) that sets state.texture_id = texture_id."""
    word = texture_id & 0x1FFF
    return NjChunk(type_id=8, flags=0x34, body=struct.pack("<H", word))


def _strip_chunk_one_tri(start_index: int) -> NjChunk:
    """Build a strip chunk (type 64 = NJD_CP_STRIP, "index only") that
    emits one triangle.

    Per ``formats/xj.py::_parse_strip_chunk``:
      * After the size word, the next u16 packs ``user_offset_size``
        (top 2 bits, multiplied by 2 bytes) and ``strip_count``
        (low 14 bits).
      * Per strip: a SIGNED u16 — negative => clockwise winding,
        absolute value => index_count.
      * Per vertex: u16 index (type 64 has no UV / normal / color
        extras).

    Body size formula for type 64-75 is ``2 + 2 * body_words`` (i.e.
    body_words is in WORDS), so the size word counts u16's of post-
    size-word content.
    """
    strip_count = 1
    nverts = 3
    # Encode the strip header u16 (user_offset=0, strip_count=1).
    main_header = (0 << 14) | (strip_count & 0x3FFF)
    # Per-strip header: -nverts means clockwise (signed -3 → 0xFFFD).
    per_strip_header = struct.pack("<h", -nverts)
    # Per-vertex indices.
    indices = struct.pack(
        "<HHH",
        start_index + 0,
        start_index + 1,
        start_index + 2,
    )
    after_size = struct.pack("<H", main_header) + per_strip_header + indices
    assert len(after_size) % 2 == 0
    body_words = len(after_size) // 2
    body = struct.pack("<H", body_words) + after_size
    return NjChunk(type_id=64, flags=0, body=bytes(body))


def _build_synthetic_model_with_n_materials(n: int) -> bytes:
    """Build an NjModel with ``n`` sub-meshes, each preceded by a Tiny
    chunk that sets a different texture id (0..n-1).

    The model has a single root node carrying one mesh that interleaves
    (vertex chunk, tiny chunk, strip chunk) blocks ``n`` times.

    Returns the encoded ``.nj`` bytes ready for ``parse_nj_file``.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    # Build the vlist: one vertex chunk per strip (with start_index
    # spaced so strips don't share vertex slots).
    vlist: list[NjChunk] = []
    for i in range(n):
        vlist.append(_vertex_chunk_3pts(start_index=i * 3))

    # Build the plist: alternating Tiny + strip chunks.
    plist: list[NjChunk] = []
    for i in range(n):
        plist.append(_tiny_chunk(texture_id=i))
        plist.append(_strip_chunk_one_tri(start_index=i * 3))

    mesh = NjMeshChunks(
        bbox=(0.0, 0.0, 0.0, 10.0),
        vlist=vlist,
        plist=plist,
    )

    # Single root node pointing at the mesh.
    root = NjNode(
        eval_flags=0,
        position=(0.0, 0.0, 0.0),
        rotation_bams=(0, 0, 0),
        scale=(1.0, 1.0, 1.0),
        mesh_index=0,
        child_index=-1,
        sibling_index=-1,
    )

    model = NjModel(
        njtl_names=[],
        nodes=[root],
        meshes=[mesh],
    )
    return encode_nj_model(model)


# ---------------------------------------------------------------------------
# Synthetic test: 5 sub-meshes with distinct material ids 0..4.
# ---------------------------------------------------------------------------


def test_synthetic_chunked_nj_propagates_distinct_material_ids():
    """Five sub-meshes, each with its own Tiny chunk, must surface five
    distinct material_ids in DFS order."""
    nj_bytes = _build_synthetic_model_with_n_materials(5)
    meshes = parse_nj_file(nj_bytes)

    # Five strips → five sub-meshes (ignoring any synthesized parser
    # artifacts; if more than 5 something else is wrong).
    assert len(meshes) == 5, (
        f"expected 5 sub-meshes from 5 (Tiny + Strip) blocks, got {len(meshes)}"
    )

    # Each sub-mesh must carry a different material_id, and the order
    # MUST follow Tiny-chunk-by-Tiny-chunk (0, 1, 2, 3, 4).
    mids = [m.material_id for m in meshes]
    assert mids == [0, 1, 2, 3, 4], (
        f"material_id propagation broken: expected [0,1,2,3,4], got {mids}"
    )


def test_synthetic_chunked_nj_all_materials_distinct():
    """Distribution sanity: with N independent Tiny+Strip pairs, no
    material_id may dominate."""
    n = 8
    nj_bytes = _build_synthetic_model_with_n_materials(n)
    meshes = parse_nj_file(nj_bytes)
    assert len(meshes) == n
    counts = Counter(m.material_id for m in meshes)
    # Every distinct material is referenced exactly once.
    assert len(counts) == n
    assert all(v == 1 for v in counts.values())


# ---------------------------------------------------------------------------
# Real-data sanity: dragon BML if available.
# ---------------------------------------------------------------------------


def _resolve_data_dir() -> Path:
    """Mirror ``server.py``'s DATA_DIR resolution but without importing
    ``server`` (which would spin up the FastAPI app + cache subsystems)."""
    return Path(
        os.environ.get("PSO_DATA_DIR")
        or r"C:/tmp_pso_dev/data"
    ).resolve()


def _extract_dragon_inner_nj() -> bytes | None:
    """Return the raw inner .nj bytes for boss1_s_nb_dragon, or None
    if the BML isn't present in this test environment."""
    data_dir = _resolve_data_dir()
    bml_path = data_dir / "bm_boss1_dragon.bml"
    if not bml_path.exists():
        return None
    try:
        from formats.bml import extract_bml
    except ImportError:
        return None
    try:
        entries = extract_bml(bml_path.read_bytes())
    except Exception:
        return None
    blob = entries.get("boss1_s_nb_dragon.nj")
    if blob is None or len(blob) == 0:
        return None
    return blob


@pytest.mark.skipif(
    _extract_dragon_inner_nj() is None,
    reason="dragon BML not available in this test data dir",
)
def test_dragon_material_distribution_not_skewed():
    """Real-data sanity: the dragon's 734 sub-meshes must span >5
    distinct material_ids and no single material_id may account for
    more than 50% of sub-meshes.

    Empirical baseline (post-fix verification, 2026-04-26):
      - 16 distinct material_ids
      - top material: id 10 (douyoko1, body) at 15.5% (114/734)
      - tail of distribution: id 2/15 at 0.5% each (4/734)

    A regression that resets state.texture_id between mesh-tree nodes
    or that fails to propagate Tiny chunks through type-4/5 cache
    replay would push the top material's share over 50% (the dragon's
    polygon stream would then default to material 0 = foot1 = the
    very first Tiny chunk in DFS order)."""
    nj_bytes = _extract_dragon_inner_nj()
    assert nj_bytes is not None  # gated by skipif

    meshes = parse_nj_file(nj_bytes)
    assert len(meshes) > 100, (
        f"dragon should have 700+ sub-meshes, got {len(meshes)}"
    )

    counts = Counter(m.material_id for m in meshes)
    distinct = len(counts)
    total = len(meshes)
    top_mid, top_count = counts.most_common(1)[0]
    top_share = top_count / total

    assert distinct >= 5, (
        f"dragon should span >=5 distinct material_ids, got {distinct} "
        f"(distribution: {dict(counts)})"
    )
    assert top_share <= 0.50, (
        f"single material_id {top_mid} dominates {top_share*100:.1f}% "
        f"of sub-meshes ({top_count}/{total}); chunk-walker is not "
        f"propagating texture_id correctly. Distribution: {dict(counts)}"
    )


@pytest.mark.skipif(
    _extract_dragon_inner_nj() is None,
    reason="dragon BML not available in this test data dir",
)
def test_dragon_eye_foot_tail_submeshes_present():
    """Spot-check the named extremities: dragon must emit sub-meshes
    bound to the eye, foot, and tail texture slots.

    The texture-id → name mapping for the dragon (per its NJTL):
      slot  0 = s_064_foot1
      slot  3 = s_064_kuti  (mouth)
      slot  5 = s_128_tail
      slot 15 = s_32_eye
      slot 17 = s_64_douue  (top of body)

    If the chunk-walker silently drops Tiny chunks for slots 15+
    (e.g. truncates the texture_id mask too aggressively) the eye
    sub-meshes would inherit some other material and slot 15 would be
    absent from ``set(m.material_id for m in meshes)``."""
    nj_bytes = _extract_dragon_inner_nj()
    assert nj_bytes is not None
    meshes = parse_nj_file(nj_bytes)
    refd = {m.material_id for m in meshes}

    # Foot, mouth, tail, eye, top-of-body: every named extremity must
    # surface at least one sub-mesh.
    for required, label in [
        (0, "foot1"),
        (3, "kuti/mouth"),
        (5, "tail"),
        (15, "eye"),
        (17, "douue/top-of-body"),
    ]:
        assert required in refd, (
            f"dragon has no sub-mesh bound to material_id {required} "
            f"({label}); referenced ids: {sorted(refd)}"
        )
