"""Oracle-parity tests for the BML reader/packer (DOMAIN bml audit, 2026-06-19).

These tests pin the two behaviours ported from the authoritative C# oracle
``_reference/PSOBMLExtract/PSOBMLExtract/BMLUtil.cs``:

  1. **SeekPadding offset resolver** — ``parse_bml`` resolves each inner and
     texture payload offset with the oracle's per-gap zero-skip scan
     (BMLUtil.cs:362-377), NOT a fixed global file-alignment guess. This test
     recomputes every offset with an independent in-test SeekPadding reimpl and
     asserts byte-for-byte agreement with the reader.

  2. **Header magic model** — the 4 bytes at +0x08 are a single u32 magic
     (== 0x150, BMLUtil.cs:20), not a (compression u8, has_textures u8) pair.

It also pins the byte-exact RAW round-trip (``parse_bml_for_pack`` ->
``pack_bml``), which the audit found to be strictly more correct than the C#
packer (the C# ``PackBML`` hardcodes Align(0x20) and so cannot reproduce the
23 ``pl[A-Z]nj.bml`` 0x800-aligned archives).

Live data: ``~/EphineaPSO/data`` and ``~/PSOBB.IO/data``.
The whole module is skipped when neither corpus is installed (CI without the
game assets). A tiny synthetic-archive test always runs so the resolver has at
least one unconditional check.
"""
from __future__ import annotations
import os

import struct
from pathlib import Path

import pytest

from formats import bml as B

_ROOTS = [
    Path(os.path.expanduser("~/EphineaPSO/data")),
    Path(os.path.expanduser("~/PSOBB.IO/data")),
]
_LIVE_ROOTS = [r for r in _ROOTS if r.exists()]
_HAVE_CORPUS = bool(_LIVE_ROOTS)


def _all_bmls():
    out = []
    for root in _LIVE_ROOTS:
        out.extend(sorted(root.glob("*.bml")))
    return out


def _oracle_offsets(buf: bytes):
    """Independent faithful port of BMLUtil.cs ExtractBML offset walk +
    SeekPadding. Returns ``[(name, inner_off, tex_off_or_None), ...]``.

    Deliberately does NOT call into ``formats.bml`` internals so it is a
    true second implementation, not a tautology.
    """
    mv = memoryview(buf)
    n = len(mv)
    file_count = struct.unpack_from("<I", mv, 4)[0]
    assert file_count <= 0xFFFF, "big-endian GC archive not expected in PC corpus"

    entries = []
    for i in range(file_count):
        ent = 0x40 + i * 0x40
        name = bytes(mv[ent:ent + 0x20]).split(b"\x00", 1)[0].decode("ascii", "replace")
        cs, _u, _d, tcs, _t, _b, _c, _e = struct.unpack_from("<8I", mv, ent + 0x20)
        entries.append((name, cs, tcs))

    # First payload starts at table-end rounded up to the next 0x800 boundary
    # (BMLUtil.cs:89-96).
    pos = 0x40 + file_count * 0x40
    offset = (pos + 0x800) & 0xFFFFF800 if (pos & 0x7FF) > 0 else pos

    def seek_padding(o: int) -> int:
        # BMLUtil.cs:362-377
        while o % 0x10 > 0:
            o += 1
        while o + 4 <= n and struct.unpack_from("<i", mv, o)[0] == 0:
            o += 0x10
        return o

    res = []
    for name, cs, tcs in entries:
        inner_off = offset
        offset = seek_padding(offset + cs)
        tex_off = None
        if tcs > 0:
            tex_off = offset
            offset = seek_padding(offset + tcs)
        res.append((name, inner_off, tex_off))
    return res


# ---------------------------------------------------------------------------
# Unconditional synthetic check — exercises the resolver without the corpus.
# ---------------------------------------------------------------------------
def test_seek_padding_resolver_synthetic():
    """A hand-built 2-entry, 0x20-aligned archive parses to the expected
    offsets via the SeekPadding resolver."""
    # Two pre-compressed (here just opaque) inner blobs, no textures.
    inner0 = b"AAAA" * 5          # 20 bytes -> padded to 0x20
    inner1 = b"BBBBBBBB"          # 8 bytes
    entries = [
        B.BmlPackEntry(name="a.nj", data=inner0, decompressed_size=999,
                       is_compressed=True),
        B.BmlPackEntry(name="b.nj", data=inner1, decompressed_size=999,
                       is_compressed=True),
    ]
    packed = B.pack_bml(entries, file_alignment=B.DATA_ALIGNMENT_HAS_TEX,
                        has_textures_override=False)
    # Magic must be the 0x150 constant.
    assert struct.unpack_from("<I", packed, 8)[0] == B.BML_MAGIC

    parsed = B.parse_bml(packed)
    oracle = {n: (io, to) for (n, io, to) in _oracle_offsets(packed)}
    assert len(parsed) == 2
    for e in parsed:
        io, _to = oracle[e.name]
        assert e.offset == io, f"{e.name}: reader 0x{e.offset:x} != oracle 0x{io:x}"
    # First payload sits at the 0x800 table boundary.
    assert parsed[0].offset == 0x800
    # Second payload is the first padded up to 0x20 (20 -> 0x20).
    assert parsed[1].offset == 0x800 + B._align_up(len(inner0), 0x20)


# ---------------------------------------------------------------------------
# Corpus-wide parity (skipped when assets absent).
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _HAVE_CORPUS, reason="no BML corpus installed")
def test_magic_is_0x150_everywhere():
    bmls = _all_bmls()
    assert len(bmls) >= 100, f"expected hundreds of BMLs, got {len(bmls)}"
    bad = []
    for p in bmls:
        buf = p.read_bytes()
        magic = struct.unpack_from("<I", buf, 8)[0]
        if magic != B.BML_MAGIC:
            bad.append((p.name, hex(magic)))
    assert not bad, f"{len(bad)} BMLs have a non-0x150 magic: {bad[:10]}"


@pytest.mark.skipif(not _HAVE_CORPUS, reason="no BML corpus installed")
def test_offsets_match_seekpadding_oracle():
    """Reader inner+texture offsets == independent SeekPadding reimpl."""
    bmls = _all_bmls()
    inner_mismatch = []
    tex_mismatch = []
    for p in bmls:
        buf = p.read_bytes()
        parsed = B.parse_bml(buf)
        oracle = {n: (io, to) for (n, io, to) in _oracle_offsets(buf)}
        # Re-derive the reader's texture offsets via the same public path
        # extract_bml_texture uses (the SeekPadding walk over parsed metas).
        metas = [
            (e.name, e.size_compressed, e.size_decompressed, e.tex_size_compressed)
            for e in parsed
        ]
        walk = B._walk_offsets(memoryview(buf), metas)
        for e, (inner_off, tex_off) in zip(parsed, walk):
            assert e.offset == inner_off  # internal consistency
            o_inner, o_tex = oracle[e.name]
            if e.offset != o_inner:
                inner_mismatch.append((p.name, e.name, hex(e.offset), hex(o_inner)))
            if e.tex_size_compressed > 0 and tex_off != o_tex:
                tex_mismatch.append((p.name, e.name, hex(tex_off or 0), hex(o_tex or 0)))
    assert not inner_mismatch, f"inner offset mismatches: {inner_mismatch[:10]}"
    assert not tex_mismatch, f"texture offset mismatches: {tex_mismatch[:10]}"


@pytest.mark.skipif(not _HAVE_CORPUS, reason="no BML corpus installed")
def test_raw_roundtrip_byte_exact():
    """parse_bml_for_pack -> pack_bml reproduces every shipped BML byte-exact.

    Uses the raw (is_compressed=True) passthrough path, which preserves the
    original PRS bytes — this is the bit-exact path. (The CLI unpack->pack
    path re-encodes PRS and is intentionally NOT bit-exact; see the module
    docstring.)
    """
    bmls = _all_bmls()
    failures = []
    for p in bmls:
        buf = p.read_bytes()
        meta = B.parse_bml_pack_meta(buf)
        pack_entries = B.parse_bml_for_pack(buf)
        rebuilt = B.pack_bml(
            pack_entries,
            compression=meta["compression"],
            file_alignment=meta["file_alignment"],
            has_textures_override=meta["has_textures"],
        )
        if rebuilt != buf:
            first = next(
                (i for i in range(min(len(rebuilt), len(buf))) if rebuilt[i] != buf[i]),
                None,
            )
            failures.append((p.name, len(buf), len(rebuilt),
                             hex(first) if first is not None else "len-only"))
    assert not failures, f"{len(failures)} non-byte-exact round-trips: {failures[:10]}"
