"""Regression test for the 2026-04-25 BML alignment fix.

The 23 ``pl[A-Z]nj.bml`` archives lie in their header — has_textures=1
but every entry has tex_size_compressed==0 — and use 0x800 alignment.
NpcApcMot.bml has the same superficial flag but uses 0x20 alignment.
The cumulative-end heuristic in ``parse_bml`` discriminates by
computing the predicted last-byte offset under each candidate
alignment and picking the one that exactly matches the on-disk size.

Live data: ``~/PSOBB.IO/data/`` — the test is skipped if
that directory isn't present.
"""
from __future__ import annotations
import os

from pathlib import Path

import pytest

from formats.bml import parse_bml, _prs_decompress

_DATA = Path(os.path.expanduser("~/PSOBB.IO/data"))
_LIVE = _DATA.exists()


@pytest.mark.skipif(not _LIVE, reason="PSOBB.IO not installed")
def test_player_nj_archives_decompress():
    """Every entry in pl[A-Z]nj.bml decompresses to NJCM-prefixed bytes."""
    pl_bmls = sorted(_DATA.glob("pl[A-Z]nj.bml"))
    assert len(pl_bmls) >= 20, f"expected ~23 player BMLs, got {len(pl_bmls)}"
    total = 0
    for p in pl_bmls:
        buf = p.read_bytes()
        entries = parse_bml(buf)
        for ent in entries:
            raw = bytes(buf[ent.offset:ent.offset + ent.size_compressed])
            decomp = _prs_decompress(raw)
            magic_ok = (
                decomp[:4] == b"NJCM"
                or (len(decomp) >= 5 and decomp[0] == 0xFF and decomp[1:5] == b"NJCM")
            )
            assert magic_ok, (
                f"{p.name}#{ent.name} decompressed without NJCM magic: "
                f"first 8 bytes = {decomp[:8].hex()}"
            )
            total += 1
    # Across all 23 player BMLs we expect 209 inner entries.
    assert total >= 200, f"expected ~209 entries, got {total}"


@pytest.mark.skipif(not _LIVE, reason="PSOBB.IO not installed")
def test_npc_apc_mot_alignment():
    """NpcApcMot.bml shares the has_textures=1 flag but uses 0x20 alignment."""
    p = _DATA / "NpcApcMot.bml"
    if not p.exists():
        pytest.skip("NpcApcMot.bml not present")
    buf = p.read_bytes()
    entries = parse_bml(buf)
    assert len(entries) >= 100, f"expected 120 NPC motions, got {len(entries)}"
    # Spot-check a handful decompress to NMDM (motion magic).
    for ent in entries[:5]:
        raw = bytes(buf[ent.offset:ent.offset + ent.size_compressed])
        decomp = _prs_decompress(raw)
        assert decomp[:4] == b"NMDM", (
            f"NpcApcMot.bml#{ent.name}: expected NMDM magic, got {decomp[:4].hex()}"
        )


@pytest.mark.skipif(not _LIVE, reason="PSOBB.IO not installed")
def test_textured_bml_still_uses_0x20():
    """A normal textured BML (e.g. bm4_ps_ma_body) MUST still pick 0x20."""
    p = _DATA / "bm4_ps_ma_body.bml"
    if not p.exists():
        pytest.skip("bm4_ps_ma_body.bml not present")
    buf = p.read_bytes()
    entries = parse_bml(buf)
    assert len(entries) >= 5
    # At least one entry should have a paired texture (otherwise the
    # alignment regression is moot).
    has_tex = sum(1 for e in entries if e.tex_size_compressed > 0)
    assert has_tex >= 1, f"expected ≥1 entry with paired XVM, got {has_tex}"
    # Every entry must decompress to a recognisable PSO-data prefix.
    for ent in entries:
        raw = bytes(buf[ent.offset:ent.offset + ent.size_compressed])
        decomp = _prs_decompress(raw)
        head = decomp[:4]
        assert head in (b"NJCM", b"NMDM", b"NJTL", b"NSSM") or head[0] == 0xFF, (
            f"{p.name}#{ent.name} unrecognised magic: {head.hex()}"
        )
