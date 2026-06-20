"""Tests for ``formats.sibling_archives`` — magic-sniffed sibling discovery.

Covers:
  * SiblingArchive.from_path on each magic (PVM, GVM, XVM, PVR, GVR, AFS)
  * discover_sibling_textures pairs a model with stem-matching texture
    files in the same directory
  * extract_tile() returns valid PVR/GVR/XVR bytes
  * round-trip: extract_tile() -> pvr_decode.decode_pvr succeeds for PVR
  * PRS-wrapped siblings are auto-decompressed
  * Real-world XVM file from the editor's data dir parses correctly

PSOBB.IO ships with XVM siblings exclusively, but the discovery helper
must also support the other texture-archive magics so future Dreamcast
or GC ports drop in without code changes. We synthesise minimal PVM /
GVM / AFS fixtures via make_test_pvr + tiny hand-rolled headers; the
real-world case uses the live ``ItemTextureEp4.afs`` cache contents.
"""
from __future__ import annotations

import struct
from pathlib import Path

import pytest

from formats.pvr_decode import decode_pvr, make_test_pvr
from formats import sibling_archives as sib
from formats import prs as prs_mod


def _make_pvr_bytes() -> bytes:
    pixels = bytes([0x80, 0x40, 0x20, 0xFF]) * (4 * 4)
    return make_test_pvr(4, 4, pixels, px_format=7, tex_format=14)


def _make_pvm_bytes(records: list[bytes]) -> bytes:
    """Build a PVMH archive containing the given PVR records.

    header_size = 4 (flags + count, no optional tables).
    """
    out = bytearray()
    out += b"PVMH"
    out += struct.pack("<I", 4)
    out += struct.pack("<H", 0)        # flags
    out += struct.pack("<H", len(records))
    for r in records:
        out += r
    return bytes(out)


def _make_gvr_bytes() -> bytes:
    """Synthesise a minimal GVRT chunk (we don't decode GVR yet — the
    test only verifies that the magic sniffer + record walker handle
    the format)."""
    body = b"\x00" * 0x40 + b"\xff" * 16  # minimal payload
    chunk = b"GVRT" + struct.pack(">I", len(body) + 8) + b"\x00" * 8 + body
    # Above is hand-rolled; +8 is the customary "size of rest" the
    # parser reads as `body_size`.
    return chunk


def _make_gvm_bytes(records: list[bytes]) -> bytes:
    out = bytearray()
    out += b"GVMH"
    out += struct.pack(">I", 4)        # big-endian header size
    out += struct.pack(">H", 0)        # flags
    out += struct.pack(">H", len(records))
    for r in records:
        out += r
    return bytes(out)


def _make_xvm_bytes(records: list[bytes]) -> bytes:
    """Build a minimal XVMH archive from given XVRT records."""
    out = bytearray()
    out += b"XVMH"
    out += struct.pack("<I", 0x38)
    out += struct.pack("<I", len(records))
    out += b"\x00" * (0x40 - 0x0C)
    for r in records:
        out += r
    return bytes(out)


def _make_xvrt_bytes(width: int, height: int) -> bytes:
    """Hand-roll a minimal XVRT record. fmt=6 (DXT1)."""
    data = b"\x00" * (max(8, ((width + 3) // 4) * ((height + 3) // 4) * 8))
    body = bytearray()
    body += b"\x00" * 4                          # reserved
    body += struct.pack("<I", 6)                 # fmt = DXT1
    body += struct.pack("<I", 0)                 # id
    body += struct.pack("<HH", width, height)
    body += struct.pack("<I", len(data))
    body += b"\x00" * (0x38 - len(body))         # pad to header end
    body += data
    record = b"XVRT" + struct.pack("<I", 0x38 + len(data)) + bytes(body)
    return record


def _make_afs_bytes(blobs: list[bytes]) -> bytes:
    """Build a minimal AFS archive from the given blob list."""
    out = bytearray()
    out += b"AFS\x00"
    out += struct.pack("<I", len(blobs))
    # Reserve the 8-byte-per-entry table.
    table_off = 0x08
    data_off = table_off + len(blobs) * 8
    table = bytearray()
    payload = bytearray()
    cur = data_off
    for blob in blobs:
        table += struct.pack("<II", cur, len(blob))
        payload += blob
        cur += len(blob)
    out += bytes(table) + bytes(payload)
    return bytes(out)


# ---------------------------------------------------------------------------
# 1. PVM sibling discovery + extract.
# ---------------------------------------------------------------------------


def test_pvm_sibling_extract_round_trip(tmp_path):
    pvr1 = _make_pvr_bytes()
    pvr2 = _make_pvr_bytes()
    pvm = _make_pvm_bytes([pvr1, pvr2])
    (tmp_path / "model.nj").write_bytes(b"NJTL\x00\x00\x00\x00")
    (tmp_path / "model.pvm").write_bytes(pvm)

    discovered = sib.discover_sibling_textures(tmp_path / "model.nj")
    assert len(discovered) == 1
    arc = discovered[0]
    assert arc.magic == "PVM"
    assert arc.list_tiles() == ["pvrt_0000", "pvrt_0001"]
    inner = arc.extract_tile(0)
    assert inner.startswith(b"PVRT")
    # round-trip
    w, h, rgba = decode_pvr(inner)
    assert (w, h) == (4, 4)
    assert len(rgba) == 4 * 4 * 4


# ---------------------------------------------------------------------------
# 2. Single PVR sibling (with GBIX optional, here without).
# ---------------------------------------------------------------------------


def test_single_pvr_sibling(tmp_path):
    pvr = _make_pvr_bytes()
    (tmp_path / "model.nj").write_bytes(b"NJTL\x00\x00\x00\x00")
    (tmp_path / "model.pvr").write_bytes(pvr)

    discovered = sib.discover_sibling_textures(tmp_path / "model.nj")
    pvr_arcs = [a for a in discovered if a.magic == "PVR"]
    assert len(pvr_arcs) == 1
    arc = pvr_arcs[0]
    assert arc.list_tiles() == ["pvr_0000"]
    inner = arc.extract_tile(0)
    assert inner.startswith(b"PVRT")
    w, h, rgba = decode_pvr(inner)
    assert (w, h) == (4, 4)


# ---------------------------------------------------------------------------
# 3. XVM sibling discovery (the PSOBB.IO-shipping format).
# ---------------------------------------------------------------------------


def test_xvm_sibling_discovery(tmp_path):
    xvrt1 = _make_xvrt_bytes(8, 8)
    xvrt2 = _make_xvrt_bytes(16, 16)
    xvm = _make_xvm_bytes([xvrt1, xvrt2])
    (tmp_path / "myModel.nj").write_bytes(b"NJTL\x00\x00\x00\x00")
    (tmp_path / "myModel.xvm").write_bytes(xvm)

    discovered = sib.discover_sibling_textures(tmp_path / "myModel.nj")
    xvm_arcs = [a for a in discovered if a.magic == "XVM"]
    assert len(xvm_arcs) == 1
    arc = xvm_arcs[0]
    tiles = arc.list_tiles()
    assert len(tiles) == 2
    inner0 = arc.extract_tile(0)
    assert inner0[:4] == b"XVRT"


# ---------------------------------------------------------------------------
# 4. GVM sibling discovery (Gamecube — magic+walker; full decode later).
# ---------------------------------------------------------------------------


def test_gvm_sibling_discovery(tmp_path):
    gvr1 = _make_gvr_bytes()
    gvr2 = _make_gvr_bytes()
    gvm = _make_gvm_bytes([gvr1, gvr2])
    (tmp_path / "model.nj").write_bytes(b"NJTL\x00\x00\x00\x00")
    (tmp_path / "model.gvm").write_bytes(gvm)

    discovered = sib.discover_sibling_textures(tmp_path / "model.nj")
    gvm_arcs = [a for a in discovered if a.magic == "GVM"]
    assert len(gvm_arcs) == 1
    arc = gvm_arcs[0]
    assert len(arc.list_tiles()) == 2
    inner = arc.extract_tile(0)
    assert inner[:4] == b"GVRT"


# ---------------------------------------------------------------------------
# 5. PRS-compressed sibling auto-unwraps.
# ---------------------------------------------------------------------------


def test_prs_compressed_sibling_unwraps(tmp_path):
    pvm = _make_pvm_bytes([_make_pvr_bytes()])
    compressed = prs_mod.compress(pvm)
    (tmp_path / "model.nj").write_bytes(b"NJTL\x00\x00\x00\x00")
    (tmp_path / "model.prs").write_bytes(compressed)

    discovered = sib.discover_sibling_textures(tmp_path / "model.nj")
    arcs_prs = [a for a in discovered if a.was_prs]
    assert len(arcs_prs) == 1
    arc = arcs_prs[0]
    assert arc.magic == "PVM"
    assert arc.list_tiles() == ["pvrt_0000"]
    inner = arc.extract_tile(0)
    assert inner.startswith(b"PVRT")


# ---------------------------------------------------------------------------
# 6. AFS sibling — record walk works.
# ---------------------------------------------------------------------------


def test_afs_sibling_records(tmp_path):
    pvr = _make_pvr_bytes()
    afs = _make_afs_bytes([pvr, pvr, pvr])
    (tmp_path / "model.nj").write_bytes(b"NJTL\x00\x00\x00\x00")
    (tmp_path / "model.afs").write_bytes(afs)

    discovered = sib.discover_sibling_textures(tmp_path / "model.nj")
    afs_arcs = [a for a in discovered if a.magic == "AFS"]
    assert len(afs_arcs) == 1
    arc = afs_arcs[0]
    assert len(arc.list_tiles()) == 3
    blob0 = arc.extract_tile(0)
    assert blob0.startswith(b"PVRT")


# ---------------------------------------------------------------------------
# 7. discover_sibling_textures: no candidate ⇒ empty list.
# ---------------------------------------------------------------------------


def test_no_sibling_returns_empty(tmp_path):
    (tmp_path / "lonely_model.nj").write_bytes(b"NJTL\x00\x00\x00\x00")
    discovered = sib.discover_sibling_textures(tmp_path / "lonely_model.nj")
    assert discovered == []


# ---------------------------------------------------------------------------
# 8. _tex suffix pairing (foo.nj + foo_tex.xvm).
# ---------------------------------------------------------------------------


def test_tex_suffix_pairing(tmp_path):
    xvm = _make_xvm_bytes([_make_xvrt_bytes(8, 8)])
    (tmp_path / "boss.nj").write_bytes(b"NJTL\x00\x00\x00\x00")
    (tmp_path / "boss_tex.xvm").write_bytes(xvm)

    discovered = sib.discover_sibling_textures(tmp_path / "boss.nj")
    assert len(discovered) == 1
    assert discovered[0].magic == "XVM"


# ---------------------------------------------------------------------------
# 9. Real PSOBB.IO XVM in the data dir parses (loose smoke).
# ---------------------------------------------------------------------------


def test_real_psobb_xvm_parses():
    candidates = [
        Path("C:/tmp_pso_dev/data/title2.xvm"),
        Path("C:/tmp_pso_dev/data/lobby_billboard.xvm"),
        Path("C:/tmp_pso_dev/data/obj_lobby_main.xvm"),
    ]
    real = next((p for p in candidates if p.is_file()), None)
    if real is None:
        pytest.skip("no real PSOBB XVM in expected paths")
    arc = sib.SiblingArchive.from_path(real)
    assert arc is not None
    assert arc.magic == "XVM"
    tiles = arc.list_tiles()
    assert len(tiles) > 0
    inner = arc.extract_tile(0)
    assert inner[:4] == b"XVRT"


# ---------------------------------------------------------------------------
# 10. SIBLING_MAGICS catalogue is complete (used as a public reference).
# ---------------------------------------------------------------------------


def test_sibling_magics_catalogue():
    labels = {label for _magic, label in sib.SIBLING_MAGICS}
    assert labels >= {"PVM", "GVM", "XVM", "PVR", "GVR", "AFS"}
