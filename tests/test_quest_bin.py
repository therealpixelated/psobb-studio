"""Tests for formats.quest_bin — the PSOBB quest .bin/.qst container codec.

The parity gate for this module is the **byte-exact decompressed-.bin
round-trip** on the phantasmal-world fixtures (quest118_e, quest27_e):

  1. parse_qst(.qst) -> extract+PRS-decompress .bin == the standalone
     *_decompressed.bin fixture, BYTE-FOR-BYTE.
  2. parse_bin(decompressed) -> serialize_bin == the decompressed bytes,
     BYTE-FOR-BYTE.

Plus a corpus sweep over the newserv quest tree and any .qst in the
tethealla fixture tree, and an always-run synthetic vector so the module
is exercised even on a bare clone with no reference data.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest

from formats import prs
from formats.quest_bin import (
    BIN_FORMAT_BB,
    CODE_OFFSET_BB,
    build_qst,
    extract_bin,
    parse_bin,
    parse_qst,
    serialize_bin,
    serialize_qst,
    to_json,
)

# ---------------------------------------------------------------------------
# Fixture locations (gated; skip-clean if absent on a bare clone)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
_PW_REL = Path("_reference") / "phantasmal-world" / "psolib" / "src" / "commonTest" / "resources"


def _find_pw_fixtures() -> Path:
    """Locate the phantasmal-world fixture dir.

    ``_reference/`` is gitignored, so in a detached git worktree it lives
    only in the main checkout. Try the current tree first, then the
    sibling ``psobb-studio`` checkout, then a couple of common roots.
    """
    candidates = [
        ROOT / _PW_REL,
        ROOT.parent / "psobb-studio" / _PW_REL,
        Path(os.path.expanduser("~/Repositories/psobb-studio")) / _PW_REL,
    ]
    for c in candidates:
        if (c / "quest118_e.qst").is_file():
            return c
    return ROOT / _PW_REL  # default (won't exist; HAS_PW gates it off)


PW_FIXTURES = _find_pw_fixtures()
HAS_PW = PW_FIXTURES.is_dir() and (PW_FIXTURES / "quest118_e.qst").is_file()

NEWSERV_QUESTS = Path(os.path.expanduser("~/Repositories/newserv/system/quests"))
TETHEALLA_QUESTS = PW_FIXTURES / "tethealla_v0.143_quests"


# ---------------------------------------------------------------------------
# Synthetic vector (always runs — no reference data required)
# ---------------------------------------------------------------------------
def _make_synthetic_bin() -> bytes:
    """Build a minimal but structurally valid BB .bin in memory.

    Header is the full 0x122C BB header with a few fields set; a tiny code
    blob; a 3-entry label table.
    """
    header = bytearray(CODE_OFFSET_BB)
    # quest_number=4242, episode=1, max_players=4, joinable=1
    struct.pack_into("<H", header, 0x10, 4242)
    header[0x14] = 1  # episode
    header[0x15] = 4  # max_players
    header[0x16] = 1  # joinable
    name = "Synthetic Quest".encode("utf-16-le")
    header[0x18:0x18 + len(name)] = name
    short = "short".encode("utf-16-le")
    header[0x58:0x58 + len(short)] = short
    # A distinctive floor assignment in the trailing block.
    header[0x3AC:0x3AC + 5] = bytes([0x01, 0x02, 0x03, 0x04, 0x05])

    code = bytes(range(0x40)) * 3  # 0xC0 bytes of "bytecode"
    labels = [0, 0x10, 0x20]
    label_blob = struct.pack(f"<{len(labels)}I", *labels)

    code_offset = len(header)
    label_table_offset = code_offset + len(code)
    total = label_table_offset + len(label_blob)
    struct.pack_into("<III", header, 0, code_offset, label_table_offset, total)
    struct.pack_into("<I", header, 0x0C, 0xFFFFFFFF)
    return bytes(header) + code + label_blob


def test_synthetic_bin_roundtrip():
    raw = _make_synthetic_bin()
    qb = parse_bin(raw)
    assert qb.fmt == BIN_FORMAT_BB
    assert qb.code_offset == CODE_OFFSET_BB
    assert qb.quest_number == 4242
    assert qb.episode == 1
    assert qb.max_players == 4
    assert qb.joinable == 1
    assert qb.name == "Synthetic Quest"
    assert qb.short_description == "short"
    assert qb.label_offsets == [0, 0x10, 0x20]
    assert qb.label_count == 3
    assert qb.floor_assignments()[0] == [1, 2, 3, 4, 5]
    # Byte-exact round trip.
    assert serialize_bin(qb) == raw


def test_synthetic_qst_roundtrip_decompressed_bin():
    """Build .qst from compressed payloads, parse it back, recover the .bin."""
    raw_bin = _make_synthetic_bin()
    raw_dat = b"DAT" + bytes(range(0x100)) * 8  # arbitrary "map" payload
    comp_bin = prs.compress(raw_bin)
    comp_dat = prs.compress(raw_dat)

    qst = build_qst(
        comp_bin,
        comp_dat,
        bin_filename="quest9999.bin",
        dat_filename="quest9999.dat",
        name="quest9999_j",
    )
    blob = serialize_qst(qst)

    reparsed = parse_qst(blob)
    assert reparsed.fmt == BIN_FORMAT_BB
    assert reparsed.online is True
    assert reparsed.bin_file is not None
    assert reparsed.dat_file is not None
    # The decompressed .bin must come back byte-exact through the transport.
    assert extract_bin(reparsed) == raw_bin
    # And the .dat too.
    assert reparsed.dat_file.decompressed() == raw_dat
    # And the .bin parses + reserializes byte-exact.
    assert serialize_bin(parse_bin(extract_bin(reparsed))) == raw_bin


def test_parse_bin_rejects_garbage():
    with pytest.raises(ValueError):
        parse_bin(b"\x00\x00\x00\x00")  # too short
    with pytest.raises(ValueError):
        # code_offset way past end
        parse_bin(struct.pack("<IIII", 0xFFFF, 0xFFFF, 0x40, 0xFFFFFFFF) + b"\x00" * 0x30)


def test_parse_qst_rejects_bad_signature():
    with pytest.raises(ValueError):
        parse_qst(b"\xde\xad\xbe\xef" + b"\x00" * 0x60)


def test_to_json_smoke():
    qb = parse_bin(_make_synthetic_bin())
    j = to_json(qb)
    assert "Synthetic Quest" in j
    assert '"format": "BB"' in j


# ---------------------------------------------------------------------------
# Parity gate — phantasmal-world fixtures (quest118_e, quest27_e)
# ---------------------------------------------------------------------------
PW_CASES = [
    ("quest118_e.qst", "quest118_e_decompressed.bin", 118, "Towards the Future"),
    ("quest27_e.qst", "quest27_e_decompressed.bin", 27, "Seat of the Heart"),
]


@pytest.mark.skipif(not HAS_PW, reason="phantasmal-world fixtures not present")
@pytest.mark.parametrize("qst_name,dec_name,quest_no,quest_name", PW_CASES)
def test_qst_to_decompressed_bin_byte_exact(qst_name, dec_name, quest_no, quest_name):
    """parse_qst -> extract+decompress .bin == the *_decompressed.bin oracle."""
    qst_bytes = (PW_FIXTURES / qst_name).read_bytes()
    oracle = (PW_FIXTURES / dec_name).read_bytes()

    qst = parse_qst(qst_bytes)
    assert qst.fmt == BIN_FORMAT_BB
    assert qst.online is True
    assert qst.bin_file is not None
    assert qst.dat_file is not None

    decompressed = extract_bin(qst)
    # THE parity assertion.
    assert decompressed == oracle, (
        f"{qst_name}: extracted .bin ({len(decompressed)}) != oracle ({len(oracle)})"
    )


@pytest.mark.skipif(not HAS_PW, reason="phantasmal-world fixtures not present")
@pytest.mark.parametrize("qst_name,dec_name,quest_no,quest_name", PW_CASES)
def test_parse_bin_serialize_bin_byte_exact(qst_name, dec_name, quest_no, quest_name):
    """parse_bin -> serialize_bin == the decompressed bytes, byte-for-byte."""
    oracle = (PW_FIXTURES / dec_name).read_bytes()
    qb = parse_bin(oracle)
    assert serialize_bin(qb) == oracle


@pytest.mark.skipif(not HAS_PW, reason="phantasmal-world fixtures not present")
@pytest.mark.parametrize("qst_name,dec_name,quest_no,quest_name", PW_CASES)
def test_header_metadata_sane(qst_name, dec_name, quest_no, quest_name):
    """Header fields decode sanely (quest_number / episode / name)."""
    oracle = (PW_FIXTURES / dec_name).read_bytes()
    qb = parse_bin(oracle)
    assert qb.fmt == BIN_FORMAT_BB
    assert qb.code_offset == CODE_OFFSET_BB
    assert qb.quest_number == quest_no
    assert qb.episode == 0  # both fixtures are Episode 1 (encoded as 0)
    assert qb.name == quest_name
    assert qb.size == len(oracle)
    # Floor assignments table is present (16 entries) and structured.
    fa = qb.floor_assignments()
    assert len(fa) == 16
    assert all(len(e) == 5 for e in fa)


@pytest.mark.skipif(not HAS_PW, reason="phantasmal-world fixtures not present")
@pytest.mark.parametrize("qst_name,dec_name,quest_no,quest_name", PW_CASES)
def test_qst_repack_preserves_decompressed_bin(qst_name, dec_name, quest_no, quest_name):
    """serialize_qst(parse_qst(x)) re-parses to the same decompressed .bin.

    (.qst re-pack need not be byte-identical, but the embedded compressed
    payloads are preserved verbatim, so the round-trip recovers the exact
    decompressed .bin.)
    """
    oracle = (PW_FIXTURES / dec_name).read_bytes()
    qst = parse_qst((PW_FIXTURES / qst_name).read_bytes())
    repacked = serialize_qst(qst)
    qst2 = parse_qst(repacked)
    assert extract_bin(qst2) == oracle


# ---------------------------------------------------------------------------
# Corpus sweep — newserv quests + tethealla .qst tree
# ---------------------------------------------------------------------------
def _iter_corpus_files():
    """Yield candidate quest files from the local corpora (if present)."""
    if NEWSERV_QUESTS.is_dir():
        for p in NEWSERV_QUESTS.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".bin", ".qst"):
                yield p
    if TETHEALLA_QUESTS.is_dir():
        for p in TETHEALLA_QUESTS.rglob("*.qst"):
            if p.is_file():
                yield p


def _decompressed_bin_from_file(path: Path):
    """Best-effort: return the decompressed .bin bytes for a corpus file.

    Returns None (skip) if the file isn't a recognized BB .qst or a
    PRS-compressed BB .bin we can carve.
    """
    data = path.read_bytes()
    if len(data) < 0x10:
        return None

    suffix = path.suffix.lower()

    if suffix == ".qst":
        # Only the BB online/download form is supported; others skip-clean.
        sig = struct.unpack_from(">I", data, 0)[0]
        if sig not in (0x58004400, 0x5800A600):
            return None
        try:
            qst = parse_qst(data)
        except ValueError:
            return None
        bf = qst.bin_file
        if bf is None:
            return None
        try:
            return bf.decompressed()
        except Exception:
            return None

    if suffix == ".bin":
        # newserv bare .bin files are PRS-compressed quest scripts.
        try:
            dec = prs.decompress(data)
        except Exception:
            return None
        # Validate it carves as a recognized quest .bin (BB code_offset).
        if len(dec) < 0x10:
            return None
        code_off = struct.unpack_from("<I", dec, 0)[0]
        if code_off != CODE_OFFSET_BB:
            return None  # non-BB / not a quest script — skip cleanly
        return dec

    return None


def test_corpus_sweep_bin_roundtrip():
    """For every recognized corpus quest: parse_bin->serialize_bin is exact."""
    files = list(_iter_corpus_files())
    if not files:
        pytest.skip("no corpus quests present (newserv / tethealla trees absent)")

    ok = 0
    skipped = 0
    failures = []
    for path in files:
        try:
            dec = _decompressed_bin_from_file(path)
        except Exception:
            skipped += 1
            continue
        if dec is None:
            skipped += 1
            continue
        try:
            qb = parse_bin(dec)
            rt = serialize_bin(qb)
        except Exception as exc:  # parse/serialize blew up on a real file
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        if rt != dec:
            failures.append(f"{path.name}: serialize_bin != decompressed input")
            continue
        ok += 1

    print(
        f"\n[corpus sweep] ok={ok} skipped={skipped} "
        f"failures={len(failures)} total={len(files)}"
    )
    if failures:
        sample = "\n  ".join(failures[:10])
        pytest.fail(f"{len(failures)} corpus round-trip failures:\n  {sample}")

    # Sanity: when the corpus is present we expect a healthy number of hits.
    assert ok > 0, f"corpus present ({len(files)} files) but nothing round-tripped"
