"""PRS parity tests against an INDEPENDENT reference decoder + real assets.

These tests back the 2026-06-19 PRS audit. The headline guarantee is
*decode byte-exactness*: ``formats.prs.decompress`` must produce output
identical to an independent decoder transcribed directly from
libpsoarchive ``src/PRS-decomp.c`` (``do_decompress``) -- which is the
ground-truth C implementation, NOT derived from our code, so agreement is
real evidence rather than a tautology.

We also assert no-build round-trips on real PSOBB assets:
    decompress(compress(payload))         == payload   (greedy)
    decompress(compress_optimal(payload)) == payload   (optimal)

Asset roots are probed in order; tests skip cleanly when no install is
present (e.g. CI). The optimal round-trip is capped to sub-MB payloads
because the pure-Python shortest-path encoder is slow on multi-MB blobs
(it is exercised on the small/medium assets, which is sufficient to prove
emission correctness -- the byte format is size-independent).
"""
from __future__ import annotations
import os

from pathlib import Path

import pytest

from formats.prs import compress, compress_optimal, decompress

# ---------------------------------------------------------------------------
# Asset discovery
# ---------------------------------------------------------------------------
_ASSET_ROOTS = [
    Path(os.path.expanduser("~/EphineaPSO/data")),
    Path(os.path.expanduser("~/PSOBB.IO/data")),
]
DATA_DIR = next((p for p in _ASSET_ROOTS if p.is_dir()), None)
HAS_ASSETS = DATA_DIR is not None

OPTIMAL_CAP = 768 * 1024  # pure-Python optimal is slow above ~1 MB


# ---------------------------------------------------------------------------
# Independent reference decoder -- byte-for-byte transcription of
# libpsoarchive src/PRS-decomp.c do_decompress() + fetch_bit/byte/short.
# Deliberately self-contained and NOT importing anything from formats.prs.
# ---------------------------------------------------------------------------
class _RefError(Exception):
    pass


def ref_decompress(src: bytes) -> bytes:
    """Reference PRS decoder (libpsoarchive do_decompress transcription)."""
    n = len(src)
    if n < 3:
        raise _RefError("src < 3 bytes")
    dst = bytearray()
    pos = 0
    flags = 0
    bit_pos = 0

    def fetch_bit() -> int:
        nonlocal pos, flags, bit_pos
        if bit_pos == 0:
            if pos >= n:
                raise _RefError("eof on bit")
            flags = src[pos]
            pos += 1
            bit_pos = 8
        rv = flags & 1
        flags >>= 1
        bit_pos -= 1
        return rv

    def fetch_byte() -> int:
        nonlocal pos
        if pos >= n:
            raise _RefError("eof on byte")
        rv = src[pos]
        pos += 1
        return rv

    def fetch_short() -> int:
        nonlocal pos
        if pos + 1 >= n:  # libpsoarchive guard
            raise _RefError("eof on short")
        rv = src[pos]
        pos += 1
        rv |= src[pos] << 8
        pos += 1
        return rv

    while True:
        if fetch_bit():
            dst.append(fetch_byte())
            continue
        if fetch_bit():
            # long copy / EOF
            offset = fetch_short()
            if offset == 0:
                return bytes(dst)
            size = offset & 0x0007
            offset >>= 3
            if size == 0:
                size = fetch_byte() + 1
            else:
                size += 2
            offset = (offset | 0xFFFFE000) - 0x100000000
        else:
            # short copy
            hi = fetch_bit()
            lo = fetch_bit()
            size = (lo | (hi << 1)) + 2
            offset = (fetch_byte() | 0xFFFFFF00) - 0x100000000
        for _ in range(size):
            tmp = len(dst) + offset
            if tmp < 0:
                raise _RefError("backref before start")
            dst.append(dst[tmp])


# ---------------------------------------------------------------------------
# Synthetic vectors: decode parity holds without any game install.
# ---------------------------------------------------------------------------
_SYNTH = [
    b"",
    b"A",
    b"\x00" * 1024,
    b"\xff" * 300,
    bytes(range(256)),
    b"ABCDABCD" * 200,
    (b"the quick brown fox jumps over the lazy dog" * 16),
    b"AB" + b"\x01" * 0xFE + b"AB",            # short-copy offset boundary
    b"ABCDEFGHIJ" + b"\x00" * (0x1FFF - 10) + b"ABCDEFGHIJ",  # long boundary
    b"A" * 257,                                # extended-copy max size
]


@pytest.mark.parametrize("payload", _SYNTH, ids=lambda p: f"len{len(p)}")
def test_decode_parity_vs_reference_synthetic(payload):
    """Our decoder agrees byte-for-byte with the libpsoarchive reference."""
    for enc in (compress(payload), compress_optimal(payload)):
        ours = decompress(enc)
        assert ours == payload
        ref = ref_decompress(enc)
        assert ours == ref, "ours.decompress disagrees with reference decoder"


# ---------------------------------------------------------------------------
# Real-asset parity. Decode byte-exact vs reference + greedy/optimal RT.
# ---------------------------------------------------------------------------
def _first_inner_prs(bml_bytes: bytes):
    from formats.bml import parse_bml

    for ent in parse_bml(bml_bytes):
        if (
            ent.size_compressed
            and ent.size_decompressed
            and ent.size_compressed != ent.size_decompressed
        ):
            return bml_bytes[ent.offset:ent.offset + ent.size_compressed]
    return None


def _collect_real_assets():
    items = []
    if not HAS_ASSETS:
        return items
    for p in sorted(DATA_DIR.glob("*.prs")):
        try:
            items.append((p.name, p.read_bytes()))
        except OSError:
            pass
    bmls = sorted(DATA_DIR.glob("*.bml"), key=lambda q: q.stat().st_size)
    for p in bmls[:25] + bmls[-10:]:
        try:
            raw = _first_inner_prs(p.read_bytes())
        except Exception:
            raw = None
        if raw:
            items.append((p.name, raw))
    return items


_REAL = _collect_real_assets()


@pytest.mark.skipif(not HAS_ASSETS, reason="no PSOBB install present")
def test_real_assets_present():
    assert _REAL, "asset dir exists but no PRS assets discovered"


@pytest.mark.skipif(not _REAL, reason="no PSOBB PRS assets present")
@pytest.mark.parametrize("name,comp", _REAL, ids=[n for n, _ in _REAL])
def test_real_asset_decode_and_roundtrip(name, comp):
    # 1) Decode byte-exact vs the independent reference decoder. Some real
    #    blobs (Sega "stub XVM") are unterminated; the strict reference
    #    raises EOF there -- in that case fall back to our tolerant decode
    #    (the documented project behavior) and skip the byte-equality leg.
    ours = decompress(comp)
    try:
        ref = ref_decompress(comp)
    except _RefError:
        ours = decompress(comp, tolerant=True)
        ref = None
    if ref is not None:
        assert ours == ref, f"{name}: decode disagrees with reference"

    # 2) Greedy round-trip on every asset.
    assert decompress(compress(ours)) == ours, f"{name}: greedy round-trip"

    # 3) Optimal round-trip (capped by size; format is size-independent).
    if len(ours) <= OPTIMAL_CAP:
        assert decompress(compress_optimal(ours)) == ours, f"{name}: optimal round-trip"
