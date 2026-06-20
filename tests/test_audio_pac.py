"""Unit tests for the pure-Python .pac PCM SFX-bank codec (formats/audio_pac).

The headline gate is BYTE-EXACTNESS:

    write_pac(parse_pac(x)) == x        for any bank x

proven both on synthetic banks (CI) and over all 27 shipped *.pac banks
(skipped cleanly when ~/PSOBB.IO/data/sound is absent). The remaining tests
cover record replacement, PCM<->WAV round-trips, trim/normalize DSP, and the
tolerant parse of a truncated bank.
"""
from __future__ import annotations

import io
import os
import struct
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from formats import audio_pac as ap  # noqa: E402

PSOBB_SOUND = Path(os.path.expanduser("~/PSOBB.IO/data/sound"))
HAS_PAC_TREE = PSOBB_SOUND.is_dir() and any(PSOBB_SOUND.glob("*.pac"))


# ---------------------------------------------------------------------------
# synthetic bank builders
# ---------------------------------------------------------------------------
def _wav_record(pcm: bytes, pad: int = 8) -> bytes:
    """A canonical Layout-A record: WFX + 'data' + size + PCM + pad."""
    return ap.WFX_SIG + b"data" + struct.pack("<I", len(pcm)) + pcm + (b"\x00" * pad)


def _synthetic_bank(n_records: int = 3, pads=(8, 12, 0)) -> bytes:
    """Build a multi-record .pac from sine-ish PCM with varied padding."""
    out = bytearray()
    for i in range(n_records):
        nsamp = 220 * (i + 1)
        t = np.arange(nsamp, dtype=np.float64)
        wave_pcm = (np.sin(t / (4 + i)) * (8000 + 1000 * i)).astype(np.int16).tobytes()
        out += _wav_record(wave_pcm, pad=pads[i % len(pads)])
    return bytes(out)


# ---------------------------------------------------------------------------
# THE byte-exact gate
# ---------------------------------------------------------------------------
def test_pac_roundtrip_synthetic_byte_exact():
    x = _synthetic_bank(4, pads=(8, 14, 0, 30))
    bank = ap.parse_pac(x)
    assert len(bank.records) == 4
    assert all(r.structured for r in bank.records)
    assert ap.write_pac(bank) == x, "write_pac(parse_pac(x)) must be byte-exact"


def test_pac_roundtrip_preserves_prefix_and_unstructured():
    # A prefix before the first sig + a trailing opaque tail must round-trip.
    body = _synthetic_bank(2)
    x = b"\xde\xad\xbe\xef" + body
    bank = ap.parse_pac(x)
    assert bank.prefix == b"\xde\xad\xbe\xef"
    assert ap.write_pac(bank) == x


def test_pac_empty_input():
    bank = ap.parse_pac(b"")
    assert bank.records == []
    assert ap.write_pac(bank) == b""


def test_pac_no_signature_is_one_opaque_record():
    x = b"not a pac at all, just bytes" * 4
    bank = ap.parse_pac(x)
    assert len(bank.records) == 1
    assert bank.records[0].structured is False
    assert ap.write_pac(bank) == x


# ---------------------------------------------------------------------------
# record replacement
# ---------------------------------------------------------------------------
def test_pac_record_replace_keeps_count_and_others_intact():
    x = _synthetic_bank(3)
    bank = ap.parse_pac(x)
    orig_rec1 = bank.records[1].pcm
    new_pcm = (np.zeros(500, dtype=np.int16) + 1234).astype(np.int16).tobytes()
    new_bank = ap.replace_record_pcm(bank, 1, new_pcm)

    assert len(new_bank.records) == len(bank.records), "record count must be stable"
    # The swapped record carries the new PCM...
    assert new_bank.records[1].pcm == new_pcm
    assert new_bank.records[1].pcm != orig_rec1
    # ...and the neighbours are byte-identical to the originals.
    assert new_bank.records[0].raw == bank.records[0].raw
    assert new_bank.records[2].raw == bank.records[2].raw
    # Re-parse the serialized result: still structured & round-trips.
    reparsed = ap.parse_pac(ap.write_pac(new_bank))
    assert reparsed.records[1].pcm == new_pcm
    assert ap.write_pac(reparsed) == ap.write_pac(new_bank)


def test_pac_replace_rejects_unstructured_record():
    bank = ap.parse_pac(b"\x00" * 64)  # opaque
    with pytest.raises(ValueError):
        ap.replace_record_pcm(bank, 0, b"\x00\x00")


def test_pac_replace_rejects_misaligned_pcm():
    bank = ap.parse_pac(_synthetic_bank(2))
    with pytest.raises(ValueError):
        ap.replace_record_pcm(bank, 0, b"\x01")  # odd byte count (not frame-aligned)


# ---------------------------------------------------------------------------
# PCM <-> WAV
# ---------------------------------------------------------------------------
def test_pcm_wav_roundtrip():
    pcm = np.arange(-2000, 2000, dtype=np.int16).tobytes()
    wav = ap.pcm_to_wav(pcm)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    got, rate, ch, bits = ap.wav_to_pcm(wav)
    assert got == pcm
    assert (rate, ch, bits) == (ap.PCM_SAMPLE_RATE, ap.PCM_CHANNELS, ap.PCM_BITS)


def test_record_to_wav_plays_back_pcm():
    bank = ap.parse_pac(_synthetic_bank(1))
    wav = ap.record_to_wav(bank.records[0])
    pcm, _r, _c, _b = ap.wav_to_pcm(wav)
    assert pcm == bank.records[0].pcm


def test_wav_to_pcm_rejects_garbage():
    with pytest.raises(ValueError):
        ap.wav_to_pcm(b"not a wav")


# ---------------------------------------------------------------------------
# DSP
# ---------------------------------------------------------------------------
def test_trim_pcm_frame_aligned():
    samples = np.arange(0, 1000, dtype=np.int16)
    pcm = samples.tobytes()
    out = ap.trim_pcm(pcm, start_frame=100, end_frame=200)
    assert np.frombuffer(out, dtype=np.int16).tolist() == list(range(100, 200))


def test_trim_pcm_clamps_out_of_range():
    pcm = np.arange(0, 50, dtype=np.int16).tobytes()
    out = ap.trim_pcm(pcm, start_frame=40, end_frame=1000)
    assert np.frombuffer(out, dtype=np.int16).tolist() == list(range(40, 50))


def test_normalize_pcm_hits_target_peak():
    # Quiet signal: peak ~1000 -> normalize to -1 dBFS (~29204).
    samples = (np.sin(np.linspace(0, 6.28, 2000)) * 1000).astype(np.int16)
    out = ap.normalize_pcm(samples.tobytes(), target_dbfs=-1.0)
    peak = int(np.max(np.abs(np.frombuffer(out, dtype=np.int16))))
    target = (10 ** (-1.0 / 20.0)) * 32767.0
    assert abs(peak - target) <= 2, f"peak {peak} not at -1 dBFS target {target:.0f}"


def test_normalize_pcm_silence_is_noop():
    pcm = np.zeros(100, dtype=np.int16).tobytes()
    assert ap.normalize_pcm(pcm) == pcm


def test_waveform_peaks_shape_and_range():
    samples = (np.sin(np.linspace(0, 60, 44100)) * 20000).astype(np.int16)
    wf = ap.waveform_peaks(samples.tobytes(), buckets=200)
    assert wf["buckets"] == 200
    assert len(wf["min"]) == len(wf["max"]) == len(wf["rms"]) == 200
    assert all(-1.0 <= v <= 1.0 for v in wf["min"])
    assert all(-1.0 <= v <= 1.0 for v in wf["max"])
    assert all(0.0 <= v <= 1.0 for v in wf["rms"])


# ---------------------------------------------------------------------------
# tolerant parse (truncation -> good prefix + warning, no exception)
# ---------------------------------------------------------------------------
def test_pac_tolerant_parse_truncated_last_record():
    x = _synthetic_bank(3)
    # Lop off the back half of the final record's PCM.
    truncated = x[: len(x) - 100]
    bank = ap.parse_pac(truncated)  # must NOT raise
    # The good records survive...
    assert bank.records[0].structured
    assert bank.records[1].structured
    # ...the truncated tail is flagged unstructured with a warning...
    assert bank.records[-1].structured is False
    assert any("truncat" in (w or "").lower() for w in bank.warnings)
    # ...and the round-trip is STILL byte-exact on the truncated input.
    assert ap.write_pac(bank) == truncated


def test_pac_truncated_after_replace_disabled():
    truncated = _synthetic_bank(2)[:-50]
    bank = ap.parse_pac(truncated)
    assert bank.replace_safe is False  # a partly-broken bank is not a target


# ---------------------------------------------------------------------------
# LIVE-tree parity: byte-exact over every shipped *.pac (CI-skip if absent)
# ---------------------------------------------------------------------------
def _shipped_pac_files() -> list:
    if not HAS_PAC_TREE:
        return []
    return sorted(PSOBB_SOUND.glob("*.pac"))


@pytest.mark.skipif(not HAS_PAC_TREE, reason="PSOBB.IO/data/sound *.pac not installed")
@pytest.mark.parametrize("path", _shipped_pac_files(), ids=lambda p: p.name)
def test_all_live_pac_roundtrip(path):
    """THE ship-gate: every real bank round-trips byte-for-byte."""
    raw = path.read_bytes()
    bank = ap.parse_pac(raw)
    assert ap.write_pac(bank) == raw, f"{path.name} is not byte-exact"
    assert len(bank.records) >= 1


@pytest.mark.skipif(not HAS_PAC_TREE, reason="PSOBB.IO/data/sound *.pac not installed")
def test_live_pac_has_structured_replaceable_banks():
    """Sanity: the live tree has at least some fully-structured (replaceable)
    banks and the codec decodes real PCM from them."""
    any_replaceable = False
    for path in _shipped_pac_files():
        bank = ap.parse_pac(path.read_bytes())
        if bank.replace_safe:
            any_replaceable = True
            rec = bank.records[0]
            assert rec.pcm_size > 0
            wav = ap.record_to_wav(rec)
            assert wav[:4] == b"RIFF"
            break
    assert any_replaceable, "expected at least one fully-structured live .pac bank"
