"""Pure-Python reader and writer for PSOBB ``.pac`` PCM SFX banks.

A ``.pac`` file (``data/sound/<area>.pac``) is a *headerless concatenation*
of WAV-style PCM sound records. Each record begins with a 16-byte
``WAVEFORMATEX`` block

    01 00 01 00 22 56 00 00 44 ac 00 00 02 00 10 00
    └fmt┘ └ch┘ └─sampleRate─┘ └blkAlign┘ └bits┘
    PCM   mono  22050 Hz       2         16-bit

immediately followed (in the common layout) by a RIFF ``data`` sub-chunk
(``"data" + u32 size + PCM``), then a few bytes of trailing padding before
the next record. There is **NO fixed stride** and **NO global file header**:
records are found by resyncing to the next ``WAVEFORMATEX`` signature, and
the trailing padding between records is variable (8..38 bytes observed, and
one outlier of ~1.2 MB in ephinea.pac). A handful of banks
(boss09/crater/desert/wilds) carry no ``data`` literal at all — their records
are an opaque variant; we still slice and round-trip them byte-for-byte but
mark them ``structured=False`` so editing tools leave them alone.

The cardinal guarantee of this module:

    write_pac(parse_pac(x)) == x      # byte-for-byte, for ANY input

is achieved by slicing each record as the raw bytes from one signature to the
next (the final record runs to EOF) and preserving any prefix before the first
signature verbatim. Structural fields (``data`` offset / PCM size / padding)
are decoded *opportunistically* on top of that verbatim slice so audition,
trim and normalize can operate without ever threatening the round-trip.

This module is pure Python. ``numpy`` is used only by the optional
``trim_pcm`` / ``normalize_pcm`` helpers (it is already a project dependency);
WAV (de)muxing uses the stdlib ``wave`` module. No subprocess, no ffmpeg.
"""
from __future__ import annotations

import io
import struct
import wave
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Format constants
# ---------------------------------------------------------------------------

# The canonical WAVEFORMATEX block every record opens with:
#   wFormatTag=1 (PCM), nChannels=1, nSamplesPerSec=22050,
#   nAvgBytesPerSec=44100, nBlockAlign=2, wBitsPerSample=16.
PCM_FORMAT_TAG = 1
PCM_CHANNELS = 1
PCM_SAMPLE_RATE = 22050
PCM_BITS = 16

WFX_SIG = struct.pack(
    "<HHIIHH",
    PCM_FORMAT_TAG,           # wFormatTag
    PCM_CHANNELS,             # nChannels
    PCM_SAMPLE_RATE,          # nSamplesPerSec
    PCM_SAMPLE_RATE * PCM_CHANNELS * (PCM_BITS // 8),  # nAvgBytesPerSec
    PCM_CHANNELS * (PCM_BITS // 8),                    # nBlockAlign
    PCM_BITS,                 # wBitsPerSample
)
assert WFX_SIG == bytes.fromhex("010001002256000044ac000002001000"), WFX_SIG.hex()

_WFX_LEN = 16
_DATA_TAG = b"data"


# ---------------------------------------------------------------------------
# Record model
# ---------------------------------------------------------------------------
@dataclass
class PacRecord:
    """One PCM record carved from a ``.pac`` bank.

    ``raw`` is the verbatim byte slice (signature .. next signature). It is
    the *single source of truth* for serialization — ``write_pac`` simply
    concatenates ``raw`` for every record, so the round-trip is byte-exact
    regardless of how much (or how little) structure we managed to decode.

    ``structured`` is True only when a clean ``data`` sub-chunk was located,
    in which case ``pcm_offset`` / ``pcm_size`` index into ``raw`` and
    ``trailing_pad`` counts the bytes after the PCM up to the next record.
    Editing helpers (decode/trim/normalize/replace) operate ONLY on
    structured records.
    """

    raw: bytes
    structured: bool = False
    pcm_offset: int = 0          # offset of PCM samples within ``raw``
    pcm_size: int = 0            # length of the PCM payload in bytes
    trailing_pad: int = 0        # padding bytes after the PCM, before next rec
    warning: Optional[str] = None

    # ---- convenience views ------------------------------------------------
    @property
    def pcm(self) -> bytes:
        """The raw little-endian 16-bit mono PCM bytes (empty if unstructured)."""
        if not self.structured:
            return b""
        return bytes(self.raw[self.pcm_offset:self.pcm_offset + self.pcm_size])

    @property
    def sample_rate(self) -> int:
        return PCM_SAMPLE_RATE

    @property
    def channels(self) -> int:
        return PCM_CHANNELS

    @property
    def bits(self) -> int:
        return PCM_BITS

    @property
    def duration_s(self) -> float:
        if not self.structured or self.pcm_size == 0:
            return 0.0
        frame = PCM_CHANNELS * (PCM_BITS // 8)
        return (self.pcm_size / frame) / PCM_SAMPLE_RATE


@dataclass
class PacBank:
    """The parsed form of a whole ``.pac`` file.

    ``prefix`` holds any bytes before the first signature (empty for every
    real PSOBB bank, but preserved for round-trip safety). ``warnings``
    aggregates per-file issues (truncation, unstructured records).
    """

    records: List[PacRecord] = field(default_factory=list)
    prefix: bytes = b""
    warnings: List[str] = field(default_factory=list)

    @property
    def replace_safe(self) -> bool:
        """True iff every record is cleanly structured (so a record swap can be
        re-serialized byte-cleanly and the bank is a sound REPLACE target)."""
        return bool(self.records) and all(r.structured for r in self.records)


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------
def _find_signatures(buf: bytes) -> List[int]:
    """Return all offsets of the WAVEFORMATEX signature in ``buf``."""
    offs: List[int] = []
    i = buf.find(WFX_SIG)
    while i != -1:
        offs.append(i)
        i = buf.find(WFX_SIG, i + 1)
    return offs


def _decode_record_structure(raw: bytes) -> Tuple[bool, int, int, int, Optional[str]]:
    """Best-effort decode of one record's PCM framing.

    Returns ``(structured, pcm_offset, pcm_size, trailing_pad, warning)``.

    Only the common ``WFX(16) + "data" + u32 size + PCM + pad`` layout is
    treated as structured. Anything else (variant banks with no ``data``
    literal, or a truncated final record) round-trips fine but is reported
    unstructured so editors skip it.
    """
    if len(raw) < _WFX_LEN:
        return (False, 0, 0, 0, "record shorter than WAVEFORMATEX header")
    # The 'data' sub-chunk normally sits immediately after the 16-byte WFX.
    if raw[_WFX_LEN:_WFX_LEN + 4] != _DATA_TAG:
        return (False, 0, 0, 0, "no 'data' sub-chunk after WAVEFORMATEX (variant/opaque record)")
    if len(raw) < _WFX_LEN + 8:
        return (False, 0, 0, 0, "truncated 'data' sub-chunk header")
    size = struct.unpack_from("<I", raw, _WFX_LEN + 4)[0]
    pcm_offset = _WFX_LEN + 8
    pcm_end = pcm_offset + size
    if pcm_end > len(raw):
        # Truncated PCM: keep the good prefix, flag it, stay unstructured so
        # nothing tries to splice a short buffer back in.
        return (
            False,
            pcm_offset,
            len(raw) - pcm_offset,
            0,
            f"declared PCM size {size} exceeds record bytes {len(raw) - pcm_offset} (truncated)",
        )
    trailing_pad = len(raw) - pcm_end
    return (True, pcm_offset, size, trailing_pad, None)


def parse_pac(data: bytes) -> PacBank:
    """Parse a ``.pac`` bank into records.

    Tolerant by contract:
      * Records are located by resyncing to the next WAVEFORMATEX signature
        (no fixed stride assumption).
      * Any bytes before the first signature are preserved as ``prefix``.
      * A truncated trailing record yields a good-prefix record plus a
        bank-level warning rather than an exception.
      * A bank with no signatures at all returns a single opaque record
        carrying the whole buffer (still byte-exact on write).

    The returned bank satisfies ``write_pac(parse_pac(x)) == x`` for any
    ``x`` (bytes-like).
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("parse_pac expects bytes-like input")
    buf = bytes(data)
    bank = PacBank()

    offs = _find_signatures(buf)
    if not offs:
        # No recognizable records. Preserve everything verbatim as one
        # opaque record so the round-trip still holds.
        if buf:
            bank.records.append(
                PacRecord(raw=buf, structured=False,
                          warning="no WAVEFORMATEX signature found"))
            bank.warnings.append("no WAVEFORMATEX signature found in bank")
        return bank

    if offs[0] != 0:
        bank.prefix = buf[:offs[0]]
        bank.warnings.append(f"{offs[0]} bytes precede the first record")

    for k, start in enumerate(offs):
        end = offs[k + 1] if k + 1 < len(offs) else len(buf)
        raw = buf[start:end]
        structured, pcm_off, pcm_sz, pad, warn = _decode_record_structure(raw)
        rec = PacRecord(
            raw=raw,
            structured=structured,
            pcm_offset=pcm_off,
            pcm_size=pcm_sz,
            trailing_pad=pad,
            warning=warn,
        )
        bank.records.append(rec)
        if warn:
            bank.warnings.append(f"record {k}: {warn}")

    return bank


# ---------------------------------------------------------------------------
# Write  (byte-exact)
# ---------------------------------------------------------------------------
def write_pac(bank: PacBank) -> bytes:
    """Serialize a :class:`PacBank` back to bytes.

    Concatenates every record's verbatim ``raw`` slice (plus any preserved
    ``prefix``). Because parsing never mutates ``raw``, an un-edited bank
    re-serializes byte-for-byte: ``write_pac(parse_pac(x)) == x``.
    """
    if not isinstance(bank, PacBank):
        raise TypeError("write_pac expects a PacBank")
    out = bytearray(bank.prefix)
    for rec in bank.records:
        out += rec.raw
    return bytes(out)


def replace_record_pcm(bank: PacBank, index: int, new_pcm: bytes) -> PacBank:
    """Return a NEW bank with record ``index``'s PCM payload swapped.

    The record header (WAVEFORMATEX + ``data`` tag), the updated 32-bit size
    field, and the original trailing padding are all rebuilt so the result
    re-parses cleanly. Only structured records may be replaced; replacing an
    unstructured record raises ``ValueError`` (mirrors the suite's
    "disable REPLACE for that bank" rule).

    The new PCM is written with the *same* trailing padding length as the
    original record so adjacent records' relative framing is preserved.
    """
    if index < 0 or index >= len(bank.records):
        raise IndexError(f"record index {index} out of range (count={len(bank.records)})")
    rec = bank.records[index]
    if not rec.structured:
        raise ValueError(f"record {index} is not structured; refusing to replace")
    if not isinstance(new_pcm, (bytes, bytearray, memoryview)):
        raise TypeError("new_pcm must be bytes-like")
    new_pcm = bytes(new_pcm)
    frame = PCM_CHANNELS * (PCM_BITS // 8)
    if len(new_pcm) % frame != 0:
        raise ValueError(f"PCM length {len(new_pcm)} is not a multiple of frame size {frame}")

    header = rec.raw[:rec.pcm_offset - 8]      # WFX(16) ... up to the 'data' tag
    # Rebuild the 'data' chunk header with the new size, keep the original pad.
    new_data_hdr = _DATA_TAG + struct.pack("<I", len(new_pcm))
    pad = rec.raw[rec.pcm_offset + rec.pcm_size:]  # exact original trailing bytes
    new_raw = header + new_data_hdr + new_pcm + pad

    new_rec = PacRecord(
        raw=new_raw,
        structured=True,
        pcm_offset=rec.pcm_offset,
        pcm_size=len(new_pcm),
        trailing_pad=len(pad),
    )
    new_records = list(bank.records)
    new_records[index] = new_rec
    return PacBank(records=new_records, prefix=bank.prefix, warnings=list(bank.warnings))


# ---------------------------------------------------------------------------
# PCM <-> WAV  (stdlib wave; no numpy required)
# ---------------------------------------------------------------------------
def pcm_to_wav(pcm: bytes,
               sample_rate: int = PCM_SAMPLE_RATE,
               channels: int = PCM_CHANNELS,
               bits: int = PCM_BITS) -> bytes:
    """Wrap raw little-endian PCM in a canonical RIFF/WAVE container."""
    if bits not in (8, 16):
        raise ValueError(f"unsupported bit depth {bits}")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(bits // 8)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_to_pcm(wav_bytes: bytes) -> Tuple[bytes, int, int, int]:
    """Extract ``(pcm, sample_rate, channels, bits)`` from a WAV container.

    Accepts any PCM WAV; callers that need the PSOBB-native 22050/mono/16
    shape should check the returned tuple and resample/convert (or reject)
    upstream. Raises ``ValueError`` on a non-PCM or unreadable WAV.
    """
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            channels = w.getnchannels()
            width = w.getsampwidth()
            rate = w.getframerate()
            frames = w.readframes(w.getnframes())
    except (wave.Error, EOFError) as e:
        raise ValueError(f"not a readable PCM WAV: {e}")
    return frames, rate, channels, width * 8


def record_to_wav(rec: PacRecord) -> bytes:
    """Render one structured record as a standalone playable WAV.

    Raises ``ValueError`` if the record is unstructured (no clean PCM).
    """
    if not rec.structured:
        raise ValueError("record is not structured; cannot render WAV")
    return pcm_to_wav(rec.pcm, rec.sample_rate, rec.channels, rec.bits)


# ---------------------------------------------------------------------------
# DSP helpers  (numpy)
# ---------------------------------------------------------------------------
def _require_numpy():
    try:
        import numpy as np  # noqa: F401
        return np
    except ImportError as e:  # pragma: no cover - numpy is a project dep
        raise RuntimeError("numpy is required for PCM DSP helpers") from e


def trim_pcm(pcm: bytes,
             start_frame: int = 0,
             end_frame: Optional[int] = None,
             channels: int = PCM_CHANNELS,
             bits: int = PCM_BITS) -> bytes:
    """Trim 16-bit PCM to the half-open frame range ``[start_frame, end_frame)``.

    Frames (not bytes) so the cut is always sample-aligned. ``end_frame=None``
    means "to the end". Out-of-range indices are clamped.
    """
    np = _require_numpy()
    if bits != 16:
        raise ValueError("trim_pcm only supports 16-bit PCM")
    dtype = np.int16
    samples = np.frombuffer(pcm, dtype=dtype)
    if channels > 1:
        samples = samples.reshape(-1, channels)
        n = samples.shape[0]
    else:
        n = samples.shape[0]
    s = max(0, int(start_frame))
    e = n if end_frame is None else min(int(end_frame), n)
    e = max(e, s)
    cut = samples[s:e]
    return cut.tobytes()


def normalize_pcm(pcm: bytes,
                  target_dbfs: float = -1.0,
                  channels: int = PCM_CHANNELS,
                  bits: int = PCM_BITS) -> bytes:
    """Peak-normalize 16-bit PCM so its loudest sample sits at ``target_dbfs``.

    A no-op (returns the input) for silence. ``target_dbfs`` is relative to
    full scale (0 dBFS = peak 32767); the default -1 dBFS leaves a hair of
    headroom. Result is clipped to the int16 range.
    """
    np = _require_numpy()
    if bits != 16:
        raise ValueError("normalize_pcm only supports 16-bit PCM")
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    if samples.size == 0:
        return pcm
    peak = float(np.max(np.abs(samples)))
    if peak <= 0.0:
        return pcm
    target_amp = (10.0 ** (target_dbfs / 20.0)) * 32767.0
    gain = target_amp / peak
    out = np.clip(np.round(samples * gain), -32768, 32767).astype(np.int16)
    return out.tobytes()


# ---------------------------------------------------------------------------
# Waveform (downsampled peaks + rms) for the frontend canvas
# ---------------------------------------------------------------------------
def waveform_peaks(pcm: bytes, buckets: int = 600,
                   channels: int = PCM_CHANNELS,
                   bits: int = PCM_BITS) -> dict:
    """Downsample 16-bit PCM to ``buckets`` (min,max,rms) triples in [-1,1].

    Returns ``{"buckets": N, "min": [...], "max": [...], "rms": [...]}``.
    Designed to feed a small overview canvas without shipping the whole
    sample buffer to the browser.
    """
    np = _require_numpy()
    if bits != 16:
        raise ValueError("waveform_peaks only supports 16-bit PCM")
    buckets = max(1, int(buckets))
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    n = samples.shape[0]
    if n == 0:
        return {"buckets": 0, "min": [], "max": [], "rms": []}
    norm = samples / 32768.0
    edges = np.linspace(0, n, buckets + 1, dtype=np.int64)
    mins: List[float] = []
    maxs: List[float] = []
    rmss: List[float] = []
    for b in range(buckets):
        lo, hi = int(edges[b]), int(edges[b + 1])
        if hi <= lo:
            mins.append(0.0); maxs.append(0.0); rmss.append(0.0)
            continue
        seg = norm[lo:hi]
        mins.append(float(seg.min()))
        maxs.append(float(seg.max()))
        rmss.append(float(np.sqrt(np.mean(seg * seg))))
    return {"buckets": buckets, "min": mins, "max": maxs, "rms": rmss}


# ---------------------------------------------------------------------------
# Summary helper for the /api/audio/info endpoint
# ---------------------------------------------------------------------------
def summarize_bank(bank: PacBank) -> List[dict]:
    """One JSON-serializable dict per record for the info endpoint."""
    out: List[dict] = []
    for i, rec in enumerate(bank.records):
        out.append({
            "index": i,
            "structured": rec.structured,
            "bytes": len(rec.raw),
            "pcm_bytes": rec.pcm_size if rec.structured else 0,
            "sample_rate": PCM_SAMPLE_RATE,
            "channels": PCM_CHANNELS,
            "bits": PCM_BITS,
            "duration_s": round(rec.duration_s, 4) if rec.structured else 0.0,
            "warning": rec.warning,
        })
    return out
