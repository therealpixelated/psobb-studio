"""Thin facade over the PSOBB audio codecs.

Re-exports the pure-Python ``.pac`` PCM-bank codec (``audio_pac``) and the
optional ffmpeg-backed decoders (``audio_codec``) behind one import surface,
plus a couple of container-classification helpers the server uses to route a
filename to the right decode path.

  from formats import audio
  bank = audio.parse_pac(blob)
  wav  = audio.record_to_wav(bank.records[0])
  if audio.ffmpeg_available():
      wav = audio.decode_to_wav(ogg_bytes, "ogg")
"""
from __future__ import annotations

from typing import Optional, Tuple

# --- pure-Python .pac codec (the main feature) -----------------------------
from formats.audio_pac import (  # noqa: F401
    PacBank,
    PacRecord,
    WFX_SIG,
    PCM_SAMPLE_RATE,
    PCM_CHANNELS,
    PCM_BITS,
    parse_pac,
    write_pac,
    replace_record_pcm,
    pcm_to_wav,
    wav_to_pcm,
    record_to_wav,
    trim_pcm,
    normalize_pcm,
    waveform_peaks,
    summarize_bank,
)

# --- optional ffmpeg facade -------------------------------------------------
from formats.audio_codec import (  # noqa: F401
    FfmpegUnavailable,
    FfmpegError,
    ffmpeg_available,
    ffmpeg_path,
    decode_to_wav,
    encode_ogg,
)


# ---------------------------------------------------------------------------
# Container classification — map a filename to (container, codec, decode-kind)
# ---------------------------------------------------------------------------
# decode_kind values:
#   "pac"     -> pure-Python per-record decode (audio_pac)
#   "ogg"     -> browser-native playback (/api/raw) + optional ffmpeg decode
#   "sfd"     -> ffmpeg-only decode (ASF/WMV); no replace
#   "wav"     -> stdlib passthrough
#   None      -> not an audio container we handle
_AUDIO_EXT = {
    ".pac": ("PAC", "PCM (mono/22050/16)", "pac"),
    ".ogg": ("OGG", "Ogg Vorbis", "ogg"),
    ".sfd": ("SFD", "ASF/WMV (WMV3 + WMAv2)", "sfd"),
    ".wav": ("WAV", "PCM", "wav"),
}

# Which containers may be a Replace target. .pac and .ogg only; .sfd is an
# A/V movie and .adx is not present live — both reject Replace (HTTP 400).
_REPLACE_TARGETS = {".pac", ".ogg"}


def classify_audio(filename: str) -> Optional[Tuple[str, str, str]]:
    """Return ``(container, codec, decode_kind)`` for an audio filename, or
    None if the extension is not an audio container this suite handles."""
    name = (filename or "").lower()
    for ext, info in _AUDIO_EXT.items():
        if name.endswith(ext):
            return info
    return None


def is_audio_file(filename: str) -> bool:
    return classify_audio(filename) is not None


def replace_supported(filename: str) -> bool:
    """True iff this filename is a valid Replace target (.pac / .ogg)."""
    name = (filename or "").lower()
    return any(name.endswith(ext) for ext in _REPLACE_TARGETS)
