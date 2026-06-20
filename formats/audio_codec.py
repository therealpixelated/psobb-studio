"""Optional ffmpeg-backed audio decode/encode for formats Python can't crack
in-process (Ogg Vorbis, ASF/WMV ``.sfd``).

Design contract:
  * ffmpeg is **optional**. If the binary is absent, every decode/encode call
    raises :class:`FfmpegUnavailable` (which the server maps to HTTP 501, never
    500) — it must NEVER crash the process or surface a 500.
  * No new Python dependency: this shells out to a system ``ffmpeg`` on PATH
    (``shutil.which``). No PyAV / soundfile / pydub.
  * There is NO ADX support anywhere (PSOBB ships zero live ``.adx`` — those
    SFX were transcoded to ``.ogg``).

Supported kinds:
  * ``"ogg"`` -> decode to 16-bit PCM WAV; encode PCM/WAV -> Ogg Vorbis.
  * ``"sfd"`` -> decode the ASF/WMV intro movie's audio track to WAV.
    (``opening_j.sfd`` is ASF + WMV3 video + WMAv2 audio — NOT CRI Sofdec/ADX.)
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Optional


class FfmpegUnavailable(RuntimeError):
    """Raised when an ffmpeg-dependent operation is requested but ffmpeg is
    not installed. The server maps this to HTTP 501 Not Implemented."""


class FfmpegError(RuntimeError):
    """Raised when ffmpeg is present but the transcode itself failed."""


_FFMPEG_TIMEOUT = 120  # seconds; generous for the 23 MB intro movie


def ffmpeg_path() -> Optional[str]:
    """Absolute path to a usable ``ffmpeg`` binary, or None if not on PATH."""
    return shutil.which("ffmpeg")


def ffmpeg_available() -> bool:
    """True iff an ffmpeg binary is discoverable on PATH."""
    return ffmpeg_path() is not None


def _run_ffmpeg(args: list, input_bytes: bytes) -> bytes:
    """Run ffmpeg reading stdin and writing stdout; return stdout bytes.

    ``args`` is the middle of the command line (between the leading
    ``ffmpeg -i -`` style input and the trailing ``-`` stdout sink) — the
    caller supplies the full arg vector after the binary. Raises
    :class:`FfmpegUnavailable` if absent, :class:`FfmpegError` on failure.
    """
    exe = ffmpeg_path()
    if exe is None:
        raise FfmpegUnavailable("ffmpeg not found on PATH")
    cmd = [exe] + args
    try:
        proc = subprocess.run(
            cmd,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_FFMPEG_TIMEOUT,
        )
    except subprocess.TimeoutExpired as e:
        raise FfmpegError(f"ffmpeg timed out after {_FFMPEG_TIMEOUT}s") from e
    except OSError as e:
        # e.g. the binary vanished between which() and exec, or a perms error.
        raise FfmpegUnavailable(f"ffmpeg could not be executed: {e}") from e
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace")[-600:]
        raise FfmpegError(f"ffmpeg failed (rc={proc.returncode}): {tail}")
    return proc.stdout


# Map our 'kind' tokens to the ffmpeg input demuxer hint. ffmpeg can usually
# autodetect, but pinning the format makes piped stdin reliable.
_DEMUX_HINT = {
    "ogg": "ogg",
    "sfd": "asf",   # opening_j.sfd is an ASF/WMV container
}


def decode_to_wav(data: bytes, kind: str,
                  sample_rate: Optional[int] = None,
                  channels: Optional[int] = None,
                  max_seconds: Optional[float] = None) -> bytes:
    """Decode ``data`` of the given ``kind`` to a 16-bit PCM WAV via ffmpeg.

    ``kind`` is one of ``"ogg"`` / ``"sfd"``. For ``"sfd"`` only the audio
    track is extracted (``-vn``). ``max_seconds`` caps the output duration
    (useful so a long intro-movie audio track stays under the raw response
    cap). Returns WAV bytes.

    Raises:
      ValueError            - unknown kind.
      FfmpegUnavailable     - ffmpeg not installed (-> HTTP 501).
      FfmpegError           - ffmpeg present but failed.
    """
    k = (kind or "").lower()
    if k not in _DEMUX_HINT:
        raise ValueError(f"unsupported decode kind: {kind!r}")
    args = ["-hide_banner", "-loglevel", "error",
            "-f", _DEMUX_HINT[k], "-i", "pipe:0"]
    if k == "sfd":
        args += ["-vn"]  # drop the WMV3 video stream; audio only
    if max_seconds and max_seconds > 0:
        args += ["-t", f"{float(max_seconds):.3f}"]
    args += ["-f", "wav", "-acodec", "pcm_s16le"]
    if sample_rate:
        args += ["-ar", str(int(sample_rate))]
    if channels:
        args += ["-ac", str(int(channels))]
    args += ["pipe:1"]
    return _run_ffmpeg(args, data)


def encode_ogg(data: bytes, in_kind: str = "wav",
               quality: int = 5) -> bytes:
    """Encode PCM/WAV ``data`` to Ogg Vorbis via ffmpeg.

    ``in_kind`` is the input container hint (``"wav"`` by default; ``"pcm"``
    is also accepted for headerless 22050/mono/16 PCM). ``quality`` is the
    libvorbis ``-q:a`` scale (0..10). Returns Ogg bytes.

    Raises FfmpegUnavailable (-> 501) / FfmpegError as ``decode_to_wav`` does.
    """
    ik = (in_kind or "wav").lower()
    args = ["-hide_banner", "-loglevel", "error"]
    if ik == "pcm":
        # Headerless PSOBB-native PCM: tell ffmpeg the framing explicitly.
        args += ["-f", "s16le", "-ar", "22050", "-ac", "1", "-i", "pipe:0"]
    else:
        args += ["-f", "wav", "-i", "pipe:0"]
    args += ["-f", "ogg", "-c:a", "libvorbis", "-q:a", str(int(quality)), "pipe:1"]
    return _run_ffmpeg(args, data)
