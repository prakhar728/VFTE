"""Audio ingestion (C1.0): decode arbitrary uploads to 16 kHz mono float32.

A single robust path via ffmpeg (must be on PATH): any container/codec ffmpeg
supports — wav/flac/mp3/m4a/aac/opus/ogg, mono or multichannel, any sample rate —
is decoded with resample + downmix handled by ffmpeg itself. Returns a writable
1-D float32 array in [-1, 1] at the target rate.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from config import TARGET_SAMPLE_RATE


class AudioDecodeError(RuntimeError):
    """Raised when input cannot be decoded to audio."""


def decode_to_mono(source: str | Path | bytes | bytearray, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Decode a file path or in-memory bytes to mono float32 at `target_sr`."""
    if isinstance(source, (bytes, bytearray)):
        # m4a/mp4 need seekable input, so stage bytes to a temp file rather than stdin.
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(source)
            tmp_path = tmp.name
        try:
            return _ffmpeg_decode(tmp_path, target_sr)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    path = Path(source)
    if not path.exists():
        raise AudioDecodeError(f"no such file: {path}")
    return _ffmpeg_decode(str(path), target_sr)


def _ffmpeg_decode(path: str, target_sr: int) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-i", path,
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(target_sr),
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True)
    except FileNotFoundError as exc:
        raise AudioDecodeError("ffmpeg not found on PATH") from exc

    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise AudioDecodeError(detail[-400:] or "ffmpeg failed to decode input")

    audio = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32, copy=True)
    if audio.size == 0:
        raise AudioDecodeError("decoded to empty audio")
    return audio
