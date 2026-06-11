"""C1.0 — audio ingestion: decode arbitrary input to 16 kHz mono float32."""
import math
import struct
import wave

import numpy as np
import pytest

from fpm.audio import AudioDecodeError, decode_to_mono


def _write_wav(path, sr: int, channels: int, seconds: float, freq: int = 440) -> None:
    n = int(sr * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)  # int16
        w.setframerate(sr)
        frames = bytearray()
        for i in range(n):
            v = int(0.3 * 32767 * math.sin(2 * math.pi * freq * i / sr))
            frames += struct.pack("<h", v) * channels
        w.writeframes(bytes(frames))


def test_resamples_and_downmixes_to_16k_mono(tmp_path):
    p = tmp_path / "stereo_44k.wav"
    _write_wav(p, sr=44_100, channels=2, seconds=1.0)
    audio = decode_to_mono(p)
    assert audio.dtype == np.float32
    assert audio.ndim == 1                       # mono
    assert abs(len(audio) - 16_000) < 200        # ~1 s at 16 kHz
    assert float(np.max(np.abs(audio))) <= 1.0
    assert audio.flags.writeable


def test_upsamples_from_8k(tmp_path):
    p = tmp_path / "mono_8k.wav"
    _write_wav(p, sr=8_000, channels=1, seconds=0.5)
    audio = decode_to_mono(p)
    assert abs(len(audio) - 8_000) < 200         # 0.5 s at 16 kHz


def test_decode_from_bytes(tmp_path):
    p = tmp_path / "in.wav"
    _write_wav(p, sr=22_050, channels=1, seconds=0.25)
    audio = decode_to_mono(p.read_bytes())
    assert abs(len(audio) - 4_000) < 200         # 0.25 s at 16 kHz


def test_garbage_raises():
    with pytest.raises(AudioDecodeError):
        decode_to_mono(b"this is definitely not audio")


def test_missing_file_raises(tmp_path):
    with pytest.raises(AudioDecodeError):
        decode_to_mono(tmp_path / "nope.wav")
