"""A.1 — pure-numpy fbank (Kaldi-compatible) feature extraction."""
import numpy as np

from fpm.embed.fbank import compute_fbank


def _tone(freq: float, n: int = 16_000, sr: int = 16_000) -> np.ndarray:
    t = np.arange(n) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_shape_and_dtype():
    f = compute_fbank(_tone(440))
    # 1s @16k: frame_len=400, shift=160 → 1 + (16000-400)//160 = 98 frames, 80 bins
    assert f.shape == (98, 80)
    assert f.dtype == np.float32
    assert np.isfinite(f).all()


def test_deterministic():
    a = _tone(300)
    assert np.array_equal(compute_fbank(a), compute_fbank(a))


def test_cmn_zero_mean_per_bin():
    f = compute_fbank(_tone(220))
    # CMN subtracts each bin's mean over frames → per-bin mean ≈ 0
    assert np.allclose(f.mean(axis=0), 0.0, atol=1e-3)


def test_short_audio_returns_empty():
    f = compute_fbank(np.zeros(100, dtype=np.float32))
    assert f.shape == (0, 80)


def test_discriminates_different_signals():
    assert not np.allclose(compute_fbank(_tone(200)), compute_fbank(_tone(2000)))
