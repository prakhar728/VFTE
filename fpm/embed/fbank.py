"""Pure-numpy Mel filterbank features (Kaldi-compatible).

Ported from VoxTerm `audio/diarization/fbank.py` (MIT). Replaces
torchaudio.compliance.kaldi.fbank() so the ONNX embedder needs no PyTorch.
Parameters match the Kaldi defaults the 3D-Speaker / WeSpeaker models expect.
No dependency beyond numpy → runs locally, torch-free.
"""
from __future__ import annotations

import numpy as np


def compute_fbank(
    audio: np.ndarray,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    preemph_coeff: float = 0.97,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
    window_type: str = "hamming",
    cmn: bool = True,
) -> np.ndarray:
    """Log-Mel filterbank features matching Kaldi conventions.

    Input:  1-D float32 audio scaled to [-1, 1] (we rescale to 16-bit internally,
            as the torchaudio path does).
    Output: (num_frames, num_mel_bins) float32.
    """
    audio = np.asarray(audio, dtype=np.float32).ravel()
    audio = audio * (1 << 15)  # Kaldi expects 16-bit PCM amplitude

    if high_freq <= 0:
        high_freq = sample_rate / 2.0

    if preemph_coeff > 0:
        audio = np.concatenate([[audio[0]], audio[1:] - preemph_coeff * audio[:-1]])

    frame_length = int(round(frame_length_ms / 1000.0 * sample_rate))
    frame_shift = int(round(frame_shift_ms / 1000.0 * sample_rate))
    n_fft = _next_power_of_2(frame_length)

    if len(audio) < frame_length:
        return np.zeros((0, num_mel_bins), dtype=np.float32)
    num_frames = 1 + (len(audio) - frame_length) // frame_shift

    window = _get_window(window_type, frame_length)
    mel_filters = _mel_filterbank(n_fft, num_mel_bins, sample_rate, low_freq, high_freq)

    features = np.empty((num_frames, num_mel_bins), dtype=np.float32)
    for i in range(num_frames):
        start = i * frame_shift
        frame = (audio[start : start + frame_length] * window).astype(np.float64)
        spectrum = np.fft.rfft(frame, n=n_fft)
        power = spectrum.real ** 2 + spectrum.imag ** 2
        # matmul has no division; numpy 2.x raises a spurious FP "divide" flag on its
        # SIMD path for large values — the result is correct, so silence the false flag.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            mel_energies = np.maximum(mel_filters @ power, 1e-10)
        features[i] = np.log(mel_energies)

    if cmn and num_frames > 0:
        features -= features.mean(axis=0)

    return features


# ── helpers ──────────────────────────────────────────────


def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _get_window(window_type: str, length: int) -> np.ndarray:
    if window_type == "hamming":
        return np.hamming(length).astype(np.float32)
    if window_type == "hanning":
        return np.hanning(length).astype(np.float32)
    if window_type == "povey":
        return (np.hanning(length).astype(np.float32)) ** 0.85
    return np.ones(length, dtype=np.float32)


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(
    n_fft: int, num_mel_bins: int, sample_rate: int, low_freq: float, high_freq: float
) -> np.ndarray:
    """(num_mel_bins, n_fft//2 + 1) triangular mel filterbank."""
    num_fft_bins = n_fft // 2 + 1
    mel_points = np.linspace(_hz_to_mel(low_freq), _hz_to_mel(high_freq), num_mel_bins + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((num_mel_bins, num_fft_bins), dtype=np.float32)
    for m in range(num_mel_bins):
        f_left, f_center, f_right = bin_points[m], bin_points[m + 1], bin_points[m + 2]
        for k in range(f_left, f_center):
            if k < num_fft_bins and f_center != f_left:
                filterbank[m, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if k < num_fft_bins and f_right != f_center:
                filterbank[m, k] = (f_right - k) / (f_right - f_center)

    return filterbank
