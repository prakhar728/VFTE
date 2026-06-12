"""Offline pipeline: audio → (Whisper ASR + diarize) → merge → attributed transcript + timings.

Reuses FPM internals: `fpm.audio.decode_to_mono` (ffmpeg → 16k mono), the diarizer factory, and
the merge. ASR and diarization run on the SAME decoded audio (shared clock). Vocab-on always runs;
vocab-off runs too when `asr.vocab_compare` so the WER delta is measurable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fpm.audio import decode_to_mono

from .asr import AsrResult, WhisperASR
from .config import ExperimentConfig
from .diarize import make_diarizer
from .merge import Turn, merge

SR = 16_000


@dataclass
class PipelineResult:
    turns: list[Turn]                      # attributed transcript (vocab-on)
    turns_vocab_off: list[Turn] | None
    asr: AsrResult
    asr_vocab_off: AsrResult | None
    speaker_segments: list                 # raw diarizer Segments
    audio_len_sec: float
    asr_sec: float
    diarize_sec: float


def load_audio(path: str | Path) -> np.ndarray:
    """Decode any audio file → 16k mono float32 (same front-end FPM uses)."""
    return decode_to_mono(Path(path).read_bytes())


def run_offline(cfg: ExperimentConfig, audio: np.ndarray) -> PipelineResult:
    audio_len = len(audio) / SR

    # ── ASR (exact Recato Whisper); vocab-on + (optionally) vocab-off ──
    asr = WhisperASR(model=cfg.asr.model, device=cfg.asr.device, compute_type=cfg.asr.compute_type)
    t0 = time.perf_counter()
    res_on = asr.transcribe(audio, vocab=cfg.vocab_or_none())
    res_off = asr.transcribe(audio, vocab=None) if cfg.asr.vocab_compare else None
    asr_sec = time.perf_counter() - t0

    # ── Diarize (fed in step-sized chunks; window is the engine's internal config) ──
    diar = make_diarizer(cfg.diarizer.engine, cfg.diarizer.window_sec, cfg.diarizer.step_sec)
    diar.start(cfg.name)
    segs = []
    step = max(1, int(cfg.diarizer.step_sec * SR))
    t1 = time.perf_counter()
    for i in range(0, len(audio), step):
        segs.extend(diar.feed(audio[i:i + step], SR))
    segs.extend(diar.finish())
    diarize_sec = time.perf_counter() - t1

    # ── Merge transcript ↔ speakers by timestamp ──
    turns_on = merge(res_on.words, segs)
    turns_off = merge(res_off.words, segs) if res_off else None

    return PipelineResult(
        turns=turns_on, turns_vocab_off=turns_off,
        asr=res_on, asr_vocab_off=res_off, speaker_segments=segs,
        audio_len_sec=audio_len, asr_sec=asr_sec, diarize_sec=diarize_sec,
    )
