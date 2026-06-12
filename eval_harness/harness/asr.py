"""Whisper ASR — replicates Recato's transcription-service EXACTLY (faster-whisper).

Source of truth: Recato/services/transcription-service/main.py — model build (209-241) and
`model.transcribe(...)` params (429-454). Same model (large-v3-turbo), same compute_type (int8),
same decoding/VAD params. Vocab biasing is the `initial_prompt` knob: pass the vocab string for
the vocab-ON pass, None for vocab-OFF.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# EXACT Recato transcribe() params (transcription-service/main.py:429-454).
_TRANSCRIBE_PARAMS = dict(
    task="transcribe",
    temperature=0.0,
    beam_size=5,
    best_of=5,
    compression_ratio_threshold=1.8,
    log_prob_threshold=-1.0,
    no_speech_threshold=0.6,
    condition_on_previous_text=False,
    repetition_penalty=1.1,
    no_repeat_ngram_size=3,
    vad_filter=True,
    vad_parameters=dict(threshold=0.5, min_silence_duration_ms=160, max_speech_duration_s=15.0),
    word_timestamps=True,
)


@dataclass
class Word:
    word: str
    start: float
    end: float
    probability: float


@dataclass
class AsrResult:
    text: str
    language: str
    words: list[Word]        # flat, time-ordered — drives the merge with speaker spans
    segments: list[dict]     # {start, end, text}


class WhisperASR:
    """Wraps faster-whisper with Recato's exact config. Lazy import keeps the dep optional."""

    def __init__(self, model: str = "large-v3-turbo", device: str = "cpu",
                 compute_type: str = "int8", download_root: str | None = None):
        from faster_whisper import WhisperModel

        self.model_name = model
        self._model = WhisperModel(
            model_size_or_path=model, device=device, compute_type=compute_type,
            download_root=download_root,
        )

    def transcribe(self, audio: np.ndarray, vocab: str | None = None,
                   language: str | None = None) -> AsrResult:
        """1-D float32 audio @16k → AsrResult. `vocab` → initial_prompt (None = vocab off)."""
        segments, info = self._model.transcribe(
            np.asarray(audio, dtype=np.float32),
            language=language,
            initial_prompt=vocab,
            **_TRANSCRIBE_PARAMS,
        )
        words: list[Word] = []
        segs: list[dict] = []
        texts: list[str] = []
        for s in segments:                       # generator — iterating runs the decode
            segs.append({"start": s.start, "end": s.end, "text": s.text})
            texts.append(s.text)
            for w in (s.words or []):
                words.append(Word(w.word, w.start, w.end, w.probability))
        return AsrResult(text="".join(texts).strip(), language=info.language,
                         words=words, segments=segs)
