"""DiariZen diarizer (batch WavLM + VBx) behind the StreamingDiarizer interface — EVAL ONLY.

DiariZen is the accuracy winner in the bake-off (22.8% DER) but it's offline/batch and its RAM
scales with audio length (docs/bakeoff-offline.md). It pins torch 2.1.1 — incompatible with diart's
2.2.2 — so DiariZen experiments run in their OWN venv (/tmp/diarizen-venv). We accumulate audio in
feed() and run the whole clip in finish(); batch-at-finish satisfies the streaming contract.
"""
from __future__ import annotations

import os
import tempfile
import wave

import numpy as np

from fpm.diarize.base import Segment, StreamingDiarizer

DIARIZEN_MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md"


class DiariZenDiarizer(StreamingDiarizer):
    def __init__(self, model: str = DIARIZEN_MODEL, sample_rate: int = 16_000):
        self._model_name = model
        self._sample_rate = sample_rate
        self._pipeline = None
        self._buf: list[np.ndarray] = []

    def _load(self):
        # DiariZen import path can vary by version — try the known ones.
        try:
            from diarizen.pipelines.inference import DiariZenPipeline
        except ImportError:
            from diarizen import DiariZenPipeline  # type: ignore
        return DiariZenPipeline.from_pretrained(self._model_name)

    def start(self, workspace_id: str) -> None:
        if self._pipeline is None:
            self._pipeline = self._load()
        self._buf = []

    def feed(self, chunk, sample_rate: int = 16_000) -> list[Segment]:
        self._buf.append(np.asarray(chunk, dtype=np.float32).ravel())
        return []  # batch engine — no incremental output

    def finish(self) -> list[Segment]:
        if not self._buf:
            return []
        audio = np.concatenate(self._buf)
        path = self._write_wav(audio)
        try:
            annotation = self._pipeline(path)
        finally:
            os.unlink(path)
        segs = [
            Segment(float(seg.start), float(seg.end), str(label))
            for seg, _, label in annotation.itertracks(yield_label=True)
        ]
        segs.sort(key=lambda s: (s.start, s.local_speaker))
        return segs

    def _write_wav(self, audio: np.ndarray) -> str:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self._sample_rate)
            w.writeframes(pcm16.tobytes())
        return path
