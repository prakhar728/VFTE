"""DiariZen diarizer (batch WavLM + VBx) — the production *post* engine behind /v1/diarize.

DiariZen is the accuracy winner in the offline bake-off (~22.8% DER; docs/bakeoff-offline.md).
It is offline/batch — there is no incremental output: we accumulate audio in feed() and run the
whole clip once in finish(). That satisfies the C1 StreamingDiarizer contract, which permits
feed() to return an empty list.

It pins torch 2.1.1, which is INCOMPATIBLE with diart's 2.2.2, so it runs in its OWN venv (see
requirements-diarizen.txt) — never alongside diart in one process. The factory keeps the import
lazy and the heavy torch/diarizen stack is loaded only in finish() (decision D1), so importing
this module and driving feed() stays torch-free; the core service venv is unaffected.

Privacy/offline: the model is loaded with HF offline mode engaged (decision D2) so there is no
network egress at runtime — weights must be pre-cached (scripts/prefetch_diarizen_model.py).
Identity always re-embeds segments with the fixed CAM++ ID layer, so DiariZen's internal
embeddings never enter the voiceprint store (the engine-independent-store invariant, C.4).
"""
from __future__ import annotations

import os
import tempfile
import wave

import numpy as np

import config
from fpm.diarize.base import Segment, StreamingDiarizer

DIARIZEN_MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md"


class ClipTooLongError(RuntimeError):
    """Accumulated audio would exceed the engine's clip cap — rejected before the model loads."""


class DiariZenDiarizer(StreamingDiarizer):
    """Batch DiariZen behind the streaming seam: buffer in feed(), decode the whole clip in finish()."""

    # Capability hint for SessionIdentifier (read via getattr(diarizer, "buffered_batch", False)):
    # this engine emits ALL segments at finish(), so the identify layer must retain the full clip
    # (not a bounded trailing buffer) to re-embed every segment. base.py is intentionally untouched.
    buffered_batch = True

    def __init__(
        self,
        model: str = DIARIZEN_MODEL,
        sample_rate: int = 16_000,
        offline: bool = True,
        max_clip_sec: float | None = None,
    ):
        self._model_name = model
        self._sample_rate = sample_rate
        self._offline = offline
        # cap defaults from config; 0 (or 0.0) disables the guard (unbounded)
        cap = config.DIARIZEN_MAX_CLIP_SEC if max_clip_sec is None else max_clip_sec
        self._max_samples = int(cap * sample_rate) if cap else None
        self._pipeline = None
        self._buf: list[np.ndarray] = []
        self._n_samples = 0
        self._started = False

    # ── model loading (lazy: importing this module / feeding must not require torch) ──

    def _load(self):
        if self._offline:
            # no network egress at runtime — weights come from the local HF cache (pre-fetched)
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        # DiariZen's import path can vary by version — try the known ones.
        try:
            from diarizen.pipelines.inference import DiariZenPipeline
        except ImportError:
            from diarizen import DiariZenPipeline  # type: ignore
        return DiariZenPipeline.from_pretrained(self._model_name)

    # ── StreamingDiarizer contract ─────────────────────────────

    def start(self, workspace_id: str) -> None:
        self._workspace_id = workspace_id
        self._buf = []
        self._n_samples = 0
        self._started = True

    def feed(self, chunk: np.ndarray, sample_rate: int = 16_000) -> list[Segment]:
        if not self._started:
            raise RuntimeError("feed() before start()")
        if sample_rate != self._sample_rate:
            raise ValueError(f"expected {self._sample_rate} Hz, got {sample_rate}")
        block = np.asarray(chunk, dtype=np.float32).ravel()
        self._n_samples += block.size
        if self._max_samples is not None and self._n_samples > self._max_samples:
            # reject before accumulating (and decoding) a clip too large for the box's RAM
            cap_sec = self._max_samples / self._sample_rate
            raise ClipTooLongError(
                f"clip exceeds DIARIZEN_MAX_CLIP_SEC ({cap_sec:.0f}s): "
                f"{self._n_samples / self._sample_rate:.0f}s and counting"
            )
        self._buf.append(block)
        return []  # batch engine — no incremental output

    def finish(self) -> list[Segment]:
        if not self._started:
            return []
        self._started = False
        if not self._buf:
            return []
        if self._pipeline is None:           # lazy load right before use (D1)
            self._pipeline = self._load()
        audio = np.concatenate(self._buf)
        self._buf = []                       # free the clip buffer
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

    # ── helpers ────────────────────────────────────────────────

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
