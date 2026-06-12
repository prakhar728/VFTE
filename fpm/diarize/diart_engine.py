"""diart adapter — implements StreamingDiarizer by driving diart in PULL mode.

The C.2 spike cleared diart as real-time on CPU (RTF ~0.28, ~150 ms/chunk,
<900 MB). This wraps diart's online `SpeakerDiarization` behind our `feed()`
contract so it's a drop-in plug — swap it later for the lean ONNX engine (E.3)
without touching the store.

How it works: diart's native API is a reactive (push) rx stream owned by
`StreamingInference`. We don't use that loop. Instead we build diart's *exact*
operator chain (`rearrange_audio_stream` → pipeline) on an rx `Subject` we push
into, so we inherit diart's windowing/latency semantics verbatim and never
reimplement its sliding-window math. The rx chain runs synchronously, so after
`subject.on_next(chunk)` any emitted annotations have already been collected —
which is exactly the pull semantics `feed()` needs.

diart emits ~0.5 s labelled fragments (session-stable labels). We stitch
contiguous same-label fragments into spans long enough for the fixed CAM++ ID
embedder (≥1 s) and split a long monologue at `MAX_SPAN_SEC` so identity can
refresh. A span is finalized once the stream's confirmed time has moved past its
end by `MERGE_GAP_SEC` (the speaker has gone quiet) — deferred close, matching
diart's own latency profile.

Privacy: models load from the local HF cache / baked dir with `HF_HUB_OFFLINE=1`
set at construction — no network egress at runtime. diart's embedder is its
INTERNAL scissors only; identity always re-embeds with our CAM++ (C.4).
"""
from __future__ import annotations

import os

import numpy as np

from .base import Segment, StreamingDiarizer

# diart's default segmentation is gated pyannote/segmentation-3.0; embedding is the
# ungated wespeaker model (pyannote loader — avoids the speechbrain use_auth_token
# trap on pyannote.audio 3.4). Both are diart-internal; the store never sees them.
DEFAULT_SEGMENTATION = "pyannote/segmentation-3.0"
DEFAULT_EMBEDDING = "pyannote/wespeaker-voxceleb-resnet34-LM"

MERGE_GAP_SEC = 0.6     # a same-label gap larger than this closes a span (> diart step 0.5)
MAX_SPAN_SEC = 10.0     # chop a long turn so identity can refresh on fresh audio


class _OpenSpan:
    __slots__ = ("start", "end")

    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class DiartDiarizer(StreamingDiarizer):
    def __init__(
        self,
        segmentation: str = DEFAULT_SEGMENTATION,
        embedding: str = DEFAULT_EMBEDDING,
        step: float = 0.5,
        latency: float = 0.5,
        duration: float | None = None,   # diart window (None = diart default ~5s); eval knob
        sample_rate: int = 16_000,
        hf_token: str | None = None,
        offline: bool = True,
    ):
        self._segmentation = segmentation
        self._embedding = embedding
        self._step = step
        self._latency = latency
        self._duration = duration
        self._sample_rate = sample_rate
        self._hf_token = hf_token or os.environ.get("HF_TOKEN")
        self._offline = offline

        self._pipeline = None
        self._subject = None
        self._raw: list[tuple[float, float, str]] = []  # collected (start,end,label) this feed
        self._open: dict[str, _OpenSpan] = {}
        self._t_final = 0.0
        self._started = False

    # ── model loading (lazy: importing this module must not require torch) ──

    def _build_pipeline(self):
        if self._offline:
            # no network egress at runtime — models come from local HF cache / baked dir
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from diart import SpeakerDiarization, SpeakerDiarizationConfig
        from diart import models as dm

        seg = dm.SegmentationModel.from_pretrained(self._segmentation, use_hf_token=self._hf_token)
        emb = dm.EmbeddingModel.from_pretrained(self._embedding, use_hf_token=self._hf_token)
        cfg_kwargs = dict(
            segmentation=seg, embedding=emb,
            step=self._step, latency=self._latency,
            device=torch.device("cpu"),
        )
        if self._duration is not None:
            cfg_kwargs["duration"] = self._duration   # override diart's default window
        return SpeakerDiarization(SpeakerDiarizationConfig(**cfg_kwargs))

    def _build_stream(self):
        import rx
        import rx.operators as ops
        from rx.subject import Subject
        from diart import operators as dops

        self._subject = Subject()
        cfg = self._pipeline.config
        stream = self._subject.pipe(
            dops.rearrange_audio_stream(cfg.duration, cfg.step, self._sample_rate),
            ops.buffer_with_count(count=1),
            ops.map(self._pipeline),
            ops.flat_map(lambda results: rx.from_iterable(results)),
        )
        stream.subscribe(on_next=self._collect)

    def _collect(self, result) -> None:
        ann = result[0]
        for seg, _, label in ann.itertracks(yield_label=True):
            self._raw.append((float(seg.start), float(seg.end), str(label)))

    # ── StreamingDiarizer contract ─────────────────────────────

    def start(self, workspace_id: str) -> None:
        self._workspace_id = workspace_id
        if self._pipeline is None:
            self._pipeline = self._build_pipeline()
        else:
            self._pipeline.reset()
        self._build_stream()
        self._raw.clear()
        self._open.clear()
        self._t_final = 0.0
        self._started = True

    def feed(self, chunk: np.ndarray, sample_rate: int = 16_000) -> list[Segment]:
        if not self._started:
            raise RuntimeError("feed() before start()")
        if sample_rate != self._sample_rate:
            raise ValueError(f"expected {self._sample_rate} Hz, got {sample_rate}")
        block = np.asarray(chunk, dtype=np.float32).ravel()
        if block.size == 0:
            return []
        self._raw.clear()
        self._subject.on_next(block.reshape(1, -1))   # synchronous → _collect already ran
        return self._ingest_and_finalize(close_all=False)

    def finish(self) -> list[Segment]:
        if not self._started:
            return []
        self._raw.clear()
        if self._subject is not None:
            self._subject.on_completed()
        self._started = False
        return self._ingest_and_finalize(close_all=True)

    # ── span stitching ─────────────────────────────────────────

    def _ingest_and_finalize(self, close_all: bool) -> list[Segment]:
        finalized: list[Segment] = []
        for rs, re, label in sorted(self._raw, key=lambda r: (r[0], r[1])):
            self._t_final = max(self._t_final, re)
            span = self._open.get(label)
            if span is not None and rs <= span.end + MERGE_GAP_SEC:
                span.end = max(span.end, re)
                if span.end - span.start >= MAX_SPAN_SEC:        # chop long turns
                    finalized.append(Segment(span.start, span.end, label))
                    del self._open[label]
            else:
                if span is not None:                              # discontinuity → close old
                    finalized.append(Segment(span.start, span.end, label))
                self._open[label] = _OpenSpan(rs, re)

        # deferred close: a speaker that's gone quiet past the merge gap is finalized
        for label in list(self._open):
            span = self._open[label]
            if close_all or self._t_final > span.end + MERGE_GAP_SEC:
                finalized.append(Segment(span.start, span.end, label))
                del self._open[label]

        finalized.sort(key=lambda s: (s.start, s.local_speaker))
        return finalized
