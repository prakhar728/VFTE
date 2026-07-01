"""Identify pre-diarized spans — VFTE's half once diarization moves to capture (migration P5).

In the new boundary, capture diarizes a recording into `{start, end, local_speaker}` spans; VFTE only
puts identity on them. This is the inverse of the old fused `/v1/diarize`: no diarizer runs here, the
caller supplies the spans.

It reuses `SessionIdentifier` UNCHANGED by feeding it a `SpanReplayDiarizer` — a StreamingDiarizer that
runs no model and simply replays the caller's spans at finish(). Marked `buffered_batch=True` so the
identifier retains the whole clip (like DiariZen's post pass) and can re-embed a span anywhere in a long
meeting. So the exact same vote-lock / mint-anonymous / retro-relabel identity logic runs — only the
source of the spans changed (capture instead of an in-process diarizer).

The Segment/StreamingDiarizer seam lives in `fpm/types.py` (the diarizer engines were removed in the P5
strip; only the contract the identifier speaks remains).
"""
from __future__ import annotations

from .identify import IdentifiedSegment, SessionIdentifier
from .types import Segment, StreamingDiarizer


class SpanReplayDiarizer(StreamingDiarizer):
    """A no-model 'diarizer' that replays caller-supplied spans — the bridge into SessionIdentifier.

    feed() emits nothing (audio is just buffered by the identifier); finish() returns all spans at once,
    so identity re-embeds each against the full retained clip. `buffered_batch=True` tells the identifier
    to keep the whole recording (not a bounded trailing window), so spans late in a long meeting still
    slice correctly.
    """

    buffered_batch = True

    def __init__(self, spans: list[Segment]):
        self._spans = sorted(spans, key=lambda s: (s.start, s.local_speaker))
        self._started = False

    def start(self, workspace_id: str) -> None:
        self._started = True

    def feed(self, chunk, sample_rate: int = 16_000) -> list[Segment]:
        if not self._started:
            raise RuntimeError("feed() before start()")
        return []  # batch: nothing emitted incrementally

    def finish(self) -> list[Segment]:
        if not self._started:
            return []
        self._started = False
        return list(self._spans)


def identified_dict(s: IdentifiedSegment) -> dict:
    """Serialize an IdentifiedSegment to the C2 wire shape (same fields the old /v1/diarize emitted)."""
    return {
        "start": round(s.start, 3),
        "end": round(s.end, 3),
        "local_speaker": s.local_speaker,
        "voiceprint_id": s.voiceprint_id,
        "name": s.name,
        "decision": s.decision,
        "confidence": round(s.confidence, 4),
    }


def identify_spans(audio, workspace, spans, *, store, embedder, sample_rate=16_000,
                   read_only=False, consumer="identify", meeting_id=None) -> list[IdentifiedSegment]:
    """Put identity on caller-supplied diarization spans → identified segments (vote-locked, relabeled).

    `spans` is an iterable of dicts `{start, end, local_speaker}` (or Segments). Returns the corrected
    session transcript (SessionIdentifier.transcript()). `read_only=False` is the authoritative writer
    (mints/updates voiceprints), matching the old `tag=offline` post pass.
    """
    segs = [s if isinstance(s, Segment)
            else Segment(float(s["start"]), float(s["end"]), str(s["local_speaker"]))
            for s in spans]
    diarizer = SpanReplayDiarizer(segs)
    ident = SessionIdentifier(store, embedder, diarizer, workspace,
                              sample_rate=sample_rate, consumer=consumer, read_only=read_only,
                              meeting_id=meeting_id)
    ident.start()
    ident.feed(audio, sample_rate)
    ident.finish()
    return ident.transcript()
