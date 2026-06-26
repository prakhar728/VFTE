"""The swappable diarizer plug — the offline path's voice-splitter.

A `StreamingDiarizer` consumes small audio chunks and emits, incrementally,
where each *engine-local* speaker is talking: `{start, end, local_speaker}`.
That is ALL it emits — never an embedding, never a voiceprint id, never text.
Identity is a separate layer (C.4): it re-embeds each emitted segment with the
fixed CAM++ ID embedder and matches the store. This separation is the
engine-independent-store invariant — swap diart (C.3) for our own ONNX engine
(E.3) and the voiceprints stay valid, because the diarizer's internals never
touch the stored space.

Contract:
  start(workspace_id)            begin a session (resets per-session state)
  feed(chunk, sr) -> [Segment]   push ~0.5–2 s of audio; return segments
                                 *finalized by this chunk* (may be empty)
  finish()        -> [Segment]   flush any trailing segments at stream end

`local_speaker` is a label local to one session/engine (e.g. "speaker0"); it is
NOT stable across sessions and is NOT a `voiceprint_id`. The identify layer maps
local_speaker → voiceprint_id and locks it once confident.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    """A finalized span of speech attributed to one engine-local speaker."""

    start: float          # seconds from session start
    end: float            # seconds from session start
    local_speaker: str    # engine-local label, e.g. "speaker0" — never a voiceprint_id

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class StreamingDiarizer(ABC):
    """Streaming voice-splitter. Bounded state regardless of stream length."""

    @abstractmethod
    def start(self, workspace_id: str) -> None:
        """Begin a session; reset per-session state."""

    @abstractmethod
    def feed(self, chunk: np.ndarray, sample_rate: int = 16_000) -> list[Segment]:
        """Push an audio chunk; return segments finalized by it (incremental)."""

    @abstractmethod
    def finish(self) -> list[Segment]:
        """End of stream: flush and return any remaining segments."""
