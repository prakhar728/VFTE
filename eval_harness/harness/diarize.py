"""Pluggable diarizer factory — diart now, DiariZen droppable in later.

Both implement FPM's `StreamingDiarizer` interface (`start/feed/finish`), so the pipeline and any
future engine are swap-only. The window (`window_sec`) is the headline eval knob: diart's default
is ~5s; try a large window (e.g. 120s) to test whether more context helps.
"""
from __future__ import annotations

from fpm.diarize.base import Segment, StreamingDiarizer


def make_diarizer(engine: str = "diart", window_sec: float = 5.0,
                  step_sec: float = 0.5) -> StreamingDiarizer:
    """Build a diarizer for the configured engine + window."""
    if engine == "diart":
        from fpm.diarize.diart_engine import DiartDiarizer
        # latency = step (min latency); duration = the configurable window.
        return DiartDiarizer(step=step_sec, latency=step_sec, duration=window_sec, offline=True)
    if engine == "diarizen":
        return _DiariZenStub(window_sec=window_sec)
    raise ValueError(f"unknown diarizer engine '{engine}' (expected 'diart' or 'diarizen')")


class _DiariZenStub(StreamingDiarizer):
    """Placeholder for the DiariZen (batch, WavLM+VBx) engine — not implemented yet.

    DiariZen is the accuracy winner (22.8% DER) but batch + RAM scales with length (parked, see
    docs/bakeoff-offline.md). It's a natural fit for the LARGE-window experiments. To add it:
    implement this behind StreamingDiarizer (accumulate audio in feed(); run DiariZen in finish()).
    """

    def __init__(self, window_sec: float = 120.0):
        self._window_sec = window_sec

    def start(self, workspace_id: str) -> None:
        raise NotImplementedError(
            "DiariZen engine not wired yet — it's a stub. diart is the implemented engine; "
            "set diarizer.engine: diart. (See _DiariZenStub docstring to add DiariZen.)"
        )

    def feed(self, chunk, sample_rate: int = 16_000) -> list[Segment]:
        raise NotImplementedError("DiariZen stub")

    def finish(self) -> list[Segment]:
        raise NotImplementedError("DiariZen stub")
