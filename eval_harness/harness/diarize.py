"""Pluggable diarizer factory — diart now, DiariZen droppable in later.

Both implement FPM's `StreamingDiarizer` interface (`start/feed/finish`), so the pipeline and any
future engine are swap-only. The window (`window_sec`) is the headline eval knob: diart's default
is ~5s; try a large window (e.g. 120s) to test whether more context helps.
"""
from __future__ import annotations

from fpm.diarize.base import StreamingDiarizer


def make_diarizer(engine: str = "diart", window_sec: float = 5.0,
                  step_sec: float = 0.5) -> StreamingDiarizer:
    """Build a diarizer for the configured engine + window."""
    if engine == "diart":
        from fpm.diarize.diart_engine import DiartDiarizer
        # latency = step (min latency); duration = the configurable window.
        return DiartDiarizer(step=step_sec, latency=step_sec, duration=window_sec, offline=True)
    if engine == "diarizen":
        # Lazy import — DiariZen lives in its own venv (torch 2.1.1) and isn't importable in the
        # diart venv. Importing here keeps the diart path clean.
        from eval_harness.harness.diarizen_engine import DiariZenDiarizer
        return DiariZenDiarizer()
    raise ValueError(f"unknown diarizer engine '{engine}' (expected 'diart' or 'diarizen')")
