"""A deterministic diarizer for tests — drives the identify pipeline (C.4) without diart.

Constructed with a fixed script of segments aligned to a known audio clip. As
audio is fed, it emits each scripted segment when the cumulative stream time
crosses that segment's `end` — modelling incremental finalization. Carries only
the bounded script as state, so it also exercises the bounded-memory contract.
"""
from __future__ import annotations

import numpy as np

from fpm.types import Segment, StreamingDiarizer


class MockDiarizer(StreamingDiarizer):
    def __init__(self, script: list[Segment]):
        self._script = sorted(script, key=lambda s: s.end)
        self._elapsed = 0.0
        self._emitted = 0
        self._started = False

    def start(self, workspace_id: str) -> None:
        self._workspace_id = workspace_id
        self._elapsed = 0.0
        self._emitted = 0
        self._started = True

    def feed(self, chunk: np.ndarray, sample_rate: int = 16_000) -> list[Segment]:
        if not self._started:
            raise RuntimeError("feed() before start()")
        self._elapsed += len(np.asarray(chunk).ravel()) / sample_rate
        out: list[Segment] = []
        while self._emitted < len(self._script) and self._script[self._emitted].end <= self._elapsed + 1e-6:
            out.append(self._script[self._emitted])
            self._emitted += 1
        return out

    def finish(self) -> list[Segment]:
        out = self._script[self._emitted:]
        self._emitted = len(self._script)
        return list(out)
