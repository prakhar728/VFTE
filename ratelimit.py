"""Per-caller fixed-window write rate limiting (D.5).

In-process and lock-guarded — FPM runs single-process in the TEE, so a shared
in-memory counter is sufficient (no Redis). Keyed by token, so one noisy caller
can't exhaust another's budget. Writes (enroll / diarize / knowledge) are the
protected ops; reads (health, vocab) are not limited here.
"""
from __future__ import annotations

import threading
import time


class RateLimiter:
    def __init__(self, max_writes: int, window_sec: float):
        self.max = max_writes
        self.window = window_sec
        self._hits: dict[str, tuple[float, int]] = {}  # key → (window_start, count)
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            start, count = self._hits.get(key, (now, 0))
            if now - start >= self.window:        # window elapsed → reset
                start, count = now, 0
            count += 1
            self._hits[key] = (start, count)
            return count <= self.max
