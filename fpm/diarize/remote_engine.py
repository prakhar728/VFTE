"""Remote DiariZen engine — forwards audio to a standalone diarization service.

Lets the FPM core stay torch-free: instead of importing DiariZen locally, this
buffers the session audio and POSTs it to the diarize microservice
(deploy/diarize-service/), which runs DiariZen and returns anonymous segments.
Identity (CAM++ re-embed against the store) still happens locally in the FPM
process, so the voiceprint store never leaves this box — the remote box only ever
sees raw audio and returns `{start, end, local_speaker}` (engine-independent-store
invariant, base.py).

Like the in-process DiariZen engine this is a *batch* diarizer: it buffers in
feed() and does all the work in finish() (one HTTP round-trip). `buffered_batch`
tells SessionIdentifier to retain the full clip so it can re-embed every returned
segment with CAM++.

Select via:  FPM_DIARIZER=remote  FPM_DIARIZER_URL=https://…  FPM_DIARIZE_TOKEN=…
"""
from __future__ import annotations

import io
import json
import wave

import httpx
import numpy as np

import config
from fpm.diarize.base import Segment, StreamingDiarizer


class RemoteDiariZenDiarizer(StreamingDiarizer):
    """Buffers session audio, then POSTs it to the remote diarize service in finish()."""

    # Same capability hint as the in-process DiariZen engine: emits ALL segments at
    # finish(), so the identify layer must retain the full clip to re-embed each one.
    buffered_batch = True

    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        sample_rate: int = 16_000,
        timeout: float | None = None,
    ):
        self._url = (url or config.DIARIZER_REMOTE_URL).rstrip("/")
        self._token = token if token is not None else config.DIARIZER_REMOTE_TOKEN
        self._sample_rate = sample_rate
        self._timeout = timeout if timeout is not None else config.DIARIZER_REMOTE_TIMEOUT
        self._buf: list[np.ndarray] = []
        self._started = False
        self._workspace_id = ""

    # ── StreamingDiarizer contract ─────────────────────────────

    def start(self, workspace_id: str) -> None:
        self._workspace_id = workspace_id
        self._buf = []
        self._started = True

    def feed(self, chunk: np.ndarray, sample_rate: int = 16_000) -> list[Segment]:
        if not self._started:
            raise RuntimeError("feed() before start()")
        if sample_rate != self._sample_rate:
            raise ValueError(f"expected {self._sample_rate} Hz, got {sample_rate}")
        self._buf.append(np.asarray(chunk, dtype=np.float32).ravel())
        return []  # batch engine — no incremental output

    def finish(self) -> list[Segment]:
        if not self._started:
            return []
        self._started = False
        if not self._buf:
            return []
        audio = np.concatenate(self._buf)
        self._buf = []
        wav = self._to_wav_bytes(audio)
        # The service heartbeats (blank lines) while DiariZen runs, then sends one
        # final JSON line — so we stream and keep the last non-blank line. The
        # heartbeats keep the gateway connection alive past its idle timeout.
        last = ""
        with httpx.stream(
            "POST",
            f"{self._url}/diarize",
            headers={"Authorization": f"Bearer {self._token}"},
            files={"file": ("clip.wav", wav, "audio/wav")},
            data={"workspace": self._workspace_id},
            timeout=self._timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.strip():
                    last = line
        if not last:
            raise RuntimeError("diarize service returned no result")
        payload = json.loads(last)
        if payload.get("error"):
            raise RuntimeError(f"diarize service: {payload.get('detail') or payload['error']}")
        segs = [
            Segment(float(s["start"]), float(s["end"]), str(s["local_speaker"]))
            for s in payload.get("segments", [])
        ]
        segs.sort(key=lambda s: (s.start, s.local_speaker))
        return segs

    # ── helpers ────────────────────────────────────────────────

    def _to_wav_bytes(self, audio: np.ndarray) -> bytes:
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self._sample_rate)
            w.writeframes(pcm16.tobytes())
        return buf.getvalue()
