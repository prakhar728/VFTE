"""Standalone DiariZen diarization microservice (the heavy torch box).

This is the *remote* half of the offline diarize path. It runs ONLY DiariZen
(WavLM-EEND + VBx) and returns anonymous `{start, end, local_speaker}` segments
— never an embedding, never a voiceprint id, never text (the engine-independent-
store invariant, see fpm/diarize/base.py). Identity (CAM++ re-embed + store match)
stays in the FPM core process, which calls this service via the `remote` engine
(fpm/diarize/remote_engine.py).

Why a separate service: DiariZen pins torch 2.1.1 and is RAM-heavy, so we isolate
it on its own (optionally cloud / TEE) box and keep the FPM core torch-free. The
box is stateless — it holds no voiceprints, so it can be torn down freely.

Auth: a single bearer token from $FPM_DIARIZE_TOKEN (constant-time compared). No
store, no per-workspace authz here — this box never sees identity.

Run:  FPM_DIARIZE_TOKEN=… uvicorn service:app --host 0.0.0.0 --port 8086
"""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import threading
import time

from fastapi import Depends, FastAPI, Form, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

import config
from fpm.audio import AudioDecodeError, decode_to_mono
from fpm.diarize.diarizen_engine import ClipTooLongError, DiariZenDiarizer

log = logging.getLogger("diarize-service")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="FPM diarization service (diart | diarizen)", version="1.0")

_TOKEN = os.environ.get("FPM_DIARIZE_TOKEN", "")
# Heartbeat cadence while DiariZen runs — keeps the Phala gateway from dropping the
# connection as idle during the (minutes-long) diarize. Must be < the gateway idle
# timeout; 8s is comfortably under typical proxy defaults (30–60s).
_HEARTBEAT_SEC = float(os.environ.get("FPM_DIARIZE_HEARTBEAT_SEC", "8"))


def require_token(authorization: str = Header(default="")) -> None:
    """Constant-time bearer-token check against $FPM_DIARIZE_TOKEN."""
    if not _TOKEN:
        raise HTTPException(503, "service token not configured (FPM_DIARIZE_TOKEN unset)")
    prefix = "Bearer "
    presented = authorization[len(prefix):] if authorization.startswith(prefix) else ""
    if not hmac.compare_digest(presented, _TOKEN):
        raise HTTPException(401, "invalid or missing bearer token")


# Engine is selectable so the SAME service deploys two ways:
#   FPM_DIARIZE_ENGINE=diart    → live/in-person acoustic diarization on CPU (P2)
#   FPM_DIARIZE_ENGINE=diarizen → post-meeting batch re-clustering on GPU (P3, default)
# Both implement the StreamingDiarizer start/feed/finish contract (fpm/diarize/base.py),
# so the batch-style _run_diarize below works for either, and FPM core stays torch-free
# and calls whichever via FPM_DIARIZER=remote.
_ENGINE = os.environ.get("FPM_DIARIZE_ENGINE", "diarizen").lower()


def _make_diarizer():
    if _ENGINE == "diart":
        from fpm.diarize.diart_engine import DiartDiarizer
        return DiartDiarizer(offline=True)
    return DiariZenDiarizer(offline=True)


# One diarizer instance reused across requests: the torch model loads lazily on the
# first finish() and stays warm. Single-session, so a lock serializes concurrent
# requests (e.g. a Conclave retry) — without it the shared buffer corrupts.
_diarizer = _make_diarizer()
_diarize_lock = threading.Lock()


def _run_diarize(audio, sr: int, workspace: str):
    """Blocking diarize, serialized. Runs in a thread off the event loop."""
    with _diarize_lock:
        _diarizer.start(workspace or "remote")
        _diarizer.feed(audio, sr)
        return _diarizer.finish()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "engine": _ENGINE, "model": getattr(_diarizer, "_model_name", None)}


@app.post("/diarize", dependencies=[Depends(require_token)])
async def diarize(file: UploadFile, workspace: str = Form("")):
    """Diarize one mixed recording → anonymous segments, as a heartbeat stream.

    DiariZen is slow (RTF ~3 on CPU) — minutes for a real clip — which outlives the
    Phala gateway's idle timeout on a plain request. So we stream: a `\\n` heartbeat
    every few seconds while DiariZen runs in a worker thread (keeps the gateway
    connection alive), then a single final JSON line:
        {segments:[{start,end,local_speaker}], sample_rate, duration_sec, elapsed_sec}
    or {error, detail} on failure. The client (remote_engine) ignores blank lines
    and parses the last non-blank line.
    """
    try:
        audio = decode_to_mono(await file.read())
    except AudioDecodeError as exc:
        raise HTTPException(400, f"audio decode failed: {exc}")
    sr = config.TARGET_SAMPLE_RATE
    dur = len(audio) / sr if sr else 0.0

    async def stream():
        t0 = time.monotonic()
        fut = asyncio.get_running_loop().run_in_executor(None, _run_diarize, audio, sr, workspace)
        while not fut.done():
            yield b"\n"  # heartbeat — blank line, keeps the connection non-idle
            await asyncio.sleep(_HEARTBEAT_SEC)
        try:
            segs = fut.result()
        except ClipTooLongError as exc:
            yield (json.dumps({"error": "clip_too_long", "detail": str(exc)}) + "\n").encode()
            return
        except Exception as exc:  # noqa: BLE001 — surface as a JSON error line, not a dropped stream
            log.exception("diarize failed")
            yield (json.dumps({"error": "diarize_failed", "detail": str(exc)[:300]}) + "\n").encode()
            return
        elapsed = time.monotonic() - t0
        log.info("diarized ws=%s dur=%.1fs -> %d segs in %.1fs (rtf=%.2f)",
                 workspace or "-", dur, len(segs), elapsed, elapsed / dur if dur else 0.0)
        yield (json.dumps({
            "segments": [
                {"start": s.start, "end": s.end, "local_speaker": s.local_speaker} for s in segs
            ],
            "sample_rate": sr,
            "duration_sec": round(dur, 3),
            "elapsed_sec": round(elapsed, 3),
        }) + "\n").encode()

    return StreamingResponse(stream(), media_type="application/x-ndjson")
