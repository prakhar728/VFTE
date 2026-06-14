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

import hmac
import logging
import os
import time

from fastapi import Depends, FastAPI, Form, Header, HTTPException, UploadFile

import config
from fpm.audio import AudioDecodeError, decode_to_mono
from fpm.diarize.diarizen_engine import ClipTooLongError, DiariZenDiarizer

log = logging.getLogger("diarize-service")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="DiariZen diarization service", version="1.0")

_TOKEN = os.environ.get("FPM_DIARIZE_TOKEN", "")


def require_token(authorization: str = Header(default="")) -> None:
    """Constant-time bearer-token check against $FPM_DIARIZE_TOKEN."""
    if not _TOKEN:
        raise HTTPException(503, "service token not configured (FPM_DIARIZE_TOKEN unset)")
    prefix = "Bearer "
    presented = authorization[len(prefix):] if authorization.startswith(prefix) else ""
    if not hmac.compare_digest(presented, _TOKEN):
        raise HTTPException(401, "invalid or missing bearer token")


# One diarizer instance reused across requests: the torch model loads lazily on the
# first finish() and stays warm. DiariZen is single-session, so requests are serial —
# fine for a batch box; scale out with replicas if you need concurrency.
_diarizer = DiariZenDiarizer(offline=True)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "engine": "diarizen", "model": _diarizer._model_name}


@app.post("/diarize", dependencies=[Depends(require_token)])
async def diarize(file: UploadFile, workspace: str = Form("")) -> dict:
    """Diarize one mixed recording → anonymous segments.

    Returns `{segments: [{start, end, local_speaker}], sample_rate, duration_sec,
    elapsed_sec}`. `workspace` is accepted for log correlation only — this box keeps
    no per-workspace state.
    """
    try:
        audio = decode_to_mono(await file.read())
    except AudioDecodeError as exc:
        raise HTTPException(400, f"audio decode failed: {exc}")
    sr = config.TARGET_SAMPLE_RATE
    dur = len(audio) / sr if sr else 0.0

    t0 = time.monotonic()
    _diarizer.start(workspace or "remote")
    try:
        _diarizer.feed(audio, sr)
        segs = _diarizer.finish()
    except ClipTooLongError as exc:
        raise HTTPException(413, str(exc))
    elapsed = time.monotonic() - t0
    log.info("diarized ws=%s dur=%.1fs -> %d segs in %.1fs (rtf=%.2f)",
             workspace or "-", dur, len(segs), elapsed, elapsed / dur if dur else 0.0)

    return {
        "segments": [
            {"start": s.start, "end": s.end, "local_speaker": s.local_speaker} for s in segs
        ],
        "sample_rate": sr,
        "duration_sec": round(dur, 3),
        "elapsed_sec": round(elapsed, 3),
    }
