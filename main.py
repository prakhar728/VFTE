"""FPM — Speaker Fingerprinting Microservice: FastAPI entrypoint.

Endpoints land per milestone (v1 endpoints get scoped auth in D):
    ✅ GET  /health
    ✅ POST /v1/enroll        (gmeet path — labeled audio → voiceprint)   [B]
       POST /v1/diarize       (offline path — streaming diarize+identify)  [C/D]
       GET  /v1/vocab/{ws}    · POST /v1/knowledge                         [D]

Store + embedder are loaded once at startup (and may be pre-set on `app.state`
for tests). All inference is local; nothing leaves the process.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile

import config
from auth import Caller, TokenAuth, require_scope
from fpm.audio import AudioDecodeError, decode_to_mono
from fpm.embed.onnx_embedder import OnnxSpeakerEmbedder
from fpm.enroll import enroll
from fpm.store.store import VoiceprintStore

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Respect pre-set state (tests inject a tmp store / embedder / auth).
    if not getattr(app.state, "auth", None):
        app.state.auth = TokenAuth.from_env()
    if not getattr(app.state, "store", None):
        app.state.store = VoiceprintStore().open()
    if not getattr(app.state, "embedder", None):
        app.state.embedder = (
            OnnxSpeakerEmbedder(config.ID_EMBEDDER_PATH).load()
            if config.ID_EMBEDDER_PATH.exists()
            else None
        )
        if app.state.embedder is None:
            log.warning("ID embedder model missing (%s) — enroll/diarize disabled", config.ID_EMBEDDER_PATH)
    yield
    if getattr(app.state, "store", None):
        app.state.store.close()


app = FastAPI(title=config.SERVICE_NAME, version=config.SERVICE_VERSION, lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": config.SERVICE_NAME, "version": config.SERVICE_VERSION}


@app.post("/v1/enroll")
async def enroll_endpoint(
    request: Request,
    file: UploadFile,
    identity: str = Form(...),
    workspace: str = Form(...),
    caller: Caller = Depends(require_scope("enroll")),
) -> dict:
    """gmeet path: a clip already attributed to `identity` → enroll its voiceprint."""
    if not caller.allows_workspace(workspace):
        raise HTTPException(403, f"caller '{caller.name}' not authorized for workspace '{workspace}'")
    embedder = getattr(request.app.state, "embedder", None)
    if embedder is None:
        raise HTTPException(503, "ID embedder not loaded")
    try:
        audio = decode_to_mono(await file.read())
    except AudioDecodeError as exc:
        raise HTTPException(400, f"audio decode failed: {exc}")
    duration = len(audio) / config.TARGET_SAMPLE_RATE
    result = enroll(
        request.app.state.store, embedder, workspace, identity, audio,
        config.TARGET_SAMPLE_RATE, duration,
    )
    return {"voiceprint_id": result.voiceprint_id, "status": result.status, "reason": result.reason}
