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

import json
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import config
from auth import Caller, TokenAuth, _extract_token, require_scope
from fpm.audio import AudioDecodeError, decode_to_mono
from fpm.embed.onnx_embedder import OnnxSpeakerEmbedder
from fpm.enroll import enroll
from fpm.identify import SessionIdentifier
from fpm.store.store import VoiceprintStore
from ratelimit import RateLimiter

log = logging.getLogger(__name__)


def enforce_write_limit(request: Request) -> None:
    """Dependency on write endpoints: 429 once a caller exceeds its write budget."""
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return
    key = _extract_token(request) or "anon"
    if not limiter.allow(key):
        raise HTTPException(429, "write rate limit exceeded; retry later")

_FEED_SEC = 0.5  # chunk size the offline pipeline is fed at


def _default_diarizer_factory():
    """Build the configured streaming diarizer (lazy import keeps core torch-free)."""
    engine = config.DIARIZATION_ENGINE
    if engine == "diart":
        from fpm.diarize.diart_engine import DiartDiarizer

        return DiartDiarizer(offline=True)
    raise HTTPException(503, f"diarizer engine '{engine}' not available")


def _segment_dict(s) -> dict:
    return {
        "start": round(s.start, 3),
        "end": round(s.end, 3),
        "voiceprint_id": s.voiceprint_id,
        "name": s.name,
        "local_speaker": s.local_speaker,
        "decision": s.decision,
        "confidence": round(s.confidence, 4),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Respect pre-set state (tests inject a tmp store / embedder / auth / diarizer).
    if not getattr(app.state, "auth", None):
        app.state.auth = TokenAuth.from_env()
    if not getattr(app.state, "diarizer_factory", None):
        app.state.diarizer_factory = _default_diarizer_factory
    if not getattr(app.state, "rate_limiter", None):
        app.state.rate_limiter = RateLimiter(config.RATE_LIMIT_WRITES, config.RATE_LIMIT_WINDOW_SEC)
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


@app.exception_handler(HTTPException)
async def _http_error(request: Request, exc: HTTPException) -> JSONResponse:
    """Uniform error envelope: {"error": {status, message}}."""
    return JSONResponse(status_code=exc.status_code,
                        content={"error": {"status": exc.status_code, "message": exc.detail}},
                        headers=getattr(exc, "headers", None))


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422,
                        content={"error": {"status": 422, "message": "invalid request",
                                           "detail": exc.errors()}})


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": config.SERVICE_NAME, "version": config.SERVICE_VERSION}


@app.post("/v1/enroll", dependencies=[Depends(enforce_write_limit)])
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


@app.post("/v1/diarize", dependencies=[Depends(enforce_write_limit)])
async def diarize_endpoint(
    request: Request,
    file: UploadFile,
    workspace: str = Form(...),
    tag: str = Form("offline"),
    identity: str | None = Form(None),
    caller: Caller = Depends(require_scope("diarize")),
):
    """offline path: stream live `{start,end,voiceprint_id,name}` for a mixed recording.

    `gmeet`-tagged audio is learn-only and routes to the enroll path (needs identity).
    `offline` audio is diarized + identified live; segments stream as NDJSON, with a
    final `transcript` line carrying the seal-corrected view (retro-relabels applied).
    """
    if not caller.allows_workspace(workspace):
        raise HTTPException(403, f"caller '{caller.name}' not authorized for workspace '{workspace}'")
    embedder = getattr(request.app.state, "embedder", None)
    if embedder is None:
        raise HTTPException(503, "ID embedder not loaded")
    try:
        audio = decode_to_mono(await file.read())
    except AudioDecodeError as exc:
        raise HTTPException(400, f"audio decode failed: {exc}")
    sr = config.TARGET_SAMPLE_RATE

    if tag == "gmeet":  # roster-labeled → learn only, no diarization
        if not identity:
            raise HTTPException(400, "gmeet tag requires an identity (roster-labeled audio)")
        result = enroll(request.app.state.store, embedder, workspace, identity, audio, sr, len(audio) / sr)
        return {"routed": "enroll", "voiceprint_id": result.voiceprint_id, "status": result.status}

    diarizer = request.app.state.diarizer_factory()
    ident = SessionIdentifier(request.app.state.store, embedder, diarizer, workspace, sample_rate=sr)
    step = int(_FEED_SEC * sr)

    def stream():
        ident.start()
        for i in range(0, len(audio), step):
            for seg in ident.feed(audio[i : i + step], sr):
                yield json.dumps(_segment_dict(seg)) + "\n"
        for seg in ident.finish():
            yield json.dumps(_segment_dict(seg)) + "\n"
        # final corrected transcript (retro-relabels applied)
        yield json.dumps({"type": "transcript",
                          "segments": [_segment_dict(s) for s in ident.transcript()]}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


class Binding(BaseModel):
    voiceprint_id: str
    name: str


class KnowledgeRequest(BaseModel):
    workspace: str
    bindings: list[Binding] = Field(default_factory=list)
    vocab_terms: list[str] = Field(default_factory=list)


@app.post("/v1/knowledge", dependencies=[Depends(enforce_write_limit)])
async def knowledge_endpoint(
    request: Request,
    body: KnowledgeRequest,
    caller: Caller = Depends(require_scope("knowledge")),
) -> dict:
    """Conclave→FPM (one-way): name anonymous voiceprints + push ASR vocab.

    Each binding is workspace-checked and audited in the store; binding a name a
    voiceprint already has, or one not in this workspace, is a no-op (returns it in
    `not_found`). Re-binding is allowed (reversible) and leaves an audit trail.
    """
    if not caller.allows_workspace(body.workspace):
        raise HTTPException(403, f"caller '{caller.name}' not authorized for workspace '{body.workspace}'")
    store = request.app.state.store
    bound, not_found = [], []
    for b in body.bindings:
        ok = store.set_name(body.workspace, b.voiceprint_id, b.name, actor=caller.name)
        (bound if ok else not_found).append(b.voiceprint_id)
    if body.vocab_terms:
        store.set_vocab(body.workspace, body.vocab_terms)
    return {"bound": bound, "not_found": not_found, "vocab_terms": len(body.vocab_terms)}


@app.get("/v1/vocab/{workspace}")
async def vocab_endpoint(
    request: Request,
    workspace: str,
    caller: Caller = Depends(require_scope("vocab")),
) -> dict:
    """Recato reads a workspace's ASR vocab (terms + a ready-to-use prompt string)."""
    if not caller.allows_workspace(workspace):
        raise HTTPException(403, f"caller '{caller.name}' not authorized for workspace '{workspace}'")
    terms = request.app.state.store.get_vocab(workspace)
    return {"workspace": workspace, "terms": terms, "prompt": ", ".join(terms)}
