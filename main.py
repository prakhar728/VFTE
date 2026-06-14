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
import consent_api
import notify
from auth import Caller, GoogleOAuth, SessionManager, TokenAuth, _extract_token, require_scope
from fpm.audio import AudioDecodeError, decode_to_mono
from fpm.embed.onnx_embedder import OnnxSpeakerEmbedder
from fpm.enroll import enroll
from fpm.identify import SessionIdentifier
from fpm.match import classify
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
    """Build the configured streaming diarizer (lazy import keeps core torch-free).

    diart and diarizen pin incompatible torch versions, so each lives in its own venv and its
    import stays lazy here — only the selected engine's stack loads. A missing engine venv
    surfaces as a clean 503, not a 500.
    """
    engine = config.DIARIZATION_ENGINE
    try:
        if engine == "diart":
            from fpm.diarize.diart_engine import DiartDiarizer

            return DiartDiarizer(offline=True)
        if engine == "diarizen":
            from fpm.diarize.diarizen_engine import DiariZenDiarizer

            return DiariZenDiarizer()
    except ImportError as exc:
        raise HTTPException(503, f"diarizer engine '{engine}' not installed: {exc}")
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
    # consent-plane end-user auth (WS2): dashboard session signer + Google OAuth.
    if not getattr(app.state, "sessions", None):
        app.state.sessions = SessionManager(config.SESSION_SECRET, config.SESSION_TTL_SEC)
    if not getattr(app.state, "oauth", None):
        app.state.oauth = GoogleOAuth(
            config.GOOGLE_CLIENT_ID, config.GOOGLE_CLIENT_SECRET, config.OAUTH_REDIRECT_URI
        )
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
app.include_router(consent_api.router)  # consent-plane web surface (sign-in + dashboard)


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
        config.TARGET_SAMPLE_RATE, duration, consumer=caller.name,
    )
    return {"voiceprint_id": result.voiceprint_id, "status": result.status, "reason": result.reason}


@app.get("/v1/voiceprints/{workspace}")
async def voiceprints_endpoint(
    request: Request,
    workspace: str,
    caller: Caller = Depends(require_scope("voiceprints")),
) -> dict:
    """List the workspace's stored fingerprints (metadata only — no centroid/exemplar bytes)."""
    if not caller.allows_workspace(workspace):
        raise HTTPException(403, f"caller '{caller.name}' not authorized for workspace '{workspace}'")
    store = request.app.state.store
    out = []
    for vid in store.list_ids(workspace):
        vp = store.get(workspace, vid)
        if vp is None:
            continue
        out.append({
            "voiceprint_id": vp.voiceprint_id,
            "name": vp.name or None,
            "enroll_count": vp.enroll_count,
            "exemplar_count": len(vp.exemplars),
            "quality_score": round(vp.quality_score, 4),
            "last_seen_at": vp.last_seen_at,
        })
    return {"workspace": workspace, "count": len(out), "voiceprints": out}


@app.post("/v1/identify", dependencies=[Depends(enforce_write_limit)])
async def identify_endpoint(
    request: Request,
    file: UploadFile,
    workspace: str = Form(...),
    caller: Caller = Depends(require_scope("identify")),
) -> dict:
    """Recognize a single clip against the workspace's enrolled voiceprints (no diarization)."""
    if not caller.allows_workspace(workspace):
        raise HTTPException(403, f"caller '{caller.name}' not authorized for workspace '{workspace}'")
    embedder = getattr(request.app.state, "embedder", None)
    if embedder is None:
        raise HTTPException(503, "ID embedder not loaded")
    try:
        audio = decode_to_mono(await file.read())
    except AudioDecodeError as exc:
        raise HTTPException(400, f"audio decode failed: {exc}")
    emb = embedder.extract(audio, config.TARGET_SAMPLE_RATE)
    if emb is None:
        return {"voiceprint_id": None, "name": None, "decision": "UNKNOWN",
                "confidence": 0.0, "score": -1.0, "reason": "audio too short to embed"}
    store = request.app.state.store
    res = classify(emb, store.centroids(workspace))
    name, decision = None, res.decision
    if res.voiceprint_id:
        if store.identify_allowed(workspace, res.voiceprint_id):
            vp = store.get(workspace, res.voiceprint_id)
            name = (vp.name or None) if vp else None
            store.log_usage(workspace, res.voiceprint_id, "identify", caller.name, "identify clip")
        else:
            # WS5 "stay anonymous": cluster preserved, name withheld, surfaced as anonymous.
            decision = "ANON"
            store.log_usage(workspace, res.voiceprint_id, "identify", caller.name,
                            "suppressed (anonymous)")
    return {"voiceprint_id": res.voiceprint_id, "name": name, "decision": decision,
            "confidence": round(res.confidence, 4), "score": round(res.score, 4)}


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
    `live`-tagged audio is the read-only diart path: identical NDJSON shape, but it
    mints nothing and writes nothing (the offline/post pass is the sole writer).
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
        result = enroll(request.app.state.store, embedder, workspace, identity, audio, sr,
                        len(audio) / sr, consumer=caller.name)
        return {"routed": "enroll", "voiceprint_id": result.voiceprint_id,
                "status": result.status, "reason": result.reason}

    diarizer = request.app.state.diarizer_factory()
    # P1: `live` (diart) is read-only — classify + stable session labels for display,
    # but mint nothing and write nothing. `offline` is the authoritative writer (post
    # pass). The streamed C2 segment shape is identical either way.
    ident = SessionIdentifier(request.app.state.store, embedder, diarizer, workspace,
                              sample_rate=sr, consumer=caller.name, read_only=(tag == "live"))
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
    email: str | None = None  # P4: when present, bind owner_email (confirmed proposal), not just a name


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

    P4: an **email-bearing** binding evolves the legacy `set_name` into an
    `owner_email` binding — a self-confirmed proposal (claim_owner + set_name) — so the
    name becomes a projection of the consented owner. A bare-name binding keeps the
    legacy `set_name` path unchanged (back-compat).
    """
    if not caller.allows_workspace(body.workspace):
        raise HTTPException(403, f"caller '{caller.name}' not authorized for workspace '{body.workspace}'")
    store = request.app.state.store
    bound, not_found = [], []
    for b in body.bindings:
        if b.email:
            if store.get(body.workspace, b.voiceprint_id) is None:
                not_found.append(b.voiceprint_id)
                continue
            p = store.propose(body.workspace, b.voiceprint_id, b.email, b.email, b.name)
            store.confirm_proposal(p["proposal_id"], actor=caller.name)
            bound.append(b.voiceprint_id)
        else:
            ok = store.set_name(body.workspace, b.voiceprint_id, b.name, actor=caller.name)
            (bound if ok else not_found).append(b.voiceprint_id)
    if body.vocab_terms:
        store.set_vocab(body.workspace, body.vocab_terms)
    return {"bound": bound, "not_found": not_found, "vocab_terms": len(body.vocab_terms)}


class ProposeRequest(BaseModel):
    workspace: str
    voiceprint_id: str
    proposed_email: str
    proposed_by: str
    proposed_name: str = ""


@app.post("/v1/propose", dependencies=[Depends(enforce_write_limit)])
async def propose_endpoint(
    request: Request,
    body: ProposeRequest,
    caller: Caller = Depends(require_scope("knowledge")),
) -> dict:
    """P4 write side (C4): host tags a voiceprint (name+email) → pending email binding.

    Idempotent per (workspace, voiceprint, email). Auto-confirms when the tag is a
    self-identification (`proposed_by == proposed_email`) OR the `CONSENT_AUTOCONFIRM`
    dev flag is on — running the shared confirm path (claim_owner + set_name, audited).
    Otherwise the proposal stays pending until the target confirms on the dashboard
    (Phase 2 fires the notify email here).
    """
    if not caller.allows_workspace(body.workspace):
        raise HTTPException(403, f"caller '{caller.name}' not authorized for workspace '{body.workspace}'")
    store = request.app.state.store
    if store.get(body.workspace, body.voiceprint_id) is None:
        raise HTTPException(404, f"voiceprint '{body.voiceprint_id}' not found in workspace '{body.workspace}'")
    p = store.propose(body.workspace, body.voiceprint_id, body.proposed_email,
                      body.proposed_by, body.proposed_name)
    self_tag = body.proposed_by.strip().lower() == body.proposed_email.strip().lower()
    if p["status"] == "confirmed" or self_tag or config.CONSENT_AUTOCONFIRM:
        binding = store.confirm_proposal(p["proposal_id"], actor=p["proposed_email"])
        return {"proposal_id": p["proposal_id"], "status": "confirmed", "auto_confirmed": True,
                "voiceprint_id": binding["voiceprint_id"], "name": binding["name"],
                "owner_email": binding["owner_email"]}
    # pending: notify the tagged target so they can confirm/deny on the dashboard
    # (FPM-routed, flag-guarded — log-only when FPM_NOTIFY_EMAIL is off). Best-effort: a
    # mail-transport failure must not roll back the (already-created) pending proposal.
    try:
        notify.notify_identification(body.proposed_email, body.workspace, body.proposed_by,
                                     p["proposal_id"])
    except Exception:  # noqa: BLE001 — surface in logs, keep the proposal
        log.warning("notify failed for proposal %s (pending stands)", p["proposal_id"], exc_info=True)
    return {"proposal_id": p["proposal_id"], "status": p["status"], "auto_confirmed": False,
            "voiceprint_id": body.voiceprint_id, "name": None, "owner_email": None}


@app.get("/v1/consent/resolve/{workspace}/{voiceprint_id}")
async def consent_resolve_endpoint(
    request: Request,
    workspace: str,
    voiceprint_id: str,
    caller: Caller = Depends(require_scope("knowledge")),
) -> dict:
    """P4 read side (C4): the projection keystone Conclave queries at display time.

    Returns `{voiceprint_id, name, owner_email, visibility}`; `name` is null whenever
    `identify_allowed=False` or the voiceprint is unbound/unnamed (the read-side consent
    gate, mirroring /v1/identify).
    """
    if not caller.allows_workspace(workspace):
        raise HTTPException(403, f"caller '{caller.name}' not authorized for workspace '{workspace}'")
    r = request.app.state.store.consent_resolve(workspace, voiceprint_id)
    return {"voiceprint_id": voiceprint_id, **r}


class ConsentResolveBatch(BaseModel):
    voiceprint_ids: list[str] = Field(default_factory=list)


@app.post("/v1/consent/resolve/{workspace}")
async def consent_resolve_batch_endpoint(
    request: Request,
    workspace: str,
    body: ConsentResolveBatch,
    caller: Caller = Depends(require_scope("knowledge")),
) -> dict:
    """Batch form of the consent-query (one transcript's worth of voiceprints at a time)."""
    if not caller.allows_workspace(workspace):
        raise HTTPException(403, f"caller '{caller.name}' not authorized for workspace '{workspace}'")
    store = request.app.state.store
    return {"resolved": {vid: store.consent_resolve(workspace, vid) for vid in body.voiceprint_ids}}


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
