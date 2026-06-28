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

import hashlib
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
from fpm.identify_spans import identified_dict, identify_spans
from fpm.match import classify
from fpm.receipts import ReceiptSigner
from fpm.store.store import VoiceprintStore
from ratelimit import RateLimiter

# Surface app INFO logs under uvicorn (root otherwise defaults to WARNING, which
# hides notify._send's "notify sent: …" line and makes mail debugging opaque).
logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)


def enforce_write_limit(request: Request) -> None:
    """Dependency on write endpoints: 429 once a caller exceeds its write budget."""
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return
    key = _extract_token(request) or "anon"
    if not limiter.allow(key):
        raise HTTPException(429, "write rate limit exceeded; retry later")

# Diarization REMOVED (migration P5): VFTE is identity-only. The diarizer factory + the fused
# /v1/diarize endpoint are gone; identity on pre-diarized spans is served by /v1/identify-spans
# (capture owns diarization). No torch engines, no diarizer state on app.state.


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
    if not getattr(app.state, "rate_limiter", None):
        app.state.rate_limiter = RateLimiter(config.RATE_LIMIT_WRITES, config.RATE_LIMIT_WINDOW_SEC)
    if not getattr(app.state, "store", None):
        app.state.store = VoiceprintStore().open()
    # Ed25519 signer for deletion receipts (Task #1): TEE sealed key → FPM_RECEIPT_KEY → keyfile.
    if not getattr(app.state, "receipt_signer", None):
        app.state.receipt_signer = ReceiptSigner.from_config()
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


@app.get("/attestation")
def attestation(request: Request, nonce: str = "") -> dict:
    """TDX attestation quote so clients can verify they're talking to the real
    enclave before trusting it. `nonce` is bound into the quote's report_data to
    prove freshness. Outside a TEE this returns a tagged stub (in_tee=false).

    The deletion-receipt public key is *also* bound into report_data
    (`sha256(nonce || pubkey)`) and echoed as `receipt_pubkey_sha256`, so a verifier
    can confirm via the quote that this pubkey belongs to the genuine deletion code at
    the published measurement — chaining: TDX quote → measurement + pubkey ⇒ receipts
    signed by that pubkey are from real deletion."""
    from fpm.enclave import IN_TEE, get_attestation_quote

    signer: ReceiptSigner | None = getattr(request.app.state, "receipt_signer", None)
    pubkey_raw = signer.public_key_raw() if signer else b""
    return {
        "in_tee": IN_TEE, "nonce": nonce,
        "quote": get_attestation_quote(nonce, bind=pubkey_raw),
        "receipt_pubkey_sha256": hashlib.sha256(pubkey_raw).hexdigest() if pubkey_raw else None,
        "receipt_key_id": signer.key_id if signer else None,
    }


@app.get("/v1/deletion-receipt-key")
def deletion_receipt_key(request: Request) -> dict:
    """Publish the deletion-receipt verification key so anyone can verify a receipt
    OFFLINE. `key_id` (= sha256(raw pubkey)[:16]) names the active key; `in_tee` says
    whether it's enclave-sealed. Bound to the enclave measurement via /attestation."""
    from fpm.enclave import IN_TEE

    signer: ReceiptSigner = request.app.state.receipt_signer
    return {
        "alg": "ed25519",
        "public_key": signer.public_key_pem(),
        "public_key_raw_hex": signer.public_key_raw().hex(),
        "key_id": signer.key_id,
        "in_tee": IN_TEE,
    }


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


# /v1/diarize REMOVED (migration P5): VFTE no longer diarizes. The old `gmeet` learn-only path is
# served by /v1/enroll directly; `offline`/`live` diarize+identify is replaced by /v1/identify-spans
# below (capture diarizes, VFTE identifies the spans).


@app.post("/v1/identify-spans", dependencies=[Depends(enforce_write_limit)])
async def identify_spans_endpoint(
    request: Request,
    file: UploadFile,
    workspace: str = Form(...),
    spans: str = Form(...),               # JSON: [{"start":..,"end":..,"local_speaker":".."}, ...]
    tag: str = Form("offline"),
    caller: Caller = Depends(require_scope("identify")),
):
    """Identity-only path (migration P5): the caller (capture) already diarized; VFTE just identifies.

    The inverse of `/v1/diarize` — no diarizer runs here. Given a recording plus the diarization spans
    `{start,end,local_speaker}`, re-embed each span with CAM++, match the workspace store, vote-lock, and
    stream the same C2 `{start,end,local_speaker,voiceprint_id,name,decision,confidence}` NDJSON + a final
    `transcript` line. `tag=offline` writes (mints/updates voiceprints); `tag=live` is read-only.
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
    try:
        span_list = json.loads(spans)
        assert isinstance(span_list, list)
    except (json.JSONDecodeError, AssertionError) as exc:
        raise HTTPException(400, f"spans must be a JSON array: {exc}")

    sr = config.TARGET_SAMPLE_RATE

    def stream():
        segs = identify_spans(audio, workspace, span_list, store=request.app.state.store,
                              embedder=embedder, sample_rate=sr, consumer=caller.name,
                              read_only=(tag == "live"))
        for s in segs:
            yield json.dumps(identified_dict(s)) + "\n"
        yield json.dumps({"type": "transcript",
                          "segments": [identified_dict(s) for s in segs]}) + "\n"

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
