"""Consent-plane web surface (WS2 + WS4): standalone Google sign-in + the user's
voiceprint dashboard. Session-authenticated (the data subject), NOT M2M-token —
these routes are the *only* ones a logged-in human reaches; everything under
`/v1/{enroll,diarize,identify,knowledge,...}` stays machine-to-machine.

A user signs in with Google → sees every voiceprint tied to their email across
workspaces, how each has been used (the ledger), and can stay-anonymous, disable
enrollment, or forget a voiceprint entirely. Authorization is per-row: you can only
touch a voiceprint whose `owner_email` matches your session.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

import config
from auth import (
    SESSION_COOKIE,
    GoogleOAuth,
    current_user,
    new_state,
    require_user,
)
from fpm import receipts
from fpm.receipts import ReceiptSigner

router = APIRouter()

_DASHBOARD_HTML = (Path(__file__).parent / "webapp" / "dashboard.html").read_text()
_OAUTH_STATE_COOKIE = "fpm_oauth_state"


def _oauth(request: Request) -> GoogleOAuth:
    oauth: GoogleOAuth | None = getattr(request.app.state, "oauth", None)
    if oauth is None or not oauth.configured:
        raise HTTPException(503, "Google sign-in not configured (set FPM_GOOGLE_CLIENT_ID/SECRET)")
    return oauth


def _set_session(resp: RedirectResponse, request: Request, email: str) -> None:
    token = request.app.state.sessions.issue(email)
    resp.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="lax",
                    max_age=config.SESSION_TTL_SEC, path="/")


# ── auth ─────────────────────────────────────────────────────

@router.get("/")
def root() -> RedirectResponse:
    return RedirectResponse("/dashboard")


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    return _DASHBOARD_HTML


@router.get("/auth/login")
def login(request: Request) -> RedirectResponse:
    oauth = _oauth(request)
    state = new_state()
    resp = RedirectResponse(oauth.auth_url(state))
    resp.set_cookie(_OAUTH_STATE_COOKIE, state, httponly=True, samesite="lax", max_age=600, path="/")
    return resp


@router.get("/auth/callback")
def callback(request: Request, code: str | None = None, state: str | None = None) -> RedirectResponse:
    oauth = _oauth(request)
    expected = request.cookies.get(_OAUTH_STATE_COOKIE)
    if not state or not expected or state != expected:
        raise HTTPException(400, "oauth state mismatch (possible CSRF) — retry sign-in")
    if not code:
        raise HTTPException(400, "missing authorization code")
    email = oauth.exchange_email(code)
    resp = RedirectResponse("/dashboard")
    _set_session(resp, request, email)
    resp.delete_cookie(_OAUTH_STATE_COOKIE, path="/")
    return resp


@router.get("/auth/dev-login")
def dev_login(request: Request, email: str) -> RedirectResponse:
    """Local-demo bypass — sign in as `email` without Google. Gated on FPM_DEV_LOGIN."""
    if not config.DEV_LOGIN:
        raise HTTPException(404, "dev login disabled")
    resp = RedirectResponse("/dashboard")
    _set_session(resp, request, email.lower())
    return resp


@router.post("/auth/logout")
def logout() -> JSONResponse:
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp


@router.get("/v1/me")
def me(request: Request) -> dict:
    email = current_user(request)
    return {"email": email, "signed_in": bool(email),
            "google_enabled": bool(getattr(request.app.state, "oauth", None)
                                   and request.app.state.oauth.configured),
            "dev_login": config.DEV_LOGIN}


# ── dashboard data + controls (WS4 + WS5) ────────────────────

def _owned_or_403(store, workspace_id: str, voiceprint_id: str, email: str):
    vp = store.get(workspace_id, voiceprint_id)
    if vp is None:
        raise HTTPException(404, "voiceprint not found")
    if (vp.owner_email or "").lower() != email.lower():
        raise HTTPException(403, "not your voiceprint")
    return vp


@router.get("/v1/me/voiceprints")
def my_voiceprints(request: Request, email: str = Depends(require_user)) -> dict:
    """Every voiceprint the signed-in user owns, across workspaces, with usage history."""
    store = request.app.state.store
    out = []
    for ws, vid in store.find_by_owner_email(email):
        vp = store.get(ws, vid)
        if vp is None:
            continue
        out.append({
            "workspace_id": ws,
            "voiceprint_id": vp.voiceprint_id,
            "name": vp.name or None,
            "owner_email": vp.owner_email,
            "enroll_allowed": vp.enroll_allowed,
            "identify_allowed": vp.identify_allowed,
            "enroll_count": vp.enroll_count,
            "quality_score": round(vp.quality_score, 4),
            "created_at": vp.created_at,
            "last_seen_at": vp.last_seen_at,
            "usage": store.usage_for_voiceprint(ws, vid),
        })
    return {"email": email, "count": len(out), "voiceprints": out}


class FlagsBody(BaseModel):
    identify_allowed: bool | None = None
    enroll_allowed: bool | None = None


@router.post("/v1/me/voiceprints/{workspace_id}/{voiceprint_id}/flags")
def set_my_flags(
    request: Request, workspace_id: str, voiceprint_id: str,
    body: FlagsBody, email: str = Depends(require_user),
) -> dict:
    store = request.app.state.store
    _owned_or_403(store, workspace_id, voiceprint_id, email)
    if body.identify_allowed is None and body.enroll_allowed is None:
        raise HTTPException(400, "no flag provided")
    store.set_flags(workspace_id, voiceprint_id,
                    identify_allowed=body.identify_allowed,
                    enroll_allowed=body.enroll_allowed, actor=email)
    vp = store.get(workspace_id, voiceprint_id)
    return {"voiceprint_id": voiceprint_id,
            "identify_allowed": vp.identify_allowed, "enroll_allowed": vp.enroll_allowed}


@router.post("/v1/me/voiceprints/{workspace_id}/{voiceprint_id}/forget")
def forget_me(
    request: Request, workspace_id: str, voiceprint_id: str,
    email: str = Depends(require_user),
) -> dict:
    """Erase the voiceprint + return a *signed, offline-verifiable deletion receipt*.

    The voiceprint row is hard-deleted (embeddings gone); the append-only `usage_ledger`
    "forget" row survives as the proof anchor (crypto-shred + tombstone deferred, §6). We
    then sign a receipt referencing that ledger row with the TEE-sealed Ed25519 key and
    persist it (deletion_receipts) so it can be re-shown and independently verified later.

    A receipt is issued ONLY on an actual deletion; an idempotent re-delete returns
    `{deleted: false}` with no receipt.
    """
    store = request.app.state.store
    _owned_or_403(store, workspace_id, voiceprint_id, email)
    result = store.delete(workspace_id, voiceprint_id, actor=email)
    # TODO(deletion cascade, decision F): emit a deletion event Conclave can subscribe to.
    if not result.deleted:
        return {"voiceprint_id": voiceprint_id, "deleted": False}

    signer: ReceiptSigner = request.app.state.receipt_signer
    payload = {
        "version": receipts.RECEIPT_VERSION,
        "voiceprint_id": voiceprint_id,
        "workspace_id": workspace_id,
        "owner_email_hash": receipts.owner_email_hash(email),
        "embedder_model": store.meta("embedder_model", config.ID_EMBEDDING_MODEL),
        "embedder_dim": int(store.meta("embedder_dim", str(config.ID_EMBEDDING_DIM))),
        "deleted_at": result.deleted_at,
        "ledger_row_id": result.ledger_row_id,
    }
    envelope = signer.sign(payload)
    store.add_deletion_receipt(envelope)
    return {"voiceprint_id": voiceprint_id, "deleted": True, "receipt": envelope}


@router.get("/v1/me/deletion-receipts")
def my_deletion_receipts(request: Request, email: str = Depends(require_user)) -> dict:
    """The signed-in user's issued deletion receipts (owner-scoped via email hash), so the
    dashboard can re-show + re-verify a past deletion. Each item is a verifiable envelope."""
    store = request.app.state.store
    receipts_out = store.deletion_receipts_for_hash(receipts.owner_email_hash(email))
    return {"email": email, "count": len(receipts_out), "receipts": receipts_out}


# ── P4 trust handshake: confirm / deny a pending email binding (WS2) ──
# Session-authed (the data subject signed into this dashboard) — these are the human
# side of the handshake; propose + consent-query stay M2M on /v1. Only the tagged
# target may act: the binding is consent, so a third party can't confirm it for you.

@router.get("/v1/me/pending")
def my_pending(request: Request, email: str = Depends(require_user)) -> dict:
    """The signed-in user's pending-identifications inbox (proposals tagged to their email)."""
    return {"email": email, "pending": request.app.state.store.list_pending_for_email(email)}


class ProposalAction(BaseModel):
    proposal_id: str


def _target_proposal_or_error(store, proposal_id: str, email: str) -> dict:
    """Load a proposal and enforce that `email` is its tagged target (404 / 403)."""
    p = store.get_proposal(proposal_id)
    if p is None:
        raise HTTPException(404, "proposal not found")
    if (p["proposed_email"] or "").lower() != email.lower():
        raise HTTPException(403, "not your proposal")
    return p


@router.post("/v1/confirm")
def confirm_proposal_endpoint(
    request: Request, body: ProposalAction, email: str = Depends(require_user),
) -> dict:
    """Confirm a pending binding → bind owner_email + name (reuses the audited store path).

    Only the tagged target (`proposed_email == session email`) may confirm.
    """
    store = request.app.state.store
    _target_proposal_or_error(store, body.proposal_id, email)
    binding = store.confirm_proposal(body.proposal_id, actor=email)
    return {"proposal_id": body.proposal_id, "status": "confirmed", **binding}


@router.post("/v1/deny")
def deny_proposal_endpoint(
    request: Request, body: ProposalAction, email: str = Depends(require_user),
) -> dict:
    """Deny a pending binding → no name attaches (the speaker stays `Speaker N`).

    Only the tagged target may deny.
    """
    store = request.app.state.store
    _target_proposal_or_error(store, body.proposal_id, email)
    res = store.deny_proposal(body.proposal_id, actor=email)
    return {"proposal_id": body.proposal_id, "status": "denied",
            "voiceprint_id": res["voiceprint_id"]}
