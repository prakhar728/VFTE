"""Scoped token auth — least-privilege per caller (D.1).

Each token maps to a `Caller` with an explicit endpoint allow-list and an optional
workspace allow-list. The two callers in the design:
  - **Recato**  → enroll + diarize + vocab (sends audio / reads ASR vocab; names no one)
  - **Conclave** → knowledge (names anonymous voiceprints; never sees audio)

A token grants ONLY its listed endpoints (out-of-scope → 403) and ONLY its listed
workspaces (cross-workspace → 403, in addition to the store-layer enforcement).
Unknown/missing token → 401. Tokens are loaded from the `FPM_AUTH_TOKENS` env var
(JSON) at startup; in the TEE these come from sealed config. `voiceprint_id` is a
public id, never a capability — authority is the token's scope, checked here.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

from fastapi import HTTPException, Request

VALID_ENDPOINTS = frozenset({"enroll", "diarize", "vocab", "knowledge", "voiceprints", "identify"})


@dataclass(frozen=True)
class Caller:
    name: str
    endpoints: frozenset[str]
    workspaces: frozenset[str] | None = None  # None → all workspaces

    def allows_workspace(self, workspace: str) -> bool:
        return self.workspaces is None or workspace in self.workspaces


class TokenAuth:
    def __init__(self, tokens: dict[str, Caller]):
        self._tokens = dict(tokens)

    @classmethod
    def from_env(cls, raw: str | None = None) -> "TokenAuth":
        """Parse `FPM_AUTH_TOKENS` JSON: {token: {name, endpoints[], workspaces?[]}}.

        Empty/unset → no tokens (deny all): fail closed, never open.
        """
        raw = raw if raw is not None else os.environ.get("FPM_AUTH_TOKENS", "")
        tokens: dict[str, Caller] = {}
        if raw.strip():
            for tok, spec in json.loads(raw).items():
                bad = set(spec.get("endpoints", [])) - VALID_ENDPOINTS
                if bad:
                    raise ValueError(f"unknown endpoint(s) for {spec.get('name', tok)}: {sorted(bad)}")
                ws = spec.get("workspaces")
                tokens[tok] = Caller(
                    name=spec["name"],
                    endpoints=frozenset(spec.get("endpoints", [])),
                    workspaces=frozenset(ws) if ws else None,
                )
        return cls(tokens)

    def caller_for(self, token: str | None) -> Caller:
        if not token or token not in self._tokens:
            raise HTTPException(401, "invalid or missing API token")
        return self._tokens[token]


def _extract_token(request: Request) -> str | None:
    auth = request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key")


def require_scope(endpoint: str):
    """FastAPI dependency: authenticate the token and require `endpoint` scope."""

    def dependency(request: Request) -> Caller:
        auth: TokenAuth | None = getattr(request.app.state, "auth", None)
        if auth is None:
            raise HTTPException(503, "auth not configured")
        caller = auth.caller_for(_extract_token(request))
        if endpoint not in caller.endpoints:
            raise HTTPException(403, f"caller '{caller.name}' not authorized for '{endpoint}'")
        return caller

    return dependency


# ─────────────────────────────────────────────────────────────
# Consent-plane end-user auth (WS2): standalone Google sign-in for the data
# subject — wholly separate from the M2M token auth above. A signed, expiring
# cookie carries the verified email; controls authorize against it.
# ─────────────────────────────────────────────────────────────

SESSION_COOKIE = "fpm_session"
_GOOGLE_AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"
_GOOGLE_USERINFO = "https://openidconnect.googleapis.com/v1/userinfo"


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _b64url_dec(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class SessionManager:
    """Sign / verify the dashboard session cookie (HMAC-SHA256 over {email, exp})."""

    def __init__(self, secret: str, ttl_sec: int):
        # random per-process secret if unset → safe in dev (cookies die on restart)
        self._secret = (secret or secrets.token_hex(32)).encode()
        self._ttl = ttl_sec

    def issue(self, email: str) -> str:
        payload = _b64url(json.dumps({"email": email, "exp": int(time.time()) + self._ttl}).encode())
        sig = _b64url(hmac.new(self._secret, payload.encode(), hashlib.sha256).digest())
        return f"{payload}.{sig}"

    def verify(self, token: str | None) -> str | None:
        """Return the email if the cookie is valid and unexpired, else None."""
        if not token or "." not in token:
            return None
        payload, _, sig = token.partition(".")
        want = _b64url(hmac.new(self._secret, payload.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, want):
            return None
        try:
            data = json.loads(_b64url_dec(payload))
        except (ValueError, json.JSONDecodeError):
            return None
        if int(data.get("exp", 0)) < time.time():
            return None
        return data.get("email") or None


class GoogleOAuth:
    """Minimal Google OAuth 2.0 authorization-code flow (openid+email scope).

    Uses urllib only (no extra runtime deps). The id-bearing `email` is read from
    Google's userinfo endpoint over a server-to-server TLS call we initiate, so the
    browser never handles a token. JWKS signature verification of the id_token is a
    productionization step (the userinfo round-trip is sufficient here).
    """

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def auth_url(self, state: str) -> str:
        q = urllib.parse.urlencode({
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": "openid email",
            "state": state,
            "access_type": "online",
            "prompt": "select_account",
        })
        return f"{_GOOGLE_AUTH}?{q}"

    def exchange_email(self, code: str) -> str:
        """Authorization code → verified email (raises HTTPException on failure)."""
        body = urllib.parse.urlencode({
            "code": code,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code",
        }).encode()
        try:
            req = urllib.request.Request(_GOOGLE_TOKEN, data=body,
                                         headers={"Content-Type": "application/x-www-form-urlencoded"})
            with urllib.request.urlopen(req, timeout=10) as r:
                tok = json.loads(r.read())
            access = tok.get("access_token")
            if not access:
                raise HTTPException(401, "google token exchange returned no access_token")
            ui = urllib.request.Request(_GOOGLE_USERINFO, headers={"Authorization": f"Bearer {access}"})
            with urllib.request.urlopen(ui, timeout=10) as r:
                info = json.loads(r.read())
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 — surface any transport/parse failure as 401
            raise HTTPException(401, f"google sign-in failed: {exc}")
        email = info.get("email")
        if not email or not info.get("email_verified", True):
            raise HTTPException(401, "google account has no verified email")
        return email.lower()


def new_state() -> str:
    return secrets.token_urlsafe(24)


def current_user(request: Request) -> str | None:
    """The signed-in email from the session cookie, or None. Never raises."""
    sm: SessionManager | None = getattr(request.app.state, "sessions", None)
    if sm is None:
        return None
    return sm.verify(request.cookies.get(SESSION_COOKIE))


def require_user(request: Request) -> str:
    """FastAPI dependency: 401 unless a valid dashboard session is present."""
    email = current_user(request)
    if not email:
        raise HTTPException(401, "sign in required")
    return email
