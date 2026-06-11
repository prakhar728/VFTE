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

import json
import os
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
