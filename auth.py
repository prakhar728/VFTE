"""Scoped per-caller authentication (enforced from M4 / C4.1).

Design (plan §5): each bearer token maps to
    {caller, allowed_endpoints, workspace_scope}

  - Recato token   -> POST /v1/diarize, GET /v1/vocab
  - Conclave token -> POST /v1/knowledge   (write-only)

Out-of-scope calls are rejected. Workspace-scoped authorization is *additionally*
enforced inside the voiceprint store (plan §4) so no code path can bind/read a
voiceprint across workspaces — a `voiceprint_id` is a public identifier, never a
capability.

Placeholder until M4; `/health` is unauthenticated.
"""
from __future__ import annotations
