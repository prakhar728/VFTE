# Branch P5 — Email-hub dashboard + redaction (OUTLINE)

**Repo:** FPM (+ Conclave companion) · **Autonomy:** human-owned/supervised ·
**Depends on:** P4 (binding) — **serial after P4, same branch/owner.** Detail + tests at branch start.

## Goal
A person signs in and sees all their voiceprints across workspaces (aggregated by `owner_email`),
with one-button self-redaction at workspace + per-meeting granularity, applied retroactively.

## Scope (to detail later)
- Aggregate voiceprints by `owner_email` across workspaces; list meetings.
- Standing per-`(email, workspace)` identify toggle (scoped `identify_allowed`) + per-
  `(email/voiceprint, session_id)` **override**. Precedence: **per-meeting → workspace → anonymous**.
- Toggle change → **re-resolve** → retroactive redaction (name → `Speaker N`).
- **Consent authority = FPM**; Conclave queries FPM at projection and caches.

## Careful about (the security-critical bit)
**Re-resolve MUST pass the live FPM consent gate.** Re-resolving an old transcript for someone who
has since revoked must NOT re-attach their name — otherwise it's a consent-bypass. Test this
explicitly.

## Test-gated steps (sketch)
aggregate-by-email → workspace toggle redacts retroactively → per-meeting override beats workspace →
revoked consent never re-attaches on re-resolve.
