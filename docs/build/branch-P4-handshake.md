# Branch P4 — Email binding + pending→confirm trust handshake (OUTLINE)

**Repo:** FPM (+ Conclave companion) · **Base:** `feat/consent-plane`/`main` ·
**Autonomy:** human-owned/supervised (authz, mailing, product judgment) ·
**Depends on:** B (P2, persisted `voiceprint_id`) + Contract **C4** (define at branch start).
**Serial with P5 — same owner, same session.** Detailed plan + test suite to be written when this
branch starts.

## Goal
Editing a speaker = a **pending email binding**. Host tags an attendee `(name+email)` → FPM emails
them → they confirm/deny on the dashboard → confirm sets `owner_email` and re-resolves the name
across all stored transcripts.

## Scope (to detail later)
- FPM: per-voiceprint **proposal state** `{voiceprint_id, proposed_email, proposed_by, status}`;
  `owner_email` set only on confirm; endpoints **propose / confirm / deny** (evolve `/v1/knowledge`
  set_name → email binding); FPM-routed notify email.
- Authz: **host tags**, **only the target confirms**; **self-tag auto-confirms** (tagger email ==
  tagged email).
- Conclave: host speaker-tag UI; dashboard "pending identifications" inbox; confirm/deny;
  verification is **context-only** (no audio); transcript read **in-app via Google login**.

## Test-gated steps (sketch)
persist proposal → notify fires (self-tag auto-confirms) → confirm flips name across transcripts →
deny leaves `Speaker N` → only-target-can-confirm authz.

## Careful about
Consent-bypass (don't surface/auto-attach for `identify_allowed=False`); idempotent proposals;
re-bind reversible + audited.
