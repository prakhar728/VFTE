# Branch P6 — Audio retention lifecycle (OUTLINE)

**Repo:** FPM + Conclave · **Autonomy:** human-owned (security infra) ·
**Depends on:** P2 (`record_routes.py` is a hot-file — coordinate/serialize through B).
Detail + tests at branch start.

## Goal
Retain the returned audio for **transcript-lifetime** so Tier-2 re-embedding corrections work, with
the confidential-layer guarantees intact.

## Scope (to detail later)
- Persist returned audio; **TEE-sealed / encrypted volume** (Phala-style); readable only by the
  post-pass + Tier-2 re-embedding.
- **Delete cascade:** forget-me + transcript delete remove audio + transcript + voiceprint together.

## Careful about
Audio is the most sensitive artifact — sealed at rest, strict reader allowlist, cascade enforced and
tested. `record_routes.py` overlaps P2 → land after B.

## Test-gated steps (sketch)
audio persisted + sealed → readable only by post-pass → forget-me/transcript-delete cascades to audio.
