# Branch C — Live read-only + confidence gate (P1 + P3, bundled)

**Repo:** FPM · **Base:** `main` (post-FF) · **Autonomy:** agent-autonomous (test-first) ·
**Depends on:** Contract **C2** (must preserve the `/v1/diarize` segment shape).
**Bundled because** both edit `fpm/identify.py:SessionIdentifier` — keep on one branch to avoid
self-conflict.

## Goal
- **P1:** a **read-only** mode for the live (diart) path — classify for display, no store writes.
- **P3:** a **confidence/min-duration gate** so weak diarization can't pollute voiceprints.

## Scope (files)
- `fpm/identify.py` — `SessionIdentifier`:
  - add `read_only: bool` (default False). When True: **skip `_mint_anonymous` and `store.upsert`**;
    still run vote-lock **in memory** for stable session labels; still classify against existing
    centroids for display.
  - add the gate: before `_exemplars[spk].append(...)` and as a **precondition in `_maybe_lock`'s
    anonymous-mint branch**, require `confidence ≥ floor` AND `segment.duration ≥ min_dur`.
- `main.py:diarize_endpoint` — select `read_only=True` for the live tag (leave offline default
  writing). **Only the endpoint body** — do not touch `_default_diarizer_factory` (branch A).
- `config.py` — reuse `MATCH_ACCEPT` as the gate floor; add one `MIN_SEGMENT_SEC` constant.

## Things to be careful about
- **Do NOT gate the vote or MATCH-lock** — only gate exemplar-append + anonymous-mint. Gating votes
  makes hard-to-ID speakers stay `PENDING` forever and breaks retro-relabel.
- The gate creates **permanently-unnameable** speakers (`voiceprint_id=None`) — that's intended; the
  UI must not offer "name this speaker" for them (Conclave-side concern, note in B/P4).
- Read-only cache may be stale re: voiceprints minted by the post pass since `open()` → reload on
  session-start (acceptable; live output is provisional).
- Preserve C2: read-only path must still emit the same segment dict shape.

## Test-gated steps (test first → green → atomic commit each)
1. **Read-only writes nothing** — `SessionIdentifier(read_only=True)` over audio with an unknown
   speaker makes **no** `store.upsert`/no new rows; asserts no mint. (Unit, mock store.)
2. **Read-only still identifies** — a known enrolled speaker still resolves to MATCH for display. (Unit.)
3. **Gate blocks exemplar-append** for sub-floor / too-short segments. (Unit.)
4. **Gate blocks anonymous-mint** for weak unknowns → stays `PENDING`/no voiceprint. (Unit.)
5. **Regression:** good segments still vote-lock, MATCH, and retro-relabel as before. (Unit.)

## Definition of done
`read_only` mode + gate implemented; offline default behavior unchanged except the additive gate;
tests 1–5 green; C2 shape preserved.
