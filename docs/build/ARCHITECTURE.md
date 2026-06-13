# Diarization + Consent/Trust Architecture — Build Spine

> **Frozen artifact.** This is the durable architecture + build plan, identical in the FPM and
> Conclave repos. Do **not** edit it as branches evolve — record deviations in the per-branch
> `docs/build/branch-*.md` plan files instead. It is written to be self-contained: an agent or
> engineer should be able to build any single branch from this + the branch plan alone.

## 1. What we're building

An in-person meeting capture path that produces an **identified, consent-governed transcript**.
The current Conclave **Record** ingress (`api/record_routes.py`) already does: browser clip →
FPM `/v1/diarize` (offline) ∥ NEAR-Whisper ASR → `merge_by_timestamp` → existing upload pipeline →
a normal meeting. We are evolving it into a **live + post diarization** system with an
**email-bound, consent-governed identity/trust layer**.

Two codebases:
- **FPM** (FastAPI) — the speaker-**identity spine** (voiceprints, diarization engines, consent).
  CPU-bound in production (no GPU TEE budget).
- **Conclave** (Next.js + FastAPI) — ingest, transcript storage, projection/display, dashboards.

## 2. The two disciplines everything falls out of

1. **Identity lives on the voiceprint, never on the transcript.**
2. **Display name is always a projection:** `voiceprint_id → confirmed owner_email → name`.

From these alone you get, with no bespoke logic: replace-safety (a post pass can overwrite the
machine transcript without clobbering human identity edits), retroactive redaction, cross-transcript
name propagation, and end-to-end forget-me.

## 3. Architecture

- **Diarization — two engines, one identity store.**
  - **diart = live, read-only.** Real-time transcript + provisional speaker labels during the
    meeting. Classifies against *existing* voiceprints for display; **mints nothing, writes nothing.**
  - **DiariZen = post, sole authoritative writer.** Runs on the returned audio file when recording
    completes; does the accurate diarize + identify; mints/updates voiceprints (confidence-gated);
    its result **replaces** the live transcript.
  - One writer ⇒ the store cache-coherence problem dissolves (live may read a slightly stale cache
    because its output is provisional and gets overwritten).
  - Engines are interchangeable behind the `StreamingDiarizer` seam (see Contracts). The seam emits
    only `{start, end, local_speaker}` — never embeddings/ids/text. Identity **always** re-embeds
    with fixed CAM++, so swapping engines never invalidates stored voiceprints.

- **Identity spine** (`fpm/identify.py:SessionIdentifier`): diarizer → CAM++ re-embed →
  `match.classify` vs enrolled centroids → vote-lock (`LOCK_MIN_VOTES=2`, clear-leader-no-tie) →
  lock `local_speaker → voiceprint_id`; mint an **anonymous** voiceprint (name="") for unknowns;
  retro-relabel earlier provisional segments on lock. Bounded memory (15s trailing buffer).

- **Trust handshake (the feedback loop).** Editing a speaker = a **pending email binding**, not a
  free-text name. The meeting **host tags** an attendee `(name + email)`; this creates a pending
  proposal; **FPM emails** the tagged address ("you've been identified in workspace X"); the person
  signs into their consent dashboard and **confirms or denies**. Confirm → `owner_email` set →
  **re-resolve** propagates the name across every stored transcript referencing that `voiceprint_id`.
  - **Self-identification auto-confirms** (tagger's logged-in email == tagged email).
  - **Verification before confirm = context-only** (meeting + timestamp + who tagged); **no audio
    playback**.
  - **Transcript delivery = in-app via Google login** (the email is notify-only; confidential
    content never leaves the enclave).

- **Control — email-hub dashboard.** Aggregates a person's voiceprints across workspaces by
  `owner_email`. **Workspace toggle + per-meeting override** for self-redaction; precedence is
  *per-meeting → workspace → anonymous*. Toggling off re-resolves retroactively (name disappears
  from stored transcripts → `Speaker N`). Two depths: **stay-anonymous** (cluster kept, name
  withheld) vs **forget-me** (delete the voiceprint).

- **Quality.** Confidence gate in the **post** pass: gate **exemplar-append** and **anonymous-mint**
  on confidence + min-duration; keep **vote-counting / MATCH-locking permissive** (else hard-to-ID
  speakers never stabilize). Audio retained for **transcript-lifetime**; delete cascades to audio +
  transcript + voiceprint; stored **TEE-sealed / encrypted**.

## 4. CONTRACTS (keystone — freeze before parallel work; do not let these drift)

These two interfaces are the FPM↔Conclave seam and the in-process engine seam. All parallel branch
work is safe **only** if these are fixed up front.

**C1 — `StreamingDiarizer` (engine seam, FPM `fpm/diarize/base.py`).** Every engine emits only:
```
Segment{ start: float, end: float, local_speaker: str }   # never embeddings / ids / text
start(workspace_id) -> None
feed(chunk, sample_rate=16000) -> list[Segment]   # segments finalized by this chunk (may be empty)
finish() -> list[Segment]                          # flush trailing segments
```

**C2 — `/v1/diarize` NDJSON per-segment shape (FPM `main.py:_segment_dict`).** Each streamed line:
```
{ start, end, voiceprint_id, name, decision, confidence, local_speaker }
```
followed by a final `{ "type": "transcript", "segments": [...] }` (seal-corrected). Conclave
**consumes** `voiceprint_id`; any live-path change must **preserve** this shape.

**C3 — `resolved_speakers` schema (Conclave `SessionMetadata`).** Per display label:
```
resolved_speakers[label] = { voiceprint_id, name, confidence }
```
Mutable JSON, no SQL migration. The **display label string is the immutable join key** for
already-enriched signals (`Signal.said_by` etc.) — never rewrite it; name is a read-time projection.

**C4 — propose/confirm/deny + consent-query (P4/P5; define when the core branch starts).** FPM is
the consent authority; Conclave queries FPM at projection time and caches the name/visibility
decision.

## 5. Dependency-ordered build sequence

- **P0 — Production DiariZen engine + two-instance topology** *(#1, post half of #5)*. Port
  `eval_harness/harness/diarizen_engine.py` → `fpm/diarize/diarizen_engine.py` (as-is; already
  implements `StreamingDiarizer`); add a `"diarizen"` branch to `_default_diarizer_factory`; add
  `requirements-diarizen.txt`. Second FPM instance (diarizen venv). **Risks:** torch 2.1.1 ≠ diart
  2.2.2 → separate venv; model weights download on first call → pre-fetch; DiariZen loads the whole
  clip in RAM (~16.6 GB on long AMI) → **cap clip length** on the no-GPU box.
- **P1 — Live diart made read-only** *(single-writer)*. `read_only` flag on `SessionIdentifier`:
  read centroids + in-memory vote-lock for stable session labels, **no `_mint_anonymous`, no
  `store.upsert`**. Cache reload-on-session-start.
- **P2 — Persist `voiceprint_id` + projection** *(#2 foundation)*. `merge_by_timestamp` keeps
  `voiceprint_id`; store per C3; **deterministic "Speaker N" numbering by `voiceprint_id`** (not
  first-appearance). Read path projects id→name; never rewrite `said_by`.
- **P3 — Confidence gate** *(#3)*. Gate exemplar-append + anonymous-mint; voting/MATCH-lock stay
  permissive. Reuse `MATCH_ACCEPT` + a min-duration constant. Creates permanently-unnameable
  speakers (`voiceprint_id=None`) → UI must not offer "name this speaker" for them.
- **P4 — Email binding + pending→confirm handshake** *(#2 core)*. Per-voiceprint proposal state;
  `owner_email` set only on confirm; host tags, target confirms; FPM-routed notify email;
  self-id auto-confirm; evolve `/v1/knowledge` set_name → email binding.
- **P5 — Email-hub dashboard + redaction (workspace + per-meeting)** *(#2 control)*. Aggregate by
  `owner_email`; toggle precedence per-meeting → workspace → anonymous; re-resolve retroactively.
  **Re-resolve MUST pass the live FPM consent gate** (revoked consent must not re-attach a name —
  consent-bypass otherwise).
- **P6 — Audio retention lifecycle**. Retain for transcript-lifetime; TEE-sealed; delete cascades.
- **Time-permitting (not dropped):** **#6 DiariZen windowing for RAM** (windows + `SessionIdentifier`
  as cross-window stitcher; coherent once post is sole writer) and **#4 TS-VAD** (another
  `StreamingDiarizer` impl). Both isolated behind the seam — slot in anytime.

## 6. Build approach & parallelization

**One intrusive core; everything else isolated by two firewalls** — the `StreamingDiarizer` seam
(engines never touch store/identity) and the repo boundary (Conclave projection vs FPM identity).

- **Intrusive core = P4 + P5** (both edit `fpm/store/store.py`, `fpm/store/models.py`,
  `consent_api.py`, dashboard; P5 depends on P4) → **one branch, serial**.
- **Wave 1 — 3 independent branches:**
  - **A (FPM): P0** — engine port + factory branch (`main.py:_default_diarizer_factory`).
  - **B (Conclave): P2** — `voiceprint_id` persistence + projection (consumes existing `/v1/diarize`).
    Unblocks the whole feedback loop.
  - **C (FPM): P1 + P3 bundled** — both edit `identify.py`; keep on one branch.
- **Wave 2:** P4 → P5 (serial, single-owner) after B + C4 contract.
- **Wave 3 (time-permitting):** P6, #6 windowing, #4 TS-VAD.
- **Critical path = B(P2) → P4 → P5.** A and C are speedups off the critical path.

**Parallel mechanism + verified non-overlap (against current files):**
- **Step 0:** freeze contracts C2 + C3 before fan-out — the only real cross-branch risk.
- One `git worktree` per branch. **Merge order: B first → A & C → integration-test A+C jointly →
  P4→P5.** **Scope discipline:** A stays out of `record_routes.py` (B owns it).
- **B ⟂ C ≈ 100%** (disjoint repos). **B ⟂ A ≈ 95%** (no shared file iff A avoids `record_routes.py`).
  **A ⟂ C ≈ 90%** (share only `main.py` in disjoint regions: factory L48 vs diarize endpoint L231).
- **The real risk is not textual conflict** (low, verified) but: (1) contract drift → killed by
  Step 0; (2) A+C integration coupling → joint test; (3) `record_routes.py` is a Conclave hot-file
  (P2/P6/live-path) → serialize through B.

## 7. Build orchestration (per-branch plans · test-gating · autonomy)

- **Each branch = a self-contained `docs/build/branch-*.md` plan**: depends-on contracts, file
  scope, an ordered **test-gated step list (test-first)**, "things to be careful about", DoD.
- **Within a branch = test-gated micro-steps** (persist → attach → state → notify): each
  independently verifiable; write its test first, implement to green, then commit.
- **Autonomy:** the foundation (A=P0, B=P2, C=P1+P3) is **agent-autonomous** given its test suite
  (bounded scope, clear contract, fully unit-testable; review at commit boundaries). The **core**
  (P4 handshake, P5 dashboard/redaction) and **P6** (security infra) are **human-owned/supervised**.
- **Commits: meaningful + atomic** — one commit per test-gated step, tests in the same commit.

## 8. Reuse (do not rebuild)

`eval_harness/harness/diarizen_engine.py` (port as-is) · `SessionIdentifier` (extend, don't fork) ·
`match.classify` + config thresholds · `store.set_name`/`/v1/knowledge` (evolve to email binding) ·
`SessionMetadata.resolved_speakers` (mutable JSON) · `merge_by_timestamp`/`_best_overlap` (keep
overlap logic) · consent plane `owner_email`/`identify_allowed`/`usage_ledger`/delete.

## 9. Verification (per phase)

- **P0/P1:** two FPM instances up; live stream returns NDJSON read-only (no new store rows); post
  instance writes.
- **P2:** recorded meeting's `resolved_speakers` carries `voiceprint_id`; numbering stable across re-run.
- **P3:** weak segments don't mint; unit tests on the gate; vote-lock still stabilizes.
- **P4:** propose→email→confirm flips the name across stored transcripts; self-id auto-confirms;
  deny leaves `Speaker N`.
- **P5:** workspace + per-meeting toggles redact retroactively; revoked consent is never re-attached.
- **P6:** forget-me deletes audio + transcript + voiceprint.

## 10. Decisions locked

- Post-pass = **async** (provisional from live, replace when DiariZen returns).
- Live diart = **read-only**; cache **reload-on-session-start**.
- Confidence gate **reuses** `MATCH_ACCEPT` + one min-duration constant.
- Notification = **FPM-routed email** on host-tag; proposer = **host**.
- Verify-before-confirm = **context-only**, no audio playback.
- Consent authority = **FPM**; Conclave queries + caches at projection.
- Audio at rest = **TEE-sealed / encrypted volume**.
- Transcript delivery = **in-app via Google login** (email is notify-only; content stays in enclave).
- Branch rule: **`eval-inperson-diarization` never merges to main** — port engine files instead.
