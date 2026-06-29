# VFTE — Voice Fingerprint TEE

> The confidential, user-owned **voice-identity layer**. *(Directory + service historically named "FPM — Speaker Fingerprinting Microservice"; the GitHub repo is `VFTE`. The two names are used interchangeably — config keys, env vars, and the SQLite store are still prefixed `fpm`/`FPM_`.)*

VFTE gives a person ownership of their own voiceprint. Every meeting assistant (Otter, Fireflies, Granola) silently stores a fingerprint of your voice to recognize you across calls. VFTE flips that: each voiceprint lives inside a **Trusted Execution Environment (TEE)** the operator can't read, a voiceprint only ever gets a **name when the named person consents**, and the data subject logs in (with the email they already use for meetings) to a dashboard to stay anonymous, pause enrollment, or delete it. It is a drop-in identity layer any meeting tool can call.

---

## 1. What VFTE is, and where it sits

VFTE is **one of three repos** in a system for confidential team-meeting intelligence with in-person diarization + voice identity (validated live end-to-end 2026-06-27, all merged to `main`):

| Repo (dir) | Role | State |
|---|---|---|
| **Conclave** (`conclave-shape-rotator`) | The product: orchestration, persistence, enrichment / knowledge base, and the in-person "finalize" flow that fans audio out to the other two. | Stateful |
| **capture** (`conclave-sync`) | Diarization + ASR microservice (diart live + DiariZen post). Emits `{start, end, local_speaker}` spans and ASR text. | **Stateless** |
| **VFTE / FPM** (**this repo**, dir `FPM/`) | **Identity-only** voiceprint layer: embed a speech span → match the workspace's consented voiceprints → tag/identify, with a consent plane. | Stateful (sealed store) |

```
                    ┌───────────────────────────────────────────────┐
                    │  Conclave  (product: orchestration + storage)  │
                    └───────┬───────────────────────────┬───────────┘
            recording +     │                           │  identify-spans(ws, audio, spans)
            "diarize this"  ▼                           ▼
                  ┌───────────────────┐         ┌─────────────────────────────┐
                  │   capture         │ spans   │   VFTE / FPM  (THIS REPO)    │
                  │  diart + DiariZen │────────▶│  embed → match → tag         │
                  │  {start,end,      │         │  consent plane + dashboard   │
                  │   local_speaker}  │         │  sealed voiceprint store     │
                  └───────────────────┘         └─────────────────────────────┘
```

### IMPORTANT: VFTE is identity-only (post-strip)

This repo **used to also diarize** (it ran diart / DiariZen behind a `/v1/diarize` endpoint). That was **removed** in the migration step **"P5b"** (commit `df03de1`). Diarization now lives entirely in **capture**. VFTE's headline endpoint is now **`/v1/identify-spans`**: it takes audio + caller-provided spans and returns a per-span identity. There are **no torch engines, no diarizer state** in this service anymore — only the `Segment` / `StreamingDiarizer` *contract* in `fpm/types.py` remains (the identifier still speaks it; capture supplies the spans).

### The flagship in-person pipeline (VFTE's role in **bold**)

1. Conclave **finalizes** a meeting and posts the recording to **capture's DiariZen** for authoritative spans `{start, end, local_speaker}`.
2. Conclave calls **VFTE `/v1/identify-spans(workspace, audio, spans)`**.
3. **VFTE embeds each span (CAM++), and for each local speaker either MATCHes an existing stored voiceprint (recognize) or mints a new anonymous voiceprint (enroll), vote-locking the decision.** It returns `{local_speaker, voiceprint_id, name, decision, confidence}` per span + a final transcript.
4. Conclave **votes that identity onto the transcript**.
5. Names attach via **consent-gated tagging**: a host proposes `(name, email)` for a voiceprint → a **self-tag auto-confirms**, otherwise the binding stays **PENDING** until the target confirms on **VFTE's own consent dashboard** (the Pending inbox).
6. A **second meeting RECOGNIZES** the same speakers from stored voiceprints — the voiceprint count does **not** grow.

This is the product's trust primitive: **a voiceprint only gets a name when the named person consents.**

> Reference docs at the monorepo root (`shape-rotator-all/`): `DIARIZATION-MIGRATION.md`, `BUILD-LOG-diarization-deployment.md`, `TROUBLESHOOTING-inperson.md`. In-repo design docs live in `docs/` and `CONSENT-PLANE.md`.

---

## 2. Core concepts

**Voiceprint.** A speaker's stored identity in a workspace. Internally (`fpm/store/models.py`): a 512-d L2-normalized **centroid** = mean of up to `MAX_EXEMPLARS=20` **exemplar** embeddings, plus optional **sub-centroids** (k-means, kick in at ≥15 exemplars) for mature profiles. Carries `name` (`""` = anonymous), `owner_email` (`""` = unclaimed), the two consent flags, and counts/quality. `voiceprint_id` is a public id (`vp_<hex>`), never a capability — authority comes from the caller's token scope.

**Workspace-scoping.** Every read/write is keyed by `workspace_id`, enforced in SQL — no code path can touch another workspace's voiceprints. The trust boundary ≡ a workspace. Matching is always scoped to the workspace's own centroids (never global).

**enroll vs match vs recognize vs mint.**
- **enroll** — a clip already attributed to an identity strengthens (or creates) that voiceprint (`/v1/enroll`, used by the older gmeet path).
- **match / recognize** — embed a span, cosine-classify it against the workspace's centroids, reuse the matched `voiceprint_id`.
- **mint anonymous** — a speaker the store doesn't know gets a new voiceprint with `name=""` (recognizable next time, nameable later via consent).

**The consent plane.** A name reaches a voiceprint only through **propose → confirm → bind**. A host proposes `(voiceprint, email, name)`; on confirmation the store binds `owner_email` **and** `name` (via `claim_owner` + `set_name`, both audited). The data subject controls their voiceprint from the dashboard: **stay-anonymous** (`identify_allowed=False` — cluster persists, name withheld everywhere), **pause-enrollment** (`enroll_allowed=False`), or **forget** (delete — returns a **signed, verifiable deletion receipt**). Flags are enforced on *every* match from a hot in-memory cache (no decrypt needed to read two booleans).

**Decision tiers.** Open-set matching (`fpm/match.py`) returns one of four raw tiers via calibrated cosine + rejection:

| Tier | Meaning |
|---|---|
| `MATCH` | best cosine ≥ `MATCH_ACCEPT` (0.45) and clearly ahead of #2 → reuse that voiceprint |
| `AMBIGUOUS` | top-2 within `AMBIGUOUS_MARGIN` (0.10) → don't name (name-leak guard) |
| `LOW` | above reject floor but below accept, not ambiguous |
| `UNKNOWN` | best < `MATCH_REJECT` (0.35) (or no centroids) → mint a new anonymous voiceprint |

The streaming identifier (`fpm/identify.py`) adds **session-level** decisions on the wire: `PENDING` (span too short to embed yet), `LOCKED` (vote-locked to a voiceprint), `RELABELED` (an earlier provisional chunk retro-corrected after a lock), and `ANON` (matched/locked but the name is withheld — either unnamed or the owner chose stay-anonymous).

**Vote-lock + retro-relabel.** Diarizer labels wobble early, so the identifier accumulates a **vote** per `local_speaker` and **locks** it to a voiceprint once ≥`LOCK_MIN_VOTES=2` agreeing segments beat the runner-up. Earlier provisional chunks for that speaker are then **retro-relabelled** to the resolved identity. `transcript()` is the authoritative corrected view.

**CAM++ window-canonicalization.** CAM++ embeds variable-length spans unstably (a partial clip can score ~0 against its own speaker). The embedder embeds fixed `EMBED_WINDOW_SEC=2.0`s windows (hop 1.0s) and averages them, so an enrollment clip and an arbitrary diarized span land in the same vector space.

---

## 3. API surface

Two planes, two auth models:

- **Machine-to-machine (`/v1/...`)** — scoped bearer-token auth (`Authorization: Bearer …` or `X-API-Key`). Callers = Conclave / capture. See §7 for tokens.
- **Consent dashboard (`/auth/...`, `/v1/me/...`, `/dashboard`)** — session-cookie auth (the human data subject, signed in via Google or dev-login).

### Open / unauthenticated
| Method · Path | Purpose |
|---|---|
| `GET /health` | `{status, service, version}` liveness. |
| `GET /attestation?nonce=` | TDX attestation quote bound to `nonce`; `{in_tee, nonce, quote}`. Outside a TEE returns a tagged stub (`in_tee:false`). |
| `GET /v1/deletion-receipt-key` | Publish the Ed25519 **deletion-receipt** public key `{alg, public_key, key_id, in_tee}` for offline verification of forget receipts; the pubkey is bound into `/attestation` report_data. |

### Identity (M2M — token scope in brackets)
| Method · Path | Body | Purpose |
|---|---|---|
| `POST /v1/identify-spans` **[identify]** | multipart: `file` (audio), `workspace`, `spans` (JSON array of `{start,end,local_speaker}`), `tag` (`offline`\|`live`) | **Headline endpoint.** Re-embed each caller-supplied span, match/mint/vote-lock, stream NDJSON of `{start,end,local_speaker,voiceprint_id,name,decision,confidence}` + a final `{"type":"transcript","segments":[…]}` line. `tag=offline` **writes** (mints/updates voiceprints); `tag=live` is read-only. |
| `POST /v1/enroll` **[enroll]** | multipart: `file`, `identity`, `workspace` | gmeet path: a clip already attributed to `identity` → enroll/strengthen its voiceprint. Returns `{voiceprint_id, status, reason}`. |
| `POST /v1/identify` **[identify]** | multipart: `file`, `workspace` | Recognize a single clip against enrolled centroids (no diarization). Returns `{voiceprint_id, name, decision, confidence, score}`. |
| `GET /v1/voiceprints/{workspace}` **[voiceprints]** | — | List the workspace's voiceprints (metadata only — never centroid/exemplar bytes). |
| `GET /v1/vocab/{workspace}` **[vocab]** | — | ASR vocab terms + a ready-to-use prompt string. |

### Consent / knowledge (M2M)
| Method · Path | Scope | Purpose |
|---|---|---|
| `POST /v1/knowledge` | knowledge | Conclave→VFTE: name anonymous voiceprints (`bindings:[{voiceprint_id,name,email?}]`) + push `vocab_terms`. An **email-bearing** binding becomes a self-confirmed proposal (binds `owner_email`); a bare-name binding takes the legacy `set_name` path. |
| `POST /v1/propose` | knowledge | Host tags a voiceprint → pending email binding `{workspace,voiceprint_id,proposed_email,proposed_by,proposed_name}`. Auto-confirms on a **self-tag** (`proposed_by==proposed_email`) or the `CONSENT_AUTOCONFIRM` flag; else pending + notify. |
| `GET /v1/consent/resolve/{workspace}/{voiceprint_id}` | knowledge | Read-side consent projection `{voiceprint_id,name,owner_email,visibility}` — `name` is null if `identify_allowed=False` or unbound. The single gate Conclave trusts at display time. |
| `POST /v1/consent/resolve/{workspace}` | knowledge | Batch form (`{voiceprint_ids:[…]}`) → `{resolved:{vid:…}}`. |

### Consent dashboard (session cookie — the data subject)
| Method · Path | Purpose |
|---|---|
| `GET /dashboard` · `GET /` | The dashboard HTML (`/` redirects to `/dashboard`). |
| `GET /auth/login` → `GET /auth/callback` | Google OAuth sign-in (hand-rolled, `urllib` only). |
| `GET /auth/dev-login?email=` | Local-demo bypass — sign in as `email` without Google. Gated on `FPM_DEV_LOGIN`. |
| `POST /auth/logout` | Clear the session cookie. |
| `GET /v1/me` | `{email, signed_in, google_enabled, dev_login}`. |
| `GET /v1/me/voiceprints` | Every voiceprint the signed-in user owns (across workspaces) + usage history. |
| `POST /v1/me/voiceprints/{ws}/{vid}/flags` | Set `identify_allowed` / `enroll_allowed` (stay-anonymous / pause). Owner-only. |
| `POST /v1/me/voiceprints/{ws}/{vid}/forget` | Hard-delete the voiceprint **and return a signed, independently-verifiable deletion receipt** (Ed25519, TEE-sealed key) referencing the surviving `usage_ledger` "forget" row; the receipt is also persisted in `deletion_receipts`. Owner-only. (Crypto-shred + re-enroll tombstone still deferred.) |
| `GET /v1/me/pending` | The signed-in user's pending-identifications inbox (proposals tagged to their email). |
| `POST /v1/confirm` · `POST /v1/deny` | Confirm (`bind owner_email+name`) or deny (`stays Speaker N`) a pending proposal. Only the tagged target may act. |

All errors use a uniform envelope: `{"error": {"status", "message"}}`.

---

## 4. Architecture / key modules

```
FPM/
├── main.py            FastAPI entrypoint: M2M /v1 endpoints, /health, /attestation, lifespan
│                      (loads store + CAM++ embedder + auth once on startup)
├── consent_api.py     Consent-plane router: Google/dev sign-in, /v1/me/*, confirm/deny, pending
├── auth.py            Two auth systems: scoped M2M TokenAuth + Caller; SessionManager (HMAC
│                      cookie) + GoogleOAuth (stdlib urllib) for the dashboard
├── config.py          All config + env knobs (loads .env via python-dotenv)
├── notify.py          Best-effort SMTP "you've been identified — sign in to confirm" mail
├── ratelimit.py       Per-caller fixed-window write rate limiter (429 over budget)
├── fpm/
│   ├── types.py       Segment + StreamingDiarizer contract (engines removed in P5 strip)
│   ├── audio.py       ffmpeg decode → 16 kHz mono float32 (AudioDecodeError on failure)
│   ├── match.py       Open-set classify(): calibrated cosine → MATCH/AMBIGUOUS/LOW/UNKNOWN
│   ├── identify.py    SessionIdentifier: diarize-stream → embed → classify → vote-lock →
│   │                  mint-anonymous → retro-relabel. The identity "brain".
│   ├── identify_spans.py  /v1/identify-spans core: SpanReplayDiarizer replays caller spans
│   │                  into SessionIdentifier (buffered_batch) → identified segments.
│   ├── enroll.py      enroll(): labeled clip → create/strengthen a voiceprint (quality-gated).
│   ├── enclave.py     dstack/Phala TEE: sealed key derivation (get_sealed_key) + TDX quotes.
│   ├── embed/
│   │   ├── fbank.py          pure-numpy 80-d log-mel fbank front-end (torch-free)
│   │   └── onnx_embedder.py  CAM++ 512-d embedder via onnxruntime; fixed-window + average
│   └── store/
│       ├── store.py   VoiceprintStore: encrypted, workspace-scoped SQLite (the heart)
│       ├── models.py  Voiceprint dataclass: exemplar retention, centroid, sub-centroids, quality
│       └── crypto.py  AES-256 blob encryption; master key from TEE seal or keyfile/env
├── webapp/dashboard.html   Self-contained dashboard served by the backend at /dashboard
├── frontend/          Next.js 16 / React 19 / Tailwind 4 consent dashboard (port 3002)
│   └── src/
│       ├── app/dashboard/page.tsx    the dashboard page (voiceprint cards + pending inbox)
│       ├── components/voiceprint-card.tsx   per-voiceprint controls (anonymize/pause/forget)
│       ├── components/pending-inbox.tsx     confirm/deny tags pending against your email
│       └── lib/api.ts                same-origin client (cookie rides along via rewrites)
├── deploy/
│   ├── backend/       Phala CVM for the stateful core (sealed store, port 8085)
│   └── diarize-service/   legacy diarize CVM (diarization now owned by `capture`)
├── models/            campplus.onnx, eres2net_large.onnx (baked into the image)
├── scripts/           fetch_models.sh, seed_consent_demo.py, seed_p4_demo.py, test_email.py, …
├── docs/              design docs (vft-scoping-model, embedder-bench, id-eval, …)
└── tests/             ~162 pytest functions
```

Note: `main.py`/`config.py` docstrings and some scopes still reference `/v1/diarize` and a `diarize` scope — those are **historical**; the diarize endpoint and engines were removed in P5 (the `diarize`/`vocab` token scopes remain valid strings for the legacy/capture-facing callers).

---

## 5. Data model & storage

A single AES-encrypted, WAL-mode SQLite DB at **`/app/data/voiceprints.db`** (`FPM_DATA_DIR/voiceprints.db`; file mode `0600`). Centroids + exemplars are encrypted blobs — **never `SELECT *` / dump them raw**; go through `VoiceprintStore`, which decrypts and caches centroids per-workspace in memory for fast matching.

| Table | Key columns | Notes |
|---|---|---|
| `voiceprints` | `voiceprint_id` PK, `workspace_id`, `name`, `owner_email`, `enroll_allowed`, `identify_allowed`, `centroid` (BLOB, encrypted), `exemplars` (BLOB, encrypted), `exemplar_count`, `enroll_count`, `total_duration_sec`, `quality_score`, `created_at/updated_at/last_seen_at` | The voiceprint. `centroid`/`exemplars` are AES blobs — opaque outside the store. |
| `proposals` | `proposal_id` PK, `workspace_id`, `voiceprint_id`, `proposed_email`, `proposed_by`, `proposed_name`, `status` (`pending`\|`confirmed`\|`denied`), `created_at/confirmed_at/denied_at` | Pending email-binding tags. **Unique** per `(workspace, voiceprint, proposed_email)` → re-tagging never duplicates. `owner_email` binds only on confirm. |
| `binding_audit` | `workspace_id`, `voiceprint_id`, `old_name`, `new_name`, `actor`, `ts` | Every name change — bindings are reversible + traceable. |
| `usage_ledger` | `workspace_id`, `voiceprint_id`, `event`, `consumer`, `purpose`, `ts` | Append-only proof trail: enroll, identify/match, name_bind, control (flag change), forget. Not telemetry. |
| `deletion_receipts` | `id` PK, `workspace_id`, `voiceprint_id`, `owner_email_hash`, `deleted_at`, `ledger_row_id`, `payload_json`, `signature`, `alg`, `key_id` | Append-only record of issued **forget receipts** (Ed25519-signed); survives the voiceprint so a deletion stays provable. No plaintext email — hash only. |
| `vocab` | `workspace_id` PK, `terms` (JSON) | Per-workspace ASR vocabulary. |
| `store_meta` | `key` PK, `value` | Records `embedder_model` + `embedder_dim`; the store **refuses to open** against a different-dim embedder (cross-model matches are meaningless). |

Consent-plane columns (`owner_email`, `enroll_allowed`, `identify_allowed`) were added after the original schema and are applied **in place via `ALTER TABLE` on open**, so an existing real DB upgrades without recreation.

**TEE / sealing.** With `IN_TEE=true` the AES master key is derived from the dstack (Phala, Intel TDX) agent over its Unix socket — bound to this enclave's identity, never written to disk, unreadable by the operator. Off-TEE (local dev) it falls back to `FPM_DB_KEY` or a generated keyfile under `FPM_DATA_DIR`, so the service still runs. `GET /attestation` lets a client verify it's talking to the real enclave before trusting it.

---

## 6. Consent flow (operational)

How a tag goes from proposed → named:

1. **Propose.** A host calls `POST /v1/propose` (or Conclave sends an email-bearing binding via `/v1/knowledge`) with `(workspace, voiceprint_id, proposed_email, proposed_by, proposed_name)`. A `pending` proposal row is created (idempotent per workspace+voiceprint+email).
2. **Auto-confirm shortcuts.** If it's a **self-tag** (`proposed_by == proposed_email`) **or** `FPM_CONSENT_AUTOCONFIRM=1`, the proposal auto-confirms immediately (runs the real `confirm_proposal` path: `claim_owner` + `set_name`, audited). Otherwise it stays `pending` and `notify.notify_identification` fires (log-only unless `FPM_NOTIFY_EMAIL=1`).
3. **Target confirms.** The tagged person signs into the dashboard — locally via `FPM_DEV_LOGIN=1` → `GET /auth/dev-login?email=…`, or in prod via Google — opens the **Pending inbox** (`/v1/me/pending`), and clicks confirm (`POST /v1/confirm`) or deny (`POST /v1/deny`). Only the tagged target may act; a third party can't confirm on your behalf.
4. **Bind.** Confirm binds `owner_email` + `name`. If the owner has `identify_allowed=False` (stay-anonymous), confirm still binds `owner_email` but writes **no name** — revoked consent can never be re-attached by a later tag.

**Workspace-scoping gotcha (important for live ops).** Callers reach VFTE under the **FPM workspace id**, not Conclave's. Conclave maps its workspace → the FPM workspace via `fpm_workspace_for` = `CONCLAVE_FPM_WORKSPACE` (e.g. `local-ws`). If identification "can't find" voiceprints, check that both sides agree on the workspace id. **Resetting voiceprints** for a workspace means deleting that `workspace_id`'s rows across **all** of `voiceprints`, `proposals`, `binding_audit`, `usage_ledger`, and `vocab` — not just the `voiceprints` table.

---

## 7. Configuration

All env vars are read in `config.py` (a local `FPM/.env` is loaded via python-dotenv; real env wins). Copy `.env.example` → `.env`.

**Consent dashboard / auth**
| Var | Meaning |
|---|---|
| `FPM_DEV_LOGIN` | `1` enables `/auth/dev-login?email=` (Google-free local sign-in). **Never in prod.** |
| `FPM_CONSENT_AUTOCONFIRM` | `1` collapses propose→confirm with no email/pending (dev only). **Never in prod.** |
| `FPM_GOOGLE_CLIENT_ID` / `FPM_GOOGLE_CLIENT_SECRET` | Google OAuth web client (unset → OAuth routes 503, dev-login is the only way in). |
| `FPM_OAUTH_REDIRECT_URI` | Must match the registered redirect, e.g. `http://localhost:3002/auth/callback`. |
| `FPM_SESSION_SECRET` | HMAC key for the session cookie. Set in prod (sealed); unset → random per-process (logs everyone out on restart). |
| `FPM_SESSION_TTL_SEC` | Session lifetime (default 7 days). |

**M2M auth + storage + matching**
| Var | Meaning |
|---|---|
| `FPM_AUTH_TOKENS` | JSON `{token:{name,endpoints[],workspaces?[]}}`. Empty → deny all (fail closed). Valid endpoints: `enroll, diarize, vocab, knowledge, voiceprints, identify`. |
| `FPM_DATA_DIR` | Store + keyfile location (default `./data`; `/app/data` in the container). |
| `FPM_DB_KEY` | AES key for the store off-TEE (unset → generated keyfile). |
| `FPM_RATE_LIMIT_WRITES` / `FPM_RATE_LIMIT_WINDOW_SEC` | Write rate-limit budget (default 120 / 60s). |
| `FPM_ID_EMBED` / `FPM_ID_EMBED_DIM` / `FPM_MODELS_DIR` | Embedder model (`campplus`), dim (512), model dir. |
| `FPM_MATCH_ACCEPT` / `FPM_MATCH_REJECT` / `FPM_AMBIGUOUS_MARGIN` | Decision thresholds (0.45 / 0.35 / 0.10, calibrated in E.1). |
| `FPM_SCORE_ALPHA` / `FPM_SCORE_BETA` | Sigmoid confidence calibration. |
| `FPM_MIN_SEGMENT_SEC` / `FPM_EMBED_WINDOW_SEC` / `FPM_EMBED_HOP_SEC` | Quality gate + embedding window/hop. |

**Notify (optional)**: `FPM_NOTIFY_EMAIL` (on → really send), `FPM_SMTP_HOST/PORT/USER/PASS`, `FPM_NOTIFY_FROM`, `FPM_DASHBOARD_URL`. Off → log-only (no provider needed).

**TEE**: `IN_TEE=true` (turns on sealed-key derivation), `DSTACK_AGENT_URL` (simulator override), `FPM_SEAL_KEY_PATH`.

**Frontend** (`frontend/next.config.ts`): `FPM_API_BASE` is the backend the Next dev server proxies `/v1`, `/auth`, `/health` to. **It must be the backend service name/URL, not `localhost`, when the frontend runs server-side in Docker** (otherwise the rewrite hits the frontend container's own localhost). Locally, point it at the backend, e.g. `FPM_API_BASE=http://localhost:8085`. The proxy keeps the httpOnly session cookie same-origin (no CORS).

---

## 8. Run + test

**Backend** (FastAPI, port 8085):
```sh
cd FPM
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
bash scripts/fetch_models.sh            # fetch campplus.onnx into models/ (if absent)
cp .env.example .env                    # set FPM_DEV_LOGIN=1 for the local demo
uvicorn main:app --reload --port 8085   # GET http://127.0.0.1:8085/health
```
Requires Python 3.11/3.12 (ML wheels) and **`ffmpeg` on PATH** for audio decode.

**Consent dashboard frontend** (Next.js, port 3002):
```sh
cd FPM/frontend
npm install
FPM_API_BASE=http://localhost:8085 npm run dev    # http://localhost:3002
```
The frontend proxies `/auth/*`, `/v1/*`, and `/health` to the backend (`next.config.ts` rewrites), so the session cookie works same-origin. (A self-contained `webapp/dashboard.html` is also served directly by the backend at `/dashboard` as a no-build fallback.)

**Tests** (~162 functions across `tests/`):
```sh
cd FPM && source .venv/bin/activate
pytest
```
> Heads-up: a worktree's `.venv` may be **unprovisioned** (missing deps). If `pytest` errors on imports, recreate it with the install step above. Tests inject a tmp store / embedder / auth on `app.state`, so most run without models or a TEE.

---

## 9. Status

- **Identity-only strip complete** — `P5a` added `/v1/identify-spans` (additive), `P5b` stripped diarization out of VFTE (commits `69d6430`, `df03de1`). Diarization now lives in **capture**.
- **Validated live end-to-end 2026-06-27** in the enroll → tag → recognize flow (meeting 1 mints + names via consent; meeting 2 recognizes the same speakers without growing the count). **Merged to `main`.**
- Deployment topology: stateful **backend CVM** (sealed store, port 8085) on Phala + the **Next.js frontend** (port 3002, typically Vercel). The `deploy/diarize-service` CVM is legacy — diarization ownership moved to `capture`.
- **Forget now returns a signed, offline-verifiable deletion receipt** (Ed25519, TEE-sealed key; `GET /v1/deletion-receipt-key` + `docs/deletion-receipt.md` + Python/JS reference verifiers). Merged to `main` 2026-06-28 (Task #1).
- Deferred (per code TODOs / docs): crypto-shred + re-enroll tombstone on forget (Levels B/C), a deletion-cascade event Conclave can subscribe to, and JWKS id_token verification on Google sign-in.
