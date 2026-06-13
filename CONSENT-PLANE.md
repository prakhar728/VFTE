# FPM Confidential Consent Plane — Design & Build Plan

> **What this is:** turning FPM (the speaker-voiceprint service) into a confidential,
> **user-controlled** identity product — and a demo that showcases in-person recording →
> diarization → "log in and control your own voiceprint."
>
> **Status:** decisions locked; demo MVP scoped; ready to build. Build is gated on an
> explicit go — this doc is the agreed direction, not a green light to ship code.

---

## 1. Why this exists

FPM stores **speaker voiceprints** (CAM++ 512-d embeddings → centroid + exemplar
profiles) to recognize people across meetings ("tag once → recognized later"). Storing
biometric data is normally a **liability** — GDPR Art. 9 special-category data, Illinois
BIPA class-action territory. This idea **flips the liability into a moat**: a TEE-sealed,
user-controlled, auditable biometric-consent plane that no incumbent (Otter, Recall,
Gong) offers.

**The concept:** a person signs into FPM with Google (the same email used as their
roster identity) and gets a dashboard to:
- see **whether** their voice embedding is stored,
- see **how it's used** (by Recato, by Conclave, …) — a usage ledger,
- **control** it: enable/disable enrollment, stay anonymous, or be forgotten entirely.

**The differentiator (the property only a TEE makes possible):** the only party who can
decrypt a voiceprint's metadata is the authenticated data subject — enforced *inside the
attested enclave*. Even the operator can't read it, and the controls execute where the
operator can't reach. "Your voice, your keys — provably," verifiable via attestation,
not marketing.

### Pitch lines
- **One-liner:** "Otter already has a fingerprint of your voice. We're the only ones who
  let you see it, control it, and delete it for good."
- **Default pitch:** "Every meeting assistant — Otter, Gong, Fireflies — quietly keeps a
  fingerprint of your voice to recognize you across calls. You never see it, never agreed
  to it, and can't delete it. We do the same thing, the right way around: your voiceprint
  lives in hardware even we can't read, you sign in and see exactly what's stored and how
  it's used, and you can wipe it permanently — with cryptographic proof it's gone. Same
  magic, but you hold the keys."
- **Investor framing:** "Voiceprints are the next biometric liability — every meeting-AI
  company is quietly hoarding them, and the lawsuits are coming. We flip that: the same
  speaker-recognition that makes transcripts useful, but sealed, user-controlled, and
  provably deletable. We turn everyone else's legal time-bomb into our moat."

---

## 2. Current state (verified in code)

**The capture + identification chain already runs:**
- `/v1/diarize` (offline tag) already returns `{start, end, voiceprint_id, name}` segments
  **plus** a final retro-corrected transcript via `SessionIdentifier` — diarize **and**
  identify in one call.
- `/v1/identify` (clip → `{voiceprint_id, name, decision, confidence}`), `/v1/voiceprints/{ws}`
  (list), `enroll.py` (name = roster email), encrypted store (`store/crypto.py`,
  `FPM_DB_KEY` from TEE sealed-key/KMS).
- Browser mic capture is proven in `eval_harness/static/record.html`
  (`getUserMedia` + `MediaRecorder` → upload).
- `store.py` already has a `binding_audit` table (workspace, voiceprint, old/new name,
  `actor`, `ts`) — the seed of the usage ledger.

**What does NOT exist yet:**
- **No product UI** — FPM is a headless FastAPI service; the only HTML is the dev
  `record.html`.
- **No user login** — only machine-to-machine scoped **token** auth (`auth.py`); no Google
  sign-in for end users.
- **No transcription** — FPM does diarize + identify, *not* ASR. Text comes from NEAR
  Whisper (which Recato wraps via `TRANSCRIPTION_SERVICE_URL`).
- **No consent/policy flags, no usage ledger view, no dashboard.**

**Conclave ↔ FPM are already connected, machine-to-machine:** `/v1/knowledge`
(`Conclave→FPM`: name anonymous voiceprints + push vocab; scope-gated, workspace-checked,
audited) and `/v1/vocab` (Recato reads). FPM **owns identity + consent**; Conclave is a
**consumer** that calls the scoped API and must honor FPM's flags. The standalone FPM
dashboard and "connect to Conclave" are not in conflict.

---

## 3. Decisions (locked)

| Topic | Decision | Note |
|---|---|---|
| **Control model** | **User-supreme** | The individual always wins; opt-out genuinely stops identification regardless of workspace. Positions privacy-first / prosumer, not compliance-surveillance buyers. |
| **Enrollment timing** | **Opt-out (MVP)** | Voiceprint created on first encounter; user views/deletes/disables via dashboard. "User-supreme" delivered via *withdraw/delete*, not *prior consent*. **Tighten to opt-in before regulated verticals.** |
| **A. Identity scope** | **Global login, per-workspace isolated voiceprints** | One Google login; dashboard lists "enrolled in N workspaces" as separate, independently-controlled entries. Tenant isolation preserved. |
| **B. Email field** | **Yes — explicit `owner_email`** on the voiceprint | Plaintext email beside the biometric inside the sealed TEE — acceptable. |
| **C. Auth / dashboard** | **Standalone — FPM has its own dashboard + own Google sign-in** | FPM pitched as a **fully standalone, sellable** product. |
| **D. Control scope** | **Pipeline-wide intent** | A flag means "across the whole pipeline" (FPM + Recato + Conclave); v1 *enforces* at FPM only, semantics won't change later. |
| **E. "Don't identify me"** | **Offer both** | "Stay anonymous" (cluster, no name) vs "forget me entirely" (no embedding). |
| **F. Erasure scope** | **FPM-only now + emit a deletion event** for future Conclave cascade | MVP scrubs the FPM voiceprint; publishes an event Conclave can later subscribe to. |
| **G. Usage ledger** | **Log enroll + match + cross-product reads** (consumer, ts, purpose); append-only, kept for life | It's the audit trail / proof, not telemetry. |
| **In-person / phone-P2P** | **Deferred (future)** | `FPM/docs/in-person-phone-p2p.md` — later quality upgrade (multi-mic). |
| **Live streaming** | **Deferred (future)** | Demo runs **record→batch**; live WS pipeline (~2–3 wk, mostly Recato-side merge) is later polish. |

---

## 4. Consent model

Two **distinct** consents — conflating them is the classic trap:

1. **Consent to record/transcribe a meeting** — per-meeting, legal (all-party-consent
   states). Satisfied by **notice**: the bot/app announces itself; continued participation
   = implied consent. No per-meeting form.
2. **Consent to enroll a persistent voiceprint** — per-**person**, biometric, the sensitive
   one. Captured **once, out-of-band, at sign-in** — never mid-meeting.

**Manual flow:**
- **Rostered member:** signs in with Google → one-time consent screen → sets `enroll_allowed`
  (default per the MVP opt-out decision). `enroll.py` honors the flag.
- **Guest (no account):** appears as an anonymous diarization label; controllable only if
  they sign in.
- **Host-tagging ≠ consent:** a host labeling "Speaker 1 = Alice" names a transcript but must
  **not** trigger voiceprint storage for Alice — you can't consent on someone else's behalf.

---

## 5. How Otter compares (the contrast that sells it)

- Otter does in-person via a **single ambient mic → acoustic diarization → Speaker 1/2/3**.
- To recognize speakers across sessions it **must store voiceprints** — there is no
  consent-free version of "recognize Alice next time."
- Bystanders/non-users are recorded + voice-processed with **no account, no toggle, no
  dashboard, no control**; consent is offloaded to the host; data is operator-readable on
  Otter's cloud. Has drawn privacy/BIPA heat.
- **FPM = same storage mechanic, inverted trust:** TEE-sealed, dashboard, audit, delete.

---

## 6. Deletion & re-enrollment (productionization — design captured, deferred from demo)

**Proof of deletion (confidential-grade):** crypto-shredding + an attested receipt.
1. Each voiceprint blob encrypted with its **own random data key**, stored wrapped by the
   enclave master (change from HKDF-from-master-only in `crypto.py`).
2. **Delete = destroy the wrapped key** → ciphertext *and every backup* unrecoverable at
   once (no backup-hunting).
3. **Enclave-signed deletion receipt** the user verifies against the published measurement
   — unforgeable by the operator. This is the *proof* Otter can't offer.
4. Logged in the append-only ledger.
> Caveat: proves the sealed copy is shredded; derived data (a transcript that printed the
> name) needs the Conclave cascade event (F).

**Re-enrollment suppression (delete is meaningless without it):** under opt-out, a deleted
person would be re-created on next encounter. So "forget me" crypto-shreds the embedding
**and** leaves a minimal **tombstone** `(scope, salted_hash(email)) → DO_NOT_ENROLL`
(no biometric, just a one-way hash + flag — GDPR-standard suppression pattern). `enroll.py`
checks it before any write; only the user can lift it. *Consequence of choosing opt-out:
the tombstone is not optional — it's what makes deletion real.*

---

## 7. Demo MVP — scope

**Showcase:** in-person recording → diarization → log into FPM → see your voiceprint is
tied to your email + how it's used → control access. Runs on **record→batch**.

- **Already implemented (reuse):** `record.html` capture, `/v1/diarize` (diarize+identify),
  `/v1/identify`, `/v1/voiceprints`, `enroll.py`, `binding_audit`.
- **Net-new (user-facing half):** FPM Google login + dashboard + access-control toggles;
  ASR text via NEAR Whisper + batch merge.
- **Explicitly deferred:** live streaming, phone-P2P, crypto-shred deletion proof,
  suppression tombstones, Conclave erasure cascade, enforcement beyond FPM.

**Two surfaces:** **Conclave** = Record button + transcript (the product); **FPM standalone
dashboard** = login + voiceprint controls (the consent plane).

---

## 8. Build plan

### WS1 — In-person capture → identified transcript (mostly assembly)
- **Reuse `/v1/diarize` (offline tag)** — already returns identity segments + corrected
  transcript. The diarize+identify half is DONE.
- **1a. Conclave "Record" button** — new `RecordMeetingButton`, built as a **direct sibling
  of `UploadTranscriptButton`** (`frontend/src/components/upload-transcript.tsx` — orange
  pill + modal, Vantage language), mounted on the **same dashboard spots**
  (`frontend/src/app/dashboard/page.tsx:149` and `:419`). Modal captures in-browser via
  `getUserMedia` + `MediaRecorder` (pattern from `FPM/eval_harness/static/record.html`).
  This is Conclave **ingress mode 3** (bot · upload · **record**).
- **1b. ASR text** — add a NEAR Whisper call on the clip (reuse Recato's
  `TRANSCRIPTION_SERVICE_URL`) → text segments with timestamps.
- **1c. Batch merge-by-timestamp** — ASR text ∥ FPM identity segments → `[name] words`
  (the parallel-then-merge-by-timestamp pattern; batch merge is simple). New small util.
- **1d. Ingest via existing path → "saved like usual" for free** — POST the merged
  transcript through `workspaces.uploadTranscript` (`frontend/src/lib/api.ts` →
  `POST /api/workspaces/{id}/transcripts`). It becomes a normal Conclave meeting and the
  **transcript view reuses `frontend/src/app/meeting/[id]/page.tsx`** — no new view.

### WS2 — FPM standalone Google login (net-new; decision C)
- New Google OAuth path in `fpm/auth.py` *alongside* the M2M token auth: sign-in → session
  cookie → email. Resolve email → voiceprint(s) across workspaces (decision A: isolated
  per-workspace entries).

### WS3 — Voiceprint owns email (decision B)
- Add `owner_email` to the `voiceprints` table (`fpm/store/store.py` `_SCHEMA` +
  `upsert`/`get`) and the `Voiceprint` dataclass (`fpm/store/models.py`).
- `fpm/enroll.py` sets `owner_email` from the roster identity (already passed in).

### WS4 — Dashboard + usage ledger (decisions A, G)
- **Ledger:** extend `binding_audit` (or add `usage_ledger`) to log enroll + identify/match
  + name-bind with `(consumer, ts, purpose)`. Hooks: `enroll()`, `classify()` /
  `SessionIdentifier`, `set_name()`.
- **Dashboard (FPM web app):** logged-in user sees their voiceprint(s) per workspace,
  `owner_email`, and "how it's used" (from the ledger).

### WS5 — Access controls (decision E subset; enforce at FPM per D)
- Add flags to `Voiceprint` + schema: `enroll_allowed`, `identify_allowed` (= "stay
  anonymous" when false), plus a **"forget me"** action.
- **Enforce:** `enroll.py` checks `enroll_allowed` before upsert; `match.py` /
  `SessionIdentifier` returns anonymous when `identify_allowed=False`; "forget me" = delete
  the voiceprint (demo: plain delete; crypto-shred + tombstone deferred per §6).
- Wire dashboard toggles to these.

### Demo seed
- Pre-enroll 2–3 people with emails (via `/v1/diarize` gmeet tag or `/v1/enroll`) so
  identify recognizes them; leave 1 unknown speaker to show the anonymous case.

---

## 9. Verification (E2E demo script)

1. In Conclave, hit **Record**, capture a ~2-min 3-person conversation (2 pre-enrolled,
   1 unknown) → after a few seconds the **merged transcript** shows `[Alice] … [Bob] …
   [Speaker 3] …` and saves as a normal meeting (reusing `meeting/[id]`).
2. **Log into FPM** (Google) as `alice@…` → dashboard shows her voiceprint tied to her
   email + usage ("enrolled by Recato, identified 4×, …").
3. Toggle **"stay anonymous"** → re-run identify → Alice returns anonymous (enforcement).
4. **"Forget me"** → voiceprint gone; a later clip no longer matches her.
- Unit: `enroll.py` refuses when `enroll_allowed=False`; `classify()` returns anonymous
  when `identify_allowed=False`; a ledger row is written per enroll/identify.

---

## 10. Files touched (map)

| Area | Files |
|---|---|
| Conclave Record button + ingest | `frontend/src/components/` (new `record-meeting.tsx`), `frontend/src/app/dashboard/page.tsx`, reuse `frontend/src/lib/api.ts` `uploadTranscript`, reuse `frontend/src/app/meeting/[id]/page.tsx` |
| ASR + merge | new FPM module (NEAR Whisper client + batch merge util) |
| FPM Google login | `fpm/auth.py` (new OAuth path), new dashboard web app |
| Voiceprint email + flags | `fpm/store/models.py`, `fpm/store/store.py` (`_SCHEMA`, `upsert`, `get`), `fpm/enroll.py` |
| Enforcement | `fpm/enroll.py`, `fpm/match.py`, `fpm/identify.py` |
| Ledger | `fpm/store/store.py` (extend `binding_audit` / add `usage_ledger`), hooks in enroll/identify/knowledge |
