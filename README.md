# VFTE — Voice Fingerprint TEE

> The confidential **voice-identity layer**. *(Service formerly named "FPM — Speaker Fingerprinting Microservice".)*

## What this is

**VFTE (Voice Fingerprint TEE)** is a confidential voice-identity layer that gives people ownership of their own voiceprint. Every meeting assistant — Otter, Fireflies, Granola — has to store a fingerprint of your voice to recognize you across calls; today they do it silently, opt-out, and are being sued for it. VFTE flips that. Each person's voiceprint is stored inside a **Trusted Execution Environment (TEE)** the operator can't read, and you **log in with the email you already use for meetings** to see exactly which workspaces and companies use your voiceprint — and control it. **Stay anonymous, pause enrollment, or delete it for good.** Opt-in by design, enforced inside the enclave. It's a drop-in identity layer any meeting tool, transcription service, or voice app can plug into — turning a biometric liability into user-owned trust. *Your voice, your keys.*

## What the service does

- Self-hosted, **CPU-only**, runs inside a **TEE**.
- **Diarization + speaker identification** — who's speaking, across meetings. **No transcription** (that lives upstream in Recato).
- **Persistent voiceprint enrollment + cross-session ID**; consent rule: only enrolled voiceprints get a name, untagged stay anonymous.
- **Standalone Google sign-in + dashboard** (Next.js) where each person sees and controls their own voiceprint — stay-anonymous · pause-enrollment · forget-me — plus a usage ledger of who used it.

## How it's made

VFTE is a confidential voice-identity layer running on TEE infrastructure, with three things wired together: a capture bot, the VFTE identity service, and an intelligence layer.

### The identity core (the hard part)
Real-time speaker diarization **and** cross-session voiceprint identification, entirely on **CPU** — no GPU, because TEEs can't afford them. Torch-free **Python/FastAPI**: a pure-NumPy fbank front-end feeds a **CAM++ 512-d speaker embedder in ONNX Runtime**; diarization runs through **diart** (pyannote segmentation-3.0), hot-swappable with a **DiariZen / WavLM** engine behind one interface. Identity is open-set — calibrated-cosine matching with a rejection tier and sigmoid-calibrated confidence.

Three things we're proud of:
- **diart in "pull mode."** diart's native API is a reactive RxPY *push* stream. We drove it synchronously by building diart's *exact* operator chain on a Subject we push into — inheriting its windowing/latency math verbatim instead of reimplementing it, so `feed()` returns segments immediately.
- **CAM++ window-canonicalization.** CAM++ embeds variable-length spans unstably (a partial clip can score ~0 against its *own* speaker). We embed fixed 2-second windows and average, so an enrollment clip and an arbitrary diarized span land in the same vector space — the difference between identification working and not.
- **Vote-lock + retro-relabel.** Live diarizer labels wobble early; we vote per speaker, lock to a voiceprint once the evidence is clear, then retro-correct earlier provisional chunks into the final transcript.

### The consent plane
Voiceprints are stored **AES-256-encrypted at rest** in a workspace-scoped SQLite store, each carrying an `owner_email` and `enroll_allowed`/`identify_allowed` flags enforced on *every* match (a hot in-memory flag cache means enforcement never has to decrypt). Around it: an append-only **usage ledger**, **standalone Google OAuth hand-rolled in Python stdlib (`urllib`)** to keep the core image lean, a signed session cookie, and a **Next.js 16 / React 19 / Tailwind 4** dashboard to stay-anonymous, pause, or forget your voiceprint.

### The confidential stack (load-bearing tech)
- **NEAR AI** runs **Whisper (whisper-large-v3) ASR + LLM enrichment inside TEEs** (`cloud-api.near.ai`), so transcripts are generated confidentially and never harvested.
- **Phala (dstack, Intel TDX)** is where the services run in confidential compute — it's what makes *"the operator can't read your voiceprint"* true, not marketing. **RedPill** (Phala's OpenAI-compatible TEE LLM) backs enrichment; **Supabase** handles auth; the bot is a **Vexa-derived (Apache-2.0)** fork. Strip out the NEAR/Phala TEEs and the privacy claim collapses — they're why it's confidential *by construction.*

### Notable hacks
The in-person flow records in the browser (`getUserMedia`/`MediaRecorder`), then the backend fans the clip to VFTE and NEAR Whisper **in parallel** (`asyncio.gather`) and **merges by timestamp** — diarizer ∥ ASR, not a pipeline. Browser MediaRecorder emits webm/opus that NEAR's Whisper rejects, so we transcode to 16 kHz mono WAV with **ffmpeg** on the way in. And we shipped the consent schema onto a **live AES-encrypted SQLite DB via an in-place `ALTER` migration** (with a `.bak` safety net).

## Endpoints

| Endpoint | Caller |
|---|---|
| `GET /health` | — |
| `POST /v1/diarize` · `POST /v1/enroll` · `POST /v1/identify` | Recato (audio → diarize / enroll / identify) |
| `GET /v1/voiceprints/{ws}` · `GET /v1/vocab/{ws}` | Recato / Conclave (read) |
| `POST /v1/knowledge` | Conclave (name anonymous voiceprints) |
| `GET /dashboard` · `GET /v1/me/voiceprints` · `POST …/flags` · `POST …/forget` · `/auth/*` | the data subject — consent dashboard (Next.js frontend, `:3002`) |

## Build milestones

- **M0** repo + DER evaluation harness (the decision instrument)
- **M1** diarization candidate pipelines (D2 clustering+VBx, D1 DiariZen, D3 EEND-TA)
- **M2** bake-off + engine decision (DER / CPU-RTF / RAM on AMI + CALLHOME)
- **M3** persistent-ID layer (calibrated-cosine matching, encrypted voiceprint store)
- **M4** FastAPI service + scoped auth
- **M5** evaluation + hardening

## Dev setup

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest
uvicorn main:app --reload --port 8085   # GET http://127.0.0.1:8085/health
```

Requires Python 3.11/3.12 (ML wheels), and `ffmpeg` on PATH for audio decoding.
