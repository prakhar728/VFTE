# FPM — Speaker Fingerprinting Microservice

Standalone service that takes a mixed audio recording and returns a speaker-labeled
timeline (**offline batch diarization**) plus **persistent voiceprint enrollment &
cross-session identification** (Otter-style "tag once → recognized next time").

- Self-hosted, **CPU-only**, intended to run inside a TEE.
- Diarization + identification only — **no transcription** (that lives upstream in Recato).
- Consent rule: only **tagged** voiceprints get a name; untagged stay anonymous.

This repo is built **standalone first**, against frozen API contracts; Recato/Conclave
integration is a separate, later phase. Build plan: see the project plan file
(`we-want-to-create-expressive-fern.md`).

## Status

Early scaffold (M0). Endpoints land per milestone:

| Endpoint | Caller | Milestone |
|---|---|---|
| `GET /health` | — | ✅ M0 |
| `POST /v1/diarize` | Recato | M4 |
| `GET /v1/vocab/{host_id}` | Recato | M4 |
| `POST /v1/knowledge` | Conclave (write-only) | M4 |

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
