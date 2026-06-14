# DiariZen diarization service

The heavy half of the offline diarize path, isolated on its own box so the FPM
core stays torch-free. Runs **only** DiariZen and returns anonymous segments —
no embeddings, no voiceprints, no text (engine-independent-store invariant). The
box is stateless and holds no identity, so it can be torn down freely.

```
FPM core (torch-free)                 diarize service (this, torch 2.1.1)
  /v1/diarize                            POST /diarize
    └─ RemoteDiariZenDiarizer  ──audio──▶  DiariZenDiarizer ──▶ segments
    └─ CAM++ re-embed + store   ◀─segments─┘   (no store here)
```

## Endpoints

### `GET /health`
```json
{ "status": "ok", "engine": "diarizen", "model": "BUT-FIT/diarizen-wavlm-large-s80-md" }
```

### `POST /diarize`  (auth: `Authorization: Bearer <FPM_DIARIZE_TOKEN>`)
Multipart form:
- `file` — the recording (any ffmpeg-decodable format; decoded to mono 16 kHz)
- `workspace` — optional, log correlation only

Response:
```json
{
  "segments": [
    { "start": 0.48, "end": 3.91, "local_speaker": "speaker0" },
    { "start": 4.10, "end": 7.22, "local_speaker": "speaker1" }
  ],
  "sample_rate": 16000,
  "duration_sec": 64.0,
  "elapsed_sec": 81.3
}
```
`local_speaker` is engine-local (not stable, not a voiceprint id). Returns **413**
if the clip exceeds `FPM_DIARIZEN_MAX_CLIP_SEC` (the RAM guard — ~360 s on tdx.large).

Quick check:
```bash
curl -sS -H "Authorization: Bearer $FPM_DIARIZE_TOKEN" \
  -F file=@eval_data/spike_2spk_64s.wav -F workspace=demo \
  https://<cvm-host>/diarize | jq
```

## Wire it into FPM

Set on the **FPM core** process (not this box):
```bash
FPM_DIARIZER=remote
FPM_DIARIZER_URL=https://<cvm-host>        # from `phala cvms list`
FPM_DIARIZE_TOKEN=<same token as the service>
# optional: FPM_DIARIZER_TIMEOUT=600
```
`/v1/diarize` then forwards audio here for segments and does CAM++ identity locally.
Falls back cleanly (503) if `FPM_DIARIZER_URL` is unset.

## Deploy

Prereqs you supply: Docker daemon running + `docker login`; `export HF_TOKEN=hf_...`
(gated model); Phala CLI authed (already done as `kinoo`).

```bash
export HF_TOKEN=hf_...
export FPM_DIARIZE_TOKEN="$(openssl rand -hex 24)"
REGISTRY=prakharojha ./deploy/diarize-service/deploy.sh
```
Deploys to **tdx.large** (4 vCPU / 8 GB, $0.23/hr, per-second billing). Tear down to
stop compute charges:
```bash
phala cvms list
phala cvms delete <id>     # stateless — nothing to preserve
```

## Sizing note
DiariZen RAM scales super-linearly (4.9 GB @30 s → 11 GB @14 min → 16.6 GB @17 min),
so on the 8 GB box clips are capped (~6 min) via `FPM_DIARIZEN_MAX_CLIP_SEC`. Full-
length meetings need either windowing or a GPU CVM (80 GB VRAM erases the wall). See
`docs/bakeoff-offline.md`.
