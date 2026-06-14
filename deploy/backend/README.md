# FPM backend — Phala CVM deployment

The confidential, always-on core: encrypted voiceprint store, CAM++ identity,
consent flags + ledger, Google sign-in, dashboard API. Its own CVM, separate
from the stateless diarize service.

## Trust posture
- **Voiceprints are sealed.** With `IN_TEE=true` the AES master key is derived
  from the dstack agent (`fpm/enclave.py` → `crypto.get_or_create_key`) — bound
  to this enclave, never written to disk, unreadable by the operator. Off-TEE it
  falls back to `FPM_DB_KEY` or a local keyfile.
- **Attestation:** `GET /attestation?nonce=…` returns a TDX quote clients verify
  before trusting the box.
- **Torch-free:** diarization is delegated to the remote diarize CVM
  (`FPM_DIARIZER=remote`), so the image is small and CPU-light.

## Prereqs (ordering)
Deploy the **diarize CVM first** — the backend needs its URL + token (already
wired into `.env.sealed`). Then:

1. **Build & push** the image (private repo):
   ```bash
   bash deploy/backend/build.sh        # → prakharojha/fpm-backend:v1
   ```
2. **Fill `.env.sealed`** — the diarize URL/token + a session secret are already
   set; add `FPM_GOOGLE_CLIENT_ID/SECRET` and `FPM_OAUTH_REDIRECT_URI`
   (your Vercel domain + `/auth/callback`).
3. **Deploy** — uses the *correct* private-pull mechanism
   (`DSTACK_DOCKER_USERNAME` / `DSTACK_DOCKER_PASSWORD` as encrypted env, **not**
   `phala docker login` — that path does nothing). The script prompts for your
   Docker Hub read-only token:
   ```bash
   bash deploy/backend/deploy.sh
   phala cvms list      # grab the public URL → https://<app-id>-8085.dstack-pha-prod5.phala.network
   ```

> Private-pull gotcha (learned the hard way on the diarize CVM): creds attach at
> **deploy** time via `DSTACK_DOCKER_*`. A `start`/`restart` of a CVM created
> without them re-runs the credless pull and fails — redeploy, don't restart.

## Security checklist (enforced by the compose)
- ✅ `IN_TEE=true` → sealed key; `FPM_DB_KEY` left unset.
- ✅ `FPM_DEV_LOGIN` **unset** (the Google-free bypass stays off).
- ✅ Secrets passed via `-e .env.sealed` (encrypted to the CVM), not baked.
- ✅ CAM++ model baked into the image; no runtime model download.
- ⚠️ Set `FPM_OAUTH_REDIRECT_URI` to the real host and register it in Google Console.

## Sizing
Backend is light (no torch, onnxruntime CAM++ only) → `tdx.large` (4 vCPU/8 GB)
is ample; `tdx.medium` likely fine. The RAM-heavy work lives on the diarize CVM.

## Wire the frontend
Vercel: root dir `frontend/`, env `FPM_API_BASE=https://<this-cvm-host>` (the
Next rewrites proxy `/v1`, `/auth`, `/health` to it, so the session cookie is
same-origin).
