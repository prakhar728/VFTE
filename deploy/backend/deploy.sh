#!/usr/bin/env bash
# Deploy the FPM backend to a NEW Phala CVM.
#
# The backend image is PUBLIC (public FPM source, ungated CAM++ model, NO baked
# secrets — secrets arrive via the sealed env below at runtime), so no registry
# creds are needed. This keeps provisioning to the simple one-step path; passing
# DSTACK_DOCKER_* at fresh-provision tripped a 400 in the Phala CLI.
#
# Prereq: deploy/backend/.env.sealed filled (Google creds + redirect URI), and
# the image pushed + made public on Docker Hub.
#
# Run:  bash deploy/backend/deploy.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# tdx.small (2 GB) — the backend is light (no torch) and it fits the account's
# resource quota. tdx.large was rejected (400) because the 16 GB diarize CVM had
# consumed the headroom; resize up later only if quota frees and load demands it.
phala deploy -n fpm-backend -t tdx.small \
  -c deploy/backend/docker-compose.yml \
  -e deploy/backend/.env.sealed \
  --wait

echo ">> deployed. Get the URL with:  phala cvms list"
echo "   Backend URL = https://<app-id>-8085.dstack-pha-prod5.phala.network"
echo "   Then set FPM_API_BASE to that on Vercel; add <vercel>/auth/callback in Google Console."
