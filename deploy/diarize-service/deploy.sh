#!/usr/bin/env bash
# Build → push → deploy the DiariZen diarization service to a Phala CPU CVM.
#
# Prereqs (the parts only you can provide):
#   - Docker daemon running + logged into the registry:  docker login
#   - HF token for the gated model:                      export HF_TOKEN=hf_...
#   - Phala CLI authed (already done as "kinoo"):        phala auth status
#
# Usage:
#   export HF_TOKEN=hf_...
#   export FPM_DIARIZE_TOKEN="$(openssl rand -hex 24)"   # the service bearer token
#   REGISTRY=prakharojha ./deploy/diarize-service/deploy.sh
set -euo pipefail

REGISTRY="${REGISTRY:-prakharojha}"
NAME="${NAME:-fpm-diarize}"
TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%s)}"
IMAGE="${REGISTRY}/${NAME}:${TAG}"

: "${HF_TOKEN:?set HF_TOKEN (gated model pull)}"
: "${FPM_DIARIZE_TOKEN:?set FPM_DIARIZE_TOKEN (service bearer token)}"

# Build from the FPM repo root (two levels up from this script).
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo ">> building $IMAGE (linux/amd64, baking gated model)…"
docker buildx build --platform linux/amd64 \
  -f deploy/diarize-service/Dockerfile \
  --secret id=hf_token,env=HF_TOKEN \
  -t "$IMAGE" --push .

echo ">> deploying to Phala as '$NAME'…"
IMAGE="$IMAGE" FPM_DIARIZE_TOKEN="$FPM_DIARIZE_TOKEN" \
  phala deploy -c deploy/diarize-service/docker-compose.yml -n "$NAME"

echo ">> done. Find the public URL with:  phala cvms list"
echo "   Then point FPM at it:"
echo "     FPM_DIARIZER=remote"
echo "     FPM_DIARIZER_URL=https://<cvm-host>"
echo "     FPM_DIARIZE_TOKEN=$FPM_DIARIZE_TOKEN"
