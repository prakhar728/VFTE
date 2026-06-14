#!/usr/bin/env bash
# Build + push the DiariZen image (build only — `phala deploy` handled separately).
# Run from anywhere:  bash deploy/diarize-service/build.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

: "${HF_TOKEN:=$(cat ~/.cache/huggingface/token)}"
export HF_TOKEN

echo ">> building prakharojha/fpm-diarize:v1 (linux/amd64) from $ROOT"
docker buildx build --platform linux/amd64 \
  -f deploy/diarize-service/Dockerfile \
  --secret id=hf_token,env=HF_TOKEN \
  -t prakharojha/fpm-diarize:v1 --push .

echo ">> pushed prakharojha/fpm-diarize:v1"
