#!/usr/bin/env bash
# Build + push the FPM backend image. No HF token needed (CAM++ is ungated,
# fetched at build by fetch_models.sh). Run:  bash deploy/backend/build.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo ">> building prakharojha/fpm-backend:v1 (linux/amd64) from $ROOT"
docker buildx build --platform linux/amd64 \
  -f deploy/backend/Dockerfile \
  -t prakharojha/fpm-backend:v1 --push .

echo ">> pushed prakharojha/fpm-backend:v1"
