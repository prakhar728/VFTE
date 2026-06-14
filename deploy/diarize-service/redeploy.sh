#!/usr/bin/env bash
# Redeploy the diarize CVM with the CORRECT private-pull mechanism:
# Phala pulls private images using DSTACK_DOCKER_USERNAME / DSTACK_DOCKER_PASSWORD
# passed as encrypted env at deploy time (NOT `phala docker login`).
#
# Run:  bash deploy/diarize-service/redeploy.sh
# Prompts for your Docker Hub read-only access token (hidden; not stored/echoed).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CVM_ID=d558b82c-33ac-4288-9f65-a9842e28f0fd
DOCKER_USER=prakharojha

read -rsp "Docker Hub access token (read-only PAT): " DSTACK_DOCKER_PASSWORD
echo
[ -n "$DSTACK_DOCKER_PASSWORD" ] || { echo "no token entered"; exit 1; }

phala deploy --cvm-id "$CVM_ID" \
  -c deploy/diarize-service/docker-compose.yml \
  -e deploy/diarize-service/.deploy.env \
  -e IMAGE=prakharojha/fpm-diarize:v1 \
  -e DSTACK_DOCKER_USERNAME="$DOCKER_USER" \
  -e DSTACK_DOCKER_PASSWORD="$DSTACK_DOCKER_PASSWORD" \
  --wait
