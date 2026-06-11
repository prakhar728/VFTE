#!/usr/bin/env bash
# Fetch the ONNX ID-embedder model at BUILD time (not committed — it's a large
# binary artifact baked into the image; runtime never downloads).
# Source: sherpa-onnx model zoo (ungated, Apache-2.0), self-contained ONNX
# (input `feats` (B,T,80) → output `embs` (B,512)).
set -euo pipefail
MODELS_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
mkdir -p "$MODELS_DIR"

CAMPP_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM++.onnx"
if [ ! -f "$MODELS_DIR/campplus.onnx" ]; then
  echo "downloading campplus.onnx ..."
  curl -sSL -o "$MODELS_DIR/campplus.onnx" "$CAMPP_URL"
fi
echo "ok: $MODELS_DIR/campplus.onnx"
