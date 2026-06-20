# FPM backend (consent plane core) — Phala CVM image.
#
# Torch-free: diarization is delegated to the remote diarize service
# (FPM_DIARIZER=remote), so this image carries only the light CPU stack
# (FastAPI + onnxruntime CAM++ + crypto). The CAM++ ONNX embedder is baked in
# at build (ungated, Apache-2.0) so runtime needs no model download.
#
# Build from the FPM repo root as context:
#   docker buildx build --platform linux/amd64 -f deploy/backend/Dockerfile \
#     -t <registry>/fpm-backend:<tag> --push .
# Target platform is set by the buildx `--platform linux/amd64` flag at build time.
FROM python:3.11-slim

# ffmpeg: decode arbitrary upload formats (fpm/audio.py). curl: fetch the model.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps first (cache-friendly): core is CPU-only + torch-free.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# runtime code only — NOT frontend/, eval_data/, tests/, docs/, the venvs.
# All root-level app modules (config, main, auth, consent_api, ratelimit, notify, …)
# — glob so a new top-level module (e.g. P4's notify.py) can't be silently missed.
COPY *.py ./
COPY fpm ./fpm
COPY webapp ./webapp
COPY scripts ./scripts

# bake the CAM++ ID embedder into the image (→ /app/models/campplus.onnx).
RUN bash scripts/fetch_models.sh

EXPOSE 8085
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8085"]
