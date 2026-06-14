# Standalone DiariZen diarization service (CPU, torch 2.1.1).
#
# Build from the FPM repo root as context so it can COPY fpm/, config.py, etc.:
#   docker buildx build --platform linux/amd64 \
#     -f deploy/diarize-service/Dockerfile \
#     --secret id=hf_token,env=HF_TOKEN \
#     -t <registry>/fpm-diarize:<tag> --push .
#
# The DiariZen install is the load-bearing recipe from requirements-diarizen.txt:
# torch 2.1.1 pin-set, then the vendored pyannote-audio fork + diarizen (both --no-deps),
# then re-pin numpy. The gated model is baked into the image so runtime is fully offline.
FROM --platform=linux/amd64 python:3.11-slim

# git: clone the DiariZen repo (vendored pyannote fork isn't on PyPI).
# ffmpeg: decode arbitrary upload formats (fpm/audio.py shells out to it).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 0) CPU-ONLY torch FIRST — this box has no GPU. Installing torch/torchaudio 2.1.1
#    from the PyTorch CPU index satisfies requirements-diarizen.txt's torch==2.1.1
#    pin with the +cpu wheels, so the next pip never pulls the ~2-3 GB NVIDIA CUDA
#    libraries (cudnn/nvrtc/…). Slashes image size + build/push/pull time.
RUN pip install --no-cache-dir torch==2.1.1 torchaudio==2.1.1 \
    --index-url https://download.pytorch.org/whl/cpu

# 1) the rest of the torch-2.1.1 pin-set + pyannote stack + DiariZen runtime deps.
# requirements-diarizen.txt already pins fastapi + python-multipart + httpx; add the server.
# torch is already satisfied (+cpu) → pip leaves it, no CUDA download here.
COPY requirements-diarizen.txt .
RUN pip install --no-cache-dir -r requirements-diarizen.txt "uvicorn[standard]>=0.30"

# 2) DiariZen itself + its vendored pyannote-audio fork (editable, no-deps), then re-pin numpy
#    (a transitive dep may bump it to 2.x; the stack needs 1.26.4).
RUN git clone --depth 1 https://github.com/BUTSpeechFIT/DiariZen /opt/DiariZen \
    && pip install --no-cache-dir -e /opt/DiariZen/pyannote-audio --no-deps \
    && pip install --no-cache-dir -e /opt/DiariZen --no-deps \
    # the vendored fork's small pure-python deps, skipped by --no-deps above
    # (none pull torch, so the 2.1.1 pin-set is untouched). numpy re-pinned last.
    && pip install --no-cache-dir "semver>=3.0.0" "omegaconf>=2.1,<3.0" "rich>=12.0.0" "tensorboardX>=2.6" \
    && pip install --no-cache-dir "numpy==1.26.4"

# 3) app code — SLIM: only what diarization needs. Deliberately excludes the
#    voiceprint store (fpm/store/), identity (fpm/identify.py), enroll/match/embed.
#    The remote box only ever sees raw audio → it must not carry crown-jewel code.
COPY config.py .
COPY fpm/__init__.py ./fpm/__init__.py
COPY fpm/audio.py ./fpm/audio.py
COPY fpm/diarize ./fpm/diarize
COPY scripts/prefetch_diarizen_model.py ./scripts/prefetch_diarizen_model.py
COPY deploy/diarize-service/service.py .

# 4) bake the gated DiariZen checkpoint into the image's HF cache so runtime needs no egress.
#    HF_TOKEN is provided as a build secret (never persisted in a layer).
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token)" python scripts/prefetch_diarizen_model.py

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

EXPOSE 8086
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8086"]
