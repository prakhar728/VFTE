"""Central configuration: service metadata, paths, audio constants.

Model and threshold choices that are decided empirically during the build
(plan §10 UNFINALIZED) default to "TBD" and are set after the relevant gate:
the diarization engine after the M2 bake-off, the ID embedding model after the
C3.1 bench, and the matching thresholds after C5.2 calibration.
"""
from __future__ import annotations

import os
from pathlib import Path

SERVICE_NAME = "fpm"
SERVICE_VERSION = "0.0.1"

# --- storage (kept out of git; see .gitignore) ---
DATA_DIR = Path(os.environ.get("FPM_DATA_DIR", "./data"))
DB_PATH = DATA_DIR / "voiceprints.db"

# --- audio ---
TARGET_SAMPLE_RATE = 16_000  # all internal processing is 16 kHz mono

# --- models (fetched at build time via scripts/fetch_models.sh; baked into image) ---
MODELS_DIR = Path(os.environ.get("FPM_MODELS_DIR", "./models"))
# The FIXED ID embedder defines the voiceprint space. CAM++ (512-d) for now;
# the CAM++ vs ERes2NetV2 bench (A.3) may revise this.
ID_EMBEDDING_MODEL = os.environ.get("FPM_ID_EMBED", "campplus")
ID_EMBEDDER_PATH = MODELS_DIR / f"{ID_EMBEDDING_MODEL}.onnx"
ID_EMBEDDING_DIM = int(os.environ.get("FPM_ID_EMBED_DIM", "512"))

# --- diarizer engine (offline path) — set after the C.2 diart spike ---
DIARIZATION_ENGINE = os.environ.get("FPM_DIARIZER", "TBD")     # diart | onnx
