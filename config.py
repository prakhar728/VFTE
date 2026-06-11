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

# --- decided during the build (do not hardcode a winner here) ---
DIARIZATION_ENGINE = os.environ.get("FPM_DIARIZER", "TBD")     # D1 / D2 / D3 — set after M2
ID_EMBEDDING_MODEL = os.environ.get("FPM_ID_EMBED", "TBD")     # CAM++ / ERes2NetV2 — set after C3.1
