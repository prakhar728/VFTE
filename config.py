"""Central configuration: service metadata, paths, audio constants.

Model and threshold choices that are decided empirically during the build
(plan §10 UNFINALIZED) default to "TBD" and are set after the relevant gate:
the diarization engine after the M2 bake-off, the ID embedding model after the
C3.1 bench, and the matching thresholds after C5.2 calibration.
"""
from __future__ import annotations

import os
from pathlib import Path

# Load a local `.env` (next to this file) before reading any os.environ below, so
# secrets like the Google OAuth client live in a gitignored file, not the shell
# history. Real environment variables still win over `.env`. No-op if python-dotenv
# isn't installed or the file is absent.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ModuleNotFoundError:
    pass

SERVICE_NAME = "fpm"
SERVICE_VERSION = "0.0.1"

# --- storage (kept out of git; see .gitignore) ---
DATA_DIR = Path(os.environ.get("FPM_DATA_DIR", "./data"))
DB_PATH = DATA_DIR / "voiceprints.db"

# --- deletion-receipt signing (Task #1: cryptographic proof of deletion) ---
# Ed25519 signing key for "forget me" receipts. Derivation mirrors crypto.get_or_create_key()'s
# priority: TEE sealed key (path RECEIPT_SEAL_KEY_PATH, distinct from the store key) → this env
# (32-byte seed, hex/base64) for off-TEE determinism in dev/CI → a 0600 dev keyfile under DATA_DIR.
RECEIPT_KEY = os.environ.get("FPM_RECEIPT_KEY", "")
# Stable seal-derivation path — separate from the store key (fpm/voiceprint-store) so the signing
# key and the at-rest encryption key are cryptographically independent. Don't change it post-deploy
# (would rotate the pubkey and orphan published key_ids).
RECEIPT_SEAL_KEY_PATH = os.environ.get("FPM_RECEIPT_SEAL_KEY_PATH", "fpm/deletion-signing")

# --- audio ---
TARGET_SAMPLE_RATE = 16_000  # all internal processing is 16 kHz mono

# --- models (fetched at build time via scripts/fetch_models.sh; baked into image) ---
MODELS_DIR = Path(os.environ.get("FPM_MODELS_DIR", "./models"))
# The FIXED ID embedder defines the voiceprint space. CAM++ (512-d) for now;
# the CAM++ vs ERes2NetV2 bench (A.3) may revise this.
ID_EMBEDDING_MODEL = os.environ.get("FPM_ID_EMBED", "campplus")
ID_EMBEDDER_PATH = MODELS_DIR / f"{ID_EMBEDDING_MODEL}.onnx"
ID_EMBEDDING_DIM = int(os.environ.get("FPM_ID_EMBED_DIM", "512"))
# Embedding-window canonicalization: CAM++ embeds variable-length spans unstably
# (a partial span of a voice can score ~0 against that same voice — see
# docs/embedder-bench.md). We embed fixed-length windows and average, so enroll
# (clip) and identify (arbitrary diarized span) live in the SAME space. 0 = off.
EMBED_WINDOW_SEC = float(os.environ.get("FPM_EMBED_WINDOW_SEC", "2.0"))
EMBED_HOP_SEC = float(os.environ.get("FPM_EMBED_HOP_SEC", "1.0"))

# --- matching / open-set rejection — CALIBRATED at E.1 (far-field AMI mix, 8 spk) ---
# E.1: genuine cos mean 0.76, impostor 0.08, EER ~0.2% @ 0.42. ACCEPT 0.45 → FAR 0.3%/
# FRR 0% (name-leak-averse); REJECT 0.35 → FAR 1.3%. See docs/id-eval.md.
MATCH_ACCEPT = float(os.environ.get("FPM_MATCH_ACCEPT", "0.45"))      # ≥ → MATCH
MATCH_REJECT = float(os.environ.get("FPM_MATCH_REJECT", "0.35"))      # < → UNKNOWN
AMBIGUOUS_MARGIN = float(os.environ.get("FPM_AMBIGUOUS_MARGIN", "0.10"))  # best within this of 2nd → AMBIGUOUS
# sigmoid-calibrated confidence: conf = sigmoid(alpha*cos + beta); fitted at E.1
SCORE_ALPHA = float(os.environ.get("FPM_SCORE_ALPHA", "15.5"))
SCORE_BETA = float(os.environ.get("FPM_SCORE_BETA", "-7.67"))
# P3 quality gate (branch C): a diarized span must be at least this long to contribute
# an exemplar / justify minting an anonymous voiceprint — weak/short spans embed
# unreliably and would pollute the centroid. Voting + MATCH-lock stay permissive (NOT
# gated), so hard-to-ID speakers still stabilize. Default 1.0 s ≈ the embedder's own
# minimum, so existing behaviour is unchanged; tune via FPM_MIN_SEGMENT_SEC.
MIN_SEGMENT_SEC = float(os.environ.get("FPM_MIN_SEGMENT_SEC", "1.0"))

# --- enrollment ---
ENROLL_QUALITY_MIN = float(os.environ.get("FPM_ENROLL_QUALITY_MIN", "0.50"))  # min self-sim to centroid

# --- diarization REMOVED (migration P5): VFTE is identity-only. Diarization lives in `capture`;
#     this service identifies pre-diarized spans (/v1/identify-spans). No diarizer engine config here. ---

# --- write rate limiting (per caller token, fixed window) ---
RATE_LIMIT_WRITES = int(os.environ.get("FPM_RATE_LIMIT_WRITES", "120"))
RATE_LIMIT_WINDOW_SEC = float(os.environ.get("FPM_RATE_LIMIT_WINDOW_SEC", "60"))

# --- consent plane: standalone Google sign-in + dashboard session (WS2) ---
# The end-user (data subject) login — distinct from the M2M token auth above.
# In the TEE these come from sealed config. If GOOGLE_CLIENT_ID is unset the OAuth
# routes 503 and the dev-login path (below) is the only way in — fine for local demo.
GOOGLE_CLIENT_ID = os.environ.get("FPM_GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("FPM_GOOGLE_CLIENT_SECRET", "")
OAUTH_REDIRECT_URI = os.environ.get("FPM_OAUTH_REDIRECT_URI", "http://localhost:3002/auth/callback")
# HMAC key for signing the dashboard session cookie. MUST be set in prod (sealed);
# a random per-process default keeps dev safe but logs everyone out on restart.
SESSION_SECRET = os.environ.get("FPM_SESSION_SECRET", "")
SESSION_TTL_SEC = int(os.environ.get("FPM_SESSION_TTL_SEC", str(7 * 24 * 3600)))
# Dev convenience: when set (and Google creds absent), /auth/dev-login?email= signs you
# in without Google — for running the demo locally. NEVER enable in production.
DEV_LOGIN = os.environ.get("FPM_DEV_LOGIN", "").lower() in ("1", "true", "yes")
# P4 dev flag: collapse propose→confirm with no email/no pending so the binding→
# re-resolve→projection spine can be exercised before email/permissions exist
# (Phase 1). Rides the real propose→confirm path. Default OFF; Phase 2 flips it off
# permanently and the only auto-confirm left is the specced self-tag. NEVER on in prod.
CONSENT_AUTOCONFIRM = os.environ.get("FPM_CONSENT_AUTOCONFIRM", "").lower() in ("1", "true", "yes")
# P4 Phase 2: FPM-routed notify email on a pending host-tag (decision §10). Off → log only
# (safe for dev/tests; no provider needed). The mail is notify-only ("you've been identified
# in workspace X — sign in to confirm") — transcript content never leaves the enclave.
NOTIFY_EMAIL = os.environ.get("FPM_NOTIFY_EMAIL", "").lower() in ("1", "true", "yes")
# SMTP transport for the notify email (used only when NOTIFY_EMAIL is on). For a Gmail
# demo: host=smtp.gmail.com, port=587, user=<you>@gmail.com, pass=<app password>. The
# message is notify-only and links to the consent dashboard; no transcript content.
SMTP_HOST = os.environ.get("FPM_SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("FPM_SMTP_PORT", "587"))
SMTP_USER = os.environ.get("FPM_SMTP_USER", "")
SMTP_PASS = os.environ.get("FPM_SMTP_PASS", "")
NOTIFY_FROM = os.environ.get("FPM_NOTIFY_FROM", "") or SMTP_USER
DASHBOARD_URL = os.environ.get("FPM_DASHBOARD_URL", "http://localhost:8091/dashboard")
