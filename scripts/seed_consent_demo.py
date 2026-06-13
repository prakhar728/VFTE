"""Seed the consent-plane demo: a few owner-bound voiceprints + usage ledger.

Two modes:
  • --synthetic (default): random unit-vector voiceprints — enough to show the
    dashboard (login → see your voiceprint, email, usage → toggle → forget).
    Does NOT match real audio, so /v1/identify won't recognize these.
  • --enroll DIR: enroll real people from wav clips so the FULL record→identify
    E2E works. DIR holds one wav per person named <email>.wav, e.g.
    alice@demo.com.wav. Requires the CAM++ model (scripts/fetch_models.sh).

Run with the FPM venv and the SAME FPM_DATA_DIR / FPM_DB_KEY the server uses:
    FPM_DATA_DIR=./data .venv/bin/python scripts/seed_consent_demo.py
    FPM_DATA_DIR=./data .venv/bin/python scripts/seed_consent_demo.py --enroll ./demo_clips
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ID_EMBEDDING_DIM  # noqa: E402
from fpm.store.models import Voiceprint  # noqa: E402
from fpm.store.store import VoiceprintStore  # noqa: E402

WORKSPACE = "demo-ws"
PEOPLE = ["alice@demo.com", "bob@demo.com"]


def _unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(ID_EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def seed_synthetic(store: VoiceprintStore) -> None:
    for i, email in enumerate(PEOPLE):
        vp = Voiceprint(f"vp_{email.split('@')[0]}", WORKSPACE, name=email, owner_email=email)
        for j in range(3):
            vp.add_exemplar(_unit(i * 10 + j))
        vp.recompute_centroid()
        vp.enroll_count = 3
        vp.quality_score = vp.compute_quality()
        store.upsert(vp)
        store.log_usage(WORKSPACE, vp.voiceprint_id, "enroll", "recato", "created voiceprint")
        store.log_usage(WORKSPACE, vp.voiceprint_id, "identify", "conclave", "matched in meeting")
        print(f"  seeded {email} → {vp.voiceprint_id}")


def seed_from_clips(store: VoiceprintStore, clip_dir: Path) -> None:
    import soundfile as sf

    from config import ID_EMBEDDER_PATH, TARGET_SAMPLE_RATE
    from fpm.embed.onnx_embedder import OnnxSpeakerEmbedder
    from fpm.enroll import enroll

    if not ID_EMBEDDER_PATH.exists():
        sys.exit(f"embedder model missing ({ID_EMBEDDER_PATH}) — run scripts/fetch_models.sh")
    emb = OnnxSpeakerEmbedder(ID_EMBEDDER_PATH).load()
    for wav in sorted(clip_dir.glob("*.wav")):
        email = wav.stem
        audio, sr = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        r = enroll(store, emb, WORKSPACE, email, audio, sr, len(audio) / sr, consumer="recato")
        print(f"  enrolled {email} → {r.voiceprint_id} ({r.status})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enroll", metavar="DIR", help="enroll real people from <email>.wav clips")
    args = ap.parse_args()

    store = VoiceprintStore().open()
    print(f"seeding workspace '{WORKSPACE}' into {store._db_path}")
    if args.enroll:
        seed_from_clips(store, Path(args.enroll))
    else:
        seed_synthetic(store)
    print("owners:", {e: store.find_by_owner_email(e) for e in PEOPLE})
    store.close()
    print("done. Sign in at /dashboard as one of:", ", ".join(PEOPLE))


if __name__ == "__main__":
    main()
