"""Run the REAL diart engine + CAM++ identify over a clip — no server needed.

This drives the exact offline implementation Conclave's /record route calls
(`DiartDiarizer` → `SessionIdentifier`): diart segments the audio, CAM++ re-embeds
each span, and it's matched/locked against the workspace's enrolled voiceprints.
Prints the live segments and the final retro-corrected transcript ([speaker] text).

Examples (diart venv):
  # built-in 2-speaker clip from the in-repo AMI fixtures, with spkA enrolled as Alice
  HF_TOKEN=hf_... .venv-diart/bin/python scripts/diarize_clip.py --enroll-fixtures
  # your own recording
  HF_TOKEN=hf_... .venv-diart/bin/python scripts/diarize_clip.py --wav meeting.wav
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import ID_EMBEDDER_PATH, TARGET_SAMPLE_RATE  # noqa: E402
from fpm.diarize.diart_engine import DiartDiarizer  # noqa: E402
from fpm.embed.onnx_embedder import OnnxSpeakerEmbedder  # noqa: E402
from fpm.enroll import enroll  # noqa: E402
from fpm.identify import SessionIdentifier  # noqa: E402
from fpm.store.store import VoiceprintStore  # noqa: E402

SR = TARGET_SAMPLE_RATE
FIX = ROOT / "tests" / "fixtures" / "speakers"
WS = "diart-demo"


def _wav(path: Path) -> np.ndarray:
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        sys.exit(f"{path.name}: expected {SR} Hz, got {sr} — resample first")
    return a


def _fixture_clip() -> np.ndarray:
    """~24 s, speakers A/B alternating in 4 s turns (in-repo CC-BY fixtures)."""
    names = ["spkA_1", "spkB_1", "spkA_2", "spkB_1", "spkA_1", "spkB_1"]
    return np.concatenate([_wav(FIX / f"{n}.wav") for n in names])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=Path, help="16 kHz mono wav to diarize (default: fixture clip)")
    ap.add_argument("--enroll-fixtures", action="store_true",
                    help="enroll spkA as Alice + spkB as Bob first, so identity names them")
    args = ap.parse_args()

    if not ID_EMBEDDER_PATH.exists():
        sys.exit(f"CAM++ model missing ({ID_EMBEDDER_PATH}) — run scripts/fetch_models.sh")

    store = VoiceprintStore(db_path="/tmp/diart-demo.db").open()
    emb = OnnxSpeakerEmbedder(ID_EMBEDDER_PATH).load()

    if args.enroll_fixtures:
        enroll(store, emb, WS, "Alice", _wav(FIX / "spkA_1.wav"), SR)
        enroll(store, emb, WS, "Bob", _wav(FIX / "spkB_1.wav"), SR)
        print(f"enrolled: {store.list_ids(WS)}")

    audio = _wav(args.wav) if args.wav else _fixture_clip()
    print(f"diarizing {len(audio) / SR:.1f}s with diart (CPU, offline)…\n")

    ident = SessionIdentifier(store, emb, DiartDiarizer(offline=True), WS, sample_rate=SR)
    ident.start()
    step = int(0.5 * SR)
    for i in range(0, len(audio), step):
        for seg in ident.feed(audio[i:i + step], SR):
            who = seg.name or f"<{seg.local_speaker}>"
            print(f"  [{seg.start:6.2f}–{seg.end:6.2f}] {who:16} {seg.decision} "
                  f"conf={seg.confidence:.2f} vp={seg.voiceprint_id}")
    for seg in ident.finish():
        who = seg.name or f"<{seg.local_speaker}>"
        print(f"  [{seg.start:6.2f}–{seg.end:6.2f}] {who:16} {seg.decision} "
              f"conf={seg.confidence:.2f} vp={seg.voiceprint_id}")

    print("\n=== final transcript (retro-corrected) ===")
    for seg in ident.transcript():
        print(f"  [{seg.name or 'Speaker ' + seg.local_speaker}]  ({seg.start:.1f}–{seg.end:.1f}s)")
    store.close()


if __name__ == "__main__":
    main()
