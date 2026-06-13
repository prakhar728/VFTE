"""One-time ONLINE prefetch of diart's pyannote models into the HF cache.

Runtime loads diart models with HF_HUB_OFFLINE=1 (no network egress), so the
segmentation + embedding checkpoints must already be in `~/.cache/huggingface`.
This builds the diart pipeline ONCE with offline=False to populate that cache.

Requires (gated segmentation):
  • accept terms at https://huggingface.co/pyannote/segmentation-3.0
  • an HF token with "read access to public gated repos", via $HF_TOKEN or
    ~/.cache/huggingface/token

Run with the diart venv:
  HF_TOKEN=hf_... .venv-diart/bin/python scripts/prefetch_diart_models.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    if not (os.environ.get("HF_TOKEN") or (Path.home() / ".cache/huggingface/token").exists()):
        sys.exit("no HF token — set $HF_TOKEN or run `huggingface-cli login` first")
    from fpm.diarize.diart_engine import DiartDiarizer

    print("building diart pipeline ONLINE to warm the HF cache (one-time)…")
    d = DiartDiarizer(offline=False)  # offline=False → allowed to download
    d._build_pipeline()               # triggers segmentation + embedding fetch
    print("ok — pyannote/segmentation-3.0 + wespeaker cached. Runtime can now load offline.")


if __name__ == "__main__":
    main()
