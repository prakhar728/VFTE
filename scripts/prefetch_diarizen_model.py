"""One-time ONLINE prefetch of the DiariZen model into the HF cache.

Runtime loads DiariZen with HF offline mode engaged (no network egress), so the
`BUT-FIT/diarizen-wavlm-large-s80-md` checkpoint must already be in `~/.cache/huggingface`.
This builds the pipeline ONCE with offline=False to populate that cache.

Requires an HF token with read access (the model pull is gated), via $HF_TOKEN or
~/.cache/huggingface/token.

Run with the diarizen venv:
  HF_TOKEN=hf_... .venv-diarizen/bin/python scripts/prefetch_diarizen_model.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    if not (os.environ.get("HF_TOKEN") or (Path.home() / ".cache/huggingface/token").exists()):
        sys.exit("no HF token — set $HF_TOKEN or run `huggingface-cli login` first")
    from fpm.diarize.diarizen_engine import DiariZenDiarizer

    print("building DiariZen pipeline ONLINE to warm the HF cache (one-time)…")
    d = DiariZenDiarizer(offline=False)  # offline=False → allowed to download
    d._load()                            # triggers the model fetch
    print("ok — BUT-FIT/diarizen-wavlm-large-s80-md cached. Runtime can now load offline.")


if __name__ == "__main__":
    main()
