"""Merge saved diarization + transcript → a readable speaker-attributed transcript.

    /tmp/diart-venv/bin/python -m eval_harness.fp_attribute <run> [seg_stem ...]

Reads results/transcripts/<stem>.json (words + timestamps) and results/diarization/<stem>.json
(speaker segments), attributes each word to the most-overlapping speaker (same merge as the eval
harness), and writes results/attributed/<stem>.txt — "[speakerN] …text…" per turn. This is the
human-readable way to judge diarization quality: read whether the speaker labels track the words.
Pure (no models) — fast. Needs fp_transcribe + fp_diarize to have run first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from eval_harness.harness.merge import merge


def attribute_piece(run_dir: Path, stem: str, engine: str = "diart") -> Path:
    suffix = "" if engine == "diart" else f".{engine}"
    tj = run_dir / "results" / "transcripts" / f"{stem}.json"          # transcript is engine-independent
    dj = run_dir / "results" / "diarization" / f"{stem}{suffix}.json"  # speakers come from this engine
    if not tj.exists():
        raise SystemExit(f"no transcript for {stem} — run fp_transcribe first ({tj})")
    if not dj.exists():
        raise SystemExit(f"no {engine} diarization for {stem} — run fp_diarize --engine {engine} ({dj})")

    words = [SimpleNamespace(word=w["word"], start=w["start"], end=w["end"])
             for w in json.loads(tj.read_text())["words"]]
    segs = [SimpleNamespace(start=s["start"], end=s["end"], local_speaker=s["speaker"])
            for s in json.loads(dj.read_text())["segments"]]
    turns = merge(words, segs)

    out_dir = run_dir / "results" / "attributed"
    out_dir.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"[{t.speaker} {t.start:6.1f}–{t.end:6.1f}s] {t.text}" for t in turns)
    out = out_dir / f"{stem}{suffix}.txt"
    out.write_text(f"# {stem} [{engine}]: {len(turns)} attributed turns "
                   f"({len({t.speaker for t in turns})} speakers)\n\n{body}\n")
    print(f"  {stem} [{engine}]: {len(turns)} turns → results/attributed/{stem}{suffix}.txt")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="eval_harness.fp_attribute")
    ap.add_argument("run_dir")
    ap.add_argument("stems", nargs="*")
    ap.add_argument("--engine", choices=["diart", "diarizen"], default="diart")
    args = ap.parse_args()
    for stem in (args.stems or ["00_enroll_000-060s", "01_test_060-240s"]):
        attribute_piece(Path(args.run_dir), stem, args.engine)


if __name__ == "__main__":
    main()
