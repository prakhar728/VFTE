"""Dump the actual diarization timeline for pieces of a fingerprint-persistence run.

    HF_TOKEN=… /tmp/diart-venv/bin/python     -m eval_harness.fp_diarize <run> [stems...]              # diart
    HF_TOKEN=… /tmp/diarizen-venv/bin/python  -m eval_harness.fp_diarize <run> [stems...] --engine diarizen

Runs the chosen diarizer (same path as fp_latency / fp_identify) and WRITES the who-spoke-when
segments so they're inspectable — not just a speaker count. diart writes unsuffixed files; other
engines write <stem>.<engine>.* so a diart run and a diarizen run sit side-by-side:
  results/diarization/<stem>[.<engine>].txt    readable: [speaker]  start–end  (dur)
  results/diarization/<stem>[.<engine>].json   segments + per-speaker totals + distinct-speaker count
(DiariZen must run in /tmp/diarizen-venv — it can't share diart's torch.)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import soundfile as sf

from eval_harness.harness.diarize import make_diarizer

SR = 16_000
DEFAULT_STEMS = ["00_enroll_000-060s", "01_test_060-240s", "02_test_240-end"]


def diarize_segments(audio, engine="diart", window_sec=5.0, step_sec=0.5):
    d = make_diarizer(engine, window_sec, step_sec)
    d.start("fp-diar")
    segs, step = [], max(1, int(step_sec * SR))
    for i in range(0, len(audio), step):
        segs.extend(d.feed(audio[i:i + step], SR))
    segs.extend(d.finish())
    return sorted(segs, key=lambda s: s.start)


def dump_piece(seg_dir: Path, stem: str, out_dir: Path, engine: str = "diart") -> dict:
    suffix = "" if engine == "diart" else f".{engine}"
    audio, sr = sf.read(seg_dir / f"{stem}.wav", dtype="float32")
    if sr != SR:
        raise SystemExit(f"{stem} is {sr} Hz, expected {SR}")
    segs = diarize_segments(audio, engine)
    totals: dict[str, float] = defaultdict(float)
    lines = []
    for s in segs:
        totals[s.local_speaker] += s.end - s.start
        lines.append(f"[{s.local_speaker}]  {s.start:7.2f}–{s.end:7.2f}s  ({s.end - s.start:5.2f}s)")
    speakers = sorted(totals)
    header = (f"# {stem}: {len(speakers)} distinct speakers, {len(segs)} segments, "
              f"{len(audio)/SR:.1f}s audio\n"
              f"# speech per speaker: " + ", ".join(f"{k}={v:.1f}s" for k, v in
                                                    sorted(totals.items(), key=lambda kv: -kv[1])) + "\n\n")
    (out_dir / f"{stem}{suffix}.txt").write_text(header + "\n".join(lines) + "\n")
    (out_dir / f"{stem}{suffix}.json").write_text(json.dumps({
        "stem": stem, "engine": engine, "audio_sec": round(len(audio) / SR, 2),
        "distinct_speakers": len(speakers),
        "speech_per_speaker": {k: round(v, 2) for k, v in totals.items()},
        "segments": [{"speaker": s.local_speaker, "start": round(s.start, 2),
                      "end": round(s.end, 2)} for s in segs],
    }, indent=2))
    print(f"  {stem} [{engine}]: {len(speakers)} speakers, {len(segs)} segments  "
          f"({', '.join(f'{k}={v:.0f}s' for k, v in sorted(totals.items(), key=lambda kv: -kv[1]))})")
    return {"stem": stem, "speakers": len(speakers)}


def dump_run(run_dir, stems=None, engine="diart"):
    run_dir = Path(run_dir)
    out_dir = run_dir / "results" / "diarization"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem in (stems or DEFAULT_STEMS):
        if not (run_dir / "segments" / f"{stem}.wav").exists():
            raise SystemExit(f"no such segment: {stem} (run fp_split first)")
        print(f"→ diarizing {stem} with {engine} …", flush=True)
        dump_piece(run_dir / "segments", stem, out_dir, engine)
    print("\n→ results/diarization/ written")


def main() -> None:
    ap = argparse.ArgumentParser(prog="eval_harness.fp_diarize")
    ap.add_argument("run_dir")
    ap.add_argument("stems", nargs="*", help="segment stems (default: all three)")
    ap.add_argument("--engine", choices=["diart", "diarizen"], default="diart")
    args = ap.parse_args()
    dump_run(args.run_dir, args.stems or None, args.engine)


if __name__ == "__main__":
    main()
