"""Dump the actual diarization timeline for pieces of a fingerprint-persistence run.

    HF_TOKEN=… /tmp/diart-venv/bin/python -m eval_harness.fp_diarize <run> [seg_stem ...]

Runs diart (same path as fp_latency / fp_identify) and WRITES the who-spoke-when segments so they're
inspectable — not just a speaker count. Per piece:
  results/diarization/<stem>.txt    readable: [speaker]  start–end  (dur)
  results/diarization/<stem>.json   segments + per-speaker totals + distinct-speaker count
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf

from eval_harness.harness.diarize import make_diarizer

SR = 16_000
DEFAULT_STEMS = ["00_enroll_000-060s", "01_test_060-240s", "02_test_240-end"]


def diarize_segments(audio, window_sec=5.0, step_sec=0.5):
    d = make_diarizer("diart", window_sec, step_sec)
    d.start("fp-diar")
    segs, step = [], max(1, int(step_sec * SR))
    for i in range(0, len(audio), step):
        segs.extend(d.feed(audio[i:i + step], SR))
    segs.extend(d.finish())
    return sorted(segs, key=lambda s: s.start)


def dump_piece(seg_dir: Path, stem: str, out_dir: Path) -> dict:
    audio, sr = sf.read(seg_dir / f"{stem}.wav", dtype="float32")
    if sr != SR:
        raise SystemExit(f"{stem} is {sr} Hz, expected {SR}")
    segs = diarize_segments(audio)
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
    (out_dir / f"{stem}.txt").write_text(header + "\n".join(lines) + "\n")
    (out_dir / f"{stem}.json").write_text(json.dumps({
        "stem": stem, "audio_sec": round(len(audio) / SR, 2),
        "distinct_speakers": len(speakers),
        "speech_per_speaker": {k: round(v, 2) for k, v in totals.items()},
        "segments": [{"speaker": s.local_speaker, "start": round(s.start, 2),
                      "end": round(s.end, 2)} for s in segs],
    }, indent=2))
    print(f"  {stem}: {len(speakers)} speakers, {len(segs)} segments  "
          f"({', '.join(f'{k}={v:.0f}s' for k, v in sorted(totals.items(), key=lambda kv: -kv[1]))})")
    return {"stem": stem, "speakers": len(speakers)}


def dump_run(run_dir, stems=None):
    run_dir = Path(run_dir)
    out_dir = run_dir / "results" / "diarization"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem in (stems or DEFAULT_STEMS):
        if not (run_dir / "segments" / f"{stem}.wav").exists():
            raise SystemExit(f"no such segment: {stem} (run fp_split first)")
        print(f"→ diarizing {stem} …", flush=True)
        dump_piece(run_dir / "segments", stem, out_dir)
    print("\n→ results/diarization/<stem>.txt + .json written")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m eval_harness.fp_diarize <run> [seg_stem ...]")
    dump_run(sys.argv[1], sys.argv[2:] or None)


if __name__ == "__main__":
    main()
