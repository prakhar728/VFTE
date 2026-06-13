"""Dump the hypothesis transcript for one or more pieces of a fingerprint-persistence run.

    /tmp/diart-venv/bin/python -m eval_harness.fp_transcribe <run> [seg_stem ...]

Transcribes the named `segments/<stem>.wav` (default: the enroll + 3-min identify pieces — the
gold-able ones) with the exact Recato Whisper, and writes:
  results/transcripts/<stem>.txt    plain text (for WER vs a gold you paste in later)
  results/transcripts/<stem>.json   words + per-segment timestamps (for span-aligned partial gold)
No diarization here — this is the what-was-said artifact; latency/RTF lives in fp_latency.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import soundfile as sf

from eval_harness.harness.asr import WhisperASR

SR = 16_000
DEFAULT_STEMS = ["00_enroll_000-060s", "01_test_060-240s"]   # the short, gold-able pieces


def transcribe_segment(asr: WhisperASR, wav: Path, out_dir: Path) -> dict:
    audio, sr = sf.read(wav, dtype="float32")
    if sr != SR:
        raise SystemExit(f"{wav.name} is {sr} Hz, expected {SR}")
    res = asr.transcribe(audio, vocab=None)
    (out_dir / f"{wav.stem}.txt").write_text(res.text.strip() + "\n")
    (out_dir / f"{wav.stem}.json").write_text(json.dumps({
        "segment": wav.name,
        "language": res.language,
        "text": res.text.strip(),
        "words": [{"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)}
                  for w in res.words],
    }, indent=2))
    return {"segment": wav.name, "words": len(res.words), "chars": len(res.text.strip())}


def transcribe_run(run_dir: str | Path, stems: list[str] | None = None) -> list[dict]:
    run_dir = Path(run_dir)
    seg_dir = run_dir / "segments"
    stems = stems or DEFAULT_STEMS
    out_dir = run_dir / "results" / "transcripts"
    out_dir.mkdir(parents=True, exist_ok=True)

    asr = WhisperASR(model="large-v3-turbo", device="cpu", compute_type="int8")
    rows = []
    for stem in stems:
        wav = seg_dir / f"{stem}.wav"
        if not wav.exists():
            raise SystemExit(f"no such segment: {wav} (run fp_split first; check the stem)")
        print(f"→ transcribing {wav.name} …", flush=True)
        r = transcribe_segment(asr, wav, out_dir)
        print(f"   {r['words']} words, {r['chars']} chars → results/transcripts/{stem}.txt")
        rows.append(r)
    return rows


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m eval_harness.fp_transcribe <run> [seg_stem ...]")
    transcribe_run(sys.argv[1], sys.argv[2:] or None)


if __name__ == "__main__":
    main()
