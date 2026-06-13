"""Split a fingerprint-persistence run's source recording into its metadata-defined pieces.

    python -m eval_harness.fp_split eval_harness/fingerprint_runs/<run>

Reads `<run>/metadata.yaml` → `segmentation.segments` (name + start_sec + end_sec; null end = EOF),
finds the single audio file in `<run>/source/`, and writes each piece as a 16 kHz mono WAV to
`<run>/segments/`. ffmpeg does the cut + downmix/resample. Idempotent (overwrites).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".mp4", ".webm", ".mov"}
TARGET_SR = 16_000


def find_source(run_dir: Path) -> Path:
    """The single audio file dropped in <run>/source/ (ignores .md markers)."""
    src = run_dir / "source"
    cands = [p for p in src.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXT] \
        if src.is_dir() else []
    if not cands:
        raise SystemExit(f"no audio in {src}/ — drop the recording there (see DROP-ORIGINAL-AUDIO-HERE.md)")
    if len(cands) > 1:
        raise SystemExit(f"more than one audio file in {src}/: {[c.name for c in cands]} — keep just one")
    return cands[0]


def load_segments(run_dir: Path) -> list[dict]:
    meta = yaml.safe_load((run_dir / "metadata.yaml").read_text()) or {}
    segs = (meta.get("segmentation") or {}).get("segments") or []
    if not segs:
        raise SystemExit("metadata.yaml has no segmentation.segments")
    return segs


def ffmpeg_cut(src: Path, start_sec: float, end_sec: float | None, out: Path) -> None:
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-ss", str(start_sec)]
    if end_sec is not None:
        cmd += ["-to", str(end_sec)]            # absolute end (since -ss is before -i? no — after -i = accurate)
    cmd += ["-ac", "1", "-ar", str(TARGET_SR), str(out)]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg failed cutting {out.name}: {proc.stderr.decode('utf-8','replace')[-300:]}")


def split_run(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    src = find_source(run_dir)
    segs = load_segments(run_dir)
    out_dir = run_dir / "segments"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    print(f"source: {src.relative_to(run_dir)}  →  {len(segs)} pieces")
    for s in segs:
        start, end = float(s["start_sec"]), s.get("end_sec")
        end_f = None if end is None else float(end)
        out = out_dir / f"{s['name']}.wav"
        ffmpeg_cut(src, start, end_f, out)
        span = f"{start:.0f}-{'EOF' if end_f is None else f'{end_f:.0f}'}s"
        print(f"  ✓ {out.name:28} [{s.get('role','?'):8}] {span}")
        written.append(out)
    print(f"→ {out_dir.relative_to(run_dir.parent)}/  ({len(written)} files)")
    return written


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m eval_harness.fp_split eval_harness/fingerprint_runs/<run>")
    split_run(sys.argv[1])


if __name__ == "__main__":
    main()
