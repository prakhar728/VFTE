"""Latency-vs-duration over a fingerprint-persistence run's growing pieces.

    /tmp/diart-venv/bin/python -m eval_harness.fp_latency eval_harness/fingerprint_runs/<run>

Runs each `segments/*.wav` through the SAME pipeline as the eval harness (Whisper + diart, no gold —
this is a timing/identity run, not WER), records asr/diarize/total latency + RTF per piece, writes
`results/latency.json`, and plots `results/latency.png` (latency + RTF vs clip duration). The 1/3/6-min
pieces give the curve: does processing time grow linearly, and does RTF stay < 1 (faster than real time)?

peak_rss_mb is the process cumulative peak after each piece (model loads once and is reused), so it
reflects the footprint up to and including that piece — a reasonable upper bound, not an isolated measure.
"""
from __future__ import annotations

import json
import resource
import sys
from pathlib import Path

import soundfile as sf

from eval_harness.harness.config import AsrConfig, DiarizerConfig, ExperimentConfig
from eval_harness.harness.metrics import offline_metrics
from eval_harness.harness.pipeline import run_offline

SR = 16_000


def _peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 ** 2 if sys.platform == "darwin" else 1024), 1)


def _cfg_for(segment: Path) -> ExperimentConfig:
    """A minimal harness config: real Recato Whisper + diart 5s, single pass, no vocab/gold."""
    return ExperimentConfig(
        name=segment.stem, dir=segment.parent,
        asr=AsrConfig(model="large-v3-turbo", compute_type="int8", device="cpu",
                      vocab=[], vocab_compare=False),
        diarizer=DiarizerConfig(engine="diart", window_sec=5.0, step_sec=0.5),
    )


def measure_segment(segment: Path) -> dict:
    audio, sr = sf.read(segment, dtype="float32")
    if sr != SR:                                  # segments are pre-resampled, but be safe
        raise SystemExit(f"{segment.name} is {sr} Hz, expected {SR}")
    res = run_offline(_cfg_for(segment), audio)
    m = offline_metrics(res.audio_len_sec, res.asr_sec, res.diarize_sec)
    return {
        "segment": segment.name,
        "duration_sec": round(res.audio_len_sec, 2),
        "duration_min": round(res.audio_len_sec / 60, 2),
        **m.as_dict(),                            # asr_sec, diarize_sec, total_sec, rtf, mode
        "peak_rss_mb": _peak_rss_mb(),
        "speakers_detected": len({s.local_speaker for s in res.speaker_segments}),
    }


def plot_latency(rows: list[dict], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["duration_sec"])
    x = [r["duration_min"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax1.plot(x, [r["asr_sec"] for r in rows], "o-", label="ASR (Whisper)")
    ax1.plot(x, [r["diarize_sec"] for r in rows], "s-", label="diarize (diart)")
    ax1.plot(x, [r["total_sec"] for r in rows], "^-", label="total", linewidth=2)
    ax1.set_ylabel("processing time (s)")
    pieces = " / ".join(f"{r['duration_min']:g}" for r in rows)
    ax1.set_title(f"Latency vs clip duration ({pieces}-min pieces)")
    ax1.grid(True, alpha=0.3); ax1.legend()

    ax2.plot(x, [r["rtf"] for r in rows], "D-", color="#b8423a", label="RTF (total / audio len)")
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="real-time (RTF = 1)")
    ax2.set_xlabel("clip duration (minutes)"); ax2.set_ylabel("RTF")
    ax2.grid(True, alpha=0.3); ax2.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def run_latency(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    segs = sorted((run_dir / "segments").glob("*.wav"))
    if not segs:
        raise SystemExit(f"no segments in {run_dir}/segments/ — run fp_split first")
    rows = []
    for seg in segs:
        print(f"→ {seg.name} …", flush=True)
        r = measure_segment(seg)
        print(f"   {r['duration_min']}min: asr {r['asr_sec']}s + diarize {r['diarize_sec']}s "
              f"= {r['total_sec']}s (RTF {r['rtf']}), {r['speakers_detected']} speakers")
        rows.append(r)

    out_dir = run_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latency.json").write_text(json.dumps(rows, indent=2))
    plot_latency(rows, out_dir / "latency.png")
    print(f"\n→ results/latency.json + results/latency.png written")
    return {"rows": rows}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m eval_harness.fp_latency eval_harness/fingerprint_runs/<run>")
    run_latency(sys.argv[1])


if __name__ == "__main__":
    main()
