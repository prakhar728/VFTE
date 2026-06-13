"""CLI runner — run one experiment folder offline.

    python -m eval_harness.run experiments/<name>
    python -m eval_harness.run experiments/<name> --engine diarizen

Reads config + audio + gold, runs Whisper + diarize → merge, scores WER (vocab-on vs -off) +
speaker attribution, writes result.json + transcript(s).txt, and prints a summary. `--engine`
overrides the config's diarizer (so the same folder runs through diart and DiariZen) and isolates
the output under `results/<engine>/` — that split is what `compare` reads back.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
from pathlib import Path

from eval_harness.harness.config import ExperimentConfig
from eval_harness.harness.metrics import offline_metrics
from eval_harness.harness.pipeline import load_audio, run_offline
from eval_harness.harness.scoring import gold_text, load_gold, speaker_accuracy, wer


def render_transcript(turns) -> str:
    return "\n".join(f"[{t.speaker}] {t.text}" for t in turns)


def _peak_rss_mb() -> float:
    """Process peak resident set, MB. ru_maxrss is bytes on macOS, kilobytes on Linux."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 ** 2 if sys.platform == "darwin" else 1024), 1)


def run_experiment(exp_dir: str | Path, engine: str | None = None) -> dict:
    cfg = ExperimentConfig.load(exp_dir)
    if engine:
        cfg.diarizer.engine = engine        # CLI override → run the SAME folder through any engine
    errs = cfg.validate()
    if errs:
        raise SystemExit("config errors:\n  - " + "\n  - ".join(errs))
    if not cfg.audio_path.exists():
        raise SystemExit(f"no recording yet — drop your audio at {cfg.audio_path}")
    if cfg.mode != "offline":
        raise SystemExit(f"mode '{cfg.mode}' not supported by the offline runner (real-time = C7)")

    audio = load_audio(cfg.audio_path)
    res = run_offline(cfg, audio)

    gold_turns = load_gold(cfg.gold_path)
    gt = gold_text(gold_turns)
    wer_on = wer(gt, res.asr.text)
    wer_off = wer(gt, res.asr_vocab_off.text) if res.asr_vocab_off else None
    spk = speaker_accuracy(gold_turns, res.turns)
    m = offline_metrics(res.audio_len_sec, res.asr_sec, res.diarize_sec)

    out = {
        "name": cfg.name,
        "config": {
            "model": cfg.asr.model, "vocab": cfg.asr.vocab,
            "diarizer": cfg.diarizer.engine, "window_sec": cfg.diarizer.window_sec,
            "step_sec": cfg.diarizer.step_sec, "mode": cfg.mode, "notes": cfg.notes,
        },
        **m.as_dict(),
        "peak_rss_mb": _peak_rss_mb(),
        "wer": round(wer_on, 4),
        "wer_vocab_off": round(wer_off, 4) if wer_off is not None else None,
        "wer_delta_vocab": round(wer_off - wer_on, 4) if wer_off is not None else None,
        "speaker_accuracy": round(spk["accuracy"], 4),
        "speaker_mapping": spk["mapping"],
        "speakers_detected": len({s.local_speaker for s in res.speaker_segments}),
    }

    # default → results/ ; with an engine override → results/<engine>/ (so compare can keep both).
    out_dir = cfg.results_dir / engine if engine else cfg.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(out, indent=2))
    (out_dir / "transcript.txt").write_text(render_transcript(res.turns) + "\n")
    if res.turns_vocab_off is not None:
        (out_dir / "transcript.vocab-off.txt").write_text(
            render_transcript(res.turns_vocab_off) + "\n")
    return out


def _print_summary(out: dict) -> None:
    print(f"\n=== {out['name']} ===")
    print(f"  audio {out['audio_length_sec']}s | mode {out['mode']} | "
          f"latency {out['total_sec']}s (RTF {out['rtf']})")
    print(f"  diarizer {out['config']['diarizer']} @ {out['config']['window_sec']}s window | "
          f"speakers detected: {out['speakers_detected']}")
    line = f"  WER: {out['wer']*100:.1f}% (vocab-on)"
    if out['wer_vocab_off'] is not None:
        line += (f"  vs  {out['wer_vocab_off']*100:.1f}% (vocab-off)  "
                 f"→ vocab Δ {out['wer_delta_vocab']*100:+.1f} pts")
    print(line)
    print(f"  speaker attribution accuracy: {out['speaker_accuracy']*100:.1f}%  "
          f"(map {out['speaker_mapping']})")
    print(f"  peak RAM: {out.get('peak_rss_mb', '?')} MB")
    print(f"  → results written\n")


def main() -> None:
    ap = argparse.ArgumentParser(prog="eval_harness.run")
    ap.add_argument("exp_dir", help="experiment folder, e.g. experiments/eval-conversation")
    ap.add_argument("--engine", choices=["diart", "diarizen"], default=None,
                    help="override config diarizer; writes to results/<engine>/")
    args = ap.parse_args()
    _print_summary(run_experiment(args.exp_dir, engine=args.engine))


if __name__ == "__main__":
    main()
