"""Fingerprint-persistence test: enroll voiceprints from one piece, identify a LATER piece.

    HF_TOKEN=… /tmp/diart-venv/bin/python -m eval_harness.fp_identify <run> \
        [--enroll 01_test_060-240s] [--identify 02_test_240-end]

The product question made concrete: build a voiceprint per speaker from the ENROLL piece, then —
on a non-overlapping FUTURE piece — re-diarize, embed each speaker, and ask FPM's open-set matcher
whether they map back to the SAME enrolled prints. Uses the exact FPM pieces:
  - diart for the split (same engine as production / the latency run),
  - the fixed CAM++ ONNX embedder (`models/campplus.onnx`) that defines the voiceprint space,
  - `fpm.match.classify` (calibrated-cosine, MATCH/AMBIGUOUS/UNKNOWN/LOW tiers).
In-process (no FPM service): an offline bench doesn't need the encrypted store / auth.

Writes results/identify.json + results/identify_simmatrix.png and prints a verdict.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from config import ID_EMBEDDER_PATH, MATCH_ACCEPT, MATCH_REJECT, AMBIGUOUS_MARGIN
from fpm.embed.onnx_embedder import OnnxSpeakerEmbedder
from fpm.match import classify

from eval_harness.harness.diarize import make_diarizer

SR = 16_000
DEFAULT_ENROLL = "01_test_060-240s"
DEFAULT_IDENTIFY = "02_test_240-end"


def diarize(audio: np.ndarray, window_sec=5.0, step_sec=0.5) -> list:
    """Run diart over the clip, return finalized Segments (same feed loop as the pipeline)."""
    d = make_diarizer("diart", window_sec, step_sec)
    d.start("fp-eval")
    segs, step = [], max(1, int(step_sec * SR))
    for i in range(0, len(audio), step):
        segs.extend(d.feed(audio[i:i + step], SR))
    segs.extend(d.finish())
    return segs


def speaker_spans(segs: list) -> dict[str, list[tuple[float, float]]]:
    spans: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for s in segs:
        spans[s.local_speaker].append((s.start, s.end))
    return spans


def _concat(audio: np.ndarray, spans: list[tuple[float, float]]) -> np.ndarray:
    parts = [audio[int(a * SR):int(b * SR)] for a, b in spans]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def embed_speakers(embedder, audio: np.ndarray, segs: list) -> list[dict]:
    """Per local speaker: total speech, concatenated-span embedding. Sorted by speech desc."""
    out = []
    for spk, spans in speaker_spans(segs).items():
        total = sum(b - a for a, b in spans)
        clip = _concat(audio, spans)
        emb = embedder.extract(clip, SR)
        out.append({"local": spk, "total_sec": round(total, 1),
                    "n_spans": len(spans), "emb": emb})
    return sorted(out, key=lambda r: r["total_sec"], reverse=True)


def run_identify(run_dir, enroll_stem=DEFAULT_ENROLL, identify_stem=DEFAULT_IDENTIFY) -> dict:
    run_dir = Path(run_dir)
    seg_dir = run_dir / "segments"
    embedder = OnnxSpeakerEmbedder(ID_EMBEDDER_PATH).load()

    # ── ENROLL ──────────────────────────────────────────────────────────
    en_audio, sr = sf.read(seg_dir / f"{enroll_stem}.wav", dtype="float32")
    assert sr == SR
    en_spk = [s for s in embed_speakers(embedder, en_audio, diarize(en_audio)) if s["emb"] is not None]
    centroids = {f"E{i}": s["emb"] for i, s in enumerate(en_spk)}     # E0 = most talkative
    summary = ", ".join(f"E{i}={s['total_sec']}s" for i, s in enumerate(en_spk))
    print(f"enroll [{enroll_stem}]: {len(en_spk)} voiceprints ({summary})")

    # within-enroll separation: are the prints actually distinct people?
    enroll_pairsim = {}
    for i in range(len(en_spk)):
        for j in range(i + 1, len(en_spk)):
            enroll_pairsim[f"E{i}-E{j}"] = round(float(en_spk[i]["emb"] @ en_spk[j]["emb"]), 3)

    # ── IDENTIFY (future piece) ─────────────────────────────────────────
    id_audio, sr = sf.read(seg_dir / f"{identify_stem}.wav", dtype="float32")
    assert sr == SR
    id_spk = [s for s in embed_speakers(embedder, id_audio, diarize(id_audio)) if s["emb"] is not None]

    rows, sim_matrix = [], []
    for i, s in enumerate(id_spk):
        r = classify(s["emb"], centroids)
        sims = {vid: round(float(s["emb"] @ c), 3) for vid, c in centroids.items()}
        sim_matrix.append([sims[f"E{k}"] for k in range(len(centroids))])
        rows.append({"identify": f"I{i}", "local": s["local"], "total_sec": s["total_sec"],
                     "decision": r.decision, "matched": r.voiceprint_id,
                     "score": round(r.score, 3), "confidence": round(r.confidence, 3),
                     "sims": sims})

    # ── verdict ─────────────────────────────────────────────────────────
    matched_prints = [r["matched"] for r in rows if r["decision"] == "MATCH"]
    bijection = len(set(matched_prints)) == len(matched_prints) == len(centroids) and \
        set(matched_prints) == set(centroids)
    result = {
        "enroll": {"stem": enroll_stem, "voiceprints": len(centroids),
                   "speech_sec": {f"E{i}": s["total_sec"] for i, s in enumerate(en_spk)},
                   "pairwise_cos": enroll_pairsim},
        "identify": {"stem": identify_stem, "speakers": len(id_spk), "rows": rows},
        "thresholds": {"accept": MATCH_ACCEPT, "reject": MATCH_REJECT, "ambiguous_margin": AMBIGUOUS_MARGIN},
        "verdict": {"bijective_match": bijection,
                    "n_match": sum(r["decision"] == "MATCH" for r in rows),
                    "n_unknown": sum(r["decision"] == "UNKNOWN" for r in rows),
                    "n_ambiguous": sum(r["decision"] == "AMBIGUOUS" for r in rows),
                    "n_low": sum(r["decision"] == "LOW" for r in rows)},
    }
    _print_report(result)
    out_dir = run_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "identify.json").write_text(json.dumps(result, indent=2))
    _plot_simmatrix(rows, len(centroids), out_dir / "identify_simmatrix.png")
    print(f"\n→ results/identify.json + results/identify_simmatrix.png written")
    return result


def _print_report(r: dict) -> None:
    print(f"\n=== fingerprint persistence: enroll [{r['enroll']['stem']}] → identify [{r['identify']['stem']}] ===")
    print(f"  enrolled prints: {r['enroll']['voiceprints']}  speech {r['enroll']['speech_sec']}")
    print(f"  enroll separation (lower = more distinct people): {r['enroll']['pairwise_cos']}")
    print(f"  thresholds: accept ≥{r['thresholds']['accept']}, reject <{r['thresholds']['reject']}, "
          f"ambiguous margin {r['thresholds']['ambiguous_margin']}\n")
    hdr = f"  {'spk':4} {'speech':>7} {'decision':>10} {'→print':>7} {'cos':>6} {'conf':>6}   sims"
    print(hdr)
    for row in r["identify"]["rows"]:
        print(f"  {row['identify']:4} {row['total_sec']:>6.1f}s {row['decision']:>10} "
              f"{str(row['matched'] or '-'):>7} {row['score']:>6.3f} {row['confidence']:>6.3f}   {row['sims']}")
    v = r["verdict"]
    print(f"\n  verdict: {v['n_match']} MATCH / {v['n_ambiguous']} AMBIGUOUS / "
          f"{v['n_unknown']} UNKNOWN / {v['n_low']} LOW")
    print(f"  bijective (each future speaker → a distinct enrolled print, all prints covered): "
          f"{'YES ✓' if v['bijective_match'] else 'NO ✗'}")


def _plot_simmatrix(rows: list[dict], n_prints: int, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = np.array([[row["sims"][f"E{k}"] for k in range(n_prints)] for row in rows])
    fig, ax = plt.subplots(figsize=(1.6 + n_prints, 1.2 + 0.6 * len(rows)))
    im = ax.imshow(mat, cmap="viridis", vmin=-0.1, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n_prints)); ax.set_xticklabels([f"E{k}" for k in range(n_prints)])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([row["identify"] for row in rows])
    ax.set_xlabel("enrolled voiceprint"); ax.set_ylabel("future-piece speaker")
    ax.set_title("cosine: future speakers × enrolled prints")
    for i in range(len(rows)):
        for j in range(n_prints):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat[i, j] < 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(prog="eval_harness.fp_identify")
    ap.add_argument("run_dir")
    ap.add_argument("--enroll", default=DEFAULT_ENROLL)
    ap.add_argument("--identify", default=DEFAULT_IDENTIFY)
    args = ap.parse_args()
    run_identify(args.run_dir, args.enroll, args.identify)


if __name__ == "__main__":
    main()
