"""Corpus-level DER harness — grades a hypothesis RTTM against a reference RTTM.

This is the bake-off decision instrument (plan M2): each diarization candidate
emits a hypothesis RTTM; this scores it (per-file + duration-weighted aggregate)
under the strict protocol. Aggregate components are summed across files so the
aggregate DER is the true corpus-level number, not a mean of per-file rates.

CLI:
    python -m evaluation.harness --reference ref.rttm --hypothesis hyp.rttm [--json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyannote.core import Annotation

from evaluation.der import compute_der
from evaluation.rttm import parse_rttm


def evaluate(
    pairs: dict[str, tuple[Annotation, Annotation]],
    collar: float = 0.0,
    skip_overlap: bool = False,
) -> dict:
    """Score {uri: (reference, hypothesis)} → per-file rows + aggregate."""
    per_file: dict[str, dict[str, float]] = {}
    agg = {"missed": 0.0, "false_alarm": 0.0, "confusion": 0.0, "total": 0.0}
    for uri, (reference, hypothesis) in pairs.items():
        row = compute_der(reference, hypothesis, collar=collar, skip_overlap=skip_overlap)
        per_file[uri] = row
        for key in agg:
            agg[key] += row[key]
    agg_der = (
        (agg["missed"] + agg["false_alarm"] + agg["confusion"]) / agg["total"]
        if agg["total"] > 0
        else 0.0
    )
    return {"per_file": per_file, "aggregate": {"der": agg_der, **agg}}


def evaluate_rttm_files(
    reference_path: str | Path,
    hypothesis_path: str | Path,
    collar: float = 0.0,
    skip_overlap: bool = False,
) -> dict:
    """Score two RTTM files. Missing hypotheses for a referenced URI score as all-missed."""
    references = parse_rttm(reference_path)
    hypotheses = parse_rttm(hypothesis_path)
    pairs = {
        uri: (ref, hypotheses.get(uri, Annotation(uri=uri)))
        for uri, ref in references.items()
    }
    return evaluate(pairs, collar=collar, skip_overlap=skip_overlap)


def _format(result: dict) -> str:
    lines = [f"{'uri':<28} {'DER':>7} {'miss':>8} {'FA':>8} {'conf':>8} {'total':>9}"]
    for uri, r in result["per_file"].items():
        lines.append(
            f"{uri[:28]:<28} {r['der']*100:>6.2f}% {r['missed']:>8.1f} "
            f"{r['false_alarm']:>8.1f} {r['confusion']:>8.1f} {r['total']:>9.1f}"
        )
    a = result["aggregate"]
    lines.append("-" * 74)
    lines.append(
        f"{'AGGREGATE':<28} {a['der']*100:>6.2f}% {a['missed']:>8.1f} "
        f"{a['false_alarm']:>8.1f} {a['confusion']:>8.1f} {a['total']:>9.1f}"
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="DER harness (no collar, overlap scored).")
    ap.add_argument("--reference", required=True, help="reference RTTM")
    ap.add_argument("--hypothesis", required=True, help="hypothesis RTTM")
    ap.add_argument("--collar", type=float, default=0.0)
    ap.add_argument("--skip-overlap", action="store_true")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    result = evaluate_rttm_files(
        args.reference, args.hypothesis, collar=args.collar, skip_overlap=args.skip_overlap
    )
    print(json.dumps(result, indent=2) if args.json else _format(result))


if __name__ == "__main__":
    main()
