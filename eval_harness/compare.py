"""One-command compare — diart vs DiariZen on the SAME experiment.

    python -m eval_harness.compare experiments/<name>

diart (torch 2.2.2) and DiariZen (torch 2.1.1) can't share a venv, so this orchestrates by
SUBPROCESS: it runs `run.py --engine diart` in the diart venv and `--engine diarizen` in the
diarizen venv, each writing `results/<engine>/result.json`, then reads both back and prints a
side-by-side (WER, speaker accuracy, RTF, peak RAM, speakers) + writes `results/compare.json` with
the DiariZen-over-diart deltas. No single process ever imports both engines.

Venv pythons are overridable:  EVAL_DIART_PY / EVAL_DIARIZEN_PY
(defaults: /tmp/diart-venv/bin/python, /tmp/diarizen-venv/bin/python).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]      # FPM/ — PYTHONPATH for the subprocess
DEFAULT_PY = {
    "diart": os.environ.get("EVAL_DIART_PY", "/tmp/diart-venv/bin/python"),
    "diarizen": os.environ.get("EVAL_DIARIZEN_PY", "/tmp/diarizen-venv/bin/python"),
}

# (json key, label, how to render, lower-is-better) — drives both the table and the deltas.
_FIELDS = [
    ("wer", "WER (vocab-on)", "pct", True),
    ("wer_vocab_off", "WER (vocab-off)", "pct", True),
    ("speaker_accuracy", "speaker accuracy", "pct", False),
    ("rtf", "RTF", "num", True),
    ("total_sec", "latency (s)", "num", True),
    ("peak_rss_mb", "peak RAM (MB)", "num", True),
    ("speakers_detected", "speakers detected", "int", False),
]


def _run_engine(exp_dir: str, engine: str) -> Path:
    """Subprocess-run the experiment through one engine's venv; return its result.json path."""
    py = DEFAULT_PY[engine]
    if not Path(py).exists():
        raise SystemExit(f"{engine} venv python not found: {py} "
                         f"(set EVAL_{engine.upper()}_PY, or build the venv)")
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    print(f"→ running {engine} ({py}) …", flush=True)
    proc = subprocess.run([py, "-m", "eval_harness.run", exp_dir, "--engine", engine],
                          cwd=str(REPO_ROOT), env=env)
    if proc.returncode != 0:
        raise SystemExit(f"{engine} run failed (exit {proc.returncode}) — see output above")
    return REPO_ROOT / exp_dir / "results" / engine / "result.json"


def _fmt(value, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "int":
        return f"{int(value)}"
    return f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"


def build_compare(diart: dict, diarizen: dict) -> dict:
    """Pure: assemble the side-by-side + DiariZen-over-diart deltas (no I/O, no models)."""
    rows = []
    for key, label, kind, lower_better in _FIELDS:
        a, b = diart.get(key), diarizen.get(key)
        delta = (b - a) if (isinstance(a, (int, float)) and isinstance(b, (int, float))) else None
        improved = None
        if delta is not None and delta != 0:
            improved = (delta < 0) if lower_better else (delta > 0)
        rows.append({"key": key, "label": label, "kind": kind,
                     "diart": a, "diarizen": b, "delta": delta, "diarizen_better": improved})
    return {
        "experiment": diart.get("name") or diarizen.get("name"),
        "diart": diart,
        "diarizen": diarizen,
        "rows": rows,
    }


def format_table(cmp: dict) -> str:
    lines = [f"\n=== compare: {cmp['experiment']} — diart vs DiariZen ===",
             f"  {'metric':<20} {'diart':>12} {'diarizen':>12} {'Δ (dzn−diart)':>16}"]
    for r in cmp["rows"]:
        a, b = _fmt(r["diart"], r["kind"]), _fmt(r["diarizen"], r["kind"])
        if r["delta"] is None:
            d = "—"
        else:
            mark = "" if r["diarizen_better"] is None else ("  ✓" if r["diarizen_better"] else "  ✗")
            d = (_fmt(r["delta"], r["kind"]) if r["kind"] != "pct"
                 else f"{r['delta'] * 100:+.1f} pts") + mark
        lines.append(f"  {r['label']:<20} {a:>12} {b:>12} {d:>16}")
    lines.append("  (✓ = DiariZen better on that metric; ✗ = worse)\n")
    return "\n".join(lines)


def compare_experiment(exp_dir: str) -> dict:
    diart_json = _run_engine(exp_dir, "diart")
    diarizen_json = _run_engine(exp_dir, "diarizen")
    diart = json.loads(diart_json.read_text())
    diarizen = json.loads(diarizen_json.read_text())
    cmp = build_compare(diart, diarizen)
    out_path = REPO_ROOT / exp_dir / "results" / "compare.json"
    out_path.write_text(json.dumps(cmp, indent=2))
    print(format_table(cmp))
    print(f"  → {out_path.relative_to(REPO_ROOT)} written\n")
    return cmp


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m eval_harness.compare experiments/<name>")
    compare_experiment(sys.argv[1])


if __name__ == "__main__":
    main()
