"""Batch DiariZen DER anchor on AMI — reproduce the bake-off number (~22.8% strict / ~14% lenient).

Runs full-batch DiariZenDiarizer over the AMI single-mic mix (IS1009a, ES2004a) and scores against
the reference RTTM under BOTH protocols:
  - strict  : collar 0,    overlap scored      (the honest number)
  - lenient : collar 0.25, overlap skipped      (≈ DiariZen's published protocol)
Aggregate DER is recomputed from summed components (not averaged), per file.

Requires DiariZen → run in the diarizen venv:
    HF_TOKEN=... PYTHONPATH=. /tmp/diarizen-venv/bin/python -m evaluation.diarizen_eval
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import soundfile as sf
from pyannote.core import Annotation
from pyannote.core import Segment as PSegment

from evaluation.der import compute_der
from evaluation.rttm import parse_rttm
from eval_harness.harness.diarizen_engine import DiariZenDiarizer

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval_data"
RTTM_DIR = EVAL / "AMI-diarization-setup" / "only_words" / "rttms" / "test"
MEETINGS = ["IS1009a", "ES2004a"]
SR = 16_000


def _peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 ** 2 if sys.platform == "darwin" else 1024), 1)


def diarize(wav: Path, uri: str) -> tuple[Annotation, float, float]:
    audio, sr = sf.read(wav, dtype="float32")
    assert sr == SR
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio_len = len(audio) / SR
    d = DiariZenDiarizer()
    d.start(uri)
    t0 = time.perf_counter()
    step = int(2.0 * SR)
    for i in range(0, len(audio), step):
        d.feed(audio[i : i + step], SR)
    segs = d.finish()
    elapsed = time.perf_counter() - t0
    ann = Annotation(uri=uri)
    for i, s in enumerate(segs):
        ann[PSegment(s.start, s.end), i] = s.local_speaker
    return ann, audio_len, elapsed


def _agg(rows: dict, key_pref: str) -> float:
    miss = sum(r[f"{key_pref}_missed"] for r in rows.values())
    fa = sum(r[f"{key_pref}_false_alarm"] for r in rows.values())
    conf = sum(r[f"{key_pref}_confusion"] for r in rows.values())
    total = sum(r[f"{key_pref}_total"] for r in rows.values())
    return (miss + fa + conf) / total if total else 0.0


def main() -> dict:
    rows: dict[str, dict] = {}
    for m in MEETINGS:
        ref = parse_rttm(RTTM_DIR / f"{m}.rttm")[m]
        hyp, audio_len, elapsed = diarize(EVAL / f"{m}.16k.wav", m)
        strict = compute_der(ref, hyp, collar=0.0, skip_overlap=False)
        lenient = compute_der(ref, hyp, collar=0.25, skip_overlap=True)
        rows[m] = {
            "audio_sec": round(audio_len, 1), "diarize_sec": round(elapsed, 1),
            "rtf": round(elapsed / audio_len, 3),
            "n_spk": len(hyp.labels()),
            "strict_der": strict["der"], "strict_missed": strict["missed"],
            "strict_false_alarm": strict["false_alarm"], "strict_confusion": strict["confusion"],
            "strict_total": strict["total"],
            "lenient_der": lenient["der"], "lenient_missed": lenient["missed"],
            "lenient_false_alarm": lenient["false_alarm"], "lenient_confusion": lenient["confusion"],
            "lenient_total": lenient["total"],
        }
        print(f"{m}: strict DER={strict['der']*100:.1f}%  lenient DER={lenient['der']*100:.1f}%  "
              f"spk={len(hyp.labels())}  RTF={rows[m]['rtf']}  ({audio_len/60:.1f}min)")

    out = {
        "engine": "diarizen-batch", "model": "BUT-FIT/diarizen-wavlm-large-s80-md",
        "meetings": MEETINGS,
        "aggregate_strict_der": round(_agg(rows, "strict"), 4),
        "aggregate_lenient_der": round(_agg(rows, "lenient"), 4),
        "peak_rss_mb": _peak_rss_mb(),
        "per_file": rows,
    }
    print(f"\nAGGREGATE DiariZen-batch:  strict DER = {out['aggregate_strict_der']*100:.1f}%   "
          f"lenient DER = {out['aggregate_lenient_der']*100:.1f}%   peak RAM = {out['peak_rss_mb']} MB")
    out_dir = EVAL / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "diarizen_anchor.json").write_text(json.dumps(out, indent=2))
    print(f"→ eval_data/results/diarizen_anchor.json written")
    return out


if __name__ == "__main__":
    main()
