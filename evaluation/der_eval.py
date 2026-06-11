"""E.2 — diart DER baseline (the bar a future lean ONNX engine must meet).

Runs the shipped DiartDiarizer over the AMI single-mic mix and scores its
diarization against the reference RTTM under the strict protocol (no collar,
overlap scored). This measures diart *as we use it* (streaming, merged spans),
not a clean offline batch. Requires diart → run in the diart venv:

    HF_TOKEN=... PYTHONPATH=. /tmp/diart-venv/bin/python -m evaluation.der_eval
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from pyannote.core import Annotation
from pyannote.core import Segment as PSegment

from evaluation.der import compute_der
from evaluation.rttm import parse_rttm
from fpm.diarize.diart_engine import DiartDiarizer

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval_data"
RTTM_DIR = EVAL / "AMI-diarization-setup" / "only_words" / "rttms" / "test"
MEETINGS = ["IS1009a", "ES2004a"]
SR = 16_000


def diarize(wav: Path, uri: str) -> Annotation:
    audio, sr = sf.read(wav, dtype="float32")
    assert sr == SR
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    d = DiartDiarizer(offline=True)
    d.start(uri)
    segs, step = [], int(0.5 * SR)
    for i in range(0, len(audio), step):
        segs.extend(d.feed(audio[i : i + step], SR))
    segs.extend(d.finish())
    ann = Annotation(uri=uri)
    for i, s in enumerate(segs):
        ann[PSegment(s.start, s.end), i] = s.local_speaker
    return ann


def main() -> dict:
    agg = {"missed": 0.0, "false_alarm": 0.0, "confusion": 0.0, "total": 0.0}
    rows = {}
    for m in MEETINGS:
        ref = parse_rttm(RTTM_DIR / f"{m}.rttm")[m]
        hyp = diarize(EVAL / f"{m}.16k.wav", m)
        row = compute_der(ref, hyp)
        rows[m] = row
        for k in agg:
            agg[k] += row[k]
        print(f"{m}: DER={row['der']*100:.1f}%  miss={row['missed']:.0f} "
              f"FA={row['false_alarm']:.0f} conf={row['confusion']:.0f} total={row['total']:.0f}")
    agg_der = (agg["missed"] + agg["false_alarm"] + agg["confusion"]) / agg["total"]
    print(f"AGGREGATE diart DER = {agg_der*100:.1f}%")
    return {"per_file": rows, "aggregate_der": agg_der}


if __name__ == "__main__":
    main()
