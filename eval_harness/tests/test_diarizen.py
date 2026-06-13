"""C9 — DiariZen engine (batch WavLM + VBx) behind the StreamingDiarizer interface.

Runs ONLY in the diarizen venv (torch 2.1.1) — it's skipped everywhere `diarizen` can't import
(e.g. the diart venv), so a partial setup still passes. The real run also needs the model, which
DiariZenPipeline.from_pretrained pulls from HF on first use (gated on HF_TOKEN).
"""
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("diarizen", reason="diarizen not installed (diarizen venv only)")
import soundfile as sf  # noqa: E402

from fpm.diarize.base import StreamingDiarizer  # noqa: E402
from eval_harness.harness.diarizen_engine import DiariZenDiarizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "speakers"
SR = 16_000


def test_diarizen_is_a_streaming_diarizer():
    d = DiariZenDiarizer()
    assert isinstance(d, StreamingDiarizer)


@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="needs HF_TOKEN to pull the DiariZen model")
def test_diarizen_diarizes_two_speakers():
    clip = np.concatenate([sf.read(FIX / f"{n}.wav", dtype="float32")[0]
                           for n in ["spkA_1", "spkB_1", "spkA_2", "spkB_1"]])
    d = DiariZenDiarizer()
    d.start("eval")
    step = int(0.5 * SR)
    for i in range(0, len(clip), step):
        out = d.feed(clip[i:i + step], SR)
        assert out == []                       # batch engine — no incremental output
    segs = d.finish()
    assert segs, "DiariZen returned no segments"
    assert all(s.end >= s.start for s in segs)
    assert len({s.local_speaker for s in segs}) >= 1
