"""C3 — diarizer factory: diart (configurable window) + DiariZen stub. diart run needs the diart
venv + HF token; the fast checks run anywhere diart imports."""
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("diart", reason="diart not installed (eval/diart venv only)")
import soundfile as sf  # noqa: E402

from fpm.diarize.base import StreamingDiarizer  # noqa: E402
from eval_harness.harness.diarize import make_diarizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "speakers"
SR = 16_000


def test_window_is_configurable():
    d = make_diarizer("diart", window_sec=10.0, step_sec=0.5)
    assert isinstance(d, StreamingDiarizer)
    assert d._duration == 10.0 and d._step == 0.5      # window threaded into the diart config


def test_diarizen_builds():
    # DiariZen is now real (its actual model only loads in start(), inside the diarizen venv).
    # Here we just confirm the factory wires the engine and it satisfies the interface.
    d = make_diarizer("diarizen", window_sec=120)
    assert isinstance(d, StreamingDiarizer)
    assert type(d).__name__ == "DiariZenDiarizer"


def test_unknown_engine_raises():
    with pytest.raises(ValueError):
        make_diarizer("whisperx")


@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="needs HF_TOKEN for diart models")
def test_diart_diarizes_two_speakers():
    clip = np.concatenate([sf.read(FIX / f"{n}.wav", dtype="float32")[0]
                           for n in ["spkA_1", "spkB_1", "spkA_2", "spkB_1"]])
    d = make_diarizer("diart", window_sec=5.0, step_sec=0.5)
    d.start("eval")
    segs, step = [], int(0.5 * SR)
    for i in range(0, len(clip), step):
        segs.extend(d.feed(clip[i:i + step], SR))
    segs.extend(d.finish())
    assert segs and len({s.local_speaker for s in segs}) == 2
