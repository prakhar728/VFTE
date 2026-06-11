"""E.2 — diart DER harness smoke test (full corpus run is manual; see docs/der-eval.md)."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("diart", reason="diart not installed (torch-free .venv); run in diart venv")
pytest.importorskip("pyannote.core")

from pyannote.core import Annotation  # noqa: E402

from evaluation.der_eval import diarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "campplus.onnx"
FIX = Path(__file__).parent / "fixtures" / "speakers"
SR = 16_000


@pytest.mark.skipif(not MODEL.exists(), reason="embedder model missing")
def test_diarize_builds_annotation(tmp_path):
    # short 2-speaker clip from fixtures → diart produces a scorable Annotation
    clip = np.concatenate([sf.read(FIX / f"{n}.wav", dtype="float32")[0]
                           for n in ["spkA_1", "spkB_1", "spkA_2", "spkB_1"]])
    wav = tmp_path / "clip.wav"
    sf.write(wav, clip, SR)
    ann = diarize(wav, "smoke")
    assert isinstance(ann, Annotation)
    assert len(ann.labels()) >= 1
    assert ann.get_timeline().duration() > 0
