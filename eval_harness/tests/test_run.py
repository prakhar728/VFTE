"""C6 — end-to-end runner. Gated on models (faster-whisper + diart + HF token); uses 'tiny' + a
2-speaker fixture clip. Verifies the runner wires everything and writes results."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("faster_whisper")
pytest.importorskip("diart")

from eval_harness.run import run_experiment  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "speakers"
SR = 16_000


@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="needs HF_TOKEN for diart models")
def test_runner_end_to_end(tmp_path):
    # build a tmp experiment: 2-speaker clip + placeholder gold + tiny-model config
    clip = np.concatenate([sf.read(FIX / f"{n}.wav", dtype="float32")[0]
                           for n in ["spkA_1", "spkB_1", "spkA_2", "spkB_1"]])
    sf.write(tmp_path / "audio.wav", clip, SR)
    (tmp_path / "gold.txt").write_text("A: alpha bravo\nB: charlie delta\n")
    (tmp_path / "config.yaml").write_text(
        "name: e2e-smoke\nmode: offline\n"
        "asr: { model: tiny, compute_type: int8, device: cpu, vocab_compare: false }\n"
        "diarizer: { engine: diart, window_sec: 5, step_sec: 0.5 }\n"
    )

    out = run_experiment(tmp_path)

    assert (tmp_path / "results" / "result.json").exists()
    assert (tmp_path / "results" / "transcript.txt").exists()
    saved = json.loads((tmp_path / "results" / "result.json").read_text())
    for key in ("wer", "speaker_accuracy", "rtf", "audio_length_sec", "speakers_detected"):
        assert key in saved
    assert saved["audio_length_sec"] == pytest.approx(len(clip) / SR, abs=0.2)
    assert saved["speakers_detected"] >= 1
