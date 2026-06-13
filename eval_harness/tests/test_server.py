"""C11 — record→save→run web UI. Upload/list run anywhere (just ffmpeg); the run endpoint is gated
on the models (faster-whisper + diart + HF token). EXPERIMENTS is monkeypatched to a tmp dir so the
test never touches the real experiments/ folder."""
import io
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import eval_harness.server as server  # noqa: E402

SR = 16_000
ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "speakers"


@pytest.fixture
def client(tmp_path, monkeypatch):
    exp = tmp_path / "demo"
    exp.mkdir()
    (exp / "config.yaml").write_text(
        "name: demo\nmode: offline\n"
        "asr: { model: tiny, compute_type: int8, device: cpu, vocab_compare: false }\n"
        "diarizer: { engine: diart, window_sec: 5, step_sec: 0.5 }\n"
    )
    (exp / "gold.txt").write_text("A: alpha bravo\nB: charlie delta\n")
    monkeypatch.setattr(server, "EXPERIMENTS", tmp_path)
    return TestClient(server.app)


def _wav_bytes(seconds=1.0):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(SR * seconds), dtype="float32"), SR, format="WAV")
    return buf.getvalue()


def test_lists_experiments(client):
    r = client.get("/api/experiments")
    assert r.status_code == 200 and r.json()["experiments"] == ["demo"]


def test_upload_saves_canonical_wav(client, tmp_path):
    r = client.post("/record/demo/upload",
                    files={"file": ("rec.wav", _wav_bytes(1.0), "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] and body["duration_sec"] == pytest.approx(1.0, abs=0.1)
    saved = tmp_path / "demo" / "audio.wav"
    assert saved.exists()
    data, sr = sf.read(saved)
    assert sr == SR and data.ndim == 1            # 16k mono


def test_unknown_experiment_404s(client):
    r = client.post("/record/nope/upload",
                    files={"file": ("rec.wav", _wav_bytes(0.2), "audio/wav")})
    assert r.status_code == 404


def test_run_without_audio_400s(client):
    assert client.post("/record/demo/run").status_code == 400


@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="run endpoint needs models + HF_TOKEN")
def test_run_returns_metrics(client, tmp_path):
    clip = np.concatenate([sf.read(FIX / f"{n}.wav", dtype="float32")[0]
                           for n in ["spkA_1", "spkB_1", "spkA_2", "spkB_1"]])
    buf = io.BytesIO(); sf.write(buf, clip, SR, format="WAV")
    up = client.post("/record/demo/upload",
                     files={"file": ("rec.wav", buf.getvalue(), "audio/wav")})
    assert up.status_code == 200, up.text
    r = client.post("/record/demo/run")
    assert r.status_code == 200, r.text
    j = r.json()
    for k in ("wer", "speaker_accuracy", "rtf", "speakers_detected", "peak_rss_mb"):
        assert k in j
