"""Branch A / step 3 — end-to-end /v1/diarize on the real DiariZen engine (C2 SHAPE).

Gated: runs ONLY where `diarizen` imports (the diarizen venv) and the CAM++ embedder model is
present; skipped everywhere else (e.g. the torch-free core venv), so this file is green by
skipping there. It asserts the C2 contract SHAPE only — NDJSON per-segment keys + a final
transcript line. Identity *correctness* across a long clip depends on the SessionIdentifier
buffer adaptation for batch engines (docs/build/branch-A-engine.md §6) and is owned by the
A+C joint integration gate, not by Branch A.
"""
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("diarizen", reason="diarizen not installed (diarizen venv only)")

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from auth import Caller, TokenAuth  # noqa: E402
from fpm.diarize.diarizen_engine import DiariZenDiarizer  # noqa: E402
from fpm.embed.onnx_embedder import OnnxSpeakerEmbedder  # noqa: E402
from fpm.store.store import VoiceprintStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "campplus.onnx"
FIX = Path(__file__).parent / "fixtures" / "speakers"
SR = 16_000
RECATO = "tok-recato"

pytestmark = pytest.mark.skipif(
    not MODEL.exists(), reason="embedder model missing — run scripts/fetch_models.sh"
)

_TURNS = ["spkA_1", "spkB_1", "spkA_2", "spkB_1"]


def _clip_bytes():
    parts = []
    for name in _TURNS:
        a, sr = sf.read(FIX / f"{name}.wav", dtype="float32")
        assert sr == SR
        parts.append(a if a.ndim == 1 else a.mean(axis=1))
    buf = io.BytesIO()
    sf.write(buf, np.concatenate(parts), SR, format="WAV")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    main.app.state.store = VoiceprintStore(db_path=tmp_path / "vp.db", key=os.urandom(32)).open()
    main.app.state.embedder = OnnxSpeakerEmbedder(MODEL).load()
    main.app.state.auth = TokenAuth({RECATO: Caller("recato", frozenset({"diarize"}))})
    main.app.state.diarizer_factory = lambda: DiariZenDiarizer()   # the real engine
    with TestClient(main.app) as c:
        c._clip = _clip_bytes()
        yield c
    for attr in ("store", "embedder", "auth", "diarizer_factory"):
        setattr(main.app.state, attr, None)


def _diarize(client):
    return client.post(
        "/v1/diarize",
        files={"file": ("clip.wav", client._clip, "audio/wav")},
        data={"workspace": "ws1", "tag": "offline"},
        headers={"Authorization": f"Bearer {RECATO}"},
    )


def _parse_ndjson(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_diarizen_streams_c2_shape(client):
    r = _diarize(client)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    rows = _parse_ndjson(r.text)
    segs = [x for x in rows if x.get("type") != "transcript"]
    assert segs, "no segments streamed"
    for s in segs:
        # C2 per-segment shape (main.py:_segment_dict)
        assert {"start", "end", "voiceprint_id", "name", "decision",
                "confidence", "local_speaker"} <= s.keys()
        assert s["end"] > s["start"]


def test_diarizen_final_transcript_line(client):
    rows = _parse_ndjson(_diarize(client).text)
    assert rows[-1].get("type") == "transcript"
    assert rows[-1]["segments"]
