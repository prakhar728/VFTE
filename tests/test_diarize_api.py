"""D.2 — POST /v1/diarize (offline streaming + gmeet->enroll routing)."""
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import main
from auth import Caller, TokenAuth
from fpm.diarize.base import Segment
from fpm.diarize.mock import MockDiarizer
from fpm.embed.onnx_embedder import OnnxSpeakerEmbedder
from fpm.store.store import VoiceprintStore

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "campplus.onnx"
FIX = Path(__file__).parent / "fixtures" / "speakers"
SR = 16_000
RECATO = "tok-recato"

pytestmark = pytest.mark.skipif(
    not MODEL.exists(), reason="embedder model missing — run scripts/fetch_models.sh"
)

_TURNS = ["spkA_1", "spkB_1", "spkA_2", "spkB_1", "spkA_1", "spkB_1"]
_LABELS = ["speaker0", "speaker1", "speaker0", "speaker1", "speaker0", "speaker1"]


def _wav_arr(name):
    a, sr = sf.read(FIX / f"{name}.wav", dtype="float32")
    assert sr == SR
    return a if a.ndim == 1 else a.mean(axis=1)


def _clip_bytes_and_script():
    parts, script, t = [], [], 0.0
    for name, label in zip(_TURNS, _LABELS):
        a = _wav_arr(name)
        dur = len(a) / SR
        parts.append(a)
        script.append(Segment(t, t + dur, label))
        t += dur
    buf = io.BytesIO()
    sf.write(buf, np.concatenate(parts), SR, format="WAV")
    return buf.getvalue(), script


def _wav_bytes(name):
    buf = io.BytesIO()
    sf.write(buf, _wav_arr(name), SR, format="WAV")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    clip, script = _clip_bytes_and_script()
    main.app.state.store = VoiceprintStore(db_path=tmp_path / "vp.db", key=os.urandom(32)).open()
    main.app.state.embedder = OnnxSpeakerEmbedder(MODEL).load()
    main.app.state.auth = TokenAuth({RECATO: Caller("recato", frozenset({"enroll", "diarize"}))})
    main.app.state.diarizer_factory = lambda: MockDiarizer(script)  # core venv: no diart
    with TestClient(main.app) as c:
        c._clip = clip
        yield c
    for attr in ("store", "embedder", "auth", "diarizer_factory"):
        setattr(main.app.state, attr, None)


def _hdr(token=RECATO):
    return {"Authorization": f"Bearer {token}"} if token else {}


def _enroll_alice(client):
    client.post("/v1/enroll", files={"file": ("a.wav", _wav_bytes("spkA_1"), "audio/wav")},
                data={"identity": "Alice", "workspace": "ws1"}, headers=_hdr())


def _diarize(client, tag="offline", ws="ws1", identity=None, token=RECATO):
    data = {"workspace": ws, "tag": tag}
    if identity:
        data["identity"] = identity
    return client.post("/v1/diarize", files={"file": ("clip.wav", client._clip, "audio/wav")},
                       data=data, headers=_hdr(token))


def _parse_ndjson(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_offline_streams_contract_valid_segments(client):
    _enroll_alice(client)
    r = _diarize(client, tag="offline")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    rows = _parse_ndjson(r.text)
    segs = [x for x in rows if x.get("type") != "transcript"]
    assert segs, "no segments streamed"
    for s in segs:
        assert {"start", "end", "voiceprint_id", "name"} <= s.keys()
        assert s["end"] > s["start"]


def test_offline_identifies_enrolled_and_anonymous(client):
    _enroll_alice(client)
    rows = _parse_ndjson(_diarize(client, tag="offline").text)
    transcript = next(x for x in rows if x.get("type") == "transcript")["segments"]
    names = {s["name"] for s in transcript}
    assert "Alice" in names                                   # enrolled speaker recognized
    anon = [s for s in transcript if s["voiceprint_id"] and s["name"] is None]
    assert anon                                               # unknown speaker stays anonymous


def test_final_transcript_line_present(client):
    _enroll_alice(client)
    rows = _parse_ndjson(_diarize(client, tag="offline").text)
    assert rows[-1].get("type") == "transcript" and rows[-1]["segments"]


def test_gmeet_tag_routes_to_enroll(client):
    r = client.post("/v1/diarize",
                    files={"file": ("a.wav", _wav_bytes("spkA_1"), "audio/wav")},
                    data={"workspace": "ws1", "tag": "gmeet", "identity": "Bob"}, headers=_hdr())
    assert r.status_code == 200
    body = r.json()
    assert body["routed"] == "enroll" and body["status"] == "created"


def test_gmeet_without_identity_400(client):
    r = client.post("/v1/diarize",
                    files={"file": ("a.wav", _wav_bytes("spkA_1"), "audio/wav")},
                    data={"workspace": "ws1", "tag": "gmeet"}, headers=_hdr())
    assert r.status_code == 400


def test_diarize_requires_auth(client):
    assert _diarize(client, token=None).status_code == 401
