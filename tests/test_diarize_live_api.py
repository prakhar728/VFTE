"""Branch C — POST /v1/diarize tag="live" is read-only (P1 at the endpoint).

Mirrors test_diarize_api.py (MockDiarizer factory) but asserts the live tag writes
nothing to the store while still streaming the C2 shape, and that the default
"offline" tag still writes — proving the branch is tag-selected.
"""
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
from fpm.enroll import identity_voiceprint_id
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
_C2_KEYS = {"start", "end", "voiceprint_id", "name", "decision", "confidence", "local_speaker"}


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
    main.app.state.diarizer_factory = lambda: MockDiarizer(script)
    with TestClient(main.app) as c:
        c._clip = clip
        yield c
    for attr in ("store", "embedder", "auth", "diarizer_factory"):
        setattr(main.app.state, attr, None)


def _hdr():
    return {"Authorization": f"Bearer {RECATO}"}


def _enroll_alice(client):
    client.post("/v1/enroll", files={"file": ("a.wav", _wav_bytes("spkA_1"), "audio/wav")},
                data={"identity": "Alice", "workspace": "ws1"}, headers=_hdr())


def _diarize(client, tag):
    return client.post("/v1/diarize", files={"file": ("clip.wav", client._clip, "audio/wav")},
                       data={"workspace": "ws1", "tag": tag}, headers=_hdr())


def _parse_ndjson(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_live_tag_preserves_c2_shape(client):
    _enroll_alice(client)
    r = _diarize(client, tag="live")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    rows = _parse_ndjson(r.text)
    segs = [x for x in rows if x.get("type") != "transcript"]
    assert segs
    for s in segs:
        assert _C2_KEYS <= s.keys()          # full 7-key C2 shape preserved
        assert s["end"] > s["start"]
    assert rows[-1].get("type") == "transcript" and rows[-1]["segments"]


def test_live_tag_writes_nothing(client):
    _enroll_alice(client)
    alice_id = identity_voiceprint_id("ws1", "Alice")
    store = main.app.state.store
    ids_before = set(store.list_ids("ws1"))
    ledger_before = len(store.usage_for_voiceprint("ws1", alice_id))

    _diarize(client, tag="live")

    assert set(store.list_ids("ws1")) == ids_before == {alice_id}   # no mint
    assert len(store.usage_for_voiceprint("ws1", alice_id)) == ledger_before  # no ledger row


def test_live_tag_still_identifies(client):
    _enroll_alice(client)
    transcript = next(x for x in _parse_ndjson(_diarize(client, tag="live").text)
                      if x.get("type") == "transcript")["segments"]
    names = {s["name"] for s in transcript}
    assert "Alice" in names                                          # enrolled recognized for display
    anon = [s for s in transcript if s["name"] is None and s["decision"] != "PENDING"]
    assert anon and all(s["voiceprint_id"] is None for s in anon)    # unknown stays id-less in live


def test_offline_tag_still_writes(client):
    """Control: default offline tag is the authoritative writer — mints the unknown."""
    _enroll_alice(client)
    store = main.app.state.store
    _diarize(client, tag="offline")
    assert len(store.list_ids("ws1")) == 2                          # Alice + minted anon
