"""B.2 — POST /v1/enroll endpoint (gmeet path)."""
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main
from auth import Caller, TokenAuth
from fpm.embed.onnx_embedder import OnnxSpeakerEmbedder
from fpm.store.store import VoiceprintStore

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "campplus.onnx"
FIX = Path(__file__).parent / "fixtures" / "speakers"
RECATO = "tok-recato"

pytestmark = pytest.mark.skipif(
    not MODEL.exists(), reason="embedder model missing — run scripts/fetch_models.sh"
)


@pytest.fixture
def client(tmp_path):
    main.app.state.store = VoiceprintStore(db_path=tmp_path / "vp.db", key=os.urandom(32)).open()
    main.app.state.embedder = OnnxSpeakerEmbedder(MODEL).load()
    main.app.state.auth = TokenAuth({RECATO: Caller("recato", frozenset({"enroll", "diarize"}))})
    with TestClient(main.app) as c:
        yield c
    main.app.state.store = None
    main.app.state.embedder = None
    main.app.state.auth = None


def _enroll(client, name, identity, ws="ws1", token=RECATO):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with open(FIX / f"{name}.wav", "rb") as fh:
        return client.post(
            "/v1/enroll",
            files={"file": (f"{name}.wav", fh, "audio/wav")},
            data={"identity": identity, "workspace": ws},
            headers=headers,
        )


def test_enroll_created_then_updated(client):
    r1 = _enroll(client, "spkA_1", "Alice")
    assert r1.status_code == 200
    b1 = r1.json()
    assert b1["status"] == "created" and b1["voiceprint_id"].startswith("vp_")

    r2 = _enroll(client, "spkA_2", "Alice")  # same speaker, strengthens
    assert r2.json()["status"] == "updated"
    assert r2.json()["voiceprint_id"] == b1["voiceprint_id"]


def test_enroll_rejects_inconsistent(client):
    _enroll(client, "spkA_1", "Alice")
    r = _enroll(client, "spkB_1", "Alice")  # different voice under Alice's identity
    assert r.json()["status"] == "rejected"


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"
