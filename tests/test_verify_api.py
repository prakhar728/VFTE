"""Verify/observability endpoints: GET /v1/voiceprints, POST /v1/identify."""
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
TOK = "tok-verify"

pytestmark = pytest.mark.skipif(
    not MODEL.exists(), reason="embedder model missing — run scripts/fetch_models.sh"
)


@pytest.fixture
def client(tmp_path):
    main.app.state.store = VoiceprintStore(db_path=tmp_path / "vp.db", key=os.urandom(32)).open()
    main.app.state.embedder = OnnxSpeakerEmbedder(MODEL).load()
    main.app.state.auth = TokenAuth({
        TOK: Caller("verifier", frozenset({"enroll", "voiceprints", "identify"})),
    })
    with TestClient(main.app) as c:
        yield c
    for attr in ("store", "embedder", "auth"):
        setattr(main.app.state, attr, None)


def _hdr(token=TOK):
    return {"Authorization": f"Bearer {token}"} if token else {}


def _enroll(client, name, identity, ws="ws1"):
    with open(FIX / f"{name}.wav", "rb") as fh:
        return client.post("/v1/enroll", files={"file": (f"{name}.wav", fh, "audio/wav")},
                           data={"identity": identity, "workspace": ws}, headers=_hdr())


def _identify(client, name, ws="ws1", token=TOK):
    with open(FIX / f"{name}.wav", "rb") as fh:
        return client.post("/v1/identify", files={"file": (f"{name}.wav", fh, "audio/wav")},
                           data={"workspace": ws}, headers=_hdr(token))


# ── GET /v1/voiceprints ──────────────────────────────────

def test_voiceprints_empty(client):
    r = client.get("/v1/voiceprints/ws1", headers=_hdr())
    assert r.status_code == 200 and r.json() == {"workspace": "ws1", "count": 0, "voiceprints": []}


def test_voiceprints_lists_enrolled(client):
    _enroll(client, "spkA_1", "Alice")
    _enroll(client, "spkA_2", "Alice")  # strengthens
    r = client.get("/v1/voiceprints/ws1", headers=_hdr())
    body = r.json()
    assert body["count"] == 1
    vp = body["voiceprints"][0]
    assert vp["name"] == "Alice" and vp["exemplar_count"] == 2 and vp["enroll_count"] == 2
    assert "centroid" not in vp and "exemplars" not in vp  # metadata only


def test_voiceprints_workspace_scoped(client):
    _enroll(client, "spkA_1", "Alice", ws="ws1")
    assert client.get("/v1/voiceprints/ws2", headers=_hdr()).json()["count"] == 0


# ── POST /v1/identify ────────────────────────────────────

def test_identify_matches_enrolled(client):
    _enroll(client, "spkA_1", "Alice")
    r = _identify(client, "spkA_2")  # same speaker, held-out clip
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "MATCH" and body["name"] == "Alice"
    assert body["confidence"] > 0.5


def test_identify_rejects_stranger(client):
    _enroll(client, "spkA_1", "Alice")
    r = _identify(client, "spkB_1")  # different speaker → not enrolled
    assert r.json()["decision"] == "UNKNOWN" and r.json()["name"] is None


def test_identify_empty_store_unknown(client):
    assert _identify(client, "spkA_1").json()["decision"] == "UNKNOWN"


# ── auth ─────────────────────────────────────────────────

def test_verify_endpoints_require_scope(client):
    main.app.state.auth = TokenAuth({"weak": Caller("weak", frozenset({"enroll"}))})
    assert client.get("/v1/voiceprints/ws1", headers=_hdr("weak")).status_code == 403
    assert _identify(client, "spkA_1", token="weak").status_code == 403
    assert client.get("/v1/voiceprints/ws1", headers=_hdr(None)).status_code == 401
