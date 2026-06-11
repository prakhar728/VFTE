"""D.1 — scoped token auth (unit + endpoint enforcement)."""
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from auth import Caller, TokenAuth, require_scope

RECATO, CONCLAVE = "tok-recato", "tok-conclave"


def _auth():
    return TokenAuth({
        RECATO: Caller("recato", frozenset({"enroll", "diarize", "vocab"})),
        CONCLAVE: Caller("conclave", frozenset({"knowledge"}), workspaces=frozenset({"ws1"})),
    })


# ── unit: TokenAuth ──────────────────────────────────────

def test_from_env_parses_and_fails_closed():
    assert TokenAuth.from_env("")._tokens == {}          # unset → deny all
    a = TokenAuth.from_env('{"t1": {"name": "recato", "endpoints": ["enroll", "diarize"]}}')
    assert a.caller_for("t1").name == "recato"
    assert a.caller_for("t1").endpoints == frozenset({"enroll", "diarize"})


def test_from_env_rejects_unknown_endpoint():
    with pytest.raises(ValueError):
        TokenAuth.from_env('{"t": {"name": "x", "endpoints": ["nope"]}}')


def test_bad_token_401():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        _auth().caller_for("garbage")
    assert ei.value.status_code == 401


def test_workspace_scope():
    c = _auth().caller_for(CONCLAVE)
    assert c.allows_workspace("ws1") and not c.allows_workspace("ws2")
    assert _auth().caller_for(RECATO).allows_workspace("anything")  # None → all


# ── endpoint enforcement via require_scope ───────────────

@pytest.fixture
def client():
    app = FastAPI()
    app.state.auth = _auth()

    @app.post("/v1/diarize")
    def diarize(caller: Caller = Depends(require_scope("diarize"))):
        return {"caller": caller.name}

    @app.post("/v1/knowledge")
    def knowledge(caller: Caller = Depends(require_scope("knowledge"))):
        return {"caller": caller.name}

    return TestClient(app)


def _post(client, path, token):
    h = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post(path, headers=h)


def test_missing_token_401(client):
    assert _post(client, "/v1/diarize", None).status_code == 401


def test_in_scope_allowed(client):
    r = _post(client, "/v1/diarize", RECATO)
    assert r.status_code == 200 and r.json()["caller"] == "recato"


def test_conclave_blocked_from_diarize_403(client):
    assert _post(client, "/v1/diarize", CONCLAVE).status_code == 403


def test_recato_blocked_from_knowledge_403(client):
    assert _post(client, "/v1/knowledge", RECATO).status_code == 403


def test_x_api_key_header_also_works(client):
    r = client.post("/v1/diarize", headers={"X-API-Key": RECATO})
    assert r.status_code == 200
