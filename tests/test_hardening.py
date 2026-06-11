"""D.5 — health, write rate-limits, structured error envelope."""
import os

import pytest
from fastapi.testclient import TestClient

import main
from auth import Caller, TokenAuth
from fpm.store.store import VoiceprintStore
from ratelimit import RateLimiter

CONCLAVE = "tok-conclave"


def test_rate_limiter_unit():
    rl = RateLimiter(max_writes=2, window_sec=10)
    assert rl.allow("k", now=0.0) and rl.allow("k", now=1.0)   # 2 allowed
    assert not rl.allow("k", now=2.0)                          # 3rd blocked
    assert rl.allow("k", now=11.0)                             # window reset
    assert rl.allow("other", now=2.0)                          # per-key isolation


@pytest.fixture
def client(tmp_path):
    main.app.state.store = VoiceprintStore(db_path=tmp_path / "vp.db", key=os.urandom(32)).open()
    main.app.state.embedder = object()
    main.app.state.auth = TokenAuth({CONCLAVE: Caller("conclave", frozenset({"knowledge"}))})
    main.app.state.rate_limiter = RateLimiter(max_writes=2, window_sec=60)
    with TestClient(main.app) as c:
        yield c
    for attr in ("store", "embedder", "auth", "rate_limiter"):
        setattr(main.app.state, attr, None)


def _knowledge(client, token=CONCLAVE):
    h = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post("/v1/knowledge", json={"workspace": "ws1", "vocab_terms": ["x"]}, headers=h)


def test_health_ok(client):
    assert client.get("/health").json()["status"] == "ok"


def test_write_flood_429(client):
    assert _knowledge(client).status_code == 200
    assert _knowledge(client).status_code == 200
    r = _knowledge(client)                                     # 3rd write in window
    assert r.status_code == 429
    assert r.json()["error"]["status"] == 429


def test_structured_error_envelope(client):
    r = _knowledge(client, token=None)                         # 401
    assert r.status_code == 401
    body = r.json()
    assert set(body["error"]) == {"status", "message"}
    assert body["error"]["status"] == 401 and body["error"]["message"]
