"""D.4 — GET /v1/vocab/{workspace}."""
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main
from auth import Caller, TokenAuth
from fpm.store.store import VoiceprintStore

RECATO, CONCLAVE = "tok-recato", "tok-conclave"


@pytest.fixture
def client(tmp_path):
    store = VoiceprintStore(db_path=tmp_path / "vp.db", key=os.urandom(32)).open()
    main.app.state.store = store
    main.app.state.embedder = object()
    main.app.state.auth = TokenAuth({
        RECATO: Caller("recato", frozenset({"enroll", "diarize", "vocab"})),
        CONCLAVE: Caller("conclave", frozenset({"knowledge"})),
    })
    with TestClient(main.app) as c:
        c._store = store
        yield c
    for attr in ("store", "embedder", "auth"):
        setattr(main.app.state, attr, None)


def _get(client, ws="ws1", token=RECATO):
    h = {"Authorization": f"Bearer {token}"} if token else {}
    return client.get(f"/v1/vocab/{ws}", headers=h)


def test_empty_vocab(client):
    r = _get(client)
    assert r.status_code == 200
    assert r.json() == {"workspace": "ws1", "terms": [], "prompt": ""}


def test_returns_terms_and_prompt(client):
    client._store.set_vocab("ws1", ["Kubernetes", "Anthropic"])
    r = _get(client)
    assert r.json()["terms"] == ["Kubernetes", "Anthropic"]
    assert r.json()["prompt"] == "Kubernetes, Anthropic"


def test_vocab_requires_scope(client):
    assert _get(client, token=CONCLAVE).status_code == 403   # conclave lacks vocab scope
    assert _get(client, token=None).status_code == 401
