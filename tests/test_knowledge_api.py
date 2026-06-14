"""D.3 — POST /v1/knowledge (bind anonymous voiceprint -> name; push vocab)."""
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main
from auth import Caller, TokenAuth
from fpm.store.models import Voiceprint
from fpm.store.store import VoiceprintStore, new_voiceprint_id

CONCLAVE, RECATO = "tok-conclave", "tok-recato"


def _anon(store, ws="ws1"):
    vp = Voiceprint(new_voiceprint_id(), ws, name="")
    vp.add_exemplar(np.ones(512, dtype=np.float32) / np.sqrt(512))
    vp.recompute_centroid()
    store.upsert(vp)
    return vp.voiceprint_id


@pytest.fixture
def client(tmp_path):
    store = VoiceprintStore(db_path=tmp_path / "vp.db", key=os.urandom(32)).open()
    main.app.state.store = store
    main.app.state.embedder = object()  # knowledge path doesn't embed
    main.app.state.auth = TokenAuth({
        CONCLAVE: Caller("conclave", frozenset({"knowledge"}), workspaces=frozenset({"ws1"})),
        RECATO: Caller("recato", frozenset({"enroll", "diarize"})),
    })
    with TestClient(main.app) as c:
        c._store = store
        yield c
    for attr in ("store", "embedder", "auth"):
        setattr(main.app.state, attr, None)


def _post(client, body, token=CONCLAVE):
    h = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post("/v1/knowledge", json=body, headers=h)


def test_bind_names_anonymous_voiceprint(client):
    vp = _anon(client._store)
    r = _post(client, {"workspace": "ws1", "bindings": [{"voiceprint_id": vp, "name": "Carol"}]})
    assert r.status_code == 200 and r.json()["bound"] == [vp]
    assert client._store.get("ws1", vp).name == "Carol"


def test_binding_is_audited_and_reversible(client):
    vp = _anon(client._store)
    _post(client, {"workspace": "ws1", "bindings": [{"voiceprint_id": vp, "name": "Carol"}]})
    _post(client, {"workspace": "ws1", "bindings": [{"voiceprint_id": vp, "name": ""}]})  # revert
    assert client._store.get("ws1", vp).name == ""
    audit = client._store.audit_entries("ws1")
    assert [(a["old_name"], a["new_name"], a["actor"]) for a in audit] == \
           [("", "Carol", "conclave"), ("Carol", "", "conclave")]


def test_unknown_voiceprint_not_found(client):
    r = _post(client, {"workspace": "ws1", "bindings": [{"voiceprint_id": "vp_nope", "name": "X"}]})
    assert r.json()["bound"] == [] and r.json()["not_found"] == ["vp_nope"]


def test_wrong_workspace_403(client):
    vp = _anon(client._store, ws="ws2")
    r = _post(client, {"workspace": "ws2", "bindings": [{"voiceprint_id": vp, "name": "X"}]})
    assert r.status_code == 403  # conclave token scoped to ws1 only


def test_vocab_pushed(client):
    r = _post(client, {"workspace": "ws1", "vocab_terms": ["Kubernetes", "Anthropic"]})
    assert r.status_code == 200 and r.json()["vocab_terms"] == 2
    assert client._store.get_vocab("ws1") == ["Kubernetes", "Anthropic"]


def test_recato_blocked_from_knowledge(client):
    r = _post(client, {"workspace": "ws1", "vocab_terms": ["x"]}, token=RECATO)
    assert r.status_code == 403


def test_missing_token_401(client):
    assert _post(client, {"workspace": "ws1"}, token=None).status_code == 401


# ── P4: email-bearing binding evolves set_name → self-confirmed proposal ──

def test_email_binding_creates_confirmed_owner_binding(client):
    vp = _anon(client._store)
    r = _post(client, {"workspace": "ws1",
                       "bindings": [{"voiceprint_id": vp, "name": "Carol", "email": "carol@x.com"}]})
    assert r.status_code == 200 and r.json()["bound"] == [vp]
    v = client._store.get("ws1", vp)
    assert v.name == "Carol" and v.owner_email == "carol@x.com"  # owner bound, not just named
    assert client._store.consent_resolve("ws1", vp)["visibility"] == "named"


def test_email_binding_unknown_voiceprint_not_found(client):
    r = _post(client, {"workspace": "ws1",
                       "bindings": [{"voiceprint_id": "vp_nope", "name": "X", "email": "x@x.com"}]})
    assert r.json()["bound"] == [] and r.json()["not_found"] == ["vp_nope"]


def test_bare_name_binding_stays_legacy_no_owner(client):
    vp = _anon(client._store)
    r = _post(client, {"workspace": "ws1", "bindings": [{"voiceprint_id": vp, "name": "Dave"}]})
    assert r.json()["bound"] == [vp]
    v = client._store.get("ws1", vp)
    assert v.name == "Dave" and v.owner_email == ""  # legacy path binds no owner_email
