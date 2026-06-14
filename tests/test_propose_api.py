"""P4 — /v1/propose + /v1/consent/resolve (M2M, `knowledge` scope).

Drives the FastAPI app with a tmp store + Conclave token, like test_knowledge_api.
Confirm/deny (session-authed) live in test_consent_api.py.
"""
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

import config
import main
from auth import Caller, TokenAuth
from fpm.store.models import Voiceprint
from fpm.store.store import VoiceprintStore, new_voiceprint_id

CONCLAVE = "tok-conclave"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONSENT_AUTOCONFIRM", False)  # explicit Phase-1 default
    store = VoiceprintStore(db_path=tmp_path / "vp.db", key=os.urandom(32)).open()
    main.app.state.store = store
    main.app.state.embedder = object()  # propose/resolve never embed
    main.app.state.auth = TokenAuth({
        CONCLAVE: Caller("conclave", frozenset({"knowledge"}), workspaces=frozenset({"ws1"})),
    })
    with TestClient(main.app) as c:
        c._store = store
        yield c, store, monkeypatch
    for attr in ("store", "embedder", "auth"):
        setattr(main.app.state, attr, None)


def _vp(store, ws="ws1", name="", email="", identify_allowed=True):
    vp = Voiceprint(new_voiceprint_id(), ws, name=name, owner_email=email,
                    identify_allowed=identify_allowed)
    vp.add_exemplar(np.ones(512, dtype=np.float32) / np.sqrt(512))
    vp.recompute_centroid()
    store.upsert(vp)
    return vp.voiceprint_id


def _propose(c, body, token=CONCLAVE):
    h = {"Authorization": f"Bearer {token}"} if token else {}
    return c.post("/v1/propose", json=body, headers=h)


# ── propose: self-tag / flag / pending ───────────────────────

def test_self_tag_autoconfirms(client):
    c, store, _ = client
    vid = _vp(store)
    r = _propose(c, {"workspace": "ws1", "voiceprint_id": vid,
                     "proposed_email": "alice@x.com", "proposed_by": "alice@x.com",
                     "proposed_name": "Alice"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "confirmed" and body["auto_confirmed"] is True
    assert body["name"] == "Alice" and body["owner_email"] == "alice@x.com"
    vp = store.get("ws1", vid)
    assert vp.owner_email == "alice@x.com" and vp.name == "Alice"
    # the bind is audited (reuses set_name's binding_audit)
    assert any(a["new_name"] == "Alice" for a in store.audit_entries("ws1"))


def test_flag_on_autoconfirms_cross_tag(client):
    c, store, monkeypatch = client
    monkeypatch.setattr(config, "CONSENT_AUTOCONFIRM", True)
    vid = _vp(store)
    r = _propose(c, {"workspace": "ws1", "voiceprint_id": vid,
                     "proposed_email": "alice@x.com", "proposed_by": "host@x.com",
                     "proposed_name": "Alice"})
    assert r.json()["status"] == "confirmed" and r.json()["auto_confirmed"] is True
    assert store.get("ws1", vid).owner_email == "alice@x.com"


def test_pending_when_flag_off_and_emails_differ(client):
    c, store, _ = client
    vid = _vp(store)
    r = _propose(c, {"workspace": "ws1", "voiceprint_id": vid,
                     "proposed_email": "alice@x.com", "proposed_by": "host@x.com",
                     "proposed_name": "Alice"})
    body = r.json()
    assert body["status"] == "pending" and body["auto_confirmed"] is False
    assert body["name"] is None and body["owner_email"] is None
    vp = store.get("ws1", vid)
    assert vp.owner_email == "" and vp.name == ""  # nothing bound yet


def test_propose_idempotent(client):
    c, store, _ = client
    vid = _vp(store)
    base = {"workspace": "ws1", "voiceprint_id": vid, "proposed_email": "alice@x.com",
            "proposed_by": "host@x.com", "proposed_name": "Alice"}
    a = _propose(c, base).json()
    b = _propose(c, base).json()
    assert a["proposal_id"] == b["proposal_id"]


def test_propose_404_unknown_voiceprint(client):
    c, _, _ = client
    r = _propose(c, {"workspace": "ws1", "voiceprint_id": "vp_nope",
                     "proposed_email": "a@x.com", "proposed_by": "h@x.com", "proposed_name": "A"})
    assert r.status_code == 404


def test_propose_403_wrong_workspace(client):
    c, store, _ = client
    vid = _vp(store, ws="ws2")
    r = _propose(c, {"workspace": "ws2", "voiceprint_id": vid,
                     "proposed_email": "a@x.com", "proposed_by": "h@x.com", "proposed_name": "A"})
    assert r.status_code == 403  # conclave token scoped to ws1


def test_propose_missing_token_401(client):
    c, store, _ = client
    vid = _vp(store)
    r = _propose(c, {"workspace": "ws1", "voiceprint_id": vid,
                     "proposed_email": "a@x.com", "proposed_by": "h@x.com", "proposed_name": "A"},
                 token=None)
    assert r.status_code == 401


# ── consent-resolve (read-side gate) ─────────────────────────

def test_consent_resolve_get_named(client):
    c, store, _ = client
    vid = _vp(store, name="Alice", email="alice@x.com")
    r = c.get(f"/v1/consent/resolve/ws1/{vid}", headers={"Authorization": f"Bearer {CONCLAVE}"})
    assert r.status_code == 200
    assert r.json() == {"voiceprint_id": vid, "name": "Alice",
                        "owner_email": "alice@x.com", "visibility": "named"}


def test_consent_resolve_anonymous_gate(client):
    c, store, _ = client
    vid = _vp(store, name="Alice", email="alice@x.com", identify_allowed=False)
    r = c.get(f"/v1/consent/resolve/ws1/{vid}", headers={"Authorization": f"Bearer {CONCLAVE}"})
    assert r.json()["name"] is None and r.json()["visibility"] == "anonymous"


def test_consent_resolve_batch(client):
    c, store, _ = client
    v1 = _vp(store, name="Alice", email="alice@x.com")
    v2 = _vp(store)  # anonymous
    r = c.post("/v1/consent/resolve/ws1", json={"voiceprint_ids": [v1, v2, "vp_nope"]},
               headers={"Authorization": f"Bearer {CONCLAVE}"})
    res = r.json()["resolved"]
    assert res[v1]["visibility"] == "named" and res[v1]["name"] == "Alice"
    assert res[v2]["visibility"] == "anonymous" and res[v2]["name"] is None
    assert res["vp_nope"]["visibility"] == "unknown"
