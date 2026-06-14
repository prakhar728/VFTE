"""Consent-plane web surface (WS2 + WS4): sign-in session + dashboard data + controls.

Drives the FastAPI app with a tmp store and a forged session cookie (dev-login path),
so no real Google round-trip is needed.
"""
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

import config
from auth import SESSION_COOKIE, GoogleOAuth, SessionManager, TokenAuth
from fpm.store.models import Voiceprint
from fpm.store.store import VoiceprintStore

DIM = 512
KEY = os.urandom(32)


def _unit(seed):
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _vp(ws, vid, name, email, seed, **kw):
    vp = Voiceprint(vid, ws, name, owner_email=email, **kw)
    for i in range(3):
        vp.add_exemplar(_unit(seed * 10 + i))
    vp.recompute_centroid()
    return vp


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DEV_LOGIN", True)
    import main
    store = VoiceprintStore(db_path=tmp_path / "vp.db", key=KEY).open()
    store.upsert(_vp("ws1", "vp_alice", "alice@x.com", "alice@x.com", 1))
    store.upsert(_vp("ws2", "vp_alice2", "alice@x.com", "alice@x.com", 2))
    store.upsert(_vp("ws1", "vp_bob", "bob@x.com", "bob@x.com", 3))
    main.app.state.store = store
    main.app.state.auth = TokenAuth.from_env("{}")
    main.app.state.sessions = SessionManager("test-secret", 3600)
    main.app.state.oauth = GoogleOAuth("", "", "http://localhost/auth/callback")
    with TestClient(main.app) as c:
        yield c, store
    store.close()


def _login(c, email):
    token = c.app.state.sessions.issue(email)
    c.cookies.set(SESSION_COOKIE, token)


def test_me_signed_out(client):
    c, _ = client
    r = c.get("/v1/me")
    assert r.status_code == 200 and r.json()["signed_in"] is False
    assert r.json()["dev_login"] is True


def test_dashboard_served(client):
    c, _ = client
    r = c.get("/dashboard")
    assert r.status_code == 200 and "Your voiceprint" in r.text


def test_voiceprints_require_auth(client):
    c, _ = client
    assert c.get("/v1/me/voiceprints").status_code == 401


def test_lists_only_own_voiceprints_across_workspaces(client):
    c, _ = client
    _login(c, "alice@x.com")
    data = c.get("/v1/me/voiceprints").json()
    wss = {v["workspace_id"] for v in data["voiceprints"]}
    assert data["count"] == 2 and wss == {"ws1", "ws2"}        # both Alice entries, isolated
    assert all(v["owner_email"] == "alice@x.com" for v in data["voiceprints"])


def test_stay_anonymous_toggle(client):
    c, store = client
    _login(c, "alice@x.com")
    r = c.post("/v1/me/voiceprints/ws1/vp_alice/flags", json={"identify_allowed": False})
    assert r.status_code == 200 and r.json()["identify_allowed"] is False
    assert store.identify_allowed("ws1", "vp_alice") is False
    # ledger recorded the control action
    assert any(u["event"] == "control" for u in store.usage_for_voiceprint("ws1", "vp_alice"))


def test_cannot_touch_someone_elses_voiceprint(client):
    c, _ = client
    _login(c, "alice@x.com")
    assert c.post("/v1/me/voiceprints/ws1/vp_bob/flags",
                  json={"identify_allowed": False}).status_code == 403
    assert c.post("/v1/me/voiceprints/ws1/vp_bob/forget").status_code == 403


def test_forget_me_deletes(client):
    c, store = client
    _login(c, "alice@x.com")
    r = c.post("/v1/me/voiceprints/ws1/vp_alice/forget")
    assert r.status_code == 200 and r.json()["deleted"] is True
    assert store.get("ws1", "vp_alice") is None
    # the other-workspace entry is untouched
    assert store.get("ws2", "vp_alice2") is not None


def test_login_503_without_google(client):
    c, _ = client
    r = c.get("/auth/login", follow_redirects=False)
    assert r.status_code == 503


def test_dev_login_sets_session(client):
    c, _ = client
    r = c.get("/auth/dev-login?email=Alice@X.com", follow_redirects=False)
    assert r.status_code in (302, 307)
    me = c.get("/v1/me").json()
    assert me["signed_in"] and me["email"] == "alice@x.com"     # lowercased


# ── P4: confirm / deny a pending binding (session-authed) ────

def test_confirm_proposal_via_session(client):
    c, store = client
    store.upsert(_vp("ws1", "vp_anon1", "", "", 9))            # anonymous voiceprint
    p = store.propose("ws1", "vp_anon1", "carol@x.com", "host@x.com", "Carol")
    _login(c, "carol@x.com")
    r = c.post("/v1/confirm", json={"proposal_id": p["proposal_id"]})
    assert r.status_code == 200
    assert r.json()["status"] == "confirmed" and r.json()["name"] == "Carol"
    vp = store.get("ws1", "vp_anon1")
    assert vp.owner_email == "carol@x.com" and vp.name == "Carol"


def test_confirm_requires_auth(client):
    c, store = client
    store.upsert(_vp("ws1", "vp_anon2", "", "", 8))
    p = store.propose("ws1", "vp_anon2", "carol@x.com", "host@x.com", "Carol")
    assert c.post("/v1/confirm", json={"proposal_id": p["proposal_id"]}).status_code == 401


def test_deny_leaves_speaker_unbound(client):
    c, store = client
    store.upsert(_vp("ws1", "vp_anon3", "", "", 7))
    p = store.propose("ws1", "vp_anon3", "carol@x.com", "host@x.com", "Carol")
    _login(c, "carol@x.com")
    r = c.post("/v1/deny", json={"proposal_id": p["proposal_id"]})
    assert r.status_code == 200 and r.json()["status"] == "denied"
    vp = store.get("ws1", "vp_anon3")
    assert vp.owner_email == "" and vp.name == ""


def test_confirm_unknown_proposal_404(client):
    c, _ = client
    _login(c, "carol@x.com")
    assert c.post("/v1/confirm", json={"proposal_id": "prop_nope"}).status_code == 404


# ── P2: only the tagged target may confirm / deny ────────────

def test_confirm_by_non_target_403(client):
    c, store = client
    store.upsert(_vp("ws1", "vp_anon4", "", "", 6))
    p = store.propose("ws1", "vp_anon4", "carol@x.com", "host@x.com", "Carol")
    _login(c, "mallory@x.com")                       # not the tagged person
    r = c.post("/v1/confirm", json={"proposal_id": p["proposal_id"]})
    assert r.status_code == 403
    vp = store.get("ws1", "vp_anon4")
    assert vp.owner_email == "" and vp.name == ""    # a non-target binds nothing


def test_deny_by_non_target_403(client):
    c, store = client
    store.upsert(_vp("ws1", "vp_anon5", "", "", 5))
    p = store.propose("ws1", "vp_anon5", "carol@x.com", "host@x.com", "Carol")
    _login(c, "mallory@x.com")
    r = c.post("/v1/deny", json={"proposal_id": p["proposal_id"]})
    assert r.status_code == 403
    assert store.get_proposal(p["proposal_id"])["status"] == "pending"  # untouched
