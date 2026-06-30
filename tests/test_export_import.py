"""Signed voiceprint export / import (Task #4) — spec §6's checks.

Drives the FastAPI app with a tmp store + an injected Ed25519 signer (fixed seed → no
keyfile side effects, deterministic signatures). Covers: export→wipe→import restores the
voiceprint (recomputed centroid == pre-wipe, /identify-style match preserved); tamper /
unsigned / foreign-key reject; model/dim mismatch reject; ownership mismatch → 403;
collision → merge (eviction), absent → create; consent flags restored; an "import" ledger
row written; round-trip determinism; and the export-all bundle round-trips.
"""
import copy
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

import config
from auth import SESSION_COOKIE, GoogleOAuth, SessionManager, TokenAuth
from fpm.match import classify
from fpm.receipts import ReceiptSigner
from fpm.store.models import MAX_EXEMPLARS, Voiceprint
from fpm.store.store import VoiceprintStore

DIM = 512
STORE_KEY = os.urandom(32)
SEED = bytes(range(32))                       # fixed → deterministic signer
OTHER_SEED = bytes(range(32, 64))             # a *foreign* signer (not ours)


def _unit(seed):
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _vp(ws, vid, email, seed, n=4, **kw):
    vp = Voiceprint(vid, ws, email.split("@")[0], owner_email=email, **kw)
    for i in range(n):
        vp.add_exemplar(_unit(seed * 100 + i))
    vp.recompute_centroid()
    return vp


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DEV_LOGIN", True)
    import main
    store = VoiceprintStore(db_path=tmp_path / "vp.db", key=STORE_KEY).open()
    store.upsert(_vp("ws1", "vp_alice", "alice@x.com", 1))
    store.upsert(_vp("ws2", "vp_alice2", "alice@x.com", 2))
    store.upsert(_vp("ws1", "vp_bob", "bob@x.com", 3))
    main.app.state.store = store
    main.app.state.auth = TokenAuth.from_env("{}")
    main.app.state.sessions = SessionManager("test-secret", 3600)
    main.app.state.oauth = GoogleOAuth("", "", "http://localhost/auth/callback")
    main.app.state.receipt_signer = ReceiptSigner(SEED)
    with TestClient(main.app) as c:
        yield c, store
    store.close()
    for attr in ("store", "receipt_signer"):
        setattr(main.app.state, attr, None)


def _login(c, email):
    c.cookies.set(SESSION_COOKIE, c.app.state.sessions.issue(email))


def _export(c, ws="ws1", vid="vp_alice"):
    return c.get(f"/v1/me/voiceprints/{ws}/{vid}/export")


def _wipe(store, ws, vid):
    store.delete(ws, vid, actor="test")
    assert store.get(ws, vid) is None


# ── auth surface ─────────────────────────────────────────────

def test_export_requires_auth(client):
    c, _ = client
    assert _export(c).status_code == 401
    assert c.get("/v1/me/voiceprints/export").status_code == 401
    assert c.post("/v1/me/voiceprints/import", json={}).status_code == 401


def test_export_is_owner_scoped(client):
    c, _ = client
    _login(c, "alice@x.com")
    assert _export(c, "ws1", "vp_bob").status_code == 403       # bob's vp, alice's session
    assert _export(c, "ws1", "nope").status_code == 404


# ── §6.1 export → wipe → import restores ─────────────────────

def test_export_wipe_import_restores(client):
    c, store = client
    _login(c, "alice@x.com")
    pre = store.get("ws1", "vp_alice")
    pre_centroid = pre.centroid.copy()
    probe = pre.exemplars[0].copy()
    # the clip identifies as vp_alice before any of this
    assert classify(probe, store.centroids("ws1")).voiceprint_id == "vp_alice"

    env = _export(c).json()
    _wipe(store, "ws1", "vp_alice")
    assert classify(probe, store.centroids("ws1")).voiceprint_id != "vp_alice"

    r = c.post("/v1/me/voiceprints/import", json=env)
    assert r.status_code == 200 and r.json()["results"][0]["status"] == "created"

    post = store.get("ws1", "vp_alice")
    assert post is not None
    np.testing.assert_allclose(post.centroid, pre_centroid, atol=1e-6)
    # identify matches as before the wipe
    assert classify(probe, store.centroids("ws1")).voiceprint_id == "vp_alice"


# ── §6.2 tamper / unsigned / foreign-key → reject ────────────

def test_tampered_vector_rejected(client):
    c, store = client
    _login(c, "alice@x.com")
    env = _export(c).json()
    env["payload"]["exemplars_b64"][0] = env["payload"]["exemplars_b64"][1]  # swap a vector
    r = c.post("/v1/me/voiceprints/import", json=env)
    assert r.status_code == 400 and r.json()["error"]["message"] == "bad-signature"


def test_tampered_flag_rejected(client):
    c, _ = client
    _login(c, "alice@x.com")
    env = _export(c).json()
    env["payload"]["identify_allowed"] = not env["payload"]["identify_allowed"]
    assert c.post("/v1/me/voiceprints/import", json=env).status_code == 400


def test_unsigned_rejected(client):
    c, _ = client
    _login(c, "alice@x.com")
    env = _export(c).json()
    env.pop("signature")
    assert c.post("/v1/me/voiceprints/import", json=env).status_code == 400


def test_foreign_key_rejected(client):
    c, store = client
    _login(c, "alice@x.com")
    # an envelope correctly signed, but by a DIFFERENT (foreign) ed25519 key
    foreign = ReceiptSigner(OTHER_SEED)
    vp = store.get("ws1", "vp_alice")
    from consent_api import _export_payload
    env = foreign.sign(_export_payload(store, vp))
    assert c.post("/v1/me/voiceprints/import", json=env).status_code == 400


# ── §6.3 model/dim mismatch → reject ─────────────────────────

def test_wrong_model_rejected(client):
    c, store = client
    _login(c, "alice@x.com")
    vp = store.get("ws1", "vp_alice")
    from consent_api import _export_payload
    signer = ReceiptSigner(SEED)
    p = _export_payload(store, vp)
    p["embedder_model"] = "some-other-model"      # validly signed, wrong model
    r = c.post("/v1/me/voiceprints/import", json=signer.sign(p))
    assert r.status_code == 400 and r.json()["error"]["message"] == "wrong-model"


def test_wrong_dim_rejected(client):
    c, store = client
    _login(c, "alice@x.com")
    vp = store.get("ws1", "vp_alice")
    from consent_api import _export_payload
    signer = ReceiptSigner(SEED)
    p = _export_payload(store, vp)
    p["embedder_dim"] = 256
    assert c.post("/v1/me/voiceprints/import", json=signer.sign(p)).status_code == 400


# ── §6.4 ownership mismatch → 403 ────────────────────────────

def test_ownership_mismatch_403(client):
    c, store = client
    # alice exports her vp, bob (signed in) tries to import it
    _login(c, "alice@x.com")
    env = _export(c).json()
    _login(c, "bob@x.com")
    r = c.post("/v1/me/voiceprints/import", json=env)
    assert r.status_code == 403 and r.json()["error"]["message"] == "not-owner"


def test_merge_into_voiceprint_you_no_longer_own_403(client):
    c, store = client
    # alice owns vp_alice now; craft a validly-signed envelope (owner_email=bob) whose
    # vid collides with alice's live voiceprint. bob (session) must NOT merge into it —
    # a signed snapshot proves ownership only at export time, not now.
    _login(c, "bob@x.com")
    from consent_api import _export_payload
    p = _export_payload(store, store.get("ws1", "vp_alice"))
    p["owner_email"] = "bob@x.com"          # bob's own snapshot, but vid is alice's live print
    env = ReceiptSigner(SEED).sign(p)
    r = c.post("/v1/me/voiceprints/import", json=env)
    assert r.status_code == 403 and r.json()["error"]["message"] == "not-owner"
    # alice's voiceprint is untouched (name/flags/owner unchanged)
    assert store.get("ws1", "vp_alice").owner_email == "alice@x.com"


# ── §6.5 collision → merge (eviction); absent → create ───────

def test_collision_merges_eviction_aware(client):
    c, store = client
    _login(c, "alice@x.com")
    # give the live vp a full exemplar set, then import a fresh snapshot on top
    vp = store.get("ws1", "vp_alice")
    while len(vp.exemplars) < MAX_EXEMPLARS:
        vp.add_exemplar(_unit(900 + len(vp.exemplars)))
    vp.recompute_centroid()
    store.upsert(vp)

    from consent_api import _export_payload
    extra = _vp("ws1", "vp_alice", "alice@x.com", 7, n=3)
    p = _export_payload(store, extra)
    p["voiceprint_id"] = "vp_alice"
    env = ReceiptSigner(SEED).sign(p)

    r = c.post("/v1/me/voiceprints/import", json=env)
    assert r.status_code == 200 and r.json()["results"][0]["status"] == "merged"
    merged = store.get("ws1", "vp_alice")
    assert len(merged.exemplars) == MAX_EXEMPLARS          # capped, eviction respected


def test_absent_creates(client):
    c, store = client
    _login(c, "alice@x.com")
    env = _export(c, "ws2", "vp_alice2").json()
    _wipe(store, "ws2", "vp_alice2")
    r = c.post("/v1/me/voiceprints/import", json=env)
    assert r.json()["results"][0]["status"] == "created"
    assert store.get("ws2", "vp_alice2") is not None


# ── §6.6 consent flags restored exactly ──────────────────────

def test_consent_flags_restored(client):
    c, store = client
    _login(c, "alice@x.com")
    # export a vp whose flags are non-default
    store.set_flags("ws1", "vp_alice", identify_allowed=False, enroll_allowed=False, actor="test")
    env = _export(c).json()
    assert env["payload"]["identify_allowed"] is False
    assert env["payload"]["enroll_allowed"] is False
    _wipe(store, "ws1", "vp_alice")
    c.post("/v1/me/voiceprints/import", json=env)
    restored = store.get("ws1", "vp_alice")
    assert restored.identify_allowed is False and restored.enroll_allowed is False
    assert restored.name == env["payload"]["name"]


# ── §6.7 import writes a usage_ledger "import" row ───────────

def test_import_ledgered(client):
    c, store = client
    _login(c, "alice@x.com")
    env = _export(c).json()
    _wipe(store, "ws1", "vp_alice")
    c.post("/v1/me/voiceprints/import", json=env)
    events = [e["event"] for e in store.usage_for_voiceprint("ws1", "vp_alice")]
    assert "import" in events
    row = next(e for e in store.usage_for_voiceprint("ws1", "vp_alice") if e["event"] == "import")
    assert row["consumer"] == "alice@x.com" and row["purpose"] == "restore"


# ── §6.8 round-trip determinism ──────────────────────────────

def test_round_trip_determinism(client):
    c, store = client
    _login(c, "alice@x.com")
    env1 = _export(c).json()
    _wipe(store, "ws1", "vp_alice")
    c.post("/v1/me/voiceprints/import", json=env1)
    env2 = _export(c).json()
    # everything but the wall-clock exported_at is equivalent
    p1, p2 = copy.deepcopy(env1["payload"]), copy.deepcopy(env2["payload"])
    for p in (p1, p2):
        p.pop("exported_at")
    assert p1 == p2


# ── export-all bundle ────────────────────────────────────────

def test_export_all_and_bundle_import(client):
    c, store = client
    _login(c, "alice@x.com")
    bundle = c.get("/v1/me/voiceprints/export").json()
    assert bundle["count"] == 2                       # both of alice's, not bob's
    vids = {e["payload"]["voiceprint_id"] for e in bundle["voiceprints"]}
    assert vids == {"vp_alice", "vp_alice2"}

    _wipe(store, "ws1", "vp_alice")
    _wipe(store, "ws2", "vp_alice2")
    r = c.post("/v1/me/voiceprints/import", json=bundle)
    assert r.status_code == 200 and r.json()["imported"] == 2
    assert store.get("ws1", "vp_alice") is not None
    assert store.get("ws2", "vp_alice2") is not None


def test_bundle_partial_restore_surfaces_rejections(client):
    c, store = client
    _login(c, "alice@x.com")
    good = _export(c).json()
    # a foreign-signed item in the same bundle must be rejected without sinking the good one
    foreign = ReceiptSigner(OTHER_SEED)
    from consent_api import _export_payload
    bad = foreign.sign(_export_payload(store, store.get("ws2", "vp_alice2")))
    _wipe(store, "ws1", "vp_alice")
    r = c.post("/v1/me/voiceprints/import", json={"voiceprints": [good, bad]})
    assert r.status_code == 200
    body = r.json()
    assert body["imported"] == 1
    statuses = {res["status"] for res in body["results"]}
    assert statuses == {"created", "rejected"}
    assert any(res.get("reason") == "bad-signature" for res in body["results"])
    assert store.get("ws1", "vp_alice") is not None
