"""Task #3 — hear-the-clip proposals + auto-recognition transparency (FPM side).

Covers:
  (b) proposals carry a clip_ref; the owner mints a signed, expiring, owner-scoped clip
      capability; FPM never stores/streams audio (only the ref).
  (c) a *consented* recognition is recorded + emailed (always, incl. read-only identify)
      and surfaced in the subject's transparency inbox; the detail page is metadata-only.
"""
import base64
import json
import os
import sqlite3
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import config
import main
from auth import SESSION_COOKIE, Caller, GoogleOAuth, SessionManager, TokenAuth
from fpm import receipts
from fpm.identify import SessionIdentifier
from fpm.receipts import ReceiptSigner
from fpm.store.models import Voiceprint
from fpm.store.store import VoiceprintStore, new_voiceprint_id
from fpm.types import Segment
from tests.mock_diarizer import MockDiarizer

DIM = 512
KEY = os.urandom(32)
SEED = os.urandom(32)
CONCLAVE = "tok-conclave"
CLIP = {"conclave_session_id": "sess-1", "native_meeting_id": "sess-1", "start": 3.0, "end": 7.5}


# ── store-level ──────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    s = VoiceprintStore(db_path=tmp_path / "vp.db", key=KEY).open()
    yield s
    s.close()


def _anon(store, ws="ws1", name="", email=""):
    vp = Voiceprint(new_voiceprint_id(), ws, name=name, owner_email=email)
    vp.add_exemplar(np.ones(DIM, dtype=np.float32) / np.sqrt(DIM))
    vp.recompute_centroid()
    store.upsert(vp)
    return vp.voiceprint_id


def test_propose_stores_and_roundtrips_clip_ref(store):
    vid = _anon(store)
    p = store.propose("ws1", vid, "alice@x.com", "host@x.com", "Alice",
                      clip_ref=CLIP, source="recognition", confidence=0.82)
    assert p["clip_ref"] == CLIP            # parsed back to a dict, not JSON text
    assert p["source"] == "recognition" and p["confidence"] == pytest.approx(0.82)
    # surfaced in the subject's pending inbox feed
    pend = store.list_pending_for_email("alice@x.com")
    assert pend and pend[0]["clip_ref"] == CLIP


def test_propose_without_clip_ref_is_none(store):
    vid = _anon(store)
    p = store.propose("ws1", vid, "alice@x.com", "host@x.com", "Alice")
    assert p["clip_ref"] is None and p["source"] == "tag"


def test_migration_adds_proposal_columns_to_legacy_db(tmp_path):
    """An existing proposals table (pre-#3) gains clip_ref/source/confidence on open."""
    db = tmp_path / "legacy.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE proposals(proposal_id TEXT PRIMARY KEY, workspace_id TEXT, "
        "voiceprint_id TEXT, proposed_email TEXT, proposed_by TEXT, proposed_name TEXT, "
        "status TEXT, created_at TEXT, confirmed_at TEXT, denied_at TEXT)"
    )
    con.commit()
    con.close()
    s = VoiceprintStore(db_path=db, key=KEY).open()
    cols = {r[1] for r in s._conn.execute("PRAGMA table_info(proposals)")}
    assert {"clip_ref", "source", "confidence"} <= cols
    vid = _anon(s)
    p = s.propose("ws1", vid, "a@x.com", "h@x.com", "A", clip_ref=CLIP)
    assert p["clip_ref"] == CLIP
    s.close()


def test_recognition_roundtrip_newest_first(store):
    store.add_recognition("ws1", "vp_a", "Alice@X.com", native_meeting_id="m1", app="conclave")
    store.add_recognition("ws1", "vp_a", "alice@x.com", native_meeting_id="m2", app="conclave",
                          meeting_title="Sync")
    rows = store.list_recognitions_for_email("alice@x.com")   # lowercased join
    assert [r["native_meeting_id"] for r in rows] == ["m2", "m1"]  # newest first
    assert rows[0]["meeting_title"] == "Sync"
    one = store.get_recognition(rows[0]["recognition_id"])
    assert one["owner_email"] == "alice@x.com" and one["app"] == "conclave"


# ── read-only identify still logs (Part c, decision 2) ───────

class _StubEmbedder:
    def __init__(self):
        v = np.random.default_rng(1).standard_normal(DIM).astype(np.float32)
        self._vec = v / np.linalg.norm(v)

    def extract(self, audio, sample_rate=16_000):
        return None if audio is None or len(audio) == 0 else self._vec.copy()


def _named_vp(seed=0):
    vp = Voiceprint("vp_alice", "ws1", "alice@x.com", owner_email="alice@x.com")
    for i in range(3):
        v = np.random.default_rng(1).standard_normal(DIM).astype(np.float32)
        vp.add_exemplar(v / np.linalg.norm(v))
    vp.recompute_centroid()
    return vp


def _run(store, *, read_only, meeting_id=None):
    emb = _StubEmbedder()
    script = [Segment(i * 2.0, i * 2.0 + 2.0, "speaker0") for i in range(4)]
    audio = np.ones(int(8.0 * 16_000), dtype=np.float32) * 0.1
    ident = SessionIdentifier(store, emb, MockDiarizer(script), "ws1", sample_rate=16_000,
                              read_only=read_only, meeting_id=meeting_id)
    ident.start()
    step = int(0.5 * 16_000)
    for i in range(0, len(audio), step):
        ident.feed(audio[i:i + step], 16_000)
    ident.finish()
    return ident.transcript()


def test_readonly_identify_is_still_logged(store):
    store.upsert(_named_vp())
    _run(store, read_only=True, meeting_id="native-42")
    rows = [r for r in store.usage_for_voiceprint("ws1", "vp_alice") if r["event"] == "identify"]
    assert rows, "read-only recognition must still be logged (Part c)"
    assert "native-42" in rows[0]["purpose"]   # meeting context carried


# ── M2M endpoints (propose + recognitions) ───────────────────

@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONSENT_AUTOCONFIRM", False)
    monkeypatch.setattr(config, "DEV_LOGIN", True)
    monkeypatch.setattr(config, "NOTIFY_EMAIL", False)     # log-only email in tests
    monkeypatch.setattr(config, "CONCLAVE_CLIP_BASE_URL", "http://conclave.test")
    monkeypatch.setattr(config, "CLIP_CAP_TTL_SEC", 300)
    s = VoiceprintStore(db_path=tmp_path / "vp.db", key=KEY).open()
    main.app.state.store = s
    main.app.state.embedder = object()
    main.app.state.auth = TokenAuth({
        CONCLAVE: Caller("conclave", frozenset({"knowledge", "identify"}),
                         workspaces=frozenset({"ws1"})),
    })
    main.app.state.sessions = SessionManager("test-secret", 3600)
    main.app.state.oauth = GoogleOAuth("", "", "http://localhost/auth/callback")
    main.app.state.receipt_signer = ReceiptSigner(SEED)
    with TestClient(main.app) as c:
        yield c, s, monkeypatch
    for attr in ("store", "embedder", "auth", "sessions", "oauth", "receipt_signer"):
        setattr(main.app.state, attr, None)


def _h(token=CONCLAVE):
    return {"Authorization": f"Bearer {token}"}


def _login(c, email):
    c.cookies.set(SESSION_COOKIE, c.app.state.sessions.issue(email))


def _mk_named(store, vid, email, *, identify_allowed=True):
    vp = Voiceprint(vid, "ws1", email, owner_email=email, identify_allowed=identify_allowed)
    vp.add_exemplar(np.ones(DIM, dtype=np.float32) / np.sqrt(DIM))
    vp.recompute_centroid()
    store.upsert(vp)


def test_propose_endpoint_carries_clip_ref(api):
    c, s, _ = api
    _mk_named(s, "vp_x", "owner@x.com")     # exists; tag targets a different person → pending
    r = c.post("/v1/propose", headers=_h(), json={
        "workspace": "ws1", "voiceprint_id": "vp_x", "proposed_email": "alice@x.com",
        "proposed_by": "host@x.com", "proposed_name": "Alice",
        "clip_ref": CLIP, "source": "recognition", "confidence": 0.9})
    assert r.status_code == 200 and r.json()["status"] == "pending"
    assert s.get_proposal(r.json()["proposal_id"])["clip_ref"] == CLIP


def test_recognition_recorded_for_consented(api):
    c, s, _ = api
    _mk_named(s, "vp_alice", "alice@x.com")
    r = c.post("/v1/recognitions", headers=_h(), json={
        "workspace": "ws1", "voiceprint_id": "vp_alice",
        "native_meeting_id": "m1", "app": "conclave", "meeting_title": "Standup"})
    assert r.status_code == 200
    body = r.json()
    assert body["recorded"] is True and body["notified"] is False   # log-only
    assert s.list_recognitions_for_email("alice@x.com")[0]["native_meeting_id"] == "m1"


def test_recognition_skipped_for_unclaimed(api):
    c, s, _ = api
    _anon(s, name="")   # no owner
    vid = s.list_ids("ws1")[0]
    r = c.post("/v1/recognitions", headers=_h(),
               json={"workspace": "ws1", "voiceprint_id": vid})
    assert r.status_code == 200 and r.json()["recorded"] is False
    assert s.list_recognitions_for_email("") == []


def test_recognition_skipped_when_identify_blocked(api):
    c, s, _ = api
    _mk_named(s, "vp_a", "alice@x.com", identify_allowed=False)
    r = c.post("/v1/recognitions", headers=_h(),
               json={"workspace": "ws1", "voiceprint_id": "vp_a"})
    assert r.json()["recorded"] is False
    assert s.list_recognitions_for_email("alice@x.com") == []


def test_recognition_unknown_vp_404(api):
    c, _, _ = api
    r = c.post("/v1/recognitions", headers=_h(),
               json={"workspace": "ws1", "voiceprint_id": "vp_nope"})
    assert r.status_code == 404


# ── owner-authed: clip-url capability + inbox + detail ───────

def _unpack(cap: str) -> dict:
    pad = "=" * (-len(cap) % 4)
    return json.loads(base64.urlsafe_b64decode(cap + pad))


def test_clip_url_mints_verifiable_capability(api):
    c, s, _ = api
    _mk_named(s, "vp_x", "owner@x.com")
    pid = s.propose("ws1", "vp_x", "alice@x.com", "host@x.com", "Alice", clip_ref=CLIP)["proposal_id"]
    _login(c, "alice@x.com")
    r = c.post(f"/v1/me/proposals/{pid}/clip-url")
    assert r.status_code == 200
    url = r.json()["url"]
    assert "/api/transcripts/sessions/sess-1/audio" in url and "start=3.0" in url and "end=7.5" in url
    cap = url.split("cap=")[1]
    env = _unpack(cap)
    signer: ReceiptSigner = main.app.state.receipt_signer
    assert receipts.verify_with_pubkey(env, signer.public_key_raw())    # genuine signature
    assert env["payload"]["purpose"] == "clip-cap"
    assert env["payload"]["sub"] == "alice@x.com"
    assert env["payload"]["clip_ref"] == CLIP
    assert env["payload"]["exp"] > time.time()


def test_clip_url_owner_scoped(api):
    c, s, _ = api
    _mk_named(s, "vp_x", "owner@x.com")
    pid = s.propose("ws1", "vp_x", "alice@x.com", "host@x.com", "Alice", clip_ref=CLIP)["proposal_id"]
    _login(c, "mallory@x.com")     # not the tagged target
    assert c.post(f"/v1/me/proposals/{pid}/clip-url").status_code == 403


def test_clip_url_no_clip_404(api):
    c, s, _ = api
    _mk_named(s, "vp_x", "owner@x.com")
    pid = s.propose("ws1", "vp_x", "alice@x.com", "host@x.com", "Alice")["proposal_id"]  # no clip
    _login(c, "alice@x.com")
    assert c.post(f"/v1/me/proposals/{pid}/clip-url").status_code == 404


def test_recognitions_inbox_lists_own(api):
    c, s, _ = api
    s.add_recognition("ws1", "vp_a", "alice@x.com", native_meeting_id="m1", app="conclave")
    _login(c, "alice@x.com")
    body = c.get("/v1/me/recognitions").json()
    assert body["email"] == "alice@x.com"
    assert [r["native_meeting_id"] for r in body["recognitions"]] == ["m1"]


def test_recognition_detail_metadata_only_and_owner_gated(api):
    c, s, _ = api
    rec = s.add_recognition("ws1", "vp_a", "alice@x.com", native_meeting_id="m1",
                            app="conclave", meeting_title="Board sync")
    rid = rec["recognition_id"]
    _login(c, "alice@x.com")
    body = c.get(f"/v1/me/recognitions/{rid}").json()
    assert body["recognition"]["meeting_title"] == "Board sync"
    assert body["transcript_access"] is False
    # transcript-access guard: NO meeting content ever crosses this surface
    blob = json.dumps(body).lower()
    assert "transcript" not in body["recognition"] and "segments" not in blob and "text" not in body["recognition"]
    assert "forget" in body["controls"] and "stay_anonymous" in body["controls"]
    # a different signed-in user cannot read it
    _login(c, "bob@x.com")
    assert c.get(f"/v1/me/recognitions/{rid}").status_code == 403
