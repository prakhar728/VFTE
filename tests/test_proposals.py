"""P4 — proposal state on the voiceprint store (propose → confirm/deny → consent_resolve).

Store-level unit tests for the email-binding handshake. Mirrors test_knowledge_api's
tmp-store style; no embedder/diarizer needed (proposals are pure metadata).
"""
import os

import numpy as np
import pytest

from fpm.store.models import Voiceprint
from fpm.store.store import VoiceprintStore, new_voiceprint_id


@pytest.fixture
def store(tmp_path):
    s = VoiceprintStore(db_path=tmp_path / "vp.db", key=os.urandom(32)).open()
    yield s
    s.close()


def _anon(store, ws="ws1", name=""):
    """Mint an anonymous voiceprint to attach proposals to."""
    vp = Voiceprint(new_voiceprint_id(), ws, name=name)
    vp.add_exemplar(np.ones(512, dtype=np.float32) / np.sqrt(512))
    vp.recompute_centroid()
    store.upsert(vp)
    return vp.voiceprint_id


def _named(store, ws, name, email, identify_allowed=True):
    vp = Voiceprint(new_voiceprint_id(), ws, name=name, owner_email=email,
                    identify_allowed=identify_allowed)
    vp.add_exemplar(np.ones(512, dtype=np.float32) / np.sqrt(512))
    vp.recompute_centroid()
    store.upsert(vp)
    return vp.voiceprint_id


# ── step 1: idempotent propose ───────────────────────────────

def test_propose_creates_pending(store):
    vid = _anon(store)
    p = store.propose("ws1", vid, "alice@x.com", "host@x.com", "Alice")
    assert p["proposal_id"].startswith("prop_")
    assert p["status"] == "pending"
    assert p["voiceprint_id"] == vid
    assert p["proposed_email"] == "alice@x.com"
    assert p["proposed_by"] == "host@x.com"
    assert p["proposed_name"] == "Alice"
    assert p["created_at"] and p["confirmed_at"] is None and p["denied_at"] is None


def test_propose_lowercases_emails(store):
    vid = _anon(store)
    p = store.propose("ws1", vid, "Alice@X.com", "Host@X.com", "Alice")
    assert p["proposed_email"] == "alice@x.com"
    assert p["proposed_by"] == "host@x.com"


def test_propose_idempotent_returns_same_id(store):
    vid = _anon(store)
    a = store.propose("ws1", vid, "alice@x.com", "host@x.com", "Alice")
    # re-propose the same (ws, vid, email): same row, original proposed_name kept
    b = store.propose("ws1", vid, "ALICE@x.com", "someone@x.com", "Alice Renamed")
    assert b["proposal_id"] == a["proposal_id"]
    assert b["proposed_name"] == "Alice"
    assert b["proposed_by"] == "host@x.com"
    assert b["status"] == "pending"


def test_propose_distinct_email_distinct_row(store):
    vid = _anon(store)
    a = store.propose("ws1", vid, "alice@x.com", "host@x.com", "Alice")
    b = store.propose("ws1", vid, "bob@x.com", "host@x.com", "Bob")
    assert a["proposal_id"] != b["proposal_id"]


def test_propose_distinct_voiceprint_distinct_row(store):
    v1, v2 = _anon(store), _anon(store)
    a = store.propose("ws1", v1, "alice@x.com", "host@x.com", "Alice")
    b = store.propose("ws1", v2, "alice@x.com", "host@x.com", "Alice")
    assert a["proposal_id"] != b["proposal_id"]


def test_get_proposal_roundtrip(store):
    vid = _anon(store)
    p = store.propose("ws1", vid, "alice@x.com", "host@x.com", "Alice")
    got = store.get_proposal(p["proposal_id"])
    assert got == p
    assert store.get_proposal("prop_nope") is None


# ── step 2: confirm / deny ───────────────────────────────────

def test_confirm_binds_owner_and_name(store):
    vid = _anon(store)
    p = store.propose("ws1", vid, "alice@x.com", "host@x.com", "Alice")
    res = store.confirm_proposal(p["proposal_id"], actor="alice@x.com")
    assert res == {"voiceprint_id": vid, "name": "Alice", "owner_email": "alice@x.com"}
    vp = store.get("ws1", vid)
    assert vp.owner_email == "alice@x.com" and vp.name == "Alice"
    got = store.get_proposal(p["proposal_id"])
    assert got["status"] == "confirmed" and got["confirmed_at"] and got["denied_at"] is None


def test_confirm_writes_binding_audit_and_ledger(store):
    vid = _anon(store)
    p = store.propose("ws1", vid, "alice@x.com", "host@x.com", "Alice")
    store.confirm_proposal(p["proposal_id"], actor="alice@x.com")
    audit = store.audit_entries("ws1")
    assert ("", "Alice", "alice@x.com") in [(a["old_name"], a["new_name"], a["actor"]) for a in audit]
    assert any(u["event"] == "name_bind" for u in store.usage_for_voiceprint("ws1", vid))


def test_confirm_idempotent_on_confirmed(store):
    vid = _anon(store)
    p = store.propose("ws1", vid, "alice@x.com", "host@x.com", "Alice")
    a = store.confirm_proposal(p["proposal_id"], actor="alice@x.com")
    b = store.confirm_proposal(p["proposal_id"], actor="alice@x.com")
    assert a == b
    # the bind ran exactly once (no second audit row from a re-confirm)
    name_binds = [u for u in store.usage_for_voiceprint("ws1", vid) if u["event"] == "name_bind"]
    assert len(name_binds) == 1


def test_deny_leaves_no_binding(store):
    vid = _anon(store)
    p = store.propose("ws1", vid, "alice@x.com", "host@x.com", "Alice")
    res = store.deny_proposal(p["proposal_id"], actor="alice@x.com")
    assert res["status"] == "denied"
    vp = store.get("ws1", vid)
    assert vp.owner_email == "" and vp.name == ""
    got = store.get_proposal(p["proposal_id"])
    assert got["status"] == "denied" and got["denied_at"] and got["confirmed_at"] is None


def test_confirm_and_deny_unknown_return_none(store):
    assert store.confirm_proposal("prop_nope") is None
    assert store.deny_proposal("prop_nope") is None


def test_list_pending_for_email(store):
    v1, v2, v3 = _anon(store), _anon(store), _anon(store)
    store.propose("ws1", v1, "alice@x.com", "host@x.com", "Alice")
    p2 = store.propose("ws1", v2, "alice@x.com", "host@x.com", "Alice")
    store.propose("ws1", v3, "bob@x.com", "host@x.com", "Bob")
    store.confirm_proposal(p2["proposal_id"], actor="alice@x.com")
    pending = store.list_pending_for_email("ALICE@x.com")
    assert {p["voiceprint_id"] for p in pending} == {v1}  # v2 confirmed, v3 is bob's


# ── step 3: consent_resolve (read-side gate) ─────────────────

def test_resolve_named(store):
    vid = _named(store, "ws1", "Alice", "alice@x.com")
    assert store.consent_resolve("ws1", vid) == {
        "name": "Alice", "owner_email": "alice@x.com", "visibility": "named"}


def test_resolve_anonymous_when_identify_blocked(store):
    vid = _named(store, "ws1", "Alice", "alice@x.com", identify_allowed=False)
    r = store.consent_resolve("ws1", vid)
    assert r["name"] is None and r["visibility"] == "anonymous"
    assert r["owner_email"] == "alice@x.com"  # owner still bound; only the name is withheld


def test_resolve_unbound_name_null(store):
    vid = _anon(store)  # exists, no name, no owner
    assert store.consent_resolve("ws1", vid) == {
        "name": None, "owner_email": None, "visibility": "anonymous"}


def test_resolve_unknown_vid(store):
    assert store.consent_resolve("ws1", "vp_nope") == {
        "name": None, "owner_email": None, "visibility": "unknown"}
