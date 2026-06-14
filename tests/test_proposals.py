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
