"""Consent plane (WS3/WS4/WS5): owner_email, access flags, enforcement, usage ledger.

Uses a stub embedder (fixed unit vector → guaranteed self-match) so the enforcement
and ledger behaviour is testable without the ONNX model present.
"""
import os
import sqlite3

import numpy as np

from config import ID_EMBEDDING_DIM as DIM
from fpm.types import Segment
from tests.mock_diarizer import MockDiarizer
from fpm.enroll import enroll, identity_voiceprint_id
from fpm.identify import SessionIdentifier
from fpm.store.models import Voiceprint
from fpm.store.store import VoiceprintStore

KEY = os.urandom(32)
SR = 16_000


def _store(tmp_path):
    return VoiceprintStore(db_path=tmp_path / "vp.db", key=KEY).open()


def _unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class _StubEmbedder:
    """Returns one fixed unit vector for any audio → every clip self-matches."""

    def __init__(self, seed: int = 1):
        self._vec = _unit(seed)

    def extract(self, audio, sample_rate=SR):
        if audio is None or len(audio) == 0:
            return None
        return self._vec.copy()


def _vp(ws, vid, name="", seed=0, **kw):
    vp = Voiceprint(vid, ws, name, **kw)
    for i in range(3):
        vp.add_exemplar(_unit(seed * 10 + i))
    vp.recompute_centroid()
    return vp


# ── WS3: owner_email + flags persistence ─────────────────────

def test_owner_email_and_flags_roundtrip(tmp_path):
    s = _store(tmp_path)
    s.upsert(_vp("ws1", "vp_a", "Alice", seed=1,
                 owner_email="alice@x.com", identify_allowed=False))
    got = s.get("ws1", "vp_a")
    assert got.owner_email == "alice@x.com"
    assert got.enroll_allowed is True and got.identify_allowed is False
    s.close()


def test_migration_adds_consent_columns(tmp_path):
    # Pre-consent-plane DB: voiceprints table without the new columns.
    db = tmp_path / "vp.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE voiceprints(voiceprint_id TEXT PRIMARY KEY, workspace_id TEXT, "
        "name TEXT, centroid BLOB, exemplars BLOB, exemplar_count INTEGER, enroll_count INTEGER, "
        "total_duration_sec REAL, quality_score REAL, created_at TEXT, updated_at TEXT, last_seen_at TEXT)"
    )
    con.commit()
    con.close()
    s = _store(tmp_path)  # open() runs the ALTER migration
    cols = {r[1] for r in s._conn.execute("PRAGMA table_info(voiceprints)")}
    assert {"owner_email", "enroll_allowed", "identify_allowed"} <= cols
    s.close()


# ── WS4: usage ledger ────────────────────────────────────────

def test_log_usage_newest_first(tmp_path):
    s = _store(tmp_path)
    s.upsert(_vp("ws1", "vp_a", seed=2))
    s.log_usage("ws1", "vp_a", "identify", "recato", "first")
    s.log_usage("ws1", "vp_a", "identify", "conclave", "second")
    rows = s.usage_for_voiceprint("ws1", "vp_a")
    assert [r["purpose"] for r in rows] == ["second", "first"]  # DESC
    assert rows[0]["consumer"] == "conclave" and rows[0]["event"] == "identify"
    s.close()


def test_set_flags_writes_ledger_and_refreshes_cache(tmp_path):
    s = _store(tmp_path)
    s.upsert(_vp("ws1", "vp_a", seed=3))
    assert s.identify_allowed("ws1", "vp_a") is True
    assert s.set_flags("ws1", "vp_a", identify_allowed=False, actor="dashboard") is True
    assert s.identify_allowed("ws1", "vp_a") is False        # hot cache refreshed
    assert s.get("ws1", "vp_a").identify_allowed is False     # and persisted
    control = [r for r in s.usage_for_voiceprint("ws1", "vp_a") if r["event"] == "control"]
    assert control and "identify_allowed=off" in control[0]["purpose"]
    s.close()


def test_set_flags_workspace_scoped(tmp_path):
    s = _store(tmp_path)
    s.upsert(_vp("ws1", "vp_a", seed=4))
    assert s.set_flags("ws2", "vp_a", identify_allowed=False) is False  # wrong workspace
    assert s.get("ws1", "vp_a").identify_allowed is True
    s.close()


def test_find_by_owner_email(tmp_path):
    s = _store(tmp_path)
    s.upsert(_vp("ws1", "vp_a", "Alice", seed=5, owner_email="alice@x.com"))
    s.upsert(_vp("ws2", "vp_b", "Alice", seed=6, owner_email="alice@x.com"))
    s.upsert(_vp("ws1", "vp_c", "Bob", seed=7, owner_email="bob@x.com"))
    found = s.find_by_owner_email("alice@x.com")
    assert set(found) == {("ws1", "vp_a"), ("ws2", "vp_b")}  # both workspaces, isolated entries
    s.close()


def test_delete_logs_forget(tmp_path):
    s = _store(tmp_path)
    s.upsert(_vp("ws1", "vp_a", seed=8))
    res = s.delete("ws1", "vp_a", actor="dashboard")
    assert res and res.deleted is True                        # DeleteResult, truthy on a real delete
    assert res.ledger_row_id is not None and res.deleted_at   # proof anchor returned
    assert s.get("ws1", "vp_a") is None                       # gone
    forget = [r for r in s.usage_for_voiceprint("ws1", "vp_a") if r["event"] == "forget"]
    assert forget and forget[0]["consumer"] == "dashboard"    # ledger survives the delete
    s.close()


# ── WS5: enforcement ─────────────────────────────────────────

def test_enroll_sets_owner_email_and_logs(tmp_path):
    s = _store(tmp_path)
    emb = _StubEmbedder()
    r = enroll(s, emb, "ws1", "alice@x.com", _unit(0), SR, consumer="recato")
    assert r.status == "created"
    vp = s.get("ws1", r.voiceprint_id)
    assert vp.owner_email == "alice@x.com" and vp.name == "alice@x.com"
    assert any(row["event"] == "enroll" for row in s.usage_for_voiceprint("ws1", r.voiceprint_id))
    s.close()


def test_enroll_refused_when_disabled(tmp_path):
    s = _store(tmp_path)
    emb = _StubEmbedder()
    r = enroll(s, emb, "ws1", "alice@x.com", _unit(0), SR)
    assert r.status == "created"
    assert s.set_flags("ws1", r.voiceprint_id, enroll_allowed=False) is True
    r2 = enroll(s, emb, "ws1", "alice@x.com", _unit(0), SR)
    assert r2.status == "rejected" and "disabled" in r2.reason
    # not strengthened: still a single exemplar
    assert len(s.get("ws1", r.voiceprint_id).exemplars) == 1
    s.close()


def _run_session(store, emb, ws="ws1"):
    # one speaker0 turn, repeated so it locks
    script = [Segment(i * 2.0, i * 2.0 + 2.0, "speaker0") for i in range(4)]
    audio = np.ones(int(8.0 * SR), dtype=np.float32) * 0.1
    ident = SessionIdentifier(store, emb, MockDiarizer(script), ws, sample_rate=SR)
    ident.start()
    step = int(0.5 * SR)
    out = []
    for i in range(0, len(audio), step):
        out.extend(ident.feed(audio[i:i + step], SR))
    out.extend(ident.finish())
    return ident.transcript()


def test_identify_named_when_allowed(tmp_path):
    s = _store(tmp_path)
    emb = _StubEmbedder()
    enroll(s, emb, "ws1", "alice@x.com", _unit(0), SR)        # Alice enrolled
    segs = _run_session(s, emb)
    named = [x for x in segs if x.voiceprint_id is not None]
    assert named and all(x.name == "alice@x.com" for x in named)
    s.close()


def test_identify_anonymous_when_disabled(tmp_path):
    s = _store(tmp_path)
    emb = _StubEmbedder()
    r = enroll(s, emb, "ws1", "alice@x.com", _unit(0), SR)
    assert s.set_flags("ws1", r.voiceprint_id, identify_allowed=False) is True
    segs = _run_session(s, emb)
    resolved = [x for x in segs if x.voiceprint_id is not None]
    assert resolved, "speaker never resolved"
    assert all(x.name is None for x in resolved)              # "stay anonymous": cluster, no name
    # ...but a usage row was still logged (the match happened)
    assert any(row["event"] == "identify"
               for row in s.usage_for_voiceprint("ws1", r.voiceprint_id))
    s.close()
