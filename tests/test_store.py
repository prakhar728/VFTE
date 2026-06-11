"""A.5–A.6 — encrypted, workspace-scoped voiceprint store."""
import sqlite3
import os

import numpy as np

from fpm.store import crypto
from fpm.store.models import Voiceprint
from fpm.store.store import VoiceprintStore

KEY = os.urandom(32)
DIM = 512


def _unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _store(tmp_path):
    return VoiceprintStore(db_path=tmp_path / "vp.db", key=KEY).open()


def _vp(ws, vid, name="", seed=0, n_ex=3):
    vp = Voiceprint(vid, ws, name)
    for i in range(n_ex):
        vp.add_exemplar(_unit(seed * 10 + i))
    vp.recompute_centroid()
    return vp


def test_upsert_get_roundtrip(tmp_path):
    s = _store(tmp_path)
    s.upsert(_vp("ws1", "vp_a", "Alice", seed=1))
    got = s.get("ws1", "vp_a")
    assert got.name == "Alice" and got.voiceprint_id == "vp_a"
    assert got.centroid.shape == (DIM,)
    assert len(got.exemplars) == 3
    s.close()


def test_encrypted_at_rest(tmp_path):
    s = _store(tmp_path)
    s.upsert(_vp("ws1", "vp_a", seed=2))
    s.close()
    raw = sqlite3.connect(tmp_path / "vp.db").execute(
        "SELECT centroid FROM voiceprints"
    ).fetchone()[0]
    assert crypto.is_encrypted(raw) and raw[:4] == b"FPE1"


def test_workspace_isolation(tmp_path):
    s = _store(tmp_path)
    s.upsert(_vp("ws1", "vp_a", "Alice", seed=3))
    assert s.get("ws2", "vp_a") is None                      # cross-ws read blocked
    assert s.set_name("ws2", "vp_a", "Mallory", "x") is False  # cross-ws write blocked
    assert s.get("ws1", "vp_a").name == "Alice"              # untouched
    assert "vp_a" not in s.centroids("ws2")
    assert "vp_a" in s.centroids("ws1")
    s.close()


def test_set_name_audited(tmp_path):
    s = _store(tmp_path)
    s.upsert(_vp("ws1", "vp_a", seed=4))
    assert s.set_name("ws1", "vp_a", "Bob", "conclave") is True
    assert s.get("ws1", "vp_a").name == "Bob"
    s.close()
    audit = sqlite3.connect(tmp_path / "vp.db").execute(
        "SELECT new_name, actor FROM binding_audit"
    ).fetchall()
    assert ("Bob", "conclave") in audit


def test_vocab(tmp_path):
    s = _store(tmp_path)
    assert s.get_vocab("ws1") == []
    s.set_vocab("ws1", ["Phala", "TDX"])
    assert s.get_vocab("ws1") == ["Phala", "TDX"]
    s.close()


def test_db_perms_0600(tmp_path):
    s = _store(tmp_path)
    s.close()
    assert ((tmp_path / "vp.db").stat().st_mode & 0o777) == 0o600
