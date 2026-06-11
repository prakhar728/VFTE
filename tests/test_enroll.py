"""A.8 — enrollment (labeled audio → quality-gated voiceprint)."""
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from fpm.embed.onnx_embedder import OnnxSpeakerEmbedder
from fpm.enroll import enroll, identity_voiceprint_id
from fpm.store.store import VoiceprintStore

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "campplus.onnx"
FIX = Path(__file__).parent / "fixtures" / "speakers"
KEY = os.urandom(32)

pytestmark = pytest.mark.skipif(
    not MODEL.exists(), reason="embedder model missing — run scripts/fetch_models.sh"
)


def _audio(name):
    a, sr = sf.read(FIX / f"{name}.wav", dtype="float32")
    return (a if a.ndim == 1 else a.mean(axis=1)), sr


@pytest.fixture
def ctx(tmp_path):
    store = VoiceprintStore(db_path=tmp_path / "vp.db", key=KEY).open()
    embedder = OnnxSpeakerEmbedder(MODEL).load()
    yield store, embedder
    store.close()


def test_create_then_strengthen(ctx):
    store, emb = ctx
    a1, sr = _audio("spkA_1")
    r1 = enroll(store, emb, "ws1", "Alice", a1, sr)
    assert r1.status == "created"
    vp = store.get("ws1", r1.voiceprint_id)
    assert vp.name == "Alice" and len(vp.exemplars) == 1

    a2, sr2 = _audio("spkA_2")  # same speaker
    r2 = enroll(store, emb, "ws1", "Alice", a2, sr2)
    assert r2.status == "updated" and r2.voiceprint_id == r1.voiceprint_id
    assert len(store.get("ws1", r1.voiceprint_id).exemplars) == 2


def test_inconsistent_clip_rejected(ctx):
    store, emb = ctx
    a1, sr = _audio("spkA_1")
    enroll(store, emb, "ws1", "Alice", a1, sr)
    b1, srb = _audio("spkB_1")  # different speaker under Alice's identity
    r = enroll(store, emb, "ws1", "Alice", b1, srb)
    assert r.status == "rejected"
    assert len(store.get("ws1", r.voiceprint_id).exemplars) == 1  # not polluted


def test_short_audio_rejected(ctx):
    store, emb = ctx
    r = enroll(store, emb, "ws1", "Alice", np.zeros(8000, dtype=np.float32), 16000)
    assert r.status == "rejected"


def test_deterministic_id_and_workspace_scoped(ctx):
    store, emb = ctx
    a1, sr = _audio("spkA_1")
    r = enroll(store, emb, "ws1", "Alice", a1, sr)
    assert r.voiceprint_id == identity_voiceprint_id("ws1", "Alice")
    assert store.get("ws2", r.voiceprint_id) is None  # not visible in another workspace
