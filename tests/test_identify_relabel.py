"""C.5 — anonymous mint + retro-relabel + nameable-later."""
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from fpm.diarize.base import Segment
from fpm.diarize.mock import MockDiarizer
from fpm.embed.onnx_embedder import OnnxSpeakerEmbedder
from fpm.enroll import enroll, identity_voiceprint_id
from fpm.identify import SessionIdentifier
from fpm.store.store import VoiceprintStore

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "campplus.onnx"
FIX = Path(__file__).parent / "fixtures" / "speakers"
SR = 16_000
KEY = os.urandom(32)

pytestmark = pytest.mark.skipif(
    not MODEL.exists(), reason="embedder model missing — run scripts/fetch_models.sh"
)


def _wav(name):
    a, sr = sf.read(FIX / f"{name}.wav", dtype="float32")
    assert sr == SR
    return a if a.ndim == 1 else a.mean(axis=1)


def _run(store, emb, audio, script, ws="ws1", **kw):
    ident = SessionIdentifier(store, emb, MockDiarizer(script), ws, **kw)
    ident.start()
    step = int(0.5 * SR)
    for i in range(0, len(audio), step):
        ident.feed(audio[i:i + step], SR)
    return ident


@pytest.fixture
def ctx(tmp_path):
    store = VoiceprintStore(db_path=tmp_path / "vp.db", key=KEY).open()
    emb = OnnxSpeakerEmbedder(MODEL).load()
    yield store, emb
    store.close()


def test_late_lock_relabels_earlier_chunks(ctx):
    store, emb = ctx
    enroll(store, emb, "ws1", "Alice", _wav("spkA_1"), SR)
    alice_id = identity_voiceprint_id("ws1", "Alice")

    # all Alice (spkA). First scripted span is 0.6 s — too short to embed → PENDING.
    # The embeddable spans are FULL fixture clips (the embedder is unreliable on
    # partial sub-spans of these short fixtures — see docs/spikes.md embedder note).
    audio = np.concatenate([_wav("spkA_1"), _wav("spkA_2")])  # 8 s
    script = [Segment(0.0, 0.6, "speaker0"),   # PENDING (no embedding)
              Segment(0.0, 4.0, "speaker0"),   # full spkA_1 → MATCH (vote 1)
              Segment(4.0, 8.0, "speaker0")]   # full spkA_2 → MATCH (vote 2 → lock + relabel)
    ident = _run(store, emb, audio, script)

    sealed = ident.seal()
    # the early PENDING chunk is retro-corrected to Alice in the final transcript
    assert sealed[0].decision == "RELABELED"
    assert sealed[0].voiceprint_id == alice_id and sealed[0].name == "Alice"
    # and every Alice chunk ends up as Alice
    assert all(s.voiceprint_id == alice_id for s in sealed)


def test_unknown_minted_then_nameable_later(ctx):
    store, emb = ctx
    # nobody enrolled: spkB is unknown → minted anonymous, then named via knowledge channel
    audio = np.concatenate([_wav("spkB_1"), _wav("spkB_1"), _wav("spkB_1")])  # 12 s
    script = [Segment(0, 4, "s0"), Segment(4, 8, "s0"), Segment(8, 12, "s0")]
    ident = _run(store, emb, audio, script)
    sealed = ident.seal()

    anon_id = next(s.voiceprint_id for s in sealed if s.voiceprint_id)
    assert store.get("ws1", anon_id).name == ""        # anonymous in store
    assert all(s.name is None for s in sealed if s.voiceprint_id)

    # name it later (tag-once via the knowledge channel), then re-identify in a NEW session
    assert store.set_name("ws1", anon_id, "Bob", actor="conclave")
    ident2 = _run(store, emb, audio, script)
    sealed2 = ident2.seal()
    named = [s for s in sealed2 if s.voiceprint_id == anon_id]
    assert named and all(s.name == "Bob" for s in named)  # recognised by name across sessions


def test_seal_freezes_corrections(ctx):
    store, emb = ctx
    enroll(store, emb, "ws1", "Alice", _wav("spkA_1"), SR)
    audio = np.concatenate([_wav("spkA_1"), _wav("spkA_2")])
    script = [Segment(0, 4, "speaker0"), Segment(4, 8, "speaker0")]
    ident = _run(store, emb, audio, script)
    sealed = ident.seal()
    n = len(sealed)
    assert ident.finish() == []          # draining is idempotent after seal
    assert len(ident.transcript()) == n  # transcript unchanged post-seal
