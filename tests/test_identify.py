"""C.4 — real-time identify pipeline (MockDiarizer-driven, real CAM++ + store)."""
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from fpm.types import Segment
from tests.mock_diarizer import MockDiarizer
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


# 6 alternating 4 s turns: speaker0 = Alice (spkA), speaker1 = unknown (spkB)
_TURNS = ["spkA_1", "spkB_1", "spkA_2", "spkB_1", "spkA_1", "spkB_1"]
_LABELS = ["speaker0", "speaker1", "speaker0", "speaker1", "speaker0", "speaker1"]


def _clip_and_script():
    parts, script, t = [], [], 0.0
    for name, label in zip(_TURNS, _LABELS):
        a = _wav(name)
        dur = len(a) / SR
        parts.append(a)
        script.append(Segment(t, t + dur, label))
        t += dur
    return np.concatenate(parts), script


@pytest.fixture
def session(tmp_path):
    store = VoiceprintStore(db_path=tmp_path / "vp.db", key=KEY).open()
    emb = OnnxSpeakerEmbedder(MODEL).load()
    enroll(store, emb, "ws1", "Alice", _wav("spkA_1"), SR)  # Alice = spkA
    audio, script = _clip_and_script()
    ident = SessionIdentifier(store, emb, MockDiarizer(script), "ws1")
    ident.start()
    out = []
    step = int(0.5 * SR)
    for i in range(0, len(audio), step):
        out.extend(ident.feed(audio[i:i + step], SR))
    out.extend(ident.finish())
    yield store, ident, out
    store.close()


def _by_speaker(out, spk):
    return [s for s in out if s.local_speaker == spk]


def test_enrolled_voice_labeled_live(session):
    store, _, out = session
    alice_id = identity_voiceprint_id("ws1", "Alice")
    sp0 = _by_speaker(out, "speaker0")
    assert sp0, "no segments for the enrolled speaker"
    # every resolved speaker0 segment names Alice (never anonymous, never silent)
    resolved = [s for s in sp0 if s.voiceprint_id is not None]
    assert resolved and all(s.voiceprint_id == alice_id and s.name == "Alice" for s in resolved)


def test_unknown_minted_anonymous(session):
    store, _, out = session
    alice_id = identity_voiceprint_id("ws1", "Alice")
    sp1 = _by_speaker(out, "speaker1")
    resolved = [s for s in sp1 if s.voiceprint_id is not None]
    assert resolved, "unknown speaker never resolved"
    anon_id = resolved[-1].voiceprint_id
    assert anon_id != alice_id
    assert all(s.name is None for s in resolved)            # anonymous → no name
    vp = store.get("ws1", anon_id)                          # minted into the store
    assert vp is not None and vp.name == "" and vp.centroid.size
    assert set(store.list_ids("ws1")) == {alice_id, anon_id}


def test_labels_stabilize(session):
    _, _, out = session
    # each local speaker resolves to exactly one voiceprint id — no flicker
    for spk in ("speaker0", "speaker1"):
        ids = {s.voiceprint_id for s in _by_speaker(out, spk) if s.voiceprint_id}
        assert len(ids) == 1, f"{spk} flickered across ids: {ids}"


def test_bounded_memory(session):
    _, ident, _ = session
    # 24 s fed into a 15 s buffer → trimmed, not grown; maps sized by speaker count
    assert ident._buf.size <= ident._max_buf
    assert len(ident._locked) <= 2 and len(ident._votes) <= 2


def test_locks_after_repeated_agreement(session):
    _, _, out = session
    # once locked, the engine stops re-deciding (emits LOCKED) for both speakers
    assert any(s.decision == "LOCKED" for s in _by_speaker(out, "speaker0"))
    assert any(s.decision == "LOCKED" for s in _by_speaker(out, "speaker1"))
