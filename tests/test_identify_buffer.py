"""Branch C §6 — trailing-buffer sizing from `diarizer.buffered_batch` (A↔C bridge).

A batch engine (DiariZen) returns [] from feed() and emits ALL segments at finish().
With the default bounded 15 s buffer, any speaker that appears only EARLY in a long
clip can't be re-embedded at finish() → stays PENDING. SessionIdentifier honours an
optional `buffered_batch` hint (read via getattr — A sets it on its own engine) to
retain the full clip for batch engines.

Uses a local batch double (NOT an edit to the shared fpm/diarize/mock.py).
"""
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from fpm.types import Segment, StreamingDiarizer
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


class _BatchDiarizer(StreamingDiarizer):
    """Emulates DiariZen: buffers everything, emits all segments only at finish()."""

    buffered_batch = True

    def __init__(self, script):
        self._script = sorted(script, key=lambda s: s.end)

    def start(self, workspace_id: str) -> None:
        self._started = True

    def feed(self, chunk, sample_rate: int = 16_000):
        return []

    def finish(self):
        return list(self._script)


class _BatchDiarizerBounded(_BatchDiarizer):
    """Same batch behaviour but WITHOUT the hint (→ bounded buffer, current default)."""

    buffered_batch = False


def _long_clip():
    """24 s: Alice (speaker0) only in the first 8 s; an unknown (speaker1) fills the
    last 16 s — i.e. > the 15 s buffer — so Alice is unreachable to a bounded buffer."""
    parts = [_wav("spkA_1"), _wav("spkA_2"),                 # 0-8 s  Alice
             _wav("spkB_1"), _wav("spkB_1"), _wav("spkB_1"), _wav("spkB_1")]  # 8-24 s unknown
    audio = np.concatenate(parts)
    script = [Segment(0, 4, "speaker0"), Segment(4, 8, "speaker0"),
              Segment(8, 12, "speaker1"), Segment(12, 16, "speaker1"),
              Segment(16, 20, "speaker1"), Segment(20, 24, "speaker1")]
    return audio, script


@pytest.fixture
def ctx(tmp_path):
    store = VoiceprintStore(db_path=tmp_path / "vp.db", key=KEY).open()
    emb = OnnxSpeakerEmbedder(MODEL).load()
    enroll(store, emb, "ws1", "Alice", _wav("spkA_1"), SR)
    yield store, emb
    store.close()


def _seal(store, emb, diarizer, audio):
    ident = SessionIdentifier(store, emb, diarizer, "ws1")
    ident.start()
    step = int(0.5 * SR)
    for i in range(0, len(audio), step):
        ident.feed(audio[i:i + step], SR)
    return ident.seal()


def test_batch_hint_identifies_early_speaker(ctx):
    store, emb = ctx
    alice_id = identity_voiceprint_id("ws1", "Alice")
    audio, script = _long_clip()
    sealed = _seal(store, emb, _BatchDiarizer(script), audio)   # buffered_batch=True
    sp0 = [s for s in sealed if s.local_speaker == "speaker0"]
    assert sp0 and all(s.voiceprint_id == alice_id for s in sp0)  # early speaker recovered


def test_bounded_buffer_loses_early_speaker(ctx):
    store, emb = ctx
    audio, script = _long_clip()
    sealed = _seal(store, emb, _BatchDiarizerBounded(script), audio)  # bounded (control)
    sp0 = [s for s in sealed if s.local_speaker == "speaker0"]
    # the early-only speaker was never re-embeddable → never identified
    assert sp0 and all(s.voiceprint_id is None for s in sp0)
