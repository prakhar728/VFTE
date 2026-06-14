"""Branch C — P1 read-only live mode + P3 quality gate.

Same substrate as test_identify.py: MockDiarizer (scripted) + real CAM++ embedder
+ real (tmp) VoiceprintStore. "Writes nothing" is asserted black-box via list_ids
and usage_for_voiceprint deltas; the gate is exercised by monkeypatching
config.MIN_SEGMENT_SEC so a real (embeddable) segment is gated by *duration*,
proving the gate — not the embedder's own min-length — blocks it.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import config
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
def ctx(tmp_path):
    store = VoiceprintStore(db_path=tmp_path / "vp.db", key=KEY).open()
    emb = OnnxSpeakerEmbedder(MODEL).load()
    yield store, emb
    store.close()


def _run(store, emb, audio, script, ws="ws1", **kw):
    ident = SessionIdentifier(store, emb, MockDiarizer(script), ws, **kw)
    ident.start()
    step = int(0.5 * SR)
    out = []
    for i in range(0, len(audio), step):
        out.extend(ident.feed(audio[i:i + step], SR))
    out.extend(ident.finish())
    return ident, out


def _by_speaker(out, spk):
    return [s for s in out if s.local_speaker == spk]


# ── P1: read-only mode ────────────────────────────────────────────────

def test_read_only_writes_nothing(ctx):
    store, emb = ctx
    enroll(store, emb, "ws1", "Alice", _wav("spkA_1"), SR)
    alice_id = identity_voiceprint_id("ws1", "Alice")
    ids_before = set(store.list_ids("ws1"))
    ledger_before = len(store.usage_for_voiceprint("ws1", alice_id))

    audio, script = _clip_and_script()
    _, out = _run(store, emb, audio, script, read_only=True)

    # no anonymous voiceprint minted — store unchanged
    assert set(store.list_ids("ws1")) == ids_before == {alice_id}
    # MATCH-lock did NOT append a usage-ledger row (log_usage suppressed)
    assert len(store.usage_for_voiceprint("ws1", alice_id)) == ledger_before
    # the unknown speaker never receives a stored id
    assert all(s.voiceprint_id is None for s in _by_speaker(out, "speaker1"))


def test_read_only_still_identifies(ctx):
    store, emb = ctx
    enroll(store, emb, "ws1", "Alice", _wav("spkA_1"), SR)
    alice_id = identity_voiceprint_id("ws1", "Alice")
    audio, script = _clip_and_script()
    _, out = _run(store, emb, audio, script, read_only=True)
    resolved = [s for s in _by_speaker(out, "speaker0") if s.voiceprint_id is not None]
    assert resolved, "enrolled speaker not identified for display in read-only"
    assert all(s.voiceprint_id == alice_id and s.name == "Alice" for s in resolved)


def test_read_only_labels_stable(ctx):
    store, emb = ctx
    enroll(store, emb, "ws1", "Alice", _wav("spkA_1"), SR)
    audio, script = _clip_and_script()
    _, out = _run(store, emb, audio, script, read_only=True)
    # speaker0 resolves to exactly one id (Alice) — no flicker
    ids0 = {s.voiceprint_id for s in _by_speaker(out, "speaker0") if s.voiceprint_id}
    assert len(ids0) == 1
    # the unknown speaker still locks to a stable (None) label → emits LOCKED
    assert any(s.decision == "LOCKED" for s in _by_speaker(out, "speaker1"))


def test_offline_default_still_writes(ctx):
    """Control: with read_only=False (default), the unknown IS minted (regression)."""
    store, emb = ctx
    enroll(store, emb, "ws1", "Alice", _wav("spkA_1"), SR)
    alice_id = identity_voiceprint_id("ws1", "Alice")
    audio, script = _clip_and_script()
    _run(store, emb, audio, script)  # default writer
    ids = set(store.list_ids("ws1"))
    assert alice_id in ids and len(ids) == 2  # Alice + one minted anon
