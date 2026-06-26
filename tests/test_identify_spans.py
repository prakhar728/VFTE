"""Identify pre-diarized spans (migration P5) — the identity-only inverse of /v1/diarize.

Same real CAM++ + store + fixtures as test_identify, but instead of running a diarizer we hand
identify_spans the spans directly (as capture would) and assert identity lands correctly: the enrolled
speaker (Alice = spkA) resolves to her voiceprint, the unknown speaker (spkB) mints an anonymous one.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from fpm.diarize.base import Segment
from fpm.embed.onnx_embedder import OnnxSpeakerEmbedder
from fpm.enroll import enroll
from fpm.identify_spans import SpanReplayDiarizer, identified_dict, identify_spans
from fpm.store.store import VoiceprintStore

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "campplus.onnx"
FIX = Path(__file__).parent / "fixtures" / "speakers"
SR = 16_000
KEY = os.urandom(32)

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="embedder model missing")

_TURNS = ["spkA_1", "spkB_1", "spkA_2", "spkB_1", "spkA_1", "spkB_1"]
_LABELS = ["speaker0", "speaker1", "speaker0", "speaker1", "speaker0", "speaker1"]


def _wav(name):
    a, sr = sf.read(FIX / f"{name}.wav", dtype="float32")
    assert sr == SR
    return a if a.ndim == 1 else a.mean(axis=1)


def _clip_and_spans():
    parts, spans, t = [], [], 0.0
    for name, label in zip(_TURNS, _LABELS):
        a = _wav(name)
        dur = len(a) / SR
        parts.append(a)
        spans.append({"start": t, "end": t + dur, "local_speaker": label})
        t += dur
    return np.concatenate(parts), spans


# ── SpanReplayDiarizer (the bridge) ──

def test_span_replay_emits_all_spans_at_finish():
    d = SpanReplayDiarizer([Segment(2.0, 3.0, "s1"), Segment(0.0, 1.0, "s0")])
    d.start("ws1")
    assert d.feed(np.zeros(8000, np.float32), SR) == []   # batch: nothing during feed
    out = d.finish()
    assert [(s.start, s.local_speaker) for s in out] == [(0.0, "s0"), (2.0, "s1")]  # sorted
    assert d.finish() == []                                # idempotent
    assert SpanReplayDiarizer([]).buffered_batch is True


def test_feed_before_start_raises():
    with pytest.raises(RuntimeError):
        SpanReplayDiarizer([]).feed(np.zeros(10, np.float32), SR)


# ── identify_spans (real CAM++ + store) ──

@pytest.fixture
def store_and_embedder(tmp_path):
    store = VoiceprintStore(db_path=tmp_path / "vp.db", key=KEY).open()
    emb = OnnxSpeakerEmbedder(MODEL).load()
    enroll(store, emb, "ws1", "Alice", _wav("spkA_1"), SR)   # Alice = spkA
    return store, emb


def test_identify_spans_resolves_enrolled_and_mints_unknown(store_and_embedder):
    store, emb = store_and_embedder
    audio, spans = _clip_and_spans()
    segs = identify_spans(audio, "ws1", spans, store=store, embedder=emb, sample_rate=SR)
    assert segs, "expected identified segments"
    by_label = {}
    for s in segs:
        by_label.setdefault(s.local_speaker, set()).add(s.name)
    # speaker0 (Alice) resolves to her name somewhere in the (relabeled) transcript
    assert any(n == "Alice" for n in by_label.get("speaker0", set()))
    # every emitted segment carries the full identity shape
    for s in segs:
        d = identified_dict(s)
        assert set(d) == {"start", "end", "local_speaker", "voiceprint_id", "name",
                          "decision", "confidence"}


def test_identify_spans_read_only_writes_nothing(store_and_embedder):
    store, emb = store_and_embedder
    audio, spans = _clip_and_spans()
    before = len(store.centroids("ws1"))
    identify_spans(audio, "ws1", spans, store=store, embedder=emb, sample_rate=SR, read_only=True)
    after = len(store.centroids("ws1"))
    assert after == before, "read-only identify must not mint/write voiceprints"


def test_identify_spans_accepts_segment_objects(store_and_embedder):
    store, emb = store_and_embedder
    audio, spans = _clip_and_spans()
    seg_objs = [Segment(s["start"], s["end"], s["local_speaker"]) for s in spans]
    segs = identify_spans(audio, "ws1", seg_objs, store=store, embedder=emb, sample_rate=SR)
    assert segs
