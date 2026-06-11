"""C — end-to-end offline path: REAL diart engine -> identify pipeline.

Cold-start integration (no enrollment): two unknown speakers in a single-mic mix
should diarize into two session-stable speakers, each minted as an anonymous
voiceprint, with a sealed corrected transcript and bounded memory. This exercises
the diart<->identify wiring on real audio; identity *recognition* accuracy is
covered by the C.4/C.5 unit tests (and gated on the embedder windowing fix, see
docs/embedder-bench.md). Skipped unless diart is importable (run in the diart venv).
"""
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("diart", reason="diart not installed (torch-free .venv); run in diart venv")

from fpm.diarize.diart_engine import DiartDiarizer  # noqa: E402
from fpm.embed.onnx_embedder import OnnxSpeakerEmbedder  # noqa: E402
from fpm.identify import SessionIdentifier  # noqa: E402
from fpm.store.store import VoiceprintStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "campplus.onnx"
FIX = Path(__file__).parent / "fixtures" / "speakers"
SR = 16_000


def _wav(name):
    a, sr = sf.read(FIX / f"{name}.wav", dtype="float32")
    assert sr == SR
    return a if a.ndim == 1 else a.mean(axis=1)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    if not MODEL.exists():
        pytest.skip("embedder model missing — run scripts/fetch_models.sh")
    db = tmp_path_factory.mktemp("e2e") / "vp.db"
    store = VoiceprintStore(db_path=db, key=os.urandom(32)).open()
    emb = OnnxSpeakerEmbedder(MODEL).load()
    clip = np.concatenate([_wav(n) for n in
                           ["spkA_1", "spkB_1", "spkA_2", "spkB_1", "spkA_1", "spkB_1"]])
    ident = SessionIdentifier(store, emb, DiartDiarizer(offline=True), "ws1")
    ident.start()
    step = int(0.5 * SR)
    for i in range(0, len(clip), step):
        ident.feed(clip[i:i + step], SR)
    transcript = ident.seal()
    yield store, ident, transcript, len(clip) / SR
    store.close()


def test_runs_offline(result):
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


def test_two_stable_speakers(result):
    _, _, transcript, _ = result
    assert transcript, "empty transcript"
    assert len({s.local_speaker for s in transcript}) == 2
    # each local speaker resolves to exactly one voiceprint — no flicker
    for spk in {s.local_speaker for s in transcript}:
        ids = {s.voiceprint_id for s in transcript if s.local_speaker == spk and s.voiceprint_id}
        assert len(ids) == 1, f"{spk} flickered: {ids}"


def test_unknown_speakers_minted_anonymous(result):
    store, _, transcript, _ = result
    resolved = {s.voiceprint_id for s in transcript if s.voiceprint_id}
    assert len(resolved) == 2                                  # two anonymous voiceprints
    assert set(store.list_ids("ws1")) == resolved
    assert all(store.get("ws1", vp).name == "" for vp in resolved)  # anonymous (no name)


def test_transcript_bounds_and_bounded_memory(result):
    _, ident, transcript, total = result
    for s in transcript:
        assert 0.0 <= s.start < s.end <= total + 1.0
    assert ident._buf.size <= ident._max_buf
