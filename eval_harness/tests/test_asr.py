"""C2 — WhisperASR wrapper (exact Recato params). Uses 'tiny' for a fast check; real runs use
large-v3-turbo per config. Skipped if faster-whisper isn't installed."""
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("faster_whisper", reason="faster-whisper not installed (eval venv only)")
import soundfile as sf  # noqa: E402

from eval_harness.harness.asr import AsrResult, WhisperASR, Word  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "speakers" / "spkA_1.wav"
MODEL = os.environ.get("FW_TEST_MODEL", "tiny")


@pytest.fixture(scope="module")
def asr():
    return WhisperASR(model=MODEL, device="cpu", compute_type="int8")


def _audio():
    a, sr = sf.read(FIX, dtype="float32")
    return (a if a.ndim == 1 else a.mean(axis=1))


def test_transcribe_shape(asr):
    r = asr.transcribe(_audio(), vocab=None)
    assert isinstance(r, AsrResult)
    assert isinstance(r.text, str) and isinstance(r.language, str)
    assert isinstance(r.words, list) and all(isinstance(w, Word) for w in r.words)
    # word timestamps are populated + ordered when there's speech
    for w in r.words:
        assert w.end >= w.start


def test_vocab_on_and_off_both_run(asr):
    audio = _audio()
    off = asr.transcribe(audio, vocab=None)
    on = asr.transcribe(audio, vocab="Recato, Priya, Arjun, Sunnyvale, cortado")
    assert isinstance(off, AsrResult) and isinstance(on, AsrResult)  # both paths work


def test_deterministic(asr):
    audio = _audio()
    assert asr.transcribe(audio).text == asr.transcribe(audio).text  # temp 0 + beam → deterministic
