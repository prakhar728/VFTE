"""A.2 — ONNX ID embedder (fixed model defining the voiceprint space).

Needs the embedder model (run `scripts/fetch_models.sh`); skips if absent so
CI without the baked model doesn't fail. Speaker fixtures are short AMI clips
(CC-BY): spkA_1/spkA_2 = same speaker, spkB_1 = different.
"""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from fpm.embed.onnx_embedder import OnnxSpeakerEmbedder

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "campplus.onnx"
FIX = Path(__file__).parent / "fixtures" / "speakers"

pytestmark = pytest.mark.skipif(
    not MODEL.exists(), reason="embedder model missing — run scripts/fetch_models.sh"
)


def _embed(embedder, name):
    audio, sr = sf.read(FIX / f"{name}.wav", dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return embedder.extract(audio, sr)


def _cos(a, b):  # embeddings are L2-normalized → dot == cosine
    return float(a @ b)


def test_loads_and_dim():
    e = OnnxSpeakerEmbedder(MODEL).load()
    assert e.is_loaded
    assert e.embedding_dim == 512


def test_discriminates_speakers():
    e = OnnxSpeakerEmbedder(MODEL).load()
    a1, a2, b1 = _embed(e, "spkA_1"), _embed(e, "spkA_2"), _embed(e, "spkB_1")
    assert a1 is not None and a1.shape == (512,)
    same, diff = _cos(a1, a2), _cos(a1, b1)
    assert same > diff + 0.2, f"weak separation: same={same:.3f} diff={diff:.3f}"
    assert same > 0.4


def test_deterministic():
    e = OnnxSpeakerEmbedder(MODEL).load()
    assert np.allclose(_embed(e, "spkA_1"), _embed(e, "spkA_1"))


def test_short_audio_returns_none():
    e = OnnxSpeakerEmbedder(MODEL).load()
    assert e.extract(np.zeros(8_000, dtype=np.float32)) is None
