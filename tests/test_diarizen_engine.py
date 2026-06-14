"""Branch A / step 1 — DiariZenDiarizer conforms to the C1 StreamingDiarizer seam.

These run in the torch-free CORE venv: the engine module imports only numpy + base, and
the heavy torch/diarizen stack is loaded lazily inside finish() (decision D1), so contract
conformance, the batch feed() shape, the guards, and the lazy seam are all testable without
the diarizen venv. The real diarize (model load + decode) is exercised by the gated e2e test
(tests/test_diarize_diarizen_api.py).
"""
import sys

import numpy as np
import pytest

from fpm.diarize.base import Segment, StreamingDiarizer
from fpm.diarize.diarizen_engine import ClipTooLongError, DiariZenDiarizer

SR = 16_000


def _chunk(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def test_is_a_streaming_diarizer():
    assert isinstance(DiariZenDiarizer(), StreamingDiarizer)


def test_exposes_contract_methods():
    d = DiariZenDiarizer()
    assert callable(d.start) and callable(d.feed) and callable(d.finish)


def test_feed_is_batch_no_incremental_output():
    d = DiariZenDiarizer()
    d.start("ws1")
    # batch engine: every feed() returns [] — segments only come at finish() (C1 allows empty)
    assert d.feed(_chunk(0.5), SR) == []
    assert d.feed(_chunk(2.0), SR) == []


def test_feed_before_start_raises():
    with pytest.raises(RuntimeError):
        DiariZenDiarizer().feed(_chunk(1.0), SR)


def test_feed_wrong_sample_rate_raises():
    d = DiariZenDiarizer()
    d.start("ws1")
    with pytest.raises(ValueError):
        d.feed(_chunk(1.0), 8_000)


def test_start_resets_buffer():
    d = DiariZenDiarizer()
    d.start("ws1")
    d.feed(_chunk(1.0), SR)
    d.start("ws2")          # new session drops the prior buffer
    assert d._buf == []


def test_finish_without_audio_is_empty():
    d = DiariZenDiarizer()
    d.start("ws1")
    assert d.finish() == []          # no audio fed → nothing to diarize, no model load
    assert "diarizen" not in sys.modules


def test_buffered_batch_capability_flag():
    # Hint the identify layer reads via getattr(diarizer, "buffered_batch", False): this engine
    # emits all segments at finish(), so the trailing-audio buffer must retain the whole clip.
    assert DiariZenDiarizer().buffered_batch is True


def test_lazy_seam_no_heavy_import_until_finish():
    # Constructing + feeding must not import torch/diarizen (D1: load only in finish()).
    d = DiariZenDiarizer()
    d.start("ws1")
    d.feed(_chunk(1.0), SR)
    assert "diarizen" not in sys.modules
    assert "torch" not in sys.modules


def test_segments_carry_no_identity_type():
    # Sanity on the contract type the engine emits (no embeddings/ids/text).
    s = Segment(0.0, 1.0, "speaker0")
    assert {f for f in s.__dataclass_fields__} == {"start", "end", "local_speaker"}


# ── step 4: clip-length cap (RAM guard) ─────────────────────────────


def test_cap_rejects_oversized_clip_before_model_load():
    # DiariZen loads the whole clip in RAM; oversize audio must be rejected in feed() before
    # any model load — proves the cap is a pre-load guard (D1+D4), testable without diarizen.
    d = DiariZenDiarizer(max_clip_sec=0.5)
    d.start("ws1")
    with pytest.raises(ClipTooLongError):
        d.feed(_chunk(1.0), SR)          # 1.0 s > 0.5 s cap
    assert "diarizen" not in sys.modules


def test_cap_accumulates_across_feeds():
    d = DiariZenDiarizer(max_clip_sec=1.0)
    d.start("ws1")
    assert d.feed(_chunk(0.5), SR) == []   # 0.5 s — under cap
    with pytest.raises(ClipTooLongError):
        d.feed(_chunk(0.6), SR)            # cumulative 1.1 s > 1.0 s cap


def test_under_cap_does_not_raise():
    d = DiariZenDiarizer(max_clip_sec=10.0)
    d.start("ws1")
    assert d.feed(_chunk(2.0), SR) == []


def test_cap_zero_disables_guard():
    # 0 / None disables the cap (unbounded) — feeding a long clip must not raise.
    d = DiariZenDiarizer(max_clip_sec=0)
    d.start("ws1")
    assert d.feed(_chunk(30.0), SR) == []


def test_cap_defaults_from_config(monkeypatch):
    import config
    monkeypatch.setattr(config, "DIARIZEN_MAX_CLIP_SEC", 1.0)
    d = DiariZenDiarizer()                  # no explicit cap → uses config default
    d.start("ws1")
    with pytest.raises(ClipTooLongError):
        d.feed(_chunk(1.5), SR)
