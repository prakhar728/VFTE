"""C.1 — StreamingDiarizer interface + MockDiarizer contract."""
import numpy as np
import pytest

from fpm.diarize.base import Segment, StreamingDiarizer
from fpm.diarize.mock import MockDiarizer

SR = 16_000


def _chunk(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def _script():
    return [
        Segment(0.0, 1.0, "speaker0"),
        Segment(1.0, 2.0, "speaker1"),
        Segment(2.0, 3.0, "speaker0"),
    ]


def test_segment_carries_no_identity():
    s = Segment(0.0, 1.5, "speaker0")
    assert {f for f in s.__dataclass_fields__} == {"start", "end", "local_speaker"}
    assert s.duration == pytest.approx(1.5)
    assert "voiceprint" not in repr(s).lower()  # diarizer never emits identity


def test_mock_is_a_streaming_diarizer():
    assert isinstance(MockDiarizer([]), StreamingDiarizer)


def test_emits_incrementally_as_time_accumulates():
    d = MockDiarizer(_script())
    d.start("ws1")
    # feed in 0.5 s chunks; segments finalize only as their end-time is crossed
    out0 = d.feed(_chunk(0.5), SR)   # t=0.5 → nothing finalized yet
    assert out0 == []
    out1 = d.feed(_chunk(0.5), SR)   # t=1.0 → speaker0 [0,1] finalized
    assert [s.local_speaker for s in out1] == ["speaker0"]
    out2 = d.feed(_chunk(1.0), SR)   # t=2.0 → speaker1 [1,2]
    assert [s.local_speaker for s in out2] == ["speaker1"]


def test_finish_flushes_remaining():
    d = MockDiarizer(_script())
    d.start("ws1")
    d.feed(_chunk(1.5), SR)          # finalizes only [0,1]
    rest = d.finish()
    assert [(s.start, s.local_speaker) for s in rest] == [(1.0, "speaker1"), (2.0, "speaker0")]
    assert d.finish() == []          # idempotent after drain


def test_start_resets_state():
    d = MockDiarizer(_script())
    d.start("ws1")
    d.feed(_chunk(3.0), SR)
    d.start("ws2")                   # new session
    assert d.feed(_chunk(1.0), SR)[0].local_speaker == "speaker0"


def test_feed_before_start_errors():
    with pytest.raises(RuntimeError):
        MockDiarizer(_script()).feed(_chunk(1.0), SR)
