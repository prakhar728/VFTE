"""C.3 — diart adapter: incremental {start,end,local_speaker}, offline at runtime.

Skipped unless diart is importable (the main .venv is intentionally torch-free;
run this in the diart eval venv). Builds a 2-speaker clip from the in-repo AMI
CC-BY fixtures so it's self-contained — no eval_data/ download needed.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("diart", reason="diart not installed (torch-free .venv); run in diart venv")

from fpm.diarize.base import Segment, StreamingDiarizer  # noqa: E402
from fpm.diarize.diart_engine import DiartDiarizer  # noqa: E402

SR = 16_000
FIX = Path(__file__).parent / "fixtures" / "speakers"


def _clip() -> np.ndarray:
    """~24 s of alternating speakers A/B (4 s turns) from the fixtures."""
    parts = []
    for name in ["spkA_1", "spkB_1", "spkA_2", "spkB_1", "spkA_1", "spkB_1"]:
        a, sr = sf.read(FIX / f"{name}.wav", dtype="float32")
        assert sr == SR
        parts.append(a if a.ndim == 1 else a.mean(axis=1))
    return np.concatenate(parts)


@pytest.fixture(scope="module")
def segments():
    # offline=True → HF_HUB_OFFLINE=1; a successful load then proves cache-only (no egress).
    d = DiartDiarizer(offline=True)
    assert isinstance(d, StreamingDiarizer)
    d.start("ws1")
    audio = _clip()
    out: list[Segment] = []
    step = int(0.5 * SR)
    for i in range(0, len(audio), step):
        out.extend(d.feed(audio[i:i + step], SR))
    out.extend(d.finish())
    return out, audio


def test_runtime_is_offline(segments):
    # adapter must have engaged HF offline mode (no network egress at runtime)
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


def test_emits_incremental_segments(segments):
    out, _ = segments
    assert out, "expected at least one segment"
    for s in out:
        assert isinstance(s, Segment)
        assert {f for f in s.__dataclass_fields__} == {"start", "end", "local_speaker"}
        assert s.end > s.start
        assert s.local_speaker.startswith("speaker")  # engine-local label, not a voiceprint id


def test_detects_two_speakers(segments):
    out, _ = segments
    labels = {s.local_speaker for s in out}
    assert len(labels) == 2, f"expected 2 speakers, got {labels}"


def test_spans_are_usable_length(segments):
    # stitching must produce spans long enough for the CAM++ embedder (≥1 s), not 0.5 s shards
    out, _ = segments
    long_enough = [s for s in out if s.duration >= 1.0]
    assert len(long_enough) >= 2
    assert max(s.duration for s in out) >= 1.0


def test_segments_within_audio_bounds(segments):
    out, audio = segments
    total = len(audio) / SR
    for s in out:
        assert 0.0 <= s.start < s.end <= total + 0.5  # small tail tolerance
