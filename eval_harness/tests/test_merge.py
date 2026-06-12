"""C4 — merge transcript ↔ speaker spans (pure logic, no models)."""
from dataclasses import dataclass

from eval_harness.harness.merge import attribute_words, group_turns, merge


@dataclass
class W:          # stand-in for asr.Word
    word: str
    start: float
    end: float


@dataclass
class S:          # stand-in for diarize Segment
    start: float
    end: float
    local_speaker: str


def test_words_attributed_by_overlap():
    words = [W(" hey", 0.0, 0.5), W(" there", 0.6, 1.0), W(" hi", 2.1, 2.5)]
    spans = [S(0.0, 1.5, "speaker0"), S(2.0, 3.0, "speaker1")]
    aw = attribute_words(words, spans)
    assert [a.speaker for a in aw] == ["speaker0", "speaker0", "speaker1"]


def test_no_overlap_falls_back_to_nearest():
    spans = [S(0.0, 1.5, "speaker0"), S(2.0, 3.0, "speaker1")]
    assert attribute_words([W(" g", 1.55, 1.65)], spans)[0].speaker == "speaker0"  # nearer span0
    assert attribute_words([W(" g", 1.85, 1.95)], spans)[0].speaker == "speaker1"  # nearer span1


def test_turns_group_consecutive_speakers():
    from eval_harness.harness.merge import AttributedWord as A
    aw = [A(" hey", 0.0, 0.5, "speaker0"), A(" there", 0.6, 1.0, "speaker0"),
          A(" hi", 2.1, 2.5, "speaker1"), A(" yo", 2.6, 3.0, "speaker1")]
    turns = group_turns(aw)
    assert [(t.speaker, t.text) for t in turns] == [("speaker0", "hey there"), ("speaker1", "hi yo")]


def test_merge_end_to_end():
    words = [W(" hello", 0.0, 0.5), W(" world", 0.6, 1.0), W(" bye", 2.1, 2.6)]
    spans = [S(0.0, 1.5, "A"), S(2.0, 3.0, "B")]
    turns = merge(words, spans)
    assert [(t.speaker, t.text) for t in turns] == [("A", "hello world"), ("B", "bye")]
    assert turns[0].start == 0.0 and turns[1].end == 2.6
