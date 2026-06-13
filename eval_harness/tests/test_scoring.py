"""C5 — WER + speaker-attribution scoring + metrics (jiwer; no models)."""
import json

from eval_harness.harness.merge import Turn
from eval_harness.harness.metrics import offline_metrics
from eval_harness.harness.scoring import (load_gold, normalize, parse_gold,
                                          parse_gold_json, speaker_accuracy, wer)

GOLD = [("A", "hello world"), ("B", "bye now")]


def test_normalize():
    assert normalize("Hello, World!") == "hello world"


def test_wer():
    assert wer("hello world", "hello world") == 0.0
    assert wer("hello world", "hello there") == 0.5            # 1 of 2 words wrong


def test_parse_gold():
    assert parse_gold("A: hi there\nB: yo\n\n# noise") == [("A", "hi there"), ("B", "yo")]


def test_parse_gold_json():
    raw = json.dumps({"turns": [{"speaker": "A", "text": "hi there"}, {"speaker": "B", "text": "yo"}]})
    assert parse_gold_json(raw) == [("A", "hi there"), ("B", "yo")]


def test_load_gold_dispatches_on_suffix(tmp_path):
    j = tmp_path / "gold.json"
    j.write_text(json.dumps({"turns": [{"speaker": "A", "text": "hi"}]}))
    assert load_gold(j) == [("A", "hi")]
    t = tmp_path / "gold.txt"
    t.write_text("A: hi\n")
    assert load_gold(t) == [("A", "hi")]


def _turns(*pairs):
    return [Turn(spk, txt, 0.0, 1.0) for spk, txt in pairs]


def test_speaker_accuracy_perfect_and_swapped_both_score_1():
    perfect = _turns(("speaker0", "hello world"), ("speaker1", "bye now"))
    assert speaker_accuracy(GOLD, perfect)["accuracy"] == 1.0
    # consistent label swap → optimal mapping recovers it
    swapped = _turns(("speaker1", "hello world"), ("speaker0", "bye now"))
    assert speaker_accuracy(GOLD, swapped)["accuracy"] == 1.0


def test_speaker_accuracy_single_speaker_half():
    everything_one = _turns(("speaker0", "hello world bye now"))
    r = speaker_accuracy(GOLD, everything_one)
    assert r["accuracy"] == 0.5 and r["aligned"] == 4


def test_offline_metrics_rtf():
    m = offline_metrics(audio_length_sec=10.0, asr_sec=2.0, diarize_sec=3.0)
    assert m.total_sec == 5.0 and m.rtf == 0.5 and m.mode == "offline"
    assert m.as_dict()["rtf"] == 0.5
