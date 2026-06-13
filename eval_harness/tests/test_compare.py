"""C10 — compare's diff/formatting on synthetic result dicts. No models, no subprocess, no venv —
this is the pure logic that turns two result.json blobs into the side-by-side + deltas."""
from eval_harness.compare import build_compare, format_table

DIART = {
    "name": "demo", "wer": 0.20, "wer_vocab_off": 0.25, "speaker_accuracy": 0.80,
    "rtf": 0.30, "total_sec": 12.0, "peak_rss_mb": 850.0, "speakers_detected": 2,
}
DIARIZEN = {
    "name": "demo", "wer": 0.20, "wer_vocab_off": 0.25, "speaker_accuracy": 0.92,
    "rtf": 1.30, "total_sec": 52.0, "peak_rss_mb": 4900.0, "speakers_detected": 2,
}


def _row(cmp, key):
    return next(r for r in cmp["rows"] if r["key"] == key)


def test_deltas_and_direction():
    cmp = build_compare(DIART, DIARIZEN)
    # DiariZen more accurate on speakers (higher is better) → flagged better, positive delta
    spk = _row(cmp, "speaker_accuracy")
    assert spk["delta"] == 0.12 and spk["diarizen_better"] is True
    # …but slower (RTF higher is worse) → flagged worse
    rtf = _row(cmp, "rtf")
    assert rtf["delta"] == 1.0 and rtf["diarizen_better"] is False
    # equal WER → delta 0, no winner
    assert _row(cmp, "wer")["delta"] == 0 and _row(cmp, "wer")["diarizen_better"] is None


def test_handles_missing_field():
    cmp = build_compare(DIART, {**DIARIZEN, "peak_rss_mb": None})
    ram = _row(cmp, "peak_rss_mb")
    assert ram["delta"] is None and ram["diarizen_better"] is None


def test_table_renders():
    table = format_table(build_compare(DIART, DIARIZEN))
    assert "diart vs DiariZen" in table
    assert "speaker accuracy" in table
    assert "✓" in table and "✗" in table        # at least one better + one worse
