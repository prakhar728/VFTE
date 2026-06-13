"""fp_split — source discovery + segment-plan parsing (no ffmpeg; the cut itself is integration)."""
import pytest

from eval_harness.fp_split import find_source, load_segments

META = """\
name: t
segmentation:
  segments:
    - { name: "00_enroll_000-060s", start_sec: 0,   end_sec: 60,   role: enroll }
    - { name: "01_test_060-240s",   start_sec: 60,  end_sec: 240,  role: identify }
    - { name: "02_test_240-end",    start_sec: 240, end_sec: null, role: identify }
"""


def _run(tmp_path):
    (tmp_path / "metadata.yaml").write_text(META)
    (tmp_path / "source").mkdir()
    return tmp_path


def test_load_segments(tmp_path):
    segs = load_segments(_run(tmp_path))
    assert [s["name"] for s in segs][0] == "00_enroll_000-060s"
    assert segs[-1]["end_sec"] is None and segs[-1]["role"] == "identify"


def test_find_source_picks_the_one_audio(tmp_path):
    run = _run(tmp_path)
    (run / "source" / "DROP-ORIGINAL-AUDIO-HERE.md").write_text("marker")   # ignored
    (run / "source" / "conversation.m4a").write_bytes(b"\x00")
    assert find_source(run).name == "conversation.m4a"


def test_find_source_errors_when_missing(tmp_path):
    with pytest.raises(SystemExit):
        find_source(_run(tmp_path))


def test_find_source_errors_when_ambiguous(tmp_path):
    run = _run(tmp_path)
    (run / "source" / "a.wav").write_bytes(b"\x00")
    (run / "source" / "b.mp3").write_bytes(b"\x00")
    with pytest.raises(SystemExit):
        find_source(run)
