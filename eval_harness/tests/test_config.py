"""C1 — ExperimentConfig load + validation."""
from pathlib import Path

from eval_harness.harness.config import ExperimentConfig

EXP = Path(__file__).resolve().parents[1] / "experiments" / "eval-conversation"


def test_loads_example():
    cfg = ExperimentConfig.load(EXP)
    assert cfg.name == "eval-conversation"
    assert cfg.mode == "offline"
    assert cfg.asr.model == "large-v3-turbo" and cfg.asr.compute_type == "int8"
    assert cfg.asr.vocab_compare is True
    assert cfg.diarizer.engine == "diart" and cfg.diarizer.window_sec == 5
    assert cfg.gold_path.exists()                    # gold ships with the example
    assert cfg.results_dir == EXP / "results"


def test_vocab_prompt():
    cfg = ExperimentConfig.load(EXP)
    assert cfg.vocab_or_none() == "Recato, Priya, Arjun, Sunnyvale, cortado"


def test_validate_ok_and_catches_bad():
    cfg = ExperimentConfig.load(EXP)
    assert cfg.validate() == []                      # example is valid
    cfg.mode = "nope"
    cfg.diarizer.engine = "whisperx"
    cfg.diarizer.step_sec = 0
    errs = cfg.validate()
    assert any("mode" in e for e in errs)
    assert any("engine" in e for e in errs)
    assert any("step_sec" in e for e in errs)


def test_missing_config_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        ExperimentConfig.load(tmp_path)
