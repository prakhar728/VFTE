"""Experiment configuration — one self-describing `config.yaml` per experiment folder.

The config IS the experiment record: model + methodology + diarizer engine/window + vocab,
so a run is reproducible and diff-able. You only drop `audio.wav`; everything else lives here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

VALID_MODES = {"offline", "realtime"}
VALID_ENGINES = {"diart", "diarizen"}


@dataclass
class AsrConfig:
    model: str = "large-v3-turbo"      # the exact Recato Whisper model (don't downgrade)
    compute_type: str = "int8"
    device: str = "cpu"                # no CUDA locally
    vocab: list[str] = field(default_factory=list)   # initial_prompt terms ([] / None = vocab off)
    vocab_compare: bool = True         # run both vocab-on and vocab-off and report the WER delta


@dataclass
class DiarizerConfig:
    engine: str = "diart"              # diart | diarizen (stub for now)
    window_sec: float = 5.0            # diart window (duration); try 120 for the 2-min experiment
    step_sec: float = 0.5


@dataclass
class ExperimentConfig:
    name: str
    dir: Path
    audio: str = "audio.wav"
    gold: str = "gold.json"
    mode: str = "offline"              # offline | realtime
    asr: AsrConfig = field(default_factory=AsrConfig)
    diarizer: DiarizerConfig = field(default_factory=DiarizerConfig)
    notes: str = ""

    @classmethod
    def load(cls, exp_dir: str | Path) -> "ExperimentConfig":
        exp_dir = Path(exp_dir)
        cfg_path = exp_dir / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"no config.yaml in {exp_dir}")
        data = yaml.safe_load(cfg_path.read_text()) or {}
        return cls(
            name=data.get("name", exp_dir.name),
            dir=exp_dir,
            audio=data.get("audio", "audio.wav"),
            gold=data.get("gold", "gold.json"),
            mode=data.get("mode", "offline"),
            asr=AsrConfig(**(data.get("asr") or {})),
            diarizer=DiarizerConfig(**(data.get("diarizer") or {})),
            notes=data.get("notes", ""),
        )

    # ── resolved paths ───────────────────────────────────────
    @property
    def audio_path(self) -> Path:
        return self.dir / self.audio

    @property
    def gold_path(self) -> Path:
        return self.dir / self.gold

    @property
    def results_dir(self) -> Path:
        return self.dir / "results"

    # ── validation ───────────────────────────────────────────
    def validate(self) -> list[str]:
        """Return a list of problems (empty = ok). Missing audio is a warning surfaced at run time."""
        errs = []
        if self.mode not in VALID_MODES:
            errs.append(f"mode '{self.mode}' not in {sorted(VALID_MODES)}")
        if self.diarizer.engine not in VALID_ENGINES:
            errs.append(f"diarizer.engine '{self.diarizer.engine}' not in {sorted(VALID_ENGINES)}")
        if self.diarizer.window_sec <= 0 or self.diarizer.step_sec <= 0:
            errs.append("diarizer window_sec/step_sec must be > 0")
        if not self.gold_path.exists():
            errs.append(f"gold transcript missing: {self.gold_path}")
        return errs

    def vocab_or_none(self) -> str | None:
        """The Whisper `initial_prompt` for the vocab-on pass (None when no vocab)."""
        return ", ".join(self.asr.vocab) if self.asr.vocab else None
