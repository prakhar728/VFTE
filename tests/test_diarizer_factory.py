"""Branch A / step 2 — _default_diarizer_factory engine selection.

Runs in the torch-free CORE venv: both engine modules import only numpy+base (their torch
stacks load lazily), so the factory can construct either without diart/diarizen installed.
"""
import builtins

import pytest
from fastapi import HTTPException

import config
import main
from fpm.diarize.diart_engine import DiartDiarizer
from fpm.diarize.diarizen_engine import DiariZenDiarizer


def test_selects_diarizen(monkeypatch):
    monkeypatch.setattr(config, "DIARIZATION_ENGINE", "diarizen")
    assert isinstance(main._default_diarizer_factory(), DiariZenDiarizer)


def test_selects_diart(monkeypatch):
    monkeypatch.setattr(config, "DIARIZATION_ENGINE", "diart")
    assert isinstance(main._default_diarizer_factory(), DiartDiarizer)


def test_unknown_engine_503(monkeypatch):
    monkeypatch.setattr(config, "DIARIZATION_ENGINE", "bogus")
    with pytest.raises(HTTPException) as ei:
        main._default_diarizer_factory()
    assert ei.value.status_code == 503


def test_missing_engine_venv_is_503_not_500(monkeypatch):
    # If the engine module can't import (missing venv/deps), the factory must surface a clean
    # 503 ("engine not available"), not let an ImportError become a 500.
    monkeypatch.setattr(config, "DIARIZATION_ENGINE", "diarizen")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if "diarizen_engine" in name:
            raise ImportError("simulated: diarizen venv not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(HTTPException) as ei:
        main._default_diarizer_factory()
    assert ei.value.status_code == 503
