"""P4 Phase 2 — FPM-routed notify email (flag-guarded, provider-agnostic)."""
import pytest

import config
import notify


def test_notify_log_only_when_flag_off(monkeypatch):
    monkeypatch.setattr(config, "NOTIFY_EMAIL", False)
    # default (flag off) → log-only, nothing actually sent, no exception
    assert notify.notify_identification("alice@x.com", "ws1", "host@x.com", "prop_1") is False


def test_notify_raises_when_enabled_without_provider(monkeypatch):
    monkeypatch.setattr(config, "NOTIFY_EMAIL", True)
    with pytest.raises(NotImplementedError):
        notify.notify_identification("alice@x.com", "ws1", "host@x.com", "prop_1")
