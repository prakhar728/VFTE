"""P4 Phase 2 — FPM-routed notify email (real SMTP behind a flag; stubbed in tests)."""
import smtplib

import pytest

import config
import notify


class _FakeSMTP:
    """Records the SMTP conversation instead of touching the network."""
    instances: list = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port = host, port
        self.started = False
        self.logged = None
        self.sent = []
        _FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self, context=None):
        self.started = True

    def login(self, user, password):
        self.logged = (user, password)

    def send_message(self, msg):
        self.sent.append(msg)


def test_notify_log_only_when_flag_off(monkeypatch):
    monkeypatch.setattr(config, "NOTIFY_EMAIL", False)
    # flag off → log-only, nothing sent, no exception
    assert notify.notify_identification("alice@x.com", "ws1", "host@x.com", "prop_1") is False


def test_notify_raises_when_enabled_without_smtp(monkeypatch):
    monkeypatch.setattr(config, "NOTIFY_EMAIL", True)
    monkeypatch.setattr(config, "SMTP_HOST", "")  # flag on but transport not configured
    with pytest.raises(RuntimeError):
        notify.notify_identification("alice@x.com", "ws1", "host@x.com", "prop_1")


def test_notify_sends_via_smtp_when_configured(monkeypatch):
    monkeypatch.setattr(config, "NOTIFY_EMAIL", True)
    monkeypatch.setattr(config, "SMTP_HOST", "smtp.test")
    monkeypatch.setattr(config, "SMTP_PORT", 587)
    monkeypatch.setattr(config, "SMTP_USER", "bot@test")
    monkeypatch.setattr(config, "SMTP_PASS", "pw")
    monkeypatch.setattr(config, "NOTIFY_FROM", "bot@test")
    monkeypatch.setattr(config, "DASHBOARD_URL", "http://localhost:8091/dashboard")
    monkeypatch.setattr(smtplib, "SMTP", _FakeSMTP)
    _FakeSMTP.instances.clear()

    ok = notify.notify_identification("alice@x.com", "ws1", "host@x.com", "prop_1")
    assert ok is True
    inst = _FakeSMTP.instances[-1]
    assert inst.host == "smtp.test" and inst.port == 587
    assert inst.started is True                      # STARTTLS
    assert inst.logged == ("bot@test", "pw")         # authenticated
    msg = inst.sent[-1]
    assert msg["To"] == "alice@x.com" and msg["From"] == "bot@test"
    assert "ws1" in msg["Subject"]
    body = msg.get_content()
    assert "http://localhost:8091/dashboard" in body  # confirm link
    assert "host@x.com" in body                        # who tagged them
