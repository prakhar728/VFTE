"""A.4 — AES-256-CBC+HMAC at-rest encryption for voiceprint BLOBs."""
import os

import numpy as np
import pytest

from fpm.store import crypto

KEY = os.urandom(32)


def test_roundtrip():
    pt = np.random.randn(512).astype(np.float32).tobytes()
    blob = crypto.encrypt_blob(KEY, pt)
    assert crypto.is_encrypted(blob)
    assert crypto.decrypt_blob(KEY, blob) == pt


def test_tamper_detected():
    blob = bytearray(crypto.encrypt_blob(KEY, b"secret voiceprint bytes"))
    blob[-1] ^= 0xFF
    with pytest.raises(ValueError):
        crypto.decrypt_blob(KEY, bytes(blob))


def test_wrong_key_rejected():
    blob = crypto.encrypt_blob(KEY, b"x" * 64)
    with pytest.raises(ValueError):
        crypto.decrypt_blob(os.urandom(32), blob)


def test_empty_and_non_blob():
    assert crypto.encrypt_blob(KEY, b"") == b""
    assert crypto.decrypt_blob(KEY, b"") == b""
    assert not crypto.is_encrypted(b"")
    with pytest.raises(ValueError):
        crypto.decrypt_blob(KEY, b"random bytes, not a blob")


def test_env_key_hex(monkeypatch):
    k = os.urandom(32)
    monkeypatch.setenv("FPM_DB_KEY", k.hex())
    assert crypto.get_or_create_key() == k
