"""AES-256-CBC + HMAC-SHA256 at-rest encryption for voiceprint BLOBs.

Ported from VoxTerm `audio/speakers/crypto.py` (MIT), simplified to one
cross-platform backend (PyCA `cryptography`) with the master key from the
environment (`FPM_DB_KEY` = 64 hex chars / 32 bytes) — drops the macOS Keychain.
In the TEE the key comes from a sealed-key/KMS (env); for local dev a 0600
keyfile is auto-created under DATA_DIR.

BLOB format:  MAGIC(4) || IV(16) || HMAC-SHA256(32) || ciphertext
AES-256-CBC, random IV per blob, encrypt-then-MAC, HKDF-derived enc/mac keys.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives import padding as _pad
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

_MAGIC = b"FPE1"
_IV_LEN = 16
_HMAC_LEN = 32
_KEY_LEN = 32
_HEADER_LEN = len(_MAGIC) + _IV_LEN + _HMAC_LEN
_ENC_LABEL = b"fpm-enc-v1"
_MAC_LABEL = b"fpm-mac-v1"


def _hkdf_expand(master: bytes, label: bytes, length: int = 32) -> bytes:
    return hmac.new(master, label + b"\x01", hashlib.sha256).digest()[:length]


def derive_keys(master: bytes) -> tuple[bytes, bytes]:
    return _hkdf_expand(master, _ENC_LABEL), _hkdf_expand(master, _MAC_LABEL)


def _aes(op: str, key: bytes, iv: bytes, data: bytes) -> bytes:
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    if op == "enc":
        padder = _pad.PKCS7(128).padder()
        padded = padder.update(data) + padder.finalize()
        enc = cipher.encryptor()
        return enc.update(padded) + enc.finalize()
    dec = cipher.decryptor()
    padded = dec.update(data) + dec.finalize()
    unpadder = _pad.PKCS7(128).unpadder()
    return unpadder.update(padded) + unpadder.finalize()


def encrypt_blob(master: bytes, plaintext: bytes) -> bytes:
    if not plaintext:
        return b""
    enc_key, mac_key = derive_keys(master)
    iv = os.urandom(_IV_LEN)
    ciphertext = _aes("enc", enc_key, iv, plaintext)
    tag = hmac.new(mac_key, _MAGIC + iv + ciphertext, hashlib.sha256).digest()
    return _MAGIC + iv + tag + ciphertext


def decrypt_blob(master: bytes, data: bytes) -> bytes:
    if not data:
        return b""
    if len(data) < _HEADER_LEN + 1 or data[: len(_MAGIC)] != _MAGIC:
        raise ValueError("not an FPM-encrypted blob")
    iv = data[len(_MAGIC) : len(_MAGIC) + _IV_LEN]
    stored = data[len(_MAGIC) + _IV_LEN : _HEADER_LEN]
    ciphertext = data[_HEADER_LEN:]
    enc_key, mac_key = derive_keys(master)
    expected = hmac.new(mac_key, _MAGIC + iv + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(stored, expected):
        raise ValueError("integrity check failed — tampered data or wrong key")
    return _aes("dec", enc_key, iv, ciphertext)


def is_encrypted(data: bytes) -> bool:
    return bool(data) and len(data) >= _HEADER_LEN + 1 and data[: len(_MAGIC)] == _MAGIC


def get_or_create_key() -> bytes:
    """Master key from `FPM_DB_KEY` (hex/base64, 32 bytes), else a 0600 dev keyfile."""
    env = os.environ.get("FPM_DB_KEY")
    if env:
        key = bytes.fromhex(env) if len(env) == 64 else base64.b64decode(env)
        if len(key) != _KEY_LEN:
            raise ValueError("FPM_DB_KEY must decode to 32 bytes (64 hex chars)")
        return key

    from config import DATA_DIR

    key_path = Path(DATA_DIR) / ".fpm.key"
    if key_path.exists():
        return key_path.read_bytes()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key = os.urandom(_KEY_LEN)
    fd, tmp = tempfile.mkstemp(dir=key_path.parent, prefix=".fpmkey_")
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        os.write(fd, key)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, key_path)
    return key
