"""Ed25519-signed deletion receipts (Task #1: cryptographic proof of deletion).

When a user deletes ("forgets") a voiceprint we hard-delete the row as before, but
also return a *signed, independently verifiable* receipt — not just `{"deleted": true}`.
The receipt is a small JSON payload plus a detached Ed25519 signature; anyone holding
the published public key can verify it OFFLINE, with no trust in the operator.

Why Ed25519 (asymmetric), not HMAC: the private key is sealed to the TEE (derived from
the CVM's hardware-bound key via a path distinct from the at-rest store key), so the
operator can't forge a receipt, while the public key is freely publishable so the data
subject — or their lawyer — can verify a deletion happened without contacting us.

Canonicalization (MUST be byte-exact — this is what gets signed):
    UTF-8 JSON, keys sorted lexicographically, separators (",", ":"), no extra whitespace.
A reference Python + JS verifier live in docs/deletion-receipt-verify.{py,js}; the test
suite asserts they agree byte-for-byte with what this module produces.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

RECEIPT_VERSION = "fpm-deletion-receipt-v1"
ALG = "ed25519"
_SEED_LEN = 32


# ── canonicalization + helpers (the signed-bytes contract) ───────────

def canonical_bytes(payload: dict) -> bytes:
    """The exact bytes that get signed/verified. See module docstring for the rules."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def owner_email_hash(email: str) -> str:
    """sha256(lowercased email) hex — what the receipt carries instead of plaintext PII.

    Lowercased to match the case-insensitive owner checks elsewhere; the owner verifies
    by recomputing this over their own address."""
    return hashlib.sha256((email or "").lower().encode("utf-8")).hexdigest()


def compute_key_id(public_key_raw: bytes) -> str:
    """key_id = first 16 hex chars of sha256(raw 32-byte ed25519 public key).

    Present from day one so future key rotation is verifiable (a receipt names the key
    that signed it)."""
    return hashlib.sha256(public_key_raw).hexdigest()[:16]


def _decode_seed(value: str) -> bytes:
    """Decode a 32-byte seed from hex (64 chars) or base64."""
    seed = bytes.fromhex(value) if len(value) == 64 else base64.b64decode(value)
    if len(seed) != _SEED_LEN:
        raise ValueError("FPM_RECEIPT_KEY must decode to 32 bytes (64 hex chars)")
    return seed


# ── verification (free function — no signer/private key needed) ──────

def verify_with_pubkey(envelope: dict, public_key_raw: bytes) -> bool:
    """Offline verify: does `envelope.signature` sign `envelope.payload` under this pubkey?

    The documented source of truth — any ed25519 library reproducing `canonical_bytes`
    can do this. Returns False on any tamper, wrong key, or malformed envelope (never
    raises)."""
    try:
        payload = envelope["payload"]
        signature = base64.b64decode(envelope["signature"])
        Ed25519PublicKey.from_public_bytes(public_key_raw).verify(
            signature, canonical_bytes(payload)
        )
        return True
    except (InvalidSignature, KeyError, TypeError, ValueError, Exception):  # noqa: BLE001 — any failure ⇒ not verified
        return False


class ReceiptSigner:
    """Holds the Ed25519 signing key and signs/verifies deletion receipts.

    Construct via `from_config()` (the production/dev path) which mirrors
    `crypto.get_or_create_key()`'s priority: TEE sealed key → `FPM_RECEIPT_KEY` env →
    0600 dev keyfile. Tests construct directly with an explicit 32-byte seed.
    """

    def __init__(self, seed: bytes, *, in_tee: bool = False):
        if len(seed) != _SEED_LEN:
            raise ValueError("ed25519 seed must be 32 bytes")
        self._priv = Ed25519PrivateKey.from_private_bytes(seed)
        self._pub = self._priv.public_key()
        self.in_tee = in_tee
        self._raw = self._pub.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.key_id = compute_key_id(self._raw)

    # ── construction (key derivation) ────────────────────────────
    @classmethod
    def from_config(cls) -> "ReceiptSigner":
        """Derive the signing key, priority order (mirrors crypto.get_or_create_key):

        1. **TEE sealed key** (`IN_TEE`): seed from the CVM's hardware-bound key via
           dstack at path `RECEIPT_SEAL_KEY_PATH` (distinct from the store key) — bound
           to this enclave, never on disk, unforgeable by the operator. Production path.
        2. **`FPM_RECEIPT_KEY`** env (hex/base64, 32 bytes): off-TEE determinism for dev/CI.
        3. **0600 dev keyfile** under DATA_DIR: local dev / fallback (auto-created).
        """
        import config
        from fpm.enclave import get_sealed_key

        sealed = get_sealed_key(path=config.RECEIPT_SEAL_KEY_PATH,
                                subject="deletion-receipt-signing")
        if sealed is not None:
            return cls(sealed[:_SEED_LEN], in_tee=True)

        if config.RECEIPT_KEY:
            return cls(_decode_seed(config.RECEIPT_KEY))

        return cls(_load_or_create_keyfile())

    # ── publishing the public key ────────────────────────────────
    def public_key_raw(self) -> bytes:
        return self._raw

    def public_key_pem(self) -> str:
        return self._pub.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("ascii")

    # ── sign / verify ────────────────────────────────────────────
    def sign(self, payload: dict) -> dict:
        """Sign a (mutable) payload dict → the envelope returned to the caller.

        Stamps `alg` + `key_id` INTO the payload before signing (so they're covered by
        the signature), then returns `{payload, signature(base64), alg, key_id}`."""
        payload = {**payload, "alg": ALG, "key_id": self.key_id}
        signature = self._priv.sign(canonical_bytes(payload))
        return {
            "payload": payload,
            "signature": base64.b64encode(signature).decode("ascii"),
            "alg": ALG,
            "key_id": self.key_id,
        }

    def verify(self, envelope: dict) -> bool:
        """Verify an envelope against THIS signer's public key (test/endpoint helper)."""
        return verify_with_pubkey(envelope, self._raw)


def _load_or_create_keyfile() -> bytes:
    """0600 dev keyfile under DATA_DIR (atomic create), mirroring crypto's keyfile path."""
    from config import DATA_DIR

    key_path = Path(DATA_DIR) / ".fpm-receipt.key"
    if key_path.exists():
        return key_path.read_bytes()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    seed = os.urandom(_SEED_LEN)
    fd, tmp = tempfile.mkstemp(dir=key_path.parent, prefix=".fpmrcpt_")
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        os.write(fd, seed)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, key_path)
    return seed
