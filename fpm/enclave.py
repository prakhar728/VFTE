"""Dstack (Phala TEE) integration — sealed key derivation + attestation quotes.

Inside a Phala CVM the dstack agent exposes a JSON-RPC interface over a Unix
socket (auto-discovered by dstack-sdk; default /var/run/dstack.sock). Two uses:

  - get_sealed_key(): derive the voiceprint-store master key from the CVM's
    hardware-bound root key. The key is bound to this app's identity, never
    written to disk, and unreadable by the operator — this is what makes the
    voiceprints sealed (crypto.py wires it in as the master). See get_or_create_key.
  - get_attestation_quote(): a TDX quote the dashboard/clients verify to prove
    they're talking to the real, unmodified enclave before trusting it.

Outside the CVM (local dev) or when the socket is unreachable, sealed-key
derivation returns None (caller falls back to env/keyfile) and quotes return
tagged stubs, so the service stays up and the UI can show an honest "no seal".
"""
from __future__ import annotations

import hashlib
import logging
import os

logger = logging.getLogger(__name__)

IN_TEE = os.environ.get("IN_TEE", "false").lower() == "true"
# Optional override for the dstack simulator (HTTP endpoint) or explicit socket.
_DSTACK_ENDPOINT = os.environ.get("DSTACK_AGENT_URL") or None
# Derivation path for the store key — stable, so the same CVM app re-derives the
# same key across restarts/redeploys (changing it would orphan existing blobs).
_KEY_PATH = os.environ.get("FPM_SEAL_KEY_PATH", "fpm/voiceprint-store")

_client = None
_client_init_failed = False


def _get_client():
    """Lazy-init a single DstackClient; cache success and failure separately."""
    global _client, _client_init_failed
    if _client is not None or _client_init_failed:
        return _client
    try:
        from dstack_sdk import DstackClient

        _client = DstackClient(_DSTACK_ENDPOINT) if _DSTACK_ENDPOINT else DstackClient()
    except Exception as e:  # noqa: BLE001 — any SDK/socket error → graceful stub mode
        logger.warning("dstack client init failed: %s", e)
        _client_init_failed = True
        _client = None
    return _client


def get_sealed_key(path: str | None = None, subject: str = "voiceprint-store-encryption") -> bytes | None:
    """Derive a 32-byte master from the CVM's hardware-bound key, or None.

    `path` selects the derivation path (defaults to the voiceprint-store path
    `_KEY_PATH`); pass a distinct path (e.g. `fpm/deletion-signing`) to derive an
    INDEPENDENT key — same CVM, different path ⇒ a different, unrelated key. So the
    receipt-signing seed is cryptographically separate from the at-rest store key.

    Returns None when not in a TEE or the agent is unreachable, so the caller can
    fall back to the env/keyfile path. The dstack key is SHA-256'd to a fixed 32
    bytes so it's the right length regardless of the SDK's raw width.
    """
    if not IN_TEE:
        return None
    client = _get_client()
    if client is None:
        return None
    try:
        resp = client.get_key(path or _KEY_PATH, subject)
        raw = resp.decode_key()
        return hashlib.sha256(raw).digest()
    except Exception as e:  # noqa: BLE001
        logger.warning("dstack get_key failed (%s) — falling back to keyfile", e)
        return None


def get_attestation_quote(nonce: str = "", bind: bytes = b"") -> str:
    """Hex-encoded TDX quote from the dstack agent; tagged stub outside the TEE.

    `bind` is extra bytes folded into report_data alongside the nonce — used to
    anchor the deletion-receipt public key (`bind=sha256(pubkey)`) to the enclave
    measurement, so a verifier can prove via the quote that this pubkey belongs to
    the genuine deletion code at the published measurement.
    """
    if not IN_TEE:
        return "stub_attestation_quote_not_in_tee"
    client = _get_client()
    if client is None:
        return "stub_attestation_quote_dstack_unreachable"
    try:
        return client.get_quote(_normalize_report_data(nonce, bind)).quote
    except Exception as e:  # noqa: BLE001
        logger.warning("dstack get_quote failed: %s", e)
        return "stub_attestation_quote_dstack_unreachable"


def _normalize_report_data(nonce: str, bind: bytes = b"") -> bytes:
    """Pack nonce (+ optional `bind` bytes) into ≤64 bytes for TDX report_data.

    With `bind`, report_data = sha256(nonce_bytes || bind) (always 32 bytes) so both
    freshness (nonce) and the bound pubkey are committed. Without it, the legacy
    nonce-only packing is unchanged (empty → 32 zero bytes)."""
    raw = nonce.encode("utf-8") if nonce else b""
    if bind:
        return hashlib.sha256(raw + bind).digest()
    if not raw:
        return b"\x00" * 32
    return hashlib.sha256(raw).digest() if len(raw) > 64 else raw
