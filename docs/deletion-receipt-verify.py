#!/usr/bin/env python3
"""Reference Python verifier for FPM deletion receipts (Task #1).

Proves a receipt is genuine OFFLINE — no contact with the service, only the published
Ed25519 public key. Mirrors fpm/receipts.py's canonicalization; kept standalone (only
`cryptography`) so a data subject / auditor can run it without the FPM codebase.

Usage:
    python deletion-receipt-verify.py <receipt.json> <pubkey.pem | rawhex>

<receipt.json> is the envelope from /forget: {payload, signature, alg, key_id}.
The second arg is a PEM SubjectPublicKeyInfo file path, or the 64-hex raw key
(GET /v1/deletion-receipt-key → public_key_raw_hex).

Prints JSON {valid, key_id, canonical_sha256}; exit 0 iff valid.
"""
import base64
import hashlib
import json
import sys

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


def canonical_bytes(payload: dict) -> bytes:
    """UTF-8 JSON, keys sorted lexicographically, separators (",", ":")."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def load_pubkey(arg: str):
    """A PEM file path or 64-hex raw key → (Ed25519PublicKey, raw 32 bytes)."""
    arg = arg.strip()
    if len(arg) == 64 and all(c in "0123456789abcdefABCDEF" for c in arg):
        raw = bytes.fromhex(arg)
        return Ed25519PublicKey.from_public_bytes(raw), raw
    with open(arg, "rb") as fh:
        pub = serialization.load_pem_public_key(fh.read())
    raw = pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return pub, raw


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python deletion-receipt-verify.py <receipt.json> <pubkey.pem|rawhex>", file=sys.stderr)
        return 2
    with open(sys.argv[1]) as fh:
        envelope = json.load(fh)
    pub, raw = load_pubkey(sys.argv[2])

    message = canonical_bytes(envelope["payload"])
    signature = base64.b64decode(envelope["signature"])
    try:
        pub.verify(signature, message)
        valid = True
    except InvalidSignature:
        valid = False

    out = {
        "valid": valid,
        "key_id": hashlib.sha256(raw).hexdigest()[:16],
        "canonical_sha256": hashlib.sha256(message).hexdigest(),
    }
    print(json.dumps(out))
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
