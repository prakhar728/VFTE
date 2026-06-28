# Cryptographic proof of fingerprint deletion (Task #1)

When a user deletes ("forgets") a voiceprint, FPM hard-deletes the row as before **and**
returns a *signed, independently verifiable* receipt — not just `{"deleted": true}`. This is
the trust primitive behind the "user-owned voice / verifiable trust" pitch: a delete the user
has to take our word for is exactly what the regulated/legal buyer won't accept.

## What you get back

`POST /v1/me/voiceprints/{ws}/{vid}/forget` →

```json
{
  "voiceprint_id": "vp_…",
  "deleted": true,
  "receipt": {
    "payload": {
      "version": "fpm-deletion-receipt-v1",
      "voiceprint_id": "vp_…",
      "workspace_id": "local-ws",
      "owner_email_hash": "<sha256(lower(email))>",
      "embedder_model": "campplus",
      "embedder_dim": 512,
      "deleted_at": "2026-06-28T12:34:56.789+00:00",
      "ledger_row_id": 1234,
      "alg": "ed25519",
      "key_id": "<sha256(pubkey)[:16]>"
    },
    "signature": "<base64 ed25519 signature over the canonical payload>",
    "alg": "ed25519",
    "key_id": "<sha256(pubkey)[:16]>"
  }
}
```

A receipt is issued **only on an actual deletion**. An idempotent re-delete returns
`{"deleted": false}` with no receipt.

## Why it proves something

- **Ed25519, TEE-sealed key.** The private signing key is derived from the CVM's
  hardware-bound key (dstack, path `fpm/deletion-signing` — *distinct* from the at-rest store
  key) so the operator can't forge a receipt. The public key is freely publishable, so anyone
  can verify **offline**.
- **The audit trail survives.** `forget` hard-deletes only the `voiceprints` row (embeddings
  gone). The append-only `usage_ledger` "forget" row is kept as the proof anchor; the receipt
  references it by `ledger_row_id`. Every issued receipt is also persisted append-only in
  `deletion_receipts` (hash only — no email plaintext).
- **The key is bound to the enclave.** `sha256(pubkey)` is folded into the `/attestation` TDX
  report_data (and echoed as `receipt_pubkey_sha256`), so a verifier can confirm via the quote
  that this pubkey belongs to the genuine deletion code at the published measurement.

## Canonicalization (the signed-bytes contract)

The signature covers the **canonical** payload bytes. This MUST be exact:

> **UTF-8 JSON, keys sorted lexicographically, separators `(",", ":")`, no extra whitespace.**

`alg` and `key_id` are stamped *into* the payload before signing, so they're covered too. Any
Ed25519 library that reproduces these bytes can verify.

## Verifying

1. Fetch the key: `GET /v1/deletion-receipt-key` → `{alg, public_key (PEM), public_key_raw_hex,
   key_id, in_tee}`. Check `key_id` matches the receipt.
2. (Optional, strongest) Confirm the key via `/attestation` → `receipt_pubkey_sha256`.
3. Run a reference verifier on the downloaded `receipt.json`:

```bash
python docs/deletion-receipt-verify.py receipt.json <public_key_raw_hex>
node   docs/deletion-receipt-verify.js receipt.json <public_key_raw_hex>
```

Both print `{"valid": true, "key_id": …, "canonical_sha256": …}` and exit 0 on a genuine
receipt. They agree byte-for-byte (asserted in `tests/test_deletion_receipt.py`). The
in-browser verifier (`frontend/src/lib/receipt.ts`) uses the same canonicalization via WebCrypto.

## Key derivation priority (mirrors `crypto.get_or_create_key()`)

1. **TEE sealed key** (`IN_TEE`): seed from dstack at `FPM_RECEIPT_SEAL_KEY_PATH`
   (`fpm/deletion-signing`). Production.
2. **`FPM_RECEIPT_KEY`** env (32-byte seed, hex or base64): off-TEE determinism for dev/CI.
3. **`DATA_DIR/.fpm-receipt.key`** (0600, auto-created): local dev fallback.

`key_id` is published from day one so future key rotation is an additive change.

## Out of scope (sequenced after — see TASK-01 §9)

- **Level B — crypto-shred** (per-voiceprint data key, destroy-key-to-delete).
- **Level C — re-enroll tombstone.**
- **Key rotation mechanics** (we ship `key_id` so it's additive later).
