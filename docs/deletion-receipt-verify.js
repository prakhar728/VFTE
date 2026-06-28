#!/usr/bin/env node
/**
 * Reference JS verifier for FPM deletion receipts (Task #1).
 *
 * Proves a receipt is genuine OFFLINE — no contact with the service, only the
 * published Ed25519 public key. This file is the documented JS source of truth for
 * the canonicalization; it must agree byte-for-byte with fpm/receipts.py.
 *
 * Usage:
 *   node deletion-receipt-verify.js <receipt.json> <pubkey.pem | rawhex>
 *
 * <receipt.json> is the envelope returned by /forget: {payload, signature, alg, key_id}.
 * The second arg is either a PEM SubjectPublicKeyInfo file path, or the 64-hex raw key
 * (as published at GET /v1/deletion-receipt-key → public_key_raw_hex).
 *
 * Prints JSON: {valid, key_id, canonical_sha256}. Exit 0 iff valid.
 */
const crypto = require("crypto");
const fs = require("fs");

/** Canonical JSON: UTF-8, keys sorted lexicographically, separators (",",":"). */
function canonical(obj) {
  if (Array.isArray(obj)) return "[" + obj.map(canonical).join(",") + "]";
  if (obj && typeof obj === "object") {
    return (
      "{" +
      Object.keys(obj)
        .sort()
        .map((k) => JSON.stringify(k) + ":" + canonical(obj[k]))
        .join(",") +
      "}"
    );
  }
  return JSON.stringify(obj); // strings, ints, bool, null — matches Python json.dumps
}

/** Accept a PEM file path or 64-hex raw key → a Node KeyObject + the raw 32 bytes. */
function loadPubkey(arg) {
  let raw, keyObject;
  if (/^[0-9a-fA-F]{64}$/.test(arg.trim())) {
    raw = Buffer.from(arg.trim(), "hex");
    // Wrap the raw key in the fixed Ed25519 SPKI DER prefix so Node can import it.
    const prefix = Buffer.from("302a300506032b6570032100", "hex");
    keyObject = crypto.createPublicKey({
      key: Buffer.concat([prefix, raw]),
      format: "der",
      type: "spki",
    });
  } else {
    const pem = fs.readFileSync(arg, "utf8");
    keyObject = crypto.createPublicKey(pem);
    const der = keyObject.export({ type: "spki", format: "der" });
    raw = der.subarray(der.length - 32); // raw key = last 32 bytes of Ed25519 SPKI
  }
  return { keyObject, raw };
}

function main() {
  const [receiptPath, pubArg] = process.argv.slice(2);
  if (!receiptPath || !pubArg) {
    console.error("usage: node deletion-receipt-verify.js <receipt.json> <pubkey.pem|rawhex>");
    process.exit(2);
  }
  const envelope = JSON.parse(fs.readFileSync(receiptPath, "utf8"));
  const { keyObject, raw } = loadPubkey(pubArg);

  const message = Buffer.from(canonical(envelope.payload), "utf8");
  const signature = Buffer.from(envelope.signature, "base64");
  const valid = crypto.verify(null, message, keyObject, signature);

  const key_id = crypto.createHash("sha256").update(raw).digest("hex").slice(0, 16);
  const canonical_sha256 = crypto.createHash("sha256").update(message).digest("hex");

  console.log(JSON.stringify({ valid, key_id, canonical_sha256 }));
  process.exit(valid ? 0 : 1);
}

main();
