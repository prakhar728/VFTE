/**
 * Client-side deletion-receipt verification + download (Task #1).
 *
 * Verifies a signed receipt OFFLINE in the browser with the published Ed25519 public
 * key (no trust in the server's "verified" claim). Canonicalization MUST match
 * fpm/receipts.py + docs/deletion-receipt-verify.js byte-for-byte: UTF-8 JSON, keys
 * sorted lexicographically, separators (",",":").
 */
import type { DeletionReceipt, ReceiptKey } from "@/lib/api";

/** Canonical JSON — see module docstring. */
export function canonical(obj: unknown): string {
  if (Array.isArray(obj)) return "[" + obj.map(canonical).join(",") + "]";
  if (obj && typeof obj === "object") {
    return (
      "{" +
      Object.keys(obj as Record<string, unknown>)
        .sort()
        .map((k) => JSON.stringify(k) + ":" + canonical((obj as Record<string, unknown>)[k]))
        .join(",") +
      "}"
    );
  }
  return JSON.stringify(obj);
}

// Build over an explicit ArrayBuffer so the result is Uint8Array<ArrayBuffer> (a strict
// BufferSource) — satisfies WebCrypto's importKey/verify typing under TS lib.dom.
function hexToBytes(hex: string): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(new ArrayBuffer(hex.length / 2));
  for (let i = 0; i < out.length; i++) out[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  return out;
}

function b64ToBytes(b64: string): Uint8Array<ArrayBuffer> {
  const bin = atob(b64);
  const out = new Uint8Array(new ArrayBuffer(bin.length));
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** Fixed Ed25519 SPKI DER prefix → wraps a 32-byte raw key so WebCrypto can import it. */
const SPKI_PREFIX = hexToBytes("302a300506032b6570032100");

/**
 * Verify a receipt against the published key. Returns true/false, or null when the
 * runtime can't verify Ed25519 (older browser) — callers should then fall back to the
 * CLI verifier rather than show a false "unverified".
 */
export async function verifyReceipt(
  receipt: DeletionReceipt,
  key: ReceiptKey,
): Promise<boolean | null> {
  try {
    const subtle = globalThis.crypto?.subtle;
    if (!subtle) return null;
    const raw = hexToBytes(key.public_key_raw_hex);
    const spki = new Uint8Array(SPKI_PREFIX.length + raw.length);
    spki.set(SPKI_PREFIX);
    spki.set(raw, SPKI_PREFIX.length);
    const pub = await subtle.importKey("spki", spki, { name: "Ed25519" }, false, ["verify"]);
    const msg = new TextEncoder().encode(canonical(receipt.payload));
    return await subtle.verify("Ed25519", pub, b64ToBytes(receipt.signature), msg);
  } catch {
    return null; // unsupported algorithm / malformed key → "verify with CLI"
  }
}

/** Trigger a download of the receipt as a .json file. */
export function downloadReceipt(receipt: DeletionReceipt): void {
  const blob = new Blob([JSON.stringify(receipt, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `deletion-receipt-${receipt.payload.voiceprint_id}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
