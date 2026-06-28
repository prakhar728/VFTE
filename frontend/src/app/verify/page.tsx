import { ShieldCheck } from "lucide-react";

export const metadata = { title: "Verify a deletion receipt" };

export default function VerifyPage() {
  return (
    <div className="mx-auto min-h-dvh w-full max-w-2xl px-6 pb-24 pt-10">
      <div className="flex items-center gap-2.5">
        <ShieldCheck className="size-5 text-primary" />
        <span className="text-sm font-semibold tracking-tight">Verify a deletion receipt</span>
      </div>

      <h1 className="mt-8 text-2xl font-semibold tracking-tight">
        Proof your voiceprint was deleted
      </h1>
      <p className="mt-1.5 max-w-lg text-sm leading-relaxed text-muted-foreground">
        When you delete a voiceprint, we return a receipt — a small JSON object plus an
        Ed25519 signature. Anyone can verify it <strong>offline</strong> with our published
        public key; you never have to take our word for it.
      </p>

      <ol className="mt-6 space-y-4 text-sm leading-relaxed">
        <li>
          <span className="font-medium">1. Get the public key.</span> Fetch{" "}
          <code className="rounded bg-secondary px-1.5 py-0.5 font-mono text-xs">
            GET /v1/deletion-receipt-key
          </code>{" "}
          — it returns the PEM public key, its <code className="font-mono text-xs">key_id</code>,
          and whether it&apos;s enclave-sealed (<code className="font-mono text-xs">in_tee</code>).
          The <code className="font-mono text-xs">key_id</code> must match the one inside your
          receipt.
        </li>
        <li>
          <span className="font-medium">2. Confirm the key is genuine.</span> The same public
          key is bound into the enclave&apos;s TDX attestation at{" "}
          <code className="rounded bg-secondary px-1.5 py-0.5 font-mono text-xs">/attestation</code>{" "}
          (as <code className="font-mono text-xs">receipt_pubkey_sha256</code>), so you can prove
          the key belongs to the real, unmodified deletion code.
        </li>
        <li>
          <span className="font-medium">3. Verify the signature.</span> Run a reference
          verifier (Python or JS) against your downloaded{" "}
          <code className="font-mono text-xs">receipt.json</code> and the key:
        </li>
      </ol>

      <pre className="mt-4 overflow-x-auto rounded-xl border border-border bg-background/60 p-4 font-mono text-xs leading-relaxed text-muted-foreground">
{`# Python (needs the 'cryptography' package)
python deletion-receipt-verify.py receipt.json <public_key_raw_hex>

# Node.js (no dependencies)
node deletion-receipt-verify.js receipt.json <public_key_raw_hex>

# both print {"valid": true, ...} and exit 0 on a genuine receipt`}
      </pre>

      <p className="mt-4 text-xs text-muted-foreground">
        The canonicalization is exact (UTF-8 JSON, sorted keys, no whitespace), so any Ed25519
        library reproduces the signed bytes. Reference verifiers:{" "}
        <code className="font-mono text-xs">docs/deletion-receipt-verify.py</code> /{" "}
        <code className="font-mono text-xs">.js</code>.
      </p>
    </div>
  );
}
