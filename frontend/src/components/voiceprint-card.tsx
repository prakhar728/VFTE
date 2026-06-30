"use client";

import { useState } from "react";
import { Activity, Download, Fingerprint, ShieldCheck, ShieldQuestion, Trash2 } from "lucide-react";

import { api, type DeletionReceipt, type Voiceprint } from "@/lib/api";
import { cn, fmtTime } from "@/lib/utils";
import { downloadReceipt, verifyReceipt } from "@/lib/receipt";
import { downloadJSON, exportFilename } from "@/lib/voiceprint-export";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";

const EVENT_TONE: Record<string, string> = {
  enroll: "text-primary",
  identify: "text-sky-400",
  name_bind: "text-violet-400",
  control: "text-amber-400",
  forget: "text-destructive",
};

export function VoiceprintCard({
  vp,
  onChanged,
}: {
  vp: Voiceprint;
  onChanged: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);
  const [receipt, setReceipt] = useState<DeletionReceipt | null>(null);
  const [verified, setVerified] = useState<boolean | null>(null);
  const anonymous = !vp.identify_allowed;

  async function flip(
    flags: { identify_allowed?: boolean; enroll_allowed?: boolean },
    msg: string,
  ) {
    setBusy(true);
    setNote(null);
    try {
      await api.setFlags(vp.workspace_id, vp.voiceprint_id, flags);
      setNote(msg);
      onChanged();
    } catch {
      setNote("Couldn't save that — try again.");
    } finally {
      setBusy(false);
    }
  }

  async function exportVp() {
    setBusy(true);
    setNote(null);
    try {
      const env = await api.exportVoiceprint(vp.workspace_id, vp.voiceprint_id);
      downloadJSON(exportFilename(env), env);
      setNote("Signed export downloaded — re-import it any time to restore.");
    } catch {
      setNote("Export failed — try again.");
    } finally {
      setBusy(false);
    }
  }

  async function forget() {
    if (!confirm("Permanently delete this voiceprint? This can't be undone.")) return;
    setBusy(true);
    try {
      const res = await api.forget(vp.workspace_id, vp.voiceprint_id);
      if (res.receipt) {
        setReceipt(res.receipt);
        // verify the receipt client-side against the published key (no trust in us)
        try {
          const key = await api.deletionReceiptKey();
          setVerified(await verifyReceipt(res.receipt, key));
        } catch {
          setVerified(null);
        }
      } else {
        onChanged(); // no receipt (shouldn't happen on a real delete) → just refresh
      }
    } catch {
      setNote("Delete failed — try again.");
    } finally {
      setBusy(false);
    }
  }

  if (receipt) {
    return (
      <ReceiptPanel
        receipt={receipt}
        verified={verified}
        onDone={onChanged}
      />
    );
  }

  const usage = showAll ? vp.usage : vp.usage.slice(0, 4);

  return (
    <div className="animate-rise rounded-2xl border border-border bg-card/60 p-5 backdrop-blur-sm">
      {/* header */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <div className="mt-0.5 flex size-9 items-center justify-center rounded-lg border border-border bg-secondary text-primary">
            <Fingerprint className="size-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-[15px] font-semibold leading-tight">
                {vp.name || "Unnamed voiceprint"}
              </h3>
              {anonymous ? <Badge tone="warn">anonymous</Badge> : null}
            </div>
            <p className="mt-1 font-mono text-[11px] text-muted-foreground">
              {vp.workspace_id} · {vp.voiceprint_id}
            </p>
          </div>
        </div>
        <div className="text-right">
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
            quality
          </div>
          <div className="font-mono text-sm text-foreground">
            {vp.quality_score.toFixed(2)}
          </div>
        </div>
      </div>

      <p className="mt-3 text-xs text-muted-foreground">
        enrolled {vp.enroll_count}× · first seen {fmtTime(vp.created_at)} · last seen{" "}
        {fmtTime(vp.last_seen_at)}
      </p>

      {/* controls */}
      <div className="mt-4 divide-y divide-border rounded-xl border border-border bg-background/40">
        <ControlRow
          label="Stay anonymous"
          desc="Still grouped as one speaker, but never named in transcripts."
          checked={anonymous}
          disabled={busy}
          onChange={(on) =>
            flip({ identify_allowed: !on }, on ? "You're now anonymous." : "Identification re-enabled.")
          }
        />
        <ControlRow
          label="Pause enrollment"
          desc="Stop strengthening this voiceprint from new audio."
          checked={!vp.enroll_allowed}
          disabled={busy}
          onChange={(on) =>
            flip({ enroll_allowed: !on }, on ? "Enrollment paused." : "Enrollment resumed.")
          }
        />
      </div>

      {/* usage ledger */}
      <div className="mt-4">
        <div className="mb-2 flex items-center gap-2 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
          <Activity className="size-3.5" /> How it&apos;s been used
        </div>
        {vp.usage.length === 0 ? (
          <p className="text-xs text-muted-foreground/70">No recorded activity.</p>
        ) : (
          <ul className="space-y-1.5">
            {usage.map((u, i) => (
              <li key={i} className="flex items-center gap-2 text-xs">
                <span className={cn("w-16 shrink-0 font-medium", EVENT_TONE[u.event] ?? "text-foreground")}>
                  {u.event}
                </span>
                <span className="truncate text-muted-foreground">
                  {u.consumer}
                  {u.purpose ? ` · ${u.purpose}` : ""}
                </span>
                <span className="ml-auto shrink-0 font-mono text-[11px] text-muted-foreground/70">
                  {fmtTime(u.ts)}
                </span>
              </li>
            ))}
          </ul>
        )}
        {vp.usage.length > 4 ? (
          <button
            onClick={() => setShowAll((s) => !s)}
            className="mt-2 text-[11px] text-muted-foreground hover:text-foreground"
          >
            {showAll ? "Show less" : `Show all ${vp.usage.length}`}
          </button>
        ) : null}
      </div>

      {/* footer */}
      <div className="mt-4 flex items-center justify-between border-t border-border pt-4">
        <span className="text-xs text-muted-foreground">
          {note ?? "Download a signed backup, or permanently delete this voiceprint."}
        </span>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" disabled={busy} onClick={exportVp}>
            <Download className="size-3.5" />
            Export
          </Button>
          <Button variant="destructive" size="sm" disabled={busy} onClick={forget}>
            <Trash2 className="size-3.5" />
            Forget me
          </Button>
        </div>
      </div>
    </div>
  );
}

function ReceiptPanel({
  receipt,
  verified,
  onDone,
}: {
  receipt: DeletionReceipt;
  verified: boolean | null;
  onDone: () => void;
}) {
  return (
    <div className="animate-rise rounded-2xl border border-primary/30 bg-card/60 p-5 backdrop-blur-sm">
      <div className="flex items-start gap-3">
        <div className="mt-0.5 flex size-9 items-center justify-center rounded-lg border border-primary/40 bg-primary/5 text-primary">
          {verified === false ? (
            <ShieldQuestion className="size-4" />
          ) : (
            <ShieldCheck className="size-4" />
          )}
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-[15px] font-semibold leading-tight">Voiceprint deleted</h3>
            {verified === true ? (
              <Badge tone="emerald">verified ✓</Badge>
            ) : verified === false ? (
              <Badge tone="warn">verification failed</Badge>
            ) : (
              <Badge tone="muted">verify with CLI</Badge>
            )}
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            A signed, independently verifiable receipt was issued. Keep it as proof the
            voiceprint was erased — anyone can verify it offline with our public key.
          </p>
        </div>
      </div>

      <dl className="mt-4 space-y-1.5 rounded-xl border border-border bg-background/40 p-4 font-mono text-[11px]">
        <Row label="voiceprint" value={receipt.payload.voiceprint_id} />
        <Row label="deleted_at" value={receipt.payload.deleted_at} />
        <Row label="ledger_row" value={String(receipt.payload.ledger_row_id)} />
        <Row label="key_id" value={receipt.payload.key_id} />
      </dl>

      <div className="mt-4 flex items-center justify-between border-t border-border pt-4">
        <a
          href="/verify"
          className="text-xs text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
        >
          How to verify
        </a>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => downloadReceipt(receipt)}>
            <Download className="size-3.5" />
            Download .json
          </Button>
          <Button size="sm" onClick={onDone}>
            Done
          </Button>
        </div>
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="truncate text-foreground">{value}</dd>
    </div>
  );
}

function ControlRow({
  label,
  desc,
  checked,
  disabled,
  onChange,
}: {
  label: string;
  desc: string;
  checked: boolean;
  disabled?: boolean;
  onChange: (next: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-4 px-4 py-3">
      <div>
        <div className="text-sm font-medium">{label}</div>
        <div className="text-xs text-muted-foreground">{desc}</div>
      </div>
      <Switch checked={checked} disabled={disabled} onCheckedChange={onChange} aria-label={label} />
    </div>
  );
}
