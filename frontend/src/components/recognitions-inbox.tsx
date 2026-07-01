"use client";

import { useCallback, useEffect, useState } from "react";
import { ChevronDown, Radar, ShieldOff, Trash2 } from "lucide-react";

import { api, type Recognition } from "@/lib/api";
import { fmtTime } from "@/lib/utils";
import { Button } from "@/components/ui/button";

/**
 * Task #3 Part (c): recognition transparency inbox. Lists every time a
 * voiceprint of the signed-in subject was auto-recognized in a meeting
 * (`GET /v1/me/recognitions`) — where, when, and in which app. Each row can
 * expand to the detail view (`GET /v1/me/recognitions/{id}`), which exposes the
 * same consent controls as the dashboard (stay anonymous / forget) but — by
 * design — NEVER any transcript. Renders nothing when there's nothing to show.
 */
export function RecognitionsInbox() {
  const [items, setItems] = useState<Recognition[] | null>(null);

  const load = useCallback(async () => {
    try {
      setItems((await api.recognitions()).recognitions);
    } catch {
      setItems([]); // not signed in / nothing to show → just hide
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  if (!items || items.length === 0) return null;

  return (
    <section className="mt-8 rounded-2xl border border-border bg-card/40 p-5">
      <div className="flex items-center gap-2.5">
        <Radar className="size-4 text-primary" />
        <h2 className="text-sm font-semibold tracking-tight">
          Where your voice was recognized
        </h2>
      </div>
      <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">
        Every meeting where a voiceprint of yours was auto-identified. This is
        metadata only — being recognized never grants anyone your transcript.
      </p>

      <ul className="mt-4 space-y-2.5">
        {items.map((r) => (
          <RecognitionRow key={r.recognition_id} r={r} onChanged={load} />
        ))}
      </ul>
    </section>
  );
}

function RecognitionRow({
  r,
  onChanged,
}: {
  r: Recognition;
  onChanged: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  async function stayAnonymous() {
    setBusy(true);
    setNote(null);
    try {
      await api.setFlags(r.workspace_id, r.voiceprint_id, { identify_allowed: false });
      setNote("You're now anonymous in this workspace.");
      onChanged();
    } catch {
      setNote("Couldn't save that — try again.");
    } finally {
      setBusy(false);
    }
  }

  async function forget() {
    if (!confirm("Permanently delete this voiceprint? This can't be undone.")) return;
    setBusy(true);
    setNote(null);
    try {
      await api.forget(r.workspace_id, r.voiceprint_id);
      setNote("Voiceprint deleted.");
      onChanged();
    } catch {
      setNote("Delete failed — try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <li className="rounded-xl border border-border bg-card px-4 py-3">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between gap-3 text-left"
      >
        <div className="min-w-0">
          <p className="truncate text-sm font-medium">
            {r.meeting_title || r.native_meeting_id || "Untitled meeting"}
          </p>
          <p className="truncate text-xs text-muted-foreground">
            {r.app || "meeting"} · workspace {r.workspace_id} · {fmtTime(r.ts)}
          </p>
        </div>
        <ChevronDown
          className={`size-4 shrink-0 text-muted-foreground transition-transform ${
            open ? "rotate-180" : ""
          }`}
        />
      </button>

      {open ? (
        <div className="mt-3 border-t border-border pt-3">
          <dl className="space-y-1.5 rounded-lg border border-border bg-background/40 p-3 font-mono text-[11px]">
            <Row label="voiceprint" value={r.voiceprint_id} />
            <Row label="workspace" value={r.workspace_id} />
            {r.native_meeting_id ? (
              <Row label="meeting" value={r.native_meeting_id} />
            ) : null}
            <Row label="when" value={fmtTime(r.ts)} />
          </dl>

          <div className="mt-3 flex items-center justify-between gap-3">
            <span className="text-xs text-muted-foreground">
              {note ?? "No transcript is shared here — only that you were identified."}
            </span>
            <div className="flex shrink-0 items-center gap-2">
              <Button variant="outline" size="sm" disabled={busy} onClick={stayAnonymous}>
                <ShieldOff className="size-3.5" />
                Stay anonymous
              </Button>
              <Button variant="destructive" size="sm" disabled={busy} onClick={forget}>
                <Trash2 className="size-3.5" />
                Forget me
              </Button>
            </div>
          </div>
        </div>
      ) : null}
    </li>
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
