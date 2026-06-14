"use client";

import { useCallback, useEffect, useState } from "react";
import { Check, Inbox, X } from "lucide-react";

import { api, type Proposal } from "@/lib/api";
import { Button } from "@/components/ui/button";

/**
 * P4 consent inbox: lists proposals where a host tagged this user's voice
 * (`GET /v1/me/pending`) and lets them Confirm (claim the voiceprint + name)
 * or Deny (stay anonymous) via `/v1/confirm` · `/v1/deny`. Renders nothing
 * when there's nothing pending. `onResolved` refreshes the voiceprint list,
 * since a confirm moves a voiceprint into the user's owned set.
 */
export function PendingInbox({ onResolved }: { onResolved: () => void }) {
  const [items, setItems] = useState<Proposal[] | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setItems((await api.pending()).pending);
    } catch {
      setItems([]); // not signed in / no inbox → just hide
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function act(p: Proposal, kind: "confirm" | "deny") {
    setBusy(p.proposal_id);
    try {
      await (kind === "confirm" ? api.confirm(p.proposal_id) : api.deny(p.proposal_id));
      await load();
      onResolved();
    } catch {
      // leave the item in place; the user can retry
    } finally {
      setBusy(null);
    }
  }

  if (!items || items.length === 0) return null;

  return (
    <section className="mt-8 rounded-2xl border border-primary/30 bg-primary/5 p-5">
      <div className="flex items-center gap-2.5">
        <Inbox className="size-4 text-primary" />
        <h2 className="text-sm font-semibold tracking-tight">
          Pending identifications
        </h2>
      </div>
      <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">
        Someone tagged your voice in a meeting. <strong>Confirm</strong> to claim
        it as yours, or <strong>Deny</strong> to stay anonymous.
      </p>

      <ul className="mt-4 space-y-2.5">
        {items.map((p) => (
          <li
            key={p.proposal_id}
            className="flex items-center justify-between gap-3 rounded-xl border border-border bg-card px-4 py-3"
          >
            <div className="min-w-0">
              <p className="truncate text-sm font-medium">
                {p.proposed_name || "(unnamed)"}
              </p>
              <p className="truncate text-xs text-muted-foreground">
                tagged by {p.proposed_by} · workspace {p.workspace_id}
              </p>
            </div>
            <div className="flex shrink-0 items-center gap-2">
              <Button
                size="sm"
                onClick={() => act(p, "confirm")}
                disabled={busy === p.proposal_id}
              >
                <Check className="size-3.5" />
                Confirm
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => act(p, "deny")}
                disabled={busy === p.proposal_id}
              >
                <X className="size-3.5" />
                Deny
              </Button>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
