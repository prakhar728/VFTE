"use client";

import { useCallback, useEffect, useState } from "react";
import { Check, Inbox, Play, X } from "lucide-react";

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
          <PendingCard
            key={p.proposal_id}
            p={p}
            busy={busy === p.proposal_id}
            onAct={(kind) => act(p, kind)}
          />
        ))}
      </ul>
    </section>
  );
}

/**
 * One pending proposal. When a `clip_ref` is attached (Task #3 Part b), a
 * "Hear the clip" button mints a short-lived signed URL and streams it in an
 * inline <audio> so the subject can listen *before* confirming or denying.
 */
function PendingCard({
  p,
  busy,
  onAct,
}: {
  p: Proposal;
  busy: boolean;
  onAct: (kind: "confirm" | "deny") => void;
}) {
  const [clipUrl, setClipUrl] = useState<string | null>(null);
  const [clipBusy, setClipBusy] = useState(false);
  const [clipError, setClipError] = useState(false);

  async function hearClip() {
    setClipBusy(true);
    setClipError(false);
    try {
      setClipUrl((await api.clipUrl(p.proposal_id)).url);
    } catch {
      setClipError(true);
    } finally {
      setClipBusy(false);
    }
  }

  return (
    <li className="rounded-xl border border-border bg-card px-4 py-3">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium">
            {p.proposed_name || "(unnamed)"}
          </p>
          <p className="truncate text-xs text-muted-foreground">
            tagged by {p.proposed_by} · workspace {p.workspace_id}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Button size="sm" onClick={() => onAct("confirm")} disabled={busy}>
            <Check className="size-3.5" />
            Confirm
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => onAct("deny")}
            disabled={busy}
          >
            <X className="size-3.5" />
            Deny
          </Button>
        </div>
      </div>

      {p.clip_ref ? (
        <div className="mt-3 border-t border-border pt-3">
          {clipUrl ? (
            <audio
              controls
              autoPlay
              src={clipUrl}
              className="h-9 w-full"
            />
          ) : (
            <div className="flex items-center gap-2.5">
              <Button
                size="sm"
                variant="outline"
                onClick={hearClip}
                disabled={clipBusy}
              >
                <Play className="size-3.5" />
                {clipBusy ? "Loading…" : "Hear the clip"}
              </Button>
              <span className="text-xs text-muted-foreground">
                {clipError
                  ? "Couldn't load the clip — try again."
                  : "Listen before you decide."}
              </span>
            </div>
          )}
        </div>
      ) : null}
    </li>
  );
}
