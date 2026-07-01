"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { LogOut, ShieldCheck } from "lucide-react";

import { api, type Voiceprint } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { VoiceprintCard } from "@/components/voiceprint-card";
import { PendingInbox } from "@/components/pending-inbox";
import { RecognitionsInbox } from "@/components/recognitions-inbox";
import { ExportImportBar } from "@/components/export-import-bar";

export default function DashboardPage() {
  const router = useRouter();
  const [email, setEmail] = useState<string | null>(null);
  const [vps, setVps] = useState<Voiceprint[] | null>(null);

  const load = useCallback(async () => {
    try {
      const data = await api.voiceprints();
      setEmail(data.email);
      setVps(data.voiceprints);
    } catch {
      router.replace("/"); // not signed in
    }
  }, [router]);

  useEffect(() => {
    load();
  }, [load]);

  async function signOut() {
    await api.logout().catch(() => {});
    router.replace("/");
  }

  return (
    <div className="mx-auto min-h-dvh w-full max-w-2xl px-6 pb-24 pt-10">
      {/* top bar */}
      <header className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <ShieldCheck className="size-5 text-primary" />
          <span className="text-sm font-semibold tracking-tight">Your voiceprint</span>
        </div>
        <div className="flex items-center gap-3">
          {email ? (
            <span className="hidden font-mono text-xs text-muted-foreground sm:inline">
              {email}
            </span>
          ) : null}
          <Button variant="ghost" size="sm" onClick={signOut}>
            <LogOut className="size-3.5" />
            Sign out
          </Button>
        </div>
      </header>

      {/* P4 consent inbox — pending tags awaiting confirm/deny (hidden when none) */}
      <PendingInbox onResolved={load} />

      {/* Task #3 Part (c): transparency inbox — where your voice was recognized (hidden when none) */}
      <RecognitionsInbox />

      <div className="mt-8">
        <h1 className="text-2xl font-semibold tracking-tight">
          Voiceprints tied to you
        </h1>
        <p className="mt-1.5 max-w-lg text-sm leading-relaxed text-muted-foreground">
          Each workspace keeps a separate, independently-controlled entry. Stay
          anonymous, pause enrollment, or delete a voiceprint for good — enforced
          inside the enclave.
        </p>
        <ExportImportBar
          email={email}
          hasVoiceprints={!!vps && vps.length > 0}
          onImported={load}
        />
      </div>

      <div className="mt-6 space-y-4">
        {vps === null ? (
          <>
            <CardSkeleton />
            <CardSkeleton />
          </>
        ) : vps.length === 0 ? (
          <EmptyState />
        ) : (
          vps.map((vp) => (
            <VoiceprintCard
              key={`${vp.workspace_id}:${vp.voiceprint_id}`}
              vp={vp}
              onChanged={load}
            />
          ))
        )}
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="rounded-2xl border border-dashed border-border bg-card/30 p-10 text-center">
      <p className="text-sm font-medium">No voiceprint is stored for you yet.</p>
      <p className="mx-auto mt-1.5 max-w-sm text-xs text-muted-foreground">
        Once you&apos;re recorded in a meeting, your voiceprint will appear here —
        always under your control.
      </p>
    </div>
  );
}

function CardSkeleton() {
  return (
    <div className="rounded-2xl border border-border bg-card/40 p-5">
      <div className="flex items-center gap-3">
        <div className="size-9 animate-pulse rounded-lg bg-secondary" />
        <div className="space-y-2">
          <div className="h-3.5 w-32 animate-pulse rounded bg-secondary" />
          <div className="h-2.5 w-48 animate-pulse rounded bg-secondary" />
        </div>
      </div>
      <div className="mt-4 h-24 animate-pulse rounded-xl bg-secondary/60" />
    </div>
  );
}
