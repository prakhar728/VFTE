"use client";

import { useRef, useState } from "react";
import { Download, Upload } from "lucide-react";

import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  bundleFilename,
  downloadJSON,
  parseImportFile,
  summarizeImport,
} from "@/lib/voiceprint-export";

/**
 * Dashboard toolbar (Task #4): download ALL of the user's voiceprints as one signed
 * bundle, or re-import a previously-exported file. Import surfaces the per-item outcome
 * (restored / merged / rejected with the reason) so a partial restore is legible.
 */
export function ExportImportBar({
  email,
  hasVoiceprints,
  onImported,
}: {
  email: string | null;
  hasVoiceprints: boolean;
  onImported: () => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [tone, setTone] = useState<"ok" | "warn" | null>(null);

  async function exportAll() {
    setBusy(true);
    setNote(null);
    try {
      const bundle = await api.exportAll();
      if (bundle.count === 0) {
        setTone("warn");
        setNote("Nothing to export yet.");
        return;
      }
      downloadJSON(bundleFilename(email), bundle);
      setTone("ok");
      setNote(`Downloaded ${bundle.count} signed voiceprint${bundle.count === 1 ? "" : "s"}.`);
    } catch {
      setTone("warn");
      setNote("Export failed — try again.");
    } finally {
      setBusy(false);
    }
  }

  async function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = ""; // allow re-selecting the same file
    if (!file) return;
    setBusy(true);
    setNote(null);
    try {
      const body = parseImportFile(await file.text());
      const resp = await api.importVoiceprints(body);
      const s = summarizeImport(resp);
      setTone(s.ok ? "ok" : "warn");
      setNote(s.message);
      if (resp.imported > 0) onImported();
    } catch (err) {
      setTone("warn");
      setNote(err instanceof Error ? err.message : "Import failed — try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mt-6 flex flex-wrap items-center gap-3">
      <Button variant="outline" size="sm" disabled={busy || !hasVoiceprints} onClick={exportAll}>
        <Download className="size-3.5" />
        Export all
      </Button>
      <Button
        variant="outline"
        size="sm"
        disabled={busy}
        onClick={() => fileRef.current?.click()}
      >
        <Upload className="size-3.5" />
        Import
      </Button>
      <input
        ref={fileRef}
        type="file"
        accept="application/json,.json"
        className="hidden"
        onChange={onFile}
      />
      {note ? (
        <span
          className={
            tone === "warn"
              ? "text-xs text-amber-400"
              : "text-xs text-muted-foreground"
          }
        >
          {note}
        </span>
      ) : null}
    </div>
  );
}
