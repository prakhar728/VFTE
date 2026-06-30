/**
 * Voiceprint export / import helpers (Task #4).
 *
 * Pure functions for file naming, parsing an uploaded export file, and summarizing an
 * import response, plus a small DOM download helper. The pure parts are unit-tested
 * (voiceprint-export.test.ts) — the browser-only download is kept trivial.
 */
import type {
  ExportBundle,
  ImportResponse,
  ImportResult,
  VoiceprintExport,
} from "@/lib/api";

/** Filename for a single voiceprint export. */
export function exportFilename(env: VoiceprintExport): string {
  return `fpm-voiceprint-${env.payload.voiceprint_id}.json`;
}

/** Filename for an export-all bundle (slug the email so it's filesystem-safe). */
export function bundleFilename(email: string | null): string {
  const slug = (email || "voiceprints").replace(/[^a-z0-9]+/gi, "-").replace(/^-+|-+$/g, "");
  return `fpm-voiceprints-${slug}.json`;
}

/** True when `o` looks like a single signed envelope (has a payload + signature). */
function isEnvelope(o: unknown): o is VoiceprintExport {
  return (
    !!o &&
    typeof o === "object" &&
    "payload" in o &&
    "signature" in o &&
    typeof (o as VoiceprintExport).signature === "string"
  );
}

/** True when `o` looks like an export-all bundle (a `voiceprints` array of envelopes). */
function isBundle(o: unknown): o is ExportBundle {
  return (
    !!o &&
    typeof o === "object" &&
    Array.isArray((o as ExportBundle).voiceprints)
  );
}

/**
 * Parse the text of an uploaded export file into an import body. Throws a human-readable
 * Error for anything that isn't a single envelope or a bundle (so the UI can show why).
 */
export function parseImportFile(text: string): VoiceprintExport | ExportBundle {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error("That file isn't valid JSON.");
  }
  if (isBundle(parsed)) {
    if (parsed.voiceprints.length === 0) throw new Error("This export contains no voiceprints.");
    if (!parsed.voiceprints.every(isEnvelope)) {
      throw new Error("This bundle has malformed entries.");
    }
    return parsed;
  }
  if (isEnvelope(parsed)) return parsed;
  throw new Error("This doesn't look like an FPM voiceprint export.");
}

const REASON_LABEL: Record<string, string> = {
  "bad-signature": "bad signature",
  "wrong-model": "wrong embedder model",
  "not-owner": "not your voiceprint",
  "bad-exemplars": "corrupt vectors",
};

/** A one-line, human-readable summary of an import response for the UI. */
export function summarizeImport(resp: ImportResponse): {
  created: number;
  merged: number;
  rejected: ImportResult[];
  message: string;
  ok: boolean;
} {
  const created = resp.results.filter((r) => r.status === "created").length;
  const merged = resp.results.filter((r) => r.status === "merged").length;
  const rejected = resp.results.filter((r) => r.status === "rejected");

  const parts: string[] = [];
  if (created) parts.push(`${created} restored`);
  if (merged) parts.push(`${merged} merged`);
  if (rejected.length) {
    const reasons = [...new Set(rejected.map((r) => REASON_LABEL[r.reason || ""] || r.reason || "rejected"))];
    parts.push(`${rejected.length} rejected (${reasons.join(", ")})`);
  }
  const message = parts.length ? parts.join(" · ") : "Nothing to import.";
  return { created, merged, rejected, message, ok: resp.imported > 0 && rejected.length === 0 };
}

/** Browser-only: trigger a download of `data` as a pretty-printed .json file. */
export function downloadJSON(filename: string, data: unknown): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
