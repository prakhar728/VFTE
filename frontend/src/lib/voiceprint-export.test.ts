/**
 * Unit tests for the Task #4 export/import pure helpers. Runs on Node's built-in test
 * runner with native TS type-stripping (Node ≥ 22.6): `node --test src/lib/*.test.ts`.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  bundleFilename,
  exportFilename,
  parseImportFile,
  summarizeImport,
} from "./voiceprint-export.ts";

function envelope(vid: string, sig = "AAAA") {
  return { payload: { voiceprint_id: vid }, signature: sig, alg: "ed25519", key_id: "k1" };
}

test("exportFilename names the file after the voiceprint id", () => {
  assert.equal(exportFilename(envelope("vp_alice") as never), "fpm-voiceprint-vp_alice.json");
});

test("bundleFilename slugs the email and falls back when null", () => {
  assert.equal(bundleFilename("Alice@X.com"), "fpm-voiceprints-Alice-X-com.json");
  assert.equal(bundleFilename(null), "fpm-voiceprints-voiceprints.json");
});

test("parseImportFile accepts a single envelope", () => {
  const env = envelope("vp_a");
  assert.deepEqual(parseImportFile(JSON.stringify(env)), env);
});

test("parseImportFile accepts a non-empty bundle", () => {
  const bundle = { version: "v1", voiceprints: [envelope("vp_a"), envelope("vp_b")] };
  assert.deepEqual(parseImportFile(JSON.stringify(bundle)), bundle);
});

test("parseImportFile rejects invalid JSON", () => {
  assert.throws(() => parseImportFile("{not json"), /valid JSON/);
});

test("parseImportFile rejects an empty bundle", () => {
  assert.throws(() => parseImportFile(JSON.stringify({ voiceprints: [] })), /no voiceprints/);
});

test("parseImportFile rejects a bundle with malformed entries", () => {
  assert.throws(
    () => parseImportFile(JSON.stringify({ voiceprints: [{ nope: true }] })),
    /malformed/,
  );
});

test("parseImportFile rejects a non-export object", () => {
  assert.throws(() => parseImportFile(JSON.stringify({ hello: "world" })), /voiceprint export/);
});

test("summarizeImport reports created / merged / rejected with reasons", () => {
  const resp = {
    imported: 2,
    count: 3,
    results: [
      { voiceprint_id: "a", status: "created" as const },
      { voiceprint_id: "b", status: "merged" as const },
      { voiceprint_id: "c", status: "rejected" as const, reason: "bad-signature" },
    ],
  };
  const s = summarizeImport(resp);
  assert.equal(s.created, 1);
  assert.equal(s.merged, 1);
  assert.equal(s.rejected.length, 1);
  assert.equal(s.ok, false); // rejections present → not a clean import
  assert.match(s.message, /1 restored/);
  assert.match(s.message, /1 merged/);
  assert.match(s.message, /bad signature/);
});

test("summarizeImport.ok is true only on a clean, non-empty restore", () => {
  const clean = summarizeImport({
    imported: 1,
    count: 1,
    results: [{ voiceprint_id: "a", status: "created" }],
  });
  assert.equal(clean.ok, true);
  assert.match(clean.message, /1 restored/);

  const empty = summarizeImport({ imported: 0, count: 0, results: [] });
  assert.equal(empty.ok, false);
  assert.match(empty.message, /Nothing to import/);
});
