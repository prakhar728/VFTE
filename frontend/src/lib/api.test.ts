/**
 * Unit tests for the Task #3 additions to the API client (Parts b + c): they pin the
 * URL + HTTP method + credentials each helper constructs, and that ids are encoded.
 * Runs on Node's built-in test runner: `node --test src/lib/*.test.ts`.
 */
import { test, afterEach } from "node:test";
import assert from "node:assert/strict";

import { api } from "./api.ts";

type Call = { url: string; init: RequestInit | undefined };
let calls: Call[] = [];

/** Stub global.fetch, recording each call and returning a canned JSON body. */
function stubFetch(body: unknown = {}) {
  calls = [];
  (globalThis as { fetch: unknown }).fetch = (url: string, init?: RequestInit) => {
    calls.push({ url, init });
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) });
  };
}

afterEach(() => {
  calls = [];
});

test("clipUrl POSTs to the proposal clip-url endpoint (no body)", async () => {
  stubFetch({ url: "https://c/audio", expires_at: 123, key_id: "k1" });
  const res = await api.clipUrl("prop_1");
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "/v1/me/proposals/prop_1/clip-url");
  assert.equal(calls[0].init?.method, "POST");
  assert.equal(calls[0].init?.credentials, "same-origin");
  assert.equal(calls[0].init?.body, undefined); // no-body POST
  assert.equal(res.url, "https://c/audio");
  assert.equal(res.key_id, "k1");
});

test("clipUrl url-encodes the proposal id", async () => {
  stubFetch();
  await api.clipUrl("prop/with space");
  assert.equal(calls[0].url, "/v1/me/proposals/prop%2Fwith%20space/clip-url");
});

test("recognitions GETs the transparency inbox", async () => {
  stubFetch({ email: "a@b.com", recognitions: [] });
  const res = await api.recognitions();
  assert.equal(calls[0].url, "/v1/me/recognitions");
  assert.equal(calls[0].init?.method, undefined); // GET
  assert.equal(calls[0].init?.credentials, "same-origin");
  assert.deepEqual(res.recognitions, []);
});

test("recognitionDetail GETs a single recognition, encoding the id", async () => {
  stubFetch({ recognition: {}, controls: {}, transcript_access: false });
  await api.recognitionDetail("rec/1");
  assert.equal(calls[0].url, "/v1/me/recognitions/rec%2F1");
  assert.equal(calls[0].init?.method, undefined); // GET
});
