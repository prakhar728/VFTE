# Contract C4 — propose / confirm / deny + consent-resolve (FROZEN)

> **Frozen seam, identical in both repos.** This file is the FPM↔Conclave contract for the P4
> trust handshake (ARCHITECTURE §4 C4, §3 handshake, §5 P4). Freeze it before any P4 code lands
> — the same discipline that made the C2/C3 freeze safe before A/B/C. The body below is **byte-identical**
> in `FPM/docs/build/contract-C4.md` and `conclave-shape-rotator/docs/plans/contract-C4.md`. Do not let
> the two drift; record per-branch deviations in the branch plan, not here.

## Roles & disciplines (recap)

- **FPM is the consent authority.** Identity lives on the voiceprint; the display name is a projection
  `voiceprint_id → confirmed owner_email → name`.
- **Conclave** queries FPM at projection time and caches the name/visibility decision; on observing a
  `confirmed` proposal it sweeps its stored transcripts and re-resolves the name. **The display label
  string is the immutable join key** — re-resolve rewrites only `resolved_speakers[label]["name"]`,
  never the label key or `Signal.said_by` (C3).

## Auth model

- **Write side (`/v1/propose`) and read side (`/v1/consent/resolve/...`) are M2M** — Bearer token under
  the existing **`knowledge`** scope (`require_scope("knowledge")`, workspace-checked via
  `caller.allows_workspace`). Reuses the binding-family scope `/v1/knowledge` already uses; no new token
  contract. Conclave's `fpm_api_token` must carry `knowledge` (in addition to `diarize`).
- **Confirm/deny (`/v1/confirm`, `/v1/deny`) are session-authed** — the data subject signed into the FPM
  consent dashboard (`require_user`, the dev-login/Google session). These are the only human-reached P4
  routes, mirroring `consent_api.py`.
- **Errors** use FPM's uniform envelope: `{"error": {"status", "message"}}`.

## Proposal state (FPM-owned)

`proposals` row: `{proposal_id, workspace_id, voiceprint_id, proposed_email, proposed_by,
proposed_name, status, created_at, confirmed_at, denied_at}`.

- `status ∈ {pending, confirmed, denied}`.
- **Unique** on `(workspace_id, voiceprint_id, proposed_email)` → propose is idempotent.
- Emails are compared **case-insensitively** and stored **lowercased** (matches `consent_api`/auth).

---

## Write side — `POST /v1/propose` (M2M, scope `knowledge`)

Request:
```json
{
  "workspace": "ws1",
  "voiceprint_id": "vp_abc",
  "proposed_email": "alice@x.com",
  "proposed_by": "host@x.com",
  "proposed_name": "Alice"
}
```

Response:
```json
{
  "proposal_id": "prop_0a1b2c3d4e5f6071",
  "status": "pending",            // "pending" | "confirmed"
  "auto_confirmed": false,
  "voiceprint_id": "vp_abc",
  "name": null,                   // set only when confirmed (and consent allows)
  "owner_email": null             // set only when confirmed
}
```

Rules:
- **Idempotent.** Re-proposing the same `(workspace, voiceprint_id, proposed_email)` returns the existing
  proposal (same `proposal_id`, current `status`) — no duplicate row, `proposed_name`/`proposed_by` of the
  first proposal are retained.
- **Auto-confirm** when **self-tag** (`proposed_by == proposed_email`, case-insensitive) **OR**
  `config.CONSENT_AUTOCONFIRM` is on. Auto-confirm runs the shared confirm path
  (`claim_owner` + `set_name`, audited): `status="confirmed"`, `auto_confirmed=true`, response carries the
  bound `name` + `owner_email`.
- Otherwise **pending**: `status="pending"`, `auto_confirmed=false`, `name=null`, `owner_email=null`
  (Phase 2 fires the FPM-routed notify email here; Phase 1 with the flag off is log-only).
- `403` if caller not authorized for `workspace`; `404` if the voiceprint is unknown in that workspace.

---

## Confirm / deny — session-authed (FPM consent dashboard)

`POST /v1/confirm`  body `{ "proposal_id": "prop_..." }`
- **Phase 1:** any valid dashboard session may confirm (single-actor spine; dev-login).
- **Phase 2:** only the target confirms — `proposal.proposed_email == session_email` else `403`.
- On confirm: `claim_owner(ws, vid, proposed_email)` + `set_name(ws, vid, proposed_name, actor=email)`;
  `status→confirmed`, `confirmed_at` stamped.
- **Consent-bypass guard (Phase 2):** if `identify_allowed == False` for the voiceprint, bind
  `owner_email` but **do not set a name**; response `name=null`. Revoked consent never re-attaches a name.
- Response:
```json
{ "proposal_id": "prop_...", "status": "confirmed",
  "voiceprint_id": "vp_abc", "name": "Alice", "owner_email": "alice@x.com" }
```

`POST /v1/deny`  body `{ "proposal_id": "prop_..." }`
- Same auth as confirm (Phase 2 target-only). `status→denied`, `denied_at` stamped. **No binding** — the
  speaker stays `Speaker N`.
- Response: `{ "proposal_id": "prop_...", "status": "denied" }`.

`404` if `proposal_id` is unknown; idempotent on a terminal status (confirming a confirmed proposal
returns its confirmed view; denying a denied one returns its denied view).

---

## Read side — consent-resolve (M2M, scope `knowledge`) — the projection keystone

`GET /v1/consent/resolve/{workspace}/{voiceprint_id}`
```json
{
  "voiceprint_id": "vp_abc",
  "name": "Alice",               // null whenever identify_allowed=False OR unbound/unnamed
  "owner_email": "alice@x.com",  // null if no owner bound
  "visibility": "named"          // "named" | "anonymous" | "unknown"
}
```

`POST /v1/consent/resolve/{workspace}`  (batch)  body `{ "voiceprint_ids": ["vp_a", "vp_b"] }`
```json
{ "resolved": {
    "vp_a": { "name": "Alice", "owner_email": "alice@x.com", "visibility": "named" },
    "vp_b": { "name": null,    "owner_email": null,          "visibility": "unknown" }
} }
```

`visibility` semantics (read-side consent gate — mirrors the `/v1/identify` gate, `main.py` ~L215):
- **`named`** — voiceprint exists, `identify_allowed=True`, and has a non-empty `name` → `name` returned.
- **`anonymous`** — voiceprint exists but `identify_allowed=False` **or** has no name → `name=null`
  (`owner_email` may still be present if bound).
- **`unknown`** — voiceprint not found in this workspace → all fields null.

`name` is **null whenever `identify_allowed=False`** regardless of binding — this is the single read-side
gate Conclave trusts; it must never surface a withheld name.

---

## Conclave consumption (the other half of the seam)

- **Host tag** → Conclave maps the meeting's display `label → voiceprint_id` (from
  `resolved_speakers`) and calls `POST /v1/propose` with `proposed_by = the host's logged-in email`,
  `workspace = fpm_workspace_for(conclave_workspace_id)`.
- On `status == "confirmed"` (self-tag / autoconfirm), Conclave immediately runs
  `reresolve_voiceprint(voiceprint_id, name)` — sweep the workspace's stored sessions, rewrite
  `resolved_speakers[label]["name"]` **only** for entries whose `voiceprint_id` matches, via
  `set_metadata`. **Never** rewrite the label key or `Signal.said_by`. Cross-transcript by construction.
- On `status == "pending"` (Phase 2), Conclave shows the tag as pending; no name flips until the target
  confirms. A read-time consent-resolve backstop is wired into `to_transcript` (cached ~60s TTL) so a
  later confirm surfaces on next load even without a re-tag.

## Scope of P4 (locked)

- **Per-room (per-workspace).** Confirm binds only that workspace's voiceprint; re-resolve stays within
  the workspace. No cross-workspace propagation in P4 (that is P5's email-hub aggregation by `owner_email`).
- **Name on confirm = host-typed `proposed_name`** (self-edit deferred to P5; no directory lookup).
- **Re-resolve = pull-by-Conclave, on-next-load** (sweep on observing `confirmed` + cached consent-query);
  **no FPM→Conclave push callback.**

## Dev flag (recorded deviation, ARCHITECTURE §7)

`FPM_CONSENT_AUTOCONFIRM` (default **OFF**) generalizes the specced self-tag auto-confirm so the spine
(binding → re-resolve → projection) can be tested before email/permissions exist. It rides the real
propose→confirm path (precedent: `FPM_DEV_LOGIN`). Phase 2 flips it off → pending + notify + target-only
authz + the consent-bypass guard.
