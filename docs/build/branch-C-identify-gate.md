# Branch C — Live read-only + confidence gate (P1 + P3, bundled)

**Repo:** FPM · **Base:** `main` (post-FF) · **Working branch:** `branch-C-identify` ·
**Autonomy:** agent-autonomous (test-first) · **Depends on:** Contract **C2** (must preserve the
`/v1/diarize` per-segment shape) and Contract **C1** (`StreamingDiarizer`).
**Bundled because** both P1 and P3 edit `fpm/identify.py:SessionIdentifier` (and both touch
`_maybe_lock`/`_identify`) — keeping them on one branch avoids self-conflict.

> This is the enriched build plan. The frozen architecture lives in `ARCHITECTURE.md` — deviations
> from it are recorded **here** (§5 Decisions), not by editing it. It supersedes the original
> 5-bullet stub; nothing in scope has been dropped, several gaps found during code review have been
> added (read-only must also skip the usage-ledger write; the gate's "confidence ≥ floor" has a
> unit bug; the A↔C buffer hand-off lands here).

## Goal
- **P1 — read-only live mode.** A `read_only` mode on `SessionIdentifier` for the **live (diart)**
  path: classify against *existing* voiceprints for display + run vote-lock **in memory** for stable
  session labels, but **mint nothing and write nothing to the store** (single-writer invariant —
  ARCHITECTURE §3: "diart = live, read-only … mints nothing, writes nothing").
- **P3 — confidence / min-duration gate.** So weak diarization can't pollute voiceprints: gate
  **exemplar-append** and **anonymous-mint** on segment quality; keep **vote-counting and
  MATCH-locking permissive** (else hard-to-ID speakers never stabilize).

---

## 0. Contract compliance — does C break C1–C4 or step on A? (audit, verified 2026-06-13)

**Verdict: C breaks no contract and edits no file branch A edits.** The only A↔C contact is a single
*logical* micro-contract (the `buffered_batch` attribute name), handled defensively via `getattr`.

| Contract | What it requires | How C complies | Edits a file A touches? |
|---|---|---|---|
| **C1** — `StreamingDiarizer` seam (`fpm/diarize/base.py`): engines emit only `Segment{start,end,local_speaker}`; identity **always** re-embeds with fixed CAM++ | C implements/modifies **no engine** and **does not edit `base.py`**. It only `getattr`-reads the optional `buffered_batch` hint. Identity still re-embeds every segment with CAM++ — the §6 unbounded-buffer change *upholds* this invariant for batch engines (without it they can't be re-embedded). | **No** — `base.py` untouched by **both** branches (A's revised §6 puts the attr on its own `diarizen_engine.py`). |
| **C2** — `/v1/diarize` NDJSON shape (`main.py:_segment_dict`): 7 keys `{start,end,voiceprint_id,name,decision,confidence,local_speaker}` + final `transcript` line | `_segment_dict` **not touched**. Read-only/gate emit the **same 7 keys** (`voiceprint_id` may be `null` — key always present, never dropped); **no new required keys, no new `decision` strings**. `tag="live"` reuses the same streaming body + final transcript line. | **No** — A doesn't touch the endpoint; C edits only **inside** `diarize_endpoint`. |
| **C3** — `resolved_speakers` (Conclave `SessionMetadata`) | Conclave/B-side schema. C is FPM-only and never touches Conclave. | N/A (disjoint repo) |
| **C4** — propose/confirm/deny + consent-query | P4/P5 core branch. Not in C's scope. | N/A |

**Shared-file verdict (A↔C):**
- `main.py` — **disjoint regions**: A = `_default_diarizer_factory` (L49-56); C = inside
  `diarize_endpoint` (L216-266). ✓
- `config.py` — **disjoint**: A = engine line L62 + `DIARIZEN_*`; C = `MIN_SEGMENT_SEC` near L56. ✓
- `fpm/diarize/base.py` — **neither edits** (A's revised §6). C only `getattr`-reads. ✓
- `fpm/diarize/mock.py` — **neither edits**; C's batch test-double lives in its own test file. ✓
- `fpm/identify.py` — **C-exclusive** (A forbidden, branch-A §2/§8). ✓
- New C test files — zero conflict surface. ✓

**The one logical (non-textual) micro-contract:** the hint name must be **exactly** `buffered_batch`.
C's `getattr(diarizer, "buffered_batch", False)` is *silent* on a name mismatch → it would fall back
to the bounded buffer and diarizen identity would silently break past 15 s (no crash). Verify the
spelling against A's `diarizen_engine.py` at the **A+C joint integration gate**. (Details in §6.)

---

## 1. Current state (verified against the tree on `main`, 2026-06-13)

What exists today and is **extended, not rebuilt** (ARCHITECTURE §8):

- `fpm/identify.py:SessionIdentifier` — the identity spine. Key facts verified by reading the file:
  - `__init__` (L66-83) takes `store, embedder, diarizer, workspace_id, *, sample_rate,
    lock_min_votes, consumer`. **No `read_only` flag yet.**
  - `_identify` (L148-180): locked→`LOCKED`; slice+embed, `emb is None`→`PENDING` (L159-160);
    **appends the exemplar BEFORE classify** (L162-164); classify (L166); vote (L167-168); `_maybe_lock`.
  - `_maybe_lock` (L182-202): clear-leader vote check (`count >= lock_min` and `count != runner`);
    `cand == _UNKNOWN` → `_mint_anonymous` (**store write**); else `store.log_usage` (**store write**,
    L195) + resolve name. Then `_relabel_history`.
  - `_mint_anonymous` (L217-224): builds a `Voiceprint`, adds the accumulated `_exemplars[spk]`,
    `store.upsert` (**store write**).
  - **Bounded 15 s trailing buffer**: `BUFFER_SEC = 15.0` (L47), trimmed in `_append` (L130-136),
    sliced in `_slice` (L138-144). **This is the A↔C coupling point — see §6.**
  - Decision vocabulary in use: `MATCH | ANON | AMBIGUOUS | LOW | UNKNOWN | PENDING | LOCKED |
    RELABELED`. We add **no new** decision strings.
- `fpm/match.py:classify` (L34-58) → `MatchResult{decision, voiceprint_id, score, confidence}`.
  **`score` is raw cosine; `confidence` is sigmoid-calibrated [0,1]** (L30-31). Decisions already
  encode `MATCH_ACCEPT`/`MATCH_REJECT`/`AMBIGUOUS_MARGIN` (L52-57). This matters for §5/G1.
- `config.py` — matching block at L48-56 (`MATCH_ACCEPT=0.45`, etc.). **No `MIN_SEGMENT_SEC` yet.**
  `DIARIZATION_ENGINE` is at L62 — **Branch A's edit region; keep our new constant far from it.**
- `main.py:diarize_endpoint` (L216-266) — dispatches on `tag`: `gmeet`→enroll, else builds
  `SessionIdentifier(...)` (L251, **no `read_only`**) and streams `_segment_dict` NDJSON + a final
  `{"type":"transcript", …}` line. `_segment_dict` (L59-68) is the **C2** shape (7 keys). **No `live`
  tag exists** (`grep` confirms). `_default_diarizer_factory` (L49-56) is **Branch A territory — do
  not touch.**
- `fpm/diarize/base.py` — **C1**: `Segment{start,end,local_speaker}` (+ `.duration` property, L38-40)
  and the abstract `StreamingDiarizer`. Frozen except Branch A's additive `buffered_batch` hint (§6).
- `fpm/diarize/mock.py:MockDiarizer(script)` — deterministic, script-driven; the test harness for
  the whole pipeline. We extend it (or subclass) for the batch-engine + buffer tests.
- `fpm/store/store.py` — writes we must suppress in read-only: `upsert` (L164), `log_usage`
  (L321-332). Reads we keep: `centroids` (L376), `get` (L348), `identify_allowed` (L250),
  `list_ids` (L368). For "writes nothing" assertions: `list_ids` + `usage_for_voiceprint`
  (L334-345, newest-first list) are the observation hooks.
- Tests: `tests/test_identify.py` (live identify), `tests/test_identify_relabel.py` (mint + relabel),
  `tests/test_diarize_api.py` (endpoint). **Convention = real `VoiceprintStore(tmp)` + real CAM++
  embedder under `skipif MODEL missing` + `MockDiarizer(script)`.** We follow this exactly (the
  stub's "mock store" wording is superseded — see §4 "test substrate").

**What does NOT exist and must be built:** the `read_only` flag + write-suppression; the quality
gate (`MIN_SEGMENT_SEC` + the two gate points); the `tag="live"` endpoint branch; the
`buffered_batch` buffer-sizing consumption (§6); and the four new/extended test files.

---

## 2. Scope (files)

| File | Action | Notes |
|---|---|---|
| `fpm/identify.py` | **EDIT** | `read_only` flag; write-suppression; gate (`_passes_gate` + 2 sites); reorder classify above exemplar-append; `buffered_batch` buffer sizing (§6); add `import config` (module-attr access for `MIN_SEGMENT_SEC`). **C owns this whole file.** |
| `config.py` | **EDIT (1 new const)** | Add `MIN_SEGMENT_SEC` in the **matching block (after L56)** — far from `DIARIZATION_ENGINE` (L62, Branch A). Reference it as `config.MIN_SEGMENT_SEC` (module-attr) so tests can monkeypatch (§4). **Do not touch `DIARIZATION_ENGINE`/`DIARIZEN_*`.** |
| `main.py` | **EDIT (endpoint body only)** | Add `tag="live"` → `SessionIdentifier(read_only=True)`. **Only inside `diarize_endpoint` (L216-266).** Do **not** touch `_default_diarizer_factory` (A), `_segment_dict` (C2-frozen). |
| `tests/test_identify_gate.py` | **NEW** | Steps 3-4 + 6-7 (gate + read-only, pipeline level). |
| `tests/test_diarize_live_api.py` | **NEW** | Step 8 (endpoint `tag="live"` read-only e2e). |
| `tests/test_identify_buffer.py` | **NEW** | Step 9 (`buffered_batch` buffer sizing, §6). |
| `fpm/diarize/mock.py` | **NOT edited** | The §6 batch test-double (feed→`[]`, finish→all, `buffered_batch=True`) lives **inside `tests/test_identify_buffer.py`** — the shared C1 harness stays untouched, so zero shared-test-file surface with A. |

**Do NOT touch:** `_default_diarizer_factory`, `_segment_dict`, `fpm/match.py`, `fpm/diarize/base.py`
(frozen C1 — **neither A nor C edits it**; C only `getattr`-reads `buffered_batch`, which A defines on
its own `diarizen_engine.py`), `fpm/diarize/mock.py` (shared C1 harness — batch double goes in C's test
file), `fpm/store/store.py`, `record_routes.py` (Conclave/B), `MATCH_*`/`DIARIZEN_*` config.

---

## 3. Exact deltas

### 3a. `read_only` (P1)

- **`__init__`**: add `read_only: bool = False` (keyword-only, after `lock_min_votes`); store as
  `self._read_only`. Default `False` ⇒ **offline/post behaviour byte-for-byte unchanged**.
- **`_maybe_lock`** — suppress both writes, lock to a session-stable label:
  ```
  if cand == _UNKNOWN:
      if self._read_only:
          vp_id = None                       # live: stable session label, no store write
      elif self._exemplars.get(spk):         # P3 mint gate: need ≥1 qualifying exemplar
          vp_id = self._mint_anonymous(spk)
      else:
          return None                        # gated weak unknown → don't lock, keep voting
      label = IdentifiedSegment(0, 0, spk, vp_id, None, "ANON", confidence)
  else:
      if not self._read_only:
          self._store.log_usage(self._ws, cand, "identify", self._consumer, "matched in meeting")
      name = self._name_of(cand)
      decision = "MATCH" if name is not None else "ANON"
      label = IdentifiedSegment(0, 0, spk, cand, name, decision, confidence)
  ```
  Notes: the MATCH branch still **locks** in read-only (it is a pure read of existing centroids) — it
  only skips the **ledger write**. `_name_of` (reads `identify_allowed`/`get`) is unchanged and
  read-only-safe. Unknowns in read-only lock to `voiceprint_id=None`, decision `"ANON"` → stable via
  `local_speaker`, no new decision string, C2 shape intact (id is `null`, a value Conclave already
  handles for `PENDING`/undecided).

### 3b. Quality gate (P3)

- **`config.py`**: `MIN_SEGMENT_SEC = float(os.environ.get("FPM_MIN_SEGMENT_SEC", "1.0"))` placed
  right after `AMBIGUOUS_MARGIN`/score-calibration (≈L56). Default **1.0 s** is deliberately
  conservative so **existing 4 s-turn fixtures and behaviour are unchanged** (the gate is a no-op for
  clean, long segments; it only bites on genuinely short/overlapped ones).
- **`_identify`** — reorder so classify precedes exemplar-append, vote stays ungated, append is gated:
  ```
  res = classify(emb, self._store.centroids(self._ws))     # moved ABOVE the append
  votes = self._votes.setdefault(spk, Counter())
  votes[res.voiceprint_id if res.decision == "MATCH" else _UNKNOWN] += 1   # vote ALWAYS (ungated)
  if self._passes_gate(seg, res):
      self._exemplars.setdefault(spk, [])
      if len(self._exemplars[spk]) < 20:
          self._exemplars[spk].append(emb)
  locked = self._maybe_lock(spk, res.confidence)
  ```
  Reordering is safe: `classify` does not depend on `_exemplars`; exemplars are consumed only at mint.
- **`_passes_gate`**:
  ```
  def _passes_gate(self, seg, res) -> bool:
      import config
      if seg.duration < config.MIN_SEGMENT_SEC:     # too short → unreliable embedding
          return False
      if res.decision == "AMBIGUOUS":               # top-2 too close → likely overlap, don't pollute
          return False
      return True
  ```
- **Mint precondition** already shown in 3a: mint only when `_exemplars.get(spk)` is non-empty, i.e.
  ≥1 gate-passing exemplar accumulated. A speaker who only ever produces short/overlapped segments
  never accumulates one → never mints → stays `voiceprint_id=None` ("permanently-unnameable", by
  design). MATCH-lock (`cand != _UNKNOWN`) is **never** gated.

### 3c. Endpoint (P1 wiring)

- In `diarize_endpoint` only, after the `gmeet` branch:
  ```
  read_only = (tag == "live")
  ident = SessionIdentifier(request.app.state.store, embedder, diarizer, workspace,
                            sample_rate=sr, consumer=caller.name, read_only=read_only)
  ```
  `tag="offline"` (default) stays the authoritative writer; `tag="live"` is read-only. Streaming body
  and `_segment_dict` are untouched (C2 preserved). Conclave chooses `live` vs `offline` per
  ARCHITECTURE §3 (live diart instance vs post writer) — its side is a B/record-routes concern,
  noted, not in this branch.

---

## 4. Test-gated steps (write test first → green → atomic commit each)

**Test substrate (supersedes the stub's "mock store").** Match the existing convention: real
`VoiceprintStore(db_path=tmp_path, key=os.urandom(32)).open()` + real CAM++ via
`OnnxSpeakerEmbedder(MODEL)` under `pytestmark = skipif(not MODEL.exists())` + `MockDiarizer(script)`.
"Writes nothing" is asserted **black-box**: `store.list_ids(ws)` count unchanged and
`store.usage_for_voiceprint(ws, id)` empty/unchanged. *(Optional: a tiny `SpyStore` wrapper that
counts `upsert`/`log_usage` calls for a precise "0 write calls" assertion in step 6 — nice-to-have,
not required; the black-box checks are authoritative and match repo style.)*

**Controlling segment quality in tests.** Duration is taken from the *script* (`Segment.end-start`),
audio from real fixtures — so we can script a short span over embeddable audio. Because the embedder
itself returns `None` on very short spans (→ `PENDING` before the gate is even reached — see the
`test_identify_relabel.py` note), tests **monkeypatch `config.MIN_SEGMENT_SEC` upward** (e.g. 5.0) so
a 4 s embeddable segment (`emb != None`) is gated by **duration**, proving the gate — not the
embedder — blocks it. This is why §3b references `config.MIN_SEGMENT_SEC` (module-attr) rather than
`from config import …`.

1. **Read-only writes nothing.** `SessionIdentifier(read_only=True)` over [enrolled Alice + unknown
   spkB]. Assert: `set(store.list_ids("ws1")) == {alice_id}` (no anon minted);
   `store.usage_for_voiceprint("ws1", alice_id) == []` (log_usage suppressed); every unknown-speaker
   segment has `voiceprint_id is None`. *(Unit, pipeline.)*
2. **Read-only still identifies (display).** In the same read-only run, Alice's resolved segments
   carry `voiceprint_id == alice_id` and `name == "Alice"`. *(Unit.)*
3. **Read-only labels stay stable.** Each `local_speaker` resolves to exactly one
   `(voiceprint_id)` across the session (unknown → a single stable `None` label, Alice → her id) and
   emits `LOCKED` after enough votes — no flicker. *(Unit.)*
4. **Gate blocks exemplar-append (counts).** `monkeypatch config.MIN_SEGMENT_SEC = 3.0`. Unknown spkB
   scripted `[Segment(0,2), Segment(2,6), Segment(6,10)]` (2 s gated; two 4 s pass). It mints (≥1
   qualifying), and the minted voiceprint's `exemplars`/`exemplar_count == 2` — the 2 s segment was
   excluded. *(Unit; **writing** path, `read_only=False`.)*
5. **Gate blocks anonymous-mint (unnameable).** `monkeypatch config.MIN_SEGMENT_SEC = 5.0`. Unknown
   spkB, all 4 s segments → 0 qualifying exemplars → **no mint**: `store.list_ids("ws1")` has no anon
   (only Alice, or empty), and **every** output segment for that speaker has `voiceprint_id is None`
   and never decision `LOCKED` (stays `UNKNOWN`/`PENDING`). *(Unit.)*
6. **Gate does NOT break the vote / MATCH-lock.** `monkeypatch config.MIN_SEGMENT_SEC = 5.0` (all
   segments gated from append). Enrolled **Alice** still vote-MATCH-**locks** to `alice_id` and
   retro-relabels earlier provisional chunks — proving voting and MATCH-lock are ungated. *(Unit.)*
7. **Regression — defaults unchanged.** With default `MIN_SEGMENT_SEC` (1.0) and `read_only=False`,
   the existing 4 s-turn scenario behaves exactly as `test_identify.py`/`test_identify_relabel.py`:
   Alice MATCHes, unknown mints anonymous, labels stabilize, late lock relabels. *(Unit — may be
   covered by leaving the existing suites green rather than a new test; assert explicitly here.)*
8. **Endpoint `tag="live"` is read-only e2e.** Mirror `test_diarize_api.py` (MockDiarizer factory).
   `POST /v1/diarize tag=live`: 200, `content-type` NDJSON, every segment carries the full C2 key set
   `{start,end,voiceprint_id,name,decision,confidence,local_speaker}` with `end>start`, Alice's name
   appears, unknown stays `voiceprint_id=None`, and **after** the call `store.list_ids("ws1")` shows
   no new rows (no mint) and Alice's ledger is empty. Plus a `tag="offline"` control on the same
   client that **does** write (one new anon row) — proving the branch is tag-selected. *(Integration.)*
9. **`buffered_batch` buffer sizing (A↔C, §6).** A **local batch double** (defined in this test file;
   feed→`[]`, finish→all segments, `buffered_batch=True`) over a **>15 s** clip (e.g. 24 s = 6×4 s) with an enrolled
   speaker. With the hint honoured, `finish()` re-embeds **every** segment → the enrolled speaker is
   identified across the whole clip (not just the final 15 s). Control: the same clip with the
   default bounded buffer leaves early segments `PENDING`. *(Integration; the *real* diarizen e2e is
   the A+C joint gate — this proves C's consumption in isolation.)*

**Regression (must stay green, unchanged):** `tests/test_identify.py`,
`tests/test_identify_relabel.py`, `tests/test_diarize_api.py`.

---

## 5. Decisions (deviations / resolutions — recorded here, not in ARCHITECTURE)

- **G1 — the gate floor is min-duration + decision-tier, NOT a raw `confidence ≥ MATCH_ACCEPT`
  compare.** The stub said "require `confidence ≥ floor`" reusing `MATCH_ACCEPT`. Two problems found
  in code: (a) **unit mismatch** — `MatchResult.confidence` is sigmoid-calibrated `[0,1]` while
  `MATCH_ACCEPT=0.45` is a **raw cosine**; comparing them is meaningless. (b) **definitional** — an
  unknown speaker (the one we want to *mint*) scores **below** the accept threshold against existing
  centroids by definition, so a literal `confidence ≥ accept` floor would block **every** mint. The
  reused threshold is therefore applied **via `classify`'s decision tiers** (which already encode
  `MATCH_ACCEPT`/`REJECT`/`AMBIGUOUS_MARGIN`): exclude only `AMBIGUOUS` (overlap/borderline) from
  exemplar-append, and gate on `MIN_SEGMENT_SEC`. *(Open sub-decision: also exclude `LOW`?
  Recommended **no** — a genuinely new speaker who happens to score `LOW` against an existing
  centroid should still be allowed to mint; excluding `LOW` would strand similar-sounding new
  speakers as permanently unnameable. Keep `LOW` admissible; revisit if false mints appear.)*
- **G2 — read-only must also suppress `log_usage`, not only `upsert`/mint.** The stub listed
  "skip `_mint_anonymous` and `store.upsert`" but missed the **MATCH-lock `log_usage` write** (L195),
  which is a row into `usage_ledger`. ARCHITECTURE §3 requires live to write **nothing**; suppressing
  it is mandatory. (Covered by step 1's ledger assertion.)
- **G3 — `tag="live"` selects read-only** (additive to the existing `gmeet`/`offline` dispatch),
  rather than a new `read_only`/`mode` form field. Reason: reuses the established tag idiom, stays
  inside the endpoint body, and the live-vs-post choice is naturally a routing decision Conclave makes
  per instance. *(Alternative: explicit `read_only: bool = Form(False)` — cleaner separation but adds
  a request field; not chosen.)*
- **G4 — unknowns in read-only lock to `voiceprint_id=None` (decision `"ANON"`), session-stable via
  `local_speaker`.** No in-memory synthetic id is minted; `local_speaker` is already the stable
  session key, and live output is provisional/overwritten by the post pass. No new decision string.
- **G5 — gate runs in read-only too, but is moot for mint** (nothing is minted). The exemplar-append
  gate still executes (cheap, harmless — the exemplars are simply unused). Keeps the two features
  orthogonal and composable: `read_only` governs **writes**, the gate governs **mint-worthiness**.
- **G6 — config default `MIN_SEGMENT_SEC=1.0`** chosen so existing tests/behaviour are unchanged
  (all current fixtures use 4 s turns with clean MATCH/UNKNOWN decisions). Tunable via
  `FPM_MIN_SEGMENT_SEC`; calibration against real meetings is post-merge, not in scope.

---

## 6. The A↔C coupling: bounded buffer vs batch engine (carried from branch-A §6)

**This consumption was explicitly handed to Branch C by `branch-A-engine.md` §6 and MUST land here.**
`SessionIdentifier` keeps a **bounded 15 s** trailing buffer and re-embeds by slicing. diart emits
segments *incrementally* (their audio still in-window) — correct. **DiariZen returns `[]` from every
`feed()` and emits ALL segments at `finish()`** — by then the buffer holds only the last 15 s, so
`_slice` returns `None` for every earlier segment → `PENDING`, no identity. A naïve batch engine
would identify only the **final 15 s** of any clip.

**Resolution (zero shared-file edit — matches branch-A's revised §6):**
1. **Branch A** declares `buffered_batch = True` as a class attribute on **`DiariZenDiarizer` in its
   own new file `fpm/diarize/diarizen_engine.py`** — **A does NOT edit `base.py`** (the frozen C1 file
   stays untouched). diart/mock simply lack the attribute.
2. **Branch C** (here) consumes it **defensively via `getattr`** in `identify.py` (C's own file), so C
   is correct **whether or not A has merged yet**:
   - `start()`: `self._max_buf = None if getattr(self._diarizer, "buffered_batch", False) else int(BUFFER_SEC * self._sr)`
   - `_append()`: only trim the front when `self._max_buf is not None`.
   Attribute absent (diart/mock) → `False` → byte-for-byte the current bounded 15 s buffer. No
   dependency on A's merge order.
3. **Test:** step 9 above, using a **local batch double defined inside `tests/test_identify_buffer.py`**
   (feed→`[]`, finish→all, `buffered_batch=True`) — *not* an edit to the shared `mock.py`. The
   real-engine identity-across-a-long-clip e2e (diarizen + CAM++) is the **A+C joint integration gate**
   (ARCHITECTURE §6 merge order: B → A & C → joint A+C test → P4→P5), not owned solely by C.

**Confirmed (answering branch-A §6's coordination request):** Branch C **owns** this buffer-sizing
consumption — it is in scope here (step 9), not deferred to the joint gate.

**Two logical micro-contracts with A (no textual overlap, but must agree):**
- **Attribute name** must be exactly `buffered_batch`. C's `getattr` is *silent* on a name mismatch
  (→ falls back to bounded buffer → diarizen identity silently breaks past 15 s, no crash). Verify the
  spelling matches A's `diarizen_engine.py` at the A+C joint gate.
- **Memory safety:** C's unbounded buffer for batch engines is safe **only because A caps clip length**
  (`DIARIZEN_MAX_CLIP_SEC`, enforced in `DiariZenDiarizer.feed()`). The live diart path is
  `buffered_batch=False` → buffer stays bounded at 15 s regardless. Documented coupling, not a break.

*(Long-term answer is ARCHITECTURE §5 "#6 DiariZen windowing for RAM"; the buffer hint is the minimal
P0-compatible bridge.)*

---

## 7. Merge-conflict / scope discipline (ARCHITECTURE §6)

- **A ⟂ C ≈ 90%** — share only `main.py`, in **disjoint regions**: A edits
  `_default_diarizer_factory` (L49-56); C edits **inside** `diarize_endpoint` (L216-266) only.
  Reciprocal discipline holds (branch-A §8 commits to not touching the endpoint/`identify.py`).
- **`config.py`** — A edits the engine line (L62) + adds `DIARIZEN_MAX_CLIP_SEC`; C adds
  `MIN_SEGMENT_SEC` in the matching block (≈L56). Disjoint regions. **C must not touch
  `DIARIZATION_ENGINE`/`DIARIZEN_*`; A must not touch `MATCH_*`/`MIN_SEGMENT_SEC`.**
- **`fpm/diarize/base.py`** — **neither branch edits it** (A's revised §6 puts `buffered_batch` on its
  own `diarizen_engine.py`, not the base class). C only `getattr`-reads the hint. Zero overlap.
  *(Note: branch-A §2's table line still marks `base.py` EDIT — a stale row vs A's revised §6; the
  operative decision is no base.py edit. Confirm at the joint gate.)*
- **`fpm/diarize/mock.py`** — neither branch edits it; C's batch double lives in its own test file.
- **`fpm/identify.py`** — **C-exclusive**; A is forbidden from touching it (branch-A §2/§8). No conflict.
- **New test files** — zero conflict surface; `mock.py` untouched (batch double is in-test).
- **B ⟂ C ≈ 100%** (disjoint repos). **Merge order:** B → A & C → joint A+C integration → P4→P5.

---

## 8. Things to be careful about

- **Do NOT gate the vote or MATCH-lock** — only exemplar-append + anonymous-mint. Gating votes makes
  hard-to-ID speakers stay `PENDING` forever and breaks retro-relabel. (Enforced by step 6.)
- **Read-only writes NOTHING** — mint, `upsert`, **and `log_usage`** all suppressed (G2). Don't leave
  the ledger write behind.
- The gate creates **permanently-unnameable** speakers (`voiceprint_id=None`) — intended; the UI must
  not offer "name this speaker" for them (Conclave/B/P4 concern — note there, not fixable here).
- **Preserve C2.** Read-only and gated paths must still emit the full 7-key `_segment_dict` shape;
  `voiceprint_id=None` is a valid value, not a missing key.
- **Read-only cache may be stale** re: voiceprints minted by the post pass — acceptable, live output
  is provisional; ARCHITECTURE §10 says reload-on-session-start (the store cache is per-`open()`).
- **Reorder, don't duplicate**: classify moves above exemplar-append; make sure the vote still fires
  for every embeddable segment (ungated) and only the append is conditional.
- **`config.MIN_SEGMENT_SEC` via module-attr**, not `from config import` — or the monkeypatch-based
  gate tests can't move the floor (same trap `match.py`'s bound `MATCH_ACCEPT` would hit).

---

## 9. Out of scope / deferred (explicit)

- **Conclave's choice of `live` vs `offline`** and the "don't offer name-this-speaker for
  `voiceprint_id=None`" UI rule — Conclave / B / P4.
- **Second FPM instance / two-instance topology** — ARCHITECTURE §5 P0 tail (Branch A defers it too).
- **Real diarizen long-clip identity e2e** — the A+C joint integration gate, not this branch.
- **`MIN_SEGMENT_SEC` calibration** against real meeting audio — post-merge tuning.
- **Pre-stream `413` for oversized clips** — endpoint-level, later (branch-A §9 defers the cap-to-413).

---

## 10. Definition of done

- `read_only` flag on `SessionIdentifier`; mint + `upsert` + `log_usage` all suppressed when set;
  unknowns lock to `voiceprint_id=None` (stable session label), enrolled speakers still resolve
  name + id for display.
- Quality gate implemented: `MIN_SEGMENT_SEC` added (matching block); `_passes_gate` (duration +
  exclude `AMBIGUOUS`); exemplar-append gated; anonymous-mint requires ≥1 qualifying exemplar;
  vote-counting and MATCH-lock left permissive.
- `buffered_batch` buffer-sizing consumed via `getattr` (§6).
- Endpoint: `tag="live"` → read-only; `offline` default unchanged; `_segment_dict`/C2 preserved.
- Tests 1-9 green; existing `test_identify*.py` + `test_diarize_api.py` unchanged and green;
  offline/post default behaviour byte-for-byte unchanged except the additive (default-1.0 s) gate.

## 11. Suggested commit sequence (one per test-gated step group)

1. `test+feat(identify): read_only mode — suppress mint/upsert/log_usage, stable None labels` —
   steps 1-3 + §3a.
2. `test+feat(identify): confidence/min-duration gate on exemplar-append + anon-mint` — steps 4-6 +
   §3b + `config.MIN_SEGMENT_SEC`.
3. `test(identify): regression — defaults leave offline behaviour unchanged` — step 7 (or fold into
   the existing green suites).
4. `test+feat(main): tag="live" selects read_only on /v1/diarize` — step 8 + §3c.
5. `test+feat(identify): size trailing buffer from diarizer.buffered_batch (A↔C bridge)` — step 9 +
   §6 + the in-test batch double (no `mock.py` edit).
