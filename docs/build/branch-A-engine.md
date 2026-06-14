# Branch A — DiariZen production engine (P0)

**Repo:** FPM · **Base:** `main` (post-FF) · **Working branch:** `branch-A-engine` ·
**Autonomy:** agent-autonomous (test-first) · **Depends on:** Contract **C1** (`StreamingDiarizer`,
`fpm/diarize/base.py`). Touches no consent/identity logic.

> This is the enriched build plan. The frozen architecture lives in `ARCHITECTURE.md` — deviations
> from it are recorded **here** (§5 Decisions, §6 the A↔C buffer coupling), not by editing it.

## Goal
Make DiariZen the production diarizer available behind the existing `/v1/diarize`, **without**
deleting diart (diart stays dormant as the future live engine). No new endpoints. This is the
trimmed P0 — **engine swap only**; the live/post two-instance split and Docker topology are later.

---

## 1. Current state (verified against the tree on `main`, 2026-06-13)

What already exists and is **reused as-is** (do not rebuild — ARCHITECTURE §8):
- `fpm/diarize/base.py` — the **C1 contract**: `Segment{start,end,local_speaker}` + abstract
  `StreamingDiarizer{start, feed, finish}`. Frozen.
- `fpm/diarize/diart_engine.py` — the **pattern to mirror** (lazy torch import, `offline=` flag that
  sets `HF_HUB_OFFLINE=1`, span stitching). **Do not touch it** — it must stay lazy-imported/dormant.
- `fpm/diarize/mock.py` + `tests/test_diarize_base.py` — the deterministic contract harness.
- `main.py:_default_diarizer_factory` (L49–56) — currently only the `diart` branch + a 503 fallback.
- `main.py:diarize_endpoint` (L216–266) — streams `_segment_dict` NDJSON + a final `transcript`
  line (**C2**). **Branch C territory — A must not touch it.**
- `main.py:_segment_dict` (L59–68) — the C2 per-line shape. Do not touch.
- `config.py:DIARIZATION_ENGINE` (L62) — `os.environ.get("FPM_DIARIZER", "diart")`, comment
  `# diart | onnx (E.3)`.
- `fpm/audio.py:decode_to_mono` — ffmpeg → 16 kHz mono float32 (used by the endpoint, not the engine).
- `scripts/prefetch_diart_models.py` + `scripts/fetch_models.sh` — the **prefetch pattern to mirror**.
- `requirements-diart.txt` — the **separate-venv requirements pattern to mirror** (pins + recipe).
- `pyproject.toml` — pytest config: `testpaths=["tests"]`, `pythonpath=["."]`, `addopts="-q"`.

**Where the port source lives.** `eval_harness/harness/diarizen_engine.py` does **not** exist on
`main` — it lives only on branch `eval2-experiments`, together with `eval_harness/requirements-
diarizen.txt` and `eval_harness/tests/test_diarizen.py`. Per the branch rule
(ARCHITECTURE §10: *"`eval-inperson-diarization` never merges to main — port engine files instead"*),
we **copy the file content in**, we do not merge the branch. Read the source with:
```
git show eval2-experiments:eval_harness/harness/diarizen_engine.py
git show eval2-experiments:eval_harness/requirements-diarizen.txt
git show eval2-experiments:eval_harness/tests/test_diarizen.py
```
The source is already clean: it imports `from fpm.diarize.base import Segment, StreamingDiarizer`,
implements the contract, buffers in `feed()` (returns `[]`), and runs the whole clip in `finish()`.

---

## 2. Scope (files)

| File | Action | Notes |
|---|---|---|
| `fpm/diarize/diarizen_engine.py` | **NEW** | Port from `eval2-experiments`; apply the §5 adaptations. |
| `requirements-diarizen.txt` | **NEW** | Trimmed runtime pin-set + build recipe (§7). |
| `scripts/prefetch_diarizen_model.py` | **NEW** | Mirror `prefetch_diart_models.py` (§7). |
| `tests/test_diarizen_engine.py` | **NEW** | Steps 1, 3, 4 (§4). |
| `tests/test_diarizer_factory.py` | **NEW** | Step 2 — factory selection (§4). |
| `main.py:_default_diarizer_factory` | **EDIT (only this fn)** | Add the `"diarizen"` branch + graceful 503. |
| `config.py` | **EDIT (engine line + 1 new const)** | Comment update; `DIARIZEN_MAX_CLIP_SEC`. **Do not touch `MATCH_*`.** |
| `fpm/diarize/base.py` | **EDIT (additive, see §6)** | Optional capability hint for the batch buffer — **needs contract-owner sign-off.** |

**Do NOT touch:** `diart_engine.py`, `diarize_endpoint`, `_segment_dict`, `fpm/identify.py`,
`record_routes.py` (Conclave), `MATCH_*` config, `Dockerfile.cpu`.

---

## 3. The port — exact source→target delta

Copy `DiariZenDiarizer` verbatim into `fpm/diarize/diarizen_engine.py`, then apply:

1. **Docstring** — drop "EVAL ONLY"; state it is the production post engine behind `/v1/diarize`,
   batch-at-finish, runs in its own torch-2.1.1 venv.
2. **Import path** — unchanged (`from fpm.diarize.base import ...` already correct once relocated).
3. **Move model load `start()` → `finish()`** (Decision D1, §5). `start()` resets the buffer only;
   `finish()` lazy-loads the pipeline right before use. This makes steps 1 & 4 testable in the
   torch-free core venv (no `diarizen` import is forced until a real diarize at `finish()`), and lets
   the clip cap reject *before* any model is loaded.
4. **Offline mode** (Decision D2) — add an `offline: bool = True` ctor flag; in the loader set
   `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` via `os.environ.setdefault(...)` before
   `from_pretrained`, mirroring `diart_engine._build_pipeline`. No runtime egress.
5. **Contract-parity guards** (Decision D3) — `feed()` raises `RuntimeError` if called before
   `start()`; validate `sample_rate == self._sample_rate` (mirrors diart). Keeps engines swap-clean.
6. **Clip-length cap** (Decision D4) — track accumulated samples in `feed()`; raise
   `ClipTooLongError(RuntimeError)` the moment the buffer would exceed `config.DIARIZEN_MAX_CLIP_SEC`,
   so we never accumulate the ~16.6 GB clip that crashes the no-GPU box.
7. **Capability hint** (§6) — set the batch/full-buffer marker the identify layer needs.

Everything else (WAV staging, `itertracks` → `Segment`, sort) is copied unchanged.

---

## 4. Test-gated steps (write test first → green → atomic commit each)

Each step is one commit (test + impl together — ARCHITECTURE §7). Venv-gating matters: the **core
`.venv` is intentionally torch-free**, so engine/model tests must `pytest.importorskip("diarizen")`
or be designed to run without it.

**Step 1 — Engine conforms to C1** *(unit; runs in the core venv — no torch needed).*
`tests/test_diarizen_engine.py`:
- `isinstance(DiariZenDiarizer(), StreamingDiarizer)` and `start/feed/finish` exist.
- After `start("ws")`, every `feed(chunk)` returns `[]` (batch engine — no incremental output).
- `feed()` before `start()` raises `RuntimeError`; wrong `sample_rate` raises `ValueError`.
- Constructing the engine and feeding does **not** import torch/diarizen (asserts the lazy seam:
  load only happens in `finish()`). This is why D1 moves the load site.

**Step 2 — Factory selection** *(unit; core venv).* `tests/test_diarizer_factory.py`:
- `monkeypatch.setattr(config, "DIARIZATION_ENGINE", "diarizen")` → `_default_diarizer_factory()`
  returns a `DiariZenDiarizer` (the `diarizen_engine` module imports only numpy+base, so this works
  torch-free).
- `"diart"` → returns `DiartDiarizer` (construct-only; diart's torch import is lazy in
  `_build_pipeline`, not `__init__`, so no skip needed).
- unknown engine → `HTTPException(503)`.
- missing-venv: if the lazy `import` raises `ImportError`, the factory returns a clean
  `HTTPException(503)` not a 500 (Decision D5).

**Step 3 — End-to-end `/v1/diarize` (shape)** *(integration; gated:
`pytest.importorskip("diarizen")` + `MODEL`/`HF_TOKEN` present → diarizen venv only).*
- Mirror `tests/test_diarize_api.py`, but `diarizer_factory = lambda: DiariZenDiarizer()`.
- Assert C2 holds: NDJSON segments stream, each has `{start,end,voiceprint_id,name,decision,
  confidence,local_speaker}`, `end>start`, and a final `{"type":"transcript","segments":[...]}`.
- **Identity-coverage caveat (see §6):** on a clip longer than `BUFFER_SEC` (15 s), the *shape* is
  valid but most segments resolve `PENDING` until the §6 buffer fix lands. So Branch A's step 3
  asserts **shape/contract only**; the *identity-correctness* e2e (enrolled speaker recognized across
  a long clip) is owned by the **A+C joint integration gate**, not by Branch A alone.

**Step 4 — Clip-length cap** *(unit; core venv — no model).* `tests/test_diarizen_engine.py`:
- `DiariZenDiarizer(max_clip_sec=small)`; `start()`; feed chunks past the cap →
  raises `ClipTooLongError` **before** any model load (proves D1+D4: cap rejects pre-load).
- Under the cap, feeding does not raise.

**Regression (no new test, must stay green):** `tests/test_diart_engine.py`,
`tests/test_diarize_base.py`, `tests/test_diarize_api.py` (MockDiarizer path) all unchanged.

---

## 5. Decisions (deviations from the as-is port — recorded here, not in ARCHITECTURE)

- **D1 — load the model in `finish()`, not `start()`.** The as-is source loads in `start()`. Moving
  it makes the cap check and contract conformance unit-testable without the diarizen venv, and avoids
  loading a multi-GB model for a clip we're about to reject. Behaviour for real runs is identical
  (batch engine — nothing happens between `start` and `finish` but buffering).
- **D2 — `offline=True` by default**, mirroring diart, so the production engine never egresses at
  runtime (TEE requirement). Prefetch warms the cache (§7).
- **D3 — add feed-before-start + sample-rate guards** for parity with diart/mock (and the existing
  `test_feed_before_start_errors` expectation across engines).
- **D4 — cap lives in the engine, not the endpoint.** The endpoint is Branch C's; A may not touch it.
  So `DiariZenDiarizer` enforces `DIARIZEN_MAX_CLIP_SEC` in `feed()`. **Limitation:** because the
  endpoint streams `200` before `finish()`, an engine-raised cap can't become a clean pre-stream
  `413` — that belongs to the endpoint (Branch C / later). A provides the engine guard + the config
  constant the endpoint will consume.
- **D5 — factory returns `503` on `ImportError`.** Selecting `diarizen` in a venv without it
  installed should be a clean 503 ("engine not available"), not a 500. Scoped to
  `_default_diarizer_factory` (allowed).
- **D6 — config default stays `diart` (RECOMMENDED); select `diarizen` via `FPM_DIARIZER=diarizen`
  in its instance.** ARCHITECTURE/branch-A says the default *"may switch."* Flipping the global
  default while the core venv is torch-free would make a bare `/v1/diarize` call `ImportError`/503 in
  the default deployment. Since the two-instance split is explicitly deferred, keep the default and
  document selection via env. The config comment becomes `# diart | diarizen | onnx (E.3)`.
  *(Open — confirm with owner; the only behaviour-visible fork in this branch.)*

---

## 6. The A↔C coupling: bounded buffer vs batch engine (THE key risk)

**Finding (verified, `fpm/identify.py`).** `SessionIdentifier` keeps a **bounded 15 s trailing audio
buffer** (`BUFFER_SEC = 15.0`, L47; stale front dropped in `_append`, L130–136) and re-embeds each
segment by slicing it (`_slice`, L138–144). This is correct for **diart**, which emits segments
*incrementally* while their audio is still in-window. **DiariZen returns `[]` from every `feed()` and
emits ALL segments at `finish()`** — by then the buffer holds only the last 15 s, so `_slice`
returns `None` for every earlier segment → `emb=None` → `PENDING`, no identity. Net: a naïve engine
swap identifies only the **final 15 s** of any clip.

This is exactly the *"A+C integration coupling"* the architecture flags (§6 risk 2). The fix is in
`fpm/identify.py` — **Branch C's file** (branch-A must not touch it; branch-C owns `SessionIdentifier`).

**Resolution (zero shared-file edit — chosen to keep A and C textually disjoint, since C is being
built in parallel right now):**
1. **Branch A** declares the capability as a **class attribute on `DiariZenDiarizer` itself**, in its
   own NEW file `fpm/diarize/diarizen_engine.py` — e.g. `buffered_batch = True`. **A does NOT edit
   `base.py`.** No default is added to the base class, so the frozen C1 contract file is untouched and
   A introduces zero edits to any file C might open.
2. **Branch C** (or the A+C joint integration step) reads it **defensively** in `identify.py`:
   `getattr(self._diarizer, "buffered_batch", False)` → when True, size `_max_buf` to retain the
   **full** clip (unbounded) so `finish()` can re-embed every segment. The `getattr` default means
   C's code is correct *whether or not A has merged yet* (diart/mock simply lack the attr → `False`).
   ~3-line change, entirely inside C's own file.
3. **The identity-correctness e2e test** (enrolled speaker recognized across a >15 s clip with the
   DiariZen engine) is the **A+C joint integration gate** (ARCHITECTURE §6 merge order:
   *B → A & C → integration-test A+C jointly*).

Because A's flag lives in its own new file and C reads it via `getattr` in C's own file, **there is no
file both branches edit for this coupling** — no textual merge conflict, and no ordering dependency
between the merges.

**Required coordination (logical, not textual):** the buffer-sizing in step 2 is **not** in the
current `branch-C-identify-gate.md` scope. Add a one-line item there —
*"size the trailing buffer via `getattr(diarizer, 'buffered_batch', False)` → unbounded for batch
engines"* — so the consumption isn't dropped. Without it there's still no merge conflict, but DiariZen
identity silently breaks past 15 s. **Owner of Branch C: please confirm you'll pick this up, or
explicitly defer it to the joint gate.**

*(Note: the long-term answer is ARCHITECTURE §5 "#6 DiariZen windowing for RAM" — windowed decode
with `SessionIdentifier` as cross-window stitcher. That's time-permitting/later; the `buffered_batch`
hint is the minimal P0-compatible bridge.)*

---

## 6.1 Contract-compliance audit (C1 / C2 — proof A cannot break Branch C)

Branch C depends on **C2** (preserve the `/v1/diarize` shape) and consumes **C1** (drives the engine
via `SessionIdentifier`). Verified against the current tree:

**C1 — `StreamingDiarizer` engine seam (`fpm/diarize/base.py`). A keeps it intact:**
- `DiariZenDiarizer` implements the exact signatures: `start(workspace_id)->None`,
  `feed(chunk, sample_rate=16000)->list[Segment]`, `finish()->list[Segment]`. Unchanged.
- It emits **only** `Segment{start,end,local_speaker}` — never embeddings/ids/text (the port maps
  `itertracks` straight to `Segment`). Invariant held.
- `feed()` returning `[]` every time is **explicitly contract-legal** — C1: *"feed() … segments
  finalized by this chunk (may be empty)."* Batch-at-`finish()` is a permitted shape, not a violation.
- **No generic/parametrized contract test spans engines** (`test_diarize_base.py` tests `MockDiarizer`
  only; the incremental-emission asserts in `test_diart_engine.py` are diart-scoped). So DiariZen's
  batch behaviour cannot fail an existing C1 test, and `Segment` is untouched → those tests stay green.
- The added guards/flags don't alter the contract surface and **don't fire under C's calls**:
  `SessionIdentifier` always calls `start()` then `feed(block, sample_rate)` with
  `sample_rate = config.TARGET_SAMPLE_RATE = 16000` (`main.py:240`,`252`), and `DiariZenDiarizer`'s
  default is `16000` → the D3 sample-rate guard never trips; the D4 clip cap is an offline guard and
  C's live path uses diart regardless; `buffered_batch` is an additive read-only attribute on A's own
  subclass. **No base-class default added → `base.py` byte-identical → zero risk to the frozen C1.**

**C2 — `/v1/diarize` NDJSON shape (`main.py:_segment_dict` L59 + transcript line L263). A never edits
it:** A touches only `_default_diarizer_factory` (L49–56). `_segment_dict`, `diarize_endpoint`, and
the final `transcript` line are untouched, so C2 is preserved **by construction** — DiariZen's
`Segment`s flow through the unchanged endpoint and serialize identically. `test_diarize_api.py`
(C2 regression) stays green.

**C3 / C4** — Conclave projection + propose/confirm; not engine concerns. A goes nowhere near them.

**Conclusion.** A's engine is fully C1-compliant and leaves C2 byte-identical, so it cannot break the
contracts C relies on. The §6 buffer item is **not** a contract repair — C1 permits batch `feed()`;
the 15 s buffer is an *implementation assumption inside `SessionIdentifier`*, and adapting it is C's
optional enhancement (read via `getattr`), never forced by A.

---

## 7. Venv, requirements & prefetch

**`requirements-diarizen.txt`** — port the eval2 pin-set, **trimmed to runtime**: keep the
torch-2.1.1 legacy block, the pyannote stack the vendored fork hard-pins, and DiariZen runtime deps
(einops/librosa/soundfile/pyyaml/h5py/joblib/pandas/scipy/tabulate/toml/torchinfo/tqdm). **Drop the
eval-only ASR+scoring block** (`faster-whisper`, `ctranslate2`, `jiwer`) — ASR is Conclave's job, not
FPM's. **Keep `pytest` + `soundfile`** for the step-3 gated test. Keep the build-recipe comment
(DiariZen is not on PyPI):
```
python3.11 -m venv .venv-diarizen
.venv-diarizen/bin/pip install -r requirements-diarizen.txt
git clone https://github.com/BUTSpeechFIT/DiariZen /tmp/DiariZen && cd /tmp/DiariZen
.venv-diarizen/bin/pip install -e ./pyannote-audio --no-deps   # vendored fork
.venv-diarizen/bin/pip install -e .          --no-deps          # diarizen
.venv-diarizen/bin/pip install "numpy==1.26.4"                  # re-pin (a transitive dep bumps it)
```
Import path: `from diarizen.pipelines.inference import DiariZenPipeline`. Model:
`BUT-FIT/diarizen-wavlm-large-s80-md` (HF, first-use download → gated on `HF_TOKEN`).

**`scripts/prefetch_diarizen_model.py`** — mirror `prefetch_diart_models.py`: require an HF token,
build `DiariZenDiarizer(offline=False)` and trigger the `from_pretrained` fetch once to warm the HF
cache so runtime (`offline=True`) loads with no egress. Run with the diarizen venv.

**Venv conflict** (ARCHITECTURE §5): DiariZen torch 2.1.1 ≠ diart 2.2.2 → separate venvs. The factory
keeps **both** engine imports lazy, so neither torch stack loads unless its engine is selected — the
core `.venv` stays torch-free.

---

## 8. Merge-conflict / scope discipline (ARCHITECTURE §6)

**Branch C is in active parallel development → A is scoped to share no file C edits.** C owns
`fpm/identify.py` (entire `SessionIdentifier`) and `main.py:diarize_endpoint`. A touches **neither**.

Per-file verdict (only two files are edited by both branches, both in disjoint regions):

| File | A's edit | C's edit | Verdict |
|---|---|---|---|
| `fpm/identify.py` | — none | whole `SessionIdentifier` | A out entirely → **no conflict** |
| `main.py` | `_default_diarizer_factory` L49–56 | `diarize_endpoint` body L216–266 | ~160 lines apart → **auto-merge** |
| `config.py` | engine line L62 + `DIARIZEN_MAX_CLIP_SEC` | reads `MATCH_ACCEPT` + adds `MIN_SEGMENT_SEC` | different sections → **auto-merge** |
| `fpm/diarize/base.py` | **— none** (hint moved to engine file, §6) | — none | **no conflict** |

- **A must not touch:** `identify.py`, `diarize_endpoint`, `_segment_dict`, `base.py`, `MATCH_*`,
  `record_routes.py` (Conclave).
- **`config.py` hygiene:** put `DIARIZEN_MAX_CLIP_SEC` in the **diarizer-engine section** (right after
  L62), *not* near the `MATCH_*` block where C adds `MIN_SEGMENT_SEC` — keeps the two new-constant
  hunks far apart so git never even has to think about it. A must not edit `MATCH_*`.
- **New files** (engine, requirements, prefetch, 2 test files) — zero conflict surface.
- **The §6 buffer coupling is now textually conflict-free** (A's flag in its own engine file; C reads
  via `getattr` in its own file) — the only remaining coordination is logical: C must add the
  buffer-sizing line (or defer it to the joint gate).
- **A stays out of** `record_routes.py` (Conclave/B), `diarize_endpoint` (C), `identify.py` (C).
- **Merge order** (§6): **B → A & C → joint A+C integration test → P4→P5.**

---

## 9. Out of scope / deferred (explicit)

- **Second FPM instance / Docker topology** — the live+post two-instance split (ARCHITECTURE §5 P0
  "second FPM instance"). Trimmed P0 is engine-swap only; `Dockerfile.cpu` untouched.
- **Pre-stream `413` for oversized clips** — needs the endpoint (Branch C / later); A ships the
  engine-level cap + config constant only (D4).
- **DiariZen windowing for RAM (#6)** and **TS-VAD (#4)** — ARCHITECTURE §5 time-permitting.
- **Full-clip identity correctness through `/v1/diarize`** — depends on the §6 buffer fix (Branch C)
  and is verified at the joint A+C gate, not within Branch A.

---

## 10. Definition of done

- `fpm/diarize/diarizen_engine.py` ported with D1–D4, D6 applied; conforms to C1.
- `_default_diarizer_factory` selects `diarizen` (lazy) and `diart` still returns `DiartDiarizer`;
  unknown/missing-venv → clean 503 (D5).
- `requirements-diarizen.txt` (trimmed runtime set + recipe) and `scripts/prefetch_diarizen_model.py`
  added.
- `config.py`: engine comment updated; `DIARIZEN_MAX_CLIP_SEC` added; `MATCH_*` untouched.
- Tests 1, 2, 4 green in the **core venv**; test 3 (shape) green in the **diarizen venv**; existing
  diart/mock/api tests unaffected (lazy imports preserved).
- §6 capability hint (`buffered_batch = True`) lives **on `DiariZenDiarizer` in its own engine file**
  — `base.py` untouched. A one-line buffer-sizing item is added to `branch-C-identify-gate.md` (read
  via `getattr`) so the consumption is scheduled, or it's explicitly deferred to the joint gate.
- diart present-but-dormant; nothing under `diarize_endpoint`/`identify.py` changed.

## 11. Suggested commit sequence (one per test-gated step)

1. `test+feat(diarize): port DiariZen engine behind StreamingDiarizer (C1 conform)` — step 1 + the
   ported `diarizen_engine.py` (D1–D3) + capability hint (§6).
2. `test+feat(main): diarizen branch in _default_diarizer_factory (+503 on missing venv)` — step 2.
3. `test+feat(diarize): clip-length cap + DIARIZEN_MAX_CLIP_SEC` — step 4 (D4).
4. `chore(diarize): requirements-diarizen.txt + prefetch_diarizen_model.py + config comment` — §7.
5. `test(diarize): gated e2e /v1/diarize shape on diarizen engine` — step 3 (shape-only).

*(Identity-correctness e2e + the `SessionIdentifier` buffer consumption land on the A+C joint
integration step, not here.)*
