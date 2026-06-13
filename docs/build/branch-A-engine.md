# Branch A — DiariZen production engine (P0)

**Repo:** FPM · **Base:** `main` (post-FF) · **Autonomy:** agent-autonomous (test-first) ·
**Depends on:** Contract **C1** (`StreamingDiarizer`). Touches no consent/identity logic.

## Goal
Make DiariZen the production diarizer behind the existing `/v1/diarize`, **without** deleting diart
(diart stays dormant as the future live engine). No new endpoints. This is the trimmed P0 — engine
swap only; the live/post two-instance split is later.

## Scope (files)
- **NEW** `fpm/diarize/diarizen_engine.py` — port from `eval_harness/harness/diarizen_engine.py`
  as-is (it already implements `StreamingDiarizer` and imports `fpm.diarize.base`).
- `main.py:_default_diarizer_factory` — add a `"diarizen"` branch (lazy import, mirroring the diart
  branch). **Only this function** — keep changes off the diarize endpoint (that's branch C).
- **NEW** `requirements-diarizen.txt` — DiariZen deps (pins torch 2.1.1).
- `config.py` — `DIARIZATION_ENGINE` default may switch to `diarizen` (env-overridable). Do **not**
  touch `MATCH_*` (branch C).
- Do **NOT** touch `fpm/diarize/diart_engine.py` (keep it; lazy-imported, dormant).

## Things to be careful about
- **Venv conflict:** DiariZen pins torch 2.1.1, diart pins 2.2.2 → must run in a **separate venv**.
  Diart's import stays lazy (inside the factory branch) so it never loads under the diarizen venv.
- **Model weights:** `BUT-FIT/diarizen-wavlm-large-s80-md` downloads on first call → pre-fetch for
  TEE/offline.
- **RAM:** `decode_to_mono` + `finish()` load the whole clip; ~16.6 GB on long AMI. **Cap clip
  length** on the no-GPU box (this is #6's reason to exist later).
- Identity always re-embeds with CAM++ — DiariZen embeddings never enter the store (invariant holds).

## Test-gated steps (write test first → green → atomic commit each)
1. **Engine conforms to C1** — `DiariZenDiarizer` instantiates and exposes `start/feed/finish`
   returning `Segment`. (Unit; mock pipeline or `skip` if model absent.)
2. **Factory selection** — `FPM_DIARIZER=diarizen` → `_default_diarizer_factory()` returns a
   `DiariZenDiarizer`; `diart` still returns `DiartDiarizer`. (Unit.)
3. **End-to-end `/v1/diarize`** on a short clip with the diarizen engine produces NDJSON segments +
   final transcript matching C2. (Integration; gate on model presence.)
4. **Clip-length cap** — oversized audio is rejected/handled before DiariZen loads it. (Unit.)

## Definition of done
DiariZen is the active engine via config; diart present-but-dormant; `requirements-diarizen.txt`
added; tests 1–4 green; existing diart tests unaffected (lazy import preserved).
