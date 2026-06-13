# Eval harness — test plan + rollback points

Branch: **`eval-inperson-diarization`** (off `dev`). Everything here is eval-only; `dev`/production
FPM is untouched. Each commit is a clean rollback point.

## How to run the tests
The harness needs the eval venv (faster-whisper + diart). Diarizen-engine tests need the separate
diarizen venv.
```
cd FPM
# diart + whisper + scoring (most tests)
HF_TOKEN=$(cat ~/.cache/huggingface/token) PYTHONPATH=$(pwd) \
  /tmp/diart-venv/bin/python -m pytest -q -p no:warnings eval_harness/tests/

# DiariZen engine tests (C9+) — separate venv
HF_TOKEN=$(cat ~/.cache/huggingface/token) PYTHONPATH=$(pwd) \
  /tmp/diarizen-venv/bin/python -m pytest -q -p no:warnings eval_harness/tests/test_diarizen.py
```
Tests gated on a missing model/venv/HF_TOKEN **skip** (not fail), so a partial setup still passes.

## Commits & what each is verified by

| Commit | Scope | Verified by |
|---|---|---|
| `71b78a4` C1 | scaffold + ExperimentConfig + example | `test_config.py` (4) |
| `fb5eae6` C2 | WhisperASR (exact Recato params) | `test_asr.py` (3, model='tiny') |
| `7b84824` C3 | diarizer factory + configurable window | `test_diarize.py` (4) |
| `753b6df` C4 | timestamp merge + offline pipeline | `test_merge.py` (4) |
| `c2539c3` C5 | scoring (WER + speaker) + metrics | `test_scoring.py` (6) |
| `9604c90` C6 | CLI runner + end-to-end | `test_run.py` (1, e2e) |
| `f0ccccb` C8 | window-comparison example + README | config load check |
| **`f0ccccb` = offline-harness known-good HEAD (22 tests green)** | | |
| `c0ccd22` C9 | real DiariZen engine + pinned venv | `test_diarizen.py` (diarizen-venv, real run ✓) |
| C10 *(next)* | one-command compare (diart vs DiariZen) | `test_compare.py` (synthetic) |
| C11 | record→save→run web UI | `test_server.py` (TestClient) |

## Rollback
- Inspect: `git log --oneline dev..HEAD`
- **Undo just the last commit, keep working tree:** `git reset --soft HEAD~1`
- **Hard-reset to a known-good checkpoint** (discards later commits + changes):
  `git reset --hard <hash>` — e.g. back to the offline harness: `git reset --hard f0ccccb`
- **Revert a specific commit (safe, keeps history):** `git revert <hash>`
- The whole feature lives on this branch + in `eval_harness/`; to drop everything, just don't merge
  the branch (or `git branch -D eval-inperson-diarization` after switching off it). `dev` is unaffected.
- Venvs (`/tmp/diart-venv`, `/tmp/diarizen-venv`) and recordings/results are **not committed**
  (gitignored / /tmp), so rollback never has to touch them.
