# Eval harness — in-person diarization + transcription

A standalone lab bench to evaluate transcription + diarization on in-person audio and A/B the
knobs: **vocab on/off**, **diarizer engine** (diart now; DiariZen pluggable later), **window size**
(diart 5s vs a 2-min window). It is **not** the production Recato↔Conclave flow — it deliberately
collapses Whisper + FPM diarize into one process so you can test fast.

Uses the **exact Recato Whisper** (`large-v3-turbo`, int8; vocab via `initial_prompt`) and reuses
FPM's diart + identify internals. Each experiment is a self-describing folder.

## Your workflow (you only upload audio)
1. Pick / create an experiment folder under `experiments/<name>/` (the assistant makes `gold.json`
   + `conversation.md` + `config.yaml`).
2. Drop your recording as `experiments/<name>/audio.wav`.
3. Run it (the harness needs the eval venv with faster-whisper + diart):
   ```
   cd FPM
   HF_TOKEN=$(cat ~/.cache/huggingface/token) PYTHONPATH=$(pwd) \
     /tmp/diart-venv/bin/python -m eval_harness.run experiments/<name>
   ```
4. Read `experiments/<name>/results/result.json` (metrics) + `transcript.txt` (attributed output).

`large-v3-turbo` is already cached locally (same repo Recato resolves it to). The vocab compare
(`asr.vocab_compare: true`) runs Whisper twice (on + off) — slower on CPU but gives the WER delta.

## Compare engines + record UI
- **diart vs DiariZen, side-by-side:** `… -m eval_harness.compare experiments/<name>` runs the same
  audio through both venvs (subprocess — they can't share one) and writes `results/compare.json` +
  a table of WER / speaker-accuracy / RTF / peak-RAM / speakers, with the DiariZen-over-diart deltas.
  Needs both `/tmp/diart-venv` and `/tmp/diarizen-venv` (see `requirements-diarizen.txt`).
- **One-button record:** `… -m eval_harness.server` → open `http://localhost:8090/record/<name>`,
  hit Record → Save (writes `experiments/<name>/audio.wav`) → Run (diart) → metrics panel. No live
  transcription — it records then batches.

## Window-comparison example
`experiments/diart-2min-window/` reuses the eval-conversation audio + gold and only changes
`diarizer.window_sec: 120` — drop the recording once at `experiments/eval-conversation/audio.wav`,
then run both to compare a 2-min window vs the 5s baseline (same audio).

## What an experiment folder holds
```
experiments/<name>/
  audio.wav        ← you drop this (gitignored)
  gold.json        canonical ground truth the scorer compares against: {"turns":[{speaker,text}]}
  conversation.md  the same script, human-readable (read it aloud to record)
  config.yaml      model + methodology + diarizer engine/window + vocab + mode
  results/         run output (gitignored): result.json + transcript.txt;
                   results/<engine>/ for compare, results/compare.json for the diff
```
Gold is `gold.json` (a legacy `gold.txt` of `A: …` lines is still accepted).

## config.yaml knobs
- `mode`: `offline` (batch a file → metrics) | `realtime` (live screen + per-chunk latency).
- `asr.vocab` + `asr.vocab_compare`: the vocab list (→ `initial_prompt`); compare on vs off.
- `diarizer.engine`: `diart` | `diarizen` (later); `diarizer.window_sec` / `step_sec` — the window
  to test (e.g. 5 vs 120).

## Metrics (results/result.json)
audio length, latency + RTF, mode, **WER** (vocab-on vs vocab-off), **speaker accuracy + DER**
(vs the gold speaker turns), and the full config (so the run is self-describing).

See the repo plan for the commit-by-commit build.
