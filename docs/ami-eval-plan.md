# AMI evaluation + chunk-size elbow sweep — plan

Goal: an **automated, repeatable** evaluation on **AMI** that (a) confirms we reproduce published
numbers (so our protocol is trustworthy), (b) finds the **chunk-size elbow** for CPU streaming
DiariZen, and (c) compares us against **diart** and **full-batch DiariZen** on DER + latency + RAM.
Everything CPU (prod is CPU-bound; GPU = local eval only). Status: **PLAN — awaiting go.**

---

## 0. Decisions already made (override if you disagree)

- **Condition = AMI-SDM** (Single Distant Microphone). This is the right analogue of our single-mic
  fallback — one far-field mic, the hard case. (IHM = per-headset = the "perfect channel" case, not
  our problem; MDM = array, future.)
- **Scoring protocol = match DiariZen's published setup** so we can *reproduce* their AMI-SDM DER as a
  correctness anchor: **DER, no collar, overlap included** (the modern hard setting), via
  `pyannote.metrics` (cross-checked with `dscore`). If we don't land near their ~13–14% with batch
  DiariZen, the *protocol* is wrong and we fix it before trusting anything.
- **DER-first.** The broad sweep scores **DER only** (pure diarization, no Whisper — much faster).
  **cpWER** (`meeteval`, needs Whisper) runs only on the **finalist** configs + baselines.
- **Latency + peak RAM** captured per run, each run in its **own subprocess** for an isolated peak-RSS.

## What needs your input (gates the run)
1. **Do you already have AMI downloaded**, or should Phase 0 fetch it? (SDM audio + RTTM refs + the
   official test/dev split; sizeable download — see Phase 0.)
2. **Overnight compute budget** — how many hours can it run? Sets the subset size (see §6 estimates).
3. OK to **build first** (AMI prep + scorers + chunked runner) before the sweep? That's several hours
   of building tonight, then the sweep runs overnight.

---

## 1. The phases

### Phase 0 — AMI data + protocol (prereq, gating)
- Fetch **AMI-SDM** audio + reference RTTMs + the official **test** (and **dev**) split (pyannote AMI
  recipe / DiariZen's `prepare_data` scripts both do this). Flag the download size/time.
- Lock the scoring config (collar=0, overlap=on) and a tiny "protocol check" meeting to validate the
  scorer end-to-end before the big run.

### Phase 1 — metric harness (build)
- `der_score(ref_rttm, hyp_rttm)` → DER + breakdown (miss / false-alarm / **speaker-confusion**) + JER,
  via `pyannote.metrics`, optimal speaker mapping.
- `cpwer_score(ref, hyp)` → via `meeteval` (finalists only).
- `run_meta` → RTF, wall-clock, **peak RSS** (subprocess `getrusage`), #speakers detected.
- Per-run output = one JSON row: `{system, config, meeting, DER, miss, FA, confusion, JER, RTF,
  peak_rss_mb, n_spk}`. All rows → one `results.jsonl` the plotter reads.

### Phase 2 — baseline reproduction (the "are we close to current implementations" anchor)
Run on the subset, score DER:
- **DiariZen full-batch** → must reproduce **~published AMI-SDM DER (≈13–14%)**. ← correctness gate.
- **diart** (default streaming) → reproduce its ballpark (our prior bake-off ~33%).
- If batch DiariZen ≠ published, STOP and fix protocol (collar/overlap/refs) before sweeping.

### Phase 3 — STAGED parameter sweep (Experiment 1 — the overnight core)
chunk_size is **not** the only knob — and probably not the biggest accuracy lever. The parameters that
target the failures we actually observed (4-vs-3 **over-segmentation**, **short-turn loss**, missed
speech) are DiariZen-internal + preprocessing. So sweep **one factor at a time around a moving
baseline**, ordered by impact (full grid = thousands of runs = infeasible on CPU).

**Full knob taxonomy** (defaults from the loaded DiariZen config):

| Stage | Param (default) | Affects | Priority |
|---|---|---|---|
| Clustering | **ahc_threshold (0.6)** | over/under-seg | **HIGH** — the 4-vs-3 over-seg we saw |
| Segmentation | **apply_median_filtering (on)** | short turns | **HIGH** — likely smooths away rapid-exchange turns |
| Preprocess | **VAD on/off + threshold** | missed speech | **HIGH** — direct DER term |
| Streaming | **chunk_size** | latency/RAM + acc | MED — the elbow (streaming feasibility) |
| Preprocess | denoise on/off + strength | embedding quality | MED |
| Segmentation | segmentation_step (0.1) | boundaries vs speed | MED |
| Streaming | overlap/hop | cross-chunk link | MED (Exp 2) |
| Clustering | Fa 0.07 / Fb 0.8 / lda_dim 128 / max_speakers 20 | granularity | LOW (hold at default) |
| Linker | matcher 0.45/0.35/0.10, embed window, min-speech floor | cross-chunk confusion | MED (Exp 2) |

**Note:** `seg_duration=16` is baked into DiariZen *training* → don't push it far. `ahc_threshold`,
`median_filtering`, `segmentation_step` are **inference-time** → safe to sweep. VAD/denoise = safe front-end.

**Staged OFAT (each stage holds others at best-so-far):**
- **Stage A — chunk_size elbow** (at defaults): {10,15,30,45,60,90,120, full}. Per-chunk DER w/ optimal
  local mapping (no linker yet) + RTF + **peak RAM**. → the streaming operating point.
- **Stage B — accuracy knobs** (at the chosen chunk): `ahc_threshold ∈ {0.4,0.5,0.6,0.7,0.8}`,
  `median_filtering ∈ {on,off}`, `VAD ∈ {off, on@2-3 thresholds}`. ← attacks the observed over-seg +
  short-turn + missed-speech failures.
- **Stage C** → Exp 2 (linker thresholds + overlap, Phase 4).
- Baselines (diart, DiariZen-batch) scored alongside every stage.
- **Adaptive:** zoom where a factor moves DER (e.g., plateau by 60s → refine 45–90s; ahc sweet spot at
  0.5 → refine 0.45–0.55).

### Phase 4 — whole-meeting chunked + cross-chunk linking (Experiment 2 — needs a build)
The *real* system: chunks linked into a consistent whole-meeting labeling.
- **Build:** overlapping-window runner + **FPM matcher as the cross-chunk linker** (MATCH → reuse id,
  UNKNOWN → new speaker; `enroll` accumulates the centroid). Overlap region links on shared speech.
- Configs: chunk = {elbow ± 1}, overlap = {0, 50%}, linker threshold = {default}.
- Score **whole-meeting DER** (now includes cross-chunk confusion — the realistic number) vs
  **diart** (streaming) and **DiariZen-batch** (ceiling).
- This is where "diart-style streaming DiariZen" actually gets measured end-to-end.

### Phase 5 — cpWER on finalists + analysis + graphs
- Run **Whisper + merge + cpWER** on the best 2–3 configs + diart + DiariZen-batch (finalists only).
- **Graphs** (matplotlib, like the latency plot):
  1. **Elbow:** DER vs chunk_size, with batch + diart reference lines.
  2. **Cost:** RTF and peak-RAM vs chunk_size.
  3. **Pareto:** DER vs RTF (which config is the best accuracy/latency trade).
  4. **Summary table:** every system/config — DER, confusion, RTF, RAM, cpWER (finalists) — vs published.
- A written readout: where's the elbow, how close are we to DiariZen-batch and to published SOTA, and
  the recommended operating config.

---

## 2. What exists vs what I build
- **Already built + committed (reuse, don't rebuild):**
  - `evaluation/` package — **`der.py` = real `pyannote.metrics` DER scorer**, `der_eval.py`, `rttm.py`,
    `harness.py`, `id_eval.py`. The DER metric + RTTM handling are done.
  - **A validated diart-vs-DiariZen AMI bake-off** (`docs/bakeoff-offline.md`, `docs/der-eval.md`):
    DiariZen 22.8% strict / **~14% lenient**, diart 32.9% strict, on IS1009a + ES2004a (SDM). The
    **lenient ~14% already matches DiariZen's published ~13–14%** → Phase 0/2 protocol + reproduction
    are **largely de-risked** on 2 meetings. AMI audio is in `eval_data/` (gitignored).
  - DiariZen batch engine, diart engine, Whisper ASR, merge, peak-RSS pattern.
- **Genuinely new to build:** `meeteval` **cpWER** scorer (der.py is DER-only); **more AMI meetings**
  (extend beyond IS1009a/ES2004a to the full test split); the **chunked-streaming DiariZen runner**
  (per-chunk for Exp 1; +FPM linker for Exp 2); the **staged-sweep orchestrator** (subprocess-per-run,
  resumable, `results.jsonl`); the **plotter**.

> **Net effect:** Phases 1–2 are mostly done. The real work is the **chunked runner + the staged
> sweep + cpWER + more meetings + plots** — and the bake-off gives us a known-good anchor to extend from.

## 3. Reusability / robustness for an overnight run
- **Resumable:** each (system, config, meeting) run is independent; skip already-done rows → safe to
  stop/restart while you sleep.
- **Isolated:** subprocess per run → one crash/OOM doesn't kill the sweep; peak-RAM is clean.
- **Logged:** progress + per-run timing streamed to a log you can glance at in the morning.

## 4. Compute budget (CPU, rough)
- DiariZen ≈ RTF 1.3 → a 30-min AMI meeting ≈ ~40 min CPU. diart ≈ RTF 0.6 ≈ ~18 min.
- Full AMI-SDM test (~16 meetings) × ~15 configs would be **days** — too much. So:
  - **Broad sweep on a representative subset** (propose **3–4 meetings**, or 10-min excerpts) → ~7–12 h
    overnight for the chunk matrix + baselines.
  - **Validate finalists on the full test set** in a second, shorter run.
- I'll print the exact estimate after Phase 0 (once meeting count/lengths are known) and scope to your
  hour budget before launching.

## 5. Honest caveats
- **Exp 1 is an upper bound** (per-chunk optimal mapping assumes perfect linking) — it isolates the
  elbow; Exp 2's whole-meeting DER (with the real linker) is the number that ships.
- On a single distant mic, **the rapid-exchange wall still applies** — this sweep optimizes the
  *fallback*; the step-change is the multi-phone path (research_doc_diarization.md §5).
- cpWER needs Whisper per meeting → that's why it's finalists-only, not in the broad sweep.
