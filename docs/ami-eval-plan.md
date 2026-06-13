# Diarization evaluation + direction — plan & conversation handoff

Goal: an **automated, repeatable** evaluation on **AMI** that (a) confirms we reproduce published
numbers (so our protocol is trustworthy), (b) finds the **chunk-size elbow** for CPU streaming
DiariZen, and (c) compares us against **diart**, **full-batch DiariZen**, and (new) **GPU Sortformer**
on DER + latency + RAM. Status: **PLAN — paused mid-thread, continuing in a new chat.**

---

## Conversation handoff — full state (2026-06-13)

What this whole session produced + decided, so a fresh chat can pick up:

### Built + committed (branch `eval-inperson-diarization`)
- **Eval harness** C9–C11: real DiariZen engine (`/tmp/diarizen-venv`), one-command `compare`
  (diart vs DiariZen), record→save→run web UI. Self-contained experiment folders
  (`config.yaml` + `gold.json` + `conversation.md`), `tag` field, 3 `initial-testing` scripts.
- **Fingerprint-run tooling** (`eval_harness/fp_*.py`): `fp_split` (cut audio per metadata),
  `fp_latency` (latency/RTF/RAM curve + plot), `fp_transcribe` (Whisper hypothesis), `fp_diarize`
  (who-spoke-when timeline, `--engine diart|diarizen`), `fp_attribute` (merged `[speaker] text`
  transcript), `fp_identify` (enroll→identify persistence).
- **AMI DER harness already existed + committed**: `evaluation/` (`der.py` = pyannote.metrics,
  `rttm.py`, `der_eval.py` diart) + the M2 bake-off (`docs/bakeoff-offline.md`, `der-eval.md`).
- **New this session:** `evaluation/diarizen_eval.py` (batch DiariZen DER anchor),
  `docs/research_doc_diarization.md`, `docs/diarization-experiment-log.md`, this plan.

### Key empirical results
- **diart-vs-DiariZen on AMI (IS1009a, ES2004a, SDM)** — DiariZen **22.7% strict / 15.3% lenient**
  (Run 001 reproduced the committed bake-off 22.8%; harness TRUSTED), diart **32.9% strict**.
  DiariZen over-detects speakers (5 vs 4 ref) — over-segmentation.
- **3-person private recording** (11 min, single mic, gitignored): latency **RTF ~0.6**,
  diarize-dominated (~63%), **bounded RAM ~2.9 GB**, linear in length. Enroll 3-min → identify 7-min:
  **2 MATCH (cos 0.95, 0.88), 1 AMBIGUOUS** (the ~25 s quiet speaker); enroll prints contaminated
  (E0–E1 cos 0.73) by diart's leaky spans. A 1-min enroll only captured **2 of 3 speakers**.

### Decisions / architecture locked
- **Merge is timestamp code** (`eval_harness/harness/merge.py`), not manual; its quality is **capped
  by the diarizer**. Whisper (ASR) ∥ diarizer run in parallel → merge by overlap. FPM fingerprint is a
  **3rd layer (identity)**, downstream of diarization — it can't recover lost short turns.
- **Rapid-exchange / short-turn loss is the fundamental failure** of acoustic diarization on a **mono
  mix** — true of *both* diart and DiariZen; no tool fixes it. Per-participant channels avoid it.
- **Multi-phone (§5 of research doc) = the real fix:** browser-link, phone-per-person, near-field
  capture → stream to TEE → per-stream ASR + **FPM as cross-talk gate** → **transcript-merge** on a
  coarse clock. Device-login = free identity **and** free ground-truth labels. **1-phone vs 2-phone =
  the experiment, where the 2-phone run is its own pseudo-gold.** VoxTerm already prototypes this
  (`external/VoxTerm`, NTP transcript-merge). TDOA/audio-merge = dropped.
- **Clustering grounding** (read the code): diart's `OnlineSpeakerClustering` and FPM's `match.py` are
  the **same family** (cosine + centroid + new-speaker threshold). diart adds **cannot-link** (better
  for cross-chunk *consistency*); FPM adds open-set + name-leak guard (for *identity*). **Neither
  trains its clustering** (DiariZen = VBx, diart = algorithmic) → clustering is swappable without
  retraining. **Streaming-DiariZen is not in the literature** (our gap), but the online-clustering tax
  is **~3–4% DER vs batch** (worse with more speakers).
- **train-on-GPU / infer-on-CPU is valid** (DiariZen already does it). **Sortformer = SOTA streaming,
  123M params (~0.25–0.5 GB), GPU-optimized;** CPU RTF unpublished → estimate **batch-viable,
  live-doubtful.** Literature support for the approach: **TS-VAD** (fingerprint-conditioned diarization,
  CHiME-6 winner), **spatial+embedding fusion** (~34% rel. gain), ad-hoc arrays.
- **GPU TEE was deemed unaffordable** → that's *why* the CPU plan exists. If that flips, **Sortformer
  on a GPU TEE dominates** (SOTA, no training, no hack). This is an **economic** call, not technical.

---

## Open questions (carry into the next chat)
1. **Where's the NVIDIA GPU?** This machine is Apple Silicon (no CUDA). Options raised: **Google Colab**
   (free T4 — recommended for a quick test; run *in a notebook*, VSCode→Colab is hacky), cloud VM
   (Lambda/RunPod/GCP), or the TEE GPU.
2. **Is the product LIVE or BATCH?** Decides everything: live needs real-time RTF (GPU likely); batch
   tolerates slow CPU (Sortformer-on-CPU may then be viable).
3. **Is a confidential GPU TEE affordable for production?** If yes → Sortformer on GPU TEE (Fork A,
   dominant). If no → CPU path (Fork B) or multi-phone (Fork C).
4. **Is Sortformer "leagues better"** than DiariZen-batch (22.7%) / diart (32.9%) on DER? — the test below.
5. **Sortformer CPU RTF** — unmeasured; the ship-on-cheap-CPU question hinges on it.
6. **Single-mic-SOTA vs multi-phone (§5)** — which to prioritize for the product? Multi-phone sidesteps
   the whole diarization-quality problem if app adoption is acceptable.
7. **Chunked-DiariZen worth building?** Only if the ~3–4% online tax beats plain diart by enough — and
   only if GPU-Sortformer isn't simply adopted instead.

## Track G — GPU Sortformer eval (separate, parallel, "ship if leagues better")
A standalone test of **`nvidia/diar_streaming_sortformer_4spk-v2`** (123M params) *as-is* — independent
of the CPU/DiariZen track. Sortformer **diarizes** (Whisper still transcribes; merge as usual).
- **Env:** isolated NeMo install (`nemo_toolkit[asr]`), on an **NVIDIA GPU** (Colab T4 fine).
- **Accuracy (platform-independent — the "leagues better?" answer):** run on AMI (IS1009a, ES2004a) →
  DER strict/lenient, **compare directly to DiariZen 22.7% / diart 32.9%** (reuse `evaluation/der.py`
  + `rttm.py`). Also run on the 3-person recording → merged transcript (with Whisper) for the
  qualitative front-end view, esp. the rapid-exchange spots.
- **Speed:** RTF on the GPU (for live feasibility) + RTF on CPU (the ship-on-CPU-TEE question).
- **Script:** `evaluation/sortformer_eval.py` mirroring `diarizen_eval.py` (drops into the same DER
  harness + experiment log → directly comparable).
- **Decision rule:** if Sortformer DER is *leagues* below DiariZen-batch on AMI + the merged transcript
  is visibly cleaner on rapid exchange → **ship the GPU-Sortformer path**; park the CPU/DiariZen sweep
  to run in parallel later. If it's only marginally better → the GPU cost isn't justified; stay CPU /
  go multi-phone.
- **Caveats:** NeMo is a heavy/finicky dep (4th env); confirm it loads + runs **offline** for the TEE;
  Apple-Silicon local = no CUDA, so this track runs on Colab/cloud/TEE, not this Mac.

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
