# Diarization experiment log

Running ledger of every diarization eval run + the **decision** it drove. Append-only; newest first.
Plan = `ami-eval-plan.md`. Order = **anchor → accuracy knobs (batch) → chunk-size elbow → linker**.

**Protocols** (both reported): **strict** = collar 0, overlap scored, Hungarian mapping (the honest
number); **lenient** = collar 0.25, overlap skipped (≈ DiariZen's published protocol).
**Dataset:** AMI single-distant-mic, `only_words` RTTM refs. Current meetings: IS1009a, ES2004a.
**Gear-change rules** (logged per transition): adopt a factor level only if it improves aggregate DER
by **>1.0 absolute pt** (else keep default); ties → lower speaker-confusion; best at range endpoint →
extend, interior min → refine ±; chunk knee = smallest chunk within 1.0pt of batch.

---

## Committed prior baselines (FPM M2 bake-off — `docs/bakeoff-offline.md`, `der-eval.md`)

| Engine | strict DER | lenient DER | RTF | peak RAM | meetings |
|---|---|---|---|---|---|
| DiariZen (batch, WavLM-EEND+VBx) | 22.8% | ~14% | 1.24–1.35 | 16.6 GB | IS1009a, ES2004a |
| diart (streaming, merged spans) | 32.9% | — | 0.28 | ~0.85 GB | IS1009a, ES2004a |

> lenient ~14% ≈ DiariZen published ~13–14% → harness/protocol validated on these 2 meetings.

---

## Runs

### Run 001 — DiariZen-batch anchor (re-reproduce with current code) — ✅ PASS
- **Date:** 2026-06-13
- **Goal:** confirm current code reproduces the committed bake-off (~22.8% strict / ~14% lenient)
  before building the sweep. The trust anchor.
- **Engine/config:** DiariZen-batch, `BUT-FIT/diarizen-wavlm-large-s80-md`, defaults
  (ahc_threshold 0.6, median_filter on, seg_duration 16, segmentation_step 0.1).
- **Command:** `HF_TOKEN=… PYTHONPATH=. /tmp/diarizen-venv/bin/python -m evaluation.diarizen_eval`
- **Output:** `eval_data/results/diarizen_anchor.json`
- **Results:**

  | meeting | strict DER | lenient DER | spk (ref 4) | RTF | len |
  |---|---|---|---|---|---|
  | IS1009a | 16.7% | 10.9% | **5** | 1.26 | 14.0 min |
  | ES2004a | 27.3% | 18.7% | **5** | 1.27 | 17.5 min |
  | **aggregate** | **22.7%** | **15.3%** | — | — | — |

  Peak RAM **15.5 GB**.
- **Verdict / decision:** **PASS — reproduced** (22.7% vs bake-off 22.8%, within 0.1pt; ES2004a exact;
  RAM/RTF match). Harness + protocol **trusted.** Both meetings over-detect by 1 speaker (5 vs 4) —
  the over-segmentation pattern; `ahc_threshold` is the lever if/when we tune internal knobs.
- **Next gear:** per the architecture discussion (2026-06-13), the chunked build will use **diart-style
  online clustering (cannot-link) for cross-chunk *consistency* + FPM fingerprint for *identity***
  (two layers, different jobs) — pending user confirm. Internal-knob sweep (ahc/median/VAD) demoted to
  optional. Next runs to use **short AMI excerpts** for quick iteration.
