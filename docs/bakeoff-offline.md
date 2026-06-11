# M2 bake-off — OFFLINE mixed-audio diarization engine

Scope: this evaluates the engine for the **offline upload path** (one mixed recording →
separate speakers). It does NOT measure the live/real-time path (per-track audio →
identification only) or identification/fingerprinting accuracy (M3/M5).

Eval: 2 AMI test meetings (IS1009a, ES2004a; 4 speakers each), official `only_words`
RTTM references, scored on our C0.2 harness. Strict = no collar, overlap scored,
Hungarian mapping. Lenient = collar 0.25, overlap skipped (≈ published protocol).

## Results

| Engine | IS1009a | ES2004a | **Aggregate (strict)** | Lenient | RTF | Peak RAM |
|---|---|---|---|---|---|---|
| **D1 — DiariZen** (WavLM-EEND + VBx) | 16.8% | 27.3% | **22.8%** | ~14% | 1.24–1.35 | **16.6 GB** |
| **D2 — sherpa clustering** (CAM++, oracle K=4) | 53.1% | 57.3% | 55.5% | ~54% | 0.06 | ~30 MB |

D1's lenient ~11–19% matches DiariZen's published SOTA. D2's failure is genuine
(embedder discriminates — same-spk 0.94 vs diff 0.74 — so it's clustering collapse on
overlap-heavy, same-gender meetings, not a bug). D2 is tunable but won't close a ~33pt gap.

## Decision (offline path)

**D1 (DiariZen) is the accuracy winner.** Caveat = resources: **RAM scales with audio
length** (4.9 GB @30s → 11 GB @14min → 16.6 GB @17min); RTF ~1.3 (fine for batch).
Adopting D1 for offline requires either a large enclave OR **windowed processing** to
bound RAM, and (later) int8-ONNX WavLM to shrink it.

## Not covered here (separate engines/metrics)

- **Live/real-time path**: per-track audio → identification only; the heavy diarizer is
  NOT used there. If real-time diarization of a *mixed* live stream is needed, that's a
  **streaming** engine (different from D1/D2) — Otter-style, to be evaluated separately.
- **Fingerprinting/identification accuracy** (FAR/FRR): M3/M5.
