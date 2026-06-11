# Feasibility spikes — diarization engine candidates (M1 gates)

Purpose: resolve the plan's gating unknowns before building full pipelines, so the
M2 bake-off only includes candidates that are actually viable.

## D3 — EEND-TA: DROPPED

No public code or pretrained weights found for EEND-TA / "Pushing the Limits of
End-to-End Diarization" (arXiv 2509.14737, 2312.06253) — papers only, no repo /
HF checkpoint / license. Per the plan gate (weights+license must be confirmed),
**D3 is out**. The bake-off reduces to **D1 vs D2**.

## D1 — DiariZen: FEASIBLE on CPU ✅

Model `BUT-FIT/diarizen-wavlm-large-s80-md` = 80%-pruned WavLM-large + Conformer
powerset-EEND + **VBx** clustering. Code MIT.

Spike: bundled 30 s AMI clip (`example/EN2002a_30s.wav`), CPU only, 12-core
Apple-silicon, torch CPU build.

| Metric | Result | Note |
|---|---|---|
| **RTF** | **≈ 0.56** | faster than real-time; ~33 min for a 1-hr recording |
| **Peak RSS** | **≈ 4.9 GB** | the real constraint — TEE memory budget |
| Output | 4 speakers, 13 segments | sane on the sample |
| Internals | WavLM + Conformer EEND + VBxClustering | confirms the §2 D1 description |

### Reproduction (legacy env required)

DiariZen pins torch 2.1.1; reproduce its dependency set exactly (it does NOT
resolve cleanly against latest torch/transformers/numpy):

```
torch==2.1.1  torchaudio==2.1.1   # AudioMetaData removed in >=2.11
transformers==4.36.2 accelerate==0.25.0   # 4.x>=4.38 / 5.x break on torch 2.1.1 _pytree API
numpy==1.26.4  "pyannote.core<6"   # pyannote.core>=6 wants numpy>=2
```
Install: clone repo → trimmed `requirements.txt` (drop onnxruntime-gpu/jupyter/
tensorboard/pesq) → `pip install -e .` → vendored `pyannote-audio` `pip install -e .`.
Inference API: `DiariZenPipeline.from_pretrained(...)(wav)` → pyannote Annotation.

### Implications

- **D1 is the accuracy + speed frontrunner** (SOTA ~13.3% DER, overlap-aware, and
  faster-than-real-time on CPU). Likely the engine, with our ID layer bolted on.
- **Memory (~4.9 GB) is the watch-item** for the TEE. Later optimization: export
  WavLM to int8 ONNX to cut RAM/image size (out of MVP scope; torch path works now).
- The legacy pin-set must be baked into the service image (or reproduced via their
  conda recipe) — a packaging task, not a blocker.

## Next

Build **D2** (clustering+VBx floor; its VAD/embedding/store components also feed the
ID layer), then run **D1 vs D2** on an AMI subset through the C0.2 DER harness for the
real comparison (M2).
