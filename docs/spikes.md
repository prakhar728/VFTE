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

> **NOTE:** the D1/D2/D3 study above is the **parked offline-batch** detour (whole-file
> DER, wrong axis). The live product axis is the real-time streaming spike below (C.2).

---

## C.2 — diart streaming (the real-time go/no-go gate) — **GO ✅**

The decision this spike gates is narrow: **does diart run real-time on CPU?** That's
RTF / per-chunk latency / bounded RAM. Accuracy (DER) is a *separate* axis (E.2) and was
deliberately **not** measured here — no reference labels needed for a latency gate (MVP
scoping: prove the engine is viable before measuring how good it is).

**Setup:** diart 0.9.2, `SpeakerDiarization`, CPU only (12-core Apple-silicon, M-series),
`step=0.5s latency=0.5s duration=5s`. Segmentation = `pyannote/segmentation-3.0` (gated,
terms accepted); embedding = `pyannote/wespeaker-voxceleb-resnet34-LM` (ungated). diart's
embedder is its **internal scissors only** — our identity layer always re-embeds with CAM++,
so this choice doesn't touch the store. Input: 64s, 2 speakers alternating (built from the
in-repo AMI CC-BY fixtures `tests/fixtures/speakers/spk{A,B}*.wav`, no download).

| Metric | Result | Verdict |
|---|---|---|
| **RTF (wall/audio)** | **0.28** | ~3.6× faster than real-time |
| **Per-chunk latency** | mean **152 ms**, median 149, p95 172, max 209 | vs **500 ms** step budget → ~3× headroom |
| **Peak RSS** | **~850 MB** | vs DiariZen's 4.9 GB — far better for the TEE budget |
| **Speakers detected** | 2 (correct) | sane on a 2-speaker stream |
| Pipeline load (warm cache) | 0.3 s | cold (first download) ~4.6 s |

**Decision: GO** — build the `C.3` diart adapter behind `fpm/diarize/base.py:StreamingDiarizer`.
diart is comfortably real-time on CPU with bounded memory. (Fallback path `E.3` — our own
lean ONNX engine — stays the long-term TEE optimization, not a blocker.)

### Reproduction (dependency pinning — friction, like the DiariZen spike)

Throwaway venv (`/tmp/diart-venv`, Python 3.11), kept separate so `FPM/.venv` stays
torch-free. `pip install diart` pulls bleeding-edge deps that break pyannote.audio 3.4.0;
pin back:

```
torch==2.2.2  torchaudio==2.2.2  torchvision==0.17.2   # 2.11+ dropped torchaudio.AudioMetaData
pytorch-lightning==2.2.1  lightning==2.2.1             # 2.6 breaks on torch 2.2 _pytree API
huggingface_hub==0.23.4                                # 1.x dropped hf_hub_download(use_auth_token=)
onnxruntime                                            # wespeaker embedder loader
# speechbrain ECAPA embedder is a TRAP on pyannote 3.4.0: it passes use_auth_token= to
# speechbrain.from_hparams which 1.1.0 rejects, and 0.5.x lacks speechbrain.inference.
# → use the wespeaker embedder (pyannote loader) instead; ungated, no speechbrain path.
```

HF token: `~/.cache/huggingface/token`; account must accept terms on
`huggingface.co/pyannote/segmentation-3.0` AND the token needs "read access to public
gated repos". Spike script: `/tmp/diart_spike.py` (throwaway).

## Next

C.3 — implement `fpm/diarize/diart_engine.py` against the `StreamingDiarizer` interface
(C.1), models baked-in, `HF_HUB_OFFLINE=1`, confirm zero network at runtime. Then C.4
wires emitted segments → CAM++ re-embed → store match.
