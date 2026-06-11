# A.3 — ID embedder bench / decision

The **fixed ID embedder** defines the voiceprint store's vector space. It's swappable
(`OnnxSpeakerEmbedder` + `config.ID_EMBEDDING_MODEL`; the store records model+dim and
refuses cross-model matches), so this can be revised later without invalidating design.

## Chosen: CAM++ (WeSpeaker VoxCeleb, self-contained ONNX)

`feats (B,T,80)` → `embs (B,512)`. Apache-2.0, ~29 MB, torch-free (our numpy fbank → onnxruntime).

| Metric | Result |
|---|---|
| Separation (AMI clips, our fbank) | same-speaker **0.624**, different **−0.001** (margin 0.63) |
| RTF, 1 s chunk | **7.3 ms** (RTF 0.0073) |
| RTF, 2 s chunk | 11.9 ms (RTF 0.0059) |
| Dim | 512 |

→ Fast enough for real-time small-chunk identification (~7 ms/chunk, ~140× real-time on CPU),
strong speaker separation. Good default.

## Deferred (UNFINALIZED #1): ERes2NetV2 comparison

ERes2NetV2 (192-d, claimed slightly better short-utterance EER) is not readily available as a
self-contained `feats→embs` ONNX in an ungated zoo; VoxTerm's released ERes2Net ONNX ships
without its `.onnx.data` weights (unusable). Sourcing it needs a torch export. Deferred to the
**E.1 FAR/FRR bench** (with a proper multi-speaker verification set) — if CAM++ accuracy is
insufficient there, export + swap ERes2NetV2. CAM++ is the working default until then.

## Note on fbank
Our Kaldi-style fbank (with CMN) yields strong separation but a different absolute operating
point than WeSpeaker's internal fbank (e.g. same-spk 0.62 vs 0.94). Irrelevant: we calibrate our
OWN thresholds on our OWN embeddings (sigmoid-calibrated cosine + open-set, A.7/E.1). If the
E.1 EER shows headroom lost to fbank mismatch, match WeSpeaker's fbank params then.
