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

## ✅ RESOLVED (fixed-window embedding): instability on variable-length spans

**Fix shipped.** `OnnxSpeakerEmbedder.extract` now embeds fixed-length windows
(`EMBED_WINDOW_SEC=2.0`, `EMBED_HOP_SEC=1.0`) and averages them, so an enrolled
clip and an arbitrary diarized span map into the same space. Validated on the live
diart→identify e2e: the enrolled speaker is recognized by name (cos→centroid
~0.99–1.00) while a different speaker stays anonymous (~0.01) — clean separation
preserved. Set `FPM_EMBED_WINDOW_SEC=0` to disable. The cos-vs-length study below
is what motivated it; E.1 still owns the systematic FAR/FRR calibration.

### Original finding — embedding unstable on partial sub-spans of short clips
Re-embedding a *sub-span* of one of the 4 s AMI fixtures can collapse the embedding. Same
speaker, same source `spkA_1`:

| span | frames | cos to full `[0:4]` |
|---|---|---|
| `[0:4.0]` (self) | 398 | 1.000 |
| `[0.1:4.0]` | 388 | 0.971 |
| `[0.3:4.0]` | 368 | 0.614 |
| `[0:2.0]` / `[2:4]` | 198 | 0.895 / 0.888 |
| **`[0:3.4]`** | **338** | **0.090** |
| **`[0.6:4.0]`** | **338** | **0.084** |

Small front-trims (and specifically ~338-frame spans) collapse cosine to ~0.08 — near
orthogonal to the same speaker's full clip. Clean halves (198 frames) are fine. This is an
**embedder/front-end robustness issue**, not a diarize/identify bug: C.4/C.5 slice, embed,
classify, and vote correctly; the vote-lock + multi-segment accumulation is exactly the
mitigation for occasional bad embeddings, which is why the full pipeline still converges.

**Why it mattered:** the real diart path emits variable-length spans, so without the fix even the
enrolled speaker fell through to anonymous (observed: Alice's own turns scored ~0.02 to her print).
**Cause:** length/offset-dependent CMN + shape-sensitivity in the fbank→CAM++ front-end on whole
variable-length spans. **Fix (above):** average fixed 2 s windows + keep CMN — robust (0.96 vs
0.08) while preserving separation. The bake-off that picked it:

| strategy | partial-span (broken case) | same-speaker | different-speaker |
|---|---|---|---|
| full + CMN (old) | 0.08 | 0.62 | −0.00 |
| full, no CMN | 0.53 | 0.59 | 0.74 ✗ (separation lost) |
| **fixed-2s avg + CMN (chosen)** | **0.96** | 0.60 | **0.01** |
| fixed-2s, no CMN | 0.98 | 0.58 | 0.75 ✗ |

E.1 still owns systematic FAR/FRR calibration on a real multi-speaker set (and may revisit window
length). Tests embed full-clip spans where they isolate non-embedder logic.
