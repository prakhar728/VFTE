# E.2 — diart diarization DER baseline

The bar a future lean ONNX engine (E.3) would have to meet. Measures the shipped
`DiartDiarizer` (streaming, merged spans) on the AMI single-mic mix vs the reference
RTTM, strict protocol: **no collar, overlap scored, Hungarian speaker mapping**.

Run (diart venv): `HF_TOKEN=… PYTHONPATH=. /tmp/diart-venv/bin/python -m evaluation.der_eval`

## Result

| meeting | DER | missed | false-alarm | confusion | ref speech (s) |
|---|---|---|---|---|---|
| IS1009a | 35.5 % | 49 | 75 | 123 | 696 |
| ES2004a | 30.9 % | 103 | 65 | 118 | 923 |
| **aggregate** | **32.9 %** | 152 | 140 | 241 | 1619 |

Error is **confusion-dominated** (speaker label swaps), as expected for *online* clustering
on 4-speaker **far-field single-mic** audio — the hardest realistic setting. (Offline,
collar-forgiving SOTA on AMI-SDM is ~20-25 %; strict online single-mic is materially worse.)

## Read — and why it doesn't sink the product

Diarization (who-spoke-when) is the weak link at **32.9 %**, but **identity is not**: E.1
showed EER ~0.2 % (genuine 0.76 vs impostor 0.08). Because the identify layer **re-embeds
every diart span with CAM++ and matches the enrolled store** (not diart's own labels), much
of diart's *confusion* is corrected downstream — a span diart mislabels `speaker1` still
re-embeds to the right enrolled voiceprint. The vote-lock then stabilizes the mapping.

So the diart DER is a diarization-segmentation quality number, not the product's identity
accuracy. It's acceptable for now. The open weaknesses diart leaves — overlap (missed) and
boundary churn — are what a future engine would target.

## Implication for E.3 (build our own lean ONNX engine?)

Building E.3 to *match* 32.9 % buys **no accuracy** — only a torch-free, leaner TEE image.
The accuracy case for a custom engine is weak (the ID layer already carries identity); the
deployment case (drop torch/pyannote, smaller image, cleaner enclave) is the real reason, and
it only bites at TEE packaging (E.5). Decision deferred to the user with this data.
