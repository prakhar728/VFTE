# E.1 — identification accuracy (FAR / FRR / EER)

The key metric for the offline path: recognize an enrolled voice, reject strangers.

**Data:** single-mic far-field AMI mix (`IS1009a` + `ES2004a`), 8 distinct speakers.
Single-speaker regions cut from the mix via the reference RTTM (≥2 s, ≤6 s; overlap
excluded). Enroll on each speaker's first-half utterances, test on the second half.
This is the HARD case on purpose — the same far-field single-mic audio the product
sees, not headset-clean. Harness: `evaluation/id_eval.py` (`python -m evaluation.id_eval`).

## Result

| metric | value |
|---|---|
| speakers / genuine pairs / impostor pairs | 8 / 44 / 308 |
| genuine cosine (mean ± std) | **0.756 ± 0.102** (min 0.479) |
| impostor cosine (mean ± std) | **0.083 ± 0.102** (max 0.498) |
| **EER** | **≈ 0.2 %** @ threshold 0.42 |
| sigmoid fit (α, β) | 15.5, −7.67 |

FAR/FRR sweep:

| threshold | FAR | FRR |
|---|---|---|
| 0.30 | 0.032 | 0.000 |
| 0.35 (REJECT) | 0.013 | 0.000 |
| 0.40 | 0.006 | 0.000 |
| **0.45 (ACCEPT)** | **0.003** | **0.000** |
| 0.50 | 0.000 | 0.023 |

## Calibration applied (config.py)

- `MATCH_ACCEPT = 0.45` — name point: FAR 0.3 %, FRR 0 % (name-leak-averse; ≥ → MATCH).
- `MATCH_REJECT = 0.35` — open-set floor: < → UNKNOWN (FAR 1.3 %).
- `AMBIGUOUS_MARGIN = 0.10` — unchanged (top-2 within → AMBIGUOUS).
- `SCORE_ALPHA = 15.5`, `SCORE_BETA = -7.67` — fitted logistic (cos → P(genuine)).

## Read

Windowed CAM++ (the E.0 embedder fix) separates speakers cleanly **even on far-field
single-mic audio** — genuine ~0.76 vs impostor ~0.08, EER ~0.2 %. The provisional
thresholds were well-placed and are now empirically grounded. Caveats: 8 speakers from
2 meetings (modest panel); pairs aren't fully independent (utterances from one meeting
share a room). A larger/multi-room panel would tighten the estimate, but the margin here
is large enough to be confident the ID layer is not the weak link.
