"""E.1 — identification accuracy (FAR / FRR / EER) on far-field AMI mix audio.

The key metric for the offline path: can we recognize an enrolled voice and
*reject* strangers? We cut single-speaker regions from the single-mic AMI mix
(via the reference RTTM), enroll each speaker on the first half of their
utterances, and score the second half against every speaker's centroid:
  - genuine  = test utterance vs its OWN speaker's centroid
  - impostor = test utterance vs every OTHER speaker's centroid
From the two score distributions we sweep a threshold → FAR/FRR, find the EER,
and fit the sigmoid (cos → P(genuine)) for calibrated confidence. This is
deliberately the HARD case (far-field, single mic) — the same audio the product
sees — so the numbers are honest, not headset-clean.

Run (core venv, embedder model present):
    python -m evaluation.id_eval
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from fpm.embed.onnx_embedder import OnnxSpeakerEmbedder

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "campplus.onnx"
EVAL = ROOT / "eval_data"
RTTM_DIR = EVAL / "AMI-diarization-setup" / "only_words" / "rttms" / "test"
MEETINGS = ["IS1009a", "ES2004a"]
SR = 16_000

MIN_UTT_SEC = 2.0      # ≥ one embed window
MAX_UTT_SEC = 6.0      # cap so utterances are comparable
GRID = 0.01            # 10 ms timeline resolution
MAX_UTTS_PER_SPK = 12  # bound the work


def parse_rttm(path: Path) -> list[tuple[float, float, str]]:
    segs = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER" and float(p[4]) > 0:
            segs.append((float(p[3]), float(p[3]) + float(p[4]), p[7]))
    return segs


def exclusive_utterances(segs: list[tuple[float, float, str]]) -> dict[str, list[tuple[float, float]]]:
    """Per-speaker intervals where ONLY that speaker is active (no overlap)."""
    speakers = sorted({s for _, _, s in segs})
    idx = {s: i for i, s in enumerate(speakers)}
    end = max(e for _, e, _ in segs)
    n = int(end / GRID) + 1
    count = np.zeros(n, dtype=np.int16)
    owner = np.full(n, -1, dtype=np.int16)
    for s, e, spk in segs:
        a, b = int(s / GRID), int(e / GRID)
        count[a:b] += 1
        owner[a:b] = idx[spk]

    out: dict[str, list[tuple[float, float]]] = {s: [] for s in speakers}
    for spk in speakers:
        mask = (count == 1) & (owner == idx[spk])
        # contiguous runs of exclusive frames
        edges = np.diff(np.concatenate([[0], mask.view(np.int8), [0]]))
        starts = np.where(edges == 1)[0]
        stops = np.where(edges == -1)[0]
        for a, b in zip(starts, stops):
            t0, t1 = a * GRID, b * GRID
            while t1 - t0 >= MIN_UTT_SEC:
                seg_end = min(t0 + MAX_UTT_SEC, t1)
                if seg_end - t0 >= MIN_UTT_SEC:
                    out[spk].append((t0, seg_end))
                t0 = seg_end
    return out


@dataclass
class Speaker:
    key: str
    embs: list[np.ndarray]


def build_speakers(embedder) -> list[Speaker]:
    speakers: list[Speaker] = []
    for meeting in MEETINGS:
        wav = EVAL / f"{meeting}.16k.wav"
        audio, sr = sf.read(wav, dtype="float32")
        assert sr == SR
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        utts = exclusive_utterances(parse_rttm(RTTM_DIR / f"{meeting}.rttm"))
        for spk, intervals in utts.items():
            embs = []
            for (t0, t1) in intervals[:MAX_UTTS_PER_SPK]:
                clip = audio[int(t0 * SR): int(t1 * SR)]
                e = embedder.extract(clip, SR)
                if e is not None:
                    embs.append(e)
            if len(embs) >= 4:                     # need ≥2 enroll + ≥2 test
                speakers.append(Speaker(f"{meeting}/{spk}", embs))
    return speakers


def _centroid(embs: list[np.ndarray]) -> np.ndarray:
    m = np.mean(embs, axis=0)
    n = float(np.linalg.norm(m))
    return (m / n if n > 1e-10 else m).astype(np.float32)


def score(speakers: list[Speaker]) -> tuple[np.ndarray, np.ndarray]:
    """Temporal split: enroll on first half, test on second half."""
    centroids, tests = {}, {}
    for sp in speakers:
        h = len(sp.embs) // 2
        centroids[sp.key] = _centroid(sp.embs[:h])
        tests[sp.key] = sp.embs[h:]
    genuine, impostor = [], []
    for sp in speakers:
        for e in tests[sp.key]:
            for key, c in centroids.items():
                (genuine if key == sp.key else impostor).append(float(e @ c))
    return np.array(genuine), np.array(impostor)


def far_frr(genuine: np.ndarray, impostor: np.ndarray):
    ths = np.linspace(-0.2, 1.0, 121)
    far = np.array([(impostor >= t).mean() for t in ths])
    frr = np.array([(genuine < t).mean() for t in ths])
    eer_i = int(np.argmin(np.abs(far - frr)))
    return ths, far, frr, eer_i


def fit_sigmoid(genuine: np.ndarray, impostor: np.ndarray, iters=4000, lr=0.5):
    """Logistic regression P(genuine) = sigmoid(a*cos + b), simple GD (deterministic)."""
    x = np.concatenate([genuine, impostor]).astype(np.float64)
    y = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)]).astype(np.float64)
    a, b = 10.0, -3.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(a * x + b)))
        ga = np.mean((p - y) * x)
        gb = np.mean(p - y)
        a -= lr * ga
        b -= lr * gb
    return float(a), float(b)


def main() -> dict:
    if not MODEL.exists():
        raise SystemExit("embedder model missing — run scripts/fetch_models.sh")
    embedder = OnnxSpeakerEmbedder(MODEL).load()
    speakers = build_speakers(embedder)
    genuine, impostor = score(speakers)
    ths, far, frr, eer_i = far_frr(genuine, impostor)
    alpha, beta = fit_sigmoid(genuine, impostor)

    # operating points: EER (balanced) and a high-precision "name" point (FAR ≈ 1%)
    far1_i = int(np.argmin(np.abs(far - 0.01)))
    report = {
        "speakers": len(speakers),
        "genuine_pairs": int(genuine.size),
        "impostor_pairs": int(impostor.size),
        "genuine_mean": round(float(genuine.mean()), 3),
        "impostor_mean": round(float(impostor.mean()), 3),
        "eer": round(float((far[eer_i] + frr[eer_i]) / 2), 3),
        "eer_threshold": round(float(ths[eer_i]), 3),
        "far1pct_threshold": round(float(ths[far1_i]), 3),
        "sigmoid_alpha": round(alpha, 3),
        "sigmoid_beta": round(beta, 3),
    }
    print("=== E.1 IDENTIFICATION ACCURACY (far-field AMI mix) ===")
    for k, v in report.items():
        print(f"  {k:22s} {v}")
    print(f"  genuine:  mean={genuine.mean():.3f} std={genuine.std():.3f} "
          f"min={genuine.min():.3f} max={genuine.max():.3f}")
    print(f"  impostor: mean={impostor.mean():.3f} std={impostor.std():.3f} "
          f"min={impostor.min():.3f} max={impostor.max():.3f}")
    print("  threshold  FAR    FRR")
    for t in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5):
        i = int(np.argmin(np.abs(ths - t)))
        print(f"    {t:.2f}    {far[i]:.3f}  {frr[i]:.3f}")
    return report


if __name__ == "__main__":
    main()
