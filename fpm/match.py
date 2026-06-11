"""Open-set voiceprint matching — calibrated cosine + rejection.

Given a query embedding and a workspace's centroids, decide:
  MATCH      → confidently the same as an enrolled voiceprint (reuse its id)
  AMBIGUOUS  → top-2 too close to safely pick (don't name — name-leak guard)
  UNKNOWN    → no centroid close enough (mint a new anonymous voiceprint)
  LOW        → above the reject floor but below accept, not ambiguous

Decisions use raw cosine tiers (thresholds in config, calibrated in E.1); a
sigmoid-calibrated [0,1] confidence is returned alongside for display.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from config import AMBIGUOUS_MARGIN, MATCH_ACCEPT, MATCH_REJECT, SCORE_ALPHA, SCORE_BETA


@dataclass
class MatchResult:
    decision: str                 # MATCH | AMBIGUOUS | UNKNOWN | LOW
    voiceprint_id: str | None     # set only on MATCH
    score: float                  # raw cosine to the best centroid
    confidence: float             # sigmoid-calibrated [0, 1]


def calibrate(cos: float) -> float:
    return 1.0 / (1.0 + math.exp(-(SCORE_ALPHA * cos + SCORE_BETA)))


def classify(query: np.ndarray, centroids: dict[str, np.ndarray]) -> MatchResult:
    """Classify a query embedding against a workspace's enrolled centroids."""
    q = np.asarray(query, dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n > 1e-10:
        q = q / n
    scored = sorted(
        ((vid, float(q @ np.asarray(c, dtype=np.float32))) for vid, c in centroids.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    if not scored:
        return MatchResult("UNKNOWN", None, -1.0, 0.0)

    best_vid, best = scored[0]
    second = scored[1][1] if len(scored) > 1 else -1.0
    conf = calibrate(best)

    if best < MATCH_REJECT:
        return MatchResult("UNKNOWN", None, best, conf)
    if best < second + AMBIGUOUS_MARGIN:
        return MatchResult("AMBIGUOUS", None, best, conf)
    if best >= MATCH_ACCEPT:
        return MatchResult("MATCH", best_vid, best, conf)
    return MatchResult("LOW", None, best, conf)
