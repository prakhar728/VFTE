"""Voiceprint profile model (adapted from VoxTerm `audio/speakers/models.py`, MIT).

Holds a speaker's centroid + exemplars and the logic to refine them: redundancy-
aware exemplar retention, L2-normalized centroid, multi-centroid (k-means) for
mature profiles, quality, and drift detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAX_EXEMPLARS = 20
MULTI_CENTROID_MIN = 15  # min exemplars before sub-centroids kick in
MULTI_CENTROID_K = 3


@dataclass
class Voiceprint:
    voiceprint_id: str
    workspace_id: str
    name: str = ""  # "" = anonymous (consent: only tagged/enrolled get a name)

    # ── consent plane (WS3/WS5) ──────────────────────────────
    # owner_email: the authenticated data subject (roster identity). Plaintext beside
    # the biometric is acceptable *inside* the sealed TEE (decision B). "" = unclaimed.
    owner_email: str = ""
    # User-supreme controls (decision E), enforced at FPM (decision D):
    #   enroll_allowed=False   → enroll.py refuses to create/strengthen this voiceprint
    #   identify_allowed=False → "stay anonymous": still clusters, never surfaces a name
    enroll_allowed: bool = True
    identify_allowed: bool = True

    centroid: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    exemplars: list[np.ndarray] = field(default_factory=list)
    sub_centroids: list[np.ndarray] = field(default_factory=list)

    enroll_count: int = 0  # times reinforced (drives cold-start adaptive threshold)
    total_duration_sec: float = 0.0
    quality_score: float = 0.0

    created_at: str = ""
    updated_at: str = ""
    last_seen_at: str = ""

    # ── exemplar / centroid maintenance ──────────────────────

    def add_exemplar(self, embedding: np.ndarray) -> None:
        """Add an exemplar; if full, replace the one most redundant with the centroid."""
        emb = embedding.astype(np.float32)
        if len(self.exemplars) < MAX_EXEMPLARS:
            self.exemplars.append(emb.copy())
        elif self.centroid.size:
            sims = [float(self.centroid @ e) for e in self.exemplars]  # unit vectors → cos
            self.exemplars[int(np.argmax(sims))] = emb.copy()

    def recompute_centroid(self) -> None:
        """Centroid = L2-normalized mean of exemplars; refresh sub-centroids."""
        if not self.exemplars:
            return
        mean = np.stack(self.exemplars).mean(axis=0)
        norm = float(np.linalg.norm(mean))
        self.centroid = (mean / norm if norm > 1e-10 else mean).astype(np.float32)
        if len(self.exemplars) >= MULTI_CENTROID_MIN:
            self._compute_sub_centroids()
        else:
            self.sub_centroids = []

    def _compute_sub_centroids(self) -> None:
        k = min(MULTI_CENTROID_K, len(self.exemplars) // 3)
        if k < 2:
            self.sub_centroids = []
            return
        data = np.stack(self.exemplars)
        centers = data[np.linspace(0, len(data) - 1, k, dtype=int)].copy()
        for _ in range(10):
            assign = (data @ centers.T).argmax(axis=1)
            new = np.zeros_like(centers)
            for ci in range(k):
                mask = assign == ci
                if mask.any():
                    m = data[mask].mean(axis=0)
                    n = float(np.linalg.norm(m))
                    new[ci] = m / n if n > 1e-10 else m
                else:
                    new[ci] = centers[ci]
            if np.allclose(centers, new, atol=1e-6):
                break
            centers = new
        self.sub_centroids = [c.astype(np.float32) for c in centers]

    # ── scoring ──────────────────────────────────────────────

    def best_match_score(self, embedding: np.ndarray) -> float:
        """Cosine to the nearest sub-centroid (mature profile) or the centroid."""
        emb = embedding.astype(np.float32)
        if self.sub_centroids:
            return max(float(emb @ c) for c in self.sub_centroids)
        return float(emb @ self.centroid) if self.centroid.size else -1.0

    def compute_quality(self) -> float:
        """Mean pairwise cosine of exemplars (tightness), clamped to [0, 1]."""
        if len(self.exemplars) < 2:
            return 0.0
        x = np.stack(self.exemplars)
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-10)
        sims = x @ x.T
        mask = np.triu(np.ones_like(sims, dtype=bool), k=1)
        return max(0.0, min(1.0, float(sims[mask].mean())))

    def detect_drift(self) -> float:
        """Cosine distance between the first-5 'golden' centroid and current centroid."""
        if len(self.exemplars) < 5 or not self.centroid.size:
            return 0.0
        golden = np.stack(self.exemplars[:5]).mean(axis=0)
        n = float(np.linalg.norm(golden))
        if n > 1e-10:
            golden /= n
        return max(0.0, 1.0 - float(golden @ self.centroid))
