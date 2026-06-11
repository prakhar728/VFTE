"""A.4 — Voiceprint profile (exemplars, centroid, quality, drift)."""
import numpy as np

from fpm.store.models import MAX_EXEMPLARS, Voiceprint


def _unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_add_and_recompute_centroid_is_unit():
    vp = Voiceprint("v1", "ws")
    for i in range(3):
        vp.add_exemplar(_unit(i))
    vp.recompute_centroid()
    assert vp.centroid.shape == (512,)
    assert abs(float(np.linalg.norm(vp.centroid)) - 1.0) < 1e-5


def test_exemplars_capped():
    vp = Voiceprint("v1", "ws")
    for i in range(MAX_EXEMPLARS + 10):
        vp.add_exemplar(_unit(i))
        vp.recompute_centroid()
    assert len(vp.exemplars) == MAX_EXEMPLARS


def test_quality_high_for_tight_cluster():
    vp = Voiceprint("v1", "ws")
    base = _unit(0)
    for i in range(5):
        e = base + _unit(100 + i) * 0.1
        vp.add_exemplar((e / np.linalg.norm(e)).astype(np.float32))
    assert vp.compute_quality() > 0.8


def test_best_match_score():
    vp = Voiceprint("v1", "ws")
    for i in range(3):
        e = _unit(0) + _unit(50 + i) * 0.05
        vp.add_exemplar((e / np.linalg.norm(e)).astype(np.float32))
    vp.recompute_centroid()
    assert vp.best_match_score(vp.centroid) > 0.9
    assert vp.best_match_score(_unit(999)) < 0.5
