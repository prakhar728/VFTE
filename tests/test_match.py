"""A.7 — open-set matching (MATCH / AMBIGUOUS / UNKNOWN / LOW)."""
import numpy as np

from fpm.match import calibrate, classify


def _unit(seed: int, dim: int = 512) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _orth_to(c: np.ndarray, seed: int) -> np.ndarray:
    o = _unit(seed)
    o = o - (o @ c) * c
    return o / np.linalg.norm(o)


def test_empty_store_unknown():
    assert classify(_unit(1), {}).decision == "UNKNOWN"


def test_exact_match():
    c1, c2 = _unit(1), _unit(2)  # near-orthogonal
    r = classify(c1, {"v1": c1, "v2": c2})
    assert r.decision == "MATCH" and r.voiceprint_id == "v1"
    assert r.score > 0.99 and r.confidence > 0.9


def test_far_query_unknown():
    r = classify(_unit(99), {"v1": _unit(1), "v2": _unit(2)})
    assert r.decision == "UNKNOWN" and r.voiceprint_id is None


def test_ambiguous_when_two_close():
    c3 = _unit(3)
    c4 = c3 + 0.05 * _unit(4)
    c4 = c4 / np.linalg.norm(c4)  # very close to c3
    r = classify(c3, {"v3": c3, "v4": c4})
    assert r.decision == "AMBIGUOUS" and r.voiceprint_id is None


def test_low_confidence_band():
    c5 = _unit(5)
    q = 0.40 * c5 + np.sqrt(1 - 0.16) * _orth_to(c5, 6)  # cosine to c5 == 0.40
    r = classify(q.astype(np.float32), {"v5": c5})
    assert r.decision == "LOW", f"score={r.score:.3f}"  # between reject 0.35 and accept 0.45


def test_calibration_monotonic_bounded():
    assert 0.0 <= calibrate(0.0) <= calibrate(0.5) <= calibrate(1.0) <= 1.0
