"""E.1 — ID accuracy harness: determinism + sane operating point."""
import numpy as np
import pytest

from evaluation.id_eval import EVAL, MODEL, far_frr, fit_sigmoid, main


def test_far_frr_and_sigmoid_deterministic():
    rng = np.random.default_rng(0)
    genuine = rng.normal(0.75, 0.1, 200).clip(-1, 1)
    impostor = rng.normal(0.08, 0.1, 200).clip(-1, 1)
    ths, far, frr, eer_i = far_frr(genuine, impostor)
    a1, b1 = fit_sigmoid(genuine, impostor)
    a2, b2 = fit_sigmoid(genuine, impostor)
    assert (a1, b1) == (a2, b2)                 # deterministic GD
    assert far[eer_i] == pytest.approx(frr[eer_i], abs=0.05)
    assert 0.3 < ths[eer_i] < 0.6               # separation lands the EER mid-range


def test_far_frr_monotonic():
    genuine = np.full(50, 0.8)
    impostor = np.full(50, 0.1)
    ths, far, frr, _ = far_frr(genuine, impostor)
    assert np.all(np.diff(far) <= 1e-9)         # FAR non-increasing in threshold
    assert np.all(np.diff(frr) >= -1e-9)        # FRR non-decreasing


@pytest.mark.skipif(
    not (MODEL.exists() and (EVAL / "IS1009a.16k.wav").exists()),
    reason="AMI eval audio / embedder model not present (gitignored eval_data)",
)
def test_full_eval_separates_speakers():
    r = main()
    assert r["speakers"] >= 4
    assert r["genuine_mean"] > r["impostor_mean"] + 0.3
    assert r["eer"] < 0.1                        # clean separation on far-field mix
