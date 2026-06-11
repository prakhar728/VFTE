"""C0.2 — DER harness correctness (the bake-off decision instrument)."""
from pyannote.core import Annotation, Segment

from evaluation.der import compute_der
from evaluation.harness import evaluate, evaluate_rttm_files
from evaluation.rttm import parse_rttm, write_rttm


def test_known_der():
    # ref: A[0,10] B[10,20] ; hyp: X[0,10] Y[12,20]  -> optimal X->A, Y->B
    # B missed 10-12 (2s); total 20s -> DER = 2/20 = 0.10
    ref = Annotation(uri="f1")
    ref[Segment(0, 10)] = "A"
    ref[Segment(10, 20)] = "B"
    hyp = Annotation(uri="f1")
    hyp[Segment(0, 10)] = "X"
    hyp[Segment(12, 20)] = "Y"

    r = compute_der(ref, hyp)
    assert abs(r["der"] - 0.10) < 1e-6
    assert abs(r["missed"] - 2.0) < 1e-6
    assert abs(r["false_alarm"]) < 1e-6
    assert abs(r["confusion"]) < 1e-6
    assert abs(r["total"] - 20.0) < 1e-6


def test_self_der_is_zero():
    ref = Annotation(uri="f1")
    ref[Segment(0, 5)] = "A"
    ref[Segment(5, 9)] = "B"
    assert compute_der(ref, ref)["der"] == 0.0


def test_rttm_roundtrip(tmp_path):
    ref = Annotation(uri="m1")
    ref[Segment(0, 10)] = "A"
    ref[Segment(10, 20)] = "B"
    path = tmp_path / "ref.rttm"
    write_rttm({"m1": ref}, path)

    loaded = parse_rttm(path)
    assert "m1" in loaded
    # scoring an RTTM against itself must be 0
    assert abs(evaluate_rttm_files(path, path)["aggregate"]["der"]) < 1e-9


def test_aggregate_is_duration_weighted():
    # file a: 20s with 2s error ; file b: 40s perfect -> aggregate = 2/60, NOT mean(0.10, 0)
    ref_a = Annotation(uri="a")
    ref_a[Segment(0, 10)] = "A"
    ref_a[Segment(10, 20)] = "B"
    hyp_a = Annotation(uri="a")
    hyp_a[Segment(0, 10)] = "X"
    hyp_a[Segment(12, 20)] = "Y"

    ref_b = Annotation(uri="b")
    ref_b[Segment(0, 40)] = "C"
    hyp_b = Annotation(uri="b")
    hyp_b[Segment(0, 40)] = "Z"

    res = evaluate({"a": (ref_a, hyp_a), "b": (ref_b, hyp_b)})
    assert abs(res["aggregate"]["der"] - (2.0 / 60.0)) < 1e-6
    assert abs(res["aggregate"]["total"] - 60.0) < 1e-6
