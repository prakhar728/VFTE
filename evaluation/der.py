"""Diarization Error Rate with the strict, honest protocol.

DER = (missed + false_alarm + confusion) / total_reference_speech

Defaults: collar=0.0 (no forgiveness window) and skip_overlap=False (overlapped
speech IS scored) — the realistic setting. Speaker mapping is optimal (Hungarian),
done internally by pyannote.metrics.
"""
from __future__ import annotations

from pyannote.core import Annotation
from pyannote.metrics.diarization import DiarizationErrorRate


def compute_der(
    reference: Annotation,
    hypothesis: Annotation,
    collar: float = 0.0,
    skip_overlap: bool = False,
) -> dict[str, float]:
    """DER + its components for one (reference, hypothesis) pair."""
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    d = metric(reference, hypothesis, detailed=True)
    return {
        "der": d["diarization error rate"],
        "missed": d["missed detection"],
        "false_alarm": d["false alarm"],
        "confusion": d["confusion"],
        "total": d["total"],
    }
