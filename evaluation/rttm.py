"""RTTM <-> pyannote Annotation helpers (no pyannote.database dependency).

RTTM line format (NIST):
    SPEAKER <uri> <chan> <start> <dur> <NA> <NA> <speaker> <NA> <NA>
A single RTTM file may contain multiple recordings (URIs); we key by URI.
"""
from __future__ import annotations

from pathlib import Path

from pyannote.core import Annotation, Segment


def parse_rttm(path: str | Path) -> dict[str, Annotation]:
    """Load an RTTM file into {uri: Annotation}."""
    annotations: dict[str, Annotation] = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            uri, start, dur, speaker = parts[1], float(parts[3]), float(parts[4]), parts[7]
            if dur <= 0:
                continue
            ann = annotations.setdefault(uri, Annotation(uri=uri))
            ann[Segment(start, start + dur)] = speaker
    return annotations


def write_rttm(annotations: dict[str, Annotation], path: str | Path) -> None:
    """Write {uri: Annotation} to an RTTM file (used to dump candidate hypotheses)."""
    with open(path, "w") as fh:
        for uri, ann in annotations.items():
            for segment, _, label in ann.itertracks(yield_label=True):
                fh.write(
                    f"SPEAKER {uri} 1 {segment.start:.3f} {segment.duration:.3f} "
                    f"<NA> <NA> {label} <NA> <NA>\n"
                )
