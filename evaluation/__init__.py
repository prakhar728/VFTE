"""Diarization evaluation harness — the bake-off decision instrument (M0/C0.2).

Scores hypothesis diarization against reference using the strict protocol
(no collar, overlap scored, optimal/Hungarian speaker mapping) via
`pyannote.metrics`. Every later engine decision (M2 bake-off) is graded here.
"""
