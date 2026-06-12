"""Standalone eval harness for in-person diarization + transcription.

A lab bench (NOT the production Recato-hub flow) to A/B the knobs that matter — vocab on/off,
diarizer engine (diart now, DiariZen pluggable later), and window size — against gold.
Each experiment is a self-describing folder under `experiments/`; run it with `run.py`.
"""
