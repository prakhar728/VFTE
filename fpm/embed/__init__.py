"""Speaker embedding: pure-numpy fbank front-end + ONNX embedder (the fixed ID model).

Ported/adapted from VoxTerm (`audio/diarization/`, MIT) — torch-free, CPU, runs
locally. This embedder defines the voiceprint store's vector space; it is fixed
and independent of whichever diarizer is in use (see plan §"Engine-independent store").
"""
