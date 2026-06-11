"""ONNX speaker embedder — the FIXED ID model that defines the voiceprint space.

Adapted from VoxTerm `audio/diarization/onnx_embedder.py` (MIT). Torch-free:
pure-numpy fbank (A.1) → onnxruntime. Loads a self-contained ONNX model from a
LOCAL path (baked into the image at build time) — no download/export at runtime,
so it runs fully offline inside the enclave.

Model contract: input `(1, T, 80)` fbank features (input name auto-detected, e.g.
`feats`/`feature`) → output `(1, D)` embedding (D=512 for CAM++).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .fbank import compute_fbank

log = logging.getLogger(__name__)

_MIN_SAMPLES = 16_000  # 1.0 s @16k — shorter audio gives unreliable embeddings


class OnnxSpeakerEmbedder:
    """Extract L2-normalized speaker embeddings via ONNX Runtime (no PyTorch)."""

    def __init__(self, model_path: str | Path, sample_rate: int = 16_000):
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        self._session = None
        self._input_name: str | None = None
        self._embedding_dim: int = 0

    def load(self) -> "OnnxSpeakerEmbedder":
        import onnxruntime as ort

        if not self.model_path.exists():
            raise FileNotFoundError(f"embedder model not found: {self.model_path}")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"], sess_options=opts
        )
        self._input_name = self._session.get_inputs()[0].name
        out_last = self._session.get_outputs()[0].shape[-1]
        self._embedding_dim = int(out_last) if isinstance(out_last, int) else 0
        log.info("Loaded ONNX embedder %s (dim=%s)", self.model_path.name, self._embedding_dim)
        return self

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def extract(self, audio: np.ndarray, sample_rate: int | None = None) -> np.ndarray | None:
        """1-D float32 audio in [-1, 1] → L2-normalized embedding, or None if too short."""
        if self._session is None:
            return None
        sr = sample_rate or self.sample_rate
        audio = np.asarray(audio, dtype=np.float32).ravel()
        if len(audio) < _MIN_SAMPLES:
            return None
        feats = compute_fbank(audio, sample_rate=sr)
        if feats.shape[0] == 0:
            return None
        feats_in = feats[np.newaxis, :, :].astype(np.float32)  # (1, T, 80)
        emb = self._session.run(None, {self._input_name: feats_in})[0].squeeze()
        norm = float(np.linalg.norm(emb))
        if norm > 1e-10:
            emb = emb / norm
        if self._embedding_dim == 0:
            self._embedding_dim = int(emb.shape[-1])
        return emb.astype(np.float32)
