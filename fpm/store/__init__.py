"""Voiceprint persistence: encrypted SQLite store + profile model + at-rest crypto.

Adapted from VoxTerm `audio/speakers/` (MIT). Embeddings are encrypted at rest;
the store is workspace-scoped and engine-independent (centroids are always from
the fixed ID embedder).
"""
