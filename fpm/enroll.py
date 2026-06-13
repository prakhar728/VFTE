"""Enrollment: labeled audio → stored voiceprint.

This is the `gmeet` path's core: Recato sends a clip already attributed to a
roster identity (the trusted source of truth); FPM turns it into / strengthens
that person's voiceprint. Idempotent per identity — repeat calls accumulate
exemplars and refine the centroid. A consistency gate rejects clips that don't
match the existing voiceprint (noise / silence / wrong speaker) so a profile
isn't polluted.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from config import ENROLL_QUALITY_MIN

from .store.models import Voiceprint
from .store.store import VoiceprintStore


def identity_voiceprint_id(workspace_id: str, identity: str) -> str:
    """Deterministic voiceprint id for a known identity → repeat enrollments find it."""
    h = hashlib.sha1(f"{workspace_id}\x00{identity}".encode()).hexdigest()[:16]
    return "vp_" + h


@dataclass
class EnrollResult:
    status: str          # created | updated | rejected
    voiceprint_id: str
    reason: str = ""


def enroll(
    store: VoiceprintStore,
    embedder,
    workspace_id: str,
    identity: str,
    audio: np.ndarray,
    sample_rate: int = 16_000,
    duration_sec: float = 0.0,
    consumer: str = "recato",
) -> EnrollResult:
    vp_id = identity_voiceprint_id(workspace_id, identity)
    emb = embedder.extract(audio, sample_rate)
    if emb is None:
        return EnrollResult("rejected", vp_id, "audio too short / no embedding")

    existing = store.get(workspace_id, vp_id)

    # WS5 — user-supreme consent: if the data subject has disabled enrollment, FPM
    # refuses to create or strengthen their voiceprint regardless of workspace.
    if existing is not None and not existing.enroll_allowed:
        return EnrollResult("rejected", vp_id, "enrollment disabled by user")

    # owner_email is the roster identity (enroll's `identity` IS the person's email).
    if existing is None:
        vp = Voiceprint(vp_id, workspace_id, name=identity, owner_email=identity)
        vp.add_exemplar(emb)
        vp.recompute_centroid()
        vp.enroll_count = 1
        vp.quality_score = vp.compute_quality()
        vp.total_duration_sec = duration_sec
        store.upsert(vp)
        store.log_usage(workspace_id, vp_id, "enroll", consumer, "created voiceprint")
        return EnrollResult("created", vp_id)

    # consistency gate: reject clips that don't match the existing voiceprint
    if existing.centroid.size and float(emb @ existing.centroid) < ENROLL_QUALITY_MIN:
        return EnrollResult("rejected", vp_id, "inconsistent with existing voiceprint")

    existing.add_exemplar(emb)
    existing.recompute_centroid()
    existing.enroll_count += 1
    existing.quality_score = existing.compute_quality()
    existing.total_duration_sec += duration_sec
    existing.name = identity  # trusted roster identity
    if not existing.owner_email:
        existing.owner_email = identity
    store.upsert(existing)
    store.log_usage(workspace_id, vp_id, "enroll", consumer, "strengthened voiceprint")
    return EnrollResult("updated", vp_id)
