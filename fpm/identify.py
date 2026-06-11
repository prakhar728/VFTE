"""Real-time identify pipeline — the offline path's brain (C.4).

Wires a `StreamingDiarizer` (diart or mock) to the fixed CAM++ ID layer:

    audio chunks ─▶ diarizer.feed ─▶ {start,end,local_speaker}
                                          │  slice that span from a bounded buffer
                                          ▼
                          CAM++ re-embed (the FIXED ID embedder)
                                          ▼
                          classify vs the workspace's enrolled centroids
                                          ▼
              vote-lock  local_speaker ─▶ voiceprint_id (+name | anonymous)

The diarizer's labels are session-local and can wobble early; we accumulate a
*vote* per `local_speaker` and **lock** it to a voiceprint once the evidence is
clear (VoxTerm's vote-lock pattern) so live labels stabilize instead of
flickering. A `local_speaker` that the store doesn't know gets an **anonymous**
voiceprint minted into the store (name="") — recognizable in future sessions,
nameable later via the knowledge channel.

Bounded memory: only a trailing window of audio is retained (enough to slice the
longest in-flight segment); per-session maps are sized by speaker count, not
stream length — so state stays flat over an arbitrarily long meeting.

Engine-independent: identity ALWAYS re-embeds with CAM++ from the raw segment
audio; the diarizer's own embeddings never enter the store.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .diarize.base import Segment, StreamingDiarizer
from .match import classify
from .store.models import Voiceprint
from .store.store import VoiceprintStore, new_voiceprint_id

BUFFER_SEC = 15.0          # trailing audio kept for slicing (> diarizer max span + latency)
LOCK_MIN_VOTES = 2         # agreeing segments before a local_speaker locks to an id
_UNKNOWN = "__unknown__"   # vote sentinel for not-in-store


@dataclass
class IdentifiedSegment:
    start: float
    end: float
    local_speaker: str
    voiceprint_id: str | None   # resolved voiceprint (enrolled or anonymous), None if undecided
    name: str | None            # human name if known, else None (anonymous / undecided)
    decision: str               # MATCH | ANON | AMBIGUOUS | LOW | UNKNOWN | PENDING | LOCKED
    confidence: float


class SessionIdentifier:
    """One live offline session: diarize → identify → emit, with a vote-lock map."""

    def __init__(
        self,
        store: VoiceprintStore,
        embedder,
        diarizer: StreamingDiarizer,
        workspace_id: str,
        *,
        sample_rate: int = 16_000,
        lock_min_votes: int = LOCK_MIN_VOTES,
    ):
        self._store = store
        self._embedder = embedder
        self._diarizer = diarizer
        self._ws = workspace_id
        self._sr = sample_rate
        self._lock_min = lock_min_votes

    # ── lifecycle ────────────────────────────────────────────

    def start(self) -> None:
        self._diarizer.start(self._ws)
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = 0                       # absolute sample index of buf[0]
        self._votes: dict[str, Counter] = {}
        self._exemplars: dict[str, list[np.ndarray]] = {}
        self._locked: dict[str, IdentifiedSegment] = {}  # local_speaker → resolved label
        self._max_buf = int(BUFFER_SEC * self._sr)

    def feed(self, chunk: np.ndarray, sample_rate: int = 16_000) -> list[IdentifiedSegment]:
        block = np.asarray(chunk, dtype=np.float32).ravel()
        self._append(block)
        return [self._identify(s) for s in self._diarizer.feed(block, sample_rate)]

    def finish(self) -> list[IdentifiedSegment]:
        return [self._identify(s) for s in self._diarizer.finish()]

    # ── bounded trailing audio buffer ────────────────────────

    def _append(self, block: np.ndarray) -> None:
        if block.size:
            self._buf = np.concatenate([self._buf, block])
        if self._buf.size > self._max_buf:                  # drop the stale front
            drop = self._buf.size - self._max_buf
            self._buf = self._buf[drop:]
            self._buf_start += drop

    def _slice(self, start_sec: float, end_sec: float) -> np.ndarray | None:
        s = int(start_sec * self._sr) - self._buf_start
        e = int(end_sec * self._sr) - self._buf_start
        s, e = max(0, s), min(self._buf.size, e)
        if e <= s:
            return None
        return self._buf[s:e]

    # ── identification ───────────────────────────────────────

    def _identify(self, seg: Segment) -> IdentifiedSegment:
        spk = seg.local_speaker

        # already locked → emit the stable label without re-deciding
        if spk in self._locked:
            r = self._locked[spk]
            return IdentifiedSegment(seg.start, seg.end, spk, r.voiceprint_id, r.name,
                                     "LOCKED", r.confidence)

        audio = self._slice(seg.start, seg.end)
        emb = self._embedder.extract(audio, self._sr) if audio is not None else None
        if emb is None:                                     # too short to ID yet
            return IdentifiedSegment(seg.start, seg.end, spk, None, None, "PENDING", 0.0)

        self._exemplars.setdefault(spk, [])
        if len(self._exemplars[spk]) < 20:
            self._exemplars[spk].append(emb)

        res = classify(emb, self._store.centroids(self._ws))
        votes = self._votes.setdefault(spk, Counter())
        votes[res.voiceprint_id if res.decision == "MATCH" else _UNKNOWN] += 1

        locked = self._maybe_lock(spk, res.confidence)
        if locked is not None:
            return IdentifiedSegment(seg.start, seg.end, spk, locked.voiceprint_id,
                                     locked.name, "LOCKED", locked.confidence)

        # provisional (pre-lock) label — reflects the current classify result
        if res.decision == "MATCH":
            name = self._name_of(res.voiceprint_id)
            return IdentifiedSegment(seg.start, seg.end, spk, res.voiceprint_id, name,
                                     "MATCH", res.confidence)
        return IdentifiedSegment(seg.start, seg.end, spk, None, None, res.decision, res.confidence)

    def _maybe_lock(self, spk: str, confidence: float) -> IdentifiedSegment | None:
        votes = self._votes[spk]
        cand, count = votes.most_common(1)[0]
        # clear leader once it reaches the vote floor (no tie with the runner-up)
        runner = votes.most_common(2)[1][1] if len(votes) > 1 else 0
        if count < self._lock_min or count == runner:
            return None

        if cand == _UNKNOWN:
            vp_id = self._mint_anonymous(spk)
            label = IdentifiedSegment(0, 0, spk, vp_id, None, "ANON", confidence)
        else:
            label = IdentifiedSegment(0, 0, spk, cand, self._name_of(cand), "MATCH", confidence)
        self._locked[spk] = label
        return label

    def _mint_anonymous(self, spk: str) -> str:
        vp = Voiceprint(new_voiceprint_id(), self._ws, name="")
        for e in self._exemplars.get(spk, []):
            vp.add_exemplar(e)
        vp.recompute_centroid()
        vp.quality_score = vp.compute_quality()
        self._store.upsert(vp)
        return vp.voiceprint_id

    def _name_of(self, vp_id: str | None) -> str | None:
        if not vp_id:
            return None
        vp = self._store.get(self._ws, vp_id)
        return vp.name or None if vp else None
