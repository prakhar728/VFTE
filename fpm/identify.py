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

Correction (C.5): live emissions are *provisional*. The session keeps a running
transcript; when a `local_speaker` finally locks, earlier provisional chunks for
that speaker are retro-relabelled to the resolved identity. `transcript()` is the
authoritative corrected view; `seal()` drains the stream, freezes corrections,
and returns the final transcript. An anonymous voiceprint can be named later via
the knowledge channel (`store.set_name`) and is then recognised by name.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

import config  # module-attr access so MIN_SEGMENT_SEC is monkeypatch-tunable in tests

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
        consumer: str = "offline",
        read_only: bool = False,
    ):
        self._store = store
        self._embedder = embedder
        self._diarizer = diarizer
        self._ws = workspace_id
        self._sr = sample_rate
        self._lock_min = lock_min_votes
        self._consumer = consumer
        # P1: live (diart) path — classify + vote-lock in memory for stable session
        # labels, but mint nothing and write nothing to the store (single-writer: the
        # post pass is the sole authoritative writer). Default False = offline writer.
        self._read_only = read_only

    # ── lifecycle ────────────────────────────────────────────

    def start(self) -> None:
        self._diarizer.start(self._ws)
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = 0                       # absolute sample index of buf[0]
        self._votes: dict[str, Counter] = {}
        self._exemplars: dict[str, list[np.ndarray]] = {}
        self._locked: dict[str, IdentifiedSegment] = {}  # local_speaker → resolved label
        self._history: list[IdentifiedSegment] = []      # running session transcript
        # A↔C bridge: incremental engines (diart/mock) emit segments while their audio is
        # still in a bounded trailing window. Batch engines (DiariZen) emit ALL segments at
        # finish(), so we must retain the whole clip to re-embed early ones. The hint is
        # read via getattr — set by the engine on its own class (A owns it); absent → bounded.
        # Memory for the unbounded case is bounded by the engine's own clip-length cap.
        self._max_buf = None if getattr(self._diarizer, "buffered_batch", False) \
            else int(BUFFER_SEC * self._sr)
        self._finished = False
        self._sealed = False

    def feed(self, chunk: np.ndarray, sample_rate: int = 16_000) -> list[IdentifiedSegment]:
        block = np.asarray(chunk, dtype=np.float32).ravel()
        self._append(block)
        return self._consume(self._diarizer.feed(block, sample_rate))

    def finish(self) -> list[IdentifiedSegment]:
        if self._finished:
            return []
        self._finished = True
        return self._consume(self._diarizer.finish())

    def _consume(self, segs: list[Segment]) -> list[IdentifiedSegment]:
        out = []
        for s in segs:
            r = self._identify(s)
            self._history.append(r)   # append before the next lock so relabel sees it
            out.append(r)
        return out

    def transcript(self) -> list[IdentifiedSegment]:
        """The corrected session transcript (relabels applied in place)."""
        return list(self._history)

    def seal(self) -> list[IdentifiedSegment]:
        """Drain the stream, freeze corrections, return the final transcript."""
        self.finish()
        self._sealed = True
        return self.transcript()

    # ── bounded trailing audio buffer ────────────────────────

    def _append(self, block: np.ndarray) -> None:
        if block.size:
            self._buf = np.concatenate([self._buf, block])
        # max_buf is None for batch engines → retain the whole clip (no trimming).
        if self._max_buf is not None and self._buf.size > self._max_buf:  # drop the stale front
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

        res = classify(emb, self._store.centroids(self._ws))
        votes = self._votes.setdefault(spk, Counter())
        votes[res.voiceprint_id if res.decision == "MATCH" else _UNKNOWN] += 1  # vote ALWAYS (ungated)

        # P3 gate: only strong spans contribute an exemplar to a (future) minted centroid.
        # Voting above is untouched, so weak speakers still vote-lock / MATCH normally.
        if self._passes_gate(seg, res):
            self._exemplars.setdefault(spk, [])
            if len(self._exemplars[spk]) < 20:
                self._exemplars[spk].append(emb)

        locked = self._maybe_lock(spk, res.confidence)
        if locked is not None:
            return IdentifiedSegment(seg.start, seg.end, spk, locked.voiceprint_id,
                                     locked.name, "LOCKED", locked.confidence)

        # provisional (pre-lock) label — reflects the current classify result
        if res.decision == "MATCH":
            name = self._name_of(res.voiceprint_id)  # None if user opted to stay anonymous
            return IdentifiedSegment(seg.start, seg.end, spk, res.voiceprint_id, name,
                                     "MATCH" if name is not None else "ANON", res.confidence)
        return IdentifiedSegment(seg.start, seg.end, spk, None, None, res.decision, res.confidence)

    def _maybe_lock(self, spk: str, confidence: float) -> IdentifiedSegment | None:
        votes = self._votes[spk]
        cand, count = votes.most_common(1)[0]
        # clear leader once it reaches the vote floor (no tie with the runner-up)
        runner = votes.most_common(2)[1][1] if len(votes) > 1 else 0
        if count < self._lock_min or count == runner:
            return None

        if cand == _UNKNOWN:
            if self._read_only:
                # P1 read-only: mint nothing — lock to a stable session-local label
                # (voiceprint_id=None), kept stable via local_speaker.
                vp_id = None
            elif self._exemplars.get(spk):
                # P3 gate: mint only with >=1 gate-passing exemplar. A speaker whose every
                # span was sub-floor has none → don't lock; keep voting (may MATCH later).
                vp_id = self._mint_anonymous(spk)
            else:
                return None
            label = IdentifiedSegment(0, 0, spk, vp_id, None, "ANON", confidence)
        else:
            # WS4 ledger: a known speaker locking in IS a use of their voiceprint.
            # P1 read-only writes nothing — skip the ledger row (still resolve for display).
            if not self._read_only:
                self._store.log_usage(self._ws, cand, "identify", self._consumer, "matched in meeting")
            name = self._name_of(cand)
            # WS5: identify_allowed=False ⇒ "stay anonymous" — keep the cluster, drop the name.
            decision = "MATCH" if name is not None else "ANON"
            label = IdentifiedSegment(0, 0, spk, cand, name, decision, confidence)
        self._locked[spk] = label
        self._relabel_history(spk, label)
        return label

    def _relabel_history(self, spk: str, label: IdentifiedSegment) -> int:
        """Retro-correct earlier provisional chunks for a speaker that just locked."""
        if self._sealed:
            return 0
        n = 0
        for h in self._history:
            if h.local_speaker == spk and h.voiceprint_id != label.voiceprint_id:
                h.voiceprint_id = label.voiceprint_id
                h.name = label.name
                h.decision = "RELABELED"
                n += 1
        return n

    def _passes_gate(self, seg: Segment, res) -> bool:
        """P3 quality gate for exemplar-append (and, via accumulated exemplars, mint).

        A span must be long enough to embed reliably and not be AMBIGUOUS (top-2 too
        close → likely overlapped speech, which would pollute a centroid). LOW stays
        admissible so a genuinely new speaker who scores near an existing centroid can
        still mint. Reads `config.MIN_SEGMENT_SEC` via module attr so tests can tune it.
        """
        if seg.duration < config.MIN_SEGMENT_SEC:
            return False
        if res.decision == "AMBIGUOUS":
            return False
        return True

    def _mint_anonymous(self, spk: str) -> str:
        vp = Voiceprint(new_voiceprint_id(), self._ws, name="")
        for e in self._exemplars.get(spk, []):
            vp.add_exemplar(e)
        vp.recompute_centroid()
        vp.quality_score = vp.compute_quality()
        self._store.upsert(vp)
        return vp.voiceprint_id

    def _name_of(self, vp_id: str | None) -> str | None:
        """Human name for a voiceprint, or None when the subject opted to stay anonymous
        (WS5 identify_allowed=False) — the cluster persists, the name is withheld."""
        if not vp_id:
            return None
        if not self._store.identify_allowed(self._ws, vp_id):
            return None
        vp = self._store.get(self._ws, vp_id)
        return vp.name or None if vp else None
