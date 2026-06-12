"""Merge transcript + diarization by timestamp — the "who said what" stitch.

ASR (Whisper) gives words with timestamps; the diarizer gives speaker spans with timestamps; they
share one clock (same audio file). Each word is attributed to the speaker span it overlaps most
(midpoint fallback when there's no overlap), then consecutive same-speaker words become a turn.
This is the standard ASR+diarization fusion (whisperX-style), the same shape the production
Recato-hub would do — here collapsed for the eval.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AttributedWord:
    word: str
    start: float
    end: float
    speaker: str


@dataclass
class Turn:
    speaker: str
    text: str
    start: float
    end: float


def attribute_words(words, spk_segments) -> list[AttributedWord]:
    """Assign each word (objects with .word/.start/.end) to a speaker span (.start/.end/.local_speaker)."""
    spans = [(s.start, s.end, s.local_speaker) for s in spk_segments]
    out: list[AttributedWord] = []
    for w in words:
        speaker = _best_speaker(w.start, w.end, spans)
        out.append(AttributedWord(w.word, w.start, w.end, speaker))
    return out


def _best_speaker(w_start: float, w_end: float, spans) -> str:
    best, best_ov = None, 0.0
    for s_start, s_end, label in spans:
        ov = max(0.0, min(w_end, s_end) - max(w_start, s_start))
        if ov > best_ov:
            best_ov, best = ov, label
    if best is not None:
        return best
    # no overlap → nearest span by midpoint distance
    mid = (w_start + w_end) / 2
    nearest, nd = "?", float("inf")
    for s_start, s_end, label in spans:
        d = 0.0 if s_start <= mid <= s_end else min(abs(mid - s_start), abs(mid - s_end))
        if d < nd:
            nd, nearest = d, label
    return nearest


def group_turns(attributed: list[AttributedWord]) -> list[Turn]:
    """Collapse consecutive same-speaker words into turns (time-ordered)."""
    turns: list[Turn] = []
    for aw in sorted(attributed, key=lambda x: (x.start, x.end)):
        if turns and turns[-1].speaker == aw.speaker:
            t = turns[-1]
            t.text = (t.text + aw.word) if aw.word.startswith(" ") else (t.text + " " + aw.word)
            t.end = max(t.end, aw.end)
        else:
            turns.append(Turn(aw.speaker, aw.word.strip(), aw.start, aw.end))
    for t in turns:
        t.text = t.text.strip()
    return turns


def merge(words, spk_segments) -> list[Turn]:
    """words (Whisper) + spk_segments (diarizer) → attributed turns."""
    return group_turns(attribute_words(words, spk_segments))
