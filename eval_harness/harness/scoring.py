"""Scoring vs gold — transcription WER + speaker-attribution accuracy.

Gold is a turn list (speaker + text, turn order = the diarization reference, no timestamps). The
canonical form is `gold.json` (`{"turns": [{"speaker","text"}, ...]}`); a plain `SPEAKER: text`
`.txt` is still accepted. From either:
- **WER** via jiwer on normalized text (vocab-on vs vocab-off → the delta is the vocab win).
- **Speaker accuracy**: align hypothesis words to gold words (jiwer alignment), map predicted
  speakers → gold speakers (greedy by co-occurrence), and score the fraction of aligned words given
  the right speaker. (Time-based DER needs a timestamped gold — a future add.)
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import jiwer

_PUNCT = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    """Lowercase, punctuation → spaces, collapse whitespace (so WER ignores casing/punctuation)."""
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


def wer(gold_text: str, hyp_text: str) -> float:
    g, h = normalize(gold_text), normalize(hyp_text)
    if not g:
        return 0.0 if not h else 1.0
    return float(jiwer.wer(g, h))


def parse_gold(gold_text: str) -> list[tuple[str, str]]:
    """`A: hello` lines → [(speaker, text), ...]."""
    turns = []
    for line in gold_text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        spk, text = line.split(":", 1)
        turns.append((spk.strip(), text.strip()))
    return turns


def parse_gold_json(raw: str) -> list[tuple[str, str]]:
    """`{"turns": [{"speaker","text"}, ...]}` → [(speaker, text), ...]."""
    data = json.loads(raw)
    return [(str(t["speaker"]).strip(), str(t["text"]).strip()) for t in data.get("turns", [])]


def load_gold(path: str | Path) -> list[tuple[str, str]]:
    """Load gold turns from a file — `.json` (canonical) or `SPEAKER: text` `.txt`."""
    path = Path(path)
    raw = path.read_text()
    return parse_gold_json(raw) if path.suffix == ".json" else parse_gold(raw)


def _words_with_speakers(turns: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """[(speaker, text)] → flat [(normalized_word, speaker)]."""
    out = []
    for spk, text in turns:
        for w in normalize(text).split():
            out.append((w, spk))
    return out


def gold_text(gold_turns: list[tuple[str, str]]) -> str:
    return " ".join(t for _, t in gold_turns)


def speaker_accuracy(gold_turns: list[tuple[str, str]], hyp_turns) -> dict:
    """Align hyp words ↔ gold words, map predicted→gold speakers, score attribution.

    hyp_turns: objects with .speaker and .text (merge.Turn). Returns {accuracy, mapping, aligned}.
    """
    gold_ws = _words_with_speakers(gold_turns)
    hyp_ws = _words_with_speakers([(t.speaker, t.text) for t in hyp_turns])
    if not gold_ws or not hyp_ws:
        return {"accuracy": 0.0, "mapping": {}, "aligned": 0}

    out = jiwer.process_words(" ".join(w for w, _ in gold_ws), " ".join(w for w, _ in hyp_ws))
    gold_spk = [s for _, s in gold_ws]
    hyp_spk = [s for _, s in hyp_ws]
    pairs: list[tuple[str, str]] = []                 # (gold_speaker, hyp_speaker) for aligned words
    for chunk in out.alignments[0]:
        if chunk.type in ("equal", "substitute"):
            for gi, hi in zip(range(chunk.ref_start_idx, chunk.ref_end_idx),
                              range(chunk.hyp_start_idx, chunk.hyp_end_idx)):
                pairs.append((gold_spk[gi], hyp_spk[hi]))
    if not pairs:
        return {"accuracy": 0.0, "mapping": {}, "aligned": 0}

    # greedy: each predicted speaker → the gold speaker it co-occurs with most
    co: dict[str, Counter] = defaultdict(Counter)
    for g, h in pairs:
        co[h][g] += 1
    mapping = {h: c.most_common(1)[0][0] for h, c in co.items()}
    correct = sum(1 for g, h in pairs if mapping.get(h) == g)
    return {"accuracy": correct / len(pairs), "mapping": mapping, "aligned": len(pairs)}
