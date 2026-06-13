# eval-coffee — readable script

Human-readable copy of this experiment's ground truth (tag: `initial-testing`). The canonical
machine-readable version the scorer uses is `gold.json` in this same folder — keep them in sync.

Read aloud and record (2 speakers, **A** and **B**), both on one mic — the in-person scenario.
Each line = one speaker turn, so the turn order below **is** the diarization ground truth. Keep it
casual and quick — this script is the looser, faster-speech counterpoint to the standup.

## Script A↔B (exact words)

1. **A:** Hey, are we still on for lunch at Tartine?
2. **B:** Yeah, noon works. Is Meera joining us?
3. **A:** She might. She's flying in from Seattle tonight.
4. **B:** Nice. Did you try that new place in the Mission?
5. **A:** The taco spot? It was incredible. Get the carnitas.
6. **B:** Noted. I'll bring the book you lent Rohan.
7. **A:** Oh, the one about Patagonia? Keep it, no rush.
8. **B:** Thanks. See you at noon then.
9. **A:** Sounds good. I'll grab us a table.
10. **B:** Perfect, see you there.

## Vocab list (biased run)
`Tartine`, `Meera`, `Rohan`, `Seattle`, `Mission`, `Patagonia` — also in `config.yaml`
(`asr.vocab`).

## What it measures
Same two axes as the other initial-testing scripts — diarization (A/B turn structure) and the
vocab WER delta — under casual, faster, more naturally overlapping speech.
