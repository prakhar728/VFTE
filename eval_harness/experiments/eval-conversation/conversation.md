# eval-conversation — readable script

Human-readable copy of this experiment's ground truth. The machine-readable canonical version
(what the scorer compares against) is `gold.json` in this same folder — keep the two in sync.

Read aloud and record (2 speakers, **A** and **B**), both on one mic — the in-person scenario.
Each line = one speaker turn, so the turn order below **is** the diarization ground truth. The
proper nouns are the vocab-biasing payoff (a plain ASR fumbles them; vocab biasing should fix them).

## Script A↔B (exact words)

1. **A:** Hey, did you finish the report for Recato?
2. **B:** Almost. I just need Priya to review the last part.
3. **A:** Cool. Is the meeting still at three?
4. **B:** Yeah, three at the Sunnyvale office. Are you driving?
5. **A:** I'll take the train. Want me to grab coffee?
6. **B:** Please — a cortado for me, thanks.
7. **A:** Got it. Did Arjun send the slides yet?
8. **B:** He sent them this morning. They look good.
9. **A:** Nice. Let's wrap up by four.
10. **B:** Sounds good. See you at three.

## Vocab list (biased run)
`Recato`, `Priya`, `Arjun`, `Sunnyvale`, `cortado` — also in `config.yaml` (`asr.vocab`).

## What it measures
- **Diarization (who-spoke-when):** the 10-turn A/B alternation is the reference. diart (real-time)
  vs DiariZen (batch) on the same clip → speaker-attribution accuracy + RTF + peak RAM.
- **Vocab (what-was-said):** transcribe twice — plain vs primed with the vocab list
  (`initial_prompt`). WER should drop on the proper nouns above.
- Short (~30–45s read) so it's quick to re-run.
