# Ground-truth eval conversation (diarization + vocab test)

Read aloud and record (2 speakers, **A** and **B**). Each line = one speaker turn, so the turn
order below **is** the diarization ground truth. Score real-time diart vs batch DiariZen against
it, and score Whisper **with** the FPM vocab vs **without** it.

Simple, natural conversation — the vocab payoff is the proper nouns (a plain ASR fumbles them;
vocab biasing should fix them).

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

## Vocab list (put these in the FPM vocab for the "biased" run)
`Recato`, `Priya`, `Arjun`, `Sunnyvale`, `cortado`

## How to use
- **Diarization (who-spoke-when):** the 10-turn A/B alternation is the reference. Run real-time
  diart and batch DiariZen on the recording; compare to this turn structure (DER + peak RAM + RTF,
  same clip = the fair head-to-head).
- **Vocab (what-was-said):** transcribe twice — Whisper plain vs Whisper primed with the vocab
  list (initial_prompt / hotwords). WER should drop on the proper nouns above, demonstrating the
  "vocab from prior transcript corrections" enrichment.
- Short (~30–45s read) so it's quick to re-run.
