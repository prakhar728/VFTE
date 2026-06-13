# eval-standup — readable script

Human-readable copy of this experiment's ground truth (tag: `initial-testing`). The canonical
machine-readable version the scorer uses is `gold.json` in this same folder — keep them in sync.

Read aloud and record (2 speakers, **A** and **B**), both on one mic — the in-person scenario.
Each line = one speaker turn, so the turn order below **is** the diarization ground truth. This
script leans into denser proper nouns + tech jargon (the harder vocab-biasing case).

## Script A↔B (exact words)

1. **A:** Morning. Did the Recato build pass on staging?
2. **B:** It did, but Kavya flagged a flaky test in the auth module.
3. **A:** Is that blocking the Bangalore demo on Friday?
4. **B:** Not yet. I'll rerun it and check the metrics in Grafana.
5. **A:** Okay. Can you sync with Devansh about the dashboard?
6. **B:** Sure. He's pushing the Kubernetes config this afternoon.
7. **A:** Great. Let's keep the standup to ten minutes.
8. **B:** Agreed. I'll post the notes in the channel.
9. **A:** Perfect. Ping me if the test fails again.
10. **B:** Will do. Talk later.

## Vocab list (biased run)
`Recato`, `Kavya`, `Devansh`, `Bangalore`, `Grafana`, `Kubernetes` — also in `config.yaml`
(`asr.vocab`).

## What it measures
Same two axes as the other initial-testing scripts — diarization (A/B turn structure) and the
vocab WER delta — but with a heavier proper-noun / jargon load to stress the biasing.
