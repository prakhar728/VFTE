# Drop the original recording in THIS folder

Put the long 3-person conversation file right here, e.g.:

    eval_harness/fingerprint_runs/3person-persistence/source/conversation.m4a

- Any container ffmpeg reads is fine: `.m4a`, `.mp3`, `.wav`, `.aac`, `.flac`, `.ogg`, `.opus`, `.mp4`.
- The file is **gitignored** (it can be large / sensitive) — only `metadata.yaml` + the READMEs commit.
- Drop exactly **one** audio file here; the splitter auto-detects it. (This `.md` is ignored by the
  splitter.)

Once it's here, the next step splits it into `../segments/` per `metadata.yaml`
(`enroll_sec: 120`, `chunk_sec: 120`):

    00_enroll_000-120s.wav   ← first 2 min, the fingerprint source
    01_test_120-240s.wav     ← "future minutes" to identify
    02_test_240-360s.wav
    …
