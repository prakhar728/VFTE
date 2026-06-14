# In-person capture via phone P2P (idea note)

**One-liner:** an app where, when someone starts an in-person meeting, **nearby phones join with a
single tap** (AirDrop / "Share WiFi password" style) and each records. Multiple phones = a
**distributed mic array** → better diarization, better transcripts, and (bonus) identity for free.

## Why it's powerful
- **Recovers the gmeet model for the physical room.** Each phone is a person's **near-field mic**
  → that person dominates their own channel → do *structural* per-channel separation instead of
  hard *acoustic* diarization on one muddy mixed mic (today's weak link: ~33% DER single-mic).
- **Identity for free** — each phone is logged into an account = the owner's **email/identity**.
  Solves the "couldn't get attendee emails" problem; no voiceprint guessing for channel owners.
- **Better ASR** — near-field = high SNR; Whisper degrades hard on far-field/overlap.
- Doesn't need one-phone-per-person: even a **few scattered phones** as extra mics improve plain
  acoustic diarization ("more mics → better"). Non-app people fall back to FPM acoustic diarize.

## Join UX (the easy, delightful part)
- Proximity tap-to-join via **Multipeer Connectivity** (iOS↔iOS, offline, BLE+peer-WiFi). Android
  equivalent = Nearby Connections.
- Proximity collapses **discovery + consent** into one gesture — ideal for a confidentiality product
  (physical presence + explicit tap = consent).
- **Cross-platform is the friction:** Multipeer (iOS-only) and Nearby (Android-only) don't interop →
  need a common BLE/local-net layer or a cloud "join code" for mixed rooms. iOS-only avoids this.

## Hard parts (the real engineering)
1. **Clock sync** across independent devices — align channels post-hoc by **cross-correlation**
   (all phones hear the same room). Known technique, but make-or-break for the merge.
2. **Cross-channel attribution** — every phone hears everyone (near + far bleed). v1 = attribute to
   the **loudest channel** / near-field dominance; general case (beamforming/TDOA, heavy overlap) is
   research-y — avoid for v1.
3. **Partial coverage / backgrounding / mic perms** — graceful degradation; iOS background limits.

## What it reuses (already built)
- **Recato** transcribes each channel. **FPM** already does single-channel diarize + identify
  (post-process) + the voiceprint store (links a person across channels/meetings). FPM's acoustic
  diarizer shifts from workhorse → **safety net** (verify channel owner, cover non-app people).

## Effort (rough, iOS-only)
- **MVP ~6–10 weeks** (1 solid iOS dev + existing backend): record app + Multipeer join + N-phone
  record/upload + cross-correlation alignment + simple loudest-channel attribution. Mostly assembly;
  the one real DSP problem is sync.
- The honest unknown isn't the *building* — it's **how much multi-mic actually improves real-room
  diarization**.

## Smart first step (merit-over-effort)
Before the polished proximity UX: a **2–3 week spike** — 3 phones record a real noisy room (even
manual start), upload, cross-correlate to align, run diarization, and **measure DER vs the single-mic
33%**. If it drops hard with cheap scattered phones, the whole direction is de-risked.
