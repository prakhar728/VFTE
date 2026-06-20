# Diarization Flow

Source of truth: `FPM/docs/build/ARCHITECTURE.md`. This is the visual companion — when
transcription happens, when diarization happens, where raw audio goes.

The spine: **identity lives on the voiceprint, never on the transcript**, and the
**display name is always a projection** (`voiceprint_id → owner_email → name`). Two
diarization engines, **one identity store, one writer**.

```mermaid
flowchart TD
    %% ================= CAPTURE =================
    subgraph CAP["🎙️ Capture (in-person)"]
        MIC["Browser Record button<br/>single-mic audio clip"]
    end
    MIC -->|"audio stream"| ING["Conclave Record ingress<br/>api/record_routes.py"]

    %% ================= LIVE (during meeting) =================
    subgraph LIVE["⏱️ LIVE — during the meeting (provisional, read-only)"]
        direction TB
        L_DIA["diart (StreamingDiarizer)<br/>emits {start, end, local_speaker}<br/>NO embeddings / ids / text"]
        L_ASR["NEAR-Whisper ASR<br/>(faster-whisper, in TEE)<br/>→ text segments"]
        L_ID["Identity (read-only):<br/>CAM++ re-embed → match.classify<br/>vs enrolled centroids → vote-lock<br/>MINTS nothing, WRITES nothing"]
        L_MERGE["merge_by_timestamp<br/>[speaker] text"]
        L_DIA --> L_ID
        L_ID --> L_MERGE
        L_ASR --> L_MERGE
    end
    ING -->|"live stream (∥)"| L_DIA
    ING -->|"live stream (∥)"| L_ASR
    L_MERGE -->|"provisional transcript<br/>+ provisional labels"| DISPLAY["Live transcript shown in UI"]

    %% ================= POST (on recording complete) =================
    subgraph POST["✅ POST — on recording complete (authoritative)"]
        direction TB
        P_DIA["DiariZen (StreamingDiarizer)<br/>accurate diarize<br/>SOLE authoritative writer"]
        P_ASR["ASR re-run → text segments"]
        P_ID["Identity (read+write):<br/>CAM++ re-embed → classify → vote-lock<br/>confidence-gated mint/update<br/>anonymous voiceprint for unknowns"]
        P_MERGE["merge_by_timestamp<br/>→ deterministic Speaker N<br/>(numbered by voiceprint_id)"]
        P_DIA --> P_ID
        P_DIA --> P_MERGE
        P_ASR --> P_MERGE
    end
    ING -->|"returned audio file"| RAW
    RAW -->|"full clip (∥)"| P_DIA
    RAW -->|"full clip (∥)"| P_ASR
    P_ID -->|"mint / update<br/>(confidence-gated)"| STORE
    P_MERGE -->|"REPLACES live transcript<br/>persists resolved_speakers{voiceprint_id,name,conf}"| DISPLAY

    %% ================= RAW AUDIO =================
    RAW["🔒 Raw audio at rest<br/>TEE-sealed / encrypted volume<br/>retained for transcript-lifetime<br/>delete cascades"]

    %% ================= IDENTITY STORE =================
    subgraph IDENT["🧬 Identity store (FPM)"]
        STORE["Voiceprints<br/>(CAM++ centroids + exemplars)<br/>identity NEVER on transcript"]
    end
    L_ID -. "reads centroids (stale cache OK)" .-> STORE

    %% ================= TRUST / CONSENT =================
    subgraph TRUST["🤝 Trust handshake + consent"]
        TAG["Host tags attendee<br/>(name + email) → pending proposal"]
        MAIL["FPM emails tagged address<br/>(notify-only, no content)"]
        CONFIRM["Person signs into consent dashboard<br/>confirm / deny (context-only, no audio)"]
        RESOLVE["owner_email set →<br/>re-resolve propagates name<br/>across ALL stored transcripts"]
        TAG --> MAIL --> CONFIRM -->|"confirm<br/>(self-id auto-confirms)"| RESOLVE
    end
    STORE -->|"voiceprint_id"| TAG
    RESOLVE -->|"name projection<br/>(passes live consent gate)"| DISPLAY
    STORE -.->|"projection at read-time<br/>voiceprint_id → owner_email → name"| DISPLAY

    %% ================= IMPLEMENTATION STATUS (verified against code 2026-06-19) =================
    classDef done    fill:#c8e6c9,stroke:#2e7d32,stroke-width:1px,color:#102610;
    classDef partial fill:#ffe0b2,stroke:#ef6c00,stroke-width:1px,color:#3d2400;
    classDef todo    fill:#ffcdd2,stroke:#c62828,stroke-width:2px,stroke-dasharray:4 3,color:#3d0a0a;

    %% GREEN — implemented & wired
    class MIC,ING,P_DIA,P_ASR,P_ID,P_MERGE,STORE,TAG,MAIL,CONFIRM,RESOLVE done;
    %% RED — the LIVE pass: no live/streaming execution path exists (engines exist in code, never wired live)
    class L_DIA,L_ASR,L_ID,L_MERGE todo;
    %% AMBER — partial: built/runs but not as drawn
    class RAW,DISPLAY partial;
```

**Status colors** (verified against the code on 2026-06-19):

| Color | Meaning |
|---|---|
| 🟩 **green** | Implemented & wired end-to-end |
| 🟧 **amber** | Partial — exists/runs, but not fully as drawn |
| 🟥 **red (dashed)** | Not implemented as a live path |

**What's actually built vs. drawn:**

- 🟩 **The whole POST pass + capture + identity + consent loop is real.** Browser Record → `record_routes.py` (ffmpeg → FPM `/v1/diarize` ∥ NEAR-Whisper ASR → `merge_by_timestamp` → `resolved_speakers`) runs as a single **offline** pass on the finished clip. DiariZen is the default engine; identity (CAM++ re-embed, `match.classify`, vote-lock, confidence gate, anonymous mint) and the full host-tag → FPM-email → confirm → re-resolve handshake are all on `main`.
- 🟥 **The entire LIVE column is not operational.** Conclave only ever calls FPM with `tag=offline` on the whole clip — there is **no streaming/during-meeting diarization, no live ASR, no live merge**. The `diart` engine and the `read_only` "live" identity mode *exist in FPM code* but are never invoked as a live path (and `diart` isn't even the default engine), so functionally the live pass doesn't run.
- 🟧 **RAW audio at rest** — audio is decoded **in-memory per request and never persisted**; voiceprints are AES-encrypted at rest, but there is no audio-at-rest sealing, transcript-lifetime retention, or delete-cascade to audio/transcript.
- 🟧 **DISPLAY** — the **final** transcript is shown in the UI; the **live/provisional** transcript and the live→post "replace" reconciliation are not (the record UI shows only an elapsed-time counter while recording).

> **Net:** what's implemented is essentially a clean single-pass **offline** diarization + the consent/identity spine. The "two-engine live + post" split this diagram depicts is currently **post-only** — the live half is the unbuilt portion (matches the P1 "live diart read-only" branch in `build/ARCHITECTURE.md`, which is not on `main`).

## Reading it in one breath

1. **Capture** — single mic, browser Record → Conclave ingress.
2. **Transcription happens twice** — ASR runs in *both* the live and post passes,
   always in **parallel (∥)** with the diarizer, never as a pipeline stage after it.
   Diarizer and ASR are merged by timestamp into `[speaker] text`.
3. **Diarization happens twice, two engines:**
   - **LIVE = `diart`** — provisional, **read-only**: labels speakers for display,
     mints nothing, writes nothing.
   - **POST = `DiariZen`** — runs on the returned audio file, **sole authoritative
     writer**: accurate diarize + identify, confidence-gated mint/update, and its
     result **replaces** the live transcript. One writer ⇒ no cache-coherence problem.
4. **Raw audio** flows to a **TEE-sealed / encrypted** store, feeds the post pass,
   is retained for transcript-lifetime, and delete cascades (audio + transcript + voiceprint).
5. **Identity is never on the transcript** — it's a voiceprint id; the **name is a
   read-time projection** `voiceprint_id → owner_email → name`, set only after the
   email-bound consent handshake confirms.
