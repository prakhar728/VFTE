# Diarization research — benchmarks, online/streaming, preprocessing

Scope: state-of-the-art numbers, existing **online/low-latency** diarization implementations (the
chunked + cross-chunk-identity idea), and whether **audio preprocessing helps**. Aimed at the FPM
in-person / single-mic fallback (CPU, TEE, real-time). Numbers below are from vendor/secondary
sources — **verify against model cards before quoting externally** (datasets + metric definitions
vary). Compiled 2026-06-13.

---

## TL;DR

1. **diart is NOT SOTA — it's the *streaming* engine.** DiariZen ≈ open-source SOTA for *offline*.
   **Streaming Sortformer** (NVIDIA, Aug 2025) is the SOTA *streaming* model — but **GPU-native, and a
   GPU TEE is NOT affordable → production is CPU-bound.** So Sortformer is an **eval-only
   reference/ceiling** (run it locally on GPU to know the gap), **not the production path.** The
   production direction for the single-mic fallback is **CPU diart-style *streaming* DiariZen + the
   chunk-size elbow** (item #1 below).
2. **Your chunked + ephemeral-fingerprint idea is a real, published architecture — you're not
   inventing it.** It's literally diart's design (rolling buffer + incremental clustering +
   *cannot-link* constraints), and Streaming Sortformer's **Arrival-Order Speaker Cache (AOSC)** is
   the same idea done with a neural cache. So the question isn't "is it possible" — it's "reimplement
   diart's pattern with a better per-chunk engine, or adopt Sortformer (if GPU)."
3. **The latency↔accuracy curve is real and tunable** (diart updates a rolling buffer every ~500 ms;
   lower latency → higher DER). Your "elbow" exists; find it with a chunk-size sweep on labeled data.
4. **Preprocessing helps** — denoise + a strong VAD cut *missed-speech* and sharpen speaker
   boundaries → lower DER. Cheap win. **But don't over-denoise** (it distorts voice and hurts the
   speaker embedding). On single-mic you get denoise + VAD; beamforming needs multi-mic.
5. **cpWER is the cross-vendor metric.** AssemblyAI/Deepgram report it; Otter doesn't (transcript-only).
   Vendor self-reports are **not apples-to-apples** — use an independent suite (SDBench) or run them
   yourself on one dataset.
6. **For our constraints (production CPU-bound — GPU TEE unaffordable):** the single-mic fallback path
   is **CPU diart-style *streaming* DiariZen + chunk-size elbow** (diart's rolling-buffer/cannot-link
   glue, DiariZen front-end, FPM matcher as the cache). Add a VAD/denoise front-end. Use GPU **locally
   only** to benchmark Sortformer as the ceiling. And remember: even SOTA hits the rapid-exchange wall
   on mono — the real fix is the **multi-phone path (§5)**, which sidesteps diarization entirely.
7. **THE PLAN (§5): engineer the diarization away.** Each person opens a **browser link** on their
   phone → near-field capture → stream to the TEE → per-stream ASR + **FPM as the cross-talk gate** →
   **transcript-merge** on a coarse-synced clock. Device-login = free identity *and* clean enrollment.
   No install, no laptop, no meeting room. Eval: **1-phone vs 2-phone, where the 2-phone run is the
   pseudo-gold that grades the 1-phone run** (solves the labeling blocker). TDOA/audio-merge = dropped.

---

## 1. Benchmark numbers (DER ↓ unless noted; treat as indicative)

| System | Type | Reported numbers | Notes |
|---|---|---|---|
| **pyannoteAI** (premium) | offline | DER **≈11.2%** (best overall in one study) | commercial |
| **DiariZen** (wavlm-large-s80) | offline | DER **≈13.3%**; VoxConverse **5.2%** | open, MIT |
| **DiariZen-large-s80-v2** | offline | AMI-SDM **13.9**, DIHARD3 **14.5**, VoxConverse **9.1**, AISHELL-4 **10.1**, AliMeeting-far **10.8** | newer release |
| **Sortformer v2 (streaming)** | **streaming** | ALI **7.0%**; beats pyannote 3.1 on VoxConverse; strong on **short utterances / rapid turns** | **GPU-only** |
| **diart** | **streaming** | ~30%+ on hard sets (accuracy traded for latency) | CPU-capable |
| **AssemblyAI Universal-3 Pro** | offline (API) | **cpWER 33.34%** on DiPCo+NOTSOFAR (hard meeting sets); claims +10.1% DER / +13.2% cpWER *relative* gains, +30% in noise | commercial, cloud |
| **Deepgram** | offline (API) | speed-first; "+53.1% vs prior", language-agnostic; users report voice-mixing on similar voices | commercial, cloud |

Caveats: different datasets, different text normalization, relative-vs-absolute gains, and Otter
publishes **no rigorous academic DER/cpWER**. Independent cross-system suites: **SDBench**
(arxiv 2507.16136), "Benchmarking Diarization Models" (researchgate). Hard meeting cpWER in the
~30%+ range even for the best APIs tells you **single-mic multi-party is genuinely hard for everyone.**

**Datasets that ship the references you need:** **AMI** (meetings; transcripts + diarization labels +
named speakers across sessions → DER *and* cpWER *and* cross-session identity), **DIHARD III**
(hardest, diarization), **VoxConverse** (diarization), **CHiME-6/7/8** (meeting cpWER).

---

## 2. Online / streaming diarization — existing implementations (≈ your idea)

Your "process the last N minutes, keep an ephemeral fingerprint across chunks, tag returning speakers,
mint new ones below a confidence floor" **is the standard online-diarization pattern.** Prior art:

- **diart** (Coria/Bredin 2021, *Overlap-Aware Low-Latency Online Diarization*, arXiv 2109.06483) —
  **this is your architecture, already built.** Incremental clustering over a **rolling buffer updated
  every ~500 ms**, with **cannot-link constraints** derived from the local end-to-end segmentation so
  two distinct local speakers are **never wrongly merged** across updates. End-to-end local
  segmentation front-end (pyannote). CPU-capable. Latency is a dial (smaller buffer/step → lower
  latency, higher DER). **Key trick to steal: cannot-link constraints** — directly addresses the
  cross-chunk wrongful-merge you were worried about.
- **Streaming Sortformer** (NVIDIA, Aug 2025) — **Arrival-Order Speaker Cache (AOSC)**: a dynamic
  memory buffer of speaker embeddings seen so far; new frames are matched against it for consistent
  labels without recomputation. Tracks 2–4+ speakers, processes small overlapping chunks, real-time —
  **but requires NVIDIA GPU** (no CPU path mentioned). This is your "ephemeral fingerprint cache"
  exactly, learned end-to-end. SOTA streaming accuracy.
- **Turn-to-Diarize** (Google, 2021, arXiv 2109.11641) — transformer-transducer speaker-turn
  detection + clustering; efficient/on-device.
- **Multi-stage clustering** (Google, arXiv 2210.13690) — *different clusterers for short / medium /
  long inputs* (fallback clusterer for short-form). **Directly relevant to your chunk-size worry** —
  short chunks need a different clustering strategy than long ones.
- **BW-EDA-EEND** (arXiv 2011.02678) — streaming EEND for a variable number of speakers.

**Takeaway:** the ephemeral-cache-across-chunks pattern is well-established (diart = explicit
clustering, Sortformer = neural cache). The **cannot-link constraint** and **multi-stage clustering
for short chunks** are the two design ideas worth importing into our version.

---

## 3. Preprocessing before diarization — does it help? (yes, with a caveat)

- **Denoising / speech enhancement helps DER**, mainly by **reducing missed-speech** and producing
  **sharper VAD + speaker boundaries**. Noise degrades the speaker-embedding model's discriminability,
  so cleaner input → better clustering. (AutoPrep arXiv 2309.13905; "Multi-Stage Diarization for Noisy
  Classrooms" arXiv 2505.10879.)
- **A good VAD is the highest-leverage front-end** — missed/false speech is a direct DER component, and
  the diarizer can't recover a turn its VAD dropped.
- **Multi-mic only:** beamforming / adaptive array processing helps a lot — **N/A for single-mic mono**
  (our hard case).
- **The caveat:** *over-aggressive* denoising **distorts the voice and can hurt the speaker
  embedding.** The goal is reliable VAD/boundaries, not pretty audio. Training/enrolling on **both
  denoised and noisy** audio is more robust than denoising hard.
- **Overlap/rapid-exchange:** the real fix is *source separation* (split overlapping speakers into
  streams) — but it's heavy, error-prone, and risky to bolt on. Don't start there.

**For single-mic FPM:** add **VAD + light denoise** before diarization (cheap DER win); skip
beamforming (needs multi-mic); treat source separation as research, not MVP.

---

## 4. What to try / implement for the product (prioritized)

1. **VAD + light denoise front-end** before diarization. Cheapest accuracy lever; cuts missed-speech.
   Use a strong VAD (Silero / pyannote VAD). **Do not over-denoise** — validate it doesn't lower the
   enroll-print quality (re-run the E0–E1 separation check with/without).
2. **Steal diart's cannot-link constraint** for the chunked design — it's the documented fix for the
   cross-chunk wrongful-merge you flagged. Don't let two distinct local speakers collapse across the
   buffer.
3. **Multi-stage clustering by chunk length** (Google 2210.13690) — short chunks need a different
   clusterer than long ones. Explains why small-chunk DiariZen over-segmented; design for it.
4. **Ephemeral cache = the FPM matcher** you already have (`match.classify` MATCH/UNKNOWN as the
   "returning speaker vs new speaker" gate; `enroll` exemplar-accumulation as the centroid update).
   Use **overlapping** windows so cross-chunk linking matches on shared speech, not a cold centroid.
5. **Chunk-size elbow sweep with real metrics** — cpWER (`meeteval`) + DER (`pyannote.metrics`) on
   **AMI** (free ground truth). Plot accuracy + RTF + RAM vs chunk size → pick the knee. This replaces
   eyeballing with numbers.
6. **Comparison table** — run diart / full-batch DiariZen / our chunked variant / pyannote / NeMo on
   the same AMI audio; add AssemblyAI + Deepgram via their batch APIs (cpWER); Otter only if you can
   feed it audio (cpWER cell, no DER).
7. **Streaming Sortformer — eval-only ceiling, NOT prod (GPU TEE unaffordable).** Run it on GPU
   *locally* against AMI to measure the gap vs the CPU path — useful as a north-star number, but it
   can't ship (production is CPU). Don't build product around it.

---

### The two locked things-to-try (consolidated 2026-06-13)

**#1 (near-term) — CPU diart-style *streaming* DiariZen + the elbow.** diart's rolling-buffer +
cannot-link glue, DiariZen as the per-chunk engine, FPM matcher as the ephemeral cache; sweep chunk
size to find the **elbow** (accuracy plateau vs latency/RAM), all on CPU. **Prereq:** a cpWER/DER
metric harness on **AMI** (public labels) so the elbow has a measurable y-axis. → single-mic fallback.

**#2 (later) — 2-phone browser-link implementation + experiment** (§5). The engineering-as-diarization
path; 1-vs-2-phone where the 2-phone run is its own pseudo-gold. Deferred, but it's the real fix and it
*generates* the ground-truth labels for free.

**Reality check (keep the north star):** even SOTA diarization hits the rapid-exchange wall on a mono
mix — that failure is fundamental, not a tool choice. The product's **per-participant audio channels**
(rostered path) sidestep diarization entirely; acoustic diarization + fingerprinting is the
**unrostered single-mic fallback**, where preprocessing + cannot-link streaming buy incremental, not
miraculous, gains.

---

## 5. THE CONCRETE PLAN — engineering the diarization away (browser-link, phone-per-person)

**Thesis: solve diarization as a *systems* problem, not an acoustic-ML problem.** Don't fight to split
a mono mix. Instead give each person a **near-field channel** (their own phone), time-sync the
streams, transcribe each, and use **voice fingerprinting (FPM) as the cross-talk gate + identity**.
Zero new hardware, no app, no laptop, no "meeting room" to join — **the host sends a link, each person
opens it in their phone browser.** Everyone already has a phone in hand.

### Architecture
- **Browser = thin capture client.** Phone browser `getUserMedia` → mic, no install. (We already use
  this — `MediaRecorder` in the eval-harness `record.html`.) Friction = **HTTPS + one permission tap.**
- **Phones can't run Whisper** → each browser **streams audio to the server/TEE**, which runs
  **ASR + FPM per stream**. Better for confidentiality: audio goes straight to the enclave, nothing
  computed on-device.
- **Transcript-merge, coarse sync.** Merge *text*, not waveforms → only need ~tens-of-ms clock
  alignment (client-offset handshake, `performance.now()` ↔ server receive time). **No sample-level
  SRO/drift problem** — that's the whole reason to merge transcripts, not audio.
- **FPM = the cross-talk gate** (the one new piece of logic). Every phone hears everyone; on phone A's
  stream, **keep only segments whose voiceprint matches A's owner; drop the bleed.** TS-VAD-flavored.
- **Device-login = identity** (free) **and a perfectly-labeled, clean near-field enrollment clip**
  (free) — so the fingerprints build themselves with correct labels, fixing the contaminated-enroll
  problem we hit on mono.

### The experiment (1 phone vs 2 phones — comparison *and* its own ground truth)
Same room, same people, same distance, same conversation, recorded **simultaneously**:
- **Condition A — 1 phone:** the mono mix → existing transcribe+diarize+merge (the hard case).
- **Condition B — 2 phones:** two near-field streams → per-stream ASR + FPM cross-talk gate → merge.
- **Key trick:** in B, each phone's owner is **known (login)** and **near-field/clean** → B is a
  **near-perfect "who-spoke-when" reference.** So **B grades A** — the multi-device run *manufactures
  the labels* to score the single-mic run. **This solves the gold-standard blocker** for cpWER/DER on
  our own recordings, with zero manual transcription. One variable changes (#devices) → a real result.

### What's supported vs novel
- **Supported by literature:** transcript-merge over synced peers (**VoxTerm**, our own prototype —
  `external/VoxTerm`, NTP `network/clock.py`); near-field per-person capture (ad-hoc arrays, PickNet);
  fingerprint-conditioned attribution (**TS-VAD**, CHiME-6 winner, arXiv 2005.07272 / online 2310.08696).
- **Novel in the assembly:** zero-install **browser-link** delivery + **login-as-free-label/enrollment**
  + the **1-vs-2-phone-as-evaluation** method. Pieces proven → it'll work; integration + eval method
  are the contribution.

### Build reuse
The eval-harness record UI already captures browser mic. Extend → (a) multi-phone session join via
link, (b) per-stream save, (c) the FPM cross-talk gate, (d) the A-vs-B compare (B = pseudo-gold).

### Caveats
HTTPS/cert needed for mobile mic; the **cross-talk gate** is the one genuinely new algorithm (but
TS-VAD-backed); phone echo/AGC can distort (usually fine for transcript-merge).

### Out of scope (future-future — deliberately dropped)
**TDOA / audio-level beamforming / spatial fusion** — richer signal but needs *sample-level* sync and
heavy DSP. Revisit only if transcript-merge + fingerprint proves insufficient. Not part of this plan.

---

## Sources
- [Benchmarking Diarization Models (ResearchGate)](https://www.researchgate.net/publication/396048562_Benchmarking_Diarization_Models)
- [DiariZen toolkit (GitHub, BUTSpeechFIT)](https://github.com/BUTSpeechFIT/DiariZen)
- [DiariZen Explained (arXiv 2604.21507)](https://arxiv.org/html/2604.21507v1)
- [SDBench: Comprehensive Benchmark Suite for Speaker Diarization (arXiv 2507.16136)](https://arxiv.org/html/2507.16136v2)
- [pyannoteAI — How to evaluate diarization performance](https://www.pyannote.ai/blog/how-to-evaluate-speaker-diarization-performance)
- [Sortformer (EmergentMind topic)](https://www.emergentmind.com/topics/sortformer-model)
- [NVIDIA Streaming Sortformer (MarkTechPost)](https://www.marktechpost.com/2025/08/21/nvidia-ai-just-released-streaming-sortformer-a-real-time-speaker-diarization-that-figures-out-whos-talking-in-meetings-and-calls-instantly/)
- [Streaming Sortformer: Speaker Cache-Based Online Diarization (ResearchGate)](https://www.researchgate.net/publication/396813275_Streaming_Sortformer_Speaker_Cache-Based_Online_Speaker_Diarization_with_Arrival-Time_Ordering)
- [Overlap-Aware Low-Latency Online Diarization / diart (arXiv 2109.06483)](https://arxiv.org/pdf/2109.06483)
- [diart (GitHub, juanmc2005)](https://github.com/juanmc2005/diart)
- [Turn-to-Diarize (arXiv 2109.11641)](https://arxiv.org/pdf/2109.11641)
- [Real-Time On-Device Diarization with Multi-Stage Clustering (arXiv 2210.13690)](https://arxiv.org/abs/2210.13690)
- [BW-EDA-EEND streaming diarization (arXiv 2011.02678)](https://arxiv.org/pdf/2011.02678)
- [AutoPrep: Automatic Preprocessing for In-the-Wild Speech (arXiv 2309.13905)](https://arxiv.org/pdf/2309.13905)
- [Multi-Stage Speaker Diarization for Noisy Classrooms (arXiv 2505.10879)](https://arxiv.org/html/2505.10879v1)
- [AssemblyAI Benchmarks](https://www.assemblyai.com/benchmarks)
- [AssemblyAI — Top speaker diarization libraries and APIs](https://www.assemblyai.com/blog/top-speaker-diarization-libraries-and-apis)
- [Deepgram — Best Speech-to-Text APIs 2026](https://deepgram.com/learn/best-speech-to-text-apis-2026)
- [Scribie — Real-world STT accuracy benchmark](https://scribie.com/blog/speech-to-text-accuracy-benchmark-assemblyai-deepgram-whisperx)

### §5 — multi-device, fingerprint-conditioned attribution, transcript-merge
- [TS-VAD: Target-Speaker VAD for diarization (arXiv 2005.07272, CHiME-6 winner)](https://arxiv.org/abs/2005.07272) — fingerprint-conditioned attribution
- [End-to-end Online Diarization with Target-Speaker Tracking (arXiv 2310.08696)](https://arxiv.org/pdf/2310.08696) — streaming TS-VAD
- VoxTerm "Party Mode" — in-house prototype (`conclave-shape-rotator/external/VoxTerm`, NTP `network/clock.py`); transcript-merge over LAN-synced peers
- [Microsoft — Meeting Transcription Using Asynchronous Distant Microphones](https://www.microsoft.com/en-us/research/publication/meeting-transcription-using-asynchronous-distant-microphones/)
- [Spatial Diarization for Meeting Transcription with Ad-Hoc Acoustic Sensor Networks (arXiv 2311.15597)](https://arxiv.org/pdf/2311.15597)
- [PickNet: Real-Time Channel Selection for Ad Hoc Microphone Arrays (arXiv 2201.09586)](https://arxiv.org/pdf/2201.09586)
- [Utterance-Wise Meeting Transcription Using Asynchronous Distributed Microphones (arXiv 2007.15868)](https://arxiv.org/pdf/2007.15868)
- [Libri-adhoc40: synchronized ad-hoc microphone array dataset (arXiv 2103.15118)](https://arxiv.org/pdf/2103.15118)
- [SAMbA: Speech enhancement with Asynchronous ad-hoc Microphone Arrays (arXiv 2307.16582)](https://arxiv.org/pdf/2307.16582)
- [MISP 2025 — multi-modal multi-device meeting transcription](https://www.isca-archive.org/interspeech_2025/gao25g_interspeech.pdf)
- [Autodirective Audio Capturing Through a Synchronized Smartphone Array (MobiSys'14)](http://xyzhang.ucsd.edu/papers/Sur_Wei_MobiSys14_Dia.pdf)
