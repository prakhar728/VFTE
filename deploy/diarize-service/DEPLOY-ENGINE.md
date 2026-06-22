# Diarization service — engine selector (P2/P3)

One service, two deployments, selected by `FPM_DIARIZE_ENGINE`:

| Env | Engine | Use | Hardware |
|---|---|---|---|
| `FPM_DIARIZE_ENGINE=diart` | `DiartDiarizer` | in-person / live acoustic diarization (**P2**) | CPU |
| `FPM_DIARIZE_ENGINE=diarizen` (default) | `DiariZenDiarizer` | post-meeting batch re-clustering (**P3**) | GPU |

Both honor the same `POST /diarize` (multipart `file` + `workspace`, NDJSON
heartbeats → final `{segments:[{start,end,local_speaker}],...}`) that
`fpm/diarize/remote_engine.py` speaks.

## Flipping FPM core to use it (T1 — config only, no FPM code change)
The FPM core (`main.py::_default_diarizer_factory`) already supports remote:
```
FPM_DIARIZER=remote
FPM_DIARIZER_URL=https://<diarize-cvm>            # the service above
FPM_DIARIZE_TOKEN=<bearer>                         # service checks this
```
Identity (CAM++ re-embed + store match) stays in FPM core — the remote box
never sees voiceprints (engine-independent-store invariant, base.py).

## diart deps (P2)
The `diart` engine needs diart + pyannote (separate from DiariZen's torch 2.1.1).
Known pin: `matplotlib<3.9` (pyannote.core `get_cmap` breakage). The diart variant
needs its own requirements/Dockerfile layer — TODO before the CPU deploy.

## NOT verified here
CPU-diart real-time headroom across multiple in-person streams — **measure before
committing** (GPU-realtime is the fallback). Plan P2 gate.
