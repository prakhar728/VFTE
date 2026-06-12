"""Run metrics — audio length, latency, RTF, mode."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Metrics:
    mode: str                     # offline | realtime
    audio_length_sec: float
    asr_sec: float                # wall time in ASR
    diarize_sec: float            # wall time in diarization
    total_sec: float              # asr + diarize (offline) or stream wall time (realtime)
    rtf: float                    # total_sec / audio_length_sec (<1 = faster than real-time)

    def as_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(self).items()}


def offline_metrics(audio_length_sec: float, asr_sec: float, diarize_sec: float) -> Metrics:
    total = asr_sec + diarize_sec
    rtf = total / audio_length_sec if audio_length_sec > 0 else 0.0
    return Metrics(mode="offline", audio_length_sec=audio_length_sec, asr_sec=asr_sec,
                   diarize_sec=diarize_sec, total_sec=total, rtf=rtf)
