"""Record → save → run web UI (C11) — one button to record, one to run.

    /tmp/diart-venv/bin/python -m eval_harness.server
    → open http://localhost:8090/record/<experiment>

Browser MediaRecorder captures audio → POST upload → ffmpeg-decoded to 16k mono → written to
`experiments/<exp>/audio.wav` (overwrite-safe). A Run button calls run_experiment (diart) and shows
result.json in a panel. Deliberately record→batch (no live transcription) — sidesteps CPU
real-time latency. Eval-only; never the production Recato↔Conclave flow.
"""
from __future__ import annotations

from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from config import TARGET_SAMPLE_RATE
from fpm.audio import AudioDecodeError, decode_to_mono
from eval_harness.run import run_experiment

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE / "experiments"
RECORD_HTML = HERE / "static" / "record.html"

app = FastAPI(title="eval-harness record UI")


def _exp_dir(exp: str) -> Path:
    # guard against path traversal — exp is a single folder name under experiments/
    if "/" in exp or "\\" in exp or exp in ("", ".", ".."):
        raise HTTPException(400, f"bad experiment name: {exp!r}")
    d = EXPERIMENTS / exp
    if not d.is_dir():
        raise HTTPException(404, f"no experiment folder: experiments/{exp}")
    return d


def _list_experiments() -> list[str]:
    if not EXPERIMENTS.is_dir():
        return []
    return sorted(p.name for p in EXPERIMENTS.iterdir()
                  if p.is_dir() and (p / "config.yaml").exists())


@app.get("/api/experiments")
def api_experiments() -> JSONResponse:
    return JSONResponse({"experiments": _list_experiments()})


@app.get("/", response_class=HTMLResponse)
@app.get("/record/{exp}", response_class=HTMLResponse)
def record_page(exp: str | None = None) -> HTMLResponse:
    return HTMLResponse(RECORD_HTML.read_text())


@app.post("/record/{exp}/upload")
async def upload(exp: str, file: UploadFile) -> JSONResponse:
    d = _exp_dir(exp)
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty upload")
    try:
        audio = decode_to_mono(raw)            # ffmpeg: any container → 16k mono float32
    except AudioDecodeError as e:
        raise HTTPException(422, f"could not decode audio: {e}")
    out = d / "audio.wav"
    sf.write(out, audio, TARGET_SAMPLE_RATE)   # overwrite-safe; canonical 16k mono wav
    return JSONResponse({"ok": True, "saved": f"experiments/{out.relative_to(EXPERIMENTS)}",
                         "duration_sec": round(len(audio) / TARGET_SAMPLE_RATE, 2)})


@app.post("/record/{exp}/run")
def run(exp: str) -> JSONResponse:
    d = _exp_dir(exp)
    if not (d / "audio.wav").exists():
        raise HTTPException(400, "no audio.wav yet — record/upload first")
    return JSONResponse(run_experiment(d))     # diart (config default); writes results/


def main() -> None:
    import uvicorn
    print("eval-harness record UI → http://localhost:8090/record/<experiment>")
    print("experiments:", ", ".join(_list_experiments()) or "(none yet)")
    uvicorn.run(app, host="127.0.0.1", port=8090)


if __name__ == "__main__":
    main()
