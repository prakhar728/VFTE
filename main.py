"""FPM — Speaker Fingerprinting Microservice: FastAPI entrypoint.

Endpoints are added per milestone (all v1 endpoints behind per-caller scoped auth):
    M4  POST /v1/diarize        (Recato-scoped)
    M4  GET  /v1/vocab/{host}   (Recato-scoped)
    M4  POST /v1/knowledge      (Conclave-scoped, write-only)

This scaffold (M0) exposes only the unauthenticated health check.
"""
from __future__ import annotations

from fastapi import FastAPI

from config import SERVICE_NAME, SERVICE_VERSION

app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)


@app.get("/health")
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION}
