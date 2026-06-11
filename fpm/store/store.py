"""Encrypted, workspace-scoped SQLite voiceprint store.

Adapted from VoxTerm `audio/speakers/store.py` (MIT). Every read/write is keyed by
`workspace_id` (enforced in SQL) → no code path can touch another workspace's
voiceprints. Centroids + exemplars are AES-encrypted at rest; centroids are cached
in memory per workspace for fast cosine matching. The store records the embedder
model+dim and refuses to open against a different-dim embedder (cross-model matches
would be meaningless).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import DB_PATH, ID_EMBEDDING_DIM, ID_EMBEDDING_MODEL

from . import crypto
from .models import Voiceprint

_SCHEMA = """
CREATE TABLE IF NOT EXISTS voiceprints(
  voiceprint_id TEXT PRIMARY KEY,
  workspace_id  TEXT NOT NULL,
  name          TEXT NOT NULL DEFAULT '',
  centroid      BLOB NOT NULL,
  exemplars     BLOB NOT NULL DEFAULT X'',
  exemplar_count     INTEGER NOT NULL DEFAULT 0,
  enroll_count       INTEGER NOT NULL DEFAULT 0,
  total_duration_sec REAL    NOT NULL DEFAULT 0,
  quality_score      REAL    NOT NULL DEFAULT 0,
  created_at TEXT, updated_at TEXT, last_seen_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_vp_ws ON voiceprints(workspace_id);
CREATE TABLE IF NOT EXISTS vocab(workspace_id TEXT PRIMARY KEY, terms TEXT NOT NULL DEFAULT '[]');
CREATE TABLE IF NOT EXISTS binding_audit(
  id INTEGER PRIMARY KEY AUTOINCREMENT, workspace_id TEXT, voiceprint_id TEXT,
  old_name TEXT, new_name TEXT, actor TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS store_meta(key TEXT PRIMARY KEY, value TEXT);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_voiceprint_id() -> str:
    return "vp_" + uuid.uuid4().hex[:16]


class VoiceprintStore:
    def __init__(self, db_path: str | Path | None = None, key: bytes | None = None):
        self._db_path = Path(db_path or DB_PATH)
        self._key = key
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._centroids: dict[str, dict[str, np.ndarray]] = {}  # ws → {vid → centroid}

    # ── lifecycle ────────────────────────────────────────────

    def open(self) -> "VoiceprintStore":
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        try:
            os.chmod(self._db_path, 0o600)
        except OSError:
            pass
        if self._key is None:
            self._key = crypto.get_or_create_key()
        self._check_meta()
        self._load_centroids()
        return self

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def _check_meta(self) -> None:
        meta = dict(self._conn.execute("SELECT key, value FROM store_meta").fetchall())
        want = {"embedder_model": ID_EMBEDDING_MODEL, "embedder_dim": str(ID_EMBEDDING_DIM)}
        if not meta:
            self._conn.executemany("INSERT INTO store_meta VALUES (?, ?)", list(want.items()))
            self._conn.commit()
        elif meta.get("embedder_dim") != want["embedder_dim"]:
            raise RuntimeError(
                f"store embedder mismatch (dim {meta.get('embedder_dim')} != {want['embedder_dim']}); "
                "cross-model voiceprint matches are forbidden"
            )

    # ── (de)serialization ────────────────────────────────────

    def _enc(self, arr: np.ndarray) -> bytes:
        return crypto.encrypt_blob(self._key, np.asarray(arr, dtype=np.float32).tobytes())

    def _dec_centroid(self, blob: bytes) -> np.ndarray:
        return np.frombuffer(crypto.decrypt_blob(self._key, blob), dtype=np.float32).copy()

    def _enc_exemplars(self, exemplars: list[np.ndarray]) -> bytes:
        if not exemplars:
            return b""
        return self._enc(np.stack(exemplars))

    def _dec_exemplars(self, blob: bytes) -> list[np.ndarray]:
        if not blob:
            return []
        flat = np.frombuffer(crypto.decrypt_blob(self._key, blob), dtype=np.float32)
        arr = flat.reshape(-1, ID_EMBEDDING_DIM)
        return [row.copy() for row in arr]

    def _load_centroids(self) -> None:
        self._centroids.clear()
        for vid, ws, blob in self._conn.execute(
            "SELECT voiceprint_id, workspace_id, centroid FROM voiceprints"
        ):
            self._centroids.setdefault(ws, {})[vid] = self._dec_centroid(blob)

    # ── writes (workspace-scoped) ────────────────────────────

    def upsert(self, vp: Voiceprint) -> None:
        with self._lock:
            ts = _now()
            row = (
                vp.voiceprint_id, vp.workspace_id, vp.name,
                self._enc(vp.centroid), self._enc_exemplars(vp.exemplars),
                len(vp.exemplars), vp.enroll_count, vp.total_duration_sec, vp.quality_score,
                vp.created_at or ts, ts, ts,
            )
            self._conn.execute(
                """INSERT INTO voiceprints
                   (voiceprint_id, workspace_id, name, centroid, exemplars, exemplar_count,
                    enroll_count, total_duration_sec, quality_score, created_at, updated_at, last_seen_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(voiceprint_id) DO UPDATE SET
                     name=excluded.name, centroid=excluded.centroid, exemplars=excluded.exemplars,
                     exemplar_count=excluded.exemplar_count, enroll_count=excluded.enroll_count,
                     total_duration_sec=excluded.total_duration_sec, quality_score=excluded.quality_score,
                     updated_at=excluded.updated_at, last_seen_at=excluded.last_seen_at""",
                row,
            )
            self._conn.commit()
            self._centroids.setdefault(vp.workspace_id, {})[vp.voiceprint_id] = (
                np.asarray(vp.centroid, dtype=np.float32).copy()
            )

    def set_name(self, workspace_id: str, voiceprint_id: str, name: str, actor: str) -> bool:
        """Bind a name to a voiceprint (tag-once). Workspace-scoped + audited."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT name FROM voiceprints WHERE voiceprint_id=? AND workspace_id=?",
                (voiceprint_id, workspace_id),
            ).fetchone()
            if cur is None:
                return False  # not found OR wrong workspace → no cross-workspace writes
            old = cur[0]
            self._conn.execute(
                "UPDATE voiceprints SET name=?, updated_at=? WHERE voiceprint_id=? AND workspace_id=?",
                (name, _now(), voiceprint_id, workspace_id),
            )
            self._conn.execute(
                "INSERT INTO binding_audit (workspace_id, voiceprint_id, old_name, new_name, actor, ts)"
                " VALUES (?,?,?,?,?,?)",
                (workspace_id, voiceprint_id, old, name, actor, _now()),
            )
            self._conn.commit()
            return True

    def delete(self, workspace_id: str, voiceprint_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM voiceprints WHERE voiceprint_id=? AND workspace_id=?",
                (voiceprint_id, workspace_id),
            )
            self._conn.commit()
            self._centroids.get(workspace_id, {}).pop(voiceprint_id, None)
            return cur.rowcount > 0

    # ── reads (workspace-scoped) ─────────────────────────────

    def get(self, workspace_id: str, voiceprint_id: str) -> Voiceprint | None:
        r = self._conn.execute(
            """SELECT voiceprint_id, workspace_id, name, centroid, exemplars, enroll_count,
                      total_duration_sec, quality_score, created_at, updated_at, last_seen_at
               FROM voiceprints WHERE voiceprint_id=? AND workspace_id=?""",
            (voiceprint_id, workspace_id),
        ).fetchone()
        if r is None:
            return None
        vp = Voiceprint(
            voiceprint_id=r[0], workspace_id=r[1], name=r[2],
            centroid=self._dec_centroid(r[3]), exemplars=self._dec_exemplars(r[4]),
            enroll_count=r[5], total_duration_sec=r[6], quality_score=r[7],
            created_at=r[8], updated_at=r[9], last_seen_at=r[10],
        )
        vp.recompute_centroid() if vp.exemplars else None  # rebuild sub-centroids
        return vp

    def list_ids(self, workspace_id: str) -> list[str]:
        return [
            row[0]
            for row in self._conn.execute(
                "SELECT voiceprint_id FROM voiceprints WHERE workspace_id=?", (workspace_id,)
            )
        ]

    def centroids(self, workspace_id: str) -> dict[str, np.ndarray]:
        """All centroids for a workspace (from cache) — for matching."""
        return dict(self._centroids.get(workspace_id, {}))

    # ── vocab ────────────────────────────────────────────────

    def set_vocab(self, workspace_id: str, terms: list[str]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO vocab (workspace_id, terms) VALUES (?, ?) "
                "ON CONFLICT(workspace_id) DO UPDATE SET terms=excluded.terms",
                (workspace_id, json.dumps(list(terms))),
            )
            self._conn.commit()

    def get_vocab(self, workspace_id: str) -> list[str]:
        r = self._conn.execute(
            "SELECT terms FROM vocab WHERE workspace_id=?", (workspace_id,)
        ).fetchone()
        return json.loads(r[0]) if r else []
