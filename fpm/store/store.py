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

from dataclasses import dataclass

from . import crypto
from .models import Voiceprint


@dataclass
class DeleteResult:
    """Outcome of a `forget` delete. Truthy iff a row was actually deleted, so legacy
    `if store.delete(...)` callers keep working; carries the surviving `usage_ledger`
    "forget" row id + timestamp that the signed deletion receipt references."""
    deleted: bool
    ledger_row_id: int | None = None
    deleted_at: str | None = None

    def __bool__(self) -> bool:
        return self.deleted

_SCHEMA = """
CREATE TABLE IF NOT EXISTS voiceprints(
  voiceprint_id TEXT PRIMARY KEY,
  workspace_id  TEXT NOT NULL,
  name          TEXT NOT NULL DEFAULT '',
  owner_email   TEXT NOT NULL DEFAULT '',
  enroll_allowed   INTEGER NOT NULL DEFAULT 1,
  identify_allowed INTEGER NOT NULL DEFAULT 1,
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
-- usage ledger (decision G): append-only audit of every touch of a voiceprint —
-- enroll, identify/match, name-bind, dashboard read, forget. The proof trail, not telemetry.
CREATE TABLE IF NOT EXISTS usage_ledger(
  id INTEGER PRIMARY KEY AUTOINCREMENT, workspace_id TEXT, voiceprint_id TEXT,
  event TEXT, consumer TEXT, purpose TEXT, ts TEXT);
CREATE INDEX IF NOT EXISTS idx_ledger_vp ON usage_ledger(workspace_id, voiceprint_id);
-- P4 trust handshake: pending email-binding proposals. A host tags a voiceprint
-- (name+email) → a pending proposal; the data subject confirms/denies on the consent
-- dashboard. owner_email binds only on confirm (via claim_owner+set_name). Idempotent
-- per (workspace, voiceprint, email) so re-tagging never duplicates.
CREATE TABLE IF NOT EXISTS proposals(
  proposal_id    TEXT PRIMARY KEY,
  workspace_id   TEXT NOT NULL,
  voiceprint_id  TEXT NOT NULL,
  proposed_email TEXT NOT NULL,
  proposed_by    TEXT NOT NULL,
  proposed_name  TEXT NOT NULL DEFAULT '',
  status         TEXT NOT NULL DEFAULT 'pending',  -- pending | confirmed | denied
  created_at TEXT, confirmed_at TEXT, denied_at TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS idx_proposal_unique
  ON proposals(workspace_id, voiceprint_id, proposed_email);
CREATE INDEX IF NOT EXISTS idx_proposal_email ON proposals(proposed_email);
CREATE TABLE IF NOT EXISTS store_meta(key TEXT PRIMARY KEY, value TEXT);
-- deletion receipts (Task #1): every signed "forget me" receipt we issue, append-only,
-- so the dashboard can re-show a past receipt and we can prove issuance after the
-- voiceprint row is gone. NO owner_email plaintext — only the sha256 hash.
CREATE TABLE IF NOT EXISTS deletion_receipts(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  workspace_id TEXT, voiceprint_id TEXT,
  owner_email_hash TEXT, deleted_at TEXT,
  ledger_row_id INTEGER,
  payload_json TEXT, signature TEXT, alg TEXT, key_id TEXT);
CREATE INDEX IF NOT EXISTS idx_receipt_owner ON deletion_receipts(owner_email_hash);
"""

# Column order for proposal-row reads → dict (the C4 wire shape).
_PROPOSAL_COLS = (
    "proposal_id", "workspace_id", "voiceprint_id", "proposed_email", "proposed_by",
    "proposed_name", "status", "created_at", "confirmed_at", "denied_at",
)

# Columns added to `voiceprints` after the original A.5 schema shipped; applied via
# ALTER on open so an existing real DB upgrades in place (CREATE IF NOT EXISTS won't
# add columns to a table that already exists).
_VP_MIGRATIONS = (
    ("owner_email", "TEXT NOT NULL DEFAULT ''"),
    ("enroll_allowed", "INTEGER NOT NULL DEFAULT 1"),
    ("identify_allowed", "INTEGER NOT NULL DEFAULT 1"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_voiceprint_id() -> str:
    return "vp_" + uuid.uuid4().hex[:16]


def new_proposal_id() -> str:
    return "prop_" + uuid.uuid4().hex[:16]


class VoiceprintStore:
    def __init__(self, db_path: str | Path | None = None, key: bytes | None = None):
        self._db_path = Path(db_path or DB_PATH)
        self._key = key
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._centroids: dict[str, dict[str, np.ndarray]] = {}  # ws → {vid → centroid}
        # consent flags cached for the hot path (enforcement on every match) so we
        # don't decrypt a voiceprint just to read two booleans: ws → vid → (enroll, identify)
        self._flags: dict[str, dict[str, tuple[bool, bool]]] = {}

    # ── lifecycle ────────────────────────────────────────────

    def open(self) -> "VoiceprintStore":
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._migrate()
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

    def _migrate(self) -> None:
        """Add consent-plane columns to a pre-existing `voiceprints` table in place."""
        have = {row[1] for row in self._conn.execute("PRAGMA table_info(voiceprints)")}
        for col, decl in _VP_MIGRATIONS:
            if col not in have:
                self._conn.execute(f"ALTER TABLE voiceprints ADD COLUMN {col} {decl}")
        # index on owner_email only after the column is guaranteed to exist (post-ALTER)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_vp_owner ON voiceprints(owner_email)")

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
        self._flags.clear()
        for vid, ws, blob, en, idn in self._conn.execute(
            "SELECT voiceprint_id, workspace_id, centroid, enroll_allowed, identify_allowed "
            "FROM voiceprints"
        ):
            self._centroids.setdefault(ws, {})[vid] = self._dec_centroid(blob)
            self._flags.setdefault(ws, {})[vid] = (bool(en), bool(idn))

    # ── writes (workspace-scoped) ────────────────────────────

    def upsert(self, vp: Voiceprint) -> None:
        with self._lock:
            ts = _now()
            row = (
                vp.voiceprint_id, vp.workspace_id, vp.name,
                vp.owner_email, int(vp.enroll_allowed), int(vp.identify_allowed),
                self._enc(vp.centroid), self._enc_exemplars(vp.exemplars),
                len(vp.exemplars), vp.enroll_count, vp.total_duration_sec, vp.quality_score,
                vp.created_at or ts, ts, ts,
            )
            self._conn.execute(
                """INSERT INTO voiceprints
                   (voiceprint_id, workspace_id, name, owner_email, enroll_allowed, identify_allowed,
                    centroid, exemplars, exemplar_count,
                    enroll_count, total_duration_sec, quality_score, created_at, updated_at, last_seen_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(voiceprint_id) DO UPDATE SET
                     name=excluded.name, owner_email=excluded.owner_email,
                     enroll_allowed=excluded.enroll_allowed, identify_allowed=excluded.identify_allowed,
                     centroid=excluded.centroid, exemplars=excluded.exemplars,
                     exemplar_count=excluded.exemplar_count, enroll_count=excluded.enroll_count,
                     total_duration_sec=excluded.total_duration_sec, quality_score=excluded.quality_score,
                     updated_at=excluded.updated_at, last_seen_at=excluded.last_seen_at""",
                row,
            )
            self._conn.commit()
            self._centroids.setdefault(vp.workspace_id, {})[vp.voiceprint_id] = (
                np.asarray(vp.centroid, dtype=np.float32).copy()
            )
            self._flags.setdefault(vp.workspace_id, {})[vp.voiceprint_id] = (
                bool(vp.enroll_allowed), bool(vp.identify_allowed)
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
            self._conn.execute(
                "INSERT INTO usage_ledger (workspace_id, voiceprint_id, event, consumer, purpose, ts)"
                " VALUES (?,?,?,?,?,?)",
                (workspace_id, voiceprint_id, "name_bind", actor, f"named '{name}'", _now()),
            )
            self._conn.commit()
            return True

    def delete(self, workspace_id: str, voiceprint_id: str, actor: str = "dashboard") -> DeleteResult:
        """'Forget me': hard-delete the FPM voiceprint row (embeddings gone). The append-
        only `usage_ledger` "forget" row survives as the proof anchor — its id + timestamp
        are returned so the signed deletion receipt can reference it. crypto-shred +
        re-enroll tombstone are deferred (doc §6).

        Returns a `DeleteResult` (truthy iff something was deleted). On a miss (not found /
        wrong workspace / already deleted) no ledger row is written and ledger_row_id/
        deleted_at are None — so an idempotent re-delete issues no receipt."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM voiceprints WHERE voiceprint_id=? AND workspace_id=?",
                (voiceprint_id, workspace_id),
            )
            deleted = cur.rowcount > 0
            ledger_row_id = deleted_at = None
            if deleted:
                deleted_at = _now()
                led = self._conn.execute(
                    "INSERT INTO usage_ledger (workspace_id, voiceprint_id, event, consumer, purpose, ts)"
                    " VALUES (?,?,?,?,?,?)",
                    (workspace_id, voiceprint_id, "forget", actor, "user erasure", deleted_at),
                )
                ledger_row_id = led.lastrowid
            self._conn.commit()
            self._centroids.get(workspace_id, {}).pop(voiceprint_id, None)
            self._flags.get(workspace_id, {}).pop(voiceprint_id, None)
            return DeleteResult(deleted=deleted, ledger_row_id=ledger_row_id, deleted_at=deleted_at)

    # ── deletion receipts (Task #1: cryptographic proof of deletion) ──

    def add_deletion_receipt(self, envelope: dict) -> int:
        """Persist an issued, signed receipt (append-only). Returns the new row id.

        Reads the anchor fields off the *signed payload* (not loose args) so what we store
        is exactly what was signed. No owner_email plaintext — only the hash in the payload."""
        p = envelope["payload"]
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO deletion_receipts (workspace_id, voiceprint_id, owner_email_hash,"
                " deleted_at, ledger_row_id, payload_json, signature, alg, key_id)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (p["workspace_id"], p["voiceprint_id"], p["owner_email_hash"], p["deleted_at"],
                 p["ledger_row_id"], json.dumps(p, sort_keys=True, separators=(",", ":")),
                 envelope["signature"], envelope["alg"], envelope["key_id"]),
            )
            self._conn.commit()
            return cur.lastrowid

    def deletion_receipts_for_hash(self, owner_email_hash: str) -> list[dict]:
        """All issued receipts for an owner (by email hash), newest first — reconstructed
        as verifiable `{payload, signature, alg, key_id}` envelopes."""
        rows = self._conn.execute(
            "SELECT payload_json, signature, alg, key_id FROM deletion_receipts "
            "WHERE owner_email_hash=? ORDER BY id DESC",
            (owner_email_hash,),
        ).fetchall()
        return [
            {"payload": json.loads(pj), "signature": sig, "alg": alg, "key_id": kid}
            for pj, sig, alg, kid in rows
        ]

    def meta(self, key: str, default: str | None = None) -> str | None:
        """Read a `store_meta` value (e.g. embedder_model / embedder_dim) stamped at open."""
        r = self._conn.execute("SELECT value FROM store_meta WHERE key=?", (key,)).fetchone()
        return r[0] if r else default

    # ── consent flags + ledger (WS3/WS4/WS5) ─────────────────

    def flags(self, workspace_id: str, voiceprint_id: str) -> tuple[bool, bool]:
        """(enroll_allowed, identify_allowed) from the hot cache; (True, True) if unknown."""
        return self._flags.get(workspace_id, {}).get(voiceprint_id, (True, True))

    def identify_allowed(self, workspace_id: str, voiceprint_id: str) -> bool:
        return self.flags(workspace_id, voiceprint_id)[1]

    def set_flags(
        self,
        workspace_id: str,
        voiceprint_id: str,
        *,
        enroll_allowed: bool | None = None,
        identify_allowed: bool | None = None,
        actor: str = "dashboard",
    ) -> bool:
        """Set one or both consent flags on a voiceprint (workspace-scoped). Audited in the ledger."""
        with self._lock:
            sets, params, events = [], [], []
            if enroll_allowed is not None:
                sets.append("enroll_allowed=?")
                params.append(int(enroll_allowed))
                events.append(("enroll_allowed", enroll_allowed))
            if identify_allowed is not None:
                sets.append("identify_allowed=?")
                params.append(int(identify_allowed))
                events.append(("identify_allowed", identify_allowed))
            if not sets:
                return False
            sets.append("updated_at=?")
            params.append(_now())
            cur = self._conn.execute(
                f"UPDATE voiceprints SET {', '.join(sets)} WHERE voiceprint_id=? AND workspace_id=?",
                (*params, voiceprint_id, workspace_id),
            )
            if cur.rowcount == 0:
                return False  # not found / wrong workspace
            for field, val in events:
                self._conn.execute(
                    "INSERT INTO usage_ledger (workspace_id, voiceprint_id, event, consumer, purpose, ts)"
                    " VALUES (?,?,?,?,?,?)",
                    (workspace_id, voiceprint_id, "control",
                     actor, f"{field}={'on' if val else 'off'}", _now()),
                )
            self._conn.commit()
            # refresh hot cache
            en, idn = self.flags(workspace_id, voiceprint_id)
            if enroll_allowed is not None:
                en = bool(enroll_allowed)
            if identify_allowed is not None:
                idn = bool(identify_allowed)
            self._flags.setdefault(workspace_id, {})[voiceprint_id] = (en, idn)
            return True

    def claim_owner(self, workspace_id: str, voiceprint_id: str, owner_email: str) -> bool:
        """Bind a voiceprint to its authenticated data subject's email (idempotent)."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE voiceprints SET owner_email=?, updated_at=? "
                "WHERE voiceprint_id=? AND workspace_id=?",
                (owner_email, _now(), voiceprint_id, workspace_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def find_by_owner_email(self, owner_email: str) -> list[tuple[str, str]]:
        """All (workspace_id, voiceprint_id) for an email — the dashboard's per-workspace list."""
        return [
            (row[0], row[1])
            for row in self._conn.execute(
                "SELECT workspace_id, voiceprint_id FROM voiceprints WHERE owner_email=? ORDER BY workspace_id",
                (owner_email,),
            )
        ]

    # ── proposals (P4 trust handshake) ───────────────────────

    def _proposal_dict(self, row) -> dict:
        return dict(zip(_PROPOSAL_COLS, row))

    def get_proposal(self, proposal_id: str) -> dict | None:
        r = self._conn.execute(
            f"SELECT {', '.join(_PROPOSAL_COLS)} FROM proposals WHERE proposal_id=?",
            (proposal_id,),
        ).fetchone()
        return self._proposal_dict(r) if r else None

    def propose(
        self, workspace_id: str, voiceprint_id: str, proposed_email: str,
        proposed_by: str, proposed_name: str = "",
    ) -> dict:
        """Create (or return the existing) pending email-binding proposal.

        Idempotent per (workspace, voiceprint, email): re-tagging the same person on the
        same voiceprint returns the original proposal unchanged (no duplicate row; the
        original proposed_name/proposed_by are retained). `owner_email` binds only on
        confirm (claim_owner+set_name), never here. Emails are stored lowercased.
        """
        email = proposed_email.lower()
        by = proposed_by.lower()
        with self._lock:
            existing = self._conn.execute(
                f"SELECT {', '.join(_PROPOSAL_COLS)} FROM proposals "
                "WHERE workspace_id=? AND voiceprint_id=? AND proposed_email=?",
                (workspace_id, voiceprint_id, email),
            ).fetchone()
            if existing:
                return self._proposal_dict(existing)
            pid = new_proposal_id()
            self._conn.execute(
                "INSERT INTO proposals (proposal_id, workspace_id, voiceprint_id, proposed_email,"
                " proposed_by, proposed_name, status, created_at) VALUES (?,?,?,?,?,?,'pending',?)",
                (pid, workspace_id, voiceprint_id, email, by, proposed_name, _now()),
            )
            self._conn.commit()
            return self.get_proposal(pid)

    def confirm_proposal(self, proposal_id: str, actor: str | None = None) -> dict | None:
        """Confirm a pending proposal → bind owner_email + name.

        Reuses `claim_owner` + `set_name` (both audited → binding_audit + usage_ledger), so
        a confirmed binding is reversible and traceable like any other name bind. Idempotent
        on an already-confirmed proposal (the bind runs exactly once). Returns
        `{voiceprint_id, name, owner_email}`, or None if the proposal is unknown.

        Consent-bypass guard: when `identify_allowed=False` (the data subject chose
        stay-anonymous), confirm still binds `owner_email` but writes/surfaces NO name —
        revoked consent can never be re-attached by a later tag. consent_resolve gates on
        the same flag, so the name stays withheld at read time too.
        """
        p = self.get_proposal(proposal_id)
        if p is None:
            return None
        ws, vid, email, name = p["workspace_id"], p["voiceprint_id"], p["proposed_email"], p["proposed_name"]
        allowed = self.identify_allowed(ws, vid)
        if p["status"] != "confirmed":
            # claim_owner / set_name each take self._lock — call them OUTSIDE the lock below.
            self.claim_owner(ws, vid, email)
            if allowed:
                self.set_name(ws, vid, name, actor=actor or email)
            with self._lock:
                self._conn.execute(
                    "UPDATE proposals SET status='confirmed', confirmed_at=? WHERE proposal_id=?",
                    (_now(), proposal_id),
                )
                self._conn.commit()
        return {"voiceprint_id": vid, "name": (name if allowed else None), "owner_email": email}

    def deny_proposal(self, proposal_id: str, actor: str | None = None) -> dict | None:
        """Deny a proposal → mark denied, no binding (the speaker stays `Speaker N`).

        Idempotent on an already-denied proposal. Returns `{voiceprint_id, status}` or None
        if unknown. Binding removal for an already-confirmed voiceprint is P5 (redaction),
        not here — deny only blocks an outstanding proposal.
        """
        p = self.get_proposal(proposal_id)
        if p is None:
            return None
        if p["status"] != "denied":
            with self._lock:
                self._conn.execute(
                    "UPDATE proposals SET status='denied', denied_at=? WHERE proposal_id=?",
                    (_now(), proposal_id),
                )
                self._conn.commit()
        return {"voiceprint_id": p["voiceprint_id"], "status": "denied"}

    def consent_resolve(self, workspace_id: str, voiceprint_id: str) -> dict:
        """Read-side consent projection: `{name, owner_email, visibility}`.

        The single gate Conclave trusts at projection time. `name` is None whenever the
        voiceprint is unknown, has `identify_allowed=False` (mirrors the /v1/identify gate),
        or carries no name. visibility ∈ {named, anonymous, unknown}.
        """
        vp = self.get(workspace_id, voiceprint_id)
        if vp is None:
            return {"name": None, "owner_email": None, "visibility": "unknown"}
        owner = vp.owner_email or None
        if not vp.identify_allowed or not vp.name:
            return {"name": None, "owner_email": owner, "visibility": "anonymous"}
        return {"name": vp.name, "owner_email": owner, "visibility": "named"}

    def list_pending_for_email(self, proposed_email: str) -> list[dict]:
        """All still-pending proposals tagged to an email (the consent inbox feed)."""
        email = proposed_email.lower()
        return [
            self._proposal_dict(r)
            for r in self._conn.execute(
                f"SELECT {', '.join(_PROPOSAL_COLS)} FROM proposals "
                "WHERE proposed_email=? AND status='pending' ORDER BY created_at",
                (email,),
            )
        ]

    def log_usage(
        self, workspace_id: str, voiceprint_id: str, event: str,
        consumer: str, purpose: str = "",
    ) -> None:
        """Append a usage-ledger row (decision G). Best-effort; never blocks the caller."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO usage_ledger (workspace_id, voiceprint_id, event, consumer, purpose, ts)"
                " VALUES (?,?,?,?,?,?)",
                (workspace_id, voiceprint_id, event, consumer, purpose, _now()),
            )
            self._conn.commit()

    def usage_for_voiceprint(self, workspace_id: str, voiceprint_id: str) -> list[dict]:
        """Usage history for one voiceprint (newest first) — 'how it's been used'."""
        cols = ("event", "consumer", "purpose", "ts")
        return [
            dict(zip(cols, row))
            for row in self._conn.execute(
                "SELECT event, consumer, purpose, ts FROM usage_ledger "
                "WHERE workspace_id=? AND voiceprint_id=? ORDER BY id DESC",
                (workspace_id, voiceprint_id),
            )
        ]

    # ── reads (workspace-scoped) ─────────────────────────────

    def get(self, workspace_id: str, voiceprint_id: str) -> Voiceprint | None:
        r = self._conn.execute(
            """SELECT voiceprint_id, workspace_id, name, owner_email, enroll_allowed,
                      identify_allowed, centroid, exemplars, enroll_count,
                      total_duration_sec, quality_score, created_at, updated_at, last_seen_at
               FROM voiceprints WHERE voiceprint_id=? AND workspace_id=?""",
            (voiceprint_id, workspace_id),
        ).fetchone()
        if r is None:
            return None
        vp = Voiceprint(
            voiceprint_id=r[0], workspace_id=r[1], name=r[2],
            owner_email=r[3], enroll_allowed=bool(r[4]), identify_allowed=bool(r[5]),
            centroid=self._dec_centroid(r[6]), exemplars=self._dec_exemplars(r[7]),
            enroll_count=r[8], total_duration_sec=r[9], quality_score=r[10],
            created_at=r[11], updated_at=r[12], last_seen_at=r[13],
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

    def audit_entries(self, workspace_id: str) -> list[dict]:
        """Naming history for a workspace (oldest first) — bindings are reversible + traceable."""
        cols = ("voiceprint_id", "old_name", "new_name", "actor", "ts")
        return [
            dict(zip(cols, row))
            for row in self._conn.execute(
                "SELECT voiceprint_id, old_name, new_name, actor, ts FROM binding_audit "
                "WHERE workspace_id=? ORDER BY id",
                (workspace_id,),
            )
        ]
