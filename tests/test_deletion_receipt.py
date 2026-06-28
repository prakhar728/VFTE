"""Cryptographic proof of fingerprint deletion (Task #1) — spec §6's 10 checks.

Drives the FastAPI app with a tmp store + an injected Ed25519 signer (fixed seed → no
keyfile side effects, deterministic signatures). Covers: receipt verifies; tamper / wrong
key fail; voiceprint row gone but the ledger "forget" row survives and matches
receipt.ledger_row_id; deletion_receipts persisted + re-fetchable; off-TEE determinism;
key endpoint stable; idempotent re-delete issues no receipt; attestation binds the pubkey;
and the Python ↔ JS reference verifiers agree byte-for-byte.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import config
from auth import SESSION_COOKIE, GoogleOAuth, SessionManager, TokenAuth
from fpm import receipts
from fpm.receipts import ReceiptSigner, canonical_bytes, owner_email_hash, verify_with_pubkey
from fpm.store.models import Voiceprint
from fpm.store.store import VoiceprintStore

DIM = 512
STORE_KEY = os.urandom(32)
SEED = bytes(range(32))                       # fixed → deterministic signer
SEED_HEX = SEED.hex()
OTHER_SEED = bytes(range(32, 64))
DOCS = Path(__file__).resolve().parents[1] / "docs"


def _unit(seed):
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _vp(ws, vid, email, seed):
    vp = Voiceprint(vid, ws, email, owner_email=email)
    for i in range(3):
        vp.add_exemplar(_unit(seed * 10 + i))
    vp.recompute_centroid()
    return vp


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DEV_LOGIN", True)
    import main
    store = VoiceprintStore(db_path=tmp_path / "vp.db", key=STORE_KEY).open()
    store.upsert(_vp("ws1", "vp_alice", "alice@x.com", 1))
    store.upsert(_vp("ws2", "vp_alice2", "alice@x.com", 2))
    store.upsert(_vp("ws1", "vp_bob", "bob@x.com", 3))
    main.app.state.store = store
    main.app.state.auth = TokenAuth.from_env("{}")
    main.app.state.sessions = SessionManager("test-secret", 3600)
    main.app.state.oauth = GoogleOAuth("", "", "http://localhost/auth/callback")
    main.app.state.receipt_signer = ReceiptSigner(SEED)
    with TestClient(main.app) as c:
        yield c, store
    store.close()
    for attr in ("store", "receipt_signer"):
        setattr(main.app.state, attr, None)


def _login(c, email):
    c.cookies.set(SESSION_COOKIE, c.app.state.sessions.issue(email))


def _forget(c, ws="ws1", vid="vp_alice"):
    return c.post(f"/v1/me/voiceprints/{ws}/{vid}/forget")


# ── §6.1 receipt is returned and verifies ────────────────────

def test_receipt_returned_and_verifies(client):
    c, _ = client
    _login(c, "alice@x.com")
    body = _forget(c).json()
    assert body["deleted"] is True
    env = body["receipt"]
    assert env["alg"] == "ed25519" and env["key_id"] == c.app.state.receipt_signer.key_id
    p = env["payload"]
    assert p["version"] == receipts.RECEIPT_VERSION
    assert p["voiceprint_id"] == "vp_alice" and p["workspace_id"] == "ws1"
    assert p["owner_email_hash"] == owner_email_hash("alice@x.com")
    assert p["embedder_dim"] == config.ID_EMBEDDING_DIM and p["embedder_model"]
    # alg + key_id are INSIDE the signed payload (covered by the signature)
    assert p["alg"] == "ed25519" and p["key_id"] == env["key_id"]
    # verifies under the signer AND under the published raw pubkey
    assert c.app.state.receipt_signer.verify(env) is True
    assert verify_with_pubkey(env, c.app.state.receipt_signer.public_key_raw()) is True


# ── §6.2 tamper any payload field → verification fails ────────

def test_tamper_any_field_fails(client):
    c, _ = client
    _login(c, "alice@x.com")
    env = _forget(c).json()["receipt"]
    raw = c.app.state.receipt_signer.public_key_raw()
    assert verify_with_pubkey(env, raw) is True
    for field, bad in {
        "voiceprint_id": "vp_evil",
        "workspace_id": "ws_evil",
        "owner_email_hash": "0" * 64,
        "ledger_row_id": (env["payload"]["ledger_row_id"] or 0) + 1,
        "deleted_at": "1999-01-01T00:00:00Z",
        "embedder_dim": 1,
        "version": "fpm-deletion-receipt-v999",
    }.items():
        tampered = json.loads(json.dumps(env))          # deep copy
        tampered["payload"][field] = bad
        assert verify_with_pubkey(tampered, raw) is False, f"{field} tamper not caught"
    # tampering the signature itself also fails
    bad_sig = json.loads(json.dumps(env))
    sig = bytearray(__import__("base64").b64decode(bad_sig["signature"]))
    sig[0] ^= 0xFF
    bad_sig["signature"] = __import__("base64").b64encode(bytes(sig)).decode()
    assert verify_with_pubkey(bad_sig, raw) is False


# ── §6.3 wrong key → verification fails ──────────────────────

def test_wrong_key_fails(client):
    c, _ = client
    _login(c, "alice@x.com")
    env = _forget(c).json()["receipt"]
    wrong = ReceiptSigner(OTHER_SEED).public_key_raw()
    assert verify_with_pubkey(env, wrong) is False


# ── §6.4 voiceprint gone; ledger forget row survives + matches ─

def test_voiceprint_gone_ledger_survives_and_matches(client):
    c, store = client
    _login(c, "alice@x.com")
    # leave an audit/proposal trail to prove it's untouched by forget
    store.propose("ws1", "vp_alice", "alice@x.com", "host@x.com", "Alice")
    audits_before = len(store.audit_entries("ws1"))
    env = _forget(c).json()["receipt"]
    # voiceprint row is gone
    assert store.get("ws1", "vp_alice") is None
    # the usage_ledger "forget" row survives and its id equals receipt.ledger_row_id
    ledger = store._conn.execute(
        "SELECT id FROM usage_ledger WHERE workspace_id=? AND voiceprint_id=? AND event='forget'",
        ("ws1", "vp_alice"),
    ).fetchall()
    assert len(ledger) == 1
    assert ledger[0][0] == env["payload"]["ledger_row_id"]
    # binding_audit + proposals untouched
    assert len(store.audit_entries("ws1")) == audits_before
    assert store._conn.execute(
        "SELECT COUNT(*) FROM proposals WHERE voiceprint_id='vp_alice'"
    ).fetchone()[0] == 1
    # other-workspace entry untouched
    assert store.get("ws2", "vp_alice2") is not None


# ── §6.5 deletion_receipts row written; GET re-returns + verifies ─

def test_receipt_persisted_and_refetchable(client):
    c, store = client
    _login(c, "alice@x.com")
    env = _forget(c).json()["receipt"]
    rows = store._conn.execute("SELECT COUNT(*) FROM deletion_receipts").fetchone()[0]
    assert rows == 1
    r = c.get("/v1/me/deletion-receipts").json()
    assert r["count"] == 1
    fetched = r["receipts"][0]
    assert fetched["payload"] == env["payload"] and fetched["signature"] == env["signature"]
    assert c.app.state.receipt_signer.verify(fetched) is True


def test_deletion_receipts_owner_scoped(client):
    c, store = client
    _login(c, "alice@x.com")
    _forget(c)                                    # alice deletes hers
    # bob's receipts are not visible to alice (owner-scoped by email hash)
    assert c.get("/v1/me/deletion-receipts").json()["count"] == 1
    _login(c, "bob@x.com")
    assert c.get("/v1/me/deletion-receipts").json()["count"] == 0


def test_deletion_receipts_requires_auth(client):
    c, _ = client
    assert c.get("/v1/me/deletion-receipts").status_code == 401


# ── §6.6 off-TEE determinism via FPM_RECEIPT_KEY ─────────────

def test_offline_determinism_via_env(monkeypatch):
    monkeypatch.setattr(config, "RECEIPT_KEY", SEED_HEX)
    a = ReceiptSigner.from_config()
    b = ReceiptSigner.from_config()
    assert a.public_key_raw() == b.public_key_raw()      # stable pubkey across "restarts"
    assert a.key_id == b.key_id == ReceiptSigner(SEED).key_id
    payload = {"version": "v", "voiceprint_id": "vp", "workspace_id": "w",
               "owner_email_hash": "h", "embedder_model": "campplus", "embedder_dim": 512,
               "deleted_at": "2026-01-01T00:00:00Z", "ledger_row_id": 7}
    assert a.sign(payload)["signature"] == b.sign(payload)["signature"]   # deterministic


def test_env_key_accepts_base64(monkeypatch):
    import base64
    monkeypatch.setattr(config, "RECEIPT_KEY", base64.b64encode(SEED).decode())
    assert ReceiptSigner.from_config().key_id == ReceiptSigner(SEED).key_id


# ── §6.7 key endpoint stable; key_id matches receipts ────────

def test_key_endpoint_stable_and_matches(client):
    c, _ = client
    k1 = c.get("/v1/deletion-receipt-key").json()
    k2 = c.get("/v1/deletion-receipt-key").json()
    assert k1 == k2                                       # stable
    assert k1["alg"] == "ed25519" and k1["in_tee"] is False
    assert k1["key_id"] == c.app.state.receipt_signer.key_id
    assert "BEGIN PUBLIC KEY" in k1["public_key"]
    assert bytes.fromhex(k1["public_key_raw_hex"]) == c.app.state.receipt_signer.public_key_raw()
    # a receipt issued now carries the same key_id the endpoint publishes
    _login(c, "alice@x.com")
    env = _forget(c).json()["receipt"]
    assert env["key_id"] == k1["key_id"]


# ── §6.8 idempotency: re-delete issues no receipt ────────────

def test_idempotent_redelete_no_receipt(client):
    c, store = client
    _login(c, "alice@x.com")
    assert _forget(c).json()["deleted"] is True
    # second delete: 403 (vp gone → _owned_or_403 raises 404 actually) — re-create to test the
    # store-level idempotency path directly: deleting an already-gone vp returns deleted False.
    res = store.delete("ws1", "vp_alice", actor="alice@x.com")
    assert res.deleted is False and res.ledger_row_id is None
    # no extra receipt, no extra forget ledger row
    assert store._conn.execute("SELECT COUNT(*) FROM deletion_receipts").fetchone()[0] == 1
    forgets = store._conn.execute(
        "SELECT COUNT(*) FROM usage_ledger WHERE voiceprint_id='vp_alice' AND event='forget'"
    ).fetchone()[0]
    assert forgets == 1


def test_forget_missing_voiceprint_404_no_receipt(client):
    c, store = client
    _login(c, "alice@x.com")
    assert c.post("/v1/me/voiceprints/ws1/vp_nope/forget").status_code == 404
    assert store._conn.execute("SELECT COUNT(*) FROM deletion_receipts").fetchone()[0] == 0


# ── §6.10 attestation binds sha256(pubkey) ───────────────────

def test_attestation_binds_pubkey(client):
    c, _ = client
    import hashlib
    r = c.get("/attestation?nonce=abc").json()
    expected = hashlib.sha256(c.app.state.receipt_signer.public_key_raw()).hexdigest()
    assert r["receipt_pubkey_sha256"] == expected
    assert r["receipt_key_id"] == c.app.state.receipt_signer.key_id
    assert r["in_tee"] is False                          # quote is a stub off-TEE


def test_attestation_passes_pubkey_as_bind(client, monkeypatch):
    """The endpoint must actually hand the raw pubkey to the quote as `bind` (not just
    echo its hash). Off-TEE the real quote is a constant stub, so we spy on the call to
    prove the pubkey is committed into report_data, not decoratively reported."""
    c, _ = client
    import fpm.enclave as enclave
    seen = {}

    def _spy(nonce="", bind=b""):
        seen["nonce"], seen["bind"] = nonce, bind
        return "stubbed"

    monkeypatch.setattr(enclave, "get_attestation_quote", _spy)
    c.get("/attestation?nonce=xyz")
    assert seen["nonce"] == "xyz"
    assert seen["bind"] == c.app.state.receipt_signer.public_key_raw()  # real pubkey bound


def test_report_data_folds_in_bind():
    """Unit-test the enclave packing: `bind` MUST be folded into report_data (32 bytes),
    distinct from the nonce-only packing, and sensitive to the bound bytes."""
    import hashlib
    from fpm.enclave import _normalize_report_data
    pk = b"\x11" * 32
    rd = _normalize_report_data("abc", pk)
    assert rd == hashlib.sha256(b"abc" + pk).digest()    # actually committed
    assert len(rd) == 32
    assert rd != _normalize_report_data("abc")           # differs from nonce-only
    assert rd != _normalize_report_data("abc", b"\x22" * 32)  # changes with bind


# ── canonicalization + key_id unit checks ────────────────────

def test_canonicalization_is_sorted_compact():
    payload = {"b": 1, "a": "x", "c": 512}
    assert canonical_bytes(payload) == b'{"a":"x","b":1,"c":512}'


def test_owner_email_hash_lowercased():
    assert owner_email_hash("Alice@X.com") == owner_email_hash("alice@x.com")


def test_key_id_derivation():
    import hashlib
    s = ReceiptSigner(SEED)
    assert s.key_id == hashlib.sha256(s.public_key_raw()).hexdigest()[:16]


# ── §6.9 Python ↔ JS reference verifiers agree byte-for-byte ──

def _write(tmp_path, env, signer):
    rp = tmp_path / "receipt.json"
    rp.write_text(json.dumps(env))
    return rp, signer.public_key_raw().hex()


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_reference_verifier_agrees(client, tmp_path):
    c, _ = client
    _login(c, "alice@x.com")
    env = _forget(c).json()["receipt"]
    rp, raw_hex = _write(tmp_path, env, c.app.state.receipt_signer)
    out = subprocess.run(
        ["node", str(DOCS / "deletion-receipt-verify.js"), str(rp), raw_hex],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    res = json.loads(out.stdout)
    assert res["valid"] is True
    assert res["key_id"] == env["key_id"]
    # byte-for-byte canonicalization agreement with Python
    import hashlib
    py_canon = hashlib.sha256(canonical_bytes(env["payload"])).hexdigest()
    assert res["canonical_sha256"] == py_canon
    # JS rejects a tampered receipt
    env["payload"]["voiceprint_id"] = "vp_evil"
    rp.write_text(json.dumps(env))
    bad = subprocess.run(
        ["node", str(DOCS / "deletion-receipt-verify.js"), str(rp), raw_hex],
        capture_output=True, text=True,
    )
    assert bad.returncode == 1 and json.loads(bad.stdout)["valid"] is False


def test_python_reference_verifier_agrees(client, tmp_path):
    c, _ = client
    _login(c, "alice@x.com")
    env = _forget(c).json()["receipt"]
    rp, raw_hex = _write(tmp_path, env, c.app.state.receipt_signer)
    out = subprocess.run(
        [os.sys.executable, str(DOCS / "deletion-receipt-verify.py"), str(rp), raw_hex],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["valid"] is True
