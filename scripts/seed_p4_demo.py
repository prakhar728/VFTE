#!/usr/bin/env python3
"""Seed an anonymous voiceprint into FPM's store for the P4 Phase-1 gate demo (no audio).

Plants a minimal anonymous voiceprint (random centroid) so the gate's tag→confirm→
consent-resolve path has a real voiceprint to bind to — without recording or enrolling.
Run with the FPM venv and the SAME FPM_DATA_DIR the demo FPM server uses (so the
encryption key + DB match). Deletes any prior demo voiceprint + its proposals first, so
re-runs always start from a clean, unbound state.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

import numpy as np

from fpm.store.models import Voiceprint
from fpm.store.store import VoiceprintStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="live-test")
    ap.add_argument("--voiceprint-id", default="vp_p4demo")
    a = ap.parse_args()

    store = VoiceprintStore().open()  # uses config DB_PATH + key (from FPM_DATA_DIR)
    # clean slate: drop any prior demo voiceprint + its proposals so the gate re-binds fresh
    store.delete(a.workspace, a.voiceprint_id, actor="seed")
    store._conn.execute(
        "DELETE FROM proposals WHERE workspace_id=? AND voiceprint_id=?",
        (a.workspace, a.voiceprint_id),
    )
    store._conn.commit()

    vp = Voiceprint(a.voiceprint_id, a.workspace, name="")  # anonymous; identify_allowed defaults True
    e = np.random.default_rng(42).standard_normal(512).astype(np.float32)
    e /= np.linalg.norm(e)
    vp.add_exemplar(e)
    vp.recompute_centroid()
    store.upsert(vp)
    store.close()
    print(f"[fpm-seed] anonymous voiceprint {a.voiceprint_id} ready in workspace {a.workspace}")


if __name__ == "__main__":
    main()
