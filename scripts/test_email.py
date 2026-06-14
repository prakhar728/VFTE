#!/usr/bin/env python3
"""Send ONE test email through the P4 mailer — confirm your SMTP creds work in 5 seconds.

Uses the exact same code path the real notify uses (notify._send over STARTTLS). Run it with
the same env vars you'd use for the demo:

    export FPM_NOTIFY_EMAIL=1
    export FPM_SMTP_HOST=smtp.gmail.com FPM_SMTP_PORT=587
    export FPM_SMTP_USER=you@gmail.com FPM_SMTP_PASS='app-password' FPM_NOTIFY_FROM=you@gmail.com
    python scripts/test_email.py --to someone@wherever.com

Prints SENT on success, or FAILED + the exact error (almost always: not using a Google
App Password, or 2-Step Verification not enabled).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

import config
import notify


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", required=True, help="recipient email (a real inbox you can check)")
    a = ap.parse_args()

    print("SMTP config:")
    print(f"  host = {config.SMTP_HOST or '(unset!)'}:{config.SMTP_PORT}")
    print(f"  user = {config.SMTP_USER or '(unset)'}")
    print(f"  from = {config.NOTIFY_FROM or '(unset!)'}")
    print(f"  pass = {'set' if config.SMTP_PASS else '(unset!)'}")
    if not (config.SMTP_HOST and config.NOTIFY_FROM):
        print("\nFAILED: set FPM_SMTP_HOST and FPM_SMTP_USER/FPM_NOTIFY_FROM first.")
        sys.exit(1)

    try:
        notify._send(
            a.to,
            "FPM test email — P4 demo",
            "This is a test from the P4 demo mailer. If you got this, your SMTP works.\n\n"
            f"Consent dashboard: {config.DASHBOARD_URL}\n",
        )
    except Exception as e:  # noqa: BLE001
        print(f"\nFAILED: {type(e).__name__}: {e}")
        sys.exit(1)
    print(f"\nSENT → {a.to}.  Check that inbox (and spam).")


if __name__ == "__main__":
    main()
