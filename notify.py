"""FPM-routed notify email for the P4 trust handshake (decision §10).

When `FPM_NOTIFY_EMAIL` is off (the default) this only logs — safe for local/dev and tests.
When on, it sends a real email over SMTP (e.g. Gmail) using stdlib `smtplib`. The message is
**notify-only** ("you've been identified in workspace X — sign in to confirm or deny"), with a
link to the consent dashboard; confidential transcript content never leaves the enclave
(delivery is in-app via Google login, architecture §10).
"""
from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage

import config

log = logging.getLogger(__name__)


def notify_identification(
    proposed_email: str, workspace: str, proposed_by: str, proposal_id: str,
) -> bool:
    """Tell a tagged person they have a pending identification to confirm/deny.

    Returns True if an email was actually sent, False if it was log-only (flag off).
    """
    subject = f"You've been identified in workspace {workspace}"
    body = (
        f"{proposed_by} tagged you as a speaker in workspace {workspace}.\n\n"
        f"Sign in to your consent dashboard to confirm or deny:\n{config.DASHBOARD_URL}\n\n"
        f"(reference: proposal {proposal_id})\n"
    )
    if not config.NOTIFY_EMAIL:
        log.info("notify (log-only): to=%s subject=%r", proposed_email, subject)
        return False
    return _send(proposed_email, subject, body)


def _send(to: str, subject: str, body: str) -> bool:
    """Send one notify email over SMTP (STARTTLS). Raises if the flag is on but SMTP isn't
    configured — fail loud rather than silently dropping a consent email."""
    if not (config.SMTP_HOST and config.NOTIFY_FROM):
        raise RuntimeError(
            "FPM_NOTIFY_EMAIL is on but SMTP isn't configured "
            "(set FPM_SMTP_HOST + FPM_NOTIFY_FROM/FPM_SMTP_USER)"
        )
    msg = EmailMessage()
    msg["From"] = config.NOTIFY_FROM
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=15) as s:
        s.starttls(context=ssl.create_default_context())
        if config.SMTP_USER:
            s.login(config.SMTP_USER, config.SMTP_PASS)
        s.send_message(msg)
    log.info("notify sent: to=%s subject=%r", to, subject)
    return True
