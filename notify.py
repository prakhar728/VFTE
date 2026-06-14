"""FPM-routed notify email for the P4 trust handshake (decision §10).

Thin and provider-agnostic. When `FPM_NOTIFY_EMAIL` is off (the default) it only logs —
safe for local/dev and tests, no provider needed. A real SMTP/SES send slots in behind the
flag later. The message is **notify-only** ("you've been identified in workspace X — sign in
to confirm or deny"); confidential transcript content never leaves the enclave (delivery is
in-app via Google login, architecture §10).
"""
from __future__ import annotations

import logging

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
        f"{proposed_by} tagged you as a speaker in workspace {workspace}. "
        f"Sign in to your consent dashboard to confirm or deny (proposal {proposal_id})."
    )
    if not config.NOTIFY_EMAIL:
        log.info("notify (log-only): to=%s subject=%r", proposed_email, subject)
        return False
    return _send(proposed_email, subject, body)


def _send(to: str, subject: str, body: str) -> bool:
    # TODO(Phase 2 productionization): wire a real provider (SMTP / SES). Fail loud rather
    # than silently dropping consent mail if the flag is on but nothing is configured.
    raise NotImplementedError("FPM_NOTIFY_EMAIL is on but no mail provider is configured")
