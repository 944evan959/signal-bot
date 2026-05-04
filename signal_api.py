"""
HTTP wrappers around signal-cli-rest-api endpoints used to send messages,
attachments, and group-quit calls. Pure I/O — no auth or routing logic.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import requests

from config import BOT_NUMBER, SIGNAL_SERVICE

log = logging.getLogger(__name__)


def send(recipient: str, text: str) -> None:
    try:
        r = requests.post(
            f"{SIGNAL_SERVICE}/v2/send",
            json={"message": text, "number": BOT_NUMBER, "recipients": [recipient]},
            timeout=10,
        )
        if not r.ok:
            log.warning("send() HTTP %s recipient=%s body=%s",
                        r.status_code, recipient, r.text[:300])
    except Exception as exc:
        log.error("send() error: %s", exc)


def send_with_attachment(recipient: str, text: str, filepath: Path) -> None:
    try:
        b64 = base64.b64encode(filepath.read_bytes()).decode()
        requests.post(
            f"{SIGNAL_SERVICE}/v2/send",
            json={
                "message": text,
                "number": BOT_NUMBER,
                "recipients": [recipient],
                "base64_attachments": [b64],
            },
            timeout=15,
        )
    except Exception as exc:
        log.error("send_with_attachment() error: %s", exc)


def quit_signal_group(group_id: str) -> bool:
    """Tell signal-api to leave the Signal group. Returns True on success.

    Failures here are logged but non-fatal — the caller should still proceed
    with the local-state cleanup (removing the group from groups.txt) so the
    bot stops responding even if the Signal-side leave didn't complete."""
    wrapped = "group." + base64.b64encode(group_id.encode()).decode()
    url = f"{SIGNAL_SERVICE}/v1/groups/{BOT_NUMBER}/{wrapped}/quit"
    try:
        r = requests.post(url, timeout=10)
        if r.ok:
            log.info("Left Signal group: %s", group_id)
            return True
        log.warning("quit_signal_group HTTP %s body=%s",
                    r.status_code, r.text[:300])
        return False
    except Exception as exc:
        log.error("quit_signal_group error: %s", exc)
        return False
