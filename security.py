"""
Authorization, input sanitization, output cleaning, rate limiting, and
the healthcheck marker.

`is_admin` and `is_owner` live here (with the rate limit) because all four
are policy/authz primitives that the rest of the codebase consults; the
underlying group-admin *state* lives in `signal_state.py`.
"""

from __future__ import annotations

import logging
import re
import time

from config import (
    HEALTH_FILE,
    MAX_INSTRUCTION,
    MAX_QUERY_CHARS,
    OWNER_ID,
    RATE_LIMIT_MAX,
    RATE_LIMIT_WINDOW,
)
from signal_state import group_admins_for

log = logging.getLogger(__name__)


# ── Authorization ────────────────────────────────────────────────────────────

def is_owner(source: str, source_uuid: str) -> bool:
    return bool(OWNER_ID) and OWNER_ID in {source, source_uuid}


def is_admin(source: str, source_uuid: str, group_key: str | None = None) -> bool:
    """Admin = owner (always), or Signal-recognized admin of THIS specific
    group (cached via /admins update). Per-group only — privilege never
    extends to other groups or DMs.

    Phone number OR UUID match, so username-only Signal users are supported.
    """
    if is_owner(source, source_uuid):
        return True
    return bool(group_admins_for(group_key) & {source, source_uuid})


# ── Input sanitization ───────────────────────────────────────────────────────

def _sanitize_user_text(raw: str, max_len: int) -> str:
    """Cap length, strip nulls, and strip role-prefix injection markers."""
    text = raw[:max_len].replace("\x00", "")
    text = re.sub(r"(?i)(system\s*:|user\s*:|assistant\s*:)", "", text)
    return text.strip()


def sanitize_instruction(raw: str) -> str:
    return _sanitize_user_text(raw, MAX_INSTRUCTION)


def sanitize_query(raw: str) -> str:
    return _sanitize_user_text(raw, MAX_QUERY_CHARS)


# ── Output cleaning ──────────────────────────────────────────────────────────

def strip_markdown(text: str) -> str:
    """
    Remove the most disruptive markdown formatting from chat replies.
    Signal renders some markdown but the model output is often littered with
    `# headers`, `**bold**`, fence blocks etc. that show as literal characters.

    Handles:
      - Code fences (``` ... ```) — keep the inner code, drop the markers
      - Headers (#, ##, … at line start)
      - Bold (**x**, __x__) and italic (*x*, _x_) markers — keep the text
      - Inline links [text](url) → "text (url)"

    Does NOT strip:
      - Inline code (`x`) — renders fine and aids readability
      - Bullet markers (- or *)
      - Numbered list prefixes
    """
    # Code fences: drop the marker lines but keep the code block content
    text = re.sub(r"```[A-Za-z0-9_-]*\n?", "", text)
    text = text.replace("```", "")

    # Headers at line start: # / ## / ### / etc.
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Bold first (so the doubled-marker doesn't get half-eaten by the italic rule)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"__(.+?)__",     r"\1", text, flags=re.DOTALL)

    # Italic — only when wrapping non-whitespace content, to avoid stripping
    # bullet `* ` markers and stray asterisks used for emphasis-by-spacing.
    text = re.sub(r"(?<!\w)\*(\S(?:.*?\S)?)\*(?!\w)", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\w)_(\S(?:.*?\S)?)_(?!\w)",   r"\1", text, flags=re.DOTALL)

    # Links: [text](url) → text (url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)

    return text.strip()


# ── Rate limiting ────────────────────────────────────────────────────────────

# Per-sender sliding-window rate limiter. Owner is exempt.
_rate_buckets: dict[str, list[float]] = {}


def is_rate_limited(source: str, source_uuid: str) -> bool:
    if not RATE_LIMIT_MAX or is_owner(source, source_uuid):
        return False
    key = source_uuid or source
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW
    bucket = _rate_buckets.setdefault(key, [])
    bucket[:] = [t for t in bucket if t > cutoff]
    if len(bucket) >= RATE_LIMIT_MAX:
        return True
    bucket.append(now)
    # Cap dict size opportunistically
    if len(_rate_buckets) > 1000:
        for k in list(_rate_buckets):
            if not _rate_buckets[k]:
                del _rate_buckets[k]
    return False


# ── Healthcheck marker ───────────────────────────────────────────────────────

def mark_healthy() -> None:
    """Touch HEALTH_FILE. Docker HEALTHCHECK reads its mtime to detect
    wedged states: any mtime newer than ~90s past means a frame arrived
    recently OR the WS just connected, so the bot is healthy."""
    try:
        HEALTH_FILE.touch()
    except Exception as exc:
        log.debug("Could not update health marker: %s", exc)


def mark_unhealthy() -> None:
    """Remove the marker so HEALTHCHECK fails fast on disconnect."""
    try:
        HEALTH_FILE.unlink(missing_ok=True)
    except Exception as exc:
        log.debug("Could not clear health marker: %s", exc)
