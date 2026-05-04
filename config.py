"""
Centralised configuration.

Reads env vars and exposes them as module-level constants for the rest of
the codebase to import. Also kicks off `dotenv` loading and the project's
logging baseline so that any module imported after this one inherits the
same log format.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)


# ── Signal API + bot identity ────────────────────────────────────────────────

SIGNAL_SERVICE = os.environ.get("SIGNAL_SERVICE", "http://localhost:8080")
BOT_NUMBER     = os.environ["BOT_NUMBER"]

# OWNER_ID — the single number or UUID that can DM the bot directly.
# The owner is implicitly admin in every approved group as well.
OWNER_ID: str = os.environ.get("OWNER_ID", "").strip()

# BOT_UUID — the bot's own Signal UUID. Required for the bot to respond to
# @mentions in groups. When unset, the bot still logs the authors of any
# structured mentions it sees (so you can discover the UUID by sending a
# message that @-mentions the bot, then reading the log line:
#   Mentions in message: ['<bot-uuid>']
# Copy that into BOT_UUID and force-recreate.
BOT_UUID: str = os.environ.get("BOT_UUID", "").strip()


# ── On-disk state paths ──────────────────────────────────────────────────────

DOC_DIR           = Path(os.environ.get("DOC_DIR",           "docs"))
PERSONA_PATH      = Path(os.environ.get("PERSONA_PATH",      "persona.md"))
GROUPS_FILE       = Path(os.environ.get("GROUPS_FILE",       "groups.txt"))
# Per-group cached Signal admin lists. Populated only on `/admins update`,
# never automatically — explicit refresh keeps Signal-side admin changes
# from silently granting bot-admin privileges. Per-group, never global.
GROUP_ADMINS_FILE = Path(os.environ.get("GROUP_ADMINS_FILE", "group_admins.json"))

# Healthcheck marker. The bot touches this file on every WS event; the
# Docker HEALTHCHECK greps its mtime to detect a wedged connection. Stored
# under /tmp because the container's rootfs is read-only.
HEALTH_FILE       = Path(os.environ.get("HEALTH_FILE",       "/tmp/signal-bot-healthy"))


# ── User-input + output caps ─────────────────────────────────────────────────

MAX_INLINE_CHARS    = int(os.environ.get("MAX_INLINE_CHARS",    "1500"))
MAX_INSTRUCTION     = int(os.environ.get("MAX_INSTRUCTION",     "500"))
MAX_QUERY_CHARS     = int(os.environ.get("MAX_QUERY_CHARS",     "1000"))
# Hard ceiling on persisted document size. write_doc() refuses anything
# larger, so a runaway model or malicious instruction can't blow up the file.
MAX_DOC_CHARS       = int(os.environ.get("MAX_DOC_CHARS",       "50000"))
# Maximum doc size accepted as input to ai_edit_doc(). Edits on docs larger
# than this are refused with a clear error rather than silently truncated
# (silent truncation + full-output replace = data loss).
MAX_DOC_INPUT_CHARS = int(os.environ.get("MAX_DOC_INPUT_CHARS", "20000"))


# ── LLM call ceilings ────────────────────────────────────────────────────────

# Per-call ceilings on every Ollama chat completion. max_tokens stops
# runaway generation (e.g. "recite the entire bee movie script"); timeout
# stops a stalled / looping model from hanging the bot indefinitely.
# Default ~3000 tokens ≈ 12K chars output — fits most doc edits and Q&A
# while bounding runaway behavior. Bump in .env if you hit truncations
# on large /doc edit operations.
LLM_MAX_TOKENS         = int(os.environ.get("LLM_MAX_TOKENS",         "3000"))
# Larger cap admins can opt into per-edit with `--long`. Used only when an
# admin explicitly asks for it — keeps the safe default tight while leaving
# room for legitimate large rewrites.
LLM_LARGE_MAX_TOKENS   = int(os.environ.get("LLM_LARGE_MAX_TOKENS",  "10000"))
LLM_TIMEOUT            = int(os.environ.get("LLM_TIMEOUT",             "180"))
# Send a "still working" nudge if a /doc edit hasn't completed in this many
# seconds. Set to 0 to disable.
PROGRESS_NUDGE_SECONDS = int(os.environ.get("PROGRESS_NUDGE_SECONDS",   "30"))


# ── Rate limiting ────────────────────────────────────────────────────────────

# Per-sender rate limit. Owner is exempt; everyone else gets at most
# RATE_LIMIT_MAX messages in any RATE_LIMIT_WINDOW-second sliding window.
# Set RATE_LIMIT_MAX=0 to disable.
RATE_LIMIT_MAX    = int(os.environ.get("RATE_LIMIT_MAX",    "10"))
RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))


# ── Ollama / search ──────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL_EDIT      = os.environ.get("MODEL_EDIT",     "llama3.3")
MODEL_SEARCH    = os.environ.get("MODEL_SEARCH",   "llama3.3")
# SEARCH_MODE: "" | "duckduckgo" | "auto"
SEARCH_MODE     = os.environ.get("SEARCH_FALLBACK", "").lower().strip()
