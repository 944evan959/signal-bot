#!/usr/bin/env python3
"""
Signal Ollama Bot
─────────────────
Sits in a Signal group chat and responds to:
  /doc [view|status|share|edit <instruction>]  — document management
  @<BOT_NAME> <question>                        — AI-powered Q&A

Inbound messages arrive via signal-cli-rest-api's json-rpc websocket
(/v1/receive), so message delivery is push-based with no polling overhead.

AI backend: Ollama (local, no API key required)

Web search modes (SEARCH_FALLBACK):
  unset        — training knowledge only (with stale-data warning footer)
  duckduckgo   — always fetch DDG snippets before answering
  auto         — keyword heuristic on the query: search only when timely

Security model:
  - Only responds inside the allowlisted group (or DMs from OWNER_ID)
  - ADMIN_IDS can use write commands (/doc edit)
  - All group members can use read commands + @mention
  - Unlisted groups are silently ignored
  - Instruction length is capped to prevent prompt injection
  - Message content is never written to logs
"""

from __future__ import annotations

import json
import os
import re
import time
import logging
import base64
import threading
import concurrent.futures
import requests
import websocket
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════════

SIGNAL_SERVICE   = os.environ.get("SIGNAL_SERVICE", "http://localhost:8080")
BOT_NUMBER       = os.environ["BOT_NUMBER"]
BOT_NAME         = os.environ.get("BOT_NAME", "Claude")
DOC_PATH         = Path(os.environ.get("DOC_PATH", "document.md"))
PERSONA_PATH     = Path(os.environ.get("PERSONA_PATH", "persona.md"))
GROUPS_FILE      = Path(os.environ.get("GROUPS_FILE", "groups.txt"))
ADMINS_FILE      = Path(os.environ.get("ADMINS_FILE", "admins.txt"))
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
# Per-call ceilings on every Ollama chat completion. max_tokens stops
# runaway generation (e.g. "recite the entire bee movie script"); timeout
# stops a stalled / looping model from hanging the bot indefinitely.
# Default ~3000 tokens ≈ 12K chars output — fits most doc edits and Q&A
# while bounding runaway behavior. Bump in .env if you hit truncations
# on large /doc edit operations.
LLM_MAX_TOKENS      = int(os.environ.get("LLM_MAX_TOKENS", "3000"))
LLM_TIMEOUT         = int(os.environ.get("LLM_TIMEOUT",   "180"))
# Send a "still working" nudge if a /doc edit hasn't completed in this many
# seconds. Set to 0 to disable.
PROGRESS_NUDGE_SECONDS = int(os.environ.get("PROGRESS_NUDGE_SECONDS", "30"))

# Per-sender rate limit. Owner is exempt; everyone else gets at most
# RATE_LIMIT_MAX messages in any RATE_LIMIT_WINDOW-second sliding window.
# Set RATE_LIMIT_MAX=0 to disable.
RATE_LIMIT_MAX    = int(os.environ.get("RATE_LIMIT_MAX",    "10"))
RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))

ALLOWED_GROUP : str      = os.environ.get("ALLOWED_GROUP", "").strip()
ADMIN_IDS     : set[str] = {
    n.strip() for n in os.environ.get("ADMIN_IDS", "").split(",") if n.strip()
}
# OWNER_ID — the single number or UUID that can DM the bot directly.
# Accepts phone number (+15551234567) or UUID, same as ADMIN_IDS.
OWNER_ID      : str      = os.environ.get("OWNER_ID", "").strip()

# BOT_UUID — the bot's own Signal UUID. When set, the bot responds to native
# Signal @mentions of itself (selected via the in-app contact picker). When
# unset, the bot falls back to text-matching `@<BOT_NAME>` and logs the
# author of any structured mentions it sees so the user can discover the
# right value.
BOT_UUID      : str      = os.environ.get("BOT_UUID", "").strip()


# ════════════════════════════════════════════════════════════════════════════════
# OLLAMA AI
# ════════════════════════════════════════════════════════════════════════════════

EDIT_SYSTEM_PROMPT = (
    "You are a precise document editor. "
    "The user provides a markdown document and an edit instruction. "
    "Return ONLY the updated document — no commentary, no code fences, "
    "no explanation. Preserve existing formatting where possible."
)

# The Q&A system prompt is loaded from PERSONA_PATH at every query so users
# can edit `data/persona.md` on the host and see changes without a restart.
# A starter template is auto-written if the file doesn't exist.

def _starter_persona() -> str:
    return (
        f"You are {BOT_NAME}, an AI assistant in a Signal group chat.\n"
        f"\n"
        f"# Identity\n"
        f"- When asked who you are, identify yourself as {BOT_NAME}.\n"
        f"- Do NOT mention being trained by Google, Meta, OpenAI, Anthropic, "
        f"or any other company. You are simply {BOT_NAME}.\n"
        f"- Keep answers concise — under 300 words — since this is a group chat.\n"
        f"\n"
        f"# Tone\n"
        f"- Be direct and helpful. Skip filler phrases like \"Great question!\".\n"
        f"- If you don't know something, say so plainly instead of guessing.\n"
        f"\n"
        f"# Context\n"
        f"Edit this section to teach the bot about the people, places, and\n"
        f"running threads in this chat. Everything below is included as system\n"
        f"context for every Q&A reply, so the bot can resolve references like\n"
        f"\"my brother\", \"the cabin\", or \"that project we discussed\".\n"
        f"\n"
        f"Useful things to put here:\n"
        f"- Group purpose (family chat, work team, friend group, etc.)\n"
        f"- Members and their roles or relationships to each other\n"
        f"- Aliases / nicknames the model wouldn't otherwise know\n"
        f"- Recurring topics, projects, or in-jokes worth recognising\n"
        f"- Locations, pets, or anything else commonly referred to by shorthand\n"
    )


def read_persona() -> str:
    if not PERSONA_PATH.exists():
        PERSONA_PATH.write_text(_starter_persona())
        log.info("Wrote starter persona to %s", PERSONA_PATH)
    return PERSONA_PATH.read_text().strip()

# Auto-mode trigger heuristic. Matches queries whose answer is likely to depend
# on data more recent than the model's training cutoff. When this matches, the
# bot fetches DDG snippets before answering; when it doesn't, it answers from
# training knowledge alone. Tune freely — false positives are cheap (an extra
# search), false negatives can mean stale answers.
_TIMELY_PATTERN = re.compile(
    r"\b("
    r"today|tonight|tomorrow|yesterday|"
    r"now|currently|current|latest|recent(ly)?|"
    r"this\s+(week|month|year)|"
    r"right\s+now|live|"
    r"price|cost|"
    r"news|headline|"
    r"version|release|"
    r"score|standings|"
    r"weather|forecast|"
    r"stock|crypto|exchange\s+rate|"
    r"who\s+won|is\s+\w+\s+(alive|dead)|"
    r"in\s+20[2-9]\d"
    r")\b",
    re.IGNORECASE,
)

# Ollama config — read once at startup
_OLLAMA_BASE_URL  = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
_MODEL_EDIT       = os.environ.get("MODEL_EDIT",       "llama3.3")
_MODEL_SEARCH     = os.environ.get("MODEL_SEARCH",     "llama3.3")
_SEARCH_MODE      = os.environ.get("SEARCH_FALLBACK",  "").lower().strip()
# _SEARCH_MODE: "" | "duckduckgo" | "auto"

def _ollama_client():
    from openai import OpenAI
    return OpenAI(base_url=_OLLAMA_BASE_URL, api_key="ollama", max_retries=0)


# Long-running shared executor for LLM calls. Each call runs in a worker
# thread so we can enforce a hard wall-clock timeout — the OpenAI SDK's own
# timeout doesn't fire reliably against Ollama. A stalled thread keeps
# running until Ollama eventually completes, but the bot moves on.
_llm_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="llm",
)


def _call_ollama(*, model: str, messages: list, **kwargs):
    """Run an Ollama chat completion with a hard wall-clock timeout.

    Raises TimeoutError if LLM_TIMEOUT elapses before the call returns —
    handle_doc and handle_mention already surface a friendly message for
    that exception class."""
    def _do_call():
        client = _ollama_client()
        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=LLM_MAX_TOKENS,
            timeout=LLM_TIMEOUT,    # SDK's own timeout, best-effort
            **kwargs,
        )

    future = _llm_executor.submit(_do_call)
    try:
        # Give the SDK a small grace window past LLM_TIMEOUT to surface
        # APITimeoutError cleanly before our watchdog fires.
        return future.result(timeout=LLM_TIMEOUT + 5)
    except concurrent.futures.TimeoutError:
        future.cancel()
        raise TimeoutError(f"Ollama call exceeded {LLM_TIMEOUT}s")


def _ddg_search(query: str, max_results: int = 5) -> str:
    """
    Fetch DuckDuckGo snippets and return them as a context block to inject
    into the model prompt. Returns empty string on any failure so the model
    can still answer from training knowledge.
    """
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return ""
        lines = []
        for r in results:
            title = r.get("title", "")
            body  = r.get("body", "")
            href  = r.get("href", "")
            lines.append(f"• {title}\n  {body}\n  {href}")
        return "Web search results:\n\n" + "\n\n".join(lines)
    except Exception as exc:
        log.warning("DuckDuckGo search failed: %s", exc)
        return ""


def _query_needs_search(query: str) -> bool:
    """
    Decide whether a query is timely enough to warrant a live web search,
    using a keyword regex. Cheap, deterministic, and easy to tune — far more
    reliable than asking a small local model to self-assess in a tight loop.
    """
    needs = bool(_TIMELY_PATTERN.search(query))
    log.info("Timeliness check: needs_search=%s", needs)
    return needs


def ai_edit_doc(doc: str, instruction: str) -> str:
    """Apply an edit instruction to a markdown document via Ollama.

    Raises ValueError if the input doc exceeds MAX_DOC_INPUT_CHARS (refusing
    rather than silently truncating, since the model's full-doc replacement
    would lose anything past the truncation point), or if the model returns
    empty/blank content."""
    if len(doc) > MAX_DOC_INPUT_CHARS:
        raise ValueError(
            f"Document is {len(doc)} chars; max for editing is "
            f"MAX_DOC_INPUT_CHARS={MAX_DOC_INPUT_CHARS}. Trim it first."
        )

    response = _call_ollama(
        model=_MODEL_EDIT,
        temperature=0.2,
        messages=[
            {"role": "system", "content": EDIT_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Document:\n\n{doc}\n\nInstruction: {instruction}"},
        ],
    )
    new_doc = response.choices[0].message.content.strip()
    if not new_doc:
        raise ValueError("Model returned empty content")
    return new_doc


def _answer(query: str, ddg_context: str = "") -> str:
    """Single Ollama chat call. If ddg_context is non-empty, the model is told
    to ground its answer in those snippets."""
    if ddg_context:
        user_content = (
            f"{ddg_context}\n\n"
            f"Using the search results above where relevant, "
            f"answer this question:\n{query}"
        )
    else:
        user_content = query

    response = _call_ollama(
        model=_MODEL_SEARCH,
        messages=[
            {"role": "system", "content": read_persona()},
            {"role": "user",   "content": user_content},
        ],
    )
    return response.choices[0].message.content.strip()


def ai_search(query: str) -> str:
    """
    Answer a question via Ollama.
      duckduckgo mode → always search first, then answer with snippets
      auto mode       → search only when the query matches timely-keyword
                        regex; otherwise answer from training alone
      unset           → answer from training only, with a stale-data warning
    """
    if _SEARCH_MODE == "duckduckgo":
        context = _ddg_search(query)
        if not context:
            log.warning("DDG returned no results, answering from training knowledge")
        return _answer(query, context)

    if _SEARCH_MODE == "auto" and _query_needs_search(query):
        context = _ddg_search(query)
        if context:
            return _answer(query, context)
        log.warning("DDG returned no results, falling back to training knowledge")
        return _answer(query)

    answer = _answer(query)
    if not _SEARCH_MODE:
        answer += "\n\n_⚠️ No live search — answered from model training data._"
    return answer


def _ai_description() -> str:
    mode_label = {
        "duckduckgo": "+ddg(always)",
        "auto":       "+ddg(auto)",
    }.get(_SEARCH_MODE, "training-only")
    return (
        f"Ollama  edit={_MODEL_EDIT}  "
        f"search={_MODEL_SEARCH}({mode_label})  "
        f"url={_OLLAMA_BASE_URL}"
    )


# ════════════════════════════════════════════════════════════════════════════════
# SECURITY HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def _load_known_groups() -> set[str]:
    """Read the persisted set of approved group IDs."""
    if not GROUPS_FILE.exists():
        return set()
    return {ln.strip() for ln in GROUPS_FILE.read_text().splitlines() if ln.strip()}


def _save_known_group(group_id: str) -> None:
    groups = _load_known_groups()
    if group_id in groups:
        return
    groups.add(group_id)
    GROUPS_FILE.write_text("\n".join(sorted(groups)) + "\n")


def _remove_known_group(group_id: str) -> bool:
    groups = _load_known_groups()
    if group_id not in groups:
        return False
    groups.discard(group_id)
    if groups:
        GROUPS_FILE.write_text("\n".join(sorted(groups)) + "\n")
    else:
        GROUPS_FILE.write_text("")
    return True


def is_allowed_group(group_id: str | None) -> bool:
    if not group_id:
        return False
    if ALLOWED_GROUP and group_id == ALLOWED_GROUP:
        return True
    return group_id in _load_known_groups()

def _load_file_admins() -> set[str]:
    """Read the runtime-managed admin set from ADMINS_FILE."""
    if not ADMINS_FILE.exists():
        return set()
    return {ln.strip() for ln in ADMINS_FILE.read_text().splitlines() if ln.strip()}


def _save_file_admin(admin_id: str) -> bool:
    """Add admin_id to ADMINS_FILE. Returns True if added, False if already present."""
    ids = _load_file_admins()
    if admin_id in ids:
        return False
    ids.add(admin_id)
    ADMINS_FILE.write_text("\n".join(sorted(ids)) + "\n")
    return True


def _remove_file_admin(admin_id: str) -> bool:
    """Remove admin_id from ADMINS_FILE. Returns True if removed, False if absent."""
    ids = _load_file_admins()
    if admin_id not in ids:
        return False
    ids.discard(admin_id)
    ADMINS_FILE.write_text(("\n".join(sorted(ids)) + "\n") if ids else "")
    return True


def _all_admins() -> set[str]:
    """Combined admin set: env-configured plus file-persisted."""
    return ADMIN_IDS | _load_file_admins()


def is_admin(source: str, source_uuid: str) -> bool:
    """Checks phone number OR UUID so username-only Signal users are supported."""
    ids = _all_admins()
    if not ids:
        log.warning("No admins configured — all write commands disabled.")
        return False
    return bool(ids & {source, source_uuid})


def is_owner(source: str, source_uuid: str) -> bool:
    return bool(OWNER_ID) and OWNER_ID in {source, source_uuid}


def _sanitize_user_text(raw: str, max_len: int) -> str:
    """Cap length, strip nulls, and strip role-prefix injection markers."""
    text = raw[:max_len].replace("\x00", "")
    text = re.sub(r"(?i)(system\s*:|user\s*:|assistant\s*:)", "", text)
    return text.strip()


def sanitize_instruction(raw: str) -> str:
    return _sanitize_user_text(raw, MAX_INSTRUCTION)


def sanitize_query(raw: str) -> str:
    return _sanitize_user_text(raw, MAX_QUERY_CHARS)


# Per-sender sliding-window rate limiter. Owner is exempt.
_rate_buckets: dict[str, list[float]] = {}


def is_rate_limited(source: str, source_uuid: str) -> bool:
    if not RATE_LIMIT_MAX or is_owner(source, source_uuid):
        return False
    key  = source_uuid or source
    now  = time.time()
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


# ════════════════════════════════════════════════════════════════════════════════
# DOCUMENT HELPERS
# ════════════════════════════════════════════════════════════════════════════════

STARTER_DOC = (
    "# Shared Document\n\n"
    "This document is empty.\n"
    "Use `/doc edit add an introduction` to get started.\n"
)

def read_doc() -> str:
    if not DOC_PATH.exists():
        DOC_PATH.write_text(STARTER_DOC)
    return DOC_PATH.read_text()

def write_doc(content: str):
    """Persist content to DOC_PATH. Refuses oversized writes and snapshots the
    previous version to DOC_PATH.bak so a bad edit can be recovered manually."""
    if len(content) > MAX_DOC_CHARS:
        raise ValueError(
            f"Document content {len(content)} chars exceeds "
            f"MAX_DOC_CHARS={MAX_DOC_CHARS}"
        )
    if DOC_PATH.exists():
        backup = DOC_PATH.with_suffix(DOC_PATH.suffix + ".bak")
        backup.write_text(DOC_PATH.read_text())
    DOC_PATH.write_text(content)
    log.info("Document saved (%d chars)", len(content))

def doc_status() -> str:
    content = read_doc()
    lines   = content.splitlines()
    words   = len(content.split())
    mtime   = datetime.fromtimestamp(DOC_PATH.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    title   = next((l.lstrip("#").strip() for l in lines if l.strip()), "Untitled")
    return (
        f"📄 *{title}*\n"
        f"Lines : {len(lines)}\n"
        f"Words : {words}\n"
        f"Saved : {mtime}"
    )


# ════════════════════════════════════════════════════════════════════════════════
# SIGNAL API HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def send(recipient: str, text: str):
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

def send_with_attachment(recipient: str, text: str, filepath: Path):
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


# ════════════════════════════════════════════════════════════════════════════════
# COMMAND HANDLERS
# ════════════════════════════════════════════════════════════════════════════════

DOC_HELP = (
    "📄 *Document commands:*\n"
    "`/doc`                    — view document inline\n"
    "`/doc status`             — word count & last-modified\n"
    "`/doc share`              — send document as a file *(admins only)*\n"
    "`/doc edit <instruction>` — AI-powered edit *(admins only)*\n\n"
    "*Examples:*\n"
    "`/doc edit add a section on Q3 goals`\n"
    "`/doc edit rewrite the intro to be more concise`"
)

def handle_doc(recipient: str, args: str, source: str, source_uuid: str):
    args = args.strip()

    if args in ("", "view"):
        content = read_doc()
        if len(content) > MAX_INLINE_CHARS:
            hint = ("use `/doc share` for the full file"
                    if is_admin(source, source_uuid)
                    else "ask an admin for the full file")
            send(recipient, content[:MAX_INLINE_CHARS] + f"\n\n…*(truncated — {hint})*")
        else:
            send(recipient, content)

    elif args == "status":
        send(recipient, doc_status())

    elif args == "share":
        if not is_admin(source, source_uuid):
            send(recipient, "🔒 Only admins can share the document as a file.")
            log.info("Share denied source=%s uuid=%s", source, source_uuid)
            return
        read_doc()  # Ensure DOC_PATH exists (writes the starter on first use)
        send_with_attachment(recipient, "📎 Current document:", DOC_PATH)

    elif args.lower().startswith("edit "):
        if not is_admin(source, source_uuid):
            send(recipient, "🔒 Only admins can edit the document.")
            log.info("Write denied source=%s uuid=%s", source, source_uuid)
            return

        raw_instruction = args[5:].strip()
        if not raw_instruction:
            send(recipient, "⚠️ Usage: `/doc edit <instruction>`")
            return

        instruction = sanitize_instruction(raw_instruction)
        if len(instruction) != len(raw_instruction):
            log.info("Instruction sanitized/truncated (source=%s)", source)

        send(recipient, "✏️ Editing document…")
        # Schedule a "still working" nudge if the call hasn't completed in
        # PROGRESS_NUDGE_SECONDS. Cancelled by finally on success/failure.
        nudge = threading.Timer(
            PROGRESS_NUDGE_SECONDS,
            lambda: send(recipient,
                         f"⏳ Still working… large edits can take up to "
                         f"{LLM_TIMEOUT}s. Hang tight."),
        )
        nudge.start()
        try:
            old_doc = read_doc()
            new_doc = ai_edit_doc(old_doc, instruction)
            write_doc(new_doc)
            old_lines = len(old_doc.splitlines())
            new_lines = len(new_doc.splitlines())
            delta     = new_lines - old_lines
            sign      = f"+{delta}" if delta > 0 else str(delta)
            send(recipient,
                 f"✅ Done. Lines: {old_lines} → {new_lines} ({sign})\n\n"
                 f"{doc_status()}")
        except ValueError as exc:
            log.warning("Edit rejected (source=%s): %s", source, exc)
            send(recipient, f"❌ Edit failed: {exc}")
        except (TimeoutError, requests.Timeout) as exc:
            log.warning("Edit timed out (source=%s): %s", source, exc)
            send(recipient,
                 f"⏱️ Edit timed out after {LLM_TIMEOUT}s — the model may be "
                 f"overloaded or the request is too large. Try a simpler instruction.")
        except Exception as exc:
            # The OpenAI client raises its own APITimeoutError; catch by name.
            if "Timeout" in type(exc).__name__:
                log.warning("Edit timed out (source=%s): %s", source, exc)
                send(recipient,
                     f"⏱️ Edit timed out after {LLM_TIMEOUT}s — the model may be "
                     f"overloaded or the request is too large. Try a simpler instruction.")
            else:
                log.error("ai_edit_doc failed: %s", exc)
                send(recipient, "❌ Edit failed. Please try again.")
        finally:
            nudge.cancel()

    else:
        send(recipient, DOC_HELP)


def handle_mention(recipient: str, query: str):
    if not query:
        send(recipient, f"👋 Ask me anything: `@{BOT_NAME} <your question>`")
        return
    send(recipient, "🔍 Looking that up…")
    nudge = threading.Timer(
        PROGRESS_NUDGE_SECONDS,
        lambda: send(recipient,
                     f"⏳ Still thinking… replies can take up to {LLM_TIMEOUT}s."),
    )
    nudge.start()
    try:
        answer = ai_search(query)
        send(recipient, answer or "🤷 I couldn't find a good answer.")
    except TimeoutError as exc:
        log.warning("Search timed out: %s", exc)
        send(recipient,
             f"⏱️ Reply timed out after {LLM_TIMEOUT}s — model may be overloaded "
             f"or the question is too complex.")
    except Exception as exc:
        log.error("ai_search failed: %s", exc)
        send(recipient, "❌ Search failed. Please try again.")
    finally:
        nudge.cancel()


ADMINS_HELP = (
    "👥 *Admin commands:*\n"
    "`/admins`              — list owner and admins\n"
    "`/admins add <id>`     — add an admin *(owner only)*\n"
    "`/admins remove <id>`  — remove a runtime-added admin *(owner only)*\n\n"
    "Use phone numbers (`+15551234567`) or UUIDs. Admins added via `.env`\n"
    "are pinned and can't be removed at runtime — edit `.env` instead."
)


def _list_admins(recipient: str):
    file_ids = _load_file_admins()
    all_ids  = ADMIN_IDS | file_ids

    lines = []
    lines.append(f"👑 *Owner:* `{OWNER_ID}`" if OWNER_ID else "👑 *Owner:* _not set_")

    if all_ids:
        lines.append(f"\n👥 *Admins ({len(all_ids)}):*")
        for aid in sorted(all_ids):
            origin = "env" if aid in ADMIN_IDS else "file"
            lines.append(f"• `{aid}` _({origin})_")
    else:
        lines.append("\n👥 *Admins:* _none configured_")

    send(recipient, "\n".join(lines))


def handle_admins(recipient: str, args: str, source: str, source_uuid: str):
    args = args.strip()

    if args in ("", "list"):
        _list_admins(recipient)
        return

    # Add / remove are owner-only
    if args.startswith("add ") or args.startswith("remove ") or args.startswith("rm "):
        if not is_owner(source, source_uuid):
            send(recipient, "🔒 Only the owner can modify the admin list.")
            log.info("Admin-mutation denied source=%s uuid=%s", source, source_uuid)
            return

    if args.startswith("add "):
        admin_id = args[4:].strip()
        if not admin_id or " " in admin_id or len(admin_id) > 128:
            send(recipient, "⚠️ Usage: `/admins add <phone-or-uuid>`")
            return
        if admin_id in ADMIN_IDS:
            send(recipient, f"⚠️ `{admin_id}` is already configured via `.env`.")
            return
        if _save_file_admin(admin_id):
            log.info("Admin added: %s by owner=%s", admin_id, source)
            send(recipient, f"✅ Added admin: `{admin_id}`")
        else:
            send(recipient, f"⚠️ `{admin_id}` is already an admin.")
        return

    if args.startswith("remove ") or args.startswith("rm "):
        admin_id = args.split(maxsplit=1)[1].strip() if " " in args else ""
        if not admin_id:
            send(recipient, "⚠️ Usage: `/admins remove <phone-or-uuid>`")
            return
        if admin_id in ADMIN_IDS and admin_id not in _load_file_admins():
            send(recipient,
                 f"⚠️ `{admin_id}` is configured via `.env`. Remove it from "
                 f"`.env` and force-recreate the container instead.")
            return
        if _remove_file_admin(admin_id):
            log.info("Admin removed: %s by owner=%s", admin_id, source)
            send(recipient, f"✅ Removed admin: `{admin_id}`")
        else:
            send(recipient, f"⚠️ `{admin_id}` is not in the admin list.")
        return

    send(recipient, ADMINS_HELP)


# ════════════════════════════════════════════════════════════════════════════════
# MESSAGE ROUTER
# ════════════════════════════════════════════════════════════════════════════════

def route(envelope: dict):
    data        = envelope.get("dataMessage", {})
    text        = (data.get("message") or "").strip()
    source      = envelope.get("source", "")
    source_uuid = envelope.get("sourceUuid", "")

    if not text or source == BOT_NUMBER:
        return

    if is_rate_limited(source, source_uuid):
        log.info("Rate-limited: dropping message from source=%s", source)
        return

    group_id = (data.get("groupInfo") or {}).get("groupId")
    is_dm    = group_id is None

    # ── DM mode — owner only, no trigger prefix needed ──────────────────────
    if is_dm:
        if not is_owner(source, source_uuid):
            log.info("Ignored DM from non-owner source=%s", source)
            return
        recipient = source
        log.info("DM source=%s uuid=%s", source, source_uuid)
        if text.startswith("/doc"):
            handle_doc(recipient, text[4:], source, source_uuid)
        elif text.strip().startswith("/admins"):
            handle_admins(recipient, text.strip()[7:], source, source_uuid)
        else:
            # In DMs every message is treated as a search query — no @mention needed
            handle_mention(recipient, sanitize_query(text))
        return

    # ── Group mode ────────────────────────────────────────────────────────────
    recipient = "group." + base64.b64encode(group_id.encode()).decode()

    if not is_allowed_group(group_id):
        # The only command accepted in unapproved groups is /group claim from
        # the owner — that's how a group gets added to the known set.
        if text.strip().startswith("/group claim") and is_owner(source, source_uuid):
            _save_known_group(group_id)
            log.info("Group claimed: %s by owner=%s", group_id, source)
            send(recipient, "✅ Group claimed. The bot will now respond here.")
            return
        log.info("Ignored message outside allowed group (source=%s group_id=%s)",
                 source, group_id)
        return

    # /group leave — owner removes the current group from the approved set.
    if text.strip().startswith("/group leave") and is_owner(source, source_uuid):
        _remove_known_group(group_id)
        log.info("Group released: %s by owner=%s", group_id, source)
        send(recipient, "👋 Group released. I'll stop responding here.")
        return

    # /admins — list / add / remove. Read is admin-or-owner; mutations are
    # checked inside handle_admins (owner only).
    if text.strip().startswith("/admins"):
        if is_owner(source, source_uuid) or is_admin(source, source_uuid):
            handle_admins(recipient, text.strip()[7:], source, source_uuid)
        else:
            send(recipient, "🔒 Only admins can use the admins command.")
        return

    mentions = data.get("mentions") or []
    bot_mentioned = bool(BOT_UUID) and any(
        m.get("author") == BOT_UUID or m.get("uuid") == BOT_UUID for m in mentions
    )

    # When BOT_UUID isn't set, surface mention authors so the user can discover
    # the bot's own UUID (it's the author that consistently appears when the
    # bot is the one being mentioned).
    if mentions and not BOT_UUID:
        log.info("Mentions in message: %s",
                 [m.get("author") or m.get("uuid") for m in mentions])

    if text.startswith("/doc"):
        cmd = "doc"
    elif bot_mentioned or f"@{BOT_NAME}" in text:
        cmd = "mention"
    else:
        return

    log.info("CMD=%s source=%s uuid=%s admin=%s",
             cmd, source, source_uuid, is_admin(source, source_uuid))

    if cmd == "doc":
        handle_doc(recipient, text[4:], source, source_uuid)
    elif cmd == "mention":
        # Strip both the structured-mention placeholder (￼) and the
        # legacy @<BOT_NAME> text trigger, then sanitize before LLM input.
        cleaned = text.replace("￼", "").replace(f"@{BOT_NAME}", "").strip()
        handle_mention(recipient, sanitize_query(cleaned))


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def _extract_envelope(frame: dict) -> dict | None:
    """
    Pull the inner Signal envelope out of a websocket frame, regardless of
    whether signal-cli-rest-api wraps it as a plain {envelope, account} object
    or as a JSON-RPC notification.
    """
    params = frame.get("params")
    if isinstance(params, dict) and "envelope" in params:
        return params["envelope"]
    if "envelope" in frame:
        return frame["envelope"]
    if any(k in frame for k in ("source", "sourceUuid", "dataMessage")):
        return frame
    return None


def receive_loop():
    """
    Subscribe to signal-api's json-rpc websocket and dispatch each inbound
    message via route(). Reconnects with exponential backoff on disconnect.
    """
    ws_scheme = "wss" if SIGNAL_SERVICE.startswith("https://") else "ws"
    ws_host   = SIGNAL_SERVICE.split("://", 1)[-1].rstrip("/")
    ws_url    = f"{ws_scheme}://{ws_host}/v1/receive/{BOT_NUMBER}"

    seen    : set[str] = set()
    backoff = 2

    while True:
        try:
            log.info("WS connecting to %s", ws_url)
            ws = websocket.create_connection(ws_url, timeout=30)
            log.info("WS connected — waiting for messages")
            backoff = 2  # reset on successful connect

            while True:
                raw = ws.recv()
                if not raw:
                    break

                try:
                    frame = json.loads(raw)
                except json.JSONDecodeError as exc:
                    log.warning("WS frame not JSON: %s", exc)
                    continue

                envelope = _extract_envelope(frame)
                if not envelope:
                    log.debug("WS frame had no envelope (keys=%s)", list(frame.keys()))
                    continue

                key = f"{envelope.get('source')}:{envelope.get('timestamp', 0)}"
                if key in seen:
                    continue
                seen.add(key)
                if len(seen) > 2000:
                    seen.clear()

                try:
                    route(envelope)
                except Exception as exc:
                    log.error("route() error: %s", exc)

        except websocket.WebSocketException as exc:
            log.warning("WS disconnected (%s) — reconnecting in %ds", exc, backoff)
        except Exception as exc:
            log.error("WS unexpected error (%s) — reconnecting in %ds", exc, backoff)

        time.sleep(backoff)
        backoff = min(backoff * 2, 60)


def main():
    log.info("━━ Signal Ollama Bot started ━━")
    log.info("AI         : %s", _ai_description())
    log.info("Number     : %s", BOT_NUMBER)
    known = _load_known_groups()
    if ALLOWED_GROUP:
        known.add(ALLOWED_GROUP)
    if known:
        log.info("Groups     : %d known (%s)", len(known),
                 ", ".join(sorted(known)[:3]) + ("…" if len(known) > 3 else ""))
    elif OWNER_ID:
        log.info("Groups     : none yet — owner can run `/group claim` in a group to approve it")
    else:
        log.warning("Groups     : none, and OWNER_ID not set — bot will reject all groups")
    log.info("Admins     : %d configured", len(ADMIN_IDS))

    if not ADMIN_IDS:
        log.warning("Set ADMIN_IDS in .env to enable /doc edit.")

    if _SEARCH_MODE in ("auto", "duckduckgo"):
        try:
            import ddgs  # noqa: F401
        except ImportError:
            log.warning(
                "SEARCH_FALLBACK=%s but the 'ddgs' package is not installed. "
                "Uncomment ddgs in requirements.txt and rebuild, or set "
                "SEARCH_FALLBACK= to disable search.", _SEARCH_MODE)

    # Touch persona/doc files at startup so they appear in ./data/ immediately
    # (and any write-permission failures surface here instead of mid-query).
    persona = read_persona()
    doc     = read_doc()
    log.info("Persona    : %s (%d chars)", PERSONA_PATH, len(persona))
    log.info("Document   : %s (%d chars)", DOC_PATH,     len(doc))

    receive_loop()


if __name__ == "__main__":
    main()
