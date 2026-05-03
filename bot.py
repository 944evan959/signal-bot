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
DOC_DIR          = Path(os.environ.get("DOC_DIR", "docs"))
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
# Larger cap admins can opt into per-edit with `--long`. Used only when an
# admin explicitly asks for it — keeps the safe default tight while leaving
# room for legitimate large rewrites.
LLM_LARGE_MAX_TOKENS = int(os.environ.get("LLM_LARGE_MAX_TOKENS", "10000"))
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
        f"- Reply in plain prose. Avoid markdown formatting (headers, bold "
        f"asterisks, bullet lists, code fences) unless explicitly asked.\n"
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


def _call_ollama(*, model: str, messages: list,
                 max_tokens: int | None = None, **kwargs):
    """Run an Ollama chat completion with a hard wall-clock timeout.

    Raises TimeoutError if LLM_TIMEOUT elapses before the call returns —
    handle_doc and handle_mention already surface a friendly message for
    that exception class.

    max_tokens defaults to LLM_MAX_TOKENS; pass an int to override per-call
    (e.g. for admin-opt-in `--long` edits)."""
    cap = max_tokens if max_tokens is not None else LLM_MAX_TOKENS

    def _do_call():
        client = _ollama_client()
        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=cap,
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


def ai_edit_doc(doc: str, instruction: str,
                max_tokens: int | None = None) -> str:
    """Apply an edit instruction to a markdown document via Ollama.

    Raises ValueError if the input doc exceeds MAX_DOC_INPUT_CHARS (refusing
    rather than silently truncating, since the model's full-doc replacement
    would lose anything past the truncation point), or if the model returns
    empty/blank content.

    max_tokens defaults to LLM_MAX_TOKENS; passed through for admin-opt-in
    long edits."""
    if len(doc) > MAX_DOC_INPUT_CHARS:
        raise ValueError(
            f"Document is {len(doc)} chars; max for editing is "
            f"MAX_DOC_INPUT_CHARS={MAX_DOC_INPUT_CHARS}. Trim it first."
        )

    response = _call_ollama(
        model=_MODEL_EDIT,
        temperature=0.2,
        max_tokens=max_tokens,
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
    return strip_markdown(response.choices[0].message.content.strip())


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
        answer += "\n\n⚠️ No live search — answered from model training data."
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

# Documents are namespaced by chat context: each Signal group gets its own
# subdirectory under DOC_DIR, and DMs share a single `dm` namespace. Within
# a context, multiple named documents can coexist, each as <name>.md.
DEFAULT_DOC_NAME = "default"
DOC_NAME_RE      = re.compile(r"^[a-zA-Z0-9_-]{1,40}$")

# Reserved verbs can't be used as document names — would collide with the
# command parser. See handle_doc().
RESERVED_DOC_VERBS = frozenset({
    "list", "view", "edit", "share", "status", "create", "delete", "help",
})


def _group_key(group_id: str | None, is_dm: bool) -> str:
    """Filesystem-safe namespace key for the current chat context."""
    if is_dm or not group_id:
        return "dm"
    # Translate base64's `/`, `=`, `+` to safe equivalents.
    return group_id.translate(str.maketrans("/=+", "_-."))


def _ctx_dir(group_key: str) -> Path:
    d = DOC_DIR / group_key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _doc_path(group_key: str, name: str) -> Path:
    return _ctx_dir(group_key) / f"{name}.md"


def is_valid_doc_name(name: str) -> bool:
    return (bool(DOC_NAME_RE.match(name))
            and name.lower() not in RESERVED_DOC_VERBS)


def list_docs(group_key: str) -> list[str]:
    d = _ctx_dir(group_key)
    return sorted(p.stem for p in d.glob("*.md"))


def read_doc(group_key: str, name: str = DEFAULT_DOC_NAME) -> str:
    path = _doc_path(group_key, name)
    if not path.exists():
        if name == DEFAULT_DOC_NAME:
            path.write_text(STARTER_DOC)
        else:
            raise FileNotFoundError(name)
    return path.read_text()


def write_doc(group_key: str, name: str, content: str) -> None:
    """Persist a doc. Refuses oversized writes and snapshots the previous
    version to <name>.md.bak so a bad edit can be recovered manually."""
    if len(content) > MAX_DOC_CHARS:
        raise ValueError(
            f"Document content {len(content)} chars exceeds "
            f"MAX_DOC_CHARS={MAX_DOC_CHARS}"
        )
    path = _doc_path(group_key, name)
    if path.exists():
        backup = path.with_suffix(".md.bak")
        backup.write_text(path.read_text())
    path.write_text(content)
    log.info("Document saved: %s/%s (%d chars)", group_key, name, len(content))


def delete_doc(group_key: str, name: str) -> bool:
    """Remove a doc and its backup. Returns False if the doc doesn't exist."""
    path = _doc_path(group_key, name)
    if not path.exists():
        return False
    backup = path.with_suffix(".md.bak")
    path.unlink()
    if backup.exists():
        backup.unlink()
    return True


def doc_status(group_key: str, name: str = DEFAULT_DOC_NAME) -> str:
    path    = _doc_path(group_key, name)
    content = read_doc(group_key, name)
    lines   = content.splitlines()
    words   = len(content.split())
    mtime   = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    title   = next((l.lstrip("#").strip() for l in lines if l.strip()), "Untitled")
    return (
        f"📄 *{name}* — {title}\n"
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
    "`/doc`                              — view default document\n"
    "`/doc list`                         — list documents in this chat\n"
    "`/doc status`                       — default doc word count & last-modified\n"
    "`/doc share`                        — send default doc as a file *(admin)*\n"
    "`/doc edit <instruction>`           — AI-edit default doc *(admin)*\n"
    "`/doc edit --long <instruction>`    — AI-edit with larger output cap *(admin)*\n"
    "`/doc create <name>`                — create new doc *(admin)*\n"
    "`/doc <name>`                       — view named doc\n"
    "`/doc <name> status`                — named doc info\n"
    "`/doc <name> share`                 — send named doc as a file *(admin)*\n"
    "`/doc <name> edit <instruction>`    — AI-edit named doc *(admin)*\n"
    "`/doc <name> delete`                — delete named doc *(admin)*\n\n"
    "`--long` raises the per-edit output cap for legitimate large rewrites.\n"
    "Doc names: letters, digits, `_`, `-`. Each chat (group or DM) has its\n"
    "own independent document set."
)


def _doc_view(recipient: str, group_key: str, name: str,
              source: str, source_uuid: str):
    try:
        content = read_doc(group_key, name)
    except FileNotFoundError:
        send(recipient, f"❌ No document named `{name}`. Use `/doc list` to see what's available.")
        return
    if len(content) > MAX_INLINE_CHARS:
        hint = ("use `/doc share` for the full file"
                if is_admin(source, source_uuid)
                else "ask an admin for the full file")
        send(recipient, content[:MAX_INLINE_CHARS] + f"\n\n…*(truncated — {hint})*")
    else:
        send(recipient, content)


def _doc_share(recipient: str, group_key: str, name: str):
    try:
        read_doc(group_key, name)  # ensure file exists; auto-creates default
    except FileNotFoundError:
        send(recipient, f"❌ No document named `{name}`.")
        return
    send_with_attachment(recipient, f"📎 {name}:", _doc_path(group_key, name))


def _doc_status(recipient: str, group_key: str, name: str):
    try:
        send(recipient, doc_status(group_key, name))
    except FileNotFoundError:
        send(recipient, f"❌ No document named `{name}`.")


def _doc_list(recipient: str, group_key: str):
    docs = list_docs(group_key)
    if not docs:
        send(recipient, "📄 No documents in this chat yet. Use `/doc edit <instruction>` "
                        "to create the default, or `/doc create <name>` to add a new one.")
        return
    lines = ["📄 *Documents in this chat:*"]
    for d in docs:
        lines.append(f"• `{d}`")
    send(recipient, "\n".join(lines))


def _doc_create(recipient: str, group_key: str, name: str,
                source: str, source_uuid: str):
    if not is_admin(source, source_uuid):
        send(recipient, "🔒 Only admins can create documents.")
        return
    if not is_valid_doc_name(name):
        send(recipient,
             f"❌ Invalid doc name `{name}`. Use letters, digits, `_`, or `-` "
             f"(1–40 chars). Reserved verbs not allowed: "
             f"{', '.join(sorted(RESERVED_DOC_VERBS))}.")
        return
    path = _doc_path(group_key, name)
    if path.exists():
        send(recipient, f"⚠️ Document `{name}` already exists.")
        return
    path.write_text(f"# {name}\n\nThis document is empty.\n")
    log.info("Doc created: %s/%s by source=%s", group_key, name, source)
    send(recipient, f"✅ Created `{name}`. Use `/doc {name} edit <instruction>` to write into it.")


def _doc_delete(recipient: str, group_key: str, name: str,
                source: str, source_uuid: str):
    if not is_admin(source, source_uuid):
        send(recipient, "🔒 Only admins can delete documents.")
        return
    if name == DEFAULT_DOC_NAME:
        send(recipient, f"❌ Can't delete the default document. "
                        f"Use `/doc edit clear all content` to empty it instead.")
        return
    if delete_doc(group_key, name):
        log.info("Doc deleted: %s/%s by source=%s", group_key, name, source)
        send(recipient, f"🗑️ Deleted `{name}`.")
    else:
        send(recipient, f"⚠️ No document named `{name}`.")


def _doc_edit(recipient: str, group_key: str, name: str, raw_instruction: str,
              source: str, source_uuid: str):
    if not is_admin(source, source_uuid):
        send(recipient, "🔒 Only admins can edit documents.")
        log.info("Write denied source=%s uuid=%s", source, source_uuid)
        return
    raw_instruction = raw_instruction.strip()

    # Admin opt-in: leading `--long` enables LLM_LARGE_MAX_TOKENS for this
    # one call. Lets admins explicitly request large rewrites while keeping
    # the safe default tight against runaway prompts.
    long_mode = False
    if raw_instruction.startswith("--long"):
        rest = raw_instruction[len("--long"):]
        if not rest or rest[0].isspace():
            long_mode = True
            raw_instruction = rest.strip()

    if not raw_instruction:
        send(recipient, f"⚠️ Usage: `/doc{' '+name if name != DEFAULT_DOC_NAME else ''} edit [--long] <instruction>`")
        return

    instruction = sanitize_instruction(raw_instruction)
    if len(instruction) != len(raw_instruction):
        log.info("Instruction sanitized/truncated (source=%s)", source)

    mode_tag    = " (long mode)" if long_mode else ""
    edit_tokens = LLM_LARGE_MAX_TOKENS if long_mode else LLM_MAX_TOKENS
    log.info("Edit start: %s/%s long_mode=%s max_tokens=%d source=%s",
             group_key, name, long_mode, edit_tokens, source)
    send(recipient, f"✏️ Editing `{name}`{mode_tag}…")
    nudge = threading.Timer(
        PROGRESS_NUDGE_SECONDS,
        lambda: send(recipient,
                     f"⏳ Still working… large edits can take up to {LLM_TIMEOUT}s. Hang tight."),
    )
    nudge.start()
    try:
        try:
            old_doc = read_doc(group_key, name)
        except FileNotFoundError:
            send(recipient, f"❌ No document named `{name}`. Create it with `/doc create {name}`.")
            return
        new_doc = ai_edit_doc(old_doc, instruction, max_tokens=edit_tokens)
        write_doc(group_key, name, new_doc)
        old_lines = len(old_doc.splitlines())
        new_lines = len(new_doc.splitlines())
        delta     = new_lines - old_lines
        sign      = f"+{delta}" if delta > 0 else str(delta)
        send(recipient,
             f"✅ Done. Lines: {old_lines} → {new_lines} ({sign})\n\n"
             f"{doc_status(group_key, name)}")
    except ValueError as exc:
        log.warning("Edit rejected (source=%s): %s", source, exc)
        send(recipient, f"❌ Edit failed: {exc}")
    except TimeoutError as exc:
        log.warning("Edit timed out (source=%s): %s", source, exc)
        send(recipient,
             f"⏱️ Edit timed out after {LLM_TIMEOUT}s — the model may be "
             f"overloaded or the request is too large. Try a simpler instruction.")
    except Exception as exc:
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


def handle_doc(recipient: str, args: str, source: str, source_uuid: str,
               group_key: str):
    args = args.strip()

    if not args:
        _doc_view(recipient, group_key, DEFAULT_DOC_NAME, source, source_uuid)
        return

    parts = args.split(maxsplit=1)
    first = parts[0].lower()
    rest  = parts[1] if len(parts) > 1 else ""

    # Top-level commands that don't operate on a "current" doc
    if first == "list":
        _doc_list(recipient, group_key)
        return
    if first == "help":
        send(recipient, DOC_HELP)
        return
    if first == "create":
        if not rest.strip():
            send(recipient, "⚠️ Usage: `/doc create <name>`")
            return
        _doc_create(recipient, group_key, rest.strip().split()[0], source, source_uuid)
        return

    # Verbs operating on the default doc: /doc <verb> ...
    if first in ("view", "status", "share", "edit", "delete"):
        if first == "delete":
            send(recipient, "❌ Use `/doc <name> delete` to delete a named doc.")
            return
        _dispatch_doc_verb(recipient, group_key, DEFAULT_DOC_NAME, first, rest,
                           source, source_uuid)
        return

    # Otherwise first should be a doc name: /doc <name> [verb ...]
    if not is_valid_doc_name(first):
        send(recipient, DOC_HELP)
        return
    doc_name = first

    if not rest:
        # Just a name → view it
        _doc_view(recipient, group_key, doc_name, source, source_uuid)
        return

    sub_parts = rest.split(maxsplit=1)
    verb      = sub_parts[0].lower()
    verb_args = sub_parts[1] if len(sub_parts) > 1 else ""

    if verb in ("view", "status", "share", "edit", "delete"):
        _dispatch_doc_verb(recipient, group_key, doc_name, verb, verb_args,
                           source, source_uuid)
    else:
        send(recipient,
             f"❌ Unknown verb `{verb}`. Try: view, edit, share, status, delete.")


def _dispatch_doc_verb(recipient: str, group_key: str, name: str, verb: str,
                       verb_args: str, source: str, source_uuid: str):
    if verb == "view":
        _doc_view(recipient, group_key, name, source, source_uuid)
    elif verb == "status":
        _doc_status(recipient, group_key, name)
    elif verb == "share":
        if not is_admin(source, source_uuid):
            send(recipient, "🔒 Only admins can share the document as a file.")
            log.info("Share denied source=%s uuid=%s", source, source_uuid)
            return
        _doc_share(recipient, group_key, name)
    elif verb == "edit":
        _doc_edit(recipient, group_key, name, verb_args, source, source_uuid)
    elif verb == "delete":
        _doc_delete(recipient, group_key, name, source, source_uuid)


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
            handle_doc(recipient, text[4:], source, source_uuid,
                       _group_key(None, is_dm=True))
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
        handle_doc(recipient, text[4:], source, source_uuid,
                   _group_key(group_id, is_dm=False))
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

    # Touch persona at startup so it appears in ./data/ immediately and any
    # write-permission failures surface here instead of mid-query.
    persona = read_persona()
    log.info("Persona    : %s (%d chars)", PERSONA_PATH, len(persona))
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Doc dir    : %s", DOC_DIR)

    receive_loop()


if __name__ == "__main__":
    main()
