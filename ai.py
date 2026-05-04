"""
Ollama integration: chat completions for `/doc edit`, free-form Q&A for
`@mention`, and the persona system prompt that drives Q&A tone/identity.

A single shared ThreadPoolExecutor wraps every Ollama call with a hard
wall-clock timeout — the OpenAI SDK's own timeout doesn't always fire
reliably against Ollama's HTTP transport.

Web search is opt-in via SEARCH_FALLBACK; when enabled, a keyword regex
on the query decides whether to ground the answer in DDG snippets.
"""

from __future__ import annotations

import concurrent.futures
import logging
import re

from config import (
    LLM_MAX_TOKENS,
    LLM_TIMEOUT,
    MAX_DOC_INPUT_CHARS,
    MODEL_EDIT,
    MODEL_SEARCH,
    OLLAMA_BASE_URL,
    PERSONA_PATH,
    SEARCH_MODE,
)
from security import strip_markdown

log = logging.getLogger(__name__)


# ── Persona ──────────────────────────────────────────────────────────────────

EDIT_SYSTEM_PROMPT = (
    "You are a precise document editor. "
    "The user provides a markdown document and an edit instruction. "
    "Return ONLY the updated document — no commentary, no code fences, "
    "no explanation. Preserve existing formatting where possible."
)


def _starter_persona() -> str:
    return (
        "You are an AI assistant in a Signal group chat.\n"
        "\n"
        "# Identity\n"
        "- Edit this section to give yourself a name. The name in your Signal\n"
        "  profile is what users see when they @-mention you, but you can\n"
        "  introduce yourself as anything in conversation.\n"
        "- Do NOT mention being trained by Google, Meta, OpenAI, Anthropic,\n"
        "  or any other company. Identify only as the name set above.\n"
        "- Keep answers concise — under 300 words — since this is a group chat.\n"
        "- Reply in plain prose. Avoid markdown formatting (headers, bold\n"
        "  asterisks, bullet lists, code fences) unless explicitly asked.\n"
        "\n"
        "# Tone\n"
        "- Be direct and helpful. Skip filler phrases like \"Great question!\".\n"
        "- If you don't know something, say so plainly instead of guessing.\n"
        "\n"
        "# Context\n"
        "Edit this section to teach the bot about the people, places, and\n"
        "running threads in this chat. Everything below is included as system\n"
        "context for every Q&A reply, so the bot can resolve references like\n"
        "\"my brother\", \"the cabin\", or \"that project we discussed\".\n"
        "\n"
        "Useful things to put here:\n"
        "- Group purpose (family chat, work team, friend group, etc.)\n"
        "- Members and their roles or relationships to each other\n"
        "- Aliases / nicknames the model wouldn't otherwise know\n"
        "- Recurring topics, projects, or in-jokes worth recognising\n"
        "- Locations, pets, or anything else commonly referred to by shorthand\n"
    )


def read_persona() -> str:
    """Load the persona system prompt from PERSONA_PATH, auto-creating the
    starter template on first call if the file is missing. Re-read on every
    Q&A so users can edit `data/persona.md` and see changes immediately."""
    if not PERSONA_PATH.exists():
        PERSONA_PATH.write_text(_starter_persona())
        log.info("Wrote starter persona to %s", PERSONA_PATH)
    return PERSONA_PATH.read_text().strip()


# ── Ollama client + watchdog ─────────────────────────────────────────────────

# Long-running shared executor for LLM calls. Each call runs in a worker
# thread so we can enforce a hard wall-clock timeout — the OpenAI SDK's own
# timeout doesn't fire reliably against Ollama. A stalled thread keeps
# running until Ollama eventually completes, but the bot moves on.
_llm_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="llm",
)


def _ollama_client():
    from openai import OpenAI
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", max_retries=0)


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


# ── Web search ───────────────────────────────────────────────────────────────

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


def _ddg_search(query: str, max_results: int = 5) -> str:
    """Fetch DuckDuckGo snippets and return them as a context block to inject
    into the model prompt. Returns empty string on any failure so the model
    can still answer from training knowledge."""
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
    """Decide whether a query is timely enough to warrant a live web search,
    using a keyword regex. Cheap, deterministic, and easy to tune — far more
    reliable than asking a small local model to self-assess in a tight loop.
    """
    needs = bool(_TIMELY_PATTERN.search(query))
    log.info("Timeliness check: needs_search=%s", needs)
    return needs


# ── Public entrypoints ───────────────────────────────────────────────────────

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
        model=MODEL_EDIT,
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
        model=MODEL_SEARCH,
        messages=[
            {"role": "system", "content": read_persona()},
            {"role": "user",   "content": user_content},
        ],
    )
    return strip_markdown(response.choices[0].message.content.strip())


def ai_search(query: str) -> str:
    """Answer a question via Ollama.

      duckduckgo mode → always search first, then answer with snippets
      auto mode       → search only when the query matches timely-keyword
                        regex; otherwise answer from training alone
      unset           → answer from training only, with a stale-data warning
    """
    if SEARCH_MODE == "duckduckgo":
        context = _ddg_search(query)
        if not context:
            log.warning("DDG returned no results, answering from training knowledge")
        return _answer(query, context)

    if SEARCH_MODE == "auto" and _query_needs_search(query):
        context = _ddg_search(query)
        if context:
            return _answer(query, context)
        log.warning("DDG returned no results, falling back to training knowledge")
        return _answer(query)

    answer = _answer(query)
    if not SEARCH_MODE:
        answer += "\n\n⚠️ No live search — answered from model training data."
    return answer


def ai_description() -> str:
    """One-line summary of AI config, used in the startup banner."""
    mode_label = {
        "duckduckgo": "+ddg(always)",
        "auto":       "+ddg(auto)",
    }.get(SEARCH_MODE, "training-only")
    return (
        f"Ollama  edit={MODEL_EDIT}  "
        f"search={MODEL_SEARCH}({mode_label})  "
        f"url={OLLAMA_BASE_URL}"
    )
