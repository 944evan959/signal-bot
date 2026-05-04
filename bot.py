#!/usr/bin/env python3
"""
Signal Ollama Bot
─────────────────
Sits in a Signal group chat and responds to:
  /doc [view|status|share|edit <instruction>]  — document management
  @<bot> <question>                             — AI-powered Q&A
                                                  (native Signal mention,
                                                   resolved by BOT_UUID)

Inbound messages arrive via signal-cli-rest-api's json-rpc websocket
(/v1/receive), so message delivery is push-based with no polling overhead.

AI backend: Ollama (local, no API key required)

Web search modes (SEARCH_FALLBACK):
  unset        — training knowledge only (with stale-data warning footer)
  duckduckgo   — always fetch DDG snippets before answering
  auto         — keyword heuristic on the query: search only when timely

Security model:
  - Only responds in approved groups (claimed via /group claim by the owner)
    or DMs from OWNER_ID
  - Owner is admin everywhere
  - Other admins are derived per-group from Signal's group-admin list,
    cached only on `/admins update` — never automatic, never cross-group
  - All group members can use read commands + @mention
  - Unapproved groups are silently ignored
  - Instruction length is capped to prevent prompt injection
  - Message content is never written to logs

The actual implementation is split across module files in this directory:
  config.py        — env vars and paths
  signal_state.py  — groups + admin caches on disk
  signal_api.py    — HTTP wrappers (send / quit)
  security.py      — auth, sanitization, rate limit, health
  ai.py            — Ollama, persona, search
  docs.py          — doc CRUD + tar.gz packing
  commands.py      — /doc, /admins, @mention handler functions
  bot.py           — this file: route + receive loop + startup
"""

from __future__ import annotations

import base64
import json
import logging
import time

import websocket

from ai import ai_description, read_persona
from commands import handle_admins, handle_doc, handle_mention
from config import (
    BOT_NUMBER,
    BOT_UUID,
    DOC_DIR,
    OWNER_ID,
    PERSONA_PATH,
    SEARCH_MODE,
    SIGNAL_SERVICE,
)
from docs import group_key as _group_key
from security import (
    is_admin,
    is_owner,
    is_rate_limited,
    mark_healthy,
    mark_unhealthy,
    sanitize_query,
)
from signal_api import quit_signal_group, send, send_with_attachment
from signal_state import (
    is_allowed_group,
    load_group_admins_map,
    load_known_groups,
    remove_known_group,
    save_known_group,
)
from docs import delete_group_docs, pack_group_docs

log = logging.getLogger(__name__)


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
            handle_admins(recipient, text.strip()[7:], source, source_uuid,
                          gkey=None, group_id=None)
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
            save_known_group(group_id)
            log.info("Group claimed: %s by owner=%s", group_id, source)
            send(recipient, "✅ Group claimed. The bot will now respond here.")
            return
        log.info("Ignored message outside allowed group (source=%s group_id=%s)",
                 source, group_id)
        return

    # /group leave — owner has the bot exit cleanly:
    #   1. Pack the group's docs into a tar.gz so the group keeps a copy
    #   2. Send the archive + a goodbye text (must happen while still a member)
    #   3. Remove from groups.txt (stop responding)
    #   4. Quit the group on Signal's side (leave member list)
    #   5. Delete the group's local doc directory (free disk + remove residue)
    if text.strip().startswith("/group leave") and is_owner(source, source_uuid):
        gkey = _group_key(group_id, is_dm=False)

        archive = None
        try:
            archive = pack_group_docs(gkey)
        except Exception as exc:
            log.error("pack_group_docs failed for %s: %s", gkey, exc)

        if archive:
            send_with_attachment(recipient,
                                 "📦 Documents from this group (saved before I leave):",
                                 archive)
        send(recipient, "👋 Group released. Leaving the group now.")

        remove_known_group(group_id)
        log.info("Group released: %s by owner=%s", group_id, source)

        if not quit_signal_group(group_id):
            log.warning("Signal-side group quit failed for %s — local state cleared but bot may still appear in member list",
                        group_id)

        # Local cleanup. Errors are logged but non-fatal — groups.txt removal
        # already stopped the bot from responding here.
        try:
            delete_group_docs(gkey)
        except Exception as exc:
            log.error("delete_group_docs failed for %s: %s", gkey, exc)
        if archive:
            try:
                archive.unlink(missing_ok=True)
            except Exception as exc:
                log.warning("Could not unlink archive %s: %s", archive, exc)
        return

    gkey = _group_key(group_id, is_dm=False)

    # /admins — list / update. Read is admin-or-owner; mutations are checked
    # inside handle_admins (owner only).
    if text.strip().startswith("/admins"):
        if is_owner(source, source_uuid) or is_admin(source, source_uuid, gkey):
            handle_admins(recipient, text.strip()[7:], source, source_uuid,
                          gkey=gkey, group_id=group_id)
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
    elif bot_mentioned:
        cmd = "mention"
    else:
        return

    log.info("CMD=%s source=%s uuid=%s admin=%s",
             cmd, source, source_uuid, is_admin(source, source_uuid, gkey))

    if cmd == "doc":
        handle_doc(recipient, text[4:], source, source_uuid, gkey)
    elif cmd == "mention":
        # Strip the structured-mention placeholder (￼), then sanitize before
        # LLM input. Native Signal mentions don't include the bot's name as
        # text — the placeholder is the only artifact.
        cleaned = text.replace("￼", "").strip()
        handle_mention(recipient, sanitize_query(cleaned))


# ════════════════════════════════════════════════════════════════════════════════
# WEBSOCKET RECEIVE LOOP
# ════════════════════════════════════════════════════════════════════════════════

def _extract_envelope(frame: dict) -> dict | None:
    """Pull the inner Signal envelope out of a websocket frame, regardless of
    whether signal-cli-rest-api wraps it as a plain {envelope, account} object
    or as a JSON-RPC notification."""
    params = frame.get("params")
    if isinstance(params, dict) and "envelope" in params:
        return params["envelope"]
    if "envelope" in frame:
        return frame["envelope"]
    if any(k in frame for k in ("source", "sourceUuid", "dataMessage")):
        return frame
    return None


def receive_loop():
    """Subscribe to signal-api's json-rpc websocket and dispatch each inbound
    message via route(). Reconnects with exponential backoff on disconnect."""
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
            mark_healthy()
            backoff = 2  # reset on successful connect

            while True:
                raw = ws.recv()
                if not raw:
                    break

                # Refresh the healthcheck marker. Each frame received = the
                # connection is alive. The mtime check in HEALTHCHECK uses
                # this to detect a wedged WS that hasn't disconnected cleanly.
                mark_healthy()

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
            mark_unhealthy()
        except Exception as exc:
            log.error("WS unexpected error (%s) — reconnecting in %ds", exc, backoff)
            mark_unhealthy()

        time.sleep(backoff)
        backoff = min(backoff * 2, 60)


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    log.info("━━ Signal Ollama Bot started ━━")
    log.info("AI         : %s", ai_description())
    log.info("Number     : %s", BOT_NUMBER)
    known = load_known_groups()
    if known:
        log.info("Groups     : %d known (%s)", len(known),
                 ", ".join(sorted(known)[:3]) + ("…" if len(known) > 3 else ""))
    elif OWNER_ID:
        log.info("Groups     : none yet — owner can run `/group claim` in a group to approve it")
    else:
        log.warning("Groups     : none, and OWNER_ID not set — bot will reject all groups")
    cached_groups = load_group_admins_map()
    cached_admins = sum(len(v) for v in cached_groups.values())
    log.info("Admins     : %d cached across %d group(s); owner=%s",
             cached_admins, len(cached_groups),
             "set" if OWNER_ID else "UNSET")

    if not OWNER_ID:
        log.warning("OWNER_ID not set — no one can DM the bot or run admin commands.")

    if SEARCH_MODE in ("auto", "duckduckgo"):
        try:
            import ddgs  # noqa: F401
        except ImportError:
            log.warning(
                "SEARCH_FALLBACK=%s but the 'ddgs' package is not installed. "
                "Uncomment ddgs in requirements.txt and rebuild, or set "
                "SEARCH_FALLBACK= to disable search.", SEARCH_MODE)

    # Touch persona at startup so it appears in ./data/ immediately and any
    # write-permission failures surface here instead of mid-query.
    persona = read_persona()
    log.info("Persona    : %s (%d chars)", PERSONA_PATH, len(persona))
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Doc dir    : %s", DOC_DIR)

    receive_loop()


if __name__ == "__main__":
    main()
