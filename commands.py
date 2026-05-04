"""
Command handlers — one function per Signal command, plus the help text
strings that get sent on `/doc help` etc.

This module is the orchestration layer: it pulls together LLM calls (ai.py),
on-disk doc state (docs.py), Signal state (signal_state.py), Signal API
wrappers (signal_api.py), and authorization checks (security.py) into the
flows that make up the bot's user-facing behavior.
"""

from __future__ import annotations

import logging
import threading

from ai import ai_edit_doc, ai_search
from config import (
    LLM_LARGE_MAX_TOKENS,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT,
    MAX_INLINE_CHARS,
    OWNER_ID,
    PROGRESS_NUDGE_SECONDS,
)
from docs import (
    DEFAULT_DOC_NAME,
    RESERVED_DOC_VERBS,
    delete_doc,
    doc_path,
    doc_status,
    is_valid_doc_name,
    list_docs,
    read_doc,
    write_doc,
)
from security import is_admin, is_owner, sanitize_instruction
from signal_api import send, send_with_attachment
from signal_state import group_admins_for, refresh_group_admins

log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════
# /doc commands
# ════════════════════════════════════════════════════════════════════════════════

DOC_HELP = (
    "📄 *Document commands:*\n"
    "`/doc`                              — view default document\n"
    "`/doc list`                         — list documents in this chat\n"
    "`/doc status`                       — default doc word count & last-modified\n"
    "`/doc share`                        — send default doc as a file *(admin)*\n"
    "`/doc edit <instruction>`           — AI-edit default doc *(admin)*\n"
    "`/doc edit --long <instruction>`    — AI-edit with larger output cap *(admin)*\n"
    "`/doc backup`                       — manually snapshot current state to .bak *(admin)*\n"
    "`/doc restore`                      — undo the most recent edit (swap with .bak) *(admin)*\n"
    "`/doc create <name>`                — create new doc *(admin)*\n"
    "`/doc <name>`                       — view named doc\n"
    "`/doc <name> status`                — named doc info\n"
    "`/doc <name> share`                 — send named doc as a file *(admin)*\n"
    "`/doc <name> edit <instruction>`    — AI-edit named doc *(admin)*\n"
    "`/doc <name> backup`                — manually snapshot current state *(admin)*\n"
    "`/doc <name> restore`               — undo the most recent edit *(admin)*\n"
    "`/doc <name> delete`                — delete named doc *(admin)*\n\n"
    "`--long` raises the per-edit output cap for legitimate large rewrites.\n"
    "`restore` swaps the doc with its `.bak`; running it twice gives you\n"
    "a one-step undo/redo. Doc names: letters, digits, `_`, `-`. Each\n"
    "chat (group or DM) has its own independent document set."
)


def _doc_view(recipient: str, gkey: str, name: str,
              source: str, source_uuid: str):
    try:
        content = read_doc(gkey, name)
    except FileNotFoundError:
        send(recipient, f"❌ No document named `{name}`. Use `/doc list` to see what's available.")
        return
    if len(content) > MAX_INLINE_CHARS:
        hint = ("use `/doc share` for the full file"
                if is_admin(source, source_uuid, gkey)
                else "ask an admin for the full file")
        send(recipient, content[:MAX_INLINE_CHARS] + f"\n\n…*(truncated — {hint})*")
    else:
        send(recipient, content)


def _doc_share(recipient: str, gkey: str, name: str):
    try:
        read_doc(gkey, name)  # ensure file exists; auto-creates default
    except FileNotFoundError:
        send(recipient, f"❌ No document named `{name}`.")
        return
    send_with_attachment(recipient, f"📎 {name}:", doc_path(gkey, name))


def _doc_status(recipient: str, gkey: str, name: str):
    try:
        send(recipient, doc_status(gkey, name))
    except FileNotFoundError:
        send(recipient, f"❌ No document named `{name}`.")


def _doc_list(recipient: str, gkey: str):
    docs = list_docs(gkey)
    if not docs:
        send(recipient, "📄 No documents in this chat yet. Use `/doc edit <instruction>` "
                        "to create the default, or `/doc create <name>` to add a new one.")
        return
    lines = ["📄 *Documents in this chat:*"]
    for d in docs:
        lines.append(f"• `{d}`")
    send(recipient, "\n".join(lines))


def _doc_create(recipient: str, gkey: str, name: str,
                source: str, source_uuid: str):
    if not is_admin(source, source_uuid, gkey):
        send(recipient, "🔒 Only admins can create documents.")
        return
    if not is_valid_doc_name(name):
        send(recipient,
             f"❌ Invalid doc name `{name}`. Use letters, digits, `_`, or `-` "
             f"(1–40 chars). Reserved verbs not allowed: "
             f"{', '.join(sorted(RESERVED_DOC_VERBS))}.")
        return
    path = doc_path(gkey, name)
    if path.exists():
        send(recipient, f"⚠️ Document `{name}` already exists.")
        return
    path.write_text(f"# {name}\n\nThis document is empty.\n")
    log.info("Doc created: %s/%s by source=%s", gkey, name, source)
    send(recipient, f"✅ Created `{name}`. Use `/doc {name} edit <instruction>` to write into it.")


def _doc_delete(recipient: str, gkey: str, name: str,
                source: str, source_uuid: str):
    if not is_admin(source, source_uuid, gkey):
        send(recipient, "🔒 Only admins can delete documents.")
        return
    if name == DEFAULT_DOC_NAME:
        send(recipient, f"❌ Can't delete the default document. "
                        f"Use `/doc edit clear all content` to empty it instead.")
        return
    if delete_doc(gkey, name):
        log.info("Doc deleted: %s/%s by source=%s", gkey, name, source)
        send(recipient, f"🗑️ Deleted `{name}`.")
    else:
        send(recipient, f"⚠️ No document named `{name}`.")


def _doc_restore(recipient: str, gkey: str, name: str,
                 source: str, source_uuid: str):
    """Restore a document from its .bak file (the snapshot taken before the
    most recent /doc edit). Going through write_doc means the current state
    is itself backed up first — so restoring twice in a row toggles between
    the two states, effectively giving a one-step undo/redo."""
    if not is_admin(source, source_uuid, gkey):
        send(recipient, "🔒 Only admins can restore documents.")
        log.info("Restore denied source=%s uuid=%s", source, source_uuid)
        return

    path   = doc_path(gkey, name)
    backup = path.with_suffix(".md.bak")

    if not path.exists():
        send(recipient, f"❌ No document named `{name}`.")
        return
    if not backup.exists():
        send(recipient,
             f"❌ No backup found for `{name}`. Backups are created automatically "
             f"on `/doc edit` — there's nothing to restore from yet.")
        return

    try:
        backup_content = backup.read_text()
        old_doc        = path.read_text()
        write_doc(gkey, name, backup_content)
        old_lines = len(old_doc.splitlines())
        new_lines = len(backup_content.splitlines())
        delta     = new_lines - old_lines
        sign      = f"+{delta}" if delta > 0 else str(delta)
        log.info("Doc restored: %s/%s by source=%s", gkey, name, source)
        send(recipient,
             f"↩️ Restored `{name}` from backup. Lines: {old_lines} → {new_lines} ({sign})\n"
             f"_Run the same command again to undo this restore._\n\n"
             f"{doc_status(gkey, name)}")
    except ValueError as exc:
        log.warning("Restore rejected (source=%s): %s", source, exc)
        send(recipient, f"❌ Restore failed: {exc}")
    except Exception as exc:
        log.error("Restore failed: %s", exc)
        send(recipient, "❌ Restore failed. Please try again.")


def _doc_backup(recipient: str, gkey: str, name: str,
                source: str, source_uuid: str):
    """Manually capture the doc's current state to its .bak. Useful for
    pinning a known-good state before risky edits, independent of the
    automatic pre-edit backup. Overwrites any existing .bak."""
    if not is_admin(source, source_uuid, gkey):
        send(recipient, "🔒 Only admins can back up documents.")
        log.info("Backup denied source=%s uuid=%s", source, source_uuid)
        return

    path = doc_path(gkey, name)
    if not path.exists():
        send(recipient, f"❌ No document named `{name}`.")
        return

    backup = path.with_suffix(".md.bak")
    backup_existed = backup.exists()
    backup.write_text(path.read_text())
    log.info("Doc backed up: %s/%s by source=%s (overwrote=%s)",
             gkey, name, source, backup_existed)

    name_arg = "" if name == DEFAULT_DOC_NAME else f" {name}"
    note = (" Replaced the previous backup." if backup_existed
            else " (No prior backup existed.)")
    send(recipient,
         f"💾 Snapshot of `{name}` saved to backup.{note}\n"
         f"Run `/doc{name_arg} restore` later to revert to this state.")


def _doc_edit(recipient: str, gkey: str, name: str, raw_instruction: str,
              source: str, source_uuid: str):
    if not is_admin(source, source_uuid, gkey):
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
             gkey, name, long_mode, edit_tokens, source)
    send(recipient, f"✏️ Editing `{name}`{mode_tag}…")
    nudge = threading.Timer(
        PROGRESS_NUDGE_SECONDS,
        lambda: send(recipient,
                     f"⏳ Still working… large edits can take up to {LLM_TIMEOUT}s. Hang tight."),
    )
    nudge.start()
    try:
        try:
            old_doc = read_doc(gkey, name)
        except FileNotFoundError:
            send(recipient, f"❌ No document named `{name}`. Create it with `/doc create {name}`.")
            return
        new_doc = ai_edit_doc(old_doc, instruction, max_tokens=edit_tokens)
        write_doc(gkey, name, new_doc)
        old_lines = len(old_doc.splitlines())
        new_lines = len(new_doc.splitlines())
        delta     = new_lines - old_lines
        sign      = f"+{delta}" if delta > 0 else str(delta)
        send(recipient,
             f"✅ Done. Lines: {old_lines} → {new_lines} ({sign})\n\n"
             f"{doc_status(gkey, name)}")
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
               gkey: str):
    args = args.strip()

    if not args:
        _doc_view(recipient, gkey, DEFAULT_DOC_NAME, source, source_uuid)
        return

    parts = args.split(maxsplit=1)
    first = parts[0].lower()
    rest  = parts[1] if len(parts) > 1 else ""

    # Top-level commands that don't operate on a "current" doc
    if first == "list":
        _doc_list(recipient, gkey)
        return
    if first == "help":
        send(recipient, DOC_HELP)
        return
    if first == "create":
        if not rest.strip():
            send(recipient, "⚠️ Usage: `/doc create <name>`")
            return
        _doc_create(recipient, gkey, rest.strip().split()[0], source, source_uuid)
        return

    # Verbs operating on the default doc: /doc <verb> ...
    if first in ("view", "status", "share", "edit", "delete", "restore", "backup"):
        if first == "delete":
            send(recipient, "❌ Use `/doc <name> delete` to delete a named doc.")
            return
        _dispatch_doc_verb(recipient, gkey, DEFAULT_DOC_NAME, first, rest,
                           source, source_uuid)
        return

    # Otherwise first should be a doc name: /doc <name> [verb ...]
    if not is_valid_doc_name(first):
        send(recipient, DOC_HELP)
        return
    doc_name = first

    if not rest:
        # Just a name → view it
        _doc_view(recipient, gkey, doc_name, source, source_uuid)
        return

    sub_parts = rest.split(maxsplit=1)
    verb      = sub_parts[0].lower()
    verb_args = sub_parts[1] if len(sub_parts) > 1 else ""

    if verb in ("view", "status", "share", "edit", "delete", "restore", "backup"):
        _dispatch_doc_verb(recipient, gkey, doc_name, verb, verb_args,
                           source, source_uuid)
    else:
        send(recipient,
             f"❌ Unknown verb `{verb}`. Try: view, edit, share, status, backup, restore, delete.")


def _dispatch_doc_verb(recipient: str, gkey: str, name: str, verb: str,
                       verb_args: str, source: str, source_uuid: str):
    if verb == "view":
        _doc_view(recipient, gkey, name, source, source_uuid)
    elif verb == "status":
        _doc_status(recipient, gkey, name)
    elif verb == "share":
        if not is_admin(source, source_uuid, gkey):
            send(recipient, "🔒 Only admins can share the document as a file.")
            log.info("Share denied source=%s uuid=%s", source, source_uuid)
            return
        _doc_share(recipient, gkey, name)
    elif verb == "edit":
        _doc_edit(recipient, gkey, name, verb_args, source, source_uuid)
    elif verb == "delete":
        _doc_delete(recipient, gkey, name, source, source_uuid)
    elif verb == "restore":
        _doc_restore(recipient, gkey, name, source, source_uuid)
    elif verb == "backup":
        _doc_backup(recipient, gkey, name, source, source_uuid)


# ════════════════════════════════════════════════════════════════════════════════
# @mention / DM Q&A
# ════════════════════════════════════════════════════════════════════════════════

def handle_mention(recipient: str, query: str):
    if not query:
        send(recipient, "👋 @-mention me with your question to ask anything.")
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


# ════════════════════════════════════════════════════════════════════════════════
# /admins commands
# ════════════════════════════════════════════════════════════════════════════════

ADMINS_HELP = (
    "👥 *Admin commands:*\n"
    "`/admins`              — list owner and this group's admins\n"
    "`/admins update`       — refresh this group's admin cache from Signal *(owner only, group only)*\n\n"
    "Bot admins are derived from Signal's group-admin list, scoped per-group.\n"
    "The owner is implicitly admin in every approved group, but no one else's\n"
    "admin status carries between groups."
)


def _list_admins(recipient: str, gkey: str | None):
    group_ids = group_admins_for(gkey)

    lines = []
    lines.append(f"👑 *Owner:* `{OWNER_ID}`" if OWNER_ID else "👑 *Owner:* _not set_")

    if group_ids:
        lines.append(f"\n👥 *Group admins ({len(group_ids)}):*")
        for aid in sorted(group_ids):
            lines.append(f"• `{aid}`")
    elif gkey:
        lines.append("\n_No group admins cached for this chat. "
                     "Run `/admins update` to fetch from Signal._")
    else:
        # DM context — no group concept. Owner is the only admin here anyway.
        lines.append("\n_(In DMs, the owner is the only admin.)_")

    send(recipient, "\n".join(lines))


def handle_admins(recipient: str, args: str, source: str, source_uuid: str,
                  gkey: str | None, group_id: str | None):
    args = args.strip()

    if args in ("", "list"):
        _list_admins(recipient, gkey)
        return

    if args == "update":
        if not group_id:
            send(recipient, "⚠️ `/admins update` only works in a group chat.")
            return
        if not is_owner(source, source_uuid):
            send(recipient, "🔒 Only the owner can refresh the group admin list.")
            log.info("Group-admin refresh denied source=%s", source)
            return
        try:
            added, removed = refresh_group_admins(gkey, group_id)
        except Exception as exc:
            log.warning("Group-admin refresh failed: %s", exc)
            send(recipient, f"❌ Refresh failed: {exc}")
            return
        log.info("Group admins refreshed for %s: +%d -%d by owner=%s",
                 gkey, len(added), len(removed), source)
        bullets = []
        if added:
            bullets.append("Added: " + ", ".join(f"`{a}`" for a in sorted(added)))
        if removed:
            bullets.append("Removed: " + ", ".join(f"`{a}`" for a in sorted(removed)))
        if not bullets:
            bullets.append("No changes.")
        send(recipient, "✅ Group admin cache refreshed.\n\n" + "\n".join(bullets))
        return

    send(recipient, ADMINS_HELP)
