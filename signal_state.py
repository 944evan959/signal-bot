"""
Persistent state about Signal groups: which groups are approved for the bot
to respond in, and the per-group cache of Signal-recognized admins.

All state lives on disk under paths from `config.py`. Both the approved-group
set and the per-group admin cache are managed at runtime via Signal commands
(`/group claim`, `/group leave`, `/admins update`) — none are configured by env.
"""

from __future__ import annotations

import json
import logging

import requests

from config import (
    BOT_NUMBER,
    GROUP_ADMINS_FILE,
    GROUPS_FILE,
    SIGNAL_SERVICE,
)

log = logging.getLogger(__name__)


# ── Approved groups ──────────────────────────────────────────────────────────

def load_known_groups() -> set[str]:
    """Read the persisted set of approved group IDs."""
    if not GROUPS_FILE.exists():
        return set()
    return {ln.strip() for ln in GROUPS_FILE.read_text().splitlines() if ln.strip()}


def save_known_group(group_id: str) -> None:
    groups = load_known_groups()
    if group_id in groups:
        return
    groups.add(group_id)
    GROUPS_FILE.write_text("\n".join(sorted(groups)) + "\n")


def remove_known_group(group_id: str) -> bool:
    groups = load_known_groups()
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
    return group_id in load_known_groups()


# ── Per-group admin cache ────────────────────────────────────────────────────

def load_group_admins_map() -> dict:
    """Read the persisted {group_key: [admin_id, ...]} map."""
    if not GROUP_ADMINS_FILE.exists():
        return {}
    try:
        return json.loads(GROUP_ADMINS_FILE.read_text())
    except json.JSONDecodeError:
        log.warning("Could not parse %s; treating as empty", GROUP_ADMINS_FILE)
        return {}


def save_group_admins_map(m: dict) -> None:
    GROUP_ADMINS_FILE.write_text(json.dumps(m, indent=2, sort_keys=True))


def group_admins_for(group_key: str | None) -> set[str]:
    """Cached Signal-group admins for the given group_key, or empty for DM /
    unknown groups. Refreshed only by the `/admins update` command."""
    if not group_key:
        return set()
    return set(load_group_admins_map().get(group_key, []))


def refresh_group_admins(group_key: str, group_id: str) -> tuple[set[str], set[str]]:
    """Pull the current Signal admin list for `group_id` from signal-api and
    persist it under `group_key`. Returns (added, removed) versus the previously
    cached list. Raises on any failure."""
    r = requests.get(f"{SIGNAL_SERVICE}/v1/groups/{BOT_NUMBER}", timeout=5)
    r.raise_for_status()
    new_admins: set[str] = set()
    found = False
    for g in r.json():
        if g.get("internal_id") == group_id:
            new_admins = set(g.get("admins") or [])
            found = True
            break
    if not found:
        raise ValueError(f"signal-api doesn't see this bot in group {group_id[:20]}…")

    m = load_group_admins_map()
    old_admins = set(m.get(group_key, []))
    m[group_key] = sorted(new_admins)
    save_group_admins_map(m)
    return (new_admins - old_admins, old_admins - new_admins)
