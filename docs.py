"""
Document storage: per-chat namespace layout, validation, CRUD operations,
and tar.gz packing for the leave-the-group archive flow.

Each Signal group (and the DM context) gets its own subdirectory under
DOC_DIR. Within a context, multiple named markdown documents can coexist
as <name>.md, with optional <name>.md.bak snapshots from the most recent
edit (used by /doc restore).

Pure file-system logic — no LLM calls, no Signal API. The command
handlers in `commands.py` orchestrate these operations.
"""

from __future__ import annotations

import logging
import re
import shutil
import tarfile
from datetime import datetime
from pathlib import Path

from config import DOC_DIR, MAX_DOC_CHARS

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

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
    "list", "view", "edit", "share", "status", "create", "delete",
    "restore", "backup", "help",
})


# ── Path helpers ─────────────────────────────────────────────────────────────

def group_key(group_id: str | None, is_dm: bool) -> str:
    """Filesystem-safe namespace key for the current chat context."""
    if is_dm or not group_id:
        return "dm"
    # Translate base64's `/`, `=`, `+` to safe equivalents.
    return group_id.translate(str.maketrans("/=+", "_-."))


def _ctx_dir(gkey: str) -> Path:
    d = DOC_DIR / gkey
    d.mkdir(parents=True, exist_ok=True)
    return d


def doc_path(gkey: str, name: str) -> Path:
    return _ctx_dir(gkey) / f"{name}.md"


def is_valid_doc_name(name: str) -> bool:
    return (bool(DOC_NAME_RE.match(name))
            and name.lower() not in RESERVED_DOC_VERBS)


def list_docs(gkey: str) -> list[str]:
    d = _ctx_dir(gkey)
    return sorted(p.stem for p in d.glob("*.md"))


# ── CRUD ─────────────────────────────────────────────────────────────────────

def read_doc(gkey: str, name: str = DEFAULT_DOC_NAME) -> str:
    path = doc_path(gkey, name)
    if not path.exists():
        if name == DEFAULT_DOC_NAME:
            path.write_text(STARTER_DOC)
        else:
            raise FileNotFoundError(name)
    return path.read_text()


def write_doc(gkey: str, name: str, content: str) -> None:
    """Persist a doc. Refuses oversized writes and snapshots the previous
    version to <name>.md.bak so a bad edit can be recovered manually."""
    if len(content) > MAX_DOC_CHARS:
        raise ValueError(
            f"Document content {len(content)} chars exceeds "
            f"MAX_DOC_CHARS={MAX_DOC_CHARS}"
        )
    path = doc_path(gkey, name)
    if path.exists():
        backup = path.with_suffix(".md.bak")
        backup.write_text(path.read_text())
    path.write_text(content)
    log.info("Document saved: %s/%s (%d chars)", gkey, name, len(content))


def delete_doc(gkey: str, name: str) -> bool:
    """Remove a doc and its backup. Returns False if the doc doesn't exist."""
    path = doc_path(gkey, name)
    if not path.exists():
        return False
    backup = path.with_suffix(".md.bak")
    path.unlink()
    if backup.exists():
        backup.unlink()
    return True


def doc_status(gkey: str, name: str = DEFAULT_DOC_NAME) -> str:
    path    = doc_path(gkey, name)
    content = read_doc(gkey, name)
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


# ── Group-leave archival ─────────────────────────────────────────────────────

def pack_group_docs(gkey: str) -> Path | None:
    """Bundle a group's current doc files into a gzipped tarball under /tmp.
    Excludes .bak files (redundant — each .bak duplicates its .md). Returns
    the archive path, or None if there are no current docs to pack."""
    src_dir = DOC_DIR / gkey
    if not src_dir.exists():
        return None
    current_docs = [p for p in src_dir.iterdir()
                    if p.is_file() and not p.name.endswith(".bak")]
    if not current_docs:
        return None

    archive = Path("/tmp") / f"{gkey}-docs.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        # arcname="docs" → recipients see a clean "docs/" folder when extracted,
        # not the long internal group_key.
        for doc in current_docs:
            tar.add(doc, arcname=f"docs/{doc.name}")
    log.info("Packed %d-byte archive of %s docs (%d files)",
             archive.stat().st_size, gkey, len(current_docs))
    return archive


def delete_group_docs(gkey: str) -> bool:
    """Recursively delete a group's doc directory. Returns True if removed."""
    target = DOC_DIR / gkey
    if not target.exists():
        return False
    shutil.rmtree(target)
    log.info("Deleted doc directory: %s", target)
    return True
