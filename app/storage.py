"""
Session scratch storage for media (video / images).

Media is NOT kept in a permanent uploads/ archive. Files live under a
per-session temp directory and are deleted when the session ends.

Model weights/classes/exp also go in the same scratch dir for the session
lifetime (needed while the model is loaded).

For large videos already on the server, prefer opening by filesystem path
(no copy at all) — that is faster and uses the user's existing disk storage.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# Root for ephemeral session files (override with MEDIA_TMP_ROOT)
_DEFAULT_ROOT = Path(tempfile.gettempdir()) / "nava-model-validation"


def media_root() -> Path:
    root = Path(os.environ.get("MEDIA_TMP_ROOT", str(_DEFAULT_ROOT)))
    root.mkdir(parents=True, exist_ok=True)
    return root


def session_scratch(session_id: str) -> Path:
    path = media_root() / session_id[:16]
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_session_scratch(session_id: str) -> None:
    path = media_root() / session_id[:16]
    if path.exists():
        try:
            shutil.rmtree(path, ignore_errors=True)
            logger.info("Cleared session scratch %s", path)
        except Exception as exc:
            logger.warning("Failed to clear scratch %s: %s", path, exc)


def save_bytes_stream(file_obj, session_id: str, prefix: str, filename: str | None) -> Path:
    """Write an upload stream into session scratch (auto-cleaned later)."""
    scratch = session_scratch(session_id)
    suffix = Path(filename or "file").suffix
    dest = scratch / f"{prefix}_{uuid.uuid4().hex}{suffix}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file_obj, out)
    return dest


def save_gallery(files, session_id: str) -> list[Path]:
    gallery = session_scratch(session_id) / f"gallery_{uuid.uuid4().hex}"
    gallery.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for file in files:
        name = Path(getattr(file, "filename", None) or "image.jpg").name
        dest = gallery / name
        if dest.exists():
            dest = gallery / f"{dest.stem}_{uuid.uuid4().hex[:8]}{dest.suffix}"
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        saved.append(dest)
    return saved
