"""Per-browser session manager for multi-user isolation.

Each browser keeps one stable session (cookie + optional header). Refresh resumes
the same session instead of allocating a new GPU/CPU slot.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from app.storage import clear_session_scratch
from app.stream.live_grid import LiveGrid
from app.stream.processor import StreamProcessor

MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "3"))
SESSION_TTL_SEC = int(os.environ.get("SESSION_TTL_SEC", "7200"))  # 2 hours
# Only reclaim truly abandoned empty seats (not a page refresh)
EMPTY_IDLE_SEC = int(os.environ.get("EMPTY_IDLE_SEC", "600"))  # 10 min
# Brand-new sessions get a grace window so refresh / slow loads never steal the seat
NEW_SESSION_GRACE_SEC = int(os.environ.get("NEW_SESSION_GRACE_SEC", "180"))  # 3 min


@dataclass
class UserSession:
    session_id: str
    processor: StreamProcessor
    live_grid: LiveGrid = field(default_factory=LiveGrid)
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_seen = time.time()

    def close(self) -> None:
        try:
            self.live_grid.stop()
        finally:
            try:
                self.processor.close()
            finally:
                clear_session_scratch(self.session_id)


class SessionManager:
    def __init__(
        self,
        max_sessions: int = MAX_SESSIONS,
        ttl_sec: int = SESSION_TTL_SEC,
    ) -> None:
        self.max_sessions = max_sessions
        self.ttl_sec = ttl_sec
        self._sessions: dict[str, UserSession] = {}
        self._lock = threading.RLock()

    def _purge_expired(self) -> None:
        now = time.time()
        dead = [
            sid
            for sid, s in self._sessions.items()
            if now - s.last_seen > self.ttl_sec
            and now - s.created_at > NEW_SESSION_GRACE_SEC
        ]
        for sid in dead:
            self._close_unlocked(sid)

    def _purge_idle_empty(self) -> None:
        """Free abandoned empty seats when at capacity — never touch recent sessions."""
        if len(self._sessions) < self.max_sessions:
            return
        now = time.time()
        candidates = []
        for sid, s in self._sessions.items():
            age = now - s.created_at
            idle = now - s.last_seen
            if age < NEW_SESSION_GRACE_SEC:
                continue
            if idle < EMPTY_IDLE_SEC:
                continue
            has_model = s.processor.predictor is not None
            has_source = bool(s.processor.status().get("source_type"))
            if not has_model and not has_source:
                candidates.append((idle, sid))
        candidates.sort(reverse=True)
        for _, sid in candidates:
            if len(self._sessions) < self.max_sessions:
                break
            self._close_unlocked(sid)

    def _close_unlocked(self, session_id: str) -> None:
        sess = self._sessions.pop(session_id, None)
        if sess is not None:
            sess.close()

    def get_or_create(self, session_id: Optional[str] = None) -> UserSession:
        """Resume existing session or create one. Same id always resumes after refresh."""
        with self._lock:
            self._purge_expired()

            if session_id:
                sid = session_id.strip()
                # Allow hex UUIDs and similar stable tokens
                if sid and all(c.isalnum() or c in "-_" for c in sid) and len(sid) <= 64:
                    existing = self._sessions.get(sid)
                    if existing is not None:
                        existing.touch()
                        return existing
                    # Recreate under the same browser id (refresh after server restart / purge)
                    if len(self._sessions) >= self.max_sessions:
                        self._purge_idle_empty()
                    if len(self._sessions) >= self.max_sessions:
                        raise RuntimeError(
                            f"Server is full ({self.max_sessions} concurrent users). "
                            "Ask someone to close their tab, or try again shortly."
                        )
                    sess = UserSession(session_id=sid, processor=StreamProcessor())
                    self._sessions[sid] = sess
                    return sess

            if len(self._sessions) >= self.max_sessions:
                self._purge_idle_empty()
            if len(self._sessions) >= self.max_sessions:
                raise RuntimeError(
                    f"Server is full ({self.max_sessions} concurrent users). "
                    "Ask someone to close their tab, or try again shortly."
                )

            sid = uuid.uuid4().hex
            sess = UserSession(session_id=sid, processor=StreamProcessor())
            self._sessions[sid] = sess
            return sess

    def get(self, session_id: Optional[str]) -> UserSession:
        with self._lock:
            self._purge_expired()
            if not session_id or session_id not in self._sessions:
                raise RuntimeError("Unknown or expired session. Refresh the page.")
            sess = self._sessions[session_id]
            sess.touch()
            return sess

    def release(self, session_id: str) -> None:
        with self._lock:
            self._close_unlocked(session_id)

    def clear_all(self) -> None:
        with self._lock:
            for sid in list(self._sessions.keys()):
                self._close_unlocked(sid)

    def info(self) -> dict:
        with self._lock:
            self._purge_expired()
            return {
                "active_sessions": len(self._sessions),
                "max_sessions": self.max_sessions,
                "ttl_sec": self.ttl_sec,
                "sessions": [
                    {
                        "session_id": s.session_id,
                        "created_at": s.created_at,
                        "last_seen": s.last_seen,
                        "idle_sec": round(time.time() - s.last_seen, 1),
                        "has_model": s.processor.predictor is not None,
                        "source_type": s.processor.status().get("source_type"),
                        "live_running": s.live_grid.is_running,
                    }
                    for s in self._sessions.values()
                ],
            }


session_manager = SessionManager()
