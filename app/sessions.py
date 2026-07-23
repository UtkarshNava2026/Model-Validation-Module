"""
Multi-user session manager.

Each browser/dev gets an isolated StreamProcessor so concurrent users
don't overwrite each other's model / video / play state.

Limits (override with env):
  MAX_SESSIONS=3          # safe default for ~8GB GPU; raise for 16GB
  SESSION_TTL_SEC=3600    # idle sessions are cleaned up
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from app.stream.processor import StreamProcessor

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class UserSession:
    session_id: str
    processor: StreamProcessor = field(default_factory=StreamProcessor)
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    label: str = ""

    def touch(self):
        self.last_seen = time.time()

    def close(self):
        try:
            self.processor.stop()
        except Exception:
            pass
        # Drop model reference so GC / CUDA can reclaim
        try:
            self.processor.set_predictor(None)  # type: ignore[arg-type]
        except Exception:
            self.processor._predictor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        # Remove ephemeral media / weights for this session (not permanent uploads/)
        try:
            from app.storage import clear_session_scratch

            clear_session_scratch(self.session_id)
        except Exception:
            pass


class SessionManager:
    def __init__(
        self,
        max_sessions: int | None = None,
        ttl_sec: int | None = None,
    ):
        self.max_sessions = max_sessions or _env_int("MAX_SESSIONS", 3)
        self.ttl_sec = ttl_sec or _env_int("SESSION_TTL_SEC", 3600)
        self._lock = threading.RLock()
        self._sessions: dict[str, UserSession] = {}

    def info(self) -> dict:
        with self._lock:
            self._purge_expired()
            return {
                "active_sessions": len(self._sessions),
                "max_sessions": self.max_sessions,
                "ttl_sec": self.ttl_sec,
                "sessions": [
                    {
                        "id": s.session_id[:8],
                        "idle_sec": int(time.time() - s.last_seen),
                        "model_loaded": s.processor.predictor is not None,
                        "source_type": (
                            s.processor.status().get("source_type")
                        ),
                    }
                    for s in self._sessions.values()
                ],
            }

    def get_or_create(self, session_id: Optional[str]) -> UserSession:
        with self._lock:
            self._purge_expired()
            if session_id and session_id in self._sessions:
                sess = self._sessions[session_id]
                sess.touch()
                return sess

            if len(self._sessions) >= self.max_sessions:
                # Reclaim sessions that never loaded a model and sat idle briefly
                self._purge_idle_empty(min_idle_sec=30)
                self._purge_expired()
                if len(self._sessions) >= self.max_sessions:
                    raise RuntimeError(
                        f"Server busy: {self.max_sessions} concurrent sessions in use. "
                        f"Wait for a free slot, or raise MAX_SESSIONS "
                        f"(current GPU guidance: 3 on 8GB, 4–6 on 16GB)."
                    )

            # Only reuse client-provided id if it looks valid; else mint new
            if session_id and len(session_id) >= 16 and session_id.isalnum():
                new_id = session_id
            else:
                new_id = uuid.uuid4().hex
            while new_id in self._sessions:
                new_id = uuid.uuid4().hex
            sess = UserSession(session_id=new_id)
            self._sessions[new_id] = sess
            logger.info(
                "Session created %s (%d/%d)",
                new_id[:8],
                len(self._sessions),
                self.max_sessions,
            )
            return sess

    def _purge_idle_empty(self, min_idle_sec: float = 30):
        """Drop unused sessions (no model) to free slots for real users."""
        now = time.time()
        dead = [
            sid
            for sid, s in self._sessions.items()
            if s.processor.predictor is None
            and (now - s.last_seen) >= min_idle_sec
        ]
        for sid in dead:
            sess = self._sessions.pop(sid, None)
            if sess:
                sess.close()
                logger.info("Reclaimed empty session %s", sid[:8])

    def clear_all(self):
        with self._lock:
            for sess in list(self._sessions.values()):
                sess.close()
            self._sessions.clear()

    def get(self, session_id: Optional[str]) -> UserSession:
        with self._lock:
            self._purge_expired()
            if not session_id or session_id not in self._sessions:
                raise RuntimeError(
                    "Unknown or expired session. Refresh the page to get a new session."
                )
            sess = self._sessions[session_id]
            sess.touch()
            return sess

    def release(self, session_id: str) -> bool:
        with self._lock:
            sess = self._sessions.pop(session_id, None)
            if not sess:
                return False
            sess.close()
            logger.info("Session released %s", session_id[:8])
            return True

    def _purge_expired(self):
        now = time.time()
        dead = [
            sid
            for sid, s in self._sessions.items()
            if (now - s.last_seen) > self.ttl_sec
        ]
        for sid in dead:
            sess = self._sessions.pop(sid, None)
            if sess:
                sess.close()
                logger.info("Session expired %s", sid[:8])


session_manager = SessionManager()
