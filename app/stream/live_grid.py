"""Up to four RTSP cameras, one shared predictor, one inference worker.

Capture threads keep only the newest frame. The worker round-robins those
frames so a single YOLOX model serves every tile without concurrent forwards.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

from app.stream.capture import open_capture

logger = logging.getLogger(__name__)

MAX_SLOTS = 4
_RECONNECT_SEC = 2.0
_JPEG_QUALITY = 75
_LIVE_SCHEMES = {"rtsp", "rtsps", "http", "https"}
_URL_IN_TEXT = re.compile(r"(?i)\b(?:rtsps?|https?)://\S+")


def redact_text(text: str) -> str:
    """Remove stream URLs so a camera password cannot leak into the UI or logs."""
    cleaned = _URL_IN_TEXT.sub("rtsp://****", text or "")
    return cleaned[:300]


def mask_stream_url(url: str) -> str:
    """Hide userinfo passwords before a URL is shown or logged."""
    raw = (url or "").strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
    except Exception:
        return "rtsp://****" if "@" in raw else raw
    if not parts.hostname:
        return "rtsp://****" if "@" in raw else raw
    host = parts.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{parts.port}" if parts.port else host
    if parts.username:
        if parts.password:
            netloc = f"{parts.username}:****@{netloc}"
        else:
            netloc = f"{parts.username}@{netloc}"
    elif parts.password or "@" in (parts.netloc or ""):
        netloc = f"****@{netloc}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, ""))


def slot_label(url: str, index: int) -> str:
    """Short camera name from host and path, never credentials."""
    try:
        parts = urlsplit((url or "").strip())
    except Exception:
        parts = None
    host = ""
    tail = ""
    if parts is not None:
        host = parts.hostname or ""
        if parts.path:
            tail = parts.path.rstrip("/").split("/")[-1]
    if host and tail:
        return f"Cam {index + 1} · {host}/{tail}"
    if host:
        return f"Cam {index + 1} · {host}"
    return f"Cam {index + 1}"


def validate_live_url(url: str) -> str:
    """Accept only network streams. Local file paths stay on the single-source path API."""
    cleaned = (url or "").strip()
    if not cleaned or len(cleaned) > 2048:
        raise ValueError("RTSP URL is empty or too long")
    parts = urlsplit(cleaned)
    if parts.scheme.lower() not in _LIVE_SCHEMES or not parts.hostname:
        raise ValueError("Use an rtsp:// URL with a host")
    return cleaned


@dataclass
class _Slot:
    index: int
    url: str
    label: str
    masked_url: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    frame: Optional[np.ndarray] = None
    seq: int = 0
    seen: int = 0
    jpeg: Optional[bytes] = None
    connected: bool = False
    error: Optional[str] = None
    fps: float = 0.0
    detections: int = 0
    last_done: float = 0.0
    has_frame: bool = False
    cap: Optional[cv2.VideoCapture] = None


class LiveGrid:
    def __init__(self) -> None:
        self._ops_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._generation = 0
        self._predictor = None
        self._slots: list[_Slot] = []
        self._threads: list[threading.Thread] = []
        self._infer_thread: Optional[threading.Thread] = None
        self._running = False
        self._status = "idle"

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self, predictor, urls: list[str]) -> None:
        cleaned = [u.strip() for u in urls if u and u.strip()]
        if predictor is None:
            raise RuntimeError("Load model weights / classes / experiment first")
        if not cleaned:
            raise ValueError("Provide at least one RTSP URL")
        if len(cleaned) > MAX_SLOTS:
            raise ValueError(f"At most {MAX_SLOTS} RTSP URLs")
        cleaned = [validate_live_url(url) for url in cleaned]

        with self._ops_lock:
            self._start_locked(predictor, cleaned)

    def _start_locked(self, predictor, cleaned: list[str]) -> None:
        self.stop()
        with self._state_lock:
            self._generation += 1
            generation = self._generation
            self._stop_event.clear()
            self._predictor = predictor
            self._slots = [
                _Slot(
                    index=i,
                    url=url,
                    label=slot_label(url, i),
                    masked_url=mask_stream_url(url),
                )
                for i, url in enumerate(cleaned)
            ]
            self._running = True
            self._status = "running"
            self._threads = []
            for slot in self._slots:
                thread = threading.Thread(
                    target=self._capture_loop,
                    args=(slot, generation),
                    name=f"live-cap-{slot.index}",
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)
            self._infer_thread = threading.Thread(
                target=self._infer_loop,
                args=(generation,),
                name="live-infer",
                daemon=True,
            )
            self._infer_thread.start()
            logger.info(
                "Live grid started (%d cameras): %s",
                len(self._slots),
                ", ".join(slot.label for slot in self._slots),
            )

    def stop(self) -> None:
        with self._ops_lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        with self._state_lock:
            threads = list(self._threads)
            infer = self._infer_thread
            slots = list(self._slots)
            self._running = False
            self._generation += 1
        self._stop_event.set()

        self._join_all(threads, timeout=2.5)
        if infer is not None:
            infer.join(timeout=2.0)

        stuck = [thread for thread in threads if thread.is_alive()]
        if infer is not None and infer.is_alive():
            stuck.append(infer)
        if stuck:
            for slot in slots:
                self._release_capture(slot)
            self._join_all(stuck, timeout=1.5)
            still = [thread.name for thread in stuck if thread.is_alive()]
            if still:
                logger.warning("Live grid threads still exiting: %s", ", ".join(still))

        with self._state_lock:
            self._threads = []
            self._infer_thread = None
            self._slots = []
            self._predictor = None
            self._status = "idle"
            self._stop_event = threading.Event()

    def get_jpeg(self, index: int) -> Optional[bytes]:
        with self._state_lock:
            if index < 0 or index >= len(self._slots):
                return None
            slot = self._slots[index]
        with slot.lock:
            return slot.jpeg

    def status(self) -> dict:
        with self._state_lock:
            running = self._running
            status = self._status
            slots = list(self._slots)
        payload = []
        for slot in slots:
            with slot.lock:
                payload.append(
                    {
                        "index": slot.index,
                        "label": slot.label,
                        "url": slot.masked_url,
                        "connected": slot.connected,
                        "error": slot.error,
                        "fps": round(slot.fps, 1),
                        "detections": slot.detections,
                        "has_frame": slot.has_frame,
                    }
                )
        return {
            "running": running,
            "status": status,
            "slot_count": len(payload),
            "slots": payload,
        }

    def _capture_loop(self, slot: _Slot, generation: int) -> None:
        while self._alive(generation):
            cap = open_capture(slot.url)
            if cap is None:
                with slot.lock:
                    slot.connected = False
                    slot.error = "Could not connect"
                logger.info("Live slot %s could not connect (%s)", slot.index, slot.label)
                if self._stop_event.wait(_RECONNECT_SEC):
                    break
                continue

            with slot.lock:
                slot.cap = cap
                slot.connected = True
                slot.error = None
            failures = 0
            while self._alive(generation):
                try:
                    ok, frame = cap.read()
                except Exception:
                    logger.exception("Live slot %s read failed", slot.index)
                    ok, frame = False, None
                if not ok or frame is None:
                    failures += 1
                    if failures >= 15:
                        break
                    if self._stop_event.wait(0.05):
                        break
                    continue
                failures = 0
                # Copy before the next read(). OpenCV may reuse the same buffer.
                owned = frame.copy()
                with slot.lock:
                    slot.frame = owned
                    slot.seq += 1
                    slot.connected = True
                    slot.error = None

            self._release_capture(slot)
            with slot.lock:
                slot.connected = False
                if self._alive(generation) and not slot.error:
                    slot.error = "Reconnecting…"
            if not self._alive(generation):
                break
            if self._stop_event.wait(_RECONNECT_SEC):
                break

    def _infer_loop(self, generation: int) -> None:
        cursor = 0
        while self._alive(generation):
            predictor = self._predictor
            with self._state_lock:
                slots = list(self._slots)
            if predictor is None or not slots:
                break
            progressed = False
            for _ in range(len(slots)):
                if not self._alive(generation):
                    break
                slot = slots[cursor % len(slots)]
                cursor += 1
                with slot.lock:
                    if slot.frame is None or slot.seq == slot.seen:
                        frame = None
                    else:
                        frame = slot.frame.copy()
                        slot.seen = slot.seq
                if frame is None:
                    continue
                progressed = True
                try:
                    annotated, count = predictor.predict_frame(frame)
                except Exception as exc:
                    logger.exception("Live inference failed on %s", slot.label)
                    with slot.lock:
                        slot.error = redact_text(str(exc)) or "Inference failed"
                    continue
                ok, buf = cv2.imencode(
                    ".jpg",
                    annotated,
                    [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY],
                )
                if not ok:
                    continue
                now = time.perf_counter()
                with slot.lock:
                    slot.jpeg = buf.tobytes()
                    slot.detections = count
                    slot.has_frame = True
                    if slot.last_done > 0:
                        delta = now - slot.last_done
                        if delta > 0:
                            instant = 1.0 / delta
                            slot.fps = (
                                instant
                                if slot.fps <= 0
                                else (0.7 * slot.fps + 0.3 * instant)
                            )
                    slot.last_done = now
                    if slot.connected:
                        slot.error = None
            if not progressed:
                self._stop_event.wait(0.01)

    def _alive(self, generation: int) -> bool:
        return generation == self._generation and not self._stop_event.is_set()

    @staticmethod
    def _join_all(threads: list[threading.Thread], timeout: float) -> None:
        """Wait for every thread, but share one deadline instead of timing each out fully."""
        deadline = time.monotonic() + timeout
        for thread in threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)

    @staticmethod
    def _release_capture(slot: _Slot) -> None:
        with slot.lock:
            cap = slot.cap
            slot.cap = None
        if cap is None:
            return
        try:
            cap.release()
        except Exception:
            pass
