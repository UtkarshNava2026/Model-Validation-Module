"""OpenCV capture helper.

Timeouts are applied before open(), which is the only point OpenCV honors
them. That keeps a dead RTSP camera from blocking the server indefinitely.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2

# Quiet FFmpeg so camera URLs with passwords are not written to the process log.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")


def open_capture(
    source: str,
    *,
    open_timeout_ms: int = 5000,
    read_timeout_ms: int = 3000,
) -> Optional[cv2.VideoCapture]:
    cap = _open_with_api(source, cv2.CAP_FFMPEG, open_timeout_ms, read_timeout_ms)
    if cap is not None:
        return cap
    return _open_with_api(source, cv2.CAP_ANY, open_timeout_ms, read_timeout_ms)


def _open_with_api(
    source: str,
    api: int,
    open_timeout_ms: int,
    read_timeout_ms: int,
) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture()
    _set_timeouts(cap, open_timeout_ms, read_timeout_ms)
    try:
        opened = cap.open(source, api)
    except Exception:
        opened = False
    if not opened or not cap.isOpened():
        cap.release()
        return None
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def _set_timeouts(cap: cv2.VideoCapture, open_timeout_ms: int, read_timeout_ms: int) -> None:
    for prop_name, msec in (
        ("CAP_PROP_OPEN_TIMEOUT_MSEC", open_timeout_ms),
        ("CAP_PROP_READ_TIMEOUT_MSEC", read_timeout_ms),
    ):
        prop = getattr(cv2, prop_name, None)
        if prop is None:
            continue
        try:
            cap.set(prop, int(msec))
        except Exception:
            pass
