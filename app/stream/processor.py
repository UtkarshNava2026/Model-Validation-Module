"""Video / RTSP / image / directory stream processor with play-pause, seek, and speed."""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class SourceType(str, Enum):
    IMAGE = "image"
    IMAGE_DIR = "image_dir"
    VIDEO = "video"
    RTSP = "rtsp"


class StreamProcessor:
    def __init__(self):
        self._lock = threading.RLock()
        self._predictor = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._source_type: Optional[SourceType] = None
        self._source_path: Optional[str] = None
        self._playing = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._last_raw: Optional[np.ndarray] = None
        self._source_image: Optional[np.ndarray] = None
        self._fps = 0.0
        self._source_fps = 25.0
        self._det_count = 0
        self._frame_idx = 0
        self._total_frames = 0
        self._duration_sec = 0.0
        self._loop_video = True
        self._speed = 1.0
        self._status = "idle"
        self._error: Optional[str] = None
        self._image_paths: list[str] = []
        self._image_index = 0
        self._current_image_name: Optional[str] = None

    # ---- state helpers ----
    def status(self) -> dict:
        with self._lock:
            current_sec = (
                self._frame_idx / self._source_fps if self._source_fps > 0 else 0.0
            )
            image_count = len(self._image_paths)
            return {
                "status": self._status,
                "playing": self._playing,
                "source_type": self._source_type.value if self._source_type else None,
                "source": self._source_path,
                "fps": round(self._fps, 1),
                "source_fps": round(self._source_fps, 2),
                "detections": self._det_count,
                "frame": self._frame_idx,
                "total_frames": self._total_frames,
                "current_sec": round(current_sec, 2),
                "duration_sec": round(self._duration_sec, 2),
                "speed": self._speed,
                "seekable": self._source_type == SourceType.VIDEO
                and self._total_frames > 0,
                "image_gallery": self._source_type == SourceType.IMAGE_DIR
                and image_count > 0,
                "image_index": self._image_index,
                "image_count": image_count,
                "image_name": self._current_image_name,
                "has_prev": image_count > 0 and self._image_index > 0,
                "has_next": image_count > 0 and self._image_index < image_count - 1,
                "model_loaded": self._predictor is not None,
                "error": self._error,
                "conf": getattr(self._predictor, "conf", None),
                "nms": getattr(self._predictor, "nms", None),
            }

    def set_predictor(self, predictor):
        with self._lock:
            self._predictor = predictor
            self._status = "model_ready"
            self._error = None

    def set_conf(self, conf: float):
        with self._lock:
            if self._predictor is None:
                raise RuntimeError("Model not loaded")
            self._predictor.set_conf(conf)
            self._reprocess_current()

    def set_nms(self, nms: float):
        with self._lock:
            if self._predictor is None:
                raise RuntimeError("Model not loaded")
            self._predictor.set_nms(nms)
            self._reprocess_current()

    def set_speed(self, speed: float):
        """Playback speed multiplier (0.25x – 8x)."""
        with self._lock:
            self._speed = float(np.clip(speed, 0.25, 8.0))

    def _reprocess_current(self):
        """Re-run inference on the last raw frame (image / paused video)."""
        if self._predictor is None:
            return
        raw = self._source_image if self._source_image is not None else self._last_raw
        if raw is None:
            return
        annotated, n = self._predictor.predict_frame(raw)
        self._det_count = n
        self._encode(annotated)

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    @staticmethod
    def list_images_in_dir(directory: str | Path) -> list[str]:
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Not a directory: {root}")
        files = [
            str(p)
            for p in sorted(root.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not files:
            raise ValueError(f"No images found in {root}")
        return files

    def open_image_list(self, paths: list[str], label: str | None = None):
        """Open a list of image paths as a browsable gallery."""
        self.stop()
        with self._lock:
            if self._predictor is None:
                raise RuntimeError("Load model weights / classes / experiment first")
            cleaned = []
            for p in paths:
                path = Path(p).expanduser().resolve()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    cleaned.append(str(path))
            if not cleaned:
                raise ValueError("No valid images provided")
            cleaned = sorted(cleaned)

            self._error = None
            self._source_type = SourceType.IMAGE_DIR
            self._source_path = label or str(Path(cleaned[0]).parent)
            self._image_paths = cleaned
            self._image_index = 0
            self._cap = None
            self._frame_idx = 0
            self._total_frames = 0
            self._duration_sec = 0.0
            self._playing = False
            self._show_image_at(0)

    def open_image_directory(self, directory: str):
        paths = self.list_images_in_dir(directory)
        self.open_image_list(paths, label=str(Path(directory).expanduser().resolve()))

    def next_image(self):
        with self._lock:
            if self._source_type != SourceType.IMAGE_DIR or not self._image_paths:
                raise RuntimeError("No image directory loaded")
            if self._image_index >= len(self._image_paths) - 1:
                raise RuntimeError("Already at the last image")
            self._show_image_at(self._image_index + 1)

    def prev_image(self):
        with self._lock:
            if self._source_type != SourceType.IMAGE_DIR or not self._image_paths:
                raise RuntimeError("No image directory loaded")
            if self._image_index <= 0:
                raise RuntimeError("Already at the first image")
            self._show_image_at(self._image_index - 1)

    def goto_image(self, index: int):
        with self._lock:
            if self._source_type != SourceType.IMAGE_DIR or not self._image_paths:
                raise RuntimeError("No image directory loaded")
            index = int(np.clip(index, 0, len(self._image_paths) - 1))
            self._show_image_at(index)

    def _show_image_at(self, index: int):
        assert self._predictor is not None
        path = self._image_paths[index]
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        self._image_index = index
        self._current_image_name = Path(path).name
        self._source_image = img
        self._last_raw = img
        t0 = time.time()
        annotated, n = self._predictor.predict_frame(img)
        elapsed = time.time() - t0
        self._fps = 1.0 / elapsed if elapsed > 0 else 0.0
        self._det_count = n
        self._encode(annotated)
        self._status = "ready"
        self._playing = False

    # ---- source control ----
    def open_source(self, source: str, source_type: str):
        st = SourceType(source_type)
        if st == SourceType.IMAGE_DIR:
            self.open_image_directory(source)
            return

        self.stop()
        with self._lock:
            if self._predictor is None:
                raise RuntimeError("Load model weights / classes / experiment first")

            self._error = None
            self._source_type = st
            self._source_path = source
            self._frame_idx = 0
            self._total_frames = 0
            self._duration_sec = 0.0
            self._source_fps = 25.0
            self._image_paths = []
            self._image_index = 0
            self._current_image_name = None

            if st == SourceType.IMAGE:
                img = cv2.imread(source)
                if img is None:
                    raise ValueError(f"Could not read image: {source}")
                self._source_image = img
                self._last_raw = img
                self._current_image_name = Path(source).name
                annotated, n = self._predictor.predict_frame(img)
                self._det_count = n
                self._encode(annotated)
                self._status = "ready"
                self._playing = False
                return

            self._source_image = None

            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise ValueError(f"Could not open source: {source}")

            if st == SourceType.RTSP:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self._cap = cap
            src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            self._source_fps = src_fps if src_fps > 1e-3 else 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self._total_frames = max(total, 0)
            if self._total_frames > 0 and self._source_fps > 0:
                self._duration_sec = self._total_frames / self._source_fps
            self._status = "ready"
            self._playing = False

            ok, frame = cap.read()
            if ok and frame is not None:
                self._process_frame(frame, set_index=0)
                if st == SourceType.VIDEO:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._frame_idx = 0

    def play(self):
        with self._lock:
            if self._source_type in (SourceType.IMAGE, SourceType.IMAGE_DIR):
                self._status = "ready"
                return
            if self._cap is None:
                raise RuntimeError("No media source opened")
            if self._playing:
                return
            self._playing = True
            self._status = "playing"
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def pause(self):
        with self._lock:
            self._playing = False
            if self._status == "playing":
                self._status = "paused"
        self._stop_event.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=2.0)
        self._thread = None

    def stop(self):
        self.pause()
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
            if self._status not in ("model_ready", "idle"):
                self._status = "stopped"

    def seek(self, frame_idx: int | None = None, time_sec: float | None = None):
        """Seek video by frame index or seconds. Keeps play/pause state."""
        with self._lock:
            if self._cap is None or self._source_type != SourceType.VIDEO:
                raise RuntimeError("Seek is only available for video files")
            if self._predictor is None:
                raise RuntimeError("Model not loaded")

            if time_sec is not None:
                frame_idx = int(max(0.0, float(time_sec)) * self._source_fps)
            if frame_idx is None:
                raise ValueError("Provide frame or time_sec")

            if self._total_frames > 0:
                frame_idx = int(np.clip(frame_idx, 0, max(0, self._total_frames - 1)))
            else:
                frame_idx = max(0, int(frame_idx))

            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = self._cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not seek to frame {frame_idx}")
            self._process_frame(frame, set_index=frame_idx)
            if not self._playing and self._status not in ("ended", "error"):
                self._status = "paused"

    # ---- internals ----
    def _encode(self, frame: np.ndarray):
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            self._latest_jpeg = buf.tobytes()
            self._latest_frame = frame

    def _process_frame(self, frame: np.ndarray, set_index: int | None = None):
        assert self._predictor is not None
        self._last_raw = frame
        t0 = time.time()
        annotated, n = self._predictor.predict_frame(frame)
        elapsed = time.time() - t0
        self._fps = 1.0 / elapsed if elapsed > 0 else 0.0
        self._det_count = n
        if set_index is not None:
            self._frame_idx = int(set_index)
        else:
            # Prefer OpenCV position after read when available
            if self._cap is not None and self._source_type == SourceType.VIDEO:
                pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                self._frame_idx = max(0, pos - 1)
            else:
                self._frame_idx += 1
        self._encode(annotated)

    def _loop(self):
        while not self._stop_event.is_set():
            with self._lock:
                if not self._playing or self._cap is None or self._predictor is None:
                    break
                cap = self._cap
                source_type = self._source_type
                speed = self._speed
                source_fps = self._source_fps

            ok, frame = cap.read()
            if not ok or frame is None:
                if source_type == SourceType.VIDEO and self._loop_video:
                    with self._lock:
                        if self._cap is not None:
                            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            self._frame_idx = 0
                    continue
                with self._lock:
                    self._playing = False
                    self._status = "ended"
                break

            try:
                with self._lock:
                    if self._predictor is None:
                        break
                    self._process_frame(frame)
            except Exception as exc:
                logger.exception("Inference error")
                with self._lock:
                    self._error = str(exc)
                    self._playing = False
                    self._status = "error"
                break

            # Pace video by source FPS / speed; RTSP runs as fast as inference allows
            if source_type == SourceType.VIDEO:
                target = source_fps * max(speed, 0.25)
                delay = (1.0 / target) if target > 0 else 0.04
                # Subtract a bit of inference time already spent
                delay = max(0.0, delay - 0.001)
                # At high speed, skip wait entirely when inference is the bottleneck
                if speed >= 4.0:
                    delay = 0.0
                if delay > 0:
                    time.sleep(delay)
