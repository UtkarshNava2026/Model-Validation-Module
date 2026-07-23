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
        # Inference video recording
        self._recording = False
        self._recorder: Optional[cv2.VideoWriter] = None
        self._record_path: Optional[Path] = None
        self._record_ready_path: Optional[Path] = None
        self._record_start_ts = 0.0
        self._record_limit_sec: Optional[float] = None
        self._record_frames = 0
        self._record_dir = Path(__file__).resolve().parents[2] / "outputs" / "recordings"
        self._record_dir.mkdir(parents=True, exist_ok=True)

    # ---- state helpers ----
    def status(self) -> dict:
        with self._lock:
            current_sec = (
                self._frame_idx / self._source_fps if self._source_fps > 0 else 0.0
            )
            image_count = len(self._image_paths)
            recorded_sec = 0.0
            if self._recording and self._record_start_ts > 0:
                recorded_sec = max(0.0, time.time() - self._record_start_ts)
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
                "recording": self._recording,
                "record_limit_sec": self._record_limit_sec,
                "record_elapsed_sec": round(recorded_sec, 1),
                "record_frames": self._record_frames,
                "download_ready": self._record_ready_path is not None
                and self._record_ready_path.exists(),
                "download_name": (
                    self._record_ready_path.name if self._record_ready_path else None
                ),
            }

    def set_predictor(self, predictor):
        with self._lock:
            self._predictor = predictor
            if predictor is None:
                self._status = "idle"
                self._error = None
            else:
                self._status = "model_ready"
                self._error = None

    @property
    def predictor(self):
        return self._predictor

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

    def start_recording(self, duration_sec: float | None = None):
        """
        Start capturing annotated frames to an MP4.
        duration_sec: if > 0, auto-stop after that many seconds; else until stop_recording().
        """
        with self._lock:
            if self._predictor is None:
                raise RuntimeError("Load model first")
            if self._source_type in (None, SourceType.IMAGE, SourceType.IMAGE_DIR):
                raise RuntimeError("Recording is for video / RTSP playback")
            if self._recording:
                return
            # Close any previous open writer
            self._close_recorder(finalize=False)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self._record_path = self._record_dir / f"inference_{stamp}.mp4"
            self._record_ready_path = None
            self._record_frames = 0
            self._record_start_ts = time.time()
            limit = float(duration_sec) if duration_sec is not None else 0.0
            self._record_limit_sec = limit if limit > 0 else None
            self._recording = True
            # Writer opens lazily on first annotated frame (needs size)
            logger.info(
                "Recording started → %s (limit=%s s)",
                self._record_path.name,
                self._record_limit_sec,
            )

    def stop_recording(self) -> Optional[Path]:
        """Stop recording and finalize file for download."""
        with self._lock:
            if not self._recording and self._recorder is None:
                return self._record_ready_path
            self._recording = False
            path = self._close_recorder(finalize=True)
            return path

    def get_download_path(self) -> Optional[Path]:
        with self._lock:
            if self._record_ready_path and self._record_ready_path.exists():
                return self._record_ready_path
            return None

    def _close_recorder(self, finalize: bool) -> Optional[Path]:
        path = self._record_path
        if self._recorder is not None:
            try:
                self._recorder.release()
            except Exception:
                pass
            self._recorder = None
        self._recording = False
        if finalize and path is not None and path.exists() and path.stat().st_size > 0:
            self._record_ready_path = path
            logger.info(
                "Recording saved: %s (%d frames)", path.name, self._record_frames
            )
            return path
        self._record_path = None
        return self._record_ready_path

    def _ensure_recorder(self, frame: np.ndarray):
        if not self._recording or self._recorder is not None:
            return
        assert self._record_path is not None
        h, w = frame.shape[:2]
        fps = self._source_fps if self._source_fps and self._source_fps > 1 else 20.0
        # Cap writer fps to a sane range
        fps = float(np.clip(fps, 5.0, 60.0))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self._record_path), fourcc, fps, (w, h))
        if not writer.isOpened():
            self._recording = False
            self._error = "Could not open video writer for recording"
            raise RuntimeError(self._error)
        self._recorder = writer

    def _write_record_frame(self, frame: np.ndarray):
        if not self._recording:
            return
        try:
            self._ensure_recorder(frame)
            if self._recorder is not None:
                self._recorder.write(frame)
                self._record_frames += 1
        except Exception as exc:
            logger.exception("Record write failed")
            self._error = str(exc)
            self._recording = False
            self._close_recorder(finalize=False)
            return
        # Auto-stop after duration (record this frame, then finalize)
        if self._record_limit_sec is not None and self._record_start_ts > 0:
            if (time.time() - self._record_start_ts) >= self._record_limit_sec:
                self._recording = False
                self._close_recorder(finalize=True)

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
            if self._recording or self._recorder is not None:
                self._recording = False
                self._close_recorder(finalize=True)
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
        self._write_record_frame(frame)

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
                # At >1x, skip extra source frames so playback feels faster
                if speed > 1.0:
                    skip = max(0, int(speed) - 1)
                    for _ in range(skip):
                        if self._stop_event.is_set():
                            break
                        cap.grab()
                target = source_fps * max(speed, 0.25)
                delay = (1.0 / target) if target > 0 else 0.04
                delay = max(0.0, delay - 0.001)
                if speed >= 2.0:
                    delay = 0.0
                if delay > 0:
                    time.sleep(delay)
