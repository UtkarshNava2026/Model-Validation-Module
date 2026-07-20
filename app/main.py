"""
Nava Vision AI Model Validation — FastAPI backend.

Upload YOLOX weights + class file + experiment file, then run inference on
image / video / RTSP (live or playback) with official YOLOX pre/post-process.
"""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.stream.processor import StreamProcessor
from app.yolox.predictor import YOLOXPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model-validation")

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "uploads"
STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Nava Vision AI Model Validation", version="1.0.0")
processor = StreamProcessor()

# Persist last uploaded artifact paths for the session
_session = {
    "weight": None,
    "classes": None,
    "exp": None,
}


class ConfBody(BaseModel):
    conf: float = Field(..., ge=0.01, le=0.99)


class NmsBody(BaseModel):
    nms: float = Field(..., ge=0.01, le=0.99)


class SpeedBody(BaseModel):
    speed: float = Field(..., ge=0.25, le=8.0)


class SeekBody(BaseModel):
    frame: Optional[int] = Field(None, ge=0)
    time_sec: Optional[float] = Field(None, ge=0.0)


class RtspBody(BaseModel):
    url: str
    source_type: str = "rtsp"  # rtsp | video (for file urls)


class DirBody(BaseModel):
    path: str


class GotoImageBody(BaseModel):
    index: int = Field(..., ge=0)


async def _save_upload(file: UploadFile, prefix: str) -> Path:
    suffix = Path(file.filename or "file").suffix
    dest = UPLOAD_DIR / f"{prefix}_{uuid.uuid4().hex}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest


async def _save_gallery_uploads(files: list[UploadFile]) -> list[Path]:
    gallery_dir = UPLOAD_DIR / f"gallery_{uuid.uuid4().hex}"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for file in files:
        name = Path(file.filename or "image.jpg").name
        # Flatten nested folder uploads: keep basename only
        dest = gallery_dir / name
        # Avoid overwrite collisions
        if dest.exists():
            dest = gallery_dir / f"{dest.stem}_{uuid.uuid4().hex[:8]}{dest.suffix}"
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        saved.append(dest)
    return saved


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/upload")
async def upload_model(
    weight: UploadFile = File(..., description="YOLOX .pth or .onnx weights"),
    classes: UploadFile = File(..., description="Class names .txt"),
    exp: UploadFile = File(..., description="YOLOX experiment .py"),
    device: str = Form("cpu"),
    conf: Optional[float] = Form(None),
    nms: Optional[float] = Form(None),
):
    try:
        weight_path = await _save_upload(weight, "weight")
        class_path = await _save_upload(classes, "classes")
        exp_path = await _save_upload(exp, "exp")

        predictor = YOLOXPredictor(
            weight_path=weight_path,
            exp_path=exp_path,
            class_path=class_path,
            device=device,
            conf=conf,
            nms=nms,
        )
        processor.set_predictor(predictor)
        _session["weight"] = str(weight_path)
        _session["classes"] = str(class_path)
        _session["exp"] = str(exp_path)

        return {
            "ok": True,
            "message": "Model loaded successfully",
            "backend": predictor.backend,
            "num_classes": predictor.num_classes,
            "class_names": predictor.class_names,
            "test_size": predictor.test_size,
            "conf": predictor.conf,
            "nms": predictor.nms,
            "device": predictor.device,
        }
    except Exception as exc:
        logger.exception("Upload/load failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/source/file")
async def open_file_source(
    file: UploadFile = File(...),
    source_type: str = Form(...),  # image | video
):
    if source_type not in ("image", "video"):
        raise HTTPException(status_code=400, detail="source_type must be image or video")
    try:
        path = await _save_upload(file, source_type)
        processor.open_source(str(path), source_type)
        return {"ok": True, **processor.status()}
    except Exception as exc:
        logger.exception("Failed to open file source")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/source/folder")
async def open_folder_upload(files: list[UploadFile] = File(...)):
    """Upload multiple images (folder picker) and browse with next/prev."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    try:
        saved = await _save_gallery_uploads(files)
        image_paths = [
            str(p)
            for p in saved
            if p.suffix.lower()
            in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        ]
        if not image_paths:
            raise ValueError("No image files in the selected folder")
        processor.open_image_list(image_paths, label=str(Path(image_paths[0]).parent))
        return {"ok": True, **processor.status()}
    except Exception as exc:
        logger.exception("Failed to open folder upload")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/source/directory")
async def open_directory_path(body: DirBody):
    """Open a server-side image directory path."""
    path = body.path.strip()
    if not path:
        raise HTTPException(status_code=400, detail="Directory path is empty")
    try:
        processor.open_image_directory(path)
        return {"ok": True, **processor.status()}
    except Exception as exc:
        logger.exception("Failed to open directory path")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/control/next-image")
async def next_image():
    try:
        processor.next_image()
        return {"ok": True, **processor.status()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/control/prev-image")
async def prev_image():
    try:
        processor.prev_image()
        return {"ok": True, **processor.status()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/control/goto-image")
async def goto_image(body: GotoImageBody):
    try:
        processor.goto_image(body.index)
        return {"ok": True, **processor.status()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/source/rtsp")
async def open_rtsp(body: RtspBody):
    url = body.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="RTSP/URL is empty")
    st = body.source_type if body.source_type in ("rtsp", "video") else "rtsp"
    # Auto-detect playback-style file URLs
    if url.lower().endswith((".mp4", ".avi", ".mkv", ".mov", ".webm")):
        st = "video"
    try:
        processor.open_source(url, st)
        return {"ok": True, **processor.status()}
    except Exception as exc:
        logger.exception("Failed to open RTSP/URL")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/control/play")
async def play():
    try:
        processor.play()
        return processor.status()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/control/pause")
async def pause():
    processor.pause()
    return processor.status()


@app.post("/api/control/stop")
async def stop():
    processor.stop()
    return processor.status()


@app.post("/api/conf")
async def set_conf(body: ConfBody):
    try:
        processor.set_conf(body.conf)
        return {"ok": True, "conf": body.conf}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/nms")
async def set_nms(body: NmsBody):
    try:
        processor.set_nms(body.nms)
        return {"ok": True, "nms": body.nms}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/control/speed")
async def set_speed(body: SpeedBody):
    try:
        processor.set_speed(body.speed)
        return {"ok": True, **processor.status()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/control/seek")
async def seek(body: SeekBody):
    if body.frame is None and body.time_sec is None:
        raise HTTPException(status_code=400, detail="Provide frame or time_sec")
    try:
        processor.seek(frame_idx=body.frame, time_sec=body.time_sec)
        return {"ok": True, **processor.status()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/status")
async def status():
    return processor.status()


@app.get("/api/frame.jpg")
async def latest_frame():
    data = processor.get_jpeg()
    if data is None:
        raise HTTPException(status_code=404, detail="No frame available")
    return StreamingResponse(iter([data]), media_type="image/jpeg")


@app.get("/api/stream")
async def mjpeg_stream():
    """Multipart MJPEG stream for live preview."""

    boundary = "frame"

    def gen():
        last = None
        idle_loops = 0
        while True:
            data = processor.get_jpeg()
            if data is None:
                idle_loops += 1
                if idle_loops > 300:
                    break
                time.sleep(0.05)
                continue
            idle_loops = 0
            if data is last:
                time.sleep(0.03)
                continue
            last = data
            yield (
                b"--" + boundary.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
            )
            time.sleep(0.02)

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
    )


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def create_app() -> FastAPI:
    return app
