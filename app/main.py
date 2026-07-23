"""
Nava Vision AI Model Validation — FastAPI backend.

Multi-user: each browser gets an isolated session (own model / stream / controls).
"""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.sessions import UserSession, session_manager
from app.yolox.predictor import YOLOXPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model-validation")

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "uploads"
STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Nava Vision AI Model Validation", version="1.1.0")


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
    source_type: str = "rtsp"


class DirBody(BaseModel):
    path: str


class GotoImageBody(BaseModel):
    index: int = Field(..., ge=0)


class RecordBody(BaseModel):
    duration_sec: Optional[float] = Field(None, ge=0, le=3600)


def _session_response(extra: dict | None = None, sess: UserSession | None = None) -> dict:
    data = dict(extra or {})
    if sess is not None:
        data["session_id"] = sess.session_id
        data["sessions"] = session_manager.info()
    return data


def _resolve_session_id(
    x_session_id: Optional[str],
    session_id: Optional[str],
) -> Optional[str]:
    # Header preferred; query param used for <img src> stream/frame
    return x_session_id or session_id


def get_session(
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-Id"),
    session_id: Optional[str] = Query(default=None),
) -> UserSession:
    try:
        return session_manager.get_or_create(
            _resolve_session_id(x_session_id, session_id)
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def require_session(
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-Id"),
    session_id: Optional[str] = Query(default=None),
) -> UserSession:
    try:
        return session_manager.get(_resolve_session_id(x_session_id, session_id))
    except RuntimeError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


async def _save_upload(file: UploadFile, prefix: str, session_id: str) -> Path:
    sess_dir = UPLOAD_DIR / session_id[:16]
    sess_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "file").suffix
    dest = sess_dir / f"{prefix}_{uuid.uuid4().hex}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest


async def _save_gallery_uploads(files: list[UploadFile], session_id: str) -> list[Path]:
    gallery_dir = UPLOAD_DIR / session_id[:16] / f"gallery_{uuid.uuid4().hex}"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for file in files:
        name = Path(file.filename or "image.jpg").name
        dest = gallery_dir / name
        if dest.exists():
            dest = gallery_dir / f"{dest.stem}_{uuid.uuid4().hex[:8]}{dest.suffix}"
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        saved.append(dest)
    return saved


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/session")
async def create_session(sess: UserSession = Depends(get_session)):
    """Create or resume a user session."""
    return _session_response(
        {
            "ok": True,
            "session_id": sess.session_id,
            **session_manager.info(),
        },
        sess,
    )


@app.delete("/api/session")
async def delete_session(
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-Id"),
):
    if not x_session_id:
        raise HTTPException(status_code=400, detail="X-Session-Id required")
    session_manager.release(x_session_id)
    return {"ok": True, **session_manager.info()}


@app.post("/api/session/reset-all")
async def reset_all_sessions():
    """Admin helper: clear all sessions (frees GPU slots)."""
    session_manager.clear_all()
    return {"ok": True, **session_manager.info()}


@app.post("/api/upload")
async def upload_model(
    weight: UploadFile = File(..., description="YOLOX .pth or .onnx weights"),
    classes: UploadFile = File(..., description="Class names .txt"),
    exp: UploadFile = File(..., description="YOLOX experiment .py"),
    device: str = Form("auto"),
    conf: Optional[float] = Form(None),
    nms: Optional[float] = Form(None),
    sess: UserSession = Depends(get_session),
):
    try:
        weight_path = await _save_upload(weight, "weight", sess.session_id)
        class_path = await _save_upload(classes, "classes", sess.session_id)
        exp_path = await _save_upload(exp, "exp", sess.session_id)

        # Free previous model in this session before loading a new one
        old = sess.processor.predictor
        if old is not None:
            sess.processor.set_predictor(None)
            del old
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        predictor = YOLOXPredictor(
            weight_path=weight_path,
            exp_path=exp_path,
            class_path=class_path,
            device=device,
            conf=conf,
            nms=nms,
        )
        sess.processor.set_predictor(predictor)

        return _session_response(
            {
                "ok": True,
                "message": "Model loaded successfully",
                "backend": predictor.backend,
                "num_classes": predictor.num_classes,
                "class_names": predictor.class_names,
                "test_size": predictor.test_size,
                "conf": predictor.conf,
                "nms": predictor.nms,
                "device": predictor.device,
                "fp16": bool(
                    getattr(predictor, "fp16", False) and predictor.device == "cuda"
                ),
            },
            sess,
        )
    except Exception as exc:
        logger.exception("Upload/load failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/source/file")
async def open_file_source(
    file: UploadFile = File(...),
    source_type: str = Form(...),
    sess: UserSession = Depends(require_session),
):
    if source_type not in ("image", "video"):
        raise HTTPException(status_code=400, detail="source_type must be image or video")
    try:
        path = await _save_upload(file, source_type, sess.session_id)
        sess.processor.open_source(str(path), source_type)
        return _session_response({"ok": True, **sess.processor.status()}, sess)
    except Exception as exc:
        logger.exception("Failed to open file source")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/source/folder")
async def open_folder_upload(
    files: list[UploadFile] = File(...),
    sess: UserSession = Depends(require_session),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    try:
        saved = await _save_gallery_uploads(files, sess.session_id)
        image_paths = [
            str(p)
            for p in saved
            if p.suffix.lower()
            in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        ]
        if not image_paths:
            raise ValueError("No image files in the selected folder")
        sess.processor.open_image_list(
            image_paths, label=str(Path(image_paths[0]).parent)
        )
        return _session_response({"ok": True, **sess.processor.status()}, sess)
    except Exception as exc:
        logger.exception("Failed to open folder upload")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/source/directory")
async def open_directory_path(
    body: DirBody, sess: UserSession = Depends(require_session)
):
    path = body.path.strip()
    if not path:
        raise HTTPException(status_code=400, detail="Directory path is empty")
    try:
        sess.processor.open_image_directory(path)
        return _session_response({"ok": True, **sess.processor.status()}, sess)
    except Exception as exc:
        logger.exception("Failed to open directory path")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/control/next-image")
async def next_image(sess: UserSession = Depends(require_session)):
    try:
        sess.processor.next_image()
        return _session_response({"ok": True, **sess.processor.status()}, sess)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/control/prev-image")
async def prev_image(sess: UserSession = Depends(require_session)):
    try:
        sess.processor.prev_image()
        return _session_response({"ok": True, **sess.processor.status()}, sess)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/control/goto-image")
async def goto_image(body: GotoImageBody, sess: UserSession = Depends(require_session)):
    try:
        sess.processor.goto_image(body.index)
        return _session_response({"ok": True, **sess.processor.status()}, sess)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/source/rtsp")
async def open_rtsp(body: RtspBody, sess: UserSession = Depends(require_session)):
    url = body.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="RTSP/URL is empty")
    st = body.source_type if body.source_type in ("rtsp", "video") else "rtsp"
    if url.lower().endswith((".mp4", ".avi", ".mkv", ".mov", ".webm")):
        st = "video"
    try:
        sess.processor.open_source(url, st)
        return _session_response({"ok": True, **sess.processor.status()}, sess)
    except Exception as exc:
        logger.exception("Failed to open RTSP/URL")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/control/play")
async def play(sess: UserSession = Depends(require_session)):
    try:
        sess.processor.play()
        return _session_response(sess.processor.status(), sess)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/control/pause")
async def pause(sess: UserSession = Depends(require_session)):
    sess.processor.pause()
    return _session_response(sess.processor.status(), sess)


@app.post("/api/control/stop")
async def stop(sess: UserSession = Depends(require_session)):
    sess.processor.stop()
    return _session_response(sess.processor.status(), sess)


@app.post("/api/conf")
async def set_conf(body: ConfBody, sess: UserSession = Depends(require_session)):
    try:
        sess.processor.set_conf(body.conf)
        return _session_response({"ok": True, "conf": body.conf}, sess)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/nms")
async def set_nms(body: NmsBody, sess: UserSession = Depends(require_session)):
    try:
        sess.processor.set_nms(body.nms)
        return _session_response({"ok": True, "nms": body.nms}, sess)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/control/speed")
async def set_speed(body: SpeedBody, sess: UserSession = Depends(require_session)):
    try:
        sess.processor.set_speed(body.speed)
        return _session_response({"ok": True, **sess.processor.status()}, sess)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/control/seek")
async def seek(body: SeekBody, sess: UserSession = Depends(require_session)):
    if body.frame is None and body.time_sec is None:
        raise HTTPException(status_code=400, detail="Provide frame or time_sec")
    try:
        sess.processor.seek(frame_idx=body.frame, time_sec=body.time_sec)
        return _session_response({"ok": True, **sess.processor.status()}, sess)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/record/start")
async def record_start(body: RecordBody, sess: UserSession = Depends(require_session)):
    try:
        sess.processor.start_recording(duration_sec=body.duration_sec)
        return _session_response({"ok": True, **sess.processor.status()}, sess)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/record/stop")
async def record_stop(sess: UserSession = Depends(require_session)):
    try:
        path = sess.processor.stop_recording()
        status = sess.processor.status()
        return _session_response(
            {"ok": True, "saved": str(path) if path else None, **status}, sess
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/record/download")
async def record_download(sess: UserSession = Depends(require_session)):
    path = sess.processor.get_download_path()
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="No recorded video ready")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=path.name,
        headers={"Content-Disposition": f'attachment; filename="{path.name}"'},
    )


@app.get("/api/system")
async def system_info():
    import torch

    cuda = bool(torch.cuda.is_available())
    name = torch.cuda.get_device_name(0) if cuda else None
    mem = None
    if cuda:
        props = torch.cuda.get_device_properties(0)
        mem = round(props.total_memory / (1024**3), 2)
    return {
        "cuda_available": cuda,
        "gpu_name": name,
        "gpu_memory_gb": mem,
        "torch": torch.__version__,
        "recommended_device": "cuda" if cuda else "cpu",
        **session_manager.info(),
        "guidance": {
            "8gb": "MAX_SESSIONS=2 or 3 recommended",
            "16gb": "MAX_SESSIONS=4 to 6 usually fine for YOLOX-S",
        },
    }


@app.get("/api/status")
async def status(
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-Id"),
    session_id: Optional[str] = Query(default=None),
):
    """Return this user's status. Does NOT create a session if id is missing."""
    sid = _resolve_session_id(x_session_id, session_id)
    if not sid:
        return {
            "status": "no_session",
            "model_loaded": False,
            "sessions": session_manager.info(),
        }
    try:
        sess = session_manager.get(sid)
    except RuntimeError:
        # Unknown id — create only via /api/session
        return {
            "status": "no_session",
            "model_loaded": False,
            "sessions": session_manager.info(),
            "detail": "Unknown session; call POST /api/session",
        }
    st = sess.processor.status()
    pred = sess.processor.predictor
    st["device"] = getattr(pred, "device", None) if pred else None
    return _session_response(st, sess)


@app.get("/api/frame.jpg")
async def latest_frame(sess: UserSession = Depends(require_session)):
    data = sess.processor.get_jpeg()
    if data is None:
        raise HTTPException(status_code=404, detail="No frame available")
    return StreamingResponse(iter([data]), media_type="image/jpeg")


@app.get("/api/stream")
async def mjpeg_stream(sess: UserSession = Depends(require_session)):
    """Multipart MJPEG stream for this session's preview."""
    boundary = "frame"
    processor = sess.processor

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
