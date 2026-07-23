# Nava Vision AI Model Validation

Web UI to validate YOLOX models: upload **weights**, **class names**, and an **experiment file**, then run inference on **image / video / RTSP** (live or playback) with bounding boxes, play/pause, and adjustable confidence / NMS thresholds.

Uses **official YOLOX** preprocessing (`preproc` / `ValTransform`) and postprocessing (`postprocess`, `demo_postprocess`, multiclass NMS) from [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).

## Features

- Upload `.pth` / `.onnx` weights, class `.txt`, and experiment `.py`
- Inputs: image, video file, RTSP live, or playback URL
- BBoxes drawn on the preview stream
- Play / Pause / Stop
- Confidence and NMS threshold sliders (live update)
- Official YOLOX letterbox (pad=114) + decode + NMS

## Quick start (Docker — recommended for server / team)

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU speed.

```bash
./dev-start.sh          # build + start (conda env inside Docker, GPU enabled)
# open http://localhost:7860
curl -s http://localhost:7860/api/system   # should show cuda_available: true
./dev-start.sh logs
./dev-start.sh down
```

**Performance tip:** In the UI, keep Device = **Auto/CUDA**. CPU-only Docker runs ~2 FPS; GPU should be much faster. Reload the model after switching device.

### Multi-user (3+ developers)

Each browser gets a **private session** (own model, video, controls). Defaults:

| GPU VRAM | Suggested `MAX_SESSIONS` |
|----------|--------------------------|
| 8 GB     | `3` (default)            |
| 16 GB    | `4`–`6`                  |

```bash
MAX_SESSIONS=5 ./dev-start.sh
# or in .env / docker-compose environment
```

Idle sessions auto-expire after `SESSION_TTL_SEC` (default 1 hour) to free GPU memory.

Local conda (no Docker):

```bash
./dev-start.sh local
```

Manual venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/setup_yolox.sh
python run.py
```

Open **http://localhost:7860**

### Download inference video

On video / RTSP: turn on **Download inference video**, set duration in seconds (or `0` = until you turn it off), play the stream, then click **Download** when recording finishes.

## Uploads

| File | Description |
|------|-------------|
| Weights | YOLOX `.pth` or `.onnx` |
| Classes | One class name per line (see `sample_data/classes_coco.txt`) |
| Experiment | YOLOX-style `Exp` class or simple attrs (see `sample_data/yolox_s_exp.py`) |

### ONNX

Export with official YOLOX tools, then load the `.onnx` file. The UI uses `demo_postprocess` when the export is raw, then official `postprocess` + NMS.

### PyTorch `.pth`

Supported. Official [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) is vendored under `third_party/YOLOX`. The app builds the network from your experiment’s `depth`, `width`, `num_classes`, and `test_size` (even if the uploaded exp can’t be imported due to dataset deps), then loads your `.pth` checkpoint.

If `third_party/YOLOX` is missing:

```bash
git clone --depth 1 https://github.com/Megvii-BaseDetection/YOLOX.git third_party/YOLOX
pip install -r requirements.txt
```

## Sample experiment attrs

```python
class Exp:
    def __init__(self):
        self.num_classes = 80
        self.test_size = (640, 640)
        self.test_conf = 0.25
        self.nmsthre = 0.45
```

## API sketch

- `POST /api/upload` — weight + classes + exp
- `POST /api/source/file` — image / video
- `POST /api/source/rtsp` — RTSP or media URL
- `POST /api/control/play|pause|stop`
- `POST /api/conf` / `POST /api/nms`
- `GET /api/stream` — MJPEG preview
- `GET /api/status`

## Notes

- RTSP needs network access and a reachable stream (`rtsp://...`).
- CUDA: select **CUDA** in the UI when a GPU + matching PyTorch / onnxruntime-gpu is available.
- First inference on CPU can be slow; FPS is shown in the top bar.
