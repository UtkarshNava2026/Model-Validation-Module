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

## Quick start

```bash
cd Model-Validation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Open **http://localhost:7860**

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
