"""
YOLOX predictor supporting .pth (via YOLOX Exp) and .onnx weights.
Uses official ValTransform preprocess and postprocess / demo_postprocess.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .exp_loader import load_class_names, load_exp_file
from .model_builder import build_model
from .postprocess import demo_postprocess, postprocess
from .preprocess import ValTransform
from .visualize import vis

logger = logging.getLogger(__name__)


class YOLOXPredictor:
    def __init__(
        self,
        weight_path: str | Path,
        exp_path: str | Path,
        class_path: str | Path,
        device: str = "cpu",
        conf: float | None = None,
        nms: float | None = None,
    ):
        self.weight_path = Path(weight_path)
        self.exp = load_exp_file(exp_path)
        self.class_names = load_class_names(class_path)
        self.device = self._resolve_device(device)
        self.conf = float(conf if conf is not None else self.exp.test_conf)
        self.nms = float(nms if nms is not None else self.exp.nmsthre)
        self.test_size = self.exp.test_size
        self.num_classes = self.exp.num_classes
        self.preproc = ValTransform(legacy=self.exp.legacy)
        self.backend = "onnx" if self.weight_path.suffix.lower() == ".onnx" else "torch"
        self.fp16 = self.device == "cuda"
        self.session = None
        self.model = None
        self.input_name = None
        self._load_model()

        # Keep model/exp class count for postprocess; only adjust display names
        if len(self.class_names) != self.num_classes:
            logger.warning(
                "Class count (%d) != exp.num_classes (%d); keeping exp.num_classes for the model.",
                len(self.class_names),
                self.num_classes,
            )
            if len(self.class_names) < self.num_classes:
                self.class_names = self.class_names + [
                    f"class_{i}" for i in range(len(self.class_names), self.num_classes)
                ]
            else:
                self.class_names = self.class_names[: self.num_classes]

    @staticmethod
    def _resolve_device(device: str) -> str:
        want = (device or "auto").lower().strip()
        if want in ("auto", "gpu", "cuda"):
            if torch.cuda.is_available():
                return "cuda"
            if want != "auto":
                logger.warning("CUDA requested but not available — falling back to CPU")
            return "cpu"
        return "cpu"

    def set_conf(self, conf: float):
        self.conf = float(np.clip(conf, 0.01, 0.99))

    def set_nms(self, nms: float):
        self.nms = float(np.clip(nms, 0.01, 0.99))

    def _load_model(self):
        if self.backend == "onnx":
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            if self.device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.session = ort.InferenceSession(
                str(self.weight_path), providers=providers
            )
            self.input_name = self.session.get_inputs()[0].name
            # Infer input size from ONNX if present
            shape = self.session.get_inputs()[0].shape
            if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
                self.test_size = (int(shape[2]), int(shape[3]))
            logger.info("Loaded ONNX model: %s size=%s", self.weight_path, self.test_size)
            return

        # Torch / .pth — build via uploaded Exp or official YOLOX from parsed hyperparams
        try:
            model, exp_obj = build_model(self.exp)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to build YOLOX model for .pth weights: {exc}"
            ) from exc

        try:
            ckpt = torch.load(
                str(self.weight_path), map_location="cpu", weights_only=False
            )
        except TypeError:
            ckpt = torch.load(str(self.weight_path), map_location="cpu")

        if isinstance(ckpt, dict):
            if "model" in ckpt:
                state = ckpt["model"]
            elif "state_dict" in ckpt:
                state = ckpt["state_dict"]
            else:
                state = ckpt
        else:
            state = ckpt

        # Strip common prefixes from exported / DDP checkpoints
        cleaned = {}
        for k, v in state.items():
            nk = k
            for prefix in ("module.", "model."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix) :]
            cleaned[nk] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            logger.warning("Missing keys while loading .pth (%d): %s...", len(missing), missing[:5])
        if unexpected:
            logger.warning(
                "Unexpected keys while loading .pth (%d): %s...",
                len(unexpected),
                unexpected[:5],
            )

        model.eval()
        if self.device == "cuda":
            model.cuda()
            # Official YOLOX speed tip: fuse conv+bn, then optional FP16
            try:
                from yolox.utils import fuse_model

                model = fuse_model(model)
                logger.info("Applied YOLOX fuse_model (conv+bn)")
            except Exception as exc:
                logger.warning("fuse_model skipped: %s", exc)
            if self.fp16:
                model = model.half()
        self.model = model
        if hasattr(exp_obj, "test_size"):
            ts = exp_obj.test_size
            self.test_size = (int(ts[0]), int(ts[1]))
        self.exp.raw_exp = exp_obj
        logger.info(
            "Loaded PyTorch .pth: %s size=%s classes=%s device=%s fp16=%s",
            self.weight_path.name,
            self.test_size,
            self.num_classes,
            self.device,
            self.fp16 and self.device == "cuda",
        )

    def inference(self, img: np.ndarray) -> tuple[Any, dict]:
        """Run official preprocess + model + postprocess. Returns (output, img_info)."""
        if img is None or img.size == 0:
            raise ValueError("Empty image")

        height, width = img.shape[:2]
        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info = {
            "height": height,
            "width": width,
            "raw_img": img,
            "ratio": ratio,
        }

        processed, _ = self.preproc(img, None, self.test_size)

        if self.backend == "onnx":
            ort_input = processed[None, :, :, :]
            outputs = self.session.run(None, {self.input_name: ort_input})
            output = np.asarray(outputs[0])

            # Official ONNX demo path: decode raw heads, then NMS
            if output.ndim == 3 and output.shape[-1] >= 6:
                # Raw heads have small pre-exp size logits; decoded boxes are larger
                needs_decode = float(np.percentile(np.abs(output[..., 2:4]), 95)) < 20.0
                if needs_decode:
                    output = demo_postprocess(
                        output, self.test_size, p6=self.exp.p6
                    )
                predictions = torch.from_numpy(np.ascontiguousarray(output)).float()
                outs = postprocess(
                    predictions,
                    self.num_classes,
                    self.conf,
                    self.nms,
                    class_agnostic=True,
                )
                return outs[0], img_info

            # Fallback: some exports already return xyxy + score + cls
            if output.ndim == 2 and output.shape[-1] >= 6:
                pred = torch.from_numpy(output).float()
                return pred, img_info

            raise RuntimeError(
                f"Unsupported ONNX output shape: {output.shape}. "
                "Expected (1, N, 5+num_classes)."
            )

        tensor = torch.from_numpy(processed).unsqueeze(0)
        if self.device == "cuda":
            tensor = tensor.cuda(non_blocking=True)
            if self.fp16:
                tensor = tensor.half()
            else:
                tensor = tensor.float()
        else:
            tensor = tensor.float()
        with torch.inference_mode():
            outputs = self.model(tensor)
            # NMS ops prefer float32
            if outputs.dtype != torch.float32:
                outputs = outputs.float()
            outputs = postprocess(
                outputs,
                self.num_classes,
                self.conf,
                self.nms,
                class_agnostic=True,
            )
        return outputs[0], img_info

    def visual(self, output, img_info: dict, cls_conf: float | None = None) -> np.ndarray:
        """Draw bboxes on original image (official demo visual)."""
        ratio = img_info["ratio"]
        img = img_info["raw_img"].copy()
        if output is None:
            return img

        if isinstance(output, torch.Tensor):
            output = output.detach().cpu()

        bboxes = output[:, 0:4]
        bboxes /= ratio
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        conf = self.conf if cls_conf is None else cls_conf
        return vis(img, bboxes, scores, cls, conf, self.class_names)

    def predict_frame(self, frame: np.ndarray) -> tuple[np.ndarray, int]:
        """Convenience: inference + visualize. Returns (annotated_bgr, num_dets)."""
        output, img_info = self.inference(frame)
        annotated = self.visual(output, img_info)
        n = 0 if output is None else int(len(output))
        return annotated, n
