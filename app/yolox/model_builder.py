"""
Build a YOLOX torch model for .pth checkpoints.

Uses the official YOLOX Exp.get_model() API. If the uploaded experiment file
cannot be imported (missing dataset deps, etc.), rebuilds a compatible Exp
from parsed depth / width / num_classes / test_size.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from .exp_loader import ExpConfig

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_VENDORED_YOLOX = _ROOT / "third_party" / "YOLOX"


def _ensure_yolox_path() -> None:
    """Prefer vendored Megvii YOLOX (PyPI package is often broken on py3.12)."""
    if _VENDORED_YOLOX.is_dir():
        path = str(_VENDORED_YOLOX)
        if path not in sys.path:
            sys.path.insert(0, path)


def _require_yolox():
    _ensure_yolox_path()
    try:
        import yolox  # noqa: F401
        from yolox.exp import Exp as BaseExp
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch .pth loading requires official YOLOX sources. "
            "Clone them with:\n"
            "  git clone --depth 1 https://github.com/Megvii-BaseDetection/YOLOX.git third_party/YOLOX\n"
            "Then: pip install tabulate loguru thop psutil tqdm"
        ) from exc
    return BaseExp


def build_exp_from_config(cfg: ExpConfig):
    """Create an official YOLOX Exp configured from parsed ExpConfig."""
    BaseExp = _require_yolox()

    class RuntimeExp(BaseExp):
        def __init__(self):
            super().__init__()
            self.num_classes = int(cfg.num_classes)
            self.depth = float(cfg.depth)
            self.width = float(cfg.width)
            self.test_size = tuple(cfg.test_size)
            self.input_size = tuple(cfg.test_size)
            self.test_conf = float(cfg.test_conf)
            self.nmsthre = float(cfg.nmsthre)
            self.exp_name = str(cfg.name or "yolox")

    return RuntimeExp()


def resolve_exp(cfg: ExpConfig) -> Any:
    """
    Prefer the uploaded Exp object when it can build a model; otherwise
    synthesize one from parsed hyperparameters (depth/width/num_classes).
    """
    exp_obj = cfg.raw_exp
    if exp_obj is not None and callable(getattr(exp_obj, "get_model", None)):
        # Keep uploaded overrides in sync with class file / UI if needed
        try:
            exp_obj.num_classes = int(cfg.num_classes)
            if hasattr(exp_obj, "test_conf"):
                exp_obj.test_conf = float(cfg.test_conf)
            if hasattr(exp_obj, "nmsthre"):
                exp_obj.nmsthre = float(cfg.nmsthre)
        except Exception:
            pass
        logger.info("Using uploaded Exp.get_model()")
        return exp_obj

    logger.info(
        "Building YOLOX from parsed exp: depth=%s width=%s classes=%s size=%s",
        cfg.depth,
        cfg.width,
        cfg.num_classes,
        cfg.test_size,
    )
    return build_exp_from_config(cfg)


def build_model(cfg: ExpConfig):
    """Return (model, exp) ready for load_state_dict."""
    exp = resolve_exp(cfg)
    model = exp.get_model()
    model.eval()
    return model, exp
