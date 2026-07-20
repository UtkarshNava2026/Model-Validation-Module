"""Load YOLOX experiment (.py) files and class name lists."""

from __future__ import annotations

import ast
import importlib.util
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExpConfig:
    """Minimal experiment config needed for inference."""

    num_classes: int = 80
    test_size: tuple[int, int] = (640, 640)
    test_conf: float = 0.25
    nmsthre: float = 0.45
    depth: float = 1.0
    width: float = 1.0
    name: str = "yolox"
    p6: bool = False
    legacy: bool = False
    raw_exp: Any = field(default=None, repr=False)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _safe_literal(raw: str):
    raw = raw.strip().rstrip(",")
    try:
        return ast.literal_eval(raw)
    except Exception:
        try:
            return eval(raw, {"__builtins__": {}}, {})
        except Exception:
            return raw.strip("\"'")


def _extract_attr(text: str, name: str, default=None):
    """
    Extract `self.<name> = <literal>` assignments only.
    Ignores kwargs like `nmsthre=self.nmsthre` that appear in method calls.
    """
    patterns = [
        # self.nmsthre = 0.45
        rf"self\.{name}\s*=\s*([^\n#]+)",
        # class-level / bare: nmsthre = 0.45  (not nmsthre=self.xxx)
        rf"(?<![\w.]){name}\s*=\s*(?!self\.)([^\n#]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return _safe_literal(m.group(1))
    return default


def _as_float(value, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _parse_test_size(value) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (int(value[0]), int(value[1]))
    if isinstance(value, int):
        return (value, value)
    return (640, 640)


def _from_exp_object(exp_obj: Any, fallback_name: str) -> ExpConfig:
    test_size = _parse_test_size(getattr(exp_obj, "test_size", (640, 640)))
    return ExpConfig(
        num_classes=_as_int(getattr(exp_obj, "num_classes", 80), 80),
        test_size=test_size,
        test_conf=_as_float(getattr(exp_obj, "test_conf", 0.25), 0.25),
        nmsthre=_as_float(getattr(exp_obj, "nmsthre", 0.45), 0.45),
        depth=_as_float(getattr(exp_obj, "depth", 1.0), 1.0),
        width=_as_float(getattr(exp_obj, "width", 1.0), 1.0),
        name=str(getattr(exp_obj, "exp_name", None) or getattr(exp_obj, "name", fallback_name)),
        p6=_as_bool(getattr(exp_obj, "p6", False)),
        legacy=_as_bool(getattr(exp_obj, "legacy", False)),
        raw_exp=exp_obj,
    )


def _try_import_exp(exp_path: Path):
    module_name = f"user_exp_{exp_path.stem}_{abs(hash(str(exp_path))) % 10_000_000}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(exp_path))
        if not spec or not spec.loader:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        parent = str(exp_path.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        spec.loader.exec_module(module)
        if hasattr(module, "Exp"):
            return module.Exp()
    except Exception as exc:
        logger.info("Could not import Exp from %s (%s); using text parse.", exp_path.name, exc)
    return None


def load_exp_file(exp_path: str | Path) -> ExpConfig:
    """
    Load a YOLOX experiment description file.
    Tries official Exp class import first; falls back to safe text parsing.
    """
    exp_path = Path(exp_path).resolve()
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {exp_path}")

    exp_obj = _try_import_exp(exp_path)
    if exp_obj is not None:
        return _from_exp_object(exp_obj, exp_path.stem)

    # Regex / text fallback for files that import missing yolox package deps
    text = _read_text(exp_path)
    raw_name = _extract_attr(text, "name", None)
    # Only keep simple string names; skip expressions like os.path.split(...)
    if not isinstance(raw_name, str) or any(ch in raw_name for ch in "()[]"):
        raw_name = exp_path.stem

    cfg = ExpConfig(
        num_classes=_as_int(_extract_attr(text, "num_classes", 80), 80),
        test_size=_parse_test_size(_extract_attr(text, "test_size", (640, 640))),
        test_conf=_as_float(_extract_attr(text, "test_conf", 0.25), 0.25),
        nmsthre=_as_float(_extract_attr(text, "nmsthre", 0.45), 0.45),
        depth=_as_float(_extract_attr(text, "depth", 1.0), 1.0),
        width=_as_float(_extract_attr(text, "width", 1.0), 1.0),
        name=str(raw_name or exp_path.stem),
        p6=_as_bool(_extract_attr(text, "p6", False)),
        legacy=_as_bool(_extract_attr(text, "legacy", False)),
    )
    logger.info(
        "Parsed exp via text fallback: classes=%s size=%s conf=%s nms=%s",
        cfg.num_classes,
        cfg.test_size,
        cfg.test_conf,
        cfg.nmsthre,
    )
    return cfg


def load_class_names(class_path: str | Path) -> list[str]:
    """Load class names from a text file (one class per line)."""
    class_path = Path(class_path)
    if not class_path.exists():
        raise FileNotFoundError(f"Class file not found: {class_path}")

    names: list[str] = []
    for line in class_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Support "0: person" or "person"
        if ":" in line and line.split(":", 1)[0].strip().isdigit():
            line = line.split(":", 1)[1].strip()
        names.append(line)
    if not names:
        raise ValueError("Class file is empty")
    return names
