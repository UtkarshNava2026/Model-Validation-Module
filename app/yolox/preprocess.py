"""
Official YOLOX preprocessing (from Megvii-BaseDetection/YOLOX).
Source: yolox/data/data_augment.py — preproc + ValTransform
"""

from __future__ import annotations

import cv2
import numpy as np


def preproc(img: np.ndarray, input_size: tuple[int, int], swap=(2, 0, 1)):
    """Letterbox resize with pad value 114, then HWC -> CHW float32."""
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class ValTransform:
    """Official YOLOX validation / inference transform."""

    def __init__(self, swap=(2, 0, 1), legacy: bool = False):
        self.swap = swap
        self.legacy = legacy

    def __call__(self, img: np.ndarray, res, input_size: tuple[int, int]):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 5))
