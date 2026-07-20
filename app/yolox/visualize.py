"""Bounding-box visualization helpers."""

from __future__ import annotations

import cv2
import numpy as np


_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    """Draw detections on BGR image. boxes are xyxy in original image space."""
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = float(scores[i])
        if score < conf:
            continue

        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id % len(_COLORS)] * 255).astype(np.uint8).tolist()
        text = (
            f"{class_names[cls_id]}:{score:.2f}"
            if class_names and cls_id < len(class_names)
            else f"{cls_id}:{score:.2f}"
        )
        txt_color = (0, 0, 0) if np.mean(color) > 127 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.5, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (int(color[0] * 0.7), int(color[1] * 0.7), int(color[2] * 0.7))
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1,
        )
        cv2.putText(
            img, text, (x0, y0 + txt_size[1]), font, 0.5, txt_color, thickness=1
        )
    return img
