"""
Run YOLO once at a low score threshold (default 0.001) and CACHE all predictions per image.
All later threshold sweeps and adaptive-threshold experiments operate on this cache —
NO need to re-run inference for every threshold.

Cache format (one .npz per image):
    boxes:  (N, 4) xyxy (pixels)
    scores: (N,)
    classes:(N,)   (always 0 for ship)

We also write a manifest CSV listing (file_name, image_path, label_path, env fields).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_pred(npz_path: Path, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray):
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, boxes=boxes, scores=scores, classes=classes)


def load_pred(npz_path: Path):
    d = np.load(npz_path)
    return d["boxes"], d["scores"], d["classes"]


def load_yolo_gt(label_txt: Path, img_w: int, img_h: int) -> np.ndarray:
    """Read YOLO-format label txt; return xyxy boxes in pixel coords."""
    if not label_txt.exists():
        return np.zeros((0, 4), dtype=np.float32)
    arr = np.loadtxt(label_txt, ndmin=2)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    # cols: cls cx cy w h (normalized)
    cx = arr[:, 1] * img_w
    cy = arr[:, 2] * img_h
    w = arr[:, 3] * img_w
    h = arr[:, 4] * img_h
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return xyxy.astype(np.float32)
