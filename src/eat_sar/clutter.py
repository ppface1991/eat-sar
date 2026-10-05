"""
Image-derived sea-state proxy features.

For a SAR image with ship annotations available, we estimate background
(sea-clutter) statistics by:
  1. Masking out all GT ship boxes (with a small dilation margin);
  2. Computing on the remaining pixels:
        - mean intensity        (mu)
        - standard deviation    (sigma)
        - normalized std        (sigma / mu, like coefficient of variation)
        - entropy of intensity histogram (Shannon, base 2)

Physical intuition:
  - Calm sea  -> low backscatter mean, low std, low entropy (uniform dark)
  - Rough sea -> stronger backscatter, larger std, higher entropy
The std (or coefficient of variation) is the most widely used proxy in CFAR
literature; we use it as the primary feature, but compute all four for
ablation/inspection.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def shannon_entropy(arr: np.ndarray, bins: int = 64) -> float:
    """Histogram-based Shannon entropy (base 2). arr is 1-D pixel values."""
    if arr.size == 0:
        return 0.0
    hist, _ = np.histogram(arr, bins=bins, range=(0, 255))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def compute_clutter_stats(img_path: Path, gt_boxes_xyxy: np.ndarray,
                          dilate_px: int = 8) -> dict:
    """
    Args:
        img_path: path to SAR image
        gt_boxes_xyxy: (N, 4) ground-truth ship boxes in pixel coords
        dilate_px: pixels to expand each box to avoid leakage of ship pixels
                   into the "background" set
    Returns:
        dict with keys: clutter_mean, clutter_std, clutter_cv, clutter_entropy,
                        bg_pixel_ratio
    """
    im = np.array(Image.open(img_path).convert("L"))  # uint8
    H, W = im.shape
    mask = np.ones((H, W), dtype=bool)
    for (x1, y1, x2, y2) in gt_boxes_xyxy:
        x1 = max(int(x1 - dilate_px), 0)
        y1 = max(int(y1 - dilate_px), 0)
        x2 = min(int(x2 + dilate_px), W)
        y2 = min(int(y2 + dilate_px), H)
        mask[y1:y2, x1:x2] = False

    bg = im[mask].astype(np.float32)
    if bg.size < 100:
        # almost no background pixels (huge ships or tiny image) -> fall back
        bg = im.astype(np.float32).ravel()

    mu = float(bg.mean())
    sigma = float(bg.std())
    cv = sigma / max(mu, 1e-6)
    ent = shannon_entropy(bg)

    return {
        "clutter_mean": mu,
        "clutter_std": sigma,
        "clutter_cv": cv,
        "clutter_entropy": ent,
        "bg_pixel_ratio": float(mask.sum() / mask.size),
    }
