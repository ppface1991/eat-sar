"""Lightweight detection metrics (mAP@0.5, F1) at a single score threshold.

We implement this here (instead of relying on Ultralytics .val) because we need
to compute metrics at MANY score thresholds and on ENVIRONMENT-STRATIFIED subsets,
which Ultralytics' API doesn't expose cleanly.
"""
from __future__ import annotations

import numpy as np


def box_iou_xywh(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """boxes_*: (N, 4) in xyxy. Return (Na, Nb) IoU matrix."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = boxes_a.T
    bx1, by1, bx2, by2 = boxes_b.T
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    iw = np.clip(inter_x2 - inter_x1, 0, None)
    ih = np.clip(inter_y2 - inter_y1, 0, None)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def match_predictions(pred_boxes: np.ndarray, pred_scores: np.ndarray,
                      gt_boxes: np.ndarray, iou_thresh: float = 0.5
                      ) -> tuple[np.ndarray, int]:
    """
    For one image, returns:
        tp_flag: (N_pred,) {0,1} after greedy matching with GT
        n_gt:    number of ground-truth boxes
    Predictions must already be score-filtered & sorted descending by score.
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    tp = np.zeros(n_pred, dtype=np.int8)
    if n_gt == 0 or n_pred == 0:
        return tp, n_gt

    iou = box_iou_xywh(pred_boxes, gt_boxes)
    gt_matched = np.zeros(n_gt, dtype=bool)
    # predictions are pre-sorted by descending score
    for i in range(n_pred):
        # candidate GTs sorted by IoU desc
        order = np.argsort(-iou[i])
        for j in order:
            if iou[i, j] < iou_thresh:
                break
            if not gt_matched[j]:
                gt_matched[j] = True
                tp[i] = 1
                break
    return tp, n_gt


def precision_recall_f1(tp_all: np.ndarray, scores_all: np.ndarray, n_gt_total: int):
    """
    Single-threshold P/R/F1: assume predictions are ALREADY filtered by score threshold.
    """
    if len(tp_all) == 0:
        return 0.0, 0.0, 0.0
    n_tp = int(tp_all.sum())
    n_fp = len(tp_all) - n_tp
    precision = n_tp / max(n_tp + n_fp, 1)
    recall = n_tp / max(n_gt_total, 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def average_precision(tp_all: np.ndarray, scores_all: np.ndarray, n_gt_total: int):
    """
    AP @ a single IoU threshold using the all-points interpolation (COCO-like).
    Predictions are NOT pre-filtered by a score threshold; all of them participate,
    sorted by descending score.
    """
    if len(tp_all) == 0 or n_gt_total == 0:
        return 0.0
    order = np.argsort(-scores_all)
    tp = tp_all[order]
    fp = 1 - tp
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / n_gt_total
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)

    # 101-point interpolation (COCO style)
    ap = 0.0
    rec_levels = np.linspace(0, 1, 101)
    for r in rec_levels:
        mask = recall >= r
        p = precision[mask].max() if mask.any() else 0.0
        ap += p / 101
    return float(ap)
