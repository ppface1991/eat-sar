"""
Scan score thresholds globally and per env-bin; output mAP50/F1 curves.

For each score threshold T in [scan_min, scan_max]:
    Filter predictions >= T  (operationally; we still use ALL preds for AP)
    Compute:
        - F1 @ T (TP/FP from filtered preds; FN from unmatched GT)
        - mAP@0.5  (uses all preds with score >= T as candidate set, then PR curve)
We do this:
    - on the WHOLE eval set
    - on each environment bin (low/mid/high wind, or calm/moderate/rough swh)

Outputs:
    runs/<tag>/threshold_scan.csv   (rows: env_bin x threshold)
    runs/<tag>/best_per_bin.csv     (best threshold per bin)

Usage:
    python scripts/04_threshold/scan_thresholds.py --config configs/default.yaml \\
        --manifest cache/preds/sardet100k_val/manifest.csv \\
        --data /root/autodl-tmp/eat-sar/yolo/sardet100k/data.yaml \\
        --split val \\
        --env-key wind_bin \\
        --tag sardet100k_val
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from eat_sar.eval_metrics import (average_precision, match_predictions,  # noqa: E402
                                    precision_recall_f1)
from eat_sar.inference_cache import load_pred, load_yolo_gt  # noqa: E402
from eat_sar.utils import ensure_dir, load_config  # noqa: E402


def gather_per_image(row, label_root: Path, iou_thresh: float):
    """For one image, load preds + GT, return per-pred (score, tp) and n_gt."""
    boxes, scores, _ = load_pred(Path(row["pred_npz"]))
    # GT
    img_path = Path(row["image_path"])
    with Image.open(img_path) as im:
        W, H = im.size
    label_txt = label_root / f"{img_path.stem}.txt"
    gt = load_yolo_gt(label_txt, W, H)

    # sort preds by score desc for matching
    if len(boxes) > 0:
        order = np.argsort(-scores)
        boxes = boxes[order]; scores = scores[order]
    tp, n_gt = match_predictions(boxes, scores, gt, iou_thresh=iou_thresh)
    return scores, tp, n_gt


def compute_metrics_at_threshold(score_tp_pairs, n_gt_total, T):
    """
    score_tp_pairs: concatenated (scores, tp) over images (already sorted? no need).
        For F1 we filter by T; for AP we keep all preds with score >= T as candidate set.
    """
    scores = score_tp_pairs[0]
    tp = score_tp_pairs[1]
    mask = scores >= T
    s_keep = scores[mask]
    t_keep = tp[mask]
    p, r, f1 = precision_recall_f1(t_keep, s_keep, n_gt_total)
    ap = average_precision(t_keep, s_keep, n_gt_total)
    return p, r, f1, ap, int(mask.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--env-key", default="wind_bin", choices=["wind_bin", "swh_bin"])
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    iou_thresh = cfg["threshold"]["iou_thresh"]
    Ts = np.arange(
        cfg["threshold"]["scan_min"],
        cfg["threshold"]["scan_max"] + 1e-9,
        cfg["threshold"]["scan_step"],
    )

    data_yaml = yaml.safe_load(open(args.data))
    label_root = Path(data_yaml["path"]) / "labels" / args.split

    mf = pd.read_csv(args.manifest)
    mf = mf.dropna(subset=[args.env_key])
    print(f"Loaded manifest: {len(mf)} rows; env-key={args.env_key}")
    print(mf[args.env_key].value_counts())

    # Step 1: gather (scores, tp) per image
    all_scores, all_tp, env_bins = [], [], []
    n_gt_per_bin: dict[str, int] = {}
    n_gt_total = 0
    pred_bin: list[str] = []  # bin label for each pred (for filtering)
    for _, row in tqdm(mf.iterrows(), total=len(mf), desc="match preds vs GT"):
        s, t, n_gt = gather_per_image(row, label_root, iou_thresh)
        all_scores.append(s); all_tp.append(t)
        b = row[args.env_key]
        pred_bin.extend([b] * len(s))
        n_gt_per_bin[b] = n_gt_per_bin.get(b, 0) + n_gt
        n_gt_total += n_gt
    all_scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
    all_tp = np.concatenate(all_tp) if all_tp else np.zeros(0, dtype=np.int8)
    pred_bin = np.array(pred_bin)

    # Step 2: scan thresholds
    rows = []
    # overall
    for T in Ts:
        p, r, f1, ap_, n_keep = compute_metrics_at_threshold((all_scores, all_tp), n_gt_total, T)
        rows.append(dict(env_bin="ALL", threshold=round(float(T), 4),
                         precision=p, recall=r, f1=f1, map50=ap_,
                         n_pred_kept=n_keep, n_gt=n_gt_total))
    # per-bin
    for b, n_gt_b in n_gt_per_bin.items():
        mask = pred_bin == b
        s_b = all_scores[mask]; t_b = all_tp[mask]
        for T in Ts:
            p, r, f1, ap_, n_keep = compute_metrics_at_threshold((s_b, t_b), n_gt_b, T)
            rows.append(dict(env_bin=b, threshold=round(float(T), 4),
                             precision=p, recall=r, f1=f1, map50=ap_,
                             n_pred_kept=n_keep, n_gt=n_gt_b))

    df = pd.DataFrame(rows)
    out_dir = ensure_dir(Path(cfg["paths"]["runs_dir"]) / args.tag)
    out_csv = out_dir / "threshold_scan.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Best threshold per bin under multiple selection criteria
    # We compute several so the paper can pick the most compelling story.
    def pick_best(group, metric, constraint=None):
        sub = group
        if constraint is not None:
            col, op, val = constraint
            if op == ">=":
                sub = group[group[col] >= val]
            elif op == "<=":
                sub = group[group[col] <= val]
        if len(sub) == 0:
            sub = group  # fallback if constraint not satisfiable
        return sub.sort_values(metric, ascending=False).head(1)

    selections = {
        "argmax_f1": ("f1", None),
        "argmax_map50": ("map50", None),
        "max_recall_at_p085": ("recall", ("precision", ">=", 0.85)),
        "max_recall_at_p090": ("recall", ("precision", ">=", 0.90)),
        "max_f1_at_p085": ("f1", ("precision", ">=", 0.85)),
    }

    best_records = []
    for crit_name, (metric, constr) in selections.items():
        for b in df["env_bin"].unique():
            grp = df[df["env_bin"] == b]
            row = pick_best(grp, metric, constr).iloc[0].to_dict()
            row["selection"] = crit_name
            best_records.append(row)
    best = pd.DataFrame(best_records)
    best = best[["selection", "env_bin", "threshold", "precision", "recall", "f1", "map50", "n_pred_kept", "n_gt"]]
    best_csv = out_dir / "best_per_bin.csv"
    best.to_csv(best_csv, index=False)
    print(f"Saved: {best_csv}")
    # Print only the most interesting selections
    print("\n=== argmax_f1 (legacy) ===")
    print(best[best.selection == "argmax_f1"].to_string(index=False))
    print("\n=== max_recall_at_p085 (recommended for the paper) ===")
    print(best[best.selection == "max_recall_at_p085"].to_string(index=False))


if __name__ == "__main__":
    main()
