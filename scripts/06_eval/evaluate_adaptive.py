"""
Final test-set evaluation:
    Baseline:  fixed threshold T0 (e.g. 0.25 or argmax-on-ALL from val scan)
    Adaptive (piecewise):  T = piecewise(env)
    Adaptive (linear):     T = a * env + b

Reports overall + stratified P/R/F1/mAP50.

Usage:
    python scripts/06_eval/evaluate_adaptive.py --config configs/default.yaml \\
        --manifest cache/preds/sardet100k_test/manifest.csv \\
        --data /root/autodl-tmp/eat-sar/yolo/sardet100k/data.yaml \\
        --split test \\
        --adaptive-json runs/sardet100k_val/adaptive_T.json \\
        --baseline-T 0.25 \\
        --tag sardet100k_test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from eat_sar.adaptive import LinearT, PiecewiseT  # noqa: E402
from eat_sar.eval_metrics import (average_precision, match_predictions,  # noqa: E402
                                    precision_recall_f1)
from eat_sar.inference_cache import load_pred, load_yolo_gt  # noqa: E402
from eat_sar.utils import ensure_dir, load_config  # noqa: E402


def per_image_eval(row, label_root, iou_thresh, T):
    """Apply per-image threshold T (scalar), return (s_kept, tp_kept, n_gt)."""
    boxes, scores, _ = load_pred(Path(row["pred_npz"]))
    img_path = Path(row["image_path"])
    with Image.open(img_path) as im:
        W, H = im.size
    gt = load_yolo_gt(label_root / f"{img_path.stem}.txt", W, H)
    if len(boxes):
        order = np.argsort(-scores)
        boxes = boxes[order]; scores = scores[order]
    mask = scores >= T
    boxes_k, scores_k = boxes[mask], scores[mask]
    tp, n_gt = match_predictions(boxes_k, scores_k, gt, iou_thresh=iou_thresh)
    return scores_k, tp, n_gt


def aggregate(scores_list, tp_list, n_gt_total):
    if len(scores_list) == 0:
        return dict(precision=0., recall=0., f1=0., map50=0., n_pred=0, n_gt=n_gt_total)
    s = np.concatenate(scores_list); t = np.concatenate(tp_list)
    p, r, f1 = precision_recall_f1(t, s, n_gt_total)
    ap = average_precision(t, s, n_gt_total)
    return dict(precision=p, recall=r, f1=f1, map50=ap, n_pred=int(len(s)), n_gt=n_gt_total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--adaptive-json", required=True)
    ap.add_argument("--baseline-T", type=float, default=0.25)
    ap.add_argument("--env-key", default="wind_bin")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    iou_thresh = cfg["threshold"]["iou_thresh"]
    params = json.load(open(args.adaptive_json))
    env_value_key = params["env_value_key"]
    piecewise = PiecewiseT(bins=params["piecewise"]["bins"],
                             thresholds=params["piecewise"]["thresholds"])
    lin = LinearT(a=params["linear"]["a"], b=params["linear"]["b"],
                  clip_min=params["linear"]["clip_min"], clip_max=params["linear"]["clip_max"])

    data_yaml = yaml.safe_load(open(args.data))
    label_root = Path(data_yaml["path"]) / "labels" / args.split

    mf = pd.read_csv(args.manifest)
    mf = mf.dropna(subset=[args.env_key, env_value_key])

    schemes = {
        "baseline": lambda env: args.baseline_T,
        "adaptive_piecewise": lambda env: float(piecewise(env)[0]),
        "adaptive_linear": lambda env: float(lin(env)[0]),
    }

    # Run all schemes; collect per-image results once per scheme
    bin_labels = sorted(mf[args.env_key].unique().tolist())
    results_rows = []

    for sch_name, T_fn in schemes.items():
        # collect (scores, tp, bin) per image
        sc_all, tp_all, bin_all = [], [], []
        n_gt_total = 0
        n_gt_per_bin: dict[str, int] = {b: 0 for b in bin_labels}
        for _, row in tqdm(mf.iterrows(), total=len(mf), desc=f"{sch_name}"):
            T = T_fn(row[env_value_key])
            s, t, n_gt = per_image_eval(row, label_root, iou_thresh, T)
            sc_all.append(s); tp_all.append(t)
            bin_all.append(np.array([row[args.env_key]] * len(s)))
            n_gt_total += n_gt
            n_gt_per_bin[row[args.env_key]] += n_gt
        bin_all = np.concatenate(bin_all) if bin_all else np.array([])
        sc_concat = np.concatenate(sc_all) if sc_all else np.array([])
        tp_concat = np.concatenate(tp_all) if tp_all else np.array([])

        # Overall
        overall = aggregate([sc_concat], [tp_concat], n_gt_total)
        results_rows.append({"scheme": sch_name, "env_bin": "ALL", **overall})
        # Per bin
        for b in bin_labels:
            mask = bin_all == b
            sub = aggregate([sc_concat[mask]], [tp_concat[mask]], n_gt_per_bin[b])
            results_rows.append({"scheme": sch_name, "env_bin": b, **sub})

    df = pd.DataFrame(results_rows)
    out_dir = ensure_dir(Path(cfg["paths"]["runs_dir"]) / args.tag)
    out_csv = out_dir / "final_eval.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    # Pretty print
    piv_f1 = df.pivot(index="env_bin", columns="scheme", values="f1").round(4)
    piv_map = df.pivot(index="env_bin", columns="scheme", values="map50").round(4)
    print("\nF1 by env bin:")
    print(piv_f1)
    print("\nmAP@0.5 by env bin:")
    print(piv_map)


if __name__ == "__main__":
    main()
