"""
Sensitivity analysis: number of environment bins (2, 3, 4, 5).

完全复用 val + test 的 preds cache，不重跑 inference。流程：
    1. 读 val manifest，取 clutter_std → 按 K 分位数重新分箱
    2. 对每个 K：
        a. 在 val 上 scan 阈值 → argmax F1 per bin → 得到 piecewise T(e)
        b. 用这个 T(e) 在 test 上评估，记录 ALL + per-bin P/R/F1/mAP
    3. 汇总成一张 sensitivity 表（行 = K，列 = ALL mAP, F1, ΔmAP vs baseline）

Usage:
    python scripts/09_sensitivity/sweep_bins.py \\
        --config configs/default.yaml \\
        --val-manifest /autodl-fs/data/eat-sar/runs/sardet100k_val/manifest.csv \\
        --test-manifest /autodl-fs/data/eat-sar/runs/sardet100k_test/manifest.csv \\
        --val-data /root/autodl-tmp/eat-sar-yolo/sardet100k/data.yaml \\
        --test-data /root/autodl-tmp/eat-sar-yolo/sardet100k/data.yaml \\
        --baseline-T 0.40 \\
        --tag sardet100k_bin_sweep
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


def per_image_eval(row, label_root, iou_eval, T):
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
    tp, n_gt = match_predictions(boxes_k, scores_k, gt, iou_thresh=iou_eval)
    return scores_k, tp, n_gt


def aggregate(scores, tp, n_gt):
    if len(scores) == 0:
        return dict(precision=0., recall=0., f1=0., map50=0.)
    p, r, f1 = precision_recall_f1(tp, scores, n_gt)
    ap = average_precision(tp, scores, n_gt)
    return dict(precision=p, recall=r, f1=f1, map50=ap)


def scan_best_T_per_bin(mf_val, label_root_val, iou_eval, env_col,
                        T_grid=np.arange(0.05, 0.91, 0.02)):
    """For each bin, find argmax-F1 threshold on val."""
    bin_labels = sorted(mf_val[env_col].unique().tolist())
    # cache per-image (scores, tp, n_gt) over the SAME T_grid by varying mask
    cache = {b: [] for b in bin_labels}  # list of (boxes_scores_all, tp_for_each_T?)
    # Easier: precompute scores+iou-matched tp at the LOWEST threshold, then
    # filter by mask at each T.
    print(f"  pre-loading cache for {len(mf_val)} val images...")
    per_img = []  # list of (env_bin, scores, tp_at_iou, n_gt) at conf>=0 (all)
    for _, row in tqdm(mf_val.iterrows(), total=len(mf_val), desc="  preload"):
        boxes, scores, _ = load_pred(Path(row["pred_npz"]))
        img_path = Path(row["image_path"])
        with Image.open(img_path) as im:
            W, H = im.size
        gt = load_yolo_gt(label_root_val / f"{img_path.stem}.txt", W, H)
        if len(boxes):
            order = np.argsort(-scores)
            boxes = boxes[order]; scores = scores[order]
        tp, n_gt = match_predictions(boxes, scores, gt, iou_thresh=iou_eval)
        per_img.append((row[env_col], scores, tp, n_gt))

    best_T = {}
    for b in bin_labels:
        # gather all scores/tp/n_gt in this bin
        sub = [(s, t, n) for (bb, s, t, n) in per_img if bb == b]
        if not sub:
            best_T[b] = 0.40
            continue
        n_gt_total = sum(n for (_, _, n) in sub)
        # for each T in grid: filter by score, compute F1
        best = -1
        best_T_b = 0.40
        for T in T_grid:
            sc, tp = [], []
            for (s, t, _) in sub:
                m = s >= T
                sc.append(s[m]); tp.append(t[m])
            sc = np.concatenate(sc) if sc else np.array([])
            tp = np.concatenate(tp) if tp else np.array([])
            if len(sc) == 0:
                continue
            _, _, f1 = precision_recall_f1(tp, sc, n_gt_total)
            if f1 > best:
                best = f1; best_T_b = float(T)
        best_T[b] = best_T_b
    return best_T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--baseline-T", type=float, default=0.40)
    ap.add_argument("--proxy-col", default="clutter_std")
    ap.add_argument("--K-list", nargs="+", type=int, default=[2, 3, 4, 5])
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    iou_eval = cfg["threshold"]["iou_thresh"]
    val_label_root = Path(yaml.safe_load(open(args.val_data))["path"]) / "labels" / args.val_split
    test_label_root = Path(yaml.safe_load(open(args.test_data))["path"]) / "labels" / args.test_split

    mf_val = pd.read_csv(args.val_manifest).dropna(subset=[args.proxy_col])
    mf_test = pd.read_csv(args.test_manifest).dropna(subset=[args.proxy_col])

    out_dir = ensure_dir(Path(cfg["paths"]["runs_dir"]) / args.tag)
    all_rows = []

    for K in args.K_list:
        print(f"\n=== K = {K} bins ===")
        # quantile bin edges on val
        qs = np.linspace(0, 1, K + 1)
        edges = np.quantile(mf_val[args.proxy_col].values, qs)
        edges[0] = -np.inf; edges[-1] = np.inf
        labels = [f"b{i}" for i in range(K)]
        col_K = f"bin_K{K}"
        mf_val[col_K] = pd.cut(mf_val[args.proxy_col], bins=edges,
                                labels=labels, include_lowest=True)
        mf_test[col_K] = pd.cut(mf_test[args.proxy_col], bins=edges,
                                 labels=labels, include_lowest=True)
        mf_val_K = mf_val.dropna(subset=[col_K])
        mf_test_K = mf_test.dropna(subset=[col_K])
        print(f"  edges = {edges[1:-1]}")
        print(f"  val populations: {mf_val_K[col_K].value_counts().to_dict()}")

        # 1) calibrate on val
        best_T = scan_best_T_per_bin(mf_val_K, val_label_root, iou_eval, col_K)
        print(f"  best_T per bin: {best_T}")

        # 2) evaluate on test
        sc_all, tp_all, bin_all = [], [], []
        n_gt_total = 0
        n_gt_per_bin = {b: 0 for b in labels}
        for _, row in tqdm(mf_test_K.iterrows(), total=len(mf_test_K),
                            desc=f"  K={K} test"):
            T = best_T.get(row[col_K], args.baseline_T)
            s, t, n_gt = per_image_eval(row, test_label_root, iou_eval, T)
            sc_all.append(s); tp_all.append(t)
            bin_all.append(np.array([row[col_K]] * len(s)))
            n_gt_total += n_gt
            n_gt_per_bin[row[col_K]] += n_gt
        bin_arr = np.concatenate(bin_all) if bin_all else np.array([])
        sc_arr = np.concatenate(sc_all) if sc_all else np.array([])
        tp_arr = np.concatenate(tp_all) if tp_all else np.array([])

        ov = aggregate(sc_arr, tp_arr, n_gt_total)
        all_rows.append({"K": K, "env_bin": "ALL", "best_T": str(best_T),
                          "n_gt": n_gt_total, **ov})
        for b in labels:
            m = bin_arr == b
            sub = aggregate(sc_arr[m], tp_arr[m], n_gt_per_bin[b])
            all_rows.append({"K": K, "env_bin": b, "best_T": best_T.get(b),
                              "n_gt": n_gt_per_bin[b], **sub})

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "bin_sweep.csv", index=False)
    print(f"\nSaved: {out_dir/'bin_sweep.csv'}")
    piv = df[df.env_bin == "ALL"].set_index("K")[["precision", "recall", "f1", "map50"]].round(4)
    print("\nOverall (ALL) by K:")
    print(piv)


if __name__ == "__main__":
    main()
