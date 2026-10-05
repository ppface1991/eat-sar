"""
Sensitivity analysis: bounding-box dilation parameter k for clutter_std.

For each k in K_LIST:
  1. Recompute clutter_std for all images (val + test) using dilate_px=k
  2. Re-bin into tertiles based on val distribution
  3. Recalibrate piecewise T(e) on val (argmax F1 per bin)
  4. Evaluate on test → ALL + per-bin mAP@0.5

预先要求：preds/*.npz 缓存已经存在（与 dilation 无关，只影响 proxy 值），label
files 已就绪。

Usage:
    python scripts/09_sensitivity/sweep_dilation.py \\
        --config configs/default.yaml \\
        --val-manifest /autodl-fs/data/eat-sar/runs/sardet100k_val/manifest.csv \\
        --test-manifest /autodl-fs/data/eat-sar/runs/sardet100k_test/manifest.csv \\
        --val-data /root/autodl-tmp/eat-sar-yolo/sardet100k/data.yaml \\
        --test-data /root/autodl-tmp/eat-sar-yolo/sardet100k/data.yaml \\
        --K-list 3 5 8 12 16 \\
        --baseline-T 0.40 \\
        --tag sardet100k_dilation_sweep
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
from eat_sar.clutter import compute_clutter_stats  # noqa: E402
from eat_sar.eval_metrics import (average_precision, match_predictions,  # noqa: E402
                                    precision_recall_f1)
from eat_sar.inference_cache import load_pred, load_yolo_gt  # noqa: E402
from eat_sar.utils import ensure_dir, load_config  # noqa: E402


def recompute_clutter(mf: pd.DataFrame, label_root: Path, k: int) -> pd.Series:
    vals = []
    for _, row in tqdm(mf.iterrows(), total=len(mf), desc=f"  recompute k={k}"):
        img_path = Path(row["image_path"])
        with Image.open(img_path) as im:
            W, H = im.size
        gt = load_yolo_gt(label_root / f"{img_path.stem}.txt", W, H)
        stats = compute_clutter_stats(img_path, gt, dilate_px=k)
        vals.append(stats["clutter_std"])
    return pd.Series(vals, index=mf.index)


def per_image_eval(row, label_root, iou_eval, T, std_col="clutter_std_k"):
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


def calibrate_per_bin(mf_val, label_root, iou_eval, bin_col,
                      T_grid=np.arange(0.05, 0.91, 0.02)):
    """Same as sweep_bins.scan_best_T_per_bin but with explicit bin_col."""
    bin_labels = sorted(mf_val[bin_col].unique().tolist())
    per_img = []
    for _, row in tqdm(mf_val.iterrows(), total=len(mf_val), desc="  preload val"):
        boxes, scores, _ = load_pred(Path(row["pred_npz"]))
        img_path = Path(row["image_path"])
        with Image.open(img_path) as im:
            W, H = im.size
        gt = load_yolo_gt(label_root / f"{img_path.stem}.txt", W, H)
        if len(boxes):
            order = np.argsort(-scores)
            boxes = boxes[order]; scores = scores[order]
        tp, n_gt = match_predictions(boxes, scores, gt, iou_thresh=iou_eval)
        per_img.append((row[bin_col], scores, tp, n_gt))

    best_T = {}
    for b in bin_labels:
        sub = [(s, t, n) for (bb, s, t, n) in per_img if bb == b]
        if not sub:
            best_T[b] = 0.40; continue
        n_gt_total = sum(n for (_, _, n) in sub)
        best = -1; best_T_b = 0.40
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
    ap.add_argument("--K-list", nargs="+", type=int, default=[3, 5, 8, 12, 16])
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    iou_eval = cfg["threshold"]["iou_thresh"]
    val_label_root = Path(yaml.safe_load(open(args.val_data))["path"]) / "labels" / args.val_split
    test_label_root = Path(yaml.safe_load(open(args.test_data))["path"]) / "labels" / args.test_split

    mf_val = pd.read_csv(args.val_manifest)
    mf_test = pd.read_csv(args.test_manifest)

    out_dir = ensure_dir(Path(cfg["paths"]["runs_dir"]) / args.tag)
    rows = []

    for k in args.K_list:
        print(f"\n=== dilation k = {k} px ===")
        mf_val_k = mf_val.copy()
        mf_test_k = mf_test.copy()
        mf_val_k["clutter_std_k"] = recompute_clutter(mf_val_k, val_label_root, k)
        mf_test_k["clutter_std_k"] = recompute_clutter(mf_test_k, test_label_root, k)

        # tertile bin edges on val
        q = np.quantile(mf_val_k["clutter_std_k"].values, [0, 1/3, 2/3, 1])
        q[0] = -np.inf; q[-1] = np.inf
        labels = ["calm", "moderate", "rough"]
        mf_val_k["bin_k"] = pd.cut(mf_val_k["clutter_std_k"], bins=q,
                                    labels=labels, include_lowest=True)
        mf_test_k["bin_k"] = pd.cut(mf_test_k["clutter_std_k"], bins=q,
                                     labels=labels, include_lowest=True)
        mf_val_kk = mf_val_k.dropna(subset=["bin_k"])
        mf_test_kk = mf_test_k.dropna(subset=["bin_k"])

        best_T = calibrate_per_bin(mf_val_kk, val_label_root, iou_eval, "bin_k")
        print(f"  edges = {q[1:-1]}  best_T = {best_T}")

        # test eval
        sc_all, tp_all, bin_all = [], [], []
        n_gt_total = 0
        n_gt_per_bin = {b: 0 for b in labels}
        for _, row in tqdm(mf_test_kk.iterrows(), total=len(mf_test_kk),
                            desc=f"  k={k} test"):
            T = best_T.get(row["bin_k"], args.baseline_T)
            s, t, n_gt = per_image_eval(row, test_label_root, iou_eval, T)
            sc_all.append(s); tp_all.append(t)
            bin_all.append(np.array([row["bin_k"]] * len(s)))
            n_gt_total += n_gt
            n_gt_per_bin[row["bin_k"]] += n_gt
        sc_arr = np.concatenate(sc_all) if sc_all else np.array([])
        tp_arr = np.concatenate(tp_all) if tp_all else np.array([])
        bin_arr = np.concatenate(bin_all) if bin_all else np.array([])

        if len(sc_arr) > 0:
            p, r, f1 = precision_recall_f1(tp_arr, sc_arr, n_gt_total)
            ap_ = average_precision(tp_arr, sc_arr, n_gt_total)
        else:
            p = r = f1 = ap_ = 0.0
        rows.append({"k": k, "env_bin": "ALL", "best_T": str(best_T),
                      "precision": p, "recall": r, "f1": f1, "map50": ap_,
                      "n_gt": n_gt_total})
        for b in labels:
            m = bin_arr == b
            if m.sum() > 0:
                p, r, f1 = precision_recall_f1(tp_arr[m], sc_arr[m], n_gt_per_bin[b])
                ap_ = average_precision(tp_arr[m], sc_arr[m], n_gt_per_bin[b])
            else:
                p = r = f1 = ap_ = 0.0
            rows.append({"k": k, "env_bin": b, "best_T": best_T.get(b),
                          "precision": p, "recall": r, "f1": f1, "map50": ap_,
                          "n_gt": n_gt_per_bin[b]})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "dilation_sweep.csv", index=False)
    print(f"\nSaved: {out_dir/'dilation_sweep.csv'}")
    piv = df[df.env_bin == "ALL"].set_index("k")[["precision", "recall", "f1", "map50"]].round(4)
    print("\nOverall (ALL) by k:")
    print(piv)


if __name__ == "__main__":
    main()
