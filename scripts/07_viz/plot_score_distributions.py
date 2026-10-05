"""
Visualize score distributions of positive (TP-eligible) vs negative (FP) predictions
within each environment bin. This is the core intuition figure for the paper:
the same global threshold T=0.40 has different P/R operating points in calm
vs rough bins.

For each image in the val manifest:
  - For each prediction, mark TP if it matches some GT at IoU>=0.5, else FP
  - Tag with env_bin
Then plot per-bin score histograms separately for TP and FP.

Usage:
    python scripts/07_viz/plot_score_distributions.py \\
        --config configs/default.yaml \\
        --manifest /autodl-fs/data/eat-sar/runs/sardet100k_val/manifest.csv \\
        --data /root/autodl-tmp/eat-sar-yolo/sardet100k/data.yaml \\
        --split val \\
        --env-key clutter_bin \\
        --out runs/sardet100k_val/fig_score_dists.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from eat_sar.eval_metrics import match_predictions  # noqa: E402
from eat_sar.inference_cache import load_pred, load_yolo_gt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--env-key", default="clutter_bin")
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_yaml = yaml.safe_load(open(args.data))
    label_root = Path(data_yaml["path"]) / "labels" / args.split

    mf = pd.read_csv(args.manifest).dropna(subset=[args.env_key])
    records = []
    for _, row in tqdm(mf.iterrows(), total=len(mf), desc="match"):
        boxes, scores, _ = load_pred(Path(row["pred_npz"]))
        if len(boxes) == 0:
            continue
        order = np.argsort(-scores)
        boxes = boxes[order]; scores = scores[order]
        img_path = Path(row["image_path"])
        with Image.open(img_path) as im:
            W, H = im.size
        gt = load_yolo_gt(label_root / f"{img_path.stem}.txt", W, H)
        tp, _ = match_predictions(boxes, scores, gt, iou_thresh=args.iou_thresh)
        for s, t in zip(scores, tp):
            records.append({"env_bin": row[args.env_key], "score": float(s),
                            "label": "TP" if t == 1 else "FP"})

    df = pd.DataFrame(records)
    print(f"Collected {len(df)} predictions  ({(df.label=='TP').sum()} TP / "
          f"{(df.label=='FP').sum()} FP)")

    bins = ["calm", "moderate", "rough"]
    bins = [b for b in bins if b in df.env_bin.unique()]
    n_bins = len(bins)

    fig, axes = plt.subplots(1, n_bins, figsize=(3.2 * n_bins, 2.6),
                              sharey=True, sharex=True)
    if n_bins == 1:
        axes = [axes]
    score_grid = np.linspace(0, 1, 41)
    palette = {"TP": "#2b8cbe", "FP": "#e34a33"}
    for ax, b in zip(axes, bins):
        sub = df[df.env_bin == b]
        for cls in ["TP", "FP"]:
            s = sub[sub.label == cls].score.values
            if len(s) == 0:
                continue
            ax.hist(s, bins=score_grid, alpha=0.55, label=cls,
                    color=palette[cls], density=True)
        ax.axvline(0.40, color="black", lw=0.8, ls="--")
        ax.text(0.405, ax.get_ylim()[1] * 0.92, "T=0.40", fontsize=7)
        ax.set_title(b, fontsize=9)
        ax.set_xlabel("Detection score", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Density", fontsize=8)
    axes[-1].legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
