"""
Visualize qualitative cases: baseline vs adaptive on the same image.

Picks N images from high-wind bin (where adaptive should help most) and draws
GT + predictions for both schemes side by side.

Usage:
    python scripts/07_viz/plot_case_examples.py --config configs/default.yaml \\
        --manifest cache/preds/sardet100k_test/manifest.csv \\
        --data /root/autodl-tmp/eat-sar/yolo/sardet100k/data.yaml \\
        --split test \\
        --adaptive-json runs/sardet100k_val/adaptive_T.json \\
        --baseline-T 0.25 \\
        --env-bin high --n 6 \\
        --out runs/sardet100k_test/case_examples.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from eat_sar.adaptive import LinearT, PiecewiseT  # noqa: E402
from eat_sar.inference_cache import load_pred, load_yolo_gt  # noqa: E402


def draw_boxes(ax, boxes, color, label=None, lw=1.6):
    for (x1, y1, x2, y2) in boxes:
        rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    fill=False, edgecolor=color, lw=lw)
        ax.add_patch(rect)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--adaptive-json", required=True)
    ap.add_argument("--baseline-T", type=float, default=0.25)
    ap.add_argument("--env-bin", default="high")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--scheme", default="piecewise", choices=["piecewise", "linear"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params = json.load(open(args.adaptive_json))
    if args.scheme == "piecewise":
        T_fn = PiecewiseT(bins=params["piecewise"]["bins"],
                          thresholds=params["piecewise"]["thresholds"])
    else:
        T_fn = LinearT(a=params["linear"]["a"], b=params["linear"]["b"])
    env_value_key = params["env_value_key"]

    data_yaml = yaml.safe_load(open(args.data))
    label_root = Path(data_yaml["path"]) / "labels" / args.split

    mf = pd.read_csv(args.manifest)
    sub = mf[mf["wind_bin"] == args.env_bin].sample(
        n=min(args.n, (mf["wind_bin"] == args.env_bin).sum()),
        random_state=0,
    )

    n = len(sub)
    fig, axes = plt.subplots(n, 2, figsize=(8, 3.2 * n))
    if n == 1:
        axes = axes.reshape(1, 2)
    for i, (_, row) in enumerate(sub.iterrows()):
        img = Image.open(row["image_path"]).convert("L")
        W, H = img.size
        gt = load_yolo_gt(label_root / f"{Path(row['image_path']).stem}.txt", W, H)
        boxes, scores, _ = load_pred(Path(row["pred_npz"]))

        T_base = args.baseline_T
        T_adapt = float(T_fn(row[env_value_key])[0])

        for j, (title, T) in enumerate([
            (f"Baseline (T={T_base:.2f})", T_base),
            (f"Adaptive {args.scheme} (T={T_adapt:.2f}, env={row[env_value_key]:.1f})", T_adapt),
        ]):
            ax = axes[i, j]
            ax.imshow(img, cmap="gray")
            keep = scores >= T
            draw_boxes(ax, gt, "lime", lw=2.0)
            draw_boxes(ax, boxes[keep], "red", lw=1.4)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    # legend
    fig.suptitle(f"Qualitative: GT (green) vs Predictions (red) — env_bin={args.env_bin}",
                 fontsize=10)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
