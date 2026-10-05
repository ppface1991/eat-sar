"""
Plot per-env-bin F1 / mAP50 vs score threshold (论文 Fig.2).

Two subplots side-by-side (F1, mAP@0.5). Each curve = one env bin.
If --adaptive-json is provided, the per-bin chosen T (vertical dashed lines)
and the baseline T (vertical solid line) are overlaid.

Usage:
    python scripts/07_viz/plot_threshold_curves.py \\
        --scan-csv runs/sardet100k_val/threshold_scan.csv \\
        --out runs/sardet100k_val/fig_threshold_curves.png \\
        [--adaptive-json runs/sardet100k_val_f1/adaptive_T.json] \\
        [--baseline-T 0.40]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COLORS = {"calm": "#2E86AB", "moderate": "#E07A5F", "rough": "#8B2C00",
          "low": "#2E86AB", "mid": "#E07A5F", "high": "#8B2C00",
          "ALL": "#444444"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adaptive-json", default=None,
                    help="Optional adaptive_T.json — overlays per-bin chosen T.")
    ap.add_argument("--baseline-T", type=float, default=None,
                    help="Optional baseline T — overlays as solid vertical line.")
    args = ap.parse_args()

    df = pd.read_csv(args.scan_csv)
    bins = [b for b in df["env_bin"].unique() if b != "ALL"]

    adaptive_T = None
    if args.adaptive_json:
        adp = json.load(open(args.adaptive_json))
        labels = adp["labels"]
        thresholds = adp["piecewise"]["thresholds"]
        adaptive_T = dict(zip(labels, thresholds))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for metric, ax in zip(["f1", "map50"], axes):
        for b in bins + ["ALL"]:
            sub = df[df["env_bin"] == b].sort_values("threshold")
            ax.plot(sub["threshold"], sub[metric], marker="o", ms=3,
                    label=b, lw=1.6, ls=("--" if b == "ALL" else "-"),
                    color=COLORS.get(b, None))
            if adaptive_T and b in adaptive_T:
                ax.axvline(adaptive_T[b], ls=":", lw=1.2,
                           color=COLORS.get(b, "k"), alpha=0.7)
        if args.baseline_T is not None:
            ax.axvline(args.baseline_T, ls="-", lw=1.0, color="k", alpha=0.5,
                       label=f"baseline T={args.baseline_T}")
        ax.set_xlabel("Score threshold T")
        ax.set_ylabel(metric.upper().replace("MAP50", "mAP@0.5"))
        ax.set_title(f"{metric.upper().replace('MAP50','mAP@0.5')} vs T")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
