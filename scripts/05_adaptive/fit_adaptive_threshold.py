"""
Fit both forms of T(e) from the (env_value, best_threshold) pairs derived from
the VALIDATION threshold scan.

For piecewise: directly take per-bin argmax from threshold_scan.csv.
For linear:    use per-image best-threshold or per-bin median env value as anchor.
               We use per-bin mean env value -> best threshold.

Outputs:
    runs/<tag>/adaptive_T.json   { "piecewise": {...}, "linear": {a, b} }
    runs/<tag>/adaptive_T.png    visualisation

Usage:
    python scripts/05_adaptive/fit_adaptive_threshold.py --config configs/default.yaml \\
        --scan-csv runs/sardet100k_val/threshold_scan.csv \\
        --manifest cache/preds/sardet100k_val/manifest.csv \\
        --env-key wind_bin --env-value-key wind_speed \\
        --tag sardet100k_val
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from eat_sar.adaptive import LinearT, PiecewiseT, fit_linear  # noqa: E402
from eat_sar.utils import ensure_dir, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--scan-csv", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--env-key", default="wind_bin")
    ap.add_argument("--env-value-key", default="wind_speed")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--selection", default="max_recall_at_p085",
                    help="selection criterion column in best_per_bin.csv "
                         "(or 'argmax_f1' for legacy F1)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    metric = cfg["threshold"]["selection_metric"]
    out_dir = ensure_dir(Path(cfg["paths"]["runs_dir"]) / args.tag)

    scan = pd.read_csv(args.scan_csv)
    mf = pd.read_csv(args.manifest)

    # ----- Piecewise: best threshold per bin -----
    # Read from the multi-criterion best_per_bin.csv produced by scan_thresholds.py
    best_csv = Path(args.scan_csv).with_name("best_per_bin.csv")
    if best_csv.exists() and "selection" in pd.read_csv(best_csv).columns:
        best_df = pd.read_csv(best_csv)
        per_bin = best_df[(best_df["selection"] == args.selection)
                          & (best_df["env_bin"] != "ALL")].sort_values("env_bin")
        if len(per_bin) == 0:
            raise ValueError(f"selection '{args.selection}' not found in {best_csv}; "
                             f"available: {sorted(best_df['selection'].unique())}")
        print(f"Using selection='{args.selection}' from {best_csv}")
    else:
        # legacy fallback: pick argmax of selection_metric on scan CSV
        per_bin = (scan[scan["env_bin"] != "ALL"]
                   .sort_values(metric, ascending=False)
                   .groupby("env_bin", as_index=False).head(1)
                   .sort_values("env_bin"))
        print(f"Using legacy argmax({metric}) on {args.scan_csv}")
    bin_to_T = dict(zip(per_bin["env_bin"], per_bin["threshold"]))

    # Determine bin order. Prefer the bins JSON written by compute_image_clutter.py
    # (which records the ACTUAL labels & numeric edges used at binning time).
    # Fall back to ERA5 wind/swh labels if running the original ERA5 pipeline.
    bins_json = Path(args.manifest).parent.parent.parent / "meta"
    # try common locations
    candidates = [
        Path(args.manifest).parent.parent / "meta" / f"{args.tag.split('_')[0]}_meta_bins.json",
        Path("cache/meta") / f"{args.tag.split('_')[0]}_meta_bins.json",
    ]
    bins_meta_path = next((p for p in candidates if p.exists()), None)
    if bins_meta_path is not None:
        bm = json.load(open(bins_meta_path))
        labels = bm["labels"]
        bins = bm["bins"]
        print(f"Using bins from {bins_meta_path}: labels={labels}, bins={bins}")
    else:
        # ERA5 fallback
        labels = (cfg["era5"]["wind_labels"] if args.env_key == "wind_bin"
                  else cfg["era5"]["swh_labels"])
        bins = (cfg["era5"]["wind_bins"] if args.env_key == "wind_bin"
                else cfg["era5"]["swh_bins"])
        print(f"No bins JSON found; using config labels={labels}")

    # Sanity check: bin_to_T should cover all labels
    missing = [lbl for lbl in labels if lbl not in bin_to_T]
    if missing:
        print(f"  [warn] selection criterion gave no result for labels {missing}; "
              f"will fall back to global ALL threshold for those")
        global_T = bin_to_T.get("ALL", 0.5)
        for lbl in missing:
            bin_to_T[lbl] = global_T

    thresholds_in_order = [bin_to_T[lbl] for lbl in labels]
    piecewise = PiecewiseT(bins=list(bins), thresholds=thresholds_in_order)

    # ----- Linear: anchor (mean env value, best_T) pairs -----
    env_means = mf.groupby(args.env_key)[args.env_value_key].mean().to_dict()
    print(f"  env_means per bin: {env_means}")
    e_arr = np.array([env_means[lbl] for lbl in labels if lbl in env_means])
    t_arr = np.array([bin_to_T[lbl] for lbl in labels if lbl in env_means])
    lin = fit_linear(e_arr, t_arr) if len(e_arr) >= 2 else LinearT(a=0.0, b=0.5)

    # ----- Save params -----
    params = {
        "env_key": args.env_key,
        "env_value_key": args.env_value_key,
        "labels": labels,
        "bins": list(bins),
        "piecewise": {"thresholds": thresholds_in_order, "bins": list(bins)},
        "linear": {"a": lin.a, "b": lin.b, "clip_min": lin.clip_min, "clip_max": lin.clip_max},
        "anchor_points": {"env": e_arr.tolist(), "best_T": t_arr.tolist()},
    }
    out_json = out_dir / "adaptive_T.json"
    json.dump(params, open(out_json, "w"), indent=2)
    print(f"Saved: {out_json}")
    print(json.dumps(params, indent=2))

    # ----- Visualisation -----
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    e_dense = np.linspace(0, max(20.0, e_arr.max() * 1.2 if len(e_arr) else 20), 200)
    ax.plot(e_dense, piecewise(e_dense), label="Piecewise T(e)", lw=2)
    ax.plot(e_dense, lin(e_dense), label=f"Linear T(e)={lin.a:.3f}e+{lin.b:.3f}", lw=2, ls="--")
    if len(e_arr):
        ax.scatter(e_arr, t_arr, c="k", zorder=5, label="Per-bin argmax")
    ax.set_xlabel(f"Environment value ({args.env_value_key})")
    ax.set_ylabel("Score threshold T(e)")
    ax.set_title(f"Adaptive threshold — {args.tag}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "adaptive_T.png", dpi=160)
    print(f"Saved: {out_dir / 'adaptive_T.png'}")


if __name__ == "__main__":
    main()
