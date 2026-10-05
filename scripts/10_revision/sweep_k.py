"""
Step R4 — Sensitivity to the box-dilation margin k (Reviewer 2).

For each k in the manifest's e_pred_k{k} columns (written by recompute_proxy.py):
  re-bin (tertiles on val) -> re-fit T_b on val -> evaluate on test.
Reports overall and high-bin ΔAP, the fitted T_b, and Spearman(e_k, e_kmain).

Usage:
  python scripts/10_revision/sweep_k.py \
      --val-manifest  $WORK/cache_preds/sarship_val/manifest_proxy.csv \
      --val-labels    $WORK/sarship_root/labels/val \
      --test-manifest $WORK/cache_preds/sarship_test/manifest_proxy.csv \
      --test-labels   $WORK/sarship_root/labels/test \
      --target-class 0 --k-main 3 --out $OUT/sarship_ksweep.json
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.append(str(Path(__file__).resolve().parent))
from revkit import (BIN_LABELS, apply_policy, assign_bins, dump_json,  # noqa: E402
                    evaluate, filter_gt, fit_per_bin_T, load_all, read_manifest, tertile_edges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--val-labels", required=True)
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--test-labels", required=True)
    ap.add_argument("--target-class", type=int, default=None)
    ap.add_argument("--baseline-T", type=float, default=0.40)
    ap.add_argument("--k-main", type=int, required=True)
    ap.add_argument("--require-gt", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    val = read_manifest(args.val_manifest); test = read_manifest(args.test_manifest)
    kcols = sorted([c for c in val.columns if re.fullmatch(r"e_pred_k\d+", c)],
                   key=lambda c: int(c[8:]))
    Rv = load_all(val, Path(args.val_labels), args.target_class, "val")
    Rt = load_all(test, Path(args.test_labels), args.target_class, "test")
    sv, _ = filter_gt(Rv, val["stem"].values, val[kcols[0]].values, args.require_gt)
    st, _ = filter_gt(Rt, test["stem"].values, test[kcols[0]].values, args.require_gt)
    val = val.set_index("stem").loc[sv].reset_index(); test = test.set_index("stem").loc[st].reset_index()
    mb = evaluate(Rt, st, args.baseline_T); base_all = mb["AP"]; base_f1 = mb["F1"]

    rows = []
    for c in kcols:
        k = int(c[8:])
        ev, et = val[c].values, test[c].values
        edges = tertile_edges(ev); bv = assign_bins(ev, edges)
        T_b = fit_per_bin_T(Rv, sv, bv, BIN_LABELS, criterion="f1")
        pol = apply_policy(et, st, edges, T_b, BIN_LABELS)
        m = evaluate(Rt, st, pol)
        bt = assign_bins(et, edges); hi = st[bt == "high"]
        d_hi = 100 * (evaluate(Rt, hi, pol)["AP"] - evaluate(Rt, hi, args.baseline_T)["AP"])
        rho = float(spearmanr(ev, val[f"e_pred_k{args.k_main}"].values).correlation)
        rows.append(dict(k=k, edge1=edges[1], edge2=edges[2], **{f"T_{lb}": T_b[lb] for lb in BIN_LABELS},
                         AP=m["AP"], F1=m["F1"], dAP_pp=100 * (m["AP"] - base_all), dF1_pp=100 * (m["F1"] - base_f1),
                         dAP_high_pp=d_hi, rho_vs_kmain=rho,
                         bgfrac_median=float(val[f"bgfrac_k{k}"].median()) if f"bgfrac_k{k}" in val else np.nan))
        print(f"k={k:2d} edges=({edges[1]:.2f},{edges[2]:.2f}) T={T_b} ΔAP={rows[-1]['dAP_pp']:+.2f}pp "
              f"ΔAP_high={d_hi:+.2f}pp ρ={rho:.3f}")
    df = pd.DataFrame(rows)
    dump_json(dict(k_main=args.k_main, baseline_AP=base_all, rows=rows,
                   dAP_range_pp=[float(df.dAP_pp.min()), float(df.dAP_pp.max())]), args.out)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
