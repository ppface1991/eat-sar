"""
Step R3 — K-fold cross-validation of the calibration (Reviewer 1, Q1).

Pool = VAL images. For each fold f (K=10, stratified by tertile so every fold
contains all clutter levels):
    * fit tertile edges + per-bin T_b on the other 9 folds
    * evaluate on the held-out fold  (in-pool generalisation)
    * evaluate on the full TEST set  (does the sampling of the calibration
      set change the test gain?)
Reports mean +- std of: edges, T_b, ΔAP on held-out fold, ΔAP on test.
Also reports how often the SIGN of (T_high - 0.40) flips across folds.

Usage:
  python scripts/10_revision/kfold_cv.py \
      --val-manifest  $WORK/cache_preds/sarship_val/manifest_proxy.csv \
      --val-labels    $WORK/sarship_root/labels/val \
      --test-manifest $WORK/cache_preds/sarship_test/manifest_proxy.csv \
      --test-labels   $WORK/sarship_root/labels/test \
      --target-class 0 --K 10 --out $OUT/sarship_kfold.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from revkit import (BIN_LABELS, apply_policy, assign_bins, dump_json,  # noqa: E402
                    evaluate, filter_gt, fit_per_bin_T, load_all, read_manifest, tertile_edges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--val-labels", required=True)
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--test-labels", required=True)
    ap.add_argument("--e-col", default="e_pred")
    ap.add_argument("--target-class", type=int, default=None)
    ap.add_argument("--baseline-T", type=float, default=0.40)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require-gt", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    val = read_manifest(args.val_manifest).dropna(subset=[args.e_col]).reset_index(drop=True)
    test = read_manifest(args.test_manifest).dropna(subset=[args.e_col])
    Rv = load_all(val, Path(args.val_labels), args.target_class, "val")
    Rt = load_all(test, Path(args.test_labels), args.target_class, "test")
    sv, ev = filter_gt(Rv, val["stem"].values, val[args.e_col].values, args.require_gt)
    st, et = filter_gt(Rt, test["stem"].values, test[args.e_col].values, args.require_gt)

    # stratified fold assignment (by full-pool tertile)
    rng = np.random.default_rng(args.seed)
    full_bins = assign_bins(ev, tertile_edges(ev))
    fold = np.empty(len(sv), int)
    for lb in BIN_LABELS:
        idx = np.where(full_bins == lb)[0]; rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % args.K

    mb = evaluate(Rt, st, args.baseline_T); base_test = mb["AP"]; base_f1 = mb["F1"]
    rows = []
    for f in range(args.K):
        tr, ho = fold != f, fold == f
        edges = tertile_edges(ev[tr])
        b_tr = assign_bins(ev[tr], edges)
        T_b = fit_per_bin_T(Rv, sv[tr], b_tr, BIN_LABELS, criterion="f1")
        # held-out fold
        pol_ho = apply_policy(ev[ho], sv[ho], edges, T_b, BIN_LABELS)
        m_ho = evaluate(Rv, sv[ho], pol_ho); m_ho_base = evaluate(Rv, sv[ho], args.baseline_T)
        # full test
        pol_te = apply_policy(et, st, edges, T_b, BIN_LABELS)
        m_te = evaluate(Rt, st, pol_te)
        # per-bin test ΔAP (high bin is what matters)
        b_te = assign_bins(et, edges)
        hi = st[b_te == "high"]
        d_hi = 100 * (evaluate(Rt, hi, pol_te)["AP"] - evaluate(Rt, hi, args.baseline_T)["AP"])
        rows.append(dict(fold=f, edge1=edges[1], edge2=edges[2],
                         T_low=T_b["low"], T_medium=T_b["medium"], T_high=T_b["high"],
                         dAP_heldout_pp=100 * (m_ho["AP"] - m_ho_base["AP"]),
                         dF1_heldout_pp=100 * (m_ho["F1"] - m_ho_base["F1"]),
                         AP_test=m_te["AP"], dAP_test_pp=100 * (m_te["AP"] - base_test),
                         F1_test=m_te["F1"], dF1_test_pp=100 * (m_te["F1"] - base_f1),
                         dAP_test_high_pp=d_hi))
        print(f"fold {f}: edges=({edges[1]:.2f},{edges[2]:.2f}) T={T_b} "
              f"ΔAP_ho={rows[-1]['dAP_heldout_pp']:+.2f}pp ΔAP_test={rows[-1]['dAP_test_pp']:+.2f}pp")

    df = pd.DataFrame(rows)
    summ = {c: dict(mean=float(df[c].mean()), std=float(df[c].std(ddof=1)),
                    min=float(df[c].min()), max=float(df[c].max()))
            for c in df.columns if c != "fold"}
    sign_hi = np.sign(df["T_high"] - args.baseline_T)
    out = dict(K=args.K, n_val=int(len(sv)), n_test=int(len(st)), baseline_T=args.baseline_T,
               baseline_AP_test=base_test, folds=rows, summary=summ,
               T_high_above_baseline_frac=float((sign_hi > 0).mean()),
               T_high_sign_consistent=bool(len(set(sign_hi[sign_hi != 0])) <= 1),
               dAP_test_all_positive=bool((df["dAP_test_pp"] > 0).all()),
               dF1_test_all_positive=bool((df["dF1_test_pp"] > 0).all()))
    dump_json(out, args.out)
    print("\nSUMMARY (mean ± std):")
    for c, s in summ.items():
        print(f"  {c:18s} {s['mean']:+.4f} ± {s['std']:.4f}   [{s['min']:.4f}, {s['max']:.4f}]")


if __name__ == "__main__":
    main()
