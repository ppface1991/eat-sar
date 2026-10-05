"""
Step R2 — Main result regeneration + control baselines (one dataset).

Calibrate on VAL (proxy manifest from recompute_proxy.py), evaluate on TEST.

Outputs (JSON):
  * bin edges (tertiles of e_pred on val), per-bin T_b (F1-balanced and
    recall-biased), linear fit (a, b) and anchor points           -> R1-Q3
  * Table-I style stratified rows: baseline T=0.40, adaptive-F1 piecewise,
    adaptive-F1 linear, adaptive-recall                            -> Table I
  * sign of (T_b - 0.40) per bin                                   -> R2
  * CONTROL A: val-optimal GLOBAL threshold T*_global              -> R2 "not just val search"
  * CONTROL B: random-bin placebo (same bin sizes, N permutations) -> R2
  * ungated (detector-only) AP on test                             -> R2 metric clarity
  * AP computed two ways on the adaptive list: gated-AP (ranked by original
    score) vs single-point P/R at T(e)                             -> R2

Usage:
  python scripts/10_revision/run_main_eval.py \
      --val-manifest  $WORK/cache_preds/sarship_val/manifest_proxy.csv \
      --val-labels    $WORK/sarship_root/labels/val \
      --test-manifest $WORK/cache_preds/sarship_test/manifest_proxy.csv \
      --test-labels   $WORK/sarship_root/labels/test \
      --target-class 0 --baseline-T 0.40 --n-perm 20 \
      --out $OUT/sarship_main.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from revkit import (BIN_LABELS, apply_policy, assign_bins, dump_json,  # noqa: E402
                    evaluate, filter_gt, fit_global_T, fit_per_bin_T, linear_fit,
                    load_all, read_manifest, stratified_eval, tertile_edges, ungated_ap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--val-labels", required=True)
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--test-labels", required=True)
    ap.add_argument("--e-col", default="e_pred")
    ap.add_argument("--target-class", type=int, default=None)
    ap.add_argument("--baseline-T", type=float, default=0.40)
    ap.add_argument("--min-precision", type=float, default=0.85)
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require-gt", action="store_true", help="keep only images with >=1 GT ship (ship subset)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    val = read_manifest(args.val_manifest).dropna(subset=[args.e_col])
    test = read_manifest(args.test_manifest).dropna(subset=[args.e_col])
    Rv = load_all(val, Path(args.val_labels), args.target_class, "val")
    Rt = load_all(test, Path(args.test_labels), args.target_class, "test")
    sv, ev = filter_gt(Rv, val["stem"].values, val[args.e_col].values, args.require_gt)
    st, et = filter_gt(Rt, test["stem"].values, test[args.e_col].values, args.require_gt)

    # ---- bins + calibration on val
    edges = tertile_edges(ev)
    bv = assign_bins(ev, edges)
    T_f1 = fit_per_bin_T(Rv, sv, bv, BIN_LABELS, criterion="f1")
    T_rc = fit_per_bin_T(Rv, sv, bv, BIN_LABELS, criterion="recall",
                         min_precision=args.min_precision)
    a, b, anchors = linear_fit(edges, ev, bv, T_f1, BIN_LABELS)
    T_glob = fit_global_T(Rv, sv)

    out = dict(
        val_manifest=args.val_manifest, test_manifest=args.test_manifest,
        n_val=int(len(sv)), n_test=int(len(st)), e_col=args.e_col, require_gt=args.require_gt,
        edges=edges, bin_sizes_val={lb: int((bv == lb).sum()) for lb in BIN_LABELS},
        T_f1=T_f1, T_recall=T_rc, T_global_valopt=T_glob,
        T_f1_minus_baseline={lb: round(T_f1[lb] - args.baseline_T, 3) for lb in BIN_LABELS},
        linear=dict(a=a, b=b, anchors_e_median=anchors,
                    T_at_anchor=[float(np.clip(a * x + b, 0.05, 0.9)) for x in anchors]),
        baseline_T=args.baseline_T,
    )

    # ---- test evaluation
    pol_f1 = apply_policy(et, st, edges, T_f1, BIN_LABELS)
    pol_lin = apply_policy(et, st, edges, T_f1, BIN_LABELS, linear=(a, b))
    pol_rc = apply_policy(et, st, edges, T_rc, BIN_LABELS)
    tables = {}
    for name, pol in (("adaptive_f1", pol_f1), ("adaptive_linear", pol_lin),
                      ("adaptive_recall", pol_rc)):
        df = stratified_eval(Rt, st, et, edges, BIN_LABELS, pol, args.baseline_T)
        tables[name] = df.to_dict(orient="records")
    out["tables"] = tables

    # ---- metric clarity
    out["ungated_AP_test"] = ungated_ap(Rt, st)
    out["baseline_gated_AP_test"] = evaluate(Rt, st, args.baseline_T)["AP"]
    out["adaptive_f1_gated_AP_test"] = evaluate(Rt, st, pol_f1)["AP"]

    # ---- CONTROL A: val-optimal global threshold
    m_glob = evaluate(Rt, st, T_glob)
    out["control_global_valopt"] = dict(T=T_glob, **m_glob,
                                        dAP_pp_vs_baseline=100 * (m_glob["AP"] - out["baseline_gated_AP_test"]),
                                        dAP_pp_adaptive_vs_global=100 * (out["adaptive_f1_gated_AP_test"] - m_glob["AP"]))

    # ---- CONTROL B: random-bin placebo
    rng = np.random.default_rng(args.seed)
    placebo, placebo_f1 = [], []
    for _ in range(args.n_perm):
        perm_v = rng.permutation(bv)             # same bin sizes, random membership
        T_rand = fit_per_bin_T(Rv, sv, perm_v, BIN_LABELS, criterion="f1")
        perm_t = rng.permutation(assign_bins(et, edges))
        pol = {s: T_rand[bb] for s, bb in zip(st, perm_t)}
        m_pl = evaluate(Rt, st, pol)
        placebo.append(m_pl["AP"]); placebo_f1.append(m_pl["F1"])
    placebo = np.array(placebo); placebo_f1 = np.array(placebo_f1)
    f1_ad = evaluate(Rt, st, pol_f1)["F1"]
    out["control_random_bins"] = dict(
        n_perm=args.n_perm, AP_mean=float(placebo.mean()), AP_std=float(placebo.std()),
        dAP_pp_mean=float(100 * (placebo.mean() - out["baseline_gated_AP_test"])),
        dAP_pp_adaptive=float(100 * (out["adaptive_f1_gated_AP_test"] - placebo.mean())),
        z_score=float((out["adaptive_f1_gated_AP_test"] - placebo.mean()) / max(placebo.std(), 1e-9)),
        F1_mean=float(placebo_f1.mean()), F1_std=float(placebo_f1.std()),
        dF1_pp_adaptive=float(100 * (f1_ad - placebo_f1.mean())),
        z_score_F1=float((f1_ad - placebo_f1.mean()) / max(placebo_f1.std(), 1e-9)))

    # ---- CONTROL C: matched-operating-point global thresholds (isolates
    #      *conditioning* from the *level* of the threshold)
    m_ad = evaluate(Rt, st, pol_f1)
    scan = np.round(np.arange(0.05, 0.95, 0.005), 3)
    glob = [(float(T), evaluate(Rt, st, float(T))) for T in scan]
    T_mr, m_mr = min(glob, key=lambda x: abs(x[1]["R"] - m_ad["R"]))      # same recall
    T_mp, m_mp = min(glob, key=lambda x: abs(x[1]["P"] - m_ad["P"]))      # same precision
    T_mn, m_mn = min(glob, key=lambda x: abs(x[1]["n_pred"] - m_ad["n_pred"]))  # same #detections
    out["control_matched_global"] = dict(
        adaptive=m_ad,
        matched_recall=dict(T=T_mr, **m_mr, dP_pp=100 * (m_ad["P"] - m_mr["P"]), dF1_pp=100 * (m_ad["F1"] - m_mr["F1"])),
        matched_precision=dict(T=T_mp, **m_mp, dR_pp=100 * (m_ad["R"] - m_mp["R"]), dF1_pp=100 * (m_ad["F1"] - m_mp["F1"])),
        matched_ndet=dict(T=T_mn, **m_mn, dF1_pp=100 * (m_ad["F1"] - m_mn["F1"]), dAP_pp=100 * (m_ad["AP"] - m_mn["AP"])))

    # ---- ORACLE: per-bin T fit directly on TEST (upper bound of what per-bin
    #      thresholding can deliver; shows how much the val-fit captures)
    bt = assign_bins(et, edges)
    T_or = fit_per_bin_T(Rt, st, bt, BIN_LABELS, criterion="f1")
    m_or = evaluate(Rt, st, apply_policy(et, st, edges, T_or, BIN_LABELS))
    out["oracle_test_fit"] = dict(T=T_or, **m_or,
                                  F1_gap_pp=100 * (m_or["F1"] - m_ad["F1"]),
                                  dF1_pp_vs_baseline=100 * (m_or["F1"] - evaluate(Rt, st, args.baseline_T)["F1"]))

    dump_json(out, args.out)
    print("\nedges:", edges)
    print("T_f1:", T_f1, " T_recall:", T_rc, " T*_global(val):", T_glob)
    print(f"linear: T = {a:.5f}*e + {b:.4f}")
    print(pd.DataFrame(tables["adaptive_f1"]).round(4).to_string(index=False))
    print("ungated AP:", round(out["ungated_AP_test"], 4),
          "| baseline gated:", round(out["baseline_gated_AP_test"], 4),
          "| adaptive gated:", round(out["adaptive_f1_gated_AP_test"], 4))
    print("CONTROL A (val-opt global):", {k: round(v, 4) if isinstance(v, float) else v
                                          for k, v in out["control_global_valopt"].items()})
    print("CONTROL B (random bins):", {k: round(v, 4) if isinstance(v, float) else v
                                       for k, v in out["control_random_bins"].items()})
    cm = out["control_matched_global"]
    print(f"CONTROL C adaptive P/R/F1/AP = {m_ad['P']:.4f}/{m_ad['R']:.4f}/{m_ad['F1']:.4f}/{m_ad['AP']:.4f} (n={m_ad['n_pred']})")
    print(f"   global@same-recall    T={cm['matched_recall']['T']:.3f}  P={cm['matched_recall']['P']:.4f}  ΔP={cm['matched_recall']['dP_pp']:+.2f}pp ΔF1={cm['matched_recall']['dF1_pp']:+.2f}pp")
    print(f"   global@same-precision T={cm['matched_precision']['T']:.3f}  R={cm['matched_precision']['R']:.4f}  ΔR={cm['matched_precision']['dR_pp']:+.2f}pp ΔF1={cm['matched_precision']['dF1_pp']:+.2f}pp")
    print(f"   global@same-#det      T={cm['matched_ndet']['T']:.3f}  F1={cm['matched_ndet']['F1']:.4f}  ΔF1={cm['matched_ndet']['dF1_pp']:+.2f}pp ΔAP={cm['matched_ndet']['dAP_pp']:+.2f}pp")
    print("ORACLE (per-bin T fit on test):", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["oracle_test_fit"].items()})


if __name__ == "__main__":
    main()
