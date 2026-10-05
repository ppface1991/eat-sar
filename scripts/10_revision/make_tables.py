"""
Step R6 — Collect all revision JSONs into (a) one compact numbers.json and
(b) ready-to-paste LaTeX table rows.

Usage:
  python scripts/10_revision/make_tables.py --out-dir $OUT
Expects in $OUT:  {sardet,sarship}_main.json, {sardet,sarship}_kfold.json,
                  {sardet,sarship}_ksweep.json, {sardet,sarship}_val_proxy_report.json,
                  transfer_sardet_to_sarship.json, transfer_sarship_to_sardet.json
Missing files are skipped with a warning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DS = {"sardet": "SARDet-100K", "sarship": "SAR-Ship-Dataset"}
LB = {"ALL": "All", "low": "Low-clutter", "medium": "Medium-clutter", "high": "High-clutter"}


def _load(p: Path):
    if not p.exists():
        print(f"[warn] missing {p}"); return None
    return json.load(open(p))


def f3(x): return f"{x:.3f}"
def pp(x): return f"{'+' if x >= 0 else '$-$'}{abs(x):.2f}"


def table1_rows(ds, main):
    """Table I: baseline vs adaptive-F1 (piecewise), with T_b column."""
    rows = []
    recs = {(r["strategy"], r["bin"]): r for r in main["tables"]["adaptive_f1"]}
    for strat, label in (("baseline", f"Baseline ($T{{=}}{main['baseline_T']:.2f}$)"), ("adaptive", "Adaptive F1 (ours)")):
        for i, b in enumerate(["ALL", "low", "medium", "high"]):
            r = recs[(strat, b)]
            T = f"{main['baseline_T']:.2f}" if strat == "baseline" else ("--" if b == "ALL" else f"{main['T_f1'][b]:.3f}")
            d = "--" if strat == "baseline" else pp(r["dAP_pp"])
            dF1 = "--" if strat == "baseline" else pp(100 * (r["F1"] - recs[('baseline', b)]["F1"]))
            first = (DS[ds] if (strat == "baseline" and i == 0) else "")
            lab = label if i == 0 else ""
            rows.append(f"{first:18s} & {lab:28s} & {LB[b]:15s} & {T:>5s} & {f3(r['P'])} & {f3(r['R'])} & {f3(r['F1'])} & {dF1:>7s} & {f3(r['AP'])} & {d:>7s} \\\\")
        rows.append("\\addlinespace[2pt]")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    O = Path(args.out_dir)
    num = {}
    tex = []

    tex.append("% ===== Table I rows (cols: Dataset & Strategy & Bin & T_b & P & R & F1 & dF1 & AP & dAP) =====")
    for ds in ("sardet", "sarship"):
        m = _load(O / f"{ds}_main.json")
        if not m:
            continue
        tex += table1_rows(ds, m)
        tex.append("\\midrule")
        num[ds] = dict(
            n_val=m["n_val"], n_test=m["n_test"], edges=m["edges"][1:3], bin_sizes_val=m["bin_sizes_val"],
            T_f1=m["T_f1"], T_recall=m["T_recall"], T_global_valopt=m["T_global_valopt"],
            T_f1_minus_baseline=m["T_f1_minus_baseline"], linear=m["linear"],
            ungated_AP=m["ungated_AP_test"], baseline_AP=m["baseline_gated_AP_test"], adaptive_AP=m["adaptive_f1_gated_AP_test"],
            control_global_valopt=m["control_global_valopt"], control_random_bins=m["control_random_bins"],
            control_matched_global=m.get("control_matched_global"), oracle=m.get("oracle_test_fit"),
            adaptive_f1_rows={r["bin"]: r for r in m["tables"]["adaptive_f1"] if r["strategy"] == "adaptive"},
            baseline_rows={r["bin"]: r for r in m["tables"]["adaptive_f1"] if r["strategy"] == "baseline"},
            linear_rows={r["bin"]: r for r in m["tables"]["adaptive_linear"] if r["strategy"] == "adaptive"},
            recall_rows={r["bin"]: r for r in m["tables"]["adaptive_recall"] if r["strategy"] == "adaptive"},
        )
        k = _load(O / f"{ds}_kfold.json")
        if k:
            s = k["summary"]
            num[ds]["kfold"] = dict(K=k["K"], **{c: (round(s[c]["mean"], 4), round(s[c]["std"], 4)) for c in s},
                                    T_high_sign_consistent=k["T_high_sign_consistent"],
                                    dAP_test_all_positive=k["dAP_test_all_positive"],
                                    dF1_test_all_positive=k.get("dF1_test_all_positive"))
        sw = _load(O / f"{ds}_ksweep.json")
        if sw:
            num[ds]["ksweep"] = {r["k"]: dict(dAP_pp=round(r["dAP_pp"], 2), dF1_pp=round(r.get("dF1_pp", float("nan")), 2),
                                             dAP_high_pp=round(r["dAP_high_pp"], 2), rho=round(r["rho_vs_kmain"], 3),
                                             T=[r["T_low"], r["T_medium"], r["T_high"]]) for r in sw["rows"]}
        pr = _load(O / f"{ds}_val_proxy_report.json")
        if pr:
            num[ds]["proxy"] = dict(spearman=pr["spearman_vs_e_pred"], tertile_agreement=pr["tertile_agreement_pred_vs_gt"],
                                    per_bin=pr["per_bin"], quantiles=pr["quantiles_e_pred"], mask_gate=pr["mask_gate"])

    # ---- Table (sensitivity): K-fold + k sweep compact rows
    tex.append("\n% ===== Robustness table rows =====")
    for ds in ("sardet", "sarship"):
        if ds not in num:
            continue
        n = num[ds]
        if "kfold" in n:
            kf = n["kfold"]
            tex.append(f"% {DS[ds]} 10-fold: T_low {kf['T_low'][0]:.3f}±{kf['T_low'][1]:.3f}, "
                       f"T_med {kf['T_medium'][0]:.3f}±{kf['T_medium'][1]:.3f}, T_high {kf['T_high'][0]:.3f}±{kf['T_high'][1]:.3f}; "
                       f"ΔAP_test {kf['dAP_test_pp'][0]:+.2f}±{kf['dAP_test_pp'][1]:.2f} pp; ΔF1_test {kf['dF1_test_pp'][0]:+.2f}±{kf['dF1_test_pp'][1]:.2f} pp")
        if "ksweep" in n:
            ks = n["ksweep"]
            tex.append(f"% {DS[ds]} k-sweep ΔAP(pp): " + ", ".join(f"k={k}:{v['dAP_pp']:+.2f}" for k, v in ks.items()))
            tex.append(f"% {DS[ds]} k-sweep ΔF1(pp): " + ", ".join(f"k={k}:{v['dF1_pp']:+.2f}" for k, v in ks.items()))

    for name in ("transfer_sardet_to_sarship", "transfer_sarship_to_sardet"):
        t = _load(O / f"{name}.json")
        if t:
            num[name] = {k: dict(AP=round(v["AP"], 4), F1=round(v["F1"], 4), dAP_pp=round(v["dAP_pp"], 2),
                                 occ=v["bin_occupancy"]) for k, v in t["results"].items()}
            tex.append(f"% {name}: " + "; ".join(f"{k}: AP {v['AP']:.3f} ({v['dAP_pp']:+.2f}pp) F1 {v['F1']:.3f}" for k, v in num[name].items()))

    (O / "numbers.json").write_text(json.dumps(num, indent=2, default=str))
    (O / "table_rows.tex").write_text("\n".join(tex))
    print("\n".join(tex))
    print(f"\n[saved] {O/'numbers.json'}  {O/'table_rows.tex'}")
    print("\n>>> Send numbers.json and table_rows.tex back to the assistant.")


if __name__ == "__main__":
    main()
