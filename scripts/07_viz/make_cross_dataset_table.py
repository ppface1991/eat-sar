"""
Make the GRSL main paper table: cross-dataset comparison.

Rows: dataset x env_bin
Cols: P, R, F1, mAP, dmAP (baseline + adaptive_f1 + adaptive_recall)

Usage:
    python scripts/07_viz/make_cross_dataset_table.py \\
        --sardet-f1 runs/sardet100k_test_f1/final_eval.csv \\
        --sardet-recall runs/sardet100k_test/final_eval.csv \\
        --sarship-f1 runs/sarship_test_f1/final_eval.csv \\
        --sarship-recall runs/sarship_test_recall/final_eval.csv \\
        --out runs/cross_dataset/table_main.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ENV_ORDER = ["ALL", "calm", "moderate", "rough"]
ENV_LABEL = {"ALL": "All", "calm": "Calm", "moderate": "Moderate", "rough": "Rough"}


def load_block(csv: str, scheme: str = "adaptive_piecewise") -> pd.DataFrame:
    df = pd.read_csv(csv)
    return df[df["scheme"] == scheme].set_index("env_bin")


def fmt(val, best, lower=False):
    s = f"{val:.3f}"
    is_best = val <= best + 1e-6 if lower else val >= best - 1e-6
    return r"\textbf{" + s + "}" if is_best else s


def fmt_dlt(v, base):
    d = (v - base) * 100
    return f"{'+' if d >= 0 else ''}{d:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sardet-f1", required=True)
    ap.add_argument("--sardet-recall", required=True)
    ap.add_argument("--sarship-f1", required=True)
    ap.add_argument("--sarship-recall", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    datasets = [
        ("SARDet-100K", args.sardet_f1, args.sardet_recall),
        ("SAR-Ship-Dataset", args.sarship_f1, args.sarship_recall),
    ]

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Adaptive environment-conditioned thresholding "
                 r"vs.\ global-threshold baseline on two SAR ship detection "
                 r"benchmarks. For each dataset and sea-state bin we report "
                 r"precision (P), recall (R), F1, mAP@0.5, and the absolute "
                 r"$\Delta$mAP@0.5 against the baseline. F1-balanced selects "
                 r"$T_b=\arg\max_T F1$ per bin; Recall-biased selects "
                 r"$T_b=\arg\max_T R$ subject to $P\geq 0.85$.}")
    lines.append(r"\label{tab:main_cross}")
    lines.append(r"\begin{tabular}{l l l c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Strategy & Bin & P & R & F1 & mAP & $\Delta$mAP \\")
    lines.append(r"\midrule")

    for ds_name, f1_csv, rec_csv in datasets:
        df_f1 = pd.read_csv(f1_csv)
        df_rc = pd.read_csv(rec_csv)
        base = df_f1[df_f1["scheme"] == "baseline"].set_index("env_bin")
        adp_f1 = df_f1[df_f1["scheme"] == "adaptive_piecewise"].set_index("env_bin")
        adp_rc = df_rc[df_rc["scheme"] == "adaptive_piecewise"].set_index("env_bin")

        # Determine best per (env, metric) across the three strategies
        best_f1, best_map = {}, {}
        for env in ENV_ORDER:
            best_f1[env] = max(base.loc[env, "f1"], adp_f1.loc[env, "f1"], adp_rc.loc[env, "f1"])
            best_map[env] = max(base.loc[env, "map50"], adp_f1.loc[env, "map50"], adp_rc.loc[env, "map50"])

        first_ds = True
        for strategy_name, df_idx, is_base in [
            ("Baseline (T=0.40)", base, True),
            ("Adaptive F1", adp_f1, False),
            ("Adaptive Recall", adp_rc, False),
        ]:
            first_strat = True
            for env in ENV_ORDER:
                r = df_idx.loc[env]
                ds_cell = ds_name if first_ds else ""
                first_ds = False
                strat_cell = strategy_name if first_strat else ""
                first_strat = False
                dmap = "--" if is_base else fmt_dlt(r["map50"], base.loc[env, "map50"])
                f1s = fmt(r["f1"], best_f1[env])
                maps = fmt(r["map50"], best_map[env])
                lines.append(
                    f"{ds_cell} & {strat_cell} & {ENV_LABEL[env]} & "
                    f"{r['precision']:.3f} & {r['recall']:.3f} & {f1s} & {maps} & {dmap} \\\\"
                )
            lines.append(r"\addlinespace[2pt]")
        # Remove final addlinespace; add a midrule between datasets
        if lines[-1].strip().startswith(r"\addlinespace"):
            lines.pop()
        lines.append(r"\midrule")

    # remove the trailing midrule and replace with bottomrule
    if lines[-1].strip() == r"\midrule":
        lines.pop()
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out_path}")
    print("\n----- preview -----")
    print(out_path.read_text())


if __name__ == "__main__":
    main()
