"""
Produce paper-ready LaTeX tables from final_eval.csv.

Two modes:

(A) Single eval (legacy):
    python scripts/07_viz/make_paper_tables.py \\
        --eval-csv runs/sardet100k_test/final_eval.csv \\
        --out-dir runs/sardet100k_test

(B) Combined main + ablation (recommended for GRSL paper):
    python scripts/07_viz/make_paper_tables.py \\
        --eval-csv runs/sardet100k_test_f1/final_eval.csv \\
        --eval-csv-ablation runs/sardet100k_test/final_eval.csv \\
        --baseline-T 0.40 \\
        --out-dir runs/sardet100k_paper

Outputs (mode B):
    table_main.tex        — Overall + per-env in one compact table, bold best, Δ vs baseline
    table_overall.tex     — Just the overall block (back-compat)
    table_per_env.tex     — Just the per-env block (back-compat)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCHEME_NAMES = {
    "baseline": "Baseline (T=0.40)",
    "adaptive_piecewise": "Adaptive (Piecewise)",
    "adaptive_linear": "Adaptive (Linear)",
}

ENV_ORDER = ["calm", "moderate", "rough"]
ENV_ALIAS = {"calm": "Calm", "moderate": "Moderate", "rough": "Rough"}


def fmt(val, best, lower_is_better=False):
    """Format a number to 3 dp, bold if it equals the best."""
    s = f"{val:.3f}"
    is_best = (val <= best + 1e-6) if lower_is_better else (val >= best - 1e-6)
    return r"\textbf{" + s + "}" if is_best else s


def fmt_delta(val, base):
    d = (val - base) * 100  # percentage points
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.2f}"


def overall_block(df: pd.DataFrame) -> pd.DataFrame:
    o = df[df["env_bin"] == "ALL"].copy()
    o["Scheme"] = o["scheme"].map(SCHEME_NAMES).fillna(o["scheme"])
    return o[["Scheme", "precision", "recall", "f1", "map50"]].reset_index(drop=True)


def per_env_block(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sch in ["baseline", "adaptive_piecewise", "adaptive_linear"]:
        if sch not in df["scheme"].unique():
            continue
        for b in ENV_ORDER:
            sub = df[(df["scheme"] == sch) & (df["env_bin"] == b)]
            if len(sub) == 0:
                continue
            r = sub.iloc[0]
            rows.append({
                "Scheme": SCHEME_NAMES.get(sch, sch),
                "Env": ENV_ALIAS.get(b, b),
                "precision": r["precision"], "recall": r["recall"],
                "f1": r["f1"], "map50": r["map50"],
            })
    return pd.DataFrame(rows)


def render_overall_latex(o: pd.DataFrame, caption: str, label: str) -> str:
    best_f1 = o["f1"].max()
    best_map = o["map50"].max()
    base_f1 = o.loc[o["Scheme"].str.startswith("Baseline"), "f1"].iloc[0]
    base_map = o.loc[o["Scheme"].str.startswith("Baseline"), "map50"].iloc[0]
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Scheme & P & R & F1 & $\Delta$F1 & mAP & $\Delta$mAP \\")
    lines.append(r"\midrule")
    for _, r in o.iterrows():
        d_f1 = "--" if r["Scheme"].startswith("Baseline") else fmt_delta(r["f1"], base_f1)
        d_map = "--" if r["Scheme"].startswith("Baseline") else fmt_delta(r["map50"], base_map)
        lines.append(
            f"{r['Scheme']} & {r['precision']:.3f} & {r['recall']:.3f} & "
            f"{fmt(r['f1'], best_f1)} & {d_f1} & "
            f"{fmt(r['map50'], best_map)} & {d_map} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def render_per_env_latex(p: pd.DataFrame, caption: str, label: str) -> str:
    # bold best per (Env, metric) across schemes
    best = p.groupby("Env")[["f1", "map50"]].max().to_dict()
    base = p[p["Scheme"].str.startswith("Baseline")].set_index("Env")[["f1", "map50"]]
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"Scheme & Env & P & R & F1 & mAP & $\Delta$mAP \\")
    lines.append(r"\midrule")
    last_sch = None
    for _, r in p.iterrows():
        sch_cell = r["Scheme"] if r["Scheme"] != last_sch else ""
        last_sch = r["Scheme"]
        is_base = r["Scheme"].startswith("Baseline")
        d_map = "--" if is_base else fmt_delta(r["map50"], base.loc[r["Env"], "map50"])
        f1_str = fmt(r["f1"], best["f1"][r["Env"]])
        map_str = fmt(r["map50"], best["map50"][r["Env"]])
        lines.append(
            f"{sch_cell} & {r['Env']} & {r['precision']:.3f} & {r['recall']:.3f} & "
            f"{f1_str} & {map_str} & {d_map} \\\\"
        )
        if r["Env"] == "Rough":
            lines.append(r"\midrule" if not is_base else r"\addlinespace")
    # remove trailing rule
    if lines[-1].strip() in (r"\midrule", r"\addlinespace"):
        lines.pop()
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def render_combined_main(df_main: pd.DataFrame, df_abl: pd.DataFrame,
                          caption: str, label: str) -> str:
    """One compact table: rows = (Strategy, Env), cols = P, R, F1, mAP, dmAP.
    Strategies: Baseline / Adaptive-F1 / Adaptive-Recall."""
    # Baseline (same in both eval files) -> take from df_main
    base_overall = df_main[(df_main["scheme"] == "baseline") & (df_main["env_bin"] == "ALL")].iloc[0]
    rows = []

    def add(strategy_label, sub_df, scheme="adaptive_piecewise"):
        # overall
        o = sub_df[(sub_df["scheme"] == scheme) & (sub_df["env_bin"] == "ALL")].iloc[0]
        rows.append({"Strategy": strategy_label, "Env": "ALL",
                     **{k: o[k] for k in ["precision", "recall", "f1", "map50"]}})
        for b in ENV_ORDER:
            r = sub_df[(sub_df["scheme"] == scheme) & (sub_df["env_bin"] == b)].iloc[0]
            rows.append({"Strategy": "", "Env": ENV_ALIAS[b],
                         **{k: r[k] for k in ["precision", "recall", "f1", "map50"]}})

    # Baseline rows
    rows.append({"Strategy": "Baseline (T=0.40)", "Env": "ALL",
                 **{k: base_overall[k] for k in ["precision", "recall", "f1", "map50"]}})
    for b in ENV_ORDER:
        r = df_main[(df_main["scheme"] == "baseline") & (df_main["env_bin"] == b)].iloc[0]
        rows.append({"Strategy": "", "Env": ENV_ALIAS[b],
                     **{k: r[k] for k in ["precision", "recall", "f1", "map50"]}})

    add("Adaptive (F1-balanced)", df_main, "adaptive_piecewise")
    add("Adaptive (Recall-biased)", df_abl, "adaptive_piecewise")

    tbl = pd.DataFrame(rows)
    # ΔmAP vs baseline (within same Env)
    base_map = {r["Env"]: r["map50"] for _, r in tbl[tbl["Strategy"].str.startswith("Baseline") | (tbl["Strategy"] == "")].iloc[:4].iterrows()}
    # Build base_map properly using first 4 baseline rows
    base_block = tbl.iloc[0:4].set_index("Env")
    base_map = base_block["map50"].to_dict()

    # bold-best
    best_per_env = tbl.groupby("Env")[["f1", "map50"]].max().to_dict()

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"Strategy & Env & P & R & F1 & mAP & $\Delta$mAP \\")
    lines.append(r"\midrule")
    for _, r in tbl.iterrows():
        is_base = r["Strategy"].startswith("Baseline") or (r["Strategy"] == "" and len(rows) and tbl.iloc[0]["Strategy"].startswith("Baseline"))
        d_map = "--" if r["Env"] in base_map and abs(r["map50"] - base_map[r["Env"]]) < 1e-9 else fmt_delta(r["map50"], base_map[r["Env"]])
        f1_s = fmt(r["f1"], best_per_env["f1"][r["Env"]])
        map_s = fmt(r["map50"], best_per_env["map50"][r["Env"]])
        lines.append(
            f"{r['Strategy']} & {r['Env']} & {r['precision']:.3f} & {r['recall']:.3f} & "
            f"{f1_s} & {map_s} & {d_map} \\\\"
        )
        if r["Env"] == "Rough":
            lines.append(r"\addlinespace")
    if lines[-1].strip() == r"\addlinespace":
        lines.pop()
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-csv", required=True, help="Main final_eval.csv")
    ap.add_argument("--eval-csv-ablation", default=None,
                    help="Optional second final_eval.csv (e.g., recall-biased) "
                         "to combine into one paper table.")
    ap.add_argument("--baseline-T", type=float, default=0.40)
    ap.add_argument("--dataset-name", default="SARDet-100K ship subset",
                    help="Dataset name shown in the table caption.")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.eval_csv)

    # Back-compat: always write the simple overall + per-env tables
    overall = overall_block(df)
    overall_disp = overall.copy()
    overall_disp.columns = ["Scheme", "Precision", "Recall", "F1", "mAP@0.5"]
    (out_dir / "table_overall.tex").write_text(
        render_overall_latex(overall, "Overall detection performance on the test set.",
                              "tab:overall")
    )
    print("== Overall ==")
    print(overall_disp.to_string(index=False))

    per_env = per_env_block(df)
    (out_dir / "table_per_env.tex").write_text(
        render_per_env_latex(per_env,
                              "Detection performance stratified by environment (sea-state proxy).",
                              "tab:per_env")
    )
    print("\n== Per Env ==")
    print(per_env.to_string(index=False))

    # Combined main + ablation
    if args.eval_csv_ablation:
        df_abl = pd.read_csv(args.eval_csv_ablation)
        ds_name = args.dataset_name
        combined = render_combined_main(
            df, df_abl,
            caption=(r"Detection performance on the " + ds_name + r". "
                     r"Two adaptive policies are derived from the same per-bin "
                     r"threshold scan but with different selection criteria: "
                     r"F1-balanced (argmax F1 per bin) and Recall-biased "
                     r"(max recall s.t. P$\geq$0.85). Baseline uses a single "
                     r"global threshold T=0.40."),
            label="tab:adaptive_main_" + ds_name.split()[0].lower().replace("-", ""),
        )
        (out_dir / "table_main.tex").write_text(combined)
        print(f"\nSaved combined main table: {out_dir / 'table_main.tex'}")
        print("\n----- table_main.tex preview -----")
        print(combined)


if __name__ == "__main__":
    main()
