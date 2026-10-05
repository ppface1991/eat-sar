"""
Build Table II — Method comparison.

Compares (per dataset):
  - Global baseline (T=0.40)
  - Soft-NMS (sigma=0.5)
  - TENT (BN affine adaptation)
  - Ours (Adaptive F1-balanced)
On four axes:
  - mAP@0.5 (ALL + Rough)
  - FPS (ms/img)
  - Need retraining? (Y/N)
  - Extra parameters / external data

Inputs:
  --baseline-csv   /autodl-fs/data/eat-sar/runs/sardet100k_test/final_eval.csv
  --softnms-csv    /autodl-fs/data/eat-sar/runs/sardet100k_test_softnms/final_eval.csv
  --tent-csv       /autodl-fs/data/eat-sar/runs/sardet100k_test_tent/final_eval.csv
  --adaptive-csv   /autodl-fs/data/eat-sar/runs/sardet100k_test_f1/final_eval.csv
  --softnms-timing /autodl-fs/data/eat-sar/runs/sardet100k_test_softnms/timing.csv
  --baseline-fps   <float>   (you measure once; same for adaptive since same forward pass)
  --tent-fps       <float>   (post-adaptation inference FPS)
  --dataset-label  SARDet-100K   (or SAR-Ship)
  --out            paper/table2_sardet.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def get_map(df, env_bin, scheme=None):
    sub = df[df.env_bin == env_bin]
    if scheme is not None and "scheme" in df.columns:
        sub = sub[sub.scheme == scheme]
    if len(sub) == 0:
        return float("nan")
    return float(sub["map50"].iloc[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-csv", required=True)
    ap.add_argument("--softnms-csv", required=True)
    ap.add_argument("--tent-csv", required=True)
    ap.add_argument("--adaptive-csv", required=True)
    ap.add_argument("--baseline-fps", type=float, required=True,
                    help="Detector forward-pass FPS (img/s)")
    ap.add_argument("--softnms-fps", type=float, default=None,
                    help="From softnms timing.csv if --softnms-timing not given")
    ap.add_argument("--softnms-timing", default=None)
    ap.add_argument("--tent-fps", type=float, required=True)
    ap.add_argument("--dataset-label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = pd.read_csv(args.baseline_csv)
    soft = pd.read_csv(args.softnms_csv)
    tent = pd.read_csv(args.tent_csv)
    adap = pd.read_csv(args.adaptive_csv)

    # baseline csv typically has scheme=baseline,adaptive_piecewise,adaptive_linear
    map_base_all = get_map(base, "ALL", scheme="baseline")
    map_base_rough = get_map(base, "rough", scheme="baseline")
    map_soft_all = get_map(soft, "ALL")
    map_soft_rough = get_map(soft, "rough")
    map_tent_all = get_map(tent, "ALL", scheme="baseline") if "scheme" in tent.columns else get_map(tent, "ALL")
    map_tent_rough = get_map(tent, "rough", scheme="baseline") if "scheme" in tent.columns else get_map(tent, "rough")
    map_ours_all = get_map(adap, "ALL", scheme="adaptive_piecewise")
    map_ours_rough = get_map(adap, "rough", scheme="adaptive_piecewise")

    soft_fps = args.softnms_fps
    if args.softnms_timing and soft_fps is None:
        t = pd.read_csv(args.softnms_timing)
        soft_fps = float(t["fps"].iloc[0])

    base_fps = args.baseline_fps
    ours_fps = base_fps  # same forward pass + cheap proxy
    tent_fps = args.tent_fps

    rows = [
        ("Global baseline ($T=0.40$)", map_base_all, map_base_rough, base_fps,
         "N", "--"),
        ("Soft-NMS \\cite{bodla2017}", map_soft_all, map_soft_rough, soft_fps,
         "N", "+post-proc."),
        ("TENT \\cite{wang2021tent}",   map_tent_all, map_tent_rough, tent_fps,
         "Adapt only", "+BN adapt"),
        ("\\textbf{Ours (F1-balanced)}", map_ours_all, map_ours_rough, ours_fps,
         "N", "$<2$~ms/tile"),
    ]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Method comparison on {args.dataset_label} test set. "
                 r"FPS measured on a single A800; \emph{Retrain?} indicates whether "
                 r"backbone weights are modified; \emph{Cost} reports any extra runtime overhead.}}")
    lines.append(rf"\label{{tab:methods_{args.dataset_label.lower().replace('-','').replace(' ','')}}}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l c c c c l}")
    lines.append(r"\toprule")
    lines.append(r"Method & mAP@0.5 (ALL) & mAP@0.5 (Rough) & FPS & Retrain? & Cost \\")
    lines.append(r"\midrule")
    for name, m_all, m_rgh, fps, retrain, cost in rows:
        lines.append(rf"{name} & {m_all:.3f} & {m_rgh:.3f} & {fps:.1f} & {retrain} & {cost} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {out_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
