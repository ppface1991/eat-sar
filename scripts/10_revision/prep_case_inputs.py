#!/usr/bin/env python3
"""Step G — prepare inputs for plot_case_examples_v2.py (paper's Fig. 3).

Reads numbers.json + the test manifest_proxy.csv and writes:
  1. <out>/adaptive_params_<ds>.json — piecewise/linear policy + env_value_key
  2. <out>/manifest_cases_<ds>.csv   — manifest + "wind_bin" column (legacy name)
  3. <out>/data_case_<ds>.yaml       — {"path": <label root parent>} so that
     plot_case_examples_v2.py resolves labels as path/labels/<split>/<stem>.txt

Then run (SAR-Ship example):
  python scripts\\07_viz\\plot_case_examples_v2.py ^
    --manifest <out>\\manifest_cases_sarship.csv ^
    --data <out>\\data_case_sarship.yaml --split all ^
    --adaptive-json <out>\\adaptive_params_sarship.json ^
    --baseline-T 0.40 --target-class 0 --auto-discriminative ^
    --dataset-label "SAR-Ship-Dataset" --out <out>\\case_examples_sarship.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from revkit import read_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers", required=True)
    ap.add_argument("--dataset", required=True, help="key in numbers.json (sardet/sarship)")
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--label-root", required=True,
                    help="e.g. D:\\Project\\eat-sar\\work\\yolo\\sarship (labels live under <root>\\labels)")
    ap.add_argument("--label-split", required=True,
                    help="subdir of labels/ holding the test labels, e.g. 'all' (SAR-Ship) or 'test' (SARDet)")
    ap.add_argument("--e-col", default="e_pred")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    num = json.load(open(args.numbers))[args.dataset]
    edges = num["edges"]
    T = num["T_f1"]
    lin = num["linear"]

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    params = {
        "piecewise": {
            "bins": edges,
            "thresholds": [T["low"], T["medium"], T["high"]],
            "bin_order": ["low", "medium", "high"],
        },
        "linear": {"a": lin["a"], "b": lin["b"]},
        "env_value_key": args.e_col,
    }
    pj = out / f"adaptive_params_{args.dataset}.json"
    pj.write_text(json.dumps(params, indent=1))

    mf = read_manifest(args.test_manifest).dropna(subset=[args.e_col]).copy()
    bins = np.digitize(mf[args.e_col].values, edges)
    mf["wind_bin"] = [["low", "medium", "high"][b] for b in bins]
    mc = out / f"manifest_cases_{args.dataset}.csv"
    mf.to_csv(mc, index=False)

    dy = out / f"data_case_{args.dataset}.yaml"
    dy.write_text(f"path: {Path(args.label_root).as_posix()}\n")

    print(f"[saved] {pj}\n[saved] {mc} ({len(mf)} rows)\n[saved] {dy}")
    print(f"next: python scripts\\07_viz\\plot_case_examples_v2.py "
          f"--manifest {mc} --data {dy} --split {args.label_split} "
          f"--adaptive-json {pj} --baseline-T 0.40 --auto-discriminative "
          f"--dataset-label {args.dataset} --out {out}/case_examples_{args.dataset}.pdf")


if __name__ == "__main__":
    main()
