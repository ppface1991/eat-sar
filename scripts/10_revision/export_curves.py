#!/usr/bin/env python3
"""Step F — export per-bin metric-vs-threshold curves for the paper's Fig. 2.

Sweeps a grid of thresholds on the TEST split, per clutter bin (edges taken
from numbers.json, i.e. fitted on the calibration split), and writes a JSON:

    {"left":  {"calm": [[Ts],[F1s]], "moderate": ..., "rough": ..., "ALL": ...},
     "right": {"calm": [[Ts],[APs]], ...}}

which is the format consumed by the paper figure scripts (replot_all.py).
Run once per dataset after `run_revision.cmd analyze`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from revkit import evaluate, filter_gt, load_all, read_manifest

BINMAP = {"low": "calm", "medium": "moderate", "high": "rough"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers", required=True, help="revision_out/numbers.json")
    ap.add_argument("--dataset", required=True, help="key in numbers.json (sardet/sarship)")
    ap.add_argument("--test-manifest", required=True, help="manifest_proxy.csv of the test split")
    ap.add_argument("--test-labels", required=True, help="label dir of the test split")
    ap.add_argument("--target-class", type=int, default=None)
    ap.add_argument("--e-col", default="e_pred")
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args()

    num = json.load(open(args.numbers))[args.dataset]
    edges = num["edges"]
    test = read_manifest(args.test_manifest).dropna(subset=[args.e_col])

    print(f"[curves:{args.dataset}] {len(test)} test images, edges {edges[0]:.2f}/{edges[1]:.2f}")
    records = load_all(test, Path(args.test_labels), args.target_class)
    stems_all = test["stem"].values
    e_all = test[args.e_col].values
    sv, ev = filter_gt(records, stems_all, e_all, require_gt=True)
    print(f"[curves:{args.dataset}] kept {len(sv)}/{len(stems_all)} images with >=1 GT ship")

    bins = np.digitize(ev, edges)  # 0,1,2 -> low, medium, high
    groups = {"ALL": np.arange(len(sv))}
    for b in range(3):
        groups[BINMAP[["low", "medium", "high"][b]]] = np.where(bins == b)[0]

    ts = np.round(np.arange(0.05, 0.8501, 0.025), 4).tolist()
    out = {"left": {}, "right": {}}
    for gname, idx in groups.items():
        gstems = sv[idx]
        f1s, aps = [], []
        for T in ts:
            r = evaluate(records, gstems, T)
            f1s.append(round(r["F1"], 4))
            aps.append(round(r["AP"], 4))
        out["left"][gname] = [ts, f1s]
        out["right"][gname] = [ts, aps]
        i = int(np.nanargmax(f1s))
        print(f"  {gname:8s} n={len(idx):5d}  best F1 {max(f1s):.4f} @T={ts[i]:.3f}  "
              f"AP@0.40={aps[ts.index(0.4)]:.4f}  AP@0.45={aps[ts.index(0.45)]:.4f}")

    Path(args.out).write_text(json.dumps(out))
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
