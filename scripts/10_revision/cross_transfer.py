"""
Step R5 — Cross-dataset / cross-sensor transfer of the policy (Reviewer 2:
"show that bin edges / T(e) transfer across sensors").

Source = dataset A (val -> edges_A, T_A).  Target = dataset B test.
Three transfer modes evaluated on B-test:
  (i)   raw       : edges_A AND T_A applied unchanged          (no target info)
  (ii)  requantile: T_A kept, edges re-estimated as tertiles of e on the
                    UNLABELLED target images (label-free, uses only images)
  (iii) in-domain : edges_B, T_B from B-val (upper reference)
plus baseline T=0.40 on B-test.

Usage:
  python scripts/10_revision/cross_transfer.py \
      --src-main $OUT/sardet_main.json \
      --tgt-main $OUT/sarship_main.json \
      --tgt-test-manifest $WORK/cache_preds/sarship_test/manifest_proxy.csv \
      --tgt-test-labels   $WORK/sarship_root/labels/test \
      --target-class 0 --out $OUT/transfer_sardet_to_sarship.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from revkit import (BIN_LABELS, apply_policy, assign_bins, dump_json,  # noqa: E402
                    evaluate, filter_gt, load_all, read_manifest, tertile_edges)


def _edges(j):
    return [(-np.inf if x in ("-inf", "-Infinity") else np.inf if x in ("inf", "Infinity") else float(x))
            for x in j["edges"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-main", required=True, help="run_main_eval.py JSON of SOURCE dataset")
    ap.add_argument("--tgt-main", required=True, help="run_main_eval.py JSON of TARGET dataset")
    ap.add_argument("--tgt-test-manifest", required=True)
    ap.add_argument("--tgt-test-labels", required=True)
    ap.add_argument("--e-col", default="e_pred")
    ap.add_argument("--target-class", type=int, default=None)
    ap.add_argument("--baseline-T", type=float, default=0.40)
    ap.add_argument("--require-gt", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = json.load(open(args.src_main)); tgt = json.load(open(args.tgt_main))
    test = read_manifest(args.tgt_test_manifest).dropna(subset=[args.e_col])
    Rt = load_all(test, Path(args.tgt_test_labels), args.target_class, "tgt-test")
    st, et = filter_gt(Rt, test["stem"].values, test[args.e_col].values, args.require_gt)

    T_A, T_B = src["T_f1"], tgt["T_f1"]
    edges_A, edges_B = _edges(src), _edges(tgt)
    edges_req = tertile_edges(et)              # label-free, target images only

    modes = {
        "baseline_T0.40": args.baseline_T,
        "raw_transfer(edges_A,T_A)": apply_policy(et, st, edges_A, T_A, BIN_LABELS),
        "requantile(edges_tgt_unlabelled,T_A)": apply_policy(et, st, edges_req, T_A, BIN_LABELS),
        "in_domain(edges_B,T_B)": apply_policy(et, st, edges_B, T_B, BIN_LABELS),
    }
    base = evaluate(Rt, st, args.baseline_T)["AP"]
    res = {}
    for name, pol in modes.items():
        m = evaluate(Rt, st, pol)
        # bin occupancy under raw transfer (diagnoses DN-scale mismatch)
        occ = None
        if "raw" in name:
            occ = {lb: int((assign_bins(et, edges_A) == lb).sum()) for lb in BIN_LABELS}
        res[name] = dict(**m, dAP_pp=100 * (m["AP"] - base), bin_occupancy=occ)
        print(f"{name:40s} AP={m['AP']:.4f} ({100*(m['AP']-base):+.2f}pp) F1={m['F1']:.4f} occ={occ}")
    dump_json(dict(source=args.src_main, target=args.tgt_main, edges_A=edges_A, edges_B=edges_B,
                   edges_requantile=edges_req, T_A=T_A, T_B=T_B, results=res), args.out)


if __name__ == "__main__":
    main()
