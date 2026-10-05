#!/usr/bin/env python3
"""Step P — sanitize a full manifest_proxy.csv for public release.

Keeps only label-free columns (stem, the e_pred* columns) and drops local
paths (image_path, pred_npz). Used to build manifests/<dataset>_<split>_e.csv
for the public repository, so that the per-image clutter proxy values
promised in the letter's code statement are available without shipping
local paths or caches.

Usage:
  python make_public_manifests.py --manifest work/cache_preds/sarship_test/manifest_proxy.csv \
      --dataset sarship --split test --out-dir ../manifests
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DROP_COLS = ["image_path", "pred_npz"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="manifest_proxy.csv")
    ap.add_argument("--dataset", required=True, choices=["sardet", "sarship"])
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, dtype={"stem": str})
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    e_cols = [c for c in df.columns if c.startswith("e_pred")]
    if "e_pred" not in e_cols:
        raise SystemExit("[error] no e_pred column found -- is this a manifest_proxy.csv?")

    # Tertile bin of the main e_pred column (tertiles = the calibration rule;
    # for the test split this is the *true* stratum used in the letter).
    q1, q2 = np.quantile(df["e_pred"].values, [1 / 3, 2 / 3])
    df["tertile"] = np.digitize(df["e_pred"].values, [q1, q2])  # 0,1,2
    df["tertile"] = df["tertile"].map({0: "low", 1: "medium", 2: "high"})

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{args.dataset}_{args.split}_e.csv"
    df.to_csv(dst, index=False)
    counts = df["tertile"].value_counts().to_dict()
    print(f"[saved] {dst}: {len(df)} rows, e columns: {e_cols}, tertiles {counts}")


if __name__ == "__main__":
    main()
