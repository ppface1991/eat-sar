"""
Summarize sample counts per environment bin (for论文 Table 1).

Usage:
    python scripts/02_env/summarize_env_strata.py \\
        --meta-csv cache/meta/sardet100k_meta_with_env.csv \\
                  cache/meta/sarship_meta_with_env.csv \\
        --out runs/env_strata.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-csv", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = [pd.read_csv(p) for p in args.meta_csv]
    df = pd.concat(frames, ignore_index=True)

    grouped = df.groupby(["dataset", "split", "wind_bin"]).agg(
        n_scenes=("file_name", "count"),
        n_ships=("num_ships", "sum"),
        wind_mean=("wind_speed", "mean"),
        swh_mean=("swh", "mean"),
    ).reset_index()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(args.out, index=False)
    print(grouped.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
