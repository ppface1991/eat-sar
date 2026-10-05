"""
For each SAR scene (lat, lon, datetime), look up ERA5 wind & wave values via
nearest-neighbour spatial interpolation + nearest-hour temporal selection.

Inputs:  cache/era5/<year>_<month>.nc  (downloaded in step 02/download_era5.py)
         cache/meta/*.csv              (from step 01)

Output:  cache/meta/<dataset>_with_env.csv  with extra columns:
            u10, v10, wind_speed, swh, wind_bin, swh_bin

Usage:
    python scripts/02_env/match_era5_to_scenes.py --config configs/default.yaml \\
        --meta-csv cache/meta/sardet100k_meta.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from eat_sar.utils import bin_value, ensure_dir, load_config  # noqa: E402


# ERA5 NetCDF 中变量短名 (CDS 输出): u10, v10, swh
VAR_MAP = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "significant_height_of_combined_wind_waves_and_swell": "swh",
}


def open_month(era5_dir: Path, year: int, month: int) -> xr.Dataset | None:
    f = era5_dir / f"{year:04d}_{month:02d}.nc"
    if not f.exists():
        return None
    ds = xr.open_dataset(f)
    return ds


def sample_point(ds: xr.Dataset, lat: float, lon: float, t: pd.Timestamp) -> dict:
    # ERA5 经度通常是 0-360; 转换:
    if lon < 0 and ds["longitude"].max() > 180:
        lon = lon + 360.0
    try:
        pt = ds.sel(latitude=lat, longitude=lon, time=t, method="nearest")
    except KeyError:
        return {"u10": np.nan, "v10": np.nan, "swh": np.nan}
    out = {}
    for short in ("u10", "v10", "swh"):
        if short in pt:
            v = float(pt[short].values)
            out[short] = v if np.isfinite(v) else np.nan
        else:
            out[short] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--meta-csv", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    era5_dir = Path(cfg["paths"]["era5_dir"])
    wind_bins = cfg["era5"]["wind_bins"]
    wind_labels = cfg["era5"]["wind_labels"]
    swh_bins = cfg["era5"]["swh_bins"]
    swh_labels = cfg["era5"]["swh_labels"]

    df = pd.read_csv(args.meta_csv)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    n_total = len(df)
    df = df.dropna(subset=["datetime", "lat", "lon"]).reset_index(drop=True)
    print(f"Scenes with valid time+geo: {len(df)} / {n_total}")

    # Cache opened datasets by (year, month)
    ds_cache: dict[tuple[int, int], xr.Dataset] = {}

    u10s, v10s, swhs = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="match ERA5"):
        t = row["datetime"]
        key = (t.year, t.month)
        if key not in ds_cache:
            ds_cache[key] = open_month(era5_dir, *key)
        ds = ds_cache[key]
        if ds is None:
            u10s.append(np.nan); v10s.append(np.nan); swhs.append(np.nan)
            continue
        vals = sample_point(ds, float(row["lat"]), float(row["lon"]), t)
        u10s.append(vals["u10"]); v10s.append(vals["v10"]); swhs.append(vals["swh"])

    df["u10"] = u10s
    df["v10"] = v10s
    df["wind_speed"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
    df["swh"] = swhs

    # 分级
    df["wind_bin"] = df["wind_speed"].apply(
        lambda v: bin_value(v, wind_bins, wind_labels) if np.isfinite(v) else "unknown"
    )
    df["swh_bin"] = df["swh"].apply(
        lambda v: bin_value(v, swh_bins, swh_labels) if np.isfinite(v) else "unknown"
    )

    out_csv = Path(args.meta_csv).with_name(Path(args.meta_csv).stem + "_with_env.csv")
    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print("Wind-speed bin counts:")
    print(df["wind_bin"].value_counts())
    print("SWH bin counts:")
    print(df["swh_bin"].value_counts())


if __name__ == "__main__":
    main()
