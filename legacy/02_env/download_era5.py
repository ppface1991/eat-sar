"""
Download ERA5 reanalysis fields covering all SAR scene timestamps.

我们不针对每张图单独发一次 CDS 请求 (会被限流), 而是:
  1. 读取 meta CSV (来自步骤 01) 得到所有 (year, month, day, hour) 组合;
  2. 按 (year, month) 批量打包下载, 整月 NetCDF;
  3. 空间范围: 用所有场景经纬度的最小/最大值 + 1° buffer;
  4. 变量: 10m u/v 风分量 + 有效波高 (significant wave height);
  5. 保存到 cache/era5/<year>_<month>.nc

前置准备 (用户只需做一次):
  1. 注册 https://cds.climate.copernicus.eu/  并接受 ERA5 license;
  2. 把 API key 写入 ~/.cdsapirc:
        url: https://cds.climate.copernicus.eu/api/v2
        key: <UID>:<API-KEY>
  3. pip install cdsapi

Usage:
    python scripts/02_env/download_era5.py --config configs/default.yaml \\
        --meta-csv cache/meta/sardet100k_meta.csv \\
        [--meta-csv cache/meta/sarship_meta.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from eat_sar.utils import ensure_dir, load_config  # noqa: E402


def collect_year_month_pairs(metas: list[Path]) -> pd.DataFrame:
    frames = []
    for p in metas:
        df = pd.read_csv(p)
        df = df.dropna(subset=["datetime", "lat", "lon"])
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full["year"] = full["datetime"].dt.year
    full["month"] = full["datetime"].dt.month
    full["day"] = full["datetime"].dt.day
    full["hour"] = full["datetime"].dt.hour
    return full


def bbox_with_buffer(df: pd.DataFrame, buffer: float = 1.0) -> list[float]:
    """ERA5 area = [north, west, south, east]."""
    n = float(df["lat"].max()) + buffer
    s = float(df["lat"].min()) - buffer
    w = float(df["lon"].min()) - buffer
    e = float(df["lon"].max()) + buffer
    return [round(n, 2), round(w, 2), round(s, 2), round(e, 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--meta-csv", action="append", required=True,
                    help="可重复传入, 例如 --meta-csv ... --meta-csv ...")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印将要下载的内容, 不实际请求 CDS")
    args = ap.parse_args()

    cfg = load_config(args.config)
    era5_cfg = cfg["era5"]
    out_dir = ensure_dir(cfg["paths"]["era5_dir"])

    metas = [Path(p) for p in args.meta_csv]
    df = collect_year_month_pairs(metas)
    print(f"Total scenes with valid time+geo: {len(df)}")

    ym_groups = df.groupby(["year", "month"])
    print(f"Distinct (year, month) groups: {len(ym_groups)}")

    if not args.dry_run:
        import cdsapi
        client = cdsapi.Client()

    for (yr, mo), grp in tqdm(ym_groups, desc="ERA5 monthly"):
        area = bbox_with_buffer(grp, buffer=1.0)
        days = sorted(grp["day"].unique().astype(int).tolist())
        hours = sorted(grp["hour"].unique().astype(int).tolist())

        out_file = out_dir / f"{int(yr):04d}_{int(mo):02d}.nc"
        if out_file.exists():
            print(f"  [skip exists] {out_file.name}")
            continue

        request = {
            "product_type": era5_cfg["product_type"],
            "format": "netcdf",
            "variable": era5_cfg["variables"],
            "year": [f"{int(yr):04d}"],
            "month": [f"{int(mo):02d}"],
            "day": [f"{d:02d}" for d in days],
            "time": [f"{h:02d}:00" for h in hours],
            "area": area,   # [N, W, S, E]
            "grid": era5_cfg["grid"],
        }

        if args.dry_run:
            print(f"  [dry] {out_file.name}  area={area}  days={len(days)}  hours={len(hours)}")
            continue

        # ERA5 single-levels 包含 10m wind; significant wave height 在 ERA5 (atmospheric) 也有
        # (waves from ERA5 single-levels reanalysis)
        client.retrieve("reanalysis-era5-single-levels", request, str(out_file))


if __name__ == "__main__":
    main()
