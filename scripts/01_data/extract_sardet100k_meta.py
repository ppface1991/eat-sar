"""
Extract per-scene metadata from SARDet-100K (ship subset).

For each ship-containing image, we record:
  image_id, file_name, split, width, height,
  num_ships, lat, lon, datetime, sensor, polarization

NOTE on coordinates/time:
  SARDet-100K 的标注 JSON 里通常每张图带有 'sensor' / 'imaging_time' / 'lat' / 'lon'
  等字段 (论文附带元数据)。不同发布版本字段名略有差异，本脚本会尝试多种命名。
  如果你的版本缺少这些字段，请按照 README 的说明手工补充 (e.g. 从原始 Sentinel-1
  产品名解析时间, 从场景 footprint 取中心经纬度)。

Usage:
    python scripts/01_data/extract_sardet100k_meta.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from eat_sar.utils import ensure_dir, load_config  # noqa: E402


# ---------- helpers ----------

# Sentinel-1 product naming: S1A_IW_GRDH_1SDV_20200101T120000_..._lat_lon
S1_TIME_RE = re.compile(r"_(\d{8}T\d{6})_")

POSSIBLE_LAT_KEYS = ["lat", "latitude", "center_lat", "scene_lat"]
POSSIBLE_LON_KEYS = ["lon", "longitude", "center_lon", "scene_lon"]
POSSIBLE_TIME_KEYS = ["imaging_time", "datetime", "acquisition_time", "time"]
POSSIBLE_SENSOR_KEYS = ["sensor", "platform", "satellite"]
POSSIBLE_POL_KEYS = ["polarization", "pol"]


def _pick(d: dict, keys: list[str], default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _parse_time_from_filename(name: str) -> str | None:
    m = S1_TIME_RE.search(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").isoformat()
    except ValueError:
        return None


def _find_ship_category_id(categories: list[dict], override: int | None) -> int:
    if override is not None:
        return int(override)
    for c in categories:
        if str(c.get("name", "")).lower() == "ship":
            return int(c["id"])
    raise ValueError(f"ship category not found in: {[c.get('name') for c in categories]}")


# ---------- main ----------

def process_split(coco_json: Path, split_name: str, ship_cat_override: int | None) -> pd.DataFrame:
    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    ship_id = _find_ship_category_id(coco["categories"], ship_cat_override)
    print(f"[{split_name}] ship category id = {ship_id}")

    # image_id -> num ships
    img_ship_count: dict[int, int] = {}
    for ann in coco["annotations"]:
        if int(ann["category_id"]) == ship_id:
            img_ship_count[int(ann["image_id"])] = img_ship_count.get(int(ann["image_id"]), 0) + 1

    rows = []
    for img in tqdm(coco["images"], desc=f"parse {split_name}"):
        num_ships = img_ship_count.get(int(img["id"]), 0)
        if num_ships == 0:
            continue  # ship-only subset

        lat = _pick(img, POSSIBLE_LAT_KEYS)
        lon = _pick(img, POSSIBLE_LON_KEYS)
        t = _pick(img, POSSIBLE_TIME_KEYS)
        if t is None:
            t = _parse_time_from_filename(str(img.get("file_name", "")))
        sensor = _pick(img, POSSIBLE_SENSOR_KEYS, default="unknown")
        pol = _pick(img, POSSIBLE_POL_KEYS, default="unknown")

        rows.append({
            "dataset": "SARDet-100K",
            "split": split_name,
            "image_id": img["id"],
            "file_name": img["file_name"],
            "width": img.get("width"),
            "height": img.get("height"),
            "num_ships": num_ships,
            "lat": lat,
            "lon": lon,
            "datetime": t,
            "sensor": sensor,
            "polarization": pol,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = cfg["datasets"]["sardet100k"]
    if not ds["enabled"]:
        print("SARDet-100K disabled in config; skip.")
        return

    root = Path(ds["root"])
    ann_dir = root / ds["annotations"]
    ship_cat = ds.get("ship_category_id")

    all_dfs = []
    for split in ["train", "val", "test"]:
        # SARDet-100K 的标注文件命名通常是 instances_train.json / val / test
        candidates = list(ann_dir.glob(f"*{split}*.json"))
        if not candidates:
            print(f"[warn] no annotation file for split={split} under {ann_dir}")
            continue
        df = process_split(candidates[0], split, ship_cat)
        all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("No annotation files found. Check configs/default.yaml paths.")

    out = pd.concat(all_dfs, ignore_index=True)
    out_csv = Path(ds["meta_csv"])
    ensure_dir(out_csv.parent)
    out.to_csv(out_csv, index=False)
    print(f"Saved metadata: {out_csv}  ({len(out)} ship-containing images)")
    print(out.head())
    # Sanity stats
    miss_geo = out[["lat", "lon"]].isna().any(axis=1).sum()
    miss_time = out["datetime"].isna().sum()
    print(f"  missing lat/lon: {miss_geo} | missing datetime: {miss_time}")
    if miss_geo == len(out) and miss_time == len(out):
        print("\n  NOTE: this SARDet-100K release has no lat/lon/time fields.")
        print("        Use the image-derived sea-state proxy pipeline instead:")
        print("        python scripts/02_env/compute_image_clutter.py \\")
        print("            --meta-csv", out_csv, "--dataset sardet100k")


if __name__ == "__main__":
    main()
