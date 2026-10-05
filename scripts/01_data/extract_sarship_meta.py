"""
Extract per-tile metadata from SAR-Ship-Dataset (GF-3 + Sentinel-1, 256x256 tiles, VOC XML).

This dataset has NO ImageSets/Main/ split files in the shipped version,
so we generate a deterministic 80/10/10 train/val/test split using the
project seed. Tiles without any 'ship' object are skipped (we only care
about positive samples for environment-conditioned threshold analysis).

Time/lat/lon are usually not recoverable from tile names alone — we leave
them NaN and rely on image-derived clutter statistics as the sea-state
proxy (same approach as SARDet-100K).

Usage:
    python scripts/01_data/extract_sarship_meta.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from eat_sar.utils import ensure_dir, load_config  # noqa: E402


S1_TIME_RE = re.compile(r"(\d{8}T\d{6})")
GF3_TIME_RE = re.compile(r"(\d{8})")  # GF-3 names often start with date


def parse_voc_xml(xml_path: Path):
    """Return (width, height, list_of_ship_boxes [(xmin,ymin,xmax,ymax), ...])."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    boxes = []
    for obj in root.findall("object"):
        name = str(obj.findtext("name", "")).lower()
        if name not in ("ship", "boat"):
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            boxes.append((
                int(float(bb.findtext("xmin", "0"))),
                int(float(bb.findtext("ymin", "0"))),
                int(float(bb.findtext("xmax", "0"))),
                int(float(bb.findtext("ymax", "0"))),
            ))
        except (TypeError, ValueError):
            continue
    return width, height, boxes


def infer_sensor_and_time(name: str):
    if name.startswith("GF"):
        # GF_VV_20170127_31 -> 2017-01-27
        parts = name.split("_")
        for p in parts:
            if len(p) == 8 and p.isdigit():
                try:
                    return "GF-3", pd.to_datetime(p, format="%Y%m%d").isoformat()
                except Exception:
                    pass
        return "GF-3", None
    if name.startswith("Gao"):
        # Gao_ship_hh_0201608254401010020 -> 2016-08-25
        m = re.search(r"02(\d{6})", name)
        if m:
            try:
                return "GF-3", pd.to_datetime(m.group(1), format="%y%m%d").isoformat()
            except Exception:
                pass
        return "GF-3", None
    m = S1_TIME_RE.search(name)
    if m:
        try:
            return "Sentinel-1", pd.to_datetime(m.group(1), format="%Y%m%dT%H%M%S").isoformat()
        except Exception:
            pass
    return "unknown", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--train-frac", type=float, default=0.80)
    ap.add_argument("--val-frac", type=float, default=0.10)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = cfg["datasets"]["sarship"]
    if not ds["enabled"]:
        print("SAR-Ship-Dataset disabled; skip.")
        return

    seed = cfg["project"]["seed"]
    root = Path(ds["root"])
    img_dir = root / ds["images_subdir"]
    ann_dir = root / ds["annotations_subdir"]
    if not ann_dir.exists():
        # Try the alt name shipped with the public release
        alt = root / "Annotations_new"
        if alt.exists():
            ann_dir = alt
            print(f"Annotations dir not found; using {ann_dir}")
        else:
            raise FileNotFoundError(f"Neither {ann_dir} nor {alt} exists")

    xmls = sorted(ann_dir.glob("*.xml"))
    print(f"Found {len(xmls)} XML files in {ann_dir}")

    rows = []
    for xml_p in tqdm(xmls, desc="parse XML"):
        name = xml_p.stem
        img_p = img_dir / f"{name}.jpg"
        if not img_p.exists():
            continue
        w, h, boxes = parse_voc_xml(xml_p)
        if len(boxes) == 0:
            continue
        sensor, dt = infer_sensor_and_time(name)
        rows.append({
            "dataset": "SAR-Ship-Dataset",
            "image_id": name,
            "file_name": f"{name}.jpg",
            "width": w, "height": h,
            "num_ships": len(boxes),
            "lat": None, "lon": None,
            "datetime": dt,
            "sensor": sensor,
            "polarization": "unknown",
        })

    if not rows:
        raise RuntimeError("No ship-containing tiles found. Check paths.")

    df = pd.DataFrame(rows)

    # Deterministic 80/10/10 split
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_train = int(len(df) * args.train_frac)
    n_val = int(len(df) * args.val_frac)
    split = np.array(["test"] * len(df), dtype=object)
    split[idx[:n_train]] = "train"
    split[idx[n_train:n_train + n_val]] = "val"
    df["split"] = split

    out_csv = Path(ds["meta_csv"])
    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)
    print(f"Saved metadata: {out_csv}  ({len(df)} ship-containing tiles)")
    print(df.groupby("split").size().to_string())
    print(f"Sensors: {df['sensor'].value_counts().to_dict()}")
    miss_time = df["datetime"].isna().sum()
    print(f"  missing datetime: {miss_time}/{len(df)} (expected; we use image-derived proxy)")


if __name__ == "__main__":
    main()
