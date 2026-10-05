"""
Compute per-image sea-state proxy features from SAR images (no ERA5 needed).

Reads:  cache/meta/<dataset>_meta.csv  (from step 01)
Writes: cache/meta/<dataset>_meta_with_env.csv  with extra columns:
            clutter_mean, clutter_std, clutter_cv, clutter_entropy,
            wind_speed (= alias of selected proxy, for compatibility),
            wind_bin   (= bin label, for compatibility with downstream scripts)

We deliberately reuse the column names 'wind_speed' / 'wind_bin' so that
the downstream scripts (04_threshold, 05_adaptive, 06_eval, 07_viz) work
WITHOUT any modification. In the paper we relabel these as 'clutter_std' /
'sea_state_bin'.

Usage:
    python scripts/02_env/compute_image_clutter.py --config configs/default.yaml \\
        --meta-csv cache/meta/sardet100k_meta.csv \\
        --dataset sardet100k \\
        --proxy clutter_std \\
        [--bins 0 12 24 256]   # override config bins
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from eat_sar.clutter import compute_clutter_stats  # noqa: E402
from eat_sar.utils import bin_value, ensure_dir, load_config  # noqa: E402


def _coco_anns_index(coco_json: Path, ship_cat_name: str = "ship"):
    """Return (img_id->{file_name,W,H}, img_id->[xyxy boxes])."""
    coco = json.load(open(coco_json, "r", encoding="utf-8"))
    ship_id = next(c["id"] for c in coco["categories"]
                   if str(c.get("name", "")).lower() == ship_cat_name)
    imgs = {int(i["id"]): i for i in coco["images"]}
    boxes_by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        if int(a["category_id"]) != ship_id:
            continue
        x, y, w, h = a["bbox"]
        boxes_by_img.setdefault(int(a["image_id"]), []).append(
            [x, y, x + w, y + h]
        )
    return imgs, boxes_by_img


def _voc_boxes(xml_path: Path):
    """Return xyxy boxes (list) for ship/boat labels in a VOC XML."""
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        if str(obj.findtext("name", "")).lower() not in ("ship", "boat"):
            continue
        bb = obj.find("bndbox")
        boxes.append([float(bb.findtext("xmin")), float(bb.findtext("ymin")),
                      float(bb.findtext("xmax")), float(bb.findtext("ymax"))])
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--meta-csv", required=True)
    ap.add_argument("--dataset", required=True, choices=["sardet100k", "sarship"])
    ap.add_argument("--proxy", default="clutter_std",
                    choices=["clutter_mean", "clutter_std", "clutter_cv", "clutter_entropy"],
                    help="which feature to alias as 'wind_speed' for downstream scripts")
    ap.add_argument("--bins", type=float, nargs="+", default=None,
                    help="bin edges, e.g. --bins 0 12 24 256")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="bin labels, e.g. --labels calm moderate rough")
    ap.add_argument("--dilate-px", type=int, default=8,
                    help="pixels to expand each GT box before masking out ship pixels. "
                         "Use 8 for ~800px imagery (SARDet-100K), 3 for 256px (SAR-Ship-Dataset).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg["datasets"][args.dataset]
    root = Path(ds_cfg["root"])

    # Build (file_name -> xyxy boxes) lookup for THIS dataset
    print(f"Building GT box lookup for {args.dataset} ...")
    gt_lookup: dict[str, list] = {}
    img_path_lookup: dict[str, Path] = {}

    if args.dataset == "sardet100k":
        ann_dir = root / ds_cfg["annotations"]
        img_dir = root / ds_cfg["images_subdir"]
        for split in ("train", "val", "test"):
            cand = list(ann_dir.glob(f"*{split}*.json"))
            if not cand:
                continue
            imgs, boxes = _coco_anns_index(cand[0])
            for iid, im_meta in imgs.items():
                fname = im_meta["file_name"]
                gt_lookup[fname] = boxes.get(iid, [])
                # SARDet-100K images are under Images/<split>/<file>
                p = img_dir / split / fname
                if p.exists():
                    img_path_lookup[fname] = p
                else:
                    p2 = img_dir / fname
                    if p2.exists():
                        img_path_lookup[fname] = p2
    else:  # sarship
        img_dir = root / ds_cfg["images_subdir"]
        ann_dir = root / ds_cfg["annotations_subdir"]
        if not ann_dir.exists():
            alt = root / "Annotations_new"
            if alt.exists():
                ann_dir = alt
                print(f"  annotations dir fallback -> {ann_dir}")
        for xml_p in ann_dir.glob("*.xml"):
            name = xml_p.stem
            gt_lookup[f"{name}.jpg"] = _voc_boxes(xml_p)
            p = img_dir / f"{name}.jpg"
            if p.exists():
                img_path_lookup[f"{name}.jpg"] = p

    print(f"  {len(img_path_lookup)} images resolved on disk; "
          f"{len(gt_lookup)} have GT entries")

    # Load meta CSV (from step 01)
    meta = pd.read_csv(args.meta_csv)
    print(f"  meta CSV has {len(meta)} ship-containing entries")

    # Compute clutter stats per image
    feats = []
    miss = 0
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="clutter"):
        fname = row["file_name"]
        img_p = img_path_lookup.get(fname)
        if img_p is None:
            miss += 1
            feats.append({k: np.nan for k in
                          ["clutter_mean", "clutter_std", "clutter_cv",
                           "clutter_entropy", "bg_pixel_ratio"]})
            continue
        boxes = np.array(gt_lookup.get(fname, []), dtype=np.float32)
        if boxes.ndim == 1:
            boxes = boxes.reshape(0, 4)
        stats = compute_clutter_stats(img_p, boxes, dilate_px=args.dilate_px)
        feats.append(stats)

    feats_df = pd.DataFrame(feats)
    out = pd.concat([meta.reset_index(drop=True), feats_df], axis=1)

    # Alias selected proxy -> wind_speed (for compatibility)
    out["wind_speed"] = out[args.proxy]

    # Bin
    if args.bins is None:
        # auto: tertiles of valid values
        v = out["wind_speed"].dropna().values
        if len(v) < 30:
            raise RuntimeError("Too few valid clutter values to bin; check image paths.")
        q33, q66 = np.quantile(v, [1/3, 2/3])
        bins = [float(v.min()) - 1e-6, float(q33), float(q66), float(v.max()) + 1e-6]
        labels = ["calm", "moderate", "rough"]
        print(f"  Auto bins (tertiles of {args.proxy}): {bins}")
    else:
        bins = list(args.bins)
        labels = args.labels or [f"bin{i}" for i in range(len(bins) - 1)]

    out["wind_bin"] = out["wind_speed"].apply(
        lambda v: bin_value(v, bins, labels) if np.isfinite(v) else "unknown"
    )

    out_csv = Path(args.meta_csv).with_name(Path(args.meta_csv).stem + "_with_env.csv")
    ensure_dir(out_csv.parent)
    out.to_csv(out_csv, index=False)

    # Persist bin definition for downstream / paper
    bin_meta = {
        "proxy_feature": args.proxy,
        "bins": bins,
        "labels": labels,
        "n_missing_image": int(miss),
        "n_total": int(len(meta)),
    }
    bin_json = Path(args.meta_csv).with_name(Path(args.meta_csv).stem + "_bins.json")
    json.dump(bin_meta, open(bin_json, "w"), indent=2)

    print(f"Saved: {out_csv}")
    print(f"Saved: {bin_json}")
    print("\nBin counts:")
    print(out["wind_bin"].value_counts())
    print(f"Missing image files: {miss}")


if __name__ == "__main__":
    main()
