"""
Convert SARDet-100K (COCO, ship subset) and SAR-Ship-Dataset (VOC)
into Ultralytics YOLO format (single class: ship).

Output layout (per dataset):
    <out_root>/<dataset>/
        images/{train,val,test}/
        labels/{train,val,test}/
        data.yaml

Usage:
    python scripts/03_train/build_yolo_dataset.py --config configs/default.yaml \\
        --out /root/autodl-tmp/eat-sar/yolo
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from eat_sar.utils import ensure_dir, load_config  # noqa: E402


def coco_bbox_to_yolo(bbox, W, H):
    x, y, w, h = bbox
    cx = (x + w / 2) / W
    cy = (y + h / 2) / H
    return cx, cy, w / W, h / H


def convert_sardet100k(cfg, out_root: Path):
    ds = cfg["datasets"]["sardet100k"]
    if not ds["enabled"]:
        return None
    src = Path(ds["root"])
    out = out_root / "sardet100k"
    for s in ("train", "val", "test"):
        ensure_dir(out / "images" / s)
        ensure_dir(out / "labels" / s)

    ann_dir = src / ds["annotations"]
    img_dir = src / ds["images_subdir"]
    for split in ("train", "val", "test"):
        cand = list(ann_dir.glob(f"*{split}*.json"))
        if not cand:
            continue
        coco = json.load(open(cand[0], encoding="utf-8"))
        cats = {c["id"]: c["name"].lower() for c in coco["categories"]}
        ship_id = next(i for i, n in cats.items() if n == "ship")

        # group anns by image
        ann_by_img: dict[int, list] = {}
        for a in coco["annotations"]:
            if int(a["category_id"]) != ship_id:
                continue
            ann_by_img.setdefault(int(a["image_id"]), []).append(a)

        for img in tqdm(coco["images"], desc=f"SARDet-100K {split}"):
            iid = int(img["id"])
            if iid not in ann_by_img:
                continue
            src_img = img_dir / img["file_name"]
            if not src_img.exists():
                # 有些版本 images 又分 train/val/test 子目录
                alt = img_dir / split / img["file_name"]
                if alt.exists():
                    src_img = alt
                else:
                    continue
            dst_img = out / "images" / split / src_img.name
            if not dst_img.exists():
                try:
                    dst_img.symlink_to(src_img.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(src_img, dst_img)

            W, H = img["width"], img["height"]
            lbl = out / "labels" / split / (Path(src_img.name).stem + ".txt")
            with open(lbl, "w") as f:
                for a in ann_by_img[iid]:
                    cx, cy, w, h = coco_bbox_to_yolo(a["bbox"], W, H)
                    f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    yaml_text = f"""# SARDet-100K ship-only (auto-generated)
path: {out.resolve()}
train: images/train
val: images/val
test: images/test
names:
  0: ship
"""
    (out / "data.yaml").write_text(yaml_text)
    print(f"data.yaml -> {out / 'data.yaml'}")
    return out / "data.yaml"


def convert_sarship(cfg, out_root: Path):
    ds = cfg["datasets"]["sarship"]
    if not ds["enabled"]:
        return None
    src = Path(ds["root"])
    out = out_root / "sarship"
    for s in ("train", "val", "test"):
        ensure_dir(out / "images" / s)
        ensure_dir(out / "labels" / s)

    img_dir = src / ds["images_subdir"]
    ann_dir = src / ds["annotations_subdir"]
    if not ann_dir.exists():
        alt = src / "Annotations_new"
        if alt.exists():
            ann_dir = alt
            print(f"  annotations dir fallback -> {ann_dir}")

    # Prefer split column from meta CSV (deterministic 80/10/10 from step 01).
    # Fall back to ImageSets/Main/{split}.txt if present.
    meta_csv = Path(ds["meta_csv"])
    sets_dir = src / "ImageSets" / "Main"
    split_to_names: dict[str, list[str]] = {}
    if meta_csv.exists():
        import pandas as _pd
        mdf = _pd.read_csv(meta_csv)
        if "split" in mdf.columns:
            for s, g in mdf.groupby("split"):
                split_to_names[s] = g["image_id"].astype(str).tolist()
            print(f"  SAR-Ship split from meta CSV: "
                  f"{ {k: len(v) for k, v in split_to_names.items()} }")

    for split in ("train", "val", "test"):
        if split in split_to_names:
            names = split_to_names[split]
        else:
            sp_file = sets_dir / f"{split}.txt"
            if not sp_file.exists():
                print(f"  [skip] no split info for {split} (no meta CSV and no {sp_file})")
                continue
            names = [ln.strip() for ln in open(sp_file) if ln.strip()]
        for name in tqdm(names, desc=f"SAR-Ship {split}"):
            xml_p = ann_dir / f"{name}.xml"
            img_p = img_dir / f"{name}.jpg"
            if not xml_p.exists() or not img_p.exists():
                continue
            root = ET.parse(xml_p).getroot()
            size = root.find("size")
            W = int(size.findtext("width"))
            H = int(size.findtext("height"))
            boxes = []
            for obj in root.findall("object"):
                if str(obj.findtext("name", "")).lower() not in ("ship", "boat"):
                    continue
                bb = obj.find("bndbox")
                xmin = float(bb.findtext("xmin")); ymin = float(bb.findtext("ymin"))
                xmax = float(bb.findtext("xmax")); ymax = float(bb.findtext("ymax"))
                cx = (xmin + xmax) / 2 / W
                cy = (ymin + ymax) / 2 / H
                w = (xmax - xmin) / W
                h = (ymax - ymin) / H
                boxes.append((cx, cy, w, h))
            if not boxes:
                continue
            dst_img = out / "images" / split / img_p.name
            if not dst_img.exists():
                try:
                    dst_img.symlink_to(img_p.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(img_p, dst_img)
            lbl = out / "labels" / split / f"{name}.txt"
            with open(lbl, "w") as f:
                for cx, cy, w, h in boxes:
                    f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    yaml_text = f"""# SAR-Ship-Dataset ship-only (auto-generated)
path: {out.resolve()}
train: images/train
val: images/val
test: images/test
names:
  0: ship
"""
    (out / "data.yaml").write_text(yaml_text)
    print(f"data.yaml -> {out / 'data.yaml'}")
    return out / "data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", required=True, help="YOLO 数据集根目录")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_root = ensure_dir(args.out)
    convert_sardet100k(cfg, out_root)
    convert_sarship(cfg, out_root)


if __name__ == "__main__":
    main()
