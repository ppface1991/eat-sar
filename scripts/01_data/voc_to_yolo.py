#!/usr/bin/env python3
"""Convert SAR-Ship-Dataset PASCAL VOC XML to YOLO txt format.

SAR-Ship has only one class: ship.
Splits: produce train.txt / val.txt list files (90/10 split, deterministic).
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import random

def convert_one(xml_path: Path, out_txt: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    W = int(size.find("width").text)
    H = int(size.find("height").text)
    lines = []
    for obj in root.iter("object"):
        b = obj.find("bndbox")
        x1 = float(b.find("xmin").text)
        y1 = float(b.find("ymin").text)
        x2 = float(b.find("xmax").text)
        y2 = float(b.find("ymax").text)
        # Clip to image bounds first (handles VOC annotations with x2>W or y2>H)
        x1 = max(0.0, min(x1, W))
        x2 = max(0.0, min(x2, W))
        y1 = max(0.0, min(y1, H))
        y2 = max(0.0, min(y2, H))
        cx = ((x1 + x2) / 2) / W
        cy = ((y1 + y2) / 2) / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H
        # Safety clip to [0,1] (handles floating point edge cases)
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        bw = min(max(bw, 0.0), 1.0)
        bh = min(max(bh, 0.0), 1.0)
        if bw <= 1e-4 or bh <= 1e-4:
            continue
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    out_txt.write_text("\n".join(lines))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc_root", required=True, help="dir with Annotations_new/*.xml and JPEGImages/*.jpg")
    ap.add_argument("--out_root", required=True, help="output root (labels/ will be created)")
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    voc_root = Path(args.voc_root)
    out_root = Path(args.out_root)
    ann_dir  = voc_root / "Annotations_new"
    img_dir  = voc_root / "JPEGImages"
    lb_dir   = out_root / "labels"
    lb_dir.mkdir(parents=True, exist_ok=True)

    xmls = sorted(ann_dir.glob("*.xml"))
    print(f"Found {len(xmls)} XML files")

    valid_stems = []
    for xml in xmls:
        stem = xml.stem
        img = img_dir / f"{stem}.jpg"
        if not img.exists():
            continue
        convert_one(xml, lb_dir / f"{stem}.txt")
        valid_stems.append(stem)
    print(f"Converted {len(valid_stems)} pairs")

    # Deterministic split → write file lists in YOLO format (relative paths from voc_root)
    random.Random(args.seed).shuffle(valid_stems)
    n_val = int(len(valid_stems) * args.val_frac)
    val   = valid_stems[:n_val]
    train = valid_stems[n_val:]

    with open(out_root / "train.txt", "w") as f:
        for s in train:
            f.write(f"./JPEGImages/{s}.jpg\n")
    with open(out_root / "val.txt", "w") as f:
        for s in val:
            f.write(f"./JPEGImages/{s}.jpg\n")
    print(f"train={len(train)} val={len(val)}  →  {out_root}/train.txt, val.txt")

    # YOLO expects labels at parallel path: replace 'JPEGImages' with 'labels' in path
    # Ultralytics handles this automatically when 'labels' dir is sibling of 'JPEGImages'.
    # Our labels are at out_root/labels (which equals voc_root/labels), so this works.

if __name__ == "__main__":
    main()
