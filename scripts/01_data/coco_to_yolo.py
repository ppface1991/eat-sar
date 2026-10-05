#!/usr/bin/env python3
"""Convert SARDet-100K COCO JSON to YOLO txt format.

Layout produced (alongside Images/):
  labels/train/<stem>.txt
  labels/val/<stem>.txt
  labels/test/<stem>.txt
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image

def convert_split(coco_json: Path, img_dir: Path, label_dir: Path):
    label_dir.mkdir(parents=True, exist_ok=True)
    with open(coco_json) as f:
        d = json.load(f)

    # Build cat_id -> contiguous YOLO id (0..N-1) following category order in json
    cats = sorted(d["categories"], key=lambda c: c["id"])
    cat2yolo = {c["id"]: i for i, c in enumerate(cats)}
    print(f"  Categories: {[(i,c['name']) for i,c in enumerate(cats)]}")

    img_by_id = {im["id"]: im for im in d["images"]}
    anns_by_img = defaultdict(list)
    for a in d["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    n_written = 0
    for img_id, im in img_by_id.items():
        fname = Path(im["file_name"]).name
        stem = Path(fname).stem
        W, H = im["width"], im["height"]
        if W <= 0 or H <= 0:
            # fallback: open image
            p = img_dir / fname
            if not p.exists():
                continue
            with Image.open(p) as I:
                W, H = I.size

        lines = []
        for a in anns_by_img.get(img_id, []):
            x, y, w, h = a["bbox"]
            cx = (x + w/2) / W
            cy = (y + h/2) / H
            bw = w / W
            bh = h / H
            if bw <= 0 or bh <= 0:
                continue
            cls = cat2yolo[a["category_id"]]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        (label_dir / f"{stem}.txt").write_text("\n".join(lines))
        n_written += 1
    print(f"  Wrote {n_written} label files to {label_dir}")
    return [c["name"] for c in cats]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", required=True, help="dir with Annotations/{train,val,test}.json and Images/{train,val,test}")
    ap.add_argument("--out_root",  required=True, help="output root (labels/ will be created here)")
    args = ap.parse_args()

    coco_root = Path(args.coco_root)
    out_root  = Path(args.out_root)

    names = None
    for split in ["train", "val", "test"]:
        j = coco_root / "Annotations" / f"{split}.json"
        im_dir = coco_root / "Images" / split
        lb_dir = out_root  / "labels" / split
        if not j.exists():
            print(f"[skip] {j} not found")
            continue
        print(f"[{split}] {j}")
        names = convert_split(j, im_dir, lb_dir)

    if names:
        print("\nUpdate configs/yolo/sardet100k.yaml 'names:' to:")
        for i, n in enumerate(names):
            print(f"  {i}: {n}")

if __name__ == "__main__":
    main()
