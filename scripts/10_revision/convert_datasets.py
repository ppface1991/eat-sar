"""
Step W1 — One-shot dataset conversion for the revision experiments (Windows /
local). Replaces the July per-dataset conversion + produces deterministic
split lists consistent with what the paper describes.

Inputs (your original dataset folders, exactly as downloaded from AutoDL):
  SAR-Ship-Dataset/  Annotations_new/*.xml  +  JPEGImages/*.jpg
  SARDet-100K/        Annotations/{train,val,test}.json + Images/{train,val,test}/

Outputs (under --out-root):
  sarship/labels/all/<stem>.txt        YOLO labels, single class 0
  sarship/{train,val,test}.txt         absolute image paths, 80/10/10, seed 42
  sardet/labels/{split}/<stem>.txt     YOLO labels, all classes, contiguous ids
  sardet/{train,val,test}.txt          absolute paths of the SHIP-SUBSET images
  sardet/class_map.json                names list + ship class id
  conversion_summary.json              counts for the paper

Notes
  * SAR-Ship split: sorted stems, random.Random(42) shuffle, 10% test then
    10% val, rest train (deterministic; documented in the paper as such).
  * SARDet uses the official train/val/test image dirs; lists contain only
    images with >=1 ship annotation (the "ship subset" of the paper), which
    also halves the CPU caching time.
  * SAR-Ship: every object is converted to class 0 regardless of its VOC
    name (the dataset is single-class).

Usage:
  python scripts\\10_revision\\convert_datasets.py ^
      --sarship-root "D:\\...\\SAR-Ship-Dataset" ^
      --sardet-root  "D:\\...\\SARDet-100K" ^
      --out-root     "D:\\...\\work\\yolo"
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

SHIP_NAMES = {"ship", "ships", "boat", "vessel"}


# --------------------------------------------------------------------------- #
# SAR-Ship-Dataset (VOC)
# --------------------------------------------------------------------------- #
def convert_sarship(root: Path, out: Path, seed: int, val_frac: float,
                    test_frac: float) -> dict:
    ann_dir, img_dir = root / "Annotations_new", root / "JPEGImages"
    lb_dir = out / "labels" / "all"
    lb_dir.mkdir(parents=True, exist_ok=True)

    stems = []
    for xml in sorted(ann_dir.glob("*.xml")):
        stem = xml.stem
        if not (img_dir / f"{stem}.jpg").exists():
            continue
        r = ET.parse(xml).getroot()
        size = r.find("size")
        W = int(size.find("width").text)
        H = int(size.find("height").text)
        lines = []
        for obj in r.iter("object"):
            b = obj.find("bndbox")
            x1 = max(0.0, min(float(b.find("xmin").text), W))
            x2 = max(0.0, min(float(b.find("xmax").text), W))
            y1 = max(0.0, min(float(b.find("ymin").text), H))
            y2 = max(0.0, min(float(b.find("ymax").text), H))
            bw, bh = (x2 - x1) / W, (y2 - y1) / H
            if bw <= 1e-4 or bh <= 1e-4:
                continue
            lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                         f"{bw:.6f} {bh:.6f}")
        (lb_dir / f"{stem}.txt").write_text("\n".join(lines))
        stems.append(stem)

    random.Random(seed).shuffle(stems)
    n_test = int(len(stems) * test_frac)
    n_val = int(len(stems) * val_frac)
    parts = dict(test=stems[:n_test], val=stems[n_test:n_test + n_val],
                 train=stems[n_test + n_val:])
    for split, ss in parts.items():
        with open(out / f"{split}.txt", "w") as f:
            for s in ss:
                f.write(str((img_dir / f"{s}.jpg").resolve()) + "\n")
    return dict(n_total=len(stems), split_sizes={k: len(v) for k, v in parts.items()},
                labels_dir=str(lb_dir), seed=seed,
                split_rule=f"sorted stems, Random({seed}) shuffle, "
                           f"{test_frac:.0%} test then {val_frac:.0%} val")


# --------------------------------------------------------------------------- #
# SARDet-100K (COCO)
# --------------------------------------------------------------------------- #
def convert_sardet(root: Path, out: Path) -> dict:
    ann_dir = root / "Annotations"
    summary = dict(split_sizes={}, ship_subset_sizes={}, labels_dirs={})
    names, ship_id = None, None

    for split in ("train", "val", "test"):
        j = ann_dir / f"{split}.json"
        img_dir = root / "Images" / split
        lb_dir = out / "labels" / split
        if not j.exists():
            # tolerate alternative naming like Annotations/<split>_.json
            cands = [p for p in ann_dir.glob("*.json")
                     if split in p.stem.lower()]
            if not cands:
                print(f"[skip] no COCO json for split '{split}' under {ann_dir}")
                continue
            j = cands[0]
        lb_dir.mkdir(parents=True, exist_ok=True)
        d = json.load(open(j, encoding="utf-8"))

        cats = sorted(d["categories"], key=lambda c: c["id"])
        cat2yolo = {c["id"]: i for i, c in enumerate(cats)}
        if names is None:
            names = [c["name"] for c in cats]
            ship_id = next((i for i, n in enumerate(names)
                            if n.strip().lower() in SHIP_NAMES), None)
            print(f"  categories: {list(enumerate(names))}")
            print(f"  ship class id = {ship_id}")

        anns = defaultdict(list)
        for a in d["annotations"]:
            anns[a["image_id"]].append(a)

        n_all, n_ship = 0, 0
        ship_list = open(out / f"{split}.txt", "w")
        for im in d["images"]:
            fname = Path(im["file_name"]).name
            stem = Path(fname).stem
            p = (img_dir / Path(im["file_name"])).resolve()
            W, H = im.get("width", 0), im.get("height", 0)
            lines, has_ship = [], False
            for a in anns.get(im["id"], []):
                x, y, w, h = a["bbox"]
                if W <= 0 or H <= 0:
                    continue
                bw, bh = w / W, h / H
                if bw <= 0 or bh <= 0:
                    continue
                cls = cat2yolo[a["category_id"]]
                if cls == ship_id:
                    has_ship = True
                lines.append(f"{cls} {(x + w / 2) / W:.6f} "
                             f"{(y + h / 2) / H:.6f} {bw:.6f} {bh:.6f}")
            (lb_dir / f"{stem}.txt").write_text("\n".join(lines))
            n_all += 1
            if has_ship and p.exists():
                ship_list.write(str(p) + "\n")
                n_ship += 1
        ship_list.close()
        summary["split_sizes"][split] = n_all
        summary["ship_subset_sizes"][split] = n_ship
        summary["labels_dirs"][split] = str(lb_dir)
        print(f"  [{split}] images={n_all} ship-subset={n_ship}")

    (out / "class_map.json").write_text(json.dumps(
        dict(names=names, ship_class_id=ship_id), indent=2))
    summary["names"] = names
    summary["ship_class_id"] = ship_id
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sarship-root", required=True)
    ap.add_argument("--sardet-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    args = ap.parse_args()

    out = Path(args.out_root)
    print("=== SAR-Ship-Dataset (VOC -> YOLO) ===")
    ss = out / "sarship"
    ss.mkdir(parents=True, exist_ok=True)
    sarship = convert_sarship(Path(args.sarship_root), ss, args.seed,
                              args.val_frac, args.test_frac)
    print(f"  total={sarship['n_total']}  splits={sarship['split_sizes']}")

    print("=== SARDet-100K (COCO -> YOLO) ===")
    sd = out / "sardet"
    sd.mkdir(parents=True, exist_ok=True)
    sardet = convert_sardet(Path(args.sardet_root), sd)

    summary = dict(sarship=sarship, sardet=sardet)
    (out / "conversion_summary.json").write_text(json.dumps(summary, indent=2))
    # cmd.exe helper for run_revision.cmd
    (out / "env.cmd").write_text(f"set SHIP_ID={sardet['ship_class_id']}\n")
    print(f"\n[saved] {out / 'conversion_summary.json'}")
    print("\nNEXT: run cache_preds.py for the 4 needed caches "
          "(sarship val+test, sardet val+test ship subsets).")
    print(f"SARDet ship --target-class = {sardet['ship_class_id']}")


if __name__ == "__main__":
    main()
