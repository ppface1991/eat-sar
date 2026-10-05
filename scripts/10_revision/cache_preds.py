"""
Step W2 — Cache YOLOv8 predictions to per-image .npz (Windows-friendly).

Same output format as scripts/06_eval/cache_predictions.py
(boxes, scores, classes, img_w, img_h) but:
  * --device defaults to "cpu"
  * list entries may be full paths or bare names with ANY extension
    (the original script hard-coded .jpg)
  * skips images whose .npz already exists (resumable)

Usage:
  python scripts\\10_revision\\cache_preds.py ^
      --weights runs_backup\\sarship\\weights\\best.pt ^
      --img-dir  "D:\\...\\SAR-Ship-Dataset\\JPEGImages" ^
      --list     "D:\\...\\work\\yolo\\sarship\\test.txt" ^
      --out-dir  "D:\\...\\work\\cache_preds\\sarship_test" ^
      --device cpu
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SUFFIXES = ["", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]


def resolve(p_txt: str, img_dir: Path) -> Path | None:
    p = Path(p_txt.strip())
    if p.is_absolute():
        if p.exists():
            return p
        return None
    for suf in SUFFIXES:
        c = img_dir / (p if p.suffix else Path(p).stem + suf)
        if c.exists():
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--list", default=None,
                    help="txt file with image paths/names (one per line)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    from tqdm import tqdm
    from ultralytics import YOLO

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.list:
        entries = [l for l in open(args.list, encoding="utf-8") if l.strip()]
        imgs = [r for r in (resolve(e, img_dir) for e in entries) if r]
        skipped = len(entries) - len(imgs)
        if skipped:
            print(f"[warn] {skipped} list entries could not be resolved in {img_dir}")
    else:
        imgs = sorted([p for p in img_dir.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png",
                                               ".tif", ".tiff", ".bmp"}])

    todo = [p for p in imgs if not (out_dir / f"{p.stem}.npz").exists()]
    print(f"{len(imgs)} images, {len(imgs) - len(todo)} already cached, "
          f"{len(todo)} to go  ->  {out_dir}")

    model = YOLO(args.weights)
    for bs in tqdm(range(0, len(todo), args.batch)):
        batch = todo[bs:bs + args.batch]
        results = model.predict([str(p) for p in batch], conf=args.conf,
                                 iou=args.iou, imgsz=args.imgsz,
                                 device=args.device, verbose=False)
        for p, r in zip(batch, results):
            boxes = (r.boxes.xyxy.cpu().numpy().astype(np.float32)
                     if r.boxes is not None else np.zeros((0, 4), np.float32))
            scores = (r.boxes.conf.cpu().numpy().astype(np.float32)
                      if r.boxes is not None else np.zeros((0,), np.float32))
            classes = (r.boxes.cls.cpu().numpy().astype(np.int32)
                       if r.boxes is not None else np.zeros((0,), np.int32))
            H, W = r.orig_shape
            np.savez_compressed(out_dir / f"{p.stem}.npz", boxes=boxes,
                                scores=scores, classes=classes,
                                img_w=W, img_h=H)
    print(f"Done. {len(list(out_dir.glob('*.npz')))} .npz in {out_dir}")


if __name__ == "__main__":
    main()
