#!/usr/bin/env python3
"""Cache YOLOv8 pre-NMS predictions to per-image .npz files.

Each .npz file:
  boxes   : (N,4)  xyxy in pixels
  scores  : (N,)
  classes : (N,)
  img_w, img_h : ints

Per-image GT (YOLO format) is also bundled into label.txt-equivalent .npy if available,
but src/eat_sar/inference_cache.py already supports load_yolo_gt() so we keep separation.
"""
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--img_dir", required=True, help="root of images")
    ap.add_argument("--list", default=None, help="optional file containing relative image paths (one per line) — for SAR-Ship val")
    ap.add_argument("--label_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--iou",  type=float, default=0.6)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.list:
        with open(args.list) as f:
            stems = [Path(l.strip()).stem for l in f if l.strip()]
        imgs = [img_dir / f"{s}.jpg" for s in stems]
        imgs = [p for p in imgs if p.exists()]
    else:
        imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

    print(f"Loading {args.weights}")
    model = YOLO(args.weights)

    print(f"Predicting {len(imgs)} images → {out_dir}")
    for batch_start in tqdm(range(0, len(imgs), 32)):
        batch = imgs[batch_start: batch_start + 32]
        results = model.predict(
            [str(p) for p in batch],
            conf=args.conf, iou=args.iou, imgsz=args.imgsz,
            device=args.device, verbose=False, stream=False
        )
        for p, r in zip(batch, results):
            boxes  = r.boxes.xyxy.cpu().numpy().astype(np.float32) if r.boxes is not None else np.zeros((0,4), np.float32)
            scores = r.boxes.conf.cpu().numpy().astype(np.float32) if r.boxes is not None else np.zeros((0,),  np.float32)
            classes= r.boxes.cls.cpu().numpy().astype(np.int32)    if r.boxes is not None else np.zeros((0,),  np.int32)
            H, W = r.orig_shape
            np.savez_compressed(
                out_dir / f"{p.stem}.npz",
                boxes=boxes, scores=scores, classes=classes,
                img_w=W, img_h=H
            )

    print(f"Done. Wrote {len(imgs)} .npz files.")

if __name__ == "__main__":
    main()
