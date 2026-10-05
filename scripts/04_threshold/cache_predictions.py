"""
Run trained YOLOv8 once at low conf threshold, cache predictions per image.

Usage:
    python scripts/04_threshold/cache_predictions.py --config configs/default.yaml \\
        --weights runs/sardet100k_yolov8s/weights/best.pt \\
        --data /root/autodl-tmp/eat-sar/yolo/sardet100k/data.yaml \\
        --split val \\
        --meta-csv cache/meta/sardet100k_meta_with_env.csv \\
        --out cache/preds/sardet100k_val
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from eat_sar.inference_cache import save_pred  # noqa: E402
from eat_sar.utils import ensure_dir, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--meta-csv", required=True, help="meta CSV with env columns")
    ap.add_argument("--out", required=True, help="output cache dir")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6, help="NMS IoU (kept fixed)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_dir(args.out)

    data_yaml = yaml.safe_load(open(args.data))
    img_root = Path(data_yaml["path"]) / data_yaml[args.split]
    images = sorted([p for p in img_root.iterdir()
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")])
    print(f"Found {len(images)} images in {img_root}")

    # filter to those with env info
    meta = pd.read_csv(args.meta_csv)
    meta["stem"] = meta["file_name"].apply(lambda s: Path(s).stem)
    keep_stems = set(meta["stem"].tolist())
    images = [p for p in images if p.stem in keep_stems]
    print(f"After env filter: {len(images)} images")

    from ultralytics import YOLO
    model = YOLO(args.weights)

    manifest = []
    bs = 16
    for i in tqdm(range(0, len(images), bs), desc="inference"):
        batch = images[i:i + bs]
        results = model.predict(
            source=[str(p) for p in batch],
            conf=args.conf, iou=args.iou,
            imgsz=cfg["train"]["imgsz"],
            device=cfg["project"]["device"],
            verbose=False,
        )
        for p, r in zip(batch, results):
            if r.boxes is None or r.boxes.xyxy is None:
                boxes = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros((0,), dtype=np.float32)
                classes = np.zeros((0,), dtype=np.int32)
            else:
                boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                scores = r.boxes.conf.cpu().numpy().astype(np.float32)
                classes = r.boxes.cls.cpu().numpy().astype(np.int32)
            save_pred(out_dir / f"{p.stem}.npz", boxes, scores, classes)
            manifest.append({
                "stem": p.stem,
                "image_path": str(p),
                "pred_npz": str(out_dir / f"{p.stem}.npz"),
            })

    mf = pd.DataFrame(manifest)
    mf = mf.merge(meta, on="stem", how="left")
    mf_csv = Path(args.out) / "manifest.csv"
    mf.to_csv(mf_csv, index=False)
    print(f"Saved manifest: {mf_csv}")


if __name__ == "__main__":
    main()
