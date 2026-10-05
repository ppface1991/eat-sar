"""
Train YOLOv8 (ship-only) on either SARDet-100K or SAR-Ship-Dataset.

Usage:
    python scripts/03_train/train_yolov8.py --config configs/default.yaml \\
        --data /root/autodl-tmp/eat-sar/yolo/sardet100k/data.yaml \\
        --name sardet100k_yolov8s
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from eat_sar.utils import load_config, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", required=True, help="path to YOLO data.yaml")
    ap.add_argument("--name", required=True, help="run name")
    ap.add_argument("--model", default=None, help="override base model (.pt)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])

    from ultralytics import YOLO

    model_path = args.model or cfg["train"]["model"]
    model = YOLO(model_path)

    model.train(
        data=args.data,
        imgsz=cfg["train"]["imgsz"],
        epochs=cfg["train"]["epochs"],
        batch=cfg["train"]["batch"],
        optimizer=cfg["train"]["optimizer"],
        lr0=cfg["train"]["lr0"],
        patience=cfg["train"]["patience"],
        single_cls=cfg["train"]["single_cls"],
        project=cfg["paths"]["runs_dir"],
        name=args.name,
        seed=cfg["project"]["seed"],
        device=cfg["project"]["device"],
        workers=cfg["project"]["workers"],
        exist_ok=True,
        verbose=True,
    )

    # Final val on test split if defined
    print("\n[final val on test]")
    model.val(
        data=args.data,
        split="test",
        imgsz=cfg["train"]["imgsz"],
        batch=cfg["train"]["batch"],
        project=cfg["paths"]["runs_dir"],
        name=f"{args.name}_test",
        device=cfg["project"]["device"],
    )


if __name__ == "__main__":
    main()
