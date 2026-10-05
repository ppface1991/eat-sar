#!/usr/bin/env bash
# Prepare SAR-Ship-Dataset for YOLO training and start training.
# Runs entirely in /root/autodl-tmp/eat-sar-work/sarship_root (physical layout, no symlinks).
# Output: /root/autodl-tmp/eat-sar-work/runs/train/sarship/weights/best.pt
#
# Prerequisite: SARDet training must be DONE (or stopped) — single GPU.

set -euo pipefail

REPO=/autodl-fs/data/eat-sar
ROOT=/root/autodl-tmp/eat-sar-work/sarship_root
mkdir -p "$ROOT/images" "$ROOT/labels"

# 1) Copy JPEGImages (476MB) to physical /images dir on temp disk
if [ ! -d "$ROOT/images/all" ]; then
  echo "[1/4] Copying 43820 jpg → $ROOT/images/all ..."
  cp -r "$REPO/SAR-Ship-Dataset/JPEGImages" "$ROOT/images/all"
fi

# 2) Convert VOC XML → YOLO txt (lands in $ROOT/labels/all)
mkdir -p "$ROOT/labels/all"
if [ "$(ls "$ROOT/labels/all" 2>/dev/null | wc -l)" -lt 40000 ]; then
  echo "[2/4] Converting VOC XML → YOLO txt → $ROOT/labels/all ..."
  python "$REPO/scripts/01_data/voc_to_yolo.py" \
    --voc_root "$REPO/SAR-Ship-Dataset" \
    --out_root "$ROOT/_voc_tmp" \
    --val_frac 0.10 --seed 42
  # voc_to_yolo writes to <out_root>/labels/*.txt and train.txt/val.txt
  find "$ROOT/_voc_tmp/labels" -maxdepth 1 -name "*.txt" -exec mv -t "$ROOT/labels/all/" {} +
  mv "$ROOT/_voc_tmp/train.txt"   "$ROOT/train.txt"
  mv "$ROOT/_voc_tmp/val.txt"     "$ROOT/val.txt"
  rm -rf "$ROOT/_voc_tmp"
fi

# 3) Rewrite train.txt/val.txt to point to the COPY on temp disk
echo "[3/4] Rewriting list files to absolute temp-disk paths..."
sed -i "s|./JPEGImages|$ROOT/images/all|g" "$ROOT/train.txt"
sed -i "s|./JPEGImages|$ROOT/images/all|g" "$ROOT/val.txt"

# Make ultralytics find labels: it replaces /images/ with /labels/, so we need
# the image paths to contain '/images/' and labels to be at the parallel '/labels/' location.
# We already have $ROOT/images/all/*.jpg and $ROOT/labels/all/*.txt → parallel. Good.

head -3 "$ROOT/train.txt"
wc -l "$ROOT/train.txt" "$ROOT/val.txt"

# 4) Write yolo data yaml
mkdir -p "$REPO/configs/yolo"
cat > "$REPO/configs/yolo/sarship.yaml" <<EOF
path: $ROOT
train: train.txt
val: val.txt
names:
  0: ship
EOF
cat "$REPO/configs/yolo/sarship.yaml"

# 5) Train (100 epochs, batch=32, img=512)
echo "[4/4] Starting SAR-Ship training (100 epochs)..."
cd "$REPO"
yolo detect train \
  data="$REPO/configs/yolo/sarship.yaml" \
  model=yolov8s.pt \
  epochs=100 imgsz=512 batch=32 \
  project=/root/autodl-tmp/eat-sar-work/runs/train name=sarship \
  device=0 workers=8 \
  patience=20 \
  pretrained=True exist_ok=True \
  save=True save_period=20 \
  2>&1 | tee /root/autodl-tmp/eat-sar-work/runs/train/sarship.log
