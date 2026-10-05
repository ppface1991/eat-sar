#!/usr/bin/env bash
# Retrain both YOLOv8s models on AutoDL (RTX 5090)
# Run from /autodl-fs/data/eat-sar
# Expected runtime: ~3h (SARDet) + ~4.5h (SAR-Ship) ≈ 7-8h total
# Outputs persist on permanent disk

set -euo pipefail

REPO=/autodl-fs/data/eat-sar
cd "$REPO"

# Make sure runs land on permanent disk
mkdir -p "$REPO/runs/train" "$REPO/cache/preds"

# 1) Reconstruct YOLO data yamls (paths point to permanent disk)
mkdir -p "$REPO/configs/yolo"

cat > "$REPO/configs/yolo/sardet100k.yaml" <<EOF
path: $REPO/SARDet-100K
train: Images/train
val: Images/val
test: Images/test
names:
  0: ship
  1: aircraft
  2: car
  3: tank
  4: bridge
  5: harbor
EOF
# NOTE: adjust class names if SARDet-100K has different categories
# Check via:  python -c "import json; d=json.load(open('$REPO/SARDet-100K/Annotations/train.json')); print([c['name'] for c in d['categories']])"

cat > "$REPO/configs/yolo/sarship.yaml" <<EOF
path: $REPO/SAR-Ship-Dataset
train: train.txt
val: val.txt
names:
  0: ship
EOF
# NOTE: SAR-Ship uses VOC XML. voc_to_yolo.py converts and writes train.txt/val.txt (90/10 split, seed=42).

# 2) Convert COCO json → YOLO txt for SARDet (if not done)
if [ ! -d "$REPO/SARDet-100K/labels/train" ]; then
  echo "[INFO] Converting SARDet COCO→YOLO..."
  python "$REPO/scripts/01_data/coco_to_yolo.py" \
    --coco_root "$REPO/SARDet-100K" \
    --out_root  "$REPO/SARDet-100K"
fi

# 3) Convert VOC XML → YOLO txt for SAR-Ship (if not done)
if [ ! -d "$REPO/SAR-Ship-Dataset/labels" ]; then
  echo "[INFO] Converting SAR-Ship VOC→YOLO..."
  python "$REPO/scripts/01_data/voc_to_yolo.py" \
    --voc_root  "$REPO/SAR-Ship-Dataset" \
    --out_root  "$REPO/SAR-Ship-Dataset"
fi

# 4) Train SARDet-100K (100 epochs, batch=16, img=640)
echo "==================== TRAINING SARDet-100K ===================="
yolo detect train \
  data="$REPO/configs/yolo/sardet100k.yaml" \
  model=yolov8s.pt \
  epochs=100 imgsz=640 batch=16 \
  project="$REPO/runs/train" name=sardet100k \
  device=0 workers=8 \
  patience=20 \
  pretrained=True \
  exist_ok=True \
  save=True save_period=20 \
  2>&1 | tee "$REPO/runs/train/sardet100k.log"

# 5) Train SAR-Ship (100 epochs, batch=32, img=512 -- smaller imgs)
echo "==================== TRAINING SAR-Ship ===================="
yolo detect train \
  data="$REPO/configs/yolo/sarship.yaml" \
  model=yolov8s.pt \
  epochs=100 imgsz=512 batch=32 \
  project="$REPO/runs/train" name=sarship \
  device=0 workers=8 \
  patience=20 \
  pretrained=True \
  exist_ok=True \
  save=True save_period=20 \
  2>&1 | tee "$REPO/runs/train/sarship.log"

echo ""
echo "==================== DONE ===================="
echo "Weights at:"
echo "  $REPO/runs/train/sardet100k/weights/best.pt"
echo "  $REPO/runs/train/sarship/weights/best.pt"
