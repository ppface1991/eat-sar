#!/usr/bin/env bash
# After retrain_both.sh finishes, run this to:
#   1) inference best.pt on test sets, save raw (pre-NMS) scores
#   2) re-fit adaptive thresholds (uses existing meta_with_env.csv — no recompute)
#   3) generate qualitative 3x3 grid figures (low/med/high clutter × Input+GT / Baseline / Ours)
# Output: case_examples_sardet.pdf, case_examples_sarship.pdf

set -euo pipefail
REPO=/autodl-fs/data/eat-sar
cd "$REPO"

SARDET_BEST="$REPO/runs/train/sardet100k/weights/best.pt"
SARSHIP_BEST="$REPO/runs/train/sarship/weights/best.pt"

[ -f "$SARDET_BEST"  ] || { echo "Missing $SARDET_BEST";  exit 1; }
[ -f "$SARSHIP_BEST" ] || { echo "Missing $SARSHIP_BEST"; exit 1; }

mkdir -p "$REPO/cache/preds/sardet100k_test"
mkdir -p "$REPO/cache/preds/sarship_val"
mkdir -p "$REPO/paper/figs"

# 1) Cache pre-NMS predictions (high recall: conf=0.01, iou_nms=0.6)
echo "==== Inference SARDet test ===="
python "$REPO/scripts/06_eval/cache_predictions.py" \
  --weights "$SARDET_BEST" \
  --img_dir "$REPO/SARDet-100K/Images/test" \
  --label_dir "$REPO/SARDet-100K/labels/test" \
  --out_dir "$REPO/cache/preds/sardet100k_test" \
  --conf 0.01 --iou 0.6 --imgsz 640

echo "==== Inference SAR-Ship val ===="
python "$REPO/scripts/06_eval/cache_predictions.py" \
  --weights "$SARSHIP_BEST" \
  --img_dir "$REPO/SAR-Ship-Dataset/JPEGImages" \
  --list "$REPO/SAR-Ship-Dataset/val.txt" \
  --label_dir "$REPO/SAR-Ship-Dataset/labels" \
  --out_dir "$REPO/cache/preds/sarship_val" \
  --conf 0.01 --iou 0.6 --imgsz 512

# 2) Re-fit adaptive thresholds (existing meta still valid — clutter_std is image-only)
echo "==== Refit adaptive thresholds ===="
python "$REPO/scripts/05_adaptive/fit_adaptive_threshold_v2.py" \
  --preds  "$REPO/cache/preds/sardet100k_test" \
  --meta   "$REPO/cache/meta/sardet100k_meta_with_env.csv" \
  --out    "$REPO/cache/preds/sardet100k_test/adaptive_T.json"

python "$REPO/scripts/05_adaptive/fit_adaptive_threshold_v2.py" \
  --preds  "$REPO/cache/preds/sarship_val" \
  --meta   "$REPO/cache/meta/sarship_meta_with_env.csv" \
  --out    "$REPO/cache/preds/sarship_val/adaptive_T.json"

# 3) Generate qualitative figures
echo "==== Plot qualitative 3x3 grids ===="
python "$REPO/scripts/07_viz/plot_case_examples_v2.py" \
  --dataset sardet \
  --preds   "$REPO/cache/preds/sardet100k_test" \
  --img_dir "$REPO/SARDet-100K/Images/test" \
  --meta    "$REPO/cache/meta/sardet100k_meta_with_env.csv" \
  --adaptive_T "$REPO/cache/preds/sardet100k_test/adaptive_T.json" \
  --T_baseline 0.40 \
  --out "$REPO/paper/figs/case_examples_sardet.pdf"

python "$REPO/scripts/07_viz/plot_case_examples_v2.py" \
  --dataset sarship \
  --preds   "$REPO/cache/preds/sarship_val" \
  --img_dir "$REPO/SAR-Ship-Dataset/JPEGImages" \
  --meta    "$REPO/cache/meta/sarship_meta_with_env.csv" \
  --adaptive_T "$REPO/cache/preds/sarship_val/adaptive_T.json" \
  --T_baseline 0.40 \
  --out "$REPO/paper/figs/case_examples_sarship.pdf"

echo ""
echo "==== DONE ===="
echo "Figures:"
ls -lh "$REPO/paper/figs/case_examples_"*.pdf
echo ""
echo "Copy back to local:"
echo "  scp -P <port> root@<host>:$REPO/paper/figs/case_examples_*.pdf  paper/figs/"
