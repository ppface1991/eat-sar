#!/bin/bash
set -e

cd /autodl-fs/data/eat-sar
export PYTHONPATH=/autodl-fs/data/eat-sar/src
WORK=/root/autodl-tmp/eat-sar-work
RUN_DIR=$WORK/runs/sarship_test
CACHE_DIR=$WORK/cache_preds/sarship_test
TRAIN_DIR=$WORK/runs/train/sarship
mkdir -p $RUN_DIR $CACHE_DIR

echo ""
echo "========== Step 0: Verify training finished =========="
ls -la $TRAIN_DIR/weights/best.pt
cp $TRAIN_DIR/weights/best.pt /autodl-fs/data/eat-sar/runs_backup/sarship_best.pt
echo "best.pt backed up to /autodl-fs/data/eat-sar/runs_backup/sarship_best.pt"

echo ""
echo "========== Step 1: Cache predictions on val.txt =========="
# val.txt has full paths; we need to build a temporary "test" image dir reference
# Use the same script as SARDet but point at images/all
python scripts/06_eval/cache_predictions.py \
    --weights $TRAIN_DIR/weights/best.pt \
    --img_dir $WORK/sarship_root/images/all \
    --list $WORK/sarship_root/val.txt \
    --out_dir $CACHE_DIR --label_dir $WORK/sarship_root/labels/all \
    --conf 0.01 --iou 0.6 --imgsz 512 2>&1 | tail -20

echo ""
echo "========== Step 2: Build manifest =========="
python scripts/05_adaptive/build_manifest_from_cache.py \
    --out_dir $CACHE_DIR --label_dir $WORK/sarship_root/labels/all \
    --img_dir $WORK/sarship_root/images/all \
    --meta-csv /autodl-fs/data/eat-sar/cache/meta/sarship_meta_with_env.csv \
    --out-csv $CACHE_DIR/manifest.csv

echo ""
echo "========== Step 3: Rename calm/moderate/rough -> low/medium/high =========="
python - << 'PYEOF'
import pandas as pd, json
p = "/root/autodl-tmp/eat-sar-work/cache_preds/sarship_test/manifest.csv"
mf = pd.read_csv(p)
m = {"calm": "low", "moderate": "medium", "rough": "high"}
mf["wind_bin"] = mf["wind_bin"].map(m)
mf.to_csv(p, index=False)
print("wind_bin counts:", mf["wind_bin"].value_counts().to_dict())

# Make a low/medium/high-labeled bins.json for fit_adaptive_simple.py
src = json.load(open("/autodl-fs/data/eat-sar/cache/meta/sarship_meta_bins.json"))
src["labels"] = ["low", "medium", "high"]
json.dump(src, open("/autodl-fs/data/eat-sar/cache/meta/sarship_meta_bins_lmh.json", "w"), indent=2)
print("Wrote bins_lmh.json")
PYEOF

echo ""
echo "========== Step 4: Fit adaptive thresholds =========="
# Build a fake data.yaml that points label_root for plot_case_examples_v2.py
cat > $WORK/sarship_root/data_test.yaml << 'YAMLEOF'
path: /root/autodl-tmp/eat-sar-work/sarship_root
train: train.txt
val: val.txt
test: val.txt
names:
  0: ship
YAMLEOF
# Note: label_root is computed as path/labels/<split>. We need labels/test/ to exist.
# Quick fix: symlink labels/test -> labels/all
if [ ! -e $WORK/sarship_root/labels/test ]; then
    ln -s $WORK/sarship_root/labels/all $WORK/sarship_root/labels/test
fi

python scripts/05_adaptive/fit_adaptive_simple.py \
    --manifest $CACHE_DIR/manifest.csv \
    --label-root $WORK/sarship_root/labels/all \
    --env-key wind_bin --env-value-key clutter_std \
    --bin-order low,medium,high \
    --bin-boundaries-json /autodl-fs/data/eat-sar/cache/meta/sarship_meta_bins_lmh.json \
    --target-class 0 \
    --out-json $RUN_DIR/adaptive_T.json

echo ""
echo "========== Step 5: Plot qualitative figure =========="
python scripts/07_viz/plot_case_examples_v2.py \
    --manifest $CACHE_DIR/manifest.csv \
    --data $WORK/sarship_root/data_test.yaml \
    --split test \
    --adaptive-json $RUN_DIR/adaptive_T.json \
    --baseline-T 0.40 \
    --auto-discriminative \
    --target-class 0 \
    --out /autodl-fs/data/eat-sar/paper/figs/case_examples_sarship.pdf \
    --dataset-label "SAR-Ship"

echo ""
echo "========== Done! =========="
ls -la /autodl-fs/data/eat-sar/paper/figs/case_examples_sarship.pdf
