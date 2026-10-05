#!/bin/bash
# inspect_autodl_v2.sh
# 深入诊断：检查 (1) ultralytics 是否装好 (2) 关键脚本能否 import (3)
# data.yaml 是否需要现写 (4) cache/meta 里有什么残留可用
# 用法（在 /autodl-fs/data/eat-sar 下）：
#   bash inspect_autodl_v2.sh > autodl_status_v2.txt 2>&1

set +e

echo "========================================"
echo "EAT-SAR Deep Inspection v2"
echo "Date: $(date)"
echo "PWD: $(pwd)"
echo "========================================"

echo
echo "### A. ultralytics 是否装好 ###"
python -c "import ultralytics; print('ultralytics:', ultralytics.__version__)" 2>&1
python -c "from ultralytics import YOLO; print('YOLO import OK')" 2>&1

echo
echo "### B. 关键源码模块能否 import ###"
cd "$(dirname "$0")/.." 2>/dev/null  # 切到 eat-sar repo 根
echo "PWD after cd: $(pwd)"
PYTHONPATH="$(pwd)/src:$PYTHONPATH" python -c "
import sys
sys.path.insert(0, 'src')
try:
    from eat_sar import inference_cache, adaptive
    print('inference_cache, adaptive import OK')
    print('inference_cache funcs:', [x for x in dir(inference_cache) if not x.startswith('_')])
    print('adaptive funcs:', [x for x in dir(adaptive) if not x.startswith('_')])
except Exception as e:
    print('IMPORT ERR:', e)
" 2>&1

echo
echo "### C. cache/meta 残留 (可能还有 clutter_std 等元数据) ###"
ls -la cache/meta/ 2>&1
for f in cache/meta/*.csv cache/meta/*.json cache/meta/*.parquet 2>/dev/null; do
    [ -e "$f" ] || continue
    echo "--- $f ---"
    if [[ "$f" == *.csv ]]; then
        head -3 "$f"
        echo "  rows: $(wc -l < "$f")"
    else
        head -c 500 "$f"
        echo
    fi
done

echo
echo "### D. 测试集图像数量 ###"
echo "SARDet-100K test:"
ls SARDet-100K/Images/test 2>/dev/null | wc -l
echo "SAR-Ship-Dataset JPEGImages:"
ls SAR-Ship-Dataset/JPEGImages 2>/dev/null | wc -l
echo "SAR-Ship-Dataset Annotations_new:"
ls SAR-Ship-Dataset/Annotations_new 2>/dev/null | wc -l

echo
echo "### E. SARDet labels 看一眼 (确认 YOLO format) ###"
find SARDet-100K -maxdepth 4 -name 'labels' -type d 2>/dev/null | head
find SARDet-100K -maxdepth 5 -name '*.txt' 2>/dev/null | head -3
echo "--- sample SARDet label content ---"
SAMPLE=$(find SARDet-100K -maxdepth 5 -name '*.txt' 2>/dev/null | head -1)
if [ -n "$SAMPLE" ]; then
    echo "==> $SAMPLE"
    head -3 "$SAMPLE"
fi

echo
echo "### F. SAR-Ship labels 看一眼 ###"
ls SAR-Ship-Dataset/Annotations_new 2>/dev/null | head -3
SAMPLE2=$(find SAR-Ship-Dataset/Annotations_new -name '*.txt' 2>/dev/null | head -1)
if [ -n "$SAMPLE2" ]; then
    echo "==> $SAMPLE2"
    head -3 "$SAMPLE2"
fi
SAMPLE3=$(find SAR-Ship-Dataset/Annotations_new -name '*.xml' 2>/dev/null | head -1)
if [ -n "$SAMPLE3" ]; then
    echo "==> $SAMPLE3 (XML PASCAL VOC format)"
    head -10 "$SAMPLE3"
fi

echo
echo "### G. configs/ 里有什么 ###"
ls -la configs/ 2>&1
for f in configs/*.yaml configs/*.yml configs/*.json 2>/dev/null; do
    [ -e "$f" ] || continue
    echo "==> $f"
    head -30 "$f"
    echo
done

echo
echo "### H. scripts/01_data 看下数据划分脚本 ###"
ls -la scripts/01_data/ 2>&1
echo "--- find any prepare/split/yolo-config script ---"
ls scripts/01_data/*.py 2>/dev/null

echo
echo "### I. scripts/03_train 训练入口 (找训练时用过的 data.yaml) ###"
ls -la scripts/03_train/ 2>&1
grep -l "data.yaml\|data=\|yaml" scripts/03_train/*.py 2>/dev/null | head -3
echo "--- search whole repo for *.yaml that look like ultralytics data configs ---"
find . -maxdepth 6 -name "*.yaml" 2>/dev/null | grep -vE "configs_old|scripts_old|src_old|/eat-sar/" | head -20
for f in $(find . -maxdepth 6 -name "*.yaml" 2>/dev/null | grep -vE "configs_old|scripts_old|src_old|/eat-sar/" | head -10); do
    if grep -qE "^(path|train|val|test|names):" "$f" 2>/dev/null; then
        echo "==> $f (looks like ultralytics data config)"
        cat "$f"
        echo
    fi
done

echo
echo "### J. scripts/04_threshold + 05_adaptive 看会跑什么 ###"
ls scripts/04_threshold/ 2>/dev/null
ls scripts/05_adaptive/ 2>/dev/null

echo
echo "### K. autodl-tmp 是否能写 / 大小 ###"
mkdir -p /root/autodl-tmp/test_write 2>&1
touch /root/autodl-tmp/test_write/ok 2>&1 && echo "autodl-tmp writable: YES" && rm /root/autodl-tmp/test_write/ok && rmdir /root/autodl-tmp/test_write
df -h /root/autodl-tmp 2>&1

echo
echo "========================================"
echo "Done. Send back: autodl_status_v2.txt"
echo "========================================"
