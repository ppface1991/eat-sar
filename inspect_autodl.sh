#!/bin/bash
# inspect_autodl.sh
# 在 AutoDL 上运行，收集 eat-sar 工程当前结构信息。
# 用法：
#   cd 到 eat-sar 仓库根目录（即包含 scripts/、src/ 这种结构的地方）
#   bash inspect_autodl.sh > autodl_status.txt 2>&1
#   然后把 autodl_status.txt 发回来
#
# 脚本不修改任何文件，只读。

set +e   # 不要因为单条命令失败就退出

echo "========================================"
echo "EAT-SAR AutoDL Workspace Inspection"
echo "Generated: $(date)"
echo "Host: $(hostname)"
echo "Working dir: $(pwd)"
echo "========================================"

echo
echo "### 1. Repo root listing (top level) ###"
ls -lah . 2>&1 | head -40

echo
echo "### 2. Git status / branch / latest commits ###"
git rev-parse --abbrev-ref HEAD 2>&1
git log --oneline -10 2>&1
git status --short 2>&1 | head -30

echo
echo "### 3. Directory tree (depth 3, exclude data) ###"
if command -v tree &>/dev/null; then
    tree -L 3 -d -I '__pycache__|*.egg-info|node_modules|.git' . 2>&1 | head -80
else
    find . -maxdepth 3 -type d \
        ! -path '*/\.*' ! -path '*/__pycache__*' \
        2>&1 | sort | head -80
fi

echo
echo "### 4. Python / package environment ###"
python --version 2>&1
which python 2>&1
pip list 2>&1 | grep -iE "^(torch|ultralytics|opencv|numpy|pandas|matplotlib|pillow|pyyaml|scipy)" | head -20

echo
echo "### 5. GPU info ###"
nvidia-smi -L 2>&1 | head -5

echo
echo "### 6. Data YAML files (ultralytics dataset configs) ###"
echo "--- find data.yaml ---"
find . -maxdepth 6 -name "data.yaml" -o -name "*.yaml" 2>/dev/null \
    | grep -iE "sardet|sarship|ssdd|hrsid|data\.yaml" | head -20
echo "--- contents of each data.yaml found ---"
for f in $(find . -maxdepth 6 -name "data.yaml" 2>/dev/null | head -5); do
    echo "==> $f"
    cat "$f" 2>&1
    echo
done

echo
echo "### 7. Cached predictions / manifest CSVs ###"
echo "--- look for cache/preds and runs/ ---"
ls -la cache/ 2>&1 | head -20
ls -la cache/preds/ 2>&1 | head -20
ls -la runs/ 2>&1 | head -20
echo "--- find manifest.csv ---"
find . -maxdepth 6 -name "manifest.csv" 2>/dev/null | head -20
echo "--- find adaptive_T*.json ---"
find . -maxdepth 6 -name "adaptive_T*.json" 2>/dev/null | head -20
echo "--- find .npz prediction files (count by parent dir) ---"
find . -maxdepth 8 -name "*.npz" 2>/dev/null | \
    awk -F'/' '{NF--; print}' OFS='/' | sort -u | head -20
echo "--- count of .npz total ---"
find . -maxdepth 8 -name "*.npz" 2>/dev/null | wc -l

echo
echo "### 8. Manifest CSV column headers (first 3 lines each) ###"
for csv in $(find . -maxdepth 6 -name "manifest.csv" 2>/dev/null | head -5); do
    echo "==> $csv"
    head -3 "$csv" 2>&1
    echo "    (rows: $(wc -l < "$csv" 2>/dev/null) )"
    echo
done

echo
echo "### 9. Adaptive_T JSON contents ###"
for j in $(find . -maxdepth 6 -name "adaptive_T*.json" 2>/dev/null | head -5); do
    echo "==> $j"
    cat "$j" 2>&1
    echo
done

echo
echo "### 10. YOLO weights ###"
find . -maxdepth 6 \( -name "*.pt" -o -name "best.pt" -o -name "last.pt" \) \
    2>/dev/null | head -20

echo
echo "### 11. Image dataset sample (first few test images per dataset) ###"
echo "--- SARDet-100K test images sample ---"
find . -maxdepth 8 -type d -name "test" 2>/dev/null | head -10
echo "--- SARDet sardet images sample ---"
find . -maxdepth 8 -type d -iname "*sardet*" 2>/dev/null | head -10
echo "--- SAR-Ship images sample ---"
find . -maxdepth 8 -type d -iname "*sarship*" -o -iname "*sar-ship*" 2>/dev/null | head -10

echo
echo "### 12. Visualization scripts ###"
ls -la scripts/07_viz/ 2>&1
echo "--- check if plot_case_examples_v2.py exists ---"
ls -la scripts/07_viz/plot_case_examples_v2.py 2>&1
echo "--- check src/eat_sar/inference_cache.py ---"
ls -la src/eat_sar/inference_cache.py 2>&1
ls -la src/eat_sar/adaptive.py 2>&1

echo
echo "### 13. Disk usage of large dirs ###"
du -sh cache/ runs/ data/ 2>/dev/null | head -10
du -sh /root/autodl-tmp/ 2>/dev/null | head -5
df -h | head -5

echo
echo "========================================"
echo "Inspection complete."
echo "Please send back: autodl_status.txt"
echo "========================================"
