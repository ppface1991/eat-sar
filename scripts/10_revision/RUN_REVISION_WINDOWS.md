# GRSL-01920-2026 二审补实验 — Windows 本地运行手册（v3，按体检结果定制）

针对体检结果定制：原始数据集、两个 best.pt 都在仓库根 `D:\Project\eat-sar\eat-sar\eat-sar`
下；缺 npz 缓存和 YOLO 标签树 → 用本 kit 的 `convert_datasets.py` + `cache_preds.py`
本地重建。CPU（无 CUDA）即可跑完全部，只有推理缓存耗时较长。

## 0. 每次开新终端先执行

```powershell
cd D:\Project\eat-sar\eat-sar\eat-sar     # 仓库根（含 scripts\、src\）
conda activate eat-sar
$env:PYTHONPATH = "$PWD\src"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [Text.Encoding]::UTF8

# 固定路径变量（本手册通用）
$REPO   = "D:\Project\eat-sar\eat-sar\eat-sar"
$WORK   = "D:\Project\eat-sar\work"
$SSROOT = "$REPO\SAR-Ship-Dataset"
$SDROOT = "$REPO\SARDet-100K"
$W_SS   = "$REPO\runs_backup\sarship\weights\best.pt"
$W_SD   = "$REPO\runs_backup\sardet100k\best.pt"
$Y      = "$WORK\yolo"; $SS = "$Y\sarship"; $SD = "$Y\sardet"
$OUT    = "$REPO\revision_out"
New-Item -ItemType Directory -Force $WORK, $OUT | Out-Null
```

解压 `revision_kit_v3.zip` 到 `$REPO`（得到 `scripts\10_revision\`）。

## 1. 数据转换（CPU，约 5–15 分钟）

SAR-Ship：VOC → YOLO + 80/10/10 确定性划分（seed 42，与论文口径一致）。
SARDet：COCO → YOLO（保留全部 6 类、类别顺序与 7 月训练一致），并生成
**ship 子集**图像清单（论文口径；顺带把推理量减半）。

```powershell
python scripts\10_revision\convert_datasets.py --sarship-root $SSROOT --sardet-root $SDROOT --out-root $Y
```

看终端打印的 `ship class id`（预计 **0**，即 ['ship','aircraft',...] 顺序）。
**若不是 0**，把后面所有 `--target-class 0` 换成该值。
结果：`$Y\sarship\labels\all\*.txt`、`$Y\sarship\{train,val,test}.txt`、
`$Y\sardet\labels\{split}\*.txt`、`$Y\sardet\{split}.txt`（ship 子集）、
`$Y\conversion_summary.json`（把内容也贴给我，论文要引用划分规模）。

## 2. 推理缓存（唯一耗时步骤，CPU）

断点可续（重跑会跳过已有 npz）。建议先跑 SAR-Ship 验证整条链，SARDet 挂夜。

| 缓存 | 图像数 | CPU 预计 |
|---|---|---|
| SAR-Ship val + test | 4381+4381（256px） | 合计 ~1–2 h |
| SARDet val + test（ship 子集） | ~6.4k+6.5k（800px） | 合计 ~3–7 h，挂夜 |

```powershell
$C = "scripts\10_revision\cache_preds.py"
# SAR-Ship（先跑这两个）
python $C --weights $W_SS --img-dir "$SSROOT\JPEGImages" --list "$SS\val.txt"  --out-dir "$WORK\cache_preds\sarship_val"  --device cpu
python $C --weights $W_SS --img-dir "$SSROOT\JPEGImages" --list "$SS\test.txt" --out-dir "$WORK\cache_preds\sarship_test" --device cpu
# SARDet（挂夜）
python $C --weights $W_SD --img-dir "$SDROOT\Images\val"  --list "$SD\val.txt"  --out-dir "$WORK\cache_preds\sardet_val"  --device cpu
python $C --weights $W_SD --img-dir "$SDROOT\Images\test" --list "$SD\test.txt" --out-dir "$WORK\cache_preds\sardet_test" --device cpu
```

## 3–7. 分析流水线（CPU，分钟级）

```powershell
$M = "scripts\10_revision\make_manifest.py"
# 3. manifest
python $M --cache-dir "$WORK\cache_preds\sarship_val"  --img-dir "$SSROOT\JPEGImages" --list "$SS\val.txt"  --out "$WORK\cache_preds\sarship_val\manifest.csv"
python $M --cache-dir "$WORK\cache_preds\sarship_test" --img-dir "$SSROOT\JPEGImages" --list "$SS\test.txt" --out "$WORK\cache_preds\sarship_test\manifest.csv"
python $M --cache-dir "$WORK\cache_preds\sardet_val"  --img-dir "$SDROOT\Images\val"  --list "$SD\val.txt"  --out "$WORK\cache_preds\sardet_val\manifest.csv"
python $M --cache-dir "$WORK\cache_preds\sardet_test" --img-dir "$SDROOT\Images\test" --list "$SD\test.txt" --out "$WORK\cache_preds\sardet_test\manifest.csv"

# 4. e 重算（预测框掩膜）+ 独立杂波统计
$P = "scripts\10_revision\recompute_proxy.py"
python $P --manifest "$WORK\cache_preds\sarship_val\manifest.csv"  --label-root "$SS\labels\all" --out-manifest "$WORK\cache_preds\sarship_val\manifest_proxy.csv"  --k-list 0 1 2 3 5 8 --k-main 3 --mask-gate 0.25 --target-class 0 --report "$OUT\sarship_val_proxy_report.json"
python $P --manifest "$WORK\cache_preds\sarship_test\manifest.csv" --label-root "$SS\labels\all" --out-manifest "$WORK\cache_preds\sarship_test\manifest_proxy.csv" --k-list 0 1 2 3 5 8 --k-main 3 --mask-gate 0.25 --target-class 0 --report "$OUT\sarship_test_proxy_report.json"
python $P --manifest "$WORK\cache_preds\sardet_val\manifest.csv"  --label-root "$SD\labels\val"  --out-manifest "$WORK\cache_preds\sardet_val\manifest_proxy.csv"  --k-list 0 3 5 8 12 16 --k-main 8 --mask-gate 0.25 --target-class 0 --report "$OUT\sardet_val_proxy_report.json"
python $P --manifest "$WORK\cache_preds\sardet_test\manifest.csv" --label-root "$SD\labels\test" --out-manifest "$WORK\cache_preds\sardet_test\manifest_proxy.csv" --k-list 0 3 5 8 12 16 --k-main 8 --mask-gate 0.25 --target-class 0 --report "$OUT\sardet_test_proxy_report.json"

# 5. 主结果 + 对照组（A 验证集最优全局 T / B 随机分箱安慰剂 / C 匹配工作点 / oracle）
$R = "scripts\10_revision\run_main_eval.py"
python $R --val-manifest "$WORK\cache_preds\sarship_val\manifest_proxy.csv" --val-labels "$SS\labels\all" --test-manifest "$WORK\cache_preds\sarship_test\manifest_proxy.csv" --test-labels "$SS\labels\all" --target-class 0 --baseline-T 0.40 --n-perm 20 --require-gt --out "$OUT\sarship_main.json"
python $R --val-manifest "$WORK\cache_preds\sardet_val\manifest_proxy.csv" --val-labels "$SD\labels\val" --test-manifest "$WORK\cache_preds\sardet_test\manifest_proxy.csv" --test-labels "$SD\labels\test" --target-class 0 --baseline-T 0.40 --n-perm 20 --require-gt --out "$OUT\sardet_main.json"

# 6. 10 折 CV / k 敏感性 / 跨数据集迁移
$K = "scripts\10_revision\kfold_cv.py"
python $K --val-manifest "$WORK\cache_preds\sarship_val\manifest_proxy.csv" --val-labels "$SS\labels\all" --test-manifest "$WORK\cache_preds\sarship_test\manifest_proxy.csv" --test-labels "$SS\labels\all" --target-class 0 --K 10 --require-gt --out "$OUT\sarship_kfold.json"
python $K --val-manifest "$WORK\cache_preds\sardet_val\manifest_proxy.csv" --val-labels "$SD\labels\val" --test-manifest "$WORK\cache_preds\sardet_test\manifest_proxy.csv" --test-labels "$SD\labels\test" --target-class 0 --K 10 --require-gt --out "$OUT\sardet_kfold.json"
$Q = "scripts\10_revision\sweep_k.py"
python $Q --val-manifest "$WORK\cache_preds\sarship_val\manifest_proxy.csv" --val-labels "$SS\labels\all" --test-manifest "$WORK\cache_preds\sarship_test\manifest_proxy.csv" --test-labels "$SS\labels\all" --target-class 0 --k-main 3 --require-gt --out "$OUT\sarship_ksweep.json"
python $Q --val-manifest "$WORK\cache_preds\sardet_val\manifest_proxy.csv" --val-labels "$SD\labels\val" --test-manifest "$WORK\cache_preds\sardet_test\manifest_proxy.csv" --test-labels "$SD\labels\test" --target-class 0 --k-main 8 --require-gt --out "$OUT\sardet_ksweep.json"
$X = "scripts\10_revision\cross_transfer.py"
python $X --src-main "$OUT\sardet_main.json"  --tgt-main "$OUT\sarship_main.json" --tgt-test-manifest "$WORK\cache_preds\sarship_test\manifest_proxy.csv" --tgt-test-labels "$SS\labels\all" --target-class 0 --require-gt --out "$OUT\transfer_sardet_to_sarship.json"
python $X --src-main "$OUT\sarship_main.json" --tgt-main "$OUT\sardet_main.json"  --tgt-test-manifest "$WORK\cache_preds\sardet_test\manifest_proxy.csv" --tgt-test-labels "$SD\labels\test" --target-class 0 --require-gt --out "$OUT\transfer_sarship_to_sardet.json"

# 7. 汇总
python scripts\10_revision\make_tables.py --out-dir $OUT
```

## 8. 发回给我的东西

```powershell
Get-Content "$Y\conversion_summary.json" -Raw
Get-Content "$OUT\numbers.json" -Raw
Get-Content "$OUT\table_rows.tex" -Raw
```

（外加任何一步的报错截图/文本。）

## 常见问题

- `--target-class`：以第 1 步打印的 `ship class id` 为准（预计 0）
- SARDet 推理慢：正常，CPU + 800px 就是这个量级；挂夜或分两次跑（断点续传）
- numpy 2.x 兼容问题：`pip install "numpy<2"`
- 符号乱码：确认开头三行 `$env:PYTHONIOENCODING` / `OutputEncoding` 已执行
- **不要**再跑 `make_sarship_test_split.py`——v3 的 `convert_datasets.py` 已直接生成
  train/val/test 三个清单

## Step: figures (new in kit v7)

After `run_revision.cmd analyze` has produced `revision_out\numbers.json`:

```
run_revision.cmd figures
```

- exports `revision_out\curves_sarship.json` and `revision_out\curves_sardet.json`
  (per-bin F1 / gated-AP vs threshold curves for paper Fig. 2), and
- prepares `revision_out\adaptive_params_*.csv/json`, `manifest_cases_*.csv`,
  `data_case_*.yaml` for `scripts\07_viz\plot_case_examples_v2.py` (paper Fig. 3);
  the exact commands to run are printed by the script.

Send back: `curves_sarship.json`, `curves_sardet.json`, and the two
`case_examples_*.pdf` files. Fig. 2 is then re-plotted and the letter
recompiled on the assistant's side.

## Step H: CFAR-style evaluation (new in kit v8)

After `analyze` has run (uses the existing caches; a few minutes):

```
run_revision.cmd cfar
```

Question tested: at the same overall false-alarm budget, does clutter
conditioning keep the false-alarm rate per image (and precision) uniform
across the true clutter strata, where a global threshold and a
random-binning placebo cannot? Targets are pre-specified (0.5x/1x/2x the
calibration FA/img of T=0.40; precision 0.85/0.90/0.95), thresholds are
fitted on the calibration split, everything is evaluated on test, with a
20-permutation placebo and a 200-sample bootstrap.

Send back: `revision_out\sarship_cfar.json` and `revision_out\sardet_cfar.json`
(console summaries are also saved as `*_cfar.txt`).
