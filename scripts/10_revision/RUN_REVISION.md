# Revision pipeline — Linux/macOS command reference

(Portable version of the original AutoDL runbook; paths are placeholders.)

全部基于 **已缓存的 .npz 预测**（conf≥0.01），除两次新的推理缓存外不需要 GPU，
CPU 上整套 ≈ 20–40 分钟。所有结果写到 `$OUT`，最后把 `numbers.json` 和
`table_rows.tex` 发回给我即可。

> 路径沿用 7 月布局；如有不同请对应修改前 6 行变量。

```bash
cd /path/to/eat-sar
source /root/eat-sar-env/bin/activate          # 你的 venv
export PYTHONPATH=/path/to/eat-sar/src
pip install -q scipy                           # recompute_proxy / sweep_k 需要

WORK=/path/to/work
OUT=/path/to/eat-sar/revision_out ; mkdir -p $OUT
W_SARDET=/path/to/eat-sar/runs_backup/sardet100k/best.pt
W_SARSHIP=/path/to/eat-sar/runs_backup/sarship/weights/best.pt
SD=$WORK/sardet_root        # images/{val,test}  labels/{val,test}
SS=$WORK/sarship_root       # images/all labels/all train.txt val.txt
unzip -o revision_kit_v1.zip -d /path/to/eat-sar/   # 解压本 kit（scripts/10_revision/）
```

---

## 0. 补齐缺失的推理缓存（唯一需要 GPU 的步骤，~1–3 min 每个）

7 月只缓存了 SARDet **test** 和 SAR-Ship **val**。审稿要求"在 val 上标定、在 test
上报告"，所以要补 SARDet **val** 与 SAR-Ship **test**。

```bash
# 0a  SAR-Ship test 划分恢复（自动判断 train+val 之外是否还有 10% 未用图像）
python scripts/10_revision/make_sarship_test_split.py --root $SS
#   -> 若打印 "wrote test.txt"   : 用 val.txt 标定、test.txt 报告（理想）
#   -> 若打印 "[WARN] ..."       : 用 val_calib.txt 标定、val_test.txt 报告，并告诉我

# 0b  SARDet val 预测缓存
python scripts/06_eval/cache_predictions.py --weights $W_SARDET \
    --img_dir $SD/images/val --label_dir $SD/labels/val \
    --out_dir $WORK/cache_preds/sardet100k_val --conf 0.01 --imgsz 640

# 0c  SAR-Ship test 预测缓存（若是 val_calib/val_test 情形，两个都在 val 缓存里，跳过本步）
python scripts/06_eval/cache_predictions.py --weights $W_SARSHIP \
    --img_dir $SS/images/all --list $SS/test.txt --label_dir $SS/labels/all \
    --out_dir $WORK/cache_preds/sarship_test --conf 0.01 --imgsz 640
```

## 1. 组装 manifest（stem, image_path, pred_npz）

```bash
M=scripts/10_revision/make_manifest.py
python $M --cache-dir $WORK/cache_preds/sardet100k_val  --img-dir $SD/images/val  --out $WORK/cache_preds/sardet100k_val/manifest.csv
python $M --cache-dir $WORK/cache_preds/sardet100k_test --img-dir $SD/images/test --out $WORK/cache_preds/sardet100k_test/manifest.csv
python $M --cache-dir $WORK/cache_preds/sarship_val  --img-dir $SS/images/all --list $SS/val.txt  --out $WORK/cache_preds/sarship_val/manifest.csv
python $M --cache-dir $WORK/cache_preds/sarship_test --img-dir $SS/images/all --list $SS/test.txt --out $WORK/cache_preds/sarship_test/manifest.csv
# (val_calib/val_test 情形：两条 sarship 命令都用 --cache-dir $WORK/cache_preds/sarship_val，--list 分别换成 val_calib.txt / val_test.txt，--out 分别写到 sarship_calib/ 与 sarship_evalhalf/ 目录)
```

## 2. 重算杂波代理 e（**预测框掩膜**，不碰 GT）+ 独立杂波统计 + 多 k

```bash
P=scripts/10_revision/recompute_proxy.py
# SARDet: k_main=8 ; ship 类 id 若不是 0 请改 --target-class
python $P --manifest $WORK/cache_preds/sardet100k_val/manifest.csv  --label-root $SD/labels/val  --out-manifest $WORK/cache_preds/sardet100k_val/manifest_proxy.csv  --k-list 0 3 5 8 12 16 --k-main 8 --mask-gate 0.25 --target-class 0 --report $OUT/sardet_val_proxy_report.json
python $P --manifest $WORK/cache_preds/sardet100k_test/manifest.csv --label-root $SD/labels/test --out-manifest $WORK/cache_preds/sardet100k_test/manifest_proxy.csv --k-list 0 3 5 8 12 16 --k-main 8 --mask-gate 0.25 --target-class 0 --report $OUT/sardet_test_proxy_report.json
# SAR-Ship: k_main=3
python $P --manifest $WORK/cache_preds/sarship_val/manifest.csv  --label-root $SS/labels/all --out-manifest $WORK/cache_preds/sarship_val/manifest_proxy.csv  --k-list 0 1 2 3 5 8 --k-main 3 --mask-gate 0.25 --target-class 0 --report $OUT/sarship_val_proxy_report.json
python $P --manifest $WORK/cache_preds/sarship_test/manifest.csv --label-root $SS/labels/all --out-manifest $WORK/cache_preds/sarship_test/manifest_proxy.csv --k-list 0 1 2 3 5 8 --k-main 3 --mask-gate 0.25 --target-class 0 --report $OUT/sarship_test_proxy_report.json
```

## 3. 主结果重生成 + 三组对照（val 标定 → test 报告）

```bash
R=scripts/10_revision/run_main_eval.py
python $R --val-manifest $WORK/cache_preds/sardet100k_val/manifest_proxy.csv --val-labels $SD/labels/val \
          --test-manifest $WORK/cache_preds/sardet100k_test/manifest_proxy.csv --test-labels $SD/labels/test \
          --target-class 0 --baseline-T 0.40 --n-perm 20 --require-gt --out $OUT/sardet_main.json
python $R --val-manifest $WORK/cache_preds/sarship_val/manifest_proxy.csv --val-labels $SS/labels/all \
          --test-manifest $WORK/cache_preds/sarship_test/manifest_proxy.csv --test-labels $SS/labels/all \
          --target-class 0 --baseline-T 0.40 --n-perm 20 --require-gt --out $OUT/sarship_main.json
```
`--require-gt` = 只统计含 ≥1 艘船的图（论文中的 "ship subset" 口径）。
如想同时给出"全部图像"口径，去掉该参数再跑一次，输出改名 `*_main_allimg.json`。

## 4. 10 折交叉验证（R1-Q1）

```bash
K=scripts/10_revision/kfold_cv.py
python $K --val-manifest $WORK/cache_preds/sardet100k_val/manifest_proxy.csv --val-labels $SD/labels/val --test-manifest $WORK/cache_preds/sardet100k_test/manifest_proxy.csv --test-labels $SD/labels/test --target-class 0 --K 10 --require-gt --out $OUT/sardet_kfold.json
python $K --val-manifest $WORK/cache_preds/sarship_val/manifest_proxy.csv --val-labels $SS/labels/all --test-manifest $WORK/cache_preds/sarship_test/manifest_proxy.csv --test-labels $SS/labels/all --target-class 0 --K 10 --require-gt --out $OUT/sarship_kfold.json
```

## 5. 膨胀核 k 敏感性（R2）

```bash
Q=scripts/10_revision/sweep_k.py
python $Q --val-manifest $WORK/cache_preds/sardet100k_val/manifest_proxy.csv --val-labels $SD/labels/val --test-manifest $WORK/cache_preds/sardet100k_test/manifest_proxy.csv --test-labels $SD/labels/test --target-class 0 --k-main 8 --require-gt --out $OUT/sardet_ksweep.json
python $Q --val-manifest $WORK/cache_preds/sarship_val/manifest_proxy.csv --val-labels $SS/labels/all --test-manifest $WORK/cache_preds/sarship_test/manifest_proxy.csv --test-labels $SS/labels/all --target-class 0 --k-main 3 --require-gt --out $OUT/sarship_ksweep.json
```

## 6. 跨数据集/跨传感器迁移（R2）

```bash
X=scripts/10_revision/cross_transfer.py
python $X --src-main $OUT/sardet_main.json  --tgt-main $OUT/sarship_main.json --tgt-test-manifest $WORK/cache_preds/sarship_test/manifest_proxy.csv   --tgt-test-labels $SS/labels/all  --target-class 0 --require-gt --out $OUT/transfer_sardet_to_sarship.json
python $X --src-main $OUT/sarship_main.json --tgt-main $OUT/sardet_main.json  --tgt-test-manifest $WORK/cache_preds/sardet100k_test/manifest_proxy.csv --tgt-test-labels $SD/labels/test --target-class 0 --require-gt --out $OUT/transfer_sarship_to_sardet.json
```

## 7. 汇总 → 发回

```bash
python scripts/10_revision/make_tables.py --out-dir $OUT
cat $OUT/numbers.json      # 整个贴给我
cat $OUT/table_rows.tex
```

## 8. （数字确定后）重出 Fig. 2–6

Fig. 2–5 曲线与 Fig. 6 定性图需用 **新的 e_pred 分箱与新的 T_b** 重出，命令与 7 月相同，
只是 `--adaptive-json` 换成从 `$OUT/*_main.json` 生成的 json（我会在拿到数字后给你一个
一行转换命令）。

---

### 常见问题
* `KeyError: 'stem'` → manifest 缺 stem 列；用本 kit 的 make_manifest.py 重建。
* 标签找不到 → `--label-root` 必须是放 `<stem>.txt` 的目录（SAR-Ship 是 labels/all）。
* SARDet ship 类 id：`python -c "import yaml;print(yaml.safe_load(open('$SD/data.yaml'))['names'])"`，ship 的下标填 `--target-class`。
* 内存：SARDet 两万张 800×800 全加载约 1–2 GB，正常。
