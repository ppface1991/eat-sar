# 实验说明（你看的版本）

下面这些是代码里几个关键设计决定的解释，便于你在跑实验和写论文时心里有数。

## 1. 为什么先把预测缓存下来 (`04_threshold/cache_predictions.py`)?

我们要在很多 score 阈值下、还要分环境层重复评估。如果每次都让 YOLO 跑一遍推理，几十次扫描就把卡跑爆了。

做法：跑 YOLO 时把 `conf=0.001` 设得很低，保留所有候选框（含 score、xyxy）。后续所有"阈值"操作都是在内存里对这个候选集做 `score >= T` 过滤——零额外推理成本。这也是 GRSL 评审会喜欢的"轻量"特性：**方法只增加 O(1) 后处理。**

## 2. 评估指标怎么选

代码里同时算 4 个指标：

- `precision`, `recall`：单阈值的工作点
- `f1`：综合工作点，**默认用来选最优 T**（在 `configs/default.yaml` 的 `threshold.selection_metric`）
- `map50`：单 IoU 下的 AP（COCO-101-point 插值）

论文里建议主表用 **F1 + mAP@0.5** 并列，因为：

- 自适应阈值的"好处"在 F1 上体现最直接（baseline 用固定 T 通常 P 高 R 低，自适应在恶劣海况下抬 R）
- 但仅看 F1 容易被审稿人质疑"是不是阈值过拟合"，加上 mAP@0.5 能反驳：mAP 不依赖具体阈值，只看 score 排序。

## 3. 自适应阈值的两种形式

- **Piecewise constant**：直观、易实现、对训练集分布无强假设，工程上最稳。
- **Linear** `T = a·e + b`：连续可导，看起来更"方法"一点；但只有 3 个锚点（low/mid/high）拟合的话其实不算严格的回归——更多是 **示意**。

GRSL 4-page 篇幅里我建议这样讲故事：
> "为了避免对环境分箱方式过度依赖，我们也给出连续形式 $T(e)=ae+b$ 的拟合结果作为参照；二者表现相当，验证了方法的鲁棒性。"

这样在论文里它就是一个 **消融实验** 的角色，而不是争"谁是主方法"。

## 4. baseline 阈值怎么定

`evaluate_adaptive.py` 默认 `--baseline-T 0.25`。两个推荐值：

- **0.25**：YOLOv8 默认 inference 阈值，最常见、最 fair 的对照。
- **VAL 集 ALL bin 的 argmax-F1**：从 `runs/<tag>/best_per_bin.csv` 里 `env_bin=ALL` 行取出。这个更"狠"，相当于让 baseline 也用最优全局阈值。

**两个 baseline 都跑、都写**进论文：

| Baseline | 解释 |
|---|---|
| `T=0.25` | 工业默认 |
| `T=T*_global` | 验证集上选出的最优全局阈值 |
| Adaptive (piecewise) | 我们的方法 |
| Adaptive (linear) | 我们的方法（连续形式） |

## 5. 数据集元数据的"软肋"

整个 pipeline 最容易卡住的就是 **第 1 步**：很多开源 SAR 数据集发布时只放了图 + 标注，把原始大场景的经纬度/成像时间丢掉了。

如果你下载的 SARDet-100K 的 `images` 字段里 **没有** lat/lon/time:

1. 看 SARDet-100K 论文附录有没有给"scene → source product"的映射表（[arXiv 2403.06534](https://arxiv.org/abs/2403.06534) 的 supplementary）；
2. 没有的话，**最稳的退路** 是只用 SAR-Ship-Dataset 那一支——因为它的切片名往往保留了 Sentinel-1 原始产品名，时间/卫星可以从文件名解析，经纬度可以查 ESA 的 Sentinel-1 产品 footprint 数据库；
3. 实在不行，做一个"代理环境分箱"：用 SAR 图像本身的全局后向散射强度统计（如均值、方差、海杂波直方图 entropy）来估海况强弱。这个 fallback 我可以再帮你写一段代码，但要做就别叫 ERA5-based 了，要诚实地写成"image-derived sea-state proxy"。

请先确认你拿到手的数据里是否有 lat/lon/time，**这个决定后续一切**。

## 6. 我建议你按这个顺序先跑通一个 dry-run

1. `extract_sardet100k_meta.py` → 看 missing lat/lon/time 是多少
2. 如果 >90% 缺失：先去找 scene mapping；不要继续往下走
3. 如果 <50% 缺失：直接 `download_era5.py --dry-run` 看 (year, month) 组合数量是否合理（10~30 个比较正常）
4. 真下 ERA5（最慢的一步，CDS 排队可能要 1~6 小时）
5. `match_era5_to_scenes.py` → 看 `wind_bin` 分布是否 3 档都有 >300 张图（少于这个量，分箱可能不稳）
6. 再开始训练 YOLOv8

如果 wind_bin 高档样本不足，可以把 `wind_bins` 改成 `[0, 6, 100]` 二档，论文里就说"低海况 vs 高海况"。

## 7. 跑完结果发我什么

按 README 末尾的"Ablations to send back"清单打包发我。我会用这些文件起草：

- Section II.A 数据 + 配准（表 1：每个环境层的样本量）
- Section II.B 阈值依赖环境的发现（图 2：F1 vs threshold 分层曲线）
- Section II.C 自适应方法（公式 + 图 3：拟合的 T(e) 曲线）
- Section III 实验（表 2：overall；表 3：stratified；图 4：定性 case）

一篇 GRSL 标准的 4–6 页 letter 完全够用。
