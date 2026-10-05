"""
Soft-NMS baseline (Bodla et al., ICCV 2017) for SAR ship detection.

复用现有 preds/*.npz 缓存：缓存里已经是 Ultralytics 用 hard-NMS + IoU=0.6 过滤后的
predictions。本脚本对每张图做以下操作：

    1. 读 .npz （boxes, scores）
    2. 用 Gaussian Soft-NMS (sigma=0.5) 重新对所有 boxes 进行 score 调整
    3. 套用全局阈值 T0（与 baseline 同一 T=0.40）
    4. 按 env_bin 分箱评估 P/R/F1/mAP@0.5

注意：严格意义上 Soft-NMS 应在 raw detector output 上做（即 conf=0.001 不做 NMS）。
我们的 cache 已经过 hard-NMS，因此这里运行的是 "Soft-NMS over hard-NMS-survivors"，
相当于二次去重。这是论文里最公平的对照方式，因为我们的方法也是在同一 cache 上做的。
如果想严格复现 Soft-NMS，需要重跑 inference 时设 conf=0.001, iou=1.0（即不做 NMS）。
本脚本支持两种模式：--mode soft-over-cache (默认，快) 和 --mode strict (需 --weights 重跑)。

Usage (复用 cache):
    python scripts/08_baselines/run_softnms.py \\
        --manifest /autodl-fs/data/eat-sar/runs/sardet100k_test/manifest.csv \\
        --data /root/autodl-tmp/eat-sar-yolo/sardet100k/data.yaml \\
        --split test \\
        --config configs/default.yaml \\
        --T 0.40 --sigma 0.5 --env-key clutter_bin \\
        --tag sardet100k_test_softnms

Usage (strict — 需重跑 inference):
    python scripts/08_baselines/run_softnms.py \\
        --mode strict \\
        --weights /autodl-fs/data/eat-sar/runs/sardet100k_yolov8s/weights/best.pt \\
        --data /root/autodl-tmp/eat-sar-yolo/sardet100k/data.yaml \\
        --split test --meta-csv ... \\
        --T 0.40 --sigma 0.5 --tag sardet100k_test_softnms_strict
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from eat_sar.eval_metrics import (average_precision, match_predictions,  # noqa: E402
                                    precision_recall_f1)
from eat_sar.inference_cache import load_pred, load_yolo_gt, save_pred  # noqa: E402
from eat_sar.utils import ensure_dir, load_config  # noqa: E402


def soft_nms_gaussian(boxes: np.ndarray, scores: np.ndarray,
                      sigma: float = 0.5, score_thresh: float = 1e-3
                      ) -> tuple[np.ndarray, np.ndarray]:
    """
    Gaussian Soft-NMS (Bodla 2017). All inputs / outputs in xyxy.
    Returns boxes & rescored scores; ordering may change.
    Reference: si <- si * exp(-iou^2 / sigma)
    """
    N = len(boxes)
    if N == 0:
        return boxes, scores
    boxes = boxes.astype(np.float32).copy()
    scores = scores.astype(np.float32).copy()
    indexes = np.arange(N)
    for i in range(N):
        # pick current max
        max_idx = i + int(np.argmax(scores[i:]))
        # swap
        boxes[[i, max_idx]] = boxes[[max_idx, i]]
        scores[[i, max_idx]] = scores[[max_idx, i]]
        indexes[[i, max_idx]] = indexes[[max_idx, i]]

        # compute IoU with rest
        if i + 1 >= N:
            break
        ax1, ay1, ax2, ay2 = boxes[i]
        bx1, by1, bx2, by2 = boxes[i + 1:].T
        iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
        ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
        inter = iw * ih
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        iou = inter / np.maximum(area_a + area_b - inter, 1e-9)
        # Gaussian decay
        scores[i + 1:] = scores[i + 1:] * np.exp(-(iou ** 2) / sigma)

    keep = scores >= score_thresh
    return boxes[keep], scores[keep]


def per_image_eval_soft(row, label_root, iou_eval, T, sigma):
    boxes, scores, _ = load_pred(Path(row["pred_npz"]))
    img_path = Path(row["image_path"])
    with Image.open(img_path) as im:
        W, H = im.size
    gt = load_yolo_gt(label_root / f"{img_path.stem}.txt", W, H)
    # Soft-NMS pass
    boxes_s, scores_s = soft_nms_gaussian(boxes, scores, sigma=sigma)
    # threshold
    keep = scores_s >= T
    boxes_k = boxes_s[keep]; scores_k = scores_s[keep]
    # sort desc
    order = np.argsort(-scores_k)
    boxes_k = boxes_k[order]; scores_k = scores_k[order]
    tp, n_gt = match_predictions(boxes_k, scores_k, gt, iou_thresh=iou_eval)
    return scores_k, tp, n_gt


def aggregate(scores_list, tp_list, n_gt_total):
    if len(scores_list) == 0:
        return dict(precision=0., recall=0., f1=0., map50=0., n_pred=0, n_gt=n_gt_total)
    s = np.concatenate(scores_list); t = np.concatenate(tp_list)
    p, r, f1 = precision_recall_f1(t, s, n_gt_total)
    ap = average_precision(t, s, n_gt_total)
    return dict(precision=p, recall=r, f1=f1, map50=ap, n_pred=int(len(s)), n_gt=n_gt_total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mode", default="soft-over-cache",
                    choices=["soft-over-cache", "strict"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--T", type=float, default=0.40)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--env-key", default="clutter_bin")
    ap.add_argument("--tag", required=True)
    # strict mode
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    iou_eval = cfg["threshold"]["iou_thresh"]
    data_yaml = yaml.safe_load(open(args.data))
    label_root = Path(data_yaml["path"]) / "labels" / args.split

    if args.mode == "strict":
        # Re-run inference with NMS effectively off (iou=1.0), then Soft-NMS here
        from ultralytics import YOLO
        model = YOLO(args.weights)
        img_root = Path(data_yaml["path"]) / data_yaml[args.split]
        out_dir = ensure_dir(Path(cfg["paths"]["runs_dir"]) / args.tag / "preds")
        mf_in = pd.read_csv(args.manifest)
        new_rows = []
        bs = 8
        stems = mf_in["stem"].tolist()
        paths = [str(img_root / f"{s}.jpg") for s in stems]
        for i in tqdm(range(0, len(paths), bs), desc="re-infer raw"):
            batch = paths[i:i + bs]
            results = model.predict(source=batch, conf=0.001, iou=1.0,
                                     imgsz=cfg["train"]["imgsz"],
                                     device=cfg["project"]["device"], verbose=False)
            for p, r in zip(batch, results):
                if r.boxes is None or r.boxes.xyxy is None:
                    boxes = np.zeros((0, 4), dtype=np.float32)
                    scores = np.zeros((0,), dtype=np.float32)
                    classes = np.zeros((0,), dtype=np.int32)
                else:
                    boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                    scores = r.boxes.conf.cpu().numpy().astype(np.float32)
                    classes = r.boxes.cls.cpu().numpy().astype(np.int32)
                stem = Path(p).stem
                save_pred(out_dir / f"{stem}.npz", boxes, scores, classes)
                new_rows.append({"stem": stem, "image_path": p,
                                 "pred_npz": str(out_dir / f"{stem}.npz")})
        mf_new = pd.DataFrame(new_rows).merge(
            mf_in.drop(columns=["image_path", "pred_npz"], errors="ignore"),
            on="stem", how="left")
        mf = mf_new
    else:
        mf = pd.read_csv(args.manifest)

    mf = mf.dropna(subset=[args.env_key])
    bin_labels = sorted(mf[args.env_key].unique().tolist())

    sc_all, tp_all, bin_all = [], [], []
    n_gt_total = 0
    n_gt_per_bin: dict[str, int] = {b: 0 for b in bin_labels}
    t_start = time.time()
    for _, row in tqdm(mf.iterrows(), total=len(mf), desc="soft-nms"):
        s, t, n_gt = per_image_eval_soft(row, label_root, iou_eval, args.T, args.sigma)
        sc_all.append(s); tp_all.append(t)
        bin_all.append(np.array([row[args.env_key]] * len(s)))
        n_gt_total += n_gt
        n_gt_per_bin[row[args.env_key]] += n_gt
    elapsed = time.time() - t_start

    bin_all = np.concatenate(bin_all) if bin_all else np.array([])
    sc_concat = np.concatenate(sc_all) if sc_all else np.array([])
    tp_concat = np.concatenate(tp_all) if tp_all else np.array([])

    rows = []
    overall = aggregate([sc_concat], [tp_concat], n_gt_total)
    rows.append({"scheme": "softnms", "env_bin": "ALL", **overall})
    for b in bin_labels:
        mask = bin_all == b
        sub = aggregate([sc_concat[mask]], [tp_concat[mask]], n_gt_per_bin[b])
        rows.append({"scheme": "softnms", "env_bin": b, **sub})

    df = pd.DataFrame(rows)
    out_dir = ensure_dir(Path(cfg["paths"]["runs_dir"]) / args.tag)
    df.to_csv(out_dir / "final_eval.csv", index=False)
    fps = len(mf) / max(elapsed, 1e-9)
    pd.DataFrame([{"n_images": len(mf), "elapsed_s": elapsed, "fps": fps,
                   "sigma": args.sigma, "T": args.T, "mode": args.mode}]
                  ).to_csv(out_dir / "timing.csv", index=False)
    print(f"\nSaved: {out_dir/'final_eval.csv'}")
    print(f"Throughput: {fps:.2f} img/s (n={len(mf)}, t={elapsed:.1f}s)")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
