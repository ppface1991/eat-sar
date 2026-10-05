"""
Plot a 3x3 qualitative grid (clutter bin x [Input+GT | Baseline | Adaptive])
for the GRSL revision response.

For each clutter bin (low / medium / high), we pick ONE illustrative test image
and render three panels:
  - Column 1: input SAR image + ground-truth boxes (green)
  - Column 2: baseline detections at fixed T (red boxes with score)
  - Column 3: adaptive-T detections per the per-bin policy (red boxes)

Output: vector PDF sized for IEEE GRSL figure* (two-column wide ~7.16in).

Usage (run on your AutoDL / WSL machine, NOT in this workspace):
    python scripts/07_viz/plot_case_examples_v2.py \\
        --manifest cache/preds/sardet100k_test/manifest.csv \\
        --data /root/autodl-tmp/eat-sar/yolo/sardet100k/data.yaml \\
        --split test \\
        --adaptive-json runs/sardet100k_val/adaptive_T.json \\
        --baseline-T 0.40 \\
        --out paper/figs/case_examples_sardet.pdf \\
        --dataset-label "SARDet-100K"

Then re-run for SAR-Ship-Dataset substituting paths accordingly, with
--out paper/figs/case_examples_sarship.pdf and --dataset-label "SAR-Ship".

Manual case picking (recommended):
    Pass --pick-low STEM --pick-med STEM --pick-high STEM (image stems) to
    deterministically choose specific images that best showcase the method.
    Otherwise, random picks are deterministic via --seed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from eat_sar.adaptive import LinearT, PiecewiseT  # noqa: E402
from eat_sar.inference_cache import load_pred, load_yolo_gt  # noqa: E402


# ---------- IEEE-friendly matplotlib style ----------
mpl.rcParams.update({
    "pdf.fonttype": 42,          # TrueType, no Type-3 (GRSL required)
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 8.0,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.0,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def draw_boxes(ax, boxes, color, lw=1.2):
    for (x1, y1, x2, y2) in boxes:
        rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  fill=False, edgecolor=color, lw=lw)
        ax.add_patch(rect)


def pick_row(mf: pd.DataFrame, bin_name: str, override_stem: str | None,
             seed: int) -> pd.Series:
    sub = mf[mf["wind_bin"] == bin_name]
    if len(sub) == 0:
        raise SystemExit(f"No rows in manifest for wind_bin={bin_name!r}")
    if override_stem:
        match = sub[sub["image_path"].str.contains(f"/{override_stem}.",
                                                    regex=False)]
        if len(match) == 0:
            raise SystemExit(f"--pick-{bin_name} {override_stem!r} not found "
                             f"in {bin_name} bin")
        return match.iloc[0]
    return sub.sample(n=1, random_state=seed).iloc[0]


def _box_iou_matrix(a, b):
    """IoU matrix between [N,4] and [M,4] xyxy boxes. Returns [N,M]."""
    import numpy as np
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def _count_tp_fp(pred_boxes, pred_keep_mask, gt_boxes, iou_thr=0.5):
    """Greedy match kept predictions to GT; returns (n_TP, n_FP)."""
    import numpy as np
    kept = pred_boxes[pred_keep_mask]
    if len(kept) == 0:
        return 0, 0
    if len(gt_boxes) == 0:
        return 0, len(kept)
    iou = _box_iou_matrix(kept, gt_boxes)
    matched_gt = set()
    n_tp = 0
    # greedy: iterate kept (already sorted by score desc by load_pred conv.)
    for i in range(len(kept)):
        # best unmatched gt
        order = np.argsort(-iou[i])
        for j in order:
            if j in matched_gt: continue
            if iou[i, j] >= iou_thr:
                matched_gt.add(int(j))
                n_tp += 1
            break
    n_fp = len(kept) - n_tp
    return n_tp, n_fp


def pick_discriminative_row(mf: pd.DataFrame, bin_name: str, T_base: float,
                             T_adapt: float, label_root: Path,
                             target_class: int | None = None,
                             max_scan: int = 800, seed: int = 0,
                             positive_only: bool = True) -> pd.Series:
    """Pick an image where adaptive thresholding visibly *helps* vs baseline.

    Positive-case definition (positive_only=True, default):
      n_TP_adapt >= n_TP_base  AND  n_FP_adapt < n_FP_base
      i.e. adaptive removes false-positives without dropping any true target.

    Score: (n_FP_base - n_FP_adapt) * 10 + n_TP_adapt + bonus_for_modest_n_det
    Ties broken by preferring 2-6 detections (good visual readability).
    """
    import numpy as np
    sub = mf[mf["wind_bin"] == bin_name].copy()
    if len(sub) == 0:
        raise SystemExit(f"No rows in manifest for wind_bin={bin_name!r}")
    if len(sub) > max_scan:
        sub = sub.sample(n=max_scan, random_state=seed)

    cands = []
    for _, row in sub.iterrows():
        try:
            boxes, scores, classes = load_pred(Path(row["pred_npz"]))
        except Exception:
            continue
        if target_class is not None and len(classes) > 0:
            m = classes == target_class
            boxes = boxes[m]; scores = scores[m]
        if len(scores) == 0:
            continue
        keep_base = scores >= T_base
        keep_adapt = scores >= T_adapt
        n_base = int(keep_base.sum())
        n_adapt = int(keep_adapt.sum())
        if n_base == n_adapt:
            continue  # no visual difference
        img_path = Path(row["image_path"])
        with Image.open(img_path) as im:
            W, H = im.size
        gt = load_yolo_gt(label_root / f"{img_path.stem}.txt", W, H)
        if target_class is not None and gt.shape[0] > 0:
            raw = np.loadtxt(label_root / f"{img_path.stem}.txt", ndmin=2)
            if raw.size > 0:
                mask = raw[:, 0].astype(int) == target_class
                gt = gt[mask]
        n_gt = len(gt)
        if n_gt < 1:
            continue
        tp_b, fp_b = _count_tp_fp(boxes, keep_base, gt)
        tp_a, fp_a = _count_tp_fp(boxes, keep_adapt, gt)
        # readability bonus: prefer images with 2-6 detections in baseline
        if 2 <= n_base <= 6:
            readable = 2
        elif 1 <= n_base <= 10:
            readable = 1
        else:
            readable = 0
        # positive case: adaptive does not lose TP, removes FP
        is_positive = (tp_a >= tp_b) and (fp_b > fp_a)
        score = (fp_b - fp_a) * 10 + tp_a + readable
        cands.append({
            "row": row, "n_base": n_base, "n_adapt": n_adapt,
            "tp_b": tp_b, "fp_b": fp_b, "tp_a": tp_a, "fp_a": fp_a,
            "n_gt": n_gt, "score": score, "is_positive": is_positive,
            "readable": readable,
        })
    if not cands:
        print(f"  [warn] bin={bin_name!r}: no candidates; falling back to random")
        return sub.sample(n=1, random_state=seed).iloc[0]
    if positive_only:
        pos = [c for c in cands if c["is_positive"]]
        if pos:
            cands = pos
        else:
            print(f"  [warn] bin={bin_name!r}: no strict-positive case; "
                  f"relaxing to any difference")
    cands.sort(key=lambda c: c["score"], reverse=True)
    best = cands[0]
    tag = "POS" if best["is_positive"] else "NEG"
    print(f"  bin={bin_name!r} [{tag}]: base TP={best['tp_b']} FP={best['fp_b']} "
          f"({best['n_base']} det) | ours TP={best['tp_a']} FP={best['fp_a']} "
          f"({best['n_adapt']} det) | GT={best['n_gt']} | "
          f"stem={Path(best['row']['image_path']).stem}")
    return best["row"]


def panel(ax, img, gt_boxes, det_boxes, title, gt_color="lime", det_color="red"):
    ax.imshow(img, cmap="gray")
    if gt_boxes is not None:
        draw_boxes(ax, gt_boxes, gt_color, lw=1.6)
    if det_boxes is not None:
        draw_boxes(ax, det_boxes, det_color, lw=1.2)
    ax.set_title(title, fontsize=8.5, pad=2.0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--adaptive-json", required=True)
    ap.add_argument("--baseline-T", type=float, default=0.40)
    ap.add_argument("--scheme", default="piecewise",
                    choices=["piecewise", "linear"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pick-low", default=None)
    ap.add_argument("--pick-med", default=None)
    ap.add_argument("--pick-high", default=None)
    ap.add_argument("--auto-discriminative", action="store_true",
                    help="Auto-pick images where baseline and adaptive disagree.")
    ap.add_argument("--target-class", type=int, default=None,
                    help="If set, restrict picks/score-filter to a single class id.")
    ap.add_argument("--dataset-label", default="Dataset")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params = json.load(open(args.adaptive_json))
    if args.scheme == "piecewise":
        T_fn = PiecewiseT(bins=params["piecewise"]["bins"],
                          thresholds=params["piecewise"]["thresholds"])
    else:
        T_fn = LinearT(a=params["linear"]["a"], b=params["linear"]["b"])
    env_value_key = params["env_value_key"]

    data_yaml = yaml.safe_load(open(args.data))
    label_root = Path(data_yaml["path"]) / "labels" / args.split

    mf = pd.read_csv(args.manifest)

    # If --pick-* override given, use it. Otherwise, auto-pick a discriminative
    # case where baseline and adaptive give different detection counts.
    if args.auto_discriminative and not (args.pick_low or args.pick_med or args.pick_high):
        print("Auto-selecting discriminative cases per bin...")
        # need a T for each bin to score candidates
        T_lookup = {}
        if args.scheme == "piecewise":
            bins = params["piecewise"]["bins"]
            ths = params["piecewise"]["thresholds"]
            order = params["piecewise"].get("bin_order", ["low", "medium", "high"])
            for bn, T in zip(order, ths):
                T_lookup[bn] = T
        else:
            # linear: estimate per-bin via mean env value
            for bn in ["low", "medium", "high"]:
                sub = mf[mf["wind_bin"] == bn]
                T_lookup[bn] = float(T_fn(sub[env_value_key].mean())[0]) if len(sub) else args.baseline_T
        picks = {
            "low": (pick_row(mf, "low", args.pick_low, args.seed) if args.pick_low
                    else pick_discriminative_row(mf, "low", args.baseline_T,
                                                  T_lookup["low"], label_root,
                                                  target_class=args.target_class,
                                                  seed=args.seed)),
            "medium": (pick_row(mf, "medium", args.pick_med, args.seed) if args.pick_med
                       else pick_discriminative_row(mf, "medium", args.baseline_T,
                                                     T_lookup["medium"], label_root,
                                                     target_class=args.target_class,
                                                     seed=args.seed + 1)),
            "high": (pick_row(mf, "high", args.pick_high, args.seed) if args.pick_high
                     else pick_discriminative_row(mf, "high", args.baseline_T,
                                                   T_lookup["high"], label_root,
                                                   target_class=args.target_class,
                                                   seed=args.seed + 2)),
        }
    else:
        picks = {
            "low": pick_row(mf, "low", args.pick_low, args.seed),
            "medium": pick_row(mf, "medium", args.pick_med, args.seed + 1),
            "high": pick_row(mf, "high", args.pick_high, args.seed + 2),
        }
    bin_labels = {
        "low": "Low-clutter",
        "medium": "Medium-clutter",
        "high": "High-clutter",
    }

    # figure* width in GRSL/IEEE double-column is ~7.16in. Three square panels
    # per row, so panel side ~= 7.16/3 = 2.39 in. Height = 3*2.39 + headroom.
    panel_side = 7.16 / 3.0
    header_in = 0.75  # space for suptitle (line 1) + legend (line 2)
    fig_h = 3 * panel_side + header_in
    fig, axes = plt.subplots(3, 3, figsize=(7.16, fig_h))

    for r, bin_name in enumerate(["low", "medium", "high"]):
        row = picks[bin_name]
        img = Image.open(row["image_path"]).convert("L")
        W, H = img.size
        gt = load_yolo_gt(label_root / f"{Path(row['image_path']).stem}.txt",
                          W, H)
        boxes, scores, _ = load_pred(Path(row["pred_npz"]))

        T_base = args.baseline_T
        T_adapt = float(T_fn(row[env_value_key])[0])
        env_val = row[env_value_key]

        # Column 0: input + GT
        panel(axes[r, 0], img, gt, None,
              f"{bin_labels[bin_name]}: Input + GT")
        # Column 1: baseline
        keep_b = scores >= T_base
        panel(axes[r, 1], img, None, boxes[keep_b],
              f"Baseline (T={T_base:.2f}): {keep_b.sum()} det.")
        # Column 2: adaptive
        keep_a = scores >= T_adapt
        panel(axes[r, 2], img, None, boxes[keep_a],
              f"Ours (T={T_adapt:.2f}, e={env_val:.1f}): {keep_a.sum()} det.")

    # Layout: top header_in inches for suptitle + legend; rest for 3x3 grid.
    top_frac = 1.0 - header_in / fig_h
    plt.subplots_adjust(left=0.005, right=0.995, top=top_frac, bottom=0.005,
                        wspace=0.03, hspace=0.16)

    # Suptitle on line 1
    suptitle_y = 1.0 - 0.18 / fig_h
    fig.suptitle(f"{args.dataset_label}: qualitative comparison across clutter strata",
                 fontsize=9.5, y=suptitle_y)

    # Inline legend on line 2: draw on an invisible full-figure axes so we have
    # explicit control over placement (avoids matplotlib's fig.legend collision
    # with the first row of axes titles).
    legend_y = 1.0 - 0.48 / fig_h
    leg_ax = fig.add_axes([0, 0, 1, 1], frameon=False, zorder=10)
    leg_ax.set_xticks([]); leg_ax.set_yticks([])
    leg_ax.set_xlim(0, 1); leg_ax.set_ylim(0, 1)
    leg_ax.patch.set_alpha(0)
    box_w, box_h = 0.022, 0.018
    text_pad = 0.005
    items = [("lime", 1.6, "Ground truth"), ("red", 1.2, "Detection")]
    # Approximate text widths (fig coords) for a fontsize-8.5 string
    text_w = {"Ground truth": 0.085, "Detection": 0.055}
    item_widths = [box_w + text_pad + text_w[label] for _, _, label in items]
    gap = 0.04
    total = sum(item_widths) + gap * (len(items) - 1)
    x = 0.5 - total / 2
    for (color, lw, label), w in zip(items, item_widths):
        rect = mpatches.Rectangle((x, legend_y - box_h / 2), box_w, box_h,
                                  edgecolor=color, facecolor="none",
                                  linewidth=lw, transform=leg_ax.transAxes)
        leg_ax.add_patch(rect)
        leg_ax.text(x + box_w + text_pad, legend_y, label,
                    fontsize=8.5, va="center", ha="left",
                    transform=leg_ax.transAxes)
        x += w + gap

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"Saved: {args.out}")
    print(f"Picked cases (stems):")
    for bn, r in picks.items():
        print(f"  {bn:6s} -> {Path(r['image_path']).stem}  "
              f"({env_value_key}={r[env_value_key]:.2f})")


if __name__ == "__main__":
    main()
