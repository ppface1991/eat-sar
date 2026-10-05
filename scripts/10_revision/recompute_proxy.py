"""
Step R1 — Recompute the clutter proxy e WITHOUT ground truth, plus independent
clutter statistics, for every image in a manifest.

Addresses:
  R2  "state whether e is computed on linear amplitude / intensity / dB / YOLO tensor"
      -> e is computed on the 8-bit grey-level DN of the distributed image
         (PIL 'L' conversion), BEFORE any YOLO preprocessing; no clipping.
  R2  "e should track an independent clutter / sea-state measure"
      -> Spearman rho of e against: mean backscatter mu, coefficient of
         variation, histogram entropy, robust spread (p99-median), and the
         K-distribution shape estimate nu_hat (method of moments).
  R2  "high-clutter images are not Rayleigh"
      -> per-bin sigma/mu ratio (Rayleigh predicts 0.5227) and nu_hat.
  R2  "sensitivity to dilation k"
      -> e_pred is written for every k in --k-list (used by sweep_k.py).
  Honesty fix: the ORIGINAL pipeline masked ground-truth boxes. This script
      masks DETECTOR boxes (score >= --mask-gate) and also writes e_gt so the
      two can be compared (rho, tertile agreement).

Usage:
  python scripts/10_revision/recompute_proxy.py \
      --manifest  $WORK/cache_preds/sarship_val/manifest.csv \
      --label-root $WORK/sarship_root/labels/val \
      --out-manifest $WORK/cache_preds/sarship_val/manifest_proxy.csv \
      --k-list 0 3 5 8 12 16 --k-main 3 --mask-gate 0.25 \
      --target-class 0 \
      --report $OUT/sarship_val_proxy_report.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))
from revkit import BIN_LABELS, assign_bins, dump_json, read_manifest, tertile_edges  # noqa: E402
from eat_sar.inference_cache import load_pred, load_yolo_gt  # noqa: E402

RAYLEIGH_CV = float(np.sqrt((4 - np.pi) / np.pi))  # 0.5227


def _mask_boxes(mask, boxes, k, W, H):
    for x1, y1, x2, y2 in boxes:
        x1 = max(int(x1 - k), 0); y1 = max(int(y1 - k), 0)
        x2 = min(int(x2 + k), W); y2 = min(int(y2 + k), H)
        mask[y1:y2, x1:x2] = False


def _bg_stats(bg: np.ndarray) -> dict:
    bg = bg.astype(np.float64)
    mu = bg.mean(); sd = bg.std()
    I = bg ** 2                                   # intensity from amplitude DN
    m2 = (I ** 2).mean() / max(I.mean() ** 2, 1e-9)  # normalised 2nd intensity moment
    # single-look K-distribution: m2 = 2(1 + 1/nu)  ->  nu = 1 / (m2/2 - 1)
    nu = 1.0 / max(m2 / 2.0 - 1.0, 1e-6)
    hist, _ = np.histogram(bg, bins=64, range=(0, 255))
    p = hist / max(hist.sum(), 1); p = p[p > 0]
    ent = float(-(p * np.log2(p)).sum())
    return dict(mu=mu, sd=sd, cv=sd / max(mu, 1e-6), m2=m2, nu_hat=min(nu, 1e3),
                entropy=ent, spread=float(np.percentile(bg, 99) - np.median(bg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--label-root", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--k-list", type=int, nargs="+", default=[0, 3, 5, 8, 12, 16])
    ap.add_argument("--k-main", type=int, required=True,
                    help="dilation used for the main e_pred column (8 for SARDet, 3 for SAR-Ship)")
    ap.add_argument("--mask-gate", type=float, default=0.25,
                    help="detector score above which boxes are masked out of the background")
    ap.add_argument("--target-class", type=int, default=None)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    mf = read_manifest(args.manifest)
    if "stem" not in mf.columns:
        mf["stem"] = mf["image_path"].map(lambda p: Path(p).stem)
    label_root = Path(args.label_root)

    try:
        from tqdm import tqdm
        it = tqdm(mf.iterrows(), total=len(mf), desc="proxy")
    except ImportError:
        it = mf.iterrows()

    rows = []
    for _, r in it:
        im = np.array(Image.open(r["image_path"]).convert("L"))  # uint8 DN, no scaling
        H, W = im.shape
        boxes, scores, classes = load_pred(Path(r["pred_npz"]))
        if args.target_class is not None and len(classes):
            m = classes == args.target_class
            boxes, scores = boxes[m], scores[m]
        pb = boxes[scores >= args.mask_gate]
        gt = load_yolo_gt(label_root / f"{r['stem']}.txt", W, H)

        rec = dict(stem=r["stem"], n_pred_masked=int(len(pb)), n_gt=int(len(gt)))
        for k in sorted(set(args.k_list) | {args.k_main}):
            mask = np.ones((H, W), bool); _mask_boxes(mask, pb, k, W, H)
            bg = im[mask]
            if bg.size < 100:
                bg = im.ravel()
            rec[f"e_pred_k{k}"] = float(bg.std())
            rec[f"bgfrac_k{k}"] = float(mask.mean())
            if k == args.k_main:
                st = _bg_stats(bg)
                rec.update({f"bg_{kk}": v for kk, v in st.items()})
        # GT-masked (reference only — NOT used by the method)
        mask = np.ones((H, W), bool); _mask_boxes(mask, gt, args.k_main, W, H)
        bg = im[mask]; bg = bg if bg.size >= 100 else im.ravel()
        rec["e_gt"] = float(bg.std())
        rec["e_full"] = float(im.std())  # no masking at all
        rows.append(rec)

    px = pd.DataFrame(rows)
    px["e_pred"] = px[f"e_pred_k{args.k_main}"]
    out = mf.merge(px, on="stem", how="left")
    Path(args.out_manifest).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_manifest, index=False)
    print(f"[saved] {args.out_manifest}  ({len(out)} rows)")

    # ----------------------------------------------------------------- report
    from scipy.stats import spearmanr
    e = out["e_pred"].values
    rep = dict(manifest=args.manifest, n=int(len(out)), k_main=args.k_main,
               mask_gate=args.mask_gate, rayleigh_cv=RAYLEIGH_CV)
    rep["spearman_vs_e_pred"] = {
        c: float(spearmanr(e, out[c].values, nan_policy="omit").correlation)
        for c in ["e_gt", "e_full", "bg_mu", "bg_cv", "bg_entropy", "bg_spread", "bg_nu_hat"]
    }
    edges_pred = tertile_edges(e); edges_gt = tertile_edges(out["e_gt"].values)
    bp = assign_bins(e, edges_pred); bg_ = assign_bins(out["e_gt"].values, edges_gt)
    rep["tertile_agreement_pred_vs_gt"] = float((bp == bg_).mean())
    rep["edges_e_pred"] = edges_pred
    rep["quantiles_e_pred"] = {q: float(np.percentile(e, q)) for q in (1, 5, 25, 50, 75, 95, 99)}
    per_bin = {}
    for lb in BIN_LABELS:
        sub = out[bp == lb]
        per_bin[lb] = dict(n=int(len(sub)),
                           e_median=float(sub["e_pred"].median()),
                           mu_median=float(sub["bg_mu"].median()),
                           cv_median=float(sub["bg_cv"].median()),
                           cv_over_rayleigh=float(sub["bg_cv"].median() / RAYLEIGH_CV),
                           nu_hat_median=float(sub["bg_nu_hat"].median()),
                           entropy_median=float(sub["bg_entropy"].median()),
                           bgfrac_median=float(sub[f"bgfrac_k{args.k_main}"].median()))
    rep["per_bin"] = per_bin
    rep["e_pred_vs_k_spearman"] = {
        f"k{k}": float(spearmanr(e, out[f"e_pred_k{k}"].values).correlation)
        for k in sorted(set(args.k_list) | {args.k_main})
    }
    dump_json(rep, args.report)
    print(pd.DataFrame(per_bin).T.to_string())
    print("Spearman vs e_pred:", {k: round(v, 3) for k, v in rep["spearman_vs_e_pred"].items()})


if __name__ == "__main__":
    main()
