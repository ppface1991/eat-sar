from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from eat_sar.eval_metrics import match_predictions, precision_recall_f1
from eat_sar.inference_cache import load_pred, load_yolo_gt


def collect_image_data(row, label_root, target_class):
    boxes, scores, classes = load_pred(Path(row["pred_npz"]))
    if target_class is not None and len(classes) > 0:
        m = classes == target_class
        boxes = boxes[m]; scores = scores[m]
    img_path = Path(row["image_path"])
    with Image.open(img_path) as im:
        W, H = im.size
    stem = img_path.stem
    label_file = label_root / f"{stem}.txt"
    gt = load_yolo_gt(label_file, W, H)
    if target_class is not None and gt.shape[0] > 0 and label_file.exists():
        raw = np.loadtxt(label_file, ndmin=2)
        if raw.size > 0:
            mask = raw[:, 0].astype(int) == target_class
            gt = gt[mask]
    if len(scores) > 0:
        order = np.argsort(-scores)
        boxes = boxes[order]; scores = scores[order]
    return scores, boxes, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--label-root", required=True)
    ap.add_argument("--env-key", default="wind_bin")
    ap.add_argument("--env-value-key", default="clutter_std")
    ap.add_argument("--bin-order", default="low,medium,high")
    ap.add_argument("--bin-boundaries-json", required=True)
    ap.add_argument("--target-class", type=int, default=None)
    ap.add_argument("--scan-min", type=float, default=0.05)
    ap.add_argument("--scan-max", type=float, default=0.85)
    ap.add_argument("--scan-step", type=float, default=0.025)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    mf = pd.read_csv(args.manifest)
    mf = mf.dropna(subset=[args.env_key, args.env_value_key])
    print(f"Manifest rows: {len(mf)}")

    label_root = Path(args.label_root)
    thresholds = np.round(np.arange(args.scan_min, args.scan_max + 1e-9, args.scan_step), 4)
    print(f"Scanning {len(thresholds)} thresholds")

    per_image = {}
    for _, row in tqdm(mf.iterrows(), total=len(mf), desc="match"):
        scores, boxes, gt = collect_image_data(row, label_root, args.target_class)
        tp, n_gt = match_predictions(boxes, scores, gt, 0.5)
        per_image[row["stem"]] = (scores, tp, n_gt)

    bin_order = args.bin_order.split(",")
    print(f"Bins in manifest: {dict(mf[args.env_key].value_counts())}")

    results = []
    for bin_name, group in mf.groupby(args.env_key):
        for T in thresholds:
            all_tp, all_n_gt = [], 0
            for stem in group["stem"]:
                scores, tp, n_gt = per_image[stem]
                keep = scores >= T
                all_tp.append(tp[keep])
                all_n_gt += n_gt
            tp_cat = np.concatenate(all_tp) if all_tp else np.zeros(0, dtype=np.int8)
            P, R, F1 = precision_recall_f1(tp_cat, None, all_n_gt)
            results.append({"env_bin": bin_name, "threshold": float(T), "P": P, "R": R, "f1": F1})
    scan_df = pd.DataFrame(results)

    best = (scan_df.sort_values("f1", ascending=False)
                   .groupby("env_bin", as_index=False).head(1)
                   .sort_values("env_bin"))
    print("\nBest threshold per bin:")
    print(best.to_string(index=False))

    bdata = json.load(open(args.bin_boundaries_json))
    if "bins" in bdata and "labels" in bdata:
        bins = list(bdata["bins"])
        labels_in_file = list(bdata["labels"])
        if set(labels_in_file) == set(bin_order):
            bin_order = labels_in_file
    else:
        bins = list(bdata.get(args.env_value_key, bdata.get("bins", [])))
    if len(bins) != len(bin_order) + 1:
        bins = [float("-inf")] + list(bins) + [float("inf")]
    print(f"Bin boundaries: {bins} (order={bin_order})")

    best_map = {row["env_bin"]: float(row["threshold"]) for _, row in best.iterrows()}
    thresholds_ordered = [best_map[b] for b in bin_order]

    out = {
        "env_key": args.env_key,
        "env_value_key": args.env_value_key,
        "piecewise": {"bins": bins, "thresholds": thresholds_ordered, "bin_order": bin_order},
        "scan_df_summary": best.to_dict(orient="records"),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.out_json}")

if __name__ == "__main__":
    main()
