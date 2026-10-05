"""
Shared helpers for the GRSL-01920-2026 revision experiments.

Everything operates on the cached predictions (one .npz per image, conf>=0.001)
plus YOLO-format label txt files, so NO GPU / re-inference is needed.

Manifest CSV must have at least:  stem, image_path, pred_npz
Optional (added by recompute_proxy.py):  e_pred, e_gt, clutter_mean, ...

Key definitions (these are what the paper reports):
  * gated AP@0.5 : 101-point interpolated AP computed on the detection list
                   AFTER the score gate (score >= T), ranked by original scores.
  * P / R / F1   : single operating point at T (same gated list).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.append(str(_SRC))

from eat_sar.eval_metrics import (average_precision, match_predictions,  # noqa: E402
                                    precision_recall_f1)
from eat_sar.inference_cache import load_pred, load_yolo_gt  # noqa: E402

def read_manifest(path) -> pd.DataFrame:
    """pd.read_csv with stem kept as string (SARDet stems are pure digits with
    leading zeros; the default dtype inference would corrupt them: 000123 -> 123)."""
    return pd.read_csv(path, dtype={"stem": str})


BIN_LABELS = ["low", "medium", "high"]
DEFAULT_SCAN = np.round(np.arange(0.05, 0.85 + 1e-9, 0.025), 4)


# --------------------------------------------------------------------------- #
# Per-image loading + matching (done once, reused by every experiment)
# --------------------------------------------------------------------------- #
def load_image_record(row, label_root: Path, target_class: int | None,
                      iou_thresh: float = 0.5):
    """Return dict(scores, tp, n_gt, boxes, gt, W, H) with preds sorted desc."""
    boxes, scores, classes = load_pred(Path(row["pred_npz"]))
    if target_class is not None and len(classes):
        m = classes == target_class
        boxes, scores = boxes[m], scores[m]
    img_path = Path(row["image_path"])
    with Image.open(img_path) as im:
        W, H = im.size
    stem = row["stem"] if "stem" in row else img_path.stem
    lbl = label_root / f"{stem}.txt"
    gt = load_yolo_gt(lbl, W, H)
    if target_class is not None and gt.shape[0] > 0:
        raw = np.loadtxt(lbl, ndmin=2)
        if raw.size:
            gt = gt[raw[:, 0].astype(int) == target_class]
    if len(scores):
        o = np.argsort(-scores)
        boxes, scores = boxes[o], scores[o]
    tp, n_gt = match_predictions(boxes, scores, gt, iou_thresh=iou_thresh)
    return dict(scores=scores.astype(np.float32), tp=tp.astype(np.int8),
                n_gt=int(n_gt), boxes=boxes, gt=gt, W=W, H=H, stem=stem)


def load_all(manifest: pd.DataFrame, label_root: Path, target_class: int | None,
             desc: str = "load") -> dict:
    try:
        from tqdm import tqdm
        it = tqdm(manifest.iterrows(), total=len(manifest), desc=desc)
    except ImportError:  # pragma: no cover
        it = manifest.iterrows()
    return {r["stem"]: load_image_record(r, label_root, target_class) for _, r in it}


# --------------------------------------------------------------------------- #
# Metrics for a set of images under a per-image threshold vector
# --------------------------------------------------------------------------- #
def evaluate(records: dict, stems, T_per_stem) -> dict:
    """T_per_stem: scalar or dict stem->T. Returns P,R,F1,AP(gated),n_pred,n_gt."""
    s_all, tp_all, n_gt = [], [], 0
    for st in stems:
        rec = records[st]
        T = T_per_stem if np.isscalar(T_per_stem) else T_per_stem[st]
        keep = rec["scores"] >= T
        s_all.append(rec["scores"][keep]); tp_all.append(rec["tp"][keep])
        n_gt += rec["n_gt"]
    if not s_all:
        return dict(P=0., R=0., F1=0., AP=0., n_pred=0, n_gt=n_gt)
    s = np.concatenate(s_all); t = np.concatenate(tp_all)
    P, R, F1 = precision_recall_f1(t, s, n_gt)
    AP = average_precision(t, s, n_gt)
    return dict(P=P, R=R, F1=F1, AP=AP, n_pred=int(len(s)), n_gt=n_gt)


def filter_gt(records: dict, stems, e, require_gt: bool):
    """Optionally keep only images that contain >=1 GT ship (paper's 'ship subset')."""
    stems = np.asarray(stems); e = np.asarray(e, dtype=float)
    if not require_gt:
        return stems, e
    keep = np.array([records[s]["n_gt"] > 0 for s in stems])
    print(f"[filter_gt] kept {keep.sum()}/{len(stems)} images with >=1 GT ship")
    return stems[keep], e[keep]


def ungated_ap(records: dict, stems) -> float:
    """Detector-only AP@0.5 (no score gate) — sanity reference."""
    return evaluate(records, stems, 0.0)["AP"]


# --------------------------------------------------------------------------- #
# Binning + calibration
# --------------------------------------------------------------------------- #
def tertile_edges(e: np.ndarray, K: int = 3) -> list[float]:
    qs = np.quantile(e, np.linspace(0, 1, K + 1)[1:-1])
    return [-np.inf, *map(float, qs), np.inf]


def assign_bins(e: np.ndarray, edges: list[float], labels=None) -> np.ndarray:
    labels = labels or (BIN_LABELS if len(edges) == 4 else [f"b{i}" for i in range(len(edges) - 1)])
    idx = np.clip(np.searchsorted(edges, e, side="right") - 1, 0, len(labels) - 1)
    return np.asarray(labels)[idx]


def fit_per_bin_T(records: dict, stems, bins: np.ndarray, labels,
                  scan=DEFAULT_SCAN, criterion: str = "f1",
                  min_precision: float = 0.85) -> dict:
    """Return {label: T_b}. criterion: 'f1' or 'recall@P>=min_precision'."""
    out = {}
    stems = np.asarray(stems)
    for lb in labels:
        sub = stems[bins == lb]
        best_T, best_val = None, -1.0
        for T in scan:
            m = evaluate(records, sub, float(T))
            if criterion == "f1":
                val = m["F1"]
            else:
                val = m["R"] if m["P"] >= min_precision else -1.0
            if val > best_val:
                best_val, best_T = val, float(T)
        out[lb] = best_T if best_T is not None else float(scan[-1])
    return out


def fit_global_T(records: dict, stems, scan=DEFAULT_SCAN) -> float:
    best_T, best = None, -1.0
    for T in scan:
        f1 = evaluate(records, stems, float(T))["F1"]
        if f1 > best:
            best, best_T = f1, float(T)
    return best_T


def linear_fit(edges: list[float], e: np.ndarray, bins: np.ndarray,
               T_b: dict, labels) -> tuple[float, float, list[float]]:
    """Least-squares T = a*e + b through (median e of bin, T_b). Returns a, b, anchors."""
    anchors = [float(np.median(e[bins == lb])) for lb in labels]
    t = np.array([T_b[lb] for lb in labels])
    A = np.stack([anchors, np.ones(len(anchors))], 1)
    a, b = np.linalg.lstsq(A, t, rcond=None)[0]
    return float(a), float(b), anchors


def apply_policy(e: np.ndarray, stems, edges, T_b: dict, labels,
                 linear: tuple[float, float] | None = None,
                 clip=(0.05, 0.90)) -> dict:
    """Return stem->T for piecewise (default) or linear policy."""
    if linear is None:
        bins = assign_bins(e, edges, labels)
        return {st: T_b[b] for st, b in zip(stems, bins)}
    a, b = linear
    return {st: float(np.clip(a * ev + b, *clip)) for st, ev in zip(stems, e)}


def stratified_eval(records, stems, e, edges, labels, T_per_stem, baseline_T):
    """Rows for ALL + each bin, for baseline and adaptive."""
    stems = np.asarray(stems)
    bins = assign_bins(e, edges, labels)
    rows = []
    for name, Tmap in (("baseline", baseline_T), ("adaptive", T_per_stem)):
        for lb in ["ALL", *labels]:
            sub = stems if lb == "ALL" else stems[bins == lb]
            m = evaluate(records, sub, Tmap)
            Tval = (baseline_T if name == "baseline"
                    else (np.nan if lb == "ALL" else float(np.median([T_per_stem[s] for s in sub])) if len(sub) else np.nan))
            rows.append(dict(strategy=name, bin=lb, n_img=int(len(sub)), T=Tval, **m))
    df = pd.DataFrame(rows)
    # delta AP columns
    base = df[df.strategy == "baseline"].set_index("bin")["AP"]
    df["dAP_pp"] = df.apply(lambda r: 100 * (r["AP"] - base[r["bin"]]), axis=1)
    return df


def dump_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_np_default)
    print(f"[saved] {path}")


def _np_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if o is np.inf:
        return "inf"
    return str(o)
