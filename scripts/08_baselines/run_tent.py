"""
TENT (Test-time Entropy Minimization, Wang et al., ICLR 2021) baseline.

Adapts YOLOv8s BatchNorm affine parameters (gamma, beta) on the TEST images by
minimizing the entropy of the per-pixel class probability map. This is the
canonical Wang-2021 TENT recipe adapted to one-class SAR ship detection:

  - Freeze all conv weights
  - Re-enable BN running stats (model.train() but with momentum=0 for non-affine)
  - For each test batch, forward, compute predicted-class entropy over all anchor
    positions, backprop, step Adam(lr=1e-4) ONCE
  - Then run normal inference for evaluation

After TENT adaptation, we cache predictions (just like cache_predictions.py)
and reuse the existing adaptive-threshold evaluator. So this script outputs a
new manifest.csv and preds/ dir; downstream eval is identical to baseline.

Usage:
    python scripts/08_baselines/run_tent.py \\
        --config configs/default.yaml \\
        --weights /autodl-fs/data/eat-sar/runs/sardet100k_yolov8s/weights/best.pt \\
        --data /root/autodl-tmp/eat-sar-yolo/sardet100k/data.yaml \\
        --split test \\
        --meta-csv /autodl-fs/data/eat-sar/cache/meta/sardet100k_meta_with_env.csv \\
        --out-dir /autodl-fs/data/eat-sar/cache/preds/sardet100k_test_tent \\
        --tent-lr 1e-4 --tent-steps 1

Then evaluate with:
    python scripts/06_eval/evaluate_adaptive.py ... \\
        --manifest <out-dir>/manifest.csv --tag sardet100k_test_tent
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from eat_sar.inference_cache import save_pred  # noqa: E402
from eat_sar.utils import ensure_dir, load_config  # noqa: E402


def configure_model_for_tent(model: nn.Module) -> nn.Module:
    """
    TENT recipe (adapted for YOLOv8 + Ultralytics):
      - All conv weights frozen (requires_grad=False)
      - BN modules kept in eval() (use running stats, do NOT update them)
      - BN affine params (gamma, beta) are the only trainable ones

    Note: the canonical Wang-2021 TENT uses BATCH statistics (BN.train mode +
    track_running_stats=False). For YOLOv8 we keep BN in eval mode so that
    (a) Ultralytics' predict() can later fuse Conv+BN (which requires
    running_var to exist), and (b) we avoid NaN explosions on small batches.
    The only difference from Wang 2021 is statistic source; the OBJECTIVE
    (entropy minimization on the test stream w.r.t. BN affine) is identical.
    """
    model.eval()  # all BN use running stats
    for p in model.parameters():
        p.requires_grad = False
    bn_params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)):
            # keep eval mode (running stats), but allow affine grad
            m.track_running_stats = True  # keep running_var for later fuse()
            for p in (m.weight, m.bias):
                if p is not None:
                    p.requires_grad = True
                    bn_params.append(p)
    return model, bn_params


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Mean entropy of objectness sigmoid output. Lower = more confident."""
    # YOLOv8 outputs include objectness/classification logits. We treat each
    # detection's max-confidence as Bernoulli and minimize H(p) = -p log p -(1-p)log(1-p).
    p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    h = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    return h.mean()


# Global handle to captured logits via forward hook
_HOOK_LOGITS: list[torch.Tensor] = []


def _det_head_hook(module, inputs, output):
    """Capture the classification logits from the Detect head.
    YOLOv8 Detect.forward in EVAL mode returns one of:
      - tuple (y, x) where y is (B, 4+nc, A) processed preds AND x is the
        raw multi-scale list [(B, no, Hi, Wi), ...]  <- we want x
      - single (B, 4+nc, A) tensor (export mode)
      - list of (B, no, Hi, Wi) tensors (train mode)
    `no = reg_max*4 + nc`. We slice the cls channels and stash in _HOOK_LOGITS.
    For the processed (B, 4+nc, A) format, cls channels are at index [4:4+nc]
    and are POST-sigmoid (probabilities), so we convert back to logits via logit().
    """
    _HOOK_LOGITS.clear()
    reg_max = getattr(module, "reg_max", 16)
    nc = module.nc
    cls_start_raw = 4 * reg_max  # raw multi-scale layout

    # Unwrap tuple: (processed, raw_list) in eval, just list in train
    candidates = []
    if isinstance(output, tuple):
        for o in output:
            if isinstance(o, list):
                candidates.extend(o)
            elif torch.is_tensor(o):
                candidates.append(o)
    elif isinstance(output, list):
        candidates.extend(output)
    elif torch.is_tensor(output):
        candidates.append(output)

    for feat in candidates:
        if not torch.is_tensor(feat):
            continue
        # Raw multi-scale format: (B, reg_max*4+nc, H, W)
        if feat.ndim == 4 and feat.shape[1] >= cls_start_raw + nc:
            cls_logits = feat[:, cls_start_raw:cls_start_raw + nc, :, :].contiguous()
            _HOOK_LOGITS.append(cls_logits.flatten())
        # Processed format: (B, 4+nc, A) — channels are POST-sigmoid probabilities
        elif feat.ndim == 3 and feat.shape[1] == 4 + nc:
            probs = feat[:, 4:4 + nc, :].contiguous().clamp(1e-6, 1 - 1e-6)
            # Convert back to logits so downstream sigmoid+entropy is well-defined
            logits = torch.log(probs) - torch.log1p(-probs)
            _HOOK_LOGITS.append(logits.flatten())


def install_detect_hook(model: nn.Module):
    """Find the Detect head module (last layer) and register the forward hook."""
    detect = None
    for m in model.modules():
        if m.__class__.__name__ == "Detect":
            detect = m
    if detect is None:
        # fallback: last module in model.model list
        detect = model.model[-1] if hasattr(model, "model") else None
    if detect is None:
        raise RuntimeError("Could not locate YOLOv8 Detect head for TENT hook.")
    handle = detect.register_forward_hook(_det_head_hook)
    return handle, detect


def tent_step(model_pt, imgs: torch.Tensor, optimizer):
    """One TENT update step.

    We rely on a forward hook on the Detect head (installed once before the loop)
    that captures classification logits regardless of whether the raw forward
    returns dict / tuple / list / tensor. Forward in eval mode (so BN uses
    running stats); only the BN affine grads flow.
    """
    optimizer.zero_grad()
    raw_model = model_pt.model  # the underlying nn.Module
    # Keep model.eval() (BN running stats); affine still has requires_grad=True
    raw_model.eval()
    _ = raw_model(imgs)  # hook fills _HOOK_LOGITS
    if not _HOOK_LOGITS:
        return None
    logits = torch.cat(_HOOK_LOGITS)
    # Guard against NaN/Inf in the head output
    if not torch.isfinite(logits).all():
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
    # Clamp logits to a safe range before sigmoid to avoid degenerate entropy
    logits = logits.clamp(-20.0, 20.0)
    loss = softmax_entropy(logits)
    if not torch.isfinite(loss):
        return None
    loss.backward()
    # Clip gradients to prevent BN affine from blowing up on small batches
    torch.nn.utils.clip_grad_norm_(
        [p for p in raw_model.parameters() if p.requires_grad], max_norm=1.0
    )
    optimizer.step()
    return float(loss.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--meta-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tent-lr", type=float, default=1e-4)
    ap.add_argument("--tent-steps", type=int, default=1,
                    help="Number of optimizer steps per batch (Wang 2021 uses 1)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_dir(args.out_dir)
    imgsz = cfg["train"]["imgsz"]
    device = cfg["project"]["device"]

    data_yaml = yaml.safe_load(open(args.data))
    img_root = Path(data_yaml["path"]) / data_yaml[args.split]
    images = sorted([p for p in img_root.iterdir()
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")])

    meta = pd.read_csv(args.meta_csv)
    meta["stem"] = meta["file_name"].apply(lambda s: Path(s).stem)
    keep_stems = set(meta["stem"].tolist())
    images = [p for p in images if p.stem in keep_stems]
    print(f"TENT will adapt on {len(images)} test images.")

    from ultralytics import YOLO
    model_pt = YOLO(args.weights)
    raw = model_pt.model.to(device)
    raw, bn_params = configure_model_for_tent(raw)
    hook_handle, _det = install_detect_hook(raw)
    optimizer = torch.optim.Adam(bn_params, lr=args.tent_lr)

    # ====== TENT adaptation pass (no labels used) ======
    print("--- TENT adaptation pass ---")
    from PIL import Image as PILImage
    import torchvision.transforms.functional as TF

    def load_batch(paths):
        tensors = []
        for p in paths:
            im = PILImage.open(p).convert("RGB").resize((imgsz, imgsz))
            t = TF.to_tensor(im)
            tensors.append(t)
        return torch.stack(tensors).to(device)

    bs = args.batch_size
    losses = []
    for i in tqdm(range(0, len(images), bs), desc="tent adapt"):
        batch_paths = images[i:i + bs]
        imgs = load_batch(batch_paths)
        for _ in range(args.tent_steps):
            loss = tent_step(model_pt, imgs, optimizer)
            if loss is not None:
                losses.append(loss)

    pd.DataFrame({"step": range(len(losses)), "entropy_loss": losses}
                  ).to_csv(Path(out_dir) / "tent_loss.csv", index=False)
    print(f"Mean adaptation entropy: {np.mean(losses):.4f} -> "
          f"{np.mean(losses[-20:]):.4f}")
    hook_handle.remove()

    # ====== Cache predictions on adapted model ======
    print("--- Caching predictions (post-TENT) ---")
    # We bypass Ultralytics' predict() to avoid Conv+BN fuse, which would
    # discard the adapted BN modules. We forward through raw model in eval()
    # mode and do NMS ourselves using torchvision.
    from torchvision.ops import nms as tv_nms
    raw.eval()
    for p in raw.parameters():
        p.requires_grad = False

    # YOLOv8 in eval mode returns (preds, feats). preds is shape
    # (B, 4 + nc, num_anchors), where the first 4 are decoded xywh.
    manifest = []
    for i in tqdm(range(0, len(images), bs), desc="post-tent infer"):
        batch_paths = images[i:i + bs]
        imgs_batch = load_batch(batch_paths)
        with torch.no_grad():
            out = raw(imgs_batch)
        # parse output
        preds = out[0] if isinstance(out, (list, tuple)) else out
        # preds shape: (B, no, A). YOLOv8 in eval already decodes box reg via DFL.
        # no = 4 + nc; rows: [cx, cy, w, h, score_class_0, ...]
        if preds.ndim == 3 and preds.shape[1] >= 5:
            # transpose to (B, A, no)
            preds = preds.permute(0, 2, 1).contiguous()
        for bi, p in enumerate(batch_paths):
            pb = preds[bi]  # (A, no)
            xywh = pb[:, :4]
            cls_scores = pb[:, 4:]  # (A, nc)
            scores, cls_idx = cls_scores.max(dim=1)
            keep0 = scores >= args.conf
            xywh = xywh[keep0]; scores = scores[keep0]; cls_idx = cls_idx[keep0]
            # xywh -> xyxy
            x1 = xywh[:, 0] - xywh[:, 2] / 2
            y1 = xywh[:, 1] - xywh[:, 3] / 2
            x2 = xywh[:, 0] + xywh[:, 2] / 2
            y2 = xywh[:, 1] + xywh[:, 3] / 2
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
            if boxes_xyxy.numel() == 0:
                boxes_np = np.zeros((0, 4), dtype=np.float32)
                scores_np = np.zeros((0,), dtype=np.float32)
                classes_np = np.zeros((0,), dtype=np.int32)
            else:
                keep = tv_nms(boxes_xyxy, scores, args.iou)
                # scale boxes back to original image size
                im_orig = PILImage.open(p)
                W0, H0 = im_orig.size
                im_orig.close()
                sx = W0 / imgsz; sy = H0 / imgsz
                b = boxes_xyxy[keep].detach().cpu().numpy().astype(np.float32)
                b[:, [0, 2]] *= sx; b[:, [1, 3]] *= sy
                boxes_np = b
                scores_np = scores[keep].detach().cpu().numpy().astype(np.float32)
                classes_np = cls_idx[keep].detach().cpu().numpy().astype(np.int32)
            save_pred(Path(out_dir) / f"{p.stem}.npz",
                       boxes_np, scores_np, classes_np)
            manifest.append({
                "stem": p.stem,
                "image_path": str(p),
                "pred_npz": str(Path(out_dir) / f"{p.stem}.npz"),
            })
    mf = pd.DataFrame(manifest)
    # ensure stem exists on the meta side for the join
    if "stem" not in meta.columns:
        meta = meta.copy()
        meta["stem"] = meta["file_name"].apply(lambda s: Path(s).stem)
    mf = mf.merge(meta, on="stem", how="left")
    mf.to_csv(Path(out_dir) / "manifest.csv", index=False)
    print(f"Saved manifest: {Path(out_dir)/'manifest.csv'}")
    print("Now run scripts/06_eval/evaluate_adaptive.py with --manifest <this>.")


if __name__ == "__main__":
    main()
