# Per-image clutter proxy values

One CSV per split, e.g. `sarship_test_e.csv`, with columns:

- `stem` — image identifier (as in the dataset)
- `e_pred` — the clutter proxy `e` (background grey-level standard
  deviation outside dilated detector boxes; main dilation k = 3 for
  SAR-Ship-Dataset, k = 8 for SARDet-100K)
- `e_pred_k<k>` — variants for the dilation sweep (Sec. III-D)
- `tertile` — low / medium / high clutter bin

`e` is label-free: it is computed from the image and the detector's own
boxes only. The letter (Sec. III-E) reports that this detector-mask
version agrees with a ground-truth-masked version to Spearman
ρ ≥ 0.99 with 98.3 / 96.6 % identical tertile assignments.

Generate (or regenerate) from the full working manifests:

```bash
python scripts/10_revision/make_public_manifests.py \
    --manifest work/cache_preds/sarship_val/manifest_proxy.csv \
    --dataset sarship --split val --out-dir .
# repeat for sarship/test, sardet/val, sardet/test
```
