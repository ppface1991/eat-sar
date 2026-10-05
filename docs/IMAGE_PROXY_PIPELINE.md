# Image-Derived Sea-State Proxy Pipeline (for datasets without lat/lon/time)

When the SAR dataset annotations DON'T contain per-image lat/lon/datetime
(this is the case for SARDet-100K's public release), the ERA5 route is not
applicable. We instead estimate sea-state strength **directly from the SAR image
itself** using background clutter statistics — this is well established in CFAR
literature.

## Method (image-derived)

For each ship-containing image:

1. Take all ground-truth ship bounding boxes; dilate by 8 px to avoid
   ship-pixel leakage into the background set.
2. Mask out these regions. The remaining pixels are treated as "sea clutter".
3. Compute four statistics over the masked region:
   - `clutter_mean`     – mean intensity
   - `clutter_std`      – standard deviation
   - `clutter_cv`       – coefficient of variation (`std / mean`)
   - `clutter_entropy`  – Shannon entropy of intensity histogram

Physical intuition:
- Calm sea  → low backscatter mean, low std, low entropy
- Rough sea → stronger / more inhomogeneous backscatter → higher std + entropy

We use **clutter_std** as the primary proxy (matches the most common CFAR-style
practice); the others are computed for ablation.

## Story for the paper

> Sea-State-Aware Adaptive Thresholding for SAR Ship Detection via
> Image-Derived Clutter Statistics

- Method: post-processing only; thresholds adapt to per-image clutter stats
- Inputs: SAR image + detector; **no external data needed** (no ERA5, no AIS)
- Plug-and-play: works with any existing detector
- Validated on SARDet-100K (ship subset) + SAR-Ship-Dataset

## Pipeline

```bash
# Step 1: extract per-image metadata (lat/lon/time fields will be empty — OK)
python scripts/01_data/extract_sardet100k_meta.py --config configs/default.yaml
python scripts/01_data/extract_sarship_meta.py    --config configs/default.yaml

# Step 2 (replaces ERA5): compute per-image clutter statistics & bin
python scripts/02_env/compute_image_clutter.py --config configs/default.yaml \
    --meta-csv cache/meta/sardet100k_meta.csv \
    --dataset sardet100k --proxy clutter_std

python scripts/02_env/compute_image_clutter.py --config configs/default.yaml \
    --meta-csv cache/meta/sarship_meta.csv \
    --dataset sarship --proxy clutter_std

# Step 2c: summarise (Table 1 of the paper)
python scripts/02_env/summarize_env_strata.py \
    --meta-csv cache/meta/sardet100k_meta_with_env.csv \
                cache/meta/sarship_meta_with_env.csv \
    --out runs/env_strata.csv
```

After this point, **all downstream scripts are unchanged** — they read
`wind_speed` and `wind_bin` columns from the CSV which we've aliased to the
clutter proxy. Just continue with steps 3–7 from `README.md`.

In the paper, relabel `wind_speed` → `clutter_std` and
`wind_bin` (low/mid/high) → `sea_state` (calm/moderate/rough).

## Ablation: which proxy is best?

Re-run step 2 with each proxy and compare downstream gains:

```bash
for P in clutter_std clutter_cv clutter_entropy clutter_mean; do
    python scripts/02_env/compute_image_clutter.py --config configs/default.yaml \
        --meta-csv cache/meta/sardet100k_meta.csv \
        --dataset sardet100k --proxy $P
    mv cache/meta/sardet100k_meta_with_env.csv \
       cache/meta/sardet100k_meta_with_env_${P}.csv
done
```

Then run threshold scan + adaptive eval for each, and report mAP/F1 in
ablation table.
