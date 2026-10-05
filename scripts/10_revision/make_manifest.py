"""
Step R0b — Build a minimal manifest.csv (stem, image_path, pred_npz) from a
prediction-cache directory and an image directory (or an explicit list file).

Usage:
  python scripts/10_revision/make_manifest.py \
      --cache-dir $WORK/cache_preds/sarship_test \
      --img-dir   $WORK/sarship_root/images/all \
      [--list $WORK/sarship_root/test.txt]   # optional: only these images
      --out $WORK/cache_preds/sarship_test/manifest.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--list", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cache = Path(args.cache_dir); img_dir = Path(args.img_dir)
    npz = {p.stem: p for p in cache.glob("*.npz")}
    if args.list:
        names = [l.strip() for l in open(args.list) if l.strip()]
        imgs = [Path(n) if Path(n).is_absolute() else img_dir / Path(n).name for n in names]
    else:
        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT]
    rows, miss = [], 0
    for p in sorted(imgs):
        if p.stem in npz:
            rows.append(dict(stem=p.stem, image_path=str(p), pred_npz=str(npz[p.stem])))
        else:
            miss += 1
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[saved] {args.out}: {len(df)} rows  (images without cache: {miss})")


if __name__ == "__main__":
    main()
