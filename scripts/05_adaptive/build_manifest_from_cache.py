from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--meta-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--img-ext", default=".jpg")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    img_dir = Path(args.img_dir)
    npz_files = sorted(cache_dir.glob("*.npz"))
    print(f"Found {len(npz_files)} npz files in {cache_dir}")

    rows = []
    missing_img = 0
    for npz in npz_files:
        stem = npz.stem
        img_path = None
        for ext in (args.img_ext, ".jpg", ".png", ".tif", ".tiff", ".jpeg"):
            cand = img_dir / f"{stem}{ext}"
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            missing_img += 1
            continue
        rows.append({"stem": stem, "image_path": str(img_path), "pred_npz": str(npz)})
    print(f"Matched {len(rows)} images; {missing_img} npz had no image")

    mf = pd.DataFrame(rows)
    meta = pd.read_csv(args.meta_csv)
    meta["stem"] = meta["file_name"].apply(lambda s: Path(s).stem)
    mf = mf.merge(meta, on="stem", how="left")
    n_with_env = mf["wind_bin"].notna().sum() if "wind_bin" in mf.columns else 0
    print(f"Rows with wind_bin: {n_with_env}/{len(mf)}")

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    mf.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")
    print(f"Columns: {list(mf.columns)}")
    if "wind_bin" in mf.columns:
        print(mf["wind_bin"].value_counts())

if __name__ == "__main__":
    main()
