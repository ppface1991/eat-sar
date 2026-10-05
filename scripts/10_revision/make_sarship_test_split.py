"""
Step R0a — Recover the SAR-Ship-Dataset TEST split.

July retrain used train.txt + val.txt under $WORK/sarship_root.
  * If images/all contains stems that are in NEITHER list  -> those are the
    untouched 10 % test split; write test.txt.
  * Otherwise (train+val cover everything) -> deterministically split val.txt
    50/50 (seed 42) into val_calib.txt (calibration) and val_test.txt (report)
    and PRINT A WARNING: the paper must then say "calibration / evaluation
    halves of the validation split".

Usage:
  python scripts/10_revision/make_sarship_test_split.py --root $WORK/sarship_root
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path


def _stems(txt: Path) -> set[str]:
    return {Path(l.strip()).stem for l in open(txt) if l.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--img-subdir", default="images/all")
    args = ap.parse_args()
    root = Path(args.root)
    imgs = {p.stem: p for p in (root / args.img_subdir).iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg")}
    tr, va = _stems(root / "train.txt"), _stems(root / "val.txt")
    rest = sorted(set(imgs) - tr - va)
    print(f"all={len(imgs)}  train={len(tr)}  val={len(va)}  remainder={len(rest)}")
    if len(rest) >= 1000:
        with open(root / "test.txt", "w") as f:
            for s in rest:
                f.write(str(imgs[s]) + "\n")
        print(f"[ok] wrote {root/'test.txt'} ({len(rest)} images)  -> use val.txt for calibration, test.txt for reporting")
    else:
        va_sorted = sorted(va); random.Random(42).shuffle(va_sorted)
        half = len(va_sorted) // 2
        for name, part in (("val_calib.txt", va_sorted[:half]), ("val_test.txt", va_sorted[half:])):
            with open(root / name, "w") as f:
                for s in part:
                    f.write(str(imgs[s]) + "\n")
            print(f"[ok] wrote {root/name} ({len(part)} images)")
        print("[WARN] no untouched test split found; paper must describe calibration/evaluation halves of val.")


if __name__ == "__main__":
    main()
