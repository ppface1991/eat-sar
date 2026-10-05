"""
Step W0 — Windows environment check for the revision kit.

Scans candidate roots for everything the revision pipeline needs:
  * .npz prediction caches   (dirs with many *.npz)
  * image directories       (dirs with many jpg/png)
  * YOLO label directories   (dirs with many txt in "cls cx cy w h" format)
  * VOC xml directories     (original SAR-Ship annotations, need conversion)
  * split lists              (train.txt / val.txt / test.txt ...)
  * .pt weights              (best.pt / last.pt)
and checks Python packages + GPU.

Usage (from the repo root, the folder that contains scripts\\ and src\\):
  python scripts\\10_revision\\check_windows_env.py [--roots D:\\data D:\\Project ...]

Default roots: the repo root, its parent, its grandparent, and the CWD.
Paste the full output back to the assistant.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv",
             ".idea", ".vscode", "figs", "paper"}
YOLO_LINE = re.compile(r"^\s*\d+(\.\d+)?(\s+-?\d+\.?\d*){4}\s*$")


def count_dir(p: Path) -> dict:
    """Count interesting files in one directory (cheap: single scandir)."""
    c = dict(npz=0, img=0, txt=0, yolo_txt=0, xml=0)
    try:
        with os.scandir(p) as it:
            for i, e in enumerate(it):
                if i > 300000:
                    break
                if not e.is_file():
                    continue
                s = e.name.lower()
                if s.endswith(".npz"):
                    c["npz"] += 1
                elif s.endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")):
                    c["img"] += 1
                elif s.endswith(".txt"):
                    c["txt"] += 1
                elif s.endswith(".xml"):
                    c["xml"] += 1
    except (PermissionError, OSError):
        pass
    return c


def sample_yolo(p: Path, n=8) -> bool:
    """True if the txt files in p look like YOLO labels."""
    try:
        files = sorted(p.glob("*.txt"))[:n]
        if not files:
            return False
        for f in files:
            try:
                lines = f.read_text(errors="ignore").strip().splitlines()
            except OSError:
                return False
            if lines and not YOLO_LINE.match(lines[0]):
                return False
        return True
    except OSError:
        return False


def scan_root(root: Path, max_depth=6):
    found = dict(npz_dirs=[], img_dirs=[], label_dirs=[], xml_dirs=[],
                 splits=[], weights=[])
    if not root.is_dir():
        return found
    base_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        if len(d.parts) - base_depth > max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [x for x in dirnames if x not in SKIP_DIRS]
        c = count_dir(d)
        if c["npz"] >= 100:
            found["npz_dirs"].append((str(d), c["npz"]))
        if c["img"] >= 500:
            found["img_dirs"].append((str(d), c["img"]))
        if c["txt"] >= 500 and sample_yolo(d):
            found["label_dirs"].append((str(d), c["txt"]))
        if c["xml"] >= 500:
            found["xml_dirs"].append((str(d), c["xml"]))
        for f in filenames:
            fl = f.lower()
            if fl.endswith(".pt") and any(k in fl for k in ("best", "last", "yolov8", "yolo")):
                try:
                    mb = (d / f).stat().st_size / 1e6
                except OSError:
                    mb = -1
                found["weights"].append((str(d / f), round(mb, 1)))
        for f in filenames:
            if f.lower() in ("train.txt", "val.txt", "test.txt",
                             "val_calib.txt", "val_test.txt"):
                fp = d / f
                try:
                    n = sum(1 for _ in open(fp, errors="ignore"))
                    if n >= 200:
                        found["splits"].append((str(fp), n))
                except OSError:
                    pass
    return found


def check_packages():
    def has(mod):
        try:
            __import__(mod)
            return "OK"
        except ImportError:
            return "MISSING"
    out = {m: has(m) for m in ("numpy", "pandas", "scipy", "PIL", "tqdm")}
    for m in ("torch", "ultralytics"):
        try:
            mod = __import__(m)
            extra = ""
            if m == "torch":
                extra = " (cuda: %s)" % ("YES" if mod.cuda.is_available() else "no")
            out[m] = "OK" + extra
        except ImportError:
            out[m] = "MISSING"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=None)
    args = ap.parse_args()

    print("=" * 70)
    print("GRSL revision kit — environment check")
    print("Repo root (this script's project):", REPO)
    print("Python:", sys.version.split()[0], "at", sys.executable)
    print("=" * 70)

    roots = [Path(p) for p in args.roots] if args.roots else []
    for cand in (REPO, REPO.parent, REPO.parent.parent, Path.cwd()):
        if cand.is_dir() and cand not in roots:
            roots.append(cand)
    roots = [r for r in roots if r.is_dir()]

    all_found = dict(npz_dirs=[], img_dirs=[], label_dirs=[], xml_dirs=[],
                     splits=[], weights=[])
    for r in roots:
        print(f"\n--- scanning {r} (this can take a minute on big dataset folders) ---")
        f = scan_root(r)
        for k in all_found:
            all_found[k].extend(f[k])

    def dedup(seq):
        seen, out = set(), []
        for x in seq:
            if x[0] not in seen:
                seen.add(x[0]); out.append(x)
        return out
    for k in all_found:
        all_found[k] = dedup(all_found[k])

    print("\n" + "=" * 70)
    print("FINDINGS")
    print("=" * 70)
    for name, key, min_n in (("Prediction caches (.npz >=100)", "npz_dirs", 1),
                             ("Image dirs (>=500 imgs)", "img_dirs", 2),
                             ("YOLO label dirs (>=500 txt)", "label_dirs", 2),
                             ("VOC xml dirs (>=500)", "xml_dirs", 1),
                             ("Split lists (>=200 lines)", "splits", 1),
                             ("Weights (.pt)", "weights", 1)):
        items = all_found[key]
        print(f"\n[{name}]  -> {len(items)} found")
        for p, n in items[:25]:
            print(f"    {p}   ({n})")

    print("\n" + "=" * 70)
    print("PYTHON PACKAGES")
    print("=" * 70)
    for m, s in check_packages().items():
        print(f"  {m:12s} {s}")

    # ---------------- verdicts ----------------
    print("\n" + "=" * 70)
    print("VERDICT (what you can run right now)")
    print("=" * 70)
    has_npz_val = any("val" in Path(p).name.lower() or "sardet100k_val" in p.lower() for p, _ in all_found["npz_dirs"])
    has_npz_test = any("test" in Path(p).name.lower() for p, _ in all_found["npz_dirs"])
    has_labels = len(all_found["label_dirs"]) >= 2
    has_weights = any("best" in Path(p).name.lower() and mb > 5 for p, mb in all_found["weights"])
    print(f"  npz cache (val)      : {'FOUND' if has_npz_val else 'NOT FOUND'}")
    print(f"  npz cache (test)     : {'FOUND' if has_npz_test else 'NOT FOUND'}")
    print(f"  YOLO label trees     : {'FOUND' if has_labels else 'NOT FOUND'}")
    print(f"  trained best.pt      : {'FOUND' if has_weights else 'NOT FOUND'}")
    if not (has_npz_val and has_npz_test):
        print("\n  -> npz caches missing: you must re-run cache_predictions.py (needs")
        print("     best.pt + the YOLO-format image/label trees), see RUN_REVISION_WINDOWS.md.")
    if not has_labels:
        print("\n  -> YOLO label trees missing: original datasets (Images/ + Annotations_new/)")
        print("     were on the server; conversion scripts are in scripts/01_data/.")
    print("\nPaste EVERYTHING above back to the assistant.")


if __name__ == "__main__":
    main()
