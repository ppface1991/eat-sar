# Reproducing the letter

Full command sequence behind every number in the letter. Windows users can
run the same stages with `scripts/10_revision/run_revision.cmd <stage>`.
Paths below are placeholders; on the authors' machine `REPO` is the
repository root, `WORK` a scratch directory, `SS` / `SD` the converted
dataset roots, `CP` the prediction-cache root and `OUT` the output
directory.

```bash
export REPO=/path/to/eat-sar
export WORK=/path/to/work
export SS=$WORK/yolo/sarship
export SD=$WORK/yolo/sardet
export CP=$WORK/cache_preds
export OUT=$REPO/revision_out
cd $REPO && export PYTHONPATH=$REPO/src
```

1. **Convert datasets** (once): `scripts/10_revision/convert_datasets.py`
   — builds the common layout and YOLO labels; also fixes the
   SAR-Ship-Dataset deterministic test split
   (`make_sarship_test_split.py`).
2. **Cache predictions** (one GPU pass per dataset):
   `scripts/10_revision/cache_preds.py` — YOLOv8 inference on val + test,
   saved as one `.npz` per image (conf ≥ 0.01). Everything after this
   step is CPU-only.
3. **Compute the proxy**: `scripts/10_revision/recompute_proxy.py` on each
   of the four manifests — writes `e_pred` (and the k-sweep variants)
   into `*_proxy.csv` and a `*_proxy_report.json` (Sec. III-E statistics).
4. **Main evaluation + controls**: `scripts/10_revision/run_main_eval.py`
   — Table II, controls A–D and the test-fit oracle (Table III);
   `make_tables.py` assembles `numbers.json`.
5. **False-alarm stability**: `scripts/10_revision/cfar_eval.py` — the 12
   pre-specified budget/precision targets, bootstrap CIs and the
   random-binning placebo (Table I, Fig. 2). Add `--n-boot 200 --n-perm 20`
   for the settings in the letter.
6. **Robustness**: `kfold_cv.py` (10-fold CV), `sweep_k.py` (dilation
   sweep) — Sec. III-D.
7. **Transfer**: `cross_transfer.py` both directions — Sec. III-F.
8. **Figures**: `export_curves.py` (Fig. 3 data) and
   `paper/figs/replot_fig3.py` / `paper/figs/plot_cfar.py` (rendering).

Expected runtime on CPU after step 2: 20–40 min per dataset for steps 3–7.

Exact commands for every stage are in `scripts/10_revision/run_revision.cmd`
(one `python` call per line) and `scripts/10_revision/RUN_REVISION.md`
(Linux command reference).
