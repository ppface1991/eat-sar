"""
Split compound figures into individual single-panel vector PDFs.

Each output PDF is designed for SINGLE-COLUMN placement (~3.5" wide) so that
in-figure text remains clearly readable on the printed page even when LaTeX
places it inside a one-column `figure` environment. This solves the
'too-small text after typesetting' problem caused by 7.16"-wide compound
figures being squeezed into a column.

Splits performed:
  - fig_curves_sardet         -> _f1.pdf  +  _map.pdf
  - fig_curves_sarship        -> _f1.pdf  +  _map.pdf
  - fig_curves_sarship_recall -> _f1.pdf  +  _map.pdf
  - score_dists_sardet        -> _low.pdf + _med.pdf + _high.pdf
  - score_dists_sarship       -> _low.pdf + _med.pdf + _high.pdf

(fig_adaptive_sardet / sarship are already single-panel — no split needed.)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

# Reuse style + helpers from main script
from replot_all import (
    LABEL_MAP, COL_LOW, COL_MED, COL_HIGH, COL_ALL,
    smooth_curve, extract_hist_bars, FIG_DIR,
)

# Slightly larger fonts since each PDF is its own single-column figure
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size":        10,
    "axes.titlesize":   10,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  8.5,
    "axes.linewidth":   0.7,
    "lines.linewidth":  1.6,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "legend.frameon":    True,
    "legend.framealpha": 0.92,
    "legend.borderpad":  0.3,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.4,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})


# ============================================================
# Split: F1 / mAP curves into one PDF per metric panel
# ============================================================
def plot_curve_panel(json_path, side, metric_title, ylabel,
                      out_pdf, baseline_T=0.40, adaptive_Ts=None):
    """Render a single metric panel (F1 or mAP) for one dataset
    at single-column (~3.5") width with the legend below."""
    data = json.load(open(json_path))
    sub = data[side]

    fig, ax = plt.subplots(figsize=(3.5, 3.1))

    handles = []
    for k in ["calm", "moderate", "rough", "ALL"]:
        if k not in sub or len(sub[k][0]) == 0:
            continue
        label, color = LABEL_MAP[k]
        xs, ys = sub[k]
        xs_s, ys_s = smooth_curve(xs, ys, n_out=80)
        ln, = ax.plot(xs_s, ys_s, color=color, lw=1.6,
                      ls=("--" if k == "ALL" else "-"),
                      label=label)
        handles.append(ln)

    if adaptive_Ts:
        for k, T in adaptive_Ts.items():
            if k in LABEL_MAP:
                _, color = LABEL_MAP[k]
                ax.axvline(T, ls=":", lw=1.1, color=color, alpha=0.85)
    ax.axvline(baseline_T, ls="-", lw=0.9, color="k", alpha=0.55)

    ax.set_xlabel("Score threshold $T$")
    ax.set_ylabel(ylabel)
    ax.set_title(metric_title)
    ax.set_xlim(0.05, 0.92)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25, lw=0.4)
    ax.tick_params(direction="in", top=True, right=True)

    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color="k", lw=0.9,
                          label=f"Baseline $T={baseline_T:.2f}$"))
    if adaptive_Ts:
        handles.append(Line2D([0], [0], color="k", lw=1.1, ls=":",
                              label="Per-bin adaptive $T$"))
    ax.legend(handles=handles, loc="upper center",
              ncol=2, bbox_to_anchor=(0.5, -0.20),
              frameon=False, fontsize=8.0,
              columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout(rect=[0, 0.0, 1, 1])
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"  ✓ {out_pdf.name}")


# ============================================================
# Split: score distributions into one PDF per clutter bin
# ============================================================
def plot_score_panel(src_jpg, out_pdf, bin_idx, bin_title,
                      dataset_label, y_max=36.0, score_bins=40,
                      show_legend=False):
    """Render a single clutter-bin histogram as a column-width PDF."""
    data = extract_hist_bars(src_jpg, n_bins=3, score_bins=score_bins)
    d = data[bin_idx]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    edges = np.linspace(0, 1, score_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    bar_w = 1.0 / score_bins * 0.95
    tp_h = d["tp"] * y_max
    fp_h = d["fp"] * y_max

    ax.bar(centers, fp_h, width=bar_w, color=COL_HIGH, alpha=0.55,
           label="FP", edgecolor="none")
    ax.bar(centers, tp_h, width=bar_w, color=COL_LOW, alpha=0.65,
           label="TP", edgecolor="none")
    ax.axvline(0.40, color="black", lw=1.0, ls="--")
    ax.text(0.42, y_max * 0.88, "$T{=}0.40$", fontsize=9)
    ax.set_title(f"{bin_title} ({dataset_label})", fontsize=10)
    ax.set_xlabel("Detection score")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, y_max)
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(alpha=0.2, lw=0.4, axis="y")
    if show_legend:
        ax.legend(loc="upper right", fontsize=9,
                  handlelength=1.4, handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"  ✓ {out_pdf.name}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("Splitting compound figures into single-panel PDFs ...")

    # --- Curves: F1-balanced SARDet ---
    print("\n[Curves] SARDet F1-balanced ...")
    plot_curve_panel(
        FIG_DIR / "fig_curves_sardet_data.json",
        side="left",  metric_title="F1 vs. $T$",
        ylabel="F1",
        out_pdf=FIG_DIR / "fig_curves_sardet_f1.pdf",
        baseline_T=0.40,
        adaptive_Ts={"calm": 0.38, "moderate": 0.42, "rough": 0.36})
    plot_curve_panel(
        FIG_DIR / "fig_curves_sardet_data.json",
        side="right", metric_title="mAP@0.5 vs. $T$",
        ylabel="mAP@0.5",
        out_pdf=FIG_DIR / "fig_curves_sardet_map.pdf",
        baseline_T=0.40,
        adaptive_Ts={"calm": 0.38, "moderate": 0.42, "rough": 0.36})

    # --- Curves: F1-balanced SAR-Ship ---
    print("\n[Curves] SAR-Ship F1-balanced ...")
    plot_curve_panel(
        FIG_DIR / "fig_curves_sarship_data.json",
        side="left",  metric_title="F1 vs. $T$", ylabel="F1",
        out_pdf=FIG_DIR / "fig_curves_sarship_f1.pdf",
        baseline_T=0.40,
        adaptive_Ts={"calm": 0.30, "moderate": 0.40, "rough": 0.30})
    plot_curve_panel(
        FIG_DIR / "fig_curves_sarship_data.json",
        side="right", metric_title="mAP@0.5 vs. $T$", ylabel="mAP@0.5",
        out_pdf=FIG_DIR / "fig_curves_sarship_map.pdf",
        baseline_T=0.40,
        adaptive_Ts={"calm": 0.30, "moderate": 0.40, "rough": 0.30})

    # --- Curves: Recall-biased SAR-Ship ---
    print("\n[Curves] SAR-Ship Recall-biased ...")
    plot_curve_panel(
        FIG_DIR / "fig_curves_sarship_recall_data.json",
        side="left",  metric_title="F1 vs. $T$", ylabel="F1",
        out_pdf=FIG_DIR / "fig_curves_sarship_recall_f1.pdf",
        baseline_T=0.40,
        adaptive_Ts={"calm": 0.05, "moderate": 0.35, "rough": 0.60})
    plot_curve_panel(
        FIG_DIR / "fig_curves_sarship_recall_data.json",
        side="right", metric_title="mAP@0.5 vs. $T$", ylabel="mAP@0.5",
        out_pdf=FIG_DIR / "fig_curves_sarship_recall_map.pdf",
        baseline_T=0.40,
        adaptive_Ts={"calm": 0.05, "moderate": 0.35, "rough": 0.60})

    # --- Score distributions: SARDet (3 bins) ---
    print("\n[Score] SARDet per-bin ...")
    for idx, (suffix, title) in enumerate(
            [("low", "Low-clutter"),
             ("med", "Medium-clutter"),
             ("high", "High-clutter")]):
        plot_score_panel(
            src_jpg=FIG_DIR / "score_dists_sardet_orig.jpg",
            out_pdf=FIG_DIR / f"score_dists_sardet_{suffix}.pdf",
            bin_idx=idx, bin_title=title,
            dataset_label="SARDet-100K",
            show_legend=(idx == 2))      # legend only on high-clutter panel

    # --- Score distributions: SAR-Ship (3 bins) ---
    print("\n[Score] SAR-Ship per-bin ...")
    for idx, (suffix, title) in enumerate(
            [("low", "Low-clutter"),
             ("med", "Medium-clutter"),
             ("high", "High-clutter")]):
        plot_score_panel(
            src_jpg=FIG_DIR / "score_dists_sarship_orig.jpg",
            out_pdf=FIG_DIR / f"score_dists_sarship_{suffix}.pdf",
            bin_idx=idx, bin_title=title,
            dataset_label="SAR-Ship",
            show_legend=(idx == 2))

    print("\nAll split panels written.")


if __name__ == "__main__":
    main()
