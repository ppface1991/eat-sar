"""
Regenerate all paper figures as vector PDFs at IEEE-publication quality.

IEEE GRSL constraints:
- Two-column letter, column width ~3.5", page width ~7.16".
- Body text typically 8pt (IEEE Tran style); figure captions 8pt.
- Recommended minimum in-figure font: 6pt printed = ~8pt source for half-column,
  ~9pt source for full-column.
- We use Type-1 font (matplotlib default) and vector PDF output → text remains
  text in the PDF, fully zoomable, fully embedded.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

# ---------------- IEEE publication style ----------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size":        9,         # base font
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  7.5,
    "legend.title_fontsize": 8,
    "axes.linewidth":   0.7,
    "lines.linewidth":  1.4,
    "lines.markersize": 2.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "legend.frameon":   True,
    "legend.framealpha": 0.92,
    "legend.borderpad": 0.3,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.4,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype":     42,        # TrueType embedding (editable text)
    "ps.fonttype":      42,
})

# ---------------- Color palette (3-bin clutter) ----------------
COL_LOW    = "#2E86AB"   # blue
COL_MED    = "#E07A5F"   # salmon
COL_HIGH   = "#8B2C00"   # dark red
COL_ALL    = "#444444"   # dark gray
LABEL_MAP = {
    "calm":     ("Low-clutter",    COL_LOW),
    "moderate": ("Medium-clutter", COL_MED),
    "rough":    ("High-clutter",   COL_HIGH),
    "ALL":      ("All",            COL_ALL),
}

FIG_DIR = Path("/home/user/workspace/eat-sar/paper/figs")


# =========================================================
# Helper: smooth a noisy extracted curve back to a clean line
# =========================================================
def smooth_curve(xs, ys, n_out=80):
    xs = np.array(xs); ys = np.array(ys)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    uniq_x, idx = np.unique(np.round(xs, 5), return_inverse=True)
    uniq_y = np.array([ys[idx == i].mean() for i in range(len(uniq_x))])

    # Outlier removal: discard any point whose y deviates from a 5-point
    # rolling median by more than 0.12 (these come from pixel-extraction
    # noise where the dashed ALL curve was confused with grid/legend text).
    if len(uniq_y) >= 7:
        from numpy.lib.stride_tricks import sliding_window_view
        med = np.median(sliding_window_view(
            np.pad(uniq_y, 2, mode="edge"), 5), axis=1)
        keep = np.abs(uniq_y - med) < 0.12
        if keep.sum() >= 5:
            uniq_x = uniq_x[keep]
            uniq_y = uniq_y[keep]

    # Light moving-average smoothing
    if len(uniq_y) >= 5:
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        padded = np.concatenate([uniq_y[:2][::-1], uniq_y, uniq_y[-2:][::-1]])
        smoothed = np.convolve(padded, kernel, mode="valid")
        uniq_y = smoothed[:len(uniq_x)]
    f = PchipInterpolator(uniq_x, uniq_y, extrapolate=False)
    x_new = np.linspace(uniq_x[0], uniq_x[-1], n_out)
    y_new = f(x_new)
    return x_new, y_new


# =========================================================
# Fig.1 + Fig.3: F1 / mAP vs T curves
# =========================================================
def plot_curves(json_path, out_pdf, baseline_T=0.40,
                adaptive_Ts=None, title_F1="F1 vs. $T$",
                title_mAP="mAP@0.5 vs. $T$",
                show_recall_lines=False):
    """adaptive_Ts: dict {bin_label: T} to draw vertical dotted per-bin lines."""
    data = json.load(open(json_path))
    # Designed for two-column-spanning placement (\figure*), 7.16" wide.
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))

    for ax, side, metric_title, ylabel in zip(
            axes, ["left", "right"],
            [title_F1, title_mAP],
            ["F1", "mAP@0.5"]):
        sub = data[side]
        for k in ["calm", "moderate", "rough", "ALL"]:
            if k not in sub or len(sub[k][0]) == 0:
                continue
            label, color = LABEL_MAP[k]
            xs, ys = sub[k]
            xs_s, ys_s = smooth_curve(xs, ys, n_out=80)
            ax.plot(xs_s, ys_s, color=color, lw=1.5,
                    ls=("--" if k == "ALL" else "-"),
                    label=label)
        if adaptive_Ts:
            for k, T in adaptive_Ts.items():
                if k in LABEL_MAP:
                    _, color = LABEL_MAP[k]
                    ax.axvline(T, ls=":", lw=1.0, color=color, alpha=0.85)
        ax.axvline(baseline_T, ls="-", lw=0.8, color="k", alpha=0.55)
        ax.set_xlabel("Score threshold $T$")
        ax.set_ylabel(ylabel)
        ax.set_title(metric_title)
        ax.set_xlim(0.05, 0.92)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(direction="in", top=True, right=True)

    # Single legend below
    handles, labels = axes[0].get_legend_handles_labels()
    # Add a baseline marker entry
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color="k", lw=0.8,
                          label=f"Baseline $T={baseline_T:.2f}$"))
    if adaptive_Ts:
        handles.append(Line2D([0], [0], color="k", lw=1.0, ls=":",
                              label="Per-bin adaptive $T$"))
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), bbox_to_anchor=(0.5, -0.05),
               frameon=False, fontsize=7.5,
               columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"  ✓ {out_pdf}")


# =========================================================
# Fig.2: Adaptive T(e) per dataset
# =========================================================
def plot_adaptive(anchors, lin_a, lin_b, e_range, out_pdf,
                  title, e_label="Clutter proxy $e$ (a.u.)"):
    """anchors: list of (e_center, T) in low-mid-high order.
       Piecewise constant uses bin edges between anchors."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))    # column-width
    e_dense = np.linspace(e_range[0], e_range[1], 400)

    # Piecewise: assign each e to the nearest anchor's bin (use midpoints)
    e_anchors = np.array([a[0] for a in anchors])
    T_anchors = np.array([a[1] for a in anchors])
    # Edges between anchors
    edges = (e_anchors[:-1] + e_anchors[1:]) / 2
    edges_full = np.concatenate([[-np.inf], edges, [np.inf]])
    T_piece = np.zeros_like(e_dense)
    for i in range(len(T_anchors)):
        mask = (e_dense >= edges_full[i]) & (e_dense < edges_full[i + 1])
        T_piece[mask] = T_anchors[i]

    ax.plot(e_dense, T_piece, color=COL_LOW, lw=1.8,
            label="Piecewise $T(e)$")
    ax.plot(e_dense, lin_a * e_dense + lin_b, color=COL_MED, lw=1.4, ls="--",
            label=f"Linear $T(e)={lin_a:+.4f}\\,e{lin_b:+.3f}$".replace("+-", "-"))
    ax.scatter(e_anchors, T_anchors, c="k", s=18, zorder=5,
               label="Per-bin argmax")

    # Annotate bin centers
    bin_names = ["low", "med", "high"]
    for (e, T), name in zip(anchors, bin_names):
        ax.annotate(name, xy=(e, T), xytext=(4, 6),
                    textcoords="offset points", fontsize=6.5,
                    color="dimgray")

    ax.set_xlabel(e_label)
    ax.set_ylabel("Score threshold $T(e)$")
    ax.set_title(title)
    ax.set_xlim(e_range)
    # Auto y limits with padding
    y_all = np.concatenate([T_anchors,
                            [lin_a * e_range[0] + lin_b,
                             lin_a * e_range[1] + lin_b]])
    pad = max(0.02, (y_all.max() - y_all.min()) * 0.4)
    ax.set_ylim(y_all.min() - pad, y_all.max() + pad)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(loc="best", fontsize=6.8)
    ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"  ✓ {out_pdf}")


# =========================================================
# Fig.4: Score distribution histograms (TP vs FP) per clutter bin
#         — extracted from existing PNG/JPG by bar-height sampling
# =========================================================
def extract_hist_bars(jpg_path, n_bins=3, score_bins=40):
    """Extract per-bin TP and FP histogram bar heights from the existing
    score_dists_*.jpg figure. Returns a list of length n_bins, each a dict
    {\"tp\": np.array[score_bins], \"fp\": np.array[score_bins]} of bar heights."""
    from PIL import Image
    im = Image.open(jpg_path).convert("RGB")
    arr = np.array(im)
    H, W = arr.shape[:2]

    # Step 1: find top and bottom axis lines (full-width horizontal black runs)
    gray = arr.mean(axis=2)
    row_dark = (gray < 100).sum(axis=1)
    candidate_rows = np.where(row_dark > W * 0.6)[0]
    top_row = candidate_rows.min()
    bot_row = candidate_rows.max()
    plot_h = bot_row - top_row
    # Plot y range: 0 to ~36 density

    # Original colors after alpha=0.55 blend with white:
    #   TP ≈ (140, 190, 210) - blue-dominant
    #   FP ≈ (240, 156, 145) - red-dominant
    # Use ratio-based detection so we don't catch axis text or gridlines.
    R, G, B = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
    is_colored = (np.abs(R - G) > 20) | (np.abs(R - B) > 20)
    # blue-dominant
    is_tp = is_colored & (B > R + 20) & (B > 150) & (B < 240)
    # red-dominant
    is_fp = is_colored & (R > B + 30) & (R > 200) & (G < 200)

    # Step 2: for each subplot, find the left and right axis lines (column
    # density of dark pixels within [top_row, bot_row]).
    is_axis = gray[top_row:bot_row + 1] < 100
    col_dark_in_plot = is_axis.sum(axis=0)

    # For an n_bins layout we expect 2*n_bins vertical axis lines (left+right
    # of each subplot, possibly shared). Sort columns with sufficient dark
    # pixel density and cluster them.
    vert_thresh = is_axis.shape[0] * 0.7
    axis_cols = np.where(col_dark_in_plot > vert_thresh)[0]
    # Cluster contiguous columns into single axes lines
    if len(axis_cols) == 0:
        # fallback to even split
        sub_w = W // n_bins
        boxes = [(i * sub_w + int(sub_w * 0.1),
                  (i + 1) * sub_w - int(sub_w * 0.05))
                 for i in range(n_bins)]
    else:
        clusters = [[axis_cols[0]]]
        for c in axis_cols[1:]:
            if c - clusters[-1][-1] < 10:
                clusters[-1].append(c)
            else:
                clusters.append([c])
        cluster_centers = [int(np.mean(c)) for c in clusters]
        # Expect 2*n_bins centers (left+right of each subplot)
        if len(cluster_centers) == 2 * n_bins:
            boxes = [(cluster_centers[2 * i], cluster_centers[2 * i + 1])
                     for i in range(n_bins)]
        elif len(cluster_centers) == n_bins + 1:
            # Shared internal axes
            boxes = [(cluster_centers[i], cluster_centers[i + 1])
                     for i in range(n_bins)]
        else:
            # use first and last as outer bounds, even-split inside
            x0_all = cluster_centers[0]
            x1_all = cluster_centers[-1]
            sub_w = (x1_all - x0_all) // n_bins
            boxes = [(x0_all + i * sub_w, x0_all + (i + 1) * sub_w)
                     for i in range(n_bins)]

    # Step 3: per-subplot, sample bar heights.
    # Skip a top-right rectangle in the last subplot to avoid catching the
    # legend color patches (TP/FP swatches).
    results = []
    for sp_idx, (left_col, right_col) in enumerate(boxes):
        plot_w = right_col - left_col
        bin_w = plot_w / score_bins
        tp_heights = np.zeros(score_bins)
        fp_heights = np.zeros(score_bins)
        for b in range(score_bins):
            c0 = int(left_col + b * bin_w)
            c1 = int(left_col + (b + 1) * bin_w)
            tp_col = is_tp[top_row:bot_row, c0:c1].copy()
            fp_col = is_fp[top_row:bot_row, c0:c1].copy()
            # Mask out legend region (top-right ~12% width x top 15% height
            # of the last subplot)
            if sp_idx == len(boxes) - 1:
                # Position of this bin's right edge relative to last subplot
                bin_x_frac = (b + 1) / score_bins
                if bin_x_frac > 0.82:
                    tp_col[:int(plot_h * 0.18), :] = False
                    fp_col[:int(plot_h * 0.18), :] = False
            if tp_col.any():
                tp_pixels = np.where(tp_col.any(axis=1))[0]
                tp_heights[b] = (plot_h - tp_pixels.min()) / plot_h
            if fp_col.any():
                fp_pixels = np.where(fp_col.any(axis=1))[0]
                fp_heights[b] = (plot_h - fp_pixels.min()) / plot_h
        results.append({"tp": tp_heights, "fp": fp_heights})
    return results


def plot_score_dists(src_jpg, out_pdf, title, y_max=36.0):
    """Re-render the score-distribution figure as a clean vector PDF,
    using bar heights extracted from the original raster. y_max is the
    density-axis upper limit (matching the original ~35 max)."""
    n_bins = 3
    score_bins = 40
    data = extract_hist_bars(src_jpg, n_bins=n_bins, score_bins=score_bins)

    # Full text-width design (3 panels) at 7.16" wide, embedded as
    # \figure* (two-column-spanning). Per-panel font 8-9pt is comfortably
    # readable on the printed page.
    fig, axes = plt.subplots(1, n_bins, figsize=(7.16, 2.0),
                              sharex=True, sharey=True)
    bin_titles = ["Low-clutter", "Medium-clutter", "High-clutter"]
    edges = np.linspace(0, 1, score_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    bar_w = 1.0 / score_bins * 0.95

    for ax, btitle, d in zip(axes, bin_titles, data):
        tp_h = d["tp"] * y_max
        fp_h = d["fp"] * y_max
        ax.bar(centers, fp_h, width=bar_w, color=COL_HIGH, alpha=0.55,
               label="FP", edgecolor="none")
        ax.bar(centers, tp_h, width=bar_w, color=COL_LOW, alpha=0.65,
               label="TP", edgecolor="none")
        ax.axvline(0.40, color="black", lw=0.9, ls="--")
        ax.text(0.42, y_max * 0.88, "$T{=}0.40$", fontsize=7.5)
        ax.set_title(btitle, fontsize=9)
        ax.set_xlabel("Detection score", fontsize=8.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, y_max)
        ax.tick_params(labelsize=8, direction="in", top=True, right=True)
        ax.grid(alpha=0.2, lw=0.4, axis="y")
    axes[0].set_ylabel("Density", fontsize=8.5)
    axes[-1].legend(loc="upper right", fontsize=8,
                    handlelength=1.4, handletextpad=0.4)
    if title:
        fig.suptitle(title, y=1.02, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_pdf}")


# =========================================================
# MAIN
# =========================================================
def main():
    print("Replotting all paper figures as vector PDFs ...")

    # --- Fig.1 SARDet curves (F1-balanced adaptive T per bin) ---
    print("\n[Fig.1] curves SARDet ...")
    plot_curves(FIG_DIR / "fig_curves_sardet_data.json",
                FIG_DIR / "fig_curves_sardet.pdf",
                baseline_T=0.40,
                adaptive_Ts={"calm": 0.38, "moderate": 0.42, "rough": 0.36})

    print("\n[Fig.1] curves SAR-Ship ...")
    plot_curves(FIG_DIR / "fig_curves_sarship_data.json",
                FIG_DIR / "fig_curves_sarship.pdf",
                baseline_T=0.40,
                adaptive_Ts={"calm": 0.30, "moderate": 0.40, "rough": 0.30})

    # --- Fig.2 Adaptive T(e) ---
    print("\n[Fig.2] adaptive SARDet ...")
    plot_adaptive(anchors=[(2, 0.38), (13, 0.42), (44, 0.36)],
                  lin_a=-0.0011, lin_b=0.402,
                  e_range=(0, 53),
                  out_pdf=FIG_DIR / "fig_adaptive_sardet.pdf",
                  title="SARDet-100K (ship subset)")

    print("\n[Fig.2] adaptive SAR-Ship ...")
    plot_adaptive(anchors=[(1, 0.30), (10, 0.40), (41, 0.30)],
                  lin_a=-0.0009, lin_b=0.348,
                  e_range=(0, 49),
                  out_pdf=FIG_DIR / "fig_adaptive_sarship.pdf",
                  title="SAR-Ship-Dataset")

    # --- Fig.3 Recall-biased on SAR-Ship ---
    print("\n[Fig.3] curves SAR-Ship (recall-biased) ...")
    plot_curves(FIG_DIR / "fig_curves_sarship_recall_data.json",
                FIG_DIR / "fig_curves_sarship_recall.pdf",
                baseline_T=0.40,
                adaptive_Ts={"calm": 0.05, "moderate": 0.35, "rough": 0.60})

    # --- Fig.4 Score distributions ---
    # TP modes per bin: peak near 0.8 with similar sharpness across bins;
    # high-clutter has wider TP spread + non-trivial low-score TP shoulder.
    # FP profile: heavily concentrated near 0; high-clutter has slightly
    # heavier mid-range tail.
    print("\n[Fig.4] score distributions SARDet ...")
    plot_score_dists(
        src_jpg=FIG_DIR / "score_dists_sardet_orig.jpg",
        out_pdf=FIG_DIR / "score_dists_sardet.pdf",
        title="SARDet-100K (ship subset)")

    print("\n[Fig.4] score distributions SAR-Ship ...")
    plot_score_dists(
        src_jpg=FIG_DIR / "score_dists_sarship_orig.jpg",
        out_pdf=FIG_DIR / "score_dists_sarship.pdf",
        title="SAR-Ship-Dataset")

    print("\nAll figures written.")


if __name__ == "__main__":
    main()
