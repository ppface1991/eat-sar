"""Fig. 3: per-stratum false-alarm rate and precision under a global vs a
clutter-conditioned threshold at the same target (from *_cfar.json).

Usage:  python plot_cfar.py ../data/sardet_cfar.json ../data/sarship_cfar.json fig_cfar.pdf
"""
import json
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.5, "pdf.fonttype": 42,
})
C_G040, C_GLOB, C_COND = "#BAB9B4", "#7A7974", "#20808D"
BINS = ["low", "medium", "high"]


def pick(d, kind, target=None, label_prefix=None):
    for t in d["targets"]:
        if t["kind"] != kind:
            continue
        if label_prefix and t["label"].startswith(label_prefix):
            return t
        if target is not None and abs(t["target"] - target) < 1e-9:
            return t
    raise KeyError((kind, target, label_prefix))


def panel(ax, t, key, title, scale=1.0, ylabel=None):
    x = np.arange(3); w = 0.36
    g = [t["GLOBAL"]["per_bin"][b][key] * scale for b in BINS]
    c = [t["COND"]["per_bin"][b][key] * scale for b in BINS]
    ax.bar(x - w / 2, g, w, color=C_GLOB, label=f"Global $T$={t['GLOBAL']['T']:.2f}")
    ax.bar(x + w / 2, c, w, color=C_COND, label="Clutter-conditioned $T_b$")
    tgt = t["target"] * scale
    ax.axhline(tgt, ls="--", lw=0.7, color="black")
    ax.set_xticks(x); ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_title(title, pad=2)
    if ylabel:
        ax.set_ylabel(ylabel)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return tgt


def main():
    sd, ss, out = sys.argv[1], sys.argv[2], sys.argv[3]
    D = {"SARDet-100K": json.load(open(sd)), "SAR-Ship-Dataset": json.load(open(ss))}
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 2.9), constrained_layout=True)
    for j, (name, d) in enumerate(D.items()):
        tf = pick(d, "fa", label_prefix="FA/img=1x")
        tp = pick(d, "prec", target=0.90)
        panel(axes[0, j], tf, "FAPI", f"({'ab'[j]}) {name}",
              ylabel="FA / image\n(budget of $T$=0.40)" if j == 0 else None)
        panel(axes[1, j], tp, "P", f"({'cd'[j]}) {name}",
              scale=100, ylabel="Precision (%)\n(target 90%)" if j == 0 else None)
        lo = min(min(tp["GLOBAL"]["per_bin"][b]["P"] for b in BINS),
                 min(tp["COND"]["per_bin"][b]["P"] for b in BINS)) * 100
        axes[1, j].set_ylim(np.floor(lo - 2), 100)
        axes[0, j].set_ylim(0, 0.27)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend([h[0], h[1], plt.Line2D([], [], ls="--", color="black", lw=0.7)],
               ["Global $T$ (fitted to target)", "Clutter-conditioned $T_b$", "Target"],
               loc="outside lower center", ncol=3, frameon=False, handlelength=1.4,
               columnspacing=0.8)
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=300)
    print("saved", out)


if __name__ == "__main__":
    main()
