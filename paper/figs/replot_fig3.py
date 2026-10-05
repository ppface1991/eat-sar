"""Regenerate Fig. 3 panels (F1 / gated AP) from revision_out/curves_*.json.

Usage: python replot_fig3.py
Expects ../data/curves_sardet.json and ../data/curves_sarship-2.json
(copies of revision_out/curves_sardet.json / curves_sarship.json) and writes
fig_curves_{sardet,sarship}_{f1,map}.pdf with the F1-balanced per-bin T_b
(fitted on the calibration split) drawn as dotted lines.
"""
from replot_split import plot_curve_panel, FIG_DIR

JOBS = [
    ("fig_curves_sardet_data.json", "SARDet-100K",
     {"calm": 0.40, "moderate": 0.425, "rough": 0.45}, "fig_curves_sardet"),
    ("fig_curves_sarship_data.json", "SAR-Ship-Dataset",
     {"calm": 0.425, "moderate": 0.475, "rough": 0.55}, "fig_curves_sarship"),
]

if __name__ == "__main__":
    for j, name, Ts, base in JOBS:
        plot_curve_panel(FIG_DIR / j, side="left", metric_title=f"{name}: F1",
                         ylabel="F1", out_pdf=FIG_DIR / f"{base}_f1.pdf",
                         baseline_T=0.40, adaptive_Ts=Ts)
        plot_curve_panel(FIG_DIR / j, side="right",
                         metric_title=f"{name}: gated AP@0.5",
                         ylabel="Gated AP@0.5",
                         out_pdf=FIG_DIR / f"{base}_map.pdf",
                         baseline_T=0.40, adaptive_Ts=Ts)
