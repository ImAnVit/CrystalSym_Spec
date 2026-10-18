"""
Figure 4: Symbolic regression results.
Panel A: Predicted vs actual D3 (PySR, pooled across folds).
Panel B: Predicted vs actual D4 (PySR, pooled across folds).
Panel C: Coefficient stability across folds for stable equations.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\andri\Projects\crystalsym-spec")
RESULTS = PROJECT_ROOT / "results" / "05_exploratory_analysis"
OUTPUT = PROJECT_ROOT / "figures" / "phase5_step5_25F_main_figures"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
})

d3_pred_path = RESULTS / "phase5_step5_23_multivariate_descriptor_pysr" / "phase5_step5_23_d3_predictions.csv"
d4_pred_path = RESULTS / "phase5_step5_23_multivariate_descriptor_pysr" / "phase5_step5_23_d4_predictions.csv"
d3_agg_path = RESULTS / "phase5_step5_23_multivariate_descriptor_pysr" / "phase5_step5_23_d3_aggregate_metrics.csv"
d4_agg_path = RESULTS / "phase5_step5_23_multivariate_descriptor_pysr" / "phase5_step5_23_d4_aggregate_metrics.csv"

d3_pred = pd.read_csv(d3_pred_path)
d4_pred = pd.read_csv(d4_pred_path)
d3_agg = pd.read_csv(d3_agg_path)
d4_agg = pd.read_csv(d4_agg_path)

d3_r2 = d3_agg.loc[d3_agg["model"] == "pysr", "r2"].values[0]
d4_r2 = d4_agg.loc[d4_agg["model"] == "pysr", "r2"].values[0]

d3_coefs = {0: 0.0015414291, 1: 0.0015809644, 2: 0.0015459491}
d4_coefs = {0: 0.00014801, 1: 0.0001498786, 2: 0.00014716524}

fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1, 1, 1.1]})

# --- Panel A: D3 predicted vs actual ---
ax = axes[0]
actual_d3 = d3_pred["target_actual"].values
pred_d3 = d3_pred["pred_pysr"].values

ax.scatter(actual_d3, pred_d3, s=8, alpha=0.5, c="#0072B2", edgecolors="none")
lims = [min(actual_d3.min(), pred_d3.min()), max(actual_d3.max(), pred_d3.max())]
ax.plot(lims, lims, "--", color="#D55E00", linewidth=0.8, label="1:1 line")

ax.set_xlabel("Actual D3", fontsize=8)
ax.set_ylabel("Predicted D3", fontsize=8)
ax.set_title("(a) D3: PySR predictions", fontweight="bold", loc="left")
ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax.text(0.05, 0.95, f"$R^2$ = {d3_r2:.3f}", transform=ax.transAxes,
        fontsize=8, va="top", bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
        edgecolor="#CCCCCC", alpha=0.9))
ax.annotate("D3 ~ 0.00155 $\\sqrt{D_1 \\cdot D_2}$",
            xy=(0.05, 0.82), xycoords="axes fraction",
            fontsize=6.5, color="#555555", va="top")

# --- Panel B: D4 predicted vs actual ---
ax2 = axes[1]
actual_d4 = d4_pred["target_actual"].values
pred_d4 = d4_pred["pred_pysr"].values

ax2.scatter(actual_d4, pred_d4, s=8, alpha=0.5, c="#0072B2", edgecolors="none")
lims2 = [min(actual_d4.min(), pred_d4.min()), max(actual_d4.max(), pred_d4.max())]
ax2.plot(lims2, lims2, "--", color="#D55E00", linewidth=0.8, label="1:1 line")

ax2.set_xlabel("Actual D4", fontsize=8)
ax2.set_ylabel("Predicted D4", fontsize=8)
ax2.set_title("(b) D4: PySR predictions", fontweight="bold", loc="left")
ax2.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax2.text(0.05, 0.95, f"$R^2$ = {d4_r2:.3f}", transform=ax2.transAxes,
         fontsize=8, va="top", bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
         edgecolor="#CCCCCC", alpha=0.9))
ax2.annotate("D4 ~ 0.000148 $D_1 \\cdot D_2$",
             xy=(0.05, 0.82), xycoords="axes fraction",
             fontsize=6.5, color="#555555", va="top")

# --- Panel C: Coefficient stability ---
ax3 = axes[2]
folds = [0, 1, 2]
d3_vals = [d3_coefs[f] for f in folds]
d4_vals = [d4_coefs[f] for f in folds]

d3_mean = np.mean(d3_vals)
d4_mean = np.mean(d4_vals)
d3_cv = np.std(d3_vals, ddof=0) / d3_mean * 100
d4_cv = np.std(d4_vals, ddof=0) / d4_mean * 100

x_pos = np.array([0, 1])
width = 0.25

for i, f in enumerate(folds):
    offset = (i - 1) * width
    ax3.bar(x_pos + offset, [d3_vals[i], d4_vals[i] * 10], width,
            label=f"Fold {f}", alpha=0.7, edgecolor="white", linewidth=0.5,
            color=["#0072B2", "#E69F00", "#009E73"][i])

ax3.bar(x_pos[0], d3_mean, width * 0.8, color="none",
        edgecolor="#D55E00", linewidth=1.5, linestyle="--", zorder=5)
ax3.bar(x_pos[1], d4_mean * 10, width * 0.8, color="none",
        edgecolor="#D55E00", linewidth=1.5, linestyle="--", zorder=5)

ax3.set_xticks(x_pos)
ax3.set_xticklabels(["D3 coefficient\n($\\times 10^{-3}$)",
                      "D4 coefficient\n($\\times 10^{-4}$)"], fontsize=7)
ax3.set_ylabel("Coefficient value", fontsize=8)
ax3.set_title("(c) Fold stability", fontweight="bold", loc="left")

ax3.text(0, d3_mean + 0.00003, f"mean={d3_mean:.5f}\nCV={d3_cv:.1f}%",
         ha="center", fontsize=6, color="#D55E00")
ax3.text(1, d4_mean * 10 + 0.0003, f"mean={d4_mean:.6f}\nCV={d4_cv:.1f}%",
         ha="center", fontsize=6, color="#D55E00")

ax3.legend(fontsize=6, loc="upper left", framealpha=0.9)

ax3.annotate("All descriptors from\nthe same spectra",
             xy=(0.98, 0.02), xycoords="axes fraction",
             fontsize=6, color="#999999", ha="right", va="bottom", fontstyle="italic")

fig.suptitle("Figure 4  Symbolic regression results",
             fontsize=11, fontweight="bold", y=1.02)
fig.tight_layout()

for ext in ("pdf", "svg", "png"):
    fig.savefig(OUTPUT / f"figure4_symbolic_regression.{ext}", dpi=300)
plt.close(fig)
print("Figure 4 saved.")
