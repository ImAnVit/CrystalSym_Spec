"""
Figure 3: Class support vs performance.
Scatter plot of n_groups (log) vs per-class F1, with mineral labels.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
})

TIER_COLORS = {
    "well_supported": "#0072B2",
    "limited": "#E69F00",
}

pcm_path = (
    RESULTS
    / "phase5_step5_24_classification_robustness"
    / "phase5_step5_24_per_class_metrics.csv"
)
cs_path = (
    RESULTS
    / "phase5_step5_24_classification_robustness"
    / "phase5_step5_24_class_support.csv"
)

pcm = pd.read_csv(pcm_path)
cs = pd.read_csv(cs_path)

merged = pcm.merge(cs[["mineral", "n_groups"]], on="mineral", how="left")

fig, ax = plt.subplots(figsize=(4.0, 3.6))

for _, row in merged.iterrows():
    color = TIER_COLORS[row["evaluation_tier"]]
    marker = "o" if row["evaluation_tier"] == "well_supported" else "D"
    ax.scatter(row["n_groups"], row["f1"], c=color, s=60, marker=marker,
               edgecolors="#333333", linewidths=0.5, zorder=3, alpha=0.85)

for _, row in merged.iterrows():
    offset_x = 8
    offset_y = 0
    ha = "left"
    if row["mineral"] in ("Enstatite", "Andradite"):
        offset_x = -8
        ha = "right"
    if row["mineral"] == "Olivine":
        offset_y = -5
    ax.annotate(row["mineral"],
                xy=(row["n_groups"], row["f1"]),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                fontsize=6.5, color="#333333", ha=ha, va="center")

ax.axhline(y=0.5, color="#999999", linestyle="--", linewidth=0.8, alpha=0.7)
ax.axvline(x=4, color="#AAAAAA", linestyle=":", linewidth=0.8, alpha=0.7)

ax.text(3.7, 0.96, "F1 = 0.5", fontsize=6.5, color="#999999", ha="right", va="top")
ax.text(4.15, 0.04, "tier\nboundary", fontsize=6, color="#AAAAAA", ha="left", va="bottom")

ax.set_xscale("log")
ax.set_xlabel("Number of provenance groups (log scale)", fontsize=9)
ax.set_ylabel("Per-class F1 score", fontsize=9)
ax.set_title("Figure 3  Class support vs performance", fontweight="bold", loc="left", fontsize=11)

ax.set_xlim(2, 60)
ax.set_ylim(-0.05, 1.05)

ax.annotate(
    "Enstatite (3 groups, F1 = 0.80):\nsmall support, high performance",
    xy=(3, 0.80), xytext=(12, 0.72),
    fontsize=6.5, color="#0072B2",
    arrowprops=dict(arrowstyle="->", color="#0072B2", lw=0.8),
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#0072B2", alpha=0.9))

ax.annotate(
    "Andradite (3 groups, F1 = 0.00):\nsmall support, zero performance",
    xy=(3, 0.00), xytext=(10, 0.15),
    fontsize=6.5, color="#D55E00",
    arrowprops=dict(arrowstyle="->", color="#D55E00", lw=0.8),
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#D55E00", alpha=0.9))

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=TIER_COLORS["well_supported"],
           markersize=7, markeredgecolor="#333333", markeredgewidth=0.5,
           label="Well-supported"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=TIER_COLORS["limited"],
           markersize=7, markeredgecolor="#333333", markeredgewidth=0.5,
           label="Limited"),
]
ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9, fontsize=7)

fig.tight_layout()

for ext in ("pdf", "svg", "png"):
    fig.savefig(OUTPUT / f"figure3_support_vs_performance.{ext}", dpi=300)
plt.close(fig)
print("Figure 3 saved.")
