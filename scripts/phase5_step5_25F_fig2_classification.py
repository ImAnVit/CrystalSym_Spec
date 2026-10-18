"""
Figure 2: Group-aware classification performance.
Panel A: 16x16 confusion matrix (logistic regression, aggregate OOF).
Panel B: Horizontal per-class F1 bar chart sorted descending, coloured by tier.
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
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
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

cm_path = (
    RESULTS
    / "phase5_step5_24_classification_robustness"
    / "phase5_step5_24_confusion_matrix.csv"
)
pcm_path = (
    RESULTS
    / "phase5_step5_24_classification_robustness"
    / "phase5_step5_24_per_class_metrics.csv"
)

cm = pd.read_csv(cm_path, index_col=0)
pcm = pd.read_csv(pcm_path)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1.2, 1]})

# --- Panel A: Confusion matrix ---
ax = axes[0]
data = cm.values.astype(float)
labels = cm.columns.tolist()
n = len(labels)

im = ax.imshow(data, cmap="Blues", aspect="equal", interpolation="nearest", vmin=0)

ax.set_xticks(range(n))
ax.set_yticks(range(n))
short_labels = [l[:6] for l in labels]
ax.set_xticklabels(short_labels, rotation=90, fontsize=6)
ax.set_yticklabels(short_labels, fontsize=6)

for i in range(n):
    for j in range(n):
        val = int(data[i, j])
        if val == 0:
            continue
        color = "white" if data[i, j] > data.max() * 0.5 else "black"
        ax.text(j, i, str(val), ha="center", va="center", fontsize=5.5, color=color)

for i in range(n):
    rect = mpatches.Rectangle((i - 0.5, i - 0.5), 1, 1,
                               linewidth=1.2, edgecolor="#D55E00", facecolor="none")
    ax.add_patch(rect)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=7)
cbar.set_label("Count", fontsize=8)

ax.set_xlabel("Predicted", fontsize=8)
ax.set_ylabel("True", fontsize=8)
ax.set_title("(a) Confusion matrix (16 classes)", fontweight="bold", loc="left")

# --- Panel B: Per-class F1 bar chart ---
ax2 = axes[1]
pcm_sorted = pcm.sort_values("f1", ascending=True)

colors = [TIER_COLORS[t] for t in pcm_sorted["evaluation_tier"]]
bars = ax2.barh(range(len(pcm_sorted)), pcm_sorted["f1"], color=colors,
                edgecolor="white", linewidth=0.5, alpha=0.85)

ax2.set_yticks(range(len(pcm_sorted)))
ax2.set_yticklabels(pcm_sorted["mineral"], fontsize=7)
ax2.set_xlabel("F1 score", fontsize=8)
ax2.set_title("(b) Per-class F1 score", fontweight="bold", loc="left")
ax2.axvline(x=0.5, color="#999999", linestyle="--", linewidth=0.8, label="F1 = 0.5")

for i, (idx, row) in enumerate(pcm_sorted.iterrows()):
    ax2.text(row["f1"] + 0.01, i, f'{row["f1"]:.2f}', va="center", fontsize=6, color="#333333")

legend_handles = [
    mpatches.Patch(facecolor=TIER_COLORS["well_supported"], alpha=0.85,
                   label="Well-supported (n_groups >= 4)"),
    mpatches.Patch(facecolor=TIER_COLORS["limited"], alpha=0.85,
                   label="Limited (n_groups < 4)"),
]
ax2.legend(handles=legend_handles, loc="lower right", framealpha=0.9, fontsize=7)

ax2.annotate("4 classes F1 < 0.5:\nAndradite 0.0, Orthoclase 0.0\nMicrocline 0.4, Rutile 0.4",
             xy=(0.97, 0.97), xycoords="axes fraction",
             fontsize=6, color="#666666", ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFCC", edgecolor="#CCCCCC", alpha=0.9))

fig.suptitle("Figure 2  Group-aware classification performance",
             fontsize=11, fontweight="bold", y=1.01)
fig.tight_layout()

for ext in ("pdf", "svg", "png"):
    fig.savefig(OUTPUT / f"figure2_classification_performance.{ext}", dpi=300)
plt.close(fig)
print("Figure 2 saved.")
