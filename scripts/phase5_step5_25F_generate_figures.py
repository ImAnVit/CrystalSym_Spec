"""
Figure 1: Dataset composition and provenance structure.
Panel A: Grouped bar chart of n_spectra and n_groups per mineral.
Panel B: Provenance hierarchy schematic using Augite as example.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
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
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

CB_COLORS = ["#0072B2", "#D55E00"]

summary_path = (
    RESULTS
    / "phase5_step5_20_descriptive_descriptor_catalogue"
    / "phase5_step5_20_mineral_descriptor_summary.csv"
)
df = pd.read_csv(summary_path)
df = df.sort_values("n_source_groups", ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [2.2, 1]})

# --- Panel A: grouped bar chart ---
ax = axes[0]
minerals = df["mineral"].values
n_spectra = df["n_spectra"].values
n_groups = df["n_source_groups"].values

y_pos = np.arange(len(minerals))
bar_h = 0.35

bars_spec = ax.barh(y_pos - bar_h / 2, n_spectra, bar_h, label="n_spectra",
                     color=CB_COLORS[0], alpha=0.85, edgecolor="white", linewidth=0.5)
bars_grp = ax.barh(y_pos + bar_h / 2, n_groups, bar_h, label="n_provenance_groups",
                    color=CB_COLORS[1], alpha=0.85, edgecolor="white", linewidth=0.5)

for i, (ns, ng) in enumerate(zip(n_spectra, n_groups)):
    if ng == 1:
        ax.barh(y_pos[i] - bar_h / 2, ns, bar_h, color=CB_COLORS[0], alpha=0.4,
                edgecolor="#999999", linewidth=0.8, hatch="//")
        ax.barh(y_pos[i] + bar_h / 2, ng, bar_h, color=CB_COLORS[1], alpha=0.4,
                edgecolor="#999999", linewidth=0.8, hatch="//")

ax.set_yticks(y_pos)
ax.set_yticklabels(minerals)
ax.set_xlabel("Count")
ax.set_title("(a) Mineral coverage", fontweight="bold", loc="left")
ax.legend(loc="lower right", framealpha=0.9)
ax.invert_yaxis()

single = mpatches.Patch(facecolor=CB_COLORS[0], alpha=0.4, edgecolor="#999999",
                         hatch="//", label="n_groups = 1 (single)")
ax.add_artist(ax.legend(handles=[
    mpatches.Patch(facecolor=CB_COLORS[0], alpha=0.85, label="n_spectra"),
    mpatches.Patch(facecolor=CB_COLORS[1], alpha=0.85, label="n_provenance_groups"),
    single
], loc="lower right", framealpha=0.9))

ax.annotate("Provenance groups are\nprovisional units",
            xy=(0.98, 0.02), xycoords="axes fraction",
            fontsize=6.5, color="#666666", ha="right", va="bottom", fontstyle="italic")

# --- Panel B: provenance hierarchy schematic (Augite example) ---
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis("off")
ax2.set_title("(b) Provenance hierarchy (Augite)", fontweight="bold", loc="left")

box_kw = dict(boxstyle="round,pad=0.3", linewidth=0.8)

def draw_box(ax, x, y, w, h, label, sublabel, color, alpha=0.85):
    rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                    boxstyle="round,pad=0.15",
                                    facecolor=color, edgecolor="#333333",
                                    linewidth=0.8, alpha=alpha)
    ax.add_patch(rect)
    ax.text(x, y + 0.12, label, ha="center", va="center", fontsize=8, fontweight="bold")
    ax.text(x, y - 0.22, sublabel, ha="center", va="center", fontsize=6.5, color="#333333")

def draw_arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.8))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.15, my, label, fontsize=6, color="#666666", va="center")

draw_box(ax2, 5, 9.0, 3.5, 0.7, "Augite", "126 spectra", "#4DBBD5")
draw_arrow(ax2, 5, 8.65, 5, 8.0)
ax2.text(5.4, 8.3, "from", fontsize=5.5, color="#888888")

draw_box(ax2, 5, 7.6, 3.5, 0.7, "82 provenance groups", "unique sample_id + grain_size", "#E64B35")
draw_arrow(ax2, 5, 7.25, 5, 6.6)

draw_box(ax2, 5, 6.2, 3.5, 0.7, "Grain-size series", "e.g. <45 um, 45-125 um", "#00A087")
draw_arrow(ax2, 5, 5.85, 5, 5.2)

draw_box(ax2, 5, 4.8, 3.5, 0.7, "Measurement replicates", "spectra per group", "#3C5488")

draw_box(ax2, 2.5, 3.2, 2.2, 0.6, "Spectrum A", "reflectance", "#959099", alpha=0.6)
draw_box(ax2, 5.0, 3.2, 2.2, 0.6, "Spectrum B", "reflectance", "#959099", alpha=0.6)
draw_box(ax2, 7.5, 3.2, 2.2, 0.6, "Spectrum C", "reflectance", "#959099", alpha=0.6)

draw_arrow(ax2, 4.2, 4.45, 2.8, 3.55)
draw_arrow(ax2, 5.0, 4.45, 5.0, 3.55)
draw_arrow(ax2, 5.8, 4.45, 7.2, 3.55)

ax2.text(5, 2.3, "n = 126 spectra  |  82 groups  |  ~1.5 spectra/group",
         ha="center", fontsize=6.5, color="#555555", fontstyle="italic")
ax2.text(5, 1.8, "True specimen independence is unverified",
         ha="center", fontsize=6, color="#999999", fontstyle="italic")

fig.suptitle("Figure 1  Dataset composition and provenance structure",
             fontsize=11, fontweight="bold", y=1.01)
fig.tight_layout()

for ext in ("pdf", "svg", "png"):
    fig.savefig(OUTPUT / f"figure1_dataset_composition.{ext}", dpi=300)
plt.close(fig)
print("Figure 1 saved.")
