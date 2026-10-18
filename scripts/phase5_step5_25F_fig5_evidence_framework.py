"""
Figure 5: Evidence framework and analysis gates.
Conceptual diagram showing 5 analysis gates with OPEN/CLOSED status
and the 3 conditions for testing the crystal symmetry hypothesis.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\andri\Projects\crystalsym-spec")
OUTPUT = PROJECT_ROOT / "figures" / "phase5_step5_25F_main_figures"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")

OPEN_COLOR = "#009E73"
CLOSED_COLOR = "#D55E00"
GATE_OPEN = "#E8F5E9"
GATE_CLOSED = "#FFF3E0"

gates = [
    ("GATE_DESC_RESTRICTIVE", "Descriptive analysis\nof available data", "OPEN"),
    ("GATE_MATCHED_STRATIFIED", "Matched stratified comparison\nrequiring specimen-specific chemistry", "CLOSED"),
    ("GATE_SPECIMEN_CHEMISTRY", "Analysis requiring\nspecimen-specific measured chemistry", "CLOSED"),
    ("GATE_INTERPRETABLE_ML", "Inferential ML requiring\nverified independence", "CLOSED"),
    ("GATE_PYSR", "Symbolic regression requiring\nindependent descriptors", "CLOSED"),
]

gate_w = 3.2
gate_h = 0.7
x_center = 2.5
y_start = 6.2
y_step = 1.05

ax.text(x_center, 6.85, "Analysis Gates", fontsize=11, fontweight="bold",
        ha="center", va="center", color="#333333")

for i, (name, criteria, status) in enumerate(gates):
    y = y_start - i * y_step
    color = OPEN_COLOR if status == "OPEN" else CLOSED_COLOR
    bg = GATE_OPEN if status == "OPEN" else GATE_CLOSED

    rect = FancyBboxPatch((x_center - gate_w / 2, y - gate_h / 2), gate_w, gate_h,
                           boxstyle="round,pad=0.1",
                           facecolor=bg, edgecolor=color, linewidth=1.2)
    ax.add_patch(rect)

    ax.text(x_center - gate_w / 2 + 0.15, y + 0.08, name.replace("GATE_", ""),
            fontsize=6.5, fontweight="bold", color="#333333", va="center")
    ax.text(x_center - gate_w / 2 + 0.15, y - 0.15, criteria,
            fontsize=5.5, color="#666666", va="center")

    status_x = x_center + gate_w / 2 - 0.15
    badge_w = 0.65
    badge_h = 0.25
    badge = FancyBboxPatch((status_x - badge_w, y - badge_h / 2), badge_w, badge_h,
                            boxstyle="round,pad=0.05",
                            facecolor=color, edgecolor="none", alpha=0.9)
    ax.add_patch(badge)
    ax.text(status_x - badge_w / 2, y, status, fontsize=6, fontweight="bold",
            color="white", ha="center", va="center")

    if i < len(gates) - 1:
        ax.annotate("", xy=(x_center, y - gate_h / 2 - 0.05),
                    xytext=(x_center, y - gate_h / 2 - y_step + gate_h + 0.05),
                    arrowprops=dict(arrowstyle="-|>", color="#BBBBBB", lw=0.8))

cond_x = 7.2
ax.text(cond_x, 6.85, "Crystal Symmetry Hypothesis", fontsize=10, fontweight="bold",
        ha="center", va="center", color="#333333")
ax.text(cond_x, 6.5, "Three necessary conditions:", fontsize=8, ha="center",
        color="#555555", fontstyle="italic")

conditions = [
    ("C1", "No complete confounding\nbetween mineral and crystal system", "NOT MET"),
    ("C2", "Specimen-specific chemistry\navailable for analysis", "NOT MET"),
    ("C3", "Verified specimen independence\n(provenance groups validated)", "NOT MET"),
]

cond_y_start = 5.8
cond_h = 0.85
cond_w = 3.0

for i, (cid, desc, status) in enumerate(conditions):
    y = cond_y_start - i * (cond_h + 0.2)

    rect = FancyBboxPatch((cond_x - cond_w / 2, y - cond_h / 2), cond_w, cond_h,
                           boxstyle="round,pad=0.1",
                           facecolor="#FAFAFA", edgecolor="#CC0000",
                           linewidth=1.0, linestyle="--")
    ax.add_patch(rect)

    ax.text(cond_x - cond_w / 2 + 0.12, y + 0.15, cid,
            fontsize=7, fontweight="bold", color="#CC0000", va="center")
    ax.text(cond_x - cond_w / 2 + 0.12, y - 0.12, desc,
            fontsize=6, color="#555555", va="center")

    badge_w2 = 0.7
    badge_h2 = 0.22
    bx = cond_x + cond_w / 2 - 0.12
    badge = FancyBboxPatch((bx - badge_w2, y + 0.18 - badge_h2 / 2), badge_w2, badge_h2,
                            boxstyle="round,pad=0.05",
                            facecolor="#CC0000", edgecolor="none", alpha=0.85)
    ax.add_patch(badge)
    ax.text(bx - badge_w2 / 2, y + 0.18, status, fontsize=5.5, fontweight="bold",
            color="white", ha="center", va="center")

verdict_y = cond_y_start - 3 * (cond_h + 0.2) + 0.3
verdict_rect = FancyBboxPatch((cond_x - cond_w / 2, verdict_y - 0.3), cond_w, 0.6,
                               boxstyle="round,pad=0.1",
                               facecolor="#FFEBEE", edgecolor="#CC0000", linewidth=1.5)
ax.add_patch(verdict_rect)
ax.text(cond_x, verdict_y + 0.05, "Hypothesis UNTESTABLE", fontsize=8,
        fontweight="bold", color="#CC0000", ha="center", va="center")
ax.text(cond_x, verdict_y - 0.15, "with current data", fontsize=6.5,
        color="#888888", ha="center", va="center", fontstyle="italic")

ax.annotate("", xy=(cond_x - cond_w / 2 - 0.1, verdict_y),
            xytext=(x_center + gate_w / 2 + 0.3, y_start - 4 * y_step + 0.2),
            arrowprops=dict(arrowstyle="->", color="#CC0000", lw=1.0,
                           connectionstyle="arc3,rad=-0.2", linestyle="--"))

ax.text(5.0, 0.3, "Gates are project-specific assessments, not universal standards",
        fontsize=6.5, color="#999999", ha="center", fontstyle="italic")

fig.suptitle("Figure 5  Evidence framework and analysis gates",
             fontsize=11, fontweight="bold", y=0.98)
fig.tight_layout()

for ext in ("pdf", "svg", "png"):
    fig.savefig(OUTPUT / f"figure5_evidence_framework.{ext}", dpi=300)
plt.close(fig)
print("Figure 5 saved.")
