"""
Generate supplementary figures S1 and S2 for the manuscript.

Figure S1: Symbolic regression results (D3/D4 predicted vs actual)
Figure S2: Regional permutation importance bar chart

Outputs saved to figures/supplementary/ in PNG, PDF, and SVG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "figures" / "supplementary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
FORMATS = ("png", "pdf", "svg")


def save_fig(fig, name: str) -> None:
    for fmt in FORMATS:
        fig.savefig(OUT_DIR / f"{name}.{fmt}", dpi=DPI, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved {name} ({', '.join(FORMATS)})")


# ---------------------------------------------------------------------------
# Figure S1: Symbolic regression predicted vs actual
# ---------------------------------------------------------------------------

def figure_s1() -> None:
    d3_path = PROJECT_ROOT / "results" / "05_exploratory_analysis" / \
        "phase5_step5_23_multivariate_descriptor_pysr" / "phase5_step5_23_d3_predictions.csv"
    d4_path = PROJECT_ROOT / "results" / "05_exploratory_analysis" / \
        "phase5_step5_23_multivariate_descriptor_pysr" / "phase5_step5_23_d4_predictions.csv"

    d3 = pd.read_csv(d3_path)
    d4 = pd.read_csv(d4_path)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel (a): D3 predicted vs actual
    ax = axes[0]
    actual_d3 = d3["target_actual"].values
    pred_d3 = d3["pred_pysr"].values
    ax.scatter(actual_d3, pred_d3, s=8, alpha=0.5, c="#2166ac", edgecolors="none")
    lim = [min(actual_d3.min(), pred_d3.min()), max(actual_d3.max(), pred_d3.max())]
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.6)
    r2_d3 = 1 - np.sum((actual_d3 - pred_d3) ** 2) / np.sum((actual_d3 - np.mean(actual_d3)) ** 2)
    ax.set_title(f"(a) D3 (first_deriv_rms)\n$n={len(d3)}$, $R^2={r2_d3:.3f}$", fontsize=10)
    ax.set_xlabel("Actual D3", fontsize=9)
    ax.set_ylabel("Predicted D3", fontsize=9)
    ax.tick_params(labelsize=8)

    # Panel (b): D4 predicted vs actual
    ax = axes[1]
    actual_d4 = d4["target_actual"].values
    pred_d4 = d4["pred_pysr"].values
    ax.scatter(actual_d4, pred_d4, s=8, alpha=0.5, c="#b2182b", edgecolors="none")
    lim = [min(actual_d4.min(), pred_d4.min()), max(actual_d4.max(), pred_d4.max())]
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.6)
    r2_d4 = 1 - np.sum((actual_d4 - pred_d4) ** 2) / np.sum((actual_d4 - np.mean(actual_d4)) ** 2)
    ax.set_title(f"(b) D4 (second_deriv_rms)\n$n={len(d4)}$, $R^2={r2_d4:.3f}$", fontsize=10)
    ax.set_xlabel("Actual D4", fontsize=9)
    ax.set_ylabel("Predicted D4", fontsize=9)
    ax.tick_params(labelsize=8)

    # Panel (c): Fold-level R² stability
    ax = axes[2]
    folds = sorted(d3["fold"].unique())
    r2_d3_folds = []
    r2_d4_folds = []
    for f in folds:
        mask3 = d3["fold"] == f
        a3 = d3.loc[mask3, "target_actual"].values
        p3 = d3.loc[mask3, "pred_pysr"].values
        r2_d3_folds.append(1 - np.sum((a3 - p3) ** 2) / np.sum((a3 - np.mean(a3)) ** 2))

        mask4 = d4["fold"] == f
        a4 = d4.loc[mask4, "target_actual"].values
        p4 = d4.loc[mask4, "pred_pysr"].values
        r2_d4_folds.append(1 - np.sum((a4 - p4) ** 2) / np.sum((a4 - np.mean(a4)) ** 2))

    x = np.arange(len(folds))
    w = 0.35
    ax.bar(x - w / 2, r2_d3_folds, w, label="D3 (first_deriv_rms)", color="#2166ac", alpha=0.8)
    ax.bar(x + w / 2, r2_d4_folds, w, label="D4 (second_deriv_rms)", color="#b2182b", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds], fontsize=8)
    ax.set_ylabel("Fold $R^2$", fontsize=9)
    ax.set_title("(c) Cross-validation stability", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(labelsize=8)

    fig.suptitle("Supplementary Figure S1. Symbolic regression results.",
                 fontsize=11, y=1.02)
    fig.text(0.5, -0.02,
             "All descriptors computed from the same spectra; relationships reflect mathematical coupling.",
             ha="center", fontsize=8, style="italic")

    save_fig(fig, "figureS1_symbolic_regression")


# ---------------------------------------------------------------------------
# Figure S2: Regional permutation importance
# ---------------------------------------------------------------------------

def figure_s2() -> None:
    imp_path = PROJECT_ROOT / "results" / "05_exploratory_analysis" / \
        "phase5_step5_24_classification_robustness" / "phase5_step5_24_region_importance.csv"
    df = pd.read_csv(imp_path).sort_values("mean_importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#2166ac" if v >= 0 else "#b2182b" for v in df["mean_importance"]]
    y = np.arange(len(df))
    ax.barh(y, df["mean_importance"], color=colors, alpha=0.85, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(df["region"], fontsize=8)
    ax.set_xlabel("Mean permutation importance", fontsize=9)
    ax.axvline(0, color="k", lw=0.5)
    ax.tick_params(labelsize=8)

    for i, (val, std) in enumerate(zip(df["mean_importance"], df["std_importance"])):
        ax.errorbar(val, i, xerr=std, fmt="none", ecolor="grey",
                    elinewidth=0.8, capsize=2)

    ax.set_title("Supplementary Figure S2. Regional permutation importance.",
                 fontsize=11)
    fig.text(0.5, -0.02,
             "11 of 12 regions have negative mean importance (permuting improved performance). "
             "Error bars: std across 3 CV folds.",
             ha="center", fontsize=8, style="italic")

    save_fig(fig, "figureS2_region_importance")


if __name__ == "__main__":
    print("Generating supplementary figures...")
    figure_s1()
    figure_s2()
    print("Done.")
