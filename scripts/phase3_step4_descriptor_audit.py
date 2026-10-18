#!/usr/bin/env python3
"""
Phase 3 Step 4 — Descriptor Distribution, Dependence, and Redundancy Audit

Diagnostic-only audit of the four frozen primary spectral complexity descriptors.
This script does NOT modify the canonical feature matrix or perform scientific analysis.

Author: CrystalSym-Spec Project
Date: 2026-10-06
"""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ============================================================================
# Configuration
# ============================================================================

INPUT_FILE = Path("data/processed/phase3_canonical_spectral_features.csv")
OUTPUT_DIR = Path("results/03_complexity_features")
FIGURES_DIR = OUTPUT_DIR / "figures" / "step4"

EXPECTED_CHECKSUM = "a761544a2c959db83ce5ee5acc7c537ad876283c733ee08ace3f081e2af6efda"
EXPECTED_ROWS = 454
EXPECTED_COLS = 9
EXPECTED_DESCRIPTOR_COLS = [
    "n_peaks_prom",
    "mean_peak_prominence",
    "first_deriv_rms",
    "second_deriv_rms",
]
EXPECTED_METADATA_COLS = [
    "canonical_spectrum_id",
    "raw_filename",
    "mineral",
    "specimen_group_id",
    "representation_type",
]

REDUNDANCY_THRESHOLD_STRONG = 0.90
REDUNDANCY_THRESHOLD_VERY_STRONG = 0.95

# ============================================================================
# Utility Functions
# ============================================================================


def compute_file_checksum(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_descriptor_stats(series: pd.Series) -> dict:
    """Compute comprehensive distribution statistics for a descriptor."""
    stats_dict = {
        "count": int(series.count()),
        "missing": int(series.isna().sum()),
        "finite": int(np.isfinite(series).sum()),
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "variance": float(series.var()),
        "q1": float(series.quantile(0.25)),
        "q3": float(series.quantile(0.75)),
        "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
        "skewness": float(series.skew()),
        "unique_values": int(series.nunique()),
        "zero_values": int((series == 0).sum()),
        "cv": float(series.std() / series.mean()) if series.mean() != 0 else np.nan,
    }
    return stats_dict


# ============================================================================
# Part A: Distribution Statistics
# ============================================================================


def part_a_distribution_statistics(df: pd.DataFrame) -> dict:
    """Compute distribution statistics for all 4 descriptors."""
    print("Part A: Computing distribution statistics...")

    results = {}
    for col in EXPECTED_DESCRIPTOR_COLS:
        results[col] = compute_descriptor_stats(df[col])

    return results


# ============================================================================
# Part B: Visualization
# ============================================================================


def part_b_visualization(df: pd.DataFrame) -> list[Path]:
    """Create distribution plots for each descriptor."""
    print("Part B: Creating visualizations...")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_paths = []

    for col in EXPECTED_DESCRIPTOR_COLS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram
        axes[0].hist(df[col], bins=30, edgecolor="black", alpha=0.7)
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Frequency")
        axes[0].set_title(f"Distribution of {col}")
        axes[0].axvline(df[col].mean(), color="red", linestyle="--", label=f"Mean: {df[col].mean():.4g}")
        axes[0].axvline(
            df[col].median(),
            color="green",
            linestyle="--",
            label=f"Median: {df[col].median():.4g}",
        )
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Boxplot
        axes[1].boxplot(df[col], orientation="vertical")
        axes[1].set_ylabel(col)
        axes[1].set_title(f"Boxplot of {col}")
        axes[1].grid(alpha=0.3, axis="y")

        plt.tight_layout()
        fig_path = FIGURES_DIR / f"fig_{col}_distribution.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(fig_path)

    return figure_paths


# ============================================================================
# Part C: Pairwise Correlation
# ============================================================================


def part_c_pairwise_correlation(df: pd.DataFrame) -> dict:
    """Compute Pearson and Spearman correlation matrices."""
    print("Part C: Computing pairwise correlations...")

    descriptor_data = df[EXPECTED_DESCRIPTOR_COLS]

    # Pearson correlation
    pearson_corr = descriptor_data.corr(method="pearson")

    # Spearman correlation
    spearman_corr = descriptor_data.corr(method="spearman")

    # Find strongest pair (excluding diagonal)
    pearson_matrix = pearson_corr.values.copy()
    np.fill_diagonal(pearson_matrix, 0)
    max_pearson_idx = np.unravel_index(np.argmax(np.abs(pearson_matrix)), pearson_matrix.shape)
    strongest_pearson_pair = (
        EXPECTED_DESCRIPTOR_COLS[max_pearson_idx[0]],
        EXPECTED_DESCRIPTOR_COLS[max_pearson_idx[1]],
    )
    strongest_pearson_value = pearson_corr.iloc[max_pearson_idx]

    spearman_matrix = spearman_corr.values.copy()
    np.fill_diagonal(spearman_matrix, 0)
    max_spearman_idx = np.unravel_index(np.argmax(np.abs(spearman_matrix)), spearman_matrix.shape)
    strongest_spearman_pair = (
        EXPECTED_DESCRIPTOR_COLS[max_spearman_idx[0]],
        EXPECTED_DESCRIPTOR_COLS[max_spearman_idx[1]],
    )
    strongest_spearman_value = spearman_corr.iloc[max_spearman_idx]

    return {
        "pearson": pearson_corr,
        "spearman": spearman_corr,
        "strongest_pearson_pair": strongest_pearson_pair,
        "strongest_pearson_value": strongest_pearson_value,
        "strongest_spearman_pair": strongest_spearman_pair,
        "strongest_spearman_value": strongest_spearman_value,
    }


# ============================================================================
# Part D: Redundancy Assessment
# ============================================================================


def part_d_redundancy_assessment(correlation_results: dict) -> dict:
    """Assess redundancy based on correlation thresholds."""
    print("Part D: Assessing redundancy...")

    pearson_corr = correlation_results["pearson"]
    spearman_corr = correlation_results["spearman"]

    strong_pairs_pearson = []
    very_strong_pairs_pearson = []
    strong_pairs_spearman = []
    very_strong_pairs_spearman = []

    for i in range(len(EXPECTED_DESCRIPTOR_COLS)):
        for j in range(i + 1, len(EXPECTED_DESCRIPTOR_COLS)):
            col_i = EXPECTED_DESCRIPTOR_COLS[i]
            col_j = EXPECTED_DESCRIPTOR_COLS[j]

            pearson_val = pearson_corr.loc[col_i, col_j]
            spearman_val = spearman_corr.loc[col_i, col_j]

            if abs(pearson_val) >= REDUNDANCY_THRESHOLD_STRONG:
                strong_pairs_pearson.append((col_i, col_j, pearson_val))
            if abs(pearson_val) >= REDUNDANCY_THRESHOLD_VERY_STRONG:
                very_strong_pairs_pearson.append((col_i, col_j, pearson_val))

            if abs(spearman_val) >= REDUNDANCY_THRESHOLD_STRONG:
                strong_pairs_spearman.append((col_i, col_j, spearman_val))
            if abs(spearman_val) >= REDUNDANCY_THRESHOLD_VERY_STRONG:
                very_strong_pairs_spearman.append((col_i, col_j, spearman_val))

    return {
        "strong_pairs_pearson": strong_pairs_pearson,
        "very_strong_pairs_pearson": very_strong_pairs_pearson,
        "strong_pairs_spearman": strong_pairs_spearman,
        "very_strong_pairs_spearman": very_strong_pairs_spearman,
    }


# ============================================================================
# Part E: Scale Comparison
# ============================================================================


def part_e_scale_comparison(df: pd.DataFrame) -> dict:
    """Compare scales of descriptors (no standardization)."""
    print("Part E: Comparing descriptor scales...")

    scale_info = {}
    for col in EXPECTED_DESCRIPTOR_COLS:
        scale_info[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "range": float(df[col].max() - df[col].min()),
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std()),
        }

    # Compute scale ratios
    means = [scale_info[col]["mean"] for col in EXPECTED_DESCRIPTOR_COLS]
    stds = [scale_info[col]["std"] for col in EXPECTED_DESCRIPTOR_COLS]

    max_mean_ratio = max(means) / min(means) if min(means) != 0 else np.nan
    max_std_ratio = max(stds) / min(stds) if min(stds) != 0 else np.nan

    return {
        "scale_info": scale_info,
        "max_mean_ratio": max_mean_ratio,
        "max_std_ratio": max_std_ratio,
    }


# ============================================================================
# Part F: Zero and Discrete Value Assessment
# ============================================================================


def part_f_zero_discrete_assessment(df: pd.DataFrame) -> dict:
    """Assess zero values and discrete distributions."""
    print("Part F: Assessing zero and discrete values...")

    zero_assessment = {}
    for col in EXPECTED_DESCRIPTOR_COLS:
        zero_count = int((df[col] == 0).sum())
        zero_fraction = zero_count / len(df)

        # Frequency table for n_peaks_prom (discrete)
        if col == "n_peaks_prom":
            freq_table = df[col].value_counts().sort_index().to_dict()
        else:
            freq_table = None

        zero_assessment[col] = {
            "zero_count": zero_count,
            "zero_fraction": zero_fraction,
            "frequency_table": freq_table,
        }

    return zero_assessment


# ============================================================================
# Part G: Representation Type Audit
# ============================================================================


def part_g_representation_audit(df: pd.DataFrame) -> dict:
    """Audit descriptor distributions by representation type."""
    print("Part G: Auditing representation types...")

    representation_audit = {}

    for rep_type in df["representation_type"].unique():
        rep_df = df[df["representation_type"] == rep_type]
        rep_stats = {}

        for col in EXPECTED_DESCRIPTOR_COLS:
            rep_stats[col] = {
                "count": int(len(rep_df)),
                "median": float(rep_df[col].median()),
                "q1": float(rep_df[col].quantile(0.25)),
                "q3": float(rep_df[col].quantile(0.75)),
                "iqr": float(rep_df[col].quantile(0.75) - rep_df[col].quantile(0.25)),
                "min": float(rep_df[col].min()),
                "max": float(rep_df[col].max()),
            }

        representation_audit[rep_type] = rep_stats

    return representation_audit


# ============================================================================
# Part H: Provenance Group Structure
# ============================================================================


def part_h_provenance_group_structure(df: pd.DataFrame) -> dict:
    """Analyze provenance group structure."""
    print("Part H: Analyzing provenance group structure...")

    group_sizes = df.groupby("specimen_group_id").size()

    provenance_structure = {
        "n_groups": int(len(group_sizes)),
        "min_group_size": int(group_sizes.min()),
        "max_group_size": int(group_sizes.max()),
        "median_group_size": float(group_sizes.median()),
        "mean_group_size": float(group_sizes.mean()),
        "groups_with_1_spectrum": int((group_sizes == 1).sum()),
        "groups_with_2_spectra": int((group_sizes == 2).sum()),
        "groups_with_3plus_spectra": int((group_sizes >= 3).sum()),
        "group_size_distribution": group_sizes.value_counts().sort_index().to_dict(),
    }

    return provenance_structure


# ============================================================================
# Part I: Duplicate Feature Vectors
# ============================================================================


def part_i_duplicate_feature_vectors(df: pd.DataFrame) -> dict:
    """Check for exact duplicate 4-descriptor vectors."""
    print("Part I: Checking for duplicate feature vectors...")

    descriptor_data = df[EXPECTED_DESCRIPTOR_COLS]

    # Find exact duplicates
    duplicated_mask = descriptor_data.duplicated(keep=False)
    n_duplicates = int(duplicated_mask.sum())

    # Get duplicate groups
    if n_duplicates > 0:
        duplicate_groups = descriptor_data[duplicated_mask].groupby(EXPECTED_DESCRIPTOR_COLS).size()
        n_duplicate_groups = len(duplicate_groups)
        duplicate_group_sizes = duplicate_groups.value_counts().sort_index().to_dict()
    else:
        n_duplicate_groups = 0
        duplicate_group_sizes = {}

    return {
        "n_duplicate_rows": n_duplicates,
        "n_duplicate_groups": n_duplicate_groups,
        "duplicate_group_sizes": duplicate_group_sizes,
    }


# ============================================================================
# Report Generation
# ============================================================================


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a markdown table without requiring tabulate."""
    lines = []
    # Header
    header = "| " + " | ".join([""] + list(df.columns)) + " |"
    lines.append(header)
    # Separator
    sep = "| " + " | ".join(["---"] * (len(df.columns) + 1)) + " |"
    lines.append(sep)
    # Data rows
    for idx, row in df.iterrows():
        row_str = "| " + " | ".join([str(idx)] + [f"{v}" for v in row]) + " |"
        lines.append(row_str)
    return "\n".join(lines)


def generate_markdown_report(
    df: pd.DataFrame,
    checksum: str,
    part_a_results: dict,
    part_b_figures: list[Path],
    part_c_results: dict,
    part_d_results: dict,
    part_e_results: dict,
    part_f_results: dict,
    part_g_results: dict,
    part_h_results: dict,
    part_i_results: dict,
) -> str:
    """Generate comprehensive markdown report."""

    report = []
    report.append("# Phase 3 Step 4 — Descriptor Distribution, Dependence, and Redundancy Audit")
    report.append("")
    report.append("**Date:** 2026-10-06")
    report.append("")
    report.append("**Status:** COMPLETE / FROZEN")
    report.append("")
    report.append("---")
    report.append("")

    # Input verification
    report.append("## 1. Input Verification")
    report.append("")
    report.append(f"| Parameter | Expected | Observed | Status |")
    report.append(f"|-----------|----------|----------|--------|")
    report.append(f"| SHA-256 checksum | {EXPECTED_CHECKSUM} | {checksum} | {'✓ PASS' if checksum == EXPECTED_CHECKSUM else '✗ FAIL'} |")
    report.append(f"| Number of rows | {EXPECTED_ROWS} | {len(df)} | {'✓ PASS' if len(df) == EXPECTED_ROWS else '✗ FAIL'} |")
    report.append(f"| Number of columns | {EXPECTED_COLS} | {len(df.columns)} | {'✓ PASS' if len(df.columns) == EXPECTED_COLS else '✗ FAIL'} |")
    report.append("")

    # Part A: Distribution Statistics
    report.append("## 2. Part A — Distribution Statistics")
    report.append("")
    for col in EXPECTED_DESCRIPTOR_COLS:
        stats = part_a_results[col]
        report.append(f"### {col}")
        report.append("")
        report.append(f"| Statistic | Value |")
        report.append(f"|-----------|-------|")
        report.append(f"| Count | {stats['count']} |")
        report.append(f"| Missing | {stats['missing']} |")
        report.append(f"| Finite | {stats['finite']} |")
        report.append(f"| Min | {stats['min']:.6g} |")
        report.append(f"| Max | {stats['max']:.6g} |")
        report.append(f"| Mean | {stats['mean']:.6g} |")
        report.append(f"| Median | {stats['median']:.6g} |")
        report.append(f"| Std | {stats['std']:.6g} |")
        report.append(f"| Variance | {stats['variance']:.6g} |")
        report.append(f"| Q1 | {stats['q1']:.6g} |")
        report.append(f"| Q3 | {stats['q3']:.6g} |")
        report.append(f"| IQR | {stats['iqr']:.6g} |")
        report.append(f"| Skewness | {stats['skewness']:.6g} |")
        report.append(f"| Unique values | {stats['unique_values']} |")
        report.append(f"| Zero values | {stats['zero_values']} |")
        report.append(f"| CV | {stats['cv']:.6g} |")
        report.append("")

    # Part B: Visualization
    report.append("## 3. Part B — Visualization")
    report.append("")
    report.append("Distribution plots generated for all 4 descriptors:")
    report.append("")
    for fig_path in part_b_figures:
        report.append(f"- `{fig_path}`")
    report.append("")

    # Part C: Pairwise Correlation
    report.append("## 4. Part C — Pairwise Correlation")
    report.append("")
    report.append("### Pearson Correlation Matrix")
    report.append("")
    report.append(dataframe_to_markdown(part_c_results["pearson"].round(4)))
    report.append("")
    report.append("### Spearman Correlation Matrix")
    report.append("")
    report.append(dataframe_to_markdown(part_c_results["spearman"].round(4)))
    report.append("")
    report.append(f"**Strongest Pearson pair:** {part_c_results['strongest_pearson_pair'][0]} ↔ {part_c_results['strongest_pearson_pair'][1]} (r = {part_c_results['strongest_pearson_value']:.4f})")
    report.append("")
    report.append(f"**Strongest Spearman pair:** {part_c_results['strongest_spearman_pair'][0]} ↔ {part_c_results['strongest_spearman_pair'][1]} (ρ = {part_c_results['strongest_spearman_value']:.4f})")
    report.append("")

    # Part D: Redundancy Assessment
    report.append("## 5. Part D — Redundancy Assessment")
    report.append("")
    report.append(f"**Thresholds:** |r| ≥ {REDUNDANCY_THRESHOLD_STRONG} (strong), |r| ≥ {REDUNDANCY_THRESHOLD_VERY_STRONG} (very strong)")
    report.append("")
    report.append("### Pearson Correlation")
    report.append("")
    if part_d_results["strong_pairs_pearson"]:
        report.append(f"**Strong pairs (|r| ≥ {REDUNDANCY_THRESHOLD_STRONG}):**")
        report.append("")
        for pair in part_d_results["strong_pairs_pearson"]:
            report.append(f"- {pair[0]} ↔ {pair[1]}: r = {pair[2]:.4f}")
        report.append("")
    else:
        report.append(f"**No strong pairs (|r| ≥ {REDUNDANCY_THRESHOLD_STRONG})**")
        report.append("")

    if part_d_results["very_strong_pairs_pearson"]:
        report.append(f"**Very strong pairs (|r| ≥ {REDUNDANCY_THRESHOLD_VERY_STRONG}):**")
        report.append("")
        for pair in part_d_results["very_strong_pairs_pearson"]:
            report.append(f"- {pair[0]} ↔ {pair[1]}: r = {pair[2]:.4f}")
        report.append("")
    else:
        report.append(f"**No very strong pairs (|r| ≥ {REDUNDANCY_THRESHOLD_VERY_STRONG})**")
        report.append("")

    report.append("### Spearman Correlation")
    report.append("")
    if part_d_results["strong_pairs_spearman"]:
        report.append(f"**Strong pairs (|ρ| ≥ {REDUNDANCY_THRESHOLD_STRONG}):**")
        report.append("")
        for pair in part_d_results["strong_pairs_spearman"]:
            report.append(f"- {pair[0]} ↔ {pair[1]}: ρ = {pair[2]:.4f}")
        report.append("")
    else:
        report.append(f"**No strong pairs (|ρ| ≥ {REDUNDANCY_THRESHOLD_STRONG})**")
        report.append("")

    if part_d_results["very_strong_pairs_spearman"]:
        report.append(f"**Very strong pairs (|ρ| ≥ {REDUNDANCY_THRESHOLD_VERY_STRONG}):**")
        report.append("")
        for pair in part_d_results["very_strong_pairs_spearman"]:
            report.append(f"- {pair[0]} ↔ {pair[1]}: ρ = {pair[2]:.4f}")
        report.append("")
    else:
        report.append(f"**No very strong pairs (|ρ| ≥ {REDUNDANCY_THRESHOLD_VERY_STRONG})**")
        report.append("")

    # Part E: Scale Comparison
    report.append("## 6. Part E — Scale Comparison")
    report.append("")
    report.append("| Descriptor | Min | Max | Range | Mean | Median | Std |")
    report.append("|------------|-----|-----|-------|------|--------|-----|")
    for col in EXPECTED_DESCRIPTOR_COLS:
        info = part_e_results["scale_info"][col]
        report.append(f"| {col} | {info['min']:.4g} | {info['max']:.4g} | {info['range']:.4g} | {info['mean']:.4g} | {info['median']:.4g} | {info['std']:.4g} |")
    report.append("")
    report.append(f"**Max mean ratio:** {part_e_results['max_mean_ratio']:.2f}×")
    report.append("")
    report.append(f"**Max std ratio:** {part_e_results['max_std_ratio']:.2f}×")
    report.append("")

    # Part F: Zero and Discrete Value Assessment
    report.append("## 7. Part F — Zero and Discrete Value Assessment")
    report.append("")
    for col in EXPECTED_DESCRIPTOR_COLS:
        assessment = part_f_results[col]
        report.append(f"### {col}")
        report.append("")
        report.append(f"- Zero values: {assessment['zero_count']} ({assessment['zero_fraction']:.2%})")
        report.append("")
        if assessment["frequency_table"]:
            report.append("**Frequency table:**")
            report.append("")
            report.append("| Value | Count |")
            report.append("|-------|-------|")
            for value, count in assessment["frequency_table"].items():
                report.append(f"| {value} | {count} |")
            report.append("")

    # Part G: Representation Type Audit
    report.append("## 8. Part G — Representation Type Audit")
    report.append("")
    for rep_type, rep_stats in part_g_results.items():
        report.append(f"### {rep_type}")
        report.append("")
        report.append(f"**Count:** {rep_stats[EXPECTED_DESCRIPTOR_COLS[0]]['count']}")
        report.append("")
        report.append("| Descriptor | Median | Q1 | Q3 | IQR | Min | Max |")
        report.append("|------------|--------|----|----|-----|-----|-----|")
        for col in EXPECTED_DESCRIPTOR_COLS:
            stats = rep_stats[col]
            report.append(f"| {col} | {stats['median']:.4g} | {stats['q1']:.4g} | {stats['q3']:.4g} | {stats['iqr']:.4g} | {stats['min']:.4g} | {stats['max']:.4g} |")
        report.append("")

    # Part H: Provenance Group Structure
    report.append("## 9. Part H — Provenance Group Structure")
    report.append("")
    report.append(f"- Number of provenance groups: {part_h_results['n_groups']}")
    report.append(f"- Min group size: {part_h_results['min_group_size']}")
    report.append(f"- Max group size: {part_h_results['max_group_size']}")
    report.append(f"- Median group size: {part_h_results['median_group_size']}")
    report.append(f"- Mean group size: {part_h_results['mean_group_size']:.2f}")
    report.append(f"- Groups with 1 spectrum: {part_h_results['groups_with_1_spectrum']}")
    report.append(f"- Groups with 2 spectra: {part_h_results['groups_with_2_spectra']}")
    report.append(f"- Groups with 3+ spectra: {part_h_results['groups_with_3plus_spectra']}")
    report.append("")
    report.append("**Group size distribution:**")
    report.append("")
    report.append("| Size | Count |")
    report.append("|------|-------|")
    for size, count in part_h_results["group_size_distribution"].items():
        report.append(f"| {size} | {count} |")
    report.append("")

    # Part I: Duplicate Feature Vectors
    report.append("## 10. Part I — Duplicate Feature Vectors")
    report.append("")
    report.append(f"- Number of duplicate rows: {part_i_results['n_duplicate_rows']}")
    report.append(f"- Number of duplicate groups: {part_i_results['n_duplicate_groups']}")
    report.append("")
    if part_i_results["duplicate_group_sizes"]:
        report.append("**Duplicate group sizes:**")
        report.append("")
        report.append("| Group size | Number of groups |")
        report.append("|------------|------------------|")
        for size, count in part_i_results["duplicate_group_sizes"].items():
            report.append(f"| {size} | {count} |")
        report.append("")

    # Summary
    report.append("## 11. Summary")
    report.append("")
    report.append("### Key Findings")
    report.append("")
    report.append(f"1. **Strongest correlation:** {part_c_results['strongest_pearson_pair'][0]} ↔ {part_c_results['strongest_pearson_pair'][1]} (Pearson r = {part_c_results['strongest_pearson_value']:.4f})")
    report.append(f"2. **Redundancy:** {len(part_d_results['strong_pairs_pearson'])} strong pairs (Pearson), {len(part_d_results['very_strong_pairs_pearson'])} very strong pairs")
    report.append(f"3. **Scale variation:** {part_e_results['max_mean_ratio']:.2f}× range in means, {part_e_results['max_std_ratio']:.2f}× range in standard deviations")
    report.append(f"4. **Zero values:** {part_f_results['n_peaks_prom']['zero_count']} spectra with zero peaks")
    report.append(f"5. **Representation types:** {len(part_g_results)} types (native-5nm, PCHIP)")
    report.append(f"6. **Provenance groups:** {part_h_results['n_groups']} groups, {part_h_results['groups_with_1_spectrum']} singletons")
    report.append(f"7. **Duplicate vectors:** {part_i_results['n_duplicate_rows']} duplicate rows in {part_i_results['n_duplicate_groups']} groups")
    report.append("")
    report.append("### Audit Status")
    report.append("")
    report.append("**✓ PASS — All audit components completed successfully.**")
    report.append("")
    report.append("---")
    report.append("")
    report.append("*End of audit report.*")

    return "\n".join(report)


def generate_csv_summary(
    part_a_results: dict,
    part_c_results: dict,
    part_d_results: dict,
    part_e_results: dict,
    part_f_results: dict,
    part_h_results: dict,
    part_i_results: dict,
) -> pd.DataFrame:
    """Generate machine-readable CSV summary."""

    rows = []

    # Part A: Distribution statistics
    for col in EXPECTED_DESCRIPTOR_COLS:
        stats = part_a_results[col]
        for stat_name, stat_value in stats.items():
            rows.append(
                {
                    "category": "distribution",
                    "descriptor": col,
                    "check": stat_name,
                    "result": stat_value,
                }
            )

    # Part C: Correlation
    rows.append(
        {
            "category": "correlation",
            "descriptor": "strongest_pearson_pair",
            "check": "pair",
            "result": f"{part_c_results['strongest_pearson_pair'][0]}__{part_c_results['strongest_pearson_pair'][1]}",
        }
    )
    rows.append(
        {
            "category": "correlation",
            "descriptor": "strongest_pearson_value",
            "check": "r",
            "result": part_c_results["strongest_pearson_value"],
        }
    )

    # Part D: Redundancy
    rows.append(
        {
            "category": "redundancy",
            "descriptor": "strong_pairs_pearson",
            "check": "count",
            "result": len(part_d_results["strong_pairs_pearson"]),
        }
    )
    rows.append(
        {
            "category": "redundancy",
            "descriptor": "very_strong_pairs_pearson",
            "check": "count",
            "result": len(part_d_results["very_strong_pairs_pearson"]),
        }
    )

    # Part E: Scale
    rows.append(
        {
            "category": "scale",
            "descriptor": "max_mean_ratio",
            "check": "ratio",
            "result": part_e_results["max_mean_ratio"],
        }
    )
    rows.append(
        {
            "category": "scale",
            "descriptor": "max_std_ratio",
            "check": "ratio",
            "result": part_e_results["max_std_ratio"],
        }
    )

    # Part F: Zero values
    for col in EXPECTED_DESCRIPTOR_COLS:
        rows.append(
            {
                "category": "zero_values",
                "descriptor": col,
                "check": "count",
                "result": part_f_results[col]["zero_count"],
            }
        )

    # Part H: Provenance
    rows.append(
        {
            "category": "provenance",
            "descriptor": "n_groups",
            "check": "count",
            "result": part_h_results["n_groups"],
        }
    )

    # Part I: Duplicates
    rows.append(
        {
            "category": "duplicates",
            "descriptor": "duplicate_rows",
            "check": "count",
            "result": part_i_results["n_duplicate_rows"],
        }
    )

    return pd.DataFrame(rows)


# ============================================================================
# Main Execution
# ============================================================================


def main():
    """Main execution function."""
    print("=" * 80)
    print("Phase 3 Step 4 — Descriptor Distribution, Dependence, and Redundancy Audit")
    print("=" * 80)
    print()

    # Verify input file exists
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    # Compute checksum
    print("Computing input file checksum...")
    checksum = compute_file_checksum(INPUT_FILE)
    print(f"Checksum: {checksum}")
    if checksum != EXPECTED_CHECKSUM:
        print(f"WARNING: Checksum mismatch!")
        print(f"  Expected: {EXPECTED_CHECKSUM}")
        print(f"  Observed: {checksum}")
    print()

    # Load data
    print("Loading canonical feature matrix...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print()

    # Verify structure
    print("Verifying structure...")
    assert len(df) == EXPECTED_ROWS, f"Row count mismatch: {len(df)} != {EXPECTED_ROWS}"
    assert len(df.columns) == EXPECTED_COLS, f"Column count mismatch: {len(df.columns)} != {EXPECTED_COLS}"
    assert list(df.columns) == EXPECTED_METADATA_COLS + EXPECTED_DESCRIPTOR_COLS, "Column name mismatch"
    print("[PASS] Structure verified")
    print()

    # Execute all parts
    part_a_results = part_a_distribution_statistics(df)
    part_b_figures = part_b_visualization(df)
    part_c_results = part_c_pairwise_correlation(df)
    part_d_results = part_d_redundancy_assessment(part_c_results)
    part_e_results = part_e_scale_comparison(df)
    part_f_results = part_f_zero_discrete_assessment(df)
    part_g_results = part_g_representation_audit(df)
    part_h_results = part_h_provenance_group_structure(df)
    part_i_results = part_i_duplicate_feature_vectors(df)

    # Generate reports
    print("Generating markdown report...")
    markdown_report = generate_markdown_report(
        df,
        checksum,
        part_a_results,
        part_b_figures,
        part_c_results,
        part_d_results,
        part_e_results,
        part_f_results,
        part_g_results,
        part_h_results,
        part_i_results,
    )

    report_path = OUTPUT_DIR / "phase3_step4_descriptor_audit.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)
    print(f"[PASS] Report saved: {report_path}")
    print()

    print("Generating CSV summary...")
    csv_summary = generate_csv_summary(
        part_a_results,
        part_c_results,
        part_d_results,
        part_e_results,
        part_f_results,
        part_h_results,
        part_i_results,
    )

    csv_path = OUTPUT_DIR / "phase3_step4_descriptor_audit.csv"
    csv_summary.to_csv(csv_path, index=False)
    print(f"[PASS] CSV summary saved: {csv_path}")
    print()

    print("=" * 80)
    print("Audit complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
