"""
Phase 3 Step 1 — Spectral Complexity Descriptor Investigation
==============================================================

Investigates candidate spectral-complexity descriptors for the frozen
Phase 2 canonical dataset (454 spectra, 320-2550 nm, 5 nm spacing).

Produces:
  - Per-spectrum descriptor values (investigation table)
  - Descriptor summary statistics
  - Descriptor redundancy matrix
  - Diagnostic figures
  - Investigation report

Does NOT:
  - Modify the canonical spectral dataset
  - Run ML, PySR, or hypothesis tests
  - Use mineral labels to optimize descriptors
  - Freeze the final feature matrix
"""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import find_peaks
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "03_complexity_features"
FIGURES_DIR = RESULTS_DIR / "figures"

CANONICAL_SPECTRA = DATA_DIR / "phase2_canonical_d2_5nm.csv"
CANONICAL_METADATA = DATA_DIR / "phase2_canonical_d2_5nm_metadata.csv"

WAVELENGTH_START = 320
WAVELENGTH_END = 2550
WAVELENGTH_STEP = 5
N_EXPECTED_SPECTRA = 454
N_EXPECTED_WAVELENGTHS = 447
WL_RANGE = WAVELENGTH_END - WAVELENGTH_START

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PROMINENCE_THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.05]
DEFAULT_PROMINENCE = 0.01

WINDOW_SIZES = [5, 10, 20, 50]

# ---------------------------------------------------------------------------
# 2. CANONICAL DATASET VERIFICATION
# ---------------------------------------------------------------------------

def compute_file_checksum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_canonical_dataset(spectra_df, metadata_df):
    checks = {}

    checks["n_spectra"] = len(spectra_df) == N_EXPECTED_SPECTRA
    wl_cols = [c for c in spectra_df.columns if c.startswith("R_")]
    checks["n_wavelengths"] = len(wl_cols) == N_EXPECTED_WAVELENGTHS

    expected_wl = np.arange(WAVELENGTH_START, WAVELENGTH_END + 1, WAVELENGTH_STEP)
    actual_wl = np.array([int(c.replace("R_", "").replace("nm", "")) for c in wl_cols])
    checks["wavelength_grid"] = np.array_equal(actual_wl, expected_wl)

    checks["n_minerals"] = metadata_df["mineral"].nunique() == 18
    checks["n_provenance_groups"] = metadata_df["specimen_group_id"].nunique() == 382

    rep_counts = metadata_df["representation_type"].value_counts()
    checks["native_5nm_count"] = rep_counts.get("native_5nm", 0) == 390
    checks["pchip_count"] = rep_counts.get("pchip_from_native_10nm", 0) == 64

    checks["no_nan"] = spectra_df[wl_cols].notna().all().all()
    checks["no_inf"] = np.all(np.isfinite(spectra_df[wl_cols].values))

    return checks


# ---------------------------------------------------------------------------
# 3. DESCRIPTOR COMPUTATION
# ---------------------------------------------------------------------------

def compute_wavelength_array():
    return np.arange(WAVELENGTH_START, WAVELENGTH_END + 1, WAVELENGTH_STEP, dtype=np.float64)


def compute_first_derivative(reflectance, wl):
    return np.gradient(reflectance, wl)


def compute_second_derivative(reflectance, wl):
    d1 = compute_first_derivative(reflectance, wl)
    return np.gradient(d1, wl)


def compute_all_descriptors(reflectance, wl, prominence=DEFAULT_PROMINENCE):
    descriptors = {}
    d1 = compute_first_derivative(reflectance, wl)
    d2 = compute_second_derivative(reflectance, wl)
    dwl = wl[1] - wl[0]

    # --- Family A: Feature / extrema structure ---

    peaks, _ = find_peaks(reflectance, prominence=prominence, distance=3)
    troughs, _ = find_peaks(-reflectance, prominence=prominence, distance=3)

    descriptors["n_peaks_prom"] = len(peaks)
    descriptors["n_troughs_prom"] = len(troughs)
    descriptors["n_extrema_prom"] = len(peaks) + len(troughs)
    descriptors["peak_density_per_1000nm"] = len(peaks) / (WL_RANGE / 1000.0)
    descriptors["extrema_density_per_1000nm"] = (len(peaks) + len(troughs)) / (WL_RANGE / 1000.0)

    peaks_unprom, _ = find_peaks(reflectance, distance=3)
    troughs_unprom, _ = find_peaks(-reflectance, distance=3)
    descriptors["n_peaks_unprom"] = len(peaks_unprom)
    descriptors["n_troughs_unprom"] = len(troughs_unprom)
    descriptors["n_extrema_unprom"] = len(peaks_unprom) + len(troughs_unprom)

    if len(peaks) > 0:
        descriptors["mean_peak_prominence"] = float(np.mean(peaks[1] if len(peaks) > 0 and isinstance(peaks, tuple) else np.array([0.0])))
    else:
        descriptors["mean_peak_prominence"] = 0.0

    peak_props = find_peaks(reflectance, prominence=prominence, distance=3)
    if len(peak_props[0]) > 0 and peak_props[1] is not None:
        descriptors["mean_peak_prominence"] = float(np.mean(peak_props[1]["prominences"]))
        descriptors["max_peak_prominence"] = float(np.max(peak_props[1]["prominences"]))
    else:
        descriptors["mean_peak_prominence"] = 0.0
        descriptors["max_peak_prominence"] = 0.0

    # --- Family B: Derivative structure ---

    descriptors["first_deriv_total_variation"] = float(np.sum(np.abs(np.diff(d1))))
    descriptors["first_deriv_rms"] = float(np.sqrt(np.mean(d1 ** 2)))
    descriptors["first_deriv_abs_integral"] = float(np.trapezoid(np.abs(d1), wl))
    descriptors["first_deriv_sign_changes"] = int(np.sum(np.diff(np.sign(d1)) != 0))

    descriptors["second_deriv_total_variation"] = float(np.sum(np.abs(np.diff(d2))))
    descriptors["second_deriv_rms"] = float(np.sqrt(np.mean(d2 ** 2)))
    descriptors["second_deriv_abs_integral"] = float(np.trapezoid(np.abs(d2), wl))
    descriptors["second_deriv_sign_changes"] = int(np.sum(np.diff(np.sign(d2)) != 0))

    # --- Family C: Curvature ---

    descriptors["integrated_abs_d2"] = float(np.trapezoid(np.abs(d2), wl))
    descriptors["curvature_rms"] = float(np.sqrt(np.mean(d2 ** 2)))
    descriptors["curvature_total_variation"] = float(np.sum(np.abs(np.diff(d2))))

    # --- Family D: Roughness / local variation ---

    descriptors["total_variation"] = float(np.sum(np.abs(np.diff(reflectance))))
    descriptors["mean_abs_adj_diff"] = float(np.mean(np.abs(np.diff(reflectance))))
    descriptors["rms_adj_diff"] = float(np.sqrt(np.mean(np.diff(reflectance) ** 2)))

    # --- Family E: Entropy / information-like ---

    abs_refl = np.abs(reflectance)
    s = abs_refl.sum()
    if s > 0:
        p = abs_refl / s
        p = p[p > 0]
        descriptors["spectral_entropy"] = float(-np.sum(p * np.log2(p)))
    else:
        descriptors["spectral_entropy"] = 0.0

    abs_d1 = np.abs(d1)
    s1 = abs_d1.sum()
    if s1 > 0:
        p1 = abs_d1 / s1
        p1 = p1[p1 > 0]
        descriptors["first_deriv_entropy"] = float(-np.sum(p1 * np.log2(p1)))
    else:
        descriptors["first_deriv_entropy"] = 0.0

    # --- Family F: Scale-dependent structure ---

    for w in WINDOW_SIZES:
        if len(reflectance) > w:
            block_starts = np.arange(0, len(reflectance) - w + 1, w)
            block_means = np.array([np.mean(reflectance[i:i + w]) for i in block_starts])
            descriptors[f"ms_variation_window{w}"] = float(np.sum(np.abs(np.diff(block_means))))
        else:
            descriptors[f"ms_variation_window{w}"] = 0.0

    return descriptors


def compute_descriptors_for_dataset(spectra_df, wl, prominence=DEFAULT_PROMINENCE):
    wl_cols = [c for c in spectra_df.columns if c.startswith("R_")]
    reflectance_matrix = spectra_df[wl_cols].values.astype(np.float64)

    all_descriptors = []
    for i in range(len(spectra_df)):
        desc = compute_all_descriptors(reflectance_matrix[i], wl, prominence=prominence)
        all_descriptors.append(desc)

    return pd.DataFrame(all_descriptors, index=spectra_df.index)


# ---------------------------------------------------------------------------
# 4. PARAMETER SENSITIVITY
# ---------------------------------------------------------------------------

def compute_parameter_sensitivity(spectra_df, wl):
    results = {}
    wl_cols = [c for c in spectra_df.columns if c.startswith("R_")]
    reflectance_matrix = spectra_df[wl_cols].values.astype(np.float64)

    for prom in PROMINENCE_THRESHOLDS:
        peak_counts = []
        trough_counts = []
        extrema_counts = []
        for i in range(len(spectra_df)):
            p, _ = find_peaks(reflectance_matrix[i], prominence=prom, distance=3)
            t, _ = find_peaks(-reflectance_matrix[i], prominence=prom, distance=3)
            peak_counts.append(len(p))
            trough_counts.append(len(t))
            extrema_counts.append(len(p) + len(t))

        results[prom] = {
            "peak_counts": peak_counts,
            "trough_counts": trough_counts,
            "extrema_counts": extrema_counts,
            "median_peaks": np.median(peak_counts),
            "median_troughs": np.median(trough_counts),
            "median_extrema": np.median(extrema_counts),
        }

    return results


# ---------------------------------------------------------------------------
# 5. REPRESENTATION SENSITIVITY
# ---------------------------------------------------------------------------

def compute_representation_sensitivity(desc_df, metadata_df):
    native_mask = metadata_df["representation_type"] == "native_5nm"
    pchip_mask = metadata_df["representation_type"] == "pchip_from_native_10nm"

    native_desc = desc_df[native_mask.values]
    pchip_desc = desc_df[pchip_mask.values]

    comparison = {}
    for col in desc_df.columns:
        n_med = float(np.median(native_desc[col]))
        p_med = float(np.median(pchip_desc[col]))
        n_mean = float(np.mean(native_desc[col]))
        p_mean = float(np.mean(pchip_desc[col]))

        if n_med != 0:
            rel_diff_median = (p_med - n_med) / abs(n_med)
        else:
            rel_diff_median = float("nan")

        if n_mean != 0:
            rel_diff_mean = (p_mean - n_mean) / abs(n_mean)
        else:
            rel_diff_mean = float("nan")

        comparison[col] = {
            "native_median": n_med,
            "pchip_median": p_med,
            "native_mean": n_mean,
            "pchip_mean": p_mean,
            "relative_diff_median": rel_diff_median,
            "relative_diff_mean": rel_diff_mean,
        }

    return pd.DataFrame(comparison).T


# ---------------------------------------------------------------------------
# 6. REDUNDANCY ANALYSIS
# ---------------------------------------------------------------------------

def compute_redundancy(desc_df):
    pearson_corr = desc_df.corr(method="pearson")
    spearman_corr = desc_df.corr(method="spearman")
    return pearson_corr, spearman_corr


# ---------------------------------------------------------------------------
# 7. NUMERICAL STABILITY
# ---------------------------------------------------------------------------

def verify_numerical_stability(spectra_df, wl):
    desc1 = compute_descriptors_for_dataset(spectra_df, wl)
    desc2 = compute_descriptors_for_dataset(spectra_df, wl)

    identical = True
    max_diff = 0.0
    for col in desc1.columns:
        col_diff = np.max(np.abs(desc1[col].values - desc2[col].values))
        if col_diff > 0:
            identical = False
        max_diff = max(max_diff, col_diff)

    return identical, max_diff


# ---------------------------------------------------------------------------
# 8. FIGURE GENERATION
# ---------------------------------------------------------------------------

def fig1_descriptor_distributions(desc_df, descriptor_meta):
    descriptors = [c for c in desc_df.columns if c in descriptor_meta.index]
    n = len(descriptors)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows))
    axes = axes.flatten()

    for i, desc_name in enumerate(descriptors):
        ax = axes[i]
        values = desc_df[desc_name].values
        ax.hist(values, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
        ax.set_title(desc_name, fontsize=8, fontweight="bold")
        ax.axvline(np.median(values), color="red", linestyle="--", linewidth=1, label=f"median={np.median(values):.3g}")
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Phase 3 Step 1 — Descriptor Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGURES_DIR / "fig1_descriptor_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig2_descriptor_correlation(pearson_corr):
    fig, ax = plt.subplots(figsize=(16, 14))

    mask_upper = np.triu(np.ones_like(pearson_corr, dtype=bool), k=1)
    cmap = LinearSegmentedColormap.from_list("div", ["#2166ac", "#f7f7f7", "#b2182b"])

    im = ax.imshow(pearson_corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    labels = pearson_corr.columns
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = pearson_corr.iloc[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=4, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson correlation")
    ax.set_title("Phase 3 Step 1 — Descriptor Correlation Matrix (Pearson)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_descriptor_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig3_representation_comparison(rep_sensitivity):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    descriptors = rep_sensitivity.index.tolist()
    rel_diffs = rep_sensitivity["relative_diff_median"].values.astype(float)
    native_medians = rep_sensitivity["native_median"].values.astype(float)
    pchip_medians = rep_sensitivity["pchip_median"].values.astype(float)

    valid = np.isfinite(rel_diffs)
    desc_valid = [descriptors[i] for i in range(len(descriptors)) if valid[i]]
    diffs_valid = rel_diffs[valid]

    idx_sorted = np.argsort(np.abs(diffs_valid))[::-1]

    ax = axes[0]
    ax.barh(range(len(desc_valid)), [diffs_valid[i] for i in idx_sorted], color="steelblue", edgecolor="black")
    ax.set_yticks(range(len(desc_valid)))
    ax.set_yticklabels([desc_valid[i] for i in idx_sorted], fontsize=6)
    ax.set_xlabel("Relative difference (PCHIP - native) / native", fontsize=9)
    ax.set_title("Representation Sensitivity\nRelative Median Difference", fontsize=10, fontweight="bold")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.tick_params(labelsize=7)

    ax = axes[1]
    n_desc = len(desc_valid)
    bar_width = 0.35
    y_pos = np.arange(n_desc)
    native_vals = [native_medians[i] for i in range(len(descriptors)) if valid[i]]
    pchip_vals = [pchip_medians[i] for i in range(len(descriptors)) if valid[i]]
    native_sorted = [native_vals[i] for i in idx_sorted]
    pchip_sorted = [pchip_vals[i] for i in idx_sorted]

    ax.barh(y_pos - bar_width / 2, native_sorted, bar_width, label="Native 5nm", color="steelblue", edgecolor="black")
    ax.barh(y_pos + bar_width / 2, pchip_sorted, bar_width, label="PCHIP 10->5nm", color="coral", edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([desc_valid[i] for i in idx_sorted], fontsize=6)
    ax.set_xlabel("Median value", fontsize=9)
    ax.set_title("Median Values by Representation", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=7)

    ax = axes[2]
    n_high = np.sum(np.abs(diffs_valid) > 0.10)
    n_med = np.sum((np.abs(diffs_valid) > 0.02) & (np.abs(diffs_valid) <= 0.10))
    n_low = np.sum(np.abs(diffs_valid) <= 0.02)
    ax.bar(["High (>|10%|)", "Medium (2-10%)", "Low (<=2%)"], [n_high, n_med, n_low],
           color=["#b2182b", "#f4a582", "#92c5de"], edgecolor="black")
    ax.set_ylabel("Number of descriptors", fontsize=9)
    ax.set_title("Sensitivity Categories", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)

    fig.suptitle("Phase 3 Step 1 — Representation Sensitivity (Native 5nm vs PCHIP)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES_DIR / "fig3_representation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig4_parameter_sensitivity(param_sensitivity):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    proms = sorted(param_sensitivity.keys())
    median_peaks = [param_sensitivity[p]["median_peaks"] for p in proms]
    median_troughs = [param_sensitivity[p]["median_troughs"] for p in proms]
    median_extrema = [param_sensitivity[p]["median_extrema"] for p in proms]

    axes[0].plot(proms, median_peaks, "o-", color="steelblue", linewidth=2, markersize=6)
    axes[0].set_xlabel("Prominence threshold", fontsize=10)
    axes[0].set_ylabel("Median peak count", fontsize=10)
    axes[0].set_title("Peak Count vs Prominence", fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(labelsize=8)

    axes[1].plot(proms, median_troughs, "o-", color="coral", linewidth=2, markersize=6)
    axes[1].set_xlabel("Prominence threshold", fontsize=10)
    axes[1].set_ylabel("Median trough count", fontsize=10)
    axes[1].set_title("Trough Count vs Prominence", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(labelsize=8)

    axes[2].plot(proms, median_extrema, "o-", color="forestgreen", linewidth=2, markersize=6)
    axes[2].set_xlabel("Prominence threshold", fontsize=10)
    axes[2].set_ylabel("Median extrema count", fontsize=10)
    axes[2].set_title("Total Extrema vs Prominence", fontsize=11, fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(labelsize=8)

    fig.suptitle("Phase 3 Step 1 — Parameter Sensitivity (Prominence Threshold)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES_DIR / "fig4_parameter_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig5_example_spectra_with_features(spectra_df, metadata_df, wl):
    wl_cols = [c for c in spectra_df.columns if c.startswith("R_")]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    minerals = metadata_df["mineral"].unique()
    np.random.seed(42)
    selected = []
    for _ in range(6):
        mineral = np.random.choice(minerals)
        subset = metadata_df[metadata_df["mineral"] == mineral]
        idx = subset.index[np.random.choice(len(subset))]
        selected.append(idx)

    for i, idx in enumerate(selected):
        ax = axes[i]
        refl = spectra_df.loc[idx, wl_cols].values.astype(np.float64)
        mineral = metadata_df.loc[idx, "mineral"]
        spec_id = metadata_df.loc[idx, "canonical_spectrum_id"]
        rep_type = metadata_df.loc[idx, "representation_type"]

        ax.plot(wl, refl, "k-", linewidth=0.8, label="Reflectance")

        peaks, peak_props = find_peaks(refl, prominence=DEFAULT_PROMINENCE, distance=3)
        troughs, _ = find_peaks(-refl, prominence=DEFAULT_PROMINENCE, distance=3)

        ax.plot(wl[peaks], refl[peaks], "rv", markersize=5, label=f"Peaks ({len(peaks)})")
        ax.plot(wl[troughs], refl[troughs], "b^", markersize=5, label=f"Troughs ({len(troughs)})")

        d1 = np.gradient(refl, wl)
        sign_changes_d1 = np.sum(np.diff(np.sign(d1)) != 0)
        ax.set_title(f"{spec_id} — {mineral}\n{len(peaks)} peaks, {sign_changes_d1} d1 sign changes",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Wavelength (nm)", fontsize=8)
        ax.set_ylabel("Reflectance", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=7)

    fig.suptitle(f"Phase 3 Step 1 — Example Spectra with Detected Features (prominence={DEFAULT_PROMINENCE})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES_DIR / "fig5_example_spectra_features.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig6_descriptor_family_comparison(desc_df, descriptor_meta):
    families = descriptor_meta["family"].unique()
    family_descriptors = {f: descriptor_meta[descriptor_meta["family"] == f].index.tolist() for f in families}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, family in enumerate(sorted(families)):
        if i >= 6:
            break
        ax = axes[i]
        descs = [d for d in family_descriptors[family] if d in desc_df.columns]
        if not descs:
            ax.set_visible(False)
            continue

        data = desc_df[descs].values
        medians = np.median(data, axis=0)
        q25 = np.percentile(data, 25, axis=0)
        q75 = np.percentile(data, 75, axis=0)

        x = np.arange(len(descs))
        ax.bar(x, medians, yerr=[medians - q25, q75 - medians], capsize=3,
               color="steelblue", edgecolor="black", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(descs, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Median value (IQR)", fontsize=9)
        ax.set_title(f"Family: {family}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Phase 3 Step 1 — Descriptor Family Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES_DIR / "fig6_descriptor_family_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 9. DESCRIPTOR METADATA TABLE
# ---------------------------------------------------------------------------

def build_descriptor_metadata():
    records = []

    # Family A
    records.append(("n_peaks_prom", "A_FeatureStructure", "count",
                     "Local maxima count with prominence >= 0.01", "Number of distinct absorption/reflection peaks",
                     "Prominence threshold, distance parameter", "0-200", "retain"))
    records.append(("n_troughs_prom", "A_FeatureStructure", "count",
                     "Local minima count with prominence >= 0.01", "Number of distinct absorption troughs",
                     "Prominence threshold, distance parameter", "0-200", "retain"))
    records.append(("n_extrema_prom", "A_FeatureStructure", "count",
                     "Total extrema count (peaks + troughs) with prominence >= 0.01",
                     "Total number of structural features",
                     "Prominence threshold, distance parameter", "0-400", "retain"))
    records.append(("peak_density_per_1000nm", "A_FeatureStructure", "count per 1000 nm",
                     "Peak count normalized by wavelength range", "Feature density independent of range",
                     "Prominence threshold", "0-100", "retain"))
    records.append(("extrema_density_per_1000nm", "A_FeatureStructure", "count per 1000 nm",
                     "Extrema count normalized by wavelength range", "Total feature density",
                     "Prominence threshold", "0-200", "retain"))
    records.append(("n_peaks_unprom", "A_FeatureStructure", "count",
                     "Local maxima count without prominence filter", "All detected peaks including noise-level",
                     "Distance parameter only", "0-447", "investigate"))
    records.append(("n_troughs_unprom", "A_FeatureStructure", "count",
                     "Local minima count without prominence filter", "All detected troughs including noise-level",
                     "Distance parameter only", "0-447", "investigate"))
    records.append(("n_extrema_unprom", "A_FeatureStructure", "count",
                     "Total extrema without prominence filter", "All structural features including noise-level",
                     "Distance parameter only", "0-447", "investigate"))
    records.append(("mean_peak_prominence", "A_FeatureStructure", "reflectance units",
                     "Mean prominence of detected peaks", "Average peak strength",
                     "Prominence threshold", "0-1", "retain"))
    records.append(("max_peak_prominence", "A_FeatureStructure", "reflectance units",
                     "Maximum prominence among detected peaks", "Strongest peak magnitude",
                     "Prominence threshold", "0-1", "investigate"))

    # Family B
    records.append(("first_deriv_total_variation", "B_DerivativeStructure", "reflectance / nm",
                     "Sum of absolute first-difference of first derivative", "Cumulative local slope change",
                     "Grid spacing", "0-inf", "retain"))
    records.append(("first_deriv_rms", "B_DerivativeStructure", "reflectance / nm",
                     "RMS of first derivative", "Average local slope magnitude",
                     "Grid spacing", "0-inf", "retain"))
    records.append(("first_deriv_abs_integral", "B_DerivativeStructure", "reflectance",
                     "Trapezoidal integral of absolute first derivative", "Total absolute spectral change",
                     "Grid spacing", "0-inf", "retain"))
    records.append(("first_deriv_sign_changes", "B_DerivativeStructure", "count",
                     "Number of sign changes in first derivative", "Number of local direction reversals",
                     "Grid spacing", "0-447", "investigate"))
    records.append(("second_deriv_total_variation", "B_DerivativeStructure", "reflectance / nm^2",
                     "Sum of absolute first-difference of second derivative", "Cumulative curvature change",
                     "Grid spacing", "0-inf", "investigate"))
    records.append(("second_deriv_rms", "B_DerivativeStructure", "reflectance / nm^2",
                     "RMS of second derivative", "Average curvature magnitude",
                     "Grid spacing", "0-inf", "retain"))
    records.append(("second_deriv_abs_integral", "B_DerivativeStructure", "reflectance / nm",
                     "Trapezoidal integral of absolute second derivative", "Total absolute curvature",
                     "Grid spacing", "0-inf", "retain"))
    records.append(("second_deriv_sign_changes", "B_DerivativeStructure", "count",
                     "Number of sign changes in second derivative", "Number of inflection points",
                     "Grid spacing", "0-447", "investigate"))

    # Family C
    records.append(("integrated_abs_d2", "C_Curvature", "reflectance / nm",
                     "Trapezoidal integral of absolute second derivative", "Total structural bending",
                     "Grid spacing", "0-inf", "retain"))
    records.append(("curvature_rms", "C_Curvature", "reflectance / nm^2",
                     "RMS of second derivative", "Average curvature intensity",
                     "Grid spacing", "0-inf", "retain"))
    records.append(("curvature_total_variation", "C_Curvature", "reflectance / nm^2",
                     "Sum of absolute first-difference of second derivative", "Cumulative curvature variation",
                     "Grid spacing", "0-inf", "investigate"))

    # Family D
    records.append(("total_variation", "D_Roughness", "reflectance",
                     "Sum of absolute adjacent differences", "Cumulative local spectral change",
                     "Grid spacing", "0-inf", "retain"))
    records.append(("mean_abs_adj_diff", "D_Roughness", "reflectance",
                     "Mean absolute adjacent difference", "Average local spectral step",
                     "Grid spacing", "0-inf", "retain"))
    records.append(("rms_adj_diff", "D_Roughness", "reflectance",
                     "RMS of adjacent differences", "RMS local spectral step",
                     "Grid spacing", "0-inf", "retain"))

    # Family E
    records.append(("spectral_entropy", "E_Entropy", "bits",
                     "Shannon entropy of normalized absolute reflectance", "Distributional unpredictability of reflectance",
                     "None (direct from values)", "0-log2(447)", "investigate"))
    records.append(("first_deriv_entropy", "E_Entropy", "bits",
                     "Shannon entropy of normalized absolute first derivative", "Distributional unpredictability of spectral slope",
                     "None (direct from values)", "0-log2(446)", "investigate"))

    # Family F
    records.append(("ms_variation_window5", "F_ScaleDependent", "reflectance",
                     "Total variation of 5-point block means", "Local-scale spectral variation",
                     "Window size", "0-inf", "investigate"))
    records.append(("ms_variation_window10", "F_ScaleDependent", "reflectance",
                     "Total variation of 10-point block means", "Meso-scale spectral variation",
                     "Window size", "0-inf", "investigate"))
    records.append(("ms_variation_window20", "F_ScaleDependent", "reflectance",
                     "Total variation of 20-point block means", "Broad-scale spectral variation",
                     "Window size", "0-inf", "investigate"))
    records.append(("ms_variation_window50", "F_ScaleDependent", "reflectance",
                     "Total variation of 50-point block means", "Very-broad-scale spectral variation",
                     "Window size", "0-inf", "investigate"))

    df = pd.DataFrame(records, columns=["descriptor", "family", "units", "definition",
                                         "interpretation", "sensitivity", "expected_range", "recommendation"])
    df = df.set_index("descriptor")
    return df


# ---------------------------------------------------------------------------
# 10. MAIN EXECUTION
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Phase 3 Step 1 — Spectral Complexity Descriptor Investigation")
    print("=" * 70)

    # --- Load data ---
    print("\n[1/12] Loading canonical dataset...")
    spectra_df = pd.read_csv(CANONICAL_SPECTRA)
    metadata_df = pd.read_csv(CANONICAL_METADATA)
    print(f"  Loaded {len(spectra_df)} spectra, {len(metadata_df)} metadata rows")

    # --- Verify canonical dataset ---
    print("\n[2/12] Verifying canonical dataset invariants...")
    checks = verify_canonical_dataset(spectra_df, metadata_df)
    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {check_name}: {status}")
    if not all(checks.values()):
        print("\nERROR: Canonical dataset verification FAILED. Stopping.")
        sys.exit(1)
    print("  All canonical dataset checks PASS.")

    # --- Compute checksums ---
    print("\n[3/12] Computing file checksums...")
    spectra_checksum = compute_file_checksum(CANONICAL_SPECTRA)
    metadata_checksum = compute_file_checksum(CANONICAL_METADATA)
    print(f"  Spectra SHA-256: {spectra_checksum}")
    print(f"  Metadata SHA-256: {metadata_checksum}")

    # --- Build descriptor metadata ---
    print("\n[4/12] Building descriptor metadata...")
    descriptor_meta = build_descriptor_metadata()
    print(f"  {len(descriptor_meta)} candidate descriptors defined")

    # --- Compute descriptors ---
    print("\n[5/12] Computing candidate descriptors for 454 spectra...")
    wl = compute_wavelength_array()
    desc_df = compute_descriptors_for_dataset(spectra_df, wl)
    print(f"  Computed {len(desc_df.columns)} descriptors for {len(desc_df)} spectra")

    # --- Parameter sensitivity ---
    print("\n[6/12] Computing parameter sensitivity (prominence thresholds)...")
    param_sensitivity = compute_parameter_sensitivity(spectra_df, wl)
    for prom in PROMINENCE_THRESHOLDS:
        med_e = param_sensitivity[prom]["median_extrema"]
        print(f"  prominence={prom:.3f}: median extrema = {med_e:.1f}")

    # --- Representation sensitivity ---
    print("\n[7/12] Computing representation sensitivity (native vs PCHIP)...")
    rep_sensitivity = compute_representation_sensitivity(desc_df, metadata_df)
    high_sens = np.sum(np.abs(rep_sensitivity["relative_diff_median"].values.astype(float)) > 0.10)
    print(f"  {high_sens} descriptors with >10% relative median difference")

    # --- Redundancy ---
    print("\n[8/12] Computing descriptor redundancy...")
    pearson_corr, spearman_corr = compute_redundancy(desc_df)
    n_high_pearson = np.sum(np.triu(np.abs(pearson_corr.values) > 0.90, k=1))
    n_high_spearman = np.sum(np.triu(np.abs(spearman_corr.values) > 0.90, k=1))
    print(f"  Pairs with |Pearson| > 0.90: {n_high_pearson}")
    print(f"  Pairs with |Spearman| > 0.90: {n_high_spearman}")

    # --- Numerical stability ---
    print("\n[9/12] Verifying numerical stability (deterministic rerun)...")
    stable, max_diff = verify_numerical_stability(spectra_df, wl)
    print(f"  Deterministic: {stable}")
    print(f"  Max absolute difference across reruns: {max_diff}")

    # --- Build output descriptor values CSV ---
    print("\n[10/12] Writing output files...")
    desc_output = desc_df.copy()
    desc_output.insert(0, "canonical_spectrum_id", metadata_df["canonical_spectrum_id"].values)
    desc_output.insert(1, "specimen_group_id", metadata_df["specimen_group_id"].values)
    desc_output.insert(2, "mineral", metadata_df["mineral"].values)
    desc_output.insert(3, "representation_type", metadata_df["representation_type"].values)
    desc_output_path = RESULTS_DIR / "phase3_step1_descriptor_values.csv"
    desc_output.to_csv(desc_output_path, index=False)
    print(f"  Descriptor values: {desc_output_path}")

    # --- Build descriptor summary CSV ---
    summary_records = []
    for desc_name in desc_df.columns:
        if desc_name not in descriptor_meta.index:
            continue
        meta = descriptor_meta.loc[desc_name]
        values = desc_df[desc_name].values
        missing = int(np.sum(np.isnan(values)))

        rep_diff = float(rep_sensitivity.loc[desc_name, "relative_diff_median"]) if desc_name in rep_sensitivity.index else float("nan")
        if np.isnan(rep_diff):
            rep_status = "unknown"
        elif abs(rep_diff) > 0.10:
            rep_status = "high_sensitivity"
        elif abs(rep_diff) > 0.02:
            rep_status = "moderate_sensitivity"
        else:
            rep_status = "low_sensitivity"

        summary_records.append({
            "descriptor": desc_name,
            "family": meta["family"],
            "units": meta["units"],
            "definition": meta["definition"],
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "missing_count": missing,
            "numerical_stability": "stable" if stable else "unstable",
            "representation_sensitivity": rep_status,
            "recommendation": meta["recommendation"],
        })

    summary_df = pd.DataFrame(summary_records)
    summary_path = RESULTS_DIR / "phase3_step1_descriptor_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Descriptor summary: {summary_path}")

    # --- Redundancy matrix ---
    pearson_path = RESULTS_DIR / "phase3_step1_descriptor_redundancy.csv"
    pearson_corr.to_csv(pearson_path)
    print(f"  Redundancy matrix (Pearson): {pearson_path}")

    # --- Figures ---
    print("\n[11/12] Generating diagnostic figures...")
    fig1_descriptor_distributions(desc_df, descriptor_meta)
    print("  fig1_descriptor_distributions.png")
    fig2_descriptor_correlation(pearson_corr)
    print("  fig2_descriptor_correlation.png")
    fig3_representation_comparison(rep_sensitivity)
    print("  fig3_representation_comparison.png")
    fig4_parameter_sensitivity(param_sensitivity)
    print("  fig4_parameter_sensitivity.png")
    fig5_example_spectra_with_features(spectra_df, metadata_df, wl)
    print("  fig5_example_spectra_features.png")
    fig6_descriptor_family_comparison(desc_df, descriptor_meta)
    print("  fig6_descriptor_family_comparison.png")

    # --- Reproducibility checksums ---
    desc_checksum = compute_file_checksum(desc_output_path)
    summary_checksum = compute_file_checksum(summary_path)
    redundancy_checksum = compute_file_checksum(pearson_path)

    # --- Verify checksums unchanged after second run ---
    print("\n[12/12] Reproducibility verification (second computation pass)...")
    desc_df2 = compute_descriptors_for_dataset(spectra_df, wl)
    desc_output2 = desc_df2.copy()
    desc_output2.insert(0, "canonical_spectrum_id", metadata_df["canonical_spectrum_id"].values)
    desc_output2.insert(1, "specimen_group_id", metadata_df["specimen_group_id"].values)
    desc_output2.insert(2, "mineral", metadata_df["mineral"].values)
    desc_output2.insert(3, "representation_type", metadata_df["representation_type"].values)
    desc_output2_path = RESULTS_DIR / "phase3_step1_descriptor_values_run2.csv"
    desc_output2.to_csv(desc_output2_path, index=False)
    desc_checksum2 = compute_file_checksum(desc_output2_path)

    print(f"  Run 1 descriptor checksum: {desc_checksum}")
    print(f"  Run 2 descriptor checksum: {desc_checksum2}")
    print(f"  Identical: {desc_checksum == desc_checksum2}")

    if desc_checksum != desc_checksum2:
        print("  WARNING: Reproducibility check FAILED!")
    else:
        print("  Reproducibility PASS.")

    desc_output2_path.unlink()

    # --- Save run metadata ---
    run_meta = {
        "spectra_checksum_sha256": spectra_checksum,
        "metadata_checksum_sha256": metadata_checksum,
        "descriptor_values_checksum_sha256": desc_checksum,
        "descriptor_summary_checksum_sha256": summary_checksum,
        "redundancy_matrix_checksum_sha256": redundancy_checksum,
        "n_spectra": len(spectra_df),
        "n_descriptors": len(desc_df.columns),
        "n_wavelengths": len(wl),
        "prominence_threshold": DEFAULT_PROMINENCE,
        "window_sizes": WINDOW_SIZES,
        "numerical_stability": stable,
        "max_rerun_difference": max_diff,
        "reproducibility_pass": desc_checksum == desc_checksum2,
        "canonical_dataset_all_checks_pass": all(checks.values()),
    }
    run_meta_path = RESULTS_DIR / "phase3_step1_run_metadata.json"
    with open(run_meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"\n  Run metadata: {run_meta_path}")

    print("\n" + "=" * 70)
    print("Phase 3 Step 1 — COMPLETE")
    print("=" * 70)
    print(f"\nOutputs in: {RESULTS_DIR}")
    print(f"  - phase3_step1_descriptor_values.csv ({len(desc_df.columns)} descriptors x {len(desc_df)} spectra)")
    print(f"  - phase3_step1_descriptor_summary.csv ({len(summary_df)} descriptors)")
    print(f"  - phase3_step1_descriptor_redundancy.csv ({len(desc_df.columns)} x {len(desc_df.columns)})")
    print(f"  - phase3_step1_run_metadata.json")
    print(f"  - 6 diagnostic figures in figures/")
    print(f"\nReproducibility: {'PASS' if desc_checksum == desc_checksum2 else 'FAIL'}")
    print(f"Canonical dataset integrity: {'PASS' if all(checks.values()) else 'FAIL'}")


if __name__ == "__main__":
    main()
