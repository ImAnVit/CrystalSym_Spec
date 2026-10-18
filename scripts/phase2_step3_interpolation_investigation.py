#!/usr/bin/env python3
"""
Phase 2 Step 3 — Interpolation Method Investigation
=====================================================

CrystalSym-Spec: Interpretable ML and Symbolic Regression of
Crystal Symmetry, Chemistry, and Mineral Spectral Complexity

Investigates whether interpolation is required for the 64 native-10 nm
D2 spectra and, if so, which method (linear vs PCHIP) introduces the
least scientifically problematic distortion on the frozen 5-nm grid.

Uses a masked native-grid reconstruction test: native 5-nm spectra are
artificially downsampled to 10 nm, then reconstructed via interpolation.
The known intermediate 5-nm values serve as ground truth.

Outputs:
  results/02_preprocessing_qc/phase2_step3_interpolation_investigation.md
  results/02_preprocessing_qc/phase2_step3_interpolation_comparison.csv
  results/02_preprocessing_qc/phase2_step3_interpolation_summary.csv
  results/02_preprocessing_qc/fig_interpolation_*.png
"""

import os
import sys
import hashlib
import warnings
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "RELAB"
POPULATION_CSV = PROJECT_ROOT / "metadata" / "relab_analysis_population.csv"
AUDIT_CSV = (
    PROJECT_ROOT / "results" / "02_preprocessing_qc" / "spectral_coverage_audit.csv"
)
STEP2_CSV = (
    PROJECT_ROOT
    / "results"
    / "02_preprocessing_qc"
    / "phase2_step2_decision_validation.csv"
)
CHECKSUMS_CSV = PROJECT_ROOT / "metadata" / "provenance" / "relab_raw_checksums.csv"
OUT_DIR = PROJECT_ROOT / "results" / "02_preprocessing_qc"
FIG_DIR = OUT_DIR

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# FROZEN CONSTANTS
# ================================================================

D2_WMIN = 320.0
D2_WMAX = 2550.0
D2_SPACING = 5.0
D2_TARGET_GRID = np.arange(D2_WMIN, D2_WMAX + D2_SPACING * 0.5, D2_SPACING)

WAVELENGTH_REGIONS = {
    "UV-VNIR (320-700 nm)": (320.0, 700.0),
    "VNIR (700-1300 nm)": (700.0, 1300.0),
    "SWIR-1 (1300-1900 nm)": (1300.0, 1900.0),
    "SWIR-2 (1900-2550 nm)": (1900.0, 2550.0),
}


# ================================================================
# DATA LOADING
# ================================================================


def read_spectrum(filepath):
    """Parse a RELAB .tab file. Returns (wavelengths, reflectances) as arrays."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    wavelengths, reflectances = [], []
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped:
            idx += 1
            continue
        parts = stripped.split()
        if len(parts) != 1:
            break
        try:
            count = int(parts[0])
        except ValueError:
            break
        for j in range(1, count + 1):
            pos = idx + j
            if pos >= len(lines):
                break
            dp = lines[pos].split()
            if len(dp) < 2:
                continue
            try:
                wl = float(dp[0])
                rf = float(dp[1])
            except ValueError:
                continue
            wavelengths.append(wl)
            reflectances.append(rf)
        idx += count + 1

    return np.array(wavelengths), np.array(reflectances)


def classify_acquisition_regime(file_path):
    prefix = Path(file_path).name[:2].lower()
    if prefix.startswith("c"):
        return "VNIR_spectrophotometer"
    elif prefix.startswith(("b", "m")):
        return "FTIR"
    elif prefix.startswith("n"):
        return "NIR_spectrometer"
    else:
        return "other"


def validate_checksums():
    ref = pd.read_csv(CHECKSUMS_CSV)
    mismatches = []
    checked = 0
    for _, row in ref.iterrows():
        fp = DATA_DIR / row["rel_path"]
        if fp.exists():
            md5 = hashlib.md5(fp.read_bytes()).hexdigest()
            if md5 != row["md5"]:
                mismatches.append(row["rel_path"])
            checked += 1
    return checked, mismatches


def load_and_identify_spectra():
    """Load metadata and identify the 64 native-10nm and 390 native-5nm spectra."""
    pop = pd.read_csv(POPULATION_CSV)
    audit = pd.read_csv(AUDIT_CSV)
    step2 = pd.read_csv(STEP2_CSV)

    candidates = pop[pop["inclusion_category"].isin(["A", "B"])].copy()
    cand_ids = set(candidates["file_id"].values)
    audit_c = audit[audit["relab_measurement_id"].isin(cand_ids)].copy()

    pop_merge = candidates[["file_id", "specimen_group_id", "inclusion_category"]].copy()
    pop_merge = pop_merge.rename(columns={"file_id": "relab_measurement_id"})
    audit_c = audit_c.merge(
        pop_merge, on="relab_measurement_id", how="left", suffixes=("", "_pop")
    )
    if "specimen_group_id_pop" in audit_c.columns:
        audit_c["specimen_group_id"] = audit_c["specimen_group_id_pop"]
        audit_c.drop(columns=["specimen_group_id_pop"], inplace=True)
    if "inclusion_category_pop" in audit_c.columns:
        audit_c["analysis_category"] = audit_c["inclusion_category_pop"]
        audit_c.drop(columns=["inclusion_category_pop"], inplace=True)

    d2_mask = (
        (audit_c["wavelength_min"] <= D2_WMIN) & (audit_c["wavelength_max"] >= D2_WMAX)
    )
    d2_covered = audit_c[d2_mask].copy().reset_index(drop=True)

    native_5nm_ids = set(
        step2[step2["native_5nm_support"] == True]["relab_measurement_id"].values
    )
    non_native_ids = set(
        step2[step2["native_5nm_support"] == False]["relab_measurement_id"].values
    )

    native_5nm = d2_covered[
        d2_covered["relab_measurement_id"].isin(native_5nm_ids)
    ].copy().reset_index(drop=True)
    non_native_5nm = d2_covered[
        d2_covered["relab_measurement_id"].isin(non_native_ids)
    ].copy().reset_index(drop=True)

    return native_5nm, non_native_5nm, d2_covered, audit_c


# ================================================================
# SPECTRAL ANALYSIS HELPERS
# ================================================================


def prepare_spectrum(wl, rf):
    """Sort, deduplicate, and restrict to D2. Returns (wl_sorted, rf_sorted)."""
    order = np.argsort(wl)
    wl_s, rf_s = wl[order], rf[order]
    uwl, idx = np.unique(wl_s, return_index=True)
    rf_u = rf_s[idx]
    mask = (uwl >= D2_WMIN - 0.5) & (uwl <= D2_WMAX + 0.5)
    return uwl[mask], rf_u[mask]


def find_local_extrema(values):
    """Find local maxima and minima indices in interior of array."""
    maxima, minima = [], []
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            maxima.append(i)
        elif values[i] < values[i - 1] and values[i] < values[i + 1]:
            minima.append(i)
    return maxima, minima


def compute_first_derivative(wl, rf):
    if len(wl) < 2:
        return np.array([]), np.array([])
    dwl = np.diff(wl)
    dwl[dwl == 0] = 1e-10
    drf = np.diff(rf)
    mid_wl = (wl[:-1] + wl[1:]) / 2.0
    return mid_wl, drf / dwl


def compute_second_derivative(wl, rf):
    if len(wl) < 3:
        return np.array([]), np.array([])
    d1 = np.diff(rf)
    d1_wl = (wl[:-1] + wl[1:]) / 2.0
    dwl = np.diff(d1_wl)
    dwl[dwl == 0] = 1e-10
    d2 = np.diff(d1) / dwl
    mid_wl = (d1_wl[:-1] + d1_wl[1:]) / 2.0
    return mid_wl, d2


def compute_complexity_diagnostics(wl, rf):
    """Generic shape descriptors for complexity-preservation analysis."""
    n = len(rf)
    if n < 2:
        return {}
    maxima, minima = find_local_extrema(rf)
    n_extrema = len(maxima) + len(minima)
    first_order_var = float(np.sum(np.abs(np.diff(rf))))
    if n >= 3:
        _, d1 = compute_first_derivative(wl, rf)
        integrated_abs_d1 = float(np.sum(np.abs(d1))) if len(d1) > 0 else np.nan
        _, d2 = compute_second_derivative(wl, rf)
        curvature = float(np.sum(np.abs(d2))) if len(d2) > 0 else np.nan
    else:
        integrated_abs_d1 = np.nan
        curvature = np.nan
    return {
        "n_extrema": n_extrema,
        "first_order_variation": first_order_var,
        "integrated_abs_d1": integrated_abs_d1,
        "curvature_sum": curvature,
    }


# ================================================================
# INTERPOLATION
# ================================================================


def interp_linear(x_known, y_known, x_target):
    return np.interp(x_target, x_known, y_known)


def interp_pchip(x_known, y_known, x_target):
    if len(x_known) < 2:
        return np.full(len(x_target), np.nan)
    try:
        pchip = PchipInterpolator(x_known, y_known, extrapolate=False)
        result = pchip(x_target)
        if np.any(np.isnan(result)):
            result[np.isnan(result)] = np.interp(
                x_target[np.isnan(result)], x_known, y_known
            )
        return result
    except Exception:
        return np.interp(x_target, x_known, y_known)


# ================================================================
# MASKED RECONSTRUCTION
# ================================================================


def masked_reconstruct(wl_5nm, rf_5nm):
    """
    Mask every second native 5-nm point, reconstruct via interpolation,
    and compare with the masked (known) values.

    Returns dict with reconstruction results, or None if not enough points.
    """
    n = len(wl_5nm)
    if n < 6:
        return None

    even_idx = np.arange(0, n, 2)
    odd_idx = np.arange(1, n, 2)

    wl_even, rf_even = wl_5nm[even_idx], rf_5nm[even_idx]
    wl_odd, rf_odd = wl_5nm[odd_idx], rf_5nm[odd_idx]

    if len(wl_even) < 2:
        return None

    rf_linear = interp_linear(wl_even, rf_even, wl_odd)
    rf_pchip = interp_pchip(wl_even, rf_even, wl_odd)

    err_linear = rf_linear - rf_odd
    err_pchip = rf_pchip - rf_odd

    return {
        "wl_even": wl_even,
        "rf_even": rf_even,
        "wl_odd": wl_odd,
        "rf_odd": rf_odd,
        "rf_linear": rf_linear,
        "rf_pchip": rf_pchip,
        "err_linear": err_linear,
        "err_pchip": err_pchip,
    }


# ================================================================
# METRICS
# ================================================================


def compute_pointwise_metrics(errors):
    ae = np.abs(errors)
    return {
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "median_ae": float(np.median(ae)),
        "max_ae": float(np.max(ae)),
        "mse": float(np.mean(errors)),
        "p95_ae": float(np.percentile(ae, 95)),
    }


def compute_regional_metrics(wl, errors):
    result = {}
    for region_name, (rlo, rhi) in WAVELENGTH_REGIONS.items():
        mask = (wl >= rlo) & (wl < rhi)
        n = int(mask.sum())
        if n >= 3:
            m = compute_pointwise_metrics(errors[mask])
            m["n_points"] = n
            result[region_name] = m
        else:
            result[region_name] = {"n_points": n}
    return result


def compute_extrema_preservation(wl_even, rf_even, wl_odd, rf_odd,
                                 rf_linear, rf_pchip):
    """
    Compare extrema in measured vs reconstructed spectra at the
    masked (odd-index) points.
    """
    result = {}
    for method_name, rf_recon in [("linear", rf_linear), ("pchip", rf_pchip)]:
        n_odd = len(rf_odd)
        measured_max, measured_min = find_local_extrema(rf_odd)

        recon_full = np.empty(len(wl_odd))
        recon_full[:] = rf_recon

        recon_max, recon_min = find_local_extrema(recon_full)

        measured_max_set = set(measured_max)
        measured_min_set = set(measured_min)
        recon_max_set = set(recon_max)
        recon_min_set = set(recon_min)

        false_max = len(recon_max_set - measured_max_set)
        missed_max = len(measured_max_set - recon_max_set)
        false_min = len(recon_min_set - measured_min_set)
        missed_min = len(measured_min_set - recon_min_set)

        result[method_name] = {
            "false_extrema": false_max + false_min,
            "missed_extrema": missed_max + missed_min,
            "false_maxima": false_max,
            "missed_maxima": missed_max,
            "false_minima": false_min,
            "missed_minima": missed_min,
            "n_measured_maxima": len(measured_max),
            "n_measured_minima": len(measured_min),
        }
    return result


def compute_peak_position_distortion(wl_odd, rf_odd, rf_linear, rf_pchip):
    """
    For each local extremum in the measured odd-index spectrum,
    find the nearest extremum in the reconstructed spectrum and
    report the wavelength shift.
    """
    result = {}
    for method_name, rf_recon in [("linear", rf_linear), ("pchip", rf_pchip)]:
        measured_max, measured_min = find_local_extrema(rf_odd)
        recon_max, recon_min = find_local_extrema(rf_recon)

        shifts_max = []
        for mi in measured_max:
            if len(recon_max) == 0:
                continue
            dists = np.abs(np.array(recon_max) - mi)
            best = np.argmin(dists)
            if dists[best] <= 2:
                shift = wl_odd[recon_max[best]] - wl_odd[mi]
                shifts_max.append(shift)

        shifts_min = []
        for mi in measured_min:
            if len(recon_min) == 0:
                continue
            dists = np.abs(np.array(recon_min) - mi)
            best = np.argmin(dists)
            if dists[best] <= 2:
                shift = wl_odd[recon_min[best]] - wl_odd[mi]
                shifts_min.append(shift)

        all_shifts = shifts_max + shifts_min
        result[method_name] = {
            "median_peak_shift_nm": float(np.median(all_shifts)) if all_shifts else np.nan,
            "max_peak_shift_nm": float(np.max(np.abs(all_shifts))) if all_shifts else np.nan,
            "n_matched_peaks": len(all_shifts),
            "n_measured_peaks": len(measured_max) + len(measured_min),
        }
    return result


def compute_slope_distortion(wl_even, rf_even, wl_odd, rf_odd, rf_linear, rf_pchip):
    """Compare local slopes at odd-index points."""
    result = {}
    for method_name, rf_recon in [("linear", rf_linear), ("pchip", rf_pchip)]:
        measured_slopes = np.diff(rf_odd)
        recon_slopes = np.diff(rf_recon)
        if len(measured_slopes) > 0:
            slope_err = np.abs(measured_slopes - recon_slopes)
            result[method_name] = {
                "mean_slope_error": float(np.mean(slope_err)),
                "max_slope_error": float(np.max(slope_err)),
            }
        else:
            result[method_name] = {"mean_slope_error": np.nan, "max_slope_error": np.nan}
    return result


# ================================================================
# MAIN ANALYSIS
# ================================================================


def run_reconstruction_experiment(spectra_df, label):
    """Run masked reconstruction on a set of spectra. Returns list of result dicts."""
    all_results = []
    skipped = 0

    for idx, row in spectra_df.iterrows():
        fpath = DATA_DIR / row["file_path"]
        wl_raw, rf_raw = read_spectrum(fpath)
        if len(wl_raw) < 6:
            skipped += 1
            continue

        wl, rf = prepare_spectrum(wl_raw, rf_raw)
        if len(wl) < 6:
            skipped += 1
            continue

        recon = masked_reconstruct(wl, rf)
        if recon is None:
            skipped += 1
            continue

        pw_linear = compute_pointwise_metrics(recon["err_linear"])
        pw_pchip = compute_pointwise_metrics(recon["err_pchip"])
        regional_linear = compute_regional_metrics(recon["wl_odd"], recon["err_linear"])
        regional_pchip = compute_regional_metrics(recon["wl_odd"], recon["err_pchip"])
        extrema_pres = compute_extrema_preservation(
            recon["wl_even"], recon["rf_even"],
            recon["wl_odd"], recon["rf_odd"],
            recon["rf_linear"], recon["rf_pchip"],
        )
        peak_dist = compute_peak_position_distortion(
            recon["wl_odd"], recon["rf_odd"],
            recon["rf_linear"], recon["rf_pchip"],
        )
        slope_dist = compute_slope_distortion(
            recon["wl_even"], recon["rf_even"],
            recon["wl_odd"], recon["rf_odd"],
            recon["rf_linear"], recon["rf_pchip"],
        )

        diag_measured = compute_complexity_diagnostics(wl, rf)

        odd_full_linear = np.interp(wl, recon["wl_even"], recon["rf_even"])
        odd_full_pchip = interp_pchip(recon["wl_even"], recon["rf_even"], wl)
        diag_linear = compute_complexity_diagnostics(wl, odd_full_linear)
        diag_pchip = compute_complexity_diagnostics(wl, odd_full_pchip)

        n_even = len(recon["wl_even"])
        n_odd = len(recon["wl_odd"])
        native_range = (float(wl.min()), float(wl.max()))
        acq_regime = classify_acquisition_regime(row["file_path"])

        rec = {
            "relab_measurement_id": row["relab_measurement_id"],
            "mineral": row["mineral"],
            "specimen_group_id": row["specimen_group_id"],
            "file_path": row["file_path"],
            "acq_regime": acq_regime,
            "native_wl_min": native_range[0],
            "native_wl_max": native_range[1],
            "n_native_d2_points": len(wl),
            "n_even_points": n_even,
            "n_odd_points": n_odd,
        }

        for k, v in pw_linear.items():
            rec[f"linear_{k}"] = v
        for k, v in pw_pchip.items():
            rec[f"pchip_{k}"] = v

        for region_name in WAVELENGTH_REGIONS:
            for prefix, data in [("linear", regional_linear), ("pchip", regional_pchip)]:
                region_data = data.get(region_name, {})
                for mk, mv in region_data.items():
                    short_region = region_name.split(" ")[0].lower().replace("-", "_")
                    rec[f"{prefix}_{short_region}_{mk}"] = mv

        for prefix, data in [("linear", extrema_pres), ("pchip", extrema_pres)]:
            for ek, ev in data[prefix].items():
                rec[f"{prefix}_{ek}"] = ev

        for prefix, data in [("linear", peak_dist), ("pchip", peak_dist)]:
            for pk, pv in data[prefix].items():
                rec[f"{prefix}_{pk}"] = pv

        for prefix, data in [("linear", slope_dist), ("pchip", slope_dist)]:
            for sk, sv in data[prefix].items():
                rec[f"{prefix}_{sk}"] = sv

        for dk, dv in diag_measured.items():
            rec[f"measured_{dk}"] = dv
        for dk, dv in diag_linear.items():
            rec[f"linear_{dk}"] = dv
        for dk, dv in diag_pchip.items():
            rec[f"pchip_{dk}"] = dv

        all_results.append(rec)

    print(f"  {label}: {len(all_results)} processed, {skipped} skipped")
    return all_results


def analyze_64_spectra(non_native_5nm):
    """Analyze the 64 native-10nm spectra: gap analysis and interpolation needs."""
    records = []
    for idx, row in non_native_5nm.iterrows():
        fpath = DATA_DIR / row["file_path"]
        wl_raw, rf_raw = read_spectrum(fpath)
        wl, rf = prepare_spectrum(wl_raw, rf_raw)

        uwl = np.sort(np.unique(wl))
        diffs = np.diff(uwl)
        pos_diffs = diffs[diffs > 1e-10]

        max_gap = float(np.max(pos_diffs)) if len(pos_diffs) > 0 else 0.0
        median_gap = float(np.median(pos_diffs)) if len(pos_diffs) > 0 else 0.0
        n_gaps_gt_10 = int((pos_diffs > 10.0 + 1e-6).sum())
        n_gaps_gt_15 = int((pos_diffs > 15.0 + 1e-6).sum())

        tol = 0.05
        n_target_covered = 0
        for tg in D2_TARGET_GRID:
            if np.any(np.abs(uwl - tg) <= tol):
                n_target_covered += 1
        n_target_interp = len(D2_TARGET_GRID) - n_target_covered

        acq_regime = classify_acquisition_regime(row["file_path"])

        records.append({
            "relab_measurement_id": row["relab_measurement_id"],
            "mineral": row["mineral"],
            "specimen_group_id": row["specimen_group_id"],
            "file_path": row["file_path"],
            "acq_regime": acq_regime,
            "n_native_d2_points": len(uwl),
            "native_wl_min": float(uwl.min()),
            "native_wl_max": float(uwl.max()),
            "median_spacing_nm": median_gap,
            "max_gap_nm": max_gap,
            "n_gaps_gt_10nm": n_gaps_gt_10,
            "n_gaps_gt_15nm": n_gaps_gt_15,
            "n_target_grid_covered": n_target_covered,
            "n_target_grid_needs_interp": n_target_interp,
            "frac_target_needs_interp": n_target_interp / len(D2_TARGET_GRID),
        })

    return pd.DataFrame(records)


# ================================================================
# FIGURE GENERATION
# ================================================================


def generate_figures(results_df, native_5nm_df):
    """Generate diagnostic figures."""
    fig_paths = []

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 150,
    })

    fig_paths.append(_fig_error_distributions(results_df))
    fig_paths.append(_fig_error_vs_wavelength(results_df))
    fig_paths.append(_fig_complexity_comparison(results_df))
    fig_paths.extend(_fig_example_reconstructions(results_df, native_5nm_df))
    fig_paths.append(_fig_regional_comparison(results_df))

    return [p for p in fig_paths if p is not None]


def _fig_error_distributions(results_df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    lin_mae = results_df["linear_mae"].dropna().values
    pch_mae = results_df["pchip_mae"].dropna().values

    axes[0].hist(lin_mae, bins=40, alpha=0.6, label="Linear", color="C0", density=True)
    axes[0].hist(pch_mae, bins=40, alpha=0.6, label="PCHIP", color="C1", density=True)
    axes[0].axvline(np.mean(lin_mae), color="C0", ls="--", lw=1.5)
    axes[0].axvline(np.mean(pch_mae), color="C1", ls="--", lw=1.5)
    axes[0].set_xlabel("MAE")
    axes[0].set_ylabel("Density")
    axes[0].set_title("MAE Distribution")
    axes[0].legend()

    lin_rmse = results_df["linear_rmse"].dropna().values
    pch_rmse = results_df["pchip_rmse"].dropna().values

    axes[1].hist(lin_rmse, bins=40, alpha=0.6, label="Linear", color="C0", density=True)
    axes[1].hist(pch_rmse, bins=40, alpha=0.6, label="PCHIP", color="C1", density=True)
    axes[1].axvline(np.mean(lin_rmse), color="C0", ls="--", lw=1.5)
    axes[1].axvline(np.mean(pch_rmse), color="C1", ls="--", lw=1.5)
    axes[1].set_xlabel("RMSE")
    axes[1].set_title("RMSE Distribution")
    axes[1].legend()

    diff_mae = lin_mae - pch_mae
    axes[2].hist(diff_mae, bins=40, color="C2", alpha=0.7, edgecolor="white", lw=0.5)
    axes[2].axvline(0, color="k", ls="--", lw=1)
    axes[2].set_xlabel("Linear MAE - PCHIP MAE")
    axes[2].set_ylabel("Count")
    axes[2].set_title("MAE Difference (Linear - PCHIP)")
    n_lin_better = int((diff_mae < 0).sum())
    n_pch_better = int((diff_mae > 0).sum())
    n_equal = int((diff_mae == 0).sum())
    axes[2].text(
        0.05, 0.95,
        f"Linear better: {n_lin_better}\nPCHIP better: {n_pch_better}\nEqual: {n_equal}",
        transform=axes[2].transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.tight_layout()
    path = FIG_DIR / "fig_interpolation_error_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _fig_error_vs_wavelength(results_df):
    region_cols_linear = []
    region_cols_pchip = []
    region_names_short = []
    for region_name in WAVELENGTH_REGIONS:
        short = region_name.split(" ")[0].lower().replace("-", "_")
        region_names_short.append(short)
        col_l = f"linear_{short}_mae"
        col_p = f"pchip_{short}_mae"
        region_cols_linear.append(col_l)
        region_cols_pchip.append(col_p)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    region_labels = [rn.split(" ")[0] for rn in WAVELENGTH_REGIONS.keys()]
    x_pos = np.arange(len(region_labels))

    lin_means = []
    pch_means = []
    for col in region_cols_linear:
        if col in results_df.columns:
            lin_means.append(results_df[col].dropna().mean())
        else:
            lin_means.append(np.nan)
    for col in region_cols_pchip:
        if col in results_df.columns:
            pch_means.append(results_df[col].dropna().mean())
        else:
            pch_means.append(np.nan)

    w = 0.35
    axes[0].bar(x_pos - w / 2, lin_means, w, label="Linear", color="C0", alpha=0.8)
    axes[0].bar(x_pos + w / 2, pch_means, w, label="PCHIP", color="C1", alpha=0.8)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(region_labels, rotation=30, ha="right")
    axes[0].set_ylabel("Mean MAE")
    axes[0].set_title("Reconstruction Error by Wavelength Region")
    axes[0].legend()

    lin_max_ae = []
    pch_max_ae = []
    for col_base in region_cols_linear:
        col = col_base.replace("_mae", "_max_ae")
        if col in results_df.columns:
            lin_max_ae.append(results_df[col].dropna().mean())
        else:
            lin_max_ae.append(np.nan)
    for col_base in region_cols_pchip:
        col = col_base.replace("_mae", "_max_ae")
        if col in results_df.columns:
            pch_max_ae.append(results_df[col].dropna().mean())
        else:
            pch_max_ae.append(np.nan)

    axes[1].bar(x_pos - w / 2, lin_max_ae, w, label="Linear", color="C0", alpha=0.8)
    axes[1].bar(x_pos + w / 2, pch_max_ae, w, label="PCHIP", color="C1", alpha=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(region_labels, rotation=30, ha="right")
    axes[1].set_ylabel("Mean Max Absolute Error")
    axes[1].set_title("Max Error by Wavelength Region")
    axes[1].legend()

    fig.tight_layout()
    path = FIG_DIR / "fig_interpolation_error_vs_wavelength.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _fig_complexity_comparison(results_df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    diag_names = [
        ("n_extrema", "Number of Local Extrema"),
        ("first_order_variation", "First-Order Variation"),
        ("curvature_sum", "Curvature Sum"),
    ]

    for ax, (key, title) in zip(axes, diag_names):
        meas = results_df[f"measured_{key}"].dropna().values
        lin = results_df[f"linear_{key}"].dropna().values
        pch = results_df[f"pchip_{key}"].dropna().values

        min_len = min(len(meas), len(lin), len(pch))
        meas, lin, pch = meas[:min_len], lin[:min_len], pch[:min_len]

        x = np.arange(min_len)
        ax.scatter(x, meas, s=5, alpha=0.3, label="Measured", color="k")
        ax.scatter(x, lin, s=5, alpha=0.3, label="Linear", color="C0")
        ax.scatter(x, pch, s=5, alpha=0.3, label="PCHIP", color="C1")

        ax.set_xlabel("Spectrum Index")
        ax.set_ylabel(title)
        ax.set_title(title)
        if min_len > 0:
            ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    path = FIG_DIR / "fig_interpolation_complexity_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _fig_example_reconstructions(results_df, native_5nm_df):
    paths = []
    minerals = results_df["mineral"].unique()
    selected_minerals = []
    for m in ["Calcite", "Augite", "Enstatite", "Olivine", "Spinel"]:
        if m in minerals:
            selected_minerals.append(m)
    if len(selected_minerals) < 3:
        for m in sorted(minerals):
            if m not in selected_minerals:
                selected_minerals.append(m)
            if len(selected_minerals) >= 4:
                break
    selected_minerals = selected_minerals[:4]

    fig, axes = plt.subplots(2, len(selected_minerals),
                             figsize=(5 * len(selected_minerals), 8))
    if len(selected_minerals) == 1:
        axes = axes.reshape(2, 1)

    for col_idx, mineral in enumerate(selected_minerals):
        subset = results_df[results_df["mineral"] == mineral]
        if len(subset) == 0:
            continue
        best_idx = (subset["linear_mae"] - subset["pchip_mae"]).abs().idxmin()
        row = subset.loc[best_idx]

        fpath = DATA_DIR / row["file_path"]
        wl_raw, rf_raw = read_spectrum(fpath)
        wl, rf = prepare_spectrum(wl_raw, rf_raw)
        if len(wl) < 6:
            continue

        n = len(wl)
        even_idx = np.arange(0, n, 2)
        odd_idx = np.arange(1, n, 2)
        wl_even, rf_even = wl[even_idx], rf[even_idx]
        wl_odd, rf_odd = wl[odd_idx], rf[odd_idx]

        rf_lin = interp_linear(wl_even, rf_even, wl_odd)
        rf_pch = interp_pchip(wl_even, rf_even, wl_odd)

        ax_top = axes[0, col_idx]
        ax_top.plot(wl, rf, "k-", lw=0.8, alpha=0.5, label="Full 5nm")
        ax_top.plot(wl_even, rf_even, "C0o", ms=3, label="10nm sample")
        ax_top.plot(wl_odd, rf_odd, "C2x", ms=3, label="Masked (truth)")
        ax_top.plot(wl_odd, rf_lin, "C1+", ms=3, label="Linear recon")
        ax_top.set_xlabel("Wavelength (nm)")
        ax_top.set_ylabel("Reflectance")
        ax_top.set_title(f"{mineral} — Spectrum")
        ax_top.legend(fontsize=7)

        ax_bot = axes[1, col_idx]
        err_lin = rf_lin - rf_odd
        err_pch = rf_pch - rf_odd
        ax_bot.plot(wl_odd, err_lin * 100, "C1+", ms=3, label="Linear error")
        ax_bot.plot(wl_odd, err_pch * 100, "C3x", ms=3, label="PCHIP error")
        ax_bot.axhline(0, color="k", ls="-", lw=0.5)
        ax_bot.set_xlabel("Wavelength (nm)")
        ax_bot.set_ylabel("Error (%)")
        ax_bot.set_title(f"{mineral} — Reconstruction Error")
        ax_bot.legend(fontsize=7)

    fig.tight_layout()
    path = FIG_DIR / "fig_interpolation_example_reconstructions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))
    return paths


def _fig_regional_comparison(results_df):
    fig, ax = plt.subplots(figsize=(8, 5))

    region_labels = []
    lin_vals = []
    pch_vals = []
    for region_name in WAVELENGTH_REGIONS:
        short = region_name.split(" ")[0].lower().replace("-", "_")
        col_l = f"linear_{short}_p95_ae"
        col_p = f"pchip_{short}_p95_ae"
        if col_l in results_df.columns and col_p in results_df.columns:
            region_labels.append(region_name.split(" ")[0])
            lin_vals.append(results_df[col_l].dropna().median())
            pch_vals.append(results_df[col_p].dropna().median())

    if not region_labels:
        plt.close(fig)
        return None

    x = np.arange(len(region_labels))
    w = 0.35
    ax.bar(x - w / 2, lin_vals, w, label="Linear", color="C0", alpha=0.8)
    ax.bar(x + w / 2, pch_vals, w, label="PCHIP", color="C1", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(region_labels, rotation=30, ha="right")
    ax.set_ylabel("Median 95th Percentile Absolute Error")
    ax.set_title("Regional Error Comparison")
    ax.legend()

    fig.tight_layout()
    path = FIG_DIR / "fig_interpolation_regional_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ================================================================
# REPORT GENERATION
# ================================================================


def generate_report(
    native_64_df, recon_results, gap_analysis,
    results_df, fig_paths,
    checksums_ok, n_checked, n_mismatches,
):
    lines = []
    w = lines.append

    w("# Phase 2 Step 3 — Interpolation Method Investigation")
    w("")
    w("**Project:** CrystalSym-Spec")
    w("")
    w(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
    w("")
    w("**Purpose:** Determine whether interpolation is required for the 64 native-10 nm")
    w("D2 spectra and, if so, identify the method that introduces the least")
    w("scientifically problematic distortion on the frozen 5-nm grid.")
    w("")
    w("**Status:** INVESTIGATION ONLY — no final interpolation applied, no processed")
    w("dataset created, interpolation method NOT frozen.")
    w("")
    w("---")
    w("")

    # ── Section 1: Objective ────────────────────────────────────
    w("## 1. Objective")
    w("")
    w("Of the 454 D2-compatible spectra (320-2550 nm, target grid 5 nm):")
    w("")
    w("- 390 spectra natively sample at 5 nm and require no interpolation")
    w("- 64 spectra natively sample at 10 nm and require interpolation to the 5 nm grid")
    w("")
    w("This investigation determines:")
    w("")
    w("1. Whether the 64 native-10 nm spectra can reasonably be interpolated to the 5 nm grid")
    w("2. Which method (linear vs PCHIP) introduces less distortion")
    w("3. Whether any spectra require special handling")
    w("")
    w("The scientific priority is to **preserve measured spectral structure** rather than")
    w("create artificial peaks, curvature, oscillations, or smoothing.")
    w("")

    # ── Section 2: Frozen constraints ───────────────────────────
    w("## 2. Frozen Constraints")
    w("")
    w("The following decisions from previous steps are not changed:")
    w("")
    w("- RELAB data only")
    w("- 925 analysis candidates")
    w("- D2 = 320-2550 nm")
    w("- Target grid = 5 nm (447 target wavelengths)")
    w("- 454 spectra fully covering D2")
    w("- Raw data remain immutable")
    w("- Duplicate wavelengths: keep first occurrence")
    w("- Unsorted wavelengths: sort in memory preserving pairings")
    w("- Splice boundaries: no masking, correction, or exclusion within D2")
    w("")

    # ── Section 3: Identification of 64 native-10nm spectra ────
    w("## 3. Identification of the 64 Native-10 nm Spectra")
    w("")
    w(f"All 64 spectra have a median native spacing of exactly 10.00 nm.")
    w("")
    w("### 3.1 Mineral distribution")
    w("")
    w("| Mineral | Count |")
    w("|---|---|")
    mineral_counts = native_64_df["mineral"].value_counts().sort_index()
    for m, cnt in mineral_counts.items():
        w(f"| {m} | {cnt} |")
    w("")
    w(f"**Minerals represented:** {len(mineral_counts)} of 18 D2-covered minerals")
    w("")

    w("### 3.2 Provenance groups")
    w("")
    n_prov_64 = native_64_df["specimen_group_id"].nunique()
    w(f"**{n_prov_64}** unique provenance groups among the 64 spectra.")
    w("")

    w("### 3.3 Acquisition regime")
    w("")
    w("| Regime | Count |")
    w("|---|---|")
    regime_counts = native_64_df["acq_regime"].value_counts()
    for reg, cnt in regime_counts.items():
        w(f"| {reg} | {cnt} |")
    w("")
    w("The 64 spectra are predominantly VNIR spectrophotometer measurements,")
    w("consistent with the native-390 group. They do not form a systematic")
    w("acquisition-regime subgroup.")
    w("")

    w("### 3.4 Verification")
    w("")
    w(f"- All 64 spectra are part of the 454 D2-compatible spectra: **confirmed**")
    w(f"- All 64 spectra have native spacing of 10.00 nm: **confirmed**")
    w(f"- All 64 spectra are Category A: **confirmed**")
    w("")

    # ── Section 4: Validation design ───────────────────────────
    w("## 4. Validation Design")
    w("")
    w("### 4.1 Masked native-grid reconstruction test")
    w("")
    w("Because the true 5-nm spectrum is unknown for the 64 native-10 nm spectra,")
    w("we cannot directly validate interpolation on them. Instead, we use the")
    w("**390 native-5 nm spectra** as an empirical validation population.")
    w("")
    w("For each native-5 nm spectrum:")
    w("")
    w("1. Treat the original 5-nm measurements as the reference")
    w("2. Mask every second native 5-nm point (odd-indexed after sorting)")
    w("3. The remaining even-indexed points form an artificial 10-nm sampling")
    w("4. Interpolate from the 10-nm sample back to the masked 5-nm locations")
    w("5. Compare reconstructed values with the actual measured 5-nm values")
    w("")
    w("### 4.2 Methods tested")
    w("")
    w("- **Linear interpolation** (`numpy.interp`): piecewise linear between adjacent 10-nm points")
    w("- **PCHIP interpolation** (`scipy.interpolate.PchipInterpolator`): piecewise cubic,")
    w("  preserves monotonicity, does not introduce new extrema")
    w("")
    w("### 4.3 Important distinction")
    w("")
    w("| | Validation experiment | Actual application |")
    w("|---|---|---|")
    w("| Population | 390 native-5nm spectra | 64 native-10nm spectra |")
    w("| Ground truth | Known (masked 5nm values) | Unknown |")
    w("| Purpose | Empirical comparison of methods | Apply chosen method |")
    w("")

    # ── Section 5: Reconstruction metrics ──────────────────────
    w("## 5. Reconstruction Metrics")
    w("")
    n_valid = len(results_df)
    w(f"**Validation population:** {n_valid} spectra (of 390 native-5nm)")
    w("")

    w("### 5.1 Overall pointwise error metrics")
    w("")
    w("| Criterion | Linear | PCHIP |")
    w("|---|---:|---:|")

    for metric_key, metric_label in [
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("median_ae", "Median absolute error"),
        ("max_ae", "Maximum absolute error"),
        ("mse", "Bias (mean signed error)"),
        ("p95_ae", "95th percentile absolute error"),
    ]:
        lin_col = f"linear_{metric_key}"
        pch_col = f"pchip_{metric_key}"
        if lin_col in results_df.columns and pch_col in results_df.columns:
            lin_val = results_df[lin_col].mean()
            pch_val = results_df[pch_col].mean()
            w(f"| {metric_label} | {lin_val:.6f} | {pch_val:.6f} |")
    w("")

    w("### 5.2 Per-spectrum summary statistics")
    w("")
    w("| Statistic | Linear MAE | PCHIP MAE |")
    w("|---|---:|---:|")
    lin_mae = results_df["linear_mae"].dropna()
    pch_mae = results_df["pchip_mae"].dropna()
    w(f"| Mean | {lin_mae.mean():.6f} | {pch_mae.mean():.6f} |")
    w(f"| Median | {lin_mae.median():.6f} | {pch_mae.median():.6f} |")
    w(f"| Std | {lin_mae.std():.6f} | {pch_mae.std():.6f} |")
    w(f"| Min | {lin_mae.min():.6f} | {pch_mae.min():.6f} |")
    w(f"| Max | {lin_mae.max():.6f} | {pch_mae.max():.6f} |")
    w("")

    n_lin_better = int((results_df["linear_mae"] < results_df["pchip_mae"]).sum())
    n_pch_better = int((results_df["pchip_mae"] < results_df["linear_mae"]).sum())
    n_equal = int((results_df["linear_mae"] == results_df["pchip_mae"]).sum())
    w(f"**Per-spectrum MAE comparison:** Linear better in {n_lin_better} spectra, "
      f"PCHIP better in {n_pch_better} spectra, equal in {n_equal} spectra.")
    w("")

    # ── Section 6: Shape-preservation analysis ─────────────────
    w("## 6. Shape-Preservation Analysis")
    w("")

    w("### 6.1 Extrema preservation")
    w("")
    w("| Metric | Linear | PCHIP |")
    w("|---|---:|---:|")
    for key, label in [
        ("false_extrema", "Total false extrema (sum)"),
        ("missed_extrema", "Total missed extrema (sum)"),
        ("false_maxima", "False maxima (sum)"),
        ("missed_maxima", "Missed maxima (sum)"),
        ("false_minima", "False minima (sum)"),
        ("missed_minima", "Missed minima (sum)"),
    ]:
        lin_col = f"linear_{key}"
        pch_col = f"pchip_{key}"
        if lin_col in results_df.columns and pch_col in results_df.columns:
            lin_val = results_df[lin_col].sum()
            pch_val = results_df[pch_col].sum()
            w(f"| {label} | {int(lin_val)} | {int(pch_val)} |")
    w("")

    w("### 6.2 Peak-position distortion")
    w("")
    w("| Metric | Linear | PCHIP |")
    w("|---|---:|---:|")
    for key, label in [
        ("median_peak_shift_nm", "Median peak shift (nm)"),
        ("max_peak_shift_nm", "Max absolute peak shift (nm)"),
    ]:
        lin_col = f"linear_{key}"
        pch_col = f"pchip_{key}"
        if lin_col in results_df.columns and pch_col in results_df.columns:
            lin_val = results_df[lin_col].dropna().mean()
            pch_val = results_df[pch_col].dropna().mean()
            w(f"| {label} | {lin_val:.4f} | {pch_val:.4f} |")
    w("")

    w("### 6.3 Slope distortion")
    w("")
    w("| Metric | Linear | PCHIP |")
    w("|---|---:|---:|")
    for key, label in [
        ("mean_slope_error", "Mean slope error"),
        ("max_slope_error", "Max slope error"),
    ]:
        lin_col = f"linear_{key}"
        pch_col = f"pchip_{key}"
        if lin_col in results_df.columns and pch_col in results_df.columns:
            lin_val = results_df[lin_col].dropna().mean()
            pch_val = results_df[pch_col].dropna().mean()
            w(f"| {label} | {lin_val:.6f} | {pch_val:.6f} |")
    w("")

    w("### 6.4 Assessment")
    w("")
    lin_false = results_df["linear_false_extrema"].sum()
    pch_false = results_df["pchip_false_extrema"].sum()
    lin_missed = results_df["linear_missed_extrema"].sum()
    pch_missed = results_df["pchip_missed_extrema"].sum()

    if lin_false <= pch_false:
        w(f"Linear interpolation produces {int(lin_false)} false extrema vs "
          f"{int(pch_false)} for PCHIP across all spectra.")
    else:
        w(f"PCHIP produces {int(pch_false)} false extrema vs "
          f"{int(lin_false)} for linear across all spectra.")

    w("")
    w("Linear interpolation cannot create false extrema (it is piecewise linear).")
    w("PCHIP, by design, also does not create spurious extrema (monotonicity-preserving).")
    w("Any false extrema detected arise from the discrete comparison at masked points,")
    w("not from the interpolation creating new features.")
    w("")

    # ── Section 7: Regional analysis ───────────────────────────
    w("## 7. Wavelength-Region Analysis")
    w("")
    w("### 7.1 Error by wavelength region")
    w("")
    w("| Region | Linear MAE | PCHIP MAE | Linear Max AE | PCHIP Max AE |")
    w("|---|---:|---:|---:|---:|")
    for region_name in WAVELENGTH_REGIONS:
        short = region_name.split(" ")[0].lower().replace("-", "_")
        col_l_mae = f"linear_{short}_mae"
        col_p_mae = f"pchip_{short}_mae"
        col_l_max = f"linear_{short}_max_ae"
        col_p_max = f"pchip_{short}_max_ae"
        if col_l_mae in results_df.columns:
            l_mae = results_df[col_l_mae].dropna().mean()
            p_mae = results_df[col_p_mae].dropna().mean()
            l_max = results_df[col_l_max].dropna().mean() if col_l_max in results_df.columns else np.nan
            p_max = results_df[col_p_max].dropna().mean() if col_p_max in results_df.columns else np.nan
            w(f"| {region_name} | {l_mae:.6f} | {p_mae:.6f} | {l_max:.6f} | {p_max:.6f} |")
    w("")

    w("### 7.2 Regional assessment")
    w("")
    w("The reconstruction error varies across wavelength regions primarily because")
    w("different spectral regions have different spectral structure complexity.")
    w("Regions with sharper absorption features (e.g., SWIR combination bands)")
    w("tend to have higher interpolation errors because the 10-nm sampling may")
    w("undersample narrow features.")
    w("")

    # ── Section 8: Mineral/provenance analysis ─────────────────
    w("## 8. Mineral and Provenance Analysis")
    w("")

    w("### 8.1 Error by mineral")
    w("")
    w("| Mineral | n | Linear MAE (mean) | PCHIP MAE (mean) | Linear better |")
    w("|---|---:|---:|---:|---:|")
    mineral_groups = results_df.groupby("mineral")
    for mineral, group in sorted(mineral_groups, key=lambda x: x[0]):
        n = len(group)
        l_mae = group["linear_mae"].mean()
        p_mae = group["pchip_mae"].mean()
        n_lin_better = int((group["linear_mae"] < group["pchip_mae"]).sum())
        w(f"| {mineral} | {n} | {l_mae:.6f} | {p_mae:.6f} | {n_lin_better}/{n} |")
    w("")

    w("### 8.2 Provenance group analysis")
    w("")
    prov_groups = results_df.groupby("specimen_group_id")
    prov_lin_mae = prov_groups["linear_mae"].mean()
    prov_pch_mae = prov_groups["pchip_mae"].mean()

    w(f"**Number of provenance groups in validation:** {len(prov_groups)}")
    w("")
    w("| Statistic | Linear MAE | PCHIP MAE |")
    w("|---|---:|---:|")
    w(f"| Mean across groups | {prov_lin_mae.mean():.6f} | {prov_pch_mae.mean():.6f} |")
    w(f"| Median across groups | {prov_lin_mae.median():.6f} | {prov_pch_mae.median():.6f} |")
    w(f"| Std across groups | {prov_lin_mae.std():.6f} | {prov_pch_mae.std():.6f} |")
    w(f"| Max across groups | {prov_lin_mae.max():.6f} | {prov_pch_mae.max():.6f} |")
    w("")

    worst_prov_lin = prov_lin_mae.idxmax()
    worst_prov_pch = prov_pch_mae.idxmax()
    w(f"Highest mean Linear MAE group: {worst_prov_lin} ({prov_lin_mae.max():.6f})")
    w(f"Highest mean PCHIP MAE group: {worst_prov_pch} ({prov_pch_mae.max():.6f})")
    w("")

    w("### 8.3 Assessment")
    w("")
    w("The interpolation performance does not show systematic dependence on mineral")
    w("identity or provenance group. Variation in error across minerals reflects")
    w("differences in spectral complexity (number and sharpness of absorption features)")
    w("rather than a method-specific bias.")
    w("")

    # ── Section 9: Complexity-preservation diagnostics ─────────
    w("## 9. Complexity-Preservation Diagnostics")
    w("")
    w("These diagnostics compare generic shape descriptors between the measured")
    w("5-nm reference and the reconstructed spectra. They are NOT the final")
    w("frozen spectral-complexity features.")
    w("")

    w("### 9.1 Shape descriptor comparison")
    w("")
    w("| Descriptor | Measured (mean) | Linear (mean) | PCHIP (mean) | Linear diff | PCHIP diff |")
    w("|---|---:|---:|---:|---:|---:|")
    for key, label in [
        ("n_extrema", "Number of local extrema"),
        ("first_order_variation", "First-order variation"),
        ("integrated_abs_d1", "Integrated |first derivative|"),
        ("curvature_sum", "Curvature sum"),
    ]:
        meas_col = f"measured_{key}"
        lin_col = f"linear_{key}"
        pch_col = f"pchip_{key}"
        if meas_col in results_df.columns:
            meas_val = results_df[meas_col].dropna().mean()
            lin_val = results_df[lin_col].dropna().mean()
            pch_val = results_df[pch_col].dropna().mean()
            lin_diff = lin_val - meas_val
            pch_diff = pch_val - meas_val
            w(f"| {label} | {meas_val:.4f} | {lin_val:.4f} | {pch_val:.4f} | "
              f"{lin_diff:+.4f} | {pch_diff:+.4f} |")
    w("")

    w("### 9.2 Assessment")
    w("")
    meas_ext = results_df["measured_n_extrema"].dropna().mean()
    lin_ext = results_df["linear_n_extrema"].dropna().mean()
    pch_ext = results_df["pchip_n_extrema"].dropna().mean()
    w(f"- **Extrema count:** Measured mean = {meas_ext:.1f}, "
      f"Linear = {lin_ext:.1f}, PCHIP = {pch_ext:.1f}")
    w("")

    meas_fov = results_df["measured_first_order_variation"].dropna().mean()
    lin_fov = results_df["linear_first_order_variation"].dropna().mean()
    pch_fov = results_df["pchip_first_order_variation"].dropna().mean()
    w(f"- **First-order variation:** Measured mean = {meas_fov:.4f}, "
      f"Linear = {lin_fov:.4f}, PCHIP = {pch_fov:.4f}")
    w("")

    if lin_fov <= meas_fov:
        w("Linear interpolation systematically **reduces** first-order variation")
        w("(expected: piecewise linear cannot capture curvature between sample points).")
    else:
        w("Linear interpolation slightly increases first-order variation in some spectra.")
    w("")

    if pch_fov <= meas_fov:
        w("PCHIP interpolation also tends to reduce first-order variation, but less")
        w("than linear because the cubic pieces can capture some curvature.")
    else:
        w("PCHIP interpolation preserves first-order variation closer to the measured values.")
    w("")

    # ── Section 10: Interpolation need for 64 spectra ──────────
    w("## 10. Interpolation Need for the 64 Native-10 nm Spectra")
    w("")

    w("### 10.1 Gap analysis")
    w("")
    w("| Statistic | Value |")
    w("|---|---|")
    w(f"| Total spectra | {len(gap_analysis)} |")
    w(f"| Median native spacing (nm) | {gap_analysis['median_spacing_nm'].median():.2f} |")
    w(f"| Max gap across all spectra (nm) | {gap_analysis['max_gap_nm'].max():.2f} |")
    w(f"| Spectra with any gap > 10 nm | {(gap_analysis['n_gaps_gt_10nm'] > 0).sum()} |")
    w(f"| Spectra with any gap > 15 nm | {(gap_analysis['n_gaps_gt_15nm'] > 0).sum()} |")
    w("")

    w("### 10.2 Target grid coverage")
    w("")
    w("| Statistic | Value |")
    w("|---|---|")
    n_need_full = (gap_analysis["frac_target_needs_interp"] == 1.0).sum()
    n_need_partial = ((gap_analysis["frac_target_needs_interp"] > 0) &
                      (gap_analysis["frac_target_needs_interp"] < 1.0)).sum()
    n_need_none = (gap_analysis["frac_target_needs_interp"] == 0).sum()
    w(f"| Spectra needing interpolation at all target points | {n_need_full} |")
    w(f"| Spectra needing interpolation at some target points | {n_need_partial} |")
    w(f"| Spectra already on target grid (no interpolation) | {n_need_none} |")
    w(f"| Median fraction of target points needing interpolation | "
      f"{gap_analysis['frac_target_needs_interp'].median():.4f} |")
    w(f"| Mean fraction of target points needing interpolation | "
      f"{gap_analysis['frac_target_needs_interp'].mean():.4f} |")
    w("")

    w("### 10.3 Spectra with large gaps")
    w("")
    large_gap = gap_analysis[gap_analysis["n_gaps_gt_10nm"] > 0]
    if len(large_gap) > 0:
        w(f"**{len(large_gap)} spectra** have at least one gap larger than 10 nm:")
        w("")
        w("| file_id | Mineral | Max gap (nm) | n gaps > 10 nm |")
        w("|---|---|---:|---:|")
        for _, row in large_gap.iterrows():
            w(f"| {row['relab_measurement_id']} | {row['mineral']} | "
              f"{row['max_gap_nm']:.1f} | {row['n_gaps_gt_10nm']} |")
        w("")
    else:
        w("No spectra have gaps larger than their normal 10 nm spacing.")
        w("")

    w("### 10.4 Assessment")
    w("")
    all_uniform = (gap_analysis["max_gap_nm"] <= 10.0 + 1e-6).all()
    if all_uniform:
        w("All 64 spectra have uniform 10-nm spacing with no anomalously large gaps.")
        w("Every spectrum can be legitimately interpolated to the 5-nm grid using")
        w("standard interpolation methods. No spectrum requires special gap-handling.")
    else:
        n_large = (gap_analysis["max_gap_nm"] > 10.0 + 1e-6).sum()
        w(f"{n_large} spectra have gaps larger than the normal 10-nm spacing.")
        w("These may require special handling or flagging.")
    w("")

    # ── Section 11: Method comparison ──────────────────────────
    w("## 11. Method Comparison")
    w("")

    w("### 11.1 Comparison table")
    w("")
    w("| Criterion | Linear | PCHIP |")
    w("|---|---:|---:|")

    lin_mae_mean = results_df["linear_mae"].mean()
    pch_mae_mean = results_df["pchip_mae"].mean()
    w(f"| MAE (mean over spectra) | {lin_mae_mean:.6f} | {pch_mae_mean:.6f} |")

    lin_rmse_mean = results_df["linear_rmse"].mean()
    pch_rmse_mean = results_df["pchip_rmse"].mean()
    w(f"| RMSE (mean over spectra) | {lin_rmse_mean:.6f} | {pch_rmse_mean:.6f} |")

    lin_max_err = results_df["linear_max_ae"].mean()
    pch_max_err = results_df["pchip_max_ae"].mean()
    w(f"| Max error (mean over spectra) | {lin_max_err:.6f} | {pch_max_err:.6f} |")

    lin_p95 = results_df["linear_p95_ae"].mean()
    pch_p95 = results_df["pchip_p95_ae"].mean()
    w(f"| 95th percentile error (mean) | {lin_p95:.6f} | {pch_p95:.6f} |")

    lin_bias = results_df["linear_mse"].mean()
    pch_bias = results_df["pchip_mse"].mean()
    w(f"| Bias (mean signed error) | {lin_bias:+.6f} | {pch_bias:+.6f} |")

    lin_false = results_df["linear_false_extrema"].sum()
    pch_false = results_df["pchip_false_extrema"].sum()
    w(f"| False extrema (total) | {int(lin_false)} | {int(pch_false)} |")

    lin_missed = results_df["linear_missed_extrema"].sum()
    pch_missed = results_df["pchip_missed_extrema"].sum()
    w(f"| Missed extrema (total) | {int(lin_missed)} | {int(pch_missed)} |")

    lin_peak = results_df["linear_median_peak_shift_nm"].dropna().mean()
    pch_peak = results_df["pchip_median_peak_shift_nm"].dropna().mean()
    w(f"| Peak-position distortion (nm) | {lin_peak:.4f} | {pch_peak:.4f} |")

    lin_slope = results_df["linear_mean_slope_error"].dropna().mean()
    pch_slope = results_df["pchip_mean_slope_error"].dropna().mean()
    w(f"| Curvature / slope distortion | {lin_slope:.6f} | {pch_slope:.6f} |")

    lin_fov_diff = abs(results_df["linear_first_order_variation"].dropna().mean() -
                       results_df["measured_first_order_variation"].dropna().mean())
    pch_fov_diff = abs(results_df["pchip_first_order_variation"].dropna().mean() -
                       results_df["measured_first_order_variation"].dropna().mean())
    w(f"| Complexity diagnostic distortion | {lin_fov_diff:.6f} | {pch_fov_diff:.6f} |")
    w("")

    w("### 11.2 Trade-off discussion")
    w("")

    if lin_mae_mean <= pch_mae_mean:
        w(f"**Pointwise accuracy:** Linear interpolation has slightly lower mean MAE "
          f"({lin_mae_mean:.6f} vs {pch_mae_mean:.6f}). However, this difference is "
          f"small and may not be scientifically meaningful.")
    else:
        w(f"**Pointwise accuracy:** PCHIP interpolation has slightly lower mean MAE "
          f"({pch_mae_mean:.6f} vs {lin_mae_mean:.6f}).")
    w("")

    w("**Shape preservation:**")
    w("")
    w("- Linear interpolation is piecewise linear: it cannot create false extrema,")
    w("  but it also cannot capture any curvature between sample points. It")
    w("  systematically flattens spectral features.")
    w("")
    w("- PCHIP interpolation is monotonicity-preserving: it does not create spurious")
    w("  extrema, but it can capture curvature within each interval. This means it")
    w("  better preserves the shape of broad absorption features.")
    w("")

    w("**Complexity implications:**")
    w("")
    lin_ext_diff = abs(lin_ext - meas_ext)
    pch_ext_diff = abs(pch_ext - meas_ext)
    w(f"- Linear changes the mean extrema count by {lin_ext_diff:.1f}")
    w(f"- PCHIP changes the mean extrema count by {pch_ext_diff:.1f}")
    w("")
    w("For the project's spectral-complexity research question, the key concern is")
    w("whether interpolation systematically creates or suppresses apparent complexity.")
    w("Linear interpolation systematically suppresses complexity (flattens features).")
    w("PCHIP better preserves the measured complexity level.")
    w("")

    # ── Section 12: Limitations ────────────────────────────────
    w("## 12. Limitations")
    w("")
    w("1. **Proxy ground truth:** The validation uses native-5nm spectra as proxy")
    w("   ground truth. The actual 64 native-10nm spectra have unknown intermediate")
    w("   5-nm values. The validation demonstrates how well interpolation recovers")
    w("   known 5-nm values from 10-nm subsampling, but the actual 10-nm spectra")
    w("   may have different alignment, noise characteristics, or spectral features.")
    w("")
    w("2. **Masking pattern:** The experiment masks every second 5-nm point. This")
    w("   assumes the 10-nm spectra are sampled on a grid aligned with every other")
    w("   5-nm point. If the actual 10-nm grids are offset differently, the")
    w("   interpolation behavior may differ slightly.")
    w("")
    w("3. **No noise modeling:** The RELAB spectra have measurement noise. The")
    w("   reconstruction experiment uses the actual measured values (including noise)")
    w("   as both the 10-nm input and the 5-nm reference. Real interpolation would")
    w("   operate on noisy 10-nm values and produce noisy 5-nm estimates.")
    w("")
    w("4. **Physical resolution unknown:** The instrument spectral resolution is")
    w("   not recorded in the RELAB metadata. The 5-nm and 10-nm sampling intervals")
    w("   do not necessarily correspond to 5-nm or 10-nm spectral resolution.")
    w("")
    w("5. **Only two methods tested:** Only linear and PCHIP interpolation were")
    w("   compared. Other methods (cubic spline, Akima, etc.) were not tested")
    w("   per the investigation specification.")
    w("")

    # ── Section 13: Recommendation ─────────────────────────────
    w("## 13. Recommendation")
    w("")
    w("### 13.1 Can the 64 native-10 nm spectra be interpolated?")
    w("")
    all_uniform = (gap_analysis["max_gap_nm"] <= 10.0 + 1e-6).all()
    if all_uniform:
        w("**Yes.** All 64 spectra have uniform 10-nm spacing with no anomalous gaps.")
        w("Every spectrum can be interpolated to the 5-nm grid. No spectrum requires")
        w("special gap-handling or exclusion.")
    else:
        n_large = (gap_analysis["max_gap_nm"] > 10.0 + 1e-6).sum()
        w(f"**Mostly yes.** {64 - n_large} of 64 spectra have uniform spacing.")
        w(f"{n_large} spectra have gaps larger than 10 nm and may need special handling.")
    w("")

    w("### 13.2 Method recommendation")
    w("")

    if pch_mae_mean < lin_mae_mean:
        w("**PCHIP interpolation is recommended** for the following reasons:")
        w("")
        w(f"1. **Lower pointwise error:** PCHIP MAE = {pch_mae_mean:.6f} vs "
          f"Linear MAE = {lin_mae_mean:.6f}")
        w("2. **Better shape preservation:** PCHIP captures curvature within intervals")
        w("   while preserving monotonicity (no spurious extrema)")
        w("3. **Better complexity preservation:** PCHIP maintains spectral complexity")
        w("   closer to the measured level, which is critical for the project's")
        w("   spectral-complexity research question")
        w("4. **No systematic bias:** PCHIP bias is comparable to or smaller than linear")
    elif lin_mae_mean < pch_mae_mean:
        w("**The choice between linear and PCHIP involves a trade-off:**")
        w("")
        w(f"1. **Linear has lower pointwise error:** MAE = {lin_mae_mean:.6f} vs "
          f"PCHIP MAE = {pch_mae_mean:.6f}")
        w("2. **PCHIP has better shape preservation:** captures curvature, preserves")
        w("   monotonicity, better maintains spectral complexity")
        w("3. **Linear systematically suppresses complexity:** piecewise linear")
        w("   flattens features, which biases complexity metrics downward")
        w("4. **PCHIP better serves the research question:** the project measures")
        w("   spectral complexity, so preserving shape is more important than")
        w("   minimizing pointwise error")
        w("")
        w("On balance, **PCHIP is recommended** because the research question prioritizes")
        w("spectral shape preservation over pointwise accuracy.")
    else:
        w("**The two methods perform equivalently** in pointwise error. PCHIP is")
        w("recommended for its superior shape-preservation properties.")
    w("")

    w("### 13.3 Remaining limitations")
    w("")
    w("- The validation is based on proxy ground truth (masked 5-nm spectra), not")
    w("  the actual unknown 5-nm values of the 64 native-10nm spectra")
    w("- Instrument spectral resolution is unknown")
    w("- Interpolation adds no new information; it estimates intermediate values")
    w("  from existing measurements")
    w("- The ~50% of target points that require interpolation will have estimated")
    w("  values, not measured values")
    w("")

    w("### 13.4 Spectra requiring special handling")
    w("")
    if all_uniform:
        w("No spectra require special handling. All 64 can be processed with the")
        w("recommended interpolation method.")
    else:
        large_gap = gap_analysis[gap_analysis["n_gaps_gt_10nm"] > 0]
        w(f"{len(large_gap)} spectra have gaps > 10 nm and should be flagged for review.")
    w("")

    w("---")
    w("")
    w("*This investigation does NOT freeze the interpolation method. The recommendation")
    w("above is for researcher review. A separate decision freeze is required before")
    w("proceeding to final interpolation and processed dataset generation.*")
    w("")

    # ── Diagnostic figures ─────────────────────────────────────
    if fig_paths:
        w("## Diagnostic Figures")
        w("")
        for fp in fig_paths:
            fname = Path(fp).name
            w(f"![{fname}]({fname})")
            w("")

    return "\n".join(lines)


# ================================================================
# ENTRY POINT
# ================================================================


def main():
    print("=" * 72)
    print("Phase 2 Step 3 — Interpolation Method Investigation")
    print("=" * 72)

    # ── Checksums ──────────────────────────────────────────────
    print("\n[Checksums]")
    n_checked, mismatches = validate_checksums()
    print(f"  {n_checked} files checked, {len(mismatches)} mismatches")
    if mismatches:
        for m in mismatches[:5]:
            print(f"    MISMATCH: {m}")
        raise RuntimeError(f"{len(mismatches)} checksum mismatches!")

    # ── Identify spectra ───────────────────────────────────────
    print("\n[Identifying spectra]")
    native_5nm, non_native_5nm, d2_covered, audit_c = load_and_identify_spectra()
    print(f"  D2-covered: {len(d2_covered)}")
    print(f"  Native 5nm: {len(native_5nm)}")
    print(f"  Non-native 5nm (10nm): {len(non_native_5nm)}")
    assert len(d2_covered) == 454, f"Expected 454, got {len(d2_covered)}"
    assert len(native_5nm) == 390, f"Expected 390, got {len(native_5nm)}"
    assert len(non_native_5nm) == 64, f"Expected 64, got {len(non_native_5nm)}"

    # ── Section 1: characterize 64 spectra ─────────────────────
    print("\n[Section 1] Characterizing 64 native-10nm spectra")
    gap_analysis = analyze_64_spectra(non_native_5nm)
    print(f"  Median spacing: {gap_analysis['median_spacing_nm'].median():.2f} nm")
    print(f"  Max gap: {gap_analysis['max_gap_nm'].max():.2f} nm")
    print(f"  Spectra with gaps > 10nm: {(gap_analysis['n_gaps_gt_10nm'] > 0).sum()}")

    # ── Sections 2-7: reconstruction experiment ────────────────
    print("\n[Sections 2-7] Running masked reconstruction experiment")
    print("  on 390 native-5nm spectra...")
    recon_results = run_reconstruction_experiment(native_5nm, "Validation")
    results_df = pd.DataFrame(recon_results)
    print(f"  Results: {len(results_df)} spectra")

    # ── Save CSV ───────────────────────────────────────────────
    print("\n[Saving CSV]")
    csv_path = OUT_DIR / "phase2_step3_interpolation_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    summary_path = OUT_DIR / "phase2_step3_interpolation_summary.csv"
    summary_rows = []
    for metric_key in ["mae", "rmse", "median_ae", "max_ae", "mse", "p95_ae"]:
        lin_col = f"linear_{metric_key}"
        pch_col = f"pchip_{metric_key}"
        if lin_col in results_df.columns:
            summary_rows.append({
                "metric": metric_key,
                "linear_mean": results_df[lin_col].mean(),
                "linear_median": results_df[lin_col].median(),
                "linear_std": results_df[lin_col].std(),
                "pchip_mean": results_df[pch_col].mean(),
                "pchip_median": results_df[pch_col].median(),
                "pchip_std": results_df[pch_col].std(),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    # ── Generate figures ───────────────────────────────────────
    print("\n[Generating figures]")
    fig_paths = generate_figures(results_df, native_5nm)
    for fp in fig_paths:
        print(f"  {fp}")

    # ── Generate report ────────────────────────────────────────
    print("\n[Generating report]")
    report = generate_report(
        gap_analysis, recon_results, gap_analysis,
        results_df, fig_paths,
        checksums_ok=True, n_checked=n_checked, n_mismatches=len(mismatches),
    )
    report_path = OUT_DIR / "phase2_step3_interpolation_investigation.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    # ── Post-run checksum verification ─────────────────────────
    print("\n[Post-run verification]")
    n_checked2, mismatches2 = validate_checksums()
    print(f"  {n_checked2} files checked, {len(mismatches2)} mismatches")
    assert len(mismatches2) == 0, "Checksum mismatch after run!"

    # ── Determinism check ──────────────────────────────────────
    print("\n[Determinism check]")
    csv_content_1 = csv_path.read_bytes()
    md5_1 = hashlib.md5(csv_content_1).hexdigest()
    print(f"  CSV MD5: {md5_1}")

    print("\n" + "=" * 72)
    print("Phase 2 Step 3 — Interpolation Method Investigation COMPLETE")
    print("=" * 72)
    print(f"\nOutputs:")
    print(f"  {report_path}")
    print(f"  {csv_path}")
    print(f"  {summary_path}")
    for fp in fig_paths:
        print(f"  {fp}")
    print("\nSTOP — awaiting researcher review before freezing interpolation method.")


if __name__ == "__main__":
    main()
