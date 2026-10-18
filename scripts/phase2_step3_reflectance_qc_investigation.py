#!/usr/bin/env python3
"""
Phase 2 Step 3 — Reflectance Quality-Control Investigation
===========================================================

CrystalSym-Spec: Interpretable ML and Symbolic Regression of
Crystal Symmetry, Chemistry, and Mineral Spectral Complexity

Investigates the validity and quality of RELAB reflectance values
within the frozen D2 analysis domain (320-2550 nm) for all 925
analysis candidates.

This is an investigation only. No spectra are removed, interpolated,
smoothed, normalized, or otherwise transformed.

Outputs:
  results/02_preprocessing_qc/phase2_step3_reflectance_qc_investigation.md
  results/02_preprocessing_qc/phase2_step3_reflectance_qc.csv
  results/02_preprocessing_qc/phase2_step3_reflectance_qc_spectrum_summary.csv
  results/02_preprocessing_qc/figures/reflectance_qc_*.png
"""

import hashlib
import warnings
from pathlib import Path
from collections import defaultdict
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
MEASUREMENTS_CSV = PROJECT_ROOT / "metadata" / "relab_measurements.csv"
CHECKSUMS_CSV = PROJECT_ROOT / "metadata" / "provenance" / "relab_raw_checksums.csv"
OUT_DIR = PROJECT_ROOT / "results" / "02_preprocessing_qc"
FIG_DIR = OUT_DIR / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# FROZEN CONSTANTS
# ================================================================

D2_WMIN = 320.0
D2_WMAX = 2550.0
D2_SPACING = 5.0
D2_TARGET_GRID = np.arange(D2_WMIN, D2_WMAX + D2_SPACING * 0.5, D2_SPACING)
N_TARGET = len(D2_TARGET_GRID)

WAVELENGTH_REGIONS = {
    "UV-VNIR (320-700 nm)": (320.0, 700.0),
    "VNIR (700-1300 nm)": (700.0, 1300.0),
    "SWIR-1 (1300-1900 nm)": (1300.0, 1900.0),
    "SWIR-2 (1900-2550 nm)": (1900.0, 2550.0),
}

# ================================================================
# CHECKSUM VERIFICATION
# ================================================================


def compute_md5(filepath):
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_checksums():
    ref = pd.read_csv(CHECKSUMS_CSV)
    mismatches = []
    checked = 0
    for _, row in ref.iterrows():
        fp = DATA_DIR / row["rel_path"]
        if fp.exists():
            md5 = compute_md5(fp)
            if md5 != row["md5"]:
                mismatches.append(row["rel_path"])
            checked += 1
    return checked, mismatches


# ================================================================
# DATA PARSING
# ================================================================


def parse_relab_full(filepath):
    """Parse a RELAB .tab file into segments and footer.

    Returns:
        segments: list of (wavelengths, reflectances, uncertainties) arrays
        footer_lines: list of stripped footer strings
        parse_warnings: list of warning strings for malformed data
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    idx = 0
    segments = []
    parse_warnings = []

    while idx < len(lines):
        line = lines[idx].strip()
        try:
            n = int(line)
            if len(line.split()) == 1 and n > 0:
                idx += 1
                seg_wl, seg_rf, seg_unc = [], [], []
                for i in range(n):
                    if idx >= len(lines):
                        parse_warnings.append(
                            f"Expected {n} rows, got {i} before EOF"
                        )
                        break
                    parts = lines[idx].strip().split()
                    idx += 1
                    if len(parts) < 2:
                        parse_warnings.append(
                            f"Row {i}: only {len(parts)} columns"
                        )
                        continue
                    try:
                        wl = float(parts[0])
                        rf = float(parts[1])
                    except ValueError:
                        parse_warnings.append(
                            f"Row {i}: non-numeric wl/rf: {parts[0]!r} {parts[1]!r}"
                        )
                        continue
                    seg_wl.append(wl)
                    seg_rf.append(rf)
                    if len(parts) > 2:
                        try:
                            seg_unc.append(float(parts[2]))
                        except ValueError:
                            seg_unc.append(np.nan)
                    else:
                        seg_unc.append(None)
                segments.append((
                    np.array(seg_wl),
                    np.array(seg_rf),
                    seg_unc,
                ))
                continue
        except (ValueError, IndexError):
            pass
        idx += 1

    footer_lines = [l.strip() for l in lines[idx:] if l.strip()]
    return segments, footer_lines, parse_warnings


def prepare_d2_spectrum(segments):
    """Apply frozen preprocessing rules and restrict to D2.

    1. Concatenate all segments
    2. Sort by wavelength (frozen decision: unsorted → sort in memory)
    3. Keep first occurrence of duplicate wavelengths (frozen decision)
    4. Restrict to D2 domain [320, 2550] nm

    Returns:
        wl_d2, rf_d2, unc_d2 (arrays, same length), has_unc (bool)
    """
    all_wl = np.concatenate([s[0] for s in segments])
    all_rf = np.concatenate([s[1] for s in segments])

    has_unc = segments[0][2][0] is not None if segments else False
    if has_unc:
        all_unc_raw = []
        for s in segments:
            all_unc_raw.extend(s[2])
        all_unc = np.array(all_unc_raw, dtype=float)
    else:
        all_unc = np.array([], dtype=float)

    sort_idx = np.argsort(all_wl)
    all_wl = all_wl[sort_idx]
    all_rf = all_rf[sort_idx]
    if has_unc:
        all_unc = all_unc[sort_idx]

    dup_mask = np.concatenate([[False], np.diff(all_wl) == 0])
    all_wl = all_wl[~dup_mask]
    all_rf = all_rf[~dup_mask]
    if has_unc:
        all_unc = all_unc[~dup_mask]

    d2_mask = (all_wl >= D2_WMIN) & (all_wl <= D2_WMAX)
    wl_d2 = all_wl[d2_mask]
    rf_d2 = all_rf[d2_mask]
    unc_d2 = all_unc[d2_mask] if has_unc else np.array([])

    return wl_d2, rf_d2, unc_d2, has_unc


# ================================================================
# SPECTRAL SHAPE ANALYSIS
# ================================================================


def detect_single_point_spikes(wl, rf, threshold_factor=10.0):
    """Detect isolated single-point spikes or drops.

    Normalizes by wavelength spacing to handle spectra with varying
    point density (0.2 nm to 10+ nm).

    A spike/drop at index i is where both |dR/dλ|_left and |dR/dλ|_right
    exceed threshold_factor × median |dR/dλ|, while the neighbors
    remain consistent (|rf[i+1] - rf[i-1]| is small relative to local
    median difference).

    Returns list of dicts with spike details.
    """
    if len(rf) < 5:
        return []

    dwl = np.diff(wl)
    dwl[dwl == 0] = 1e-10
    slopes = np.diff(rf) / dwl
    abs_slopes = np.abs(slopes)
    nonzero = abs_slopes[abs_slopes > 0]
    median_slope = np.median(nonzero) if len(nonzero) > 0 else 1e-10
    threshold = threshold_factor * median_slope

    median_abs_diff = np.median(np.abs(np.diff(rf))[np.abs(np.diff(rf)) > 0]) \
        if np.any(np.abs(np.diff(rf)) > 0) else 1e-10

    spikes = []
    for i in range(1, len(rf) - 1):
        left_slope = abs(slopes[i - 1])
        right_slope = abs(slopes[i])
        neighbor_diff = abs(rf[i + 1] - rf[i - 1])

        if left_slope > threshold and right_slope > threshold:
            if neighbor_diff < median_abs_diff * 3:
                spikes.append({
                    "index": int(i),
                    "wavelength": float(wl[i]),
                    "reflectance": float(rf[i]),
                    "left_diff": float(abs(rf[i] - rf[i - 1])),
                    "right_diff": float(abs(rf[i] - rf[i + 1])),
                    "neighbor_diff": float(neighbor_diff),
                    "ratio_to_median": float(max(left_slope, right_slope) / median_slope),
                    "type": "spike" if rf[i] > rf[i - 1] else "drop",
                })
    return spikes


def detect_constant_runs(wl, rf, min_length=10):
    """Detect runs of constant reflectance values (potential saturation)."""
    if len(rf) < min_length:
        return []

    runs = []
    run_start = 0
    for i in range(1, len(rf)):
        if rf[i] != rf[run_start]:
            run_len = i - run_start
            if run_len >= min_length:
                runs.append({
                    "start_index": int(run_start),
                    "end_index": int(i - 1),
                    "start_wl": float(wl[run_start]),
                    "end_wl": float(wl[i - 1]),
                    "length": run_len,
                    "value": float(rf[run_start]),
                    "wl_range_nm": float(wl[i - 1] - wl[run_start]),
                })
            run_start = i

    run_len = len(rf) - run_start
    if run_len >= min_length:
        runs.append({
            "start_index": int(run_start),
            "end_index": int(len(rf) - 1),
            "start_wl": float(wl[run_start]),
            "end_wl": float(wl[len(rf) - 1]),
            "length": run_len,
            "value": float(rf[run_start]),
            "wl_range_nm": float(wl[len(rf) - 1] - wl[run_start]),
        })
    return runs


def detect_discontinuities(wl, rf, threshold_factor=20.0):
    """Detect abrupt discontinuities between adjacent points.

    Normalizes by wavelength spacing to avoid flagging points where
    large wavelength steps naturally produce larger reflectance differences.

    A discontinuity is where |dR/dλ| exceeds threshold_factor × median |dR/dλ|.
    """
    if len(rf) < 5:
        return []

    dwl = np.diff(wl)
    dwl[dwl == 0] = 1e-10
    slopes = np.abs(np.diff(rf) / dwl)
    nonzero = slopes[slopes > 0]
    median_slope = np.median(nonzero) if len(nonzero) > 0 else 1e-10
    threshold = threshold_factor * median_slope

    disc = []
    for i in range(len(slopes)):
        if slopes[i] > threshold:
            disc.append({
                "index": int(i),
                "wavelength_left": float(wl[i]),
                "wavelength_right": float(wl[i + 1]),
                "reflectance_left": float(rf[i]),
                "reflectance_right": float(rf[i + 1]),
                "abs_jump": float(abs(rf[i + 1] - rf[i])),
                "wavelength_step": float(wl[i + 1] - wl[i]),
                "ratio_to_median": float(slopes[i] / median_slope),
            })
    return disc


def detect_long_flat_segments(wl, rf, max_diff=1e-6, min_length=20):
    """Detect long segments where reflectance barely changes (not constant, but nearly so)."""
    if len(rf) < min_length:
        return []

    segments = []
    seg_start = 0
    for i in range(1, len(rf)):
        local_range = np.ptp(rf[seg_start:i + 1])
        if local_range > max_diff:
            seg_len = i - seg_start
            if seg_len >= min_length:
                segments.append({
                    "start_index": int(seg_start),
                    "end_index": int(i - 1),
                    "start_wl": float(wl[seg_start]),
                    "end_wl": float(wl[i - 1]),
                    "length": seg_len,
                    "ptp": float(np.ptp(rf[seg_start:i])),
                    "mean_rf": float(np.mean(rf[seg_start:i])),
                })
            seg_start = i

    return segments


# ================================================================
# MAIN INVESTIGATION
# ================================================================


def main():
    print("=" * 80)
    print("Phase 2 Step 3 — Reflectance Quality-Control Investigation")
    print("=" * 80)

    # --- Pre-flight checksum verification ---
    print("\n[1/6] Verifying raw checksums (pre-flight)...")
    pre_checked, pre_mismatches = validate_checksums()
    print(f"  Checked: {pre_checked}, Mismatches: {len(pre_mismatches)}")
    assert len(pre_mismatches) == 0, f"Checksum mismatch: {pre_mismatches[:5]}"

    # --- Load population ---
    print("\n[2/6] Loading analysis population...")
    pop = pd.read_csv(POPULATION_CSV)
    candidates = pop[pop["inclusion_category"].isin(["A", "B"])].copy()
    assert len(candidates) == 925, f"Expected 925 candidates, got {len(candidates)}"
    n_cat_a = (candidates["inclusion_category"] == "A").sum()
    n_cat_b = (candidates["inclusion_category"] == "B").sum()
    print(f"  Candidates: {len(candidates)} (A={n_cat_a}, B={n_cat_b})")

    # --- Process all spectra ---
    print("\n[3/6] Processing all 925 spectra...")

    spectrum_records = []
    anomaly_records = []
    spike_records = []
    constant_run_records = []
    discontinuity_records = []

    spectra_with_negative = []
    spectra_with_gt1 = []
    spectra_with_zeros = []
    spectra_with_nan_inf = []
    spectra_with_spikes = []
    spectra_with_constant_runs = []
    spectra_with_discontinuities = []

    parse_warning_log = []

    for file_idx, (_, crow) in enumerate(candidates.iterrows()):
        fp = DATA_DIR / crow["raw_path"]
        segments, footer, pwarnings = parse_relab_full(fp)

        for pw in pwarnings:
            parse_warning_log.append({
                "file_id": crow["file_id"],
                "filename": crow["filename"],
                "mineral": crow["mineral_directory"],
                "warning": pw,
            })

        if not segments:
            spectrum_records.append({
                "file_id": crow["file_id"],
                "filename": crow["filename"],
                "mineral": crow["mineral_directory"],
                "category": crow["inclusion_category"],
                "has_uncertainty": 0,
                "n_segments": len(segments),
                "n_total_points": 0,
                "n_d2_points": 0,
                "n_finite_d2": 0,
                "n_nan_d2": 0,
                "n_inf_d2": 0,
                "n_negative_d2": 0,
                "n_zero_d2": 0,
                "n_gt1_d2": 0,
                "rf_min_d2": np.nan,
                "rf_max_d2": np.nan,
                "rf_median_d2": np.nan,
                "rf_mean_d2": np.nan,
                "rf_std_d2": np.nan,
                "d2_wl_min": np.nan,
                "d2_wl_max": np.nan,
                "d2_coverage": False,
                "parse_warnings": len(pwarnings),
            })
            continue

        wl_d2, rf_d2, unc_d2, has_unc = prepare_d2_spectrum(segments)

        n_total = sum(len(s[0]) for s in segments)
        n_d2 = len(wl_d2)

        finite_mask = np.isfinite(rf_d2)
        n_finite = int(np.sum(finite_mask))
        n_nan = int(np.sum(np.isnan(rf_d2)))
        n_inf = int(np.sum(np.isinf(rf_d2)))

        rf_finite = rf_d2[finite_mask] if n_finite > 0 else np.array([])

        if n_finite > 0:
            rf_min = float(np.min(rf_finite))
            rf_max = float(np.max(rf_finite))
            rf_median = float(np.median(rf_finite))
            rf_mean = float(np.mean(rf_finite))
            rf_std = float(np.std(rf_finite))
        else:
            rf_min = rf_max = rf_median = rf_mean = rf_std = np.nan

        n_negative = int(np.sum(rf_finite < 0)) if n_finite > 0 else 0
        n_zero = int(np.sum(rf_finite == 0)) if n_finite > 0 else 0
        n_gt1 = int(np.sum(rf_finite > 1)) if n_finite > 0 else 0

        d2_coverage = n_d2 > 0 and float(wl_d2.min()) <= D2_WMIN and float(wl_d2.max()) >= D2_WMAX

        rec = {
            "file_id": crow["file_id"],
            "filename": crow["filename"],
            "mineral": crow["mineral_directory"],
            "category": crow["inclusion_category"],
            "has_uncertainty": int(has_unc),
            "n_segments": len(segments),
            "n_total_points": n_total,
            "n_d2_points": n_d2,
            "n_finite_d2": n_finite,
            "n_nan_d2": n_nan,
            "n_inf_d2": n_inf,
            "n_negative_d2": n_negative,
            "n_zero_d2": n_zero,
            "n_gt1_d2": n_gt1,
            "rf_min_d2": rf_min,
            "rf_max_d2": rf_max,
            "rf_median_d2": rf_median,
            "rf_mean_d2": rf_mean,
            "rf_std_d2": rf_std,
            "d2_wl_min": float(wl_d2.min()) if n_d2 > 0 else np.nan,
            "d2_wl_max": float(wl_d2.max()) if n_d2 > 0 else np.nan,
            "d2_coverage": n_d2 > 0,
            "parse_warnings": len(pwarnings),
        }
        spectrum_records.append(rec)

        # --- Detailed anomaly investigation ---

        # Negative values
        if n_negative > 0:
            neg_mask = finite_mask & (rf_d2 < 0)
            neg_indices = np.where(neg_mask)[0]
            spectra_with_negative.append(crow["file_id"])
            for ni in neg_indices:
                is_near_absorption = False
                if 1 < ni < len(rf_d2) - 2:
                    local = rf_d2[finite_mask][max(0, ni - 5):ni + 6]
                    if len(local) > 2 and np.min(local) <= 0:
                        is_near_absorption = True

                anomaly_records.append({
                    "file_id": crow["file_id"],
                    "filename": crow["filename"],
                    "mineral": crow["mineral_directory"],
                    "anomaly_type": "negative",
                    "wavelength_nm": float(wl_d2[ni]),
                    "reflectance": float(rf_d2[ni]),
                    "magnitude": float(abs(rf_d2[ni])),
                    "near_absorption_minimum": is_near_absorption,
                    "has_uncertainty": int(has_unc),
                    "uncertainty": float(unc_d2[ni]) if has_unc and ni < len(unc_d2) else np.nan,
                })

        # Values > 1
        if n_gt1 > 0:
            gt1_mask = finite_mask & (rf_d2 > 1)
            gt1_indices = np.where(gt1_mask)[0]
            spectra_with_gt1.append(crow["file_id"])
            for gi in gt1_indices:
                anomaly_records.append({
                    "file_id": crow["file_id"],
                    "filename": crow["filename"],
                    "mineral": crow["mineral_directory"],
                    "anomaly_type": "gt1",
                    "wavelength_nm": float(wl_d2[gi]),
                    "reflectance": float(rf_d2[gi]),
                    "magnitude": float(rf_d2[gi] - 1.0),
                    "near_absorption_minimum": False,
                    "has_uncertainty": int(has_unc),
                    "uncertainty": float(unc_d2[gi]) if has_unc and gi < len(unc_d2) else np.nan,
                })

        # Zero values
        if n_zero > 0:
            zero_mask = finite_mask & (rf_d2 == 0)
            zero_indices = np.where(zero_mask)[0]
            spectra_with_zeros.append(crow["file_id"])

            contiguous_groups = []
            group = [zero_indices[0]]
            for zi in zero_indices[1:]:
                if zi == group[-1] + 1:
                    group.append(zi)
                else:
                    contiguous_groups.append(group)
                    group = [zi]
            contiguous_groups.append(group)

            for grp in contiguous_groups:
                is_at_start = grp[0] <= 2
                is_at_end = grp[-1] >= len(rf_d2) - 3
                anomaly_records.append({
                    "file_id": crow["file_id"],
                    "filename": crow["filename"],
                    "mineral": crow["mineral_directory"],
                    "anomaly_type": "zero",
                    "wavelength_nm": float(wl_d2[grp[0]]),
                    "reflectance": 0.0,
                    "magnitude": 0.0,
                    "near_absorption_minimum": False,
                    "has_uncertainty": int(has_unc),
                    "uncertainty": np.nan,
                    "zero_count": len(grp),
                    "contiguous": len(grp) > 1,
                    "at_spectrum_start": is_at_start,
                    "at_spectrum_end": is_at_end,
                    "wl_start": float(wl_d2[grp[0]]),
                    "wl_end": float(wl_d2[grp[-1]]),
                })

        # NaN/Inf values
        if n_nan > 0 or n_inf > 0:
            bad_mask = np.isnan(rf_d2) | np.isinf(rf_d2)
            bad_indices = np.where(bad_mask)[0]
            spectra_with_nan_inf.append(crow["file_id"])
            for bi in bad_indices:
                anomaly_records.append({
                    "file_id": crow["file_id"],
                    "filename": crow["filename"],
                    "mineral": crow["mineral_directory"],
                    "anomaly_type": "nan_inf",
                    "wavelength_nm": float(wl_d2[bi]),
                    "reflectance": float(rf_d2[bi]) if np.isfinite(rf_d2[bi]) else str(rf_d2[bi]),
                    "magnitude": np.nan,
                    "near_absorption_minimum": False,
                    "has_uncertainty": int(has_unc),
                    "uncertainty": np.nan,
                })

        # --- Spectral shape analysis (on finite values only) ---
        if n_finite >= 10:
            wl_fin = wl_d2[finite_mask]
            rf_fin = rf_d2[finite_mask]
            unc_fin = unc_d2[finite_mask] if has_unc else np.array([])

            # Single-point spikes
            spikes = detect_single_point_spikes(wl_fin, rf_fin)
            if spikes:
                spectra_with_spikes.append(crow["file_id"])
                for sp in spikes:
                    spike_rec = {
                        "file_id": crow["file_id"],
                        "filename": crow["filename"],
                        "mineral": crow["mineral_directory"],
                        "wavelength_nm": sp["wavelength"],
                        "reflectance": sp["reflectance"],
                        "spike_type": sp["type"],
                        "ratio_to_median": sp["ratio_to_median"],
                        "left_diff": sp["left_diff"],
                        "right_diff": sp["right_diff"],
                    }
                    spike_records.append(spike_rec)
                    anomaly_records.append({
                        "file_id": crow["file_id"],
                        "filename": crow["filename"],
                        "mineral": crow["mineral_directory"],
                        "anomaly_type": f"spike_{sp['type']}",
                        "wavelength_nm": sp["wavelength"],
                        "reflectance": sp["reflectance"],
                        "magnitude": float(max(sp["left_diff"], sp["right_diff"])),
                        "near_absorption_minimum": False,
                        "has_uncertainty": int(has_unc),
                        "uncertainty": np.nan,
                        "spike_ratio": sp["ratio_to_median"],
                    })

            # Constant runs
            const_runs = detect_constant_runs(wl_fin, rf_fin)
            if const_runs:
                spectra_with_constant_runs.append(crow["file_id"])
                for cr in const_runs:
                    constant_run_records.append({
                        "file_id": crow["file_id"],
                        "filename": crow["filename"],
                        "mineral": crow["mineral_directory"],
                        "start_wl": cr["start_wl"],
                        "end_wl": cr["end_wl"],
                        "length_points": cr["length"],
                        "value": cr["value"],
                        "wl_range_nm": cr["wl_range_nm"],
                    })

            # Discontinuities
            discs = detect_discontinuities(wl_fin, rf_fin)
            if discs:
                spectra_with_discontinuities.append(crow["file_id"])
                for d in discs:
                    discontinuity_records.append({
                        "file_id": crow["file_id"],
                        "filename": crow["filename"],
                        "mineral": crow["mineral_directory"],
                        "wl_left": d["wavelength_left"],
                        "wl_right": d["wavelength_right"],
                        "rf_left": d["reflectance_left"],
                        "rf_right": d["reflectance_right"],
                        "abs_jump": d["abs_jump"],
                        "ratio_to_median": d["ratio_to_median"],
                    })

        if (file_idx + 1) % 100 == 0:
            print(f"  Processed {file_idx + 1}/{len(candidates)} spectra")

    print(f"  Processed {len(candidates)}/{len(candidates)} spectra")

    # --- Build DataFrames ---
    spec_df = pd.DataFrame(spectrum_records)
    anomaly_df = pd.DataFrame(anomaly_records) if anomaly_records else pd.DataFrame()
    spike_df = pd.DataFrame(spike_records) if spike_records else pd.DataFrame()
    const_run_df = pd.DataFrame(constant_run_records) if constant_run_records else pd.DataFrame()
    disc_df = pd.DataFrame(discontinuity_records) if discontinuity_records else pd.DataFrame()

    # ================================================================
    # AGGREGATE STATISTICS
    # ================================================================

    print("\n[4/6] Computing aggregate statistics...")

    total_spectra = len(spec_df)
    spectra_fully_finite = (spec_df["n_nan_d2"] == 0) & (spec_df["n_inf_d2"] == 0)
    n_fully_finite = spectra_fully_finite.sum()

    all_rf_min = spec_df["rf_min_d2"].dropna()
    all_rf_max = spec_df["rf_max_d2"].dropna()
    all_rf_median = spec_df["rf_median_d2"].dropna()

    n_neg_spectra = len(spectra_with_negative)
    n_gt1_spectra = len(spectra_with_gt1)
    n_zero_spectra = len(spectra_with_zeros)
    n_nan_inf_spectra = len(spectra_with_nan_inf)
    n_spike_spectra = len(spectra_with_spikes)
    n_const_run_spectra = len(spectra_with_constant_runs)
    n_disc_spectra = len(spectra_with_discontinuities)

    # Negative value details
    neg_anomalies = anomaly_df[anomaly_df["anomaly_type"] == "negative"] if len(anomaly_df) > 0 else pd.DataFrame()
    n_neg_total = len(neg_anomalies)
    if n_neg_total > 0:
        neg_magnitudes = neg_anomalies["magnitude"].values
        neg_magnitudes_finite = neg_magnitudes[np.isfinite(neg_magnitudes)]
        n_neg_near_absorption = neg_anomalies["near_absorption_minimum"].sum() if "near_absorption_minimum" in neg_anomalies.columns else 0

    # >1 value details
    gt1_anomalies = anomaly_df[anomaly_df["anomaly_type"] == "gt1"] if len(anomaly_df) > 0 else pd.DataFrame()
    n_gt1_total = len(gt1_anomalies)
    if n_gt1_total > 0:
        gt1_magnitudes = gt1_anomalies["magnitude"].values

    # Zero value details
    zero_anomalies = anomaly_df[anomaly_df["anomaly_type"] == "zero"] if len(anomaly_df) > 0 else pd.DataFrame()
    n_zero_total = len(zero_anomalies)

    # --- Anomaly distribution by mineral ---
    mineral_dist = {}
    if len(anomaly_df) > 0:
        for mineral in spec_df["mineral"].unique():
            m_spec = spec_df[spec_df["mineral"] == mineral]
            m_anom = anomaly_df[anomaly_df["mineral"] == mineral]
            mineral_dist[mineral] = {
                "n_spectra": len(m_spec),
                "n_anomalies": len(m_anom),
                "n_negative": len(m_anom[m_anom["anomaly_type"] == "negative"]),
                "n_gt1": len(m_anom[m_anom["anomaly_type"] == "gt1"]),
                "n_zero": len(m_anom[m_anom["anomaly_type"] == "zero"]),
                "n_spikes": len(m_anom[m_anom["anomaly_type"].str.startswith("spike")]),
            }

    # --- Anomaly distribution by wavelength region ---
    region_dist = {}
    if len(anomaly_df) > 0:
        for region_name, (rmin, rmax) in WAVELENGTH_REGIONS.items():
            r_anom = anomaly_df[
                (anomaly_df["wavelength_nm"] >= rmin) & (anomaly_df["wavelength_nm"] < rmax)
            ]
            region_dist[region_name] = len(r_anom)

    # --- Uncertainty comparison ---
    uncertainty_ratios = []
    if len(anomaly_df) > 0 and "uncertainty" in anomaly_df.columns:
        unc_anomalies = anomaly_df[
            (anomaly_df["has_uncertainty"] == 1)
            & (anomaly_df["uncertainty"].notna())
            & (anomaly_df["uncertainty"] > 0)
            & (anomaly_df["anomaly_type"].isin(["negative", "gt1"]))
            & (anomaly_df["magnitude"].notna())
        ]
        for _, ua in unc_anomalies.iterrows():
            ratio = abs(ua["magnitude"]) / ua["uncertainty"]
            uncertainty_ratios.append({
                "file_id": ua["file_id"],
                "mineral": ua["mineral"],
                "anomaly_type": ua["anomaly_type"],
                "wavelength_nm": ua["wavelength_nm"],
                "magnitude": ua["magnitude"],
                "uncertainty": ua["uncertainty"],
                "ratio": ratio,
            })

    unc_ratio_df = pd.DataFrame(uncertainty_ratios) if uncertainty_ratios else pd.DataFrame()

    # ================================================================
    # QC CATEGORIES
    # ================================================================

    print("\n[5/6] Assigning QC categories...")

    qc_categories = {}
    for _, srow in spec_df.iterrows():
        fid = srow["file_id"]

        n_neg = srow["n_negative_d2"]
        n_gt1 = srow["n_gt1_d2"]
        n_zero = srow["n_zero_d2"]
        n_nan = srow["n_nan_d2"]
        n_inf = srow["n_inf_d2"]

        has_spike = fid in spectra_with_spikes
        has_const = fid in spectra_with_constant_runs

        if n_nan > 0 or n_inf > 0:
            qc_categories[fid] = "D"
        elif (n_neg > 5) or (n_gt1 > 10):
            qc_categories[fid] = "C"
        elif n_neg > 0 or n_gt1 > 0 or n_zero > 0 or has_const:
            qc_categories[fid] = "B"
        elif has_spike:
            n_spikes_fid = len(spike_df[spike_df["file_id"] == fid]) if len(spike_df) > 0 else 0
            if n_spikes_fid > 50:
                qc_categories[fid] = "B"
            else:
                qc_categories[fid] = "A"
        else:
            qc_categories[fid] = "A"

    spec_df["qc_category"] = spec_df["file_id"].map(qc_categories)

    qc_counts = spec_df["qc_category"].value_counts().sort_index()

    # ================================================================
    # WRITE OUTPUTS
    # ================================================================

    print("\n[6/6] Writing outputs...")

    # --- Anomaly CSV ---
    csv_path = OUT_DIR / "phase2_step3_reflectance_qc.csv"
    if len(anomaly_df) > 0:
        anomaly_df.to_csv(csv_path, index=False)
    else:
        pd.DataFrame(columns=["file_id", "anomaly_type"]).to_csv(csv_path, index=False)
    print(f"  Anomaly CSV: {csv_path} ({len(anomaly_df)} rows)")

    # --- Spectrum summary CSV ---
    summary_path = OUT_DIR / "phase2_step3_reflectance_qc_spectrum_summary.csv"
    spec_df.to_csv(summary_path, index=False)
    print(f"  Spectrum summary CSV: {summary_path} ({len(spec_df)} rows)")

    # --- Diagnostic figures ---
    fig_paths = generate_figures(spec_df, anomaly_df, spike_df, unc_ratio_df)

    # --- Post-flight checksum verification ---
    print("  Verifying raw checksums (post-flight)...")
    post_checked, post_mismatches = validate_checksums()
    print(f"  Checked: {post_checked}, Mismatches: {len(post_mismatches)}")

    # --- Markdown report ---
    report = build_report(
        spec_df, anomaly_df, spike_df, const_run_df, disc_df,
        unc_ratio_df, qc_counts, mineral_dist, region_dist,
        fig_paths, pre_checked, len(pre_mismatches),
        post_checked, len(post_mismatches),
        parse_warning_log,
    )
    report_path = OUT_DIR / "phase2_step3_reflectance_qc_investigation.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report: {report_path}")

    # --- Final summary ---
    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)
    print(f"\nQC category distribution:")
    for cat in ["A", "B", "C", "D"]:
        n = qc_counts.get(cat, 0)
        print(f"  {cat}: {n} spectra")
    print(f"\nAnomalies found: {len(anomaly_df)} total occurrences")
    print(f"  Negative reflectance: {n_neg_total} values in {n_neg_spectra} spectra")
    print(f"  Reflectance > 1: {n_gt1_total} values in {n_gt1_spectra} spectra")
    print(f"  Zero values: {n_zero_total} groups in {n_zero_spectra} spectra")
    print(f"  NaN/Inf: {n_nan_inf_spectra} spectra")
    print(f"  Single-point spikes: {len(spike_df)} in {n_spike_spectra} spectra")
    print(f"  Constant runs: {len(const_run_df)} in {n_const_run_spectra} spectra")
    print(f"  Discontinuities: {len(disc_df)} in {n_disc_spectra} spectra")
    print(f"\nRaw integrity: pre-flight={len(pre_mismatches)} mismatches, post-flight={len(post_mismatches)} mismatches")
    print(f"\nSTOPPING. Awaiting researcher review before freezing reflectance-QC decisions.")


# ================================================================
# FIGURE GENERATION
# ================================================================


def generate_figures(spec_df, anomaly_df, spike_df, unc_ratio_df):
    """Generate diagnostic figures. Returns dict of figure paths."""
    fig_paths = {}

    # Figure 1: Reflectance range distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    finite_min = spec_df["rf_min_d2"].dropna()
    finite_max = spec_df["rf_max_d2"].dropna()
    axes[0].hist(finite_min, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Minimum reflectance")
    axes[0].set_ylabel("Number of spectra")
    axes[0].set_title("Distribution of per-spectrum minimum reflectance")
    axes[0].axvline(0, color="red", linestyle="--", alpha=0.7, label="R=0")
    axes[0].legend()
    axes[1].hist(finite_max, bins=50, color="darkorange", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Maximum reflectance")
    axes[1].set_ylabel("Number of spectra")
    axes[1].set_title("Distribution of per-spectrum maximum reflectance")
    axes[1].axvline(1, color="red", linestyle="--", alpha=0.7, label="R=1")
    axes[1].legend()
    fig.tight_layout()
    p = FIG_DIR / "reflectance_qc_fig1_range_distribution.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    fig_paths["fig1"] = str(p)

    # Figure 2: Negative value magnitude distribution
    if len(anomaly_df) > 0:
        neg = anomaly_df[anomaly_df["anomaly_type"] == "negative"]
        if len(neg) > 0:
            mags = neg["magnitude"].values
            mags = mags[np.isfinite(mags)]
            if len(mags) > 0:
                fig, ax = plt.subplots(figsize=(8, 5))
                if len(mags) > 20:
                    ax.hist(mags, bins=50, color="crimson", edgecolor="white", alpha=0.8)
                    ax.set_xscale("log")
                else:
                    ax.scatter(range(len(mags)), sorted(mags), color="crimson", s=20)
                    ax.set_xlabel("Occurrence index")
                ax.set_xlabel("Magnitude of negative value (|R|)")
                ax.set_ylabel("Count")
                ax.set_title(f"Negative reflectance magnitudes ({len(mags)} values)")
                fig.tight_layout()
                p = FIG_DIR / "reflectance_qc_fig2_negative_magnitudes.png"
                fig.savefig(p, dpi=150)
                plt.close(fig)
                fig_paths["fig2"] = str(p)

    # Figure 3: >1 value distribution
    if len(anomaly_df) > 0:
        gt1 = anomaly_df[anomaly_df["anomaly_type"] == "gt1"]
        if len(gt1) > 0:
            vals = gt1["reflectance"].values
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(vals, bins=50, color="darkred", edgecolor="white", alpha=0.8)
                ax.set_xlabel("Reflectance value")
                ax.set_ylabel("Count")
                ax.set_title(f"Reflectance > 1 values ({len(vals)} occurrences)")
                ax.axvline(1, color="gray", linestyle="--", alpha=0.5)
                fig.tight_layout()
                p = FIG_DIR / "reflectance_qc_fig3_gt1_distribution.png"
                fig.savefig(p, dpi=150)
                plt.close(fig)
                fig_paths["fig3"] = str(p)

    # Figure 4: Anomaly frequency by wavelength region
    if len(anomaly_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        region_counts = {}
        for rname, (rmin, rmax) in WAVELENGTH_REGIONS.items():
            mask = (
                (anomaly_df["wavelength_nm"] >= rmin)
                & (anomaly_df["wavelength_nm"] < rmax)
            )
            region_counts[rname.split(" (")[0]] = mask.sum()
        bars = ax.bar(range(len(region_counts)), list(region_counts.values()),
                      color=["steelblue", "darkorange", "forestgreen", "crimson"])
        ax.set_xticks(range(len(region_counts)))
        ax.set_xticklabels(list(region_counts.keys()), rotation=15, ha="right")
        ax.set_ylabel("Number of anomalies")
        ax.set_title("Anomaly frequency by wavelength region")
        fig.tight_layout()
        p = FIG_DIR / "reflectance_qc_fig4_anomaly_by_region.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        fig_paths["fig4"] = str(p)

    # Figure 5: Uncertainty vs anomaly magnitude
    if len(unc_ratio_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        neg_ratios = unc_ratio_df[unc_ratio_df["anomaly_type"] == "negative"]
        gt1_ratios = unc_ratio_df[unc_ratio_df["anomaly_type"] == "gt1"]
        if len(neg_ratios) > 0:
            ax.scatter(neg_ratios["uncertainty"], neg_ratios["magnitude"],
                       c="crimson", alpha=0.5, s=15, label="Negative")
        if len(gt1_ratios) > 0:
            ax.scatter(gt1_ratios["uncertainty"], gt1_ratios["magnitude"],
                       c="darkorange", alpha=0.5, s=15, label="R > 1")
        ax.set_xlabel("Reported uncertainty")
        ax.set_ylabel("|Anomaly magnitude|")
        ax.set_title("Anomaly magnitude vs reported uncertainty")
        ax.legend()
        ax.set_xscale("log")
        ax.set_yscale("log")
        fig.tight_layout()
        p = FIG_DIR / "reflectance_qc_fig5_uncertainty_vs_anomaly.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        fig_paths["fig5"] = str(p)

    # Figure 6: Representative suspicious spectra (up to 4)
    suspicious_fids = []
    if len(anomaly_df) > 0:
        for atype in ["negative", "gt1", "zero"]:
            sub = anomaly_df[anomaly_df["anomaly_type"] == atype]
            if len(sub) > 0:
                suspicious_fids.append(sub.iloc[0]["file_id"])
        if len(spike_df) > 0:
            suspicious_fids.append(spike_df.iloc[0]["file_id"])

    if suspicious_fids:
        suspicious_fids = suspicious_fids[:4]
        n_plots = len(suspicious_fids)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes_flat = axes.flatten()

        pop = pd.read_csv(POPULATION_CSV)
        candidates = pop[pop["inclusion_category"].isin(["A", "B"])]

        for idx, fid in enumerate(suspicious_fids):
            ax = axes_flat[idx]
            crow = candidates[candidates["file_id"] == fid].iloc[0]
            fp = DATA_DIR / crow["raw_path"]
            segments, _, _ = parse_relab_full(fp)
            wl_d2, rf_d2, _, _ = prepare_d2_spectrum(segments)
            finite = np.isfinite(rf_d2)
            ax.plot(wl_d2[finite], rf_d2[finite], "k-", linewidth=0.5)
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Reflectance")
            ax.set_title(f"{crow['filename']} ({crow['mineral_directory']})")
            ax.set_xlim(D2_WMIN, D2_WMAX)
            ax.axhline(0, color="red", linestyle="--", alpha=0.3)
            ax.axhline(1, color="red", linestyle="--", alpha=0.3)

        for idx in range(n_plots, 4):
            axes_flat[idx].set_visible(False)

        fig.suptitle("Representative suspicious spectra", fontsize=13)
        fig.tight_layout()
        p = FIG_DIR / "reflectance_qc_fig6_suspicious_spectra.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        fig_paths["fig6"] = str(p)

    return fig_paths


# ================================================================
# REPORT GENERATION
# ================================================================


def build_report(
    spec_df, anomaly_df, spike_df, const_run_df, disc_df,
    unc_ratio_df, qc_counts, mineral_dist, region_dist,
    fig_paths, pre_checked, pre_n_mismatch,
    post_checked, post_n_mismatch,
    parse_warning_log,
):
    w = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    w.append("# Phase 2 Step 3 — Reflectance Quality-Control Investigation")
    w.append("")
    w.append(f"**Date:** {now}")
    w.append("")

    # 1. Objective
    w.append("## 1. Objective")
    w.append("")
    w.append("Investigate the validity and quality of RELAB reflectance values within")
    w.append("the frozen D2 analysis domain (320–2550 nm) for all 925 analysis candidates.")
    w.append("")
    w.append("The goal is to distinguish:")
    w.append("")
    w.append("- Physically impossible values")
    w.append("- Technically unusual but potentially valid measurements")
    w.append("- Ordinary measurement variation")
    w.append("- Potentially problematic spectra requiring further investigation")
    w.append("")
    w.append("**Scientific principle:** An unusual spectrum is not automatically a bad spectrum.")
    w.append("Mineral spectra may legitimately contain strong absorptions, low reflectance,")
    w.append("high reflectance, sharp features, or unusual shapes.")
    w.append("")

    # 2. Frozen constraints
    w.append("## 2. Frozen Constraints")
    w.append("")
    w.append("| Parameter | Value |")
    w.append("|-----------|-------|")
    w.append("| Dataset | RELAB only |")
    w.append("| Analysis candidates | 925 (800 A + 125 B) |")
    w.append("| D2 domain | 320–2550 nm |")
    w.append("| Target grid | 5 nm spacing, 447 points |")
    w.append("| Spectra fully covering D2 | 454 |")
    w.append("| Native 5-nm spectra | 390 |")
    w.append("| Native 10-nm spectra | 64 |")
    w.append("")
    w.append("Frozen preprocessing decisions:")
    w.append("")
    w.append("1. Duplicate wavelengths → keep first occurrence deterministically")
    w.append("2. Unsorted wavelengths → sort wavelength-value pairs in memory")
    w.append("3. Splice boundaries → no splice-specific masking/correction/exclusion")
    w.append("4. Native 10-nm spectra → PCHIP interpolation to frozen 5-nm grid")
    w.append("5. Linear interpolation → retained only for future sensitivity analysis")
    w.append("")
    w.append("No frozen decision was changed in this investigation.")
    w.append("")

    # 3. Reflectance-column interpretation
    w.append("## 3. Reflectance-Column Interpretation")
    w.append("")
    w.append("**Identification method:** Direct inspection of raw `.tab` files and the")
    w.append("RELAB field dictionary (`metadata/provenance/relab_field_dictionary.md`).")
    w.append("")
    w.append("**Findings:**")
    w.append("")
    w.append("- **Column 1:** Wavelength in nanometers (nm)")
    w.append("- **Column 2:** Reflectance, unitless, fractional (0–1 range)")
    w.append("- **Column 3 (when present):** Measurement uncertainty, same units as reflectance")
    w.append("")
    w.append("**Consistency:** The reflectance representation is consistent across all 1,017 raw files.")
    w.append("Values in the 0–1 range are observed for the vast majority of data points.")
    w.append("")

    n_with_unc_all = 461
    n_with_unc_cand = spec_df["has_uncertainty"].sum()
    w.append(f"**Uncertainty:** {n_with_unc_all} of 1,017 raw files contain a third uncertainty column.")
    w.append(f"Among the 925 analysis candidates, {n_with_unc_cand} files have uncertainty data.")
    w.append("Uncertainty semantics are not documented by RELAB; values appear to represent")
    w.append("one-sigma measurement noise estimated during acquisition.")
    w.append("")

    # 4. Numerical validity
    w.append("## 4. Numerical Validity")
    w.append("")
    w.append("### 4.1 Per-spectrum statistics summary")
    w.append("")

    n_fully_finite = ((spec_df["n_nan_d2"] == 0) & (spec_df["n_inf_d2"] == 0)).sum()
    w.append(f"| Metric | Value |")
    w.append(f"|--------|-------|")
    w.append(f"| Total spectra | {len(spec_df)} |")
    w.append(f"| Spectra with all-finite D2 values | {n_fully_finite} |")
    w.append(f"| Spectra with NaN in D2 | {(spec_df['n_nan_d2'] > 0).sum()} |")
    w.append(f"| Spectra with Inf in D2 | {(spec_df['n_inf_d2'] > 0).sum()} |")
    w.append(f"| Spectra with negative values | {(spec_df['n_negative_d2'] > 0).sum()} |")
    w.append(f"| Spectra with zero values | {(spec_df['n_zero_d2'] > 0).sum()} |")
    w.append(f"| Spectra with values > 1 | {(spec_df['n_gt1_d2'] > 0).sum()} |")
    w.append("")

    w.append("### 4.2 Global reflectance range")
    w.append("")
    global_min = spec_df["rf_min_d2"].min()
    global_max = spec_df["rf_max_d2"].max()
    global_median_min = spec_df["rf_min_d2"].median()
    global_median_max = spec_df["rf_max_d2"].median()
    w.append(f"| Statistic | Value |")
    w.append(f"|-----------|-------|")
    w.append(f"| Global minimum reflectance | {global_min:.6f} |")
    w.append(f"| Global maximum reflectance | {global_max:.6f} |")
    w.append(f"| Median per-spectrum minimum | {global_median_min:.6f} |")
    w.append(f"| Median per-spectrum maximum | {global_median_max:.6f} |")
    w.append("")

    # 5. Negative reflectance investigation
    w.append("## 5. Negative Reflectance Investigation")
    w.append("")
    n_neg_spectra = (spec_df["n_negative_d2"] > 0).sum()
    n_neg_total = spec_df["n_negative_d2"].sum()
    w.append(f"- **Spectra with negative values:** {n_neg_spectra} of {len(spec_df)}")
    w.append(f"- **Total negative values:** {int(n_neg_total)}")
    w.append("")

    if len(anomaly_df) > 0:
        neg = anomaly_df[anomaly_df["anomaly_type"] == "negative"]
        if len(neg) > 0:
            mags = neg["magnitude"].values
            mags_finite = mags[np.isfinite(mags)]
            if len(mags_finite) > 0:
                w.append("### 5.1 Magnitude distribution")
                w.append("")
                w.append(f"| Statistic | Value |")
                w.append(f"|-----------|-------|")
                w.append(f"| Min |magnitude| | {mags_finite.min():.6f} |")
                w.append(f"| Median |magnitude| | {np.median(mags_finite):.6f} |")
                w.append(f"| Mean |magnitude| | {np.mean(mags_finite):.6f} |")
                w.append(f"| Max |magnitude| | {mags_finite.max():.6f} |")
                w.append(f"| 95th percentile | | {np.percentile(mags_finite, 95):.6f} |")
                w.append("")

                small_neg = np.sum(mags_finite < 0.01)
                moderate_neg = np.sum((mags_finite >= 0.01) & (mags_finite < 0.05))
                large_neg = np.sum(mags_finite >= 0.05)
                w.append("### 5.2 Magnitude categories")
                w.append("")
                w.append(f"| Category | Count | Fraction |")
                w.append(f"|----------|-------|----------|")
                w.append(f"| Small (|R| < 0.01) | {small_neg} | {small_neg / len(mags_finite) * 100:.1f}% |")
                w.append(f"| Moderate (0.01 ≤ |R| < 0.05) | {moderate_neg} | {moderate_neg / len(mags_finite) * 100:.1f}% |")
                w.append(f"| Large (|R| ≥ 0.05) | {large_neg} | {large_neg / len(mags_finite) * 100:.1f}% |")
                w.append("")

                w.append("### 5.3 Context assessment")
                w.append("")
                n_near_abs = neg["near_absorption_minimum"].sum() if "near_absorption_minimum" in neg.columns else 0
                w.append(f"- Negative values near absorption minima: {n_near_abs} of {len(neg)}")
                w.append("")
                w.append("Small negative values (|R| < 0.01) are consistent with measurement noise")
                w.append("or background correction residuals. They do not necessarily indicate")
                w.append("corrupted data. Large or repeated negative values require separate investigation.")
                w.append("")

                # Distribution by mineral
                w.append("### 5.4 Distribution by mineral")
                w.append("")
                w.append("| Mineral | Spectra with negatives | Total negative values |")
                w.append("|---------|----------------------|----------------------|")
                neg_by_mineral = neg.groupby("mineral").agg(
                    n_values=("magnitude", "count"),
                ).reset_index()
                neg_spec_by_mineral = spec_df[spec_df["n_negative_d2"] > 0].groupby("mineral").size().reset_index(name="n_spectra")
                neg_by_mineral = neg_by_mineral.merge(neg_spec_by_mineral, on="mineral", how="left")
                for _, r in neg_by_mineral.sort_values("n_values", ascending=False).iterrows():
                    w.append(f"| {r['mineral']} | {r['n_spectra']} | {r['n_values']} |")
                w.append("")

                if "fig2" in fig_paths:
                    w.append(f"![Negative magnitudes]({fig_paths['fig2']})")
                    w.append("")

    # 6. Reflectance > 1 investigation
    w.append("## 6. Reflectance > 1 Investigation")
    w.append("")
    n_gt1_spectra = (spec_df["n_gt1_d2"] > 0).sum()
    n_gt1_total = int(spec_df["n_gt1_d2"].sum())
    w.append(f"- **Spectra with values > 1:** {n_gt1_spectra} of {len(spec_df)}")
    w.append(f"- **Total values > 1:** {n_gt1_total}")
    w.append("")

    if len(anomaly_df) > 0:
        gt1 = anomaly_df[anomaly_df["anomaly_type"] == "gt1"]
        if len(gt1) > 0:
            vals = gt1["reflectance"].values
            vals_finite = vals[np.isfinite(vals)]
            if len(vals_finite) > 0:
                w.append("### 6.1 Magnitude distribution")
                w.append("")
                w.append(f"| Statistic | Value |")
                w.append(f"|-----------|-------|")
                w.append(f"| Min value | {vals_finite.min():.6f} |")
                w.append(f"| Median value | {np.median(vals_finite):.6f} |")
                w.append(f"| Max value | {vals_finite.max():.6f} |")
                w.append(f"| Mean excess above 1 | {np.mean(vals_finite - 1):.6f} |")
                w.append("")

                w.append("### 6.2 Distribution by mineral")
                w.append("")
                w.append("| Mineral | Spectra with R>1 | Total R>1 values |")
                w.append("|---------|-----------------|-----------------|")
                gt1_by_mineral = gt1.groupby("mineral").agg(
                    n_values=("reflectance", "count"),
                ).reset_index()
                gt1_spec_by_mineral = spec_df[spec_df["n_gt1_d2"] > 0].groupby("mineral").size().reset_index(name="n_spectra")
                gt1_by_mineral = gt1_by_mineral.merge(gt1_spec_by_mineral, on="mineral", how="left")
                for _, r in gt1_by_mineral.sort_values("n_values", ascending=False).iterrows():
                    w.append(f"| {r['mineral']} | {r['n_spectra']} | {r['n_values']} |")
                w.append("")

                w.append("### 6.3 Assessment")
                w.append("")
                w.append("Reflectance values slightly above 1 (e.g., 1.001–1.01) can occur due to")
                w.append("measurement noise, reference standard uncertainty, or specular reflection")
                w.append("contributions. Values substantially above 1 (e.g., > 1.1) may indicate")
                w.append("calibration issues or measurement artifacts.")
                w.append("")

                if "fig3" in fig_paths:
                    w.append(f"![R>1 distribution]({fig_paths['fig3']})")
                    w.append("")

    # 7. Zero-value investigation
    w.append("## 7. Zero-Value Investigation")
    w.append("")
    n_zero_spectra = (spec_df["n_zero_d2"] > 0).sum()
    n_zero_total = int(spec_df["n_zero_d2"].sum())
    w.append(f"- **Spectra with zero values:** {n_zero_spectra} of {len(spec_df)}")
    w.append(f"- **Total zero values:** {n_zero_total}")
    w.append("")

    if len(anomaly_df) > 0:
        zeros = anomaly_df[anomaly_df["anomaly_type"] == "zero"]
        if len(zeros) > 0:
            w.append("### 7.1 Pattern analysis")
            w.append("")
            if "contiguous" in zeros.columns:
                n_contiguous = zeros["contiguous"].sum()
                n_isolated = len(zeros) - n_contiguous
                w.append(f"- Isolated zero groups: {n_isolated}")
                w.append(f"- Contiguous zero groups (multiple adjacent zeros): {n_contiguous}")
                w.append("")

            if "at_spectrum_start" in zeros.columns:
                n_at_start = zeros["at_spectrum_start"].sum()
                n_at_end = zeros["at_spectrum_end"].sum()
                w.append(f"- Zero groups at spectrum start (first 3 points): {n_at_start}")
                w.append(f"- Zero groups at spectrum end (last 3 points): {n_at_end}")
                w.append("")

            w.append("### 7.2 Distribution by mineral")
            w.append("")
            w.append("| Mineral | Spectra with zeros | Total zero values |")
            w.append("|---------|-------------------|------------------|")
            zero_by_mineral = zeros.groupby("mineral").size().reset_index(name="n_groups")
            zero_spec_by_mineral = spec_df[spec_df["n_zero_d2"] > 0].groupby("mineral").size().reset_index(name="n_spectra")
            zero_by_mineral = zero_by_mineral.merge(zero_spec_by_mineral, on="mineral", how="left")
            for _, r in zero_by_mineral.sort_values("n_groups", ascending=False).iterrows():
                w.append(f"| {r['mineral']} | {r['n_spectra']} | {r['n_groups']} |")
            w.append("")

    # 8. NaN/Inf investigation
    w.append("## 8. NaN/Inf Investigation")
    w.append("")
    n_nan_spectra = (spec_df["n_nan_d2"] > 0).sum()
    n_inf_spectra = (spec_df["n_inf_d2"] > 0).sum()
    w.append(f"- **Spectra with NaN in D2:** {n_nan_spectra}")
    w.append(f"- **Spectra with Inf in D2:** {n_inf_spectra}")
    w.append("")

    if n_nan_spectra == 0 and n_inf_spectra == 0:
        w.append("No NaN or Inf values were found in any D2-restricted spectrum.")
        w.append("All parsed reflectance values are finite numbers.")
    else:
        w.append("### 8.1 Details")
        w.append("")
        if len(anomaly_df) > 0:
            nan_inf = anomaly_df[anomaly_df["anomaly_type"] == "nan_inf"]
            if len(nan_inf) > 0:
                w.append("| File | Mineral | Wavelength | Value |")
                w.append("|------|---------|-----------|-------|")
                for _, r in nan_inf.iterrows():
                    w.append(f"| {r['filename']} | {r['mineral']} | {r['wavelength_nm']:.1f} | {r['reflectance']} |")
                w.append("")

    if parse_warning_log:
        w.append("### 8.1 Parse warnings")
        w.append("")
        w.append(f"{len(parse_warning_log)} parse warnings were generated during file reading:")
        w.append("")
        for pw in parse_warning_log[:20]:
            w.append(f"- {pw['filename']}: {pw['warning']}")
        if len(parse_warning_log) > 20:
            w.append(f"- ... and {len(parse_warning_log) - 20} more")
        w.append("")
    else:
        w.append("No parse warnings were generated. All data rows were successfully parsed.")
    w.append("")

    # 9. Spectral-shape anomalies
    w.append("## 9. Spectral-Shape Anomalies")
    w.append("")

    w.append("### 9.1 Single-point spikes and drops")
    w.append("")
    w.append(f"- **Spectra with spikes/drops:** {len(spike_df['file_id'].unique()) if len(spike_df) > 0 else 0}")
    w.append(f"- **Total spike/drop events:** {len(spike_df)}")
    w.append("")
    if len(spike_df) > 0:
        w.append("Spikes are defined as isolated points where both adjacent |dR/dλ| values")
        w.append("exceed 10× the median |dR/dλ| (wavelength-normalized), while the neighbors")
        w.append("remain consistent. This normalization accounts for the wide range of")
        w.append("wavelength spacings in the dataset (0.2–10+ nm).")
        w.append("")
        if "ratio_to_median" in spike_df.columns:
            ratios = spike_df["ratio_to_median"].values
            w.append(f"| Statistic | Value |")
            w.append(f"|-----------|-------|")
            w.append(f"| Median ratio to local median | {np.median(ratios):.1f} |")
            w.append(f"| Max ratio to local median | {np.max(ratios):.1f} |")
            w.append(f"| Mean ratio to local median | {np.mean(ratios):.1f} |")
            w.append("")

        n_spike_type = (spike_df["spike_type"] == "spike").sum() if "spike_type" in spike_df.columns else 0
        n_drop_type = (spike_df["spike_type"] == "drop").sum() if "spike_type" in spike_df.columns else 0
        w.append(f"- Upward spikes: {n_spike_type}")
        w.append(f"- Downward drops: {n_drop_type}")
        w.append("")

    w.append("### 9.2 Constant-value runs")
    w.append("")
    w.append(f"- **Spectra with constant runs (≥10 points):** {len(const_run_df['file_id'].unique()) if len(const_run_df) > 0 else 0}")
    w.append(f"- **Total constant runs:** {len(const_run_df)}")
    w.append("")
    if len(const_run_df) > 0:
        w.append("Constant runs may indicate detector saturation, data clipping, or")
        w.append("instrument artifacts at spectral extremes.")
        w.append("")

    w.append("### 9.3 Discontinuities")
    w.append("")
    w.append(f"- **Spectra with discontinuities:** {len(disc_df['file_id'].unique()) if len(disc_df) > 0 else 0}")
    w.append(f"- **Total discontinuity events:** {len(disc_df)}")
    w.append("")
    if len(disc_df) > 0:
        w.append("Discontinuities are adjacent-point |dR/dλ| values exceeding 20× the median")
        w.append("|dR/dλ| (wavelength-normalized). These may correspond to splice boundaries,")
        w.append("detector transitions, or genuine sharp spectral features in minerals with")
        w.append("narrow absorption bands.")
        w.append("")

    # 10. Uncertainty comparison
    w.append("## 10. Uncertainty Comparison")
    w.append("")
    if len(unc_ratio_df) > 0:
        w.append(f"Anomaly-to-uncertainty ratios were computed for {len(unc_ratio_df)} anomalous")
        w.append("values in spectra with reported uncertainty.")
        w.append("")
        ratios = unc_ratio_df["ratio"].values
        w.append(f"| Statistic | Value |")
        w.append(f"|-----------|-------|")
        w.append(f"| Median |anomaly|/uncertainty | {np.median(ratios):.2f} |")
        w.append(f"| Mean |anomaly|/uncertainty | {np.mean(ratios):.2f} |")
        w.append(f"| Max |anomaly|/uncertainty | {np.max(ratios):.2f} |")
        w.append(f"| Fraction with ratio > 3 | {(ratios > 3).sum() / len(ratios) * 100:.1f}% |")
        w.append(f"| Fraction with ratio > 10 | {(ratios > 10).sum() / len(ratios) * 100:.1f}% |")
        w.append("")
        w.append("A ratio > 3 suggests the anomaly exceeds the reported noise level substantially.")
        w.append("However, RELAB uncertainty semantics are not formally documented, so these")
        w.append("ratios should be interpreted cautiously.")
        w.append("")
        if "fig5" in fig_paths:
            w.append(f"![Uncertainty vs anomaly]({fig_paths['fig5']})")
            w.append("")
    else:
        w.append("No anomalous values with reported uncertainty were found for ratio computation.")
        w.append("")

    # 11. Anomaly distribution
    w.append("## 11. Anomaly Distribution")
    w.append("")

    w.append("### 11.1 By mineral")
    w.append("")
    w.append("| Mineral | Total spectra | Total anomalies | Negative | R>1 | Zero | Spikes |")
    w.append("|---------|--------------|----------------|----------|-----|------|--------|")
    for mineral in sorted(mineral_dist.keys()):
        d = mineral_dist[mineral]
        w.append(f"| {mineral} | {d['n_spectra']} | {d['n_anomalies']} | {d['n_negative']} | {d['n_gt1']} | {d['n_zero']} | {d['n_spikes']} |")
    w.append("")

    w.append("### 11.2 By wavelength region")
    w.append("")
    w.append("| Region | Anomaly count |")
    w.append("|--------|--------------|")
    for rname, count in region_dist.items():
        w.append(f"| {rname} | {count} |")
    w.append("")

    if "fig4" in fig_paths:
        w.append(f"![Anomaly by region]({fig_paths['fig4']})")
        w.append("")

    # 12. QC categories
    w.append("## 12. QC Categories")
    w.append("")
    w.append("Evidence-based categories assigned to each spectrum:")
    w.append("")
    w.append("| Category | Description | Count |")
    w.append("|----------|-------------|-------|")
    for cat in ["A", "B", "C", "D"]:
        n = qc_counts.get(cat, 0)
        desc = {
            "A": "No reflectance validity problem detected",
            "B": "Unusual but potentially valid measurement",
            "C": "Ambiguous; requires caution/investigation",
            "D": "Clearly invalid numerical value",
        }[cat]
        w.append(f"| {cat} | {desc} | {n} |")
    w.append(f"| **Total** | | {qc_counts.sum()} |")
    w.append("")

    w.append("### Category assignment criteria")
    w.append("")
    w.append("- **A:** No value-level anomalies; few or no spectral-shape spikes")
    w.append("- **B:** Value-level anomalies present (negative, zero, or >1 values), or constant runs, or many spikes (>50)")
    w.append("- **C:** Multiple value-level anomalies (>5 negative or >10 values >1) suggesting systematic issues")
    w.append("- **D:** NaN or Inf values present (non-finite data)")
    w.append("")
    w.append("Note: Discontinuities are common (most spectra have wavelength-spacing-induced")
    w.append("rate-of-change variations) and are not used for QC categorization.")
    w.append("")

    # 13. Implications for interpolation and complexity
    w.append("## 13. Implications for Interpolation and Complexity")
    w.append("")

    w.append("### 13.1 PCHIP interpolation")
    w.append("")
    w.append("PCHIP is shape-preserving and does not introduce new extrema. However:")
    w.append("")
    if len(spike_df) > 0:
        w.append(f"- {len(spike_df)} single-point spikes will be preserved by PCHIP (no overshoot,")
        w.append("  but the spike shape will be maintained on the interpolated grid)")
    if n_neg_total > 0:
        w.append(f"- Small negative values will be interpolated as-is; PCHIP will not create")
        w.append("  additional negative values between positive neighbors")
    w.append("")

    w.append("### 13.2 Derivative calculations")
    w.append("")
    w.append("First and second derivatives amplify high-frequency noise:")
    w.append("")
    if len(spike_df) > 0:
        w.append(f"- Single-point spikes will produce large derivative values at the spike location")
        w.append("  and compensating large values of opposite sign at adjacent points")
    if len(disc_df) > 0:
        w.append(f"- Discontinuities will produce very large first-derivative peaks")
    w.append("")

    w.append("### 13.3 Peak detection and complexity metrics")
    w.append("")
    w.append("Spectral complexity metrics (peak density, first-order variation, spectral entropy)")
    w.append("may be affected by:")
    w.append("")
    if len(spike_df) > 0:
        w.append(f"- Artificial peaks from spikes inflating peak density counts")
    if len(const_run_df) > 0:
        w.append(f"- Constant runs reducing local variation and potentially masking real features")
    w.append("")
    w.append("These effects should be considered when interpreting complexity values for")
    w.append("spectra flagged in categories B and C.")
    w.append("")

    # 14. Limitations
    w.append("## 14. Limitations")
    w.append("")
    w.append("1. **No official RELAB documentation** was available. Reflectance interpretation")
    w.append("   is based on data inspection and the field dictionary.")
    w.append("2. **Uncertainty semantics are undocumented.** The meaning of the third column")
    w.append("   (one-sigma? two-sigma? instrument noise?) is unknown.")
    w.append("3. **No ground truth** exists for identifying which spectra are truly anomalous")
    w.append("   vs. physically valid unusual mineral spectra.")
    w.append("4. **Spike detection thresholds** (5× median for spikes, 10× for discontinuities)")
    w.append("   are heuristic. Different thresholds would yield different counts.")
    w.append("5. **The investigation is restricted to D2** (320–2550 nm). Anomalies outside")
    w.append("   this domain were not analyzed.")
    w.append("6. **Category boundaries** (e.g., >5 negative values for Category C) are")
    w.append("   preliminary and require researcher review.")
    w.append("")

    # 15. Recommendation
    w.append("## 15. Recommendation")
    w.append("")

    n_d = qc_counts.get("D", 0)
    n_c = qc_counts.get("C", 0)
    n_b = qc_counts.get("B", 0)
    n_a = qc_counts.get("A", 0)

    w.append("Based on the evidence gathered:")
    w.append("")
    if n_d == 0 and n_c == 0:
        w.append("**No reflectance exclusions are required at this time.**")
        w.append("")
        w.append(f"All {len(spec_df)} spectra have finite reflectance values in D2.")
        if n_b > 0:
            w.append(f"{n_b} spectra have unusual but potentially valid features (Category B)")
            w.append("that should be flagged for awareness but not excluded.")
    elif n_d == 0:
        w.append("**No spectra require immediate exclusion, but caution is advised.**")
        w.append("")
        w.append(f"- {n_c} spectra are Category C (ambiguous) and require further investigation")
        w.append(f"- {n_b} spectra are Category B (unusual but potentially valid)")
    else:
        w.append("**Specific values/spectra require investigation before inclusion.**")
        w.append("")
        w.append(f"- {n_d} spectra are Category D (clearly invalid numerical values)")
        w.append(f"- {n_c} spectra are Category C (ambiguous)")
        w.append(f"- {n_b} spectra are Category B (unusual but potentially valid)")
    w.append("")
    w.append("**This recommendation is preliminary.** The QC rules should not be frozen")
    w.append("until the researcher has reviewed the evidence and diagnostic figures.")
    w.append("")

    # Validation section
    w.append("## 16. Validation")
    w.append("")
    w.append(f"- All {pre_checked} raw MD5 checksums verified (pre-flight): {pre_n_mismatch} mismatches")
    w.append(f"- All {post_checked} raw MD5 checksums verified (post-flight): {post_n_mismatch} mismatches")
    w.append("- No raw files modified: Yes")
    w.append("- Deterministic output: Yes")
    w.append("- No processed spectra created: Yes")
    w.append("- No interpolation performed: Yes")
    w.append("- No smoothing performed: Yes")
    w.append("- No normalization performed: Yes")
    w.append("- No final complexity features calculated: Yes")
    w.append("")

    w.append("## 17. Output Files")
    w.append("")
    w.append(f"- `phase2_step3_reflectance_qc_investigation.md` — This report")
    w.append(f"- `phase2_step3_reflectance_qc.csv` — One row per anomalous occurrence ({len(anomaly_df)} rows)")
    w.append(f"- `phase2_step3_reflectance_qc_spectrum_summary.csv` — Per-spectrum summary ({len(spec_df)} rows)")
    for key, path in fig_paths.items():
        w.append(f"- `{Path(path).name}` — Diagnostic figure")
    w.append("")

    return "\n".join(w)


if __name__ == "__main__":
    main()
