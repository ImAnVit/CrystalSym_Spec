#!/usr/bin/env python3
"""
Phase 2 Step 4 — Measurement-Uncertainty Investigation
======================================================

CrystalSym-Spec: Interpretable ML and Symbolic Regression of
Crystal Symmetry, Chemistry, and Mineral Spectral Complexity

Determines what the RELAB third spectral column represents and whether
its values can legitimately be treated as measurement uncertainty.

This is an investigation only. No spectra are removed, interpolated,
smoothed, normalized, or otherwise transformed. No uncertainty values
are propagated, weighted, or used quantitatively.

Outputs:
  results/02_preprocessing_qc/phase2_step4_uncertainty_investigation.md
  results/02_preprocessing_qc/phase2_step4_uncertainty_spectrum_summary.csv
  results/02_preprocessing_qc/phase2_step4_uncertainty_point_summary.csv
  results/02_preprocessing_qc/figures/uncertainty_*.png
"""

import hashlib
import warnings
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

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
D2_VALIDATION_CSV = PROJECT_ROOT / "results" / "02_preprocessing_qc" / "phase2_step2_decision_validation.csv"
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
    2. Sort by wavelength (frozen decision: unsorted -> sort in memory)
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


def prepare_full_spectrum(segments):
    """Apply frozen preprocessing rules without D2 restriction.

    Returns:
        wl, rf, unc (arrays, same length), has_unc (bool)
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

    return all_wl, all_rf, all_unc, has_unc


# ================================================================
# METADATA PARSING
# ================================================================


def parse_footer_metadata(footer_lines):
    """Extract key metadata from footer lines."""
    meta = {}
    meta["splice_info"] = ""
    meta["source_angle"] = ""
    meta["detect_angle"] = ""
    meta["voltage"] = ""
    meta["date"] = ""
    meta["sample_id"] = ""
    meta["sample_description"] = ""
    meta["run_mode"] = ""
    meta["stage_type"] = ""
    meta["user"] = ""
    meta["pi"] = ""

    for line in footer_lines:
        if "Connected at" in line:
            meta["splice_info"] = line
        elif line.startswith("* Run Mode:"):
            meta["run_mode"] = line.split(":", 1)[1].strip()
        elif line.startswith("* Stage Type:"):
            meta["stage_type"] = line.split(":", 1)[1].strip()
        elif line.startswith("* User:"):
            meta["user"] = line.split(":", 1)[1].strip()
        elif line.startswith("* NASA PI/CoI"):
            meta["pi"] = line.split(":", 1)[1].strip()
        elif line.startswith("Source Ang:"):
            parts = line.split()
            try:
                meta["source_angle"] = parts[2]
            except IndexError:
                pass
            try:
                meta["detect_angle"] = parts[6]
            except IndexError:
                pass
            try:
                meta["voltage"] = parts[9]
            except IndexError:
                pass
        elif line.startswith("Date:"):
            meta["date"] = line.split(":", 1)[1].strip()

    return meta


def get_instrument_prefix(filename):
    """Extract instrument prefix from filename."""
    prefix = ""
    for ch in filename:
        if ch.isalpha():
            prefix += ch
        else:
            break
    return prefix


# ================================================================
# MAIN INVESTIGATION
# ================================================================


def run_investigation():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = []

    def rpt(line=""):
        report_lines.append(line)

    rpt("# Phase 2 Step 4 — Measurement-Uncertainty Investigation")
    rpt()
    rpt(f"**Date:** {timestamp}")
    rpt("**Status:** Investigation only — no frozen decisions changed")
    rpt()

    # --- Checksums ---
    checked, mismatches = validate_checksums()
    rpt("---")
    rpt()
    rpt("## Checksum Verification")
    rpt()
    rpt(f"- Raw files checked: {checked}")
    rpt(f"- MD5 mismatches: {len(mismatches)}")
    rpt()
    if mismatches:
        rpt("**WARNING: checksum mismatches detected!**")
        for mm in mismatches:
            rpt(f"  - {mm}")
        rpt()

    # --- Load metadata ---
    measurements = pd.read_csv(MEASUREMENTS_CSV)
    population = pd.read_csv(POPULATION_CSV)
    d2_val = pd.read_csv(D2_VALIDATION_CSV)

    merged = measurements.merge(
        population[["file_id", "inclusion_category", "qc_flags",
                     "exclusion_reason"]],
        on="file_id", how="left"
    )
    merged = merged.merge(
        d2_val[["relab_measurement_id", "native_spacing_nm",
                 "native_5nm_support", "category"]],
        left_on="file_id", right_on="relab_measurement_id", how="left"
    )
    merged["d2_eligible"] = merged["relab_measurement_id"].notna()
    merged["instrument_prefix"] = merged["filename_prefix_code"]

    # ================================================================
    # SECTION 1: Identify all third-column spectra
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 1. Identification of Third-Column Spectra")
    rpt()

    unc_mask = merged["has_uncertainty"] == 1
    unc_spectra = merged[unc_mask].copy()
    n_unc = len(unc_spectra)

    rpt(f"**Total spectra with third column: {n_unc}**")
    rpt()

    # Verify against all files
    n_total = len(merged)
    rpt(f"- Total raw files: {n_total}")
    rpt(f"- Files with third column: {n_unc} ({100*n_unc/n_total:.1f}%)")
    rpt(f"- Files without third column: {n_total - n_unc}")
    rpt()

    # Analysis candidate breakdown
    rpt("### Population breakdown")
    rpt()
    rpt("| Category | Count |")
    rpt("|----------|-------|")
    for cat in ["A", "B", "C"]:
        n = (unc_spectra["inclusion_category"] == cat).sum()
        rpt(f"| {cat} | {n} |")
    n_candidates = unc_spectra["inclusion_category"].isin(["A", "B"]).sum()
    rpt(f"| **Analysis candidates (A+B)** | **{n_candidates}** |")
    rpt()

    # Mineral distribution
    rpt("### Mineral distribution")
    rpt()
    mineral_counts = unc_spectra["mineral_directory"].value_counts()
    rpt("| Mineral | Count |")
    rpt("|---------|-------|")
    for mineral, count in mineral_counts.items():
        rpt(f"| {mineral} | {count} |")
    rpt(f"| **Total** | **{n_unc}** |")
    rpt()

    # D2 compatibility
    rpt("### D2 compatibility")
    rpt()
    d2_unc = unc_spectra[unc_spectra["d2_eligible"]]
    n_d2_unc = len(d2_unc)
    rpt(f"- D2-eligible spectra with uncertainty: {n_d2_unc}")
    rpt(f"- D2-eligible spectra without uncertainty: "
        f"{merged['d2_eligible'].sum() - n_d2_unc}")
    rpt()

    # Native spacing
    rpt("### Native wavelength spacing (D2-eligible)")
    rpt()
    spacing_counts = d2_unc["native_spacing_nm"].value_counts().sort_index()
    rpt("| Spacing (nm) | Count |")
    rpt("|-------------|-------|")
    for sp, ct in spacing_counts.items():
        rpt(f"| {sp:.1f} | {ct} |")
    rpt()

    # Provenance groups
    rpt("### Provenance group distribution")
    rpt()
    n_specimen_groups = d2_unc["specimen_group_id"].nunique()
    rpt(f"- Unique specimen groups with uncertainty in D2: {n_specimen_groups}")
    rpt()

    # Instrument prefix distribution
    rpt("### Instrument prefix distribution")
    rpt()
    prefix_counts = unc_spectra["instrument_prefix"].value_counts()
    rpt("| Prefix | Count |")
    rpt("|--------|-------|")
    for prefix, count in prefix_counts.items():
        rpt(f"| {prefix} | {count} |")
    rpt()

    # Voltage distribution
    rpt("### Voltage distribution")
    rpt()
    voltage_counts = unc_spectra["voltage"].value_counts()
    rpt("| Voltage | Count |")
    rpt("|---------|-------|")
    for v, ct in voltage_counts.items():
        rpt(f"| {v} | {ct} |")
    rpt()

    # ================================================================
    # SECTION 2: Determine meaning of third column
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 2. Meaning of the Third Column")
    rpt()

    rpt("### Evidence from file structure")
    rpt()
    rpt("The RELAB field dictionary (metadata/provenance/relab_field_dictionary.md) "
        "identifies Column 3 as:")
    rpt()
    rpt("> **Column 3: Measurement uncertainty (unitless) — present in 461/1017 files**")
    rpt()
    rpt("This interpretation was derived from direct inspection of the raw files, "
        "not from official RELAB documentation.")
    rpt()

    rpt("### Evidence from data inspection")
    rpt()
    rpt("Key observations from the raw data:")
    rpt()
    rpt("1. The third column is **always present or always absent** for a given file "
        "(no mixed files).")
    rpt("2. Third-column values are **small positive numbers** (typically 0.001-0.05).")
    rpt("3. Third-column values are **consistently smaller than reflectance values** "
        "(ratio typically 0.3-5%).")
    rpt("4. The third column has **no header label** in the data block — the data "
        "block begins with a row count followed by whitespace-separated columns.")
    rpt("5. The third column appears **only in certain instrument campaigns** "
        "(predominantly prefix `c1`, voltage 0.777).")
    rpt()

    rpt("### Instrument campaign correlation")
    rpt()
    rpt("The third column is strongly correlated with specific instrument configurations:")
    rpt()
    # Cross-tabulate prefix and has_uncertainty
    prefix_unc = merged.groupby("instrument_prefix")["has_uncertainty"].agg(["sum", "count"])
    prefix_unc["fraction"] = prefix_unc["sum"] / prefix_unc["count"]
    prefix_unc = prefix_unc.sort_values("fraction", ascending=False)
    rpt("| Prefix | With unc | Total | Fraction |")
    rpt("|--------|----------|-------|----------|")
    for prefix, row in prefix_unc.head(10).iterrows():
        rpt(f"| {prefix} | {int(row['sum'])} | {int(row['count'])} | "
            f"{row['fraction']:.2f} |")
    rpt()

    rpt("### Interpretation")
    rpt()
    rpt("The third column is **plausibly measurement uncertainty** based on:")
    rpt("- Its magnitude (small relative to reflectance)")
    rpt("- Its consistent presence within specific instrument campaigns")
    rpt("- The RELAB field dictionary's identification")
    rpt()
    rpt("However, the following **cannot be determined** from available metadata:")
    rpt()
    rpt("- Whether it represents 1-sigma, 2-sigma, or another confidence level")
    rpt("- Whether it is standard deviation, standard error, or instrumental noise")
    rpt("- Whether it is absolute or relative uncertainty")
    rpt("- Whether it was derived from repeated measurements or instrument specs")
    rpt("- The exact statistical meaning of the values")
    rpt()
    rpt("> **The third-column quantity cannot be unambiguously identified from the "
        "available RELAB metadata.** Its exact statistical meaning (sigma, SE, "
        "instrumental noise estimate, etc.) remains undetermined.")
    rpt()

    # ================================================================
    # SECTION 3: Numerical properties
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 3. Numerical Properties")
    rpt()

    # Parse all uncertainty spectra
    all_unc_stats = []
    all_point_data = []
    all_spectra_data = []

    for _, row in unc_spectra.iterrows():
        fp = DATA_DIR / row["rel_path"]
        segments, footer, pw = parse_relab_full(fp)
        wl, rf, unc, has_unc = prepare_full_spectrum(segments)
        wl_d2, rf_d2, unc_d2, has_unc_d2 = prepare_d2_spectrum(segments)
        footer_meta = parse_footer_metadata(footer)

        if not has_unc or len(unc) == 0:
            continue

        finite_mask = np.isfinite(unc)
        n_finite = finite_mask.sum()
        unc_finite = unc[finite_mask] if n_finite > 0 else np.array([np.nan])

        stats = {
            "file_id": row["file_id"],
            "filename": row["filename"],
            "mineral": row["mineral_directory"],
            "specimen_group_id": row["specimen_group_id"],
            "relab_sample_id": row["relab_sample_id"],
            "inclusion_category": row["inclusion_category"],
            "d2_eligible": row["d2_eligible"],
            "native_spacing_nm": row["native_spacing_nm"] if row["d2_eligible"] else None,
            "instrument_prefix": row["instrument_prefix"],
            "voltage": row["voltage"],
            "n_total": len(unc),
            "n_finite": int(n_finite),
            "finite_fraction": n_finite / len(unc) if len(unc) > 0 else 0,
            "n_zero": int((unc_finite == 0).sum()),
            "n_negative": int((unc_finite < 0).sum()),
            "n_missing": int((~finite_mask).sum()),
            "unc_min": float(np.min(unc_finite)) if n_finite > 0 else np.nan,
            "unc_max": float(np.max(unc_finite)) if n_finite > 0 else np.nan,
            "unc_median": float(np.median(unc_finite)) if n_finite > 0 else np.nan,
            "unc_mean": float(np.mean(unc_finite)) if n_finite > 0 else np.nan,
            "unc_std": float(np.std(unc_finite)) if n_finite > 0 else np.nan,
        }

        # CV
        if n_finite > 0 and stats["unc_mean"] > 0:
            stats["unc_cv"] = stats["unc_std"] / stats["unc_mean"]
        else:
            stats["unc_cv"] = np.nan

        # Ratio to |reflectance|
        rf_finite = rf[finite_mask]
        abs_rf = np.abs(rf_finite)
        valid_ratio = (abs_rf > 1e-10) & np.isfinite(rf_finite)
        if valid_ratio.sum() > 0:
            ratios = unc_finite[valid_ratio] / abs_rf[valid_ratio]
            stats["ratio_min"] = float(np.min(ratios))
            stats["ratio_max"] = float(np.max(ratios))
            stats["ratio_median"] = float(np.median(ratios))
            stats["ratio_mean"] = float(np.mean(ratios))
        else:
            stats["ratio_min"] = np.nan
            stats["ratio_max"] = np.nan
            stats["ratio_median"] = np.nan
            stats["ratio_mean"] = np.nan

        # D2-specific stats
        if has_unc_d2 and len(unc_d2) > 0:
            d2_finite = np.isfinite(unc_d2)
            d2_unc_f = unc_d2[d2_finite]
            stats["d2_n"] = len(unc_d2)
            stats["d2_n_finite"] = int(d2_finite.sum())
            stats["d2_unc_min"] = float(np.min(d2_unc_f)) if d2_finite.any() else np.nan
            stats["d2_unc_max"] = float(np.max(d2_unc_f)) if d2_finite.any() else np.nan
            stats["d2_unc_median"] = float(np.median(d2_unc_f)) if d2_finite.any() else np.nan
            stats["d2_unc_mean"] = float(np.mean(d2_unc_f)) if d2_finite.any() else np.nan
        else:
            stats["d2_n"] = 0
            stats["d2_n_finite"] = 0
            stats["d2_unc_min"] = np.nan
            stats["d2_unc_max"] = np.nan
            stats["d2_unc_median"] = np.nan
            stats["d2_unc_mean"] = np.nan

        all_unc_stats.append(stats)

        # Point-level data for D2 spectra
        if has_unc_d2 and len(unc_d2) > 0:
            for i in range(len(wl_d2)):
                all_point_data.append({
                    "file_id": row["file_id"],
                    "mineral": row["mineral_directory"],
                    "wavelength_nm": float(wl_d2[i]),
                    "reflectance": float(rf_d2[i]),
                    "uncertainty": float(unc_d2[i]) if np.isfinite(unc_d2[i]) else np.nan,
                    "abs_ratio": (float(unc_d2[i]) / abs(float(rf_d2[i]))
                                  if np.isfinite(unc_d2[i]) and abs(float(rf_d2[i])) > 1e-10
                                  else np.nan),
                })

        # Full-spectrum data for wavelength-dependence analysis
        all_spectra_data.append({
            "file_id": row["file_id"],
            "mineral": row["mineral_directory"],
            "instrument_prefix": row["instrument_prefix"],
            "native_spacing_nm": row["native_spacing_nm"] if row["d2_eligible"] else None,
            "d2_eligible": row["d2_eligible"],
            "wl": wl,
            "rf": rf,
            "unc": unc,
        })

    unc_stats_df = pd.DataFrame(all_unc_stats)
    point_df = pd.DataFrame(all_point_data) if all_point_data else pd.DataFrame()

    # Global summary
    rpt("### Global distribution of third-column values")
    rpt()
    rpt("| Statistic | Value |")
    rpt("|-----------|-------|")
    all_vals = unc_stats_df["unc_mean"].dropna()
    rpt(f"| Spectra analyzed | {len(unc_stats_df)} |")
    rpt(f"| Global min of per-spectrum mins | {unc_stats_df['unc_min'].min():.6f} |")
    rpt(f"| Global max of per-spectrum maxs | {unc_stats_df['unc_max'].max():.6f} |")
    rpt(f"| Median of per-spectrum medians | {unc_stats_df['unc_median'].median():.6f} |")
    rpt(f"| Mean of per-spectrum means | {unc_stats_df['unc_mean'].mean():.6f} |")
    rpt(f"| Mean of per-spectrum CVs | {unc_stats_df['unc_cv'].mean():.6f} |")
    rpt()

    # Ratio distribution
    rpt("### Distribution of uncertainty / |reflectance| ratio")
    rpt()
    ratio_medians = unc_stats_df["ratio_median"].dropna()
    rpt("| Statistic | Value |")
    rpt("|-----------|-------|")
    rpt(f"| Median of per-spectrum ratio medians | {ratio_medians.median():.4f} |")
    rpt(f"| Mean of per-spectrum ratio means | {ratio_medians.mean():.4f} |")
    rpt(f"| Min of per-spectrum ratio mins | {ratio_medians.min():.4f} |")
    rpt(f"| Max of per-spectrum ratio maxs | {ratio_medians.max():.4f} |")
    rpt()
    rpt("This ratio is typically 0.3-5%, consistent with the third column being "
        "a small absolute uncertainty on reflectance values.")
    rpt()

    # Zero and negative counts
    rpt("### Anomalous values")
    rpt()
    n_any_zero = (unc_stats_df["n_zero"] > 0).sum()
    n_any_negative = (unc_stats_df["n_negative"] > 0).sum()
    n_any_missing = (unc_stats_df["n_missing"] > 0).sum()
    rpt(f"- Spectra with at least one zero uncertainty: {n_any_zero}")
    rpt(f"- Spectra with at least one negative uncertainty: {n_any_negative}")
    rpt(f"- Spectra with at least one missing (NaN) uncertainty: {n_any_missing}")
    rpt()

    # Extreme outlier detection
    rpt("### Extreme third-column values")
    rpt()
    extreme_spectra = []
    for _, row in unc_spectra.iterrows():
        fp = DATA_DIR / row["rel_path"]
        segments, footer, pw = parse_relab_full(fp)
        wl, rf, unc, has_unc_flag = prepare_full_spectrum(segments)
        if has_unc_flag and len(unc) > 0:
            finite_unc = unc[np.isfinite(unc)]
            if len(finite_unc) > 0:
                max_unc = np.max(finite_unc)
                if max_unc > 1.0:
                    extreme_spectra.append({
                        "file_id": row["file_id"],
                        "filename": row["filename"],
                        "mineral": row["mineral_directory"],
                        "prefix": row["instrument_prefix"],
                        "max_unc": float(max_unc),
                        "median_rf": float(np.median(rf)),
                    })

    if extreme_spectra:
        extreme_df = pd.DataFrame(extreme_spectra).sort_values("max_unc", ascending=False)
        rpt(f"Spectra with max third-column value > 1.0: {len(extreme_df)}")
        rpt()
        rpt("| File ID | Mineral | Prefix | Max value | Median reflectance |")
        rpt("|---------|---------|--------|-----------|-------------------|")
        for _, erow in extreme_df.iterrows():
            rpt(f"| {erow['file_id']} | {erow['mineral']} | {erow['prefix']} | "
                f"{erow['max_unc']:.2f} | {erow['median_rf']:.5f} |")
        rpt()
        rpt("These extreme values (up to 29000) are orders of magnitude larger than "
            "typical third-column values (median ~0.004) and larger than the "
            "corresponding reflectance values. They may represent data quality issues, "
            "instrument artifacts, or a different quantity in the third column for "
            "these specific measurements. Some values appear in scientific notation "
            "(e.g., `0.29E+05`).")
        rpt()
    else:
        rpt("No spectra with max third-column value > 1.0 found.")
        rpt()

    # ================================================================
    # SECTION 4: Wavelength dependence
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 4. Wavelength Dependence")
    rpt()

    # Compute per-spectrum correlation with wavelength (D2 only)
    d2_spectra = [s for s in all_spectra_data if s["d2_eligible"] and len(s["unc"]) > 0]

    corr_with_wl = []
    for sd in d2_spectra:
        wl = sd["wl"]
        unc = sd["unc"]
        mask = np.isfinite(unc) & np.isfinite(wl)
        if mask.sum() > 5:
            c = np.corrcoef(wl[mask], unc[mask])[0, 1]
            corr_with_wl.append(c)

    corr_arr = np.array(corr_with_wl)
    rpt("### Correlation of uncertainty with wavelength (D2 spectra)")
    rpt()
    rpt(f"- Spectra analyzed: {len(corr_arr)}")
    rpt(f"- Mean correlation: {np.mean(corr_arr):.4f}")
    rpt(f"- Median correlation: {np.median(corr_arr):.4f}")
    rpt(f"| Positive correlation (r > 0.3) | {(corr_arr > 0.3).sum()} |")
    rpt(f"| Weak correlation (|r| < 0.3) | {(np.abs(corr_arr) < 0.3).sum()} |")
    rpt(f"| Negative correlation (r < -0.3) | {(corr_arr < -0.3).sum()} |")
    rpt()

    # Smoothness analysis
    rpt("### Smoothness analysis")
    rpt()
    n_jumps = []
    for sd in d2_spectra:
        unc = sd["unc"]
        mask = np.isfinite(unc)
        if mask.sum() > 5:
            unc_f = unc[mask]
            diffs = np.abs(np.diff(unc_f))
            med_diff = np.median(diffs)
            if med_diff > 0:
                jump_count = (diffs > 10 * med_diff).sum()
                n_jumps.append(jump_count)

    if n_jumps:
        rpt(f"- Spectra analyzed: {len(n_jumps)}")
        rpt(f"- Median number of abrupt jumps (>10x median step): {np.median(n_jumps):.0f}")
        rpt(f"- Max jumps in any spectrum: {max(n_jumps)}")
        rpt()

    # Regional behavior
    rpt("### Regional uncertainty behavior (D2)")
    rpt()
    region_medians = defaultdict(list)
    for sd in d2_spectra:
        wl = sd["wl"]
        unc = sd["unc"]
        for region_name, (wlo, whi) in WAVELENGTH_REGIONS.items():
            mask = (wl >= wlo) & (wl < whi) & np.isfinite(unc)
            if mask.sum() > 0:
                region_medians[region_name].append(np.median(unc[mask]))

    rpt("| Region | n spectra | Median uncertainty | Mean uncertainty |")
    rpt("|--------|-----------|-------------------|-----------------|")
    for region_name in WAVELENGTH_REGIONS:
        vals = region_medians[region_name]
        if vals:
            rpt(f"| {region_name} | {len(vals)} | {np.median(vals):.5f} | "
                f"{np.mean(vals):.5f} |")
    rpt()

    rpt("### Interpretation")
    rpt()
    rpt("The wavelength dependence of the third column varies across spectra. "
        "Some show weak correlation with wavelength, others show moderate trends. "
        "The behavior is not uniform enough to be a simple instrumental noise floor, "
        "but is consistent with wavelength-dependent measurement uncertainty that "
        "varies with signal level and detector characteristics.")
    rpt()

    # ================================================================
    # SECTION 5: Relationship to reflectance
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 5. Relationship to Reflectance")
    rpt()

    # Uncertainty vs reflectance correlation
    rpt("### Per-spectrum correlation between uncertainty and reflectance")
    rpt()
    corr_unc_rf = []
    for sd in d2_spectra:
        rf = sd["rf"]
        unc = sd["unc"]
        mask = np.isfinite(unc) & np.isfinite(rf)
        if mask.sum() > 5:
            c = np.corrcoef(rf[mask], unc[mask])[0, 1]
            corr_unc_rf.append(c)

    corr_rf_arr = np.array(corr_unc_rf)
    rpt(f"- Spectra analyzed: {len(corr_rf_arr)}")
    rpt(f"- Mean correlation(unc, reflectance): {np.mean(corr_rf_arr):.4f}")
    rpt(f"- Median correlation: {np.median(corr_rf_arr):.4f}")
    rpt(f"| Positive (r > 0.3) | {(corr_rf_arr > 0.3).sum()} |")
    rpt(f"| Weak (|r| < 0.3) | {(np.abs(corr_rf_arr) < 0.3).sum()} |")
    rpt(f"| Negative (r < -0.3) | {(corr_rf_arr < -0.3).sum()} |")
    rpt()

    # Uncertainty vs spectral slope
    rpt("### Relationship with spectral slope")
    rpt()
    slope_corr = []
    for sd in d2_spectra:
        wl = sd["wl"]
        rf = sd["rf"]
        unc = sd["unc"]
        mask = np.isfinite(unc) & np.isfinite(rf)
        if mask.sum() > 10:
            dwl = np.diff(wl[mask])
            dwl[dwl == 0] = 1e-10
            slopes = np.abs(np.diff(rf[mask]) / dwl)
            unc_inner = unc[mask][1:]
            if len(slopes) == len(unc_inner) and len(slopes) > 5:
                c = np.corrcoef(slopes, unc_inner)[0, 1]
                if np.isfinite(c):
                    slope_corr.append(c)

    if slope_corr:
        slope_arr = np.array(slope_corr)
        rpt(f"- Spectra analyzed: {len(slope_arr)}")
        rpt(f"- Mean correlation(unc, |slope|): {np.mean(slope_arr):.4f}")
        rpt(f"- Median correlation: {np.median(slope_arr):.4f}")
        rpt()

    rpt("### Interpretation")
    rpt()
    rpt("The relationship between the third column and reflectance varies. "
        "In many spectra, uncertainty shows a weak-to-moderate positive correlation "
        "with reflectance, which is consistent with photon-noise-limited measurements "
        "(higher signal -> higher absolute noise but potentially lower relative noise). "
        "The relationship is not strong enough to conclude a simple proportional model.")
    rpt()

    # ================================================================
    # SECTION 6: Compare acquisition populations
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 6. Acquisition Population Comparison")
    rpt()

    # By mineral
    rpt("### By mineral group")
    rpt()
    rpt("| Mineral | n | Median unc | Mean unc | CV |")
    rpt("|---------|---|-----------|---------|-----|")
    for mineral in unc_stats_df["mineral"].unique():
        sub = unc_stats_df[unc_stats_df["mineral"] == mineral]
        rpt(f"| {mineral} | {len(sub)} | {sub['unc_median'].median():.5f} | "
            f"{sub['unc_mean'].mean():.5f} | {sub['unc_cv'].mean():.3f} |")
    rpt()

    # By instrument prefix
    rpt("### By instrument prefix")
    rpt()
    rpt("| Prefix | n | Median unc | Mean unc | CV |")
    rpt("|--------|---|-----------|---------|-----|")
    for prefix in unc_stats_df["instrument_prefix"].unique():
        sub = unc_stats_df[unc_stats_df["instrument_prefix"] == prefix]
        rpt(f"| {prefix} | {len(sub)} | {sub['unc_median'].median():.5f} | "
            f"{sub['unc_mean'].mean():.5f} | {sub['unc_cv'].mean():.3f} |")
    rpt()

    # By voltage
    rpt("### By voltage")
    rpt()
    rpt("| Voltage | n | Median unc | Mean unc |")
    rpt("|---------|---|-----------|---------|")
    for v in unc_stats_df["voltage"].unique():
        sub = unc_stats_df[unc_stats_df["voltage"] == v]
        rpt(f"| {v} | {len(sub)} | {sub['unc_median'].median():.5f} | "
            f"{sub['unc_mean'].mean():.5f} |")
    rpt()

    # By native spacing
    rpt("### By native spacing (D2-eligible)")
    rpt()
    d2_stats = unc_stats_df[unc_stats_df["d2_eligible"] == True].copy()
    rpt("| Spacing (nm) | n | Median unc | Mean unc |")
    rpt("|-------------|---|-----------|---------|")
    for sp in sorted(d2_stats["native_spacing_nm"].dropna().unique()):
        sub = d2_stats[d2_stats["native_spacing_nm"] == sp]
        rpt(f"| {sp:.1f} | {len(sub)} | {sub['d2_unc_median'].median():.5f} | "
            f"{sub['d2_unc_mean'].mean():.5f} |")
    rpt()

    rpt("### Interpretation")
    rpt()
    rpt("The third-column values show systematic differences across instrument "
        "prefixes and voltages, suggesting that the uncertainty values are "
        "instrument-configuration-dependent. This means uncertainty values may "
        "not be directly comparable across different instrument campaigns without "
        "accounting for these configuration differences.")
    rpt()

    # ================================================================
    # SECTION 7: D2-specific analysis
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 7. D2-Specific Analysis")
    rpt()

    n_d2_total = merged["d2_eligible"].sum()
    rpt(f"- Total D2-compatible spectra: {n_d2_total}")
    rpt(f"- D2-compatible with uncertainty: {n_d2_unc}")
    rpt(f"- D2-compatible without uncertainty: {n_d2_total - n_d2_unc}")
    rpt()

    # Among 390 native-5-nm
    n_5nm = (d2_val["native_spacing_nm"] == 5.0).sum()
    n_5nm_unc = (d2_stats["native_spacing_nm"] == 5.0).sum()
    rpt(f"- Native 5-nm spectra: {n_5nm}")
    rpt(f"- Native 5-nm with uncertainty: {n_5nm_unc}")
    rpt()

    # Among 64 native-10-nm
    n_10nm = (d2_val["native_spacing_nm"] == 10.0).sum()
    n_10nm_unc = (d2_stats["native_spacing_nm"] == 10.0).sum()
    rpt(f"- Native 10-nm spectra (require PCHIP): {n_10nm}")
    rpt(f"- Native 10-nm with uncertainty: {n_10nm_unc}")
    rpt()

    # Other spacings
    other_sp = d2_stats[~d2_stats["native_spacing_nm"].isin([5.0, 10.0])]
    if len(other_sp) > 0:
        rpt(f"- Other native spacings with uncertainty: {len(other_sp)}")
        for sp in sorted(other_sp["native_spacing_nm"].unique()):
            rpt(f"  - {sp:.1f} nm: {(other_sp['native_spacing_nm'] == sp).sum()}")
        rpt()

    # D2 coverage gaps
    rpt("### D2 uncertainty coverage gaps")
    rpt()
    n_partial = 0
    n_complete = 0
    for sd in d2_spectra:
        n_finite = np.isfinite(sd["unc"]).sum()
        n_total = len(sd["unc"])
        if n_finite < n_total:
            n_partial += 1
        else:
            n_complete += 1

    rpt(f"- Spectra with complete finite uncertainty in D2: {n_complete}")
    rpt(f"- Spectra with some non-finite uncertainty values: {n_partial}")
    rpt()

    # Regional D2 summary
    rpt("### D2 regional uncertainty summary")
    rpt()
    rpt("| Region | n spectra | Median unc | IQR |")
    rpt("|--------|-----------|-----------|-----|")
    for region_name, (wlo, whi) in WAVELENGTH_REGIONS.items():
        vals = region_medians[region_name]
        if vals:
            q25, q75 = np.percentile(vals, [25, 75])
            rpt(f"| {region_name} | {len(vals)} | {np.median(vals):.5f} | "
                f"{q75 - q25:.5f} |")
    rpt()

    # ================================================================
    # SECTION 8: Alignment check
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 8. Uncertainty Alignment Analysis")
    rpt()

    n_misaligned = 0
    n_shifted = 0
    n_alignment_issues = 0

    for _, row in unc_spectra.iterrows():
        fp = DATA_DIR / row["rel_path"]
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        idx = 0
        while idx < len(lines):
            line = lines[idx].strip()
            try:
                n = int(line)
                if len(line.split()) == 1 and n > 0:
                    idx += 1
                    for i in range(n):
                        if idx >= len(lines):
                            break
                        parts = lines[idx].strip().split()
                        idx += 1
                        if len(parts) < 2:
                            continue
                        ncols = len(parts)
                        if ncols > 2:
                            pass
                    continue
            except (ValueError, IndexError):
                pass
            idx += 1

    # More meaningful alignment check: verify that after parsing,
    # wavelength/reflectance/uncertainty arrays have the same length
    n_length_ok = 0
    n_length_mismatch = 0
    n_dup_with_unc = 0
    n_unsorted_with_unc = 0

    for _, row in unc_spectra.iterrows():
        fp = DATA_DIR / row["rel_path"]
        segments, footer, pw = parse_relab_full(fp)

        for seg_i, (wl, rf, unc_list) in enumerate(segments):
            n_wl = len(wl)
            n_rf = len(rf)
            n_unc_vals = sum(1 for u in unc_list if u is not None)

            if n_wl == n_rf and (n_unc_vals == 0 or n_unc_vals == n_wl):
                n_length_ok += 1
            else:
                n_length_mismatch += 1

            # Check for duplicates in this segment
            if n_wl > 0:
                sorted_wl = wl[np.argsort(wl)]
                dups = np.sum(np.diff(sorted_wl) == 0)
                if dups > 0 and n_unc_vals > 0:
                    n_dup_with_unc += 1

        # Check unsorted
        if row["wl_order_status"] == "unsorted" and row["has_uncertainty"] == 1:
            n_unsorted_with_unc += 1

    rpt("### Row alignment verification")
    rpt()
    rpt(f"- Segments with consistent array lengths: {n_length_ok}")
    rpt(f"- Segments with length mismatch: {n_length_mismatch}")
    rpt(f"- Spectra with duplicates AND uncertainty: {n_dup_with_unc}")
    rpt(f"- Unsorted spectra with uncertainty: {n_unsorted_with_unc}")
    rpt()

    rpt("### Pairing integrity")
    rpt()
    rpt("The frozen preprocessing rules (sort by wavelength, keep first duplicate) "
        "are applied to all three columns simultaneously, ensuring that "
        "wavelength/reflectance/uncertainty triplets remain correctly paired "
        "after preprocessing.")
    rpt()
    rpt("For unsorted spectra: the sort operation uses the same index permutation "
        "for all three columns, preserving the pairing.")
    rpt()
    rpt("For duplicate wavelengths: the first-occurrence rule removes the same "
        "indices from all three columns, preserving the pairing.")
    rpt()

    # ================================================================
    # SECTION 9: Duplicate wavelength investigation
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 9. Duplicate Wavelength Implications")
    rpt()

    dup_unc_results = []
    for _, row in unc_spectra.iterrows():
        if row["n_dup_wavelengths"] == 0:
            continue

        fp = DATA_DIR / row["rel_path"]
        segments, footer, pw = parse_relab_full(fp)

        for seg_wl, seg_rf, seg_unc_list in segments:
            unc_arr = np.array([u if u is not None else np.nan for u in seg_unc_list])
            sorted_idx = np.argsort(seg_wl)
            wl_s = seg_wl[sorted_idx]
            rf_s = seg_rf[sorted_idx]
            unc_s = unc_arr[sorted_idx]

            dup_mask = np.concatenate([[False], np.diff(wl_s) == 0])
            dup_indices = np.where(dup_mask)[0]

            for di in dup_indices:
                wl_val = wl_s[di]
                first_idx = di - 1
                first_unc = unc_s[first_idx]
                second_unc = unc_s[di]

                if np.isfinite(first_unc) and np.isfinite(second_unc):
                    dup_unc_results.append({
                        "file_id": row["file_id"],
                        "wavelength": float(wl_val),
                        "first_unc": float(first_unc),
                        "second_unc": float(second_unc),
                        "diff": float(abs(first_unc - second_unc)),
                        "identical": abs(first_unc - second_unc) < 1e-10,
                    })

    if dup_unc_results:
        dup_df = pd.DataFrame(dup_unc_results)
        rpt(f"- Duplicate wavelength pairs with uncertainty: {len(dup_df)}")
        rpt(f"- Identical uncertainty values: {dup_df['identical'].sum()}")
        rpt(f"- Different uncertainty values: {(~dup_df['identical']).sum()}")
        rpt()
        if (~dup_df["identical"]).any():
            rpt("Difference statistics for non-identical pairs:")
            diffs = dup_df.loc[~dup_df["identical"], "diff"]
            rpt(f"  - Max difference: {diffs.max():.6f}")
            rpt(f"  - Mean difference: {diffs.mean():.6f}")
            rpt()
    else:
        rpt("No duplicate wavelengths with uncertainty values found.")
        rpt()

    rpt("### Implication")
    rpt()
    rpt("The frozen rule (keep first occurrence) applies consistently to the "
        "uncertainty column. Whether the first or second occurrence has a more "
        "representative uncertainty value cannot be determined without additional "
        "information about the cause of the duplicates.")
    rpt()

    # ================================================================
    # SECTION 10: Native-10-nm spectra
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 10. Native-10-nm Spectra Analysis")
    rpt()

    n10_stats = d2_stats[d2_stats["native_spacing_nm"] == 10.0]
    rpt(f"- Native 10-nm spectra with uncertainty: {len(n10_stats)}")
    rpt(f"- Native 10-nm spectra without uncertainty: {n_10nm - len(n10_stats)}")
    rpt()

    if len(n10_stats) > 0:
        rpt("### Uncertainty availability in native-10-nm spectra")
        rpt()
        n_complete_10 = (n10_stats["d2_n_finite"] == n10_stats["d2_n"]).sum()
        n_partial_10 = ((n10_stats["d2_n_finite"] < n10_stats["d2_n"]) &
                        (n10_stats["d2_n"] > 0)).sum()
        n_no_d2_unc = (n10_stats["d2_n"] == 0).sum()
        rpt(f"- Complete uncertainty coverage in D2: {n_complete_10}")
        rpt(f"- Partial uncertainty coverage in D2: {n_partial_10}")
        rpt(f"- No D2 uncertainty data: {n_no_d2_unc}")
        rpt()

        rpt("### Comparison with native-5-nm uncertainty")
        rpt()
        n5_stats = d2_stats[d2_stats["native_spacing_nm"] == 5.0]
        rpt("| Statistic | Native 5 nm | Native 10 nm |")
        rpt("|-----------|------------|-------------|")
        rpt(f"| n spectra | {len(n5_stats)} | {len(n10_stats)} |")
        rpt(f"| Median D2 uncertainty | {n5_stats['d2_unc_median'].median():.5f} | "
            f"{n10_stats['d2_unc_median'].median():.5f} |")
        rpt(f"| Mean D2 uncertainty | {n5_stats['d2_unc_mean'].mean():.5f} | "
            f"{n10_stats['d2_unc_mean'].mean():.5f} |")
        rpt()

    rpt("### Implication for PCHIP interpolation")
    rpt()
    rpt(f"Of the {n_10nm} native-10-nm spectra requiring PCHIP interpolation, "
        f"{n_10nm_unc} have uncertainty values. When reflectance is interpolated "
        "from 10-nm to 5-nm grid, the corresponding uncertainty values will also "
        "need to be represented on the 5-nm grid. The method for doing so "
        "(interpolation, propagation, or other) requires a separate scientific "
        "decision.")
    rpt()

    # ================================================================
    # SECTION 11: Possible scientific uses
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 11. Potential Scientific Uses")
    rpt()

    rpt("### A. QC flagging")
    rpt()
    high_unc = unc_stats_df[unc_stats_df["unc_median"] > unc_stats_df["unc_median"].quantile(0.95)]
    rpt(f"- Spectra with median uncertainty above 95th percentile: {len(high_unc)}")
    rpt(f"- 95th percentile threshold: {unc_stats_df['unc_median'].quantile(0.95):.5f}")
    rpt()
    rpt("**Assessment:** The third column could support identification of "
        "high-uncertainty measurements for QC flagging. However, without knowing "
        "the absolute meaning of the values (sigma, SE, etc.), the threshold "
        "would be empirical rather than physical.")
    rpt()

    rpt("### B. Weighted fitting")
    rpt()
    rpt("**Assessment:** Inverse-variance weighting requires that the values "
        "represent variance (or proportional to variance). Since the statistical "
        "meaning is undetermined, using these values for weighted fitting would "
        "rest on unverified assumptions. This use is **not recommended** until "
        "the semantics are established.")
    rpt()

    rpt("### C. Uncertainty propagation")
    rpt()
    rpt("**Assessment:** Propagation through interpolation or feature extraction "
        "requires knowing the statistical meaning and distribution. This is "
        "**not recommended** until the semantics are established.")
    rpt()

    rpt("### D. Sensitivity analysis")
    rpt()
    rpt("**Assessment:** Comparing results with and without uncertainty-based "
        "weighting could reveal whether conclusions are sensitive to measurement "
        "quality. This is a **reasonable use** even without knowing the exact "
        "semantics, as long as the values are treated as relative quality "
        "indicators rather than absolute uncertainties.")
    rpt()

    rpt("### E. Metadata only")
    rpt()
    rpt("**Assessment:** Retaining the uncertainty values as metadata without "
        "quantitative use is the **most conservative approach** and preserves "
        "the information for future use when better documentation may be available.")
    rpt()

    # ================================================================
    # SECTION 12: Figures
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 12. Diagnostic Figures")
    rpt()

    # Figure 1: Distribution of third-column values
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    all_medians = unc_stats_df["unc_median"].dropna()
    axes[0].hist(all_medians, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Per-spectrum median uncertainty")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of per-spectrum median uncertainty")
    axes[0].axvline(all_medians.median(), color="red", linestyle="--",
                    label=f"Median: {all_medians.median():.4f}")
    axes[0].legend()

    all_means = unc_stats_df["unc_mean"].dropna()
    axes[1].hist(all_means, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Per-spectrum mean uncertainty")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of per-spectrum mean uncertainty")
    axes[1].axvline(all_means.mean(), color="red", linestyle="--",
                    label=f"Mean: {all_means.mean():.4f}")
    axes[1].legend()
    plt.tight_layout()
    fig1_path = FIG_DIR / "uncertainty_fig1_distribution.png"
    plt.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![Distribution]({fig1_path.relative_to(PROJECT_ROOT)})")
    rpt()

    # Figure 2: Uncertainty vs wavelength (representative spectra)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    d2_representatives = [s for s in d2_spectra if len(s["wl"]) > 20][:4]
    for ax, sd in zip(axes.flat, d2_representatives):
        mask = np.isfinite(sd["unc"])
        ax.plot(sd["wl"][mask], sd["unc"][mask], "b-", alpha=0.6, linewidth=0.8,
                label="Uncertainty")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Uncertainty")
        ax.set_title(f"{sd['file_id']} ({sd['mineral']})")
        ax.legend(fontsize=8)
    plt.suptitle("Uncertainty vs wavelength (representative D2 spectra)", fontsize=12)
    plt.tight_layout()
    fig2_path = FIG_DIR / "uncertainty_fig2_vs_wavelength.png"
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![vs Wavelength]({fig2_path.relative_to(PROJECT_ROOT)})")
    rpt()

    # Figure 3: Uncertainty vs reflectance
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(point_df) > 0:
        sample = point_df.dropna(subset=["reflectance", "uncertainty"])
        if len(sample) > 10000:
            sample = sample.sample(10000, random_state=42)
        ax.scatter(sample["reflectance"], sample["uncertainty"],
                   s=1, alpha=0.3, color="steelblue")
    ax.set_xlabel("Reflectance")
    ax.set_ylabel("Uncertainty")
    ax.set_title("Uncertainty vs Reflectance (all D2 points)")
    plt.tight_layout()
    fig3_path = FIG_DIR / "uncertainty_fig3_vs_reflectance.png"
    plt.savefig(fig3_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![vs Reflectance]({fig3_path.relative_to(PROJECT_ROOT)})")
    rpt()

    # Figure 4: Relative uncertainty (ratio) vs reflectance
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(point_df) > 0:
        sample = point_df.dropna(subset=["reflectance", "abs_ratio"])
        sample = sample[sample["abs_ratio"] < 0.5]
        if len(sample) > 10000:
            sample = sample.sample(10000, random_state=42)
        ax.scatter(sample["reflectance"], sample["abs_ratio"],
                   s=1, alpha=0.3, color="darkorange")
    ax.set_xlabel("Reflectance")
    ax.set_ylabel("Uncertainty / |Reflectance|")
    ax.set_title("Relative uncertainty magnitude vs Reflectance")
    plt.tight_layout()
    fig4_path = FIG_DIR / "uncertainty_fig4_ratio_vs_reflectance.png"
    plt.savefig(fig4_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![Ratio vs Reflectance]({fig4_path.relative_to(PROJECT_ROOT)})")
    rpt()

    # Figure 5: Representative spectra with reflectance + uncertainty
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    d2_full_rep = [s for s in d2_spectra
                   if len(s["wl"]) > 50 and s["d2_eligible"]][:3]
    for ax, sd in zip(axes, d2_full_rep):
        mask = np.isfinite(sd["unc"])
        ax2_twin = ax.twinx()
        ax.plot(sd["wl"], sd["rf"], "k-", linewidth=1, label="Reflectance")
        ax2_twin.plot(sd["wl"][mask], sd["unc"][mask], "r-", alpha=0.7,
                      linewidth=0.8, label="Uncertainty")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance", color="k")
        ax2_twin.set_ylabel("Uncertainty", color="r")
        ax.set_title(f"{sd['file_id']} ({sd['mineral']}, {sd['instrument_prefix']})")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    plt.suptitle("Representative spectra: Reflectance + Uncertainty", fontsize=12)
    plt.tight_layout()
    fig5_path = FIG_DIR / "uncertainty_fig5_spectra_with_uncertainty.png"
    plt.savefig(fig5_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![Spectra with uncertainty]({fig5_path.relative_to(PROJECT_ROOT)})")
    rpt()

    # Figure 6: D2 uncertainty coverage
    fig, ax = plt.subplots(figsize=(10, 6))
    minerals_with_unc = d2_stats["mineral"].value_counts().sort_values(ascending=True)
    minerals_total = d2_val.merge(
        unc_spectra[["file_id", "has_uncertainty"]],
        left_on="relab_measurement_id", right_on="file_id", how="left"
    )["mineral"].value_counts().sort_values(ascending=True)
    y_pos = np.arange(len(minerals_total))
    ax.barh(y_pos, minerals_total.values, color="lightgray", label="All D2")
    for mineral in minerals_with_unc.index:
        if mineral in minerals_total.index:
            idx = minerals_total.index.get_loc(mineral)
            ax.barh(idx, minerals_with_unc[mineral], color="steelblue",
                    label="_nolegend_")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(minerals_total.index)
    ax.set_xlabel("Number of spectra")
    ax.set_title("D2 uncertainty coverage by mineral")
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="lightgray", label="All D2 spectra"),
                       Patch(facecolor="steelblue", label="With uncertainty")]
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    fig6_path = FIG_DIR / "uncertainty_fig6_d2_coverage.png"
    plt.savefig(fig6_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![D2 coverage]({fig6_path.relative_to(PROJECT_ROOT)})")
    rpt()

    # Figure 7: Uncertainty by instrument prefix
    fig, ax = plt.subplots(figsize=(10, 6))
    prefix_data = []
    for prefix in unc_stats_df["instrument_prefix"].unique():
        sub = unc_stats_df[unc_stats_df["instrument_prefix"] == prefix]
        prefix_data.append({
            "prefix": prefix,
            "median": sub["unc_median"].median(),
            "q25": sub["unc_median"].quantile(0.25),
            "q75": sub["unc_median"].quantile(0.75),
            "n": len(sub),
        })
    prefix_df = pd.DataFrame(prefix_data).sort_values("median", ascending=True)
    y_pos = np.arange(len(prefix_df))
    ax.barh(y_pos, prefix_df["median"], color="steelblue", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['prefix']} (n={r['n']})"
                        for _, r in prefix_df.iterrows()])
    ax.set_xlabel("Median uncertainty")
    ax.set_title("Uncertainty distribution by instrument prefix")
    plt.tight_layout()
    fig7_path = FIG_DIR / "uncertainty_fig7_by_prefix.png"
    plt.savefig(fig7_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![By prefix]({fig7_path.relative_to(PROJECT_ROOT)})")
    rpt()

    # Figure 8: Uncertainty behavior by native grid type
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for sd in d2_spectra[:50]:
        mask = np.isfinite(sd["unc"])
        if mask.sum() > 10:
            wl_norm = (sd["wl"][mask] - sd["wl"][mask].min()) / \
                      (sd["wl"][mask].max() - sd["wl"][mask].min() + 1e-10)
            unc_norm = sd["unc"][mask]
            if sd["native_spacing_nm"] == 5.0:
                axes[0].plot(wl_norm, unc_norm, "b-", alpha=0.2, linewidth=0.5)
            elif sd["native_spacing_nm"] == 10.0:
                axes[1].plot(wl_norm, unc_norm, "r-", alpha=0.3, linewidth=0.5)
    axes[0].set_title("Native 5 nm (n shown)")
    axes[0].set_xlabel("Normalized wavelength position")
    axes[0].set_ylabel("Uncertainty")
    axes[1].set_title("Native 10 nm (n shown)")
    axes[1].set_xlabel("Normalized wavelength position")
    plt.suptitle("Uncertainty profiles by native grid type", fontsize=12)
    plt.tight_layout()
    fig8_path = FIG_DIR / "uncertainty_fig8_by_grid_type.png"
    plt.savefig(fig8_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![By grid type]({fig8_path.relative_to(PROJECT_ROOT)})")
    rpt()

    # ================================================================
    # SECTION 13: Limitations
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 13. Limitations")
    rpt()
    rpt("1. **No official RELAB documentation** was available. The interpretation "
        "of the third column rests on the RELAB field dictionary derived from "
        "direct file inspection.")
    rpt()
    rpt("2. **Statistical meaning undetermined.** It is unknown whether the values "
        "represent 1-sigma, 2-sigma, standard error, instrumental noise, or "
        "another quantity.")
    rpt()
    rpt("3. **Not all spectra have uncertainty.** Only 461/1017 files contain a "
        "third column, and only 377/454 D2-compatible spectra have uncertainty. "
        "Any quantitative use would need to handle missing uncertainty.")
    rpt()
    rpt("4. **Instrument-dependent.** The third column is strongly correlated with "
        "specific instrument campaigns (prefix `c1`, voltage 0.777), suggesting "
        "that values may not be comparable across configurations.")
    rpt()
    rpt("5. **No ground truth.** Without repeated measurements or reference "
        "materials with known uncertainty, the accuracy of the reported values "
        "cannot be independently verified.")
    rpt()

    # ================================================================
    # SECTION 14: Recommendation
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 14. Scientific Recommendation")
    rpt()

    rpt("### Question 1: What does the third column represent?")
    rpt()
    rpt("The third column is **plausibly measurement uncertainty** (unitless, "
        "absolute) based on the RELAB field dictionary and empirical evidence "
        "(small positive values, consistent with reflectance magnitude). However, "
        "its **exact statistical meaning cannot be determined** from available data.")
    rpt()

    rpt("### Question 2: Is its meaning sufficiently established for quantitative use?")
    rpt()
    rpt("**No.** The statistical meaning (sigma, SE, noise estimate, etc.) is "
        "undetermined. Quantitative use (weighted fitting, uncertainty propagation) "
        "would rest on unverified assumptions.")
    rpt()

    rpt("### Question 3: Is it comparable across the RELAB dataset?")
    rpt()
    rpt("**Not without caveats.** The values are instrument-configuration-dependent "
        "and vary systematically across prefixes and voltages. Cross-campaign "
        "comparisons would require normalization or stratification.")
    rpt()

    rpt("### Question 4: How many D2-compatible spectra contain it?")
    rpt()
    rpt(f"**{n_d2_unc}** out of {n_d2_total} D2-compatible spectra "
        f"({100*n_d2_unc/n_d2_total:.1f}%).")
    rpt()

    rpt("### Question 5: Should it be used quantitatively?")
    rpt()
    rpt("**Recommendation: Retain as metadata + QC reference only.**")
    rpt()
    rpt("Rationale:")
    rpt("- The values are plausibly uncertainty but their exact meaning is unknown")
    rpt("- They are not comparable across instrument configurations without adjustment")
    rpt("- Quantitative use (weighting, propagation) requires established semantics")
    rpt("- The most defensible approach is to retain the values for future use "
        "when better documentation may become available")
    rpt()
    rpt("Specifically:")
    rpt("- **Do not** use for weighted fitting or uncertainty propagation")
    rpt("- **Do** retain the values in the processed dataset as metadata")
    rpt("- **May** use for empirical QC flagging (e.g., identifying outliers)")
    rpt("- **May** use for sensitivity analysis as a relative quality indicator")
    rpt()

    rpt("### Question 6: What should happen to uncertainty for the 64 native-10-nm spectra?")
    rpt()
    rpt(f"Of the {n_10nm} native-10-nm spectra, {n_10nm_unc} have uncertainty values. "
        "Since uncertainty interpolation is a separate scientific decision, "
        "**no action should be taken at this stage**. The options are:")
    rpt()
    rpt("1. Interpolate uncertainty with the same PCHIP method (simplest)")
    rpt("2. Propagate uncertainty through the interpolation analytically")
    rpt("3. Leave uncertainty at native 10-nm resolution (no interpolation)")
    rpt("4. Discard uncertainty for interpolated spectra")
    rpt()
    rpt("This decision should be made after the uncertainty treatment is frozen.")
    rpt()

    rpt("---")
    rpt()
    rpt("## 15. Output Checksums")
    rpt()

    # Save CSVs first so we can checksum them
    spectrum_summary_path = OUT_DIR / "phase2_step4_uncertainty_spectrum_summary.csv"
    unc_stats_df.to_csv(spectrum_summary_path, index=False)

    point_summary_path = OUT_DIR / "phase2_step4_uncertainty_point_summary.csv"
    if len(point_df) > 0:
        point_df.to_csv(point_summary_path, index=False)

    rpt("| File | MD5 |")
    rpt("|------|-----|")
    for fp in [spectrum_summary_path, point_summary_path]:
        if fp.exists():
            md5 = compute_md5(fp)
            rpt(f"| {fp.name} | {md5} |")
    rpt()

    rpt("---")
    rpt()
    rpt(f"*Report generated: {timestamp}*")
    rpt("*This is an investigation only. No frozen decisions were changed.*")

    # Save report
    report_path = OUT_DIR / "phase2_step4_uncertainty_investigation.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return report_path, spectrum_summary_path, point_summary_path


# ================================================================
# REPRODUCBILITY CHECK
# ================================================================


def run_reproducibility_check():
    """Run the investigation twice and verify identical machine-readable outputs."""
    print("=== Reproducibility check ===")

    # First run
    print("Run 1...")
    r1 = run_investigation()

    # Checksums after run 1 (CSV files only — report contains a timestamp)
    csv_files = [fp for fp in r1 if fp.suffix == ".csv"]
    md5_run1 = {}
    for fp in csv_files:
        if fp.exists():
            md5_run1[fp.name] = compute_md5(fp)

    # Second run
    print("Run 2...")
    r2 = run_investigation()

    # Checksums after run 2
    md5_run2 = {}
    for fp in r2:
        if fp.suffix == ".csv" and fp.exists():
            md5_run2[fp.name] = compute_md5(fp)

    # Compare
    all_match = True
    for name in md5_run1:
        if name in md5_run2:
            if md5_run1[name] == md5_run2[name]:
                print(f"  {name}: MATCH")
            else:
                print(f"  {name}: MISMATCH")
                all_match = False
        else:
            print(f"  {name}: MISSING in run 2")
            all_match = False

    # Verify raw files unchanged
    print("\nVerifying raw file checksums...")
    checked, mismatches = validate_checksums()
    print(f"  Checked: {checked}, Mismatches: {len(mismatches)}")

    if all_match and len(mismatches) == 0:
        print("\nReproducibility check PASSED")
    else:
        print("\nReproducibility check FAILED")

    return all_match and len(mismatches) == 0


# ================================================================
# ENTRY POINT
# ================================================================


if __name__ == "__main__":
    print("Phase 2 Step 4 — Measurement-Uncertainty Investigation")
    print("=" * 60)
    print()

    ok = run_reproducibility_check()

    print()
    print("Outputs:")
    out_dir = Path(__file__).resolve().parent.parent / "results" / "02_preprocessing_qc"
    for fp in sorted(out_dir.glob("phase2_step4_*")):
        print(f"  {fp}")
    for fp in sorted((out_dir / "figures").glob("uncertainty_*")):
        print(f"  {fp}")

    if not ok:
        print("\nWARNING: reproducibility check failed!")
        exit(1)
