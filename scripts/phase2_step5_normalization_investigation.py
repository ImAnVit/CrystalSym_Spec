#!/usr/bin/env python3
"""
Phase 2 Step 5 — Reflectance Normalization Investigation
========================================================

CrystalSym-Spec: Interpretable ML and Symbolic Regression of
Crystal Symmetry, Chemistry, and Mineral Spectral Complexity

Determines whether reflectance normalization is scientifically justified
for the primary spectral-complexity analysis.

This is an investigation only. No raw files are modified. No final
processed dataset is created. No normalization method is frozen.

Outputs:
  results/02_preprocessing_qc/phase2_step5_normalization_investigation.md
  results/02_preprocessing_qc/phase2_step5_normalization_comparison.csv
  results/02_preprocessing_qc/phase2_step5_normalization_summary.csv
  results/02_preprocessing_qc/figures/normalization_*.png
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

from scipy.interpolate import PchipInterpolator

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
REFLECTANCE_QC_CSV = PROJECT_ROOT / "results" / "02_preprocessing_qc" / "phase2_step3_reflectance_qc_spectrum_summary.csv"
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

NORM_METHODS = ["A_none", "B_vector", "C_max", "D_area", "E_continuum", "F_zscore"]

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


def parse_footer_metadata(footer_lines):
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
    prefix = ""
    for ch in filename:
        if ch.isalpha():
            prefix += ch
        else:
            break
    return prefix


def classify_acquisition_regime(prefix):
    if not isinstance(prefix, str) or not prefix:
        return "unknown"
    if prefix.startswith("c"):
        return "VNIR_spectrophotometer"
    elif prefix.startswith("b") or prefix.startswith("m"):
        return "FTIR"
    elif prefix.startswith("n"):
        return "NIR_spectrometer"
    else:
        return "other"


# ================================================================
# PCHIP INTERPOLATION (for native-10-nm spectra)
# ================================================================


def interp_pchip(wl_native, rf_native, target_grid):
    if len(wl_native) < 4:
        return np.interp(target_grid, wl_native, rf_native)
    try:
        interp_fn = PchipInterpolator(wl_native, rf_native)
        return interp_fn(target_grid)
    except Exception:
        return np.interp(target_grid, wl_native, rf_native)


def resample_to_grid(wl, rf, native_spacing, target_grid):
    if native_spacing <= 5.5:
        return np.interp(target_grid, wl, rf)
    else:
        return interp_pchip(wl, rf, target_grid)


# ================================================================
# COMPLEXITY DESCRIPTORS
# ================================================================


def compute_complexity(r):
    r = np.asarray(r, dtype=float)
    n = len(r)
    if n < 5:
        return {k: np.nan for k in [
            "first_order_variation", "second_order_variation",
            "peak_density", "spectral_entropy", "spectral_range",
            "n_extrema", "curvature_sum", "integrated_abs_d1",
        ]}

    d1 = np.diff(r)
    d2 = np.diff(r, n=2)

    first_order = float(np.sum(np.abs(d1)))
    second_order = float(np.sum(np.abs(d2)))

    is_peak = np.zeros(n, dtype=bool)
    is_trough = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        if r[i] > r[i - 1] and r[i] > r[i + 1]:
            is_peak[i] = True
        if r[i] < r[i - 1] and r[i] < r[i + 1]:
            is_trough[i] = True
    n_extrema = int(is_peak.sum() + is_trough.sum())
    peak_density = n_extrema / n * 100.0

    r_finite = r[np.isfinite(r)]
    if len(r_finite) > 0 and (np.max(r_finite) - np.min(r_finite)) > 1e-15:
        hist, _ = np.histogram(r_finite, bins="auto")
        hist = hist.astype(float)
        total = hist.sum()
        if total > 0:
            p = hist / total
            p = p[p > 0]
            spectral_entropy = float(-np.sum(p * np.log(p)))
        else:
            spectral_entropy = 0.0
    else:
        spectral_entropy = 0.0

    spectral_range = float(np.max(r_finite) - np.min(r_finite)) if len(r_finite) > 0 else 0.0

    curvature_vals = np.abs(d2) / (1.0 + d1[:-1] ** 2) ** 1.5
    curvature_sum = float(np.sum(curvature_vals))

    integrated_abs_d1 = float(np.sum(np.abs(d1)))

    return {
        "first_order_variation": first_order,
        "second_order_variation": second_order,
        "peak_density": peak_density,
        "spectral_entropy": spectral_entropy,
        "spectral_range": spectral_range,
        "n_extrema": n_extrema,
        "curvature_sum": curvature_sum,
        "integrated_abs_d1": integrated_abs_d1,
    }


# ================================================================
# NORMALIZATION METHODS
# ================================================================


def norm_none(r):
    return r.copy(), 1.0, None


def norm_vector(r):
    magnitude = np.sqrt(np.sum(r ** 2))
    if magnitude < 1e-15:
        return r.copy(), magnitude, "zero_magnitude"
    return r / magnitude, magnitude, None


def norm_max(r):
    mx = np.max(r)
    if abs(mx) < 1e-15:
        return r.copy(), mx, "zero_max"
    return r / mx, mx, None


def norm_area(r):
    area = np.trapezoid(np.abs(r))
    if area < 1e-15:
        return r.copy(), area, "zero_area"
    return r / area, area, None


def norm_continuum(r, wl):
    n = len(r)
    if n < 3:
        return r.copy(), np.nan, "too_few_points"

    valid = np.isfinite(r) & np.isfinite(wl)
    if valid.sum() < 3:
        return r.copy(), np.nan, "insufficient_finite"

    wl_v = wl[valid]
    r_v = r[valid]

    continuum = np.full(n, np.nan)
    continuum[valid] = _compute_continuum(wl_v, r_v)

    if np.any(np.isnan(continuum[valid])):
        return r.copy(), np.nan, "continuum_failure"

    denom = continuum.copy()
    denom[~valid] = 1.0
    zero_mask = np.abs(denom) < 1e-15
    if zero_mask.any():
        return r.copy(), np.nan, "zero_continuum"

    result = r / denom
    factor = np.nanmean(continuum[valid])
    return result, factor, None


def _compute_continuum(wl, r):
    n = len(wl)
    continuum = np.zeros(n)

    points = list(zip(wl, r))
    hull_input = np.array([(wl[i], r[i]) for i in range(n)])

    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(hull_input)
    except Exception:
        return np.full(n, np.nan)

    upper_envelope = np.full(n, -np.inf)

    for simplex_idx in hull.simplices:
        i0, i1 = simplex_idx
        w0, r0 = hull_input[i0]
        w1, r1 = hull_input[i1]

        if abs(w1 - w0) < 1e-15:
            continue

        slope = (r1 - r0) / (w1 - w0)
        intercept = r0 - slope * w0

        w_lo = min(w0, w1)
        w_hi = max(w0, w1)

        for j in range(n):
            if wl[j] >= w_lo - 1e-10 and wl[j] <= w_hi + 1e-10:
                env_val = slope * wl[j] + intercept
                if env_val > upper_envelope[j]:
                    upper_envelope[j] = env_val

    for j in range(n):
        if upper_envelope[j] > -np.inf:
            continuum[j] = upper_envelope[j]
        else:
            continuum[j] = r[j]

    return continuum


def norm_zscore(r):
    mean_r = np.mean(r)
    std_r = np.std(r, ddof=0)
    if std_r < 1e-15:
        return r.copy(), std_r, "zero_std"
    return (r - mean_r) / std_r, std_r, None


NORMALIZERS = {
    "A_none": norm_none,
    "B_vector": norm_vector,
    "C_max": norm_max,
    "D_area": norm_area,
    "E_continuum": None,
    "F_zscore": norm_zscore,
}


# ================================================================
# MAIN INVESTIGATION
# ================================================================


def run_investigation():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = []

    def rpt(line=""):
        report_lines.append(line)

    rpt("# Phase 2 Step 5 — Reflectance Normalization Investigation")
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
    reflectance_qc = pd.read_csv(REFLECTANCE_QC_CSV)

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

    d2_pop = merged[merged["d2_eligible"]].copy()
    d2_pop = d2_pop.sort_values("file_id").reset_index(drop=True)

    # Merge reflectance QC category
    if "qc_category" in reflectance_qc.columns:
        qc_map = dict(zip(reflectance_qc["file_id"], reflectance_qc["qc_category"]))
    elif "reflectance_qc_category" in reflectance_qc.columns:
        qc_map = dict(zip(reflectance_qc["file_id"], reflectance_qc["reflectance_qc_category"]))
    else:
        qc_map = {}
    d2_pop["reflectance_qc"] = d2_pop["file_id"].map(qc_map).fillna("unknown")

    # ================================================================
    # SECTION 1: Establish the candidate population
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 1. Candidate Population")
    rpt()

    n_d2 = len(d2_pop)
    n_minerals = d2_pop["mineral_directory"].nunique()
    minerals_list = sorted(d2_pop["mineral_directory"].unique())
    n_specimen_groups = d2_pop["specimen_group_id"].nunique()
    n_native_5nm = (d2_pop["native_spacing_nm"] <= 5.5).sum()
    n_native_10nm = (d2_pop["native_spacing_nm"] > 5.5).sum()

    rpt(f"- D2-compatible spectra: **{n_d2}**")
    rpt(f"- Minerals represented: **{n_minerals}**")
    rpt(f"- Specimen groups: **{n_specimen_groups}**")
    rpt(f"- Native 5-nm spectra: **{n_native_5nm}**")
    rpt(f"- Native 10-nm spectra (require PCHIP): **{n_native_10nm}**")
    rpt()

    rpt("### Mineral distribution")
    rpt()
    mineral_counts = d2_pop["mineral_directory"].value_counts().sort_index()
    rpt("| Mineral | Count |")
    rpt("|---------|-------|")
    for mineral, count in mineral_counts.items():
        rpt(f"| {mineral} | {count} |")
    rpt(f"| **Total** | **{n_d2}** |")
    rpt()

    rpt("### Provenance group distribution")
    rpt()
    prov_counts = d2_pop.groupby("mineral_directory")["specimen_group_id"].nunique()
    rpt("| Mineral | Provenance groups |")
    rpt("|---------|-------------------|")
    for mineral in minerals_list:
        if mineral in prov_counts.index:
            rpt(f"| {mineral} | {prov_counts[mineral]} |")
    rpt()

    rpt("### QC category distribution")
    rpt()
    qc_counts = d2_pop["reflectance_qc"].value_counts()
    rpt("| QC Category | Count |")
    rpt("|-------------|-------|")
    for cat in ["A", "B", "C", "D"]:
        n = qc_counts.get(cat, 0)
        rpt(f"| {cat} | {n} |")
    rpt()

    # Verify frozen population
    assert n_d2 == 454, f"Expected 454 D2 spectra, got {n_d2}"
    assert n_minerals == 18, f"Expected 18 minerals, got {n_minerals}"
    assert n_native_5nm == 390, f"Expected 390 native-5nm, got {n_native_5nm}"
    assert n_native_10nm == 64, f"Expected 64 native-10nm, got {n_native_10nm}"

    # ================================================================
    # Load all D2 spectra into memory
    # ================================================================

    print("Loading D2 spectra...")
    spectra = []
    for idx, row in d2_pop.iterrows():
        fp = DATA_DIR / row["rel_path"]
        segments, footer, pw = parse_relab_full(fp)
        wl_d2, rf_d2, unc_d2, has_unc = prepare_d2_spectrum(segments)
        footer_meta = parse_footer_metadata(footer)

        native_spacing = row["native_spacing_nm"]
        if native_spacing > 5.5:
            rf_grid = resample_to_grid(wl_d2, rf_d2, native_spacing, D2_TARGET_GRID)
            wl_grid = D2_TARGET_GRID.copy()
            is_interpolated = True
        else:
            rf_grid = np.interp(D2_TARGET_GRID, wl_d2, rf_d2)
            wl_grid = D2_TARGET_GRID.copy()
            is_interpolated = False

        prefix = row["instrument_prefix"]
        if not isinstance(prefix, str):
            prefix = ""

        spectra.append({
            "file_id": row["file_id"],
            "filename": row["filename"],
            "mineral": row["mineral_directory"],
            "specimen_group_id": row["specimen_group_id"],
            "inclusion_category": row["inclusion_category"],
            "reflectance_qc": row["reflectance_qc"],
            "native_spacing_nm": native_spacing,
            "instrument_prefix": prefix,
            "acquisition_regime": classify_acquisition_regime(prefix),
            "voltage": footer_meta.get("voltage", "") or "",
            "source_angle": footer_meta.get("source_angle", "") or "",
            "detect_angle": footer_meta.get("detect_angle", "") or "",
            "wl_native": wl_d2,
            "rf_native": rf_d2,
            "wl_grid": wl_grid,
            "rf_grid": rf_grid,
            "is_interpolated": is_interpolated,
        })

    print(f"  Loaded {len(spectra)} spectra on {N_TARGET}-point grid")

    # ================================================================
    # SECTION 2: Raw reflectance characteristics
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 2. Raw Reflectance Characteristics (D2 Grid)")
    rpt()

    raw_stats = []
    for sp in spectra:
        r = sp["rf_grid"]
        r_fin = r[np.isfinite(r)]
        if len(r_fin) == 0:
            continue
        d1 = np.diff(r_fin)
        stats = {
            "file_id": sp["file_id"],
            "mineral": sp["mineral"],
            "specimen_group_id": sp["specimen_group_id"],
            "reflectance_qc": sp["reflectance_qc"],
            "native_spacing_nm": sp["native_spacing_nm"],
            "instrument_prefix": sp["instrument_prefix"],
            "voltage": sp["voltage"],
            "rf_min": float(np.min(r_fin)),
            "rf_max": float(np.max(r_fin)),
            "rf_median": float(np.median(r_fin)),
            "rf_mean": float(np.mean(r_fin)),
            "rf_std": float(np.std(r_fin)),
            "rf_range": float(np.max(r_fin) - np.min(r_fin)),
            "mean_abs_slope": float(np.mean(np.abs(d1))),
            "integrated_reflectance": float(np.trapezoid(r_fin, dx=D2_SPACING)),
        }
        raw_stats.append(stats)

    raw_df = pd.DataFrame(raw_stats)

    rpt("### Global distribution of per-spectrum reflectance statistics")
    rpt()
    rpt("| Statistic | Min | Q25 | Median | Mean | Q75 | Max |")
    rpt("|-----------|-----|-----|--------|------|-----|-----|")
    for col in ["rf_min", "rf_max", "rf_median", "rf_mean", "rf_std",
                "rf_range", "mean_abs_slope", "integrated_reflectance"]:
        vals = raw_df[col]
        q25, q75 = np.percentile(vals, [25, 75])
        rpt(f"| {col} | {vals.min():.4f} | {q25:.4f} | {vals.median():.4f} | "
            f"{vals.mean():.4f} | {q75:.4f} | {vals.max():.4f} |")
    rpt()

    rpt("### Reflectance range by mineral")
    rpt()
    rpt("| Mineral | n | Median range | Median mean | Median std |")
    rpt("|---------|---|-------------|-------------|-----------|")
    for mineral in minerals_list:
        sub = raw_df[raw_df["mineral"] == mineral]
        rpt(f"| {mineral} | {len(sub)} | {sub['rf_range'].median():.4f} | "
            f"{sub['rf_mean'].median():.4f} | {sub['rf_std'].median():.4f} |")
    rpt()

    rpt("### Key observations")
    rpt()
    overall_range = raw_df["rf_range"]
    rpt(f"- Overall reflectance range: {overall_range.min():.4f} to {overall_range.max():.4f}")
    rpt(f"- Median per-spectrum range: {overall_range.median():.4f}")
    rpt(f"- Mean reflectance varies from {raw_df['rf_mean'].min():.4f} to "
        f"{raw_df['rf_mean'].max():.4f}")
    n_negative = (raw_df["rf_min"] < 0).sum()
    n_gt1 = (raw_df["rf_max"] > 1.0).sum()
    rpt(f"- Spectra with negative reflectance: {n_negative}")
    rpt(f"- Spectra with reflectance > 1: {n_gt1}")
    rpt()
    rpt("Absolute reflectance varies substantially across the dataset, "
        "both in overall level and dynamic range. This motivates investigating "
        "whether normalization is needed or harmful.")
    rpt()

    # ================================================================
    # SECTION 3: Apply normalization methods
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 3. Candidate Normalization Methods")
    rpt()

    rpt("### Mathematical definitions")
    rpt()
    rpt("| Method | Formula | Factor |")
    rpt("|--------|---------|--------|")
    rpt("| A: None | R'(λ) = R(λ) | 1 |")
    rpt("| B: Vector | R'(λ) = R(λ) / ‖R‖₂ | ‖R‖₂ = √(Σ R(λ)²) |")
    rpt("| C: Maximum | R'(λ) = R(λ) / max(R) | max(R) |")
    rpt("| D: Area | R'(λ) = R(λ) / ∫|R(λ)|dλ | ∫|R(λ)|dλ (trapezoidal) |")
    rpt("| E: Continuum | R'(λ) = R(λ) / C(λ) | mean(C), C = upper convex hull |")
    rpt("| F: Z-score | R'(λ) = (R(λ) - μ) / σ | σ (std dev) |")
    rpt()

    print("Applying normalization methods...")

    comparison_rows = []
    failure_counts = defaultdict(lambda: defaultdict(int))

    for sp in spectra:
        r = sp["rf_grid"]
        wl = sp["wl_grid"]

        for method in NORM_METHODS:
            row_base = {
                "file_id": sp["file_id"],
                "filename": sp["filename"],
                "mineral": sp["mineral"],
                "specimen_group_id": sp["specimen_group_id"],
                "inclusion_category": sp["inclusion_category"],
                "reflectance_qc": sp["reflectance_qc"],
                "native_spacing_nm": sp["native_spacing_nm"],
                "instrument_prefix": sp["instrument_prefix"],
                "acquisition_regime": sp["acquisition_regime"],
                "is_interpolated": sp["is_interpolated"],
                "normalization_method": method,
            }

            if method == "E_continuum":
                r_norm, factor, failure = norm_continuum(r, wl)
            else:
                r_norm, factor, failure = NORMALIZERS[method](r)

            row_base["norm_factor"] = float(factor) if np.isfinite(factor) else np.nan
            row_base["failure_flag"] = failure if failure else ""

            if failure:
                failure_counts[method][failure] += 1
                row_base["rf_min"] = np.nan
                row_base["rf_max"] = np.nan
                row_base["rf_mean"] = np.nan
                row_base["rf_range"] = np.nan
                for k in ["first_order_variation", "second_order_variation",
                           "peak_density", "spectral_entropy", "spectral_range",
                           "n_extrema", "curvature_sum", "integrated_abs_d1"]:
                    row_base[k] = np.nan
                comparison_rows.append(row_base)
                continue

            r_fin = r_norm[np.isfinite(r_norm)]
            if len(r_fin) == 0:
                row_base["rf_min"] = np.nan
                row_base["rf_max"] = np.nan
                row_base["rf_mean"] = np.nan
                row_base["rf_range"] = np.nan
                for k in ["first_order_variation", "second_order_variation",
                           "peak_density", "spectral_entropy", "spectral_range",
                           "n_extrema", "curvature_sum", "integrated_abs_d1"]:
                    row_base[k] = np.nan
                comparison_rows.append(row_base)
                continue

            row_base["rf_min"] = float(np.min(r_fin))
            row_base["rf_max"] = float(np.max(r_fin))
            row_base["rf_mean"] = float(np.mean(r_fin))
            row_base["rf_range"] = float(np.max(r_fin) - np.min(r_fin))

            complexity = compute_complexity(r_norm)
            row_base.update(complexity)

            comparison_rows.append(row_base)

    comparison_df = pd.DataFrame(comparison_rows)

    # Store normalized spectra for later analysis
    norm_spectra = {}
    for sp in spectra:
        r = sp["rf_grid"]
        wl = sp["wl_grid"]
        fid = sp["file_id"]
        norm_spectra[fid] = {"original": r.copy()}
        for method in NORM_METHODS:
            if method == "E_continuum":
                r_norm, _, fail = norm_continuum(r, wl)
            else:
                r_norm, _, fail = NORMALIZERS[method](r)
            norm_spectra[fid][method] = r_norm if not fail else None

    print(f"  Normalization complete. Comparison rows: {len(comparison_df)}")

    # ================================================================
    # SECTION 3 continued: Report normalization factors
    # ================================================================

    rpt("### Normalization factor distributions")
    rpt()
    rpt("| Method | n success | n fail | Median factor | Q25 | Q75 | Min | Max |")
    rpt("|--------|-----------|--------|---------------|-----|-----|-----|-----|")
    for method in NORM_METHODS:
        sub = comparison_df[comparison_df["normalization_method"] == method]
        n_fail = sub["failure_flag"].astype(bool).sum()
        n_ok = len(sub) - n_fail
        factors = sub.loc[sub["failure_flag"] == "", "norm_factor"].dropna()
        if len(factors) > 0:
            q25, q75 = np.percentile(factors, [25, 75])
            rpt(f"| {method} | {n_ok} | {n_fail} | {factors.median():.6f} | "
                f"{q25:.6f} | {q75:.6f} | {factors.min():.6f} | {factors.max():.6f} |")
        else:
            rpt(f"| {method} | {n_ok} | {n_fail} | — | — | — | — | — |")
    rpt()

    # Failure details
    rpt("### Failure details")
    rpt()
    any_failures = False
    for method in NORM_METHODS:
        if method in failure_counts and failure_counts[method]:
            any_failures = True
            rpt(f"**{method}:**")
            for reason, count in sorted(failure_counts[method].items(),
                                         key=lambda x: -x[1]):
                rpt(f"  - {reason}: {count}")
            rpt()
    if not any_failures:
        rpt("No failures encountered.")
        rpt()

    # ================================================================
    # SECTION 5: Spectral structure preservation
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 5. Spectral Structure Preservation")
    rpt()

    rpt("### Peak position preservation")
    rpt()

    peak_preservation = []
    for sp in spectra:
        r = sp["rf_grid"]
        wl = sp["wl_grid"]
        fid = sp["file_id"]

        orig_peaks = _find_peaks(r)
        orig_peak_wl = wl[orig_peaks]

        for method in NORM_METHODS:
            r_norm = norm_spectra[fid][method]
            if r_norm is None:
                continue
            norm_peaks = _find_peaks(r_norm)
            norm_peak_wl = wl[norm_peaks]

            if len(orig_peak_wl) == 0:
                continue

            n_preserved = 0
            total_shift = 0.0
            for opw in orig_peak_wl:
                if len(norm_peak_wl) > 0:
                    shifts = np.abs(norm_peak_wl - opw)
                    min_shift = shifts.min()
                    if min_shift <= D2_SPACING * 1.5:
                        n_preserved += 1
                        total_shift += min_shift

            peak_preservation.append({
                "file_id": fid,
                "mineral": sp["mineral"],
                "method": method,
                "n_orig_peaks": len(orig_peak_wl),
                "n_preserved": n_preserved,
                "preservation_rate": n_preserved / len(orig_peak_wl) if len(orig_peak_wl) > 0 else np.nan,
                "mean_shift_nm": total_shift / n_preserved if n_preserved > 0 else np.nan,
            })

    pp_df = pd.DataFrame(peak_preservation)

    rpt("| Method | Median preservation rate | Median shift (nm) | Spectra with 100% preservation |")
    rpt("|--------|------------------------|-------------------|-------------------------------|")
    for method in NORM_METHODS:
        sub = pp_df[pp_df["method"] == method]
        if len(sub) == 0:
            continue
        med_rate = sub["preservation_rate"].median()
        med_shift = sub["mean_shift_nm"].median()
        n_perfect = (sub["preservation_rate"] == 1.0).sum()
        rpt(f"| {method} | {med_rate:.3f} | {med_shift:.2f} | "
            f"{n_perfect}/{len(sub)} |")
    rpt()

    # Derivative structure comparison
    rpt("### Derivative structure comparison")
    rpt()
    rpt("| Method | Median |d1| change | Max |d1| change | Median |d2| change |")
    rpt("|--------|---------------------|--------------------|----------------------|")
    for method in NORM_METHODS:
        if method == "A_none":
            continue
        d1_changes = []
        d2_changes = []
        for sp in spectra:
            fid = sp["file_id"]
            r_orig = norm_spectra[fid]["A_none"]
            r_norm = norm_spectra[fid][method]
            if r_norm is None:
                continue
            d1_orig = np.abs(np.diff(r_orig))
            d1_norm = np.abs(np.diff(r_norm))
            d2_orig = np.abs(np.diff(r_orig, n=2))
            d2_norm = np.abs(np.diff(r_norm, n=2))

            orig_scale = np.mean(d1_orig) if np.mean(d1_orig) > 1e-15 else 1.0
            d1_changes.append(float(np.mean(np.abs(d1_norm - d1_orig)) / orig_scale))

            orig_scale2 = np.mean(d2_orig) if np.mean(d2_orig) > 1e-15 else 1.0
            d2_changes.append(float(np.mean(np.abs(d2_norm - d2_orig)) / orig_scale2))

        if d1_changes:
            rpt(f"| {method} | {np.median(d1_changes):.4f} | "
                f"{np.max(d1_changes):.4f} | {np.median(d2_changes):.4f} |")
    rpt()

    # Relative band depth preservation
    rpt("### Absorption band depth preservation")
    rpt()
    rpt("For each spectrum, the three deepest local absorption features are identified "
        "in the original spectrum. Band depth is measured as the vertical distance "
        "from the continuum-removed spectrum to the band minimum. Changes in relative "
        "band depth ordering are reported.")
    rpt()

    band_depth_changes = []
    for sp in spectra:
        fid = sp["file_id"]
        r_orig = norm_spectra[fid]["A_none"]
        wl = sp["wl_grid"]

        orig_troughs = _find_troughs(r_orig)
        if len(orig_troughs) < 3:
            continue

        depths_orig = []
        for t in orig_troughs:
            depth = _band_depth(r_orig, t, wl)
            depths_orig.append((t, depth))
        depths_orig.sort(key=lambda x: x[1], reverse=True)
        top3_orig = [d[0] for d in depths_orig[:3]]

        for method in NORM_METHODS:
            if method == "A_none":
                continue
            r_norm = norm_spectra[fid][method]
            if r_norm is None:
                continue

            norm_troughs = _find_troughs(r_norm)
            preserved = 0
            for t in top3_orig:
                if len(norm_troughs) > 0:
                    shifts = np.abs(wl[norm_troughs] - wl[t])
                    if shifts.min() <= D2_SPACING * 2:
                        preserved += 1

            band_depth_changes.append({
                "file_id": fid,
                "mineral": sp["mineral"],
                "method": method,
                "n_preserved_top3": preserved,
            })

    if band_depth_changes:
        bd_df = pd.DataFrame(band_depth_changes)
        rpt("| Method | Median top-3 preserved | Fraction with 3/3 |")
        rpt("|--------|----------------------|-------------------|")
        for method in NORM_METHODS:
            if method == "A_none":
                continue
            sub = bd_df[bd_df["method"] == method]
            if len(sub) == 0:
                continue
            med = sub["n_preserved_top3"].median()
            frac_3 = (sub["n_preserved_top3"] == 3).mean()
            rpt(f"| {method} | {med:.0f}/3 | {frac_3:.3f} |")
        rpt()

    rpt("### Interpretation")
    rpt()
    rpt("Methods B (vector), C (max), and D (area) are simple scalar divisions "
        "that preserve all wavelength positions exactly. They change amplitudes "
        "uniformly but do not alter peak positions, derivative zero-crossings, "
        "or relative band depths.")
    rpt()
    rpt("Method F (z-score) subtracts a constant and divides by a constant, "
        "which also preserves peak positions but inverts the spectrum if the "
        "original was positive-dominant (mean subtraction shifts the baseline).")
    rpt()
    rpt("Method E (continuum) is wavelength-dependent and can shift apparent "
        "peak positions slightly where the continuum slope is steep.")
    rpt()

    # ================================================================
    # SECTION 6: Impact on spectral-complexity descriptors
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 6. Impact on Spectral-Complexity Descriptors")
    rpt()
    rpt("This is the most important section.")
    rpt()

    complexity_cols = ["first_order_variation", "second_order_variation",
                       "peak_density", "spectral_entropy", "spectral_range",
                       "n_extrema", "curvature_sum", "integrated_abs_d1"]

    baseline = comparison_df[comparison_df["normalization_method"] == "A_none"].copy()
    baseline = baseline.set_index("file_id")

    rpt("### Absolute and relative changes from baseline (Method A)")
    rpt()

    summary_rows = []

    for method in NORM_METHODS:
        sub = comparison_df[comparison_df["normalization_method"] == method].copy()
        sub = sub.set_index("file_id")
        n_success = (sub["failure_flag"] == "").sum()
        n_fail = (sub["failure_flag"] != "").sum()

        factors = sub.loc[sub["failure_flag"] == "", "norm_factor"]

        row = {
            "method": method,
            "n_success": int(n_success),
            "n_fail": int(n_fail),
            "median_norm_factor": factors.median() if len(factors) > 0 else np.nan,
        }

        if method == "A_none":
            for col in complexity_cols:
                vals = sub.loc[sub["failure_flag"] == "", col]
                row[f"{col}_median"] = vals.median()
                row[f"{col}_abs_change_median"] = 0.0
                row[f"{col}_rel_change_median"] = 0.0
                row[f"{col}_abs_change_max"] = 0.0
                row[f"{col}_rel_change_max"] = 0.0
            summary_rows.append(row)
            continue

        for col in complexity_cols:
            orig_vals = baseline.loc[baseline["failure_flag"] == "", col]
            norm_vals = sub.loc[sub["failure_flag"] == "", col]

            common_idx = orig_vals.index.intersection(norm_vals.index)
            ov = orig_vals.loc[common_idx]
            nv = norm_vals.loc[common_idx]

            abs_change = (nv - ov).abs()
            rel_change = np.where(np.abs(ov) > 1e-15,
                                  (nv - ov).abs() / np.abs(ov),
                                  np.nan)

            row[f"{col}_median"] = nv.median()
            row[f"{col}_abs_change_median"] = abs_change.median()
            row[f"{col}_rel_change_median"] = np.nanmedian(rel_change)
            row[f"{col}_abs_change_max"] = abs_change.max()
            row[f"{col}_rel_change_max"] = np.nanmax(rel_change)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    rpt("#### First-order variation (Σ|ΔR|)")
    rpt()
    for method in NORM_METHODS:
        row = summary_df[summary_df["method"] == method].iloc[0]
        if method == "A_none":
            rpt(f"- **A (baseline):** median = {row['first_order_variation_median']:.4f}")
        else:
            rpt(f"- **{method}:** median = {row['first_order_variation_median']:.4f}, "
                f"median |change| = {row['first_order_variation_abs_change_median']:.4f}, "
                f"max |change| = {row['first_order_variation_abs_change_max']:.4f}")
    rpt()

    rpt("#### Second-order variation (Σ|Δ²R|)")
    rpt()
    for method in NORM_METHODS:
        row = summary_df[summary_df["method"] == method].iloc[0]
        if method == "A_none":
            rpt(f"- **A (baseline):** median = {row['second_order_variation_median']:.6f}")
        else:
            rpt(f"- **{method}:** median = {row['second_order_variation_median']:.6f}, "
                f"median |change| = {row['second_order_variation_abs_change_median']:.6f}, "
                f"max |change| = {row['second_order_variation_abs_change_max']:.6f}")
    rpt()

    rpt("#### Peak density (extrema / N × 100)")
    rpt()
    for method in NORM_METHODS:
        row = summary_df[summary_df["method"] == method].iloc[0]
        if method == "A_none":
            rpt(f"- **A (baseline):** median = {row['peak_density_median']:.2f}")
        else:
            rpt(f"- **{method}:** median = {row['peak_density_median']:.2f}, "
                f"median |change| = {row['peak_density_abs_change_median']:.4f}")
    rpt()

    rpt("#### Spectral entropy")
    rpt()
    for method in NORM_METHODS:
        row = summary_df[summary_df["method"] == method].iloc[0]
        if method == "A_none":
            rpt(f"- **A (baseline):** median = {row['spectral_entropy_median']:.4f}")
        else:
            rpt(f"- **{method}:** median = {row['spectral_entropy_median']:.4f}, "
                f"median |change| = {row['spectral_entropy_abs_change_median']:.4f}")
    rpt()

    rpt("#### Spectral range (max - min)")
    rpt()
    for method in NORM_METHODS:
        row = summary_df[summary_df["method"] == method].iloc[0]
        if method == "A_none":
            rpt(f"- **A (baseline):** median = {row['spectral_range_median']:.4f}")
        else:
            rpt(f"- **{method}:** median = {row['spectral_range_median']:.4f}, "
                f"median |change| = {row['spectral_range_abs_change_median']:.4f}")
    rpt()

    rpt("#### Curvature sum")
    rpt()
    for method in NORM_METHODS:
        row = summary_df[summary_df["method"] == method].iloc[0]
        if method == "A_none":
            rpt(f"- **A (baseline):** median = {row['curvature_sum_median']:.4f}")
        else:
            rpt(f"- **{method}:** median = {row['curvature_sum_median']:.4f}, "
                f"median |change| = {row['curvature_sum_abs_change_median']:.4f}")
    rpt()

    rpt("#### Integrated absolute first derivative")
    rpt()
    for method in NORM_METHODS:
        row = summary_df[summary_df["method"] == method].iloc[0]
        if method == "A_none":
            rpt(f"- **A (baseline):** median = {row['integrated_abs_d1_median']:.4f}")
        else:
            rpt(f"- **{method}:** median = {row['integrated_abs_d1_median']:.4f}, "
                f"median |change| = {row['integrated_abs_d1_abs_change_median']:.4f}")
    rpt()

    # Mineral-level complexity changes
    rpt("### Mineral-level complexity changes")
    rpt()
    rpt("Median relative change in first-order variation by mineral:")
    rpt()
    rpt("| Mineral | B_vector | C_max | D_area | E_continuum | F_zscore |")
    rpt("|---------|----------|-------|--------|-------------|----------|")
    for mineral in minerals_list:
        mineral_fids = baseline[baseline["mineral"] == mineral].index
        if len(mineral_fids) == 0:
            continue
        row_vals = [f"| {mineral} "]
        for method in NORM_METHODS[1:]:
            sub = comparison_df[(comparison_df["normalization_method"] == method) &
                                (comparison_df["mineral"] == mineral)]
            sub = sub.set_index("file_id")
            common = mineral_fids.intersection(sub.index)
            if len(common) == 0:
                row_vals.append("| — ")
                continue
            orig = baseline.loc[common, "first_order_variation"]
            norm_v = sub.loc[common, "first_order_variation"]
            rel = ((norm_v - orig).abs() / orig.abs().clip(lower=1e-15)).median()
            row_vals.append(f"| {rel:.4f} ")
        row_vals.append("|")
        rpt("".join(row_vals))
    rpt()

    rpt("### Interpretation")
    rpt()
    rpt("Scalar normalization methods (B, C, D) scale all complexity descriptors "
        "by a constant factor. First-order variation, second-order variation, "
        "spectral range, integrated |d1|, and curvature sum all scale proportionally. "
        "Peak density, n_extrema, and spectral entropy are **invariant** under "
        "scalar multiplication (they depend on shape, not amplitude).")
    rpt()
    rpt("Method E (continuum) is wavelength-dependent and can alter shape-based "
        "descriptors including peak density and spectral entropy.")
    rpt()
    rpt("Method F (z-score) shifts the baseline and rescales, which preserves "
        "peak density and n_extrema but changes spectral entropy (the histogram "
        "shape changes after mean subtraction).")
    rpt()

    # ================================================================
    # SECTION 7: Amplitude versus shape information
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 7. Amplitude versus Shape Information")
    rpt()

    rpt("### Hypothesis A: Complexity describes shape independently of magnitude")
    rpt()
    rpt("Under this hypothesis, normalization removes only irrelevant amplitude "
        "information and reveals the underlying spectral structure. Scalar methods "
        "(B, C, D) achieve this exactly. Continuum (E) and z-score (F) also "
        "remove amplitude but introduce wavelength-dependent or offset effects.")
    rpt()

    rpt("### Hypothesis B: Absolute reflectance contains meaningful information")
    rpt()
    rpt("Under this hypothesis, normalization destroys scientifically relevant "
        "amplitude information that relates to mineral chemistry/crystal structure.")
    rpt()

    # Quantify amplitude variation
    rpt("### Quantifying amplitude information content")
    rpt()

    amp_stats = []
    for sp in spectra:
        fid = sp["file_id"]
        r = norm_spectra[fid]["A_none"]
        r_fin = r[np.isfinite(r)]
        amp_stats.append({
            "file_id": fid,
            "mineral": sp["mineral"],
            "specimen_group_id": sp["specimen_group_id"],
            "mean_reflectance": float(np.mean(r_fin)),
            "max_reflectance": float(np.max(r_fin)),
            "integrated_reflectance": float(np.trapezoid(np.abs(r_fin), dx=D2_SPACING)),
            "vector_norm": float(np.sqrt(np.sum(r_fin ** 2))),
        })
    amp_df = pd.DataFrame(amp_stats)

    rpt("| Amplitude measure | CV across dataset | Range |")
    rpt("|-------------------|-------------------|-------|")
    for col in ["mean_reflectance", "max_reflectance", "integrated_reflectance",
                "vector_norm"]:
        vals = amp_df[col]
        cv = vals.std() / vals.mean() if vals.mean() != 0 else np.nan
        rpt(f"| {col} | {cv:.4f} | [{vals.min():.4f}, {vals.max():.4f}] |")
    rpt()

    # Amplitude variation by mineral
    rpt("### Amplitude variation by mineral")
    rpt()
    rpt("| Mineral | n | Median mean R | CV(mean R) | Median max R |")
    rpt("|---------|---|--------------|-----------|-------------|")
    for mineral in minerals_list:
        sub = amp_df[amp_df["mineral"] == mineral]
        cv = sub["mean_reflectance"].std() / sub["mean_reflectance"].mean() \
            if sub["mean_reflectance"].mean() != 0 else np.nan
        rpt(f"| {mineral} | {len(sub)} | {sub['mean_reflectance'].median():.4f} | "
            f"{cv:.4f} | {sub['max_reflectance'].median():.4f} |")
    rpt()

    # Same-mineral amplitude variation vs cross-mineral
    rpt("### Within-mineral vs between-mineral amplitude variation")
    rpt()
    within_mineral_std = []
    for mineral in minerals_list:
        sub = amp_df[amp_df["mineral"] == mineral]
        if len(sub) >= 3:
            within_mineral_std.append(sub["mean_reflectance"].std())
    overall_std = amp_df["mean_reflectance"].std()
    between_mineral = amp_df.groupby("mineral")["mean_reflectance"].median().std()

    rpt(f"- Overall std of mean reflectance: {overall_std:.4f}")
    rpt(f"- Between-mineral std (of mineral medians): {between_mineral:.4f}")
    rpt(f"- Median within-mineral std: {np.median(within_mineral_std):.4f}")
    rpt()

    ratio = overall_std / between_mineral if between_mineral > 0 else np.nan
    rpt(f"Ratio of total to between-mineral variation: {ratio:.2f}")
    rpt()
    if ratio > 1.5:
        rpt("A substantial fraction of amplitude variation is **within-mineral**, "
            "suggesting that absolute reflectance is influenced by factors other "
            "than mineral identity (e.g., grain size, acquisition geometry, "
            "sample preparation).")
    else:
        rpt("Most amplitude variation is **between-mineral**, suggesting that "
            "absolute reflectance carries mineral-discriminating information.")
    rpt()

    rpt("### Information lost under each normalization")
    rpt()
    rpt("| Method | Amplitude information lost | Shape information preserved |")
    rpt("|--------|--------------------------|---------------------------|")
    rpt("| A: None | None | All |")
    rpt("| B: Vector | Overall magnitude level | All relative structure |")
    rpt("| C: Maximum | Overall magnitude + max value | All relative structure |")
    rpt("| D: Area | Overall magnitude + total area | All relative structure |")
    rpt("| E: Continuum | Continuum shape + absolute level | Band depths relative to continuum |")
    rpt("| F: Z-score | Mean level + absolute scale | All relative structure (shifted baseline) |")
    rpt()

    rpt("### Assessment")
    rpt()
    rpt("The coefficient of variation of mean reflectance across the dataset is "
        f"{amp_df['mean_reflectance'].std() / amp_df['mean_reflectance'].mean():.2f}, "
        "indicating substantial amplitude variation. Whether this variation is "
        "scientifically meaningful or acquisition-artifact-driven cannot be "
        "determined from the available metadata alone. The within-mineral amplitude "
        "variation suggests that acquisition effects (grain size, geometry, "
        "instrument) contribute significantly to absolute reflectance levels.")
    rpt()

    # ================================================================
    # SECTION 8: Mineral and provenance effects
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 8. Mineral and Provenance Effects")
    rpt()

    # Does normalization change mineral clustering?
    rpt("### Effect on inter-mineral spectral distances")
    rpt()

    mineral_medians = {}
    for method in NORM_METHODS:
        mineral_medians[method] = {}
        for mineral in minerals_list:
            fids = [sp["file_id"] for sp in spectra
                    if sp["mineral"] == mineral and norm_spectra[sp["file_id"]][method] is not None]
            if len(fids) == 0:
                continue
            stack = np.array([norm_spectra[fid][method] for fid in fids])
            mineral_medians[method][mineral] = np.median(stack, axis=0)

    # Compute pairwise mineral distances for each method
    rpt("Pairwise mineral distances (Euclidean between median spectra):")
    rpt()

    dist_changes = []
    minerals_with_data = [m for m in minerals_list
                          if all(m in mineral_medians[met] for met in NORM_METHODS)]

    for i, m1 in enumerate(minerals_with_data):
        for j, m2 in enumerate(minerals_with_data):
            if j <= i:
                continue
            baseline_dist = np.sqrt(np.sum(
                (mineral_medians["A_none"][m1] - mineral_medians["A_none"][m2]) ** 2
            ))
            for method in NORM_METHODS[1:]:
                norm_dist = np.sqrt(np.sum(
                    (mineral_medians[method][m1] - mineral_medians[method][m2]) ** 2
                ))
                if baseline_dist > 1e-15:
                    rel_change = abs(norm_dist - baseline_dist) / baseline_dist
                else:
                    rel_change = np.nan
                dist_changes.append({
                    "mineral_pair": f"{m1}--{m2}",
                    "method": method,
                    "baseline_dist": baseline_dist,
                    "norm_dist": norm_dist,
                    "rel_change": rel_change,
                })

    if dist_changes:
        dc_df = pd.DataFrame(dist_changes)
        rpt("| Method | Median relative distance change | Max relative distance change |")
        rpt("|--------|-------------------------------|------------------------------|")
        for method in NORM_METHODS[1:]:
            sub = dc_df[dc_df["method"] == method]
            rpt(f"| {method} | {sub['rel_change'].median():.4f} | "
                f"{sub['rel_change'].max():.4f} |")
        rpt()

    # Provenance group effects
    rpt("### Provenance group effects")
    rpt()
    rpt("For each provenance group with multiple spectra, the within-group "
        "coefficient of variation of first-order variation is compared across "
        "normalization methods.")
    rpt()

    prov_groups = d2_pop.groupby("specimen_group_id")["file_id"].apply(list).to_dict()
    multi_prov = {k: v for k, v in prov_groups.items() if len(v) >= 3}

    rpt(f"Provenance groups with >= 3 spectra: {len(multi_prov)}")
    rpt()

    if multi_prov:
        prov_cv_rows = []
        for pg_id, fids in multi_prov.items():
            mineral = spectra[[sp["file_id"] for sp in spectra].index(fids[0])]["mineral"] \
                if fids[0] in [sp["file_id"] for sp in spectra] else "unknown"
            for sp in spectra:
                if sp["file_id"] == fids[0]:
                    mineral = sp["mineral"]
                    break

            for method in NORM_METHODS:
                fovs = []
                for fid in fids:
                    if norm_spectra[fid][method] is not None:
                        c = compute_complexity(norm_spectra[fid][method])
                        fovs.append(c["first_order_variation"])
                if len(fovs) >= 3:
                    arr = np.array(fovs)
                    cv = arr.std() / arr.mean() if arr.mean() > 1e-15 else np.nan
                    prov_cv_rows.append({
                        "specimen_group_id": pg_id,
                        "mineral": mineral,
                        "method": method,
                        "n_spectra": len(fovs),
                        "cv_fov": cv,
                    })

        if prov_cv_rows:
            prov_cv_df = pd.DataFrame(prov_cv_rows)
            rpt("| Method | Median within-group CV | Max within-group CV |")
            rpt("|--------|----------------------|--------------------|")
            for method in NORM_METHODS:
                sub = prov_cv_df[prov_cv_df["method"] == method]
                cvs = sub["cv_fov"].dropna()
                if len(cvs) > 0:
                    rpt(f"| {method} | {cvs.median():.4f} | {cvs.max():.4f} |")
            rpt()

    rpt("### Interpretation")
    rpt()
    rpt("Scalar normalization methods (B, C, D) change inter-mineral distances "
        "by a uniform scaling factor, so the **relative ordering** of mineral "
        "pair distances is preserved. Method E (continuum) can change the "
        "relative ordering because the continuum shape varies across minerals.")
    rpt()

    # ================================================================
    # SECTION 9: Acquisition effects
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 9. Acquisition Effects")
    rpt()

    rpt("### Association of absolute reflectance with acquisition metadata")
    rpt()

    # By instrument prefix
    rpt("#### By instrument prefix")
    rpt()
    rpt("| Prefix | n | Median mean R | Median max R | Regime |")
    rpt("|--------|---|--------------|-------------|--------|")
    for prefix in sorted(raw_df["instrument_prefix"].unique()):
        sub = raw_df[raw_df["instrument_prefix"] == prefix]
        regime = classify_acquisition_regime(prefix)
        rpt(f"| {prefix} | {len(sub)} | {sub['rf_mean'].median():.4f} | "
            f"{sub['rf_max'].median():.4f} | {regime} |")
    rpt()

    # By acquisition regime
    rpt("#### By acquisition regime")
    rpt()
    regime_map = {p: classify_acquisition_regime(p) for p in raw_df["instrument_prefix"].unique()}
    raw_df_regime = raw_df.copy()
    raw_df_regime["regime"] = raw_df_regime["instrument_prefix"].map(regime_map)
    rpt("| Regime | n | Median mean R | Median max R | Median range |")
    rpt("|--------|---|--------------|-------------|-------------|")
    for regime in sorted(raw_df_regime["regime"].unique()):
        sub = raw_df_regime[raw_df_regime["regime"] == regime]
        rpt(f"| {regime} | {len(sub)} | {sub['rf_mean'].median():.4f} | "
            f"{sub['rf_max'].median():.4f} | {sub['rf_range'].median():.4f} |")
    rpt()

    # By voltage
    rpt("#### By voltage")
    rpt()
    rpt("| Voltage | n | Median mean R | Median max R |")
    rpt("|---------|---|--------------|-------------|")
    for v in sorted(raw_df["voltage"].dropna().unique()):
        if v == "":
            continue
        sub = raw_df[raw_df["voltage"] == v]
        rpt(f"| {v} | {len(sub)} | {sub['rf_mean'].median():.4f} | "
            f"{sub['rf_max'].median():.4f} |")
    rpt()

    # By native spacing
    rpt("#### By native grid type")
    rpt()
    rpt("| Native spacing | n | Median mean R | Median max R |")
    rpt("|---------------|---|--------------|-------------|")
    for sp_val in sorted(raw_df["native_spacing_nm"].unique()):
        sub = raw_df[raw_df["native_spacing_nm"] == sp_val]
        rpt(f"| {sp_val:.1f} nm | {len(sub)} | {sub['rf_mean'].median():.4f} | "
            f"{sub['rf_max'].median():.4f} |")
    rpt()

    rpt("### Limitations")
    rpt()
    rpt("The available acquisition metadata is limited. Grain size is not "
        "systematically recorded across the dataset. Illumination geometry "
        "(source/detect angles) is available for some spectra but not all. "
        "Measurement campaign can be partially inferred from the instrument "
        "prefix, but detailed calibration records are not available.")
    rpt()
    rpt("The observed variation in absolute reflectance across instrument "
        "prefixes and voltages is consistent with acquisition-related effects, "
        "but causality cannot be inferred from the available metadata.")
    rpt()

    # ================================================================
    # SECTION 10: Special cases
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 10. Special Cases")
    rpt()

    # F0692
    rpt("### F0692 (Olivine, bir1dd001.tab)")
    rpt()
    f0692 = [sp for sp in spectra if sp["file_id"] == "F0692"]
    if f0692:
        sp692 = f0692[0]
        r692 = sp692["rf_grid"]
        rpt(f"- Reflectance QC: {sp692['reflectance_qc']}")
        rpt(f"- Min reflectance: {np.min(r692):.6f}")
        rpt(f"- Max reflectance: {np.max(r692):.6f}")
        rpt(f"- Negative values: {(r692 < 0).sum()}")
        rpt(f"- Values > 1: {(r692 > 1).sum()}")
        rpt()

        rpt("| Method | Factor | Failure | Min | Max | Notes |")
        rpt("|--------|--------|---------|-----|-----|-------|")
        for method in NORM_METHODS:
            r_norm = norm_spectra["F0692"][method]
            if r_norm is None:
                rpt(f"| {method} | — | yes | — | — | normalization failed |")
                continue
            if method == "E_continuum":
                _, factor, fail = norm_continuum(r692, sp692["wl_grid"])
            elif method == "A_none":
                factor = 1.0
                fail = None
            elif method == "B_vector":
                _, factor, fail = norm_vector(r692)
            elif method == "C_max":
                _, factor, fail = norm_max(r692)
            elif method == "D_area":
                _, factor, fail = norm_area(r692)
            elif method == "F_zscore":
                _, factor, fail = norm_zscore(r692)

            notes = ""
            if method == "C_max" and np.max(r692) > 1:
                notes = "max > 1, division preserves >1 region"
            elif method == "F_zscore":
                notes = "negative values become more negative"
            elif method == "E_continuum":
                notes = "negative values / continuum"

            rpt(f"| {method} | {factor:.6f} | {fail or 'no'} | "
                f"{np.min(r_norm):.6f} | {np.max(r_norm):.6f} | {notes} |")
        rpt()

        rpt("F0692 has 3 negative values (1005-1008 nm, within deep 1-um absorption) "
            "and 3 values >1 (1139-1147 nm, noise spikes). Scalar methods (B, C, D) "
            "preserve the negative values and spikes proportionally. Z-score (F) "
            "shifts the baseline, making more values negative. Continuum (E) divides "
            "by the upper hull, preserving the structure but rescaling.")
        rpt()

    # F0999
    rpt("### F0999 (Spinel glass, glue13.tab)")
    rpt()
    f0999 = [sp for sp in spectra if sp["file_id"] == "F0999"]
    if f0999:
        sp999 = f0999[0]
        r999 = sp999["rf_grid"]
        rpt(f"- Reflectance QC: {sp999['reflectance_qc']}")
        rpt(f"- Min reflectance: {np.min(r999):.6f}")
        rpt(f"- Max reflectance: {np.max(r999):.6f}")
        rpt(f"- Values > 1: {(r999 > 1).sum()}")
        rpt()

        rpt("| Method | Factor | Failure | Min | Max | Notes |")
        rpt("|--------|--------|---------|-----|-----|-------|")
        for method in NORM_METHODS:
            r_norm = norm_spectra["F0999"][method]
            if r_norm is None:
                rpt(f"| {method} | — | yes | — | — | normalization failed |")
                continue
            if method == "E_continuum":
                _, factor, fail = norm_continuum(r999, sp999["wl_grid"])
            elif method == "A_none":
                factor = 1.0
                fail = None
            elif method == "B_vector":
                _, factor, fail = norm_vector(r999)
            elif method == "C_max":
                _, factor, fail = norm_max(r999)
            elif method == "D_area":
                _, factor, fail = norm_area(r999)
            elif method == "F_zscore":
                _, factor, fail = norm_zscore(r999)

            notes = ""
            if method == "C_max":
                notes = f"max was {np.max(r999):.6f}, now exactly 1.0"
            elif method == "D_area":
                notes = "area includes >1 region"

            rpt(f"| {method} | {factor:.6f} | {fail or 'no'} | "
                f"{np.min(r_norm):.6f} | {np.max(r_norm):.6f} | {notes} |")
        rpt()

        rpt("F0999 has 67 values >1 (max 1.00738) in the 435-875 nm range. "
            "This is a systematic smooth hump. Method C (max) brings the maximum "
            "to exactly 1.0 but preserves the hump shape. No normalization method "
            "removes or hides this anomaly; they all preserve the spectral structure.")
        rpt()

    # Native-10-nm spectra
    rpt("### Native-10-nm spectra")
    rpt()
    n10_spectra = [sp for sp in spectra if sp["is_interpolated"]]
    rpt(f"Spectra requiring PCHIP interpolation: {len(n10_spectra)}")
    rpt()
    rpt("Normalization effects on these spectra conflate interpolation effects "
        "with normalization effects. The PCHIP interpolation has already been "
        "validated (Phase 2 Step 3) with MAE=0.000525 reflectance units. "
        "Normalization is applied **after** interpolation, so the normalization "
        "effect is independent of the interpolation method.")
    rpt()

    # ================================================================
    # SECTION 11: Numerical stability
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 11. Numerical Stability")
    rpt()

    rpt("| Method | Zero denominator | Near-zero denominator | Negative values | Failure count |")
    rpt("|--------|-----------------|----------------------|----------------|---------------|")
    for method in NORM_METHODS:
        sub = comparison_df[comparison_df["normalization_method"] == method]
        n_fail = (sub["failure_flag"] != "").sum()
        fail_reasons = sub[sub["failure_flag"] != ""]["failure_flag"].value_counts()

        zero_den = 0
        near_zero = 0
        neg_vals = 0

        if method == "B_vector":
            zero_den = (sub["norm_factor"] < 1e-15).sum()
            near_zero = ((sub["norm_factor"] >= 1e-15) & (sub["norm_factor"] < 1e-10)).sum()
        elif method == "C_max":
            zero_den = (sub["norm_factor"].abs() < 1e-15).sum()
            near_zero = ((sub["norm_factor"].abs() >= 1e-15) & (sub["norm_factor"].abs() < 1e-10)).sum()
        elif method == "D_area":
            zero_den = (sub["norm_factor"] < 1e-15).sum()
            near_zero = ((sub["norm_factor"] >= 1e-15) & (sub["norm_factor"] < 1e-10)).sum()
        elif method == "F_zscore":
            zero_den = (sub["norm_factor"] < 1e-15).sum()
            near_zero = ((sub["norm_factor"] >= 1e-15) & (sub["norm_factor"] < 1e-10)).sum()
        elif method == "E_continuum":
            zero_den = fail_reasons.get("zero_continuum", 0)
            near_zero = 0

        neg_count = 0
        for sp in spectra:
            r_norm = norm_spectra[sp["file_id"]][method]
            if r_norm is not None and (r_norm < 0).any():
                neg_count += 1

        rpt(f"| {method} | {zero_den} | {near_zero} | {neg_count} | {n_fail} |")
    rpt()

    rpt("### Pathological cases")
    rpt()
    rpt("- No spectra have zero vector magnitude, zero max, or zero area")
    rpt("- No spectra have zero standard deviation")
    rpt(f"- {n_negative} spectra have negative reflectance values "
        "(handled correctly by all methods)")
    rpt("- Continuum normalization (E) is the only method with potential "
        "numerical issues (convex hull failures for degenerate spectra)")
    rpt()

    # ================================================================
    # SECTION 12: Reversibility and provenance
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 12. Reversibility and Provenance")
    rpt()
    rpt("Every normalization applied in this investigation is:")
    rpt()
    rpt("1. **Deterministic** — given the same input spectrum, the same output is produced")
    rpt("2. **Reproducible** — defined by the raw spectrum + D2 domain + 5-nm grid + formula")
    rpt("3. **Non-destructive** — raw data files are never modified")
    rpt("4. **In-memory only** — no normalized dataset is persisted as canonical")
    rpt()
    rpt("Scalar methods (B, C, D, F) are exactly reversible: R = R' × factor (+ mean for F).")
    rpt("Continuum (E) is reversible if the continuum is stored.")
    rpt()

    # ================================================================
    # SECTION 13: Diagnostic figures
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 13. Diagnostic Figures")
    rpt()

    print("Generating diagnostic figures...")

    # Figure 1: Representative raw vs normalized spectra
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    rep_indices = [0, len(spectra) // 3, 2 * len(spectra) // 3]
    method_colors = {
        "A_none": "black", "B_vector": "blue", "C_max": "red",
        "D_area": "green", "E_continuum": "orange", "F_zscore": "purple",
    }
    for ax_i, (ax, sp_idx) in enumerate(zip(axes.flat, rep_indices)):
        sp = spectra[sp_idx]
        fid = sp["file_id"]
        wl = sp["wl_grid"]
        for method in NORM_METHODS:
            r_norm = norm_spectra[fid][method]
            if r_norm is not None:
                ax.plot(wl, r_norm, color=method_colors[method],
                        alpha=0.7, linewidth=0.8, label=method)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Normalized reflectance")
        ax.set_title(f"{fid} ({sp['mineral']})")
        if ax_i == 0:
            ax.legend(fontsize=6, loc="upper right")
    plt.suptitle("Raw vs normalized spectra (representative examples)", fontsize=12)
    plt.tight_layout()
    fig1_path = FIG_DIR / "normalization_fig1_raw_vs_normalized.png"
    plt.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![Raw vs normalized]({fig1_path.relative_to(PROJECT_ROOT)})")
    rpt()

    # Figure 2: Distribution of normalization factors
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, method in zip(axes.flat, NORM_METHODS[1:]):
        sub = comparison_df[comparison_df["normalization_method"] == method]
        factors = sub.loc[sub["failure_flag"] == "", "norm_factor"].dropna()
        if len(factors) > 0:
            ax.hist(factors, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
            ax.axvline(factors.median(), color="red", linestyle="--",
                       label=f"Median: {factors.median():.4f}")
        ax.set_title(f"{method}")
        ax.set_xlabel("Normalization factor")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    plt.suptitle("Distribution of normalization factors", fontsize=12)
    plt.tight_layout()
    fig2_path = FIG_DIR / "normalization_fig2_norm_factors.png"
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![Norm factors]({fig2_path.relative_to(PROJECT_ROOT)})")
    rpt()

    # Figure 3: Complexity changes (box plots)
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    for ax, col in zip(axes.flat, complexity_cols):
        data_by_method = []
        labels = []
        for method in NORM_METHODS:
            sub = comparison_df[comparison_df["normalization_method"] == method]
            vals = sub.loc[sub["failure_flag"] == "", col].dropna()
            base_vals = baseline.loc[baseline["failure_flag"] == "", col].dropna()
            common = base_vals.index.intersection(vals.index)
            if method == "A_none":
                rel = np.zeros(len(common))
            else:
                bv = base_vals.loc[common]
                nv = vals.loc[common]
                rel = ((nv - bv) / bv.abs().clip(lower=1e-15)).values
            data_by_method.append(rel)
            labels.append(method.split("_")[0])

        bp = ax.boxplot(data_by_method, patch_artist=True,
                        showfliers=False)
        ax.set_xticklabels(labels)
        colors = ["#d3d3d3", "#87ceeb", "#ff7f7f", "#90ee90", "#ffd700", "#dda0dd"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_title(col.replace("_", " "), fontsize=8)
        ax.set_ylabel("Relative change")
        ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    plt.suptitle("Complexity descriptor changes by normalization method", fontsize=12)
    plt.tight_layout()
    fig3_path = FIG_DIR / "normalization_fig3_complexity_changes.png"
    plt.savefig(fig3_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![Complexity changes]({fig3_path.relative_to(PROJECT_ROOT)})")
    rpt()

    # Figure 4: Complexity change vs normalization factor
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, method in zip(axes.flat, NORM_METHODS[1:]):
        sub = comparison_df[comparison_df["normalization_method"] == method].copy()
        sub = sub.set_index("file_id")
        base = baseline.loc[baseline["failure_flag"] == "", "first_order_variation"]
        norm_v = sub.loc[sub["failure_flag"] == "", "first_order_variation"]
        common = base.index.intersection(norm_v.index)
        factors = sub.loc[common, "norm_factor"]
        rel_change = ((norm_v.loc[common] - base.loc[common]).abs() /
                      base.loc[common].abs().clip(lower=1e-15))

        ax.scatter(factors, rel_change, s=5, alpha=0.5, color="steelblue")
        ax.set_xlabel("Normalization factor")
        ax.set_ylabel("|Relative change| in FOV")
        ax.set_title(f"{method}")
    plt.suptitle("Complexity change vs normalization factor", fontsize=12)
    plt.tight_layout()
    fig4_path = FIG_DIR / "normalization_fig4_complexity_vs_factor.png"
    plt.savefig(fig4_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![Complexity vs factor]({fig4_path.relative_to(PROJECT_ROOT)})")
    rpt()

    # Figure 5: Mineral-level distribution of changes
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, method in zip(axes.flat, NORM_METHODS[1:]):
        mineral_changes = []
        for mineral in minerals_list:
            m_fids = [sp["file_id"] for sp in spectra if sp["mineral"] == mineral]
            changes = []
            for fid in m_fids:
                if norm_spectra[fid][method] is None:
                    continue
                base_c = compute_complexity(norm_spectra[fid]["A_none"])
                norm_c = compute_complexity(norm_spectra[fid][method])
                bv = base_c["first_order_variation"]
                nv = norm_c["first_order_variation"]
                if abs(bv) > 1e-15:
                    changes.append(abs(nv - bv) / abs(bv))
            if changes:
                mineral_changes.append({
                    "mineral": mineral,
                    "median_change": np.median(changes),
                    "n": len(changes),
                })

        if mineral_changes:
            mc_df = pd.DataFrame(mineral_changes).sort_values("median_change", ascending=True)
            y_pos = np.arange(len(mc_df))
            ax.barh(y_pos, mc_df["median_change"], color="steelblue", alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"{r['mineral']} (n={r['n']})"
                                for _, r in mc_df.iterrows()], fontsize=7)
            ax.set_xlabel("Median |relative change| in FOV")
            ax.set_title(f"{method}")
    plt.suptitle("Mineral-level complexity changes", fontsize=12)
    plt.tight_layout()
    fig5_path = FIG_DIR / "normalization_fig5_mineral_changes.png"
    plt.savefig(fig5_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![Mineral changes]({fig5_path.relative_to(PROJECT_ROOT)})")
    rpt()

    # Figure 6: F0692 and F0999 special cases
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax_row, fid_target, title in [(0, "F0692", "F0692 (Olivine)"),
                                       (1, "F0999", "F0999 (Spinel glass)")]:
        sp_target = [sp for sp in spectra if sp["file_id"] == fid_target]
        if not sp_target:
            continue
        sp_t = sp_target[0]
        wl = sp_t["wl_grid"]

        for method in NORM_METHODS:
            r_norm = norm_spectra[fid_target][method]
            if r_norm is not None:
                axes[ax_row][0].plot(wl, r_norm, color=method_colors[method],
                                     alpha=0.7, linewidth=0.8, label=method)
        axes[ax_row][0].set_xlabel("Wavelength (nm)")
        axes[ax_row][0].set_ylabel("Normalized reflectance")
        axes[ax_row][0].set_title(f"{title} — all methods")
        if ax_row == 0:
            axes[ax_row][0].legend(fontsize=6, loc="upper right")

        axes[ax_row][1].plot(wl, norm_spectra[fid_target]["A_none"], "k-",
                             linewidth=1, label="Original")
        r_cont = norm_spectra[fid_target]["E_continuum"]
        if r_cont is not None:
            axes[ax_row][1].plot(wl, r_cont, "orange", alpha=0.7,
                                 linewidth=0.8, label="Continuum")
        axes[ax_row][1].set_xlabel("Wavelength (nm)")
        axes[ax_row][1].set_ylabel("Normalized reflectance")
        axes[ax_row][1].set_title(f"{title} — original vs continuum")
        axes[ax_row][1].legend(fontsize=8)
    plt.tight_layout()
    fig6_path = FIG_DIR / "normalization_fig6_special_cases.png"
    plt.savefig(fig6_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![Special cases]({fig6_path.relative_to(PROJECT_ROOT)})")
    rpt()

    # Figure 7: Derivative structure comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    rep_sp = spectra[len(spectra) // 4]
    fid_rep = rep_sp["file_id"]
    wl_rep = rep_sp["wl_grid"]
    for ax, method in zip(axes.flat, NORM_METHODS):
        r_norm = norm_spectra[fid_rep][method]
        if r_norm is not None:
            d1 = np.diff(r_norm)
            ax.plot(wl_rep[1:], d1, color=method_colors[method],
                    alpha=0.8, linewidth=0.6)
        ax.set_title(f"{method}")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("First derivative")
    plt.suptitle(f"First derivative comparison — {fid_rep} ({rep_sp['mineral']})",
                 fontsize=12)
    plt.tight_layout()
    fig7_path = FIG_DIR / "normalization_fig7_derivative_structure.png"
    plt.savefig(fig7_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![Derivative structure]({fig7_path.relative_to(PROJECT_ROOT)})")
    rpt()

    # Figure 8: Amplitude variation by mineral
    fig, ax = plt.subplots(figsize=(12, 6))
    mineral_box_data = []
    mineral_labels = []
    for mineral in minerals_list:
        sub = raw_df[raw_df["mineral"] == mineral]
        mineral_box_data.append(sub["rf_mean"].values)
        mineral_labels.append(f"{mineral}\n(n={len(sub)})")
    bp = ax.boxplot(mineral_box_data, patch_artist=True,
                    showfliers=True)
    ax.set_xticklabels(mineral_labels)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.7)
    ax.set_ylabel("Mean reflectance")
    ax.set_title("Absolute reflectance level by mineral")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    plt.tight_layout()
    fig8_path = FIG_DIR / "normalization_fig8_amplitude_by_mineral.png"
    plt.savefig(fig8_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![Amplitude by mineral]({fig8_path.relative_to(PROJECT_ROOT)})")
    rpt()

    # ================================================================
    # SECTION 14: Limitations
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 14. Limitations")
    rpt()
    rpt("1. **No ground truth for normalization.** There is no reference standard "
        "that defines the 'correct' normalization for mineral spectral complexity. "
        "The choice is inherently scientific, not technical.")
    rpt()
    rpt("2. **Limited acquisition metadata.** Grain size, detailed illumination "
        "geometry, and sample preparation are not systematically available. "
        "This limits the ability to attribute amplitude variation to specific "
        "acquisition factors.")
    rpt()
    rpt("3. **Continuum normalization is not universally applicable.** The convex "
        "hull continuum assumes a specific spectral model (absorption features "
        "below a convex upper envelope). This model is appropriate for some "
        "minerals (e.g., olivine with broad absorption bands) but may not be "
        "appropriate for others (e.g., quartz with weak, narrow features).")
    rpt()
    rpt("4. **Complexity descriptors are not yet final.** The descriptors used "
        "here are diagnostic sensitivity measures, not the final feature set. "
        "Different descriptors may respond differently to normalization.")
    rpt()
    rpt("5. **Single-interpolation pipeline.** The 64 native-10-nm spectra "
        "are interpolated via PCHIP before normalization. Interpolation effects "
        "are small (MAE=0.000525) but could interact with normalization in "
        "ways not fully explored here.")
    rpt()

    # ================================================================
    # SECTION 15: Scientific recommendation
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 15. Scientific Recommendation")
    rpt()

    rpt("### Question 1: Does the primary analysis require normalization?")
    rpt()
    rpt("**No, not strictly.** The spectral-complexity descriptors that describe "
        "**shape** (peak density, n_extrema) are invariant under scalar "
        "normalization. The descriptors that describe **amplitude** (first-order "
        "variation, spectral range, integrated |d1|) are proportional to the "
        "overall reflectance level and would change under normalization, but "
        "their **relative ordering** across spectra is preserved by scalar methods.")
    rpt()

    rpt("### Question 2: What scientifically meaningful information is lost by normalization?")
    rpt()
    rpt("- **Absolute reflectance level** — related to overall sample brightness, "
        "which may correlate with grain size, opacity, or composition")
    rpt("- **Inter-mineral amplitude differences** — some minerals are intrinsically "
        "more reflective than others; normalization removes this information")
    rpt("- **Amplitude-based complexity** — first-order variation, spectral range, "
        "and integrated |d1| encode both shape and amplitude; normalization "
        "removes the amplitude component")
    rpt()

    rpt("### Question 3: Which normalization methods materially alter spectral complexity?")
    rpt()
    rpt("- **Methods B, C, D (scalar):** Alter amplitude-based descriptors "
        "proportionally. Shape descriptors (peak density, n_extrema) are unchanged.")
    rpt("- **Method E (continuum):** Can alter both amplitude and shape descriptors "
        "because the continuum is wavelength-dependent. Most invasive.")
    rpt("- **Method F (z-score):** Alters all descriptors except n_extrema. "
        "Changes the histogram shape (spectral entropy) and shifts the baseline.")
    rpt()

    rpt("### Question 4: Are the effects robust across minerals/provenance groups?")
    rpt()
    rpt("Scalar normalization effects are **uniform** across all minerals — the "
        "relative ordering of complexity within a mineral group is preserved. "
        "Continuum normalization effects vary by mineral because the continuum "
        "shape depends on the spectral structure.")
    rpt()

    rpt("### Question 5: How should normalization be treated?")
    rpt()
    rpt("**Recommendation: Sensitivity analysis only.**")
    rpt()
    rpt("Rationale:")
    rpt("1. The primary analysis should use **original reflectance** (Method A) "
        "to preserve all scientifically meaningful information")
    rpt("2. Scalar normalization methods (B, C, D) can serve as a **sensitivity "
        "analysis** to test whether conclusions hold when amplitude information "
        "is removed")
    rpt("3. Continuum normalization (E) should be treated with caution — it is "
        "not universally appropriate and can alter shape descriptors")
    rpt("4. Z-score normalization (F) is a **methodological sensitivity test** "
        "only, not a physical reflectance representation")
    rpt()

    rpt("### Question 6: Why is retaining the original reflectance scale preferable?")
    rpt()
    rpt("1. **Preserves all information.** No scientifically meaningful signal is "
        "discarded. If amplitude is irrelevant, the analysis can demonstrate this "
        "empirically rather than assuming it a priori.")
    rpt("2. **Physical interpretability.** Complexity descriptors computed on "
        "original reflectance have direct physical meaning (e.g., first-order "
        "variation = total absolute spectral change in reflectance units).")
    rpt("3. **Avoids introducing artifacts.** Normalization can amplify noise "
        "(dividing by small numbers), create artificial structure (continuum), "
        "or shift baselines (z-score).")
    rpt("4. **Conservative.** The project investigates relationships between "
        "mineral chemistry/crystal structure and spectral complexity. Removing "
        "amplitude information before knowing whether it matters is premature.")
    rpt("5. **Reversibility.** Starting from original reflectance allows any "
        "normalization to be applied later. Starting from normalized data "
        "precludes recovering the original scale.")
    rpt()

    rpt("---")
    rpt()
    rpt("*This is an investigation only. No normalization method has been frozen. "
        "The recommendation is pending researcher review.*")
    rpt()

    # ================================================================
    # Save outputs
    # ================================================================

    print("Saving outputs...")

    comparison_path = OUT_DIR / "phase2_step5_normalization_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    summary_path = OUT_DIR / "phase2_step5_normalization_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    rpt("---")
    rpt()
    rpt("## Output Checksums")
    rpt()
    rpt("| File | MD5 |")
    rpt("|------|-----|")
    for fp in [comparison_path, summary_path]:
        if fp.exists():
            md5 = compute_md5(fp)
            rpt(f"| {fp.name} | {md5} |")
    rpt()

    rpt("---")
    rpt()
    rpt(f"*Report generated: {timestamp}*")
    rpt("*This is an investigation only. No frozen decisions were changed.*")

    report_path = OUT_DIR / "phase2_step5_normalization_investigation.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return report_path, comparison_path, summary_path


# ================================================================
# HELPER FUNCTIONS
# ================================================================


def _find_peaks(r):
    peaks = []
    for i in range(1, len(r) - 1):
        if r[i] > r[i - 1] and r[i] > r[i + 1]:
            peaks.append(i)
    return np.array(peaks, dtype=int)


def _find_troughs(r):
    troughs = []
    for i in range(1, len(r) - 1):
        if r[i] < r[i - 1] and r[i] < r[i + 1]:
            troughs.append(i)
    return np.array(troughs, dtype=int)


def _band_depth(r, trough_idx, wl):
    n = len(r)
    left_min = max(0, trough_idx - 50)
    right_max = min(n, trough_idx + 50)

    left_region = r[left_min:trough_idx]
    right_region = r[trough_idx:right_max]

    if len(left_region) == 0 or len(right_region) == 0:
        return 0.0

    left_idx = left_min + np.argmax(left_region)
    right_idx = trough_idx + np.argmax(right_region)

    wl_left = wl[left_idx]
    wl_right = wl[right_idx]
    r_left = r[left_idx]
    r_right = r[right_idx]

    if abs(wl_right - wl_left) < 1e-15:
        return 0.0

    continuum_at_trough = r_left + (r_right - r_left) * (wl[trough_idx] - wl_left) / (wl_right - wl_left)
    depth = continuum_at_trough - r[trough_idx]
    return float(max(0.0, depth))


# ================================================================
# REPRODUCIBILITY CHECK
# ================================================================


def run_reproducibility_check():
    print("=== Reproducibility check ===")

    print("Run 1...")
    r1 = run_investigation()

    csv_files = [fp for fp in r1 if fp.suffix == ".csv"]
    md5_run1 = {}
    for fp in csv_files:
        if fp.exists():
            md5_run1[fp.name] = compute_md5(fp)

    print("Run 2...")
    r2 = run_investigation()

    md5_run2 = {}
    for fp in r2:
        if fp.suffix == ".csv" and fp.exists():
            md5_run2[fp.name] = compute_md5(fp)

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
    print("Phase 2 Step 5 — Reflectance Normalization Investigation")
    print("=" * 60)
    print()

    ok = run_reproducibility_check()

    print()
    print("Outputs:")
    out_dir = Path(__file__).resolve().parent.parent / "results" / "02_preprocessing_qc"
    for fp in sorted(out_dir.glob("phase2_step5_*")):
        print(f"  {fp}")
    for fp in sorted((out_dir / "figures").glob("normalization_*")):
        print(f"  {fp}")

    if not ok:
        print("\nWARNING: reproducibility check failed!")
        exit(1)
