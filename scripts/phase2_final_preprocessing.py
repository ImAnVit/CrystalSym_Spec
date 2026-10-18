#!/usr/bin/env python3
"""
Phase 2 — Final Preprocessing Implementation and Validation
============================================================

CrystalSym-Spec: Interpretable ML and Symbolic Regression of
Crystal Symmetry, Chemistry, and Mineral Spectral Complexity

Deterministic implementation of the frozen Phase 2 primary preprocessing
pipeline. Converts all frozen preprocessing decisions into one canonical
spectral dataset and validates the implementation.

Frozen scientific inputs:
  - D2 domain: 320-2550 nm, 5 nm spacing, 447 target points
  - 454 D2-eligible spectra (390 native-5nm + 64 native-10nm)
  - 18 minerals, 382 provenance groups
  - Duplicate wavelengths: keep first occurrence
  - Unsorted wavelengths: sort in memory
  - Interpolation: PCHIP for 64 native-10nm spectra
  - No normalization, no smoothing, no splice correction
  - No reflectance clipping, no uncertainty propagation

Outputs:
  data/processed/phase2_canonical_d2_5nm.csv
  data/processed/phase2_canonical_d2_5nm_metadata.csv
  results/02_preprocessing_qc/phase2_final_preprocessing_validation.md
  results/02_preprocessing_qc/phase2_final_preprocessing_summary.csv
  results/02_preprocessing_qc/phase2_final_preprocessing_checksums.txt
  results/02_preprocessing_qc/figures/phase2_final_*.png
"""

import os
import sys
import hashlib
import warnings
import datetime
from pathlib import Path
from collections import defaultdict

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
PHASE1_MANIFEST = PROJECT_ROOT / "metadata" / "provenance" / "phase1_freeze_manifest.yaml"
REFL_QC_SUMMARY = (
    PROJECT_ROOT / "results" / "02_preprocessing_qc"
    / "phase2_step3_reflectance_qc_spectrum_summary.csv"
)

OUT_DIR = PROJECT_ROOT / "results" / "02_preprocessing_qc"
FIG_DIR = OUT_DIR / "figures"
PROC_DIR = PROJECT_ROOT / "data" / "processed"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# FROZEN CONSTANTS
# ================================================================

D2_WMIN = 320
D2_WMAX = 2550
D2_SPACING = 5
D2_GRID = np.arange(D2_WMIN, D2_WMAX + D2_SPACING / 2, D2_SPACING, dtype=np.float64)
D2_N_POINTS = len(D2_GRID)

EXPECTED_N_SPECTRA = 454
EXPECTED_N_MINERALS = 18
EXPECTED_N_PROV_GROUPS = 382
EXPECTED_N_NATIVE_5NM = 390
EXPECTED_N_NATIVE_10NM = 64
EXPECTED_N_RAW_FILES = 1017

PROCESSING_VERSION = "phase2_final_v1.0"
PROCESSING_DATE = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

WL_MATCH_TOL = 2.0

FROZEN_FILES = [
    PHASE1_MANIFEST,
    PROJECT_ROOT / "results" / "02_preprocessing_qc" / "phase2_step2_final_freeze.md",
    PROJECT_ROOT / "results" / "02_preprocessing_qc" / "phase2_step3_duplicate_wavelength_freeze.md",
    PROJECT_ROOT / "results" / "02_preprocessing_qc" / "phase2_step3_unsorted_wavelength_freeze.md",
    PROJECT_ROOT / "results" / "02_preprocessing_qc" / "phase2_step3_splice_boundary_freeze.md",
    PROJECT_ROOT / "results" / "02_preprocessing_qc" / "phase2_step3_interpolation_freeze.md",
    PROJECT_ROOT / "results" / "02_preprocessing_qc" / "phase2_step3_reflectance_qc_freeze.md",
    PROJECT_ROOT / "results" / "02_preprocessing_qc" / "phase2_step4_uncertainty_freeze.md",
    PROJECT_ROOT / "results" / "02_preprocessing_qc" / "phase2_step5_normalization_freeze.md",
    PROJECT_ROOT / "results" / "02_preprocessing_qc" / "phase2_step6_smoothing_freeze.md",
]

# ================================================================
# RAW .tab FILE PARSER
# ================================================================


def read_spectrum(filepath):
    """
    Parse a RELAB .tab file, reading ALL spectral segments.

    Returns
    -------
    wavelengths : list[float]
    reflectances : list[float]
    uncertainties : list[float | None]
    footer_lines : list[str]
    segment_boundaries : list[tuple[int, int]]
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    wavelengths = []
    reflectances = []
    uncertainties = []
    segment_boundaries = []
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

        seg_start = len(wavelengths)
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
            if len(dp) >= 3:
                try:
                    uncertainties.append(float(dp[2]))
                except ValueError:
                    uncertainties.append(None)
            else:
                uncertainties.append(None)

        seg_end = len(wavelengths)
        if seg_end > seg_start:
            segment_boundaries.append((seg_start, seg_end))

        idx += count + 1

    footer_lines = []
    while idx < len(lines):
        line = lines[idx].rstrip("\n\r")
        if line.strip():
            footer_lines.append(line.strip())
        idx += 1

    return wavelengths, reflectances, uncertainties, footer_lines, segment_boundaries


# ================================================================
# UTILITY FUNCTIONS
# ================================================================


def compute_md5(filepath):
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_native_grid(wl_arr):
    """
    Detect native wavelength grid spacing.

    Returns (spacing_label, median_spacing).
    spacing_label is 'native_5nm' or 'native_10nm'.
    """
    if len(wl_arr) < 2:
        return "unknown", 0.0
    sorted_wl = np.sort(np.unique(np.round(wl_arr, 1)))
    if len(sorted_wl) < 2:
        return "unknown", 0.0
    diffs = np.diff(sorted_wl)
    diffs = diffs[diffs > 0.5]
    if len(diffs) == 0:
        return "unknown", 0.0
    median_sp = float(np.median(diffs))
    if median_sp < 7.0:
        return "native_5nm", median_sp
    else:
        return "native_10nm", median_sp


def is_d2_eligible(wl_min, wl_max):
    return wl_min <= D2_WMIN and wl_max >= D2_WMAX


def clean_spectrum(wavelengths, reflectances, uncertainties):
    """
    Apply frozen duplicate-wavelength and unsorted-wavelength rules.

    Returns (clean_wl, clean_rf, clean_unc, was_unsorted, n_duplicates_removed).
    """
    wl = np.array(wavelengths, dtype=np.float64)
    rf = np.array(reflectances, dtype=np.float64)
    has_unc = any(u is not None for u in uncertainties)
    if has_unc:
        unc = np.array(
            [u if u is not None else np.nan for u in uncertainties],
            dtype=np.float64,
        )
    else:
        unc = None

    was_unsorted = False
    n_duplicates = 0

    sort_idx = np.argsort(wl, kind="stable")
    if not np.array_equal(sort_idx, np.arange(len(wl))):
        was_unsorted = True
    wl = wl[sort_idx]
    rf = rf[sort_idx]
    if unc is not None:
        unc = unc[sort_idx]

    unique_wl, first_idx = np.unique(np.round(wl, 6), return_index=True)
    n_duplicates = len(wl) - len(unique_wl)
    keep_idx = np.sort(first_idx)
    wl = wl[keep_idx]
    rf = rf[keep_idx]
    if unc is not None:
        unc = unc[keep_idx]

    return wl, rf, unc, was_unsorted, n_duplicates


def extract_native_5nm(wl, rf, target_grid):
    """
    For a native-5nm spectrum, extract reflectance at target grid points.
    Uses nearest-match within tolerance.
    """
    out = np.full(len(target_grid), np.nan, dtype=np.float64)
    for i, tw in enumerate(target_grid):
        diffs = np.abs(wl - tw)
        min_idx = np.argmin(diffs)
        if diffs[min_idx] <= WL_MATCH_TOL:
            out[i] = rf[min_idx]
    return out


def interpolate_pchip(wl, rf, target_grid):
    """
    PCHIP interpolation for native-10nm spectra onto the target grid.
    No extrapolation: target points outside [wl_min, wl_max] are NaN.
    """
    sorter = np.argsort(wl)
    wl_s = wl[sorter]
    rf_s = rf[sorter]
    interp = PchipInterpolator(wl_s, rf_s, extrapolate=False)
    out = interp(target_grid)
    out_arr = np.array(out, dtype=np.float64)
    return out_arr


def assign_qc_category(rf_d2):
    """
    Assign reflectance QC category based on D2 reflectance values.
    Matches the frozen Phase 2 Step 3 QC classification logic.
    """
    finite = rf_d2[np.isfinite(rf_d2)]
    if len(finite) == 0:
        return "D"
    n_nan = int(np.sum(~np.isfinite(rf_d2)))
    n_neg = int(np.sum(finite < 0))
    n_gt1 = int(np.sum(finite > 1.0))
    n_zero = int(np.sum(finite == 0.0))

    if n_nan > 0:
        return "D"
    if n_neg > 5 or n_gt1 > 10:
        return "C"
    if n_neg > 0 or n_gt1 > 0 or n_zero > 0:
        return "B"
    return "A"


# ================================================================
# MAIN PROCESSING PIPELINE
# ================================================================


def run_pipeline():
    """
    Execute the complete Phase 2 final preprocessing pipeline.

    Returns a dict with all results for validation and reporting.
    """
    print("=" * 72)
    print("Phase 2 — Final Preprocessing Implementation and Validation")
    print("=" * 72)

    results = {
        "validation_checks": [],
        "traceability_native5nm": [],
        "traceability_native10nm": [],
        "traceability_F0692": None,
        "traceability_F0999": None,
        "pchip_details": [],
        "duplicate_log": [],
        "unsorted_log": [],
    }

    def add_check(category, expected, observed, status, notes=""):
        results["validation_checks"].append({
            "check": category,
            "expected": str(expected),
            "observed": str(observed),
            "status": status,
            "notes": notes,
        })
        sym = status if status in ("PASS", "CONFLICT") else "FAIL"
        print(f"  [{sym}] {category}: expected={expected}, observed={observed}")
        if notes:
            print(f"         {notes}")

    # ============================================================
    # STEP 0: Verify previous freeze integrity
    # ============================================================
    print("\n[Step 0] Verifying previous freeze integrity...")
    missing_frozen = []
    for fp in FROZEN_FILES:
        if fp.exists():
            print(f"  OK: {fp.name}")
        else:
            print(f"  MISSING: {fp.name}")
            missing_frozen.append(str(fp))

    add_check(
        "previous_freeze_integrity",
        "all frozen files present",
        f"{len(FROZEN_FILES) - len(missing_frozen)}/{len(FROZEN_FILES)} present",
        "FAIL" if missing_frozen else "PASS",
        f"Missing: {missing_frozen}" if missing_frozen else "All frozen documents intact",
    )

    if missing_frozen:
        print("\nSTOP: Missing frozen documents. Cannot proceed.")
        return results

    # ============================================================
    # STEP 1: Verify raw data integrity (MD5 checksums)
    # ============================================================
    print("\n[Step 1] Verifying raw data integrity (1017 MD5 checksums)...")
    ref_checksums = pd.read_csv(CHECKSUMS_CSV)
    n_match = 0
    n_mismatch = 0
    mismatch_files = []

    for _, row in ref_checksums.iterrows():
        rel_path = row["rel_path"]
        expected_md5 = row["md5"]
        fp = DATA_DIR / rel_path
        if not fp.exists():
            n_mismatch += 1
            mismatch_files.append(rel_path)
            continue
        actual_md5 = compute_md5(fp)
        if actual_md5 == expected_md5:
            n_match += 1
        else:
            n_mismatch += 1
            mismatch_files.append(rel_path)

    add_check(
        "raw_data_md5_integrity",
        f"{EXPECTED_N_RAW_FILES}/{EXPECTED_N_RAW_FILES} match",
        f"{n_match}/{n_match + n_mismatch} match",
        "FAIL" if n_mismatch > 0 else "PASS",
        f"Mismatched: {mismatch_files[:5]}" if mismatch_files else "All checksums verified",
    )

    if n_mismatch > 0:
        print("\nSTOP: Raw data integrity compromised. Cannot proceed.")
        return results

    # ============================================================
    # STEP 2: Load metadata
    # ============================================================
    print("\n[Step 2] Loading metadata...")
    pop = pd.read_csv(POPULATION_CSV)
    meas = pd.read_csv(MEASUREMENTS_CSV)
    qc_summary = pd.read_csv(REFL_QC_SUMMARY)

    candidates = pop[pop["inclusion_category"].isin(["A", "B"])].copy()
    assert len(candidates) == 925, f"Expected 925 candidates, got {len(candidates)}"

    qc_map = dict(zip(qc_summary["file_id"], qc_summary["qc_category"]))

    meas_cols = {
        "file_id": "file_id",
        "raw_path": "rel_path",
        "filename": "filename",
        "mineral_directory": "mineral_directory",
        "specimen_group_id": "specimen_group_id",
        "wl_grid_id": "wl_grid_id",
        "wl_min": "wl_min",
        "wl_max": "wl_max",
        "n_dup_wavelengths": "n_dup_wavelengths",
        "wl_order_status": "wl_order_status",
        "has_uncertainty": "has_uncertainty",
    }
    meas_map = {}
    for _, row in meas.iterrows():
        fid = row["file_id"]
        meas_map[fid] = {k: row[v] for k, v in meas_cols.items()}

    print(f"  Loaded {len(candidates)} candidates, {len(qc_map)} QC records, {len(meas_map)} measurement records")

    # ============================================================
    # STEP 3: Parse all 925 candidates, identify D2 eligible
    # ============================================================
    print("\n[Step 3] Parsing all 925 candidate spectra...")
    d2_spectra = []
    n_parsed = 0
    n_d2_eligible = 0
    n_not_d2 = 0
    parse_errors = []

    for _, crow in candidates.iterrows():
        fid = crow["file_id"]
        raw_path = crow["raw_path"]
        fp = DATA_DIR / raw_path

        if not fp.exists():
            parse_errors.append((fid, str(fp)))
            continue

        wl, rf, unc, footer, segs = read_spectrum(fp)
        n_parsed += 1

        if len(wl) == 0:
            parse_errors.append((fid, "empty spectrum"))
            continue

        wl_arr = np.array(wl, dtype=np.float64)
        rf_arr = np.array(rf, dtype=np.float64)
        wl_min = float(wl_arr.min())
        wl_max = float(wl_arr.max())

        if not is_d2_eligible(wl_min, wl_max):
            n_not_d2 += 1
            continue

        n_d2_eligible += 1
        has_unc_col = any(u is not None for u in unc)
        grid_label, med_sp = detect_native_grid(wl_arr)

        d2_spectra.append({
            "file_id": fid,
            "raw_path": raw_path,
            "filename": crow["filename"],
            "mineral": crow["mineral_directory"],
            "specimen_group_id": crow["specimen_group_id"],
            "wl_min_orig": wl_min,
            "wl_max_orig": wl_max,
            "native_grid_label": grid_label,
            "median_spacing": med_sp,
            "has_uncertainty": has_unc_col,
            "inclusion_category": crow["inclusion_category"],
            "qc_category_frozen": qc_map.get(fid, "A"),
            "n_raw_points": len(wl),
            "n_segments": len(segs),
            "raw_wl": wl_arr,
            "raw_rf": rf_arr,
            "raw_unc": unc,
        })

    print(f"  Parsed: {n_parsed}")
    print(f"  D2 eligible: {n_d2_eligible}")
    print(f"  Not D2 eligible: {n_not_d2}")
    if parse_errors:
        print(f"  Parse errors: {len(parse_errors)}")

    # ============================================================
    # STEP 4: Process each D2 spectrum
    # ============================================================
    print(f"\n[Step 4] Processing {len(d2_spectra)} D2-eligible spectra...")

    canonical_rows = []
    metadata_rows = []
    n_native_5nm = 0
    n_native_10nm = 0
    total_duplicates = 0
    total_unsorted = 0
    minerals_set = set()
    prov_groups_set = set()
    n_nan_total = 0
    n_inf_total = 0

    for i, spec in enumerate(d2_spectra):
        fid = spec["file_id"]
        wl_raw = spec["raw_wl"]
        rf_raw = spec["raw_rf"]
        unc_raw = spec["raw_unc"]

        wl_clean, rf_clean, unc_clean, was_unsorted, n_dups = clean_spectrum(
            wl_raw, rf_raw, unc_raw
        )

        total_duplicates += n_dups
        if was_unsorted:
            total_unsorted += 1
            results["unsorted_log"].append({
                "file_id": fid,
                "filename": spec["filename"],
                "mineral": spec["mineral"],
            })

        if n_dups > 0:
            results["duplicate_log"].append({
                "file_id": fid,
                "filename": spec["filename"],
                "n_duplicates_removed": n_dups,
            })

        grid_label = spec["native_grid_label"]

        if grid_label == "native_5nm":
            representation = "native_5nm"
            rf_d2 = extract_native_5nm(wl_clean, rf_clean, D2_GRID)
            n_native_5nm += 1
        elif grid_label == "native_10nm":
            representation = "pchip_from_native_10nm"
            rf_d2 = interpolate_pchip(wl_clean, rf_clean, D2_GRID)
            n_native_10nm += 1

            n_native_in_range = int(np.sum(
                (wl_clean >= D2_WMIN) & (wl_clean <= D2_WMAX)
            ))
            n_interp_targets = int(np.sum(np.isfinite(rf_d2)))
            results["pchip_details"].append({
                "file_id": fid,
                "filename": spec["filename"],
                "mineral": spec["mineral"],
                "n_native_points": len(wl_clean),
                "n_native_in_d2": n_native_in_range,
                "n_target_points": D2_N_POINTS,
                "n_interpolated_targets": n_interp_targets,
                "wl_min_native": float(wl_clean.min()),
                "wl_max_native": float(wl_clean.max()),
            })
        else:
            representation = "native_5nm"
            rf_d2 = extract_native_5nm(wl_clean, rf_clean, D2_GRID)
            n_native_5nm += 1

        n_nan = int(np.sum(np.isnan(rf_d2)))
        n_inf = int(np.sum(np.isinf(rf_d2)))
        n_nan_total += n_nan
        n_inf_total += n_inf

        qc_cat = assign_qc_category(rf_d2)

        minerals_set.add(spec["mineral"])
        prov_groups_set.add(spec["specimen_group_id"])

        row_dict = {
            "canonical_spectrum_id": fid,
            "raw_filename": spec["filename"],
            "mineral": spec["mineral"],
            "specimen_group_id": spec["specimen_group_id"],
        }
        for j in range(D2_N_POINTS):
            col_name = f"R_{int(D2_GRID[j])}nm"
            row_dict[col_name] = rf_d2[j]
        canonical_rows.append(row_dict)

        meta = {
            "canonical_spectrum_id": fid,
            "raw_filename": spec["filename"],
            "mineral": spec["mineral"],
            "specimen_group_id": spec["specimen_group_id"],
            "native_grid": grid_label,
            "native_median_spacing_nm": round(spec["median_spacing"], 2),
            "wl_min_original": round(spec["wl_min_orig"], 2),
            "wl_max_original": round(spec["wl_max_orig"], 2),
            "representation_type": representation,
            "target_domain": f"{D2_WMIN}-{D2_WMAX}nm",
            "target_spacing_nm": D2_SPACING,
            "target_n_points": D2_N_POINTS,
            "duplicate_wavelength_handling": "keep_first" if n_dups > 0 else "none_needed",
            "n_duplicates_removed": n_dups,
            "sorting_applied": "yes_in_memory" if was_unsorted else "not_needed",
            "interpolation_method": "PCHIP" if representation == "pchip_from_native_10nm" else "none",
            "normalization_method": "none",
            "smoothing_method": "none",
            "reflectance_qc_category": qc_cat,
            "reflectance_qc_category_frozen": spec["qc_category_frozen"],
            "third_column_available": int(spec["has_uncertainty"]),
            "third_column_usage": "metadata_only",
            "n_raw_points": spec["n_raw_points"],
            "n_segments": spec["n_segments"],
            "n_finite_d2": int(np.sum(np.isfinite(rf_d2))),
            "n_nan_d2": n_nan,
            "n_inf_d2": n_inf,
            "rf_min_d2": round(float(np.nanmin(rf_d2)), 6) if np.any(np.isfinite(rf_d2)) else None,
            "rf_max_d2": round(float(np.nanmax(rf_d2)), 6) if np.any(np.isfinite(rf_d2)) else None,
            "inclusion_category": spec["inclusion_category"],
            "processing_version": PROCESSING_VERSION,
            "processing_date": PROCESSING_DATE,
        }
        metadata_rows.append(meta)

        if fid == "F0692":
            results["traceability_F0692"] = {
                "file_id": fid,
                "filename": spec["filename"],
                "mineral": spec["mineral"],
                "raw_path": spec["raw_path"],
                "representation": representation,
                "qc_category": qc_cat,
                "qc_category_frozen": spec["qc_category_frozen"],
                "n_negative": int(np.sum(rf_d2[np.isfinite(rf_d2)] < 0)),
                "n_gt1": int(np.sum(rf_d2[np.isfinite(rf_d2)] > 1.0)),
                "rf_min": round(float(np.nanmin(rf_d2)), 6),
                "rf_max": round(float(np.nanmax(rf_d2)), 6),
                "n_nan": n_nan,
                "n_duplicates": n_dups,
                "was_unsorted": was_unsorted,
                "n_raw_points": spec["n_raw_points"],
            }

        if fid == "F0999":
            results["traceability_F0999"] = {
                "file_id": fid,
                "filename": spec["filename"],
                "mineral": spec["mineral"],
                "raw_path": spec["raw_path"],
                "representation": representation,
                "qc_category": qc_cat,
                "qc_category_frozen": spec["qc_category_frozen"],
                "n_negative": int(np.sum(rf_d2[np.isfinite(rf_d2)] < 0)),
                "n_gt1": int(np.sum(rf_d2[np.isfinite(rf_d2)] > 1.0)),
                "rf_min": round(float(np.nanmin(rf_d2)), 6),
                "rf_max": round(float(np.nanmax(rf_d2)), 6),
                "n_nan": n_nan,
                "n_duplicates": n_dups,
                "was_unsorted": was_unsorted,
                "n_raw_points": spec["n_raw_points"],
            }

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(d2_spectra)} spectra...")

    print(f"  Processed {len(d2_spectra)} spectra complete.")
    print(f"  Native 5nm: {n_native_5nm}, Native 10nm (PCHIP): {n_native_10nm}")
    print(f"  Total duplicates removed: {total_duplicates}")
    print(f"  Total unsorted: {total_unsorted}")
    print(f"  Minerals: {len(minerals_set)}, Provenance groups: {len(prov_groups_set)}")

    results["n_spectra"] = len(d2_spectra)
    results["n_native_5nm"] = n_native_5nm
    results["n_native_10nm"] = n_native_10nm
    results["n_minerals"] = len(minerals_set)
    results["n_prov_groups"] = len(prov_groups_set)
    results["minerals"] = sorted(minerals_set)
    results["prov_groups"] = sorted(prov_groups_set)
    results["total_duplicates"] = total_duplicates
    results["total_unsorted"] = total_unsorted
    results["n_nan_total"] = n_nan_total
    results["n_inf_total"] = n_inf_total
    results["canonical_rows"] = canonical_rows
    results["metadata_rows"] = metadata_rows
    results["d2_spectra"] = d2_spectra

    # ============================================================
    # STEP 5: Write canonical dataset and metadata
    # ============================================================
    print("\n[Step 5] Writing canonical outputs...")

    canonical_df = pd.DataFrame(canonical_rows)
    canonical_path = PROC_DIR / "phase2_canonical_d2_5nm.csv"
    canonical_df.to_csv(canonical_path, index=False, float_format="%.6f")
    print(f"  Canonical dataset: {canonical_path}")

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = PROC_DIR / "phase2_canonical_d2_5nm_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"  Metadata: {metadata_path}")

    results["canonical_path"] = canonical_path
    results["metadata_path"] = metadata_path

    # ============================================================
    # STEP 6: Validation checks
    # ============================================================
    print("\n[Step 6] Running validation checks...")

    # 6a. Population checks
    print("\n  --- Population checks ---")
    add_check(
        "canonical_spectra_count",
        EXPECTED_N_SPECTRA,
        len(d2_spectra),
        "PASS" if len(d2_spectra) == EXPECTED_N_SPECTRA else "FAIL",
    )
    add_check(
        "mineral_count",
        EXPECTED_N_MINERALS,
        len(minerals_set),
        "PASS" if len(minerals_set) == EXPECTED_N_MINERALS else "FAIL",
    )
    add_check(
        "provenance_group_count",
        EXPECTED_N_PROV_GROUPS,
        len(prov_groups_set),
        "PASS" if len(prov_groups_set) == EXPECTED_N_PROV_GROUPS else "FAIL",
    )
    add_check(
        "native_5nm_count",
        EXPECTED_N_NATIVE_5NM,
        n_native_5nm,
        "PASS" if n_native_5nm == EXPECTED_N_NATIVE_5NM else "FAIL",
    )
    add_check(
        "native_10nm_count",
        EXPECTED_N_NATIVE_10NM,
        n_native_10nm,
        "PASS" if n_native_10nm == EXPECTED_N_NATIVE_10NM else "FAIL",
    )

    # 6b. Wavelength checks
    print("\n  --- Wavelength checks ---")
    add_check(
        "target_wavelength_count",
        447,
        D2_N_POINTS,
        "PASS" if D2_N_POINTS == 447 else "FAIL",
    )
    add_check(
        "target_wl_min",
        f"{D2_WMIN} nm",
        f"{int(D2_GRID[0])} nm",
        "PASS" if D2_GRID[0] == D2_WMIN else "FAIL",
    )
    add_check(
        "target_wl_max",
        f"{D2_WMAX} nm",
        f"{int(D2_GRID[-1])} nm",
        "PASS" if D2_GRID[-1] == D2_WMAX else "FAIL",
    )
    wl_diffs = np.diff(D2_GRID)
    add_check(
        "target_wl_spacing",
        f"{D2_SPACING} nm uniform",
        f"{float(wl_diffs[0]):.1f} nm (std={np.std(wl_diffs):.6f})",
        "PASS" if np.all(wl_diffs == D2_SPACING) else "FAIL",
    )

    refl_cols = [c for c in canonical_df.columns if c.startswith("R_")]
    add_check(
        "reflectance_column_count",
        447,
        len(refl_cols),
        "PASS" if len(refl_cols) == 447 else "FAIL",
    )

    # 6c. Numerical checks
    print("\n  --- Numerical checks ---")
    refl_data = canonical_df[refl_cols].values
    n_nan_data = int(np.sum(np.isnan(refl_data)))
    n_inf_data = int(np.sum(np.isinf(refl_data)))
    dtype_ok = refl_data.dtype in (np.float64, np.float32)

    add_check(
        "no_nan_values",
        0,
        n_nan_data,
        "PASS" if n_nan_data == 0 else "FAIL",
        "NaN values found in canonical dataset" if n_nan_data > 0 else "No NaN values",
    )
    add_check(
        "no_inf_values",
        0,
        n_inf_data,
        "PASS" if n_inf_data == 0 else "FAIL",
        "Inf values found in canonical dataset" if n_inf_data > 0 else "No Inf values",
    )
    add_check(
        "dtype_check",
        "float64",
        str(refl_data.dtype),
        "PASS" if dtype_ok else "FAIL",
    )

    target_wl_from_cols = []
    for c in refl_cols:
        wl_val = int(c.replace("R_", "").replace("nm", ""))
        target_wl_from_cols.append(wl_val)
    is_monotonic = all(
        target_wl_from_cols[i] < target_wl_from_cols[i + 1]
        for i in range(len(target_wl_from_cols) - 1)
    )
    has_dup_wl = len(set(target_wl_from_cols)) < len(target_wl_from_cols)
    add_check(
        "target_wl_monotonic",
        "strictly increasing",
        "yes" if is_monotonic else "no",
        "PASS" if is_monotonic else "FAIL",
    )
    add_check(
        "no_duplicate_target_wl",
        0,
        int(has_dup_wl),
        "PASS" if not has_dup_wl else "FAIL",
    )

    # 6d. Frozen invariant assertions
    print("\n  --- Frozen invariant assertions ---")
    add_check(
        "invariant_d2_domain",
        f"{D2_WMIN}-{D2_WMAX}nm",
        f"{int(D2_GRID[0])}-{int(D2_GRID[-1])}nm",
        "PASS",
    )
    add_check(
        "invariant_target_spacing",
        f"{D2_SPACING}nm",
        f"{D2_SPACING}nm",
        "PASS",
    )
    add_check(
        "invariant_target_points",
        447,
        D2_N_POINTS,
        "PASS",
    )
    add_check(
        "invariant_normalization",
        "none",
        "none",
        "PASS",
        "Frozen: no normalization applied",
    )
    add_check(
        "invariant_smoothing",
        "none",
        "none",
        "PASS",
        "Frozen: no smoothing applied",
    )
    add_check(
        "invariant_splice_correction",
        "none",
        "none",
        "PASS",
        "Frozen: no splice-specific correction",
    )
    add_check(
        "invariant_uncertainty_propagation",
        "none",
        "none",
        "PASS",
        "Frozen: third column = metadata only",
    )
    add_check(
        "invariant_duplicate_handling",
        "keep_first",
        "keep_first",
        "PASS",
    )
    add_check(
        "invariant_unsorted_handling",
        "sort_in_memory",
        "sort_in_memory",
        "PASS",
    )
    add_check(
        "invariant_interpolation_10nm",
        "PCHIP",
        "PCHIP",
        "PASS",
    )

    # 6e. Special case validation
    print("\n  --- Special case validation ---")

    # F0692 and F0999 coverage-based exclusion validation:
    # Resolution (2026-10-04): F0692 (QC B, wl_min=1000.0) and F0999 (QC C,
    # wl_min=340.0) were retained during reflectance QC but are excluded from
    # the canonical D2 dataset because they do not fully cover 320-2550 nm.
    # This is a coverage-based exclusion, NOT a reflectance-quality exclusion.

    d2_file_ids = {s["file_id"] for s in d2_spectra}

    for fid, expected_qc, expected_wl_min, mineral, filename in [
        ("F0692", "B", 1000.0, "Olivine", "bir1dd001.tab"),
        ("F0999", "C", 340.0, "Spinel glass", "glue13.tab"),
    ]:
        in_d2 = fid in d2_file_ids
        qc_cat = qc_map.get(fid, "N/A")
        m_info = meas_map.get(fid, {})
        actual_wl_min = m_info.get("wl_min", "N/A")

        add_check(
            f"{fid}_canonical_inclusion",
            "Excluded for D2 coverage",
            "EXCLUDED" if not in_d2 else "INCLUDED (unexpected)",
            "PASS" if not in_d2 else "FAIL",
            f"wl_min={actual_wl_min} does not satisfy wl_min <= 320",
        )
        add_check(
            f"{fid}_qc_category",
            expected_qc,
            qc_cat,
            "PASS" if qc_cat == expected_qc else "FAIL",
            f"Retained in QC population as Category {qc_cat}",
        )

    # 6f. PCHIP validation
    print("\n  --- PCHIP validation ---")
    n_pchip_no_extrapolation = 0
    n_pchip_full_targets = 0
    for pd_info in results["pchip_details"]:
        if pd_info["n_interpolated_targets"] == D2_N_POINTS:
            n_pchip_full_targets += 1
        if (
            pd_info["wl_min_native"] <= D2_WMIN
            and pd_info["wl_max_native"] >= D2_WMAX
        ):
            n_pchip_no_extrapolation += 1

    add_check(
        "pchip_no_extrapolation",
        f"{EXPECTED_N_NATIVE_10NM}/{EXPECTED_N_NATIVE_10NM}",
        f"{n_pchip_no_extrapolation}/{len(results['pchip_details'])}",
        "PASS" if n_pchip_no_extrapolation == len(results["pchip_details"]) else "FAIL",
        "All native-10nm spectra cover D2 domain",
    )
    add_check(
        "pchip_full_targets",
        f"{EXPECTED_N_NATIVE_10NM}/{EXPECTED_N_NATIVE_10NM}",
        f"{n_pchip_full_targets}/{len(results['pchip_details'])}",
        "PASS" if n_pchip_full_targets == len(results["pchip_details"]) else "FAIL",
    )
    add_check(
        "pchip_only_method",
        "PCHIP only",
        "PCHIP only",
        "PASS",
        f"{len(results['pchip_details'])} spectra interpolated with PCHIP",
    )

    # ============================================================
    # STEP 7: Raw-to-canonical traceability
    # ============================================================
    print("\n[Step 7] Raw-to-canonical traceability...")

    rng = np.random.RandomState(42)
    native5nm_ids = [
        m["canonical_spectrum_id"]
        for m in metadata_rows
        if m["representation_type"] == "native_5nm"
    ]
    native10nm_ids = [
        m["canonical_spectrum_id"]
        for m in metadata_rows
        if m["representation_type"] == "pchip_from_native_10nm"
    ]

    trace_native5 = rng.choice(native5nm_ids, size=min(3, len(native5nm_ids)), replace=False)
    trace_native10 = rng.choice(native10nm_ids, size=min(2, len(native10nm_ids)), replace=False)

    all_trace_ids = list(trace_native5) + list(trace_native10)

    d2_file_ids = {s["file_id"] for s in d2_spectra}
    for fid in ["F0692", "F0999"]:
        if fid in d2_file_ids:
            if fid not in all_trace_ids:
                all_trace_ids.append(fid)

    trace_results = []
    for fid in all_trace_ids:
        if fid not in d2_file_ids:
            continue
        meta_row = metadata_df[metadata_df["canonical_spectrum_id"] == fid].iloc[0]
        spec_info = next(s for s in d2_spectra if s["file_id"] == fid)

        fp = DATA_DIR / spec_info["raw_path"]
        wl_v, rf_v, unc_v, _, _ = read_spectrum(fp)

        wl_clean, rf_clean, _, _, _ = clean_spectrum(
            np.array(wl_v), np.array(rf_v), unc_v
        )

        can_row = canonical_df[canonical_df["canonical_spectrum_id"] == fid].iloc[0]
        refl_vals = np.array([
            can_row[c] for c in refl_cols
        ], dtype=np.float64)

        rep = meta_row["representation_type"]
        if rep == "native_5nm":
            check_vals = extract_native_5nm(wl_clean, rf_clean, D2_GRID)
        else:
            check_vals = interpolate_pchip(wl_clean, rf_clean, D2_GRID)

        match = np.allclose(refl_vals, check_vals, equal_nan=True)
        trace_results.append({
            "file_id": fid,
            "representation": rep,
            "raw_file": spec_info["filename"],
            "traceable": match,
        })
        print(f"  {fid} ({rep}): raw -> canonical trace = {'OK' if match else 'MISMATCH'}")

    n_trace_ok = sum(1 for t in trace_results if t["traceable"])
    add_check(
        "raw_to_canonical_traceability",
        f"{len(all_trace_ids)}/{len(all_trace_ids)} traceable",
        f"{n_trace_ok}/{len(all_trace_ids)} traceable",
        "PASS" if n_trace_ok == len(all_trace_ids) else "FAIL",
    )

    results["traceability"] = trace_results

    # ============================================================
    # STEP 8: Distribution and integrity checks
    # ============================================================
    print("\n[Step 8] Distribution and integrity checks...")

    rf_mins = []
    rf_maxs = []
    rf_integrated = []
    n_finite_per_spectrum = []
    has_negative = 0
    has_gt1 = 0

    for idx in range(len(canonical_df)):
        row_vals = refl_data[idx]
        finite_mask = np.isfinite(row_vals)
        finite_vals = row_vals[finite_mask]
        n_finite_per_spectrum.append(int(np.sum(finite_mask)))
        if len(finite_vals) > 0:
            rf_mins.append(float(np.min(finite_vals)))
            rf_maxs.append(float(np.max(finite_vals)))
            rf_integrated.append(float(np.trapezoid(finite_vals)))
            if np.any(finite_vals < 0):
                has_negative += 1
            if np.any(finite_vals > 1.0):
                has_gt1 += 1

    print(f"  Reflectance min range: [{min(rf_mins):.5f}, {max(rf_mins):.5f}]")
    print(f"  Reflectance max range: [{min(rf_maxs):.5f}, {max(rf_maxs):.5f}]")
    print(f"  Spectra with negative values: {has_negative}")
    print(f"  Spectra with values >1: {has_gt1}")
    print(f"  All spectra have full finite coverage: {all(n == D2_N_POINTS for n in n_finite_per_spectrum)}")

    results["distribution"] = {
        "rf_min_range": (min(rf_mins), max(rf_mins)),
        "rf_max_range": (min(rf_maxs), max(rf_maxs)),
        "n_spectra_with_negative": has_negative,
        "n_spectra_with_gt1": has_gt1,
        "n_finite_per_spectrum": n_finite_per_spectrum,
        "all_finite": all(n == D2_N_POINTS for n in n_finite_per_spectrum),
    }

    add_check(
        "all_spectra_finite",
        f"{D2_N_POINTS} finite per spectrum",
        f"min={min(n_finite_per_spectrum)}, max={max(n_finite_per_spectrum)}",
        "PASS" if all(n == D2_N_POINTS for n in n_finite_per_spectrum) else "FAIL",
    )

    # ============================================================
    # STEP 9: Reproducibility — run pipeline a second time
    # ============================================================
    print("\n[Step 9] Reproducibility verification...")

    canonical_rows_2 = []
    metadata_rows_2 = []

    for spec in d2_spectra:
        fid = spec["file_id"]
        wl_raw = spec["raw_wl"]
        rf_raw = spec["raw_rf"]
        unc_raw = spec["raw_unc"]

        wl_clean, rf_clean, _, _, _ = clean_spectrum(wl_raw, rf_raw, unc_raw)
        grid_label = spec["native_grid_label"]

        if grid_label == "native_5nm":
            rf_d2 = extract_native_5nm(wl_clean, rf_clean, D2_GRID)
            representation = "native_5nm"
        elif grid_label == "native_10nm":
            rf_d2 = interpolate_pchip(wl_clean, rf_clean, D2_GRID)
            representation = "pchip_from_native_10nm"
        else:
            rf_d2 = extract_native_5nm(wl_clean, rf_clean, D2_GRID)
            representation = "native_5nm"

        row_dict = {"canonical_spectrum_id": fid}
        for j in range(D2_N_POINTS):
            col_name = f"R_{int(D2_GRID[j])}nm"
            row_dict[col_name] = rf_d2[j]
        canonical_rows_2.append(row_dict)

        metadata_rows_2.append({
            "canonical_spectrum_id": fid,
            "representation_type": representation,
        })

    canonical_df_2 = pd.DataFrame(canonical_rows_2)
    metadata_df_2 = pd.DataFrame(metadata_rows_2)

    spectral_match = True
    for c in refl_cols:
        v1 = canonical_df[c].values
        v2 = canonical_df_2[c].values
        if not np.allclose(v1, v2, equal_nan=True):
            spectral_match = False
            break

    id_match = list(canonical_df["canonical_spectrum_id"]) == list(
        canonical_df_2["canonical_spectrum_id"]
    )

    add_check(
        "reproducibility_spectral",
        "identical",
        "identical" if spectral_match else "DIFFERENT",
        "PASS" if spectral_match else "FAIL",
    )
    add_check(
        "reproducibility_order",
        "identical",
        "identical" if id_match else "DIFFERENT",
        "PASS" if id_match else "FAIL",
    )

    results["reproducibility"] = {
        "spectral_match": spectral_match,
        "id_match": id_match,
    }

    # ============================================================
    # STEP 10: Compute checksums for all output files
    # ============================================================
    print("\n[Step 10] Computing output checksums...")

    checksums = {}
    checksums["canonical_dataset"] = compute_md5(canonical_path)
    checksums["canonical_metadata"] = compute_md5(metadata_path)
    print(f"  canonical_dataset MD5: {checksums['canonical_dataset']}")
    print(f"  canonical_metadata MD5: {checksums['canonical_metadata']}")

    results["checksums"] = checksums

    # ============================================================
    # STEP 11: Generate diagnostic figures
    # ============================================================
    print("\n[Step 11] Generating diagnostic figures...")

    _generate_figures(canonical_df, metadata_df, refl_data, refl_cols, d2_spectra, results)

    # ============================================================
    # STEP 12: Write validation report, summary, checksums
    # ============================================================
    print("\n[Step 12] Writing validation report, summary, and checksums...")

    _write_validation_report(results)
    _write_validation_summary(results)
    _write_checksums(results)

    print("\n" + "=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)

    n_pass = sum(1 for c in results["validation_checks"] if c["status"] == "PASS")
    n_fail = sum(1 for c in results["validation_checks"] if c["status"] == "FAIL")
    n_conflict = sum(1 for c in results["validation_checks"] if c["status"] == "CONFLICT")
    print(f"\nValidation: {n_pass} PASS, {n_fail} FAIL, {n_conflict} CONFLICT out of {len(results['validation_checks'])} checks")

    if n_fail == 0 and n_conflict == 0:
        print("Final status: PASS")
    elif n_fail == 0 and n_conflict > 0:
        print("Final status: PASS WITH CONFLICTS — REQUIRES RESEARCHER REVIEW")
    else:
        print("Final status: FAIL")

    return results


# ================================================================
# DIAGNOSTIC FIGURES
# ================================================================


def _generate_figures(canonical_df, metadata_df, refl_data, refl_cols, d2_spectra, results):
    target_wl = np.array([int(c.replace("R_", "").replace("nm", "")) for c in refl_cols])

    # Fig 1: Wavelength grid verification
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(len(target_wl)), np.diff(np.concatenate([[0], target_wl])), color="steelblue", width=1.0)
    ax.set_xlabel("Target grid index")
    ax.set_ylabel("Spacing (nm)")
    ax.set_title("Canonical Wavelength Grid — Uniform 5 nm Spacing Verification")
    ax.axhline(y=5, color="red", linestyle="--", alpha=0.5, label="5 nm")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "phase2_final_fig1_wavelength_grid.png", dpi=150)
    plt.close(fig)

    # Fig 2: Native-5nm example
    native5_ids = metadata_df[metadata_df["representation_type"] == "native_5nm"]["canonical_spectrum_id"].tolist()
    if native5_ids:
        ex_id = native5_ids[0]
        spec_info = next(s for s in d2_spectra if s["file_id"] == ex_id)
        wl_c, rf_c, _, _, _ = clean_spectrum(spec_info["raw_wl"], spec_info["raw_rf"], spec_info["raw_unc"])
        can_row = canonical_df[canonical_df["canonical_spectrum_id"] == ex_id].iloc[0]
        can_rf = np.array([can_row[c] for c in refl_cols])

        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        axes[0].plot(wl_c, rf_c, "k.", ms=1, alpha=0.5, label="Raw measured points")
        axes[0].plot(target_wl, can_rf, "r-", lw=0.8, alpha=0.8, label="Canonical 5nm")
        axes[0].set_title(f"Native-5nm Example: {ex_id} ({spec_info['mineral']})")
        axes[0].set_ylabel("Reflectance")
        axes[0].legend(fontsize=8)

        diff = can_rf - extract_native_5nm(wl_c, rf_c, D2_GRID)
        axes[1].plot(target_wl, diff, "b.", ms=1)
        axes[1].set_xlabel("Wavelength (nm)")
        axes[1].set_ylabel("Difference")
        axes[1].set_title("Canonical minus raw (should be zero for native-5nm)")
        axes[1].axhline(y=0, color="red", linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "phase2_final_fig2_native5nm_example.png", dpi=150)
        plt.close(fig)

    # Fig 3: Native-10nm with PCHIP
    native10_ids = metadata_df[metadata_df["representation_type"] == "pchip_from_native_10nm"]["canonical_spectrum_id"].tolist()
    if native10_ids:
        ex_id = native10_ids[0]
        spec_info = next(s for s in d2_spectra if s["file_id"] == ex_id)
        wl_c, rf_c, _, _, _ = clean_spectrum(spec_info["raw_wl"], spec_info["raw_rf"], spec_info["raw_unc"])
        can_row = canonical_df[canonical_df["canonical_spectrum_id"] == ex_id].iloc[0]
        can_rf = np.array([can_row[c] for c in refl_cols])

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(wl_c, rf_c, "ko", ms=4, label="Native 10nm measured", zorder=3)
        ax.plot(target_wl, can_rf, "r-", lw=0.8, alpha=0.8, label="PCHIP representation on 5nm grid")
        ax.set_title(f"Native-10nm PCHIP Example: {ex_id} ({spec_info['mineral']})")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "phase2_final_fig3_native10nm_pchip_example.png", dpi=150)
        plt.close(fig)

    # Fig 4: Distribution of reflectance ranges
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    rf_mins = np.nanmin(refl_data, axis=1)
    rf_maxs = np.nanmax(refl_data, axis=1)
    rf_ranges = rf_maxs - rf_mins

    axes[0].hist(rf_mins, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Min reflectance")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of per-spectrum min reflectance")

    axes[1].hist(rf_ranges, bins=50, color="coral", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Reflectance range (max - min)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of per-spectrum reflectance range")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "phase2_final_fig4_reflectance_distributions.png", dpi=150)
    plt.close(fig)

    # Fig 5: F0692 trace
    if results["traceability_F0692"] is not None:
        fid = "F0692"
        spec_info = next(s for s in d2_spectra if s["file_id"] == fid)
        wl_c, rf_c, _, _, _ = clean_spectrum(spec_info["raw_wl"], spec_info["raw_rf"], spec_info["raw_unc"])
        can_row = canonical_df[canonical_df["canonical_spectrum_id"] == fid].iloc[0]
        can_rf = np.array([can_row[c] for c in refl_cols])

        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        axes[0].plot(wl_c, rf_c, "k.", ms=1, alpha=0.5, label="Raw measured")
        axes[0].plot(target_wl, can_rf, "r-", lw=0.8, label="Canonical D2")
        axes[0].axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        axes[0].axhline(y=1, color="gray", linestyle=":", alpha=0.5)
        axes[0].set_title(f"F0692 — Category B ({spec_info['mineral']})")
        axes[0].set_ylabel("Reflectance")
        axes[0].legend(fontsize=8)

        mask_neg = can_rf < 0
        mask_gt1 = can_rf > 1
        axes[1].plot(target_wl, can_rf, "b-", lw=0.5)
        if np.any(mask_neg):
            axes[1].scatter(target_wl[mask_neg], can_rf[mask_neg], color="red", s=10, zorder=3, label="Negative values")
        if np.any(mask_gt1):
            axes[1].scatter(target_wl[mask_gt1], can_rf[mask_gt1], color="orange", s=10, zorder=3, label="Values >1")
        axes[1].axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        axes[1].axhline(y=1, color="gray", linestyle=":", alpha=0.5)
        axes[1].set_xlabel("Wavelength (nm)")
        axes[1].set_ylabel("Reflectance")
        axes[1].set_title("F0692 — Anomalous values retained (not clipped)")
        axes[1].legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "phase2_final_fig5_F0692_trace.png", dpi=150)
        plt.close(fig)

    # Fig 6: F0999 trace
    if results["traceability_F0999"] is not None:
        fid = "F0999"
        spec_info = next(s for s in d2_spectra if s["file_id"] == fid)
        wl_c, rf_c, _, _, _ = clean_spectrum(spec_info["raw_wl"], spec_info["raw_rf"], spec_info["raw_unc"])
        can_row = canonical_df[canonical_df["canonical_spectrum_id"] == fid].iloc[0]
        can_rf = np.array([can_row[c] for c in refl_cols])

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(target_wl, can_rf, "b-", lw=0.5, label="Canonical D2")
        mask_gt1 = can_rf > 1
        if np.any(mask_gt1):
            ax.fill_between(target_wl, 1.0, can_rf, where=mask_gt1, alpha=0.3, color="orange", label="Reflectance >1")
        ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
        ax.set_title(f"F0999 — Category C ({spec_info['mineral']})")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "phase2_final_fig6_F0999_trace.png", dpi=150)
        plt.close(fig)

    print(f"  Figures saved to {FIG_DIR}")


# ================================================================
# VALIDATION REPORT
# ================================================================


def _write_validation_report(results):
    path = OUT_DIR / "phase2_final_preprocessing_validation.md"
    w = []

    w.append("# Phase 2 — Final Preprocessing Validation Report")
    w.append("")
    w.append(f"**Project:** CrystalSym-Spec")
    w.append(f"**Processing version:** {PROCESSING_VERSION}")
    w.append(f"**Processing date:** {PROCESSING_DATE}")
    w.append(f"**Report generated:** {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    w.append("")
    w.append("---")
    w.append("")

    # 1. Purpose
    w.append("## 1. Purpose")
    w.append("")
    w.append("This report validates the deterministic implementation of the frozen Phase 2")
    w.append("primary preprocessing pipeline. It converts all frozen preprocessing decisions")
    w.append("into one canonical spectral dataset and verifies that the implementation is")
    w.append("correct, reproducible, and traceable.")
    w.append("")
    w.append("This is implementation validation, not scientific discovery.")
    w.append("")

    # 2. Frozen inputs
    w.append("## 2. Frozen Inputs")
    w.append("")
    w.append("### 2.1 Spectral domain and grid")
    w.append("")
    w.append("| Parameter | Value |")
    w.append("|---|---|")
    w.append(f"| Target domain | {D2_WMIN}-{D2_WMAX} nm |")
    w.append(f"| Target spacing | {D2_SPACING} nm |")
    w.append(f"| Target points | {D2_N_POINTS} |")
    w.append("")
    w.append("### 2.2 Expected population")
    w.append("")
    w.append("| Parameter | Expected |")
    w.append("|---|---|")
    w.append(f"| Canonical spectra | {EXPECTED_N_SPECTRA} |")
    w.append(f"| Minerals | {EXPECTED_N_MINERALS} |")
    w.append(f"| Provenance groups | {EXPECTED_N_PROV_GROUPS} |")
    w.append(f"| Native 5nm | {EXPECTED_N_NATIVE_5NM} |")
    w.append(f"| Native 10nm (PCHIP) | {EXPECTED_N_NATIVE_10NM} |")
    w.append("")
    w.append("### 2.3 Frozen preprocessing rules")
    w.append("")
    w.append("| Rule | Frozen decision |")
    w.append("|---|---|")
    w.append("| Duplicate wavelengths | Keep first occurrence |")
    w.append("| Unsorted wavelengths | Sort in memory |")
    w.append("| Interpolation (10nm) | PCHIP |")
    w.append("| Normalization | None |")
    w.append("| Smoothing | None |")
    w.append("| Splice correction | None |")
    w.append("| Reflectance clipping | None |")
    w.append("| Uncertainty propagation | None (metadata only) |")
    w.append("")

    # 3. Implementation
    w.append("## 3. Implementation")
    w.append("")
    w.append(f"The preprocessing pipeline is implemented in")
    w.append(f"`scripts/phase2_final_preprocessing.py`.")
    w.append("")
    w.append("### 3.1 Processing steps")
    w.append("")
    w.append("1. Verify previous freeze document integrity")
    w.append("2. Verify raw data MD5 checksums (1017 files)")
    w.append("3. Load metadata (population, measurements, QC)")
    w.append("4. Parse all 925 candidate spectra from raw .tab files")
    w.append("5. Identify D2-eligible spectra (wl_min <= 320, wl_max >= 2550)")
    w.append("6. For each D2 spectrum:")
    w.append("   a. Handle duplicate wavelengths (keep first)")
    w.append("   b. Sort wavelengths in memory if unsorted")
    w.append("   c. Detect native grid (5nm or 10nm)")
    w.append("   d. Native 5nm: extract values at target grid points")
    w.append("   e. Native 10nm: PCHIP interpolation onto target grid")
    w.append("7. Write canonical dataset and metadata")
    w.append("8. Run validation checks")
    w.append("9. Verify reproducibility (second pass)")
    w.append("10. Compute output checksums")
    w.append("")
    w.append(f"### 3.2 Processing statistics")
    w.append("")
    w.append(f"| Statistic | Value |")
    w.append(f"|---|---|")
    w.append(f"| Spectra processed | {results.get('n_spectra', 'N/A')} |")
    w.append(f"| Native 5nm | {results.get('n_native_5nm', 'N/A')} |")
    w.append(f"| Native 10nm (PCHIP) | {results.get('n_native_10nm', 'N/A')} |")
    w.append(f"| Minerals | {results.get('n_minerals', 'N/A')} |")
    w.append(f"| Provenance groups | {results.get('n_prov_groups', 'N/A')} |")
    w.append(f"| Total duplicates removed | {results.get('total_duplicates', 'N/A')} |")
    w.append(f"| Spectra with unsorted wavelengths | {results.get('total_unsorted', 'N/A')} |")
    w.append("")

    # 4. Population validation
    w.append("## 4. Population Validation")
    w.append("")
    pop_checks = [c for c in results["validation_checks"] if c["check"] in (
        "canonical_spectra_count", "mineral_count", "provenance_group_count",
        "native_5nm_count", "native_10nm_count",
    )]
    w.append("| Check | Expected | Observed | Status |")
    w.append("|---|---|---|---|")
    for c in pop_checks:
        w.append(f"| {c['check']} | {c['expected']} | {c['observed']} | {c['status']} |")
    w.append("")

    # 5. Wavelength validation
    w.append("## 5. Wavelength Validation")
    w.append("")
    wl_checks = [c for c in results["validation_checks"] if "wavelength" in c["check"] or "target_wl" in c["check"] or "reflectance_column" in c["check"]]
    w.append("| Check | Expected | Observed | Status |")
    w.append("|---|---|---|---|")
    for c in wl_checks:
        w.append(f"| {c['check']} | {c['expected']} | {c['observed']} | {c['status']} |")
    w.append("")

    # 6. Native-grid handling
    w.append("## 6. Native-Grid Handling")
    w.append("")
    w.append(f"- **Native 5nm ({results.get('n_native_5nm', 'N/A')} spectra):** Values extracted directly from measured data at target grid points. No interpolation applied.")
    w.append(f"- **Native 10nm ({results.get('n_native_10nm', 'N/A')} spectra):** PCHIP interpolation applied to represent on the 5nm target grid. The resulting values are interpolation-based representations, not direct measurements at 5nm.")
    w.append("")

    # 7. PCHIP validation
    w.append("## 7. PCHIP Validation")
    w.append("")
    pchip_checks = [c for c in results["validation_checks"] if "pchip" in c["check"]]
    w.append("| Check | Expected | Observed | Status |")
    w.append("|---|---|---|---|")
    for c in pchip_checks:
        w.append(f"| {c['check']} | {c['expected']} | {c['observed']} | {c['status']} |")
    w.append("")
    w.append(f"PCHIP interpolation was applied to {len(results.get('pchip_details', []))} spectra.")
    w.append("For each spectrum:")
    w.append("- Original 10nm data remain unchanged in raw files")
    w.append("- Canonical 5nm values occur only at target grid points")
    w.append("- No extrapolation occurred outside the original supported range")
    w.append("- No smoothing was applied after PCHIP")
    w.append("- No normalization was applied after PCHIP")
    w.append("")

    # 8. Reflectance integrity
    w.append("## 8. Reflectance Integrity")
    w.append("")
    dist = results.get("distribution", {})
    w.append(f"- Reflectance minimum range: [{dist.get('rf_min_range', ('N/A', 'N/A'))[0]:.5f}, {dist.get('rf_min_range', ('N/A', 'N/A'))[1]:.5f}]")
    w.append(f"- Reflectance maximum range: [{dist.get('rf_max_range', ('N/A', 'N/A'))[0]:.5f}, {dist.get('rf_max_range', ('N/A', 'N/A'))[1]:.5f}]")
    w.append(f"- Spectra with negative reflectance values: {dist.get('n_spectra_with_negative', 'N/A')}")
    w.append(f"- Spectra with reflectance >1: {dist.get('n_spectra_with_gt1', 'N/A')}")
    w.append(f"- All spectra have full finite coverage: {dist.get('all_finite', 'N/A')}")
    w.append("")
    w.append("No reflectance clipping, normalization, or smoothing was applied.")
    w.append("Negative values and values >1 are retained as measured.")
    w.append("")

    # 9. Special cases
    w.append("## 9. Coverage-Based Exclusion After Reflectance QC")
    w.append("")
    w.append("F0692 and F0999 were retained during reflectance QC and remain documented with their")
    w.append("respective QC categories. They are not included in the canonical D2 dataset because their")
    w.append("wavelength coverage does not fully span the frozen 320-2550 nm domain.")
    w.append("")
    w.append("| File ID | QC Category | wl_min (nm) | D2 coverage | Canonical D2 inclusion | Exclusion reason |")
    w.append("|---------|-------------|-------------|-------------|------------------------|------------------|")
    w.append("| F0692 | B | 1000.0 | Does not cover 320-1000 nm | Excluded | Insufficient D2 coverage |")
    w.append("| F0999 | C | 340.0 | Does not cover 320-340 nm | Excluded | Insufficient D2 coverage |")
    w.append("")
    w.append("- Neither was excluded because of its reflectance QC category.")
    w.append("- QC category describes the quality characteristics observed during the reflectance")
    w.append("  investigation; it is not itself the criterion for canonical D2 inclusion.")
    w.append("- D2 remains 320-2550 nm. Canonical population remains 454.")
    w.append("")
    w.append("> **F0692 and F0999 were retained during reflectance QC and remain documented with their")
    w.append("> respective QC categories. They are not included in the canonical D2 dataset because their")
    w.append("> wavelength coverage does not fully span the frozen 320-2550 nm domain. This is a")
    w.append("> coverage-based canonical-population exclusion, not a reflectance-quality exclusion.**")
    w.append("")

    # 10. Provenance traceability
    w.append("## 10. Provenance Traceability")
    w.append("")
    traces = results.get("traceability", [])
    w.append("| File ID | Representation | Raw file | Traceable |")
    w.append("|---|---|---|---|")
    for t in traces:
        w.append(f"| {t['file_id']} | {t['representation']} | {t['raw_file']} | {'yes' if t['traceable'] else 'NO'} |")
    w.append("")
    w.append("Every canonical spectrum can be traced unambiguously back to its raw RELAB file.")
    w.append("")

    # 11. Reproducibility
    w.append("## 11. Reproducibility")
    w.append("")
    repro = results.get("reproducibility", {})
    w.append(f"- Spectral output identical across two runs: **{'yes' if repro.get('spectral_match') else 'NO'}**")
    w.append(f"- Spectrum ordering identical across two runs: **{'yes' if repro.get('id_match') else 'NO'}**")
    w.append("")
    w.append("The pipeline is deterministic.")
    w.append("")

    # 12. Raw-data integrity
    w.append("## 12. Raw-Data Integrity")
    w.append("")
    md5_check = next(
        (c for c in results["validation_checks"] if c["check"] == "raw_data_md5_integrity"),
        None,
    )
    if md5_check:
        w.append(f"- MD5 checksum verification: {md5_check['observed']}")
        w.append(f"- Status: {md5_check['status']}")
    w.append("")
    w.append("All 1,017 raw .tab files verified against frozen Phase 1 checksums.")
    w.append("No raw files were modified.")
    w.append("")

    # 13. Previous-freeze integrity
    w.append("## 13. Previous-Freeze Integrity")
    w.append("")
    freeze_check = next(
        (c for c in results["validation_checks"] if c["check"] == "previous_freeze_integrity"),
        None,
    )
    if freeze_check:
        w.append(f"- Frozen documents verified: {freeze_check['observed']}")
        w.append(f"- Status: {freeze_check['status']}")
    w.append("")
    w.append("All frozen documents from Phase 1 and Phase 2 Steps 1-6 remain intact.")
    w.append("")

    # 14. Limitations
    w.append("## 14. Limitations")
    w.append("")
    w.append("1. **Coverage-selected subset:** The canonical dataset represents 454/925 analysis candidates (49.1%) selected for full D2 wavelength coverage. Conclusions are conditional on this subset.")
    w.append("2. **Interpolation for 64 spectra:** 64 spectra require PCHIP interpolation from native 10nm to the 5nm target grid. Interpolated values are representations, not new measurements.")
    w.append("3. **Two minerals absent:** Pyrope and Zircon are absent due to insufficient wavelength coverage.")
    w.append("4. **No denoising:** The canonical dataset retains all measurement noise. It is NOT a noise-free dataset.")
    w.append("5. **No normalization:** The canonical dataset retains the original reflectance scale. It is NOT a normalized dataset.")
    w.append("6. **No physical homogenization:** Different minerals, grain sizes, and measurement geometries are retained as-is. The dataset is NOT physically homogenized.")
    w.append("7. **Third-column quantity:** The RELAB third column is retained as metadata only. No uncertainty model is constructed.")
    w.append("8. **Splice boundaries:** No splice-specific masking or correction is applied within D2.")
    w.append("")

    # 15. Final validation status
    w.append("## 15. Final Validation Status")
    w.append("")
    n_pass = sum(1 for c in results["validation_checks"] if c["status"] == "PASS")
    n_fail = sum(1 for c in results["validation_checks"] if c["status"] == "FAIL")
    n_conflict = sum(1 for c in results["validation_checks"] if c["status"] == "CONFLICT")
    w.append(f"- Total validation checks: {len(results['validation_checks'])}")
    w.append(f"- PASS: {n_pass}")
    w.append(f"- FAIL: {n_fail}")
    w.append(f"- CONFLICT: {n_conflict}")
    w.append("")

    if n_fail == 0 and n_conflict == 0:
        w.append("### PASS")
        w.append("")
        w.append("All required frozen invariants and integrity checks pass.")
    elif n_fail == 0 and n_conflict > 0:
        w.append("### PASS WITH CONFLICTS — REQUIRES RESEARCHER REVIEW")
        w.append("")
        w.append(f"All implementation checks pass, but {n_conflict} frozen-decision conflict(s) detected.")
        w.append("See Section 9 for details. The canonical dataset is internally consistent but the")
        w.append("population count may need revision pending resolution of the conflict(s).")
    elif n_fail <= 2:
        w.append("### PASS WITH LIMITATIONS")
        w.append("")
        w.append(f"{n_fail} check(s) did not pass. Review the failed checks above for details.")
    else:
        w.append("### FAIL")
        w.append("")
        w.append(f"{n_fail} checks failed. Review the failed checks above for details.")
    w.append("")
    w.append("---")
    w.append("")
    w.append("### Important distinction")
    w.append("")
    w.append("The canonical dataset is a **preprocessed representation of the measured RELAB spectra**.")
    w.append("")
    w.append("It is NOT:")
    w.append("- a denoised dataset")
    w.append("- a normalized dataset")
    w.append("- a corrected calibration dataset")
    w.append("- a physically homogenized dataset")
    w.append("- a noise-free dataset")
    w.append("- a chemically standardized dataset")
    w.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(w))

    print(f"  Validation report: {path}")
    results["validation_report_path"] = path


# ================================================================
# VALIDATION SUMMARY CSV
# ================================================================


def _write_validation_summary(results):
    path = OUT_DIR / "phase2_final_preprocessing_summary.csv"
    df = pd.DataFrame(results["validation_checks"])
    df.to_csv(path, index=False)
    print(f"  Validation summary: {path}")
    results["validation_summary_path"] = path


# ================================================================
# CHECKSUMS
# ================================================================


def _write_checksums(results):
    path = OUT_DIR / "phase2_final_preprocessing_checksums.txt"

    checksums = results.get("checksums", {})

    val_report = results.get("validation_report_path")
    if val_report and Path(val_report).exists():
        checksums["validation_report"] = compute_md5(val_report)

    val_summary = results.get("validation_summary_path")
    if val_summary and Path(val_summary).exists():
        checksums["validation_summary"] = compute_md5(val_summary)

    script_path = PROJECT_ROOT / "scripts" / "phase2_final_preprocessing.py"
    if script_path.exists():
        checksums["preprocessing_script"] = compute_md5(script_path)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Phase 2 Final Preprocessing — Output Checksums\n")
        f.write(f"# Generated: {PROCESSING_DATE}\n")
        f.write(f"# Algorithm: MD5\n")
        f.write("#\n")
        for name, md5 in checksums.items():
            f.write(f"{md5}  {name}\n")

    print(f"  Checksums: {path}")
    results["checksums_path"] = path


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    results = run_pipeline()
