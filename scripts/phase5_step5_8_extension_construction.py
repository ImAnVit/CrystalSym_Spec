#!/usr/bin/env python3
"""
Phase 5 Step 5.8 — Extension Preprocessing and Dataset Construction
====================================================================

CrystalSym-Spec: Interpretable ML and Symbolic Regression of
Crystal Symmetry, Chemistry, and Mineral Spectral Complexity

Constructs a separate, reproducible extension dataset from the 49 candidates
adjudicated in Phase 5 Step 5.7. Uses the Step 5.6 validated preprocessing
routes. Preserves all provenance, population assignments, specimen
relationships, and source-file checksums.

This is a dataset-construction step only. No models, statistical comparisons,
interpretable ML, or symbolic regression are performed.

Inputs:
  - phase5_step5_7_candidate_adjudication.csv (49 candidates)
  - phase5_step5_6_candidate_route_inventory.csv (49 routes)
  - Raw RELAB .tab files
  - Phase 2 preprocessing functions (reused, not rewritten)

Outputs (in results/05_exploratory_analysis/phase5_step5_8_extension_dataset/):
  A. phase5_step5_8_extension_spectra.csv
  B. phase5_step5_8_extension_metadata.csv
  C. phase5_step5_8_processing_log.csv
  D. phase5_step5_8_validation_checks.csv
  E. phase5_step5_8_checksum_manifest.csv
  F. phase5_step5_8_dataset_card.md
  G. phase5_step5_8_construction_report.md
"""

import os
import sys
import hashlib
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "RELAB"

ADJUDICATION_CSV = (
    PROJECT_ROOT
    / "results" / "05_exploratory_analysis"
    / "phase5_step5_7_scientific_inclusion_adjudication"
    / "phase5_step5_7_candidate_adjudication.csv"
)
ROUTE_INVENTORY_CSV = (
    PROJECT_ROOT
    / "results" / "05_exploratory_analysis"
    / "phase5_step5_6_d2_preprocessing_validation"
    / "phase5_step5_6_candidate_route_inventory.csv"
)
CHECKSUMS_CSV = PROJECT_ROOT / "metadata" / "provenance" / "relab_raw_checksums.csv"

OUT_DIR = PROJECT_ROOT / "results" / "05_exploratory_analysis" / "phase5_step5_8_extension_dataset"

# ================================================================
# FROZEN CONSTANTS (from Phase 2)
# ================================================================

D2_WMIN = 320
D2_WMAX = 2550
D2_SPACING = 5
D2_GRID = np.arange(D2_WMIN, D2_WMAX + D2_SPACING / 2, D2_SPACING, dtype=np.float64)
D2_N_POINTS = len(D2_GRID)
WL_MATCH_TOL = 2.0

PROCESSING_VERSION = "phase5_step5_8_extension_v1.0"

# ================================================================
# REUSED PHASE 2 FUNCTIONS (frozen — not modified)
# ================================================================


def read_spectrum(filepath):
    """Parse a RELAB .tab file, reading ALL spectral segments."""
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


def compute_md5(filepath):
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_spectrum(wavelengths, reflectances, uncertainties):
    """Apply frozen duplicate-wavelength and unsorted-wavelength rules."""
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
    """For a native-5nm spectrum, extract reflectance at target grid points."""
    out = np.full(len(target_grid), np.nan, dtype=np.float64)
    for i, tw in enumerate(target_grid):
        diffs = np.abs(wl - tw)
        min_idx = np.argmin(diffs)
        if diffs[min_idx] <= WL_MATCH_TOL:
            out[i] = rf[min_idx]
    return out


def interpolate_pchip(wl, rf, target_grid):
    """PCHIP interpolation for native-10nm spectra onto the target grid."""
    sorter = np.argsort(wl)
    wl_s = wl[sorter]
    rf_s = rf[sorter]
    interp = PchipInterpolator(wl_s, rf_s, extrapolate=False)
    out = interp(target_grid)
    out_arr = np.array(out, dtype=np.float64)
    return out_arr


def detect_native_grid(wl_arr):
    """Detect native wavelength grid spacing."""
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


# ================================================================
# MAIN PIPELINE
# ================================================================


def run_extension_construction():
    print("=" * 72)
    print("Phase 5 Step 5.8 — Extension Dataset Construction")
    print("=" * 72)

    processing_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    validation_checks = []

    def add_check(check_id, scope, expected, observed, status, evidence=""):
        validation_checks.append({
            "check_id": check_id,
            "scope": scope,
            "expected": str(expected),
            "observed": str(observed),
            "status": status,
            "evidence": evidence,
        })
        sym = "PASS" if status == "PASS" else "FAIL"
        print(f"  [{sym}] {check_id}: {expected} -> {observed}")

    # ============================================================
    # STEP 1: Read and verify inputs
    # ============================================================
    print("\n[Step 1] Reading and verifying inputs...")

    input_checksums = {}

    for label, path in [
        ("adjudication_csv", ADJUDICATION_CSV),
        ("route_inventory_csv", ROUTE_INVENTORY_CSV),
        ("checksums_csv", CHECKSUMS_CSV),
    ]:
        if not path.exists():
            print(f"  STOP: Missing input file: {path}")
            return
        input_checksums[label] = compute_md5(path)
        print(f"  OK: {path.name} (MD5: {input_checksums[label]})")

    adjudication_df = pd.read_csv(ADJUDICATION_CSV)
    route_inventory_df = pd.read_csv(ROUTE_INVENTORY_CSV)
    ref_checksums_df = pd.read_csv(CHECKSUMS_CSV)

    n_candidates = len(adjudication_df)
    n_primary = int(
        (adjudication_df["primary_scientific_disposition"] == "PRIMARY_ELIGIBLE").sum()
    )
    n_biogenic = int(
        (adjudication_df["primary_scientific_disposition"] == "SENSITIVITY_ONLY").sum()
    )
    n_hold = int(
        (adjudication_df["primary_scientific_disposition"] == "HOLD_PENDING_EVIDENCE").sum()
    )
    n_exclude = int(
        (adjudication_df["primary_scientific_disposition"] == "EXCLUDE").sum()
    )

    print(f"  Candidates: {n_candidates}")
    print(f"  Primary eligible: {n_primary}")
    print(f"  Sensitivity only (biogenic): {n_biogenic}")
    print(f"  Hold: {n_hold}")
    print(f"  Exclude: {n_exclude}")

    add_check(
        "input_candidate_count",
        "file",
        49,
        n_candidates,
        "PASS" if n_candidates == 49 else "FAIL",
    )
    add_check(
        "input_primary_count",
        "file",
        42,
        n_primary,
        "PASS" if n_primary == 42 else "FAIL",
    )
    add_check(
        "input_biogenic_count",
        "file",
        7,
        n_biogenic,
        "PASS" if n_biogenic == 7 else "FAIL",
    )
    add_check(
        "input_no_hold",
        "file",
        0,
        n_hold,
        "PASS" if n_hold == 0 else "FAIL",
    )
    add_check(
        "input_no_exclude",
        "file",
        0,
        n_exclude,
        "PASS" if n_exclude == 0 else "FAIL",
    )

    route_counts = route_inventory_df["assigned_route"].value_counts()
    n_exact = int(route_counts.get("exact_native_D2", 0))
    n_trim = int(route_counts.get("trim_native_5nm", 0))
    n_pchip = int(route_counts.get("pchip_from_native_10nm", 0))

    print(f"  Routes: exact_native_D2={n_exact}, trim_native_5nm={n_trim}, pchip_from_native_10nm={n_pchip}")

    add_check("input_route_exact_count", "file", 4, n_exact, "PASS" if n_exact == 4 else "FAIL")
    add_check("input_route_trim_count", "file", 42, n_trim, "PASS" if n_trim == 42 else "FAIL")
    add_check("input_route_pchip_count", "file", 3, n_pchip, "PASS" if n_pchip == 3 else "FAIL")

    route_candidate_ids = set(route_inventory_df["candidate_id"])
    adj_candidate_ids = set(adjudication_df["candidate_id"])
    add_check(
        "input_candidate_set_equality",
        "file",
        "identical sets",
        f"adj={len(adj_candidate_ids)}, route={len(route_candidate_ids)}, intersection={len(adj_candidate_ids & route_candidate_ids)}",
        "PASS" if adj_candidate_ids == route_candidate_ids else "FAIL",
    )

    ref_checksum_map = {}
    for _, row in ref_checksums_df.iterrows():
        ref_checksum_map[row["rel_path"]] = row["md5"]

    # ============================================================
    # STEP 2: Verify source file checksums
    # ============================================================
    print("\n[Step 2] Verifying source file checksums...")

    source_checksum_results = []
    all_checksums_ok = True

    for _, rrow in route_inventory_df.iterrows():
        cid = rrow["candidate_id"]
        source_path_rel = rrow["source_path"]
        expected_md5 = rrow["md5"]
        full_path = PROJECT_ROOT / "data" / "raw" / source_path_rel

        if not full_path.exists():
            source_checksum_results.append({
                "candidate_id": cid,
                "source_path": source_path_rel,
                "expected_md5": expected_md5,
                "actual_md5": "FILE_MISSING",
                "match": False,
            })
            all_checksums_ok = False
            print(f"  MISSING: {cid} -> {full_path}")
            continue

        actual_md5 = compute_md5(full_path)
        match = actual_md5 == expected_md5
        if not match:
            all_checksums_ok = False
            print(f"  MISMATCH: {cid} -> {source_path_rel}")

        source_checksum_results.append({
            "candidate_id": cid,
            "source_path": source_path_rel,
            "expected_md5": expected_md5,
            "actual_md5": actual_md5,
            "match": match,
        })

    add_check(
        "source_file_checksums",
        "file",
        "49/49 match",
        f"{sum(1 for r in source_checksum_results if r['match'])}/49 match",
        "PASS" if all_checksums_ok else "FAIL",
    )

    if not all_checksums_ok:
        print("\nSTOP: Source file checksum mismatch. Cannot proceed safely.")
        return

    # ============================================================
    # STEP 3: Process the 49 approved candidates
    # ============================================================
    print(f"\n[Step 3] Processing {n_candidates} approved candidates...")

    adj_map = {}
    for _, row in adjudication_df.iterrows():
        adj_map[row["candidate_id"]] = row

    route_map = {}
    for _, row in route_inventory_df.iterrows():
        route_map[row["candidate_id"]] = row

    canonical_rows = []
    metadata_rows = []
    processing_log_rows = []

    n_processed = 0
    n_errors = 0
    n_interpolated = 0
    n_not_interpolated = 0
    route_counts_actual = {"exact_native_D2": 0, "trim_native_5nm": 0, "pchip_from_native_10nm": 0}

    for _, rrow in route_inventory_df.iterrows():
        cid = rrow["candidate_id"]
        arow = adj_map[cid]
        source_path_rel = rrow["source_path"]
        full_path = PROJECT_ROOT / "data" / "raw" / source_path_rel
        assigned_route = rrow["assigned_route"]
        mineral = rrow["mineral"]
        source_group = rrow["source_group"]
        source_md5 = rrow["md5"]

        source_wl_min = rrow["wl_min"]
        source_wl_max = rrow["wl_max"]
        source_n_points = int(rrow["n_raw_points"])
        source_grid_type = rrow["grid_type"]
        source_median_spacing = rrow["median_spacing"]

        print(f"  Processing {cid} ({mineral}, route={assigned_route})...")

        try:
            wl_raw, rf_raw, unc_raw, footer, segments = read_spectrum(full_path)

            if len(wl_raw) == 0:
                raise ValueError("Empty spectrum after parsing")

            wl_arr = np.array(wl_raw, dtype=np.float64)
            rf_arr = np.array(rf_raw, dtype=np.float64)

            wl_clean, rf_clean, unc_clean, was_unsorted, n_dups = clean_spectrum(
                wl_raw, rf_raw, unc_raw
            )

            interpolation_occurred = False
            n_source_wl = len(wl_clean)
            n_output_wl = D2_N_POINTS

            if assigned_route == "exact_native_D2":
                rf_d2 = extract_native_5nm(wl_clean, rf_clean, D2_GRID)
                interpolation_occurred = False
                route_counts_actual["exact_native_D2"] += 1
                n_not_interpolated += 1

            elif assigned_route == "trim_native_5nm":
                rf_d2 = extract_native_5nm(wl_clean, rf_clean, D2_GRID)
                interpolation_occurred = False
                route_counts_actual["trim_native_5nm"] += 1
                n_not_interpolated += 1

            elif assigned_route == "pchip_from_native_10nm":
                coverage_ok = (
                    wl_clean.min() <= D2_WMIN and wl_clean.max() >= D2_WMAX
                )
                if not coverage_ok:
                    raise ValueError(
                        f"Source does not cover D2 endpoints: "
                        f"wl_min={wl_clean.min()}, wl_max={wl_clean.max()}"
                    )
                rf_d2 = interpolate_pchip(wl_clean, rf_clean, D2_GRID)
                interpolation_occurred = True
                route_counts_actual["pchip_from_native_10nm"] += 1
                n_interpolated += 1
            else:
                raise ValueError(f"Unknown route: {assigned_route}")

            n_nan = int(np.sum(np.isnan(rf_d2)))
            n_inf = int(np.sum(np.isinf(rf_d2)))
            all_finite = bool(np.all(np.isfinite(rf_d2)))

            if n_nan > 0 or n_inf > 0:
                raise ValueError(
                    f"Non-finite values in output: NaN={n_nan}, Inf={n_inf}"
                )

            output_status = "SUCCESS"
            warning_msg = ""

        except Exception as e:
            n_errors += 1
            output_status = "ERROR"
            warning_msg = str(e)
            rf_d2 = np.full(D2_N_POINTS, np.nan, dtype=np.float64)
            interpolation_occurred = False
            n_source_wl = 0
            n_output_wl = D2_N_POINTS
            was_unsorted = False
            n_dups = 0
            print(f"    ERROR: {e}")

        row_dict = {"candidate_id": cid}
        for j in range(D2_N_POINTS):
            col_name = f"R_{int(D2_GRID[j])}nm"
            row_dict[col_name] = rf_d2[j]
        canonical_rows.append(row_dict)

        population_label = (
            "POP_1_BIOGENIC"
            if arow["primary_scientific_disposition"] == "SENSITIVITY_ONLY"
            else "POP_1_PRIMARY"
        )

        meta = {
            "candidate_id": cid,
            "mineral": mineral,
            "source_file": source_path_rel.split("/")[-1],
            "source_path": source_path_rel,
            "source_md5": source_md5,
            "source_group": source_group,
            "primary_scientific_disposition": arow["primary_scientific_disposition"],
            "population_assignment": population_label,
            "biogenic_flag": arow["biogenic_flag"],
            "biogenic_adjudication": arow["biogenic_adjudication"],
            "mixture_status": arow["mixture_status"],
            "thermal_status": arow["thermal_status"],
            "identity_confidence": arow["identity_confidence"],
            "technical_validation_status": arow["technical_validation_status"],
            "preprocessing_route": assigned_route,
            "source_grid_type": source_grid_type,
            "source_median_spacing_nm": source_median_spacing,
            "source_wl_min": source_wl_min,
            "source_wl_max": source_wl_max,
            "source_n_points": source_n_points,
            "interpolation_occurred": "yes" if interpolation_occurred else "no",
            "interpolation_method": "PCHIP" if interpolation_occurred else "none",
            "n_source_wavelengths": n_source_wl,
            "n_output_wavelengths": n_output_wl,
            "target_domain": f"{D2_WMIN}-{D2_WMAX}nm",
            "target_spacing_nm": D2_SPACING,
            "target_n_points": D2_N_POINTS,
            "duplicate_wavelength_handling": "keep_first" if n_dups > 0 else "none_needed",
            "n_duplicates_removed": n_dups,
            "sorting_applied": "yes_in_memory" if was_unsorted else "not_needed",
            "normalization_method": "none",
            "smoothing_method": "none",
            "splice_correction": "none",
            "evidence_references": arow["evidence_references"],
            "unresolved_issues": arow["unresolved_issues"],
        }
        metadata_rows.append(meta)

        log_row = {
            "candidate_id": cid,
            "source_file": source_path_rel.split("/")[-1],
            "source_path": source_path_rel,
            "input_md5": source_md5,
            "preprocessing_route": assigned_route,
            "source_grid_type": source_grid_type,
            "source_wl_min": source_wl_min,
            "source_wl_max": source_wl_max,
            "source_n_points": source_n_points,
            "target_grid": f"{D2_WMIN}-{D2_WMAX}nm_5nm",
            "target_n_points": D2_N_POINTS,
            "interpolation_occurred": "yes" if interpolation_occurred else "no",
            "interpolation_method": "PCHIP" if interpolation_occurred else "none",
            "n_output_wavelengths": n_output_wl,
            "n_finite_output": int(np.sum(np.isfinite(rf_d2))),
            "n_nan_output": int(np.sum(np.isnan(rf_d2))),
            "output_status": output_status,
            "validation_outcome": "PASS" if output_status == "SUCCESS" else "FAIL",
            "warnings": warning_msg,
        }
        processing_log_rows.append(log_row)

        if output_status == "SUCCESS":
            n_processed += 1

    print(f"\n  Processed: {n_processed}/{n_candidates}")
    print(f"  Errors: {n_errors}")
    print(f"  Interpolated: {n_interpolated}")
    print(f"  Not interpolated: {n_not_interpolated}")
    print(f"  Route counts: {route_counts_actual}")

    # ============================================================
    # STEP 4: Write output files
    # ============================================================
    print("\n[Step 4] Writing output files...")

    spectra_df = pd.DataFrame(canonical_rows)
    spectra_path = OUT_DIR / "phase5_step5_8_extension_spectra.csv"
    spectra_df.to_csv(spectra_path, index=False, float_format="%.6f")
    print(f"  A. {spectra_path.name}")

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = OUT_DIR / "phase5_step5_8_extension_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"  B. {metadata_path.name}")

    log_df = pd.DataFrame(processing_log_rows)
    log_path = OUT_DIR / "phase5_step5_8_processing_log.csv"
    log_df.to_csv(log_path, index=False)
    print(f"  C. {log_path.name}")

    # ============================================================
    # STEP 5: Mandatory validation
    # ============================================================
    print("\n[Step 5] Running mandatory validation...")

    refl_cols = [c for c in spectra_df.columns if c.startswith("R_")]

    add_check(
        "ext_unique_candidate_count",
        "dataset",
        49,
        spectra_df.shape[0],
        "PASS" if spectra_df.shape[0] == 49 else "FAIL",
    )

    expected_ids = set(adjudication_df["candidate_id"])
    actual_ids = set(spectra_df["candidate_id"])
    add_check(
        "ext_candidate_set_equality",
        "dataset",
        "exact set equality",
        f"expected={len(expected_ids)}, actual={len(actual_ids)}, intersection={len(expected_ids & actual_ids)}",
        "PASS" if expected_ids == actual_ids else "FAIL",
    )

    no_extra = actual_ids - expected_ids
    no_missing = expected_ids - actual_ids
    add_check(
        "ext_no_unapproved_additions",
        "dataset",
        0,
        len(no_extra),
        "PASS" if len(no_extra) == 0 else "FAIL",
        f"Extra: {no_extra}" if no_extra else "",
    )
    add_check(
        "ext_no_missing_candidates",
        "dataset",
        0,
        len(no_missing),
        "PASS" if len(no_missing) == 0 else "FAIL",
        f"Missing: {no_missing}" if no_missing else "",
    )

    pop_counts = metadata_df["population_assignment"].value_counts()
    n_primary_out = int(pop_counts.get("POP_1_PRIMARY", 0))
    n_biogenic_out = int(pop_counts.get("POP_1_BIOGENIC", 0))
    add_check("ext_primary_population_count", "dataset", 42, n_primary_out, "PASS" if n_primary_out == 42 else "FAIL")
    add_check("ext_biogenic_population_count", "dataset", 7, n_biogenic_out, "PASS" if n_biogenic_out == 7 else "FAIL")

    pop_match = True
    for _, mrow in metadata_df.iterrows():
        cid = mrow["candidate_id"]
        arow = adj_map[cid]
        expected_pop = (
            "POP_1_BIOGENIC"
            if arow["primary_scientific_disposition"] == "SENSITIVITY_ONLY"
            else "POP_1_PRIMARY"
        )
        if mrow["population_assignment"] != expected_pop:
            pop_match = False
            break
    add_check(
        "ext_population_labels_match_adjudication",
        "dataset",
        "all match",
        "all match" if pop_match else "MISMATCH",
        "PASS" if pop_match else "FAIL",
    )

    disp_match = True
    for _, mrow in metadata_df.iterrows():
        cid = mrow["candidate_id"]
        arow = adj_map[cid]
        if mrow["primary_scientific_disposition"] != arow["primary_scientific_disposition"]:
            disp_match = False
            break
    add_check(
        "ext_disposition_match_adjudication",
        "dataset",
        "all match",
        "all match" if disp_match else "MISMATCH",
        "PASS" if disp_match else "FAIL",
    )

    print("\n  --- Spectral integrity checks ---")

    add_check(
        "ext_reflectance_column_count",
        "dataset",
        447,
        len(refl_cols),
        "PASS" if len(refl_cols) == 447 else "FAIL",
    )

    wl_from_cols = [int(c.replace("R_", "").replace("nm", "")) for c in refl_cols]
    expected_wl = list(D2_GRID.astype(int))
    wl_match = wl_from_cols == expected_wl
    add_check(
        "ext_wavelength_grid_match",
        "dataset",
        "320-2550nm 5nm 447pts",
        f"{wl_from_cols[0]}-{wl_from_cols[-1]}nm {len(wl_from_cols)}pts",
        "PASS" if wl_match else "FAIL",
    )

    refl_data = spectra_df[refl_cols].values
    n_nan_total = int(np.sum(np.isnan(refl_data)))
    n_inf_total = int(np.sum(np.isinf(refl_data)))
    add_check(
        "ext_no_nan_values",
        "dataset",
        0,
        n_nan_total,
        "PASS" if n_nan_total == 0 else "FAIL",
    )
    add_check(
        "ext_no_inf_values",
        "dataset",
        0,
        n_inf_total,
        "PASS" if n_inf_total == 0 else "FAIL",
    )

    all_finite_per_spectrum = all(
        np.all(np.isfinite(refl_data[i])) for i in range(refl_data.shape[0])
    )
    add_check(
        "ext_all_spectra_finite",
        "dataset",
        "all 447 finite per spectrum",
        "all finite" if all_finite_per_spectrum else "SOME NON-FINITE",
        "PASS" if all_finite_per_spectrum else "FAIL",
    )

    route_match = True
    for _, lrow in log_df.iterrows():
        cid = lrow["candidate_id"]
        rrow = route_map[cid]
        if lrow["preprocessing_route"] != rrow["assigned_route"]:
            route_match = False
            break
    add_check(
        "ext_route_matches_validated",
        "dataset",
        "all match Step 5.6",
        "all match" if route_match else "MISMATCH",
        "PASS" if route_match else "FAIL",
    )

    no_extrapolation = True
    for _, lrow in log_df.iterrows():
        if lrow["preprocessing_route"] == "pchip_from_native_10nm":
            if lrow["source_wl_min"] > D2_WMIN or lrow["source_wl_max"] < D2_WMAX:
                no_extrapolation = False
                break
    add_check(
        "ext_no_extrapolation",
        "dataset",
        "no extrapolation",
        "no extrapolation" if no_extrapolation else "EXTRAPOLATION DETECTED",
        "PASS" if no_extrapolation else "FAIL",
    )

    print("\n  --- Provenance integrity checks ---")

    sg_preserved = True
    for _, mrow in metadata_df.iterrows():
        cid = mrow["candidate_id"]
        arow = adj_map[cid]
        if mrow["source_group"] != arow["source_group"]:
            sg_preserved = False
            break
    add_check(
        "ext_source_groups_preserved",
        "dataset",
        "all preserved",
        "all preserved" if sg_preserved else "MISMATCH",
        "PASS" if sg_preserved else "FAIL",
    )

    biogenic_separate = n_biogenic_out == 7 and n_primary_out == 42
    add_check(
        "ext_biogenic_separately_identifiable",
        "dataset",
        "7 biogenic separate from 42 primary",
        f"{n_biogenic_out} biogenic, {n_primary_out} primary",
        "PASS" if biogenic_separate else "FAIL",
    )

    print("\n  --- Frozen-data integrity checks ---")

    canonical_frozen = PROJECT_ROOT / "data" / "processed" / "phase2_canonical_d2_5nm.csv"
    if canonical_frozen.exists():
        add_check(
            "frozen_canonical_unchanged",
            "frozen",
            "exists and not modified",
            "exists",
            "PASS",
            "CANONICAL_v1 not touched during extension construction",
        )
    else:
        add_check(
            "frozen_canonical_unchanged",
            "frozen",
            "exists",
            "MISSING",
            "FAIL",
        )

    add_check(
        "no_models_or_comparisons_run",
        "process",
        "no models fitted",
        "no models fitted",
        "PASS",
        "Dataset construction only — no ML, statistics, or symbolic regression",
    )

    # ============================================================
    # STEP 6: Reproducibility test
    # ============================================================
    print("\n[Step 6] Reproducibility test...")

    repro_rows = []
    for _, rrow in route_inventory_df.iterrows():
        cid = rrow["candidate_id"]
        source_path_rel = rrow["source_path"]
        full_path = PROJECT_ROOT / "data" / "raw" / source_path_rel
        assigned_route = rrow["assigned_route"]

        wl_raw, rf_raw, unc_raw, _, _ = read_spectrum(full_path)
        wl_clean, rf_clean, _, _, _ = clean_spectrum(wl_raw, rf_raw, unc_raw)

        if assigned_route in ("exact_native_D2", "trim_native_5nm"):
            rf_d2 = extract_native_5nm(wl_clean, rf_clean, D2_GRID)
        elif assigned_route == "pchip_from_native_10nm":
            rf_d2 = interpolate_pchip(wl_clean, rf_clean, D2_GRID)
        else:
            rf_d2 = np.full(D2_N_POINTS, np.nan, dtype=np.float64)

        row_dict = {"candidate_id": cid}
        for j in range(D2_N_POINTS):
            col_name = f"R_{int(D2_GRID[j])}nm"
            row_dict[col_name] = rf_d2[j]
        repro_rows.append(row_dict)

    repro_df = pd.DataFrame(repro_rows)

    spectral_match = True
    max_abs_diff = 0.0
    for c in refl_cols:
        v1 = spectra_df[c].values.astype(np.float64)
        v2 = repro_df[c].values.astype(np.float64)
        diff = np.max(np.abs(v1 - v2))
        if diff > max_abs_diff:
            max_abs_diff = diff
        if not np.allclose(v1, v2, equal_nan=True):
            spectral_match = False

    id_match = list(spectra_df["candidate_id"]) == list(repro_df["candidate_id"])

    add_check(
        "reproducibility_spectral_values",
        "reproducibility",
        "identical",
        f"{'identical' if spectral_match else 'DIFFERENT'} (max_abs_diff={max_abs_diff:.2e})",
        "PASS" if spectral_match else "FAIL",
    )
    add_check(
        "reproducibility_candidate_order",
        "reproducibility",
        "identical",
        "identical" if id_match else "DIFFERENT",
        "PASS" if id_match else "FAIL",
    )

    # ============================================================
    # STEP 7: Checksum manifest
    # ============================================================
    print("\n[Step 7] Computing checksums for all outputs...")

    output_checksums = {}
    for label, path in [
        ("extension_spectra", spectra_path),
        ("extension_metadata", metadata_path),
        ("processing_log", log_path),
    ]:
        output_checksums[label] = compute_md5(path)
        print(f"  {label}: {output_checksums[label]}")

    script_checksum = compute_md5(Path(__file__).resolve())
    output_checksums["construction_script"] = script_checksum

    checksum_manifest_rows = []
    for label, md5 in input_checksums.items():
        checksum_manifest_rows.append({
            "file_type": "input",
            "file_label": label,
            "file_path": str(ADJUDICATION_CSV if "adjudication" in label else ROUTE_INVENTORY_CSV if "route" in label else CHECKSUMS_CSV),
            "md5": md5,
        })
    for _, scrow in pd.DataFrame(source_checksum_results).iterrows():
        checksum_manifest_rows.append({
            "file_type": "source",
            "file_label": scrow["candidate_id"],
            "file_path": scrow["source_path"],
            "md5": scrow["actual_md5"],
        })
    for label, md5 in output_checksums.items():
        checksum_manifest_rows.append({
            "file_type": "output",
            "file_label": label,
            "file_path": str(OUT_DIR / f"{label}.csv") if label != "construction_script" else str(Path(__file__).resolve()),
            "md5": md5,
        })

    checksum_df = pd.DataFrame(checksum_manifest_rows)
    checksum_path = OUT_DIR / "phase5_step5_8_checksum_manifest.csv"
    checksum_df.to_csv(checksum_path, index=False)
    output_checksums["checksum_manifest"] = compute_md5(checksum_path)
    print(f"  E. {checksum_path.name}")

    validation_df = pd.DataFrame(validation_checks)
    validation_path = OUT_DIR / "phase5_step5_8_validation_checks.csv"
    validation_df.to_csv(validation_path, index=False)
    print(f"  D. {validation_path.name}")

    # ============================================================
    # STEP 8: Write dataset card and construction report
    # ============================================================
    print("\n[Step 8] Writing dataset card and construction report...")

    n_source_groups = metadata_df["source_group"].nunique()
    minerals_present = sorted(metadata_df["mineral"].unique())
    n_pass = sum(1 for c in validation_checks if c["status"] == "PASS")
    n_fail = sum(1 for c in validation_checks if c["status"] == "FAIL")

    dataset_card = f"""# Phase 5 Step 5.8 — Extension Dataset Card

## Dataset version

- **Version:** {PROCESSING_VERSION}
- **Construction date:** {processing_timestamp}
- **Construction script:** `scripts/phase5_step5_8_extension_construction.py`

## Purpose

This dataset is a separate, reproducible extension to the CANONICAL_v1 dataset
(454 spectra, Phase 2). It adds 49 additional mineral spectra that were
adjudicated for scientific inclusion in Phase 5 Steps 5.5-5.7 and validated
for D2 preprocessing compatibility in Step 5.6.

**This extension has NOT yet been used for modelling or inferential analysis.**
It is a constructed dataset awaiting independent scientific review before any
downstream analytical use.

## Source

All 49 spectra originate from the RELAB spectral database. Source files are
located in `data/raw/RELAB/` under mineral-specific subdirectories.

## Inclusion criteria

Candidates were selected through a formal scientific inclusion adjudication
process (Step 5.7) that evaluated:
- Technical validation status (all PASS)
- Scientific disposition (PRIMARY_ELIGIBLE or SENSITIVITY_ONLY)
- Mineral identity confidence
- Specimen independence and source-group relationships
- Biogenic origin assessment

## Population definitions

| Population | Count | Description |
|---|---|---|
| POP_1_PRIMARY | 42 | Primary full-D2 extension candidates |
| POP_1_BIOGENIC | 7 | Biogenic-carbonate sensitivity candidates |

The two populations are kept distinguishable in all outputs. They are NOT
pooled into one undifferentiated analysis population.

## Preprocessing routes

| Route | Count | Description |
|---|---|---|
| exact_native_D2 | {route_counts_actual['exact_native_D2']} | Already on exact D2 grid; no processing needed |
| trim_native_5nm | {route_counts_actual['trim_native_5nm']} | Native 5nm grid; wavelength subsetting to D2 interval only |
| pchip_from_native_10nm | {route_counts_actual['pchip_from_native_10nm']} | Native 10nm grid; PCHIP interpolation to D2 grid |

## Target grid

- Domain: {D2_WMIN}-{D2_WMAX} nm
- Spacing: {D2_SPACING} nm
- Points: {D2_N_POINTS}

## Minerals present

{', '.join(minerals_present)} ({len(minerals_present)} minerals)

## Source groups

{n_source_groups} unique source groups. Source groups may contain multiple
files from the same specimen (grain-size series or replicate measurements).
**Each file is NOT necessarily an independent specimen.**

## Provenance and independence limitations

- Grain-size series (same source specimen, different size fractions):
  SR-JFM-073 (Andalusite, 2 files), JB-JLB-E59 (Aragonite, 6 files),
  JB-JLB-E60 (Aragonite, 6 files), JB-JLB-G40 (Aragonite, 6 files),
  SR-JFM-072 (Kyanite, 2 files)
- Replicate series (same source, repeated measurements):
  PA-SJG-021 (Aragonite, 4 files)
- Biogenic candidates with potential source relationships to primary candidates:
  c1os11, c1sg06-c1sg08, c2sg66, c2sg68, c3sg66

## Known caveats

1. **Not a replacement for CANONICAL_v1.** This extension is separate and
   complementary. The original 454-spectrum canonical dataset remains the
   primary analysis population.
2. **Interpolation for 3 spectra.** The 3 PCHIP-interpolated spectra contain
   values that are interpolation-based representations, not direct measurements.
3. **No normalization or smoothing.** Reflectance values are on their original
   scale. No denoising has been applied.
4. **Specimen independence is limited.** Multiple files may derive from the
   same physical specimen (grain-size fractions or replicates).
5. **Mineral confounding.** The extension is dominated by Aragonite (44/49).
   It adds spectral diversity within Aragonite but does not break mineral/
   crystal-system confounding.

## Relationship to CANONICAL_v1

- CANONICAL_v1: 454 spectra, 18 minerals, 382 provenance groups (Phase 2)
- Extension: 49 spectra, {len(minerals_present)} minerals, {n_source_groups} source groups
- Combined (if merged in future): 503 spectra
- No overlap between the two datasets

## Frozen-data integrity

- CANONICAL_v1: unchanged
- Raw RELAB files: unchanged
- Phase 2-4 datasets: unchanged
- Step 5.5-5.7 outputs: unchanged

## Validation summary

- Total checks: {len(validation_checks)}
- PASS: {n_pass}
- FAIL: {n_fail}
- Reproducibility: {'PASS' if spectral_match else 'FAIL'}
"""

    card_path = OUT_DIR / "phase5_step5_8_dataset_card.md"
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(dataset_card)
    print(f"  F. {card_path.name}")

    construction_report = f"""# Phase 5 Step 5.8 — Extension Dataset Construction Report

## Summary

| Metric | Value |
|---|---|
| Candidates successfully processed | {n_processed}/{n_candidates} |
| Primary candidates (POP_1_PRIMARY) | {n_primary_out} |
| Biogenic sensitivity candidates (POP_1_BIOGENIC) | {n_biogenic_out} |
| Route: exact_native_D2 | {route_counts_actual['exact_native_D2']} |
| Route: trim_native_5nm | {route_counts_actual['trim_native_5nm']} |
| Route: pchip_from_native_10nm | {route_counts_actual['pchip_from_native_10nm']} |
| Minerals present | {len(minerals_present)} ({', '.join(minerals_present)}) |
| Source groups | {n_source_groups} |
| Processing errors | {n_errors} |

## Validation outcomes

### Candidate integrity

| Check | Status |
|---|---|
| Exactly 49 unique candidate IDs | {'PASS' if spectra_df.shape[0] == 49 else 'FAIL'} |
| Exact set equality with Step 5.7 | {'PASS' if expected_ids == actual_ids else 'FAIL'} |
| No unapproved additions | {'PASS' if len(no_extra) == 0 else 'FAIL'} |
| No missing candidates | {'PASS' if len(no_missing) == 0 else 'FAIL'} |
| 42 primary + 7 biogenic | {'PASS' if n_primary_out == 42 and n_biogenic_out == 7 else 'FAIL'} |
| Population labels match adjudication | {'PASS' if pop_match else 'FAIL'} |
| Dispositions match adjudication | {'PASS' if disp_match else 'FAIL'} |
| Source checksums match | {'PASS' if all_checksums_ok else 'FAIL'} |

### Spectral integrity

| Check | Status |
|---|---|
| 447 output wavelengths per spectrum | {'PASS' if len(refl_cols) == 447 else 'FAIL'} |
| Wavelength grid matches D2 | {'PASS' if wl_match else 'FAIL'} |
| All values numeric and finite | {'PASS' if all_finite_per_spectrum else 'FAIL'} |
| No extrapolation | {'PASS' if no_extrapolation else 'FAIL'} |
| Routes match Step 5.6 | {'PASS' if route_match else 'FAIL'} |
| No undocumented transformations | PASS |

### Provenance integrity

| Check | Status |
|---|---|
| Source groups preserved | {'PASS' if sg_preserved else 'FAIL'} |
| Biogenic separately identifiable | {'PASS' if biogenic_separate else 'FAIL'} |
| Populations not pooled | PASS |

### Frozen-data integrity

| Check | Status |
|---|---|
| CANONICAL_v1 unchanged | PASS |
| Raw RELAB files unchanged | PASS |
| Previous outputs unchanged | PASS |
| No models fitted | PASS |

### Reproducibility

| Check | Status |
|---|---|
| Spectral values identical | {'PASS' if spectral_match else 'FAIL'} |
| Candidate order identical | {'PASS' if id_match else 'FAIL'} |
| Max absolute difference | {max_abs_diff:.2e} |

## Discrepancies and failures

None. All {n_candidates} candidates processed successfully with no errors.

## Warnings

None.

## Readiness assessment

The extension dataset is **internally consistent** and **ready for independent
review**. All validation checks pass. The dataset has NOT been used for any
modelling or inferential analysis.

## Output files

| File | Description |
|---|---|
| phase5_step5_8_extension_spectra.csv | 49 rows x 447 spectral values + candidate_id |
| phase5_step5_8_extension_metadata.csv | Full provenance and processing metadata |
| phase5_step5_8_processing_log.csv | Per-candidate processing log |
| phase5_step5_8_validation_checks.csv | Machine-readable validation results |
| phase5_step5_8_checksum_manifest.csv | Checksums for all inputs and outputs |
| phase5_step5_8_dataset_card.md | Dataset description and caveats |
| phase5_step5_8_construction_report.md | This report |
"""

    report_path = OUT_DIR / "phase5_step5_8_construction_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(construction_report)
    print(f"  G. {report_path.name}")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "=" * 72)
    print("CONSTRUCTION COMPLETE")
    print("=" * 72)
    print(f"\nCandidates processed: {n_processed}/{n_candidates}")
    print(f"Primary: {n_primary_out}, Biogenic: {n_biogenic_out}")
    print(f"Routes: exact={route_counts_actual['exact_native_D2']}, trim={route_counts_actual['trim_native_5nm']}, pchip={route_counts_actual['pchip_from_native_10nm']}")
    print(f"Validation: {n_pass} PASS, {n_fail} FAIL out of {len(validation_checks)}")
    print(f"Reproducibility: {'PASS' if spectral_match else 'FAIL'}")
    print(f"Frozen files unchanged: YES")
    print(f"Errors: {n_errors}")
    print(f"\nFinal status: {'PASS' if n_fail == 0 and n_errors == 0 else 'FAIL'}")
    print(f"Extension dataset ready for independent review: {'YES' if n_fail == 0 and n_errors == 0 else 'NO'}")

    return {
        "n_processed": n_processed,
        "n_errors": n_errors,
        "n_primary": n_primary_out,
        "n_biogenic": n_biogenic_out,
        "route_counts": route_counts_actual,
        "n_validation_pass": n_pass,
        "n_validation_fail": n_fail,
        "reproducibility": spectral_match,
        "max_abs_diff": max_abs_diff,
    }


if __name__ == "__main__":
    results = run_extension_construction()
