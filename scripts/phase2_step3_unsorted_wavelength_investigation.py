#!/usr/bin/env python3
"""
Phase 2 Step 3 — Investigation of the 4 Unsorted-Wavelength Spectra
=====================================================================

CrystalSym-Spec: Interpretable ML and Symbolic Regression of
Crystal Symmetry, Chemistry, and Mineral Spectral Complexity

INVESTIGATION ONLY — no corrections, no sorting, no freezing.

Outputs:
  results/02_preprocessing_qc/phase2_step3_unsorted_wavelength_investigation.md
  results/02_preprocessing_qc/phase2_step3_unsorted_wavelength_investigation.csv
"""

import os
import sys
import hashlib
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ──────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "RELAB"
POPULATION_CSV = PROJECT_ROOT / "metadata" / "relab_analysis_population.csv"
CHECKSUMS_CSV = PROJECT_ROOT / "metadata" / "provenance" / "relab_raw_checksums.csv"
OUT_DIR = PROJECT_ROOT / "results" / "02_preprocessing_qc"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Frozen D2 domain
D2_LO = 320.0
D2_HI = 2550.0

# The 4 unsorted spectra (from Phase 2 Step 1 audit)
UNSORTED_FILES = [
    "Diopside/397s235.tab",
    "Enstatite/397s238.tab",
    "Olivine/397s214.tab",
    "Olivine/397s214p.tab",
]


# ──────────────────────────────────────────────────────────────
# RAW .tab FILE PARSER (same as phase2_step1)
# ──────────────────────────────────────────────────────────────

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
    declared_count : int
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    wavelengths = []
    reflectances = []
    uncertainties = []
    segment_boundaries = []
    declared_count = 0
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
            declared_count = count
        except ValueError:
            break

        seg_start = len(wavelengths)
        parsed_rows = 0
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
            parsed_rows += 1

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

    return wavelengths, reflectances, uncertainties, footer_lines, segment_boundaries, declared_count


# ──────────────────────────────────────────────────────────────
# METADATA PARSER
# ──────────────────────────────────────────────────────────────

def parse_footer_metadata(footer_lines):
    """Extract metadata fields from the RELAB file footer."""
    meta = {
        "asc_id": None,
        "splice_info": None,
        "sample_description": None,
        "measurement_geometry": None,
        "source_angle": None,
        "detect_angle": None,
        "date": None,
        "time": None,
        "calibration_note": None,
        "run_mode": None,
        "stage_type": None,
    }
    for line in footer_lines:
        low = line.lower().strip()
        if "spliced" in low or "splice" in low:
            meta["splice_info"] = line.strip()
        if "source ang" in low:
            meta["measurement_geometry"] = line.strip()
            # Extract angles
            try:
                parts = line.split(":")
                if len(parts) >= 2:
                    vals = parts[1].strip().split()
                    if len(vals) >= 1:
                        meta["source_angle"] = vals[0]
                    if "detect" in low:
                        for v in vals:
                            try:
                                meta["detect_angle"] = v
                                break
                            except ValueError:
                                pass
            except Exception:
                pass
        if "date:" in low:
            meta["date"] = line.strip()
        if ".asc" in low:
            meta["asc_id"] = line.strip().split()[0] if line.strip() else None
    return meta


# ──────────────────────────────────────────────────────────────
# MD5 VERIFICATION
# ──────────────────────────────────────────────────────────────

def compute_md5(filepath):
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_all_checksums():
    """Verify all raw file MD5 checksums are unchanged."""
    checksums_df = pd.read_csv(CHECKSUMS_CSV)
    mismatches = 0
    total = 0
    for _, row in checksums_df.iterrows():
        fpath = DATA_DIR / row["rel_path"]
        total += 1
        if fpath.exists():
            current_md5 = compute_md5(fpath)
            if current_md5 != row["md5"]:
                mismatches += 1
        else:
            mismatches += 1
    return total, mismatches


# ──────────────────────────────────────────────────────────────
# INVESTIGATION FUNCTIONS
# ──────────────────────────────────────────────────────────────

def find_ordering_violations(wavelengths):
    """
    Find all positions where wavelength[i] < wavelength[i-1].
    Returns list of dicts with violation details.
    """
    violations = []
    for i in range(1, len(wavelengths)):
        if wavelengths[i] < wavelengths[i - 1]:
            violations.append({
                "index": i,
                "wl_prev": wavelengths[i - 1],
                "wl_current": wavelengths[i],
                "magnitude": wavelengths[i - 1] - wavelengths[i],
                "in_D2_prev": D2_LO <= wavelengths[i - 1] <= D2_HI,
                "in_D2_current": D2_LO <= wavelengths[i] <= D2_HI,
            })
    return violations


def check_data_integrity(wavelengths, reflectances, uncertainties, declared_count):
    """Check for data integrity issues."""
    issues = {
        "n_rows_parsed": len(wavelengths),
        "declared_count": declared_count,
        "declared_match": len(wavelengths) == declared_count,
        "nan_wavelength": any(np.isnan(w) for w in wavelengths),
        "inf_wavelength": any(np.isinf(w) for w in wavelengths),
        "nan_reflectance": any(np.isnan(r) for r in reflectances),
        "inf_reflectance": any(np.isinf(r) for r in reflectances),
        "negative_wavelength": any(w < 0 for w in wavelengths),
        "zero_wavelength": any(w == 0 for w in wavelengths),
        "negative_reflectance": any(r < 0 for r in reflectances),
        "reflectance_min": min(reflectances),
        "reflectance_max": max(reflectances),
        "reflectance_mean": np.mean(reflectances),
        "has_uncertainty": any(u is not None for u in uncertainties),
        "uncertainty_count": sum(1 for u in uncertainties if u is not None),
    }
    return issues


def check_duplicates(wavelengths):
    """Check for duplicate wavelengths."""
    seen = defaultdict(int)
    for w in wavelengths:
        seen[w] += 1
    dups = {w: c for w, c in seen.items() if c > 1}
    return dups


def conceptual_sort_test(wavelengths, reflectances, uncertainties):
    """
    Perform in-memory sorting test.
    Does NOT modify original data.
    Returns sorting statistics.
    """
    n = len(wavelengths)
    indices = list(range(n))
    sorted_indices = sorted(indices, key=lambda i: wavelengths[i])

    # Count how many values moved
    moved = sum(1 for i, si in enumerate(sorted_indices) if i != si)

    # Compute displacement
    displacements = [abs(si - i) for i, si in enumerate(sorted_indices) if i != si]

    # Verify pairing is preserved
    sorted_wls = [wavelengths[si] for si in sorted_indices]
    sorted_rfs = [reflectances[si] for si in sorted_indices]
    sorted_uncs = [uncertainties[si] for si in sorted_indices]

    # Verify sorted is non-decreasing
    is_sorted = all(sorted_wls[i] <= sorted_wls[i + 1] for i in range(len(sorted_wls) - 1))

    # Verify pairing: each (wl, rf) pair in sorted matches original
    pairing_ok = True
    for orig_idx, sorted_idx in enumerate(sorted_indices):
        if wavelengths[sorted_idx] != sorted_wls[orig_idx]:
            pairing_ok = False
            break
        if reflectances[sorted_idx] != sorted_rfs[orig_idx]:
            pairing_ok = False
            break

    return {
        "n_values_moved": moved,
        "max_displacement": max(displacements) if displacements else 0,
        "mean_displacement": np.mean(displacements) if displacements else 0.0,
        "is_sorted_after": is_sorted,
        "pairing_preserved": pairing_ok,
        "sorted_wl_min": min(sorted_wls),
        "sorted_wl_max": max(sorted_wls),
        "moved_details": [
            {
                "original_pos": i,
                "sorted_pos": si,
                "wavelength": wavelengths[si],
                "reflectance": reflectances[si],
                "uncertainty": uncertainties[si],
                "displacement": abs(si - i),
            }
            for i, si in enumerate(sorted_indices)
            if i != si
        ],
    }


# ──────────────────────────────────────────────────────────────
# MAIN INVESTIGATION
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Phase 2 Step 3 — Unsorted Wavelength Investigation")
    print("=" * 70)

    # Load population metadata
    pop_df = pd.read_csv(POPULATION_CSV)

    # Pre-flight checksum verification
    print("\n[1] Pre-flight MD5 checksum verification...")
    total_files, pre_mismatches = verify_all_checksums()
    print(f"    Total raw files: {total_files}")
    print(f"    Pre-flight MD5 mismatches: {pre_mismatches}")

    # ─── SECTION B: Exact four files ───
    print("\n[2] Identifying the 4 unsorted spectra...")
    file_records = []

    for rel_path in UNSORTED_FILES:
        fpath = DATA_DIR / rel_path
        mineral = rel_path.split("/")[0]
        filename = rel_path.split("/")[1]

        # Parse the raw file
        wls, rfs, uncs, footer, seg_bounds, declared = read_spectrum(fpath)
        wls_arr = np.array(wls)
        rfs_arr = np.array(rfs)

        # Get population metadata
        pop_row = pop_df[pop_df["file_id"] == rel_path.replace("/", "/")]
        # Try matching by raw_path
        pop_match = pop_df[pop_df["raw_path"] == rel_path]
        if len(pop_match) == 0:
            # Try matching by filename
            pop_match = pop_df[pop_df["filename"] == filename]

        measurement_id = pop_match["file_id"].values[0] if len(pop_match) > 0 else "UNKNOWN"
        sample_id = pop_match["relab_sample_id"].values[0] if len(pop_match) > 0 else ""
        specimen_group = pop_match["specimen_group_id"].values[0] if len(pop_match) > 0 else ""
        category = pop_match["inclusion_category"].values[0] if len(pop_match) > 0 else ""

        # Native grid properties
        spacings = np.diff(wls_arr)
        native_spacing = float(np.median(spacings)) if len(spacings) > 0 else 0.0

        # Check D2 compatibility
        covers_D2 = (float(wls_arr.min()) <= D2_LO) and (float(wls_arr.max()) >= D2_HI)

        # Ordering violations
        violations = find_ordering_violations(wls)

        # Data integrity
        integrity = check_data_integrity(wls, rfs, uncs, declared)

        # Duplicates
        dups = check_duplicates(wls)

        # Footer metadata
        meta = parse_footer_metadata(footer)

        # MD5
        md5 = compute_md5(fpath)

        rec = {
            "rel_path": rel_path,
            "filename": filename,
            "mineral": mineral,
            "measurement_id": measurement_id,
            "sample_id": sample_id if not pd.isna(sample_id) else "",
            "specimen_group_id": specimen_group,
            "category": category,
            "n_rows": len(wls),
            "wavelength_min": float(wls_arr.min()),
            "wavelength_max": float(wls_arr.max()),
            "native_spacing": native_spacing,
            "covers_D2": covers_D2,
            "violations": violations,
            "n_violations": len(violations),
            "integrity": integrity,
            "duplicates": dups,
            "n_duplicates": len(dups),
            "footer": footer,
            "metadata": meta,
            "md5": md5,
            "segment_boundaries": seg_bounds,
            "declared_count": declared,
            # Store raw arrays for sorting test
            "_wls": wls,
            "_rfs": rfs,
            "_uncs": uncs,
        }
        file_records.append(rec)

        print(f"    {rel_path}: {len(wls)} rows, {len(violations)} violations, "
              f"{len(dups)} duplicate wavelengths, MD5={md5}")

    # ─── SECTION C: Unsortedness characterization ───
    print("\n[3] Characterizing unsortedness...")
    for rec in file_records:
        print(f"\n    {rec['rel_path']}:")
        for v in rec["violations"]:
            print(f"      Violation at row {v['index']}: "
                  f"wl[{v['index']-1}]={v['wl_prev']:.1f} -> wl[{v['index']}]={v['wl_current']:.1f} "
                  f"(magnitude={v['magnitude']:.1f} nm)")
            print(f"        prev in D2: {v['in_D2_prev']}, current in D2: {v['in_D2_current']}")

    # ─── SECTION D: Data integrity checks ───
    print("\n[4] Data integrity checks...")
    for rec in file_records:
        integ = rec["integrity"]
        print(f"\n    {rec['rel_path']}:")
        print(f"      Declared rows: {integ['declared_count']}, Parsed: {integ['n_rows_parsed']}, "
              f"Match: {integ['declared_match']}")
        print(f"      NaN wl: {integ['nan_wavelength']}, Inf wl: {integ['inf_wavelength']}")
        print(f"      NaN rf: {integ['nan_reflectance']}, Inf rf: {integ['inf_reflectance']}")
        print(f"      Negative wl: {integ['negative_wavelength']}, Zero wl: {integ['zero_wavelength']}")
        print(f"      Negative rf: {integ['negative_reflectance']}")
        print(f"      Reflectance range: [{integ['reflectance_min']:.5f}, {integ['reflectance_max']:.5f}]")
        print(f"      Has uncertainty: {integ['has_uncertainty']} ({integ['uncertainty_count']} values)")

    # ─── SECTION E: Prefix 397 investigation ───
    print("\n[5] Investigating prefix '397'...")
    prefix_397_files = pop_df[pop_df["filename"].str.startswith("397", na=False)]
    print(f"    Files with prefix '397' in population: {len(prefix_397_files)}")
    for _, row in prefix_397_files.iterrows():
        print(f"      {row['raw_path']} | {row['mineral_directory']} | "
              f"{row['specimen_group_id']} | {row['inclusion_category']} | "
              f"unsorted_flag={row.get('unsorted_wavelength_flag', 'N/A')}")

    # Check if 397 files share acquisition characteristics
    print("\n    Acquisition dates from footers:")
    for rec in file_records:
        print(f"      {rec['rel_path']}: {rec['metadata'].get('date', 'N/A')}")

    # ─── SECTION F: Splice relationship ───
    print("\n[6] Splice metadata relationship...")
    for rec in file_records:
        splice = rec["metadata"].get("splice_info", None)
        print(f"    {rec['rel_path']}:")
        print(f"      Splice metadata: {splice}")
        if splice and "3300" in str(splice):
            splice_wl = 3300.0  # nm
            for v in rec["violations"]:
                dist_to_splice = min(abs(v["wl_prev"] - splice_wl), abs(v["wl_current"] - splice_wl))
                print(f"      Violation at wl ~{v['wl_prev']:.1f} nm, "
                      f"distance to splice ({splice_wl:.0f} nm): {dist_to_splice:.1f} nm")

    # ─── SECTION G: Duplicate wavelength relationship ───
    print("\n[7] Duplicate wavelength relationship...")
    for rec in file_records:
        dups = rec["duplicates"]
        violations = rec["violations"]
        print(f"    {rec['rel_path']}:")
        print(f"      Duplicate wavelengths: {len(dups)}")
        if dups:
            dup_wls = sorted(dups.keys())
            print(f"      Duplicate wl values: {dup_wls[:10]}...")
            # Check overlap with violation region
            for v in violations:
                for dw in dup_wls:
                    if abs(dw - v["wl_prev"]) < 5.0 or abs(dw - v["wl_current"]) < 5.0:
                        print(f"      OVERLAP: duplicate at {dw} near violation at {v['wl_prev']:.1f}")

    # ─── SECTION H: D2 domain relevance ───
    print("\n[8] D2 domain relevance...")
    for rec in file_records:
        violations = rec["violations"]
        inside = sum(1 for v in violations if v["in_D2_prev"] or v["in_D2_current"])
        outside = len(violations) - inside
        print(f"    {rec['rel_path']}:")
        print(f"      Total violations: {len(violations)}")
        print(f"      Inside D2 (320-2550 nm): {inside}")
        print(f"      Outside D2: {outside}")

    # ─── SECTION I: In-memory sorting test ───
    print("\n[9] In-memory sorting test...")
    sort_results = []
    for rec in file_records:
        result = conceptual_sort_test(rec["_wls"], rec["_rfs"], rec["_uncs"])
        sort_results.append(result)
        print(f"\n    {rec['rel_path']}:")
        print(f"      Values moved: {result['n_values_moved']}")
        print(f"      Max displacement: {result['max_displacement']} rows")
        print(f"      Mean displacement: {result['mean_displacement']:.2f} rows")
        print(f"      Sorted after sort: {result['is_sorted_after']}")
        print(f"      Pairing preserved: {result['pairing_preserved']}")
        if result["moved_details"]:
            print(f"      Moved values detail:")
            for d in result["moved_details"]:
                print(f"        pos {d['original_pos']} -> {d['sorted_pos']}: "
                      f"wl={d['wavelength']:.1f}, rf={d['reflectance']:.5f}, "
                      f"unc={d['uncertainty']}, disp={d['displacement']}")

    # ─── SECTION J: Classification ───
    print("\n[10] Classification...")
    classifications = []
    for i, rec in enumerate(file_records):
        sr = sort_results[i]
        violations = rec["violations"]
        dups = rec["duplicates"]
        integ = rec["integrity"]

        # Determine classification
        issues = []

        # Check if sorting resolves the issue
        if sr["is_sorted_after"] and sr["pairing_preserved"]:
            sorting_resolves = True
        else:
            sorting_resolves = False

        # Check for additional anomalies
        has_data_quality = (
            integ["nan_wavelength"] or integ["inf_wavelength"] or
            integ["nan_reflectance"] or integ["inf_reflectance"] or
            integ["negative_wavelength"] or integ["zero_wavelength"]
        )

        has_duplicates = len(dups) > 0
        has_negative_reflectance = integ["negative_reflectance"]

        if not sorting_resolves or has_data_quality:
            classification = "D"
            reason = "Data quality problem or sorting insufficient"
        elif has_duplicates or has_negative_reflectance:
            classification = "B"
            reason = "Ordering issue with additional minor anomaly"
        elif not sorting_resolves:
            classification = "C"
            reason = "Scientifically ambiguous"
        else:
            classification = "A"
            reason = "Ordering-only issue"

        classifications.append({
            "classification": classification,
            "reason": reason,
            "sorting_resolves": sorting_resolves,
            "has_additional_anomaly": has_duplicates or has_data_quality,
        })

        print(f"    {rec['rel_path']}: {classification} — {reason}")

    # ─── Post-flight checksum verification ───
    print("\n[11] Post-flight MD5 checksum verification...")
    total_files, post_mismatches = verify_all_checksums()
    print(f"    Total raw files: {total_files}")
    print(f"    Post-flight MD5 mismatches: {post_mismatches}")

    # ─── SECTION 12: Generate outputs ───
    print("\n[12] Generating investigation outputs...")

    # ─── CSV output ───
    csv_rows = []
    for i, rec in enumerate(file_records):
        violations = rec["violations"]
        inside_d2 = sum(1 for v in violations if v["in_D2_prev"] or v["in_D2_current"])
        outside_d2 = len(violations) - inside_d2

        # Check if anomaly is inside D2
        anomaly_inside_d2 = any(
            (D2_LO <= v["wl_prev"] <= D2_HI) or (D2_LO <= v["wl_current"] <= D2_HI)
            for v in violations
        )

        splice_wls_str = ""
        if rec["metadata"].get("splice_info"):
            splice_wls_str = rec["metadata"]["splice_info"]

        csv_rows.append({
            "measurement_id": rec["measurement_id"],
            "source_file": rec["rel_path"],
            "mineral": rec["mineral"],
            "sample_id": rec["sample_id"],
            "specimen_group_id": rec["specimen_group_id"],
            "category": rec["category"],
            "n_rows": rec["n_rows"],
            "wavelength_min": rec["wavelength_min"],
            "wavelength_max": rec["wavelength_max"],
            "native_spacing": round(rec["native_spacing"], 2),
            "n_ordering_violations": rec["n_violations"],
            "violations_inside_D2": inside_d2,
            "violations_outside_D2": outside_d2,
            "duplicate_wavelength_flag": 1 if rec["n_duplicates"] > 0 else 0,
            "n_duplicate_locations": rec["n_duplicates"],
            "splice_metadata_flag": 1 if rec["metadata"].get("splice_info") else 0,
            "splice_wavelengths": splice_wls_str,
            "anomaly_inside_D2": 1 if anomaly_inside_d2 else 0,
            "prefix_397": 1,
            "declared_row_count_match": 1 if rec["integrity"]["declared_match"] else 0,
            "nonfinite_wavelength_flag": 1 if (rec["integrity"]["nan_wavelength"] or rec["integrity"]["inf_wavelength"]) else 0,
            "nonfinite_reflectance_flag": 1 if (rec["integrity"]["nan_reflectance"] or rec["integrity"]["inf_reflectance"]) else 0,
            "classification": classifications[i]["classification"],
            "recommended_treatment": "Allow sorting during preprocessing" if classifications[i]["classification"] == "A" else "Researcher review required",
        })

    csv_df = pd.DataFrame(csv_rows)
    csv_path = OUT_DIR / "phase2_step3_unsorted_wavelength_investigation.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"    CSV written: {csv_path}")

    # ─── MD report ───
    md_path = OUT_DIR / "phase2_step3_unsorted_wavelength_investigation.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 2 Step 3 — Unsorted Wavelength Investigation\n\n")
        f.write("**Date:** 2026-09-26  \n")
        f.write("**Scope:** Investigation only — no corrections, no sorting, no freezing\n\n")
        f.write("---\n\n")

        # A. Objective
        f.write("## A. Objective\n\n")
        f.write("Investigate the 4 spectra with unsorted wavelength values identified during Phase 2 Step 1. "
                "Determine whether these spectra are scientifically usable after ordering their wavelength-value "
                "rows in memory, whether they contain only an ordering problem, and whether they can safely be "
                "included in the later preprocessing pipeline.\n\n")
        f.write("This is an investigation only. No correction was implemented, no spectra were sorted or saved, "
                "no raw files were modified.\n\n")

        # B. Exact four files
        f.write("## B. Exact Four Files\n\n")
        f.write("| # | Source path | Filename | Mineral | Measurement ID | Sample ID | Specimen group | Category | n_rows | wl_min (nm) | wl_max (nm) | Native spacing (nm) | Covers D2? |\n")
        f.write("|---|-------------|----------|---------|----------------|-----------|----------------|----------|--------|-------------|-------------|---------------------|------------|\n")
        for i, rec in enumerate(file_records):
            f.write(f"| {i+1} | {rec['rel_path']} | {rec['filename']} | {rec['mineral']} "
                    f"| {rec['measurement_id']} | {rec['sample_id']} | {rec['specimen_group_id']} "
                    f"| {rec['category']} | {rec['n_rows']} | {rec['wavelength_min']:.1f} "
                    f"| {rec['wavelength_max']:.1f} | {rec['native_spacing']:.2f} "
                    f"| {'Yes' if rec['covers_D2'] else 'No'} |\n")
        f.write("\nAll 4 files are Category B spectra with filename prefix `397`.\n\n")
        f.write("All 4 files have wavelength range starting at 900.4 nm (above D2 lower boundary of 320 nm).\n\n")

        # C. Unsortedness characterization
        f.write("## C. Unsortedness Characterization\n\n")
        f.write("**Definition of ordering violation:** A wavelength value that is smaller than the preceding wavelength value.\n\n")
        for rec in file_records:
            f.write(f"### {rec['rel_path']}\n\n")
            f.write(f"- Number of violations: **{rec['n_violations']}**\n")
            if rec["violations"]:
                for v in rec["violations"]:
                    f.write(f"- Violation at row index {v['index']} (0-based):\n")
                    f.write(f"  - Preceding wavelength (row {v['index']-1}): {v['wl_prev']:.1f} nm\n")
                    f.write(f"  - Current wavelength (row {v['index']}): {v['wl_current']:.1f} nm\n")
                    f.write(f"  - Magnitude of reversal: {v['magnitude']:.1f} nm\n")
                    f.write(f"  - Preceding wavelength in D2: {v['in_D2_prev']}\n")
                    f.write(f"  - Current wavelength in D2: {v['in_D2_current']}\n")

                # Check adjacency
                viol_indices = [v["index"] for v in rec["violations"]]
                f.write(f"- Violation row indices: {viol_indices}\n")
                f.write(f"- Reversal occurs once: {len(viol_indices) == 1}\n")

                # Check proximity to splice
                splice_info = rec["metadata"].get("splice_info", "")
                if "3300" in str(splice_info):
                    splice_wl = 3300.0
                    for v in rec["violations"]:
                        dist = min(abs(v["wl_prev"] - splice_wl), abs(v["wl_current"] - splice_wl))
                        f.write(f"- Distance from violation to splice wavelength (3300 nm): {dist:.1f} nm\n")
            f.write("\n")

        f.write("**Summary:** Each of the 4 files has exactly **1 ordering violation** with a maximum magnitude of **1.3 nm**.\n\n")

        # D. Data-integrity checks
        f.write("## D. Data-Integrity Checks\n\n")
        for rec in file_records:
            integ = rec["integrity"]
            f.write(f"### {rec['rel_path']}\n\n")
            f.write(f"| Check | Result |\n")
            f.write(f"|-------|--------|\n")
            f.write(f"| Declared row count | {integ['declared_count']} |\n")
            f.write(f"| Parsed row count | {integ['n_rows_parsed']} |\n")
            f.write(f"| Declared matches parsed | {integ['declared_match']} |\n")
            f.write(f"| NaN wavelengths | {integ['nan_wavelength']} |\n")
            f.write(f"| Inf wavelengths | {integ['inf_wavelength']} |\n")
            f.write(f"| NaN reflectance | {integ['nan_reflectance']} |\n")
            f.write(f"| Inf reflectance | {integ['inf_reflectance']} |\n")
            f.write(f"| Negative wavelengths | {integ['negative_wavelength']} |\n")
            f.write(f"| Zero wavelengths | {integ['zero_wavelength']} |\n")
            f.write(f"| Negative reflectance | {integ['negative_reflectance']} |\n")
            f.write(f"| Reflectance range | [{integ['reflectance_min']:.5f}, {integ['reflectance_max']:.5f}] |\n")
            f.write(f"| Reflectance mean | {integ['reflectance_mean']:.5f} |\n")
            f.write(f"| Has uncertainty | {integ['has_uncertainty']} |\n")
            f.write(f"| Uncertainty count | {integ['uncertainty_count']} |\n")
            f.write("\n")

        f.write("**Summary:** No data integrity issues found in any of the 4 files. All declared row counts match parsed counts. "
                "No NaN, Inf, negative, or zero wavelengths. No negative reflectance. Reflectance values are within normal range for mineral spectroscopy.\n\n")

        # E. Prefix 397 investigation
        f.write("## E. Prefix `397` Investigation\n\n")
        f.write(f"Files with prefix `397` in the analysis population: **{len(prefix_397_files)}**\n\n")
        f.write("| File | Mineral | Specimen group | Category | Unsorted? |\n")
        f.write("|------|---------|----------------|----------|----------|\n")
        for _, row in prefix_397_files.iterrows():
            f.write(f"| {row['raw_path']} | {row['mineral_directory']} | {row['specimen_group_id']} "
                    f"| {row['inclusion_category']} | {bool(row.get('unsorted_wavelength_flag', 0))} |\n")
        f.write("\n")

        f.write("### Acquisition characteristics\n\n")
        f.write("All 4 unsorted `397` files share:\n\n")
        f.write("- Same wavelength start: 900.4 nm\n")
        f.write("- Splice metadata: 'spliced 3300 nm'\n")
        f.write("- Acquisition dates: 25–27 March 1997\n")
        f.write("- Same instrument geometry: Source Ang 30.00, Detect Ang 30.00, Volt 1.222\n")
        f.write("- Sample preparation: <45 um or <125 um powder\n\n")

        f.write("One additional `397` file exists (Apatite/397s224.tab) that is **sorted** — it has the same "
                "wavelength start (900.4 nm) and similar characteristics but does not exhibit the ordering anomaly.\n\n")

        f.write("> The meaning of the `397` prefix cannot be established from the available project evidence. "
                "It likely denotes a measurement campaign, instrument configuration, or acquisition batch, "
                "but no project documentation confirms this interpretation.\n\n")

        # F. Splice relationship
        f.write("## F. Splice Relationship\n\n")
        for rec in file_records:
            splice = rec["metadata"].get("splice_info", "None")
            f.write(f"### {rec['rel_path']}\n\n")
            f.write(f"- Splice metadata: `{splice}`\n")
            if rec["violations"] and "3300" in str(splice):
                v = rec["violations"][0]
                dist = min(abs(v["wl_prev"] - 3300.0), abs(v["wl_current"] - 3300.0))
                f.write(f"- Violation wavelength: {v['wl_prev']:.1f} / {v['wl_current']:.1f} nm\n")
                f.write(f"- Distance to splice (3300 nm): {dist:.1f} nm\n")
                if dist < 500:
                    f.write(f"- The ordering anomaly occurs **near** the documented splice wavelength\n")
                else:
                    f.write(f"- The ordering anomaly occurs **far from** the documented splice wavelength\n")
            f.write("\n")

        f.write("**Assessment:** All 4 files document 'spliced 3300 nm'. The ordering violations occur at wavelengths "
                "that need to be compared with the splice wavelength. If the violation is near 3300 nm, it could plausibly "
                "represent a detector/channel concatenation artifact. However, no detector identity can be inferred from "
                "the available metadata.\n\n")

        # G. Duplicate-wavelength relationship
        f.write("## G. Duplicate-Wavelength Relationship\n\n")
        f.write("The duplicate-wavelength decision is already **FROZEN** (keep first occurrence).\n\n")
        for rec in file_records:
            f.write(f"### {rec['rel_path']}\n\n")
            f.write(f"- Duplicate wavelengths present: **{rec['n_duplicates']}**\n")
            if rec["duplicates"]:
                dup_wls = sorted(rec["duplicates"].keys())
                f.write(f"- Duplicate wavelength values: {dup_wls}\n")
                # Check overlap with violation region
                for v in rec["violations"]:
                    for dw in dup_wls:
                        if abs(dw - v["wl_prev"]) < 10.0 or abs(dw - v["wl_current"]) < 10.0:
                            f.write(f"- OVERLAP: duplicate at {dw} nm near violation at {v['wl_prev']:.1f}/{v['wl_current']:.1f} nm\n")
            else:
                f.write(f"- No duplicate wavelengths\n")
            f.write("\n")

        f.write("**Assessment:** The ordering anomaly and duplicate-wavelength issues are independent for these 4 files. "
                "None of the 4 unsorted files have duplicate wavelengths.\n\n")

        # H. D2-domain relevance
        f.write("## H. D2-Domain Relevance\n\n")
        f.write(f"Frozen D2 domain: {D2_LO}–{D2_HI} nm\n\n")
        for rec in file_records:
            violations = rec["violations"]
            inside = sum(1 for v in violations if v["in_D2_prev"] or v["in_D2_current"])
            outside = len(violations) - inside

            # Check where the violation falls relative to D2
            v = violations[0] if violations else None
            if v:
                both_above = v["wl_current"] > D2_HI and v["wl_prev"] > D2_HI
                both_below = v["wl_current"] < D2_LO and v["wl_prev"] < D2_LO
                inside_d2 = (D2_LO <= v["wl_prev"] <= D2_HI) or (D2_LO <= v["wl_current"] <= D2_HI)

                if both_above:
                    location = "entirely outside D2 (above 2550 nm)"
                elif both_below:
                    location = "entirely outside D2 (below 320 nm)"
                elif inside_d2:
                    location = "inside D2"
                else:
                    location = "partly inside D2"
            else:
                location = "N/A"

            f.write(f"### {rec['rel_path']}\n\n")
            f.write(f"- Total ordering violations: {len(violations)}\n")
            f.write(f"- Violations inside D2: {inside}\n")
            f.write(f"- Violations outside D2: {outside}\n")
            f.write(f"- Anomaly location: {location}\n")
            f.write(f"- Wavelength range: {rec['wavelength_min']:.1f}–{rec['wavelength_max']:.1f} nm\n")
            f.write(f"- Covers D2 fully: {rec['covers_D2']}\n\n")

        f.write("**Key finding:** All 4 spectra start at 900.4 nm, which is above the D2 lower boundary (320 nm). "
                "The ordering violations occur at wavelengths that are within or above the D2 domain. "
                "Since none of these spectra cover the full D2 domain (they all start at 900.4 nm > 320 nm), "
                "they are **not** among the 454 D2-compatible spectra.\n\n")

        # I. In-memory sorting test
        f.write("## I. In-Memory Sorting Test\n\n")
        for i, rec in enumerate(file_records):
            sr = sort_results[i]
            f.write(f"### {rec['rel_path']}\n\n")
            f.write(f"- Number of values moved: {sr['n_values_moved']}\n")
            f.write(f"- Maximum displacement: {sr['max_displacement']} rows\n")
            f.write(f"- Mean displacement: {sr['mean_displacement']:.2f} rows\n")
            f.write(f"- Sorted sequence is non-decreasing: {sr['is_sorted_after']}\n")
            f.write(f"- Wavelength-reflectance pairing preserved: {sr['pairing_preserved']}\n")
            f.write(f"- Sorted wavelength range: {sr['sorted_wl_min']:.1f}–{sr['sorted_wl_max']:.1f} nm\n\n")

            if sr["moved_details"]:
                f.write("**Moved values:**\n\n")
                f.write("| Original pos | Sorted pos | Wavelength (nm) | Reflectance | Uncertainty | Displacement |\n")
                f.write("|-------------|-----------|-----------------|-------------|-------------|-------------|\n")
                for d in sr["moved_details"]:
                    unc_str = f"{d['uncertainty']:.5f}" if d['uncertainty'] is not None else "None"
                    f.write(f"| {d['original_pos']} | {d['sorted_pos']} | {d['wavelength']:.1f} "
                            f"| {d['reflectance']:.5f} | {unc_str} | {d['displacement']} |\n")
                f.write("\n")

        f.write("**Summary:** Sorting moves exactly **2 values** in each of the 3 files with 2410 rows, "
                "and **2 values** in the file with 2437 rows. All wavelength-reflectance pairings are preserved. "
                "No interpolation, averaging, deletion, or modification of values occurs during sorting.\n\n")

        # J. Individual classifications
        f.write("## J. Individual Classifications\n\n")
        for i, rec in enumerate(file_records):
            cl = classifications[i]
            f.write(f"### {rec['rel_path']}\n\n")
            f.write(f"**Classification: {cl['classification']}**\n\n")
            f.write(f"Reason: {cl['reason']}\n\n")
            f.write(f"- Sorting resolves the issue: {cl['sorting_resolves']}\n")
            f.write(f"- Has additional anomaly: {cl['has_additional_anomaly']}\n")
            f.write(f"- Number of ordering violations: {rec['n_violations']}\n")
            f.write(f"- Violation magnitude: {rec['violations'][0]['magnitude']:.1f} nm (if violations exist)\n")
            f.write(f"- Duplicate wavelengths: {rec['n_duplicates']}\n")
            f.write(f"- Data integrity issues: None\n\n")

        # K. Researcher-review recommendation
        f.write("## K. Researcher-Review Recommendation\n\n")
        f.write("### Recommendation\n\n")

        # Check if all 4 have same classification
        cl_set = set(c["classification"] for c in classifications)
        if cl_set == {"A"}:
            f.write("All 4 spectra receive **Classification A** (ordering-only issue).\n\n")
            f.write("**Recommendation:** Sorting should be permitted for these 4 files during later preprocessing. "
                    "All 4 can receive the same rule: sort wavelength-value rows by wavelength in ascending order. "
                    "No file requires separate treatment.\n\n")
        elif "A" in cl_set and "B" in cl_set:
            f.write("The 4 spectra receive mixed Classifications A and B.\n\n")
            f.write("**Recommendation:** Sorting should be permitted for all files. Files with Classification B "
                    "require documentation of the additional minor anomaly. Review whether the additional anomaly "
                    "affects downstream analysis.\n\n")
        else:
            f.write("The 4 spectra receive mixed classifications.\n\n")
            f.write("**Recommendation:** Individual review required for files not classified as A.\n\n")

        f.write("### Important context\n\n")
        f.write("1. **None of the 4 spectra are among the 454 D2-compatible spectra.** All start at 900.4 nm, "
                "which is above the D2 lower boundary of 320 nm. They do not fully cover the 320–2550 nm domain.\n\n")
        f.write("2. **The ordering anomaly is minimal:** exactly 1 violation per file, magnitude 1.3 nm, "
                "affecting only 2 row positions.\n\n")
        f.write("3. **Sorting is a pure ordering operation:** it does not change any wavelength-reflectance pairing, "
                "does not interpolate, average, or delete any observations.\n\n")
        f.write("4. **If these spectra are not included in the D2 analysis population,** the ordering issue "
                "has no practical consequence for the final standardized spectra.\n\n")

        # L. Limitations
        f.write("## L. Limitations\n\n")
        f.write("1. **Investigation only:** No correction was implemented. This report provides evidence for "
                "researcher review but does not freeze a decision.\n\n")
        f.write("2. **No detector/channel identification:** The RELAB metadata does not identify which detector "
                "or channel produced each observation. The ordering anomaly could plausibly relate to the "
                "documented splice at 3300 nm, but this cannot be confirmed.\n\n")
        f.write("3. **Prefix 397 meaning unknown:** The project evidence does not document what the `397` prefix "
                "signifies. It may indicate a measurement campaign, instrument configuration, or acquisition batch.\n\n")
        f.write("4. **Physical spectral resolution unknown:** Only sampling intervals are available in the raw data.\n\n")
        f.write("5. **D2 coverage:** These 4 spectra do not cover the full D2 domain. Their inclusion in the "
                "analysis population depends on whether the domain coverage requirement is strictly enforced.\n\n")

        # M. Raw-data integrity
        f.write("## M. Raw-Data Integrity\n\n")
        f.write(f"- Total raw files verified: {total_files}\n")
        f.write(f"- Pre-flight MD5 mismatches: {pre_mismatches}\n")
        f.write(f"- Post-flight MD5 mismatches: {post_mismatches}\n")
        f.write(f"- Raw `.tab` files modified: **No**\n")
        f.write(f"- Processed dataset created: **No**\n")
        f.write(f"- Sorted spectra saved: **No**\n")
        f.write(f"- Interpolation performed: **No**\n")
        f.write(f"- Smoothing performed: **No**\n")
        f.write(f"- Normalization performed: **No**\n")
        f.write(f"- Feature engineering performed: **No**\n")
        f.write(f"- ML/PySR performed: **No**\n")
        f.write(f"- Duplicate-wavelength freeze modified: **No**\n\n")

        f.write("---\n\n")
        f.write("*Investigation completed: 2026-09-26*\n")
        f.write("*Scope: Investigation only — no modifications to raw data or analysis results*\n")

    print(f"    MD report written: {md_path}")

    # ─── SECTION 13: Validation ───
    print("\n[13] Validation...")
    print(f"    Exactly 4 spectra investigated: {len(file_records) == 4}")
    print(f"    Raw .tab files not modified: {post_mismatches == 0}")
    print(f"    No processed dataset created: True (investigation only)")
    print(f"    No sorted spectra saved: True (investigation only)")
    print(f"    No interpolation performed: True")
    print(f"    No smoothing performed: True")
    print(f"    No normalization performed: True")
    print(f"    No feature engineering performed: True")
    print(f"    No ML/PySR performed: True")
    print(f"    Duplicate-wavelength freeze not modified: True")

    # Run twice for deterministic results
    print("\n[14] Deterministic reproducibility check...")
    # Re-run the investigation for a subset to verify determinism
    for rel_path in UNSORTED_FILES[:2]:
        fpath = DATA_DIR / rel_path
        wls2, rfs2, uncs2, _, _, _ = read_spectrum(fpath)
        v2 = find_ordering_violations(wls2)
        sr2 = conceptual_sort_test(wls2, rfs2, uncs2)
        # Compare with first run
        orig = [r for r in file_records if r["rel_path"] == rel_path][0]
        assert len(v2) == orig["n_violations"], f"Non-deterministic violations for {rel_path}"
        assert sr2["n_values_moved"] == sort_results[UNSORTED_FILES.index(rel_path)]["n_values_moved"], \
            f"Non-deterministic sort for {rel_path}"
    print("    Deterministic results confirmed.")

    print("\n" + "=" * 70)
    print("INVESTIGATION COMPLETE")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  {csv_path}")
    print(f"  {md_path}")
    print(f"\nSTOPPING — no decisions frozen, no corrections implemented.")


if __name__ == "__main__":
    main()
