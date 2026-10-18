"""
Phase 3 Step 3 — Canonical Feature Matrix Construction

This script constructs the canonical spectral-complexity feature matrix using the frozen
Phase 2 canonical dataset and the frozen Phase 3 Step 2 descriptor specification.

Frozen primary descriptors (4):
  D1: n_peaks_prom          — prominent peak count (prominence >= 0.01)
  D2: mean_peak_prominence  — mean prominence of detected peaks (0.0 if zero peaks)
  D3: first_deriv_rms       — RMS of first spectral derivative
  D4: second_deriv_rms      — RMS of second spectral derivative

Frozen computational conventions:
  - Prominence threshold: 0.01 reflectance units
  - Peak detection: scipy.signal.find_peaks(distance=3, prominence=0.01)
  - Derivative method: numpy.gradient (central differences)
  - Wavelength grid: 447 points, 320-2550 nm, 5 nm spacing
  - Zero-peak convention: mean_peak_prominence = 0.0
  - No smoothing, no normalization

Input:
  data/processed/phase2_canonical_d2_5nm.csv
  data/processed/phase2_canonical_d2_5nm_metadata.csv

Output:
  data/processed/phase3_canonical_spectral_features.csv
  results/03_complexity_features/phase3_step3_feature_matrix_validation.md
  results/03_complexity_features/phase3_step3_feature_matrix_validation.csv
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import hashlib
import json
from datetime import datetime
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

INPUT_SPECTRA_PATH = "data/processed/phase2_canonical_d2_5nm.csv"
INPUT_METADATA_PATH = "data/processed/phase2_canonical_d2_5nm_metadata.csv"
OUTPUT_FEATURES_PATH = "data/processed/phase3_canonical_spectral_features.csv"
OUTPUT_VALIDATION_MD = "results/03_complexity_features/phase3_step3_feature_matrix_validation.md"
OUTPUT_VALIDATION_CSV = "results/03_complexity_features/phase3_step3_feature_matrix_validation.csv"

# Frozen parameters
PROMINENCE_THRESHOLD = 0.01
PEAK_DISTANCE = 3
WAVELENGTH_START = 320
WAVELENGTH_END = 2550
WAVELENGTH_SPACING = 5
EXPECTED_N_POINTS = 447
EXPECTED_N_SPECTRA = 454
EXPECTED_ZERO_PEAK_COUNT = 14

# ============================================================================
# Helper functions
# ============================================================================

def compute_sha256(filepath):
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def compute_descriptors(reflectance_spectrum, wavelength_array):
    """
    Compute the four frozen primary descriptors for a single spectrum.

    Parameters:
    -----------
    reflectance_spectrum : array-like
        Reflectance values (447 points)
    wavelength_array : array-like
        Wavelength values (447 points, 320-2550 nm, 5 nm spacing)

    Returns:
    --------
    dict with keys: n_peaks_prom, mean_peak_prominence, first_deriv_rms, second_deriv_rms
    """
    R = np.asarray(reflectance_spectrum, dtype=float)
    wl = np.asarray(wavelength_array, dtype=float)

    # D1: n_peaks_prom
    peaks, properties = find_peaks(R, distance=PEAK_DISTANCE, prominence=PROMINENCE_THRESHOLD)
    n_peaks_prom = len(peaks)

    # D2: mean_peak_prominence
    if n_peaks_prom > 0:
        mean_peak_prominence = np.mean(properties['prominences'])
    else:
        mean_peak_prominence = 0.0

    # D3: first_deriv_rms
    d1 = np.gradient(R, wl)
    first_deriv_rms = np.sqrt(np.mean(d1**2))

    # D4: second_deriv_rms
    d2 = np.gradient(d1, wl)
    second_deriv_rms = np.sqrt(np.mean(d2**2))

    return {
        'n_peaks_prom': int(n_peaks_prom),
        'mean_peak_prominence': float(mean_peak_prominence),
        'first_deriv_rms': float(first_deriv_rms),
        'second_deriv_rms': float(second_deriv_rms)
    }

def validate_dataset(df_spectra, df_metadata):
    """Validate dataset integrity."""
    checks = {}

    # Check row counts
    checks['n_spectra'] = len(df_spectra) == EXPECTED_N_SPECTRA
    checks['n_metadata'] = len(df_metadata) == EXPECTED_N_SPECTRA

    # Check unique spectrum IDs
    checks['unique_spectrum_ids'] = df_spectra['canonical_spectrum_id'].nunique() == EXPECTED_N_SPECTRA

    # Check wavelength columns
    wl_cols = [c for c in df_spectra.columns if c.startswith('R_') and c.endswith('nm')]
    checks['n_wavelength_columns'] = len(wl_cols) == EXPECTED_N_POINTS

    # Check wavelength range and spacing
    wl_values = []
    for col in wl_cols:
        wl_str = col.replace('R_', '').replace('nm', '')
        wl_values.append(int(wl_str))
    wl_values = sorted(wl_values)

    checks['wl_min'] = wl_values[0] == WAVELENGTH_START
    checks['wl_max'] = wl_values[-1] == WAVELENGTH_END
    checks['wl_spacing'] = all(wl_values[i+1] - wl_values[i] == WAVELENGTH_SPACING for i in range(len(wl_values)-1))

    # Check representation types
    checks['native_5nm_count'] = (df_metadata['representation_type'] == 'native_5nm').sum() == 390
    checks['pchip_count'] = (df_metadata['representation_type'] == 'pchip_from_native_10nm').sum() == 64

    # Check mineral count
    checks['mineral_count'] = df_metadata['mineral'].nunique() == 18

    # Check provenance group count
    checks['provenance_group_count'] = df_metadata['specimen_group_id'].nunique() == 382

    return checks

def validate_features(df_features):
    """Validate feature integrity."""
    checks = {}

    # Check row count
    checks['n_rows'] = len(df_features) == EXPECTED_N_SPECTRA

    # Check unique spectrum IDs
    checks['unique_spectrum_ids'] = df_features['canonical_spectrum_id'].nunique() == EXPECTED_N_SPECTRA

    # Check no NaN/Inf
    feature_cols = ['n_peaks_prom', 'mean_peak_prominence', 'first_deriv_rms', 'second_deriv_rms']
    checks['no_nan'] = all(not df_features[col].isna().any() for col in feature_cols)
    checks['no_inf'] = all(not np.isinf(df_features[col]).any() for col in feature_cols)

    # Check non-negativity
    checks['n_peaks_prom_nonneg'] = (df_features['n_peaks_prom'] >= 0).all()
    checks['mean_peak_prominence_nonneg'] = (df_features['mean_peak_prominence'] >= 0).all()
    checks['first_deriv_rms_nonneg'] = (df_features['first_deriv_rms'] >= 0).all()
    checks['second_deriv_rms_nonneg'] = (df_features['second_deriv_rms'] >= 0).all()

    # Check integer-valued n_peaks_prom
    checks['n_peaks_prom_integer'] = (df_features['n_peaks_prom'] == df_features['n_peaks_prom'].round()).all()

    # Check zero-peak count
    zero_peak_mask = df_features['n_peaks_prom'] == 0
    checks['zero_peak_count'] = zero_peak_mask.sum() == EXPECTED_ZERO_PEAK_COUNT

    # Check zero-peak convention
    checks['zero_peak_mean_prominence'] = (df_features.loc[zero_peak_mask, 'mean_peak_prominence'] == 0.0).all()

    return checks

def compute_feature_statistics(df_features):
    """Compute descriptive statistics for each descriptor."""
    feature_cols = ['n_peaks_prom', 'mean_peak_prominence', 'first_deriv_rms', 'second_deriv_rms']
    stats = {}

    for col in feature_cols:
        stats[col] = {
            'min': float(df_features[col].min()),
            'max': float(df_features[col].max()),
            'median': float(df_features[col].median()),
            'mean': float(df_features[col].mean()),
            'std': float(df_features[col].std()),
            'missing': int(df_features[col].isna().sum()),
            'invalid': int(np.isinf(df_features[col]).sum())
        }

    # Zero-peak count
    stats['zero_peak_count'] = int((df_features['n_peaks_prom'] == 0).sum())

    return stats

# ============================================================================
# Main execution
# ============================================================================

def main():
    print("=" * 80)
    print("Phase 3 Step 3 — Canonical Feature Matrix Construction")
    print("=" * 80)
    print()

    # Record start time
    start_time = datetime.now()

    # =========================================================================
    # Load data
    # =========================================================================

    print("Loading Phase 2 canonical dataset...")
    df_spectra = pd.read_csv(INPUT_SPECTRA_PATH)
    df_metadata = pd.read_csv(INPUT_METADATA_PATH)

    print(f"  Spectra: {len(df_spectra)} rows")
    print(f"  Metadata: {len(df_metadata)} rows")
    print()

    # Compute input checksum
    input_checksum = compute_sha256(INPUT_SPECTRA_PATH)
    print(f"Input checksum (SHA-256): {input_checksum}")
    print()

    # =========================================================================
    # Validate dataset
    # =========================================================================

    print("Validating dataset integrity...")
    dataset_checks = validate_dataset(df_spectra, df_metadata)

    all_dataset_checks_pass = all(dataset_checks.values())
    for check_name, check_result in dataset_checks.items():
        status = "PASS" if check_result else "FAIL"
        print(f"  {check_name}: {status}")

    if not all_dataset_checks_pass:
        print("\nERROR: Dataset validation failed. Aborting.")
        return

    print()

    # =========================================================================
    # Extract wavelength array
    # =========================================================================

    wl_cols = [c for c in df_spectra.columns if c.startswith('R_') and c.endswith('nm')]
    wl_cols_sorted = sorted(wl_cols, key=lambda x: int(x.replace('R_', '').replace('nm', '')))
    wavelength_array = np.array([int(c.replace('R_', '').replace('nm', '')) for c in wl_cols_sorted])

    print(f"Wavelength array: {wavelength_array[0]} to {wavelength_array[-1]} nm, {len(wavelength_array)} points")
    print()

    # =========================================================================
    # Compute descriptors (Run 1)
    # =========================================================================

    print("Computing descriptors (Run 1)...")
    features_run1 = []

    for idx, row in df_spectra.iterrows():
        spectrum_id = row['canonical_spectrum_id']
        reflectance_values = row[wl_cols_sorted].values

        descriptors = compute_descriptors(reflectance_values, wavelength_array)
        descriptors['canonical_spectrum_id'] = spectrum_id

        features_run1.append(descriptors)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df_spectra)} spectra")

    df_features_run1 = pd.DataFrame(features_run1)

    # Reorder columns
    df_features_run1 = df_features_run1[['canonical_spectrum_id', 'n_peaks_prom', 'mean_peak_prominence',
                                         'first_deriv_rms', 'second_deriv_rms']]

    print(f"  Computed descriptors for {len(df_features_run1)} spectra")
    print()

    # =========================================================================
    # Validate features (Run 1)
    # =========================================================================

    print("Validating features (Run 1)...")
    feature_checks_run1 = validate_features(df_features_run1)

    all_feature_checks_run1_pass = all(feature_checks_run1.values())
    for check_name, check_result in feature_checks_run1.items():
        status = "PASS" if check_result else "FAIL"
        print(f"  {check_name}: {status}")

    if not all_feature_checks_run1_pass:
        print("\nERROR: Feature validation failed. Aborting.")
        return

    print()

    # =========================================================================
    # Compute descriptors (Run 2 — reproducibility)
    # =========================================================================

    print("Computing descriptors (Run 2 — reproducibility)...")
    features_run2 = []

    for idx, row in df_spectra.iterrows():
        spectrum_id = row['canonical_spectrum_id']
        reflectance_values = row[wl_cols_sorted].values

        descriptors = compute_descriptors(reflectance_values, wavelength_array)
        descriptors['canonical_spectrum_id'] = spectrum_id

        features_run2.append(descriptors)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df_spectra)} spectra")

    df_features_run2 = pd.DataFrame(features_run2)
    df_features_run2 = df_features_run2[['canonical_spectrum_id', 'n_peaks_prom', 'mean_peak_prominence',
                                         'first_deriv_rms', 'second_deriv_rms']]

    print(f"  Computed descriptors for {len(df_features_run2)} spectra")
    print()

    # =========================================================================
    # Reproducibility check
    # =========================================================================

    print("Checking reproducibility...")

    # Check identical row order
    identical_order = (df_features_run1['canonical_spectrum_id'] == df_features_run2['canonical_spectrum_id']).all()
    print(f"  Identical row order: {'PASS' if identical_order else 'FAIL'}")

    # Check identical feature values
    feature_cols = ['n_peaks_prom', 'mean_peak_prominence', 'first_deriv_rms', 'second_deriv_rms']
    identical_values = all(
        np.allclose(df_features_run1[col], df_features_run2[col], rtol=1e-10, atol=1e-10)
        for col in feature_cols
    )
    print(f"  Identical feature values: {'PASS' if identical_values else 'FAIL'}")

    reproducibility_pass = identical_order and identical_values

    if not reproducibility_pass:
        print("\nERROR: Reproducibility check failed. Aborting.")
        return

    print()

    # =========================================================================
    # Merge features with metadata
    # =========================================================================

    print("Merging features with metadata...")
    df_features_final = df_features_run1.copy()

    # Merge with metadata
    df_features_final = df_features_final.merge(
        df_metadata[['canonical_spectrum_id', 'raw_filename', 'mineral', 'specimen_group_id', 'representation_type']],
        on='canonical_spectrum_id',
        how='left'
    )

    # Reorder columns
    df_features_final = df_features_final[[
        'canonical_spectrum_id', 'raw_filename', 'mineral', 'specimen_group_id', 'representation_type',
        'n_peaks_prom', 'mean_peak_prominence', 'first_deriv_rms', 'second_deriv_rms'
    ]]

    print(f"  Final feature matrix: {len(df_features_final)} rows, {len(df_features_final.columns)} columns")
    print()

    # =========================================================================
    # Compute feature statistics
    # =========================================================================

    print("Computing feature statistics...")
    feature_stats = compute_feature_statistics(df_features_final)

    for col in ['n_peaks_prom', 'mean_peak_prominence', 'first_deriv_rms', 'second_deriv_rms']:
        print(f"  {col}:")
        print(f"    min: {feature_stats[col]['min']:.6g}")
        print(f"    max: {feature_stats[col]['max']:.6g}")
        print(f"    median: {feature_stats[col]['median']:.6g}")
        print(f"    mean: {feature_stats[col]['mean']:.6g}")
        print(f"    std: {feature_stats[col]['std']:.6g}")

    print(f"  Zero-peak count: {feature_stats['zero_peak_count']}")
    print()

    # =========================================================================
    # Write output files
    # =========================================================================

    print("Writing output files...")

    # Write feature matrix
    df_features_final.to_csv(OUTPUT_FEATURES_PATH, index=False)
    print(f"  Feature matrix: {OUTPUT_FEATURES_PATH}")

    # Compute output checksum
    output_checksum = compute_sha256(OUTPUT_FEATURES_PATH)
    print(f"  Output checksum (SHA-256): {output_checksum}")

    # =========================================================================
    # Write validation report (Markdown)
    # =========================================================================

    validation_report_md = f"""# Phase 3 Step 3 — Canonical Feature Matrix Construction — Validation Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Status:** {'COMPLETE / FROZEN' if all_dataset_checks_pass and all_feature_checks_run1_pass and reproducibility_pass else 'NOT FROZEN — issue(s) requiring review'}

---

## 1. Dataset Integrity Validation

| Check | Result |
|-------|--------|
| Number of spectra = {EXPECTED_N_SPECTRA} | {'PASS' if dataset_checks['n_spectra'] else 'FAIL'} |
| Number of metadata rows = {EXPECTED_N_SPECTRA} | {'PASS' if dataset_checks['n_metadata'] else 'FAIL'} |
| Unique spectrum IDs = {EXPECTED_N_SPECTRA} | {'PASS' if dataset_checks['unique_spectrum_ids'] else 'FAIL'} |
| Number of wavelength columns = {EXPECTED_N_POINTS} | {'PASS' if dataset_checks['n_wavelength_columns'] else 'FAIL'} |
| Wavelength min = {WAVELENGTH_START} nm | {'PASS' if dataset_checks['wl_min'] else 'FAIL'} |
| Wavelength max = {WAVELENGTH_END} nm | {'PASS' if dataset_checks['wl_max'] else 'FAIL'} |
| Wavelength spacing = {WAVELENGTH_SPACING} nm | {'PASS' if dataset_checks['wl_spacing'] else 'FAIL'} |
| Native-5nm count = 390 | {'PASS' if dataset_checks['native_5nm_count'] else 'FAIL'} |
| PCHIP count = 64 | {'PASS' if dataset_checks['pchip_count'] else 'FAIL'} |
| Mineral count = 18 | {'PASS' if dataset_checks['mineral_count'] else 'FAIL'} |
| Provenance group count = 382 | {'PASS' if dataset_checks['provenance_group_count'] else 'FAIL'} |

**Dataset integrity:** {'PASS' if all_dataset_checks_pass else 'FAIL'}

---

## 2. Feature Integrity Validation

| Check | Result |
|-------|--------|
| Number of feature rows = {EXPECTED_N_SPECTRA} | {'PASS' if feature_checks_run1['n_rows'] else 'FAIL'} |
| Unique spectrum IDs = {EXPECTED_N_SPECTRA} | {'PASS' if feature_checks_run1['unique_spectrum_ids'] else 'FAIL'} |
| No NaN values | {'PASS' if feature_checks_run1['no_nan'] else 'FAIL'} |
| No infinite values | {'PASS' if feature_checks_run1['no_inf'] else 'FAIL'} |
| n_peaks_prom >= 0 | {'PASS' if feature_checks_run1['n_peaks_prom_nonneg'] else 'FAIL'} |
| mean_peak_prominence >= 0 | {'PASS' if feature_checks_run1['mean_peak_prominence_nonneg'] else 'FAIL'} |
| first_deriv_rms >= 0 | {'PASS' if feature_checks_run1['first_deriv_rms_nonneg'] else 'FAIL'} |
| second_deriv_rms >= 0 | {'PASS' if feature_checks_run1['second_deriv_rms_nonneg'] else 'FAIL'} |
| n_peaks_prom integer-valued | {'PASS' if feature_checks_run1['n_peaks_prom_integer'] else 'FAIL'} |
| Zero-peak count = {EXPECTED_ZERO_PEAK_COUNT} | {'PASS' if feature_checks_run1['zero_peak_count'] else 'FAIL'} |
| Zero-peak mean_peak_prominence = 0.0 | {'PASS' if feature_checks_run1['zero_peak_mean_prominence'] else 'FAIL'} |

**Feature integrity:** {'PASS' if all_feature_checks_run1_pass else 'FAIL'}

---

## 3. Reproducibility Validation

| Check | Result |
|-------|--------|
| Identical row order (Run 1 vs Run 2) | {'PASS' if identical_order else 'FAIL'} |
| Identical feature values (Run 1 vs Run 2) | {'PASS' if identical_values else 'FAIL'} |

**Reproducibility:** {'PASS' if reproducibility_pass else 'FAIL'}

---

## 4. Feature Statistics

### D1 — n_peaks_prom (prominent peak count)

| Statistic | Value |
|-----------|-------|
| Min | {feature_stats['n_peaks_prom']['min']} |
| Max | {feature_stats['n_peaks_prom']['max']} |
| Median | {feature_stats['n_peaks_prom']['median']:.1f} |
| Mean | {feature_stats['n_peaks_prom']['mean']:.3f} |
| Std | {feature_stats['n_peaks_prom']['std']:.3f} |
| Missing | {feature_stats['n_peaks_prom']['missing']} |
| Invalid | {feature_stats['n_peaks_prom']['invalid']} |

### D2 — mean_peak_prominence (mean prominent peak prominence)

| Statistic | Value |
|-----------|-------|
| Min | {feature_stats['mean_peak_prominence']['min']:.6f} |
| Max | {feature_stats['mean_peak_prominence']['max']:.6f} |
| Median | {feature_stats['mean_peak_prominence']['median']:.6f} |
| Mean | {feature_stats['mean_peak_prominence']['mean']:.6f} |
| Std | {feature_stats['mean_peak_prominence']['std']:.6f} |
| Missing | {feature_stats['mean_peak_prominence']['missing']} |
| Invalid | {feature_stats['mean_peak_prominence']['invalid']} |

### D3 — first_deriv_rms (first-derivative RMS)

| Statistic | Value |
|-----------|-------|
| Min | {feature_stats['first_deriv_rms']['min']:.6g} |
| Max | {feature_stats['first_deriv_rms']['max']:.6g} |
| Median | {feature_stats['first_deriv_rms']['median']:.6g} |
| Mean | {feature_stats['first_deriv_rms']['mean']:.6g} |
| Std | {feature_stats['first_deriv_rms']['std']:.6g} |
| Missing | {feature_stats['first_deriv_rms']['missing']} |
| Invalid | {feature_stats['first_deriv_rms']['invalid']} |

### D4 — second_deriv_rms (second-derivative RMS)

| Statistic | Value |
|-----------|-------|
| Min | {feature_stats['second_deriv_rms']['min']:.6g} |
| Max | {feature_stats['second_deriv_rms']['max']:.6g} |
| Median | {feature_stats['second_deriv_rms']['median']:.6g} |
| Mean | {feature_stats['second_deriv_rms']['mean']:.6g} |
| Std | {feature_stats['second_deriv_rms']['std']:.6g} |
| Missing | {feature_stats['second_deriv_rms']['missing']} |
| Invalid | {feature_stats['second_deriv_rms']['invalid']} |

### Zero-peak spectra

| Statistic | Value |
|-----------|-------|
| Zero-peak count | {feature_stats['zero_peak_count']} |
| Expected zero-peak count | {EXPECTED_ZERO_PEAK_COUNT} |
| All zero-peak spectra have mean_peak_prominence = 0.0 | {'YES' if feature_checks_run1['zero_peak_mean_prominence'] else 'NO'} |

---

## 5. Checksums

| File | SHA-256 |
|------|---------|
| Input (Phase 2 canonical spectra) | {input_checksum} |
| Output (Phase 3 canonical features) | {output_checksum} |

---

## 6. Frozen Descriptor Specification

### Primary descriptors (4)

1. **D1 — n_peaks_prom**: Number of peaks with prominence >= 0.01 reflectance units
2. **D2 — mean_peak_prominence**: Mean prominence of detected peaks (0.0 if zero peaks)
3. **D3 — first_deriv_rms**: RMS of first spectral derivative (numpy.gradient)
4. **D4 — second_deriv_rms**: RMS of second spectral derivative (numpy.gradient applied twice)

### Frozen computational conventions

- Prominence threshold: {PROMINENCE_THRESHOLD} reflectance units
- Peak detection algorithm: scipy.signal.find_peaks
- Minimum peak distance: {PEAK_DISTANCE} points ({PEAK_DISTANCE * WAVELENGTH_SPACING} nm)
- Derivative method: numpy.gradient (central differences)
- Wavelength grid: {EXPECTED_N_POINTS} points, {WAVELENGTH_START}-{WAVELENGTH_END} nm, {WAVELENGTH_SPACING} nm spacing
- Zero-peak convention: mean_peak_prominence = 0.0
- Smoothing: None
- Normalization: None

### Sensitivity descriptor (NOT included in primary matrix)

- **S1 — total_variation**: Sum of absolute adjacent differences (retained for sensitivity analysis only)

---

## 7. Final Status

**Phase 3 Step 3 — Canonical Feature Matrix Construction:** {'COMPLETE / FROZEN' if all_dataset_checks_pass and all_feature_checks_run1_pass and reproducibility_pass else 'NOT FROZEN — issue(s) requiring review'}

**Date frozen:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Output file:** {OUTPUT_FEATURES_PATH}

**Rows:** {len(df_features_final)}

**Columns:** {len(df_features_final.columns)}

**Primary feature columns:** n_peaks_prom, mean_peak_prominence, first_deriv_rms, second_deriv_rms

**Metadata columns:** canonical_spectrum_id, raw_filename, mineral, specimen_group_id, representation_type

---

*End of validation report.*
"""

    with open(OUTPUT_VALIDATION_MD, 'w', encoding='utf-8') as f:
        f.write(validation_report_md)

    print(f"  Validation report (Markdown): {OUTPUT_VALIDATION_MD}")

    # =========================================================================
    # Write validation report (CSV)
    # =========================================================================

    validation_report_csv_data = []

    # Dataset integrity checks
    for check_name, check_result in dataset_checks.items():
        validation_report_csv_data.append({
            'category': 'dataset_integrity',
            'check': check_name,
            'result': 'PASS' if check_result else 'FAIL'
        })

    # Feature integrity checks
    for check_name, check_result in feature_checks_run1.items():
        validation_report_csv_data.append({
            'category': 'feature_integrity',
            'check': check_name,
            'result': 'PASS' if check_result else 'FAIL'
        })

    # Reproducibility checks
    validation_report_csv_data.append({
        'category': 'reproducibility',
        'check': 'identical_row_order',
        'result': 'PASS' if identical_order else 'FAIL'
    })
    validation_report_csv_data.append({
        'category': 'reproducibility',
        'check': 'identical_feature_values',
        'result': 'PASS' if identical_values else 'FAIL'
    })

    df_validation = pd.DataFrame(validation_report_csv_data)
    df_validation.to_csv(OUTPUT_VALIDATION_CSV, index=False)

    print(f"  Validation report (CSV): {OUTPUT_VALIDATION_CSV}")
    print()

    # =========================================================================
    # Final summary
    # =========================================================================

    end_time = datetime.now()
    duration = end_time - start_time

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Dataset integrity: {'PASS' if all_dataset_checks_pass else 'FAIL'}")
    print(f"Feature integrity: {'PASS' if all_feature_checks_run1_pass else 'FAIL'}")
    print(f"Reproducibility: {'PASS' if reproducibility_pass else 'FAIL'}")
    print()
    print(f"Rows: {len(df_features_final)}")
    print(f"Columns: {len(df_features_final.columns)}")
    print(f"Primary feature columns: n_peaks_prom, mean_peak_prominence, first_deriv_rms, second_deriv_rms")
    print(f"Metadata columns: canonical_spectrum_id, raw_filename, mineral, specimen_group_id, representation_type")
    print()
    print(f"Zero-peak count: {feature_stats['zero_peak_count']}")
    print(f"All zero-peak spectra have mean_peak_prominence = 0.0: {'YES' if feature_checks_run1['zero_peak_mean_prominence'] else 'NO'}")
    print()
    print(f"Duration: {duration}")
    print()

    if all_dataset_checks_pass and all_feature_checks_run1_pass and reproducibility_pass:
        print("Phase 3 Step 3 — COMPLETE / FROZEN")
    else:
        print("Phase 3 Step 3 — NOT FROZEN — issue(s) requiring review")

    print("=" * 80)

if __name__ == "__main__":
    main()
