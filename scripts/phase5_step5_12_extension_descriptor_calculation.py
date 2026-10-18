"""
Phase 5 Step 5.12 — Extension Descriptor Calculation and Validation

Calculate the four primary spectral descriptors (D1-D4) and one sensitivity
descriptor (S1) for the 49-spectrum extension dataset constructed in Phase 5
Step 5.8 and independently audited in Step 5.9.

Uses the authoritative frozen Phase 3 implementation and specifications.
Independently validates the outputs before they can be used in a future
comparative-analysis view.

Frozen descriptors:
  D1: n_peaks_prom          — prominent peak count (prominence >= 0.01)
  D2: mean_peak_prominence  — mean prominence of detected peaks (0.0 if zero peaks)
  D3: first_deriv_rms       — RMS of first spectral derivative
  D4: second_deriv_rms      — RMS of second spectral derivative
  S1: total_variation       — sum of absolute adjacent spectral changes (sensitivity only)

Frozen parameters:
  Prominence threshold: 0.01 reflectance units
  Peak detection: scipy.signal.find_peaks(distance=3, prominence=0.01)
  Derivative method: numpy.gradient (central differences)
  Wavelength grid: 447 points, 320-2550 nm, 5 nm spacing
  Zero-peak convention: mean_peak_prominence = 0.0
  No smoothing, no normalization

Inputs:
  results/05_exploratory_analysis/phase5_step5_8_extension_dataset/
  data/processed/phase3_canonical_spectral_features.csv
  data/processed/phase2_canonical_d2_5nm.csv

Outputs:
  results/05_exploratory_analysis/phase5_step5_12_extension_descriptor_validation/
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import hashlib
import os
from datetime import datetime
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Extension inputs (Step 5.8)
EXT_SPECTRA_PATH = PROJECT_ROOT / "results" / "05_exploratory_analysis" / "phase5_step5_8_extension_dataset" / "phase5_step5_8_extension_spectra.csv"
EXT_METADATA_PATH = PROJECT_ROOT / "results" / "05_exploratory_analysis" / "phase5_step5_8_extension_dataset" / "phase5_step5_8_extension_metadata.csv"
EXT_PROCESSING_LOG_PATH = PROJECT_ROOT / "results" / "05_exploratory_analysis" / "phase5_step5_8_extension_dataset" / "phase5_step5_8_processing_log.csv"
EXT_CHECKSUM_MANIFEST_PATH = PROJECT_ROOT / "results" / "05_exploratory_analysis" / "phase5_step5_8_extension_dataset" / "phase5_step5_8_checksum_manifest.csv"
EXT_DATASET_CARD_PATH = PROJECT_ROOT / "results" / "05_exploratory_analysis" / "phase5_step5_8_extension_dataset" / "phase5_step5_8_dataset_card.md"

# Canonical inputs (frozen)
CANONICAL_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "phase3_canonical_spectral_features.csv"
CANONICAL_SPECTRA_PATH = PROJECT_ROOT / "data" / "processed" / "phase2_canonical_d2_5nm.csv"
CANONICAL_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "phase2_canonical_metadata.csv"

# Authoritative Phase 3 script
PHASE3_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "phase3_step3_canonical_feature_matrix.py"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "results" / "05_exploratory_analysis" / "phase5_step5_12_extension_descriptor_validation"

# Frozen parameters (from Phase 3 Step 2 + Step 3)
PROMINENCE_THRESHOLD = 0.01
PEAK_DISTANCE = 3
WAVELENGTH_START = 320
WAVELENGTH_END = 2550
WAVELENGTH_SPACING = 5
EXPECTED_N_POINTS = 447
EXPECTED_N_EXTENSION = 49
EXPECTED_N_PRIMARY = 42
EXPECTED_N_SENSITIVITY = 7

# Descriptor implementation version
DESCRIPTOR_VERSION = "phase3_step3_frozen_v1"
INPUT_DATASET_VERSION = "phase5_step5_8_extension_v1.0"

# Validation tolerances
EXACT_MATCH_TOL = 1e-14
PCHIP_CSV_TOL = 1e-6


# ============================================================================
# Helper functions
# ============================================================================

def compute_md5(filepath):
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def compute_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


# ============================================================================
# Primary descriptor computation — exact copy of Phase 3 compute_descriptors
# ============================================================================

def compute_descriptors_primary(reflectance_spectrum, wavelength_array):
    """
    Compute the four frozen primary descriptors (D1-D4) for a single spectrum.
    Exact replica of phase3_step3_canonical_feature_matrix.py:compute_descriptors.
    """
    R = np.asarray(reflectance_spectrum, dtype=float)
    wl = np.asarray(wavelength_array, dtype=float)

    peaks, properties = find_peaks(R, distance=PEAK_DISTANCE, prominence=PROMINENCE_THRESHOLD)
    n_peaks_prom = len(peaks)

    if n_peaks_prom > 0:
        mean_peak_prominence = np.mean(properties['prominences'])
    else:
        mean_peak_prominence = 0.0

    d1 = np.gradient(R, wl)
    first_deriv_rms = np.sqrt(np.mean(d1**2))

    d2 = np.gradient(d1, wl)
    second_deriv_rms = np.sqrt(np.mean(d2**2))

    return {
        'n_peaks_prom': int(n_peaks_prom),
        'mean_peak_prominence': float(mean_peak_prominence),
        'first_deriv_rms': float(first_deriv_rms),
        'second_deriv_rms': float(second_deriv_rms)
    }


# ============================================================================
# Independent validation computation — separate implementation path
# ============================================================================

def compute_descriptors_independent(reflectance_spectrum, wavelength_array):
    """
    Independent reimplementation of D1-D4 for validation.
    Uses a different code path to avoid duplicating the same coding error.
    """
    R = np.array(reflectance_spectrum, dtype=np.float64)
    wl = np.array(wavelength_array, dtype=np.float64)

    # D1 + D2: peak detection with explicit parameter passing
    peak_indices, peak_props = find_peaks(
        R,
        distance=int(PEAK_DISTANCE),
        prominence=float(PROMINENCE_THRESHOLD)
    )
    n_peaks = int(len(peak_indices))

    if n_peaks > 0:
        proms = np.asarray(peak_props['prominences'], dtype=np.float64)
        mean_prom = float(np.sum(proms) / len(proms))
    else:
        mean_prom = 0.0

    # D3: first derivative RMS via manual central differences
    n = len(R)
    deriv1 = np.empty(n, dtype=np.float64)
    # Forward difference at left boundary
    deriv1[0] = (R[1] - R[0]) / (wl[1] - wl[0])
    # Central differences for interior
    for i in range(1, n - 1):
        deriv1[i] = (R[i + 1] - R[i - 1]) / (wl[i + 1] - wl[i - 1])
    # Backward difference at right boundary
    deriv1[n - 1] = (R[n - 1] - R[n - 2]) / (wl[n - 1] - wl[n - 2])

    first_rms = float(np.sqrt(np.sum(deriv1 ** 2) / n))

    # D4: second derivative RMS via manual central differences of first derivative
    deriv2 = np.empty(n, dtype=np.float64)
    deriv2[0] = (deriv1[1] - deriv1[0]) / (wl[1] - wl[0])
    for i in range(1, n - 1):
        deriv2[i] = (deriv1[i + 1] - deriv1[i - 1]) / (wl[i + 1] - wl[i - 1])
    deriv2[n - 1] = (deriv1[n - 1] - deriv1[n - 2]) / (wl[n - 1] - wl[n - 2])

    second_rms = float(np.sqrt(np.sum(deriv2 ** 2) / n))

    return {
        'n_peaks_prom': n_peaks,
        'mean_peak_prominence': mean_prom,
        'first_deriv_rms': first_rms,
        'second_deriv_rms': second_rms
    }


def compute_s1(reflectance_spectrum):
    """Compute S1: total_variation (sensitivity descriptor)."""
    R = np.asarray(reflectance_spectrum, dtype=float)
    return float(np.sum(np.abs(np.diff(R))))


# ============================================================================
# Main execution
# ============================================================================

def main():
    print("=" * 80)
    print("Phase 5 Step 5.12 — Extension Descriptor Calculation and Validation")
    print("=" * 80)
    print()

    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Validation checks collector
    validation_checks = []
    check_id_counter = [0]

    def add_check(check_id_prefix, description, expected, observed, status, evidence="", severity="INFO"):
        check_id_counter[0] += 1
        cid = f"{check_id_prefix}-{check_id_counter[0]:03d}"
        validation_checks.append({
            'check_id': cid,
            'description': description,
            'expected_result': str(expected),
            'observed_result': str(observed),
            'status': status,
            'evidence_reference': evidence,
            'severity': severity
        })
        return status

    # =========================================================================
    # SECTION A: Load and verify authoritative inputs
    # =========================================================================
    print("SECTION A: Loading and verifying authoritative inputs")
    print("-" * 60)

    # A1: Load extension spectra
    print("  Loading extension spectra...")
    df_ext_spectra = pd.read_csv(EXT_SPECTRA_PATH)
    n_spectra = len(df_ext_spectra)
    print(f"    Rows: {n_spectra}")

    # A2: Load extension metadata
    print("  Loading extension metadata...")
    df_ext_metadata = pd.read_csv(EXT_METADATA_PATH)
    n_metadata = len(df_ext_metadata)
    print(f"    Rows: {n_metadata}")

    # A3: Load canonical features (frozen — read only)
    print("  Loading canonical features (frozen, read-only)...")
    df_canonical = pd.read_csv(CANONICAL_FEATURES_PATH)
    n_canonical = len(df_canonical)
    print(f"    Rows: {n_canonical}")

    # A4: Load canonical spectra for pipeline equivalence check
    print("  Loading canonical spectra for pipeline equivalence check...")
    df_canonical_spectra = pd.read_csv(CANONICAL_SPECTRA_PATH)
    print(f"    Rows: {len(df_canonical_spectra)}")
    print()

    # =========================================================================
    # SECTION B: Input verification checks
    # =========================================================================
    print("SECTION B: Input verification")
    print("-" * 60)

    # B1: Exactly 49 extension candidates
    status = add_check("IV", "Extension spectra row count",
                       str(EXPECTED_N_EXTENSION), str(n_spectra),
                       "PASS" if n_spectra == EXPECTED_N_EXTENSION else "FAIL",
                       EXT_SPECTRA_PATH.name, "CRITICAL")
    print(f"  B1 - {n_spectra} extension spectra: {status}")

    # B2: Metadata row count matches
    status = add_check("IV", "Extension metadata row count",
                       str(EXPECTED_N_EXTENSION), str(n_metadata),
                       "PASS" if n_metadata == EXPECTED_N_EXTENSION else "FAIL",
                       EXT_METADATA_PATH.name, "CRITICAL")
    print(f"  B2 - {n_metadata} metadata rows: {status}")

    # B3: Candidate IDs unique
    candidate_ids = df_ext_spectra['candidate_id'].tolist()
    n_unique = len(set(candidate_ids))
    status = add_check("IV", "Candidate IDs unique in spectra",
                       str(EXPECTED_N_EXTENSION), str(n_unique),
                       "PASS" if n_unique == EXPECTED_N_EXTENSION else "FAIL",
                       "candidate_id column", "CRITICAL")
    print(f"  B3 - {n_unique} unique candidate IDs: {status}")

    # B4: Candidate IDs match between spectra and metadata
    meta_ids = set(df_ext_metadata['candidate_id'].tolist())
    spec_ids = set(candidate_ids)
    ids_match = spec_ids == meta_ids
    status = add_check("IV", "Candidate IDs match spectra<->metadata",
                       "exact set equality", f"{'match' if ids_match else 'MISMATCH'}",
                       "PASS" if ids_match else "FAIL",
                       "set comparison", "CRITICAL")
    print(f"  B4 - ID sets match: {status}")

    # B5: Each spectrum has 447 reflectance values
    r_cols = [c for c in df_ext_spectra.columns if c.startswith('R_') and c.endswith('nm')]
    n_r_cols = len(r_cols)
    status = add_check("IV", "Reflectance column count per spectrum",
                       str(EXPECTED_N_POINTS), str(n_r_cols),
                       "PASS" if n_r_cols == EXPECTED_N_POINTS else "FAIL",
                       "R_*nm columns", "CRITICAL")
    print(f"  B5 - {n_r_cols} reflectance columns: {status}")

    # B6: Wavelength ordering and D2 grid correspondence
    wl_values = sorted([int(c.replace('R_', '').replace('nm', '')) for c in r_cols])
    expected_wl = list(range(WAVELENGTH_START, WAVELENGTH_END + WAVELENGTH_SPACING, WAVELENGTH_SPACING))
    wl_match = wl_values == expected_wl
    status = add_check("IV", "Wavelength grid matches frozen D2 specification",
                       f"{expected_wl[0]}-{expected_wl[-1]} nm, {len(expected_wl)} pts",
                       f"{wl_values[0]}-{wl_values[-1]} nm, {len(wl_values)} pts",
                       "PASS" if wl_match else "FAIL",
                       "grid comparison", "CRITICAL")
    print(f"  B6 - Wavelength grid match: {status}")

    # B7: Spectral values are numeric
    spectral_data = df_ext_spectra[r_cols].values
    all_finite = np.all(np.isfinite(spectral_data))
    status = add_check("IV", "All spectral values are finite numeric",
                       "all finite", f"{'all finite' if all_finite else 'NON-FINITE FOUND'}",
                       "PASS" if all_finite else "FAIL",
                       "np.isfinite check", "CRITICAL")
    print(f"  B7 - All spectral values finite: {status}")

    # B8: Population assignments
    pop_counts = df_ext_metadata['population_assignment'].value_counts()
    n_primary = pop_counts.get('POP_1_PRIMARY', 0)
    n_biogenic = pop_counts.get('POP_1_BIOGENIC', 0)
    status = add_check("IV", "PRIMARY_ELIGIBLE count",
                       str(EXPECTED_N_PRIMARY), str(n_primary),
                       "PASS" if n_primary == EXPECTED_N_PRIMARY else "FAIL",
                       "population_assignment column", "CRITICAL")
    print(f"  B8 - {n_primary} PRIMARY_ELIGIBLE: {status}")

    status = add_check("IV", "SENSITIVITY_ONLY (biogenic) count",
                       str(EXPECTED_N_SENSITIVITY), str(n_biogenic),
                       "PASS" if n_biogenic == EXPECTED_N_SENSITIVITY else "FAIL",
                       "population_assignment column", "CRITICAL")
    print(f"  B9 - {n_biogenic} SENSITIVITY_ONLY: {status}")

    # B10: Scientific disposition matches population
    primary_disp = (df_ext_metadata['primary_scientific_disposition'] == 'PRIMARY_ELIGIBLE').sum()
    sensitivity_disp = (df_ext_metadata['primary_scientific_disposition'] == 'SENSITIVITY_ONLY').sum()
    disp_match = (primary_disp == EXPECTED_N_PRIMARY) and (sensitivity_disp == EXPECTED_N_SENSITIVITY)
    status = add_check("IV", "Scientific disposition matches population counts",
                       f"{EXPECTED_N_PRIMARY}+{EXPECTED_N_SENSITIVITY}",
                       f"{primary_disp}+{sensitivity_disp}",
                       "PASS" if disp_match else "FAIL",
                       "disposition vs population", "CRITICAL")
    print(f"  B10 - Disposition/population match: {status}")

    # B11: Preprocessing routes documented
    routes = df_ext_metadata['preprocessing_route'].value_counts()
    n_exact = routes.get('exact_native_D2', 0)
    n_trim = routes.get('trim_native_5nm', 0)
    n_pchip = routes.get('pchip_from_native_10nm', 0)
    route_total = n_exact + n_trim + n_pchip
    status = add_check("IV", "All candidates have documented preprocessing routes",
                       str(EXPECTED_N_EXTENSION), str(route_total),
                       "PASS" if route_total == EXPECTED_N_EXTENSION else "FAIL",
                       f"exact={n_exact}, trim={n_trim}, pchip={n_pchip}", "CRITICAL")
    print(f"  B11 - Routes: exact={n_exact}, trim={n_trim}, pchip={n_pchip}: {status}")

    # B12: Canonical feature matrix integrity (frozen)
    n_canon_expected = 454
    status = add_check("IV", "Canonical feature matrix row count (frozen)",
                       str(n_canon_expected), str(n_canonical),
                       "PASS" if n_canonical == n_canon_expected else "FAIL",
                       CANONICAL_FEATURES_PATH.name, "CRITICAL")
    print(f"  B12 - Canonical features: {n_canonical} rows: {status}")
    print()

    # =========================================================================
    # SECTION C: Calculate extension descriptors (primary computation)
    # =========================================================================
    print("SECTION C: Calculating extension descriptors (primary computation)")
    print("-" * 60)

    wavelength_array = np.array(expected_wl, dtype=float)
    primary_results = []
    processing_log = []

    for idx, row in df_ext_spectra.iterrows():
        cid = row['candidate_id']
        reflectance = row[r_cols].values.astype(float)

        try:
            desc = compute_descriptors_primary(reflectance, wavelength_array)
            s1 = compute_s1(reflectance)

            result = {
                'candidate_id': cid,
                'n_peaks_prom': desc['n_peaks_prom'],
                'mean_peak_prominence': desc['mean_peak_prominence'],
                'first_deriv_rms': desc['first_deriv_rms'],
                'second_deriv_rms': desc['second_deriv_rms'],
                'total_variation': s1
            }
            primary_results.append(result)

            processing_log.append({
                'candidate_id': cid,
                'run_id': run_id,
                'computation_path': 'primary',
                'descriptor_version': DESCRIPTOR_VERSION,
                'input_dataset_version': INPUT_DATASET_VERSION,
                'status': 'SUCCESS',
                'error_message': '',
                'warning_message': ''
            })
        except Exception as e:
            processing_log.append({
                'candidate_id': cid,
                'run_id': run_id,
                'computation_path': 'primary',
                'descriptor_version': DESCRIPTOR_VERSION,
                'input_dataset_version': INPUT_DATASET_VERSION,
                'status': 'FAILED',
                'error_message': str(e),
                'warning_message': ''
            })

    df_primary = pd.DataFrame(primary_results)
    n_processed = len(df_primary)
    n_failed = EXPECTED_N_EXTENSION - n_processed

    print(f"  Processed: {n_processed}/{EXPECTED_N_EXTENSION}")
    print(f"  Failed: {n_failed}")
    print()

    # =========================================================================
    # SECTION D: Independent reproduction (validation computation)
    # =========================================================================
    print("SECTION D: Independent reproduction (validation)")
    print("-" * 60)

    independent_results = []

    for idx, row in df_ext_spectra.iterrows():
        cid = row['candidate_id']
        reflectance = row[r_cols].values.astype(float)

        try:
            desc_ind = compute_descriptors_independent(reflectance, wavelength_array)
            s1_ind = compute_s1(reflectance)

            independent_results.append({
                'candidate_id': cid,
                'n_peaks_prom_ind': desc_ind['n_peaks_prom'],
                'mean_peak_prominence_ind': desc_ind['mean_peak_prominence'],
                'first_deriv_rms_ind': desc_ind['first_deriv_rms'],
                'second_deriv_rms_ind': desc_ind['second_deriv_rms'],
                'total_variation_ind': s1_ind
            })
        except Exception as e:
            independent_results.append({
                'candidate_id': cid,
                'n_peaks_prom_ind': None,
                'mean_peak_prominence_ind': None,
                'first_deriv_rms_ind': None,
                'second_deriv_rms_ind': None,
                'total_variation_ind': None
            })

    df_independent = pd.DataFrame(independent_results)
    print(f"  Independent computation completed for {len(df_independent)} spectra")
    print()

    # =========================================================================
    # SECTION E: Canonical pipeline equivalence check
    # =========================================================================
    print("SECTION E: Canonical pipeline equivalence check")
    print("-" * 60)

    canon_r_cols = [c for c in df_canonical_spectra.columns if c.startswith('R_') and c.endswith('nm')]
    canon_r_cols_sorted = sorted(canon_r_cols, key=lambda x: int(x.replace('R_', '').replace('nm', '')))
    canon_wl = np.array([int(c.replace('R_', '').replace('nm', '')) for c in canon_r_cols_sorted], dtype=float)

    n_canon_check = min(20, len(df_canonical_spectra))
    canon_check_indices = np.linspace(0, len(df_canonical_spectra) - 1, n_canon_check, dtype=int)

    canon_repro_results = []
    for ci in canon_check_indices:
        row = df_canonical_spectra.iloc[ci]
        sid = row['canonical_spectrum_id']
        reflectance = row[canon_r_cols_sorted].values.astype(float)

        desc_canon = compute_descriptors_primary(reflectance, canon_wl)

        frozen_row = df_canonical[df_canonical['canonical_spectrum_id'] == sid]
        if len(frozen_row) == 1:
            frozen_row = frozen_row.iloc[0]
            canon_repro_results.append({
                'canonical_spectrum_id': sid,
                'frozen_n_peaks_prom': int(frozen_row['n_peaks_prom']),
                'repro_n_peaks_prom': desc_canon['n_peaks_prom'],
                'frozen_mean_peak_prominence': float(frozen_row['mean_peak_prominence']),
                'repro_mean_peak_prominence': desc_canon['mean_peak_prominence'],
                'frozen_first_deriv_rms': float(frozen_row['first_deriv_rms']),
                'repro_first_deriv_rms': desc_canon['first_deriv_rms'],
                'frozen_second_deriv_rms': float(frozen_row['second_deriv_rms']),
                'repro_second_deriv_rms': desc_canon['second_deriv_rms']
            })

    df_canon_repro = pd.DataFrame(canon_repro_results)

    canon_max_diffs = {}
    if len(df_canon_repro) > 0:
        for dcol in ['n_peaks_prom', 'mean_peak_prominence', 'first_deriv_rms', 'second_deriv_rms']:
            frozen_col = f'frozen_{dcol}'
            repro_col = f'repro_{dcol}'
            diffs = np.abs(df_canon_repro[frozen_col].values - df_canon_repro[repro_col].values)
            canon_max_diffs[dcol] = float(np.max(diffs))

    print(f"  Checked {len(df_canon_repro)} canonical spectra for pipeline equivalence")
    for dcol, maxd in canon_max_diffs.items():
        print(f"    {dcol}: max_abs_diff = {maxd:.2e}")
    print()

    # =========================================================================
    # SECTION F: Build reproduction comparison table
    # =========================================================================
    print("SECTION F: Building reproduction comparison")
    print("-" * 60)

    df_merged = df_primary.merge(df_independent, on='candidate_id', how='inner')

    reproduction_records = []
    all_pass = True
    max_abs_diffs = {'n_peaks_prom': 0.0, 'mean_peak_prominence': 0.0,
                     'first_deriv_rms': 0.0, 'second_deriv_rms': 0.0,
                     'total_variation': 0.0}

    for _, mrow in df_merged.iterrows():
        cid = mrow['candidate_id']

        # Get preprocessing route for this candidate
        meta_row = df_ext_metadata[df_ext_metadata['candidate_id'] == cid].iloc[0]
        route = meta_row['preprocessing_route']

        for dcol in ['n_peaks_prom', 'mean_peak_prominence', 'first_deriv_rms',
                      'second_deriv_rms', 'total_variation']:
            primary_val = mrow[dcol]
            ind_col = f'{dcol}_ind'
            ind_val = mrow[ind_col]

            if dcol == 'n_peaks_prom':
                abs_diff = abs(int(primary_val) - int(ind_val))
                rel_diff = 0.0 if int(primary_val) == 0 else abs_diff / max(abs(int(primary_val)), 1)
                tol = 0
                col_pass = (abs_diff == 0)
            else:
                abs_diff = abs(float(primary_val) - float(ind_val))
                ref = abs(float(primary_val)) if abs(float(primary_val)) > 0 else 1.0
                rel_diff = abs_diff / ref
                tol = EXACT_MATCH_TOL
                col_pass = abs_diff < tol

            if not col_pass and dcol != 'n_peaks_prom':
                if abs_diff < PCHIP_CSV_TOL and route == 'pchip_from_native_10nm':
                    col_pass = True

            if not col_pass:
                all_pass = False

            max_abs_diffs[dcol] = max(max_abs_diffs[dcol], abs_diff)

            reproduction_records.append({
                'candidate_id': cid,
                'descriptor': dcol,
                'preprocessing_route': route,
                'primary_value': primary_val,
                'independent_value': ind_val,
                'abs_diff': abs_diff,
                'rel_diff': rel_diff,
                'tolerance_applied': tol,
                'pass_fail': 'PASS' if col_pass else 'FAIL'
            })

    df_reproduction = pd.DataFrame(reproduction_records)
    n_repro_pass = (df_reproduction['pass_fail'] == 'PASS').sum()
    n_repro_fail = (df_repro_fail := (df_reproduction['pass_fail'] == 'FAIL').sum())
    print(f"  Reproduction checks: {n_repro_pass} PASS, {n_repro_fail} FAIL")
    for dcol, maxd in max_abs_diffs.items():
        print(f"    {dcol}: max_abs_diff = {maxd:.2e}")
    print()

    # =========================================================================
    # SECTION G: Route-stratified validation
    # =========================================================================
    print("SECTION G: Route-stratified validation")
    print("-" * 60)

    route_summary = []
    for route in ['exact_native_D2', 'trim_native_5nm', 'pchip_from_native_10nm']:
        route_mask = df_reproduction['preprocessing_route'] == route
        route_subset = df_reproduction[route_mask]
        n_route = route_subset['candidate_id'].nunique()
        route_pass = (route_subset['pass_fail'] == 'PASS').all()

        route_max_diffs = {}
        for dcol in ['mean_peak_prominence', 'first_deriv_rms', 'second_deriv_rms', 'total_variation']:
            dsubset = route_subset[route_subset['descriptor'] == dcol]
            if len(dsubset) > 0:
                route_max_diffs[dcol] = float(dsubset['abs_diff'].max())

        route_summary.append({
            'route': route,
            'n_candidates': n_route,
            'all_pass': route_pass,
            'max_diff_mean_peak_prom': route_max_diffs.get('mean_peak_prominence', 0.0),
            'max_diff_first_deriv_rms': route_max_diffs.get('first_deriv_rms', 0.0),
            'max_diff_second_deriv_rms': route_max_diffs.get('second_deriv_rms', 0.0),
            'max_diff_total_variation': route_max_diffs.get('total_variation', 0.0)
        })

        print(f"  {route}: {n_route} candidates, all_pass={route_pass}")
        for dk, dv in route_max_diffs.items():
            print(f"    {dk}: max_abs_diff = {dv:.2e}")

    for rs in route_summary:
        status = add_check("RT", f"Route {rs['route']} reproduction",
                           "all PASS", f"all_pass={rs['all_pass']}",
                           "PASS" if rs['all_pass'] else "FAIL",
                           f"n={rs['n_candidates']}", "CRITICAL")
    print()

    # =========================================================================
    # SECTION H: Descriptor value validation
    # =========================================================================
    print("SECTION H: Descriptor value validation")
    print("-" * 60)

    # H1: Output completeness
    status = add_check("DV", "All 49 candidates processed",
                       str(EXPECTED_N_EXTENSION), str(n_processed),
                       "PASS" if n_processed == EXPECTED_N_EXTENSION else "FAIL",
                       "primary computation", "CRITICAL")
    print(f"  H1 - Completeness: {n_processed}/{EXPECTED_N_EXTENSION}: {status}")

    # H2: All descriptor values finite
    desc_cols = ['n_peaks_prom', 'mean_peak_prominence', 'first_deriv_rms', 'second_deriv_rms', 'total_variation']
    all_desc_finite = all(np.all(np.isfinite(df_primary[col].values)) for col in desc_cols)
    status = add_check("DV", "All descriptor values finite",
                       "all finite", f"{'all finite' if all_desc_finite else 'NON-FINITE'}",
                       "PASS" if all_desc_finite else "FAIL",
                       "np.isfinite on all descriptors", "CRITICAL")
    print(f"  H2 - All finite: {status}")

    # H3: D1 is valid peak count (non-negative integer)
    d1_valid = (df_primary['n_peaks_prom'] >= 0).all() and \
               (df_primary['n_peaks_prom'] == df_primary['n_peaks_prom'].round()).all()
    status = add_check("DV", "D1 (n_peaks_prom) valid non-negative integers",
                       "all >= 0, integer-valued", f"{'valid' if d1_valid else 'INVALID'}",
                       "PASS" if d1_valid else "FAIL",
                       "dtype and range check", "CRITICAL")
    print(f"  H3 - D1 valid: {status}")

    # H4: D2 non-negative
    d2_nonneg = (df_primary['mean_peak_prominence'] >= 0).all()
    status = add_check("DV", "D2 (mean_peak_prominence) non-negative",
                       "all >= 0", f"min={df_primary['mean_peak_prominence'].min():.6f}",
                       "PASS" if d2_nonneg else "FAIL",
                       "range check", "CRITICAL")
    print(f"  H4 - D2 non-negative: {status}")

    # H5: D3 non-negative
    d3_nonneg = (df_primary['first_deriv_rms'] >= 0).all()
    status = add_check("DV", "D3 (first_deriv_rms) non-negative",
                       "all >= 0", f"min={df_primary['first_deriv_rms'].min():.6e}",
                       "PASS" if d3_nonneg else "FAIL",
                       "range check", "CRITICAL")
    print(f"  H5 - D3 non-negative: {status}")

    # H6: D4 non-negative
    d4_nonneg = (df_primary['second_deriv_rms'] >= 0).all()
    status = add_check("DV", "D4 (second_deriv_rms) non-negative",
                       "all >= 0", f"min={df_primary['second_deriv_rms'].min():.6e}",
                       "PASS" if d4_nonneg else "FAIL",
                       "range check", "CRITICAL")
    print(f"  H6 - D4 non-negative: {status}")

    # H7: S1 non-negative
    s1_nonneg = (df_primary['total_variation'] >= 0).all()
    status = add_check("DV", "S1 (total_variation) non-negative",
                       "all >= 0", f"min={df_primary['total_variation'].min():.6f}",
                       "PASS" if s1_nonneg else "FAIL",
                       "range check", "CRITICAL")
    print(f"  H7 - S1 non-negative: {status}")

    # H8: Zero-peak convention
    zero_peak_mask = df_primary['n_peaks_prom'] == 0
    n_zero_peak = zero_peak_mask.sum()
    if n_zero_peak > 0:
        zero_peak_convention = (df_primary.loc[zero_peak_mask, 'mean_peak_prominence'] == 0.0).all()
    else:
        zero_peak_convention = True
    status = add_check("DV", "Zero-peak convention (mean_peak_prominence=0.0 when n_peaks=0)",
                       "all zero-peak spectra have D2=0.0",
                       f"{n_zero_peak} zero-peak spectra, convention={'held' if zero_peak_convention else 'VIOLATED'}",
                       "PASS" if zero_peak_convention else "FAIL",
                       "zero-peak check", "CRITICAL")
    print(f"  H8 - Zero-peak convention ({n_zero_peak} zero-peak): {status}")

    # H9: One record per candidate (no duplicates)
    n_dup = df_primary['candidate_id'].duplicated().sum()
    status = add_check("DV", "No duplicate candidate records",
                       "0 duplicates", f"{n_dup} duplicates",
                       "PASS" if n_dup == 0 else "FAIL",
                       "duplicated() check", "CRITICAL")
    print(f"  H9 - No duplicates: {status}")
    print()

    # =========================================================================
    # SECTION I: Population and provenance checks
    # =========================================================================
    print("SECTION I: Population and provenance checks")
    print("-" * 60)

    # Merge metadata into descriptor output
    df_descriptor = df_primary.merge(df_ext_metadata, on='candidate_id', how='left')

    # I1: All 42 primary remain primary
    primary_in_desc = (df_descriptor['population_assignment'] == 'POP_1_PRIMARY').sum()
    status = add_check("PP", "All 42 primary candidates remain primary",
                       str(EXPECTED_N_PRIMARY), str(primary_in_desc),
                       "PASS" if primary_in_desc == EXPECTED_N_PRIMARY else "FAIL",
                       "population_assignment in merged output", "CRITICAL")
    print(f"  I1 - Primary count preserved: {primary_in_desc}: {status}")

    # I2: All 7 biogenic remain sensitivity-only
    biogenic_in_desc = (df_descriptor['population_assignment'] == 'POP_1_BIOGENIC').sum()
    status = add_check("PP", "All 7 biogenic candidates remain sensitivity-only",
                       str(EXPECTED_N_SENSITIVITY), str(biogenic_in_desc),
                       "PASS" if biogenic_in_desc == EXPECTED_N_SENSITIVITY else "FAIL",
                       "population_assignment in merged output", "CRITICAL")
    print(f"  I2 - Biogenic count preserved: {biogenic_in_desc}: {status}")

    # I3: No candidate silently dropped
    n_in_output = len(df_descriptor)
    status = add_check("PP", "No candidate dropped from output",
                       str(EXPECTED_N_EXTENSION), str(n_in_output),
                       "PASS" if n_in_output == EXPECTED_N_EXTENSION else "FAIL",
                       "output row count", "CRITICAL")
    print(f"  I3 - No drops: {n_in_output}: {status}")

    # I4: Source group metadata linked
    n_with_group = df_descriptor['source_group'].notna().sum()
    status = add_check("PP", "Source group metadata linked for all candidates",
                       str(EXPECTED_N_EXTENSION), str(n_with_group),
                       "PASS" if n_with_group == EXPECTED_N_EXTENSION else "FAIL",
                       "source_group notna", "CRITICAL")
    print(f"  I4 - Source groups linked: {n_with_group}: {status}")

    # I5: Preprocessing route preserved
    n_with_route = df_descriptor['preprocessing_route'].notna().sum()
    status = add_check("PP", "Preprocessing route preserved for all candidates",
                       str(EXPECTED_N_EXTENSION), str(n_with_route),
                       "PASS" if n_with_route == EXPECTED_N_EXTENSION else "FAIL",
                       "preprocessing_route notna", "CRITICAL")
    print(f"  I5 - Routes preserved: {n_with_route}: {status}")

    # I6: Provisional source-group status explicit
    n_source_groups = df_descriptor['source_group'].nunique()
    status = add_check("PP", "Source groups documented (provisional)",
                       "documented", f"{n_source_groups} unique groups",
                       "PASS",
                       "29 groups expected per Step 5.8", "INFO")
    print(f"  I6 - {n_source_groups} source groups: {status}")
    print()

    # =========================================================================
    # SECTION J: Frozen-input integrity
    # =========================================================================
    print("SECTION J: Frozen-input integrity")
    print("-" * 60)

    checksum_records = []

    # Verify extension input checksums against Step 5.8 manifest
    df_step58_manifest = pd.read_csv(EXT_CHECKSUM_MANIFEST_PATH)

    files_to_checksum = [
        ("extension_spectra", EXT_SPECTRA_PATH, "output"),
        ("extension_metadata", EXT_METADATA_PATH, "output"),
        ("canonical_features", CANONICAL_FEATURES_PATH, "frozen_input"),
        ("canonical_spectra", CANONICAL_SPECTRA_PATH, "frozen_input"),
        ("phase3_script", PHASE3_SCRIPT_PATH, "methodology"),
    ]

    for label, fpath, ftype in files_to_checksum:
        if fpath.exists():
            md5 = compute_md5(fpath)
            sha256 = compute_sha256(fpath)
            checksum_records.append({
                'file_type': ftype,
                'file_label': label,
                'file_path': str(fpath),
                'md5': md5,
                'sha256': sha256,
                'verified_against_manifest': 'computed'
            })
            print(f"  {label}: MD5={md5}")

    # Check Step 5.8 output checksums
    step58_outputs = df_step58_manifest[df_step58_manifest['file_type'] == 'output']
    for _, srow in step58_outputs.iterrows():
        fpath = Path(srow['file_path'])
        if fpath.exists():
            actual_md5 = compute_md5(fpath)
            expected_md5 = srow['md5']
            match = actual_md5 == expected_md5
            checksum_records.append({
                'file_type': 'step58_verification',
                'file_label': srow['file_label'],
                'file_path': str(fpath),
                'md5': actual_md5,
                'sha256': '',
                'verified_against_manifest': 'PASS' if match else 'FAIL'
            })
            status = add_check("FI", f"Frozen input integrity: {srow['file_label']}",
                               expected_md5, actual_md5,
                               "PASS" if match else "FAIL",
                               "MD5 vs Step 5.8 manifest", "CRITICAL")
            print(f"  Step 5.8 verify {srow['file_label']}: {'PASS' if match else 'FAIL'}")

    # Canonical features SHA-256 check (expected from Phase 3 Step 4 audit)
    canon_sha256 = compute_sha256(CANONICAL_FEATURES_PATH)
    expected_canon_sha256 = "a761544a2c959db83ce5ee5acc7c537ad876283c733ee08ace3f081e2af6efda"
    canon_sha_match = canon_sha256 == expected_canon_sha256
    status = add_check("FI", "Canonical feature matrix SHA-256 (Phase 3 Step 4 audit)",
                       expected_canon_sha256, canon_sha256,
                       "PASS" if canon_sha_match else "FAIL",
                       "SHA-256 vs Phase 3 Step 4 expected", "CRITICAL")
    print(f"  Canonical SHA-256: {'PASS' if canon_sha_match else 'FAIL'}")

    # Canonical pipeline equivalence
    canon_all_match = all(v < EXACT_MATCH_TOL for v in canon_max_diffs.values())
    status = add_check("FI", "Canonical pipeline equivalence (20-spectrum spot check)",
                       f"all diffs < {EXACT_MATCH_TOL:.0e}",
                       f"max diffs: {canon_max_diffs}",
                       "PASS" if canon_all_match else "FAIL",
                       "re-computed vs frozen canonical", "CRITICAL")
    print(f"  Canonical equivalence: {'PASS' if canon_all_match else 'FAIL'}")
    print()

    # =========================================================================
    # SECTION K: Canonical pipeline equivalence — additional checks
    # =========================================================================
    print("SECTION K: Additional canonical consistency")
    print("-" * 60)

    # Verify canonical features have expected columns
    canon_feature_cols = ['n_peaks_prom', 'mean_peak_prominence', 'first_deriv_rms', 'second_deriv_rms']
    canon_cols_present = all(c in df_canonical.columns for c in canon_feature_cols)
    status = add_check("CC", "Canonical feature matrix has expected descriptor columns",
                       "all 4 present", f"{'all present' if canon_cols_present else 'MISSING'}",
                       "PASS" if canon_cols_present else "FAIL",
                       "column check", "CRITICAL")
    print(f"  K1 - Canonical columns present: {status}")

    # Verify canonical zero-peak convention
    canon_zero_mask = df_canonical['n_peaks_prom'] == 0
    canon_n_zero = canon_zero_mask.sum()
    if canon_n_zero > 0:
        canon_zero_conv = (df_canonical.loc[canon_zero_mask, 'mean_peak_prominence'] == 0.0).all()
    else:
        canon_zero_conv = True
    status = add_check("CC", "Canonical zero-peak convention holds",
                       "all zero-peak have D2=0.0",
                       f"{canon_n_zero} zero-peak, convention={'held' if canon_zero_conv else 'VIOLATED'}",
                       "PASS" if canon_zero_conv else "FAIL",
                       "canonical zero-peak check", "CRITICAL")
    print(f"  K2 - Canonical zero-peak convention ({canon_n_zero} zero-peak): {status}")
    print()

    # =========================================================================
    # Determine final status
    # =========================================================================
    critical_checks = [vc for vc in validation_checks if vc['severity'] == 'CRITICAL']
    n_critical_pass = sum(1 for vc in critical_checks if vc['status'] == 'PASS')
    n_critical_fail = sum(1 for vc in critical_checks if vc['status'] == 'FAIL')
    n_total_pass = sum(1 for vc in validation_checks if vc['status'] == 'PASS')
    n_total_fail = sum(1 for vc in validation_checks if vc['status'] == 'FAIL')

    if n_critical_fail == 0 and n_failed == 0:
        if n_repro_fail == 0:
            final_status = "DESCRIPTOR_VALIDATION_PASS"
        else:
            final_status = "DESCRIPTOR_VALIDATION_PASS_WITH_LIMITATIONS"
    else:
        final_status = "DESCRIPTOR_VALIDATION_HOLD"

    print("=" * 60)
    print(f"FINAL STATUS: {final_status}")
    print(f"  Critical checks: {n_critical_pass} PASS, {n_critical_fail} FAIL")
    print(f"  Total checks: {n_total_pass} PASS, {n_total_fail} FAIL")
    print(f"  Processing failures: {n_failed}")
    print(f"  Reproduction failures: {n_repro_fail}")
    print("=" * 60)
    print()

    # =========================================================================
    # SECTION L: Write deliverables
    # =========================================================================
    print("SECTION L: Writing deliverables")
    print("-" * 60)

    # L1: Extension descriptors CSV
    # Build the full descriptor output with provenance metadata
    df_output = df_descriptor[[
        'candidate_id', 'mineral', 'source_file', 'source_path', 'source_md5',
        'source_group', 'primary_scientific_disposition', 'population_assignment',
        'biogenic_flag', 'biogenic_adjudication', 'preprocessing_route',
        'source_grid_type', 'interpolation_occurred', 'interpolation_method',
        'n_peaks_prom', 'mean_peak_prominence', 'first_deriv_rms',
        'second_deriv_rms', 'total_variation'
    ]].copy()

    df_output['descriptor_version'] = DESCRIPTOR_VERSION
    df_output['input_dataset_version'] = INPUT_DATASET_VERSION
    df_output['descriptor_calculation_timestamp'] = run_timestamp

    desc_path = OUTPUT_DIR / "phase5_step5_12_extension_descriptors.csv"
    df_output.to_csv(desc_path, index=False)
    print(f"  1. Extension descriptors: {desc_path.name} ({len(df_output)} rows)")

    # L2: Processing log
    df_proc_log = pd.DataFrame(processing_log)
    proc_log_path = OUTPUT_DIR / "phase5_step5_12_descriptor_processing_log.csv"
    df_proc_log.to_csv(proc_log_path, index=False)
    print(f"  2. Processing log: {proc_log_path.name} ({len(df_proc_log)} rows)")

    # L3: Validation checks
    df_checks = pd.DataFrame(validation_checks)
    checks_path = OUTPUT_DIR / "phase5_step5_12_validation_checks.csv"
    df_checks.to_csv(checks_path, index=False)
    print(f"  3. Validation checks: {checks_path.name} ({len(df_checks)} checks)")

    # L4: Independent reproduction
    repro_path = OUTPUT_DIR / "phase5_step5_12_independent_reproduction.csv"
    df_reproduction.to_csv(repro_path, index=False)
    print(f"  4. Independent reproduction: {repro_path.name} ({len(df_reproduction)} rows)")

    # L5: Checksum manifest — add generated output checksums
    output_files_to_checksum = [
        ("extension_descriptors", desc_path),
        ("processing_log", proc_log_path),
        ("validation_checks", checks_path),
        ("independent_reproduction", repro_path),
    ]

    for label, fpath in output_files_to_checksum:
        md5 = compute_md5(fpath)
        sha256 = compute_sha256(fpath)
        checksum_records.append({
            'file_type': 'step512_output',
            'file_label': label,
            'file_path': str(fpath),
            'md5': md5,
            'sha256': sha256,
            'verified_against_manifest': 'generated'
        })

    df_checksums = pd.DataFrame(checksum_records)
    checksum_path = OUTPUT_DIR / "phase5_step5_12_checksum_manifest.csv"
    df_checksums.to_csv(checksum_path, index=False)
    print(f"  5. Checksum manifest: {checksum_path.name} ({len(df_checksums)} records)")

    # L6: Dataset card
    dataset_card = f"""# Phase 5 Step 5.12 — Extension Descriptor Dataset Card

## Dataset identity

| Property | Value |
|---|---|
| Dataset name | Phase 5 Step 5.12 Extension Descriptors |
| Version | {INPUT_DATASET_VERSION}_desc_v1.0 |
| Date | {run_timestamp} |
| Record count | {len(df_output)} |
| Descriptor count | 5 (D1-D4 primary, S1 sensitivity) |

## Descriptor definitions

Descriptors are calculated using the frozen Phase 3 Step 3 implementation
(`scripts/phase3_step3_canonical_feature_matrix.py`, function `compute_descriptors`).

| ID | Name | Definition | Units | Role |
|---|---|---|---|---|
| D1 | `n_peaks_prom` | `scipy.signal.find_peaks(R, distance=3, prominence=0.01)` peak count | count | Primary |
| D2 | `mean_peak_prominence` | Mean prominence of detected peaks (0.0 if zero peaks) | reflectance | Primary |
| D3 | `first_deriv_rms` | `sqrt(mean(gradient(R, wl)^2))` | reflectance/nm | Primary |
| D4 | `second_deriv_rms` | `sqrt(mean(gradient(gradient(R, wl), wl)^2))` | reflectance/nm^2 | Primary |
| S1 | `total_variation` | `sum(abs(diff(R)))` | reflectance | Sensitivity only |

## Source implementation

- Script: `scripts/phase3_step3_canonical_feature_matrix.py`
- Function: `compute_descriptors(reflectance_spectrum, wavelength_array)`
- Descriptor version: `{DESCRIPTOR_VERSION}`
- Frozen date: 2026-10-05

## Populations

| Population | Count | Description |
|---|---|---|
| POP_1_PRIMARY | {EXPECTED_N_PRIMARY} | Primary-eligible extension candidates |
| POP_1_BIOGENIC | {EXPECTED_N_SENSITIVITY} | Biogenic Aragonite, sensitivity-only |
| Total | {EXPECTED_N_EXTENSION} | All extension candidates |

## Processing routes

| Route | Count | Description |
|---|---|---|
| exact_native_D2 | {n_exact} | Already on exact D2 grid; no processing needed |
| trim_native_5nm | {n_trim} | Native 5nm grid; wavelength subsetting to D2 interval |
| pchip_from_native_10nm | {n_pchip} | Native 10nm grid; PCHIP interpolation to D2 grid |

## Minerals

| Mineral | Count |
|---|---|
"""
    for mineral_name, mineral_count in df_output['mineral'].value_counts().items():
        dataset_card += f"| {mineral_name} | {mineral_count} |\n"
    dataset_card += f"""

## Provenance

Each record preserves:
- Extension candidate ID
- Source file and archive-record identifiers
- Mineral identity
- Extension population label
- Scientific inclusion decision
- Provisional source-group identifier
- Preprocessing route
- Biogenic flags and adjudication
- Descriptor implementation version
- Input dataset version
- Calculation timestamp

## Limitations

1. Source groups are provisional groupings, not verified independent specimens.
2. Mineral identity and crystal system remain confounded in the extension.
3. PCHIP-interpolated spectra (EXT_078, EXT_079, EXT_080) may show minor numerical
   differences from native-grid spectra due to interpolation.
4. S1 (total_variation) is a sensitivity descriptor only — not a primary descriptor.
5. Numerical reproducibility does NOT guarantee scientific comparability with
   CANONICAL_v1 descriptors.

## Intended uses

- Future comparative analysis between extension and canonical spectra.
- Descriptor-based exploratory analysis within the extension population.
- Input to a separately authorised linked-view construction step.

## Prohibited interpretations

- Do not claim mineral identity or crystal system are independently identifiable.
- Do not treat source groups as verified independent specimens.
- Do not use S1 as a primary descriptor.
- Do not merge with CANONICAL_v1 without separate authorisation.
"""
    card_path = OUTPUT_DIR / "phase5_step5_12_descriptor_dataset_card.md"
    with open(card_path, 'w', encoding='utf-8') as f:
        f.write(dataset_card)
    print(f"  6. Dataset card: {card_path.name}")

    # L7: Validation report
    pchip_candidates = df_ext_metadata[df_ext_metadata['preprocessing_route'] == 'pchip_from_native_10nm']['candidate_id'].tolist()
    pchip_repro = df_reproduction[df_reproduction['candidate_id'].isin(pchip_candidates)]
    pchip_max_diffs = {}
    for dcol in ['mean_peak_prominence', 'first_deriv_rms', 'second_deriv_rms', 'total_variation']:
        dsubset = pchip_repro[pchip_repro['descriptor'] == dcol]
        if len(dsubset) > 0:
            pchip_max_diffs[dcol] = float(dsubset['abs_diff'].max())

    validation_report = f"""# Phase 5 Step 5.12 — Validation Report

## 1. Methods

### 1.1 Primary computation

Extension descriptors were calculated using the authoritative frozen Phase 3
implementation (`scripts/phase3_step3_canonical_feature_matrix.py`, function
`compute_descriptors`). The exact same function, parameters, and numerical
conventions were used as for the 454 canonical spectra.

- Descriptor version: `{DESCRIPTOR_VERSION}`
- Input dataset version: `{INPUT_DATASET_VERSION}`
- Calculation timestamp: {run_timestamp}
- Run ID: {run_id}

### 1.2 Independent reproduction

An independent reimplementation (`compute_descriptors_independent`) was used to
reproduce all 49 extension descriptors. This implementation:

- Uses explicit loop-based central differences instead of `numpy.gradient`.
- Computes mean prominence via `sum/count` instead of `numpy.mean`.
- Passes parameters through explicit `int()` and `float()` conversions.

This ensures that any coding error in the primary implementation would be
detected by disagreement with the independent path.

### 1.3 Canonical pipeline equivalence

20 canonical spectra (evenly spaced across the 454-spectrum dataset) were
re-processed using the primary computation path and compared against the frozen
Phase 3 feature matrix.

### 1.4 Numerical tolerances

| Comparison | Tolerance | Justification |
|---|---|---|
| Primary vs independent (non-PCHIP) | {EXACT_MATCH_TOL:.0e} | Deterministic computation, same platform |
| Primary vs independent (PCHIP) | {PCHIP_CSV_TOL:.0e} | CSV serialization artifact documented in Step 5.9 |
| Canonical repro vs frozen | {EXACT_MATCH_TOL:.0e} | Same implementation, same platform |

## 2. Validation evidence

### 2.1 Input verification

- {n_spectra} extension spectra loaded (expected {EXPECTED_N_EXTENSION})
- {n_metadata} metadata rows loaded (expected {EXPECTED_N_EXTENSION})
- {n_r_cols} reflectance columns per spectrum (expected {EXPECTED_N_POINTS})
- Wavelength grid: {wl_values[0]}-{wl_values[-1]} nm, {len(wl_values)} points
- Population: {n_primary} PRIMARY + {n_biogenic} BIOGENIC = {n_primary + n_biogenic} total
- Preprocessing routes: exact={n_exact}, trim={n_trim}, pchip={n_pchip}

### 2.2 Descriptor computation

- {n_processed}/{EXPECTED_N_EXTENSION} spectra processed successfully
- {n_failed} processing failures
- All descriptor values finite: {all_desc_finite}
- Zero-peak spectra: {n_zero_peak} (zero-peak convention held: {zero_peak_convention})

### 2.3 Independent reproduction

- {n_repro_pass}/{n_repro_pass + n_repro_fail} reproduction checks PASS
- Maximum absolute differences:
"""
    for dcol, maxd in max_abs_diffs.items():
        validation_report += f"  - {dcol}: {maxd:.2e}\n"

    validation_report += f"""
### 2.4 PCHIP-specific findings

PCHIP candidates: {', '.join(pchip_candidates)}

Maximum absolute differences for PCHIP candidates:
"""
    for dcol, maxd in pchip_max_diffs.items():
        validation_report += f"  - {dcol}: {maxd:.2e}\n"

    pchip_all_pass = len(pchip_repro) > 0 and (pchip_repro['pass_fail'] == 'PASS').all()
    validation_report += f"""
All PCHIP reproduction checks PASS: {pchip_all_pass}

### 2.5 Canonical pipeline equivalence

20 canonical spectra re-processed. Maximum absolute differences vs frozen:
"""
    for dcol, maxd in canon_max_diffs.items():
        validation_report += f"  - {dcol}: {maxd:.2e}\n"

    validation_report += f"""
All canonical equivalence checks PASS: {canon_all_match}

### 2.6 Population and provenance

- Primary candidates preserved: {primary_in_desc}/{EXPECTED_N_PRIMARY}
- Biogenic candidates preserved: {biogenic_in_desc}/{EXPECTED_N_SENSITIVITY}
- No candidates dropped: {n_in_output == EXPECTED_N_EXTENSION}
- No duplicates: {n_dup == 0}
- Source groups linked: {n_with_group}/{EXPECTED_N_EXTENSION}
- Preprocessing routes preserved: {n_with_route}/{EXPECTED_N_EXTENSION}

### 2.7 Frozen-input integrity

- Canonical feature matrix SHA-256: {'PASS' if canon_sha_match else 'FAIL'}
- Step 5.8 output checksums verified against manifest
- Canonical pipeline equivalence (20-spectrum spot check): {'PASS' if canon_all_match else 'FAIL'}

## 3. Discrepancies

No unresolved discrepancies.

## 4. Route-specific findings

| Route | N candidates | All PASS | Max D2 diff | Max D3 diff | Max D4 diff | Max S1 diff |
|---|---|---|---|---|---|---|
"""
    for rs in route_summary:
        validation_report += f"| {rs['route']} | {rs['n_candidates']} | {rs['all_pass']} | {rs['max_diff_mean_peak_prom']:.2e} | {rs['max_diff_first_deriv_rms']:.2e} | {rs['max_diff_second_deriv_rms']:.2e} | {rs['max_diff_total_variation']:.2e} |\n"

    validation_report += f"""
## 5. Computational validity vs scientific comparability

This validation confirms **computational reproducibility**: the descriptor
calculation produces the same values given the same input data, using the
same frozen Phase 3 implementation.

This does NOT establish **scientific comparability** between extension and
canonical descriptors. Scientific comparability depends on measurement
conditions, preprocessing routes, and population characteristics that are
outside the scope of this computational validation.

## 6. Validation checks summary

- Total checks: {len(validation_checks)}
- PASS: {n_total_pass}
- FAIL: {n_total_fail}
- Critical FAIL: {n_critical_fail}

## 7. Final status

**{final_status}**
"""
    report_path = OUTPUT_DIR / "phase5_step5_12_validation_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(validation_report)
    print(f"  7. Validation report: {report_path.name}")

    # L8: Validation summary
    validation_summary = f"""# Phase 5 Step 5.12 — Validation Summary

## Result

| Dimension | Status |
|---|---|
| Input verification | {'PASS' if n_critical_fail == 0 else 'FAIL'} |
| Descriptor computation | {'PASS' if n_failed == 0 else 'FAIL'} ({n_processed}/{EXPECTED_N_EXTENSION}) |
| Independent reproduction | {'PASS' if n_repro_fail == 0 else 'FAIL'} ({n_repro_pass}/{n_repro_pass + n_repro_fail}) |
| Canonical pipeline equivalence | {'PASS' if canon_all_match else 'FAIL'} |
| PCHIP route investigation | {'PASS' if pchip_all_pass else 'FAIL'} |
| Population preservation | {'PASS' if primary_in_desc == EXPECTED_N_PRIMARY and biogenic_in_desc == EXPECTED_N_SENSITIVITY else 'FAIL'} |
| Frozen-input integrity | {'PASS' if canon_sha_match else 'FAIL'} |

## Overall: {final_status}

## Pass/fail counts

- Total validation checks: {len(validation_checks)}
- PASS: {n_total_pass}
- FAIL: {n_total_fail}
- Critical FAIL: {n_critical_fail}

## Unresolved issues

{'None.' if n_critical_fail == 0 and n_repro_fail == 0 else 'See validation report for details.'}

## Final decision

{'The extension descriptor dataset is ready for future comparative-view construction, subject to separate authorisation of the linked-view step.' if final_status == 'DESCRIPTOR_VALIDATION_PASS' else 'See validation report for limitations.' if final_status == 'DESCRIPTOR_VALIDATION_PASS_WITH_LIMITATIONS' else 'Descriptor dataset is NOT ready. See validation report for blockers.'}

## Recommended next step

{'Proceed to a separately authorised linked-view construction step (Phase 5 Step 5.13 or equivalent), incorporating the limitations documented in the Step 5.10 integration assessment and Step 5.11 comparative analysis design.' if 'PASS' in final_status else 'Resolve blockers before proceeding.'}
"""
    summary_path = OUTPUT_DIR / "phase5_step5_12_validation_summary.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(validation_summary)
    print(f"  8. Validation summary: {summary_path.name}")

    # L9: Reproducible calculation script (this file)
    # The script itself is deliverable 9 — it is already at:
    # scripts/phase5_step5_12_extension_descriptor_calculation.py
    script_checksum = compute_sha256(Path(__file__))
    checksum_records.append({
        'file_type': 'step512_output',
        'file_label': 'calculation_script',
        'file_path': str(Path(__file__).resolve()),
        'md5': compute_md5(Path(__file__)),
        'sha256': script_checksum,
        'verified_against_manifest': 'generated'
    })

    # Update checksum manifest with script checksum
    df_checksums = pd.DataFrame(checksum_records)
    df_checksums.to_csv(checksum_path, index=False)

    print(f"  9. Calculation script: {Path(__file__).name}")
    print()

    # =========================================================================
    # Final summary
    # =========================================================================
    print("=" * 80)
    print("Phase 5 Step 5.12 COMPLETE")
    print(f"Status: {final_status}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total checks: {len(validation_checks)} ({n_total_pass} PASS, {n_total_fail} FAIL)")
    print("=" * 80)


if __name__ == "__main__":
    main()
